#
# Copyright (c) 2024 University of York and others
#
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0.
# 
# SPDX-License-Identifier: EPL-2.0
#
# Contributors:
#   * Alan Millard - initial contributor
#   * Pedro Ribeiro - revised implementation
#
 
import sys

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.qos import QoSPresetProfiles
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_msgs.msg import Float32
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from assessment_interfaces.msg import Item, ItemList, ItemHolder
from auro_interfaces.msg import StringWithPose
from auro_interfaces.srv import ItemRequest

from tf_transformations import euler_from_quaternion
import angles

from enum import Enum
import random
import math

LINEAR_VELOCITY  = 0.3 # Metres per second
ANGULAR_VELOCITY = 0.5 # Radians per second

TURN_LEFT = 1 # Postive angular velocity turns left
TURN_RIGHT = -1 # Negative angular velocity turns right

SCAN_THRESHOLD = 0.5 # Metres per second
PICKUP_DISTANCE = 0.35
 # Array indexes for sensor sectors
SCAN_FRONT = 0
SCAN_LEFT = 1
SCAN_BACK = 2
SCAN_RIGHT = 3

# Zone positions (example coordinates - adjust based on actual arena)
ZONES = {
    'cyan': {'x': -3.0, 'y': 3.0},
    'purple': {'x': 3.0, 'y': 3.0},
    'green': {'x': -3.0, 'y': -3.0},
    'pink': {'x': 3.0, 'y': -3.0}
}

# Finite state machine (FSM) states
class State(Enum):
    SEARCHING = 0      # Looking for items
    APPROACHING = 1    # Moving towards detected item
    COLLECTING = 2     # Picking up item
    NAVIGATING = 3     # Moving to appropriate zone
    DEPOSITING = 4     # Depositing item in zone
    AVOIDING = 5       # Avoiding obstacles


class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')
        
        # Class variables used to store persistent values between executions of callbacks and control loop
        self.state = State.SEARCHING # Current FSM state
        self.pose = Pose() # Current pose (position and orientation), relative to the odom reference frame
        self.previous_pose = Pose() # Store a snapshot of the pose for comparison against future poses
        self.yaw = 0.0 # Angle the robot is facing (rotation around the Z axis, in radians), relative to the odom reference frame
        self.previous_yaw = 0.0 # Snapshot of the angle for comparison against future angles
        self.turn_angle = 0.0 # Relative angle to turn to in the TURNING state
        self.turn_direction = TURN_LEFT # Direction to turn in the TURNING state
        self.goal_distance = random.uniform(1.0, 2.0) # Goal distance to travel in FORWARD state
        self.scan_triggered = [False] * 4 # Boolean value for each of the 4 LiDAR sensor sectors. True if obstacle detected within SCAN_THRESHOLD
        self.items = ItemList()
        self.current_item = None
        self.target_zone = None

        self.declare_parameter('robot_id', 'robot1')
        self.robot_id = self.get_parameter('robot_id').value

        # Here we use two callback groups, to ensure that those in 'client_callback_group' can be executed
        # independently from those in 'timer_callback_group'. This allos calling the services below within
        # a callback handled by the timer_callback_group. See https://docs.ros.org/en/humble/How-To-Guides/Using-callback-groups.html
        # for a detailed discussion on the ROS executors and callback groups.
        client_callback_group = MutuallyExclusiveCallbackGroup()
        timer_callback_group = MutuallyExclusiveCallbackGroup()

        self.pick_up_service = self.create_client(ItemRequest, '/pick_up_item', callback_group=client_callback_group)
        self.offload_service = self.create_client(ItemRequest, '/offload_item', callback_group=client_callback_group)

        self.item_subscriber = self.create_subscription(
            ItemList,
            'items',
            self.item_callback,
            10, callback_group=timer_callback_group
        )

        self.holder_subscriber = self.create_subscription(
            ItemHolder,
            '/item_holders',
            self.holder_callback,
            10,
            callback_group=timer_callback_group
        )

        # Subscribes to Odometry messages published on /odom topic
        # http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html
        #
        # Final argument can either be an integer representing the history depth, or a Quality of Service (QoS) profile
        # https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html
        # https://github.com/ros2/rclpy/blob/humble/rclpy/rclpy/node.py#L1335-L1338
        # https://github.com/ros2/rclpy/blob/humble/rclpy/rclpy/node.py#L1187-L1196
        #
        # If you only specify a history depth, rclpy defaults to QoSHistoryPolicy.KEEP_LAST
        # https://github.com/ros2/rclpy/blob/humble/rclpy/rclpy/qos.py#L80-L83
        self.odom_subscriber = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10, callback_group=timer_callback_group)
        
        # Subscribes to LaserScan messages on the /scan topic
        # http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/LaserScan.html
        #
        # QoSPresetProfiles.SENSOR_DATA specifices "best effort" reliability and a small queue size
        # https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html
        # https://github.com/ros2/rclpy/blob/humble/rclpy/rclpy/qos.py#L455
        # https://github.com/ros2/rclpy/blob/humble/rclpy/rclpy/qos.py#L428-L431
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            QoSPresetProfiles.SENSOR_DATA.value, callback_group=timer_callback_group)

        # Publishes Twist messages (linear and angular velocities) on the /cmd_vel topic
        # http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/Twist.html
        # 
        # Gazebo ROS differential drive plugin subscribes to these messages, and converts them into left and right wheel speeds
        # https://github.com/ros-simulation/gazebo_ros_pkgs/blob/ros2/gazebo_plugins/src/gazebo_ros_diff_drive.cpp#L537-L555
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        #self.orientation_publisher = self.create_publisher(Float32, '/orientation', 10)

        # Publishes custom StringWithPose (see auro_interfaces/msg/StringWithPose.msg) messages on the /marker_input topic
        # The week3/rviz_text_marker node subscribes to these messages, and ouputs a Marker message on the /marker_output topic
        # ros2 run week_3 rviz_text_marker
        # This can be visualised in RViz: Add > By topic > /marker_output
        #
        # http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/Marker.html
        # http://wiki.ros.org/rviz/DisplayTypes/Marker
        self.marker_publisher = self.create_publisher(StringWithPose, 'marker_input', 10, callback_group=timer_callback_group)

        # Creates a timer that calls the control_loop method repeatedly - each loop represents single iteration of the FSM
        self.timer_period = 0.1 # 100 milliseconds = 10 Hz
        self.timer = self.create_timer(self.timer_period, self.control_loop, callback_group=timer_callback_group)

    def item_callback(self, msg):
        self.items = msg
        if len(msg.data) > 0:
            self.get_logger().info(f'Items detected: {len(msg.data)}')

    def holder_callback(self, msg):
        # Track which robot is holding what item
        for holder in msg.data:
            if holder.robot_id == self.robot_id:
                self.current_item = holder.item
                return
        self.current_item = None

    # Called every time odom_subscriber receives an Odometry message from the /odom topic
    #
    # The Gazebo ROS differential drive plugin generates these messages using kinematic equations, and publishes them
    # https://github.com/ros-simulation/gazebo_ros_pkgs/blob/ros2/gazebo_plugins/src/gazebo_ros_diff_drive.cpp#L434-L535
    #
    # This plugin is configured with physical measurements of the TurtleBot3 in the SDF file that defines the robot model
    # https://github.com/ROBOTIS-GIT/turtlebot3_simulations/blob/humble-devel/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf#L476-L507
    #
    # The pose estimates are expressed in a coordinate system relative to the starting pose of the robot
    def odom_callback(self, msg):
        self.pose = msg.pose.pose # Store the pose in a class variable

        # Uses tf_transformations package to convert orientation from quaternion to Euler angles (RPY = roll, pitch, yaw)
        # https://github.com/DLu/tf_transformations
        #
        # Roll (rotation around X axis) and pitch (rotation around Y axis) are discarded
        (roll, pitch, yaw) = euler_from_quaternion([self.pose.orientation.x,
                                                    self.pose.orientation.y,
                                                    self.pose.orientation.z,
                                                    self.pose.orientation.w])
        
        
        self.yaw = yaw # Store the yaw in a class variable

    # Called every time scan_subscriber recieves a LaserScan message from the /scan topic
    #
    # The Gazebo RaySensor calculates distance at which rays intersect with obstacles
    # The data is published by the Gazebo ROS ray sensor plugin
    # https://github.com/gazebosim/gazebo-classic/tree/gazebo11/gazebo/sensors
    # https://github.com/ros-simulation/gazebo_ros_pkgs/blob/ros2/gazebo_plugins/src/gazebo_ros_ray_sensor.cpp#L178-L205
    #
    # This plugin is configured to match the LiDAR on the TurtleBot3 in the SDF file that defines the robot model
    # http://wiki.ros.org/hls_lfcd_lds_driver
    # https://github.com/ROBOTIS-GIT/turtlebot3_simulations/blob/humble-devel/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf#L132-L165
    def scan_callback(self, msg):
        # Group scan ranges into 4 segments
        # Front, left, and right segments are each 60 degrees
        # Back segment is 180 degrees
        front_ranges = msg.ranges[331:359] + msg.ranges[0:30] # 30 to 331 degrees (30 to -30 degrees)
        left_ranges  = msg.ranges[31:90] # 31 to 90 degrees (31 to 90 degrees)
        back_ranges  = msg.ranges[91:270] # 91 to 270 degrees (91 to -90 degrees)
        right_ranges = msg.ranges[271:330] # 271 to 330 degrees (-30 to -91 degrees)

        # Store True/False values for each sensor segment, based on whether the nearest detected obstacle is closer than SCAN_THRESHOLD
        self.scan_triggered[SCAN_FRONT] = min(front_ranges) < SCAN_THRESHOLD 
        self.scan_triggered[SCAN_LEFT]  = min(left_ranges)  < SCAN_THRESHOLD
        self.scan_triggered[SCAN_BACK]  = min(back_ranges)  < SCAN_THRESHOLD
        self.scan_triggered[SCAN_RIGHT] = min(right_ranges) < SCAN_THRESHOLD

    def move_to_pose(self, target_x, target_y):
        # Calculate distance and angle to target
        dx = target_x - self.pose.position.x
        dy = target_y - self.pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)
        target_angle = math.atan2(dy, dx)
        
        # Calculate angle difference
        angle_diff = angles.normalize_angle(target_angle - self.yaw)
        
        msg = Twist()
        
        # If not facing target, turn first
        if abs(angle_diff) > 0.1:
            msg.angular.z = ANGULAR_VELOCITY * (1 if angle_diff > 0 else -1)
        else:
            # Move towards target
            msg.linear.x = min(LINEAR_VELOCITY, distance)
            msg.angular.z = 0.5 * angle_diff  # Proportional control for steering
            
        self.cmd_vel_publisher.publish(msg)
        return distance < 0.1  # Return True if at target

    # Control loop for the FSM - called periodically by self.timer
    def control_loop(self):
        self.get_logger().info(f'State: {self.state.name}')
        
        match self.state:
            case State.SEARCHING:
                if self.scan_triggered[SCAN_FRONT]:
                    self.state = State.AVOIDING
                    return

                if len(self.items.data) > 0:
                    self.state = State.APPROACHING
                    return

                # Random walk behavior
                msg = Twist()
                msg.linear.x = LINEAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

            case State.APPROACHING:
                if len(self.items.data) == 0:
                    self.state = State.SEARCHING
                    return

                # Find closest item
                closest_item = max(self.items.data, key=lambda x: x.diameter)
                
                # Calculate approach parameters
                angle_to_item = closest_item.x / 320.0
                distance = 32.4 * float(closest_item.diameter) ** -0.75

                if distance < PICKUP_DISTANCE:
                    self.state = State.COLLECTING
                    return

                # Proportional control for approach
                msg = Twist()
                msg.linear.x = 0.25 * distance
                msg.angular.z = angle_to_item
                self.cmd_vel_publisher.publish(msg)

            case State.COLLECTING:
                # Stop movement
                self.cmd_vel_publisher.publish(Twist())
                
                # Attempt pickup
                request = ItemRequest.Request()
                request.robot_id = self.robot_id
                
                try:
                    future = self.pick_up_service.call_async(request)
                    rclpy.spin_until_future_complete(self, future)
                    response = future.result()
                    
                    if response.success:
                        self.get_logger().info('Item collected successfully')
                        self.state = State.NAVIGATING
                    else:
                        self.get_logger().info(f'Failed to collect item: {response.message}')
                        self.state = State.APPROACHING
                except Exception as e:
                    self.get_logger().error(f'Service call failed: {str(e)}')
                    self.state = State.SEARCHING

            case State.NAVIGATING:
                if not self.current_item:
                    self.state = State.SEARCHING
                    return
                
                # Select appropriate zone based on item color
                # This is where you'd implement zone selection logic
                target_zone = ZONES['cyan']  # Example - implement proper selection
                
                if self.move_to_pose(target_zone['x'], target_zone['y']):
                    self.state = State.DEPOSITING

            case State.DEPOSITING:
                request = ItemRequest.Request()
                request.robot_id = self.robot_id
                
                try:
                    future = self.offload_service.call_async(request)
                    rclpy.spin_until_future_complete(self, future)
                    response = future.result()
                    
                    if response.success:
                        self.get_logger().info('Item deposited successfully')
                    else:
                        self.get_logger().info(f'Failed to deposit item: {response.message}')
                    
                    self.state = State.SEARCHING
                except Exception as e:
                    self.get_logger().error(f'Service call failed: {str(e)}')
                    self.state = State.SEARCHING

            case State.AVOIDING:
                # Basic obstacle avoidance
                msg = Twist()
                if self.scan_triggered[SCAN_LEFT] and self.scan_triggered[SCAN_RIGHT]:
                    msg.angular.z = ANGULAR_VELOCITY * random.choice([TURN_LEFT, TURN_RIGHT])
                elif self.scan_triggered[SCAN_LEFT]:
                    msg.angular.z = ANGULAR_VELOCITY * TURN_RIGHT
                else:
                    msg.angular.z = ANGULAR_VELOCITY * TURN_LEFT
                
                self.cmd_vel_publisher.publish(msg)
                
                if not any(self.scan_triggered):
                    self.state = State.SEARCHING

    def destroy_node(self):
        msg = Twist()
        self.cmd_vel_publisher.publish(msg)
        self.get_logger().info(f"Stopping: {msg}")
        super().destroy_node()


def main(args=None):

    rclpy.init(args = args, signal_handler_options = SignalHandlerOptions.NO)

    node = RobotController()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
