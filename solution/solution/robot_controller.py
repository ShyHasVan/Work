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
    FORWARD = 0       # Basic movement state (from week 4/5)
    TURNING = 1       # Turning to avoid obstacles or explore
    COLLECTING = 2    # Moving towards and picking up items
    OFFLOADING = 3    # Depositing items in zones


class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')
        
        # Class variables used to store persistent values between executions of callbacks and control loop
        self.state = State.FORWARD # Current FSM state
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

        # Track zone states internally
        self.zone_colors = {
            'cyan': None,
            'purple': None,
            'green': None,
            'pink': None
        }
        self.current_zone = None  # Track which zone we're in

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
        # Check for obstacles first
        if self.scan_triggered[SCAN_FRONT]:
            self.get_logger().info('Front obstacle detected while moving to pose')
            return False
        
        if self.scan_triggered[SCAN_LEFT] or self.scan_triggered[SCAN_RIGHT]:
            self.get_logger().info('Side obstacle detected while moving to pose')
            return False
        
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

    # Add new method to check if in a zone
    def check_current_zone(self):
        for zone_name, pos in ZONES.items():
            if math.dist((self.pose.position.x, self.pose.position.y), (pos['x'], pos['y'])) < 0.5:
                return zone_name
        return None

    def handle_obstacle_avoidance(self):
        """Handle obstacle avoidance and return True if obstacle detected"""
        if self.scan_triggered[SCAN_FRONT]:
            self.get_logger().info('Front obstacle detected, making large turn')
            self.previous_yaw = self.yaw
            self.state = State.TURNING
            self.turn_angle = random.uniform(150, 170)
            self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
            return True
        
        if self.scan_triggered[SCAN_LEFT] or self.scan_triggered[SCAN_RIGHT]:
            self.get_logger().info('Side obstacle detected, turning away')
            self.previous_yaw = self.yaw
            self.state = State.TURNING
            self.turn_angle = random.uniform(45, 90)
            self.turn_direction = TURN_RIGHT if self.scan_triggered[SCAN_LEFT] else TURN_LEFT
            return True
        
        return False

    # Control loop for the FSM - called periodically by self.timer
    def control_loop(self):
        self.get_logger().info(f'State: {self.state.name}')
        
        match self.state:
            case State.FORWARD:
                self.get_logger().info(f"Moving forward {self.goal_distance:.2f} metres")
                
                # Check for items first
                if len(self.items.data) > 0:
                    self.state = State.COLLECTING
                    return

                # Handle obstacle detection and movement
                msg = Twist()
                if self.scan_triggered[SCAN_FRONT]:
                    msg.linear.x = 0.0
                    self.cmd_vel_publisher.publish(msg)
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = random.uniform(150, 170)  # Larger turn angle for front obstacles
                    self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                    return
                
                if self.scan_triggered[SCAN_LEFT] or self.scan_triggered[SCAN_RIGHT]:
                    msg.linear.x = 0.0
                    self.cmd_vel_publisher.publish(msg)
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = random.uniform(45, 90)
                    self.turn_direction = TURN_RIGHT if self.scan_triggered[SCAN_LEFT] else TURN_LEFT
                    return

                # Forward movement
                msg.linear.x = LINEAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

                # In FORWARD state
                if self.handle_obstacle_avoidance():
                    return

                # In FORWARD state after obstacle check
                difference_x = self.pose.position.x - self.previous_pose.position.x
                difference_y = self.pose.position.y - self.previous_pose.position.y
                distance_travelled = math.sqrt(difference_x ** 2 + difference_y ** 2)

                if distance_travelled >= self.goal_distance:
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = random.uniform(30, 150)
                    self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                    self.get_logger().info("Goal reached, turning " + ("left" if self.turn_direction == TURN_LEFT else "right") + f" by {self.turn_angle:.2f} degrees")
                    return

            case State.TURNING:
                self.get_logger().info(f"Turning {self.turn_direction} by {self.turn_angle:.2f} degrees")
                if len(self.items.data) > 0:
                    self.state = State.COLLECTING
                    return

                msg = Twist()
                msg.angular.z = self.turn_direction * ANGULAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

                yaw_difference = angles.normalize_angle(self.yaw - self.previous_yaw)

                if math.fabs(yaw_difference) >= math.radians(self.turn_angle):
                    self.previous_pose = self.pose
                    self.goal_distance = random.uniform(0.5, 1.0)
                    self.state = State.FORWARD
                    self.get_logger().info(f"Finished turning, driving forward by {self.goal_distance:.2f} metres")

            case State.COLLECTING:
                if len(self.items.data) == 0:
                    self.previous_pose = self.pose
                    self.state = State.FORWARD
                    return

                item = self.items.data[0]
                estimated_distance = 32.4 * float(item.diameter) ** -0.75
                self.get_logger().info(f"Collecting item at distance {estimated_distance:.2f}")

                if estimated_distance <= PICKUP_DISTANCE:
                    request = ItemRequest.Request()
                    request.robot_id = self.robot_id
                    try:
                        future = self.pick_up_service.call_async(request)
                        rclpy.spin_until_future_complete(self, future)
                        response = future.result()
                        if response.success:
                            self.get_logger().info('Item picked up')
                            self.current_item = item
                            self.state = State.OFFLOADING
                        else:
                            self.get_logger().info('Unable to pick up item: ' + response.message)
                            # Move away slightly and try again
                            self.previous_pose = self.pose
                            self.goal_distance = random.uniform(0.5, 1.0)
                            self.state = State.FORWARD
                    except Exception as e:
                        self.get_logger().error(f'Service call failed: {str(e)}')
                        self.state = State.FORWARD
                        return

                msg = Twist()
                msg.linear.x = 0.25 * estimated_distance
                msg.angular.z = item.x / 320.0
                self.cmd_vel_publisher.publish(msg)
                return

            case State.OFFLOADING:
                if not self.current_item:
                    self.get_logger().info('No item held, returning to FORWARD')
                    self.state = State.FORWARD
                    return

                # Check if we're already in a zone
                self.current_zone = self.check_current_zone()
                
                if self.current_zone:
                    zone_color = self.zone_colors[self.current_zone]
                    if zone_color is None or zone_color == self.current_item.colour:
                        request = ItemRequest.Request()
                        request.robot_id = self.robot_id
                        try:
                            future = self.offload_service.call_async(request)
                            rclpy.spin_until_future_complete(self, future)
                            response = future.result()
                            if response.success:
                                self.get_logger().info(f'Successfully offloaded item in {self.current_zone} zone')
                                # Only update zone color if this was a successful deposit
                                if 'deposited' in response.message.lower():
                                    if self.zone_colors[self.current_zone] is None:
                                        self.zone_colors[self.current_zone] = self.current_item.colour
                                        self.get_logger().info(f'Zone {self.current_zone} now accepts {self.current_item.colour} items')
                            else:
                                # Item was offloaded but not deposited
                                self.get_logger().info(f'Failed to deposit in {self.current_zone} zone: {response.message}')
                                self.state = State.FORWARD
                        except Exception as e:
                            self.get_logger().error(f'Service call failed: {str(e)}')
                            self.state = State.FORWARD
                        return
                
                # Find and move to suitable zone
                target_zone = None
                blocked_attempts = 0  # Track failed attempts

                # Try each zone until we find one we can reach
                for zone_name, color in self.zone_colors.items():
                    if color is None or color == self.current_item.colour:
                        target_zone = zone_name
                        self.get_logger().info(f'Attempting to move to {zone_name} zone')
                        
                        # Try to move to this zone
                        if self.move_to_pose(ZONES[target_zone]['x'], ZONES[target_zone]['y']):
                            return
                        
                        # If blocked by obstacles, try another zone
                        blocked_attempts += 1
                        if blocked_attempts >= len(ZONES):
                            self.get_logger().info('All zones blocked, offloading item in current location')
                            request = ItemRequest.Request()
                            request.robot_id = self.robot_id
                            try:
                                future = self.offload_service.call_async(request)
                                rclpy.spin_until_future_complete(self, future)
                                response = future.result()
                                if response.success:
                                    self.get_logger().info('Item offloaded in current location')
                                    self.state = State.FORWARD
                                else:
                                    self.get_logger().error('Offload failed: ' + response.message)
                                    self.state = State.FORWARD
                            except Exception as e:
                                self.get_logger().error(f'Offload failed: {str(e)}')
                                self.state = State.FORWARD
                            return

                if target_zone:
                    # More aggressive obstacle avoidance
                    if self.handle_obstacle_avoidance():
                        return
                    
                    if self.move_to_pose(ZONES[target_zone]['x'], ZONES[target_zone]['y']):
                        return
                else:
                    # No suitable zone, offload item where we are
                    self.get_logger().info('No suitable zone found, offloading item in current location')
                    request = ItemRequest.Request()
                    request.robot_id = self.robot_id
                    try:
                        future = self.offload_service.call_async(request)
                        rclpy.spin_until_future_complete(self, future)
                        response = future.result()
                        if response.success:
                            self.get_logger().info('Item offloaded in current location')
                    except Exception as e:
                        self.get_logger().error(f'Service call failed: {str(e)}')
                    
                self.state = State.FORWARD

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
