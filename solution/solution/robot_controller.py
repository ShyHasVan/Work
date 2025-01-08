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
from assessment_interfaces.msg import Item, ItemList, Zone, ZoneList
from auro_interfaces.msg import StringWithPose
from auro_interfaces.srv import ItemRequest

from tf_transformations import euler_from_quaternion
import angles

from enum import Enum
import random
import math

from nav2_simple_commander.robot_navigatorimport BasicNavigator
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

LINEAR_VELOCITY  = 0.3 # Metres per second
ANGULAR_VELOCITY = 0.5 # Radians per second

TURN_LEFT = 1 # Postive angular velocity turns left
TURN_RIGHT = -1 # Negative angular velocity turns right

SCAN_THRESHOLD = 0.5 # Metres per second
 # Array indexes for sensor sectors
SCAN_FRONT = 0
SCAN_LEFT = 1
SCAN_BACK = 2
SCAN_RIGHT = 3

# Finite state machine (FSM) states
class State(Enum):
    FORWARD = 0
    TURNING = 1
    COLLECTING = 2
    OFFLOADING = 3  


class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')
        
        # Basic initialization first
        self.pose = Pose()
        self.previous_pose = Pose()
        self.state = State.FORWARD
        
        # Create publishers and basic services first
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Initialize Nav2 after basic setup
        self.navigator = BasicNavigator()
        
        # Wait for transform to be ready
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Set initial pose for Nav2 using odom frame
        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'odom'  # Changed from 'map' to 'odom'
        initial_pose.header.stamp = self.get_clock().now().to_msg()
        initial_pose.pose = self.pose
        
        try:
            self.navigator.setInitialPose(initial_pose)
            self.navigator.waitUntilNav2Active(timeout_sec=10.0)
        except Exception as e:
            self.get_logger().warn(f'Nav2 initialization failed: {str(e)}. Continuing with basic navigation.')
        
        # Class variables used to store persistent values between executions of callbacks and control loop
        self.yaw = 0.0 # Angle the robot is facing (rotation around the Z axis, in radians), relative to the odom reference frame
        self.previous_yaw = 0.0 # Snapshot of the angle for comparison against future angles
        self.turn_angle = 0.0 # Relative angle to turn to in the TURNING state
        self.turn_direction = TURN_LEFT # Direction to turn in the TURNING state
        self.goal_distance = random.uniform(1.0, 2.0) # Goal distance to travel in FORWARD state
        self.scan_triggered = [False] * 4 # Boolean value for each of the 4 LiDAR sensor sectors. True if obstacle detected within SCAN_THRESHOLD
        self.items = ItemList()

        # Add new variables for item handling
        self.holding_item = False
        self.item_color = None
        self.target_zone = None
        
        # Define zones with their positions
        self.zones = {
            'cyan': {'x': -3.0, 'y': 3.0},      # Bottom left
            'purple': {'x': 3.0, 'y': -3.0},    # Bottom right
            'deeppink': {'x': 3.0, 'y': 3.0},   # Top right
            'seagreen': {'x': -3.0, 'y': -3.0}  # Top left
        }

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
        self.marker_publisher = self.create_publisher(StringWithPose, 'marker_input', 10, callback_group=timer_callback_group)

        # Creates a timer that calls the control_loop method repeatedly - each loop represents single iteration of the FSM
        self.timer_period = 0.1 # 100 milliseconds = 10 Hz
        self.timer = self.create_timer(self.timer_period, self.control_loop, callback_group=timer_callback_group)

        # Subscribe to zone information
        self.zone_subscriber = self.create_subscription(
            ZoneList,
            'zones',
            self.zone_callback,
            10,
            callback_group=timer_callback_group
        )

        self.get_logger().info(f'Robot controller initialized with ID: {self.robot_id}')
        self.get_logger().info(f'Initial position: ({self.pose.position.x:.2f}, {self.pose.position.y:.2f})')
        self.get_logger().info(f'Initial state: {self.state}')

    def item_callback(self, msg):
        if len(msg.data) > 0 and len(self.items.data) == 0:
            # Only log when we first see an item
            self.get_logger().info(f'New item detected! Count: {len(msg.data)}')
        self.items = msg

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


    # Control loop for the FSM - called periodically by self.timer
    def control_loop(self):
        # Single clear state logging message
        self.get_logger().info(f'STATE: {self.state.name} | Position: ({self.pose.position.x:.2f}, {self.pose.position.y:.2f}) | ' +
                              f'Holding item: {self.holding_item} | Item color: {self.item_color if self.holding_item else "None"}')
        
        match self.state:
            case State.FORWARD:
                # Check if we see any items
                if len(self.items.data) > 0:
                    self.state = State.COLLECTING
                    return

                # Move forward
                msg = Twist()
                msg.linear.x = LINEAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

                # Calculate distance traveled
                difference_x = self.pose.position.x - self.previous_pose.position.x
                difference_y = self.pose.position.y - self.previous_pose.position.y
                distance_travelled = math.sqrt(difference_x ** 2 + difference_y ** 2)

                if distance_travelled >= self.goal_distance:
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = random.uniform(30, 150)
                    self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                    self.get_logger().info(f"Goal reached, turning {'left' if self.turn_direction == TURN_LEFT else 'right'} by {self.turn_angle:.2f} degrees")

            case State.TURNING:
                if len(self.items.data) > 0:
                    self.state = State.COLLECTING
                    return

                msg = Twist()
                msg.angular.z = self.turn_direction * ANGULAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

                yaw_difference = angles.normalize_angle(self.yaw - self.previous_yaw)

                if math.fabs(yaw_difference) >= math.radians(self.turn_angle):
                    self.previous_pose = self.pose
                    self.goal_distance = random.uniform(1.0, 2.0)
                    self.state = State.FORWARD
                    self.get_logger().info(f"Finished turning, driving forward by {self.goal_distance:.2f} metres")

            case State.COLLECTING:
                if len(self.items.data) == 0:
                    self.state = State.FORWARD
                    return
                
                closest_item = self.items.data[0]
                estimated_distance = 32.4 * float(closest_item.diameter) ** -0.75

                if estimated_distance <= 0.35:
                    # Pickup logic remains the same
                    rqt = ItemRequest.Request()
                    rqt.robot_id = self.robot_id
                    try:
                        future = self.pick_up_service.call_async(rqt)
                        rclpy.spin_until_future_complete(self, future)
                        response = future.result()
                        if response.success:
                            self.holding_item = True
                            self.item_color = closest_item.colour
                            self.state = State.OFFLOADING
                            return
                    except Exception as e:
                        self.get_logger().info(f'Service call failed: {str(e)}')
                
                # Basic movement for collection
                msg = Twist()
                msg.linear.x = max(0.1, min(0.3, 0.25 * estimated_distance))
                msg.angular.z = max(-0.5, min(0.5, closest_item.x / 320.0))
                self.cmd_vel_publisher.publish(msg)

            case State.OFFLOADING:
                if not self.holding_item:
                    self.state = State.FORWARD
                    return

                target, distance = self.get_nearest_zone()
                
                if not target:
                    # No compatible zone found, do random walk (from week 5)
                    msg = Twist()
                    msg.linear.x = LINEAR_VELOCITY
                    self.cmd_vel_publisher.publish(msg)

                    difference_x = self.pose.position.x - self.previous_pose.position.x
                    difference_y = self.pose.position.y - self.previous_pose.position.y
                    distance_travelled = math.sqrt(difference_x ** 2 + difference_y ** 2)

                    if distance_travelled >= self.goal_distance:
                        self.state = State.TURNING
                        self.previous_yaw = self.yaw
                        self.turn_angle = random.uniform(30, 150)
                        self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                        self.previous_pose = self.pose
                    return

                # We found a compatible zone - use week 8's approach
                target_pos = self.zones[target]
                if distance < 0.5:  # Close enough to try offloading
                    rqt = ItemRequest.Request()
                    rqt.robot_id = self.robot_id
                    try:
                        future = self.offload_service.call_async(rqt)
                        rclpy.spin_until_future_complete(self, future)
                        response = future.result()
                        if response.success:
                            self.holding_item = False
                            self.item_color = None
                            self.state = State.FORWARD
                            self.previous_pose = self.pose
                            self.goal_distance = random.uniform(1.0, 2.0)
                    except Exception as e:
                        self.get_logger().info(f'Service call failed: {str(e)}')
                else:
                    # Move towards zone using week 8's movement approach
                    msg = self.move_to_zone(target_pos, distance)
                    self.cmd_vel_publisher.publish(msg)

            case _:
                pass
        
    def destroy_node(self):
        msg = Twist()
        self.cmd_vel_publisher.publish(msg)
        self.get_logger().info(f"Stopping: {msg}")
        super().destroy_node()

    def zone_callback(self, msg):
        self.get_logger().info('Received zone update')
        for zone in msg.data:
            if zone.colour in self.zones:
                # Store the zone's current color assignment if it has one
                self.zones[zone.colour]['assigned_color'] = zone.assigned_colour if zone.assigned_colour else None
                self.get_logger().info(f'Zone {zone.colour} - Position: ({self.zones[zone.colour]["x"]}, {self.zones[zone.colour]["y"]}), Assigned color: {zone.assigned_colour}')

    def get_nearest_zone(self):
        min_dist = float('inf')
        nearest = None
        
        for color, pos in self.zones.items():
            # Check if zone is compatible with our item
            assigned_color = pos.get('assigned_color')
            
            if assigned_color is not None and assigned_color != self.item_color:
                continue
            
            dx = pos['x'] - self.pose.position.x
            dy = pos['y'] - self.pose.position.y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < min_dist:
                min_dist = dist
                nearest = color
        
        return nearest, min_dist

    def move_to_zone(self, target_pos, distance):
        msg = Twist()
        
        # Calculate angle to target (like week 8's RETURNING state)
        angle_to_target = math.atan2(
            target_pos['y'] - self.pose.position.y,
            target_pos['x'] - self.pose.position.x
        )
        angle_diff = angles.normalize_angle(angle_to_target - self.yaw)
        
        # Use week 8's scaling approach
        scale_rotation_rate = 0.5
        msg.angular.z = scale_rotation_rate * angle_diff
        
        scale_forward_speed = 0.25
        msg.linear.x = scale_forward_speed * distance
        
        return msg

    def navigate_to_pose(self, x, y):
        try:
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'odom'  # Changed from 'map' to 'odom'
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            goal_pose.pose.position.x = x
            goal_pose.pose.position.y = y
            goal_pose.pose.orientation.w = 1.0
            
            # Set a shorter timeout for navigation
            self.navigator.goToPose(goal_pose)
            
            timeout_count = 0
            while not self.navigator.isTaskComplete():
                feedback = self.navigator.getFeedback()
                timeout_count += 1
                if timeout_count > 50:  # Shorter timeout
                    self.navigator.cancelTask()
                    return False
                self.get_logger().info('Navigating to goal...')
                
            return self.navigator.isTaskComplete()
            
        except Exception as e:
            self.get_logger().error(f'Navigation error: {str(e)}')
            return False

def main(args=None):
    rclpy.init(args=args)
    
    try:
        robot_controller = RobotController()
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error occurred: {str(e)}')
    finally:
        if 'robot_controller' in locals():
            robot_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()