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
from assessment_interfaces.msg import Item, ItemList
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
 # Array indexes for sensor sectors
SCAN_FRONT = 0
SCAN_LEFT = 1
SCAN_BACK = 2
SCAN_RIGHT = 3

# Zone positions (x, y coordinates)
ZONES = {
    'cyan': (2.57, 2.5),    # Top right
    'purple': (-2.57, 2.5),  # Top left
    'green': (-2.57, -2.5),  # Bottom left
    'pink': (2.57, -2.5)    # Bottom right
}

# Color mapping for items to zones
COLOR_TO_ZONE = {
    'red': 'cyan',
    'green': 'green',
    'blue': 'purple'
}

# Finite state machine (FSM) states
class State(Enum):
    FORWARD = 0
    TURNING = 1
    COLLECTING = 2
    OFFLOADING = 3


class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')
        
        # Class variables
        self.state = State.FORWARD
        self.pose = Pose()
        self.yaw = 0.0
        self.items = ItemList()
        self.current_item_color = None
        
        # Parameters
        self.declare_parameter('robot_id', 'robot1')
        self.robot_id = self.get_parameter('robot_id').value
        
        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.item_subscriber = self.create_subscription(
            ItemList,
            '/items',
            self.item_callback,
            10
        )
        
        # Services
        self.pick_up_service = self.create_client(ItemRequest, '/pick_up_item')
        self.offload_service = self.create_client(ItemRequest, '/offload_item')
        
        # Timer
        self.timer = self.create_timer(0.1, self.control_loop)

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
        self.pose = msg.pose.pose
        orientation = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

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
        # Process LiDAR data into 4 sectors (front, left, back, right)
        sector_size = len(msg.ranges) // 4
        for i in range(4):
            start_idx = i * sector_size
            end_idx = (i + 1) * sector_size
            sector_ranges = msg.ranges[start_idx:end_idx]
            # Filter out invalid readings
            valid_ranges = [r for r in sector_ranges if msg.range_min <= r <= msg.range_max]
            if valid_ranges:
                self.scan_triggered[i] = min(valid_ranges) < SCAN_THRESHOLD

    # Control loop for the FSM - called periodically by self.timer
    def control_loop(self):
        # Add debug info about items near the start of control_loop
        if len(self.items.data) > 0:
            item = self.items.data[0]
            self.get_logger().info(f'Current state: {self.state}, Items in view: {len(self.items.data)}, '
                                 f'Nearest item at: ({item.x:.2f}, {item.y:.2f})')
        elif self.state == State.COLLECTING:
            self.get_logger().info('In COLLECTING state but no items visible!')

        # Send message to rviz_text_marker node
        marker_input = StringWithPose()
        marker_input.text = str(self.state) # Visualise robot state as an RViz marker
        marker_input.pose = self.pose # Set the pose of the RViz marker to track the robot's pose
        self.marker_publisher.publish(marker_input)

        #self.get_logger().info(f"{self.state}")
        
        match self.state:

            case State.FORWARD:

                if self.scan_triggered[SCAN_FRONT]:
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = random.uniform(150, 170)
                    self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                    self.get_logger().info("Detected obstacle in front, turning " + ("left" if self.turn_direction == TURN_LEFT else "right") + f" by {self.turn_angle:.2f} degrees")
                    return
                
                if self.scan_triggered[SCAN_LEFT] or self.scan_triggered[SCAN_RIGHT]:
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = 45

                    if self.scan_triggered[SCAN_LEFT] and self.scan_triggered[SCAN_RIGHT]:
                        self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                        self.get_logger().info("Detected obstacle to both the left and right, turning " + ("left" if self.turn_direction == TURN_LEFT else "right") + f" by {self.turn_angle:.2f} degrees")
                    elif self.scan_triggered[SCAN_LEFT]:
                        self.turn_direction = TURN_RIGHT
                        self.get_logger().info(f"Detected obstacle to the left, turning right by {self.turn_angle} degrees")
                    else: # self.scan_triggered[SCAN_RIGHT]
                        self.turn_direction = TURN_LEFT
                        self.get_logger().info(f"Detected obstacle to the right, turning left by {self.turn_angle} degrees")
                    return
                
                if len(self.items.data) > 0:
                    self.state = State.COLLECTING
                    return

                msg = Twist()
                msg.linear.x = LINEAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

                difference_x = self.pose.position.x - self.previous_pose.position.x
                difference_y = self.pose.position.y - self.previous_pose.position.y
                distance_travelled = math.sqrt(difference_x ** 2 + difference_y ** 2)

                # self.get_logger().info(f"Driven {distance_travelled:.2f} out of {self.goal_distance:.2f} metres")

                if distance_travelled >= self.goal_distance:
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = random.uniform(30, 150)
                    self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                    self.get_logger().info("Goal reached, turning " + ("left" if self.turn_direction == TURN_LEFT else "right") + f" by {self.turn_angle:.2f} degrees")

            case State.TURNING:

                self.get_logger().info("Turning state")

                if len(self.items.data) > 0:
                    self.state = State.COLLECTING
                    return

                msg = Twist()
                msg.angular.z = self.turn_direction * ANGULAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

                # self.get_logger().info(f"Turned {math.degrees(math.fabs(yaw_difference)):.2f} out of {self.turn_angle:.2f} degrees")

                yaw_difference = angles.normalize_angle(self.yaw - self.previous_yaw)                

                if math.fabs(yaw_difference) >= math.radians(self.turn_angle):
                    self.previous_pose = self.pose
                    self.goal_distance = random.uniform(1.0, 2.0)
                    self.state = State.FORWARD
                    self.get_logger().info(f"Finished turning, driving forward by {self.goal_distance:.2f} metres")

            case State.COLLECTING:
                if len(self.items.data) == 0:
                    self.previous_pose = self.pose
                    self.goal_distance = random.uniform(1.0, 2.0)
                    self.state = State.FORWARD
                    return
                
                # Choose the closest item based on diameter (larger diameter = closer)
                closest_item = max(self.items.data, key=lambda x: x.diameter)
                msg = Twist()

                # x is left/right in camera view (negative is right)
                # y is up/down in camera view (negative is down)
                angle_to_item = closest_item.x / 320.0  # Convert pixel x to normalized angle
                estimated_distance = 32.4 * float(closest_item.diameter) ** -0.75
                
                self.get_logger().info(f'Item detected - Distance: {estimated_distance:.2f}m, Angle: {math.degrees(angle_to_item):.2f}°')

                if estimated_distance < 0.35:  # DISTANCE from item_manager.py
                    msg.linear.x = 0.0
                    msg.angular.z = 0.0
                    self.cmd_vel_publisher.publish(msg)
                    
                    rqt = ItemRequest.Request()
                    rqt.robot_id = self.robot_id
                    try:
                        future = self.pick_up_service.call_async(rqt)
                        rclpy.spin_until_future_complete(self, future)
                        response = future.result()
                        if response.success:
                            self.get_logger().info('Successfully picked up item!')
                            self.current_item_color = closest_item.colour
                            self.items.data = []  # Clear items list after pickup
                            self.state = State.OFFLOADING
                            return
                        else:
                            self.get_logger().info('Failed to pick up item: ' + response.message)
                            # Move slightly closer if pickup failed
                            msg.linear.x = 0.05
                            self.cmd_vel_publisher.publish(msg)
                    except Exception as e:
                        self.get_logger().info(f'Service call failed: {str(e)}')
                else:
                    # Use proportional control for smoother approach
                    msg.linear.x = 0.25 * estimated_distance  # Proportional to distance
                    msg.angular.z = angle_to_item  # Proportional to angle offset
                    self.cmd_vel_publisher.publish(msg)
            case State.OFFLOADING:
                # Get the target zone for this item's color
                target_zone = COLOR_TO_ZONE.get(self.current_item_color)
                if not target_zone:
                    self.get_logger().error(f'No zone mapping for color {self.current_item_color}')
                    self.state = State.FORWARD
                    return
                
                zone_pos = ZONES[target_zone]
                
                # Calculate distance to target zone
                dx = zone_pos[0] - self.pose.position.x
                dy = zone_pos[1] - self.pose.position.y
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < 0.5:  # Close enough to zone
                    # Try to offload
                    rqt = ItemRequest.Request()
                    rqt.robot_id = self.robot_id
                    try:
                        future = self.offload_service.call_async(rqt)
                        rclpy.spin_until_future_complete(self, future)
                        response = future.result()
                        if response.success:
                            self.get_logger().info(f'Successfully offloaded {self.current_item_color} item in {target_zone} zone!')
                            self.current_item_color = None
                            self.state = State.FORWARD
                        else:
                            self.get_logger().info('Failed to offload item: ' + response.message)
                    except Exception as e:
                        self.get_logger().info(f'Service call failed: {str(e)}')
                else:
                    # Move towards zone
                    msg = Twist()
                    msg.linear.x = min(LINEAR_VELOCITY, distance * 0.5)
                    # Calculate angle to zone
                    target_angle = math.atan2(dy, dx)
                    angle_diff = angles.normalize_angle(target_angle - self.yaw)
                    msg.angular.z = max(-ANGULAR_VELOCITY, min(ANGULAR_VELOCITY, angle_diff))
                    self.cmd_vel_publisher.publish(msg)
            case _:
                pass
        
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