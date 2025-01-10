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
    OFFLOADING = 3  # New state for offloading items

# Color to zone mapping
ITEM_TO_ZONE = {
    'red': Zone.ZONE_PINK,
    'green': Zone.ZONE_GREEN,
    'blue': Zone.ZONE_CYAN,
    'purple': Zone.ZONE_PURPLE
}

class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')
        
        # Class variables used to store persistent values between executions of callbacks and control loop
        self.state = State.FORWARD
        self.pose = Pose()
        self.previous_pose = Pose()
        self.yaw = 0.0
        self.previous_yaw = 0.0
        self.turn_angle = 0.0
        self.turn_direction = TURN_LEFT
        self.goal_distance = random.uniform(1.0, 2.0)
        self.scan_triggered = [False] * 4
        self.items = ItemList()
        self.zones = ZoneList()
        self.current_item = None
        self.offload_attempted = False  # New flag to track offload attempts
        
        self.declare_parameter('robot_id', 'robot1')
        self.robot_id = self.get_parameter('robot_id').value

        # Create callback groups
        client_callback_group = MutuallyExclusiveCallbackGroup()
        timer_callback_group = MutuallyExclusiveCallbackGroup()

        # Services in client callback group
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

        self.zone_subscriber = self.create_subscription(
            ZoneList,
            'zone',
            self.zone_callback,
            10, callback_group=timer_callback_group
        )

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

    def zone_callback(self, msg):
        self.zones = msg

    # Control loop for the FSM - called periodically by self.timer
    def control_loop(self):
        # Create a status message that combines state and sensor information
        status_msg = f'State: {self.state}'
        
        # Add zone information
        if len(self.zones.data) > 0:
            zones_info = [f"{z.zone}({z.x}, {z.y}, size={z.size:.2f})" for z in self.zones.data]
            status_msg += f', Zones in view: {", ".join(zones_info)}'
        else:
            status_msg += ', No zones in view'

        # Add item information if relevant
        if self.state == State.COLLECTING:
            if len(self.items.data) > 0:
                item = self.items.data[0]
                status_msg += f', Nearest item at: ({item.x:.2f}, {item.y:.2f}), Color: {item.colour}'
            else:
                status_msg += ', No items visible'
        elif self.state == State.OFFLOADING:
            if self.current_item:
                status_msg += f', Holding: {self.current_item}'
                target_zone = ITEM_TO_ZONE.get(self.current_item)
                if target_zone:
                    matching_zones = [z for z in self.zones.data if z.zone == target_zone]
                    if matching_zones:
                        zone = matching_zones[0]
                        status_msg += f', Target zone at: ({zone.x}, {zone.y}), Size: {zone.size:.3f}'
                    else:
                        status_msg += f', Target zone ({target_zone}) not visible'
                else:
                    status_msg += f', No matching zone for {self.current_item}'

        # Log the combined status message
        self.get_logger().info(status_msg)

        # Send message to rviz_text_marker node
        marker_input = StringWithPose()
        marker_input.text = str(self.state)
        marker_input.pose = self.pose
        self.marker_publisher.publish(marker_input)
        
        match self.state:
            case State.FORWARD:
                # If we're holding an item, prioritize going to offloading state
                if self.current_item is not None:
                    self.state = State.OFFLOADING
                    self.offload_attempted = False  # Reset the flag when entering offloading state
                    return
                
                if self.scan_triggered[SCAN_FRONT]:
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = random.uniform(150, 170)
                    self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                    return

                if self.scan_triggered[SCAN_LEFT] or self.scan_triggered[SCAN_RIGHT]:
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = 45

                    if self.scan_triggered[SCAN_LEFT] and self.scan_triggered[SCAN_RIGHT]:
                        self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                    elif self.scan_triggered[SCAN_LEFT]:
                        self.turn_direction = TURN_RIGHT
                    else:  # self.scan_triggered[SCAN_RIGHT]
                        self.turn_direction = TURN_LEFT
                    return
                
                if len(self.items.data) > 0 and self.current_item is None:
                    self.state = State.COLLECTING
                    return

                msg = Twist()
                msg.linear.x = LINEAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

                difference_x = self.pose.position.x - self.previous_pose.position.x
                difference_y = self.pose.position.y - self.previous_pose.position.y
                distance_travelled = math.sqrt(difference_x ** 2 + difference_y ** 2)

                if distance_travelled >= self.goal_distance:
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = random.uniform(30, 150)
                    self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])

            case State.TURNING:
                # If we're holding an item, prioritize going to offloading state
                if self.current_item is not None:
                    self.state = State.OFFLOADING
                    self.offload_attempted = False  # Reset the flag when entering offloading state
                    return
                
                if len(self.items.data) > 0 and self.current_item is None:
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

            case State.COLLECTING:
                if len(self.items.data) == 0:
                    self.previous_pose = self.pose
                    self.goal_distance = random.uniform(1.0, 2.0)
                    self.state = State.FORWARD
                    return
                
                closest_item = max(self.items.data, key=lambda x: x.diameter)
                msg = Twist()

                angle_to_item = closest_item.x / 320.0
                estimated_distance = 32.4 * float(closest_item.diameter) ** -0.75

                if estimated_distance < 0.35:
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
                            self.current_item = closest_item.colour
                            self.get_logger().info(f'Successfully picked up {self.current_item} item')
                            
                            # After pickup, search for the correct zone
                            target_zone = ITEM_TO_ZONE.get(self.current_item)
                            if not target_zone:
                                self.get_logger().info(f'No matching zone type for item color: {self.current_item} in mapping {ITEM_TO_ZONE}')
                                self.current_item = None
                                self.state = State.FORWARD
                                return

                            matching_zones = [z for z in self.zones.data if z.zone == target_zone]
                            if matching_zones:
                                closest_zone = matching_zones[0]
                                self.get_logger().info(f'Found matching zone {target_zone} for {self.current_item} item')
                                if closest_zone.size > 0.3:  # If we're already in a valid zone
                                    rqt = ItemRequest.Request()
                                    rqt.robot_id = self.robot_id
                                    try:
                                        future = self.offload_service.call_async(rqt)
                                        rclpy.spin_until_future_complete(self, future)
                                        response = future.result()
                                        if response.success:
                                            self.get_logger().info(f'Successfully offloaded {self.current_item} item')
                                            self.current_item = None
                                            self.state = State.FORWARD
                                        else:
                                            self.get_logger().info(f'Failed to offload item: {response.message}')
                                            self.state = State.OFFLOADING
                                    except Exception as e:
                                        self.get_logger().info(f'Offload service call failed: {str(e)}')
                                        self.state = State.OFFLOADING
                                else:
                                    self.state = State.OFFLOADING
                            else:
                                # No matching zone visible, start searching by turning
                                self.get_logger().info(f'No matching zone visible for {self.current_item} item, starting search')
                                msg = Twist()
                                msg.angular.z = ANGULAR_VELOCITY  # Start turning to search for zone
                                self.cmd_vel_publisher.publish(msg)
                                
                                # Keep turning until we see the zone or complete a full rotation
                                search_start_time = self.get_clock().now()
                                while len([z for z in self.zones.data if z.zone == target_zone]) == 0:
                                    # Check if we've completed a full rotation (approximately 6.28 radians)
                                    if (self.get_clock().now() - search_start_time).nanoseconds / 1e9 > (2 * math.pi / ANGULAR_VELOCITY):
                                        break
                                    rclpy.spin_once(self, timeout_sec=0.1)
                                
                                # Stop turning
                                msg.angular.z = 0.0
                                self.cmd_vel_publisher.publish(msg)
                                
                                # If we still haven't found the zone, move forward and try again
                                if len([z for z in self.zones.data if z.zone == target_zone]) == 0:
                                    msg.linear.x = LINEAR_VELOCITY
                                    self.cmd_vel_publisher.publish(msg)
                                    rclpy.sleep(1.0)  # Move forward for 1 second
                                    msg.linear.x = 0.0
                                    self.cmd_vel_publisher.publish(msg)
                                
                                self.state = State.OFFLOADING
                        else:
                            self.get_logger().info(f'Failed to pick up item: {response.message}')
                            msg.linear.x = 0.05
                            self.cmd_vel_publisher.publish(msg)
                    except Exception as e:
                        self.get_logger().info(f'Service call failed: {str(e)}')
                else:
                    msg.linear.x = 0.25 * estimated_distance
                    msg.angular.z = angle_to_item
                    self.cmd_vel_publisher.publish(msg)

            case State.OFFLOADING:
                if not self.current_item:
                    self.get_logger().info('No item to offload, returning to FORWARD state')
                    self.state = State.FORWARD
                    return

                target_zone = ITEM_TO_ZONE.get(self.current_item)
                if not target_zone:
                    self.get_logger().info(f'No matching zone for item color: {self.current_item}, dropping item')
                    self.current_item = None
                    self.state = State.FORWARD
                    return

                matching_zones = [z for z in self.zones.data if z.zone == target_zone]
                if not matching_zones:
                    if not self.offload_attempted:
                        self.get_logger().info('No matching zone visible, turning to search')
                        self.previous_yaw = self.yaw
                        self.state = State.TURNING
                        self.turn_angle = random.uniform(30, 90)
                        self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                    else:
                        # If we've already attempted to offload, try moving forward to find a zone
                        msg = Twist()
                        msg.linear.x = LINEAR_VELOCITY
                        self.cmd_vel_publisher.publish(msg)
                    return

                closest_zone = matching_zones[0]
                msg = Twist()

                angle_to_zone = closest_zone.x / 320.0
                zone_size = closest_zone.size

                self.get_logger().info(f'Approaching zone: {target_zone}, Angle: {angle_to_zone:.2f}, Size: {zone_size:.2f}')

                if zone_size > 0.3:
                    msg.linear.x = 0.0
                    msg.angular.z = 0.0
                    self.cmd_vel_publisher.publish(msg)
                    
                    if not self.offload_attempted:
                        self.offload_attempted = True
                        rqt = ItemRequest.Request()
                        rqt.robot_id = self.robot_id
                        try:
                            future = self.offload_service.call_async(rqt)
                            rclpy.spin_until_future_complete(self, future)
                            response = future.result()
                            if response.success:
                                self.get_logger().info(f'Successfully offloaded {self.current_item} item')
                                self.current_item = None
                                self.state = State.FORWARD
                            else:
                                self.get_logger().info(f'Failed to offload item: {response.message}')
                                # If offload fails, back up slightly and try again
                                msg.linear.x = -0.1
                                self.cmd_vel_publisher.publish(msg)
                        except Exception as e:
                            self.get_logger().info(f'Service call failed: {str(e)}')
                    else:
                        # If we've already attempted to offload, try backing up and approaching again
                        msg.linear.x = -0.1
                        self.cmd_vel_publisher.publish(msg)
                        if zone_size < 0.25:  # If we've backed up enough
                            self.offload_attempted = False
                else:
                    # Approach the zone more carefully when close
                    approach_speed = 0.15 if zone_size > 0.2 else 0.25
                    msg.linear.x = approach_speed * (1.0 - zone_size)
                    msg.angular.z = angle_to_zone
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
