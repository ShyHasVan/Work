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
import random
import math

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.qos import QoSPresetProfiles
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration

from geometry_msgs.msg import Twist, Pose, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from assessment_interfaces.msg import Item, ItemList
from auro_interfaces.msg import StringWithPose
from auro_interfaces.srv import ItemRequest

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import angles

from enum import Enum

# Constants
LINEAR_VELOCITY = 0.3  # Metres per second
ANGULAR_VELOCITY = 0.5  # Radians per second
TURN_LEFT = 1  # Positive angular velocity turns left
TURN_RIGHT = -1  # Negative angular velocity turns right
SCAN_THRESHOLD = 0.5  # Metres
SCAN_FRONT = 0
SCAN_LEFT = 1
SCAN_BACK = 2
SCAN_RIGHT = 3

# FSM States
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
        self.previous_pose = Pose()
        self.yaw = 0.0
        self.previous_yaw = 0.0
        self.turn_angle = 0.0
        self.turn_direction = TURN_LEFT
        self.goal_distance = random.uniform(1.0, 2.0)
        self.scan_triggered = [False] * 4
        self.items = ItemList()
        
        # Initialize Nav2 later when needed
        self.navigator = None
        
        # Create callback groups
        client_callback_group = MutuallyExclusiveCallbackGroup()
        timer_callback_group = MutuallyExclusiveCallbackGroup()
        
        # Initialize publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.marker_publisher = self.create_publisher(StringWithPose, 'rviz_text_marker', 10)
        
        # Initialize subscribers
        self.odom_subscriber = self.create_subscription(Odometry, 'odom', self.odom_callback, QoSPresetProfiles.SENSOR_DATA.value)
        self.scan_subscriber = self.create_subscription(LaserScan, 'scan', self.scan_callback, QoSPresetProfiles.SENSOR_DATA.value)
        self.items_subscriber = self.create_subscription(ItemList, 'items', self.items_callback, 10)
        
        # Initialize services
        self.pick_up_service = self.create_client(ItemRequest, 'pick_up_item', callback_group=client_callback_group)
        self.offload_service = self.create_client(ItemRequest, 'offload_item', callback_group=client_callback_group)
        
        # Initialize timer (needed for control loop)
        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.control_loop, callback_group=timer_callback_group)
        
        # Parameters
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
                            self.state = State.OFFLOADING
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
                # Initialize Nav2 if needed
                if self.navigator is None:
                    if self.init_navigation():
                        self.get_logger().info('Nav2 initialized')
                    return
                
                # Set goal pose for the zone
                goal_pose = PoseStamped()
                goal_pose.header.frame_id = 'map'
                goal_pose.header.stamp = self.get_clock().now().to_msg()
                goal_pose.pose.position.x = 2.57
                goal_pose.pose.position.y = 2.5
                goal_pose.pose.orientation.w = 1.0
                
                # Start navigation if not already started
                if not hasattr(self, 'nav_started'):
                    self.get_logger().info('Starting navigation to zone')
                    # Stop any current movement
                    stop_msg = Twist()
                    self.cmd_vel_publisher.publish(stop_msg)
                    # Start navigation
                    self.navigator.goToPose(goal_pose)
                    self.nav_started = True
                    return
                
                # Check navigation progress
                if not self.navigator.isTaskComplete():
                    # Optional: Add timeout like in week_8
                    return
                    
                # Navigation completed
                result = self.navigator.getResult()
                if result == TaskResult.SUCCEEDED:
                    # Try to offload
                    rqt = ItemRequest.Request()
                    rqt.robot_id = self.robot_id
                    try:
                        future = self.offload_service.call_async(rqt)
                        rclpy.spin_until_future_complete(self, future)
                        response = future.result()
                        if response.success:
                            self.get_logger().info('Successfully offloaded item!')
                            self.nav_started = False
                            self.state = State.FORWARD
                    except Exception as e:
                        self.get_logger().info(f'Service call failed: {str(e)}')
                else:
                    self.get_logger().warn(f'Navigation failed with result: {result}')
                    self.nav_started = False
                    self.state = State.FORWARD
            case _:
                pass
        
    def destroy_node(self):
        msg = Twist()
        self.cmd_vel_publisher.publish(msg)
        self.get_logger().info(f"Stopping: {msg}")
        super().destroy_node()

    def init_navigation(self):
        if self.navigator is None:
            self.navigator = BasicNavigator()
            
            # Create transform buffer and listener
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            
            # Wait for transform to be available
            try:
                self.get_logger().info('Waiting for transform...')
                # Wait for transform between map and base_link
                while not self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time()):
                    self.get_logger().info('Waiting for transform between map and base_link...')
                    rclpy.sleep(1.0)
                
                # Get the current transform
                transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
                
                # Set initial pose using the transform
                initial_pose = PoseStamped()
                initial_pose.header.frame_id = 'map'
                initial_pose.header.stamp = self.get_clock().now().to_msg()
                initial_pose.pose.position.x = transform.transform.translation.x
                initial_pose.pose.position.y = transform.transform.translation.y
                initial_pose.pose.orientation = transform.transform.rotation
                
                self.navigator.setInitialPose(initial_pose)
                self.navigator.waitUntilNav2Active()
                self.get_logger().info('Nav2 initialized successfully')
                return True
                
            except TransformException as ex:
                self.get_logger().error(f'Could not transform: {str(ex)}')
                return False
                
        return False

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
