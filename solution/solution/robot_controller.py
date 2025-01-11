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
from rclpy.duration import Duration

from std_msgs.msg import Float32
from geometry_msgs.msg import Twist, Pose, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from assessment_interfaces.msg import Item, ItemList, Zone, ZoneList
from auro_interfaces.msg import StringWithPose
from auro_interfaces.srv import ItemRequest

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

from tf_transformations import euler_from_quaternion
import angles

from enum import Enum
import random
import math

LINEAR_VELOCITY  = 0.3 # Metres per second
ANGULAR_VELOCITY = 0.5 # Radians per second

TURN_LEFT = 1 # Postive angular velocity turns left
TURN_RIGHT = -1 # Negative angular velocity turns right

SCAN_THRESHOLD = 0.3 # Metres per second
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
    SET_GOAL = 3
    NAVIGATING_TO_ZONE = 4

# Map item colors to zones with their map coordinates
ZONE_POSITIONS = {
    'ZONE_PINK': {'x': 2.5, 'y': 2.5},    # Pink zone in top left
    'ZONE_GREEN': {'x': 2.5, 'y': -2.5},   # Green zone in top right
    'ZONE_PURPLE': {'x': -2.5, 'y': -2.5}, # Purple zone in bottom right
    'ZONE_CYAN': {'x': -3.5, 'y': 2.5}     # Cyan zone in bottom left
}

# Map item colors to zones
ITEM_TO_ZONE = {
    'RED': 'ZONE_CYAN',     # Red items go to cyan zone
    'GREEN': 'ZONE_GREEN',  # Green items go to green zone
    'BLUE': 'ZONE_PINK'     # Blue items go to pink zone
}

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
        self.zones = ZoneList()
        self.current_zone_target = None
        self.obstacle_counter = 0  # Counter for persistent obstacles
        self.current_item_id = None  # Track the current item being pursued
        self.blocked_items = set()  # Set to store IDs of items that were blocked by obstacles

        self.robot_id = self.get_namespace().strip("/")
        
        # Initialize navigation
        self.navigator = BasicNavigator()
        
        # Set initial pose for navigation based on robot ID
        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.get_clock().now().to_msg()
        
        # Hardcode different initial poses for each robot - matching assessment/config/initial_poses.yaml
        if self.robot_id == 'robot1':
            initial_pose.pose.position.x = -3.5  # Left side
            initial_pose.pose.position.y = 2.0   # Top position
            initial_yaw = 0.0  # Facing right (towards center)
        elif self.robot_id == 'robot2':
            initial_pose.pose.position.x = -3.5  # Left side
            initial_pose.pose.position.y = 0.0   # Middle position
            initial_yaw = 0.0  # Facing right (towards center)
        elif self.robot_id == 'robot3':
            initial_pose.pose.position.x = -3.5  # Left side
            initial_pose.pose.position.y = -2.0  # Bottom position
            initial_yaw = 0.0  # Facing right (towards center)
            
        # Convert yaw to quaternion
        initial_pose.pose.orientation.z = math.sin(initial_yaw / 2.0)
        initial_pose.pose.orientation.w = math.cos(initial_yaw / 2.0)
        
        self.navigator.setInitialPose(initial_pose)

        # Wait for navigation to be ready
        self.navigator.waitUntilNav2Active()

       

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

        self.zone_subscriber = self.create_subscription(
            ZoneList,
            'zone',
            self.zone_callback,
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

    def zone_callback(self, msg):
        self.zones = msg


    # Control loop for the FSM - called periodically by self.timer
    def control_loop(self):

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
                    self.state = State.FORWARD
                    self.obstacle_counter = 0
                    self.current_item_id = None
                    return
                
                # Get the closest non-blocked item
                available_items = [item for item in self.items.data if id(item) not in self.blocked_items]
                if not available_items:
                    self.blocked_items.clear()
                    self.state = State.FORWARD
                    self.obstacle_counter = 0
                    self.current_item_id = None
                    self.get_logger().info('All visible items are blocked, returning to FORWARD state')
                    return
                
                item = available_items[0]
                
                # Reset counter for new items
                if self.current_item_id != id(item):
                    self.current_item_id = id(item)
                    self.obstacle_counter = 0
                    self.get_logger().info('Pursuing new item')

                # Calculate distance to item
                estimated_distance = 32.4 * float(item.diameter) ** -0.75
            
                # Try pickup if close enough
                if estimated_distance <= 0.35:
                    rqt = ItemRequest.Request()
                    rqt.robot_id = self.robot_id
                    try:
                        future = self.pick_up_service.call_async(rqt)
                        self.executor.spin_until_future_complete(future)
                        response = future.result()
                        if response.success:
                            self.get_logger().info('Item picked up.')
                            self.obstacle_counter = 0
                            self.current_item_id = None
                            target_zone = ITEM_TO_ZONE.get(item.colour.upper())
                            if target_zone:
                                self.current_zone_target = target_zone
                                self.state = State.SET_GOAL
                                self.get_logger().info(f'Setting goal to {target_zone} for {item.colour} item')
                            else:
                                self.get_logger().info(f'No zone mapping for item color: {item.colour}')
                                self.state = State.FORWARD
                            self.items.data = []
                            return
                        else:
                            self.get_logger().info('Unable to pick up item: ' + response.message)
                    except Exception as e:
                        self.get_logger().info('Exception during pickup: ' + str(e))   

                msg = Twist()

                # Check for persistent obstacles
                if self.scan_triggered[SCAN_FRONT] or \
                   (self.scan_triggered[SCAN_LEFT] and self.scan_triggered[SCAN_RIGHT]):
                    self.obstacle_counter += 1
                    if self.obstacle_counter > 30:  # 3 seconds at 10Hz
                        self.get_logger().info(f'Abandoning item due to persistent obstacles')
                        self.blocked_items.add(self.current_item_id)
                        self.obstacle_counter = 0
                        self.current_item_id = None
                        self.state = State.FORWARD
                        return

                # Obstacle avoidance and movement control
                if self.scan_triggered[SCAN_FRONT]:
                    # Stop and turn away from obstacle
                    msg.linear.x = 0.0
                    msg.angular.z = ANGULAR_VELOCITY * 1.5 * (TURN_LEFT if item.x < 320 else TURN_RIGHT)
                    self.get_logger().info('Front obstacle detected, turning to avoid')
                elif self.scan_triggered[SCAN_LEFT]:
                    # Bias right
                    msg.linear.x = 0.15 * estimated_distance
                    msg.angular.z = -0.3
                    self.get_logger().info('Left obstacle detected, biasing right')
                elif self.scan_triggered[SCAN_RIGHT]:
                    # Bias left
                    msg.linear.x = 0.15 * estimated_distance
                    msg.angular.z = 0.3
                    self.get_logger().info('Right obstacle detected, biasing left')
                else:
                    # Normal approach with centering
                    centering_factor = 1.0 - min(abs(item.x - 320) / 320.0, 0.7)
                    msg.linear.x = 0.2 * estimated_distance * centering_factor
                    msg.angular.z = item.x / 320.0
                
                # Safety velocity limits
                msg.linear.x = max(min(msg.linear.x, 0.3), -0.2)
                msg.angular.z = max(min(msg.angular.z, ANGULAR_VELOCITY * 1.5), -ANGULAR_VELOCITY * 1.5)
                
                self.cmd_vel_publisher.publish(msg)

            case State.SET_GOAL:
                if not self.current_zone_target:
                    self.state = State.FORWARD
                    return

                # Get the zone position from our mapping
                zone_pos = ZONE_POSITIONS.get(self.current_zone_target)
                if not zone_pos:
                    self.get_logger().info(f'No position mapping for zone: {self.current_zone_target}')
                    self.state = State.FORWARD
                    return

                # Create the goal pose
                goal_pose = PoseStamped()
                goal_pose.header.frame_id = 'map'
                goal_pose.header.stamp = self.get_clock().now().to_msg()
                goal_pose.pose.position.x = zone_pos['x']
                goal_pose.pose.position.y = zone_pos['y']
                
                # Calculate orientation to face center of arena
                dx = 0.0 - zone_pos['x']  # Vector to center x
                dy = 0.0 - zone_pos['y']  # Vector to center y
                angle = math.atan2(dy, dx)  # Calculate angle to face center
                
                # Convert angle to quaternion
                goal_pose.pose.orientation.z = math.sin(angle / 2.0)
                goal_pose.pose.orientation.w = math.cos(angle / 2.0)

                # Send the goal to the navigator
                nav_path = self.navigator.goToPose(goal_pose)
                self.get_logger().info(f'Goal set to zone at x:{zone_pos["x"]:.2f} y:{zone_pos["y"]:.2f} angle:{math.degrees(angle):.2f}°')
                
                # Move to navigation state
                self.state = State.NAVIGATING_TO_ZONE

            case State.NAVIGATING_TO_ZONE:
                if not self.navigator.isTaskComplete():
                    feedback = self.navigator.getFeedback()
                    if feedback:
                        self.get_logger().info('Estimated time to zone: ' + '{0:.0f}'.format(
                            Duration.from_msg(feedback.estimated_time_remaining).nanoseconds / 1e9) + ' seconds.')
                else:
                    result = self.navigator.getResult()
                    if result == TaskResult.SUCCEEDED:
                        self.get_logger().info('Reached zone!')
                        # Try to offload the item
                        rqt = ItemRequest.Request()
                        rqt.robot_id = self.robot_id
                        try:
                            future = self.offload_service.call_async(rqt)
                            self.executor.spin_until_future_complete(future)
                            response = future.result()
                            if response.success:
                                self.get_logger().info('Item offloaded successfully')
                            else:
                                self.get_logger().info('Failed to offload item: ' + response.message)
                        except Exception as e:
                            self.get_logger().info('Exception during offload: ' + str(e))
                    elif result == TaskResult.CANCELED:
                        self.get_logger().info('Navigation canceled!')
                    elif result == TaskResult.FAILED:
                        self.get_logger().info('Navigation failed!')
                    else:
                        self.get_logger().info('Navigation has an invalid return status!')
                    
                    self.current_zone_target = None
                    self.state = State.FORWARD

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