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

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from assessment_interfaces.msg import Item, ItemList
from auro_interfaces.msg import StringWithPose
from auro_interfaces.srv import ItemRequest

from tf_transformations import euler_from_quaternion
import angles

from enum import Enum

# Constants
LINEAR_VELOCITY = 0.3
ANGULAR_VELOCITY = 0.5
TURN_LEFT = 1
TURN_RIGHT = -1
SCAN_THRESHOLD = 0.5
SCAN_FRONT = 0
SCAN_LEFT = 1
SCAN_BACK = 2
SCAN_RIGHT = 3

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
        
        # Create callback groups
        client_callback_group = MutuallyExclusiveCallbackGroup()
        timer_callback_group = MutuallyExclusiveCallbackGroup()
        
        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscribers
        self.odom_subscriber = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 
            QoSPresetProfiles.SENSOR_DATA.value
        )
        
        self.scan_subscriber = self.create_subscription(
            LaserScan, 'scan', self.scan_callback,
            QoSPresetProfiles.SENSOR_DATA.value
        )
        
        self.items_subscriber = self.create_subscription(
            ItemList, 'items', self.item_callback, 10
        )
        
        # Services
        self.pick_up_service = self.create_client(
            ItemRequest, '/pick_up_item', 
            callback_group=client_callback_group
        )
        
        self.offload_service = self.create_client(
            ItemRequest, '/offload_item',
            callback_group=client_callback_group
        )
        
        # Timer
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Parameters
        self.declare_parameter('robot_id', 'robot1')
        self.robot_id = self.get_parameter('robot_id').value

    def item_callback(self, msg):
        self.items = msg

    def odom_callback(self, msg):
        self.pose = msg.pose.pose
        (roll, pitch, yaw) = euler_from_quaternion([
            self.pose.orientation.x,
            self.pose.orientation.y,
            self.pose.orientation.z,
            self.pose.orientation.w
        ])
        self.yaw = yaw

    def scan_callback(self, msg):
        front_ranges = msg.ranges[331:359] + msg.ranges[0:30]
        left_ranges = msg.ranges[31:90]
        back_ranges = msg.ranges[91:270]
        right_ranges = msg.ranges[271:330]
        
        self.scan_triggered[SCAN_FRONT] = min(front_ranges) < SCAN_THRESHOLD
        self.scan_triggered[SCAN_LEFT] = min(left_ranges) < SCAN_THRESHOLD
        self.scan_triggered[SCAN_BACK] = min(back_ranges) < SCAN_THRESHOLD
        self.scan_triggered[SCAN_RIGHT] = min(right_ranges) < SCAN_THRESHOLD

    def control_loop(self):
        match self.state:
            case State.FORWARD:
                if len(self.items.data) > 0:
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
                    self.get_logger().info("Goal reached, turning " + ("left" if self.turn_direction == TURN_LEFT else "right") + f" by {self.turn_angle:.2f} degrees")

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
                    self.previous_pose = self.pose
                    self.state = State.FORWARD
                    return
                
                item = self.items.data[0]
                estimated_distance = 32.4 * float(item.diameter) ** -0.75

                if estimated_distance <= 0.35:
                    rqt = ItemRequest.Request()
                    rqt.robot_id = self.robot_id
                    try:
                        future = self.pick_up_service.call_async(rqt)
                        rclpy.spin_until_future_complete(self, future)
                        response = future.result()
                        if response.success:
                            self.get_logger().info('Item picked up!')
                            self.state = State.OFFLOADING
                            self.items.data = []
                        else:
                            self.get_logger().info('Unable to pick up item: ' + response.message)
                    except Exception as e:
                        self.get_logger().info(f'Service call failed: {str(e)}')

                msg = Twist()
                msg.linear.x = 0.25 * estimated_distance
                msg.angular.z = item.x / 320.0
                self.cmd_vel_publisher.publish(msg)

            case State.OFFLOADING:
                # Try to offload in the zone
                rqt = ItemRequest.Request()
                rqt.robot_id = self.robot_id
                try:
                    future = self.offload_service.call_async(rqt)
                    rclpy.spin_until_future_complete(self, future)
                    response = future.result()
                    if response.success:
                        self.get_logger().info('Successfully offloaded item!')
                        self.state = State.FORWARD
                except Exception as e:
                    self.get_logger().info(f'Service call failed: {str(e)}')
                    self.state = State.FORWARD

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
