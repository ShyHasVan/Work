# Copyright (c) 2024 University of York and others
# SPDX-License-Identifier: EPL-2.0

import sys
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from auro_interfaces.msg import Item, ItemList
from auro_interfaces.srv import ItemRequest
from tf_transformations import euler_from_quaternion
import angles
from enum import Enum
import random
import math

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

class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')
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

        self.declare_parameter('robot_id', 'robot1')
        self.robot_id = self.get_parameter('robot_id').value

        client_callback_group = MutuallyExclusiveCallbackGroup()
        timer_callback_group = MutuallyExclusiveCallbackGroup()

        self.pick_up_service = self.create_client(ItemRequest, '/pick_up_item', callback_group=client_callback_group)
        self.item_subscriber = self.create_subscription(
            ItemList, '/items', self.item_callback, 10, callback_group=timer_callback_group)
        self.odom_subscriber = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10, callback_group=timer_callback_group)
        self.scan_subscriber = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10, callback_group=timer_callback_group)
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.control_loop, callback_group=timer_callback_group)

    def item_callback(self, msg):
        self.items = msg

    def odom_callback(self, msg):
        self.pose = msg.pose.pose
        _, _, self.yaw = euler_from_quaternion([
            self.pose.orientation.x,
            self.pose.orientation.y,
            self.pose.orientation.z,
            self.pose.orientation.w])

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
                if self.scan_triggered[SCAN_FRONT]:
                    self.transition_to_turning()
                    return
                if len(self.items.data) > 0:
                    self.state = State.COLLECTING
                    return
                self.move_forward()
            case State.TURNING:
                if len(self.items.data) > 0:
                    self.state = State.COLLECTING
                    return
                self.turn_robot()
            case State.COLLECTING:
                if len(self.items.data) == 0:
                    self.state = State.FORWARD
                    return
                self.collect_item()

    def transition_to_turning(self):
        self.previous_yaw = self.yaw
        self.state = State.TURNING
        self.turn_angle = random.uniform(150, 170)
        self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])

    def move_forward(self):
        msg = Twist()
        msg.linear.x = LINEAR_VELOCITY
        self.cmd_vel_publisher.publish(msg)
        dx = self.pose.position.x - self.previous_pose.position.x
        dy = self.pose.position.y - self.previous_pose.position.y
        distance_travelled = math.sqrt(dx ** 2 + dy ** 2)
        if distance_travelled >= self.goal_distance:
            self.transition_to_turning()

    def turn_robot(self):
        msg = Twist()
        msg.angular.z = self.turn_direction * ANGULAR_VELOCITY
        self.cmd_vel_publisher.publish(msg)
        yaw_diff = angles.normalize_angle(self.yaw - self.previous_yaw)
        if abs(yaw_diff) >= math.radians(self.turn_angle):
            self.previous_pose = self.pose
            self.goal_distance = random.uniform(1.0, 2.0)
            self.state = State.FORWARD

    def collect_item(self):
        item = self.items.data[0]
        estimated_distance = 32.4 * float(item.diameter) ** -0.75
        if estimated_distance <= 0.35:
            req = ItemRequest.Request()
            req.robot_id = self.robot_id
            try:
                future = self.pick_up_service.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                response = future.result()
                if response.success:
                    self.items.data.pop(0)
                    self.state = State.FORWARD
            except Exception as e:
                self.get_logger().error(f"Failed to pick up item: {e}")
        else:
            msg = Twist()
            msg.linear.x = 0.25 * estimated_distance
            msg.angular.z = item.x / 320.0
            self.cmd_vel_publisher.publish(msg)

    def destroy_node(self):
        self.cmd_vel_publisher.publish(Twist())
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
