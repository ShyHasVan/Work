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
from auro_interfaces.msg import StringWithPose
from auro_interfaces.srv import ItemRequest
from assessment_interfaces.msg import Item, ItemList

from tf_transformations import euler_from_quaternion
import angles

from enum import Enum
import random
import math

LINEAR_VELOCITY  = 0.3 # Metres per second
ANGULAR_VELOCITY = 0.5 # Radians per second

TURN_LEFT = 1 # Positive angular velocity turns left
TURN_RIGHT = -1 # Negative angular velocity turns right

SCAN_THRESHOLD = 0.5 # Metres per second
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
        self.offload_service = self.create_client(ItemRequest, '/offload_item', callback_group=client_callback_group)

        self.item_subscriber = self.create_subscription(
            ItemList,
            '/items',
            self.item_callback,
            10, callback_group=timer_callback_group
        )

        self.odom_subscriber = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10, callback_group=timer_callback_group
        )

        self.scan_subscriber = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            QoSPresetProfiles.SENSOR_DATA.value, callback_group=timer_callback_group
        )

        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        self.marker_publisher = self.create_publisher(StringWithPose, 'marker_input', 10, callback_group=timer_callback_group)

        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.control_loop, callback_group=timer_callback_group)

    def item_callback(self, msg):
        self.items = msg

        for item in self.items.data:
            self.get_logger().info(f"Detected item - Colour: {item.colour}, Position: ({item.x}, {item.y}), Diameter: {item.diameter}, Value: {item.value}")


    def odom_callback(self, msg):
        self.pose = msg.pose.pose
        (roll, pitch, yaw) = euler_from_quaternion([self.pose.orientation.x,
                                                    self.pose.orientation.y,
                                                    self.pose.orientation.z,
                                                    self.pose.orientation.w])
        self.yaw = yaw

    def scan_callback(self, msg):
        front_ranges = msg.ranges[331:359] + msg.ranges[0:30]
        left_ranges  = msg.ranges[31:90]
        back_ranges  = msg.ranges[91:270]
        right_ranges = msg.ranges[271:330]

        self.scan_triggered[SCAN_FRONT] = min(front_ranges) < SCAN_THRESHOLD 
        self.scan_triggered[SCAN_LEFT]  = min(left_ranges)  < SCAN_THRESHOLD
        self.scan_triggered[SCAN_BACK]  = min(back_ranges)  < SCAN_THRESHOLD
        self.scan_triggered[SCAN_RIGHT] = min(right_ranges) < SCAN_THRESHOLD

    def control_loop(self):
     marker_input = StringWithPose()
     marker_input.text = str(self.state)
     marker_input.pose = self.pose
     self.marker_publisher.publish(marker_input)

     match self.state:
        case State.FORWARD:
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
                else:
                    self.turn_direction = TURN_LEFT
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

            if distance_travelled >= self.goal_distance:
                self.previous_yaw = self.yaw
                self.state = State.TURNING
                self.turn_angle = random.uniform(30, 150)
                self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])

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

        case State.COLLECTING:
            if len(self.items.data) == 0:
                self.previous_pose = self.pose
                self.state = State.FORWARD
                return

            # Sort items by value (highest first) or by distance
            sorted_items = sorted(self.items.data, key=lambda item: item.value, reverse=True)
            target_item = sorted_items[0]

            self.get_logger().info(f"Approaching item: Colour={target_item.colour}, Position=({target_item.x}, {target_item.y}), Diameter={target_item.diameter}, Value={target_item.value}")

            # Calculate estimated distance to item based on diameter
            estimated_distance = 32.4 * float(target_item.diameter) ** -0.75
            self.get_logger().info(f"Estimated distance to item: {estimated_distance:.2f} meters")

            if estimated_distance <= 0.35:
                # Attempt to pick up the item
                rqt = ItemRequest.Request()
                rqt.robot_id = self.robot_id
                try:
                    future = self.pick_up_service.call_async(rqt)
                    self.executor.spin_until_future_complete(future)
                    response = future.result()
                    if response.success:
                        self.get_logger().info(f"Successfully picked up {target_item.colour} item with value {target_item.value}.")
                        self.items.data.remove(target_item)
                    else:
                        self.get_logger().info(f"Failed to pick up item: {response.message}")
                except Exception as e:
                    self.get_logger().info(f"Pick-up service exception: {e}")

                self.state = State.FORWARD
                return

            # Adjust robot movement to approach the target item
            msg = Twist()
            msg.linear.x = min(0.25 * estimated_distance, LINEAR_VELOCITY)
            msg.angular.z = target_item.x / 320.0
            self.cmd_vel_publisher.publish(msg)

        case _:
            self.get_logger().warn("Unknown state in control loop.")

    def destroy_node(self):
        msg = Twist()
        self.cmd_vel_publisher.publish(msg)
        self.get_logger().info(f"Stopping: {msg}")
        super().destroy_node()


def main(args=None):

    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)

    node = RobotController()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass