import sys
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import ExternalShutdownException
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSPresetProfiles

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from assessment_interfaces.msg import ItemList, ZoneList
from auro_interfaces.srv import ItemRequest

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from tf_transformations import euler_from_quaternion
import angles
import random
import math

# Constants for movement
LINEAR_VELOCITY = 0.3
ANGULAR_VELOCITY = 0.5
TURN_LEFT = 1
TURN_RIGHT = -1

# Constants for LiDAR sectors
SCAN_THRESHOLD = 0.5
SCAN_FRONT = 0
SCAN_LEFT = 1
SCAN_BACK = 2
SCAN_RIGHT = 3

# Navigation constants
ROAMING_RADIUS = 2.0
HOME_RADIUS = 0.5
ITEM_PICKUP_DISTANCE = 0.35
ZONE_DEPOSIT_DISTANCE = 0.5

class State(Enum):
    FORWARD = 0
    TURNING = 1
    COLLECTING = 2
    RETURNING = 3
    DEPOSITING = 4

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Robot state
        self.state = State.FORWARD
        self.pose = Pose()
        self.yaw = 0.0
        self.previous_yaw = 0.0
        self.turn_angle = 0.0
        self.turn_direction = TURN_LEFT
        self.scan_triggered = [False] * 4
        
        # Item and zone tracking
        self.items = ItemList()
        self.zones = ZoneList()
        self.holding_item = False
        self.held_item_color = None
        
        # Navigation
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.x_to_target = 0.0
        self.y_to_target = 0.0
        self.distance_to_target = 0.0
        self.angle_to_target = 0.0
        
        # Robot ID parameter
        self.declare_parameter('robot_id', 'robot1')
        self.robot_id = self.get_parameter('robot_id').value

        # Callback groups for service calls
        client_cb_group = MutuallyExclusiveCallbackGroup()
        timer_cb_group = MutuallyExclusiveCallbackGroup()

        # Services
        self.pick_up_service = self.create_client(
            ItemRequest, 
            '/pick_up_item',
            callback_group=client_cb_group
        )
        self.offload_service = self.create_client(
            ItemRequest, 
            '/offload_item',
            callback_group=client_cb_group
        )

        # Subscribers
        self.item_subscriber = self.create_subscription(
            ItemList,
            'items',
            self.item_callback,
            10,
            callback_group=timer_cb_group
        )
        
        self.zone_subscriber = self.create_subscription(
            ZoneList,
            'zone',
            self.zone_callback,
            10,
            callback_group=timer_cb_group
        )
        
        self.odom_subscriber = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10,
            callback_group=timer_cb_group
        )
        
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            QoSPresetProfiles.SENSOR_DATA.value,
            callback_group=timer_cb_group
        )

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(
            Twist, 
            'cmd_vel', 
            10
        )

        # Control loop timer
        self.timer_period = 0.1
        self.timer = self.create_timer(
            self.timer_period, 
            self.control_loop,
            callback_group=timer_cb_group
        )

    def item_callback(self, msg):
        self.items = msg

    def zone_callback(self, msg):
        self.zones = msg

    def odom_callback(self, msg):
        self.pose = msg.pose.pose
        (_, _, self.yaw) = euler_from_quaternion([
            self.pose.orientation.x,
            self.pose.orientation.y,
            self.pose.orientation.z,
            self.pose.orientation.w
        ])

    def scan_callback(self, msg):
        front_ranges = msg.ranges[331:359] + msg.ranges[0:30]
        left_ranges = msg.ranges[31:90]
        back_ranges = msg.ranges[91:270]
        right_ranges = msg.ranges[271:330]

        self.scan_triggered[SCAN_FRONT] = min(front_ranges) < SCAN_THRESHOLD
        self.scan_triggered[SCAN_LEFT] = min(left_ranges) < SCAN_THRESHOLD
        self.scan_triggered[SCAN_BACK] = min(back_ranges) < SCAN_THRESHOLD
        self.scan_triggered[SCAN_RIGHT] = min(right_ranges) < SCAN_THRESHOLD

    def find_suitable_zone(self):
        if not self.zones.data:
            return None
            
        for zone in self.zones.data:
            # If zone matches our item color or is unused (no color assigned)
            if zone.color == self.held_item_color or zone.color == "":
                return zone
        return None

    def navigate_to_target(self, x, y, close_distance):
        try:
            t = self.tf_buffer.lookup_transform(
                'base_link',
                'odom',
                rclpy.time.Time())
            
            self.x_to_target = x - self.pose.position.x
            self.y_to_target = y - self.pose.position.y
            self.distance_to_target = math.sqrt(
                self.x_to_target ** 2 + 
                self.y_to_target ** 2
            )
            self.angle_to_target = math.atan2(
                self.y_to_target,
                self.x_to_target
            )

            if self.distance_to_target < close_distance:
                return True

            msg = Twist()
            msg.angular.z = 0.5 * self.angle_to_target
            msg.linear.x = 0.25 * self.distance_to_target
            self.cmd_vel_publisher.publish(msg)
            return False

        except TransformException as e:
            self.get_logger().error(f'Transform error: {e}')
            return False

    def control_loop(self):
        self.get_logger().info(f"STATE: {self.state}")
        
        match self.state:
            case State.FORWARD:
                # Handle obstacle avoidance
                if self.scan_triggered[SCAN_FRONT]:
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = random.uniform(150, 170)
                    self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                    return

                # Look for items if not holding one
                if not self.holding_item and len(self.items.data) > 0:
                    self.state = State.COLLECTING
                    return

                # Look for zones if holding an item
                if self.holding_item:
                    suitable_zone = self.find_suitable_zone()
                    if suitable_zone:
                        self.state = State.DEPOSITING
                        return

                # Move forward
                msg = Twist()
                msg.linear.x = LINEAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

            case State.TURNING:
                msg = Twist()
                msg.angular.z = self.turn_direction * ANGULAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

                yaw_difference = angles.normalize_angle(self.yaw - self.previous_yaw)
                if math.fabs(yaw_difference) >= math.radians(self.turn_angle):
                    self.state = State.FORWARD

            case State.COLLECTING:
                if len(self.items.data) == 0:
                    self.state = State.FORWARD
                    return

                item = self.items.data[0]
                estimated_distance = 32.4 * float(item.diameter) ** -0.75

                if estimated_distance <= ITEM_PICKUP_DISTANCE:
                    request = ItemRequest.Request()
                    request.robot_id = self.robot_id
                    try:
                        future = self.pick_up_service.call_async(request)
                        rclpy.spin_until_future_complete(self, future)
                        response = future.result()
                        if response.success:
                            self.holding_item = True
                            self.held_item_color = item.colour
                            self.state = State.FORWARD
                        else:
                            self.get_logger().warn(f'Failed to pick up: {response.message}')
                    except Exception as e:
                        self.get_logger().error(f'Service call failed: {e}')
                    return

                msg = Twist()
                msg.linear.x = 0.25 * estimated_distance
                msg.angular.z = item.x / 320.0
                self.cmd_vel_publisher.publish(msg)

            case State.DEPOSITING:
                suitable_zone = self.find_suitable_zone()
                if not suitable_zone:
                    self.state = State.FORWARD
                    return

                if self.navigate_to_target(suitable_zone.x, suitable_zone.y, ZONE_DEPOSIT_DISTANCE):
                    request = ItemRequest.Request()
                    request.robot_id = self.robot_id
                    try:
                        future = self.offload_service.call_async(request)
                        rclpy.spin_until_future_complete(self, future)
                        response = future.result()
                        if response.success:
                            self.holding_item = False
                            self.held_item_color = None
                        else:
                            self.get_logger().warn(f'Failed to offload: {response.message}')
                    except Exception as e:
                        self.get_logger().error(f'Service call failed: {e}')
                    self.state = State.FORWARD

    def destroy_node(self):
        msg = Twist()
        self.cmd_vel_publisher.publish(msg)
        super().destroy_node()


def main(args=None):

    rclpy.init(args = args, signal_handler_options = SignalHandlerOptions.NO)

    node = RobotController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
