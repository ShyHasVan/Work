import sys
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import ExternalShutdownException
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSPresetProfiles
from enum import Enum

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from assessment_interfaces.msg import ItemList, ZoneList
from auro_interfaces.srv import ItemRequest
from auro_interfaces.msg import StringWithPose

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

# Zone colors mapping
ZONE_COLORS = {
    1: "cyan",    # ZONE_CYAN
    2: "purple",  # ZONE_PURPLE
    3: "green",   # ZONE_GREEN
    4: "pink"     # ZONE_PINK
}

ITEM_PICKUP_DISTANCE = 0.35
ZONE_DEPOSIT_DISTANCE = 0.2  # Size threshold for being close enough to deposit

class State(Enum):
    SEARCHING = 0  # Looking for items or zones
    COLLECTING = 1 # Moving to collect an item
    DEPOSITING = 2 # Moving to deposit in a zone
    ROTATING = 3   # Rotating to find zones

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Robot state
        self.state = State.SEARCHING
        self.pose = Pose()
        self.yaw = 0.0
        self.previous_yaw = 0.0
        self.scan_triggered = [False] * 4
        
        # Item and zone tracking
        self.items = ItemList()
        self.zones = ZoneList()
        self.holding_item = False
        self.held_item_color = None
        
        # Robot ID parameter
        self.declare_parameter('robot_id', 'robot1')
        self.robot_id = self.get_parameter('robot_id').value

        # Callback groups
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
            'zones',
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
        
        self.marker_publisher = self.create_publisher(
            StringWithPose, 
            'marker_input', 
            10
        )

        # Control loop timer
        self.timer = self.create_timer(
            0.1, 
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
            zone_color = ZONE_COLORS.get(zone.zone, "")
            # If zone matches our item color or is unused (no color assigned)
            if zone_color == self.held_item_color or zone_color == "":
                return zone
        return None

    def move_to_target(self, x, y, size):
        """Visual servoing to target using camera coordinates"""
        msg = Twist()
        
        # If obstacle in front, turn left
        if self.scan_triggered[SCAN_FRONT]:
            msg.angular.z = ANGULAR_VELOCITY
            self.cmd_vel_publisher.publish(msg)
            return False
            
        # x is negative when target is to the right
        msg.angular.z = -0.003 * x

        # Only move forward if roughly aligned with target
        if abs(x) < 100:  # Within ~100 pixels of center
            # y is negative when target is down/forward
            msg.linear.x = min(0.15, max(-0.002 * y, 0))  # Cap between 0 and 0.15
        else:
            msg.linear.x = 0.0

        self.cmd_vel_publisher.publish(msg)
        
        # Return True if we're close enough (based on size)
        return size > 0.2

    def rotate_in_place(self):
        """Rotate the robot in place at a constant speed"""
        msg = Twist()
        msg.angular.z = ANGULAR_VELOCITY
        self.cmd_vel_publisher.publish(msg)

    def control_loop(self):
        # Update marker for visualization
        marker = StringWithPose()
        marker.text = str(self.state)
        marker.pose = self.pose
        self.marker_publisher.publish(marker)

        self.get_logger().info(f"STATE: {self.state}, Holding: {self.held_item_color if self.holding_item else 'No'}")
        
        match self.state:
            case State.SEARCHING:
                # If we see an item and aren't holding one, collect it
                if not self.holding_item and len(self.items.data) > 0:
                    self.state = State.COLLECTING
                    return
                    
                # If holding an item, switch to rotating to find a zone
                if self.holding_item:
                    self.state = State.ROTATING
                    return

                # If not holding an item and no item visible, explore
                if self.scan_triggered[SCAN_FRONT]:
                    msg = Twist()
                    msg.angular.z = ANGULAR_VELOCITY
                    self.cmd_vel_publisher.publish(msg)
                else:
                    msg = Twist()
                    msg.linear.x = LINEAR_VELOCITY
                    self.cmd_vel_publisher.publish(msg)

            case State.ROTATING:
                # Keep rotating until we find a suitable zone
                suitable_zone = self.find_suitable_zone()
                if suitable_zone:
                    self.state = State.DEPOSITING
                    self.get_logger().info(f'Found {ZONE_COLORS.get(suitable_zone.zone, "unknown")} zone, moving to deposit')
                    return
                
                # Just rotate in place
                msg = Twist()
                msg.angular.z = ANGULAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

            case State.COLLECTING:
                if len(self.items.data) == 0:
                    self.state = State.SEARCHING
                    return
                
                item = self.items.data[0]
                # Obtained by curve fitting from experimental runs
                estimated_distance = 32.4 * float(item.diameter) ** -0.75

                self.get_logger().info(f'Estimated distance {estimated_distance}')

                if estimated_distance <= ITEM_PICKUP_DISTANCE:
                    request = ItemRequest.Request()
                    request.robot_id = self.robot_id
                    try:
                        future = self.pick_up_service.call_async(request)
                        rclpy.spin_until_future_complete(self, future)
                        response = future.result()
                        if response.success:
                            self.get_logger().info(f'Picked up {item.colour} item')
                            self.holding_item = True
                            self.held_item_color = item.colour
                            self.state = State.ROTATING
                        else:
                            self.get_logger().info('Failed to pick up: ' + response.message)
                    except Exception as e:
                        self.get_logger().error(f'Service call failed: {e}')
                    return

                # Move towards item using visual servoing
                msg = Twist()
                msg.linear.x = 0.25 * estimated_distance
                msg.angular.z = item.x / 320.0
                self.cmd_vel_publisher.publish(msg)

            case State.DEPOSITING:
                suitable_zone = self.find_suitable_zone()
                if not suitable_zone:
                    self.state = State.ROTATING
                    return

                # Navigate to zone exactly like we do for items
                msg = Twist()
                msg.linear.x = 0.25 * (1.0 - suitable_zone.size)  # Slow down as we get closer
                msg.angular.z = suitable_zone.x / 320.0  # Center the zone in view
                self.cmd_vel_publisher.publish(msg)

                # Try to deposit when zone is large enough in view
                if suitable_zone.size >= ZONE_DEPOSIT_DISTANCE:
                    request = ItemRequest.Request()
                    request.robot_id = self.robot_id
                    try:
                        future = self.offload_service.call_async(request)
                        rclpy.spin_until_future_complete(self, future)
                        response = future.result()
                        if response.success:
                            self.get_logger().info(f'Deposited {self.held_item_color} item in {ZONE_COLORS.get(suitable_zone.zone, "unknown")} zone')
                            self.holding_item = False
                            self.held_item_color = None
                            self.state = State.SEARCHING
                        else:
                            self.get_logger().warn(f'Failed to deposit: {response.message}')
                            self.state = State.ROTATING
                    except Exception as e:
                        self.get_logger().error(f'Service call failed: {e}')
                        self.state = State.ROTATING

def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        # Stop the robot before shutting down
        msg = Twist()
        node.cmd_vel_publisher.publish(msg)
        node.destroy_node()
        rclpy.try_shutdown()
