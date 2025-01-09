import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSPresetProfiles

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from assessment_interfaces.msg import ItemList, ZoneList
from auro_interfaces.srv import ItemRequest

import random
from enum import Enum

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

class State(Enum):
    SEARCHING = 0  # Looking for items or zones
    COLLECTING = 1 # Moving to collect an item
    DEPOSITING = 2 # Moving to deposit in a zone

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Robot state
        self.state = State.SEARCHING
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
        
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            QoSPresetProfiles.SENSOR_DATA.value,
            callback_group=timer_cb_group
        )

        # Publisher
        self.cmd_vel_publisher = self.create_publisher(
            Twist, 
            'cmd_vel', 
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

    def control_loop(self):
        self.get_logger().info(f"STATE: {self.state}, Holding: {self.held_item_color if self.holding_item else 'No'}")
        
        match self.state:
            case State.SEARCHING:
                # If we see an item and aren't holding one, collect it
                if not self.holding_item and len(self.items.data) > 0:
                    self.state = State.COLLECTING
                    return
                    
                # If holding an item, turn in place until we find a zone
                if self.holding_item:
                    suitable_zone = self.find_suitable_zone()
                    if suitable_zone:
                        self.state = State.DEPOSITING
                        self.get_logger().info(f'Found {ZONE_COLORS.get(suitable_zone.zone, "unknown")} zone, moving to deposit')
                        return
                    else:
                        # Turn in place to search for zones
                        msg = Twist()
                        msg.angular.z = ANGULAR_VELOCITY
                        self.cmd_vel_publisher.publish(msg)
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

            case State.COLLECTING:
                if len(self.items.data) == 0:
                    self.state = State.SEARCHING
                    return
                
                item = self.items.data[0]
                if self.move_to_target(item.x, item.y, float(item.diameter)):
                    # Try to pick up the item
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
                            self.state = State.SEARCHING
                        else:
                            self.get_logger().info('Failed to pick up: ' + response.message)
                    except Exception as e:
                        self.get_logger().error(f'Service call failed: {e}')

            case State.DEPOSITING:
                suitable_zone = self.find_suitable_zone()
                if not suitable_zone:
                    self.state = State.SEARCHING
                    return

                if self.move_to_target(suitable_zone.x, suitable_zone.y, suitable_zone.size):
                    # Try to deposit the item
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
                        else:
                            self.get_logger().warn(f'Failed to deposit: {response.message}')
                    except Exception as e:
                        self.get_logger().error(f'Service call failed: {e}')
                    self.state = State.SEARCHING

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot before shutting down
        msg = Twist()
        node.cmd_vel_publisher.publish(msg)
        node.destroy_node()
        rclpy.shutdown()
