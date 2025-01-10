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
from assessment_interfaces.msg import ItemList, ZoneList, Zone
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
    Zone.ZONE_CYAN: "cyan",
    Zone.ZONE_PURPLE: "purple",
    Zone.ZONE_GREEN: "green",
    Zone.ZONE_PINK: "pink"
}

ITEM_PICKUP_DISTANCE = 0.35
ZONE_DEPOSIT_DISTANCE = 1.0  # Increased to ensure we're well inside the zone
ZONE_APPROACH_SPEED = 0.2   # Consistent approach speed

# Add color mappings at the top with other constants
ITEM_TO_ZONE = {
    "red": Zone.ZONE_GREEN,     # Red items go to green zone
    "green": Zone.ZONE_PURPLE,  # Green items go to purple zone
    "blue": Zone.ZONE_CYAN      # Blue items go to cyan zone
}

class State(Enum):
    FORWARD = 0    # Moving forward, looking for items
    TURNING = 1    # Turning to avoid obstacles or find zones
    COLLECTING = 2 # Moving to collect an item
    DEPOSITING = 3 # Moving to deposit in a zone

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Robot state
        self.state = State.FORWARD
        self.pose = Pose()
        self.previous_pose = Pose()
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

        # Wait for services to be available
        while not self.pick_up_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Pick up service not available, waiting...')
        while not self.offload_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Offload service not available, waiting...')

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
        
        self.get_logger().info('Robot controller initialized')

    def item_callback(self, msg):
        self.items = msg

    def zone_callback(self, msg):
        """Process incoming zone information"""
        self.zones = msg
        if len(msg.data) > 0:
            self.get_logger().info(f"Detected {len(msg.data)} zones: " + 
                                 ", ".join([ZONE_COLORS.get(z.zone, "unknown") for z in msg.data]))

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
        """Find a zone matching our held item's color"""
        if not self.zones.data or not self.holding_item:
            return None
        
        # Get the target zone type for our held item
        target_zone = ITEM_TO_ZONE.get(self.held_item_color)
        if not target_zone:
            self.get_logger().warn(f"No target zone defined for {self.held_item_color} items")
            return None

        # Find matching zone
        for zone in self.zones.data:
            if zone.zone == target_zone:
                self.get_logger().info(
                    f"Found matching zone for {self.held_item_color} item: "
                    f"{ZONE_COLORS.get(zone.zone, 'unknown')}, "
                    f"size: {zone.size:.2f}, position: ({zone.x}, {zone.y})"
                )
                return zone
        
        return None

    def control_loop(self):
        # Update marker for visualization
        marker = StringWithPose()
        marker.text = str(self.state)
        marker.pose = self.pose
        self.marker_publisher.publish(marker)

        self.get_logger().info(f"STATE: {self.state}, Holding: {self.held_item_color if self.holding_item else 'None'}, "
                             f"Zones visible: {len(self.zones.data)}")
        
        if self.state == State.FORWARD:
            # First check for items
            if len(self.items.data) > 0:
                self.state = State.COLLECTING
                self.get_logger().info(f"Found {self.items.data[0].colour} item, moving to collect")
                return

            # Handle obstacles like week 5
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
                self.turn_direction = TURN_RIGHT if self.scan_triggered[SCAN_LEFT] else TURN_LEFT
                return

            # Move forward if no obstacles
            msg = Twist()
            msg.linear.x = LINEAR_VELOCITY
            self.cmd_vel_publisher.publish(msg)

        elif self.state == State.TURNING:
            # Just handle turning like week 5
            angle_turned = abs(angles.normalize_angle(self.yaw - self.previous_yaw))
            angle_to_turn = math.radians(self.turn_angle)

            if angle_turned >= angle_to_turn:
                self.state = State.FORWARD
                return

            msg = Twist()
            msg.angular.z = ANGULAR_VELOCITY * self.turn_direction
            self.cmd_vel_publisher.publish(msg)

        elif self.state == State.COLLECTING:
            if len(self.items.data) == 0:
                self.previous_pose = self.pose
                self.state = State.FORWARD
                return
            
            item = self.items.data[0]
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
                        self.items.data = []
                        self.state = State.DEPOSITING
                        self.get_logger().info("Switching to DEPOSITING state")
                    else:
                        self.get_logger().info('Failed to pick up: ' + response.message)
                        self.state = State.FORWARD
                except Exception as e:
                    self.get_logger().error(f'Service call failed: {e}')
                    self.state = State.FORWARD
                return

            # Move towards item using visual servoing
            msg = Twist()
            msg.linear.x = 0.25 * estimated_distance
            msg.angular.z = item.x / 320.0
            self.cmd_vel_publisher.publish(msg)

        elif self.state == State.DEPOSITING:
            # First try to find our target zone
            zone = self.find_suitable_zone()
            if not zone:
                # If no matching zone found, turn in place to search
                self.get_logger().info(f"Searching for zone for {self.held_item_color} item...")
                msg = Twist()
                msg.angular.z = ANGULAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)
                return

            # Once we see our target zone, move towards it
            self.get_logger().info(f"Moving to zone, size: {zone.size:.2f}, x_offset: {zone.x}")
            msg = Twist()
            msg.linear.x = ZONE_APPROACH_SPEED
            msg.angular.z = zone.x / 320.0
            self.cmd_vel_publisher.publish(msg)

            # When close enough, try to deposit
            if zone.size >= ZONE_DEPOSIT_DISTANCE:
                self.get_logger().info("In zone, attempting to deposit...")
                request = ItemRequest.Request()
                request.robot_id = self.robot_id
                try:
                    future = self.offload_service.call_async(request)
                    rclpy.spin_until_future_complete(self, future)
                    response = future.result()
                    if response.success:
                        self.get_logger().info(f'Successfully deposited {self.held_item_color} item')
                        self.holding_item = False
                        self.held_item_color = None
                        self.state = State.FORWARD
                    else:
                        self.get_logger().warn(f'Failed to deposit: {response.message}')
                        # Keep trying if deposit fails
                except Exception as e:
                    self.get_logger().error(f'Service call failed: {e}')
                return

def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    
    node = RobotController()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
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
