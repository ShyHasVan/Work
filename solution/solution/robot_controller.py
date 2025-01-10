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
        """Find a visible zone"""
        if not self.zones.data:
            return None
        
        # Log all visible zones for debugging
        for zone in self.zones.data:
            self.get_logger().info(f"Zone: {ZONE_COLORS.get(zone.zone, 'unknown')}, "
                                 f"size: {zone.size:.2f}, x: {zone.x}, y: {zone.y}")
        
        # Return the largest visible zone (closest to robot)
        largest_zone = max(self.zones.data, key=lambda z: z.size) if self.zones.data else None
        
        if largest_zone:
            self.get_logger().info(
                f"Selected zone: {ZONE_COLORS.get(largest_zone.zone, 'unknown')}, "
                f"size: {largest_zone.size:.2f}, "
                f"position: ({largest_zone.x}, {largest_zone.y})"
            )
            
        return largest_zone

    def control_loop(self):
        # Update marker for visualization
        marker = StringWithPose()
        marker.text = str(self.state)
        marker.pose = self.pose
        self.marker_publisher.publish(marker)

        self.get_logger().info(f"STATE: {self.state}, Holding item: {self.holding_item}, "
                             f"Item color: {self.held_item_color}, Zones visible: {len(self.zones.data)}")
        
        # Stop moving if transitioning states
        if self.state != State.FORWARD and self.state != State.COLLECTING:
            msg = Twist()
            self.cmd_vel_publisher.publish(msg)

        match self.state:
            case State.FORWARD:
                # If holding an item, force transition to TURNING to find a zone
                if self.holding_item:
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = 45  # Turn in increments to scan for zones
                    self.turn_direction = TURN_LEFT
                    self.get_logger().info("Holding item, turning to find a zone")
                    return

                # If we see an item and aren't holding one, go to COLLECTING state
                if not self.holding_item and len(self.items.data) > 0:
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

                # Move forward if no obstacles and not holding an item
                msg = Twist()
                msg.linear.x = LINEAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

            case State.TURNING:
                # Log the current state and what we're doing
                if self.holding_item:
                    self.get_logger().info(f"TURNING: Looking for zone while holding {self.held_item_color} item")
                else:
                    self.get_logger().info("TURNING: Avoiding obstacle")

                # If holding an item, prioritize finding a zone
                if self.holding_item:
                    zone = self.find_suitable_zone()
                    if zone:
                        self.state = State.DEPOSITING
                        self.get_logger().info(f"Found {ZONE_COLORS.get(zone.zone, 'unknown')} zone while turning")
                        return
                    
                    # Keep turning until we find a zone
                    msg = Twist()
                    msg.angular.z = ANGULAR_VELOCITY * self.turn_direction
                    self.cmd_vel_publisher.publish(msg)
                    self.get_logger().info(f"Still turning to find zone, current yaw: {math.degrees(self.yaw):.1f} degrees")
                    return

                # Normal turning behavior for obstacle avoidance
                angle_turned = abs(angles.normalize_angle(self.yaw - self.previous_yaw))
                angle_to_turn = math.radians(self.turn_angle)

                if angle_turned >= angle_to_turn:
                    self.state = State.FORWARD
                    return

                msg = Twist()
                msg.angular.z = ANGULAR_VELOCITY * self.turn_direction
                self.cmd_vel_publisher.publish(msg)

            case State.COLLECTING:
                if len(self.items.data) == 0:
                    self.get_logger().info("Lost sight of item, returning to FORWARD state")
                    self.state = State.FORWARD
                    return
                
                item = self.items.data[0]
                # Use week 5's proven collection logic
                estimated_distance = 32.4 * float(item.diameter) ** -0.75

                if estimated_distance <= ITEM_PICKUP_DISTANCE:
                    self.get_logger().info("Attempting to pick up item...")
                    request = ItemRequest.Request()
                    request.robot_id = self.robot_id
                    try:
                        future = self.pick_up_service.call_async(request)
                        rclpy.spin_until_future_complete(self, future)
                        if future.done():
                            response = future.result()
                            if response.success:
                                self.get_logger().info(f'Successfully picked up {item.colour} item')
                                self.holding_item = True
                                self.held_item_color = item.colour
                                msg = Twist()
                                self.cmd_vel_publisher.publish(msg)
                                # Immediately start searching for a zone
                                self.previous_yaw = self.yaw
                                self.state = State.TURNING
                                self.turn_angle = 45
                                self.turn_direction = TURN_LEFT
                                self.get_logger().info("Item picked up, switching to TURNING to find a zone")
                            else:
                                self.get_logger().warn(f'Failed to pick up: {response.message}')
                                self.state = State.FORWARD
                        else:
                            self.get_logger().error("Pickup request timed out")
                            self.state = State.FORWARD
                    except Exception as e:
                        self.get_logger().error(f'Service call failed: {str(e)}')
                        self.state = State.FORWARD
                    return

                # Move towards item using visual servoing
                msg = Twist()
                msg.linear.x = 0.25 * estimated_distance
                msg.angular.z = item.x / 320.0
                self.cmd_vel_publisher.publish(msg)
                self.get_logger().info(f'Moving to item: distance={estimated_distance:.2f}, x_offset={item.x}')

            case State.DEPOSITING:
                zone = self.find_suitable_zone()
                if not zone:
                    self.state = State.TURNING
                    self.get_logger().warn("Lost sight of zone, turning to find it again")
                    return

                if self.scan_triggered[SCAN_FRONT]:
                    self.get_logger().warn("Obstacle detected while approaching zone, adjusting...")
                    msg = Twist()
                    msg.angular.z = ANGULAR_VELOCITY * TURN_LEFT
                    self.cmd_vel_publisher.publish(msg)
                    return

                msg = Twist()
                if zone.size >= ZONE_DEPOSIT_DISTANCE:
                    self.get_logger().info(f'Fully inside {ZONE_COLORS.get(zone.zone, "unknown")} zone! Depositing item...')
                    request = ItemRequest.Request()
                    request.robot_id = self.robot_id
                    try:
                        future = self.offload_service.call_async(request)
                        rclpy.spin_until_future_complete(self, future)
                        if future.done():
                            response = future.result()
                            if response.success:
                                self.get_logger().info(f'Successfully deposited {self.held_item_color} item')
                                self.holding_item = False
                                self.held_item_color = None
                                self.state = State.FORWARD
                            else:
                                self.get_logger().warn(f'Failed to deposit: {response.message}')
                                self.state = State.FORWARD
                        else:
                            self.get_logger().error("Deposit request timed out")
                            self.state = State.FORWARD
                    except Exception as e:
                        self.get_logger().error(f'Service call failed: {str(e)}')
                        self.state = State.FORWARD
                    return
                else:
                    msg.linear.x = ZONE_APPROACH_SPEED
                    turn_factor = zone.x / 320.0
                    msg.angular.z = turn_factor * ANGULAR_VELOCITY
                    self.cmd_vel_publisher.publish(msg)
                    self.get_logger().info(
                        f'Moving to zone: size={zone.size:.2f}, '
                        f'target={ZONE_DEPOSIT_DISTANCE:.2f}, '
                        f'x_offset={zone.x}, '
                        f'turning={msg.angular.z:.2f}'
                    )

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
