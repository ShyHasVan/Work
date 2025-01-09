import sys
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import ExternalShutdownException
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSPresetProfiles

from geometry_msgs.msg import Twist, Pose, PoseStamped
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

# Navigation constants
ITEM_PICKUP_DISTANCE = 0.35
ZONE_DEPOSIT_DISTANCE = 0.5

# Add these constants after the navigation constants
ZONE_COLORS = {
    1: "cyan",    # ZONE_CYAN
    2: "purple",  # ZONE_PURPLE
    3: "green",   # ZONE_GREEN
    4: "pink"     # ZONE_PINK
}

class State(Enum):
    FORWARD = 0
    TURNING = 1
    COLLECTING = 2
    DEPOSITING = 3

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Initial pose parameters
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('yaw', 0.0)
        
        self.initial_x = self.get_parameter('x').value
        self.initial_y = self.get_parameter('y').value
        self.initial_yaw = self.get_parameter('yaw').value
        
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
        self.target_zone = None  # Store the current target zone
        
        # Navigation
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Robot ID parameter
        self.declare_parameter('robot_id', 'robot1')
        self.robot_id = self.get_parameter('robot_id').value

        # Callback groups
        client_cb_group = MutuallyExclusiveCallbackGroup()
        timer_cb_group = MutuallyExclusiveCallbackGroup()

        # Services (using absolute paths due to namespacing)
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

        # Subscribers (using relative paths due to namespacing)
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

        # Control loop timer
        self.timer = self.create_timer(
            0.1, 
            self.control_loop,
            callback_group=timer_cb_group
        )

        # Add zone tracking
        self.known_zones = {}  # Dictionary to store zone positions {zone_id: (x, y)}

    def item_callback(self, msg):
        self.items = msg

    def zone_callback(self, msg):
        self.zones = msg
        # Update our knowledge of zone positions
        for zone in msg.data:
            zone_color = ZONE_COLORS.get(zone.zone, "")
            if zone.size > 0.5:  # If we're close enough to the zone
                self.known_zones[zone.zone] = (zone.x, zone.y)
                
        # Update target zone if we're holding an item and not already depositing
        if self.holding_item and self.state != State.DEPOSITING:
            suitable_zone = self.find_suitable_zone()
            if suitable_zone:
                self.target_zone = suitable_zone
                self.state = State.DEPOSITING
                self.get_logger().info(f'Found suitable zone of color {ZONE_COLORS.get(suitable_zone.zone, "unknown")}')

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

    def navigate_to_target(self, x, y, close_distance):
        """Navigate using camera-relative coordinates (x, y)
        x: pixels from center of image (positive is left)
        y: pixels from center of image (positive is up)
        """
        try:
            # If we can't see the zone anymore, stop
            if not self.zones.data:
                msg = Twist()
                self.cmd_vel_publisher.publish(msg)
                return False

            # First check if there's an obstacle in front
            if self.scan_triggered[SCAN_FRONT]:
                msg = Twist()
                msg.angular.z = TURN_LEFT * ANGULAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)
                return False

            msg = Twist()
            
            # x is negative when target is to the right
            # Scale down the angular velocity for smoother motion
            msg.angular.z = -0.003 * x

            # y is negative when target is down/forward
            # Only move forward if roughly aligned
            if abs(x) < 100:  # Within ~100 pixels of center
                # Scale down the linear velocity for smoother motion
                # Negative y means move forward
                msg.linear.x = -0.002 * y
                msg.linear.x = min(0.2, max(msg.linear.x, 0))  # Cap between 0 and 0.2
            else:
                msg.linear.x = 0.0

            self.cmd_vel_publisher.publish(msg)

            # Consider we've reached the target if we're close enough (based on size)
            for zone in self.zones.data:
                if zone.size > 0.2:  # Adjust this threshold as needed
                    return True
            
            return False

        except Exception as e:
            self.get_logger().error(f'Navigation error: {e}')
            return False

    def control_loop(self):
        self.get_logger().info(f"STATE: {self.state}, Holding item: {self.held_item_color if self.holding_item else 'No'}")
        
        match self.state:
            case State.FORWARD:
                # First priority: Look for items if not holding one
                if not self.holding_item and len(self.items.data) > 0:
                    self.state = State.COLLECTING
                    return

                # Second priority: Look for zones if holding an item
                if self.holding_item:
                    suitable_zone = self.find_suitable_zone()
                    if suitable_zone:
                        self.target_zone = suitable_zone
                        self.state = State.DEPOSITING
                        self.get_logger().info(f'Moving to deposit in zone of color {ZONE_COLORS.get(suitable_zone.zone, "unknown")}')
                        return

                # Third priority: Handle obstacle avoidance
                if self.scan_triggered[SCAN_FRONT]:
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = random.uniform(150, 170)
                    self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                    return

                # Move forward if no other priorities
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

                self.get_logger().info(f'Estimated distance to item: {estimated_distance}')

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
                            # Check for suitable zone immediately
                            suitable_zone = self.find_suitable_zone()
                            if suitable_zone:
                                self.target_zone = suitable_zone
                                self.state = State.DEPOSITING
                                self.get_logger().info(f'Found suitable zone of color {ZONE_COLORS.get(suitable_zone.zone, "unknown")}')
                            else:
                                self.state = State.FORWARD
                            self.items.data = []
                        else:
                            self.get_logger().info('Unable to pick up item: ' + response.message)
                    except Exception as e:
                        self.get_logger().error(f'Service call failed: {e}')
                    return

                msg = Twist()
                msg.linear.x = 0.25 * estimated_distance
                msg.angular.z = item.x / 320.0
                self.cmd_vel_publisher.publish(msg)

            case State.DEPOSITING:
                # If we lost sight of all zones, keep the target_zone if we had one
                if not self.zones.data and not self.target_zone:
                    self.state = State.FORWARD
                    return

                # Use target_zone if we have one, otherwise find a new suitable zone
                zone_to_navigate = self.target_zone if self.target_zone else self.find_suitable_zone()
                if not zone_to_navigate:
                    self.state = State.FORWARD
                    return

                # Log navigation attempt
                self.get_logger().info(f'Navigating to {ZONE_COLORS.get(zone_to_navigate.zone, "unknown")} zone at ({zone_to_navigate.x}, {zone_to_navigate.y})')
                
                if self.navigate_to_target(zone_to_navigate.x, zone_to_navigate.y, ZONE_DEPOSIT_DISTANCE):
                    request = ItemRequest.Request()
                    request.robot_id = self.robot_id
                    try:
                        future = self.offload_service.call_async(request)
                        rclpy.spin_until_future_complete(self, future)
                        response = future.result()
                        if response.success:
                            self.get_logger().info(f'Successfully deposited {self.held_item_color} item in {ZONE_COLORS.get(zone_to_navigate.zone, "unknown")} zone')
                            self.holding_item = False
                            self.held_item_color = None
                            self.target_zone = None
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
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    
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
