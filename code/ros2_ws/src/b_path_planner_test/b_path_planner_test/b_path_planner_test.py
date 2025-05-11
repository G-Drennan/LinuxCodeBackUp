import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
import tf_transformations
import numpy as np
from itertools import permutations


class WaypointOptimizer(Node):
    def __init__(self):
        super().__init__('waypoint_optimizer')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.path_sub = self.create_subscription(
            Path,
            '/input_waypoints',
            self.path_callback,
            10
        )

        self.nav_to_pose_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        self.current_pose = None
        self.get_logger().info('Waypoint Optimizer Node has been started.')

    def path_callback(self, msg: Path):
        self.get_logger().info(f'Received {len(msg.poses)} waypoints.')

        self.current_pose = self.get_current_pose()
        if not self.current_pose:
            self.get_logger().error('Could not get current pose.')
            return

        optimized_path = self.optimize_path(self.current_pose, msg.poses)
        self.execute_path(optimized_path)

    def get_current_pose(self):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('map', 'base_link', now, timeout=rclpy.duration.Duration(seconds=1))
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = trans.transform.translation.x
            pose.pose.position.y = trans.transform.translation.y
            pose.pose.position.z = trans.transform.translation.z
            pose.pose.orientation = trans.transform.rotation
            return pose
        except Exception as e:
            self.get_logger().error(f'Error getting current pose: {e}')
            return None

    def optimize_path(self, start_pose, waypoints):
        def distance(p1, p2):
            dx = p1.pose.position.x - p2.pose.position.x
            dy = p1.pose.position.y - p2.pose.position.y
            return np.hypot(dx, dy)

        min_path = None
        min_cost = float('inf')

        for perm in permutations(waypoints):
            cost = distance(start_pose, perm[0])
            cost += sum(distance(perm[i], perm[i+1]) for i in range(len(perm)-1))
            if cost < min_cost:
                min_cost = cost
                min_path = perm

        self.get_logger().info(f'Optimized path with cost {min_cost:.2f}')
        return list(min_path)

    def execute_path(self, waypoints):
        for idx, pose in enumerate(waypoints):
            self.get_logger().info(f'Sending goal {idx+1}/{len(waypoints)}')
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = pose

            if not self.nav_to_pose_client.wait_for_server(timeout_sec=2.0):
                self.get_logger().error('NavigateToPose action server not available.')
                return

            future = self.nav_to_pose_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()

            if not goal_handle.accepted:
                self.get_logger().error(f'Goal {idx+1} was rejected.')
                continue

            self.get_logger().info(f'Goal {idx+1} accepted. Waiting for result...')
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result().result
            self.get_logger().info(f'Goal {idx+1} result: {result}')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointOptimizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
