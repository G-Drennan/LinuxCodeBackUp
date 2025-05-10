import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped
from typing import List, Tuple
import time

class PathPlanner:
    def __init__(self, navigator: BasicNavigator):
        self.navigator = navigator

    def compute_path(self, waypoints: List[PoseStamped]) -> Tuple[List[PoseStamped], float]:
        if len(waypoints) < 2:
            return [], float('inf')

        total_path = []
        total_length = 0.0

        for i in range(len(waypoints) - 1):
            self.navigator.clear_costmap()
            path = self.navigator.compute_path(waypoints[i], waypoints[i + 1])
            if not path or len(path.poses) == 0:
                return [], float('inf')
            total_path.extend(path.poses)
            total_length += self._compute_path_length(path.poses)

        return total_path, total_length

    def _compute_path_length(self, poses: List[PoseStamped]) -> float:
        length = 0.0
        for i in range(len(poses) - 1):
            dx = poses[i + 1].pose.position.x - poses[i].pose.position.x
            dy = poses[i + 1].pose.position.y - poses[i].pose.position.y
            length += (dx ** 2 + dy ** 2) ** 0.5
        return length


class DynamicWaypointNavigator(Node):
    def __init__(self):
        super().__init__('dynamic_waypoint_navigator')
        self.navigator = BasicNavigator()
        self.planner = PathPlanner(self.navigator)

    def navigate_through_waypoints(self, waypoints: List[PoseStamped]):
        remaining_waypoints = waypoints.copy()

        while remaining_waypoints:
            current_pose = self.navigator.get_current_pose()

            candidate_paths = []
            for i in range(len(remaining_waypoints)):
                candidate = [current_pose] + remaining_waypoints[i:]
                path, cost = self.planner.compute_path(candidate)
                if path:
                    candidate_paths.append((path, cost, i))

            if not candidate_paths:
                self.get_logger().error("No valid path found to any remaining waypoint.")
                break

            best_path, _, best_index = min(candidate_paths, key=lambda x: x[1])

            self.navigator.follow_path(best_path)
            while not self.navigator.is_task_complete():
                time.sleep(0.5)

            # Remove visited waypoints
            remaining_waypoints = remaining_waypoints[best_index + 1:]

            self.get_logger().info("Reached a waypoint. Recalculating next best path...")

        self.get_logger().info("All waypoints reached or no paths remaining.")
