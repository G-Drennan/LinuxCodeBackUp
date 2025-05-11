import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import math
from python_tsp.heuristics import solve_tsp_local_search  
from nav_msgs.msg import Path 

class PathPlannerTestNode(Node):
    def __init__(self):
        super().__init__('nav2_test_node')
        self.navigator = BasicNavigator()

        # Set an initial pose
        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.pose.position.x = 0.0
        initial_pose.pose.position.y = 0.0
        initial_pose.pose.orientation.w = 1.0
        self.navigator.setInitialPose(initial_pose)

        self.get_logger().info('Nav2 Test Node Initialized')

    def compute_path_length(self, path: Path):
        """Compute the total Euclidean length of a nav_msgs/Path."""
        if not path.poses or len(path.poses) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(path.poses)):
            p1 = path.poses[i - 1].pose.position
            p2 = path.poses[i].pose.position
            total += math.hypot(p2.x - p1.x, p2.y - p1.y)
        return total

    def find_optimal_path(self, waypoints):
        n = len(waypoints)
        cost_matrix = [[0.0] * n for _ in range(n)]

        # Compute the cost matrix (distance matrix)
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Use computePath to get the path between two waypoints
                    path = self.navigator.getPath(waypoints[i], waypoints[j])
                    cost_matrix[i][j] = self.compute_path_length(path) 

        # Solve TSP
        best_order, _ = solve_tsp_local_search  (cost_matrix,
                                                    perturbation_scheme="two_opt_and_swap",
                                                    max_neighbours=20,
                                                    max_iterations=5000,
                                                    seed=42) 

        # Re-order waypoints based on the optimal path
        opt_waypoints = [waypoints[i] for i in best_order]
        
        opt_path = self.navigator.getPathThroughPoses(opt_waypoints)
        return opt_path  
    
    def run(self):
        # Wait for Nav2 to activate
        services = self.get_service_names_and_types()
        amcl_present = any('/amcl/get_state' in name for name, _ in services)  
        if amcl_present:
            self.get_logger().info("AMCL detected; waiting for Nav2 to become active...")
        else:
            self.get_logger().info("AMCL not found; assuming SLAM is being used.")

        self.navigator.lifecycleStartup()  

        # Define a goal pose
        goal_pose1 = PoseStamped()
        goal_pose1.header.frame_id = 'map'
        goal_pose1.pose.position.x = 2.0
        goal_pose1.pose.position.y = 2.0
        goal_pose1.pose.orientation.w = 1.0

        goal_pose2 = PoseStamped()
        goal_pose2.header.frame_id = 'map'
        goal_pose2.pose.position.x = 3.0
        goal_pose2.pose.position.y = 1.0
        goal_pose2.pose.orientation.w = 0.0

        goal_pose3 = PoseStamped()
        goal_pose3.header.frame_id = 'map'
        goal_pose3.pose.position.x = -1.0
        goal_pose3.pose.position.y = 2.0
        goal_pose3.pose.orientation.w = 1.0

        # Navigate to the goal
        waypoints = [goal_pose3, goal_pose1, goal_pose2]
        opt_path = self.find_optimal_path(waypoints)
        self.navigator.followPath(opt_path)   

        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback:
                self.get_logger().info(f"Distance remaining: {feedback.distance_remaining:.2f} meters")

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("Navigation succeeded!")
        elif result == TaskResult.CANCELED:
            self.get_logger().info("Navigation was canceled.")
        elif result == TaskResult.FAILED:
            self.get_logger().info("Navigation failed.")
        

def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerTestNode() 
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()