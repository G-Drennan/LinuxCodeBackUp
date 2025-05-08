import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

class Nav2TestNode(Node):
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

    def run(self):
        # Wait for Nav2 to activate
        self.navigator.waitUntilNav2Active()

        # Define a goal pose
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = 2.0
        goal_pose.pose.position.y = 2.0
        goal_pose.pose.orientation.w = 1.0

        # Navigate to the goal
        self.navigator.goToPose(goal_pose)

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
    node = Nav2TestNode()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()