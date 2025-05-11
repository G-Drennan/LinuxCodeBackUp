import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav2_simple_commander.robot_navigator import BasicNavigator
from rclpy.duration import Duration  


class MyWayPointFollower(Node):
    def __init__(self):
        super().__init__('unique_waypoint_follower')  # Initialize the parent Node class
        self.navigator = BasicNavigator()  # Initialize the BasicNavigator

        self.subscription1 = self.create_subscription( 
            Path, 
            'my_waypoints_set',  
            self.listener_callback_waypoint_follower,
            10) 
        self.subscription1 

    def get_current_pose(self):
        """Get the robot's current pose using the navigator."""
        return self.navigator.get_current_pose()
    
    def move_to_goal(self, goal_pose):
        """Move the robot to a specified goal pose."""
        self.navigator.waitUntilNav2Active()  # Ensure Nav2 is active

        # Set the goal pose
        self.navigator.goToPose(goal_pose)

        # Wait for the robot to reach the goal
        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback:
                self.get_logger().info(f"Distance remaining: {feedback.distance_remaining:.2f} meters")

        # Check the result
        result = self.navigator.getResult()
        if result == BasicNavigator.Result.SUCCEEDED:
            self.get_logger().info("Goal succeeded!")
        elif result == BasicNavigator.Result.CANCELED:
            self.get_logger().warn("Goal was canceled!")
        elif result == BasicNavigator.Result.FAILED:
            self.get_logger().error("Goal failed!")

    def listener_callback_waypoint_follower(self, msg):
        """Callback for processing waypoints."""
        # Example usage of getting the current pose
        current_pose = self.get_current_pose()
        if current_pose:
            self.get_logger().info(f"Current Pose: {current_pose.pose.position.x}, {current_pose.pose.position.y}")
        else:
            self.get_logger().error("Failed to get the current pose.")

        # Example: Move to the first waypoint in the Path message
        if msg.poses:
            goal_pose = msg.poses[0]  # Use the first pose in the Path as the goal
            self.move_to_goal(goal_pose)



def main(args=None):
    rclpy.init(args=args)
    node = MyWayPointFollower()
    rclpy.spin(node) 
    node.destroy_node()  
    rclpy.shutdown()

if __name__ == '__main__':
    main()