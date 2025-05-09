from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

#ros2 launch nav2_test_pkg nav2_test_launch.py

def generate_launch_description():
    #ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
    #ros2 launch nav2_bringup navigation_launch.py
    #ros2 launch slam_toolbox online_async_launch.py 
    gazebo_sim = ExecuteProcess(
        cmd=[
            "ros2",
            "launch",
            "turtlebot3_gazebo",
            "turtlebot3_world.launch.py",
        ], 
        #output="screen",  
    )

    Nav2 = ExecuteProcess(
        cmd=[
            "ros2",
            "launch",
            "nav2_bringup",
            "navigation_launch.py",
        ], 
        #output="screen",  
    ) 

    Slam = ExecuteProcess(
        cmd=[
            "ros2",
            "launch",
            "slam_toolbox",
            "online_async_launch.py", 
        ], 
        #output="screen",  
    )

    return LaunchDescription([gazebo_sim, Slam, Nav2]) 