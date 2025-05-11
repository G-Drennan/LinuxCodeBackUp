from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 1) Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    params_file_arg = LaunchConfiguration('params_file')

    declare_args = [
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use simulation (Gazebo) clock'
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('path_planner_test'),
                'config',
                'nav2_params.yaml'
            ]),
            description='Full path to the ROS2 parameters file for Nav2'
        ),
    ]

    # 2) Launch Gazebo with TurtleBot3 world
    gazebo_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('turtlebot3_gazebo'),
                'worlds',
                'turtlebot3_world.world'
            ]),
            'verbose': 'true'  # Enables more Gazebo logs in the terminal
        }.items()
    )

    # 3) Launch SLAM Toolbox
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('slam_toolbox'),
                'launch',
                'online_async_launch.py'
            ])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # 4) Launch Nav2 bringup
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'navigation_launch.py'
            ])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': params_file_arg
        }.items()
    )

    # 5) Launch your custom path planner node 
    path_planner_node = Node(
        package='path_planner_test',
        executable='path_planner_test',
        name='path_planner_test_node',
        output='screen',
        parameters=[params_file_arg, {'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        *declare_args,
        LogInfo(msg=['Using Nav2 params file: ', params_file_arg]),
        gazebo_world,
        slam, 
        nav2,
        path_planner_node,
    ])
