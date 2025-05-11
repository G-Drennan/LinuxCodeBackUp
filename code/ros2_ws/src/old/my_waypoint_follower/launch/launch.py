from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_file = PathJoinSubstitution([
        FindPackageShare('turtlebot3_gazebo'),
        'worlds',
        'turtlebot3_world.model'
    ])
    nav2_params_file = PathJoinSubstitution([
        FindPackageShare('my_waypoint_follower'),
        'config',
        'nav2_params.yaml' 
    ])

    # Include Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={'world': world_file}.items()
    )

    # Include SLAM launch
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('slam_toolbox'),
                'launch',
                'online_async_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()  # Removed slam_params_file
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': nav2_params_file
        }.items()
    )

    # Waypoint Follower Node
    my_waypoint_follower_node = Node(
        package='my_waypoint_follower',
        executable='my_waypoint_follower',
        name='unique_waypoint_follower', 
        output='screen', 
        parameters=[{'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation time'),
        gazebo,
        slam,
        nav2, 
        my_waypoint_follower_node,
    ]) 