from setuptools import setup
import os
from glob import glob 

package_name = 'my_waypoint_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name], 
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include the launch directory
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include the config directory
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'geometry_msgs',
        'gazebo_ros',
        'nav2_simple_commander',
        'nav_msgs',
        'std_msgs',
        'turtlebot3_msgs',
        'slam_toolbox',  
        'nav2_bringup', 
        'turtlebot3_gazebo', 
    ], 
    zip_safe=True,
    maintainer='rogue',
    maintainer_email='roguenobody247@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_waypoint_follower = my_waypoint_follower.my_waypoint_follower:main',  
        ],
    },
)
