from setuptools import setup
import os
from glob import glob

package_name = 'path_planner_test'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],  # Keep this as the package name
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include the launch directory
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include the config directory
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Path planning test node using Nav2, SLAM Toolbox, and Gazebo.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_planner_test = path_planner_test.path_planner_test:main',  # Update this path
        ],
    },
) 