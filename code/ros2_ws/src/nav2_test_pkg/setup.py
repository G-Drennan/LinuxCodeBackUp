from setuptools import setup
import os
from glob import glob

package_name = 'nav2_test_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),  # Add this line

    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'geometry_msgs',
        'nav2_simple_commander',
        'python-tsp',  # for TSP solver
    ],
    zip_safe=True,
    maintainer='rogue',
    maintainer_email='roguenobody247@gmail.com',
    description='Runs Nav2 and SLAM in Gazebo, and navigates to multiple goals optimally.',
    license='MIT',  # replace with actual license
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nav2_test_node = nav2_test_pkg.nav2_test_node:main',
        ],
    },
)
