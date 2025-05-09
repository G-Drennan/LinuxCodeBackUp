from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'nav2_test_pkg'

setup(
    name=package_name, 
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],  
    install_requires=[
        'setuptools', 
        "rclpy",
        "std_msgs",
        "geometry_msgs",
        "nav_msgs",
        "numpy",
        "sensor_msgs",
        "os",
        "glob",
        "launch",
        "nav2_simple_commander"
        ],
    zip_safe=True,
    maintainer='rogue',
    maintainer_email='roguenobody247@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nav2_test_node = nav2_test_pkg.nav2_test_node:main',

        ],
    },
)
