import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/rogue/codeSpace/repos/LinuxCodeBackUp/code/ros2_ws/install/nav2_test_pkg'
