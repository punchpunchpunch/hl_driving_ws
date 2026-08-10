from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # waypoint_follower
        Node(
            package='auto_car_test_pkg',
            executable='waypoint_follower',
            name='waypoint_follower',
            output='screen'
        ),

        # lane_hls_dynamic
        Node(
            package='auto_car_test_pkg',
            executable='lane_hls_dynamic',
            name='lane_hls_dynamic',
            output='screen'
        ),

        # lidar_obstacle_detector
        Node(
            package='auto_car_test_pkg',
            executable='lidar_obstacle_detector',
            name='lidar_obstacle_detector',
            output='screen'
        ),

        # comm_car
        Node(
            package='auto_car_test_pkg',
            executable='comm_car',
            name='comm_car',
            output='screen'
        ),

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', '/home/yeong/ros2_ws/rviz/autonomous_car.rviz'],
            output='screen'
        ),
    ])