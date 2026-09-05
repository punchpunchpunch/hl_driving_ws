#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import math
import time

from sensor_msgs.msg import LaserScan
from auto_driving_msgs.msg import SteerMsg

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from rclpy.qos import QoSProfile
from rclpy.qos import DurabilityPolicy


class ReverseWallFollowing(Node):

    def __init__(self):
        super().__init__('reverse_wall_following')

        # ============================================================
        # Parameters
        # ============================================================

        # 목표 벽 거리
        self.target_distance = 0.80

        # ============================================================
        # Wall ROI
        #
        # 차량 기준
        #
        #             +X 전방
        #                ↑
        #                |
        #       +Y  ← 차량 →  -Y
        #                |
        #                ↓
        #             -X 후방
        #
        # 오른쪽 후방 영역을 검사
        # ============================================================

        self.wall_x_min = -1.5
        self.wall_x_max = 0.3

        self.wall_y_min = -2.0
        self.wall_y_max = -0.0

        # 벽으로 인정할 최소 / 최대 거리
        self.wall_min_distance = 0.15
        self.wall_max_distance = 1.5

        # 최소 검출 포인트 수
        self.min_wall_points = 5

        # ============================================================
        # Controller Parameters
        # ============================================================

        # 거리 오차 Gain
        self.kp = 80.0

        # 조향 최대값
        self.max_steer = 25.0

        # Deadband
        self.deadband = 0.03

        # 벽 방향 보정 Gain
        self.angle_gain = 15.0

        # ============================================================
        # Message
        # ============================================================

        self.steer_msg = SteerMsg()
        self.steer_msg.steer = 0.0
        self.steer_msg.is_ok = False

        # ============================================================
        # Subscriber
        # ============================================================

        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # ============================================================
        # Publisher
        # ============================================================

        self.steer_publisher = self.create_publisher(
            SteerMsg,
            '/lidar_parking_steer',
            10
        )

        # ============================================================
        # RViz Marker
        # ============================================================

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.roi_pub = self.create_publisher(
            Marker,
            '/wall_following_roi',
            qos
        )

        self.wall_pub = self.create_publisher(
            Marker,
            '/detected_wall',
            qos
        )

        self.create_timer(
            0.5,
            self.publish_roi
        )

        self.get_logger().info(
            '========================================'
        )

        self.get_logger().info(
            'Reverse Wall Following Started'
        )

        self.get_logger().info(
            f'Target distance : {self.target_distance:.2f} m'
        )

        self.get_logger().info(
            f'Kp              : {self.kp:.2f}'
        )

        self.get_logger().info(
            '========================================'
        )

    # ================================================================
    # LiDAR Callback
    # ================================================================

    def scan_callback(self, msg: LaserScan):

        ranges = np.array(msg.ranges)

        angles = (
            msg.angle_min
            + np.arange(len(ranges))
            * msg.angle_increment
        )

        # ============================================================
        # Valid Scan
        # ============================================================

        valid = (
            np.isfinite(ranges)
            & (ranges > msg.range_min)
            & (ranges < msg.range_max)
        )

        ranges = ranges[valid]
        angles = angles[valid]

        if len(ranges) == 0:

            self.stop_steering()

            return

        # ============================================================
        # Polar → Cartesian
        # ============================================================

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        points = np.column_stack((x, y))

        # ============================================================
        # Right Rear Wall ROI
        # ============================================================

        wall_indices = (
            (points[:, 0] >= self.wall_x_min)
            & (points[:, 0] <= self.wall_x_max)
            & (points[:, 1] >= self.wall_y_min)
            & (points[:, 1] <= self.wall_y_max)
        )

        wall_points = points[wall_indices]

        # ============================================================
        # Wall Detection
        # ============================================================

        if len(wall_points) < self.min_wall_points:

            self.get_logger().warn(
                f'Wall not detected '
                f'({len(wall_points)} points)'
            )

            self.stop_steering()

            return

        # ============================================================
        # Wall Distance
        #
        # 오른쪽은 y < 0
        #
        # 따라서 벽과의 횡방향 거리는 abs(y)
        # ============================================================

        distances = np.abs(wall_points[:, 1])

        # Median을 사용하여 순간적인 LiDAR 이상값 완화
        wall_distance = np.median(distances)

        # ============================================================
        # Wall Distance Range Check
        # ============================================================

        if (
            wall_distance < self.wall_min_distance
            or wall_distance > self.wall_max_distance
        ):

            self.get_logger().warn(
                f'Wall distance out of range: '
                f'{wall_distance:.3f} m'
            )

            self.stop_steering()

            return

        # ============================================================
        # Wall Angle
        # ============================================================

        wall_angle = self.calculate_wall_angle(
            wall_points
        )

        # ============================================================
        # Steering
        # ============================================================

        steer = self.calculate_steering(
            wall_distance,
            wall_angle
        )

        # ============================================================
        # Publish
        # ============================================================

        self.steer_msg.steer = steer
        self.steer_msg.is_ok = True

        self.steer_publisher.publish(
            self.steer_msg
        )

        # ============================================================
        # Debug
        # ============================================================

        self.get_logger().info(
            f'Wall points={len(wall_points):3d} | '
            f'distance={wall_distance:.3f} m | '
            f'angle={math.degrees(wall_angle):6.2f} deg | '
            f'steer={steer:6.2f}'
        )

        # ============================================================
        # RViz
        # ============================================================

        self.publish_wall_points(
            wall_points
        )

    # ================================================================
    # Calculate Wall Angle
    # ================================================================

    def calculate_wall_angle(self, points):

        if len(points) < 2:

            return 0.0

        x = points[:, 0]
        y = points[:, 1]

        try:

            # y = ax + b
            a, b = np.polyfit(
                x,
                y,
                1
            )

            angle = math.atan(a)

            return angle

        except Exception:

            return 0.0

    # ================================================================
    # Steering Controller
    # ================================================================

    def calculate_steering(
        self,
        wall_distance,
        wall_angle
    ):

        # ============================================================
        # Distance Error
        #
        # error = 실제 거리 - 목표 거리
        #
        # 예:
        #
        # 벽 0.30m
        # error = 0.30 - 0.50 = -0.20
        #
        # 벽 0.80m
        # error = 0.80 - 0.50 = +0.30
        # ============================================================

        distance_error = (
            wall_distance
            - self.target_distance
        )

        # ============================================================
        # Deadband
        # ============================================================

        if abs(distance_error) < self.deadband:

            distance_error = 0.0

        # ============================================================
        # P Controller
        # ============================================================

        steer = (
            self.kp
            * distance_error
        )

        # ============================================================
        # Wall Angle Compensation
        # ============================================================

        steer += (
            self.angle_gain
            * wall_angle
        )

        # ============================================================
        # Steering Saturation
        # ============================================================

        steer = np.clip(
            steer,
            -self.max_steer,
            self.max_steer
        )

        return float(steer)

    # ================================================================
    # Stop Steering
    # ================================================================

    def stop_steering(self):

        self.steer_msg.steer = 0.0
        self.steer_msg.is_ok = False

        self.steer_publisher.publish(
            self.steer_msg
        )

    # ================================================================
    # Publish ROI Marker
    # ================================================================

    def publish_roi(self):

        marker = Marker()

        marker.header.frame_id = 'laser'

        marker.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        marker.ns = 'wall_following'
        marker.id = 0

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.03

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        x1 = self.wall_x_min
        x2 = self.wall_x_max

        y1 = self.wall_y_min
        y2 = self.wall_y_max

        marker.points = [

            Point(
                x=x1,
                y=y1,
                z=0.0
            ),

            Point(
                x=x2,
                y=y1,
                z=0.0
            ),

            Point(
                x=x2,
                y=y2,
                z=0.0
            ),

            Point(
                x=x1,
                y=y2,
                z=0.0
            ),

            Point(
                x=x1,
                y=y1,
                z=0.0
            )
        ]

        self.roi_pub.publish(
            marker
        )

    # ================================================================
    # Publish Detected Wall Points
    # ================================================================

    def publish_wall_points(self, points):

        marker = Marker()

        marker.header.frame_id = 'laser'

        marker.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        marker.ns = 'detected_wall'
        marker.id = 0

        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        marker.scale.x = 0.05
        marker.scale.y = 0.05

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.points = [

            Point(
                x=float(p[0]),
                y=float(p[1]),
                z=0.0
            )

            for p in points
        ]

        self.wall_pub.publish(
            marker
        )

    # ================================================================
    # Shutdown
    # ================================================================

    def destroy_node(self):

        self.steer_msg.steer = 0.0
        self.steer_msg.is_ok = False

        self.steer_publisher.publish(
            self.steer_msg
        )

        time.sleep(0.1)

        super().destroy_node()


# ====================================================================
# Main
# ====================================================================

def main(args=None):

    rclpy.init(args=args)

    node = ReverseWallFollowing()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        node.get_logger().info(
            'KeyboardInterrupt'
        )

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()