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
        self.target_distance = 1.00

        # ============================================================
        # Wall ROI
        # ============================================================

        self.wall_x_min = -1.0
        self.wall_x_max = 1.0

        self.wall_y_min = -2.0
        self.wall_y_max = -0.0

        # ============================================================
        # Wall distance
        # ============================================================

        self.wall_min_distance = 0.15
        self.wall_max_distance = 2.0

        # ============================================================
        # ROI Wall Point
        # ============================================================

        self.min_wall_points = 5

        # ============================================================
        # F1TENTH Beam
        # ============================================================

        self.theta_deg = 30.0

        self.b_angle_deg = -90.0

        self.a_angle_deg = (
            self.b_angle_deg
            + self.theta_deg
        )

        self.max_beam_angle_error_deg = 5.0

        # ============================================================
        # Lookahead
        # ============================================================

        self.lookahead_distance = 0.50

        # ============================================================
        # Steering
        # ============================================================

        self.max_steer = 25.0

        # ============================================================
        # Geometric Controller
        #
        # PID를 사용하지 않는다.
        #
        # 벽 방향 + 목표 거리 관계를 이용해서
        # 기하학적으로 조향각을 계산한다.
        # ============================================================

        # 벽과 평행하게 만들기 위한 비율
        self.wall_angle_weight = 1.0

        # 벽 거리 오차를 조향에 반영하는 비율
        self.distance_weight = 0.5

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

        qos.durability = (
            DurabilityPolicy.TRANSIENT_LOCAL
        )

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

        # ============================================================
        # Log
        # ============================================================

        self.get_logger().info(
            '========================================'
        )

        self.get_logger().info(
            'Reverse Geometric Wall Following Started'
        )

        self.get_logger().info(
            f'Target distance : '
            f'{self.target_distance:.2f} m'
        )

        self.get_logger().info(
            f'ROI X           : '
            f'{self.wall_x_min:.2f} ~ '
            f'{self.wall_x_max:.2f}'
        )

        self.get_logger().info(
            f'ROI Y           : '
            f'{self.wall_y_min:.2f} ~ '
            f'{self.wall_y_max:.2f}'
        )

        self.get_logger().info(
            f'Beam b angle    : '
            f'{self.b_angle_deg:.1f} deg'
        )

        self.get_logger().info(
            f'Beam a angle    : '
            f'{self.a_angle_deg:.1f} deg'
        )

        self.get_logger().info(
            f'Lookahead       : '
            f'{self.lookahead_distance:.2f} m'
        )

        self.get_logger().info(
            'PID              : DISABLED'
        )

        self.get_logger().info(
            '========================================'
        )

    # ================================================================
    # LiDAR Callback
    # ================================================================

    def scan_callback(self, msg: LaserScan):

        # ============================================================
        # Raw scan
        # ============================================================

        raw_ranges = np.array(
            msg.ranges,
            dtype=np.float64
        )

        raw_angles = (
            msg.angle_min
            + np.arange(len(raw_ranges))
            * msg.angle_increment
        )

        # ============================================================
        # Valid Scan
        # ============================================================

        valid = (
            np.isfinite(raw_ranges)
            & (raw_ranges > msg.range_min)
            & (raw_ranges < msg.range_max)
        )

        ranges = raw_ranges[valid]
        angles = raw_angles[valid]

        if len(ranges) == 0:

            self.stop_steering()

            return

        # ============================================================
        # Polar → Cartesian
        # ============================================================

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        points = np.column_stack(
            (x, y)
        )

        # ============================================================
        # ROI
        # ============================================================

        roi_mask = (
            (x >= self.wall_x_min)
            & (x <= self.wall_x_max)
            & (y >= self.wall_y_min)
            & (y <= self.wall_y_max)
        )

        roi_points = points[roi_mask]

        roi_ranges = ranges[roi_mask]
        roi_angles = angles[roi_mask]

        # ============================================================
        # ROI Point Check
        # ============================================================

        if len(roi_points) < self.min_wall_points:

            self.get_logger().warn(
                f'ROI wall points insufficient: '
                f'{len(roi_points)}'
            )

            self.stop_steering()

            return

        # ============================================================
        # Beam a / b
        # ============================================================

        a = self.get_roi_beam_range(
            roi_ranges,
            roi_angles,
            self.a_angle_deg
        )

        b = self.get_roi_beam_range(
            roi_ranges,
            roi_angles,
            self.b_angle_deg
        )

        # ============================================================
        # Beam validity
        # ============================================================

        if a is None or b is None:

            self.get_logger().warn(
                f'ROI beam not detected | '
                f'a={a} | b={b}'
            )

            self.stop_steering()

            return

        # ============================================================
        # Beam distance check
        # ============================================================

        if (
            a < self.wall_min_distance
            or a > self.wall_max_distance
            or b < self.wall_min_distance
            or b > self.wall_max_distance
        ):

            self.get_logger().warn(
                f'ROI beam distance invalid | '
                f'a={a:.3f} | '
                f'b={b:.3f}'
            )

            self.stop_steering()

            return

        # ============================================================
        # Wall Angle
        # ============================================================

        wall_angle = self.calculate_wall_angle(
            a,
            b
        )

        # ============================================================
        # Current Distance
        # ============================================================

        current_distance = (
            b * math.cos(wall_angle)
        )

        if (
            current_distance < self.wall_min_distance
            or current_distance > self.wall_max_distance
        ):

            self.get_logger().warn(
                f'Current wall distance invalid: '
                f'{current_distance:.3f}'
            )

            self.stop_steering()

            return

        # ============================================================
        # Reverse Lookahead Distance
        # ============================================================

        future_distance = (
            self.calculate_future_distance(
                b,
                wall_angle
            )
        )

        if (
            future_distance < self.wall_min_distance
            or future_distance > self.wall_max_distance
        ):

            self.get_logger().warn(
                f'Future wall distance invalid: '
                f'{future_distance:.3f}'
            )

            self.stop_steering()

            return

        # ============================================================
        # GEOMETRIC STEERING
        #
        # PID 없음
        #
        # 벽 방향 오차
        # +
        # 미래 벽 거리 오차
        #
        # 를 이용해 조향각 계산
        # ============================================================

        steer = self.calculate_geometric_steering(
            wall_angle,
            future_distance
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

        distance_error = (
            future_distance
            - self.target_distance
        )

        self.get_logger().info(
            f'ROI={len(roi_points):3d} | '
            f'a={a:.3f} | '
            f'b={b:.3f} | '
            f'alpha={math.degrees(wall_angle):6.2f} deg | '
            f'D={current_distance:.3f} | '
            f'Dfuture={future_distance:.3f} | '
            f'Derror={distance_error:+.3f} | '
            f'steer={steer:+6.2f}'
        )

        # ============================================================
        # RViz
        # ============================================================

        self.publish_wall_points(
            roi_points
        )

    # ================================================================
    # Get Beam From ROI
    # ================================================================

    def get_roi_beam_range(
        self,
        roi_ranges,
        roi_angles,
        target_angle_deg
    ):

        target_angle = math.radians(
            target_angle_deg
        )

        angle_error = np.arctan2(
            np.sin(
                roi_angles - target_angle
            ),
            np.cos(
                roi_angles - target_angle
            )
        )

        index = np.argmin(
            np.abs(angle_error)
        )

        angle_diff = abs(
            angle_error[index]
        )

        max_angle_error = math.radians(
            self.max_beam_angle_error_deg
        )

        if angle_diff > max_angle_error:

            return None

        value = roi_ranges[index]

        if not np.isfinite(value):

            return None

        return float(value)

    # ================================================================
    # Calculate Wall Angle
    # ================================================================

    def calculate_wall_angle(
        self,
        a,
        b
    ):

        theta = math.radians(
            self.theta_deg
        )

        denominator = (
            a * math.sin(theta)
        )

        if abs(denominator) < 1e-6:

            return 0.0

        numerator = (
            a * math.cos(theta)
            - b
        )

        alpha = math.atan2(
            numerator,
            denominator
        )

        return alpha

    # ================================================================
    # Calculate Future Distance
    # ================================================================

    def calculate_future_distance(
        self,
        b,
        wall_angle
    ):

        slope = math.tan(
            wall_angle
        )

        future_distance = (
            b
            + slope * self.lookahead_distance
        )

        return future_distance

    # ================================================================
    # Geometric Steering
    #
    # PID 완전 제거
    #
    # 벽의 방향과 목표 거리만 이용해서
    # 현재 필요한 조향각을 직접 계산한다.
    # ================================================================

    def calculate_geometric_steering(
        self,
        wall_angle,
        future_distance
    ):

        # ------------------------------------------------------------
        # 1. 벽 방향을 따라가도록 하는 조향 성분
        # ------------------------------------------------------------

        angle_component = (
            self.wall_angle_weight
            * wall_angle
        )

        # ------------------------------------------------------------
        # 2. 미래 벽 거리 오차
        # ------------------------------------------------------------

        distance_error = (
            future_distance
            - self.target_distance
        )

        # ------------------------------------------------------------
        # 3. 거리 오차 성분
        # ------------------------------------------------------------

        distance_component = (
            self.distance_weight
            * distance_error
        )

        # ------------------------------------------------------------
        # 4. 두 성분을 합산
        # ------------------------------------------------------------

        steering_angle = (
            angle_component
            + distance_component
        )

        # ------------------------------------------------------------
        # 5. rad → deg
        # ------------------------------------------------------------

        steer_deg = math.degrees(
            steering_angle
        )

        # ------------------------------------------------------------
        # 6. Saturation
        # ------------------------------------------------------------

        steer_deg = np.clip(
            steer_deg,
            -self.max_steer,
            self.max_steer
        )

        return float(steer_deg)

    # ================================================================
    # Stop
    # ================================================================

    def stop_steering(self):

        self.steer_msg.steer = 0.0

        self.steer_msg.is_ok = False

        self.steer_publisher.publish(
            self.steer_msg
        )

    # ================================================================
    # ROI Marker
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

        self.roi_pub.publish(marker)

    # ================================================================
    # Detected Wall Points
    # ================================================================

    def publish_wall_points(
        self,
        points
    ):

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

        self.wall_pub.publish(marker)

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