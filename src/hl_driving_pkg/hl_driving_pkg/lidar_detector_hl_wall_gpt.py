import rclpy
from rclpy.node import Node

import numpy as np
import math
import time

from sensor_msgs.msg import LaserScan
from auto_driving_msgs.msg import SteerMsg
from std_msgs.msg import Int32

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from rclpy.qos import QoSProfile
from rclpy.qos import DurabilityPolicy


class LidarObstacleDetector(Node):

    def __init__(self):

        super().__init__('lidar_obstacle_detector')

        # ============================================================
        # Parking Flag
        # ============================================================

        self.t_parking_flag = 8
        self.p_parking_flag = 9
        self.r_parking_flag = 1

        self.flag = 0

        # ============================================================
        # General LiDAR
        # ============================================================

        self.slot_min_points = 5
        self.grid_size = 0.02

        # ============================================================
        # Center ROI
        # ============================================================

        self.center_roi_x_min = 0.1
        self.center_roi_x_max = 4.0
        self.center_roi_y_width = 1.4

        # ============================================================
        # Slow ROI
        # ============================================================

        self.slow_roi_x_min = 0.0
        self.slow_roi_x_max = 3.5
        self.slow_roi_y_width = 1.8

        # ============================================================
        # Side ROI
        # ============================================================

        self.side_roi_x_min = -0.3
        self.side_roi_x_max = 1.5
        self.side_roi_y_width = 2.0

        # ============================================================
        # T Parking Slot ROI
        # ============================================================

        self.t_slot1_x_min = 0.5
        self.t_slot1_x_max = 2.0

        self.t_slot1_y_min = 2.0
        self.t_slot1_y_max = 6.0

        # ============================================================
        # P Parking Slot ROI
        # ============================================================

        self.p_slot1_x_min = 1.0
        self.p_slot1_x_max = 6.0

        self.p_slot1_y_min = -1.0
        self.p_slot1_y_max = 1.0

        # ============================================================
        # Parking Slot Decision
        # ============================================================

        self.t_parking_target_slot = 0
        self.t_parking_decided = False

        self.p_parking_target_slot = 0
        self.p_parking_decided = False

        # ============================================================
        # Center Stop
        # ============================================================

        self.center_condition_start_time = None

        self.center_hold_duration = 6.0

        # ============================================================
        # Existing Steering Messages
        # ============================================================

        self.center_steer_msg = SteerMsg()
        self.center_steer_msg.steer = 0.0
        self.center_steer_msg.is_ok = False

        self.slow_steer_msg = SteerMsg()
        self.slow_steer_msg.steer = 0.0
        self.slow_steer_msg.is_ok = False

        self.side_steer_msg = SteerMsg()
        self.side_steer_msg.steer = 0.0
        self.side_steer_msg.is_ok = False

        # ============================================================
        # Reverse Parking Wall Following
        # ============================================================

        # 목표 벽 거리
        self.target_distance = 1.00

        # Wall ROI
        #
        # +X : 전방
        # -X : 후방
        # +Y : 좌측
        # -Y : 우측
        #
        # 후진 주차 시 오른쪽 벽을 사용
        #
        self.wall_x_min = -1.0
        self.wall_x_max = 1.0

        self.wall_y_min = -2.0
        self.wall_y_max = 0.0

        # Wall distance
        self.wall_min_distance = 0.15
        self.wall_max_distance = 2.0

        # 최소 벽 포인트
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
        # PID 없음
        # ============================================================

        self.wall_angle_weight = 1.0
        self.distance_weight = 0.5

        # ============================================================
        # Parking Steering Message
        # ============================================================

        self.parking_steer_msg = SteerMsg()

        self.parking_steer_msg.steer = 0.0
        self.parking_steer_msg.is_ok = False

        # ============================================================
        # RDDF Select
        # ============================================================

        self.lidar_rddf_select_msg = Int32()
        self.lidar_rddf_select_msg.data = 0

        # ============================================================
        # Subscribers
        # ============================================================

        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.flag_subscription = self.create_subscription(
            Int32,
            '/waypoints_flag',
            self.flag_callback,
            10
        )

        # ============================================================
        # Existing Publishers
        # ============================================================

        self.lidar_center_steer_publisher = self.create_publisher(
            SteerMsg,
            'lidar_center_steer',
            10
        )

        self.lidar_slow_steer_publisher = self.create_publisher(
            SteerMsg,
            'lidar_slow_steer',
            10
        )

        self.lidar_side_steer_publisher = self.create_publisher(
            SteerMsg,
            'lidar_side_steer',
            10
        )

        self.lidar_rddf_select_publisher = self.create_publisher(
            Int32,
            '/lidar_rddf_select',
            10
        )

        # ============================================================
        # Parking Steering Publisher
        # ============================================================

        self.parking_steer_publisher = self.create_publisher(
            SteerMsg,
            '/lidar_parking_steer',
            10
        )

        # ============================================================
        # RViz
        # ============================================================

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.roi_pub = self.create_publisher(
            Marker,
            'roi_marker',
            qos
        )

        self.wall_roi_pub = self.create_publisher(
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
            'LidarObstacleDetector Started'
        )

        self.get_logger().info(
            'T Parking Flag : 8'
        )

        self.get_logger().info(
            'P Parking Flag : 9'
        )

        self.get_logger().info(
            'Reverse Wall Following : ENABLED'
        )

        self.get_logger().info(
            'PID : DISABLED'
        )

        self.get_logger().info(
            '========================================'
        )

    # ================================================================
    # Flag
    # ================================================================

    def flag_callback(self, msg: Int32):

        self.flag = msg.data

    # ================================================================
    # LiDAR Callback
    # ================================================================

    def scan_callback(self, msg: LaserScan):

        # ============================================================
        # Raw Scan
        # ============================================================

        ranges = np.array(
            msg.ranges,
            dtype=np.float64
        )

        angles = (
            msg.angle_min
            + np.arange(len(ranges))
            * msg.angle_increment
        )

        # ============================================================
        # Valid
        # ============================================================

        valid_indices = (
            np.isfinite(ranges)
            & (ranges > msg.range_min)
            & (ranges < msg.range_max)
        )

        r = ranges[valid_indices]
        theta = angles[valid_indices]

        if len(r) == 0:

            self.stop_parking_steering()

            return

        # ============================================================
        # 기존 LidarObstacleDetector 좌표
        #
        # 기존 코드와 동일하게 +pi
        # ============================================================

        parking_theta = theta + np.pi

        x = r * np.cos(parking_theta)
        y = r * np.sin(parking_theta)

        points = np.vstack(
            (x, y)
        ).T

        # ============================================================
        # Downsampling
        # ============================================================

        points = self.grid_downsampling(
            points,
            self.grid_size
        )

        # ============================================================
        # 기존 ROI
        # ============================================================

        self.process_roi(points)

        # ============================================================
        # 기존 Parking Slot
        # ============================================================

        self.process_parking(points)

        # ============================================================
        # Reverse Wall Following
        #
        # 주차 플래그 8일 때만 작동
        #
        # 여기서는 원본 LaserScan 좌표를 사용한다.
        # ============================================================

        if self.flag == self.r_parking_flag:

            self.process_reverse_wall(
                r,
                theta
            )

        else:

            self.stop_parking_steering()

    # ================================================================
    # Grid Downsampling
    # ================================================================

    def grid_downsampling(
        self,
        points,
        grid_size
    ):

        if len(points) == 0:

            return points

        keys = (
            points / grid_size
        ).astype(int)

        _, indices = np.unique(
            keys,
            axis=0,
            return_index=True
        )

        return points[indices]

    # ================================================================
    # Existing ROI
    # ================================================================

    def process_roi(self, points):

        self.center_steer_msg.is_ok = False
        self.slow_steer_msg.is_ok = False
        self.side_steer_msg.is_ok = False

        self.side_steer_msg.steer = 0.0

        if len(points) == 0:

            self.publish_existing_steering()

            return

        # ============================================================
        # Center
        # ============================================================

        center_indices = (
            (points[:, 0] > self.center_roi_x_min)
            & (points[:, 0] < self.center_roi_x_max)
            & (
                points[:, 1]
                > -self.center_roi_y_width / 2
            )
            & (
                points[:, 1]
                < self.center_roi_y_width / 2
            )
        )

        center_points = points[
            center_indices
        ]

        # ============================================================
        # Slow
        # ============================================================

        slow_indices = (
            (points[:, 0] > self.slow_roi_x_min)
            & (points[:, 0] < self.slow_roi_x_max)
            & (
                points[:, 1]
                > -self.slow_roi_y_width / 2
            )
            & (
                points[:, 1]
                < self.slow_roi_y_width / 2
            )
        )

        slow_points = points[
            slow_indices
        ]

        # ============================================================
        # Side
        # ============================================================

        side_indices = (
            (points[:, 0] > self.side_roi_x_min)
            & (points[:, 0] < self.side_roi_x_max)
            & (
                points[:, 1]
                > -self.side_roi_y_width / 2
            )
            & (
                points[:, 1]
                < self.side_roi_y_width / 2
            )
        )

        side_points = points[
            side_indices
        ]

        # ============================================================
        # Center Stop
        # ============================================================

        condition = (
            len(center_points) > 5
        )

        current_time = time.time()

        if condition:

            if self.center_condition_start_time is None:

                self.center_condition_start_time = (
                    current_time
                )

            if (
                current_time
                - self.center_condition_start_time
                < self.center_hold_duration
            ):

                self.center_steer_msg.steer = 0.0
                self.center_steer_msg.is_ok = True

            else:

                self.center_steer_msg.is_ok = False

        else:

            self.center_condition_start_time = None

            self.center_steer_msg.is_ok = False

        # ============================================================
        # Slow
        # ============================================================

        if len(slow_points) > 8:

            self.slow_steer_msg.is_ok = True

        # ============================================================
        # Side
        # ============================================================

        left_pts = side_points[
            side_points[:, 1] > 0.0
        ]

        right_pts = side_points[
            side_points[:, 1] < 0.0
        ]

        if (
            len(left_pts) > 8
            and len(left_pts) > len(right_pts)
        ):

            self.side_steer_msg.steer = 20.0
            self.side_steer_msg.is_ok = True

        elif (
            len(right_pts) > 8
            and len(right_pts) > len(left_pts)
        ):

            self.side_steer_msg.steer = -20.0
            self.side_steer_msg.is_ok = True

        self.publish_existing_steering()

    # ================================================================
    # Existing Steering Publish
    # ================================================================

    def publish_existing_steering(self):

        self.lidar_center_steer_publisher.publish(
            self.center_steer_msg
        )

        self.lidar_slow_steer_publisher.publish(
            self.slow_steer_msg
        )

        self.lidar_side_steer_publisher.publish(
            self.side_steer_msg
        )

    # ================================================================
    # Reverse Wall Processing
    # ================================================================

    def process_reverse_wall(
        self,
        ranges,
        angles
    ):

        # ============================================================
        # Wall ROI
        # ============================================================

        roi_mask = (
            (angles >= math.radians(-180.0))
            & (angles <= math.radians(0.0))
        )

        # 실제 x/y ROI
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        roi_mask = (
            (x >= self.wall_x_min)
            & (x <= self.wall_x_max)
            & (y >= self.wall_y_min)
            & (y <= self.wall_y_max)
        )

        roi_ranges = ranges[
            roi_mask
        ]

        roi_angles = angles[
            roi_mask
        ]

        roi_points = np.column_stack(
            (
                roi_ranges * np.cos(roi_angles),
                roi_ranges * np.sin(roi_angles)
            )
        )

        # ============================================================
        # Point Check
        # ============================================================

        if len(roi_points) < self.min_wall_points:

            self.get_logger().warn(
                f'[PARKING WALL] Not enough points: '
                f'{len(roi_points)}'
            )

            self.stop_parking_steering()

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

        if a is None or b is None:

            self.get_logger().warn(
                f'[PARKING WALL] Beam missing '
                f'a={a}, b={b}'
            )

            self.stop_parking_steering()

            return

        # ============================================================
        # Distance Check
        # ============================================================

        if (
            a < self.wall_min_distance
            or a > self.wall_max_distance
            or b < self.wall_min_distance
            or b > self.wall_max_distance
        ):

            self.get_logger().warn(
                f'[PARKING WALL] Invalid beam '
                f'a={a:.3f}, b={b:.3f}'
            )

            self.stop_parking_steering()

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
            current_distance
            < self.wall_min_distance
            or current_distance
            > self.wall_max_distance
        ):

            self.get_logger().warn(
                f'[PARKING WALL] Invalid distance '
                f'{current_distance:.3f}'
            )

            self.stop_parking_steering()

            return

        # ============================================================
        # Future Distance
        # ============================================================

        future_distance = (
            self.calculate_future_distance(
                b,
                wall_angle
            )
        )

        if (
            future_distance
            < self.wall_min_distance
            or future_distance
            > self.wall_max_distance
        ):

            self.get_logger().warn(
                f'[PARKING WALL] Invalid future '
                f'distance={future_distance:.3f}'
            )

            self.stop_parking_steering()

            return

        # ============================================================
        # Geometric Steering
        # ============================================================

        steer = self.calculate_geometric_steering(
            wall_angle,
            future_distance
        )

        self.parking_steer_msg.steer = steer
        self.parking_steer_msg.is_ok = True

        self.parking_steer_publisher.publish(
            self.parking_steer_msg
        )

        # ============================================================
        # Debug
        # ============================================================

        distance_error = (
            future_distance
            - self.target_distance
        )

        self.get_logger().info(
            f'[PARKING WALL] '
            f'points={len(roi_points):3d} | '
            f'a={a:.3f} | '
            f'b={b:.3f} | '
            f'alpha={math.degrees(wall_angle):+.2f} | '
            f'D={current_distance:.3f} | '
            f'Dfuture={future_distance:.3f} | '
            f'Derror={distance_error:+.3f} | '
            f'steer={steer:+.2f}'
        )

        # ============================================================
        # RViz
        # ============================================================

        self.publish_wall_points(
            roi_points
        )

    # ================================================================
    # Get Beam
    # ================================================================

    def get_roi_beam_range(
        self,
        roi_ranges,
        roi_angles,
        target_angle_deg
    ):

        if len(roi_ranges) == 0:

            return None

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
    # Wall Angle
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
    # Future Distance
    # ================================================================

    def calculate_future_distance(
        self,
        b,
        wall_angle
    ):

        slope = math.tan(
            wall_angle
        )

        return (
            b
            + slope * self.lookahead_distance
        )

    # ================================================================
    # Geometric Steering
    #
    # PID 없음
    # ================================================================

    def calculate_geometric_steering(
        self,
        wall_angle,
        future_distance
    ):

        angle_component = (
            self.wall_angle_weight
            * wall_angle
        )

        distance_error = (
            future_distance
            - self.target_distance
        )

        distance_component = (
            self.distance_weight
            * distance_error
        )

        steering_angle = (
            angle_component
            + distance_component
        )

        steer_deg = math.degrees(
            steering_angle
        )

        steer_deg = np.clip(
            steer_deg,
            -self.max_steer,
            self.max_steer
        )

        return float(steer_deg)

    # ================================================================
    # Stop Parking Steering
    # ================================================================

    def stop_parking_steering(self):

        self.parking_steer_msg.steer = 0.0
        self.parking_steer_msg.is_ok = False

        self.parking_steer_publisher.publish(
            self.parking_steer_msg
        )

    # ================================================================
    # Slot Occupancy
    # ================================================================

    def get_slot_occupancy(
        self,
        points,
        x_min,
        x_max,
        y_min,
        y_max
    ):

        if points.size == 0:

            return False

        indices = (
            (points[:, 0] > x_min)
            & (points[:, 0] < x_max)
            & (points[:, 1] > y_min)
            & (points[:, 1] < y_max)
        )

        slot_points = points[
            indices
        ]

        self.get_logger().info(
            f'[SLOT ROI] '
            f'x=({x_min:.2f},{x_max:.2f}), '
            f'y=({y_min:.2f},{y_max:.2f}), '
            f'points={len(slot_points)}'
        )

        if len(slot_points) > 0:

            self.get_logger().info(
                f'[SLOT POINT] '
                f'min={slot_points.min(axis=0)}, '
                f'max={slot_points.max(axis=0)}'
            )

        if len(slot_points) < self.slot_min_points:

            return False

        return (
            len(slot_points)
            >= self.slot_min_points
        )

    # ================================================================
    # Parking Slot Selection
    # ================================================================

    def process_parking(
        self,
        points
    ):

        if self.flag == self.t_parking_flag:

            if self.t_parking_decided:

                slot_msg = Int32()

                slot_msg.data = (
                    self.t_parking_target_slot
                )

                self.lidar_rddf_select_publisher.publish(
                    slot_msg
                )

                return

            slot1_occupied = (
                self.get_slot_occupancy(
                    points,
                    self.t_slot1_x_min,
                    self.t_slot1_x_max,
                    self.t_slot1_y_min,
                    self.t_slot1_y_max
                )
            )

            if slot1_occupied:

                self.t_parking_target_slot = 2

            else:

                self.t_parking_target_slot = 1

            if (
                self.t_parking_target_slot
                != 0
            ):

                self.t_parking_decided = True

            slot_msg = Int32()

            slot_msg.data = (
                self.t_parking_target_slot
            )

            self.lidar_rddf_select_publisher.publish(
                slot_msg
            )

            self.get_logger().info(
                f'[PARKING] '
                f'target_slot='
                f'{self.t_parking_target_slot}'
            )

        elif self.flag == self.p_parking_flag:

            if self.p_parking_decided:

                slot_msg = Int32()

                slot_msg.data = (
                    self.p_parking_target_slot
                )

                self.lidar_rddf_select_publisher.publish(
                    slot_msg
                )

                return

            slot1_occupied = (
                self.get_slot_occupancy(
                    points,
                    self.p_slot1_x_min,
                    self.p_slot1_x_max,
                    self.p_slot1_y_min,
                    self.p_slot1_y_max
                )
            )

            if slot1_occupied:

                self.p_parking_target_slot = 2

            else:

                self.p_parking_target_slot = 1

            if (
                self.p_parking_target_slot
                != 0
            ):

                self.p_parking_decided = True

            slot_msg = Int32()

            slot_msg.data = (
                self.p_parking_target_slot
            )

            self.lidar_rddf_select_publisher.publish(
                slot_msg
            )

            self.get_logger().info(
                f'[P-PARKING] '
                f'target_slot='
                f'{self.p_parking_target_slot}'
            )

    # ================================================================
    # Wall ROI Marker
    # ================================================================

    def publish_wall_roi(self):

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

        marker.points = [

            Point(
                x=self.wall_x_min,
                y=self.wall_y_min,
                z=0.0
            ),

            Point(
                x=self.wall_x_max,
                y=self.wall_y_min,
                z=0.0
            ),

            Point(
                x=self.wall_x_max,
                y=self.wall_y_max,
                z=0.0
            ),

            Point(
                x=self.wall_x_min,
                y=self.wall_y_max,
                z=0.0
            ),

            Point(
                x=self.wall_x_min,
                y=self.wall_y_min,
                z=0.0
            )
        ]

        self.wall_roi_pub.publish(
            marker
        )

    # ================================================================
    # Wall Points
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

        self.wall_pub.publish(
            marker
        )

    # ================================================================
    # Existing ROI Marker
    # ================================================================

    def publish_roi(self):

        # ============================================================
        # Slow ROI
        # ============================================================

        marker = Marker()

        marker.header.frame_id = 'laser'

        marker.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        marker.ns = 'roi'
        marker.id = 0

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.03

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        x_min = self.slow_roi_x_min
        x_max = self.slow_roi_x_max

        y_min = -self.slow_roi_y_width / 2
        y_max = self.slow_roi_y_width / 2

        marker.points = [

            Point(
                x=x_min,
                y=y_min,
                z=0.0
            ),

            Point(
                x=x_max,
                y=y_min,
                z=0.0
            ),

            Point(
                x=x_max,
                y=y_max,
                z=0.0
            ),

            Point(
                x=x_min,
                y=y_max,
                z=0.0
            ),

            Point(
                x=x_min,
                y=y_min,
                z=0.0
            )
        ]

        self.roi_pub.publish(
            marker
        )

        # ============================================================
        # Wall ROI
        # ============================================================

        self.publish_wall_roi()

    # ================================================================
    # Shutdown
    # ================================================================

    def destroy_node(self):

        self.center_steer_msg.steer = 0.0
        self.center_steer_msg.is_ok = False

        self.slow_steer_msg.steer = 0.0
        self.slow_steer_msg.is_ok = False

        self.side_steer_msg.steer = 0.0
        self.side_steer_msg.is_ok = False

        self.parking_steer_msg.steer = 0.0
        self.parking_steer_msg.is_ok = False

        self.lidar_center_steer_publisher.publish(
            self.center_steer_msg
        )

        self.lidar_slow_steer_publisher.publish(
            self.slow_steer_msg
        )

        self.lidar_side_steer_publisher.publish(
            self.side_steer_msg
        )

        self.parking_steer_publisher.publish(
            self.parking_steer_msg
        )

        time.sleep(0.1)

        super().destroy_node()


# ====================================================================
# Main
# ====================================================================

def main(args=None):

    rclpy.init(args=args)

    node = LidarObstacleDetector()

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