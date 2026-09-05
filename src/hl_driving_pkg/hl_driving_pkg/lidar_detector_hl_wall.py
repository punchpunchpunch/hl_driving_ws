import rclpy
from rclpy.node import Node

import numpy as np
import time
import math
import random

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

        self.t_parking_flag = 8     # 직각주차 플래그
        self.p_parking_flag = 9     # 평행주차 플래그
        self.r_parking_flag = 1
        self.flag = 0

        self.slot_min_points = 5    # 막혀 있다고 판단하는 기준이 되는 포인트 수
        self.grid_size = 0.02

        # Center ROI
        self.center_roi_x_min = 0.1
        self.center_roi_x_max = 4.0
        self.center_roi_y_width = 1.4

        # Slow ROI
        self.slow_roi_x_min = 0.0
        self.slow_roi_x_max = 3.5
        self.slow_roi_y_width = 1.8

        # Side ROI
        self.side_roi_x_min = -0.3
        self.side_roi_x_max = 1.5
        self.side_roi_y_width = 2.0

        # T-Parking ROI
        self.t_slot1_x_min = 0.5
        self.t_slot1_x_max = 2.0
        self.t_slot1_y_min = 2.0
        self.t_slot1_y_max = 6.0

        # P-Parking ROI
        self.p_slot1_x_min = 1.0
        self.p_slot1_x_max = 6.0
        self.p_slot1_y_min = -1.0
        self.p_slot1_y_max = 1.0

        self.t_parking_target_slot = 0
        self.t_parking_decided = False

        self.p_parking_target_slot = 0
        self.p_parking_decided = False

        self.center_condition_start_time = None
        self.center_prev_detected = False
        self.center_hold_active = False
        self.center_cooldown = False
        self.center_hold_start_time = 0.0
        self.center_hold_duration = 6.0

        # ============================================================
        # SteerMsg
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

        self.lidar_rddf_select_msg = Int32()
        self.lidar_rddf_select_msg.data = 0

        # ============================================================
        # 직각주차 wall_following SteerMsg
        # ============================================================

        self.parking_steer_msg = SteerMsg()
        self.parking_steer_msg.steer = 0.0
        self.parking_steer_msg.is_ok = False

        # ============================================================
        # RANSAC / Wall parameters
        # ============================================================

        # 목표 오른쪽 벽 거리
        self.wall_target_distance = 0.7

        self.wall_roi_x_min = -1.0
        self.wall_roi_x_max = 1.0

        self.wall_roi_y_min = 0.0
        self.wall_roi_y_max = 1.5

        # 거리 제한
        self.wall_min_range = 0.5
        self.wall_max_range = 1.5

        # RANSAC
        self.ransac_threshold = 0.04
        self.ransac_iterations = 100
        self.wall_min_inliers = 5

        # Control gain
        self.wall_distance_gain = 20.0
        self.wall_yaw_gain = 20.0

        # 최대 steering
        self.wall_max_steer = 10.0

        # steering 방향
        # 실제 차량에서 반대로 움직이면 -1.0
        self.wall_steering_sign = 1.0

        # ============================================================
        # 마지막 wall detection 결과
        # ============================================================

        self.wall_distance = None
        self.wall_yaw = None
        self.wall_inliers = None

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
        # Publishers
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

        # NEW
        self.lidar_parking_steer_publisher = self.create_publisher(
            SteerMsg,
            'lidar_parking_steer',
            10
        )

        self.lidar_rddf_select_publisher = self.create_publisher(
            Int32,
            '/lidar_rddf_select',
            10
        )

        # ============================================================
        # Marker
        # ============================================================

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.roi_pub = self.create_publisher(
            Marker,
            'roi_marker',
            qos
        )

        self.create_timer(0.5, self.publish_roi)

    # ================================================================
    # Flag callback
    # ================================================================

    def flag_callback(self, msg: Int32):

        self.flag = msg.data

    # ================================================================
    # Scan callback
    # ================================================================

    def scan_callback(self, msg):

        ranges = np.array(msg.ranges)   # 각도 별 거리(m)

        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment   # 각도

        # 유효한 거리 데이터만 추출 (0 < r < inf)?
        valid_indices = (ranges > msg.range_min) & (ranges < msg.range_max) # boolean 배열
        r = ranges[valid_indices]
        theta = angles[valid_indices]
        theta = theta + np.pi

        # 극좌표(Polar) -> 직교좌표(XY) 변환
        x = r * np.cos(theta)   # 전방 거리
        y = r * np.sin(theta)   # 좌우 거리

        points = np.vstack((x, y)).T # point[x, y]

        # 다운샘플링 (Voxel/Grid Filter 방식)
        points = self.grid_downsampling(points, self.grid_size)

        # ROI 필터링 및 구역별 감지
        self.process_roi(points)

        # 직각주차 rddf 선택 판단
        self.process_parking(points)

        # ------------------------------------------------------------
        # 직각주차일 경우 오른쪽 벽 검출
        # ------------------------------------------------------------

        if self.flag == self.r_parking_flag:

            self.process_parking_wall(points)

        else:

            self.parking_steer_msg.steer = 0.0
            self.parking_steer_msg.is_ok = False

            self.lidar_parking_steer_publisher.publish(self.parking_steer_msg)

    # ================================================================
    # Grid downsampling
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

        return points[
            indices
        ]

    # ================================================================
    # 기존 process_roi
    # ================================================================

    def process_roi(self, points):

        self.center_steer_msg.is_ok = False
        self.slow_steer_msg.is_ok = False
        self.side_steer_msg.is_ok = False

        self.side_steer_msg.steer = 0.0

        if len(points) == 0:
            return

        # ------------------------------------------------------------
        # Center ROI
        # ------------------------------------------------------------

        center_indices = (
            (points[:, 0] > self.center_roi_x_min)
            &
            (points[:, 0] < self.center_roi_x_max)
            &
            (points[:, 1] >
             -self.center_roi_y_width / 2)
            &
            (points[:, 1] <
             self.center_roi_y_width / 2)
        )

        center_points = points[
            center_indices
        ]

        # ------------------------------------------------------------
        # Slow ROI
        # ------------------------------------------------------------

        slow_indices = (
            (points[:, 0] > self.slow_roi_x_min)
            &
            (points[:, 0] < self.slow_roi_x_max)
            &
            (points[:, 1] >
             -self.slow_roi_y_width / 2)
            &
            (points[:, 1] <
             self.slow_roi_y_width / 2)
        )

        slow_points = points[
            slow_indices
        ]

        # ------------------------------------------------------------
        # Side ROI
        # ------------------------------------------------------------

        side_indices = (
            (points[:, 0] > self.side_roi_x_min)
            &
            (points[:, 0] < self.side_roi_x_max)
            &
            (points[:, 1] >
             -self.side_roi_y_width / 2)
            &
            (points[:, 1] <
             self.side_roi_y_width / 2)
        )

        side_points = points[
            side_indices
        ]

        if (
            len(center_points) > 0
            or
            len(slow_points) > 0
            or
            len(side_points) > 0
        ):

            # --------------------------------------------------------
            # Left / Right
            # --------------------------------------------------------

            left_pts = side_points[
                side_points[:, 1] > 0.0
            ]

            right_pts = side_points[
                side_points[:, 1] < 0.0
            ]

            current_time = time.time()

            condition = False

            if len(center_points) > 5:

                condition = True

            if condition:

                if (
                    self.center_condition_start_time
                    is None
                ):

                    self.center_condition_start_time = (
                        current_time
                    )

                if (
                    current_time
                    -
                    self.center_condition_start_time
                    <
                    6.0
                ):

                    self.center_steer_msg.steer = 0.0
                    self.center_steer_msg.is_ok = True

                else:

                    self.center_steer_msg.is_ok = False

            else:

                self.center_condition_start_time = None
                self.center_steer_msg.is_ok = False

            if len(slow_points) > 8:

                self.slow_steer_msg.is_ok = True

            if (
                len(left_pts) > 8
                and
                len(left_pts) > len(right_pts)
            ):

                self.side_steer_msg.steer = 20.0
                self.side_steer_msg.is_ok = True

            elif (
                len(right_pts) > 8
                and
                len(right_pts) > len(left_pts)
            ):

                self.side_steer_msg.steer = -20.0
                self.side_steer_msg.is_ok = True

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
    # NEW
    # Parking wall processing
    # ================================================================

    def process_parking_wall(
        self,
        points
    ):

        if len(points) == 0:

            self.publish_parking_stop()
            return

        # ------------------------------------------------------------
        # 1. Right wall ROI
        # ------------------------------------------------------------

        wall_indices = (
            (points[:, 0] >
             self.wall_roi_x_min)
            &
            (points[:, 0] <
             self.wall_roi_x_max)
            &
            (points[:, 1] >
             self.wall_roi_y_min)
            &
            (points[:, 1] <
             self.wall_roi_y_max)
        )

        wall_points = points[
            wall_indices
        ]

        if len(wall_points) < self.wall_min_inliers:

            self.get_logger().warn(
                '[PARKING WALL] '
                'Not enough points'
            )

            self.publish_parking_stop()
            return

        # ------------------------------------------------------------
        # 2. RANSAC
        # ------------------------------------------------------------

        result = self.ransac_line(
            wall_points
        )

        if result is None:

            self.get_logger().warn(
                '[PARKING WALL] '
                'RANSAC failed'
            )

            self.publish_parking_stop()
            return

        _, inliers = result

        # ------------------------------------------------------------
        # 3. Least-square refinement
        # ------------------------------------------------------------

        line = self.least_square_line(
            inliers
        )

        if line is None:

            self.publish_parking_stop()
            return

        a, b, c = line

        # ------------------------------------------------------------
        # 4. Wall distance
        # ------------------------------------------------------------

        distance = abs(c) / math.sqrt(
            a * a
            +
            b * b
        )

        # ------------------------------------------------------------
        # 5. Wall yaw
        # ------------------------------------------------------------

        yaw = math.atan2(
            a,
            -b
        )

        # ------------------------------------------------------------
        # Normalize -90° ~ +90°
        # ------------------------------------------------------------

        while yaw > math.pi / 2:

            yaw -= math.pi

        while yaw < -math.pi / 2:

            yaw += math.pi

        # ------------------------------------------------------------
        # 저장
        # ------------------------------------------------------------

        self.wall_distance = distance
        self.wall_yaw = yaw
        self.wall_inliers = inliers

        # ------------------------------------------------------------
        # 6. Distance error
        # ------------------------------------------------------------

        distance_error = (
            self.wall_target_distance
            -
            distance
        )

        # ------------------------------------------------------------
        # 7. Steering
        # ------------------------------------------------------------

        steering = (
            self.wall_distance_gain
            *
            distance_error
            +
            self.wall_yaw_gain
            *
            yaw
        )

        # rad → degree
        steering_deg = math.degrees(
            steering
        )

        steering_deg *= (
            self.wall_steering_sign
        )

        # ------------------------------------------------------------
        # 8. Steering limit
        # ------------------------------------------------------------

        steering_deg = max(
            -self.wall_max_steer,
            min(
                self.wall_max_steer,
                steering_deg
            )
        )

        # ------------------------------------------------------------
        # Debug
        # ------------------------------------------------------------

        self.get_logger().info(
            f'[PARKING WALL] '
            f'distance={distance:.3f} m | '
            f'target={self.wall_target_distance:.3f} m | '
            f'd_error={distance_error:.3f} | '
            f'yaw={math.degrees(yaw):.2f} deg | '
            f'steer={steering_deg:.2f} deg | '
            f'inliers={len(inliers)}'
        )

        # ------------------------------------------------------------
        # 9. Publish
        # ------------------------------------------------------------

        self.parking_steer_msg.steer = float(
            steering_deg
        )

        self.parking_steer_msg.is_ok = True

        self.lidar_parking_steer_publisher.publish(
            self.parking_steer_msg
        )

    # ================================================================
    # RANSAC
    # ================================================================

    def ransac_line(
        self,
        points
    ):

        if len(points) < 2:
            return None

        best_line = None
        best_inliers = np.empty(
            (0, 2)
        )

        n_points = len(points)

        for _ in range(
            self.ransac_iterations
        ):

            # --------------------------------------------------------
            # random 2 points
            # --------------------------------------------------------

            idx1, idx2 = random.sample(
                range(n_points),
                2
            )

            x1, y1 = points[idx1]
            x2, y2 = points[idx2]

            # --------------------------------------------------------
            # 동일점
            # --------------------------------------------------------

            if (
                abs(x2 - x1) < 1e-6
                and
                abs(y2 - y1) < 1e-6
            ):
                continue

            # --------------------------------------------------------
            # ax + by + c = 0
            # --------------------------------------------------------

            a = y1 - y2
            b = x2 - x1

            c = (
                x1 * y2
                -
                x2 * y1
            )

            norm = math.sqrt(
                a * a
                +
                b * b
            )

            if norm < 1e-6:
                continue

            # --------------------------------------------------------
            # 모든 점과 직선의 거리
            # --------------------------------------------------------

            distances = np.abs(
                a * points[:, 0]
                +
                b * points[:, 1]
                +
                c
            ) / norm

            # --------------------------------------------------------
            # inlier
            # --------------------------------------------------------

            mask = (
                distances
                <
                self.ransac_threshold
            )

            inliers = points[
                mask
            ]

            # --------------------------------------------------------
            # best model
            # --------------------------------------------------------

            if (
                len(inliers)
                >
                len(best_inliers)
            ):

                best_inliers = inliers

                best_line = (
                    a,
                    b,
                    c
                )

        if (
            best_line is None
            or
            len(best_inliers)
            <
            self.wall_min_inliers
        ):

            return None

        return (
            best_line,
            best_inliers
        )

    # ================================================================
    # Least Squares / PCA
    # ================================================================

    def least_square_line(
        self,
        points
    ):

        if len(points) < 2:
            return None

        # ------------------------------------------------------------
        # centroid
        # ------------------------------------------------------------

        centroid = np.mean(
            points,
            axis=0
        )

        centered = (
            points
            -
            centroid
        )

        # ------------------------------------------------------------
        # covariance
        # ------------------------------------------------------------

        covariance = np.cov(
            centered.T
        )

        # ------------------------------------------------------------
        # eigen decomposition
        # ------------------------------------------------------------

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        # 가장 큰 eigenvalue의 방향
        direction = eigenvectors[
            :,
            np.argmax(eigenvalues)
        ]

        dx = direction[0]
        dy = direction[1]

        # ------------------------------------------------------------
        # line normal
        # ------------------------------------------------------------

        a = -dy
        b = dx

        # centroid를 지나는 직선
        c = -(
            a * centroid[0]
            +
            b * centroid[1]
        )

        # normalize
        norm = math.sqrt(
            a * a
            +
            b * b
        )

        a /= norm
        b /= norm
        c /= norm

        return (
            a,
            b,
            c
        )

    # ================================================================
    # Parking stop
    # ================================================================

    def publish_parking_stop(self):

        self.parking_steer_msg.steer = 0.0

        self.parking_steer_msg.is_ok = False

        self.lidar_parking_steer_publisher.publish(
            self.parking_steer_msg
        )

    # ================================================================
    # Slot occupancy
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
            &
            (points[:, 0] < x_max)
            &
            (points[:, 1] > y_min)
            &
            (points[:, 1] < y_max)
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

        return (
            len(slot_points)
            >=
            self.slot_min_points
        )

    # ================================================================
    # Parking decision
    # ================================================================

    def process_parking(
        self,
        points
    ):

        if self.flag == self.t_parking_flag:

            self.get_logger().info(
                '[PARKING] FLAG 8 ACTIVE'
            )

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
                !=
                0
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
                f'{self.t_parking_target_slot}, '
                f'RDDF={slot_msg.data}'
            )

        elif self.flag == self.p_parking_flag:

            self.get_logger().info(
                '[PARKING] FLAG 9 ACTIVE'
            )

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
                !=
                0
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
                f'[PARKING] '
                f'target_slot='
                f'{self.p_parking_target_slot}, '
                f'RDDF={slot_msg.data}'
            )

    # ================================================================
    # ROI Marker
    # ================================================================

    def publish_roi(self):

        marker = Marker()

        marker.header.frame_id = "laser"

        marker.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        marker.ns = "roi"
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

        y_min = (
            -self.slow_roi_y_width / 2
        )

        y_max = (
            self.slow_roi_y_width / 2
        )

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

        # ------------------------------------------------------------
        # Parking wall ROI
        # ------------------------------------------------------------

        wall_marker = Marker()

        wall_marker.header.frame_id = "laser"

        wall_marker.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        wall_marker.ns = "parking_wall"
        wall_marker.id = 10

        wall_marker.type = Marker.LINE_STRIP
        wall_marker.action = Marker.ADD

        wall_marker.scale.x = 0.03

        wall_marker.color.r = 1.0
        wall_marker.color.g = 1.0
        wall_marker.color.b = 0.0
        wall_marker.color.a = 1.0

        wall_marker.points = [

            Point(
                x=self.wall_roi_x_min,
                y=self.wall_roi_y_min,
                z=0.0
            ),

            Point(
                x=self.wall_roi_x_max,
                y=self.wall_roi_y_min,
                z=0.0
            ),

            Point(
                x=self.wall_roi_x_max,
                y=self.wall_roi_y_max,
                z=0.0
            ),

            Point(
                x=self.wall_roi_x_min,
                y=self.wall_roi_y_max,
                z=0.0
            ),

            Point(
                x=self.wall_roi_x_min,
                y=self.wall_roi_y_min,
                z=0.0
            )
        ]

        self.roi_pub.publish(
            wall_marker
        )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = LidarObstacleDetector()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        node.get_logger().info(
            'KeyboardInterrupt'
        )

    finally:

        node.center_steer_msg.steer = 0.0
        node.center_steer_msg.is_ok = False

        node.slow_steer_msg.steer = 0.0
        node.slow_steer_msg.is_ok = False

        node.side_steer_msg.steer = 0.0
        node.side_steer_msg.is_ok = False

        node.parking_steer_msg.steer = 0.0
        node.parking_steer_msg.is_ok = False

        node.lidar_center_steer_publisher.publish(
            node.center_steer_msg
        )

        node.lidar_slow_steer_publisher.publish(
            node.slow_steer_msg
        )

        node.lidar_side_steer_publisher.publish(
            node.side_steer_msg
        )

        node.lidar_parking_steer_publisher.publish(
            node.parking_steer_msg
        )

        time.sleep(0.1)

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()