import rclpy
from rclpy.node import Node

import numpy as np
import time

from sensor_msgs.msg import LaserScan
from auto_driving_msgs.msg import SteerMsg
from std_msgs.msg import Int32

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from rclpy.qos import QoSProfile
from rclpy.qos import DurabilityPolicy

class LidarObstacleDetector(Node):
    def __init__(self):
        super().__init__('lidar_obstacle_detector')

        self.t_parking_flag = 8 # 직각주차 플래그
        self.p_parking_flag = 9 # 평행주차 플래그

        self.center_min_points = 5
        self.slow_min_points = 8
        self.side_min_points = 8
        self.slot_min_points = 5 # 장애물이 있다고 판단하는 기준이 되는 최소 포인트 수

        self.grid_size = 0.02

        self.jump_threshold = 0.2   # m
        self.cluster_min_points = 5
        self.cluster_max_points = 30
        self.cluster_min_range = 0.1
        self.cluster_max_range = 15.0

        # Center ROI
        self.center_roi_x_min, self.center_roi_x_max = 0.1, 3.0
        self.center_roi_y_width = 2.0  # 좌우 전체

        # Slow ROI
        self.slow_roi_x_min, self.slow_roi_x_max = 0.0, 4.0
        self.slow_roi_y_width = 3.0  # 좌우 전체

        # Side ROI
        self.side_roi_x_min, self.side_roi_x_max = -0.3, 1.5
        self.side_roi_y_width = 2.0  # 좌우 전체

        # T-Parking ROI
        self.t_slot1_x_min, self.t_slot1_x_max = 0.0, 6.0
        self.t_slot1_y_min, self.t_slot1_y_max = 0.0, 10.0

        # P-Parking ROI
        self.p_slot1_x_min, self.p_slot1_x_max = 10.0, 15.0
        self.p_slot1_y_min, self.p_slot1_y_max = 0.0, 2.0

        # Back ROI
        self.back_roi_x_min = -3.0
        self.back_roi_x_max = -0.8

        self.back_left_y_min = 0.3
        self.back_left_y_max = 1.3

        self.back_right_y_min = -1.3
        self.back_right_y_max = -0.0

        # 후진 라이다 최소 포인트 수
        self.back_min_points = 1

        # 후진 라이다 최대 회피 조향각
        self.back_max_steer = 20.0


        self.t_parking_target_slot = 0   # 0: 미결정, 1: Slot 1, 2: Slot 2
        self.t_parking_decided = False

        self.p_parking_target_slot = 0
        self.p_parking_decided = False

        self.center_condition_start_time = None
        self.center_prev_detected = False   # 이전 상태
        self.center_hold_active = False     # n초 유지 상태
        self.center_cooldown = False
        self.center_hold_start_time = 0.0   # 시작 시간
        self.center_hold_duration = 6.0     # 유지 시간 (초)

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

        self.flag = 0

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
        
        self.lidar_center_steer_publisher = self.create_publisher(
            SteerMsg,
            '/lidar_center_steer',
            10
        )

        self.lidar_slow_steer_publisher = self.create_publisher(
            SteerMsg,
            '/lidar_slow_steer',
            10
        )

        self.lidar_side_steer_publisher = self.create_publisher(
            SteerMsg,
            '/lidar_side_steer',
            10
        )

        self.lidar_rddf_select_publisher = self.create_publisher(
            Int32,
            '/lidar_rddf_select',
            10
        )

        self.cluster_pub = self.create_publisher(
            MarkerArray,
            '/jump_clusters',
            10
        )

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.roi_pub = self.create_publisher(
            Marker, 'roi_marker', qos
        )
        
        self.create_timer(0.5, self.publish_roi)

    def flag_callback(self, msg: Int32):
        self.flag = msg.data

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges) # 각도 별 거리(m)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment # 각도 
        
        # 유효한 거리 데이터만 추출 (0 < r < inf)?
        valid_indices = (ranges > msg.range_min) & (ranges < msg.range_max) # boolean 배열
        r = ranges[valid_indices]
        theta = angles[valid_indices]
        theta = theta + np.pi

        # 극좌표(Polar) -> 직교좌표(XY) 변환
        x = r * np.cos(theta) # 전방 거리
        y = r * np.sin(theta) # 좌우 거리

        points = np.vstack((x, y)).T # point[x, y]

        # 다운샘플링 (Voxel/Grid Filter 방식)
        points = self.grid_downsampling(points, self.grid_size)

        # ROI 필터링 및 구역별 감지
        self.process_roi(points)

        # 주차 rddf 선택 판단
        self.process_parking(points)

    # 다운샘플링
    def grid_downsampling(self, points, grid_size):
        if len(points) == 0: return points
        # 좌표를 격자 크기로 나누고 정수화하여 인덱스 부여
        keys = (points / grid_size).astype(int)
        # 중복된 격자 키 제거 (격자당 포인트 1개 유지)
        _, indices = np.unique(keys, axis=0, return_index=True)
        return points[indices]

    def process_roi(self, points):
        self.center_steer_msg.is_ok = False
        self.slow_steer_msg.is_ok = False
        self.side_steer_msg.is_ok = False
        self.side_steer_msg.steer = 0.0

        # ==========================================
        # FLAG 1 : 후진 주차 충돌 방지
        # ==========================================
        if self.flag == 1:
            back_steer, back_ok = self.process_back_obstacle(points)

            self.get_logger().info(
                f"[BACK RESULT] "
                f"steer={back_steer:.2f}, "
                f"ok={back_ok}"
            )

            self.side_steer_msg.steer = back_steer
            self.side_steer_msg.is_ok = back_ok

            self.lidar_side_steer_publisher.publish(
                self.side_steer_msg
            )

            # FLAG 1에서는 일반 Side ROI 로직을 사용하지 않음
            return
        
        # Center ROI
        center_indices = (
            (points[:, 0] > self.center_roi_x_min) & (points[:, 0] < self.center_roi_x_max) &
            (points[:, 1] > -self.center_roi_y_width/2) & (points[:, 1] < self.center_roi_y_width/2)
        )
        center_points = points[center_indices]

        # Slow ROI
        slow_indices = (
            (points[:, 0] > self.slow_roi_x_min) & (points[:, 0] < self.slow_roi_x_max) &
            (points[:, 1] > -self.slow_roi_y_width/2) & (points[:, 1] < self.slow_roi_y_width/2)
        )
        slow_points = points[slow_indices]

        # Side ROI
        side_indices = (
            (points[:, 0] > self.side_roi_x_min) & (points[:, 0] < self.side_roi_x_max) &
            (points[:, 1] > -self.side_roi_y_width/2) & (points[:, 1] < self.side_roi_y_width/2)
        )
        side_points = points[side_indices]

        # 구역 세분화 (Left, Right)
        left_pts = side_points[side_points[:, 1] > 0.0]
        right_pts = side_points[side_points[:, 1] < 0.0]

        current_time = time.time()

        if len(center_points) > self.center_min_points:

            print(f"center_points={len(center_points)}")

            # 처음 조건 들어온 순간
            if self.center_condition_start_time is None:
                self.center_condition_start_time = current_time

            # 3초 안 지났으면 정지 유지
            elapsed = current_time - self.center_condition_start_time
            if elapsed < self.center_hold_duration:
                self.center_steer_msg.steer = 0.0
                self.center_steer_msg.is_ok = True

        else:
            # 조건 깨지면 타이머 리셋
            self.center_condition_start_time = None
                    
        if len(slow_points) >= self.slow_min_points:
            self.slow_steer_msg.is_ok = True

        if len(left_pts) >= self.side_min_points and len(left_pts) > len(right_pts):
            #self.get_logger().info("Obstacle in Left !!!")
            self.side_steer_msg.steer = 20.0 # 우측 조향
            self.side_steer_msg.is_ok = True
            
        elif len(right_pts) >= self.side_min_points and len(right_pts) > len(left_pts):
            #self.get_logger().info("Obstacle in Right !!!")
            self.side_steer_msg.steer = -20.0 # 좌측 조향
            self.side_steer_msg.is_ok = True
        
        self.lidar_center_steer_publisher.publish(self.center_steer_msg)
        self.lidar_slow_steer_publisher.publish(self.slow_steer_msg)
        self.lidar_side_steer_publisher.publish(self.side_steer_msg)

    def get_slot_occupancy(self, points, x_min, x_max, y_min, y_max):
        """
            True  -> slot 1 막힘
            False -> slot 1 비어있음 -> slot 1 주차
        """


        if points.size == 0:
            return False

        indices = (
            (points[:, 0] > x_min) & (points[:, 0] < x_max) &
            (points[:, 1] > y_min) & (points[:, 1] < y_max)
        )
        slot_points = points[indices]

        '''
        self.get_logger().info(
            f"[SLOT ROI] "
            f"x=({x_min:.2f},{x_max:.2f}), "
            f"y=({y_min:.2f},{y_max:.2f}), "
            f"points={len(slot_points)}"
        )

        if len(slot_points) > 0:
            self.get_logger().info(
                f"[SLOT POINT] min={slot_points.min(axis=0)}, "
                f"max={slot_points.max(axis=0)}"
            )
        '''

        self.get_logger().info(f"slot_points={len(slot_points)}")

        return len(slot_points) >= self.slot_min_points
    

    def process_parking(self, points):
        if self.flag == self.t_parking_flag:
            self.get_logger().info("[PARKING] T-PARKING FLAG 8 ACTIVE")

            # 이미 결정했으면 계속 같은 RDDF 선택값 publish
            if self.t_parking_decided:

                slot_msg = Int32()
                slot_msg.data = self.t_parking_target_slot

                self.lidar_rddf_select_publisher.publish(slot_msg)

                return

            if points.size == 0:
                return


            # -------------------------------------------------
            # 1. T-Parking 전체 ROI
            # -------------------------------------------------
            t_indices = (
                (points[:, 0] > self.t_slot1_x_min) &
                (points[:, 0] < self.t_slot1_x_max) &
                (points[:, 1] > self.t_slot1_y_min) &
                (points[:, 1] < self.t_slot1_y_max)
            )

            t_points = points[t_indices]

            if len(t_points) == 0:
                self.get_logger().info(
                    "[T-PARKING] ROI 내부 포인트 없음"
                )
                return

            self.get_logger().info(
                f"[T-PARKING] ROI points={len(t_points)}"
            )

            # -------------------------------------------------
            # 2. 각도순 정렬
            #
            # jump distance clustering은 인접한 LiDAR
            # point가 실제 스캔 순서여야 하므로 다시 정렬
            # -------------------------------------------------
            angles = np.arctan2(
                t_points[:, 1],
                t_points[:, 0]
            )

            sort_idx = np.argsort(angles)
            t_points_sorted = t_points[sort_idx]

            # -------------------------------------------------
            # 3. Jump Distance Clustering
            # -------------------------------------------------
            clusters = self.jump_distance_clustering(
                t_points_sorted
            )

            self.get_logger().info(
                f"[T-PARKING] clusters={len(clusters)}"
            )

            # -------------------------------------------------
            # 4. 클러스터가 너무 적으면 판단하지 않음
            # -------------------------------------------------
            if len(clusters) < 2:
                self.get_logger().info(
                    "[T-PARKING] 클러스터 부족 -> 대기"
                )

                self.publish_cluster_markers(clusters)

                return

            # -------------------------------------------------
            # 5. 클러스터 특징 계산
            # -------------------------------------------------
            cluster_info = []

            for i, cluster in enumerate(clusters):

                cluster_np = np.asarray(cluster)

                min_x = np.min(cluster_np[:, 0])
                max_x = np.max(cluster_np[:, 0])

                min_y = np.min(cluster_np[:, 1])
                max_y = np.max(cluster_np[:, 1])

                cx = np.mean(cluster_np[:, 0])
                cy = np.mean(cluster_np[:, 1])

                width_x = max_x - min_x
                width_y = max_y - min_y

                size = len(cluster_np)

                cluster_info.append({
                    "index": i,
                    "points": cluster_np,
                    "cx": cx,
                    "cy": cy,
                    "min_x": min_x,
                    "max_x": max_x,
                    "min_y": min_y,
                    "max_y": max_y,
                    "width_x": width_x,
                    "width_y": width_y,
                    "size": size
                })

            # -------------------------------------------------
            # 6. 클러스터를 y 기준으로 정렬
            #
            # y > 0 : 왼쪽
            # y < 0 : 오른쪽
            # -------------------------------------------------
            cluster_info.sort(
                key=lambda c: c["cx"]
            )

            # -------------------------------------------------
            # 7. 양 끝 클러스터 제거
            #
            # T자 주차장의 양쪽 풀/경계 구조물이
            # 하나의 큰 클러스터로 잡히는 경우를 제거.
            #
            # 단, 클러스터가 2개뿐이면 제거하지 않음.
            # -------------------------------------------------
            filtered_clusters = cluster_info.copy()

            if len(filtered_clusters) >= 4:

                # y 기준 가장 바깥쪽 클러스터 제거
                filtered_clusters = filtered_clusters[1:-1]

                self.get_logger().info(
                    "[T-PARKING] 양 끝 클러스터 제거"
                )

            # -------------------------------------------------
            # 8. 너무 큰 클러스터 제거
            #
            # 라바콘보다 지나치게 긴 구조물 제거
            # -------------------------------------------------
            cone_clusters = []

            for c in filtered_clusters:

                # 지나치게 많은 point를 가진 클러스터 제거
                if c["size"] > self.cluster_max_points:
                    self.get_logger().info(
                        f"[T-PARKING] 큰 클러스터 제거 "
                        f"id={c['index']} "
                        f"points={c['size']}"
                    )
                    continue

                # 지나치게 긴 클러스터 제거
                if c["width_x"] > 1.0 or c["width_y"] > 1.0:
                    self.get_logger().info(
                        f"[T-PARKING] 선형 클러스터 제거 "
                        f"id={c['index']} "
                        f"wx={c['width_x']:.2f} "
                        f"wy={c['width_y']:.2f}"
                    )
                    continue

                cone_clusters.append(c)

            # -------------------------------------------------
            # 9. 라바콘 후보가 없으면 대기
            # -------------------------------------------------
            if len(cone_clusters) == 0:

                self.get_logger().info(
                    "[T-PARKING] 라바콘 후보 없음"
                )

                self.publish_cluster_markers(
                    [c["points"] for c in filtered_clusters]
                )

                return

            # -------------------------------------------------
            # 10. 라바콘 후보 RViz 표시
            # -------------------------------------------------
            self.publish_cluster_markers(
                [c["points"] for c in cone_clusters]
            )

            # -------------------------------------------------
            # 11. 라바콘 분포 분석
            #
            # 현재 차량 기준:
            #
            # y > 0 : 왼쪽
            # y < 0 : 오른쪽
            #
            # Slot 1 / Slot 2를 y 위치로 분리
            # -------------------------------------------------

            left_cones = [
                c for c in cone_clusters
                if c["cx"] > 0.0
            ]

            right_cones = [
                c for c in cone_clusters
                if c["cx"] < 0.0
            ]

            self.get_logger().info(
                f"[T-PARKING] "
                f"left={len(left_cones)}, "
                f"right={len(right_cones)}"
            )

            # -------------------------------------------------
            # 12. 슬롯 판단
            #
            # 한쪽에 라바콘 구조가 집중되어 있으면
            # 반대쪽 슬롯을 선택
            # -------------------------------------------------

            # 왼쪽에 라바콘이 더 많음
            if len(left_cones) > len(right_cones):

                self.t_parking_target_slot = 2

                self.get_logger().info(
                    "[T-PARKING] "
                    "LEFT structure detected -> SLOT 2"
                )

            # 오른쪽에 라바콘이 더 많음
            elif len(right_cones) > len(left_cones):

                self.t_parking_target_slot = 1

                self.get_logger().info(
                    "[T-PARKING] "
                    "RIGHT structure detected -> SLOT 1"
                )

            # -------------------------------------------------
            # 양쪽 개수가 같으면 위치 분포로 추가 판단
            # -------------------------------------------------
            else:

                if len(left_cones) == 0 and len(right_cones) == 0:

                    self.get_logger().info(
                        "[T-PARKING] "
                        "유효한 라바콘 분포 없음 -> 대기"
                    )

                    return

                left_y = np.mean(
                    [abs(c["cx"]) for c in left_cones]
                ) if left_cones else 0.0

                right_y = np.mean(
                    [abs(c["cx"]) for c in right_cones]
                ) if right_cones else 0.0

                # 더 멀리 있는 쪽을 구조물 쪽으로 판단
                if right_y > left_y:

                    self.t_parking_target_slot = 2

                    self.get_logger().info(
                        "[T-PARKING] "
                        "LEFT distribution stronger -> SLOT 2"
                    )

                elif left_y > right_y:

                    self.t_parking_target_slot = 1

                    self.get_logger().info(
                        "[T-PARKING] "
                        "RIGHT distribution stronger -> SLOT 1"
                    )

                else:

                    self.get_logger().info(
                        "[T-PARKING] "
                        "판단 불확실 -> 대기"
                    )

                    return

            # -------------------------------------------------
            # 13. 선택 확정
            # -------------------------------------------------
            if self.t_parking_target_slot != 0:

                self.t_parking_decided = True

                slot_msg = Int32()
                slot_msg.data = self.t_parking_target_slot

                self.lidar_rddf_select_publisher.publish(
                    slot_msg
                )

                self.get_logger().info(
                    f"[T-PARKING] "
                    f"TARGET SLOT = "
                    f"{self.t_parking_target_slot}"
                )

            # 1. 왼쪽 전방 고정 ROI

            # 2. ROI 내 포인트들 클러스터링하여 큰 덩어리는 제거하고 라바콘만 남김

            # 3. 라바콘 분포 분석

            # 4. 슬롯 1 / 2 중 선택


        elif self.flag == self.p_parking_flag:

            self.get_logger().info("[PARKING] FLAG 9 ACTIVE")
            
            if self.p_parking_decided:
                slot_msg = Int32()
                slot_msg.data = self.p_parking_target_slot
                self.lidar_rddf_select_publisher.publish(slot_msg)
                return
            
            slot1_occupied = self.get_slot_occupancy(
                points,
                self.p_slot1_x_min, self.p_slot1_x_max,
                self.p_slot1_y_min, self.p_slot1_y_max
            )

            if slot1_occupied:
                self.p_parking_target_slot = 2
            else:
                self.p_parking_target_slot = 1

            # 하나라도 주차 가능한 공간을 찾으면 선택 확정
            if self.p_parking_target_slot != 0:
                self.p_parking_decided = True

            slot_msg = Int32()
            slot_msg.data = self.p_parking_target_slot
            self.lidar_rddf_select_publisher.publish(slot_msg)

            self.get_logger().info(
                f"[PARKING] target_slot={self.p_parking_target_slot}, RDDF={slot_msg.data}"
            )


    # =========================================================
    # Jump Distance Clustering
    # =========================================================
    def jump_distance_clustering(self, points):
        """
        인접한 LiDAR point 사이의 거리를 기준으로 클러스터링.

        distance > jump_threshold
            -> 새로운 cluster

        cluster_min_points ~ cluster_max_points
            -> 최종 표시
        """

        if points is None or len(points) == 0:
            return []

        clusters = []
        current_cluster = [points[0]]

        for i in range(1, len(points)):
            prev = points[i - 1]
            curr = points[i]

            distance = np.linalg.norm(curr - prev)

            if distance > self.jump_threshold:
                if (
                    self.cluster_min_points
                    <= len(current_cluster)
                    <= self.cluster_max_points
                ):
                    clusters.append(current_cluster)

                current_cluster = [curr]

            else:
                current_cluster.append(curr)

        # 마지막 cluster
        if (
            self.cluster_min_points
            <= len(current_cluster)
            <= self.cluster_max_points
        ):
            clusters.append(current_cluster)

        return clusters

    def publish_cluster_markers(self, clusters):
        """
        각 클러스터를 180도 회전한 좌표계로 RViz에 표시.
        
        원본 LiDAR 좌표:
            x, y
        
        180도 회전:
            x' = -x
            y' = -y
        """

        marker_array = MarkerArray()

        # -------------------------------------------------
        # 이전 marker 전체 삭제
        # -------------------------------------------------
        delete_marker = Marker()

        delete_marker.header.frame_id = "laser"
        delete_marker.header.stamp = self.get_clock().now().to_msg()

        delete_marker.ns = "jump_clusters"
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL

        marker_array.markers.append(delete_marker)

        # -------------------------------------------------
        # 클러스터별 색상
        # -------------------------------------------------
        colors = [
            (1.0, 0.0, 0.0),  # 빨강
            (0.0, 1.0, 0.0),  # 초록
            (0.0, 0.5, 1.0),  # 파랑
            (1.0, 0.0, 1.0),  # 보라
            (0.0, 1.0, 1.0),  # 하늘
            (1.0, 0.5, 0.0),  # 주황
            (1.0, 1.0, 0.0),  # 노랑
            (0.5, 0.0, 1.0),  # 자주
        ]

        # -------------------------------------------------
        # 클러스터 표시
        # -------------------------------------------------
        for i, cluster in enumerate(clusters):

            color = colors[i % len(colors)]
            r, g, b = color

            marker = Marker()

            marker.header.frame_id = "laser"
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.ns = "jump_clusters"
            marker.id = i + 1

            marker.type = Marker.POINTS
            marker.action = Marker.ADD

            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.06
            marker.scale.y = 0.06

            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1.0

            # -------------------------------------------------
            # Point
            # -------------------------------------------------
            for point in cluster:

                p = Point()

                # ★ 180도 회전
                p.x = float(-point[0])
                p.y = float(-point[1])
                p.z = 0.0

                marker.points.append(p)

            marker_array.markers.append(marker)

            # -------------------------------------------------
            # Centroid
            # -------------------------------------------------
            cluster_np = np.asarray(cluster)

            cx = float(np.mean(cluster_np[:, 0]))
            cy = float(np.mean(cluster_np[:, 1]))

            # ★ centroid도 180도 회전
            rotated_cx = -cx
            rotated_cy = -cy

            text_marker = Marker()

            text_marker.header.frame_id = "laser"
            text_marker.header.stamp = self.get_clock().now().to_msg()

            text_marker.ns = "jump_cluster_text"
            text_marker.id = 1000 + i

            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = rotated_cx
            text_marker.pose.position.y = rotated_cy
            text_marker.pose.position.z = 0.15

            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = 0.15

            text_marker.color.r = r
            text_marker.color.g = g
            text_marker.color.b = b
            text_marker.color.a = 1.0

            text_marker.text = f"{i}: {len(cluster)} pts"

            marker_array.markers.append(text_marker)

        # -------------------------------------------------
        # Publish
        # -------------------------------------------------
        self.cluster_pub.publish(marker_array)

    def process_back_obstacle(self, points):
        """
        FLAG 1 후진 주차 충돌 방지

        좌측 장애물  -> 우측 조향 (+)
        우측 장애물  -> 좌측 조향 (-)

        반환:
            steer : 회피 조향각
            is_ok : 조향 명령 사용 여부
        """

        if points.size == 0:
            return 0.0, False

        # -----------------------------
        # Back Left ROI
        # -----------------------------
        left_indices = (
            (points[:, 0] > self.back_roi_x_min) &
            (points[:, 0] < self.back_roi_x_max) &
            (points[:, 1] > self.back_left_y_min) &
            (points[:, 1] < self.back_left_y_max)
        )

        left_points = points[left_indices]

        # -----------------------------
        # Back Right ROI
        # -----------------------------
        right_indices = (
            (points[:, 0] > self.back_roi_x_min) &
            (points[:, 0] < self.back_roi_x_max) &
            (points[:, 1] > self.back_right_y_min) &
            (points[:, 1] < self.back_right_y_max)
        )

        right_points = points[right_indices]

        # 포인트가 너무 적으면 장애물로 판단하지 않음
        left_valid = len(left_points) >= self.back_min_points
        right_valid = len(right_points) >= self.back_min_points

        # -----------------------------
        # 디버깅
        # -----------------------------
        if left_valid or right_valid:
            self.get_logger().info(
                f"[BACK] "
                f"left={len(left_points)}pts, "
                f"right={len(right_points)}pts"
            )

        # -----------------------------
        # 양쪽 모두 위험
        # -----------------------------
        if left_valid and right_valid:
            self.get_logger().warn(
                "[BACK] BOTH SIDES DANGER !!!"
            )
            # 양쪽 모두 막혀있으므로 한쪽으로 강제 조향하지 않음
            return 0.0, False

        # -----------------------------
        # 왼쪽 장애물
        # -----------------------------
        if left_valid:
            self.get_logger().warn(
                f"[BACK] LEFT BLOCKED "
            )

            # 후진 시 좌측 장애물 -> 우측 조향
            return self.back_max_steer, True

        # -----------------------------
        # 오른쪽 장애물
        # -----------------------------
        if right_valid:
            self.get_logger().warn(
                f"[BACK] RIGHT BLOCKED "
            )

            # 후진 시 우측 장애물 -> 좌측 조향
            return -self.back_max_steer, True

        # -----------------------------
        # 장애물 없음
        # -----------------------------
        return 0.0, False

    def publish_roi(self):

        def make_marker(marker_id, x_min, x_max, y_min, y_max, r, g, b):
            marker = Marker()

            marker.header.frame_id = "laser"
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.ns = "roi"
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            marker.scale.x = 0.03

            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1.0

            marker.points = [
                Point(x=-x_min, y=-y_min, z=0.0),
                Point(x=-x_max, y=-y_min, z=0.0),
                Point(x=-x_max, y=-y_max, z=0.0),
                Point(x=-x_min, y=-y_max, z=0.0),
                Point(x=-x_min, y=-y_min, z=0.0)
            ]

            return marker

        # Slow ROI
        self.roi_pub.publish(
            make_marker(
                0,
                self.slow_roi_x_min,
                self.slow_roi_x_max,
                -self.slow_roi_y_width / 2,
                self.slow_roi_y_width / 2,
                0.0, 0.0, 1.0
            )
        )

        # Side ROI
        self.roi_pub.publish(
            make_marker(
                1,
                self.side_roi_x_min,
                self.side_roi_x_max,
                -self.side_roi_y_width / 2,
                self.side_roi_y_width / 2,
                1.0, 1.0, 0.0
            )
        )

        # Center ROI
        self.roi_pub.publish(
            make_marker(
                2,
                self.center_roi_x_min,
                self.center_roi_x_max,
                -self.center_roi_y_width / 2,
                self.center_roi_y_width / 2,
                0.0, 1.0, 0.0
            )
        )

        # T-Parking Slot 1 ROI
        self.roi_pub.publish(
            make_marker(
                3,
                self.t_slot1_x_min,
                self.t_slot1_x_max,
                self.t_slot1_y_min,
                self.t_slot1_y_max,
                1.0, 0.0, 0.0
            )
        )

        # P-Parking Slot 1 ROI
        self.roi_pub.publish(
            make_marker(
                4,
                self.p_slot1_x_min,
                self.p_slot1_x_max,
                self.p_slot1_y_min,
                self.p_slot1_y_max,
                1.0, 0.0, 0.0
            )
        )

        # Back Left ROI
        self.roi_pub.publish(
            make_marker(
                5,
                self.back_roi_x_min,
                self.back_roi_x_max,
                self.back_left_y_min,
                self.back_left_y_max,
                1.0, 0.0, 1.0
            )
        )

        # Back Right ROI
        self.roi_pub.publish(
            make_marker(
                6,
                self.back_roi_x_min,
                self.back_roi_x_max,
                self.back_right_y_min,
                self.back_right_y_max,
                1.0, 0.0, 1.0
            )
        )

def main(args=None):
    rclpy.init(args=args)
    node = LidarObstacleDetector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt')

    finally:
        node.center_steer_msg.steer = 0.0
        node.center_steer_msg.is_ok = False
        node.slow_steer_msg.steer = 0.0
        node.slow_steer_msg.is_ok = False
        node.side_steer_msg.steer = 0.0
        node.side_steer_msg.is_ok = False
        node.lidar_center_steer_publisher.publish(node.center_steer_msg)
        node.lidar_slow_steer_publisher.publish(node.slow_steer_msg)
        node.lidar_side_steer_publisher.publish(node.side_steer_msg)
        time.sleep(0.1)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()