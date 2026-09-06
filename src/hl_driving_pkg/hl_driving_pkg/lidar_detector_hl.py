import rclpy
from rclpy.node import Node

import numpy as np
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

        self.t_parking_flag = 8 # 직각주차 플래그
        self.p_parking_flag = 9 # 평행주차 플래그

        self.center_min_points = 5
        self.slow_min_points = 8
        self.side_min_points = 8
        self.slot_min_points = 5 # 막혀 있다고 판단하는 기준이 되는 포인트 수

        self.grid_size = 0.02

        # Center ROI
        self.center_roi_x_min, self.center_roi_x_max = 0.1, 4.0
        self.center_roi_y_width = 1.4  # 좌우 전체

        # Slow ROI
        self.slow_roi_x_min, self.slow_roi_x_max = 0.0, 3.5
        self.slow_roi_y_width = 1.8  # 좌우 전체

        # Side ROI
        self.side_roi_x_min, self.side_roi_x_max = -0.3, 1.5
        self.side_roi_y_width = 2.0  # 좌우 전체

        # T-Parking ROI
        self.t_slot1_x_min, self.t_slot1_x_max = 0.5, 2.0
        self.t_slot1_y_min, self.t_slot1_y_max = 2.0, 6.0

        # P-Parking ROI
        self.p_slot1_x_min, self.p_slot1_x_max = 8.0, 12.0
        self.p_slot1_y_min, self.p_slot1_y_max = 0.0, 2.0


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

        # 직각주차 rddf 선택 판단
        self.process_parking(points)

    # 다운샘플링 centroid 방식으로 수정 요 edit
    def grid_downsampling(self, points, grid_size):
        if len(points) == 0: return points
        # 좌표를 격자 크기로 나누고 정수화하여 인덱스 부여
        keys = (points / grid_size).astype(int)
        # 중복된 격자 키 제거 (격자당 포인트 1개 유지)
        _, indices = np.unique(keys, axis=0, return_index=True)
        return points[indices]
    '''
    def clustering(self, points):
        if len(points) == 0:
            return np.array([])
        db = DBSCAN(eps=0.15, min_samples=5).fit(points)
        return db.labels_
    '''

    def process_roi(self, points):
        self.center_steer_msg.is_ok = False
        self.slow_steer_msg.is_ok = False
        self.side_steer_msg.is_ok = False
        self.side_steer_msg.steer = 0.0

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

        if len(center_points) > 0 or len(slow_points) > 0 or len(side_points) > 0:
            """
            # 구역 세분화 (Center, Left, Right)
            center_pts = roi_points[(roi_points[:, 1] >= -0.3) & (roi_points[:, 1] <= 0.3)]
            left_pts = roi_points[roi_points[:, 1] > 0.0]
            right_pts = roi_points[roi_points[:, 1] < 0.0]

            if len(center_pts) > 10: # 임계값 설정
                center_detected = True
            """
            # 구역 세분화 (Left, Right)
            left_pts = side_points[side_points[:, 1] > 0.0]
            right_pts = side_points[side_points[:, 1] < 0.0]

            current_time = time.time()
            condition = False
            
            if len(center_points) > self.center_min_points:
                    condition = True
                    print(len(center_points))

            if condition:
                # 처음 조건 들어온 순간
                if self.center_condition_start_time is None:
                    self.center_condition_start_time = current_time

                # 3초 안 지났으면 정지 유지
                if current_time - self.center_condition_start_time < self.center_hold_duration:
                    self.center_steer_msg.steer = 0.0
                    self.center_steer_msg.is_ok = True
                else:
                    # 3초 지나면 무조건 해제
                    self.center_steer_msg.is_ok = False

            else:
                # 조건 깨지면 타이머 리셋
                self.center_condition_start_time = None
                self.center_steer_msg.is_ok = False
                    
            if len(slow_points) > self.slow_min_points:
                self.slow_steer_msg.is_ok = True

            if len(left_pts) > self.side_min_points and len(left_pts) > len(right_pts):
                #self.get_logger().info("Obstacle in Left !!!")
                self.side_steer_msg.steer = 20.0 # 우측 조향
                self.side_steer_msg.is_ok = True
            
            elif len(right_pts) > self.side_min_points and len(right_pts) > len(left_pts):
                #self.get_logger().info("Obstacle in Right !!!")
                self.side_steer_msg.steer = -20.0 # 좌측 조향
                self.side_steer_msg.is_ok = True
        
        self.lidar_center_steer_publisher.publish(self.center_steer_msg)
        self.lidar_slow_steer_publisher.publish(self.slow_steer_msg)
        self.lidar_side_steer_publisher.publish(self.side_steer_msg)

    def get_slot_occupancy(self, points, x_min, x_max, y_min, y_max):
        """
            True  -> 막힘
            False -> 비어있음 -> 주차
        """
        if points.size == 0:
            return False

        indices = (
            (points[:, 0] > x_min) & (points[:, 0] < x_max) &
            (points[:, 1] > y_min) & (points[:, 1] < y_max)
        )
        slot_points = points[indices]

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

        if len(slot_points) < self.slot_min_points:
            return False
        """
        # DBSCAN으로 실제 물체가 뭉쳐 있는지 확인
        if len(slot_points) >= 5:
            labels = self.clustering(slot_points)
            valid_labels = labels[labels >= 0]

            if len(valid_labels) > 0:
                _, counts = np.unique(valid_labels, return_counts=True)
                if np.max(counts) >= 5:
                    return True
                    """

        # DBSCAN이 잘 안 되는 경우 그냥 점 개수로 판단
        return len(slot_points) >= self.slot_min_points
    

    def process_parking(self, points):
        if self.flag == self.t_parking_flag:

            self.get_logger().info("[PARKING] FLAG 8 ACTIVE")
            
            if self.t_parking_decided:
                slot_msg = Int32()
                slot_msg.data = self.t_parking_target_slot
                self.lidar_rddf_select_publisher.publish(slot_msg)
                return
            
            slot1_occupied = self.get_slot_occupancy(
                points,
                self.t_slot1_x_min, self.t_slot1_x_max,
                self.t_slot1_y_min, self.t_slot1_y_max
            )

            if slot1_occupied:
                self.t_parking_target_slot = 2
            else:
                self.t_parking_target_slot = 1

            # 하나라도 주차 가능한 공간을 찾으면 선택 확정
            if self.t_parking_target_slot != 0:
                self.t_parking_decided = True

            slot_msg = Int32()
            slot_msg.data = self.t_parking_target_slot
            self.lidar_rddf_select_publisher.publish(slot_msg)

            self.get_logger().info(
                f"[PARKING] target_slot={self.t_parking_target_slot}, RDDF={slot_msg.data}"
            )

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
        """
        parking_flag 구간에서 차량이 계속 주행하면서
        Slot 1 / Slot 2의 점유 여부를 여러 프레임 누적한다.

        최종 결과:
            /parking_slot = 1 -> Slot 1 선택
            /parking_slot = 2 -> Slot 2 선택
            /parking_slot = 0 -> 아직 미결정
        """

    def publish_roi(self):

        slow_marker = Marker()

        slow_marker.header.frame_id = "laser"
        slow_marker.header.stamp = self.get_clock().now().to_msg()

        slow_marker.ns = "roi"
        slow_marker.id = 0
        slow_marker.type = Marker.LINE_STRIP
        slow_marker.action = Marker.ADD

        slow_marker.scale.x = 0.03  # 선 두께

        # 색상 + 투명도
        slow_marker.color.r = 0.0
        slow_marker.color.g = 0.0
        slow_marker.color.b = 1.0
        slow_marker.color.a = 1.0

        # ROI 좌표
        x_min = self.slow_roi_x_min
        x_max = self.slow_roi_x_max
        y_min = -self.slow_roi_y_width / 2
        y_max = self.slow_roi_y_width / 2

        # 사각형
        points = [
            Point(x=-x_min, y=y_min, z=0.0),
            Point(x=-x_max, y=y_min, z=0.0),
            Point(x=-x_max, y=y_max, z=0.0),
            Point(x=-x_min, y=y_max, z=0.0),
            Point(x=-x_min, y=y_min, z=0.0)
        ]

        slow_marker.points = points

        self.roi_pub.publish(slow_marker)

        t_parking_marker = Marker()

        t_parking_marker.header.frame_id = "laser"
        t_parking_marker.header.stamp = self.get_clock().now().to_msg()

        t_parking_marker.ns = "roi"
        t_parking_marker.id = 1
        t_parking_marker.type = Marker.LINE_STRIP
        t_parking_marker.action = Marker.ADD

        t_parking_marker.scale.x = 0.03

        # T-Parking ROI 색상
        t_parking_marker.color.r = 1.0
        t_parking_marker.color.g = 0.0
        t_parking_marker.color.b = 0.0
        t_parking_marker.color.a = 1.0

        x_min = self.t_slot1_x_min
        x_max = self.t_slot1_x_max
        y_min = self.t_slot1_y_min
        y_max = self.t_slot1_y_max

        t_parking_marker.points = [
            Point(x=-x_min, y=-y_min, z=0.0),
            Point(x=-x_max, y=-y_min, z=0.0),
            Point(x=-x_max, y=-y_max, z=0.0),
            Point(x=-x_min, y=-y_max, z=0.0),
            Point(x=-x_min, y=-y_min, z=0.0)
        ]

        self.roi_pub.publish(t_parking_marker)

        p_parking_marker = Marker()

        p_parking_marker.header.frame_id = "laser"
        p_parking_marker.header.stamp = self.get_clock().now().to_msg()

        p_parking_marker.ns = "roi"
        p_parking_marker.id = 2
        p_parking_marker.type = Marker.LINE_STRIP
        p_parking_marker.action = Marker.ADD

        p_parking_marker.scale.x = 0.03

        # T-Parking ROI 색상
        p_parking_marker.color.r = 1.0
        p_parking_marker.color.g = 0.0
        p_parking_marker.color.b = 0.0
        p_parking_marker.color.a = 1.0

        x_min = self.p_slot1_x_min
        x_max = self.p_slot1_x_max
        y_min = self.p_slot1_y_min
        y_max = self.p_slot1_y_max

        p_parking_marker.points = [
            Point(x=-x_min, y=-y_min, z=0.0),
            Point(x=-x_max, y=-y_min, z=0.0),
            Point(x=-x_max, y=-y_max, z=0.0),
            Point(x=-x_min, y=-y_max, z=0.0),
            Point(x=-x_min, y=-y_min, z=0.0)
        ]

        self.roi_pub.publish(p_parking_marker)

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