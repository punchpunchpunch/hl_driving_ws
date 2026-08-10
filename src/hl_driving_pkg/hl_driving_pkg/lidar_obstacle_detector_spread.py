import rclpy
from rclpy.node import Node

import numpy as np
import time

from sensor_msgs.msg import LaserScan
from auto_car_msgs.msg import SteerMsg

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile
from rclpy.qos import DurabilityPolicy

class LidarObstacleDetector(Node):
    def __init__(self):
        super().__init__('lidar_obstacle_detector')

        self.grid_size = 0.02

        # Center ROI
        self.center_roi_x_min, self.center_roi_x_max = 0.3, 3.0
        self.center_roi_y_width = 0.6  # 좌우 전체

        # Slow ROI
        self.slow_roi_x_min, self.slow_roi_x_max = 0.0, 3.5
        self.slow_roi_y_width = 1.8  # 좌우 전체

        # Side ROI
        self.side_roi_x_min, self.side_roi_x_max = -0.3, 1.5
        self.side_roi_y_width = 2.0  # 좌우 전체

        self.center_condition_start_time = None
        self.center_prev_detected = False   # 이전 상태
        self.center_hold_active = False     # 5초 유지 상태
        self.center_cooldown = False
        self.center_hold_start_time = 0.0   # 시작 시간
        self.center_hold_duration = 3.0     # 유지 시간 (초)

        self.center_steer_msg = SteerMsg()
        self.center_steer_msg.steer = 0.0
        self.center_steer_msg.is_ok = False

        self.slow_steer_msg = SteerMsg()
        self.slow_steer_msg.steer = 0.0
        self.slow_steer_msg.is_ok = False

        self.side_steer_msg = SteerMsg()
        self.side_steer_msg.steer = 0.0
        self.side_steer_msg.is_ok = False

        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
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

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.roi_pub = self.create_publisher(
            Marker, 'roi_marker', qos
        )
        
        self.create_timer(0.5, self.publish_roi)

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

        # 시각화
        #self.visualize(points)

    # 다운샘플링 centroid 방식으로 수정 요 edit
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

        # ROI 내부 포인트 필터링 (전방 0~3m, 좌우 -0.5~0.5m)
        """
        roi_indices = (points[:, 0] > self.roi_x_min) & (points[:, 0] < self.roi_x_max) & \
                      (points[:, 1] > -self.roi_y_width/2) & (points[:, 1] < self.roi_y_width/2)
        roi_points = points[roi_indices]
        """
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
            
            if len(center_points) > 20:
                center = np.mean(center_points, axis=0)
                spread = np.mean(np.linalg.norm(center_points - center, axis=1))
                print(spread)
                if spread < 0.25:
                    condition = True

            if condition:
                # 처음 조건 들어온 순간
                if self.center_condition_start_time is None:
                    self.center_condition_start_time = current_time

                # 3초 안 지났으면 정지 유지
                if current_time - self.center_condition_start_time < 3.0:
                    self.center_steer_msg.steer = 0.0
                    self.center_steer_msg.is_ok = True
                else:
                    # 3초 지나면 무조건 해제
                    self.center_steer_msg.is_ok = False

            else:
                # 조건 깨지면 타이머 리셋
                self.center_condition_start_time = None
                self.center_steer_msg.is_ok = False
                    
            if len(slow_points) > 8:
                self.slow_steer_msg.is_ok = True

            if len(left_pts) > 8 and len(left_pts) > len(right_pts):
                #self.get_logger().info("Obstacle in Left !!!")
                self.side_steer_msg.steer = 20.0 # 우측 조향
                self.side_steer_msg.is_ok = True
            
            elif len(right_pts) > 8 and len(right_pts) > len(left_pts):
                #self.get_logger().info("Obstacle in Right !!!")
                self.side_steer_msg.steer = -20.0 # 좌측 조향
                self.side_steer_msg.is_ok = True
        
        self.lidar_center_steer_publisher.publish(self.center_steer_msg)
        self.lidar_slow_steer_publisher.publish(self.slow_steer_msg)
        self.lidar_side_steer_publisher.publish(self.side_steer_msg)

    def publish_roi(self):
        marker = Marker()

        marker.header.frame_id = "laser"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "roi"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.03  # 선 두께

        # 색상 (빨간색 + 투명도)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

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

        marker.points = points

        self.roi_pub.publish(marker)

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