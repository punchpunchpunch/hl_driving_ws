import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32
from sensor_msgs.msg import NavSatFix
from ublox_msgs.msg import NavPVT
from auto_car_msgs.msg import SteerMsg
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from rclpy.qos import QoSProfile
from rclpy.qos import DurabilityPolicy

import os
import math
import time
import csv
import utm
import subprocess

class WaypointFollower(Node):

    def __init__(self):
        super().__init__('waypoint_follower')

        # =====================
        # parameters
        # =====================
        self.declare_parameter('waypoint_csv', '')

        # =====================
        # waypoints CSV file
        # =====================
        self.file_dir = os.path.expanduser('~/hl_driving_ws/src/hl_driving_pkg/hl_driving_pkg/waypoints')
        self.rddf_files = [
            'waypoints_20260812_162603.csv',  #1 src/hl_driving_pkg/hl_driving_pkg/waypoints/waypoints_20260812_162603.csv
            'waypoints_20260812_160905.csv',  #2 src/hl_driving_pkg/hl_driving_pkg/waypoints/waypoints_20260812_160905.csv
            'waypoints_20260812_161444.csv',  #3 src/hl_driving_pkg/hl_driving_pkg/waypoints/waypoints_20260812_161444.csv
            'waypoints_20260810_013624.csv',  #4
            'waypoints_20260810_013624.csv',  #5
            'waypoints_20260810_013624.csv',  #6
            'waypoints_20260810_013624.csv'   #7
        ]

        # =====================
        # internal state
        # =====================
        self.ego_x = None
        self.ego_y = None
        self.ego_heading = None
        self.closest_idx = 0
        self.flag = 0
        self.rddf_num = 0
        self.lidar_select = 0
        self.rddf_finished = False

        self.current_group = 0
        self.waypoint_groups = []

        self.lookahead_distance = 3.0   # lookahad (랩뷰에서는 2.5, 작은차는 1.0, 큰차는 일단 2m?)
        self.wheelbase = 0.7            # 대강 측정했을 때 70cm
        self.control_rate = 20.0        # 제어루프 주기

        self.steer_msg = SteerMsg()
        self.steer_msg.steer = 0.0
        self.steer_msg.is_ok = True

        self.load_rddf()

        # =====================
        # subscriptions
        # =====================
        self.gps_subscription = self.create_subscription(
            NavSatFix,
            '/ublox_gps_node/fix',
            self.gps_callback,
            10 # 큐 크기
        )

        self.navpvt_subscription = self.create_subscription(
            NavPVT,
            '/ublox_gps_node/navpvt',
            self.navpvt_callback,
            10 # 큐 크기
        )

        self.lidar_subscription = self.create_subscription(
            Int32,
            '/lidar_rddf_select',
            self.lidar_select_callback,
            10
        )

        # =====================
        # publishers (RViz)
        # =====================
        self.gps_steer_publisher = self.create_publisher(
            SteerMsg,
            'gps_steer',
            10
        )

        self.ego_pose_pub = self.create_publisher(
            PoseStamped, '/ego_pose', 10
        )

        self.closest_wp_pub = self.create_publisher(
            PoseStamped, '/closest_waypoint', 10
        )

        self.target_wp_pub = self.create_publisher(
            PoseStamped, '/target_waypoint', 10
        )

        self.waypoints_flag_pub = self.create_publisher(
            Int32, '/waypoints_flag', 10
        )

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.waypoints_path_pub = self.create_publisher(
            Path, '/waypoints_path', qos
        )

        # timer - loop period
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)

        self.create_timer(0.5, self.publish_waypoints_path)

        self.get_logger().info('WaypointFollower start')

    # =====================================================
    # Load Waypoints CSV
    # =====================================================

    def load_waypoints(self, csv_path):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f'Waypoint CSV not found: {csv_path}')
        waypoints = []

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                waypoints.append(
                    (float(row['utm_x']), float(row['utm_y']), int(row['flag']))
                )
        return waypoints

    def create_waypoint_groups(self):

        """
        flag == 1  → 하나의 별도 trajectory
        flag != 1  → 하나의 별도 trajectory
        waypoint 순서상 연속된 구간을 하나의 group으로 취급
        """

        self.waypoint_groups = []

        if not self.waypoints:
            return

        start_idx = 0

        current_is_reverse = (self.waypoints[0][2] == 1)

        for i in range(1, len(self.waypoints)):

            is_reverse = (self.waypoints[i][2] == 1)

            # flag == 1 / flag != 1 상태가 바뀌면
            # 새로운 trajectory 시작
            if is_reverse != current_is_reverse:

                self.waypoint_groups.append(
                    {
                        'start': start_idx,
                        'end': i - 1,
                        'reverse': current_is_reverse
                    }
                )

                start_idx = i
                current_is_reverse = is_reverse

        # 마지막 group
        self.waypoint_groups.append(
            {
                'start': start_idx,
                'end': len(self.waypoints) - 1,
                'reverse': current_is_reverse
            }
        )

        """
        self.get_logger().info(
            f'Trajectory groups: {len(self.waypoint_groups)}'
        )

        for i, group in enumerate(self.waypoint_groups):

            group_type = (
                'FLAG 1'
                if group['reverse']
                else 'FLAG != 1'
            )

            self.get_logger().info(
                f'  Group {i}: '
                f'{group["start"]} ~ {group["end"]} '
                f'({group_type})'
            )
        """

    def load_rddf(self):
            
        csv_path = os.path.join(self.file_dir, self.rddf_files[self.rddf_num])
    
        self.waypoints = self.load_waypoints(csv_path)
    
        # flag == 1 / flag != 1 기준으로 trajectory group 생성
        self.create_waypoint_groups()

        # 첫 번째 group부터 시작
        self.current_group = 0

        self.closest_idx = (
            self.waypoint_groups[0]['start']
        )

        self.origin_x = self.waypoints[0][0]
        self.origin_y = self.waypoints[0][1]

        group = self.waypoint_groups[
            self.current_group
        ]

        group_type = (
            'FLAG 1'
            if group['reverse']
            else 'FLAG != 1'
        )

        self.get_logger().info(
            f'Loaded {len(self.waypoints)} waypoints '
            f'from RDDF {self.rddf_num + 1}'
        )

        self.get_logger().info(
            f'Start Group {self.current_group}: '
            f'{group["start"]} ~ {group["end"]} '
            f'({group_type})'
        )

    def load_next_group(self):

        next_group = self.current_group + 1

        # 현재 RDDF 안에 다음 group이 존재
        if next_group < len(self.waypoint_groups):

            self.current_group = next_group

            group = self.waypoint_groups[
                self.current_group
            ]

            # 새로운 trajectory의 시작점으로 이동
            self.closest_idx = group['start']

            group_type = (
                'FLAG 1'
                if group['reverse']
                else 'FLAG != 1'
            )

            self.get_logger().info(
                f'================================'
            )

            self.get_logger().info(
                f'New trajectory group: '
                f'{self.current_group}'
            )

            self.get_logger().info(
                f'Waypoint range: '
                f'{group["start"]} ~ {group["end"]}'
            )

            self.get_logger().info(
                f'Type: {group_type}'
            )

            self.get_logger().info(
                f'================================'
            )

            return True

        # 현재 RDDF의 모든 group 완료
        return False

    def load_next_rddf(self):
        if self.rddf_num == 0:
            # 2번 RDDF
            if self.lidar_select == 2:
                self.rddf_num = 1
            # 3번 RDDF
            else:
                self.rddf_num = 2

            self.get_logger().info(f'lidar_select = {self.lidar_select}')
            self.lidar_select = 0

        elif self.rddf_num == 1 or self.rddf_num == 2:
            self.rddf_num = 3

        elif self.rddf_num == 3:
            # 5번 RDDF
            if self.lidar_select == 5:
                self.rddf_num = 4
            # 6번 RDDF
            else:
                self.rddf_num = 5

            self.get_logger().info(f'lidar_select = {self.lidar_select}')
            self.lidar_select = 0

        elif self.rddf_num == 4 or self.rddf_num == 5:
            self.rddf_num = 6
        elif self.rddf_num >= len(self.rddf_files) - 1:
            self.get_logger().info('All RDDF completed!')
            return False

        self.load_rddf()

        return True

    # =====================================================
    # Callbacks
    # =====================================================

    def lidar_select_callback(self, msg: Int32):
        self.lidar_select = msg.data

        self.get_logger().info(
            f'LiDAR selection updated: {self.lidar_select}'
        )

    def gps_callback(self, msg: NavSatFix):
        #if msg.status.status < 0:
        #    return

        self.ego_x, self.ego_y, _, _ = utm.from_latlon(msg.latitude, msg.longitude)
    
    def navpvt_callback(self, msg: NavPVT):
        """
        NavPVT.heading
        - Heading of motion 2-D [deg / 1e-5]
        - 방위각(True North, CW(+))
        """

        #if msg.heading == 0x7FFFFFFF:
        #    return

        # 1) heading (deg)
        heading_deg = msg.heading * 1e-5

        # 여기서부턴 콜백에서 옮겨야할지도
        # 2) deg → rad
        heading_rad = math.radians(heading_deg)

        # 3) North-CW → East-CCW 변환
        #    Pure Pursuit 좌표계에 맞춤
        self.ego_heading = math.pi / 2.0 - heading_rad

        # [-pi, pi] 정규화
        self.ego_heading = math.atan2(
            math.sin(self.ego_heading),
            math.cos(self.ego_heading)
        )

    # =====================
    # Control Loop
    # =====================
    def control_loop(self):
        if None in (self.ego_x, self.ego_y, self.ego_heading):
            self.get_logger().info('No Input')
            return

        if self.rddf_finished:
            return

        group = self.waypoint_groups[self.current_group]
        group_start = group['start']
        group_end = group['end']

        # 1. closest waypoint
        self.closest_idx = self.find_closest_waypoint(
            self.ego_x,
            self.ego_y,
            self.closest_idx,
            group_start,
            group_end,
            window=50
        )

        # 마지막 waypoint 도착하면 다음 RDDF 열기
        if self.closest_idx >= group_end:
            # 다음 group 존재
            if self.load_next_group():
                return
            
            if not self.load_next_rddf():
                self.rddf_finished = True
            return

        # 2. lookahead target
        target_x, target_y, flag = self.find_target_PurePursuit(
            self.closest_idx,
            self.lookahead_distance,
            group_end
        )

        # 3. pure pursuit
        steer = self.compute_steer_PurePursuit(
            self.ego_x,
            self.ego_y,
            self.ego_heading,
            target_x,
            target_y,
            self.lookahead_distance,
            self.wheelbase
        )
    
        # 4. steer(deg) 퍼블리시
        #self.get_logger().info(f'heading={self.ego_heading:.2f}, steer={steer:.2f}')       
        self.steer_msg.steer = -steer
        self.gps_steer_publisher.publish(self.steer_msg)

        # flag publish
        self.publish_flag()

        # RViz publish
        self.publish_ego_pose()
        self.publish_closest_waypoint()
        self.publish_target_waypoint(target_x, target_y)

    # =====================
    # Core Logic
    # =====================
    # 로컬 경로 + 로컬 경로 내 가장 가까운 waypoint 지정(현재 코드에서는 앞으로만. 뒤로 돌아갈 수 없음)
    def find_closest_waypoint(self, ego_x, ego_y, start, group_start, group_end, window):
        min_d = float('inf')
        idx = start

        # 현재 group 안에서만 탐색
        search_start = max(start, group_start)
        search_end = min(start + window, group_end + 1)
     
        for i in range(search_start, search_end):
            dx = self.waypoints[i][0] - ego_x
            dy = self.waypoints[i][1] - ego_y
            d = dx*dx + dy*dy
            if d < min_d:
                min_d = d
                idx = i

        return idx

    # Pure Pursuit: 가장 가까운 waypoint에서 lookahad만큼 앞에 있는 목표 waypoint 지정
    def find_target_PurePursuit(self, idx, lookahead, group_end):
        dist = 0.0
        for i in range(idx, group_end):
            dx = self.waypoints[i+1][0] - self.waypoints[i][0]
            dy = self.waypoints[i+1][1] - self.waypoints[i][1]
            seg = math.hypot(dx, dy)
            dist += seg
            if dist >= lookahead:
                return self.waypoints[i+1]
        return self.waypoints[group_end]

    # Pure Pursuit: Pure Pursuit 알고리즘으로 조향값 계산
    def compute_steer_PurePursuit(self, ego_x, ego_y, ego_heading, tgt_x, tgt_y, lookahead, wheelbase):
        dx = tgt_x - ego_x
        dy = tgt_y - ego_y
        Ld = max(math.hypot(dx, dy), 1.0)
        #Ld = math.hypot(dx, dy)

        target_heading = math.atan2(dy, dx)
        alpha = target_heading - ego_heading
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        # 랩뷰에서는 lookahead 그대로 안쓰고 누적 거리 사용?
        curvature = 2.0 * math.sin(alpha) / Ld
        steer_rad = math.atan(wheelbase * curvature)
        steer_deg = math.degrees(steer_rad)

        return steer_deg

    # =====================
    # RViz Publish
    # =====================
    def publish_waypoints_path(self):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()

        for wp in self.waypoints:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            # 상대좌표 적용
            pose.pose.position.x = wp[0] - self.origin_x
            pose.pose.position.y = wp[1] - self.origin_y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self.waypoints_path_pub.publish(path)

    def publish_ego_pose(self):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.position.x = self.ego_x - self.origin_x
        msg.pose.position.y = self.ego_y - self.origin_y
        msg.pose.position.z = 0.0

        # yaw -> quaternion (z축 회전만)
        yaw = self.ego_heading
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)

        self.ego_pose_pub.publish(msg)

    def publish_closest_waypoint(self):
        wp = self.waypoints[self.closest_idx]

        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.position.x = wp[0] - self.origin_x
        msg.pose.position.y = wp[1] - self.origin_y
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0

        self.closest_wp_pub.publish(msg)

    def publish_target_waypoint(self, x, y):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.position.x = x - self.origin_x
        msg.pose.position.y = y - self.origin_y
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0

        self.target_wp_pub.publish(msg)

    def publish_flag(self):
        flag = Int32()

        new_flag = self.waypoints[self.closest_idx][2]

        if new_flag != self.flag:
            self.get_logger().info(f'FLAG CHANGE: {self.flag} -> {new_flag}')
            self.paplay_sound()

        self.flag = new_flag
        flag.data = self.flag

        self.waypoints_flag_pub.publish(flag)

    def paplay_sound(self):
        try:
            subprocess.Popen(
                [
                    'paplay', '/usr/share/sounds/freedesktop/stereo/bell.oga'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            self.get_logger().error(f'Failed to play sound: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt')

    finally:
        node.steer_msg.steer = 0.0
        node.steer_msg.is_ok = False
        node.gps_steer_publisher.publish(node.steer_msg)
        time.sleep(0.1)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()