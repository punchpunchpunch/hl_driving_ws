import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32
from auto_driving_msgs.msg import SteerMsg
from ublox_msgs.msg import NavPVT
from visualization_msgs.msg import Marker

import serial
import time

class CommCar(Node):
    def __init__(self):
        super().__init__('comm_car')
        self.gps_steer = 0.0
        self.lane_steer = 0.0
        self.side_steer = 0.0
        
        self.gps_speed = 700        # 차량 속도(아두이노 코드에도 속도 제한 있음)
        self.lane_speed = 700
        self.side_speed = 300
        self.fast_speed = 1000
        self.slow_speed = 300
        self.back_speed = 300
        self.stop_speed = 0
        self.stop_time = 4.0
        
        self.gps_ok = False
        self.lane_ok = False
        self.side_ok = False
        self.slow_ok = False
        self.center_ok = False

        self.MAX_STEER_DEG = 20     # 차량 하드웨어가 할 수 있는 최대 조향각 (일단 랩뷰에서 설정한 20도로 맞춤)
        self.MAX_HST = 2000         # 아두이노 코드 상으로는 2000이 맞는듯
        self.control_rate = 30.0    # 제어루프 주기

        self.rtk_status = "No"
        self.num_sv = 0
        self.h_acc = 0.0

        self.flag = 0

        self.stop_start_time = None

        # =====================
        # car serial
        # =====================
        try:
            self.ser = serial.Serial(
                port='/dev/tty_powerpack',  # 차량 시리얼포트
                baudrate=115200,            # 차량 보레이트
                timeout=0.1                 # 차량 타임아웃
            )
        except Exception as e:
            self.get_logger().error(f'시리얼 포트 연결 실패: {e}')
            self.ser = None

        # =====================
        # subscriptions
        # =====================
        self.gps_steer_subscription = self.create_subscription(
            SteerMsg,
            'gps_steer',
            self.gps_steer_callback,
            10 # 큐 크기
        )

        self.lane_steer_subscription = self.create_subscription(
            SteerMsg,
            'lane_steer',
            self.lane_steer_callback,
            10 # 큐 크기
        )

        self.lidar_side_steer_subscription = self.create_subscription(
            SteerMsg,
            'lidar_side_steer',
            self.lidar_side_steer_callback,
            10 # 큐 크기
        )

        self.lidar_slow_steer_subscription = self.create_subscription(
            SteerMsg,
            'lidar_slow_steer',
            self.lidar_slow_steer_callback,
            10 # 큐 크기
        )

        self.lidar_center_steer_subscription = self.create_subscription(
            SteerMsg,
            'lidar_center_steer',
            self.lidar_center_steer_callback,
            10 # 큐 크기
        )

        self.rtk_subscription = self.create_subscription(
            NavPVT,
            '/ublox_gps_node/navpvt', 
            self.rtk_callback,
            10
        )

        self.flag_subscription = self.create_subscription(
            Int32,
            '/waypoints_flag',
            self.flag_callback,
            10
        )

        # =====================
        # publishers (RViz)
        # =====================
        self.marker_pub = self.create_publisher(
            Marker, '/comm_car_marker', 10
        )

        # timer - loop period
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)
        self.get_logger().info('차량 통신 시작')

    # =====================================================
    # Callbacks
    # =====================================================
    def gps_steer_callback(self, msg: SteerMsg):
        self.gps_steer = msg.steer
        self.gps_ok = msg.is_ok
        
        
    def lane_steer_callback(self, msg: SteerMsg):
        self.lane_steer = msg.steer
        self.lane_ok = msg.is_ok

    def lidar_side_steer_callback(self, msg: SteerMsg):
        self.side_steer = msg.steer
        self.side_ok = msg.is_ok

    def lidar_slow_steer_callback(self, msg: SteerMsg):
        self.slow_ok = msg.is_ok

    def lidar_center_steer_callback(self, msg: SteerMsg):
        self.center_ok = msg.is_ok

    # =====================================================
    # Control Loop
    # =====================================================
    def control_loop(self):

        # 조향
        if self.side_ok and self.flag == 7: #lidar_side
            steer = self.side_steer
            speed = self.side_speed
            mode = 'side'
        elif self.gps_ok:   #gps
            if self.flag == 1:
                steer = -self.gps_steer
            else: 
                steer = self.gps_steer
            speed = self.gps_speed
            mode = 'gps'
        elif self.lane_ok and self.flag == 12: #lane
            steer = self.lane_steer
            speed = self.lane_speed
            mode = 'lane'
        else:
            if self.flag == 1:
                steer = -self.gps_steer
            else: 
                steer = self.gps_steer
            speed = self.gps_speed
            mode = 'gps'

        # 속도
        if self.flag == 2:   # 경사로 정지
                    if self.stop_start_time is None:
                        self.stop_start_time = time.time()
                    if time.time() - self.stop_start_time < self.stop_time:
                        speed = self.stop_speed
                        mode = mode + ' hill_stop'
                    else:
                        speed = self.gps_speed
                        mode = mode + ' hill_go'
        elif self.flag == 10:  # 주차 정지
            if self.stop_start_time is None:
                self.stop_start_time = time.time()
            if time.time() - self.stop_start_time < self.stop_time:
                speed = self.stop_speed
                mode = mode + ' parking_stop'
            else:
                speed = self.gps_speed
                mode = mode + ' parking_go'
        elif self.center_ok and self.flag == 4: # 긴급제동 정지
            speed = self.stop_speed
            mode = mode + ' center'
        elif self.slow_ok and self.flag == 4:
            speed = self.slow_speed
            mode = mode + ' slow'
        elif self.flag == 1: # 후진
            speed = self.back_speed
            mode = mode + ' back'
        elif self.flag == 5: # 속도 느리게
            speed = self.slow_speed
            mode = mode + ' slow'
        elif self.flag == 4: # 가속
            speed = self.fast_speed
            mode = mode + ' fast'
        elif self.flag == 6: # 속도 빠르게
            speed = self.fast_speed
            mode = mode + ' fast'
        elif self.slow_ok and self.flag == 7:
            speed = self.slow_speed
            mode = mode + ' slow'
    
        hst = self.steer_to_hst(steer)

        # 후진
        if self.flag == 1:
            hge = 2
        else:
            hge = 0

        self.send_frame(speed, hst, hge)
        self.publish_text_marker(steer, speed, mode)
        print(steer, speed)

    def destroy_node(self):
        try:
            self.send_frame(0, 0)
        except Exception:
            pass
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        super().destroy_node()

    def steer_to_hst(self, steer_deg):
        # rate limiter 추가 고려
        hst = int(
            max(-self.MAX_HST, min(self.MAX_HST, steer_deg / self.MAX_STEER_DEG * self.MAX_HST))
        )
        return hst

        '''
        hst = int(steer_rad * 100)

        if hst > 2000:
            hst = 2000
        elif hst < -2000:
            hst = -2000

        return(hst)
        '''
        
    def send_frame(self, hsp, hst, hge=0):
        """
        Arduino로 13바이트 프레임 송신
        """
        if self.ser is None:
            return

        Ham = 1      # ROS 제어 활성
        Hes = 0      # ESTOP 해제
        Hal = 0

        frame = bytearray(13)
        frame[0:3] = b'STX'
        frame[3] = Ham
        frame[4] = Hes
        frame[5] = hge      # 전진 0, 후진 2

        frame[6] = (hsp >> 8) & 0xFF
        frame[7] = hsp & 0xFF

        frame[8] = (hst >> 8) & 0xFF
        frame[9] = hst & 0xFF

        frame[10] = Hal
        frame[11] = 0x0D
        frame[12] = 0x0A

        self.ser.write(frame)

    def rtk_callback(self, msg: NavPVT):
        # flags 필드에서 Carrier Phase Status 추출 (192 = 0b11000000)
        self.rtk_status = msg.flags & 192
        '''
        0: "❌ No RTK (Single/DGPS)",
        64: "🟡 RTK Float (Searching...)",
        128: "🟢 RTK Fixed (Perfect!)"
        '''
        self.num_sv = msg.num_sv
        self.h_acc = msg.h_acc / 1000.0 # mm -> m 단위 변환

    def flag_callback(self, msg: Int32):
        if self.flag != msg.data:
            self.stop_start_time = None
        self.flag = msg.data

    def publish_text_marker(self, steer, speed, mode):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = 3.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.0
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.9

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.text = (
            f"Steer : {steer:.2f}\n"
            f"Speed : {speed:.2f}\n"
            f"Mode : {mode}\n"
            f"RTK          : {self.rtk_status}\n"
            f"Satellites        : {self.num_sv}\n"        # Number of SVs used in Nav Solution
            f"Horizontal Accuracy : {self.h_acc:.3f} m"     # Horizontal Accuracy Estimate [mm]
        )

        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = CommCar()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt')

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()