import rclpy
from rclpy.node import Node
# UBX NAV-PVT 메시지 타입 (사용하시는 패키지명에 따라 import 경로를 수정하세요)
# 보통 'ublox_msgs.msg'를 많이 사용합니다.
from ublox_msgs.msg import NavPVT 

class RtkStatusChecker(Node):
    def __init__(self):
        super().__init__('rtk_status_checker')
        
        # NAV-PVT 토픽 구독 (일반적으로 /ublox/navpvt)
        self.subscription = self.create_subscription(
            NavPVT,
            '/ublox_gps_node/navpvt', 
            self.listener_callback,
            10)
        
        # 상태 매핑 테이블
        self.rtk_modes = {
            0: "❌ No RTK (Single/DGPS)",
            64: "🟡 RTK Float (Searching...)",
            128: "🟢 RTK Fixed (Perfect!)"
        }

    def listener_callback(self, msg):
        # flags 필드에서 Carrier Phase Status 추출 (192 = 0b11000000)
        rtk_status = msg.flags & 192
        
        status_text = self.rtk_modes.get(rtk_status, "Unknown")
        num_sv = msg.num_sv
        h_acc = msg.h_acc / 1000.0 # mm -> m 단위 변환

        # 터미널 출력
        print(f"--- GNSS Status ---")
        print(f"Mode: {status_text}")
        print(f"Satellites: {num_sv}") # Number of SVs used in Nav Solution
        print(f"Horizontal Accuracy: {h_acc:.3f} m") # Horizontal Accuracy Estimate [mm]
        
        # 정밀도가 중요한 작업 시 로직 처리
        if rtk_status == 128:
            # 주행 시작 가능
            pass
        else:
            # 주의 알림 또는 정지
            pass

def main(args=None):
    rclpy.init(args=args)
    node = RtkStatusChecker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()