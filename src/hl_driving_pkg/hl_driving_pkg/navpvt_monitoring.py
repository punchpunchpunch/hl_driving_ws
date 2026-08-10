#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ublox_msgs.msg import NavPVT  # ublox_msgs 설치 필요

class HeadingMonitor(Node):
    def __init__(self):
        super().__init__('heading_monitor')
        # /ublox_gps_node/navpvt 토픽 구독
        self.subscription = self.create_subscription(
            NavPVT,
            '/ublox_gps_node/navpvt',
            self.navpvt_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def navpvt_callback(self, msg: NavPVT):
        if msg.fix_type < 2:  # fix 없으면 무시
            self.get_logger().info("No GPS fix yet.")
            return

        # heading: 1e-5 deg 단위 → deg 변환
        heading_deg = msg.heading * 1e-5

        # head_veh: vehicle heading도 필요하면 동일 방식
        head_veh_deg = msg.head_veh * 1e-5

        self.get_logger().info(
            f"Heading (Course over Ground): {heading_deg:.2f} deg, "
            f"Vehicle Heading: {head_veh_deg:.2f} deg, "
            f"Speed: {msg.g_speed:.2f} cm/s, "
            f"Satellites: {msg.num_sv}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = HeadingMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()