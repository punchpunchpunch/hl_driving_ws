import rclpy
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data

class CameraPublisher(Node):

    def __init__(self):
        super().__init__('camera_publisher')

        self.camera_pub = self.create_publisher(
            Image, '/camera/image_raw', qos_profile_sensor_data
        )
        
        self.bridge = CvBridge()

        # 카메라 열기 (0은 기본 웹캠, 1, 2!!, 3)
        # 기본 해상도는 640*480
        camera_id = 2
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(f'카메라를 열 수 없습니다! (/dev/video{camera_id})')
            raise RuntimeError("Camera open failed")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        #width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        #height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #self.get_logger().info(f"Camera resolution: {width} x {height}")
        #self.get_logger().info(f"Camera FPS: {fps}")
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        #self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 수동 모드 - 빛번짐 효과 없음
        #self.cap.set(cv2.CAP_PROP_EXPOSURE, -1)     # 노출을 낮춰서 어둡게 찍히도록 함 - 빛번짐 효과 없음

        self.control_rate = 30.0    # 제어루프 주기

        self.timer = self.create_timer(1.0 / self.control_rate, self.timer_callback)

    def timer_callback(self):

        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn('프레임을 가져오지 못했습니다.')
            return

        self.camera_pub.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt')

    finally:
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()