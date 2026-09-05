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

        # 카메라 열기 (0, 1, 2, 3 등)
        # 기본 해상도 640*480
        camera_id = 2
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.get_logger().error(f'[CAMERA] Failed to open /dev/video{camera_id}')
            raise RuntimeError("Camera open failed")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.get_logger().info(f"[CAMERA] Resolution: {width} x {height}")
        #self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 자동 노출 - 빛번짐 효과 관련
        #self.cap.set(cv2.CAP_PROP_EXPOSURE, -1)     # 노출 설정 - 빛번짐 효과 관련

        self.process_rate = 30.0    # 제어루프 주기

        self.timer = self.create_timer(1.0 / self.process_rate, self.timer_callback)

    def timer_callback(self):

        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn('[CAMERA] Failed to capture frame')
            return

        image_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')

        self.camera_pub.publish(image_msg)        

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