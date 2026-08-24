import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from auto_driving_msgs.msg import TrafficLightMsg, TrafficSignMsg

from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

import cv2
import numpy as np
import time
from ultralytics import YOLO

class VisionAI(Node):

    def __init__(self):
        super().__init__('vision_ai')

        # =====================
        # subscriptions & publishers
        # =====================
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )
        
        self.object_detection_image_publisher = self.create_publisher(
            Image, 'object_detection_image', 1
        )

        self.traffic_light_publisher = self.create_publisher(
            TrafficLightMsg, 'traffic_light_result', 10
        )

        self.traffic_sign_publisher = self.create_publisher(
            TrafficSignMsg, 'traffic_sign_result', 10
        )

        self.bridge = CvBridge()
        
        self.steer_msg = SteerMsg()
        self.steer_msg.steer = 0.0
        self.steer_msg.is_ok = False

        self.real_lane = False

        self.sensitivity = 0.2
        
        self.get_logger().info('hough transform 차선 인식 시작')


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











from ultralytics import YOLO

self.model = YOLO("yolo11n.pt")

results = self.model(frame, conf=0.25, verbose=False)


for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        label = result.names[cls]