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

        self.frame = None

        self.model = YOLO('best.pt')
        self.process_rate = 30.0    # 프로세스 루프 주기
        self.conf_threshold = 0.25

        self.timer = self.create_timer(1.0 / self.process_rate, self.timer_callback)

        self.prev_time = time.time()
    
        self.get_logger().info('YOLO Node started')

    def image_callback(self, image_msg):
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # 하위 1/2 제거
        height = int(frame.shape[0])
        self.frame = frame[:height//2, :]


    def timer_callback(self):
        if self.frame is None:
            return
        
        results = self.model(self.frame, conf=self.conf_threshold, verbose=False)

        display_frame = self.frame.copy()

        detection_count = 0

        for result in results:
            for box in result.boxes:
                detection_count += 1

                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                label = result.names[cls]

                # bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                # =====================
                # Bounding Box
                # =====================
                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # =====================
                # Label
                # =====================
                text = f'{label} {confidence:.2f}'

                cv2.putText(
                    display_frame,
                    text,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # =====================
                # Center point
                # =====================
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                cv2.circle(
                    display_frame,
                    (center_x, center_y),
                    4,
                    (0, 0, 255),
                    -1
                )

                # 터미널 출력
                self.get_logger().info(
                    f'Detected: {label}, '
                    f'confidence={confidence:.2f}, '
                    f'bbox=({x1}, {y1}, {x2}, {y2})'
                )

                # =================================================
                # TrafficLight / TrafficSign 처리
                # =================================================
                self.process_detection(
                    label,
                    confidence,
                    x1,
                    y1,
                    x2,
                    y2
                )

        # =====================
        # FPS
        # =====================
        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time)
        self.prev_time = current_time

        cv2.putText(
            display_frame,
            f'FPS: {fps:.1f}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f'Detections: {detection_count}',
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # =====================
        # Publish detection image
        # =====================
        try:

            detection_image_msg = self.bridge.cv2_to_imgmsg(
                display_frame,
                encoding='bgr8'
            )

            detection_image_msg.header.stamp = self.get_clock().now().to_msg()

            self.object_detection_image_publisher.publish(
                detection_image_msg
            )

        except Exception as e:

            self.get_logger().error(
                f'Image publish error: {e}'
            )

    # =========================================================
    # Detection processing
    # =========================================================
    def process_detection(self, label, confidence):

        """
        YOLO에서 검출된 객체를
        TrafficLight / TrafficSign으로 분류하는 부분.

        실제 TrafficLightMsg / TrafficSignMsg의
        필드명에 맞춰 수정해야 함.
        """

        # -----------------------------------------------------
        # Traffic Light
        # -----------------------------------------------------
        traffic_light_labels = [
            'red',
            'yellow',
            'green',
            'left',
            'right',
            'straight'
        ]

        if label.lower() in traffic_light_labels:

            self.get_logger().info(
                f'Traffic Light detected: {label}'
            )

            # -------------------------------------------------
            # TrafficLightMsg 필드에 맞춰 작성
            # -------------------------------------------------
            msg = TrafficLightMsg()

            # 예시:
            #
            # msg.label = label
            # msg.confidence = confidence
            #
            # 실제 메시지 정의 확인 후 사용해야 함.

            self.traffic_light_publisher.publish(msg)

        # -----------------------------------------------------
        # Traffic Sign
        # -----------------------------------------------------
        else:

            self.get_logger().info(
                f'Traffic Sign detected: {label}'
            )

            # -------------------------------------------------
            # TrafficSignMsg 필드에 맞춰 작성
            # -------------------------------------------------
            msg = TrafficSignMsg()

            # 예시:
            #
            # msg.label = label
            # msg.confidence = confidence
            #
            # 실제 메시지 정의 확인 후 사용해야 함.

            self.traffic_sign_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisionAI()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt')

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()