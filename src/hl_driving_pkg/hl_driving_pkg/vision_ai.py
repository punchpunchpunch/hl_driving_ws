import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from auto_driving_msgs.msg import TrafficLightMsg

from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

import cv2
import os

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
            TrafficLightMsg, 'traffic_light_result', 1
        )

        self.selected_lane_publisher = self.create_publisher(
            Int32, '/selected_lane', 1
        )

        self.bridge = CvBridge()

        self.frame = None

        #self.model = YOLO('/home/yeong/runs/detect/train/weights/best.pt')

        model_path = os.path.expanduser('~/hl_driving_ws/src/hl_driving_pkg/hl_driving_pkg/hl_model/best.pt')  # 09/18 버전

        self.model = YOLO(model_path)

        self.process_rate = 30.0    # 프로세스 루프 주기
        self.conf_threshold = 0.50

        self.traffic_light_labels = {
            'green',
            'left',
            'stop',
            'yellow'
        }

        self.lane_sign_labels = {
            'go_line',
            'no_line'
        }

        self.selected_lane = 0

        # 최근 프레임에서 검출된 차로 결과
        self.lane_history = []

        # 몇 번 연속 같은 결과가 나와야 확정할지
        self.lane_confirm_count = 5

        # 한번 차로를 선택하면 고정
        self.lane_locked = False

        self.timer = self.create_timer(1.0 / self.process_rate, self.timer_callback)
    
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
        lane_detections = []

        for result in results:
            for box in result.boxes:
                detection_count += 1

                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                label = result.names[cls].lower()

                # bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                if label in self.lane_sign_labels:

                    lane_detections.append({
                        'label': label,
                        'confidence': confidence,
                        'x1': x1
                    })

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

                # 터미널 출력
                self.get_logger().info(
                    f'Detected: {label}, '
                    f'confidence={confidence:.2f}, '
                    f'bbox=({x1}, {y1}, {x2}, {y2})'
                )

                # =================================================
                # TrafficLight / TrafficSign 처리
                # =================================================
                self.process_traffic_light(
                    label,
                    confidence,
                )

        self.process_lane_control(
            lane_detections,
            display_frame
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
    def process_traffic_light(self, label, confidence):

        if label.lower() in self.traffic_light_labels:

            self.get_logger().info(
                f'Traffic Light detected: {label}'
            )

            msg = TrafficLightMsg()

            msg.label = label
            msg.confidence = confidence

            self.traffic_light_publisher.publish(msg)

    def process_lane_control(self, lane_detections, display_frame):

        if self.lane_locked:
            return

        # 3개 LCS 모두 검출되어야 판단
        if len(lane_detections) != 3:
            return

        # 왼쪽 -> 오른쪽 정렬
        lane_detections.sort(key=lambda d: d['x1'])

        lcs1 = lane_detections[0]
        lcs2 = lane_detections[1]

        if lcs1['label'] == 'go_line':
            candidate_lane = 1

        elif lcs2['label'] == 'go_line':
            candidate_lane = 2

        # 첫째 둘째 둘 다 no_line
        else:
            return

        # ---------------------------------
        # 5 프레임 안정화
        # ---------------------------------
        self.lane_history.append(candidate_lane)

        if len(self.lane_history) > 5:
            self.lane_history.pop(0)

        if len(self.lane_history) < 5:
            return

        # 5프레임 모두 같은 차로
        if all(
            lane == candidate_lane
            for lane in self.lane_history
        ):

            self.selected_lane = candidate_lane
            self.lane_locked = True

            msg = Int32()
            msg.data = self.selected_lane

            self.selected_lane_publisher.publish(msg)

            self.get_logger().info(
                f'[LANE SELECTED] {self.selected_lane}차로'
            )

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