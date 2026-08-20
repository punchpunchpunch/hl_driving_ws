import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from auto_driving_msgs.msg import SteerMsg

from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class LaneDetector(Node):
    def __init__(self):
        super().__init__('lane_detector')

        # =====================
        # subscriptions & publishers
        # =====================
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )
        
        self.lane_image_publisher = self.create_publisher(
            Image, 'lane_image', 1
        )

        self.lane_steer_publisher = self.create_publisher(
            SteerMsg, 'lane_steer', 10
        )

        self.bridge = CvBridge()
        
        self.steer_msg = SteerMsg()
        self.steer_msg.steer = 0.0
        self.steer_msg.is_ok = False

        self.real_lane = False

        self.sensitivity = 0.2
        
        self.get_logger().info('hough transform 차선 인식 시작')

    def image_callback(self, msg):
        # --- 차선 인식 알고리즘 ---
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # 상위 1/2 제거
        height = int(frame.shape[0])
        frame = frame[height//2:, :]

        # HLS 변환 h는 색상 l은 밝기 s는 채도. 그림자 졌을 때 밝기 180까지 해야함
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        # 현재 화면의 평균 밝기 계산
        l_channel = hls[:, :, 1]
        avg_brightness = np.mean(l_channel)
        #print(avg_brightness)

        # 평균 밝기에 따라 하한선 결정. 최솟값 150. 현재 + 30
        dynamic_lower_l = int(max(avg_brightness + 30, 150)) 
        #print(dynamic_lower_l)

        lower_white = np.array([0, dynamic_lower_l, 120])
        upper_white = np.array([179, 255, 255])
        '''
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([179, 255, 100])
        '''

        # 색상 마스크
        mask = cv2.inRange(hls, lower_white, upper_white)

        # 마스크 적용
        #frame = cv2.bitwise_and(frame, frame, mask=mask)

        # 전처리 (그레이스케일 변환 및 블러)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # 엣지 검출 (Canny Edge) 80 150
        canny = cv2.Canny(blur, 80, 180)
        cv2.imshow("Edge Detection", canny)

        # Hough Transform으로 직선 검출 (cv2.HoughLinesP 사용)
        rho = 1
        theta = np.pi/180
        threshold = 30
        lines = cv2.HoughLinesP(canny, rho, theta, threshold, minLineLength=30, maxLineGap=10)

        target_center = frame.shape[1] // 2
        mid_point = frame.shape[1] / 2

        left_x_list = []
        right_x_list = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0

                if y1 > y2:
                    x_near, y_near = x1, y1
                    x_far, y_far = x2, y2
                else:
                    x_near, y_near = x2, y2
                    x_far, y_far = x1, y1
    
                # 차량과 가까운 아래쪽 x좌표 위주로 수집 (y가 큰 값)
                if slope < -0.1 and x_near < (frame.shape[1] // 2) * 1.1:
                    left_x_list.append(x1)
                    left_x_list.append(x2)
                elif slope > 0.1 and x_near > (frame.shape[1] // 2) * 0.9:
                    right_x_list.append(x1)
                    right_x_list.append(x2)

            # 조향을 위한 target_center 결정
            if left_x_list and right_x_list:
                # 양쪽 다 있을 땐 순수 평균
                target_center = (np.mean(left_x_list) + np.mean(right_x_list)) / 2
                self.real_lane = True
            elif left_x_list:
                # 왼쪽만 보이면 '이전 프레임의 평균 차선 폭'을 쓰거나 현재는 고정값 유지
                target_center = np.mean(left_x_list) + 200 # 150이 너무 작을 수 있음
                self.real_lane = True
            elif right_x_list:
                target_center = np.mean(right_x_list) - 200
                self.real_lane = True
            else:
                target_center = mid_point # 아무것도 안 보이면 직진
                self.real_lane = False

        # 에러 계산 및 조향
        # 감도 설정: 너무 낮으면 반응이 느리고 높으면 흔들림
        error = (target_center - mid_point) * self.sensitivity
        self.steer_msg.steer = error
        #self.steer_msg.is_ok = True if lines else False
        #self.steer_msg.is_ok = bool(lines)
        self.steer_msg.is_ok = (lines is not None) and self.real_lane
        self.lane_steer_publisher.publish(self.steer_msg)
        
        # 5. 시각화 (중앙점 표시)
        cv2.line(frame, (int(target_center), 0), (int(target_center), frame.shape[0]), (255, 255, 0), 3)

        # 5. 결과 그리기
        line_frame = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)        
        result = cv2.addWeighted(frame, 0.8, line_frame, 1.0, 0.0)

        # 6. ROS 2 메시지로 변환하여 발행
        self.lane_image_publisher.publish(self.bridge.cv2_to_imgmsg(result, 'bgr8'))
        
        # 로컬 화면 확인 (선택 사항)
        #cv2.imshow('Lane Detection', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.destroy_node()

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()
        
def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt')

    finally:
        node.steer_msg.steer = 0.0
        node.steer_msg.is_ok = False
        node.lane_steer_publisher.publish(node.steer_msg)
        time.sleep(0.1)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()