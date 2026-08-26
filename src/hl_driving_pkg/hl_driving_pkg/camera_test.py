import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import time
from rclpy.qos import qos_profile_sensor_data


class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_test')

        self.count = 0
        self.last_time = time.perf_counter()

        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.callback,
            qos_profile_sensor_data
        )

    def callback(self, msg):
        self.count += 1

        now = time.perf_counter()

        if now - self.last_time >= 1.0:
            print(f"Received FPS: {self.count}")
            self.count = 0
            self.last_time = now


def main():
    rclpy.init()
    node = ImageSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()