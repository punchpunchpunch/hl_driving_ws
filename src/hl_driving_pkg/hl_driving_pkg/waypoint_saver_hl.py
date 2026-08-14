import rclpy
from rclpy.node import Node

from sensor_msgs.msg import NavSatFix
from ublox_msgs.msg import NavPVT

import os
import sys

from datetime import datetime
from pyproj import CRS, Transformer
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtCore import QTimer

class WaypointSaver(Node):

    def __init__(self):
        super().__init__('waypoint_saver')

        self.interval = 0.2 # meter
        self.prev_x = None  # UTM X(E)
        self.prev_y = None  # UTM Y(N)
        self.flag = 0
        self.waypoint_count = 0
        self.rtk_status = "No"
        self.num_sv = 0
        self.h_acc = 0.0

        # --------------------
        # CSV file
        # --------------------
        self.file_dir = os.path.expanduser('~/hl_driving_ws/src/hl_driving_pkg/hl_driving_pkg/waypoints')
        os.makedirs(self.file_dir, exist_ok=True)

        now = datetime.now()
        self.file_name = now.strftime('waypoints_%Y%m%d_%H%M%S.csv')

        self.csv_path = os.path.join(self.file_dir, self.file_name)
        self.ofs = open(self.csv_path, 'a')
        self.ofs.write('utm_x,utm_y,flag\n')

        # --------------------
        # UTM transformer
        # --------------------
        self.epsg = 32652
        self.transformer = Transformer.from_crs(
            CRS.from_epsg(4326),        # WGS84
            CRS.from_epsg(self.epsg),   # UTM(한국 기준)
            always_xy=True
        )

        # --------------------
        # subscriptions
        # --------------------
        self.gps_subscription = self.create_subscription(
            NavSatFix,
            '/ublox_gps_node/fix',
            self.gps_callback,
            10
        )

        self.rtk_subscription = self.create_subscription(
            NavPVT,
            '/ublox_gps_node/navpvt', 
            self.rtk_callback,
            10
        )

        self.get_logger().info(f'Waypoint Saver started (EPSG:{self.epsg})')

    def gps_callback(self, msg: NavSatFix):
        if msg.status.status < 0:
            return

        # lon, lat → UTM
        utm_x, utm_y = self.transformer.transform(
            msg.longitude,
            msg.latitude
        )

        if self.prev_x is None:
            self.prev_x = utm_x
            self.prev_y = utm_y
            return

        dx = utm_x - self.prev_x
        dy = utm_y - self.prev_y
        distance_sq = dx * dx + dy * dy

        if distance_sq < self.interval * self.interval:
            return

        self.get_logger().info(
            f'UTM: {utm_x:.6f}, {utm_y:.6f}, flag={self.flag}'
        )

        self.ofs.write(
            f'{utm_x:.6f},{utm_y:.6f},{self.flag}\n'
        )
        self.ofs.flush()

        self.waypoint_count += 1

        self.prev_x = utm_x
        self.prev_y = utm_y

    def rtk_callback(self, msg: NavPVT):
        # flags 필드에서 Carrier Phase Status 추출 (192 = 0b11000000)
        carr_soln = msg.flags & 192

        if carr_soln == 0:
            self.rtk_status = "No"
        elif carr_soln == 64:
            self.rtk_status = "Float"
        elif carr_soln == 128:
            self.rtk_status = "Fixed"
        else:
            self.rtk_status = "Unknown"

        self.num_sv = msg.num_sv
        self.h_acc = msg.h_acc / 1000.0 # mm -> m 단위 변환

    def set_flag(self, flag):
        self.flag = flag

class UI(QWidget):

    def __init__(self, node):
        super().__init__()

        self.node = node

        self.setWindowTitle("Waypoint Saver")
        self.resize(400, 1000)
        font = self.font()
        font.setPointSize(20)
        self.setFont(font)

        layout = QVBoxLayout()

        # 상태 표시
        self.rtk_label = QLabel("RTK Status: No")
        self.sv_label = QLabel("Satellites: 0")
        self.hacc_label = QLabel("Horizontal Accuracy: 0.00 m")
        self.flag_label = QLabel("Current Flag: 0")
        self.count_label = QLabel("Waypoint Count: 0")
        layout.addWidget(self.rtk_label)
        layout.addWidget(self.sv_label)
        layout.addWidget(self.hacc_label)
        layout.addWidget(self.flag_label)
        layout.addWidget(self.count_label)

        # flag 버튼
        for i in range(13):
            btn = QPushButton(f"Flag {i}")
            btn.setMinimumHeight(50)
            btn.clicked.connect(lambda _, flag=i: node.set_flag(flag))
            layout.addWidget(btn)

        # 종료 버튼
        btn_exit = QPushButton("Exit")
        btn_exit.setMinimumHeight(50)
        btn_exit.clicked.connect(QApplication.quit)
        layout.addWidget(btn_exit)

        self.setLayout(layout)

        # UI 업데이트 타이머
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_status)
        self.ui_timer.start(200)

    def update_status(self):
        self.rtk_label.setText(f"RTK Status: {self.node.rtk_status}")
        self.sv_label.setText(f"Satellites: {self.node.num_sv}")
        self.hacc_label.setText(f"Horizontal Accuracy: {self.node.h_acc:.3f} m")
        self.flag_label.setText(f"Current Flag: {self.node.flag}")
        self.count_label.setText(f"Waypoint Count: {self.node.waypoint_count}")
    
def main(args=None):
    rclpy.init(args=args)
    node = WaypointSaver()

    app = QApplication(sys.argv)
    ui = UI(node)
    ui.show()

    # ROS spin timer
    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0))
    timer.start(10)

    app.exec_()

    node.get_logger().info(f'~~~~~~ CSV file path: {node.csv_path}')
    node.ofs.close()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()