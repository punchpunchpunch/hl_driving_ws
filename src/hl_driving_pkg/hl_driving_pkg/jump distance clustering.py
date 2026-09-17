#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import numpy as np
import math


class JumpDistanceClustering(Node):

    def __init__(self):
        super().__init__('jump_distance_clustering')

        # =========================
        # Parameters
        # =========================
        self.jump_threshold = 0.15   # [m]
        self.min_points = 3          # 최소 클러스터 점 개수
        self.max_points = 50         # 너무 큰 클러스터 제거

        # 관심 거리
        self.min_range = 0.10
        self.max_range = 8.0

        # =========================
        # Subscriber
        # =========================
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # =========================
        # Publisher
        # =========================
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/jump_clusters',
            10
        )

        self.get_logger().info(
            'Jump Distance Clustering started'
        )

    def scan_callback(self, msg):

        points = []

        angle = msg.angle_min

        # =====================================
        # LaserScan → XY point
        # =====================================
        for r in msg.ranges:

            if (
                math.isfinite(r)
                and self.min_range < r < self.max_range
            ):
                x = r * math.cos(angle)
                y = r * math.sin(angle)

                points.append([x, y])

            else:
                # invalid point
                points.append(None)

            angle += msg.angle_increment

        # =====================================
        # Jump Distance Clustering
        # =====================================
        clusters = []
        current_cluster = []

        previous_point = None

        for point in points:

            # ---------------------------------
            # Invalid point
            # ---------------------------------
            if point is None:

                if current_cluster:
                    clusters.append(current_cluster)

                current_cluster = []
                previous_point = None

                continue

            # ---------------------------------
            # 첫 번째 point
            # ---------------------------------
            if previous_point is None:

                current_cluster = [point]
                previous_point = point

                continue

            # ---------------------------------
            # Jump distance 계산
            # ---------------------------------
            dx = point[0] - previous_point[0]
            dy = point[1] - previous_point[1]

            distance = math.sqrt(
                dx * dx + dy * dy
            )

            # ---------------------------------
            # Jump 발생
            # ---------------------------------
            if distance > self.jump_threshold:

                if current_cluster:
                    clusters.append(current_cluster)

                current_cluster = [point]

            else:

                current_cluster.append(point)

            previous_point = point

        # 마지막 cluster
        if current_cluster:
            clusters.append(current_cluster)

        # =====================================
        # Cluster filtering
        # =====================================
        filtered_clusters = []

        for cluster in clusters:

            size = len(cluster)

            if (
                self.min_points
                <= size
                <= self.max_points
            ):
                filtered_clusters.append(cluster)

        # =====================================
        # RViz visualization
        # =====================================
        self.publish_markers(
            filtered_clusters,
            'laser'
        )

    def publish_markers(self, clusters, frame_id):

        marker_array = MarkerArray()

        # 이전 marker 삭제
        delete_marker = Marker()
        delete_marker.header.frame_id = frame_id
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.ns = 'clusters'
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL

        marker_array.markers.append(delete_marker)

        # =====================================
        # 클러스터별 색상
        # =====================================
        colors = [
            (1.0, 0.0, 0.0),  # 빨강
            (0.0, 1.0, 0.0),  # 초록
            (0.0, 0.5, 1.0),  # 파랑
            (1.0, 0.0, 1.0),  # 보라
            (0.0, 1.0, 1.0),  # 하늘
            (1.0, 0.5, 0.0),  # 주황
            (1.0, 1.0, 0.0),  # 노랑
            (0.5, 0.0, 1.0),  # 보라색
        ]

        # =====================================
        # Cluster마다 Marker 생성
        # =====================================
        for i, cluster in enumerate(clusters):

            marker = Marker()

            marker.header.frame_id = frame_id
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.ns = 'clusters'
            marker.id = i + 1

            marker.type = Marker.POINTS
            marker.action = Marker.ADD

            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.06
            marker.scale.y = 0.06

            # ---------------------------------
            # 클러스터별 색상
            # ---------------------------------
            r, g, b = colors[i % len(colors)]

            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1.0

            for x, y in cluster:

                p = Point()
                p.x = x
                p.y = y
                p.z = 0.0

                marker.points.append(p)

            marker_array.markers.append(marker)

            # =================================
            # Cluster centroid
            # =================================
            cx = np.mean(
                [p[0] for p in cluster]
            )

            cy = np.mean(
                [p[1] for p in cluster]
            )

            text_marker = Marker()

            text_marker.header.frame_id = frame_id
            text_marker.header.stamp = \
                self.get_clock().now().to_msg()

            text_marker.ns = 'cluster_text'
            text_marker.id = 1000 + i

            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            # ---------------------------------
            # 텍스트도 해당 클러스터와 같은 색
            # ---------------------------------
            text_marker.color.r = r
            text_marker.color.g = g
            text_marker.color.b = b
            text_marker.color.a = 1.0

            text_marker.pose.position.x = cx
            text_marker.pose.position.y = cy
            text_marker.pose.position.z = 0.15

            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = 0.15

            text_marker.text = (
                f'{i}: {len(cluster)} pts'
            )

            marker_array.markers.append(text_marker)

        self.marker_pub.publish(marker_array)


def main(args=None):

    rclpy.init(args=args)

    node = JumpDistanceClustering()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()