from yolov5_ros2.yolov5_shim import YOLOv5
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from std_msgs.msg import Bool, Int16
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time

# 订阅 /mavros/altitude 的消息类型
from mavros_msgs.msg import Altitude

# 高度缓存
from collections import deque


# Get the ROS distribution version and set the shared directory for YoloV5 configuration files.
ros_distribution = os.environ.get("ROS_DISTRO")
package_share_directory = get_package_share_directory('yolov5_ros2')


# Create a ROS 2 Node class YoloV5Ros2.
class YoloV5Ros2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')
        self.get_logger().info(f"Current ROS 2 distribution: {ros_distribution}")

        # ---------------- 参数声明 ----------------
        self.declare_parameter("device", "cuda", ParameterDescriptor(
            name="device", description="Compute device selection, default: cpu, options: cuda:0"))

        self.declare_parameter("model", "yolov5s", ParameterDescriptor(
            name="model", description="Default model selection: yolov5s"))

        self.declare_parameter("image_topic", "/image_raw", ParameterDescriptor(
            name="image_topic", description="Image topic, default: /image_raw"))

        self.declare_parameter("camera_info_topic", "/camera/camera_info", ParameterDescriptor(
            name="camera_info_topic", description="Camera information topic, default: /camera/camera_info"))

        self.declare_parameter("camera_info_file", "", ParameterDescriptor(
            name="camera_info_file",
            description="Path to camera intrinsics YAML; empty = <share>/config/down_camera_info.yaml"))

        self.declare_parameter("show_result", False, ParameterDescriptor(
            name="show_result", description="Whether to display detection results, default: False"))

        self.declare_parameter("pub_result_img", False, ParameterDescriptor(
            name="pub_result_img", description="Whether to publish detection result images, default: False"))

        # ---------------- 1. 加载模型 ----------------
        model_path = "/home/lin/uav_ws/src/yolov5_ros2/config/best_fp16.engine"
        device = self.get_parameter('device').value
        self.yolov5 = YOLOv5(model_path=model_path, device=device)

        # ---------------- 2. 相机内参（从配置文件读取，便于实机部署） ----------------
        self.camera_info = self.load_camera_info()
        self.camera_info_received = True
        self.print_camera_info()

        # ---------------- 3. 图像格式转换 ----------------
        self.bridge = CvBridge()

        # ---------------- 4. 控制参数 ----------------
        self.show_result = self.get_parameter('show_result').value
        self.pub_result_img = self.get_parameter('pub_result_img').value

        self.active = True
        self.rate = 30
        self.frame_counter = 0
        self.inference_interval = 1

        # ---------------- 5. 时间同步相关（★ 在创建订阅之前初始化） ----------------
        # 高度缓存：存 (本地时钟秒, 高度米) 二元组，按时间递增
        # 使用本地 ROS 时钟做统一基准，避免 mavros 与相机驱动 header.stamp 时间源不一致
        self.height_buffer = deque(maxlen=200)
        self.height_sync_tolerance = 0.2   # 允许查高度的最大时间偏差（秒）
        self.last_height = 0.0
        self.height_received = False
        self.height_offset = 0.6           # 相机-机身垂直偏移（米）

        # ---------------- 6. 发布器 ----------------
        self.yolo_result_pub = self.create_publisher(Detection2DArray, "yolo_result", 10)
        self.result_msg = Detection2DArray()

        self.result_img_pub = self.create_publisher(Image, "result_img", 10)

        self.relative_result_pub = self.create_publisher(
            Detection2DArray, '/down_detections_relative', 10)
        self.relative_result_msg = Detection2DArray()
        self.relative_result_msg.header.frame_id = "camera"

        # ---------------- 7. 订阅器 ----------------
        image_topic = self.get_parameter('image_topic').value
        _img_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, _img_qos)

        self.active_sub = self.create_subscription(
            Bool, '/down_camera/activate', self.active_callback, 10)

        self.rate_sub = self.create_subscription(
            Int16, '/down_camera/rate', self.rate_callback, 10)

        # mavros 高度：使用 BEST_EFFORT QoS（与 mavros 发布端一致）
        qos_best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.altitude_sub = self.create_subscription(
            Altitude, '/mavros/altitude', self.altitude_callback, qos_best_effort)

        self.get_logger().info("YoloV5Ros2 节点初始化完成")

    # ---------------- 工具函数 ----------------
    def load_camera_info(self):
        """从内参文件读取相机参数；失败时回退到默认值"""
        import yaml

        default_path = os.path.join(
            get_package_share_directory('yolov5_ros2'), 'config', 'down_camera_info.yaml')
        path = (self.get_parameter('camera_info_file').value or '').strip() or default_path

        fallback = {
            'k': [269.9682, 0.0, 320.0,
                  0.0, 269.9682, 240.0,
                  0.0, 0.0, 1.0],
            'width': 640,
            'height': 480
        }

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}

            if isinstance(data.get('k'), (list, tuple)) and len(data['k']) >= 9:
                k = [float(v) for v in data['k'][:9]]
                width = int(data.get('width') or data.get('image_width') or 0)
                height = int(data.get('height') or data.get('image_height') or 0)
            elif isinstance(data.get('camera_matrix'), dict):
                cm = data['camera_matrix']
                fx, fy = float(cm['fx']), float(cm['fy'])
                cx, cy = float(cm['cx']), float(cm['cy'])
                k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
                width = int(data.get('image_width') or data.get('width') or 0)
                height = int(data.get('image_height') or data.get('height') or 0)
            elif all(key in data for key in ('fx', 'fy', 'cx', 'cy')):
                fx, fy = float(data['fx']), float(data['fy'])
                cx, cy = float(data['cx']), float(data['cy'])
                k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
                width = int(data.get('image_width') or data.get('width') or 0)
                height = int(data.get('image_height') or data.get('height') or 0)
            else:
                self.get_logger().warning(f"内参文件缺少必要字段: {path}，使用默认值")
                return fallback

            if width <= 0 or height <= 0:
                self.get_logger().warning(f"内参文件缺少图像尺寸: {path}，使用默认值")
                return fallback

            self.get_logger().info(f"下视相机内参来自: {path}")
            return {'k': k, 'width': width, 'height': height}

        except FileNotFoundError:
            self.get_logger().warning(f"内参文件不存在: {path}，使用默认值")
        except Exception as e:
            self.get_logger().warning(f"读取内参文件失败: {path}: {e}，使用默认值")
        return fallback

    def print_camera_info(self):
        """打印当前下视相机内参"""
        if not self.camera_info_received:
            self.get_logger().warning("尚未收到下视相机内参")
            return

        k = self.camera_info.get('k', [])
        if len(k) < 9:
            self.get_logger().warning("内参数据不完整")
            return

        fx, fy = k[0], k[4]
        cx, cy = k[2], k[5]
        width = self.camera_info.get('width', '未知')
        height = self.camera_info.get('height', '未知')

        self.get_logger().info(
            f"下视相机内参:\n"
            f"  fx = {fx:.3f}\n"
            f"  fy = {fy:.3f}\n"
            f"  cx = {cx:.3f}\n"
            f"  cy = {cy:.3f}\n"
            f"  图像尺寸: {width} x {height}"
        )

    # ---------------- 回调函数 ----------------
    def active_callback(self, msg):
        self.active = msg.data

    def rate_callback(self, msg):
        self.rate = msg.data
        self.inference_interval = max(1, int(30 / self.rate)) if self.rate > 0 else 1

    def altitude_callback(self, msg: Altitude):
        """
        订阅 /mavros/altitude。
        ★ 关键修复：使用本地 ROS 时钟作为时间基准，而不是 msg.header.stamp，
        因为相机驱动发出的图像 header.stamp 与 mavros 不在同一时间源，
        直接使用 header.stamp 会导致 Δt ~ 1.79e9 秒，永远匹配不上。
        """
        t_sec = self.get_clock().now().nanoseconds / 1e9

        # 高度：相对起飞点的 local 高度减去相机-机身偏移
        h = msg.local - self.height_offset
        if h < 0.05:
            h = 0.05

        self.height_buffer.append((t_sec, h))
        self.last_height = h
        self.height_received = True

    def get_height_at(self, query_sec):
        """
        按给定本地时钟时间戳在缓存里查最接近的高度值。
        返回 (height, dt)：height 为 None 表示未命中或超过容差。
        """
        if not self.height_buffer:
            return None, None

        best_h = None
        best_dt = float('inf')
        for t_sec, h in self.height_buffer:
            dt = abs(t_sec - query_sec)
            if dt < best_dt:
                best_dt = dt
                best_h = h

        if best_dt > self.height_sync_tolerance:
            return None, best_dt
        return best_h, best_dt

    def image_callback(self, msg: Image):
        if not self.active:
            return
        self.frame_counter += 1
        if self.frame_counter % self.inference_interval != 0:
            return

        # ---------- 使用本地时钟查询对齐高度 ----------
        now = self.get_clock().now()
        now_sec = now.nanoseconds / 1e9

        height, height_dt = self.get_height_at(now_sec)
        height_valid = (height is not None)
        if height_valid:
            self.last_height = height

        # ---------- 推理 ----------
        image = self.bridge.imgmsg_to_cv2(msg)
        detect_result = self.yolov5.predict(image)

        self.result_msg.detections.clear()
        self.result_msg.header.frame_id = "camera"
        self.result_msg.header.stamp = now.to_msg()

        # ---------- 解析结果 ----------
        predictions = detect_result.pred[0]
        boxes = predictions[:, :4]      # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        target_info_list = []
        CONFIDENCE_THRESHOLD = 0.7

        for index in range(len(categories)):
            if float(scores[index]) < CONFIDENCE_THRESHOLD:
                continue

            name = detect_result.names[int(categories[index])]
            detection2d = Detection2D()
            detection2d.id = name

            x1, y1, x2, y2 = boxes[index]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            if ros_distribution == 'galactic':
                detection2d.bbox.center.x = center_x
                detection2d.bbox.center.y = center_y
            else:
                detection2d.bbox.center.position.x = center_x
                detection2d.bbox.center.position.y = center_y

            detection2d.bbox.size_x = float(x2 - x1)
            detection2d.bbox.size_y = float(y2 - y1)

            obj_pose = ObjectHypothesisWithPose()
            obj_pose.hypothesis.class_id = name
            obj_pose.hypothesis.score = float(scores[index])

            detection2d.results.append(obj_pose)
            self.result_msg.detections.append(detection2d)

            # ---------- 像素 → 米（使用与本地时钟对齐的高度） ----------
            if self.camera_info_received and height_valid:
                fx = self.camera_info['k'][0]
                fy = self.camera_info['k'][4]
                cx = self.camera_info['k'][2]
                cy = self.camera_info['k'][5]
                h_use = height

                x_cam = (center_x - cx) * h_use / fx   # 右（东）
                y_cam = (center_y - cy) * h_use / fy   # 下（南）

                front_offset = -y_cam   # 北向
                right_offset = -x_cam    # 东向
            else:
                front_offset = 0.0
                right_offset = 0.0

            target_info_list.append(f"{name}:({front_offset:.2f},{right_offset:.2f})")

            # ---------- 相对偏移消息 ----------
            detection = Detection2D()
            detection.id = name
            detection.bbox.center.position.x = center_x
            detection.bbox.center.position.y = center_y

            obj = ObjectHypothesisWithPose()
            obj.hypothesis.class_id = name
            obj.hypothesis.score = float(scores[index])
            obj.pose.pose.position.x = front_offset
            obj.pose.pose.position.y = right_offset
            obj.pose.pose.position.z = 0.0

            detection.results.append(obj)
            self.relative_result_msg.detections.append(detection)

            # ---------- 绘制 ----------
            if self.show_result or self.pub_result_img:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{name}({front_offset:.2f},{right_offset:.2f})",
                            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.circle(image, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)
                cv2.drawMarker(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)),
                               (255, 0, 0), cv2.MARKER_CROSS, 10, 2)

        # ---------- 显示 / 发布结果图 ----------
        if self.show_result:
            cv2.imshow('result', image)
            cv2.waitKey(1)

        if self.pub_result_img:
            result_img_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            result_img_msg.header = msg.header
            self.result_img_pub.publish(result_img_msg)

        # ---------- 发布检测结果 ----------
        if len(self.relative_result_msg.detections) > 0:
            self.yolo_result_pub.publish(self.result_msg)
            self.relative_result_msg.header.stamp = now.to_msg()
            self.relative_result_pub.publish(self.relative_result_msg)
            self.relative_result_msg.detections.clear()
        else:
            self.result_msg.detections.clear()

        # ---------- 日志（每秒最多一次） ----------
        height_dt_str = f"{height_dt*1000:.1f}ms" if height_dt is not None else "N/A"
        if len(target_info_list) > 0:
            self.get_logger().info(
                f"检测到 {len(target_info_list)} 个目标(置信度≥{CONFIDENCE_THRESHOLD}): "
                f"{'; '.join(target_info_list)} | 高度: {self.last_height:.2f}m "
                f"(Δt={height_dt_str}, valid={height_valid})",
                throttle_duration_sec=1
            )
        else:
            self.get_logger().info(
                f"未检测到满足置信度≥{CONFIDENCE_THRESHOLD} 的目标 | "
                f"高度: {self.last_height:.2f}m (Δt={height_dt_str}, valid={height_valid})",
                throttle_duration_sec=1
            )


def main():
    rclpy.init()
    rclpy.spin(YoloV5Ros2())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
