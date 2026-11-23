from yolov5 import YOLOv5
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
import torch
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from collections import deque
import time
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import message_filters
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import math
from std_msgs.msg import Bool, Int16

# Get the ROS distribution version
ros_distribution = os.environ.get("ROS_DISTRO")
package_share_directory = get_package_share_directory('yolov5_ros2')

class YoloV5Ros2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')
        self.get_logger().info(f"Current ROS 2 distribution: {ros_distribution}")

        # ROS parameters
        self.declare_parameter("device", "cuda", ParameterDescriptor(
            name="device", description="Compute device selection"))

        self.declare_parameter("model", "best", ParameterDescriptor(
            name="model", description="Default model selection"))

        self.declare_parameter("image_topic", "/camera", ParameterDescriptor(
            name="image_topic", description="Color image topic (Gazebo simulation)"))
        
        self.declare_parameter("depth_topic", "/depth_camera", ParameterDescriptor(
            name="depth_topic", description="Depth image topic (Gazebo simulation)"))
        
        self.declare_parameter("camera_info_topic", "/camera_info", ParameterDescriptor(
            name="camera_info_topic", description="Camera information topic"))

        self.declare_parameter("show_result", True, ParameterDescriptor(
            name="show_result", description="Display detection results in OpenCV window"))

        self.declare_parameter("pub_result_img", True, ParameterDescriptor(
            name="pub_result_img", description="Publish result images"))

        self.declare_parameter("depth_scale", 1.0, ParameterDescriptor(
            name="depth_scale", description="Depth scale factor to convert to meters (simulation usually uses meters)"))

        self.declare_parameter("confidence_threshold", 0.5, ParameterDescriptor(
            name="confidence_threshold", description="Confidence threshold for detections"))
        
        self.declare_parameter("nms_iou_threshold", 0.4, ParameterDescriptor(
            name="nms_iou_threshold", description="NMS IoU threshold"))

        # 3D output options
        self.declare_parameter("target_frame", "", ParameterDescriptor(
            name="target_frame", description="Target frame for 3D transformations (empty = camera frame)"))
        
        self.declare_parameter("use_tf", False, ParameterDescriptor(
            name="use_tf", description="Use TF to transform to target frame"))

        # Visualization parameters
        self.declare_parameter("publish_markers", True, ParameterDescriptor(
            name="publish_markers", description="Publish visualization markers for 3D detections"))
        
        self.declare_parameter("marker_frame", "camera_link", ParameterDescriptor(
            name="marker_frame", description="Frame for 3D markers"))
        
        self.declare_parameter("show_3d_labels", True, ParameterDescriptor(
            name="show_3d_labels", description="Show 3D coordinates in visualization"))
        
        self.declare_parameter("show_depth", True, ParameterDescriptor(
            name="show_depth", description="Show depth value in visualization"))
        
        self.declare_parameter("show_confidence", True, ParameterDescriptor(
            name="show_confidence", description="Show confidence score in visualization"))
        
        self.declare_parameter("visualization_scale", 1.0, ParameterDescriptor(
            name="visualization_scale", description="Scale factor for visualization elements"))
        
        self.declare_parameter("bbox_line_thickness", 2, ParameterDescriptor(
            name="bbox_line_thickness", description="Thickness of bounding box lines"))
        
        self.declare_parameter("show_track_id", True, ParameterDescriptor(
            name="show_track_id", description="Show track ID in visualization"))
        
        # 新增参数：深度相机内参话题
        self.declare_parameter("depth_camera_info_topic", "/camera_info", ParameterDescriptor(
            description="Depth camera info topic (if empty, auto-derive from depth_topic)"))
            
        # 先声明彩色相机内参话题
        self.declare_parameter("color_camera_info_topic", 
                               "/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/camera_info",
                               ParameterDescriptor(description="Color camera info topic"))

        # 然后再获取参数值
        color_camera_info_topic = self.get_parameter('color_camera_info_topic').value
            
        # 深度相机内参（动态获取）
        self.depth_fx = 0.0
        self.depth_fy = 0.0
        self.depth_cx = 0.0
        self.depth_cy = 0.0
        self.depth_info_received = False

        # Load model
        model_path = "/home/ling/best.pt"
        device = self.get_parameter('device').value
        self.get_logger().info(f"Loading model from {model_path} on device {device}")
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found: {model_path}")
        else:
            self.get_logger().info(f"Model file found: {model_path}")
            
        self.yolov5 = YOLOv5(model_path=model_path, device=device)

        # Create publishers
        self.stable_result_pub = self.create_publisher(Detection2DArray, "stable_yolo_result", 10)
        self.result_img_pub = self.create_publisher(Image, "result_img", 10)
        
        # New publisher for 3D detections
        self.detections_3d_pub = self.create_publisher(Detection2DArray, "detections_3d", 10)
        
        # Visualization markers publisher
        self.marker_pub = self.create_publisher(MarkerArray, "detection_markers", 10)

        # Define QoS profiles for simulation sensor data (Best Effort)
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Synchronize image and depth topics
        image_topic = self.get_parameter('image_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        
        self.image_sub = message_filters.Subscriber(self, Image, image_topic, qos_profile=best_effort_qos)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=best_effort_qos)
        
        # Approximate time synchronization
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)

        # Camera info subscriber
        camera_info_topic = self.get_parameter('camera_info_topic').value
        # 获取彩色相机内参话题
        color_camera_info_topic = self.get_parameter('color_camera_info_topic').value
        self.color_info_sub = self.create_subscription(
            CameraInfo, color_camera_info_topic, self.camera_info_callback, best_effort_qos)
        # 订阅深度相机内参
        depth_info_topic = self.get_parameter('depth_camera_info_topic').value
        if not depth_info_topic:
            # 如果未指定，根据深度图像话题推导（假设深度话题以 'image' 结尾）
            depth_info_topic = depth_topic.replace('/image', '/camera_info')
            self.get_logger().info(f"Auto-derived depth camera info topic: {depth_info_topic}")

        self.depth_info_sub = self.create_subscription(
            CameraInfo, depth_info_topic, self.depth_camera_info_callback, best_effort_qos)
        self.bridge = CvBridge()

        # Get parameter values
        self.show_result = self.get_parameter('show_result').value
        self.pub_result_img = self.get_parameter('pub_result_img').value
        self.depth_scale = self.get_parameter('depth_scale').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.nms_iou_threshold = self.get_parameter('nms_iou_threshold').value
        self.target_frame = self.get_parameter('target_frame').value
        self.use_tf = self.get_parameter('use_tf').value
        
        # Visualization parameters
        self.publish_markers = self.get_parameter('publish_markers').value
        self.marker_frame = self.get_parameter('marker_frame').value
        self.show_3d_labels = self.get_parameter('show_3d_labels').value
        self.show_depth = self.get_parameter('show_depth').value
        self.show_confidence = self.get_parameter('show_confidence').value
        self.viz_scale = self.get_parameter('visualization_scale').value
        self.bbox_line_thickness = self.get_parameter('bbox_line_thickness').value
        self.show_track_id = self.get_parameter('show_track_id').value

        # Camera information
        self.camera_info = None
        self.camera_info_received = False
        
        # TF2 setup
        if self.use_tf and self.target_frame:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.get_logger().info(f"TF enabled, target frame: {self.target_frame}")
        
        # Publish control
        self.last_publish_time = 0
        self.publish_interval = 0.5
        
        # Image dimensions
        self.image_width = None
        self.image_height = None
        
        # Frame counter
        self.frame_count = 0
        
        # Colors for different classes (for visualization)
        self.class_colors = {}
        
        self.get_logger().info(f"YoloV5 ROS2 Node initialized with 3D support and visualization (adapted for Gazebo simulation)")
        
        self.active = True          # 默认激活（可调整）
        self.rate = 30
        self.frame_counter = 0
        self.inference_interval = 1

        self.active_sub = self.create_subscription(Bool, '/front_camera/activate', self.active_callback, 10)
        self.rate_sub = self.create_subscription(Int16, '/front_camera/rate', self.rate_callback, 10)
        # 可选：初始发布一次默认设置（但订阅会触发回调）
        # 主动发送一次初始设置（如果控制节点已运行）
        # self.active_callback(Bool(data=True))
        # self.rate_callback(Int16(data=30))
        
    def active_callback(self, msg):
        self.active = msg.data
        self.get_logger().info(f"Front camera active: {self.active}")

    def rate_callback(self, msg):
        self.rate = msg.data
        if self.rate <= 0:
            self.inference_interval = 1   # 若率为0，仍推理（但可通过停用避免）
        else:
            # 假设图像话题频率约为30Hz，计算跳过帧数
            self.inference_interval = max(1, int(30 / self.rate))
        self.get_logger().info(f"Front camera rate={self.rate}, interval={self.inference_interval}")

    def camera_info_callback(self, msg: CameraInfo):
        """Camera information callback"""
        self.camera_info = {
            'k': msg.k,
            'p': msg.p,
            'd': msg.d,
            'r': msg.r,
            'roi': msg.roi,
            'width': msg.width,
            'height': msg.height
        }
        if not self.camera_info_received:
            self.get_logger().info(f"Camera info received: fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}, cx={msg.k[2]:.2f}, cy={msg.k[5]:.2f}")
            self.camera_info_received = True

    def get_color_for_class(self, class_name):
        """Generate consistent color for each class"""
        if class_name not in self.class_colors:
            # Generate random but consistent color based on class name hash
            hash_val = hash(class_name) % 256
            np.random.seed(hash_val)
            self.class_colors[class_name] = tuple(np.random.randint(0, 255, 3).tolist())
        return self.class_colors[class_name]
        
    def depth_camera_info_callback(self, msg: CameraInfo):
        """获取深度相机内参"""
        self.depth_fx = msg.k[0]
        self.depth_fy = msg.k[4]
        self.depth_cx = msg.k[2]
        self.depth_cy = msg.k[5]
        self.depth_info_received = True
        if not hasattr(self, '_depth_info_logged'):
            self.get_logger().info(f"Depth camera intrinsics: fx={self.depth_fx:.2f}, fy={self.depth_fy:.2f}, cx={self.depth_cx:.2f}, cy={self.depth_cy:.2f}")
            self._depth_info_logged = True

    def get_depth_at_point(self, depth_image, x, y):
        """Get depth value at specified point"""
        if depth_image is None:
            return float('nan')

        try:
            h, w = depth_image.shape[:2]
            x = int(round(x))
            y = int(round(y))
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))

            # Take median of 3x3 region for robustness
            x_min = max(0, x-1)
            x_max = min(w-1, x+1)
            y_min = max(0, y-1)
            y_max = min(h-1, y+1)
            
            region = depth_image[y_min:y_max+1, x_min:x_max+1]
            valid_depths = region[region > 0]  # Filter out invalid depths (0)
            
            if len(valid_depths) > 0:
                return np.median(valid_depths) * self.depth_scale
            return float('nan')
            
        except Exception as e:
            return float('nan')

    def pixel_to_3d(self, pixel, depth, camera_info=None):
        if np.isnan(depth) or depth <= 0:
            return None
        try:
            fx = self.depth_fx
            fy = self.depth_fy
            cx = self.depth_cx
            cy = self.depth_cy
            # 若内参尚未收到，使用一个保守默认值（但此时应该还未处理图像）
            if not self.depth_info_received:
                self.get_logger().warning("Depth camera info not yet received, using default intrinsics (may be inaccurate)")
                fx = 480.0  # 临时值
                fy = 480.0
                cx = 320.0
                cy = 240.0
            x = (pixel[0] - cx) * depth / fx
            y = (pixel[1] - cy) * depth / fy
            z = depth
            return np.array([x, y, z])
        except Exception as e:
            self.get_logger().error(f"Error in pixel_to_3d: {e}")
            return None
            
    def transform_point(self, point_np, from_frame, to_frame, time):
        """Transform point from one frame to another using TF"""
        if not self.use_tf or not self.target_frame:
            return point_np
            
        try:
            # Look up transform
            transform = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                time,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Apply translation
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            tz = transform.transform.translation.z
            
            # For simplicity, we're only applying translation
            # For full transformation including rotation, use tf2_geometry_msgs
            transformed_point = point_np.copy()
            transformed_point[0] += tx
            transformed_point[1] += ty
            transformed_point[2] += tz
            
            return transformed_point
            
        except Exception as e:
            self.get_logger().warning(f"TF transform failed: {e}")
            return point_np

    def non_max_suppression(self, boxes, scores, iou_threshold):
        """NMS to remove overlapping boxes"""
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        indices = np.argsort(scores)[::-1]
        
        keep = []
        
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            
            if len(indices) == 1:
                break
                
            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            intersection = w * h
            
            iou = intersection / (area[i] + area[indices[1:]] - intersection)
            indices = indices[1:][iou < iou_threshold]
        
        return keep

    def process_predictions(self, predictions):
        """Process YOLO predictions"""
        if predictions is None or len(predictions) == 0:
            return [], [], []
        
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        
        # Confidence filtering
        mask = predictions[:, 4] >= self.confidence_threshold
        predictions = predictions[mask]
        
        if len(predictions) == 0:
            return [], [], []
        
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        categories = predictions[:, 5]
        
        return boxes, scores, categories

    def create_detection_markers(self, detections, stamp, frame_id):
        """Create visualization markers for 3D detections"""
        marker_array = MarkerArray()
        
        for i, detection in enumerate(detections):
            if detection['point_3d'] is None:
                continue
                
            point_3d = detection['point_3d']
            class_name = detection['category']
            track_id = detection['track_id']
            confidence = detection['score']
            depth = detection['depth']
            
            # Get color for this class
            color = self.get_color_for_class(class_name)
            
            # Create sphere marker for 3D position
            sphere_marker = Marker()
            sphere_marker.header.stamp = stamp
            sphere_marker.header.frame_id = frame_id
            sphere_marker.ns = "detections_3d"
            sphere_marker.id = track_id
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            
            sphere_marker.pose.position.x = float(point_3d[0])
            sphere_marker.pose.position.y = float(point_3d[1])
            sphere_marker.pose.position.z = float(point_3d[2])
            sphere_marker.pose.orientation.w = 1.0
            
            # Scale based on depth (farther objects appear smaller)
            scale = 0.1 * self.viz_scale * (1.0 / max(depth, 0.5))
            sphere_marker.scale.x = scale
            sphere_marker.scale.y = scale
            sphere_marker.scale.z = scale
            
            sphere_marker.color.r = color[0] / 255.0
            sphere_marker.color.g = color[1] / 255.0
            sphere_marker.color.b = color[2] / 255.0
            sphere_marker.color.a = 0.7  # Semi-transparent
            
            sphere_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            
            marker_array.markers.append(sphere_marker)
            
            # Create text marker for label
            text_marker = Marker()
            text_marker.header.stamp = stamp
            text_marker.header.frame_id = frame_id
            text_marker.ns = "detection_labels"
            text_marker.id = track_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = float(point_3d[0])
            text_marker.pose.position.y = float(point_3d[1])
            text_marker.pose.position.z = float(point_3d[2]) + 0.15  # Offset above sphere
            
            # Build label text
            label_parts = [f"{class_name}"]
            if self.show_confidence:
                label_parts.append(f"{confidence:.2f}")
            if self.show_depth:
                label_parts.append(f"{depth:.2f}m")
            if self.show_3d_labels:
                label_parts.append(f"({point_3d[0]:.2f},{point_3d[1]:.2f},{point_3d[2]:.2f})")
            
            text_marker.text = " ".join(label_parts)
            
            text_marker.scale.x = 0.1 * self.viz_scale
            text_marker.scale.y = 0.1 * self.viz_scale
            text_marker.scale.z = 0.1 * self.viz_scale
            
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            text_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            
            marker_array.markers.append(text_marker)
        
        return marker_array

    def draw_detections_simple(self, image, detections):
        """
        在图像上绘制检测框和标签，直接基于检测结果列表。
        detections: 每个元素为 dict，包含 'bbox': (x1,y1,x2,y2), 'category', 'score', 'depth', 'point_3d'。
        """
        viz_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # 确保整数坐标
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # 裁剪到图像范围
            x1 = max(0, min(x1, image.shape[1] - 1))
            y1 = max(0, min(y1, image.shape[0] - 1))
            x2 = max(0, min(x2, image.shape[1] - 1))
            y2 = max(0, min(y2, image.shape[0] - 1))
        
            color = self.get_color_for_class(det['category'])
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, self.bbox_line_thickness)
        
            # 绘制中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(viz_image, (center_x, center_y), 5, (0, 0, 255), -1)
        
            # 构建标签文本
            label_parts = [det['category']]
            if self.show_confidence:
                label_parts.append(f"{det['score']:.2f}")
            if self.show_depth and det['depth'] is not None and not np.isnan(det['depth']):
                label_parts.append(f"{det['depth']:.2f}m")
            if self.show_3d_labels and det.get('point_3d') is not None:
                pt = det['point_3d']
                label_parts.append(f"({pt[0]:.2f},{pt[1]:.2f},{pt[2]:.2f})")
            # 添加 ID 显示
            if self.show_track_id and 'id' in det:
                label_parts.append(f"ID:{det['id']}")
        
            label = " ".join(label_parts)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_bg_x1 = x1
            label_bg_y1 = y1 - label_size[1] - 10
            label_bg_x2 = x1 + label_size[0]
            label_bg_y2 = y1
            label_bg_y1 = max(0, label_bg_y1)
            
            cv2.rectangle(viz_image, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, -1)
        
            text_y = y1 - 5
            if text_y - 5 < 0:
                text_y = y2 + 20
            cv2.putText(viz_image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
        return viz_image

    def synced_callback(self, image_msg: Image, depth_msg: Image):
        if not self.active:
            return
        self.frame_counter += 1
        if self.frame_counter % self.inference_interval != 0:
            return

        # 检查深度内参
        if not self.depth_info_received:
            self.get_logger().warning("Waiting for depth camera info", throttle_duration_sec=5.0)
            return

        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            self.image_height, self.image_width = image.shape[:2]
            depth_h, depth_w = depth_image.shape[:2]
            scale_x = depth_w / self.image_width
            scale_y = depth_h / self.image_height
        except Exception as e:
            self.get_logger().error(f"Failed to convert images: {e}")
            return

        try:
            detect_result = self.yolov5.predict(image)
            predictions = detect_result.pred[0] if hasattr(detect_result, 'pred') and len(detect_result.pred) > 0 else None
        except Exception as e:
            self.get_logger().error(f"YOLO prediction failed: {e}")
            return

        # 准备 3D 检测消息
        detections_3d_msg = Detection2DArray()
        if self.use_tf and self.target_frame:
            detections_3d_msg.header.frame_id = self.target_frame
        else:
            detections_3d_msg.header.frame_id = image_msg.header.frame_id
        detections_3d_msg.header.stamp = self.get_clock().now().to_msg()

        boxes, scores, categories = self.process_predictions(predictions)
        detections_for_viz = []   # 用于可视化

        if len(boxes) > 0:
            keep_indices = self.non_max_suppression(boxes, scores, self.nms_iou_threshold)
            for idx in keep_indices:
                # 获取类别名
                if detect_result.names and len(detect_result.names) > 0:
                    if isinstance(detect_result.names, dict):
                        name = detect_result.names.get(int(categories[idx]), "unknown")
                    else:
                        name = detect_result.names[int(categories[idx])] if int(categories[idx]) < len(detect_result.names) else "unknown"
                else:
                    name = f"class_{int(categories[idx])}"

                x1, y1, x2, y2 = boxes[idx]
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                # 深度映射
                if self.camera_info is not None and self.depth_info_received:
                    fx_c = self.camera_info['k'][0]
                    fy_c = self.camera_info['k'][4]
                    cx_c = self.camera_info['k'][2]
                    cy_c = self.camera_info['k'][5]
                    fx_d = self.depth_fx
                    fy_d = self.depth_fy
                    cx_d = self.depth_cx
                    cy_d = self.depth_cy
                    x_norm = (center_x - cx_c) / fx_c
                    y_norm = (center_y - cy_c) / fy_c
                    u_d = x_norm * fx_d + cx_d
                    v_d = y_norm * fy_d + cy_d
                else:
                    u_d = center_x * scale_x
                    v_d = center_y * scale_y

                depth_value = self.get_depth_at_point(depth_image, u_d, v_d)
                self.get_logger().info(f"Depth at ({center_x:.1f},{center_y:.1f}) = {depth_value:.3f}")

                point_3d = None
                if not np.isnan(depth_value) and depth_value > 0:
                    point_3d = self.pixel_to_3d([u_d, v_d], depth_value)
                    if point_3d is not None and self.use_tf and self.target_frame:
                        point_3d = self.transform_point(point_3d, image_msg.header.frame_id, self.target_frame, image_msg.header.stamp)
    
                # 添加到可视化列表
                detections_for_viz.append({
                    'id': idx, 
                    'bbox': (x1, y1, x2, y2),
                    'category': name,
                    'score': scores[idx],
                    'depth': depth_value,
                    'point_3d': point_3d
                })

                # 构建 3D 检测消息
                detection = Detection2D()
                detection.id = f"{name}_{idx}"
                detection.bbox.center.position.x = float(center_x) / float(self.image_width)
                detection.bbox.center.position.y = float(center_y) / float(self.image_height)
                detection.bbox.size_x = float(x2 - x1) / float(self.image_width)
                detection.bbox.size_y = float(y2 - y1) / float(self.image_height)

                obj = ObjectHypothesisWithPose()
                obj.hypothesis.class_id = name
                obj.hypothesis.score = float(scores[idx])
                if point_3d is not None:
                    obj.pose.pose.position.x = float(point_3d[0])
                    obj.pose.pose.position.y = float(point_3d[1])
                    obj.pose.pose.position.z = float(point_3d[2])
                    # 可选协方差
                    if not np.isnan(depth_value):
                        depth_uncertainty = 0.01 * depth_value
                        obj.pose.covariance[0] = depth_uncertainty
                        obj.pose.covariance[7] = depth_uncertainty
                        obj.pose.covariance[14] = depth_uncertainty
                detection.results.append(obj)
                detections_3d_msg.detections.append(detection)

        # 发布 3D 检测结果
        self.detections_3d_pub.publish(detections_3d_msg)
        self.get_logger().info(f"Published {len(detections_3d_msg.detections)} detections")

        # 可视化
        if self.show_result:
            viz_image = self.draw_detections_simple(image, detections_for_viz)
            cv2.imshow('YOLO Detection with 3D', viz_image)
            cv2.waitKey(1)

        # 发布结果图像
        if self.pub_result_img:
            try:
                viz_image = self.draw_detections_simple(image, detections_for_viz)

                # 规范化：确保是 uint8、3 通道、连续内存
                if viz_image.dtype != np.uint8:
                    viz_image = viz_image.astype(np.uint8)
                if viz_image.ndim == 2:
                    viz_image = cv2.cvtColor(viz_image, cv2.COLOR_GRAY2BGR)
                elif viz_image.ndim == 3 and viz_image.shape[2] == 4:
                    viz_image = cv2.cvtColor(viz_image, cv2.COLOR_BGRA2BGR)
                elif viz_image.ndim == 3 and viz_image.shape[2] == 1:
                    viz_image = cv2.cvtColor(viz_image, cv2.COLOR_GRAY2BGR)
                viz_image = np.ascontiguousarray(viz_image, dtype=np.uint8)

                # ===== 手动构造 Image 消息，绕开 cv_bridge =====
                result_img_msg = Image()
                result_img_msg.header = image_msg.header
                result_img_msg.height = int(viz_image.shape[0])
                result_img_msg.width = int(viz_image.shape[1])
                result_img_msg.encoding = "bgr8"
                result_img_msg.is_bigendian = 0
                result_img_msg.step = int(viz_image.shape[1] * 3)
                result_img_msg.data = viz_image.tobytes()
                # =============================================

                self.result_img_pub.publish(result_img_msg)
            except Exception as e:
                self.get_logger().error(
                    f"Failed to publish result image: {type(e).__name__}: {e}")
                    
def main(args=None):
    rclpy.init(args=args)
    node = YoloV5Ros2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
