#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped, Point
from mavros_msgs.srv import CommandBool, SetMode,CommandLong
from mavros_msgs.msg import State
import numpy as np
import time
from builtin_interfaces.msg import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy
import math
import traceback
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Pose, Quaternion
from rcl_interfaces.msg import ParameterDescriptor
from mavros_msgs.msg import PositionTarget
from std_msgs.msg import Bool, Int16
from collections import deque
import cmath  # 用于偏航角复数平均
from std_msgs.msg import Float32

class UavTracker(Node):
    def __init__(self):
        super().__init__('uav_tracker')
        
        # 参数声明
        self.declare_parameters(
            namespace='',
            parameters=[
                ('target_height', 1.0),
                ('bomb_scan_height_offset', 0.5),
                ('bomb_recheck_hover_time', 3.0),      # 新增：下降后二次修正采集时间
                ('bomb_stabilize_time', 2.0),          # 新增：二次修正后稳定时间
                ('bomb_post_drop_hover_time', 2.0),   # 新增：投弹后悬停时长
                ('k_p', 0.5),
                ('k_i', 0.01),
                ('k_d', 0.1),
                ('arm_timeout', 10),
                ('max_speed', 1.0),
                ('home_position', [0.0, 0.0, 2.0]),
                ('camera_pitch_deg', 90),
                ('camera_fov', 90.0),
                ('ground_threshold', 0.1),
                ('position_tolerance', 0.3),
                ('velocity_tolerance', 0.2),
                ('height_tolerance', 0.3),
                ('flight_controller', 'PX4'),
                ('position_timeout', 3.0),
                ('verbose_logging', True),
                ('queue_size', 10),
                ('scan_radius', 50.0),
                ('scan_speed', 0.1),
                ('spiral_step', 0.2),
                ('target_timeout', 2.0),
                ('point1_x', 28.8),
                ('point2_x', 56),
                ('scout_scan_hover_time', 2.0),   # ← 2 秒悬停
                ('num_bomb_targets', 3),
                # 偏航采样参数
                ('yaw_stabilize_time', 4.0),
                ('yaw_sample_count', 50),
                ('down_settle_time', 3.0),   # 高度变化后，下视数据稳定所需的时间（秒）
                # 下视搜索（前视未检测到目标时）
                ('drop_zone_center_offset', 6.0),  # 投弹区中心 = point1 前 offset 米
                ('down_search_height', 4.0),       # 下视搜索抬升高度(m)
                ('down_search_timeout', 10.0),     # 下视搜索超时(s)
            ]
        )
        
        self.declare_parameter(
            'camera_rotation',
            [0.0, 0.0, 0.0, 1.0],
            ParameterDescriptor(description='Quaternion from camera frame to body frame')
        )

        self.declare_parameter(
            'camera_translation',
            [0.0, 0.0, 0.0],
            ParameterDescriptor(description='Translation from camera to body center (NED)')
        )

        self.declare_parameter(
            'bomb_hover_duration',
            15.0,
            ParameterDescriptor(description='Hover time over bomb targets (seconds)')
        )

        self.declare_parameter(
            'scout_hover_duration',
            4.0,
            ParameterDescriptor(description='Hover time over scout targets (seconds)')
        )

        self.declare_parameter(
            'target_collect_timeout',
            10.0,
            ParameterDescriptor(description='Timeout for collecting targets (seconds)')
        )
        self.declare_parameter('point1_distance', 28.5)
        self.declare_parameter('point2_distance', 56.0)
        self.point1_distance = self.get_parameter('point1_distance').value
        self.point2_distance = self.get_parameter('point2_distance').value
        
        # QoS配置
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        
        # 状态常量
        self.MAX_RETRIES = 5
        self.RETRY_DELAY = 2.0
        
        # 状态机初始化
        self.state = 'INIT'
        
        # 控制变量初始化
        self.dt = 0.1
        self.last_control_time = None      # 上次控制循环时间戳，用于实测 dt
        self.control_error_count = 0       # 控制循环连续异常计数
        self.init_retry_after = 0.0        # 初始化重试的下次允许时间（非阻塞）
        self.integral = np.zeros(2)
        self.last_error = np.zeros(2)
        self.current_pose = None
        self.position_timeout = self.get_parameter('position_timeout').value
        self.verbose = self.get_parameter('verbose_logging').value
        self.home_pos = self.get_parameter('home_position').value
        
        # 扫描相关变量
        self.scan_radius = self.get_parameter('scan_radius').value
        self.scan_speed = self.get_parameter('scan_speed').value
        self.spiral_step = self.get_parameter('spiral_step').value
        self.scan_angle = 0.0
        self.scan_center = np.array([0.0, 0.0])
        self.scan_center_initialized = False
        self.spiral_radius = 0.0
        
        # 偏航采样相关（替换 origin_yaw）
        self.initial_yaw = None
        self.yaw_samples = []
        self.yaw_sampling_start = None
        self.yaw_sampled = False
        self.yaw_sample_count = self.get_parameter('yaw_sample_count').value
        self.yaw_stabilize_time = self.get_parameter('yaw_stabilize_time').value
        
        # 着陆模式设置的异步追踪变量
        self.land_mode_pending = False
        
        # 状态变量
        self.current_state = State()
        self.target_detected = False
        self.target_pos = np.array([0.0, 0.0])
        self.armed = False
        self.bomb_dropped = False
        self.in_landing_phase = False
        self.on_ground = True
        self.return_home_complete = False
        self.connected = False
        self.position_received = False
        self.initial_setpoint_published = False
        self.last_state_log_time = time.time()
        self.last_ascend_log = time.time()
        self.last_search_log = time.time()
        self.last_target_time = 0.0
        self.velocity_control_active = False
        self.last_debug_log = time.time()
        self.last_pose_log_time = 0.0
        
        self.bomb_stabilize_start = None
        
        # 飞控特定模式名称
        self.flight_controller = self.get_parameter('flight_controller').value
        if self.flight_controller == 'PX4':
            self.position_control_mode = "OFFBOARD"
            self.land_mode = "AUTO.LAND"
        else:
            self.position_control_mode = "GUIDED"
            self.land_mode = "LAND"
        
        # 初始化状态
        self.init_substate = 0
        self.init_retry_count = 0
        self.mode_set_attempted = False
        self.arm_attempted = False
        self.set_mode_future = None
        self.arm_future = None
        
        # 服务客户端
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.cmd_client = self.create_client(CommandLong, '/mavros/cmd/command')
        
        # 订阅话题
        self.state_sub = self.create_subscription(
            State, '/mavros/state', self.state_cb, qos_profile)
        
        self.pos_sub = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.pose_cb, qos_profile)
        
        # 发布话题
        self.pos_pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', qos_profile)
        
        self.vel_pub = self.create_publisher(
            TwistStamped, '/mavros/setpoint_velocity/cmd_vel', qos_profile)
        
        # 目标检测订阅
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections_3d',
            self.detections_3d_callback,
            qos_profile          # 必须与发布端一致（best-effort），否则收不到
        )
        
        # 控制定时器
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        # 初始高度设定点
        self.target_height = self.get_parameter('target_height').value
        
        self.get_logger().info("节点已启动,等待飞控连接...")
        self.get_logger().info(f'飞行控制模式:{self.position_control_mode}')
        self.get_logger().info(f"着陆模式:{self.land_mode}")
        self.get_logger().info(f"初始home位置:{self.home_pos}")
        self.get_logger().info(f'扫描半径:{self.scan_radius}m,扫描速度:{self.scan_speed}rad/s')
        
        self.point1_hover_start_time = None
        
        # 目标收集列表
        self.bomb_targets = []
        self.target_index = 0
        self.detection_start_time = 0.0
        self.last_bomb_time = 0.0
        self.bomb_targets_filtered = []
        # 下视搜索状态（前视未检测到目标时）
        self.down_search_active = False
        self.down_search_mode = False
        self.down_search_start = None
        self.down_search_targets = []
        self.down_search_center = None
        self._pid_align_stable_start = None

        # 偏航控制
        self.desired_yaw = None
        
        self.raw_local_pub = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)
        self.yaw_stabilized = False
        self.low_altitude = 0.5
        self.mission_complete = False
        self._auto_land_set = False
        self._landing_stable = False
        self._landing_stable_start = None
        self._low_alt_start_time = None
        
        self.front_activate_pub = self.create_publisher(Bool, '/front_camera/activate', 10)
        self.front_rate_pub = self.create_publisher(Int16, '/front_camera/rate', 10)
        self.down_activate_pub = self.create_publisher(Bool, '/down_camera/activate', 10)
        self.down_rate_pub = self.create_publisher(Int16, '/down_camera/rate', 10)
        self.HIGH_RATE = 30
        self.LOW_RATE = 0
        
        # 订阅下视相机相对偏移检测
        self.down_detection_sub = self.create_subscription(
            Detection2DArray,
            '/down_detections_relative',
            self.down_detection_callback,
            qos_profile)         # ros_gz_bridge 发 best-effort，reliable 会收不到
        self.latest_down_offset = None
        
        self.declare_parameter('target_merge_distance', 1.0, ParameterDescriptor(description='合并重复目标的最小距离(m)'))
        self.target_merge_distance = self.get_parameter('target_merge_distance').value

        # ===== 新增：目标过远时航点后退参数 =====
        self.declare_parameter(
            'far_target_depth_threshold', 5.0,
            ParameterDescriptor(description='触发航点后退的目标深度阈值(m)'))
        self.declare_parameter(
            'far_target_back_offset', 0.3,
            ParameterDescriptor(description='目标过远时航点向后偏移距离(m)'))
        self.far_target_depth_threshold = self.get_parameter('far_target_depth_threshold').value
        self.far_target_back_offset = self.get_parameter('far_target_back_offset').value
        # ========================================
        
        self.declare_parameter('align_threshold', 0.10, 
                               ParameterDescriptor(description='对准偏移阈值(m)，投弹门槛'))
        self.declare_parameter('align_stable_time', 2.0,
                               ParameterDescriptor(description='对准稳定时间(s)'))
        self.align_threshold = self.get_parameter('align_threshold').value
        self.align_stable_time = self.get_parameter('align_stable_time').value

        # 偏移历史（滑动窗口）—— 增大窗口使解算出的偏移更稳定
        self.offset_history = deque(maxlen=8)
        self.latest_down_offset_time = 0.0
        
        # 图像尺寸（下视相机）
        self.declare_parameter('down_image_width', 1920, ParameterDescriptor(description='下视相机图像宽度(px)'))
        self.declare_parameter('down_image_height', 1080, ParameterDescriptor(description='下视相机图像高度(px)'))
        self.down_image_width = self.get_parameter('down_image_width').value
        self.down_image_height = self.get_parameter('down_image_height').value

        # 像素对准阈值和稳定时间 —— 提高精度：20px -> 8px
        self.declare_parameter('pixel_align_threshold', 8, ParameterDescriptor(description='像素对准阈值(px)'))
        self.declare_parameter('pixel_align_stable_time', 2.0, ParameterDescriptor(description='像素对准稳定时间(s)'))
        self.pixel_align_threshold = self.get_parameter('pixel_align_threshold').value
        self.pixel_align_stable_time = self.get_parameter('pixel_align_stable_time').value
        self.down_settle_time = self.get_parameter('down_settle_time').value

        # 像素坐标历史（滑动平均）—— 增大窗口使像素偏移更平滑
        self.pixel_history = deque(maxlen=8)
        self.latest_pixel_center = None
        self.latest_pixel_time = 0.0
        
        self.bomb_precise_wait_start = None
        self.bomb_align_start_time = None
        self.bomb_precise_xy = None
        
        self.accept_down_data = True
        
        self.bomb_precise_delay_start = None
        
        self.point1_stable_start_time = None   # 记录稳定开始时间
        self.point2_stable_start_time = None   # 记录到达 point2 后稳定开始时间
        
        # 在 __init__ 中添加
        self.compass_hdg_sub = self.create_subscription(
            Float32,  # 修改此处
            '/mavros/global_position/compass_hdg',
            self.compass_hdg_callback,
            qos_profile
        )
        self.compass_hdg = None
        
        self.stable_start_time = None
        
        # 侦察区矩形扫描（新增）
        self.scout_scan_waypoints = []
        self.scout_scan_index = 0
        self.scout_scan_substate = None
        self.scout_scan_hover_start = None
        self.scout_scan_hover_time = self.get_parameter('scout_scan_hover_time').value
        
        # 全局坐标原点（解锁时记录）
        self.origin_offset_ned = None
        
        self.bomb_scan_height_offset = self.get_parameter('bomb_scan_height_offset').value
        self.bomb_rise_start = None
        self.bomb_descend_start = None
        self.bomb_recheck_hover_time = self.get_parameter('bomb_recheck_hover_time').value
        self.bomb_stabilize_time = self.get_parameter('bomb_stabilize_time').value
        self.bomb_recheck_start = None
        self.bomb_post_drop_hover_time = self.get_parameter('bomb_post_drop_hover_time').value
        self.bomb_post_drop_start = None

    def _now(self):
        """统一时间基准：优先 ROS 时钟（仿真 use_sim_time 时才是正确的）。"""
        return self.get_clock().now().nanoseconds / 1e9

    def set_front_camera(self, active: bool, rate: int):
        self.front_activate_pub.publish(Bool(data=active))
        self.front_rate_pub.publish(Int16(data=rate))
        self.get_logger().info(f"Front camera: active={active}, rate={rate}")

    def set_down_camera(self, active: bool, rate: int):
        self.down_activate_pub.publish(Bool(data=active))
        self.down_rate_pub.publish(Int16(data=rate))
        self.get_logger().info(f"Down camera: active={active}, rate={rate}")

    def set_mode(self, mode_name):
        try:
            if not self.set_mode_client.service_is_ready():
                self.get_logger().warning(f"设置模式服务不可用，无法设置模式: {mode_name}")
                return False
        
            req = SetMode.Request(custom_mode=mode_name)
            future = self.set_mode_client.call_async(req)
            future.add_done_callback(lambda f: self._set_mode_done_callback(f, mode_name))
            return True
        except Exception as e:
            self.get_logger().error(f"设置模式异常: {e}")
            return False

    def _set_mode_done_callback(self, future, mode_name):
        try:
            if future.done():
                response = future.result()
                if response.mode_sent:
                    self.get_logger().info(f"成功设置飞行模式: {mode_name}")
                else:
                    self.get_logger().warning(f"设置模式失败: {mode_name}")
        except Exception as e:
            self.get_logger().error(f"设置模式回调异常: {e}")

    def state_cb(self, msg):
        self.get_logger().info(f"飞控状态更新:连接={msg.connected},模式={msg.mode},解锁={msg.armed}", 
                              throttle_duration_sec=2)
        try:
            self.current_state = msg
            self.armed = msg.armed
            self.connected = msg.connected
            if hasattr(self, 'land_mode_pending') and self.land_mode_pending and msg.mode == self.land_mode:
                self.land_mode_pending = False
                self.get_logger().info(f"确认已切换到{self.land_mode}模式")
            
            current_time = time.time()
            if current_time - self.last_state_log_time > 5.0:
                self.last_state_log_time = current_time
                if self.connected:
                    if not self.position_received:
                        self.get_logger().info("已连接MAVROS,等待位置数据...")
                    elif self.verbose:
                        self.get_logger().info(f'MAVROS已连接,模式:{msg.mode},解锁:{msg.armed}')
                else:
                    self.get_logger().warning("MAVROS连接断开!")
        except Exception as e:
            self.get_logger().error(f'状态回调异常: {e}')
    
    def pose_cb(self, msg):
        try:
            # ===== 首次收到位置时记录为全局坐标原点 =====
            if self.origin_offset_ned is None:
                self.origin_offset_ned = np.array([
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z
                ])
                self.get_logger().info(
                    f"全局坐标原点已设置（首次收到位置）: "
                    f"x={self.origin_offset_ned[0]:.3f}, "
                    f"y={self.origin_offset_ned[1]:.3f}, "
                    f"z={self.origin_offset_ned[2]:.3f}"
                )
            # ============================================

            # 把原始 NED 坐标转换为相对原点的坐标
            msg.pose.position.x -= self.origin_offset_ned[0]
            msg.pose.position.y -= self.origin_offset_ned[1]
            msg.pose.position.z -= self.origin_offset_ned[2]

            self.current_pose = msg
            self.position_received = True
            self.last_position_time = self.get_clock().now().to_msg()

            ground_threshold = self.get_parameter('ground_threshold').value
            ground_buffer = 0.5
            self.on_ground = self.current_pose.pose.position.z < (ground_threshold + ground_buffer)

            if not self.scan_center_initialized:
                self.scan_center = np.array([
                    self.current_pose.pose.position.x,
                    self.current_pose.pose.position.y
                ])
                self.scan_center_initialized = True
                self.get_logger().info(
                    f"初始化扫描中心:[{self.scan_center[0]:.1f},{self.scan_center[1]:.1f}]")

            if self.state == "LANDING" and self.on_ground and not self.armed:
                self._handle_landing_complete()

        except Exception as e:
            self.get_logger().error(f"位置回调异常:{e}")
            self.get_logger().error(traceback.format_exc())
    
    def _handle_landing_complete(self):
        self.get_logger().info("着陆完成!系统待命")
        self.mission_complete = True
        self.armed = False
        self.bomb_dropped = False
        self.target_detected = False
        self.state = "MISSION_COMPLETE"
        self.integral = np.zeros(2)
        self.last_error = np.zeros(2)

    def get_yaw_from_pose(self):
        if self.current_pose is None:
            return 0.0
        
        q = self.current_pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw
        
    def is_duplicate_target(self, new_point, target_list, threshold=0.2):
        for existing in target_list:
            dist = np.linalg.norm(
                np.array(new_point[:2]) - np.array(existing[:2]))   # ← existing 也只取前两个
            if dist < threshold:
                return True
        return False

    def detections_3d_callback(self, msg: Detection2DArray):
        if self.current_pose is None:
            return
    
        if self.state != "WAIT_BOMB_DETECTIONS":
            return
    
        current_time = time.time()
        if self.state != "WAIT_BOMB_DETECTIONS":
            return

        last_time = getattr(self, '_last_bomb_time', 0)
        if current_time - last_time < 1.5:
            return
        self._last_bomb_time = current_time
        target_list = self.bomb_targets
        required = self.get_parameter('num_bomb_targets').value
        state_after = "GO_BOMB_TARGETS"
    
        for detection in msg.detections:
            if not detection.results:
                continue
            pos = detection.results[0].pose.pose.position
            cam_point = np.array([pos.x, pos.y, pos.z])
            if cam_point[2] > 25.0:
                self.get_logger().debug(f"跳过深度过大的目标: {cam_point[2]:.1f}m")
                continue
            global_point = self.camera_point_to_global_ned(cam_point)
            if global_point is None:
                continue

            # ===== 深度距离超过阈值时，航点向后偏移 =====
            if cam_point[2] > self.far_target_depth_threshold:
                yaw = self.get_yaw_from_pose()
                back_offset = self.far_target_back_offset
                global_point[0] -= back_offset * math.cos(yaw)
                global_point[1] -= back_offset * math.sin(yaw)
                self.get_logger().info(
                    f"目标深度 {cam_point[2]:.2f}m > {self.far_target_depth_threshold:.2f}m，"
                    f"航点向后偏移 {back_offset:.2f}m -> "
                    f"NED=({global_point[0]:.2f}, {global_point[1]:.2f})"
                )
            # ============================================

            # ===== 新增：读前视图像的归一化横坐标（bbox 中心 x）=====
            px = detection.bbox.center.position.x   # 0~1 归一化值
            # ========================================================

            if not self.is_duplicate_target(global_point, target_list, self.target_merge_distance):
                # 存 [N, E, px]
                target_list.append([global_point[0], global_point[1], px])
                self.get_logger().info(
                    f"收集目标 #{len(target_list)}: "
                    f"NED=({global_point[0]:.1f}, {global_point[1]:.1f}), "
                    f"norm_x={px:.3f}"
                )
            else:
                self.get_logger().debug("目标与已有目标重复，忽略")
        
            if len(target_list) >= required:
                self.get_logger().info(f"目标收集完成，准备访问")
                self.state = state_after
                self.target_index = 0
                self.set_front_camera(True, self.LOW_RATE)
                self.set_down_camera(True, self.HIGH_RATE)
                break
                
    def down_detection_callback(self, msg: Detection2DArray):
        if getattr(self, 'down_search_mode', False):
            self._collect_down_search_targets(msg)
            return
        if not self.accept_down_data:
            return
        if not msg.detections:
            return

        candidates = []
        for det in msg.detections:
            if not det.results:
                continue
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            front = det.results[0].pose.pose.position.x
            right = det.results[0].pose.pose.position.y
            dist = math.hypot(front, right)
            candidates.append((cx, cy, front, right, dist))

        if not candidates:
            return

        # 选择策略：距离最近优先；距离相同（0.02m 内视为相同）则选画面最右侧
        best = min(candidates, key=lambda x: (round(x[4], 2), -x[0]))
        cx, cy, front, right, dist_best = best

        if len(candidates) > 1:
            self.get_logger().info(
                f"下视检测到 {len(candidates)} 个目标，选择最近(或最右)的在 "
                f"(cx={cx:.1f}px, cy={cy:.1f}px, front={front:.3f}m, "
                f"right={right:.3f}m, dist={dist_best:.3f}m)"
            )

        self.pixel_history.append((cx, cy))
        if len(self.pixel_history) > 0:
            avg_cx = np.mean([p[0] for p in self.pixel_history])
            avg_cy = np.mean([p[1] for p in self.pixel_history])
            self.latest_pixel_center = (avg_cx, avg_cy)
            self.latest_pixel_time = time.time()

        self.offset_history.append((front, right))
        if len(self.offset_history) > 0:
            avg_front = np.mean([o[0] for o in self.offset_history])
            avg_right = np.mean([o[1] for o in self.offset_history])
            self.latest_down_offset = (avg_front, avg_right)
            self.latest_down_offset_time = time.time()
            self.get_logger().debug(f"平滑后偏移: front={avg_front:.3f}, right={avg_right:.3f}")
                
    def drop_bomb_servo(self):
        if not self.cmd_client.service_is_ready():
            self.get_logger().warning("命令服务不可用，无法投弹")
            return False

        # 根据目标序号选择舵机 PWM 输出:
        #   第 1 个目标 (target_index == 0): param1 = 0.0 -> PWM 1500 (中位)
        #   第 2 个目标 (target_index == 1): param1 = 1.0 -> PWM 2000
        if self.target_index == 0:
            param1_value = 0.0
            pwm_desc = "param1=0.0 (PWM 1500)"
        elif self.target_index == 1:
            param1_value = 1.0
            pwm_desc = "param1=1.0 (PWM 2000)"
        else:
            self.get_logger().warning(
                f"目标索引 {self.target_index} 超出投弹范围 (仅支持 0/1)，忽略本次投弹")
            return False

        req = CommandLong.Request()
        req.command = 187            # MAV_CMD_DO_SET_ACTUATOR
        req.param1 = param1_value    # 对应 Actuator Set 1 (即 PWM_MAIN_FUNC5=301 的通道)
        req.param2 = 0.0
        req.param3 = 0.0
        req.param4 = 0.0
        req.param5 = 0.0
        req.param6 = 0.0
        req.param7 = 0.0

        self.get_logger().info(
            f"执行投弹 (第 {self.target_index + 1} 个目标): "
            f"DO_SET_ACTUATOR {pwm_desc}")

        future = self.cmd_client.call_async(req)

        def callback(fut):
            try:
                response = fut.result()
                if response.success:
                    self.get_logger().info(
                        f"投弹指令执行成功 (第 {self.target_index + 1} 个目标): {pwm_desc}")
                else:
                    self.get_logger().warning(
                        f"投弹指令执行失败 (第 {self.target_index + 1} 个目标)，"
                        f"结果码: {response.result}")
            except Exception as e:
                self.get_logger().error(f"投弹服务回调异常: {e}")

        future.add_done_callback(callback)
        return True

    def calculate_velocity(self):
        if self.current_pose is None:
            return 0.0, 0.0
        
        try:
            current_pos = np.array([
                self.current_pose.pose.position.x,
                self.current_pose.pose.position.y
            ])
            error = self.target_pos - current_pos
            
            # 用实测 dt：定时器抖动或被短暂阻塞时，积分/微分才不会失真
            now = self.get_clock().now().nanoseconds / 1e9
            if self.last_control_time is not None:
                dt = now - self.last_control_time
            else:
                dt = self.dt
            if dt <= 0.0 or dt > 1.0:
                dt = self.dt
            self.last_control_time = now

            self.integral += error * dt
            max_integral = 2.0
            self.integral = np.clip(self.integral, -max_integral, max_integral)
            
            derivative = (error - self.last_error) / dt
            
            control = (
                self.get_parameter('k_p').value * error +
                self.get_parameter('k_i').value * self.integral +
                self.get_parameter('k_d').value * derivative
            )
            
            self.last_error = error.copy()
            
            max_speed = self.get_parameter('max_speed').value
            return (
                np.clip(control[0], -max_speed, max_speed),
                np.clip(control[1], -max_speed, max_speed)
            )
        except Exception as e:
            self.get_logger().error(f'速度计算错误:{e}')
            return 0.0, 0.0
    
    def drop_bomb(self):
        try:
            target_height = self.get_parameter('target_height').value
            height_error = abs(self.current_pose.pose.position.z - target_height)
            height_tol = self.get_parameter('height_tolerance').value
            
            current_pos = np.array([
                self.current_pose.pose.position.x,
                self.current_pose.pose.position.y
            ])
            position_error = np.linalg.norm(self.target_pos - current_pos)
            position_tol = self.get_parameter('position_tolerance').value
            
            if (height_error <= height_tol and
                position_error <= position_tol):
                
                self.get_logger().info(
                    f"模拟投弹:高度={self.current_pose.pose.position.z:.2f}m | "
                    f"位置误差={position_error:.2f}m")
                self.bomb_dropped = True
                return True
            else:
                self.get_logger().warning(
                    f"投弹条件不满足: "
                    f"高度误差={height_error:.2f}m(需<{height_tol:.2f}) | "
                    f"位置误差={position_error:.2f}m(需<{position_tol:.2f})")
                return False
        except Exception as e:
            self.get_logger().error(f'投弹失败:{e}')
            return False
    
    def _transition_to_velocity_control(self):
        """发布零速指令完成模式切换。非阻塞：不在控制定时器里 sleep。"""
        self.get_logger().info("切换到速度控制模式")
        try:
            vel_cmd = TwistStamped()
            vel_cmd.header.stamp = self.get_clock().now().to_msg()
            self.vel_pub.publish(vel_cmd)
        except Exception as e:
            self.get_logger().error(f'速度模式切换失败:{e}')
    
    def attempt_initialization(self):
        if self.state != "PREFLIGHT":
            return
        # 非阻塞重试：未到允许时间就直接返回，由定时器下次再推进
        if self.init_retry_after and self._now() < self.init_retry_after:
            return
        self.init_retry_after = 0.0
        
        try:
            if self.init_substate == 0:
                self.init_substate = 1
                self.init_retry_count = 0
            
            if self.init_substate == 1:
                if not self.mode_set_attempted:
                    self.publish_position_setpoint(0, 0, self.target_height)
                    req = SetMode.Request(custom_mode=self.position_control_mode)
                    self.set_mode_future = self.set_mode_client.call_async(req)
                    self.mode_set_attempted = True
                    self.get_logger().info(f"尝试设置模式:{self.position_control_mode}")
                
                if self.set_mode_future.done():
                    try:
                        response = self.set_mode_future.result()
                        if response.mode_sent:
                            self.get_logger().info(f"成功设置模式:{self.position_control_mode}")
                            self.init_substate = 2
                            self.arm_attempted = False
                        else:
                            self.get_logger().warning("设置模式失败，重试中...")
                            self._retry_init_step()
                    except Exception as e:
                        self.get_logger().error(f"设置模式异常: {e}")
                        self._retry_init_step()
            
            if self.init_substate == 2:
                if not self.arm_attempted:
                    self.publish_position_setpoint(0, 0, self.target_height)
                    req = CommandBool.Request(value=True)
                    self.arm_future = self.arm_client.call_async(req)
                    self.arm_attempted = True
                    self.get_logger().info("尝试解锁无人机")
                
                if self.arm_future.done():
                    try:
                        response = self.arm_future.result()
                        if response.success:
                            self.get_logger().info("解锁成功!")
                            self.armed = True

                            self.state = "YAW_SAMPLING"
                            self.yaw_sampled = False
                            self.yaw_samples = []
                            self.yaw_sampling_start = None
                            self.get_logger().info("解锁成功，开始地面偏航角采样...")
                            self.stable_start_time = None
                        else:
                            self.get_logger().warning('解锁失败，请检查飞控状态')
                            self._retry_init_step()
                    except Exception as e:
                        self.get_logger().error(f"解锁异常:{e}")
                        self._retry_init_step()
        except Exception as e:
            self.get_logger().error(f"初始化异常:{e}")
            self.state = "INIT"
            self.init_substate = 0
    
    def _retry_init_step(self):
        self.init_retry_count += 1
        self.mode_set_attempted = False
        self.arm_attempted = False
        if self.init_retry_count >= self.MAX_RETRIES:
            self.get_logger().error("初始化失败次数过多!进入安全模式")
            self.state = "SAFE"
            self.publish_position_setpoint(0, 0, 0)
        else:
            self.init_substate = 1
            # 用时间戳代替 sleep，避免阻塞控制循环
            self.init_retry_after = self._now() + self.RETRY_DELAY

    def publish_position_setpoint(self, x, y, z):
        try:
            target = PoseStamped()
            target.header.stamp = self.get_clock().now().to_msg()
            target.header.frame_id = "map"
            # 相对坐标 → 绝对 NED
            if self.origin_offset_ned is not None:
                x = float(x) + self.origin_offset_ned[0]
                y = float(y) + self.origin_offset_ned[1]
                z = float(z) + self.origin_offset_ned[2]
            target.pose.position = Point(x=float(x), y=float(y), z=float(z))
            self.pos_pub.publish(target)
            self.initial_setpoint_published = True
        except Exception as e:
            self.get_logger().error(f"发布位置设定点时出错:{e}")
            
    def publish_pose_with_yaw(self, x, y, z, yaw):
        target = PoseStamped()
        target.header.stamp = self.get_clock().now().to_msg()
        target.header.frame_id = "map"
        if self.origin_offset_ned is not None:
            x = float(x) + self.origin_offset_ned[0]
            y = float(y) + self.origin_offset_ned[1]
            z = float(z) + self.origin_offset_ned[2]
        target.pose.position = Point(x=float(x), y=float(y), z=float(z))
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        target.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
        self.pos_pub.publish(target)        
        
    def publish_ned_position(self, north, east, down, yaw=None):
        msg = PositionTarget()
        msg.coordinate_frame = 1  # FRAME_LOCAL_NED
        msg.type_mask = (PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ |
                         PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                         PositionTarget.IGNORE_YAW_RATE)
        if self.origin_offset_ned is not None:
            msg.position.x = float(north + self.origin_offset_ned[0])
            msg.position.y = float(east + self.origin_offset_ned[1])
            msg.position.z = float(down + self.origin_offset_ned[2])
        else:
            msg.position.x = float(north)
            msg.position.y = float(east)
            msg.position.z = float(down)
        if yaw is not None:
            msg.yaw = yaw
        else:
            msg.yaw = self.get_yaw_from_pose()
        self.raw_local_pub.publish(msg)
        
    def _handle_yaw_sampling(self):
        self.publish_pose_with_yaw(0.0, 0.0, 0.0, 0.0)

        if self.yaw_sampling_start is None:
            self.yaw_sampling_start = time.time()
            self.yaw_samples = []
            return

        current_yaw = self.get_yaw_from_pose()
        self.yaw_samples.append(current_yaw)

        elapsed = time.time() - self.yaw_sampling_start
        if elapsed >= self.yaw_stabilize_time or len(self.yaw_samples) >= self.yaw_sample_count:
            avg_complex = sum(cmath.exp(1j * y) for y in self.yaw_samples) / len(self.yaw_samples)
            avg_yaw = cmath.phase(avg_complex)
            self.initial_yaw = avg_yaw
            self.yaw_sampled = True
            self.get_logger().info(f"偏航采样完成，均值: {avg_yaw:.4f} rad")
            self.state = "ASCENDING"
            self.stable_start_time = None
            self.yaw_samples = []
            self.yaw_sampling_start = None

    def calculate_spiral_position(self):
        self.scan_angle += self.scan_speed * self.dt
        radius_increment = self.spiral_step * self.dt
        self.spiral_radius += radius_increment
        if self.spiral_radius >= self.scan_radius:
            self.spiral_radius = self.scan_radius
        x = self.scan_center[0] + self.spiral_radius * math.cos(self.scan_angle)
        y = self.scan_center[1] + self.spiral_radius * math.sin(self.scan_angle)
        return x, y
        
    def is_position_valid(self):
        if not self.position_received or self.current_pose is None:
            return False
        
        try:
            now = self.get_clock().now()
            last_time = Time()
            last_time.sec = self.last_position_time.sec
            last_time.nanosec = self.last_position_time.nanosec
            time_diff = (now.nanoseconds - self.last_position_time.sec * 1e9 - self.last_position_time.nanosec) / 1e9
            return time_diff < self.position_timeout
        except Exception as e:
            self.get_logger().error(f"时间计算错误:{e}")
            return False
            
    def _log_pose_info(self, prefix=""):
        if self.current_pose is None:
            return
    
        now = time.time()
        if now - self.last_pose_log_time < 1.0:
            return
        self.last_pose_log_time = now

        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y
        z = self.current_pose.pose.position.z

        q = self.current_pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        yaw_deg = math.degrees(yaw)

        height_error = abs(z - self.home_pos[2])
        pos_error = np.linalg.norm([x - self.home_pos[0], y - self.home_pos[1]])

        mode = self.current_state.mode if hasattr(self, 'current_state') else "N/A"
        armed = self.armed

        self.get_logger().info(
            f"{prefix} 位姿 | 位置: ({x:.3f}, {y:.3f}, {z:.3f}) | "
            f"偏航: {yaw_deg:.1f}° | "
            f"水平误差: {pos_error:.3f}m | 高度误差: {height_error:.3f}m | "
            f"模式: {mode} | 解锁: {armed}"
        )
        
    def camera_point_to_global_ned(self, cam_point):
        body_ned = np.array([cam_point[2], cam_point[0], cam_point[1]])
        trans = self.get_parameter('camera_translation').value
        body_ned += trans
        body_enu = np.array([-body_ned[1], body_ned[0], -body_ned[2]])

        if self.current_pose is None:
            return None
        uav_x = self.current_pose.pose.position.x
        uav_y = self.current_pose.pose.position.y
        uav_z = self.current_pose.pose.position.z
        yaw = self.get_yaw_from_pose()
        yaw_deg = math.degrees(yaw)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        global_enu = np.array([
            uav_x + body_enu[1] * cos_y - body_enu[0] * sin_y,
            uav_y + body_enu[1] * sin_y + body_enu[0] * cos_y,
            uav_z + body_enu[2]
        ])

        global_ned = np.array([
            global_enu[0],
            global_enu[1],
            -global_enu[2]
        ])
        
        self.get_logger().info(
            f"cam_point=({cam_point[0]:.2f}, {cam_point[1]:.2f}, {cam_point[2]:.2f}) | "
            f"body_ned=({body_ned[0]:.2f}, {body_ned[1]:.2f}, {body_ned[2]:.2f}) | "
            f"body_enu=({body_enu[0]:.2f}, {body_enu[1]:.2f}, {body_enu[2]:.2f}) | "
            f"yaw={yaw_deg:.1f}° | "
            f"global_enu=({global_enu[0]:.2f}, {global_enu[1]:.2f}, {global_enu[2]:.2f}) | "
            f"global_ned=({global_ned[0]:.2f}, {global_ned[1]:.2f}, {global_ned[2]:.2f})"
    )

        return global_ned
        
    def enu_to_ned(self, enu_x, enu_y, enu_z):
        return (enu_y, enu_x, -enu_z)

    def ned_to_enu(self, ned_x, ned_y, ned_z):
        return (ned_y, ned_x, -ned_z)
              
    def compass_hdg_callback(self, msg):
        self.compass_hdg = msg.data

    def control_loop(self):
        try:
            if self.mission_complete:
                if self.current_pose is not None:
                    self.publish_position_setpoint(
                        self.current_pose.pose.position.x,
                        self.current_pose.pose.position.y,
                        0.0
                    )
                return
                
            if self.state == "SAFE":
                self.publish_position_setpoint(0, 0, 0)
                return
            
            if not self._basic_checks():
                return
            
            if self.state == "INIT":
                self._handle_init_state()
            elif self.state == 'PREFLIGHT':
                self._handle_preflight_state()
            elif self.state == "YAW_SAMPLING":
                self._handle_yaw_sampling()
            elif self.state == "ASCENDING":
                self._handle_ascending_state()
            elif self.state == "SEARCHING":
                self._handle_searching_state()
            elif self.state == "GO_POINT1":
                self._handle_go_point1()
            elif self.state == "WAIT_BOMB_DETECTIONS":
                self._handle_wait_bomb_detections()
            elif self.state == "GO_BOMB_TARGETS":
                self._handle_go_bomb_targets()
            elif self.state == "GO_POINT2":
                self._handle_go_point2()
            elif self.state == "SCOUT_SCAN":
                self._handle_scout_scan()
            elif self.state == 'TRACKING':
                self._handle_tracking_state()
            elif self.state == 'RETURNING':
                self._handle_returning_state()
            elif self.state == 'LANDING':
                self._handle_landing_state()
            else:
                self.get_logger().error(f'未知状态:{self.state}')
            
            now = time.time()
            if now - self.last_state_log_time > 1.0:
                self.get_logger().info(f"状态:{self.state} | "
                                      f"高度:{self.current_pose.pose.position.z:.1f}m | "
                                      f"解锁:{self.armed} | "
                                      f"目标:{'有' if self.target_detected else '无'}")
                self.last_state_log_time = now
        except Exception as e:
            self.control_error_count += 1
            self.get_logger().error(
                f'控制循环异常(连续第{self.control_error_count}次):{e}\n'
                f'{traceback.format_exc()}')
            if self.armed and not self.in_landing_phase:
                self._trigger_emergency_landing()
            if self.control_error_count >= 10:
                self.get_logger().fatal('控制循环连续异常，进入安全模式')
                self.state = 'SAFE'
                self.control_error_count = 0
        else:
            self.control_error_count = 0

    def _basic_checks(self):
        if not self.is_position_valid():
            if not self.position_received:
                self.get_logger().warning("等待位置数据...", throttle_duration_sec=1)
            return False

        # 已移除“高度过低触发紧急着陆”的保护逻辑

        return True

    def _handle_init_state(self):
        if self.connected and self.position_received:
            self.get_logger().info("系统已连接并准备好，进入预飞行状态")
            self.state = "PREFLIGHT"
        else:
            if not self.connected:
                self.get_logger().info("等待飞控连接...")
            elif not self.position_received:
                self.get_logger().info("等待位置数据...")
            self.publish_ned_position(0, 0, self.target_height, self.initial_yaw)

    def _handle_preflight_state(self):
        self.attempt_initialization()
        self.publish_ned_position(0, 0, self.target_height, self.initial_yaw)

    def _handle_ascending_state(self):
        if not self.armed:
            self.state = "PREFLIGHT"
            return

        target_height = self.get_parameter('target_height').value
        self.publish_ned_position(0, 0, target_height, self.initial_yaw)

        current_height = self.current_pose.pose.position.z
        height_error = abs(target_height - current_height)

        if height_error < 0.2:
            if self.stable_start_time is None:
                self.stable_start_time = time.time()
            elif time.time() - self.stable_start_time > 1.0:
                self.get_logger().info("起飞完成，计算航点")
                yaw = self.initial_yaw
                start_enu = (0.0, 0.0, target_height)
                dist1 = self.get_parameter('point1_distance').value
                dist2 = self.get_parameter('point2_distance').value
                target1_enu = (start_enu[0] + dist1 * math.cos(yaw),
                               start_enu[1] + dist1 * math.sin(yaw),
                               start_enu[2])
                target2_enu = (start_enu[0] + dist2 * math.cos(yaw),
                               start_enu[1] + dist2 * math.sin(yaw),
                               start_enu[2])
                self.point1_ned = np.array(self.enu_to_ned(target1_enu[1], target1_enu[0], target1_enu[2]))
                self.point2_ned = np.array(self.enu_to_ned(target2_enu[1], target2_enu[0], target2_enu[2]))
                self.get_logger().info(f"point1 NED: {self.point1_ned}")
                self.get_logger().info(f"point2 NED: {self.point2_ned}")
                self.state = "GO_POINT1"
                self.stable_start_time = None
        else:
            self.stable_start_time = None
            
    def _handle_go_point1(self):
        self.publish_ned_position(self.point1_ned[0], self.point1_ned[1], -self.point1_ned[2], yaw=self.initial_yaw)

        current_xy = np.array([self.current_pose.pose.position.x, self.current_pose.pose.position.y])
        target_xy = self.point1_ned[:2]
        dist = np.linalg.norm(current_xy - target_xy)
        height_error = abs(self.current_pose.pose.position.z - self.target_height)
        pos_tol = self.get_parameter('position_tolerance').value
        height_tol = self.get_parameter('height_tolerance').value

        if dist < pos_tol and height_error < height_tol:
            if self.point1_stable_start_time is None:
                self.point1_stable_start_time = time.time()
                self.get_logger().info(f"已到达 point1，开始稳定悬停（误差：位置 {dist:.2f}m，高度 {height_error:.2f}m）")
            elif time.time() - self.point1_stable_start_time >= 4.0:
                self.get_logger().info("稳定悬停完成，开始收集炸弹目标")
                self.state = "WAIT_BOMB_DETECTIONS"
                self.set_front_camera(True, self.HIGH_RATE)
                self.set_down_camera(False, 0)
                self.detection_start_time = time.time()
                self.bomb_targets.clear()
                self.point1_stable_start_time = None
        else:
            self.point1_stable_start_time = None
            
    def _handle_wait_bomb_detections(self):
        timeout = self.get_parameter('target_collect_timeout').value
        if time.time() - self.detection_start_time > timeout:
            self.get_logger().warning("炸弹目标收集超时，使用已收集目标")
            self.state = "GO_BOMB_TARGETS"
            self.target_index = 0
            self.set_front_camera(False, 0)
            self.set_down_camera(True, self.HIGH_RATE)
            return
        self.publish_ned_position(self.point1_ned[0], self.point1_ned[1],
                                  -self.point1_ned[2], yaw=self.initial_yaw)        
        
    def _handle_go_bomb_targets(self):
        # ===== 前视无目标时：飞到投弹区中心 + 抬升 + 下视搜索 =====
        if self.down_search_active:
            self._handle_down_search()
            return

        if not hasattr(self, 'bomb_targets_filtered') or not self.bomb_targets_filtered:
            if len(self.bomb_targets) == 0:
                self._start_down_search()
                return
            elif len(self.bomb_targets) == 1:
                # ===== 情况 B：只有 1 个目标，两发弹药都投于它 =====
                self.get_logger().warning(
                    "前视仅检测到 1 个目标，两发弹药都投于同一目标")
                t = list(self.bomb_targets[0])   # 复制一份
                self.bomb_targets_filtered = [t, list(t)]
                # ===============================================
            else:
                # ===== 情况 C：≥ 2 个目标，按原逻辑选画面最左侧两个 =====
                sorted_targets = sorted(self.bomb_targets, key=lambda p: p[2])
                self.bomb_targets_filtered = sorted_targets[:2]
                self.get_logger().info(
                    f"筛选出 {len(self.bomb_targets_filtered)} 个炸弹目标"
                    f"（前视画面最左侧的两个）: "
                    + ", ".join([f"(NED=({t[0]:.2f},{t[1]:.2f}), norm_x={t[2]:.3f})"
                                 for t in self.bomb_targets_filtered])
                )
                # ========================================================

            self.target_index = 0
            self.bomb_substate = 'APPROACH'
            self.bomb_precise_xy = None
            self.bomb_align_start_time = None
            self.bomb_stabilize_start = None

        if self.target_index >= len(self.bomb_targets_filtered):
            self.get_logger().info("所有炸弹目标完成，飞往 point2")
            self.state = "GO_POINT2"
            self.set_down_camera(False, 0)
            self.set_front_camera(True, self.HIGH_RATE)
            self.bomb_targets_filtered = []
            self.bomb_substate = None
            self.bomb_precise_xy = None
            self.bomb_align_start_time = None
            self.bomb_stabilize_start = None
            return

        tx, ty = self.bomb_targets_filtered[self.target_index][:2]
        target_height = self.get_parameter('target_height').value
        # ===== 使用参数化的抬升高度 =====
        scan_height = target_height + self.bomb_scan_height_offset   # 下视采集/对准时的高度
        # ================================
        pos_tol = self.get_parameter('position_tolerance').value
        height_tol = self.get_parameter('height_tolerance').value

        current_xy = np.array([self.current_pose.pose.position.x,
                               self.current_pose.pose.position.y])
        current_z = self.current_pose.pose.position.z
        dist = np.linalg.norm(current_xy - np.array([tx, ty]))
        height_error = abs(current_z - target_height)
        height_error_scan = abs(current_z - scan_height)

        # ---------- 子状态 1：飞向粗略目标 ----------
        if self.bomb_substate == 'APPROACH':
            self.publish_ned_position(tx, ty, target_height, yaw=self.initial_yaw)
            if dist < pos_tol and height_error < height_tol:
                self.get_logger().info(
                    f"到达粗略目标 (误差 {dist:.2f}m)，抬升 "
                    f"{self.bomb_scan_height_offset:.2f}m 到 {scan_height:.2f}m")
                self.bomb_substate = 'RISE'
                self.bomb_rise_start = time.time()

        # ---------- 子状态 1.5：抬升 ----------
        elif self.bomb_substate == 'RISE':
            self.publish_ned_position(tx, ty, scan_height, yaw=self.initial_yaw)
            if height_error_scan < height_tol:
                self.get_logger().info("抬升完成，稳定 3 秒")
                self.bomb_substate = 'STABILIZE_BEFORE_COLLECT'
                self.bomb_stabilize_start = time.time()

        # ---------- 子状态 1.7：稳定 3 秒 ----------
        elif self.bomb_substate == 'STABILIZE_BEFORE_COLLECT':
            self.publish_ned_position(tx, ty, scan_height, yaw=self.initial_yaw)
            if time.time() - self.bomb_stabilize_start >= 3.0:
                self.get_logger().info("稳定完成，开始 4 秒采集下视数据")
                self.bomb_substate = 'HOVER_COLLECT'
                self.bomb_hover_start = time.time()
                # 重置下视数据
                self.latest_pixel_center = None
                self.latest_down_offset = None
                self.pixel_history.clear()
                self.offset_history.clear()
                self.accept_down_data = True
                self.set_down_camera(True, self.HIGH_RATE)

        # ---------- 子状态 2：悬停等待高度稳定 + 采集下视数据 ----------
        elif self.bomb_substate == 'HOVER_COLLECT':
            self.publish_ned_position(tx, ty, scan_height, yaw=self.initial_yaw)

            elapsed = time.time() - self.bomb_hover_start

            # 前 down_settle_time 秒内，气压计高度尚未稳定，
            # 此时下视数据算出的偏移会随高度漂移，持续清空缓冲区丢弃
            if elapsed < self.down_settle_time:
                # 丢弃不稳定期间的数据，只保留稳定后的
                self.pixel_history.clear()
                self.offset_history.clear()
                self.latest_pixel_center = None
                self.latest_down_offset = None
                # 每 1 秒打一次进度
                if int(elapsed * 10) % 10 == 0:
                    self.get_logger().info(
                        f"[HOVER_COLLECT] 等待下视数据稳定中 "
                        f"{elapsed:.1f}/{self.down_settle_time}s (高度 {self.current_pose.pose.position.z:.2f}m)")
                return

            # 稳定后再额外采集 1 秒数据作为最终平均
            extra_collect = 1.0
            if elapsed >= self.down_settle_time + extra_collect:
                self.get_logger().info(
                    f"悬停稳定 {self.down_settle_time:.1f}s + 采集 {extra_collect:.1f}s 完成，"
                    f"计算精确目标位置")
                self.set_down_camera(False, 0)
                self.accept_down_data = False

                if self.latest_down_offset is not None:
                    front, right = self.latest_down_offset
                    yaw = self.get_yaw_from_pose()
                    cos_y = math.cos(yaw)
                    sin_y = math.sin(yaw)

                    uav_x = self.current_pose.pose.position.x
                    uav_y = self.current_pose.pose.position.y
                    target_x = uav_x + front * cos_y + right * sin_y
                    target_y = uav_y + front * sin_y - right * cos_y
                    self.bomb_precise_xy = (target_x, target_y)

                    self.get_logger().info(
                        f"采集偏移: front={front:.4f}, right={right:.4f}, "
                        f"yaw={math.degrees(yaw):.1f}° | "
                        f"无人机=({uav_x:.4f}, {uav_y:.4f}) | "
                        f"精确目标=({target_x:.4f}, {target_y:.4f})"
                    )
                else:
                    self.bomb_precise_xy = (tx, ty)
                    self.get_logger().warning(
                        f"未采集到下视偏移，使用粗略目标 ({tx:.4f}, {ty:.4f})")

                # ===== 改用 PID + 位姿速度控制对准 =====
                self.target_pos = np.array(self.bomb_precise_xy)
                self.integral = np.zeros(2)
                self.last_error = np.zeros(2)
                self._pid_align_stable_start = None
                self.accept_down_data = True
                self.set_down_camera(True, self.HIGH_RATE)
                self.bomb_substate = 'PID_ALIGN'
                self.bomb_align_start_time = None

        # ---------- 子状态 2.5：PID + 位姿速度对准（下视闭环） ----------
        elif self.bomb_substate == 'PID_ALIGN':
            target_height = self.get_parameter('target_height').value
            # 投弹阶段用更严的对准阈值：position_tolerance(0.3m) 会让目标
            # 偏离画面中心约 60 像素（1.4m 高度），这里改用 align_threshold
            pos_tol = self.get_parameter('align_threshold').value
            height_tol = self.get_parameter('height_tolerance').value

            # 下视相机持续更新目标（闭环）
            if (self.latest_down_offset is not None and
                    time.time() - self.latest_down_offset_time < 1.0):
                front, right = self.latest_down_offset
                yaw = self.get_yaw_from_pose()
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                uav_x = self.current_pose.pose.position.x
                uav_y = self.current_pose.pose.position.y
                self.target_pos = np.array([
                    uav_x + front * cos_y + right * sin_y,
                    uav_y + front * sin_y - right * cos_y,
                ])

            # PID 速度（世界 N/E）
            vn, ve = self.calculate_velocity()

            # 发布：XY 用速度、Z 用位置（FRAME_LOCAL_NED）
            msg = PositionTarget()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            msg.coordinate_frame = 1  # FRAME_LOCAL_NED
            msg.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY |
                             PositionTarget.IGNORE_VZ |
                             PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY |
                             PositionTarget.IGNORE_AFZ | PositionTarget.IGNORE_YAW_RATE)
            z_abs = target_height + (self.origin_offset_ned[2] if self.origin_offset_ned is not None else 0.0)
            msg.position.z = float(z_abs)
            msg.velocity.x = float(vn)
            msg.velocity.y = float(ve)
            msg.yaw = self.initial_yaw
            self.raw_local_pub.publish(msg)

            current_xy = np.array([self.current_pose.pose.position.x,
                                   self.current_pose.pose.position.y])
            pos_err = np.linalg.norm(self.target_pos - current_xy)
            height_err = abs(self.current_pose.pose.position.z - target_height)

            if pos_err < pos_tol and height_err < height_tol:
                if self._pid_align_stable_start is None:
                    self._pid_align_stable_start = time.time()
                elif time.time() - self._pid_align_stable_start >= 1.0:
                    self.get_logger().info(
                        f"PID 对准完成 (位置误差 {pos_err:.2f}m, 高度误差 {height_err:.2f}m)，投弹")
                    self.publish_ned_position(self.target_pos[0], self.target_pos[1],
                                              target_height, yaw=self.initial_yaw)
                    self.set_down_camera(False, 0)
                    self.accept_down_data = False
                    if self.drop_bomb_servo():
                        self.bomb_dropped = True
                        self.bomb_substate = 'POST_DROP_HOVER'
                        self.bomb_post_drop_start = time.time()
                        self.get_logger().info(
                            f"弹药已发出，保持悬停 {self.bomb_post_drop_hover_time} 秒")
            else:
                self._pid_align_stable_start = None

        # ---------- 子状态 3：飞向精确目标（在抬升高度） ----------
        elif self.bomb_substate == 'GO_PRECISE':
            px, py = self.bomb_precise_xy
            self.publish_ned_position(px, py, scan_height, yaw=self.initial_yaw)

            precise_dist = np.linalg.norm(current_xy - np.array([px, py]))
            if precise_dist < pos_tol and height_error_scan < height_tol:
                self.get_logger().info(
                    f"已对准精确目标 (误差 {precise_dist:.2f}m)，"
                    f"开始下降到 {target_height:.2f}m")
                self.bomb_substate = 'DESCEND'

        # ---------- 子状态 3.5：下降到 target_height ----------
        elif self.bomb_substate == 'DESCEND':
            px, py = self.bomb_precise_xy
            self.publish_ned_position(px, py, target_height, yaw=self.initial_yaw)

            if height_error < height_tol:
                self.get_logger().info(
                    f"下降完成，开启下视相机二次修正 "
                    f"{self.bomb_recheck_hover_time} 秒")
                # 打开下视相机、清空历史，准备二次采集
                self.latest_pixel_center = None
                self.latest_down_offset = None
                self.pixel_history.clear()
                self.offset_history.clear()
                self.accept_down_data = True
                self.set_down_camera(True, self.HIGH_RATE)
                self.bomb_substate = 'RE_CORRECT'
                self.bomb_recheck_start = time.time()
                
        # ---------- 子状态 3.7：下视相机二次修正（等待稳定后） ----------
        elif self.bomb_substate == 'RE_CORRECT':
            px, py = self.bomb_precise_xy
            self.publish_ned_position(px, py, target_height, yaw=self.initial_yaw)

            elapsed = time.time() - self.bomb_recheck_start
            # 二次修正时是下降后的高度，气压计同样需要稳定时间
            wait_time = max(self.bomb_recheck_hover_time, self.down_settle_time)

      # 稳定期：丢弃漂移数据
            if elapsed < wait_time:
                self.pixel_history.clear()
                self.offset_history.clear()
                self.latest_pixel_center = None
                self.latest_down_offset = None
                if int(elapsed * 10) % 10 == 0:
                    self.get_logger().info(
                        f"[RE_CORRECT] 等待下视数据稳定中 "
                        f"{elapsed:.1f}/{wait_time:.1f}s (高度 {self.current_pose.pose.position.z:.2f}m)")
                return

            # 稳定后再采集 1 秒
            extra_collect = 1.0
            if elapsed >= wait_time + extra_collect:
                self.get_logger().info("下视二次修正采集结束，计算修正后位置")
                self.set_down_camera(False, 0)
                self.accept_down_data = False

                if self.latest_down_offset is not None:
                    front, right = self.latest_down_offset
                    yaw = self.get_yaw_from_pose()
                    cos_y = math.cos(yaw)
                    sin_y = math.sin(yaw)

                    uav_x = self.current_pose.pose.position.x
                    uav_y = self.current_pose.pose.position.y
                    target_x = uav_x + front * cos_y + right * sin_y
                    target_y = uav_y + front * sin_y - right * cos_y
                    self.bomb_precise_xy = (target_x, target_y)

                    self.get_logger().info(
                        f"二次修正偏移: front={front:.4f}, right={right:.4f}, "
                        f"yaw={math.degrees(yaw):.1f}° | "
                        f"无人机=({uav_x:.4f}, {uav_y:.4f}) | "
                        f"修正后目标=({target_x:.4f}, {target_y:.4f})"
                    )
                else:
                    self.get_logger().warning(
                        "二次修正未采集到下视偏移，保持原有精确目标")

                self.bomb_substate = 'STABILIZE'
                self.bomb_align_start_time = None
                
        # ---------- 子状态 4：稳定 N 秒后投弹 ----------
        elif self.bomb_substate == 'STABILIZE':
            px, py = self.bomb_precise_xy
            self.publish_ned_position(px, py, target_height, yaw=self.initial_yaw)

            precise_dist = np.linalg.norm(current_xy - np.array([px, py]))
            is_stable = (precise_dist < pos_tol and height_error < height_tol)

            if is_stable:
                if self.bomb_align_start_time is None:
                    self.bomb_align_start_time = time.time()
                    self.get_logger().info(
                        f"位置稳定 (误差 {precise_dist:.2f}m)，"
                        f"开始 {self.bomb_stabilize_time} 秒稳定计时")
                else:
                    elapsed = time.time() - self.bomb_align_start_time
                    if int(elapsed * 10) % 10 == 0:
                        self.get_logger().info(
                            f"稳定悬停中 {elapsed:.1f}/{self.bomb_stabilize_time} 秒 "
                            f"(误差 {precise_dist:.2f}m)")
                    if elapsed >= self.bomb_stabilize_time:
                        self.get_logger().info(
                            f"稳定悬停 {self.bomb_stabilize_time} 秒达到 "
                            f"(误差 {precise_dist:.2f}m)，执行投弹！")
                        if self.drop_bomb_servo():
                            self.bomb_dropped = True
                            # 投弹成功后进入悬停等待，避免弹药未滑出时无人机移动
                            self.bomb_substate = 'POST_DROP_HOVER'
                            self.bomb_post_drop_start = time.time()
                            self.get_logger().info(
                                f"弹药已发出，保持悬停 "
                                f"{self.bomb_post_drop_hover_time} 秒等待滑出")
                        else:
                            self.get_logger().warning("投弹指令发送失败，保持位置等待重试")
            else:
                if self.bomb_align_start_time is not None:
                    self.get_logger().warning(
                        f"位置偏离 (误差 {precise_dist:.2f}m 超过容差 {pos_tol:.2f}m)，"
                        f"重置稳定计时")
                    self.bomb_align_start_time = None
                    
        # ---------- 子状态 5：投弹后悬停，等待弹药滑出导轨 ----------
        elif self.bomb_substate == 'POST_DROP_HOVER':
            px, py = self.bomb_precise_xy
            self.publish_ned_position(px, py, target_height, yaw=self.initial_yaw)

            elapsed = time.time() - self.bomb_post_drop_start
            if int(elapsed * 10) % 10 == 0:
                self.get_logger().info(
                    f"投弹后悬停中 {elapsed:.1f}/{self.bomb_post_drop_hover_time} 秒")
            if elapsed >= self.bomb_post_drop_hover_time:
                self.get_logger().info(
                    f"投弹后悬停 {self.bomb_post_drop_hover_time} 秒完成，"
                    f"切换到下一目标")
                self.target_index += 1     # ← 新增，悬停完成后才推进索引
                self.bomb_substate = 'APPROACH'
                self.bomb_precise_xy = None
                self.bomb_align_start_time = None
                self.bomb_stabilize_start = None
                self.bomb_rise_start = None
                self.bomb_recheck_start = None
                self.bomb_post_drop_start = None
    
    # ------------------------------------------------------------------
    # 下视搜索：前视未检测到目标时，飞到投弹区中心抬升到指定高度，
    # 用下视相机搜索目标；找到 1 个则两发都投它，2 个以上按原逻辑取 2 个。
    # ------------------------------------------------------------------
    def _start_down_search(self):
        self.down_search_active = True
        self.down_search_mode = True
        self.down_search_start = time.time()
        self.down_search_targets = []
        self.bomb_substate = 'DOWN_SEARCH'
        off = self.get_parameter('drop_zone_center_offset').value
        cx = self.point1_ned[0] + off * math.cos(self.initial_yaw)
        cy = self.point1_ned[1] + off * math.sin(self.initial_yaw)
        self.down_search_center = (cx, cy)
        h = self.get_parameter('down_search_height').value
        self.get_logger().warning(
            f"前视未检测到目标：飞往投弹区中心 ({cx:.2f},{cy:.2f})，"
            f"抬升到 {h:.1f}m 用下视相机搜索")
        self.set_front_camera(False, 0)
        self.set_down_camera(True, self.HIGH_RATE)

    def _handle_down_search(self):
        h = self.get_parameter('down_search_height').value
        timeout = self.get_parameter('down_search_timeout').value
        cx, cy = self.down_search_center
        self.publish_ned_position(cx, cy, h, yaw=self.initial_yaw)

        pos_tol = self.get_parameter('position_tolerance').value
        height_tol = self.get_parameter('height_tolerance').value
        current_xy = np.array([self.current_pose.pose.position.x,
                               self.current_pose.pose.position.y])
        dist = np.linalg.norm(current_xy - np.array([cx, cy]))
        height_err = abs(self.current_pose.pose.position.z - h)
        elapsed = time.time() - self.down_search_start
        arrived = dist < pos_tol and height_err < height_tol

        if int(elapsed * 10) % 10 == 0:
            self.get_logger().info(
                f"[DOWN_SEARCH] {elapsed:.1f}/{timeout:.0f}s 到达={arrived} "
                f"距中心={dist:.2f}m 高度误差={height_err:.2f}m "
                f"已找到={len(self.down_search_targets)}")

        if (arrived and elapsed >= 2.0) or elapsed >= timeout:
            self.down_search_active = False
            self.down_search_mode = False
            self.set_down_camera(False, 0)
            targets = self.down_search_targets
            if len(targets) == 1:
                t = list(targets[0])
                self.bomb_targets_filtered = [t, list(t)]
                self.get_logger().info(
                    f"下视搜索到 1 个目标，两发都投它: ({t[0]:.2f},{t[1]:.2f})")
            elif len(targets) >= 2:
                self.bomb_targets_filtered = sorted(targets, key=lambda p: p[2])[:2]
                self.get_logger().info(
                    f"下视搜索到 {len(targets)} 个目标，取 2 个: "
                    + ", ".join([f"({t[0]:.2f},{t[1]:.2f})" for t in self.bomb_targets_filtered]))
            else:
                self.bomb_targets_filtered = [[cx, cy, 0.5], [cx, cy, 0.5]]
                self.get_logger().warning(
                    f"下视也未搜索到目标，回退到投弹区中心 ({cx:.2f},{cy:.2f}) 投弹")
            self.target_index = 0
            self.bomb_substate = 'APPROACH'
            self.bomb_precise_xy = None
            self.bomb_align_start_time = None
            self.bomb_stabilize_start = None

    def _collect_down_search_targets(self, msg: Detection2DArray):
        if self.current_pose is None or not msg.detections:
            return
        yaw = self.get_yaw_from_pose()
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        uav_x = self.current_pose.pose.position.x
        uav_y = self.current_pose.pose.position.y
        for det in msg.detections:
            if not det.results:
                continue
            front = det.results[0].pose.pose.position.x
            right = det.results[0].pose.pose.position.y
            if math.hypot(front, right) > 25.0:
                continue
            tx = uav_x + front * cos_y + right * sin_y
            ty = uav_y + front * sin_y - right * cos_y
            if not any(math.hypot(tx - t[0], ty - t[1]) < self.target_merge_distance
                       for t in self.down_search_targets):
                self.down_search_targets.append([tx, ty, right])
                self.get_logger().info(
                    f"下视搜索到目标 #{len(self.down_search_targets)}: "
                    f"NED=({tx:.2f},{ty:.2f}) right={right:.2f}")

    def _handle_go_point2(self):
        self.publish_ned_position(self.point2_ned[0], self.point2_ned[1], -self.point2_ned[2], yaw=self.initial_yaw)

        current_xy = np.array([self.current_pose.pose.position.x, self.current_pose.pose.position.y])
        target_xy = self.point2_ned[:2]
        dist = np.linalg.norm(current_xy - target_xy)
        height_error = abs(self.current_pose.pose.position.z - self.target_height)
        pos_tol = self.get_parameter('position_tolerance').value
        height_tol = self.get_parameter('height_tolerance').value

        if dist < pos_tol and height_error < height_tol:
            if self.point2_stable_start_time is None:
                self.point2_stable_start_time = time.time()
                self.get_logger().info(f"已到达 point2，开始稳定悬停（误差：位置 {dist:.2f}m，高度 {height_error:.2f}m）")
            elif time.time() - self.point2_stable_start_time >= 1.0:   # ← 3.0 改为 1.0
                self.get_logger().info("point2 悬停 1 秒完成，开启下视相机，开始侦察区 6 点扫描")
                self.set_front_camera(False, 0)                 # 关闭前视相机
                self.set_down_camera(False, 0)      # 关闭下视相机
                self._start_scout_scan()
                self.state = "SCOUT_SCAN"
                self.point2_stable_start_time = None
        else:
            self.point2_stable_start_time = None
            
        # ------------------------------------------------------------------
    # 侦察区矩形扫描（6 点：3 列 × 2 行）
    # ------------------------------------------------------------------
    def _start_scout_scan(self):
        """计算 3列×2行 割草机式扫描航点，初始化扫描子状态。

        网格：
          - 列偏移（机体左正右负）: -2.5, 0, +2.5
          - 行偏移（机体前向）    : 1.25, 3.75
        所有 6 个悬停点均严格位于矩形 x∈[-4,4], y∈[0,5] 内。
        """
        yaw = self.initial_yaw
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        p2x = self.point2_ned[0]
        p2y = self.point2_ned[1]

        # 扫描网格（3 列 × 2 行）
        xs = [-2.5, 0.0, 2.5]   # 左正右负
        ys = [1.25, 3.75]       # 前偏移

        waypoints = []
        for row, y_off in enumerate(ys):
            # 割草机：偶数行左→右，奇数行右→左
            xs_order = xs if row % 2 == 0 else list(reversed(xs))
            for x_off in xs_order:
                # 机体前向在 NED 中 = ( cos(yaw), sin(yaw) )
                # 机体左向在 NED 中 = (-sin(yaw), cos(yaw) )
                ex = p2x + y_off * cos_y - x_off * sin_y
                ey = p2y + y_off * sin_y + x_off * cos_y
                waypoints.append((ex, ey))

        self.scout_scan_waypoints = waypoints
        self.scout_scan_index = 0
        self.scout_scan_substate = 'GO_TO_POINT'
        self.scout_scan_hover_start = None

        self.get_logger().info(
            f"侦察区扫描初始化: {len(waypoints)} 点 (3列×2行), "
            f"每点悬停 {self.scout_scan_hover_time}s, "
            f"yaw={math.degrees(yaw):.1f}°"
        )
        for i, (ex, ey) in enumerate(waypoints):
            self.get_logger().info(
                f"  扫描点 {i+1:>2}: x={ex:.2f}, y={ey:.2f}")
                
    def _handle_scout_scan(self):
        target_height = self.target_height
        pos_tol = self.get_parameter('position_tolerance').value
        height_tol = self.get_parameter('height_tolerance').value

        # ---------- 所有扫描点已完成 → 返回 point2 ----------
        if not self.scout_scan_waypoints or self.scout_scan_index >= len(self.scout_scan_waypoints):
            self.publish_ned_position(
                self.point2_ned[0], self.point2_ned[1], target_height,
                yaw=self.initial_yaw)
            current_xy = np.array([
                self.current_pose.pose.position.x,
                self.current_pose.pose.position.y])
            dist = np.linalg.norm(
                current_xy - np.array([self.point2_ned[0], self.point2_ned[1]]))
            if dist < pos_tol:
                self.get_logger().info(
                    f"已返回 point2 (误差 {dist:.2f}m)，侦察区扫描结束，开始返航")
                self.set_down_camera(False, 0)
                self.set_front_camera(False, 0)
                self.scout_scan_substate = None
                self.state = "RETURNING"
            return

        # ---------- 当前扫描点 ----------
        ex, ey = self.scout_scan_waypoints[self.scout_scan_index]
        current_xy = np.array([
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y])
        current_z = self.current_pose.pose.position.z
        dist = np.linalg.norm(current_xy - np.array([ex, ey]))
        height_error = abs(current_z - target_height)

        # ---------- 前往扫描点 ----------
        if self.scout_scan_substate == 'GO_TO_POINT':
            self.publish_ned_position(ex, ey, target_height, yaw=self.initial_yaw)
            if dist < pos_tol and height_error < height_tol:
                self.get_logger().info(
                    f"到达扫描点 {self.scout_scan_index + 1}/{len(self.scout_scan_waypoints)} "
                    f"(误差 {dist:.2f}m)，开始悬停 {self.scout_scan_hover_time}s")
                self.scout_scan_substate = 'HOVER'
                self.scout_scan_hover_start = time.time()

        # ---------- 在扫描点悬停 ----------
        elif self.scout_scan_substate == 'HOVER':
            self.publish_ned_position(ex, ey, target_height, yaw=self.initial_yaw)
            elapsed = time.time() - self.scout_scan_hover_start
            if elapsed >= self.scout_scan_hover_time:
                self.get_logger().info(
                    f"扫描点 {self.scout_scan_index + 1} 悬停完成 ({elapsed:.1f}s)")
                self.scout_scan_index += 1
                self.scout_scan_substate = 'GO_TO_POINT'
                self.scout_scan_hover_start = None
    
    def _handle_returning_state(self):
        self._log_pose_info("[返航]")
        self.publish_ned_position(self.home_pos[0], self.home_pos[1], self.home_pos[2], yaw=self.initial_yaw)
        
        current_pos = [
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            self.current_pose.pose.position.z
        ]
        position_error = np.linalg.norm(np.array(self.home_pos[:2]) - np.array(current_pos[:2]))
        height_error = abs(self.home_pos[2] - current_pos[2])
        
        if position_error < 1.0 and height_error < 0.5:
            self.get_logger().info("已到达归位点，准备着陆")
            self.state = 'LANDING'
            self.set_front_camera(False, 0)
            self.set_down_camera(False, 0)
    
    def _send_descend_velocity(self):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.twist.linear.x = 0.0
        cmd.twist.linear.y = 0.0
        cmd.twist.linear.z = 0.05
        self.vel_pub.publish(cmd)
        
    def disarm(self):
        if not self.arm_client.service_is_ready():
            self.get_logger().warning("Arm 服务不可用，无法 disarm")
            return

        req = CommandBool.Request(value=False)
        future = self.arm_client.call_async(req)
        future.add_done_callback(
            lambda f: self.get_logger().info("Disarm 请求已发送")
        )

    def _handle_landing_state(self):
        if self.current_pose is not None:
            x = self.current_pose.pose.position.x
            y = self.current_pose.pose.position.y
            z = self.current_pose.pose.position.z
            self.get_logger().info(f"[降落实时] 位置: ({x:.3f}, {y:.3f}, {z:.3f}) | 模式: {self.current_state.mode} | 解锁: {self.armed}")

        if self._auto_land_set and self.armed:
            if not hasattr(self, '_force_disarm_time'):
                self._force_disarm_time = time.time()
            elif time.time() - self._force_disarm_time > 5.0:
                self.get_logger().warning("AUTO.LAND 超时，强制上锁")
                self.disarm()
                return
        else:
            if hasattr(self, '_force_disarm_time'):
                delattr(self, '_force_disarm_time')

        if not self.in_landing_phase:
            self.set_front_camera(False, 0)
            self.set_down_camera(False, 0)
            self.in_landing_phase = True
            self._auto_land_set = False
            self._landing_stable = False
            self._landing_stable_start = None
            self._low_alt_start_time = None
            self._landing_entry_time = time.time()
            self.get_logger().info("开始精确降落至起飞点(0,0)")

        if not self._landing_stable:
            self.publish_ned_position(0.0, 0.0, 2.0, yaw=self.initial_yaw)

            current_pos = np.array([
                self.current_pose.pose.position.x,
                self.current_pose.pose.position.y,
                self.current_pose.pose.position.z
            ])
            home_pos = np.array([0.0, 0.0, 2.0])
            pos_error = np.linalg.norm(current_pos[:2] - home_pos[:2])
            height_error = abs(current_pos[2] - home_pos[2])

            pos_tol = 2.0
            height_tol = 0.5
            stable_duration = 3.0
            elapsed = time.time() - self._landing_entry_time

            if (pos_error < pos_tol and height_error < height_tol) or elapsed > 30.0:
                if self._landing_stable_start is None:
                    self._landing_stable_start = time.time()
                    self.get_logger().info(f"接近归位点 (pos_err={pos_error:.2f}m, ht_err={height_error:.2f}m)，开始稳定计时")
                elif time.time() - self._landing_stable_start > stable_duration:
                    self._landing_stable = True
                    self.get_logger().info(f"归位点已稳定或超时，开始下降")
            else:
                self._landing_stable_start = None
            return

        current_z = self.current_pose.pose.position.z
        self.get_logger().info(f"[降落] 当前高度 {current_z:.2f}m")

        if current_z > 1.0:
            self.publish_ned_position(0.0, 0.0, 0.5, yaw=self.initial_yaw)
            self._low_alt_start_time = None
            return

        if current_z > 0.5:
            self.publish_ned_position(0.0, 0.0, 0.0, yaw=self.initial_yaw)
            self._low_alt_start_time = None
            return

        if not self._auto_land_set:
            self.get_logger().info("高度低于0.5m，尝试切换到 AUTO.LAND")
            if self.set_mode(self.land_mode):
                self._auto_land_set = True
                self._landing_entry_time = time.time()
                return
            else:
                self.get_logger().warning("切换到 AUTO.LAND 失败，继续使用位置控制")

            if self._auto_land_set:
                if not self.armed:
                    self.get_logger().info("AUTO.LAND 完成，已上锁")
                    self._handle_landing_complete()
                    return
                if time.time() - self._landing_entry_time > 10.0:
                    self.get_logger().warning("AUTO.LAND 超时，强制 disarm")
                    self.disarm()
                return

        self.publish_ned_position(0.0, 0.0, 0.0, yaw=self.initial_yaw)

        if current_z < 0.1:
            if self.armed:
                self.get_logger().info("高度低于0.1m，执行上锁")
                self.disarm()
            return
       
        if self._low_alt_start_time is None:
            self._low_alt_start_time = time.time()
            self.get_logger().info(f"进入低空区域，高度 {current_z:.2f}m，开始计时")
        elif time.time() - self._low_alt_start_time > 3.0:
            if self.armed:
                self.get_logger().info("低空悬停超时，强制上锁")
                self.disarm()
            return
        
        # 注意：本函数目前不再从 _basic_checks 自动触发
    # 仅保留作为将来手动安全兜底使用
    def _trigger_emergency_landing(self):
        self.get_logger().error("触发紧急着陆程序!")
        self.in_landing_phase = True
        if not self.set_mode(self.land_mode):
            self.get_logger().warning("无法设置着陆模式,尝试直接下降")
            self.publish_position_setpoint(
                self.current_pose.pose.position.x,
                self.current_pose.pose.position.y,
                0.0
            )

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = UavTracker()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("节点被用户中断")
        else:
            print("节点被用户中断")
    except Exception as e:
        if node:
            node.get_logger().error(f"节点初始化异常: {e}")
        else:
            import logging
            logging.basicConfig(level=logging.ERROR)
            logging.error(f"节点初始化失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
