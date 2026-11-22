# cuadc-ros2-px4

基于 **ROS 2 (Eloquent) + PX4 (MAVROS) + YOLOv5/TensorRT** 的无人机自主任务系统，面向 **NVIDIA Jetson Nano** 部署。

> ⚠️ **本仓库为 Gazebo 仿真验证版本，尚未在 Jetson Nano 实机测试。** 实机测试版本（Jetson Orin Nano + 雷讯 X7+ 飞控）及高性能电脑 PX4+Gazebo 仿真版本尚未上传。实机需配备**高精度 RTK 定位模块**与**定高激光雷达**。使用前请务必阅读下方「[项目状态与适用范围](#项目状态与适用范围)」。

包含两个 ROS 2 包：

| 包 | 作用 |
|---|---|
| `uav_tracker` | 自主任务主控（状态机）：起飞、搜索、目标跟踪、投弹、返航、降落 |
| `yolov5_ros2` | YOLOv5 目标检测节点（前视 / 下视），支持 PyTorch 与 TensorRT 推理 |

---

## 项目状态与适用范围

> ⚠️ **请务必先读这一节再使用本仓库。**

本仓库上传的源码是**面向 Gazebo 仿真的版本**，仅用于算法验证与仿真联调。请注意以下几点：

- **本仓库版本仅通过 Gazebo 仿真测试**，**尚未在 Jetson Nano 实机上测试**。
- 本项目实际做过**实机测试**的是另一套部署版本：**Jetson Orin Nano + 雷讯 X7+ 飞控**。
- 以下两个版本**尚未上传**到本仓库：
  1. **高性能电脑上用于 PX4 固件 + Gazebo 仿真的版本**
  2. **Jetson Orin Nano 实机部署版本**
- 因此，本仓库代码**不能直接用于实机飞行**，实机使用请参考上述未上传的版本。

### ⚠️ 使用前必须修改的参数（相机标定）

源码中的**相机内参**和**相机位置外参**均为示例/仿真值，**必须根据实际硬件修改**后才能使用：

| 类型 | 文件 | 需修改内容 |
|---|---|---|
| 前视相机内参 | `src/yolov5_ros2/config/camera_info.yaml` | 焦距 `fx/fy`、主点 `cx/cy`、畸变系数 |
| 下视相机内参 | `src/yolov5_ros2/config/down_camera_info.yaml` | 焦距 `fx/fy`、主点 `cx/cy`、畸变系数 |
| 相机外参（安装位置/姿态） | `uav_tracker` 节点内（相机相对机体系的位置/朝向） | 相机相对飞控/机体的平移与旋转 |

> 内参错误会导致检测框反投影到 3D 位置的偏差，外参错误会导致目标定位/跟踪坐标错误，直接影响投弹与跟踪精度。

### ⚠️ 硬件要求（实机必须）

本项目实机运行时对定位与高度感知要求较高，**必须配备以下硬件**：

- **高精度 RTK 定位模块**：系统依赖高精度全局定位，普通 GPS 精度不足，需 RTK 级定位（厘米级）才能保证目标定位与投弹精度。
- **定高激光雷达（LiDAR）**：用于高精度定高/测距，保证低空飞行与投弹阶段的高度准确性，仅靠气压计/普通 GPS 高度无法满足要求。

### 📹 实机测试视频

实机测试（**Jetson Orin Nano + 雷讯 X7+ 飞控**）视频：夜间操场，地面放置白色桶状目标（模型中的 `tong` 类），无人机自主搜索、跟踪并投弹。

[▶ 点击查看实机测试视频](docs/media/real_flight_test.mp4)

> 注：本视频为 Jetson Orin Nano 实机版本测试，对应源码尚未上传本仓库；本仓库代码为 Gazebo 仿真版本。

---

## 系统架构

```
                    ┌──────────────────────┐
   PX4 飞控 ◄──────►│  MAVROS  (mavros_node)│
                    └──────────┬───────────┘
                               │ /mavros/*
                    ┌──────────▼───────────┐
                    │   uav_tracker        │  状态机主控
                    │  (mission control)   │
                    └──┬───────────────┬───┘
      /front_camera/*  │               │  /down_camera/*
                       ▼               ▼
              ┌────────────────┐ ┌────────────────┐
              │ yolo_detect_2d │ │yolo_detect_2d_ │
              │   (前视检测)    │ │  down (下视)   │
              └───────┬────────┘ └───────┬────────┘
                      │ /detections_3d    │ /down_detections_relative
                      └────────┬──────────┘
                               ▼
                         uav_tracker 消费检测结果
```

### 关键话题

| 方向 | 话题 | 类型 |
|---|---|---|
| 订阅 | `/mavros/state` | mavros_msgs/State |
| 订阅 | `/mavros/local_position/pose` | geometry_msgs/PoseStamped |
| 订阅 | `/mavros/global_position/compass_hdg` | std_msgs/Float64 |
| 订阅 | `/detections_3d` | vision_msgs/Detection2DArray（前视） |
| 订阅 | `/down_detections_relative` | vision_msgs/Detection2DArray（下视） |
| 发布 | `/mavros/setpoint_position/local` | geometry_msgs/PoseStamped |
| 发布 | `/mavros/setpoint_raw/local` | mavros_msgs/PositionTarget |
| 发布 | `/mavros/setpoint_velocity/cmd_vel` | geometry_msgs/TwistStamped |
| 发布 | `/front_camera/activate`、`/front_camera/rate` | std_msgs/Bool、Int16 |
| 发布 | `/down_camera/activate`、`/down_camera/rate` | std_msgs/Bool、Int16 |

### 状态机流程

```
INIT → PREFLIGHT → YAW_SAMPLING → ASCENDING → SEARCHING
   → GO_POINT1 → WAIT_BOMB_DETECTIONS → GO_BOMB_TARGETS
   → GO_POINT2 → SCOUT_SCAN → TRACKING → RETURNING → LANDING
   → MISSION_COMPLETE           （异常 → SAFE）
```

---

## 环境依赖

> 本项目在 **Jetson Nano (JetPack 4.6.6 / L4T R32.7.6, Ubuntu 18.04)** + **ROS 2 Eloquent** 上验证通过。

- **ROS 2 Eloquent**（`/opt/ros/eloquent`）
- **MAVROS 2.2.0**（源码编译，仅支持 Eloquent 的版本；含 `libmavconn` / `mavros_msgs` / `mavros` / `mavros_extras`）
- **vision_msgs**：`sudo apt install ros-eloquent-vision-msgs`
- **YOLOv5 运行环境**：PyTorch 1.10 (Qengineering sm_53 wheel) + TensorRT 8.2 + pycuda + onnx + numpy 1.16.6
  （详见 `yolov5_env.sh` 相关配置；无需在此仓库内）
- 相机：Intel RealSense D435i（`realsense2_camera`）或 USB 相机（`usb_cam`）

### 模型文件

仓库仅包含 `src/yolov5_ros2/config/best.pt`（YOLOv5 v7.0，2 类：`' '` 和 `'tong'`=桶）。

`best.onnx` 与 `best_fp16.engine` **未上传**（体积大、可由 best.pt 重新生成）：

```bash
# 1. 导出 ONNX
python3 /mnt/sda/yolov5-7.0/export.py \
    --weights src/yolov5_ros2/config/best.pt \
    --include onnx --img 640 --opset 12

# 2. 转 TensorRT FP16 engine（约 7 分钟）
python3 /mnt/sda/yolov5-6.0/onnx2trt.py \
    best.onnx best_fp16.engine fp16
```

> 检测节点默认加载 `config/best_fp16.engine`；如需用 PyTorch，把节点里的 `model_path` 改回 `best.pt`。

---

## 编译

```bash
# 1. 先编译 mavros（若尚未编译），并 source 其 install
source /mnt/sda/mavros_ws/install/setup.bash

# 2. 编译本工作空间
cd ~/uav_ws
colcon build --packages-select uav_tracker yolov5_ros2 \
    --symlink-install
```

---

## 运行

```bash
# 加载统一环境（ROS + mavros + 本工作空间 + YOLOv5/TensorRT）
source ~/uav_ws/uav_env.sh

# 终端 1：MAVROS（连接飞控，按实际串口/波特率修改）
ros2 run mavros mavros_node --ros-args -p fcu_url:=/dev/ttyUSB0:921600

# 终端 2：前视检测
ros2 run yolov5_ros2 yolo_detect_2d

# 终端 3：下视检测
ros2 run yolov5_ros2 yolo_detect_2d_down

# 终端 4：主控状态机
ros2 run uav_tracker uav_tracker
```

> **可执行名注意**：`yolo_detect_2d`（不是源文件名 `yolo_detect_2d_014106`）。

### 主控参数（可在启动时覆盖）

```bash
ros2 run uav_tracker uav_tracker --ros-args \
    -p target_height:=1.0 \
    -p max_speed:=1.0 \
    -p home_position:="[0.0, 0.0, 2.0]"
```

常用参数：`target_height`、`bomb_scan_height_offset`、`bomb_recheck_hover_time`、
`bomb_stabilize_time`、`bomb_post_drop_hover_time`、`k_p/k_i/k_d`、`arm_timeout`、`max_speed`、`home_position`。

---

## 目录结构

```
uav_ws/
├── src/
│   ├── uav_tracker/          # 自主任务主控（状态机）
│   │   ├── uav_tracker/uav_tracker.py
│   │   ├── package.xml
│   │   └── setup.py
│   └── yolov5_ros2/          # YOLOv5 检测节点
│       ├── config/
│       │   ├── best.pt                 # 模型权重（2 类）
│       │   ├── camera_info.yaml        # 前视相机标定
│       │   └── down_camera_info.yaml   # 下视相机标定
│       ├── yolov5_ros2/
│       │   ├── yolo_detect_2d_014106.py   # 前视检测节点
│       │   ├── yolo_detect_2d_down.py     # 下视检测节点
│       │   ├── yolov5_shim.py             # YOLOv5 v7 兼容垫片
│       │   └── cv_tool.py
│       ├── package.xml
│       └── setup.py
└── uav_env.sh                # 统一环境加载脚本
```

---

## 已知注意事项

- **⚠️ 实机使用前必读**：见上文「[项目状态与适用范围](#项目状态与适用范围)」。本仓库仅通过 Gazebo 仿真测试，未在 Jetson Nano 实机验证；相机内参/外参需按实际硬件修改。
- **⚠️ 硬件要求**：实机需配备**高精度 RTK 定位模块**和**定高激光雷达**，普通 GPS 与气压计精度不足。
- **Python 3.6 / GCC 7.5**：Eloquent 环境较老，`yolov5_shim.py` 用于让 YOLOv5 v7.0 模型在 v6.0 运行时下加载。
- **TensorRT engine 与 GPU 绑定**：`best_fp16.engine` 只能在构建它的机器（Jetson Nano, sm_53）上使用，换机需重新生成。
- **内存**：双检测节点同时运行内存吃紧（Jetson Nano 仅 4G），建议配置 swap 或降低输入尺寸。
- **mavros 版本**：必须用 2.2.0（Eloquent 对应版本），2.3+ 需要 Foxy 及以上。

---

## License

[Apache-2.0](LICENSE)
