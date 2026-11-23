# 仿真版本（PX4 + Gazebo）

本目录收录**高性能电脑上的 PX4 + Gazebo 仿真验证版本**，与仓库根目录的 Jetson 实机版本分开存放。

## 目录说明

### `uav_ws_sim/` — 仿真版 ROS 2 工作空间源码

仿真环境下的 `uav_ws/src`，与实机版的主要差异：

- 检测节点为 `yolov5_ros2/yolov5_ros2/yolo_detect_2d_front.py`（前视）+ `yolo_detect_2d_down.py`（下视）
- 模型为 PyTorch `best.pt`（未转 TensorRT engine，仿真电脑算力充足无需 TRT）
- 无 `yolov5_shim.py`（直接用 pip 版 yolov5 加载）
- `config/` 含 `camera_info.yaml` / `down_camera_info.yaml` 相机内参

主控 `uav_tracker/uav_tracker/uav_tracker.py` 为状态机（搜索 / 跟踪 / 投弹）。

### `px4_gz_models/` — Gazebo (gz) 仿真模型库定制部分

从 `PX4-Autopilot/Tools/simulation/gz`（PX4-gazebo-models 子模块）中提取的**本地新增/修改**内容：

**新增模型**（`models/`）
- `bomb1` `bomb2` `bomb3` — 投弹任务用的炸弹模型
- `drop_cam` — 投放相机
- `recon_cam` — 侦察相机
- `mono_cam_1` — 单目相机
- `overview_cam` — 全局俯视相机
- `playground_top` — 场地顶部模型

**修改模型**（`models/`）
- `x500_depth/model.sdf` — 在 x500_depth 机体上挂载相机 / 投弹机构

**世界文件**（`worlds/`）
- `default.sdf` — 默认仿真世界（含场地与目标布置）

**飞控机型**（`airframes/`）
- `4002_gz_x500_depth` — PX4 机型配置，含 `NAV_DLL_ACT 0` 等仿真参数调整

## 还原方式

将 `px4_gz_models/models/*` 复制到 `PX4-Autopilot/Tools/simulation/gz/models/`，
`worlds/default.sdf` 覆盖到 `worlds/`，
`airframes/4002_gz_x500_depth` 复制到 `ROMFS/px4fmu_common/init.d-posix/airframes/`。

`uav_ws_sim/src` 放入 ROS 2 工作空间 `src/` 后 `colcon build` 即可。

> 注：仿真版本尚未在实机上测试；实机版本见仓库根目录。
