#!/bin/bash
# ============================================================
# uav_ws 统一环境脚本 (Jetson Nano / ROS 2 Eloquent)
# 用法: source /home/lin/uav_ws/uav_env.sh
# ============================================================

# 1. ROS 2 Eloquent
source /opt/ros/eloquent/setup.bash

# 2. mavros_msgs (本地编译，含 libmavconn)
source /mnt/sda/mavros_ws/install/setup.bash

# 3. uav_ws 本工作空间 (uav_tracker + yolov5_ros2)
source /home/lin/uav_ws/install/setup.bash

# 4. YOLOv5 + PyTorch + TensorRT 环境 (torch/cuda/pylibs)
source /mnt/sda/yolov5_env.sh

# 5. matplotlib 后端 (无显示环境)
export MPLBACKEND=Agg

echo "✅ uav_ws 环境已加载"
echo "   启动检测(前视): ros2 run yolov5_ros2 yolo_detect_2d_014106"
echo "   启动检测(下视): ros2 run yolov5_ros2 yolo_detect_2d_down"
echo "   启动跟踪节点:   ros2 run uav_tracker uav_tracker"
