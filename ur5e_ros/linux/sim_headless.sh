#!/usr/bin/env bash
# Chay Isaac Sim HEADLESS (khong cua so, khong dung toi man hinh dang dung) tren 1 GPU rieng.
#   bash ur5e_ros/linux/sim_headless.sh                 # GPU=auto: chon GPU trong nhat; GPU=1 de ep
#   UR5E_OBJECTS=itodd bash ur5e_ros/linux/sim_headless.sh   # chi tiet co khi thay do gia dung
# Bien: GPU (auto|0|1), UR5E_DEMO_CAM (1), UR5E_OBJECTS (hope|itodd|cube), ISAAC_ROOT, ISAAC_PORTABLE_ROOT, NICE (10)
set -eo pipefail  # khong dung -u: setup.bash cua ROS tham chieu bien chua dat
GPU="${GPU:-auto}"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# GPU=auto: chon GPU con trong nhieu VRAM nhat luc khoi dong (may dung chung)
if [ "$GPU" = "auto" ]; then
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
    echo "GPU=auto -> chon GPU $GPU (nhieu VRAM trong nhat)"
fi

export UR5E_DEMO_CAM="${UR5E_DEMO_CAM:-1}"
# KHONG dat CUDA_VISIBLE_DEVICES: RTX renderer phai ghep GPU Vulkan voi GPU CUDA cung chi so,
# che bot CUDA se loi "No device could be created". Chon GPU bang activeGpu / cudaDevice.
unset CUDA_VISIBLE_DEVICES
# Thu muc cau hinh / cache / log rieng cho Kit (khong dung chung voi Isaac Sim cua nguoi khac)
PORTABLE="${ISAAC_PORTABLE_ROOT:-$REPO/.isaac_portable}"
mkdir -p "$PORTABLE" "$REPO/logs"
# Thu vien CUDA kem Isaac Sim (giong ~/run_isaac.sh tren may lab)
CUDA_LIB="$ISAAC_ROOT/exts/isaacsim.pip.nv/pip_prebundle/nvidia/cuda_runtime/lib"
[ -d "$CUDA_LIB" ] && export LD_LIBRARY_PATH="$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset DISPLAY

LOG="$REPO/logs/sim_$(date +%Y%m%d_%H%M%S).log"
echo "Isaac Sim headless: GPU $GPU, log $LOG"
cd "$ISAAC_ROOT"
nice -n "${NICE:-10}" ./isaac-sim.sh --no-window \
    --portable-root "$PORTABLE" \
    --/renderer/multiGpu/enabled=false --/renderer/activeGpu="$GPU" --/physics/cudaDevice="$GPU" \
    --/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge \
    --exec "$REPO_ROS/isaac/ur5e_scene.py" "$@" 2>&1 | tee "$LOG"
