#!/usr/bin/env bash
# Chay Isaac Sim HEADLESS (khong cua so, khong dung toi man hinh dang dung) tren 1 GPU rieng.
#   GPU=1 bash ur5e_ros/linux/sim_headless.sh          # mac dinh GPU 1, co camera toan canh
# Bien: GPU, UR5E_DEMO_CAM (1), UR5E_OBJECTS (hope|cube), ISAAC_ROOT, ISAAC_PORTABLE_ROOT, NICE (10)
set -eo pipefail  # khong dung -u: setup.bash cua ROS tham chieu bien chua dat
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export UR5E_DEMO_CAM="${UR5E_DEMO_CAM:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
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
    --/renderer/multiGpu/enabled=false --/renderer/activeGpu=0 \
    --/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge \
    --exec "$REPO_ROS/isaac/ur5e_scene.py" "$@" 2>&1 | tee "$LOG"
