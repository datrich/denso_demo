#!/usr/bin/env bash
# Cai dat lan dau tren Linux (Ubuntu 24.04 + ROS 2 Jazzy apt + Isaac Sim 6.x):
#   venv (dung chung rclpy/numpy cua he thong) + usd-core + imageio-ffmpeg, tai 5 vat HOPE.
#   bash ur5e_ros/linux/setup.sh
set -eo pipefail  # khong dung -u: setup.bash cua ROS tham chieu bien chua dat
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [ ! -d "$VENV" ]; then
    python3 -m venv --system-site-packages "$VENV"
    source "$VENV/bin/activate"
fi
python -m pip install -q --upgrade pip
python -m pip install -q usd-core imageio-ffmpeg
python - <<'EOF'
import imageio_ffmpeg, pxr, rclpy, numpy
print("ffmpeg:", imageio_ffmpeg.get_ffmpeg_exe())
print("usd-core", pxr.Usd.GetVersion(), "| numpy", numpy.__version__, "| rclpy OK")
EOF
cd "$REPO_ROS"
python tools/bop_to_usd.py --fetch
python -m pytest -q -p no:launch_testing -p no:launch_ros tests 2>/dev/null || python -m unittest 2>/dev/null || true
echo "Xong. Chay sim: bash ur5e_ros/linux/sim_headless.sh"
