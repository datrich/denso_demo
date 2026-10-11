#!/usr/bin/env bash
# Quay video demo pick & place tu camera toan canh (sim headless phai dang chay, UR5E_DEMO_CAM=1):
#   bash ur5e_ros/linux/record_demo.sh            # -> recordings/<thoi gian>/pick_place.mp4 + anh PNG
#   bash ur5e_ros/linux/record_demo.sh grasp_test # quay tool khac
set -eo pipefail  # khong dung -u: setup.bash cua ROS tham chieu bien chua dat
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
tool="${1:-pick_place_demo}"
OUT="$REPO/recordings/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
cd "$REPO_ROS"

python tools/record_camera.py --out "$OUT/${tool}.mp4" --snapshot-every 5 > "$OUT/recorder.log" 2>&1 &
REC=$!
sleep 3
python "tools/${tool}.py" 2>&1 | tee "$OUT/${tool}.log" || true
sleep 2
kill -INT "$REC" 2>/dev/null || true
wait "$REC" || true
cat "$OUT/recorder.log"
ls -la "$OUT"
