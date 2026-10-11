#!/usr/bin/env bash
# Quay video demo pick & place tu camera toan canh (sim headless phai dang chay, UR5E_DEMO_CAM=1):
#   bash ur5e_ros/linux/record_demo.sh                         # -> recordings/<thoi gian>/
#   RECORD_DIR=~/denso_demo_videos bash ur5e_ros/linux/record_demo.sh   # thu muc rieng (de tai qua AnyDesk)
#   bash ur5e_ros/linux/record_demo.sh grasp_test              # quay tool khac
# Ket qua: <tool>.mp4 (1280x720), <tool>_small.mp4 (960p, nen manh de tai qua mang cham), anh PNG moi 5 s, log
set -eo pipefail  # khong dung -u: setup.bash cua ROS tham chieu bien chua dat
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
tool="${1:-pick_place_demo}"
OUT="${RECORD_DIR:-$REPO/recordings}/$(date +%Y%m%d_%H%M%S)_${UR5E_OBJECTS:-hope}"
mkdir -p "$OUT"
cd "$REPO_ROS"

python tools/record_camera.py --out "$OUT/${tool}.mp4" --snapshot-every 5 > "$OUT/recorder.log" 2>&1 < /dev/null &
REC=$!
sleep 3
python "tools/${tool}.py" < /dev/null 2>&1 | tee "$OUT/${tool}.log" || true
sleep 2
kill -INT "$REC" 2>/dev/null || true
wait "$REC" || true
cat "$OUT/recorder.log"

FF=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" < /dev/null)
nice -n 10 "$FF" -nostdin -loglevel error -y -i "$OUT/${tool}.mp4" -vf scale=960:-2 -c:v libx264 -crf 30 \
    -preset veryfast "$OUT/${tool}_small.mp4" < /dev/null || echo "khong nen duoc video"
echo "== ket qua: $OUT"
ls -la "$OUT" | grep -v "\.png$"
echo "anh PNG: $(ls "$OUT"/*.png 2>/dev/null | wc -l)"
