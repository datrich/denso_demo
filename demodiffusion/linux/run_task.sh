#!/usr/bin/env bash
# Ghi demo "nguoi" tong hop va/hoac danh gia pi0 / kinematic / DemoDiffusion cho 1 task (Isaac Lab Arena, headless).
#   bash demodiffusion/linux/run_task.sh banana_on_plate record
#   bash demodiffusion/linux/run_task.sh banana_on_plate eval --episodes 10 --methods pi0 kinematic dd0.2 dd0.4
# Bien: GPU (auto|0|1), DD_OUT (thu muc ket qua, mac dinh $WS/results/<task>), PORT (server openpi)
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
TASK="$1"; MODE="$2"; shift 2
GPU_ID="$(pick_gpu)"
cd "$WS/IsaacLab-Arena"
PY=.venv/bin/python
ENV_SPEC=$(cd "$REPO/demodiffusion" && "$WS/IsaacLab-Arena/$PY" -c "
import sys; sys.path[:0] = ['.', 'arena']
from tasks import TASKS; print(TASKS['$TASK']['env_spec'])")
OUT="${DD_OUT:-$WS/results/$TASK}"
mkdir -p "$OUT"
export FFMPEG="$($PY -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
echo "task=$TASK mode=$MODE env=$ENV_SPEC GPU=$GPU_ID out=$OUT"
exec nice -n "${NICE:-10}" "$PY" "$REPO/demodiffusion/arena/run_demodiffusion.py" --mode "$MODE" --task "$TASK" \
    --env_spec "$ENV_SPEC" --device "cuda:$GPU_ID" --out "$OUT" --ffmpeg "$FFMPEG" \
    --policy_port "${PORT:-8000}" --seed "${SEED:-$([ "$MODE" = record ] && echo 0 || echo 100)}" "$@"
