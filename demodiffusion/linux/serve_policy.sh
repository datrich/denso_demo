#!/usr/bin/env bash
# Server pi0-DROID jointpos co ho tro DemoDiffusion (openpi, JAX).
#   GPU=auto PORT=8000 bash demodiffusion/linux/serve_policy.sh
# JAX mac dinh chiem 75% VRAM -> gioi han XLA_MEM (mac dinh 0.45) de Isaac Sim chay chung GPU.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GPU_ID="$(pick_gpu)"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM:-0.45}"
echo "openpi server: GPU $GPU_ID, cong ${PORT:-8000}, XLA_MEM $XLA_PYTHON_CLIENT_MEM_FRACTION"
CUDA_VISIBLE_DEVICES="$GPU_ID" exec nice -n "${NICE:-10}" "$WS/openpi/.venv/bin/python" \
    "$REPO/demodiffusion/openpi_server/serve_demodiffusion.py" --port "${PORT:-8000}" "$@"
