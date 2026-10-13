# Bien moi truong chung cho workspace DemoDiffusion (source file nay).
# WS: thu muc chua thu vien ngoai (Arena, openpi, cache) - mac dinh tren SSD cua server lab.
export WS="${DD_WS:-$HOME/ros2_ws/demodiffusion_ws}"
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="$WS/.local/bin:$PATH"
export UV_CACHE_DIR="$WS/.cache/uv"
export OPENPI_DATA_HOME="$WS/.cache/openpi"     # openpi luu checkpoint tai day
export HF_HOME="$WS/.cache/huggingface"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export UV_HTTP_TIMEOUT=300
# KHONG dung CUDA_VISIBLE_DEVICES cho Isaac (renderer RTX loi); openpi (JAX) thi dung duoc
unset CUDA_VISIBLE_DEVICES

# GPU=auto: chon GPU con trong nhieu VRAM nhat
pick_gpu() {
    if [ "${GPU:-auto}" = "auto" ]; then
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1
    else
        echo "$GPU"
    fi
}
