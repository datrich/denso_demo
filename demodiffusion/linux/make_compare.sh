#!/usr/bin/env bash
# Ghep video so sanh pi0 / kinematic / DemoDiffusion (luoi 2x2, cung 1 episode = cung canh test) + chep ket qua
# vao thu muc de tai ve:  bash demodiffusion/linux/make_compare.sh [~/demodiffusion_videos]
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
DST="${1:-$HOME/demodiffusion_videos}/$(date +%Y%m%d_%H%M)"
mkdir -p "$DST"
FF="$("$WS/IsaacLab-Arena/.venv/bin/python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
for D in "$WS"/results/*/; do
    task=$(basename "$D"); task=${task%_clean}
    cp "$D"/results_*.json "$DST/" 2>/dev/null || true
    for f in "$D"/demo_*_expert.mp4; do [ -f "$f" ] && cp "$f" "$DST/$(basename "$D")_$(basename "$f")"; done
    for ep in 0 1; do
        V=(); L=()
        for m in pi0 kinematic dd0.2 dd0.4; do
            f="$D/${task}_${m}_ep${ep}.mp4"; [ -f "$f" ] && { V+=("$f"); L+=("$m"); }
        done
        [ ${#V[@]} -eq 4 ] || continue
        IN=(); FL=""
        for k in 0 1 2 3; do
            IN+=(-i "${V[$k]}")
            # moi khung da co nhan ten phuong phap (ve luc ghi); keo dai video ngan (ket thuc som khi thanh cong)
            FL+="[$k:v]scale=1280:-2,tpad=stop_mode=clone:stop_duration=30[v$k];"
        done
        "$FF" -nostdin -loglevel error -y "${IN[@]}" -filter_complex \
            "${FL}[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:shortest=0[o]" -map "[o]" \
            -c:v libx264 -crf 24 -preset veryfast -pix_fmt yuv420p "$DST/$(basename "$D")_so_sanh_ep${ep}.mp4" < /dev/null \
            || echo "khong ghep duoc $D ep$ep"
    done
done
ls -la "$DST"
