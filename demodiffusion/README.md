# DemoDiffusion trong Isaac Lab Arena

Tái hiện [DemoDiffusion](https://arxiv.org/abs/2506.20668) (Park, Bharadhwaj, Tulsiani, ICRA 2026) trong mô phỏng: robot học làm một task từ **một** demo của người, nhờ policy diffusion tổng quát đã huấn luyện sẵn.

**Ý tưởng của bài báo:**
1. Retarget quỹ đạo tay người thành quỹ đạo đầu cuối của robot, theo công thức hình học.
2. Thêm nhiễu Gaussian vào quỹ đạo đó tới mức s\*. Sau đó để policy tổng quát (π0) **khử nhiễu từ s\* về 0** theo vòng kín:

   `a_{s*} = s*·ε + (1 − s*)·â`

3. Kết quả: hành động bám theo ý đồ trong demo, đồng thời nằm trong phân bố hành động khả thi của robot.

Giai đoạn này dùng đúng cấu hình của bài báo:
- **Robot:** DROID (Franka + Robotiq 2F-85, camera ngoài và camera cổ tay), điều khiển 15 Hz.
- **Policy nền:** **π0-DROID**. Bài báo dùng `pi0_droid` (vận tốc khớp) trên robot thật. Trong sim, mình dùng `pi0_droid_jointpos`, là checkpoint π0-DROID xuất vị trí khớp do Physical Intelligence phát hành cho đánh giá mô phỏng, và cũng là bản Arena dùng.

UR5e + 2F-85 là bước tiếp theo.

## Thành phần

| File | Vai trò |
|---|---|
| [`openpi_server/serve_demodiffusion.py`](openpi_server/serve_demodiffusion.py) | Server openpi. Nhận thêm `retargeted_actions` (H×8) và `time` (s\*). Quỹ đạo retarget được đưa qua đúng chuỗi biến đổi đầu vào của policy (delta so với trạng thái, chuẩn hóa), trộn nhiễu theo công thức trên, rồi tích phân flow của π0 từ s\* về 0 (tổng cộng 20 bước như bài báo). Không sửa mã openpi: `Pi0.sample_actions` được vá lúc chạy để nhận `start_time`. |
| [`demodiffusion/hand_demo.py`](demodiffusion/hand_demo.py) | "Demo người" tổng hợp, 21 keypoint kiểu MANO, dựng từ quỹ đạo của expert. Có thêm các sai lệch kiểu người: lệch cố định, trôi chậm, rung keypoint (giống sai số HaMeR), lệch hướng, nắm/nhả sớm hoặc muộn. Kèm hàm retarget dùng **đúng công thức** trong `deploy/preprocess.py` của tác giả: EMA α = 0.1, wrist / thumb / trung bình các ngón, offset 6.2 cm, tay kẹp đóng khi độ mở < 80%. |
| [`demodiffusion/franka_kinematics.py`](demodiffusion/franka_kinematics.py) | FK/IK cho Franka (DH, DLS có giới hạn khớp). Khoảng cách link8 → base_link của Robotiq được hiệu chỉnh từ sim. |
| [`arena/tasks.py`](arena/tasks.py) | Danh sách task Arena tương tự task trong bài báo, cùng expert gắp–đặt có thông tin đặc quyền (dùng để sinh demo). |
| [`arena/run_demodiffusion.py`](arena/run_demodiffusion.py) | `--mode record`: ghi demo người tổng hợp. `--mode eval`: so sánh π0 / kinematic retargeting / DemoDiffusion (s\* = 0.2, 0.4) trên các cảnh test, kèm video và `results_<task>.json`. |

## Task

| DemoDiffusion | Arena (RoboLab, robot DROID) | Prompt |
|---|---|---|
| Pick and place the banana | `banana_on_plate` | Pick up the banana and place it on the plate. |
| Pick and place the bowl | `bowl_in_bin` | Pick up the bowl and place it in the grey bin. |
| Pick up the teddy bear | `toy_in_bin` (tượng đồ chơi) | place the lizard figurine into the grey bin on the table |

## Chạy trên server Linux

```bash
bash demodiffusion/linux/setup_ws.sh                     # uv, Arena (+Isaac Lab, Isaac Sim 6.1 pip), openpi, checkpoint 12 GB
GPU=auto bash demodiffusion/linux/serve_policy.sh &      # server pi0 / DemoDiffusion (cổng 8000)
bash demodiffusion/linux/run_task.sh banana_on_plate record
bash demodiffusion/linux/run_task.sh banana_on_plate eval --episodes 10 --methods pi0 kinematic dd0.2 dd0.4
```

- Thư viện ngoài nằm trong `$WS = ~/ros2_ws/demodiffusion_ws` (SSD), không đưa vào git.
- JAX bị giới hạn 45% VRAM (`XLA_MEM`), để Isaac Sim chạy chung GPU.
