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

## Kết quả (Isaac Lab Arena, server lab, 2026-10-14)

**Thiết lập đánh giá:**
- Mỗi ô là 10 episode.
- **Cảnh test** = cảnh lúc ghi demo, nhưng vật cần gắp bị xê dịch trong bán kính 4 cm và xoay ±20°. Như bài báo, vật được đặt ngẫu nhiên trong vùng dán băng keo, và mọi phương pháp gặp **cùng một biến thể**.
- **Demo người** tổng hợp với nhiễu "realistic": lệch cố định 1 cm, trôi 1 cm, rung keypoint 4 mm, lệch hướng 5°, nắm/nhả sớm hoặc muộn ±3 khung.
- **Độ dài episode** = độ dài demo (16.5 s) + 2 s.

| Task (Arena) | Tương ứng trong bài báo | π0 | Kinematic | DemoDiffusion s\*=0.2 | DemoDiffusion s\*=0.4 |
|---|---|---|---|---|---|
| `banana_on_plate` | Pick & place banana | 60% | 80% | **90%** | 80% |
| `bowl_in_bin` | Pick & place bowl | 0% | 40% | 30% | **70%** |
| `toy_in_bin` | Pick up teddy bear | 0% | 60% | 50% | **70%** |
| **Trung bình** | | 20% | 60% | 57% | **73%** |

Tham chiếu từ bài báo (robot thật, cùng 3 task): π0 0 / 0 / 0%, kinematic 70 / 70 / 40%, DemoDiffusion 90 / 100 / 60%.

**Thí nghiệm phụ (`banana_on_plate`):**
- **Demo người không nhiễu:** kinematic 80%, DemoDiffusion s\*=0.2 **100%**, s\*=0.4 70%.
- **Kiểm tra nền** (demo sạch, không xê dịch vật): kinematic 3/3, DemoDiffusion s\*=0.2 3/3.

**Nhận xét:**
- **DemoDiffusion tốt hơn cả π0 lẫn kinematic khi chọn đúng s\*.** Mức tối ưu tùy theo task:
  - Với vật cần kẹp chính xác (vành bát, tượng đồ chơi), s\*=0.4 tốt hơn, vì π0 có nhiều bước để sửa vị trí kẹp.
  - Với chuối, s\*=0.2 đã đủ.
  - Bài báo cũng nhận xét s\* nên chọn theo từng task.
- **π0 cần đủ thời gian.** Với episode 10 s, π0 thất bại hoàn toàn với chuối (0/10). Với 18.5 s, π0 đạt 6/10, vì thường phải thử kẹp vài lần.
- **Chỉ 10 episode/ô** nên mỗi ô có sai số khoảng ±15%. Các chênh lệch nhỏ (ví dụ 30% so với 40%) chưa đủ để kết luận.

**Các bẫy kỹ thuật đã gặp** (đã sửa trong mã):
- **Demo cần nhịp kiểu người.** Bộ lọc EMA α=0.1 của tác giả (hằng số thời gian khoảng 0.67 s ở 15 Hz) làm quỹ đạo retarget trễ tới vài cm. Expert di chuyển liên tục thì tay kẹp đóng khi còn cao 3 cm, và lệnh nhả vật bị lọc mất. Demo phải dừng lại khi nắm và khi nhả, giống người thật.
- **Arena kết thúc episode ngay khi task thành công**, nên phần nhả vật và rút tay phải được nối thêm vào demo.
- **Vật trong cảnh test được đặt bằng `write_root_pose_to_sim_index`**, vì Arena xếp vật không theo seed của `env.reset`.
- **`rot_err` phải tính bằng `atan2`.** Dữ liệu float32 từ sim cộng với `arccos` làm phần quay của Jacobian bằng 0.
- **Asset theo đơn vị cm** (scale nằm ở prim gốc): khi đọc điểm mesh phải giữ lại tỉ lệ của prim gốc.
- **Mảng nhận qua msgpack là chỉ-đọc**, phải sao chép trước khi qua `DeltaActions`.
- **RTX trên GPU phụ thiếu VRAM thì cho ảnh trắng**, nên Isaac phải chạy trên GPU 0.

## Chạy trên server Linux

```bash
bash demodiffusion/linux/setup_ws.sh                     # uv, Arena (+Isaac Lab, Isaac Sim 6.1 pip), openpi, checkpoint 12 GB
GPU=auto bash demodiffusion/linux/serve_policy.sh &      # server pi0 / DemoDiffusion (cổng 8000)
bash demodiffusion/linux/run_task.sh banana_on_plate record
bash demodiffusion/linux/run_task.sh banana_on_plate eval --episodes 10 --methods pi0 kinematic dd0.2 dd0.4
bash demodiffusion/linux/make_compare.sh ~/demodiffusion_videos   # video so sanh 2x2 + json ket qua
```

Các tùy chọn của `run_task.sh`:
- `--human_noise none|realistic|plus5cm`: mức sai lệch của demo người.
- `--var_radius` / `--var_yaw`: mức xê dịch vật trong cảnh test.
- `--methods dd0.6` (hoặc mức khác): đặt s\* tùy ý.
- `SEED`, `DD_OUT`, `GPU`: biến môi trường cho seed, thư mục kết quả và GPU.

- Thư viện ngoài nằm trong `$WS = ~/ros2_ws/demodiffusion_ws` (SSD), không đưa vào git.
- JAX bị giới hạn 45% VRAM (`XLA_MEM`), để Isaac Sim chạy chung GPU.
