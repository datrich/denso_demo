# denso_demo — UR5e + Robotiq 2F-85 trong Isaac Sim 6, điều khiển qua ROS 2 Jazzy

Demo cánh tay UR5e gắn tay gắp 2 ngón Robotiq 2F-85 trong NVIDIA Isaac Sim 6.0.1. Robot được điều khiển qua ROS 2 Jazzy chạy native trên Windows 11 (pixi/RoboStack, RMW Zenoh), gồm:

- **Panel điều khiển** (PySide6): thanh trượt từng khớp, IK Cartesian (MoveJ / MoveL / jog), gripper, camera cổ tay RGB + độ sâu.
- **IK của UR5e**: giải tích (8 nghiệm) cho MoveJ, damped least-squares cho MoveL/jog (ổn định gần điểm kỳ dị).
- **Gripper Robotiq 2F-85** (asset chính thức của Robotiq, PhysX `parallel_grip`), tự kiểm tra cấu hình khi dựng scene.
- **Demo gắp**: gắp – nâng – đặt khối lập phương, và pick & place 5 vật thật từ bộ dữ liệu [BOP HOPE](https://bop.felk.cvut.cz/datasets/).

![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-6.0.1-76b900) ![ROS 2](https://img.shields.io/badge/ROS%202-Jazzy-22314e) ![Platform](https://img.shields.io/badge/Windows-11-0078d4)

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| [`ur5e_ros/`](ur5e_ros/) | Toàn bộ code: scene Isaac Sim, panel, IK, công cụ và demo (xem [README chi tiết](ur5e_ros/README.md)) |
| `ur5eworkspace-env/` | *Submodule* — scene bàn + vị trí robot + gain đã tune ([Veus09/ur5eworkspace-env](https://github.com/Veus09/ur5eworkspace-env)) |
| `robotiq_isaacsim_assets/` | *Submodule* — asset gripper của Robotiq, ghim ở commit `fbfcd0b` ([robotiq/isaacsim_assets](https://github.com/robotiq/isaacsim_assets)) |
| `datasets/bop_hope/` | Không có trong repo, tạo bằng `pixi run fetch-hope` |

## Yêu cầu

- Windows 11, GPU NVIDIA RTX (đã thử với RTX 5090), driver **≥ 595.97**
- [Isaac Sim **6.0.1**](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/download.html) giải nén tại `C:\isaacsim`
- [pixi](https://pixi.sh) (`winget install prefix-dev.pixi`), Git + [Git LFS](https://git-lfs.com)
- RAM: Isaac Sim + scene dùng khoảng 13 GB. Máy 32 GB nên đóng bớt ứng dụng nặng khi chạy.

## Cài đặt

```powershell
git lfs install
git clone --recursive https://github.com/datrich/denso_demo.git
cd denso_demo\ur5e_ros
pixi install          # ROS 2 Jazzy + Zenoh + PySide6 + usd-core
pixi run test         # 209 test FK/IK
pixi run fetch-hope   # tải 5 vật HOPE (~20 MB, chỉ phần cần) và chuyển sang USD
```

## Chạy

Mở 3 terminal trong `ur5e_ros`:

```powershell
pixi run zenoh        # 1. Zenoh router — luôn chạy trước
pixi run sim          # 2. Isaac Sim + scene + cầu ROS 2 (tự bấm Play)
pixi run panel        # 3. Panel điều khiển
```

Scene tự kiểm tra cấu hình và in `[ur5e_scene] CONFIG OK 20/20`. Sau đó có thể chạy:

| Lệnh | Việc làm |
|---|---|
| `pixi run check-sim -- --move` | Kiểm tra topic, camera, FK so với `/tf`, gripper, bám lệnh khớp |
| `pixi run ik-demo` | MoveL vẽ hình vuông 15 cm, hạ/nâng 10 cm |
| `pixi run pick-place` | Gắp 5 vật HOPE, xếp ngay ngắn vào hàng gần robot |
| `pixi run grasp-test` | Gắp – nâng – đặt khối lập phương (chạy sim với `$env:UR5E_OBJECTS="cube"`) |

## Chạy headless trên server Linux (quay video, không cần màn hình)

Đã thử trên Ubuntu 24.04, ROS 2 Jazzy cài bằng apt, Isaac Sim 6.1.0, 2× RTX 4090. Không cần pixi: script dùng ROS của hệ thống (`~/ros2_ws/env.sh` nếu có) và một venv nhỏ.

```bash
bash ur5e_ros/linux/setup.sh                         # venv + usd-core + imageio-ffmpeg, tải 5 vật HOPE
GPU=1 bash ur5e_ros/linux/sim_headless.sh &          # Isaac Sim headless trên GPU 1
bash ur5e_ros/linux/run.sh check_sim --move          # kiểm tra
bash ur5e_ros/linux/record_demo.sh                   # chạy pick & place + quay MP4 -> recordings/<thời gian>/
```

- `--no-window`: không mở cửa sổ, không đụng tới desktop của người đang dùng máy.
- GPU được chọn bằng `--/renderer/activeGpu` và `--/physics/cudaDevice`. Đừng dùng `CUDA_VISIBLE_DEVICES`: renderer RTX sẽ lỗi "No device could be created".
- Cấu hình, cache và log của Kit nằm riêng trong `.isaac_portable/`, nên không lẫn với Isaac Sim của người khác. Chạy với `nice 10`.
- Camera toàn cảnh 1280×720 (`UR5E_DEMO_CAM=1`) phát lên `/demo_cam/rgb`. `record_camera.py` đặt khung hình theo thời gian sim, nên video phát đúng tốc độ kể cả khi sim chậm. Ở góc trên bên trái, video ghép thêm 2 khung của camera D435 ở cổ tay: ảnh RGB và ảnh độ sâu tô màu jet trong dải 0.1–3 m. Dùng `--no-inset` để tắt phần ghép này.
- Demo và các khoảng chờ chạy theo thời gian sim (`/clock`). Trên server, sim chỉ đạt RTF khoảng 0.46 (vật lý chạy CPU) mà demo vẫn đúng.

## Kết quả đã kiểm chứng

| Bài kiểm tra | Kết quả |
|---|---|
| Cấu hình scene (gripper, vật lý, vật thể) | 20/20 |
| FK của panel so với `/tf` của Isaac Sim | lệch 0.04 mm / 0.004° |
| Gắp khối lập phương 4 cm | 8/8. Nâng 150.1/150 mm, trượt 0.0 mm |
| Pick & place 5 vật HOPE (Windows, RTX 5090) | **4/5**. Butter 0.5 mm, Pudding 1.2 mm, GreenBeans 0.2 mm, Raisins 17.5 mm |
| Pick & place 5 vật HOPE (server Linux headless, chờ theo thời gian sim) | **5/5**. Lệch 1 / 1 / 1 / 4 / 14 mm, nghiêng ≤ 1° |
| Pick & place 5 chi tiết cơ khí ITODD (server Linux headless) | **5/5**. Lệch 0.6–2.5 mm, nghiêng 0° |

### Bộ chi tiết cơ khí (BOP ITODD)

`UR5E_OBJECTS=itodd` thay đồ gia dụng bằng 5 chi tiết công nghiệp của [MVTec ITODD](https://bop.felk.cvut.cz/datasets/). Tải bằng `tools/bop_to_usd.py --dataset itodd --fetch` (Windows: `pixi run fetch-itodd`):

| Chi tiết | ITODD | Đặt trên bàn (cao × hẹp × dài) | Kẹp ở |
|---|---|---|---|
| HexBolt (bu lông lục giác) | 23 | 56 × 17 × 19 mm | thân ren 8 mm |
| SteppedPin (chốt bậc) | 24 | 60 × 30 × 30 mm | trục Ø20 trên vai |
| PipeTee (khớp nối chữ T) | 26 | 41 × 25 × 63 mm | thân ống Ø24 |
| ClampBlock (khối kẹp) | 9 | 38 × 60 × 78 mm | cạnh 60 mm |
| SlottedClamp (kẹp có rãnh) | 10 | 50 × 15 × 20 mm | cạnh 15 mm |

Mesh CAD công nghiệp không đặt sẵn ở tư thế đứng, nên bộ chuyển đổi thử 6 hướng úp xuống bàn. Hướng được chọn phải đứng vững (tâm khối nằm sâu trong vùng tựa) và kẹp được (cạnh hẹp 10–70 mm, cao ≥ 30 mm). Ngoài ra bộ chuyển đổi:
- Tính khối lượng từ thể tích mesh với mật độ nhôm.
- Dùng va chạm `convexDecomposition` cho vật lõm.
- Đo độ cao kẹp bằng mặt cắt chính xác qua các tam giác, nên dùng được cả với mesh CAD ít đỉnh.
| Tốc độ mô phỏng | Thời gian thực (60 Hz), camera 30 Hz |

**Hạn chế đã biết**: *Yogurt* là cốc thuôn (54 → 66 mm), nên hai má kẹp phẳng song song chỉ chạm cốc ở 2 điểm. Cốc xoay quanh trục nối 2 điểm đó và bị lật dù đã thêm ma sát xoắn và kẹp ở vành. Muốn gắp được loại vật này cần biến thể `compliant` của 2F-85 hoặc tay gắp hút.

## Ghi chú kỹ thuật

- Scene gốc của `ur5eworkspace-env` được nạp làm *sublayer*, nên file của repo đó không bị sửa. Repo đó hiện thiếu thư mục robot gốc (`ur_1/`, gripper 3 ngón), nên scene thay bằng UR5e chuẩn của Isaac Sim và áp gain đã tune.
- PhysX chạy trên CPU, 120 Hz. Để GPU dynamics (mặc định) thì sim chỉ đạt khoảng 20% thời gian thực với cấu hình solver này.
- Chi tiết về gripper, vật lý và các lỗi thường gặp: xem [`ur5e_ros/README.md`](ur5e_ros/README.md).

## Bản quyền bên thứ ba

Repo này chỉ chứa code. Asset bên thứ ba được tải từ nguồn gốc và giữ nguyên giấy phép của chúng:

- Robotiq 2F-85: [robotiq/isaacsim_assets](https://github.com/robotiq/isaacsim_assets) (BSD-3-Clause và các giấy phép đi kèm asset)
- UR5e, bàn, chai: thư viện asset của NVIDIA Isaac Sim (tải lúc chạy)
- Vật thể HOPE: [BOP benchmark](https://bop.felk.cvut.cz/datasets/), xem giấy phép của bộ dữ liệu trước khi dùng lại
