# UR5e · Isaac Sim 6.0.1 · ROS 2 Jazzy (pixi) · Control Panel

Điều khiển cánh tay UR5e trong Isaac Sim qua ROS 2 Jazzy chạy native trên Windows 11 (RoboStack/pixi, RMW Zenoh).

```
┌──────────────┐  /joint_command (JointState)  ┌──────────────────────────────┐
│  Panel (Qt)  │ ────────────────────────────▶ │ Isaac Sim 6.0.1              │
│  pixi run    │ ◀──────────────────────────── │  Action Graph ROS 2 bridge   │
│  panel       │  /joint_states, /clock, /tf,  │  UR5e + bàn + camera D435    │
└──────────────┘  /camera/color|depth/...      └──────────────────────────────┘
          ▲                   Zenoh router (pixi run zenoh)                ▲
          └────────────────────────────────────────────────────────────────┘
```

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `pixi.toml` | Môi trường ROS 2 Jazzy (ros-base, rmw_zenoh, PySide6) + các task |
| `isaac/ur5e_scene.py` | Chạy bên trong Isaac Sim: nạp scene của repo `ur5eworkspace-env`, dựng robot, gripper, camera, khối thử, Action Graph ROS 2, tự kiểm tra cấu hình |
| `isaac/robotiq_2f85.py` | Gắn Robotiq 2F-85 lên `wrist_3_link` + hàm kiểm tra cấu hình (chỉ dùng `pxr`, test được offline) |
| `config/gripper.json` | Do scene ghi ra: loại gripper, khớp điều khiển, TCP (`grip_frame`), commit asset |
| `tools/grasp_test.py` | Thử gắp – nâng – đặt khối (`pixi run grasp-test`) |
| `tools/ik_demo.py` | Demo IK: hình vuông MoveL (`pixi run ik-demo`) |
| `tools/pick_place_demo.py` | Pick & place 5 vật BOP HOPE (`pixi run pick-place`) |
| `tools/bop_to_usd.py` | Tải (`--fetch`) và chuyển model BOP (PLY, mm) sang USD có vật lý (`pixi run fetch-hope`) |
| `tools/remote_zip.py` | Đọc từng file trong zip qua HTTP range request (không tải cả zip) |
| `config/objects.json` | Do scene ghi ra: vật trên bàn, ô gắp / ô đặt |
| `panel/ur5e_panel.py` | Panel điều khiển (PySide6 + rclpy) |
| `panel/ur5e_kinematics.py` | FK, IK giải tích (8 nghiệm), Jacobian, IK DLS của UR5e |
| `tools/fake_sim.py` | Sim giả lập để thử panel khi chưa mở Isaac Sim |
| `tools/check_sim.py` | Kiểm tra kết nối Isaac Sim ↔ ROS 2 (`pixi run check-sim`) |
| `tests/` | Test FK/IK (`pixi run test`) |
| `config/poses.json` | Tư thế đã lưu (tự tạo khi bấm "Lưu") |

## Yêu cầu

- Driver NVIDIA **≥ 595.97** (Isaac Sim 6.x yêu cầu)
- Isaac Sim **6.0.1** tại `C:\isaacsim` (đổi `isaac_sim_package_path` trong `pixi.toml` nếu cài chỗ khác)
- pixi (`winget install prefix-dev.pixi`), sau đó `pixi install` trong thư mục này
- Repo `ur5eworkspace-env` nằm cạnh thư mục này (`..\ur5eworkspace-env`)
- Asset gripper của Robotiq nằm cạnh thư mục này (cần Git LFS):
  ```powershell
  git lfs install
  git clone https://github.com/robotiq/isaacsim_assets.git ..\robotiq_isaacsim_assets
  ```
  Đã kiểm thử ở commit `fbfcd0b` (repo được cập nhật thường xuyên; commit đang dùng được ghi vào `config/gripper.json`).
- Isaac Sim + scene dùng khoảng 13 GB RAM (commit). Máy 32 GB nên đóng bớt ứng dụng nặng (Edge nhiều tab…) trước khi chạy.

## Chạy

Mở 3 terminal trong thư mục `ur5e_ros`:

```powershell
pixi run zenoh     # 1. Zenoh router — luôn chạy TRƯỚC
pixi run sim       # 2. Isaac Sim + scene + ROS 2 bridge (tự bấm Play)
pixi run panel     # 3. Panel điều khiển
```

Lần đầu chạy `pixi run sim` mất khoảng 4–5 phút vì phải biên dịch shader và tải asset UR5e, bàn, chai từ server của NVIDIA. Trong lúc này cửa sổ có thể báo "Not Responding". Scene sẵn sàng khi log in ra `[ur5e_scene] Da bam Play`. Từ lần thứ hai sẽ nhanh hơn.

Thử panel khi chưa có Isaac Sim: thay bước 2 bằng `pixi run fake-sim`.

Khi dựng xong, scene tự kiểm tra cấu hình và in từng dòng `[ur5e_scene] CHECK PASS/FAIL …`, cuối cùng là `CONFIG OK n/n`.

Kiểm tra kết nối: `pixi run check-sim` (thêm `-- --move` để thử cho robot chuyển động). Lệnh này kiểm tra `/joint_states` (cả khớp gripper), `/clock`, camera, so FK của panel với `/tf` (vị trí và hướng), frame TF của gripper, và việc bám lệnh khớp.

Thử gắp khối lập phương (chạy sim với `$env:UR5E_OBJECTS="cube"`): `pixi run grasp-test` sẽ mở gripper, đi tới khối, MoveL xuống, đóng, nâng 15 cm, đi ngang 10 cm, rồi đặt lại.

Liệt kê topic: `pixi run topics` (dùng `--no-daemon` vì ros2 daemon hay treo trên Windows).

`isaac-sim.bat` chỉ dùng thư viện ROS nội bộ khi chưa có `ROS_DISTRO`. Chạy qua pixi thì Isaac Sim dùng ROS 2 Jazzy và Zenoh của pixi (log có dòng `Attempting to load system rclpy … rclpy loaded`).

## Panel

- **Thanh trên**: trạng thái kết nối, thanh trượt **Tốc độ** (áp cho mọi chế độ), nút **STOP** (phím `Space`).
- **Gripper** (hiện khi `config/gripper.json` có gripper): nút *Mở* / *Đóng*, thanh 0–100% (0 = mở, 100 = đóng = `finger_joint` 0.8 rad), cột *thực tế*. Nếu đang kẹp vật thì độ đóng thực tế dừng giữa chừng (khối 4 cm khoảng 57%).
- **Khớp**: kéo thanh trượt hoặc gõ góc cho từng khớp, cột phải là góc thực tế từ `/joint_states`. Danh sách tư thế đã lưu: double-click hoặc "Đi tới (MoveJ)".
- **Cartesian (IK)**: pose TCP trong khung `base_link` của robot.
  - *MoveJ*: IK giải tích, chọn nghiệm gần tư thế hiện tại, nội suy trong không gian khớp.
  - *MoveL*: đường thẳng trong không gian Cartesian (IK DLS, ổn định gần điểm kỳ dị).
  - *Jog*: nhấn giữ nút để di chuyển liên tục theo khung Base hoặc Tool.
  - *TCP offset Z*: khoảng cách từ `tool0` tới đầu kẹp. Có 2F-85 thì panel tự đặt bằng `grip_frame` (133.7 mm).
- **Camera D435**: ảnh màu và ảnh độ sâu (tô màu 0.1–3 m).

Tham số topic: `pixi run panel -- --joint-command-topic /x --rgb-topic /y` (xem `--help`).

> Tư thế *Home (scene)* của Hung có `wrist_2 = 0°`, là **điểm kỳ dị cổ tay** của UR. MoveL và Jog vẫn chạy được (dùng DLS), nhưng gần điểm này robot sẽ chậm lại. Nên chuyển sang *Ready (hướng xuống)* trước khi điều khiển Cartesian.

## Robot trong scene

Scene gốc `ur5etunewithtable.usd` tham chiếu `./ur_1/payloads/base.usda` (UR5e + Robotiq 3F) và `./d435_import/d435_1/completed435ros.usd`, nhưng **hai thư mục này chưa có trên GitHub**. Vì vậy `ur5e_scene.py` sẽ:

- **Khi thiếu**: tắt `/ur`, thay bằng UR5e chuẩn của Isaac Sim ở cùng vị trí, áp stiffness/damping/maxForce/target đã tune trong scene, rồi gắn camera D435 tạm tại đúng `d435_mount`.
- **Khi có đủ** (chép `ur_1/` và `d435_import/` vào `ur5eworkspace-env/`): tự dùng robot gốc. Panel không cần sửa vì tên khớp giống nhau.

File scene gốc được nạp làm *sublayer*, nên mọi thay đổi nằm ở layer tạm và file của repo không bị ghi đè.

Khi thiếu `ur_1/`, log Isaac Sim có 3 dòng "Encountered USD Warnings but USD Diagnostics are currently muted". Đây là USD báo không mở được các file đang thiếu, sẽ hết khi repo có đủ.

## Gripper Robotiq 2F-85

Dùng asset [robotiq/isaacsim_assets](https://github.com/robotiq/isaacsim_assets), làm theo `grippers/GRIPPER_SIMULATION_GUIDE.md` và scene demo UR5e của Robotiq (`grippers/demo/teleop/ur5robot_with_2F-85.usda`):

| Mục | Giá trị |
|---|---|
| Variant | `Physics = Physx_parallel_grip` (mimic joint, ngón luôn song song), `Fingertip = Standard` |
| Gắn | `/ur5e/wrist_3_link/Robotiq_2F_85_edit` (pose đơn vị) + `FixedJoint` `/ur5e/joints/robotiq_2f85_mount_joint` |
| Articulation | Xóa `ArticulationRootAPI` của gripper, tắt `root_joint`. Gripper thuộc articulation `/ur5e/root_joint` |
| `finger_joint` | stiffness 50, damping 3, maxForce 10 N·m (guide §3.1). Giới hạn 0–47°, lệnh đóng 0.8 rad |
| Cơ cấu ngón | 4 thân outer/inner finger: solver iterations 64/8 |
| Vật lý | PhysX CPU (`enableGPUDynamics = 0`, broadphase MBP), 120 Hz, `enableStabilization`, articulation 255/4 |
| TF | `base_link` của gripper đổi tên frame thành `robotiq_base_link` (`isaac:nameOverride`) để không trùng `base_link` của UR5e |
| TCP | `grip_frame` của asset, `(0, 0, 0.1337)` m trong `wrist_3_link` |
| Khối thử | `/TestObjects/TestCube` 4 cm, 0.1 kg, contactOffset 2 mm, restOffset 0.5 mm, linearDamping 0.1, velocity iterations 4, ma sát 1.0/0.9. Thêm `TableTopCollider` vì bàn từ S3 chỉ có hình |

Gửi lệnh gripper từ ROS: `sensor_msgs/JointState` lên `/joint_command` với `name = ["finger_joint"]`, `position = [0.0 … 0.8]`.

Tắt gripper: `$env:UR5E_GRIPPER="none"` trước `pixi run sim`. Khi repo của Hung có robot gốc (gripper 3F), script tự dùng robot đó và không gắn 2F-85.

Ghi chú:
- GPU dynamics (mặc định của scene) làm sim chỉ còn khoảng 20% thời gian thực với cấu hình solver này. Vì vậy dùng PhysX CPU như demo của Robotiq.
- `/tf` vẫn dùng `targetPrims` của `ROS2PublishTransformTree` (deprecated). `IsaacComputeTransformTree` không mở rộng cây link khi articulation root nằm trên joint prim (`/ur5e/root_joint` của asset UR5e).
- Biến thể Newton (theo blog Robotiq) chưa dùng, vì phải vá file trong bản cài Isaac Sim 6.0.1 (Newton 1.2.1) và chạy `apply_gripper_tuning.py` sau mỗi lần Play.

## Demo pick & place với vật BOP HOPE

`pixi run fetch-hope` chỉ lấy 5 vật cần dùng từ zip HOPE 87.5 MB trên Hugging Face (khoảng 20 MB) rồi chuyển sang USD trong `../datasets/bop_hope/usd/`:

| Vật | Kích thước (mm) | Độ cao kẹp tính từ đáy |
|---|---|---|
| Butter (obj 3) | 24 × 103 × 53 | 26 mm |
| ChocolatePudding (obj 5) | 30 × 84 × 50 | 26 mm |
| GreenBeans (obj 10) | 71 × 66 × 58 | 28 mm |
| Raisins (obj 23) | 40 × 86 × 123 | 62 mm (gần trọng tâm) |
| Yogurt (obj 28) | 68 × 68 × 54, thuôn | 44 mm (vành miệng) |

Mỗi USD có mesh theo mét, texture, rigid body, collision `convexHull`, các offset va chạm theo guide của Robotiq, ma sát 1.0/0.9 và ma sát xoắn (`torsionalPatchRadius` 10 mm). Độ cao kẹp được tính từ profile bề rộng của mesh: chỗ rộng nhất, nếu bằng nhau thì chỗ gần giữa.

Mặc định scene (`UR5E_OBJECTS=hope`) xếp 5 vật thành hàng ở x = 0.52 m (khung `base_link`), mỗi vật xoay một góc khác nhau. `pixi run pick-place` lần lượt:
1. Đọc pose của vật từ `/tf`.
2. Xoay gripper cho hướng kẹp trùng cạnh hẹp của vật.
3. MoveJ tới phía trên vật, MoveL xuống, rồi kẹp.
4. Nâng lên, xoay để vật thẳng hàng theo trục X, đặt vào hàng x = 0.30 m.
5. Kiểm tra vị trí và độ nghiêng của vật qua `/tf`.

Kết quả: 4/5 vật đạt (lệch 0.2–17.5 mm). Yogurt là cốc thuôn nên bị lật trong tay gắp song song (xem README gốc).

### Băng chuyền và thùng (`isaac/conveyor_line.py`)

Với bộ vật BOP, scene mặc định thêm 2 băng chuyền phía sau robot. Mỗi băng chạy ra xa bàn và có thùng KLT ở cuối. Đặt `UR5E_CONVEYORS=0` để quay lại kiểu xếp hàng trên bàn.

| | Băng 1 (y = −0.33 m) | Băng 2 (y = +0.33 m) |
|---|---|---|
| Vật | 1, 2 (HexBolt, SteppedPin) | 3, 4, 5 (PipeTee, ClampBlock, SlottedClamp) |

- **Băng chuyền**: asset NVIDIA `ConveyorBelt_A06`, thu nhỏ đều ×0.428 để mặt băng ngang mặt bàn (0.762 m). Băng dài 0.86 m, rộng 0.39 m.
- **Chuyển động**: mặt băng là rigid body kinematic, đặt `PhysxSurfaceVelocityAPI` theo hệ thế giới (−0.12 m/s theo X). Node `IsaacConveyor` (`UR5E_CONVEYOR_MODE=node`) cuộn được texture, nhưng với băng đã xoay và thu nhỏ thì kéo vật sai chiều hoặc gần như không kéo.
- **Thùng**: KLT nhỏ 20 × 30 cm, cố định (kinematic), đặt trên bệ sao cho miệng thùng thấp hơn mặt băng 4 cm.
- **Tự kiểm tra**: sau khi bấm Play, scene thả một khối 4 cm lên mỗi băng, đo vận tốc thực rồi xóa khối, và in `CONVEYOR TEST OK`. Chờ dòng này rồi mới chạy demo.
- **Demo**: robot thả vật cách mặt băng 1 cm, cạnh hẹp nằm ngang băng để vật không đổ. Khi xoay ra sau, robot chạy chậm (hệ số tốc độ 0.15) để vật nặng không trượt khỏi ngón. Cuối demo, chương trình kiểm tra qua `/tf` rằng từng vật nằm trong đúng thùng.
- **Kết quả (server Linux, ITODD)**: **5/5** vật vào đúng thùng. Tự kiểm tra đo được −0.120 m/s ở cả hai băng.

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Panel báo "chưa nhận /joint_states" | Zenoh router chưa chạy trước Isaac Sim, hoặc sim chưa Play |
| Panel báo "sim đang dừng" | Bấm Play trong Isaac Sim |
| `ros2 topic list` treo | Dùng `pixi run topics` (`--no-daemon`) |
| Lỗi đường dẫn quá dài khi `pixi install` | Chuyển workspace tới đường dẫn ngắn hơn, ví dụ `C:\ur5e_ros` |
