"""Ghi topic anh ROS 2 thanh video MP4 (+ anh chup dinh ky) - dung khi chay Isaac Sim headless.

    python tools/record_camera.py --out recordings/demo.mp4                 # toi khi Ctrl+C / SIGTERM
    python tools/record_camera.py --out a.mp4 --duration 20 --snapshot-every 5

Mac dinh ghep 2 khung camera D435 co tay (RGB tren, do sau to mau jet 0.1-3 m duoi) thanh cot doc
o goc tren ben trai (--inset-horizontal: xep ngang, --no-inset: tat).

Can: imageio-ffmpeg (pip) de ghi MP4; anh chup PNG dung imageio-ffmpeg luon (1 frame) hoac numpy thuan.
"""

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def to_rgb(msg: Image) -> np.ndarray:
    buf = np.frombuffer(bytes(msg.data), np.uint8)
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}[msg.encoding.lower()]
    arr = buf.reshape(msg.height, msg.step)[:, : msg.width * ch].reshape(msg.height, msg.width, ch)[:, :, :3]
    if msg.encoding.lower().startswith("bgr"):
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


def depth_to_rgb(msg: Image, rng=(0.1, 3.0)) -> np.ndarray:
    """Anh do sau (32FC1 m / 16UC1 mm) -> RGB bang mau jet; vung khong hop le (nan, 0, ngoai dai) mau den."""
    enc = msg.encoding.upper()
    dt, scale = {"32FC1": (np.float32, 1.0), "16UC1": (np.uint16, 1e-3)}[enc]
    d = np.frombuffer(bytes(msg.data), dt).reshape(msg.height, -1)[:, : msg.width].astype(np.float32) * scale
    lo, hi = rng
    valid = np.isfinite(d) & (d >= lo)
    d = np.nan_to_num(d, nan=hi, posinf=hi, neginf=lo)
    t = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    rgb = np.stack([np.clip(1.5 - np.abs(4 * t - k), 0, 1) for k in (3, 2, 1)], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def downscale(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Thu nho anh: trung binh khoi neu chia het, khong thi lay mau gan nhat."""
    H, W = img.shape[:2]
    if H % h == 0 and W % w == 0:
        fy, fx = H // h, W // w
        return img.reshape(h, fy, w, fx, -1).mean(axis=(1, 3)).astype(np.uint8)
    ys = (np.arange(h) * H / h).astype(int)
    xs = (np.arange(w) * W / w).astype(int)
    return img[ys][:, xs]


# Font bitmap 5x7 cho nhan inset (chi cac ky tu can dung, ky tu khac -> khoang trang)
_FONT = {
    "0": "01110 10001 10011 10101 11001 10001 01110", "1": "00100 01100 00100 00100 00100 00100 01110",
    "3": "11110 00001 00001 01110 00001 00001 11110", "4": "00010 00110 01010 10010 11111 00010 00010",
    "5": "11111 10000 11110 00001 00001 10001 01110", "B": "11110 10001 10001 11110 10001 10001 11110",
    "D": "11110 10001 10001 10001 10001 10001 11110", "E": "11111 10000 10000 11110 10000 10000 11111",
    "G": "01110 10001 10000 10111 10001 10001 01111", "H": "10001 10001 10001 11111 10001 10001 10001",
    "M": "10001 11011 10101 10101 10001 10001 10001", "P": "11110 10001 10001 11110 10000 10000 10000",
    "R": "11110 10001 10001 11110 10100 10010 10001", "T": "11111 00100 00100 00100 00100 00100 00100",
    ".": "00000 00000 00000 00000 00000 01100 01100", "-": "00000 00000 00000 11111 00000 00000 00000",
}


def draw_text(img: np.ndarray, x: int, y: int, text: str, scale: int = 2, color=(255, 255, 255)):
    for i, ch in enumerate(text.upper()):
        rows = _FONT.get(ch)
        if rows is None:
            continue
        g = np.array([[c == "1" for c in r] for r in rows.split()], bool)
        g = np.kron(g, np.ones((scale, scale), bool))
        x0 = x + i * 6 * scale
        sub = img[y: y + g.shape[0], x0: x0 + g.shape[1]]
        sub[g[: sub.shape[0], : sub.shape[1]]] = color


def compose_insets(frame: np.ndarray, panels, width: int = 320, margin: int = 12, border: int = 2,
                   vertical: bool = True) -> np.ndarray:
    """Dan cac panel (anh RGB hoac None, nhan) o goc tren ben trai: cot doc (mac dinh) hoac hang ngang."""
    out = frame.copy()
    x = y = margin
    for img, label in panels:
        h = width * 3 // 4 if img is None else round(width * img.shape[0] / img.shape[1])
        if x + width + border > out.shape[1] or y + h + border > out.shape[0]:
            break
        out[y - border: y + h + border, x - border: x + width + border] = 255  # vien trang
        out[y: y + h, x: x + width] = 0 if img is None else downscale(img, width, h)
        out[y: y + 22, x: x + width] //= 3  # nen toi cho nhan
        draw_text(out, x + 6, y + 4, label if img is not None else label + " -")
        if vertical:
            y += h + margin
        else:
            x += width + margin
    return out


def save_png(path: Path, rgb: np.ndarray):
    import zlib
    import struct

    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


class Recorder(Node):
    def __init__(self, args):
        super().__init__("camera_recorder")
        self.args = args
        self.out = Path(args.out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.frames = 0
        self.t0 = None
        self.last_snap = -1e9
        self.lock = threading.Lock()
        self.create_subscription(Image, args.topic, self.on_image, qos_profile_sensor_data)
        # Anh moi nhat cua camera D435 co tay de ghep vao goc tren ben trai
        self.inset_rgb = self.inset_depth = None
        if not args.no_inset:
            self.create_subscription(Image, args.inset_rgb_topic, self.on_inset_rgb, qos_profile_sensor_data)
            self.create_subscription(Image, args.inset_depth_topic, self.on_inset_depth, qos_profile_sensor_data)

    def on_inset_rgb(self, msg):
        self.inset_rgb = to_rgb(msg)

    def on_inset_depth(self, msg):
        self.inset_depth = depth_to_rgb(msg)

    def on_image(self, msg):
        rgb = to_rgb(msg)
        if not self.args.no_inset:
            rgb = compose_insets(rgb, [(self.inset_rgb, "D435 RGB"), (self.inset_depth, "D435 DEPTH 0.1-3M")],
                                 width=self.args.inset_width, vertical=not self.args.inset_horizontal)
        now = time.monotonic()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9  # thoi gian sim
        with self.lock:
            if self.writer is None:
                import imageio_ffmpeg

                self.writer = imageio_ffmpeg.write_frames(
                    str(self.out), (rgb.shape[1], rgb.shape[0]), fps=self.args.fps,
                    codec="libx264", pix_fmt_in="rgb24", quality=8, macro_block_size=8)
                self.writer.send(None)
                self.t0 = now
                self.next_t = stamp
                print(f"bat dau ghi {rgb.shape[1]}x{rgb.shape[0]} -> {self.out}", flush=True)
            # Dat frame theo thoi gian SIM: sim cham hon thoi gian thuc thi lap frame, video van dung toc do
            if stamp <= 0 or stamp < self.next_t - 1.0:  # khong co stamp / sim reset -> ghi thang
                self.next_t = stamp
            n = 0
            while self.next_t <= stamp + 1e-6 and n < 100:
                self.writer.send(rgb.tobytes())
                self.next_t += 1.0 / self.args.fps
                self.frames += 1
                n += 1
            self.received = getattr(self, "received", 0) + 1
        if self.args.snapshot_every > 0 and now - self.last_snap >= self.args.snapshot_every:
            self.last_snap = now
            snap = self.out.with_name(f"{self.out.stem}_{int(now - self.t0):04d}s.png")
            save_png(snap, rgb)

    def close(self):
        with self.lock:
            if self.writer is not None:
                self.writer.close()
                self.writer = None
        dur = (time.monotonic() - self.t0) if self.t0 else 0
        rec = getattr(self, "received", 0)
        print(f"xong: nhan {rec} anh trong {dur:.1f} s that ({rec / max(dur, 1e-6):.1f} Hz) -> video {self.frames} frame "
              f"= {self.frames / self.args.fps:.1f} s thoi gian sim @ {self.args.fps:.0f} fps -> {self.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/demo_cam/rgb")
    ap.add_argument("--out", default="recordings/demo.mp4")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--duration", type=float, default=0.0, help="giay; 0 = toi khi bi dung")
    ap.add_argument("--snapshot-every", type=float, default=0.0, help="chup PNG moi N giay (0 = tat)")
    ap.add_argument("--inset-rgb-topic", default="/camera/color/image_raw")
    ap.add_argument("--inset-depth-topic", default="/camera/depth/image_rect_raw")
    ap.add_argument("--inset-width", type=int, default=320, help="chieu rong moi khung D435 ghep vao (px)")
    ap.add_argument("--inset-horizontal", action="store_true", help="xep 2 khung D435 thanh hang ngang (mac dinh cot doc)")
    ap.add_argument("--no-inset", action="store_true", help="khong ghep khung D435")
    args, _ = ap.parse_known_args()

    rclpy.init()
    rec = Recorder(args)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    end = time.monotonic() + args.duration if args.duration > 0 else None
    while not stop.is_set() and (end is None or time.monotonic() < end):
        rclpy.spin_once(rec, timeout_sec=0.05)
    rec.close()
    rec.destroy_node()
    rclpy.try_shutdown()
    return 0 if rec.frames else 1


if __name__ == "__main__":
    sys.exit(main())
