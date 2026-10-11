"""Ghi topic anh ROS 2 thanh video MP4 (+ anh chup dinh ky) - dung khi chay Isaac Sim headless.

    python tools/record_camera.py --out recordings/demo.mp4                 # toi khi Ctrl+C / SIGTERM
    python tools/record_camera.py --out a.mp4 --duration 20 --snapshot-every 5

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

    def on_image(self, msg):
        rgb = to_rgb(msg)
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
