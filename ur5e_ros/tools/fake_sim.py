"""Sim gia lap toi gian de thu panel khi chua mo Isaac Sim.

Nhan /joint_command, bam theo bang khau quan tinh bac 1, phat /joint_states (60 Hz),
/clock, anh mau va anh do sau gia (15 Hz). Chay: pixi run fake-sim
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, JointState

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
HOME = np.radians([0.0, -51.4, 66.5, -67.3, 0.0, 0.0])
GRIPPER_JOINT = "finger_joint"  # gia lap 2F-85: 0 = mo, 0.8 rad = dong


class FakeSim(Node):
    def __init__(self):
        super().__init__("fake_isaac_sim")
        self.q = HOME.copy()
        self.target = HOME.copy()
        self.grip = 0.0
        self.grip_target = 0.0
        self.t = 0.0
        self.js_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.rgb_pub = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_rect_raw", 10)
        self.create_subscription(JointState, "/joint_command", self._on_cmd, 10)
        self.create_timer(1 / 60, self._step)
        self.create_timer(1 / 15, self._camera)
        self.get_logger().info("fake sim dang chay: /joint_command -> /joint_states")

    def _on_cmd(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in JOINT_NAMES:
                self.target[JOINT_NAMES.index(name)] = pos
            elif name == GRIPPER_JOINT:
                self.grip_target = min(max(pos, 0.0), 0.82)

    def _step(self):
        dt = 1 / 60
        self.t += dt
        self.q += (self.target - self.q) * min(1.0, dt / 0.05)
        self.grip += (self.grip_target - self.grip) * min(1.0, dt / 0.15)
        stamp = self.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = stamp
        js.name = JOINT_NAMES + [GRIPPER_JOINT]
        js.position = self.q.tolist() + [self.grip]
        self.js_pub.publish(js)
        clk = Clock()
        clk.clock.sec = int(self.t)
        clk.clock.nanosec = int((self.t % 1) * 1e9)
        self.clock_pub.publish(clk)

    def _camera(self):
        w, h = 320, 240
        x = np.linspace(0, 1, w)[None, :]
        y = np.linspace(0, 1, h)[:, None]
        phase = self.q[0]
        rgb = np.zeros((h, w, 3), np.uint8)
        rgb[..., 0] = (255 * (0.5 + 0.5 * np.sin(6 * x + phase))).astype(np.uint8)
        rgb[..., 1] = (255 * y).astype(np.uint8)
        rgb[..., 2] = 128
        depth = (0.3 + 2.5 * (0.5 + 0.5 * np.sin(4 * math.pi * x + 3 * y + phase))).astype(np.float32)
        stamp = self.get_clock().now().to_msg()
        for pub, arr, enc, step in ((self.rgb_pub, rgb, "rgb8", w * 3), (self.depth_pub, depth, "32FC1", w * 4)):
            m = Image()
            m.header.stamp = stamp
            m.header.frame_id = "camera_color_optical_frame"
            m.height, m.width, m.encoding, m.step = h, w, enc, step
            m.data = arr.tobytes()
            pub.publish(m)


def main():
    rclpy.init()
    node = FakeSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
