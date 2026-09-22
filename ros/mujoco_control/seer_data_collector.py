#!/usr/bin/env python3
"""
Seer-format data collector for the MuJoCo Panda pick-and-place pipeline.

Output per step (matches Seer's documented format):
  - image_primary.jpg
  - image_wrist.jpg
  - other.npz
    - joints
    - gripper_pose
    - gripper_open_state,
    - action_gripper_pose
    - delta_cur_2_last_action
    - language_instruction
"""

import os
import cv2
import time
import rclpy
import numpy as np
import argparse
import threading
import subprocess

from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge
from rclpy.time import Time
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, JointState
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.transform import Rotation as R


JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

# ============================= Pose helpers =============================
# Uses identical conventions to Seer: xyz + euler "xyz" in radians)

def pose6d_to_mat(p):
    T = np.eye(4)
    T[:3, 3] = p[:3]
    T[:3, :3] = R.from_euler("xyz", p[3:6]).as_matrix()
    return T


def mat_to_pose6d(T):
    out = np.zeros(6)
    out[:3] = T[:3, 3]
    out[3:6] = R.from_matrix(T[:3, :3]).as_euler("xyz")
    return out


def compute_delta_actions(frames):
    """
    frames[i] has 'gripper_pose' (6,) and 'action_gripper_pose' (7,).
    Returns list of (7,) delta_cur_2_last_action, mirroring Seer's compute_delta_action.
    """
    deltas = []
    for i, f in enumerate(frames):
        d = np.zeros(7)
        d[6] = f["action_gripper_pose"][6]
        last = pose6d_to_mat(f["gripper_pose"] if i == 0 else frames[i - 1]["action_gripper_pose"][:6])
        cur = pose6d_to_mat(f["action_gripper_pose"][:6])
        d[:6] = mat_to_pose6d(np.linalg.inv(last) @ cur)
        deltas.append(d)
    return deltas


class DataCollector(Node):
    def __init__(self, args):
        super().__init__("data_collector_node")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.args = args
        self.logger = self.get_logger()
        self.bridge = CvBridge()

        # root/<dataset>/<exp_id>/<demo_id>/steps/<step>/
        self.exp_dir = os.path.join(os.path.expanduser(args.root), args.dataset_name, f"{args.exp_id:04d}")
        os.makedirs(self.exp_dir, exist_ok=True)
        # metadata lives OUTSIDE the dataset tree: Seer's index builder globs every entry in there as a demo
        self.meta_dir = os.path.join(os.path.expanduser(args.root), "_meta", args.dataset_name, f"{args.exp_id:04d}")
        os.makedirs(self.meta_dir, exist_ok=True)

        self.lock = threading.Lock()
        self.latest_primary = None      # sensor_msgs/Image
        self.latest_wrist = None
        self.latest_joints = None       # dict name -> pos
        self.gripper_cmd = -1.0         # pick_place.reset_panda closes the gripper first
        self.is_recording = False      # pick_place subprocess running
        self.episode_active = None     # /episode_active from pick_place (None = never received -> ignore)
        self.buffer = []

        freq = 15
        self.create_subscription(Image, args.primary_topic, self.on_primary, freq)
        self.create_subscription(Image, args.wrist_topic, self.on_wrist, freq)
        self.create_subscription(JointState, "/joint_states", self.on_joints, freq)
        self.create_subscription(JointState, "/gripper_command", self.on_gripper_cmd, freq)
        self.create_subscription(Bool, "/episode_active", self.on_episode_active, 1)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(1.0 / args.hz, self.record_timestep)
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.logger.info(f"Collector ready. Writing to {self.exp_dir}")

    # ============================= Callbacks =============================
    def on_primary(self, msg):
        self.latest_primary = msg

    def on_wrist(self, msg):
        self.latest_wrist = msg

    def on_joints(self, msg):
        self.latest_joints = dict(zip(msg.name, msg.position))

    def on_gripper_cmd(self, msg):
        if msg.position:
            self.gripper_cmd = 1.0 if msg.position[0] > 0 else -1.0

    def on_episode_active(self, msg):
        self.episode_active = msg.data

    # ---------------- helpers ----------------
    def ee_pose6d(self, stamp):
        """EE pose in panda_link0 at the image stamp (falls back to latest)."""
        for t in (Time.from_msg(stamp), Time()):
            try:
                tf = self.tf_buffer.lookup_transform("panda_link0", "panda_hand_tcp", t)
                tr, q = tf.transform.translation, tf.transform.rotation
                eul = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
                return np.array([tr.x, tr.y, tr.z, *eul])
            except Exception:
                continue
        return None

    def encode(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
        return buf.tobytes() if ok else None


    def record_timestep(self):
        if not self.is_recording or self.episode_active is False:
            return
        if self.latest_primary is None or self.latest_wrist is None or self.latest_joints is None:
            self.logger.warn("waiting for images / joint states", throttle_duration_sec=2.0)
            return
        pose = self.ee_pose6d(self.latest_primary.header.stamp)
        if pose is None:
            self.logger.warn("TF panda_link0->panda_hand_tcp unavailable", throttle_duration_sec=1.0)
            return
        try:
            primary = self.encode(self.latest_primary)
            wrist = self.encode(self.latest_wrist)
        except Exception as e:
            self.logger.error(f"image conversion failed: {e}")
            return
        joints = np.array([self.latest_joints.get(n, np.nan) for n in JOINT_NAMES], dtype=np.float32)
        with self.lock:
            self.buffer.append({
                "primary": primary,
                "wrist": wrist,
                "joints": joints,
                "gripper_pose": pose,
                "gripper_open_state": self.gripper_cmd,
            })

    # ---------------- post-processing ----------------
    def build_episode(self, raw):
        """
        raw frames -> Seer steps: attach next-frame targets, drop idle frames, trim after release.
        """
        if len(raw) < 2:
            return []
        # action at t = pose/gripper at t+1 (last frame has no target -> dropped)
        frames = []
        for t in range(len(raw) - 1):
            f = dict(raw[t])
            nxt = raw[t + 1]
            f["action_gripper_pose"] = np.concatenate([nxt["gripper_pose"], [nxt["gripper_open_state"]]])
            frames.append(f)

        # trim: keep up to `post_release` frames after the last close->open transition (the release)
        g = [f["action_gripper_pose"][6] for f in frames]
        release = [i for i in range(1, len(g)) if g[i - 1] < 0 and g[i] > 0]
        if release:
            frames = frames[: min(len(frames), release[-1] + 1 + self.args.post_release)]

        # filter idle frames the same way Seer's filter_real_data does
        kept, prev_g = [], None
        for f in frames:
            a = f["action_gripper_pose"]
            moving = np.any(np.abs(a[:3] - f["gripper_pose"][:3]) >= self.args.idle_thresh)
            g_changed = prev_g is not None and a[6] != prev_g
            if moving or g_changed or prev_g is None:
                kept.append(f)
            prev_g = a[6]
        # recompute targets on the filtered sequence so consecutive kept frames chain correctly
        for i in range(len(kept) - 1):
            nxt = kept[i + 1]
            kept[i]["action_gripper_pose"] = np.concatenate([nxt["gripper_pose"], [nxt["action_gripper_pose"][6]]])
        deltas = compute_delta_actions(kept)
        for f, d in zip(kept, deltas):
            f["delta"] = d
        return kept

    def save_episode(self, demo_idx):
        with self.lock:
            raw, self.buffer = self.buffer, []
        steps = self.build_episode(raw)
        if len(steps) < self.args.min_steps:
            self.logger.warn(f"episode {demo_idx}: only {len(steps)} usable steps (raw {len(raw)}); skipping")
            return False
        steps_dir = os.path.join(self.exp_dir, f"{demo_idx:06d}", "steps")
        os.makedirs(steps_dir, exist_ok=True)
        self.logger.info(f"episode {demo_idx}: saving {len(steps)} steps (raw {len(raw)}) -> {steps_dir}")

        def write(t, f):
            d = os.path.join(steps_dir, f"{t:04d}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "image_primary.jpg"), "wb") as fh:
                fh.write(f["primary"])
            with open(os.path.join(d, "image_wrist.jpg"), "wb") as fh:
                fh.write(f["wrist"])
            np.savez_compressed(
                os.path.join(d, "other.npz"),
                joints=f["joints"],
                gripper_pose=f["gripper_pose"].astype(np.float32),
                gripper_open_state=np.array([f["gripper_open_state"]], dtype=np.float32),
                action_gripper_pose=f["action_gripper_pose"].astype(np.float32),
                delta_cur_2_last_action=f["delta"].astype(np.float32),
                # Seer reads this with .tobytes().decode("utf-8"); a unicode array would come back
                # UTF-32 encoded (NUL-padded), so store the raw UTF-8 bytes instead.
                language_instruction=np.frombuffer(self.args.instruction.encode("utf-8"), dtype=np.uint8),
            )

        list(self.pool.map(lambda tf: write(*tf), enumerate(steps)))
        # attach the latest pick_place metadata (cube pose, place target, start joints) to this demo
        meta_dir = self.meta_dir
        if os.path.isdir(meta_dir):
            metas = sorted(f for f in os.listdir(meta_dir) if f.startswith("episode_meta_"))
            if metas:
                os.replace(os.path.join(meta_dir, metas[-1]),
                           os.path.join(self.exp_dir, f"{demo_idx:06d}", "episode_meta.json"))
        pos = np.array([s["delta"][:3] for s in steps])
        self.logger.info(f"episode {demo_idx} saved. |dpos| max={np.abs(pos).max():.4f} m, "
                         f"mean={np.abs(pos).mean():.4f} m (training normalises by 0.02)")
        return True

    # ---------------- episode driver ----------------
    def run_pick_place(self):
        # pick_place randomises the cube pose (via /set_body_state) and the arm start in its reset_env();
        # forward the randomisation flags and collect its per-episode metadata beside the data.
        meta_dir = self.meta_dir
        pp_flags = (f"--episodes 1 --set_body_service {self.args.set_body_service} --cube_body {self.args.cube_body} "
                    f"--cube_range {' '.join(map(str, self.args.cube_range))} "
                    f"--place_range {' '.join(map(str, self.args.place_range))} "
                    f"--start_jitter {self.args.start_jitter} --meta_dir {meta_dir} {self.args.pp_extra}")
        cmd = f"cd {self.args.ws} && source install/setup.bash && ros2 run mujoco_control mujoco_pick_place {pp_flags}"
        self.logger.info(f"launching: {cmd}")
        self.is_recording = True
        t0 = time.time()
        if self.args.use_terminal:
            # NOTE: the return code is gnome-terminal's, not the script's, and the window closes on a crash,
            # so tracebacks are lost. Prefer the default (direct child process) when debugging.
            res = subprocess.run(["gnome-terminal", "--wait", "--", "bash", "-c", cmd], capture_output=True, text=True)
            rc, err = res.returncode, res.stderr
        else:
            # direct child: stdout/stderr go to THIS terminal, so a crash in pick_place is visible here
            res = subprocess.run(["bash", "-c", cmd])
            rc, err = res.returncode, ""
        self.is_recording = False
        elapsed = time.time() - t0
        if rc != 0:
            self.logger.error(f"pick-and-place exited with code {rc} after {elapsed:.1f}s\n{err}")
            return False
        if elapsed < self.args.min_episode_sec:
            self.logger.error(f"pick-and-place finished after only {elapsed:.1f}s - it almost certainly crashed "
                              f"on startup (argparse / import error?). Run the command above by hand to see why.")
            return False
        return True

    def collect(self):
        idx, done = self.args.start_from, 0
        while done < self.args.episodes and rclpy.ok():
            self.logger.info(f"--- episode {idx} ---")
            ok = self.run_pick_place() and self.save_episode(idx)
            if ok:
                done += 1
                idx += 1
            else:
                with self.lock:
                    self.buffer = []
                self.logger.warn("retrying episode")
            time.sleep(self.args.cooldown)
        self.logger.info("data collection finished.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--start_from", type=int, default=0, help="demo index to continue from")
    ap.add_argument("--root", default="~/mujoco_data", help="Seer root_dir")
    ap.add_argument("--dataset_name", default="panda_pick_place", help="Seer real_dataset_names")
    ap.add_argument("--exp_id", type=int, default=0)
    ap.add_argument("--instruction", default="Pick up the blue cube, and place it.")
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--idle_thresh", type=float, default=5e-4, help="m; Seer filter_real_data threshold")
    ap.add_argument("--post_release", type=int, default=5, help="frames kept after the gripper opens at place")
    ap.add_argument("--min_steps", type=int, default=15, help="skip episodes shorter than this (window is 10)")
    ap.add_argument("--cooldown", type=float, default=5.0, help="s between episodes (pick_place already waits+resets)")
    # ---- randomisation (forwarded to mujoco_pick_place) ----
    ap.add_argument("--set_body_service", default="/set_body_state")
    ap.add_argument("--cube_body", default="cube", help="MJCF body name of the cube")
    ap.add_argument("--cube_range", type=float, nargs=4, default=[0.30, 0.60, -0.30, 0.30],
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"), help="cube start region (panda_link0)")
    ap.add_argument("--place_range", type=float, nargs=4, default=[0.30, 0.60, -0.30, 0.30],
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"), help="place target region (panda_link0)")
    ap.add_argument("--start_jitter", type=float, default=0.05, help="rad of random offset on the home joints")
    ap.add_argument("--pp_extra", default="", help="any extra flags for mujoco_pick_place, e.g. '--link0_in_world 0 0 0 180'")
    ap.add_argument("--ws", default=os.path.expanduser("~/multipanda_ws"), help="ROS 2 workspace to source before running pick_place")
    ap.add_argument("--use_terminal", action="store_true", help="run pick_place in a gnome-terminal window (hides crashes)")
    ap.add_argument("--min_episode_sec", type=float, default=8.0, help="runs shorter than this are treated as crashes")
    ap.add_argument("--primary_topic", default="/mujoco_server/cameras/third_person_camera/rgb/image_raw")
    ap.add_argument("--wrist_topic", default="/mujoco_server/cameras/wrist_camera/rgb/image_raw")
    return ap.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = DataCollector(args)
    threading.Thread(target=node.collect, daemon=True).start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pool.shutdown(wait=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()