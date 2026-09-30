"""
ROS2 node for Seer

Approach:
  - Observe required data for Seer.
  - Convert the observations to the Seer-expected format.
  - Send the observations to the policy server.
  - Fetch the returned target delta actions predicted by the policy server.
  - Execute the fetched actions.
"""

import io
import pickle
import socket
import struct
import threading
 
import cv2
import numpy as np
import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R
from PIL import Image as PILImage
from tf2_ros import Buffer, TransformListener
from franka_msgs.action import Grasp, Move
 
OPEN = 1.0
CLOSE = -1.0
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


class RemoteVLA():
    def __init__(self, host="127.0.0.1", port=5001, timeout=5.0, jpeg_quality=90):
        self.lock = threading.Lock()
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.jpeg_quality = jpeg_quality


    def call(self, req):
        """
        Send a given request to the policy server, and return the result.
        """
        with self.lock:
            self.send(req)
            (n,) = struct.unpack(">I", self.recv_exact(4))
            response = pickle.loads(self.recv_exact(n))
        if not response.get("ok"):
            raise RuntimeError(f"server error: {response.get('error')}")
        return response


    def convert_to_jpeg(self, rgb):
        """
        Convert NumPy array to JPEG bytes, then return those bytes.
        Since sending images over network is expensive, JPEG compression is used for optimization.
        """
        buf = io.BytesIO()
        PILImage.fromarray(rgb).save(buf, format="JPEG", quality=self.jpeg_quality)
        return buf.getvalue()


    def send(self, observation):
        """ 
        Send the observation to the policy server.
        """
        payload = pickle.dumps(observation, protocol=pickle.HIGHEST_PROTOCOL)
        self.sock.sendall(struct.pack(">I", len(payload)) + payload)


    def recv_exact(self, n):
        """
        Receive exactly n bytes from the policy server. (NOTE: socket.recv() receives 1024 bytes)
        """
        buf = bytearray()
        while len(buf) < n:
            # Receive until n bytes have been filled in the buffer.
            left = self.sock.recv(n - len(buf))
            if not left:
                raise ConnectionError("server closed the connection")
            buf.extend(left)
        return bytes(buf)


    def predict(self, primary_rgb, wrist_rgb, state7, instruction):
        """
        Run inference and return the raw normalized 7D action.
        """
        rep = self.call({
            "cmd": "predict",
            "primary": self.convert_to_jpeg(primary_rgb),
            "wrist": self.convert_to_jpeg(wrist_rgb),
            "state": [float(v) for v in state7],
            "instruction": instruction,
        })
        return np.asarray(rep["delta"], dtype=np.float64)


    def reset(self):
        self.call({"cmd": "reset"})


    def close(self):
        self.sock.close()


class SeerNode(Node):
    def __init__(self):
        super().__init__("seer")
        self.logger = self.get_logger()

        # ======================== Server Parameters ========================
        self.declare_parameter("server_host", "127.0.0.1")
        self.declare_parameter("server_port", 5001)
        self.declare_parameter("server_timeout", 5.0)
        self.declare_parameter("jpeg_quality", 90)

        # ======================= Inference Parameters ======================
        self.declare_parameter("control_frequency", 15.0)
        self.declare_parameter("instruction", "Pick up the blue cube, and place it.")

        # ======================== Control Parameters =======================
        self.declare_parameter("gripper_eps", 0.002)
        self.declare_parameter("base_link", "panda_link0")
        self.declare_parameter("hand_link", "panda_hand_tcp")
        self.declare_parameter("max_rel_pos", 0.02) # this was used in the official repo for normalization scale
        self.declare_parameter("max_rel_orn", 0.05) # this as well
        self.declare_parameter("grasp_force", 50.0)

        # ========================  Topic Parameters ========================
        self.declare_parameter("third_person_image_topic", "/mujoco_server/cameras/third_person_camera/rgb/image_raw")
        self.declare_parameter("wrist_image_topic", "/mujoco_server/cameras/wrist_camera/rgb/image_raw")
        self.declare_parameter("gripper_state_topic", "/panda_gripper_sim_node/joint_states")
        self.declare_parameter("instruction_topic", "/language_instruction")
        self.declare_parameter("goal_pose_topic", "/seer/goal_pose")
        self.declare_parameter("arm_joint_state_topic", "/joint_states")
        self.declare_parameter("gripper_topic_ns", "/panda_gripper_sim_node")

        # ========================== Miscellaneous ==========================
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        # ============================ Variables ============================
        self.eps = p("gripper_eps")
        self.vla = RemoteVLA(host=p("server_host"), port=p("server_port"), timeout=p("server_timeout"), jpeg_quality=p("jpeg_quality"))
        self.bridge = CvBridge()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.base_link = p("base_link")
        self.hand_link = p("hand_link")

        self.max_rel_pos = p("max_rel_pos")
        self.max_rel_orn = p("max_rel_orn")
        self.grasp_force = p("grasp_force")

        self.latest_third_person_image = None
        self.latest_wrist_image = None
        self.latest_joint_states = None
        self.latest_gripper_state = None
        self.last_target_T = None
        self.instruction = p("instruction") 

        # =========================== Subs & Pubs ===========================
        sensor_cb = ReentrantCallbackGroup()
        control_cb = MutuallyExclusiveCallbackGroup()   # control_step never overlaps itself
        depth = 15

        self.create_subscription(Image, p("third_person_image_topic"), self.on_third_person_image, depth, callback_group=sensor_cb)
        self.create_subscription(Image, p("wrist_image_topic"), self.on_wrist_image, depth, callback_group=sensor_cb)
        self.create_subscription(JointState, p("arm_joint_state_topic"), self.on_arm_joint_state, depth, callback_group=sensor_cb)
        self.create_subscription(JointState, p("gripper_state_topic"), self.on_gripper_state, depth, callback_group=sensor_cb)
        self.create_subscription(String, p("instruction_topic"), self.on_instruction, 1, callback_group=sensor_cb)
 
        self.goal_pub = self.create_publisher(JointState, p("goal_pose_topic"), 1)

        self.move_client = ActionClient(self, Move, f"{p('gripper_topic_ns')}/move", callback_group=sensor_cb)
        self.grasp_client = ActionClient(self, Grasp, f"{p('gripper_topic_ns')}/grasp", callback_group=sensor_cb)
 
        self.create_timer(1.0 / p("control_frequency"), self.control_step, callback_group=control_cb)
        self.logger.info("Seer node started.")

        

    # ======================== Callbacks ========================
    def on_third_person_image(self, msg):
        if self.latest_third_person_image is None:
            self.logger.info(f"First third-person image({msg.width}x{msg.height}) received.")
        self.latest_third_person_image = msg

    
    def on_wrist_image(self, msg):
        if self.latest_wrist_image is None:
            self.logger.info(f"First wrist image({msg.width}x{msg.height}) received.")
        self.latest_wrist_image = msg


    def on_arm_joint_states(self, msg):
        self.latest_joint_states = msg


    def on_gripper_state(self, msg):
        # NOTE: maybe utilize the object width to compute the threshold?
        self.latest_gripper_state = OPEN if msg.position[0] > 0.04 - self.eps else CLOSE


    def on_instruction(self, msg):
        if msg.data != self.instruction:
            self.logger.info(f"New instruction: '{msg.data}' received. Resetting history...")
            self.instruction = msg.data
            self.vla.reset()
            self.last_target_T = None
            self.reset_episode_state()
            self.logger.info("Episode state reset, policy is running.")

    
    # ===================== Commanding Panda ====================
    def get_ee_T(self):
        """
        Looks up the pose of the hand relative to the base as a 4x4 transformation matrix.
        
        The matrix is shaped as:
          
          [ R T ]
          [ 0 1 ]
        """
        try:
            tf = self.tf_buffer.lookup_transform(self.base_link, self.hand_link, rclpy.time.Time())
        except Exception as e:
            self.logger.warn(f"Transform lookup failed: {e}", throttle_duration_sec=1.0)
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        T = np.eye(4)
        T[:3, 3] = [t.x, t.y, t.z]
        T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return T


    def decode_action(self, action, base_T):
        """
        Applies one predicted delta to a base pose.

        Args:
        - action: `[dx, dy, dz, drx, dry, drz, gripper]`; normalized, expressed in base_T's local frame.
        - base_T: 4x4 pose the delta is applied to (the previous *target*, in base_link frame).

        Returns:
        - target_T: 4x4 target end-effector pose in base_link frame.
        - gripper: OPEN or CLOSE.
        """
        action = np.asarray(action, dtype=np.float64)
        delta = np.concatenate([action[:3] * self.max_rel_pos, action[3:6] * self.max_rel_orn])

        target_T = base_T @ pose6d_to_mat(delta)                        # local-frame composition
        target_T[:3, :3] = R.from_matrix(target_T[:3, :3]).as_matrix()  # re-orthonormalize

        gripper = OPEN if action[6] > 0.5 else CLOSE # NOTE: Seer outputs sigmoid value for gripper
        return target_T, gripper


    def control_step(self):
        if self.last_target_T is None: # first step of an episode
            ee_T = self.get_ee_T()
            if ee_T is None:
                self.logger.error("Unable to fetch the end-effector pose. Aborting...")
                return
            self.last_target_T = ee_T

        third_person_image = self.to_rgb(self.latest_third_person_image)
        wrist_image = self.to_rgb(self.latest_wrist_image)
        state7d = np.zeros(7)
        state7d[:3] = ee_T[:3, 3] # translation values (x, y, z)
        state7d[3:6] = R.from_matrix(ee_T[:3, :3]).as_euler("xyz")
        state7d[6] = self.latest_gripper_state
        
        action = self.vla.predict(third_person_image, wrist_image, state7d, self.instruction)
        self.last_target_T, gripper = self.decode_action(action, self.last_target_T)
        self.send_command(self.last_target_T, gripper)


    def send_command(self, target_T, gripper):
        """
        Command the Panda arm to move according to the predicted target delta actions.
        """
        msg = JointState()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = target_T[:3, 3]
        qx, qy, qz, qw = R.from_matrix(target_T[:3, :3]).as_quat()
        msg.pose.orientation.x, msg.pose.orientation.y = qx, qy
        msg.pose.orientation.z, msg.pose.orientation.w = qz, qw
        msg.q_n = list(self.latest_joint_states)
        self.goal_pub.publish(msg)
        self.send_gripper(gripper)


    def send_gripper(self, cmd):
        """
        Send the gripper goal to the GripperAction client. This fires and returns immediately.
        """
        if cmd == OPEN:
            goal = Move.Goal(width=0.08, speed=0.1)
            client = self.move_client
        else:
            goal = Grasp.Goal(width=0.0, speed=0.1, force=self.grasp_force)
            goal.epsilon.inner = 0.08 # accept any object width
            goal.epsilon.outer = 0.08
            client = self.grasp_client
        if not client.server_is_ready():
            self.logger.error("Gripper action server not available.", throttle_duration_sec=2.0)
            return
        client.send_goal_async(goal) # use async so it doesn't block
        self.logger.info(f"step {self.step_count}: gripper {'OPEN' if cmd == OPEN else 'CLOSE'}")


    def destroy_node(self):
        try:
            self.vla.close()
        finally:
            super().destroy_node()

    
    def to_rgb(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ======================== Utility Functions ========================


def pose6d_to_mat(pose6d):
    T = np.eye(4)
    T[:3, 3] = pose6d[:3]
    T[:3, :3] = R.from_euler("xyz", pose6d[3:6]).as_matrix()
    return T


def mat_to_pose6d(T):
    out = np.zeros(6)
    out[:3] = T[:3, 3]
    out[3:6] = R.from_matrix(T[:3, :3]).as_euler("xyz")
    return out


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi




# class VLANode(Node):
#     def __init__(self):
#         super().__init__("vla_node")
#         self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

#         # --- server (through the ssh -L tunnel) ---
#         self.declare_parameter("server_host", "127.0.0.1")
#         self.declare_parameter("server_port", 5001)
#         self.declare_parameter("jpeg_quality", 90)
#         # --- control ---
#         self.declare_parameter("instruction", "Pick up the blue cube, and place it.")  # same string as training
#         self.declare_parameter("control_freq", 15.0)
#         self.declare_parameter("action_mode", "seer_relative")  # "seer_relative" | "goal_offset"
#         self.declare_parameter("integrate_on", "leash")         # "leash" | "measured" | "target"
#         self.declare_parameter("leash_dist", 0.03)              # leash: max distance the target may run ahead of the arm (m)
#         self.declare_parameter("z_min", 0.015)                  # floor for the commanded TCP height (m); <=0 disables
#         self.declare_parameter("max_rel_pos", 0.02)             # normalization used at training
#         self.declare_parameter("max_rel_orn", 0.05)
#         self.declare_parameter("base_frame", "panda_link0")
#         self.declare_parameter("tcp_frame", "panda_hand_tcp")
#         self.declare_parameter("dump_dir", "")        # if set: save every step's obs/action here for debugging
#         self.declare_parameter("gripper_debounce", 3)  # consecutive ticks a new gripper state must be predicted before acting
#         self.declare_parameter("stop_after_release", True)  # done once the gripper has closed and then re-opened
#         self.declare_parameter("max_steps", 600)       # hard cap on policy steps per episode (600 = 40 s at 15 Hz)
#         self.declare_parameter("release_min_z", 0.0)   # >0: ignore a predicted release while TCP z is above this (m)
#         # --- topics ---
#         self.declare_parameter("primary_image_topic", "/mujoco_server/cameras/third_person_camera/rgb/image_raw")
#         self.declare_parameter("wrist_image_topic", "/mujoco_server/cameras/wrist_camera/rgb/image_raw")
#         self.declare_parameter("gripper_topic", "/gripper_command")
#         self.declare_parameter("instruction_topic", "/language_instruction")
#         self.declare_parameter("target_pose_topic", "/vla/target_pose")
#         self.declare_parameter("target_gripper_topic", "/vla/target_gripper")
#         p = lambda n: self.get_parameter(n).value  # noqa: E731

#         self.instruction = p("instruction")
#         self.action_mode = p("action_mode")
#         self.integrate_on = p("integrate_on")
#         self.leash_dist = p("leash_dist")
#         self.z_min = p("z_min")
#         self.max_rel_pos = p("max_rel_pos")
#         self.max_rel_orn = p("max_rel_orn")
#         self.base_frame = p("base_frame")
#         self.tcp_frame = p("tcp_frame")
#         self.bridge = CvBridge()
#         self.dump_dir = p("dump_dir")
#         self.dump_idx = 0
#         self.gripper_debounce = p("gripper_debounce")
#         self.stop_after_release = p("stop_after_release")
#         self.max_steps = p("max_steps")
#         self.release_min_z = p("release_min_z")
#         self._reset_episode_state()
#         if self.dump_dir:
#             import os
#             os.makedirs(self.dump_dir, exist_ok=True)
#             self.get_logger().info(f"dumping observations/actions to {self.dump_dir}")

#         # latest raw messages (converted in step(), like data_collector.record_timestep)
#         self.latest_primary_msg = None
#         self.latest_wrist_msg = None
#         self.gripper_open = -1.0          # closed by default, same as collector
#         self.last_target_pose6d = None

#         # --- connect to the GPU server ---
#         self.get_logger().info(f"Connecting to Seer server at {p('server_host')}:{p('server_port')} ...")
#         try:
#             self.vla = RemoteVLA(p("server_host"), p("server_port"), jpeg_quality=p("jpeg_quality"))
#         except OSError as e:
#             raise RuntimeError(f"could not reach the Seer server through the tunnel: {e}") from e
#         self.get_logger().info(f"Connected. action_mode={self.action_mode} integrate_on={self.integrate_on} leash={self.leash_dist} z_min={self.z_min}")

#         # --- TF ---
#         self.tf_buffer = Buffer()
#         self.tf_listener = TransformListener(self.tf_buffer, self)

#         # --- subscribers / publishers ---
#         cb = ReentrantCallbackGroup()
#         depth = 15
#         self.create_subscription(Image, p("primary_image_topic"), self.on_primary, depth, callback_group=cb)
#         self.create_subscription(Image, p("wrist_image_topic"), self.on_wrist, depth, callback_group=cb)
#         self.create_subscription(JointState, p("gripper_topic"), self.on_gripper, depth, callback_group=cb)
#         self.create_subscription(String, p("instruction_topic"), self.on_instruction, 1, callback_group=cb)
        
#         self.pose_pub = self.create_publisher(PoseStamped, p("target_pose_topic"), 1)
#         self.gripper_pub = self.create_publisher(Float64, p("target_gripper_topic"), 1)

#         self.busy = threading.Lock()
#         self.timer = self.create_timer(1.0 / p("control_freq"), self.step, callback_group=cb)
#         self.get_logger().info("VLA Node started.")

#     # ========================== Callbacks ========================== #
#     def on_primary(self, msg: Image):
#         if self.latest_primary_msg is None:
#             self.get_logger().info(f"first third-person image: {msg.width}x{msg.height} {msg.encoding}")
#         self.latest_primary_msg = msg

#     def on_wrist(self, msg: Image):
#         if self.latest_wrist_msg is None:
#             self.get_logger().info(f"first wrist image: {msg.width}x{msg.height} {msg.encoding}")
#         self.latest_wrist_msg = msg

#     def on_gripper(self, msg: JointState):
#         if len(msg.position) == 0:
#             return
#         self.gripper_open = 1.0 if msg.position[0] > 0 else -1.0

#     def on_instruction(self, msg: String):
#         if msg.data != self.instruction:
#             self.get_logger().info(f"New instruction: '{msg.data}' -> resetting history")
#             self.instruction = msg.data
#             self.vla.reset()
#             self.last_target_pose6d = None
#             self._reset_episode_state()
#             self.get_logger().info("episode state reset; policy running")

#     def get_ee_pose6d(self):
#         try:
#             tf = self.tf_buffer.lookup_transform(self.base_frame, self.tcp_frame, rclpy.time.Time())
#         except Exception as e:
#             self.get_logger().warn(f"Transform lookup failed: {e}", throttle_duration_sec=1.0)
#             return None
#         t, q = tf.transform.translation, tf.transform.rotation
#         euler = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
#         return np.array([t.x, t.y, t.z, *euler], dtype=np.float64)

#     def _reset_episode_state(self):
#         self.step_count = 0
#         self.grip_cmd = -1.0          # what we have actually commanded (training starts closed)
#         self.grip_pending = None      # candidate new state being debounced
#         self.grip_pending_n = 0
#         self.has_closed = False       # grasp happened
#         self.done = False

#     def filter_gripper(self, raw, ee_pose6d=None):
#         """Debounce the model's +-1 gripper output; returns the state to command."""
#         zinfo = f" at TCP z={ee_pose6d[2]:.3f} m" if ee_pose6d is not None else ""
#         if raw == self.grip_cmd:
#             self.grip_pending, self.grip_pending_n = None, 0
#             return self.grip_cmd
#         # optional release gate: a release predicted while still high is deferred (motion continues)
#         if raw > 0 and self.has_closed and self.release_min_z > 0 and ee_pose6d is not None \
#                 and ee_pose6d[2] > self.release_min_z:
#             self.get_logger().info(f"release predicted{zinfo} - deferred until z < {self.release_min_z}",
#                                    throttle_duration_sec=1.0)
#             self.grip_pending, self.grip_pending_n = None, 0
#             return self.grip_cmd
#         if self.grip_pending == raw:
#             self.grip_pending_n += 1
#         else:
#             self.grip_pending, self.grip_pending_n = raw, 1
#         if self.grip_pending_n >= self.gripper_debounce:
#             self.grip_cmd = raw
#             self.grip_pending, self.grip_pending_n = None, 0
#             if raw < 0:
#                 self.has_closed = True
#                 self.get_logger().info(f"step {self.step_count}: gripper CLOSE{zinfo}")
#             else:
#                 self.get_logger().info(f"step {self.step_count}: gripper OPEN" + (" (release)" if self.has_closed else "") + zinfo)
#                 if self.has_closed and self.stop_after_release:
#                     self.done = True
#         return self.grip_cmd

#     def to_rgb(self, msg):
#         # Decode exactly like the collector (bgr8 via cv_bridge -> cv2.imwrite), then to RGB,
#         # which is what PIL produced when the training loader opened those JPEGs.
#         bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#         return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

#     # ========================== Action decoding ==========================
#     def decode_action(self, act, ee_pose6d):
#         d = act.copy()
#         d[:3] *= self.max_rel_pos
#         d[3:6] *= self.max_rel_orn
#         gripper = 1.0 if d[6] > 0 else -1.0

#         if self.action_mode == "seer_relative":
#             # Seer's deploy.py composes onto the last *target*, which assumes the controller reaches every
#             # target before the next one. With a lagging controller that lets the target run ahead of the
#             # arm (and into the table). "measured" composes onto the current pose instead: closed-loop.
#             # "target": pure open-loop (runs away if the controller lags)
#             # "measured": pure closed-loop (stalls with a soft controller: a 5 mm target gives ~no force)
#             # "leash": integrate on the target so the command stays ahead and produces force, but never
#             #          let it get more than leash_dist from the measured pose.
#             if self.integrate_on in ("target", "leash") and self.last_target_pose6d is not None:
#                 base = self.last_target_pose6d
#             else:
#                 base = ee_pose6d
#             target = mat_to_pose6d(pose6d_to_mat(base) @ pose6d_to_mat(d[:6]))
#             if self.integrate_on == "leash":
#                 off = target[:3] - ee_pose6d[:3]
#                 n = np.linalg.norm(off)
#                 if n > self.leash_dist:
#                     target[:3] = ee_pose6d[:3] + off * (self.leash_dist / n)
#                 # orientation: keep the target within ~0.15 rad of the measured orientation
#                 rel = R.from_euler("xyz", ee_pose6d[3:6]).inv() * R.from_euler("xyz", target[3:6])
#                 rv = rel.as_rotvec(); a = np.linalg.norm(rv)
#                 if a > 0.15:
#                     target[3:6] = (R.from_euler("xyz", ee_pose6d[3:6]) * R.from_rotvec(rv * 0.15 / a)).as_euler("xyz")
#         else:  # "goal_offset": label was (last_frame_pose - current_pose) in the base frame
#             target = ee_pose6d.copy()
#             target[:3] += d[:3]
#             target[3:6] = wrap_pi(target[3:6] + d[3:6])
#         if self.z_min > 0 and target[2] < self.z_min:
#             target[2] = self.z_min
#         return target, gripper

#     # ========================== Control step ==========================
#     def step(self):
#         if self.done:
#             self.get_logger().info("episode finished (holding). Publish a new /language_instruction to restart.",
#                                    throttle_duration_sec=10.0)
#             return
#         ee_pose6d = self.get_ee_pose6d()
#         missing = [n for n, v in (("third person image", self.latest_primary_msg),
#                                   ("wrist image", self.latest_wrist_msg),
#                                   (f"TF {self.base_frame}->{self.tcp_frame}", ee_pose6d)) if v is None]
#         if missing:
#             self.get_logger().warn(f"Waiting for: {', '.join(missing)}", throttle_duration_sec=2.0)
#             return
#         if not self.busy.acquire(blocking=False):
#             return
#         try:
#             try:
#                 primary_rgb = self.to_rgb(self.latest_primary_msg)
#                 wrist_rgb = self.to_rgb(self.latest_wrist_msg)
#             except Exception as e:
#                 self.get_logger().error(f"Error converting image: {e}", throttle_duration_sec=2.0)
#                 return

#             state7 = np.concatenate([ee_pose6d, [self.gripper_open]])
#             try:
#                 act = self.vla.predict(primary_rgb, wrist_rgb, state7, self.instruction)
#             except Exception as e:
#                 self.get_logger().error(f"inference failed: {e}", throttle_duration_sec=2.0)
#                 return

#             target_pose6d, raw_gripper = self.decode_action(act, ee_pose6d)
#             self.last_target_pose6d = target_pose6d
#             target_gripper = self.filter_gripper(raw_gripper, ee_pose6d)
#             self.step_count += 1
#             if self.step_count >= self.max_steps:
#                 self.get_logger().warn(f"max_steps ({self.max_steps}) reached; stopping")
#                 self.done = True
#             if self.dump_dir:
#                 import os
#                 d = os.path.join(self.dump_dir, f"{self.dump_idx:04d}")
#                 os.makedirs(d, exist_ok=True)
#                 cv2.imwrite(os.path.join(d, "image_primary.jpg"), cv2.cvtColor(primary_rgb, cv2.COLOR_RGB2BGR))
#                 cv2.imwrite(os.path.join(d, "image_wrist.jpg"), cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))
#                 np.savez(os.path.join(d, "step.npz"), state7=state7, act=act,
#                          target_pose6d=target_pose6d, target_gripper=target_gripper, instruction=self.instruction)
#                 self.dump_idx += 1
#             self.get_logger().debug(f"act={np.round(act, 3)} target={np.round(target_pose6d, 3)} grip={target_gripper}")

#             msg = PoseStamped()
#             msg.header.stamp = self.get_clock().now().to_msg()
#             msg.header.frame_id = self.base_frame
#             msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, target_pose6d[:3])
#             qx, qy, qz, qw = R.from_euler("xyz", target_pose6d[3:6]).as_quat()
#             msg.pose.orientation.x, msg.pose.orientation.y = float(qx), float(qy)
#             msg.pose.orientation.z, msg.pose.orientation.w = float(qz), float(qw)
#             self.pose_pub.publish(msg)

#             g = Float64()
#             g.data = float(target_gripper)
#             self.gripper_pub.publish(g)
#         finally:
#             self.busy.release()

#     def destroy_node(self):
#         try:
#             self.vla.close()
#         finally:
#             super().destroy_node()


def main():
    rclpy.init()
    node = SeerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()