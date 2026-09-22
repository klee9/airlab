"""
An ROS2 node that sends observations to the policy server.

ssh -N -L 8000:127.0.0.1:8000 keon@165.194.27.147
ssh -p 408 -N -L 8080:127.0.0.1:8080 yina@165.194.27.149

*port 8000 is shared by all pi clients, so change it to something else.

REPLAY mode (no policy):
    ros2 run <pkg> <node> --ros-args -p replay_file:=~/ep1.npz
"""

import os      # REPLAY
import cv2
import time
import rclpy
import threading
import numpy as np

from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
from openpi_client import image_tools, websocket_client_policy

from rclpy.action import ActionClient
from franka_msgs.action import Grasp, Move

FREQ = 15
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

# Franka Panda joint limits (rad)
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

CHUNK_SIZE = 16
EXEC_STEPS = 16
VEL_LIMIT = 1.0        # rad/s
MARGIN = 0.05          # joint limit margin

class PI05Node(Node):
    def __init__(self):
        super().__init__("pi05")

        self.busy = threading.Lock()
        self.bridge = CvBridge()
        self.logger = self.get_logger()

        # ===================== Parameters =====================
        self.declare_parameter("sim_time", True)
        self.declare_parameter("instruction", "Pick up the blue cube, and place it.")
        self.declare_parameter("send_cmd", True)
        self.declare_parameter("replay_file", "")          # REPLAY: path to .npz from export_episode.py
        self.declare_parameter("replay_start_tol", 0.05)   # REPLAY: max |q - q_demo[0]| (rad) to start

        p = lambda n: self.get_parameter(n).value # noqa: E731

        if p("sim_time"):
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        # ============== Observation target data ==============
        self.latest_base_img = None
        self.latest_wrist_img = None
        self.latest_joint_pos = None
        self.latest_gripper_pos = None
        self.prompt = p("instruction")

        # ================== Subscriptions ====================
        depth = 10
        cb = ReentrantCallbackGroup()

        self.create_subscription(Image, "/mujoco_server/cameras/third_person_camera/rgb/image_raw", self.on_base_img, depth, callback_group=cb)
        self.create_subscription(Image, "/mujoco_server/cameras/wrist_camera/rgb/image_raw", self.on_wrist_img, depth, callback_group=cb)
        self.create_subscription(JointState, "/joint_states", self.on_joint_pos, depth, callback_group=cb)
        self.create_subscription(JointState, "/panda_gripper_sim_node/joint_states", self.on_gripper_pos, depth, callback_group=cb)

        self.timer = self.create_timer(1.0/FREQ, self.step, callback_group=cb)

        # =============== For sending cmds later ===============
        self.k = 0
        self.chunk = None
        self.enabled = p("send_cmd")
        self.traj_pub = self.create_publisher(JointTrajectory, "/panda_arm_controller/joint_trajectory", 1)

        self.grasp_client = ActionClient(self, Grasp, "/panda_gripper_sim_node/grasp")
        self.move_client  = ActionClient(self, Move,  "/panda_gripper_sim_node/move")
        self.last_grip = None

        # ===================== REPLAY =====================
        self.replay = None
        self.replay_t = 0
        self.replay_done = False
        self.replay_tol = float(p("replay_start_tol"))
        path = p("replay_file")
        if path:
            d = np.load(os.path.expanduser(path))
            self.replay = {"actions": d["actions"].astype(np.float64),
                           "q":       d["q"].astype(np.float64)}
            T = len(self.replay["actions"])
            self.logger.info(f"[replay] loaded {path}: T={T}  q0={np.round(self.replay['q'][0], 3)}")

    # ===================== Callbacks =====================
    def on_base_img(self, msg: Image):
        if self.latest_base_img is None:
            self.logger.info(f"first third-person image: {msg.width}x{msg.height} {msg.encoding}")
        self.latest_base_img = msg

    
    def on_wrist_img(self, msg: Image):
        if self.latest_wrist_img is None:
            self.logger.info(f"first wrist image: {msg.width}x{msg.height} {msg.encoding}")
        self.latest_wrist_img = msg

    
    def on_joint_pos(self, msg: JointState):
        self.latest_joint_pos = msg


    def on_gripper_pos(self, msg: JointState):
        self.latest_gripper_pos = msg


    def step(self):
        missing = [n for n, v in (("base img", self.latest_base_img),
                                ("wrist img", self.latest_wrist_img),
                                ("joints", self.latest_joint_pos),
                                ("gripper", self.latest_gripper_pos)) if v is None]
        if missing:
            self.logger.info(f"waiting for: {', '.join(missing)}", throttle_duration_sec=2.0)
            return

        # still executing the current chunk:
        # advance the gripper in lockstep with the arm trajectory
        if self.chunk is not None and self.k < min(EXEC_STEPS, len(self.chunk)):
            self.command_gripper(bool(self.chunk[self.k, 7] > 0.5))
            self.k += 1
            return

        # REPLAY: ground-truth actions instead of the policy
        if self.replay is not None:
            self.replay_step()
            return

        if not self.busy.acquire(blocking=False):
            self.logger.info("previous inference still running, skipping tick", throttle_duration_sec=2.0)
            return
        try:
            obs, q_now = self.build_observation()
            if obs is None:
                return

            t0 = time.time()
            self.chunk = self.client.infer(obs)["actions"]
            self.logger.info(f"infer {(time.time()-t0)*1000:.0f} ms  shape={self.chunk.shape} "
                            f"vel[0]={np.round(self.chunk[0,:7],3)} grip={self.chunk[0,7]:.2f}")

            self.publish_chunk(self.chunk, q_now)
            self.command_gripper(bool(self.chunk[0, 7] > 0.5))   # action 0 of the new chunk
            self.k = 1                                           # next tick consumes index 1
        finally:
            self.busy.release()


    # ===================== REPLAY =====================
    def read_q(self):
        msg = self.latest_joint_pos
        try:
            idx = [msg.name.index(n) for n in JOINT_NAMES]
        except ValueError:
            return None
        return np.asarray(msg.position, dtype=np.float64)[idx]


    def replay_step(self):
        q_now = self.read_q()
        if q_now is None:
            return

        acts, q_demo = self.replay["actions"], self.replay["q"]
        T, t = len(acts), self.replay_t

        if t >= T:
            if not self.replay_done:
                self.logger.info(f"[replay] done. final |q - q_demo[-1]| = "
                                 f"{np.round(q_now - q_demo[-1], 3)}")
                self.replay_done = True
            return

        # the demo's velocities only make sense from the demo's start pose
        if t == 0:
            err0 = np.abs(q_now - q_demo[0]).max()
            if err0 > self.replay_tol:
                self.logger.error(f"[replay] start pose off by {err0:.3f} rad (tol {self.replay_tol}). "
                                  f"q_now={np.round(q_now, 3)}  q_demo[0]={np.round(q_demo[0], 3)}",
                                  throttle_duration_sec=2.0)
                return

        chunk = acts[t:t + CHUNK_SIZE]
        vmax = np.abs(chunk[:, :7]).max()
        if vmax > VEL_LIMIT:
            self.logger.warn(f"[replay] t={t}: |v|max={vmax:.2f} > VEL_LIMIT={VEL_LIMIT} -> GT is being clipped")

        # how far has the real arm drifted from the demo so far?
        err = q_now - q_demo[t]
        self.logger.info(f"[replay] t={t:3d}/{T}  |q-q_demo|max={np.abs(err).max():.3f}  "
                         f"per-joint={np.round(err, 3)}")

        self.chunk = chunk
        self.publish_chunk(self.chunk, q_now)
        self.command_gripper(bool(self.chunk[0, 7] > 0.5))
        self.k = 1
        self.replay_t = t + min(EXEC_STEPS, len(self.chunk))


    def publish_chunk(self, actions, q_now):
        dt = 1.0 / FREQ
        msg = JointTrajectory()
        # msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = JOINT_NAMES

        q = q_now.astype(np.float64).copy()
        for k in range(len(actions)):
            v = np.clip(actions[k, :7], -VEL_LIMIT, VEL_LIMIT)
            q = np.clip(q + v * dt, Q_MIN + MARGIN, Q_MAX - MARGIN)

            pt = JointTrajectoryPoint()
            pt.positions  = [float(x) for x in q]
            pt.velocities = [float(x) for x in v]
            t = (k + 1) * dt
            pt.time_from_start = Duration(sec=int(t), nanosec=int(round((t % 1.0) * 1e9)))
            msg.points.append(pt)

        if self.enabled:
            self.traj_pub.publish(msg)
        else:
            self.logger.info(f"[dry] would send {len(msg.points)} pts, "
                            f"Δq={np.round(q - q_now, 4)}", throttle_duration_sec=1.0)
        
    def connect(self):
        self.client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=8080)
        self.logger.info("connected to policy server")


    def build_observation(self):
        msg = self.latest_joint_pos
        try:
            idx = [msg.name.index(n) for n in JOINT_NAMES]
        except ValueError:
            return None, None
        joint_position = np.asarray(msg.position, dtype=np.float32)[idx]

        finger = self.latest_gripper_pos.position[0]
        gripper_position = 0.0 if finger > 0.035 else 1.0   # open only when near fully open

        obs = {
            "observation/exterior_image_1_left": self.process_image(self.latest_base_img),
            "observation/wrist_image_left":      self.process_image(self.latest_wrist_img),
            "observation/joint_position":        joint_position,
            "observation/gripper_position":      np.array([gripper_position], np.float32),
            "prompt": self.prompt,
        }
        return obs, joint_position


    def command_gripper(self, closed: bool):
        if closed == self.last_grip:
            return
        if closed:
            g = Grasp.Goal()
            g.width, g.speed, g.force = 0.0, 0.5, 20.0
            g.epsilon.inner = g.epsilon.outer = 0.08
            self.grasp_client.send_goal_async(g)
        else:
            m = Move.Goal()
            m.width, m.speed = 0.08, 0.5
            self.move_client.send_goal_async(m)
        self.last_grip = closed
        tag = f" @ replay t={self.replay_t - 1 + self.k}" if self.replay is not None else ""   # REPLAY
        self.logger.info(f"gripper {'CLOSE' if closed else 'OPEN'}{tag}")


    def process_image(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = image_tools.resize_with_pad(rgb, 224, 224)
        # For debugging only
        cv2.imwrite("/tmp/infer.png", cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR))
        return image_tools.convert_to_uint8(resized_rgb)


def main():
    rclpy.init()
    node = PI05Node()
    if node.replay is None:     # REPLAY: no policy server needed
        node.connect()
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