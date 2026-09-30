"""
ROS2 node for Seer

Approach:
  - Observe required data for Seer.
  - Convert the observations to the Seer-expected format.
  - Send the observations to the policy server.
  - Fetch the returned target delta actions predicted by the policy server.
  - Execute the fetched actions.

Usage:
  - client: ssh -N -L 5001:localhost:5001 user@165.194.27.147
  - server: python seer_server.py --model_path your-checkpoint-path --vit_path mae_vit_path --port 5001
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
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R
from PIL import Image as PILImage
from tf2_ros import Buffer, TransformListener
from franka_msgs.action import Grasp, Move
from moveit_msgs.srv import GetPositionIK
from builtin_interfaces.msg import Duration
 
OPEN = 1.0
CLOSE = -1.0
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


class RemoteModel():
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
        self.declare_parameter("ik_group", "panda_arm")
        self.declare_parameter("ik_tip_link", "panda_link8")
        self.declare_parameter("trajectory_duration", 0.1)
        self.declare_parameter("leash_pos", 0.04) # for preventing sudden movements
        self.declare_parameter("leash_rot", 0.01)
        self.declare_parameter("max_rel_pos", 0.02) # this was used in the official repo for normalization scale
        self.declare_parameter("max_rel_orn", 0.05) # this as well
        self.declare_parameter("grasp_force", 50.0)

        # ========================  Topic Parameters ========================
        self.declare_parameter("third_person_image_topic", "/mujoco_server/cameras/third_person_camera/rgb/image_raw")
        self.declare_parameter("wrist_image_topic", "/mujoco_server/cameras/wrist_camera/rgb/image_raw")
        self.declare_parameter("gripper_state_topic", "/panda_gripper_sim_node/joint_states")
        self.declare_parameter("instruction_topic", "/language_instruction")
        self.declare_parameter("goal_pose_topic", "/panda_arm_controller/joint_trajectory")
        self.declare_parameter("arm_joint_state_topic", "/joint_states")
        self.declare_parameter("gripper_topic_ns", "/panda_gripper_sim_node")

        # ========================== Miscellaneous ==========================
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        # ============================ Variables ============================
        self.eps = p("gripper_eps")
        self.model = RemoteModel(host=p("server_host"), port=p("server_port"), timeout=p("server_timeout"), jpeg_quality=p("jpeg_quality"))
        self.bridge = CvBridge()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.tcp_in_tip = None
        self.ik_group = p("ik_group")
        self.traj_duration = p("trajectory_duration")

        self.base_link = p("base_link")
        self.hand_link = p("hand_link")
        self.ik_tip_link = p("ik_tip_link")

        self.leash_pos = p("leash_pos")
        self.leash_rot = p("leash_rot")
        self.max_rel_pos = p("max_rel_pos")
        self.max_rel_orn = p("max_rel_orn")
        self.grasp_force = p("grasp_force")

        self.latest_third_person_image = None
        self.latest_wrist_image = None
        self.latest_joint_states = None
        self.latest_gripper_state = None
        self.last_target_pose = None
        self.last_gripper_cmd = None
        self.instruction = p("instruction") 

        # =========================== Subs & Pubs ===========================
        sensor_cb = ReentrantCallbackGroup()
        control_cb = MutuallyExclusiveCallbackGroup()   # control_step never overlaps itself
        depth = 15

        self.create_subscription(Image, p("third_person_image_topic"), self.on_third_person_image, depth, callback_group=sensor_cb)
        self.create_subscription(Image, p("wrist_image_topic"), self.on_wrist_image, depth, callback_group=sensor_cb)
        self.create_subscription(JointState, p("arm_joint_state_topic"), self.on_arm_joint_states, depth, callback_group=sensor_cb)
        self.create_subscription(JointState, p("gripper_state_topic"), self.on_gripper_state, depth, callback_group=sensor_cb)
        self.create_subscription(String, p("instruction_topic"), self.on_instruction, 1, callback_group=sensor_cb)
 
        self.goal_pub = self.create_publisher(JointTrajectory, p("goal_pose_topic"), 1)

        self.ik_client = self.create_client(GetPositionIK, "/compute_ik", callback_group=sensor_cb)
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
        if len(msg.position) == 0: 
            self.logger.info("Unable to fetch gripper position.")
            return
        self.latest_gripper_state = OPEN if msg.position[0] > 0.04 - self.eps else CLOSE


    def on_instruction(self, msg):
        if msg.data != self.instruction:
            self.logger.info(f"New instruction: '{msg.data}' received. Resetting history...")
            self.instruction = msg.data
            self.model.reset()
            self.last_target_pose = None
            self.last_gripper_cmd = None
            # self.reset_episode_state()
            self.logger.info("Episode state reset, policy is running.")


    # ======================== Solving IK =======================
    def get_tcp_in_tip(self):
        """
        Static offset of the TCP in the IK tip frame, cached after the first lookup.
        """
        if self.tcp_in_tip is None:
            try:
                tf = self.tf_buffer.lookup_transform(self.ik_tip_link, self.hand_link, rclpy.time.Time())
            except Exception as e:
                self.logger.warn(f"TCP offset lookup failed: {e}", throttle_duration_sec=1.0)
                return None
            t, q = tf.transform.translation, tf.transform.rotation
            self.tcp_in_tip = (np.array([t.x, t.y, t.z]), R.from_quat([q.x, q.y, q.z, q.w]))
        return self.tcp_in_tip


    def tcp_to_tip(self, target_pose):
        """
        TCP target pose (6D) -> IK tip target (pos, Rotation), both in base_link.
        """
        p_off, r_off = self.tcp_in_tip
        r_tip = R.from_euler("xyz", target_pose[3:6]) * r_off.inv()
        p_tip = target_pose[:3] - r_tip.apply(p_off)
        return p_tip, r_tip


    def solve_ik(self, pos, rot):
        """
        Returns 7 joint angles, or None if IK fails.
        """
        if not self.ik_client.service_is_ready(): 
            self.logger.warn("IK service not ready.", throttle_duration_sec=2.0)
            return None
        req = GetPositionIK.Request()
        ik = req.ik_request
        ik.group_name = self.ik_group
        ik.robot_state.joint_state = self.latest_joint_states   # seed = current joints (avoids elbow flips)
        ik.avoid_collisions = False
        ik.timeout.nanosec = 20_000_000                          # 20 ms

        ik.pose_stamped.header.frame_id = self.base_link
        ik.pose_stamped.pose.position.x, ik.pose_stamped.pose.position.y, ik.pose_stamped.pose.position.z = map(float, pos)
        qx, qy, qz, qw = rot.as_quat()
        o = ik.pose_stamped.pose.orientation
        o.x, o.y, o.z, o.w = float(qx), float(qy), float(qz), float(qw)

        res = self.ik_client.call(req)
        if res.error_code.val != 1:   # 1 = SUCCESS
            return None
        js = res.solution.joint_state
        idx = {n: i for i, n in enumerate(js.name)}
        return [js.position[idx[j]] for j in ARM_JOINTS]

    
    # ===================== Commanding Panda ====================
    def decode_action(self, action, base_pose):
        """
        Applies one predicted delta to a base pose.

        Args:
        - action: [dx, dy, dz, drx, dry, drz, gripper]
        - base_pose: 6D pose the delta is applied to.

        Returns:
        - target_pose: 6D target end-effector pose in base_link frame.
        - target_gripper_state: OPEN or CLOSE.
        """
        action = np.asarray(action, dtype=np.float64)
        delta = np.concatenate([action[:3] * self.max_rel_pos, action[3:6] * self.max_rel_orn])

        base_rot = R.from_euler("xyz", base_pose[3:6])
        target_pos = base_pose[:3] + base_rot.apply(delta[:3])
        target_rot = base_rot * R.from_euler("xyz", delta[3:6])
        target_pose = np.concatenate([target_pos, target_rot.as_euler("xyz")])

        target_gripper_state = OPEN if action[6] > 0.5 else CLOSE # NOTE: Seer outputs sigmoid value for gripper
        return target_pose, target_gripper_state


    def leash(self, target_pose, ee_pose):
        """
        Keep the target within leash_pos / leash_rot of the measured end-effector pose.
        """
        target_pose = target_pose.copy()

        # Position
        off = target_pose[:3] - ee_pose[:3]
        n = np.linalg.norm(off)
        if n > self.leash_pos:
            target_pose[:3] = ee_pose[:3] + off * (self.leash_pos / n)

        # Orientation
        ee_rot = R.from_euler("xyz", ee_pose[3:6])
        rv = (ee_rot.inv() * R.from_euler("xyz", target_pose[3:6])).as_rotvec()
        a = np.linalg.norm(rv)
        if a > self.leash_rot:
            target_pose[3:6] = (ee_rot * R.from_rotvec(rv * (self.leash_rot / a))).as_euler("xyz")

        return target_pose


    def control_step(self):
        ee_pose = self.get_ee_pose()
        if ee_pose is None:
            self.logger.error("Unable to fetch the end-effector pose. Aborting...")
            return
        
        if self.last_target_pose is None: # first step of an episode
            self.last_target_pose = ee_pose

        if None in (self.latest_third_person_image, self.latest_wrist_image, self.latest_gripper_state, self.latest_joint_states):
            self.logger.info("Unable to run control step yet.", throttle_duration_sec=2.0)
            return
        
        third_person_image = self.to_rgb(self.latest_third_person_image)
        wrist_image = self.to_rgb(self.latest_wrist_image)
        state7d = np.append(ee_pose, np.array(self.latest_gripper_state))
        try:
            delta_action = self.model.predict(third_person_image, wrist_image, state7d, self.instruction)
        except Exception as e:
            self.logger.error(f"Error occurred while running inference: {e}", throttle_duration_sec=2.0)
            return
        if self.last_target_pose is None: 
            self.logger.info("Instruction has been changed during prediction. Aborting...", throttle_duration_sec=2.0)
            return
        target_pose, target_gripper_state = self.decode_action(delta_action, self.last_target_pose)
        self.last_target_pose = self.leash(target_pose, ee_pose)
        self.send_command(self.last_target_pose, target_gripper_state)


    def send_command(self, target_pose, target_gripper_state):
        """
        Converts the TCP target to joint angles via IK and sends them to the trajectory controller.
        """
        if self.get_tcp_in_tip() is not None:
            q = self.solve_ik(*self.tcp_to_tip(target_pose))
            if q is None:
                self.logger.warn("IK failed; holding previous goal.", throttle_duration_sec=1.0)
            else:
                traj = JointTrajectory()
                traj.joint_names = ARM_JOINTS
                d = self.traj_duration
                traj.points = [JointTrajectoryPoint(
                    positions=[float(v) for v in q],
                    time_from_start=Duration(sec=int(d), nanosec=int((d % 1) * 1e9)),
                )]
                self.goal_pub.publish(traj)
        self.send_gripper_command(target_gripper_state)


    def send_gripper_command(self, cmd):
        """
        Send the gripper goal to the GripperAction client. This fires and returns immediately.
        """
        if self.last_gripper_cmd != cmd:
            if cmd == OPEN:
                goal = Move.Goal(width=0.08, speed=0.1)
                client = self.move_client
            else:
                goal = Grasp.Goal(width=0.0, speed=0.1, force=self.grasp_force)
                goal.epsilon.inner = 0.08 # accept any object width
                goal.epsilon.outer = 0.08
                client = self.grasp_client
        else:
            return
        if not client.server_is_ready():
            self.logger.error("Gripper action server not available.", throttle_duration_sec=2.0)
            return
        client.send_goal_async(goal) # use async so it doesn't block
        self.last_gripper_cmd = cmd
        self.logger.info(f"Gripper {'OPEN' if cmd == OPEN else 'CLOSE'}")


    # ======================== Utility Functions ========================
    def get_ee_pose(self):
        """
        Looks up the pose of the hand relative to the base.

        Returns:
          - The pose of the 6D end-effector: [x, y, z, rx, ry, rz]
        """
        try:
            tf = self.tf_buffer.lookup_transform(self.base_link, self.hand_link, rclpy.time.Time())
        except Exception as e:
            self.logger.warn(f"Transform lookup failed: {e}", throttle_duration_sec=1.0)
            return None
        
        t, q = tf.transform.translation, tf.transform.rotation
        pos = np.array([t.x, t.y, t.z], dtype=np.float64)
        rot = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")

        return np.concatenate([pos, rot], dtype=np.float64)


    def to_rgb(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    

    def destroy_node(self):
            try:
                self.model.close()
            finally:
                super().destroy_node()


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