#!/usr/bin/env python3
"""
Bridge between vla_node's outputs and the MuJoCo Panda controllers.

    /vla/target_pose     (PoseStamped, panda_link0)  --IK-->  /panda_arm_controller/joint_trajectory (JointTrajectory)
    /vla/target_gripper  (Float64, +1 open / -1 close) -->     /panda_gripper_sim_node/gripper_action (GripperCommand action)
                                                              + /gripper_command (JointState, so vla_node sees gripper state,
                                                                exactly as motions.move_gripper publishes it)

Why not MoveItPy plan()/execute() like pick_place.py?  Planning with OMPL takes seconds; a
closed-loop policy needs a new target ~15x per second.  IK on the current robot state takes
milliseconds and the joint-trajectory controller happily tracks a stream of single-point
trajectories (that is what reset_panda already uses).
"""
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand
from moveit.core.robot_state import RobotState
from scipy.spatial.transform import Rotation as R

from .moveit_setup import create_panda_moveit

JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]


class VLABridge(Node):
    def __init__(self):
        super().__init__("vla_bridge")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.declare_parameter("group", "panda_arm")
        self.declare_parameter("ee_link", "panda_hand_tcp")
        self.declare_parameter("traj_topic", "/panda_arm_controller/joint_trajectory")
        self.declare_parameter("gripper_action", "/panda_gripper_sim_node/gripper_action")
        self.declare_parameter("time_from_start", 0.15)   # s the controller gets to reach each target
        self.declare_parameter("max_step_pos", 0.02)      # m  clamp per command (safety against big jumps)
        self.declare_parameter("max_step_rot", 0.1)       # rad
        self.declare_parameter("max_joint_jump", 0.5)     # rad; reject IK solutions that flip configuration
        self.declare_parameter("ik_timeout", 0.05)
        self.declare_parameter("open_width", 0.04)
        self.declare_parameter("close_effort", 100.0)
        self.declare_parameter("open_effort", 20.0)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.group, self.ee_link = p("group"), p("ee_link")
        self.tfs = p("time_from_start")
        self.max_step_pos, self.max_step_rot = p("max_step_pos"), p("max_step_rot")
        self.max_joint_jump, self.ik_timeout = p("max_joint_jump"), p("ik_timeout")
        self.open_width = p("open_width")
        self.close_effort, self.open_effort = p("close_effort"), p("open_effort")

        # MoveIt (for IK only) — same helper pick_place.py uses
        self.get_logger().info("Starting MoveItPy for IK ...")
        self.panda, _ = create_panda_moveit()
        self.robot_model = self.panda.get_robot_model()
        self.get_logger().info("MoveItPy ready.")

        self.q = None                     # current arm joints (dict name -> pos)
        self.gripper_cmd = None           # last commanded +1/-1
        self.lock = threading.Lock()

        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        self.create_subscription(PoseStamped, "/vla/target_pose", self.on_target_pose, 1)
        self.create_subscription(Float64, "/vla/target_gripper", self.on_target_gripper, 1)
        self.traj_pub = self.create_publisher(JointTrajectory, p("traj_topic"), 15)
        self.gripper_state_pub = self.create_publisher(JointState, "/gripper_command", 15)
        self.gripper_client = ActionClient(self, GripperCommand, p("gripper_action"))
        self.get_logger().info("VLA bridge started; waiting for /vla/target_pose ...")

    # ------------------------------------------------------------------ #
    def on_joints(self, msg: JointState):
        d = dict(zip(msg.name, msg.position))
        if all(n in d for n in JOINT_NAMES):
            self.q = d

    def current_ee_pose(self, rs: RobotState):
        T = np.asarray(rs.get_global_link_transform(self.ee_link))
        return T[:3, 3], R.from_matrix(T[:3, :3])

    # ------------------------------------------------------------------ #
    def on_target_pose(self, msg: PoseStamped):
        if self.q is None:
            self.get_logger().warn("no /joint_states yet", throttle_duration_sec=2.0)
            return
        if not self.lock.acquire(blocking=False):
            return  # previous IK still running; drop this target
        try:
            q_now = np.array([self.q[n] for n in JOINT_NAMES])
            rs = RobotState(self.robot_model)
            rs.set_joint_group_positions(self.group, q_now)
            rs.update()

            # --- clamp the requested motion relative to the current EE pose ---
            cur_p, cur_r = self.current_ee_pose(rs)
            tgt_p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            o = msg.pose.orientation
            tgt_r = R.from_quat([o.x, o.y, o.z, o.w])
            dp = tgt_p - cur_p
            n = np.linalg.norm(dp)
            if n > self.max_step_pos:
                dp *= self.max_step_pos / n
            drot = cur_r.inv() * tgt_r
            ang = np.linalg.norm(drot.as_rotvec())
            if ang > self.max_step_rot:
                drot = R.from_rotvec(drot.as_rotvec() * self.max_step_rot / ang)
            cmd_p = cur_p + dp
            cmd_r = cur_r * drot

            # --- IK ---
            pose = msg.pose.__class__()
            pose.position.x, pose.position.y, pose.position.z = map(float, cmd_p)
            qx, qy, qz, qw = cmd_r.as_quat()
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, (qx, qy, qz, qw))
            if not rs.set_from_ik(self.group, pose, self.ee_link, timeout=self.ik_timeout):
                self.get_logger().warn("IK failed for target", throttle_duration_sec=1.0)
                return
            q_new = np.array(rs.get_joint_group_positions(self.group))
            if np.max(np.abs(q_new - q_now)) > self.max_joint_jump:
                self.get_logger().warn("IK solution jumps configuration; skipping", throttle_duration_sec=1.0)
                return

            # --- stream a single-point trajectory ---
            traj = JointTrajectory()
            traj.joint_names = JOINT_NAMES
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in q_new]
            pt.time_from_start.sec = int(self.tfs)
            pt.time_from_start.nanosec = int((self.tfs % 1) * 1e9)
            traj.points.append(pt)
            self.traj_pub.publish(traj)
        finally:
            self.lock.release()

    # ------------------------------------------------------------------ #
    def on_target_gripper(self, msg: Float64):
        cmd = 1.0 if msg.data > 0 else -1.0
        if cmd == self.gripper_cmd:
            return
        self.gripper_cmd = cmd
        width = self.open_width if cmd > 0 else 0.0
        effort = self.open_effort if cmd > 0 else self.close_effort
        self.get_logger().info(f"gripper -> {'open' if cmd > 0 else 'close'}")

        # same JointState that motions.move_gripper publishes (vla_node reads this as gripper state)
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = ["panda_finger_joint1"]
        js.position = [width]
        self.gripper_state_pub.publish(js)

        goal = GripperCommand.Goal()
        goal.command.position = width
        goal.command.max_effort = effort
        if not self.gripper_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("gripper action server not available")
            return
        self.gripper_client.send_goal_async(goal)  # fire and forget; don't block the pose stream


def main():
    rclpy.init()
    node = VLABridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()