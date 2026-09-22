import argparse
import json
import math
import os
import random
import time
import traceback

import rclpy
from rclpy.parameter import Parameter
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from mujoco_ros_msgs.srv import SetBodyState, GetBodyState

from .moveit_setup import create_panda_moveit
from .motions import (
    make_grasp_pose, make_lift_pose, make_place_pose,
    open_gripper, close_gripper, move_to_pose,
)
from .planning_scene_utils import (
    attach_object_to_gripper, add_collision_object, detach_object_from_gripper,
)

JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]


# =========================== Env reset ===========================
def get_initial_joint_positions(node, timeout_sec=10.0):
    """Reads one /joint_states message using the main node (no throw-away node needed)."""
    got = {}

    def cb(msg):
        if not got:
            got.update(zip(msg.name, msg.position))

    sub = node.create_subscription(JointState, "/joint_states", cb, 10)
    t0 = time.time()
    while not got and time.time() - t0 < timeout_sec:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    if not got:
        raise RuntimeError("no /joint_states received")
    return got


def wait_for_joints(node, logger, target_q, tol=0.02, timeout=30.0, settle=0.5):
    """
    Block until all arm joints are within `tol` rad of target_q for `settle` s of SIM time
    (or `timeout` s of sim time). Sim time is used because the simulator may run slower than
    wall-clock; a wall-clock cap of 4x timeout guards against a paused sim.
    """
    latest = {}

    def cb(msg):
        latest.update(zip(msg.name, msg.position))

    sub = node.create_subscription(JointState, "/joint_states", cb, 10)
    clk = node.get_clock()
    t0_sim, t0_wall, reached_at = clk.now(), time.time(), None
    sim_elapsed = 0.0
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.05)
            sim_elapsed = (clk.now() - t0_sim).nanoseconds * 1e-9
            if sim_elapsed > timeout or time.time() - t0_wall > 4 * timeout:
                break
            if all(n in latest for n in JOINT_NAMES):
                err = max(abs(latest[n] - target_q[n]) for n in JOINT_NAMES)
                if err < tol:
                    reached_at = reached_at if reached_at is not None else sim_elapsed
                    if sim_elapsed - reached_at >= settle:
                        logger.info(f"home pose reached (max joint err {err:.4f} rad, {sim_elapsed:.1f}s sim)")
                        return True
                else:
                    reached_at = None
        errs = {n: round(latest[n] - target_q[n], 3) for n in JOINT_NAMES if n in latest}
        logger.warn(f"home pose NOT reached after {sim_elapsed:.1f}s sim / {time.time()-t0_wall:.1f}s wall; "
                    f"joint errors (rad): {errs}; continuing")
        return False
    finally:
        node.destroy_subscription(sub)


def reset_panda_moveit(panda, panda_arm, logger, initial_q, planning_time=5.0, velocity_scaling=0.5):
    """Plan + execute a joint-space move to initial_q with MoveIt (same path as the other motions)."""
    from moveit.core.robot_state import RobotState
    from moveit.planning import PlanRequestParameters
    import numpy as np
    rs = RobotState(panda.get_robot_model())
    rs.set_to_default_values()
    # moveit_py exposes only the group-level setter; the panda_arm group is ordered panda_joint1..7
    rs.set_joint_group_positions("panda_arm", np.array([float(initial_q[n]) for n in JOINT_NAMES]))
    rs.update()
    panda_arm.set_start_state_to_current_state()
    panda_arm.set_goal_state(robot_state=rs)
    params = PlanRequestParameters(panda)
    params.planning_pipeline = "ompl"
    params.planner_id = "RRTConnectkConfigDefault"     # joint-space goal: fast and reliable
    params.planning_time = planning_time
    params.max_velocity_scaling_factor = velocity_scaling
    params.max_acceleration_scaling_factor = velocity_scaling
    try:
        result = panda_arm.plan(params)
    except Exception as e:
        logger.error(f"home planning raised: {e}")
        return False
    if not result:
        logger.error("home planning failed")
        return False
    panda.execute(group_name="panda_arm", robot_trajectory=result.trajectory)   # blocks until done
    return True


def reset_panda_topic(node, logger, traj_pub, initial_q, duration=4.0):
    while traj_pub.get_subscription_count() == 0:
        rclpy.spin_once(node, timeout_sec=0.1)
    msg = JointTrajectory()
    msg.joint_names = JOINT_NAMES
    pt = JointTrajectoryPoint()
    pt.positions = [float(initial_q[n]) for n in JOINT_NAMES]
    pt.time_from_start.sec = int(duration)
    pt.time_from_start.nanosec = int((duration % 1) * 1e9)
    msg.points.append(pt)
    traj_pub.publish(msg)
    logger.info(f"Panda reset command sent on topic ({duration:.1f}s trajectory)")
    return True


def reset_panda(node, logger, traj_pub, gripper_pub, initial_q, panda=None, panda_arm=None,
                mode="moveit", duration=4.0, timeout=30.0, tol=0.02):
    """
    Return the arm to initial_q and CONFIRM arrival via /joint_states.
    mode="moveit": plan/execute a joint goal through MoveIt (default, reliable).
    mode="topic":  single-point JointTrajectory on the controller topic (original behaviour).
    Falls back from moveit to topic if planning fails, and re-tries once if the arm stops short.
    """
    for attempt in range(2):
        if mode == "moveit" and panda is not None:
            ok = reset_panda_moveit(panda, panda_arm, logger, initial_q)
            if not ok:
                reset_panda_topic(node, logger, traj_pub, initial_q, duration)
        else:
            reset_panda_topic(node, logger, traj_pub, initial_q, duration)
        if wait_for_joints(node, logger, initial_q, tol=tol, timeout=timeout):
            break
        logger.warn(f"reset attempt {attempt+1} did not reach home; retrying")
    close_gripper(node, gripper_pub, max_effort=100.0, logger=logger)
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)


# =========================== MuJoCo world <-> panda_link0 ===========================
# MuJoCo's world frame and MoveIt's panda_link0 differ by a rigid transform (here: 180 deg about z,
# plus an optional translation). --link0_in_world gives panda_link0's pose in the world frame.
def _link0_in_world(args):
    x, y, z, yaw_deg = args.link0_in_world
    return x, y, z, math.radians(yaw_deg)


def link0_to_world(px, py, pz, yaw, args):
    """Point + yaw in panda_link0 -> MuJoCo world."""
    tx, ty, tz, tyaw = _link0_in_world(args)
    c, s_ = math.cos(tyaw), math.sin(tyaw)
    return (tx + c * px - s_ * py, ty + s_ * px + c * py, tz + pz, yaw + tyaw)


def world_to_link0(px, py, pz, args):
    """Point in MuJoCo world -> panda_link0."""
    tx, ty, tz, tyaw = _link0_in_world(args)
    c, s_ = math.cos(-tyaw), math.sin(-tyaw)
    dx, dy = px - tx, py - ty
    return (c * dx - s_ * dy, s_ * dx + c * dy, pz - tz)


def quat_world_to_link0(q, args):
    """Orientation quaternion (geometry_msgs) in world -> panda_link0 (rotate by -yaw about z)."""
    tyaw = _link0_in_world(args)[3]
    hz, hw = math.sin(-tyaw / 2), math.cos(-tyaw / 2)          # q_rot = (0, 0, hz, hw)
    # q_out = q_rot * q
    x = hw * q.x - hz * q.y
    y = hw * q.y + hz * q.x
    z = hw * q.z + hz * q.w
    w = hw * q.w - hz * q.z
    return x, y, z, w


def sample_cube_pose(args):
    """Uniform over the reachable table region; yaw in [0, 90 deg) since a cube is 4-fold symmetric."""
    x = random.uniform(args.cube_range[0], args.cube_range[1])
    y = random.uniform(args.cube_range[2], args.cube_range[3])
    yaw = random.uniform(0.0, math.pi / 2) if args.cube_yaw_deg < 0 else math.radians(args.cube_yaw_deg)
    return x, y, yaw


def reset_cube(node, logger, client, x, y, yaw, args):
    """Teleports the cube with mujoco_ros SetBodyState (pose in the MuJoCo world frame)."""
    if not client.wait_for_service(timeout_sec=3.0):
        logger.error(f"service '{args.set_body_service}' not available "
                     f"(check: ros2 service list -t | grep SetBodyState)")
        return False
    req = SetBodyState.Request()
    req.state.name = args.cube_body
    req.state.pose.header.frame_id = "world"
    wx, wy, wz, wyaw = link0_to_world(x, y, args.cube_z, yaw, args)
    req.state.pose.pose.position.x = wx
    req.state.pose.pose.position.y = wy
    req.state.pose.pose.position.z = wz
    req.state.pose.pose.orientation.z = math.sin(wyaw / 2)
    req.state.pose.pose.orientation.w = math.cos(wyaw / 2)
    req.set_pose = True
    req.set_twist = True          # twist left at zero -> cube stops moving
    fut = client.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=3.0)
    res = fut.result()
    if res is None or not res.success:
        logger.error(f"set_body_state failed: {getattr(res, 'status_message', 'timeout')}")
        return False
    for _ in range(5):            # let physics settle
        rclpy.spin_once(node, timeout_sec=0.1)
    return True


def cube_pose_stamped(x, y, yaw, args):
    """PoseStamped of the cube in panda_link0 from the commanded (x, y, yaw)."""
    ps = PoseStamped()
    ps.header.frame_id = "panda_link0"
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, args.cube_z
    ps.pose.orientation.z = math.sin(yaw / 2)
    ps.pose.orientation.w = math.cos(yaw / 2)
    return ps


def get_cube_pose(node, logger, get_client, x, y, yaw, args):
    """
    Ground-truth cube pose from /get_body_state (MuJoCo state, never stale).
    Falls back to the commanded pose if the service is unavailable.
    """
    commanded = cube_pose_stamped(x, y, yaw, args)
    if not get_client.wait_for_service(timeout_sec=1.0):
        logger.warn("get_body_state unavailable; using commanded cube pose")
        return commanded
    req = GetBodyState.Request()
    req.name = args.cube_body
    fut = get_client.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=2.0)
    res = fut.result()
    if res is None or not res.success:
        logger.warn(f"get_body_state failed ({getattr(res, 'status_message', 'timeout')}); using commanded pose")
        return commanded
    p = res.state.pose.pose
    ps = PoseStamped()
    ps.header.frame_id = "panda_link0"
    lx, ly, lz = world_to_link0(p.position.x, p.position.y, p.position.z, args)
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = lx, ly, lz
    qx, qy, qz, qw = quat_world_to_link0(p.orientation, args)
    ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = qx, qy, qz, qw
    dx, dy = ps.pose.position.x - x, ps.pose.position.y - y
    if math.hypot(dx, dy) > 0.02:
        logger.warn(f"cube is {math.hypot(dx, dy)*100:.1f} cm from where it was placed (settled/rolled?) - using measured pose")
    return ps


def reset_env(node, logger, traj_pub, gripper_pub, cube_client, initial_q, args, panda=None, panda_arm=None):
    """Arm to home (+ small random joint offset so episodes don't all start from one frame), then cube."""
    q = dict(initial_q)
    if args.start_jitter > 0:
        for n in JOINT_NAMES:
            q[n] = initial_q[n] + random.uniform(-args.start_jitter, args.start_jitter)
    reset_panda(node, logger, traj_pub, gripper_pub, q, panda=panda, panda_arm=panda_arm,
                mode=args.reset_mode, duration=args.reset_duration, timeout=args.reset_timeout, tol=args.reset_tol)
    x, y, yaw = sample_cube_pose(args)
    ok = reset_cube(node, logger, cube_client, x, y, yaw, args)
    return {"cube_x": x, "cube_y": y, "cube_yaw": yaw, "start_q": [q[n] for n in JOINT_NAMES], "cube_reset_ok": ok}


# =========================== Helpers ===========================
def get_current_ee_pose(panda, link="panda_hand_tcp"):
    with panda.get_planning_scene_monitor().read_only() as scene:
        return scene.current_state.get_pose(link)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--planner", default="rrtstar", choices=["rrtstar", "rrtconnect", "lin"])
    ap.add_argument("--velocity_scaling", type=float, default=1.0, help="1.0 = original speed")
    ap.add_argument("--place_range", type=float, nargs=4, default=[0.30, 0.60, -0.30, 0.30],
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"), help="place target region in panda_link0")
    ap.add_argument("--settle", type=float, default=1.0, help="s to wait after release before reset")
    ap.add_argument("--reset_mode", default="moveit", choices=["moveit", "topic"],
                    help="how to return home: MoveIt joint goal (default) or raw controller topic")
    ap.add_argument("--reset_duration", type=float, default=4.0, help="topic mode: s given to reach home")
    ap.add_argument("--reset_timeout", type=float, default=30.0, help="max SIM seconds to wait for the home pose")
    ap.add_argument("--reset_tol", type=float, default=0.02, help="rad; all joints within this = arrived")
    # ---- randomisation ----
    ap.add_argument("--cube_body", default="cube", help="MJCF body name of the cube")
    ap.add_argument("--set_body_service", default="/set_body_state",
                    help="find with: ros2 service list -t | grep SetBodyState")
    ap.add_argument("--get_body_service", default="/get_body_state")
    ap.add_argument("--cube_range", type=float, nargs=4, default=[0.30, 0.60, -0.30, 0.30],
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"), help="cube start region in panda_link0")
    ap.add_argument("--cube_z", type=float, default=0.02, help="cube centre height (half edge)")
    ap.add_argument("--cube_yaw_deg", type=float, default=-1.0,
                    help="fixed cube yaw in degrees; negative (default) = random in [0, 90)")
    ap.add_argument("--link0_in_world", type=float, nargs=4, default=[0.0, 0.0, 0.0, 180.0],
                    metavar=("X", "Y", "Z", "YAW_DEG"),
                    help="pose of panda_link0 in the MuJoCo world frame (default: 180 deg about z)")
    ap.add_argument("--min_place_dist", type=float, default=0.10, help="m between cube start and place target")
    ap.add_argument("--start_jitter", type=float, default=0.05, help="rad of random offset on the home joints")
    ap.add_argument("--meta_dir", default="", help="if set, write episode_meta_<n>.json here")
    return ap.parse_args()


# =========================== Main ===========================
def main():
    args = parse_args()
    rclpy.init()
    node = rclpy.create_node("pick_place")
    node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
    logger = node.get_logger()

    gripper_pub = node.create_publisher(JointState, "/gripper_command", 15)
    traj_pub = node.create_publisher(JointTrajectory, "/panda_arm_controller/joint_trajectory", 15)
    episode_pub = node.create_publisher(Bool, "/episode_active", 1)   # collector records only while True
    cube_client = node.create_client(SetBodyState, args.set_body_service)
    get_cube_client = node.create_client(GetBodyState, args.get_body_service)

    def set_active(flag: bool):
        episode_pub.publish(Bool(data=flag))

    approach_offset_z, grasp_offset_z, lift_offset_z = 0.2, 0.0, 0.15

    table_pose = PoseStamped()
    table_pose.header.frame_id = "panda_link0"
    table_pose.pose.position.z = -0.04
    table_pose.pose.orientation.w = 1.0

    def move(panda, panda_arm, pose, what):
        ok = move_to_pose(panda, panda_arm, pose, "panda_hand_tcp", "panda_arm",
                          planner=args.planner, velocity_scaling=args.velocity_scaling,
                          acceleration_scaling=args.velocity_scaling, logger=logger)
        if not ok:
            logger.error(f"Failed: {what}")
        return ok

    try:
        initial_q = get_initial_joint_positions(node)
        panda, panda_arm = create_panda_moveit()

        # start every run from the home configuration with the gripper closed (before recording)
        set_active(False)
        meta = reset_env(node, logger, traj_pub, gripper_pub, cube_client, initial_q, args, panda, panda_arm)

        for i in range(args.episodes):          # no early `return` any more
            xmin, xmax, ymin, ymax = args.place_range
            while True:   # place target at least min_place_dist from the cube
                place_x, place_y, place_z = random.uniform(xmin, xmax), random.uniform(ymin, ymax), 0.02
                if math.hypot(place_x - meta["cube_x"], place_y - meta["cube_y"]) >= args.min_place_dist:
                    break
            meta.update({"place_x": place_x, "place_y": place_y})
            logger.info(f"[{i+1}/{args.episodes}] place target: ({place_x:.3f}, {place_y:.3f}, {place_z:.3f})")

            # we teleported the cube ourselves -> use that pose, verified against MuJoCo's ground truth
            object_pose = get_cube_pose(node, logger, get_cube_client, meta["cube_x"], meta["cube_y"], meta["cube_yaw"], args)
            logger.info(f"object pose = ({object_pose.pose.position.x:.3f}, {object_pose.pose.position.y:.3f}, "
                        f"{object_pose.pose.position.z:.3f}) in {object_pose.header.frame_id}")

            for obj_id, spec in {"cube": (object_pose, [0.04, 0.04, 0.04]),
                                 "table": (table_pose, [1.6, 1.6, 0.04])}.items():
                add_collision_object(object_pose=spec[0], logger=logger, node=node, object_id=obj_id, dimensions=spec[1])

            set_active(True)   # ---- recording window starts ----
            ok = False
            try:
                # pre-grasp (closed gripper, above the cube)
                if not move(panda, panda_arm, make_grasp_pose(object_pose, approach_offset_z), "pre-grasp"):
                    continue
                open_gripper(node, gripper_pub, logger)
                # descend & grasp
                if not move(panda, panda_arm, make_grasp_pose(object_pose, grasp_offset_z), "grasp"):
                    continue
                close_gripper(node, gripper_pub, max_effort=100.0, logger=logger)
                attach_object_to_gripper(logger, node, "cube")
                # lift, keeping the grasp orientation (previously snapped to yaw=0 and twisted the cube)
                if not move(panda, panda_arm, make_lift_pose(object_pose, lift_offset_z, current_ee_pose=get_current_ee_pose(panda)), "lift"):
                    continue
                # place
                if not move(panda, panda_arm, make_place_pose(place_x, place_y, place_z, current_ee_pose=get_current_ee_pose(panda)), "place"):
                    continue
                detach_object_from_gripper(make_place_pose(place_x, place_y, place_z, current_ee_pose=get_current_ee_pose(panda)), logger, node, "cube")
                open_gripper(node, gripper_pub, logger)
                ok = True
                time.sleep(args.settle)
            finally:
                set_active(False)   # ---- recording window ends (before any reset motion) ----

            meta["success"] = ok
            logger.info(f"Episode {i+1}: {'success' if ok else 'FAILED'}  meta={meta}")
            if args.meta_dir:
                os.makedirs(args.meta_dir, exist_ok=True)
                with open(os.path.join(args.meta_dir, f"episode_meta_{int(time.time())}.json"), "w") as fh:
                    json.dump(meta, fh, indent=1)
            logger.info("Resetting the environment...")
            meta = reset_env(node, logger, traj_pub, gripper_pub, cube_client, initial_q, args, panda, panda_arm)

    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
    finally:
        set_active(False)
        node.destroy_node()
        rclpy.shutdown()


def run():
    main()


if __name__ == "__main__":
    main()