import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit.planning import PlanRequestParameters
from geometry_msgs.msg import Pose, PoseStamped
from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState
from tf_transformations import euler_from_quaternion, quaternion_from_euler

# ================================ Gripper actions ================================

_gripper_clients = {}

def _get_gripper_client(node: Node, action_name: str) -> ActionClient:
    """One ActionClient per (node, action) instead of a new one per call (they were never destroyed)."""
    key = (id(node), action_name)
    if key not in _gripper_clients:
        _gripper_clients[key] = ActionClient(node, GripperCommand, action_name)
    return _gripper_clients[key]


def move_gripper(
    node: Node,
    gripper_pub,
    width: float,
    max_effort: float = 40.0,
    gripper_action_name: str = "/panda_gripper_sim_node/gripper_action",
    logger=None,
    wait_result: bool = True,
) -> bool:
    """Sends a GripperCommand goal to the MuJoCo Franka gripper and publishes the command state."""
    logger = logger or node.get_logger()
    logger.info(f"Commanding gripper width: {width:.3f} m (force: {max_effort} N)")
    client = _get_gripper_client(node, gripper_action_name)
    if not client.wait_for_server(timeout_sec=3.0):
        logger.error(f"Gripper action server '{gripper_action_name}' not available!")
        return False

    # Publish the command for the dataset collector / VLA node (open: width > 0, close: 0)
    msg = JointState()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.name = ["panda_finger_joint1"]
    msg.position = [width]
    gripper_pub.publish(msg)

    goal = GripperCommand.Goal()
    goal.command.position = width
    goal.command.max_effort = max_effort
    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)
    handle = future.result()
    if handle is None or not handle.accepted:
        logger.error("Gripper command was rejected by the server.")
        return False
    if wait_result:
        res_future = handle.get_result_async()
        rclpy.spin_until_future_complete(node, res_future, timeout_sec=5.0)
    return True


def open_gripper(node: Node, gripper_pub, logger=None) -> bool:
    return move_gripper(node, gripper_pub, width=0.04, max_effort=20.0, logger=logger)


def close_gripper(node: Node, gripper_pub, max_effort: float = 100.0, logger=None) -> bool:
    return move_gripper(node, gripper_pub, width=0.0, max_effort=max_effort, logger=logger)


# ================================ Pose helpers ================================
def _top_down_quat(yaw: float):
    """Gripper pointing straight down (roll = pi), rotated by yaw about z."""
    return quaternion_from_euler(3.14159265, 0.0, yaw)


def _set_orientation(pose: PoseStamped, current_ee_pose, yaw: float = 0.0):
    if current_ee_pose is not None:
        o = current_ee_pose.pose.orientation if isinstance(current_ee_pose, PoseStamped) else current_ee_pose.orientation
        pose.pose.orientation.x, pose.pose.orientation.y = o.x, o.y
        pose.pose.orientation.z, pose.pose.orientation.w = o.z, o.w
    else:
        qx, qy, qz, qw = _top_down_quat(yaw)
        pose.pose.orientation.x, pose.pose.orientation.y = qx, qy
        pose.pose.orientation.z, pose.pose.orientation.w = qz, qw


def make_grasp_pose(object_pose: PoseStamped, approach_offset_z: float = 0.12, **_):
    """Top-down pose above the object, yaw aligned with the object."""
    pose = PoseStamped()
    pose.header.frame_id = object_pose.header.frame_id
    pose.pose.position.x = object_pose.pose.position.x
    pose.pose.position.y = object_pose.pose.position.y
    pose.pose.position.z = object_pose.pose.position.z + approach_offset_z
    q = object_pose.pose.orientation
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    _set_orientation(pose, None, yaw)
    return pose


def make_lift_pose(object_pose: PoseStamped, lift_offset_z: float = 0.15, current_ee_pose=None, **_):
    """Straight up from the object; keeps the current (grasp) orientation so the cube is not twisted."""
    pose = PoseStamped()
    pose.header.frame_id = object_pose.header.frame_id
    pose.pose.position.x = object_pose.pose.position.x
    pose.pose.position.y = object_pose.pose.position.y
    pose.pose.position.z = object_pose.pose.position.z + lift_offset_z
    _set_orientation(pose, current_ee_pose)
    return pose


def make_place_pose(place_x: float, place_y: float, place_z: float, current_ee_pose=None, **_):
    pose = PoseStamped()
    pose.header.frame_id = "panda_link0"
    pose.pose.position.x = place_x
    pose.pose.position.y = place_y
    pose.pose.position.z = place_z
    _set_orientation(pose, current_ee_pose)
    return pose


# ================================ Motion Planning ================================
def move_to_pose(
    panda,
    panda_arm,
    target_pose: PoseStamped,
    ee_link: str = "panda_hand_tcp",
    group_name: str = "panda_arm",
    planner: str = "rrtstar",            # original, reliable setup: RRT* with 10 s (uses the full time)
    planning_time: float = 10.0,
    velocity_scaling: float = 1.0,       # 1.0 = unchanged; e.g. 0.3 for slower, smoother demos
    acceleration_scaling: float = 1.0,
    logger=None,
) -> bool:
    """Same plan/execute flow as the original; planner and scaling are only overridable."""
    panda_arm.set_goal_state(pose_stamped_msg=target_pose, pose_link=ee_link)

    plan_params = PlanRequestParameters(panda)
    if planner == "lin":
        plan_params.planning_pipeline = "pilz_industrial_motion_planner"
        plan_params.planner_id = "LIN"
    elif planner == "rrtconnect":
        plan_params.planning_pipeline = "ompl"
        plan_params.planner_id = "RRTConnectkConfigDefault"
    else:
        plan_params.planning_pipeline = "ompl"
        plan_params.planner_id = "RRTstarkConfigDefault"
    plan_params.planning_time = planning_time
    if velocity_scaling != 1.0:
        plan_params.max_velocity_scaling_factor = velocity_scaling
    if acceleration_scaling != 1.0:
        plan_params.max_acceleration_scaling_factor = acceleration_scaling

    try:
        plan_result = panda_arm.plan(plan_params)
        if not plan_result:
            if logger:
                logger.error(f"planning failed ({plan_params.planner_id})")
            return False
    except Exception as e:
        if logger:
            logger.error(f"planning raised: {e}")
        return False

    panda.execute(group_name=group_name, robot_trajectory=plan_result.trajectory)
    return True