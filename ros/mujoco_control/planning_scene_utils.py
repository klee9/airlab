import rclpy

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive


def _extract_pose(pose_input) -> tuple[Pose, str]:
    """Helper to unpack Pose/PoseStamped and return (Pose, frame_id)."""
    if isinstance(pose_input, PoseStamped):
        return pose_input.pose, pose_input.header.frame_id
    return pose_input, "world"


def _apply_scene_diff(node, scene_diff: PlanningScene, logger=None) -> bool:
    """Synchronously applies a PlanningScene diff via ROS 2 service using the provided node."""
    log = logger or node.get_logger()

    client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
    if not client.wait_for_service(timeout_sec=5.0):
        log.error("Service '/apply_planning_scene' is not available!")
        return False

    req = ApplyPlanningScene.Request()
    req.scene = scene_diff

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    response = future.result()

    if response is None or not response.success:
        log.error("Failed to apply planning scene update.")
        return False

    return True


def add_collision_object(
    object_pose,
    logger=None,
    node=None,
    object_id: str = "target_object",
    dimensions: list = None,
    frame_id: str = "world",
):
    """Adds a collision object to the MoveIt planning scene."""
    log = logger or (node.get_logger() if node else None)
    if dimensions is None:
        dimensions = [0.04, 0.04, 0.04]

    pose, extracted_frame = _extract_pose(object_pose)
    ref_frame = extracted_frame if isinstance(object_pose, PoseStamped) else frame_id

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = dimensions

    collision_object = CollisionObject()
    collision_object.header.frame_id = ref_frame
    collision_object.id = object_id
    collision_object.primitives.append(primitive)
    collision_object.primitive_poses.append(pose)
    collision_object.operation = CollisionObject.ADD

    scene_diff = PlanningScene()
    scene_diff.is_diff = True
    scene_diff.world.collision_objects.append(collision_object)

    if _apply_scene_diff(node, scene_diff, logger=log):
        if log:
            log.info(f"Added collision object '{object_id}' to planning scene.")


def attach_object_to_gripper(
    logger=None,
    node=None,
    object_id: str = "target_cube",
    ee_link: str = "panda_hand_tcp",
):
    """Attaches an object to the gripper link in MoveIt's planning scene."""
    log = logger or (node.get_logger() if node else None)

    attached_obj = AttachedCollisionObject()
    attached_obj.link_name = ee_link
    attached_obj.object.id = object_id
    attached_obj.touch_links = [
        "panda_leftfinger",
        "panda_rightfinger",
        "panda_hand",
        "panda_hand_tcp",
    ]
    attached_obj.object.operation = CollisionObject.ADD

    scene_diff = PlanningScene()
    scene_diff.is_diff = True
    scene_diff.robot_state.is_diff = True
    scene_diff.robot_state.attached_collision_objects.append(attached_obj)

    if _apply_scene_diff(node, scene_diff, logger=log):
        if log:
            log.info(f"Attached object '{object_id}' to link '{ee_link}'.")


def detach_object_from_gripper(
    place_pose,
    logger=None,
    node=None,
    object_id: str = "target_cube",
    ee_link: str = "panda_hand_tcp",
    frame_id: str = "world",
):
    """Detaches an object from the gripper and updates its placement pose in the world scene."""
    log = logger or (node.get_logger() if node else None)

    pose, extracted_frame = _extract_pose(place_pose)
    ref_frame = extracted_frame if isinstance(place_pose, PoseStamped) else frame_id

    # Detach from end-effector
    detach_obj = AttachedCollisionObject()
    detach_obj.link_name = ee_link
    detach_obj.object.id = object_id
    detach_obj.object.operation = CollisionObject.REMOVE

    # Re-position in the world
    world_obj = CollisionObject()
    world_obj.header.frame_id = ref_frame
    world_obj.id = object_id
    world_obj.primitive_poses.append(pose)
    world_obj.operation = CollisionObject.MOVE

    scene_diff = PlanningScene()
    scene_diff.is_diff = True
    scene_diff.robot_state.is_diff = True
    scene_diff.robot_state.attached_collision_objects.append(detach_obj)
    scene_diff.world.collision_objects.append(world_obj)

    if _apply_scene_diff(node, scene_diff, logger=log):
        if log:
            log.info(f"Detached object '{object_id}' and updated world pose at '{ref_frame}'.")