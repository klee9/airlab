import time
import rclpy

from geometry_msgs.msg import PoseStamped

from .moveit_setup import create_panda_moveit
from .object_pose_sub import ObjectPoseSubscriber
from .motions import (
	make_grasp_pose,
	move_to_pose,
	open_gripper
)

def get_current_ee_pose(panda):
	''' 
	Returns the current end-effector pose
	'''
	planning_scene_monitor = panda.get_planning_scene_monitor()
	with planning_scene_monitor.read_only() as scene:
		robot_state = scene.current_state
		return robot_state.get_pose("panda_hand_tcp")
		

def main():
	rclpy.init()
	node = rclpy.create_node("pregrasp")
	logger = node.get_logger()

	try:
		# 1. Prepare MoveItPy and planning component
		panda, panda_arm = create_panda_moveit()
		
		# 2. Fetch object pose
		object_sub = ObjectPoseSubscriber(node, topic_name="/object_pose")
		logger.info("topic: /object_pose is ready.")
		
		object_pose: PoseStamped = object_sub.wait_for_pose(timeout_sec=5.0)
		
		if not object_pose:
			logger.error("Unable to fetch object position")
			return
			
		logger.info(
            f"object pose = "
            f"({object_pose.pose.position.x:.3f}, "
            f"{object_pose.pose.position.y:.3f}, "
            f"{object_pose.pose.position.z:.3f}) "
            f"in frame {object_pose.header.frame_id}"
		)
		
		# 3. Move the EE to the object (pre-grasp state)
		approach_offset_z = 0.05  # 5 cm above the object
		
		#  3.1. Fetch target EE pose
		grasp_pose = make_grasp_pose(
			object_pose=object_pose,
			approach_offset_z=approach_offset_z,
		)
		logger.info(f"grasp pose: {grasp_pose}")

		#  3.2. Move the panda arm to grasp_pose
		go_to_grasp = move_to_pose(panda, panda_arm, grasp_pose, ee_link="panda_hand_tcp", group_name="panda_arm")
		
		if not go_to_grasp:
			logger.error("Failed")
			return
		
		logger.info("Pre-grasp completed.")
		time.sleep(0.5)
		
		
	except Exception as e:
		logger.error(f"Error: {e}")
		import traceback
		traceback.print_exc()
		
	finally:
		node.destroy_node()
		rclpy.shutdown()
		
if __name__=="__main__":
	main()
