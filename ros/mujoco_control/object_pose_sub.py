# A node for subscribing to /object_pose topic

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class ObjectPoseSubscriber:

	def __init__(self, node: Node, topic_name: str = "/object_pose"):
		self.node = node
		self.pose = None
		
		self.subscription = self.node.create_subscription(
			PoseStamped,
			topic_name,
			self.pose_callback,
			10
		)
		
		
	def pose_callback(self, msg: PoseStamped):
		self.pose = msg
		
		
	def wait_for_pose(self, timeout_sec: float = 5.0):
		start_t = time.time()
		
		while rclpy.ok():
		
			rclpy.spin_once(self.node, timeout_sec=0.1)
			
			if self.pose is not None: 
				return self.pose
			
			if time.time() - start_t >= timeout_sec:
				return None
				
		return None
