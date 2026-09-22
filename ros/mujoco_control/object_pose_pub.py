# A node for publishing an object's position.

import rclpy

from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import PointStamped, QuaternionStamped, PoseStamped


class ObjectPosePublisher(Node):
	
	def __init__(self):
		super().__init__('object_pose_publisher')

		# Receivers for cube pose
		self.cube_pos = None
		self.cube_quat = None

		self.cube_pos_sub = self.create_subscription(
			PointStamped,
			'/cube_pos',
			self.cube_pos_callback,
			10
		)

		self.cub_quat_sub = self.create_subscription(
			QuaternionStamped,
			'/cube_quat',
			self.cube_quat_callback,
			10
		)

		# Publisher for the transformed cube pose
		self.publisher = self.create_publisher(
			PoseStamped,
			'/object_pose',
			10
		)
		
		self.timer = self.create_timer(
			0.1,
			self.publish_cube_pose
		)
		
		
	def publish_cube_pose(self):
		if self.cube_pos is None or self.cube_quat is None:
			return

		# Construct pose in world frame
		pose = PoseStamped()

		pose.header.frame_id = self.cube_pos.header.frame_id
		pose.header.stamp = self.cube_pos.header.stamp

		pose.pose.position = self.cube_pos.point
		pose.pose.orientation = self.cube_quat.quaternion

		self.publisher.publish(pose)


	def cube_pos_callback(self, msg):
		self.cube_pos = msg


	def cube_quat_callback(self, msg):
		self.cube_quat = msg

		
def main(args=None):
	rclpy.init(args=args)
	
	# init a publisher node
	node = ObjectPosePublisher()
	
	try: 
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()
	
		
if __name__=='__main__':
	main()
