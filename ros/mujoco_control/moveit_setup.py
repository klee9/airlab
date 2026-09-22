import os
import yaml
import subprocess

from moveit.planning import MoveItPy

def create_panda_moveit():
	franka_xacro = os.path.expanduser("~/multipanda_ws/src/franka_description/robots/sim/panda_arm_sim.urdf.xacro")
	semantic_xacro = os.path.expanduser("~/multipanda_ws/src/franka_moveit_config/srdf/panda_arm.srdf.xacro")
	config_dir = os.path.expanduser("~/multipanda_ws/src/franka_moveit_config/config")
	
	robot_description = subprocess.check_output(
		[
			"xacro",
			franka_xacro,
			"hand:=true",
		],
		text=True,
	)
	
	robot_description_semantic = subprocess.check_output(
		[
			"xacro",
			semantic_xacro,
			"hand:=true",
		],
		text=True,
	)
	
	with open(os.path.join(config_dir, "kinematics.yaml")) as f:
		kinematics = yaml.safe_load(f)
		
	with open(os.path.join(config_dir, "ompl_planning.yaml")) as f:
		ompl = yaml.safe_load(f)
		
	with open(os.path.join(config_dir, "panda_controllers.yaml")) as f:
		controllers = yaml.safe_load(f)
	
	config_dict = {
		"robot_description": robot_description,
		"robot_description_semantic": robot_description_semantic,
		"robot_description_kinematics": kinematics,
		"planning_pipelines": {
			"pipeline_names": ["ompl"],
		},
		"ompl": {
			**ompl,
			"planning_plugin": "ompl_interface/OMPLPlanner",
			'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization '
								'default_planner_request_adapters/ResolveConstraintFrames '
								'default_planner_request_adapters/FixWorkspaceBounds '
								'default_planner_request_adapters/FixStartStateBounds '
								'default_planner_request_adapters/FixStartStateCollision '
								'default_planner_request_adapters/FixStartStatePathConstraints',
			'start_state_max_bounds_error': 0.1,
		},
		'moveit_simple_controller_manager': controllers,
		'moveit_controller_manager': 'moveit_simple_controller_manager'
										'/MoveItSimpleControllerManager',
		'moveit_manage_controllers': True,
		'trajectory_execution.allowed_execution_duration_scaling': 5.0,
		'trajectory_execution.allowed_goal_duration_margin': 30.0,
		'trajectory_execution.allowed_start_tolerance': 0.05,
	}
	
	panda = MoveItPy(
		node_name="panda_moveit",
		config_dict=config_dict
	)
	
	panda_arm = panda.get_planning_component("panda_arm")
	
	return panda, panda_arm
