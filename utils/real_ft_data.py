import numpy as np  
import pandas as pd
import json  
import os  
from pdb import set_trace 
from tqdm import tqdm
import glob  
from PIL import Image as PILImage
import h5py
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import random
import imageio
import cv2
import shutil


def exists_or_mkdir(
    path
):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass 

def _6d_to_pose(
    pose6d,
    degrees=False
):
    pose = np.eye(4)
    pose[:3, 3] = pose6d[:3]
    pose[:3, :3] = R.from_euler("xyz", pose6d[3:6], degrees=degrees).as_matrix()
    return pose

def pose_to_6d(
    pose, 
    degrees=False
):
    pose6d = np.zeros(6)
    pose6d[:3] = pose[:3, 3]
    pose6d[3:6] = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)
    return pose6d

def compute_delta_action(
    data_list,
):
    delta_cur_2_last_action_list = []
    for step_id, step_data in enumerate(data_list):
        delta_cur_2_last_action = np.zeros(7)
        delta_cur_2_last_action[-1] = step_data["action_gripper_pose"][-1]
        if step_id == 0: # the first timestep
            last2world = _6d_to_pose(step_data["gripper_pose"])
        else:
            last2world = _6d_to_pose(data_list[step_id-1]["action_gripper_pose"][:6], degrees=False)  
        cur2world = _6d_to_pose(step_data["action_gripper_pose"][:6], degrees=False)
        cur2last = np.linalg.inv(last2world) @ cur2world
        delta_cur_2_last_action[:6] = pose_to_6d(cur2last)
        delta_cur_2_last_action_list.append(delta_cur_2_last_action)

def filter_real_data(
    exp_id, 
    root_path, 
    save_data_path, 
    save_gif_path
):
    root_path = os.path.join(root_path, exp_id)
    save_data_path = os.path.join(save_data_path, exp_id)
    save_gif_path = os.path.join(save_gif_path, exp_id)
    length = len(glob.glob(os.path.join(root_path, exp_id, "*")))
    exists_or_mkdir(save_gif_path)
    exists_or_mkdir(save_data_path)
    for j in range(0, length): # Here we only have 100 demos, change it accordingly.
        episode_idx = str(j).zfill(6)
        npz_path_list = glob.glob(os.path.join(root_path, episode_idx, "steps", "*", "other.npz"))
        npz_path_list.sort()
        step_id_list = []
        img_list = []
        for idx, npz_path in enumerate(npz_path_list):
            this_npz = np.load(npz_path)
            if idx == 0:
                prev_gripper_action = this_npz["action_gripper_pose"][-1]
            curr_gripper_action = this_npz["action_gripper_pose"][-1]
            step_id = npz_path.split('/')[-2]
            action = this_npz["delta_cur_2_last_action"]
            if (abs(action[0]) >= 5e-4) or (abs(action[1]) >= 5e-4) or (abs(action[2]) >= 5e-4) or (curr_gripper_action != prev_gripper_action):
                step_id_list.append(step_id)
            prev_gripper_action = curr_gripper_action
        save_last_step_id = step_id_list[-1]
        last_step_id = step_id
        add_step_id_list = [str(k).zfill(4) for k in range(int(save_last_step_id)+1, int(last_step_id)+1)]
        step_id_list += add_step_id_list
        for new_step_id, old_step_id in tqdm(enumerate(step_id_list)):
            new_step_id = str(new_step_id).zfill(4)
            new_step_path = os.path.join(save_data_path, episode_idx, "steps", new_step_id)
            old_step_path = os.path.join(root_path, episode_idx, "steps", old_step_id)
            shutil.copytree(old_step_path, new_step_path)
            img_list.append(PILImage.open(os.path.join(new_step_path, f"image_primary.jpg")))
        imageio.mimsave(os.path.join(save_gif_path, f"{episode_idx}.mp4"), img_list, fps=15)

def make_aug_short_real_dataset_info(
    root_path, 
    root_info_path,
    dataset_name,
    select_ratio=1.0,
    sequence_length=7, 
    action_pred_steps=3, 
    replicate_steps=10
):
    save_json_path = os.path.join(root_info_path, f"{dataset_name}.json")
    data_list = []
    window_size = sequence_length + action_pred_steps
    exp_path_list = glob.glob(os.path.join(root_path, "*"))
    exp_path_list.sort()
    for exp_path in tqdm(exp_path_list):
        length = len(glob.glob(os.path.join(exp_path, "*")))
        for j in tqdm(range(length)):
            exp_id = exp_path.split('/')[-1]
            demo_id = str(j).zfill(6)
            npz_path_list = glob.glob(os.path.join(exp_path, demo_id, "steps", "*", "other.npz"))
            npz_path_list.sort()
            this_demo_list = [f"{exp_id}/{demo_id}"]
            for npz_path in npz_path_list:
                this_npz = np.load(npz_path)
                step_id = npz_path.split('/')[-2]
                int_step_id = int(step_id)
                if int_step_id >= window_size:
                    this_demo_list.append([int_step_id - window_size, int_step_id])
                curr_gripper_action = this_npz["delta_cur_2_last_action"][-1]
                if step_id == "0000":
                    prev_gripper_action = curr_gripper_action
                if curr_gripper_action != prev_gripper_action:
                    print(
                        "curr_gripper_action :", curr_gripper_action, 
                        "prev_gripper_action :", prev_gripper_action,
                        "step_id :", step_id
                        )
                    for _ in range(replicate_steps):
                        for k in range(action_pred_steps):
                            if int_step_id + k < len(npz_path_list):
                                this_demo_list.append([int_step_id - window_size + k, int_step_id + k])
                prev_gripper_action = curr_gripper_action
            demo_length = len(this_demo_list)
            this_demo_list.insert(1, demo_length-1+window_size)
            data_list.append(this_demo_list)
    if select_ratio < 1.0:
        interval_len = 10
        start_id = 0
        select_num = int(interval_len * select_ratio)
        end_id = interval_len
        new_data_list = []
        while end_id <= len(data_list):
            selected_data_list = random.sample(data_list[start_id:end_id], select_num)
            new_data_list += selected_data_list
            start_id += interval_len
            end_id += interval_len
        data_list = new_data_list
    json_string = json.dumps(data_list, indent=1)
    with open(save_json_path, 'w') as json_file:
        json_file.write(json_string)


def oxe_dataset_info():
    dataset_names = [
        # {
        # "dataset_name": f"bridge_dataset",
        # "wrist_image": "Normal",
        # "s_ratio": 1.0,
        # }, # zheng

        # {
        # "dataset_name": f"cmu_stretch",
        # "wrist_image": "Normal",
        # "s_ratio": 1.0,
        # }, # zheng

        {
        "dataset_name": f"fractal20220817_data",
        "wrist_image": "Normal",
        "s_ratio": 0.54087122203,
        }, # zheng ###

        # {
        # "dataset_name": f"dlr_edan_shared_control_converted_externally_to_rlds",
        # "wrist_image": "Normal",
        # "s_ratio": 1.0,
        # }, # zheng

        # {
        # "dataset_name": f"kuka",
        # "wrist_image": "Normal",
        # "s_ratio": 0.8341046294,
        # }, # zheng ###

        # {
        # "dataset_name": f"roboturk",
        # "wrist_image": "Normal",
        # "s_ratio": 1.0,
        # }, # zheng

        # {
        # "dataset_name": f"ucsd_kitchen_dataset_converted_externally_to_rlds",
        # "wrist_image": "Normal",
        # "s_ratio": 1.0,
        # }, # zheng

        # {
        # "dataset_name" : f"berkeley_autolab_ur5", 
        # "wrist_image": "Flip vertically & horizontally", 
        # "s_ratio": 1.0,
        # }, # fan, 
        
        # {
        # "dataset_name" : f"berkeley_fanuc_manipulation", 
        # "wrist_image": "Flip vertically & horizontally",
        # "s_ratio": 1.0,
        # }, # fan

        # {
        # "dataset_name" : f"jaco_play", 
        # "wrist_image": "Flip vertically & horizontally",
        # "s_ratio": 1.0,
        # }, # fan
        
        # {
        # "dataset_name" : f"iamlab_cmu_pickup_insert_converted_externally_to_rlds", 
        # "wrist_image": "Normal",
        # "s_ratio": 1.0,
        # }, # zheng
        
        # {
        # "dataset_name" : f"viola", 
        # "wrist_image": "Flip vertically & horizontally",
        # "s_ratio": 2.0,
        # }, # fan
        
        # {
        # "dataset_name" : f"stanford_hydra_dataset_converted_externally_to_rlds", 
        # "wrist_image": "Flip vertically & horizontally",
        # "s_ratio": 2.0,
        # }, # fan
        
        # {
        # "dataset_name" : f"austin_buds_dataset_converted_externally_to_rlds", 
        # "wrist_image": "Flip vertically & horizontally",
        # "s_ratio": 1.0,
        # }, # fan
        
        # {
        # "dataset_name" : f"utaustin_mutex", 
        # "wrist_image": "Normal",
        # "s_ratio": 1.0,
        # }, # zheng
        
        # {
        # "dataset_name" : f"taco_play", 
        # "wrist_image": "Flip vertically & horizontally",
        # "s_ratio": 2.0,
        # }, # fan
        
        # {
        # "dataset_name" : f"austin_sailor_dataset_converted_externally_to_rlds", 
        # "wrist_image": "Flip vertically & horizontally",
        # "s_ratio": 1.0,
        # }, # fan
        
        # {
        # "dataset_name" : f"austin_sirius_dataset_converted_externally_to_rlds", 
        # "wrist_image": "Flip vertically & horizontally",
        # "s_ratio": 1.0,
        # }, # fan
        
        # {
        # "dataset_name" : f"furniture_bench_dataset_converted_externally_to_rlds", 
        # "wrist_image": "Normal",
        # "s_ratio": 0.1,
        # }, # zheng        
    ]

    # total_data_list = []

    for info in tqdm(dataset_names):
        dataset_name = info["dataset_name"]
        wrist_image_info = info["wrist_image"]
        s_ratio = info["s_ratio"]
        root_path = f"/xxx/preprocess/oxe/{dataset_name}"
        save_json_path = f"/xxx/data_info/{dataset_name}.json"
        root_path_list = glob.glob(os.path.join(root_path, "*", "*"))
        root_path_list.sort()
        data_list = []
        data_list.append(info)
        accumulated_num_steps = 0
        for this_path in tqdm(root_path_list):
            exp_id = this_path.split('/')[-2]
            demo_id = this_path.split('/')[-1]
            num_step = len(glob.glob(os.path.join(this_path, "steps", "*")))
            if s_ratio >= 1.0:
                for _ in range(int(s_ratio)):
                    accumulated_num_steps += num_step
                    data_list.append([exp_id+'/'+demo_id, num_step])
            else:
                this_p = np.random.random()
                if this_p < s_ratio:
                    accumulated_num_steps += num_step
                    data_list.append([exp_id+'/'+demo_id, num_step])
        
        data_list[0]["accumulated_num_steps"] = accumulated_num_steps
        json_string = json.dumps(data_list, indent=1)
        with open(save_json_path, 'w') as json_file:
            json_file.write(json_string)

