"""
Converts custom Seer dataset into LeRobot format, and uploads to HuggingFace.

The custom dataset is in the format of:

0000/
└── 000000/
    └── steps/
        ├── 0000/
        │   ├── image_primary.jpg
        │   ├── image_wrist.jpg
        │   └── other.npz
        ├── 0001/
        │   ├── image_primary.jpg
        │   ├── image_wrist.jpg
        │   └── other.npz
        └── ...

The goal is to reshape each frame to roughly:

{
    "observation.images.primary": image_primary,
    "observation.images.wrist": image_wrist,
    "observation.state": joints,
    "action": delta_action,
}
"""

import os
import cv2
import logging
import shutil
import numpy as np

from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi_client import image_tools
from huggingface_hub import login

load_dotenv()
logger = logging.getLogger(__name__)



FPS = 15.0
REPO_NAME = "klee9/pi05_no_ep2"

SKIP_EPISODES = {2} # Episode to exclude from the train set.
EXPECTED_SKIP_LEN = {2: 104} # For sanity check

HF_TOKEN = os.getenv("HF_TOKEN")

# ============================= Utility functions =============================
def convert_gripper_unit(state: float) -> np.ndarray:
    """ Returns 0 if the gripper is open, 1 if closed """
    return np.asarray((1 - state)/2, dtype=np.float64)


def resize_image(image):
    # Use a fixed 224x224 resizing to keep aspect ratio intact
    # NOTE: This may vary across different setups!
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))


# ============================= Dataset functions =============================
def load_episode(ep_dir: Path) -> dict:
    """ 
    Loads a single episode from the custom Seer dataset and returns its data 
    
    Seer contains:
      - image_primary: an image from a 3rd person camera
      - image_wrist: an image from a wrist-mounted camera
      - gripper_pose: [x, y, z, euler_x, euler_y, euler_z]
      - gripper_open_state: 1 if open, -1 if closed
      - joints: [q0, q1, q2, q3, q4, q5, q6]
      - action_gripper_pose = [target_x, target_y, target_z, target_euler_x, target_euler_y, target_euler_z]

    Returns:
      - A dictionary of steps, base image path, wrist image path, joint positions, joint velocity, gripper position, gripper action position, instruction
    """
    steps = sorted(d for d in (ep_dir / "steps").iterdir() if d.is_dir())
    if not steps:
        logger.error(f"No steps found in episode directory: {ep_dir}")
        return {}
    
    logger.info(f"Loading episode from {ep_dir}, {len(steps)} steps found.")
    
    # Load other.npz from each step
    base_img = [str(d / "image_primary.jpg") for d in steps]
    wrist_img = [str(d / "image_wrist.jpg") for d in steps]
    data = [np.load(d / "other.npz") for d in steps]

    joints = np.stack([d["joints"] for d in data]) # (T, 7) (T = number of steps)
    gripper_pose = np.stack([d["gripper_pose"] for d in data]) # (T, 7)
    action_gripper_pose = np.stack([d["action_gripper_pose"] for d in data]) # (T, 7)
    gripper_open_state = np.stack([d["gripper_open_state"] for d in data]) # (T, 1)
    language_instruction = bytes(data[0]["language_instruction"].astype(np.uint8)).decode("utf-8") # string

    # Convert joint positions to joint velocities
    # NOTE: the last frame must command zero velocity
    # NOTE: the current method of computing velocities may be wrong!
    joint_velocity = np.zeros_like(joints)
    joint_velocity[:-1] = np.diff(joints, axis=0) * FPS
    joint_velocity[-1] = 0.0 # last frame must command zero velocity

    # Extract gripper position and gripper action
    # NOTE: DROID expects gripper open state to be in [0, 1] range, where 0 is open and 1 is closed
    gripper_position = convert_gripper_unit(gripper_open_state.reshape(-1))[:, None]  # (T, 1)
    gripper_action_position = convert_gripper_unit(action_gripper_pose[:, 6])[:, None]  # (T, 1)

    return {
        "steps": steps,
        "base_image": base_img,
        "wrist_image": wrist_img,
        "joint_positions": joints.astype(np.float32),
        "joint_velocity": joint_velocity.astype(np.float32),
        "gripper_position": gripper_position.astype(np.float32),
        "gripper_action_position": gripper_action_position.astype(np.float32),
        "language_instruction": language_instruction,
    }


def build_lerobot_dataset(data_root: Path):
    """ Builds a LeRobot dataset from the custom Seer dataset """
    output_dir = HF_LEROBOT_HOME / REPO_NAME
    if os.path.exists(output_dir):
        logger.warning(f"Output directory {output_dir} already exists. It will be overwritten. Continue? (y/n)")
        response = input()
        if response != "y":
            logger.info("Aborting dataset building.")
            return
        shutil.rmtree(output_dir)

    # Create an empty LeRobot dataset (copied from convert_droid_data_to_lerobot.py)
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=int(FPS),
        features={
             "exterior_image_1_left": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),  # We will use joint *velocity* actions here (7D) + gripper position (1D)
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Write to the dataset
    ep_idx = 0   # CHANGED: counts episodes exactly as klee9/pi05 numbered them
    for ep_dir in tqdm(sorted(d for d in data_root.iterdir() if d.is_dir()), desc="Processing episodes"):   # CHANGED: dirs only
        data = load_episode(ep_dir=ep_dir)
        if not data: 
            continue

        # CHANGED: skip listed episodes (after the empty check, so numbering matches the original)
        if ep_idx in SKIP_EPISODES:
            T = len(data["steps"])
            exp = EXPECTED_SKIP_LEN.get(ep_idx)
            if exp is not None and T != exp:
                raise RuntimeError(f"episode {ep_idx} ({ep_dir.name}) has T={T}, expected {exp}. "
                                   f"Wrong episode? Check the ordering before skipping.")
            logger.info(f"SKIP episode {ep_idx} ({ep_dir.name}, T={T})")
            ep_idx += 1
            continue

        for i, step in enumerate(data["steps"]):
            base_img = cv2.imread(data["base_image"][i])
            wrist_img = cv2.imread(data["wrist_image"][i])

            if base_img is None:
                raise FileNotFoundError(data["base_image"][i])
            if wrist_img is None:
                raise FileNotFoundError(data["wrist_image"][i])

            base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
            wrist_img = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)

            dataset.add_frame({
                "exterior_image_1_left": resize_image(base_img),
                "exterior_image_2_left": resize_image(base_img),  # Placeholder for second camera
                "wrist_image_left": resize_image(wrist_img),
                "joint_position": data["joint_positions"][i],
                "gripper_position": data["gripper_position"][i],
                # Important: we use joint velocity actions here since pi05-droid was pre-trained on joint velocity actions
                "actions": np.concatenate([data["joint_velocity"][i], data["gripper_action_position"][i]]),
                "task": data["language_instruction"]
            })

        dataset.save_episode()
        logger.info(f"KEEP episode {ep_idx} ({ep_dir.name}, T={len(data['steps'])})")   # CHANGED
        ep_idx += 1   # CHANGED

    logger.info(f"LeRobot dataset saved to: {output_dir}")

    dataset.push_to_hub(
        tags=["libero", "panda", "rlds"],
        private=False,
        push_videos=True,
        license="apache-2.0",
    )

def main():
    login(token=HF_TOKEN)
    data_root = Path(os.path.expanduser("~/mujoco_data/panda_pick_place_fixed/0000"))
    build_lerobot_dataset(data_root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()