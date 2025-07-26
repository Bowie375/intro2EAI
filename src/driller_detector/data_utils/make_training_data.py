import os
import sys
from PIL import Image
import time
import argparse
import numpy as np
import cv2
from pyapriltags import Detector
import trimesh.transformations as tra

import random
import torch

def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed: int
        Random seed between 0 and 2**32 - 1
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def make_data(idxs, display, output_dir="data/est_pose_new"):
    sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__),"../../../..")))
    f = open(f"logs/data_{idxs[0]}.txt", "w")
    #sys.stdout = f
    os.environ["DISPLAY"] = f":{display}"
    set_seed(display)
    
    from src.utils import to_pose, rand_rot_mat
    from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv
    
    # nearly all the functions of the simulation is implemeted in this demo
    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=1)
    parser.add_argument("--customize_scene", type=int, default=0)
    parser.add_argument("--reset_wait_steps", type=int, default=100)

    args = parser.parse_args()
    
    for idx in range(idxs[0], idxs[1]):
        print(f"making data {idx}/{idxs[1]}")
        #f.write(f"making data {idx}/{idxs[1]}")
        env_config = WrapperEnvConfig(
            humanoid_robot=args.robot,
            obj_name=args.obj,
            headless=args.headless,
            ctrl_dt=args.ctrl_dt,
            reset_wait_steps=args.reset_wait_steps,
        )

        env = WrapperEnv(env_config)
        if args.customize_scene:
            # Customize the table and object for testing with random environment
            table_pose = to_pose(trans=np.array([0.6, 0.35, 0.72]))
            table_size = np.array([0.68, 0.36, 0.02])
            obj_trans = np.array([0.5, 0.3, 0.82])
            obj_rot = np.eye(3)
            obj_pose = to_pose(obj_trans, obj_rot)
            env.set_table_obj_config(
                table_pose=table_pose,
                table_size=table_size,
                obj_pose=obj_pose
            )

        default_table_trans=np.array([0.6, 0.35, 0.72])
        default_table_size=np.array([0.68, 0.36, 0.02])
        default_obj_trans=np.array([0.5, 0.3, 0.82])

        #table_shift = np.array([0, -0.17, 0.1])
        #table_shift = np.array([0, -0.25, -0.15])

        #table_shift = np.array([np.random.uniform(-0.25, -0.25), np.random.uniform(-0.25, 0.16), np.random.uniform(-0.15, 0.1)])
        #table_shift = np.array([np.random.uniform(-0.33, -0.33), np.random.uniform(-0.25, 0.16), np.random.uniform(0.1, 0.1)])
        #table_shift = np.array([np.random.uniform(0.24, 0.24), np.random.uniform(-0.25, 0.16), np.random.uniform(0.1, 0.1)])
        #table_shift = np.array([np.random.uniform(-0.36, -0.36), np.random.uniform(-0.25, 0.16), np.random.uniform(-0.15, -0.15)])
        #table_shift = np.array([np.random.uniform(0.45, 0.45), np.random.uniform(-0.25, 0.16), np.random.uniform(-0.15, -0.15)])
        z = np.random.uniform(-0.15, 0.1)
        x = np.random.uniform(0.12*z-0.342, -0.84*z+0.324)
        y = np.random.uniform(0.32*z-0.202, 0.112-0.32*z)
        table_shift = np.array([x,y,z])
        table_trans = default_table_trans + table_shift
        table_size = default_table_size

        #obj_shift = np.array([np.random.uniform(-0.15, 0.15), np.random.uniform(0.15, 0.15)+table_shift[1], table_shift[2]])
        obj_shift = table_shift # np.array([np.random.uniform(-0.15, 0.15), np.random.uniform(-0.06, 0.16)+table_shift[1], table_shift[2]])
        obj_trans = default_obj_trans + obj_shift
        # obj_trans[1] = np.clip(obj_trans[1], 
        #                        a_max=min(-0.6*table_trans[2]+0.772, table_trans[1]+table_size[1]/2-0.04), 
        #                        a_min=table_trans[1]-table_size[1]/2+0.04)
        # obj_trans[1] = min(-0.6*table_trans[2]+0.81, table_trans[1]+table_size[1]/2-0.04)
        obj_rot = tra.euler_matrix(
            0., 0., np.random.uniform(0, 2 * np.pi), 
            axes='sxyz'
        )[:3, :3]  # Random rotation matrix

        table_pose = to_pose(trans=table_trans)
        obj_pose = to_pose(
            obj_trans, obj_rot
        )

        env.set_table_obj_config(
            table_pose=table_pose,
            table_size=table_size,
            obj_pose=obj_pose
        )

        env.launch()
        env.reset()

        head_init_qpos = np.array([-0.05, 0.35])  # [horizontal, vertical]
        humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos

        env.step_env(
            humanoid_head_qpos=head_init_qpos, # head joint qpos is for adjusting the camera pose
            humanoid_action=humanoid_init_qpos[:7],
            quad_command=[0,0,0]
        )

        obj_pose = env.get_driller_pose()
        obs_wrist = env.get_obs(camera_id=1) # wrist camera
        camera_pose = obs_wrist.camera_pose
        camera_intrinsics = env.sim.humanoid_robot_cfg.camera_cfg[1].intrinsics
        output_path = os.path.join(output_dir, f"{idx}")
        env.debug_save_obs(obs_wrist, output_path)
        #np.save(os.path.join(output_path, "camera_pose.npy"), camera_pose)
        np.save(os.path.join(output_path, "camera_intrinsics.npy"), camera_intrinsics)
        np.save(os.path.join(output_path, "obj_pose.npy"), obj_pose)

        env.close()

if __name__ == "__main__":
    import multiprocessing as mp

    processes = []
    tasks = {
        0: [2000, 3000, 100],
        1: [3000, 4000, 101],
        2: [4000, 5000, 102],
        3: [5000, 6000, 103],
        4: [6000, 7000, 104],
        5: [7000, 8000, 105],
    }
    for i in range(0,6):
        p = mp.Process(target=make_data, args=(tasks[i][:2], tasks[i][2]))
        processes.append(p)
        
    for p in processes:
        p.start()
        
    for p in processes:
        p.join()