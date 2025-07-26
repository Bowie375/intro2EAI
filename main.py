import argparse
from typing import Optional, Tuple, List

import torch
import numpy as np

import cv2
import trimesh
import PIL.Image as Image

from pyapriltags import Detector

from src.type import Grasp
from src.utils import rand_rot_mat, to_pose
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv
from src.sim.wrapper_env import get_grasps
# from src.real.wrapper_env import WrapperEnvConfig, WrapperEnv, get_grasps
from src.test.load_test import load_test_data
from src.constants import DEPTH_IMG_SCALE

from src.config import Config
from src.driller_detector.model.est_coord import EstCoordNet
from src.driller_detector.segment import InteractiveSAM
# from model.est_coord import EstCoordNet

from segment_anything import sam_model_registry, SamPredictor

def detect_driller_mask(image):
    """
    Detect the driller mask in the image.
    return a binary mask of the driller, shape (H, W)
    """
    interactive_sam = InteractiveSAM(SAM_CKPT)
    seg = interactive_sam.run(image)
    
    return seg.astype(np.float32)

def get_pc(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Convert depth image into point cloud using intrinsics

    All points with depth=0 are filtered out

    Parameters
    ----------
    depth: np.ndarray
        Depth image, shape (H, W)
    intrinsics: np.ndarray
        Intrinsics matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        Point cloud with shape (N, 3)
    """
    # Get image dimensions
    height, width = depth.shape
    # Create meshgrid for pixel coordinates
    v, u = np.meshgrid(range(height), range(width), indexing="ij")
    # Flatten the arrays
    u = u.flatten()
    v = v.flatten()
    depth_flat = depth.flatten()
    # Filter out invalid depth values
    valid = depth_flat > 0
    u = u[valid]
    v = v[valid]
    depth_flat = depth_flat[valid]
    # Create homogeneous pixel coordinates
    pixels = np.stack([u, v, np.ones_like(u)], axis=0)
    # Convert pixel coordinates to camera coordinates
    rays = np.linalg.inv(intrinsics) @ pixels
    # Scale rays by depth
    points = rays * depth_flat
    return points.T
    
def rgbd_to_pointcloud(depth, camera_matrix, seg):
    """
    Convert RGB-D image to point cloud.
    return a point cloud in camera frame, shape (N, 3)
    """
    full_pc = get_pc(depth, camera_matrix) * np.array([-1, -1, 1])
    if seg is not None:
        valid_mask = (seg.flatten() > 0.5) & (depth.flatten() > 0)
        pc = full_pc[valid_mask]
    else:
        valid_mask = depth.flatten() > 0
        pc = full_pc[valid_mask]

    # downsample and filter the point cloud.
    def FPS(pc, num_points):
        """
        Perform Farthest Point Sampling to downsample the point cloud.
        """
        if pc.shape[0] <= num_points:
            return pc
        indices = np.zeros(num_points, dtype=int)
        indices[0] = np.random.randint(pc.shape[0])
        distances = np.linalg.norm(pc - pc[indices[0]], axis=1)
        for i in range(1, num_points):
            indices[i] = np.argmax(distances)
            distances = np.minimum(distances, np.linalg.norm(pc - pc[indices[i]], axis=1))
        return pc[indices]
    
    pc_sampled = FPS(pc, 5000)

    pc_sampled[:,:2] *= np.array([-1., -1.])
    return pc_sampled

def rotation_matrix_to_axis_angle(R):
    trace = np.trace(R)
    angle = np.arccos((trace - 1) / 2)
    return angle * 180 / np.pi

def detect_driller_pose(img, depth, seg, camera_matrix, camera_pose, model, *args, **kwargs):
    """
    Detects the pose of driller, you can include your policy in args
    return the pose of driller in world frame, shape (4, 4)
    """
    # implement the detection logic here
    #

    if seg is None:
        seg = detect_driller_mask(img)

    pc = rgbd_to_pointcloud(depth, camera_matrix, seg)
    
    if DEBUG:
        trimesh.PointCloud(vertices=pc).export("tmp/driller_pc.ply")
    
    pc = torch.tensor(pc, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
    
    with torch.no_grad():
        est_t, est_R = model.est(pc)
        est_t = est_t.squeeze(0).numpy()  # (3,)
        est_R = est_R.squeeze(0).numpy()  # (3, 3)
        
    pose = np.eye(4)
    pose[:3, :3] = est_R
    pose[:3, 3] = est_t
    
    
    if camera_pose.shape == (4, 4):
        pose = camera_pose @ pose

    elif len(camera_pose) == 3:
        world_pose = np.eye(4)
        world_pose[:3, 3] = camera_pose
        pose = world_pose @ pose
    
    else:
        raise ValueError(
            f"expected camera_pose to be (4, 4) or (3,), but got shape {camera_pose.shape}")

    return pose


def detect_marker_pose(
    detector: Detector,
    img: np.ndarray,
    camera_params: tuple,
    camera_pose: np.ndarray,
    tag_size: float = 0.12
) -> Optional[Tuple[np.ndarray, np.ndarray]]:

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    tags = detector.detect(
        gray, estimate_tag_pose=True,
        camera_params=camera_params, tag_size=tag_size
    )

    tag = tags[0] if len(tags) > 0 else None
    trans_marker_world = None
    rot_marker_world = None

    if tag is not None:
        # Extract translation and rotation from the detected tag
        trans_marker_camera = tag.pose_t.flatten()
        rot_marker_camera = tag.pose_R

        marker_pose_camera = np.eye(4)
        marker_pose_camera[:3, :3] = rot_marker_camera
        marker_pose_camera[:3, 3] = trans_marker_camera

        marker_pose_world = camera_pose @ marker_pose_camera
        trans_marker_world = marker_pose_world[:3, 3]
        rot_marker_world = marker_pose_world[:3, :3]

    return trans_marker_world, rot_marker_world


def forward_quad_policy(pose, target_pose, *args, **kwargs):
    """ guide the quadruped to position where you drop the driller """
    # implement
    v_max = kwargs.get('v_max', 0.6)
    omega_max = kwargs.get('omega_max', 1.2)
    pos_stop_thresh = kwargs.get('pos_stop_thresh', 0.05)
    angle_stop_thresh = kwargs.get('angle_stop_thresh', 0.05)
    Kp_pos = kwargs.get('Kp_pos', 2.0)
    Kp_yaw = kwargs.get('Kp_yaw', 3.0)

    # convert to robot frame
    target_robot = np.linalg.inv(pose) @ target_pose

    # P-control
    trans_error = target_robot[:2, 3]
    distance = np.linalg.norm(trans_error)

    yaw_error = np.arctan2(-target_robot[0, 1], target_robot[1, 1])

    # early stop for position
    if distance < pos_stop_thresh:
        vx, vy = 0.0, 0.0
    else:
        # Proportional velocity
        direction = trans_error / distance
        speed = min(Kp_pos * distance, v_max)
        vx, vy = direction * speed

    # early stop for yaw
    if abs(yaw_error) < angle_stop_thresh:
        vz = 0.0
    else:
        vz = np.clip(Kp_yaw * yaw_error, -omega_max, omega_max)

    return np.array([vy, vx, -vz])


def backward_quad_policy(pose, target_pose, *args, **kwargs):
    """ guide the quadruped back to its initial position """
    # implement
    action = np.array([0, 0, 0])
    return action

def plan_approach(
    env: WrapperEnv,
    begin_qpos: np.ndarray,
    orig_eef_pose: np.ndarray,
    targ_eef_pose: np.ndarray,
    approach_config: dict[str, int | float],
) -> list[np.ndarray] | None:
    """Try to plan a grasp trajectory for the given grasp.
    The trajectory is a list of joint positions. Return None if the trajectory is not valid."""
    # Get Config
    approach_steps: int = approach_config["approach_steps"]
    delta_thresh: float = approach_config["delta_thresh"]

    orig_eef_trans = orig_eef_pose[:3, 3]
    orig_eef_rot = orig_eef_pose[:3, :3]

    targ_eef_trans = targ_eef_pose[:3, 3]
    targ_eef_rot = targ_eef_pose[:3, :3]

    # Pregrasp
    succ, approach_qpos = env.humanoid_robot_model.ik_cam(
        targ_eef_trans,
        orig_eef_rot,
        init_qpos=begin_qpos,
        delta_thresh=delta_thresh * 5
    )
    if not succ:
        print("Failed to find approach IK solution.")
        return None
    
    traj_approach = plan_move_qpos(env, begin_qpos, approach_qpos, approach_steps)
    
    return traj_approach, approach_qpos

def plan_grasp(
    env: WrapperEnv,
    begin_qpos: np.ndarray,
    orig_eef_pose: np.ndarray,
    grasp: Grasp,
    grasp_config: dict[str, int | float],
) -> list[np.ndarray] | None:
    """Try to plan a grasp trajectory for the given grasp.
    The trajectory is a list of joint positions. Return None if the trajectory is not valid."""
    # Get Config
    pregrasp_steps: int = grasp_config["pregrasp_steps"]
    reach_steps: int = grasp_config["reach_steps"]
    lift_steps: int = grasp_config["lift_steps"]
    postgrasp_steps: int = grasp_config["postgrasp_steps"]
    delta_thresh: float = grasp_config["delta_thresh"]

    orig_eef_trans = orig_eef_pose[:3, 3]
    orig_eef_rot = orig_eef_pose[:3, :3]

    lift_hight = 0.2
    pregrasp_trans = grasp.trans + np.array([0.0, 0.0, lift_hight])
    ## Pregrasp
    # succ, pregrasp_qpos = env.humanoid_robot_model.ik(
    #     pregrasp_trans,
    #     #grasp.rot,
    #     orig_eef_rot,
    #     init_qpos=begin_qpos,
    #     delta_thresh=delta_thresh * 5
    # )
    # if not succ:
    #     print("Failed to find pregrasp IK solution.")
    #     return None
    
    # traj_pregrasp = plan_move_qpos(env, begin_qpos, pregrasp_qpos, pregrasp_steps)
    
    # Reach
    succ, reach_qpos = env.humanoid_robot_model.ik(
        grasp.trans,
        grasp.rot,
        init_qpos=begin_qpos,
        #init_qpos=pregrasp_qpos,
        delta_thresh=delta_thresh * 2
    )
    if not succ:
        print("Failed to find reach IK solution.")
        return None

    #traj_reach = plan_move_qpos(env, pregrasp_qpos, reach_qpos, reach_steps)
    traj_reach = plan_move_qpos(env, begin_qpos, reach_qpos, reach_steps)

    # Lift
    delta_trans = np.array([0.0, 0.0, 0.2])
    succ, lift_qpos = env.humanoid_robot_model.ik(
        grasp.trans + delta_trans,
        orig_eef_rot,
        #grasp.rot,
        init_qpos=reach_qpos,
        delta_thresh=delta_thresh * 2
    )
    if not succ:
        print("Failed to find lift IK solution.")
        return None

    traj_lift = plan_move_qpos(env, reach_qpos, lift_qpos, lift_steps)

    # Move Back
    # succ, postgrasp_qpos = env.humanoid_robot_model.ik(
    #     orig_eef_trans,
    #     grasp.rot,
    #     init_qpos=lift_qpos
    # )
    # if not succ:
    #     return None

    # traj_postgrasp = plan_move_qpos(env, lift_qpos, postgrasp_qpos, postgrasp_steps)

    return [traj_reach, traj_lift]


def plan_move(
    env: WrapperEnv,
    begin_qpos: np.ndarray,
    end_trans: np.ndarray,
    end_rot: np.ndarray,
    steps: int = 50,
) -> np.ndarray | None:
    """Plan a trajectory moving the driller from table to dropping position"""
    succ, end_qpos = env.humanoid_robot_model.ik(
        end_trans, end_rot,
        init_qpos=begin_qpos
    )
    if not succ:    
        for i in range(100):
            rot = rand_rot_mat()
            succ, end_qpos = env.humanoid_robot_model.ik(
                end_trans, rot,
                init_qpos=begin_qpos
            )
            if succ:
                print("success rot: ", rot)
                break
        else:
            return None
    
    traj = plan_move_qpos(env, begin_qpos, end_qpos, steps)
    stable_plan = traj[-1,None].repeat(40, axis=0)
    
    return np.concatenate([traj, stable_plan], axis=0)


def open_gripper(env: WrapperEnv, steps=10):
    for _ in range(steps):
        env.step_env(gripper_open=1)


def close_gripper(env: WrapperEnv, steps=10):
    for _ in range(steps):
        env.step_env(gripper_open=0)


def plan_move_qpos(env, begin_qpos, end_qpos, steps=50) -> np.ndarray:
    delta_qpos = (end_qpos - begin_qpos) / steps
    cur_qpos = begin_qpos.copy()
    traj = []

    for _ in range(steps):
        cur_qpos += delta_qpos
        traj.append(cur_qpos.copy())

    return np.array(traj)


def execute_plan(env: WrapperEnv, plan):
    """Execute the plan in the environment."""
    for step in range(len(plan)):
        env.step_env(
            humanoid_action=plan[step],
        )

DEBUG = False
TESTING = True
DISABLE_GRASP = False
DISABLE_MOVE = False
DETECTOR_CKPT = "checkpoint/est_coord/checkpoint_80000.pth"

SAM_CKPT = None
## We don't provide the the checkpoint of SAM model, you should download it from
## https://huggingface.co/ybelkada/segment-anything/blob/main/checkpoints/sam_vit_h_4b8939.pth
## and put it in "checkpoint/sam/sam_vit_h_4b8939.pth"
# SAM_CKPT = "checkpoint/sam/sam_vit_h_4b8939.pth"

def main():
    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=0)
    parser.add_argument("--reset_wait_steps", type=int, default=100)
    parser.add_argument("--test_id", type=int, default=0)

    args = parser.parse_args()

    detector = Detector(
        families="tagStandard52h13",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    env_config = WrapperEnvConfig(
        humanoid_robot=args.robot,
        obj_name=args.obj,
        headless=args.headless,
        ctrl_dt=args.ctrl_dt,
        reset_wait_steps=args.reset_wait_steps,
    )

    env = WrapperEnv(env_config)
    if TESTING:
        data_dict = load_test_data(args.test_id)
        env.set_table_obj_config(
            table_pose=data_dict['table_pose'],
            table_size=data_dict['table_size'],
            obj_pose=data_dict['obj_pose']
        )
        env.set_quad_reset_pos(data_dict['quad_reset_pos'])

    env.launch()
    env.reset(humanoid_qpos=env.sim.humanoid_robot_cfg.joint_init_qpos)
    humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos[:7]
    Metric = {
        'obj_pose': False,
        'drop_precision': False,
        'quad_return': False,
    }

    # you can adjust the head init qpos to find the driller
    head_init_qpos = np.array([0.0, 0.0])

    env.step_env(humanoid_head_qpos=head_init_qpos)

    # you can customize observing qpos to get wrist obs
    observing_qpos = humanoid_init_qpos + np.array([0.01, 0, 0, 0, 0, 0, 0])
    init_plan = plan_move_qpos(env, humanoid_init_qpos, observing_qpos, steps=20)
    execute_plan(env, init_plan)

    # load models
    driller_detector = EstCoordNet(Config())
    driller_detector.load_state_dict(
        torch.load(DETECTOR_CKPT, map_location="cpu", weights_only=True)['model']
        # torch.load("checkpoint/coord.pth", map_location="cpu")['model']
    )
    driller_detector = driller_detector.eval().to('cpu')


    # --------------------------------------step 1: move quadruped to dropping position--------------------------------------

    def move_quad(target_list, forward_steps=500, phase="phase1"):
        move_head = True
        head_qpos = head_init_qpos.copy()
        # number of steps per camera shot, increase this to reduce the frequency of camera shots and speed up the simulation
        steps_per_camera_shot = 10

        # judge if the quadruped is close to the target pose
        def is_close(pose1, target_list, threshold):
            pose2 = target_list[0]
            target_robot = np.linalg.inv(pose1) @ pose2
            trans_error = np.linalg.norm(target_robot[:2, 3])
            yaw_error = np.arctan2(-target_robot[0, 1], target_robot[1, 1])
            return trans_error < threshold and abs(yaw_error) < threshold

        for step in range(forward_steps):
            if step % steps_per_camera_shot == 0:
                obs_head = env.get_obs(camera_id=0)  # head camera
                # env.debug_save_obs(obs_head, f'data/{phase}/{step}/obs_head') # obs has rgb, depth, and camera pose
                trans_marker_world, rot_marker_world = detect_marker_pose(
                    detector, obs_head.rgb, head_camera_params,
                    obs_head.camera_pose, tag_size=0.12)
                pose_marker_world = to_pose(trans_marker_world, rot_marker_world)
                #if trans_marker_world is not None:
                #    # the container's pose is given by follows:
                #    trans_container_world = rot_marker_world @ np.array(
                #        [0, -0.31, 0.02]) + trans_marker_world
                #    rot_container_world = rot_marker_world
                #    pose_container_world = to_pose(
                #        trans_container_world, rot_container_world)

            quad_command = forward_quad_policy(
                pose_marker_world, target_list[0])
            move_head = step % steps_per_camera_shot == 0
            if move_head:
                pose_container_head = np.linalg.inv(
                    obs_head.camera_pose) @ pose_marker_world
                direct_head = np.ones((4, 1), dtype=np.float32)
                direct_head[:3, 0] = pose_container_head[:3, -1]
                direct = (world_init_head @ obs_head.camera_pose @
                          direct_head).reshape(-1)
                q1 = -np.arctan2(direct[0], direct[2])
                q2 = np.arctan2(direct[1], direct[2])
                head_qpos = np.array([q1, q2]).clip(
                    [-1.57, -0.36], [1.57, 0.36])
                env.step_env(
                    humanoid_head_qpos=head_qpos,
                    quad_command=quad_command
                )
            else:
                env.step_env(quad_command=quad_command)
            if is_close(pose_marker_world, target_list, threshold=0.05):
                target_list = target_list[1:]
                if len(target_list) == 0:
                    break

    if not DISABLE_MOVE:
        ## get init container pose and head pose for phase 5
        obs_head = env.get_obs(camera_id=0)  # head camera
        world_init_head = np.linalg.inv(
            env.get_obs(camera_id=0).camera_pose.copy())
        head_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[0].intrinsics
        head_camera_params = (
            head_camera_matrix[0, 0], head_camera_matrix[1, 1], 
            head_camera_matrix[0, 2], head_camera_matrix[1, 2])
        trans_marker_world, rot_marker_world = detect_marker_pose(
            detector, obs_head.rgb, head_camera_params,
            obs_head.camera_pose, tag_size=0.12)
        init_marker_pose = to_pose(trans_marker_world, rot_marker_world)
        #trans_container_world = rot_marker_world @ np.array(
        #    [0, -0.31, 0.02]) + trans_marker_world
        #rot_container_world = rot_marker_world
        #init_container_pose = to_pose(
        #    trans_container_world, rot_container_world)
        
        ## move quadruped to the target position
        target_list = []
        target_1 = np.array([[0.0, 1.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, -1., 0.0],
                             [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        target_1[:3, -1] = np.array([0.9, init_marker_pose[1,3], 0.278])
        target_list.append(target_1)
        target_2 = target_1.copy()
        target_2[1, 3] = 0.0
        target_list.append(target_2)
        
        target_container_trans = target_2[:3,:3] @ np.array(
            [0, -0.31, 0.02]) + target_2[:3, 3]
        target_container_rot = target_2[:3, :3].copy()
        target_container_pose = to_pose(
            target_container_trans, target_container_rot)
        
        move_quad(target_list)

    # --------------------------------------step 2: detect driller pose------------------------------------------------------
    if not DISABLE_GRASP:
        # init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos.copy()
        # orig_cam_trans, orig_cam_rot = env.humanoid_robot_model.fk_camera(init_qpos, 1)
        
        obs_wrist = env.get_obs(camera_id=1)  # wrist camera
        rgb, depth, camera_pose, seg = obs_wrist.rgb, obs_wrist.depth, obs_wrist.camera_pose, obs_wrist.seg
        #rgb, depth, camera_pose = obs_wrist.rgb, obs_wrist.depth, obs_wrist.camera_pose
        wrist_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[1].intrinsics
        
        """

        driller_pose = detect_driller_pose(
            rgb, depth, seg[:,:,0], wrist_camera_matrix, camera_pose, driller_detector)
 
        if DEBUG:
            gt_pose = env.get_driller_pose()       
            R_pred = driller_pose[:3, :3]
            R_gt = gt_pose[:3, :3]

            angle_diff = rotation_matrix_to_axis_angle(R_pred @ R_gt.T)
            angle_pred = rotation_matrix_to_axis_angle(R_pred)
            angle_gt = rotation_matrix_to_axis_angle(R_gt)

            print("------First pose estimation-------")
            print(f"Predicted angle: {angle_pred:.2f} degrees, Ground truth angle: {angle_gt:.2f} degrees")
            print(f"Detected driller pose with angle difference: {angle_diff:.2f} degrees")
            print(f"Detected driller pose:\n {driller_pose}")
            print(f"gt driller pose: \n {gt_pose}")
            
            Image.fromarray(rgb).save("tmp/driller_rgb.png")
            Image.fromarray(seg).save("tmp/driller_seg.png")
        
        """
        
        orig_cam_trans, orig_cam_rot = camera_pose[:3, 3], camera_pose[:3, :3]
        orig_cam_pose = camera_pose.copy()
        
        ## move the camera closer the the object (using heuristic of the environment setting) 
        ## for better detection of the driller
        approach_delta = np.array([0.0, 0.1]) # driller_pose[:2, 3] + np.array([-0.48, -0.3]) * 0.1
        targ_cam_trans = orig_cam_trans.copy()
        targ_cam_trans[:2] += approach_delta
        targ_cam_pose = to_pose(targ_cam_trans, orig_cam_rot)
        
        approach_config = dict(approach_steps=100, delta_thresh=1.0)
        approach_plan, approach_qpos = plan_approach(env, 
            observing_qpos, orig_cam_pose, targ_cam_pose, approach_config)
        execute_plan(env, approach_plan)

        obs_wrist = env.get_obs(camera_id=1)  # wrist camera
        rgb, depth, camera_pose, seg = obs_wrist.rgb, obs_wrist.depth, obs_wrist.camera_pose, obs_wrist.seg
        #rgb, depth, camera_pose = obs_wrist.rgb, obs_wrist.depth, obs_wrist.camera_pose
        wrist_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[1].intrinsics
        driller_pose = detect_driller_pose(
            rgb, depth, seg[:,:,0], wrist_camera_matrix, camera_pose, driller_detector)

        if DEBUG:
            gt_pose = env.get_driller_pose()
            R_pred = driller_pose[:3, :3]
            R_gt = gt_pose[:3, :3]
            
            angle_diff = rotation_matrix_to_axis_angle(R_pred @ R_gt.T)
            angle_pred = rotation_matrix_to_axis_angle(R_pred)
            angle_gt = rotation_matrix_to_axis_angle(R_gt)

            print("------Second pose estimation-------")
            print(f"Predicted angle: {angle_pred:.2f} degrees, Ground truth angle: {angle_gt:.2f} degrees")
            print(f"Detected driller pose with angle difference: {angle_diff:.2f} degrees")
            print(f"Detected driller pose:\n {driller_pose}")
            print(f"gt driller pose: \n {gt_pose}")

            Image.fromarray(rgb).save("tmp/driller_rgb_2.png")
            #Image.fromarray(seg).save("tmp/driller_seg_2.png")

        # metric judgement
        Metric['obj_pose'] = env.metric_obj_pose(driller_pose)

    # --------------------------------------step 3: plan grasp and lift------------------------------------------------------
    if not DISABLE_GRASP:
        obj_pose = driller_pose.copy()
        grasps = get_grasps(args.obj)
        valid_grasps = []
        for grasp in grasps:
            valid_grasps.append(grasp)
            valid_grasps.append(
                Grasp(grasps[0].trans, grasps[0].rot @
                      np.diag([-1, -1, 1]), grasps[0].width))

        # try all possible grasps
        orig_eef_trans, orig_eef_rot = env.humanoid_robot_model.fk_eef(observing_qpos)
        orig_eef_pose = to_pose(orig_eef_trans, orig_eef_rot)
        grasp_config = dict(pregrasp_steps=100, reach_steps=50, 
                            lift_steps=50, postgrasp_steps=100, delta_thresh=1.0)
        
        for obj_frame_grasp in valid_grasps:
            robot_frame_grasp = Grasp(
                trans=obj_pose[:3, :3] @ obj_frame_grasp.trans +
                obj_pose[:3, 3],
                rot=obj_pose[:3, :3] @ obj_frame_grasp.rot,
                width=obj_frame_grasp.width,
            )
            grasp_plan = plan_grasp(
                env, approach_qpos, orig_eef_pose, robot_frame_grasp, grasp_config)
                #env, observing_qpos, orig_eef_pose, robot_frame_grasp, grasp_config)
            if grasp_plan is not None:
                break

        if grasp_plan is None:
            print("No valid grasp plan found.")
            env.close()
            return
        
        # pregrasp_plan, reach_plan, lift_plan, postgrasp_plan = grasp_plan
        reach_plan, lift_plan = grasp_plan

        # pregrasp, change if you want
        # pregrasp_plan = plan_move_qpos(
        #     env, observing_qpos, reach_plan[0], steps=50)
        #execute_plan(env, pregrasp_plan)
        open_gripper(env)
        execute_plan(env, reach_plan)
        close_gripper(env)
        execute_plan(env, lift_plan)
        #execute_plan(env, postgrasp_plan)

    # --------------------------------------step 4: plan to move and drop----------------------------------------------------
    if not DISABLE_GRASP and not DISABLE_MOVE:
        move_plan = plan_move(
            env, lift_plan[-1],
            target_container_pose[:3, 3] + np.array([0.0, 0.0, 0.6]),
            end_rot = np.array([
                [ 0.5       ,-0.75      ,-0.4330127],
                [ 0.        ,-0.5       , 0.8660254],
                [-0.8660254 ,-0.4330127 ,-0.25     ],])

            # end_rot = np.array([
            #     [ np.cos(np.pi/6), 0.0, -np.cos(np.pi/3)],
            #     [             0.0, -1.,              0.0],
            #     [-np.cos(np.pi/3), 0.0, -np.cos(np.pi/6)]])
        )
        execute_plan(env, move_plan)
        open_gripper(env)

        ## move back
        init_plan = plan_move_qpos(env, move_plan[-1], observing_qpos, steps=100)
        execute_plan(env, init_plan)
    
    # --------------------------------------step 5: move quadruped backward to initial position------------------------------
    if not DISABLE_MOVE:
        target_list = []
        target_1 = np.eye(4, dtype=np.float32)
        target_1[:3, :3] = target_container_pose[:3, :3]
        target_1[:3, -1] = init_marker_pose[:3, -1]
        target_list.append(target_1)
        target_2 = target_1.copy()
        target_2[:3, :3] = init_marker_pose[:3, :3]
        target_list.append(target_2)
        move_quad(target_list, phase="phase5")

    # test the metrics
    Metric["drop_precision"] = Metric["drop_precision"] or env.metric_drop_precision()
    Metric["quad_return"] = Metric["quad_return"] or env.metric_quad_return()

    print("Metrics:", Metric)

    print("Simulation completed.")
    env.close()


if __name__ == "__main__":
    main()
