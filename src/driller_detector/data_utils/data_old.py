import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, Optional
import random
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader

from config import Config
from vis import Vis

from constants import DEPTH_IMG_SCALE, TABLE_HEIGHT, PC_MAX, PC_MIN, OBJ_INIT_TRANS
from utils import get_pc, get_workspace_mask
from robot.cfg import get_robot_cfg


class PoseDataset(Dataset):
    def __init__(self, config: Config, mode: str, scale: int = 1):
        """
        Dataset for pose estimation

        Parameters
        ----------
        config: Config
            Configuration object
        mode: str
            Mode of the dataset (e.g. train or val)
        scale: int
            Scale of the dataset, used to make the dataset larger
            so that we don't need to wait for the restart of the dataloader
        """
        super().__init__()
        self.config = config
        self.robot_cfg = get_robot_cfg(config.robot)
        if mode == "train":
            self.data_root = os.path.join("data", "est_pose_new")
        elif mode == "val":
            self.data_root = os.path.join("data", "est_pose_val_new")
        self.files = sorted(os.listdir(self.data_root))
        self.files = self.files * scale
        random.shuffle(self.files)
        

    def __len__(self) -> int:
        """
        For a torch dataset, a __len__ is required.

        Returns
        -------
        int
            Length of the dataset
        """
        return len(self.files)

    def __getitem__(self, idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        For a torch dataset, a __getitem__ is required.

        Parameters
        ----------
        idx: Optional[int]
            Index of the item to get. If None, a random index is used.

        Returns
        -------
        A dict of:

            pc: the point cloud in camera frame with shape (N, 3)

            trans: the ground truth translation vector with shape (3,)

            rot: the ground truth rotation matrix with shape (3, 3)

            coord: the ground truth coordinates in the object frame with shape (N, 3)

            camera_pose: the camera pose with shape (4, 4) (Used in simulation evaluation)

            obj_pose_in_world: the object pose in world frame with shape (4, 4) (Used in simulation evaluation)

        Note that they will be converted to torch tensors in the dataloader.

        The shape will be (B, ...) for batch size B when you get the data from the dataloader.
        """
        try:
            f = self.files[idx] if idx is not None else random.choice(self.files)
            fdir = os.path.join(self.data_root, f)
            print(f"Loading file: {fdir}")
            obj_pose = np.load(os.path.join(fdir, "obj_pose.npy"))
            # if not np.linalg.norm(obj_pose[:2, 3] - OBJ_INIT_TRANS[:2]) < 0.1:
            #     # some times the object will be out of the workspace
            #     # so we need to skip this sample
            #     # this rarely happens so we don't need to worry about it
            #     print(f"Object pose is not in the workspace, skipping {fdir}")
            #     return self.__getitem__()
            camera_pose = np.load(os.path.join(fdir, "camera_pose.npy"))
            depth_array = (
                np.array(
                    cv2.imread(os.path.join(fdir, "depth.png"), cv2.IMREAD_UNCHANGED)
                )
                / DEPTH_IMG_SCALE
            )

            intrinsics = np.load(os.path.join(fdir, "camera_intrinsics.npy"))
            
            
            ## mask and sample the point cloud
            seg = cv2.imread(
                os.path.join(fdir, "obj_seg.png"), cv2.THRESH_BINARY
            )
            
            object_mask = seg > 0.5
            
            full_pc_camera = get_pc(
                depth_array, intrinsics
            ) * np.array([-1, -1, 1])
            
            
            valid_depth_mask = depth_array > 0
            
            object_mask_flat = object_mask.flatten()
            valid_depth_flat = valid_depth_mask.flatten()
            
            combined_mask = object_mask_flat & valid_depth_flat
            
            pc_camera_obj = full_pc_camera[combined_mask]
            
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
    
            #pc_camera_sampled = FPS(pc_camera_obj, 5000)
            #pc_camera_sampled += np.random.normal(0, 0.01, pc_camera_sampled.shape)  # add some noise
            
            # if pc_camera_obj.shape[0] > 15000:
            #     target_points = min(10000, max(3000, pc_camera_obj.shape[0] // 3))
            #     downsample_idx = np.random.choice(pc_camera_obj.shape[0], target_points, replace=False)
            #     pc_camera_obj = pc_camera_obj[downsample_idx]
            #     # print(f"Adaptively downsampled to: {pc_camera_obj.shape[0]} points")
            
            # full_pc_world = (
            #     np.einsum("ab,nb->na", camera_pose[:3, :3], full_pc_camera)
            #     + camera_pose[:3, 3]
            # )
            
            # full_coord = np.einsum(
            #     "ba,nb->na", obj_pose[:3, :3], full_pc_world - obj_pose[:3, 3]
            # )
            
            #pc_camera_obj[:, :2] *= np.array([-1.,-1.])
            pc_world_obj =  (
                np.einsum("ab,nb->na", camera_pose[:3, :3], pc_camera_obj)
                + camera_pose[:3, 3]
            )
            
            coord_obj = np.einsum(
                "ba,nb->na", obj_pose[:3, :3], pc_world_obj - obj_pose[:3, 3]
            )


            if pc_camera_obj.shape[0] == 0:
                #print(f"dir {fdir}: No object points found, skipping this sample")
                #return fdir
                return self.__getitem__()
            elif pc_camera_obj.shape[0] >= self.config.point_num:
                sel_idx = np.random.choice(pc_camera_obj.shape[0], self.config.point_num, replace=False)
            else:
                sel_idx = np.random.choice(pc_camera_obj.shape[0], self.config.point_num, replace=True)

            pc_camera = pc_camera_obj[sel_idx]
            coord = coord_obj[sel_idx]
            
            import trimesh
            trimesh.PointCloud(pc_world_obj).export("/root/workspace/Assignment4-final/tmp/pcd_w.ply")
            trimesh.PointCloud(pc_camera).export("/root/workspace/Assignment4-final/tmp/pcd_r.ply")
            trimesh.PointCloud(coord).export("/root/workspace/Assignment4-final/tmp/pcd_std.ply")
            cv2.imwrite(
                "/root/workspace/Assignment4-final/tmp/pcd_seg.png",
                (object_mask * 255).astype(np.uint8)
            )
            
            # print(f"pc_camera: {pc_camera.shape}, coord_obj: {coord_obj.shape}")
            rel_obj_pose = np.linalg.inv(camera_pose) @ obj_pose

            # print("translation:", rel_obj_pose[:3, 3])
            # print("rotation:", rel_obj_pose[:3, :3])
            
            # cv2.imwrite(
            #     "/root/workspace/Assignment4-final/seg.png",
            #     (object_mask * 255).astype(np.uint8)
            # )
            # exit(0)
        
            return dict(
                pc=pc_camera.astype(np.float32),
                coord=coord.astype(np.float32),
                trans=rel_obj_pose[:3, 3].astype(np.float32),
                rot=rel_obj_pose[:3, :3].astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                obj_pose_in_world=obj_pose.astype(np.float32),
            )

        except Exception as e:
            print(f"Error in {fdir}: {e}")
            return self.__getitem__()


class Loader:
    # a simple wrapper for DataLoader which can get data infinitely
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iter = iter(self.loader)

    def get(self) -> dict:
        try:
            data = next(self.iter)
        except StopIteration:
            print("Restarting iterator...")
            self.iter = iter(self.loader)
            data = next(self.iter)
        return data

if __name__ == "__main__":
    # Example usage
    config = Config()
    dataset = PoseDataset(config, mode="train", scale=1)
    loader = Loader(DataLoader(dataset, batch_size=1, shuffle=False))

    l = []
    for i in range(1):
        data = loader.get()