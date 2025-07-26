from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn

from ..config import Config
from ..vis import Vis

from .pointnet2.pointnet2_part_seg_msg import get_model

def myprint(msg: str, **kwargs):
    verbose = False
    if verbose:
        print(msg, **kwargs)
    else:
        pass

class EstCoordNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Estimate the coordinates in the object frame for each object point.
        """
        super().__init__()
        self.config = config
        self.model = get_model(num_classes=3, normal_channel=False)

    def forward(
        self, pc: torch.Tensor, coord: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstCoordNet

        Parameters
        ----------
        pc: torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        coord: torch.Tensor
            Ground truth coordinates in the object frame, shape \(B, N, 3\)

        Returns
        -------
        float
            The loss value according to ground truth coordinates
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """

        coord_pred, feat = self.model(pc.permute(0,2,1))

        loss = torch.mean(torch.norm(coord_pred - coord, dim=-1))

        metric = dict(
            loss=loss,
            # additional metrics you want to log
        )
        return loss, metric

    def est(self, pc: torch.Tensor):
        """
        Estimate translation and rotation in the camera frame

        Parameters
        ----------
            Point cloud in camera frame, shape (B, N, 3)
            Point cloud in camera frame, shape \(B, N, 3\)

        Returns
        -------
        trans: torch.Tensor
            Estimated translation vector in camera frame, shape \(B, 3\)
        rot: torch.Tensor
            Estimated rotation matrix in camera frame, shape \(B, 3, 3\)

        Note
        ----
        The rotation matrix should satisfy the requirement of orthogonality and determinant 1.

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """

        def SE3(R: torch.Tensor, t: torch.Tensor):
            """ Construct SE(3) matrix from rotation matrix and translation vector """
            m = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
            m[:, :3, :3] = R
            m[:, :3, 3] = t
            m[:, 3, 3] = 1
            return m

        def ICP(pc: torch.Tensor, coord: torch.Tensor, max_iter: int = 100, eps: float = 1e-5):
            """
            params:
                pc: [B, N, 3]
                coord: [B, N, 3]
            return:
                trans: [B, 3]
                rot: [B, 3, 3]
                error: float
            """
            
            B, N, _ = pc.shape
            tf = torch.eye(4, device=pc.device, dtype=pc.dtype).unsqueeze(0).repeat(B, 1, 1)
            prev_error = float('inf')

            for i in range(max_iter):
                coord_centroid = coord.mean(1, keepdim=True)	# [B, 1, 3]
                pc_centroid = pc.mean(1, keepdim=True)			# [B, 1, 3]

                coord_centered = coord - coord_centroid
                pc_centered = pc - pc_centroid

                H = pc_centered.transpose(1, 2) @ coord_centered  # [B, 3, 3]
                U, S, V_T = torch.linalg.svd(H)

                R_T = V_T.transpose(1, 2) @ U.transpose(1, 2)

                # Fix reflection
                mask = torch.det(R_T) < 0
                V_T[mask, :, -1] *= -1
                R_T[mask] = V_T[mask].transpose(1, 2) @ U[mask].transpose(1, 2)

                t = pc_centroid.squeeze(1) - (coord_centroid @ R_T).squeeze(1)

	            # Update coord
                coord = coord @ R_T + t.unsqueeze(1)

                # Accumulate transform (apply last → first)
                T = SE3(R_T.transpose(1, 2), t)
                tf = T @ tf

                # Check convergence
                mean_error = torch.mean(torch.norm(coord - pc, dim=-1))
                myprint(f"In ICP iter {i}: mean_error={mean_error:.6f}")
                if torch.abs(prev_error - mean_error) < eps:
                    myprint("Converged.")
                    prev_error = mean_error
                    break
                prev_error = mean_error

            return tf[:, :3, 3], tf[:, :3, :3], prev_error

        def RANSAC(pc: torch.Tensor, coord: torch.Tensor, 
                   max_iter: int = 10, inlier_thresh: float = 0.01):
            """
            params:
                pc: [N, 3]
                coord: [N, 3]
            return:
                trans: [3]
                rot: [3, 3]
                error: float
            """

            N, _ = pc.shape
            num_inliers = 0
            inlier_mask = None

            for i in range(max_iter):
                # Randomly sample 3 points
                idxs = np.random.choice(N, 3, replace=False)
                pc_sampled = pc[idxs, :]
                coord_sampled = coord[idxs, :]

                # Estimate transform
                trans, rot, error = ICP(pc_sampled.unsqueeze(0), coord_sampled.unsqueeze(0))
                trans = trans.squeeze(0)
                rot = rot.squeeze(0)

                # Count inliers
                dist = torch.norm(pc - (coord @ rot.T + trans), dim=-1)
                mask = (dist < inlier_thresh)
                inliers = mask.sum()

                myprint(f"In RANSAC iter {i}: min_dist={dist.min():.6f}, max_dist={dist.max():.6f}, inliers={inliers}/{N}")

                if inliers > num_inliers:
                    num_inliers = inliers
                    inlier_mask = mask

			# use all inliers to refine the transform
            if inlier_mask is None:
                inlier_mask = torch.ones(N, dtype=torch.bool, device=pc.device)
            trans, rot, _ = ICP(pc[inlier_mask].unsqueeze(0), coord[inlier_mask].unsqueeze(0))
            return trans.squeeze(0), rot.squeeze(0)

        B, N, _ = pc.shape
        coord, feat = self.model(pc.permute(0,2,1))
        trans = []
        rot = []
        for i in range(B):
            # Estimate translation and rotation
            trans_i, rot_i = RANSAC(pc[i], coord[i])
            trans.append(trans_i)
            rot.append(rot_i)

        trans = torch.stack(trans, dim=0)
        rot = torch.stack(rot, dim=0)

        return trans, rot