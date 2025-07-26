from typing import Tuple, Dict
import torch
from torch import nn

from ..config import Config
from .pointnet2.pointnet2_cls_msg import get_model

class EstPoseNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Directly estimate the translation vector and rotation matrix.
        """
        super().__init__()
        self.config = config
        self.model = get_model(num_class=12, normal_channel=False)

    def forward(
        self, pc: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstPoseNet

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        trans : torch.Tensor
            Ground truth translation vector in camera frame, shape \(B, 3\)
        rot : torch.Tensor
            Ground truth rotation matrix in camera frame, shape \(B, 3, 3\)

        Returns
        -------
        float
            The loss value according to ground truth translation and rotation
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """

        def rot_dist(r1: torch.tensor, r2: torch.tensor) -> float:
            """
            The relative rotation angle between two rotation matrices.

            Parameters
            ----------
            r1 : torch.tensor
                The first rotation matrix (N, 3, 3).
            r2 : torch.tensor
                The second rotation matrix (N, 3, 3).

            Returns
            -------
            float
                The relative rotation angle in radians.
            """

            trace = torch.diagonal(r1 @ r2.transpose(1,2), dim1=-2, dim2=-1).sum(dim=1)
            return torch.arccos(torch.clip((trace - 1) / 2, -1, 1)).mean()

        trans_pred, rot_pred = self.est(pc)

        if self.config.use_mse_loss_on_rot:
            #gt_tf = torch.cat([trans, rot.reshape(-1, 9)], dim=1)
            #pred_tf = torch.cat([trans_pred, rot_pred.reshape(-1, 9)], dim=1)
            #loss = nn.functional.mse_loss(pred_tf, gt_tf)
            rot_loss = nn.functional.mse_loss(rot_pred, rot)
            trans_loss = nn.functional.mse_loss(trans_pred, trans)
            loss = rot_loss + trans_loss
        else:
            rot_loss = rot_dist(rot_pred, rot)
            trans_loss = nn.functional.mse_loss(trans_pred, trans)
            gamma = 10.0
            loss = gamma * trans_loss + rot_loss


        metric = dict(
            loss=loss,
            # additional metrics you want to log
            trans_distance=torch.norm(trans_pred - trans, dim=1).mean(),
            rot_distance=rot_dist(rot_pred, rot),
        )
        return loss, metric

    def est(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate translation and rotation in the camera frame

        Parameters
        ----------
        pc : torch.Tensor
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
        """
        pred, feat = self.model(pc.permute(0,2,1))
        trans_pred = pred[:, :3]
        rot_pred_raw = pred[:, 3:].reshape(-1, 3, 3)

        U, S, V_T = torch.linalg.svd(rot_pred_raw)
        U_prime = U.clone()
        det = torch.det(U @ V_T)
        mask = (det < 0)

        U_prime[mask, :, -1] *= -1
        rot_pred = torch.zeros_like(rot_pred_raw, device=pc.device, dtype=pc.dtype)
        rot_pred[mask] = U_prime[mask] @ V_T[mask]
        rot_pred[~mask] = U[~mask] @ V_T[~mask]

        return trans_pred, rot_pred


if __name__ == '__main__':
    model = EstPoseNet(None)
    pc = torch.rand(16, 1024, 3)

    trans, rot = model.est(pc)
    print(trans.shape, rot.shape)

    loss, metric = model(pc, trans, rot)
    print(loss, metric)