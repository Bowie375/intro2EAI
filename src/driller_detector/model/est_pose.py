from typing import Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F

from . import pointnet
from ..config import Config


# @torch.compile
def convert_6D_to_mat(rot: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation into rotation matrix.

    Parameters
    ----------
    rot : torch.Tensor
        6D rotation representation, shape \(N, 6\)

    Returns
    -------
    torch.Tensor
        Corresponding rotation matrix, shape \(N, 3, 3\)
    """
    a1, a2 = rot[:, :3], rot[:, 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class EstPoseNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Directly estimate the translation vector and rotation matrix.
        """
        super().__init__()
        self.config = config
        self.model = pointnet.Classification(d=9)

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
        B, N, _ = pc.shape

        # Run model
        out = self.model(pc)
        t, r_raw = out[:, :3], out[:, 3:]
        r_mat = convert_6D_to_mat(r_raw)

        # Compute Loss
        # Note: Here we use MSELoss for translation `t`
        loss_t = F.mse_loss(t, trans)
        # Note: This loss is equivalent to minimizing `(1 - cos(theta))`
        loss_r = (3 - (r_mat * rot).sum() / B) / 2
        # Note: Here we simply add them. We can adjust their importance by adding coefficients.
        loss = loss_t + loss_r

        # Log
        metric = dict(
            loss=loss,
            loss_t=loss_t,
            loss_r=loss_r,
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
        out = self.model(pc)
        trans, rot = out[:, :3], out[:, 3:]
        return trans, convert_6D_to_mat(rot)
