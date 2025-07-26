from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn

from . import pointnet
from .algorithm import Procrustes, Ransac
from ..config import Config
from ..vis import Vis


class EstCoordNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Estimate the coordinates in the object frame for each object point.
        """
        super().__init__()
        self.config = config
        self.model = pointnet.Segmentation()
        self.loss = torch.nn.MSELoss()

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
        # Forward
        predict = self.model(pc)
        # Loss
        loss = self.loss(predict, coord)
        # Log
        metric = dict(loss=loss)
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

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """
        USE_RANSAC = True
        # Run model
        coord: torch.Tensor = self.model(pc)  # Shape: [B, N, 3]
        # Into numpy
        pc_numpy = pc.to(torch.float64).numpy(force=True)
        coord_numpy = coord.to(torch.float64).numpy(force=True)
        # Solve the rotation + translation
        if USE_RANSAC:
            solver = Ransac(epoch=128, sample=16, thres=3e-5, proportion=0.5)
            param = solver.run(coord_numpy, pc_numpy)
        else:
            solver = Procrustes()
            param = solver.fit(coord_numpy, pc_numpy)
        # Convert back to torch.Tensor
        return (
            torch.from_numpy(param.trans).clone().to(dtype=pc.dtype, device=pc.device),
            torch.from_numpy(param.rot).clone().to(dtype=pc.dtype, device=pc.device),
        )
