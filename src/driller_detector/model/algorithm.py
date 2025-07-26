import numpy as np
from numpy.typing import NDArray
import dataclasses


@dataclasses.dataclass
class Transform:
    trans: NDArray  # Shape: [*, 3]
    rot: NDArray  # Shape: [*, 3, 3]

    def apply(self, points: NDArray) -> NDArray:
        return points @ np.swapaxes(self.rot, -1, -2) + self.trans[..., None, :]


class Procrustes:

    def __init__(self):
        pass

    def fit(self, src: NDArray[np.float64], dst: NDArray[np.float64]) -> Transform:
        """
        Run orthogonal Procrustes algorithm. Find matrix `R` and translation `t` to minimize
            `||Q - (RP + t)||`

        Parameters
        ----------
        src : NDArray
            matrix P in the formula above, shape [*, 3, 3]
        dst : NDArray
            matrix Q in the formula above, shape [*, 3, 3]

        Returns
        -------
        Transform
            Contains rotation and translation of the transformation.
        """

        # Step 1: Get zero-mean coordinates
        src_mean: NDArray = src.mean(axis=-2, keepdims=True)
        dst_mean: NDArray = dst.mean(axis=-2, keepdims=True)
        src_0 = src - src_mean
        dst_0 = dst - dst_mean

        # Step 2: Orthogonal Procrustes
        rot_raw = dst_0.swapaxes(-1, -2) @ src_0  # Shape: [B, 3, 3]
        U, _, Vh = np.linalg.svd(rot_raw)  # Shape: [B, 3, 3]
        det_UV: NDArray = np.linalg.det(U @ Vh)  # Shape: [B]
        U[..., -1] *= det_UV[..., None]
        rot: NDArray = U @ Vh

        # Step 3: Compute Translation
        trans = np.squeeze(dst_mean - (src_mean @ rot.swapaxes(-1, -2)), axis=-2)

        return Transform(trans=trans, rot=rot)


class Ransac:

    def __init__(self, epoch=256, sample=6, thres=1e-4, proportion=0.6):
        self.epoch = epoch
        self.sample = sample
        self.thres = thres
        self.proportion = proportion

        self.solver = Procrustes()

    def run_single(self, src: NDArray, dst: NDArray) -> Transform:
        """
        Run RANSAC orthogonal Procrustes algorithm.

        Parameters
        ----------
        src : NDArray
            shape [N, 3]
        dst : NDArray
            shape [N, 3]

        Returns
        -------
        Transform
            Contains rotation and translation of the transformation.
        """

        N, _ = dst.shape

        max_inliers = 0
        best_param = None

        for _ in range(self.epoch):
            # Select candidates
            selected = np.random.choice(N, self.sample, replace=False)
            # Compute transformation
            param = self.solver.fit(src[selected], dst[selected])
            # Compute estimation loss
            dist = np.sum((dst - param.apply(src)) ** 2, axis=-1)  # Shape: [N]
            # Find inliers
            inlier: NDArray = dist <= self.thres  # Shape: [N]
            inlier_num = inlier.sum().item()
            # Recompute for those with much inliers
            if inlier_num < self.proportion * N:
                continue
            # Compute transformation
            param = self.solver.fit(src[inlier], dst[inlier])
            # Compute estimation loss
            dist = np.sum((dst - param.apply(src)) ** 2, axis=-1)  # Shape: [N]
            # Find inliers
            inlier: NDArray = dist <= self.thres  # Shape: [N]
            inlier_num = inlier.sum().item()
            # Update for those with more inliers
            if inlier_num > max_inliers:
                max_inliers = inlier_num
                best_param = param

        if best_param is None:
            best_param = self.solver.fit(src, dst)

        return best_param

    def run(self, src: NDArray, dst: NDArray) -> Transform:
        """
        Run batched RANSAC orthogonal Procrustes algorithm through iteration.

        Parameters
        ----------
        src : NDArray
            shape [B, N, 3]
        dst : NDArray
            shape [B, N, 3]

        Returns
        -------
        Transform
            Contains rotation and translation of the transformation.
        """

        B, N, _ = src.shape
        trans, rot = [], []

        for i in range(B):
            param = self.run_single(src[i], dst[i])
            trans.append(param.trans)
            rot.append(param.rot)

        return Transform(trans=np.stack(trans, axis=0), rot=np.stack(rot, axis=0))
