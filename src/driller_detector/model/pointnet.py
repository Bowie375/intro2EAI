"""
Implements PointNet as the backbone model, including the classification and
segmentation models.

https://arxiv.org/pdf/1612.00593:
    PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
"""

import torch
import torch.nn as nn

# torch.set_float32_matmul_precision("high")


class Classification(nn.Module):

    def __init__(self, d=3):
        super().__init__()

        self.d = d

        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc = nn.Linear(256, self.d)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    # @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        # MLP
        x = x.reshape((B * N, 3))
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        # Max Pooling
        x = x.reshape((B, N, -1)).max(dim=1).values
        # MLP
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.relu(self.bn5(self.fc5(x)))
        out = self.fc(x)
        return out


class Segmentation(nn.Module):

    def __init__(self, d=3):
        super().__init__()

        self.d = d

        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(64 + 1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc = nn.Linear(128, self.d)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

    # @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        # MLP
        x = x.reshape((B * N, 3))
        x = self.relu(self.bn1(self.fc1(x)))
        # Compute Feature
        feature = self.relu(self.bn2(self.fc2(x)))
        feature = self.relu(self.bn3(self.fc3(feature)))
        feature = torch.reshape(feature, (B, N, -1)).max(dim=1, keepdim=True).values
        # Concat Feature
        x = torch.concat((x.reshape(B, N, -1), feature.expand(-1, N, -1)), dim=-1)
        x = x.reshape((B * N, -1))
        # MLP
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.relu(self.bn5(self.fc5(x)))
        x = self.relu(self.bn6(self.fc6(x)))
        out = self.fc(x)
        # Reshape
        return torch.reshape(out, (B, N, -1))
