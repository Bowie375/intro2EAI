import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetSetAbstraction2


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()

        ## network for rotation
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        #self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        #self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        #self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)

        self.sa1 = PointNetSetAbstractionMsg(512, [0.05, 0.10, 0.20], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.10, 0.15, 0.25], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

        # self.sa1 = PointNetSetAbstractionMsg(512, [0.07, 0.15, 0.18], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # self.sa2 = PointNetSetAbstractionMsg(128, [0.17, 0.2, 0.25], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        # self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        # self.sa4 = PointNetSetAbstraction2(None, None, None, 640 + 3, [256], True)

        #self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        #self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        #self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 9)

        ## network for translation
        self.fc4 = nn.Linear(3, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.fc7 = nn.Linear(1024, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.fc8 = nn.Linear(512, 256)
        self.bn8 = nn.BatchNorm1d(256)
        self.fc9 = nn.Linear(256, 3)


    def forward(self, xyz: torch.Tensor):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        ## predict rotation
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        x1 = l3_points.view(B, 1024)
        x1 = self.drop1(F.relu(self.bn1(self.fc1(x1))))
        x1 = self.drop2(F.relu(self.bn2(self.fc2(x1))))
        x1 = self.fc3(x1)
        
        ## predict translation
        x2 = F.relu(self.bn4(self.fc4(xyz.transpose(1,2)).transpose(1,2)))
        x2 = F.relu(self.bn5(self.fc5(x2.transpose(1,2)).transpose(1,2)))
        x2 = F.relu(self.bn6(self.fc6(x2.transpose(1,2)).transpose(1,2)))
        x2 = torch.max(x2, dim=2)[0]
        x2 = F.relu(self.bn7(self.fc7(x2)))
        x2 = F.relu(self.bn8(self.fc8(x2)))
        x2 = self.fc9(x2)

        x = torch.cat([x2, x1], dim=1)
        return x,l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


if __name__ == '__main__':
    import torch
    model = get_model(num_class=12, normal_channel=False)
    xyz = torch.rand(32, 1024, 3).permute(0, 2, 1)
    x, trans_feat = model(xyz)
    print(x.shape)
    print(trans_feat.shape)
