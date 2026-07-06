import torch
import torch.nn as nn
import torch.nn.functional as F

from aan.data_structures.batch_neurotree import BatchNeuroTree


class GroupNorm(nn.Module):
    def __init__(self, num_groups=16):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, inputs):
        return F.group_norm(inputs, self.num_groups)


class M5_GroupNorm(nn.Module):
    """Sound feature extractor psi_sound based on M5 (paper, Table XIII).

    Batch normalization is replaced with group normalization because the
    per-domain mini-batch size varies during multi-feature extraction
    (paper, Section III-F1).
    """

    def __init__(self, output_size=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.bn1 = GroupNorm()
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.bn2 = GroupNorm()
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = GroupNorm()
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = GroupNorm()
        self.pool4 = nn.MaxPool1d(4)

        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, output_size)

        self.parameter_init()

    def parameter_init(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)

    def forward(self, batch_tree: BatchNeuroTree):
        device = self.conv1.weight.device
        x = torch.stack(batch_tree.getX()).to(device, dtype=torch.float)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.avgPool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return F.leaky_relu(x)

    def setting_pretrained_model(self, model):
        self.conv1 = model.conv1
        self.conv2 = model.conv2
        self.conv3 = model.conv3
        self.conv4 = model.conv4


class FullyConnectedLayer_MFCC(nn.Module):
    """psi for MFCC frames: one FC layer 40 -> output (paper, Experiment 3)."""

    def __init__(self, input_dim=40, output_dim=128):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, output_dim)
        self.parameter_init()

    def parameter_init(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)

    def forward(self, batch_tree: BatchNeuroTree):
        device = self.fc_in.weight.device
        x = torch.stack(batch_tree.getX(), dim=0).to(device, dtype=torch.float)
        x = self.fc_in(x)
        x = F.normalize(x)
        return F.relu(x)
