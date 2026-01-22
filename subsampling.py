import torch
import torch.nn as nn
from model_layers import RepVGGBlock, SEBlock


class RepVGGSEInput(nn.Module):

    def __init__(self, in_channels=1, hidden_dim=256, deploy=False):
        super(RepVGGSEInput, self).__init__()

        self.conv1 = RepVGGBlock(in_channels, 32, stride=2, deploy=deploy)
        self.se1 = SEBlock(32)
        self.conv2 = RepVGGBlock(32, 64, stride=2, deploy=deploy)
        self.se2 = SEBlock(64)
        self.out_projection = nn.Linear(64 * 20, hidden_dim)

    def forward(self, x):

        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.se1(x)
        x = self.conv2(x)
        x = self.se2(x)
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, -1)
        x = self.out_projection(x)
        return x

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()