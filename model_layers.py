import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=stride, padding=1, bias=True
            )
        else:
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.rbr_identity = (
                nn.BatchNorm2d(in_channels)
                if in_channels == out_channels and stride == 1
                else None
            )

    def forward(self, x):
        if self.deploy:
            return self.relu(self.rbr_reparam(x))

        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        return self.relu(out)

    @staticmethod
    def _fuse_conv_bn(conv, bn):
        w = conv.weight
        mean = bn.running_mean
        var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = torch.sqrt(var + eps)
        w_fused = w * (gamma / std).reshape(-1, 1, 1, 1)
        b_fused = beta - mean * gamma / std
        return w_fused, b_fused

    def _pad_1x1_to_3x3(self, kernel):
        if kernel is None:
            return 0
        return F.pad(kernel, [1, 1, 1, 1])

    def _identity_kernel(self):
        kernel = torch.zeros(
            (self.out_channels, self.in_channels, 3, 3),
            device=self.rbr_dense[0].weight.device # 修改点：确保 device 一致
        )
        for i in range(self.out_channels):
            kernel[i, i, 1, 1] = 1
        return kernel

    def get_equivalent_kernel_bias(self):
        k3, b3 = self._fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        k1, b1 = self._fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        k1 = self._pad_1x1_to_3x3(k1)

        if self.rbr_identity is not None:
            kid = self._identity_kernel()
            mean = self.rbr_identity.running_mean
            var = self.rbr_identity.running_var
            gamma = self.rbr_identity.weight
            beta = self.rbr_identity.bias
            eps = self.rbr_identity.eps
            std = torch.sqrt(var + eps)
            kid = kid * (gamma / std).reshape(-1, 1, 1, 1)
            bid = beta - mean * gamma / std
        else:
            kid = 0
            bid = 0

        return k3 + k1 + kid, b3 + b1 + bid

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=3, stride=self.stride, padding=1, bias=True
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        for attr in ['rbr_dense', 'rbr_1x1', 'rbr_identity']:
            if hasattr(self, attr):
                self.__delattr__(attr)
        self.deploy = True