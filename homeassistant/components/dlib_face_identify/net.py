from collections import namedtuple

import torch
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    LeakyReLU,
    Linear,
    MaxPool2d,
    Module,
    PReLU,
    ReLU,
    Sequential,
    Sigmoid,
)
from torch.nn.functional import interpolate, relu


# flake8: noqa


def Conv(inp, oup, k=3, stride=1, p=1, act=True):
    if k == 1:
        p = 0
    if act:
        return Sequential(
            Conv2d(inp, oup, k, stride, padding=p, bias=False),
            BatchNorm2d(oup),
            LeakyReLU(negative_slope=0, inplace=True),
        )
    return Sequential(
        Conv2d(inp, oup, k, stride, padding=p, bias=False),
        BatchNorm2d(oup),
    )


class SSH(Module):
    def __init__(self):
        super().__init__()
        self.conv3X3 = Conv(256, 256 // 2, stride=1, act=False)
        self.conv5X5_1 = Conv(256, 256 // 4, stride=1)
        self.conv5X5_2 = Conv(256 // 4, 256 // 4, stride=1, act=False)
        self.conv7X7_2 = Conv(256 // 4, 256 // 4, stride=1)
        self.conv7x7_3 = Conv(256 // 4, 256 // 4, stride=1, act=False)

    def forward(self, inpt):
        conv3X3 = self.conv3X3(inpt)
        conv5X5_1 = self.conv5X5_1(inpt)
        conv5X5 = self.conv5X5_2(conv5X5_1)
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = relu(out)
        return out


class FPN(Module):
    def __init__(self):
        super().__init__()
        in_channels_list = [512, 1024, 2048]
        self.output1 = Conv(in_channels_list[0], 256, k=1, stride=1)
        self.output2 = Conv(in_channels_list[1], 256, k=1, stride=1)
        self.output3 = Conv(in_channels_list[2], 256, k=1, stride=1)
        self.merge1 = Conv(256, 256)
        self.merge2 = Conv(256, 256)

    def forward(self, inpt):
        inpt = list(inpt.values())

        output1 = self.output1(inpt[0])
        output2 = self.output2(inpt[1])
        output3 = self.output3(inpt[2])

        up3 = interpolate(
            output3, size=[output2.size(2), output2.size(3)], mode="nearest"
        )
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = interpolate(
            output2, size=[output1.size(2), output1.size(3)], mode="nearest"
        )
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out


class ClassHead(Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = Conv2d(256, 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = Conv2d(256, 8, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = Conv2d(256, 20, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


#  Original Arcface Model ######################################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBlock(Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc = Sequential(
            Linear(channel, channel // reduction),
            PReLU(),
            Linear(channel // reduction, channel),
            Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class IRBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super().__init__()
        self.bn0 = BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = BatchNorm2d(inplanes)
        self.prelu = PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out

    def fuseforward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out
