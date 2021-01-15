from collections import namedtuple

import torch
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    LeakyReLU,
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
    if act == True:
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


class Flatten(Module):
    def forward(self, inpt):
        return inpt.view(inpt.size(0), -1)


class SEModule(Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False
        )
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False
        )
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for i in range(num_units - 1)
    ]


def get_blocks(num_layers):
    blocks = [
        get_block(in_channel=64, depth=64, num_units=3),
        get_block(in_channel=64, depth=128, num_units=4),
        get_block(in_channel=128, depth=256, num_units=14),
        get_block(in_channel=256, depth=512, num_units=3),
    ]
    return blocks
