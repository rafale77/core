from collections import namedtuple

import torch
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    LeakyReLU,
    Linear,
    MaxPool2d,
    Module,
    ModuleList,
    PReLU,
    ReLU,
    Sequential,
    Sigmoid,
)
import torch.nn.functional as F
import torchvision.models as models

# flake8: noqa

def Conv(inp, oup, k=3, stride=1, p=1, act=True):
    activation = Identity()
    if k == 1:
        p = 0
    if act==True:
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
        out = F.relu(out)
        return out


class FPN(Module):
    def __init__(self):
        super().__init__()
        in_channels_list = [512,1024,2048]
        self.output1 = Conv(in_channels_list[0], 256, k=1, stride=1)
        self.output2 = Conv(in_channels_list[1], 256, k=1, stride=1)
        self.output3 = Conv(in_channels_list[2], 256, k=1, stride=1)
        self.merge1 = Conv(256, 256)
        self.merge2 = Conv(256, 256)

    def forward(self, inpt):
        # names = list(inpt.keys())
        inpt = list(inpt.values())

        output1 = self.output1(inpt[0])
        output2 = self.output2(inpt[1])
        output3 = self.output3(inpt[2])

        up3 = F.interpolate(
            output3, size=[output2.size(2), output2.size(3)], mode="nearest"
        )
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(
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


class RetinaFace(Module):
    def __init__(self):
        """Define Retina Module."""
        super().__init__()

        return_layers = {"layer2": 1, "layer3": 2, "layer4": 3}

        self.body = models._utils.IntermediateLayerGetter(
            models.resnet50(pretrained=True), return_layers
        )
        in_channels_list = [512,1024,2048]
        self.fpn = FPN()
        self.ssh1 = SSH()
        self.ssh2 = SSH()
        self.ssh3 = SSH()
        self.ClassHead = self._make_class_head()
        self.BboxHead = self._make_bbox_head()
        self.LandmarkHead = self._make_landmark_head()

    def _make_class_head(self):
        classhead = ModuleList()
        for _ in range(3):
            classhead.append(ClassHead())
        return classhead

    def _make_bbox_head(self):
        bboxhead = ModuleList()
        for _ in range(3):
            bboxhead.append(BboxHead())
        return bboxhead

    def _make_landmark_head(self):
        landmarkhead = ModuleList()
        for _ in range(3):
            landmarkhead.append(LandmarkHead())
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat(
            [self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        classifications = torch.cat(
            [self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        ldm_regressions = torch.cat(
            [self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        output = (
            bbox_regressions,
            F.softmax(classifications, dim=-1).select(2, 1),
            ldm_regressions,
        )
        return output


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


class Arcface(Module):
    def __init__(self):
        super().__init__()
        blocks = get_blocks(50)
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        self.output_layer = Sequential(
            BatchNorm2d(512),
            Dropout(0.6),
            Flatten(),
            Linear(512 * 7 * 7, 512),
            BatchNorm1d(512),
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    bottleneck_IR_SE(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        return torch.floor_divide(x, norm)
