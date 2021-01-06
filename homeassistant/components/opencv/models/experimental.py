# This file contains experimental modules
import numpy as np
from torch import arange, cat, linspace, sigmoid
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Identity,
    LeakyReLU,
    Module,
    ModuleList,
    Parameter,
    Sequential,
)

from .common import Conv, DWConv

# flake8: noqa


class CrossConv(Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(Module):
    # Cross Convolution CSP
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = LeakyReLU(0.1, inplace=True)
        self.m = Sequential(
            *[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(cat((y1, y2), dim=1))))


class Sum(Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = Parameter(
                -arange(1.0, n) / 2, requires_grad=True
            )  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class GhostConv(Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(
        self, c1, c2, k=1, s=1, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return cat([y, self.cv2(y)], 1)


class GhostBottleneck(Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k, s):
        super().__init__()
        c_ = c2 // 2
        self.conv = Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False))
            if s == 2
            else Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class MixConv2d(Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super().__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = linspace(0, groups - 1e-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[
                0
            ].round()  # solve for equal weight indices, ax = b

        self.m = ModuleList(
            [
                Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False)
                for g in range(groups)
            ]
        )
        self.bn = BatchNorm2d(c2)
        self.act = LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(cat([m(x) for m in self.m], 1)))
