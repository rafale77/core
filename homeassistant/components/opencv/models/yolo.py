from ast import literal_eval
from copy import deepcopy
import logging
import math
from pathlib import Path

import torch
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    LeakyReLU,
    Module,
    ModuleList,
    Parameter,
    ReLU,
    ReLU6,
    Sequential,
    Upsample,
)
import yaml  # for torch hub

from .common import SPPCSP, Bottleneck, BottleneckCSP, BottleneckCSP2, Concat, Conv

_LOGGER = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

# flake8: noqa


def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        _LOGGER.warning("Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is Conv2d:
            pass  # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [LeakyReLU, ReLU, ReLU6]:
            m.inplace = True


def model_info(model, verbose=False, imgsz=64):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(
        x.numel() for x in model.parameters() if x.requires_grad
    )  # number gradients
    if verbose:
        _LOGGER.warning(
            "{:>5} {:>40} {:>9} {:>12} {:>20} {:>10} {:>10}".format(
                "layer", "name", "gradient", "parameters", "shape", "mu", "sigma"
            )
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            _LOGGER.warning(
                "%5g %40s %9s %12g %20s %10.3g %10.3g"
                % (
                    i,
                    name,
                    p.requires_grad,
                    p.numel(),
                    list(p.shape),
                    p.mean(),
                    p.std(),
                )
            )
    _LOGGER.warning(
        "Model Summary: {:g} layers, {:g} parameters, {:g} gradients".format(
            len(list(model.parameters())), n_p, n_g
        )
    )


class Detect(Module):
    stride = None  # strides computed during build

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super().__init__()
        self.stride = []  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1, device=device)] * self.nl  # init grid
        a = torch.as_tensor(anchors, dtype=dtype, device=device).view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        self.m = ModuleList(Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny)

            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[
                i
            ]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, self.no))

        return (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).to(device)


class Model(Module):
    def __init__(
        self, cfg="yolov4-p5.yaml", ch=3, nc=None
    ):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml["nc"]:
            _LOGGER.warning(
                "Overriding {} nc={:g} with nc={:g}".format(cfg, self.yaml["nc"], nc)
            )
            self.yaml["nc"] = nc  # override yaml value
        self.model, self.save = parse_model(
            deepcopy(self.yaml), ch=[ch]
        )  # model, savelist, ch_out

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            m.stride = torch.as_tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))],
                device,
            )  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()

    def forward(self, x):
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def _initialize_biases(self, cf=None):
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            _LOGGER.warning(
                ("%6g Conv2d.bias:" + "%10.3g" * 6)
                % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean())
            )

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        _LOGGER.warning("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, Conv):
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
                m.conv = torch.nn.utils.fuse_conv_bn_eval(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self, verbose=False):  # print model information
        model_info(self, verbose)


def parse_model(d, ch):  # model_dict, input_channels(3)
    _LOGGER.warning(
        "\n{:>3}{:>18}{:>3}{:>10}  {:<40}{:<30}".format(
            "", "from", "n", "params", "module", "arguments"
        )
    )
    anchors, nc, gd, gw = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
    )
    na = (
        (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    )  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(
        d["backbone"] + d["head"]
    ):  # from, number, module, args
        m = literal_eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = literal_eval(a) if isinstance(a, str) else a
            except Exception:
                pass
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [
            Conv2d,
            Conv,
            Bottleneck,
            BottleneckCSP,
            BottleneckCSP2,
            SPPCSP,
        ]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2
            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, BottleneckCSP2, SPPCSP]:
                args.insert(2, n)
                n = 1
        elif m is BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]
        m_ = Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        _LOGGER.warning(f"{i:>3}{f:>18}{n:>3}{np:10.0f}  {t:<40}{args:<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        ch.append(c2)
    return Sequential(*layers), sorted(save)
