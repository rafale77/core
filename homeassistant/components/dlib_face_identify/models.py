from collections import OrderedDict
import logging
from pathlib import Path

import cv2
from skimage.transform import SimilarityTransform
import numpy as np
import torch
from torch.nn import BatchNorm1d, BatchNorm2d, Conv2d, Dropout, Identity, Linear, Module, ModuleList, PReLU, Sequential
from torch.nn.functional import softmax
import torchvision.models as models

from .net import bottleneck_IR_SE, get_blocks, Flatten, FPN, SSH, ClassHead, BboxHead, LandmarkHead, SEModule

# flake8: noqa


Bottleneck = models.resnet.Bottleneck
_LOGGER = logging.getLogger(__name__)
home = str(Path.home()) + "/.homeassistant/"
DEFAULT_CROP_SIZE = (96, 112)
REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 87],
    [62.72990036, 87],
]
face_size = (112, 112)
variances = [0.1, 0.2]
trans = SimilarityTransform()


def get_reference_facial_points():

    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    x_scale = face_size[0] / DEFAULT_CROP_SIZE[0]
    y_scale = face_size[1] / DEFAULT_CROP_SIZE[1]
    tmp_5pts[:, 0] *= x_scale
    tmp_5pts[:, 1] *= y_scale
    return tmp_5pts

def fuse_conv_and_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = Conv2d(conv.in_channels,
                              conv.out_channels,
                              kernel_size=conv.kernel_size,
                              stride=conv.stride,
                              padding=conv.padding,
                              bias=True).to(conv.weight.device)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv

def fuse(model):  # fuse model Conv2d() + BatchNorm2d() layers
    for m in model.modules():
        if type(m) is Bottleneck:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
            m.conv1 = fuse_conv_and_bn(m.conv1, m.bn1)  # update conv
            m.bn1 = Identity()  # remove batchnorm
            m.conv2 = fuse_conv_and_bn(m.conv2, m.bn2)  # update conv
            m.bn2 = Identity()  # remove batchnorm
            m.conv3 = fuse_conv_and_bn(m.conv3, m.bn3)  # update conv
            m.bn3 = Identity()  # remove batchnorm
            #m.forward = m.fuseforward  # update forward
    return model


def fuse_bn_sequential(block):
    """
    This function takes a sequential block and fuses the batch normalization with convolution
    :param model: nn.Sequential. Source resnet model
    :return: nn.Sequential. Converted block
    """
    if not isinstance(block, Sequential):
        return block
    stack = []
    for m in block.children():
        if len(stack) == 0:
            stack.append(m)
        elif isinstance(m, BatchNorm2d):
            if isinstance(stack[-1], Conv2d):
                bn_st_dict = m.state_dict()
                conv_st_dict = stack[-1].state_dict()

                # BatchNorm params
                eps = m.eps
                mu = bn_st_dict['running_mean']
                var = bn_st_dict['running_var']
                gamma = bn_st_dict['weight']

                if 'bias' in bn_st_dict:
                    beta = bn_st_dict['bias']
                else:
                    beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

                # Conv params
                W = conv_st_dict['weight']
                if 'bias' in conv_st_dict:
                    bias = conv_st_dict['bias']
                else:
                    bias = torch.zeros(W.size(0)).float().to(gamma.device)

                denom = torch.sqrt(var + eps)
                b = beta - gamma.mul(mu).div(denom)
                A = gamma.div(denom)
                bias *= A
                A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

                W.mul_(A)
                bias.add_(b)

                stack[-1].weight.data.copy_(W)
                if stack[-1].bias is None:
                    stack[-1].bias = torch.nn.Parameter(bias)
                else:
                    stack[-1].bias.data.copy_(bias)

        else:
            stack.append(m)

    if len(stack) > 1:
        return Sequential(*stack)
    else:
        return stack[0]


def fuse_bn_recursively(model):
    for module_name in model._modules:
        model._modules[module_name] = fuse_bn_sequential(model._modules[module_name])
        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])
    return model


class RetinaFace(Module):
    def __init__(self):
        """Define Retina Module."""
        super().__init__()

        return_layers = {"layer2": 1, "layer3": 2, "layer4": 3}

        self.body = models._utils.IntermediateLayerGetter(
            models.resnet50(pretrained=True), return_layers
        )
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
            softmax(classifications, dim=-1).select(2, 1),
            ldm_regressions,
        )
        return output

class FaceDetector:
    def __init__(self):
        """RetinaFace Detector with 5points landmarks."""

        self.thresh = 0.99
        self.top_k = 5000
        self.nms_thresh = 0.4
        self.keep_top_k = 750
        self.ref_pts = get_reference_facial_points()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#        self.model = torch.jit.load("/home/anhman/.homeassistant/model/RetinaJIT.pth", map_location=self.device)
#        self.model = torch.load("/home/anhman/.homeassistant/model/RetinaFace.pth", map_location=self.device)
        self.model = RetinaFace().to(self.device)
        self.model.load_state_dict(torch.load(
            home + "model/Resnet50_Final.pth", map_location=self.device
        ))
        self.model = fuse(self.model)
        self.model = fuse_bn_recursively(self.model)
        self.model.eval()
#        self.traced = False
#        torch.save(self.model, "/home/anhman/.homeassistant/model/RetinaFace.pth")

    def decode(self, loc, priors):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """

        boxes = torch.cat(
            (
                priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
            ),
            1,
        )
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def decode_landmark(self, pre, priors):
        """Decode landm from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            pre (tensor): landm predictions for loc layers,
                Shape: [num_priors,10]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded landm predictions
        """
        landms = torch.cat(
            (
                priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
            ),
            dim=1,
        )
        return landms

    def detect_faces(self, img, priors):
        """
        get a image from ndarray, detect faces in image
        Args:
            img_raw: original image from cv2(BGR) or PIL(RGB)
        Notes:
            coordinate is corresponding to original image
            and type of return image is corresponding to input(cv2, PIL)
        Returns:
            boxes:
                faces bounding box for each face
            scores:
                percentage of each face
            landmarks:
                faces landmarks for each face
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast():
#                if self.traced == False:
#                   self.model = torch.jit.trace(self.model, (img))
#                    self.traced = True
                loc, conf, landmarks = self.model(img)  # forward pass
        boxes = self.decode(loc.data.squeeze(0), priors)
        h, w = img.shape[2], img.shape[3]
        boxes = boxes * torch.as_tensor([w, h, w, h], device=self.device)
        scores = conf.squeeze(0)
        landmarks = self.decode_landmark(landmarks.squeeze(0), priors)
        landmarks = landmarks * torch.as_tensor(
            [w, h, w, h, w, h, w, h, w, h], device=self.device
        )

        # ignore low scores
        index = torch.where(scores > self.thresh)[0]
        boxes = boxes[index]
        landmarks = landmarks[index]
        scores = scores[index]

        # keep top-K before NMS
        order = scores.argsort(dim=0, descending=True)[: self.top_k]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        # Do NMS
        keep = torch.ops.torchvision.nms(boxes, scores, self.nms_thresh)
        scores = scores[:, None][keep, :]
        landmarks = landmarks[keep, :].reshape(-1, 5, 2)

        # # keep top-K faster NMS
        landmarks = landmarks[: self.keep_top_k, :]
        scores = scores[: self.keep_top_k, :]

        return scores, landmarks

    def detect_align(self, image, img, priors):
        """
        get a image from ndarray, detect faces in image,
        cropped face and align face
        Args:
            image: original image from cv2(BGR) or PIL(RGB)
            img: tensorized image
        Returns:
            faces:
                a tensor(n, 112, 112, 3) of faces that aligned
            boxes:
                face bounding box for each face
            landmarks:
                face landmarks for each face
        """
        scores, landmarks = self.detect_faces(img, priors)
        warped = []
        for src_pts in landmarks:
            if max(src_pts.shape) < 3 or min(src_pts.shape) != 2:
                raise _LOGGER.warning(
                    "RetinaFace facial_pts.shape must be (K,2) or (2,K) and K>2"
                )
            if src_pts.shape[0] == 2:
                src_pts = src_pts.T
            if src_pts.shape != self.ref_pts.shape:
                raise _LOGGER.warning(
                    "RetinaFace facial_pts and reference_pts must have the same shape"
                )
            trans.estimate(src_pts.cpu().numpy(), self.ref_pts)
            face_img = cv2.warpAffine(image, trans.params[0:2, :], face_size)
            warped.append(face_img)
        faces = torch.as_tensor(warped, dtype=torch.float32, device=self.device)
        return faces, scores


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

class FaceEncoder:

    def __init__(self):
        """ArcFace Recognizer with 5points landmarks."""

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#        self.arcmodel = torch.load("/home/anhman/.homeassistant/model/ArcFace.pth", map_location=self.device)
        self.arcmodel = Arcface().to(self.device)
        self.arcmodel.load_state_dict(
            torch.load(home + "model/model_ir_se50.pth", map_location=self.device)
        )
        self.arcmodel = fuse_bn_recursively(self.arcmodel)
        self.arcmodel.eval()
#        torch.save(self.arcmodel, "/home/anhman/.homeassistant/model/ArcFace.pth")
#        self.arcmodel = torch.jit.script(self.arcmodel)
#        self.arcmodel.save("/home/anhman/.homeassistant/model/ArcfaceJIT.pth")

    def recog(self, x):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                return self.arcmodel(x)
