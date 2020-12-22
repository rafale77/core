from collections import OrderedDict
from itertools import product
import logging
from math import ceil

import cv2
import numpy as np
from skimage import transform
import torch
from torch.cuda.amp import autocast

from .model import RetinaFace

# flake8: noqa
# from torch2trt import torch2trt
# from torch2trt import TRTModule

_LOGGER = logging.getLogger(__name__)

cfg = {
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 1,
    "ngpu": 4,
    "epoch": 100,
    "decay1": 70,
    "decay2": 90,
    "image_size": 840,
    "pretrain": True,
    "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
    "in_channel": 256,
    "out_channel": 256,
}

REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 87],
    [62.72990036, 87],
]

DEFAULT_CROP_SIZE = (96, 112)


def prior_box(cf, image_size=None, device="cpu"):
    """Boxes for Face."""
    steps = cf["steps"]
    feature_maps = [
        [ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps
    ]
    min_sizes_ = cf["min_sizes"]
    anchors = []

    for k, f in enumerate(feature_maps):
        min_sizes = min_sizes_[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]

    # back to torch land
    output = torch.as_tensor(anchors, device=device).view(-1, 4)
    return output


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
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


def decode_landmark(pre, priors, variances):
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


def nms(box, scores, thresh):
    x1 = box[:, 0]
    y1 = box[:, 1]
    x2 = box[:, 2]
    y2 = box[:, 3]
    zero = torch.as_tensor([0.0], device=scores.device)

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort(descending=True)

    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i)
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.max(zero, xx2 - xx1 + 1)
        h = torch.max(zero, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = torch.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def get_reference_facial_points(output_size=(112, 112)):

    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    # size_diff = max(tmp_crop_size) - tmp_crop_size
    # tmp_5pts += size_diff / 2
    # tmp_crop_size += size_diff
    # return tmp_5pts

    x_scale = output_size[0] / tmp_crop_size[0]
    y_scale = output_size[1] / tmp_crop_size[1]
    tmp_5pts[:, 0] *= x_scale
    tmp_5pts[:, 1] *= y_scale

    return tmp_5pts


class FaceDetector:
    def __init__(
        self,
        weight_path,
        device="cpu",
        confidence_threshold=0.99,
        top_k=5000,
        nms_threshold=0.4,
        keep_top_k=750,
        face_size=(112, 112),
    ):
        """
        RetinaFace Detector with 5points landmarks
        Args:
            weight_path: path of network weight
            device: running device (cuda, cpu)
            face_size: final face size
            face_padding: padding for bounding boxes
        """
        # setting for model
        # model = TRTModule()
        # model.load_state_dict(torch.load("/home/anhman/.homeassistant/model/retina_trt.pth"))
        self.device = device
        state_dict = torch.load(weight_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.model = RetinaFace(cfg).to(device)
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        self.cfg = cfg
        # setting for face detection
        self.thresh = confidence_threshold
        self.top_k = top_k
        self.nms_thresh = nms_threshold
        self.keep_top_k = keep_top_k
        # setting for face align
        self.trans = transform.SimilarityTransform()
        self.out_size = face_size
        self.ref_pts = get_reference_facial_points(output_size=face_size)

    def preprocessor(self, img_raw):
        img = torch.as_tensor(img_raw, dtype=torch.float32, device=self.device)
        scale = torch.as_tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]], device=self.device
        )
        img -= torch.tensor([104, 117, 123]).to(self.device)
        img = img.permute(2, 0, 1).unsqueeze(0)
        return img, scale

    def detect_faces(self, img_raw):
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

        img, scale = self.preprocessor(img_raw)
        # tic = time.time()
        with torch.no_grad():
            with autocast():
                loc, conf, landmarks = self.model(img)  # forward pass

        priors = prior_box(self.cfg, image_size=img.shape[2:], device=self.device)
        boxes = decode(loc.data.squeeze(0), priors, self.cfg["variance"])
        boxes = boxes * scale
        scores = conf.squeeze(0)[:, 1]
        landmarks = decode_landmark(landmarks.squeeze(0), priors, self.cfg["variance"])
        scale1 = torch.as_tensor(
            [
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
            ],
            device=(self.device),
        )
        landmarks = landmarks * scale1

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
        keep = nms(boxes, scores, self.nms_thresh)
        boxes = torch.abs(boxes[keep, :])
        scores = scores[:, None][keep, :]
        landmarks = landmarks[keep, :].reshape(-1, 5, 2)

        # # keep top-K faster NMS
        landmarks = landmarks[: self.keep_top_k, :]
        scores = scores[: self.keep_top_k, :]
        boxes = boxes[: self.keep_top_k, :]

        return boxes, scores, landmarks

    def detect_align(self, img):
        """
        get a image from ndarray, detect faces in image,
        cropped face and align face
        Args:
            img: original image from cv2(BGR) or PIL(RGB)
        Notes:
            coordinate is corresponding to original image
            and type of return image is corresponding to input(cv2, PIL)

        Returns:
            faces:
                a tensor(n, 112, 112, 3) of faces that aligned
            boxes:
                face bounding box for each face
            landmarks:
                face landmarks for each face
        """
        boxes, scores, landmarks = self.detect_faces(img)

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

            self.trans.estimate(src_pts.cpu().numpy(), self.ref_pts)
            face_img = cv2.warpAffine(img, self.trans.params[0:2, :], self.out_size)
            warped.append(face_img)

        faces = torch.as_tensor(warped, dtype=torch.float32, device=self.device)
        return faces, boxes, scores, landmarks
