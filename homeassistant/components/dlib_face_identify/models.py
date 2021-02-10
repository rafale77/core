import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from skimage.transform import SimilarityTransform
import torch
from torchvision.transforms.functional import normalize

# flake8: noqa


_LOGGER = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
trans = SimilarityTransform()


def get_reference_facial_points():
    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    x_scale = face_size[0] / DEFAULT_CROP_SIZE[0]
    y_scale = face_size[1] / DEFAULT_CROP_SIZE[1]
    tmp_5pts[:, 0] *= x_scale
    tmp_5pts[:, 1] *= y_scale
    return tmp_5pts


@torch.jit.script
def l2_norm(inp):
    return torch.nn.functional.normalize(inp, 2.0, 1)


def faces_preprocessing(faces):
    """Prepare face tensor."""
    dev = torch.device("cuda:0")
    faces = (
        torch.as_tensor(faces, dtype=torch.float32, device=dev)
        .permute(0, 3, 1, 2)
        .div(255)
    )
    return normalize(faces, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)


@torch.jit.script
def decode_landmark(pre, priors, variances: List[float]):
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


@torch.jit.script
def decode(loc, priors, variances: List[float]):
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


@torch.jit.script
def postprocess(input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], img, priors):
    """Decode-NMS to landmarks."""
    dev = torch.device("cuda:0")
    thresh = 0.99
    top_k = 5000
    nms_thresh = 0.4
    keep_top_k = 750
    variances = [0.1, 0.2]

    # ignore low scores
    scores = input[1].squeeze(0)
    index = torch.where(scores > thresh)[0]
    if len(index) > 0:
        h, w = img.shape[2], img.shape[3]
        landmarks = decode_landmark(input[2].squeeze(0), priors, variances)
        landmarks *= torch.as_tensor([w, h, w, h, w, h, w, h, w, h], device=dev)
        landmarks = landmarks[index]
        if len(index) == 1:
            return landmarks.reshape(-1, 5, 2)

        boxes = decode(input[0].data.squeeze(0), priors, variances)
        boxes *= torch.as_tensor([w, h, w, h], device=dev)

        boxes = boxes[index]
        scores = scores[index]

        # keep top-K before NMS
        order = scores.argsort(dim=0, descending=True)[:top_k]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        # Do NMS
        keep = torch.ops.torchvision.nms(boxes, scores, nms_thresh)
        landmarks = landmarks[keep, :].reshape(-1, 5, 2)

        # # keep top-K faster NMS
        return landmarks[:keep_top_k, :]


class FaceDetector:
    def __init__(self):
        """RetinaFace Detector with 5points landmarks."""

        self.ref_pts = get_reference_facial_points()
        self.model = torch.jit.load(home + "model/RetinaFaceJIT.pth", map_location=device)
        self.arcmodel = torch.jit.load(home + "model/epoch_16_7.pth", map_location=device)

    def detect_align(self, image, img, priors):
        """
        get a image from ndarray, detect faces in image,
        cropped face and align face
        Args:
            image: original image from cv2(BGR) or PIL(RGB)
            img: tensorized image
            priors: tensorized anchors
        Returns:
            embeddings: tensor
        """
        landmarks = []
        embs = []
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = self.model(img)
        if len(output[2]) > 0:
            landmarks = postprocess(output, img, priors)
        if isinstance(landmarks, torch.Tensor):
            warped =[]
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
            with torch.no_grad():
                embs = l2_norm(self.arcmodel(faces_preprocessing(warped)))
        return embs
