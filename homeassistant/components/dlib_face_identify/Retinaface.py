from collections import OrderedDict
import logging
from pathlib import Path

import cv2
from skimage.transform import SimilarityTransform
import numpy as np
import torch
from torch.cuda.amp import autocast

from .model import RetinaFace

# flake8: noqa


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


class FaceDetector:
    def __init__(self):
        """RetinaFace Detector with 5points landmarks."""

        self.thresh = 0.99
        self.top_k = 5000
        self.nms_thresh = 0.4
        self.keep_top_k = 750
        self.ref_pts = get_reference_facial_points()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = RetinaFace().to(self.device)
        state_dict = torch.load(
            home + "model/Resnet50_Final.pth", map_location=self.device
        )
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove "module".
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        #self.model = torch.jit.script(self.model)

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

        # tic = time.time()
        with torch.no_grad():
            with autocast():
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
