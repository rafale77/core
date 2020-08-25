"""Component that will process object detection with opencv."""
import logging
import cv2
import numpy as np
import torch
import torchvision
# pylint: disable=import-error
import voluptuous as vol
from pathlib import Path
import os
import sys


from homeassistant.components.image_processing import (
    CONF_ENTITY_ID,
    CONF_CONFIDENCE,
    CONF_NAME,
    CONF_SOURCE,
    PLATFORM_SCHEMA,
    ImageProcessingEntity,
)

from homeassistant.core import split_entity_id
import homeassistant.helpers.config_validation as cv

_LOGGER = logging.getLogger(__name__)
home = str(Path.home())+"/.homeassistant/model/"

CONF_CLASSIFIER = "classifier"
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
sys.path.insert(0, str(Path.home())+'/.local/lib/python3.7/site-packages/homeassistant/components/opencv/')

with open(home+'cococlasses.txt', "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

ATTR_MATCHES = "matches"
ATTR_TOTAL_MATCHES = "total_matches"

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_CLASSIFIER, default="person"): cv.string,
        vol.Optional(CONF_CONFIDENCE, default=0.6): vol.Coerce(float),
    }
)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
    return output

def preprocessor(img_raw, w, h, device):
    img_raw = cv2.resize(img_raw,(w,h))
    #img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    #img = torch.tensor(img_raw, dtype=torch.float32).to(self.device)
    img = torch.tensor(img_raw, dtype=torch.float16).div(255).to(device)
    #img = torch.from_numpy(img_raw).half().to(device)
    #img = torch.true_divide(img, 255)
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img

def setup_platform(hass, config, add_entities, discovery_info=None):
    """Set up the opencv object detection platform."""
    entities = []
    for camera in config[CONF_SOURCE]:
        entities.append(
            OpenCVImageProcessor(
                hass,
                camera[CONF_ENTITY_ID],
                camera.get(CONF_NAME),
                config[CONF_CONFIDENCE],
                config[CONF_CLASSIFIER],
            )
        )
    add_entities(entities)


class OpenCVImageProcessor(ImageProcessingEntity):
    """OpenCV Object API entity for identify."""

    def __init__(self, hass, camera_entity, name, confidence, classifiers):
        """Initialize the OpenCV entity."""

        super().__init__()

        self.mode = "5"

        if self.mode == "5":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#            self.model = torch.load(home+'yolov5l.pt', self.device)['model'].fuse().eval().half()
            self.model = torch.load(home+'yolov4x-mish.pt', self.device)['model'].fuse().eval().half()

        else:
            self.net = cv2.dnn.readNet(home+'yolov4.weights', home+'yolov4.cfg')
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            #self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            self.model = cv2.dnn_DetectionModel(self.net)
            self.model.setInputParams(size=(608, 608), scale=1/256)

        self.hass = hass
        self._camera = camera_entity
        if name:
            self._name = name
        else:
            self._name = f"OpenCV {split_entity_id(camera_entity)[1]}"
        self._confidence = confidence
        self._classifiers = classifiers.split(',')
        self._matches = []
        self._total_matches = 0

    @property
    def camera_entity(self):
        """Return camera entity id from process pictures."""
        return self._camera

    @property
    def name(self):
        """Return the name of the entity."""
        return self._name

    @property
    def state(self):
        """Return the state of the entity."""
        return self._total_matches

    @property
    def state_attributes(self):
        """Return device specific state attributes."""
        return {ATTR_MATCHES: self._matches, ATTR_TOTAL_MATCHES: self._total_matches}

    def process_image(self, image):
        """Process image."""

        if self.mode == "5":
            imgsz = int(672)
            img = preprocessor(image, imgsz, imgsz, self.device)
            with torch.no_grad():
                pred = self.model(img)[0]
            pred = non_max_suppression(pred, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            self._matches = []
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    for c in det[:, -1].unique():
                        if class_names[int(c)] in self._classifiers:
                            n = (det[:, -1] == c).sum()  # detections per class
                            label= "%g %ss, " % (n, class_names[int(c)])
                            self._matches.append(label)
        else:
            classes, scores, boxes = self.model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            self._matches = []
            for (classid, score, box) in zip(classes, scores, boxes):
                if score >= self._confidence:
                    if class_names[classid[0]] in self._classifiers:
                        label = "%s : %.2f" % (class_names[classid[0]], score * 100)
                        self._matches.append(label)

        self._total_matches = len(self._matches)
