"""Component that will help set the Dlib face detect processing."""
from itertools import product
import logging
from math import ceil
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from homeassistant.components.image_processing import (
    CONF_ENTITY_ID,
    CONF_NAME,
    CONF_SOURCE,
    ImageProcessingFaceEntity,
)
from homeassistant.const import ATTR_NAME
from homeassistant.core import split_entity_id

from .models import FaceDetector

_LOGGER = logging.getLogger(__name__)
home = str(Path.home()) + "/.homeassistant/"
ATTR_NAME = "name"
ATTR_FACES = "faces"
ATTR_TOTAL_FACES = "total_faces"
ATTR_MOTION = "detection"
# torch.backends.cudnn.benchmark = True


async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    """Set up the Dlib Face detection platform."""

    entities = []
    for camera in config[CONF_SOURCE]:
        entities.append(
            DlibFaceIdentifyEntity(
                camera[CONF_ENTITY_ID],
                camera.get(CONF_NAME),
            )
        )
    async_add_entities(entities)


class DlibFaceIdentifyEntity(ImageProcessingFaceEntity):
    """Dlib Face API entity for identify."""

    def __init__(self, camera_entity, name):
        """Initialize Dlib face identify entry."""

        super().__init__()
        self.facebank_path = Path(home + "recogface/")
        self.threshold = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.face_detector = FaceDetector()
        self._camera = camera_entity
        if name:
            self._name = name
        else:
            self._name = f"Dlib Face {split_entity_id(camera_entity)[1]}"
        self.train_faces()
        self.priors = []
        self._det = "on"

    def enable_detection(self):
        """Enable detection."""
        self._det = "on"

    def disable_detection(self):
        """Disable detection."""
        self._det = "off"

    def prior_box(self, image_size):
        """Boxes for Face."""
        steps = [8, 16, 32]
        feature_maps = [
            [ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps
        ]
        min_sizes = [[16, 32], [64, 128], [256, 512]]
        anchors = []

        for k, f in enumerate(feature_maps):
            min_size = min_sizes[k]
            mat = np.array(list(product(range(f[0]), range(f[1]), min_size))).astype(np.float32)
            mat[:, 0], mat[:, 1] = ((mat[:, 1] + 0.5) * steps[k] / image_size[1],
                                    (mat[:, 0] + 0.5) * steps[k] / image_size[0])
            mat = np.concatenate([mat, mat[:, 2:3]], axis=1)
            mat[:, 2] = mat[:, 2] / image_size[1]
            mat[:, 3] = mat[:, 3] / image_size[0]
            anchors.append(mat)
        output = np.concatenate(anchors, axis=0)
        return torch.as_tensor(output, device=self.device)

    def preprocessor(self, img_raw):
        """Convert cv2/PIL image to tensor."""
        # img_raw = np.float32(img_raw)
        # img_raw -= [104.0, 117.0, 123.0]
        # img = img_raw.transpose(2,0,1)
        # return img[np.newaxis, ...].astype('float32')
        img = torch.as_tensor(img_raw, dtype=torch.float32, device=self.device)
        img -= torch.as_tensor([104, 117, 123], device=self.device)  # BGR
        return img.permute(2, 0, 1).unsqueeze(0)

    def train_faces(self):
        """Train and load faces."""
        try:
            self.targets = torch.load(self.facebank_path / "facebank.pth")
            # self.targets = np.load(self.facebank_path / "facebank.npy")
            self.names = np.load(self.facebank_path / "names.npy")
            _LOGGER.warning("Faces Loaded")
        except Exception:
            _LOGGER.warning("Model not trained, retraining...")
            faces = []
            names = ["Unknown"]
            folder = home + "recogface/faces/"
            train_dir = os.listdir(folder)
            for person in train_dir:
                pix = os.listdir(folder + person)
                embs = []
                for person_img in pix:
                    pic = cv2.imread(folder + person + "/" + person_img)
                    img = self.preprocessor(pic)
                    priors = self.prior_box(img.shape[2:])
                    emb = self.face_detector.detect_align(pic, img, priors)[0]
                    if len(emb) == 1:
                        embs.append(emb)
                    else:
                        _LOGGER.error(person_img + " can't be used for training")
                # faces.append(embs)
                faces.append(torch.cat(embs).mean(0, keepdim=True))
                names.append(person)
            self.targets = torch.cat(faces)
            torch.save(self.targets, str(self.facebank_path) + "/facebank.pth")
            # np.save(str(self.facebank_path) + "/facebank.pth", self.targets)
            self.names = np.array(names)
            np.save(str(self.facebank_path) + "/names", self.names)
            _LOGGER.warning("Model training completed and saved...")

    @property
    def state_attributes(self):
        """Return device specific state attributes."""
        return {
            ATTR_FACES: self.faces,
            ATTR_TOTAL_FACES: self.total_faces,
            ATTR_MOTION: self._det,
        }

    @property
    def camera_entity(self):
        """Return camera entity id from process pictures."""
        return self._camera

    @property
    def name(self):
        """Return the name of the entity."""
        return self._name

    def process_image(self, image):
        """Process image."""
        found = embs = []
        if self._det == "on":
            img = self.preprocessor(image)
            if len(self.priors) < 1:
                self.priors = self.prior_box(img.shape[2:])
            embs = self.face_detector.detect_align(image, img, self.priors)
            if len(embs) > 0:
                diff = embs.unsqueeze(-1) - self.targets.transpose(1, 0).unsqueeze(0)
                dist = torch.sum(torch.pow(diff, 2), dim=1)
                minimum, min_idx = torch.min(dist, dim=1)
                min_idx[minimum > self.threshold] = -1  # if no match
                for idx, _ in enumerate(embs):
                    found.append({ATTR_NAME: self.names[min_idx[idx] + 1]})
        self.process_faces(found, len(embs))
