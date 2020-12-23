"""Component that will help set the Dlib face detect processing."""
import logging
import os

# pylint: disable=import-error
from pathlib import Path

import cv2
from easydict import EasyDict as edict
import numpy as np
import torch
from torch.cuda.amp import autocast
from torchvision import transforms as trans

from homeassistant.components.image_processing import (
    CONF_ENTITY_ID,
    CONF_NAME,
    CONF_SOURCE,
    ImageProcessingFaceEntity,
)
from homeassistant.core import split_entity_id

from .Retinaface import FaceDetector
from .model import Backbone

_LOGGER = logging.getLogger(__name__)
home = str(Path.home()) + "/.homeassistant/"

ATTR_NAME = "name"


def get_config():
    """Set configuration settings."""
    conf = edict()
    conf.model_path = home + "/model/"
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = "ir_se"  # or "ir"
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.batch_size = 1  # irse net depth 50
    conf.facebank_path = Path(home + "recogface/")
    conf.threshold = 1
    return conf


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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.face_detector = FaceDetector(
            weight_path=home + "model/Resnet50_Final.pth", device=self.device
        )
        self._camera = camera_entity
        self.conf = get_config()
        self.arcmodel = Backbone(
            self.conf.net_depth, self.conf.drop_ratio, self.conf.net_mode
        ).to(self.device)
        try:
            self.arcmodel.load_state_dict(
                torch.load(
                    f"{self.conf.model_path}/model_ir_se50.pth", 
                    map_location=self.device,
                )
            )
        except OSError:
            _LOGGER.warning("Arcface weight does not exist")
        self.arcmodel.eval()
        if name:
            self._name = name
        else:
            self._name = f"Dlib Face {split_entity_id(camera_entity)[1]}"
        self.train_faces()

    def faces_preprocessing(self, faces):
        """Forward."""
        norma = trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        faces = norma(faces.permute(0, 3, 1, 2).div(255))
        return faces

    def train_faces(self):
        """Train and load faces."""
        try:
            self.targets = torch.load(self.conf.facebank_path / "facebank.pth")
            self.names = np.load(self.conf.facebank_path / "names.npy")
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
                    face = self.face_detector.detect_align(pic)[0]
                    if len(face) == 1:
                        with torch.no_grad():
                            embs.append(self.arcmodel(self.faces_preprocessing(face)))
                    else:
                        _LOGGER.error(person_img + " can't be used for training")
                faces.append(torch.cat(embs).mean(0, keepdim=True))
                names.append(person)
            self.targets = torch.cat(faces)
            torch.save(self.targets, str(self.conf.facebank_path) + "/facebank.pth")
            self.names = np.array(names)
            np.save(str(self.conf.facebank_path) + "/names", self.names)
            _LOGGER.warning("Model training completed and saved...")

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
        unknowns = []
        found = []
        faces, unknowns, scores, _ = self.face_detector.detect_align(image)
        if len(scores) > 0:
            with autocast():
                embs = self.arcmodel(self.faces_preprocessing(faces))
            diff = embs.unsqueeze(-1) - self.targets.transpose(1, 0).unsqueeze(0)
            dist = torch.sum(torch.pow(diff, 2), dim=1)
            minimum, min_idx = torch.min(dist, dim=1)
            min_idx[minimum > self.conf.threshold] = -1  # if no match, set idx to -1
            for idx, _ in enumerate(unknowns):
                found.append({ATTR_NAME: self.names[min_idx[idx] + 1]})
        self.process_faces(found, len(unknowns))
