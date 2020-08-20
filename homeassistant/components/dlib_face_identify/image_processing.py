"""Component that will help set the Dlib face detect processing."""
import logging
import numpy as np
from easydict import EasyDict as edict
import cv2
import torch

from .Retinaface import FaceDetector
from .Arcface import Backbone

# pylint: disable=import-error
import voluptuous as vol
from pathlib import Path
import os

from homeassistant.components.image_processing import (
    CONF_ENTITY_ID,
    CONF_NAME,
    CONF_SOURCE,
    ImageProcessingFaceEntity,
)
from homeassistant.core import split_entity_id
import homeassistant.helpers.config_validation as cv

_LOGGER = logging.getLogger(__name__)
home = str(Path.home())+"/.homeassistant/"
ATTR_NAME = "name"

def get_config():

    conf = edict()
    conf.model_path = home+'/model/'
    conf.log_path = home
    conf.save_path = home
    conf.input_size = [112, 112]
    conf.embedding_size = 512
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se'  # or 'ir'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.data_mode = 'emore'
    conf.batch_size = 5  # irse net depth 50
    conf.facebank_path = Path(home+'recogface/')
    conf.threshold = 1.5
    conf.face_limit = 10        # when inference, at maximum detect 10 faces in one image
    return conf


def setup_platform(hass, config, add_entities, discovery_info=None):
    """Set up the Dlib Face detection platform."""

    entities = []
    for camera in config[CONF_SOURCE]:
        entities.append(
            DlibFaceIdentifyEntity(
                camera[CONF_ENTITY_ID],
                camera.get(CONF_NAME),
            )
        )
    add_entities(entities)


class DlibFaceIdentifyEntity(ImageProcessingFaceEntity):
    """Dlib Face API entity for identify."""

    def __init__(self, camera_entity, name):
        """Initialize Dlib face identify entry."""

        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.face_detector = FaceDetector(weight_path=home+'model/Resnet50_Final.pth', device=self.device)
        self._camera = camera_entity
        self.conf = get_config()
        self.arcmodel = Backbone(self.conf.net_depth, self.conf.drop_ratio, self.conf.net_mode).to(self.device)
        try:
            self.arcmodel.load_state_dict(torch.load(f'{self.conf.model_path}/model_ir_se50.pth'))
        except IOError as e:
            _LOGGER.warning("Arcface weight does not exist")
        self.arcmodel.eval()
        if name:
            self._name = name
        else:
            self._name = f"Dlib Face {split_entity_id(camera_entity)[1]}"
        self.train_faces()

    def faces_preprocessing(self, faces):

        faces = faces.permute(0, 3, 1, 2).float()
        faces = faces.div(255).to(self.device)
        mu = torch.as_tensor([.5, .5, .5], dtype=faces.dtype, device=self.device)
        faces[:].sub_(mu[:, None, None]).div_(mu[:, None, None])
        return faces

    def train_faces(self):

        try:
            self.targets = torch.load(self.conf.facebank_path/'facebank.pth')
            self.names = np.load(self.conf.facebank_path/'names.npy')
            _LOGGER.warning("Faces Loaded")
        except:
            _LOGGER.warning("Model not trained, retraining...")
            faces = []
            names = ['Unknown']
            dir = home+"recogface/faces/"
            train_dir = os.listdir(dir)
            for person in train_dir:
                pix = os.listdir(dir + person)
                embs = []
                for person_img in pix:
                    pic = cv2.imread(dir + person + "/" + person_img)
                    face = self.face_detector.detect_align(pic)[0]
                    if len(face) == 1:
                        with torch.no_grad():
                            embs.append(self.arcmodel(self.faces_preprocessing(face)))
                    else:
                        _LOGGER.error(person_img+" can't be used for training")
                faces.append(torch.cat(embs).mean(0, keepdim=True))
                names.append(person)
            self.targets = torch.cat(faces)
            torch.save(self.targets, str(self.conf.facebank_path)+'/facebank.pth')
            self.names = np.array(names)
            np.save(str(self.conf.facebank_path)+'/names', self.names)
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
        faces, unknowns, scores, landmarks = self.face_detector.detect_align(image)
        if len(faces)>0:
            embs = self.arcmodel(self.faces_preprocessing(faces))
            diff = embs.unsqueeze(-1) - self.targets.transpose(1, 0).unsqueeze(0)
            dist = torch.sum(torch.pow(diff, 2), dim=1)
            minimum, min_idx = torch.min(dist, dim=1)
            min_idx[minimum > self.conf.threshold] = -1  # if no match, set idx to -1
            for idx, bbox in enumerate(unknowns):
                found.append({ATTR_NAME: self.names[min_idx[idx]+1]})
        self.process_faces(found, len(unknowns))
