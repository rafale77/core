"""Component that will help set the Dlib face detect processing."""
import logging
import numpy as np
from easydict import EasyDict as edict
#import dlib
import cv2
import torch

from .Retinaface import FaceDetector
from .Arcface import Backbone

# pylint: disable=import-error
import voluptuous as vol
#from sklearn import svm
#from joblib import dump, load
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
    conf.face_limit = 10        # when inference, at maximum detect 10 faces in one image, my laptop is slow
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
        self.model = "arcface"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.face_detector = FaceDetector(weight_path=home+'model/Resnet50_Final.pth', device=self.device)
        self._camera = camera_entity
        if self.model == "arcface":
            self.conf = get_config()
            self.arcmodel = Backbone(self.conf.net_depth, self.conf.drop_ratio, self.conf.net_mode).to(self.device)
            try:
                self.arcmodel.load_state_dict(torch.load(f'{self.conf.model_path}/model_ir_se50.pth'))
            except IOError as e:
                _LOGGER.warning("Arcface weight does not exist")
            self.arcmodel.eval()
        else:
            self.face_encoder = dlib.face_recognition_model_v1(home+"model/dlib_face_recognition_resnet_model_v1.dat")
            self.fmodel = "large"
        if name:
            self._name = name
        else:
            self._name = f"Dlib Face {split_entity_id(camera_entity)[1]}"

        self.train_faces()

    def _raw_face_landmarks(self, face_image, face_locations, model="large"):

        face_locations = [dlib.rectangle(face_location[3], face_location[0], face_location[1], face_location[2]) for face_location in face_locations]
        if model == "small":
            pose_predictor = dlib.shape_predictor(home+"model/shape_predictor_5_face_landmarks.dat")
        else:
            pose_predictor = dlib.shape_predictor(home+"model/shape_predictor_68_face_landmarks.dat")
        return [pose_predictor(face_image, face_location) for face_location in face_locations]


    def face_encodings(self, face_image, known_face_locations, num_jitters=1, model="large"):
        """
        Given an image, return the 128-dimension face encoding for each face in the image.

        :param face_image: The image that contains one or more faces
        :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
        :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
        :param model: Optional - which model to use. "large" (default) or "small" which only returns 5 points but is faster.
        :return: A list of 128-dimensional face encodings (one for each face in the image)
        """
        raw_landmarks = self._raw_face_landmarks(face_image, known_face_locations, model)
        return [np.array(self.face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


    def faces_preprocessing(self, faces):

        faces = faces.permute(0, 3, 1, 2).float()
        faces = faces.div(255).to(self.device)
        mu = torch.as_tensor([.5, .5, .5], dtype=faces.dtype, device=self.device)
        faces[:].sub_(mu[:, None, None]).div_(mu[:, None, None])
        return faces


    def locate(self, image):

        dlibrect = []
        (h, w) = image.shape[:2]
        ratio = 2
        image = cv2.resize(image, (int(w/ratio), int(h/ratio)))
        boxes, scores, landmarks = self.face_detector.detect_faces(image)
        if len(boxes) > 0:
            for box in boxes:
                face = dlib.rectangle(int(box[0]*ratio), int(box[1]*ratio), int(box[2]*ratio), int(box[3])*ratio)
                dlibrect.append(face)

        return [(max(face.top(), 0), min(face.right(), w), min(face.bottom(), h), max(face.left(), 0)) for face in dlibrect]


    def train_faces(self):

            try:
                if self.model == "arcface":
                    self.targets = torch.load(self.conf.facebank_path/'facebank.pth')
                    self.names = np.load(self.conf.facebank_path/'names.npy')
                else:
                    self.clf = load(home+'model.joblib')
                _LOGGER.warning("Faces Loaded")
            except:
                _LOGGER.warning("Model not trained, retraining...")
                faces = []
                names = []
                faces_embs = torch.empty(0).to(self.device)
                dir = home+"recogface/faces/"
                train_dir = os.listdir(dir)

                for person in train_dir:
                    pix = os.listdir(dir + person)
                    embs = []
                    for person_img in pix:
                        pic = cv2.imread(dir + person + "/" + person_img)
                        if self.model == "arcface":
                            face = self.face_detector.detect_align(pic)[0]
                            with torch.no_grad():
                                face = self.faces_preprocessing(face)
                                embs.append(self.arcmodel(face))
                        else:
                            boxes = self.locate(pic)
                            if len(boxes) == 1:
                                face = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
                                face_enc = self.face_encodings(face, boxes, 100, model=self.fmodel)[0]
                                faces.append(face_enc)
                                names.append(person)
                            else:
                                _LOGGER.error(person_img+" can't be used for training", person, person_img)
                    if self.model == "arcface":
                        faces_emb = torch.cat(embs).mean(0, keepdim=True)
                        faces.append(faces_emb)
                        names = np.append(names, person)
                if self.model == "arcface":
                    self.targets = torch.cat(faces)
                    torch.save(self.targets, str(self.conf.facebank_path)+'/facebank.pth')
                    self.names = names
                    np.save(str(self.conf.facebank_path)+'/names', names)
                else:
                # Create and train the SVC classifier
                    self.clf = svm.SVC(gamma ='scale')
                    self.clf.fit(faces, names)
                    dump(self.clf, home+'model.joblib')
            _LOGGER.warning("Model training done and saved...")

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
        if self.model == "arcface":
            faces, unknowns, scores, landmarks = self.face_detector.detect_align(image)
            if len(faces)>0:
                face = self.faces_preprocessing(faces)
                embs = self.arcmodel(face)
                diff = embs.unsqueeze(-1) - self.targets.transpose(1, 0).unsqueeze(0)
                dist = torch.sum(torch.pow(diff, 2), dim=1)
                minimum, min_idx = torch.min(dist, dim=1)
                min_idx[minimum > self.conf.threshold] = -1  # if no match, set idx to -1

                for idx, bbox in enumerate(unknowns):
                    found.append({ATTR_NAME: self.names[min_idx[idx]]})
        else:
            face_locations = self.locate(image)
            if face_locations:
                im = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                unknowns = self.face_encodings(im, face_locations, 10, model=self.fmodel)
                for unknown_face in unknowns:
                    name = self.clf.predict([unknown_face])
                    name = np.array2string(name)
                    found.append({ATTR_NAME: name})
        self.process_faces(found, len(unknowns))
