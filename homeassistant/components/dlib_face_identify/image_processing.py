"""Component that will help set the Dlib face detect processing."""
import logging
import numpy as np
import dlib
import cv2
from .Retinaface import FaceDetector

# pylint: disable=import-error
import voluptuous as vol
from sklearn import svm
from joblib import dump, load
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
CONF_MODEL = "retina"

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

        self.face_encoder = dlib.face_recognition_model_v1(home+"model/dlib_face_recognition_resnet_model_v1.dat")
        self.fmodel = "large"
        self._camera = camera_entity
        if dlib.DLIB_USE_CUDA:
            self.face_detector = FaceDetector(weight_path=home+'model/Resnet50_Final.pth', device='cuda', face_size=(224, 224))
        else:
            self.face_detector = FaceDetector(weight_path=home+'model/Resnet50_Final.pth', device='cpu', face_size=(112, 112))
            self.fmodel = "small"

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

    def locate(self, image, conf, mode="inf"):

        dlibrect = []
        (h, w) = image.shape[:2]
        if mode == "inf":
            image = cv2.resize(image, (int(h/2), int(w/2)))
        boxes, scores, landmarks = self.face_detector.detect_faces(image)
        if len(boxes) > 0:
            for box in boxes:
                face = dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                dlibrect.append(face)

        return [(max(face.top(), 0), min(face.right(), w), min(face.bottom(), h), max(face.left(), 0)) for face in dlibrect]

    def train_faces(self):

        try:
            self.clf = load(home+'model.joblib')
        except:
            _LOGGER.warning("Model not trained, retraining...")
            faces = []
            names = []
            dir = home+"recogface/faces/"
            train_dir = os.listdir(dir)

            for person in train_dir:
                pix = os.listdir(dir + person)

                for person_img in pix:
                    pic = cv2.imread(dir + person + "/" + person_img)
                    boxes = self.locate(pic, 0.9, "retina", "train")
                    if len(boxes) == 1:
                        face = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
                        face_enc = self.face_encodings(face, boxes, 100, model=self.fmodel)[0]
                        faces.append(face_enc)
                        names.append(person)
                    else:
                        _LOGGER.error("%s/%s can't be used for training", person, person_img)
        # Create and train the SVC classifier
            self.clf = svm.SVC(gamma ='scale')
            self.clf.fit(faces, names)
            dump(self.clf, home+'model.joblib')

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

        face_locations = self.locate(image, 0.7)
        found = []
        unknowns =[]
        if face_locations:
            im = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            unknowns = self.face_encodings(im, face_locations, 10, model=self.fmodel)
            for unknown_face in unknowns:
                name = self.clf.predict([unknown_face])
                name = np.array2string(name)
                found.append({ATTR_NAME: name})
        self.process_faces(found, len(unknowns))
