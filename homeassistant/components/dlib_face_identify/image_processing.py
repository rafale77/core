"""Component that will help set the Dlib face detect processing."""
import io
import logging
import numpy as np
import dlib
import cv2

# pylint: disable=import-error
import face_recognition
import voluptuous as vol
from sklearn import svm
from joblib import dump, load
from pathlib import Path
import os

from homeassistant.components.image_processing import (
    CONF_CONFIDENCE,
    CONF_ENTITY_ID,
    CONF_NAME,
    CONF_SOURCE,
    PLATFORM_SCHEMA,
    ImageProcessingFaceEntity,
)
from homeassistant.core import split_entity_id
import homeassistant.helpers.config_validation as cv

_LOGGER = logging.getLogger(__name__)
home = str(Path.home())+"/.homeassistant/"

ATTR_NAME = "name"
CONF_FACES = "faces"

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_FACES): {cv.string: cv.isfile},
        vol.Optional(CONF_CONFIDENCE, default=0.6): vol.Coerce(float),
    }
)

def setup_platform(hass, config, add_entities, discovery_info=None):
    """Set up the Dlib Face detection platform."""
    entities = []
    for camera in config[CONF_SOURCE]:
        entities.append(
            DlibFaceIdentifyEntity(
                camera[CONF_ENTITY_ID],
                config[CONF_FACES],
                camera.get(CONF_NAME),
                config[CONF_CONFIDENCE],
            )
        )

    add_entities(entities)


class DlibFaceIdentifyEntity(ImageProcessingFaceEntity):
    """Dlib Face API entity for identify."""

    def __init__(self, camera_entity, faces, name, model = "dnn"):
        """Initialize Dlib face identify entry."""

        super().__init__()

        self.dnn_face_detector = cv2.dnn.readNetFromCaffe(home+"model/deploy.prototxt.txt", home+"model/res10_300x300_ssd_iter_140000.caffemodel")

        #switch dnn to GPU
        self.dnn_face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.dnn_face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self._camera = camera_entity
        self.fmodel = "large"

        if name:
            self._name = name
        else:
            self._name = f"Dlib Face {split_entity_id(camera_entity)[1]}"

        self.train_faces()

    def locate(self, image):
        def _rect_to_css(rect):
            return rect.top(), rect.right(), rect.bottom(), rect.left()

        def _trim_css_to_bounds(css, image_shape):
            return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

        face_locations = []
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
            # pass the blob through the network and obtain the detections and
            # predictions
        self.dnn_face_detector.setInput(blob)
        detections = self.dnn_face_detector.forward()

        dlibrect = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = dlib.rectangle(startX, startY, endX, endY)
                dlibrect.append(face)
                face_locations = [_trim_css_to_bounds(_rect_to_css(face), image.shape) for face in dlibrect]
            return face_locations

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

            # Loop through each training image for the current person
                for person_img in pix:
            # Get the face encodings for the face in each image file
                    face = cv2.imread(dir + person + "/" + person_img)
                    face_bounding_boxes = self.locate(face)
            # If training image contains exactly one face
                    if len(face_bounding_boxes) == 1:
                        face = cv2.cvtColor(np.array(face),cv2.COLOR_RGB2BGR)
                        face_enc = face_recognition.face_encodings(face, face_bounding_boxes, model=self.fmodel)[0]
                # Add face encoding for current image
                # with corresponding label (name) to the training data
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

        face_locations = self.locate(image)
        found = []
        unknowns =[]
        if face_locations:
            im = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
            unknowns = face_recognition.face_encodings(im, face_locations, model=self.fmodel)
            for unknown_face in unknowns:
                name = self.clf.predict([unknown_face])
                name = np.array2string(name)
                found.append({ATTR_NAME: name})
        self.process_faces(found, len(unknowns))
