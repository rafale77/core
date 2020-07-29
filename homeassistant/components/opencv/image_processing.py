"""Component that will process object detection with opencv."""
import io
import logging
import numpy as np
import cv2
# pylint: disable=import-error

import voluptuous as vol

from pathlib import Path
import os

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

class_names = []
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
        self.net = cv2.dnn.readNet(home+'yolov4.weights', home+'yolov4.cfg')
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(608, 608), scale=1/256)

        super().__init__()
        self.hass = hass
        self._camera = camera_entity
        if name:
            self._name = name
        else:
            self._name = f"OpenCV {split_entity_id(camera_entity)[1]}"
        self._confidence = confidence
        self._classifiers = classifiers.split(',')
        self._matches = {}
        self._total_matches = 0
        self._last_image = None

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

        classes, scores, boxes = self.model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        matches = []
        total_matches = 0
        for (classid, score, box) in zip(classes, scores, boxes):
            if score >= self._confidence:
                if class_names[classid[0]] in self._classifiers:
                    label = "%s : %.2f" % (class_names[classid[0]], score * 100)
                    matches.append(label)

        self._matches = matches
        self._total_matches = len(matches)
