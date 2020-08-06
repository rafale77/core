"""Support for Cameras with FFmpeg as decoder."""
import asyncio
import logging
import os

import voluptuous as vol
import cv2
from threading import Thread

from homeassistant.components.camera import PLATFORM_SCHEMA, SUPPORT_STREAM, Camera
from homeassistant.const import CONF_NAME
import homeassistant.helpers.config_validation as cv

from . import CONF_EXTRA_ARGUMENTS, CONF_INPUT, DATA_FFMPEG

_LOGGER = logging.getLogger(__name__)

DEFAULT_NAME = "FFmpeg"

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_INPUT): cv.string,
        vol.Optional(CONF_EXTRA_ARGUMENTS): cv.string,
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    }
)

async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    """Set up a FFmpeg camera."""
    async_add_entities([FFmpegCamera(hass, config)])

class Client:
    """ Maintain live RTSP feed without buffering. """
    _stream = None

    def __init__(self, rtsp_server_uri, extra_cmd):
        """
            rtsp_server_uri: the path to an RTSP server. should start with "rtsp://"
        """
        self.rtsp_server_uri = rtsp_server_uri
        self.extra_cmd = extra_cmd
        self.open()
        t = Thread(target=self._update, args=())
        t.daemon = True
        t.start()

    def open(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "hwaccel;cuvid|video_codec;h264_cuvid|vsync;0"
        if self.extra_cmd == "h265":
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "hwaccel;cuvid|video_codec;hevc_cuvid|vsync;0"
        self._stream = cv2.VideoCapture(self.rtsp_server_uri, cv2.CAP_FFMPEG)

    def _update(self):
        while True:
            grabbed = self._stream.grab()
            if not grabbed:
                _LOGGER.warning("Stream Interrupted, Retrying")
                self._stream.release()
                self.open()

    def read(self):
        """ Retrieve most recent frame and decode"""
        (read, frame) = self._stream.retrieve()
        return frame

class FFmpegCamera(Camera):
    """An implementation of an FFmpeg camera."""

    def __init__(self, hass, config):
        """Initialize a FFmpeg camera."""
        super().__init__()

        self._manager = hass.data[DATA_FFMPEG]
        self._name = config.get(CONF_NAME)
        self._input = config.get(CONF_INPUT)
        self._extra_arguments = config.get(CONF_EXTRA_ARGUMENTS)
        self.client = Client(rtsp_server_uri = self._input, extra_cmd=self._extra_arguments)

    @property
    def supported_features(self):
        """Return supported features."""
        return SUPPORT_STREAM

    async def stream_source(self):
        """Return the stream source."""
        return self._input.split(" ")[-1]

    async def async_camera_raw_image(self):
        """Return a still image response from the camera."""
        frame = self.client.read()
        return frame

    async def async_camera_image(self):
        """Return a still image response from the camera."""
        frame = self.client.read()
        ret, image = cv2.imencode('.jpg', frame)
        return image.tobytes()

    async def handle_async_mjpeg_stream(self, request):
        """Generate an HTTP MJPEG stream from the camera."""
        return await super().handle_async_mjpeg_stream(request)

    @property
    def name(self):
        """Return the name of this camera."""
        return self._name
