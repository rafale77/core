Home Assistant
=================================================================================

Open source home automation that puts local control and privacy first. Powered by a worldwide community of tinkerers and DIY enthusiasts. Perfect to run on a Raspberry Pi or a local server.

Check out `home-assistant.io <https://home-assistant.io>`__ for `a
demo <https://home-assistant.io/demo/>`__, `installation instructions <https://home-assistant.io/getting-started/>`__,
`tutorials <https://home-assistant.io/getting-started/automation/>`__ and `documentation <https://home-assistant.io/docs/>`__.

My own running fork of home-assistant with improved performance
Changes implemented:

1. Image Processing component got a new "get raw image" function to avoid unnecessary image conversion between image formats for processing.
2. Camera component backend moved from ffmpeg to much more efficient opencv
3. DLib Face Identify component switched from dlib to use a pytorch implementation of retinaface(resnet50) + arcface(resnet101).
4. opencv object detection component changed to pytorch implementation of ScaledYoloV4.
5. json encoder/decoder changed to orjson.
6. Added on/off switching capability to image processing component.


Installation instructions:

1. nVidia GPU driver, cuda and cudnn installation: Follow the nvidia instructions to install the drivers and library for your platform.
*driver found `here <https://www.nvidia.com/Download/index.aspx?lang=en-us>`__
*cuda library found `here <https://developer.nvidia.com/cuda-downloads?target_os=Linux>`__
*cudnn library which requires signing up to an nvidia dev account found `here <https://developer.nvidia.com/cudnn>`__
 

2. Install ffmpeg with GPU acceleration buy building from source. Instructions `here <https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/index.html>`__

3. Install opencv with GPU acceleration by compiling it from source. I found that the latest version, 4.5.2 had some breaking changes to the video handling so I recommend sticking with 4.5.1

.. code-block:: bash

 git clone --branch 4.5.1 https://github.com/opencv/opencv.git
 git clone --branch 4.5.1 https://github.com/opencv/opencv_contrib.git
 cd opencv
 mkdir build && cd build
 cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D OPENCV_ENABLE_NONFREE=ON -D WITH_CUDA=ON -D WITH_CUDNN=ON -D WITH_CAFFE=ON -D WITH_NVCUVID=ON -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D CUDA_ARCH_BIN=8.6 -D WITH_CUBLAS=ON -D OPENCV_EXTRA_MODULES_PATH=~/source/opencv_contrib/modules-D HAVE_opencv_python3=ON -D PYTHON_EXECUTABLE=/usr/bin/python3 -D BUILD_NEW_PYTHON_SUPPORT=ON -D CMAKE_CUDA_FLAGS=-lineinfo --use_fast_math -rdc=true -lcudadevrt -D BUILD_EXAMPLES=OFF ..
 make
 sudo make install

Note that I am using the cuda architecture 8.6 which corresponds to RTX30xx GPUs. make sure that you use the correct one for your GPU.

4. Install pytorch for your platform and cuda version following this `page <https://pytorch.org/get-started/locally/>`__

5. Download pretrained models:
   create a folder called "model" under your ".homeassistant/" configuration folder.
.. code-block:: bash

 mkdir ~/.homeassistant/model
   
For object detections using enhanced yolov4 see `here <https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-large>`__ by default the repo uses "yolov4-p5_.pt" 
For face detection, download the file resnet50 model from this retinaface repo: https://github.com/biubug6/Pytorch_Retinaface
The file can be obtained `here <https://drive.google.com/file/d/1wyvxIvjH1Xxvc4Qa4tvgV8ibWro1SM35/view?usp=sharing>`__
For facial recognition, download the IR-100 file from this repo: https://github.com/cavalleria/cavaface.pytorch/blob/master/docs/MODEL_ZOO.md
The file is contained `here <https://drive.google.com/file/d/1xp1IqsiArqf0XEqc7O5aq8KMhrvw3DbE/view?usp=sharing>`__
Upload these files to the model folder set in the previous step
 
6. Face database:
    create a face databade folder under your ".homeassistant/" configuration folder.
.. code-block:: bash

 mkdir ~/.homeassistant/recogface
    
create folders for each of the faces you want recognized i.e mkdir ~/.homeassistant/recogface/me and upload face pictures (I recommend at least dozen) for each of the people in their corresponding folder.
    
7. Configure homeassistant:
In your configuration.yaml file first setup your camera streams using the ffmpeg component which again has been modified to use opencv. Note the use of the extra argument, cuda for h264 decoding and hevc for h265 decoding. i.e.
 
.. code-block:: bash

 camera
  - platform: ffmpeg
    input: rtsp://user:pwd@ip:port/cam/realmonitor?channel=1&subtype=0
    name: Porch
    extra_arguments: cuda
  - platform: ffmpeg
    input: rtsp://user:pwd@ip:port/cam/realmonitor?channel=1&subtype=0
    name: kitchen
    extra_arguments: hevc
 
then setup the image processing components like you would for dlib and opencv i.e.
 
.. code-block:: bash

 image_processing:
  - platform: dlib_face_identify
    scan_interval: 0.5
    source:
    - entity_id: camera.doorbell
      name: Doorbell
  - platform: opencv
    confidence: 0.8
    scan_interval: 0.5
    source:
      - entity_id: camera.pelouse
        name: Pelouse

 
|screenshot-states|

Featured integrations
---------------------

|screenshot-components|

The system is built using a modular approach so support for other devices or actions can be implemented easily. See also the `section on architecture <https://developers.home-assistant.io/docs/architecture_index/>`__ and the `section on creating your own
components <https://developers.home-assistant.io/docs/creating_component_index/>`__.

If you run into issues while using Home Assistant or during development
of a component, check the `Home Assistant help section <https://home-assistant.io/help/>`__ of our website for further help and information.

.. |Chat Status| image:: https://img.shields.io/discord/330944238910963714.svg
   :target: https://discord.gg/c5DvZ4e
.. |screenshot-states| image:: https://raw.github.com/home-assistant/home-assistant/master/docs/screenshots.png
   :target: https://home-assistant.io/demo/
.. |screenshot-components| image:: https://raw.github.com/home-assistant/home-assistant/dev/docs/screenshot-components.png
   :target: https://home-assistant.io/integrations/
