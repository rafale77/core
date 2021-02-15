Home Assistant
=================================================================================

Open source home automation that puts local control and privacy first. Powered by a worldwide community of tinkerers and DIY enthusiasts. Perfect to run on a Raspberry Pi or a local server.

Check out `home-assistant.io <https://home-assistant.io>`__ for `a
demo <https://home-assistant.io/demo/>`__, `installation instructions <https://home-assistant.io/getting-started/>`__,
`tutorials <https://home-assistant.io/getting-started/automation/>`__ and `documentation <https://home-assistant.io/docs/>`__.

My own running fork of home-assistant with improved performance
Changes implemented:

Image Processing component got a new "get raw image" function to avoid unnecessary image conversion between image formats for processing.

D-Lib Face Identify component switched from dlib to use a pytorch implementation of retinaface(resnet50) + arcface(resnet101).

opencv object detection component changed to pytorch implementation of ScaledYoloV4.

json encoder/decoder changed to orjson.

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
