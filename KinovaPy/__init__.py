import os

from .interface import KinovaHardwareInterface

# Kuka robot description paths
ASSETS_PATH = os.path.join(os.path.dirname(__file__), "assets")
MESHES_PATH = os.path.join(ASSETS_PATH, "meshes")
URDF_PATH = os.path.join(ASSETS_PATH, "urdf")
XML_PATH = os.path.join(ASSETS_PATH, "xml")
SCENE_PATH = os.path.join(ASSETS_PATH, "scenes")