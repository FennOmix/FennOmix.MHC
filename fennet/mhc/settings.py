import os

from fennet.mhc.constants._const import CONST_FOLDER
from alphabase.yaml_utils import load_yaml


global_settings = load_yaml(os.path.join(CONST_FOLDER, "default_settings.yaml"))
