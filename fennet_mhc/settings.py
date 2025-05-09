import os

from alphabase.yaml_utils import load_yaml

from fennet_mhc.constants._const import CONST_FOLDER

global_settings = load_yaml(os.path.join(CONST_FOLDER, "default_settings.yaml"))
