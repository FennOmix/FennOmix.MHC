import os

from alphabase.yaml_utils import load_yaml

CONST_FOLDER = os.path.dirname(__file__)

global_settings = load_yaml(os.path.join(CONST_FOLDER, "default_settings.yaml"))

FENNETMHC_HOME = os.path.expanduser(global_settings["FENNETMHC_HOME"])

FENNETMHC_MODEL_DIR = os.path.join(FENNETMHC_HOME, "foundation_model")

HLA_MODEL_PATH = os.path.join(FENNETMHC_MODEL_DIR, global_settings["hla_model"])

PEPTIDE_MODEL_PATH = os.path.join(FENNETMHC_MODEL_DIR, global_settings["peptide_model"])

HLA_EMBEDDING_PATH = os.path.join(FENNETMHC_MODEL_DIR, global_settings["hla_embedding"])
