import logging

import esm

from fennet_mhc.pipeline_api import _download_pretrained_models

_download_pretrained_models()
logging.info("Downloading esm2 model ...")
esm.pretrained.esm2_t12_35M_UR50D()
