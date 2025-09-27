import os

from alphabase.yaml_utils import load_yaml

CONST_FOLDER = os.path.dirname(__file__)

global_settings = load_yaml(os.path.join(CONST_FOLDER, "global_settings.yaml"))

FENNOMIXMHC_HOME = os.path.expanduser(global_settings["FENNOMIXMHC_HOME"])

FENNOMIXMHC_MODEL_DIR = os.path.join(FENNOMIXMHC_HOME, "foundation_model")

MHC_MODEL_KEY = "mhc_model"
PEPTIDE_MODEL_KEY = "peptide_model"
MHC_EMBEDDING_KEY = "mhc_embedding"
MHC_MODEL_PATH = os.path.join(FENNOMIXMHC_MODEL_DIR, global_settings[MHC_MODEL_KEY])
PEPTIDE_MODEL_PATH = os.path.join(
    FENNOMIXMHC_MODEL_DIR, global_settings[PEPTIDE_MODEL_KEY]
)
MHC_EMBEDDING_PATH = os.path.join(
    FENNOMIXMHC_MODEL_DIR, global_settings[MHC_EMBEDDING_KEY]
)

MHC_MODEL_PSEUDO_KEY = "mhc_model_pseudo"
PEPTIDE_MODEL_PSEUDO_KEY = "peptide_model_pseudo"
MHC_EMBEDDING_PSEUDO_KEY = "mhc_embedding_pseudo"
MHC_MODEL_PSEUDO_PATH = os.path.join(
    FENNOMIXMHC_MODEL_DIR, global_settings[MHC_MODEL_PSEUDO_KEY]
)
PEPTIDE_MODEL_PSEUDO_PATH = os.path.join(
    FENNOMIXMHC_MODEL_DIR, global_settings[PEPTIDE_MODEL_PSEUDO_KEY]
)
MHC_EMBEDDING_PSEUDO_PATH = os.path.join(
    FENNOMIXMHC_MODEL_DIR, global_settings[MHC_EMBEDDING_PSEUDO_KEY]
)

BACKGROUND_FASTA_PATH = os.path.join(
    FENNOMIXMHC_MODEL_DIR, global_settings["background_fasta"]
)

PEPTIDE_DF_FOR_MHC_TSV = "peptide_df_for_MHC.tsv"
MHC_DF_FOR_EPITOPES_TSV = "MHC_df_for_epitopes.tsv"
PEPTIDE_DECONVOLUTION_CLUSTER_DF_TSV = "peptide_deconvolution_cluster_df.tsv"
PEPTIDES_FOR_MHC_FASTA = "peptides_for_MHC.fasta"

D_MODEL = 480
