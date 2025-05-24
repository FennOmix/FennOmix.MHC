import os

from alphabase.yaml_utils import load_yaml

CONST_FOLDER = os.path.dirname(__file__)

global_settings = load_yaml(os.path.join(CONST_FOLDER, "global_settings.yaml"))

FENNETMHC_HOME = os.path.expanduser(global_settings["FENNETMHC_HOME"])

FENNETMHC_MODEL_DIR = os.path.join(FENNETMHC_HOME, "foundation_model")

HLA_MODEL_PATH = os.path.join(FENNETMHC_MODEL_DIR, global_settings["hla_model"])

PEPTIDE_MODEL_PATH = os.path.join(FENNETMHC_MODEL_DIR, global_settings["peptide_model"])

HLA_EMBEDDING_PATH = os.path.join(FENNETMHC_MODEL_DIR, global_settings["hla_embedding"])

BACKGROUND_FASTA_PATH = os.path.join(
    FENNETMHC_MODEL_DIR, global_settings["background_fasta"]
)

PEPTIDE_DF_FOR_MHC_TSV = "peptide_df_for_MHC.tsv"
MHC_DF_FOR_EPITOPES_TSV = "MHC_df_for_epitopes.tsv"
PEPTIDE_DECONVOLUTION_CLUSTER_DF_TSV = "peptide_deconvolution_cluster_df.tsv"
PEPTIDES_FOR_MHC_FASTA = "peptides_for_MHC.fasta"
