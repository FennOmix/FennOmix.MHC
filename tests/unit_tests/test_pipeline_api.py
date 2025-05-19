import os

import numpy as np
import pandas as pd

from fennet_mhc.constants._const import (
    MHC_DF_FOR_EPITOPES_TSV,
    PEPTIDE_DF_FOR_MHC_TSV,
    PEPTIDES_FOR_MHC_FASTA,
)
from fennet_mhc.pipeline_api import (
    PretrainedModels,
    predict_epitopes_for_mhc,
    predict_mhc_binders_for_epitopes,
)

TEST_PEPTIDE_TSV = os.path.abspath("./test_data/test_peptides.tsv")
TEST_MHC_FASTA = os.path.abspath("./test_data/test_MHC_proteins.fasta")
TEST_PEPTIDE_FASTA = os.path.abspath("./test_data/test_peptides.fasta")
OUT_DIR = os.path.abspath("./nogit")


def test_pretrained_models():
    pretrained_models = PretrainedModels(device="cuda")

    assert len(pretrained_models.background_protein_df) == 4357

    protein_df, hla_embeds = pretrained_models.embed_proteins(TEST_MHC_FASTA)
    assert len(protein_df) > 0
    assert hla_embeds.shape[0] == len(protein_df)
    assert hla_embeds.shape[1] == pretrained_models.esm2_model.embed_dim

    peptide_list, pept_embeds = pretrained_models.embed_peptides_from_fasta(
        TEST_PEPTIDE_FASTA,
        min_peptide_length=8,
        max_peptide_length=12,
    )
    assert len(peptide_list) > 0
    assert pept_embeds.shape[0] == len(peptide_list)
    assert pept_embeds.shape[1] == pretrained_models.esm2_model.embed_dim

    peptide_list, pept_embeds = pretrained_models.embed_peptides_tsv(
        TEST_PEPTIDE_TSV,
        min_peptide_length=8,
        max_peptide_length=1000,
    )
    assert len(peptide_list) > 0
    assert pept_embeds.shape[0] == len(peptide_list)
    assert pept_embeds.shape[1] == pretrained_models.esm2_model.embed_dim

    peptide_df = pd.read_csv(TEST_PEPTIDE_TSV, sep="\t")

    _idx = pretrained_models.hla_df.query("allele=='A02_01'").index[0]
    A0201_embed = pretrained_models.hla_embeddings[_idx]
    peptide_df["dist"] = np.linalg.norm(pept_embeds - A0201_embed, axis=1)

    peptide_df["TP"] = (peptide_df["dist"] < 0.4) & (peptide_df["Rank_A0201"] < 2)
    peptide_df["TN"] = (peptide_df["dist"] > 0.4) & (peptide_df["Rank_A0201"] > 2)
    agrees = (peptide_df["TP"] | peptide_df["TN"]).astype(float)
    assert agrees.mean() >= 0.95


def test_predict_peptide_binders_for_MHC_fasta():
    predict_epitopes_for_mhc(
        TEST_PEPTIDE_TSV,
        ["A02_01"],
        OUT_DIR,
        out_fasta_format=True,
        min_peptide_length=8,
        max_peptide_length=12,
        outlier_distance=0.4,
        hla_file_path=None,
        device="cuda",
    )
    assert os.path.exists(f"{OUT_DIR}/{PEPTIDES_FOR_MHC_FASTA}")
    assert os.path.getsize(f"{OUT_DIR}/{PEPTIDES_FOR_MHC_FASTA}") > 0


def test_predict_peptide_binders_for_MHC():
    predict_epitopes_for_mhc(
        TEST_PEPTIDE_TSV,
        ["A02_01"],
        OUT_DIR,
        out_fasta_format=False,
        min_peptide_length=8,
        max_peptide_length=12,
        outlier_distance=0.4,
        hla_file_path=None,
        device="cuda",
    )
    assert os.path.exists(f"{OUT_DIR}/{PEPTIDE_DF_FOR_MHC_TSV}")
    assert os.path.getsize(f"{OUT_DIR}/{PEPTIDE_DF_FOR_MHC_TSV}") > 0


def test_predict_binders_for_epitopes():
    predict_mhc_binders_for_epitopes(
        TEST_PEPTIDE_TSV,
        OUT_DIR,
        min_peptide_length=8,
        max_peptide_length=12,
        outlier_distance=0.4,
        hla_file_path=None,
        device="cuda",
    )
    assert os.path.exists(f"{OUT_DIR}/{MHC_DF_FOR_EPITOPES_TSV}")
    assert os.path.getsize(f"{OUT_DIR}/{MHC_DF_FOR_EPITOPES_TSV}") > 0


# def test_peptide_deconvolution():
#     deconvolute_peptides(
#         TEST_PEPTIDE_TSV,
#         2,
#         OUT_DIR,
#         min_peptide_length=8,
#         max_peptide_length=12,
#         outlier_distance=100,
#         hla_file_path=None,
#         device="cuda",
#     )
#     assert os.path.exists(f"{OUT_DIR}/{PEPTIDE_DECONVOLUTION_CLUSTER_DF_TSV}")
#     assert os.path.getsize(f"{OUT_DIR}/{PEPTIDE_DECONVOLUTION_CLUSTER_DF_TSV}") > 0
