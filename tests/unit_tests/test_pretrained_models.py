import os

import numpy as np
import pandas as pd

from fennet_mhc.pipeline_api import PretrainedModels


def test_pretrained_models():
    pretrained_models = PretrainedModels(device="cpu")

    assert len(pretrained_models.background_protein_df) == 4357
    assert pretrained_models.device == "cpu"

    protein_df, hla_embeds = pretrained_models.embed_proteins(
        os.path.abspath("./test_data/test_MHC_proteins.fasta")
    )
    assert len(protein_df) > 0
    assert hla_embeds.shape[0] == len(protein_df)
    assert hla_embeds.shape[1] == pretrained_models.esm2_model.embed_dim

    peptide_list, pept_embeds = pretrained_models.embed_peptides_from_fasta(
        os.path.abspath("./test_data/test_peptides.fasta"),
        min_peptide_length=8,
        max_peptide_length=12,
    )
    assert len(peptide_list) > 0
    assert pept_embeds.shape[0] == len(peptide_list)
    assert pept_embeds.shape[1] == pretrained_models.esm2_model.embed_dim

    peptide_tsv = os.path.abspath("./test_data/test_peptides.tsv")
    peptide_list, pept_embeds = pretrained_models.embed_peptides_tsv(
        peptide_tsv,
        min_peptide_length=8,
        max_peptide_length=1000,
    )
    assert len(peptide_list) > 0
    assert pept_embeds.shape[0] == len(peptide_list)
    assert pept_embeds.shape[1] == pretrained_models.esm2_model.embed_dim

    peptide_df = pd.read_csv(peptide_tsv, sep="\t")

    _idx = pretrained_models.hla_df.query("allele=='A02_01'").index[0]
    A0201_embed = pretrained_models.hla_embeddings[_idx]
    peptide_df["dist"] = np.linalg.norm(pept_embeds - A0201_embed, axis=1)

    peptide_df["TP"] = (peptide_df["dist"] < 0.4) & (peptide_df["Rank_A0201"] < 2)
    peptide_df["TN"] = (peptide_df["dist"] > 0.4) & (peptide_df["Rank_A0201"] > 2)
    agrees = (peptide_df["TP"] | peptide_df["TN"]).astype(float)
    assert agrees.mean() >= 0.95
