import os
import pickle
import typing

import numpy as np
import pandas as pd
from alphabase.protein.fasta import load_all_proteins
from alphabase.protein.lcp_digest import get_substring_indices


def load_esm_pkl(fname="hla1_esm_embeds.pkl"):
    with open(fname, "rb") as f:
        _dict = pickle.load(f)
        return _dict["protein_df"], _dict["embedding_list"]


def load_hla_pep_df(folder=r"x:\Feng\HLA-DB\all_alleles\mixmhcpred", rank=2):
    df_list = []
    for fname in os.listdir(folder):
        df = pd.read_table(os.path.join(folder, fname), skiprows=11)
        df = df.query(f"`%Rank_bestAllele`<={rank}").copy()
        df["sequence"] = df["Peptide"]
        df = df[["sequence"]]
        df["allele"] = fname[:-4]
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


class NonSpecificDigest:
    def __init__(
        self, protein_data: typing.Tuple[pd.DataFrame, list, str], lens=[8, 14]
    ):
        if isinstance(protein_data, pd.DataFrame):
            self.cat_protein_sequence = (
                "$" + "$".join(protein_data.sequence.values) + "$"
            )
        else:
            if isinstance(protein_data, str):
                protein_data = [protein_data]
            protein_dict = load_all_proteins(protein_data)
            self.cat_protein_sequence = (
                "$" + "$".join([_["sequence"] for _ in protein_dict.values()]) + "$"
            )
        self.digest_starts, self.digest_stops = get_substring_indices(
            self.cat_protein_sequence, lens[0], lens[1]
        )

    def get_random_pept_df(self, n=5000):
        idxes = np.random.randint(0, len(self.digest_starts), size=n)
        df = pd.DataFrame(
            [
                self.cat_protein_sequence[start:stop]
                for start, stop in zip(
                    self.digest_starts[idxes], self.digest_stops[idxes]
                )
            ],
            columns=["sequence"],
        )
        df["allele"] = "random"
        return df

    def get_peptide_seqs_from_idxes(self, idxes):
        return [
            self.cat_protein_sequence[start:stop]
            for start, stop in zip(self.digest_starts[idxes], self.digest_stops[idxes])
        ]
