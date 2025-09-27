"""Utility helpers for handling peptide and protein sequences."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import pandas as pd
from alphabase.protein.fasta import load_all_proteins
from alphabase.protein.lcp_digest import get_substring_indices


def load_peptide_df_from_mixmhcpred(mixmhcpred_dir: str, rank: int = 2) -> pd.DataFrame:
    """Load peptides predicted by MixMHCpred.

    This function reads all result files from a directory produced by
    MixMHCpred and collects peptide sequences with a rank below a specified
    threshold.

    Args:
        mixmhcpred_dir (str): Path to the directory containing MixMHCpred
            output ``.tsv`` files.
        rank (int): Maximum ``%Rank_bestAllele`` value to keep. Only peptides
            with a rank less than or equal to this value are returned.

    Returns:
        pandas.DataFrame: A DataFrame with columns ``sequence`` and ``allele``
            containing the filtered peptides.
    """

    df_list: list[pd.DataFrame] = []
    for fname in os.listdir(mixmhcpred_dir):
        df = pd.read_table(os.path.join(mixmhcpred_dir, fname), skiprows=11)
        df = df.query(f"`%Rank_bestAllele`<={rank}").copy()
        df["sequence"] = df["Peptide"]
        df = df[["sequence"]]
        df["allele"] = fname[:-4]
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


class NonSpecificDigest:
    """Generate peptides from a protein sequence without specific cleavage."""

    def __init__(
        self,
        protein_data: pd.DataFrame | list[str] | str,
        min_peptide_len: int = 8,
        max_peptide_len: int = 14,
    ) -> None:
        """Create a digestion helper.

        Args:
            protein_data (Union[pandas.DataFrame, list[str], str]): Either a
                DataFrame with a ``sequence`` column, a path to a fasta file, or
                a list of fasta file paths containing protein sequences.
            min_peptide_len (int): Minimum peptide length produced by the
                digestion.
            max_peptide_len (int): Maximum peptide length produced by the
                digestion.
        """

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
            self.cat_protein_sequence, min_peptide_len, max_peptide_len
        )

    def get_random_pept_df(self, n: int = 5000) -> pd.DataFrame:
        """Randomly sample digested peptides.

        Args:
            n (int): Number of peptides to sample.

        Returns:
            pandas.DataFrame: DataFrame with a ``sequence`` column containing the
            sampled peptides and an ``allele`` column set to ``"random"``.
        """

        idxes = np.random.randint(0, len(self.digest_starts), size=n)
        df = pd.DataFrame(
            [
                self.cat_protein_sequence[start:stop]
                for start, stop in zip(
                    self.digest_starts[idxes], self.digest_stops[idxes], strict=False
                )
            ],
            columns=["sequence"],
        )
        df["allele"] = "random"
        return df

    def get_peptide_seqs_from_idxes(
        self, idxes: Sequence[int] | np.ndarray
    ) -> list[str]:
        """Return peptide sequences for the given digest indices.

        Args:
            idxes (Sequence[int] | numpy.ndarray): Indices into the digested
                peptide list.

        Returns:
            list[str]: The peptide sequences corresponding to ``idxes``.
        """

        return [
            self.cat_protein_sequence[start:stop]
            for start, stop in zip(
                self.digest_starts[idxes], self.digest_stops[idxes], strict=False
            )
        ]
