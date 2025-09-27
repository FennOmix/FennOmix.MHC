# We use pipeline_api to avoid unnecessary imports of cli
import logging
import os
import pickle
import ssl
import urllib.request
from pathlib import Path

import esm
import faiss
import numpy as np
import pandas as pd
import torch
import tqdm
from alphabase.protein.fasta import load_fasta_list_as_protein_df
from peptdeep.utils import _get_delimiter, set_logger

from fennomix_mhc.constants._const import (
    BACKGROUND_FASTA_PATH,
    D_MODEL,
    FENNOMIXMHC_MODEL_DIR,
    MHC_DF_FOR_EPITOPES_TSV,
    MHC_EMBEDDING_KEY,
    MHC_EMBEDDING_PATH,
    MHC_EMBEDDING_PSEUDO_KEY,
    MHC_EMBEDDING_PSEUDO_PATH,
    MHC_MODEL_KEY,
    MHC_MODEL_PATH,
    MHC_MODEL_PSEUDO_KEY,
    MHC_MODEL_PSEUDO_PATH,
    PEPTIDE_DECONVOLUTION_CLUSTER_DF_TSV,
    PEPTIDE_DF_FOR_MHC_TSV,
    PEPTIDE_MODEL_KEY,
    PEPTIDE_MODEL_PATH,
    PEPTIDE_MODEL_PSEUDO_KEY,
    PEPTIDE_MODEL_PSEUDO_PATH,
    PEPTIDES_FOR_MHC_FASTA,
    global_settings,
)
from fennomix_mhc.mhc_binding_model import (
    ModelHlaEncoder,
    ModelSeqEncoder,
    embed_hla_esm_list,
    embed_peptides,
)
from fennomix_mhc.mhc_binding_retriever import MHCBindingRetriever
from fennomix_mhc.mhc_utils import NonSpecificDigest


class PretrainedModels:
    """Container for pretrained models used in the pipeline.

    This class lazily downloads the required model weights and provides
    convenience methods to embed proteins and peptides as well as to predict
    peptide-MHC interactions.
    """

    def __init__(self, device: str = "cuda", use_pseudo: bool = False):
        """Initialize the pretrained models and load model weights.

        Parameters
        ----------
        device : str, optional
            Device used for inference (``"cuda"``, ``"cpu"`` or ``"mps"``),
            by default ``"cuda"``.
        use_pseudo : bool, optional
            Whether to use the pseudo model, by default ``False``.
        """
        self.device = _set_device(device)
        self._use_pseudo = use_pseudo
        _download_pretrained_models()
        self.hla_encoder = (
            self._load_mhc_model_pseudo() if use_pseudo else self._load_mhc_model()
        )
        self.pept_encoder = (
            self._load_peptide_model_pseudo()
            if use_pseudo
            else self._load_peptide_model()
        )
        self.hla_encoder.eval()
        self.pept_encoder.eval()

        if not use_pseudo:
            self.esm2_model, self.esm2_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            self.esm2_model.to(self.device)
            self.esm2_model.eval()
            self.batch_converter = self.esm2_alphabet.get_batch_converter()

        self.background_protein_df = load_fasta_list_as_protein_df(
            [BACKGROUND_FASTA_PATH]
        )
        self.hla_df, self.hla_embeddings = _load_hla_embedding_pkl(
            MHC_EMBEDDING_PSEUDO_PATH if use_pseudo else MHC_EMBEDDING_PATH
        )
        self.hla_df.reset_index(drop=True, inplace=True)

    def _embed_proteins_pseudo(self, protein_df: pd.DataFrame):
        total_mhc_embeds = np.empty((0, D_MODEL), dtype=np.float32)
        batch_size = 1024

        logging.info(
            f"Embedding MHC protein sequences using peptdeep model ...\n"
            f"  Total sequences: {len(protein_df)}\n"
            f"  Model embedding dimension: {D_MODEL}\n"
            f"  Device: {self.device}\n"
            f"  Batch size: {batch_size}\n"
        )

        with torch.no_grad():
            for i in tqdm.tqdm(range(0, len(protein_df), batch_size)):
                sequences = (
                    protein_df["pseudo_sequence"].values[i : i + batch_size].astype(str)
                )

                mhc_embeds = embed_peptides(
                    self.pept_encoder,
                    sequences,
                    d_model=D_MODEL,
                    batch_size=batch_size,
                    device=self.device,
                )
                total_mhc_embeds = np.concatenate(
                    (total_mhc_embeds, mhc_embeds), axis=0
                )

        return total_mhc_embeds

    def embed_proteins(self, fasta: str):
        """Embed HLA protein sequences from a FASTA file.

        Parameters
        ----------
        fasta : str
            Path to a FASTA file containing HLA sequences.

        Returns
        -------
        tuple[pd.DataFrame, np.ndarray]
            A tuple of the loaded protein dataframe and the corresponding
            embeddings.
        """
        if not os.path.exists(fasta):
            raise FileNotFoundError(f"Fasta file not found: {fasta}")
        logging.info(f"Loading MHC protein sequences from `{fasta}` ...\n")
        protein_df = load_fasta_list_as_protein_df([fasta])
        protein_df.rename(columns={"protein_id": "allele"}, inplace=True)

        if self._use_pseudo:
            return protein_df, self._embed_proteins_pseudo(protein_df)

        hla_esm_embedding_list = []
        batch_size = 100

        logging.info(
            f"Embedding MHC protein sequences using ESM-2 model ...\n"
            f"  Total sequences: {len(protein_df)}\n"
            f"  ESM-2 model: {self.esm2_model.__class__.__name__}\n"
            f"  ESM-2 model device: {self.device}\n"
            f"  ESM-2 model embedding dimension: {D_MODEL}\n"
            f"  Batch size: {batch_size}\n"
        )
        with torch.no_grad():
            for i in tqdm.tqdm(range(0, len(protein_df), batch_size)):
                sequences = protein_df.sequence.values[i : i + batch_size]
                data = list(zip(["_"] * len(sequences), sequences, strict=False))
                batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
                results = self.esm2_model(
                    batch_tokens.to(self.device),
                    repr_layers=[12],
                    return_contacts=False,
                )
                hla_esm_embedding_list.extend(
                    list(
                        results["representations"][12]
                        .cpu()
                        .detach()
                        .numpy()[:, 1:-1]
                        .copy()
                    )
                )
        logging.info("Embedding MHC proteins after ESM-2 model ...\n")
        hla_embeds = embed_hla_esm_list(
            self.hla_encoder, hla_esm_embedding_list, device=self.device, verbose=True
        )

        return protein_df, hla_embeds

    def embed_peptides_from_fasta(
        self,
        fasta: str,
        min_peptide_length: int = 8,
        max_peptide_length: int = 12,
    ):
        """Digest proteins in a FASTA file and embed the resulting peptides.

        Parameters
        ----------
        fasta : str
            Path to a FASTA file containing proteins.
        min_peptide_length : int, optional
            Minimum peptide length to generate, by default ``8``.
        max_peptide_length : int, optional
            Maximum peptide length to generate, by default ``12``.

        Returns
        -------
        tuple[list[str], np.ndarray]
            A list of peptide sequences and their embeddings.
        """
        logging.info(f"Loading peptide sequences from `{fasta}` ...\n")
        digest = NonSpecificDigest(fasta, min_peptide_length, max_peptide_length)
        total_peptides_num = len(digest.digest_starts)

        if total_peptides_num == 0:
            raise ValueError("No valid peptides found in fasta file")

        batch_size = 1000000
        batches = range(0, total_peptides_num, batch_size)
        batches = tqdm.tqdm(batches)

        logging.info(
            f"Embedding peptide sequences ...\n"
            f"  Total sequences: {total_peptides_num}\n"
            f"  Batch size: {batch_size}\n"
        )
        total_peptide_list = []
        total_pept_embeds = np.empty((0, D_MODEL), dtype=np.float32)

        for start_major in batches:
            if start_major + batch_size >= total_peptides_num:
                stop_major = total_peptides_num
            else:
                stop_major = start_major + batch_size

            peptide_list = digest.get_peptide_seqs_from_idxes(
                np.arange(start_major, stop_major)
            )

            pept_embeds = embed_peptides(
                self.pept_encoder,
                peptide_list,
                d_model=D_MODEL,
                batch_size=1024,
                device=self.device,
            )

            total_pept_embeds = np.concatenate((total_pept_embeds, pept_embeds), axis=0)
            total_peptide_list.extend(peptide_list)
        return total_peptide_list, total_pept_embeds

    def embed_peptides_tsv(
        self,
        peptide_tsv: str,
        min_peptide_length: int = 8,
        max_peptide_length: int = 12,
    ):
        """Embed peptides listed in a TSV/CSV file.

        Parameters
        ----------
        peptide_tsv : str
            Path to a delimited file containing a ``sequence`` column.
        min_peptide_length : int, optional
            Minimum allowed peptide length, by default ``8``.
        max_peptide_length : int, optional
            Maximum allowed peptide length, by default ``12``.

        Returns
        -------
        tuple[list[str], np.ndarray]
            A list of peptide sequences and their embeddings.
        """
        logging.info(f"Loading peptide sequences from `{peptide_tsv}` ...\n")
        delimiter = _get_delimiter(peptide_tsv)
        input_peptide_df = pd.read_table(peptide_tsv, sep=delimiter, index_col=False)
        before_filter_num = input_peptide_df.shape[0]
        input_peptide_df["peptide_length"] = input_peptide_df["sequence"].str.len()
        input_peptide_df = input_peptide_df[
            (input_peptide_df["peptide_length"] >= min_peptide_length)
            & (input_peptide_df["peptide_length"] <= max_peptide_length)
        ]
        after_filter_num = input_peptide_df.shape[0]
        if before_filter_num != after_filter_num:
            logging.info(
                f"Filter {before_filter_num - after_filter_num} peptides due to invalid length"
            )
        input_peptide_list = input_peptide_df["sequence"].tolist()

        if len(input_peptide_list) == 0:
            raise ValueError("No valid peptides found in tsv file")

        batch_size = 1000000
        batches = range(0, len(input_peptide_list), batch_size)
        batches = tqdm.tqdm(batches)

        logging.info(
            f"Embedding peptide sequences ...\n"
            f"  Total sequences: {len(input_peptide_list)}\n"
            f"  Batch size: {batch_size}\n"
        )
        total_pept_embeds = np.empty((0, D_MODEL), dtype=np.float32)

        for start_major in batches:
            if start_major + batch_size >= len(input_peptide_list):
                stop_major = len(input_peptide_list)
            else:
                stop_major = start_major + batch_size

            peptide_list = input_peptide_list[start_major:stop_major]

            pept_embeds = embed_peptides(
                self.pept_encoder,
                peptide_list,
                d_model=D_MODEL,
                batch_size=1024,
                device=self.device,
            )

            total_pept_embeds = np.concatenate((total_pept_embeds, pept_embeds), axis=0)

        return input_peptide_list, total_pept_embeds

    def predict_mhc_binders_for_epitopes(
        self,
        peptide_list: list,
        peptide_embeddings: np.ndarray,
        hla_df: pd.DataFrame = None,
        hla_embeddings: np.ndarray = None,
        min_peptide_length: int = 8,
        max_peptide_length: int = 12,
        outlier_distance: float = 0.4,
    ):
        """Find the best binding epitope for each HLA allele.

        Parameters
        ----------
        peptide_list : list
            List of peptide sequences.
        peptide_embeddings : np.ndarray
            Embeddings corresponding to ``peptide_list``.
        hla_df : pandas.DataFrame, optional
            DataFrame containing HLA information. If ``None`` the builtin
            embeddings are used.
        hla_embeddings : np.ndarray, optional
            Embeddings for the HLAs in ``hla_df``.
        min_peptide_length : int, optional
            Minimum peptide length considered, by default ``8``.
        max_peptide_length : int, optional
            Maximum peptide length considered, by default ``12``.
        outlier_distance : float, optional
            Distance threshold used to filter outliers, by default ``0.4``.

        Returns
        -------
        pandas.DataFrame
            A dataframe mapping each allele to its closest peptide.
        """
        logging.info("Predicting MHC binders for epitopes ...\n")
        peptide_lengths = np.array([len(pep) for pep in peptide_list])
        valid_indices = np.where(
            (peptide_lengths >= min_peptide_length)
            & (peptide_lengths <= max_peptide_length)
        )[0]
        peptide_list = [peptide_list[i] for i in valid_indices]
        peptide_embeddings = peptide_embeddings[valid_indices, :]

        if len(peptide_list) == 0:
            raise ValueError("No valid peptide sequences")

        if hla_df is None or hla_embeddings is None:
            hla_df = self.hla_df
            hla_embeddings = self.hla_embeddings

        retriever = MHCBindingRetriever(
            self.hla_encoder,
            self.pept_encoder,
            hla_df,
            hla_embeddings,
            self.background_protein_df,
            min_peptide_length,
            max_peptide_length,
            device=self.device,
        )

        all_allele_array = hla_df["allele"].tolist()

        logging.info(
            "Retrieving MHC binding distances for epitopes ...\n"
            f"  Total peptide sequences: {len(peptide_list)}\n"
            f"  MHC protein sequences: {len(hla_df)}\n"
        )
        ret_dists = retriever.get_embedding_distances(
            hla_embeddings, peptide_embeddings
        )
        best_peptide_idxes = np.argmin(ret_dists, axis=0)
        best_peptide_dists = ret_dists[
            best_peptide_idxes, np.arange(ret_dists.shape[1])
        ]
        best_peptide_list = [peptide_list[k] for k in best_peptide_idxes]

        allele_df = pd.DataFrame(
            {
                "allele": all_allele_array,
                "best_peptide": best_peptide_list,
                "best_peptide_dist": best_peptide_dists,
            }
        )
        allele_df = allele_df[allele_df["best_peptide_dist"] <= outlier_distance]
        allele_df.sort_values("allele", ascending=True, inplace=True, ignore_index=True)

        return allele_df

    def predict_epitopes_for_mhc(
        self,
        peptide_list: list,
        peptide_embeddings: np.ndarray,
        alleles: list,
        hla_df: pd.DataFrame = None,
        hla_embeddings: np.ndarray = None,
        min_peptide_length: int = 8,
        max_peptide_length: int = 12,
        outlier_distance: float = 0.4,
    ):
        """Predict the most likely allele binder for each peptide.

        Parameters
        ----------
        peptide_list : list
            Peptide sequences to evaluate.
        peptide_embeddings : np.ndarray
            Embeddings for ``peptide_list``.
        alleles : list
            List of allele names to consider.
        hla_df : pandas.DataFrame, optional
            DataFrame containing HLA sequence information. If ``None`` the
            pretrained HLA database is used.
        hla_embeddings : np.ndarray, optional
            Embeddings for the HLAs in ``hla_df``.
        min_peptide_length : int, optional
            Minimum peptide length considered, by default ``8``.
        max_peptide_length : int, optional
            Maximum peptide length considered, by default ``12``.
        outlier_distance : float, optional
            Distance threshold used to filter outliers, by default ``0.4``.

        Returns
        -------
        pandas.DataFrame
            Table of peptides with their best matching allele and distance.
        """
        logging.info("Predicting peptide binders for MHC molecules...\n")
        peptide_lengths = np.array([len(pep) for pep in peptide_list])
        valid_indices = np.where(
            (peptide_lengths >= min_peptide_length)
            & (peptide_lengths <= max_peptide_length)
        )[0]
        peptide_list = [peptide_list[i] for i in valid_indices]
        peptide_embeddings = peptide_embeddings[valid_indices, :]

        if len(peptide_list) == 0:
            raise ValueError("No valid peptide sequences")

        if hla_df is None or hla_embeddings is None:
            hla_df = self.hla_df
            hla_embeddings = self.hla_embeddings

        retriever = MHCBindingRetriever(
            self.hla_encoder,
            self.pept_encoder,
            hla_df,
            hla_embeddings,
            self.background_protein_df,
            min_peptide_len=min_peptide_length,
            max_peptide_len=max_peptide_length,
            device=self.device,
        )

        selected_alleles = []
        for allele in alleles:
            if allele in retriever.dataset.allele_idxes_dict:
                selected_alleles.append(allele)
            else:
                logging.warning(
                    f"Ignore allele '{allele}' which is not available in allele library."
                )

        logging.info(
            f"Retrieving MHC binding distances for {len(selected_alleles)} alleles ...\n"
            f"  Total peptide sequences: {len(peptide_list)}\n"
            f"  MHC protein sequences in database: {len(hla_df)}\n"
        )
        peptide_df = retriever.get_binding_metrics_for_peptides(
            selected_alleles, peptide_embeddings
        )

        logging.info("Converting resulted peptide_df ...\n")
        peptide_df["sequence"] = peptide_list
        peptide_df = peptide_df.drop(columns=["best_allele_id"], errors="ignore")
        peptide_df = peptide_df[["sequence", "best_allele", "best_allele_dist"]]

        peptide_df = peptide_df[
            peptide_df["best_allele_dist"] <= outlier_distance
        ].sort_values(by="best_allele_dist", ascending=True, ignore_index=True)

        return peptide_df

    def deconvolute_peptides(
        self,
        peptide_list: list,
        pept_embeddings: np.ndarray,
        n_centroids: int = 8,
        outlier_distance: float = 0.2,
    ):
        """Cluster peptides based on their embeddings using k-means.

        Parameters
        ----------
        peptide_list : list
            List of peptide sequences.
        pept_embeddings : np.ndarray
            Embedding matrix corresponding to ``peptide_list``.
        n_centroids : int, optional
            Number of clusters to form, by default ``8``.
        outlier_distance : float, optional
            Distance threshold for optional centroid refinement, by default ``0.2``.

        Returns
        -------
        tuple[pandas.DataFrame, np.ndarray]
            DataFrame assigning each peptide to a cluster and the cluster
            centroid embeddings.
        """
        d = pept_embeddings.shape[1]
        kmeans = faiss.Kmeans(d, n_centroids)
        kmeans.niter = 20
        kmeans.verbose = True
        kmeans.min_points_per_centroid = 10
        kmeans.max_points_per_centroid = len(peptide_list) // n_centroids * 3

        logging.info(
            f"Using faiss.Kmeans to cluster peptides ...\n"
            f"  Total peptide sequences: {len(peptide_list)}\n"
            f"  Number of clusters: {n_centroids}\n"
            f"  Number of iterations: {kmeans.niter}\n"
            f"  Minimum points per centroid: {kmeans.min_points_per_centroid}\n"
            f"  Maximum points per centroid: {kmeans.max_points_per_centroid}\n"
        )

        kmeans.train(pept_embeddings)
        centroids = kmeans.centroids
        logging.info(f"Got {len(centroids)} clusters ...\n")

        trained_index = faiss.IndexFlatL2(d)
        trained_index.add(centroids)

        logging.info(
            f"Finding the closest cluster for each peptide ...\n"
            f"  Total peptide sequences: {len(peptide_list)}\n"
        )
        return_dists, return_labels = trained_index.search(pept_embeddings, 1)
        cluster_labels = return_labels.flatten()
        cluster_dists = return_dists.flatten()

        cluster_df = pd.DataFrame(
            {
                "sequence": peptide_list,
                "cluster_id": cluster_labels,
                "dist_to_cluster": cluster_dists,
            }
        )

        if outlier_distance > 0 and outlier_distance < 1:
            logging.info(
                f"Refining the cluster centers for {len(cluster_df)} peptides"
                f"based on {len(centroids)} centers with outliers>{outlier_distance:.2f} ...\n"
            )
            cluster_df = cluster_df.query(f"dist_to_cluster <= {outlier_distance}")

            refined_center_df: pd.DataFrame = cluster_df.groupby("cluster_id").apply(
                lambda g: pept_embeddings[g.index].mean(axis=0)
            )
            refined_center_df = refined_center_df.reset_index(names=["new_center"])
            id_map = refined_center_df["cluster_id"].to_dict()
            id_map = {v: k for k, v in id_map.items()}
            cluster_df["cluster_id"] = cluster_df["cluster_id"].map(id_map)
            centroids = np.array(refined_center_df["new_center"].tolist())
            logging.info(
                f"Cluster refinement resulted in {len(centroids)} clusters with {len(cluster_df)} peptides ...\n"
            )

        return cluster_df, centroids

    def _load_mhc_model(self):
        """Load the pretrained MHC encoder model."""
        hla_encoder = ModelHlaEncoder()
        hla_encoder.to(self.device)
        hla_encoder.load_state_dict(
            torch.load(MHC_MODEL_PATH, weights_only=True, map_location=self.device)
        )
        return hla_encoder

    def _load_mhc_model_pseudo(self):
        """Load the pretrained pseudo MHC encoder model."""
        hla_encoder = ModelSeqEncoder()
        hla_encoder.to(self.device)
        hla_encoder.load_state_dict(
            torch.load(
                MHC_MODEL_PSEUDO_PATH, weights_only=True, map_location=self.device
            )
        )
        return hla_encoder

    def _load_peptide_model(self):
        """Load the pretrained peptide encoder model."""
        pept_encoder = ModelSeqEncoder()
        pept_encoder.to(self.device)
        pept_encoder.load_state_dict(
            torch.load(PEPTIDE_MODEL_PATH, weights_only=True, map_location=self.device)
        )
        return pept_encoder

    def _load_peptide_model_pseudo(self):
        """Load the pretrained pseudo peptide encoder model."""
        pept_encoder = ModelSeqEncoder()
        pept_encoder.to(self.device)
        pept_encoder.load_state_dict(
            torch.load(
                PEPTIDE_MODEL_PSEUDO_PATH, weights_only=True, map_location=self.device
            )
        )
        return pept_encoder


def embed_proteins(fasta: str, out_folder: str, device: str = "cuda"):
    """Embed HLA protein sequences and save the embeddings to disk.

    Parameters
    ----------
    fasta : str
        Path to a FASTA file containing HLA protein sequences.
    out_folder : str
        Directory where the resulting ``hla_embeddings.pkl`` is stored.
    device : str, optional
        Device used for embedding (``"cuda"``, ``"cpu"`` or ``"mps"``), by
        default ``"cuda"``.

    Returns
    -------
    None
    """
    os.makedirs(out_folder, exist_ok=True)
    set_logger(
        log_file_name=os.path.join(out_folder, global_settings["log_file_name"]),
        log_level=global_settings["log_level"].lower(),
    )

    pretrained_models = PretrainedModels(device=device)

    logging.info("Embedding MHC protein sequences ...\n")
    protein_df, hla_embeds = pretrained_models.embed_proteins(fasta)

    logging.info(f"Saving hla_embeddings.pkl to `{out_folder}` ...\n")
    with open(os.path.join(out_folder, "hla_embeddings.pkl"), "wb") as f:
        pickle.dump(
            {"protein_df": protein_df, "embeds": hla_embeds},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    logging.info("Finished `embed_proteins()`.")


def embed_peptides_from_file(
    peptide_file_path: str,
    out_folder: str,
    min_peptide_length: int = 8,
    max_peptide_length: int = 12,
    device: str = "cuda",
    use_pseudo: bool = False,
):
    """Embed peptides provided in a FASTA or tabular file and save them.

    Parameters
    ----------
    peptide_file_path : str
        Input file containing peptide sequences. Supported formats are FASTA and
        TSV/CSV files with a ``sequence`` column.
    out_folder : str
        Directory where the resulting ``peptide_embeddings.pkl`` is stored.
    min_peptide_length : int, optional
        Minimum length of peptides to keep, by default ``8``.
    max_peptide_length : int, optional
        Maximum length of peptides to keep, by default ``12``.
    device : str, optional
        Device used for embedding (``"cuda"``, ``"cpu"`` or ``"mps"``), by
        default ``"cuda"``.
    use_pseudo : bool, optional
        Whether to use the pseudo model for embedding, by default ``False``.

    Returns
    -------
    None
    """
    os.makedirs(out_folder, exist_ok=True)
    set_logger(
        log_file_name=os.path.join(out_folder, global_settings["log_file_name"]),
        log_level=global_settings["log_level"].lower(),
    )

    pretrained_models = PretrainedModels(device=device, use_pseudo=use_pseudo)

    logging.info(f"Embedding peptides from file `{peptide_file_path}` ...\n")
    if peptide_file_path.lower().endswith(".fasta"):
        peptide_list, peptide_embeds = pretrained_models.embed_peptides_from_fasta(
            peptide_file_path, min_peptide_length, max_peptide_length
        )
    elif peptide_file_path[-4:].lower() in [".tsv", ".txt", ".csv"]:
        peptide_list, peptide_embeds = pretrained_models.embed_peptides_tsv(
            peptide_file_path, min_peptide_length, max_peptide_length
        )
    else:
        raise ValueError(
            f"Unsupported peptide file format: {peptide_file_path}. "
            "Please provide a .fasta or .tsv/.csv file."
        )

    logging.info(f"Saving peptide_embeddings.pkl to `{out_folder}` ...\n")
    with open(os.path.join(out_folder, "peptide_embeddings.pkl"), "wb") as f:
        pickle.dump(
            {"peptide_list": peptide_list, "pept_embeds": peptide_embeds},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    logging.info("Finished `embed_peptides_from_file()`.")


def predict_epitopes_for_mhc(
    peptide_file_path: str,
    alleles: list,
    out_folder: str,
    out_fasta_format: bool = False,
    min_peptide_length: int = 8,
    max_peptide_length: int = 12,
    outlier_distance: float = 0.4,
    hla_file_path: str = None,
    device: str = "cuda",
    use_pseudo: bool = False,
):
    """Predict peptide binders for the given MHC alleles.

    Parameters
    ----------
    peptide_file_path : str
        Path to peptide embeddings or sequence file.
    alleles : list
        Alleles to consider during prediction.
    out_folder : str
        Directory where results are written.
    out_fasta_format : bool, optional
        Whether to output a FASTA file of peptides instead of TSV, by default
        ``False``.
    min_peptide_length : int, optional
        Minimum peptide length, by default ``8``.
    max_peptide_length : int, optional
        Maximum peptide length, by default ``12``.
    outlier_distance : float, optional
        Distance threshold used to filter predictions, by default ``0.4``.
    hla_file_path : str, optional
        Optional path to custom HLA embeddings or FASTA file.
    device : str, optional
        Device for running the model (``"cuda"``, ``"cpu"`` or ``"mps"``),
        by default ``"cuda"``.
    use_pseudo : bool, optional
        Whether to use the pseudo model for embedding, by default ``False``.

    Returns
    -------
    None
    """
    os.makedirs(out_folder, exist_ok=True)
    set_logger(
        log_file_name=os.path.join(out_folder, global_settings["log_file_name"]),
        log_level=global_settings["log_level"].lower(),
    )

    pretrained_models = PretrainedModels(device=device, use_pseudo=use_pseudo)

    logging.info(f"Loading peptide embeddings from `{peptide_file_path}` ...\n")
    peptide_list, pept_embeds = _load_peptide_embeddings(
        pretrained_models, peptide_file_path, min_peptide_length, max_peptide_length
    )

    logging.info(f"Loading MHC protein embeddings from `{hla_file_path}` ...\n")
    protein_df, hla_embeds = _load_protein_embeddings(pretrained_models, hla_file_path)

    logging.info("Calling `pretrained_models.predict_peptide_binders_for_MHC()` ...\n")
    peptide_df = pretrained_models.predict_epitopes_for_mhc(
        peptide_list,
        pept_embeds,
        alleles=alleles,
        hla_df=protein_df,
        hla_embeddings=hla_embeds,
        min_peptide_length=min_peptide_length,
        max_peptide_length=max_peptide_length,
        outlier_distance=outlier_distance,
    )

    output_dir = Path(out_folder)
    if out_fasta_format:
        output_file_path = output_dir.joinpath(PEPTIDES_FOR_MHC_FASTA)
        logging.info(f"Saving peptide sequences to `{output_file_path}` ...\n")
        with open(output_file_path, "w") as f:
            for seq, allele, dist in peptide_df[
                ["sequence", "best_allele", "best_allele_dist"]
            ].values:
                f.write(f">{seq} {allele} dist={dist:.3f}\n{seq}\n")
    else:
        output_file_path = output_dir.joinpath(PEPTIDE_DF_FOR_MHC_TSV)
        logging.info(f"Saving peptide sequences to `{output_file_path}` ...\n")
        peptide_df = peptide_df.round(3)
        peptide_df.to_csv(output_file_path, sep="\t", index=False)
    print(f"Peptide results saved to: {output_file_path}")


def predict_mhc_binders_for_epitopes(
    peptide_file_path: str,
    out_folder: str,
    min_peptide_length: int = 8,
    max_peptide_length: int = 12,
    outlier_distance: float = 0.4,
    hla_file_path: str = None,
    device: str = "cuda",
    use_pseudo: bool = False,
):
    """Find MHC binders for the given peptides (epitopes).

    Parameters
    ----------
    peptide_file_path : str
        Path to peptide embeddings or sequence file.
    out_folder : str
        Directory where the resulting predictions are saved.
    min_peptide_length : int, optional
        Minimum peptide length considered, by default ``8``.
    max_peptide_length : int, optional
        Maximum peptide length considered, by default ``12``.
    outlier_distance : float, optional
        Distance threshold used to filter predictions, by default ``0.4``.
    hla_file_path : str, optional
        Optional path to custom HLA embeddings or FASTA file.
    device : str, optional
        Device for running the model (``"cuda"``, ``"cpu"`` or ``"mps"``),
        by default ``"cuda"``.
    use_pseudo : bool, optional
        Whether to use the pseudo model for embedding, by default ``False``.

    Returns
    -------
    None
    """
    os.makedirs(out_folder, exist_ok=True)
    set_logger(
        log_file_name=os.path.join(out_folder, global_settings["log_file_name"]),
        log_level=global_settings["log_level"].lower(),
    )

    pretrained_models = PretrainedModels(device=device, use_pseudo=use_pseudo)

    logging.info(f"Loading peptide embeddings from {peptide_file_path} ...\n")
    peptide_list, pept_embeds = _load_peptide_embeddings(
        pretrained_models, peptide_file_path, min_peptide_length, max_peptide_length
    )

    logging.info(f"Loading MHC protein embeddings from {hla_file_path} ...\n")
    protein_df, hla_embeds = _load_protein_embeddings(pretrained_models, hla_file_path)

    logging.info("Predicting peptide binders for MHC molecules ...\n")
    allele_df = pretrained_models.predict_mhc_binders_for_epitopes(
        peptide_list,
        pept_embeds,
        hla_df=protein_df,
        hla_embeddings=hla_embeds,
        min_peptide_length=min_peptide_length,
        max_peptide_length=max_peptide_length,
        outlier_distance=outlier_distance,
    )

    logging.info(f"Saving retrieved hla_df to {out_folder} ...\n")
    allele_df = allele_df.round(3)
    output_dir = Path(out_folder)
    output_file_path = output_dir.joinpath(MHC_DF_FOR_EPITOPES_TSV)
    allele_df.to_csv(output_file_path, sep="\t", index=False)
    logging.info(f"hla_df saved to: {output_file_path}")


def deconvolute_peptides(
    peptide_file_path: str,
    n_centroids: int,
    out_folder: str,
    min_peptide_length: int = 8,
    max_peptide_length: int = 12,
    outlier_distance: float = 100,
    hla_file_path: str = None,
    device: str = "cuda",
    use_pseudo: bool = False,
):
    """Cluster peptides and assign a representative allele to each cluster.

    Parameters
    ----------
    peptide_file_path : str
        Path to peptide embeddings or sequence file used for clustering.
    n_centroids : int
        Number of clusters to form.
    out_folder : str
        Directory where clustering results are saved.
    min_peptide_length : int, optional
        Minimum peptide length, by default ``8``.
    max_peptide_length : int, optional
        Maximum peptide length, by default ``12``.
    outlier_distance : float, optional
        Distance threshold for refining clusters, by default ``100`` (disabled).
    hla_file_path : str, optional
        Optional path to custom HLA embeddings or FASTA file.
    device : str, optional
        Device for running the model (``"cuda"``, ``"cpu"`` or ``"mps"``),
        by default ``"cuda"``.
    use_pseudo : bool, optional
        Whether to use the pseudo model for embedding, by default ``False``.

    Returns
    -------
    None
    """
    os.makedirs(out_folder, exist_ok=True)
    set_logger(
        log_file_name=os.path.join(out_folder, global_settings["log_file_name"]),
        log_level=global_settings["log_level"].lower(),
    )

    pretrained_models = PretrainedModels(device=device, use_pseudo=use_pseudo)

    logging.info(f"Loading peptide embeddings from `{peptide_file_path}` ...\n")
    peptide_list, pept_embeds = _load_peptide_embeddings(
        pretrained_models, peptide_file_path, min_peptide_length, max_peptide_length
    )

    logging.info(f"Loading MHC protein embeddings from `{hla_file_path}` ...\n")
    protein_df, hla_embeds = _load_protein_embeddings(pretrained_models, hla_file_path)

    cluster_df, _ = _deconvolute_without_save(
        pretrained_models,
        peptide_list,
        pept_embeds,
        hla_embeds,
        protein_df,
        n_centroids,
        outlier_distance,
    )

    output_dir = Path(out_folder)
    output_file_path = output_dir.joinpath(PEPTIDE_DECONVOLUTION_CLUSTER_DF_TSV)
    logging.info(f"Saving peptide clusters to `{output_file_path}` ...\n")
    cluster_df = cluster_df.round(3)
    cluster_df.to_csv(output_file_path, sep="\t", index=False)
    logging.info(f"Deconvolution results saved to: `{output_file_path}`")

    # matplotlib.rcParams["axes.grid"] = False
    # kmers = [8, 9, 10, 11]

    # for i in range(n_centroids):
    #     plot_motif_multi_mer(
    #         cluster_df.copy(),
    #         allele_col="cluster_label",
    #         allele=i,
    #         kmers=kmers,
    #         fig_width_per_kmer=4,
    #     )
    #     plt.savefig(f"{out_folder}/{i}_cluster_motif.svg")


def deconvolute_and_predict_peptides(
    peptide_file_path_to_deconv: str | Path,
    peptide_file_path_to_predict: str | Path,
    n_centroids: int,
    out_folder: str | Path,
    out_fasta_format: bool,
    min_peptide_length: int = 8,
    max_peptide_length: int = 12,
    outlier_distance: float = 0.2,
    hla_file_path: str | Path = None,
    device: str = "cuda",
    use_pseudo: bool = False,
):
    """Cluster peptides, deduce "pseudo" alleles, and predict binders for
    another peptide set.

    The first peptide file is used to derive clusters representing allele
    specificities. The second file is queried against these clusters to
    identify candidate binders.

    Parameters
    ----------
    peptide_file_path_to_deconv : str | Path
        File containing peptides used for deconvolution (clustering).
    peptide_file_path_to_predict : str | Path
        File containing peptides for which binders should be predicted.
    n_centroids : int
        Number of clusters to form during deconvolution.
    out_folder : str | Path
        Directory where the results are stored.
    out_fasta_format : bool
        If ``True`` write results to FASTA instead of TSV.
    min_peptide_length : int, optional
        Minimum peptide length considered, by default ``8``.
    max_peptide_length : int, optional
        Maximum peptide length considered, by default ``12``.
    outlier_distance : float, optional
        Distance threshold used during clustering and prediction, by default ``0.2``.
    hla_file_path : str | Path, optional
        Optional path to custom HLA embeddings or FASTA file.
    device : str, optional
        Device for running the model (``"cuda"``, ``"cpu"`` or ``"mps"``),
        by default ``"cuda"``.
    use_pseudo : bool, optional
        Whether to use the pseudo model for embedding, by default ``False``.

    Returns
    -------
    None
    """
    os.makedirs(out_folder, exist_ok=True)
    set_logger(
        log_file_name=os.path.join(out_folder, global_settings["log_file_name"]),
        log_level=global_settings["log_level"].lower(),
    )

    pretrained_models = PretrainedModels(device=device, use_pseudo=use_pseudo)

    logging.info(
        f"Loading peptide embeddings to deconv from `{peptide_file_path_to_deconv}` ...\n"
    )
    peptide_list, pept_embeds = _load_peptide_embeddings(
        pretrained_models,
        peptide_file_path_to_deconv,
        min_peptide_length,
        max_peptide_length,
    )

    logging.info(f"Loading MHC protein embeddings from `{hla_file_path}` ...\n")
    protein_df, hla_embeds = _load_protein_embeddings(pretrained_models, hla_file_path)

    cluster_df, _ = _deconvolute_without_save(
        pretrained_models,
        peptide_list,
        pept_embeds,
        hla_embeds,
        protein_df,
        n_centroids,
        outlier_distance,
    )

    logging.info(
        f"Loading peptide embeddings to predict from `{peptide_file_path_to_predict}` ...\n"
    )
    peptide_list, pept_embeds = _load_peptide_embeddings(
        pretrained_models,
        peptide_file_path_to_predict,
        min_peptide_length,
        max_peptide_length,
    )

    logging.info("Calling `pretrained_models.predict_peptide_binders_for_MHC()` ...\n")
    peptide_df = pretrained_models.predict_epitopes_for_mhc(
        peptide_list,
        pept_embeds,
        alleles=cluster_df["closest_allele"].unique().tolist(),
        hla_df=protein_df,
        hla_embeddings=hla_embeds,
        min_peptide_length=min_peptide_length,
        max_peptide_length=max_peptide_length,
        outlier_distance=outlier_distance,
    )

    output_dir = Path(out_folder)
    if out_fasta_format:
        logging.info(
            f"Saving peptide sequences to `{output_dir}/peptides_for_MHC.fasta` ...\n"
        )
        output_file_path = output_dir.joinpath("peptides_for_MHC.fasta")
        with open(output_file_path, "w") as f:
            for seq, allele, dist in peptide_df[
                ["sequence", "best_allele", "best_allele_dist"]
            ].values:
                f.write(f">{seq} {allele} dist={dist:.3f}\n{seq}\n")
    else:
        output_file_path = output_dir.joinpath(PEPTIDE_DF_FOR_MHC_TSV)
        logging.info(f"Saving peptide sequences to `{output_file_path}` ...\n")
        peptide_df = peptide_df.round(3)
        peptide_df.to_csv(output_file_path, sep="\t", index=False)
    print(f"Peptide results saved to: {output_file_path}")


def _deconvolute_without_save(
    pretrained_models: PretrainedModels,
    peptide_list: list,
    pept_embeds: np.ndarray,
    hla_embeds: np.ndarray,
    hla_df: pd.DataFrame,
    n_centroids: int,
    outlier_distance,
):
    """Helper that performs deconvolution and allele assignment without saving.

    Parameters
    ----------
    pretrained_models : PretrainedModels
        Instance containing the models used for clustering.
    peptide_list : list
        List of peptides to cluster.
    pept_embeds : np.ndarray
        Embeddings for ``peptide_list``.
    hla_embeds : np.ndarray
        Embeddings of HLA proteins.
    hla_df : pandas.DataFrame
        DataFrame with HLA allele information corresponding to ``hla_embeds``.
    n_centroids : int
        Number of clusters to form.
    outlier_distance : float
        Distance threshold for refining clusters.

    Returns
    -------
    tuple[pandas.DataFrame, np.ndarray]
        Cluster assignment dataframe and centroid embeddings.
    """
    logging.info("Calling `pretrained_models.deconvolute_peptides()` ...\n")
    cluster_df, centroids = pretrained_models.deconvolute_peptides(
        peptide_list,
        pept_embeds,
        n_centroids=n_centroids,
        outlier_distance=outlier_distance,
    )

    logging.info("Assigning closest HLA alleles to each cluster ...\n")
    d = centroids.shape[1]
    trained_index = faiss.IndexFlatL2(d)
    trained_index.add(hla_embeds)
    dists, idxes = trained_index.search(centroids, 1)
    closest_alleles = hla_df["allele"].values[idxes.flatten()]

    cluster_df["closest_allele"] = closest_alleles[cluster_df["cluster_id"].values]
    cluster_df["closest_allele_dist"] = dists.flatten()[cluster_df["cluster_id"].values]
    return cluster_df, centroids


def _set_device(device: str) -> str:
    """
    Select the appropriate device based on availability.

    Args:
        device (str): The desired device ('cpu', 'cuda', or 'mps').

    Returns:
        str: The selected device ('cpu', 'cuda', or 'mps').
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available. Change to use CPU.")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("MPS (Apple Silicon GPU) not available. Change to use CPU.")
        device = "cpu"

    print(f"Using device: {device}")
    return device


def _download_pretrained_models(
    base_url: str = None, model_dir: str = FENNOMIXMHC_MODEL_DIR
):
    """
    Download pretrained models from a given URL.

    Args:
        base_url (str): The base URL to download the models from.
        model_dir (str): The directory to save the downloaded models.
    """
    if base_url is None:
        base_url = global_settings["download_url"]
    base_url += "" if base_url.endswith("/") else "/"
    os.makedirs(model_dir, exist_ok=True)

    peptide_url = base_url + global_settings[PEPTIDE_MODEL_KEY]
    hla_url = base_url + global_settings[MHC_MODEL_KEY]
    hla_embedding_url = base_url + global_settings[MHC_EMBEDDING_KEY]
    peptide_pseudo_url = base_url + global_settings[PEPTIDE_MODEL_PSEUDO_KEY]
    hla_pseudo_url = base_url + global_settings[MHC_MODEL_PSEUDO_KEY]
    hla_embedding_pseudo_url = base_url + global_settings[MHC_EMBEDDING_PSEUDO_KEY]
    background_fasta_url = base_url + global_settings["background_fasta"]

    if os.path.exists(MHC_MODEL_PATH) and os.path.exists(PEPTIDE_MODEL_PATH):
        logging.info("Pretrained models already exist. Skipping download.")
        return

    logging.info(
        f"Downloading required files ...:\n"
        f"  `{global_settings[PEPTIDE_MODEL_KEY]}` from `{peptide_url}`\n"
        f"  `{global_settings[MHC_MODEL_KEY]}` from`{hla_url}`\n"
        f"  `{global_settings[MHC_EMBEDDING_KEY]}` from`{hla_embedding_url}`\n"
        f"  `{global_settings[PEPTIDE_MODEL_PSEUDO_KEY]}` from `{peptide_pseudo_url}`\n"
        f"  `{global_settings[MHC_MODEL_PSEUDO_KEY]}` from`{hla_pseudo_url}`\n"
        f"  `{global_settings[MHC_EMBEDDING_PSEUDO_KEY]}` from`{hla_embedding_pseudo_url}`\n"
        f"  `{global_settings['background_fasta']}` from`{background_fasta_url}`"
    )
    try:
        context = ssl._create_unverified_context()

        logging.info(f"Downloading `{global_settings[MHC_EMBEDDING_KEY]}` ...")
        requests = urllib.request.urlopen(
            hla_embedding_url, context=context, timeout=10
        )
        with open(MHC_EMBEDDING_PATH, "wb") as f:
            f.write(requests.read())

        logging.info(f"Downloading `{global_settings[MHC_EMBEDDING_PSEUDO_KEY]}` ...")
        requests = urllib.request.urlopen(
            hla_embedding_pseudo_url, context=context, timeout=10
        )
        with open(MHC_EMBEDDING_PSEUDO_PATH, "wb") as f:
            f.write(requests.read())

        logging.info(f"Downloading `{global_settings['background_fasta']}` ...")
        requests = urllib.request.urlopen(
            background_fasta_url, context=context, timeout=10
        )
        with open(BACKGROUND_FASTA_PATH, "wb") as f:
            f.write(requests.read())

        logging.info(f"Downloading `{global_settings[PEPTIDE_MODEL_KEY]}` ...")
        requests = urllib.request.urlopen(peptide_url, context=context, timeout=10)
        with open(PEPTIDE_MODEL_PATH, "wb") as f:
            f.write(requests.read())

        logging.info(f"Downloading `{global_settings[PEPTIDE_MODEL_PSEUDO_KEY]}` ...")
        requests = urllib.request.urlopen(
            peptide_pseudo_url, context=context, timeout=10
        )
        with open(PEPTIDE_MODEL_PSEUDO_PATH, "wb") as f:
            f.write(requests.read())

        logging.info(f"Downloading `{global_settings[MHC_MODEL_KEY]}` ...")
        requests = urllib.request.urlopen(hla_url, context=context, timeout=10)
        with open(MHC_MODEL_PATH, "wb") as f:
            f.write(requests.read())

        logging.info(f"Downloading `{global_settings[MHC_MODEL_PSEUDO_KEY]}` ...")
        requests = urllib.request.urlopen(hla_pseudo_url, context=context, timeout=10)
        with open(MHC_MODEL_PSEUDO_PATH, "wb") as f:
            f.write(requests.read())
    except Exception as e:
        raise RuntimeError(f"Failed to download models: {e}") from e


def _load_protein_embeddings(pretrained_models: PretrainedModels, hla_file_path):
    """Load or generate embeddings for HLA proteins.

    Parameters
    ----------
    pretrained_models : PretrainedModels
        Instance providing embedding functionality.
    hla_file_path : str
        Path to a pickle file with precomputed embeddings or a FASTA file with
        sequences. If ``None`` the pretrained embeddings bundled with the
        package are used.

    Returns
    -------
    tuple[pandas.DataFrame, np.ndarray]
        The protein dataframe and corresponding embeddings.
    """
    if hla_file_path is None:
        return pretrained_models.hla_df, pretrained_models.hla_embeddings
    if hla_file_path.lower().endswith(".pkl"):
        try:
            return _load_hla_embedding_pkl(hla_file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load MHC protein embeddings: {e}") from e
    if hla_file_path.lower().endswith(".fasta"):
        return pretrained_models.embed_proteins(hla_file_path)
    raise ValueError(
        f"Unsupported MHC file format: {hla_file_path}. "
        "Please provide a .pkl or .fasta file."
    )


def _load_peptide_embeddings(
    pretrained_models, peptide_file_path, min_peptide_length, max_peptide_length
):
    """Load or compute embeddings for a peptide file.

    Parameters
    ----------
    pretrained_models : PretrainedModels
        Instance providing embedding functionality.
    peptide_file_path : str
        File containing peptide sequences or a pickle file with embeddings.
    min_peptide_length : int
        Minimum peptide length when generating embeddings from sequences.
    max_peptide_length : int
        Maximum peptide length when generating embeddings from sequences.

    Returns
    -------
    tuple[list[str], np.ndarray]
        List of peptide sequences and their embeddings.
    """
    if not os.path.exists(peptide_file_path):
        raise FileNotFoundError(f"Peptide file not found: {peptide_file_path}")
    if peptide_file_path.lower().endswith(".pkl"):
        try:
            with open(peptide_file_path, "rb") as f:
                data_dict = pickle.load(f)
                return data_dict["peptide_list"], data_dict["pept_embeds"]
        except Exception as e:
            raise RuntimeError(f"Failed to load Peptide embeddings: {e}") from e
    if peptide_file_path.lower().endswith(".fasta"):
        return pretrained_models.embed_peptides_from_fasta(
            peptide_file_path, min_peptide_length, max_peptide_length
        )
    if peptide_file_path[-4:].lower() in [".tsv", ".txt", ".csv"]:
        return pretrained_models.embed_peptides_tsv(
            peptide_file_path, min_peptide_length, max_peptide_length
        )
    raise ValueError(
        f"Unsupported peptide file format: {peptide_file_path}. "
        "Please provide a .pkl or .tsv/.csv file."
    )


def _load_hla_embedding_pkl(fname=None):
    """Load HLA protein embeddings from a pickle file.

    Parameters
    ----------
    fname : str, optional
        Path to the ``.pkl`` file. If ``None``, the bundled embeddings are
        loaded.

    Returns
    -------
    tuple[pandas.DataFrame, np.ndarray]
        The protein dataframe and corresponding embeddings.
    """
    if fname is None:
        fname = MHC_EMBEDDING_PATH
    if not os.path.exists(fname):
        raise FileNotFoundError(f".pkl file not found: {fname}")
    with open(fname, "rb") as f:
        _dict = pickle.load(f)
        return _dict["protein_df"], _dict["embeds"]


def load_peptide_embedding_pkl(fname):
    """Load peptide embeddings from a pickle file.

    Parameters
    ----------
    fname : str
        Path to the pickle file.

    Returns
    -------
    tuple[list[str], np.ndarray]
        The peptide list and corresponding embeddings.
    """
    with open(fname, "rb") as f:
        _dict = pickle.load(f)
        return _dict["peptide_list"], _dict["pept_embeds"]
