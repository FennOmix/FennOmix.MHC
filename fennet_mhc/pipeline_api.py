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

from fennet_mhc.constants._const import (
    BACKGROUND_FASTA_PATH,
    FENNETMHC_MODEL_DIR,
    HLA_EMBEDDING_PATH,
    HLA_MODEL_PATH,
    PEPTIDE_MODEL_PATH,
    global_settings,
)
from fennet_mhc.mhc_binding_model import (
    ModelHlaEncoder,
    ModelSeqEncoder,
    embed_hla_esm_list,
    embed_peptides,
)
from fennet_mhc.mhc_binding_retriever import MHCBindingRetriever
from fennet_mhc.mhc_utils import NonSpecificDigest


class PretrainedModels:
    def __init__(self, device: str = "cuda"):
        self.device = _set_device(device)
        _download_pretrained_models()
        self.hla_encoder = self._load_hla_model()
        self.pept_encoder = self._load_peptide_model()
        self.hla_encoder.eval()
        self.pept_encoder.eval()

        self.esm2_model, self.esm2_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.esm2_model.to(device)
        self.esm2_model.eval()
        self.batch_converter = self.esm2_alphabet.get_batch_converter()

        self.background_protein_df = load_fasta_list_as_protein_df(
            [BACKGROUND_FASTA_PATH]
        )
        self.hla_df, self.hla_embeddings = _load_hla_embedding_pkl(HLA_EMBEDDING_PATH)
        self.hla_df.reset_index(drop=True, inplace=True)

    def embed_proteins(self, fasta: str):
        if not os.path.exists(fasta):
            raise FileNotFoundError(f"Fasta file not found: {fasta}")
        protein_df = load_fasta_list_as_protein_df([fasta])
        protein_df.rename(columns={"protein_id": "allele"}, inplace=True)

        hla_esm_embedding_list = []
        batch_size = 100

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
        digest = NonSpecificDigest(fasta, (min_peptide_length, max_peptide_length))
        total_peptides_num = len(digest.digest_starts)

        if total_peptides_num == 0:
            raise ValueError("No valid peptides found in fasta file")

        batch_size = 1000000
        batches = range(0, total_peptides_num, batch_size)
        batches = tqdm.tqdm(batches)

        total_peptide_list = []
        total_pept_embeds = np.empty((0, 480), dtype=np.float32)

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
                d_model=480,
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

        total_pept_embeds = np.empty((0, self.esm2_model.embed_dim), dtype=np.float32)

        for start_major in batches:
            if start_major + batch_size >= len(input_peptide_list):
                stop_major = len(input_peptide_list)
            else:
                stop_major = start_major + batch_size

            peptide_list = input_peptide_list[start_major:stop_major]

            pept_embeds = embed_peptides(
                self.pept_encoder,
                peptide_list,
                d_model=self.esm2_model.embed_dim,
                batch_size=1024,
                device=self.device,
            )

            total_pept_embeds = np.concatenate((total_pept_embeds, pept_embeds), axis=0)

        return input_peptide_list, total_pept_embeds

    def predict_MHC_binders_for_epitopes(
        self,
        peptide_list: list,
        peptide_embeddings: np.ndarray,
        hla_df: pd.DataFrame = None,
        hla_embeddings: np.ndarray = None,
        min_peptide_length: int = 8,
        max_peptide_length: int = 12,
        distance_threshold: float = 0.4,
    ):
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
            digested_pept_lens=(min_peptide_length, max_peptide_length),
        )

        all_allele_array = hla_df["allele"].tolist()

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
        allele_df = allele_df[allele_df["best_peptide_dist"] <= distance_threshold]
        allele_df.sort_values("allele", ascending=True, inplace=True, ignore_index=True)

        return allele_df

    def predict_peptide_binders_for_MHC(
        self,
        peptide_list: list,
        peptide_embeddings: np.ndarray,
        alleles: list,
        hla_df: pd.DataFrame = None,
        hla_embeddings: np.ndarray = None,
        min_peptide_length: int = 8,
        max_peptide_length: int = 12,
        distance_threshold: float = 0.4,
    ):
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

        all_allele_array = hla_df["allele"].unique()
        alleles = np.array(alleles.split(","))
        return_check_array = np.isin(alleles, all_allele_array)
        selected_alleles = []
        for allele, found in zip(alleles, return_check_array, strict=False):
            if found:
                selected_alleles.append(allele)
            else:
                logging.warning(
                    f"Ignore allele '{allele}' which is not available in allele db."
                )

        retriever = MHCBindingRetriever(
            self.hla_encoder,
            self.pept_encoder,
            hla_df,
            hla_embeddings,
            self.background_protein_df,
            digested_pept_lens=(min_peptide_length, max_peptide_length),
        )
        peptide_df = retriever.get_binding_metrics_for_peptides(
            selected_alleles, peptide_embeddings
        )
        peptide_df["sequence"] = peptide_list
        peptide_df = peptide_df.drop(columns=["best_allele_id"], errors="ignore")
        peptide_df = peptide_df[["sequence", "best_allele", "best_allele_dist"]]

        peptide_df = peptide_df[
            peptide_df["best_allele_dist"] <= distance_threshold
        ].sort_values(by="best_allele_dist", ascending=True, ignore_index=True)

        return peptide_df

    def deconvolute_peptides(
        peptide_list: list,
        pept_embeddings: np.ndarray,
        n_centroids: int = 8,
    ):
        d = pept_embeddings.shape[1]

        kmeans = faiss.Kmeans(d, n_centroids)
        kmeans.niter = 20
        kmeans.verbose = True
        kmeans.min_points_per_centroid = 10
        kmeans.max_points_per_centroid = 1000

        kmeans.train(pept_embeddings)
        centroids = kmeans.centroids
        trained_index = faiss.IndexFlatL2(d)
        trained_index.add(centroids)

        return_dists, return_labels = trained_index.search(pept_embeddings, 1)
        cluster_labels = return_labels.flatten()
        cluster_dists = return_dists.flatten()

        return pd.DataFrame(
            {
                "sequence": peptide_list,
                "cluster_id": cluster_labels,
                "dist_to_cluster": cluster_dists,
            }
        ), centroids

    def _load_hla_model(self):
        hla_encoder = ModelHlaEncoder()
        hla_encoder.to(self.device)
        hla_encoder.load_state_dict(
            torch.load(HLA_MODEL_PATH, weights_only=True, map_location=self.device)
        )
        return hla_encoder

    def _load_peptide_model(self):
        pept_encoder = ModelSeqEncoder()
        pept_encoder.to(self.device)
        pept_encoder.load_state_dict(
            torch.load(PEPTIDE_MODEL_PATH, weights_only=True, map_location=self.device)
        )
        return pept_encoder


def embed_proteins(fasta: str, save_pkl_path: str, device: str = "cuda"):
    set_logger(log_file_name=global_settings["log_file_name"])

    pretrained_models = PretrainedModels(device=device)

    protein_df, hla_embeds = pretrained_models.embed_proteins(fasta)

    with open(save_pkl_path, "wb") as f:
        pickle.dump(
            {"protein_df": protein_df, "embeds": hla_embeds},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def embed_peptides_from_file(
    peptide_file_path: str,
    save_pkl_path: str,
    min_peptide_length: int = 8,
    max_peptide_length: int = 12,
    device: str = "cuda",
):
    set_logger(log_file_name=global_settings["log_file_name"])

    pretrained_models = PretrainedModels(device=device)

    if peptide_file_path.lower().endswith(".fasta"):
        peptide_list, peptide_embeds = pretrained_models.embed_peptides_from_fasta(
            peptide_file_path, min_peptide_length, max_peptide_length
        )
    elif peptide_file_path[-4:].lower() in [".tsv", ".txt", "csv"]:
        peptide_list, peptide_embeds = pretrained_models.embed_peptides_tsv(
            peptide_file_path, min_peptide_length, max_peptide_length
        )

    with open(save_pkl_path, "wb") as f:
        pickle.dump(
            {"peptide_list": peptide_list, "pept_embeds": peptide_embeds},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def predict_peptide_binders_for_MHC(
    peptide_file_path: str,
    alleles: list,
    out_folder: str,
    min_peptide_length: int = 8,
    max_peptide_length: int = 12,
    distance_threshold: float = 0.4,
    hla_file_path: str = None,
    device: str = "cuda",
):
    set_logger(log_file_name=global_settings["log_file_name"])

    pretrained_models = PretrainedModels(device=device)

    peptide_list, pept_embeds = _load_peptide_embeddings(
        pretrained_models, peptide_file_path, min_peptide_length, max_peptide_length
    )

    protein_df, hla_embeds = _load_protein_embeddings(pretrained_models, hla_file_path)

    peptide_df = pretrained_models.predict_peptide_binders_for_MHC(
        peptide_list,
        pept_embeds,
        alleles=alleles,
        hla_df=protein_df,
        hla_embeddings=hla_embeds,
        min_peptide_length=min_peptide_length,
        max_peptide_length=max_peptide_length,
        distance_threshold=distance_threshold,
    )

    peptide_df = peptide_df.round(3)
    os.makedirs(out_folder, exist_ok=True)
    output_dir = Path(out_folder)
    output_file_path = output_dir.joinpath("peptide_df_for_MHC.tsv")
    peptide_df.to_csv(output_file_path, sep="\t", index=False)
    print(f"File saved to: {output_file_path}")


def predict_binders_for_epitopes(
    peptide_file_path: str,
    out_folder: str,
    min_peptide_length: int = 8,
    max_peptide_length: int = 12,
    distance_threshold: float = 0.4,
    hla_file_path: str = None,
    device: str = "cuda",
):
    set_logger(log_file_name=global_settings["log_file_name"])

    pretrained_models = PretrainedModels(device=device)

    peptide_list, pept_embeds = _load_peptide_embeddings(
        pretrained_models, peptide_file_path, min_peptide_length, max_peptide_length
    )

    protein_df, hla_embeds = _load_protein_embeddings(pretrained_models, hla_file_path)

    allele_df = pretrained_models.predict_MHC_binders_for_epitopes(
        peptide_list,
        pept_embeds,
        hla_df=protein_df,
        hla_embeddings=hla_embeds,
        min_peptide_length=min_peptide_length,
        max_peptide_length=max_peptide_length,
        distance_threshold=distance_threshold,
    )

    allele_df = allele_df.round(3)
    os.makedirs(out_folder, exist_ok=True)
    output_dir = Path(out_folder)
    output_file_path = output_dir.joinpath("allele_df_for_epitopes.tsv")
    allele_df.to_csv(output_file_path, sep="\t", index=False)
    logging.info(f"File saved to: {output_file_path}")


def deconvolute_peptides(
    peptide_file_path: str,
    n_centroids: int,
    out_folder: str,
    min_peptide_length: int = 8,
    max_peptide_length: int = 12,
    hla_file_path: str = None,
    device: str = "cuda",
):
    set_logger(log_file_name=global_settings["log_file_name"])

    pretrained_models = PretrainedModels(device=device)

    peptide_list, pept_embeds = _load_peptide_embeddings(
        pretrained_models, peptide_file_path, min_peptide_length, max_peptide_length
    )

    protein_df, hla_embeds = _load_protein_embeddings(pretrained_models, hla_file_path)

    cluster_df, centroids = pretrained_models.deconvolute_peptides(
        peptide_list,
        pept_embeds,
        n_centroids,
    )

    d = centroids.shape[1]
    trained_index = faiss.IndexFlatL2(d)
    trained_index.add(hla_embeds)
    dists, idxes = trained_index.search(centroids, 1)
    closest_alleles = protein_df["allele"].values[idxes.flatten()]

    cluster_df["closest_allele"] = closest_alleles[cluster_df["cluster_id"].values]
    cluster_df["closest_allele_dist"] = dists.flatten()[cluster_df["cluster_id"].values]

    cluster_df = cluster_df.round(3)
    os.makedirs(out_folder, exist_ok=True)
    output_dir = Path(out_folder)
    output_file_path = output_dir.joinpath("peptide_deconvolution_cluster_df.tsv")
    cluster_df.to_csv(output_file_path, sep="\t", index=False)

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
    base_url: str = None, model_dir: str = FENNETMHC_MODEL_DIR
):
    """
    Download pretrained models from a given URL.

    Args:
        base_url (str): The base URL to download the models from.
        model_dir (str): The directory to save the downloaded models.
    """
    if base_url is None:
        base_url = global_settings["hla_url"]
    base_url += "" if base_url.endswith("/") else "/"
    os.makedirs(model_dir, exist_ok=True)

    peptide_url = base_url + global_settings["peptide_model"]
    hla_url = base_url + global_settings["hla_model"]
    hla_embedding_url = base_url + global_settings["hla_embedding"]
    background_fasta_url = base_url + global_settings["background_fasta"]

    if os.path.exists(HLA_MODEL_PATH) and os.path.exists(PEPTIDE_MODEL_PATH):
        logging.info("Pretrained models already exist. Skipping download.")
        return

    logging.info(
        f"Downloading required files ...:\n"
        f"  `{global_settings['peptide_model']} from `{peptide_url}`\n"
        f"  `{global_settings['hla_model']} from`{hla_url}`\n"
        f"  `{global_settings['hla_embedding']} from`{hla_embedding_url}`\n"
        f"  `{global_settings['background_fasta']} from`{background_fasta_url}`"
    )
    try:
        context = ssl._create_unverified_context()
        requests = urllib.request.urlopen(peptide_url, context=context, timeout=10)
        with open(PEPTIDE_MODEL_PATH, "wb") as f:
            f.write(requests.read())

        requests = urllib.request.urlopen(hla_url, context=context, timeout=10)
        with open(HLA_MODEL_PATH, "wb") as f:
            f.write(requests.read())

        requests = urllib.request.urlopen(
            hla_embedding_url, context=context, timeout=10
        )
        with open(HLA_EMBEDDING_PATH, "wb") as f:
            f.write(requests.read())

        requests = urllib.request.urlopen(
            background_fasta_url, context=context, timeout=10
        )
        with open(BACKGROUND_FASTA_PATH, "wb") as f:
            f.write(requests.read())
    except Exception as e:
        raise RuntimeError(f"Failed to download models: {e}") from e


def _load_protein_embeddings(pretrained_models: PretrainedModels, hla_file_path):
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
    if peptide_file_path[-4:].lower() in [".tsv", ".txt", "csv"]:
        return pretrained_models.embed_peptides_tsv(
            peptide_file_path, min_peptide_length, max_peptide_length
        )
    raise ValueError(
        f"Unsupported peptide file format: {peptide_file_path}. "
        "Please provide a .pkl or .tsv (.csv) file."
    )


def _load_hla_embedding_pkl(fname=None):
    if fname is None:
        fname = HLA_EMBEDDING_PATH
    if not os.path.exists(fname):
        raise FileNotFoundError(f".pkl file not found: {fname}")
    with open(fname, "rb") as f:
        _dict = pickle.load(f)
        return _dict["protein_df"], _dict["embeds"]


def load_peptide_embedding_pkl(fname):
    with open(fname, "rb") as f:
        _dict = pickle.load(f)
        return _dict["peptide_list"], _dict["pept_embeds"]
