import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import click
import esm
import faiss
import numpy as np
import pandas as pd
import torch
import tqdm

import fennet_mhc
from fennet_mhc.mhc_binding_model import (
    ModelHlaEncoder,
    ModelSeqEncoder,
    embed_hla_esm_list,
    embed_peptides,
)
from fennet_mhc.mhc_binding_retriever import MHCBindingRetriever
from fennet_mhc.mhc_utils import FOUNDATION_MODEL_DIR, NonSpecificDigest, prepare_models


@click.group(
    context_settings=dict(help_option_names=["-h", "--help"]),
    invoke_without_command=True,
)
@click.pass_context
@click.version_option(fennet_mhc.__version__, "-v", "--version")
def run(ctx, **kwargs):
    click.echo(
        rf"""
                   _____                     _
                  |  ___|__ _ __  _ __   ___| |_
                  | |_ / _ \ '_ \| '_ \ / _ \ __|
                  |  _|  __/ | | | | | |  __/ |_
                  |_|  \___|_| |_|_| |_|\___|\__|
        ...................................................
        .{fennet_mhc.__version__.center(50)}.
        .{fennet_mhc.__github__.center(50)}.
        .{fennet_mhc.__license__.center(50)}.
        ...................................................
        """
    )
    if ctx.invoked_subcommand is None:
        click.echo(run.get_help(ctx))


def set_device(device: str) -> str:
    """
    Select the appropriate device based on availability.

    Args:
        device (str): The desired device ('cpu', 'cuda', or 'mps').

    Returns:
        str: The selected device ('cpu', 'cuda', or 'mps').
    """
    if device == "cuda" and not torch.cuda.is_available():
        click.echo("CUDA not available. Change to use CPU.")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        click.echo("MPS (Apple Silicon GPU) not available. Change to use CPU.")
        device = "cpu"

    click.echo(f"Using device: {device}")
    return device


@click.group(
    "mhc",
    context_settings=dict(help_option_names=["-h", "--help"]),
    invoke_without_command=True,
    help="Foundation embedding model for MHC class I molecules and peptides binding prediction",
)
@click.pass_context
def mhc(ctx, **kwargs):
    if ctx.invoked_subcommand is None:
        click.echo(mhc.get_help(ctx))


@mhc.command(
    "embed-proteins", help="Embed MHC class I proteins using Fennet-MHC HLA encoder"
)
@click.option(
    "--fasta",
    type=click.Path(exists=True),
    required=True,
    help="Path to fasta file containing MHC class I protein sequences.",
)
@click.option(
    "--save-pkl-path",
    type=click.Path(),
    required=True,
    help="Path to .pkl Binary file for saving MHC protein embeddings.",
)
@click.option(
    "--hla-model-path",
    type=click.Path(exists=True),
    default=None,
    show_default=False,
    help="Path to model parameter file of HLA encoder module. "
    "If not provided, a default model will be downloaded and used. ",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cuda",
    show_default=True,
    help="Device to use. Options: 'cpu', 'cuda' (for NVIDIA GPUs), or 'mps' (for Apple Silicon GPUs).",
)
def embed_proteins(fasta, save_pkl_path, hla_model_path, device):
    device = set_device(device)

    protein_id_list = []
    protein_seq_list = []
    with open(fasta) as f:
        for line in f.readlines():
            line = line.strip()
            if ">" in line:
                protein_name = line.split(">")[1].split(" ")[0]
                protein_id_list.append(protein_name)
                protein_seq_list.append("")
            else:
                protein_seq_list[-1] += line
    if (not protein_id_list) | (not protein_seq_list):
        click.echo("No valid MHC protein sequences found in fasta file")
        sys.exit(1)

    protein_df = pd.DataFrame({"allele": protein_id_list, "sequence": protein_seq_list})

    esm2_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    esm2_model.to(device)
    esm2_model.eval()
    batch_converter = alphabet.get_batch_converter()

    hla_esm_embedding_list = []
    batch_size = 100

    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(protein_df), batch_size)):
            sequences = protein_df.sequence.values[i : i + batch_size]
            data = list(zip(["_"] * len(sequences), sequences, strict=False))
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            results = esm2_model(
                batch_tokens.to(device), repr_layers=[12], return_contacts=False
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

    hla_encoder = ModelHlaEncoder().to(device)
    if hla_model_path is None:
        default_hla_model_path = os.path.join(FOUNDATION_MODEL_DIR, "HLA_model.pt")
        # check if downloaded
        if os.path.exists(default_hla_model_path):
            hla_model_path = default_hla_model_path
        else:
            hla_model_path, _ = prepare_models()
            if not hla_model_path:
                sys.exit(1)

    try:
        hla_encoder.load_state_dict(
            torch.load(hla_model_path, weights_only=True, map_location=device)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    hla_embeds = embed_hla_esm_list(
        hla_encoder, hla_esm_embedding_list, device=device, verbose=True
    )

    with open(save_pkl_path, "wb") as f:
        pickle.dump(
            {"protein_df": protein_df, "embeds": hla_embeds},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


@mhc.command(
    "embed-peptides-fasta",
    help="Embed peptides that non-specifically digested from fasta using Fennet-MHC peptide encoder",
)
@click.option(
    "--fasta",
    type=click.Path(exists=True),
    required=True,
    help="Path to fasta file.",
)
@click.option(
    "--save-pkl-path",
    type=click.Path(),
    required=True,
    help="Path to .pkl Binary file for saving peptide embeddings.",
)
@click.option(
    "--min-peptide-length",
    type=int,
    default=8,
    show_default=True,
    help="Minimum peptide length.",
)
@click.option(
    "--max-peptide-length",
    type=int,
    default=14,
    show_default=True,
    help="Maximum peptide length.",
)
@click.option(
    "--peptide-model-path",
    type=click.Path(exists=True),
    default=None,
    show_default=False,
    help="Path to peptide model parameter file. "
    "If not provided, a default model will be downloaded and used. ",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cuda",
    show_default=True,
    help="Device to use. Options: 'cpu', 'cuda' (for NVIDIA GPUs), or 'mps' (for Apple Silicon GPUs).",
)
def embed_peptides_fasta(
    fasta,
    save_pkl_path,
    min_peptide_length,
    max_peptide_length,
    peptide_model_path,
    device,
):
    device = set_device(device)

    pept_encoder = ModelSeqEncoder().to(device)
    if peptide_model_path is None:
        default_peptide_model_path = os.path.join(FOUNDATION_MODEL_DIR, "pept_model.pt")
        # check if downloaded
        if os.path.exists(default_peptide_model_path):
            peptide_model_path = default_peptide_model_path
        else:
            _, peptide_model_path = prepare_models()
            if not peptide_model_path:
                sys.exit(1)

    try:
        pept_encoder.load_state_dict(
            torch.load(peptide_model_path, weights_only=True, map_location=device)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    digest = NonSpecificDigest(fasta, (min_peptide_length, max_peptide_length))
    total_peptides_num = len(digest.digest_starts)

    if total_peptides_num == 0:
        click.echo("No valid peptides found in fasta file")
        sys.exit(1)

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
            pept_encoder,
            peptide_list,
            d_model=480,
            batch_size=1024,
            device=device,
        )

        total_pept_embeds = np.concatenate((total_pept_embeds, pept_embeds), axis=0)
        total_peptide_list.extend(peptide_list)

    with open(save_pkl_path, "wb") as f:
        pickle.dump(
            {"peptide_list": total_peptide_list, "pept_embeds": total_pept_embeds},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


@mhc.command(
    "embed-peptides-tsv",
    help="Embed peptides from given tsv using Fennet-MHC peptide encoder",
)
@click.option(
    "--tsv",
    type=click.Path(exists=True),
    required=True,
    help="Path to tsv file containing peptide list.",
)
@click.option(
    "--save-pkl-path",
    type=click.Path(),
    required=True,
    help="Path to .pkl Binary file for saving peptide embeddings.",
)
@click.option(
    "--min-peptide-length",
    type=int,
    default=8,
    show_default=True,
    help="Minimum peptide length.",
)
@click.option(
    "--max-peptide-length",
    type=int,
    default=14,
    show_default=True,
    help="Maximum peptide length.",
)
@click.option(
    "--peptide-model-path",
    type=click.Path(exists=True),
    default=None,
    show_default=False,
    help="Path to peptide model parameter file."
    "If not provided, a default model will be downloaded and used. ",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cuda",
    show_default=True,
    help="Device to use. Options: 'cpu', 'cuda' (for NVIDIA GPUs), or 'mps' (for Apple Silicon GPUs).",
)
def embed_peptides_tsv(
    tsv,
    save_pkl_path,
    min_peptide_length,
    max_peptide_length,
    peptide_model_path,
    device,
):
    device = set_device(device)

    pept_encoder = ModelSeqEncoder().to(device)
    if peptide_model_path is None:
        default_peptide_model_path = os.path.join(FOUNDATION_MODEL_DIR, "pept_model.pt")
        # check if downloaded
        if os.path.exists(default_peptide_model_path):
            peptide_model_path = default_peptide_model_path
        else:
            _, peptide_model_path = prepare_models()
            if not peptide_model_path:
                sys.exit(1)

    try:
        pept_encoder.load_state_dict(
            torch.load(peptide_model_path, weights_only=True, map_location=device)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    input_peptide_df = pd.read_table(tsv, sep="\t", index_col=False)
    before_filter_num = input_peptide_df.shape[0]
    input_peptide_df["peptide_length"] = input_peptide_df.iloc[:, 0].str.len()
    input_peptide_df = input_peptide_df[
        (input_peptide_df["peptide_length"] >= min_peptide_length)
        & (input_peptide_df["peptide_length"] <= max_peptide_length)
    ]
    after_filter_num = input_peptide_df.shape[0]
    if before_filter_num != after_filter_num:
        click.echo(
            f"Filter {before_filter_num-after_filter_num} peptides due to invalid length"
        )
    input_peptide_list = input_peptide_df.iloc[:, 0].tolist()

    if len(input_peptide_list) == 0:
        click.echo("No valid peptides found in tsv file")
        sys.exit(1)

    batch_size = 1000000
    batches = range(0, len(input_peptide_list), batch_size)
    batches = tqdm.tqdm(batches)

    total_pept_embeds = np.empty((0, 480), dtype=np.float32)

    for start_major in batches:
        if start_major + batch_size >= len(input_peptide_list):
            stop_major = len(input_peptide_list)
        else:
            stop_major = start_major + batch_size

        peptide_list = input_peptide_list[start_major:stop_major]

        pept_embeds = embed_peptides(
            pept_encoder,
            peptide_list,
            d_model=480,
            batch_size=1024,
            device=device,
        )

        total_pept_embeds = np.concatenate((total_pept_embeds, pept_embeds), axis=0)

    with open(save_pkl_path, "wb") as f:
        pickle.dump(
            {"peptide_list": input_peptide_list, "pept_embeds": total_pept_embeds},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


@mhc.command(
    "predict-binding-for-MHC",
    help="Predict binding of peptides to MHC class I molecules",
)
@click.option(
    "--peptide-pkl-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to Peptide pre-embeddings file (.pkl).",
)
@click.option(
    "--protein-pkl-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the pre-computed MHC protein embeddings file (.pkl). "
    "A default embeddings file cotaining 15672 alleles is provided. "
    "If your desired alleles are not included in the default file, "
    "you can generate a custom embeddings file using the *embed_proteins* command.",
)
@click.option(
    "--alleles",
    type=str,
    required=True,
    help="List of MHC class I alleles, separated by commas. Example: A01_01,B07_02,C07_02.",
)
@click.option(
    "--out-folder",
    type=click.Path(),
    required=True,
    help="Output folder for the results.",
)
@click.option(
    "--min-peptide-length",
    type=int,
    default=8,
    show_default=True,
    help="Minimum peptide length.",
)
@click.option(
    "--max-peptide-length",
    type=int,
    default=14,
    show_default=True,
    help="Maximum peptide length.",
)
@click.option(
    "--filter-distance",
    type=float,
    default=2,
    show_default=True,
    help="Filter peptide by best allele binding distance.",
)
@click.option(
    "--background-fasta",
    type=click.Path(exists=True),
    required=True,
    help="Path to background human proteins fasta file.",
)
@click.option(
    "--hla-model-path",
    type=click.Path(exists=True),
    default=None,
    show_default=False,
    help="Path to HLA model parameter file. "
    "If not provided, a default model will be downloaded and used. ",
)
@click.option(
    "--peptide-model-path",
    type=click.Path(exists=True),
    default=None,
    show_default=True,
    help="Path to peptide model parameter file. "
    "If not provided, a default model will be downloaded and used. ",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cuda",
    show_default=True,
    help="Device to use. Options: 'cpu', 'cuda' (for NVIDIA GPUs), or 'mps' (for Apple Silicon GPUs).",
)
def predict_binding_for_MHC(
    peptide_pkl_path,
    protein_pkl_path,
    alleles,
    out_folder,
    min_peptide_length,
    max_peptide_length,
    filter_distance,
    background_fasta,
    hla_model_path,
    peptide_model_path,
    device,
):
    # check input peptide source
    try:
        with open(peptide_pkl_path, "rb") as f:
            data_dict = pickle.load(f)
            peptide_list = data_dict["peptide_list"]
            pept_embeds = data_dict["pept_embeds"]
    except Exception as e:
        raise RuntimeError(f"Failed to load Peptide embeddings: {e}") from e

    peptide_lengths = np.array([len(pep) for pep in peptide_list])
    valid_indices = np.where(
        (peptide_lengths >= min_peptide_length)
        & (peptide_lengths <= max_peptide_length)
    )[0]
    peptide_list = [peptide_list[i] for i in valid_indices]
    pept_embeds = pept_embeds[valid_indices, :]

    if len(peptide_list) == 0:
        click.echo("No valid peptide sequences")
        sys.exit(1)

    # check input MHC protein source
    # if protein_pkl_path is None:
    #     protein_pkl_path = os.path.join(CONST_DIR, "hla_v0819_embeds.pkl")

    try:
        with open(protein_pkl_path, "rb") as f:
            data_dict = pickle.load(f)
            protein_df = data_dict["protein_df"]
            hla_embeds = data_dict["embeds"]
    except Exception as e:
        raise RuntimeError(f"Failed to load MHC protein embeddings: {e}") from e

    all_allele_array = protein_df["allele"].unique()
    selected_alleles_array = np.array(alleles.split(","))
    return_check_array = np.isin(selected_alleles_array, all_allele_array)
    exit_flag = False
    for allele, result in zip(selected_alleles_array, return_check_array, strict=False):
        if not result:
            click.echo(f"The allele {allele} is not available.")
            exit_flag = True
    if exit_flag:
        sys.exit(1)

    device = set_device(device)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    pept_encoder = ModelSeqEncoder().to(device)
    hla_encoder = ModelHlaEncoder().to(device)

    if peptide_model_path is None:
        default_peptide_model_path = os.path.join(FOUNDATION_MODEL_DIR, "pept_model.pt")
        if os.path.exists(default_peptide_model_path):
            peptide_model_path = default_peptide_model_path
        else:
            _, peptide_model_path = prepare_models()
            if not peptide_model_path:
                sys.exit(1)

    if hla_model_path is None:
        default_hla_model_path = os.path.join(FOUNDATION_MODEL_DIR, "HLA_model.pt")
        if os.path.exists(default_hla_model_path):
            hla_model_path = default_hla_model_path
        else:
            hla_model_path, _ = prepare_models()
            if not hla_model_path:
                sys.exit(1)

    try:
        hla_encoder.load_state_dict(
            torch.load(hla_model_path, weights_only=True, map_location=device)
        )
        pept_encoder.load_state_dict(
            torch.load(peptide_model_path, weights_only=True, map_location=device)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    retriever = MHCBindingRetriever(
        hla_encoder,
        pept_encoder,
        protein_df,
        hla_embeds,
        background_fasta,
        digested_pept_lens=(min_peptide_length, max_peptide_length),
    )
    peptide_df = retriever.get_binding_metrics_for_peptides(
        selected_alleles_array, pept_embeds
    )
    peptide_df["sequence"] = peptide_list
    peptide_df = peptide_df.drop(columns=["best_allele_id"], errors="ignore")
    peptide_df = peptide_df[["sequence", "best_allele", "best_allele_dist"]]
    # peptide_df = peptide_df[["sequence", "best_allele", "best_allele_dist", "best_allele_rank"]]

    peptide_df = peptide_df[
        peptide_df["best_allele_dist"] <= filter_distance
    ].sort_values(by="best_allele_dist", ascending=True)

    peptide_df = peptide_df.round(3)

    output_dir = Path(out_folder)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_file_path = output_dir.joinpath(f"peptide_df_for_MHC_{current_time}.tsv")
    peptide_df.to_csv(output_file_path, sep="\t", index=False)
    print(f"File saved to: {output_file_path}")


@mhc.command(
    "predict-binding-for-epitope",
    help="Predict binding of MHC class I molecules to epitope",
)
@click.option(
    "--peptide-pkl-path",
    type=click.Path(exists=True),
    help="Path to Peptide pre-embeddings file (.pkl).",
)
@click.option(
    "--protein-pkl-path",
    type=click.Path(exists=True),
    default=None,
    show_default=False,
    help="Path to the pre-computed MHC protein embeddings file (.pkl). "
    "If not provided, a default embeddings file will be used. "
    "If your desired alleles are not included in the default file, "
    "you can generate a custom embeddings file using the *embed_proteins* command.",
)
@click.option(
    "--out-folder",
    type=click.Path(),
    required=True,
    help="Output folder for the results.",
)
@click.option(
    "--min-peptide-length",
    type=int,
    default=8,
    show_default=True,
    help="Minimum peptide length.",
)
@click.option(
    "--max-peptide-length",
    type=int,
    default=14,
    show_default=True,
    help="Maximum peptide length.",
)
@click.option(
    "--filter-distance",
    type=float,
    default=2,
    show_default=True,
    help="Filter by binding distance.",
)
@click.option(
    "--background-fasta",
    type=click.Path(exists=True),
    required=True,
    help="Path to background human proteins fasta file.",
)
@click.option(
    "--hla-model-path",
    type=click.Path(exists=True),
    default=None,
    show_default=False,
    help="Path to HLA model parameter file. "
    "If not provided, a default model will be downloaded and used. ",
)
@click.option(
    "--peptide-model-path",
    type=click.Path(exists=True),
    default=None,
    show_default=False,
    help="Path to peptide model parameter file. "
    "If not provided, a default model will be downloaded and used. ",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cuda",
    show_default=True,
    help="Device to use. Options: 'cpu', 'cuda' (for NVIDIA GPUs), or 'mps' (for Apple Silicon GPUs).",
)
def predict_binding_for_epitope(
    peptide_pkl_path,
    protein_pkl_path,
    out_folder,
    min_peptide_length,
    max_peptide_length,
    filter_distance,
    background_fasta,
    hla_model_path,
    peptide_model_path,
    device,
):
    # check input peptide source
    try:
        with open(peptide_pkl_path, "rb") as f:
            data_dict = pickle.load(f)
            peptide_list = data_dict["peptide_list"]
            pept_embeds = data_dict["pept_embeds"]
    except Exception as e:
        raise RuntimeError(f"Failed to load Peptide embeddings: {e}") from e

    peptide_lengths = np.array([len(pep) for pep in peptide_list])
    valid_indices = np.where(
        (peptide_lengths >= min_peptide_length)
        & (peptide_lengths <= max_peptide_length)
    )[0]
    peptide_list = [peptide_list[i] for i in valid_indices]
    pept_embeds = pept_embeds[valid_indices, :]

    if len(peptide_list) == 0:
        click.echo("No valid peptide sequences")
        sys.exit(1)

    # check input MHC protein source
    # if protein_pkl_path is None:
    #     protein_pkl_path = os.path.join(CONST_DIR, "hla_v0819_embeds.pkl")

    try:
        with open(protein_pkl_path, "rb") as f:
            data_dict = pickle.load(f)
            protein_df = data_dict["protein_df"]
            hla_embeds = data_dict["embeds"]
    except Exception as e:
        raise RuntimeError(f"Failed to load MHC protein embeddings: {e}") from e

    device = set_device(device)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    pept_encoder = ModelSeqEncoder().to(device)
    hla_encoder = ModelHlaEncoder().to(device)

    if peptide_model_path is None:
        default_peptide_model_path = os.path.join(FOUNDATION_MODEL_DIR, "pept_model.pt")
        if os.path.exists(default_peptide_model_path):
            peptide_model_path = default_peptide_model_path
        else:
            _, peptide_model_path = prepare_models()
            if not peptide_model_path:
                sys.exit(1)

    if hla_model_path is None:
        default_hla_model_path = os.path.join(FOUNDATION_MODEL_DIR, "HLA_model.pt")
        if os.path.exists(default_hla_model_path):
            hla_model_path = default_hla_model_path
        else:
            hla_model_path, _ = prepare_models()
            if not hla_model_path:
                sys.exit(1)

    try:
        hla_encoder.load_state_dict(
            torch.load(hla_model_path, weights_only=True, map_location=device)
        )
        pept_encoder.load_state_dict(
            torch.load(peptide_model_path, weights_only=True, map_location=device)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    retriever = MHCBindingRetriever(
        hla_encoder,
        pept_encoder,
        protein_df,
        hla_embeds,
        background_fasta,
        digested_pept_lens=(min_peptide_length, max_peptide_length),
    )

    all_allele_array = protein_df["allele"].tolist()

    ret_dists = retriever.get_embedding_distances(hla_embeds, pept_embeds)
    best_peptide_idxes = np.argmin(ret_dists, axis=0)
    best_peptide_dists = ret_dists[best_peptide_idxes, np.arange(ret_dists.shape[1])]
    best_peptide_list = [peptide_list[k] for k in best_peptide_idxes]

    allele_df = pd.DataFrame(
        {
            "allele": all_allele_array,
            "best_peptide": best_peptide_list,
            "best_peptide_dist": best_peptide_dists,
        }
    )
    allele_df = allele_df[allele_df["best_peptide_dist"] <= filter_distance]
    allele_df = allele_df.round(3)
    allele_df.sort_values("allele", ascending=True, inplace=True)

    output_dir = Path(out_folder)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_file_path = output_dir.joinpath(f"allele_df_for_epitope_{current_time}.tsv")
    allele_df.to_csv(output_file_path, sep="\t", index=False)
    print(f"File saved to: {output_file_path}")


@mhc.command(
    "deconvolute-peptides",
    help="Peptides deconvolution to clusters with corresponding binding motifs.",
)
@click.option(
    "--peptide-pkl-path",
    type=click.Path(exists=True),
    help="Path to Peptide pre-embeddings file (.pkl).",
)
@click.option(
    "--n-centroids",
    type=int,
    default=8,
    show_default=True,
    help="Number of kmeans centroids to cluster. It's better to add 1-2 to the number you expect, otherwise; some outliers may affect the clustering.",
)
@click.option(
    "--out-folder",
    type=click.Path(),
    required=True,
    help="Output folder for the results.",
)
@click.option(
    "--peptide-model-path",
    type=click.Path(exists=True),
    default=None,
    show_default=False,
    help="Path to peptide model parameter file. "
    "If not provided, a default model will be downloaded and used. ",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cuda",
    show_default=True,
    help="Device to use. Options: 'cpu', 'cuda' (for NVIDIA GPUs), or 'mps' (for Apple Silicon GPUs).",
)
def deconvolute_peptides(
    peptide_pkl_path,
    n_centroids,
    out_folder,
    peptide_model_path,
    device,
):
    # check input peptide source
    try:
        with open(peptide_pkl_path, "rb") as f:
            data_dict = pickle.load(f)
            peptide_list = data_dict["peptide_list"]
            pept_embeds = data_dict["pept_embeds"]
    except Exception as e:
        raise RuntimeError(f"Failed to load Peptide embeddings: {e}") from e

    if len(peptide_list) == 0:
        click.echo("No valid peptide sequences")
        sys.exit(1)

    device = set_device(device)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    pept_encoder = ModelSeqEncoder().to(device)

    if peptide_model_path is None:
        default_peptide_model_path = os.path.join(FOUNDATION_MODEL_DIR, "pept_model.pt")
        if os.path.exists(default_peptide_model_path):
            peptide_model_path = default_peptide_model_path
        else:
            _, peptide_model_path = prepare_models()
            if not peptide_model_path:
                sys.exit(1)

    try:
        pept_encoder.load_state_dict(
            torch.load(peptide_model_path, weights_only=True, map_location=device)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    d = pept_embeds.shape[1]

    kmeans = faiss.Kmeans(d, n_centroids)
    kmeans.niter = 20
    kmeans.verbose = True
    kmeans.min_points_per_centroid = 10
    kmeans.max_points_per_centroid = 1000

    kmeans.train(pept_embeds)
    centroids = kmeans.centroids
    trained_index = faiss.IndexFlatL2(d)
    trained_index.add(centroids)

    return_dists, return_labels = trained_index.search(pept_embeds, 1)
    cluster_labels = return_labels.flatten()
    cluster_dists = return_dists.flatten()

    cluster_df = pd.DataFrame(
        {
            "sequence": peptide_list,
            "cluster_label": cluster_labels,
            "cluster_dist": cluster_dists,
        }
    )

    output_dir = Path(out_folder)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_file_path = output_dir.joinpath(
        f"peptides_deconvolution_cluster_df_{current_time}.tsv"
    )
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


if __name__ == "__main__":
    run()
