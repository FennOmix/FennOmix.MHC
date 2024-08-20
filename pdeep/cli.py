import os
import pickle
import sys
from pathlib import Path

import click
import esm
import numpy as np
import pandas as pd
import torch
import tqdm

import pdeep
from pdeep.mhc.mhc_binding_model import (
    ModelHlaEncoder,
    ModelSeqEncoder,
    embed_hla_esm_list,
    embed_peptides,
)
from pdeep.mhc.mhc_binding_retriever import MHCBindingRetriever
from pdeep.mhc.mhc_utils import NonSpecificDigest


@click.group(
    context_settings=dict(help_option_names=["-h", "--help"]),
    invoke_without_command=True,
)
@click.pass_context
@click.version_option(pdeep.__version__, "-v", "--version")
def run(ctx, **kwargs):
    click.echo(
        rf"""
                          ____
                    _ __ |  _ \  ___  ___ _ __
                   | '_ \| | | |/ _ \/ _ \ '_ \
                   | |_) | |_| |  __/  __/ |_) |
                   | .__/|____/ \___|\___| .__/
                   |_|                   |_|
        ...................................................
        .{pdeep.__version__.center(50)}.
        .{pdeep.__github__.center(50)}.
        .{pdeep.__license__.center(50)}.
        ...................................................
        """
    )
    if ctx.invoked_subcommand is None:
        click.echo(run.get_help(ctx))


@run.group(
    "mhc",
    context_settings=dict(help_option_names=["-h", "--help"]),
    invoke_without_command=True,
    help="Joint embedding of MHC molecules and immunopeptides",
)
@click.pass_context
def mhc(ctx, **kwargs):
    if ctx.invoked_subcommand is None:
        click.echo(mhc.get_help(ctx))


@mhc.command("embed_proteins", help="Embed MHC proteins using pDeepMHC HLA encoder")
@click.option(
    "--fasta",
    type=click.Path(exists=True),
    required=True,
    help="Path to fasta file containing MHC class I protein sequences",
)
@click.option(
    "--save_pkl",
    type=click.Path(),
    required=True,
    help="Path to .pkl Binary file for saving MHC protein embeddings",
)
@click.option(
    "--load_model_hla",
    type=click.Path(exists=True),
    default="./model/HLA_model_v0819.pt",
    show_default=True,
    help="Path to model parameter file of HLA encoder module.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default="cuda",
    show_default=True,
    help="Device to use",
)
def embed_proteins(fasta, save_pkl, load_model_hla, device):
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
            data = list(zip(["_"] * len(sequences), sequences))
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

    try:
        hla_encoder.load_state_dict(
            torch.load(load_model_hla, weights_only=True, map_location=device)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    hla_embeds = embed_hla_esm_list(
        hla_encoder, hla_esm_embedding_list, device=device, verbose=True
    )

    with open(save_pkl, "wb") as f:
        pickle.dump(
            {"protein_df": protein_df, "embeds": hla_embeds},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


@mhc.command(
    "embed_peptides_fasta",
    help="Embed peptides from fasta using pDeepMHC peptide encoder",
)
@click.option(
    "--fasta",
    type=click.Path(exists=True),
    required=True,
    help="Path to fasta file",
)
@click.option(
    "--save_pkl",
    type=click.Path(),
    required=True,
    help="Path to .pkl Binary file for saving peptide embeddings",
)
@click.option(
    "--min_peptide_length",
    type=int,
    default=8,
    show_default=True,
    help="Minimum peptide length.",
)
@click.option(
    "--max_peptide_length",
    type=int,
    default=14,
    show_default=True,
    help="Maximum peptide length.",
)
@click.option(
    "--load_model_pept",
    type=click.Path(exists=True),
    default="./model/pept_model_v0819.pt",
    show_default=True,
    help="Path to peptide model parameter file.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default="cuda",
    show_default=True,
    help="Device to use",
)
def embed_peptides_fasta(
    fasta, save_pkl, min_peptide_length, max_peptide_length, load_model_pept, device
):
    pept_encoder = ModelSeqEncoder().to(device)
    pept_encoder.load_state_dict(
        torch.load(load_model_pept, weights_only=True, map_location=device)
    )

    digest = NonSpecificDigest(fasta, (min_peptide_length, max_peptide_length))
    total_peptides_num = len(digest.digest_starts)

    if total_peptides_num == 0:
        click.echo("No valid peptide sequences found in fasta file")
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

    with open(save_pkl, "wb") as f:
        pickle.dump(
            {"peptide_list": total_peptide_list, "pept_embeds": total_pept_embeds},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


@mhc.command("embed_peptides_tsv", help="Embed peptides from given tsv")
@click.option(
    "--tsv",
    type=click.Path(exists=True),
    default="",
    help="Path to tsv file containing peptides list. (Mutually exclusive with the previous --fasta option, don't provide fasta file and tsv file at the same time)",
)
@click.option(
    "--save_pkl",
    type=click.Path(),
    required=True,
    help="Path to .pkl Binary file for saving peptide embeddings",
)
@click.option(
    "--min_peptide_length",
    type=int,
    default=8,
    show_default=True,
    help="Minimum peptide length.",
)
@click.option(
    "--max_peptide_length",
    type=int,
    default=14,
    show_default=True,
    help="Maximum peptide length.",
)
@click.option(
    "--load_model_pept",
    type=click.Path(exists=True),
    default="./model/pept_model_v0819.pt",
    show_default=True,
    help="Path to peptide model parameter file.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default="cuda",
    show_default=True,
    help="Device to use",
)
def embed_peptides_tsv(
    tsv, save_pkl, min_peptide_length, max_peptide_length, load_model_pept, device
):
    pept_encoder = ModelSeqEncoder().to(device)
    pept_encoder.load_state_dict(
        torch.load(load_model_pept, weights_only=True, map_location=device)
    )

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
        click.echo("No valid peptide sequences found in tsv file")
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

    with open(save_pkl, "wb") as f:
        pickle.dump(
            {"peptide_list": input_peptide_list, "pept_embeds": total_pept_embeds},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


@mhc.command(
    "predict_binding_for_MHC",
    help="Predict binding of peptides to MHC class I molecules",
)
@click.option(
    "--peptide_pkl",
    type=click.Path(exists=True),
    help="Path to Peptide pre-embeddings file (.pkl)",
)
@click.option(
    "--protein_pkl",
    type=click.Path(exists=True),
    default="./embeds/hla_v0819_embeds.pkl",
    show_default=True,
    help="Path to MHC protein pre-embeddings file (.pkl), If the alleles you want do not exist in the our list, "
    "you can provide the sequences yourself and use the *embed_proteins* function to generate custom protein pkl file",
)
@click.option(
    "--alleles",
    type=str,
    required=True,
    help="List of MHC class I alleles, separated by commas. Example: A01_01,B07_02,C07_02.",
)
@click.option(
    "--fasta",
    type=click.Path(exists=True),
    default="./uniprotkb_UP000005640_AND_reviewed_true_2024_03_01.fasta",
    show_default=True,
    help="Path to human reviewed protein fasta file",
)
@click.option(
    "--out-folder",
    type=click.Path(),
    required=True,
    help="Output folder for the results.",
)
@click.option(
    "--min_peptide_length",
    type=int,
    default=8,
    show_default=True,
    help="Minimum peptide length.",
)
@click.option(
    "--max_peptide_length",
    type=int,
    default=14,
    show_default=True,
    help="Maximum peptide length.",
)
@click.option(
    "--filter_distance",
    type=float,
    default=2,
    show_default=True,
    help="Filter peptide by best allele embedding distance.",
)
@click.option(
    "--load_model_hla",
    type=click.Path(exists=True),
    default="./model/HLA_model_v0819.pt",
    show_default=True,
    help="Path to HLA model parameter file.",
)
@click.option(
    "--load_model_pept",
    type=click.Path(exists=True),
    default="./model/pept_model_v0819.pt",
    show_default=True,
    help="Path to peptide model parameter file.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default="cuda",
    show_default=True,
    help="Device to use",
)
def predict_binding(
    peptide_pkl,
    protein_pkl,
    alleles,
    fasta,
    out_folder,
    min_peptide_length,
    max_peptide_length,
    filter_distance,
    load_model_hla,
    load_model_pept,
    device,
):
    # check input peptide source
    try:
        with open(peptide_pkl, "rb") as f:
            data_dict = pickle.load(f)
            peptide_list = data_dict["peptide_list"]
            pept_embeds = data_dict["pept_embeds"]
    except Exception as e:
        raise RuntimeError(f"Failed to load Peptide embeddings: {e}") from e

    # check input MHC protein source
    try:
        with open(protein_pkl, "rb") as f:
            data_dict = pickle.load(f)
            protein_df = data_dict["protein_df"]
            hla_embeds = data_dict["embeds"]
    except Exception as e:
        raise RuntimeError(f"Failed to load MHC protein embeddings: {e}") from e

    all_allele_array = protein_df["allele"].unique()
    selected_alleles_array = np.array(alleles.split(","))
    return_check_array = np.isin(selected_alleles_array, all_allele_array)
    exit_flag = False
    for allele, result in zip(selected_alleles_array, return_check_array):
        if not result:
            click.echo(f"The allele {allele} is not available.")
            exit_flag = True
    if exit_flag:
        sys.exit(1)

    if device == "cuda" and not torch.cuda.is_available():
        click.echo("CUDA not available. Change to use CPU")
        device = "cpu"

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    pept_encoder = ModelSeqEncoder().to(device)
    hla_encoder = ModelHlaEncoder().to(device)

    try:
        hla_encoder.load_state_dict(
            torch.load(load_model_hla, weights_only=True, map_location=device)
        )
        pept_encoder.load_state_dict(
            torch.load(load_model_pept, weights_only=True, map_location=device)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    retriever = MHCBindingRetriever(
        hla_encoder,
        pept_encoder,
        protein_df,
        hla_embeds,
        fasta,
        digested_pept_lens=(min_peptide_length, max_peptide_length),
    )
    peptide_df = retriever.get_binding_metrics_for_peptides(
        selected_alleles_array, pept_embeds
    )
    peptide_df["sequence"] = peptide_list

    peptide_df = peptide_df[
        peptide_df["best_allele_dist"] <= filter_distance
    ].sort_values(by="best_allele_dist", ascending=True)

    output_dir = Path(out_folder)
    output_file_path = output_dir.joinpath("peptide_df_for_MHC.tsv")
    peptide_df.to_csv(output_file_path, sep="\t", index=False)


@mhc.command(
    "predict_binding_for_epitope",
    help="Predict binding of MHC class I molecules to epitope",
)
@click.option(
    "--peptide_pkl",
    type=click.Path(exists=True),
    help="Path to Peptide pre-embeddings file (.pkl)",
)
@click.option(
    "--protein_pkl",
    type=click.Path(exists=True),
    default="./embeds/hla_v0819_embeds.pkl",
    show_default=True,
    help="Path to MHC protein pre-embeddings file (.pkl), If the alleles you want do not exist in the our list, "
    "you can provide the sequences yourself and use the *embed_proteins* function to generate custom protein pkl file",
)
@click.option(
    "--fasta",
    type=click.Path(exists=True),
    default="./uniprotkb_UP000005640_AND_reviewed_true_2024_03_01.fasta",
    show_default=True,
    help="Path to human reviewed protein fasta file",
)
@click.option(
    "--out-folder",
    type=click.Path(),
    required=True,
    help="Output folder for the results.",
)
@click.option(
    "--min_peptide_length",
    type=int,
    default=8,
    show_default=True,
    help="Minimum peptide length.",
)
@click.option(
    "--max_peptide_length",
    type=int,
    default=14,
    show_default=True,
    help="Maximum peptide length.",
)
@click.option(
    "--load_model_hla",
    type=click.Path(exists=True),
    default="./model/HLA_model_v0819.pt",
    show_default=True,
    help="Path to HLA model parameter file.",
)
@click.option(
    "--load_model_pept",
    type=click.Path(exists=True),
    default="./model/pept_model_v0819.pt",
    show_default=True,
    help="Path to peptide model parameter file.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default="cuda",
    show_default=True,
    help="Device to use",
)
def predict_binding_for_epitope(
    peptide_pkl,
    protein_pkl,
    fasta,
    out_folder,
    min_peptide_length,
    max_peptide_length,
    load_model_hla,
    load_model_pept,
    device,
):
    # check input peptide source
    try:
        with open(peptide_pkl, "rb") as f:
            data_dict = pickle.load(f)
            peptide_list = data_dict["peptide_list"]
            pept_embeds = data_dict["pept_embeds"]
    except Exception as e:
        raise RuntimeError(f"Failed to load Peptide embeddings: {e}") from e

    # check input MHC protein source
    try:
        with open(protein_pkl, "rb") as f:
            data_dict = pickle.load(f)
            protein_df = data_dict["protein_df"]
            hla_embeds = data_dict["embeds"]
    except Exception as e:
        raise RuntimeError(f"Failed to load MHC protein embeddings: {e}") from e

    if device == "cuda" and not torch.cuda.is_available():
        click.echo("CUDA not available. Change to use CPU")
        device = "cpu"

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    pept_encoder = ModelSeqEncoder().to(device)
    hla_encoder = ModelHlaEncoder().to(device)

    try:
        hla_encoder.load_state_dict(
            torch.load(load_model_hla, weights_only=True, map_location=device)
        )
        pept_encoder.load_state_dict(
            torch.load(load_model_pept, weights_only=True, map_location=device)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    retriever = MHCBindingRetriever(
        hla_encoder,
        pept_encoder,
        protein_df,
        hla_embeds,
        fasta,
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
    allele_df.sort_values("best_peptide_dist", ascending=True, inplace=True)
    output_dir = Path(out_folder)
    output_file_path = output_dir.joinpath("allele_df_for_epitope.tsv")
    allele_df.to_csv(output_file_path, sep="\t", index=False)


if __name__ == "__main__":
    run()
