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
)
from pdeep.mhc.mhc_binding_retriever import MHCBindingRetriever


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
    default="./model/HLA_model_v0613.pt",
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
                protein_name = line.split(">")[1]
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


# @mhc.command("embed_peptides_fasta", help="Generate peptides from given fasta and embed")

# @mhc.command("embed_peptides_tsv", help="Embed peptides from given tsv")


@mhc.command(
    "predict_binding_for_MHC", help="Predict binding of peptides to MHC molecules"
)
@click.option(
    "--fasta",
    type=click.Path(exists=True),
    default="",
    help="Path to Fasta file containing proteins that will be digested to generate peptides.",
)
@click.option(
    "--tsv",
    type=click.Path(exists=True),
    default="",
    help="Path to tsv file containing peptides list. (Mutually exclusive with the previous --fasta option, don't provide fasta file and tsv file at the same time)",
)
@click.option(
    "--protein_pkl",
    type=click.Path(exists=True),
    default="./embeds/HLA_model_v0613.pt.embed",
    help="Path to MHC protein pre-embeddings binary file (.pkl), If the alleles you want do not exist in the original provided list, "
    "you can provide the sequences yourself and use the previous embed_proteins function to generate custom protein pkl file",
)
@click.option(
    "--alleles",
    type=str,
    required=True,
    help="List of HLA class I alleles, separated by commas. Example: A01_01,B07_02,C07_02.",
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
    help="Filter peptide by best allele embedding distance.",
)
@click.option(
    "--filter_fdr",
    type=float,
    default=1,
    help="Filter peptide by best allele fdr. (Mutually exclusive with the previous --filter_distance option, Dont't set two filtering standards at the same time.)",
)
@click.option(
    "--load_model_hla",
    type=click.Path(exists=True),
    default="./model/HLA_model_v0613.pt",
    show_default=True,
    help="Path to HLA model parameter file.",
)
@click.option(
    "--load_model_pept",
    type=click.Path(exists=True),
    default="./model/pept_model_v0613.pt",
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
    fasta,
    tsv,
    protein_pkl,
    alleles,
    out_folder,
    min_peptide_length,
    max_peptide_length,
    filter_distance,
    filter_fdr,
    load_model_hla,
    load_model_pept,
    device,
):
    # check input peptide source
    if fasta and tsv:
        click.echo("Don't provide fasta and tsv files at the same time.")
        sys.exit(1)
    elif not fasta and not tsv:
        click.echo("Please provide at least one fasta or tsv files.")
        sys.exit(1)

    # check input MHC protein source
    try:
        with open(protein_pkl, "rb") as f:
            data_dict = pickle.load(f)
            protein_df = data_dict["protein_df"]
            embeds = data_dict["embeds"]
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

    if filter_distance and filter_fdr:
        click.echo("Don't provide filter_distance and filter_fdr at the same time.")
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
        embeds,
        fasta,
        digested_pept_lens=(min_peptide_length, max_peptide_length),
    )
    if fasta:
        peptide_df = retriever.get_binding_metrics_for_self_proteins(
            selected_alleles_array,
            dist_threshold=filter_distance,
            fdr=filter_fdr,
            get_sequence=True,
        )
        peptide_df.sort_values(by="best_allele_dist", inplace=True)
        # rerank depend on filter standards
    elif tsv:
        input_peptide_df = pd.read_table(tsv, sep="\t", index_col=False)
        input_peptide_list = input_peptide_df.iloc[:, 0].tolist()
        peptide_df = retriever.get_binding_metrics_for_peptides(
            selected_alleles_array, input_peptide_list
        )
        # filter and rerank
    else:
        click.echo("Please provide at least one fasta or tsv files.")
        sys.exit(1)

    output_dir = Path(out_folder)
    output_file_path = output_dir.joinpath("peptide_df.tsv")
    peptide_df.to_csv(output_file_path, sep="\t", index=False)


# @mhc.command("predict_binding_for_epitope", help="Predict binding of MHC molecules to epitope")


if __name__ == "__main__":
    run()
