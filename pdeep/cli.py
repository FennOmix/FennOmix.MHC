import pickle
from pathlib import Path

import click

import pdeep
from pdeep.mhc.mhc_binding_model import get_model
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
        pDeep
        ...................................................
        .{pdeep.__version__.center(50)}.
        .{pdeep.__github__.center(50)}.
        .{pdeep.__license__.center(50)}.
        ...................................................
        """
    )
    if ctx.invoked_subcommand is None:
        click.echo(run.get_help(ctx))


@run.command("mhc", help="Predict binding of peptides to HLA class Ⅰ proteins")
@click.option(
    "--fasta", type=click.Path(exists=True), required=True, help="Path to Fasta file."
)
@click.option(
    "--alleles",
    type=str,
    default="A03_01,B07_02,C07_02",
    show_default=True,
    help="list of HLA class Ⅰ alleles, sperated by comma.",
)
@click.option(
    "--out-folder",
    type=click.Path(exists=True),
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
    default=1e10,
    show_default=True,
    help="Filter peptide by best allele embedding distance.",
)
@click.option(
    "--filter_fdr",
    type=float,
    default=1e10,
    show_default=True,
    help="Filter peptide by best allele %rank.",
)
def mhc(
    fasta,
    alleles,
    out_folder,
    min_peptide_length,
    max_peptide_length,
    filter_distance,
    filter_fdr,
):
    model_version = "v0613"
    hla_encoder, pept_encoder = get_model(model_version)

    with open(f"embeds/HLA_model_{model_version}.pt.embed", "rb") as f:
        data_dict = pickle.load(f)
        protein_df = data_dict["protein_df"]
        embeds = data_dict["embeds"]

    retriever = MHCBindingRetriever(
        hla_encoder,
        pept_encoder,
        protein_df,
        embeds,
        fasta,
        digested_pept_lens=(min_peptide_length, max_peptide_length),
    )

    input_alleles_list = alleles.split(",")

    peptide_df = retriever.get_binding_metrics_for_self_proteins(
        input_alleles_list,
        dist_threshold=filter_distance,
        fdr=filter_fdr,
        get_sequence=True,
    )
    peptide_df.sort_values(by="best_allele_dist", inplace=True)

    output_dir = Path(out_folder)
    output_file_path = output_dir.joinpath("peptide_df.tsv")
    peptide_df.to_csv(output_file_path, sep="\t", index=False)


if __name__ == "__main__":
    run()
