"""Command-line interface for Fennet-MHC.

This module exposes subcommands to check the installation, embed proteins
and peptides, and predict peptide binders for specified MHC class I alleles.
"""

import click

import fennet_mhc


@click.group(
    context_settings=dict(help_option_names=["-h", "--help"]),
    invoke_without_command=True,
    help="Foundation model to embed molecules and peptides for MHC class I binding prediction",
)
@click.pass_context
@click.version_option(fennet_mhc.__version__, "-v", "--version")
def run(ctx, **kwargs):
    """Show banner and delegate to subcommands.

    If called without a subcommand, prints the help text.
    """
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


@run.command(
    "check",
    help="Check if this package works, and download the model files if missing.",
)
def check():
    """Verify package setup and download models if missing."""
    import fennet_mhc.pipeline_api as pipeline_api

    pipeline_api.PretrainedModels(device="cpu")


@run.command(
    "embed-proteins", help="Embed MHC class I proteins using Fennet-MHC MHC encoder"
)
@click.option(
    "--fasta",
    type=click.Path(exists=True),
    required=True,
    help="Path to fasta file containing MHC class I protein sequences. "
    "    Format: >A01_01\nSEQUENCE",
)
@click.option(
    "--out-folder",
    type=click.Path(),
    required=True,
    help="Folder path to save hla_embeddings.pkl.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cuda",
    show_default=True,
    help="Device to use. Options: 'cpu', 'cuda' (for NVIDIA GPUs), or 'mps' (for Apple Silicon GPUs).",
)
def embed_proteins(fasta, out_folder, device):
    """Embed MHC class I protein sequences from a FASTA file.

    Parameters
    ----------
    fasta : str
        Path to FASTA file containing allele sequences (header as allele id).
    out_folder : str
        Output folder to write the serialized embeddings.
    device : {"cpu","cuda","mps"}
        Device used for inference.
    """
    import fennet_mhc.pipeline_api as pipeline_api

    pipeline_api.embed_proteins(fasta, out_folder, device)


@run.command(
    "embed-peptides",
    help="Embed peptides that non-specifically digested from fasta/tsv using Fennet-MHC peptide encoder",
)
@click.option(
    "--peptide-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to fasta/tsv file containing peptides.",
)
@click.option(
    "--out-folder",
    type=click.Path(),
    required=True,
    help="Folder path to save peptide_embeddings.pkl.",
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
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cuda",
    show_default=True,
    help="Device to use. Options: 'cpu', 'cuda' (for NVIDIA GPUs), or 'mps' (for Apple Silicon GPUs).",
)
def embed_peptides(
    peptide_file,
    out_folder,
    min_peptide_length,
    max_peptide_length,
    device,
):
    """Embed peptides listed in a FASTA/TSV file.

    Parameters
    ----------
    peptide_file : str
        Input FASTA or delimited file with a column "sequence".
    out_folder : str
        Output folder to write the serialized embeddings.
    min_peptide_length, max_peptide_length : int
        Peptide length range used when digesting FASTA inputs.
    device : {"cpu","cuda","mps"}
        Device used for inference.
    """
    import fennet_mhc.pipeline_api as pipeline_api

    pipeline_api.embed_peptides_from_file(
        peptide_file,
        out_folder,
        min_peptide_length,
        max_peptide_length,
        device,
    )


@run.command(
    "predict-epitopes-for-mhc",
    help="Predict peptide binders to MHC class I molecules",
)
@click.option(
    "--peptide-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to tsv file containing peptides or fasta file for non-specific digestion.",
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
    "--out-fasta-format", is_flag=True, help="If output the results in fasta format."
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
    "--outlier-distance",
    type=float,
    default=0.4,
    show_default=True,
    help="Filter peptide by best allele binding distance.",
)
@click.option(
    "--hla-file",
    default=None,
    required=False,
    help="Path to the fasta file or pre-computed MHC protein embeddings file (.pkl) or fasta file. "
    "If None, a default embeddings file cotaining 15672 alleles is provided. "
    "If your desired alleles are not included in the default file, ",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cuda",
    show_default=True,
    help="Device to use. Options: 'cpu', 'cuda' (for NVIDIA GPUs), or 'mps' (for Apple Silicon GPUs).",
)
def predict_epitopes_for_mhc(
    peptide_file,
    alleles,
    out_folder,
    out_fasta_format,
    min_peptide_length,
    max_peptide_length,
    outlier_distance,
    hla_file,
    device,
):
    """Predict peptide binders for the given MHC class I alleles.

    Parameters
    ----------
    peptide_file : str
        Input peptides (TSV/CSV with a "sequence" column or FASTA for digestion).
    alleles : str
        Comma-separated allele names, e.g. "A01_01,B07_02".
    out_folder : str
        Output folder for result files.
    out_fasta_format : bool
        If True, also write results in FASTA format.
    min_peptide_length, max_peptide_length : int
        Peptide length range for FASTA digestion.
    outlier_distance : float
        Threshold on best-allele distance used to filter peptides.
    hla_file : str | None
        Optional path to precomputed HLA embeddings (.pkl) or FASTA.
    device : {"cpu","cuda","mps"}
        Device used for inference.
    """
    import fennet_mhc.pipeline_api as pipeline_api

    alleles = [x.strip() for x in alleles.split(",")]
    pipeline_api.predict_epitopes_for_mhc(
        peptide_file,
        alleles,
        out_folder,
        out_fasta_format,
        min_peptide_length,
        max_peptide_length,
        outlier_distance,
        hla_file,
        device,
    )


@run.command(
    "predict-mhc-binders-for-epitopes",
    help="Predict binding MHC class I molecules to the given epitopes",
)
@click.option(
    "--peptide-file",
    type=click.Path(exists=True),
    help="Path to tsv file containing peptides or fasta file for non-specific digestion.",
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
    "--outlier-distance",
    type=float,
    default=0.4,
    show_default=True,
    help="Filter outliers by binding distance.",
)
@click.option(
    "--hla-file",
    default=None,
    help="Path to the pre-computed MHC protein embeddings file (.pkl). "
    "If None, a default embeddings file will be used. "
    "If your desired alleles are not included in the default file, "
    "you can generate a custom embeddings file using the *embed_proteins* command.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cuda",
    show_default=True,
    help="Device to use. Options: 'cpu', 'cuda' (for NVIDIA GPUs), or 'mps' (for Apple Silicon GPUs).",
)
def predict_mhc_binders_for_epitopes(
    peptide_file,
    out_folder,
    min_peptide_length,
    max_peptide_length,
    outlier_distance,
    hla_file,
    device,
):
    import fennet_mhc.pipeline_api as pipeline_api

    pipeline_api.predict_mhc_binders_for_epitopes(
        peptide_file,
        out_folder,
        min_peptide_length,
        max_peptide_length,
        outlier_distance,
        hla_file,
        device,
    )


@run.command(
    "deconvolute-peptides",
    help="De-convolute peptides into clusters.",
)
@click.option(
    "--peptide-file",
    type=click.Path(exists=True),
    help="Path to fasta/peptide_tsv or peptide pre-embedding file (.pkl).",
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
    "--min-peptide-length",
    type=int,
    default=8,
    show_default=True,
    help="Minimum peptide length.",
)
@click.option(
    "--max-peptide-length",
    type=int,
    default=12,
    show_default=True,
    help="Maximum peptide length.",
)
@click.option(
    "--hla-file",
    default=None,
    required=False,
    help="Path to the fasta file or pre-computed MHC protein embeddings file (.pkl) or fasta file. "
    "If None, a default embeddings file cotaining 15672 alleles is provided. "
    "If your desired alleles are not included in the default file, ",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cuda",
    show_default=True,
    help="Device to use. Options: 'cpu', 'cuda' (for NVIDIA GPUs), or 'mps' (for Apple Silicon GPUs).",
)
def deconvolute_peptides(
    peptide_file,
    n_centroids,
    out_folder,
    min_peptide_length,
    max_peptide_length,
    hla_file,
    device,
):
    import fennet_mhc.pipeline_api as pipeline_api

    pipeline_api.deconvolute_peptides(
        peptide_file,
        n_centroids,
        out_folder,
        min_peptide_length=min_peptide_length,
        max_peptide_length=max_peptide_length,
        outlier_distance=100,  # no distance refinement
        hla_file_path=hla_file,
        device=device,
    )


@run.command(
    "deconvolute-and-predict-peptides",
    help="De-convolute peptides into clusters, and then predict a peptide library based on the cluster.",
)
@click.option(
    "--peptide-file-to-deconv",
    type=click.Path(exists=True),
    help="Path to fasta/peptide_tsv or peptide pre-embedding file (.pkl).",
)
@click.option(
    "--peptide-file-to-predict",
    type=click.Path(exists=True),
    help="Path to fasta/peptide_tsv or peptide pre-embedding file (.pkl).",
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
    "--out-fasta-format", is_flag=True, help="If output the results in fasta format."
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
    default=12,
    show_default=True,
    help="Maximum peptide length.",
)
@click.option(
    "--outlier-distance",
    type=float,
    default=0.2,
    show_default=True,
    help="Distance threshold for outliers.",
)
@click.option(
    "--hla-file",
    default=None,
    required=False,
    help="Path to the fasta file or pre-computed MHC protein embeddings file (.pkl) or fasta file. "
    "If None, a default embeddings file cotaining 15672 alleles is provided. "
    "If your desired alleles are not included in the default file, ",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cuda",
    show_default=True,
    help="Device to use. Options: 'cpu', 'cuda' (for NVIDIA GPUs), or 'mps' (for Apple Silicon GPUs).",
)
def deconvolute_and_predict_peptides(
    peptide_file_to_deconv,
    peptide_file_to_predict,
    n_centroids,
    out_folder,
    out_fasta_format,
    min_peptide_length,
    max_peptide_length,
    outlier_distance,
    hla_file,
    device,
):
    import fennet_mhc.pipeline_api as pipeline_api

    pipeline_api.deconvolute_and_predict_peptides(
        peptide_file_to_deconv,
        peptide_file_to_predict,
        n_centroids,
        out_folder,
        out_fasta_format,
        min_peptide_length,
        max_peptide_length,
        outlier_distance,
        hla_file,
        device,
    )


if __name__ == "__main__":
    run()
