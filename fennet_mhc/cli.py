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
    "--save-pkl-path",
    type=click.Path(),
    required=True,
    help="Path to .pkl Binary file for saving MHC protein embeddings.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cuda",
    show_default=True,
    help="Device to use. Options: 'cpu', 'cuda' (for NVIDIA GPUs), or 'mps' (for Apple Silicon GPUs).",
)
def embed_proteins(fasta, save_pkl_path, device):
    import fennet_mhc.pipeline_api as pipeline_api

    pipeline_api.embed_proteins(fasta, save_pkl_path, device)


@run.command(
    "embed-peptides",
    help="Embed peptides that non-specifically digested from fasta/tsv using Fennet-MHC peptide encoder",
)
@click.option(
    "--peptide-file-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to fasta/tsv file containing peptides.",
)
@click.option(
    "--save-pkl-path",
    type=click.Path(),
    required=True,
    help="Path to .pkl file for saving peptide embeddings.",
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
def embed_peptides_from_file(
    peptide_file_path,
    save_pkl_path,
    min_peptide_length,
    max_peptide_length,
    device,
):
    import fennet_mhc.pipeline_api as pipeline_api

    pipeline_api.embed_peptides_from_file(
        peptide_file_path,
        save_pkl_path,
        min_peptide_length,
        max_peptide_length,
        device,
    )


@run.command(
    "predict-peptide-binders-for-MHC",
    help="Predict peptide binders to MHC class I molecules",
)
@click.option(
    "--peptide-file-path",
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
    "--distance-threshold",
    type=float,
    default=2,
    show_default=True,
    help="Filter peptide by best allele binding distance.",
)
@click.option(
    "--hla-file-path",
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
def predict_peptide_binders_for_MHC(
    peptide_file_path,
    alleles,
    out_folder,
    min_peptide_length,
    max_peptide_length,
    distance_threshold,
    hla_file_path,
    device,
):
    import fennet_mhc.pipeline_api as pipeline_api

    pipeline_api.predict_peptide_binders_for_MHC(
        peptide_file_path,
        alleles,
        out_folder,
        min_peptide_length,
        max_peptide_length,
        distance_threshold,
        hla_file_path,
        device,
    )


@run.command(
    "predict-hla-binders-for-epitopes",
    help="Predict binding MHC class I molecules to the given epitopes",
)
@click.option(
    "--peptide-file-path",
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
    default=12,
    show_default=True,
    help="Maximum peptide length.",
)
@click.option(
    "--distance-threshold",
    type=float,
    default=2,
    show_default=True,
    help="Filter by binding distance.",
)
@click.option(
    "--hla-file-path",
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
def predict_binders_for_epitopes(
    peptide_file_path,
    out_folder,
    min_peptide_length,
    max_peptide_length,
    distance_threshold,
    hla_file_path,
    device,
):
    import fennet_mhc.pipeline_api as pipeline_api

    pipeline_api.predict_binders_for_epitopes(
        peptide_file_path,
        out_folder,
        min_peptide_length,
        max_peptide_length,
        distance_threshold,
        hla_file_path,
        device,
    )


@run.command(
    "deconvolute-peptides",
    help="De-convolute peptides into clusters.",
)
@click.option(
    "--peptide-file-path",
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
    "--hla-file-path",
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
    peptide_file_path,
    n_centroids,
    out_folder,
    min_peptide_length,
    max_peptide_length,
    hla_file_path,
    device,
):
    import fennet_mhc.pipeline_api as pipeline_api

    pipeline_api.deconvolute_peptides(
        peptide_file_path,
        n_centroids,
        out_folder,
        min_peptide_length,
        max_peptide_length,
        hla_file_path,
        device,
    )


if __name__ == "__main__":
    run()
