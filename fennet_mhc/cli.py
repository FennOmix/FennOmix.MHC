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
    "embed-proteins", help="Embed MHC class I proteins using Fennet-MHC HLA encoder"
)
@click.option(
    "--fasta",
    type=click.Path(exists=True),
    required=True,
    help="Path to fasta file containing MHC class I protein sequences. "
    "    Format: >HLA-A*01:01\nSEQUENCE",
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
    import fennet_mhc.pipeline_api as pipeline_api

    pipeline_api.embed_proteins(fasta, save_pkl_path, hla_model_path, device)


@run.command(
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
    import fennet_mhc.pipeline_api as pipeline_api

    pipeline_api.embed_peptides_fasta(
        fasta,
        save_pkl_path,
        min_peptide_length,
        max_peptide_length,
        peptide_model_path,
        device,
    )


@run.command(
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
    import fennet_mhc.pipeline_api as pipeline_api

    pipeline_api.embed_peptides_tsv(
        tsv,
        save_pkl_path,
        min_peptide_length,
        max_peptide_length,
        peptide_model_path,
        device,
    )


@run.command(
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
    import fennet_mhc.pipeline_api as pipeline_api

    pipeline_api.predict_binding_for_MHC(
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
    )


@run.command(
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
    import fennet_mhc.pipeline_api as pipeline_api

    pipeline_api.predict_binding_for_epitope(
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
    )


@run.command(
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
    import fennet_mhc.pipeline_api as pipeline_api

    pipeline_api.deconvolute_peptides(
        peptide_pkl_path, n_centroids, out_folder, peptide_model_path, device
    )


if __name__ == "__main__":
    run()
