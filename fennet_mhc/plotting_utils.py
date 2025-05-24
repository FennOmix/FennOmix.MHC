from collections.abc import Sequence

import logomaker as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from matplotlib.axes import Axes
from sklearn.manifold import MDS, TSNE

Turbo256 = (
    "#30123b",
    "#311542",
    "#32184a",
    "#341b51",
    "#351e58",
    "#36215f",
    "#372365",
    "#38266c",
    "#392972",
    "#3a2c79",
    "#3b2f7f",
    "#3c3285",
    "#3c358b",
    "#3d3791",
    "#3e3a96",
    "#3f3d9c",
    "#4040a1",
    "#4043a6",
    "#4145ab",
    "#4148b0",
    "#424bb5",
    "#434eba",
    "#4350be",
    "#4353c2",
    "#4456c7",
    "#4458cb",
    "#455bce",
    "#455ed2",
    "#4560d6",
    "#4563d9",
    "#4666dd",
    "#4668e0",
    "#466be3",
    "#466de6",
    "#4670e8",
    "#4673eb",
    "#4675ed",
    "#4678f0",
    "#467af2",
    "#467df4",
    "#467ff6",
    "#4682f8",
    "#4584f9",
    "#4587fb",
    "#4589fc",
    "#448cfd",
    "#438efd",
    "#4291fe",
    "#4193fe",
    "#4096fe",
    "#3f98fe",
    "#3e9bfe",
    "#3c9dfd",
    "#3ba0fc",
    "#39a2fc",
    "#38a5fb",
    "#36a8f9",
    "#34aaf8",
    "#33acf6",
    "#31aff5",
    "#2fb1f3",
    "#2db4f1",
    "#2bb6ef",
    "#2ab9ed",
    "#28bbeb",
    "#26bde9",
    "#25c0e6",
    "#23c2e4",
    "#21c4e1",
    "#20c6df",
    "#1ec9dc",
    "#1dcbda",
    "#1ccdd7",
    "#1bcfd4",
    "#1ad1d2",
    "#19d3cf",
    "#18d5cc",
    "#18d7ca",
    "#17d9c7",
    "#17dac4",
    "#17dcc2",
    "#17debf",
    "#18e0bd",
    "#18e1ba",
    "#19e3b8",
    "#1ae4b6",
    "#1be5b4",
    "#1de7b1",
    "#1ee8af",
    "#20e9ac",
    "#22eba9",
    "#24eca6",
    "#27eda3",
    "#29eea0",
    "#2cef9d",
    "#2ff09a",
    "#32f197",
    "#35f394",
    "#38f491",
    "#3bf48d",
    "#3ff58a",
    "#42f687",
    "#46f783",
    "#4af880",
    "#4df97c",
    "#51f979",
    "#55fa76",
    "#59fb72",
    "#5dfb6f",
    "#61fc6c",
    "#65fc68",
    "#69fd65",
    "#6dfd62",
    "#71fd5f",
    "#74fe5c",
    "#78fe59",
    "#7cfe56",
    "#80fe53",
    "#84fe50",
    "#87fe4d",
    "#8bfe4b",
    "#8efe48",
    "#92fe46",
    "#95fe44",
    "#98fe42",
    "#9bfd40",
    "#9efd3e",
    "#a1fc3d",
    "#a4fc3b",
    "#a6fb3a",
    "#a9fb39",
    "#acfa37",
    "#aef937",
    "#b1f836",
    "#b3f835",
    "#b6f735",
    "#b9f534",
    "#bbf434",
    "#bef334",
    "#c0f233",
    "#c3f133",
    "#c5ef33",
    "#c8ee33",
    "#caed33",
    "#cdeb34",
    "#cfea34",
    "#d1e834",
    "#d4e735",
    "#d6e535",
    "#d8e335",
    "#dae236",
    "#dde036",
    "#dfde36",
    "#e1dc37",
    "#e3da37",
    "#e5d838",
    "#e7d738",
    "#e8d538",
    "#ead339",
    "#ecd139",
    "#edcf39",
    "#efcd39",
    "#f0cb3a",
    "#f2c83a",
    "#f3c63a",
    "#f4c43a",
    "#f6c23a",
    "#f7c039",
    "#f8be39",
    "#f9bc39",
    "#f9ba38",
    "#fab737",
    "#fbb537",
    "#fbb336",
    "#fcb035",
    "#fcae34",
    "#fdab33",
    "#fda932",
    "#fda631",
    "#fda330",
    "#fea12f",
    "#fe9e2e",
    "#fe9b2d",
    "#fe982c",
    "#fd952b",
    "#fd9229",
    "#fd8f28",
    "#fd8c27",
    "#fc8926",
    "#fc8624",
    "#fb8323",
    "#fb8022",
    "#fa7d20",
    "#fa7a1f",
    "#f9771e",
    "#f8741c",
    "#f7711b",
    "#f76e1a",
    "#f66b18",
    "#f56817",
    "#f46516",
    "#f36315",
    "#f26014",
    "#f15d13",
    "#ef5a11",
    "#ee5810",
    "#ed550f",
    "#ec520e",
    "#ea500d",
    "#e94d0d",
    "#e84b0c",
    "#e6490b",
    "#e5460a",
    "#e3440a",
    "#e24209",
    "#e04008",
    "#de3e08",
    "#dd3c07",
    "#db3a07",
    "#d93806",
    "#d73606",
    "#d63405",
    "#d43205",
    "#d23005",
    "#d02f04",
    "#ce2d04",
    "#cb2b03",
    "#c92903",
    "#c72803",
    "#c52602",
    "#c32402",
    "#c02302",
    "#be2102",
    "#bb1f01",
    "#b91e01",
    "#b61c01",
    "#b41b01",
    "#b11901",
    "#ae1801",
    "#ac1601",
    "#a91501",
    "#a61401",
    "#a31201",
    "#a01101",
    "#9d1001",
    "#9a0e01",
    "#970d01",
    "#940c01",
    "#910b01",
    "#8e0a01",
    "#8b0901",
    "#870801",
    "#840701",
    "#810602",
    "#7d0502",
    "#7a0402",
)


def fit_hla_umap_reducer(
    hla_embeds: np.ndarray, random_state: int = 1337
) -> umap.UMAP:
    """Fit a UMAP reducer on HLA embeddings.

    Args:
        hla_embeds (np.ndarray): Array of HLA embeddings of shape ``(n, d)``.
        random_state (int, optional): Random seed for the UMAP algorithm.
            Defaults to ``1337``.

    Returns:
        umap.UMAP: The fitted reducer.
    """

    hla_reducer = umap.UMAP(random_state=random_state)
    hla_reducer.fit(hla_embeds)
    return hla_reducer


def transform_embeds_to_umap_df(
    hla_reducer: umap.UMAP,
    embeds: np.ndarray,
    alleles: Sequence[str],
) -> pd.DataFrame:
    """Transform embeddings using a fitted UMAP reducer.

    Args:
        hla_reducer (umap.UMAP): Reducer fitted with :func:`fit_hla_umap_reducer`.
        embeds (np.ndarray): Embeddings to project with shape ``(n, d)``.
        alleles (Sequence[str]): Allele label for each embedding.

    Returns:
        pandas.DataFrame: UMAP coordinates with an ``allele`` column.
    """

    embedding = hla_reducer.transform(embeds)
    df = pd.DataFrame(embedding, columns=("UMAP1", "UMAP2"))
    df["allele"] = alleles
    return df


def transform_matrix_to_mds_df(
    matrix: np.ndarray, labels: Sequence[str], seed: int
) -> pd.DataFrame:
    """Convert a distance matrix to two dimensions using MDS.

    Args:
        matrix (np.ndarray): Pairwise distance matrix.
        labels (Sequence[str]): Labels for each sample in ``matrix``.
        seed (int): Random seed for MDS initialization.

    Returns:
        pandas.DataFrame: DataFrame with ``MDS1`` and ``MDS2`` columns.
    """

    mds = MDS(
        n_components=2,
        random_state=seed,
        dissimilarity="precomputed",
        normalized_stress="auto",
    )
    embedding = mds.fit_transform(matrix)
    df = pd.DataFrame(embedding, columns=("MDS1", "MDS2"))
    df["label"] = labels
    return df


def transform_embeds_to_tSNE_df(
    embeds: np.ndarray, labels: Sequence[str], seed: int
) -> pd.DataFrame:
    """Project embeddings to two dimensions using t-SNE.

    Args:
        embeds (np.ndarray): Array of embeddings with shape ``(n, d)``.
        labels (Sequence[str]): Labels for each embedding.
        seed (int): Random seed for t-SNE.

    Returns:
        pandas.DataFrame: DataFrame with ``t-SNE 1`` and ``t-SNE 2`` columns.
    """

    tsne = TSNE(
        n_components=2, random_state=seed, perplexity=30, learning_rate=200, n_iter=1000
    )
    embedding = tsne.fit_transform(embeds)
    df = pd.DataFrame(embedding, columns=("t-SNE 1", "t-SNE 2"))
    df["label"] = labels
    return df


def plot_umap_df(
    df: pd.DataFrame,
    color_col: str,
    hover_col: str,
    size: int = 1,
    jump_color: bool = True,
    image_width: int = 700,
    image_height: int = 600,
    save_as: str = "",
) -> go.Figure:
    """Plot UMAP embeddings using Plotly.

    Args:
        df (pandas.DataFrame): DataFrame produced by
            :func:`transform_embeds_to_umap_df`.
        color_col (str): Column used to color the scatter points.
        hover_col (str): Column shown when hovering over a point.
        size (int, optional): Marker size. Defaults to ``1``.
        jump_color (bool, optional): Use a spaced colour palette to
            differentiate categories. Defaults to ``True``.
        image_width (int, optional): Width of the image in pixels.
            Defaults to ``700``.
        image_height (int, optional): Height of the image in pixels.
            Defaults to ``600``.
        save_as (str, optional): File path to save the figure as a static
            image. If empty the figure is not saved.

    Returns:
        plotly.graph_objs.Figure: The generated scatter plot.
    """
    factors = df[color_col].drop_duplicates().to_list()
    color_mapping = {}
    for i, factor in enumerate(factors):
        if jump_color:
            color_mapping[factor] = Turbo256[i * 23 % len(Turbo256)]
        else:
            color_mapping[factor] = Turbo256[
                int(i * 1.0 * len(Turbo256) / len(factors))
            ]
    fig = px.scatter(
        df,
        x="UMAP1",
        y="UMAP2",
        color=color_col,
        hover_data=[hover_col],
        color_discrete_map=color_mapping,
        width=image_width,
        height=image_height,
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=size))
    fig.update_layout(
        legend=dict(
            itemsizing="constant",
            orientation="h",
            x=1,
            xanchor="right",
            yanchor="bottom",
            y=-1.0 / 11 * (len(factors) // 7 + 1),
        ),
        xaxis=dict(
            showgrid=False,
            visible=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False,
            visible=False,
            showticklabels=False,
        ),
    )
    if save_as:
        fig.write_image(save_as)
    # fig.show()
    return fig


def plot_motif_multi_mer(
    df: pd.DataFrame,
    allele_col: str,
    allele: str,
    kmers: Sequence[int],
    axes: Axes | Sequence[Axes] | None = None,
    logo_scale: int = 20,
    fig_width_per_kmer: int = 4,
    fig_height: int = 3,
) -> list[lm.Logo]:
    """Plot sequence logos for multiple peptide lengths.

    Args:
        df (pandas.DataFrame): DataFrame containing a ``sequence`` column.
        allele_col (str): Name of the column with allele identifiers.
        allele (str): Allele value to filter ``df``.
        kmers (Sequence[int]): Peptide lengths to plot.
        axes (matplotlib.axes.Axes or Sequence[Axes], optional): Axes to draw
            the logos on. When ``None`` a new figure is created.
        logo_scale (int, optional): Scaling factor for the information
            content. Defaults to ``20``.
        fig_width_per_kmer (int, optional): Width of each subplot.
            Defaults to ``4``.
        fig_height (int, optional): Height of the figure. Defaults to ``3``.

    Returns:
        list[logomaker.Logo]: List of logo objects for the plotted motifs.
    """
    df["nAA"] = df.sequence.str.len()
    df = df.drop_duplicates(
        [allele_col, "sequence"]
        + (["mods", "mod_sites"] if "mods" in df.columns else [])
    )
    if axes is None:
        fig, axes = plt.subplots(
            1,
            len(kmers),
            figsize=(len(kmers) * fig_width_per_kmer, fig_height),
            sharey="row",
        )
    logo_plots = []
    max_y_vals = []

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for i, ax in enumerate([axes] if len(kmers) == 1 else axes):
        logo_plots.append(
            plot_motif(df, allele_col, allele, kmers[i], ax=ax, logo_scale=logo_scale)
        )
        max_y_vals.append(np.ceil(ax.get_ylim()[1] * 5 + 1) * 0.2)

    max_y = np.max(max_y_vals)
    for i, logos in enumerate(logo_plots):
        logos.ax.text(
            3,
            max_y + 0.1,
            f"n={len(df[(df[allele_col] == allele) & (df.nAA == kmers[i])])}",
        )
    adjust_axes(logo_plots, max_y)
    return logo_plots


def plot_motif(
    df: pd.DataFrame,
    allele_col: str,
    allele: str,
    kmer: int,
    ax: Axes | None = None,
    logo_scale: int = 20,
) -> lm.Logo:
    """Plot a single sequence logo for peptides of a given length.

    Args:
        df (pandas.DataFrame): DataFrame of peptides.
        allele_col (str): Column containing allele identifiers.
        allele (str): Allele to plot.
        kmer (int): Peptide length to include in the motif.
        ax (matplotlib.axes.Axes, optional): Axis to draw on. When ``None`` a
            new axis is created.
        logo_scale (int, optional): Scaling factor for information content.
            Defaults to ``20``.

    Returns:
        logomaker.Logo: The created Logo object.
    """

    motif_df = count_motif_bits(df, allele_col, allele, kmer, logo_scale=logo_scale)
    logo_plot = lm.Logo(
        motif_df,
        font_name="DejaVu Sans",
        color_scheme="chemistry",
        ax=ax,
    )
    logo_plot.style_xticks(anchor=0, spacing=1, rotation=0)
    return logo_plot


def count_motif_bits(
    df: pd.DataFrame,
    allele_col: str,
    allele: str,
    kmer: int,
    logo_scale: int = 20,
) -> pd.DataFrame:
    """Calculate the information content matrix for a motif.

    Args:
        df (pandas.DataFrame): DataFrame of peptides.
        allele_col (str): Column with allele identifiers.
        allele (str): Allele value to select.
        kmer (int): Peptide length to include.
        logo_scale (int, optional): Scaling factor used when converting
            frequencies to bits. Defaults to ``20``.

    Returns:
        pandas.DataFrame: Matrix suitable for :class:`logomaker.Logo`.
    """

    df = df[(df[allele_col] == allele) & (df.nAA == kmer)]
    data = np.zeros((kmer, 26), dtype=float)
    for seq in df.sequence.values:
        data[(np.arange(kmer), np.array(seq, "c").view(np.int8) - ord("A"))] += 1
    data /= len(df)
    df = pd.DataFrame(data=data, columns=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

    df = df[list("ACDEFGHIKLMNPQRSTVWY")]
    df = df.apply(lambda p: p * np.log2(p * logo_scale + 1e-100))
    df.values[df.isin([np.nan, np.inf, -np.inf])] = 0
    return df


def adjust_axes(
    logo_plots: Sequence[lm.Logo] | Sequence[list[lm.Logo]],
    max_y: float,
) -> None:
    """Synchronise the y-axis limits across logo plots.

    Args:
        logo_plots (Sequence[logomaker.Logo] | Sequence[list[logomaker.Logo]]):
            Logos returned by :func:`plot_motif` or :func:`plot_motif_multi_mer`.
        max_y (float): Maximum y value to set for all axes.

    Returns:
        None
    """

    for logos in logo_plots:
        if isinstance(logos, list):
            logos[0].ax.set_ylim(top=max_y)
        else:
            logos.ax.set_ylim(top=max_y)
            break


def select_optimal_a_cover(
    a_to_b_map: dict[str, set[str]],
    uncovered_b: set[str],
    coverage_threshold: float,
    max_a_elements: int,
) -> set[str]:
    """Greedily select elements of ``a`` to cover ``b`` as well as possible.

    Args:
        a_to_b_map (dict[str, set[str]]): Mapping from an ``a`` element to the
            ``b`` elements it covers.
        uncovered_b (set[str]): Set of ``b`` elements that need to be covered.
        coverage_threshold (float): Fraction of ``b`` that should be covered
            before stopping.
        max_a_elements (int): Maximum number of ``a`` elements to select.

    Returns:
        set[str]: The selected ``a`` elements.
    """

    total_b = len(uncovered_b)
    selected_a: set[str] = set()
    current_coverage = 0.0
    selected_count = 0

    while current_coverage < coverage_threshold and selected_count < max_a_elements:
        best_a = None
        best_coverage_count = 0

        for a, b_set in a_to_b_map.items():
            if a in selected_a:
                continue

            cover_count = sum(1 for b in b_set if b in uncovered_b)

            if cover_count > best_coverage_count:
                best_a = a
                best_coverage_count = cover_count

        if best_a is not None:
            selected_a.add(best_a)
            selected_count += 1

            for b in a_to_b_map[best_a]:
                if b in uncovered_b:
                    uncovered_b.remove(b)

            current_coverage = 1.0 - len(uncovered_b) / total_b
        else:
            break

    return selected_a
