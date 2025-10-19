import numba
import numpy as np
import pandas as pd
import torch
import tqdm
from peptdeep.utils import get_device

from .mhc_binding_model import (
    HlaDataSet,
    embed_peptides,
)
from .tda_fmm import DecoyModel, select_best_fmm


@numba.njit
def get_fdrs(
    dists: np.ndarray,
    rnd_dists: np.ndarray,
    alpha: float,
    remove_rnd_top_rank: float = 0.01,
) -> np.ndarray:
    """Calculate FDRs using the target-decoy approach.

    This function computes False Discovery Rates (FDRs) for target distances by comparing them
    against decoy distances. It uses a simple counting method based on rank comparison.

    Args:
        dists: 1D array of target distances between peptide and MHC embeddings.
            Shape: (n_targets,)
        rnd_dists: 1D array of decoy distances used for FDR estimation.
            Shape: (n_decoys,)
        alpha: Ratio of number of targets to decoys, i.e., len(dists) / len(rnd_dists).
        remove_rnd_top_rank: Fraction of lowest-ranked decoy values to exclude as "binders"
            when estimating FDR (default: 0.01).

    Returns:
        fdrs: Array of FDR values corresponding to each entry in `dists`, unsorted.
              Shape: (n_targets,)

    Example:
        >>> targets = np.array([0.3, 0.5, 0.7])
        >>> decoys = np.random.normal(1.0, 0.2, size=1000)
        >>> fdrs = get_fdrs(targets, decoys, alpha=1.0)
    """
    sorted_idxes = np.argsort(dists)
    sorted_rnd = np.argsort(rnd_dists)
    still_binder_rnd_idx = int(remove_rnd_top_rank * len(rnd_dists))

    fdrs = np.zeros_like(dists)

    j = 0
    for idx in sorted_idxes:
        while j < len(rnd_dists) and rnd_dists[sorted_rnd[j]] < dists[idx]:
            j += 1
        if j > still_binder_rnd_idx:
            # do we need D+1 correction?
            fdrs[idx] = (alpha * (j - still_binder_rnd_idx)) / (idx + 1)

    return fdrs


@numba.njit
def get_q_values(fdrs: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Convert FDR estimates into q-values via monotonic minimization.

    Q-values are computed by enforcing that they are non-decreasing with increasing distance,
    ensuring proper multiple testing correction.

    Args:
        fdrs: Input array of FDR values (not necessarily monotonic).
            Shape: (n_samples,)
        distances: Distance values used to sort peptides; larger distances mean weaker binding.
            Used to reverse-sort for q-value computation.
            Shape: (n_samples,)

    Returns:
        qvals: Monotonic q-values, same shape as input.
               Values are adjusted so that q[i] <= q[j] for all j < i in ranking order.
               Shape: (n_samples,)

    Note:
        The algorithm traverses from highest to lowest distance, maintaining minimum seen FDR.
    """
    sorted_idxes = np.argsort(distances)[::-1]
    min_pep = 100000.0
    for idx in sorted_idxes:
        if fdrs[idx] > min_pep:
            fdrs[idx] = min_pep
        else:
            min_pep = fdrs[idx]
    return fdrs


def get_binding_fdrs(
    distances_1D: np.ndarray,
    decoys_1D: np.ndarray,
    max_fitting_samples: int = 200000,
    random_state: int = 1337,
    outlier_threshold: float = 0.01,
    fmm_fdr: bool = False,
) -> np.ndarray:
    """Estimate FDRs for peptide-MHC binding distances using either TDA or FMM.

    Supports two modes:
      - Target-Decoy Analysis (TDA): Simple empirical FDR.
      - Finite Mixture Model (FMM): Probabilistic modeling of binders vs. non-binders.

    Args:
        distances_1D: Observed distances for real peptides.
            Shape: (n_peptides,)
        decoys_1D: Distances for randomly generated decoy peptides.
            Shape: (n_decoys,)
        max_fitting_samples: Maximum number of samples to use in FMM fitting if dataset is large.
        random_state: Random seed for reproducibility during subsampling.
        outlier_threshold: Fraction of smallest decoy distances to ignore as strong binders.
        fmm_fdr: If True, use FMM-based FDR estimation; otherwise use standard TDA.

    Returns:
        fdrs: Estimated FDR for each peptide in `distances_1D`.
              Shape: (n_peptides,)

    Raises:
        ValueError: If `decoys_1D` is empty or invalid.
    """
    if fmm_fdr:
        decoy_fmm = DecoyModel(gaussian_outlier_sigma=outlier_threshold)
        decoy_fmm.fit(decoys_1D)
        if len(distances_1D) >= max_fitting_samples:
            np.random.seed(random_state)
            target_fmm = select_best_fmm(
                np.random.choice(distances_1D, max_fitting_samples, replace=False),
                decoy_fmm,
                verbose=True,
            )
        else:
            target_fmm = select_best_fmm(distances_1D, decoy_fmm, verbose=True)

        # target_fmm.plot("test", distances, decoy_fmm.data)
        print(f"Estimated pi0 = {target_fmm.get_pi0()}")
        peps = target_fmm.pep(distances_1D)
        peps = get_q_values(peps, distances_1D)
        sorted_idxes = np.argsort(distances_1D)
        sorted_fdrs = np.cumsum(peps[sorted_idxes]) / np.arange(1, len(peps) + 1)
        fdrs = np.zeros_like(peps)
        fdrs[sorted_idxes] = sorted_fdrs
    else:
        alpha = 1.0 * len(distances_1D) / len(decoys_1D)
        fdrs = get_fdrs(
            distances_1D, decoys_1D, alpha, remove_rnd_top_rank=outlier_threshold
        )

    fdrs = get_q_values(fdrs, distances_1D)
    return fdrs


def get_binding_fdr_for_best_allele(
    distances: np.ndarray,
    rnd_dist: np.ndarray,
    outlier_threshold: float = 0.01,
    fmm_fdr: bool = False,
) -> np.ndarray:
    """Compute FDRs for the best-binding allele per peptide.

    For each peptide, finds the allele with minimum distance and calculates its FDR
    independently across alleles using decoy distributions.

    Args:
        distances: Distance matrix between peptides and alleles.
            Shape: (n_peptides, n_alleles)
        rnd_dist: Sorted decoy distance matrix (precomputed), one column per allele.
            Should be pre-sorted along axis 0.
            Shape: (n_decoys, n_alleles)
        outlier_threshold: Fraction of top decoys to treat as true binders (ignored in FDR).
        fmm_fdr: Whether to use FMM-based FDR instead of basic TDA.

    Returns:
        best_allele_fdrs: FDR value for the best allele of each peptide.
                          Shape: (n_peptides,)

    Example:
        >>> dists = np.random.rand(100, 6).astype(np.float32)
        >>> decoy_dists = np.sort(np.random.rand(1000, 6), axis=0)
        >>> fdrs = get_binding_fdr_for_best_allele(dists, decoy_dists)
    """
    best_allele_idxes = np.argmin(distances, axis=1)
    min_allele_distances = distances[
        np.arange(len(best_allele_idxes)), best_allele_idxes
    ]
    best_allele_fdrs = np.zeros(len(best_allele_idxes))
    # best_allele_peps = np.zeros_like(best_allele_fdrs)

    for i in range(distances.shape[-1]):
        selected_alleles = best_allele_idxes == i
        fdrs = get_binding_fdrs(
            min_allele_distances[selected_alleles],
            rnd_dist[:, i],
            fmm_fdr=fmm_fdr,
            outlier_threshold=outlier_threshold,
        )
        # best_allele_peps[selected_alleles] = peps
        best_allele_fdrs[selected_alleles] = fdrs
    return best_allele_fdrs


@numba.njit
def get_binding_ranks(distances: np.ndarray, sorted_rnd_dist: np.ndarray) -> np.ndarray:
    """Compute FDRs for the best-binding allele per peptide.

    For each peptide, finds the allele with minimum distance and calculates its FDR
    independently across alleles using decoy distributions.

    Args:
        distances: Distance matrix between peptides and alleles.
            Shape: (n_peptides, n_alleles)
        rnd_dist: Sorted decoy distance matrix (precomputed), one column per allele.
            Should be pre-sorted along axis 0.
            Shape: (n_decoys, n_alleles)
        outlier_threshold: Fraction of top decoys to treat as true binders (ignored in FDR).
        fmm_fdr: Whether to use FMM-based FDR instead of basic TDA.

    Returns:
        best_allele_fdrs: FDR value for the best allele of each peptide.
                          Shape: (n_peptides,)

    Example:
        >>> dists = np.random.rand(100, 6).astype(np.float32)
        >>> decoy_dists = np.sort(np.random.rand(1000, 6), axis=0)
        >>> fdrs = get_binding_fdr_for_best_allele(dists, decoy_dists)
    """
    best_allele_idxes = np.argmin(distances, axis=1)
    best_allele_ranks = np.zeros(len(best_allele_idxes))
    len_rnd = float(sorted_rnd_dist.shape[0])
    for i, allele_idx in enumerate(best_allele_idxes):
        rank = np.searchsorted(sorted_rnd_dist[:, allele_idx], distances[i, allele_idx])
        best_allele_ranks[i] = rank / len_rnd * 100
    return best_allele_ranks


class MHCBindingRetriever:
    """A retriever class to compute peptide-MHC binding metrics including distance, rank, and FDR.

    This class wraps trained encoders for peptides and HLAs, enabling fast retrieval of binding
    predictions through embedding space distance. It supports both single-peptide queries and
    genome-wide screening against self-proteins.

    Attributes:
        hla_encoder: Trained neural network model for encoding HLA sequences.
        pept_encoder: Trained neural network model for encoding peptide sequences.
        device (torch.device): Computation device (e.g., 'cuda' or 'cpu').
        dataset (HlaDataSet): Dataset handler containing protein digestion and HLA info.
        hla_embeds (np.ndarray): Precomputed HLA embeddings. Shape: (n_alleles, d_model)
        n_decoy_samples (int): Number of random decoy peptides to generate for FDR estimation.
        outlier_threshold (float): Fraction of strongest decoy binders to exclude.
        use_fmm_fdr (bool): Whether to use finite mixture model for FDR calculation.
        decoy_rnd_seed (int): Seed for reproducible decoy generation.
        d_model (int): Embedding dimension size.
        verbose (bool): Enable progress bars and logging.
    """
    def __init__(
        self,
        hla_encoder,
        pept_encoder,
        hla_df: pd.DataFrame,
        hla_embeds: np.ndarray,
        protein_data,
        min_peptide_len: int = 8,
        max_peptide_len: int = 14,
        device: str = "cuda",
    ) -> None:
        """Initialize the MHCBindingRetriever.

        Args:
            hla_encoder: Model to encode HLA alleles into fixed-length vectors.
            pept_encoder: Model to encode peptides into fixed-length vectors.
            hla_df: DataFrame containing HLA allele metadata (e.g., names, sequences).
            hla_embeds: Precomputed embeddings for all HLA alleles. Shape: (n_alleles, d_model)
            protein_data: Protein sequences used for generating decoy/non-self peptides.
            min_peptide_len: Minimum length for digested peptides (default: 8).
            max_peptide_len: Maximum length for digested peptides (default: 14).
            device: Torch device identifier ('cuda', 'cpu', etc.). Auto-detected if needed.

        Raises:
            ValueError: If `hla_embeds` has incorrect dimensions or incompatible encoder types.
        """
        self.hla_encoder = hla_encoder
        self.pept_encoder = pept_encoder
        self.device = get_device(device)[0]

        self.dataset = HlaDataSet(
            hla_df,
            [],
            None,
            protein_data,
            min_peptide_len=min_peptide_len,
            max_peptide_len=max_peptide_len,
        )
        self.hla_embeds = hla_embeds

        self.n_decoy_samples = 10000
        self.outlier_threshold = 0.005
        self.use_fmm_fdr = False
        self.decoy_rnd_seed = 1337
        self.d_model = 480
        self.verbose = True

    def get_embedding_distances(
        self, prot_embeds: np.ndarray, pept_embeds: np.ndarray, batch_size=1000000
    ) -> np.ndarray:
        """Compute pairwise Euclidean distances between protein and peptide embeddings.

        Uses `torch.cdist` for efficient batched computation on GPU.

        Args:
            prot_embeds: Embeddings for MHC alleles. Shape: (n_alleles, d_model)
            pept_embeds: Embeddings for peptides. Shape: (n_peptides, d_model)
            batch_size: Number of peptides processed per batch to avoid memory overflow.

        Returns:
            dist_matrix: Pairwise distance matrix. Shape: (n_peptides, n_alleles)

        Example:
            >>> prot_emb = np.random.rand(6, 480).astype(np.float32)
            >>> pept_emb = np.random.rand(100, 480).astype(np.float32)
            >>> dists = retriever.get_embedding_distances(prot_emb, pept_emb)
        """
        ret_dists = np.zeros((len(pept_embeds), len(prot_embeds)), dtype=np.float32)
        prot_embeds = torch.tensor(prot_embeds, device=self.device)

        for i in range(0, len(pept_embeds), batch_size):
            _embeds = torch.tensor(pept_embeds[i : i + batch_size], device=self.device)
            ret_dists[i : i + batch_size, :] = (
                torch.cdist(
                    _embeds.unsqueeze(0),
                    prot_embeds.unsqueeze(0),
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )

        return ret_dists

    def get_binding_distances(
        self,
        prot_embeds: np.ndarray,
        peptide_list,
        cdist_batch_size: int = 1000000,
        embed_batch_size: int = 1024,
    ) -> np.ndarray:
        """Embed peptides and compute their distances to given MHC allele embeddings.

        Args:
            prot_embeds: Precomputed MHC embeddings. Shape: (n_alleles, d_model)
            peptide_list: List or array of peptide sequences (strings).
            cdist_batch_size: Batch size for distance computation.
            embed_batch_size: Batch size for peptide embedding.

        Returns:
            dist_matrix: Distance from each peptide to each allele. Shape: (n_peptides, n_alleles)
        """
        if isinstance(peptide_list, np.ndarray):
            peptide_list = peptide_list.astype("U")

        pept_embeds = embed_peptides(
            self.pept_encoder,
            peptide_list,
            d_model=self.d_model,
            batch_size=embed_batch_size,
            device=self.device,
        )

        return self.get_embedding_distances(
            prot_embeds,
            pept_embeds,
            batch_size=cdist_batch_size,
        )

    def _get_decoy_distances(self, prot_embeds):
        """Generate decoy peptides and compute their sorted distances to alleles.

        Used for FDR/rank estimation.

        Args:
            prot_embeds: MHC embeddings for scoring. Shape: (n_alleles, d_model)

        Returns:
            sorted_decoy_dists: Sorted distances of decoy peptides. Shape: (n_decoys, n_alleles)
        """
        np.random.seed(self.decoy_rnd_seed)
        rnd_pept_df = self.dataset.digest.get_random_pept_df(self.n_decoy_samples)

        rnd_dist = self.get_binding_distances(
            prot_embeds, rnd_pept_df.sequence.values.astype("U")
        )
        return np.sort(rnd_dist, axis=0)

    def get_binding_metrics_for_embeds(
        self,
        prot_embeds: np.ndarray,
        peptide_list,
        keep_not_best_alleles: bool = False,
    ) -> pd.DataFrame:
        """Compute binding metrics for a list of peptides given their sequences or embeddings.

        Args:
            prot_embeds: Allele embeddings. Shape: (n_alleles, d_model)
            peptide_list: Either a list of peptide sequences or a numpy array of embeddings.
            keep_not_best_alleles: If True, include full distance matrix in output.

        Returns:
            df: DataFrame with columns:
                - sequence (if input was sequences)
                - best_allele_id: Index of best-matching allele
                - best_allele_dist: Minimum distance
                - best_allele_rank: Percentile rank among decoys (0–100)
        """
        if len(prot_embeds.shape) == 1:
            prot_embeds = prot_embeds[None, :]

        if isinstance(peptide_list, np.ndarray) and peptide_list.dtype == np.float32:
            has_seqs = False
            dist = self.get_embedding_distances(prot_embeds, peptide_list)
        else:
            has_seqs = True
            dist = self.get_binding_distances(prot_embeds, peptide_list)

        rnd_dist = self._get_decoy_distances(prot_embeds)

        best_allele_idxes = np.argmin(dist, axis=1)
        min_allele_distances = dist[
            np.arange(len(best_allele_idxes)), best_allele_idxes
        ]

        best_allele_ranks = get_binding_ranks(dist, rnd_dist)

        # fdrs = get_binding_fdr_for_best_allele(
        #     dist,
        #     rnd_dist,
        #     outlier_threshold=self.outlier_threshold,
        #     fmm_fdr=self.use_fmm_fdr,
        # )

        _dict = {}
        if has_seqs:
            _dict["sequence"] = peptide_list
        _dict.update(
            {
                "best_allele_id": best_allele_idxes,
                "best_allele_dist": min_allele_distances,
                "best_allele_rank": best_allele_ranks,
                # "best_allele_fdr": fdrs,
            }
        )
        df = pd.DataFrame(_dict)
        if keep_not_best_alleles:
            df.loc[:, list(range(prot_embeds.shape[0]))] = dist
        return df

    def get_binding_metrics_for_self_proteins(
        self,
        alleles,
        dist_threshold: float = 0,
        fdr: float = 0.02,
        cdist_batch_size: int = 1000000,
        embed_batch_size: int = 1024,
        get_sequence: bool = True,
    ) -> pd.DataFrame:
        """Screen internal proteome for potential self-reactive binders.

        Args:
            alleles: List of HLA allele names to consider.
            dist_threshold: Maximum allowed embedding distance.
            fdr: Maximum allowed false discovery rate.
            cdist_batch_size: Batch size for distance computation.
            embed_batch_size: Batch size for embedding peptides.
            get_sequence: If True, return actual sequences; else return indices.

        Returns:
            df: DataFrame of qualifying peptides with binding metrics and optionally sequences.
        """
        selected_embeds = self.hla_embeds[
            [self.dataset.allele_idxes_dict[allele][0] for allele in alleles]
        ].copy()

        decoy_dists = self._get_decoy_distances(selected_embeds)

        best_allele_idxes = np.empty_like(
            self.dataset.digest.digest_starts, dtype=np.int64
        )
        best_allele_dists = np.empty_like(best_allele_idxes, dtype=np.float32)
        best_allele_ranks = np.empty_like(best_allele_idxes, dtype=np.int32)
        best_allele_fdrs = np.empty_like(best_allele_dists)

        batches = range(0, len(best_allele_dists), cdist_batch_size)
        if self.verbose:
            batches = tqdm.tqdm(batches)
        for start_major in batches:
            if start_major + cdist_batch_size >= len(best_allele_dists):
                stop_major = len(best_allele_dists)
            else:
                stop_major = start_major + cdist_batch_size

            peptide_list = self.dataset.digest.get_peptide_seqs_from_idxes(
                np.arange(start_major, stop_major)
            )

            dist = self.get_binding_distances(
                selected_embeds,
                peptide_list,
                cdist_batch_size=cdist_batch_size,
                embed_batch_size=embed_batch_size,
            )

            best_allele_idxes[start_major:stop_major] = np.argmin(dist, axis=1)
            best_allele_dists[start_major:stop_major] = dist[
                np.arange(stop_major - start_major),
                best_allele_idxes[start_major:stop_major],
            ]

            best_allele_ranks[start_major:stop_major] = get_binding_ranks(
                dist, decoy_dists
            )
        for i in range(len(alleles)):
            idxes = best_allele_idxes == i
            best_allele_fdrs[idxes] = get_binding_fdrs(
                best_allele_dists[idxes],
                decoy_dists[:, i],
                outlier_threshold=self.outlier_threshold,
                fmm_fdr=self.use_fmm_fdr,
            )
        idxes = (best_allele_dists <= dist_threshold) & (best_allele_fdrs <= fdr)

        df = pd.DataFrame(
            dict(
                best_allele_id=best_allele_idxes[idxes],
                best_allele_dist=best_allele_dists[idxes],
                best_allele_rank=best_allele_ranks[idxes],
                best_allele_fdr=best_allele_fdrs[idxes],
            )
        )
        if get_sequence:
            peptides = self.dataset.digest.get_peptide_seqs_from_idxes(
                np.arange(len(best_allele_idxes))[idxes]
            )
            df["sequence"] = peptides
        else:
            df["peptide_id"] = np.arange(len(best_allele_idxes))[idxes]

        return df

    def get_binding_metrics_for_peptides(
        self,
        alleles,
        peptide_list,
        keep_not_best_alleles: bool = False,
    ) -> pd.DataFrame:
        """Score a list of peptides against specified HLA alleles.

        Args:
            alleles: Names of HLA alleles to evaluate.
            peptide_list: List of peptide sequences.
            keep_not_best_alleles: Whether to retain scores for all alleles.

        Returns:
            df: Binding metrics with added `best_allele` column mapping ID to name.
        """
        selected_embeds = self.hla_embeds[
            [self.dataset.allele_idxes_dict[allele][0] for allele in alleles]
        ].copy()

        df = self.get_binding_metrics_for_embeds(selected_embeds, peptide_list)

        if keep_not_best_alleles:
            df.rename(
                columns=dict(zip(list(range(len(alleles))), alleles, strict=False)),
                inplace=True,
            )
        df["best_allele"] = df.best_allele_id.apply(lambda i: alleles[i])

        return df
