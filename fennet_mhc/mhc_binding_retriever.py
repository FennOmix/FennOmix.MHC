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

    Args:
        dists: Distances between target embeddings.
        rnd_dists: Distances for decoy embeddings.
        alpha: Scaling factor defined as ``len(targets) / len(decoys)``.
        remove_rnd_top_rank: Fraction of the lowest decoy distances to ignore
            when computing FDRs.

    Returns:
        Array of FDR values for each element in ``dists``.
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
    """Convert FDRs to q-values.

    Args:
        fdrs: Array of FDR values.
        distances: Distances used for ranking.

    Returns:
        Array of q-values sorted according to ``distances``.
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
    """Estimate FDRs for a set of distances.

    This function can either perform a simple target-decoy FDR estimation or use
    a finite mixture model (FMM) when ``fmm_fdr`` is ``True``.

    Args:
        distances_1D: Target distance values.
        decoys_1D: Decoy distance values.
        max_fitting_samples: Maximum number of target samples used for fitting
            the FMM.
        random_state: Random seed used when subsampling ``distances_1D``.
        outlier_threshold: Fraction of decoys ignored as outliers.
        fmm_fdr: Whether to estimate FDR using an FMM model.

    Returns:
        Array of FDR values corresponding to ``distances_1D``.
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
    """Calculate FDRs for the best allele of each peptide.

    Args:
        distances: Matrix of distances with shape ``(n_peptides, n_alleles)``.
        rnd_dist: Sorted decoy distance matrix with the same second dimension as
            ``distances``.
        outlier_threshold: Fraction of decoys ignored when estimating FDR.
        fmm_fdr: Whether to use the FMM based FDR estimation.

    Returns:
        Array of FDR values for the best allele of each peptide.
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
    """Rank each peptide against a decoy distribution.

    Args:
        distances: Distance matrix of shape ``(n_peptides, n_alleles)``.
        sorted_rnd_dist: Sorted decoy distance matrix of the same shape as
            ``distances``.

    Returns:
        Array containing the percentile rank of the best allele distance for
        each peptide.
    """
    best_allele_idxes = np.argmin(distances, axis=1)
    best_allele_ranks = np.zeros(len(best_allele_idxes))
    len_rnd = float(sorted_rnd_dist.shape[0])
    for i, allele_idx in enumerate(best_allele_idxes):
        rank = np.searchsorted(sorted_rnd_dist[:, allele_idx], distances[i, allele_idx])
        best_allele_ranks[i] = rank / len_rnd * 100
    return best_allele_ranks


class MHCBindingRetriever:
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
        """Create a retriever for peptide--MHC binding metrics.

        Args:
            hla_encoder: Trained HLA encoder model.
            pept_encoder: Trained peptide encoder model.
            hla_df: DataFrame describing the HLA alleles.
            hla_embeds: Pre-computed embeddings for each allele in ``hla_df``.
            protein_data: Protein sequences used to generate decoy peptides.
            min_peptide_len: Minimum peptide length used for digestion.
            max_peptide_len: Maximum peptide length used for digestion.
            device: Torch device string used for computations.
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
        """Compute Euclidean distances between peptide and allele embeddings.

        Args:
            prot_embeds: Array of allele embeddings of shape ``(n_alleles, d)``.
            pept_embeds: Array of peptide embeddings of shape ``(n_peptides, d)``.
            batch_size: Number of peptides processed per batch when computing
                pairwise distances.

        Returns:
            Distance matrix with shape ``(n_peptides, n_alleles)``.
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
        """Embed peptides and compute distances to allele embeddings.

        Args:
            prot_embeds: Allele embeddings.
            peptide_list: Iterable of peptide sequences.
            cdist_batch_size: Batch size used when computing pairwise distances.
            embed_batch_size: Batch size used when embedding peptides.

        Returns:
            Distance matrix with shape ``(n_peptides, n_alleles)``.
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
        """Generate random peptides and compute their distances to alleles."""
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
        """Return binding metrics for a set of peptides.

        Args:
            prot_embeds: Allele embedding matrix.
            peptide_list: Iterable of peptide sequences or precomputed peptide
                embeddings.
            keep_not_best_alleles: If ``True`` also return the full distance
                matrix for each peptide.

        Returns:
            DataFrame with columns ``best_allele_id``, ``best_allele_dist`` and
            ``best_allele_rank``. If ``peptide_list`` consists of sequences the
            ``sequence`` column is also included.
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
        """Screen the internal protein database for potential binders.

        Args:
            alleles: List of allele names to consider.
            dist_threshold: Distance cut-off for reporting peptides.
            fdr: Maximum FDR allowed for reported peptides.
            cdist_batch_size: Batch size used when computing pairwise distances.
            embed_batch_size: Batch size used when embedding peptides.
            get_sequence: If ``True`` return peptide sequences, otherwise return
                peptide indices.

        Returns:
            DataFrame of peptides passing the specified thresholds.
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
        """Calculate binding metrics for the provided peptide list.

        Args:
            alleles: List of allele names to score against.
            peptide_list: Iterable of peptide sequences.
            keep_not_best_alleles: If ``True`` also keep the full distance matrix
                for each allele.

        Returns:
            DataFrame with binding metrics for each peptide.
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
