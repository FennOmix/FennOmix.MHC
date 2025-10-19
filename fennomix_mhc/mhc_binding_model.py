import math
import random

import numpy as np
import pandas as pd
import torch
import tqdm
from peptdeep.model.building_block import (
    Hidden_HFace_Transformer,
    PositionalEncoding,
    SeqAttentionSum,
    ascii_embedding,
)
from peptdeep.utils import get_available_device, logging
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from .constants._const import D_MODEL
from .mhc_utils import NonSpecificDigest

random.seed(1337)
np.random.seed(1337)
torch.random.manual_seed(1337)


# peptdeep has removed this function,
# copy it here as a local method.
def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """Creates a learning rate scheduler with linear warmup and cosine decay.

    The learning rate rises linearly during warmup steps, then follows a cosine
    decay curve. Useful for stabilizing early training.

    Args:
        optimizer: Optimizer to wrap with the scheduler.
        num_warmup_steps: Number of steps for linear warmup.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cosine cycles (default 0.5 for half-cycle).
        last_epoch: Index of last epoch (-1 for new training).

    Returns:
        LambdaLR: A PyTorch learning rate scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        # linear warmup phase
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)

        # cosine
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )

        cosine_lr_multiple = 0.5 * (
            1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)
        )
        return max(0.0, cosine_lr_multiple)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_ascii_indices(seq_array: list[str]) -> torch.LongTensor:
    """Converts a list of peptide sequences into ASCII-encoded index tensors.

    Each character in the peptide string is represented by its ASCII code,
    reshaped into a 2D tensor.

    Args:
        seq_array: List of peptide sequence strings (e.g., ['GLCTLVAML', ...]).

    Returns:
        A tensor of shape (batch_size, sequence_length), dtype=torch.long.
    """
    return torch.tensor(
        np.array(seq_array).view(np.int32).reshape(len(seq_array), -1),
        dtype=torch.long,
    )


class ModelSeqEncoder(torch.nn.Module):
    """Transformer-based encoder for peptide sequences."""

    def __init__(
        self, d_model: int = D_MODEL, layer_num: int = 4, dropout: float = 0.2
    ) -> None:
        """Initialize the sequence encoder.

        Args:
            d_model: Embedding dimension.
            layer_num: Number of Transformer layers.
            dropout: Dropout rate for Transformer layers.
        """

        super().__init__()
        self.embedding = ascii_embedding(d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=100)
        self.bert = Hidden_HFace_Transformer(
            hidden_dim=d_model, nlayers=layer_num, dropout=dropout
        )
        self.out_nn = SeqAttentionSum(d_model)

    def forward(self, aa_idxes: torch.Tensor) -> torch.Tensor:
        """Encode peptide sequences to embeddings.

        Args:
            aa_idxes: Tensor of shape (batch_size, seq_len) with ASCII indices.

        Returns:
            Normalized embedding tensor of shape (batch_size, d_model).
        """

        attention_mask = aa_idxes > 0
        x = self.embedding(aa_idxes)
        x = self.pos_encoder(x)
        x = self.bert(x, attention_mask)[0] * attention_mask.unsqueeze(-1)
        return torch.nn.functional.normalize(self.out_nn(x))


class ModelHlaEncoder(torch.nn.Module):
    """Transformer-based encoder for HLA embeddings."""

    def __init__(
        self, d_model: int = D_MODEL, layer_num: int = 1, dropout: float = 0.2
    ) -> None:
        """Initialize the HLA encoder.

        Args:
            d_model: Embedding dimension.
            layer_num: Number of Transformer layers.
            dropout: Dropout rate for Transformer layers.
        """

        super().__init__()
        self.nn = Hidden_HFace_Transformer(d_model, nlayers=layer_num, dropout=dropout)
        self.out_nn = SeqAttentionSum(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes variable-length HLA embeddings into fixed-size vectors.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model),
               typically from ESM models.

        Returns:
            Normalized embedding tensor of shape (batch_size, d_model).
        """

        attn_mask = (x != 0).any(dim=2)
        x = self.nn(x, attn_mask)[0] * attn_mask.unsqueeze(-1)
        return torch.nn.functional.normalize(self.out_nn(x))


class HlaDataSet(Dataset):
    """Dataset providing paired HLA embeddings and peptides for training."""

    def __init__(
        self,
        hla_df: pd.DataFrame,
        hla_esm_list: list[np.ndarray],
        pept_df: pd.DataFrame | None,
        protein_data: pd.DataFrame | list | str,
        min_peptide_len: int = 8,
        max_peptide_len: int = 14,
    ) -> None:
        """Initialize the dataset.

        Args:
            hla_df: DataFrame with HLA information; must have 'allele' column.
            hla_esm_list: List of HLA ESM embeddings corresponding to hla_df rows.
            pept_df: Peptide DataFrame with columns 'sequence' and 'allele'.
            protein_data: Protein FASTA path(s) or DataFrame to generate negatives.
            min_peptide_len: Minimum length for random digestion.
            max_peptide_len: Maximum length for random digestion.
        """
        self.hla_esm_list = hla_esm_list
        hla_df["hla_id"] = range(len(hla_df))
        self.allele_idxes_dict: dict = (
            hla_df.groupby("allele")["hla_id"].apply(list).to_dict()
        )
        self._expand_allele_names()
        self.hla_df = hla_df

        if pept_df is not None:
            self.pept_df = (
                pept_df.groupby("sequence")[["allele"]]
                .agg(list)
                .reset_index(drop=False)
            )
            self.pept_seq_list = self.pept_df.sequence
            self.pept_allele_list = self.pept_df.allele

        self.digest = NonSpecificDigest(protein_data, min_peptide_len, max_peptide_len)
        self.prob_pept_from_hla_df = 0.8

    def _expand_allele_names(self) -> None:
        """Add underscore-free allele names to ``allele_idxes_dict``."""

        self.allele_idxes_dict.update(
            [
                (allele.replace("_", ""), val)
                for allele, val in self.allele_idxes_dict.items()
            ]
        )

    def get_neg_pept(self) -> str:
        """Sample a negative peptide sequence.

        Returns:
            Random peptide string from the dataset or digested proteins.
        """
        if random.random() > self.prob_pept_from_hla_df:
            return self.pept_seq_list[random.randint(0, len(self.pept_seq_list) - 1)]
        idx = random.randint(0, len(self.digest.digest_starts) - 1)
        return self.digest.cat_protein_sequence[
            self.digest.digest_starts[idx] : self.digest.digest_stops[idx]
        ]

    def get_allele_embed(self, index: int) -> np.ndarray:
        """Get HLA embedding for a specific peptide.

        Args:
            index: Index of the peptide.

        Returns:
            Corresponding HLA embedding.
        """
        alleles = self.pept_allele_list[index]
        allele = alleles[random.randint(0, len(alleles) - 1)]
        hla_ids = self.allele_idxes_dict[allele]
        return self.hla_esm_list[hla_ids[random.randint(0, len(hla_ids) - 1)]]

    def __getitem__(self, index: int) -> tuple[np.ndarray, str, str]:
        """Returns a training triplet: (HLA embed, positive peptide, negative peptide).

        Args:
            index: Index of the sample.

        Returns:
            A tuple containing:
                - hla_embedding: HLA ESM embedding.
                - pos_peptide: Known binding peptide.
                - neg_peptide: Non-binding (negative) peptide.
        """
        return (
            self.get_allele_embed(index),
            self.pept_seq_list[index],
            self.get_neg_pept(),
        )

    def __len__(self) -> int:
        """Return number of peptide samples."""

        return len(self.pept_df)


def batchify_hla_esm_list(batch_esm_list: list[np.ndarray]) -> torch.Tensor:
    """Converts a list of variable-length HLA ESM embeddings into a padded tensor.

    Args:
        batch_esm_list: List of arrays, each of shape (1, seq_len, d_model).

    Returns:
        Padded tensor of shape (batch_size, max_seq_len, d_model).
    """
    max_hla_len = max(len(x) for x in batch_esm_list)
    hla_x = np.zeros(
        (len(batch_esm_list), max_hla_len, batch_esm_list[0].shape[-1]),
        dtype=np.float32,
    )
    for i, x in enumerate(batch_esm_list):
        hla_x[i, : len(x[0]), :] = x[0]
    return torch.tensor(hla_x, dtype=torch.float32)


def pept_hla_collate(
    batch: list[tuple[np.ndarray, str, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for creating batches from HlaDataSet.

    Handles variable-length HLA embeddings and ASCII-encodes peptides.

    Args:
        batch: List of tuples (hla_embed, pos_peptide, neg_peptide).

    Returns:
        A tuple of:
            - hla_tensor: Padded HLA embeddings.
            - pos_pept_tensor: ASCII-encoded positive peptides.
            - neg_pept_tensor: ASCII-encoded negative peptides.
    """

    hla_embeds = [x[0] for x in batch]
    pos_pept_array = [x[1] for x in batch]
    neg_pept_array = [x[2] for x in batch]
    max_hla_len = max(len(x) for x in hla_embeds)
    hla_x = np.zeros(
        (len(batch), max_hla_len, hla_embeds[0].shape[-1]), dtype=np.float32
    )
    for i, x in enumerate(hla_embeds):
        hla_x[i, : len(x[0]), :] = x[0]
    return (
        torch.tensor(hla_x, dtype=torch.float32),
        get_ascii_indices(pos_pept_array),
        get_ascii_indices(neg_pept_array),
    )


def get_hla_dataloader(
    dataset: HlaDataSet, batch_size: int, shuffle: bool
) -> DataLoader:
    """Creates a DataLoader for HlaDataSet with custom collation.

    Args:
        dataset: The dataset to load.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data each epoch.

    Returns:
        A DataLoader with pept_hla_collate as collate_fn.
    """
    return DataLoader(
        dataset=dataset,
        collate_fn=pept_hla_collate,
        batch_size=batch_size,
        shuffle=shuffle,
    )


class SiameseCELoss:
    """Contrastive Siamese loss for HLA-peptide similarity learning.

    Encourages the model to bring positive pairs closer and push negative pairs apart.
    Uses margin-based contrastive loss.
    """
    margin: float = 1

    def get_loss(
        self, hla_x: torch.Tensor, x: torch.Tensor, y: float = 1.0
    ) -> torch.Tensor:
        """Computes contrastive loss for one pair.

        Args:
            hla_x: HLA embedding tensor.
            x: Peptide embedding tensor.
            y: Label (1.0 for positive pair, 0.0 for negative).

        Returns:
            Scalar loss tensor.
        """
        diff = hla_x - x
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.mean(loss) / 2.0
        return loss

    def __call__(
        self, hla_x: torch.Tensor, pos_x: torch.Tensor, neg_x: torch.Tensor
    ) -> torch.Tensor:
        """Computes total Siamese loss from positive and negative triplets.

        Args:
            hla_x: HLA embedding.
            pos_x: Positive (binding) peptide embedding.
            neg_x: Negative (non-binding) peptide embedding.

        Returns:
            Combined loss tensor.
        """
        loss0 = self.get_loss(hla_x, pos_x, 1)
        loss1 = self.get_loss(hla_x, neg_x, 0)
        return (loss0 + loss1) / 2


def train(
    hla_encoder: ModelHlaEncoder,
    pept_encoder: ModelSeqEncoder,
    dataset: HlaDataSet,
    batch_size: int = 256,
    lr: float = 1e-4,
    epoch: int = 100,
    warmup_epoch: int = 20,
    verbose: bool = True,
    device: str = "cuda",
    test_bundle: tuple | None = None,
    neptune_run=None,
) -> None:
    """Train the peptide/HLA encoders.

    Args:
        hla_encoder: Encoder for HLA embeddings.
        pept_encoder: Encoder for peptide sequences.
        dataset: Training dataset.
        batch_size: Number of samples per batch.
        lr: Learning rate for the optimizer.
        epoch: Total number of epochs.
        warmup_epoch: Number of warmup epochs for the scheduler.
        verbose: Whether to print training progress.
        device: Device identifier for ``torch.device``.
        test_bundle: Optional tuple of test data passed to :func:`test`.
        neptune_run: Optional Neptune experiment for logging.
    """
    loss_func = SiameseCELoss()
    dataloader = get_hla_dataloader(dataset, batch_size, True)
    device = torch.device(device)
    hla_encoder.to(device)
    pept_encoder.to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": pept_encoder.parameters()},
            {"params": hla_encoder.parameters()},
        ],
        lr=lr,
    )
    if warmup_epoch > 0:
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_epoch,
            num_training_steps=epoch,
        )
    else:
        lr_scheduler = None

    if verbose:
        logging.info(f"{len(dataset)} training samples")
    for i_epoch in range(epoch):
        hla_encoder.train()
        pept_encoder.train()
        loss_list = []
        for hla_x, pos_x, neg_x in dataloader:
            hla_x = hla_encoder(hla_x.to(device))
            pos_x = pept_encoder(pos_x.to(device))
            neg_x = pept_encoder(neg_x.to(device))
            loss = loss_func(hla_x, pos_x, neg_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        if lr_scheduler:
            lr_scheduler.step()
            _lr = lr_scheduler.get_last_lr()[0]
        else:
            _lr = lr
        mean_loss = np.mean(loss_list)
        if verbose:
            logging.info(f"[Epoch={i_epoch}] loss={mean_loss:.5f}, lr={_lr:.3e}")

        if test_bundle:
            test_df, test_allele_list, hla_df, hla_esm_list, fasta_list = test_bundle
            (
                mean_rank01_recall_rate,
                mean_rank05_recall_rate,
                mean_rank20_recall_rate,
            ) = test(
                test_df,
                test_allele_list,
                hla_encoder,
                pept_encoder,
                hla_df,
                hla_esm_list,
                fasta_list,
            )
            print(
                f"test alleles rank%<0.1 average recall rate: {mean_rank01_recall_rate:.2f}"
            )
            print(
                f"test alleles rank%<0.5 average recall rate: {mean_rank05_recall_rate:.2f}"
            )
            print(
                f"test alleles rank%<2 average recall rate: {mean_rank20_recall_rate:.2f}"
            )
        if neptune_run:
            neptune_run["train/loss"].log(mean_loss)
            neptune_run["train/lr"].log(_lr)
            if test_bundle:
                neptune_run["test/loss1"].log(mean_rank01_recall_rate)
                neptune_run["test/loss2"].log(mean_rank05_recall_rate)
                neptune_run["test/loss3"].log(mean_rank20_recall_rate)


def embed_hla_esm_list(
    hla_encoder: ModelHlaEncoder,
    hla_esm_list: list[np.ndarray],
    batch_size: int = 200,
    device: str | torch.device | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Generates fixed-size embeddings for a list of HLA ESM features.

    Args:
        hla_encoder: Trained HLA encoder model.
        hla_esm_list: List of raw ESM embeddings for HLA alleles.
        batch_size: Inference batch size.
        device: Device to use. Auto-detected if None.
        verbose: Show progress bar.

    Returns:
        Array of shape (num_hla, d_model) containing encoded HLA embeddings.
    """
    if not device:
        device = get_available_device()[0]
    hla_encoder.to(device)
    hla_encoder.eval()
    embeds = np.zeros((len(hla_esm_list), hla_esm_list[0].shape[-1]), dtype=np.float32)
    with torch.no_grad():
        batches = range(0, len(hla_esm_list), batch_size)
        if verbose:
            batches = tqdm.tqdm(batches)
        for i in batches:
            x = batchify_hla_esm_list(hla_esm_list[i : i + batch_size]).to(device)
            embeds[i : i + batch_size] = hla_encoder(x).detach().cpu().numpy()
    torch.cuda.empty_cache()
    return embeds


def embed_peptides(
    pept_encoder: ModelSeqEncoder,
    seqs: list[str],
    d_model: int = D_MODEL,
    batch_size: int = 512,
    device: str | torch.device | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Encodes a list of peptide sequences into embeddings.

    Args:
        pept_encoder: Trained peptide encoder model.
        seqs: List of peptide strings.
        d_model: Expected embedding dimension.
        batch_size: Inference batch size.
        device: Device to use (auto-detected if None).
        verbose: Show progress bar.

    Returns:
        Array of shape (num_peptides, d_model) with peptide embeddings.
    """
    if not device:
        device = get_available_device()[0]
    pept_encoder.to(device)
    pept_encoder.eval()
    embeds = np.zeros((len(seqs), d_model), dtype=np.float32)
    with torch.no_grad():
        batches = range(0, len(seqs), batch_size)
        if verbose:
            batches = tqdm.tqdm(batches)
        for i in batches:
            x = get_ascii_indices(seqs[i : i + batch_size]).to(device)
            embeds[i : i + batch_size, :] = pept_encoder(x).detach().cpu().numpy()
    torch.cuda.empty_cache()
    return embeds


def test(
    test_df: pd.DataFrame,
    test_allele_list,
    hla_encoder: ModelHlaEncoder,
    pept_encoder: ModelSeqEncoder,
    hla_df: pd.DataFrame,
    hla_esm_list: list[np.ndarray],
    fasta_list: list[str],
) -> tuple[float, float, float]:
    """Evaluates model performance on test alleles using rank-based recall.

    Args:
        test_df: DataFrame with test peptide-allele pairs.
        test_allele_list: List of HLA alleles to evaluate.
        hla_encoder: Trained HLA encoder.
        pept_encoder: Trained peptide encoder.
        hla_df: HLA metadata DataFrame.
        hla_esm_list: List of raw HLA ESM embeddings.
        fasta_list: List of protein FASTA file paths.

    Returns:
        Tuple of mean recall rates at rank < 0.1, < 0.5, and < 2.0.
    """
    from .mhc_binding_retriever import MHCBindingRetriever

    hla_embeds = embed_hla_esm_list(hla_encoder, hla_esm_list)
    retriever = MHCBindingRetriever(
        hla_encoder, pept_encoder, hla_df, hla_embeds, fasta_list
    )

    retriever.n_decoy_samples = 1000000
    pept_groups = test_df.groupby("allele")
    rank01_list = []
    rank05_list = []
    rank20_list = []
    for i in range(len(test_allele_list)):
        tmp_allele = test_allele_list[i]
        pept_df = pept_groups.get_group(tmp_allele)
        embed = retriever.hla_embeds[retriever.dataset.allele_idxes_dict[tmp_allele][0]]
        df = retriever.get_binding_metrics_for_embeds(embed, pept_df.sequence.values)
        rank01_list.append(len(df.query("best_allele_rank<=0.1")) / len(df))
        rank05_list.append(len(df.query("best_allele_rank<=0.5")) / len(df))
        rank20_list.append(len(df.query("best_allele_rank<=2")) / len(df))

    return np.mean(rank01_list), np.mean(rank05_list), np.mean(rank20_list)
