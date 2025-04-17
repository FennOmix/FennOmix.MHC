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
    """
    Create a learning rate schedule that linearly increases the learning rate from
    0.0 to lr over num_warmup_steps, then decreases to 0.0 on a cosine schedule over
    the remaining num_training_steps-num_warmup_steps (assuming num_cycles = 0.5).

    This is based on the Hugging Face implementation
    https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to
            schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (float): The number of waves in the cosine schedule. Defaults to 0.5
            (decrease from the max value to 0 following a half-cosine).
        last_epoch (int): The index of the last epoch when resuming training. Defaults to -1

    Returns:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule.
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


def get_ascii_indices(seq_array) -> torch.LongTensor:
    return torch.tensor(
        np.array(seq_array).view(np.int32).reshape(len(seq_array), -1),
        dtype=torch.long,
    )


class ModelSeqEncoder(torch.nn.Module):
    def __init__(self, d_model=480, layer_num=4, dropout=0.2):
        super().__init__()
        self.embedding = ascii_embedding(d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=100)
        self.bert = Hidden_HFace_Transformer(
            hidden_dim=d_model, nlayers=layer_num, dropout=dropout
        )
        self.out_nn = SeqAttentionSum(d_model)

    def forward(self, aa_idxes):
        attention_mask = aa_idxes > 0
        x = self.embedding(aa_idxes)
        x = self.pos_encoder(x)
        x = self.bert(x, attention_mask)[0] * attention_mask.unsqueeze(-1)
        return torch.nn.functional.normalize(self.out_nn(x))


class ModelHlaEncoder(torch.nn.Module):
    def __init__(self, d_model=480, layer_num=1, dropout=0.2):
        super().__init__()
        self.nn = Hidden_HFace_Transformer(d_model, nlayers=layer_num, dropout=dropout)
        self.out_nn = SeqAttentionSum(d_model)

    def forward(self, x: torch.Tensor):
        attn_mask = (x != 0).any(dim=2)
        x = self.nn(x, attn_mask)[0] * attn_mask.unsqueeze(-1)
        return torch.nn.functional.normalize(self.out_nn(x))


class HlaDataSet(Dataset):
    def __init__(
        self,
        hla_df,
        hla_esm_list,
        pept_df,
        protein_data,
        digested_pept_lens=(8, 14),
    ):
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

        self.digest = NonSpecificDigest(protein_data, digested_pept_lens)
        self.prob_pept_from_hla_df = 0.8

    def _expand_allele_names(self):
        self.allele_idxes_dict.update(
            [
                (allele.replace("_", ""), val)
                for allele, val in self.allele_idxes_dict.items()
            ]
        )

    def get_neg_pept(self):
        if random.random() > self.prob_pept_from_hla_df:
            return self.pept_seq_list[random.randint(0, len(self.pept_seq_list) - 1)]
        else:
            idx = random.randint(0, len(self.digest.digest_starts) - 1)
            return self.digest.cat_protein_sequence[
                self.digest.digest_starts[idx] : self.digest.digest_stops[idx]
            ]

    def get_allele_embed(self, index):
        alleles = self.pept_allele_list[index]
        allele = alleles[random.randint(0, len(alleles) - 1)]
        hla_ids = self.allele_idxes_dict[allele]
        return self.hla_esm_list[hla_ids[random.randint(0, len(hla_ids) - 1)]]

    def __getitem__(self, index):
        return (
            self.get_allele_embed(index),
            self.pept_seq_list[index],
            self.get_neg_pept(),
        )

    def __len__(self):
        return len(self.pept_df)


def batchify_hla_esm_list(batch_esm_list):
    max_hla_len = max(len(x) for x in batch_esm_list)
    hla_x = np.zeros(
        (len(batch_esm_list), max_hla_len, batch_esm_list[0].shape[-1]),
        dtype=np.float32,
    )
    for i, x in enumerate(batch_esm_list):
        hla_x[i, : len(x[0]), :] = x[0]
    return torch.tensor(hla_x, dtype=torch.float32)


def pept_hla_collate(batch):
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


def get_hla_dataloader(dataset: HlaDataSet, batch_size, shuffle):
    return DataLoader(
        dataset=dataset,
        collate_fn=pept_hla_collate,
        batch_size=batch_size,
        shuffle=shuffle,
    )


class SiameseCELoss:
    margin: float = 1

    def get_loss(self, hla_x, x, y=1.0):
        diff = hla_x - x
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.mean(loss) / 2.0
        return loss

    def __call__(self, hla_x, pos_x, neg_x):
        # euclidian distance
        loss0 = self.get_loss(hla_x, pos_x, 1)
        loss1 = self.get_loss(hla_x, neg_x, 0)
        return (loss0 + loss1) / 2


def train(
    hla_encoder: ModelHlaEncoder,
    pept_encoder: ModelSeqEncoder,
    dataset: HlaDataSet,
    batch_size=256,
    lr=1e-4,
    epoch=100,
    warmup_epoch=20,
    verbose=True,
    device="cuda",
    test_bundle=None,
    neptune_run=None,
):
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
    hla_encoder, hla_esm_list, batch_size=200, device=None, verbose=False
):
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
    pept_encoder, seqs, d_model=480, batch_size=512, device=None, verbose=False
):
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
    hla_encoder,
    pept_encoder,
    hla_df,
    hla_esm_list,
    fasta_list,
):
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
