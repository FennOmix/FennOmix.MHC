import os
import pickle
import typing

import numpy as np
import pandas as pd
from alphabase.protein.fasta import load_all_proteins
from alphabase.protein.lcp_digest import get_substring_indices

import logging
import ssl
import urllib
import re
from zipfile import ZipFile
from fennet.mhc.settings import global_settings

FOUNDATION_MODEL_DIR = os.path.join(
    os.path.join(
        os.path.expanduser(global_settings["FENNETMHC_HOME"]), "foundation_model"
    )
)

LOCAL_MODEL_ZIP_NAME = global_settings["local_model_zip_name"]
MODEL_URL = global_settings["model_url"]
MODEL_ZIP_FILE_PATH = os.path.join(FOUNDATION_MODEL_DIR, LOCAL_MODEL_ZIP_NAME)

MODEL_DOWNLOAD_INSTRUCTIONS = (
    "Please download the "
    f'zip. file by yourself from "{MODEL_URL}".'
)


def prepare_models(
    url: str = MODEL_URL,
    target_zip_path: str = MODEL_ZIP_FILE_PATH,
    extract_to_path: str = FOUNDATION_MODEL_DIR,
    overwrite: bool = True,
):
    """
    Download and extract model files.

    Parameters
    ----------
    url : str, optional
        Remote path to the model zip file.
        Defaults to :data:`fennet.mhc.models.MODEL_URL`.

    target_zip_path : str, optional
        Target file path for the downloaded zip file.
        Defaults to :data:`fennet.mhc.models.MODEL_ZIP_FILE_PATH`.

    extract_to_path : str, optional
        Directory where the extracted model files will be saved.
        Defaults to :data:`fennet.mhc.models.FOUNDATION_MODEL_DIR`.

    overwrite : bool, optional
        Overwrite existing model files if they already exist.
        Defaults to True.

    Returns
    -------
    tuple[str, str] or bool
        If successful, returns paths to the extracted HLA model file and Pept model file.
        If unsuccessful, returns False.

    """

    download_models(url=url, target_path=target_zip_path, overwrite=overwrite)

    HLA_model_path, pept_model_path = extract_model_zip(downloaded_zip=target_zip_path, extract_to_path=extract_to_path)
    
    if HLA_model_path and pept_model_path:
        if os.path.exists(HLA_model_path) and os.path.exists(pept_model_path):
            return HLA_model_path, pept_model_path

    return None, None


def download_models(
    url: str = MODEL_URL, target_path: str = MODEL_ZIP_FILE_PATH, overwrite: bool = True
):
    if not overwrite and os.path.exists(target_path):
        raise FileExistsError(f"Model file already exists: {target_path}")
    
    logging.info(f"Downloading models from {url} to {target_path} ...")
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        context = ssl._create_unverified_context()
        requests = urllib.request.urlopen(url, context=context, timeout=10)
        with open(target_path, "wb") as f:
            f.write(requests.read())
    except Exception as e:
        raise FileNotFoundError(
            f"Downloading model failed: {e}.\n" + MODEL_DOWNLOAD_INSTRUCTIONS
        ) from e

    logging.info(f"Successfully downloaded models.")


def extract_model_zip(downloaded_zip, extract_to_path):
    HLA_pattern = re.compile(r"HLA_model_v\d+\.pt")
    pept_pattern = re.compile(r"pept_model_v\d+\.pt")

    HLA_model_path = None
    pept_model_path = None

    os.makedirs(extract_to_path, exist_ok=True)

    with ZipFile(download_models) as zip:
        for filename in zip.namelist():
            if HLA_pattern.match(filename):
                target_path = os.path.join(extract_to_path, os.path.basename(filename))
                try:
                    with zip.open(filename) as source, open(target_path, "wb") as target:
                        target.write(source.read())
                    HLA_model_path = target_path
                    logging.info(f"Successfully extracted HLA model file: {filename} -> {target_path}")
                except Exception as e:
                    logging.error(f"Extracting HLA model file failed: {e}.\n")

            if pept_pattern.match(filename):
                target_path = os.path.join(extract_to_path, os.path.basename(filename))
                try:
                    with zip.open(filename) as source, open(target_path, "wb") as target:
                        target.write(source.read())
                    pept_model_path = target_path
                    logging.info(f"Successfully extracted pept model file: {filename} -> {target_path}")
                except Exception as e:
                    logging.error(f"Extracting Pept model file failed: {e}.\n")

    return HLA_model_path, pept_model_path


def load_esm_pkl(fname="hla1_esm_embeds.pkl"):
    with open(fname, "rb") as f:
        _dict = pickle.load(f)
        return _dict["protein_df"], _dict["embedding_list"]


def load_hla_pep_df(folder=r"x:\Feng\HLA-DB\all_alleles\mixmhcpred", rank=2):
    df_list = []
    for fname in os.listdir(folder):
        df = pd.read_table(os.path.join(folder, fname), skiprows=11)
        df = df.query(f"`%Rank_bestAllele`<={rank}").copy()
        df["sequence"] = df["Peptide"]
        df = df[["sequence"]]
        df["allele"] = fname[:-4]
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


class NonSpecificDigest:
    def __init__(
        self, protein_data: typing.Tuple[pd.DataFrame, list, str], lens=[8, 14]
    ):
        if isinstance(protein_data, pd.DataFrame):
            self.cat_protein_sequence = (
                "$" + "$".join(protein_data.sequence.values) + "$"
            )
        else:
            if isinstance(protein_data, str):
                protein_data = [protein_data]
            protein_dict = load_all_proteins(protein_data)
            self.cat_protein_sequence = (
                "$" + "$".join([_["sequence"] for _ in protein_dict.values()]) + "$"
            )
        self.digest_starts, self.digest_stops = get_substring_indices(
            self.cat_protein_sequence, lens[0], lens[1]
        )

    def get_random_pept_df(self, n=5000):
        idxes = np.random.randint(0, len(self.digest_starts), size=n)
        df = pd.DataFrame(
            [
                self.cat_protein_sequence[start:stop]
                for start, stop in zip(
                    self.digest_starts[idxes], self.digest_stops[idxes]
                )
            ],
            columns=["sequence"],
        )
        df["allele"] = "random"
        return df

    def get_peptide_seqs_from_idxes(self, idxes):
        return [
            self.cat_protein_sequence[start:stop]
            for start, stop in zip(self.digest_starts[idxes], self.digest_stops[idxes])
        ]
