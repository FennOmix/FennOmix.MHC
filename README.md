# Fennet-MHC

[![GitHub Release](https://img.shields.io/github/v/release/FennOmix/FeNNet.MHC?logoColor=green&color=brightgreen)](https://github.com/FennOmix/FeNNet.MHC/releases)
![Versions](https://img.shields.io/badge/python-3.10_%7C_3.11_%7C_3.12-brightgreen)
![License](https://img.shields.io/badge/License-Apache-brightgreen)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/FennOmix/FeNNet.MHC/e2e_testing.yml?branch=main&label=E2E%20Tests)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/FennOmix/FeNNet.MHC/pip_installation.yml?branch=main&label=Unit%20Tests)
![Docs](https://readthedocs.org/projects/fennet.mhc/badge/?version=latest)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/FennOmix/FeNNet.MHC/publish_docker_image.yml?branch=main&label=Deploy%20Docker)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/FennOmix/FeNNet.MHC/publish_on_pypi.yml?branch=main&label=Deploy%20PyPi)
![Coverage](https://github.com/FennOmix/FeNNet.MHC/blob/main/coverage.svg)
[![branch-checks](https://github.com/FennOmix/FeNNet.MHC/actions/workflows/branch-checks.yml/badge.svg)](https://github.com/FennOmix/FeNNet.MHC/actions/workflows/branch-checks.yml)

Foundation model for MHC class I peptide binding prediction built on deep contrastive learning.

See the [online documentation](https://fennet.mhc.readthedocs.io/en/latest) for
full API details and tutorials.

## Installation

Install the latest release from PyPI:

```bash
pip install fennet-mhc
```

Or install the development version directly from GitHub:

```bash
pip install git+https://github.com/FennOmix/FeNNet.MHC.git
```

## Command line interface

After installation the `fennet-mhc` command exposes several sub-commands.  The examples below assume your peptide or protein sequences are stored in FASTA or tabular files.

### Embed MHC proteins

```bash
fennet-mhc embed-proteins --fasta my_hla.fasta --out-folder ./output
```

### Embed peptides

```bash
fennet-mhc embed-peptides --peptide-file peptides.tsv --out-folder ./output
```

### Predict epitopes for MHC alleles

```bash
fennet-mhc predict-epitopes-for-mhc --peptide-file peptides.tsv \
    --alleles A02_01,B07_02 --out-folder ./output
```

### Predict MHC binders for given epitopes

```bash
fennet-mhc predict-mhc-binders-for-epitopes --peptide-file peptides.tsv \
    --out-folder ./output
```

Additional commands `deconvolute-peptides` and `deconvolute-and-predict-peptides` are also available.

## Pipeline API

All functionality of the command line interface is available through the `fennet_mhc.pipeline_api` module:

```python
from fennet_mhc.pipeline_api import (
    embed_proteins,
    embed_peptides_from_file,
    predict_epitopes_for_mhc,
    predict_mhc_binders_for_epitopes,
)

# compute and save embeddings
embed_proteins("my_hla.fasta", "./output")
embed_peptides_from_file("peptides.tsv", "./output")

# run predictions using the saved files
predict_epitopes_for_mhc(
    "peptides.tsv",
    ["A02_01"],
    "./output",
)
```
