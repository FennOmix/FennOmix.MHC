# Fennet-MHC

Foundation model for MHC class I peptide binding prediction built on deep contrastive learning.

See the [online documentation](https://alphabase.readthedocs.io/en/latest) for
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

