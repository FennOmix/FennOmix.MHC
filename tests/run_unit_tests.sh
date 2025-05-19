#!/bin/bash

ENV_NAME=${1:-mhc}
INCLUDE_SLOW_TESTS=${2:-false}

if [ "$(echo $INCLUDE_SLOW_TESTS | tr '[:upper:]' '[:lower:]')" = "true" ]; then
  conda run -n $ENV_NAME --no-capture-output coverage run --source=../fennet_mhc -m pytest
else
  conda run -n $ENV_NAME --no-capture-output coverage run --source=../fennet_mhc -m pytest -k 'not slow'
fi

# Test cli
conda run -n $ENV_NAME --no-capture-output fennet-mhc embed-proteins --fasta ./test_data/test_MHC_proteins.fasta --out-folder ./nogit
conda run -n $ENV_NAME --no-capture-output fennet-mhc embed-peptides --peptide-file ./test_data/test_peptides.fasta --out-folder ./nogit
conda run -n $ENV_NAME --no-capture-output fennet-mhc embed-peptides --peptide-file ./test_data/test_peptides.tsv --out-folder ./nogit
conda run -n $ENV_NAME --no-capture-output fennet-mhc predict-epitopes-for-mhc --peptide-file ./test_data/test_peptides.tsv --alleles A02_01 --out-folder nogit
conda run -n $ENV_NAME --no-capture-output fennet-mhc predict-mhc-binders-for-epitopes --peptide-file ./test_data/test_peptides.tsv --out-folder nogit
# conda run -n $ENV_NAME --no-capture-output fennet-mhc deconvolute-peptides --peptide-file ./test_data/test_peptides.tsv --n-centroids 2 --out-folder nogit
