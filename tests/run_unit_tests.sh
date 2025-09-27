#!/bin/bash

ENV_NAME=${1:-mhc}
INCLUDE_SLOW_TESTS=${2:-false}

if [ "$(echo $INCLUDE_SLOW_TESTS | tr '[:upper:]' '[:lower:]')" = "true" ]; then
  conda run -n $ENV_NAME --no-capture-output coverage run --source=../fennomix_mhc -m pytest
else
  conda run -n $ENV_NAME --no-capture-output coverage run --source=../fennomix_mhc -m pytest -k 'not slow'
fi
