#!/bin/bash

HERE=$(dirname "$(readlink -f $0)")
PARENT_DIR=$(dirname "$HERE")

docker run -it --rm --name triton_tritony \
    -p8100:8000   \
    -p8101:8001   \
    -p8102:8002    \
    -v "${PARENT_DIR}"/model_repository:/model_repository:ro \
    -e OMP_NUM_THREADS=4 \
    -e OPENBLAS_NUM_THREADS=4 \
    -e HF_TOKEN=$HF_TOKEN \
    --shm-size=4g  \
    triton-vad-server:latest \
    tritonserver --model-repository=/model_repository \
    --exit-timeout-secs 15 \
    --min-supported-compute-capability 7.0 \
