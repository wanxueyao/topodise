#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <gpu-index> <config-path> <worker-id>" >&2
  exit 2
fi

GPU_INDEX="$1"
CONFIG_PATH="$2"
WORKER_ID="$3"
CODES_ROOT="/data/regulusWXY/neo_PD"
PYTHON_BIN="$CODES_ROOT/.venv-c8/bin/python"
PARENT_SITE="/data/regulusWXY/anaconda3/envs/mmgraphrag/lib/python3.10/site-packages"
LOCAL_SITE="$CODES_ROOT/.venv-c8/lib/python3.10/site-packages"

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$LOCAL_SITE/nvidia/nccl/lib:$LOCAL_SITE/nvidia/nvshmem/lib:$PARENT_SITE/nvidia/cublas/lib:$PARENT_SITE/nvidia/cuda_cupti/lib:$PARENT_SITE/nvidia/cuda_nvrtc/lib:$PARENT_SITE/nvidia/cuda_runtime/lib:$PARENT_SITE/nvidia/cudnn/lib:$PARENT_SITE/nvidia/cufft/lib:$PARENT_SITE/nvidia/cufile/lib:$PARENT_SITE/nvidia/curand/lib:$PARENT_SITE/nvidia/cusolver/lib:$PARENT_SITE/nvidia/cusparse/lib:$PARENT_SITE/nvidia/cusparselt/lib:$PARENT_SITE/nvidia/nvjitlink/lib:$PARENT_SITE/nvidia/nvtx/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$CODES_ROOT"

"$PYTHON_BIN" -u -m experiments.unified_stage2_encoder_comparison.run_matrix \
  --config "$CONFIG_PATH" \
  --encoders topodise_stage1 \
  --tasks pd_stage pd_subtype_tdrest \
  --worker-id "$WORKER_ID"

"$PYTHON_BIN" -u -m experiments.unified_stage2_encoder_comparison.analyze \
  --config "$CONFIG_PATH" \
  --bootstrap 10000
