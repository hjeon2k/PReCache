#!/usr/bin/env bash
# Python 3.12 env for the vLLM benchmark (vLLM 0.28.0, CUDA 12.9).  TORCH_CUDA_ARCH_LIST=8.0 bash scripts/install_vllm.sh
set -euo pipefail
cd "$(dirname "$0")/.."

pip install "https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu129
pip install ninja packaging setuptools wheel
pip install --no-build-isolation ./src/flash_lora_attn
python -c "import torch, vllm, flash_lora_attn_2_cuda as k; assert hasattr(k, 'fwd_kvcache'); print('vLLM', vllm.__version__, 'ready')"
