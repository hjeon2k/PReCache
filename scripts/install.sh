#!/usr/bin/env bash
# Python 3.11 env: pinned stack, PReCache patches, Flash-LoRA-Attention.  TORCH_CUDA_ARCH_LIST=8.0 bash scripts/install.sh
set -euo pipefail
cd "$(dirname "$0")/.."

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install ninja packaging setuptools wheel
pip install --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
for p in patches/*.patch; do
    patch -d "$SITE" -p1 < "$p"
done

pip install --no-build-isolation ./src/flash_lora_attn
python -c "import flash_lora_attn, transformers.cache_utils as c; assert hasattr(c, 'LRCache'); print('PReCache ready')"
