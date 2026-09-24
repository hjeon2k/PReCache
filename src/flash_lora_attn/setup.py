"""Flash-LoRA-Attention extension: pip install --no-build-isolation ./src/flash_lora_attn (arch: TORCH_CUDA_ARCH_LIST)."""
import os
import subprocess
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = Path(__file__).parent.resolve()
csrc = here / "csrc"
cutlass = here / "third_party" / "cutlass"

if not (cutlass / "include").is_dir():
    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", "v3.6.0",
                           "https://github.com/NVIDIA/cutlass.git", str(cutlass)])

archs = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0").replace(" ", ";").split(";")
gencode = []
for a in filter(None, archs):
    sm = a.replace(".", "").replace("+PTX", "")
    gencode += ["-gencode", f"arch=compute_{sm},code=sm_{sm}"]

setup(
    name="flash_lora_attn",
    version="0.1.0",
    packages=["flash_lora_attn"],
    ext_modules=[
        CUDAExtension(
            name="flash_lora_attn_2_cuda",
            sources=[str(csrc / "flash_api.cpp")] + sorted(str(p) for p in (csrc / "src").glob("*.cu")),
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3", "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
                    "-DFLASHATTENTION_DISABLE_DROPOUT", "-DFLASHATTENTION_DISABLE_ALIBI",
                    "--threads", "2",
                ] + gencode,
            },
            include_dirs=[csrc, csrc / "src", cutlass / "include"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
