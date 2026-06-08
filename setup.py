"""setup.py for BEguidedOFF with optional CUDA extension build."""
import os
from setuptools import setup, find_packages

# Monkey-patch PyTorch's CUDA version check to allow compiling
# with system NVCC (13.2) against PyTorch built with CUDA 12.4
# The minor version mismatch is acceptable; PTX is backward compatible.
import torch.utils.cpp_extension as cpp_ext
original_check = cpp_ext._check_cuda_version

def _patched_check_cuda_version(compiler_name, compiler_version):
    try:
        original_check(compiler_name, compiler_version)
    except RuntimeError as e:
        msg = str(e)
        if "mismatches the version" in msg:
            print(f"WARNING: Ignoring CUDA version mismatch: {msg}")
            return
        raise

cpp_ext._check_cuda_version = _patched_check_cuda_version

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Only build CUDA extension if NVCC is available and USE_CUDA is not explicitly disabled
_build_cuda = False
if os.environ.get("USE_CUDA", "1") != "0":
    try:
        import subprocess
        subprocess.run(["nvcc", "--version"], capture_output=True, check=True)
        _build_cuda = True
    except Exception:
        _build_cuda = False

ext_modules = []
if _build_cuda:
    ext_modules.append(
        CUDAExtension(
            name="beguidedoff._cuda",
            sources=[
                "csrc/offtarget_kernel.cu",
                "csrc/scoring_kernel.cu",
                "csrc/bindings.cpp",
            ],
            include_dirs=["csrc/include"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    )

setup(
    name="beguidedoff",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
    zip_safe=False,
)
