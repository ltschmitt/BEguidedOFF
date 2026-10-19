"""setup.py for BEguidedOFF.

Two native extensions are built, both optional, so that a plain install never
touches torch/CUDA:

* ``beguidedoff._fmindex`` -- sdsl-lite FM-index binding used by the **fast**
  CPU mode.  Built when the sdsl-lite headers can be found (see below), or
  forced on with ``BEGUIDEDOFF_FAST=1``.
* ``beguidedoff._cuda`` -- the GPU kernel extension used by the **extensive**
  mode.  Built only when ``BEGUIDEDOFF_CUDA=1`` (or, for backwards
  compatibility, when ``nvcc`` is present and ``USE_CUDA != 0``); requires
  ``torch`` and must be built with ``--no-build-isolation`` after torch is
  installed.

Environment variables
---------------------
``BEGUIDEDOFF_FAST``   "0" to skip / "1" to require the FM-index binding.
``BEGUIDEDOFF_CUDA``   "0" to skip / "1" to require the CUDA extension.
``SDSL_INCLUDE_DIR``   Directory that contains ``sdsl/suffix_arrays.hpp``.
``SDSL_LIB_DIR``       Directory that contains ``libsdsl.a`` / ``libsdsl.so``.
``USE_CUDA``           Legacy switch: "0" disables the CUDA build.

Building the fast binding against a conda/pixi environment (recommended)::

    pixi run pip install --no-build-isolation .

or pass the paths explicitly::

    SDSL_INCLUDE_DIR=$CONDA_PREFIX/include SDSL_LIB_DIR=$CONDA_PREFIX/lib \\
        pip install --no-build-isolation .
"""
import os
import sys

from setuptools import Extension, find_packages, setup

ext_modules = []


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _find_sdsl_include():
    candidates = [
        os.environ.get("SDSL_INCLUDE_DIR"),
        os.path.join(sys.prefix, "include"),
        "/usr/local/include",
        "/usr/include",
    ]
    for cand in candidates:
        if not cand:
            continue
        if os.path.exists(os.path.join(cand, "sdsl", "suffix_arrays.hpp")):
            return cand
    return None


def _find_sdsl_lib():
    candidates = [
        os.environ.get("SDSL_LIB_DIR"),
        os.path.join(sys.prefix, "lib"),
        "/usr/local/lib",
        "/usr/lib",
    ]
    for cand in candidates:
        if not cand:
            continue
        for name in ("libsdsl.a", "libsdsl.so", "libsdsl.dylib"):
            if os.path.exists(os.path.join(cand, name)):
                return cand
    return None


def _pybind11_include():
    try:
        import pybind11

        return pybind11.get_include()
    except Exception:
        return os.path.join(sys.prefix, "include")


def _maybe_add_fast_extension():
    force = os.environ.get("BEGUIDEDOFF_FAST")
    include_dir = _find_sdsl_include()
    lib_dir = _find_sdsl_lib()
    available = include_dir is not None and lib_dir is not None

    if force is not None and not _truthy(force):
        return
    if not available:
        if force is not None and _truthy(force):
            raise SystemExit(
                "BEGUIDEDOFF_FAST=1 but sdsl-lite was not found; set "
                "SDSL_INCLUDE_DIR and SDSL_LIB_DIR."
            )
        print(
            "beguidedoff: sdsl-lite not found -- skipping the fast "
            "FM-index extension (set BEGUIDEDOFF_FAST=1 to require it)."
        )
        return

    ext_modules.append(
        Extension(
            name="beguidedoff._fmindex",
            sources=["csrc/fmindex_binding.cpp"],
            include_dirs=[include_dir, _pybind11_include()],
            library_dirs=[lib_dir],
            libraries=["sdsl", "divsufsort", "divsufsort64", "z"],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17"],
        )
    )


def _maybe_add_cuda_extension():
    force = os.environ.get("BEGUIDEDOFF_CUDA")
    if force is not None and not _truthy(force):
        return
    if force is None and os.environ.get("USE_CUDA", "1") == "0":
        return

    import shutil
    import subprocess

    if shutil.which("nvcc") is None:
        if force is not None and _truthy(force):
            raise SystemExit("BEGUIDEDOFF_CUDA=1 but nvcc was not found.")
        print("beguidedoff: nvcc not found -- skipping the CUDA extension.")
        return

    try:
        import torch.utils.cpp_extension as cpp_ext
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except Exception as exc:  # torch not installed / no CUDA support
        if force is not None and _truthy(force):
            raise
        print(f"beguidedoff: torch unavailable ({exc}); skipping the CUDA extension.")
        return

    # Allow compiling with system NVCC newer than the torch CUDA version
    # (minor mismatch is acceptable; PTX is backward compatible).
    original_check = cpp_ext._check_cuda_version

    def _patched_check_cuda_version(compiler_name, compiler_version):
        try:
            original_check(compiler_name, compiler_version)
        except RuntimeError as e:
            if "mismatches the version" in str(e):
                print(f"WARNING: ignoring CUDA version mismatch: {e}")
                return
            raise

    cpp_ext._check_cuda_version = _patched_check_cuda_version

    ext_modules.append(
        CUDAExtension(
            name="beguidedoff._cuda",
            sources=[
                "csrc/offtarget_kernel.cu",
                "csrc/scoring_kernel.cu",
                "csrc/bindings.cpp",
            ],
            include_dirs=["csrc/include"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
        )
    )
    return BuildExtension


_maybe_add_fast_extension()
_build_ext_cls = _maybe_add_cuda_extension()

setup(
    name="beguidedoff",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": _build_ext_cls} if _build_ext_cls else {},
    zip_safe=False,
)
