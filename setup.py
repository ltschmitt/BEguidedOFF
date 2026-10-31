"""setup.py for BEguidedOFF.

Two native extensions are built, both optional, so that a plain install never
touches torch/CUDA:

* ``beguidedoff._fmindex`` -- sdsl-lite FM-index binding used by the **fast**
  CPU mode.  Built when the sdsl-lite headers can be found (see below), or
  forced on with ``BEGUIDEDOFF_FAST=1``.
* ``beguidedoff._cuda`` -- optional GPU kernel extension.  **Opt-in only**:
  built when ``BEGUIDEDOFF_CUDA=1`` is set explicitly, which requires
  ``nvcc`` and ``torch`` and must be built with ``--no-build-isolation``
  after torch is installed.  Default installs never touch CUDA (nothing
  imports the extension yet; the extensive mode drives torch directly).

Environment variables
---------------------
``BEGUIDEDOFF_FAST``   "0" to skip / "1" to require the FM-index binding.
``BEGUIDEDOFF_CUDA``   "1" to build the CUDA extension (default: skipped).
``SDSL_INCLUDE_DIR``   Directory that contains ``sdsl/suffix_arrays.hpp``.
``SDSL_LIB_DIR``       Directory that contains ``libsdsl.a`` / ``libsdsl.so``.

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
    """Opt-in build of the CUDA kernel extension.

    The kernel is compiled only when ``BEGUIDEDOFF_CUDA=1`` is set
    explicitly; the build then fails loudly if nvcc or torch are missing
    instead of silently producing a kernel-less wheel.
    """
    if not _truthy(os.environ.get("BEGUIDEDOFF_CUDA", "")):
        return

    import shutil

    if shutil.which("nvcc") is None:
        raise SystemExit(
            "BEGUIDEDOFF_CUDA=1 but nvcc was not found; install the CUDA "
            "toolkit or drop BEGUIDEDOFF_CUDA to skip the GPU kernel."
        )

    try:
        import torch.utils.cpp_extension as cpp_ext
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except Exception as exc:
        raise SystemExit(
            f"BEGUIDEDOFF_CUDA=1 but torch is unavailable ({exc}); install "
            "torch with CUDA support or drop BEGUIDEDOFF_CUDA."
        ) from exc

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

    # Absolute paths: torch's ninja build runs nvcc from the build temp
    # directory, where relative source/include paths do not resolve.
    here = os.path.abspath(os.path.dirname(__file__))
    ext_modules.append(
        CUDAExtension(
            name="beguidedoff._cuda",
            sources=[
                os.path.join(here, "csrc", "offtarget_kernel.cu"),
                os.path.join(here, "csrc", "scoring_kernel.cu"),
                os.path.join(here, "csrc", "bindings.cpp"),
            ],
            include_dirs=[os.path.join(here, "csrc", "include")],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
        )
    )

    class _CudaOnlyBuildExtension(BuildExtension):
        """``BuildExtension`` that keeps torch's ABI flags off ``_fmindex``.

        torch's BuildExtension injects its own C++ ABI defaults (notably the
        pre-C++11 ``-D_GLIBCXX_USE_CXX11_ABI=0``) into every extension it
        builds. The FM-index binding links against sdsl-lite instead, whose
        conda-forge build uses the new C++11 ABI, so it must be compiled with
        plain setuptools flags; only the CUDA kernel goes through torch.
        """

        def build_extensions(self):
            # Plain setuptools first, before BuildExtension.build_extensions
            # monkey-patches the compiler with its ninja wrapper.
            plain_exts = [e for e in self.extensions if e.name != "beguidedoff._cuda"]
            cuda_exts = [e for e in self.extensions if e.name == "beguidedoff._cuda"]
            if plain_exts:
                self.extensions = plain_exts
                # Skip BuildExtension.build_extensions (torch flag injection)
                # and use the plain setuptools build_ext for the rest.
                super(BuildExtension, self).build_extensions()
            if cuda_exts:
                self.extensions = cuda_exts
                super().build_extensions()
            # Restore the full list: setuptools' build_ext.run() copies the
            # built libraries inplace by iterating self.extensions.
            self.extensions = plain_exts + cuda_exts

    return _CudaOnlyBuildExtension


_maybe_add_fast_extension()
_build_ext_cls = _maybe_add_cuda_extension()

setup(
    name="beguidedoff",
    version="0.2.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": _build_ext_cls} if _build_ext_cls else {},
    zip_safe=False,
)
