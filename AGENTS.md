# BEguidedOFF — Agent Notes

## Environment
- **Package manager:** Pixi (`pixi.toml` at root). Channels: `pytorch`, `conda-forge`, `bioconda`. Platform: `linux-64` only.
- **Run commands in env:** `pixi run <cmd>`
- **Add deps:** `pixi add <package>` (not `pip install`)
- **Installed:** base: numpy, pandas, pysam, pytest. GPU extra: pytorch + numba (Numba CUDA). Fast mode: sdsl-lite FM-index binding (`beguidedoff._fmindex`, built from `csrc/fmindex_binding.cpp`).

## Architecture
- One Python package `beguidedoff`. The GPU search is Numba CUDA (`beguidedoff/cuda_search.py`); the `csrc/*.cu` files are placeholder stubs for a future C++ CUDA extension, while `csrc/fmindex_binding.cpp` is the live sdsl-lite pybind11 binding.
- Two main entry points:
  1. `genome_converter.convert()` — one-time FASTA → 2-bit `.npz` (forward + reverse complement).
  2. `beguidedoff.beguidedoff(mode=...)` — factory returning the **extensive** (GPU brute force) or **fast** (CPU FM-index) finder; `finder.search()` runs the batch off-target search and outputs a long-format thresholded list.
- CLI: `python -m beguidedoff.cli {encode-genome,build-index,search,verify}`
- API signatures and data layouts are in `nogit/relevant_files/chat.md` sections 4, 7, 10.

## Key Data Files (Do Not Guess Formats)
- `base_editor_profiles.tsv` — base editor lookup. **Use `general_edit_start`/`general_edit_end`** as the search window for off-targets. Columns include `pam`, `cas`, `editing`.
- `nogit/relevant_files/input_example.tsv` — example input. Relevant columns: `guide_rna_sequence`, `base_editor` (and optional `pam_sequence`). Other columns are metadata.
- `nogit/relevant_files/cfd.py` — legacy Doench 2016 CFD scorer; **no longer used by the package** (scoring is CCTOP-only).
- `nogit/relevant_files/Homo_sapiens.GRCh38.dna.primary_assembly.fa` — human genome FASTA for converter.
- `nogit/` is gitignored and not part of the repository; these files exist only in the local working copy. Heavy benchmark data lives in `nogit/bench/`; the tracked benchmark scripts and reports are in `benchmarks/`.

## Conventions & Quirks
- Scoring is **CCTOP-only**: score = Σ 1.2^p over 1-indexed mismatch positions; **lower = more concerning**; PAM mismatches are ignored.
- Editing window is **1-indexed from the 5' end of the 20 bp guide** (not PAM-relative).
- Mixed PAM batches are expected; the design groups guides by `(pam, pam_position)` for separate kernel launches.
- There is no CI and no formatter/typechecker config yet.
- pytest is available. From the repo root, run tests with `PYTHONPATH=$PWD pixi run python -m pytest tests/ -v`.
- pip is blocked by PEP 668 in this environment — do not use `pip install`. Use `pixi add` instead, or run with `PYTHONPATH=...` for local development.

## What Exists vs. What Is Planned
| Exists | Planned / Stubbed |
|--------|---------------------|
| `beguidedoff` package: types, encoding, PAM, CCTOP scoring, base_editing, regions, io | CUDA kernels (`csrc/*.cu`) — placeholder stubs only |
| `genome_converter.py` — full implementation | C++ CUDA extension build in `setup.py` — opt-in via `BEGUIDEDOFF_CUDA=1`, off by default |
| `finder.py` — extensive GPU search (Numba CUDA) | — |
| `fast/` — CPU FM-index mode (sdsl-lite binding) | — |
| `cli.py` — argparse CLI | — |
| `benchmarks/` — validation/benchmark scripts + reports | — |
| `tests/` — 68 tests passing | — |
