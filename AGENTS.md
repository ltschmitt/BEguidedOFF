# BEguidedOFF — Agent Notes

## Environment
- **Package manager:** Pixi (`pixi.toml` at root). Channels: `conda-forge`, `bioconda`. Platform: `linux-64` only.
- **Run commands in env:** `pixi run <cmd>`
- **Add deps:** `pixi add <package>` (not `pip install`)
- **Installed:** pytorch >=2.10, numpy >=2.4.6, pandas >=3.0.3, pysam >=0.24.0, pytest >=9.0

## Architecture
- One Python package `beguidedoff` with placeholder CUDA kernels in `csrc/`.
- Two main entry points:
  1. `genome_converter.convert()` — one-time FASTA → 2-bit `.npz` + reverse complement.
  2. `beguidedoff.search()` — off-target search (CPU fallback), batch 100+ guides, outputs long-format thresholded list.
- CLI: `python -m beguidedoff.cli {encode-genome,search,verify}`
- API signatures and data layouts are in `relevant_files/chat.md` sections 4, 7, 10.

## Key Data Files (Do Not Guess Formats)
- `base_editor_profiles.tsv` — base editor lookup. **Use `general_edit_start`/`general_edit_end`** as the search window for off-targets. Columns include `pam`, `cas`, `editing`.
- `relevant_files/input_example.tsv` — example input. Relevant columns: `guide_rna_sequence`, `base_editor` (and optional `pam_sequence`). Other columns are metadata.
- `relevant_files/cfd.py` — **existing, working CFD scorer** (Doench 2016). Guide bases must be converted T→U; target bases use reverse complement for lookup. The scoring module ports this logic directly.
- `relevant_files/Homo_sapiens.GRCh38.dna.primary_assembly.fa` — human genome FASTA for converter.

## Conventions & Quirks
- CFD scoring: guide uses RNA bases (U instead of T). Mismatch keys are `r{guide}:d{rc_target},{position}`. PAM scores are in `PAM_SCORES` dict (2-nt key, e.g. `"GG"`).
- Editing window is **1-indexed from the 5' end of the 20 bp guide** (not PAM-relative).
- Mixed PAM batches are expected; the design groups guides by `(pam, pam_position)` for separate kernel launches.
- There is no CI and no formatter/typechecker config yet.
- pytest is available. Run tests with `PYTHONPATH=/data/lschmitt/apps/BEguidedOFF pixi run python -m pytest tests/ -v`.
- pip is blocked by PEP 668 in this environment — do not use `pip install`. Use `pixi add` instead, or run with `PYTHONPATH=...` for local development.

## What Exists vs. What Is Planned
| Exists | Planned / Stubbed |
|--------|---------------------|
| `beguidedoff` package, types, encoding, PAM, scoring, base_editing, io | CUDA kernels (`csrc/*.cu`) — placeholder stubs only |
| `genome_converter.py` — full implementation | JIT CUDA extension build in `setup.py` — disabled because torch is CPU-only |
| `finder.py` — CPU search with PyTorch tensor loading | GPU kernel launch in `_search_gpu()` — falls back to CPU loop |
| `cfd.py` (reference CPU scorer) | `pyproject.toml`, `setup.py` |
| `cli.py` — argparse CLI | `tests/` — 26 tests passing |
