# BEguidedOFF

GPU-Accelerated Off-Target Base Editing Seeker for CRISPR-Cas9 base editors.

BEguidedOFF searches the human genome for off-target sites where base editing is possible, using GPU acceleration for 2-bit packed genome matching. It is designed for batch processing of 100+ guide RNAs simultaneously and produces off-target calls with CCTOP penalty scores, base-editing window filtering, and trinucleotide context annotation.

Two search modes share one API:

- **extensive** (GPU brute force via Numba CUDA) — exhaustive scan for every site with ≤ `max_mismatches`.
- **fast** (CPU FM-index) — a sound filter at a CCTOP score cutoff; same remove/keep verdicts without a GPU, for web/service use.

---

## What it does

Given a set of base editing guide RNAs, BEguidedOFF will:

1. **Encode the reference genome** once into a compact 2-bit `.npz` format (forward + reverse complement).
2. **Search** all guide RNAs in parallel against the entire genome on GPU via Numba CUDA.
3. **Score** each off-target with the CCTOP penalty model.
4. **Filter** hits by editable-base presence within the base editor's activity window (e.g. A-to-G editors require an **A** in the window).
5. **Report** chromosome, 1-based position, strand, mismatch positions, score, and trinucleotide context of editable bases.

---

## Environment

All commands below run inside the [`pixi`](https://pixi.sh) environment: `pixi install` once, then `pixi run <command>` (or `pixi shell`).

---

## Input File Format

BEguidedOFF accepts two TSV formats.

### Format 1: Profile-based (most common)

Use this when your data comes from a base-editing pipeline and you have `base_editor` values.

| Column | Description |
|--------|-------------|
| `chromosome` | Target chromosome (e.g. `1`, `X`, `7`) — used to exclude the on-target from off-target reports |
| `position` | 1-based genomic position of the target — used to exclude the on-target (see Position Conventions below) |
| `guide_rna_sequence` | 20 bp guide RNA sequence (DNA, ACGT) |
| `pam_sequence` | *On-target* PAM sequence (e.g. `TGG`) — **not** used for off-target scanning; scanning uses the PAM family from the profile |
| `base_editor` | Base editor profile name matching a row in `base_editor_profiles.tsv` |
| `strand_used` *(optional)* | `+` or `-` — on-target strand (required when `chromosome` and `position` are provided, used for correct on-target exclusion) |

Other metadata columns (gene, mutation IDs, etc.) are ignored and preserved in the output by row index. No unique identifier is required. See `examples/input_example.tsv` for a sample.

### Format 2: Explicit columns

Use this when you want full control over every parameter.

| Column | Description |
|--------|-------------|
| `guide_seq` | 20 bp guide sequence |
| `pam` | PAM **family** for off-target scanning (e.g. `NGG`, `NNNRRT`) |
| `edit_window_start` | 1-indexed start of editing window from 5' end of guide |
| `edit_window_end` | 1-indexed end of editing window (inclusive) |
| `editor_type` | `ABE`, `CBE`, or `CGE` |
| `chromosome` *(optional)* | On-target chromosome for 0-mm exclusion |
| `position` *(optional)* | 1-based on-target position for 0-mm exclusion |
| `on_target_strand` *(optional, required with chromosome/position)* | `+` or `-` |
| `base_editor` *(optional)* | Base editor profile name (carried through to output) |

**Note:** Only **3' PAM (`3prime`)** is supported. 5' PAM (`5prime`) is not supported.

---

## Base Editor Profiles (`base_editor_profiles.tsv`)

The profile file defines the editing window and PAM family for each base editor variant:

| Column | Description |
|--------|-------------|
| `name` | Profile name (e.g. `ABE8e`, `SaKKH-ABE8e`, `NG-ABE8e`, `SpRY-ABE8e`) |
| `editing` | Base conversion type (`ABE` for A→G editors, `CBE` for C→T editors) |
| `general_edit_start` | 1-indexed start of the high-activity window |
| `general_edit_end` | 1-indexed end of the high-activity window |
| `pam` | **PAM family** used for genome scanning (`NGG`, `NNGRRT`, `NNNRRT`, `NGN`, `NNN`) |
| `cas` | Cas protein family (`SpCas9`, `SaCas9`, `SpG / SpCas9-NG`, `SpRY`) |

**Important:** The `pam` column in `base_editor_profiles.tsv` is the IUPAC PAM **family** used for genome scanning (e.g. `NGG` matches any `N` then `G` then `G`). The `pam_sequence` column in your input data is the on-target concrete PAM and is **not** used for scanning.

---

## Data Files

Reference data lives in `nogit/` (git-ignored): encoded genomes, FM-indexes and
large inputs. The human reference used here is GRCh38 (Ensembl 115):

```
nogit/GRCh38.115/Homo_sapiens.GRCh38.dna.primary_assembly.fa   # source FASTA
nogit/hg38_beguidedoff.npz                                     # extensive mode (2-bit .npz)
nogit/hg38_beguidedoff.fmi                                     # fast mode (FM-index)
```

---

## One-Time Genome Encoding

### Extensive mode (2-bit `.npz`)

```bash
pixi run python -m beguidedoff.cli encode-genome \
    --fasta nogit/GRCh38.115/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --output nogit/hg38_beguidedoff.npz \
    --include-rc \
    --include-n-mask
```

This produces `hg38_beguidedoff.npz` containing the 2-bit packed forward strand and pre-computed reverse complement (~739 MB each), the chromosome index map and offsets, and an N-position mask.

### Fast mode (FM-index)

```bash
pixi run python -m beguidedoff.cli build-index \
    --fasta nogit/GRCh38.115/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --output nogit/hg38_beguidedoff.fmi
```

The FM-index stores only the forward strand (~2–3 GB); the reverse strand is
handled via reverse-complement anchors. It is memory-mapped at load time, so a
search process needs little RAM regardless of genome size.

---

## Searching for Off-Targets

### CLI — extensive mode (GPU)

```bash
pixi run python -m beguidedoff.cli search \
    --mode extensive \
    --genome nogit/hg38_beguidedoff.npz \
    --guides examples/input_example.tsv \
    --max-mismatches 4 \
    --n-top 10 \
    --output results.tsv \
    --return-all
```

### CLI — fast mode (CPU FM-index)

```bash
pixi run python -m beguidedoff.cli search \
    --mode fast \
    --index nogit/hg38_beguidedoff.fmi \
    --guides examples/input_example.tsv \
    --search-cutoff 5.0 \
    --n-top 10 \
    --output results.tsv
```

Options:
- `--mode extensive|fast` — GPU brute force (default) or CPU FM-index filter
- `--max-mismatches N` — extensive mode: up to N mismatches (default: 4, max: 5 tested)
- `--search-cutoff X` — fast mode: CCTOP score cutoff; every site with score ≤ X is enumerated (default: **5.0**)
- `--n-top N` — keep top N hits per guide (default: 3)
- `--return-all` — return all off-targets instead of only top-N
- `--count-score-threshold X` — CCTOP score threshold for counting relevant off-targets. Only editable off-targets with score ≤ X are tallied into the `n_offtargets_below_threshold` column (default: **5.0**). In fast mode a value above `--search-cutoff` widens the search region (slower); values above ~38 are not supported (the exact "cold" region becomes empty).
- `--include-non-editable` — report off-targets even if no editable base is in the window
- `--device cuda:0` — extensive mode: force CUDA; auto-detected otherwise

### Mode semantics

- **extensive** exhaustively scans every genome position on GPU; it reports every
  site with ≤ `max_mismatches`, regardless of score.
- **fast** is a sound *filter* at the score cutoff: it enumerates every site that
  can score ≤ `search_cutoff` (the PAM-proximal positions of any such site must
  match exactly). Guides with no hit at or below the cutoff are not in the
  output ("passes filter"). Remove/keep verdicts therefore agree with extensive
  mode at the same cutoff — without needing a GPU.

### Python API

```python
from beguidedoff import beguidedoff

# Extensive mode (GPU; device auto-detected: cuda:0 if available)
finder = beguidedoff(mode="extensive", genome_path="nogit/hg38_beguidedoff.npz", max_mismatches=5)

# Fast mode (CPU FM-index)
finder = beguidedoff(mode="fast", index_path="nogit/hg38_beguidedoff.fmi", search_cutoff=5.0)

# Search from TSV (identical call for both modes)
df = finder.search(input_file="guides.tsv", return_all=True)

# Or search from GuideInput objects directly
from beguidedoff.io import load_guides_from_tsv
guides = load_guides_from_tsv("guides.tsv", presets_path="base_editor_profiles.tsv")
df = finder.search(guides=guides, return_all=True)
```

### Advanced: Custom profile path

If `base_editor_profiles.tsv` is not in the app root directory, provide it explicitly via the Python API:

```python
guides = load_guides_from_tsv("my_input.tsv", presets_path="/path/to/base_editor_profiles.tsv")
```

---

## Output Format

The default output is a long-format TSV with one row per off-target hit.

| Column | Description |
|--------|-------------|
| `guide_rna_sequence` | Original on-target guide sequence |
| `base_editor` | Base editor profile name |
| `ot_rank` | Rank by CCTOP score (ascending; lower = more concerning) |
| `ot_chromosome` | Off-target chromosome |
| `ot_position` | **1-based** genomic position of the guide start (PAM-adjacent) |
| `ot_strand` | `+` or `-` |
| `ot_sequence` | 20 bp off-target sequence |
| `ot_pam` | Full genomic PAM sequence |
| `n_mismatches` | Number of mismatches vs guide |
| `mismatch_positions` | 1-indexed mismatch positions (comma-separated) |
| `score` | CCTOP penalty score (higher = less concerning) |
| `n_offtargets_below_threshold` | Number of **editable** off-targets for this guide with a CCTOP score ≤ `count_score_threshold`. This count includes *all* qualifying off-targets in the genome, regardless of `--n-top`. `0` when no threshold is set. |

---

## Architecture

`beguidedoff/` is a Python package. A lazy mode factory (`__init__.py`, no torch/numba on
import) dispatches to the **extensive** GPU brute-force finder (`finder.py` + `cuda_search.py`,
Numba CUDA over a 2-bit packed genome, 200 Mbp chunks, ≤2,048 unique scan keys per launch) or
the **fast** CPU FM-index finder (`fast/`, backed by the sdsl-lite pybind11 binding in
`csrc/fmindex_binding.cpp`, compiled to `beguidedoff._fmindex`; the separate C++ CUDA extension
is only built with `BEGUIDEDOFF_CUDA=1`). Shared modules cover 2-bit encoding, IUPAC PAM
handling, CCTOP scoring, editable-base window filtering, score-cutoff search regions, TSV I/O,
and FASTA→.npz conversion.

A guide whose raw hits overflow the per-guide buffer (`results_per_guide=10240`) is
**re-scanned automatically** with the buffer raised ×16 (up to 2,621,440) and its hits
replaced — the returned hit list is exact. A guide that still overflows at that ceiling fails
with a clear `RuntimeError` instead of returning silent truncations; use the fast CPU mode for
such guides.

---

## Validation

BEguidedOFF was validated against [Cas-OFFinder](https://github.com/snugel/cas-offinder) v2.4.1 on 394 patient-mutation guides across three PAM families (≤ 5 mismatches, `return_all`, all hits incl. non-editable). Re-run 2026-10-25 after the overflow re-scan fix, with inputs and methodology identical to the original comparison (scripts `benchmarks/casoffinder_rerun.py` + `benchmarks/compare_rerun.py`; full report in `benchmarks/REPORT.md` and `benchmarks/casoffinder_rerun_report.json`).

| PAM Family | Guides | Total Hits | Overlap vs Cas-OFFinder |
|-----------|--------|-----------|------------------------|
| NGG | 223 | 414,438 | **99.9998%** (1 difference: N in reference) |
| NNGRRT | 49 | 20,218 | **100%** |
| NNNRRT | 122 | 811,624 | **99.9999%** (4 differing calls, all at N-containing loci) |

Every difference involves a locus where the reference genome carries an N: BEguidedOFF's 2-bit encoding maps N→A, so an N-containing site can match where Cas-OFFinder counts a mismatch. A single guide (`ATCGAGACCATCCTGGCTAA`) is a 7 bp tandem repeat producing 523,773 off-targets; the GPU mode re-scans such guides automatically with an escalating hit buffer (10,240 → 163,840 → 2,621,440) so they are enumerated completely, and beyond the ceiling it raises a clear error — `--results-per-guide` (or `results_per_guide` on the `beguidedoff` factory) only changes the *initial* buffer size.

### Speed comparison

Whole-process wall time on an NVIDIA GeForce RTX 3060 (2026-10-25 re-run; Cas-OFFinder v2.4.1 OpenCL vs BEguidedOFF extensive, CUDA; one run per PAM family):

| Tool | NGG (223 guides) | NNGRRT (49) | NNNRRT (122) | **Total** |
|------|------------------|-------------|---------------|-----------|
| Cas-OFFinder GPU | 140.7 s | 20.6 s | 67.6 s | **228.9 s** |
| **BEguidedOFF GPU** | 59.9 s | 16.7 s | 78.7 s | **155.3 s** |

BEguidedOFF is ~1.5× faster end-to-end in this configuration, while producing CCTOP-scored, editable-base-filtered, trinucleotide-annotated TSV output for every hit, whereas Cas-OFFinder dumps raw alignments only. (The NNNRRT batch is the one slow spot: it contains the 7 bp tandem-repeat guide whose 523,773 hits cost two extra re-scan passes plus per-hit assembly.) The 2026-10-25 kernel rework made the CUDA scan 7.6–9.2× faster with byte-identical outputs; the kernel is now 33–78% of search time, with Python-side per-hit assembly co-dominant on hit-heavy batches. The interactive path is the **fast** (CPU FM-index) mode, which is ~12–31× faster than the exhaustive GPU scan for the removal decision. Full methodology, per-stage profiling, and benchmark history: [`benchmarks/REPORT.md`](benchmarks/REPORT.md).

Search scales with the number of *unique* `(guide_seq, PAM)` pairs because identical sequences are deduplicated internally.

---

## Running Tests

```bash
PYTHONPATH=$PWD pixi run python -m pytest tests/ -v
```

68 tests cover encoding round-trips, PAM validation (including exhaustive
IUPAC-encoding checks and a GPU test that plants S/W/V-code PAM sites and
compares the CUDA extensive search against a brute-force reference),
base-editing window logic, search-region derivation, genome conversion,
fast-vs-brute-force equivalence, import hygiene (no torch/numba on import),
and end-to-end integration for both modes.

---

## CCTOP Scoring

For every mismatch at position *p* (1-indexed from the 5′ end of the 20 bp guide), the penalty is `1.2**p`; the overall score is the **sum** of these penalties. PAM mismatches are **ignored** — only the 20 bp guide alignment contributes to the score.

- **Lower scores** = more likely off-target activity (mismatches in the seed region, near the PAM)
- **Higher scores** = less concerning (mismatches far from the PAM); a perfect match scores **0.0**

The model treats mismatches uniformly, weighted only by distance from the seed, so it is **not biased toward any single Cas family** — it works equally well for NGG, NNGRRT, NNNRRT, NGN, or any other PAM family.

---

## Important Notes

- **PAM support:** Only **3' PAM (`3prime`)** is supported. 5' PAM (`5prime`) is not supported.
- **Position conventions:** All input and output positions are **1-based**. The position is the **5' end of the guide RNA sequence** on the reported strand:
  - For `+` strand off-targets: the **leftmost** position (guide starts here, PAM follows downstream).
  - For `-` strand off-targets: the **rightmost** position (guide ends here, PAM precedes it upstream).
  - For on-target exclusion, provide the position of the **5' end** accordingly (`+` = leftmost, `-` = rightmost).
- **0-mismatch exclusion:** Only the *on-target* location (from `chromosome` / `position` / `strand` in the input) is excluded. Other exact-match off-targets elsewhere in the genome **are** reported.
- **Mixed PAMs:** Guides with different PAM families are grouped and scanned separately. Identical `(seq, PAM)` pairs are scanned once and results mapped back to all rows.
- **CPU fallback:** There is no CPU brute-force fallback. If CUDA is unavailable, `extensive` mode fails with a clear error; use the `fast` (CPU FM-index) mode instead.

---

## References

- Richter et al. (2020) — *Phage-assisted evolution of an adenine base editor with improved Cas domain compatibility and activity*

---

## License

MIT
