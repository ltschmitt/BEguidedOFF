# BEguidedOFF

GPU-Accelerated Off-Target Base Editing Seeker for CRISPR-Cas9 base editors.

BEguidedOFF searches the human genome for off-target sites where base editing is possible, using GPU acceleration for 2-bit packed genome matching. It is designed for batch processing of 100+ guide RNAs simultaneously and produces off-target calls with CCTOP penalty scores (or optionally Doench 2016 CFD scores), base-editing window filtering, and trinucleotide context annotation.

Two search modes share one API:

- **extensive** (GPU brute force via Numba CUDA) — exhaustive scan for every site with ≤ `max_mismatches`.
- **fast** (CPU FM-index) — a sound filter at a CCTOP score cutoff; same remove/keep verdicts without a GPU, for web/service use.

---

## What it does

Given a set of base editing guide RNAs, BEguidedOFF will:

1. **Encode the reference genome** once into a compact 2-bit `.npz` format (forward + reverse complement).
2. **Search** all guide RNAs in parallel against the entire genome on GPU via Numba CUDA.
3. **Score** each off-target with the CCTOP penalty model (default) or the Doench 2016 CFD model (optional).
4. **Filter** hits by editable-base presence within the base editor's activity window (e.g. A-to-G editors require an **A** in the window).
5. **Report** chromosome, 1-based position, strand, mismatch positions, score, and trinucleotide context of editable bases.

---

## Environment

This project is managed with [`pixi`](https://pixi.sh).  All commands below should be run inside the pixi environment.

```bash
pixi install          # first-time setup
pixi run <command>    # run any command inside the env
pixi shell            # open a shell inside the env
```

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

Other metadata columns (gene, mutation IDs, etc.) are ignored and preserved in the output by row index. No unique identifier is required.

**Example row:**

```
chromosome  position  mutated_base  corrected_base  target_base  strand_used  editing_position  bystander_positions  guide_rna_sequence      pam_sequence  base_editor  gene
1           18700668  T             C               C            -            4                 8                    GGCAGTTACTGAACCAGACC  TGG           ABE-NW1    PAX7
```

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

This produces `hg38_beguidedoff.npz` containing:
- 2-bit packed forward strand (`~739 MB`)
- Pre-computed reverse complement (`~739 MB`)
- Chromosome index map and offsets
- N-position mask

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
    --guides nogit/relevant_files/input_example.tsv \
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
    --guides nogit/relevant_files/input_example.tsv \
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

```
beguidedoff/
├── __init__.py          # lazy mode factory (extensive/fast; no torch on import)
├── cli.py               # argparse CLI (encode-genome, build-index, search, verify)
├── types.py             # GuideInput, OffTarget, GuideResult
├── encoding.py          # 2-bit DNA encoding, 40-bit guide packing
├── pam.py               # IUPAC PAM masks, guide grouping
├── scoring.py           # CCTOP score, Doench 2016 CFD scorer (T→U, RC lookup)
├── base_editing.py      # Editable-base window filtering, trinucleotide contexts
├── regions.py           # score cutoff → cold/hot search regions (fast mode)
├── io.py                # TSV loading (preset-based or explicit), output formatting
├── genome_converter.py  # FASTA → .npz (forward + RC, chromosome index)
├── finder.py            # extensive mode: beguidedoff (CUDA only)
├── cuda_search.py       # Numba CUDA kernel (scans 2-bit genome in chunks)
└── fast/
    ├── backend.py       # symbol codec (2-bit-compatible) + backend protocol
    ├── fmindex.py       # sdsl-lite FM-index build/load wrapper
    └── finder.py        # fast mode: beguidedofffast (CPU, streaming search)
```

The sdsl-lite pybind11 binding for the fast mode lives in `csrc/fmindex_binding.cpp`
and is compiled to `beguidedoff._fmindex` (built by default; the CUDA extension is
built only with `BEGUIDEDOFF_CUDA=1` or when nvcc + torch are present).

### GPU kernel design

- **Genome packing:** 2 bits per base, 16 bases per `uint32`
- **Guide packing:** 40-bit integer for 20-base sequence
- **Mismatch detection:** XOR + popcount on the 40-bit value
- **PAM matching:** IUPAC 4-bit mask per position, checked before guide comparison
- **Chunking:** Genome scanned in 200 Mbp chunks to manage GPU memory
- **Buffer limit:** `results_per_guide=10240` hits stored per guide per chunk; overflow triggers a `UserWarning`

---

## Validation

BEguidedOFF was validated against [Cas-OFFinder](https://github.com/snugel/cas-offinder) v2.4.1 on 361 patient mutation guides across three PAM families.

| PAM Family | Guides | Total Hits | Overlap vs Cas-OFFinder |
|-----------|--------|-----------|------------------------|
| NGG | 233 | ~434K | **99.9998%** (1 difference: N in genome) |
| NNGRRT | 7 | ~4.7K | **100%** |
| NNNRRT | 121 | ~814K | **100% per guide** (1 repetitive guide overflows buffer) |

A single guide (`ATCGAGACCATCCTGGCTAA`) is a 7 bp tandem repeat producing >500K off-targets; the GPU buffer of 10,240 hits truncates it. Increase `--results-per-guide` (or pass `results_per_guide` to `beguidedoff`) if needed for extremely repetitive guides.

### Speed comparison

| Tool | NGG (~225 guides) | NNGRRT (~50) | NNNRRT (~124) | **Total** |
|------|------------------|-------------|---------------|-----------|
| Cas-OFFinder GPU | ~337 s | ~39 s | ~247 s | **~623 s** |
| **BEguidedOFF GPU** | **~255 s** | **~7 s** | **~187 s** | **~448 s** |

**Speedup: ~1.4×** on an NVIDIA GeForce RTX 3060.

Search scales with the number of *unique* `(guide_seq, PAM)` pairs because identical sequences are deduplicated internally.

---

## Running Tests

```bash
PYTHONPATH=/data/lschmitt/apps/BEguidedOFF pixi run python -m pytest tests/ -v
```

65 tests cover encoding round-trips, PAM validation, CFD scoring, base-editing
window logic, search-region derivation, genome conversion, fast-vs-brute-force
equivalence, import hygiene (no torch/numba on import), and end-to-end
integration for both modes. A captured test log is kept in
`nogit/test_results.txt`.

---

## CCTOP Scoring (Default)

BEguidedOFF now uses the **CCTOP off-target penalty score** as its default scoring method.

### How it works

For every mismatch at position *p* (1-indexed from the 5′ end of the 20 bp guide), the penalty is `1.2**p`. The overall score is the **sum** of these penalties:

```
cctop_score = Σ 1.2^p  (for each mismatch at position p)
```

PAM mismatches are **ignored** — only the 20 bp guide alignment contributes to the score.

### Interpretation

- **Lower scores** = more likely off-target activity (mismatches in the seed region, near the PAM)
- **Higher scores** = less likely off-target activity (mismatches far from the PAM)
- A perfect match yields a score of **0.0**

This model is intentionally simple and **not biased toward any single Cas family** (e.g. SpCas9 NGG). It works equally well for NGG, NNGRRT, NNNRRT, NGN, or any future PAM family.

### Why CCTOP over CFD?

The Doench 2016 CFD penalty matrix is specific to SpCas9 (NGG). Its position-dependent mismatch penalties and especially its PAM score table are not valid for other PAM families such as `NNGRRT` (saCas9) or `NGN` (ngCas9), which can produce misleading off-target rankings when guides with mixed PAM families are searched together.

The CCTOP formula (from the CCTOP tool) treats mismatches uniformly weighted by distance from the seed, making it unbiased across Cas families.

### Using the CFD scorer (not recommended for mixed PAM families)

CFD scoring can be re-enabled for backward compatibility or for purely SpCas9 NGG datasets via `--scoring cfd` (CLI) or `scoring_method="CFD"` (Python API). The PAM multiplier remains **deactivated** in CFD mode for the same cross-family bias reasons described above.

---

## Important Notes

- **PAM support:** Only **3' PAM (`3prime`)** is supported. 5' PAM (`5prime`) is not supported.
- **Scoring:** CCTOP is now the default. CFD (Doench 2016) is still available via `--scoring cfd` but is not recommended for mixed-PAM datasets.
- **Score direction:** With CCTOP, lower scores are more concerning. Results are sorted ascending. With CFD, higher scores are more concerning and results are sorted descending.
- **`--score-threshold`:** For CCTOP, the threshold acts as a **maximum** penalty (only off-targets with score ≤ threshold are kept). For CFD it is a **minimum** score as before.
- **Position conventions:** All input and output positions are **1-based**. The position is the **5' end of the guide RNA sequence** on the reported strand:
  - For `+` strand off-targets: the **leftmost** position (guide starts here, PAM follows downstream).
  - For `-` strand off-targets: the **rightmost** position (guide ends here, PAM precedes it upstream).
  - For on-target exclusion, provide the position of the **5' end** accordingly (`+` = leftmost, `-` = rightmost).
- **0-mismatch exclusion:** Only the *on-target* location (from `chromosome` / `position` / `strand` in the input) is excluded. Other exact-match off-targets elsewhere in the genome **are** reported.
- **Mixed PAMs:** Guides with different PAM families are grouped and scanned separately. Identical `(seq, PAM)` pairs are scanned once and results mapped back to all rows.
- **CPU fallback:** If CUDA is unavailable, the search falls back to a pure-CPU loop (correct but orders of magnitude slower).

---

## References

- Doench et al. (2016) — *Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9*
- Richter et al. (2020) — *Phage-assisted evolution of an adenine base editor with improved Cas domain compatibility and activity*

---

## License

MIT
