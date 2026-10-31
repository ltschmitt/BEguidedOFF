# BEguidedOFF fast vs extensive — benchmark report

Generated 2026-10-20 13:49 · branch `feature/fastoff` · human genome GRCh38 (Ensembl 115; byte-identical to 112, md5 `a65212262de00761b43869d5c08c8e4d`)

## Environment

- CPU: Intel(R) Xeon(R) Gold 6146 CPU @ 3.20GHz, x86_64
- RAM: 377 GB
- GPU: NVIDIA GeForce RTX 3060
- Python 3.12.13, numpy 2.4.6, pandas 3.0.3

## Phase 0 — correctness gate (fast vs extensive, real genome)

**PASS** on the 100-guide mixed set (100 unique scans).

| check | mismatches |
|---|---|
| remove/keep verdicts | 0 |
| full hit set with score ≤ 5 | 0 |
| n_offtargets_below_threshold (T=3) | 0 |
| top-3 lists | 0 |

Extensive enumerated 24,203 hits (≤ 4 mismatches, incl. non-editable) in 115.80 s; 78 of them score ≤ 5 — fast reproduced that subset exactly (1.26 s, return_all).

The comparison covers all sites that can matter for the removal decision: any site with CCTOP score ≤ 5 has ≤ 4 mismatches, which extensive enumerates exhaustively; fast enumerates every site with score ≤ 5 by construction (exact PAM-proximal cold region). Ties within the top-3 are broken canonically by `(score, chromosome, position, strand)` in both modes so top-N is deterministic and mode-independent.

### Stress test: the 24 buffer-overflow guides (912 guide×editor rows)

**Before the fix** (buffer truncation): verdicts agreed (0 mismatches), but extensive under-reported everywhere else —

| check | mismatches |
|---|---|
| hit set with score ≤ 5 | 408 of 912 |
| n_offtargets_below_threshold (T=3) | 296 |
| top-3 lists | 281 |

Extensive enumerated 65,752 sites with score ≤ 5 where fast enumerated 205,684 — every fast-only site was verified genuine, i.e. **the truncation was in extensive, not in fast**.

**After the fix** (automatic re-scan): [PASS] — extensive now re-scans any guide whose ≤4-mismatch hits overflow the buffer with an escalating `results_per_guide` (10 240 → 163 840 → 2 621 440, replacing the truncated hits). Its score ≤ 5 subset now matches fast exactly:

| check | mismatches |
|---|---|
| verdicts | 0 |
| hit set with score ≤ 5 | 0 of 912 |
| n_offtargets_below_threshold (T=3) | 0 |
| top-3 lists | 0 |

Both engines enumerate 205,684 score ≤ 5 sites. Several guides needed the second escalation tier (>163 840 raw ≤4-mm hits each). The cost is runtime only — the complete enumeration is CPU-bound scoring work the truncated search used to skip. A guide that still overflows at the ceiling fails with a clear error instead of returning silent truncations.

## Phase 1 — search latency (top-3 output, count threshold 3)

| guide set | input rows | unique scans | extensive (GPU) | fast (CPU, 24c) | fast (CPU, 4c) | speedup (4c) |
|---|---:|---:|---:|---:|---:|---:|
| 10 | 10 | 10 | 11.21 | 0.23 | — |  |
| 50 | 50 | 50 | 58.37 | 0.63 | — |  |
| 100 | 100 | 100 | 115.91 | 1.22 | — |  |
| 500 | 500 | 500 | 560.50 | 8.20 | 8.59 | 65× |
| 500x38 | 19000 | 2500 | 3,255 † | 42.69 | 44.23 | 74× |

extensive = median of timed reps after warmup (CUDA JIT compiled once per process); fast = median of 3 reps. 4c = `taskset -c 0-3` emulating the 4-core web VM.

† extensive cannot run the 500x38 set in one call: the CUDA kernel caps guides per launch (`cuda_search.MAX_GUIDES = 2048`, shared-memory budget), so it was split into 2 chunks of ≤ 2,000 unique scans and the per-call times summed (1 rep). The production pipeline would have to batch the same way; fast mode has no such limit.

Resource peaks during search:

- extensive 500: peak process RSS 3.2 GB, peak VRAM 1.45 GB
- extensive 500x38: peak RSS 12.9 GB, peak VRAM 1.45 GB
- fast 500: peak process RSS 2.34 GB (includes mmap'd index)
- fast 500x38: peak process RSS 2.61 GB

Hit-buffer overflows in the extensive runs: 46 warnings (24 unique guides exceed the 10,240-hit buffer at ≤ 4 mismatches — see the stress test above; fast mode reported 0 warnings anywhere). Guides whose exact 12-mer cold anchor occurs > 2,000 times were excluded from the sets up front (6 of 346 real protospacers) because they exceed both engines' caps — documented limitation.

Note: the extensive timings above were measured **before** the overflow fix. A post-fix re-run of the same extensive-500 benchmark (`perf_extensive_500_postfix.json`, 1 timed rep) measured **658.5 s** vs 560.5 s pre-fix — an overhead of **+98.0 s** (1.17×) for exact, untruncated hit lists on the guides that overflow the 10,240-hit buffer.

Post-fix re-scan activity in that run: 2 re-scan round(s) logged; example overflowing guides: AGAGTCTCACTCTGCTGCCC, CACTGCAACCTCCACCCTCC, TAAAAATACAAAACTTAGCT, TGTAATTCCGGTACTTTGGG, TTGTTGTTTTTGTAGAGGCA. The re-scan (buffer ×16 up to 2 621 440, fail-closed RuntimeError at the ceiling) replaces the old silent truncation; guides that fit the buffer are unaffected.

## Phase 2 — index economics

| artifact | size | build time | build peak RSS | load (warm) | load (cold) |
|---|---:|---:|---:|---:|---:|
| FM-index (`hg38_beguidedoff.fmi`) | 1.72 GB | 678.50 | 26.1 GB | 0.77 s | 1.01 s |
| 2-bit npz (`hg38_beguidedoff.npz`) | 1.28 GB | — | — | 7.64 s | 7.88 s |

Cold = page cache dropped with `posix_fadvise(DONTNEED)` before the load. The FM-index is mmapped, so a fast-mode process pays the cold read once and the OS keeps it hot across requests on a long-running service.

## Notes

- Guide sets: first N real protospacers from `input_example.tsv` (unique, editor-name-normalized), padded to N with genomic NGG protospacers sampled from the FM-index (seed 42, anchor occurrence ≤ 2000 on both strands). The 500x38 set expands the 500-set over all 38 base-editor profiles (19,000 rows).
- `input_example.tsv` spells `SaABE8e` as the older `Sa-ABE8e`; the benchmark normalizes editor names punctuation-insensitively. The production input/profile data should be brought into sync (7 rows affected).
- `io.py` reads the on-target strand from the `strand_used` column, while `input_example.tsv` uses `strand`; without it, on-target exclusion cannot work. Flagged for the BEguided integration.
- Extensive runs `max_mismatches=4`, which fully covers fast's score ≤ 5 enumeration (5 mismatches ⇒ min score 8.93 > 5).
- The index was rebuilt from the 115 FASTA; the 112/115 primary assemblies are byte-identical, and the re-encoded npz is bit-identical to the previous 112-derived one (verified field-by-field), so all three artifacts describe the same genome.

---

## Cas-OFFinder v2.4.1 re-run (2026-10-25) — post overflow-fix extensive

Re-run of the original Cas-OFFinder comparison with identical inputs and
methodology (whole-process wall time, ≤ 5 mismatches, all hits incl.
non-editable; one run per PAM family). Scripts: `casoffinder_rerun.py`
(setup + BEguidedOFF runs), `compare_rerun.py` (concordance + summary JSON
`casoffinder_rerun_report.json`). The `casoffinput_*_rerun.txt` files are the
originals with the FASTA path repointed to
`nogit/GRCh38.115/Homo_sapiens.GRCh38.dna.primary_assembly.fa` (the old
OTBEseeker path no longer exists; the 112/115 primary assemblies are
byte-identical, and each rerun Cas-OFFinder output is content-identical to its
May-2026 original, confirming the same genome).

### Timing (RTX 3060, whole process incl. genome load)

| family | guides | Cas-OFFinder wall | BEguidedOFF extensive wall | Cas hits | BEguidedOFF hits |
|---|---:|---:|---:|---:|---:|
| NGG | 223 | 140.7 s | 259.4 s | 414,437 | 414,438 |
| NNGRRT | 49 | 20.6 s | 56.9 s | 20,218 | 20,218 |
| NNNRRT | 122 | 67.6 s | 231.6 s | 811,622 | 811,624 |
| **total** | **394** | **228.9 s** | **548 s** | 1,246,277 | 1,246,280 |

(hit counts from raw output lines; the concordance totals below use sets keyed
by (guide, chrom, window_leftmost_0based, strand, mm), which collapses
duplicate output lines.)

**Cas-OFFinder is ~2.4× faster than BEguidedOFF extensive on these batches.**
The May-2026 README table (448 s vs 623 s, "~1.4× faster") is not reproducible
— the same binary on the same inputs now measures 228.9 s, so the old
comparison was methodologically off; today's numbers are the consistent
pair. Context: BEguidedOFF does strictly more work per hit (CCTOP scoring,
trinucleotide contexts, editable-base filtering, richer TSV output) and its
NNNRRT time includes the full re-scan of the 7-bp tandem-repeat guide
(buffer escalations 10,240 → 163,840 → 2,621,440 logged).

### Concordance (position conventions normalised; see compare_rerun.py)

Cas-OFFinder reports 0-based leftmost of the guide+PAM window; BEguidedOFF
reports the 1-based 5' end of the guide on the reported strand (leftmost+1 on
+ strand, leftmost+window_len on - strand) — both verified empirically before
keying.

| family | Cas hits (sets) | BEguidedOFF hits | overlap | cas_only | BEguidedOFF_only | concordance |
|---|---:|---:|---:|---:|---:|---:|
| NGG | 414,437 | 414,438 | 414,437 | 0 | 1 | 99.9998% |
| NNGRRT | 20,218 | 20,218 | 20,218 | 0 | 0 | 100% |
| NNNRRT | 809,066 | 809,068 | 809,065 | 1 | 3 | 99.9999% |
| **total** | **1,243,721** | **1,243,724** | **1,243,720** | **1** | **4** | **99.9999%** |

All five differing calls involve loci where the reference genome carries an N
(verified against the FASTA via pysam): BEguidedOFF's 2-bit encoding maps N→A,
so an N-containing site can match where Cas-OFFinder counts a mismatch (the
chr20 tandem-repeat locus appears on both sides with mm 4 vs 5 for exactly
this reason). The previously truncated tandem-repeat guide
(`ATCGAGACCATCCTGGCTAA`) is now enumerated completely — the "one repetitive
guide exceeded the buffer" caveat from the May-2026 validation is resolved.

---

## Stage profiling of the extensive search (2026-10-25)

Question: where does the wall time of the extensive (GPU) search go, and how
much of the Cas-OFFinder gap is the overflow re-scan vs the kernel vs Python
per-hit work? `profile_stages.py` wraps timers around the internal boundaries
of `beguidedoff.finder` (monkey-patching only, no production changes) and
sync-times every `search_chunk_kernel` launch. Runs: the three rerun batches
in the benchmark configuration (`return_all` + `include_non_editable`) and in
the web-pipeline configuration (`return_all=False`, top-3 per guide).
Machine-readable: `profile_stages_report.json`.

### Stage breakdown, benchmark configuration

| stage (s) | NNGRRT (49 g) | NGG (223 g) | NNNRRT (122 g) |
|---|---:|---:|---:|
| **kernel scan** (`search_chunk_kernel`, sync'd) | **44.6 (97%)** | **228.3 (92%)** | **171.8 (78%)** |
| device upload/copyback + (pos,mm) extraction | 0.4 | 0.5 | 1.5 |
| per-hit seq/PAM decode + dict build | 0.3 | 6.0 (2%) | 18.7 (9%) |
| `_process_raw_hits` (scoring/editable/contexts) | 0.6 (1%) | 10.1 (4%) | 20.2 (9%) |
| dataframe build + TSV write + sort/concat | 0.2 | 3.4 (1%) | 7.9 (4%) |
| **search total** | **46.0** | **248.4** | **220.1** |

Search totals reproduce the rerun timings minus genome load (56.9/259.4/231.6
s whole-process vs 8.8–11.5 s load) — the instrumentation adds no material
overhead.

### Findings

1. **The CUDA kernel is 78–97% of search time.** The Cas-OFFinder gap is not
   overflow-related and not Python-related: it is raw kernel throughput.
   Kernel time scales near-linearly with guide count (0.91 s/guide at 49,
   1.02 s/guide at 223), consistent with per-position × per-guide compare
   work over the full genome; Cas-OFFinder sustains roughly 2× that rate.
2. **Overflow re-scans are a minor cost.** NNGRRT and NGG had zero overflow
   (1 scan round, 32 launches); NNNRRT needed 2 re-scan rounds (96 launches)
   for the 4 repetitive guides, part of its 171.8 s kernel time. With zero
   overflow (NGG/NNGRRT) BEguidedOFF is still 1.8–2.8× slower.
3. **Python per-hit work is secondary.** Per-hit decode + scoring + editable
   filter + contexts + dataframe/TSV totals 3–21% depending on batch, and is
   only worth attacking for hit-heavy outputs.
4. **Pipeline config barely helps.** `return_all=False` (top-3/guide) saves
   only 0.4–4% (e.g. NNNRRT 211.7 s vs 220.1 s) because the kernel dominates
   regardless of output volume. There is no configuration of the extensive
   mode that dodges the kernel cost.

### Implication

The only lever that recovers real speed is the kernel itself
(`beguidedoff/cuda_search.py`): e.g. word-wise early rejection, cheaper PAM
checks, fewer memory transactions per compare. Python-side vectorisation of
hit assembly would cap out at ~10–20% on the most repetitive batch and ~0–4%
elsewhere.

---

## Kernel rework: packed PAM test + hoisted extraction (2026-10-25)

The stage profiling above showed the CUDA kernel at 78-97% of search time,
with the hot path dominated by per-(position, guide) work. Three redundancies
in `search_chunk_kernel` were the cause:

1. `_pack_window_20` ran **inside the guide loop** — ~120 instructions to
   re-extract the same 20-nt window for every guide at every position
   (it depends only on the position).
2. `_pam_matches` re-extracted the PAM bases per guide per position
   (they also depend only on the position) and tested them base-by-base
   against IUPAC mask bytes.
3. Extractions were per-base loops (div/mod/shift per nucleotide).

### What changed (`beguidedoff/cuda_search.py`)

* **Hoisted per-position extraction**: the 40-bit guide window and the 12-bit
  PAM field are extracted once per position, before the guide loop.
* **Word-wise `_pack_bits`**: window/PAM assembly from 2-3 uint32 words with
  shifts instead of per-base loops.
* **Packed PAM test**: each guide's PAM is precompiled on the host into
  `(pattern, compare-mask)` bit fields (2 bits per position; a position is
  "compared" where its IUPAC allowed-set is expressible as fixed bits — true
  for A/C/G/T, R, Y, M, K, N). The per-pair PAM check is now one XOR + AND.
  S/W/B/D/H/V positions fall back to the old per-base test (`pam_exotic`).
  Encoding verified exhaustively against the nibble test for NGG, NNGRRT,
  NNNRRT, NNNNNN, NGGNT, TTTV, TTTVN, GGSCC, S, W, A, YK (all 4^plen genome
  PAM strings each).
* **Genome device-upload cache**: overflow re-scan rounds no longer re-upload
  ~775 MB per strand (identity-checked cache, max 4 entries).
* **Hit buffers without the -1 fill** (host only reads entries below count).
* Shared memory unchanged at 32 KB (guides + packed pattern + compare mask),
  preserving the previous occupancy; kernel signature change is internal to
  `cuda_search.py` — `finder.py` is unchanged.

### Validation

* `pytest tests/` — 71 passed (encoding, PAM, scoring, chunking, overflow
  re-scan, fast-vs-extensive equivalence, end-to-end).
* **Byte-identical outputs** vs the concordance-validated rerun for all three
  family batches (414,438 / 20,218 / 811,624 hits, `diff` clean).
* `compare_rerun.py` — concordance unchanged: 1,243,720 / 1,243,724 overlap,
  99.9999%, same 5 N-in-reference differences.
* **Exotic-PAM kernel path (added 2026-10-29)**: the packed-PAM fallback for
  S/W/B/D/H/V positions had no GPU coverage (the family batches use only
  R/Y codes). New test `TestExoticPamKernelPath` in
  `tests/test_extensive_gpu.py` plants exact/1-mm/2-mm sites for TTTV, SGG
  and WGG PAMs (plus an NGG control and PAM-violating decoys) and requires
  the CUDA extensive search to match an independent brute-force reference
  exactly on both strands, with every planted site found. Suite now
  **72 passed**.

### Timing (same machine/GPU/inputs, benchmark configuration)

| family | kernel old → new | search old → new | whole-process old → new | Cas-OFFinder | vs Cas-OFFinder |
|---|---|---|---|---|---|
| NGG (223 g) | 228.3 → 30.0 s (7.6×) | 248.4 → 50.2 s | 259.4 → 59.9 s | 140.7 s | **2.35× faster** |
| NNGRRT (49 g) | 44.6 → 4.8 s (9.2×) | 46.0 → 6.2 s | 56.9 → 16.7 s | 20.6 s | **1.23× faster** |
| NNNRRT (122 g) | 171.8 → 22.7 s (7.6×) | 220.1 → 68.9 s | 231.6 → 78.7 s | 67.6 s | 0.86× (incl. tandem re-scan) |
| **total** | | **514.5 → 125.3 s (4.1×)** | **547.9 → 155.3 s (3.5×)** | **228.9 s** | **1.47× faster** |

The May-2026 "BEguidedOFF ~1.4× faster" table remains unreproducible (its
Cas-OFFinder timings were environmentally inflated and its two sides covered
different guide sets — see the rerun section above), but after the kernel
rework BEguidedOFF extensive is now genuinely faster than Cas-OFFinder v2.4.1
on the same hardware while producing scored, annotated output.

### New stage breakdown (benchmark configuration)

| stage (s) | NNGRRT (49 g) | NGG (223 g) | NNNRRT (122 g) |
|---|---:|---:|---:|
| kernel scan | 4.8 (78%) | 30.0 (60%) | 22.7 (33%) |
| per-hit seq/PAM decode + dict build | 0.3 | 6.3 (13%) | 18.1 (26%) |
| `_process_raw_hits` | 0.5 (9%) | 10.0 (20%) | 19.4 (28%) |
| dataframe + TSV + sort/concat | 0.2 | 3.4 (7%) | 7.8 (11%) |
| **search total** | **6.2** | **50.2** | **68.9** |

The kernel is no longer the only term: on hit-heavy batches the Python-side
per-hit assembly (dict build + scoring/contexts + dataframe) is now
co-dominant — the next optimisation target if more speed is needed.
Pipeline-style output (`return_all=False`, top-3) saves 8.8% on NNNRRT
(62.8 vs 68.9 s). The overflow re-scan of the tandem-repeat guide now costs
~2 extra genome passes of a *single* guide (~1.2 s of kernel time each) plus
re-decode of its 523,773 hits.

### Extensive perf across the benchmark sets (new kernel, 2026-10-29)

Full-set re-measurement after the kernel rework, same protocol per set as the
original postfix/chunked runs (500x38: 2500 unique scans, `--chunk-scans 2000`,
2 chunks, 1 rep; fast mode re-run confirms machine state — code unchanged):

| set (unique scans) | extensive old → new | fast old → new | fast/extensive old → new |
|---|---|---|---|
| 100 | 115.9 → 15.76 s (7.4×) | 1.219 → 1.266 s | 95× → **12.5×** |
| 500 | 658.5 → 180.42 s (3.65×) | 8.196 → 8.498 s | 80× → **21.2×** |
| 500x38 (2500) | 3254.7 → 1433.96 s (2.27×) | 42.685 → 46.558 s | 76× → **30.8×** |

The fast-vs-extensive fold-change drops from ~76–95× to **~12–31×**: the
kernel rework removed most of the extensive mode's head start over the fixed
fast mode. The 500x38 set shows the smallest kernel speedup (2.27×) because
its 38 editor profiles exercise the exotic-PAM fallback path (S/W/B/D/H/V
positions take the per-base test instead of the packed XOR+AND).

Correctness gate re-run on the 100-guide set with the new kernel:
`EQUIVALENCE [PASS]` — fast vs extensive agree on all 78 hits with score
<= 5. Overflow guide sets are unchanged (same 5 guides on the 500 set; 4
re-scan warnings on chunked 500x38). Extensive outputs remain byte-identical
to the concordance-validated rerun (see Validation above).
