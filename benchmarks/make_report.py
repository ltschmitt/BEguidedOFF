#!/usr/bin/env python
"""Assemble benchmarks/REPORT.md from the benchmark result JSONs."""
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).resolve().parent    # this dir: benchmarks/ (scripts + reports)
NOGIT = BENCH.parent / "nogit"             # local heavy data (not in the repository)
REPO = BENCH.parent
RESULTS = NOGIT / "bench" / "results"
sys.path.insert(0, str(REPO))

SETS = ["10", "50", "100", "500", "500x38"]


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt_s(x):
    if x is None:
        return "—"
    return f"{x:,.2f}" if x < 1000 else f"{x:,.0f}"


def cpu_model():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def ram_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal"):
            return int(line.split()[1]) / 2**20
    return float("nan")


def main():
    equiv = load("equivalence_100.json")
    loadres = load("load.json") or {}
    meta = load("build_meta.json") or {}

    gpu = None
    perf = {}
    for mode in ("extensive", "fast"):
        for s in SETS:
            rec = load(f"perf_{mode}_{s}.json")
            if not rec and mode == "extensive" and s == "500x38":
                rec = load(f"perf_{mode}_{s}_chunked.json")
            if rec:
                perf[(mode, s)] = rec
                gpu = gpu or rec.get("gpu_name")

    lines = []
    w = lines.append
    w("# BEguidedOFF fast vs extensive — benchmark report")
    w("")
    w(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · branch `feature/fastoff` · "
      "human genome GRCh38 (Ensembl 115; byte-identical to 112, md5 `a65212262de00761b43869d5c08c8e4d`)")
    w("")
    w("## Environment")
    w("")
    w(f"- CPU: {cpu_model()}, {platform.machine()}")
    w(f"- RAM: {ram_gb():.0f} GB")
    if gpu:
        w(f"- GPU: {gpu}")
    w(f"- Python {platform.python_version()}, numpy {__import__('numpy').__version__}, "
      f"pandas {__import__('pandas').__version__}")
    w("")
    w("## Phase 0 — correctness gate (fast vs extensive, real genome)")
    w("")
    if equiv:
        status = "PASS" if equiv["pass"] else "FAIL"
        w(f"**{status}** on the 100-guide mixed set ({equiv['n_unique_scans']} unique scans).")
        w("")
        w("| check | mismatches |")
        w("|---|---|")
        w(f"| remove/keep verdicts | {len(equiv['verdict_mismatches'])} |")
        w(f"| full hit set with score ≤ 5 | {len(equiv['hitset_mismatches'])} |")
        w(f"| n_offtargets_below_threshold (T=3) | {len(equiv['count_mismatches'])} |")
        w(f"| top-3 lists | {len(equiv['top3_mismatches'])} |")
        w("")
        w(f"Extensive enumerated {equiv['n_ext_rows_all']:,} hits (≤ 4 mismatches, incl. non-editable) "
          f"in {fmt_s(equiv['ext_search_s'])} s; {equiv['n_ext_rows_le_cutoff']} of them score ≤ 5 — "
          f"fast reproduced that subset exactly ({fmt_s(equiv['fast_search_s'])} s, return_all).")
    w("")
    w("The comparison covers all sites that can matter for the removal decision: any site with "
      "CCTOP score ≤ 5 has ≤ 4 mismatches, which extensive enumerates exhaustively; fast "
      "enumerates every site with score ≤ 5 by construction (exact PAM-proximal cold region). "
      "Ties within the top-3 are broken canonically by `(score, chromosome, position, strand)` "
      "in both modes so top-N is deterministic and mode-independent.")
    w("")
    # stress test: guides that overflow extensive's hit buffer
    ovf_eq = load("equivalence_overflow.json")
    ovf_pre = load("equivalence_overflow_prefix.json")
    if ovf_eq or ovf_pre:
        n_rows = (ovf_eq or ovf_pre)["n_guides"]
        w(f"### Stress test: the 24 buffer-overflow guides ({n_rows} guide×editor rows)")
        w("")
        if ovf_pre and not ovf_pre["pass"]:
            w("**Before the fix** (buffer truncation): verdicts agreed "
              f"({len(ovf_pre['verdict_mismatches'])} mismatches), but extensive "
              "under-reported everywhere else —")
            w("")
            w("| check | mismatches |")
            w("|---|---|")
            w(f"| hit set with score ≤ 5 | {len(ovf_pre['hitset_mismatches'])} of {ovf_pre['n_guides']} |")
            w(f"| n_offtargets_below_threshold (T=3) | {len(ovf_pre['count_mismatches'])} |")
            w(f"| top-3 lists | {len(ovf_pre['top3_mismatches'])} |")
            w("")
            w(f"Extensive enumerated {ovf_pre['n_ext_rows_le_cutoff']:,} sites with score ≤ 5 "
              f"where fast enumerated {ovf_pre['n_fast_rows_le_cutoff']:,} — every fast-only site "
              "was verified genuine, i.e. **the truncation was in extensive, not in fast**.")
            w("")
        if ovf_eq and ovf_eq["pass"]:
            w("**After the fix** (automatic re-scan): [PASS] — extensive now re-scans any guide "
              "whose ≤4-mismatch hits overflow the buffer with an escalating `results_per_guide` "
              "(10 240 → 163 840 → 2 621 440, replacing the truncated hits). Its score ≤ 5 subset "
              "now matches fast exactly:")
            w("")
            w("| check | mismatches |")
            w("|---|---|")
            w(f"| verdicts | {len(ovf_eq['verdict_mismatches'])} |")
            w(f"| hit set with score ≤ 5 | {len(ovf_eq['hitset_mismatches'])} of {ovf_eq['n_guides']} |")
            w(f"| n_offtargets_below_threshold (T=3) | {len(ovf_eq['count_mismatches'])} |")
            w(f"| top-3 lists | {len(ovf_eq['top3_mismatches'])} |")
            w("")
            w(f"Both engines enumerate {ovf_eq['n_ext_rows_le_cutoff']:,} score ≤ 5 sites. Several "
              "guides needed the second escalation tier (>163 840 raw ≤4-mm hits each). The cost is "
              "runtime only — the complete enumeration is CPU-bound scoring work the truncated "
              "search used to skip. A guide that still overflows at the ceiling fails with a clear "
              "error instead of returning silent truncations.")
        elif ovf_eq and not ovf_eq["pass"]:
            w("**After the fix**: FAIL — see results/equivalence_overflow.json.")
    w("")
    w("## Phase 1 — search latency (top-3 output, count threshold 3)")
    w("")
    w("| guide set | input rows | unique scans | extensive (GPU) | fast (CPU, 24c) | fast (CPU, 4c) | speedup (4c) |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for s in SETS:
        e = perf.get(("extensive", s))
        f = perf.get(("fast", s))
        f4 = load(f"perf_fast_{s}_4core.json")
        rows = f["n_input_rows"] if f else (e["n_input_rows"] if e else "")
        scans = f["n_unique_scans"] if f else (e["n_unique_scans"] if e else "")
        em = fmt_s(e["median_s"]) if e else "—"
        fm = fmt_s(f["median_s"]) if f else "—"
        f4m = fmt_s(f4["median_s"]) if f4 else "—"
        sp = ""
        if e and f4:
            sp = f"{e['median_s'] / f4['median_s']:.0f}×"
        note = " †" if s == "500x38" and e and e.get("chunked") else ""
        w(f"| {s} | {rows} | {scans} | {em}{note} | {fm} | {f4m} | {sp} |")
    w("")
    w("extensive = median of timed reps after warmup (CUDA JIT compiled once per process); "
      "fast = median of 3 reps. 4c = `taskset -c 0-3` emulating the 4-core web VM.")
    w("")
    w("† extensive cannot run the 500x38 set in one call: the CUDA kernel caps guides per "
      "launch (`cuda_search.MAX_GUIDES = 2048`, shared-memory budget), so it was split into "
      "2 chunks of ≤ 2,000 unique scans and the per-call times summed (1 rep). The production "
      "pipeline would have to batch the same way; fast mode has no such limit.")
    w("")
    # resource footnotes
    e500 = perf.get(("extensive", "500"))
    f500 = perf.get(("fast", "500"))
    e38 = perf.get(("extensive", "500x38"))
    f38 = perf.get(("fast", "500x38"))
    if e500 or e38:
        w("Resource peaks during search:")
        w("")
        if e500:
            w(f"- extensive 500: peak process RSS {e500['peak_rss_gb']:.1f} GB, "
              f"peak VRAM {e500['peak_vram_gb']:.2f} GB")
        if e38:
            w(f"- extensive 500x38: peak RSS {e38['peak_rss_gb']:.1f} GB, "
              f"peak VRAM {e38['peak_vram_gb']:.2f} GB")
        if f500:
            w(f"- fast 500: peak process RSS {f500['peak_rss_gb']:.2f} GB (includes mmap'd index)")
        if f38:
            w(f"- fast 500x38: peak process RSS {f38['peak_rss_gb']:.2f} GB")
    w("")
    ovf = sum(rec.get("overflow_warnings", 0) for rec in perf.values())
    n_ovf_guides = len((perf.get(("extensive", "500x38")) or {}).get("overflow_guides", []))
    w(f"Hit-buffer overflows in the extensive runs: {ovf} warnings "
      f"({n_ovf_guides} unique guides exceed the 10,240-hit buffer at ≤ 4 mismatches — see the "
      "stress test above; fast mode reported 0 warnings anywhere). Guides whose exact 12-mer "
      "cold anchor occurs > 2,000 times were excluded from the sets up front (6 of 346 real "
      "protospacers) because they exceed both engines' caps — documented limitation.")
    w("")
    post = load("perf_extensive_500_postfix.json")
    if post:
        pre = perf.get(("extensive", "500")) or {}
        pre_med = pre.get("median_s")
        post_med = post.get("median_s")
        ovf_guides = post.get("overflow_guides") or []
        ovf_rounds = post.get("overflow_warnings", 0)
        if pre_med and post_med:
            delta = post_med - pre_med
            w(f"Note: the extensive timings above were measured **before** the overflow fix. "
              f"A post-fix re-run of the same extensive-500 benchmark "
              f"(`perf_extensive_500_postfix.json`, 1 timed rep) measured "
              f"**{post_med:.1f} s** vs {pre_med:.1f} s pre-fix — an overhead of "
              f"**{delta:+.1f} s** ({post_med / pre_med:.2f}×) for exact, untruncated "
              f"hit lists on the guides that overflow the 10,240-hit buffer.")
        else:
            w("Note: the extensive timings above were measured **before** the overflow fix; "
              "post-fix re-run data is present but lacks a comparable median.")
        w("")
        ex = ", ".join(ovf_guides[:5]) if ovf_guides else "—"
        w(f"Post-fix re-scan activity in that run: {ovf_rounds} re-scan round(s) logged; "
          f"example overflowing guides: {ex}. The re-scan (buffer ×16 up to 2 621 440, "
          f"fail-closed RuntimeError at the ceiling) replaces the old silent truncation; "
          f"guides that fit the buffer are unaffected.")
    else:
        w("Note: the extensive timings above were measured **before** the overflow fix. Post-fix, "
          "overflowing guides are re-scanned automatically (buffer ×16 up to 2 621 440), which adds "
          "a full-genome kernel pass per escalation tier plus the CPU scoring of the complete hit "
          "list — only for the ~5 % of guides that overflow; guides that fit the buffer are "
          "unaffected.")
    w("")
    w("## Phase 2 — index economics")
    w("")
    w("| artifact | size | build time | build peak RSS | load (warm) | load (cold) |")
    w("|---|---:|---:|---:|---:|---:|")
    fmi_size = loadres.get("fmindex_size_gb")
    npz_size = loadres.get("npz_size_gb")
    bw = fmt_s(meta.get("fmindex_build_wall_s"))
    br = f"{meta['fmindex_build_peak_rss_gb']:.1f} GB" if meta.get("fmindex_build_peak_rss_gb") else "—"
    fs = f"{fmi_size:.2f} GB" if fmi_size is not None else "—"
    ns = f"{npz_size:.2f} GB" if npz_size is not None else "—"
    w(f"| FM-index (`hg38_beguidedoff.fmi`) | {fs} | {bw} | {br} | "
      f"{fmt_s(loadres.get('fmindex_load_warm'))} s | {fmt_s(loadres.get('fmindex_load_cold'))} s |")
    w(f"| 2-bit npz (`hg38_beguidedoff.npz`) | {ns} | — | — | "
      f"{fmt_s(loadres.get('npz_load_warm'))} s | {fmt_s(loadres.get('npz_load_cold'))} s |")
    w("")
    w("Cold = page cache dropped with `posix_fadvise(DONTNEED)` before the load. The FM-index "
      "is mmapped, so a fast-mode process pays the cold read once and the OS keeps it hot "
      "across requests on a long-running service.")
    w("")
    w("## Notes")
    w("")
    w("- Guide sets: first N real protospacers from `input_example.tsv` (unique, "
      "editor-name-normalized), padded to N with genomic NGG protospacers sampled from the "
      "FM-index (seed 42, anchor occurrence ≤ 2000 on both strands). The 500x38 set expands "
      "the 500-set over all 38 base-editor profiles (19,000 rows).")
    w("- `input_example.tsv` spells `SaABE8e` as the older `Sa-ABE8e`; the benchmark normalizes "
      "editor names punctuation-insensitively. The production input/profile data should be "
      "brought into sync (7 rows affected).")
    w("- `io.py` reads the on-target strand from the `strand_used` column, while "
      "`input_example.tsv` uses `strand`; without it, on-target exclusion cannot work. "
      "Flagged for the BEguided integration.")
    w("- Extensive runs `max_mismatches=4`, which fully covers fast's score ≤ 5 enumeration "
      "(5 mismatches ⇒ min score 8.93 > 5).")
    w("- The index was rebuilt from the 115 FASTA; the 112/115 primary assemblies are "
      "byte-identical, and the re-encoded npz is bit-identical to the previous 112-derived "
      "one (verified field-by-field), so all three artifacts describe the same genome.")

    out = BENCH / "REPORT.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
