#!/usr/bin/env python
"""BEguidedOFF benchmark harness.

Phases
------
correctness  extensive (GPU, return_all) vs fast (CPU) on a guide set:
             compare all hits with score <= 5 per guide (must be identical
             sets), per-guide remove/keep verdicts, per-guide
             n_offtargets_below_threshold (recomputed from extensive rows with
             the guide's editor window), and the top-3 lists.
perf         search() wall time per guide set, warmup + N reps, median;
             peak RSS (process), peak VRAM (extensive), overflow warnings.
load         index/npz load times, warm and cold (posix_fadvise DONTNEED).

Usage:
  run_bench.py correctness --set 100
  run_bench.py perf --mode extensive --sets 10,50,100,500,500x38
  run_bench.py perf --mode fast --sets 10,50,100,500,500x38
  run_bench.py load
"""
import argparse
import gc
import json
import resource
import subprocess
import sys
import time
import warnings
from pathlib import Path

BENCH = Path(__file__).resolve().parent    # this dir: benchmarks/ (scripts + reports)
NOGIT = BENCH.parent / "nogit"             # local heavy data (not in the repository)
REPO = BENCH.parent
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from beguidedoff.base_editing import editable_base_in_window  # noqa: E402
from beguidedoff.io import load_guides_from_tsv  # noqa: E402

NPZ = NOGIT / "hg38_beguidedoff.npz"
FMI = NOGIT / "hg38_beguidedoff.fmi"
PROFILES = REPO / "base_editor_profiles.tsv"
GUIDES = NOGIT / "bench" / "guides"
RESULTS = NOGIT / "bench" / "results"

CUTOFF = 5.0
COUNT_THRESHOLD = 3.0
N_TOP = 3
COMPARE_COLS = [
    "guide_rna_sequence", "base_editor", "ot_chromosome", "ot_position",
    "ot_strand", "ot_sequence", "ot_pam", "n_mismatches",
    "mismatch_positions", "score",
]


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0


def load_guides(name):
    return load_guides_from_tsv(str(GUIDES / f"guides_{name}.tsv"),
                                presets_path=str(PROFILES))


def unique_scans(guides):
    return len({(g.guide_seq, g.pam) for g in guides})


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------
def rows_le(df, cutoff):
    if df is None or df.empty:
        return pd.DataFrame(columns=COMPARE_COLS)
    return df[df["score"] <= cutoff][COMPARE_COLS]


def hit_key(row):
    return tuple(str(row[c]) for c in COMPARE_COLS)


def run_correctness(set_name):
    from beguidedoff import beguidedoff

    guides = load_guides(set_name)
    by_key = {(g.guide_seq, g.base_editor or ""): g for g in guides}

    ext = beguidedoff(mode="extensive", genome_path=str(NPZ), max_mismatches=4,
                      n_top_offtargets=N_TOP)
    fast = beguidedoff(mode="fast", index_path=str(FMI), search_cutoff=CUTOFF,
                       n_top_offtargets=N_TOP)

    t0 = time.perf_counter()
    df_ext = ext.search(guides=guides, return_all=True,
                        count_score_threshold=COUNT_THRESHOLD, verbose=False)
    ext_search_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    df_fast = fast.search(guides=guides, return_all=True,
                          count_score_threshold=COUNT_THRESHOLD, verbose=False)
    fast_search_s = time.perf_counter() - t0

    ext_le = rows_le(df_ext, CUTOFF)
    fast_le = rows_le(df_fast, CUTOFF)

    ext_sets, fast_sets = {}, {}
    for _, r in ext_le.iterrows():
        ext_sets.setdefault((r["guide_rna_sequence"], r["base_editor"]), set()).add(hit_key(r))
    for _, r in fast_le.iterrows():
        fast_sets.setdefault((r["guide_rna_sequence"], r["base_editor"]), set()).add(hit_key(r))

    all_keys = set(ext_sets) | set(fast_sets)
    verdict_mismatch, hitset_mismatch = [], []
    for key in sorted(all_keys):
        e, f = ext_sets.get(key, set()), fast_sets.get(key, set())
        if bool(e) != bool(f):
            verdict_mismatch.append({"guide": key[0], "editor": key[1],
                                     "ext_hits": len(e), "fast_hits": len(f)})
        if e != f:
            only_ext = sorted(e - f)[:5]
            only_fast = sorted(f - e)[:5]
            hitset_mismatch.append({"guide": key[0], "editor": key[1],
                                    "n_ext": len(e), "n_fast": len(f),
                                    "only_ext": only_ext, "only_fast": only_fast})

    # n_offtargets_below_threshold: fast's column vs recomputed from extensive rows
    count_mismatch = []
    if not df_fast.empty:
        fast_counts = df_fast.groupby(["guide_rna_sequence", "base_editor"])[
            "n_offtargets_below_threshold"].first().to_dict()
        # editable + score<=threshold rows from the extensive enumeration
        ext_counts = {}
        for _, r in ext_le.iterrows():
            g = by_key[(r["guide_rna_sequence"], r["base_editor"])]
            if r["score"] <= COUNT_THRESHOLD and editable_base_in_window(
                r["ot_sequence"], g.editor_type, g.edit_window_start, g.edit_window_end
            ):
                ext_counts[(r["guide_rna_sequence"], r["base_editor"])] = \
                    ext_counts.get((r["guide_rna_sequence"], r["base_editor"]), 0) + 1
        for key in set(fast_counts) | set(ext_counts):
            if fast_counts.get(key, 0) != ext_counts.get(key, 0):
                count_mismatch.append({"guide": key[0], "editor": key[1],
                                       "fast": fast_counts.get(key, 0),
                                       "ext_recomputed": ext_counts.get(key, 0)})

    # top-3 lists (the production output shape)
    top3_mismatch = []
    df_ext3 = ext.search(guides=guides, return_all=False,
                         count_score_threshold=COUNT_THRESHOLD, verbose=False)
    df_fast3 = fast.search(guides=guides, return_all=False,
                           count_score_threshold=COUNT_THRESHOLD, verbose=False)
    e3 = rows_le(df_ext3, CUTOFF)
    f3 = rows_le(df_fast3, CUTOFF)
    e3_sets, f3_sets = {}, {}
    for _, r in e3.iterrows():
        e3_sets.setdefault((r["guide_rna_sequence"], r["base_editor"]), set()).add(hit_key(r))
    for _, r in f3.iterrows():
        f3_sets.setdefault((r["guide_rna_sequence"], r["base_editor"]), set()).add(hit_key(r))
    for key in sorted(set(e3_sets) | set(f3_sets)):
        if e3_sets.get(key, set()) != f3_sets.get(key, set()):
            top3_mismatch.append({"guide": key[0], "editor": key[1],
                                  "n_ext": len(e3_sets.get(key, set())),
                                  "n_fast": len(f3_sets.get(key, set()))})

    ok = not (verdict_mismatch or hitset_mismatch or count_mismatch or top3_mismatch)
    out = {
        "set": set_name,
        "pass": ok,
        "n_guides": len(guides),
        "n_unique_scans": unique_scans(guides),
        "n_ext_rows_all": int(len(df_ext)),
        "n_ext_rows_le_cutoff": int(len(ext_le)),
        "n_fast_rows_le_cutoff": int(len(fast_le)),
        "ext_search_s": round(ext_search_s, 3),
        "fast_search_s": round(fast_search_s, 3),
        "verdict_mismatches": verdict_mismatch,
        "hitset_mismatches": hitset_mismatch,
        "count_mismatches": count_mismatch,
        "top3_mismatches": top3_mismatch,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"equivalence_{set_name}.json").write_text(json.dumps(out, indent=2))
    print(f"EQUIVALENCE [{'PASS' if ok else 'FAIL'}] set={set_name} "
          f"guides={len(guides)} hits<=5 ext={len(ext_le)} fast={len(fast_le)}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# performance
# ---------------------------------------------------------------------------
def chunk_guides_by_scans(guides, max_scans):
    """Split guides into chunks with <= max_scans unique (seq, pam) scan keys.

    Rows sharing a scan key stay in the same chunk so per-guide counting is
    unaffected.  Needed because extensive mode's CUDA kernel caps guides per
    call (cuda_search.MAX_GUIDES = 2048).
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for g in guides:
        groups[(g.guide_seq, g.pam)].append(g)
    chunks, cur, cur_scans = [], [], set()
    for key, rows in groups.items():
        if cur_scans and key not in cur_scans and len(cur_scans) >= max_scans:
            chunks.append(cur)
            cur, cur_scans = [], set()
        cur.extend(rows)
        cur_scans.add(key)
    if cur:
        chunks.append(cur)
    return chunks


class _LogCapCounter:
    """Count fast-mode 'anchor hit cap' log records during a search."""

    def __init__(self):
        self.count = 0
        self.records = []

    def __call__(self, record):
        msg = record.getMessage()
        if "cap" in msg.lower() or "overflow" in msg.lower():
            self.count += 1
            self.records.append(msg)
        return False  # never swallow the record


def _search_with_warning_count(finder, guides, chunk_scans=None):
    import logging
    import re

    counter = _LogCapCounter()
    handler = logging.Handler()
    handler.emit = counter  # type: ignore[method-assign]
    log = logging.getLogger("beguidedoff")
    log.addHandler(handler)
    ovf_guides = set()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            if chunk_scans:
                nrows = 0
                for ch in chunk_guides_by_scans(guides, chunk_scans):
                    df = finder.search(guides=ch, return_all=False,
                                       count_score_threshold=COUNT_THRESHOLD,
                                       verbose=False)
                    nrows += int(len(df))
            else:
                df = finder.search(guides=guides, return_all=False,
                                   count_score_threshold=COUNT_THRESHOLD, verbose=False)
                nrows = int(len(df))
            dt = time.perf_counter() - t0
    finally:
        log.removeHandler(handler)
    for w in caught:
        m = re.search(r"guide (\S+)", str(w.message))
        if m:
            ovf_guides.add(m.group(1))
    for rec in getattr(counter, "records", []):
        m = re.search(r"e\.g\. ([^;)]+)", rec)
        if m:
            for name in m.group(1).split(","):
                ovf_guides.add(name.strip())
            continue
        m = re.search(r"guide (\S+)", rec)
        if m:
            ovf_guides.add(m.group(1))
    n_ovf = counter.count + sum(1 for w in caught if "overflow" in str(w.message).lower())
    return dt, nrows, n_ovf, sorted(ovf_guides)


def run_perf(mode, set_names, reps_override=None, tag=None, chunk_scans=None):
    import torch
    from beguidedoff import beguidedoff

    suffix = f"_{tag}" if tag else ""
    RESULTS.mkdir(exist_ok=True)
    if mode == "extensive":
        t0 = time.perf_counter()
        finder = beguidedoff(mode="extensive", genome_path=str(NPZ),
                             max_mismatches=4, n_top_offtargets=N_TOP)
        load_s = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        finder = beguidedoff(mode="fast", index_path=str(FMI),
                             search_cutoff=CUTOFF, n_top_offtargets=N_TOP)
        load_s = time.perf_counter() - t0

    default_reps = {"10": 3, "50": 3, "100": 3, "500": 2, "500x38": 1}
    if mode == "fast":
        default_reps = {k: 3 for k in default_reps}
    if reps_override:
        default_reps = {k: reps_override for k in default_reps}

    # first search compiles the numba CUDA kernels -- do it on the smallest set
    first_guides = load_guides(set_names[0])
    t0 = time.perf_counter()
    finder.search(guides=first_guides[:1], return_all=False,
                  count_score_threshold=COUNT_THRESHOLD, verbose=False)
    jit_compile_s = time.perf_counter() - t0

    for set_name in set_names:
        guides = load_guides(set_name)
        n_scans = unique_scans(guides)
        reps = default_reps.get(set_name, 2)
        chunks = chunk_guides_by_scans(guides, chunk_scans) if chunk_scans else None

        # warmup
        _search_with_warning_count(finder, guides, chunk_scans)
        timings, rows, ovf, ovf_guides = [], [], 0, set()
        for _ in range(reps):
            if mode == "extensive":
                torch.cuda.reset_peak_memory_stats()
            dt, nrows, n_ovf, g_ovf = _search_with_warning_count(finder, guides, chunk_scans)
            timings.append(round(dt, 3))
            rows.append(nrows)
            ovf += n_ovf
            ovf_guides.update(g_ovf)
        rec = {
            "mode": mode, "set": set_name, "n_input_rows": len(guides),
            "n_unique_scans": n_scans, "reps": reps,
            "chunked": bool(chunk_scans),
            "n_chunks": len(chunks) if chunks else 1,
            "timings_s": timings,
            "median_s": sorted(timings)[len(timings) // 2],
            "output_rows_last": rows[-1],
            "overflow_warnings": ovf,
            "overflow_guides": sorted(ovf_guides),
            "load_s": round(load_s, 3),
            "jit_compile_s": round(jit_compile_s, 3) if set_name == set_names[0] else None,
            "peak_rss_gb": round(peak_rss_gb(), 2),
        }
        if mode == "extensive":
            rec["peak_vram_gb"] = round(
                torch.cuda.max_memory_allocated() / 2**30, 2)
            rec["gpu_name"] = torch.cuda.get_device_name(0)
        (RESULTS / f"perf_{mode}_{set_name}{suffix}.json").write_text(json.dumps(rec, indent=2))
        print(f"PERF {mode:9s} set={set_name:7s} scans={n_scans:5d} "
              f"median={rec['median_s']:9.3f}s reps={reps} ovf_warn={ovf}")
        gc.collect()


# ---------------------------------------------------------------------------
# load timings
# ---------------------------------------------------------------------------
def _drop_cache(path):
    """Best-effort page-cache drop so the next load is cold."""
    import os
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return True
    except OSError:
        return False


def run_load():
    from beguidedoff.fast.fmindex import SdslGenomeIndex
    from beguidedoff.genome_converter import load_genome

    out = {}

    def timed(fn, label, cold=False):
        t0 = time.perf_counter()
        obj = fn()
        dt = time.perf_counter() - t0
        del obj
        out[label] = round(dt, 3)
        print(f"LOAD {label:28s} {dt:8.3f}s {'(cold)' if cold else '(warm)'}")

    index_path = str(FMI)
    timed(lambda: SdslGenomeIndex.load(index_path), "fmindex_load_warm")
    if _drop_cache(FMI) and _drop_cache(Path(str(FMI) + ".meta")):
        timed(lambda: SdslGenomeIndex.load(index_path), "fmindex_load_cold", cold=True)

    npz_path = str(NPZ)
    timed(lambda: load_genome(npz_path), "npz_load_warm")
    if _drop_cache(NPZ):
        timed(lambda: load_genome(npz_path), "npz_load_cold", cold=True)

    out["fmindex_size_gb"] = round(FMI.stat().st_size / 2**30, 3)
    out["npz_size_gb"] = round(NPZ.stat().st_size / 2**30, 3)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "load.json").write_text(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    c = sub.add_parser("correctness")
    c.add_argument("--set", default="100")
    p = sub.add_parser("perf")
    p.add_argument("--mode", choices=["extensive", "fast"], required=True)
    p.add_argument("--sets", default="10,50,100,500,500x38")
    p.add_argument("--reps", type=int, default=None)
    p.add_argument("--tag", default=None, help="suffix for the result JSON filename")
    p.add_argument("--chunk-scans", type=int, default=None,
                   help="split input into chunks of <= N unique scans per search call "
                        "(extensive mode caps at cuda_search.MAX_GUIDES=2048)")
    sub.add_parser("load")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    if args.phase == "correctness":
        sys.exit(run_correctness(args.set))
    if args.phase == "perf":
        run_perf(args.mode, [s for s in args.sets.split(",") if s], args.reps, args.tag,
                 args.chunk_scans)
    if args.phase == "load":
        run_load()


if __name__ == "__main__":
    main()
