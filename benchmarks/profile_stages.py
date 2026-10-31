#!/usr/bin/env python
"""Stage-level profiling of the BEguidedOFF extensive (GPU) search.

Decomposes the wall time of one ``finder.search()`` call into stages by
wrapping timers around the internal boundaries of ``beguidedoff.finder``
(monkey-patching only -- no production code is modified):

  dedup             _deduplicate_guides
  scan_gpu_total    _search_gpu (all strands + all overflow re-scan rounds)
  scan_strands      _scan_strands (per round; call count = number of rounds)
  scan_strand       _search_strand_cuda (per strand, per batch)
  scan_genome       search_genome_cuda (device alloc, uploads, launches, D2H)
  scan_kernel       search_chunk_kernel launches only (sync'd, summed)
  kernel_xfer       scan_genome - scan_kernel  (uploads + copyback + (pos,mm) extraction)
  hit_dict_build    scan_strand - scan_genome  (per-hit seq/PAM decode + dict creation)
  rescan_overhead   scan_gpu_total - scan_strands
  process_raw_hits  _process_raw_hits (per-hit Python scoring/assembly)
    cctop           calculate_cctop_score
    editable        editable_base_in_window
    contexts        get_trinucleotide_contexts
  df_build          offtargets_to_dataframe (summed over per-guide calls)
  to_csv            pd.DataFrame.to_csv
  df_other          search_total - all of the above (per-guide sort, concat, ...)

Two configurations per family batch:
  benchmark  return_all + include_non_editable  (same enumeration space as
             Cas-OFFinder; the configuration used for the rerun comparison)
  pipeline   return_all=False  (web-server style: keeps only the top
             n_top_offtargets hits per guide in the output table)

Results accumulate in ``profile_stages_report.json`` keyed by
``<FAMILY>/<config>`` so the six runs build one report.

Usage:
  python profile_stages.py --family NNGRRT --config benchmark
  python profile_stages.py --family NGG --config benchmark --limit-guides 5
"""
import argparse
import functools
import json
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent    # this dir: benchmarks/ (scripts + reports)
NOGIT = BENCH.parent / "nogit"             # local heavy data (not in the repository)
REPO = BENCH.parent
sys.path.insert(0, str(REPO))

BASE = NOGIT / "relevant_files" / "cas-offinder"
NPZ = NOGIT / "hg38_beguidedoff.npz"
REPORT_JSON = BENCH / "profile_stages_report.json"
FAMILIES = ("NGG", "NNGRRT", "NNNRRT")

STAGE_ORDER = [
    "dedup",
    "scan_gpu_total",
    "scan_strands",
    "scan_strand",
    "scan_genome",
    "scan_kernel",
    "kernel_xfer",
    "hit_dict_build",
    "rescan_overhead",
    "process_raw_hits",
    "cctop",
    "editable",
    "contexts",
    "df_build",
    "to_csv",
    "df_other",
]


class Timer:
    def __init__(self):
        self.t = 0.0
        self.n = 0

    def add(self, dt):
        self.t += dt
        self.n += 1


T = {name: Timer() for name in STAGE_ORDER}


def _wrap_fn(timer_name, fn):
    tim = T[timer_name]

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            tim.add(time.perf_counter() - t0)

    return wrapper


def _wrap_method(cls, name, timer_name):
    fn = getattr(cls, name)
    setattr(cls, name, _wrap_fn(timer_name, fn))


def install_patches():
    import pandas as pd

    import beguidedoff.finder as F
    import beguidedoff.cuda_search as CS

    # --- class methods -----------------------------------------------------
    cls = F.beguidedoff
    _wrap_method(cls, "_deduplicate_guides", "dedup")
    _wrap_method(cls, "_search_gpu", "scan_gpu_total")
    _wrap_method(cls, "_scan_strands", "scan_strands")
    _wrap_method(cls, "_process_raw_hits", "process_raw_hits")

    # --- module-level functions referenced inside finder.py ----------------
    F._search_strand_cuda = _wrap_fn("scan_strand", F._search_strand_cuda)
    F.search_genome_cuda = _wrap_fn("scan_genome", F.search_genome_cuda)
    F.calculate_cctop_score = _wrap_fn("cctop", F.calculate_cctop_score)
    F.editable_base_in_window = _wrap_fn("editable", F.editable_base_in_window)
    F.get_trinucleotide_contexts = _wrap_fn("contexts", F.get_trinucleotide_contexts)
    F.offtargets_to_dataframe = _wrap_fn("df_build", F.offtargets_to_dataframe)

    # --- raw CUDA kernel launches (sync'd) ---------------------------------
    orig_kernel = CS.search_chunk_kernel

    class KernelShim:
        def __init__(self, dispatcher):
            self._d = dispatcher

        def __getitem__(self, cfg):
            launcher = self._d[cfg]

            def call(*args, **kwargs):
                CS.cuda.synchronize()
                t0 = time.perf_counter()
                try:
                    return launcher(*args, **kwargs)
                finally:
                    CS.cuda.synchronize()
                    T["scan_kernel"].add(time.perf_counter() - t0)

            return call

    CS.search_chunk_kernel = KernelShim(orig_kernel)

    # --- output write -------------------------------------------------------
    orig_to_csv = pd.DataFrame.to_csv

    def timed_to_csv(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig_to_csv(self, *args, **kwargs)
        finally:
            T["to_csv"].add(time.perf_counter() - t0)

    pd.DataFrame.to_csv = timed_to_csv


def run(family, config, limit_guides):
    import beguidedoff.finder as F

    tsv = BASE / f"beguidedoff_input_{family}.tsv"
    if limit_guides:
        lines = tsv.read_text().strip().splitlines()
        tsv = BASE / f"beguidedoff_input_{family}_limit{limit_guides}.tsv"
        tsv.write_text("\n".join([lines[0]] + lines[1 : 1 + limit_guides]) + "\n")
    out = BASE / f"beguidedoffoutput_{family}_{config}_profile.tsv"

    benchmark = config == "benchmark"

    t0 = time.perf_counter()
    finder = F.beguidedoff(str(NPZ), max_mismatches=5, device="cuda:0")
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    df = finder.search(
        input_file=str(tsv),
        return_all=benchmark,
        include_non_editable=benchmark,
        verbose=True,
        output_file=str(out),
    )
    t_search = time.perf_counter() - t1

    stages = {name: T[name].t for name in STAGE_ORDER}

    # derived buckets
    stages["kernel_xfer"] = stages["scan_genome"] - stages["scan_kernel"]
    stages["hit_dict_build"] = stages["scan_strand"] - stages["scan_genome"]
    stages["rescan_overhead"] = stages["scan_gpu_total"] - stages["scan_strands"]
    stages["df_other"] = (
        t_search
        - stages["dedup"]
        - stages["scan_gpu_total"]
        - stages["process_raw_hits"]
        - stages["df_build"]
        - stages["to_csv"]
    )

    counts = {
        "scan_genome_calls": T["scan_genome"].n,
        "kernel_launches": T["scan_kernel"].n,
        "scan_rounds": T["scan_strands"].n,
        "cctop_calls": T["cctop"].n,
        "editable_calls": T["editable"].n,
        "contexts_calls": T["contexts"].n,
        "df_build_calls": T["df_build"].n,
    }

    result = {
        "family": family,
        "config": config,
        "n_guides_input": limit_guides or (len(tsv.read_text().strip().splitlines()) - 1),
        "n_rows_out": int(len(df)),
        "genome_load_s": round(t_load, 2),
        "search_total_s": round(t_search, 2),
        "stages_s": {k: round(v, 3) for k, v in stages.items()},
        "counts": counts,
    }

    # accumulate across runs
    report = {}
    if REPORT_JSON.exists():
        report = json.loads(REPORT_JSON.read_text())
    report[f"{family}/{config}"] = result
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n")

    # human-readable summary
    print("\n=== stage breakdown ===")
    print(f"{'stage':<20} {'seconds':>10} {'% of search':>12} {'calls':>10}")
    for name in STAGE_ORDER + ["search_total"]:
        if name == "search_total":
            val, pct, n = t_search, 100.0, ""
        else:
            val = stages[name]
            pct = 100.0 * val / t_search if t_search else 0.0
            n = T[name].n
        print(f"{name:<20} {val:>10.3f} {pct:>11.1f}% {n if n != '' else '':>10}")
    print(json.dumps({k: result[k] for k in ("n_guides_input", "n_rows_out", "genome_load_s")}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--config", default="benchmark", choices=("benchmark", "pipeline"))
    ap.add_argument("--limit-guides", type=int, default=0, help="smoke-test: only first N guides")
    args = ap.parse_args()

    install_patches()
    run(args.family, args.config, args.limit_guides)


if __name__ == "__main__":
    main()
