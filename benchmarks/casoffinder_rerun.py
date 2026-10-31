#!/usr/bin/env python
"""Re-run of the Cas-OFFinder v2.4.1 comparison (post overflow-fix extensive).

The original validation (May 2026) timed Cas-OFFinder vs BEguidedOFF on the
per-PAM-family guide batches in ``nogit/relevant_files/cas-offinder/`` and
compared hit sets at <=5 mismatches. Since then the extensive (GPU) engine
gained the automatic buffer-overflow re-scan, so the comparison is repeated
with identical inputs and methodology:

* ``setup``     -- write ``casoffinput_<FAM>_rerun.txt`` (FASTA path repointed
  to the current GRCh38.115 primary assembly; 112/115 are byte-identical, see
  benchmarks/REPORT.md) and the matching explicit-format BEguidedOFF input TSVs.
* ``beguided``  -- run + time BEguidedOFF extensive on one family batch
  (return_all + include_non_editable, i.e. the same enumeration space as
  Cas-OFFinder) and write the hit TSV + a timing JSON.

Cas-OFFinder itself is invoked from the shell with /usr/bin/time; the
concordance comparison lives in ``compare_rerun.py`` next to this script.

Usage:
  python casoffinder_rerun.py setup
  python casoffinder_rerun.py beguided --family NNGRRT
"""
import argparse
import json
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent    # this dir: benchmarks/ (scripts + reports)
NOGIT = BENCH.parent / "nogit"             # local heavy data (not in the repository)
REPO = BENCH.parent
sys.path.insert(0, str(REPO))

BASE = NOGIT / "relevant_files" / "cas-offinder"
FASTA = NOGIT / "GRCh38.115" / "Homo_sapiens.GRCh38.dna.primary_assembly.fa"
NPZ = NOGIT / "hg38_beguidedoff.npz"
FAMILIES = ("NGG", "NNGRRT", "NNNRRT")


def read_casoffinput(path):
    """Return (fasta_line, pam_pattern, [(target_token, max_mm), ...]).

    The target token keeps its full length (20 nt guide + N placeholder)
    because Cas-OFFinder requires target and pattern lengths to match;
    the BEguidedOFF TSV uses the first 20 nt only.
    """
    lines = path.read_text().strip().splitlines()
    return lines[0].strip(), lines[1].strip(), [(l.split()[0], int(l.split()[1])) for l in lines[2:] if l.strip()]


def cmd_setup():
    for fam in FAMILIES:
        fasta_line, pam, guides = read_casoffinput(BASE / f"casoffinput_{fam}.txt")
        # identical content, only the (stale) FASTA path is repointed
        rerun = BASE / f"casoffinput_{fam}_rerun.txt"
        rerun.write_text("\n".join([str(FASTA), pam] + [f"{g} {mm}" for g, mm in guides]) + "\n")
        # explicit-format BEguidedOFF input; window values are irrelevant for the
        # hit set because the search runs with include_non_editable=True
        tsv = BASE / f"beguidedoff_input_{fam}.tsv"
        rows = ["guide_seq\tpam\tedit_window_start\tedit_window_end\teditor_type"]
        rows += [f"{g[:20]}\t{fam}\t4\t8\tABE" for g, _ in guides]
        tsv.write_text("\n".join(rows) + "\n")
        print(f"{fam}: {len(guides)} guides -> {rerun.name}, {tsv.name}")


def cmd_beguided(family):
    from beguidedoff import beguidedoff

    tsv = BASE / f"beguidedoff_input_{family}.tsv"
    out = BASE / f"beguidedoffoutput_{family}_rerun.tsv"
    timing = BASE / f"beguidedofftiming_{family}_rerun.json"

    t0 = time.perf_counter()
    finder = beguidedoff(str(NPZ), max_mismatches=5, device="cuda:0")
    t_load = time.perf_counter() - t0

    df = finder.search(
        input_file=str(tsv),
        return_all=True,
        include_non_editable=True,
        verbose=True,
        output_file=str(out),
    )
    t_total = time.perf_counter() - t0
    stats = {
        "family": family,
        "n_guides": len(tsv.read_text().strip().splitlines()) - 1,
        "genome_load_s": round(t_load, 2),
        "total_s": round(t_total, 2),
        "search_s": round(t_total - t_load, 2),
        "n_hits": int(len(df)),
    }
    timing.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    run = sub.add_parser("beguided")
    run.add_argument("--family", required=True, choices=FAMILIES)
    args = ap.parse_args()
    if args.cmd == "setup":
        cmd_setup()
    else:
        cmd_beguided(args.family)


if __name__ == "__main__":
    main()
