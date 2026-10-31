#!/usr/bin/env python
"""Concordance summary for the Cas-OFFinder v2.4.1 re-run (post overflow-fix).

Compares, per PAM family, the hit sets of
* Cas-OFFinder rerun output      (casoffoutput_<FAM>_rerun.txt)
* BEguidedOFF extensive rerun    (beguidedoffoutput_<FAM>_rerun.tsv)

Position conventions differ, so both sides are normalised to Cas-OFFinder's
native key -- 0-based leftmost base of the genomic guide+PAM window:
* Cas-OFFinder reports that leftmost position directly (verified empirically
  on + and - strand hits of the rerun outputs).
* BEguidedOFF reports the 1-based 5' end of the guide on the reported strand,
  i.e. leftmost+1 (+ strand) / leftmost+window_len (- strand).

Key per hit: (guide20, chromosome, window_leftmost_0based, strand, n_mismatches).
PAM mismatches are ignored by both tools (mismatches counted over the 20-nt
guide region only). Differences are checked against the reference FASTA for
N bases (the known cause of the single NGG discrepancy).

Also verifies the rerun Cas-OFFinder outputs are content-identical to the
May-2026 originals (repointed FASTA describes the same genome).

Usage:
  pixi run python benchmarks/compare_rerun.py
"""
import json
from collections import defaultdict
from pathlib import Path

BENCH = Path(__file__).resolve().parent    # this dir: benchmarks/ (scripts + reports)
NOGIT = BENCH.parent / "nogit"             # local heavy data (not in the repository)
BASE = NOGIT / "relevant_files" / "cas-offinder"
FAMILIES = ("NGG", "NNGRRT", "NNNRRT")
GUIDE_LEN = 20

# whole-process wall times (bash `time`), measured 2026-10-25 on the RTX 3060
CAS_WALL_S = {"NGG": 140.70, "NNGRRT": 20.60, "NNNRRT": 67.64}


def parse_cas(path):
    """guide -> set of (chrom, window_leftmost_0based, strand, mm)."""
    d = defaultdict(set)
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 6:
                continue
            d[parts[0][:GUIDE_LEN]].add(
                (parts[1].split()[0], int(parts[2]), parts[4], int(parts[5]))
            )
    return d


def parse_otb(path):
    """guide -> set of (chrom, window_leftmost_0based, strand, mm), normalised."""
    d = defaultdict(set)
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            guide = parts[idx["guide_rna_sequence"]]
            chrom = parts[idx["ot_chromosome"]]
            pos = int(parts[idx["ot_position"]])
            strand = parts[idx["ot_strand"]]
            mm = int(parts[idx["n_mismatches"]])
            pam_len = len(parts[idx["ot_pam"]])
            window_len = GUIDE_LEN + pam_len
            leftmost = pos - 1 if strand == "+" else pos - window_len
            d[guide].add((chrom, leftmost, strand, mm))
    return d


def has_n_in_window(fasta, chrom, leftmost, window_len):
    """True if the genomic window contains an N (case-insensitive)."""
    seq = fasta.fetch(chrom, max(leftmost, 0), leftmost + window_len).upper()
    return "N" in seq


def main():
    import pysam

    report = {"families": {}, "cas_offinder_wall_s": CAS_WALL_S}
    tot = defaultdict(int)
    for fam in FAMILIES:
        cas = parse_cas(BASE / f"casoffoutput_{fam}_rerun.txt")
        otb = parse_otb(BASE / f"beguidedoffoutput_{fam}_rerun.tsv")

        # genome identity check: rerun vs May-2026 Cas-OFFinder output
        with open(BASE / f"casoffoutput_{fam}.txt") as a, open(BASE / f"casoffoutput_{fam}_rerun.txt") as b:
            same = sorted(a.readlines()) == sorted(b.readlines())

        overlap = cas_only_n = otb_only_n = 0
        cas_only_examples, otb_only_examples = [], []
        n_with_n = 0
        window_len_fam = None
        for guide in set(cas) | set(otb):
            c = cas.get(guide, set())
            o = otb.get(guide, set())
            overlap += len(c & o)
            cas_only_n += len(c - o)
            otb_only_n += len(o - c)
            for hit in list(c - o)[:3]:
                cas_only_examples.append((guide,) + hit)
            for hit in list(o - c)[:10]:
                otb_only_examples.append((guide,) + hit)

        # check the differing loci for N bases in the reference
        fasta = pysam.FastaFile(str(NOGIT / "GRCh38.115" / "Homo_sapiens.GRCh38.dna.primary_assembly.fa"))
        for guide, chrom, leftmost, strand, mm in otb_only_examples[:10]:
            if window_len_fam is None:
                # window length from any hit of this guide (pam via re-derivation not
                # needed: diff count is what matters; use +strand convention probe)
                window_len_fam = 20 + len(fam) if fam != "NGG" else 23
            try:
                if has_n_in_window(fasta, chrom, leftmost, window_len_fam):
                    n_with_n += 1
            except (KeyError, ValueError):
                pass
        fasta.close()

        n_cas = sum(len(v) for v in cas.values())
        n_otb = sum(len(v) for v in otb.values())
        tot["cas"] += n_cas; tot["otb"] += n_otb; tot["overlap"] += overlap
        tot["cas_only"] += cas_only_n; tot["otb_only"] += otb_only_n

        beguided_json = json.loads((BASE / f"beguidedofftiming_{fam}_rerun.json").read_text())
        report["families"][fam] = {
            "cas_offinder_hits": n_cas,
            "beguidedoff_hits": n_otb,
            "overlap": overlap,
            "cas_only": cas_only_n,
            "beguidedoff_only": otb_only_n,
            "concordance_pct": round(100.0 * overlap / max(n_cas, 1), 4),
            "rerun_output_identical_to_may2026": same,
            "differing_loci_checked": len(otb_only_examples[:10]),
            "differing_loci_with_N_in_reference": n_with_n,
            "cas_only_examples": cas_only_examples[:5],
            "beguidedoff_only_examples": otb_only_examples[:5],
            "beguidedoff_total_s": beguided_json["total_s"],
            "beguidedoff_search_s": beguided_json["search_s"],
        }

    report["totals"] = {
        "cas_offinder_hits": tot["cas"],
        "beguidedoff_hits": tot["otb"],
        "overlap": tot["overlap"],
        "cas_only": tot["cas_only"],
        "beguidedoff_only": tot["otb_only"],
        "concordance_pct": round(100.0 * tot["overlap"] / max(tot["cas"], 1), 4),
        "cas_offinder_wall_s_total": round(sum(CAS_WALL_S.values()), 2),
        "beguidedoff_wall_s_total": round(sum(report["families"][f]["beguidedoff_total_s"] for f in FAMILIES), 2),
    }
    out = BENCH / "casoffinder_rerun_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
