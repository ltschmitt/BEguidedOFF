"""Compare beguidedoff output with Cas-OFFinder reference outputs.

guide_id is internal-only (auto-generated); output keyed by guide_seq.

Usage:
    python compare_with_casoffinder.py
"""
import sys
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent.parent / "relevant_files" / "cas-offinder"


def parse_cas_output(path):
    """Parse Cas-OFFinder output into dict: guide_seq -> set of (chrom,pos,strand,mm)."""
    results = defaultdict(set)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 6:
                continue
            guide_seq = parts[0][:20]
            chrom = parts[1].split()[0]
            pos = int(parts[2])
            strand = parts[4]
            mm = int(parts[5])
            results[guide_seq].add((chrom, pos, strand, mm))
    return results


def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from beguidedoff import beguidedoff
    from beguidedoff.io import load_guides_from_tsv

    guides = load_guides_from_tsv(
        "relevant_files/input_example.tsv",
        presets_path="base_editor_profiles.tsv",
    )

    # Cas-OFFinder outputs
    ngg_cas = parse_cas_output(BASE / "casoffoutput_NGG.txt")
    nngrrt_cas = parse_cas_output(BASE / "casoffoutput_NNGRRT.txt")
    nnnrrt_cas = parse_cas_output(BASE / "casoffoutput_NNNRRT.txt")

    # Compare guide-by-guide for overlapping guides
    seeker = beguidedoff(
        "hg38_otbe.npz",
        max_mismatches=5,
        n_top_offtargets=20000,
        device="cuda:0",
    )

    total_overlap = 0
    total_cas_only = 0
    total_otb_only = 0

    for pam_name, cas_dict in [
        ("NGG", ngg_cas),
        ("NNGRRT", nngrrt_cas),
        ("NNNRRT", nnnrrt_cas),
    ]:
        for guide_seq in sorted(cas_dict):
            # Find matching guide(s) in input data
            otbs = [g for g in guides if g.guide_seq == guide_seq and g.pam == pam_name]
            if not otbs:
                continue
            for guide in otbs:
                print(f"\n--- Guide: {guide.guide_seq} | PAM={guide.pam} ---")
                # Run beguidedoff for this single guide
                df = seeker.search(
                    guides=[guide],
                    return_all=True,
                    include_non_editable=True,
                    verbose=False,
                )
                otb_set = {
                    (str(row["chromosome"]), int(row["position"]), row["strand"], int(row["n_mismatches"]))
                    for _, row in df.iterrows()
                }
                cas_set = cas_dict.get(guide_seq, set())
                overlap = otb_set & cas_set
                cas_only = cas_set - otb_set
                otb_only = otb_set - cas_set
                print(f"  beguidedoff hits: {len(otb_set)}  Cas-OFFinder hits: {len(cas_set)}")
                print(f"  Overlap: {len(overlap)}  Cas-only: {len(cas_only)}  OTB-only: {len(otb_only)}")
                if cas_only:
                    print(f"  Sample Cas-only: {sorted(list(cas_only))[:3]}")
                if otb_only:
                    # Filter to show at most 3
                    sample = sorted(list(otb_only))[:3]
                    print(f"  Sample OTB-only: {sample}")
                total_overlap += len(overlap)
                total_cas_only += len(cas_only)
                total_otb_only += len(otb_only)
                break  # only test first matching guide

    print(f"\n===== TOTALS =====")
    print(f"Overlap:   {total_overlap}")
    print(f"Cas-only:  {total_cas_only}")
    print(f"OTB-only:  {total_otb_only}")


if __name__ == "__main__":
    main()
