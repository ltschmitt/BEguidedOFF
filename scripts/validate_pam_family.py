"""Validate beguidedoff against Cas-OFFinder outputs grouped by PAM family.

guide_id is internal-only (auto-generated); output is keyed by guide_seq.
"""
import sys
from pathlib import Path
from collections import defaultdict

from beguidedoff import beguidedoff
from beguidedoff.io import load_guides_from_tsv

BASE = Path(__file__).parent.parent / "relevant_files" / "cas-offinder"
INPUT_DIR = Path(__file__).parent.parent / "relevant_files"


def parse_cas_file(path):
    """Parse Cas-OFFinder output into dict: guide_seq -> set of (chrom, pos, strand, mm)."""
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
            strand = parts[4].upper()
            mm = int(parts[5])
            results[guide_seq].add((chrom, pos, strand, mm))
    return results


def validate_family(guides, cas_path, name):
    """Run beguidedoff for a PAM family and validate against Cas-OFFinder."""
    if not guides:
        print(f"No {name} guides to validate")
        return

    seeker = beguidedoff(
        "hg38_otbe.npz",
        max_mismatches=5,
        n_top_offtargets=50000,
        device="cuda:0",
    )

    cas_data = parse_cas_file(cas_path)

    # Run beguidedoff for ALL guides in this family at once
    import time
    t0 = time.time()
    df = seeker.search(guides=guides, return_all=True, include_non_editable=True, verbose=False)
    t1 = time.time()

    print(f"\n{'='*60}")
    print(f"PAM Family: {name}")
    print(f"Guides: {len(guides)}")
    print(f"Search time: {t1-t0:.2f}s")

    # Group beguidedoff results by guide_seq (guide_id is internal-only now)
    otb_by_guide = defaultdict(set)
    for _, row in df.iterrows():
        gseq = row["guide_seq"]
        otb_by_guide[gseq].add(
            (str(row["chromosome"]), int(row["position"]), str(row["strand"]).upper(), int(row["n_mismatches"]))
        )

    total_overlap = 0
    total_cas = 0
    total_otb = 0
    mismatched_guides = 0

    for g in guides:
        gseq = g.guide_seq
        otb_set = otb_by_guide.get(gseq, set())
        cas_set = cas_data.get(gseq, set())

        overlap = otb_set & cas_set
        cas_only = cas_set - otb_set
        otb_only = otb_set - cas_set

        total_overlap += len(overlap)
        total_cas += len(cas_set)
        total_otb += len(otb_set)

        if overlap != cas_set or overlap != otb_set:
            mismatched_guides += 1
            if len(cas_only) > 0 or len(otb_only) > 0:
                print(f"  {gseq}: overlap={len(overlap)}, cas_only={len(cas_only)}, otb_only={len(otb_only)}")
                if cas_only:
                    print(f"    sample cas_only: {sorted(list(cas_only))[:2]}")
                if otb_only:
                    print(f"    sample otb_only: {sorted(list(otb_only))[:2]}")

    print(f"\n{name} Summary:")
    print(f"  Total Cas hits: {total_cas}")
    print(f"  Total OTB hits: {total_otb}")
    print(f"  Overlap: {total_overlap}")
    print(f"  Cas-only: {total_cas - total_overlap}")
    print(f"  OTB-only: {total_otb - total_overlap}")
    print(f"  Guides with differences: {mismatched_guides} / {len(guides)}")
    print(f"{'='*60}\n")


def main():
    guides = load_guides_from_tsv(
        INPUT_DIR / "input_data_example.tsv",
        presets_path=Path(__file__).parent.parent / "be_presets.tsv",
    )

    ngg_guides = [g for g in guides if g.pam == "NGG"]
    nngrrt_guides = [g for g in guides if g.pam == "NNGRRT"]
    nnnrrt_guides = [g for g in guides if g.pam == "NNNRRT"]

    validate_family(ngg_guides, BASE / "casoffoutput_NGG.txt", "NGG")
    validate_family(nngrrt_guides, BASE / "casoffoutput_NNGRRT.txt", "NNGRRT")
    validate_family(nnnrrt_guides, BASE / "casoffoutput_NNNRRT.txt", "NNNRRT")


if __name__ == "__main__":
    main()
