"""Final validation of beguidedoff against Cas-OFFinder with all fixes.

- Deduplicated scans
- 0-mm self-targets excluded only at on-target location
- 1-based positions
- Buffer-overflow warnings
- Per-guide edit-window evaluation
- guide_id is internal-only (auto-generated); output keyed by guide_seq
"""
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

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
            pos = int(parts[2])  # Cas-OFFinder is 0-based
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

    t0 = time.time()
    df = seeker.search(guides=guides, return_all=True, include_non_editable=True, verbose=False)
    t1 = time.time()

    print(f"\n{'='*60}")
    print(f"PAM Family: {name}")
    print(f"Input rows: {len(guides)}  |  Scan keys (deduped): {len(seeker._deduplicate_guides(guides)[0])}")
    print(f"Search time: {t1-t0:.2f}s")

    # Group beguidedoff results by guide_seq (guide_id is internal-only now)
    otb_by_guide = defaultdict(set)
    for _, row in df.iterrows():
        gseq = row["guide_seq"]
        # Convert OTBE 1-based -> 0-based for comparison with Cas-OFFinder
        pos_0based = int(row["position"]) - 1
        otb_by_guide[gseq].add(
            (str(row["chromosome"]), pos_0based, str(row["strand"]).upper(), int(row["n_mismatches"]))
        )

    total_overlap = 0
    total_cas = 0
    total_otb = 0
    mismatched_guides = 0
    diff_samples = []

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

        if cas_only or otb_only:
            mismatched_guides += 1
            if len(diff_samples) < 5:
                diff_samples.append((gseq, len(overlap), len(cas_only), len(otb_only), list(cas_only)[:2], list(otb_only)[:2]))

    print(f"Summary:")
    print(f"  Total Cas hits: {total_cas}")
    print(f"  Total OTB hits: {total_otb}")
    print(f"  Overlap:        {total_overlap}")
    print(f"  Cas-only:       {total_cas - total_overlap}")
    print(f"  OTB-only:       {total_otb - total_overlap}")
    print(f"  Guides diffs:   {mismatched_guides} / {len(guides)}")

    for gseq, ov, co, oo, cs, os in diff_samples:
        print(f"  Diff example: {gseq}  ov={ov} cas_only={co} otb_only={oo}")
        if cs:
            print(f"    cas_only sample: {cs}")
        if os:
            print(f"    otb_only sample: {os}")

    print(f"{'='*60}\n")
    return t1 - t0


def main():
    guides = load_guides_from_tsv(
        INPUT_DIR / "input_example.tsv",
        presets_path=Path(__file__).parent.parent / "base_editor_profiles.tsv",
    )

    ngg_guides = [g for g in guides if g.pam == "NGG"]
    nngrrt_guides = [g for g in guides if g.pam == "NNGRRT"]
    nnnrrt_guides = [g for g in guides if g.pam == "NNNRRT"]

    t_ngg = validate_family(ngg_guides, BASE / "casoffoutput_NGG.txt", "NGG")
    t_nngrrt = validate_family(nngrrt_guides, BASE / "casoffoutput_NNGRRT.txt", "NNGRRT")
    t_nnnrrt = validate_family(nnnrrt_guides, BASE / "casoffoutput_NNNRRT.txt", "NNNRRT")

    total_otb = t_ngg + t_nngrrt + t_nnnrrt
    print(f"TOTAL beguidedoff GPU time: {total_otb:.2f}s")
    print()
    print("Reference Cas-OFFinder GPU times:")
    print("  NGG:     ~337s")
    print("  NNGRRT:  ~39s")
    print("  NNNRRT:  ~247s")
    print("  TOTAL:   ~623s")
    print()
    speedup = 623 / total_otb if total_otb > 0 else float('inf')
    print(f"Speedup vs Cas-OFFinder: {speedup:.2f}x")


if __name__ == "__main__":
    main()
