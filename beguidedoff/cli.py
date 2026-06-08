"""CLI for BEguidedOFF."""
import argparse
import logging
import sys
from pathlib import Path

from beguidedoff import beguidedoff, genome_converter


def cmd_encode_genome(args):
    """Encode a FASTA genome to beguidedoff format."""
    genome_converter.convert(
        fasta_path=args.fasta,
        output_path=args.output,
        include_rc=args.include_rc,
        include_n_mask=args.include_n_mask,
        verbose=True,
    )


def cmd_search(args):
    """Run off-target search."""
    finder = beguidedoff(
        genome_path=args.genome,
        max_mismatches=args.max_mismatches,
        device=args.device,
        n_top_offtargets=args.n_top,
        results_per_guide=args.results_per_guide,
    )

    results = finder.search(
        input_file=args.guides,
        return_all=args.return_all,
        count_score_threshold=args.count_score_threshold,
        output_file=args.output,
        include_non_editable=args.include_non_editable,
        presets_path=args.presets_path,
        verbose=True,
    )

    if not args.output:
        print(results.to_string(index=False))


def cmd_verify(args):
    """Verify an encoded genome against original FASTA."""
    ok = genome_converter.verify(
        encoded_path=args.encoded,
        fasta_path=args.fasta,
        sample_n=args.sample_n,
        verbose=True,
    )
    sys.exit(0 if ok else 1)


def main():
    parser = argparse.ArgumentParser(
        prog="beguidedoff",
        description="beguidedoff — GPU-Accelerated Off-Target Base Editing Seeker",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # encode-genome
    enc = subparsers.add_parser("encode-genome", help="Encode a FASTA genome")
    enc.add_argument("--fasta", required=True, help="Input FASTA file")
    enc.add_argument("--output", required=True, help="Output .npz path")
    enc.add_argument("--include-rc", action="store_true", default=True, help="Include reverse complement")
    enc.add_argument("--include-n-mask", action="store_true", default=True, help="Include N-mask")
    enc.set_defaults(func=cmd_encode_genome)

    # search
    srch = subparsers.add_parser("search", help="Search for off-targets")
    srch.add_argument("--genome", required=True, help="Encoded genome .npz")
    srch.add_argument("--guides", required=True, help="Guide TSV input")
    srch.add_argument("--max-mismatches", type=int, default=4, help="Max mismatches")
    srch.add_argument("--scoring", choices=["cctop", "cfd"], default="cctop", help="Scoring method (default: CCTOP)")
    srch.add_argument("--device", default=None, help="torch device (cuda:0 or cpu)")
    srch.add_argument("--n-top", type=int, default=3, help="Top N off-targets per guide")
    srch.add_argument("--results-per-guide", type=int, default=10240, help="Max off-target hits stored per guide per chunk (default: 10240)")
    srch.add_argument("--return-all", action="store_true", help="Return all off-targets above threshold")
    srch.add_argument("--count-score-threshold", type=float, default=15.0, help="CCTOP score threshold for counting relevant off-targets (<= threshold, default: 15.0)")
    srch.add_argument("--output", default=None, help="Output TSV file")
    srch.add_argument("--include-non-editable", action="store_true", help="Include off-targets without editable bases (also implied by --return-all)")
    srch.add_argument("--presets-path", default=None, help="Path to be_presets.tsv (default: be_presets.tsv in app root)")
    srch.set_defaults(func=cmd_search)

    # verify
    ver = subparsers.add_parser("verify", help="Verify encoded genome")
    ver.add_argument("--encoded", required=True, help="Encoded .npz")
    ver.add_argument("--fasta", required=True, help="Original FASTA")
    ver.add_argument("--sample-n", type=int, default=10000, help="Number of samples")
    ver.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
