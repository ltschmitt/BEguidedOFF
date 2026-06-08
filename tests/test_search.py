"""Tests for search engine."""
import tempfile
from pathlib import Path
from beguidedoff import beguidedoff
from beguidedoff.types import GuideInput
from beguidedoff.genome_converter import convert
from beguidedoff.encoding import reverse_complement_seq


def _make_fasta(path: str, entries: dict):
    with open(path, "w") as f:
        for name, seq in entries.items():
            f.write(f">{name}\n{seq}\n")


def test_exact_match_excluded():
    """On-target (0 mismatches) should not be reported."""
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        npz = Path(td) / "genome.npz"
        # Simple genome with one NGG PAM site
        seq = "ACGTACGTACGTACGTACGT" + "GG" + "A" * 100
        _make_fasta(fa, {"chr1": seq})
        convert(str(fa), str(npz), verbose=False)

        finder = beguidedoff(
            genome_path=str(npz),
            max_mismatches=4,
            scoring_method="CCTOP",
            device="cpu",
            n_top_offtargets=3,
        )
        guides = [
            GuideInput("g1", "ACGTACGTACGTACGTACGT", "NGG", "3prime", 4, 8, "CBE"),
        ]
        df = finder.search(guides=guides, verbose=False)
        # The exact match should be excluded
        assert len(df) == 0


def test_mismatched_ontarget_excluded():
    """On-target position with mismatches should also be excluded."""
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        npz = Path(td) / "genome.npz"
        # Genome: guide starts at pos 0 (1-based pos=1), but has 2 mismatches vs guide
        # Guide: ACGTACGTACGTACGTACGT
        # Genome: ACGTACGAACGTACGTACGA (mismatches at pos 7 and 20)
        # Wait, need 20 bp + PAM. Let's make mismatch at pos 7 and keep PAM as GG.
        # So genome segment = ACGTACGAACGTACGTACGT + GG
        ontarget = "ACGTACGAACGTACGTACGT" + "GG"
        # Also add a 1-mismatch off-target further down
        offtarget = "ACGTACGTACGTACGTACGA" + "AGG"
        seq = ontarget + "A" * 30 + offtarget + "A" * 100
        _make_fasta(fa, {"chr1": seq})
        convert(str(fa), str(npz), verbose=False)

        finder = beguidedoff(
            genome_path=str(npz),
            max_mismatches=4,
            scoring_method="CCTOP",
            device="cpu",
            n_top_offtargets=3,
        )
        guides = [
            GuideInput(
                "g1",
                "ACGTACGTACGTACGTACGT",
                "NGG",
                "3prime",
                4,
                8,
                "CBE",
                on_target_chrom="chr1",
                on_target_pos=1,
                on_target_strand="+",
            ),
        ]
        df = finder.search(guides=guides, verbose=False)
        # The mismatched on-target at position 1 must be excluded;
        # only the 1-mismatch off-target at position 53 (1-based) should remain.
        assert len(df) == 1
        assert df.iloc[0]["n_mismatches"] == 1
        assert df.iloc[0]["ot_position"] == 53


def test_minus_strand_offtarget_position():
    """Minus strand off-target should report the rightmost position."""
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        npz = Path(td) / "genome.npz"
        guide = "ACGTACGTACGTACGTACGT"
        pam = "AGG"
        rc_target = reverse_complement_seq(guide + pam)
        genome = "A" * 14 + rc_target + "A" * 23 + rc_target + "A" * 17  # total 100
        _make_fasta(fa, {"chr1": genome})
        convert(str(fa), str(npz), verbose=False)

        finder = beguidedoff(
            genome_path=str(npz),
            max_mismatches=4,
            scoring_method="CCTOP",
            device="cpu",
            n_top_offtargets=3,
        )
        guides = [
            GuideInput("g1", guide, "NRG", "3prime", 4, 8, "CBE"),
        ]
        df = finder.search(guides=guides, verbose=False)
        # Two identical minus-strand hits at positions 37 and 83
        assert len(df) == 2
        positions = sorted(df["ot_position"].tolist())
        assert positions == [37, 83]
        assert df["ot_strand"].nunique() == 1
        assert df.iloc[0]["ot_strand"] == "-"


def test_minus_strand_ontarget_excluded_correct_position():
    """Minus strand on-target excluded with correct rightmost position and strand."""
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        npz = Path(td) / "genome.npz"
        guide = "ACGTACGTACGTACGTACGT"
        pam = "AGG"
        rc_target = reverse_complement_seq(guide + pam)
        genome = "A" * 14 + rc_target + "A" * 23 + rc_target + "A" * 17  # total 100
        _make_fasta(fa, {"chr1": genome})
        convert(str(fa), str(npz), verbose=False)

        finder = beguidedoff(
            genome_path=str(npz),
            max_mismatches=4,
            scoring_method="CCTOP",
            device="cpu",
            n_top_offtargets=3,
        )
        guides = [
            GuideInput(
                "g1",
                guide,
                "NRG",
                "3prime",
                4,
                8,
                "CBE",
                on_target_chrom="chr1",
                on_target_pos=37,   # rightmost 1-based position of minus strand on-target
                on_target_strand="-",
            ),
        ]
        df = finder.search(guides=guides, verbose=False)
        # The minus-strand on-target should be excluded; only the second hit remains
        assert len(df) == 1
        assert df.iloc[0]["ot_position"] == 83
        assert df.iloc[0]["ot_strand"] == "-"


def test_minus_strand_ontarget_not_excluded_wrong_strand():
    """Minus strand on-target NOT excluded when strand is wrong — proves strand check works."""
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        npz = Path(td) / "genome.npz"
        guide = "ACGTACGTACGTACGTACGT"
        pam = "AGG"
        rc_target = reverse_complement_seq(guide + pam)
        genome = "A" * 14 + rc_target + "A" * 23 + rc_target + "A" * 17  # total 100
        _make_fasta(fa, {"chr1": genome})
        convert(str(fa), str(npz), verbose=False)

        finder = beguidedoff(
            genome_path=str(npz),
            max_mismatches=4,
            scoring_method="CCTOP",
            device="cpu",
            n_top_offtargets=3,
        )
        guides = [
            GuideInput(
                "g1",
                guide,
                "NRG",
                "3prime",
                4,
                8,
                "CBE",
                on_target_chrom="chr1",
                on_target_pos=37,
                on_target_strand="+",  # WRONG strand — should NOT exclude anything
            ),
        ]
        df = finder.search(guides=guides, verbose=False)
        # Both minus-strand hits should still be returned
        assert len(df) == 2


def test_ontarget_not_excluded_shifted_position():
    """On-target NOT excluded when position is shifted by 1 bp — proves position conversion works."""
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        npz = Path(td) / "genome.npz"
        guide = "ACGTACGTACGTACGTACGT"
        pam = "AGG"
        rc_target = reverse_complement_seq(guide + pam)
        genome = "A" * 14 + rc_target + "A" * 23 + rc_target + "A" * 17  # total 100
        _make_fasta(fa, {"chr1": genome})
        convert(str(fa), str(npz), verbose=False)

        finder = beguidedoff(
            genome_path=str(npz),
            max_mismatches=4,
            scoring_method="CCTOP",
            device="cpu",
            n_top_offtargets=3,
        )
        # Provide shifted rightmost position (36 instead of 37)
        guides = [
            GuideInput(
                "g1",
                guide,
                "NRG",
                "3prime",
                4,
                8,
                "CBE",
                on_target_chrom="chr1",
                on_target_pos=36,   # shifted by 1 — should NOT exclude anything
                on_target_strand="-",
            ),
        ]
        df = finder.search(guides=guides, verbose=False)
        # Both minus-strand hits should still be returned
        assert len(df) == 2


def test_edit_window_filter():
    """Off-target without editable base in window is excluded."""
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        npz = Path(td) / "genome.npz"
        # Guide: ACGTACGTACGTACGTACGT
        # Off-target with 1 mismatch but no C in window 4-8
        # Window positions 4-8 = TACGT -> contains no C
        ot = "ACGTACGAACGTACGTACGT" + "GG"
        # Put the off-target in the genome, plus other filler
        seq = ot + "A" * 100
        _make_fasta(fa, {"chr1": seq})
        convert(str(fa), str(npz), verbose=False)

        finder = beguidedoff(
            genome_path=str(npz),
            max_mismatches=4,
            scoring_method="CCTOP",
            device="cpu",
            n_top_offtargets=3,
        )
        guides = [
            GuideInput("g1", "ACGTACGTACGTACGTACGT", "NGG", "3prime", 4, 8, "CBE"),
        ]
        df = finder.search(guides=guides, verbose=False)
        # Should find 0 because off-target has no editable C in window 4-8
        assert len(df) == 0
