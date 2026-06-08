"""Tests for count_score_threshold and n_offtargets_below_threshold."""
import tempfile
from pathlib import Path
from beguidedoff import beguidedoff
from beguidedoff.types import GuideInput
from beguidedoff.genome_converter import convert


def _make_fasta(path: str, entries: dict):
    with open(path, "w") as f:
        for name, seq in entries.items():
            f.write(f">{name}\n{seq}\n")


def _setup_genome_with_offtargets(tmpdir: str) -> tuple:
    """Create a genome and npz with several off-targets for one guide.

    Guide: ACGTACGTACGTACGTACGT
    On-target at pos 1: ACGTACGTACGTACGTACGT + AGG
    Off-target A (1 mm pos 1):  TCGTACGTACGTACGTACGT + AGG   score=1.2
    Off-target B (1 mm pos 10): ACGTACGTACAGTACGTACGT + AGG  score=~6.19
    Off-target C (1 mm pos 20): ACGTACGTACGTACGTACGA  + AGG   score=~38.34
    All have an editable C in window 4-8 (pos 6 = C).
    """
    fa = Path(tmpdir) / "genome.fa"
    npz = Path(tmpdir) / "genome.npz"

    guide = "ACGTACGTACGTACGTACGT"
    pam = "AGG"

    ot_a = "TCGTACGTACGTACGTACGT" + pam  # mismatch at pos 1
    ot_b = "ACGTACGTACAGTACGTACGT" + pam  # mismatch at pos 10 (20bp) -- wait

    # Let's recalculate. Guide is 20bp.
    # pos 1: A -> T for ot_a
    # pos 10: C -> A for ot_b
    # pos 20: T -> A for ot_c
    ot_b = guide[:9] + "A" + guide[10:] + pam
    ot_c = guide[:19] + "A" + pam

    # Place on-target at position 1 (1-based), then off-targets spaced out
    seq = (
        guide + pam +  # on-target at 1-23
        "A" * 30 +
        ot_a + "A" * 30 +
        ot_b + "A" * 30 +
        ot_c + "A" * 100
    )

    _make_fasta(fa, {"chr1": seq})
    convert(str(fa), str(npz), verbose=False)
    return str(npz)


def test_count_score_threshold_basic():
    """n_offtargets_below_threshold reflects only editable, score-thresholded hits."""
    with tempfile.TemporaryDirectory() as td:
        npz = _setup_genome_with_offtargets(td)

        finder = beguidedoff(
            genome_path=npz,
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

        # threshold 5.0 -> only ot_a (score 1.2)
        df = finder.search(
            guides=guides,
            count_score_threshold=5.0,
            verbose=False,
        )
        assert len(df) == 3  # 3 off-targets returned, top-3
        assert "n_offtargets_below_threshold" in df.columns
        assert df["n_offtargets_below_threshold"].iloc[0] == 1
        assert df["n_offtargets_below_threshold"].nunique() == 1


def test_count_score_threshold_all_below():
    """When threshold is high enough, all editable off-targets are counted."""
    with tempfile.TemporaryDirectory() as td:
        npz = _setup_genome_with_offtargets(td)

        finder = beguidedoff(
            genome_path=npz,
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

        # threshold 50.0 -> ot_a, ot_b, ot_c all below threshold
        df = finder.search(
            guides=guides,
            count_score_threshold=50.0,
            verbose=False,
        )
        assert len(df) == 3
        assert df["n_offtargets_below_threshold"].iloc[0] == 3


def test_count_score_threshold_none_below():
    """When threshold is very low, count is zero even though some hits are returned."""
    with tempfile.TemporaryDirectory() as td:
        npz = _setup_genome_with_offtargets(td)

        finder = beguidedoff(
            genome_path=npz,
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

        # threshold 0.5 -> no off-target scores <= 0.5
        df = finder.search(
            guides=guides,
            count_score_threshold=0.5,
            verbose=False,
        )
        assert len(df) == 3
        assert df["n_offtargets_below_threshold"].iloc[0] == 0


def test_count_score_threshold_unaffected_by_n_top():
    """The count must include all off-targets below threshold, even if n_top slices them."""
    with tempfile.TemporaryDirectory() as td:
        npz = _setup_genome_with_offtargets(td)

        finder = beguidedoff(
            genome_path=npz,
            max_mismatches=4,
            scoring_method="CCTOP",
            device="cpu",
            n_top_offtargets=1,  # only top 1 returned
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

        # threshold 10.0 -> ot_a (1.2) and ot_b (~6.19) are below; ot_c (~38) is above
        # Only ot_a is returned because n_top=1, but count should be 2
        df = finder.search(
            guides=guides,
            count_score_threshold=10.0,
            verbose=False,
        )
        assert len(df) == 1
        assert df["n_offtargets_below_threshold"].iloc[0] == 2


def test_count_score_threshold_omitted_when_none():
    """If no off-targets are found at all, the DataFrame is empty."""
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        npz = Path(td) / "genome.npz"
        # Genome with no NGG PAM matches for the guide
        seq = "A" * 300
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
        df = finder.search(
            guides=guides,
            count_score_threshold=10.0,
            verbose=False,
        )
        assert df.empty
