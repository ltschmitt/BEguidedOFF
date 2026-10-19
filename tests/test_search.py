"""Tests for the fast (FM-index) search engine."""
import tempfile
from pathlib import Path

from _fastutil import make_fast_finder, write_fasta
from beguidedoff.encoding import reverse_complement_seq
from beguidedoff.types import GuideInput

_GUIDE = "ACGTACGTACGTACGTACGT"


def test_exact_ontarget_excluded():
    """The exact on-target site is excluded when on-target coordinates are given."""
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        write_fasta(str(fa), {"chr1": _GUIDE + "AGG" + "A" * 100})
        finder = make_fast_finder(str(fa), td, search_cutoff=5.0, n_top_offtargets=3)
        guides = [
            GuideInput("g1", _GUIDE, "NGG", "3prime", 4, 8, "CBE",
                       on_target_chrom="chr1", on_target_pos=1, on_target_strand="+"),
        ]
        df = finder.search(guides=guides, verbose=False)
        assert len(df) == 0


def test_mismatched_ontarget_excluded():
    """A mismatched on-target (same coordinates) is excluded too."""
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        ontarget = "ACGTACGTACGAACGTACGT" + "AGG"  # 1 mismatch at pos 12
        write_fasta(str(fa), {"chr1": ontarget + "A" * 100})
        finder = make_fast_finder(str(fa), td, search_cutoff=5.0, n_top_offtargets=3)
        guides = [
            GuideInput("g1", _GUIDE, "NGG", "3prime", 4, 8, "CBE",
                       on_target_chrom="chr1", on_target_pos=1, on_target_strand="+"),
        ]
        df = finder.search(guides=guides, verbose=False)
        assert len(df) == 0


def test_offtarget_found_with_correct_position_and_score():
    """A PAM-distal off-target is reported with the right position and score."""
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        ot = "ACGTACGTACGTACGTACGA"[:19] + "A"  # mismatch at pos 20?
        # mismatch at position 3 instead (PAM-distal, score 1.2**3)
        ot = _GUIDE[:2] + "A" + _GUIDE[3:] + "AGG"
        write_fasta(str(fa), {"chr1": "A" * 30 + ot + "A" * 100})
        finder = make_fast_finder(str(fa), td, search_cutoff=5.0, n_top_offtargets=3)
        guides = [GuideInput("g1", _GUIDE, "NGG", "3prime", 4, 8, "CBE")]
        df = finder.search(guides=guides, verbose=False)
        assert len(df) == 1
        assert df.iloc[0]["n_mismatches"] == 1
        assert df.iloc[0]["ot_position"] == 31
        assert abs(df.iloc[0]["score"] - 1.2 ** 3) < 1e-9


def test_minus_strand_offtarget_position():
    """Minus-strand off-targets report the rightmost position."""
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        rc_target = reverse_complement_seq(_GUIDE + "AGG")
        genome = "A" * 14 + rc_target + "A" * 23 + rc_target + "A" * 17  # total 100
        write_fasta(str(fa), {"chr1": genome})
        finder = make_fast_finder(str(fa), td, search_cutoff=5.0, n_top_offtargets=3)
        guides = [GuideInput("g1", _GUIDE, "NRG", "3prime", 4, 8, "CBE")]
        df = finder.search(guides=guides, verbose=False)
        assert len(df) == 2
        assert sorted(df["ot_position"].tolist()) == [37, 83]
        assert df["ot_strand"].nunique() == 1
        assert df.iloc[0]["ot_strand"] == "-"


def test_minus_strand_ontarget_excluded():
    """Minus-strand on-target (rightmost coordinate) is excluded."""
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        rc_target = reverse_complement_seq(_GUIDE + "AGG")
        genome = "A" * 14 + rc_target + "A" * 23 + rc_target + "A" * 17
        write_fasta(str(fa), {"chr1": genome})
        finder = make_fast_finder(str(fa), td, search_cutoff=5.0, n_top_offtargets=3)
        guides = [
            GuideInput("g1", _GUIDE, "NRG", "3prime", 4, 8, "CBE",
                       on_target_chrom="chr1", on_target_pos=37, on_target_strand="-"),
        ]
        df = finder.search(guides=guides, verbose=False)
        assert len(df) == 1
        assert df.iloc[0]["ot_position"] == 83
        assert df.iloc[0]["ot_strand"] == "-"


def test_minus_strand_ontarget_wrong_strand_not_excluded():
    """A wrong-strand on-target coordinate must not exclude anything."""
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        rc_target = reverse_complement_seq(_GUIDE + "AGG")
        genome = "A" * 14 + rc_target + "A" * 23 + rc_target + "A" * 17
        write_fasta(str(fa), {"chr1": genome})
        finder = make_fast_finder(str(fa), td, search_cutoff=5.0, n_top_offtargets=3)
        guides = [
            GuideInput("g1", _GUIDE, "NRG", "3prime", 4, 8, "CBE",
                       on_target_chrom="chr1", on_target_pos=37, on_target_strand="+"),
        ]
        df = finder.search(guides=guides, verbose=False)
        assert len(df) == 2


def test_edit_window_filter():
    """A site without an editable base in the window is not reported."""
    guide = "GGGACGTACGTACGTACGTA"  # window 1-3 = GGG -> no C for CBE
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        mutated = guide[:4] + "T" + guide[5:]  # PAM-distal mismatch (pos 5), no C added
        write_fasta(str(fa), {"chr1": mutated + "AGG" + "A" * 100})
        finder = make_fast_finder(str(fa), td, search_cutoff=5.0, n_top_offtargets=3)
        guides = [GuideInput("g1", guide, "NGG", "3prime", 1, 3, "CBE")]
        df = finder.search(guides=guides, verbose=False)
        assert len(df) == 0
