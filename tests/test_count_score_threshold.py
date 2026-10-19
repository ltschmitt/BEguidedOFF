"""Tests for count_score_threshold / n_offtargets_below_threshold (fast mode)."""
import tempfile
from pathlib import Path

from _fastutil import make_fast_finder, write_fasta
from beguidedoff.types import GuideInput

_GUIDE = "ACGTACGTACGTACGTACGT"
_PAM = "NGG"


def _mutate(seq: str, pos: int) -> str:
    """Return *seq* with a non-identity base at 1-indexed *pos*."""
    alt = {"A": "C", "C": "A", "G": "T", "T": "G"}[seq[pos - 1]]
    return seq[: pos - 1] + alt + seq[pos:]


def _setup_genome(tmpdir: str) -> str:
    """Genome with an on-target and three PAM-distal editable off-targets.

    Guide: ACGTACGTACGTACGTACGT (editable C at position 6, window 4-8).
      ot_a: mismatch at pos 1 -> score 1.2
      ot_b: mismatch at pos 4 -> score 1.2**4  = 2.0736
      ot_c: mismatch at pos 8 -> score 1.2**8  = 4.2998
    Positions 4 and 8 are mutated but the editable C at position 6 is kept.
    """
    fa = Path(tmpdir) / "genome.fa"
    ot_a = _mutate(_GUIDE, 1) + "AGG"
    ot_b = _mutate(_GUIDE, 4) + "AGG"
    ot_c = _mutate(_GUIDE, 8) + "AGG"
    seq = (
        _GUIDE + "AGG"                    # on-target at 1-23
        + "A" * 30 + ot_a
        + "A" * 30 + ot_b
        + "A" * 30 + ot_c
        + "A" * 100
    )
    write_fasta(str(fa), {"chr1": seq})
    return str(fa)


def _guide():
    return GuideInput(
        "g1", _GUIDE, _PAM, "3prime", 4, 8, "CBE",
        on_target_chrom="chr1", on_target_pos=1, on_target_strand="+",
    )


def test_count_below_high_threshold():
    with tempfile.TemporaryDirectory() as td:
        finder = make_fast_finder(_setup_genome(td), td, search_cutoff=5.0, n_top_offtargets=3)
        df = finder.search(guides=[_guide()], count_score_threshold=5.0, verbose=False)
        assert len(df) == 3
        assert df["n_offtargets_below_threshold"].iloc[0] == 3
        assert df["n_offtargets_below_threshold"].nunique() == 1


def test_count_below_partial_threshold():
    with tempfile.TemporaryDirectory() as td:
        finder = make_fast_finder(_setup_genome(td), td, search_cutoff=5.0, n_top_offtargets=3)
        df = finder.search(guides=[_guide()], count_score_threshold=3.0, verbose=False)
        assert df["n_offtargets_below_threshold"].iloc[0] == 2


def test_count_below_low_threshold():
    with tempfile.TemporaryDirectory() as td:
        finder = make_fast_finder(_setup_genome(td), td, search_cutoff=5.0, n_top_offtargets=3)
        df = finder.search(guides=[_guide()], count_score_threshold=2.0, verbose=False)
        assert df["n_offtargets_below_threshold"].iloc[0] == 1


def test_count_zero_when_threshold_very_low():
    with tempfile.TemporaryDirectory() as td:
        finder = make_fast_finder(_setup_genome(td), td, search_cutoff=5.0, n_top_offtargets=3)
        df = finder.search(guides=[_guide()], count_score_threshold=0.5, verbose=False)
        assert len(df) == 3
        assert df["n_offtargets_below_threshold"].iloc[0] == 0


def test_count_is_independent_of_n_top():
    with tempfile.TemporaryDirectory() as td:
        finder = make_fast_finder(_setup_genome(td), td, search_cutoff=5.0, n_top_offtargets=1)
        df = finder.search(guides=[_guide()], count_score_threshold=5.0, verbose=False)
        assert len(df) == 1
        assert df["n_offtargets_below_threshold"].iloc[0] == 3


def test_count_threshold_above_cutoff_widens_search():
    """A count threshold above search_cutoff is covered by widening the search."""
    with tempfile.TemporaryDirectory() as td:
        finder = make_fast_finder(_setup_genome(td), td, search_cutoff=5.0, n_top_offtargets=3)
        df = finder.search(guides=[_guide()], count_score_threshold=8.0, verbose=False)
        assert df["n_offtargets_below_threshold"].iloc[0] == 3


def test_empty_result():
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        write_fasta(str(fa), {"chr1": "A" * 300})
        finder = make_fast_finder(str(fa), td, search_cutoff=5.0, n_top_offtargets=3)
        df = finder.search(guides=[_guide()], count_score_threshold=10.0, verbose=False)
        assert df.empty
