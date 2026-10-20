"""Tests for the extensive (GPU) mode's large-input handling.

Covers the two scaling fixes:

* auto-chunking — inputs with more unique scan keys than one kernel launch
  (``cuda_search.MAX_GUIDES = 2048``) are split automatically instead of
  raising "Too many guides";
* overflow re-scan — a guide whose <= max_mismatches matches exceed the
  per-guide raw-hit buffer is re-scanned with a larger buffer (exact
  result), and the search fails loudly when even the maximum buffer is not
  enough instead of returning a silently truncated hit list.

These tests need a CUDA GPU; they skip elsewhere.
"""
from itertools import product

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numba")

from beguidedoff import beguidedoff as make_finder  # noqa: E402
from beguidedoff import finder as finder_module  # noqa: E402
from beguidedoff.genome_converter import convert  # noqa: E402
from beguidedoff.types import GuideInput  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required for extensive mode"
)

_GUIDE = "ACGTACGTACGTACGTACGT"
_N_COPIES = 12_000  # > the default 10240-hit buffer


def _repeat_genome() -> str:
    """Genome of ``_N_COPIES`` copies of a 1-mismatch neighbour of ``_GUIDE``
    (each followed by an NGG PAM), padded so the last copy fits."""
    unit = "T" + _GUIDE[1:] + "AGG"  # mismatch at guide position 1
    return unit * _N_COPIES + "A" * 30


def _write_npz(fa_path, npz_path, genome, chrom="chr1"):
    with open(fa_path, "w") as fh:
        fh.write(f">{chrom}\n")
        for i in range(0, len(genome), 80):
            fh.write(genome[i : i + 80] + "\n")
    convert(str(fa_path), str(npz_path), verbose=False)
    return npz_path


@pytest.fixture(scope="module")
def overflow_finder(tmp_path_factory):
    """Extensive finder on a genome whose repeat guide has ~2x buffer hits."""
    work = tmp_path_factory.mktemp("extensive_overflow")
    npz = _write_npz(work / "genome.fa", work / "genome.npz", _repeat_genome())
    return make_finder(genome_path=str(npz), max_mismatches=4, n_top_offtargets=3)


def _single_guide():
    return [GuideInput("g1", _GUIDE, "NGG", "3prime", 4, 8, "CBE")]


class TestOverflowRescan:
    def test_overflowed_guide_is_rescanned_completely(self, overflow_finder):
        """The repeat guide has more matches than the initial buffer holds;
        without the re-scan the count would silently cap at results_per_guide
        (10 240)."""
        df = overflow_finder.search(
            guides=_single_guide(),
            return_all=False,
            count_score_threshold=5.0,
            verbose=False,
        )
        # Top-3 rows for the single guide; all planted hits tie at score 1.2,
        # so the top-3 are the three leftmost sites.
        assert len(df) == 3
        row = df.iloc[0]
        # The planted 1-mismatch neighbour is the top hit.
        assert row["n_mismatches"] == 1
        # All _N_COPIES planted sites on the (+) strand are enumerated (the
        # ACGT-repeat guide has no <=4mm neighbours on the minus strand).
        assert row["n_offtargets_below_threshold"] == _N_COPIES

    def test_cap_reached_fails_loudly(self, overflow_finder, monkeypatch):
        """With the escalation ceiling at the initial buffer size, an
        overflowing guide must raise instead of returning truncated hits."""
        monkeypatch.setattr(
            finder_module, "_MAX_RESULTS_PER_GUIDE", finder_module._DEFAULT_RESULTS_PER_GUIDE
        )
        with pytest.raises(RuntimeError, match="still exceed the raw-hit buffer"):
            overflow_finder.search(
                guides=_single_guide(), return_all=False, verbose=False
            )


class TestAutoChunking:
    def test_more_scan_keys_than_max_guides(self, tmp_path):
        """2 100 unique scan keys (> MAX_GUIDES=2048) search without the
        historical "Too many guides" error, and a planted exact site is
        found."""
        rng = np.random.default_rng(7)
        genome = list(rng.choice(list("ACGT"), size=30_000))
        bases = product("ACGT", repeat=6)
        guides = [
            GuideInput(f"g{i}", "ACGTACGTACGTAC" + "".join(p), "NGG",
                       "3prime", 4, 8, "CBE")
            for i, p in enumerate(bases)
        ]
        guides = guides[:2_100]
        planted = guides[137]
        # Plant an exact NGG site for guide #137 inside the genome.
        for pos in (5_000, 15_000, 25_000):
            genome[pos : pos + 20] = list(planted.guide_seq)
            genome[pos + 20 : pos + 23] = list("AGG")
        npz = _write_npz(tmp_path / "genome.fa", tmp_path / "genome.npz",
                         "".join(genome))
        finder = make_finder(genome_path=str(npz), max_mismatches=4,
                             n_top_offtargets=3)
        df = finder.search(guides=guides, return_all=False, verbose=False)
        planted_rows = df[df["guide_rna_sequence"] == planted.guide_seq]
        assert len(planted_rows) == 3
        assert (planted_rows["n_mismatches"] == 0).all()
