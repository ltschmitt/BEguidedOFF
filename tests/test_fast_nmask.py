"""Fast-mode N semantics: a reference N never matches (Cas-OFFinder rule).

The FM-index alphabet has no fifth symbol, so ``build_index`` encodes every
non-ACGT base as A and writes an N-run sidecar (``<index>.nmask``).  The
finder then counts a guide base that equals the N-decoded symbol as a
mismatch — A on the forward strand, T on the reverse-complement strand —
mirroring the extensive mode exactly.

These tests need the compiled sdsl-lite binding; they skip elsewhere.
"""
import shutil

import pytest

from _fastutil import bruteforce_hits, fast_hits, make_fast_finder, write_fasta
from beguidedoff.fast.finder import beguidedofffast
from beguidedoff.types import GuideInput

# G flanks never match the poly-A guide; the planted A20+AAA site and its
# <=5-mm neighbours are the only N-free hits; the 200-base N block would
# yield ~180 phantom 0-mm hits without the N correction.
_GENOME = "G" * 30 + "A" * 20 + "AAA" + "G" * 30 + "N" * 200 + "G" * 30
_N_SPAN_1BASED = (84, 283)   # 0-based [83, 283)


def _guide():
    return [GuideInput("polyA", "A" * 20, "AAA", "3prime", 4, 8, "ABE")]


def _build(tmp_path):
    fa = tmp_path / "genome.fa"
    write_fasta(str(fa), {"chr1": _GENOME})
    finder = make_fast_finder(str(fa), str(tmp_path))
    return finder


def test_n_window_sites_are_excluded(tmp_path):
    """Phantom N matches must not be reported; the real site must be."""
    finder = _build(tmp_path)
    df = finder.search(guides=_guide(), return_all=True, verbose=False)
    hits = fast_hits(df)

    # The planted exact site (1-based 31) plus its <=5-mm neighbours in the
    # G flank (windows starting 1-based 28-30). Without the N correction the
    # 200-base N block (encoded as A) would add ~180 phantom 0-mm hits.
    assert len(hits) == 4, hits
    for chrom, pos, _strand, _n_mm, _score in hits:
        assert chrom == "chr1"
        assert not (_N_SPAN_1BASED[0] <= pos <= _N_SPAN_1BASED[1]), (
            f"hit reported inside the N block: {pos}"
        )
    exact = [h for h in hits if h[1] == 31]
    assert len(exact) == 1 and exact[0][3] == 0


def test_fast_matches_bruteforce_on_n_genome(tmp_path):
    """Independent oracle parity on a genome containing N."""
    finder = _build(tmp_path)
    df = finder.search(guides=_guide(), return_all=True, verbose=False)
    expected = bruteforce_hits({"chr1": _GENOME}, "A" * 20, "AAA", 5.0, "ABE", 4, 8)
    assert sorted(fast_hits(df)) == sorted(expected)


def test_index_without_sidecar_keeps_old_behaviour(tmp_path):
    """An index copied without its sidecar searches ungated (pre-mask behaviour)."""
    fa = tmp_path / "genome.fa"
    write_fasta(str(fa), {"chr1": _GENOME})
    index_path = tmp_path / "genome.fmi"
    from beguidedoff.fast.fmindex import build_index

    build_index(str(fa), str(index_path), verbose=False)
    assert (tmp_path / "genome.fmi.nmask").exists()

    gated = beguidedofffast(str(index_path))
    df_gated = gated.search(guides=_guide(), return_all=True, verbose=False)

    # copy the index and its .meta sidecar, but NOT the .nmask -> the loader
    # finds no mask and keeps the historical (ungated) behaviour
    legacy_path = tmp_path / "legacy.fmi"
    shutil.copy(index_path, legacy_path)
    shutil.copy(str(index_path) + ".meta", str(legacy_path) + ".meta")
    assert not (tmp_path / "legacy.fmi.nmask").exists()
    legacy = beguidedofffast(str(legacy_path))
    df_legacy = legacy.search(guides=_guide(), return_all=True, verbose=False)

    hits_gated = fast_hits(df_gated)
    hits_legacy = fast_hits(df_legacy)
    assert len(hits_legacy) > 100, "expected ungated phantom N hits"
    assert len(hits_gated) < 10
    assert len(hits_gated) < len(hits_legacy)


def test_n_in_range_helper(tmp_path):
    """The index exposes the sidecar runs as a per-position mask."""
    finder = _build(tmp_path)
    idx = finder._index
    assert idx.has_n_mask
    # single chromosome, no separator offset: N block at 0-based [83, 283)
    mask = idx.n_in_range(78, 20)
    assert list(mask) == [False] * 5 + [True] * 15
    assert idx.n_in_range(0, 30) is not None
    assert not idx.n_in_range(0, 30).any()
    assert idx.n_in_range(300, 13) is not None
    assert not idx.n_in_range(300, 13).any()


def test_fast_matches_extensive_on_n_genome(tmp_path):
    """Both engines must agree exactly on a genome containing N."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for extensive mode")
    from beguidedoff import beguidedoff as make_extensive
    from beguidedoff.genome_converter import convert

    fa = tmp_path / "genome.fa"
    write_fasta(str(fa), {"chr1": _GENOME})
    npz = tmp_path / "genome.npz"
    convert(str(fa), str(npz), verbose=False)

    # score cutoff 5.0 (fast) allows at most 3 guide mismatches (the 4-mm
    # CCTOP minimum is 1.2+1.44+1.73+2.07 = 6.44), so the extensive side
    # must use the same budget for an exact-set comparison.
    ext = make_extensive(genome_path=str(npz), max_mismatches=3, n_top_offtargets=100000)
    df_ext = ext.search(
        guides=_guide(), return_all=True, include_non_editable=True, verbose=False
    )
    ext_hits = {
        (r.ot_chromosome, int(r.ot_position), r.ot_strand,
         int(r.n_mismatches), round(float(r.score), 6))
        for _, r in df_ext.iterrows()
    }

    fast = _build(tmp_path)
    df_fast = fast.search(
        guides=_guide(), return_all=True, include_non_editable=True, verbose=False
    )
    fast_set = {
        (r.ot_chromosome, int(r.ot_position), r.ot_strand,
         int(r.n_mismatches), round(float(r.score), 6))
        for _, r in df_fast.iterrows()
    }
    assert fast_set == ext_hits, (
        f"fast-only: {sorted(fast_set - ext_hits)[:5]}; "
        f"extensive-only: {sorted(ext_hits - fast_set)[:5]}"
    )
