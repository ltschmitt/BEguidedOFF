"""Tests for the extensive (GPU) mode's large-input handling.

Covers the two scaling fixes:

* auto-chunking — inputs with more unique scan keys than one kernel launch
  (``cuda_search.MAX_GUIDES = 2048``) are split automatically instead of
  raising "Too many guides";
* overflow re-scan — a guide whose <= max_mismatches matches exceed the
  per-guide raw-hit buffer is re-scanned with a larger buffer (exact
  result); when even the maximum buffer is not enough the genome-order
  prefix of hits is kept and the truncation is logged loudly (guides at
  the ceiling are extreme multi-mappers that no off-target filter would
  pass).

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

    def test_cap_reached_truncates_and_warns(self, overflow_finder, monkeypatch, caplog):
        """With the escalation ceiling at the initial buffer size, an
        overflowing guide is truncated at the buffer (deterministic
        genome-order prefix) and a warning with the exact raw-hit count is
        logged instead of raising."""
        import logging

        monkeypatch.setattr(
            finder_module, "_MAX_RESULTS_PER_GUIDE", finder_module._DEFAULT_RESULTS_PER_GUIDE
        )
        with caplog.at_level(logging.WARNING):
            df = overflow_finder.search(
                guides=_single_guide(), return_all=True, verbose=False
            )
        # No exception: the recorded genome-order prefix is returned...
        assert len(df) == finder_module._DEFAULT_RESULTS_PER_GUIDE
        # ...and the truncation is logged loudly, with the exact count the
        # kernel counted (all _N_COPIES planted (+)-strand sites).
        messages = [r.getMessage() for r in caplog.records]
        assert any("Truncated off-target enumeration" in m for m in messages)
        assert any(str(_N_COPIES) in m for m in messages)


class TestNMaskGate:
    """N positions are encoded as A in the 2-bit genome (T on the
    reverse-complement strand). The kernel must skip candidate sites whose
    26-bp window overlaps an N; without the gate, A/T-rich guides collect a
    phantom hit at every N position of the assembly."""

    @staticmethod
    def _n_genome():
        # Two real sites embedded in G flanks (G never matches A/T-rich
        # guides), separated by a 500-bp N block:
        #   poly-A guide (A20, PAM AAA): real (+)-strand site "A20+AAA",
        #     real (-)-strand site = revcomp("T20+TTT") inside the rc genome;
        #   poly-T guide (T20, PAM TTT): mirrored sites.
        # Inside the N block both the guide window and the PAM are faked
        # (N -> A on '+', N -> T on '-'), so without the gate each guide
        # gains ~500 phantom hits at 0 mismatches.
        return ("G" * 40 + "T" * 20 + "TTT" + "G" * 40
                + "N" * 500
                + "G" * 40 + "A" * 20 + "AAA" + "G" * 40)

    @staticmethod
    def _two_guides():
        return [
            GuideInput("polyA", "A" * 20, "AAA", "3prime", 4, 8, "CBE"),
            GuideInput("polyT", "T" * 20, "TTT", "3prime", 4, 8, "CBE"),
        ]

    def test_n_overlapping_sites_are_excluded(self, tmp_path):
        npz = _write_npz(tmp_path / "genome.fa", tmp_path / "genome.npz",
                         self._n_genome())
        finder = make_finder(genome_path=str(npz), max_mismatches=4,
                             n_top_offtargets=3)
        df = finder.search(guides=self._two_guides(), return_all=True,
                           verbose=False)
        assert not df.empty
        n_start, n_end = 104, 603   # 1-based span of the 500-bp N block
        for guide_seq, group in df.groupby("guide_rna_sequence"):
            # Without the N gate each guide collects ~530 hits (phantom N
            # hits + boundary straddlers); with it only the real sites and
            # their <=4-mm neighbours in the G flanks remain.
            assert len(group) < 15, f"phantom N hits for {guide_seq}"
            # No hit may fall inside the N block.
            assert not ((group["ot_chromosome"] == "chr1")
                        & (group["ot_position"] >= n_start)
                        & (group["ot_position"] <= n_end)).any()
            # The planted real sites are found with 0 mismatches.
            assert (group["n_mismatches"] == 0).any()

    def test_site_adjacent_to_n_run_is_kept(self, tmp_path):
        """Regression: a minus-strand NGG site whose forward window starts
        3 bp after an N run must be found. The gate must cover only the
        20-nt guide window — over-extending into the PAM/flank (as a fixed
        26-bp check does for short PAMs) wrongly drops real sites next to
        N stretches."""
        genome = "G" * 30 + "N" * 10 + "CCC" + "T" * 20 + "G" * 30
        npz = _write_npz(tmp_path / "genome.fa", tmp_path / "genome.npz", genome)
        finder = make_finder(genome_path=str(npz), max_mismatches=4,
                             n_top_offtargets=3)
        df = finder.search(
            guides=[GuideInput("polyA", "A" * 20, "NGG", "3prime", 4, 8, "CBE")],
            return_all=True, verbose=False,
        )
        # The minus-strand site (forward CCC+T20 = revcomp of A20+NGG) and
        # its 1-mm neighbour (guide window shifted by one, PAM AGG) are the
        # only possible hits; the plus strand cannot match poly-A vs T20.
        # Both guide windows ([43,63) and [44,64)) clear the N block, so the
        # old fixed-26-bp gate — which over-checked into the N run — dropped
        # them both.
        assert len(df) == 2
        assert (df["ot_strand"] == "-").all()
        zero_mm = df[df["n_mismatches"] == 0]
        assert len(zero_mm) == 1
        assert zero_mm.iloc[0]["ot_sequence"] == "A" * 20

    def test_genome_without_n_mask_still_searches(self, tmp_path):
        """Backward compatibility: a genome encoded without an N mask
        searches ungated (historical behaviour) and still finds sites."""
        fa = tmp_path / "genome.fa"
        npz = tmp_path / "genome.npz"
        with open(fa, "w") as fh:
            fh.write(">chr1\n")
            genome = "G" * 40 + "A" * 20 + "AAA" + "G" * 40
            for i in range(0, len(genome), 80):
                fh.write(genome[i : i + 80] + "\n")
        convert(str(fa), str(npz), include_n_mask=False, verbose=False)
        finder = make_finder(genome_path=str(npz), max_mismatches=4,
                             n_top_offtargets=3)
        df = finder.search(
            guides=[GuideInput("polyA", "A" * 20, "AAA", "3prime", 4, 8, "CBE")],
            return_all=True, verbose=False,
        )
        # The homopolymer flank legitimately matches with <=4 mismatches;
        # only assert the planted exact site is among the hits.
        exact = df[(df["n_mismatches"] == 0)]
        assert len(exact) >= 1


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


# ---------------------------------------------------------------------------
# Exotic-PAM kernel path (packed-PAM fallback for S/W/B/D/H/V positions)
# ---------------------------------------------------------------------------
import random  # noqa: E402

from beguidedoff.base_editing import editable_base_in_window  # noqa: E402
from beguidedoff.encoding import (  # noqa: E402
    encode_dna,
    encode_guide_64bit,
    mismatch_positions_2bit,
    reverse_complement_seq,
)
from beguidedoff.pam import IUPAC_MASK  # noqa: E402
from beguidedoff.scoring import calculate_cctop_score  # noqa: E402

_NUC_2BIT = {"A": 0, "C": 1, "G": 2, "T": 3}
# One concrete base per IUPAC code (used to plant valid sites).
_RESOLVE = {"A": "A", "C": "C", "G": "G", "T": "T", "R": "A", "Y": "C",
            "M": "A", "K": "G", "S": "C", "W": "A", "H": "A", "B": "C",
            "V": "A", "D": "A", "N": "A"}
# A concrete base NOT allowed by the code (only meaningful for non-N codes).
_VIOLATE = {"V": "T", "S": "A", "W": "C", "B": "A", "D": "C", "H": "G"}


def _rand_seq(n):
    return "".join(random.choice("ACGT") for _ in range(n))


def _pam_ok(genome_pam, pam):
    return all((1 << _NUC_2BIT[gb]) & IUPAC_MASK[pb] for gb, pb in zip(genome_pam, pam))


def _bruteforce_sites(seq, guide_seq, pam, editor_type, cutoff):
    """Independent reference: all editable sites with score <= cutoff.

    Both strands, IUPAC-aware PAM test; tuples
    ``(chrom, position_1based, strand, n_mismatches, score)``.
    """
    win = 20 + len(pam)
    gval = encode_guide_64bit(encode_dna(guide_seq))
    hits = set()

    def consider(oriented, i, strand, position):
        w = oriented[i : i + 20]
        p = oriented[i + 20 : i + win]
        if len(p) < len(pam) or not _pam_ok(p, pam):
            return
        mm = mismatch_positions_2bit(
            gval, encode_guide_64bit(encode_dna(w)), length=20
        )
        score = calculate_cctop_score(mm)
        if score > cutoff:
            return
        if not editable_base_in_window(w, editor_type, 4, 8):
            return
        hits.add((position, strand, len(mm), round(score, 6)))

    for i in range(len(seq) - win + 1):
        consider(seq, i, "+", i + 1)
    rc = reverse_complement_seq(seq)
    for i in range(len(rc) - win + 1):
        leftmost = len(seq) - i - win
        consider(rc, i, "-", leftmost + win)
    return hits


class TestExoticPamKernelPath:
    """The packed-PAM fast path covers A/C/G/T, R/Y/M/K/N exactly; codes with
    allowed-set {S,W,B,D,H,V} take the per-base fallback in the kernel.  This
    test plants sites for such PAMs and checks the GPU extensive search
    against an independent brute-force reference."""

    def test_exotic_pam_matches_bruteforce(self, tmp_path):
        random.seed(11)
        specs = [("TTTV", "CBE"), ("SGG", "CBE"), ("WGG", "CBE"), ("NGG", "CBE")]
        # Guides carry a C inside the CBE window (0-based 3..7) so every
        # planted site passes the editability filter.
        guides = []
        c_idx = []
        for _ in specs:
            g = list(_rand_seq(20))
            c = random.randrange(3, 8)
            g[c] = "C"
            c_idx.append(c)
            guides.append("".join(g))
        seq = _rand_seq(24_000)
        planted = []  # (position_1based, strand, n_mm) expected as hits
        pos = 200  # sequential, non-overlapping plant positions
        for guide, c0, (pam, _) in zip(guides, c_idx, specs):
            # valid sites: exact x2, 1 mm, 2 mm (exotic position resolved).
            # Mutations stay in the 5'-distal half: the CCTOP penalty
            # (sum of 1.2**p over 1-based positions) must stay within the
            # 5.0 cutoff, and away from the guide's C so sites stay editable.
            mut_range = [i for i in range(5) if i != c0]
            for n_mm in (0, 0, 1, 2):
                site = list(guide)
                muts = random.sample(mut_range, n_mm)
                for p in muts:
                    site[p] = random.choice([b for b in "ACGT" if b != site[p]])
                pam_seq = "".join(_RESOLVE[c] for c in pam)
                seq = seq[:pos] + "".join(site) + pam_seq + seq[pos + 24:]
                score = calculate_cctop_score(sorted(p + 1 for p in muts))
                if (score <= 5.0
                        and editable_base_in_window("".join(site), "CBE", 4, 8)):
                    planted.append((pos + 1, "+", n_mm))
                pos += 60
            # decoys: window matches exactly but the exotic PAM base is invalid
            bad_code = next((c for c in pam if c in _VIOLATE), None)
            if bad_code is not None:
                pam_bad = "".join(
                    _VIOLATE[c] if c == bad_code else _RESOLVE[c] for c in pam
                )
                seq = seq[:pos] + guide + pam_bad + seq[pos + 24:]
                pos += 60

        npz = _write_npz(tmp_path / "exotic.fa", tmp_path / "exotic.npz", seq)
        finder = make_finder(genome_path=str(npz), max_mismatches=4,
                             n_top_offtargets=3)
        guide_inputs = [
            GuideInput(f"g{i}", g, pam, "3prime", 4, 8, et)
            for i, (g, (pam, et)) in enumerate(zip(guides, specs))
        ]
        df = finder.search(guides=guide_inputs, return_all=True, verbose=False)

        got = set()
        editor_by_seq = {gi.guide_seq: gi.editor_type for gi in guide_inputs}
        for _, row in df.iterrows():
            if row["score"] > 5.0:
                continue
            if not editable_base_in_window(row["ot_sequence"],
                                           editor_by_seq[row["guide_rna_sequence"]],
                                           4, 8):
                continue
            got.add((int(row["ot_position"]), row["ot_strand"],
                     int(row["n_mismatches"]), round(float(row["score"]), 6)))

        ref = set()
        for g, (pam, et) in zip(guides, specs):
            ref |= _bruteforce_sites(seq, g, pam, et, 5.0)

        assert got == ref, (
            f"only in extensive: {sorted(got - ref)[:5]}\n"
            f"only in reference: {sorted(ref - got)[:5]}"
        )
        # sanity: the search is not vacuous — every planted valid site is found
        got_keys = {(p, s, m) for (p, s, m, _sc) in got}
        missing = [pk for pk in planted if pk not in got_keys]
        assert not missing, f"planted sites missing from results: {missing}"
