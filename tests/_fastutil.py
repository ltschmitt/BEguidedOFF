"""Shared helpers for the fast-mode tests.

Includes an *independent* brute-force reference matcher so the FM-index search
can be checked for equivalence without relying on the removed CPU code path.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

try:  # the fast tests need the compiled sdsl-lite binding
    from beguidedoff import _fmindex  # noqa: F401
except ImportError:  # pragma: no cover - depends on build env
    pytest.skip(
        "beguidedoff._fmindex is not built (install the fast extra)",
        allow_module_level=True,
    )

from beguidedoff.base_editing import editable_base_in_window
from beguidedoff.encoding import (
    encode_dna,
    encode_guide_64bit,
    mismatch_positions_2bit,
    reverse_complement_seq,
)
from beguidedoff.fast.finder import beguidedofffast
from beguidedoff.fast.fmindex import build_index
from beguidedoff.pam import IUPAC_MASK
from beguidedoff.scoring import calculate_cctop_score

_GUIDE_LENGTH = 20
_NUC_2BIT = {"A": 0, "C": 1, "G": 2, "T": 3}


def write_fasta(path: str, entries: Dict[str, str]) -> None:
    with open(path, "w") as f:
        for name, seq in entries.items():
            f.write(f">{name}\n{seq}\n")


def make_fast_finder(fasta_path: str, tmpdir: str, **kwargs) -> beguidedofffast:
    index_path = Path(tmpdir) / "genome.fmi"
    build_index(str(fasta_path), str(index_path), verbose=False)
    return beguidedofffast(str(index_path), **kwargs)


def _pam_ok(genome_pam: str, pattern: str) -> bool:
    if len(genome_pam) != len(pattern):
        return False
    for gb, pb in zip(genome_pam, pattern):
        if not ((1 << _NUC_2BIT.get(gb, 0)) & IUPAC_MASK.get(pb, 0)):
            return False
    return True


def bruteforce_hits(
    chroms: Dict[str, str],
    guide_seq: str,
    pam: str,
    cutoff: float,
    editor_type: str = "CBE",
    edit_window_start: int = 4,
    edit_window_end: int = 8,
) -> List[Tuple[str, int, str, int, float]]:
    """Naive exhaustive matcher (both strands).

    Returns editable sites with score <= cutoff as tuples
    ``(chrom, position_1based, strand, n_mismatches, score)``.
    """
    window = _GUIDE_LENGTH + len(pam)
    gval = encode_guide_64bit(encode_dna(guide_seq))
    hits: List[Tuple[str, int, str, int, float]] = []

    def consider(chrom, oriented_seq, i, strand, position):
        w = oriented_seq[i:i + _GUIDE_LENGTH]
        p = oriented_seq[i + _GUIDE_LENGTH:i + window]
        if not _pam_ok(p, pam):
            return
        mm = mismatch_positions_2bit(gval, encode_guide_64bit(encode_dna(w)), length=_GUIDE_LENGTH)
        score = calculate_cctop_score(mm)
        if score > cutoff:
            return
        if not editable_base_in_window(w, editor_type, edit_window_start, edit_window_end):
            return
        hits.append((chrom, position, strand, len(mm), round(score, 6)))

    for chrom, seq in chroms.items():
        for i in range(0, len(seq) - window + 1):
            consider(chrom, seq, i, "+", i + 1)
        rc = reverse_complement_seq(seq)
        for i in range(0, len(rc) - window + 1):
            leftmost = len(seq) - i - window
            consider(chrom, rc, i, "-", leftmost + window)

    return hits


def fast_hits(df) -> List[Tuple[str, int, str, int, float]]:
    """Extract comparable hit tuples from a fast-mode result DataFrame."""
    if df is None or df.empty:
        return []
    return [
        (row["ot_chromosome"], int(row["ot_position"]), row["ot_strand"],
         int(row["n_mismatches"]), round(float(row["score"]), 6))
        for _, row in df.iterrows()
    ]
