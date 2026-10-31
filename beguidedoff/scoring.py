"""CCTOP off-target scoring."""
from typing import List


def calculate_cctop_score(mismatch_positions: List[int]) -> float:
    """Calculate CCTOP off-target penalty score.

    The CCTOP score sums ``1.2**p`` for every mismatch at position *p*
    (1-indexed from the 5' end of the 20 bp guide). PAM mismatches are
    ignored and should not be included in *mismatch_positions*.

    Lower scores mean more concerning off-targets (fewer mismatches, and
    mismatches far from the PAM). Higher scores mean PAM-proximal
    mismatches, which the model penalizes most heavily.

    Args:
        mismatch_positions: 1-indexed mismatch positions (1 = 5' end).

    Returns:
        Raw CCTOP penalty score.
    """
    if not mismatch_positions:
        return 0.0
    return sum(1.2 ** pos for pos in mismatch_positions)
