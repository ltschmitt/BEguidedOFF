"""Threshold-derived search regions for the fast off-target engine.

The CCTOP score is ``sum(1.2 ** p)`` over the 1-indexed mismatch positions ``p``
of a 20 bp guide (``p = 1`` is the 5' / PAM-distal end, ``p = 20`` is the 3' /
PAM-proximal end, adjacent to the PAM).  A *lower* score means a *more*
concerning off-target; a site counts as an off-target when its score is below a
cutoff ``T``.

Given ``T`` we split the guide into two regions:

* **cold** -- positions where a *single* mismatch already pushes the score to or
  above ``T`` (``1.2 ** p >= T``).  Any concerning site must therefore be
  *exact* over the cold region.  This is the PAM-proximal end.
* **hot** -- the remaining (PAM-distal) positions, where mismatches are
  tolerated as long as the running score stays below ``T``.

Enumerating the cold region exactly, then the hot positions within a bounded
mismatch budget, is *sound*: every site with score ``< T`` satisfies the
predicate, so the remove/keep verdict is identical to a full mismatch scan at the
same cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

GUIDE_LENGTH = 20
_BASE = 1.2


def position_cost(position: int) -> float:
    """CCTOP cost contributed by a mismatch at 1-indexed *position*."""
    return _BASE ** position


@dataclass(frozen=True)
class SearchRegions:
    """Cold/hot split derived from a CCTOP score cutoff.

    Attributes:
        cutoff: The score cutoff ``T`` used for the derivation.
        cold: 1-indexed positions that must match exactly (PAM-proximal).
        hot: 1-indexed positions where mismatches are allowed (PAM-distal).
        budget: Maximum number of mismatches in the hot region whose cheapest
            layout still keeps the total score below ``cutoff``.
        length: Guide length (20).
    """

    cutoff: float
    cold: Tuple[int, ...]
    hot: Tuple[int, ...]
    budget: int
    length: int = GUIDE_LENGTH

    @property
    def cold_0based(self) -> Tuple[int, ...]:
        """Cold positions as 0-indexed offsets (for sequence slicing)."""
        return tuple(p - 1 for p in self.cold)

    @property
    def hot_0based(self) -> Tuple[int, ...]:
        """Hot positions as 0-indexed offsets (for sequence slicing)."""
        return tuple(p - 1 for p in self.hot)

    @property
    def cold_anchor(self) -> str:
        """Human-readable span of the cold region, e.g. ``"9-20"``."""
        return f"{self.cold[0]}-{self.cold[-1]}" if self.cold else ""

    def min_cost(self, n_mismatches: int) -> float:
        """Cheapest possible score for exactly *n_mismatches* hot mismatches."""
        if n_mismatches <= 0:
            return 0.0
        costs = sorted(position_cost(p) for p in self.hot)
        if n_mismatches > len(costs):
            return float("inf")
        return float(sum(costs[:n_mismatches]))


def derive_regions(cutoff: float, guide_length: int = GUIDE_LENGTH) -> SearchRegions:
    """Derive the cold/hot split and mismatch budget for a score *cutoff*.

    Args:
        cutoff: CCTOP score cutoff ``T`` (sites with score ``< T`` are
            off-targets worth reporting).
        guide_length: Guide length (default 20).

    Returns:
        A :class:`SearchRegions` describing the exact (cold) positions, the
        fuzzy (hot) positions and the hot mismatch budget.

    Raises:
        ValueError: If *cutoff* is negative.
    """
    if cutoff < 0:
        raise ValueError(f"cutoff must be non-negative, got {cutoff}")

    cold = []
    hot = []
    for p in range(1, guide_length + 1):
        if position_cost(p) >= cutoff:
            cold.append(p)
        else:
            hot.append(p)

    # Budget: the largest k whose cheapest hot layout still sums below cutoff.
    hot_costs = sorted(position_cost(p) for p in hot)
    budget = 0
    running = 0.0
    for cost in hot_costs:
        if running + cost < cutoff:
            running += cost
            budget += 1
        else:
            break

    return SearchRegions(
        cutoff=float(cutoff),
        cold=tuple(cold),
        hot=tuple(hot),
        budget=budget,
        length=guide_length,
    )
