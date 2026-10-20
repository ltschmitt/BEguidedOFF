"""Tests for threshold-derived search regions."""
import pytest

from beguidedoff.regions import derive_regions, position_cost


def test_cutoff_five_split():
    """T=5 -> exact PAM-proximal 9-20, budget 3 in the PAM-distal 1-8."""
    r = derive_regions(5.0)
    assert r.cold == tuple(range(9, 21))
    assert r.hot == tuple(range(1, 9))
    assert r.budget == 3
    assert r.cold_anchor == "9-20"
    assert r.cold_0based == tuple(range(8, 20))


def test_cutoff_three_split():
    """T=3 -> exact 7-20 (1.2**6 = 2.99 < 3), budget 2."""
    r = derive_regions(3.0)
    assert r.cold == tuple(range(7, 21))
    assert r.hot == tuple(range(1, 7))
    assert r.budget == 2


def test_cutoff_fifteen_split():
    """T=15 -> exact 15-20, larger hot budget."""
    r = derive_regions(15.0)
    assert r.cold == tuple(range(15, 21))
    assert r.hot == tuple(range(1, 15))
    assert r.budget == 6


def test_lower_cutoff_longer_cold_anchor():
    """A smaller cutoff must yield a longer (or equal) exact anchor."""
    assert len(derive_regions(3.0).cold) > len(derive_regions(5.0).cold)
    assert len(derive_regions(5.0).cold) > len(derive_regions(15.0).cold)


def test_budget_is_consistent_with_soundness():
    """The budget's cheapest layout is < cutoff; adding one more is not."""
    for cutoff in (2.0, 3.0, 5.0, 8.0, 15.0):
        r = derive_regions(cutoff)
        assert r.min_cost(r.budget) < cutoff
        # one more hot mismatch must cross the cutoff (unless hot is exhausted)
        if r.budget < len(r.hot):
            assert r.min_cost(r.budget + 1) >= cutoff


def test_position_cost_monotonic():
    """Cost strictly increases toward the PAM (higher position index)."""
    assert position_cost(20) > position_cost(19) > position_cost(1)
    assert position_cost(1) == pytest.approx(1.2)


def test_invalid_cutoff():
    with pytest.raises(ValueError):
        derive_regions(-1.0)


def test_zero_cutoff_is_exact_match_only():
    """A cutoff of 0 degenerates to a fully cold guide (perfect matches only)."""
    r = derive_regions(0.0)
    assert r.cold == tuple(range(1, 21))
    assert r.hot == ()
    assert r.budget == 0
    # Only a 0-mismatch hit can pass: cheapest 1-mismatch score exceeds 0.
    assert r.min_cost(1) > 0.0
