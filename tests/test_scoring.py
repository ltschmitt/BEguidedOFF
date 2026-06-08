"""Tests for scoring."""
import pytest

from beguidedoff.scoring import calculate_cfd_score, calculate_cctop_score, PAM_SCORES


# ---------------------------------------------------------------------------
# CCTOP score tests (default)
# ---------------------------------------------------------------------------


def test_cctop_perfect_match():
    score = calculate_cctop_score([])
    assert score == 0.0


def test_cctop_single_mismatch_far_from_pam():
    # Position 20 (furthest from PAM = seed = most important)
    score = calculate_cctop_score([20])
    expected = 1.2 ** 20
    assert abs(score - expected) < 1e-9


def test_cctop_single_mismatch_seed():
    # Position 1 (closest to PAM = seed = least penalised)
    score = calculate_cctop_score([1])
    expected = 1.2 ** 1
    assert abs(score - expected) < 1e-9


def test_cctop_multiple_mismatches():
    score = calculate_cctop_score([1, 5, 10])
    expected = 1.2 ** 1 + 1.2 ** 5 + 1.2 ** 10
    assert abs(score - expected) < 1e-9


def test_cctop_score_increases_with_mismatch_distance():
    """Mismatch farther from PAM (higher position index) receives higher penalty."""
    score_near_pam = calculate_cctop_score([1])
    score_far_pam = calculate_cctop_score([20])
    assert score_far_pam > score_near_pam


def test_cctop_score_is_additive():
    """Multiple mismatches sum independently."""
    score1 = calculate_cctop_score([10])
    score2 = calculate_cctop_score([15])
    score_combined = calculate_cctop_score([10, 15])
    assert abs(score_combined - (score1 + score2)) < 1e-9


# ---------------------------------------------------------------------------
# CFD score tests (deprecated reference, kept for backward compatibility)
# ---------------------------------------------------------------------------


def test_cfd_perfect_match():
    score = calculate_cfd_score("ACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGTGG")
    assert abs(score - 1.0) < 1e-9


def test_cfd_single_mismatch():
    score = calculate_cfd_score("ACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGAGG")
    # Single mismatch at position 20, penalty is ~0.4-0.7 depending on base
    assert 0.0 < score < 1.0


def test_cfd_bad_pam():
    # PAM scoring is currently deactivated to avoid misleading scores for
    # non-SpCas9 PAM families, so a non-GG dinucleotide no longer forces the
    # score to zero.
    score = calculate_cfd_score("ACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGTAA")
    assert score == 1.0


def test_pam_scores():
    assert PAM_SCORES["GG"] == 1.0
    assert PAM_SCORES["AA"] == 0.0
