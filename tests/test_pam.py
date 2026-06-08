"""Tests for PAM encoding."""
from beguidedoff.pam import (
    validate_pam,
    encode_pam_mask,
    all_possible_pams,
    group_guides_by_pam,
)
from beguidedoff.types import GuideInput
import numpy as np


def test_validate_pam_ok():
    validate_pam("NGG")
    validate_pam("NNGRRT")


def test_validate_pam_too_long():
    try:
        validate_pam("NNNNNNN")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_encode_pam_mask():
    mask = encode_pam_mask("NGG")
    assert mask[0] == 0b1111  # N
    assert mask[1] == 0b0100  # G
    assert mask[2] == 0b0100  # G
    assert mask[3] == 0  # padding


def test_all_possible_pams():
    pams = all_possible_pams("NGG")
    assert len(pams) == 4
    assert "AGG" in pams
    assert "GGG" in pams


def test_group_guides_by_pam():
    guides = [
        GuideInput("g1", "ACGTACGTACGTACGTACGT", "NGG", "3prime", 4, 8, "CBE"),
        GuideInput("g2", "ACGTACGTACGTACGTACGT", "NNGRRT", "3prime", 4, 8, "CBE"),
        GuideInput("g3", "ACGTACGTACGTACGTACGT", "NGG", "3prime", 4, 8, "ABE"),
    ]
    groups = group_guides_by_pam(guides)
    assert len(groups) == 2
    assert len(groups[("NGG", "3prime")]) == 2
    assert len(groups[("NNGRRT", "3prime")]) == 1
