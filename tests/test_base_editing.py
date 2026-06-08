"""Tests for base editing logic."""
from beguidedoff.base_editing import (
    editable_base_in_window,
    get_trinucleotide_contexts,
    has_editable_base,
)


def test_cbe_editable():
    seq = "ACGTACGTACGTACGTACGT"
    # CBE edits C->T; window 4-8 contains positions 4,5,6,7,8
    # seq[3] = T, seq[4] = A, seq[5] = C, ...
    # positions (1-indexed): 1=A,2=C,3=G,4=T,5=A,6=C,7=G,8=T...
    positions = editable_base_in_window(seq, "CBE", 4, 8)
    assert 6 in positions  # C at position 6


def test_abe_editable():
    seq = "ACGTACGTACGTACGTACGT"
    positions = editable_base_in_window(seq, "ABE", 1, 5)
    # A at positions 1, 5
    assert 1 in positions
    assert 5 in positions


def test_no_editable_base():
    seq = "GGGGGGGGGGGGGGGGGGGG"
    positions = editable_base_in_window(seq, "CBE", 1, 10)
    assert positions == []


def test_trinucleotide_context():
    seq = "ACGTACGT"
    contexts = get_trinucleotide_contexts(seq, [3])
    # Position 3 = G, context = CGT
    assert contexts == ["CGT"]


def test_has_editable_base():
    assert has_editable_base("ACGTACGTACGTACGTACGT", "CBE", 4, 8)
    assert not has_editable_base("GGGGGGGGGGGGGGGGGGGG", "CBE", 1, 10)
