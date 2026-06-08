"""Tests for encoding utilities."""
import numpy as np
from beguidedoff.encoding import (
    encode_dna,
    decode_dna,
    reverse_complement_2bit,
    reverse_complement_seq,
    pack_2bit_to_uint32,
    unpack_uint32_to_2bit,
    encode_guide_64bit,
    decode_guide_64bit,
    hamming_distance_2bit,
    mismatch_positions_2bit,
)


def test_encode_decode_roundtrip():
    seq = "ACGTACGTNNACGT"
    encoded = encode_dna(seq)
    decoded = decode_dna(encoded)
    # Ns are encoded as A (0b00)
    assert decoded == seq.replace("N", "A")


def test_reverse_complement():
    seq = "ACGT"
    rc = reverse_complement_seq(seq)
    assert rc == "ACGT"  # RC of ACGT is ACGT (palindrome)
    assert reverse_complement_seq("ATCG") == "CGAT"


def test_pack_unpack_roundtrip():
    seq = "ACGT" * 10  # 40 bases
    encoded = encode_dna(seq)
    packed = pack_2bit_to_uint32(encoded)
    unpacked = unpack_uint32_to_2bit(packed, len(encoded))
    assert np.array_equal(encoded, unpacked)


def test_guide_64bit_roundtrip():
    seq = "ACGTACGTACGTACGTACGT"  # 20 bases
    encoded = encode_dna(seq)
    val = encode_guide_64bit(encoded)
    decoded = decode_guide_64bit(val, length=20)
    assert np.array_equal(encoded, decoded)


def test_hamming_distance():
    a = encode_dna("ACGTACGTACGTACGTACGT")
    b = encode_dna("ACGTACGTACGTACGTACGA")  # 1 mismatch at end
    va = encode_guide_64bit(a)
    vb = encode_guide_64bit(b)
    assert hamming_distance_2bit(va, vb, length=20) == 1

    c = encode_dna("ACGTACGTACGTACGTACGT")
    vc = encode_guide_64bit(c)
    assert hamming_distance_2bit(va, vc, length=20) == 0


def test_mismatch_positions():
    a = encode_dna("ACGTACGTACGTACGTACGT")
    b = encode_dna("ACGTACGTACGTACGTACGA")  # mismatch at pos 20
    va = encode_guide_64bit(a)
    vb = encode_guide_64bit(b)
    assert mismatch_positions_2bit(va, vb, length=20) == [20]

    c = encode_dna("TCGTACGTACGTACGTACGT")  # mismatch at pos 1
    vc = encode_guide_64bit(c)
    assert mismatch_positions_2bit(va, vc, length=20) == [1]
