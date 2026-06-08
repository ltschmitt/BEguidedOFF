"""2-bit encoding utilities for DNA sequences."""
import numpy as np
from typing import List

# Nucleotide → 2-bit: A = 0b00, C = 0b01, G = 0b10, T = 0b11
NUC_TO_2BIT = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}

BIT_TO_NUC = {0: "A", 1: "C", 2: "G", 3: "T"}

# Complement in 2-bit space: A↔T, C↔G
COMPLEMENT_2BIT = {0: 3, 1: 2, 2: 1, 3: 0}


def encode_dna(seq: str) -> np.ndarray:
    """Encode a DNA sequence to 2-bit uint8 array."""
    seq = seq.upper()
    return np.array([NUC_TO_2BIT.get(base, 0) for base in seq], dtype=np.uint8)


def decode_dna(encoded: np.ndarray) -> str:
    """Decode a 2-bit uint8 array back to DNA sequence."""
    return "".join(BIT_TO_NUC.get(int(b), "N") for b in encoded)


def reverse_complement_2bit(encoded: np.ndarray) -> np.ndarray:
    """Return 2-bit reverse complement."""
    return np.array([COMPLEMENT_2BIT[int(b)] for b in reversed(encoded)], dtype=np.uint8)


def reverse_complement_seq(seq: str) -> str:
    """Return reverse complement of DNA sequence."""
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(comp.get(base, base) for base in reversed(seq.upper()))


def pack_2bit_to_uint32(encoded: np.ndarray) -> np.ndarray:
    """Pack array of 2-bit values (0-3) into uint32, 16 nucleotides per element."""
    n_packed = (len(encoded) + 15) // 16
    padded = np.zeros(n_packed * 16, dtype=np.uint8)
    padded[: len(encoded)] = encoded

    packed = np.zeros(n_packed, dtype=np.uint32)
    for i in range(16):
        packed |= padded[i::16].astype(np.uint32) << (i * 2)

    return packed


def unpack_uint32_to_2bit(packed: np.ndarray, total_bp: int) -> np.ndarray:
    """Unpack uint32 array back to 2-bit uint8 array."""
    n_packed = len(packed)
    encoded = np.zeros(n_packed * 16, dtype=np.uint8)
    for i in range(16):
        encoded[i::16] = (packed >> (i * 2)) & 0x3
    return encoded[:total_bp]


def extract_window(packed: np.ndarray, start_bp: int, length: int) -> np.ndarray:
    """Extract a window of `length` bases from a packed uint32 genome starting at `start_bp`."""
    encoded = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        gpos = start_bp + i
        word_idx = gpos // 16
        bit_offset = (gpos % 16) * 2
        nuc_val = (packed[word_idx] >> bit_offset) & 0x3
        encoded[i] = nuc_val
    return encoded


def encode_guide_64bit(encoded: np.ndarray) -> int:
    """Pack a 20-base 2-bit encoded guide into a 40-bit integer."""
    val = 0
    for i, b in enumerate(encoded):
        val |= int(b) << (i * 2)
    return val


def decode_guide_64bit(val: int, length: int = 20) -> np.ndarray:
    """Unpack a 40-bit integer back to 2-bit encoded guide."""
    encoded = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        encoded[i] = (val >> (i * 2)) & 0x3
    return encoded


def hamming_distance_2bit(val1: int, val2: int, length: int = 20) -> int:
    """Count mismatches between two 2-bit packed sequences of given length."""
    xor_val = val1 ^ val2
    # For each 2-bit pair, detect if either bit differs
    mismatch_bits = (xor_val | (xor_val >> 1)) & 0x5555555555  # 0b01 mask for 20 positions
    return mismatch_bits.bit_count()


def mismatch_positions_2bit(val1: int, val2: int, length: int = 20) -> List[int]:
    """Return 1-indexed mismatch positions between two 2-bit packed sequences."""
    xor_val = val1 ^ val2
    mismatch_bits = (xor_val | (xor_val >> 1)) & 0x5555555555
    positions = []
    for i in range(length):
        if (mismatch_bits >> (i * 2)) & 1:
            positions.append(i + 1)
    return positions
