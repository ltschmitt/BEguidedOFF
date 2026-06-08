"""PAM encoding and validation."""
import re
from typing import Dict, List, Tuple
import numpy as np

# IUPAC degenerate base → 4-bit mask [bit3=T, ..., bit0=A]
IUPAC_MASK = {
    "A": 0b0001,  # Adenine
    "C": 0b0010,  # Cytosine
    "G": 0b0100,  # Guanine
    "T": 0b1000,  # Thymine
    "R": 0b0101,  # Purine (A/G)
    "Y": 0b1010,  # Pyrimidine (C/T)
    "M": 0b0011,  # Amino (A/C)
    "K": 0b1100,  # Keto (G/T)
    "S": 0b0110,  # Strong (G/C)
    "W": 0b1001,  # Weak (A/T)
    "H": 0b1011,  # Not G (A/C/T)
    "B": 0b1110,  # Not A (C/G/T)
    "V": 0b0111,  # Not T (A/C/G)
    "D": 0b1101,  # Not C (A/G/T)
    "N": 0b1111,  # Any
}

VALID_IUPAC = set(IUPAC_MASK.keys())


def validate_pam(pam: str) -> None:
    """Validate a PAM sequence."""
    pam = pam.upper()
    if not (1 <= len(pam) <= 6):
        raise ValueError(f"PAM must be 1-6 characters, got {len(pam)}")
    invalid = set(pam) - VALID_IUPAC
    if invalid:
        raise ValueError(f"Invalid IUPAC codes in PAM: {invalid}")


def encode_pam_mask(pam: str) -> np.ndarray:
    """Encode a PAM string to a uint8 mask array of length 6 (zero-padded)."""
    pam = pam.upper()
    validate_pam(pam)
    mask = np.zeros(6, dtype=np.uint8)
    for i, base in enumerate(pam):
        mask[i] = IUPAC_MASK[base]
    return mask


def pam_matches_at_position(
    genome_2bit: np.ndarray, position: int, pam_mask: np.ndarray, pam_len: int
) -> bool:
    """Check if the genome at `position` matches the PAM mask."""
    for i in range(pam_len):
        gpos = position + i
        if gpos >= len(genome_2bit):
            return False
        nuc_2bit = int(genome_2bit[gpos])
        if not ((1 << nuc_2bit) & pam_mask[i]):
            return False
    return True


def group_guides_by_pam(guides: List) -> Dict[Tuple[str, str], List]:
    """Group guide objects by (pam, pam_position) for efficient batching."""
    from collections import defaultdict

    groups = defaultdict(list)
    for g in guides:
        key = (g.pam.upper(), g.pam_position.lower())
        groups[key].append(g)
    return dict(groups)


def all_possible_pams(pam: str) -> List[str]:
    """Expand a degenerate PAM to all possible concrete sequences."""
    pam = pam.upper()
    validate_pam(pam)

    # IUPAC expansion
    expansion = {
        "A": ["A"],
        "C": ["C"],
        "G": ["G"],
        "T": ["T"],
        "R": ["A", "G"],
        "Y": ["C", "T"],
        "M": ["A", "C"],
        "K": ["G", "T"],
        "S": ["G", "C"],
        "W": ["A", "T"],
        "H": ["A", "C", "T"],
        "B": ["C", "G", "T"],
        "V": ["A", "C", "G"],
        "D": ["A", "G", "T"],
        "N": ["A", "C", "G", "T"],
    }

    results = [""]
    for base in pam:
        new_results = []
        for seq in results:
            for b in expansion[base]:
                new_results.append(seq + b)
        results = new_results
    return results
