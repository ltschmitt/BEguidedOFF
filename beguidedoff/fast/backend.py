"""Backend abstraction for the fast (CPU) off-target engine.

The fast engine only needs three operations from a genome index:

* count / locate exact occurrences of a short DNA *anchor*, and
* extract the reference DNA at a global position.

Keeping this behind a small protocol lets the concrete index library change
(the production backend is sdsl-lite; see :mod:`beguidedoff.fast.fmindex`)
without touching the finder.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

# --- symbol codec -----------------------------------------------------------
#
# The index alphabet is 1..4 for A/C/G/T plus a separator symbol.  Anything that
# is not A/C/G/T is mapped to A -- this deliberately mirrors the 2-bit genome
# encoding used by the extensive mode (``NUC_TO_2BIT.get(base, 0)``), so that the
# two modes agree on genomes that contain N / soft-masked bases.
SEPARATOR = 6

_ENCODE = bytearray([1]) * 256  # default: A
for _base, _code in ((b"A", 1), (b"C", 2), (b"G", 3), (b"T", 4)):
    _ENCODE[_base[0]] = _code
    _ENCODE[_base[0] + 32] = _code  # lowercase
_ENCODE = bytes(_ENCODE)

_DECODE = bytearray(b"A") * 256
for _code, _base in ((1, b"A"), (2, b"C"), (3, b"G"), (4, b"T")):
    _DECODE[_code] = _base[0]
_DECODE = bytes(_DECODE)


def encode_symbols(sequence: str) -> bytes:
    """Map a DNA string to index symbols (A/C/G/T -> 1..4, else A)."""
    return sequence.encode("ascii", "replace").upper().translate(_ENCODE)


def decode_symbols(symbols: bytes) -> str:
    """Map index symbols back to a DNA string."""
    return bytes(symbols).translate(_DECODE).decode("ascii")


@runtime_checkable
class GenomeIndexBackend(Protocol):
    """Minimal read-only genome index interface used by the fast finder."""

    def count(self, anchor: bytes) -> int:
        """Number of exact occurrences of *anchor* (already symbol-encoded)."""

    def locate(self, anchor: bytes, max_hits: int = 0) -> np.ndarray:
        """Global 0-based offsets of *anchor* occurrences (``max_hits<=0`` = all)."""

    def locate_interval(self, l: int, r: int, max_hits: int = 0) -> np.ndarray:
        """Global offsets for a pre-computed SA interval ``[l, r]``."""

    def interval(self, anchor: bytes):
        """Return the SA interval ``(l, r)`` for *anchor* (``(1, 0)`` if none)."""

    def extract(self, pos: int, length: int) -> bytes:
        """Return *length* symbols starting at global offset *pos*."""
