"""Concrete fast-mode genome index backed by sdsl-lite (``beguidedoff._fmindex``).

The compiled extension is imported lazily so that ``import beguidedoff.fast``
works even on installations that do not ship the FM-index binding.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from beguidedoff.fast.backend import (
    SEPARATOR,
    decode_symbols,
    encode_symbols,
)

_LOGGER = logging.getLogger("beguidedoff")

_EXTENSION_HINT = (
    "The BEguidedOFF FM-index extension ('beguidedoff._fmindex') is not "
    "available. Install the fast extra (requires sdsl-lite), e.g. "
    "`pip install 'beguidedoff[fast]'`, or build it from source."
)


def _load_extension():
    try:
        from beguidedoff import _fmindex  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on build env
        raise ImportError(_EXTENSION_HINT) from exc
    return _fmindex


def _open_fasta(fasta_path: str):
    import pysam

    try:
        return pysam.FastaFile(fasta_path)
    except (ValueError, OSError):
        # No .fai index yet -- build one, then retry.
        pysam.faidx(fasta_path)
        return pysam.FastaFile(fasta_path)


def build_index(
    fasta_path: str,
    output_path: str,
    verbose: bool = False,
) -> None:
    """Build a fast-mode FM-index from a FASTA genome.

    Writes ``output_path`` (the serialized sdsl CSA) and a sidecar
    ``output_path + ".meta"`` recording the chromosome table.

    Args:
        fasta_path: Reference genome FASTA.
        output_path: Destination path for the index (e.g. ``hg38.fmi``).
        verbose: Print per-chromosome progress.
    """
    fmi = _load_extension()
    fasta = _open_fasta(fasta_path)

    text_path = str(output_path) + ".seq.tmp"
    names: List[str] = []
    offsets: List[int] = []
    lengths: List[int] = []

    global_offset = 0
    try:
        with open(text_path, "wb") as fh:
            first = True
            for chrom in fasta.references:
                seq = fasta.fetch(chrom)
                symbols = encode_symbols(seq)
                if not first:
                    fh.write(bytes([SEPARATOR]))
                    global_offset += 1
                names.append(chrom)
                offsets.append(global_offset)
                lengths.append(len(symbols))
                fh.write(symbols)
                global_offset += len(symbols)
                first = False
                if verbose:
                    _LOGGER.info("  indexed %s (%d bp)", chrom, len(symbols))

        _LOGGER.info("Constructing CSA (%d bp total) ...", global_offset)
        fmi.FMIndex.build(
            text_path,
            str(output_path),
            names,
            np.asarray(offsets, dtype=np.int64),
            np.asarray(lengths, dtype=np.int64),
        )
    finally:
        if os.path.exists(text_path):
            os.remove(text_path)

    if verbose:
        _LOGGER.info("Wrote %s (+ .meta)", output_path)


class SdslGenomeIndex:
    """Read-only genome index backed by the sdsl-lite FM-index binding."""

    def __init__(self, handle, source: Optional[str] = None):
        self._handle = handle
        self.source = source
        self._chrom_names = [str(n) for n in handle.chrom_names()]
        self._chrom_offsets = np.asarray(handle.chrom_offsets(), dtype=np.int64)
        self._chrom_lengths = np.asarray(handle.chrom_lengths(), dtype=np.int64)

    @classmethod
    def load(cls, index_path: str) -> "SdslGenomeIndex":
        fmi = _load_extension()
        handle = fmi.FMIndex.load(str(index_path))
        return cls(handle, source=str(index_path))

    # -- metadata ----------------------------------------------------------
    @property
    def chrom_names(self) -> List[str]:
        return self._chrom_names

    @property
    def chrom_offsets(self) -> np.ndarray:
        return self._chrom_offsets

    @property
    def chrom_lengths(self) -> np.ndarray:
        return self._chrom_lengths

    @property
    def total_bp(self) -> int:
        return int(self._handle.size()) - 1  # sdsl index includes the sentinel

    def chromosome_of(self, position: int) -> Tuple[int, int]:
        """Map a global 0-based position to ``(chrom_idx, local_pos)``."""
        idx = int(np.searchsorted(self._chrom_offsets, position, side="right") - 1)
        local = int(position - self._chrom_offsets[idx])
        return idx, local

    def chromosome_name(self, position: int) -> str:
        idx, _ = self.chromosome_of(position)
        return self._chrom_names[idx]

    # -- queries -----------------------------------------------------------
    def count(self, anchor: bytes) -> int:
        return int(self._handle.count(anchor))

    def interval(self, anchor: bytes):
        l, r = self._handle.interval(anchor)
        return int(l), int(r)

    def locate(self, anchor: bytes, max_hits: int = 0) -> np.ndarray:
        return np.asarray(self._handle.locate(anchor, int(max_hits)), dtype=np.int64)

    def locate_interval(self, l: int, r: int, max_hits: int = 0) -> np.ndarray:
        return np.asarray(
            self._handle.locate_interval(int(l), int(r), int(max_hits)),
            dtype=np.int64,
        )

    def extract(self, pos: int, length: int) -> bytes:
        return bytes(self._handle.extract(int(pos), int(length)))

    def extract_dna(self, pos: int, length: int) -> str:
        return decode_symbols(self.extract(pos, length))
