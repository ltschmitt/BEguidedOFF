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


def _n_runs(seq: str) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` half-open runs of N in *seq* (0-based local)."""
    arr = np.frombuffer(seq.upper().encode("ascii", "replace"), dtype=np.uint8)
    if arr.size == 0:
        return []
    mask = (arr == ord("N")).view(np.int8)
    if not mask.any():
        return []
    diff = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def build_index(
    fasta_path: str,
    output_path: str,
    verbose: bool = False,
) -> None:
    """Build a fast-mode FM-index from a FASTA genome.

    Writes ``output_path`` (the serialized sdsl CSA) plus two sidecars:
    ``output_path + ".meta"`` records the chromosome table and
    ``output_path + ".nmask"`` lists the N runs of the assembly in global
    index coordinates (0-based, half-open). The FM-index alphabet has no
    fifth symbol, so non-ACGT bases are indexed as A; the N mask is what
    lets the fast search apply the same N semantics as the extensive mode
    (a reference N never matches).

    Args:
        fasta_path: Reference genome FASTA.
        output_path: Destination path for the index (e.g. ``hg38.fmi``).
        verbose: Print per-chromosome progress.
    """
    fmi = _load_extension()
    fasta = _open_fasta(fasta_path)

    text_path = str(output_path) + ".seq.tmp"
    nmask_path = str(output_path) + ".nmask"
    names: List[str] = []
    offsets: List[int] = []
    lengths: List[int] = []

    global_offset = 0
    try:
        with open(text_path, "wb") as fh, open(nmask_path, "w") as nmask_fh:
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
                for start, end in _n_runs(seq):
                    nmask_fh.write(
                        f"{global_offset + start}\t{global_offset + end}\n"
                    )
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
        self._n_starts = np.empty(0, dtype=np.int64)
        self._n_ends = np.empty(0, dtype=np.int64)
        self.has_n_mask = False

    @classmethod
    def load(cls, index_path: str) -> "SdslGenomeIndex":
        fmi = _load_extension()
        handle = fmi.FMIndex.load(str(index_path))
        index = cls(handle, source=str(index_path))
        index._load_n_mask(index_path)
        return index

    def _load_n_mask(self, index_path: str) -> None:
        """Read the N-run sidecar written by :func:`build_index`, if present.

        Indices built before the sidecar existed have no N information; the
        finder then falls back to the historical behaviour (N treated as A).
        """
        nmask_path = str(index_path) + ".nmask"
        if not os.path.exists(nmask_path):
            return
        starts: List[int] = []
        ends: List[int] = []
        with open(nmask_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                s, e = line.split("\t")
                starts.append(int(s))
                ends.append(int(e))
        self._n_starts = np.asarray(starts, dtype=np.int64)
        self._n_ends = np.asarray(ends, dtype=np.int64)
        self.has_n_mask = True

    def n_in_range(self, start: int, length: int) -> Optional[np.ndarray]:
        """Boolean array of ``length`` marks: True where the genome has N.

        ``None`` when the index carries no N-run sidecar (pre-mask index),
        so callers can distinguish "no N in range" from "unknown".
        """
        if not self.has_n_mask:
            return None
        out = np.zeros(length, dtype=bool)
        end = start + length
        idx = int(np.searchsorted(self._n_ends, start, side="right"))
        while idx < len(self._n_starts) and self._n_starts[idx] < end:
            s = max(int(self._n_starts[idx]), start)
            e = min(int(self._n_ends[idx]), end)
            if e > s:
                out[s - start: e - start] = True
            idx += 1
        return out

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
