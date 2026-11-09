"""Main BEguidedOFF search engine."""
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import logging

import numpy as np
import pandas as pd
from pathlib import Path

import torch

from beguidedoff.types import GuideInput, OffTarget
from beguidedoff.encoding import (
    encode_dna,
    encode_guide_64bit,
    extract_window,
    decode_dna,
    mismatch_positions_2bit,
    reverse_complement_seq,
)
from beguidedoff.pam import encode_pam_mask, pam_matches_at_position, group_guides_by_pam
from beguidedoff.scoring import calculate_cctop_score
from beguidedoff.base_editing import editable_base_in_window, get_trinucleotide_contexts
from beguidedoff.io import (
    load_guides_from_tsv,
    offtargets_to_dataframe,
)
from beguidedoff.genome_converter import load_genome

_LOGGER = logging.getLogger("beguidedoff")


def _position_to_chrom(pos: int, chrom_offsets: np.ndarray, chrom_lengths: np.ndarray) -> tuple:
    """Map a global position to (chrom_idx, local_pos).  local_pos is 0-based."""
    idx = int(np.searchsorted(chrom_offsets, pos, side="right") - 1)
    local = int(pos - chrom_offsets[idx])
    return idx, local


def _rc_pos_to_forward(
    rc_pos: int,
    pam_len: int,
    chrom_offsets: np.ndarray,
    chrom_lengths: np.ndarray,
) -> int:
    """Convert a position in the RC-packed genome to the corresponding
    forward (0-based) genome coordinate.

    For a 20-nt guide with a 3' PAM of length *pam_len* the search
    window length is ``20 + pam_len``.  The hit at *rc_pos* in the
    RC array maps to::

        leftmost_fwd = chrom_offset + chrom_length - local_rc - (20 + pam_len)
    """
    idx = int(np.searchsorted(chrom_offsets, rc_pos, side="right") - 1)
    local_rc = int(rc_pos - chrom_offsets[idx])
    window = 20 + pam_len
    local_fwd = int(chrom_lengths[idx] - local_rc - window)
    return int(chrom_offsets[idx] + local_fwd)


def _chrom_name_to_idx(chrom_name: str, chrom_names: np.ndarray) -> Optional[int]:
    """Return index of chromosome name, or None if not found."""
    for i, name in enumerate(chrom_names):
        if str(name) == chrom_name:
            return i
    return None


# ---------------------------------------------------------------------------
# Internal scan-guide representation (deduplicated)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ScanGuide:
    """Deduped search key — one genome scan per unique (seq, pam)."""
    guide_seq: str
    pam: str
    # Pre-computed on-target global position for 0-mm exclusion (0-based leftmost)
    on_target_chrom_idx: Optional[int] = None
    on_target_local_pos: Optional[int] = None
    on_target_strand: Optional[str] = None


# ---------------------------------------------------------------------------
# Numba CUDA strand search
# ---------------------------------------------------------------------------

try:
    from beguidedoff.cuda_search import search_genome_cuda
    from beguidedoff.cuda_search import MAX_GUIDES as _MAX_GUIDES_PER_LAUNCH
    _HAS_NUMBA_CUDA = True
except Exception:
    _HAS_NUMBA_CUDA = False
    _MAX_GUIDES_PER_LAUNCH = 2048

# Initial per-guide raw-hit buffer (also the constructor default). A guide
# whose <= max_mismatches matches exceed this is re-scanned automatically
# with a larger buffer; see ``_MAX_RESULTS_PER_GUIDE``.
_DEFAULT_RESULTS_PER_GUIDE = 10240
# Ceiling for the escalating re-scan (10240 * 16**2). A guide that still
# overflows at this size matches the genome extraordinarily often — with the
# N-mask gate active this means genuine repetitive sequence (poly-A/T tracts
# and similar). Enumeration stops there and the deterministic genome-order
# prefix of hits is kept: the affected guides are extreme multi-mappers that
# no off-target filter would pass, and the exact raw-hit count is logged so
# the truncation is loud, not silent.
_MAX_RESULTS_PER_GUIDE = 2_621_440
# Raw hits are decoded into one dict per hit on the host. When the buffer is
# escalated beyond the default, guides are processed in batches whose
# batch_size * results_per_guide product stays below this budget so a
# re-scan cannot exhaust host memory.
_DECODE_DICT_BUDGET = 4_000_000


def _search_strand_cuda(
    genome_packed: np.ndarray,
    total_bp: int,
    scan_guides: List[_ScanGuide],
    strand_label: str,
    max_mismatches: int,
    chrom_offsets: np.ndarray,
    chrom_lengths: np.ndarray,
    results_per_guide: int = 10240,
    n_mask_packed: Optional[np.ndarray] = None,
    n_encoded: int = 0,
) -> Tuple[dict, dict, dict]:
    """GPU search on one strand using Numba CUDA.

    Guides are processed in launches of at most ``_MAX_GUIDES_PER_LAUNCH``
    keys (the kernel loads them into shared memory); when the raw-hit buffer
    is escalated beyond the default, the batch size is additionally capped so
    the host-side decode of ``batch_size * results_per_guide`` hits stays
    bounded. Larger inputs are therefore split automatically instead of
    raising the historical "Too many guides" error.

    ``n_encoded`` is the 2-bit value an N base encodes as on this strand
    (0 = A on the forward-packed genome, 3 = T on the reverse complement);
    it drives the mismatch correction for guide windows overlapping N.

    Returns (results_dict, overflow_dict, counts_dict).
    results_dict: scan_key -> list of raw hit dicts.
    overflow_dict: scan_key -> True if results_per_guide was exceeded.
    counts_dict: scan_key -> exact number of raw matches on this strand
        (counted by the kernel even beyond the recorded buffer).
    """
    results: dict = {sg: [] for sg in scan_guides}
    overflow: dict = {sg: False for sg in scan_guides}
    counts: dict = {sg: 0 for sg in scan_guides}

    if not _HAS_NUMBA_CUDA:
        raise RuntimeError("Numba CUDA search is not available")

    if results_per_guide <= _DEFAULT_RESULTS_PER_GUIDE:
        batch_limit = _MAX_GUIDES_PER_LAUNCH
    else:
        batch_limit = max(1, _DECODE_DICT_BUDGET // results_per_guide)

    decode_map = {0: "A", 1: "C", 2: "G", 3: "T"}

    for batch_start in range(0, len(scan_guides), batch_limit):
        batch = scan_guides[batch_start : batch_start + batch_limit]
        n_guides = len(batch)
        guide_vals = np.zeros(n_guides, dtype=np.uint64)
        pam_masks_arr = np.zeros((n_guides, 6), dtype=np.uint8)
        pam_lens_arr = np.zeros(n_guides, dtype=np.uint8)
        pam_3prime_arr = np.zeros(n_guides, dtype=np.uint8)

        for i, sg in enumerate(batch):
            guide_vals[i] = encode_guide_64bit(encode_dna(sg.guide_seq))
            mask = encode_pam_mask(sg.pam)
            pam_masks_arr[i, : len(mask)] = mask
            pam_lens_arr[i] = len(sg.pam)
            pam_3prime_arr[i] = 1  # only 3prime is supported

        gpu_results, overflow_flags, count_flags = search_genome_cuda(
            genome_packed,
            total_bp,
            guide_vals,
            pam_masks_arr,
            pam_lens_arr,
            pam_3prime_arr,
            max_mismatches,
            return_overflow=True,
            results_per_guide=results_per_guide,
            n_mask_packed=n_mask_packed,
            n_encoded=n_encoded,
        )

        for g_idx, sg in enumerate(batch):
            pam_len = len(sg.pam)
            window = 20 + pam_len

            for guide_start, n_mm in gpu_results[g_idx]:
                if strand_label == "+":
                    leftmost_pos = int(guide_start)
                    report_pos = leftmost_pos
                else:
                    leftmost_pos = _rc_pos_to_forward(
                        int(guide_start), pam_len, chrom_offsets, chrom_lengths
                    )
                    # rightmost 0-based position for minus strand
                    report_pos = leftmost_pos + window - 1

                # Exclude on-target regardless of mismatch count
                if sg.on_target_chrom_idx is not None:
                    chrom_idx, local_pos = _position_to_chrom(
                        leftmost_pos, chrom_offsets, chrom_lengths
                    )
                    if (
                        chrom_idx == sg.on_target_chrom_idx
                        and local_pos == sg.on_target_local_pos
                        and strand_label == sg.on_target_strand
                    ):
                        continue

                # Extract 20-nt guide sequence from scanned genome
                offtarget_seq = decode_dna(
                    extract_window(genome_packed, int(guide_start), 20)
                )

                # Full genomic PAM (always 3prime in the scanned genome coords)
                pam_start_scanned = int(guide_start) + 20

                genomic_pam = ""
                for i in range(pam_len):
                    gpos = pam_start_scanned + i
                    word_idx = gpos // 16
                    bit_offset = (gpos % 16) * 2
                    nuc_2bit = int((genome_packed[word_idx] >> bit_offset) & 0x3)
                    genomic_pam += decode_map[nuc_2bit]

                results[sg].append(
                    {
                        "position": report_pos,  # 0-based global forward coord
                        "n_mismatches": n_mm,
                        "offtarget_seq": offtarget_seq,
                        "genomic_pam": genomic_pam,
                        "strand": strand_label,
                    }
                )

            overflow[sg] = overflow[sg] or overflow_flags.get(g_idx, False)
            counts[sg] += int(count_flags.get(g_idx, 0))

    return results, overflow, counts


# ---------------------------------------------------------------------------
# Main BEguidedOFF class
# ---------------------------------------------------------------------------

class beguidedoff:
    def __init__(
        self,
        genome_path: str,
        max_mismatches: int = 4,
        scoring_method: str = "CCTOP",
        device: Optional[str] = None,
        n_top_offtargets: int = 3,
        results_per_guide: int = 10240,
        max_results_per_guide: Optional[int] = None,
    ):
        self.genome_path = genome_path
        self.max_mismatches = max_mismatches
        if scoring_method.upper() != "CCTOP":
            raise ValueError("extensive mode only supports scoring_method='CCTOP'")
        self.scoring_method = "CCTOP"
        self.n_top_offtargets = n_top_offtargets
        self.results_per_guide = results_per_guide
        # Re-scan escalation ceiling. None -> fall back to the module-level
        # ``_MAX_RESULTS_PER_GUIDE`` (read at call time, so tests can
        # monkeypatch it).
        self.max_results_per_guide = max_results_per_guide

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if self.device.type != "cuda" or not _HAS_NUMBA_CUDA:
            raise RuntimeError(
                "Extensive mode requires a CUDA GPU (torch reports device "
                f"'{self.device}' and numba CUDA is "
                f"{'available' if _HAS_NUMBA_CUDA else 'unavailable'}). "
                "Use the fast CPU mode instead: "
                "`beguidedoff(mode='fast', index_path=...)`."
            )

        self.genome_data = load_genome(genome_path)
        self.total_bp = self.genome_data["total_bp"]
        self.chrom_names = [str(c) for c in self.genome_data["chrom_names"]]
        self.chrom_offsets = self.genome_data["chrom_offsets"]
        self.chrom_lengths = self.genome_data["chrom_lengths"]
        self._prepare_n_masks()

        self.genome_forward_t = torch.from_numpy(
            self.genome_data["genome_forward"].copy()
        ).to(self.device)

        if "genome_rc" in self.genome_data:
            self.genome_rc_t = torch.from_numpy(
                self.genome_data["genome_rc"].copy()
            ).to(self.device)
        else:
            self.genome_rc_t = None

    def _prepare_n_masks(self) -> None:
        """Build packed N-position bitmasks for both strands.

        The converter stores a bitmask of N positions (bit i = forward
        position i, ``np.packbits`` default big bit order). The GPU kernel
        consumes little bit order; the reverse-complement strand needs the
        mask reversed within each chromosome (the RC genome is reversed per
        chromosome, so an N at forward local position *l* sits at RC local
        position ``chrom_length - 1 - l``).

        Genomes encoded without an N mask (``--include-n-mask`` off) search
        ungated — the historical behaviour.
        """
        packed_big = self.genome_data.get("n_mask")
        if packed_big is None or np.asarray(packed_big).size == 0:
            self._n_mask_fwd = None
            self._n_mask_rc = None
            return
        packed_big = np.asarray(packed_big, dtype=np.uint8)
        bits = np.unpackbits(packed_big, bitorder="big", count=self.total_bp)
        if not bits.any():
            self._n_mask_fwd = None   # no N in this assembly: gate is a no-op
            self._n_mask_rc = None
            return
        fwd = np.packbits(bits, bitorder="little")
        self._n_mask_fwd = np.ascontiguousarray(
            np.pad(fwd, (0, 16), constant_values=0), dtype=np.uint8
        )
        rc_bits = np.empty(self.total_bp, dtype=np.uint8)
        for i, clen in enumerate(self.chrom_lengths):
            off = int(self.chrom_offsets[i])
            rc_bits[off : off + clen] = bits[off : off + clen][::-1]
        rc = np.packbits(rc_bits, bitorder="little")
        self._n_mask_rc = np.ascontiguousarray(
            np.pad(rc, (0, 16), constant_values=0), dtype=np.uint8
        )

    @property
    def genome_info(self) -> dict:
        return {
            "build": "unknown",
            "total_bp": self.total_bp,
            "n_chromosomes": len(self.chrom_names),
            "chromosomes": self.chrom_names,
        }

    def _resolve_chromosome(self, pos: int) -> str:
        """Map global 0-based position to chromosome name."""
        idx, _local = _position_to_chrom(pos, self.chrom_offsets, self.chrom_lengths)
        return self.chrom_names[idx]

    def _deduplicate_guides(self, guides: List[GuideInput]) -> Tuple[List[_ScanGuide], Dict[_ScanGuide, List[GuideInput]]]:
        """Deduplicate guides by (seq, pam).

        Returns:
            scan_guides: list of unique _ScanGuide objects
            mapping: scan_guide -> list of original GuideInput objects sharing it
        """
        mapping: Dict[_ScanGuide, List[GuideInput]] = defaultdict(list)

        for g in guides:
            # Resolve on-target chromosome index
            ot_chrom_idx = None
            ot_local_pos = None
            ot_strand = None
            if g.on_target_chrom and g.on_target_pos is not None:
                ot_chrom_idx = _chrom_name_to_idx(g.on_target_chrom, self.genome_data["chrom_names"])
                if ot_chrom_idx is not None:
                    if g.on_target_strand is None:
                        raise ValueError(
                            f"Guide {g.guide_id} has on-target chromosome and position but no strand. "
                            f"Please provide on_target_strand ('+' or '-')."
                        )
                    # Convert on-target position to leftmost 0-based local coordinate for exclusion
                    pam_len = len(g.pam)
                    window = 20 + pam_len
                    if g.on_target_strand == "+":
                        ot_local_pos = g.on_target_pos - 1
                    else:
                        # User provides rightmost 1-based position for minus strand
                        ot_local_pos = g.on_target_pos - window
                    ot_strand = g.on_target_strand

            sg = _ScanGuide(
                guide_seq=g.guide_seq,
                pam=g.pam,
                on_target_chrom_idx=ot_chrom_idx,
                on_target_local_pos=ot_local_pos,
                on_target_strand=ot_strand,
            )
            mapping[sg].append(g)

        scan_guides = list(mapping.keys())
        return scan_guides, mapping

    def _process_raw_hits(
        self,
        raw_hits_dict: Dict[_ScanGuide, List[dict]],
        mapping: Dict[_ScanGuide, List[GuideInput]],
        count_score_threshold: Optional[float],
        include_non_editable: bool,
    ) -> Tuple[Dict[str, List[OffTarget]], Dict[str, int]]:
        """Take raw search hits, score them, apply per-guide edit window, and map back.

        Returns:
            group_results: dict mapping guide_id -> list of OffTarget objects
            below_threshold_counts: dict mapping guide_id -> number of off-targets
                with editable bases and a CCTOP score <= count_score_threshold
        """
        group_results: Dict[str, List[OffTarget]] = defaultdict(list)
        below_threshold_counts: Dict[str, int] = defaultdict(int)

        for sg, hits in raw_hits_dict.items():
            for raw in hits:
                pos_0based_global = raw["position"]
                chrom = self._resolve_chromosome(pos_0based_global)
                idx, local = _position_to_chrom(pos_0based_global, self.chrom_offsets, self.chrom_lengths)

                # 1-based position for output
                pos_1based = local + 1

                offtarget_seq = raw["offtarget_seq"]
                genomic_pam = raw["genomic_pam"]
                n_mm = raw["n_mismatches"]

                guide_val = encode_guide_64bit(encode_dna(sg.guide_seq))
                window_val = encode_guide_64bit(encode_dna(offtarget_seq))
                mm_positions = mismatch_positions_2bit(guide_val, window_val, length=20)

                score = calculate_cctop_score(mm_positions)

                # Map to each original GuideInput that shares this scan key
                for g in mapping[sg]:
                    editable_positions = editable_base_in_window(
                        offtarget_seq, g.editor_type, g.edit_window_start, g.edit_window_end
                    )
                    is_editable = bool(editable_positions)
                    if not include_non_editable and not is_editable:
                        continue

                    contexts = get_trinucleotide_contexts(offtarget_seq, editable_positions)

                    ot = OffTarget(
                        chromosome=chrom,
                        position=pos_1based,  # 1-based output
                        strand=raw["strand"],
                        offtarget_seq=offtarget_seq,
                        genomic_pam=genomic_pam,
                        input_pam=sg.pam,
                        guide_seq=sg.guide_seq,
                        base_editor=g.base_editor or "",
                        n_mismatches=n_mm,
                        mismatch_positions=mm_positions,
                        score=score,
                        editable_bases=editable_positions,
                        base_edit_context=contexts,
                    )
                    group_results[g.guide_id].append(ot)

                    if count_score_threshold is not None and is_editable and score <= count_score_threshold:
                        below_threshold_counts[g.guide_id] += 1

        return group_results, below_threshold_counts

    def search(
        self,
        input_file: Optional[str] = None,
        guides: Optional[List[GuideInput]] = None,
        return_all: bool = False,
        count_score_threshold: Optional[float] = None,
        output_file: Optional[str] = None,
        include_non_editable: bool = False,
        presets_path: Optional[str] = None,
        verbose: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> pd.DataFrame:
        if (input_file is None) == (guides is None):
            raise ValueError("Exactly one of input_file or guides must be provided")

        if input_file is not None:
            if presets_path is None:
                presets_path = str(Path(__file__).parent.parent / "base_editor_profiles.tsv")
            guides = load_guides_from_tsv(input_file, presets_path=presets_path)

        if not guides:
            raise ValueError("No guides provided")

        log = logger if logger is not None else _LOGGER

        # Validate all guides have 3prime PAM (5prime is not supported)
        for g in guides:
            if g.pam_position.lower() != "3prime":
                raise ValueError(
                    f"Guide {g.guide_id} has pam_position '{g.pam_position}'. "
                    f"Only 3prime PAM is supported."
                )

        # When return_all is requested, also include sites without editable bases
        effective_include_non_editable = return_all

        scan_guides, mapping = self._deduplicate_guides(guides)

        if verbose:
            log.info(
                "Guides after deduplication: %d (from %d input rows)",
                len(scan_guides),
                len(guides),
            )

        if self.device.type != "cuda" or not _HAS_NUMBA_CUDA:
            raise RuntimeError(
                "Extensive mode requires a CUDA GPU (torch reports device "
                f"'{self.device}' and numba CUDA is "
                f"{'available' if _HAS_NUMBA_CUDA else 'unavailable'}). "
                "Use the fast CPU mode instead: "
                "`beguidedoff(mode='fast', index_path=...)`."
            )
        if verbose:
            log.info("Running GPU search on %s ...", self.device)
        raw_hits_dict, overflow, _hit_counts = self._search_gpu(
            scan_guides, mapping, log=log, verbose=verbose
        )
        truncated = [sg for sg, ovf in overflow.items() if ovf]
        if truncated:
            log.warning(
                "%d guide(s) returned truncated off-target lists (the "
                "re-scan escalation ceiling was reached); their off-target "
                "counts are lower bounds. See the per-guide warnings above.",
                len(truncated),
            )

        group_results, below_threshold_counts = self._process_raw_hits(
            raw_hits_dict, mapping, count_score_threshold, effective_include_non_editable
        )

        dfs = []
        for g in guides:
            gid = g.guide_id
            hits = group_results.get(gid, [])
            # CCTOP: lower = more concerning off-target (sort ascending)
            # Canonical tie-break so top-N is deterministic and identical
            # between modes when several hits share the same score.
            hits.sort(key=lambda x: (x.score, x.chromosome, x.position, x.strand))
            if not return_all:
                hits = hits[: self.n_top_offtargets]
            n_below = below_threshold_counts.get(gid, 0)
            if hits:
                dfs.append(offtargets_to_dataframe(hits, n_offtargets_below_threshold=n_below))
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.DataFrame()
        if output_file:
            df.to_csv(output_file, sep="\t", index=False)
            if verbose:
                log.info("Wrote results to %s", output_file)
        return df

    def _search_gpu(
        self,
        scan_guides: List[_ScanGuide],
        mapping: Dict[_ScanGuide, List[GuideInput]],
        log: Optional[logging.Logger],
        verbose: bool,
    ) -> Tuple[Dict[_ScanGuide, List[dict]], Dict[_ScanGuide, bool]]:
        """GPU search using Numba CUDA.

        Inputs larger than one kernel launch (``_MAX_GUIDES_PER_LAUNCH``
        unique scan keys) are split into several launches automatically.

        A guide whose raw-hit buffer overflows is **re-scanned** with a 16x
        larger buffer; its raw hits are replaced, so the returned hit list is
        exact. Escalation repeats until every guide fits or
        ``_MAX_RESULTS_PER_GUIDE`` is reached — in the latter case the
        genome-order prefix of hits is kept and a warning with the exact raw
        hit count is logged (guides that hit this ceiling are extreme
        multi-mappers; see the ``max_results_per_guide`` constructor arg).
        """
        raw_combined: Dict[_ScanGuide, List[dict]] = {sg: [] for sg in scan_guides}
        overflow_combined: Dict[_ScanGuide, bool] = {sg: False for sg in scan_guides}
        counts_combined: Dict[_ScanGuide, int] = {sg: 0 for sg in scan_guides}

        pending = list(scan_guides)
        results_per_guide = self.results_per_guide
        while pending:
            round_raw, round_overflow, round_counts = self._scan_strands(
                pending, results_per_guide, log=log, verbose=verbose
            )
            # Replace (not extend): a re-scan covers the whole genome, so its
            # hits supersede anything recorded before.
            for sg in pending:
                raw_combined[sg] = round_raw[sg]
                overflow_combined[sg] = round_overflow[sg]
                counts_combined[sg] = round_counts[sg]
            pending = [sg for sg in pending if round_overflow[sg]]
            if not pending:
                break
            cap = (self.max_results_per_guide if self.max_results_per_guide is not None
                   else _MAX_RESULTS_PER_GUIDE)
            if results_per_guide >= cap:
                # Ceiling reached: keep the deterministic genome-order prefix
                # of hits instead of failing the whole search. The kernel
                # counted every match exactly, so the true raw-hit count is
                # known and logged; the affected guides are extreme
                # multi-mappers that no off-target filter would pass anyway.
                for sg in sorted(pending, key=lambda s: s.guide_seq):
                    log.warning(
                        "Truncated off-target enumeration for guide %s "
                        "(PAM=%s): %d raw genome-wide hits exceed the "
                        "per-strand buffer of %d; keeping the first %d "
                        "(genome order). The off-target count for this "
                        "guide is a lower bound.",
                        sg.guide_seq, sg.pam, counts_combined[sg],
                        results_per_guide, len(raw_combined[sg]),
                    )
                break
            results_per_guide = min(results_per_guide * 16, cap)
            if log is not None:
                names = ", ".join(sorted({sg.guide_seq for sg in pending})[:5])
                log.warning(
                    "Hit buffer overflow for %d guide(s) at results_per_guide=%s "
                    "(e.g. %s); re-scanning them with results_per_guide=%d.",
                    len(pending),
                    self.results_per_guide,
                    names,
                    results_per_guide,
                )

        return raw_combined, overflow_combined, counts_combined

    def _scan_strands(
        self,
        scan_guides: List[_ScanGuide],
        results_per_guide: int,
        log: Optional[logging.Logger],
        verbose: bool,
    ) -> Tuple[Dict[_ScanGuide, List[dict]], Dict[_ScanGuide, bool], Dict[_ScanGuide, int]]:
        """Run the kernel over both strands once with the given buffer size."""
        raw: Dict[_ScanGuide, List[dict]] = {sg: [] for sg in scan_guides}
        overflow: Dict[_ScanGuide, bool] = {sg: False for sg in scan_guides}
        counts: Dict[_ScanGuide, int] = {sg: 0 for sg in scan_guides}

        strands = [("+", self.genome_data["genome_forward"], self._n_mask_fwd)]
        if "genome_rc" in self.genome_data:
            strands.append(("-", self.genome_data["genome_rc"], self._n_mask_rc))

        for strand_label, genome_packed, n_mask in strands:
            if verbose:
                log.info("  GPU searching %s strand ...", strand_label)
            strand_results, strand_overflow, strand_counts = _search_strand_cuda(
                genome_packed,
                self.total_bp,
                scan_guides,
                strand_label,
                self.max_mismatches,
                self.chrom_offsets,
                self.chrom_lengths,
                results_per_guide=results_per_guide,
                n_mask_packed=n_mask,
                n_encoded=0 if strand_label == "+" else 3,
            )
            for sg, hits in strand_results.items():
                raw[sg].extend(hits)
            for sg, ovf in strand_overflow.items():
                if ovf:
                    overflow[sg] = True
            for sg, cnt in strand_counts.items():
                counts[sg] += cnt

        return raw, overflow, counts
