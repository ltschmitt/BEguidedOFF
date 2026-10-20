"""CPU FM-index off-target search (fast mode).

The fast engine is a *sound filter*: for a CCTOP score cutoff ``T`` it enumerates
every genome site that could score below ``T`` by requiring the PAM-proximal
"cold" positions to match exactly (one CSA backward search) and then scoring the
full 20 bp window.  Any site with score below ``T`` must have an exact cold
region, so the remove/keep verdict agrees with the exhaustive (GPU) mode at the
same cutoff; sites above ``T`` are removed by exact scoring.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from beguidedoff.base_editing import editable_base_in_window, get_trinucleotide_contexts
from beguidedoff.encoding import (
    encode_dna,
    encode_guide_64bit,
    mismatch_positions_2bit,
    reverse_complement_seq,
)
from beguidedoff.fast.backend import encode_symbols
from beguidedoff.io import load_guides_from_tsv, offtargets_to_dataframe
from beguidedoff.pam import IUPAC_MASK
from beguidedoff.regions import derive_regions
from beguidedoff.scoring import calculate_cctop_score
from beguidedoff.types import GuideInput, OffTarget

_LOGGER = logging.getLogger("beguidedoff")

_GUIDE_LENGTH = 20
_NUC_TO_2BIT = {"A": 0, "C": 1, "G": 2, "T": 3}


def _pam_matches(genomic_pam: str, pattern: str) -> bool:
    """IUPAC-aware PAM match (genomic bases are concrete A/C/G/T)."""
    if len(genomic_pam) != len(pattern):
        return False
    for gb, pb in zip(genomic_pam, pattern):
        nuc = _NUC_TO_2BIT.get(gb, 0)
        if not ((1 << nuc) & IUPAC_MASK.get(pb, 0)):
            return False
    return True


def _mismatch_positions(guide_seq: str, site_seq: str) -> List[int]:
    gval = encode_guide_64bit(encode_dna(guide_seq))
    sval = encode_guide_64bit(encode_dna(site_seq))
    return mismatch_positions_2bit(gval, sval, length=_GUIDE_LENGTH)


@dataclass
class _Candidate:
    """A scored genome site matching one guide's cold anchor."""

    chrom: str
    chrom_idx: int
    local_leftmost: int  # 0-based, within chromosome
    position: int  # 1-based output position
    strand: str
    site_guide: str  # 20 bp, oriented as the guide (5'->3')
    genomic_pam: str
    n_mismatches: int
    mismatch_positions: List[int]
    score: float


@dataclass
class _ScanGuide:
    guide_seq: str
    pam: str
    _cache: Optional[Tuple[List[_Candidate], bool]] = None


class beguidedofffast:
    """Fast CPU off-target finder over a pre-built FM-index.

    Args:
        index_path: Path to an index built by
            :func:`beguidedoff.fast.fmindex.build_index`.
        search_cutoff: CCTOP score cutoff ``T``.  All sites with score ``<= T``
            are enumerated.  Default 5.0 (``remove_guides_score``).
        scoring_method: Only ``"CCTOP"`` is supported.
        n_top_offtargets: Number of worst (lowest-score) hits returned per guide.
        max_hits_per_anchor: Safety cap on occurrences enumerated per anchor
            (guards against low-complexity anchors).  ``<= 0`` disables the cap.
        verbose: Log progress.
    """

    def __init__(
        self,
        index_path: str = "",
        search_cutoff: float = 5.0,
        scoring_method: str = "CCTOP",
        n_top_offtargets: int = 3,
        max_hits_per_anchor: int = 10000,
        verbose: bool = False,
        index: Optional["SdslGenomeIndex"] = None,
    ):
        if scoring_method.upper() != "CCTOP":
            raise ValueError("fast mode only supports scoring_method='CCTOP'")

        self.index_path = str(index_path)
        self.search_cutoff = float(search_cutoff)
        self.scoring_method = "CCTOP"
        self.n_top_offtargets = int(n_top_offtargets)
        self.max_hits_per_anchor = int(max_hits_per_anchor)
        self.verbose = verbose

        if index is not None:
            # A preloaded index handle (e.g. from a server-side cache). The
            # handle is shared: locate/extract are read-only, so concurrent
            # finders may use the same object.
            self._index = index
        else:
            # Lazy import so that `import beguidedoff.fast` works without the
            # compiled extension (it is only required to actually run a search).
            from beguidedoff.fast.fmindex import SdslGenomeIndex

            if not self.index_path:
                raise ValueError(
                    "fast mode requires either index_path or a preloaded index"
                )
            self._index = SdslGenomeIndex.load(self.index_path)
        self._regions_cache: Dict[float, object] = {}

    # ------------------------------------------------------------------
    # index helpers
    # ------------------------------------------------------------------
    @property
    def total_bp(self) -> int:
        return self._index.total_bp  # type: ignore[attr-defined]

    @property
    def chrom_names(self) -> List[str]:
        return self._index.chrom_names  # type: ignore[attr-defined]

    def _regions(self, cutoff: float):
        key = round(cutoff, 9)
        if key not in self._regions_cache:
            self._regions_cache[key] = derive_regions(cutoff)
        return self._regions_cache[key]

    def _locate_capped(self, anchor: bytes, max_hits: int) -> Tuple[np.ndarray, bool]:
        if max_hits and max_hits > 0:
            l, r = self._index.interval(anchor)  # type: ignore[attr-defined]
            if r < l:
                return np.empty(0, dtype=np.int64), False
            count = r - l + 1
            overflow = count > max_hits
            hits = self._index.locate_interval(l, r, max_hits)  # type: ignore[attr-defined]
            return hits, overflow
        return self._index.locate(anchor, 0), False  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # candidate generation
    # ------------------------------------------------------------------
    def _generate_candidates(
        self, guide_seq: str, pam: str, cutoff: float
    ) -> Tuple[List[_Candidate], bool]:
        regions = self._regions(cutoff)
        cold0 = regions.cold_0based
        if not cold0:
            raise ValueError(
                f"search cutoff {cutoff} leaves no exact 'cold' region; fast mode "
                "cannot enumerate that many tolerated mismatches. Use a lower "
                "cutoff (the PAM-proximal 20 bp caps it at 1.2**20 ~= 38.3) or "
                "the extensive mode."
            )
        if cold0 != tuple(range(cold0[0], cold0[-1] + 1)):
            raise ValueError("fast mode requires a contiguous cold region")

        pam = pam.upper()
        pam_len = len(pam)
        window_len = _GUIDE_LENGTH + pam_len
        rc_guide = reverse_complement_seq(guide_seq)

        # The cold (PAM-proximal) anchor as it appears on the forward genome:
        #   + strand: guide[8:20]  (3' end of the guide block)
        #   - strand: rc(guide)[0:12] (5' end of the rc block)
        searches = [
            ("+", guide_seq[cold0[0]: cold0[-1] + 1], cold0[0]),
            ("-", rc_guide[: len(cold0)], 0),
        ]

        candidates: List[_Candidate] = []
        overflow = False
        for strand, anchor_dna, block_offset in searches:
            anchor_sym = encode_symbols(anchor_dna)
            hits, ovf = self._locate_capped(anchor_sym, self.max_hits_per_anchor)
            overflow = overflow or ovf

            for gpos in hits.tolist():
                if strand == "+":
                    leftmost = gpos - block_offset
                else:
                    leftmost = gpos - pam_len

                chrom_idx, _ = self._index.chromosome_of(leftmost)  # type: ignore[attr-defined]
                chrom_off = int(self._index.chrom_offsets[chrom_idx])  # type: ignore[attr-defined]
                chrom_len = int(self._index.chrom_lengths[chrom_idx])  # type: ignore[attr-defined]

                # The anchor and the full window must lie inside one chromosome.
                if gpos < chrom_off or gpos + len(anchor_sym) > chrom_off + chrom_len:
                    continue
                if leftmost < chrom_off or leftmost + window_len > chrom_off + chrom_len:
                    continue

                window = self._index.extract_dna(leftmost, window_len)  # type: ignore[attr-defined]
                if strand == "+":
                    site_guide = window[:_GUIDE_LENGTH]
                    site_pam = window[_GUIDE_LENGTH: window_len]
                else:
                    site_pam = reverse_complement_seq(window[:pam_len])
                    site_guide = reverse_complement_seq(window[pam_len:])

                if not _pam_matches(site_pam, pam):
                    continue

                mm = _mismatch_positions(guide_seq, site_guide)
                score = calculate_cctop_score(mm)
                if score > cutoff:
                    continue

                local_leftmost = leftmost - chrom_off
                if strand == "+":
                    position = local_leftmost + 1
                else:
                    position = local_leftmost + window_len

                candidates.append(
                    _Candidate(
                        chrom=self._index.chrom_names[chrom_idx],  # type: ignore[attr-defined]
                        chrom_idx=chrom_idx,
                        local_leftmost=local_leftmost,
                        position=position,
                        strand=strand,
                        site_guide=site_guide,
                        genomic_pam=site_pam,
                        n_mismatches=len(mm),
                        mismatch_positions=mm,
                        score=score,
                    )
                )
        return candidates, overflow

    # ------------------------------------------------------------------
    # on-target handling
    # ------------------------------------------------------------------
    def _on_target_key(self, guide: GuideInput) -> Optional[Tuple[int, int, str]]:
        if not guide.on_target_chrom or guide.on_target_pos is None:
            return None
        names = self._index.chrom_names  # type: ignore[attr-defined]
        chrom_idx = next((i for i, n in enumerate(names) if n == guide.on_target_chrom), None)
        if chrom_idx is None:
            return None
        if guide.on_target_strand is None:
            raise ValueError(
                f"Guide {guide.guide_id} has on-target chromosome and position but "
                f"no strand. Please provide on_target_strand ('+' or '-')."
            )
        window = _GUIDE_LENGTH + len(guide.pam)
        if guide.on_target_strand == "+":
            local_leftmost = guide.on_target_pos - 1
        else:
            local_leftmost = guide.on_target_pos - window
        return (chrom_idx, local_leftmost, guide.on_target_strand)

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
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
                presets_path = str(Path(__file__).parent.parent.parent / "base_editor_profiles.tsv")
            guides = load_guides_from_tsv(input_file, presets_path=presets_path)

        if not guides:
            raise ValueError("No guides provided")

        log = logger if logger is not None else _LOGGER

        for g in guides:
            if g.pam_position.lower() != "3prime":
                raise ValueError(
                    f"Guide {g.guide_id} has pam_position '{g.pam_position}'. "
                    f"Only 3prime PAM is supported."
                )

        # The search must cover every score we might need to resolve.  Removal
        # uses the score cutoff; counting uses ``count_score_threshold``.
        cutoff = self.search_cutoff
        if count_score_threshold is not None and count_score_threshold > cutoff:
            log.warning(
                "count_score_threshold=%s exceeds search_cutoff=%s; widening the "
                "search region (this is slower).",
                count_score_threshold,
                cutoff,
            )
            cutoff = float(count_score_threshold)

        effective_include_non_editable = return_all

        # Deduplicate the expensive genome scan by (guide_seq, pam).
        scan_guides: Dict[Tuple[str, str], List[GuideInput]] = {}
        for g in guides:
            scan_guides.setdefault((g.guide_seq, g.pam), []).append(g)

        if verbose:
            log.info(
                "Fast search: %d unique protospacers (from %d input rows), cutoff=%s",
                len(scan_guides),
                len(guides),
                cutoff,
            )

        group_results: Dict[str, List[OffTarget]] = {g.guide_id: [] for g in guides}
        below_threshold_counts: Dict[str, int] = {g.guide_id: 0 for g in guides}

        for scan_idx, ((guide_seq, pam), group) in enumerate(scan_guides.items(), start=1):
            candidates, overflow = self._generate_candidates(guide_seq, pam, cutoff)
            if overflow:
                log.warning(
                    "Anchor hit cap (%d) exceeded for guide %s (PAM=%s); results may "
                    "be incomplete.",
                    self.max_hits_per_anchor,
                    guide_seq,
                    pam,
                )
            if verbose and scan_idx % 200 == 0:
                log.info("  ... %d/%d protospacers scanned", scan_idx, len(scan_guides))

            for g in group:
                on_target = self._on_target_key(g)
                hits = group_results[g.guide_id]
                for cand in candidates:
                    if on_target is not None and (
                        cand.chrom_idx,
                        cand.local_leftmost,
                        cand.strand,
                    ) == on_target:
                        continue

                    editable_positions = editable_base_in_window(
                        cand.site_guide, g.editor_type, g.edit_window_start, g.edit_window_end
                    )
                    is_editable = bool(editable_positions)
                    if not effective_include_non_editable and not is_editable:
                        continue

                    contexts = get_trinucleotide_contexts(cand.site_guide, editable_positions)
                    hits.append(
                        OffTarget(
                            chromosome=cand.chrom,
                            position=cand.position,
                            strand=cand.strand,
                            offtarget_seq=cand.site_guide,
                            genomic_pam=cand.genomic_pam,
                            input_pam=pam,
                            guide_seq=guide_seq,
                            base_editor=g.base_editor or "",
                            n_mismatches=cand.n_mismatches,
                            mismatch_positions=cand.mismatch_positions,
                            score=cand.score,
                            editable_bases=editable_positions,
                            base_edit_context=contexts,
                        )
                    )
                    if (
                        count_score_threshold is not None
                        and is_editable
                        and cand.score <= count_score_threshold
                    ):
                        below_threshold_counts[g.guide_id] += 1

        dfs = []
        for g in guides:
            hits = group_results[g.guide_id]
            # Canonical tie-break so top-N is deterministic and identical
            # between modes when several hits share the same score.
            hits.sort(key=lambda x: (x.score, x.chromosome, x.position, x.strand))
            if not return_all:
                hits = hits[: self.n_top_offtargets]
            n_below = below_threshold_counts[g.guide_id]
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
