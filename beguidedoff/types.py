"""Core types for BEguidedOFF."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class GuideInput:
    """Input specification for a single guide RNA."""

    guide_id: str
    guide_seq: str  # 20bp, ACGT only
    pam: str  # 1-6 chars, IUPAC
    pam_position: str  # must be "3prime" (5prime is not supported)
    edit_window_start: int  # 1-indexed, from 5' end of guide
    edit_window_end: int  # 1-indexed, inclusive
    editor_type: str  # "CBE", "ABE", or "CGE"
    on_target_chrom: Optional[str] = None  # on-target chromosome (e.g. "1", "X")
    on_target_pos: Optional[int] = None    # on-target position (1-based)
    on_target_strand: Optional[str] = None  # "+" or "-"
    base_editor: Optional[str] = None  # base editor profile name (e.g. "ABE-NW1")


@dataclass
class OffTarget:
    """A single off-target hit."""

    chromosome: str
    position: int  # 1-based genomic position of the 5' end of the guide RNA (+ strand)
                    # for the - strand this is the rightmost position of the alignment
    strand: str  # "+" or "-"
    offtarget_seq: str  # 20bp off-target guide sequence
    genomic_pam: str  # full genomic PAM sequence that matched the pattern
    input_pam: str  # PAM pattern used for search (e.g. "NGG")
    guide_seq: str  # original on-target guide sequence
    base_editor: str  # base editor profile name
    n_mismatches: int
    mismatch_positions: List[int]  # 1-indexed mismatch positions
    score: float
    editable_bases: List[int]  # 1-indexed positions with editable base in window
    base_edit_context: List[str]  # trinucleotide contexts of editable bases


@dataclass
class GuideResult:
    """Results for a single guide RNA."""

    guide_id: str
    total_offtargets: int
    top_offtargets: List[OffTarget] = field(default_factory=list)
