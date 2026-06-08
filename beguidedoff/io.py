"""Input parsing and output formatting."""
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union

from beguidedoff.types import GuideInput, OffTarget
from beguidedoff.base_editing import get_edit_window_from_preset


def load_guides_from_tsv(
    input_file: str,
    presets_path: Optional[str] = None,
) -> List[GuideInput]:
    """Load guides from a TSV file.

    Supports two formats:
    1. Full format with explicit columns: guide_seq, pam,
       edit_window_start, edit_window_end, editor_type (optional: chromosome,
       position, on_target_strand)
    2. Preset-based format (like input_data_example.tsv): guide_rna_sequence,
       be_preset (optional: chromosome, position, strand_used)

    guide_id is NOT read from the input -- it is auto-generated internally as
    row_0, row_1, ... so that the output pipeline can map results back to the
    correct input row without requiring a user-supplied identifier.

    pam_position is always "3prime" (5prime is not supported).

    Args:
        input_file: Path to TSV file.
        presets_path: Path to be_presets.tsv (required if using preset-based input).

    Returns:
        List of GuideInput objects.
    """
    df = pd.read_csv(input_file, sep="\t")
    guides = []

    # Determine format
    if "guide_seq" in df.columns:
        # Full explicit format
        for idx, row in df.iterrows():
            on_target_chrom = (
                str(row.get("chromosome", ""))
                if pd.notna(row.get("chromosome"))
                else None
            )
            on_target_pos = (
                int(row.get("position"))
                if pd.notna(row.get("position"))
                else None
            )
            on_target_strand = None
            if pd.notna(row.get("on_target_strand")):
                on_target_strand = str(row.get("on_target_strand"))
            guides.append(
                GuideInput(
                    guide_id=f"row_{idx}",
                    guide_seq=str(row["guide_seq"]).upper(),
                    pam=str(row["pam"]).upper(),
                    pam_position="3prime",
                    edit_window_start=int(row["edit_window_start"]),
                    edit_window_end=int(row["edit_window_end"]),
                    editor_type=str(row["editor_type"]).upper(),
                    on_target_chrom=on_target_chrom or None,
                    on_target_pos=on_target_pos,
                    on_target_strand=on_target_strand,
                    be_preset=str(row.get("be_preset", "")) or None,
                )
            )
    elif "guide_rna_sequence" in df.columns and "be_preset" in df.columns:
        # Preset-based format
        if presets_path is None:
            raise ValueError("presets_path is required when using preset-based input")
        for idx, row in df.iterrows():
            be_preset = str(row["be_preset"])
            edit_start, edit_end, pam, _cas, editing = get_edit_window_from_preset(
                be_preset, presets_path
            )
            # Infer editor type from editing column
            editor_type = editing.split("-")[0].upper()  # "A-to-G" -> "A"
            # Map to CBE/ABE/CGE
            if editor_type == "A":
                editor_type = "ABE"
            elif editor_type == "C":
                editor_type = "CBE"
            else:
                raise ValueError(f"Cannot infer editor type from editing: {editing}")

            on_target_chrom = (
                str(row.get("chromosome", ""))
                if pd.notna(row.get("chromosome"))
                else None
            )
            on_target_pos = (
                int(row.get("position"))
                if pd.notna(row.get("position"))
                else None
            )
            on_target_strand = None
            if pd.notna(row.get("strand_used")):
                on_target_strand = str(row.get("strand_used"))

            guides.append(
                GuideInput(
                    guide_id=f"row_{idx}",
                    guide_seq=str(row["guide_rna_sequence"]).upper(),
                    pam=pam.upper(),
                    pam_position="3prime",
                    edit_window_start=edit_start,
                    edit_window_end=edit_end,
                    editor_type=editor_type,
                    on_target_chrom=on_target_chrom or None,
                    on_target_pos=on_target_pos,
                    on_target_strand=on_target_strand,
                    be_preset=be_preset,
                )
            )
    else:
        raise ValueError(
            f"Input file must contain either (guide_seq, pam, ...) or "
            f"(guide_rna_sequence, be_preset) columns. Found: {list(df.columns)}"
        )
    return guides


def offtargets_to_dataframe(
    offtargets: List[OffTarget],
    n_offtargets_below_threshold: int = 0,
) -> pd.DataFrame:
    """Convert a flat list of OffTarget hits to a long-format DataFrame."""
    rows = []
    rank = 1
    for ot in offtargets:
        rows.append(
            {
                "guide_rna_sequence": ot.guide_seq,
                "be_preset": ot.be_preset,
                "ot_rank": rank,
                "ot_chromosome": ot.chromosome,
                "ot_position": ot.position,
                "ot_strand": ot.strand,
                "ot_sequence": ot.offtarget_seq,
                "ot_pam": ot.genomic_pam,
                "n_mismatches": ot.n_mismatches,
                "mismatch_positions": ",".join(map(str, ot.mismatch_positions)),
                "score": round(ot.score, 6),
                "n_offtargets_below_threshold": n_offtargets_below_threshold,
            }
        )
        rank += 1
    return pd.DataFrame(rows)


