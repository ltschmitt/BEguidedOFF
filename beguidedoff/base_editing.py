"""Base editing window logic."""
from typing import List, Tuple

from beguidedoff.encoding import BIT_TO_NUC

_GUIDE_LENGTH = 20


def editable_base_in_window(
    offtarget_seq: str, editor_type: str, edit_window_start: int, edit_window_end: int
) -> List[int]:
    """Return 1-indexed positions within the editing window that contain an editable base.

    Args:
        offtarget_seq: 20bp off-target sequence (5'->3', DNA).
        editor_type: "CBE", "ABE", or "CGE".
        edit_window_start: 1-indexed start of editing window from 5' end.
        edit_window_end: 1-indexed inclusive end.

    Returns:
        List of 1-indexed positions with editable bases.
    """
    seq = offtarget_seq.upper()
    positions = []
    target_base = {"CBE": "C", "ABE": "A", "CGE": "C"}.get(editor_type)
    if target_base is None:
        raise ValueError(f"Unknown editor_type: {editor_type}")

    for i in range(edit_window_start - 1, edit_window_end):
        if i < len(seq) and seq[i] == target_base:
            positions.append(i + 1)  # 1-indexed
    return positions


def get_trinucleotide_contexts(seq: str, positions: List[int]) -> List[str]:
    """Return trinucleotide context for each 1-indexed position."""
    seq = seq.upper()
    contexts = []
    for pos in positions:
        # 0-indexed center
        idx = pos - 1
        left = seq[idx - 1] if idx > 0 else "N"
        center = seq[idx]
        right = seq[idx + 1] if idx < len(seq) - 1 else "N"
        contexts.append(left + center + right)
    return contexts


def has_editable_base(offtarget_seq: str, editor_type: str, edit_window_start: int, edit_window_end: int) -> bool:
    """Quick check if off-target has at least one editable base in window."""
    return len(editable_base_in_window(offtarget_seq, editor_type, edit_window_start, edit_window_end)) > 0


def get_edit_window_from_profile(base_editor: str, profiles_path: str) -> Tuple[int, int, str]:
    """Look up editing window from a base_editor_profiles.tsv file.

    Args:
        base_editor: Profile name (must match a row in the ``name`` column).
        profiles_path: Path to the base editor profiles TSV.

    Returns:
        (general_edit_start, general_edit_end, pam).

    Raises:
        ValueError: If the profile name is not found, or required columns are missing.
    """
    import pandas as pd

    required_cols = {"name", "general_edit_start", "general_edit_end", "pam"}
    df = pd.read_csv(profiles_path, sep="\t")
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Base editor profiles file {profiles_path} is missing required columns: "
            f"{sorted(missing)}"
        )

    row = df[df["name"] == base_editor]
    if row.empty:
        available = ", ".join(sorted(df["name"].head(5).tolist()))
        more = "..." if len(df) > 5 else ""
        raise ValueError(
            f"Unknown base editor '{base_editor}' in {profiles_path}. "
            f"Available profiles include: {available}{more}"
        )
    row = row.iloc[0]
    return (
        int(row["general_edit_start"]),
        int(row["general_edit_end"]),
        str(row["pam"]),
    )
