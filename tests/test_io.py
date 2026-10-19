"""Tests for TSV loading, base editor profile parsing, and output formatting."""
import tempfile
from pathlib import Path

import pandas as pd

from beguidedoff.base_editing import (
    get_edit_window_from_profile,
)
from beguidedoff.io import (
    load_guides_from_tsv,
    offtargets_to_dataframe,
)
from beguidedoff.types import OffTarget


_PROFILE_TSV = (
    "name\tediting\tgeneral_edit_start\tgeneral_edit_end\t"
    "high_edit_start\thigh_edit_end\tpam\tcas\tsource\n"
    "ABE8e\tABE\t2\t13\t3\t8\tNGG\tSpCas9\tRichter2020\n"
    "CBE6d\tCBE\t2\t10\t6\t6\tNGG\tSpCas9\tZhang2024\n"
    "SaABE8e\tABE\t5\t13\t5\t13\tNNGRRT\tSaCas9\tRichter2020\n"
)


def _write_profile(path: Path) -> Path:
    path.write_text(_PROFILE_TSV)
    return path


def _make_guides_tsv(path: Path, header_line: str, rows: list) -> Path:
    lines = [header_line] + rows
    path.write_text("\n".join(lines) + "\n")
    return path


def test_get_edit_window_from_profile_basic():
    """get_edit_window_from_profile returns (start, end, pam) from new TSV."""
    with tempfile.TemporaryDirectory() as td:
        profiles_path = _write_profile(Path(td) / "profiles.tsv")
        gs, ge, pam = get_edit_window_from_profile("ABE8e", str(profiles_path))
        assert (gs, ge, pam) == (2, 13, "NGG")


def test_get_edit_window_from_profile_cbe():
    """CBE profiles look up C-to-T editing window correctly."""
    with tempfile.TemporaryDirectory() as td:
        profiles_path = _write_profile(Path(td) / "profiles.tsv")
        gs, ge, pam = get_edit_window_from_profile("CBE6d", str(profiles_path))
        assert (gs, ge, pam) == (2, 10, "NGG")


def test_get_edit_window_from_profile_unknown_raises():
    """Unknown profile name raises ValueError with helpful message."""
    with tempfile.TemporaryDirectory() as td:
        profiles_path = _write_profile(Path(td) / "profiles.tsv")
        try:
            get_edit_window_from_profile("DoesNotExist", str(profiles_path))
        except ValueError as exc:
            assert "DoesNotExist" in str(exc)
            assert "ABE8e" in str(exc)  # available profile listed
        else:
            raise AssertionError("Expected ValueError")


def test_load_guides_preset_format_new_columns():
    """Profile-based loader reads 'base_editor' column (new format)."""
    with tempfile.TemporaryDirectory() as td:
        profiles_path = _write_profile(Path(td) / "profiles.tsv")
        tsv = _make_guides_tsv(
            Path(td) / "guides.tsv",
            "chromosome\tposition\tstrand_used\tguide_rna_sequence\tbase_editor",
            [
                "1\t100\t+\tACGTACGTACGTACGTACGT\tABE8e",
                "2\t200\t-\tGGGGGGGGGGGGGGGGGGGG\tCBE6d",
            ],
        )
        guides = load_guides_from_tsv(str(tsv), presets_path=str(profiles_path))
        assert len(guides) == 2

        g0 = guides[0]
        assert g0.guide_seq == "ACGTACGTACGTACGTACGT"
        assert g0.pam == "NGG"
        assert g0.edit_window_start == 2
        assert g0.edit_window_end == 13
        assert g0.editor_type == "ABE"
        assert g0.base_editor == "ABE8e"
        assert g0.on_target_chrom == "1"
        assert g0.on_target_pos == 100
        assert g0.on_target_strand == "+"

        g1 = guides[1]
        assert g1.editor_type == "CBE"
        assert g1.edit_window_start == 2
        assert g1.edit_window_end == 10
        assert g1.base_editor == "CBE6d"


def test_load_guides_sa_abe_renamed_profile():
    """The renamed 'SaABE8e' profile loads correctly (was 'saABE8e')."""
    with tempfile.TemporaryDirectory() as td:
        profiles_path = _write_profile(Path(td) / "profiles.tsv")
        tsv = _make_guides_tsv(
            Path(td) / "guides.tsv",
            "guide_rna_sequence\tbase_editor",
            ["ACGTACGTACGTACGTACGT\tSaABE8e"],
        )
        guides = load_guides_from_tsv(str(tsv), presets_path=str(profiles_path))
        assert len(guides) == 1
        assert guides[0].base_editor == "SaABE8e"
        assert guides[0].pam == "NNGRRT"
        assert guides[0].editor_type == "ABE"
        assert guides[0].edit_window_start == 5
        assert guides[0].edit_window_end == 13


def test_load_guides_unknown_profile_raises():
    """Unknown profile name in TSV raises ValueError."""
    with tempfile.TemporaryDirectory() as td:
        profiles_path = _write_profile(Path(td) / "profiles.tsv")
        tsv = _make_guides_tsv(
            Path(td) / "guides.tsv",
            "guide_rna_sequence\tbase_editor",
            ["ACGTACGTACGTACGTACGT\tNoSuchEditor"],
        )
        try:
            load_guides_from_tsv(str(tsv), presets_path=str(profiles_path))
        except ValueError as exc:
            assert "NoSuchEditor" in str(exc)
        else:
            raise AssertionError("Expected ValueError")


def test_load_guides_missing_required_column_raises():
    """TSV without base_editor/guide_seq columns raises ValueError."""
    with tempfile.TemporaryDirectory() as td:
        profiles_path = _write_profile(Path(td) / "profiles.tsv")
        tsv = _make_guides_tsv(
            Path(td) / "guides.tsv",
            "foo\tbar",
            ["1\t2"],
        )
        try:
            load_guides_from_tsv(str(tsv), presets_path=str(profiles_path))
        except ValueError as exc:
            msg = str(exc)
            assert "guide_rna_sequence" in msg
            assert "base_editor" in msg
        else:
            raise AssertionError("Expected ValueError")


def test_load_guides_profile_based_requires_presets_path():
    """Profile-based TSV requires presets_path to be passed."""
    with tempfile.TemporaryDirectory() as td:
        tsv = _make_guides_tsv(
            Path(td) / "guides.tsv",
            "guide_rna_sequence\tbase_editor",
            ["ACGTACGTACGTACGTACGT\tABE8e"],
        )
        try:
            load_guides_from_tsv(str(tsv))
        except ValueError as exc:
            assert "presets_path" in str(exc)
        else:
            raise AssertionError("Expected ValueError")


def test_load_guides_explicit_format_still_works():
    """The 'explicit' format (guide_seq, pam, edit_window_*, editor_type) still works."""
    with tempfile.TemporaryDirectory() as td:
        tsv = _make_guides_tsv(
            Path(td) / "guides.tsv",
            "guide_seq\tpam\tedit_window_start\tedit_window_end\teditor_type",
            ["ACGTACGTACGTACGTACGT\tNGG\t4\t8\tCBE"],
        )
        guides = load_guides_from_tsv(str(tsv))
        assert len(guides) == 1
        assert guides[0].editor_type == "CBE"
        assert guides[0].pam == "NGG"
        assert guides[0].base_editor is None  # not set in explicit format


def test_offtargets_dataframe_uses_base_editor_column():
    """Output DataFrame uses 'base_editor' column name (renamed from 'be_preset')."""
    ot = OffTarget(
        chromosome="chr1",
        position=100,
        strand="+",
        offtarget_seq="ACGTACGTACGTACGTACGA",
        genomic_pam="AGG",
        input_pam="NGG",
        guide_seq="ACGTACGTACGTACGTACGT",
        base_editor="ABE8e",
        n_mismatches=1,
        mismatch_positions=[20],
        score=1.2,
        editable_bases=[5],
        base_edit_context=["TAC"],
    )
    df = offtargets_to_dataframe([ot], n_offtargets_below_threshold=3)
    assert "base_editor" in df.columns
    assert "be_preset" not in df.columns
    assert df.iloc[0]["base_editor"] == "ABE8e"
    assert df.iloc[0]["n_offtargets_below_threshold"] == 3


def test_guide_input_base_editor_field_default_none():
    """GuideInput.base_editor defaults to None when not supplied."""
    from beguidedoff.types import GuideInput

    g = GuideInput(
        guide_id="g1",
        guide_seq="ACGTACGTACGTACGTACGT",
        pam="NGG",
        pam_position="3prime",
        edit_window_start=4,
        edit_window_end=8,
        editor_type="ABE",
    )
    assert g.base_editor is None
