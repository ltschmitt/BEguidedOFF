"""Integration test: build an FM-index and search with a guide (fast mode)."""
import tempfile
from pathlib import Path

from _fastutil import make_fast_finder, write_fasta
from beguidedoff.types import GuideInput


def test_build_index_and_search_roundtrip():
    """End-to-end fast mode: build the index and find a PAM-distal off-target."""
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"

        guide = "ACGTACGTACGTACGTACGT"
        # On-target (excluded via coordinates) + one PAM-distal off-target.
        seq = guide + "AGG" + "A" * 200
        ot = guide[:3] + "A" + guide[4:] + "AGG"  # mismatch at pos 4
        genome = seq[:50] + ot + seq[50 + len(ot):]
        write_fasta(str(fa), {"chr1": genome})

        finder = make_fast_finder(str(fa), td, search_cutoff=5.0, n_top_offtargets=3)
        guides = [
            GuideInput("g1", guide, "NGG", "3prime", 4, 8, "CBE",
                       on_target_chrom="chr1", on_target_pos=1, on_target_strand="+"),
        ]
        df = finder.search(guides=guides, verbose=False)
        assert len(df) >= 1
        assert df.iloc[0]["n_mismatches"] == 1
        assert df.iloc[0]["ot_position"] == 51


def test_factory_fast_mode():
    """The public factory dispatches to fast mode."""
    from beguidedoff import beguidedoff as factory
    from beguidedoff.fast.finder import beguidedofffast

    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        write_fasta(str(fa), {"chr1": "ACGT" * 50})
        make_fast_finder(str(fa), td)  # writes td/genome.fmi
        finder = factory(mode="fast", index_path=str(Path(td) / "genome.fmi"), n_top_offtargets=1)
        assert isinstance(finder, beguidedofffast)
