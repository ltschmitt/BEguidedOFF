"""The fast finder must accept a pre-loaded index handle.

Long-lived processes (e.g. the BEguided web server) keep one open FM-index
per genome and share it across searches; ``beguidedofffast(index=...)`` is
the API seam for that. The results must be identical to a path-loaded
finder, and constructing without either ``index_path`` or ``index`` must
fail loudly.
"""
import tempfile
from pathlib import Path

from _fastutil import write_fasta
from beguidedoff.fast.finder import beguidedofffast
from beguidedoff.fast.fmindex import SdslGenomeIndex, build_index
from beguidedoff.types import GuideInput

import pytest

pytest.importorskip("beguidedoff._fmindex")

_CHR1 = (
    "ACGTACGTACGTGGTTGACAGCTAGCTAGCTAGCTGGGGTACAGCTAGCTAGCTAGC"
    "TGGGGTACAGCTAGCTAGCTAGCTGGGGTACAGCTAGCTAGCTAGCTGGGGACGTAC"
)
_GUIDES = [
    GuideInput("g0", "TTGACAGCTAGCTAGCTAGC", "NGG", "3prime", 4, 8, "ABE"),
    GuideInput("g1", "GTACAGCTAGCTAGCTAGCT", "NGG", "3prime", 4, 8, "CBE"),
]


def _write_genome(tmpdir: str) -> str:
    fa = Path(tmpdir) / "genome.fa"
    write_fasta(str(fa), {"chr1": _CHR1})
    return str(fa)


def test_preloaded_index_matches_path_load():
    with tempfile.TemporaryDirectory() as td:
        fa = _write_genome(td)
        index_path = str(Path(td) / "genome.fmi")
        build_index(fa, index_path, verbose=False)

        by_path = beguidedofffast(index_path, search_cutoff=5.0, verbose=False)
        handle = SdslGenomeIndex.load(index_path)
        preloaded = beguidedofffast(
            "", search_cutoff=5.0, verbose=False, index=handle
        )

        assert preloaded.index_path == ""
        assert preloaded.total_bp == by_path.total_bp

        df_path = by_path.search(guides=_GUIDES, verbose=False,
                                 count_score_threshold=5.0)
        df_pre = preloaded.search(guides=_GUIDES, verbose=False,
                                  count_score_threshold=5.0)
        cols = ["guide_rna_sequence", "ot_chromosome", "ot_position",
                "ot_strand", "score"]
        assert (
            df_path[cols].sort_values(cols).reset_index(drop=True)
            .equals(df_pre[cols].sort_values(cols).reset_index(drop=True))
        )


def test_missing_index_input_raises():
    with pytest.raises(ValueError, match="index_path or a preloaded index"):
        beguidedofffast("", verbose=False)
