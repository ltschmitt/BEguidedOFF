"""Integration test for end-to-end pipeline."""
import tempfile
from pathlib import Path
from beguidedoff import beguidedoff, genome_converter
from beguidedoff.types import GuideInput


def _make_fasta(path: str, entries: dict):
    with open(path, "w") as f:
        for name, seq in entries.items():
            f.write(f">{name}\n{seq}\n")


def test_encode_search_roundtrip():
    """End-to-end: encode genome and search with a guide."""
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "genome.fa"
        npz = Path(td) / "genome.npz"

        # Guide on-target: ACGTACGTACGTACGTACGT + NGG
        seq = "ACGTACGTACGTACGTACGTGG" + "A" * 200
        # Add a 1-mismatch off-target with editable C in window 4-8
        # Window 4-8 in "ACGTACGTACGTACGTACGa" (mismatch at pos 19)
        # Let's make an off-target at position 50:
        ot = "ACGTACGTACGTACGTACGA" + "GG"  # PAM = GG -> matches NGG, but need 3-nt PAM
        # For NGG, the PAM is 3 nt. Let's use a 2-nt PAM for scoring but actual 3-nt PAM mask.
        # The genome at pos 70-72 must match NGG: any, G, G.
        # Let's construct: guide_seq + GG + next base = any.
        # So the genome segment at pos 50-72 should be: guide(20) + GG + X where X is anything.
        # The 3-nt PAM is GG(X), which matches NGG because N matches anything and G matches G.
        # Wait, NGG means position0=N (any), position1=G, position2=G.
        # So the PAM must be XGG where X is anything.
        # But my ot ends with "GG" (2 nt). I need 3 nt PAM: let's append "A" -> ot = 20bp + "AGG"
        # So the full 23-nt sequence at pos 50 is: 20bp guide + AGG.
        ot = "ACGTACGTACGTACGTACGA" + "AGG"
        # Insert the off-target into the genome at position 50
        genome = seq[:50] + ot + seq[50 + len(ot):]
        _make_fasta(fa, {"chr1": genome})

        genome_converter.convert(str(fa), str(npz), verbose=False)

        finder = beguidedoff(
            genome_path=str(npz),
            max_mismatches=4,
            scoring_method="CCTOP",
            device="cpu",
            n_top_offtargets=3,
        )

        guides = [
            GuideInput("g1", "ACGTACGTACGTACGTACGT", "NGG", "3prime", 4, 8, "CBE"),
        ]

        df = finder.search(guides=guides, verbose=False)
        # Should find the 1-mismatch off-target
        assert len(df) >= 1
        assert df.iloc[0]["n_mismatches"] == 1
