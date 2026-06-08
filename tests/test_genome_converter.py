"""Tests for genome converter."""
import tempfile
from pathlib import Path
from beguidedoff.genome_converter import convert, verify, load_genome


def _make_fasta(path: str, entries: dict):
    with open(path, "w") as f:
        for name, seq in entries.items():
            f.write(f">{name}\n{seq}\n")


def test_convert_and_load():
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "mini.fa"
        npz = Path(td) / "mini.npz"
        _make_fasta(fa, {"chr1": "ACGTACGTACGT", "chr2": "TGCATGCATGCA"})
        convert(str(fa), str(npz), verbose=False)
        data = load_genome(str(npz))
        assert data["total_bp"] == 24
        assert list(data["chrom_names"]) == ["chr1", "chr2"]
        assert "genome_rc" in data


def test_verify():
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "mini.fa"
        npz = Path(td) / "mini.npz"
        _make_fasta(fa, {"chr1": "ACGTACGTACGT"})
        convert(str(fa), str(npz), verbose=False)
        assert verify(str(npz), str(fa), sample_n=100, verbose=False)
