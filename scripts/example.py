#!/usr/bin/env python
"""Example: encode a genome and search for off-targets.

guide_id is internal-only (auto-generated) and not exposed in input/output.
"""
from pathlib import Path
from beguidedoff import beguidedoff, genome_converter
from beguidedoff.types import GuideInput

# Paths
root = Path(__file__).parent.parent
relevant = root / "relevant_files"
fasta = relevant / "Homo_sapiens.GRCh38.dna.primary_assembly.fa"
npz = root / "hg38_otbe.npz"
profiles = root / "base_editor_profiles.tsv"

# Step 1: Encode genome (one-time)
if not npz.exists():
    print("Encoding genome...")
    genome_converter.convert(
        fasta_path=str(fasta),
        output_path=str(npz),
        include_rc=True,
        include_n_mask=True,
    )

# Step 2: Initialize finder (default scoring is CCTOP)
finder = beguidedoff(
    genome_path=str(npz),
    max_mismatches=4,
    scoring_method="CCTOP",
    n_top_offtargets=3,
)

# Step 3a: Search from file
results = finder.search(
    input_file=str(relevant / "input_example.tsv"),
    presets_path=str(profiles),
    verbose=True,
)
print(results.head())

# Step 3b: Search programmatically
# guide_id is auto-generated internally; any value can be passed here.
guides = [
    GuideInput(
        guide_id="internal_001",
        guide_seq="GGCAGTTACTGAACCAGACC",
        pam="NGG",
        pam_position="3prime",
        edit_window_start=3,
        edit_window_end=8,
        editor_type="ABE",
    ),
]
results2 = finder.search(guides=guides, verbose=True)
print(results2)
