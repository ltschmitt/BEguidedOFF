#!/usr/bin/env python
"""Generate benchmark guide sets.

Sets: 10 / 50 / 100 / 500 unique protospacers (real guides from
``input_example.tsv`` first, padded with genomic NGG protospacers sampled from
the FM-index), plus ``500x38`` (the 500-set expanded over all 38 base-editor
profiles = 19,000 input rows).

Sampled guides are prefiltered so that neither strand's 12-mer cold anchor
occurs more than ``MAX_ANCHOR_OCC`` times (guards the fast mode's hit cap and
keeps extensive's output finite).

Outputs: ``nogit/bench/guides/guides_<name>.tsv`` (profile-format columns).
"""
import dataclasses
import random
import re
import sys
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parent    # this dir: benchmarks/ (scripts + reports)
NOGIT = BENCH.parent / "nogit"             # local heavy data (not in the repository)
REPO = BENCH.parent
sys.path.insert(0, str(REPO))

from beguidedoff.encoding import reverse_complement_seq  # noqa: E402
from beguidedoff.fast.backend import encode_symbols  # noqa: E402
from beguidedoff.fast.fmindex import SdslGenomeIndex  # noqa: E402
from beguidedoff.io import load_guides_from_tsv  # noqa: E402

INDEX = NOGIT / "hg38_beguidedoff.fmi"
INPUT = NOGIT / "relevant_files" / "input_example.tsv"
PROFILES = REPO / "base_editor_profiles.tsv"
OUT = NOGIT / "bench" / "guides"

MAX_ANCHOR_OCC = 2000
SET_SIZES = [10, 50, 100, 500]
SEED = 42


def _canon(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def real_guides():
    """Unique real guides with on-target info (strand column patched in).

    Editor names are normalized against the profile file (the example input
    spells ``SaABE8e`` as the older ``Sa-ABE8e``).
    """
    profiles = pd.read_csv(PROFILES, sep="\t")
    canon = {_canon(n): str(n) for n in profiles["name"]}
    raw = pd.read_csv(INPUT, sep="\t")
    raw["base_editor"] = [
        canon.get(_canon(e), str(e)) for e in raw["base_editor"]
    ]
    norm_path = OUT / "_input_example_normalized.tsv"
    OUT.mkdir(exist_ok=True)
    raw.to_csv(norm_path, sep="\t", index=False)

    guides = load_guides_from_tsv(str(norm_path), presets_path=str(PROFILES))
    for i, g in enumerate(guides):  # io.py reads 'strand_used'; the file uses 'strand'
        row = raw.iloc[int(g.guide_id.split("_")[1])]
        strand = str(row.get("strand", "")).strip()
        if strand in ("+", "-"):
            guides[i] = dataclasses.replace(g, on_target_strand=strand)
    uniq = {}
    for g in guides:
        uniq.setdefault((g.guide_seq, g.pam), g)
    return list(uniq.values())


def anchor_ok(index, guide: str) -> bool:
    for anchor in (guide[8:20], reverse_complement_seq(guide)[:12]):
        lo, hi = index.interval(encode_symbols(anchor))
        if hi - lo + 1 > MAX_ANCHOR_OCC:
            return False
    return True


def sample_guides(index, n, rng, exclude_seqs):
    """Sample genomic NGG protospacers with bounded anchor occurrence counts."""
    names = index.chrom_names
    offsets = index.chrom_offsets
    lengths = index.chrom_lengths
    weights = [int(x) for x in lengths]
    seen = set(exclude_seqs)
    out = []
    tries = 0
    while len(out) < n and tries < n * 2000:
        tries += 1
        c = rng.choices(range(len(names)), weights=weights)[0]
        local = rng.randint(1000, int(lengths[c]) - 1100)
        gpos = int(offsets[c]) + local
        w = index.extract_dna(gpos, 23)
        if len(w) != 23 or w[21:23] != "GG":
            continue
        guide = w[:20]
        if guide in seen or not anchor_ok(index, guide):
            continue
        seen.add(guide)
        out.append({"chrom": names[c], "pos": local + 1, "strand": "+", "guide": guide})
    if len(out) < n:
        raise SystemExit(f"only sampled {len(out)}/{n} guides after {tries} tries")
    return out


def write_tsv(path, rows):
    df = pd.DataFrame(rows, columns=["chromosome", "position", "strand_used",
                                     "guide_rna_sequence", "base_editor"])
    df.to_csv(path, sep="\t", index=False)
    print(f"wrote {path} ({len(df)} rows)")


def main():
    OUT.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    index = SdslGenomeIndex.load(str(INDEX))
    reals_all = real_guides()
    # Drop guides whose cold anchor is ultra-repetitive: they overflow both
    # engines' hit caps (fast: max_hits_per_anchor, extensive: results_per_guide)
    # and are documented as a known limitation, not a soundness case.
    reals = [g for g in reals_all if anchor_ok(index, g.guide_seq)]
    dropped = len(reals_all) - len(reals)
    print(f"real unique protospacers: {len(reals)} "
          f"({dropped} dropped: 12-mer anchor occurs > {MAX_ANCHOR_OCC} times)")

    # canonical row dicts for the real guides
    real_rows = [
        {
            "chromosome": g.on_target_chrom or "",
            "position": g.on_target_pos if g.on_target_pos is not None else "",
            "strand_used": g.on_target_strand or "",
            "guide_rna_sequence": g.guide_seq,
            "base_editor": g.base_editor or "",
        }
        for g in reals
    ]

    need = max(SET_SIZES)
    sampled = []
    if len(reals) < need:
        sampled = sample_guides(index, need - len(reals), rng,
                                {g.guide_seq for g in reals})
        print(f"sampled genomic NGG protospacers: {len(sampled)}")

    for size in SET_SIZES:
        rows = real_rows[:size]
        if len(rows) < size:
            for s in sampled[: size - len(rows)]:
                rows.append({
                    "chromosome": s["chrom"], "position": s["pos"],
                    "strand_used": s["strand"], "guide_rna_sequence": s["guide"],
                    "base_editor": "ABE8e",
                })
        write_tsv(OUT / f"guides_{size}.tsv", rows)

    # 500-set expanded over every profile
    profiles = pd.read_csv(PROFILES, sep="\t")
    names = profiles["name"].astype(str).tolist()
    print(f"profiles: {len(names)} | distinct PAMs: {sorted(set(profiles['pam']))}")
    rows = []
    base = (real_rows + [
        {"chromosome": s["chrom"], "position": s["pos"], "strand_used": s["strand"],
         "guide_rna_sequence": s["guide"], "base_editor": "ABE8e"}
        for s in sampled
    ])[:need]
    for r in base:
        for prof in names:
            rows.append({**r, "base_editor": prof})
    write_tsv(OUT / "guides_500x38.tsv", rows)


if __name__ == "__main__":
    main()
