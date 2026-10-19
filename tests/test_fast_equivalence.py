"""Fast FM-index search must match an independent brute-force reference."""
import random
import tempfile
from pathlib import Path

from _fastutil import bruteforce_hits, fast_hits, make_fast_finder, write_fasta
from beguidedoff.types import GuideInput

_PAMS = ["NGG", "NRG", "NNGRRT", "NGA"]


def _rand(n):
    return "".join(random.choice("ACGT") for _ in range(n))


def _mutate(seq, positions):
    out = list(seq)
    for p in positions:
        out[p] = random.choice([b for b in "ACGT" if b != out[p]])
    return "".join(out)


def _plant(genome, guide, pam, positions):
    site = _mutate(guide, positions) + pam
    pos = random.randint(10, len(genome) - len(site) - 10)
    return genome[:pos] + site + genome[pos + len(site):]


def _build_case(seed):
    random.seed(seed)
    specs = [(_rand(20), random.choice(_PAMS)) for _ in range(20)]
    chroms = {}
    for chrom in ("chr1", "chr2"):
        seq = _rand(12000)
        for g, pam in specs:
            for _ in range(random.randint(0, 2)):
                seq = _plant(seq, g, pam, random.sample(range(20), random.randint(0, 2)))
            for _ in range(random.randint(0, 3)):
                seq = _plant(seq, g, pam, random.sample(range(20), random.randint(3, 5)))
        chroms[chrom] = seq
    return specs, chroms


def _run(cutoff):
    for seed in (1, 2, 3):
        specs, chroms = _build_case(seed)
        guides = [
            GuideInput(f"g{i}", g, pam, "3prime", 4, 8, random.choice(["CBE", "ABE"]))
            for i, (g, pam) in enumerate(specs)
        ]
        with tempfile.TemporaryDirectory() as td:
            fa = Path(td) / "genome.fa"
            write_fasta(str(fa), chroms)
            finder = make_fast_finder(str(fa), td, search_cutoff=cutoff, n_top_offtargets=100000)
            df = finder.search(guides=guides, return_all=False,
                               count_score_threshold=cutoff, verbose=False)

        fast = set(fast_hits(df))
        ref = set()
        for g in guides:
            ref.update(bruteforce_hits(
                chroms, g.guide_seq, g.pam, cutoff,
                editor_type=g.editor_type, edit_window_start=4, edit_window_end=8,
            ))

        # On-target coordinates were not supplied, so no site is excluded.
        assert fast == ref, (
            f"seed={seed} cutoff={cutoff}\n"
            f"  only in fast: {sorted(fast - ref)[:5]}\n"
            f"  only in ref:  {sorted(ref - fast)[:5]}"
        )


def test_equivalence_cutoff_5():
    _run(5.0)


def test_equivalence_cutoff_3():
    _run(3.0)
