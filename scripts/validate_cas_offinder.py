"""Validate BEguidedOFF against Cas-OFFinder outputs.

Usage:
    python validate_cas_offinder.py
"""
import gzip
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent.parent / "relevant_files" / "cas-offinder"

def parse_casoffinput(path):
    """Parse Cas-OFFinder input, return list of (guide_seq, pam_len, max_mm)."""
    with open(path) as f:
        lines = f.read().strip().splitlines()
    # line 0 = fasta path, line 1 = pam pattern, rest = guide + mm
    pam_pattern = lines[1].strip()
    pam_len = len(pam_pattern)
    entries = []
    for line in lines[2:]:
        parts = line.strip().split()
        guide_seq = parts[0][:20]  # first 20 nt = guide
        max_mm = int(parts[1])
        entries.append((guide_seq, pam_len, max_mm))
    return entries


def parse_casoffoutput(path):
    """Parse Cas-OFFinder output, return dict guide_seq -> list of (seq,target,pos,strand,mm)."""
    results = defaultdict(list)

    # Cas-OFFinder output format:
    # guide_seq>PAM  chrom_desc  position  target_seq(with mismatches in lower case)  strand  mismatches
    # Example: AACCTCATCGAACACTCGTGTCNNN\t1 dna:chromosome...\t5578703\tAAtCTCATgGAACTCGaGgtTGG\t+\t5
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle the first field carefully because it may contain spaces
            # Split on tab
            parts = line.split('\t')
            if len(parts) != 6:
                continue
            guide_with_pam = parts[0]
            # The guide is the first 20 characters of the first field (before the PAM)
            # Actually the first field is like:  AACCTCATCGAACTCGTGTCNNN
            # For 6 N PAM entries, the length may be 26
            # Deduce guide length from pam length (not directly given in output)
            guide_seq = guide_with_pam[:20]
            target_seq = parts[3].upper()  # uppercase for comparison
            strand = parts[4]
            mm = int(parts[5])
            # Chromosome from description: first token is chromosome name
            chrom_field = parts[1]
            chrom = chrom_field.split()[0]
            pos = int(parts[2])

            results[guide_seq].append({
                'chrom': chrom,
                'pos': pos,
                'strand': strand,
                'mm': mm,
                'target_seq': target_seq,
            })
    return results


def main():
    input_data_path = ""

    ngg_input_e = parse_casoffinput(BASE / "casoffinput_NGG.txt")
    nngrrt_input_e = parse_casoffinput(BASE / "casoffinput_NNGRRT.txt")
    nnnrrt_input_e = parse_casoffinput(BASE / "casoffinput_NNNRRT.txt")

    print(f"NGG guides: {len(ngg_input_e)}")
    print(f"NNGRRT guides: {len(nngrrt_input_e)}")
    print(f"NNNRRT guides: {len(nnnrrt_input_e)}")
    print()

    ngg_out = parse_casoffoutput(BASE / "casoffoutput_NGG.txt")
    nngrrt_out = parse_casoffoutput(BASE / "casoffoutput_NNGRRT.txt")
    nnnrrt_out = parse_casoffoutput(BASE / "casoffoutput_NNNRRT.txt")

    for name, input_e, output in [
        ("NGG", ngg_input_e, ngg_out),
        ("NNGRRT", nngrrt_input_e, nngrrt_out),
        ("NNNRRT", nnnrrt_input_e, nnnrrt_out),
    ]:
        total_hits = sum(len(output[g]) for g in output)
        print(f"{name}: {len(input_e)} guides, {total_hits} total hits")

    # Build Cas-OFFinder output per guide
    all_cas = {}
    for d in [ngg_out, nngrrt_out, nnnrrt_out]:
        for gseq, hits in d.items():
            if gseq not in all_cas:
                all_cas[gseq] = hits
            else:
                all_cas[gseq].extend(hits)

    print(f"\nUnique guides in all Cas-OFFinder outputs: {len(all_cas)}")
    print(f"Total Cas-OFFinder hits across all outputs: {sum(len(v) for v in all_cas.values())}")


if __name__ == "__main__":
    main()
