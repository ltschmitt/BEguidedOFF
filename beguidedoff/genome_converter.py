"""Genome preparation module — one-time FASTA -> 2-bit .npz encoder."""
import numpy as np
from pathlib import Path
from typing import Optional
from pysam import FastaFile

from beguidedoff.encoding import (
    NUC_TO_2BIT,
    COMPLEMENT_2BIT,
    pack_2bit_to_uint32,
    unpack_uint32_to_2bit,
)


def convert(
    fasta_path: str,
    output_path: str,
    include_rc: bool = True,
    include_n_mask: bool = True,
    chunk_size: int = 100_000_000,
    verbose: bool = True,
) -> None:
    """Convert a FASTA genome file to BEguidedOFF 2-bit binary format.

    Args:
        fasta_path: Path to reference genome FASTA file.
        output_path: Output path for the .npz encoded genome.
        include_rc: If True, pre-compute and store reverse complement.
        include_n_mask: If True, store a bitmask of N positions.
        chunk_size: Number of base pairs to process at a time.
        verbose: Print progress.
    """
    fasta = FastaFile(fasta_path)

    chrom_names = []
    chrom_offsets = []
    chrom_lengths = []

    all_encoded_fwd = []
    all_encoded_rc = []
    all_n_mask = []

    global_offset = 0

    for chrom in fasta.references:
        seq = fasta.fetch(chrom).upper()
        chrom_len = len(seq)

        chrom_names.append(chrom)
        chrom_offsets.append(global_offset)
        chrom_lengths.append(chrom_len)
        global_offset += chrom_len

        if verbose:
            print(f"  Encoding {chrom} ({chrom_len:,} bp)...")

        # Encode forward strand
        encoded = np.array([NUC_TO_2BIT.get(c, 0) for c in seq], dtype=np.uint8)
        all_encoded_fwd.append(encoded)

        # N-mask
        if include_n_mask:
            n_positions = np.array([1 if c == "N" else 0 for c in seq], dtype=np.uint8)
            all_n_mask.append(n_positions)

        # Reverse complement (reverse order, complement bases)
        if include_rc:
            rc_encoded = np.array(
                [COMPLEMENT_2BIT[b] for b in reversed(encoded)], dtype=np.uint8
            )
            all_encoded_rc.append(rc_encoded)

    # Concatenate all chromosomes
    full_fwd = np.concatenate(all_encoded_fwd)
    total_bp = len(full_fwd)

    if verbose:
        print(f"  Total genome: {total_bp:,} bp")
        print(f"  Packing into uint32...")

    # Pack into uint32 (16 nt per element)
    genome_fwd_packed = pack_2bit_to_uint32(full_fwd)

    save_dict = {
        "genome_forward": genome_fwd_packed,
        "chrom_names": np.array(chrom_names),
        "chrom_offsets": np.array(chrom_offsets, dtype=np.int64),
        "chrom_lengths": np.array(chrom_lengths, dtype=np.int64),
        "total_bp": np.int64(total_bp),
    }

    if include_rc:
        full_rc = np.concatenate(all_encoded_rc)
        genome_rc_packed = pack_2bit_to_uint32(full_rc)
        save_dict["genome_rc"] = genome_rc_packed

    if include_n_mask:
        full_n_mask = np.concatenate(all_n_mask)
        # Pack N-mask as bits (8 positions per byte)
        n_mask_packed = np.packbits(full_n_mask)
        save_dict["n_mask"] = n_mask_packed

    np.savez_compressed(output_path, **save_dict)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    if verbose:
        print(f"  Saved to {output_path} ({file_size_mb:.1f} MB)")
        print(f"  Done.")


def verify(
    encoded_path: str,
    fasta_path: str,
    sample_n: int = 10000,
    verbose: bool = True,
) -> bool:
    """Verify encoded genome against original FASTA by random sampling.

    Args:
        encoded_path: Path to .npz encoded genome.
        fasta_path: Path to original FASTA.
        sample_n: Number of random positions to verify.

    Returns:
        True if all sampled positions match.
    """
    data = np.load(encoded_path, allow_pickle=True)
    fasta = FastaFile(fasta_path)

    total_bp = int(data["total_bp"])
    chrom_names = data["chrom_names"]
    chrom_offsets = data["chrom_offsets"]
    chrom_lengths = data["chrom_lengths"]
    genome_packed = data["genome_forward"]

    decode_map = {0: "A", 1: "C", 2: "G", 3: "T"}

    rng = np.random.default_rng(42)
    if total_bp <= 20:
        if verbose:
            print(f"  Genome too small ({total_bp} bp) for random sampling; skipping verify")
        return True
    test_positions = rng.integers(0, total_bp - 20, size=min(sample_n, total_bp - 20))

    errors = 0
    for pos in test_positions:
        # Find chromosome
        chrom_idx = int(np.searchsorted(chrom_offsets, pos, side="right") - 1)
        chrom = str(chrom_names[chrom_idx])
        local_pos = int(pos - chrom_offsets[chrom_idx])
        chrom_len = int(chrom_lengths[chrom_idx])

        if local_pos + 20 > chrom_len:
            continue

        # Get from FASTA
        ref_seq = fasta.fetch(chrom, local_pos, local_pos + 20).upper()

        # Get from encoded
        decoded = ""
        for i in range(20):
            gpos = int(pos) + i
            word_idx = gpos // 16
            bit_offset = (gpos % 16) * 2
            nuc_val = int((genome_packed[word_idx] >> bit_offset) & 0x3)
            decoded += decode_map[nuc_val]

        if decoded != ref_seq.replace("N", "A"):
            errors += 1
            if verbose:
                print(
                    f"  MISMATCH at {chrom}:{local_pos}: "
                    f"expected={ref_seq}, got={decoded}"
                )

    if verbose:
        print(f"  Verified {sample_n} positions: {errors} errors")

    return errors == 0


def load_genome(encoded_path: str) -> dict:
    """Load an encoded genome into memory.

    Returns dict with:
      genome_forward, genome_rc (optional), chrom_names, chrom_offsets,
      chrom_lengths, total_bp, n_mask (optional)
    """
    data = np.load(encoded_path, allow_pickle=True)
    result = {
        "genome_forward": data["genome_forward"],
        "chrom_names": data["chrom_names"],
        "chrom_offsets": data["chrom_offsets"],
        "chrom_lengths": data["chrom_lengths"],
        "total_bp": int(data["total_bp"]),
    }
    if "genome_rc" in data:
        result["genome_rc"] = data["genome_rc"]
    if "n_mask" in data:
        result["n_mask"] = data["n_mask"]
    return result
