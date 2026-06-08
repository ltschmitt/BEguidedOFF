"""Numba CUDA off-target search kernel.

Each kernel call scans a chunk of the packed genome on one strand.
One GPU thread = one genome position.  Threads loop over all guides
stored in shared memory and write hits to a global device buffer
with an atomic counter.
"""
import math
import logging
import numpy as np
from numba import cuda, uint32, uint64, int32, int64, int8, uint8

logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)

MAX_GUIDES = 2048         # shared-memory budget allows ~3 k on 48 KB GPUs
MAX_PAM_LEN = 6
_DEFAULT_RESULTS_PER_GUIDE = 10240  # max off-targets recorded per guide per chunk


@cuda.jit(device=True)
def _get_2bit(genome_packed, gpos):
    """Extract 2-bit nucleotide from packed uint32 genome."""
    word_idx = gpos // 16
    bit_offset = (gpos % 16) * 2
    return uint8((genome_packed[word_idx] >> bit_offset) & uint32(3))


@cuda.jit(device=True)
def _popcount_40(val):
    """Count set bits in the 0x55... masked 40-bit value."""
    lo = uint32(val & uint64(0xFFFFFFFF))
    hi = uint32((val >> uint64(32)) & uint64(0xFFFFFFFF))
    return cuda.popc(lo) + cuda.popc(hi)


@cuda.jit(device=True)
def _pack_window_20(genome_packed, start):
    """Extract 20 bases starting at `start` and pack into 40 bits."""
    w = uint64(0)
    for i in range(20):
        nuc = _get_2bit(genome_packed, start + i)
        w |= uint64(nuc) << (i * 2)
    return w


@cuda.jit(device=True)
def _pam_matches(genome_packed, pam_start, pam_len, pam_mask):
    """Check if genome matches IUPAC PAM mask."""
    for i in range(pam_len):
        nuc = _get_2bit(genome_packed, pam_start + i)
        mask = pam_mask[i]
        if (uint8(1) << nuc) & mask == uint8(0):
            return False
    return True


@cuda.jit
def search_chunk_kernel(
    genome_packed,       # uint32[:]
    total_bp,            # int
    guide_encoded,       # uint64[:]
    n_guides,            # int
    pam_masks,           # uint8[:, :]  (n_guides, MAX_PAM_LEN)
    pam_lens,            # uint8[:]
    pam_3prime,          # uint8[:]  (1 = 3prime, 0 = 5prime)
    max_mismatches,      # int
    chunk_start,         # int  – offset into genome for this chunk
    chunk_end,           # int  – exclusive bound for this chunk
    results_per_guide,   # int

    # Output
    d_hit_counts,        # int32[:]  per-guide hit counter
    d_hit_positions,     # int64[:, :] (n_guides, results_per_guide)
    d_hit_mismatches,    # int8[:, :]  (n_guides, results_per_guide)
):
    """Scan one chunk of the genome.

    Thread layout: 1-D grid. Each thread processes exactly one genome position
    in the range [chunk_start, chunk_end).
    """
    # Shared memory for guide batch
    s_guides = cuda.shared.array(MAX_GUIDES, uint64)
    s_pam_masks = cuda.shared.array((MAX_GUIDES, MAX_PAM_LEN), uint8)
    s_pam_lens = cuda.shared.array(MAX_GUIDES, uint8)
    s_pam_3prime = cuda.shared.array(MAX_GUIDES, uint8)

    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x

    # Cooperative load into shared memory
    ng = int(n_guides)
    for i in range(tid, ng, bdim):
        s_guides[i] = guide_encoded[i]
        s_pam_lens[i] = pam_lens[i]
        s_pam_3prime[i] = pam_3prime[i]
        for p in range(MAX_PAM_LEN):
            s_pam_masks[i, p] = pam_masks[i, p]
    cuda.syncthreads()

    # Thread’s genomic position
    pos = chunk_start + cuda.blockIdx.x * bdim + tid

    # Do not overshoot the genome or the current chunk
    valid_end = total_bp - 26   # safe for any PAM up to 6
    if pos >= chunk_end or pos >= valid_end:
        return

    for g in range(ng):
        pam_len = int(s_pam_lens[g])
        if pam_len == 0:
            continue

        if s_pam_3prime[g]:
            pam_start = pos + 20
            guide_start = pos
        else:
            pam_start = pos
            guide_start = pos + pam_len

        # Boundary check for this specific guide
        if guide_start + 20 > total_bp:
            continue
        if pam_start + pam_len > total_bp:
            continue

        # PAM match
        if not _pam_matches(genome_packed, pam_start, pam_len, s_pam_masks[g]):
            continue

        # Extract guide window and compare
        window = _pack_window_20(genome_packed, guide_start)
        gval = s_guides[g]
        xor_val = window ^ gval
        mm = uint64((xor_val | (xor_val >> uint64(1))) & uint64(0x5555555555))
        n_mm = _popcount_40(mm)

        if n_mm > max_mismatches:
            continue

        # Store hit atomically (host code filters on-target at provided coordinates)
        idx = cuda.atomic.add(d_hit_counts, g, int32(1))
        if idx < results_per_guide:
            d_hit_positions[g, idx] = pos
            d_hit_mismatches[g, idx] = int8(n_mm)


def search_genome_cuda(
    genome_packed,
    total_bp,
    guide_encoded,
    pam_masks,
    pam_lens,
    pam_3prime,
    max_mismatches,
    chunk_size=200_000_000,
    return_overflow=False,
    results_per_guide=_DEFAULT_RESULTS_PER_GUIDE,
):
    """Host-side driver for the Numba CUDA search kernel.

    Returns:
        all_results: dict guide_idx -> list of (position, n_mismatches)
        (if return_overflow=True): dict guide_idx -> bool (True if buffer was exceeded)
    """
    n_guides = len(guide_encoded)
    ng = int(n_guides)

    if ng > MAX_GUIDES:
        raise RuntimeError(
            f"Too many guides ({ng}) for GPU search. "
            f"Maximum supported is {MAX_GUIDES}. "
            f"Consider splitting the input into smaller batches, "
            f"or increase MAX_GUIDES in cuda_search.py."
        )

    # Device arrays
    d_genome = cuda.to_device(np.ascontiguousarray(genome_packed, dtype=np.uint32))
    d_guides = cuda.to_device(np.ascontiguousarray(guide_encoded, dtype=np.uint64))
    d_pam_masks = cuda.to_device(np.ascontiguousarray(pam_masks, dtype=np.uint8))
    d_pam_lens = cuda.to_device(np.ascontiguousarray(pam_lens, dtype=np.uint8))
    d_pam_3prime = cuda.to_device(np.ascontiguousarray(pam_3prime, dtype=np.uint8))

    d_hit_counts = cuda.to_device(np.zeros(ng, dtype=np.int32))
    d_hit_positions = cuda.to_device(np.full((ng, results_per_guide), -1, dtype=np.int64))
    d_hit_mismatches = cuda.to_device(np.full((ng, results_per_guide), -1, dtype=np.int8))

    block_size = 256
    valid_end = total_bp - 26
    n_chunks = math.ceil(valid_end / chunk_size)

    all_results = {i: [] for i in range(ng)}

    for c in range(n_chunks):
        chunk_start = c * chunk_size
        chunk_end = min((c + 1) * chunk_size, valid_end)
        chunk_len = chunk_end - chunk_start
        if chunk_len <= 0:
            continue

        grid_size = (chunk_len + block_size - 1) // block_size

        search_chunk_kernel[grid_size, block_size](
            d_genome,
            total_bp,
            d_guides,
            ng,
            d_pam_masks,
            d_pam_lens,
            d_pam_3prime,
            max_mismatches,
            chunk_start,
            chunk_end,
            results_per_guide,
            d_hit_counts,
            d_hit_positions,
            d_hit_mismatches,
        )
        cuda.synchronize()

    # Copy back
    h_counts = d_hit_counts.copy_to_host()
    h_positions = d_hit_positions.copy_to_host()
    h_mismatches = d_hit_mismatches.copy_to_host()

    overflow = {}
    for g in range(ng):
        cnt = int(h_counts[g])
        overflow[g] = cnt > results_per_guide
        recorded = min(cnt, results_per_guide)
        for i in range(recorded):
            all_results[g].append((int(h_positions[g, i]), int(h_mismatches[g, i])))

    if return_overflow:
        return all_results, overflow
    return all_results
