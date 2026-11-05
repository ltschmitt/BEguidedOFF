"""Numba CUDA off-target search kernel.

Each kernel call scans a chunk of the packed genome on one strand.
One GPU thread = one genome position.  Threads loop over all guides
stored in shared memory and write hits to a global device buffer
with an atomic counter.

Performance notes (2026-10-25 kernel rework; stage profiling had shown
the kernel scan at 78-97% of extensive-search wall time):

* The 20-nt guide window and the PAM bases depend only on the genome
  position, so they are extracted once per position (word-wise, hoisted
  out of the guide loop) instead of once per (position, guide) pair.
* PAM matching is a packed bit-field test: each guide's PAM is
  precompiled on the host into (pattern, compare-mask) fields, so the
  per-pair check is one XOR + AND instead of a per-base IUPAC mask loop.
  PAM positions whose IUPAC allowed-set cannot be expressed as fixed
  bits (S, W, B, D, H, V) fall back to the per-base mask test (rare).
* The genome device upload is cached across calls, so overflow re-scan
  rounds do not re-upload ~775 MB per strand.
* Hit buffers are allocated without the redundant -1 fill (the host
  only reads the first `count` entries per guide).
* An N-position bitmask (stored in the encoded genome by the converter)
  gates every candidate: a site whose 26-bp window overlaps an unknown
  base cannot be a real match, because N is encoded as A in the 2-bit
  genome. Without the gate, A/T-rich guides collect millions of phantom
  hits across the N gaps/centromeres of the assembly.
"""
import math
import logging
import numpy as np
from numba import cuda, uint32, uint64, int32, int8, uint8

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
def _pack_bits(genome_packed, start, nbits):
    """Pack `nbits` genome bits (2 per base) starting at base `start`.

    Word-wise extraction: reads 2-3 consecutive uint32 words and assembles
    them with shifts (the naive per-base loop costs ~6 instructions per
    base).  Bit `i` of the result corresponds to genome bit `2*start + i`,
    i.e. base `start + i//2` -- bases above `nbits//2` are neighbouring
    genome bases, not zeros; callers must mask them off (the guide compare
    masks to 40 bits, the PAM test to the compared 12 bits).
    """
    n_words = genome_packed.shape[0]
    b0 = start // 16
    off = (start % 16) * 2          # 0..30, even
    w0 = genome_packed[b0]
    if off == 0:
        v = uint64(w0)
        if nbits > 32 and b0 + 1 < n_words:
            v |= uint64(genome_packed[b0 + 1]) << uint64(32)
    else:
        w1 = genome_packed[b0 + 1] if b0 + 1 < n_words else uint32(0)
        pair = uint64(w0) | (uint64(w1) << uint64(32))
        v = pair >> uint64(off)
        if nbits + off > 64 and b0 + 2 < n_words:
            # Window spans a third word: its low bits continue at bit 64-off.
            v |= uint64(genome_packed[b0 + 2]) << uint64(64 - off)
    return v


@cuda.jit(device=True)
def _pam_matches(genome_packed, pam_start, pam_len, pam_masks, g):
    """Check if genome matches IUPAC PAM mask (per-base fallback path)."""
    for i in range(pam_len):
        nuc = _get_2bit(genome_packed, pam_start + i)
        mask = pam_masks[g, i]
        if (uint8(1) << nuc) & mask == uint8(0):
            return False
    return True


def _pam_nibble_fields(nib):
    """Map a 4-bit IUPAC allowed-nibble to (pattern2, compare2) bit fields.

    The accepted base set must be expressible as "fixed bits" of the 2-bit
    encoding: single bases (cmp=11), pairs sharing one bit (R, Y, M, K),
    and N (cmp=00).  S, W and the 3-way codes (B, D, H, V) return None and
    are handled by the per-base fallback in the kernel.
    """
    bases = [b for b in range(4) if nib & (1 << b)]
    k = len(bases)
    if k == 1:
        return bases[0], 0b11
    if k == 4:
        return 0, 0b00
    if k == 2:
        b_lo, b_hi = bases
        if (b_lo & 1) == (b_hi & 1) and (b_lo >> 1) != (b_hi >> 1):
            return b_lo & 1, 0b01        # bit 0 fixed, bit 1 free (R, Y)
        if (b_lo >> 1) == (b_hi >> 1) and (b_lo & 1) != (b_hi & 1):
            return b_lo & 2, 0b10        # bit 1 fixed, bit 0 free (M, K)
    return None


def encode_pam_packed(pam_masks, pam_lens, pam_3prime):
    """Precompile per-guide PAM tests into packed bit fields.

    Returns (pats, cmps) uint32 arrays:
      pats[g] = pattern (bits 0-11, 2 bits per PAM position)
                | pam_len << 12        (0 = guide skipped)
                | exotic << 15         (per-base fallback required)
                | 3prime << 16
      cmps[g] = compared bits (subset of bits 0-11; 0 for exotic guides)
    """
    n = int(pam_masks.shape[0])
    pats = np.zeros(n, dtype=np.uint32)
    cmps = np.zeros(n, dtype=np.uint32)
    for g in range(n):
        plen = int(pam_lens[g])
        if plen == 0 or plen > MAX_PAM_LEN:
            continue                 # len field stays 0 -> kernel skips
        pat = 0
        cmp_mask = 0
        exotic = False
        for i in range(plen):
            res = _pam_nibble_fields(int(pam_masks[g, i]) & 0xF)
            if res is None:
                exotic = True
                break
            p2, c2 = res
            pat |= p2 << (2 * i)
            cmp_mask |= c2 << (2 * i)
        info = (pat & 0xFFF) | (plen << 12)
        if exotic:
            info |= 1 << 15
            cmp_mask = 0
        if pam_3prime[g]:
            info |= 1 << 16
        pats[g] = info
        cmps[g] = cmp_mask
    return pats, cmps


@cuda.jit
def search_chunk_kernel(
    genome_packed,       # uint32[:]
    n_mask_packed,       # uint8[:]  packed N bitmask (bit i = genome position i,
                         #           little bit order); empty -> no N gating
    total_bp,            # int
    guide_encoded,       # uint64[:]
    n_guides,            # int
    pam_masks,           # uint8[:, :]  (n_guides, MAX_PAM_LEN) -- global, fallback only
    pam_pats,            # uint32[:]  packed pattern/len/exotic/3prime fields
    pam_cmps,            # uint32[:]  compared bits (<= 12 bits)
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
    # Shared memory for guide batch: 16 + 8 + 8 = 32 KB at MAX_GUIDES=2048
    s_guides = cuda.shared.array(MAX_GUIDES, uint64)
    s_pats = cuda.shared.array(MAX_GUIDES, uint32)
    s_cmps = cuda.shared.array(MAX_GUIDES, uint32)

    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x

    # Cooperative load into shared memory
    ng = int(n_guides)
    for i in range(tid, ng, bdim):
        s_guides[i] = guide_encoded[i]
        s_pats[i] = pam_pats[i]
        s_cmps[i] = pam_cmps[i]
    cuda.syncthreads()

    # Thread's genomic position
    pos = chunk_start + cuda.blockIdx.x * bdim + tid

    # Do not overshoot the genome or the current chunk
    valid_end = total_bp - 26   # safe for any PAM up to 6
    if pos >= chunk_end or pos >= valid_end:
        return

    # N gate: skip positions whose 26-bp site window (20-nt guide + up to
    # 6-nt PAM; the union is [pos, pos+26) in both PAM layouts) overlaps an
    # unknown (N) reference base. N bases are encoded as A in the 2-bit
    # genome, so without this gate every N position masquerades as an A and
    # A/T-rich guides accumulate millions of phantom "matches" across the
    # ~160 Mbp of N in a typical mammalian genome assembly.
    if n_mask_packed.shape[0] > 0:
        b = pos >> 3
        off = uint64(pos & 7)
        w = (uint64(n_mask_packed[b])
             | (uint64(n_mask_packed[b + 1]) << uint64(8))
             | (uint64(n_mask_packed[b + 2]) << uint64(16))
             | (uint64(n_mask_packed[b + 3]) << uint64(24))
             | (uint64(n_mask_packed[b + 4]) << uint64(32)))
        if (w >> off) & uint64(0x3FFFFFF):
            return

    # Per-position extraction, hoisted out of the guide loop (3' layout)
    window3 = _pack_bits(genome_packed, pos, 40)
    pam6 = _pack_bits(genome_packed, pos + 20, 12)

    for g in range(ng):
        info = s_pats[g]
        plen = int((info >> uint32(12)) & uint32(7))
        if plen == 0:
            continue

        if (info >> uint32(16)) & uint32(1):
            # 3' PAM: window and PAM bases come from the hoisted extraction
            guide_start = pos
            pam_start = pos + 20
            window = window3
            pam_packed = pam6
        else:
            guide_start = pos + plen
            pam_start = pos
            if guide_start + 20 > total_bp or pam_start + plen > total_bp:
                continue
            window = _pack_bits(genome_packed, guide_start, 40)
            pam_packed = _pack_bits(genome_packed, pam_start, 12)

        # PAM match
        if (info >> uint32(15)) & uint32(1):
            # Exotic PAM (S/W/B/D/H/V position): per-base fallback
            if not _pam_matches(genome_packed, pam_start, plen, pam_masks, g):
                continue
        elif ((pam_packed ^ info) & s_cmps[g]) != uint32(0):
            # Bits >= 12 of the XOR are masked off by s_cmps (<= 12 bits)
            continue

        # Guide compare: per-base mismatch bits from the 2-bit encoding
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


# ---------------------------------------------------------------------------
# Host-side driver
# ---------------------------------------------------------------------------

_GENOME_DEVICE_CACHE = {}   # id(host_array) -> (host_array, device_array)
_GENOME_CACHE_LIMIT = 4


def _genome_to_device(genome_packed):
    """Upload the packed genome once and reuse the device copy across calls.

    Re-scan rounds re-enter search_genome_cuda with the same host arrays;
    without the cache each round re-uploads ~775 MB per strand.  The cache
    is keyed by id() and validated by identity, so a recycled id can never
    return a stale device copy.
    """
    host = np.ascontiguousarray(genome_packed, dtype=np.uint32)
    entry = _GENOME_DEVICE_CACHE.get(id(host))
    if entry is not None and entry[0] is host:
        return entry[1]
    d_genome = cuda.to_device(host)
    if len(_GENOME_DEVICE_CACHE) >= _GENOME_CACHE_LIMIT:
        _GENOME_DEVICE_CACHE.clear()
    _GENOME_DEVICE_CACHE[id(host)] = (host, d_genome)
    return d_genome


_N_MASK_DEVICE_CACHE = {}   # id(host_array) -> (host_array, device_array)


def _n_mask_to_device(n_mask_packed):
    """Upload the packed N bitmask once and reuse the device copy.

    Same id()-keyed identity cache as ``_genome_to_device``: the mask is
    ~390 MB for a mammalian genome and overflow re-scan rounds must not
    re-upload it per call.
    """
    host = np.ascontiguousarray(n_mask_packed, dtype=np.uint8)
    entry = _N_MASK_DEVICE_CACHE.get(id(host))
    if entry is not None and entry[0] is host:
        return entry[1]
    d_mask = cuda.to_device(host)
    if len(_N_MASK_DEVICE_CACHE) >= _GENOME_CACHE_LIMIT:
        _N_MASK_DEVICE_CACHE.clear()
    _N_MASK_DEVICE_CACHE[id(host)] = (host, d_mask)
    return d_mask


def _prepare_n_mask_device(n_mask_packed, total_bp):
    """Validate/pad the N bitmask and upload it (empty array -> no gating).

    The kernel reads bytes ``[pos >> 3, pos >> 3 + 4]`` for every position
    below ``total_bp - 26``, so at least ``((total_bp - 27) >> 3) + 5``
    bytes must exist; the host pads the packed mask accordingly.
    """
    if n_mask_packed is None:
        return cuda.to_device(np.zeros(0, dtype=np.uint8))
    host = np.ascontiguousarray(n_mask_packed, dtype=np.uint8)
    if host.size == 0:
        return cuda.to_device(host)
    need = ((total_bp - 27) >> 3) + 5
    if host.size < need:
        host = np.pad(host, (0, need - host.size), constant_values=0)
    return _n_mask_to_device(host)


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
    n_mask_packed=None,
):
    """Host-side driver for the Numba CUDA search kernel.

    Returns:
        all_results: dict guide_idx -> list of (position, n_mismatches)
        (if return_overflow=True): tuple of
            dict guide_idx -> bool (True if buffer was exceeded),
            dict guide_idx -> int (exact number of raw matches; the kernel
            counts every match even beyond the recorded buffer, so this is
            exact also for overflowing guides)
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
    d_genome = _genome_to_device(genome_packed)
    d_n_mask = _prepare_n_mask_device(n_mask_packed, total_bp)
    d_guides = cuda.to_device(np.ascontiguousarray(guide_encoded, dtype=np.uint64))

    pam_masks_c = np.ascontiguousarray(pam_masks, dtype=np.uint8)
    pats, cmps = encode_pam_packed(pam_masks_c, pam_lens, pam_3prime)
    d_pam_masks = cuda.to_device(pam_masks_c)
    d_pats = cuda.to_device(pats)
    d_cmps = cuda.to_device(cmps)

    d_hit_counts = cuda.to_device(np.zeros(ng, dtype=np.int32))
    # Entries [0, count) are always written by the kernel before being read,
    # so the buffers need no -1 fill.
    d_hit_positions = cuda.to_device(np.empty((ng, results_per_guide), dtype=np.int64))
    d_hit_mismatches = cuda.to_device(np.empty((ng, results_per_guide), dtype=np.int8))

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
            d_n_mask,
            total_bp,
            d_guides,
            ng,
            d_pam_masks,
            d_pats,
            d_cmps,
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
    hit_counts = {}
    for g in range(ng):
        cnt = int(h_counts[g])
        overflow[g] = cnt > results_per_guide
        hit_counts[g] = cnt
        recorded = min(cnt, results_per_guide)
        for i in range(recorded):
            all_results[g].append((int(h_positions[g, i]), int(h_mismatches[g, i])))

    if return_overflow:
        return all_results, overflow, hit_counts
    return all_results
