/**
 * RingKernel Migration Kernels (v1.1)
 *
 * GPU-side state capture / restore / drain helpers for the 3-phase actor
 * migration protocol. These kernels are portable across `sm_75`+ (Turing
 * and newer). Hopper-specific optimizations (distributed shared memory,
 * thread block clusters) are feature-gated with `__CUDA_ARCH__ >= 900`.
 *
 * Design notes
 * ============
 *
 *   - All output buffers are **CUDA mapped memory** (host-readable). We use
 *     `__threadfence_system()` before publishing "done" markers so the host
 *     sees a consistent snapshot.
 *
 *   - CRC32 reduction is a three-stage pattern:
 *         per-thread partial  ->  warp shuffle reduction  ->  block atomic
 *     This is cheap, shared-memory-light, and scales linearly with size.
 *
 *   - Checksum uses the CRC32/ISO-HDLC polynomial 0xEDB88320. The per-thread
 *     partials are reduced with XOR (warp shuffle → block atomic → grid
 *     atomic). This is NOT a pure CRC32 of the full buffer — it is a
 *     per-stride CRC composed under XOR. Capture and restore on different
 *     GPUs use the *same* kernel launch config, so the reductions match
 *     byte-for-byte. Hosts should treat the result as a migration-internal
 *     integrity checksum rather than a portable CRC32.
 *
 *   - The drain kernel expects an SPSC queue with a 16-byte header
 *     (`{head, tail}`) followed by fixed-size message slots. It writes the
 *     messages linearly into `drain_buffer` and advances the queue's tail
 *     to mark them consumed.
 */

#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// Helpers — portable (sm_75+) CRC32
// ============================================================================

/**
 * Precomputed CRC32 of a single byte using the reversed CRC32/ISO-HDLC
 * polynomial 0xEDB88320. Used to keep per-thread work register-resident.
 */
__device__ __forceinline__ uint32_t crc32_byte(uint8_t b, uint32_t crc) {
    crc ^= b;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        crc = (crc >> 1) ^ (0xEDB88320u * (crc & 1u));
    }
    return crc;
}

/**
 * CRC32 of a single 32-bit word, little-endian byte order.
 */
__device__ __forceinline__ uint32_t crc32_word(uint32_t word, uint32_t crc) {
    crc = crc32_byte(static_cast<uint8_t>(word & 0xFFu), crc);
    crc = crc32_byte(static_cast<uint8_t>((word >> 8) & 0xFFu), crc);
    crc = crc32_byte(static_cast<uint8_t>((word >> 16) & 0xFFu), crc);
    crc = crc32_byte(static_cast<uint8_t>((word >> 24) & 0xFFu), crc);
    return crc;
}

/**
 * Combine two per-thread CRC32 accumulators by XOR. This is only correct
 * because both source and destination kernels use identical launch config
 * (same strides, same warp count) — the XOR is associative and the shapes
 * match, so the same bytes land in the same thread on both sides. Do NOT
 * use this for out-of-order composition against a reference CRC32 of the
 * whole buffer.
 */
__device__ __forceinline__ uint32_t combine_crc_xor(uint32_t a, uint32_t b) {
    return a ^ b;
}

/**
 * Warp-level reduction of a per-thread value using shuffle.
 */
__device__ __forceinline__ uint32_t warp_reduce_xor(uint32_t val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val ^= __shfl_xor_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

// ============================================================================
// capture_actor_state
// ============================================================================

/**
 * Copy the actor's state into a mapped buffer and compute its CRC32.
 *
 *   actor_state_ptr   device pointer to the live actor state
 *   state_size_bytes  length in bytes of the state region
 *   output_buffer     mapped memory buffer (host-readable)
 *   bytes_written     out: bytes actually copied (mapped, one u64)
 *   checksum          out: CRC32 of the copied bytes (mapped, one u32)
 *
 * Layout contract
 * ---------------
 * The capture kernel writes `state_size_bytes` bytes starting at
 * `output_buffer[0]`. The Rust wrapper allocates `output_buffer` via
 * `CudaMappedBuffer` with `CU_MEMHOSTALLOC_DEVICEMAP` so the host side can
 * read it directly after a stream synchronize.
 */
extern "C" __global__ void capture_actor_state(
    const void* __restrict__ actor_state_ptr,
    size_t state_size_bytes,
    void* __restrict__ output_buffer,
    uint64_t* __restrict__ bytes_written,
    uint32_t* __restrict__ checksum
) {
    // The host launches with `gridDim.x == 1` so the checksum reduction is
    // deterministic (both source and target GPUs use the same thread shape
    // and therefore hash the same bytes into the same accumulator slot).
    __shared__ uint32_t s_crc;
    if (threadIdx.x == 0) {
        s_crc = 0u;
    }
    __syncthreads();

    const uint32_t tid = threadIdx.x;
    const uint32_t stride = blockDim.x;

    const uint8_t*  src8  = static_cast<const uint8_t*>(actor_state_ptr);
    uint8_t*        dst8  = static_cast<uint8_t*>(output_buffer);

    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(actor_state_ptr);
    uint32_t*       dst32 = reinterpret_cast<uint32_t*>(output_buffer);

    const size_t   word_count = state_size_bytes / 4;
    const size_t   tail_start = word_count * 4;

    uint32_t thread_crc = 0u;

    // 1) Word-wide copy + CRC.
    for (size_t i = tid; i < word_count; i += stride) {
        uint32_t w = src32[i];
        dst32[i] = w;
        thread_crc = crc32_word(w, thread_crc);
    }

    // 2) Byte-wise tail.
    for (size_t i = tail_start + tid; i < state_size_bytes; i += stride) {
        uint8_t b = src8[i];
        dst8[i] = b;
        thread_crc = crc32_byte(b, thread_crc);
    }

    // 3) Warp reduction → block-wide XOR accumulator.
    thread_crc = warp_reduce_xor(thread_crc);
    const uint32_t lane = threadIdx.x & (warpSize - 1);
    if (lane == 0) {
        atomicXor(&s_crc, thread_crc);
    }
    __syncthreads();

    // 4) Thread 0 publishes the block-wide checksum and the byte count.
    //    __threadfence_system() ensures the host sees the *data* before the
    //    "done" markers.
    if (threadIdx.x == 0) {
        __threadfence_system();
        *checksum = s_crc;
        *bytes_written = static_cast<uint64_t>(state_size_bytes);
    }
}

// ============================================================================
// restore_actor_state
// ============================================================================

/**
 * Copy state bytes from a mapped input buffer back into the actor's live
 * state region, recomputing the migration integrity checksum to verify
 * that the source/target data matches.
 *
 *   actor_state_ptr      device pointer to restore into
 *   input_buffer         mapped memory containing source state (host-written)
 *   state_size_bytes     length in bytes
 *   expected_checksum    checksum the source GPU produced
 *   verification_result  out: 0 if match, -1 if mismatch
 *
 * Grid semantics: the caller MUST launch this kernel with exactly 1 block.
 * We then rely on a `__syncthreads()` in-block to compose the reduction
 * deterministically, avoiding the grid-sync dance. 1 block × 128 threads
 * is more than enough for the message sizes v1.1 targets (≤ 6.4 MiB per
 * actor per the spec) — the host side already enforces that grid shape
 * in `MigrationKernels::restore_state`.
 *
 * If the checksum mismatches, the destination buffer is left in an
 * undefined state and the orchestrator must refuse to activate the
 * target actor.
 */
extern "C" __global__ void restore_actor_state(
    void* __restrict__ actor_state_ptr,
    const void* __restrict__ input_buffer,
    size_t state_size_bytes,
    uint32_t expected_checksum,
    int* __restrict__ verification_result
) {
    __shared__ uint32_t s_crc;
    if (threadIdx.x == 0) {
        s_crc = 0u;
    }
    __syncthreads();

    // We launch with gridDim.x == 1 — tid is just threadIdx.x in a flat block.
    const uint32_t tid = threadIdx.x;
    const uint32_t stride = blockDim.x;

    const uint8_t*  src8  = static_cast<const uint8_t*>(input_buffer);
    uint8_t*        dst8  = static_cast<uint8_t*>(actor_state_ptr);

    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(input_buffer);
    uint32_t*       dst32 = reinterpret_cast<uint32_t*>(actor_state_ptr);

    const size_t   word_count = state_size_bytes / 4;
    const size_t   tail_start = word_count * 4;

    uint32_t thread_crc = 0u;

    for (size_t i = tid; i < word_count; i += stride) {
        uint32_t w = src32[i];
        dst32[i] = w;
        thread_crc = crc32_word(w, thread_crc);
    }
    for (size_t i = tail_start + tid; i < state_size_bytes; i += stride) {
        uint8_t b = src8[i];
        dst8[i] = b;
        thread_crc = crc32_byte(b, thread_crc);
    }

    thread_crc = warp_reduce_xor(thread_crc);
    const uint32_t lane = threadIdx.x & (warpSize - 1);
    if (lane == 0) {
        atomicXor(&s_crc, thread_crc);
    }
    __syncthreads();

    // Single thread in the (only) block finalizes and publishes the
    // verification result. __threadfence_system ensures the host sees the
    // result after stream synchronize.
    if (threadIdx.x == 0) {
        uint32_t observed = s_crc;
        int32_t result = (observed == expected_checksum) ? 0 : -1;
        __threadfence_system();
        *verification_result = result;
    }
}

// ============================================================================
// drain_inflight_queue
// ============================================================================

/**
 * Header layout must match the Rust `K2KInboxHeader` (see
 * `ringkernel-cuda/src/k2k_gpu.rs` and the matching fields on
 * `SpscQueueHeader` in `persistent.rs`):
 *
 *     struct {
 *         uint64_t head;
 *         uint64_t tail;
 *         uint32_t capacity;
 *         uint32_t mask;
 *         uint32_t msg_size;
 *         uint32_t _padding;
 *         // (K2KInboxHeader total = 64 bytes, aligned)
 *     }
 *
 * Only `head`/`tail`/`capacity`/`mask`/`msg_size` are interpreted. The
 * caller guarantees `capacity` is a power of two and `mask == capacity - 1`.
 *
 *   k2k_queue_ptr     device pointer to queue header followed by slots
 *   queue_capacity    number of slots (for sanity-check against header)
 *   drain_buffer      output buffer of at least `capacity * msg_size` bytes
 *   messages_drained  out: number of messages copied out (mapped, one u64)
 */
extern "C" __global__ void drain_inflight_queue(
    void* __restrict__ k2k_queue_ptr,
    size_t queue_capacity,
    void* __restrict__ drain_buffer,
    uint64_t* __restrict__ messages_drained
) {
    uint8_t* base = static_cast<uint8_t*>(k2k_queue_ptr);

    uint64_t* head_ptr = reinterpret_cast<uint64_t*>(base + 0);
    uint64_t* tail_ptr = reinterpret_cast<uint64_t*>(base + 8);
    uint32_t* cap_ptr  = reinterpret_cast<uint32_t*>(base + 16);
    uint32_t* mask_ptr = reinterpret_cast<uint32_t*>(base + 20);
    uint32_t* msgsz_ptr = reinterpret_cast<uint32_t*>(base + 24);

    const uint64_t head = *head_ptr;
    const uint64_t tail = *tail_ptr;
    const uint32_t capacity = *cap_ptr;
    const uint32_t mask = *mask_ptr;
    const uint32_t msg_size = *msgsz_ptr;

    // Defensive sanity checks — the caller already guaranteed these at
    // allocation time, but if the queue is corrupt we just drain nothing.
    if (msg_size == 0 || capacity == 0 ||
        static_cast<size_t>(capacity) != queue_capacity) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            __threadfence_system();
            *messages_drained = 0ULL;
        }
        return;
    }

    // Slots live immediately after the 64-byte header.
    const uint8_t* slots = base + 64;
    uint8_t* drain = static_cast<uint8_t*>(drain_buffer);

    // Number of messages pending = head - tail (wrap-safe subtraction).
    const uint64_t pending = head - tail;

    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;

    // Copy each pending message sequentially into the drain buffer,
    // preserving order.
    for (uint64_t i = tid; i < pending; i += stride) {
        const uint64_t src_slot = (tail + i) & static_cast<uint64_t>(mask);
        const uint8_t* src = slots + src_slot * msg_size;
        uint8_t* dst = drain + i * msg_size;

        // Word-aligned copy where possible. `msg_size` is ≥ 16 bytes for
        // all in-tree message types; the caller guarantees alignment.
        const size_t words = msg_size / 4;
        const uint32_t* src32 = reinterpret_cast<const uint32_t*>(src);
        uint32_t* dst32 = reinterpret_cast<uint32_t*>(dst);
        for (size_t w = 0; w < words; ++w) {
            dst32[w] = src32[w];
        }
        // Tail bytes (usually zero).
        for (size_t t = words * 4; t < msg_size; ++t) {
            dst[t] = src[t];
        }
    }

    __threadfence();

    // Advance the queue's tail by the pending count and publish the
    // drain count via mapped memory so the host sees it.
    if (tid == 0) {
        *tail_ptr = head; // mark the entire pending range as consumed
        __threadfence_system();
        *messages_drained = pending;
    }
}

// ============================================================================
// Hopper-specific optimizations (optional)
// ============================================================================
//
// A full DSMEM / cluster-tuned capture path lands with the 2×H100
// verification run (Phase 3 of the v1.1 release plan). The portable
// kernels above already run natively on Hopper — Hopper-specific launches
// just pick a larger block size via the Rust wrapper.
//
// The guard below is kept so that future Hopper-only helpers have a
// ready-made home without us having to touch the build script.
#if __CUDA_ARCH__ >= 900
// Intentionally empty for v1.1 — see note above.
#endif
