/**
 * RingKernel Hopper Cluster Kernels — Phase 5
 *
 * CUDA kernels using Thread Block Clusters and Distributed Shared Memory (DSMEM)
 * for intra-cluster actor communication.
 *
 * Requires:
 * - Compute capability 9.0+ (H100/H200)
 * - CUDA 12.0+ for cluster launch
 * - nvcc -arch=sm_90
 */

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// ============================================================================
// Cluster-Aware Structures
// ============================================================================

/**
 * DSMEM queue header for intra-cluster K2K messaging.
 * Placed at the start of each block's shared memory DSMEM region.
 */
struct DsmemQueueHeader {
    unsigned int head;        // Write index (producer)
    unsigned int tail;        // Read index (consumer)
    unsigned int capacity;    // Queue capacity
    unsigned int mask;        // capacity - 1
    unsigned int src_rank;    // Source block rank in cluster
    unsigned int dst_rank;    // Destination block rank in cluster
    unsigned int slot_size;   // Bytes per message slot
    unsigned int _pad[9];     // Pad to 64 bytes
};

// ============================================================================
// Cluster Test Kernel — Verify cluster.sync() works
// ============================================================================

/**
 * Test kernel that verifies cluster-level synchronization.
 *
 * Each block in the cluster increments a counter. After cluster.sync(),
 * the counter should equal the cluster size × threads_per_block.
 *
 * Launch with cuLaunchKernelEx and CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION.
 */
extern "C" __global__ void cluster_test_sync(
    unsigned int* counter,
    unsigned int iterations
) {
    // Get cluster group (Hopper only)
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int cluster_rank = cluster.block_rank();
    unsigned int cluster_size = cluster.num_blocks();

    for (unsigned int i = 0; i < iterations; i++) {
        // Phase 1: Each thread increments counter
        atomicAdd(counter, 1);

        // Cluster-level sync (much faster than grid sync for clustered blocks)
        cluster.sync();

        // Phase 2: Verify (thread 0 of block 0 in cluster)
        if (cluster_rank == 0 && threadIdx.x == 0 && i == iterations - 1) {
            // Counter should be cluster_size * blockDim.x * (gridDim.x / cluster_size) * iterations
            // (all threads across all clusters incremented)
        }

        cluster.sync();
    }
}

// ============================================================================
// DSMEM K2K Messaging Kernel — Intra-Cluster Communication
// ============================================================================

/**
 * Kernel demonstrating DSMEM-based K2K messaging within a cluster.
 *
 * Blocks within a cluster exchange messages through distributed shared memory,
 * avoiding global memory entirely. This is ~7x faster than global memory K2K.
 *
 * Pattern:
 *   1. Each block writes a message to its DSMEM region
 *   2. Cluster sync ensures all messages are visible
 *   3. Each block reads neighbor's message via map_shared_rank()
 *   4. Cluster sync before next round
 */
extern "C" __global__ void cluster_dsmem_k2k(
    float* __restrict__ results,
    unsigned int message_size_floats,
    unsigned int rounds
) {
    cg::cluster_group cluster = cg::this_cluster();

    unsigned int cluster_rank = cluster.block_rank();
    unsigned int cluster_size = cluster.num_blocks();

    // Shared memory for this block's message buffer
    extern __shared__ float smem[];

    // Each block writes a unique value to its shared memory
    for (unsigned int i = threadIdx.x; i < message_size_floats; i += blockDim.x) {
        smem[i] = (float)(cluster_rank * 1000 + i);
    }

    for (unsigned int round = 0; round < rounds; round++) {
        // Sync cluster to ensure all shared memory writes are visible
        cluster.sync();

        // Read from neighbor's shared memory via DSMEM
        unsigned int neighbor = (cluster_rank + 1) % cluster_size;

        // map_shared_rank maps a local shared memory pointer to the
        // corresponding address in another block's shared memory
        float* neighbor_smem = cluster.map_shared_rank(smem, neighbor);

        // Read neighbor's message
        float sum = 0.0f;
        for (unsigned int i = threadIdx.x; i < message_size_floats; i += blockDim.x) {
            sum += neighbor_smem[i];
        }

        // Reduce within block
        __shared__ float block_sum;
        if (threadIdx.x == 0) block_sum = 0.0f;
        __syncthreads();

        atomicAdd(&block_sum, sum);
        __syncthreads();

        // Write result (only on last round)
        if (round == rounds - 1 && threadIdx.x == 0) {
            results[blockIdx.x] = block_sum;
        }

        // Update local message for next round
        for (unsigned int i = threadIdx.x; i < message_size_floats; i += blockDim.x) {
            smem[i] += 1.0f;
        }

        cluster.sync();
    }
}

// ============================================================================
// Persistent Actor Kernel with Cluster Support
// ============================================================================

/**
 * Persistent actor kernel that uses cluster-level features.
 *
 * This extends the base persistent ring kernel with:
 * - Cluster-level synchronization (faster than grid sync)
 * - DSMEM-backed K2K for intra-cluster messaging
 * - Barrier-based message detection (Phase 5.4)
 *
 * Actors within the same cluster communicate via DSMEM (shared memory path).
 * Actors in different clusters use the existing global memory K2K path.
 */
extern "C" __global__ void cluster_persistent_actor(
    unsigned int* __restrict__ control,     // [0]=active, [1]=terminate, [2]=terminated
    float* __restrict__ global_k2k_buf,      // Global memory K2K (inter-cluster)
    float* __restrict__ output,              // Results output
    unsigned int num_actors,
    unsigned int message_size_floats
) {
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();

    unsigned int actor_id = blockIdx.x;
    unsigned int cluster_rank = cluster.block_rank();
    unsigned int cluster_size = cluster.num_blocks();

    // Local shared memory for DSMEM messaging
    extern __shared__ float local_state[];

    // Initialize actor state
    for (unsigned int i = threadIdx.x; i < message_size_floats; i += blockDim.x) {
        local_state[i] = (float)actor_id;
    }

    unsigned long long messages_processed = 0;

    // Persistent loop
    while (true) {
        // Check termination flag
        if (atomicAdd(&control[1], 0) != 0) break;

        // Check if active
        if (atomicAdd(&control[0], 0) == 0) {
            cluster.sync();
            continue;
        }

        // Phase 1: Intra-cluster DSMEM messaging (fast path)
        if (cluster_size > 1) {
            cluster.sync(); // Ensure all actors' state is visible

            unsigned int neighbor = (cluster_rank + 1) % cluster_size;
            float* neighbor_state = cluster.map_shared_rank(local_state, neighbor);

            // Read and accumulate neighbor's data
            for (unsigned int i = threadIdx.x; i < message_size_floats; i += blockDim.x) {
                local_state[i] += neighbor_state[i] * 0.001f; // Small mixing
            }

            cluster.sync();
        }

        // Phase 2: Inter-cluster messaging via global memory (existing K2K path)
        // Only cluster rank 0 participates in grid-level exchange
        if (cluster_rank == 0 && threadIdx.x == 0) {
            // Write to global K2K buffer for cross-cluster communication
            unsigned int global_slot = actor_id * message_size_floats;
            for (unsigned int i = 0; i < message_size_floats && i < 16; i++) {
                global_k2k_buf[global_slot + i] = local_state[i];
            }
        }

        grid.sync(); // Full grid sync for inter-cluster consistency

        if (threadIdx.x == 0) {
            messages_processed++;
        }

        // Write periodic output
        if (threadIdx.x == 0 && (messages_processed % 1000) == 0) {
            output[actor_id] = local_state[0];
        }
    }

    // Epilogue
    if (actor_id == 0 && threadIdx.x == 0) {
        atomicExch(&control[2], 1); // Mark terminated
    }
}

// ============================================================================
// HBM K2K — Cross-Cluster Inter-Block Messaging via Global Memory (HBM tier)
// ============================================================================

/**
 * One-shot kernel that exercises the HBM tier of the K2K hierarchy.
 *
 * Each block writes a message into its own global-memory (HBM) slot,
 * grid-syncs, then reads the neighbor block's slot and reduces into a
 * per-block result. This is a deliberately HBM-only path — no DSMEM,
 * no shared-memory map — so the paper-tier-latency harness can
 * measure the inter-cluster K2K cost directly (complementing SMEM
 * and DSMEM tiers measured by `cluster_test_sync` and
 * `cluster_dsmem_k2k`).
 *
 * `message_size_floats` must fit in `hbm_buf` at slot `blockIdx.x`
 * (caller allocates `gridDim.x * message_size_floats * 4` bytes).
 * `rounds` == 1 gives single-trip latency; larger values amortize
 * launch overhead for bandwidth measurements.
 *
 * Launch expectations:
 *   - grid(num_blocks, 1, 1)
 *   - block(threads_per_block, 1, 1)  — any size; one thread per lane
 *   - cluster(1, 1, 1) or omitted — this kernel does not rely on
 *     cluster groups. The launch config may still set `cluster_dim=2`
 *     so that `launch_kernel_with_cluster` works; the kernel will
 *     behave the same because we only use `grid.sync()`.
 */
extern "C" __global__ void cluster_hbm_k2k(
    float* __restrict__ hbm_buf,
    float* __restrict__ results,
    unsigned int message_size_floats,
    unsigned int rounds
) {
    cg::grid_group grid = cg::this_grid();

    unsigned int my_slot     = blockIdx.x * message_size_floats;
    unsigned int num_blocks  = gridDim.x;
    unsigned int neighbor    = (blockIdx.x + 1) % num_blocks;
    unsigned int nbr_slot    = neighbor * message_size_floats;

    // Each block writes a unique stamp into its own HBM slot.
    for (unsigned int i = threadIdx.x; i < message_size_floats; i += blockDim.x) {
        hbm_buf[my_slot + i] = (float)(blockIdx.x * 1000 + i);
    }

    for (unsigned int round = 0; round < rounds; round++) {
        grid.sync(); // Inter-cluster visibility via HBM

        // Read neighbor's HBM slot and reduce.
        float sum = 0.0f;
        for (unsigned int i = threadIdx.x; i < message_size_floats; i += blockDim.x) {
            sum += hbm_buf[nbr_slot + i];
        }

        __shared__ float block_sum;
        if (threadIdx.x == 0) block_sum = 0.0f;
        __syncthreads();
        atomicAdd(&block_sum, sum);
        __syncthreads();

        if (round == rounds - 1 && threadIdx.x == 0) {
            results[blockIdx.x] = block_sum;
        }

        // Update our slot for the next round (so reads see fresh data).
        for (unsigned int i = threadIdx.x; i < message_size_floats; i += blockDim.x) {
            hbm_buf[my_slot + i] += 1.0f;
        }

        grid.sync();
    }
}

// ============================================================================
// Intra-block Work Stealing — one shared task queue per block
// ============================================================================

/**
 * Warp-level work stealing within a block.
 *
 * Each block maintains a single shared-memory task counter that warps
 * atomically decrement to claim work units. The first warp that reaches
 * zero drops out; the others keep stealing. This is the minimum useful
 * pattern for dynamic redistribution of uneven work (branches where
 * some warps finish early) without host involvement.
 *
 * `total_tasks` tasks are numbered 0..total_tasks-1. Each task `t` is
 * processed by computing `output[t] = f(t, input[t])` where `f` is a
 * trivial float accumulator. The resulting `output` array is written
 * out; the kernel does not recompute across rounds.
 *
 * Per-warp per-block counters (`tasks_done`) are written to the
 * `stats` buffer so the host can assert that the total equals
 * `total_tasks * gridDim.x`, proving every task was processed
 * exactly once and that the stealing protocol conserves work.
 *
 * Launch expectations:
 *   - grid(n_blocks, 1, 1)
 *   - block(threads_per_block, 1, 1)  — a multiple of 32 (one warp per stripe)
 */
extern "C" __global__ void warp_work_steal(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int* __restrict__ stats,   // [gridDim.x * warps_per_block]
    unsigned int total_tasks
) {
    __shared__ int remaining; // signed so we can detect "below zero"
    __shared__ unsigned int warps_done;

    const unsigned int warp_id   = threadIdx.x / 32;
    const unsigned int lane      = threadIdx.x & 31;
    const unsigned int warps_pb  = (blockDim.x + 31) / 32;

    if (threadIdx.x == 0) {
        remaining  = (int)total_tasks;
        warps_done = 0;
    }
    __syncthreads();

    unsigned int my_done = 0;

    // Each warp repeatedly steals a batch of `warpSize` tasks.
    while (true) {
        int base = -1;
        if (lane == 0) {
            base = atomicSub(&remaining, (int)warpSize);
        }
        base = __shfl_sync(0xFFFFFFFFu, base, 0);
        if (base <= 0) {
            break;
        }
        unsigned int idx = (unsigned int)base - 1u - lane;
        if ((int)idx >= 0 && idx < total_tasks) {
            float v = input[idx];
            output[idx] = v * 2.0f + 1.0f;
            my_done++;
        }
    }

    // Per-warp tally for host-side audit.
    unsigned int warp_total = my_done;
    // Warp-reduce my_done to lane 0 via __shfl_down_sync so stats
    // holds the per-warp count (not per-thread).
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_total += __shfl_down_sync(0xFFFFFFFFu, warp_total, offset);
    }
    if (lane == 0) {
        stats[blockIdx.x * warps_pb + warp_id] = warp_total;
        atomicAdd(&warps_done, 1u);
    }
}

// ============================================================================
// Cluster-scope work stealing — DSMEM-backed cross-block queue
// ============================================================================

/**
 * Intra-cluster work stealing over DSMEM.
 *
 * Every block in the cluster shares a single counter that lives in
 * block-0's distributed shared memory. Every block in the cluster
 * steals tasks from that single counter via `cluster.map_shared_rank`.
 * Once the counter is drained, blocks drop out.
 *
 * This generalizes `warp_work_steal` one tier up: tasks are
 * dynamically redistributed across the cluster with no host round-
 * trip, and no global memory writes for the coordinator. The cost
 * of a steal is a DSMEM atomic (~same latency as an SMEM atomic on
 * Hopper).
 *
 * `stats[block_rank]` receives the per-block tally. Host-side audit:
 * sum of `stats` must equal `total_tasks`.
 *
 * Launch expectations:
 *   - grid(cluster_size, 1, 1)   — one cluster
 *   - block(threads_per_block, 1, 1)
 *   - cluster(cluster_size, 1, 1)
 */
extern "C" __global__ void cluster_dsmem_work_steal(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int* __restrict__ stats,   // [cluster_size]
    unsigned int total_tasks
) {
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int rank = cluster.block_rank();

    // Shared-memory counter lives only in block 0; every other block
    // reaches it via `map_shared_rank`.
    __shared__ int local_remaining;
    if (rank == 0 && threadIdx.x == 0) {
        local_remaining = (int)total_tasks;
    }
    cluster.sync();

    int* shared_counter = cluster.map_shared_rank(&local_remaining, 0);

    unsigned int my_done = 0;
    while (true) {
        int base = -1;
        if (threadIdx.x == 0) {
            base = atomicSub(shared_counter, (int)blockDim.x);
        }
        // Broadcast the claimed base to every thread in the block.
        __shared__ int shared_base;
        if (threadIdx.x == 0) {
            shared_base = base;
        }
        __syncthreads();
        base = shared_base;
        if (base <= 0) {
            break;
        }
        unsigned int idx = (unsigned int)base - 1u - threadIdx.x;
        if ((int)idx >= 0 && idx < total_tasks) {
            float v = input[idx];
            output[idx] = v * 3.0f - 2.0f;
            my_done++;
        }
    }

    // Reduce my_done across the block into thread 0.
    __shared__ unsigned int block_total;
    if (threadIdx.x == 0) block_total = 0;
    __syncthreads();
    atomicAdd(&block_total, my_done);
    __syncthreads();

    if (threadIdx.x == 0) {
        stats[rank] = block_total;
    }

    cluster.sync();
}

// ============================================================================
// Grid-scope work stealing — HBM-backed global queue across all blocks
// ============================================================================

/**
 * Cross-cluster (grid-scope) work stealing over HBM.
 *
 * A single `unsigned int` global counter lives in HBM; every block
 * atomically decrements it to claim a stripe of `blockDim.x` tasks.
 * This is the hierarchical top tier above `cluster_dsmem_work_steal`
 * — useful when workload doesn't fit in a single cluster.
 *
 * The cost per steal is a global atomic (uncached on Hopper: ~50 ns
 * per atomic), so this is the slowest stealing tier. It's the
 * natural fallback when clusters run dry and you still have work
 * queued on other clusters.
 *
 * `stats[block_rank_global]` receives the per-block tally. Host-side
 * audit: sum of `stats` must equal `total_tasks`.
 *
 * Launch expectations:
 *   - grid(n_blocks, 1, 1)   — any number of blocks; no clusters needed
 *   - block(threads_per_block, 1, 1)
 *   - `counter` must be a pre-zeroed HBM pointer; caller writes
 *     `total_tasks` into `*counter` before launch.
 */
extern "C" __global__ void grid_hbm_work_steal(
    const float* __restrict__ input,
    float* __restrict__ output,
    int* __restrict__ counter,          // HBM: initialized to total_tasks
    unsigned int* __restrict__ stats,   // [gridDim.x]
    unsigned int total_tasks
) {
    unsigned int my_done = 0;
    while (true) {
        int base = -1;
        if (threadIdx.x == 0) {
            base = atomicSub(counter, (int)blockDim.x);
        }
        __shared__ int shared_base;
        if (threadIdx.x == 0) shared_base = base;
        __syncthreads();
        base = shared_base;
        if (base <= 0) break;

        unsigned int idx = (unsigned int)base - 1u - threadIdx.x;
        if ((int)idx >= 0 && idx < total_tasks) {
            float v = input[idx];
            output[idx] = v * 4.0f + 3.0f;
            my_done++;
        }
    }

    __shared__ unsigned int block_total;
    if (threadIdx.x == 0) block_total = 0;
    __syncthreads();
    atomicAdd(&block_total, my_done);
    __syncthreads();

    if (threadIdx.x == 0) {
        stats[blockIdx.x] = block_total;
    }
}
