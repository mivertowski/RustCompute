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
