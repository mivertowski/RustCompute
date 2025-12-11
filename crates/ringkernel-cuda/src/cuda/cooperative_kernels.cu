/**
 * RingKernel Cooperative Groups Kernels
 *
 * Pre-compiled CUDA kernels that use cooperative groups for grid-wide synchronization.
 * These kernels require cuLaunchCooperativeKernel for launch.
 *
 * Compiled with nvcc at build time and embedded as PTX in the Rust binary.
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// ============================================================================
// Shared Structures (must match Rust definitions)
// ============================================================================

/**
 * Control block for ring kernel lifecycle management.
 * Layout must match ringkernel_core::memory::ControlBlock
 */
struct ControlBlock {
    unsigned int is_active;
    unsigned int should_terminate;
    unsigned int has_terminated;
    unsigned int _pad1;

    unsigned long long messages_processed;
    unsigned long long messages_in_flight;

    unsigned long long input_head;
    unsigned long long input_tail;
    unsigned long long output_head;
    unsigned long long output_tail;

    unsigned int input_capacity;
    unsigned int output_capacity;
    unsigned int input_mask;
    unsigned int output_mask;

    struct {
        unsigned long long physical;
        unsigned long long logical;
    } hlc_state;

    unsigned int last_error;
    unsigned int error_count;

    unsigned char _reserved[24];
};

// ============================================================================
// Block-Based FDTD Structures (for wavesim3d)
// ============================================================================

#define BLOCK_SIZE 8
#define CELLS_PER_BLOCK 512  // 8 * 8 * 8
#define FACE_SIZE 64         // 8 * 8

/**
 * Per-block state for FDTD simulation.
 * Must match wavesim3d BlockState.
 */
struct BlockState {
    float pressure[CELLS_PER_BLOCK];
    float pressure_prev[CELLS_PER_BLOCK];
    float reflection_coeff[CELLS_PER_BLOCK];
    unsigned char cell_type[CELLS_PER_BLOCK];
};

/**
 * Ghost faces for inter-block communication.
 * 6 faces: -X, +X, -Y, +Y, -Z, +Z
 */
struct GhostFaces {
    float faces[6][FACE_SIZE];
};

/**
 * Parameters for persistent FDTD kernel.
 */
struct PersistentFdtdParams {
    unsigned int blocks_x;
    unsigned int blocks_y;
    unsigned int blocks_z;
    unsigned int cells_x;
    unsigned int cells_y;
    unsigned int cells_z;
    float c_squared;
    float damping;
    unsigned int num_steps;
    unsigned int current_step;
};

// ============================================================================
// Device Helper Functions
// ============================================================================

/**
 * Convert local 3D index to linear index within a block.
 */
__device__ __forceinline__ int local_to_idx(int x, int y, int z) {
    return z * BLOCK_SIZE * BLOCK_SIZE + y * BLOCK_SIZE + x;
}

/**
 * Get neighbor block index given current block position and face.
 * Returns -1 if neighbor doesn't exist (boundary).
 */
__device__ int get_neighbor_block(
    int bx, int by, int bz,
    int face,
    int blocks_x, int blocks_y, int blocks_z
) {
    int nx = bx, ny = by, nz = bz;
    switch (face) {
        case 0: nx = bx - 1; break;  // NegX
        case 1: nx = bx + 1; break;  // PosX
        case 2: ny = by - 1; break;  // NegY
        case 3: ny = by + 1; break;  // PosY
        case 4: nz = bz - 1; break;  // NegZ
        case 5: nz = bz + 1; break;  // PosZ
    }
    if (nx < 0 || nx >= blocks_x || ny < 0 || ny >= blocks_y || nz < 0 || nz >= blocks_z) {
        return -1;
    }
    return nz * blocks_y * blocks_x + ny * blocks_x + nx;
}

// ============================================================================
// Cooperative FDTD Kernel - Persistent Multi-Step Execution
// ============================================================================

/**
 * Persistent FDTD kernel with cooperative groups for grid-wide synchronization.
 *
 * This kernel runs multiple simulation steps in a single launch:
 * 1. All blocks extract boundary faces to neighbor ghost buffers
 * 2. Grid-wide sync ensures all face data is written
 * 3. All blocks compute FDTD using ghost data
 * 4. Grid-wide sync before next step
 *
 * Launch with cuLaunchCooperativeKernel.
 * Grid size must not exceed device's max concurrent blocks.
 */
extern "C" __global__ void coop_persistent_fdtd(
    BlockState* __restrict__ blocks,
    GhostFaces* __restrict__ ghost_a,
    GhostFaces* __restrict__ ghost_b,
    PersistentFdtdParams* __restrict__ params
) {
    // Get cooperative grid for grid-wide synchronization
    cg::grid_group grid = cg::this_grid();

    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    int total_blocks = params->blocks_x * params->blocks_y * params->blocks_z;
    if (block_idx >= total_blocks) return;

    // Calculate 3D block position
    int bz = block_idx / (params->blocks_y * params->blocks_x);
    int rem = block_idx % (params->blocks_y * params->blocks_x);
    int by = rem / params->blocks_x;
    int bx = rem % params->blocks_x;

    BlockState* my_block = &blocks[block_idx];

    // Run all steps in a single kernel launch
    for (unsigned int step = 0; step < params->num_steps; step++) {
        // Double-buffer ghost faces
        GhostFaces* ghost_write = (step % 2 == 0) ? ghost_b : ghost_a;
        GhostFaces* ghost_read = (step % 2 == 0) ? ghost_a : ghost_b;

        // ================================================================
        // Phase 1: Extract faces and write to neighbor ghost buffers
        // ================================================================
        for (int i = tid; i < 6 * FACE_SIZE; i += blockDim.x) {
            int face = i / FACE_SIZE;
            int face_idx = i % FACE_SIZE;
            int fy = face_idx / BLOCK_SIZE;
            int fx = face_idx % BLOCK_SIZE;

            int neighbor_idx = get_neighbor_block(bx, by, bz, face,
                params->blocks_x, params->blocks_y, params->blocks_z);

            if (neighbor_idx < 0) continue;

            // Get local cell index for this face position
            int local_idx;
            switch (face) {
                case 0: local_idx = local_to_idx(0, fy, fx); break;              // NegX face
                case 1: local_idx = local_to_idx(BLOCK_SIZE-1, fy, fx); break;   // PosX face
                case 2: local_idx = local_to_idx(fx, 0, fy); break;              // NegY face
                case 3: local_idx = local_to_idx(fx, BLOCK_SIZE-1, fy); break;   // PosY face
                case 4: local_idx = local_to_idx(fx, fy, 0); break;              // NegZ face
                case 5: local_idx = local_to_idx(fx, fy, BLOCK_SIZE-1); break;   // PosZ face
            }

            float pressure = my_block->pressure[local_idx];

            // Write to neighbor's ghost buffer (opposite face)
            int opposite_face = face ^ 1;
            ghost_write[neighbor_idx].faces[opposite_face][face_idx] = pressure;
        }

        // ================================================================
        // GRID-WIDE SYNC - All blocks wait here until faces extracted
        // ================================================================
        grid.sync();

        // ================================================================
        // Phase 2: Compute FDTD using shared memory for local block
        // ================================================================
        __shared__ float s_pressure[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];
        __shared__ float s_pressure_prev[CELLS_PER_BLOCK];
        __shared__ float s_reflection[CELLS_PER_BLOCK];
        __shared__ unsigned char s_cell_type[CELLS_PER_BLOCK];

        GhostFaces* my_ghosts = &ghost_write[block_idx];

        // Load current pressure to shared memory with halo
        for (int i = tid; i < CELLS_PER_BLOCK; i += blockDim.x) {
            int lz = i / (BLOCK_SIZE * BLOCK_SIZE);
            int lrem = i % (BLOCK_SIZE * BLOCK_SIZE);
            int ly = lrem / BLOCK_SIZE;
            int lx = lrem % BLOCK_SIZE;

            s_pressure[lz + 1][ly + 1][lx + 1] = my_block->pressure[i];
            s_pressure_prev[i] = my_block->pressure_prev[i];
            s_reflection[i] = my_block->reflection_coeff[i];
            s_cell_type[i] = my_block->cell_type[i];
        }

        __syncthreads();

        // Load ghost faces into halo regions
        for (int i = tid; i < FACE_SIZE; i += blockDim.x) {
            int fy = i / BLOCK_SIZE;
            int fx = i % BLOCK_SIZE;

            bool has_neg_x = (bx > 0);
            bool has_pos_x = (bx < (int)params->blocks_x - 1);
            bool has_neg_y = (by > 0);
            bool has_pos_y = (by < (int)params->blocks_y - 1);
            bool has_neg_z = (bz > 0);
            bool has_pos_z = (bz < (int)params->blocks_z - 1);

            // X faces (note: shared memory indexing is [z][y][x])
            s_pressure[fx + 1][fy + 1][0] = has_neg_x ? my_ghosts->faces[0][i] : s_pressure[fx + 1][fy + 1][1];
            s_pressure[fx + 1][fy + 1][BLOCK_SIZE + 1] = has_pos_x ? my_ghosts->faces[1][i] : s_pressure[fx + 1][fy + 1][BLOCK_SIZE];

            // Y faces
            s_pressure[fy + 1][0][fx + 1] = has_neg_y ? my_ghosts->faces[2][i] : s_pressure[fy + 1][1][fx + 1];
            s_pressure[fy + 1][BLOCK_SIZE + 1][fx + 1] = has_pos_y ? my_ghosts->faces[3][i] : s_pressure[fy + 1][BLOCK_SIZE][fx + 1];

            // Z faces
            s_pressure[0][fy + 1][fx + 1] = has_neg_z ? my_ghosts->faces[4][i] : s_pressure[1][fy + 1][fx + 1];
            s_pressure[BLOCK_SIZE + 1][fy + 1][fx + 1] = has_pos_z ? my_ghosts->faces[5][i] : s_pressure[BLOCK_SIZE][fy + 1][fx + 1];
        }

        __syncthreads();

        // Compute FDTD update for all cells in block
        for (int i = tid; i < CELLS_PER_BLOCK; i += blockDim.x) {
            int lz = i / (BLOCK_SIZE * BLOCK_SIZE);
            int lrem = i % (BLOCK_SIZE * BLOCK_SIZE);
            int ly = lrem / BLOCK_SIZE;
            int lx = lrem % BLOCK_SIZE;

            // Skip obstacles (cell_type == 3)
            if (s_cell_type[i] == 3) continue;

            // Shared memory indices (with +1 offset for halo)
            int sz = lz + 1, sy = ly + 1, sx = lx + 1;

            float center = s_pressure[sz][sy][sx];

            // 7-point Laplacian stencil
            float laplacian =
                s_pressure[sz][sy][sx - 1] + s_pressure[sz][sy][sx + 1] +  // X neighbors
                s_pressure[sz][sy - 1][sx] + s_pressure[sz][sy + 1][sx] +  // Y neighbors
                s_pressure[sz - 1][sy][sx] + s_pressure[sz + 1][sy][sx] -  // Z neighbors
                6.0f * center;

            // Wave equation: P_new = 2*P - P_prev + c^2 * laplacian
            float p_prev = s_pressure_prev[i];
            float p_new = 2.0f * center - p_prev + params->c_squared * laplacian;

            // Apply damping
            p_new *= params->damping;

            // Apply reflection at boundaries (cell_type == 1)
            if (s_cell_type[i] == 1) {
                p_new *= s_reflection[i];
            }

            // Write results (swap buffers)
            my_block->pressure_prev[i] = center;
            my_block->pressure[i] = p_new;
        }

        // ================================================================
        // GRID-WIDE SYNC - All blocks wait here before next step
        // ================================================================
        grid.sync();
    }
}

// ============================================================================
// Generic Cooperative Ring Kernel
// ============================================================================

/**
 * Generic cooperative ring kernel for persistent actor execution.
 *
 * This provides a framework for user-defined handlers with grid-wide sync.
 * User code is injected via codegen into the handler section.
 */
extern "C" __global__ void coop_ring_kernel_entry(
    ControlBlock* __restrict__ control,
    unsigned char* __restrict__ input_buffer,
    unsigned char* __restrict__ output_buffer,
    void* __restrict__ shared_state,
    unsigned int msg_size,
    unsigned int resp_size,
    unsigned int queue_mask
) {
    cg::grid_group grid = cg::this_grid();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // HLC state (per-thread for now, could be per-block)
    unsigned long long hlc_physical = 0;
    unsigned long long hlc_logical = 0;

    // Persistent loop
    while (true) {
        // Check termination
        if (atomicAdd(&control->should_terminate, 0) != 0) {
            break;
        }

        // Check if active
        if (atomicAdd(&control->is_active, 0) == 0) {
            // Grid-wide sync before checking again
            grid.sync();
            __nanosleep(1000);  // 1us sleep to reduce polling
            continue;
        }

        // Check for messages (only thread 0 of each block checks queue)
        unsigned long long head = 0, tail = 0;
        if (threadIdx.x == 0) {
            head = atomicAdd(&control->input_head, 0);
            tail = atomicAdd(&control->input_tail, 0);
        }

        // Broadcast to all threads in block
        head = __shfl_sync(0xFFFFFFFF, head, 0);
        tail = __shfl_sync(0xFFFFFFFF, tail, 0);

        if (head == tail) {
            // No messages - grid sync before next iteration
            grid.sync();
            continue;
        }

        // === User handler code would be inserted here by codegen ===

        // Process message (placeholder - actual code injected by transpiler)
        if (tid == 0) {
            unsigned int msg_idx = (unsigned int)(tail & queue_mask);
            // unsigned char* msg_ptr = &input_buffer[msg_idx * msg_size];

            // Advance queue
            atomicAdd(&control->input_tail, 1);
            atomicAdd(&control->messages_processed, 1);
        }

        // Grid-wide sync after processing
        grid.sync();
    }

    // Epilogue: store HLC state and mark terminated
    if (tid == 0) {
        control->hlc_state.physical = hlc_physical;
        control->hlc_state.logical = hlc_logical;
        atomicExch(&control->has_terminated, 1);
    }
}

// ============================================================================
// Utility Kernels
// ============================================================================

/**
 * Simple cooperative kernel for testing grid.sync() functionality.
 * Increments a counter in two phases with grid sync between them.
 */
extern "C" __global__ void coop_test_grid_sync(
    unsigned int* counter,
    unsigned int iterations
) {
    cg::grid_group grid = cg::this_grid();
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (unsigned int i = 0; i < iterations; i++) {
        // Phase 1: All threads increment counter
        atomicAdd(counter, 1);

        // Grid-wide sync
        grid.sync();

        // Phase 2: Thread 0 can read final count
        if (tid == 0 && i == iterations - 1) {
            // Counter should equal gridDim.x * blockDim.x * iterations
        }

        grid.sync();
    }
}
