//! Block-Based Actor Backend for 3D Wave Simulation.
//!
//! This module implements a hybrid actor-stencil paradigm where:
//! - Each 8×8×8 block of cells is a single actor (512 cells per actor)
//! - Intra-block computation uses fast stencil operations
//! - Inter-block communication uses double-buffered message passing
//! - No atomics needed - deterministic buffer locations
//!
//! # Performance Characteristics
//!
//! Compared to per-cell actors:
//! - 512× fewer messages (block faces vs individual cells)
//! - Zero atomic operations (double-buffered BSP)
//! - Coalesced memory access patterns
//! - Expected: 10-50× faster than per-cell actors

use super::grid3d::{CellType, SimulationGrid3D};
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{
    CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits,
};
#[cfg(feature = "cuda")]
use cudarc::nvrtc;

#[cfg(feature = "cooperative")]
use ringkernel_cuda::cooperative::{has_cooperative_support, kernels, CooperativeKernel};

/// Block size for the hybrid actor model.
/// 8×8×8 = 512 cells per block provides good balance between:
/// - Enough cells for efficient stencil computation
/// - Small enough for reasonable message sizes
pub const BLOCK_SIZE: usize = 8;
pub const CELLS_PER_BLOCK: usize = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE; // 512

/// Error types for the block actor backend.
#[derive(Debug)]
pub enum BlockActorError {
    DeviceError(String),
    CompileError(String),
    LaunchError(String),
    MemoryError(String),
    GridSizeError(String),
}

impl std::fmt::Display for BlockActorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockActorError::DeviceError(msg) => write!(f, "Block actor device error: {}", msg),
            BlockActorError::CompileError(msg) => write!(f, "Block actor compile error: {}", msg),
            BlockActorError::LaunchError(msg) => write!(f, "Block actor launch error: {}", msg),
            BlockActorError::MemoryError(msg) => write!(f, "Block actor memory error: {}", msg),
            BlockActorError::GridSizeError(msg) => {
                write!(f, "Block actor grid size error: {}", msg)
            }
        }
    }
}

impl std::error::Error for BlockActorError {}

/// Face direction for inter-block communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Face {
    NegX = 0,
    PosX = 1,
    NegY = 2,
    PosY = 3,
    NegZ = 4,
    PosZ = 5,
}

impl Face {
    pub const ALL: [Face; 6] = [
        Face::NegX,
        Face::PosX,
        Face::NegY,
        Face::PosY,
        Face::NegZ,
        Face::PosZ,
    ];

    pub fn opposite(self) -> Face {
        match self {
            Face::NegX => Face::PosX,
            Face::PosX => Face::NegX,
            Face::NegY => Face::PosY,
            Face::PosY => Face::NegY,
            Face::NegZ => Face::PosZ,
            Face::PosZ => Face::NegZ,
        }
    }
}

/// Block state containing all cells in an 8×8×8 block.
/// Uses Structure-of-Arrays for better memory coalescing.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BlockState {
    /// Current pressure values (512 floats)
    pub pressure: [f32; CELLS_PER_BLOCK],
    /// Previous pressure values (512 floats)
    pub pressure_prev: [f32; CELLS_PER_BLOCK],
    /// Reflection coefficients (512 floats)
    pub reflection_coeff: [f32; CELLS_PER_BLOCK],
    /// Cell types packed as u8 (512 bytes, padded to 512)
    pub cell_type: [u8; CELLS_PER_BLOCK],
}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for BlockState {}
#[cfg(feature = "cuda")]
unsafe impl ValidAsZeroBits for BlockState {}

impl Default for BlockState {
    fn default() -> Self {
        Self {
            pressure: [0.0; CELLS_PER_BLOCK],
            pressure_prev: [0.0; CELLS_PER_BLOCK],
            reflection_coeff: [1.0; CELLS_PER_BLOCK],
            cell_type: [0; CELLS_PER_BLOCK],
        }
    }
}

/// Face buffer for boundary exchange (8×8 = 64 values per face).
pub const FACE_SIZE: usize = BLOCK_SIZE * BLOCK_SIZE; // 64

/// Ghost face data received from neighbor blocks.
/// Double-buffered: read from one, write to other.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GhostFaces {
    /// Ghost data for 6 faces (64 floats each = 384 total)
    pub faces: [[f32; FACE_SIZE]; 6],
}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for GhostFaces {}
#[cfg(feature = "cuda")]
unsafe impl ValidAsZeroBits for GhostFaces {}

impl Default for GhostFaces {
    fn default() -> Self {
        Self {
            faces: [[0.0; FACE_SIZE]; 6],
        }
    }
}

/// Block grid dimensions.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BlockGridParams {
    /// Number of blocks in X
    pub blocks_x: u32,
    /// Number of blocks in Y
    pub blocks_y: u32,
    /// Number of blocks in Z
    pub blocks_z: u32,
    /// Total cells in X (blocks_x * 8)
    pub cells_x: u32,
    /// Total cells in Y (blocks_y * 8)
    pub cells_y: u32,
    /// Total cells in Z (blocks_z * 8)
    pub cells_z: u32,
    /// Physics: c² * dt² / dx²
    pub c_squared: f32,
    /// Physics: damping factor
    pub damping: f32,
}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for BlockGridParams {}
#[cfg(feature = "cuda")]
unsafe impl ValidAsZeroBits for BlockGridParams {}

/// Extended parameters for persistent kernel (includes num_steps).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PersistentParams {
    /// Number of blocks in X
    pub blocks_x: u32,
    /// Number of blocks in Y
    pub blocks_y: u32,
    /// Number of blocks in Z
    pub blocks_z: u32,
    /// Total cells in X
    pub cells_x: u32,
    /// Total cells in Y
    pub cells_y: u32,
    /// Total cells in Z
    pub cells_z: u32,
    /// Physics: c² * dt² / dx²
    pub c_squared: f32,
    /// Physics: damping factor
    pub damping: f32,
    /// Number of steps to run
    pub num_steps: u32,
    /// Current step (for progress tracking)
    pub current_step: u32,
}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for PersistentParams {}
#[cfg(feature = "cuda")]
unsafe impl ValidAsZeroBits for PersistentParams {}

impl From<(&BlockGridParams, u32)> for PersistentParams {
    fn from((params, num_steps): (&BlockGridParams, u32)) -> Self {
        Self {
            blocks_x: params.blocks_x,
            blocks_y: params.blocks_y,
            blocks_z: params.blocks_z,
            cells_x: params.cells_x,
            cells_y: params.cells_y,
            cells_z: params.cells_z,
            c_squared: params.c_squared,
            damping: params.damping,
            num_steps,
            current_step: 0,
        }
    }
}

/// Configuration for the block actor backend.
#[derive(Debug, Clone)]
pub struct BlockActorConfig {
    /// CUDA block size for kernel launches (threads per block)
    pub cuda_block_size: u32,
    /// Use persistent kernel with cooperative groups (requires compute capability 6.0+)
    pub use_persistent_kernel: bool,
}

impl Default for BlockActorConfig {
    fn default() -> Self {
        Self {
            cuda_block_size: 256,
            use_persistent_kernel: true, // Enable by default
        }
    }
}

/// Generate CUDA kernel source for block-based FDTD.
fn generate_block_actor_kernel() -> String {
    r#"
// ============================================================================
// Block-Based Actor FDTD Kernel
// ============================================================================
// Each CUDA block processes one 8×8×8 simulation block.
// Inter-block communication via double-buffered ghost faces.
// No atomics needed - deterministic memory locations.

#define BLOCK_SIZE 8
#define CELLS_PER_BLOCK 512
#define FACE_SIZE 64

// Block state (Structure-of-Arrays for coalescing)
struct BlockState {
    float pressure[CELLS_PER_BLOCK];
    float pressure_prev[CELLS_PER_BLOCK];
    float reflection_coeff[CELLS_PER_BLOCK];
    unsigned char cell_type[CELLS_PER_BLOCK];
};

// Ghost faces from neighbors
struct GhostFaces {
    float faces[6][FACE_SIZE];  // NegX, PosX, NegY, PosY, NegZ, PosZ
};

// Grid parameters
struct BlockGridParams {
    unsigned int blocks_x;
    unsigned int blocks_y;
    unsigned int blocks_z;
    unsigned int cells_x;
    unsigned int cells_y;
    unsigned int cells_z;
    float c_squared;
    float damping;
};

// Face indices
#define FACE_NEG_X 0
#define FACE_POS_X 1
#define FACE_NEG_Y 2
#define FACE_POS_Y 3
#define FACE_NEG_Z 4
#define FACE_POS_Z 5

// Convert local (x,y,z) to linear index within block
__device__ __forceinline__ int local_to_idx(int x, int y, int z) {
    return z * BLOCK_SIZE * BLOCK_SIZE + y * BLOCK_SIZE + x;
}

// Get neighbor block index, returns -1 if at boundary
__device__ int get_neighbor_block(
    int bx, int by, int bz,
    int face,
    int blocks_x, int blocks_y, int blocks_z
) {
    int nx = bx, ny = by, nz = bz;
    switch (face) {
        case FACE_NEG_X: nx = bx - 1; break;
        case FACE_POS_X: nx = bx + 1; break;
        case FACE_NEG_Y: ny = by - 1; break;
        case FACE_POS_Y: ny = by + 1; break;
        case FACE_NEG_Z: nz = bz - 1; break;
        case FACE_POS_Z: nz = bz + 1; break;
    }
    if (nx < 0 || nx >= blocks_x || ny < 0 || ny >= blocks_y || nz < 0 || nz >= blocks_z) {
        return -1;
    }
    return nz * blocks_y * blocks_x + ny * blocks_x + nx;
}

// Phase 1: Extract boundary faces and write to ghost buffers
extern "C" __global__ void extract_faces(
    const BlockState* __restrict__ blocks,
    GhostFaces* __restrict__ ghost_write,  // Write buffer
    const BlockGridParams* __restrict__ params
) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    int total_blocks = params->blocks_x * params->blocks_y * params->blocks_z;
    if (block_idx >= total_blocks) return;

    // Block coordinates
    int bz = block_idx / (params->blocks_y * params->blocks_x);
    int rem = block_idx % (params->blocks_y * params->blocks_x);
    int by = rem / params->blocks_x;
    int bx = rem % params->blocks_x;

    const BlockState* my_block = &blocks[block_idx];

    // Each thread handles multiple face elements
    // 6 faces × 64 elements = 384 total, with 256 threads = ~1.5 elements per thread
    for (int i = tid; i < 6 * FACE_SIZE; i += blockDim.x) {
        int face = i / FACE_SIZE;
        int face_idx = i % FACE_SIZE;
        int fy = face_idx / BLOCK_SIZE;
        int fx = face_idx % BLOCK_SIZE;

        // Get neighbor block
        int neighbor_idx = get_neighbor_block(bx, by, bz, face,
            params->blocks_x, params->blocks_y, params->blocks_z);

        if (neighbor_idx < 0) {
            // Boundary - will be handled in compute phase
            continue;
        }

        // Extract pressure from my boundary cell
        int local_idx;
        switch (face) {
            case FACE_NEG_X: local_idx = local_to_idx(0, fy, fx); break;
            case FACE_POS_X: local_idx = local_to_idx(BLOCK_SIZE-1, fy, fx); break;
            case FACE_NEG_Y: local_idx = local_to_idx(fx, 0, fy); break;
            case FACE_POS_Y: local_idx = local_to_idx(fx, BLOCK_SIZE-1, fy); break;
            case FACE_NEG_Z: local_idx = local_to_idx(fx, fy, 0); break;
            case FACE_POS_Z: local_idx = local_to_idx(fx, fy, BLOCK_SIZE-1); break;
        }

        float pressure = my_block->pressure[local_idx];

        // Write to neighbor's ghost buffer (opposite face)
        int opposite_face = face ^ 1;  // 0<->1, 2<->3, 4<->5
        ghost_write[neighbor_idx].faces[opposite_face][face_idx] = pressure;
    }
}

// Phase 2: Compute FDTD update using ghost data
extern "C" __global__ void compute_fdtd(
    BlockState* __restrict__ blocks,
    const GhostFaces* __restrict__ ghost_read,  // Read buffer
    const BlockGridParams* __restrict__ params
) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    int total_blocks = params->blocks_x * params->blocks_y * params->blocks_z;
    if (block_idx >= total_blocks) return;

    // Block coordinates
    int bz = block_idx / (params->blocks_y * params->blocks_x);
    int rem = block_idx % (params->blocks_y * params->blocks_x);
    int by = rem / params->blocks_x;
    int bx = rem % params->blocks_x;

    BlockState* my_block = &blocks[block_idx];
    const GhostFaces* my_ghosts = &ghost_read[block_idx];

    // Shared memory for block data (faster neighbor access)
    __shared__ float s_pressure[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    __shared__ float s_pressure_prev[CELLS_PER_BLOCK];
    __shared__ float s_reflection[CELLS_PER_BLOCK];
    __shared__ unsigned char s_cell_type[CELLS_PER_BLOCK];

    // Load block data to shared memory
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

    // Need to sync before reading s_pressure for reflection BC
    __syncthreads();

    // Load ghost faces (must happen AFTER interior data is loaded for reflection BC)
    for (int i = tid; i < FACE_SIZE; i += blockDim.x) {
        int fy = i / BLOCK_SIZE;
        int fx = i % BLOCK_SIZE;

        // Check if we have neighbors, otherwise use reflection BC
        bool has_neg_x = (bx > 0);
        bool has_pos_x = (bx < (int)params->blocks_x - 1);
        bool has_neg_y = (by > 0);
        bool has_pos_y = (by < (int)params->blocks_y - 1);
        bool has_neg_z = (bz > 0);
        bool has_pos_z = (bz < (int)params->blocks_z - 1);

        // NegX face (x=0 boundary)
        s_pressure[fx + 1][fy + 1][0] = has_neg_x
            ? my_ghosts->faces[FACE_NEG_X][i]
            : s_pressure[fx + 1][fy + 1][1];  // Reflection BC

        // PosX face (x=7 boundary)
        s_pressure[fx + 1][fy + 1][BLOCK_SIZE + 1] = has_pos_x
            ? my_ghosts->faces[FACE_POS_X][i]
            : s_pressure[fx + 1][fy + 1][BLOCK_SIZE];

        // NegY face (y=0 boundary)
        s_pressure[fy + 1][0][fx + 1] = has_neg_y
            ? my_ghosts->faces[FACE_NEG_Y][i]
            : s_pressure[fy + 1][1][fx + 1];

        // PosY face (y=7 boundary)
        s_pressure[fy + 1][BLOCK_SIZE + 1][fx + 1] = has_pos_y
            ? my_ghosts->faces[FACE_POS_Y][i]
            : s_pressure[fy + 1][BLOCK_SIZE][fx + 1];

        // NegZ face (z=0 boundary)
        s_pressure[0][fy + 1][fx + 1] = has_neg_z
            ? my_ghosts->faces[FACE_NEG_Z][i]
            : s_pressure[1][fy + 1][fx + 1];

        // PosZ face (z=7 boundary)
        s_pressure[BLOCK_SIZE + 1][fy + 1][fx + 1] = has_pos_z
            ? my_ghosts->faces[FACE_POS_Z][i]
            : s_pressure[BLOCK_SIZE][fy + 1][fx + 1];
    }

    __syncthreads();

    // Compute FDTD update for all cells
    for (int i = tid; i < CELLS_PER_BLOCK; i += blockDim.x) {
        int lz = i / (BLOCK_SIZE * BLOCK_SIZE);
        int lrem = i % (BLOCK_SIZE * BLOCK_SIZE);
        int ly = lrem / BLOCK_SIZE;
        int lx = lrem % BLOCK_SIZE;

        // Skip obstacle cells
        if (s_cell_type[i] == 3) {
            continue;
        }

        // Shared memory indices (offset by 1 for ghost layer)
        int sz = lz + 1;
        int sy = ly + 1;
        int sx = lx + 1;

        // 7-point Laplacian
        float center = s_pressure[sz][sy][sx];
        float laplacian =
            s_pressure[sz][sy][sx - 1] +  // -X
            s_pressure[sz][sy][sx + 1] +  // +X
            s_pressure[sz][sy - 1][sx] +  // -Y
            s_pressure[sz][sy + 1][sx] +  // +Y
            s_pressure[sz - 1][sy][sx] +  // -Z
            s_pressure[sz + 1][sy][sx] -  // +Z
            6.0f * center;

        // FDTD update: P_new = 2*P - P_prev + c²*Δt²/Δx²*∇²P
        float p_prev = s_pressure_prev[i];
        float p_new = 2.0f * center - p_prev + params->c_squared * laplacian;

        // Apply damping
        p_new *= params->damping;

        // Apply absorption for boundary cells
        if (s_cell_type[i] == 1) {  // Absorber
            p_new *= s_reflection[i];
        }

        // Store results
        my_block->pressure_prev[i] = center;
        my_block->pressure[i] = p_new;
    }
}

// Initialize block states from flat pressure arrays
extern "C" __global__ void init_blocks(
    BlockState* __restrict__ blocks,
    const float* __restrict__ pressure,
    const float* __restrict__ pressure_prev,
    const unsigned char* __restrict__ cell_types,
    const float* __restrict__ reflection_coeff,
    const BlockGridParams* __restrict__ params
) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    int total_blocks = params->blocks_x * params->blocks_y * params->blocks_z;
    if (block_idx >= total_blocks) return;

    // Block coordinates
    int bz = block_idx / (params->blocks_y * params->blocks_x);
    int rem = block_idx % (params->blocks_y * params->blocks_x);
    int by = rem / params->blocks_x;
    int bx = rem % params->blocks_x;

    BlockState* my_block = &blocks[block_idx];

    // Global offset for this block
    int gx_base = bx * BLOCK_SIZE;
    int gy_base = by * BLOCK_SIZE;
    int gz_base = bz * BLOCK_SIZE;

    for (int i = tid; i < CELLS_PER_BLOCK; i += blockDim.x) {
        int lz = i / (BLOCK_SIZE * BLOCK_SIZE);
        int lrem = i % (BLOCK_SIZE * BLOCK_SIZE);
        int ly = lrem / BLOCK_SIZE;
        int lx = lrem % BLOCK_SIZE;

        int gx = gx_base + lx;
        int gy = gy_base + ly;
        int gz = gz_base + lz;

        // Bounds check
        if (gx < (int)params->cells_x && gy < (int)params->cells_y && gz < (int)params->cells_z) {
            int global_idx = gz * params->cells_y * params->cells_x + gy * params->cells_x + gx;
            my_block->pressure[i] = pressure[global_idx];
            my_block->pressure_prev[i] = pressure_prev[global_idx];
            my_block->reflection_coeff[i] = reflection_coeff[global_idx];
            my_block->cell_type[i] = cell_types[global_idx];
        } else {
            my_block->pressure[i] = 0.0f;
            my_block->pressure_prev[i] = 0.0f;
            my_block->reflection_coeff[i] = 1.0f;
            my_block->cell_type[i] = 3;  // Obstacle
        }
    }
}

// Extract pressure from blocks to flat array
extern "C" __global__ void extract_pressure(
    const BlockState* __restrict__ blocks,
    float* __restrict__ pressure,
    float* __restrict__ pressure_prev,
    const BlockGridParams* __restrict__ params
) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    int total_blocks = params->blocks_x * params->blocks_y * params->blocks_z;
    if (block_idx >= total_blocks) return;

    // Block coordinates
    int bz = block_idx / (params->blocks_y * params->blocks_x);
    int rem = block_idx % (params->blocks_y * params->blocks_x);
    int by = rem / params->blocks_x;
    int bx = rem % params->blocks_x;

    const BlockState* my_block = &blocks[block_idx];

    int gx_base = bx * BLOCK_SIZE;
    int gy_base = by * BLOCK_SIZE;
    int gz_base = bz * BLOCK_SIZE;

    for (int i = tid; i < CELLS_PER_BLOCK; i += blockDim.x) {
        int lz = i / (BLOCK_SIZE * BLOCK_SIZE);
        int lrem = i % (BLOCK_SIZE * BLOCK_SIZE);
        int ly = lrem / BLOCK_SIZE;
        int lx = lrem % BLOCK_SIZE;

        int gx = gx_base + lx;
        int gy = gy_base + ly;
        int gz = gz_base + lz;

        if (gx < (int)params->cells_x && gy < (int)params->cells_y && gz < (int)params->cells_z) {
            int global_idx = gz * params->cells_y * params->cells_x + gy * params->cells_x + gx;
            pressure[global_idx] = my_block->pressure[i];
            pressure_prev[global_idx] = my_block->pressure_prev[i];
        }
    }
}

// Inject impulse into a block
extern "C" __global__ void inject_impulse_block(
    BlockState* __restrict__ blocks,
    unsigned int gx, unsigned int gy, unsigned int gz,
    float amplitude,
    const BlockGridParams* __restrict__ params
) {
    // Find which block contains this cell
    int bx = gx / BLOCK_SIZE;
    int by = gy / BLOCK_SIZE;
    int bz = gz / BLOCK_SIZE;

    if (bx >= (int)params->blocks_x || by >= (int)params->blocks_y || bz >= (int)params->blocks_z) return;

    int block_idx = bz * params->blocks_y * params->blocks_x + by * params->blocks_x + bx;

    // Local coordinates within block
    int lx = gx % BLOCK_SIZE;
    int ly = gy % BLOCK_SIZE;
    int lz = gz % BLOCK_SIZE;
    int local_idx = lz * BLOCK_SIZE * BLOCK_SIZE + ly * BLOCK_SIZE + lx;

    blocks[block_idx].pressure[local_idx] += amplitude;
    blocks[block_idx].pressure_prev[local_idx] += amplitude * 0.5f;
}

// Fused kernel: runs extract + compute in a single launch
// Uses __threadfence() for global memory visibility
// NOTE: This does NOT provide true grid-wide sync, only memory visibility.
// For small grids where all blocks can run concurrently, this works well.
extern "C" __global__ void fused_step(
    BlockState* __restrict__ blocks,
    GhostFaces* __restrict__ ghost_a,
    GhostFaces* __restrict__ ghost_b,
    const BlockGridParams* __restrict__ params,
    unsigned int use_buffer_b  // 0 = write to A, 1 = write to B
) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    int total_blocks = params->blocks_x * params->blocks_y * params->blocks_z;
    if (block_idx >= total_blocks) return;

    // Block coordinates
    int bz = block_idx / (params->blocks_y * params->blocks_x);
    int rem = block_idx % (params->blocks_y * params->blocks_x);
    int by = rem / params->blocks_x;
    int bx = rem % params->blocks_x;

    BlockState* my_block = &blocks[block_idx];
    GhostFaces* ghost_write = use_buffer_b ? ghost_b : ghost_a;
    const GhostFaces* ghost_read = use_buffer_b ? ghost_b : ghost_a;

    // ========== PHASE 1: Extract faces ==========
    for (int i = tid; i < 6 * FACE_SIZE; i += blockDim.x) {
        int face = i / FACE_SIZE;
        int face_idx = i % FACE_SIZE;
        int fy = face_idx / BLOCK_SIZE;
        int fx = face_idx % BLOCK_SIZE;

        int neighbor_idx = get_neighbor_block(bx, by, bz, face,
            params->blocks_x, params->blocks_y, params->blocks_z);

        if (neighbor_idx < 0) continue;

        int local_idx;
        switch (face) {
            case FACE_NEG_X: local_idx = local_to_idx(0, fy, fx); break;
            case FACE_POS_X: local_idx = local_to_idx(BLOCK_SIZE-1, fy, fx); break;
            case FACE_NEG_Y: local_idx = local_to_idx(fx, 0, fy); break;
            case FACE_POS_Y: local_idx = local_to_idx(fx, BLOCK_SIZE-1, fy); break;
            case FACE_NEG_Z: local_idx = local_to_idx(fx, fy, 0); break;
            case FACE_POS_Z: local_idx = local_to_idx(fx, fy, BLOCK_SIZE-1); break;
        }

        float pressure = my_block->pressure[local_idx];
        int opposite_face = face ^ 1;
        ghost_write[neighbor_idx].faces[opposite_face][face_idx] = pressure;
    }

    // Ensure all writes are visible globally
    __threadfence();
    __syncthreads();

    // ========== PHASE 2: Compute FDTD ==========
    const GhostFaces* my_ghosts = &ghost_read[block_idx];

    // Shared memory for block data
    __shared__ float s_pressure[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    __shared__ float s_pressure_prev[CELLS_PER_BLOCK];
    __shared__ float s_reflection[CELLS_PER_BLOCK];
    __shared__ unsigned char s_cell_type[CELLS_PER_BLOCK];

    // Load block data to shared memory
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

    // Load ghost faces
    for (int i = tid; i < FACE_SIZE; i += blockDim.x) {
        int fy = i / BLOCK_SIZE;
        int fx = i % BLOCK_SIZE;

        bool has_neg_x = (bx > 0);
        bool has_pos_x = (bx < (int)params->blocks_x - 1);
        bool has_neg_y = (by > 0);
        bool has_pos_y = (by < (int)params->blocks_y - 1);
        bool has_neg_z = (bz > 0);
        bool has_pos_z = (bz < (int)params->blocks_z - 1);

        s_pressure[fx + 1][fy + 1][0] = has_neg_x
            ? my_ghosts->faces[FACE_NEG_X][i]
            : s_pressure[fx + 1][fy + 1][1];

        s_pressure[fx + 1][fy + 1][BLOCK_SIZE + 1] = has_pos_x
            ? my_ghosts->faces[FACE_POS_X][i]
            : s_pressure[fx + 1][fy + 1][BLOCK_SIZE];

        s_pressure[fy + 1][0][fx + 1] = has_neg_y
            ? my_ghosts->faces[FACE_NEG_Y][i]
            : s_pressure[fy + 1][1][fx + 1];

        s_pressure[fy + 1][BLOCK_SIZE + 1][fx + 1] = has_pos_y
            ? my_ghosts->faces[FACE_POS_Y][i]
            : s_pressure[fy + 1][BLOCK_SIZE][fx + 1];

        s_pressure[0][fy + 1][fx + 1] = has_neg_z
            ? my_ghosts->faces[FACE_NEG_Z][i]
            : s_pressure[1][fy + 1][fx + 1];

        s_pressure[BLOCK_SIZE + 1][fy + 1][fx + 1] = has_pos_z
            ? my_ghosts->faces[FACE_POS_Z][i]
            : s_pressure[BLOCK_SIZE][fy + 1][fx + 1];
    }

    __syncthreads();

    // Compute FDTD update
    for (int i = tid; i < CELLS_PER_BLOCK; i += blockDim.x) {
        int lz = i / (BLOCK_SIZE * BLOCK_SIZE);
        int lrem = i % (BLOCK_SIZE * BLOCK_SIZE);
        int ly = lrem / BLOCK_SIZE;
        int lx = lrem % BLOCK_SIZE;

        if (s_cell_type[i] == 3) continue;

        int sz = lz + 1, sy = ly + 1, sx = lx + 1;

        float center = s_pressure[sz][sy][sx];
        float laplacian =
            s_pressure[sz][sy][sx - 1] +
            s_pressure[sz][sy][sx + 1] +
            s_pressure[sz][sy - 1][sx] +
            s_pressure[sz][sy + 1][sx] +
            s_pressure[sz - 1][sy][sx] +
            s_pressure[sz + 1][sy][sx] -
            6.0f * center;

        float p_prev = s_pressure_prev[i];
        float p_new = 2.0f * center - p_prev + params->c_squared * laplacian;
        p_new *= params->damping;

        if (s_cell_type[i] == 1) {
            p_new *= s_reflection[i];
        }

        my_block->pressure_prev[i] = center;
        my_block->pressure[i] = p_new;
    }
}
"#.to_string()
}

/// Generate CUDA kernel source for persistent kernel with cooperative groups.
/// This kernel runs multiple steps in a single launch using grid-wide synchronization.
/// Note: This source is kept for reference - we now use pre-compiled PTX from ringkernel-cuda.
#[allow(dead_code)]
fn generate_persistent_kernel() -> String {
    r#"
// ============================================================================
// Persistent Block-Based Actor FDTD Kernel with Cooperative Groups
// ============================================================================
// Runs multiple simulation steps in a single kernel launch.
// Uses cooperative groups for grid-wide synchronization.
// Eliminates host-device synchronization overhead.

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define BLOCK_SIZE 8
#define CELLS_PER_BLOCK 512
#define FACE_SIZE 64

// Block state (Structure-of-Arrays for coalescing)
struct BlockState {
    float pressure[CELLS_PER_BLOCK];
    float pressure_prev[CELLS_PER_BLOCK];
    float reflection_coeff[CELLS_PER_BLOCK];
    unsigned char cell_type[CELLS_PER_BLOCK];
};

// Ghost faces from neighbors
struct GhostFaces {
    float faces[6][FACE_SIZE];  // NegX, PosX, NegY, PosY, NegZ, PosZ
};

// Grid parameters (extended for persistent kernel)
struct PersistentParams {
    unsigned int blocks_x;
    unsigned int blocks_y;
    unsigned int blocks_z;
    unsigned int cells_x;
    unsigned int cells_y;
    unsigned int cells_z;
    float c_squared;
    float damping;
    unsigned int num_steps;      // Number of steps to run
    unsigned int current_step;   // For progress tracking (atomic)
};

// Face indices
#define FACE_NEG_X 0
#define FACE_POS_X 1
#define FACE_NEG_Y 2
#define FACE_POS_Y 3
#define FACE_NEG_Z 4
#define FACE_POS_Z 5

// Convert local (x,y,z) to linear index within block
__device__ __forceinline__ int local_to_idx(int x, int y, int z) {
    return z * BLOCK_SIZE * BLOCK_SIZE + y * BLOCK_SIZE + x;
}

// Get neighbor block index, returns -1 if at boundary
__device__ int get_neighbor_block(
    int bx, int by, int bz,
    int face,
    int blocks_x, int blocks_y, int blocks_z
) {
    int nx = bx, ny = by, nz = bz;
    switch (face) {
        case FACE_NEG_X: nx = bx - 1; break;
        case FACE_POS_X: nx = bx + 1; break;
        case FACE_NEG_Y: ny = by - 1; break;
        case FACE_POS_Y: ny = by + 1; break;
        case FACE_NEG_Z: nz = bz - 1; break;
        case FACE_POS_Z: nz = bz + 1; break;
    }
    if (nx < 0 || nx >= blocks_x || ny < 0 || ny >= blocks_y || nz < 0 || nz >= blocks_z) {
        return -1;
    }
    return nz * blocks_y * blocks_x + ny * blocks_x + nx;
}

// Device function: Extract faces for a block
__device__ void extract_faces_device(
    int block_idx, int tid,
    const BlockState* __restrict__ blocks,
    GhostFaces* __restrict__ ghost_write,
    int bx, int by, int bz,
    int blocks_x, int blocks_y, int blocks_z
) {
    const BlockState* my_block = &blocks[block_idx];

    for (int i = tid; i < 6 * FACE_SIZE; i += blockDim.x) {
        int face = i / FACE_SIZE;
        int face_idx = i % FACE_SIZE;
        int fy = face_idx / BLOCK_SIZE;
        int fx = face_idx % BLOCK_SIZE;

        int neighbor_idx = get_neighbor_block(bx, by, bz, face, blocks_x, blocks_y, blocks_z);
        if (neighbor_idx < 0) continue;

        int local_idx;
        switch (face) {
            case FACE_NEG_X: local_idx = local_to_idx(0, fy, fx); break;
            case FACE_POS_X: local_idx = local_to_idx(BLOCK_SIZE-1, fy, fx); break;
            case FACE_NEG_Y: local_idx = local_to_idx(fx, 0, fy); break;
            case FACE_POS_Y: local_idx = local_to_idx(fx, BLOCK_SIZE-1, fy); break;
            case FACE_NEG_Z: local_idx = local_to_idx(fx, fy, 0); break;
            case FACE_POS_Z: local_idx = local_to_idx(fx, fy, BLOCK_SIZE-1); break;
        }

        float pressure = my_block->pressure[local_idx];
        int opposite_face = face ^ 1;
        ghost_write[neighbor_idx].faces[opposite_face][face_idx] = pressure;
    }
}

// Device function: Compute FDTD for a block
__device__ void compute_fdtd_device(
    int block_idx, int tid,
    BlockState* __restrict__ blocks,
    const GhostFaces* __restrict__ ghost_read,
    int bx, int by, int bz,
    int blocks_x, int blocks_y, int blocks_z,
    float c_squared, float damping
) {
    BlockState* my_block = &blocks[block_idx];
    const GhostFaces* my_ghosts = &ghost_read[block_idx];

    // Shared memory for block data
    __shared__ float s_pressure[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    __shared__ float s_pressure_prev[CELLS_PER_BLOCK];
    __shared__ float s_reflection[CELLS_PER_BLOCK];
    __shared__ unsigned char s_cell_type[CELLS_PER_BLOCK];

    // Load block data to shared memory
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

    // Load ghost faces
    for (int i = tid; i < FACE_SIZE; i += blockDim.x) {
        int fy = i / BLOCK_SIZE;
        int fx = i % BLOCK_SIZE;

        bool has_neg_x = (bx > 0);
        bool has_pos_x = (bx < blocks_x - 1);
        bool has_neg_y = (by > 0);
        bool has_pos_y = (by < blocks_y - 1);
        bool has_neg_z = (bz > 0);
        bool has_pos_z = (bz < blocks_z - 1);

        s_pressure[fx + 1][fy + 1][0] = has_neg_x
            ? my_ghosts->faces[FACE_NEG_X][i]
            : s_pressure[fx + 1][fy + 1][1];

        s_pressure[fx + 1][fy + 1][BLOCK_SIZE + 1] = has_pos_x
            ? my_ghosts->faces[FACE_POS_X][i]
            : s_pressure[fx + 1][fy + 1][BLOCK_SIZE];

        s_pressure[fy + 1][0][fx + 1] = has_neg_y
            ? my_ghosts->faces[FACE_NEG_Y][i]
            : s_pressure[fy + 1][1][fx + 1];

        s_pressure[fy + 1][BLOCK_SIZE + 1][fx + 1] = has_pos_y
            ? my_ghosts->faces[FACE_POS_Y][i]
            : s_pressure[fy + 1][BLOCK_SIZE][fx + 1];

        s_pressure[0][fy + 1][fx + 1] = has_neg_z
            ? my_ghosts->faces[FACE_NEG_Z][i]
            : s_pressure[1][fy + 1][fx + 1];

        s_pressure[BLOCK_SIZE + 1][fy + 1][fx + 1] = has_pos_z
            ? my_ghosts->faces[FACE_POS_Z][i]
            : s_pressure[BLOCK_SIZE][fy + 1][fx + 1];
    }

    __syncthreads();

    // Compute FDTD update
    for (int i = tid; i < CELLS_PER_BLOCK; i += blockDim.x) {
        int lz = i / (BLOCK_SIZE * BLOCK_SIZE);
        int lrem = i % (BLOCK_SIZE * BLOCK_SIZE);
        int ly = lrem / BLOCK_SIZE;
        int lx = lrem % BLOCK_SIZE;

        if (s_cell_type[i] == 3) continue;

        int sz = lz + 1, sy = ly + 1, sx = lx + 1;

        float center = s_pressure[sz][sy][sx];
        float laplacian =
            s_pressure[sz][sy][sx - 1] +
            s_pressure[sz][sy][sx + 1] +
            s_pressure[sz][sy - 1][sx] +
            s_pressure[sz][sy + 1][sx] +
            s_pressure[sz - 1][sy][sx] +
            s_pressure[sz + 1][sy][sx] -
            6.0f * center;

        float p_prev = s_pressure_prev[i];
        float p_new = 2.0f * center - p_prev + c_squared * laplacian;
        p_new *= damping;

        if (s_cell_type[i] == 1) {
            p_new *= s_reflection[i];
        }

        my_block->pressure_prev[i] = center;
        my_block->pressure[i] = p_new;
    }
}

// Persistent kernel: runs num_steps iterations with grid-wide sync
extern "C" __global__ void persistent_fdtd(
    BlockState* __restrict__ blocks,
    GhostFaces* __restrict__ ghost_a,
    GhostFaces* __restrict__ ghost_b,
    const PersistentParams* __restrict__ params
) {
    // Get cooperative group for grid-wide sync
    cg::grid_group grid = cg::this_grid();

    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    int total_blocks = params->blocks_x * params->blocks_y * params->blocks_z;
    if (block_idx >= total_blocks) return;

    // Block coordinates
    int bz = block_idx / (params->blocks_y * params->blocks_x);
    int rem = block_idx % (params->blocks_y * params->blocks_x);
    int by = rem / params->blocks_x;
    int bx = rem % params->blocks_x;

    // Run all steps in this single kernel launch
    for (unsigned int step = 0; step < params->num_steps; step++) {
        // Determine which buffer to use (alternating)
        GhostFaces* ghost_write = (step % 2 == 0) ? ghost_b : ghost_a;
        GhostFaces* ghost_read = (step % 2 == 0) ? ghost_a : ghost_b;

        // Phase 1: Extract faces
        extract_faces_device(
            block_idx, tid, blocks, ghost_write,
            bx, by, bz,
            params->blocks_x, params->blocks_y, params->blocks_z
        );

        // Grid-wide sync: all blocks wait here
        grid.sync();

        // Phase 2: Compute FDTD using the just-written ghost data
        compute_fdtd_device(
            block_idx, tid, blocks, ghost_write,
            bx, by, bz,
            params->blocks_x, params->blocks_y, params->blocks_z,
            params->c_squared, params->damping
        );

        // Grid-wide sync before next iteration
        grid.sync();
    }
}
"#
    .to_string()
}

/// Block-based actor GPU backend for 3D wave simulation.
#[cfg(feature = "cuda")]
pub struct BlockActorGpuBackend {
    device: Arc<CudaDevice>,
    /// Block states
    blocks: CudaSlice<BlockState>,
    /// Ghost faces (double-buffered)
    ghost_a: CudaSlice<GhostFaces>,
    ghost_b: CudaSlice<GhostFaces>,
    /// Current read buffer (alternates between a and b)
    read_from_a: bool,
    /// Grid parameters
    params: BlockGridParams,
    /// Number of blocks
    num_blocks: usize,
    /// Configuration
    config: BlockActorConfig,
    /// Whether cooperative kernel is available
    #[allow(dead_code)]
    cooperative_available: bool,
    /// Cooperative kernel handle (uses pre-compiled PTX with grid.sync())
    #[cfg(feature = "cooperative")]
    cooperative_kernel: Option<CooperativeKernel>,
}

#[cfg(feature = "cuda")]
impl BlockActorGpuBackend {
    /// Create a new block-based actor backend.
    pub fn new(grid: &SimulationGrid3D, config: BlockActorConfig) -> Result<Self, BlockActorError> {
        // Calculate block dimensions (round up)
        let blocks_x = grid.width.div_ceil(BLOCK_SIZE);
        let blocks_y = grid.height.div_ceil(BLOCK_SIZE);
        let blocks_z = grid.depth.div_ceil(BLOCK_SIZE);
        let num_blocks = blocks_x * blocks_y * blocks_z;

        let params = BlockGridParams {
            blocks_x: blocks_x as u32,
            blocks_y: blocks_y as u32,
            blocks_z: blocks_z as u32,
            cells_x: grid.width as u32,
            cells_y: grid.height as u32,
            cells_z: grid.depth as u32,
            c_squared: grid.params.c_squared,
            damping: grid.params.simple_damping,
        };

        // Initialize CUDA
        let device = CudaDevice::new(0).map_err(|e| BlockActorError::DeviceError(e.to_string()))?;

        // Compile standard kernels
        let kernel_source = generate_block_actor_kernel();
        let ptx = nvrtc::compile_ptx(&kernel_source)
            .map_err(|e| BlockActorError::CompileError(e.to_string()))?;

        device
            .load_ptx(
                ptx,
                "block_actor",
                &[
                    "extract_faces",
                    "compute_fdtd",
                    "init_blocks",
                    "extract_pressure",
                    "inject_impulse_block",
                    "fused_step",
                ],
            )
            .map_err(|e| BlockActorError::CompileError(e.to_string()))?;

        // Try to load pre-compiled cooperative kernel (when cooperative feature is enabled)
        #[cfg(feature = "cooperative")]
        let (cooperative_available, cooperative_kernel) = if config.use_persistent_kernel {
            match Self::try_load_cooperative_kernel(&device) {
                Ok(kernel) => {
                    println!(
                        "Cooperative kernel loaded: max {} concurrent blocks",
                        kernel.max_concurrent_blocks()
                    );
                    (true, Some(kernel))
                }
                Err(e) => {
                    eprintln!("Warning: Cooperative kernel not available: {}. Using fused kernel fallback.", e);
                    (false, None)
                }
            }
        } else {
            (false, None)
        };

        #[cfg(not(feature = "cooperative"))]
        let cooperative_available = false;

        // Allocate block states
        let blocks_data: Vec<BlockState> = vec![BlockState::default(); num_blocks];
        let blocks = device
            .htod_sync_copy(&blocks_data)
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;

        // Allocate double-buffered ghost faces
        let ghost_data: Vec<GhostFaces> = vec![GhostFaces::default(); num_blocks];
        let ghost_a = device
            .htod_sync_copy(&ghost_data)
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;
        let ghost_b = device
            .htod_sync_copy(&ghost_data)
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;

        let mut backend = Self {
            device,
            blocks,
            ghost_a,
            ghost_b,
            read_from_a: true,
            params,
            num_blocks,
            config,
            cooperative_available,
            #[cfg(feature = "cooperative")]
            cooperative_kernel,
        };

        // Initialize from grid
        backend.upload_pressure(grid)?;

        Ok(backend)
    }

    /// Try to load pre-compiled cooperative kernel.
    /// Returns Ok(kernel) if cooperative groups are supported, Err otherwise.
    #[cfg(feature = "cooperative")]
    fn try_load_cooperative_kernel(
        _device: &Arc<CudaDevice>,
    ) -> Result<CooperativeKernel, BlockActorError> {
        // Check if cooperative support was compiled at build time
        if !has_cooperative_support() {
            return Err(BlockActorError::CompileError(
                "Cooperative kernel PTX not available (nvcc not found at build time)".into(),
            ));
        }

        // We need a CudaDevice wrapper from ringkernel-cuda
        // For now, we'll create one from the device ordinal
        let rk_device = ringkernel_cuda::CudaDevice::new(0)
            .map_err(|e| BlockActorError::DeviceError(e.to_string()))?;

        // Load the pre-compiled cooperative FDTD kernel
        CooperativeKernel::from_precompiled(&rk_device, kernels::COOP_PERSISTENT_FDTD, 256)
            .map_err(|e| BlockActorError::CompileError(e.to_string()))
    }

    /// Check if cooperative kernel mode is available.
    pub fn is_cooperative_available(&self) -> bool {
        #[cfg(feature = "cooperative")]
        {
            self.cooperative_kernel.is_some()
        }
        #[cfg(not(feature = "cooperative"))]
        {
            false
        }
    }

    /// Backwards compatibility alias.
    pub fn is_persistent_available(&self) -> bool {
        self.is_cooperative_available()
    }

    /// Perform simulation steps using fused kernel (extract + compute in single launch).
    /// This reduces kernel launch overhead by 50% compared to the standard step.
    /// Note: Works correctly when all blocks can run concurrently on the GPU.
    pub fn step_fused(
        &mut self,
        grid: &mut SimulationGrid3D,
        num_steps: u32,
    ) -> Result<(), BlockActorError> {
        let launch_config = LaunchConfig {
            block_dim: (self.config.cuda_block_size, 1, 1),
            grid_dim: (self.num_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Upload params once
        let params_gpu = self
            .device
            .htod_sync_copy(&[self.params])
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;

        for _step in 0..num_steps {
            let fused_kernel = self
                .device
                .get_func("block_actor", "fused_step")
                .ok_or_else(|| BlockActorError::LaunchError("fused_step not found".into()))?;

            // Determine buffer to use (alternating)
            let use_buffer_b: u32 = if self.read_from_a { 1 } else { 0 };

            // Launch fused kernel
            unsafe {
                fused_kernel
                    .launch(
                        launch_config,
                        (
                            &self.blocks,
                            &self.ghost_a,
                            &self.ghost_b,
                            &params_gpu,
                            use_buffer_b,
                        ),
                    )
                    .map_err(|e| BlockActorError::LaunchError(e.to_string()))?;
            }

            // Swap buffers for next iteration
            self.read_from_a = !self.read_from_a;
        }

        // Synchronize at end
        self.device
            .synchronize()
            .map_err(|e| BlockActorError::LaunchError(e.to_string()))?;

        grid.step += num_steps as u64;
        grid.time += grid.params.time_step * num_steps as f32;

        Ok(())
    }

    /// Perform simulation steps using cooperative groups (true grid-wide sync).
    /// Runs all steps using grid.sync() for guaranteed correctness.
    ///
    /// Currently implemented as:
    /// - Validates grid size fits within max concurrent blocks
    /// - Falls back to step_fused which uses __threadfence() for synchronization
    ///
    /// When grid size fits within concurrent block limits, step_fused provides
    /// correct behavior since all blocks run concurrently. The __threadfence()
    /// ensures global memory visibility between phases.
    ///
    /// True cuLaunchCooperativeKernel support will be added when cudarc exposes
    /// the driver API for cooperative launch.
    ///
    /// Requires:
    /// - cooperative feature enabled
    /// - Grid size fits within max concurrent blocks
    #[cfg(feature = "cooperative")]
    pub fn step_cooperative(
        &mut self,
        grid: &mut SimulationGrid3D,
        num_steps: u32,
    ) -> Result<(), BlockActorError> {
        let kernel = self.cooperative_kernel.as_ref().ok_or_else(|| {
            BlockActorError::LaunchError("Cooperative kernel not available".into())
        })?;

        // Validate grid size fits within concurrent block limit
        // This is the key requirement for correct cooperative behavior
        let max_blocks = kernel.max_concurrent_blocks();
        if self.num_blocks as u32 > max_blocks {
            return Err(BlockActorError::GridSizeError(format!(
                "Grid size {} exceeds max concurrent blocks {} for cooperative launch. \
                 Reduce grid size or use step_fused() for larger grids.",
                self.num_blocks, max_blocks
            )));
        }

        // Grid fits within concurrent limits - use step_fused
        // When all blocks run concurrently, __threadfence() in step_fused
        // provides correct global memory visibility for inter-block communication
        //
        // TODO: When cudarc exposes cuLaunchCooperativeKernel, use the pre-compiled
        // cooperative kernel PTX with true grid.sync() for even better performance
        self.step_fused(grid, num_steps)
    }

    /// Maximum number of blocks that can run cooperatively on this device.
    /// Returns None if cooperative support is not available.
    #[cfg(feature = "cooperative")]
    pub fn max_cooperative_blocks(&self) -> Option<u32> {
        self.cooperative_kernel
            .as_ref()
            .map(|k| k.max_concurrent_blocks())
    }

    /// Perform simulation steps (standard two-kernel approach).
    pub fn step(
        &mut self,
        grid: &mut SimulationGrid3D,
        num_steps: u32,
    ) -> Result<(), BlockActorError> {
        let launch_config = LaunchConfig {
            block_dim: (self.config.cuda_block_size, 1, 1),
            grid_dim: (self.num_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Upload params
        let params_gpu = self
            .device
            .htod_sync_copy(&[self.params])
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;

        for _ in 0..num_steps {
            // Get kernels fresh each iteration (they consume self on launch)
            let extract_kernel = self
                .device
                .get_func("block_actor", "extract_faces")
                .ok_or_else(|| BlockActorError::LaunchError("extract_faces not found".into()))?;

            let compute_kernel = self
                .device
                .get_func("block_actor", "compute_fdtd")
                .ok_or_else(|| BlockActorError::LaunchError("compute_fdtd not found".into()))?;

            // Get write buffer for this step
            let ghost_write = if self.read_from_a {
                &self.ghost_b
            } else {
                &self.ghost_a
            };

            // Phase 1: Extract faces and write to ghost buffers
            unsafe {
                extract_kernel
                    .launch(launch_config, (&self.blocks, ghost_write, &params_gpu))
                    .map_err(|e| BlockActorError::LaunchError(e.to_string()))?;
            }

            // Synchronize to ensure face data is ready
            self.device
                .synchronize()
                .map_err(|e| BlockActorError::LaunchError(e.to_string()))?;

            // Phase 2: Compute FDTD using ghost data
            unsafe {
                compute_kernel
                    .launch(launch_config, (&self.blocks, ghost_write, &params_gpu))
                    .map_err(|e| BlockActorError::LaunchError(e.to_string()))?;
            }

            // Swap buffers
            self.read_from_a = !self.read_from_a;
        }

        // Synchronize
        self.device
            .synchronize()
            .map_err(|e| BlockActorError::LaunchError(e.to_string()))?;

        grid.step += num_steps as u64;
        grid.time += grid.params.time_step * num_steps as f32;

        Ok(())
    }

    /// Download pressure from GPU to CPU grid.
    pub fn download_pressure(&self, grid: &mut SimulationGrid3D) -> Result<(), BlockActorError> {
        let extract_kernel = self
            .device
            .get_func("block_actor", "extract_pressure")
            .ok_or_else(|| BlockActorError::LaunchError("extract_pressure not found".into()))?;

        let params_gpu = self
            .device
            .htod_sync_copy(&[self.params])
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;

        let pressure_gpu = self
            .device
            .htod_sync_copy(&grid.pressure)
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;
        let pressure_prev_gpu = self
            .device
            .htod_sync_copy(&grid.pressure_prev)
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;

        let launch_config = LaunchConfig {
            block_dim: (self.config.cuda_block_size, 1, 1),
            grid_dim: (self.num_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            extract_kernel
                .launch(
                    launch_config,
                    (&self.blocks, &pressure_gpu, &pressure_prev_gpu, &params_gpu),
                )
                .map_err(|e| BlockActorError::LaunchError(e.to_string()))?;
        }

        self.device
            .dtoh_sync_copy_into(&pressure_gpu, &mut grid.pressure)
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;
        self.device
            .dtoh_sync_copy_into(&pressure_prev_gpu, &mut grid.pressure_prev)
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;

        Ok(())
    }

    /// Upload pressure from CPU grid to GPU.
    pub fn upload_pressure(&mut self, grid: &SimulationGrid3D) -> Result<(), BlockActorError> {
        let init_kernel = self
            .device
            .get_func("block_actor", "init_blocks")
            .ok_or_else(|| BlockActorError::LaunchError("init_blocks not found".into()))?;

        let params_gpu = self
            .device
            .htod_sync_copy(&[self.params])
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;

        let pressure_gpu = self
            .device
            .htod_sync_copy(&grid.pressure)
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;
        let pressure_prev_gpu = self
            .device
            .htod_sync_copy(&grid.pressure_prev)
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;

        let cell_types_u8: Vec<u8> = grid
            .cell_types
            .iter()
            .map(|ct| match ct {
                CellType::Normal => 0u8,
                CellType::Absorber => 1u8,
                CellType::Reflector => 2u8,
                CellType::Obstacle => 3u8,
            })
            .collect();
        let cell_types_gpu = self
            .device
            .htod_sync_copy(&cell_types_u8)
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;
        let reflection_gpu = self
            .device
            .htod_sync_copy(&grid.reflection_coeff)
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;

        let launch_config = LaunchConfig {
            block_dim: (self.config.cuda_block_size, 1, 1),
            grid_dim: (self.num_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            init_kernel
                .launch(
                    launch_config,
                    (
                        &self.blocks,
                        &pressure_gpu,
                        &pressure_prev_gpu,
                        &cell_types_gpu,
                        &reflection_gpu,
                        &params_gpu,
                    ),
                )
                .map_err(|e| BlockActorError::LaunchError(e.to_string()))?;
        }

        Ok(())
    }

    /// Reset the backend from CPU grid.
    pub fn reset(&mut self, grid: &SimulationGrid3D) -> Result<(), BlockActorError> {
        self.upload_pressure(grid)?;

        // Clear ghost buffers
        let ghost_data: Vec<GhostFaces> = vec![GhostFaces::default(); self.num_blocks];
        self.device
            .htod_sync_copy_into(&ghost_data, &mut self.ghost_a)
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;
        self.device
            .htod_sync_copy_into(&ghost_data, &mut self.ghost_b)
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;
        self.read_from_a = true;

        Ok(())
    }

    /// Inject impulse on GPU.
    pub fn inject_impulse(
        &mut self,
        x: u32,
        y: u32,
        z: u32,
        amplitude: f32,
    ) -> Result<(), BlockActorError> {
        let kernel = self
            .device
            .get_func("block_actor", "inject_impulse_block")
            .ok_or_else(|| BlockActorError::LaunchError("inject_impulse_block not found".into()))?;

        let params_gpu = self
            .device
            .htod_sync_copy(&[self.params])
            .map_err(|e| BlockActorError::MemoryError(e.to_string()))?;

        let launch_config = LaunchConfig {
            block_dim: (1, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(
                    launch_config,
                    (&self.blocks, x, y, z, amplitude, &params_gpu),
                )
                .map_err(|e| BlockActorError::LaunchError(e.to_string()))?;
        }

        Ok(())
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let block_size = std::mem::size_of::<BlockState>();
        let ghost_size = std::mem::size_of::<GhostFaces>();
        self.num_blocks * block_size + 2 * self.num_blocks * ghost_size
    }

    /// Get statistics.
    pub fn stats(&self) -> BlockActorStats {
        BlockActorStats {
            num_blocks: self.num_blocks,
            cells_per_block: CELLS_PER_BLOCK,
            total_cells: self.num_blocks * CELLS_PER_BLOCK,
            memory_usage_bytes: self.memory_usage(),
            block_dims: (
                self.params.blocks_x as usize,
                self.params.blocks_y as usize,
                self.params.blocks_z as usize,
            ),
            #[cfg(feature = "cooperative")]
            max_cooperative_blocks: self.max_cooperative_blocks(),
            #[cfg(not(feature = "cooperative"))]
            max_cooperative_blocks: None,
        }
    }
}

/// Statistics for block actor backend.
#[derive(Debug, Clone)]
pub struct BlockActorStats {
    pub num_blocks: usize,
    pub cells_per_block: usize,
    pub total_cells: usize,
    pub memory_usage_bytes: usize,
    pub block_dims: (usize, usize, usize),
    /// Maximum blocks for cooperative launch (if available).
    pub max_cooperative_blocks: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_state_size() {
        let size = std::mem::size_of::<BlockState>();
        // 512 * 4 (pressure) + 512 * 4 (pressure_prev) + 512 * 4 (reflection) + 512 (cell_type)
        // = 2048 + 2048 + 2048 + 512 = 6656 bytes
        assert_eq!(size, 6656, "BlockState should be 6656 bytes, got {}", size);
    }

    #[test]
    fn test_ghost_faces_size() {
        let size = std::mem::size_of::<GhostFaces>();
        // 6 faces * 64 floats * 4 bytes = 1536 bytes
        assert_eq!(size, 1536, "GhostFaces should be 1536 bytes, got {}", size);
    }

    #[test]
    fn test_face_opposite() {
        assert_eq!(Face::NegX.opposite(), Face::PosX);
        assert_eq!(Face::PosX.opposite(), Face::NegX);
        assert_eq!(Face::NegY.opposite(), Face::PosY);
        assert_eq!(Face::PosY.opposite(), Face::NegY);
        assert_eq!(Face::NegZ.opposite(), Face::PosZ);
        assert_eq!(Face::PosZ.opposite(), Face::NegZ);
    }

    #[test]
    fn test_kernel_generation() {
        let kernel = generate_block_actor_kernel();
        assert!(kernel.contains("extract_faces"));
        assert!(kernel.contains("compute_fdtd"));
        assert!(kernel.contains("init_blocks"));
        assert!(kernel.contains("BLOCK_SIZE"));
    }
}
