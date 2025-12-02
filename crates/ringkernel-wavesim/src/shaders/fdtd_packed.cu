// CUDA Kernels for Packed Tile-Based FDTD Wave Simulation
//
// This implementation keeps ALL tile data in a single packed GPU buffer,
// enabling GPU-only halo exchange with ZERO host transfers during simulation.
//
// Memory Layout:
// All tiles are packed contiguously: [Tile(0,0)][Tile(1,0)]...[Tile(n,m)]
// Each tile is 18x18 floats (16x16 interior + 1-cell halo border)
//
// Tile indexing:
//   tile_idx = tile_y * tiles_x + tile_x
//   tile_offset = tile_idx * buffer_width * buffer_width
//
// Benefits:
// - Zero host<->GPU transfers during simulation steps
// - Only 2 kernel launches per step (halo exchange + FDTD)
// - All tiles computed in parallel
// - Halo exchange is just GPU memory copies

extern "C" {

// =============================================================================
// Halo Exchange Kernel
// =============================================================================
// Copies all halo edges between adjacent tiles in a single kernel launch.
// The copy list is precomputed on initialization and stored on GPU.
//
// Each copy operation is (src_index, dst_index) in the packed buffer.
// Total copies = tiles * ~4 edges * 16 cells = tiles * 64 (internal tiles)
// Edge tiles have fewer neighbors, so actual count varies.
//
// Grid: ceil(num_copies / 256) blocks
// Block: 256 threads

__global__ void exchange_all_halos(
    float* __restrict__ packed_buffer,  // All tiles packed contiguously
    const unsigned int* __restrict__ copies,  // [src0, dst0, src1, dst1, ...]
    int num_copies
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_copies) return;

    // Each copy is 2 uints: (src_index, dst_index)
    unsigned int src_idx = copies[idx * 2];
    unsigned int dst_idx = copies[idx * 2 + 1];

    packed_buffer[dst_idx] = packed_buffer[src_idx];
}

// =============================================================================
// Batched FDTD Kernel
// =============================================================================
// Computes FDTD wave equation for ALL tiles in a single kernel launch.
//
// Grid: (tiles_x, tiles_y, 1) - one block per tile
// Block: (16, 16, 1) - one thread per interior cell
//
// After halo exchange, each tile's halo cells contain neighbor data,
// so FDTD computation can proceed without any additional coordination.

__global__ void fdtd_all_tiles(
    const float* __restrict__ packed_curr,   // Current pressure (all tiles)
    float* __restrict__ packed_prev,         // Previous pressure (write output here)
    int tiles_x,
    int tiles_y,
    int tile_size,      // Interior size (16)
    int buffer_width,   // tile_size + 2 (18)
    float c2,           // Courant number squared
    float damping       // Damping factor
) {
    // Block indices identify the tile
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;

    // Thread indices identify the interior cell within the tile
    int lx = threadIdx.x;
    int ly = threadIdx.y;

    // Bounds check
    if (tile_x >= tiles_x || tile_y >= tiles_y) return;
    if (lx >= tile_size || ly >= tile_size) return;

    // Calculate offsets
    int tile_buffer_size = buffer_width * buffer_width;  // 18*18 = 324
    int tile_idx = tile_y * tiles_x + tile_x;
    int tile_offset = tile_idx * tile_buffer_size;

    // Buffer index for this interior cell (account for halo offset)
    int idx = tile_offset + (ly + 1) * buffer_width + (lx + 1);

    // Load current and previous pressure
    float p = packed_curr[idx];
    float p_prev = packed_prev[idx];

    // Load neighbors (halos were filled by exchange_all_halos)
    float p_n = packed_curr[idx - buffer_width];  // North
    float p_s = packed_curr[idx + buffer_width];  // South
    float p_w = packed_curr[idx - 1];             // West
    float p_e = packed_curr[idx + 1];             // East

    // FDTD wave equation
    float laplacian = p_n + p_s + p_e + p_w - 4.0f * p;
    float p_new = 2.0f * p - p_prev + c2 * laplacian;

    // Apply damping and write to output
    packed_prev[idx] = p_new * damping;
}

// =============================================================================
// Upload Initial State Kernel
// =============================================================================
// Copies initial pressure data from a staging buffer to the packed tile buffer.
// Used once at initialization to set up the simulation.
//
// Grid: (tiles_x, tiles_y, 1)
// Block: (18, 18, 1) - covers full tile buffer including halos

__global__ void upload_tile_data(
    float* __restrict__ packed_buffer,
    const float* __restrict__ staging,  // 18x18 data for one tile
    int tile_x,
    int tile_y,
    int tiles_x,
    int buffer_width
) {
    int lx = threadIdx.x;
    int ly = threadIdx.y;

    if (lx >= buffer_width || ly >= buffer_width) return;

    int tile_buffer_size = buffer_width * buffer_width;
    int tile_idx = tile_y * tiles_x + tile_x;
    int tile_offset = tile_idx * tile_buffer_size;

    int local_idx = ly * buffer_width + lx;
    int global_idx = tile_offset + local_idx;

    packed_buffer[global_idx] = staging[local_idx];
}

// =============================================================================
// Read All Interiors Kernel
// =============================================================================
// Extracts all tile interiors to a flat output buffer for visualization.
// Called only when the host needs to display the simulation state.
//
// Grid: ceil(grid_width/16) x ceil(grid_height/16)
// Block: (16, 16, 1)

__global__ void read_all_interiors(
    const float* __restrict__ packed_buffer,
    float* __restrict__ output,  // grid_width x grid_height flat array
    int tiles_x,
    int tiles_y,
    int tile_size,
    int buffer_width,
    int grid_width,
    int grid_height
) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= grid_width || gy >= grid_height) return;

    // Find which tile this global coordinate belongs to
    int tile_x = gx / tile_size;
    int tile_y = gy / tile_size;

    // Local position within the tile
    int lx = gx % tile_size;
    int ly = gy % tile_size;

    // Source index in packed buffer
    int tile_buffer_size = buffer_width * buffer_width;
    int tile_idx = tile_y * tiles_x + tile_x;
    int tile_offset = tile_idx * tile_buffer_size;
    int src_idx = tile_offset + (ly + 1) * buffer_width + (lx + 1);

    // Destination index in output
    int dst_idx = gy * grid_width + gx;

    output[dst_idx] = packed_buffer[src_idx];
}

// =============================================================================
// Inject Impulse Kernel
// =============================================================================
// Adds an impulse to a specific cell. Called when user clicks to add energy.
//
// Grid: (1, 1, 1)
// Block: (1, 1, 1)

__global__ void inject_impulse(
    float* __restrict__ packed_buffer,
    int tile_x,
    int tile_y,
    int local_x,
    int local_y,
    int tiles_x,
    int buffer_width,
    float amplitude
) {
    int tile_buffer_size = buffer_width * buffer_width;
    int tile_idx = tile_y * tiles_x + tile_x;
    int tile_offset = tile_idx * tile_buffer_size;
    int idx = tile_offset + (local_y + 1) * buffer_width + (local_x + 1);

    packed_buffer[idx] += amplitude;
}

// =============================================================================
// Apply Boundary Conditions Kernel
// =============================================================================
// For tiles at the grid edge, applies boundary conditions to halos.
// Called after halo exchange to handle domain boundaries.
//
// This kernel reflects interior values to boundary halos (rigid boundary).
// Can be modified for absorbing boundaries by multiplying by a coefficient.
//
// Grid: 4 blocks (one per edge: N, S, W, E)
// Block: max(tiles_x, tiles_y) * tile_size threads

__global__ void apply_boundary_conditions(
    float* __restrict__ packed_buffer,
    int tiles_x,
    int tiles_y,
    int tile_size,
    int buffer_width,
    float reflection_coeff  // 1.0 for rigid, < 1.0 for absorbing
) {
    int edge = blockIdx.x;  // 0=North, 1=South, 2=West, 3=East
    int idx = threadIdx.x;

    int tile_buffer_size = buffer_width * buffer_width;

    if (edge == 0) {
        // North boundary: tiles with tile_y == 0
        // Reflect row 1 to row 0 for each tile
        int tile_x = idx / tile_size;
        int cell_x = idx % tile_size;
        if (tile_x >= tiles_x) return;

        int tile_idx = 0 * tiles_x + tile_x;  // tile_y = 0
        int tile_offset = tile_idx * tile_buffer_size;
        int src_idx = tile_offset + 1 * buffer_width + (cell_x + 1);
        int dst_idx = tile_offset + 0 * buffer_width + (cell_x + 1);
        packed_buffer[dst_idx] = packed_buffer[src_idx] * reflection_coeff;
    }
    else if (edge == 1) {
        // South boundary: tiles with tile_y == tiles_y - 1
        int tile_x = idx / tile_size;
        int cell_x = idx % tile_size;
        if (tile_x >= tiles_x) return;

        int tile_idx = (tiles_y - 1) * tiles_x + tile_x;
        int tile_offset = tile_idx * tile_buffer_size;
        int src_idx = tile_offset + tile_size * buffer_width + (cell_x + 1);
        int dst_idx = tile_offset + (tile_size + 1) * buffer_width + (cell_x + 1);
        packed_buffer[dst_idx] = packed_buffer[src_idx] * reflection_coeff;
    }
    else if (edge == 2) {
        // West boundary: tiles with tile_x == 0
        int tile_y = idx / tile_size;
        int cell_y = idx % tile_size;
        if (tile_y >= tiles_y) return;

        int tile_idx = tile_y * tiles_x + 0;  // tile_x = 0
        int tile_offset = tile_idx * tile_buffer_size;
        int src_idx = tile_offset + (cell_y + 1) * buffer_width + 1;
        int dst_idx = tile_offset + (cell_y + 1) * buffer_width + 0;
        packed_buffer[dst_idx] = packed_buffer[src_idx] * reflection_coeff;
    }
    else if (edge == 3) {
        // East boundary: tiles with tile_x == tiles_x - 1
        int tile_y = idx / tile_size;
        int cell_y = idx % tile_size;
        if (tile_y >= tiles_y) return;

        int tile_idx = tile_y * tiles_x + (tiles_x - 1);
        int tile_offset = tile_idx * tile_buffer_size;
        int src_idx = tile_offset + (cell_y + 1) * buffer_width + tile_size;
        int dst_idx = tile_offset + (cell_y + 1) * buffer_width + (tile_size + 1);
        packed_buffer[dst_idx] = packed_buffer[src_idx] * reflection_coeff;
    }
}

}  // extern "C"
