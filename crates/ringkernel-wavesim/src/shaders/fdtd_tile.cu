// CUDA Kernels for Tile-Based FDTD Wave Simulation
//
// These kernels operate on 18x18 tile buffers (16x16 interior + 1-cell halo border).
// The design enables GPU-resident state with minimal host transfers.
//
// Buffer Layout (18x18 = 324 floats = 1,296 bytes):
//
//   +---+----------------+---+
//   | NW|   North Halo   |NE |  <- Row 0 (col 1-16 from neighbor)
//   +---+----------------+---+
//   |   |                |   |
//   | W |   16x16 Tile   | E |  <- Rows 1-16, Cols 1-16 (owned interior)
//   |   |    Interior    |   |
//   +---+----------------+---+
//   | SW|   South Halo   |SE |  <- Row 17 (col 1-16 from neighbor)
//   +---+----------------+---+
//
// Index calculation: idx = y * 18 + x
//   - Interior cell (lx, ly): idx = (ly + 1) * 18 + (lx + 1)
//   - North halo for col x: idx = 0 * 18 + (x + 1)
//   - South halo for col x: idx = 17 * 18 + (x + 1)
//   - West halo for row y: idx = (y + 1) * 18 + 0
//   - East halo for row y: idx = (y + 1) * 18 + 17

extern "C" {

// FDTD kernel for 18x18 tile buffer
//
// Computes the FDTD wave equation for all 16x16 interior cells:
//   p_new = 2*p - p_prev + c2 * (p_north + p_south + p_east + p_west - 4*p)
//   p_damped = p_new * damping
//
// Reads from pressure, writes to pressure_prev (ping-pong buffering).
// After this kernel, caller swaps buffer pointers.
//
// Grid: 1x1 blocks
// Block: 16x16 threads (one per interior cell)
__global__ void fdtd_tile_step(
    const float* __restrict__ pressure,        // 18x18 buffer (324 floats), read
    float* __restrict__ pressure_prev,         // 18x18 buffer (324 floats), write
    float c2,                                  // Courant number squared
    float damping                              // Damping factor (1 - damping_coeff)
) {
    // Thread indices map to interior cell positions
    int lx = threadIdx.x;  // 0..15 -> local x
    int ly = threadIdx.y;  // 0..15 -> local y

    // Bounds check (should be 16x16 threads)
    if (lx >= 16 || ly >= 16) return;

    // Buffer index for this interior cell
    // Interior (lx, ly) -> buffer (lx+1, ly+1) -> idx = (ly+1)*18 + (lx+1)
    int idx = (ly + 1) * 18 + (lx + 1);

    // Load current and previous pressure
    float p = pressure[idx];
    float p_prev = pressure_prev[idx];

    // Load neighbors (halos provide boundary values)
    float p_n = pressure[idx - 18];  // North (row above)
    float p_s = pressure[idx + 18];  // South (row below)
    float p_w = pressure[idx - 1];   // West (col left)
    float p_e = pressure[idx + 1];   // East (col right)

    // FDTD wave equation: discrete Laplacian + time stepping
    float laplacian = p_n + p_s + p_e + p_w - 4.0f * p;
    float p_new = 2.0f * p - p_prev + c2 * laplacian;

    // Apply damping
    pressure_prev[idx] = p_new * damping;
}

// Extract halo data from interior edge to linear buffer
//
// Extracts 16 values from the interior edge that will be sent to a neighbor tile.
// The neighbor will inject these values into their halo region.
//
// edge values:
//   0 = North: Extract row 1 (first interior row) -> send to south neighbor
//   1 = South: Extract row 16 (last interior row) -> send to north neighbor
//   2 = West: Extract col 1 (first interior col) -> send to east neighbor
//   3 = East: Extract col 16 (last interior col) -> send to west neighbor
//
// Grid: 1x1 blocks
// Block: 16 threads (one per halo cell)
__global__ void extract_halo(
    const float* __restrict__ pressure,  // 18x18 buffer
    float* __restrict__ halo_out,        // 16-element output
    int edge                             // 0=North, 1=South, 2=West, 3=East
) {
    int i = threadIdx.x;
    if (i >= 16) return;

    int idx;
    switch (edge) {
        case 0:  // North - extract row 1 (first interior row)
            idx = 1 * 18 + (i + 1);
            break;
        case 1:  // South - extract row 16 (last interior row)
            idx = 16 * 18 + (i + 1);
            break;
        case 2:  // West - extract col 1 (first interior col)
            idx = (i + 1) * 18 + 1;
            break;
        case 3:  // East - extract col 16 (last interior col)
        default:
            idx = (i + 1) * 18 + 16;
            break;
    }

    halo_out[i] = pressure[idx];
}

// Inject halo data from linear buffer to halo region
//
// Takes 16 values received from a neighbor tile and writes them to the
// appropriate halo region in this tile's buffer.
//
// edge values:
//   0 = North: Inject to row 0 (north halo) <- received from north neighbor's south edge
//   1 = South: Inject to row 17 (south halo) <- received from south neighbor's north edge
//   2 = West: Inject to col 0 (west halo) <- received from west neighbor's east edge
//   3 = East: Inject to col 17 (east halo) <- received from east neighbor's west edge
//
// Grid: 1x1 blocks
// Block: 16 threads (one per halo cell)
__global__ void inject_halo(
    float* __restrict__ pressure,        // 18x18 buffer
    const float* __restrict__ halo_in,   // 16-element input
    int edge                             // 0=North, 1=South, 2=West, 3=East
) {
    int i = threadIdx.x;
    if (i >= 16) return;

    int idx;
    switch (edge) {
        case 0:  // North - inject to row 0 (halo row)
            idx = 0 * 18 + (i + 1);
            break;
        case 1:  // South - inject to row 17 (halo row)
            idx = 17 * 18 + (i + 1);
            break;
        case 2:  // West - inject to col 0 (halo col)
            idx = (i + 1) * 18 + 0;
            break;
        case 3:  // East - inject to col 17 (halo col)
        default:
            idx = (i + 1) * 18 + 17;
            break;
    }

    pressure[idx] = halo_in[i];
}

// Read interior pressure to linear buffer for visualization
//
// Extracts the 16x16 interior cells to a linear buffer for host readback.
// Called once per frame when GUI needs to render.
//
// Grid: 1x1 blocks
// Block: 16x16 threads (one per interior cell)
__global__ void read_interior(
    const float* __restrict__ pressure,  // 18x18 buffer
    float* __restrict__ output           // 256-element output (16*16)
) {
    int lx = threadIdx.x;
    int ly = threadIdx.y;

    if (lx >= 16 || ly >= 16) return;

    // Source: interior cell in 18x18 buffer
    int src_idx = (ly + 1) * 18 + (lx + 1);

    // Destination: linear buffer in row-major order
    int dst_idx = ly * 16 + lx;

    output[dst_idx] = pressure[src_idx];
}

// Apply boundary reflection for tiles at grid edges
//
// For tiles without a neighbor in a given direction, this kernel
// reflects the interior edge to the halo (simulates rigid boundary).
//
// edge values:
//   0 = North: Reflect row 1 to row 0
//   1 = South: Reflect row 16 to row 17
//   2 = West: Reflect col 1 to col 0
//   3 = East: Reflect col 16 to col 17
//
// Grid: 1x1 blocks
// Block: 16 threads
__global__ void apply_boundary_reflection(
    float* __restrict__ pressure,  // 18x18 buffer
    int edge,                      // 0=North, 1=South, 2=West, 3=East
    float reflection_coeff         // Typically 0.95 for slight absorption
) {
    int i = threadIdx.x;
    if (i >= 16) return;

    int src_idx, dst_idx;
    switch (edge) {
        case 0:  // North - reflect row 1 to row 0
            src_idx = 1 * 18 + (i + 1);
            dst_idx = 0 * 18 + (i + 1);
            break;
        case 1:  // South - reflect row 16 to row 17
            src_idx = 16 * 18 + (i + 1);
            dst_idx = 17 * 18 + (i + 1);
            break;
        case 2:  // West - reflect col 1 to col 0
            src_idx = (i + 1) * 18 + 1;
            dst_idx = (i + 1) * 18 + 0;
            break;
        case 3:  // East - reflect col 16 to col 17
        default:
            src_idx = (i + 1) * 18 + 16;
            dst_idx = (i + 1) * 18 + 17;
            break;
    }

    pressure[dst_idx] = pressure[src_idx] * reflection_coeff;
}

}  // extern "C"
