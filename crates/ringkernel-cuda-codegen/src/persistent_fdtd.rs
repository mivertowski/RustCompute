//! Persistent FDTD kernel code generation.
//!
//! Generates CUDA code for truly persistent GPU actors that run for the
//! entire simulation lifetime with:
//!
//! - Single kernel launch
//! - H2K/K2H command/response messaging
//! - K2K halo exchange between blocks
//! - Grid-wide synchronization via cooperative groups
//!
//! # Architecture
//!
//! The generated kernel is structured as:
//!
//! 1. **Coordinator (Block 0)**: Processes H2K commands, sends K2H responses
//! 2. **Worker Blocks (1..N)**: Compute FDTD steps, exchange halos via K2K
//! 3. **Persistent Loop**: Runs until Terminate command received
//!
//! # Memory Layout
//!
//! ```text
//! Kernel Parameters:
//! - control_ptr:      PersistentControlBlock* (mapped memory)
//! - pressure_a_ptr:   float* (device memory, buffer A)
//! - pressure_b_ptr:   float* (device memory, buffer B)
//! - h2k_header_ptr:   SpscQueueHeader* (mapped memory)
//! - h2k_slots_ptr:    H2KMessage* (mapped memory)
//! - k2h_header_ptr:   SpscQueueHeader* (mapped memory)
//! - k2h_slots_ptr:    K2HMessage* (mapped memory)
//! - routes_ptr:       K2KRouteEntry* (device memory)
//! - halo_ptr:         float* (device memory, halo buffers)
//! ```

/// Configuration for persistent FDTD kernel generation.
#[derive(Debug, Clone)]
pub struct PersistentFdtdConfig {
    /// Kernel function name.
    pub name: String,
    /// Tile size per dimension (cells per block).
    pub tile_size: (usize, usize, usize),
    /// Whether to use cooperative groups for grid sync.
    pub use_cooperative: bool,
    /// Progress report interval (steps).
    pub progress_interval: u64,
    /// Enable energy calculation.
    pub track_energy: bool,
}

impl Default for PersistentFdtdConfig {
    fn default() -> Self {
        Self {
            name: "persistent_fdtd3d".to_string(),
            tile_size: (8, 8, 8),
            use_cooperative: true,
            progress_interval: 100,
            track_energy: true,
        }
    }
}

impl PersistentFdtdConfig {
    /// Create a new config with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Set tile size.
    pub fn with_tile_size(mut self, tx: usize, ty: usize, tz: usize) -> Self {
        self.tile_size = (tx, ty, tz);
        self
    }

    /// Set cooperative groups usage.
    pub fn with_cooperative(mut self, use_coop: bool) -> Self {
        self.use_cooperative = use_coop;
        self
    }

    /// Set progress reporting interval.
    pub fn with_progress_interval(mut self, interval: u64) -> Self {
        self.progress_interval = interval;
        self
    }

    /// Calculate threads per block.
    pub fn threads_per_block(&self) -> usize {
        self.tile_size.0 * self.tile_size.1 * self.tile_size.2
    }

    /// Calculate shared memory size per block (tile + halo).
    pub fn shared_mem_size(&self) -> usize {
        let (tx, ty, tz) = self.tile_size;
        // Tile + 1 cell halo on each side
        let with_halo = (tx + 2) * (ty + 2) * (tz + 2);
        with_halo * std::mem::size_of::<f32>()
    }
}

/// Generate a complete persistent FDTD kernel.
///
/// The generated kernel includes:
/// - H2K command processing
/// - K2H response generation
/// - K2K halo exchange
/// - FDTD computation with shared memory
/// - Grid-wide synchronization
pub fn generate_persistent_fdtd_kernel(config: &PersistentFdtdConfig) -> String {
    let mut code = String::new();

    // Header with includes
    code.push_str(&generate_header(config));

    // Structure definitions
    code.push_str(&generate_structures());

    // Device helper functions
    code.push_str(&generate_device_functions(config));

    // Main kernel
    code.push_str(&generate_main_kernel(config));

    code
}

fn generate_header(config: &PersistentFdtdConfig) -> String {
    let mut code = String::new();

    code.push_str("// Generated Persistent FDTD Kernel\n");
    code.push_str("// RingKernel GPU Actor System\n\n");

    if config.use_cooperative {
        code.push_str("#include <cooperative_groups.h>\n");
        code.push_str("namespace cg = cooperative_groups;\n\n");
    }

    code.push_str("#include <cuda_runtime.h>\n");
    code.push_str("#include <stdint.h>\n\n");

    code
}

fn generate_structures() -> String {
    r#"
// ============================================================================
// STRUCTURE DEFINITIONS (must match Rust persistent.rs)
// ============================================================================

// Control block for persistent kernel (256 bytes, cache-aligned)
typedef struct __align__(256) {
    // Lifecycle control (host -> GPU)
    uint32_t should_terminate;
    uint32_t _pad0;
    uint64_t steps_remaining;

    // State (GPU -> host)
    uint64_t current_step;
    uint32_t current_buffer;
    uint32_t has_terminated;
    float total_energy;
    uint32_t _pad1;

    // Configuration
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint32_t sim_size[3];
    uint32_t tile_size[3];

    // Acoustic parameters
    float c2_dt2;
    float damping;
    float cell_size;
    float dt;

    // Synchronization
    uint32_t barrier_counter;
    uint32_t barrier_generation;

    // Statistics
    uint64_t messages_processed;
    uint64_t k2k_messages_sent;
    uint64_t k2k_messages_received;

    uint64_t _reserved[16];
} PersistentControlBlock;

// SPSC queue header (128 bytes, cache-aligned)
typedef struct __align__(128) {
    uint64_t head;
    uint64_t tail;
    uint32_t capacity;
    uint32_t mask;
    uint64_t _padding[12];
} SpscQueueHeader;

// H2K message (64 bytes)
typedef struct __align__(64) {
    uint32_t cmd;
    uint32_t flags;
    uint64_t cmd_id;
    uint64_t param1;
    uint32_t param2;
    uint32_t param3;
    float param4;
    float param5;
    uint64_t _reserved[4];
} H2KMessage;

// K2H message (64 bytes)
typedef struct __align__(64) {
    uint32_t resp_type;
    uint32_t flags;
    uint64_t cmd_id;
    uint64_t step;
    uint64_t steps_remaining;
    float energy;
    uint32_t error_code;
    uint64_t _reserved[3];
} K2HMessage;

// Command types
#define CMD_NOP 0
#define CMD_RUN_STEPS 1
#define CMD_PAUSE 2
#define CMD_RESUME 3
#define CMD_TERMINATE 4
#define CMD_INJECT_IMPULSE 5
#define CMD_SET_SOURCE 6
#define CMD_GET_PROGRESS 7

// Response types
#define RESP_ACK 0
#define RESP_PROGRESS 1
#define RESP_ERROR 2
#define RESP_TERMINATED 3
#define RESP_ENERGY 4

// Neighbor block IDs
typedef struct {
    int32_t pos_x, neg_x;
    int32_t pos_y, neg_y;
    int32_t pos_z, neg_z;
    int32_t _padding[2];
} BlockNeighbors;

// K2K route entry
typedef struct {
    BlockNeighbors neighbors;
    uint32_t block_pos[3];
    uint32_t _padding;
    uint32_t cell_offset[3];
    uint32_t _padding2;
} K2KRouteEntry;

// Face indices
#define FACE_POS_X 0
#define FACE_NEG_X 1
#define FACE_POS_Y 2
#define FACE_NEG_Y 3
#define FACE_POS_Z 4
#define FACE_NEG_Z 5

"#
    .to_string()
}

fn generate_device_functions(config: &PersistentFdtdConfig) -> String {
    let (tx, ty, tz) = config.tile_size;
    let face_size = tx * ty;

    let mut code = String::new();

    // Software grid barrier (fallback when cooperative not available)
    code.push_str(
        r#"
// ============================================================================
// SYNCHRONIZATION FUNCTIONS
// ============================================================================

// Software grid barrier (atomic counter + generation)
__device__ void software_grid_sync(
    volatile uint32_t* barrier_counter,
    volatile uint32_t* barrier_gen,
    int num_blocks
) {
    __syncthreads();  // First sync within block

    if (threadIdx.x == 0) {
        unsigned int gen = *barrier_gen;

        unsigned int arrived = atomicAdd((unsigned int*)barrier_counter, 1) + 1;
        if (arrived == num_blocks) {
            *barrier_counter = 0;
            __threadfence();
            atomicAdd((unsigned int*)barrier_gen, 1);
        } else {
            while (atomicAdd((unsigned int*)barrier_gen, 0) == gen) {
                __threadfence();
            }
        }
    }

    __syncthreads();
}

"#,
    );

    // H2K/K2H queue operations
    code.push_str(
        r#"
// ============================================================================
// MESSAGE QUEUE OPERATIONS
// ============================================================================

// Copy H2K message from volatile source (field-by-field to handle volatile)
__device__ void copy_h2k_message(H2KMessage* dst, volatile H2KMessage* src) {
    dst->cmd = src->cmd;
    dst->flags = src->flags;
    dst->cmd_id = src->cmd_id;
    dst->param1 = src->param1;
    dst->param2 = src->param2;
    dst->param3 = src->param3;
    dst->param4 = src->param4;
    dst->param5 = src->param5;
    // Skip reserved fields for performance
}

// Copy K2H message to volatile destination (field-by-field to handle volatile)
__device__ void copy_k2h_message(volatile K2HMessage* dst, const K2HMessage* src) {
    dst->resp_type = src->resp_type;
    dst->flags = src->flags;
    dst->cmd_id = src->cmd_id;
    dst->step = src->step;
    dst->steps_remaining = src->steps_remaining;
    dst->energy = src->energy;
    dst->error_code = src->error_code;
    // Skip reserved fields for performance
}

// Try to receive H2K message (returns true if message available)
__device__ bool h2k_try_recv(
    volatile SpscQueueHeader* header,
    volatile H2KMessage* slots,
    H2KMessage* out_msg
) {
    // Fence BEFORE reading to ensure we see host writes
    __threadfence_system();

    uint64_t head = header->head;
    uint64_t tail = header->tail;

    if (head == tail) {
        return false;  // Empty
    }

    uint32_t slot = tail & header->mask;
    copy_h2k_message(out_msg, &slots[slot]);

    __threadfence();
    header->tail = tail + 1;

    return true;
}

// Send K2H message
__device__ bool k2h_send(
    volatile SpscQueueHeader* header,
    volatile K2HMessage* slots,
    const K2HMessage* msg
) {
    uint64_t head = header->head;
    uint64_t tail = header->tail;
    uint32_t capacity = header->capacity;

    if (head - tail >= capacity) {
        return false;  // Full
    }

    uint32_t slot = head & header->mask;
    copy_k2h_message(&slots[slot], msg);

    __threadfence_system();  // Ensure host sees our writes
    header->head = head + 1;

    return true;
}

"#,
    );

    // Energy reduction function
    code.push_str(
        r#"
// ============================================================================
// ENERGY CALCULATION (Parallel Reduction)
// ============================================================================

// Block-level parallel reduction for energy: E = sum(p^2)
// Uses shared memory for efficient reduction within a block
__device__ float block_reduce_energy(
    float my_energy,
    float* shared_reduce,
    int threads_per_block
) {
    int tid = threadIdx.x;

    // Store initial value in shared memory
    shared_reduce[tid] = my_energy;
    __syncthreads();

    // Parallel reduction tree
    for (int stride = threads_per_block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_reduce[tid] += shared_reduce[tid + stride];
        }
        __syncthreads();
    }

    // Return the total for this block (only thread 0 has final sum)
    return shared_reduce[0];
}

"#,
    );

    // Halo exchange functions
    code.push_str(&format!(
        r#"
// ============================================================================
// K2K HALO EXCHANGE
// ============================================================================

#define TILE_X {tx}
#define TILE_Y {ty}
#define TILE_Z {tz}
#define FACE_SIZE {face_size}

// Pack halo faces from shared memory to device halo buffer
__device__ void pack_halo_faces(
    float tile[TILE_Z + 2][TILE_Y + 2][TILE_X + 2],
    float* halo_buffers,
    int block_id,
    int pingpong,
    int num_blocks
) {{
    int tid = threadIdx.x;
    int face_stride = FACE_SIZE;
    int block_stride = 6 * face_stride * 2;  // 6 faces, 2 ping-pong
    int pp_offset = pingpong * 6 * face_stride;

    float* block_halo = halo_buffers + block_id * block_stride + pp_offset;

    if (tid < FACE_SIZE) {{
        int fx = tid % TILE_X;
        int fy = tid / TILE_X;

        // +X face (x = TILE_X in local coords, which is index TILE_X)
        block_halo[FACE_POS_X * face_stride + tid] = tile[fy + 1][fx + 1][TILE_X];
        // -X face (x = 1 in local coords)
        block_halo[FACE_NEG_X * face_stride + tid] = tile[fy + 1][fx + 1][1];
        // +Y face
        block_halo[FACE_POS_Y * face_stride + tid] = tile[fy + 1][TILE_Y][fx + 1];
        // -Y face
        block_halo[FACE_NEG_Y * face_stride + tid] = tile[fy + 1][1][fx + 1];
        // +Z face
        block_halo[FACE_POS_Z * face_stride + tid] = tile[TILE_Z][fy + 1][fx + 1];
        // -Z face
        block_halo[FACE_NEG_Z * face_stride + tid] = tile[1][fy + 1][fx + 1];
    }}
}}

// Unpack halo faces from device buffer to shared memory ghost cells
__device__ void unpack_halo_faces(
    float tile[TILE_Z + 2][TILE_Y + 2][TILE_X + 2],
    const float* halo_buffers,
    const K2KRouteEntry* route,
    int pingpong,
    int num_blocks
) {{
    int tid = threadIdx.x;
    int face_stride = FACE_SIZE;
    int block_stride = 6 * face_stride * 2;
    int pp_offset = pingpong * 6 * face_stride;

    if (tid < FACE_SIZE) {{
        int fx = tid % TILE_X;
        int fy = tid / TILE_X;

        // My +X ghost comes from neighbor's -X face
        if (route->neighbors.pos_x >= 0) {{
            const float* n_halo = halo_buffers + route->neighbors.pos_x * block_stride + pp_offset;
            tile[fy + 1][fx + 1][TILE_X + 1] = n_halo[FACE_NEG_X * face_stride + tid];
        }}

        // My -X ghost comes from neighbor's +X face
        if (route->neighbors.neg_x >= 0) {{
            const float* n_halo = halo_buffers + route->neighbors.neg_x * block_stride + pp_offset;
            tile[fy + 1][fx + 1][0] = n_halo[FACE_POS_X * face_stride + tid];
        }}

        // My +Y ghost
        if (route->neighbors.pos_y >= 0) {{
            const float* n_halo = halo_buffers + route->neighbors.pos_y * block_stride + pp_offset;
            tile[fy + 1][TILE_Y + 1][fx + 1] = n_halo[FACE_NEG_Y * face_stride + tid];
        }}

        // My -Y ghost
        if (route->neighbors.neg_y >= 0) {{
            const float* n_halo = halo_buffers + route->neighbors.neg_y * block_stride + pp_offset;
            tile[fy + 1][0][fx + 1] = n_halo[FACE_POS_Y * face_stride + tid];
        }}

        // My +Z ghost
        if (route->neighbors.pos_z >= 0) {{
            const float* n_halo = halo_buffers + route->neighbors.pos_z * block_stride + pp_offset;
            tile[TILE_Z + 1][fy + 1][fx + 1] = n_halo[FACE_NEG_Z * face_stride + tid];
        }}

        // My -Z ghost
        if (route->neighbors.neg_z >= 0) {{
            const float* n_halo = halo_buffers + route->neighbors.neg_z * block_stride + pp_offset;
            tile[0][fy + 1][fx + 1] = n_halo[FACE_POS_Z * face_stride + tid];
        }}
    }}
}}

"#,
        tx = tx,
        ty = ty,
        tz = tz,
        face_size = face_size
    ));

    code
}

fn generate_main_kernel(config: &PersistentFdtdConfig) -> String {
    let (tx, ty, tz) = config.tile_size;
    let threads_per_block = tx * ty * tz;
    let progress_interval = config.progress_interval;

    // For cooperative groups: declare grid variable once at init, then just call sync
    // For software sync: no variable needed, just call the function each time
    let grid_sync = if config.use_cooperative {
        "grid.sync();"
    } else {
        "software_grid_sync(&ctrl->barrier_counter, &ctrl->barrier_generation, num_blocks);"
    };

    format!(
        r#"
// ============================================================================
// MAIN PERSISTENT KERNEL
// ============================================================================

extern "C" __global__ void __launch_bounds__({threads_per_block}, 2)
{name}(
    PersistentControlBlock* __restrict__ ctrl,
    float* __restrict__ pressure_a,
    float* __restrict__ pressure_b,
    SpscQueueHeader* __restrict__ h2k_header,
    H2KMessage* __restrict__ h2k_slots,
    SpscQueueHeader* __restrict__ k2h_header,
    K2HMessage* __restrict__ k2h_slots,
    const K2KRouteEntry* __restrict__ routes,
    float* __restrict__ halo_buffers
) {{
    // Block and thread indices
    int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int num_blocks = gridDim.x * gridDim.y * gridDim.z;
    int tid = threadIdx.x;
    bool is_coordinator = (block_id == 0 && tid == 0);

    // Shared memory for tile + ghost cells
    __shared__ float tile[{tz} + 2][{ty} + 2][{tx} + 2];

    // Shared memory for energy reduction
    __shared__ float energy_reduce[{threads_per_block}];

    // Get my routing info
    const K2KRouteEntry* my_route = &routes[block_id];

    // Pointers to pressure buffers (will swap)
    float* p_curr = pressure_a;
    float* p_prev = pressure_b;

    // Grid dimensions for indexing
    int sim_x = ctrl->sim_size[0];
    int sim_y = ctrl->sim_size[1];
    int sim_z = ctrl->sim_size[2];

    // Local cell coordinates within tile
    int lx = tid % {tx};
    int ly = (tid / {tx}) % {ty};
    int lz = tid / ({tx} * {ty});

    // Global cell coordinates
    int gx = my_route->cell_offset[0] + lx;
    int gy = my_route->cell_offset[1] + ly;
    int gz = my_route->cell_offset[2] + lz;
    int global_idx = gx + gy * sim_x + gz * sim_x * sim_y;

    // Physics parameters
    float c2_dt2 = ctrl->c2_dt2;
    float damping = ctrl->damping;

    {grid_sync_init}

    // ========== PERSISTENT LOOP ==========
    while (true) {{
        // --- Phase 1: Command Processing (coordinator only) ---
        if (is_coordinator) {{
            H2KMessage cmd;
            while (h2k_try_recv(h2k_header, h2k_slots, &cmd)) {{
                ctrl->messages_processed++;

                switch (cmd.cmd) {{
                    case CMD_RUN_STEPS:
                        ctrl->steps_remaining += cmd.param1;
                        break;

                    case CMD_TERMINATE:
                        ctrl->should_terminate = 1;
                        break;

                    case CMD_INJECT_IMPULSE: {{
                        // Extract position from params
                        uint32_t ix = (cmd.param1 >> 32) & 0xFFFFFFFF;
                        uint32_t iy = cmd.param1 & 0xFFFFFFFF;
                        uint32_t iz = cmd.param2;
                        float amplitude = cmd.param4;

                        if (ix < sim_x && iy < sim_y && iz < sim_z) {{
                            int idx = ix + iy * sim_x + iz * sim_x * sim_y;
                            p_curr[idx] += amplitude;
                        }}
                        break;
                    }}

                    case CMD_GET_PROGRESS: {{
                        K2HMessage resp;
                        resp.resp_type = RESP_PROGRESS;
                        resp.cmd_id = cmd.cmd_id;
                        resp.step = ctrl->current_step;
                        resp.steps_remaining = ctrl->steps_remaining;
                        resp.energy = ctrl->total_energy;
                        k2h_send(k2h_header, k2h_slots, &resp);
                        break;
                    }}

                    default:
                        break;
                }}
            }}
        }}

        // Grid-wide sync so all blocks see updated control state
        {grid_sync}

        // Check for termination
        if (ctrl->should_terminate) {{
            break;
        }}

        // --- Phase 2: Check if we have work ---
        if (ctrl->steps_remaining == 0) {{
            // No work - brief spinwait then check again
            // Use volatile counter to prevent optimization
            volatile int spin_count = 0;
            for (int i = 0; i < 1000; i++) {{
                spin_count++;
            }}
            {grid_sync}
            continue;
        }}

        // --- Phase 3: Load tile from global memory ---
        // Load interior cells
        if (gx < sim_x && gy < sim_y && gz < sim_z) {{
            tile[lz + 1][ly + 1][lx + 1] = p_curr[global_idx];
        }} else {{
            tile[lz + 1][ly + 1][lx + 1] = 0.0f;
        }}

        // Initialize ghost cells to boundary (zero pressure)
        if (lx == 0) tile[lz + 1][ly + 1][0] = 0.0f;
        if (lx == {tx} - 1) tile[lz + 1][ly + 1][{tx} + 1] = 0.0f;
        if (ly == 0) tile[lz + 1][0][lx + 1] = 0.0f;
        if (ly == {ty} - 1) tile[lz + 1][{ty} + 1][lx + 1] = 0.0f;
        if (lz == 0) tile[0][ly + 1][lx + 1] = 0.0f;
        if (lz == {tz} - 1) tile[{tz} + 1][ly + 1][lx + 1] = 0.0f;

        __syncthreads();

        // --- Phase 4: K2K Halo Exchange ---
        int pingpong = ctrl->current_step & 1;

        // Pack my boundary faces into halo buffers
        pack_halo_faces(tile, halo_buffers, block_id, pingpong, num_blocks);
        __threadfence();  // Ensure writes visible to other blocks

        // Grid-wide sync - wait for all blocks to finish packing
        {grid_sync}

        // Unpack neighbor faces into my ghost cells
        unpack_halo_faces(tile, halo_buffers, my_route, pingpong, num_blocks);
        __syncthreads();

        // --- Phase 5: FDTD Computation ---
        if (gx < sim_x && gy < sim_y && gz < sim_z) {{
            // 7-point Laplacian from shared memory
            float center = tile[lz + 1][ly + 1][lx + 1];
            float lap = tile[lz + 1][ly + 1][lx + 2]   // +X
                      + tile[lz + 1][ly + 1][lx]       // -X
                      + tile[lz + 1][ly + 2][lx + 1]   // +Y
                      + tile[lz + 1][ly][lx + 1]       // -Y
                      + tile[lz + 2][ly + 1][lx + 1]   // +Z
                      + tile[lz][ly + 1][lx + 1]       // -Z
                      - 6.0f * center;

            // FDTD update: p_new = 2*p - p_prev + c^2*dt^2*lap
            float p_prev_val = p_prev[global_idx];
            float p_new = 2.0f * center - p_prev_val + c2_dt2 * lap;
            p_new *= damping;

            // Write to "previous" buffer (will become current after swap)
            p_prev[global_idx] = p_new;
        }}

        // --- Phase 5b: Energy Calculation (periodic) ---
        // Check if this step will report progress (after counter increment)
        bool is_progress_step = ((ctrl->current_step + 1) % {progress_interval}) == 0;

        if (is_progress_step) {{
            // Reset energy accumulator (coordinator only, before reduction)
            if (is_coordinator) {{
                ctrl->total_energy = 0.0f;
            }}
            __threadfence();  // Ensure reset is visible

            // Compute this thread's energy contribution: E = p^2
            float my_energy = 0.0f;
            if (gx < sim_x && gy < sim_y && gz < sim_z) {{
                float p_val = p_prev[global_idx];  // Just written value
                my_energy = p_val * p_val;
            }}

            // Block-level parallel reduction
            float block_energy = block_reduce_energy(my_energy, energy_reduce, {threads_per_block});

            // Block leader accumulates to global energy
            if (tid == 0) {{
                atomicAdd(&ctrl->total_energy, block_energy);
            }}
        }}

        // Grid-wide sync before buffer swap
        {grid_sync}

        // --- Phase 6: Buffer Swap & Progress ---
        if (is_coordinator) {{
            // Swap buffer pointers (toggle index)
            ctrl->current_buffer ^= 1;

            // Update counters
            ctrl->current_step++;
            ctrl->steps_remaining--;

            // Send progress update periodically
            if (ctrl->current_step % {progress_interval} == 0) {{
                K2HMessage resp;
                resp.resp_type = RESP_PROGRESS;
                resp.cmd_id = 0;
                resp.step = ctrl->current_step;
                resp.steps_remaining = ctrl->steps_remaining;
                resp.energy = ctrl->total_energy;  // Computed in Phase 5b
                k2h_send(k2h_header, k2h_slots, &resp);
            }}
        }}

        // Swap our local pointers
        float* tmp = p_curr;
        p_curr = p_prev;
        p_prev = tmp;

        {grid_sync}
    }}

    // ========== FINAL ENERGY CALCULATION ==========
    // Compute final energy before termination (all threads participate)
    if (is_coordinator) {{
        ctrl->total_energy = 0.0f;
    }}
    __threadfence();

    // Each thread's final energy contribution
    float final_energy = 0.0f;
    if (gx < sim_x && gy < sim_y && gz < sim_z) {{
        float p_val = p_curr[global_idx];  // Current buffer has final state
        final_energy = p_val * p_val;
    }}

    // Block-level parallel reduction
    float block_final_energy = block_reduce_energy(final_energy, energy_reduce, {threads_per_block});

    // Block leader accumulates to global energy
    if (tid == 0) {{
        atomicAdd(&ctrl->total_energy, block_final_energy);
    }}

    {grid_sync}

    // ========== CLEANUP ==========
    if (is_coordinator) {{
        ctrl->has_terminated = 1;

        K2HMessage resp;
        resp.resp_type = RESP_TERMINATED;
        resp.cmd_id = 0;
        resp.step = ctrl->current_step;
        resp.steps_remaining = 0;
        resp.energy = ctrl->total_energy;  // Final computed energy
        k2h_send(k2h_header, k2h_slots, &resp);
    }}
}}
"#,
        name = config.name,
        tx = tx,
        ty = ty,
        tz = tz,
        threads_per_block = threads_per_block,
        progress_interval = progress_interval,
        grid_sync_init = if config.use_cooperative {
            "cg::grid_group grid = cg::this_grid();"
        } else {
            ""
        },
        grid_sync = grid_sync,
    )
}

/// Generate PTX for the persistent FDTD kernel using nvcc.
///
/// This requires nvcc to be installed.
#[cfg(feature = "nvcc")]
pub fn compile_persistent_fdtd_to_ptx(config: &PersistentFdtdConfig) -> Result<String, String> {
    let cuda_code = generate_persistent_fdtd_kernel(config);

    // Use nvcc to compile to PTX
    use std::process::Command;

    // Write to temp file
    let temp_dir = std::env::temp_dir();
    let cuda_file = temp_dir.join("persistent_fdtd.cu");
    let ptx_file = temp_dir.join("persistent_fdtd.ptx");

    std::fs::write(&cuda_file, &cuda_code)
        .map_err(|e| format!("Failed to write CUDA file: {}", e))?;

    // Compile with nvcc
    // Use -arch=native to automatically detect the GPU architecture
    // This ensures compatibility with newer CUDA versions that dropped older sm_* support
    let mut args = vec![
        "-ptx".to_string(),
        "-o".to_string(),
        ptx_file.to_string_lossy().to_string(),
        cuda_file.to_string_lossy().to_string(),
        "-arch=native".to_string(),
        "-std=c++17".to_string(),
    ];

    if config.use_cooperative {
        args.push("-rdc=true".to_string()); // Required for cooperative groups
    }

    let output = Command::new("nvcc")
        .args(&args)
        .output()
        .map_err(|e| format!("Failed to run nvcc: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "nvcc compilation failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    std::fs::read_to_string(&ptx_file).map_err(|e| format!("Failed to read PTX: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PersistentFdtdConfig::default();
        assert_eq!(config.name, "persistent_fdtd3d");
        assert_eq!(config.tile_size, (8, 8, 8));
        assert_eq!(config.threads_per_block(), 512);
    }

    #[test]
    fn test_config_builder() {
        let config = PersistentFdtdConfig::new("my_kernel")
            .with_tile_size(4, 4, 4)
            .with_cooperative(false)
            .with_progress_interval(50);

        assert_eq!(config.name, "my_kernel");
        assert_eq!(config.tile_size, (4, 4, 4));
        assert_eq!(config.threads_per_block(), 64);
        assert!(!config.use_cooperative);
        assert_eq!(config.progress_interval, 50);
    }

    #[test]
    fn test_shared_mem_calculation() {
        let config = PersistentFdtdConfig::default(); // 8x8x8
                                                      // With halo: 10x10x10 = 1000 floats = 4000 bytes
        assert_eq!(config.shared_mem_size(), 4000);
    }

    #[test]
    fn test_generate_kernel_cooperative() {
        let config = PersistentFdtdConfig::new("test_kernel").with_cooperative(true);

        let code = generate_persistent_fdtd_kernel(&config);

        // Check includes
        assert!(code.contains("#include <cooperative_groups.h>"));
        assert!(code.contains("namespace cg = cooperative_groups;"));

        // Check structures
        assert!(code.contains("typedef struct __align__(256)"));
        assert!(code.contains("PersistentControlBlock"));
        assert!(code.contains("SpscQueueHeader"));
        assert!(code.contains("H2KMessage"));
        assert!(code.contains("K2HMessage"));

        // Check device functions
        assert!(code.contains("__device__ bool h2k_try_recv"));
        assert!(code.contains("__device__ bool k2h_send"));
        assert!(code.contains("__device__ void pack_halo_faces"));
        assert!(code.contains("__device__ void unpack_halo_faces"));

        // Check main kernel
        assert!(code.contains("extern \"C\" __global__ void"));
        assert!(code.contains("test_kernel"));
        assert!(code.contains("cg::grid_group grid = cg::this_grid()"));
        assert!(code.contains("grid.sync()"));

        // Check FDTD computation
        assert!(code.contains("7-point Laplacian"));
        assert!(code.contains("c2_dt2 * lap"));
    }

    #[test]
    fn test_generate_kernel_software_sync() {
        let config = PersistentFdtdConfig::new("test_kernel").with_cooperative(false);

        let code = generate_persistent_fdtd_kernel(&config);

        // Should NOT have cooperative groups
        assert!(!code.contains("#include <cooperative_groups.h>"));

        // Should have software barrier
        assert!(code.contains("software_grid_sync"));
        assert!(code.contains("atomicAdd"));
    }

    #[test]
    fn test_generate_kernel_command_handling() {
        let config = PersistentFdtdConfig::default();
        let code = generate_persistent_fdtd_kernel(&config);

        // Check command handling
        assert!(code.contains("CMD_RUN_STEPS"));
        assert!(code.contains("CMD_TERMINATE"));
        assert!(code.contains("CMD_INJECT_IMPULSE"));
        assert!(code.contains("CMD_GET_PROGRESS"));

        // Check response handling
        assert!(code.contains("RESP_PROGRESS"));
        assert!(code.contains("RESP_TERMINATED"));
    }

    #[test]
    fn test_generate_kernel_halo_exchange() {
        let config = PersistentFdtdConfig::new("test").with_tile_size(8, 8, 8);

        let code = generate_persistent_fdtd_kernel(&config);

        // Check halo defines
        assert!(code.contains("#define TILE_X 8"));
        assert!(code.contains("#define TILE_Y 8"));
        assert!(code.contains("#define TILE_Z 8"));
        assert!(code.contains("#define FACE_SIZE 64"));

        // Check face indices
        assert!(code.contains("FACE_POS_X"));
        assert!(code.contains("FACE_NEG_Z"));

        // Check K2K operations
        assert!(code.contains("pack_halo_faces"));
        assert!(code.contains("unpack_halo_faces"));
    }

    #[test]
    fn test_kernel_contains_persistent_loop() {
        let config = PersistentFdtdConfig::default();
        let code = generate_persistent_fdtd_kernel(&config);

        // Must have persistent loop structure
        assert!(code.contains("while (true)"));
        assert!(code.contains("if (ctrl->should_terminate)"));
        assert!(code.contains("break;"));

        // Must handle no-work case
        assert!(code.contains("if (ctrl->steps_remaining == 0)"));
        assert!(code.contains("volatile int spin_count"));
    }

    #[test]
    fn test_kernel_contains_energy_calculation() {
        let config = PersistentFdtdConfig::new("test_energy")
            .with_tile_size(8, 8, 8)
            .with_progress_interval(100);

        let code = generate_persistent_fdtd_kernel(&config);

        // Must have energy reduction function
        assert!(code.contains("block_reduce_energy"));
        assert!(code.contains("E = sum(p^2)"));
        assert!(code.contains("Parallel reduction tree"));

        // Must have shared memory for energy reduction
        assert!(code.contains("energy_reduce[512]")); // 8*8*8 = 512 threads

        // Must compute energy periodically (progress interval check)
        assert!(code.contains("is_progress_step"));
        assert!(code.contains("% 100"));

        // Must reset energy accumulator before reduction
        assert!(code.contains("ctrl->total_energy = 0.0f"));

        // Must compute p^2 for energy
        assert!(code.contains("p_val * p_val"));

        // Must accumulate via atomicAdd
        assert!(code.contains("atomicAdd(&ctrl->total_energy"));

        // Must use energy in progress response
        assert!(code.contains("resp.energy = ctrl->total_energy"));

        // Must compute final energy at termination
        assert!(code.contains("FINAL ENERGY CALCULATION"));
        assert!(code.contains("block_final_energy"));
    }
}
