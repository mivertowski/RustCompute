//! GPU-native actor system backend for 3D wave simulation.
//!
//! This module implements a cell-as-actor paradigm where each spatial cell
//! is an independent actor that:
//! - Holds its own state (pressure current/previous, cell type)
//! - Computes Laplacian using neighbor data received via messages
//! - Exchanges halo data with 6 neighbors via K2K messaging
//! - Uses HLC for temporal alignment across cells
//!
//! # Architecture
//!
//! The actor system is structured as follows:
//!
//! ```text
//! +----------+    HaloMessage    +----------+    HaloMessage    +----------+
//! | Cell(0,0)|  <------------>   | Cell(1,0)|  <------------>   | Cell(2,0)|
//! +----------+                   +----------+                   +----------+
//!      ^                              ^                              ^
//!      | HaloMessage                  | HaloMessage                  |
//!      v                              v                              v
//! +----------+                   +----------+                   +----------+
//! | Cell(0,1)|  <------------>   | Cell(1,1)|  <------------>   | Cell(2,1)|
//! +----------+                   +----------+                   +----------+
//! ```
//!
//! Each cell actor:
//! 1. Waits for halo messages from all 6 neighbors (using HLC for ordering)
//! 2. Computes the 7-point Laplacian using received neighbor pressures
//! 3. Updates its pressure state using the FDTD equation
//! 4. Sends its new pressure to all 6 neighbors
//! 5. Repeats for the next timestep
//!
//! # Features
//!
//! - **Message-based halo exchange**: No shared memory, pure actor model
//! - **HLC temporal alignment**: Ensures causal consistency across cells
//! - **GPU-native persistent kernels**: Each actor runs as a persistent GPU thread
//! - **Lock-free K2K messaging**: Direct kernel-to-kernel communication

use super::grid3d::{CellType, GridParams, SimulationGrid3D};
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{
    CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits,
};
#[cfg(feature = "cuda")]
use cudarc::nvrtc;

/// Error types for the actor backend.
#[derive(Debug)]
pub enum ActorError {
    /// GPU device error.
    DeviceError(String),
    /// Kernel compilation error.
    CompileError(String),
    /// Kernel launch error.
    LaunchError(String),
    /// Memory allocation error.
    MemoryError(String),
    /// Message routing error.
    RoutingError(String),
    /// Synchronization error.
    SyncError(String),
}

impl std::fmt::Display for ActorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActorError::DeviceError(msg) => write!(f, "Actor device error: {}", msg),
            ActorError::CompileError(msg) => write!(f, "Actor compile error: {}", msg),
            ActorError::LaunchError(msg) => write!(f, "Actor launch error: {}", msg),
            ActorError::MemoryError(msg) => write!(f, "Actor memory error: {}", msg),
            ActorError::RoutingError(msg) => write!(f, "Actor routing error: {}", msg),
            ActorError::SyncError(msg) => write!(f, "Actor sync error: {}", msg),
        }
    }
}

impl std::error::Error for ActorError {}

// ============================================================================
// Actor Message Types
// ============================================================================

/// Direction of a neighbor in 3D space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Direction3D {
    /// -X direction (west)
    West = 0,
    /// +X direction (east)
    East = 1,
    /// -Y direction (south)
    South = 2,
    /// +Y direction (north)
    North = 3,
    /// -Z direction (down)
    Down = 4,
    /// +Z direction (up)
    Up = 5,
}

impl Direction3D {
    /// Get all 6 directions.
    pub const ALL: [Direction3D; 6] = [
        Direction3D::West,
        Direction3D::East,
        Direction3D::South,
        Direction3D::North,
        Direction3D::Down,
        Direction3D::Up,
    ];

    /// Get the opposite direction.
    pub fn opposite(self) -> Direction3D {
        match self {
            Direction3D::West => Direction3D::East,
            Direction3D::East => Direction3D::West,
            Direction3D::South => Direction3D::North,
            Direction3D::North => Direction3D::South,
            Direction3D::Down => Direction3D::Up,
            Direction3D::Up => Direction3D::Down,
        }
    }

    /// Get the offset for this direction.
    pub fn offset(self) -> (i32, i32, i32) {
        match self {
            Direction3D::West => (-1, 0, 0),
            Direction3D::East => (1, 0, 0),
            Direction3D::South => (0, -1, 0),
            Direction3D::North => (0, 1, 0),
            Direction3D::Down => (0, 0, -1),
            Direction3D::Up => (0, 0, 1),
        }
    }
}

/// Halo message for communicating pressure values between neighbor cells.
///
/// This is the primary message type for actor-to-actor communication.
/// Each cell sends its pressure to neighbors, and receives pressure from neighbors.
///
/// MUST match CUDA struct exactly! Layout (40 bytes):
/// - source_idx: u32 (4)
/// - dest_idx: u32 (4)
/// - direction: u8 (1)
/// - _pad: [u8; 3] (3)
/// - pressure: f32 (4)
/// - pressure_prev: f32 (4)
/// - hlc_physical: u64 (8)
/// - hlc_logical: u64 (8)
///
/// Total: 36 bytes data, padded to 40 for u64 alignment
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct HaloMessage {
    /// Source cell linear index.
    pub source_idx: u32,
    /// Destination cell linear index.
    pub dest_idx: u32,
    /// Direction from source to destination.
    pub direction: u8,
    /// Padding for alignment.
    pub _pad: [u8; 3],
    /// Current pressure value at source cell.
    pub pressure: f32,
    /// Previous pressure value at source cell.
    pub pressure_prev: f32,
    /// HLC physical timestamp (microseconds).
    pub hlc_physical: u64,
    /// HLC logical counter.
    pub hlc_logical: u64,
}

impl HaloMessage {
    /// Create a new halo message.
    pub fn new(
        source_idx: u32,
        dest_idx: u32,
        direction: Direction3D,
        pressure: f32,
        pressure_prev: f32,
    ) -> Self {
        Self {
            source_idx,
            dest_idx,
            direction: direction as u8,
            pressure,
            pressure_prev,
            _pad: [0; 3],
            hlc_physical: 0,
            hlc_logical: 0,
        }
    }

    /// Set HLC timestamp.
    pub fn with_hlc(mut self, physical: u64, logical: u64) -> Self {
        self.hlc_physical = physical;
        self.hlc_logical = logical;
        self
    }
}

/// Cell actor state stored in GPU memory.
///
/// Each cell actor maintains its own state and processes messages
/// to update its pressure values.
///
/// IMPORTANT: This struct MUST match the CUDA struct layout exactly!
/// Layout (64 bytes total, aligned to 64):
/// - pressure: f32 (4)
/// - pressure_prev: f32 (4)
/// - cell_type: u8 (1)
/// - halos_received: u8 (1)
/// - _pad: [u8; 2] (2)
/// - reflection_coeff: f32 (4)
/// - hlc_physical: u64 (8)
/// - hlc_logical: u64 (8)
/// - timestep: u64 (8)
/// - neighbor_pressures: [f32; 6] (24)
#[derive(Debug, Clone, Copy)]
#[repr(C, align(64))]
pub struct CellActorState {
    /// Current pressure value.
    pub pressure: f32,
    /// Previous pressure value (for FDTD leapfrog).
    pub pressure_prev: f32,
    /// Cell type (0=normal, 1=absorber, 2=reflector, 3=obstacle).
    pub cell_type: u8,
    /// Number of halo messages received this step.
    pub halos_received: u8,
    /// Padding for alignment.
    pub _pad: [u8; 2],
    /// Reflection coefficient (for boundary cells).
    pub reflection_coeff: f32,
    /// HLC physical timestamp.
    pub hlc_physical: u64,
    /// HLC logical counter.
    pub hlc_logical: u64,
    /// Timestep counter.
    pub timestep: u64,
    /// Neighbor pressures (W, E, S, N, D, U).
    pub neighbor_pressures: [f32; 6],
}

// Implement traits required for cudarc
#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for CellActorState {}
#[cfg(feature = "cuda")]
unsafe impl ValidAsZeroBits for CellActorState {}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for HaloMessage {}
#[cfg(feature = "cuda")]
unsafe impl ValidAsZeroBits for HaloMessage {}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for ActorTileControl {}
#[cfg(feature = "cuda")]
unsafe impl ValidAsZeroBits for ActorTileControl {}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for K2KRoutingTable3D {}
#[cfg(feature = "cuda")]
unsafe impl ValidAsZeroBits for K2KRoutingTable3D {}

impl Default for CellActorState {
    fn default() -> Self {
        Self {
            pressure: 0.0,
            pressure_prev: 0.0,
            cell_type: 0,
            halos_received: 0,
            _pad: [0; 2],
            reflection_coeff: 1.0,
            hlc_physical: 0,
            hlc_logical: 0,
            timestep: 0,
            neighbor_pressures: [0.0; 6],
        }
    }
}

/// Inbox header for message queues.
/// MUST match the CUDA InboxHeader struct exactly.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(32))]
pub struct InboxHeader {
    /// Head pointer (write position).
    pub head: u64,
    /// Tail pointer (read position).
    pub tail: u64,
    /// Capacity of the inbox buffer.
    pub capacity: u32,
    /// Mask for modulo operations (capacity - 1).
    pub mask: u32,
    /// Padding to 32 bytes.
    pub _pad: [u8; 8],
}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for InboxHeader {}
#[cfg(feature = "cuda")]
unsafe impl ValidAsZeroBits for InboxHeader {}

// ============================================================================
// Actor Control Block
// ============================================================================

/// Control block for an actor tile (group of cell actors).
///
/// Each tile contains multiple cell actors and has its own message queues.
/// The control block manages the tile's lifecycle and message routing.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ActorTileControl {
    /// Tile is active (processing messages).
    pub is_active: u32,
    /// Tile should terminate.
    pub should_terminate: u32,
    /// Tile has terminated.
    pub has_terminated: u32,
    /// Current simulation step.
    pub current_step: u32,
    /// Number of cells in this tile.
    pub num_cells: u32,
    /// Number of cells that completed current step.
    pub cells_completed: u32,
    /// Tile X coordinate.
    pub tile_x: u32,
    /// Tile Y coordinate.
    pub tile_y: u32,
    /// Tile Z coordinate.
    pub tile_z: u32,
    /// Tile width.
    pub tile_width: u32,
    /// Tile height.
    pub tile_height: u32,
    /// Tile depth.
    pub tile_depth: u32,
    /// Global grid width.
    pub grid_width: u32,
    /// Global grid height.
    pub grid_height: u32,
    /// Global grid depth.
    pub grid_depth: u32,
    /// Slice size (width * height).
    pub slice_size: u32,
    /// HLC physical timestamp.
    pub hlc_physical: u64,
    /// HLC logical counter.
    pub hlc_logical: u64,
    /// Messages sent this step.
    pub messages_sent: u64,
    /// Messages received this step.
    pub messages_received: u64,
    /// Total messages processed.
    pub total_messages: u64,
    /// Physics: c^2 * dt^2 / dx^2.
    pub c_squared: f32,
    /// Physics: damping factor.
    pub damping: f32,
}

impl ActorTileControl {
    /// Create a new tile control block.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tile_x: u32,
        tile_y: u32,
        tile_z: u32,
        tile_width: u32,
        tile_height: u32,
        tile_depth: u32,
        grid_width: u32,
        grid_height: u32,
        grid_depth: u32,
        c_squared: f32,
        damping: f32,
    ) -> Self {
        Self {
            is_active: 0,
            should_terminate: 0,
            has_terminated: 0,
            current_step: 0,
            num_cells: tile_width * tile_height * tile_depth,
            cells_completed: 0,
            tile_x,
            tile_y,
            tile_z,
            tile_width,
            tile_height,
            tile_depth,
            grid_width,
            grid_height,
            grid_depth,
            slice_size: grid_width * grid_height,
            hlc_physical: 0,
            hlc_logical: 0,
            messages_sent: 0,
            messages_received: 0,
            total_messages: 0,
            c_squared,
            damping,
        }
    }
}

// ============================================================================
// K2K Routing for 3D Grid Topology
// ============================================================================

/// K2K routing entry for a neighbor tile.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct K2KRoute3D {
    /// Target tile linear index.
    pub target_tile_idx: u32,
    /// Direction to target (Direction3D as u8).
    pub direction: u8,
    /// Is this an edge tile (boundary)?
    pub is_boundary: u8,
    /// Padding.
    pub _pad: [u8; 2],
    /// Target tile's inbox head pointer (device address placeholder).
    pub target_inbox_head: u64,
    /// Target tile's inbox tail pointer (device address placeholder).
    pub target_inbox_tail: u64,
}

/// K2K routing table for a tile in the 3D grid.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct K2KRoutingTable3D {
    /// Number of valid routes (0-6 for 3D).
    pub num_routes: u32,
    /// Own tile index.
    pub own_tile_idx: u32,
    /// Routes to neighbor tiles.
    pub routes: [K2KRoute3D; 6],
}

impl K2KRoutingTable3D {
    /// Create routing table for a tile at (tx, ty, tz) in a grid of (nx, ny, nz) tiles.
    pub fn for_tile(tx: u32, ty: u32, tz: u32, nx: u32, ny: u32, nz: u32) -> Self {
        let own_tile_idx = tz * ny * nx + ty * nx + tx;
        let mut routes = [K2KRoute3D {
            target_tile_idx: 0,
            direction: 0,
            is_boundary: 1,
            _pad: [0; 2],
            target_inbox_head: 0,
            target_inbox_tail: 0,
        }; 6];
        let mut num_routes = 0u32;

        // West neighbor (-X)
        if tx > 0 {
            routes[num_routes as usize] = K2KRoute3D {
                target_tile_idx: tz * ny * nx + ty * nx + (tx - 1),
                direction: Direction3D::West as u8,
                is_boundary: 0,
                _pad: [0; 2],
                target_inbox_head: 0,
                target_inbox_tail: 0,
            };
            num_routes += 1;
        }

        // East neighbor (+X)
        if tx < nx - 1 {
            routes[num_routes as usize] = K2KRoute3D {
                target_tile_idx: tz * ny * nx + ty * nx + (tx + 1),
                direction: Direction3D::East as u8,
                is_boundary: 0,
                _pad: [0; 2],
                target_inbox_head: 0,
                target_inbox_tail: 0,
            };
            num_routes += 1;
        }

        // South neighbor (-Y)
        if ty > 0 {
            routes[num_routes as usize] = K2KRoute3D {
                target_tile_idx: tz * ny * nx + (ty - 1) * nx + tx,
                direction: Direction3D::South as u8,
                is_boundary: 0,
                _pad: [0; 2],
                target_inbox_head: 0,
                target_inbox_tail: 0,
            };
            num_routes += 1;
        }

        // North neighbor (+Y)
        if ty < ny - 1 {
            routes[num_routes as usize] = K2KRoute3D {
                target_tile_idx: tz * ny * nx + (ty + 1) * nx + tx,
                direction: Direction3D::North as u8,
                is_boundary: 0,
                _pad: [0; 2],
                target_inbox_head: 0,
                target_inbox_tail: 0,
            };
            num_routes += 1;
        }

        // Down neighbor (-Z)
        if tz > 0 {
            routes[num_routes as usize] = K2KRoute3D {
                target_tile_idx: (tz - 1) * ny * nx + ty * nx + tx,
                direction: Direction3D::Down as u8,
                is_boundary: 0,
                _pad: [0; 2],
                target_inbox_head: 0,
                target_inbox_tail: 0,
            };
            num_routes += 1;
        }

        // Up neighbor (+Z)
        if tz < nz - 1 {
            routes[num_routes as usize] = K2KRoute3D {
                target_tile_idx: (tz + 1) * ny * nx + ty * nx + tx,
                direction: Direction3D::Up as u8,
                is_boundary: 0,
                _pad: [0; 2],
                target_inbox_head: 0,
                target_inbox_tail: 0,
            };
            num_routes += 1;
        }

        Self {
            num_routes,
            own_tile_idx,
            routes,
        }
    }
}

// ============================================================================
// CUDA Kernel Generation for Cell Actors
// ============================================================================

/// Generate the CUDA kernel source for the cell actor ring kernel.
///
/// This kernel implements a persistent actor that:
/// 1. Waits for halo messages from neighbors
/// 2. Computes the FDTD update
/// 3. Sends new pressure to neighbors
/// 4. Uses HLC for temporal ordering
pub fn generate_cell_actor_kernel() -> String {
    r#"
// ============================================================================
// Cell Actor Ring Kernel for 3D Wave Simulation
// ============================================================================
// Each thread represents a cell actor in the 3D simulation grid.
// Actors communicate via message passing (halo exchange) instead of
// shared memory access, following the actor model paradigm.

// Halo message structure (40 bytes)
// MUST match Rust HaloMessage struct exactly!
struct HaloMessage {
    unsigned int source_idx;      // 4 bytes, offset 0
    unsigned int dest_idx;        // 4 bytes, offset 4
    unsigned char direction;      // 1 byte, offset 8
    unsigned char _pad[3];        // 3 bytes padding, offset 9
    float pressure;               // 4 bytes, offset 12
    float pressure_prev;          // 4 bytes, offset 16
    unsigned long long hlc_physical;  // 8 bytes, offset 24 (aligned)
    unsigned long long hlc_logical;   // 8 bytes, offset 32
    // Total: 40 bytes
};

// Cell actor state (64 bytes, aligned)
// MUST match Rust CellActorState struct exactly!
struct __align__(64) CellActorState {
    float pressure;               // 4 bytes, offset 0
    float pressure_prev;          // 4 bytes, offset 4
    unsigned char cell_type;      // 1 byte, offset 8
    unsigned char halos_received; // 1 byte, offset 9
    unsigned char _pad[2];        // 2 bytes, offset 10
    float reflection_coeff;       // 4 bytes, offset 12
    unsigned long long hlc_physical;  // 8 bytes, offset 16
    unsigned long long hlc_logical;   // 8 bytes, offset 24
    unsigned long long timestep;      // 8 bytes, offset 32
    float neighbor_pressures[6];  // 24 bytes, offset 40
    // Total: 64 bytes
};

// Tile control block (128 bytes, aligned)
struct __align__(128) ActorTileControl {
    unsigned int is_active;
    unsigned int should_terminate;
    unsigned int has_terminated;
    unsigned int current_step;
    unsigned int num_cells;
    unsigned int cells_completed;
    unsigned int tile_x;
    unsigned int tile_y;
    unsigned int tile_z;
    unsigned int tile_width;
    unsigned int tile_height;
    unsigned int tile_depth;
    unsigned int grid_width;
    unsigned int grid_height;
    unsigned int grid_depth;
    unsigned int slice_size;
    unsigned long long hlc_physical;
    unsigned long long hlc_logical;
    unsigned long long messages_sent;
    unsigned long long messages_received;
    unsigned long long total_messages;
    float c_squared;
    float damping;
    unsigned char _reserved[16];
};

// K2K routing entry
struct K2KRoute3D {
    unsigned int target_tile_idx;
    unsigned char direction;
    unsigned char is_boundary;
    unsigned char _pad[2];
    unsigned long long target_inbox_head;
    unsigned long long target_inbox_tail;
};

// K2K routing table
struct K2KRoutingTable3D {
    unsigned int num_routes;
    unsigned int own_tile_idx;
    K2KRoute3D routes[6];
};

// Inbox header for message queues
struct InboxHeader {
    unsigned long long head;
    unsigned long long tail;
    unsigned int capacity;
    unsigned int mask;
};

// Direction constants
#define DIR_WEST  0
#define DIR_EAST  1
#define DIR_SOUTH 2
#define DIR_NORTH 3
#define DIR_DOWN  4
#define DIR_UP    5

// Get neighbor index, returns -1 if boundary
__device__ int get_neighbor_idx(
    int x, int y, int z,
    int direction,
    int width, int height, int depth,
    int slice_size
) {
    int nx = x, ny = y, nz = z;
    switch (direction) {
        case DIR_WEST:  nx = x - 1; break;
        case DIR_EAST:  nx = x + 1; break;
        case DIR_SOUTH: ny = y - 1; break;
        case DIR_NORTH: ny = y + 1; break;
        case DIR_DOWN:  nz = z - 1; break;
        case DIR_UP:    nz = z + 1; break;
    }

    // Boundary check
    if (nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= depth) {
        return -1;
    }

    return nz * slice_size + ny * width + nx;
}

// Get opposite direction
__device__ int opposite_direction(int dir) {
    switch (dir) {
        case DIR_WEST:  return DIR_EAST;
        case DIR_EAST:  return DIR_WEST;
        case DIR_SOUTH: return DIR_NORTH;
        case DIR_NORTH: return DIR_SOUTH;
        case DIR_DOWN:  return DIR_UP;
        case DIR_UP:    return DIR_DOWN;
    }
    return -1;
}

// Send halo message to neighbor via inbox
__device__ void send_halo_message(
    InboxHeader* inbox_headers,
    HaloMessage* inbox_buffers,
    int neighbor_idx,
    unsigned int source_idx,
    float pressure,
    float pressure_prev,
    int direction,
    unsigned long long hlc_physical,
    unsigned long long hlc_logical,
    unsigned int inbox_capacity
) {
    if (neighbor_idx < 0) return;

    InboxHeader* header = &inbox_headers[neighbor_idx];
    HaloMessage* inbox = &inbox_buffers[neighbor_idx * inbox_capacity];

    // Atomically claim a slot in the inbox
    unsigned long long slot = atomicAdd(&header->head, 1ULL);
    unsigned int idx = (unsigned int)(slot & header->mask);

    // Write message
    inbox[idx].source_idx = source_idx;
    inbox[idx].dest_idx = (unsigned int)neighbor_idx;
    inbox[idx].direction = (unsigned char)direction;
    inbox[idx].pressure = pressure;
    inbox[idx].pressure_prev = pressure_prev;
    inbox[idx].hlc_physical = hlc_physical;
    inbox[idx].hlc_logical = hlc_logical;

    __threadfence();  // Ensure message is visible to receiver
}

// Try to receive a halo message from inbox
__device__ int try_receive_halo(
    InboxHeader* my_header,
    HaloMessage* my_inbox,
    HaloMessage* out_msg,
    unsigned int inbox_capacity
) {
    unsigned long long head = atomicAdd(&my_header->head, 0ULL);
    unsigned long long tail = atomicAdd(&my_header->tail, 0ULL);

    if (head == tail) {
        return 0;  // No messages
    }

    // Read message
    unsigned int idx = (unsigned int)(tail & my_header->mask);
    *out_msg = my_inbox[idx];

    // Advance tail
    atomicAdd(&my_header->tail, 1ULL);
    return 1;
}

// Main cell actor kernel
extern "C" __global__ void cell_actor_kernel(
    ActorTileControl* __restrict__ tile_control,
    CellActorState* __restrict__ cell_states,
    InboxHeader* __restrict__ inbox_headers,
    HaloMessage* __restrict__ inbox_buffers,
    K2KRoutingTable3D* __restrict__ routing_tables,
    unsigned int inbox_capacity,
    unsigned int max_iterations
) {
    // Global cell index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = tile_control->grid_width * tile_control->grid_height * tile_control->grid_depth;

    if (tid >= total_cells) return;

    // Get my position in 3D grid
    int slice_size = tile_control->slice_size;
    int width = tile_control->grid_width;
    int height = tile_control->grid_height;
    int depth = tile_control->grid_depth;

    int z = tid / slice_size;
    int remainder = tid % slice_size;
    int y = remainder / width;
    int x = remainder % width;

    // Get physics parameters
    float c_squared = tile_control->c_squared;
    float damping = tile_control->damping;

    // Local HLC state
    unsigned long long hlc_physical = cell_states[tid].hlc_physical;
    unsigned long long hlc_logical = cell_states[tid].hlc_logical;

    // Get my inbox
    InboxHeader* my_inbox_header = &inbox_headers[tid];
    HaloMessage* my_inbox = &inbox_buffers[tid * inbox_capacity];

    // Count how many neighbors I have (non-boundary)
    int num_neighbors = 0;
    int neighbor_indices[6];
    for (int dir = 0; dir < 6; dir++) {
        neighbor_indices[dir] = get_neighbor_idx(x, y, z, dir, width, height, depth, slice_size);
        if (neighbor_indices[dir] >= 0) {
            num_neighbors++;
        }
    }

    // Main actor loop
    for (unsigned int iter = 0; iter < max_iterations; iter++) {
        // Check for termination
        if (atomicAdd(&tile_control->should_terminate, 0) != 0) {
            break;
        }

        // Wait until active
        while (atomicAdd(&tile_control->is_active, 0) == 0) {
            if (atomicAdd(&tile_control->should_terminate, 0) != 0) {
                goto terminate;
            }
            __nanosleep(100);
        }

        // Skip boundary cells (they don't compute, just reflect)
        if (cell_states[tid].cell_type == 3) {  // Obstacle
            __syncthreads();
            continue;
        }

        // Step 1: Send halo messages to all neighbors
        float my_pressure = cell_states[tid].pressure;
        float my_pressure_prev = cell_states[tid].pressure_prev;

        hlc_logical++;  // Tick HLC

        for (int dir = 0; dir < 6; dir++) {
            int neighbor_idx = neighbor_indices[dir];
            if (neighbor_idx >= 0) {
                send_halo_message(
                    inbox_headers,
                    inbox_buffers,
                    neighbor_idx,
                    (unsigned int)tid,
                    my_pressure,
                    my_pressure_prev,
                    dir,
                    hlc_physical,
                    hlc_logical,
                    inbox_capacity
                );
            }
        }

        __threadfence();  // Ensure all sends are visible
        __syncthreads();  // Synchronize before receiving

        // Step 2: Receive halo messages from all neighbors
        float neighbor_pressures[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        int halos_received = 0;

        // For boundary cells, use reflection
        for (int dir = 0; dir < 6; dir++) {
            if (neighbor_indices[dir] < 0) {
                // Boundary: use own pressure (reflecting BC)
                neighbor_pressures[dir] = my_pressure * cell_states[tid].reflection_coeff;
                halos_received++;
            }
        }

        // Receive messages from actual neighbors
        int max_recv_attempts = num_neighbors * 10;  // Allow some retries
        int recv_attempts = 0;

        while (halos_received < 6 && recv_attempts < max_recv_attempts) {
            HaloMessage msg;
            if (try_receive_halo(my_inbox_header, my_inbox, &msg, inbox_capacity)) {
                // Determine which direction this message came from
                int source_dir = opposite_direction(msg.direction);
                neighbor_pressures[source_dir] = msg.pressure;

                // Update HLC based on received timestamp
                if (msg.hlc_physical > hlc_physical) {
                    hlc_physical = msg.hlc_physical;
                    hlc_logical = msg.hlc_logical + 1;
                } else if (msg.hlc_physical == hlc_physical && msg.hlc_logical >= hlc_logical) {
                    hlc_logical = msg.hlc_logical + 1;
                }

                halos_received++;
            }
            recv_attempts++;
        }

        __syncthreads();  // Ensure all actors have received

        // Step 3: Compute 7-point Laplacian using received neighbor pressures
        float laplacian =
            neighbor_pressures[DIR_WEST] + neighbor_pressures[DIR_EAST] +
            neighbor_pressures[DIR_SOUTH] + neighbor_pressures[DIR_NORTH] +
            neighbor_pressures[DIR_DOWN] + neighbor_pressures[DIR_UP] -
            6.0f * my_pressure;

        // Step 4: FDTD update equation
        float new_pressure = 2.0f * my_pressure - my_pressure_prev + c_squared * laplacian;

        // Apply damping
        new_pressure *= damping;

        // Apply boundary conditions for absorber cells
        if (cell_states[tid].cell_type == 1) {  // Absorber
            new_pressure *= cell_states[tid].reflection_coeff;
        }

        // Step 5: Update state
        cell_states[tid].pressure_prev = my_pressure;
        cell_states[tid].pressure = new_pressure;
        cell_states[tid].hlc_physical = hlc_physical;
        cell_states[tid].hlc_logical = hlc_logical;
        cell_states[tid].timestep++;

        // Copy neighbor pressures for debugging/visualization
        for (int dir = 0; dir < 6; dir++) {
            cell_states[tid].neighbor_pressures[dir] = neighbor_pressures[dir];
        }

        __syncthreads();  // Ensure all updates complete before next iteration
    }

terminate:
    // Mark as terminated
    if (tid == 0) {
        tile_control->hlc_physical = hlc_physical;
        tile_control->hlc_logical = hlc_logical;
        atomicExch(&tile_control->has_terminated, 1);
    }
}

// Kernel to initialize cell states from pressure arrays
extern "C" __global__ void init_cell_states(
    CellActorState* __restrict__ cell_states,
    const float* __restrict__ pressure,
    const float* __restrict__ pressure_prev,
    const unsigned char* __restrict__ cell_types,
    const float* __restrict__ reflection_coeff,
    unsigned int total_cells
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_cells) return;

    cell_states[tid].pressure = pressure[tid];
    cell_states[tid].pressure_prev = pressure_prev[tid];
    cell_states[tid].cell_type = cell_types[tid];
    cell_states[tid].halos_received = 0;
    cell_states[tid].reflection_coeff = reflection_coeff[tid];
    cell_states[tid].hlc_physical = 0;
    cell_states[tid].hlc_logical = 0;
    cell_states[tid].timestep = 0;

    for (int i = 0; i < 6; i++) {
        cell_states[tid].neighbor_pressures[i] = 0.0f;
    }
}

// Kernel to extract pressure from cell states
extern "C" __global__ void extract_pressure(
    const CellActorState* __restrict__ cell_states,
    float* __restrict__ pressure,
    float* __restrict__ pressure_prev,
    unsigned int total_cells
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_cells) return;

    pressure[tid] = cell_states[tid].pressure;
    pressure_prev[tid] = cell_states[tid].pressure_prev;
}

// Kernel to inject impulse into cell states
extern "C" __global__ void inject_impulse_actor(
    CellActorState* __restrict__ cell_states,
    unsigned int x, unsigned int y, unsigned int z,
    unsigned int width, unsigned int slice_size,
    float amplitude
) {
    unsigned int idx = z * slice_size + y * width + x;
    cell_states[idx].pressure += amplitude;
    cell_states[idx].pressure_prev += amplitude * 0.5f;
}

// Kernel to initialize inbox headers
extern "C" __global__ void init_inbox_headers(
    InboxHeader* __restrict__ inbox_headers,
    unsigned int total_cells,
    unsigned int inbox_capacity
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_cells) return;

    inbox_headers[tid].head = 0;
    inbox_headers[tid].tail = 0;
    inbox_headers[tid].capacity = inbox_capacity;
    inbox_headers[tid].mask = inbox_capacity - 1;
}
"#
    .to_string()
}

// ============================================================================
// Actor GPU Backend
// ============================================================================

/// Configuration for the actor-based GPU backend.
#[derive(Debug, Clone)]
pub struct ActorBackendConfig {
    /// Inbox capacity per cell (power of 2, default: 16).
    pub inbox_capacity: u32,
    /// Maximum iterations per kernel launch (default: 1).
    pub max_iterations: u32,
    /// Block size for kernel launches (default: 256).
    pub block_size: u32,
}

impl Default for ActorBackendConfig {
    fn default() -> Self {
        Self {
            inbox_capacity: 16,
            max_iterations: 1,
            block_size: 256,
        }
    }
}

/// GPU-native actor backend for 3D wave simulation.
///
/// This backend implements the cell-as-actor paradigm where each spatial cell
/// is an independent actor that communicates with neighbors via message passing.
#[cfg(feature = "cuda")]
pub struct ActorGpuBackend3D {
    /// CUDA device.
    device: Arc<CudaDevice>,
    /// Cell actor states.
    cell_states: CudaSlice<CellActorState>,
    /// Inbox headers (one per cell).
    inbox_headers: CudaSlice<InboxHeader>,
    /// Inbox buffers (messages for each cell).
    inbox_buffers: CudaSlice<HaloMessage>,
    /// Tile control block.
    tile_control: CudaSlice<ActorTileControl>,
    /// Grid parameters.
    params: GridParams,
    /// Total number of cells.
    total_cells: usize,
    /// Configuration.
    config: ActorBackendConfig,
}

#[cfg(feature = "cuda")]
impl ActorGpuBackend3D {
    /// Create a new actor-based GPU backend from a CPU grid.
    pub fn new(grid: &SimulationGrid3D, config: ActorBackendConfig) -> Result<Self, ActorError> {
        // Validate inbox capacity is power of 2
        if !config.inbox_capacity.is_power_of_two() {
            return Err(ActorError::MemoryError(
                "Inbox capacity must be power of 2".into(),
            ));
        }

        // Initialize CUDA device
        let device = CudaDevice::new(0).map_err(|e| ActorError::DeviceError(e.to_string()))?;

        // Compile CUDA kernel
        let kernel_source = generate_cell_actor_kernel();
        let ptx = nvrtc::compile_ptx(&kernel_source)
            .map_err(|e| ActorError::CompileError(e.to_string()))?;

        device
            .load_ptx(
                ptx,
                "actor_wavesim",
                &[
                    "cell_actor_kernel",
                    "init_cell_states",
                    "extract_pressure",
                    "inject_impulse_actor",
                    "init_inbox_headers",
                ],
            )
            .map_err(|e| ActorError::CompileError(e.to_string()))?;

        let total_cells = grid.width * grid.height * grid.depth;
        let params = GridParams::from(grid);

        // Allocate cell states
        let initial_states: Vec<CellActorState> = (0..total_cells)
            .map(|i| CellActorState {
                pressure: grid.pressure[i],
                pressure_prev: grid.pressure_prev[i],
                cell_type: match grid.cell_types[i] {
                    CellType::Normal => 0,
                    CellType::Absorber => 1,
                    CellType::Reflector => 2,
                    CellType::Obstacle => 3,
                },
                halos_received: 0,
                _pad: [0; 2],
                reflection_coeff: grid.reflection_coeff[i],
                hlc_physical: 0,
                hlc_logical: 0,
                timestep: 0,
                neighbor_pressures: [0.0; 6],
            })
            .collect();

        let cell_states = device
            .htod_sync_copy(&initial_states)
            .map_err(|e| ActorError::MemoryError(e.to_string()))?;

        // Allocate inbox headers (InboxHeader struct, 32 bytes each)
        let inbox_headers_data: Vec<InboxHeader> = vec![InboxHeader::default(); total_cells];
        let inbox_headers = device
            .htod_sync_copy(&inbox_headers_data)
            .map_err(|e| ActorError::MemoryError(e.to_string()))?;

        // Allocate inbox buffers
        let inbox_buffer_size = total_cells * config.inbox_capacity as usize;
        let inbox_buffers_data: Vec<HaloMessage> =
            vec![HaloMessage::new(0, 0, Direction3D::West, 0.0, 0.0); inbox_buffer_size];
        let inbox_buffers = device
            .htod_sync_copy(&inbox_buffers_data)
            .map_err(|e| ActorError::MemoryError(e.to_string()))?;

        // Allocate tile control
        let tile_ctrl = ActorTileControl::new(
            0,
            0,
            0,
            params.width,
            params.height,
            params.depth,
            params.width,
            params.height,
            params.depth,
            params.c_squared,
            params.damping,
        );
        let tile_control = device
            .htod_sync_copy(&[tile_ctrl])
            .map_err(|e| ActorError::MemoryError(e.to_string()))?;

        // Initialize inbox headers on GPU
        {
            let init_kernel = device
                .get_func("actor_wavesim", "init_inbox_headers")
                .ok_or_else(|| ActorError::LaunchError("init_inbox_headers not found".into()))?;

            let blocks = total_cells.div_ceil(config.block_size as usize);
            let launch_config = LaunchConfig {
                block_dim: (config.block_size, 1, 1),
                grid_dim: (blocks as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                init_kernel
                    .launch(
                        launch_config,
                        (&inbox_headers, total_cells as u32, config.inbox_capacity),
                    )
                    .map_err(|e| ActorError::LaunchError(e.to_string()))?;
            }
        }

        Ok(Self {
            device,
            cell_states,
            inbox_headers,
            inbox_buffers,
            tile_control,
            params,
            total_cells,
            config,
        })
    }

    /// Perform one or more simulation steps using the actor model.
    pub fn step(&mut self, grid: &mut SimulationGrid3D, num_steps: u32) -> Result<(), ActorError> {
        // Activate the tile
        {
            let mut ctrl = vec![ActorTileControl::new(
                0,
                0,
                0,
                self.params.width,
                self.params.height,
                self.params.depth,
                self.params.width,
                self.params.height,
                self.params.depth,
                self.params.c_squared,
                self.params.damping,
            )];
            ctrl[0].is_active = 1;
            ctrl[0].should_terminate = 0;
            ctrl[0].has_terminated = 0;
            self.device
                .htod_sync_copy_into(&ctrl, &mut self.tile_control)
                .map_err(|e| ActorError::MemoryError(e.to_string()))?;
        }

        // Launch the actor kernel
        let actor_kernel = self
            .device
            .get_func("actor_wavesim", "cell_actor_kernel")
            .ok_or_else(|| ActorError::LaunchError("cell_actor_kernel not found".into()))?;

        let blocks = self.total_cells.div_ceil(self.config.block_size as usize);
        let launch_config = LaunchConfig {
            block_dim: (self.config.block_size, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Create dummy routing table (not used in single-tile mode)
        let routing_table = K2KRoutingTable3D::for_tile(0, 0, 0, 1, 1, 1);
        let routing_tables = self
            .device
            .htod_sync_copy(&[routing_table])
            .map_err(|e| ActorError::MemoryError(e.to_string()))?;

        unsafe {
            actor_kernel
                .launch(
                    launch_config,
                    (
                        &self.tile_control,
                        &self.cell_states,
                        &self.inbox_headers,
                        &self.inbox_buffers,
                        &routing_tables,
                        self.config.inbox_capacity,
                        num_steps,
                    ),
                )
                .map_err(|e| ActorError::LaunchError(e.to_string()))?;
        }

        // Synchronize
        self.device
            .synchronize()
            .map_err(|e| ActorError::SyncError(e.to_string()))?;

        // Update grid state
        grid.step += num_steps as u64;
        grid.time += grid.params.time_step * num_steps as f32;

        Ok(())
    }

    /// Download pressure data from GPU to CPU grid.
    pub fn download_pressure(&self, grid: &mut SimulationGrid3D) -> Result<(), ActorError> {
        // Extract pressure from cell states
        let extract_kernel = self
            .device
            .get_func("actor_wavesim", "extract_pressure")
            .ok_or_else(|| ActorError::LaunchError("extract_pressure not found".into()))?;

        // Allocate temporary buffers on GPU
        let pressure_gpu = self
            .device
            .htod_sync_copy(&grid.pressure)
            .map_err(|e| ActorError::MemoryError(e.to_string()))?;
        let pressure_prev_gpu = self
            .device
            .htod_sync_copy(&grid.pressure_prev)
            .map_err(|e| ActorError::MemoryError(e.to_string()))?;

        let blocks = self.total_cells.div_ceil(self.config.block_size as usize);
        let launch_config = LaunchConfig {
            block_dim: (self.config.block_size, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            extract_kernel
                .launch(
                    launch_config,
                    (
                        &self.cell_states,
                        &pressure_gpu,
                        &pressure_prev_gpu,
                        self.total_cells as u32,
                    ),
                )
                .map_err(|e| ActorError::LaunchError(e.to_string()))?;
        }

        // Download
        self.device
            .dtoh_sync_copy_into(&pressure_gpu, &mut grid.pressure)
            .map_err(|e| ActorError::MemoryError(e.to_string()))?;
        self.device
            .dtoh_sync_copy_into(&pressure_prev_gpu, &mut grid.pressure_prev)
            .map_err(|e| ActorError::MemoryError(e.to_string()))?;

        Ok(())
    }

    /// Upload pressure data from CPU grid to GPU.
    pub fn upload_pressure(&mut self, grid: &SimulationGrid3D) -> Result<(), ActorError> {
        // Convert cell types to u8
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

        // Upload to GPU
        let pressure_gpu = self
            .device
            .htod_sync_copy(&grid.pressure)
            .map_err(|e| ActorError::MemoryError(e.to_string()))?;
        let pressure_prev_gpu = self
            .device
            .htod_sync_copy(&grid.pressure_prev)
            .map_err(|e| ActorError::MemoryError(e.to_string()))?;
        let cell_types_gpu = self
            .device
            .htod_sync_copy(&cell_types_u8)
            .map_err(|e| ActorError::MemoryError(e.to_string()))?;
        let reflection_coeff_gpu = self
            .device
            .htod_sync_copy(&grid.reflection_coeff)
            .map_err(|e| ActorError::MemoryError(e.to_string()))?;

        // Initialize cell states
        let init_kernel = self
            .device
            .get_func("actor_wavesim", "init_cell_states")
            .ok_or_else(|| ActorError::LaunchError("init_cell_states not found".into()))?;

        let blocks = self.total_cells.div_ceil(self.config.block_size as usize);
        let launch_config = LaunchConfig {
            block_dim: (self.config.block_size, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            init_kernel
                .launch(
                    launch_config,
                    (
                        &self.cell_states,
                        &pressure_gpu,
                        &pressure_prev_gpu,
                        &cell_types_gpu,
                        &reflection_coeff_gpu,
                        self.total_cells as u32,
                    ),
                )
                .map_err(|e| ActorError::LaunchError(e.to_string()))?;
        }

        Ok(())
    }

    /// Reset the GPU buffers to match the CPU grid.
    pub fn reset(&mut self, grid: &SimulationGrid3D) -> Result<(), ActorError> {
        self.upload_pressure(grid)?;

        // Reset inbox headers
        let init_inbox_kernel = self
            .device
            .get_func("actor_wavesim", "init_inbox_headers")
            .ok_or_else(|| ActorError::LaunchError("init_inbox_headers not found".into()))?;

        let blocks = self.total_cells.div_ceil(self.config.block_size as usize);
        let launch_config = LaunchConfig {
            block_dim: (self.config.block_size, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            init_inbox_kernel
                .launch(
                    launch_config,
                    (
                        &self.inbox_headers,
                        self.total_cells as u32,
                        self.config.inbox_capacity,
                    ),
                )
                .map_err(|e| ActorError::LaunchError(e.to_string()))?;
        }

        Ok(())
    }

    /// Inject an impulse directly on the GPU.
    pub fn inject_impulse(
        &mut self,
        x: u32,
        y: u32,
        z: u32,
        amplitude: f32,
    ) -> Result<(), ActorError> {
        let kernel = self
            .device
            .get_func("actor_wavesim", "inject_impulse_actor")
            .ok_or_else(|| ActorError::LaunchError("inject_impulse_actor not found".into()))?;

        let launch_config = LaunchConfig {
            block_dim: (1, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(
                    launch_config,
                    (
                        &self.cell_states,
                        x,
                        y,
                        z,
                        self.params.width,
                        self.params.slice_size,
                        amplitude,
                    ),
                )
                .map_err(|e| ActorError::LaunchError(e.to_string()))?;
        }

        Ok(())
    }

    /// Get the current GPU memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let cell_state_size = std::mem::size_of::<CellActorState>();
        let inbox_header_size = 32; // sizeof(InboxHeader)
        let halo_msg_size = std::mem::size_of::<HaloMessage>();
        let control_size = std::mem::size_of::<ActorTileControl>();

        self.total_cells * cell_state_size
            + self.total_cells * inbox_header_size
            + self.total_cells * self.config.inbox_capacity as usize * halo_msg_size
            + control_size
    }

    /// Get statistics about the actor system.
    pub fn stats(&self) -> ActorStats {
        ActorStats {
            total_cells: self.total_cells,
            inbox_capacity: self.config.inbox_capacity as usize,
            memory_usage_bytes: self.memory_usage(),
        }
    }
}

/// Statistics about the actor backend.
#[derive(Debug, Clone)]
pub struct ActorStats {
    /// Total number of cell actors.
    pub total_cells: usize,
    /// Inbox capacity per cell.
    pub inbox_capacity: usize,
    /// GPU memory usage in bytes.
    pub memory_usage_bytes: usize,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direction_opposite() {
        assert_eq!(Direction3D::West.opposite(), Direction3D::East);
        assert_eq!(Direction3D::East.opposite(), Direction3D::West);
        assert_eq!(Direction3D::South.opposite(), Direction3D::North);
        assert_eq!(Direction3D::North.opposite(), Direction3D::South);
        assert_eq!(Direction3D::Down.opposite(), Direction3D::Up);
        assert_eq!(Direction3D::Up.opposite(), Direction3D::Down);
    }

    #[test]
    fn test_direction_offset() {
        assert_eq!(Direction3D::West.offset(), (-1, 0, 0));
        assert_eq!(Direction3D::East.offset(), (1, 0, 0));
        assert_eq!(Direction3D::South.offset(), (0, -1, 0));
        assert_eq!(Direction3D::North.offset(), (0, 1, 0));
        assert_eq!(Direction3D::Down.offset(), (0, 0, -1));
        assert_eq!(Direction3D::Up.offset(), (0, 0, 1));
    }

    #[test]
    fn test_halo_message_size() {
        // HaloMessage: 40 bytes (36 bytes data + padding for u64 alignment)
        let size = std::mem::size_of::<HaloMessage>();
        assert_eq!(
            size, 40,
            "HaloMessage should be exactly 40 bytes, got {}",
            size
        );
    }

    #[test]
    fn test_cell_actor_state_size() {
        // CellActorState: 64 bytes aligned
        let size = std::mem::size_of::<CellActorState>();
        assert_eq!(
            size, 64,
            "CellActorState should be exactly 64 bytes, got {}",
            size
        );
    }

    #[test]
    fn test_inbox_header_size() {
        // InboxHeader: 32 bytes aligned
        let size = std::mem::size_of::<InboxHeader>();
        assert_eq!(
            size, 32,
            "InboxHeader should be exactly 32 bytes, got {}",
            size
        );
    }

    #[test]
    fn test_actor_tile_control_size() {
        // ActorTileControl: varies based on padding
        let size = std::mem::size_of::<ActorTileControl>();
        assert!(
            size >= 104,
            "ActorTileControl should be at least 104 bytes, got {}",
            size
        );
    }

    #[test]
    fn test_routing_table_corner() {
        // Corner cell (0, 0, 0) should have 3 neighbors
        let table = K2KRoutingTable3D::for_tile(0, 0, 0, 4, 4, 4);
        assert_eq!(table.num_routes, 3);
        assert_eq!(table.own_tile_idx, 0);
    }

    #[test]
    fn test_routing_table_edge() {
        // Edge cell (1, 0, 0) should have 4 neighbors
        let table = K2KRoutingTable3D::for_tile(1, 0, 0, 4, 4, 4);
        assert_eq!(table.num_routes, 4);
    }

    #[test]
    fn test_routing_table_center() {
        // Center cell (1, 1, 1) should have 6 neighbors
        let table = K2KRoutingTable3D::for_tile(1, 1, 1, 4, 4, 4);
        assert_eq!(table.num_routes, 6);
    }

    #[test]
    fn test_kernel_generation() {
        let kernel = generate_cell_actor_kernel();
        assert!(kernel.contains("cell_actor_kernel"));
        assert!(kernel.contains("HaloMessage"));
        assert!(kernel.contains("CellActorState"));
        assert!(kernel.contains("ActorTileControl"));
        assert!(kernel.contains("send_halo_message"));
        assert!(kernel.contains("try_receive_halo"));
        assert!(kernel.contains("hlc_physical"));
        assert!(kernel.contains("hlc_logical"));
    }

    #[test]
    fn test_halo_message_creation() {
        let msg = HaloMessage::new(0, 1, Direction3D::East, 1.5, 1.2).with_hlc(1000, 5);

        assert_eq!(msg.source_idx, 0);
        assert_eq!(msg.dest_idx, 1);
        assert_eq!(msg.direction, Direction3D::East as u8);
        assert_eq!(msg.pressure, 1.5);
        assert_eq!(msg.pressure_prev, 1.2);
        assert_eq!(msg.hlc_physical, 1000);
        assert_eq!(msg.hlc_logical, 5);
    }

    #[test]
    fn test_actor_config_default() {
        let config = ActorBackendConfig::default();
        assert!(config.inbox_capacity.is_power_of_two());
        assert_eq!(config.inbox_capacity, 16);
        assert_eq!(config.max_iterations, 1);
        assert_eq!(config.block_size, 256);
    }
}
