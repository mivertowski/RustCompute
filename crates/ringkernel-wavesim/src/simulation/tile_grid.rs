//! Tile-based GPU-native actor grid for wave simulation.
//!
//! This module implements a scalable actor-based simulation where each actor
//! manages a tile (e.g., 16x16) of cells. Tiles communicate via K2K messaging
//! to exchange boundary (halo) data, showcasing RingKernel's GPU-native actor model.
//!
//! ## Architecture
//!
//! ```text
//! +--------+--------+--------+
//! | Tile   | Tile   | Tile   |
//! | (0,0)  | (1,0)  | (2,0)  |
//! +--------+--------+--------+
//! | Tile   | Tile   | Tile   |
//! | (0,1)  | (1,1)  | (2,1)  |
//! +--------+--------+--------+
//! ```
//!
//! Each tile:
//! - Has a pressure buffer with 1-cell halo for neighbor access
//! - Exchanges halo data with 4 neighbors via K2K messaging
//! - Computes FDTD for its interior cells in parallel (GPU or CPU)
//!
//! ## Halo Exchange Pattern
//!
//! ```text
//! +--+----------------+--+
//! |  | North Halo     |  |  <- Received from north neighbor
//! +--+----------------+--+
//! |W |                |E |
//! |  |   Interior     |  |
//! |  |    16x16       |  |
//! +--+----------------+--+
//! |  | South Halo     |  |  <- Received from south neighbor
//! +--+----------------+--+
//! ```

use super::AcousticParams;
use ringkernel::prelude::*;
use ringkernel_core::hlc::HlcTimestamp;
use ringkernel_core::k2k::{DeliveryStatus, K2KBroker, K2KBuilder, K2KEndpoint};
use ringkernel_core::message::{MessageEnvelope, MessageHeader};
use ringkernel_core::runtime::KernelId;
use std::collections::HashMap;
use std::sync::Arc;

// Legacy GPU compute (upload/download per step)
#[cfg(feature = "wgpu")]
use super::gpu_compute::{init_wgpu, TileBuffers, TileGpuComputePool};

// New GPU-persistent backends using unified trait
#[cfg(feature = "cuda")]
use super::cuda_compute::CudaTileBackend;
#[cfg(any(feature = "wgpu", feature = "cuda"))]
use super::gpu_backend::{Edge, FdtdParams, TileGpuBackend, TileGpuBuffers};
#[cfg(feature = "wgpu")]
use super::wgpu_compute::{WgpuBuffer, WgpuTileBackend};

/// Default tile size (16x16 cells per tile).
pub const DEFAULT_TILE_SIZE: u32 = 16;

/// Direction for halo exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HaloDirection {
    North,
    South,
    East,
    West,
}

impl HaloDirection {
    /// Get the opposite direction.
    pub fn opposite(self) -> Self {
        match self {
            HaloDirection::North => HaloDirection::South,
            HaloDirection::South => HaloDirection::North,
            HaloDirection::East => HaloDirection::West,
            HaloDirection::West => HaloDirection::East,
        }
    }

    /// All directions.
    pub const ALL: [HaloDirection; 4] = [
        HaloDirection::North,
        HaloDirection::South,
        HaloDirection::East,
        HaloDirection::West,
    ];
}

/// Message type IDs for K2K communication.
const HALO_EXCHANGE_TYPE_ID: u64 = 200;

/// A single tile actor managing a region of the simulation grid.
pub struct TileActor {
    /// Tile position in tile coordinates.
    pub tile_x: u32,
    pub tile_y: u32,

    /// Size of the tile (cells per side).
    tile_size: u32,

    /// Buffer width including halos: tile_size + 2.
    buffer_width: usize,

    /// Current pressure values (buffer_width * buffer_width).
    /// Layout: row-major with 1-cell halo on each side.
    pressure: Vec<f32>,

    /// Previous pressure values for time-stepping.
    pressure_prev: Vec<f32>,

    /// Halo buffers for async exchange (received from neighbors).
    halo_north: Vec<f32>,
    halo_south: Vec<f32>,
    halo_east: Vec<f32>,
    halo_west: Vec<f32>,

    /// Flags indicating which halos have been received this step.
    halos_received: [bool; 4],

    /// Neighbor tile IDs (None if at grid boundary).
    neighbor_north: Option<KernelId>,
    neighbor_south: Option<KernelId>,
    neighbor_east: Option<KernelId>,
    neighbor_west: Option<KernelId>,

    /// K2K endpoint for this tile.
    endpoint: K2KEndpoint,

    /// Kernel ID for this tile (used for K2K routing).
    #[allow(dead_code)]
    kernel_id: KernelId,

    /// Is this tile on the grid boundary?
    is_north_boundary: bool,
    is_south_boundary: bool,
    is_east_boundary: bool,
    is_west_boundary: bool,

    /// Reflection coefficient for boundaries.
    reflection_coeff: f32,
}

impl TileActor {
    /// Create a new tile actor.
    pub fn new(
        tile_x: u32,
        tile_y: u32,
        tile_size: u32,
        tiles_x: u32,
        tiles_y: u32,
        broker: &Arc<K2KBroker>,
        reflection_coeff: f32,
    ) -> Self {
        let buffer_width = (tile_size + 2) as usize;
        let buffer_size = buffer_width * buffer_width;

        let kernel_id = Self::tile_kernel_id(tile_x, tile_y);
        let endpoint = broker.register(kernel_id.clone());

        // Determine neighbors
        let neighbor_north = if tile_y > 0 {
            Some(Self::tile_kernel_id(tile_x, tile_y - 1))
        } else {
            None
        };
        let neighbor_south = if tile_y < tiles_y - 1 {
            Some(Self::tile_kernel_id(tile_x, tile_y + 1))
        } else {
            None
        };
        let neighbor_west = if tile_x > 0 {
            Some(Self::tile_kernel_id(tile_x - 1, tile_y))
        } else {
            None
        };
        let neighbor_east = if tile_x < tiles_x - 1 {
            Some(Self::tile_kernel_id(tile_x + 1, tile_y))
        } else {
            None
        };

        Self {
            tile_x,
            tile_y,
            tile_size,
            buffer_width,
            pressure: vec![0.0; buffer_size],
            pressure_prev: vec![0.0; buffer_size],
            halo_north: vec![0.0; tile_size as usize],
            halo_south: vec![0.0; tile_size as usize],
            halo_east: vec![0.0; tile_size as usize],
            halo_west: vec![0.0; tile_size as usize],
            halos_received: [false; 4],
            neighbor_north,
            neighbor_south,
            neighbor_east,
            neighbor_west,
            endpoint,
            kernel_id,
            is_north_boundary: tile_y == 0,
            is_south_boundary: tile_y == tiles_y - 1,
            is_east_boundary: tile_x == tiles_x - 1,
            is_west_boundary: tile_x == 0,
            reflection_coeff,
        }
    }

    /// Generate kernel ID for a tile.
    pub fn tile_kernel_id(tile_x: u32, tile_y: u32) -> KernelId {
        KernelId::new(format!("tile_{}_{}", tile_x, tile_y))
    }

    /// Convert local tile coordinates to buffer index.
    #[inline(always)]
    fn buffer_idx(&self, local_x: usize, local_y: usize) -> usize {
        // Add 1 to account for halo
        (local_y + 1) * self.buffer_width + (local_x + 1)
    }

    /// Get pressure at local tile coordinates.
    pub fn get_pressure(&self, local_x: u32, local_y: u32) -> f32 {
        self.pressure[self.buffer_idx(local_x as usize, local_y as usize)]
    }

    /// Set pressure at local tile coordinates.
    pub fn set_pressure(&mut self, local_x: u32, local_y: u32, value: f32) {
        let idx = self.buffer_idx(local_x as usize, local_y as usize);
        self.pressure[idx] = value;
    }

    /// Extract edge data for sending to neighbors.
    fn extract_edge(&self, direction: HaloDirection) -> Vec<f32> {
        let size = self.tile_size as usize;
        let mut edge = vec![0.0; size];

        match direction {
            HaloDirection::North => {
                // First interior row (y = 0)
                for (x, cell) in edge.iter_mut().enumerate().take(size) {
                    *cell = self.pressure[self.buffer_idx(x, 0)];
                }
            }
            HaloDirection::South => {
                // Last interior row (y = size - 1)
                for (x, cell) in edge.iter_mut().enumerate().take(size) {
                    *cell = self.pressure[self.buffer_idx(x, size - 1)];
                }
            }
            HaloDirection::West => {
                // First interior column (x = 0)
                for (y, cell) in edge.iter_mut().enumerate().take(size) {
                    *cell = self.pressure[self.buffer_idx(0, y)];
                }
            }
            HaloDirection::East => {
                // Last interior column (x = size - 1)
                for (y, cell) in edge.iter_mut().enumerate().take(size) {
                    *cell = self.pressure[self.buffer_idx(size - 1, y)];
                }
            }
        }

        edge
    }

    /// Apply received halo data to the buffer.
    fn apply_halo(&mut self, direction: HaloDirection, data: &[f32]) {
        let size = self.tile_size as usize;
        let bw = self.buffer_width;

        match direction {
            HaloDirection::North => {
                // Top halo row (y = -1 in local coords, index row 0 in buffer)
                self.pressure[1..(size + 1)].copy_from_slice(&data[..size]);
            }
            HaloDirection::South => {
                // Bottom halo row (y = size in local coords)
                for (x, &val) in data.iter().enumerate().take(size) {
                    self.pressure[(size + 1) * bw + (x + 1)] = val;
                }
            }
            HaloDirection::West => {
                // Left halo column (x = -1 in local coords, index col 0 in buffer)
                for (y, &val) in data.iter().enumerate().take(size) {
                    self.pressure[(y + 1) * bw] = val;
                }
            }
            HaloDirection::East => {
                // Right halo column (x = size in local coords)
                for (y, &val) in data.iter().enumerate().take(size) {
                    self.pressure[(y + 1) * bw + (size + 1)] = val;
                }
            }
        }
    }

    /// Apply boundary reflection for edges without neighbors.
    fn apply_boundary_reflection(&mut self) {
        let size = self.tile_size as usize;
        let bw = self.buffer_width;
        let refl = self.reflection_coeff;

        if self.is_north_boundary {
            // Reflect interior row 0 to halo row
            for x in 0..size {
                let interior_idx = self.buffer_idx(x, 0);
                self.pressure[x + 1] = self.pressure[interior_idx] * refl;
            }
        }

        if self.is_south_boundary {
            // Reflect interior row (size-1) to halo row
            for x in 0..size {
                let interior_idx = self.buffer_idx(x, size - 1);
                self.pressure[(size + 1) * bw + (x + 1)] = self.pressure[interior_idx] * refl;
            }
        }

        if self.is_west_boundary {
            // Reflect interior col 0 to halo col
            for y in 0..size {
                let interior_idx = self.buffer_idx(0, y);
                self.pressure[(y + 1) * bw] = self.pressure[interior_idx] * refl;
            }
        }

        if self.is_east_boundary {
            // Reflect interior col (size-1) to halo col
            for y in 0..size {
                let interior_idx = self.buffer_idx(size - 1, y);
                self.pressure[(y + 1) * bw + (size + 1)] = self.pressure[interior_idx] * refl;
            }
        }
    }

    /// Send halo data to all neighbors via K2K.
    pub async fn send_halos(&self) -> Result<()> {
        for direction in HaloDirection::ALL {
            let neighbor = match direction {
                HaloDirection::North => &self.neighbor_north,
                HaloDirection::South => &self.neighbor_south,
                HaloDirection::East => &self.neighbor_east,
                HaloDirection::West => &self.neighbor_west,
            };

            if let Some(neighbor_id) = neighbor {
                let edge_data = self.extract_edge(direction);

                // Serialize halo data into envelope
                let envelope = Self::create_halo_envelope(direction.opposite(), &edge_data);

                let receipt = self.endpoint.send(neighbor_id.clone(), envelope).await?;
                if receipt.status != DeliveryStatus::Delivered {
                    tracing::warn!(
                        "Halo send failed: tile ({},{}) -> {:?}, status: {:?}",
                        self.tile_x,
                        self.tile_y,
                        neighbor_id,
                        receipt.status
                    );
                }
            }
        }

        Ok(())
    }

    /// Receive halo data from all neighbors via K2K.
    pub fn receive_halos(&mut self) {
        // Reset received flags
        self.halos_received = [false; 4];

        // Process all pending messages
        while let Some(message) = self.endpoint.try_receive() {
            if let Some((direction, data)) = Self::parse_halo_envelope(&message.envelope) {
                // Store in halo buffer
                match direction {
                    HaloDirection::North => {
                        self.halo_north.copy_from_slice(&data);
                        self.halos_received[0] = true;
                    }
                    HaloDirection::South => {
                        self.halo_south.copy_from_slice(&data);
                        self.halos_received[1] = true;
                    }
                    HaloDirection::East => {
                        self.halo_east.copy_from_slice(&data);
                        self.halos_received[2] = true;
                    }
                    HaloDirection::West => {
                        self.halo_west.copy_from_slice(&data);
                        self.halos_received[3] = true;
                    }
                }
            }
        }

        // Apply received halos to pressure buffer
        if self.halos_received[0] && self.neighbor_north.is_some() {
            self.apply_halo(HaloDirection::North, &self.halo_north.clone());
        }
        if self.halos_received[1] && self.neighbor_south.is_some() {
            self.apply_halo(HaloDirection::South, &self.halo_south.clone());
        }
        if self.halos_received[2] && self.neighbor_east.is_some() {
            self.apply_halo(HaloDirection::East, &self.halo_east.clone());
        }
        if self.halos_received[3] && self.neighbor_west.is_some() {
            self.apply_halo(HaloDirection::West, &self.halo_west.clone());
        }

        // Apply boundary reflection for edges without neighbors
        self.apply_boundary_reflection();
    }

    /// Create a K2K envelope containing halo data.
    fn create_halo_envelope(direction: HaloDirection, data: &[f32]) -> MessageEnvelope {
        // Pack direction and data into payload
        // Format: [direction_byte, f32 data...]
        let mut payload = Vec::with_capacity(1 + data.len() * 4);
        payload.push(direction as u8);
        for &value in data {
            payload.extend_from_slice(&value.to_le_bytes());
        }

        let header = MessageHeader::new(
            HALO_EXCHANGE_TYPE_ID,
            0, // source_kernel (host)
            0, // dest_kernel (will be set by K2K routing)
            payload.len(),
            HlcTimestamp::now(0),
        );

        MessageEnvelope { header, payload }
    }

    /// Parse a K2K envelope to extract halo data.
    fn parse_halo_envelope(envelope: &MessageEnvelope) -> Option<(HaloDirection, Vec<f32>)> {
        if envelope.header.message_type != HALO_EXCHANGE_TYPE_ID {
            return None;
        }

        if envelope.payload.is_empty() {
            return None;
        }

        let direction = match envelope.payload[0] {
            0 => HaloDirection::North,
            1 => HaloDirection::South,
            2 => HaloDirection::East,
            3 => HaloDirection::West,
            _ => return None,
        };

        let data: Vec<f32> = envelope.payload[1..]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Some((direction, data))
    }

    /// Compute FDTD for all interior cells.
    pub fn compute_fdtd(&mut self, c2: f32, damping: f32) {
        let size = self.tile_size as usize;
        let bw = self.buffer_width;

        // Process all interior cells
        for local_y in 0..size {
            for local_x in 0..size {
                let idx = (local_y + 1) * bw + (local_x + 1);

                let p_curr = self.pressure[idx];
                let p_prev = self.pressure_prev[idx];

                // Neighbor access (halos provide boundary values)
                let p_north = self.pressure[idx - bw];
                let p_south = self.pressure[idx + bw];
                let p_west = self.pressure[idx - 1];
                let p_east = self.pressure[idx + 1];

                // FDTD computation
                let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
                let p_new = 2.0 * p_curr - p_prev + c2 * laplacian;

                self.pressure_prev[idx] = p_new * damping;
            }
        }
    }

    /// Swap pressure buffers after FDTD step.
    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.pressure, &mut self.pressure_prev);
    }

    /// Reset tile to initial state.
    pub fn reset(&mut self) {
        self.pressure.fill(0.0);
        self.pressure_prev.fill(0.0);
    }
}

/// GPU compute resources for tile-based FDTD (wgpu feature only).
/// This is the legacy mode with upload/download per step.
#[cfg(feature = "wgpu")]
struct GpuComputeResources {
    /// Shared compute pool for all tiles.
    pool: TileGpuComputePool,
    /// Per-tile GPU buffers.
    tile_buffers: HashMap<(u32, u32), TileBuffers>,
}

/// GPU-persistent state using WGPU backend.
/// Pressure stays GPU-resident, only halos are transferred.
#[cfg(feature = "wgpu")]
struct WgpuPersistentState {
    /// WGPU backend.
    backend: WgpuTileBackend,
    /// Per-tile GPU buffers.
    tile_buffers: HashMap<(u32, u32), TileGpuBuffers<WgpuBuffer>>,
}

/// GPU-persistent state using CUDA backend.
/// Pressure stays GPU-resident, only halos are transferred.
#[cfg(feature = "cuda")]
struct CudaPersistentState {
    /// CUDA backend.
    backend: CudaTileBackend,
    /// Per-tile GPU buffers.
    tile_buffers: HashMap<(u32, u32), TileGpuBuffers<ringkernel_cuda::CudaBuffer>>,
}

/// Which GPU backend is active for persistent state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPersistentBackend {
    /// WGPU (WebGPU) backend.
    Wgpu,
    /// CUDA (NVIDIA) backend.
    Cuda,
}

/// A simulation grid using tile actors with K2K messaging.
pub struct TileKernelGrid {
    /// Grid dimensions in cells.
    pub width: u32,
    pub height: u32,

    /// Tile size (cells per tile side).
    tile_size: u32,

    /// Grid dimensions in tiles.
    tiles_x: u32,
    tiles_y: u32,

    /// Tile actors indexed by (tile_x, tile_y).
    tiles: HashMap<(u32, u32), TileActor>,

    /// K2K broker for inter-tile messaging.
    broker: Arc<K2KBroker>,

    /// Acoustic simulation parameters.
    pub params: AcousticParams,

    /// The backend being used.
    backend: Backend,

    /// RingKernel runtime (for kernel handle management).
    #[allow(dead_code)]
    runtime: Arc<RingKernel>,

    /// Legacy GPU compute pool for tile FDTD (optional, requires wgpu feature).
    /// This mode uploads/downloads full buffers each step.
    #[cfg(feature = "wgpu")]
    gpu_compute: Option<GpuComputeResources>,

    /// GPU-persistent WGPU state (optional).
    /// Pressure stays GPU-resident, only halos are transferred.
    #[cfg(feature = "wgpu")]
    wgpu_persistent: Option<WgpuPersistentState>,

    /// GPU-persistent CUDA state (optional).
    /// Pressure stays GPU-resident, only halos are transferred.
    #[cfg(feature = "cuda")]
    cuda_persistent: Option<CudaPersistentState>,
}

impl std::fmt::Debug for TileKernelGrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileKernelGrid")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("tile_size", &self.tile_size)
            .field("tiles_x", &self.tiles_x)
            .field("tiles_y", &self.tiles_y)
            .field("tile_count", &self.tiles.len())
            .field("backend", &self.backend)
            .finish()
    }
}

impl TileKernelGrid {
    /// Create a new tile-based kernel grid.
    pub async fn new(
        width: u32,
        height: u32,
        params: AcousticParams,
        backend: Backend,
    ) -> Result<Self> {
        Self::with_tile_size(width, height, params, backend, DEFAULT_TILE_SIZE).await
    }

    /// Create a new tile-based kernel grid with custom tile size.
    pub async fn with_tile_size(
        width: u32,
        height: u32,
        params: AcousticParams,
        backend: Backend,
        tile_size: u32,
    ) -> Result<Self> {
        let runtime = Arc::new(RingKernel::with_backend(backend).await?);

        // Calculate tile grid dimensions
        let tiles_x = width.div_ceil(tile_size);
        let tiles_y = height.div_ceil(tile_size);

        // Create K2K broker
        let broker = K2KBuilder::new()
            .max_pending_messages(tiles_x as usize * tiles_y as usize * 8)
            .build();

        // Create tile actors
        let mut tiles = HashMap::new();
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let tile = TileActor::new(tx, ty, tile_size, tiles_x, tiles_y, &broker, 0.95);
                tiles.insert((tx, ty), tile);
            }
        }

        tracing::info!(
            "Created TileKernelGrid: {}x{} cells, {}x{} tiles ({}x{} per tile), {} tile actors",
            width,
            height,
            tiles_x,
            tiles_y,
            tile_size,
            tile_size,
            tiles.len()
        );

        Ok(Self {
            width,
            height,
            tile_size,
            tiles_x,
            tiles_y,
            tiles,
            broker,
            params,
            backend,
            runtime,
            #[cfg(feature = "wgpu")]
            gpu_compute: None,
            #[cfg(feature = "wgpu")]
            wgpu_persistent: None,
            #[cfg(feature = "cuda")]
            cuda_persistent: None,
        })
    }

    /// Convert global cell coordinates to (tile_x, tile_y, local_x, local_y).
    fn global_to_tile_coords(&self, x: u32, y: u32) -> (u32, u32, u32, u32) {
        let tile_x = x / self.tile_size;
        let tile_y = y / self.tile_size;
        let local_x = x % self.tile_size;
        let local_y = y % self.tile_size;
        (tile_x, tile_y, local_x, local_y)
    }

    /// Perform one simulation step using tile actors with K2K messaging.
    pub async fn step(&mut self) -> Result<()> {
        let c2 = self.params.courant_number().powi(2);
        let damping = 1.0 - self.params.damping;

        // Phase 1: All tiles send halo data to neighbors (K2K)
        for tile in self.tiles.values() {
            tile.send_halos().await?;
        }

        // Phase 2: All tiles receive halo data from neighbors (K2K)
        for tile in self.tiles.values_mut() {
            tile.receive_halos();
        }

        // Phase 3: All tiles compute FDTD for interior cells
        for tile in self.tiles.values_mut() {
            tile.compute_fdtd(c2, damping);
        }

        // Phase 4: All tiles swap buffers
        for tile in self.tiles.values_mut() {
            tile.swap_buffers();
        }

        Ok(())
    }

    /// Enable GPU compute for tile FDTD acceleration.
    ///
    /// This initializes WGPU resources and creates GPU buffers for each tile.
    /// Once enabled, use `step_gpu()` for GPU-accelerated simulation steps.
    #[cfg(feature = "wgpu")]
    pub async fn enable_gpu_compute(&mut self) -> Result<()> {
        let (device, queue) = init_wgpu().await?;
        let pool = TileGpuComputePool::new(device, queue, self.tile_size)?;

        // Create GPU buffers for each tile
        let mut tile_buffers = HashMap::new();
        for (tile_coords, _) in &self.tiles {
            let buffers = pool.create_tile_buffers();
            tile_buffers.insert(*tile_coords, buffers);
        }

        let num_tiles = tile_buffers.len();
        self.gpu_compute = Some(GpuComputeResources { pool, tile_buffers });

        tracing::info!(
            "GPU compute enabled for TileKernelGrid: {} tiles with GPU buffers",
            num_tiles
        );

        Ok(())
    }

    /// Check if GPU compute is enabled.
    #[cfg(feature = "wgpu")]
    pub fn is_gpu_enabled(&self) -> bool {
        self.gpu_compute.is_some()
    }

    /// Perform one simulation step using GPU compute for FDTD.
    ///
    /// This is the hybrid approach:
    /// - K2K messaging handles halo exchange between tiles (actor model showcase)
    /// - GPU compute shaders handle FDTD for tile interiors (parallel compute)
    #[cfg(feature = "wgpu")]
    pub async fn step_gpu(&mut self) -> Result<()> {
        let gpu = self.gpu_compute.as_ref().ok_or_else(|| {
            ringkernel_core::error::RingKernelError::BackendError(
                "GPU compute not enabled. Call enable_gpu_compute() first.".to_string(),
            )
        })?;

        let c2 = self.params.courant_number().powi(2);
        let damping = 1.0 - self.params.damping;

        // Phase 1: All tiles send halo data to neighbors (K2K)
        for tile in self.tiles.values() {
            tile.send_halos().await?;
        }

        // Phase 2: All tiles receive halo data from neighbors (K2K)
        for tile in self.tiles.values_mut() {
            tile.receive_halos();
        }

        // Phase 3: GPU compute FDTD for all tile interiors
        // Each tile dispatches to GPU and reads back results
        for ((tx, ty), tile) in self.tiles.iter_mut() {
            if let Some(buffers) = gpu.tile_buffers.get(&(*tx, *ty)) {
                // Get tile's pressure buffer (with halos already applied from K2K)
                let pressure = &tile.pressure;
                let pressure_prev = &tile.pressure_prev;

                // Dispatch GPU compute
                let result = gpu
                    .pool
                    .compute_fdtd(buffers, pressure, pressure_prev, c2, damping);

                // Copy results back to tile's pressure_prev buffer
                tile.pressure_prev.copy_from_slice(&result);
            } else {
                // Fallback to CPU if no GPU buffers (shouldn't happen)
                tile.compute_fdtd(c2, damping);
            }
        }

        // Phase 4: All tiles swap buffers
        for tile in self.tiles.values_mut() {
            tile.swap_buffers();
        }

        Ok(())
    }

    /// Enable GPU-persistent WGPU compute.
    ///
    /// This mode keeps pressure state GPU-resident and only transfers halos
    /// for K2K communication. Much faster than the legacy mode which uploads
    /// and downloads full buffers each step.
    #[cfg(feature = "wgpu")]
    pub async fn enable_wgpu_persistent(&mut self) -> Result<()> {
        let backend = WgpuTileBackend::new(self.tile_size).await?;

        // Create GPU buffers for each tile and upload initial state
        let mut tile_buffers = HashMap::new();
        for ((tx, ty), tile) in &self.tiles {
            let buffers = backend.create_tile_buffers(self.tile_size)?;
            // Upload initial state (all zeros typically)
            backend.upload_initial_state(&buffers, &tile.pressure, &tile.pressure_prev)?;
            tile_buffers.insert((*tx, *ty), buffers);
        }

        let num_tiles = tile_buffers.len();
        self.wgpu_persistent = Some(WgpuPersistentState {
            backend,
            tile_buffers,
        });

        tracing::info!(
            "WGPU GPU-persistent compute enabled for TileKernelGrid: {} tiles",
            num_tiles
        );

        Ok(())
    }

    /// Enable GPU-persistent CUDA compute.
    ///
    /// This mode keeps pressure state GPU-resident and only transfers halos
    /// for K2K communication. Much faster than upload/download per step.
    #[cfg(feature = "cuda")]
    pub fn enable_cuda_persistent(&mut self) -> Result<()> {
        let backend = CudaTileBackend::new(self.tile_size)?;

        // Create GPU buffers for each tile and upload initial state
        let mut tile_buffers = HashMap::new();
        for ((tx, ty), tile) in &self.tiles {
            let buffers = backend.create_tile_buffers(self.tile_size)?;
            // Upload initial state
            backend.upload_initial_state(&buffers, &tile.pressure, &tile.pressure_prev)?;
            tile_buffers.insert((*tx, *ty), buffers);
        }

        let num_tiles = tile_buffers.len();
        self.cuda_persistent = Some(CudaPersistentState {
            backend,
            tile_buffers,
        });

        tracing::info!(
            "CUDA GPU-persistent compute enabled for TileKernelGrid: {} tiles",
            num_tiles
        );

        Ok(())
    }

    /// Check if GPU-persistent mode is enabled.
    pub fn is_gpu_persistent_enabled(&self) -> bool {
        #[cfg(feature = "wgpu")]
        if self.wgpu_persistent.is_some() {
            return true;
        }
        #[cfg(feature = "cuda")]
        if self.cuda_persistent.is_some() {
            return true;
        }
        false
    }

    /// Get which GPU-persistent backend is active.
    pub fn gpu_persistent_backend(&self) -> Option<GpuPersistentBackend> {
        #[cfg(feature = "cuda")]
        if self.cuda_persistent.is_some() {
            return Some(GpuPersistentBackend::Cuda);
        }
        #[cfg(feature = "wgpu")]
        if self.wgpu_persistent.is_some() {
            return Some(GpuPersistentBackend::Wgpu);
        }
        None
    }

    /// Perform one simulation step using GPU-persistent WGPU compute.
    ///
    /// State stays GPU-resident. Only halos are transferred for K2K messaging.
    #[cfg(feature = "wgpu")]
    pub fn step_wgpu_persistent(&mut self) -> Result<()> {
        let state = self.wgpu_persistent.as_ref().ok_or_else(|| {
            ringkernel_core::error::RingKernelError::BackendError(
                "WGPU persistent compute not enabled. Call enable_wgpu_persistent() first."
                    .to_string(),
            )
        })?;

        let c2 = self.params.courant_number().powi(2);
        let damping = 1.0 - self.params.damping;
        let fdtd_params = FdtdParams::new(self.tile_size, c2, damping);

        // Phase 1: Extract halos from GPU and send via K2K
        // Build halo messages for all tiles
        let mut halo_messages: Vec<(KernelId, HaloDirection, Vec<f32>)> = Vec::new();

        for ((tx, ty), _tile) in &self.tiles {
            if let Some(buffers) = state.tile_buffers.get(&(*tx, *ty)) {
                // Extract halos based on tile's neighbors
                let tile = self.tiles.get(&(*tx, *ty)).unwrap();

                if tile.neighbor_north.is_some() {
                    let halo = state.backend.extract_halo(buffers, Edge::North)?;
                    halo_messages.push((
                        tile.neighbor_north.clone().unwrap(),
                        HaloDirection::South,
                        halo,
                    ));
                }
                if tile.neighbor_south.is_some() {
                    let halo = state.backend.extract_halo(buffers, Edge::South)?;
                    halo_messages.push((
                        tile.neighbor_south.clone().unwrap(),
                        HaloDirection::North,
                        halo,
                    ));
                }
                if tile.neighbor_west.is_some() {
                    let halo = state.backend.extract_halo(buffers, Edge::West)?;
                    halo_messages.push((
                        tile.neighbor_west.clone().unwrap(),
                        HaloDirection::East,
                        halo,
                    ));
                }
                if tile.neighbor_east.is_some() {
                    let halo = state.backend.extract_halo(buffers, Edge::East)?;
                    halo_messages.push((
                        tile.neighbor_east.clone().unwrap(),
                        HaloDirection::West,
                        halo,
                    ));
                }
            }
        }

        // Phase 2: Inject halos into GPU buffers
        // Group messages by destination tile
        for (dest_id, direction, halo_data) in halo_messages {
            // Find the tile coords from kernel ID
            for (tx, ty) in self.tiles.keys() {
                if TileActor::tile_kernel_id(*tx, *ty) == dest_id {
                    if let Some(buffers) = state.tile_buffers.get(&(*tx, *ty)) {
                        let edge = match direction {
                            HaloDirection::North => Edge::North,
                            HaloDirection::South => Edge::South,
                            HaloDirection::East => Edge::East,
                            HaloDirection::West => Edge::West,
                        };
                        state.backend.inject_halo(buffers, edge, &halo_data)?;
                    }
                    break;
                }
            }
        }

        // Phase 3: Apply boundary reflection for edge tiles (on GPU)
        // For now, we handle this by keeping boundaries at zero (absorbing)
        // TODO: Add GPU kernel for boundary reflection

        // Phase 4: Compute FDTD on GPU for all tiles
        for ((tx, ty), _tile) in &self.tiles {
            if let Some(buffers) = state.tile_buffers.get(&(*tx, *ty)) {
                state.backend.fdtd_step(buffers, &fdtd_params)?;
            }
        }

        // Synchronize
        state.backend.synchronize()?;

        // Phase 5: Swap buffers (pointer swap only, no data movement)
        // NOTE: We need mutable access to state for this
        let state = self.wgpu_persistent.as_mut().unwrap();
        for buffers in state.tile_buffers.values_mut() {
            state.backend.swap_buffers(buffers);
        }

        Ok(())
    }

    /// Perform one simulation step using GPU-persistent CUDA compute.
    ///
    /// State stays GPU-resident. Only halos are transferred for K2K messaging.
    #[cfg(feature = "cuda")]
    pub fn step_cuda_persistent(&mut self) -> Result<()> {
        let state = self.cuda_persistent.as_ref().ok_or_else(|| {
            ringkernel_core::error::RingKernelError::BackendError(
                "CUDA persistent compute not enabled. Call enable_cuda_persistent() first."
                    .to_string(),
            )
        })?;

        let c2 = self.params.courant_number().powi(2);
        let damping = 1.0 - self.params.damping;
        let fdtd_params = FdtdParams::new(self.tile_size, c2, damping);

        // Phase 1: Extract halos from GPU and build messages
        let mut halo_messages: Vec<(KernelId, HaloDirection, Vec<f32>)> = Vec::new();

        for (tx, ty) in self.tiles.keys() {
            if let Some(buffers) = state.tile_buffers.get(&(*tx, *ty)) {
                let tile = self.tiles.get(&(*tx, *ty)).unwrap();

                if tile.neighbor_north.is_some() {
                    let halo = state.backend.extract_halo(buffers, Edge::North)?;
                    halo_messages.push((
                        tile.neighbor_north.clone().unwrap(),
                        HaloDirection::South,
                        halo,
                    ));
                }
                if tile.neighbor_south.is_some() {
                    let halo = state.backend.extract_halo(buffers, Edge::South)?;
                    halo_messages.push((
                        tile.neighbor_south.clone().unwrap(),
                        HaloDirection::North,
                        halo,
                    ));
                }
                if tile.neighbor_west.is_some() {
                    let halo = state.backend.extract_halo(buffers, Edge::West)?;
                    halo_messages.push((
                        tile.neighbor_west.clone().unwrap(),
                        HaloDirection::East,
                        halo,
                    ));
                }
                if tile.neighbor_east.is_some() {
                    let halo = state.backend.extract_halo(buffers, Edge::East)?;
                    halo_messages.push((
                        tile.neighbor_east.clone().unwrap(),
                        HaloDirection::West,
                        halo,
                    ));
                }
            }
        }

        // Phase 2: Inject halos into GPU buffers
        for (dest_id, direction, halo_data) in halo_messages {
            for (tx, ty) in self.tiles.keys() {
                if TileActor::tile_kernel_id(*tx, *ty) == dest_id {
                    if let Some(buffers) = state.tile_buffers.get(&(*tx, *ty)) {
                        let edge = match direction {
                            HaloDirection::North => Edge::North,
                            HaloDirection::South => Edge::South,
                            HaloDirection::East => Edge::East,
                            HaloDirection::West => Edge::West,
                        };
                        state.backend.inject_halo(buffers, edge, &halo_data)?;
                    }
                    break;
                }
            }
        }

        // Phase 3: Compute FDTD on GPU for all tiles
        for (tx, ty) in self.tiles.keys() {
            if let Some(buffers) = state.tile_buffers.get(&(*tx, *ty)) {
                state.backend.fdtd_step(buffers, &fdtd_params)?;
            }
        }

        // Synchronize
        state.backend.synchronize()?;

        // Phase 4: Swap buffers
        let state = self.cuda_persistent.as_mut().unwrap();
        for buffers in state.tile_buffers.values_mut() {
            state.backend.swap_buffers(buffers);
        }

        Ok(())
    }

    /// Read pressure grid from GPU for visualization.
    ///
    /// This is the only time we transfer full tile data from GPU to host.
    /// Call this only when you need to render the simulation.
    #[cfg(feature = "wgpu")]
    pub fn read_pressure_from_wgpu(&self) -> Result<Vec<Vec<f32>>> {
        let state = self.wgpu_persistent.as_ref().ok_or_else(|| {
            ringkernel_core::error::RingKernelError::BackendError(
                "WGPU persistent compute not enabled.".to_string(),
            )
        })?;

        let mut grid = vec![vec![0.0; self.width as usize]; self.height as usize];

        for ((tx, ty), buffers) in &state.tile_buffers {
            let interior = state.backend.read_interior_pressure(buffers)?;

            // Copy tile interior to grid
            let tile_start_x = tx * self.tile_size;
            let tile_start_y = ty * self.tile_size;

            for ly in 0..self.tile_size {
                for lx in 0..self.tile_size {
                    let gx = tile_start_x + lx;
                    let gy = tile_start_y + ly;
                    if gx < self.width && gy < self.height {
                        grid[gy as usize][gx as usize] =
                            interior[(ly * self.tile_size + lx) as usize];
                    }
                }
            }
        }

        Ok(grid)
    }

    /// Read pressure grid from CUDA for visualization.
    #[cfg(feature = "cuda")]
    pub fn read_pressure_from_cuda(&self) -> Result<Vec<Vec<f32>>> {
        let state = self.cuda_persistent.as_ref().ok_or_else(|| {
            ringkernel_core::error::RingKernelError::BackendError(
                "CUDA persistent compute not enabled.".to_string(),
            )
        })?;

        let mut grid = vec![vec![0.0; self.width as usize]; self.height as usize];

        for ((tx, ty), buffers) in &state.tile_buffers {
            let interior = state.backend.read_interior_pressure(buffers)?;

            let tile_start_x = tx * self.tile_size;
            let tile_start_y = ty * self.tile_size;

            for ly in 0..self.tile_size {
                for lx in 0..self.tile_size {
                    let gx = tile_start_x + lx;
                    let gy = tile_start_y + ly;
                    if gx < self.width && gy < self.height {
                        grid[gy as usize][gx as usize] =
                            interior[(ly * self.tile_size + lx) as usize];
                    }
                }
            }
        }

        Ok(grid)
    }

    /// Upload a pressure impulse to GPU-persistent state.
    ///
    /// For GPU-persistent mode, we need to upload the impulse to the GPU.
    #[cfg(feature = "wgpu")]
    pub fn inject_impulse_wgpu(&mut self, x: u32, y: u32, amplitude: f32) -> Result<()> {
        if x >= self.width || y >= self.height {
            return Ok(());
        }

        let (tile_x, tile_y, local_x, local_y) = self.global_to_tile_coords(x, y);

        // First, update the CPU tile (for consistency)
        if let Some(tile) = self.tiles.get_mut(&(tile_x, tile_y)) {
            let current = tile.get_pressure(local_x, local_y);
            tile.set_pressure(local_x, local_y, current + amplitude);

            // Then upload to GPU
            if let Some(state) = &self.wgpu_persistent {
                if let Some(buffers) = state.tile_buffers.get(&(tile_x, tile_y)) {
                    state.backend.upload_initial_state(
                        buffers,
                        &tile.pressure,
                        &tile.pressure_prev,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Upload a pressure impulse to CUDA GPU-persistent state.
    #[cfg(feature = "cuda")]
    pub fn inject_impulse_cuda(&mut self, x: u32, y: u32, amplitude: f32) -> Result<()> {
        if x >= self.width || y >= self.height {
            return Ok(());
        }

        let (tile_x, tile_y, local_x, local_y) = self.global_to_tile_coords(x, y);

        if let Some(tile) = self.tiles.get_mut(&(tile_x, tile_y)) {
            let current = tile.get_pressure(local_x, local_y);
            tile.set_pressure(local_x, local_y, current + amplitude);

            if let Some(state) = &self.cuda_persistent {
                if let Some(buffers) = state.tile_buffers.get(&(tile_x, tile_y)) {
                    state.backend.upload_initial_state(
                        buffers,
                        &tile.pressure,
                        &tile.pressure_prev,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Inject an impulse at the given global grid position.
    pub fn inject_impulse(&mut self, x: u32, y: u32, amplitude: f32) {
        if x >= self.width || y >= self.height {
            return;
        }

        let (tile_x, tile_y, local_x, local_y) = self.global_to_tile_coords(x, y);

        if let Some(tile) = self.tiles.get_mut(&(tile_x, tile_y)) {
            let current = tile.get_pressure(local_x, local_y);
            tile.set_pressure(local_x, local_y, current + amplitude);
        }
    }

    /// Get the pressure grid for visualization.
    pub fn get_pressure_grid(&self) -> Vec<Vec<f32>> {
        let mut grid = vec![vec![0.0; self.width as usize]; self.height as usize];

        for y in 0..self.height {
            for x in 0..self.width {
                let (tile_x, tile_y, local_x, local_y) = self.global_to_tile_coords(x, y);

                if let Some(tile) = self.tiles.get(&(tile_x, tile_y)) {
                    // Handle tiles at the edge that may have fewer cells
                    if local_x < self.tile_size && local_y < self.tile_size {
                        grid[y as usize][x as usize] = tile.get_pressure(local_x, local_y);
                    }
                }
            }
        }

        grid
    }

    /// Get the maximum absolute pressure in the grid.
    pub fn max_pressure(&self) -> f32 {
        self.tiles
            .values()
            .flat_map(|tile| {
                (0..self.tile_size).flat_map(move |y| {
                    (0..self.tile_size).map(move |x| tile.get_pressure(x, y).abs())
                })
            })
            .fold(0.0, f32::max)
    }

    /// Get total energy in the system.
    pub fn total_energy(&self) -> f32 {
        self.tiles
            .values()
            .flat_map(|tile| {
                (0..self.tile_size).flat_map(move |y| {
                    (0..self.tile_size).map(move |x| {
                        let p = tile.get_pressure(x, y);
                        p * p
                    })
                })
            })
            .sum()
    }

    /// Reset all tiles to initial state.
    pub fn reset(&mut self) {
        for tile in self.tiles.values_mut() {
            tile.reset();
        }
    }

    /// Get the number of cells.
    pub fn cell_count(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// Get the number of tile actors.
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Get the current backend.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Get K2K messaging statistics.
    pub fn k2k_stats(&self) -> ringkernel_core::k2k::K2KStats {
        self.broker.stats()
    }

    /// Update acoustic parameters.
    pub fn set_speed_of_sound(&mut self, speed: f32) {
        self.params.set_speed_of_sound(speed);
    }

    /// Update cell size.
    pub fn set_cell_size(&mut self, size: f32) {
        self.params.set_cell_size(size);
    }

    /// Resize the grid.
    pub async fn resize(&mut self, new_width: u32, new_height: u32) -> Result<()> {
        self.width = new_width;
        self.height = new_height;

        // Recalculate tile grid dimensions
        self.tiles_x = new_width.div_ceil(self.tile_size);
        self.tiles_y = new_height.div_ceil(self.tile_size);

        // Create new K2K broker
        self.broker = K2KBuilder::new()
            .max_pending_messages(self.tiles_x as usize * self.tiles_y as usize * 8)
            .build();

        // Recreate tile actors
        self.tiles.clear();
        for ty in 0..self.tiles_y {
            for tx in 0..self.tiles_x {
                let tile = TileActor::new(
                    tx,
                    ty,
                    self.tile_size,
                    self.tiles_x,
                    self.tiles_y,
                    &self.broker,
                    0.95,
                );
                self.tiles.insert((tx, ty), tile);
            }
        }

        tracing::info!(
            "Resized TileKernelGrid: {}x{} cells, {} tile actors",
            new_width,
            new_height,
            self.tiles.len()
        );

        Ok(())
    }

    /// Shutdown the grid.
    pub async fn shutdown(self) -> Result<()> {
        // K2K broker and tiles are dropped automatically
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tile_grid_creation() {
        let params = AcousticParams::new(343.0, 1.0);
        let grid = TileKernelGrid::new(64, 64, params, Backend::Cpu)
            .await
            .unwrap();

        assert_eq!(grid.width, 64);
        assert_eq!(grid.height, 64);
        assert_eq!(grid.tile_size, DEFAULT_TILE_SIZE);
        assert_eq!(grid.tiles_x, 4); // 64 / 16 = 4
        assert_eq!(grid.tiles_y, 4);
        assert_eq!(grid.tile_count(), 16); // 4 * 4 = 16 tiles
    }

    #[tokio::test]
    async fn test_tile_grid_impulse() {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = TileKernelGrid::with_tile_size(32, 32, params, Backend::Cpu, 16)
            .await
            .unwrap();

        grid.inject_impulse(16, 16, 1.0);

        let pressure_grid = grid.get_pressure_grid();
        assert_eq!(pressure_grid[16][16], 1.0);
    }

    #[tokio::test]
    async fn test_tile_grid_step() {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = TileKernelGrid::with_tile_size(32, 32, params, Backend::Cpu, 16)
            .await
            .unwrap();

        // Inject impulse at center
        grid.inject_impulse(16, 16, 1.0);

        // Run several steps
        for _ in 0..10 {
            grid.step().await.unwrap();
        }

        // Wave should have propagated
        let pressure_grid = grid.get_pressure_grid();
        let neighbor_pressure = pressure_grid[16][17];
        assert!(
            neighbor_pressure.abs() > 0.0,
            "Wave should have propagated to neighbor"
        );
    }

    #[tokio::test]
    async fn test_tile_grid_k2k_stats() {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = TileKernelGrid::with_tile_size(32, 32, params, Backend::Cpu, 16)
            .await
            .unwrap();

        grid.inject_impulse(16, 16, 1.0);
        grid.step().await.unwrap();

        let stats = grid.k2k_stats();
        assert!(
            stats.messages_delivered > 0,
            "K2K messages should have been exchanged"
        );
    }

    #[tokio::test]
    async fn test_tile_grid_reset() {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = TileKernelGrid::with_tile_size(32, 32, params, Backend::Cpu, 16)
            .await
            .unwrap();

        grid.inject_impulse(16, 16, 1.0);
        grid.step().await.unwrap();

        grid.reset();

        let pressure_grid = grid.get_pressure_grid();
        for row in pressure_grid {
            for p in row {
                assert_eq!(p, 0.0);
            }
        }
    }

    #[tokio::test]
    async fn test_tile_boundary_handling() {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = TileKernelGrid::with_tile_size(32, 32, params, Backend::Cpu, 16)
            .await
            .unwrap();

        // Inject near boundary
        grid.inject_impulse(1, 1, 1.0);

        // Run several steps
        for _ in 0..5 {
            grid.step().await.unwrap();
        }

        // Should not crash and energy should be bounded
        let energy = grid.total_energy();
        assert!(energy.is_finite(), "Energy should be finite");
    }

    #[cfg(feature = "wgpu")]
    #[tokio::test]
    #[ignore] // May not have GPU
    async fn test_tile_grid_gpu_step() {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = TileKernelGrid::with_tile_size(32, 32, params, Backend::Cpu, 16)
            .await
            .unwrap();

        // Enable GPU compute
        grid.enable_gpu_compute().await.unwrap();
        assert!(grid.is_gpu_enabled());

        // Inject impulse at center
        grid.inject_impulse(16, 16, 1.0);

        // Run several steps using GPU
        for _ in 0..10 {
            grid.step_gpu().await.unwrap();
        }

        // Wave should have propagated
        let pressure_grid = grid.get_pressure_grid();
        let neighbor_pressure = pressure_grid[16][17];
        assert!(
            neighbor_pressure.abs() > 0.0,
            "Wave should have propagated to neighbor (GPU compute)"
        );
    }

    #[cfg(feature = "wgpu")]
    #[tokio::test]
    #[ignore] // May not have GPU
    async fn test_tile_grid_gpu_matches_cpu() {
        let params = AcousticParams::new(343.0, 1.0);

        // Create two grids - one CPU, one GPU
        let mut grid_cpu = TileKernelGrid::with_tile_size(32, 32, params.clone(), Backend::Cpu, 16)
            .await
            .unwrap();
        let mut grid_gpu = TileKernelGrid::with_tile_size(32, 32, params, Backend::Cpu, 16)
            .await
            .unwrap();

        grid_gpu.enable_gpu_compute().await.unwrap();

        // Inject same impulse
        grid_cpu.inject_impulse(16, 16, 1.0);
        grid_gpu.inject_impulse(16, 16, 1.0);

        // Run same number of steps
        for _ in 0..5 {
            grid_cpu.step().await.unwrap();
            grid_gpu.step_gpu().await.unwrap();
        }

        // Results should match closely
        let cpu_grid = grid_cpu.get_pressure_grid();
        let gpu_grid = grid_gpu.get_pressure_grid();

        for y in 0..32 {
            for x in 0..32 {
                let diff = (cpu_grid[y][x] - gpu_grid[y][x]).abs();
                assert!(
                    diff < 1e-4,
                    "CPU/GPU mismatch at ({},{}): cpu={}, gpu={}, diff={}",
                    x,
                    y,
                    cpu_grid[y][x],
                    gpu_grid[y][x],
                    diff
                );
            }
        }
    }
}
