//! Kernel checkpointing for persistent state snapshot and restore.
//!
//! This module provides infrastructure for checkpointing persistent GPU kernels,
//! enabling fault tolerance, migration, and debugging capabilities.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    CheckpointableKernel                         │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
//! │  │ Control     │  │ Queue       │  │ Device Memory           │  │
//! │  │ Block       │  │ State       │  │ (pressure, halo, etc.)  │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        Checkpoint                               │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
//! │  │ Header      │  │ Metadata    │  │ Compressed Data Chunks  │  │
//! │  │ (magic,ver) │  │ (kernel_id) │  │ (control,queues,memory) │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   CheckpointStorage                             │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
//! │  │ File        │  │ Memory      │  │ Cloud (S3/GCS)          │  │
//! │  │ Backend     │  │ Backend     │  │ Backend                 │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_core::checkpoint::{Checkpoint, FileStorage, CheckpointableKernel};
//!
//! // Create checkpoint from running kernel
//! let checkpoint = kernel.create_checkpoint()?;
//!
//! // Save to file
//! let storage = FileStorage::new("/checkpoints");
//! storage.save(&checkpoint, "sim_step_1000")?;
//!
//! // Later: restore from checkpoint
//! let checkpoint = storage.load("sim_step_1000")?;
//! kernel.restore_from_checkpoint(&checkpoint)?;
//! ```

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::error::{Result, RingKernelError};
use crate::hlc::HlcTimestamp;

// ============================================================================
// Checkpoint Format Constants
// ============================================================================

/// Magic number for checkpoint files: "RKCKPT01" in ASCII.
pub const CHECKPOINT_MAGIC: u64 = 0x524B434B50543031;

/// Current checkpoint format version.
pub const CHECKPOINT_VERSION: u32 = 1;

/// Maximum supported checkpoint size (1 GB).
pub const MAX_CHECKPOINT_SIZE: usize = 1024 * 1024 * 1024;

/// Chunk types for checkpoint data sections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ChunkType {
    /// Control block state (256 bytes typically).
    ControlBlock = 1,
    /// H2K queue header and pending messages.
    H2KQueue = 2,
    /// K2H queue header and pending messages.
    K2HQueue = 3,
    /// HLC timestamp state.
    HlcState = 4,
    /// Device memory region (e.g., pressure field).
    DeviceMemory = 5,
    /// K2K routing table.
    K2KRouting = 6,
    /// Halo exchange buffers.
    HaloBuffers = 7,
    /// Telemetry statistics.
    Telemetry = 8,
    /// Custom application data.
    Custom = 100,
}

impl ChunkType {
    /// Convert from raw u32 value.
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            1 => Some(Self::ControlBlock),
            2 => Some(Self::H2KQueue),
            3 => Some(Self::K2HQueue),
            4 => Some(Self::HlcState),
            5 => Some(Self::DeviceMemory),
            6 => Some(Self::K2KRouting),
            7 => Some(Self::HaloBuffers),
            8 => Some(Self::Telemetry),
            100 => Some(Self::Custom),
            _ => None,
        }
    }
}

// ============================================================================
// Checkpoint Header
// ============================================================================

/// Checkpoint file header (64 bytes, fixed size).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CheckpointHeader {
    /// Magic number for format identification.
    pub magic: u64,
    /// Format version number.
    pub version: u32,
    /// Header size in bytes.
    pub header_size: u32,
    /// Total checkpoint size in bytes (including header).
    pub total_size: u64,
    /// Number of data chunks.
    pub chunk_count: u32,
    /// Compression algorithm (0 = none, 1 = lz4, 2 = zstd).
    pub compression: u32,
    /// CRC32 checksum of all data after header.
    pub checksum: u32,
    /// Flags (reserved for future use).
    pub flags: u32,
    /// Timestamp when checkpoint was created (UNIX epoch microseconds).
    pub created_at: u64,
    /// Reserved for alignment.
    pub _reserved: [u8; 8],
}

impl CheckpointHeader {
    /// Create a new checkpoint header.
    pub fn new(chunk_count: u32, total_size: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO);

        Self {
            magic: CHECKPOINT_MAGIC,
            version: CHECKPOINT_VERSION,
            header_size: std::mem::size_of::<Self>() as u32,
            total_size,
            chunk_count,
            compression: 0,
            checksum: 0,
            flags: 0,
            created_at: now.as_micros() as u64,
            _reserved: [0; 8],
        }
    }

    /// Validate the header.
    pub fn validate(&self) -> Result<()> {
        if self.magic != CHECKPOINT_MAGIC {
            return Err(RingKernelError::InvalidCheckpoint(
                "Invalid magic number".to_string(),
            ));
        }
        if self.version > CHECKPOINT_VERSION {
            return Err(RingKernelError::InvalidCheckpoint(format!(
                "Unsupported version: {} (max: {})",
                self.version, CHECKPOINT_VERSION
            )));
        }
        if self.total_size as usize > MAX_CHECKPOINT_SIZE {
            return Err(RingKernelError::InvalidCheckpoint(format!(
                "Checkpoint too large: {} bytes (max: {})",
                self.total_size, MAX_CHECKPOINT_SIZE
            )));
        }
        Ok(())
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut bytes = [0u8; 64];
        bytes[0..8].copy_from_slice(&self.magic.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.version.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.header_size.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.total_size.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.chunk_count.to_le_bytes());
        bytes[28..32].copy_from_slice(&self.compression.to_le_bytes());
        bytes[32..36].copy_from_slice(&self.checksum.to_le_bytes());
        bytes[36..40].copy_from_slice(&self.flags.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.created_at.to_le_bytes());
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8; 64]) -> Self {
        Self {
            magic: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            version: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            header_size: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            total_size: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            chunk_count: u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            compression: u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
            checksum: u32::from_le_bytes(bytes[32..36].try_into().unwrap()),
            flags: u32::from_le_bytes(bytes[36..40].try_into().unwrap()),
            created_at: u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
            _reserved: bytes[48..56].try_into().unwrap(),
        }
    }
}

// ============================================================================
// Chunk Header
// ============================================================================

/// Header for each data chunk (32 bytes).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ChunkHeader {
    /// Chunk type identifier.
    pub chunk_type: u32,
    /// Chunk flags (compression, etc.).
    pub flags: u32,
    /// Uncompressed data size.
    pub uncompressed_size: u64,
    /// Compressed data size (same as uncompressed if not compressed).
    pub compressed_size: u64,
    /// Chunk-specific identifier (e.g., memory region name hash).
    pub chunk_id: u64,
}

impl ChunkHeader {
    /// Create a new chunk header.
    pub fn new(chunk_type: ChunkType, data_size: usize) -> Self {
        Self {
            chunk_type: chunk_type as u32,
            flags: 0,
            uncompressed_size: data_size as u64,
            compressed_size: data_size as u64,
            chunk_id: 0,
        }
    }

    /// Set the chunk ID.
    pub fn with_id(mut self, id: u64) -> Self {
        self.chunk_id = id;
        self
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[0..4].copy_from_slice(&self.chunk_type.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.flags.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.uncompressed_size.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.compressed_size.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.chunk_id.to_le_bytes());
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        Self {
            chunk_type: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            flags: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            uncompressed_size: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            compressed_size: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            chunk_id: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
        }
    }
}

// ============================================================================
// Checkpoint Metadata
// ============================================================================

/// Kernel-specific metadata stored in checkpoint.
#[derive(Debug, Clone, Default)]
pub struct CheckpointMetadata {
    /// Unique kernel identifier.
    pub kernel_id: String,
    /// Kernel type (e.g., "fdtd_3d", "wave_sim").
    pub kernel_type: String,
    /// Current simulation step.
    pub current_step: u64,
    /// Grid dimensions.
    pub grid_size: (u32, u32, u32),
    /// Tile/block dimensions.
    pub tile_size: (u32, u32, u32),
    /// HLC timestamp at checkpoint time.
    pub hlc_timestamp: HlcTimestamp,
    /// Custom key-value metadata.
    pub custom: HashMap<String, String>,
}

impl CheckpointMetadata {
    /// Create new metadata for a kernel.
    pub fn new(kernel_id: impl Into<String>, kernel_type: impl Into<String>) -> Self {
        Self {
            kernel_id: kernel_id.into(),
            kernel_type: kernel_type.into(),
            ..Default::default()
        }
    }

    /// Set current step.
    pub fn with_step(mut self, step: u64) -> Self {
        self.current_step = step;
        self
    }

    /// Set grid size.
    pub fn with_grid_size(mut self, width: u32, height: u32, depth: u32) -> Self {
        self.grid_size = (width, height, depth);
        self
    }

    /// Set tile size.
    pub fn with_tile_size(mut self, x: u32, y: u32, z: u32) -> Self {
        self.tile_size = (x, y, z);
        self
    }

    /// Set HLC timestamp.
    pub fn with_hlc(mut self, hlc: HlcTimestamp) -> Self {
        self.hlc_timestamp = hlc;
        self
    }

    /// Add custom metadata.
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }

    /// Serialize metadata to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Kernel ID (length-prefixed string)
        let kernel_id_bytes = self.kernel_id.as_bytes();
        bytes.extend_from_slice(&(kernel_id_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(kernel_id_bytes);

        // Kernel type
        let kernel_type_bytes = self.kernel_type.as_bytes();
        bytes.extend_from_slice(&(kernel_type_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(kernel_type_bytes);

        // Current step
        bytes.extend_from_slice(&self.current_step.to_le_bytes());

        // Grid size
        bytes.extend_from_slice(&self.grid_size.0.to_le_bytes());
        bytes.extend_from_slice(&self.grid_size.1.to_le_bytes());
        bytes.extend_from_slice(&self.grid_size.2.to_le_bytes());

        // Tile size
        bytes.extend_from_slice(&self.tile_size.0.to_le_bytes());
        bytes.extend_from_slice(&self.tile_size.1.to_le_bytes());
        bytes.extend_from_slice(&self.tile_size.2.to_le_bytes());

        // HLC timestamp
        bytes.extend_from_slice(&self.hlc_timestamp.physical.to_le_bytes());
        bytes.extend_from_slice(&self.hlc_timestamp.logical.to_le_bytes());
        bytes.extend_from_slice(&self.hlc_timestamp.node_id.to_le_bytes());

        // Custom metadata count
        bytes.extend_from_slice(&(self.custom.len() as u32).to_le_bytes());

        // Custom key-value pairs
        for (key, value) in &self.custom {
            let key_bytes = key.as_bytes();
            bytes.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(key_bytes);

            let value_bytes = value.as_bytes();
            bytes.extend_from_slice(&(value_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(value_bytes);
        }

        bytes
    }

    /// Deserialize metadata from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut offset = 0;

        // Helper to read u32
        let read_u32 = |off: &mut usize| -> Result<u32> {
            if *off + 4 > bytes.len() {
                return Err(RingKernelError::InvalidCheckpoint(
                    "Unexpected end of metadata".to_string(),
                ));
            }
            let val = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap());
            *off += 4;
            Ok(val)
        };

        // Helper to read u64
        let read_u64 = |off: &mut usize| -> Result<u64> {
            if *off + 8 > bytes.len() {
                return Err(RingKernelError::InvalidCheckpoint(
                    "Unexpected end of metadata".to_string(),
                ));
            }
            let val = u64::from_le_bytes(bytes[*off..*off + 8].try_into().unwrap());
            *off += 8;
            Ok(val)
        };

        // Helper to read string
        let read_string = |off: &mut usize| -> Result<String> {
            let len = read_u32(off)? as usize;
            if *off + len > bytes.len() {
                return Err(RingKernelError::InvalidCheckpoint(
                    "Unexpected end of metadata".to_string(),
                ));
            }
            let s = String::from_utf8(bytes[*off..*off + len].to_vec())
                .map_err(|e| RingKernelError::InvalidCheckpoint(e.to_string()))?;
            *off += len;
            Ok(s)
        };

        let kernel_id = read_string(&mut offset)?;
        let kernel_type = read_string(&mut offset)?;
        let current_step = read_u64(&mut offset)?;

        let grid_size = (
            read_u32(&mut offset)?,
            read_u32(&mut offset)?,
            read_u32(&mut offset)?,
        );

        let tile_size = (
            read_u32(&mut offset)?,
            read_u32(&mut offset)?,
            read_u32(&mut offset)?,
        );

        let hlc_timestamp = HlcTimestamp {
            physical: read_u64(&mut offset)?,
            logical: read_u64(&mut offset)?,
            node_id: read_u64(&mut offset)?,
        };

        let custom_count = read_u32(&mut offset)? as usize;
        let mut custom = HashMap::new();

        for _ in 0..custom_count {
            let key = read_string(&mut offset)?;
            let value = read_string(&mut offset)?;
            custom.insert(key, value);
        }

        Ok(Self {
            kernel_id,
            kernel_type,
            current_step,
            grid_size,
            tile_size,
            hlc_timestamp,
            custom,
        })
    }
}

// ============================================================================
// Checkpoint Data Chunk
// ============================================================================

/// A single data chunk in a checkpoint.
#[derive(Debug, Clone)]
pub struct DataChunk {
    /// Chunk header.
    pub header: ChunkHeader,
    /// Chunk data (may be compressed).
    pub data: Vec<u8>,
}

impl DataChunk {
    /// Create a new data chunk.
    pub fn new(chunk_type: ChunkType, data: Vec<u8>) -> Self {
        Self {
            header: ChunkHeader::new(chunk_type, data.len()),
            data,
        }
    }

    /// Create a chunk with a custom ID.
    pub fn with_id(chunk_type: ChunkType, data: Vec<u8>, id: u64) -> Self {
        Self {
            header: ChunkHeader::new(chunk_type, data.len()).with_id(id),
            data,
        }
    }

    /// Get the chunk type.
    pub fn chunk_type(&self) -> Option<ChunkType> {
        ChunkType::from_u32(self.header.chunk_type)
    }
}

// ============================================================================
// Checkpoint
// ============================================================================

/// Complete checkpoint containing all kernel state.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Checkpoint header.
    pub header: CheckpointHeader,
    /// Kernel metadata.
    pub metadata: CheckpointMetadata,
    /// Data chunks.
    pub chunks: Vec<DataChunk>,
}

impl Checkpoint {
    /// Create a new checkpoint.
    pub fn new(metadata: CheckpointMetadata) -> Self {
        Self {
            header: CheckpointHeader::new(0, 0),
            metadata,
            chunks: Vec::new(),
        }
    }

    /// Add a data chunk.
    pub fn add_chunk(&mut self, chunk: DataChunk) {
        self.chunks.push(chunk);
    }

    /// Add control block data.
    pub fn add_control_block(&mut self, data: Vec<u8>) {
        self.add_chunk(DataChunk::new(ChunkType::ControlBlock, data));
    }

    /// Add H2K queue data.
    pub fn add_h2k_queue(&mut self, data: Vec<u8>) {
        self.add_chunk(DataChunk::new(ChunkType::H2KQueue, data));
    }

    /// Add K2H queue data.
    pub fn add_k2h_queue(&mut self, data: Vec<u8>) {
        self.add_chunk(DataChunk::new(ChunkType::K2HQueue, data));
    }

    /// Add HLC state.
    pub fn add_hlc_state(&mut self, data: Vec<u8>) {
        self.add_chunk(DataChunk::new(ChunkType::HlcState, data));
    }

    /// Add device memory region.
    pub fn add_device_memory(&mut self, name: &str, data: Vec<u8>) {
        // Use hash of name as chunk ID
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        let id = hasher.finish();

        self.add_chunk(DataChunk::with_id(ChunkType::DeviceMemory, data, id));
    }

    /// Get a chunk by type.
    pub fn get_chunk(&self, chunk_type: ChunkType) -> Option<&DataChunk> {
        self.chunks
            .iter()
            .find(|c| c.chunk_type() == Some(chunk_type))
    }

    /// Get all chunks of a type.
    pub fn get_chunks(&self, chunk_type: ChunkType) -> Vec<&DataChunk> {
        self.chunks
            .iter()
            .filter(|c| c.chunk_type() == Some(chunk_type))
            .collect()
    }

    /// Calculate total size in bytes.
    pub fn total_size(&self) -> usize {
        let header_size = std::mem::size_of::<CheckpointHeader>();
        let metadata_bytes = self.metadata.to_bytes();
        let metadata_size = 4 + metadata_bytes.len(); // length prefix + data

        let chunks_size: usize = self
            .chunks
            .iter()
            .map(|c| std::mem::size_of::<ChunkHeader>() + c.data.len())
            .sum();

        header_size + metadata_size + chunks_size
    }

    /// Serialize checkpoint to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Metadata as bytes
        let metadata_bytes = self.metadata.to_bytes();

        // Calculate total size
        let total_size = self.total_size();

        // Create header with correct values
        let header = CheckpointHeader::new(self.chunks.len() as u32, total_size as u64);

        // Write header
        bytes.extend_from_slice(&header.to_bytes());

        // Write metadata (length-prefixed)
        bytes.extend_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&metadata_bytes);

        // Write chunks
        for chunk in &self.chunks {
            bytes.extend_from_slice(&chunk.header.to_bytes());
            bytes.extend_from_slice(&chunk.data);
        }

        // Calculate checksum (simple CRC32 of data after header) and update in place
        let checksum = crc32_simple(&bytes[64..]);
        bytes[32..36].copy_from_slice(&checksum.to_le_bytes());

        bytes
    }

    /// Deserialize checkpoint from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 64 {
            return Err(RingKernelError::InvalidCheckpoint(
                "Checkpoint too small".to_string(),
            ));
        }

        // Read header
        let header = CheckpointHeader::from_bytes(bytes[0..64].try_into().unwrap());
        header.validate()?;

        // Verify checksum
        let expected_checksum = crc32_simple(&bytes[64..]);
        if header.checksum != expected_checksum {
            return Err(RingKernelError::InvalidCheckpoint(format!(
                "Checksum mismatch: expected {}, got {}",
                expected_checksum, header.checksum
            )));
        }

        let mut offset = 64;

        // Read metadata length
        if offset + 4 > bytes.len() {
            return Err(RingKernelError::InvalidCheckpoint(
                "Missing metadata length".to_string(),
            ));
        }
        let metadata_len =
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        // Read metadata
        if offset + metadata_len > bytes.len() {
            return Err(RingKernelError::InvalidCheckpoint(
                "Metadata truncated".to_string(),
            ));
        }
        let metadata = CheckpointMetadata::from_bytes(&bytes[offset..offset + metadata_len])?;
        offset += metadata_len;

        // Read chunks
        let mut chunks = Vec::new();
        for _ in 0..header.chunk_count {
            if offset + 32 > bytes.len() {
                return Err(RingKernelError::InvalidCheckpoint(
                    "Chunk header truncated".to_string(),
                ));
            }

            let chunk_header =
                ChunkHeader::from_bytes(bytes[offset..offset + 32].try_into().unwrap());
            offset += 32;

            let data_len = chunk_header.compressed_size as usize;
            if offset + data_len > bytes.len() {
                return Err(RingKernelError::InvalidCheckpoint(
                    "Chunk data truncated".to_string(),
                ));
            }

            let data = bytes[offset..offset + data_len].to_vec();
            offset += data_len;

            chunks.push(DataChunk {
                header: chunk_header,
                data,
            });
        }

        Ok(Self {
            header,
            metadata,
            chunks,
        })
    }
}

// ============================================================================
// Simple CRC32 Implementation
// ============================================================================

/// Simple CRC32 checksum (IEEE polynomial).
fn crc32_simple(data: &[u8]) -> u32 {
    const CRC32_TABLE: [u32; 256] = crc32_table();

    let mut crc = 0xFFFFFFFF;
    for byte in data {
        let index = ((crc ^ (*byte as u32)) & 0xFF) as usize;
        crc = CRC32_TABLE[index] ^ (crc >> 8);
    }
    !crc
}

/// Generate CRC32 lookup table at compile time.
const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

// ============================================================================
// CheckpointableKernel Trait
// ============================================================================

/// Trait for kernels that support checkpointing.
pub trait CheckpointableKernel {
    /// Create a checkpoint of the current kernel state.
    ///
    /// This should pause the kernel, serialize all state, and return a checkpoint.
    fn create_checkpoint(&self) -> Result<Checkpoint>;

    /// Restore kernel state from a checkpoint.
    ///
    /// This should pause the kernel, deserialize state, and resume.
    fn restore_from_checkpoint(&mut self, checkpoint: &Checkpoint) -> Result<()>;

    /// Get the kernel ID for checkpointing.
    fn checkpoint_kernel_id(&self) -> &str;

    /// Get the kernel type for checkpointing.
    fn checkpoint_kernel_type(&self) -> &str;

    /// Check if the kernel supports incremental checkpoints.
    fn supports_incremental(&self) -> bool {
        false
    }

    /// Create an incremental checkpoint (only changed state since last checkpoint).
    fn create_incremental_checkpoint(&self, _base: &Checkpoint) -> Result<Checkpoint> {
        // Default: fall back to full checkpoint
        self.create_checkpoint()
    }
}

// ============================================================================
// Checkpoint Storage Trait
// ============================================================================

/// Trait for checkpoint storage backends.
pub trait CheckpointStorage: Send + Sync {
    /// Save a checkpoint with the given name.
    fn save(&self, checkpoint: &Checkpoint, name: &str) -> Result<()>;

    /// Load a checkpoint by name.
    fn load(&self, name: &str) -> Result<Checkpoint>;

    /// List all available checkpoints.
    fn list(&self) -> Result<Vec<String>>;

    /// Delete a checkpoint.
    fn delete(&self, name: &str) -> Result<()>;

    /// Check if a checkpoint exists.
    fn exists(&self, name: &str) -> bool;
}

// ============================================================================
// File Storage Backend
// ============================================================================

/// File-based checkpoint storage.
pub struct FileStorage {
    /// Base directory for checkpoint files.
    base_path: PathBuf,
}

impl FileStorage {
    /// Create a new file storage backend.
    pub fn new(base_path: impl AsRef<Path>) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
        }
    }

    /// Get the full path for a checkpoint.
    fn checkpoint_path(&self, name: &str) -> PathBuf {
        self.base_path.join(format!("{}.rkcp", name))
    }
}

impl CheckpointStorage for FileStorage {
    fn save(&self, checkpoint: &Checkpoint, name: &str) -> Result<()> {
        // Ensure directory exists
        std::fs::create_dir_all(&self.base_path).map_err(|e| {
            RingKernelError::IoError(format!("Failed to create checkpoint directory: {}", e))
        })?;

        let path = self.checkpoint_path(name);
        let bytes = checkpoint.to_bytes();

        let mut file = std::fs::File::create(&path).map_err(|e| {
            RingKernelError::IoError(format!("Failed to create checkpoint file: {}", e))
        })?;

        file.write_all(&bytes)
            .map_err(|e| RingKernelError::IoError(format!("Failed to write checkpoint: {}", e)))?;

        Ok(())
    }

    fn load(&self, name: &str) -> Result<Checkpoint> {
        let path = self.checkpoint_path(name);

        let mut file = std::fs::File::open(&path).map_err(|e| {
            RingKernelError::IoError(format!("Failed to open checkpoint file: {}", e))
        })?;

        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| RingKernelError::IoError(format!("Failed to read checkpoint: {}", e)))?;

        Checkpoint::from_bytes(&bytes)
    }

    fn list(&self) -> Result<Vec<String>> {
        let entries = std::fs::read_dir(&self.base_path).map_err(|e| {
            RingKernelError::IoError(format!("Failed to read checkpoint directory: {}", e))
        })?;

        let mut names = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "rkcp").unwrap_or(false) {
                if let Some(stem) = path.file_stem() {
                    names.push(stem.to_string_lossy().to_string());
                }
            }
        }

        names.sort();
        Ok(names)
    }

    fn delete(&self, name: &str) -> Result<()> {
        let path = self.checkpoint_path(name);
        std::fs::remove_file(&path)
            .map_err(|e| RingKernelError::IoError(format!("Failed to delete checkpoint: {}", e)))?;
        Ok(())
    }

    fn exists(&self, name: &str) -> bool {
        self.checkpoint_path(name).exists()
    }
}

// ============================================================================
// Memory Storage Backend
// ============================================================================

/// In-memory checkpoint storage (for testing and fast operations).
pub struct MemoryStorage {
    /// Stored checkpoints.
    checkpoints: std::sync::RwLock<HashMap<String, Vec<u8>>>,
}

impl MemoryStorage {
    /// Create a new memory storage backend.
    pub fn new() -> Self {
        Self {
            checkpoints: std::sync::RwLock::new(HashMap::new()),
        }
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckpointStorage for MemoryStorage {
    fn save(&self, checkpoint: &Checkpoint, name: &str) -> Result<()> {
        let bytes = checkpoint.to_bytes();
        let mut checkpoints = self
            .checkpoints
            .write()
            .map_err(|_| RingKernelError::IoError("Failed to acquire write lock".to_string()))?;
        checkpoints.insert(name.to_string(), bytes);
        Ok(())
    }

    fn load(&self, name: &str) -> Result<Checkpoint> {
        let checkpoints = self
            .checkpoints
            .read()
            .map_err(|_| RingKernelError::IoError("Failed to acquire read lock".to_string()))?;

        let bytes = checkpoints
            .get(name)
            .ok_or_else(|| RingKernelError::IoError(format!("Checkpoint not found: {}", name)))?;

        Checkpoint::from_bytes(bytes)
    }

    fn list(&self) -> Result<Vec<String>> {
        let checkpoints = self
            .checkpoints
            .read()
            .map_err(|_| RingKernelError::IoError("Failed to acquire read lock".to_string()))?;

        let mut names: Vec<_> = checkpoints.keys().cloned().collect();
        names.sort();
        Ok(names)
    }

    fn delete(&self, name: &str) -> Result<()> {
        let mut checkpoints = self
            .checkpoints
            .write()
            .map_err(|_| RingKernelError::IoError("Failed to acquire write lock".to_string()))?;

        checkpoints
            .remove(name)
            .ok_or_else(|| RingKernelError::IoError(format!("Checkpoint not found: {}", name)))?;

        Ok(())
    }

    fn exists(&self, name: &str) -> bool {
        self.checkpoints
            .read()
            .map(|c| c.contains_key(name))
            .unwrap_or(false)
    }
}

// ============================================================================
// Checkpoint Builder
// ============================================================================

/// Builder for creating checkpoints incrementally.
pub struct CheckpointBuilder {
    metadata: CheckpointMetadata,
    chunks: Vec<DataChunk>,
}

impl CheckpointBuilder {
    /// Create a new checkpoint builder.
    pub fn new(kernel_id: impl Into<String>, kernel_type: impl Into<String>) -> Self {
        Self {
            metadata: CheckpointMetadata::new(kernel_id, kernel_type),
            chunks: Vec::new(),
        }
    }

    /// Set the current step.
    pub fn step(mut self, step: u64) -> Self {
        self.metadata.current_step = step;
        self
    }

    /// Set grid size.
    pub fn grid_size(mut self, width: u32, height: u32, depth: u32) -> Self {
        self.metadata.grid_size = (width, height, depth);
        self
    }

    /// Set tile size.
    pub fn tile_size(mut self, x: u32, y: u32, z: u32) -> Self {
        self.metadata.tile_size = (x, y, z);
        self
    }

    /// Set HLC timestamp.
    pub fn hlc(mut self, hlc: HlcTimestamp) -> Self {
        self.metadata.hlc_timestamp = hlc;
        self
    }

    /// Add custom metadata.
    pub fn custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.custom.insert(key.into(), value.into());
        self
    }

    /// Add control block data.
    pub fn control_block(mut self, data: Vec<u8>) -> Self {
        self.chunks
            .push(DataChunk::new(ChunkType::ControlBlock, data));
        self
    }

    /// Add H2K queue data.
    pub fn h2k_queue(mut self, data: Vec<u8>) -> Self {
        self.chunks.push(DataChunk::new(ChunkType::H2KQueue, data));
        self
    }

    /// Add K2H queue data.
    pub fn k2h_queue(mut self, data: Vec<u8>) -> Self {
        self.chunks.push(DataChunk::new(ChunkType::K2HQueue, data));
        self
    }

    /// Add device memory region.
    pub fn device_memory(mut self, name: &str, data: Vec<u8>) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        let id = hasher.finish();

        self.chunks
            .push(DataChunk::with_id(ChunkType::DeviceMemory, data, id));
        self
    }

    /// Add a custom chunk.
    pub fn chunk(mut self, chunk: DataChunk) -> Self {
        self.chunks.push(chunk);
        self
    }

    /// Build the checkpoint.
    pub fn build(self) -> Checkpoint {
        let mut checkpoint = Checkpoint::new(self.metadata);
        checkpoint.chunks = self.chunks;
        checkpoint
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_header_roundtrip() {
        let header = CheckpointHeader::new(5, 1024);
        let bytes = header.to_bytes();
        let restored = CheckpointHeader::from_bytes(&bytes);

        assert_eq!(restored.magic, CHECKPOINT_MAGIC);
        assert_eq!(restored.version, CHECKPOINT_VERSION);
        assert_eq!(restored.chunk_count, 5);
        assert_eq!(restored.total_size, 1024);
    }

    #[test]
    fn test_chunk_header_roundtrip() {
        let header = ChunkHeader::new(ChunkType::DeviceMemory, 4096).with_id(12345);
        let bytes = header.to_bytes();
        let restored = ChunkHeader::from_bytes(&bytes);

        assert_eq!(restored.chunk_type, ChunkType::DeviceMemory as u32);
        assert_eq!(restored.uncompressed_size, 4096);
        assert_eq!(restored.chunk_id, 12345);
    }

    #[test]
    fn test_metadata_roundtrip() {
        let metadata = CheckpointMetadata::new("kernel_1", "fdtd_3d")
            .with_step(1000)
            .with_grid_size(64, 64, 64)
            .with_tile_size(8, 8, 8)
            .with_custom("version", "1.0");

        let bytes = metadata.to_bytes();
        let restored = CheckpointMetadata::from_bytes(&bytes).unwrap();

        assert_eq!(restored.kernel_id, "kernel_1");
        assert_eq!(restored.kernel_type, "fdtd_3d");
        assert_eq!(restored.current_step, 1000);
        assert_eq!(restored.grid_size, (64, 64, 64));
        assert_eq!(restored.tile_size, (8, 8, 8));
        assert_eq!(restored.custom.get("version"), Some(&"1.0".to_string()));
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let checkpoint = CheckpointBuilder::new("test_kernel", "test_type")
            .step(500)
            .grid_size(32, 32, 32)
            .control_block(vec![1, 2, 3, 4])
            .device_memory("pressure_a", vec![5, 6, 7, 8, 9, 10])
            .build();

        let bytes = checkpoint.to_bytes();
        let restored = Checkpoint::from_bytes(&bytes).unwrap();

        assert_eq!(restored.metadata.kernel_id, "test_kernel");
        assert_eq!(restored.metadata.current_step, 500);
        assert_eq!(restored.chunks.len(), 2);

        let control = restored.get_chunk(ChunkType::ControlBlock).unwrap();
        assert_eq!(control.data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_memory_storage() {
        let storage = MemoryStorage::new();

        let checkpoint = CheckpointBuilder::new("mem_test", "test").step(100).build();

        storage.save(&checkpoint, "test_001").unwrap();
        assert!(storage.exists("test_001"));

        let loaded = storage.load("test_001").unwrap();
        assert_eq!(loaded.metadata.kernel_id, "mem_test");
        assert_eq!(loaded.metadata.current_step, 100);

        let list = storage.list().unwrap();
        assert_eq!(list, vec!["test_001"]);

        storage.delete("test_001").unwrap();
        assert!(!storage.exists("test_001"));
    }

    #[test]
    fn test_crc32() {
        // Known CRC32 values
        assert_eq!(crc32_simple(b""), 0);
        assert_eq!(crc32_simple(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn test_checkpoint_validation() {
        // Test invalid magic
        let mut bytes = vec![0u8; 64];
        bytes[0..8].copy_from_slice(&0u64.to_le_bytes()); // Wrong magic

        let header = CheckpointHeader::from_bytes(bytes[0..64].try_into().unwrap());
        assert!(header.validate().is_err());
    }

    #[test]
    fn test_large_checkpoint() {
        // Test with larger data
        let large_data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        let checkpoint = CheckpointBuilder::new("large_kernel", "stress_test")
            .step(999)
            .device_memory("field_a", large_data.clone())
            .device_memory("field_b", large_data.clone())
            .build();

        let bytes = checkpoint.to_bytes();
        let restored = Checkpoint::from_bytes(&bytes).unwrap();

        assert_eq!(restored.chunks.len(), 2);
        let chunks = restored.get_chunks(ChunkType::DeviceMemory);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].data.len(), 100_000);
    }
}
