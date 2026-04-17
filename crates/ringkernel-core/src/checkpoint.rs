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
        // All slice-to-array conversions below are infallible because the input
        // is a fixed-size [u8; 64] and all subslice ranges are compile-time constants.
        Self {
            magic: u64::from_le_bytes(bytes[0..8].try_into().expect("slice is exactly 8 bytes")),
            version: u32::from_le_bytes(bytes[8..12].try_into().expect("slice is exactly 4 bytes")),
            header_size: u32::from_le_bytes(
                bytes[12..16].try_into().expect("slice is exactly 4 bytes"),
            ),
            total_size: u64::from_le_bytes(
                bytes[16..24].try_into().expect("slice is exactly 8 bytes"),
            ),
            chunk_count: u32::from_le_bytes(
                bytes[24..28].try_into().expect("slice is exactly 4 bytes"),
            ),
            compression: u32::from_le_bytes(
                bytes[28..32].try_into().expect("slice is exactly 4 bytes"),
            ),
            checksum: u32::from_le_bytes(
                bytes[32..36].try_into().expect("slice is exactly 4 bytes"),
            ),
            flags: u32::from_le_bytes(bytes[36..40].try_into().expect("slice is exactly 4 bytes")),
            created_at: u64::from_le_bytes(
                bytes[40..48].try_into().expect("slice is exactly 8 bytes"),
            ),
            _reserved: bytes[48..56].try_into().expect("slice is exactly 8 bytes"),
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
        // All slice-to-array conversions below are infallible because the input
        // is a fixed-size [u8; 32] and all subslice ranges are compile-time constants.
        Self {
            chunk_type: u32::from_le_bytes(
                bytes[0..4].try_into().expect("slice is exactly 4 bytes"),
            ),
            flags: u32::from_le_bytes(bytes[4..8].try_into().expect("slice is exactly 4 bytes")),
            uncompressed_size: u64::from_le_bytes(
                bytes[8..16].try_into().expect("slice is exactly 8 bytes"),
            ),
            compressed_size: u64::from_le_bytes(
                bytes[16..24].try_into().expect("slice is exactly 8 bytes"),
            ),
            chunk_id: u64::from_le_bytes(
                bytes[24..32].try_into().expect("slice is exactly 8 bytes"),
            ),
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
            // Bounds checked above, so the 4-byte slice conversion is infallible
            let val = u32::from_le_bytes(
                bytes[*off..*off + 4]
                    .try_into()
                    .expect("bounds checked 4-byte slice"),
            );
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
            // Bounds checked above, so the 8-byte slice conversion is infallible
            let val = u64::from_le_bytes(
                bytes[*off..*off + 8]
                    .try_into()
                    .expect("bounds checked 8-byte slice"),
            );
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

        // Read header - slice is exactly 64 bytes (checked above)
        let header = CheckpointHeader::from_bytes(
            bytes[0..64]
                .try_into()
                .expect("input validated to be >= 64 bytes"),
        );
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
        // Bounds checked above, so the 4-byte slice conversion is infallible
        let metadata_len = u32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .expect("bounds checked 4-byte slice"),
        ) as usize;
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

            // Bounds checked above, so the 32-byte slice conversion is infallible
            let chunk_header = ChunkHeader::from_bytes(
                bytes[offset..offset + 32]
                    .try_into()
                    .expect("bounds checked 32-byte slice"),
            );
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
// Checkpoint Configuration
// ============================================================================

/// Configuration for periodic actor state checkpointing.
///
/// Controls how often snapshots are taken, how many are retained,
/// and where they are stored.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Interval between periodic snapshots.
    pub interval: Duration,
    /// Maximum number of checkpoints to retain per kernel.
    /// When exceeded, the oldest checkpoint is deleted.
    /// A value of 0 means unlimited retention.
    pub max_snapshots: usize,
    /// Storage path for file-based checkpoints.
    pub storage_path: PathBuf,
    /// Whether checkpointing is enabled.
    pub enabled: bool,
    /// Prefix for checkpoint names (e.g., "actor_0").
    pub name_prefix: String,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            max_snapshots: 5,
            storage_path: PathBuf::from("/tmp/ringkernel/checkpoints"),
            enabled: true,
            name_prefix: "checkpoint".to_string(),
        }
    }
}

impl CheckpointConfig {
    /// Create a new checkpoint config with the given interval.
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            ..Default::default()
        }
    }

    /// Set the maximum number of retained snapshots.
    pub fn with_max_snapshots(mut self, max: usize) -> Self {
        self.max_snapshots = max;
        self
    }

    /// Set the storage path.
    pub fn with_storage_path(mut self, path: impl AsRef<Path>) -> Self {
        self.storage_path = path.as_ref().to_path_buf();
        self
    }

    /// Set the name prefix for checkpoint files.
    pub fn with_name_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.name_prefix = prefix.into();
        self
    }

    /// Enable or disable checkpointing.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

// ============================================================================
// Snapshot Request / Response (backend-agnostic protocol)
// ============================================================================

/// A request to snapshot a specific actor's state.
///
/// This is the backend-agnostic representation of a snapshot command.
/// The CUDA backend maps this to an H2K `SnapshotActor` message.
#[derive(Debug, Clone)]
pub struct SnapshotRequest {
    /// Unique ID for this snapshot request (used for correlation).
    pub request_id: u64,
    /// Actor slot index to snapshot.
    pub actor_slot: u32,
    /// Offset in snapshot buffer (backend-specific).
    pub buffer_offset: u32,
    /// Timestamp when this request was issued.
    pub issued_at: SystemTime,
}

/// A completed snapshot response from the device.
///
/// The CUDA backend maps this from a K2H `SnapshotComplete` message.
#[derive(Debug, Clone)]
pub struct SnapshotResponse {
    /// The request ID this responds to.
    pub request_id: u64,
    /// Actor slot that was snapshotted.
    pub actor_slot: u32,
    /// Whether the snapshot succeeded.
    pub success: bool,
    /// Snapshot data (copied from device/mapped memory by the backend).
    pub data: Vec<u8>,
    /// Simulation step at snapshot time.
    pub step: u64,
}

// ============================================================================
// Checkpoint Manager
// ============================================================================

/// Tracks the state of a pending snapshot request.
#[derive(Debug, Clone)]
struct PendingSnapshot {
    /// The original request.
    request: SnapshotRequest,
    /// Kernel ID this snapshot belongs to.
    kernel_id: String,
    /// Kernel type for metadata.
    kernel_type: String,
}

/// Manages periodic checkpointing for persistent GPU actors.
///
/// The `CheckpointManager` orchestrates the checkpoint lifecycle:
///
/// 1. Periodically determines when a snapshot is due
/// 2. Issues `SnapshotRequest`s (caller sends as H2K commands)
/// 3. Processes `SnapshotResponse`s (caller feeds from K2H responses)
/// 4. Persists completed checkpoints to storage
/// 5. Enforces retention policy (deletes old checkpoints)
///
/// # Usage
///
/// ```ignore
/// use ringkernel_core::checkpoint::{CheckpointConfig, CheckpointManager};
/// use std::time::Duration;
///
/// let config = CheckpointConfig::new(Duration::from_secs(10))
///     .with_max_snapshots(3)
///     .with_storage_path("/tmp/checkpoints");
///
/// let mut manager = CheckpointManager::new(config);
/// manager.register_actor(0, "wave_sim_0", "fdtd_3d");
///
/// // In your poll loop:
/// for request in manager.poll_due_snapshots() {
///     // Send as H2K SnapshotActor command
///     h2k_queue.send(H2KMessage::snapshot_actor(
///         request.request_id,
///         request.actor_slot,
///         request.buffer_offset,
///     ));
/// }
///
/// // When K2H SnapshotComplete arrives:
/// manager.complete_snapshot(SnapshotResponse { ... })?;
/// ```
pub struct CheckpointManager {
    /// Configuration.
    config: CheckpointConfig,
    /// Storage backend.
    storage: Box<dyn CheckpointStorage>,
    /// Registered actors: slot -> (kernel_id, kernel_type).
    actors: HashMap<u32, (String, String)>,
    /// Last snapshot time per actor slot.
    last_snapshot: HashMap<u32, std::time::Instant>,
    /// Pending snapshot requests awaiting completion.
    pending: HashMap<u64, PendingSnapshot>,
    /// Next request ID counter.
    next_request_id: u64,
    /// Ordered list of checkpoint names per actor (oldest first) for retention.
    checkpoint_history: HashMap<u32, Vec<String>>,
    /// Total snapshots completed (lifetime counter).
    total_completed: u64,
    /// Total snapshots failed (lifetime counter).
    total_failed: u64,
}

impl CheckpointManager {
    /// Create a new checkpoint manager with file storage at the configured path.
    pub fn new(config: CheckpointConfig) -> Self {
        let storage = Box::new(FileStorage::new(&config.storage_path));
        Self {
            config,
            storage,
            actors: HashMap::new(),
            last_snapshot: HashMap::new(),
            pending: HashMap::new(),
            next_request_id: 1,
            checkpoint_history: HashMap::new(),
            total_completed: 0,
            total_failed: 0,
        }
    }

    /// Create a checkpoint manager with a custom storage backend.
    pub fn with_storage(config: CheckpointConfig, storage: Box<dyn CheckpointStorage>) -> Self {
        Self {
            config,
            storage,
            actors: HashMap::new(),
            last_snapshot: HashMap::new(),
            pending: HashMap::new(),
            next_request_id: 1,
            checkpoint_history: HashMap::new(),
            total_completed: 0,
            total_failed: 0,
        }
    }

    /// Register an actor for periodic checkpointing.
    pub fn register_actor(
        &mut self,
        actor_slot: u32,
        kernel_id: impl Into<String>,
        kernel_type: impl Into<String>,
    ) {
        self.actors
            .insert(actor_slot, (kernel_id.into(), kernel_type.into()));
    }

    /// Unregister an actor from checkpointing.
    pub fn unregister_actor(&mut self, actor_slot: u32) {
        self.actors.remove(&actor_slot);
        self.last_snapshot.remove(&actor_slot);
    }

    /// Check if checkpointing is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the checkpoint configuration.
    pub fn config(&self) -> &CheckpointConfig {
        &self.config
    }

    /// Get the number of pending snapshot requests.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get total completed snapshots.
    pub fn total_completed(&self) -> u64 {
        self.total_completed
    }

    /// Get total failed snapshots.
    pub fn total_failed(&self) -> u64 {
        self.total_failed
    }

    /// Poll for actors that are due for a snapshot.
    ///
    /// Returns a list of `SnapshotRequest`s that should be sent to the device
    /// as H2K `SnapshotActor` commands.
    ///
    /// Each actor is only requested once per interval, and only if no prior
    /// request for that actor is still pending.
    pub fn poll_due_snapshots(&mut self) -> Vec<SnapshotRequest> {
        if !self.config.enabled {
            return Vec::new();
        }

        let now = std::time::Instant::now();
        let interval = self.config.interval;
        let mut requests = Vec::new();

        // Collect actor slots that are due (to avoid borrow conflict)
        let due_slots: Vec<u32> = self
            .actors
            .keys()
            .filter(|slot| {
                // Skip if there's already a pending request for this actor
                let has_pending = self
                    .pending
                    .values()
                    .any(|p| p.request.actor_slot == **slot);
                if has_pending {
                    return false;
                }

                // Check if interval has elapsed since last snapshot
                match self.last_snapshot.get(slot) {
                    Some(last) => now.duration_since(*last) >= interval,
                    None => true, // Never snapshotted, due immediately
                }
            })
            .copied()
            .collect();

        for slot in due_slots {
            let request_id = self.next_request_id;
            self.next_request_id += 1;

            let request = SnapshotRequest {
                request_id,
                actor_slot: slot,
                buffer_offset: 0, // Backend fills in actual offset
                issued_at: SystemTime::now(),
            };

            if let Some((kernel_id, kernel_type)) = self.actors.get(&slot) {
                self.pending.insert(
                    request_id,
                    PendingSnapshot {
                        request: request.clone(),
                        kernel_id: kernel_id.clone(),
                        kernel_type: kernel_type.clone(),
                    },
                );
            }

            requests.push(request);
        }

        requests
    }

    /// Process a completed snapshot response from the device.
    ///
    /// If the snapshot succeeded, the data is persisted to storage and
    /// the retention policy is enforced.
    ///
    /// Returns the checkpoint name on success.
    pub fn complete_snapshot(&mut self, response: SnapshotResponse) -> Result<Option<String>> {
        let pending = match self.pending.remove(&response.request_id) {
            Some(p) => p,
            None => {
                // Unknown request ID -- may have been cancelled or already completed
                return Ok(None);
            }
        };

        // Record the snapshot time regardless of success
        self.last_snapshot
            .insert(pending.request.actor_slot, std::time::Instant::now());

        if !response.success {
            self.total_failed += 1;
            return Err(RingKernelError::InvalidCheckpoint(format!(
                "Snapshot failed for actor slot {}",
                response.actor_slot
            )));
        }

        // Build a checkpoint from the snapshot data
        let checkpoint = CheckpointBuilder::new(&pending.kernel_id, &pending.kernel_type)
            .step(response.step)
            .custom("actor_slot", pending.request.actor_slot.to_string())
            .custom(
                "snapshot_request_id",
                pending.request.request_id.to_string(),
            )
            .device_memory("actor_state", response.data)
            .build();

        // Generate checkpoint name
        let name = format!(
            "{}_{}_step_{}",
            self.config.name_prefix, pending.request.actor_slot, response.step
        );

        // Persist to storage
        self.storage.save(&checkpoint, &name)?;
        self.total_completed += 1;

        // Track in history for retention
        let history = self
            .checkpoint_history
            .entry(pending.request.actor_slot)
            .or_default();
        history.push(name.clone());

        // Enforce retention policy
        if self.config.max_snapshots > 0 {
            while history.len() > self.config.max_snapshots {
                let oldest = history.remove(0);
                if let Err(e) = self.storage.delete(&oldest) {
                    tracing::warn!(
                        checkpoint = oldest,
                        error = %e,
                        "Failed to delete old checkpoint during retention cleanup"
                    );
                }
            }
        }

        Ok(Some(name))
    }

    /// Manually request a snapshot for a specific actor, bypassing the interval timer.
    ///
    /// This is useful for on-demand snapshots (e.g., before a risky operation)
    /// or in tests. Returns `None` if the actor is not registered.
    pub fn request_snapshot(&mut self, actor_slot: u32) -> Option<SnapshotRequest> {
        let (kernel_id, kernel_type) = self.actors.get(&actor_slot)?.clone();

        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let request = SnapshotRequest {
            request_id,
            actor_slot,
            buffer_offset: 0,
            issued_at: SystemTime::now(),
        };

        self.pending.insert(
            request_id,
            PendingSnapshot {
                request: request.clone(),
                kernel_id,
                kernel_type,
            },
        );

        Some(request)
    }

    /// Cancel a pending snapshot request.
    ///
    /// Returns true if the request was found and cancelled.
    pub fn cancel_pending(&mut self, request_id: u64) -> bool {
        self.pending.remove(&request_id).is_some()
    }

    /// Cancel all pending snapshot requests.
    pub fn cancel_all_pending(&mut self) {
        self.pending.clear();
    }

    /// Load the most recent checkpoint for an actor.
    pub fn load_latest(&self, actor_slot: u32) -> Result<Option<Checkpoint>> {
        if let Some(history) = self.checkpoint_history.get(&actor_slot) {
            if let Some(latest_name) = history.last() {
                return self.storage.load(latest_name).map(Some);
            }
        }

        // Fallback: scan storage for checkpoints matching this actor's prefix
        let prefix = format!("{}_{}_", self.config.name_prefix, actor_slot);
        let all = self.storage.list()?;
        let matching: Vec<_> = all.iter().filter(|n| n.starts_with(&prefix)).collect();

        if let Some(latest) = matching.last() {
            return self.storage.load(latest).map(Some);
        }

        Ok(None)
    }

    /// List all checkpoint names for an actor.
    pub fn list_checkpoints(&self, actor_slot: u32) -> Result<Vec<String>> {
        let prefix = format!("{}_{}_", self.config.name_prefix, actor_slot);
        let all = self.storage.list()?;
        Ok(all.into_iter().filter(|n| n.starts_with(&prefix)).collect())
    }

    /// Get a reference to the storage backend.
    pub fn storage(&self) -> &dyn CheckpointStorage {
        &*self.storage
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
        let mut bytes = [0u8; 64];
        bytes[0..8].copy_from_slice(&0u64.to_le_bytes()); // Wrong magic

        let header = CheckpointHeader::from_bytes(&bytes);
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

    // ========================================================================
    // CheckpointConfig tests
    // ========================================================================

    #[test]
    fn test_checkpoint_config_defaults() {
        let config = CheckpointConfig::default();
        assert_eq!(config.interval, Duration::from_secs(30));
        assert_eq!(config.max_snapshots, 5);
        assert!(config.enabled);
        assert_eq!(config.name_prefix, "checkpoint");
    }

    #[test]
    fn test_checkpoint_config_builder() {
        let config = CheckpointConfig::new(Duration::from_secs(10))
            .with_max_snapshots(3)
            .with_storage_path("/var/checkpoints")
            .with_name_prefix("actor")
            .with_enabled(false);

        assert_eq!(config.interval, Duration::from_secs(10));
        assert_eq!(config.max_snapshots, 3);
        assert_eq!(config.storage_path, PathBuf::from("/var/checkpoints"));
        assert_eq!(config.name_prefix, "actor");
        assert!(!config.enabled);
    }

    // ========================================================================
    // CheckpointManager tests
    // ========================================================================

    #[test]
    fn test_manager_disabled() {
        let config = CheckpointConfig::new(Duration::from_millis(1)).with_enabled(false);
        let mut manager = CheckpointManager::with_storage(config, Box::new(MemoryStorage::new()));
        manager.register_actor(0, "kernel_0", "test");

        // Should return no requests when disabled
        let requests = manager.poll_due_snapshots();
        assert!(requests.is_empty());
    }

    #[test]
    fn test_manager_register_and_poll() {
        let config = CheckpointConfig::new(Duration::from_millis(1));
        let mut manager = CheckpointManager::with_storage(config, Box::new(MemoryStorage::new()));
        manager.register_actor(0, "sim_0", "fdtd_3d");
        manager.register_actor(1, "sim_1", "fdtd_3d");

        // First poll: both actors are due immediately (never snapshotted)
        let requests = manager.poll_due_snapshots();
        assert_eq!(requests.len(), 2);

        // Verify request fields
        let slots: Vec<u32> = requests.iter().map(|r| r.actor_slot).collect();
        assert!(slots.contains(&0));
        assert!(slots.contains(&1));

        // Request IDs should be unique
        assert_ne!(requests[0].request_id, requests[1].request_id);
    }

    #[test]
    fn test_manager_no_duplicate_pending() {
        let config = CheckpointConfig::new(Duration::from_millis(1));
        let mut manager = CheckpointManager::with_storage(config, Box::new(MemoryStorage::new()));
        manager.register_actor(0, "sim_0", "fdtd_3d");

        // First poll: actor is due
        let requests = manager.poll_due_snapshots();
        assert_eq!(requests.len(), 1);

        // Second poll: actor already has a pending request
        let requests2 = manager.poll_due_snapshots();
        assert!(requests2.is_empty());
    }

    #[test]
    fn test_manager_complete_snapshot() {
        let config = CheckpointConfig::new(Duration::from_secs(3600)).with_name_prefix("test");
        let mut manager = CheckpointManager::with_storage(config, Box::new(MemoryStorage::new()));
        manager.register_actor(0, "sim_0", "fdtd_3d");

        // Generate a snapshot request
        let requests = manager.poll_due_snapshots();
        assert_eq!(requests.len(), 1);
        let req = &requests[0];

        // Complete the snapshot
        let response = SnapshotResponse {
            request_id: req.request_id,
            actor_slot: 0,
            success: true,
            data: vec![1, 2, 3, 4, 5],
            step: 1000,
        };

        let name = manager.complete_snapshot(response).unwrap();
        assert!(name.is_some());
        let name = name.unwrap();
        assert_eq!(name, "test_0_step_1000");

        // Verify checkpoint was persisted
        assert!(manager.storage().exists(&name));

        // Load and verify
        let loaded = manager.storage().load(&name).unwrap();
        assert_eq!(loaded.metadata.kernel_id, "sim_0");
        assert_eq!(loaded.metadata.kernel_type, "fdtd_3d");
        assert_eq!(loaded.metadata.current_step, 1000);

        assert_eq!(manager.total_completed(), 1);
        assert_eq!(manager.total_failed(), 0);
    }

    #[test]
    fn test_manager_failed_snapshot() {
        let config = CheckpointConfig::new(Duration::from_secs(3600));
        let mut manager = CheckpointManager::with_storage(config, Box::new(MemoryStorage::new()));
        manager.register_actor(0, "sim_0", "fdtd_3d");

        let requests = manager.poll_due_snapshots();
        let req = &requests[0];

        let response = SnapshotResponse {
            request_id: req.request_id,
            actor_slot: 0,
            success: false,
            data: Vec::new(),
            step: 500,
        };

        let result = manager.complete_snapshot(response);
        assert!(result.is_err());
        assert_eq!(manager.total_failed(), 1);
        assert_eq!(manager.total_completed(), 0);
    }

    #[test]
    fn test_manager_retention_policy() {
        let config = CheckpointConfig::new(Duration::from_secs(3600))
            .with_max_snapshots(2)
            .with_name_prefix("ret");
        let storage = Box::new(MemoryStorage::new());
        let mut manager = CheckpointManager::with_storage(config, storage);
        manager.register_actor(0, "sim_0", "test");

        // Create 3 snapshots using manual requests (bypasses interval timer)
        for step in [100u64, 200, 300] {
            let req = manager.request_snapshot(0).unwrap();

            let response = SnapshotResponse {
                request_id: req.request_id,
                actor_slot: 0,
                success: true,
                data: vec![step as u8],
                step,
            };
            manager.complete_snapshot(response).unwrap();
        }

        // Oldest checkpoint (step 100) should have been deleted
        assert!(!manager.storage().exists("ret_0_step_100"));
        // Steps 200 and 300 should remain
        assert!(manager.storage().exists("ret_0_step_200"));
        assert!(manager.storage().exists("ret_0_step_300"));

        assert_eq!(manager.total_completed(), 3);
    }

    #[test]
    fn test_manager_unknown_response() {
        let config = CheckpointConfig::new(Duration::from_secs(3600));
        let mut manager = CheckpointManager::with_storage(config, Box::new(MemoryStorage::new()));

        // Response with unknown request_id
        let response = SnapshotResponse {
            request_id: 9999,
            actor_slot: 0,
            success: true,
            data: vec![1, 2, 3],
            step: 100,
        };

        let result = manager.complete_snapshot(response).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_manager_cancel_pending() {
        let config = CheckpointConfig::new(Duration::from_millis(1));
        let mut manager = CheckpointManager::with_storage(config, Box::new(MemoryStorage::new()));
        manager.register_actor(0, "sim_0", "test");

        let requests = manager.poll_due_snapshots();
        assert_eq!(manager.pending_count(), 1);

        let cancelled = manager.cancel_pending(requests[0].request_id);
        assert!(cancelled);
        assert_eq!(manager.pending_count(), 0);
    }

    #[test]
    fn test_manager_cancel_all_pending() {
        let config = CheckpointConfig::new(Duration::from_millis(1));
        let mut manager = CheckpointManager::with_storage(config, Box::new(MemoryStorage::new()));
        manager.register_actor(0, "sim_0", "test");
        manager.register_actor(1, "sim_1", "test");

        let _requests = manager.poll_due_snapshots();
        assert_eq!(manager.pending_count(), 2);

        manager.cancel_all_pending();
        assert_eq!(manager.pending_count(), 0);
    }

    #[test]
    fn test_manager_load_latest() {
        let config = CheckpointConfig::new(Duration::from_secs(3600)).with_name_prefix("lat");
        let mut manager = CheckpointManager::with_storage(config, Box::new(MemoryStorage::new()));
        manager.register_actor(0, "sim_0", "test");

        // No checkpoints yet
        let latest = manager.load_latest(0).unwrap();
        assert!(latest.is_none());

        // Create two checkpoints using manual requests
        for step in [100u64, 200] {
            let req = manager.request_snapshot(0).unwrap();
            let response = SnapshotResponse {
                request_id: req.request_id,
                actor_slot: 0,
                success: true,
                data: vec![step as u8],
                step,
            };
            manager.complete_snapshot(response).unwrap();
        }

        // Latest should be step 200
        let latest = manager.load_latest(0).unwrap().unwrap();
        assert_eq!(latest.metadata.current_step, 200);
    }

    #[test]
    fn test_manager_list_checkpoints() {
        let config = CheckpointConfig::new(Duration::from_secs(3600))
            .with_max_snapshots(10)
            .with_name_prefix("list");
        let mut manager = CheckpointManager::with_storage(config, Box::new(MemoryStorage::new()));
        manager.register_actor(0, "sim_0", "test");
        manager.register_actor(1, "sim_1", "test");

        // Create checkpoints for both actors using manual requests
        for step in [100u64, 200] {
            for actor_slot in [0u32, 1] {
                let req = manager.request_snapshot(actor_slot).unwrap();
                let response = SnapshotResponse {
                    request_id: req.request_id,
                    actor_slot,
                    success: true,
                    data: vec![step as u8],
                    step,
                };
                manager.complete_snapshot(response).unwrap();
            }
        }

        let actor0_checkpoints = manager.list_checkpoints(0).unwrap();
        let actor1_checkpoints = manager.list_checkpoints(1).unwrap();

        assert_eq!(actor0_checkpoints.len(), 2);
        assert_eq!(actor1_checkpoints.len(), 2);

        // Actor 0's checkpoints should only contain its own
        for name in &actor0_checkpoints {
            assert!(name.starts_with("list_0_"));
        }
        // Actor 1's checkpoints should only contain its own
        for name in &actor1_checkpoints {
            assert!(name.starts_with("list_1_"));
        }
    }

    #[test]
    fn test_manager_unregister_actor() {
        let config = CheckpointConfig::new(Duration::from_millis(1));
        let mut manager = CheckpointManager::with_storage(config, Box::new(MemoryStorage::new()));
        manager.register_actor(0, "sim_0", "test");

        let requests = manager.poll_due_snapshots();
        assert_eq!(requests.len(), 1);

        // Unregister and poll again
        manager.unregister_actor(0);
        // Complete the pending request first
        manager.cancel_all_pending();

        let requests = manager.poll_due_snapshots();
        assert!(requests.is_empty());
    }

    #[test]
    fn test_snapshot_request_response_roundtrip() {
        // Test that the request/response types work correctly
        let request = SnapshotRequest {
            request_id: 42,
            actor_slot: 7,
            buffer_offset: 4096,
            issued_at: SystemTime::now(),
        };

        assert_eq!(request.request_id, 42);
        assert_eq!(request.actor_slot, 7);
        assert_eq!(request.buffer_offset, 4096);

        let response = SnapshotResponse {
            request_id: 42,
            actor_slot: 7,
            success: true,
            data: vec![0xDE, 0xAD, 0xBE, 0xEF],
            step: 5000,
        };

        assert_eq!(response.request_id, request.request_id);
        assert_eq!(response.actor_slot, request.actor_slot);
        assert!(response.success);
        assert_eq!(response.step, 5000);
    }

    #[test]
    fn test_manager_interval_respected() {
        // Use a long interval so second poll returns nothing
        let config = CheckpointConfig::new(Duration::from_secs(3600));
        let mut manager = CheckpointManager::with_storage(config, Box::new(MemoryStorage::new()));
        manager.register_actor(0, "sim_0", "test");

        // First poll: due immediately
        let requests = manager.poll_due_snapshots();
        assert_eq!(requests.len(), 1);

        // Complete the snapshot
        let response = SnapshotResponse {
            request_id: requests[0].request_id,
            actor_slot: 0,
            success: true,
            data: vec![1],
            step: 100,
        };
        manager.complete_snapshot(response).unwrap();

        // Second poll: not due yet (interval is 1 hour)
        let requests = manager.poll_due_snapshots();
        assert!(requests.is_empty());
    }

    #[test]
    fn test_file_storage_roundtrip() {
        // Use a temp directory for file storage
        let tmp_dir = std::env::temp_dir().join("ringkernel_checkpoint_test");

        let config = CheckpointConfig::new(Duration::from_millis(1))
            .with_storage_path(&tmp_dir)
            .with_name_prefix("file_test");
        let mut manager = CheckpointManager::new(config);
        manager.register_actor(0, "file_kernel", "test_type");

        let requests = manager.poll_due_snapshots();
        assert_eq!(requests.len(), 1);

        let response = SnapshotResponse {
            request_id: requests[0].request_id,
            actor_slot: 0,
            success: true,
            data: vec![10, 20, 30, 40, 50],
            step: 42,
        };

        let name = manager.complete_snapshot(response).unwrap().unwrap();

        // Verify the file exists on disk
        let file_path = tmp_dir.join(format!("{}.rkcp", name));
        assert!(file_path.exists());

        // Load it back
        let loaded = manager.load_latest(0).unwrap().unwrap();
        assert_eq!(loaded.metadata.kernel_id, "file_kernel");
        assert_eq!(loaded.metadata.current_step, 42);

        // Clean up
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}
