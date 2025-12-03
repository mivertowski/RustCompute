//! GPU intrinsic mapping for CUDA code generation.
//!
//! This module provides mappings from high-level Rust operations to
//! CUDA intrinsics and built-in functions.

use std::collections::HashMap;

/// GPU intrinsic operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuIntrinsic {
    /// Thread synchronization.
    SyncThreads,
    /// Thread fence (memory ordering).
    ThreadFence,
    ThreadFenceBlock,
    ThreadFenceSystem,

    /// Atomic operations.
    AtomicAdd,
    AtomicSub,
    AtomicMin,
    AtomicMax,
    AtomicExch,
    AtomicCas,

    /// Math functions.
    Sqrt,
    Rsqrt,
    Abs,
    Fabs,
    Floor,
    Ceil,
    Round,
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Pow,
    Fma,
    Min,
    Max,

    /// Warp-level operations.
    WarpShfl,
    WarpShflUp,
    WarpShflDown,
    WarpShflXor,
    WarpActiveMask,
    WarpBallot,
    WarpAll,
    WarpAny,

    /// CUDA thread/block indices.
    ThreadIdxX,
    ThreadIdxY,
    ThreadIdxZ,
    BlockIdxX,
    BlockIdxY,
    BlockIdxZ,
    BlockDimX,
    BlockDimY,
    BlockDimZ,
    GridDimX,
    GridDimY,
    GridDimZ,
}

impl GpuIntrinsic {
    /// Convert to CUDA function/intrinsic name.
    pub fn to_cuda_string(&self) -> &'static str {
        match self {
            GpuIntrinsic::SyncThreads => "__syncthreads()",
            GpuIntrinsic::ThreadFence => "__threadfence()",
            GpuIntrinsic::ThreadFenceBlock => "__threadfence_block()",
            GpuIntrinsic::ThreadFenceSystem => "__threadfence_system()",
            GpuIntrinsic::AtomicAdd => "atomicAdd",
            GpuIntrinsic::AtomicSub => "atomicSub",
            GpuIntrinsic::AtomicMin => "atomicMin",
            GpuIntrinsic::AtomicMax => "atomicMax",
            GpuIntrinsic::AtomicExch => "atomicExch",
            GpuIntrinsic::AtomicCas => "atomicCAS",
            GpuIntrinsic::Sqrt => "sqrtf",
            GpuIntrinsic::Rsqrt => "rsqrtf",
            GpuIntrinsic::Abs => "abs",
            GpuIntrinsic::Fabs => "fabsf",
            GpuIntrinsic::Floor => "floorf",
            GpuIntrinsic::Ceil => "ceilf",
            GpuIntrinsic::Round => "roundf",
            GpuIntrinsic::Sin => "sinf",
            GpuIntrinsic::Cos => "cosf",
            GpuIntrinsic::Tan => "tanf",
            GpuIntrinsic::Exp => "expf",
            GpuIntrinsic::Log => "logf",
            GpuIntrinsic::Pow => "powf",
            GpuIntrinsic::Fma => "fmaf",
            GpuIntrinsic::Min => "fminf",
            GpuIntrinsic::Max => "fmaxf",
            GpuIntrinsic::WarpShfl => "__shfl_sync",
            GpuIntrinsic::WarpShflUp => "__shfl_up_sync",
            GpuIntrinsic::WarpShflDown => "__shfl_down_sync",
            GpuIntrinsic::WarpShflXor => "__shfl_xor_sync",
            GpuIntrinsic::WarpActiveMask => "__activemask()",
            GpuIntrinsic::WarpBallot => "__ballot_sync",
            GpuIntrinsic::WarpAll => "__all_sync",
            GpuIntrinsic::WarpAny => "__any_sync",
            GpuIntrinsic::ThreadIdxX => "threadIdx.x",
            GpuIntrinsic::ThreadIdxY => "threadIdx.y",
            GpuIntrinsic::ThreadIdxZ => "threadIdx.z",
            GpuIntrinsic::BlockIdxX => "blockIdx.x",
            GpuIntrinsic::BlockIdxY => "blockIdx.y",
            GpuIntrinsic::BlockIdxZ => "blockIdx.z",
            GpuIntrinsic::BlockDimX => "blockDim.x",
            GpuIntrinsic::BlockDimY => "blockDim.y",
            GpuIntrinsic::BlockDimZ => "blockDim.z",
            GpuIntrinsic::GridDimX => "gridDim.x",
            GpuIntrinsic::GridDimY => "gridDim.y",
            GpuIntrinsic::GridDimZ => "gridDim.z",
        }
    }
}

/// Registry for mapping Rust function names to GPU intrinsics.
#[derive(Debug)]
pub struct IntrinsicRegistry {
    mappings: HashMap<String, GpuIntrinsic>,
}

impl Default for IntrinsicRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl IntrinsicRegistry {
    /// Create a new registry with default mappings.
    pub fn new() -> Self {
        let mut mappings = HashMap::new();

        // Synchronization
        mappings.insert("sync_threads".to_string(), GpuIntrinsic::SyncThreads);
        mappings.insert("thread_fence".to_string(), GpuIntrinsic::ThreadFence);
        mappings.insert(
            "thread_fence_block".to_string(),
            GpuIntrinsic::ThreadFenceBlock,
        );
        mappings.insert(
            "thread_fence_system".to_string(),
            GpuIntrinsic::ThreadFenceSystem,
        );

        // Atomics (common naming)
        mappings.insert("atomic_add".to_string(), GpuIntrinsic::AtomicAdd);
        mappings.insert("atomic_sub".to_string(), GpuIntrinsic::AtomicSub);
        mappings.insert("atomic_min".to_string(), GpuIntrinsic::AtomicMin);
        mappings.insert("atomic_max".to_string(), GpuIntrinsic::AtomicMax);
        mappings.insert("atomic_exchange".to_string(), GpuIntrinsic::AtomicExch);
        mappings.insert("atomic_cas".to_string(), GpuIntrinsic::AtomicCas);

        // Math functions (Rust std naming)
        mappings.insert("sqrt".to_string(), GpuIntrinsic::Sqrt);
        mappings.insert("abs".to_string(), GpuIntrinsic::Fabs);
        mappings.insert("floor".to_string(), GpuIntrinsic::Floor);
        mappings.insert("ceil".to_string(), GpuIntrinsic::Ceil);
        mappings.insert("round".to_string(), GpuIntrinsic::Round);
        mappings.insert("sin".to_string(), GpuIntrinsic::Sin);
        mappings.insert("cos".to_string(), GpuIntrinsic::Cos);
        mappings.insert("tan".to_string(), GpuIntrinsic::Tan);
        mappings.insert("exp".to_string(), GpuIntrinsic::Exp);
        mappings.insert("ln".to_string(), GpuIntrinsic::Log);
        mappings.insert("log".to_string(), GpuIntrinsic::Log);
        mappings.insert("powf".to_string(), GpuIntrinsic::Pow);
        mappings.insert("powi".to_string(), GpuIntrinsic::Pow);
        mappings.insert("mul_add".to_string(), GpuIntrinsic::Fma);
        mappings.insert("min".to_string(), GpuIntrinsic::Min);
        mappings.insert("max".to_string(), GpuIntrinsic::Max);

        // CUDA thread/block indices (function-style access in Rust DSL)
        mappings.insert("thread_idx_x".to_string(), GpuIntrinsic::ThreadIdxX);
        mappings.insert("thread_idx_y".to_string(), GpuIntrinsic::ThreadIdxY);
        mappings.insert("thread_idx_z".to_string(), GpuIntrinsic::ThreadIdxZ);
        mappings.insert("block_idx_x".to_string(), GpuIntrinsic::BlockIdxX);
        mappings.insert("block_idx_y".to_string(), GpuIntrinsic::BlockIdxY);
        mappings.insert("block_idx_z".to_string(), GpuIntrinsic::BlockIdxZ);
        mappings.insert("block_dim_x".to_string(), GpuIntrinsic::BlockDimX);
        mappings.insert("block_dim_y".to_string(), GpuIntrinsic::BlockDimY);
        mappings.insert("block_dim_z".to_string(), GpuIntrinsic::BlockDimZ);
        mappings.insert("grid_dim_x".to_string(), GpuIntrinsic::GridDimX);
        mappings.insert("grid_dim_y".to_string(), GpuIntrinsic::GridDimY);
        mappings.insert("grid_dim_z".to_string(), GpuIntrinsic::GridDimZ);

        Self { mappings }
    }

    /// Look up an intrinsic by Rust function name.
    pub fn lookup(&self, name: &str) -> Option<&GpuIntrinsic> {
        self.mappings.get(name)
    }

    /// Register a custom intrinsic mapping.
    pub fn register(&mut self, rust_name: &str, intrinsic: GpuIntrinsic) {
        self.mappings.insert(rust_name.to_string(), intrinsic);
    }

    /// Check if a name is a known intrinsic.
    pub fn is_intrinsic(&self, name: &str) -> bool {
        self.mappings.contains_key(name)
    }
}

/// Stencil-specific intrinsics for neighbor access.
///
/// These are special intrinsics that the transpiler handles
/// differently based on stencil configuration.
#[derive(Debug, Clone, PartialEq)]
pub enum StencilIntrinsic {
    /// Get current cell index: `pos.idx()`
    Index,
    /// Access north neighbor: `pos.north(buf)`
    North,
    /// Access south neighbor: `pos.south(buf)`
    South,
    /// Access east neighbor: `pos.east(buf)`
    East,
    /// Access west neighbor: `pos.west(buf)`
    West,
    /// Access neighbor at offset: `pos.at(buf, dx, dy)`
    At,
    /// 3D: Access neighbor above: `pos.up(buf)`
    Up,
    /// 3D: Access neighbor below: `pos.down(buf)`
    Down,
}

impl StencilIntrinsic {
    /// Parse a method name to stencil intrinsic.
    pub fn from_method_name(name: &str) -> Option<Self> {
        match name {
            "idx" => Some(StencilIntrinsic::Index),
            "north" => Some(StencilIntrinsic::North),
            "south" => Some(StencilIntrinsic::South),
            "east" => Some(StencilIntrinsic::East),
            "west" => Some(StencilIntrinsic::West),
            "at" => Some(StencilIntrinsic::At),
            "up" => Some(StencilIntrinsic::Up),
            "down" => Some(StencilIntrinsic::Down),
            _ => None,
        }
    }
}

/// Ring kernel intrinsics for persistent actor kernels.
///
/// These intrinsics provide access to control block state, queue operations,
/// and HLC (Hybrid Logical Clock) functionality within ring kernel handlers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingKernelIntrinsic {
    // === Control Block Access ===
    /// Check if kernel is active: `is_active()`
    IsActive,
    /// Check if termination requested: `should_terminate()`
    ShouldTerminate,
    /// Mark kernel as terminated: `mark_terminated()`
    MarkTerminated,
    /// Get messages processed count: `messages_processed()`
    GetMessagesProcessed,

    // === Queue Operations ===
    /// Get input queue size: `input_queue_size()`
    InputQueueSize,
    /// Get output queue size: `output_queue_size()`
    OutputQueueSize,
    /// Check if input queue empty: `input_queue_empty()`
    InputQueueEmpty,
    /// Check if output queue empty: `output_queue_empty()`
    OutputQueueEmpty,
    /// Enqueue a response: `enqueue_response(&response)`
    EnqueueResponse,

    // === HLC Operations ===
    /// Increment HLC logical counter: `hlc_tick()`
    HlcTick,
    /// Update HLC with received timestamp: `hlc_update(received_ts)`
    HlcUpdate,
    /// Get current HLC timestamp: `hlc_now()`
    HlcNow,

    // === K2K Operations ===
    /// Send message to another kernel: `k2k_send(target_id, &msg)`
    K2kSend,
    /// Try to receive K2K message: `k2k_try_recv()`
    K2kTryRecv,
    /// Check for K2K messages: `k2k_has_message()`
    K2kHasMessage,
    /// Peek at next K2K message without consuming: `k2k_peek()`
    K2kPeek,
    /// Get number of pending K2K messages: `k2k_pending_count()`
    K2kPendingCount,

    // === Timing ===
    /// Sleep for nanoseconds: `nanosleep(ns)`
    Nanosleep,
}

impl RingKernelIntrinsic {
    /// Get the CUDA code for this intrinsic.
    pub fn to_cuda(&self, args: &[String]) -> String {
        match self {
            Self::IsActive => "atomicAdd(&control->is_active, 0) != 0".to_string(),
            Self::ShouldTerminate => "atomicAdd(&control->should_terminate, 0) != 0".to_string(),
            Self::MarkTerminated => "atomicExch(&control->has_terminated, 1)".to_string(),
            Self::GetMessagesProcessed => "atomicAdd(&control->messages_processed, 0)".to_string(),

            Self::InputQueueSize => {
                "(atomicAdd(&control->input_head, 0) - atomicAdd(&control->input_tail, 0))"
                    .to_string()
            }
            Self::OutputQueueSize => {
                "(atomicAdd(&control->output_head, 0) - atomicAdd(&control->output_tail, 0))"
                    .to_string()
            }
            Self::InputQueueEmpty => {
                "(atomicAdd(&control->input_head, 0) == atomicAdd(&control->input_tail, 0))"
                    .to_string()
            }
            Self::OutputQueueEmpty => {
                "(atomicAdd(&control->output_head, 0) == atomicAdd(&control->output_tail, 0))"
                    .to_string()
            }
            Self::EnqueueResponse => {
                if !args.is_empty() {
                    format!(
                        "{{ unsigned long long _out_idx = atomicAdd(&control->output_head, 1) & control->output_mask; \
                         memcpy(&output_buffer[_out_idx * RESP_SIZE], {}, RESP_SIZE); }}",
                        args[0]
                    )
                } else {
                    "/* enqueue_response requires response pointer */".to_string()
                }
            }

            Self::HlcTick => "hlc_logical++".to_string(),
            Self::HlcUpdate => {
                if !args.is_empty() {
                    format!(
                        "{{ if ({} > hlc_physical) {{ hlc_physical = {}; hlc_logical = 0; }} else {{ hlc_logical++; }} }}",
                        args[0], args[0]
                    )
                } else {
                    "hlc_logical++".to_string()
                }
            }
            Self::HlcNow => "(hlc_physical << 32) | (hlc_logical & 0xFFFFFFFF)".to_string(),

            Self::K2kSend => {
                if args.len() >= 2 {
                    // k2k_send(target_id, msg_ptr) -> k2k_send(k2k_routes, target_id, msg_ptr, sizeof(*msg_ptr))
                    format!(
                        "k2k_send(k2k_routes, {}, {}, sizeof(*{}))",
                        args[0], args[1], args[1]
                    )
                } else {
                    "/* k2k_send requires target_id and msg_ptr */".to_string()
                }
            }
            Self::K2kTryRecv => "k2k_try_recv(k2k_inbox)".to_string(),
            Self::K2kHasMessage => "k2k_has_message(k2k_inbox)".to_string(),
            Self::K2kPeek => "k2k_peek(k2k_inbox)".to_string(),
            Self::K2kPendingCount => "k2k_pending_count(k2k_inbox)".to_string(),

            Self::Nanosleep => {
                if !args.is_empty() {
                    format!("__nanosleep({})", args[0])
                } else {
                    "__nanosleep(1000)".to_string()
                }
            }
        }
    }

    /// Parse a function name to get the intrinsic.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "is_active" | "is_kernel_active" => Some(Self::IsActive),
            "should_terminate" => Some(Self::ShouldTerminate),
            "mark_terminated" => Some(Self::MarkTerminated),
            "messages_processed" | "get_messages_processed" => Some(Self::GetMessagesProcessed),

            "input_queue_size" => Some(Self::InputQueueSize),
            "output_queue_size" => Some(Self::OutputQueueSize),
            "input_queue_empty" => Some(Self::InputQueueEmpty),
            "output_queue_empty" => Some(Self::OutputQueueEmpty),
            "enqueue_response" | "enqueue" => Some(Self::EnqueueResponse),

            "hlc_tick" => Some(Self::HlcTick),
            "hlc_update" => Some(Self::HlcUpdate),
            "hlc_now" => Some(Self::HlcNow),

            "k2k_send" => Some(Self::K2kSend),
            "k2k_try_recv" => Some(Self::K2kTryRecv),
            "k2k_has_message" => Some(Self::K2kHasMessage),
            "k2k_peek" => Some(Self::K2kPeek),
            "k2k_pending_count" | "k2k_pending" => Some(Self::K2kPendingCount),

            "nanosleep" => Some(Self::Nanosleep),

            _ => None,
        }
    }

    /// Check if this intrinsic requires the control block.
    pub fn requires_control_block(&self) -> bool {
        matches!(
            self,
            Self::IsActive
                | Self::ShouldTerminate
                | Self::MarkTerminated
                | Self::GetMessagesProcessed
                | Self::InputQueueSize
                | Self::OutputQueueSize
                | Self::InputQueueEmpty
                | Self::OutputQueueEmpty
                | Self::EnqueueResponse
        )
    }

    /// Check if this intrinsic requires HLC state.
    pub fn requires_hlc(&self) -> bool {
        matches!(self, Self::HlcTick | Self::HlcUpdate | Self::HlcNow)
    }

    /// Check if this intrinsic requires K2K support.
    pub fn requires_k2k(&self) -> bool {
        matches!(
            self,
            Self::K2kSend
                | Self::K2kTryRecv
                | Self::K2kHasMessage
                | Self::K2kPeek
                | Self::K2kPendingCount
        )
    }
}

impl StencilIntrinsic {
    /// Get the index offset for 2D stencil (relative to buffer_width).
    ///
    /// Returns (row_offset, col_offset) where final offset is:
    /// `row_offset * buffer_width + col_offset`
    pub fn get_offset_2d(&self) -> Option<(i32, i32)> {
        match self {
            StencilIntrinsic::Index => Some((0, 0)),
            StencilIntrinsic::North => Some((-1, 0)),
            StencilIntrinsic::South => Some((1, 0)),
            StencilIntrinsic::East => Some((0, 1)),
            StencilIntrinsic::West => Some((0, -1)),
            StencilIntrinsic::At => None, // Requires runtime offset
            StencilIntrinsic::Up | StencilIntrinsic::Down => None, // 3D only
        }
    }

    /// Generate CUDA index expression for 2D stencil.
    ///
    /// # Arguments
    /// * `buffer_name` - Name of the buffer variable
    /// * `buffer_width` - Width expression (e.g., "18" for tile_size + 2*halo)
    /// * `idx_var` - Name of the current index variable
    pub fn to_cuda_index_2d(&self, buffer_name: &str, buffer_width: &str, idx_var: &str) -> String {
        match self {
            StencilIntrinsic::Index => format!("{}[{}]", buffer_name, idx_var),
            StencilIntrinsic::North => {
                format!("{}[{} - {}]", buffer_name, idx_var, buffer_width)
            }
            StencilIntrinsic::South => {
                format!("{}[{} + {}]", buffer_name, idx_var, buffer_width)
            }
            StencilIntrinsic::East => format!("{}[{} + 1]", buffer_name, idx_var),
            StencilIntrinsic::West => format!("{}[{} - 1]", buffer_name, idx_var),
            StencilIntrinsic::At => {
                // This should be handled specially with provided offsets
                format!("{}[{}]", buffer_name, idx_var)
            }
            _ => format!("{}[{}]", buffer_name, idx_var),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intrinsic_lookup() {
        let registry = IntrinsicRegistry::new();

        assert_eq!(
            registry.lookup("sync_threads"),
            Some(&GpuIntrinsic::SyncThreads)
        );
        assert_eq!(registry.lookup("sqrt"), Some(&GpuIntrinsic::Sqrt));
        assert_eq!(registry.lookup("unknown_func"), None);
    }

    #[test]
    fn test_intrinsic_cuda_output() {
        assert_eq!(
            GpuIntrinsic::SyncThreads.to_cuda_string(),
            "__syncthreads()"
        );
        assert_eq!(GpuIntrinsic::AtomicAdd.to_cuda_string(), "atomicAdd");
        assert_eq!(GpuIntrinsic::Sqrt.to_cuda_string(), "sqrtf");
    }

    #[test]
    fn test_stencil_intrinsic_parsing() {
        assert_eq!(
            StencilIntrinsic::from_method_name("north"),
            Some(StencilIntrinsic::North)
        );
        assert_eq!(
            StencilIntrinsic::from_method_name("idx"),
            Some(StencilIntrinsic::Index)
        );
        assert_eq!(StencilIntrinsic::from_method_name("unknown"), None);
    }

    #[test]
    fn test_stencil_cuda_index() {
        let north = StencilIntrinsic::North;
        assert_eq!(
            north.to_cuda_index_2d("p", "buffer_width", "idx"),
            "p[idx - buffer_width]"
        );

        let east = StencilIntrinsic::East;
        assert_eq!(east.to_cuda_index_2d("p", "18", "idx"), "p[idx + 1]");
    }

    #[test]
    fn test_stencil_offset() {
        assert_eq!(StencilIntrinsic::North.get_offset_2d(), Some((-1, 0)));
        assert_eq!(StencilIntrinsic::East.get_offset_2d(), Some((0, 1)));
        assert_eq!(StencilIntrinsic::Index.get_offset_2d(), Some((0, 0)));
    }

    #[test]
    fn test_ring_kernel_intrinsic_lookup() {
        assert_eq!(
            RingKernelIntrinsic::from_name("is_active"),
            Some(RingKernelIntrinsic::IsActive)
        );
        assert_eq!(
            RingKernelIntrinsic::from_name("should_terminate"),
            Some(RingKernelIntrinsic::ShouldTerminate)
        );
        assert_eq!(
            RingKernelIntrinsic::from_name("hlc_tick"),
            Some(RingKernelIntrinsic::HlcTick)
        );
        assert_eq!(
            RingKernelIntrinsic::from_name("enqueue_response"),
            Some(RingKernelIntrinsic::EnqueueResponse)
        );
        assert_eq!(RingKernelIntrinsic::from_name("unknown"), None);
    }

    #[test]
    fn test_ring_kernel_intrinsic_cuda_output() {
        assert!(RingKernelIntrinsic::IsActive
            .to_cuda(&[])
            .contains("is_active"));
        assert!(RingKernelIntrinsic::ShouldTerminate
            .to_cuda(&[])
            .contains("should_terminate"));
        assert!(RingKernelIntrinsic::HlcTick
            .to_cuda(&[])
            .contains("hlc_logical"));
        assert!(RingKernelIntrinsic::InputQueueEmpty
            .to_cuda(&[])
            .contains("input_head"));
    }

    #[test]
    fn test_ring_kernel_queue_intrinsics() {
        let enqueue = RingKernelIntrinsic::EnqueueResponse;
        let cuda = enqueue.to_cuda(&["&response".to_string()]);
        assert!(cuda.contains("output_head"));
        assert!(cuda.contains("memcpy"));
    }

    #[test]
    fn test_k2k_intrinsics() {
        // Test k2k_send
        let send = RingKernelIntrinsic::K2kSend;
        let cuda = send.to_cuda(&["target_id".to_string(), "&msg".to_string()]);
        assert!(cuda.contains("k2k_send"));
        assert!(cuda.contains("k2k_routes"));
        assert!(cuda.contains("target_id"));

        // Test k2k_try_recv
        assert_eq!(
            RingKernelIntrinsic::K2kTryRecv.to_cuda(&[]),
            "k2k_try_recv(k2k_inbox)"
        );

        // Test k2k_has_message
        assert_eq!(
            RingKernelIntrinsic::K2kHasMessage.to_cuda(&[]),
            "k2k_has_message(k2k_inbox)"
        );

        // Test k2k_peek
        assert_eq!(
            RingKernelIntrinsic::K2kPeek.to_cuda(&[]),
            "k2k_peek(k2k_inbox)"
        );

        // Test k2k_pending_count
        assert_eq!(
            RingKernelIntrinsic::K2kPendingCount.to_cuda(&[]),
            "k2k_pending_count(k2k_inbox)"
        );
    }

    #[test]
    fn test_k2k_intrinsic_lookup() {
        assert_eq!(
            RingKernelIntrinsic::from_name("k2k_send"),
            Some(RingKernelIntrinsic::K2kSend)
        );
        assert_eq!(
            RingKernelIntrinsic::from_name("k2k_try_recv"),
            Some(RingKernelIntrinsic::K2kTryRecv)
        );
        assert_eq!(
            RingKernelIntrinsic::from_name("k2k_has_message"),
            Some(RingKernelIntrinsic::K2kHasMessage)
        );
        assert_eq!(
            RingKernelIntrinsic::from_name("k2k_peek"),
            Some(RingKernelIntrinsic::K2kPeek)
        );
        assert_eq!(
            RingKernelIntrinsic::from_name("k2k_pending_count"),
            Some(RingKernelIntrinsic::K2kPendingCount)
        );
    }

    #[test]
    fn test_intrinsic_requirements() {
        // K2K intrinsics require K2K
        assert!(RingKernelIntrinsic::K2kSend.requires_k2k());
        assert!(RingKernelIntrinsic::K2kTryRecv.requires_k2k());
        assert!(RingKernelIntrinsic::K2kPeek.requires_k2k());
        assert!(!RingKernelIntrinsic::HlcTick.requires_k2k());

        // HLC intrinsics require HLC
        assert!(RingKernelIntrinsic::HlcTick.requires_hlc());
        assert!(RingKernelIntrinsic::HlcNow.requires_hlc());
        assert!(!RingKernelIntrinsic::K2kSend.requires_hlc());

        // Control block intrinsics require control block
        assert!(RingKernelIntrinsic::IsActive.requires_control_block());
        assert!(RingKernelIntrinsic::EnqueueResponse.requires_control_block());
        assert!(!RingKernelIntrinsic::HlcTick.requires_control_block());
    }
}
