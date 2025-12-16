//! Ring kernel configuration and code generation for persistent actor kernels.
//!
//! This module provides the infrastructure for generating CUDA code for
//! persistent actor kernels that process messages in a loop.
//!
//! # Overview
//!
//! Ring kernels are persistent GPU kernels that:
//! - Run continuously until terminated
//! - Process messages from input queues
//! - Produce responses to output queues
//! - Use HLC (Hybrid Logical Clocks) for causal ordering
//! - Support kernel-to-kernel (K2K) messaging
//!
//! # Generated Kernel Structure
//!
//! ```cuda
//! extern "C" __global__ void ring_kernel_NAME(
//!     ControlBlock* __restrict__ control,
//!     unsigned char* __restrict__ input_buffer,
//!     unsigned char* __restrict__ output_buffer,
//!     void* __restrict__ shared_state
//! ) {
//!     // Preamble: thread setup
//!     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//!
//!     // Persistent message loop
//!     while (!atomicLoad(&control->should_terminate)) {
//!         if (!atomicLoad(&control->is_active)) {
//!             __nanosleep(1000);
//!             continue;
//!         }
//!
//!         // Message processing...
//!         // (user handler code inserted here)
//!     }
//!
//!     // Epilogue: mark terminated
//!     if (tid == 0) {
//!         atomicStore(&control->has_terminated, 1);
//!     }
//! }
//! ```

use std::fmt::Write;

/// Configuration for a ring kernel.
#[derive(Debug, Clone)]
pub struct RingKernelConfig {
    /// Kernel identifier (used in function name).
    pub id: String,
    /// Block size (threads per block).
    pub block_size: u32,
    /// Input queue capacity (must be power of 2).
    pub queue_capacity: u32,
    /// Enable kernel-to-kernel messaging.
    pub enable_k2k: bool,
    /// Enable HLC clock operations.
    pub enable_hlc: bool,
    /// Message size in bytes (for buffer offset calculations).
    pub message_size: usize,
    /// Response size in bytes.
    pub response_size: usize,
    /// Use cooperative thread groups.
    pub cooperative_groups: bool,
    /// Nanosleep duration when idle (0 to spin).
    pub idle_sleep_ns: u32,
    /// Use MessageEnvelope format (256-byte header + payload).
    /// When enabled, messages in queues are full envelopes with headers.
    pub use_envelope_format: bool,
    /// Kernel numeric ID (for response routing).
    pub kernel_id_num: u64,
    /// HLC node ID for this kernel.
    pub hlc_node_id: u64,
}

impl Default for RingKernelConfig {
    fn default() -> Self {
        Self {
            id: "ring_kernel".to_string(),
            block_size: 128,
            queue_capacity: 1024,
            enable_k2k: false,
            enable_hlc: true,
            message_size: 64,
            response_size: 64,
            cooperative_groups: false,
            idle_sleep_ns: 1000,
            use_envelope_format: true, // Default to envelope format for proper serialization
            kernel_id_num: 0,
            hlc_node_id: 0,
        }
    }
}

impl RingKernelConfig {
    /// Create a new ring kernel configuration with the given ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Default::default()
        }
    }

    /// Set the block size.
    pub fn with_block_size(mut self, size: u32) -> Self {
        self.block_size = size;
        self
    }

    /// Set the queue capacity (must be power of 2).
    pub fn with_queue_capacity(mut self, capacity: u32) -> Self {
        debug_assert!(
            capacity.is_power_of_two(),
            "Queue capacity must be power of 2"
        );
        self.queue_capacity = capacity;
        self
    }

    /// Enable kernel-to-kernel messaging.
    pub fn with_k2k(mut self, enabled: bool) -> Self {
        self.enable_k2k = enabled;
        self
    }

    /// Enable HLC clock operations.
    pub fn with_hlc(mut self, enabled: bool) -> Self {
        self.enable_hlc = enabled;
        self
    }

    /// Set message and response sizes.
    pub fn with_message_sizes(mut self, message: usize, response: usize) -> Self {
        self.message_size = message;
        self.response_size = response;
        self
    }

    /// Set idle sleep duration in nanoseconds.
    pub fn with_idle_sleep(mut self, ns: u32) -> Self {
        self.idle_sleep_ns = ns;
        self
    }

    /// Enable or disable MessageEnvelope format.
    /// When enabled, messages use the 256-byte header + payload format.
    pub fn with_envelope_format(mut self, enabled: bool) -> Self {
        self.use_envelope_format = enabled;
        self
    }

    /// Set the kernel numeric ID (for K2K routing and response headers).
    pub fn with_kernel_id(mut self, id: u64) -> Self {
        self.kernel_id_num = id;
        self
    }

    /// Set the HLC node ID for this kernel.
    pub fn with_hlc_node_id(mut self, node_id: u64) -> Self {
        self.hlc_node_id = node_id;
        self
    }

    /// Generate the kernel function name.
    pub fn kernel_name(&self) -> String {
        format!("ring_kernel_{}", self.id)
    }

    /// Generate CUDA kernel signature.
    pub fn generate_signature(&self) -> String {
        let mut sig = String::new();

        writeln!(sig, "extern \"C\" __global__ void {}(", self.kernel_name()).unwrap();
        writeln!(sig, "    ControlBlock* __restrict__ control,").unwrap();
        writeln!(sig, "    unsigned char* __restrict__ input_buffer,").unwrap();
        writeln!(sig, "    unsigned char* __restrict__ output_buffer,").unwrap();

        if self.enable_k2k {
            writeln!(sig, "    K2KRoutingTable* __restrict__ k2k_routes,").unwrap();
            writeln!(sig, "    unsigned char* __restrict__ k2k_inbox,").unwrap();
            writeln!(sig, "    unsigned char* __restrict__ k2k_outbox,").unwrap();
        }

        write!(sig, "    void* __restrict__ shared_state").unwrap();
        write!(sig, "\n)").unwrap();

        sig
    }

    /// Generate the kernel preamble (thread setup, variable declarations).
    pub fn generate_preamble(&self, indent: &str) -> String {
        let mut code = String::new();

        // Thread identification
        writeln!(code, "{}// Thread identification", indent).unwrap();
        writeln!(
            code,
            "{}int tid = threadIdx.x + blockIdx.x * blockDim.x;",
            indent
        )
        .unwrap();
        writeln!(code, "{}int lane_id = threadIdx.x % 32;", indent).unwrap();
        writeln!(code, "{}int warp_id = threadIdx.x / 32;", indent).unwrap();
        writeln!(code).unwrap();

        // Kernel ID constant
        writeln!(code, "{}// Kernel identity", indent).unwrap();
        writeln!(
            code,
            "{}const unsigned long long KERNEL_ID = {}ULL;",
            indent, self.kernel_id_num
        )
        .unwrap();
        writeln!(
            code,
            "{}const unsigned long long HLC_NODE_ID = {}ULL;",
            indent, self.hlc_node_id
        )
        .unwrap();
        writeln!(code).unwrap();

        // Message size constants
        writeln!(code, "{}// Message buffer constants", indent).unwrap();
        if self.use_envelope_format {
            // With envelope format, MSG_SIZE is envelope size (header + payload)
            writeln!(
                code,
                "{}const unsigned int PAYLOAD_SIZE = {};  // User payload size",
                indent, self.message_size
            )
            .unwrap();
            writeln!(
                code,
                "{}const unsigned int MSG_SIZE = MESSAGE_HEADER_SIZE + PAYLOAD_SIZE;  // Full envelope",
                indent
            )
            .unwrap();
            writeln!(
                code,
                "{}const unsigned int RESP_PAYLOAD_SIZE = {};",
                indent, self.response_size
            )
            .unwrap();
            writeln!(
                code,
                "{}const unsigned int RESP_SIZE = MESSAGE_HEADER_SIZE + RESP_PAYLOAD_SIZE;",
                indent
            )
            .unwrap();
        } else {
            // Legacy: raw message data
            writeln!(
                code,
                "{}const unsigned int MSG_SIZE = {};",
                indent, self.message_size
            )
            .unwrap();
            writeln!(
                code,
                "{}const unsigned int RESP_SIZE = {};",
                indent, self.response_size
            )
            .unwrap();
        }
        writeln!(
            code,
            "{}const unsigned int QUEUE_MASK = {};",
            indent,
            self.queue_capacity - 1
        )
        .unwrap();
        writeln!(code).unwrap();

        // HLC state (if enabled)
        if self.enable_hlc {
            writeln!(code, "{}// HLC clock state", indent).unwrap();
            writeln!(code, "{}unsigned long long hlc_physical = 0;", indent).unwrap();
            writeln!(code, "{}unsigned long long hlc_logical = 0;", indent).unwrap();
            writeln!(code).unwrap();
        }

        code
    }

    /// Generate the persistent message loop header.
    pub fn generate_loop_header(&self, indent: &str) -> String {
        let mut code = String::new();

        writeln!(code, "{}// Persistent message processing loop", indent).unwrap();
        writeln!(code, "{}while (true) {{", indent).unwrap();
        writeln!(code, "{}    // Check for termination signal", indent).unwrap();
        writeln!(
            code,
            "{}    if (atomicAdd(&control->should_terminate, 0) != 0) {{",
            indent
        )
        .unwrap();
        writeln!(code, "{}        break;", indent).unwrap();
        writeln!(code, "{}    }}", indent).unwrap();
        writeln!(code).unwrap();

        // Check if active
        writeln!(code, "{}    // Check if kernel is active", indent).unwrap();
        writeln!(
            code,
            "{}    if (atomicAdd(&control->is_active, 0) == 0) {{",
            indent
        )
        .unwrap();
        if self.idle_sleep_ns > 0 {
            writeln!(
                code,
                "{}        __nanosleep({});",
                indent, self.idle_sleep_ns
            )
            .unwrap();
        }
        writeln!(code, "{}        continue;", indent).unwrap();
        writeln!(code, "{}    }}", indent).unwrap();
        writeln!(code).unwrap();

        // Check for messages
        writeln!(code, "{}    // Check input queue for messages", indent).unwrap();
        writeln!(
            code,
            "{}    unsigned long long head = atomicAdd(&control->input_head, 0);",
            indent
        )
        .unwrap();
        writeln!(
            code,
            "{}    unsigned long long tail = atomicAdd(&control->input_tail, 0);",
            indent
        )
        .unwrap();
        writeln!(code).unwrap();
        writeln!(code, "{}    if (head == tail) {{", indent).unwrap();
        writeln!(code, "{}        // No messages, yield", indent).unwrap();
        if self.idle_sleep_ns > 0 {
            writeln!(
                code,
                "{}        __nanosleep({});",
                indent, self.idle_sleep_ns
            )
            .unwrap();
        }
        writeln!(code, "{}        continue;", indent).unwrap();
        writeln!(code, "{}    }}", indent).unwrap();
        writeln!(code).unwrap();

        // Calculate message pointer
        writeln!(code, "{}    // Get message from queue", indent).unwrap();
        writeln!(
            code,
            "{}    unsigned int msg_idx = (unsigned int)(tail & QUEUE_MASK);",
            indent
        )
        .unwrap();
        writeln!(
            code,
            "{}    unsigned char* envelope_ptr = &input_buffer[msg_idx * MSG_SIZE];",
            indent
        )
        .unwrap();

        if self.use_envelope_format {
            // Envelope format: parse header and get payload pointer
            writeln!(code).unwrap();
            writeln!(code, "{}    // Parse message envelope", indent).unwrap();
            writeln!(
                code,
                "{}    MessageHeader* msg_header = message_get_header(envelope_ptr);",
                indent
            )
            .unwrap();
            writeln!(
                code,
                "{}    unsigned char* msg_ptr = message_get_payload(envelope_ptr);",
                indent
            )
            .unwrap();
            writeln!(code).unwrap();
            writeln!(code, "{}    // Validate message (skip invalid)", indent).unwrap();
            writeln!(
                code,
                "{}    if (!message_header_validate(msg_header)) {{",
                indent
            )
            .unwrap();
            writeln!(
                code,
                "{}        atomicAdd(&control->input_tail, 1);",
                indent
            )
            .unwrap();
            writeln!(
                code,
                "{}        atomicAdd(&control->last_error, 1);  // Track errors",
                indent
            )
            .unwrap();
            writeln!(code, "{}        continue;", indent).unwrap();
            writeln!(code, "{}    }}", indent).unwrap();

            // Update HLC from incoming message timestamp
            if self.enable_hlc {
                writeln!(code).unwrap();
                writeln!(code, "{}    // Update HLC from message timestamp", indent).unwrap();
                writeln!(
                    code,
                    "{}    if (msg_header->timestamp.physical > hlc_physical) {{",
                    indent
                )
                .unwrap();
                writeln!(
                    code,
                    "{}        hlc_physical = msg_header->timestamp.physical;",
                    indent
                )
                .unwrap();
                writeln!(code, "{}        hlc_logical = 0;", indent).unwrap();
                writeln!(code, "{}    }}", indent).unwrap();
                writeln!(code, "{}    hlc_logical++;", indent).unwrap();
            }
        } else {
            // Legacy raw format
            writeln!(
                code,
                "{}    unsigned char* msg_ptr = envelope_ptr;  // Raw message data",
                indent
            )
            .unwrap();
        }
        writeln!(code).unwrap();

        code
    }

    /// Generate message processing completion code.
    pub fn generate_message_complete(&self, indent: &str) -> String {
        let mut code = String::new();

        writeln!(code).unwrap();
        writeln!(code, "{}    // Mark message as processed", indent).unwrap();
        writeln!(code, "{}    atomicAdd(&control->input_tail, 1);", indent).unwrap();
        writeln!(
            code,
            "{}    atomicAdd(&control->messages_processed, 1);",
            indent
        )
        .unwrap();

        if self.enable_hlc {
            writeln!(code).unwrap();
            writeln!(code, "{}    // Update HLC", indent).unwrap();
            writeln!(code, "{}    hlc_logical++;", indent).unwrap();
        }

        code
    }

    /// Generate the loop footer (end of while loop).
    pub fn generate_loop_footer(&self, indent: &str) -> String {
        let mut code = String::new();

        writeln!(code, "{}    __syncthreads();", indent).unwrap();
        writeln!(code, "{}}}", indent).unwrap();

        code
    }

    /// Generate kernel epilogue (termination marking).
    pub fn generate_epilogue(&self, indent: &str) -> String {
        let mut code = String::new();

        writeln!(code).unwrap();
        writeln!(code, "{}// Mark kernel as terminated", indent).unwrap();
        writeln!(code, "{}if (tid == 0) {{", indent).unwrap();

        if self.enable_hlc {
            writeln!(code, "{}    // Store final HLC state", indent).unwrap();
            writeln!(
                code,
                "{}    control->hlc_state.physical = hlc_physical;",
                indent
            )
            .unwrap();
            writeln!(
                code,
                "{}    control->hlc_state.logical = hlc_logical;",
                indent
            )
            .unwrap();
        }

        writeln!(
            code,
            "{}    atomicExch(&control->has_terminated, 1);",
            indent
        )
        .unwrap();
        writeln!(code, "{}}}", indent).unwrap();

        code
    }

    /// Generate complete kernel wrapper (without handler body).
    pub fn generate_kernel_wrapper(&self, handler_placeholder: &str) -> String {
        let mut code = String::new();

        // Struct definitions
        code.push_str(&generate_control_block_struct());
        code.push('\n');

        if self.enable_hlc {
            code.push_str(&generate_hlc_struct());
            code.push('\n');
        }

        // MessageEnvelope structs (requires HLC for HlcTimestamp)
        if self.use_envelope_format {
            code.push_str(&generate_message_envelope_structs());
            code.push('\n');
        }

        if self.enable_k2k {
            code.push_str(&generate_k2k_structs());
            code.push('\n');
        }

        // Kernel signature
        code.push_str(&self.generate_signature());
        code.push_str(" {\n");

        // Preamble
        code.push_str(&self.generate_preamble("    "));

        // Message loop
        code.push_str(&self.generate_loop_header("    "));

        // Handler placeholder
        writeln!(code, "        // === USER HANDLER CODE ===").unwrap();
        for line in handler_placeholder.lines() {
            writeln!(code, "        {}", line).unwrap();
        }
        writeln!(code, "        // === END HANDLER CODE ===").unwrap();

        // Message completion
        code.push_str(&self.generate_message_complete("    "));

        // Loop footer
        code.push_str(&self.generate_loop_footer("    "));

        // Epilogue
        code.push_str(&self.generate_epilogue("    "));

        code.push_str("}\n");

        code
    }
}

/// Generate CUDA ControlBlock struct definition.
pub fn generate_control_block_struct() -> String {
    r#"// Control block for kernel state management (128 bytes, cache-line aligned)
struct __align__(128) ControlBlock {
    // Lifecycle state
    unsigned int is_active;
    unsigned int should_terminate;
    unsigned int has_terminated;
    unsigned int _pad1;

    // Counters
    unsigned long long messages_processed;
    unsigned long long messages_in_flight;

    // Queue pointers
    unsigned long long input_head;
    unsigned long long input_tail;
    unsigned long long output_head;
    unsigned long long output_tail;

    // Queue metadata
    unsigned int input_capacity;
    unsigned int output_capacity;
    unsigned int input_mask;
    unsigned int output_mask;

    // HLC state
    struct {
        unsigned long long physical;
        unsigned long long logical;
    } hlc_state;

    // Error state
    unsigned int last_error;
    unsigned int error_count;

    // Reserved padding
    unsigned char _reserved[24];
};
"#
    .to_string()
}

/// Generate CUDA HLC helper struct.
pub fn generate_hlc_struct() -> String {
    r#"// Hybrid Logical Clock state
struct HlcState {
    unsigned long long physical;
    unsigned long long logical;
};

// HLC timestamp (24 bytes)
struct __align__(8) HlcTimestamp {
    unsigned long long physical;
    unsigned long long logical;
    unsigned long long node_id;
};
"#
    .to_string()
}

/// Generate CUDA MessageEnvelope structs matching ringkernel-core layout.
///
/// This generates the GPU-side structures that match the Rust `MessageHeader`
/// and `MessageEnvelope` types, enabling proper serialization between host and device.
pub fn generate_message_envelope_structs() -> String {
    r#"// Magic number for message validation
#define MESSAGE_MAGIC 0x52494E474B45524E ULL  // "RINGKERN"
#define MESSAGE_VERSION 1
#define MESSAGE_HEADER_SIZE 256
#define MAX_PAYLOAD_SIZE (64 * 1024)

// Message priority levels
#define PRIORITY_LOW 0
#define PRIORITY_NORMAL 1
#define PRIORITY_HIGH 2
#define PRIORITY_CRITICAL 3

// Message header structure (256 bytes, cache-line aligned)
// Matches ringkernel_core::message::MessageHeader exactly
struct __align__(64) MessageHeader {
    // Magic number for validation (0xRINGKERN)
    unsigned long long magic;
    // Header version
    unsigned int version;
    // Message flags
    unsigned int flags;
    // Unique message identifier
    unsigned long long message_id;
    // Correlation ID for request-response
    unsigned long long correlation_id;
    // Source kernel ID (0 for host)
    unsigned long long source_kernel;
    // Destination kernel ID (0 for host)
    unsigned long long dest_kernel;
    // Message type discriminator
    unsigned long long message_type;
    // Priority level
    unsigned char priority;
    // Reserved for alignment
    unsigned char _reserved1[7];
    // Payload size in bytes
    unsigned long long payload_size;
    // Checksum of payload (CRC32)
    unsigned int checksum;
    // Reserved for alignment
    unsigned int _reserved2;
    // HLC timestamp when message was created
    HlcTimestamp timestamp;
    // Deadline timestamp (0 = no deadline)
    HlcTimestamp deadline;
    // Reserved for future use (104 bytes total: 32+32+32+8)
    unsigned char _reserved3[104];
};

// Validate message header
__device__ inline int message_header_validate(const MessageHeader* header) {
    return header->magic == MESSAGE_MAGIC &&
           header->version <= MESSAGE_VERSION &&
           header->payload_size <= MAX_PAYLOAD_SIZE;
}

// Get payload pointer from header
__device__ inline unsigned char* message_get_payload(unsigned char* envelope_ptr) {
    return envelope_ptr + MESSAGE_HEADER_SIZE;
}

// Get header from envelope pointer
__device__ inline MessageHeader* message_get_header(unsigned char* envelope_ptr) {
    return (MessageHeader*)envelope_ptr;
}

// Create a response header based on request
__device__ inline void message_create_response_header(
    MessageHeader* response,
    const MessageHeader* request,
    unsigned long long this_kernel_id,
    unsigned long long payload_size,
    unsigned long long hlc_physical,
    unsigned long long hlc_logical,
    unsigned long long hlc_node_id
) {
    response->magic = MESSAGE_MAGIC;
    response->version = MESSAGE_VERSION;
    response->flags = 0;
    // Generate new message ID (simple increment from request)
    response->message_id = request->message_id + 0x100000000ULL;
    // Preserve correlation ID for request-response matching
    response->correlation_id = request->correlation_id != 0
        ? request->correlation_id
        : request->message_id;
    response->source_kernel = this_kernel_id;
    response->dest_kernel = request->source_kernel;  // Response goes back to sender
    response->message_type = request->message_type + 1;  // Convention: response type = request + 1
    response->priority = request->priority;
    response->payload_size = payload_size;
    response->checksum = 0;  // TODO: compute checksum
    response->timestamp.physical = hlc_physical;
    response->timestamp.logical = hlc_logical;
    response->timestamp.node_id = hlc_node_id;
    response->deadline.physical = 0;
    response->deadline.logical = 0;
    response->deadline.node_id = 0;
}

// Calculate total envelope size
__device__ inline unsigned int message_envelope_size(const MessageHeader* header) {
    return MESSAGE_HEADER_SIZE + (unsigned int)header->payload_size;
}
"#
    .to_string()
}

/// Generate CUDA K2K routing structs and helper functions.
pub fn generate_k2k_structs() -> String {
    r#"// Kernel-to-kernel routing table entry (48 bytes)
// Matches ringkernel_cuda::k2k_gpu::K2KRouteEntry
struct K2KRoute {
    unsigned long long target_kernel_id;
    unsigned long long target_inbox;      // device pointer
    unsigned long long target_head;       // device pointer to head
    unsigned long long target_tail;       // device pointer to tail
    unsigned int capacity;
    unsigned int mask;
    unsigned int msg_size;
    unsigned int _pad;
};

// K2K routing table (8 + 48*16 = 776 bytes)
struct K2KRoutingTable {
    unsigned int num_routes;
    unsigned int _pad;
    K2KRoute routes[16];  // Max 16 K2K connections
};

// K2K inbox header (64 bytes, cache-line aligned)
// Matches ringkernel_cuda::k2k_gpu::K2KInboxHeader
struct __align__(64) K2KInboxHeader {
    unsigned long long head;
    unsigned long long tail;
    unsigned int capacity;
    unsigned int mask;
    unsigned int msg_size;
    unsigned int _pad;
};

// Send a message envelope to another kernel via K2K
// The entire envelope (header + payload) is copied to the target's inbox
__device__ inline int k2k_send_envelope(
    K2KRoutingTable* routes,
    unsigned long long target_id,
    unsigned long long source_kernel_id,
    const void* payload_ptr,
    unsigned int payload_size,
    unsigned long long message_type,
    unsigned long long hlc_physical,
    unsigned long long hlc_logical,
    unsigned long long hlc_node_id
) {
    // Find route for target
    for (unsigned int i = 0; i < routes->num_routes; i++) {
        if (routes->routes[i].target_kernel_id == target_id) {
            K2KRoute* route = &routes->routes[i];

            // Calculate total envelope size
            unsigned int envelope_size = MESSAGE_HEADER_SIZE + payload_size;
            if (envelope_size > route->msg_size) {
                return -1;  // Message too large
            }

            // Atomically claim a slot in target's inbox
            unsigned long long* target_head_ptr = (unsigned long long*)route->target_head;
            unsigned long long slot = atomicAdd(target_head_ptr, 1);
            unsigned int idx = (unsigned int)(slot & route->mask);

            // Calculate destination pointer
            unsigned char* dest = ((unsigned char*)route->target_inbox) +
                                  sizeof(K2KInboxHeader) + idx * route->msg_size;

            // Build message header
            MessageHeader* header = (MessageHeader*)dest;
            header->magic = MESSAGE_MAGIC;
            header->version = MESSAGE_VERSION;
            header->flags = 0;
            header->message_id = (source_kernel_id << 32) | (slot & 0xFFFFFFFF);
            header->correlation_id = 0;
            header->source_kernel = source_kernel_id;
            header->dest_kernel = target_id;
            header->message_type = message_type;
            header->priority = PRIORITY_NORMAL;
            header->payload_size = payload_size;
            header->checksum = 0;
            header->timestamp.physical = hlc_physical;
            header->timestamp.logical = hlc_logical;
            header->timestamp.node_id = hlc_node_id;
            header->deadline.physical = 0;
            header->deadline.logical = 0;
            header->deadline.node_id = 0;

            // Copy payload after header
            if (payload_size > 0 && payload_ptr != NULL) {
                memcpy(dest + MESSAGE_HEADER_SIZE, payload_ptr, payload_size);
            }

            __threadfence();  // Ensure write is visible
            return 1;  // Success
        }
    }
    return 0;  // Route not found
}

// Legacy k2k_send for raw message data (no envelope)
__device__ inline int k2k_send(
    K2KRoutingTable* routes,
    unsigned long long target_id,
    const void* msg_ptr,
    unsigned int msg_size
) {
    // Find route for target
    for (unsigned int i = 0; i < routes->num_routes; i++) {
        if (routes->routes[i].target_kernel_id == target_id) {
            K2KRoute* route = &routes->routes[i];

            // Atomically claim a slot in target's inbox
            unsigned long long* target_head_ptr = (unsigned long long*)route->target_head;
            unsigned long long slot = atomicAdd(target_head_ptr, 1);
            unsigned int idx = (unsigned int)(slot & route->mask);

            // Copy message to target inbox
            unsigned char* dest = ((unsigned char*)route->target_inbox) +
                                  sizeof(K2KInboxHeader) + idx * route->msg_size;
            memcpy(dest, msg_ptr, msg_size < route->msg_size ? msg_size : route->msg_size);

            __threadfence();
            return 1;  // Success
        }
    }
    return 0;  // Route not found
}

// Check if there are K2K messages in inbox
__device__ inline int k2k_has_message(unsigned char* k2k_inbox) {
    K2KInboxHeader* header = (K2KInboxHeader*)k2k_inbox;
    unsigned long long head = atomicAdd(&header->head, 0);
    unsigned long long tail = atomicAdd(&header->tail, 0);
    return head != tail;
}

// Try to receive a K2K message envelope
// Returns pointer to MessageHeader if available, NULL otherwise
__device__ inline MessageHeader* k2k_try_recv_envelope(unsigned char* k2k_inbox) {
    K2KInboxHeader* header = (K2KInboxHeader*)k2k_inbox;

    unsigned long long head = atomicAdd(&header->head, 0);
    unsigned long long tail = atomicAdd(&header->tail, 0);

    if (head == tail) {
        return NULL;  // No messages
    }

    // Get message pointer
    unsigned int idx = (unsigned int)(tail & header->mask);
    unsigned char* data_start = k2k_inbox + sizeof(K2KInboxHeader);
    MessageHeader* msg_header = (MessageHeader*)(data_start + idx * header->msg_size);

    // Validate header
    if (!message_header_validate(msg_header)) {
        // Invalid message, skip it
        atomicAdd(&header->tail, 1);
        return NULL;
    }

    // Advance tail (consume message)
    atomicAdd(&header->tail, 1);

    return msg_header;
}

// Legacy k2k_try_recv for raw data
__device__ inline void* k2k_try_recv(unsigned char* k2k_inbox) {
    K2KInboxHeader* header = (K2KInboxHeader*)k2k_inbox;

    unsigned long long head = atomicAdd(&header->head, 0);
    unsigned long long tail = atomicAdd(&header->tail, 0);

    if (head == tail) {
        return NULL;  // No messages
    }

    // Get message pointer
    unsigned int idx = (unsigned int)(tail & header->mask);
    unsigned char* data_start = k2k_inbox + sizeof(K2KInboxHeader);

    // Advance tail (consume message)
    atomicAdd(&header->tail, 1);

    return data_start + idx * header->msg_size;
}

// Peek at next K2K message without consuming
__device__ inline MessageHeader* k2k_peek_envelope(unsigned char* k2k_inbox) {
    K2KInboxHeader* header = (K2KInboxHeader*)k2k_inbox;

    unsigned long long head = atomicAdd(&header->head, 0);
    unsigned long long tail = atomicAdd(&header->tail, 0);

    if (head == tail) {
        return NULL;  // No messages
    }

    unsigned int idx = (unsigned int)(tail & header->mask);
    unsigned char* data_start = k2k_inbox + sizeof(K2KInboxHeader);

    return (MessageHeader*)(data_start + idx * header->msg_size);
}

// Legacy k2k_peek
__device__ inline void* k2k_peek(unsigned char* k2k_inbox) {
    K2KInboxHeader* header = (K2KInboxHeader*)k2k_inbox;

    unsigned long long head = atomicAdd(&header->head, 0);
    unsigned long long tail = atomicAdd(&header->tail, 0);

    if (head == tail) {
        return NULL;  // No messages
    }

    unsigned int idx = (unsigned int)(tail & header->mask);
    unsigned char* data_start = k2k_inbox + sizeof(K2KInboxHeader);

    return data_start + idx * header->msg_size;
}

// Get number of pending K2K messages
__device__ inline unsigned int k2k_pending_count(unsigned char* k2k_inbox) {
    K2KInboxHeader* header = (K2KInboxHeader*)k2k_inbox;
    unsigned long long head = atomicAdd(&header->head, 0);
    unsigned long long tail = atomicAdd(&header->tail, 0);
    return (unsigned int)(head - tail);
}
"#
    .to_string()
}

/// Intrinsic functions available in ring kernel handlers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingKernelIntrinsic {
    // Control block access
    IsActive,
    ShouldTerminate,
    MarkTerminated,
    GetMessagesProcessed,

    // Queue operations
    InputQueueSize,
    OutputQueueSize,
    InputQueueEmpty,
    OutputQueueEmpty,
    EnqueueResponse,

    // HLC operations
    HlcTick,
    HlcUpdate,
    HlcNow,

    // K2K operations
    K2kSend,
    K2kTryRecv,
    K2kHasMessage,
    K2kPeek,
    K2kPendingCount,
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

            _ => None,
        }
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

    /// Check if this intrinsic requires HLC support.
    pub fn requires_hlc(&self) -> bool {
        matches!(self, Self::HlcTick | Self::HlcUpdate | Self::HlcNow)
    }

    /// Check if this intrinsic requires control block access.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RingKernelConfig::default();
        assert_eq!(config.block_size, 128);
        assert_eq!(config.queue_capacity, 1024);
        assert!(config.enable_hlc);
        assert!(!config.enable_k2k);
    }

    #[test]
    fn test_config_builder() {
        let config = RingKernelConfig::new("processor")
            .with_block_size(256)
            .with_queue_capacity(2048)
            .with_k2k(true)
            .with_hlc(true);

        assert_eq!(config.id, "processor");
        assert_eq!(config.kernel_name(), "ring_kernel_processor");
        assert_eq!(config.block_size, 256);
        assert_eq!(config.queue_capacity, 2048);
        assert!(config.enable_k2k);
        assert!(config.enable_hlc);
    }

    #[test]
    fn test_kernel_signature() {
        let config = RingKernelConfig::new("test");
        let sig = config.generate_signature();

        assert!(sig.contains("extern \"C\" __global__ void ring_kernel_test"));
        assert!(sig.contains("ControlBlock* __restrict__ control"));
        assert!(sig.contains("input_buffer"));
        assert!(sig.contains("output_buffer"));
        assert!(sig.contains("shared_state"));
    }

    #[test]
    fn test_kernel_signature_with_k2k() {
        let config = RingKernelConfig::new("k2k_test").with_k2k(true);
        let sig = config.generate_signature();

        assert!(sig.contains("K2KRoutingTable"));
        assert!(sig.contains("k2k_inbox"));
        assert!(sig.contains("k2k_outbox"));
    }

    #[test]
    fn test_preamble_generation() {
        let config = RingKernelConfig::new("test").with_hlc(true);
        let preamble = config.generate_preamble("    ");

        assert!(preamble.contains("int tid = threadIdx.x + blockIdx.x * blockDim.x"));
        assert!(preamble.contains("int lane_id"));
        assert!(preamble.contains("int warp_id"));
        assert!(preamble.contains("MSG_SIZE"));
        assert!(preamble.contains("hlc_physical"));
        assert!(preamble.contains("hlc_logical"));
    }

    #[test]
    fn test_loop_header() {
        let config = RingKernelConfig::new("test");
        let header = config.generate_loop_header("    ");

        assert!(header.contains("while (true)"));
        assert!(header.contains("should_terminate"));
        assert!(header.contains("is_active"));
        assert!(header.contains("input_head"));
        assert!(header.contains("input_tail"));
        assert!(header.contains("msg_ptr"));
    }

    #[test]
    fn test_epilogue() {
        let config = RingKernelConfig::new("test").with_hlc(true);
        let epilogue = config.generate_epilogue("    ");

        assert!(epilogue.contains("has_terminated"));
        assert!(epilogue.contains("hlc_state.physical"));
        assert!(epilogue.contains("hlc_state.logical"));
    }

    #[test]
    fn test_control_block_struct() {
        let code = generate_control_block_struct();

        assert!(code.contains("struct __align__(128) ControlBlock"));
        assert!(code.contains("is_active"));
        assert!(code.contains("should_terminate"));
        assert!(code.contains("has_terminated"));
        assert!(code.contains("messages_processed"));
        assert!(code.contains("input_head"));
        assert!(code.contains("input_tail"));
        assert!(code.contains("hlc_state"));
    }

    #[test]
    fn test_full_kernel_wrapper() {
        let config = RingKernelConfig::new("example")
            .with_block_size(128)
            .with_hlc(true);

        let kernel = config.generate_kernel_wrapper("// Process message here");

        assert!(kernel.contains("struct __align__(128) ControlBlock"));
        assert!(kernel.contains("extern \"C\" __global__ void ring_kernel_example"));
        assert!(kernel.contains("while (true)"));
        assert!(kernel.contains("// Process message here"));
        assert!(kernel.contains("has_terminated"));

        println!("Generated kernel:\n{}", kernel);
    }

    #[test]
    fn test_intrinsic_lookup() {
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
        assert_eq!(RingKernelIntrinsic::from_name("unknown"), None);
    }

    #[test]
    fn test_intrinsic_cuda_output() {
        assert!(RingKernelIntrinsic::IsActive
            .to_cuda(&[])
            .contains("is_active"));
        assert!(RingKernelIntrinsic::ShouldTerminate
            .to_cuda(&[])
            .contains("should_terminate"));
        assert!(RingKernelIntrinsic::HlcTick
            .to_cuda(&[])
            .contains("hlc_logical++"));
    }

    #[test]
    fn test_k2k_structs_generation() {
        let k2k_code = generate_k2k_structs();

        // Check struct definitions
        assert!(
            k2k_code.contains("struct K2KRoute"),
            "Should have K2KRoute struct"
        );
        assert!(
            k2k_code.contains("struct K2KRoutingTable"),
            "Should have K2KRoutingTable struct"
        );
        assert!(
            k2k_code.contains("K2KInboxHeader"),
            "Should have K2KInboxHeader struct"
        );

        // Check helper functions
        assert!(
            k2k_code.contains("__device__ inline int k2k_send"),
            "Should have k2k_send function"
        );
        assert!(
            k2k_code.contains("__device__ inline int k2k_has_message"),
            "Should have k2k_has_message function"
        );
        assert!(
            k2k_code.contains("__device__ inline void* k2k_try_recv"),
            "Should have k2k_try_recv function"
        );
        assert!(
            k2k_code.contains("__device__ inline void* k2k_peek"),
            "Should have k2k_peek function"
        );
        assert!(
            k2k_code.contains("__device__ inline unsigned int k2k_pending_count"),
            "Should have k2k_pending_count function"
        );

        println!("K2K code:\n{}", k2k_code);
    }

    #[test]
    fn test_full_k2k_kernel() {
        let config = RingKernelConfig::new("k2k_processor")
            .with_block_size(128)
            .with_k2k(true)
            .with_hlc(true);

        let kernel = config.generate_kernel_wrapper("// K2K handler code");

        // Check K2K-specific components
        assert!(
            kernel.contains("K2KRoutingTable"),
            "Should have K2KRoutingTable"
        );
        assert!(kernel.contains("K2KRoute"), "Should have K2KRoute struct");
        assert!(
            kernel.contains("K2KInboxHeader"),
            "Should have K2KInboxHeader"
        );
        assert!(
            kernel.contains("k2k_routes"),
            "Should have k2k_routes param"
        );
        assert!(kernel.contains("k2k_inbox"), "Should have k2k_inbox param");
        assert!(
            kernel.contains("k2k_outbox"),
            "Should have k2k_outbox param"
        );
        assert!(kernel.contains("k2k_send"), "Should have k2k_send function");
        assert!(
            kernel.contains("k2k_try_recv"),
            "Should have k2k_try_recv function"
        );

        println!("Full K2K kernel:\n{}", kernel);
    }

    #[test]
    fn test_k2k_intrinsic_requirements() {
        assert!(RingKernelIntrinsic::K2kSend.requires_k2k());
        assert!(RingKernelIntrinsic::K2kTryRecv.requires_k2k());
        assert!(RingKernelIntrinsic::K2kHasMessage.requires_k2k());
        assert!(RingKernelIntrinsic::K2kPeek.requires_k2k());
        assert!(RingKernelIntrinsic::K2kPendingCount.requires_k2k());

        assert!(!RingKernelIntrinsic::HlcTick.requires_k2k());
        assert!(!RingKernelIntrinsic::IsActive.requires_k2k());
    }
}
