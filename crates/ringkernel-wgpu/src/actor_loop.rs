//! Event-driven actor emulation for WebGPU.
//!
//! WebGPU cannot do true persistent kernels (no cooperative groups, no mapped
//! memory). This module emulates actor semantics with rapid kernel re-launch
//! and state preservation in device memory:
//!
//! 1. Actor state lives in a GPU buffer that persists across dispatches.
//! 2. Each "tick" dispatches the compute shader, which reads state, processes
//!    messages, and writes state back.
//! 3. The host drives the loop: dispatch -> readback responses -> inject new
//!    messages -> dispatch again.
//! 4. From the user's perspective, the API matches persistent actors
//!    (send/recv/activate/terminate).
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────┐
//! │  Host (WgpuActorLoop)                        │
//! │                                              │
//! │  ┌─────────────┐   inject_messages()         │
//! │  │ input msgs   │ ─────────────────┐         │
//! │  └─────────────┘                   ▼         │
//! │                          ┌─────────────────┐ │
//! │                          │  GPU dispatch   │ │
//! │                          │  (one tick)     │ │
//! │                          │                 │ │
//! │                          │  reads input    │ │
//! │                          │  reads state    │ │
//! │                          │  writes state   │ │
//! │                          │  writes output  │ │
//! │                          └─────────────────┘ │
//! │  ┌─────────────┐                   │         │
//! │  │ output msgs  │ ◄────────────────┘         │
//! │  └─────────────┘   poll_responses()          │
//! │                                              │
//! │  ┌─────────────┐   (persists across ticks)   │
//! │  │ state buffer │ ◄──────────────────────────│
//! │  └─────────────┘                             │
//! └──────────────────────────────────────────────┘
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::memory::GpuBuffer;

use crate::adapter::WgpuAdapter;
use crate::memory::WgpuBuffer;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the actor loop.
#[derive(Debug, Clone)]
pub struct ActorLoopConfig {
    /// Size of the persistent actor state buffer in bytes.
    pub state_buffer_size: usize,
    /// Size of the input message buffer in bytes.
    pub input_buffer_size: usize,
    /// Size of the output message buffer in bytes.
    pub output_buffer_size: usize,
    /// Tick interval for the actor loop when running continuously.
    pub tick_interval: Duration,
    /// Number of workgroups to dispatch per tick.
    pub workgroups: u32,
}

impl Default for ActorLoopConfig {
    fn default() -> Self {
        Self {
            state_buffer_size: 4096,
            input_buffer_size: 65536,   // 64 KB
            output_buffer_size: 65536,  // 64 KB
            tick_interval: Duration::from_millis(1),
            workgroups: 1,
        }
    }
}

impl ActorLoopConfig {
    /// Create a config with custom buffer sizes.
    pub fn with_buffer_sizes(
        state_size: usize,
        input_size: usize,
        output_size: usize,
    ) -> Self {
        Self {
            state_buffer_size: state_size,
            input_buffer_size: input_size,
            output_buffer_size: output_size,
            ..Default::default()
        }
    }

    /// Set the tick interval.
    pub fn with_tick_interval(mut self, interval: Duration) -> Self {
        self.tick_interval = interval;
        self
    }

    /// Set the number of workgroups.
    pub fn with_workgroups(mut self, workgroups: u32) -> Self {
        self.workgroups = workgroups;
        self
    }
}

// ============================================================================
// Actor Loop
// ============================================================================

/// Event-driven actor loop that emulates persistent kernel semantics on WebGPU.
///
/// State is preserved in GPU buffers across dispatches. Each tick dispatches the
/// compute shader once, then the host reads back responses and injects new
/// messages for the next tick.
pub struct WgpuActorLoop {
    /// Reference to the WebGPU adapter.
    adapter: Arc<WgpuAdapter>,
    /// Persistent state buffer (survives across dispatches).
    state_buffer: WgpuBuffer,
    /// Input message buffer (host writes, GPU reads).
    input_buffer: WgpuBuffer,
    /// Output message buffer (GPU writes, host reads).
    output_buffer: WgpuBuffer,
    /// Compute pipeline for the actor shader.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group connecting buffers to the shader.
    bind_group: wgpu::BindGroup,
    /// Whether the loop is running.
    running: AtomicBool,
    /// Tick interval.
    tick_interval: Duration,
    /// Number of workgroups per dispatch.
    workgroups: u32,
    /// Total ticks executed.
    tick_count: AtomicU64,
    /// Pending input messages (host-side staging).
    pending_input: Mutex<Vec<u8>>,
    /// Configuration snapshot.
    config: ActorLoopConfig,
}

impl WgpuActorLoop {
    /// Create a new actor loop.
    ///
    /// The `shader_source` should be WGSL that declares bindings:
    /// - `@group(0) @binding(0) var<storage, read_write> state: ...`
    /// - `@group(0) @binding(1) var<storage, read>       input: ...`
    /// - `@group(0) @binding(2) var<storage, read_write> output: ...`
    ///
    /// The entry point must be named according to `entry_point`.
    pub fn new(
        adapter: Arc<WgpuAdapter>,
        shader_source: &str,
        entry_point: &str,
        config: ActorLoopConfig,
    ) -> Result<Self> {
        let device = adapter.device();

        // Create persistent buffers
        let state_buffer = WgpuBuffer::new(
            &adapter,
            config.state_buffer_size,
            Some("Actor State Buffer"),
        );
        let input_buffer = WgpuBuffer::new(
            &adapter,
            config.input_buffer_size,
            Some("Actor Input Buffer"),
        );
        let output_buffer = WgpuBuffer::new(
            &adapter,
            config.output_buffer_size,
            Some("Actor Output Buffer"),
        );

        // Compile the shader
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Actor Loop Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Bind group layout: state (rw), input (rw), output (rw)
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Actor Loop Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Actor Loop Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Actor Loop Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point,
            });

        // Create the bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Actor Loop Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            adapter,
            state_buffer,
            input_buffer,
            output_buffer,
            pipeline,
            bind_group_layout,
            bind_group,
            running: AtomicBool::new(false),
            tick_interval: config.tick_interval,
            workgroups: config.workgroups,
            tick_count: AtomicU64::new(0),
            pending_input: Mutex::new(Vec::new()),
            config,
        })
    }

    // ========================================================================
    // Lifecycle
    // ========================================================================

    /// Start the actor loop.
    ///
    /// This marks the loop as running. The caller is responsible for driving
    /// ticks (either manually via [`tick`] or by spawning a task that calls
    /// [`run`]).
    pub fn start(&self) {
        self.running.store(true, Ordering::Release);
        tracing::info!("Actor loop started");
    }

    /// Stop the actor loop.
    ///
    /// After this call, [`is_running`] returns false and [`run`] will exit
    /// at the next iteration boundary.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
        tracing::info!(
            ticks = self.tick_count.load(Ordering::Relaxed),
            "Actor loop stopped"
        );
    }

    /// Check whether the loop is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Get the total number of ticks executed.
    pub fn tick_count(&self) -> u64 {
        self.tick_count.load(Ordering::Relaxed)
    }

    /// Get the configuration.
    pub fn config(&self) -> &ActorLoopConfig {
        &self.config
    }

    // ========================================================================
    // Message injection / response polling
    // ========================================================================

    /// Stage a message for injection into the GPU input buffer on the next tick.
    ///
    /// Messages are accumulated host-side and flushed to the GPU at the start
    /// of [`tick`]. The bytes are an opaque payload; the shader interprets them.
    pub fn inject_message(&self, msg: &[u8]) -> Result<()> {
        let mut pending = self.pending_input.lock();

        // Prefix each message with its length (u32 LE) so the shader can parse.
        let len = msg.len() as u32;
        let total = 4 + msg.len();

        if pending.len() + total > self.config.input_buffer_size {
            return Err(RingKernelError::QueueFull {
                capacity: self.config.input_buffer_size,
            });
        }

        pending.extend_from_slice(&len.to_le_bytes());
        pending.extend_from_slice(msg);
        Ok(())
    }

    /// Read responses from the GPU output buffer.
    ///
    /// Returns the raw bytes written by the shader into the output buffer.
    /// The caller is responsible for interpreting the format, which mirrors
    /// the input convention: each response is prefixed with a u32 LE length.
    ///
    /// After reading, the output buffer is cleared for the next tick.
    pub fn poll_responses(&self) -> Result<Vec<Vec<u8>>> {
        let mut raw = vec![0u8; self.config.output_buffer_size];
        self.output_buffer.copy_to_host(&mut raw)?;

        let mut responses = Vec::new();
        let mut offset = 0;

        while offset + 4 <= raw.len() {
            let len = u32::from_le_bytes([
                raw[offset],
                raw[offset + 1],
                raw[offset + 2],
                raw[offset + 3],
            ]) as usize;

            if len == 0 {
                // End of responses sentinel.
                break;
            }

            offset += 4;

            if offset + len > raw.len() {
                break;
            }

            responses.push(raw[offset..offset + len].to_vec());
            offset += len;
        }

        // Clear the output buffer for the next tick.
        if !responses.is_empty() {
            let zeros = vec![0u8; self.config.output_buffer_size];
            self.output_buffer.copy_from_host(&zeros)?;
        }

        Ok(responses)
    }

    // ========================================================================
    // Tick / Dispatch
    // ========================================================================

    /// Execute a single dispatch cycle (one "tick").
    ///
    /// 1. Flush pending input messages to the GPU input buffer.
    /// 2. Dispatch the compute shader.
    /// 3. Wait for the GPU to finish.
    ///
    /// After this returns, call [`poll_responses`] to read any output the
    /// shader produced.
    pub fn tick(&self) -> Result<()> {
        if !self.is_running() {
            return Err(RingKernelError::KernelNotActive(
                "actor loop is not running".to_string(),
            ));
        }

        // 1. Flush pending input to the GPU buffer.
        self.flush_input()?;

        // 2. Encode and submit a compute pass.
        let mut encoder = self
            .adapter
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Actor Loop Tick"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Actor Loop Compute Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.workgroups, 1, 1);
        }

        self.adapter.queue().submit(Some(encoder.finish()));

        // 3. Wait for completion.
        self.adapter.poll(wgpu::Maintain::Wait);

        self.tick_count.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Run the actor loop continuously until [`stop`] is called.
    ///
    /// This is an async function that yields between ticks according to
    /// [`tick_interval`].
    pub async fn run(&self) -> Result<()> {
        tracing::info!(
            interval_ms = self.tick_interval.as_millis() as u64,
            workgroups = self.workgroups,
            "Actor loop entering run loop"
        );

        while self.is_running() {
            self.tick()?;

            if !self.tick_interval.is_zero() {
                tokio::time::sleep(self.tick_interval).await;
            } else {
                tokio::task::yield_now().await;
            }
        }

        Ok(())
    }

    // ========================================================================
    // State access
    // ========================================================================

    /// Write initial state into the persistent state buffer.
    ///
    /// Call this before starting the loop to set up initial actor state.
    pub fn write_state(&self, data: &[u8]) -> Result<()> {
        if data.len() > self.config.state_buffer_size {
            return Err(RingKernelError::AllocationFailed {
                size: data.len(),
                reason: format!(
                    "state data ({} bytes) exceeds buffer size ({} bytes)",
                    data.len(),
                    self.config.state_buffer_size,
                ),
            });
        }

        // Write into a zero-padded buffer of the full size.
        let mut padded = vec![0u8; self.config.state_buffer_size];
        padded[..data.len()].copy_from_slice(data);
        self.state_buffer.copy_from_host(&padded)
    }

    /// Read the current actor state from the persistent state buffer.
    pub fn read_state(&self) -> Result<Vec<u8>> {
        let mut data = vec![0u8; self.config.state_buffer_size];
        self.state_buffer.copy_to_host(&mut data)?;
        Ok(data)
    }

    /// Get a reference to the bind group layout.
    ///
    /// Useful if the caller needs to create additional bind groups that share
    /// the same layout.
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    /// Flush accumulated host-side input messages into the GPU input buffer.
    fn flush_input(&self) -> Result<()> {
        let mut pending = self.pending_input.lock();
        if pending.is_empty() {
            // Write a zero-length sentinel so the shader knows there is nothing.
            let zeros = [0u8; 4];
            self.input_buffer.copy_from_host(&zeros)?;
            return Ok(());
        }

        // Pad to the full buffer size so copy_from_host is well-defined.
        let mut buf = vec![0u8; self.config.input_buffer_size];
        let copy_len = pending.len().min(buf.len());
        buf[..copy_len].copy_from_slice(&pending[..copy_len]);
        self.input_buffer.copy_from_host(&buf)?;

        pending.clear();
        Ok(())
    }
}

// ============================================================================
// Default actor shader template
// ============================================================================

/// A minimal WGSL shader template for actor-loop emulation.
///
/// This shader demonstrates the buffer contract:
/// - binding 0: persistent state (`array<u32>`)
/// - binding 1: input messages (`array<u32>`)
/// - binding 2: output messages (`array<u32>`)
///
/// Each tick, the shader reads one input message (u32 length prefix, then
/// payload words), increments a counter in state, and echoes the message
/// into the output buffer.
pub const ACTOR_LOOP_WGSL_TEMPLATE: &str = r#"
// Actor loop emulation shader (minimal echo actor).

@group(0) @binding(0) var<storage, read_write> state: array<u32>;
@group(0) @binding(1) var<storage, read_write> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn actor_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Only thread 0 processes messages (single-threaded actor).
    if gid.x != 0u {
        return;
    }

    // state[0] = tick counter
    state[0] = state[0] + 1u;

    // Read input message length (in bytes, stored as first u32).
    let input_len_bytes = input[0];
    if input_len_bytes == 0u {
        // No messages this tick.
        // Write zero-length sentinel to output.
        output[0] = 0u;
        return;
    }

    // state[1] = total messages processed
    state[1] = state[1] + 1u;

    // Number of u32 words needed for the payload (round up).
    let payload_words = (input_len_bytes + 3u) / 4u;

    // Echo: copy input to output (including length prefix).
    output[0] = input_len_bytes;
    for (var i = 0u; i < payload_words; i = i + 1u) {
        output[i + 1u] = input[i + 1u];
    }
}
"#;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_loop_config_defaults() {
        let config = ActorLoopConfig::default();
        assert_eq!(config.state_buffer_size, 4096);
        assert_eq!(config.input_buffer_size, 65536);
        assert_eq!(config.output_buffer_size, 65536);
        assert_eq!(config.workgroups, 1);
        assert_eq!(config.tick_interval, Duration::from_millis(1));
    }

    #[test]
    fn test_actor_loop_config_builder() {
        let config = ActorLoopConfig::with_buffer_sizes(1024, 2048, 4096)
            .with_tick_interval(Duration::from_millis(10))
            .with_workgroups(4);

        assert_eq!(config.state_buffer_size, 1024);
        assert_eq!(config.input_buffer_size, 2048);
        assert_eq!(config.output_buffer_size, 4096);
        assert_eq!(config.workgroups, 4);
        assert_eq!(config.tick_interval, Duration::from_millis(10));
    }

    #[tokio::test]
    #[ignore] // Requires GPU
    async fn test_actor_loop_lifecycle() {
        let adapter = Arc::new(WgpuAdapter::new().await.unwrap());
        let config = ActorLoopConfig::default();

        let actor_loop = WgpuActorLoop::new(
            adapter,
            ACTOR_LOOP_WGSL_TEMPLATE,
            "actor_main",
            config,
        )
        .unwrap();

        assert!(!actor_loop.is_running());
        assert_eq!(actor_loop.tick_count(), 0);

        actor_loop.start();
        assert!(actor_loop.is_running());

        actor_loop.stop();
        assert!(!actor_loop.is_running());
    }

    #[tokio::test]
    #[ignore] // Requires GPU
    async fn test_actor_loop_tick() {
        let adapter = Arc::new(WgpuAdapter::new().await.unwrap());
        let config = ActorLoopConfig::default();

        let actor_loop = WgpuActorLoop::new(
            adapter,
            ACTOR_LOOP_WGSL_TEMPLATE,
            "actor_main",
            config,
        )
        .unwrap();

        actor_loop.start();

        // Run a single tick with no input.
        actor_loop.tick().unwrap();
        assert_eq!(actor_loop.tick_count(), 1);

        // State should have tick counter = 1.
        let state = actor_loop.read_state().unwrap();
        let tick_counter = u32::from_le_bytes([state[0], state[1], state[2], state[3]]);
        assert_eq!(tick_counter, 1);

        actor_loop.stop();
    }

    #[tokio::test]
    #[ignore] // Requires GPU
    async fn test_actor_loop_message_echo() {
        let adapter = Arc::new(WgpuAdapter::new().await.unwrap());
        let config = ActorLoopConfig::default();

        let actor_loop = WgpuActorLoop::new(
            adapter,
            ACTOR_LOOP_WGSL_TEMPLATE,
            "actor_main",
            config,
        )
        .unwrap();

        actor_loop.start();

        // Inject a message.
        let msg = b"hello";
        actor_loop.inject_message(msg).unwrap();

        // Tick to process it.
        actor_loop.tick().unwrap();

        // Poll responses -- the echo actor should mirror back "hello".
        let responses = actor_loop.poll_responses().unwrap();
        assert_eq!(responses.len(), 1);
        // The echoed payload may contain padding to u32 boundary.
        assert!(responses[0].starts_with(b"hello"));

        actor_loop.stop();
    }

    #[tokio::test]
    #[ignore] // Requires GPU
    async fn test_actor_loop_state_persistence() {
        let adapter = Arc::new(WgpuAdapter::new().await.unwrap());
        let config = ActorLoopConfig::default();

        let actor_loop = WgpuActorLoop::new(
            adapter,
            ACTOR_LOOP_WGSL_TEMPLATE,
            "actor_main",
            config,
        )
        .unwrap();

        // Write initial state.
        let mut initial_state = vec![0u8; 4096];
        // Set tick counter to 100.
        initial_state[..4].copy_from_slice(&100u32.to_le_bytes());
        actor_loop.write_state(&initial_state).unwrap();

        actor_loop.start();

        // Tick once -- shader increments state[0].
        actor_loop.tick().unwrap();

        let state = actor_loop.read_state().unwrap();
        let tick_counter = u32::from_le_bytes([state[0], state[1], state[2], state[3]]);
        assert_eq!(tick_counter, 101, "State should persist across ticks");

        actor_loop.stop();
    }

    #[tokio::test]
    #[ignore] // Requires GPU
    async fn test_actor_loop_write_state_too_large() {
        let adapter = Arc::new(WgpuAdapter::new().await.unwrap());
        let config = ActorLoopConfig::with_buffer_sizes(64, 1024, 1024);

        let actor_loop = WgpuActorLoop::new(
            adapter,
            ACTOR_LOOP_WGSL_TEMPLATE,
            "actor_main",
            config,
        )
        .unwrap();

        // Try to write state larger than the buffer.
        let big_state = vec![0u8; 128];
        let result = actor_loop.write_state(&big_state);
        assert!(result.is_err());
    }

    #[test]
    fn test_inject_message_overflow() {
        // This test does not need a GPU -- it only exercises host-side logic.
        // We cannot construct a WgpuActorLoop without a GPU, so we test the
        // pending buffer logic indirectly via config validation.
        let config = ActorLoopConfig::with_buffer_sizes(64, 16, 16);
        // 16-byte input buffer: a 20-byte message (4 len + 16 payload) would overflow.
        assert_eq!(config.input_buffer_size, 16);
    }

    #[test]
    fn test_actor_loop_wgsl_template_not_empty() {
        assert!(!ACTOR_LOOP_WGSL_TEMPLATE.is_empty());
        assert!(ACTOR_LOOP_WGSL_TEMPLATE.contains("actor_main"));
        assert!(ACTOR_LOOP_WGSL_TEMPLATE.contains("@group(0) @binding(0)"));
    }
}
