//! GPU actor runtime for process intelligence.
//!
//! Manages GPU resources and kernel execution.

use std::time::{Duration, Instant};

/// GPU actor runtime configuration.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Target frames per second for processing.
    pub target_fps: u32,
    /// Maximum batch size.
    pub max_batch_size: usize,
    /// Enable GPU acceleration.
    pub use_gpu: bool,
    /// GPU device ID (0 = default).
    pub device_id: u32,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            target_fps: 60,
            max_batch_size: 4096,
            use_gpu: true,
            device_id: 0,
        }
    }
}

/// GPU actor runtime.
#[derive(Debug)]
pub struct GpuActorRuntime {
    /// Configuration.
    config: RuntimeConfig,
    /// Is initialized.
    initialized: bool,
    /// GPU device info.
    device_info: Option<DeviceInfo>,
    /// Frame timing.
    frame_timer: FrameTimer,
}

impl Default for GpuActorRuntime {
    fn default() -> Self {
        Self::new(RuntimeConfig::default())
    }
}

impl GpuActorRuntime {
    /// Create a new runtime.
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            config,
            initialized: false,
            device_info: None,
            frame_timer: FrameTimer::new(60),
        }
    }

    /// Initialize the runtime.
    pub fn init(&mut self) -> Result<(), String> {
        if self.initialized {
            return Ok(());
        }

        // Check for GPU availability
        if self.config.use_gpu {
            self.device_info = Self::detect_gpu();
        }

        self.frame_timer = FrameTimer::new(self.config.target_fps);
        self.initialized = true;

        Ok(())
    }

    /// Detect available GPU.
    fn detect_gpu() -> Option<DeviceInfo> {
        // Check for CUDA (simplified check)
        #[cfg(feature = "cuda")]
        {
            Some(DeviceInfo {
                name: "CUDA Device".to_string(),
                backend: GpuBackend::Cuda,
                memory_mb: 8192,
                compute_capability: "8.6".to_string(),
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    /// Check if GPU is available.
    pub fn has_gpu(&self) -> bool {
        self.device_info.is_some()
    }

    /// Get device info.
    pub fn device_info(&self) -> Option<&DeviceInfo> {
        self.device_info.as_ref()
    }

    /// Get backend name.
    pub fn backend_name(&self) -> &'static str {
        match &self.device_info {
            Some(info) => info.backend.name(),
            None => "CPU",
        }
    }

    /// Start frame timing.
    pub fn begin_frame(&mut self) {
        self.frame_timer.begin_frame();
    }

    /// End frame and get timing info.
    pub fn end_frame(&mut self) -> FrameTiming {
        self.frame_timer.end_frame()
    }

    /// Get target frame duration.
    pub fn target_frame_duration(&self) -> Duration {
        Duration::from_secs_f64(1.0 / self.config.target_fps as f64)
    }

    /// Should skip frame (for throttling).
    pub fn should_skip_frame(&self) -> bool {
        self.frame_timer.should_skip()
    }

    /// Get current FPS.
    pub fn current_fps(&self) -> f32 {
        self.frame_timer.current_fps()
    }
}

/// GPU device information.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device name.
    pub name: String,
    /// GPU backend.
    pub backend: GpuBackend,
    /// Available memory in MB.
    pub memory_mb: u32,
    /// Compute capability (CUDA) or version.
    pub compute_capability: String,
}

/// GPU backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// NVIDIA CUDA backend.
    Cuda,
    /// WebGPU cross-platform backend.
    Wgpu,
    /// Apple Metal backend.
    Metal,
    /// CPU fallback.
    Cpu,
}

impl GpuBackend {
    /// Get backend name.
    pub fn name(&self) -> &'static str {
        match self {
            GpuBackend::Cuda => "CUDA",
            GpuBackend::Wgpu => "WebGPU",
            GpuBackend::Metal => "Metal",
            GpuBackend::Cpu => "CPU",
        }
    }
}

/// Frame timing helper.
#[derive(Debug)]
pub struct FrameTimer {
    /// Target FPS.
    target_fps: u32,
    /// Last frame start.
    frame_start: Instant,
    /// Last frame duration.
    last_frame_duration: Duration,
    /// Smoothed FPS.
    smoothed_fps: f32,
    /// Frame count.
    frame_count: u64,
    /// Accumulated time for FPS calculation.
    accumulated_time: Duration,
}

impl FrameTimer {
    /// Create a new frame timer.
    pub fn new(target_fps: u32) -> Self {
        Self {
            target_fps,
            frame_start: Instant::now(),
            last_frame_duration: Duration::ZERO,
            smoothed_fps: target_fps as f32,
            frame_count: 0,
            accumulated_time: Duration::ZERO,
        }
    }

    /// Begin frame timing.
    pub fn begin_frame(&mut self) {
        self.frame_start = Instant::now();
    }

    /// End frame timing.
    pub fn end_frame(&mut self) -> FrameTiming {
        self.last_frame_duration = self.frame_start.elapsed();
        self.accumulated_time += self.last_frame_duration;
        self.frame_count += 1;

        // Update smoothed FPS
        if self.accumulated_time >= Duration::from_secs(1) {
            self.smoothed_fps = self.frame_count as f32 / self.accumulated_time.as_secs_f32();
            self.accumulated_time = Duration::ZERO;
            self.frame_count = 0;
        }

        FrameTiming {
            duration: self.last_frame_duration,
            fps: self.smoothed_fps,
            frame_budget_used: self.last_frame_duration.as_secs_f32()
                / (1.0 / self.target_fps as f32),
        }
    }

    /// Check if should skip frame (over budget).
    pub fn should_skip(&self) -> bool {
        let target_duration = Duration::from_secs_f64(1.0 / self.target_fps as f64);
        self.last_frame_duration > target_duration * 2
    }

    /// Get current FPS.
    pub fn current_fps(&self) -> f32 {
        self.smoothed_fps
    }
}

/// Frame timing information.
#[derive(Debug, Clone, Copy)]
pub struct FrameTiming {
    /// Frame duration.
    pub duration: Duration,
    /// Current FPS.
    pub fps: f32,
    /// Percentage of frame budget used (1.0 = exactly on target).
    pub frame_budget_used: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_creation() {
        let runtime = GpuActorRuntime::new(RuntimeConfig::default());
        assert!(!runtime.initialized);
    }

    #[test]
    fn test_runtime_init() {
        let mut runtime = GpuActorRuntime::new(RuntimeConfig::default());
        let result = runtime.init();
        assert!(result.is_ok());
        assert!(runtime.initialized);
    }

    #[test]
    fn test_frame_timer() {
        let mut timer = FrameTimer::new(60);
        timer.begin_frame();
        std::thread::sleep(Duration::from_millis(10));
        let timing = timer.end_frame();
        assert!(timing.duration >= Duration::from_millis(10));
    }
}
