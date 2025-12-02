//! Main Iced Application for the wave simulation.

use super::canvas::GridCanvas;
use super::controls;
use crate::simulation::{AcousticParams, CellType, KernelGrid, SimulationGrid};

#[cfg(feature = "cuda")]
use crate::simulation::CudaPackedBackend;

use iced::widget::{container, row, Canvas};
use iced::{Element, Length, Size, Subscription, Task, Theme};
use ringkernel::prelude::Backend;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Available compute backends for the GUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    /// CPU with SoA + SIMD + Rayon optimization.
    Cpu,
    /// GPU kernel actor model (WGPU/CUDA per-tile).
    GpuActor,
    /// CUDA Packed backend (GPU-only halo exchange).
    #[cfg(feature = "cuda")]
    CudaPacked,
}

impl std::fmt::Display for ComputeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeBackend::Cpu => write!(f, "CPU (SIMD+Rayon)"),
            ComputeBackend::GpuActor => write!(f, "GPU Actor"),
            #[cfg(feature = "cuda")]
            ComputeBackend::CudaPacked => write!(f, "CUDA Packed"),
        }
    }
}

/// Drawing mode for cell manipulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DrawMode {
    /// Normal mode - left click injects impulse.
    #[default]
    Impulse,
    /// Draw absorber cells (absorb waves).
    Absorber,
    /// Draw reflector cells (reflect waves like walls).
    Reflector,
    /// Erase cell types (reset to normal).
    Erase,
}

impl std::fmt::Display for DrawMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DrawMode::Impulse => write!(f, "Impulse"),
            DrawMode::Absorber => write!(f, "Absorber"),
            DrawMode::Reflector => write!(f, "Reflector"),
            DrawMode::Erase => write!(f, "Erase"),
        }
    }
}

/// Simulation engine abstraction for CPU or GPU modes.
enum SimulationEngine {
    /// CPU-based synchronous simulation.
    Cpu(SimulationGrid),
    /// GPU-based kernel actor simulation.
    Gpu(Arc<Mutex<KernelGrid>>),
    /// CUDA Packed backend (GPU-only halo exchange).
    #[cfg(feature = "cuda")]
    CudaPacked(Arc<std::sync::Mutex<CudaPackedBackend>>),
    /// Transitioning state (while switching backends).
    Switching,
}


/// The main wave simulation application.
pub struct WaveSimApp {
    /// The simulation engine (CPU or GPU).
    engine: SimulationEngine,
    /// The canvas for rendering.
    canvas: GridCanvas,

    // Cached grid dimensions (for GPU mode)
    grid_width: u32,
    grid_height: u32,

    // UI state
    /// Current grid width setting.
    grid_width_input: String,
    /// Current grid height setting.
    grid_height_input: String,
    /// Current speed of sound.
    speed_of_sound: f32,
    /// Current cell size in meters.
    cell_size: f32,
    /// Selected compute backend.
    compute_backend: ComputeBackend,
    /// Legacy backend for GPU actor mode.
    legacy_backend: Backend,
    /// Whether simulation is running.
    is_running: bool,
    /// Whether to show stats panel.
    show_stats: bool,
    /// Impulse amplitude for clicks.
    impulse_amplitude: f32,
    /// Current drawing mode.
    draw_mode: DrawMode,
    /// Cell types for drawing (CPU mode uses grid's internal, CUDA uses this).
    cell_types: Vec<Vec<CellType>>,

    // Performance tracking
    /// Last frame time for FPS calculation.
    last_frame: Instant,
    /// Current FPS (frames per second).
    fps: f32,
    /// Steps per second (simulation steps).
    steps_per_sec: f32,
    /// Cell throughput (cells updated per second).
    throughput: f64,
    /// Steps executed in the last frame.
    steps_last_frame: u32,
    /// Accumulated step count for averaging.
    step_accumulator: u32,
    /// Time accumulator for averaging.
    time_accumulator: Duration,
    /// Last time stats were updated.
    last_stats_update: Instant,

    // Stats (updated from grid)
    cell_count: usize,
    courant_number: f32,
    max_pressure: f32,
    total_energy: f32,
}

/// Messages for the application.
#[derive(Debug, Clone)]
pub enum Message {
    /// Toggle simulation running state.
    ToggleRunning,
    /// Perform a single simulation step.
    Step,
    /// Reset the simulation.
    Reset,

    /// Speed of sound slider changed.
    SpeedChanged(f32),
    /// Grid width input changed.
    GridWidthChanged(String),
    /// Grid height input changed.
    GridHeightChanged(String),
    /// Apply grid size changes.
    ApplyGridSize,

    /// Impulse amplitude changed.
    ImpulseAmplitudeChanged(f32),

    /// Cell size changed.
    CellSizeChanged(f32),

    /// Compute backend changed.
    ComputeBackendChanged(ComputeBackend),
    /// Backend switch completed (for GPU actor mode).
    BackendSwitched(Result<Arc<Mutex<KernelGrid>>, String>),
    /// CUDA Packed backend switch completed.
    #[cfg(feature = "cuda")]
    CudaPackedSwitched(Result<Arc<std::sync::Mutex<CudaPackedBackend>>, String>),

    /// User clicked on the canvas (left click).
    CanvasClick(f32, f32),
    /// User right-clicked on the canvas (for drawing).
    CanvasRightClick(f32, f32),
    /// Drawing mode changed.
    DrawModeChanged(DrawMode),
    /// Clear all drawn cell types.
    ClearCellTypes,

    /// Animation tick.
    Tick,
    /// GPU step completed (for actor mode).
    GpuStepCompleted(Vec<Vec<f32>>, f32, f32),
    /// CUDA Packed step completed.
    #[cfg(feature = "cuda")]
    CudaPackedStepCompleted(Vec<f32>, u32),

    /// Toggle stats display.
    ToggleStats,
}

impl WaveSimApp {
    /// Create a new application instance.
    pub fn new() -> (Self, Task<Message>) {
        let params = AcousticParams::new(343.0, 1.0);
        let grid = SimulationGrid::new(64, 64, params.clone());
        let pressure_grid = grid.get_pressure_grid();

        let cell_count = grid.cell_count();
        let courant_number = grid.params.courant_number();
        let now = Instant::now();

        (
            Self {
                engine: SimulationEngine::Cpu(grid),
                canvas: GridCanvas::new(pressure_grid),
                grid_width: 64,
                grid_height: 64,
                grid_width_input: "64".to_string(),
                grid_height_input: "64".to_string(),
                speed_of_sound: 343.0,
                cell_size: 1.0,
                compute_backend: ComputeBackend::Cpu,
                legacy_backend: Backend::Cpu,
                is_running: false,
                show_stats: true,
                impulse_amplitude: 1.0,
                draw_mode: DrawMode::Impulse,
                cell_types: vec![vec![CellType::Normal; 64]; 64],
                last_frame: now,
                fps: 0.0,
                steps_per_sec: 0.0,
                throughput: 0.0,
                steps_last_frame: 0,
                step_accumulator: 0,
                time_accumulator: Duration::ZERO,
                last_stats_update: now,
                cell_count,
                courant_number,
                max_pressure: 0.0,
                total_energy: 0.0,
            },
            Task::none(),
        )
    }

    /// Application title.
    pub fn title(&self) -> String {
        format!(
            "RingKernel WaveSim - {}x{} Grid ({})",
            self.grid_width, self.grid_height, self.compute_backend
        )
    }

    /// Handle messages.
    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::ToggleRunning => {
                self.is_running = !self.is_running;
                // Reset performance stats when starting
                if self.is_running {
                    self.step_accumulator = 0;
                    self.time_accumulator = Duration::ZERO;
                    self.last_stats_update = Instant::now();
                }
            }

            Message::Step => {
                return self.step_simulation();
            }

            Message::Reset => {
                self.reset_performance_stats();
                match &self.engine {
                    SimulationEngine::Cpu(_) => {
                        if let SimulationEngine::Cpu(grid) = &mut self.engine {
                            grid.reset();
                            let pressure_grid = grid.get_pressure_grid();
                            self.max_pressure = grid.max_pressure();
                            self.total_energy = grid.total_energy();
                            self.canvas.update_pressure(pressure_grid);
                        }
                    }
                    SimulationEngine::Gpu(kernel_grid) => {
                        let grid = kernel_grid.clone();
                        return Task::perform(
                            async move {
                                let mut g = grid.lock().await;
                                g.reset();
                                (
                                    g.get_pressure_grid(),
                                    g.max_pressure(),
                                    g.total_energy(),
                                )
                            },
                            |(pressure, max_p, energy)| {
                                Message::GpuStepCompleted(pressure, max_p, energy)
                            },
                        );
                    }
                    #[cfg(feature = "cuda")]
                    SimulationEngine::CudaPacked(_) => {
                        // For CUDA Packed, we need to recreate the backend
                        return self.switch_to_cuda_packed();
                    }
                    SimulationEngine::Switching => {}
                }
            }

            Message::SpeedChanged(speed) => {
                self.speed_of_sound = speed;
                let params = AcousticParams::new(speed, self.cell_size);
                let c2 = params.courant_number().powi(2);
                let damping = 1.0 - params.damping;
                self.courant_number = params.courant_number();

                match &mut self.engine {
                    SimulationEngine::Cpu(grid) => {
                        grid.set_speed_of_sound(speed);
                    }
                    SimulationEngine::Gpu(kernel_grid) => {
                        let grid = kernel_grid.clone();
                        tokio::spawn(async move {
                            grid.lock().await.set_speed_of_sound(speed);
                        });
                    }
                    #[cfg(feature = "cuda")]
                    SimulationEngine::CudaPacked(backend) => {
                        if let Ok(mut b) = backend.lock() {
                            b.set_params(c2, damping);
                        }
                    }
                    SimulationEngine::Switching => {}
                }
            }

            Message::GridWidthChanged(s) => {
                self.grid_width_input = s;
            }

            Message::GridHeightChanged(s) => {
                self.grid_height_input = s;
            }

            Message::ApplyGridSize => {
                if let (Ok(w), Ok(h)) = (
                    self.grid_width_input.parse::<u32>(),
                    self.grid_height_input.parse::<u32>(),
                ) {
                    // Allow larger grids for CUDA Packed
                    #[cfg(feature = "cuda")]
                    let max_size = if matches!(self.compute_backend, ComputeBackend::CudaPacked) {
                        512
                    } else {
                        256
                    };
                    #[cfg(not(feature = "cuda"))]
                    let max_size = 256;

                    let w = w.clamp(16, max_size);
                    let h = h.clamp(16, max_size);
                    self.grid_width = w;
                    self.grid_height = h;
                    self.grid_width_input = w.to_string();
                    self.grid_height_input = h.to_string();
                    self.cell_count = (w * h) as usize;
                    self.reset_performance_stats();

                    // Recreate the engine with new size
                    return self.switch_backend(self.compute_backend);
                }
            }

            Message::ImpulseAmplitudeChanged(amp) => {
                self.impulse_amplitude = amp;
            }

            Message::CellSizeChanged(size) => {
                self.cell_size = size;
                let params = AcousticParams::new(self.speed_of_sound, size);
                let c2 = params.courant_number().powi(2);
                let damping = 1.0 - params.damping;
                self.courant_number = params.courant_number();

                match &mut self.engine {
                    SimulationEngine::Cpu(grid) => {
                        grid.set_cell_size(size);
                    }
                    SimulationEngine::Gpu(kernel_grid) => {
                        let grid = kernel_grid.clone();
                        tokio::spawn(async move {
                            grid.lock().await.set_cell_size(size);
                        });
                    }
                    #[cfg(feature = "cuda")]
                    SimulationEngine::CudaPacked(backend) => {
                        if let Ok(mut b) = backend.lock() {
                            b.set_params(c2, damping);
                        }
                    }
                    SimulationEngine::Switching => {}
                }
            }

            Message::ComputeBackendChanged(new_backend) => {
                if new_backend == self.compute_backend {
                    return Task::none();
                }
                tracing::info!("Switching compute backend from {} to {}", self.compute_backend, new_backend);
                self.compute_backend = new_backend;
                self.reset_performance_stats();
                return self.switch_backend(new_backend);
            }

            Message::BackendSwitched(result) => {
                match result {
                    Ok(kernel_grid) => {
                        self.engine = SimulationEngine::Gpu(kernel_grid.clone());
                        return Task::perform(
                            async move {
                                let g = kernel_grid.lock().await;
                                (
                                    g.get_pressure_grid(),
                                    g.max_pressure(),
                                    g.total_energy(),
                                )
                            },
                            |(pressure, max_p, energy)| {
                                Message::GpuStepCompleted(pressure, max_p, energy)
                            },
                        );
                    }
                    Err(e) => {
                        tracing::error!("Failed to switch backend: {}", e);
                        self.compute_backend = ComputeBackend::Cpu;
                        let params = AcousticParams::new(self.speed_of_sound, self.cell_size);
                        let grid = SimulationGrid::new(self.grid_width, self.grid_height, params);
                        self.cell_count = grid.cell_count();
                        self.courant_number = grid.params.courant_number();
                        let pressure = grid.get_pressure_grid();
                        self.engine = SimulationEngine::Cpu(grid);
                        self.canvas.update_pressure(pressure);
                    }
                }
            }

            #[cfg(feature = "cuda")]
            Message::CudaPackedSwitched(result) => {
                match result {
                    Ok(backend) => {
                        self.engine = SimulationEngine::CudaPacked(backend.clone());
                        // Read initial state
                        let pressure = {
                            let b = backend.lock().unwrap();
                            b.read_pressure_grid().unwrap_or_else(|_| vec![0.0; self.cell_count])
                        };
                        self.update_canvas_from_flat(&pressure);
                    }
                    Err(e) => {
                        tracing::error!("Failed to create CUDA Packed backend: {}", e);
                        self.compute_backend = ComputeBackend::Cpu;
                        let params = AcousticParams::new(self.speed_of_sound, self.cell_size);
                        let grid = SimulationGrid::new(self.grid_width, self.grid_height, params);
                        self.cell_count = grid.cell_count();
                        self.courant_number = grid.params.courant_number();
                        let pressure = grid.get_pressure_grid();
                        self.engine = SimulationEngine::Cpu(grid);
                        self.canvas.update_pressure(pressure);
                    }
                }
            }

            Message::CanvasClick(x, y) => {
                let gx = (x * self.grid_width as f32) as u32;
                let gy = (y * self.grid_height as f32) as u32;
                let gx = gx.min(self.grid_width - 1);
                let gy = gy.min(self.grid_height - 1);

                // Handle based on draw mode
                match self.draw_mode {
                    DrawMode::Impulse => {
                        // Original impulse behavior
                        match &self.engine {
                            SimulationEngine::Cpu(_) => {
                                if let SimulationEngine::Cpu(grid) = &mut self.engine {
                                    grid.inject_impulse(gx, gy, self.impulse_amplitude);
                                    let pressure_grid = grid.get_pressure_grid();
                                    self.max_pressure = grid.max_pressure();
                                    self.total_energy = grid.total_energy();
                                    self.canvas.update_pressure(pressure_grid);
                                }
                            }
                            SimulationEngine::Gpu(kernel_grid) => {
                                let grid = kernel_grid.clone();
                                let amp = self.impulse_amplitude;
                                return Task::perform(
                                    async move {
                                        let mut g = grid.lock().await;
                                        g.inject_impulse(gx, gy, amp);
                                        (
                                            g.get_pressure_grid(),
                                            g.max_pressure(),
                                            g.total_energy(),
                                        )
                                    },
                                    |(pressure, max_p, energy)| {
                                        Message::GpuStepCompleted(pressure, max_p, energy)
                                    },
                                );
                            }
                            #[cfg(feature = "cuda")]
                            SimulationEngine::CudaPacked(backend) => {
                                let backend = backend.clone();
                                let amp = self.impulse_amplitude;
                                return Task::perform(
                                    async move {
                                        let b = backend.lock().unwrap();
                                        let _ = b.inject_impulse(gx, gy, amp);
                                        let pressure = b.read_pressure_grid().unwrap_or_default();
                                        (pressure, 0u32)
                                    },
                                    |(pressure, steps)| {
                                        Message::CudaPackedStepCompleted(pressure, steps)
                                    },
                                );
                            }
                            SimulationEngine::Switching => {}
                        }
                    }
                    DrawMode::Absorber | DrawMode::Reflector | DrawMode::Erase => {
                        // Set cell type
                        let cell_type = match self.draw_mode {
                            DrawMode::Absorber => CellType::Absorber,
                            DrawMode::Reflector => CellType::Reflector,
                            DrawMode::Erase => CellType::Normal,
                            _ => CellType::Normal,
                        };
                        self.set_cell_type_at(gx, gy, cell_type);
                    }
                }
            }

            Message::CanvasRightClick(x, y) => {
                // Right-click always sets cell type based on current draw mode
                let gx = (x * self.grid_width as f32) as u32;
                let gy = (y * self.grid_height as f32) as u32;
                let gx = gx.min(self.grid_width - 1);
                let gy = gy.min(self.grid_height - 1);

                let cell_type = match self.draw_mode {
                    DrawMode::Impulse => CellType::Normal, // Right-click in impulse mode erases
                    DrawMode::Absorber => CellType::Absorber,
                    DrawMode::Reflector => CellType::Reflector,
                    DrawMode::Erase => CellType::Normal,
                };
                self.set_cell_type_at(gx, gy, cell_type);
            }

            Message::DrawModeChanged(mode) => {
                self.draw_mode = mode;
            }

            Message::ClearCellTypes => {
                // Clear all cell types
                for row in &mut self.cell_types {
                    for cell in row {
                        *cell = CellType::Normal;
                    }
                }
                // Also clear in the CPU grid if applicable
                if let SimulationEngine::Cpu(grid) = &mut self.engine {
                    grid.clear_cell_types();
                }
                self.canvas.update_cell_types(self.cell_types.clone());
            }

            Message::Tick => {
                if self.is_running {
                    return self.step_simulation();
                }
            }

            Message::GpuStepCompleted(pressure_grid, max_pressure, total_energy) => {
                self.canvas.update_pressure(pressure_grid);
                self.max_pressure = max_pressure;
                self.total_energy = total_energy;
                self.update_frame_stats(self.steps_last_frame);
            }

            #[cfg(feature = "cuda")]
            Message::CudaPackedStepCompleted(pressure, steps) => {
                self.update_canvas_from_flat(&pressure);
                self.update_frame_stats(steps);
            }

            Message::ToggleStats => {
                self.show_stats = !self.show_stats;
            }
        }

        Task::none()
    }

    /// Switch to a new compute backend.
    fn switch_backend(&mut self, backend: ComputeBackend) -> Task<Message> {
        match backend {
            ComputeBackend::Cpu => {
                let params = AcousticParams::new(self.speed_of_sound, self.cell_size);
                let grid = SimulationGrid::new(self.grid_width, self.grid_height, params);
                self.cell_count = grid.cell_count();
                self.courant_number = grid.params.courant_number();
                let pressure = grid.get_pressure_grid();
                self.engine = SimulationEngine::Cpu(grid);
                self.canvas.update_pressure(pressure);
                Task::none()
            }
            ComputeBackend::GpuActor => {
                self.engine = SimulationEngine::Switching;
                let width = self.grid_width;
                let height = self.grid_height;
                let speed = self.speed_of_sound;
                let cell_size = self.cell_size;
                let legacy_backend = self.legacy_backend;

                Task::perform(
                    async move {
                        let params = AcousticParams::new(speed, cell_size);
                        match KernelGrid::new(width, height, params, legacy_backend).await {
                            Ok(grid) => Ok(Arc::new(Mutex::new(grid))),
                            Err(e) => Err(format!("{:?}", e)),
                        }
                    },
                    Message::BackendSwitched,
                )
            }
            #[cfg(feature = "cuda")]
            ComputeBackend::CudaPacked => {
                self.switch_to_cuda_packed()
            }
        }
    }

    /// Switch to CUDA Packed backend.
    #[cfg(feature = "cuda")]
    fn switch_to_cuda_packed(&mut self) -> Task<Message> {
        self.engine = SimulationEngine::Switching;
        let width = self.grid_width;
        let height = self.grid_height;
        let speed = self.speed_of_sound;
        let cell_size = self.cell_size;

        Task::perform(
            async move {
                let params = AcousticParams::new(speed, cell_size);
                let c2 = params.courant_number().powi(2);
                let damping = 1.0 - params.damping;

                match CudaPackedBackend::new(width, height, 16) {
                    Ok(mut backend) => {
                        backend.set_params(c2, damping);
                        Ok(Arc::new(std::sync::Mutex::new(backend)))
                    }
                    Err(e) => Err(format!("{:?}", e)),
                }
            },
            Message::CudaPackedSwitched,
        )
    }

    /// Reset performance statistics.
    fn reset_performance_stats(&mut self) {
        self.fps = 0.0;
        self.steps_per_sec = 0.0;
        self.throughput = 0.0;
        self.steps_last_frame = 0;
        self.step_accumulator = 0;
        self.time_accumulator = Duration::ZERO;
        self.last_stats_update = Instant::now();
    }

    /// Update frame statistics after a step completes.
    fn update_frame_stats(&mut self, steps: u32) {
        let now = Instant::now();
        let delta = now.duration_since(self.last_frame);
        self.fps = 1.0 / delta.as_secs_f32();
        self.last_frame = now;
        self.steps_last_frame = steps;

        // Accumulate for averaging
        self.step_accumulator += steps;
        self.time_accumulator += delta;

        // Update stats every 500ms
        let stats_delta = now.duration_since(self.last_stats_update);
        if stats_delta >= Duration::from_millis(500) {
            let secs = self.time_accumulator.as_secs_f64();
            if secs > 0.0 {
                self.steps_per_sec = self.step_accumulator as f32 / secs as f32;
                self.throughput = (self.step_accumulator as f64 * self.cell_count as f64) / secs;
            }
            self.step_accumulator = 0;
            self.time_accumulator = Duration::ZERO;
            self.last_stats_update = now;
        }
    }

    /// Update canvas from flat pressure array.
    fn update_canvas_from_flat(&mut self, pressure: &[f32]) {
        let mut grid = Vec::with_capacity(self.grid_height as usize);
        for y in 0..self.grid_height as usize {
            let start = y * self.grid_width as usize;
            let end = start + self.grid_width as usize;
            if end <= pressure.len() {
                grid.push(pressure[start..end].to_vec());
            } else {
                grid.push(vec![0.0; self.grid_width as usize]);
            }
        }

        // Calculate max pressure and energy
        self.max_pressure = pressure.iter().fold(0.0f32, |acc, &p| acc.max(p.abs()));
        self.total_energy = pressure.iter().map(|&p| p * p).sum();

        self.canvas.update_pressure(grid);
    }

    /// Set cell type at the given grid position.
    fn set_cell_type_at(&mut self, gx: u32, gy: u32, cell_type: CellType) {
        let gx = gx as usize;
        let gy = gy as usize;

        // Resize cell_types if needed
        let height = self.grid_height as usize;
        let width = self.grid_width as usize;
        if self.cell_types.len() != height || (height > 0 && self.cell_types[0].len() != width) {
            self.cell_types = vec![vec![CellType::Normal; width]; height];
        }

        // Update local cell_types array
        if gy < self.cell_types.len() && gx < self.cell_types[gy].len() {
            self.cell_types[gy][gx] = cell_type;
        }

        // Update simulation grid for CPU mode
        if let SimulationEngine::Cpu(grid) = &mut self.engine {
            grid.set_cell_type(gx as u32, gy as u32, cell_type);
        }

        // Update canvas to show cell types
        self.canvas.update_cell_types(self.cell_types.clone());
    }

    /// Perform simulation step(s).
    fn step_simulation(&mut self) -> Task<Message> {
        match &self.engine {
            SimulationEngine::Cpu(_) => {
                let start = Instant::now();
                if let SimulationEngine::Cpu(grid) = &mut self.engine {
                    // Run steps for approximately 16ms (60 FPS target)
                    let target_frame_time = Duration::from_millis(16);
                    let dt = grid.params.time_step;
                    let max_steps = ((0.01 / dt) as u32).clamp(1, 2000);

                    let mut steps = 0u32;
                    while start.elapsed() < target_frame_time && steps < max_steps {
                        grid.step();
                        steps += 1;
                    }

                    self.steps_last_frame = steps;
                    let pressure_grid = grid.get_pressure_grid();
                    self.max_pressure = grid.max_pressure();
                    self.total_energy = grid.total_energy();
                    self.canvas.update_pressure(pressure_grid);
                    self.update_frame_stats(steps);
                }
                Task::none()
            }
            SimulationEngine::Gpu(kernel_grid) => {
                let grid = kernel_grid.clone();

                Task::perform(
                    async move {
                        let mut g = grid.lock().await;
                        let dt = g.params.time_step;
                        let steps = ((0.01 / dt) as u32).clamp(1, 500);

                        for _ in 0..steps {
                            if let Err(e) = g.step().await {
                                tracing::error!("GPU step error: {:?}", e);
                                break;
                            }
                        }
                        (g.get_pressure_grid(), g.max_pressure(), g.total_energy())
                    },
                    |(pressure, max_p, energy)| Message::GpuStepCompleted(pressure, max_p, energy),
                )
            }
            #[cfg(feature = "cuda")]
            SimulationEngine::CudaPacked(backend) => {
                let backend = backend.clone();
                let cell_count = self.cell_count;

                Task::perform(
                    async move {
                        let mut b = backend.lock().unwrap();
                        // Run many steps per frame for CUDA Packed (it's fast!)
                        let steps = 100u32;
                        if let Err(e) = b.step_batch(steps) {
                            tracing::error!("CUDA step error: {:?}", e);
                        }
                        let pressure = b.read_pressure_grid().unwrap_or_else(|_| vec![0.0; cell_count]);
                        (pressure, steps)
                    },
                    |(pressure, steps)| Message::CudaPackedStepCompleted(pressure, steps),
                )
            }
            SimulationEngine::Switching => Task::none(),
        }
    }

    /// Build the view.
    pub fn view(&self) -> Element<'_, Message> {
        let canvas = Canvas::new(&self.canvas)
            .width(Length::Fill)
            .height(Length::Fill);

        let controls = controls::view_controls(
            self.is_running,
            self.speed_of_sound,
            self.cell_size,
            &self.grid_width_input,
            &self.grid_height_input,
            self.impulse_amplitude,
            self.compute_backend,
            self.draw_mode,
            self.show_stats,
            self.fps,
            self.steps_per_sec,
            self.throughput,
            self.cell_count,
            self.courant_number,
            self.max_pressure,
            self.total_energy,
        );

        let content = row![
            container(canvas)
                .width(Length::FillPortion(3))
                .height(Length::Fill)
                .padding(10),
            container(controls)
                .width(Length::FillPortion(1))
                .height(Length::Fill)
                .padding(10),
        ];

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    /// Subscriptions for animation.
    pub fn subscription(&self) -> Subscription<Message> {
        if self.is_running {
            iced::time::every(Duration::from_millis(16)).map(|_| Message::Tick)
        } else {
            Subscription::none()
        }
    }

    /// Theme.
    pub fn theme(&self) -> Theme {
        Theme::Dark
    }
}

impl Default for WaveSimApp {
    fn default() -> Self {
        Self::new().0
    }
}

/// Run the application.
pub fn run() -> iced::Result {
    iced::application(WaveSimApp::title, WaveSimApp::update, WaveSimApp::view)
        .subscription(WaveSimApp::subscription)
        .theme(WaveSimApp::theme)
        .window_size(Size::new(1200.0, 800.0))
        .antialiasing(true)
        .run_with(WaveSimApp::new)
}
