//! WaveSim3D - Interactive 3D Acoustic Wave Simulation
//!
//! A comprehensive 3D acoustic wave simulation with:
//! - Realistic physics (temperature, humidity, frequency-dependent damping)
//! - Multiple propagation media (air, water, metal)
//! - Binaural audio output
//! - Interactive 3D visualization
//! - GPU acceleration (CUDA)

use ringkernel_wavesim3d::audio::{AudioSystem, VirtualHead};
use ringkernel_wavesim3d::gui::GuiState;
use ringkernel_wavesim3d::simulation::{Position3D, SimulationConfig, SimulationEngine};
use ringkernel_wavesim3d::visualization::{MouseButton, Renderer3D};

use std::sync::Arc;
use std::time::Instant;
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

/// Application state.
struct WaveSim3DApp {
    window: Arc<Window>,
    renderer: Renderer3D,
    engine: SimulationEngine,
    gui_state: GuiState,
    audio_system: AudioSystem,
    last_frame: Instant,
    needs_source_update: bool,
}

impl WaveSim3DApp {
    fn new(window: Arc<Window>, config: SimulationConfig) -> Self {
        let engine = config.build();
        let audio_config = ringkernel_wavesim3d::audio::AudioConfig::default();
        let audio_system = AudioSystem::new(audio_config);

        // Initialize renderer
        let grid_size = engine.grid.physical_size();
        let grid_dimensions = engine.dimensions();
        let renderer = pollster::block_on(async {
            Renderer3D::new(&window, grid_size, grid_dimensions).await
        })
        .expect("Failed to create renderer");

        Self {
            window,
            renderer,
            engine,
            gui_state: GuiState::default(),
            audio_system,
            last_frame: Instant::now(),
            needs_source_update: true,
        }
    }

    fn init_audio(&mut self) {
        let grid_size = self.engine.grid.physical_size();
        let listener_pos = Position3D::new(grid_size.0 * 0.5, grid_size.1 * 0.5, grid_size.2 * 0.7);
        let head = VirtualHead::new(listener_pos);
        self.audio_system
            .init_microphone(head, self.engine.grid.params.time_step);

        // Set initial source marker
        let source_pos = self.gui_state.source_position(grid_size);
        self.renderer.add_source(source_pos, [1.0, 0.5, 0.0, 1.0]);
        self.renderer.set_listener(listener_pos, 1.0);
    }

    fn update(&mut self) {
        let now = Instant::now();
        let _dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Run simulation steps if playing
        if self.gui_state.is_playing {
            let steps =
                (self.gui_state.steps_per_frame as f32 * self.gui_state.simulation_speed) as usize;
            for _ in 0..steps.max(1) {
                // Update audio sources
                for source in self.audio_system.sources.iter_mut() {
                    if source.active && !source.is_finished() {
                        let sample = source.next_sample(self.engine.grid.params.time_step);
                        if sample.abs() > 0.001 {
                            let (x, y, z) = source
                                .position
                                .to_grid_indices(self.engine.grid.params.cell_size);
                            if x < self.engine.grid.width
                                && y < self.engine.grid.height
                                && z < self.engine.grid.depth
                            {
                                self.engine.grid.inject_impulse(x, y, z, sample);
                            }
                        }
                    }
                }

                // Step simulation
                self.engine.step();

                // Capture binaural audio
                if let Some(ref mut mic) = self.audio_system.microphone {
                    mic.capture(&self.engine.grid);
                }
            }
        }

        // Update source marker positions
        if self.needs_source_update {
            self.update_source_markers();
            self.needs_source_update = false;
        }
    }

    fn update_source_markers(&mut self) {
        self.renderer.clear_sources();

        let grid_size = self.engine.grid.physical_size();

        // Add source marker
        let source_pos = self.gui_state.source_position(grid_size);
        self.renderer.add_source(source_pos, [1.0, 0.5, 0.0, 1.0]);

        // Add listener
        if self.gui_state.show_listener {
            let listener_pos = self.gui_state.listener_position(grid_size);
            self.renderer.set_listener(listener_pos, 1.0);
        }
    }

    fn render(&mut self) {
        // Update slice configuration
        self.renderer.slice_renderer.slices = self.gui_state.slice_configs();

        // Update render config
        self.renderer.config.auto_scale = self.gui_state.auto_scale;
        self.renderer.config.max_pressure = self.gui_state.max_pressure;
        self.renderer.config.show_bounding_box = self.gui_state.show_bounding_box;
        self.renderer.config.show_floor_grid = self.gui_state.show_floor_grid;
        self.renderer.config.show_sources = self.gui_state.show_sources;
        self.renderer.config.show_listener = self.gui_state.show_listener;

        // Render
        if let Err(e) = self.renderer.render(&self.engine.grid) {
            eprintln!("Render error: {}", e);
        }
    }

    fn inject_source(&mut self) {
        let grid_size = self.engine.grid.physical_size();
        let source = self.gui_state.create_source(grid_size);

        // For impulse, inject directly into grid
        if self.gui_state.source_type_idx == 0 {
            let pos = self.gui_state.source_position(grid_size);
            self.engine
                .grid
                .inject_spherical_impulse(pos, self.gui_state.source_amplitude, 2.0);
        } else {
            // For continuous sources, add to audio system
            self.audio_system.add_source(source);
        }

        self.needs_source_update = true;
    }

    fn reset_simulation(&mut self) {
        self.engine.reset();
        self.audio_system.reset();
        self.gui_state.is_playing = false;
    }

    fn handle_window_event(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CloseRequested => {
                return true; // Signal to exit
            }
            WindowEvent::Resized(size) => {
                self.renderer.resize(*size);
            }
            WindowEvent::RedrawRequested => {
                self.update();
                self.render();
                self.window.request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Space) => {
                            self.gui_state.is_playing = !self.gui_state.is_playing;
                        }
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            self.reset_simulation();
                        }
                        PhysicalKey::Code(KeyCode::KeyI) => {
                            self.inject_source();
                        }
                        PhysicalKey::Code(KeyCode::Digit1) => {
                            self.gui_state.show_xy_slice = !self.gui_state.show_xy_slice;
                        }
                        PhysicalKey::Code(KeyCode::Digit2) => {
                            self.gui_state.show_xz_slice = !self.gui_state.show_xz_slice;
                        }
                        PhysicalKey::Code(KeyCode::Digit3) => {
                            self.gui_state.show_yz_slice = !self.gui_state.show_yz_slice;
                        }
                        PhysicalKey::Code(KeyCode::KeyV) => {
                            // Toggle between volume and slice rendering
                            use ringkernel_wavesim3d::visualization::VisualizationMode;
                            self.renderer.config.mode = match self.renderer.config.mode {
                                VisualizationMode::VolumeRender => VisualizationMode::MultiSlice,
                                _ => VisualizationMode::VolumeRender,
                            };
                        }
                        PhysicalKey::Code(KeyCode::Escape) => {
                            return true; // Signal to exit
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let btn = match button {
                    winit::event::MouseButton::Left => MouseButton::Left,
                    winit::event::MouseButton::Middle => MouseButton::Middle,
                    winit::event::MouseButton::Right => MouseButton::Right,
                    _ => return false,
                };

                match state {
                    ElementState::Pressed => {
                        self.renderer.camera_controller.on_mouse_down(btn, 0.0, 0.0);
                    }
                    ElementState::Released => {
                        self.renderer.camera_controller.on_mouse_up(btn);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.renderer.camera_controller.on_mouse_move(
                    position.x as f32,
                    position.y as f32,
                    &mut self.renderer.camera,
                );
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
                };
                self.renderer
                    .camera_controller
                    .on_scroll(scroll, &mut self.renderer.camera);
            }
            _ => {}
        }
        false
    }
}

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║          WaveSim3D - 3D Acoustic Wave Simulation              ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  Controls:                                                    ║");
    println!("║    Space    - Play/Pause simulation                           ║");
    println!("║    R        - Reset simulation                                ║");
    println!("║    I        - Inject impulse at source position               ║");
    println!("║    V        - Toggle volume/slice rendering mode              ║");
    println!("║    1/2/3    - Toggle XY/XZ/YZ slice visibility                ║");
    println!("║    Mouse    - Rotate (left), Pan (right), Zoom (scroll)       ║");
    println!("║    Escape   - Quit                                            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    // Create simulation configuration
    let config = SimulationConfig {
        width: 64,
        height: 32,
        depth: 64,
        cell_size: 0.1, // 10cm cells
        ..Default::default()
    };

    println!("\nSimulation Configuration:");
    println!(
        "  Grid: {}×{}×{} cells",
        config.width, config.height, config.depth
    );
    let (pw, ph, pd) = config.physical_dimensions();
    println!("  Physical size: {:.1}×{:.1}×{:.1} m", pw, ph, pd);
    println!("  Cell size: {:.0} cm", config.cell_size * 100.0);
    println!(
        "  Max frequency: {:.0} Hz (accurate)",
        config.max_frequency(343.0)
    );

    // Create event loop
    let event_loop = EventLoop::new().expect("Failed to create event loop");

    // Create window
    let window_attributes = Window::default_attributes()
        .with_title("WaveSim3D - 3D Acoustic Wave Simulation")
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 800));
    #[allow(deprecated)]
    let window = Arc::new(
        event_loop
            .create_window(window_attributes)
            .expect("Failed to create window"),
    );

    // Create app
    let mut app = WaveSim3DApp::new(window.clone(), config);
    app.init_audio();

    // Run event loop (winit 0.29 closure-based API)
    // Use WaitUntil to limit frame rate to ~60fps for better efficiency
    use std::time::Duration;
    let frame_duration = Duration::from_millis(16); // ~60fps

    #[allow(deprecated)]
    event_loop
        .run(move |event, elwt| {
            match event {
                Event::WindowEvent { event, .. } => {
                    if app.handle_window_event(&event) {
                        elwt.exit();
                    }
                }
                Event::AboutToWait => {
                    // Schedule next frame at ~60fps
                    elwt.set_control_flow(ControlFlow::WaitUntil(
                        std::time::Instant::now() + frame_duration,
                    ));
                    app.window.request_redraw();
                }
                _ => {}
            }
        })
        .expect("Event loop failed");
}
