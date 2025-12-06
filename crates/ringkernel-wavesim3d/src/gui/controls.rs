//! GUI controls for the 3D wave simulation.
//!
//! Uses egui for immediate-mode UI rendering.

use crate::audio::AudioSource;
use crate::simulation::{Medium, Position3D, SimulationEngine};
use crate::visualization::{ColorMap, SliceAxis, SliceConfig};

/// GUI state for the simulation.
#[derive(Debug, Clone)]
pub struct GuiState {
    // Simulation controls
    pub is_playing: bool,
    pub simulation_speed: f32,
    pub steps_per_frame: u32,

    // Environment
    pub temperature_c: f32,
    pub humidity_percent: f32,
    pub medium: Medium,

    // Slice controls
    pub xy_slice_position: f32,
    pub xz_slice_position: f32,
    pub yz_slice_position: f32,
    pub show_xy_slice: bool,
    pub show_xz_slice: bool,
    pub show_yz_slice: bool,

    // Source controls
    pub source_x: f32,
    pub source_y: f32,
    pub source_z: f32,
    pub source_amplitude: f32,
    pub source_frequency: f32,
    pub source_type_idx: usize,

    // Listener controls
    pub listener_x: f32,
    pub listener_y: f32,
    pub listener_z: f32,
    pub listener_yaw: f32,
    pub show_listener: bool,

    // Visualization
    pub color_map_idx: usize,
    pub auto_scale: bool,
    pub max_pressure: f32,
    pub slice_opacity: f32,
    pub show_bounding_box: bool,
    pub show_floor_grid: bool,
    pub show_sources: bool,

    // Info
    pub show_info_panel: bool,
}

impl Default for GuiState {
    fn default() -> Self {
        Self {
            is_playing: false,
            simulation_speed: 1.0,
            steps_per_frame: 10,

            temperature_c: 20.0,
            humidity_percent: 50.0,
            medium: Medium::Air,

            xy_slice_position: 0.5,
            xz_slice_position: 0.5,
            yz_slice_position: 0.5,
            show_xy_slice: true,
            show_xz_slice: false,
            show_yz_slice: false,

            source_x: 0.5,
            source_y: 0.5,
            source_z: 0.5,
            source_amplitude: 1.0,
            source_frequency: 440.0,
            source_type_idx: 0,

            listener_x: 0.5,
            listener_y: 0.5,
            listener_z: 0.7,
            listener_yaw: 0.0,
            show_listener: true,

            color_map_idx: 0,
            auto_scale: true,
            max_pressure: 1.0,
            slice_opacity: 0.85,
            show_bounding_box: true,
            show_floor_grid: true,
            show_sources: true,

            show_info_panel: true,
        }
    }
}

impl GuiState {
    /// Get the selected color map.
    pub fn color_map(&self) -> ColorMap {
        match self.color_map_idx {
            0 => ColorMap::BlueWhiteRed,
            1 => ColorMap::CoolWarm,
            2 => ColorMap::Hot,
            3 => ColorMap::Diverging,
            _ => ColorMap::Grayscale,
        }
    }

    /// Get the selected source type name.
    pub fn source_type_name(&self) -> &'static str {
        match self.source_type_idx {
            0 => "Impulse",
            1 => "Tone",
            2 => "Chirp",
            3 => "White Noise",
            4 => "Pink Noise",
            _ => "Gaussian Pulse",
        }
    }

    /// Create an audio source from current settings.
    pub fn create_source(&self, grid_size: (f32, f32, f32)) -> AudioSource {
        let position = Position3D::new(
            self.source_x * grid_size.0,
            self.source_y * grid_size.1,
            self.source_z * grid_size.2,
        );

        match self.source_type_idx {
            0 => AudioSource::impulse(0, position, self.source_amplitude),
            1 => AudioSource::tone(0, position, self.source_frequency, self.source_amplitude),
            2 => AudioSource::chirp(0, position, 100.0, 2000.0, 0.5, self.source_amplitude),
            3 => AudioSource::noise(0, position, self.source_amplitude, false),
            4 => AudioSource::noise(0, position, self.source_amplitude, true),
            _ => AudioSource::gaussian_pulse(0, position, 0.01, 0.002, self.source_amplitude),
        }
    }

    /// Get source position in world coordinates.
    pub fn source_position(&self, grid_size: (f32, f32, f32)) -> Position3D {
        Position3D::new(
            self.source_x * grid_size.0,
            self.source_y * grid_size.1,
            self.source_z * grid_size.2,
        )
    }

    /// Get listener position in world coordinates.
    pub fn listener_position(&self, grid_size: (f32, f32, f32)) -> Position3D {
        Position3D::new(
            self.listener_x * grid_size.0,
            self.listener_y * grid_size.1,
            self.listener_z * grid_size.2,
        )
    }

    /// Create slice configurations from state.
    pub fn slice_configs(&self) -> Vec<SliceConfig> {
        let mut configs = Vec::new();

        if self.show_xy_slice {
            configs.push(SliceConfig {
                axis: SliceAxis::XY,
                position: self.xy_slice_position,
                opacity: self.slice_opacity,
                color_map: self.color_map(),
                show_grid: false,
            });
        }

        if self.show_xz_slice {
            configs.push(SliceConfig {
                axis: SliceAxis::XZ,
                position: self.xz_slice_position,
                opacity: self.slice_opacity,
                color_map: self.color_map(),
                show_grid: false,
            });
        }

        if self.show_yz_slice {
            configs.push(SliceConfig {
                axis: SliceAxis::YZ,
                position: self.yz_slice_position,
                opacity: self.slice_opacity,
                color_map: self.color_map(),
                show_grid: false,
            });
        }

        configs
    }
}

/// Simulation controls panel.
pub struct SimulationControls {
    state: GuiState,
}

impl SimulationControls {
    pub fn new() -> Self {
        Self {
            state: GuiState::default(),
        }
    }

    pub fn state(&self) -> &GuiState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut GuiState {
        &mut self.state
    }

    /// Render the main control panel.
    pub fn ui(&mut self, ctx: &egui::Context, engine: &SimulationEngine) {
        egui::SidePanel::left("controls")
            .default_width(280.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.simulation_section(ui, engine);
                    ui.separator();
                    self.environment_section(ui);
                    ui.separator();
                    self.source_section(ui);
                    ui.separator();
                    self.listener_section(ui);
                    ui.separator();
                    self.visualization_section(ui);
                    ui.separator();
                    self.slice_section(ui);
                });
            });

        if self.state.show_info_panel {
            egui::Window::new("Simulation Info")
                .anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])
                .show(ctx, |ui| {
                    self.info_panel(ui, engine);
                });
        }
    }

    fn simulation_section(&mut self, ui: &mut egui::Ui, engine: &SimulationEngine) {
        ui.heading("Simulation");

        ui.horizontal(|ui| {
            if ui
                .button(if self.state.is_playing { "⏸ Pause" } else { "▶ Play" })
                .clicked()
            {
                self.state.is_playing = !self.state.is_playing;
            }
            if ui.button("⟲ Reset").clicked() {
                self.state.is_playing = false;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Speed:");
            ui.add(egui::Slider::new(&mut self.state.simulation_speed, 0.1..=5.0));
        });

        ui.horizontal(|ui| {
            ui.label("Steps/frame:");
            ui.add(egui::Slider::new(&mut self.state.steps_per_frame, 1..=50));
        });

        let using_gpu = engine.is_using_gpu();
        ui.label(format!(
            "Backend: {}",
            if using_gpu { "GPU (CUDA)" } else { "CPU" }
        ));
    }

    fn environment_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Environment");

        ui.horizontal(|ui| {
            ui.label("Temperature (°C):");
            ui.add(egui::Slider::new(&mut self.state.temperature_c, -20.0..=40.0));
        });

        ui.horizontal(|ui| {
            ui.label("Humidity (%):");
            ui.add(egui::Slider::new(&mut self.state.humidity_percent, 0.0..=100.0));
        });

        ui.horizontal(|ui| {
            ui.label("Medium:");
            egui::ComboBox::from_id_source("medium_select")
                .selected_text(match self.state.medium {
                    Medium::Air => "Air",
                    Medium::Water => "Water",
                    Medium::Steel => "Steel",
                    Medium::Aluminum => "Aluminum",
                    Medium::Custom => "Custom",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.state.medium, Medium::Air, "Air");
                    ui.selectable_value(&mut self.state.medium, Medium::Water, "Water");
                    ui.selectable_value(&mut self.state.medium, Medium::Steel, "Steel");
                    ui.selectable_value(&mut self.state.medium, Medium::Aluminum, "Aluminum");
                });
        });
    }

    fn source_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Audio Source");

        ui.horizontal(|ui| {
            ui.label("Type:");
            egui::ComboBox::from_id_source("source_type")
                .selected_text(self.state.source_type_name())
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.state.source_type_idx, 0, "Impulse");
                    ui.selectable_value(&mut self.state.source_type_idx, 1, "Tone");
                    ui.selectable_value(&mut self.state.source_type_idx, 2, "Chirp");
                    ui.selectable_value(&mut self.state.source_type_idx, 3, "White Noise");
                    ui.selectable_value(&mut self.state.source_type_idx, 4, "Pink Noise");
                    ui.selectable_value(&mut self.state.source_type_idx, 5, "Gaussian Pulse");
                });
        });

        ui.label("Position:");
        ui.horizontal(|ui| {
            ui.label("X:");
            ui.add(egui::Slider::new(&mut self.state.source_x, 0.0..=1.0).text(""));
        });
        ui.horizontal(|ui| {
            ui.label("Y:");
            ui.add(egui::Slider::new(&mut self.state.source_y, 0.0..=1.0).text(""));
        });
        ui.horizontal(|ui| {
            ui.label("Z:");
            ui.add(egui::Slider::new(&mut self.state.source_z, 0.0..=1.0).text(""));
        });

        ui.horizontal(|ui| {
            ui.label("Amplitude:");
            ui.add(egui::Slider::new(&mut self.state.source_amplitude, 0.1..=5.0));
        });

        if self.state.source_type_idx == 1 {
            // Tone
            ui.horizontal(|ui| {
                ui.label("Frequency (Hz):");
                ui.add(egui::Slider::new(&mut self.state.source_frequency, 20.0..=2000.0).logarithmic(true));
            });
        }

        if ui.button("🔊 Fire Source").clicked() {
            // Signal to main loop to inject source
        }
    }

    fn listener_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Virtual Listener");

        ui.checkbox(&mut self.state.show_listener, "Show listener");

        ui.label("Position:");
        ui.horizontal(|ui| {
            ui.label("X:");
            ui.add(egui::Slider::new(&mut self.state.listener_x, 0.0..=1.0).text(""));
        });
        ui.horizontal(|ui| {
            ui.label("Y:");
            ui.add(egui::Slider::new(&mut self.state.listener_y, 0.0..=1.0).text(""));
        });
        ui.horizontal(|ui| {
            ui.label("Z:");
            ui.add(egui::Slider::new(&mut self.state.listener_z, 0.0..=1.0).text(""));
        });

        ui.horizontal(|ui| {
            ui.label("Yaw (°):");
            ui.add(egui::Slider::new(&mut self.state.listener_yaw, -180.0..=180.0));
        });
    }

    fn visualization_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Visualization");

        ui.horizontal(|ui| {
            ui.label("Color map:");
            egui::ComboBox::from_id_source("color_map")
                .selected_text(match self.state.color_map_idx {
                    0 => "Blue-White-Red",
                    1 => "Cool-Warm",
                    2 => "Hot",
                    3 => "Diverging",
                    _ => "Grayscale",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.state.color_map_idx, 0, "Blue-White-Red");
                    ui.selectable_value(&mut self.state.color_map_idx, 1, "Cool-Warm");
                    ui.selectable_value(&mut self.state.color_map_idx, 2, "Hot");
                    ui.selectable_value(&mut self.state.color_map_idx, 3, "Diverging");
                    ui.selectable_value(&mut self.state.color_map_idx, 4, "Grayscale");
                });
        });

        ui.checkbox(&mut self.state.auto_scale, "Auto-scale colors");

        if !self.state.auto_scale {
            ui.horizontal(|ui| {
                ui.label("Max pressure:");
                ui.add(egui::Slider::new(&mut self.state.max_pressure, 0.1..=10.0).logarithmic(true));
            });
        }

        ui.horizontal(|ui| {
            ui.label("Slice opacity:");
            ui.add(egui::Slider::new(&mut self.state.slice_opacity, 0.1..=1.0));
        });

        ui.checkbox(&mut self.state.show_bounding_box, "Show bounding box");
        ui.checkbox(&mut self.state.show_floor_grid, "Show floor grid");
        ui.checkbox(&mut self.state.show_sources, "Show source markers");
        ui.checkbox(&mut self.state.show_info_panel, "Show info panel");
    }

    fn slice_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Slices");

        ui.horizontal(|ui| {
            ui.checkbox(&mut self.state.show_xy_slice, "XY");
            if self.state.show_xy_slice {
                ui.add(egui::Slider::new(&mut self.state.xy_slice_position, 0.0..=1.0).text("Z"));
            }
        });

        ui.horizontal(|ui| {
            ui.checkbox(&mut self.state.show_xz_slice, "XZ");
            if self.state.show_xz_slice {
                ui.add(egui::Slider::new(&mut self.state.xz_slice_position, 0.0..=1.0).text("Y"));
            }
        });

        ui.horizontal(|ui| {
            ui.checkbox(&mut self.state.show_yz_slice, "YZ");
            if self.state.show_yz_slice {
                ui.add(egui::Slider::new(&mut self.state.yz_slice_position, 0.0..=1.0).text("X"));
            }
        });
    }

    fn info_panel(&self, ui: &mut egui::Ui, engine: &SimulationEngine) {
        let (w, h, d) = engine.dimensions();
        let (pw, ph, pd) = engine.grid.physical_size();

        ui.label(format!("Grid: {}×{}×{} cells", w, h, d));
        ui.label(format!("Size: {:.1}×{:.1}×{:.1} m", pw, ph, pd));
        ui.label(format!("Step: {}", engine.step_count()));
        ui.label(format!("Time: {:.4} s", engine.time()));
        ui.label(format!("Max pressure: {:.4}", engine.grid.max_pressure()));
        ui.label(format!("Total energy: {:.4}", engine.grid.total_energy()));

        let speed = engine.grid.params.medium.speed_of_sound;
        ui.label(format!("Speed of sound: {:.1} m/s", speed));
    }
}

impl Default for SimulationControls {
    fn default() -> Self {
        Self::new()
    }
}
