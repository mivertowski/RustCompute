//! Control panel widgets.

use super::app::{ComputeBackend, DrawMode, Message};
use crate::simulation::SimulationMode;
use iced::widget::{button, column, container, pick_list, row, slider, text, text_input, vertical_space};
use iced::{Alignment, Element, Length};

/// Build the controls panel.
#[allow(clippy::too_many_arguments)]
pub fn view_controls(
    is_running: bool,
    speed_of_sound: f32,
    cell_size: f32,
    grid_width_input: &str,
    grid_height_input: &str,
    impulse_amplitude: f32,
    compute_backend: ComputeBackend,
    draw_mode: DrawMode,
    simulation_mode: SimulationMode,
    show_stats: bool,
    fps: f32,
    steps_per_sec: f32,
    throughput: f64,
    cell_count: usize,
    courant: f32,
    max_pressure: f32,
    total_energy: f32,
) -> Element<'static, Message> {
    // Play/Pause and Step buttons
    let play_text = if is_running { "Pause" } else { "Play" };
    let play_btn = button(text(play_text).size(16))
        .on_press(Message::ToggleRunning)
        .width(80);

    let step_btn = button(text("Step").size(16))
        .on_press(Message::Step)
        .width(80);

    let reset_btn = button(text("Reset").size(16))
        .on_press(Message::Reset)
        .width(80);

    let control_buttons = row![play_btn, step_btn, reset_btn]
        .spacing(10)
        .align_y(Alignment::Center);

    // Speed of sound slider
    let speed_section = column![
        text(format!("Speed of Sound: {:.0} m/s", speed_of_sound)).size(14),
        slider(10.0..=1000.0, speed_of_sound, Message::SpeedChanged).width(Length::Fill),
        text("(343 m/s in air)").size(11),
    ]
    .spacing(5);

    // Cell size slider
    let cell_size_section = column![
        text(format!("Cell Size: {:.2} m", cell_size)).size(14),
        slider(0.01..=10.0, cell_size, Message::CellSizeChanged).width(Length::Fill),
        text("(spatial resolution)").size(11),
    ]
    .spacing(5);

    // Grid size inputs
    let width_input = text_input("Width", grid_width_input)
        .on_input(Message::GridWidthChanged)
        .width(60)
        .size(14);

    let height_input = text_input("Height", grid_height_input)
        .on_input(Message::GridHeightChanged)
        .width(60)
        .size(14);

    let apply_btn = button(text("Apply").size(14))
        .on_press(Message::ApplyGridSize)
        .width(60);

    let grid_section = column![
        text("Grid Size").size(14),
        row![width_input, text(" x ").size(14), height_input, apply_btn]
            .spacing(5)
            .align_y(Alignment::Center),
        text("(4-256 cells)").size(11),
    ]
    .spacing(5);

    // Impulse amplitude slider
    let impulse_section = column![
        text(format!("Impulse Amplitude: {:.2}", impulse_amplitude)).size(14),
        slider(0.1..=5.0, impulse_amplitude, Message::ImpulseAmplitudeChanged).width(Length::Fill),
        text("(click on grid to inject)").size(11),
    ]
    .spacing(5);

    // Backend picker
    #[cfg(feature = "cuda")]
    let backend_options: Vec<ComputeBackend> = vec![
        ComputeBackend::Cpu,
        ComputeBackend::GpuActor,
        ComputeBackend::CudaPacked,
    ];
    #[cfg(not(feature = "cuda"))]
    let backend_options: Vec<ComputeBackend> = vec![
        ComputeBackend::Cpu,
        ComputeBackend::GpuActor,
    ];

    let backend_section = column![
        text("Compute Backend").size(14),
        pick_list(backend_options, Some(compute_backend), Message::ComputeBackendChanged).width(Length::Fill),
        text("(CPU, GPU Actor, or CUDA Packed)").size(11),
    ]
    .spacing(5);

    // Draw mode picker
    let draw_mode_options: Vec<DrawMode> = vec![
        DrawMode::Impulse,
        DrawMode::Absorber,
        DrawMode::Reflector,
        DrawMode::Erase,
    ];

    let clear_btn = button(text("Clear All").size(12))
        .on_press(Message::ClearCellTypes)
        .width(80);

    let draw_mode_section = column![
        text("Draw Mode").size(14),
        row![
            pick_list(draw_mode_options, Some(draw_mode), Message::DrawModeChanged).width(Length::FillPortion(2)),
            clear_btn,
        ].spacing(5),
        text(match draw_mode {
            DrawMode::Impulse => "(Click to inject wave)",
            DrawMode::Absorber => "(Click to place absorber)",
            DrawMode::Reflector => "(Click to place reflector)",
            DrawMode::Erase => "(Click to remove cell type)",
        }).size(11),
    ]
    .spacing(5);

    // Simulation mode picker (educational modes)
    let sim_mode_options: Vec<SimulationMode> = SimulationMode::all().to_vec();

    let sim_mode_section = column![
        text("Simulation Mode").size(14),
        pick_list(sim_mode_options, Some(simulation_mode), Message::SimulationModeChanged).width(Length::Fill),
        text(simulation_mode.description()).size(11),
    ]
    .spacing(5);

    // Stats toggle button
    let stats_btn = button(text(if show_stats { "Hide Stats" } else { "Show Stats" }).size(14))
        .on_press(Message::ToggleStats)
        .width(Length::Fill);

    // Format throughput nicely
    let throughput_str = if throughput >= 1_000_000_000.0 {
        format!("{:.2} G", throughput / 1_000_000_000.0)
    } else if throughput >= 1_000_000.0 {
        format!("{:.2} M", throughput / 1_000_000.0)
    } else if throughput >= 1_000.0 {
        format!("{:.2} K", throughput / 1_000.0)
    } else {
        format!("{:.0}", throughput)
    };

    // Format steps per second nicely
    let steps_str = if steps_per_sec >= 1_000_000.0 {
        format!("{:.2} M", steps_per_sec / 1_000_000.0)
    } else if steps_per_sec >= 1_000.0 {
        format!("{:.1} K", steps_per_sec / 1_000.0)
    } else {
        format!("{:.0}", steps_per_sec)
    };

    // Stats panel (conditionally shown)
    let stats_panel = if show_stats {
        column![
            text("Performance").size(16),
            text(format!("FPS: {:.1}", fps)).size(12),
            text(format!("Steps/sec: {}", steps_str)).size(12),
            text(format!("Throughput: {} cells/s", throughput_str)).size(12),
            vertical_space().height(5),
            text("Simulation").size(16),
            text(format!("Cells: {}", cell_count)).size(12),
            text(format!("Courant #: {:.3}", courant)).size(12),
            text(format!("Max |p|: {:.4}", max_pressure)).size(12),
            text(format!("Energy: {:.4}", total_energy)).size(12),
        ]
        .spacing(3)
    } else {
        column![]
    };

    // Instructions
    let instructions = column![
        text("Instructions").size(16),
        text("Click to draw (based on mode)").size(12),
        text("Use Absorber/Reflector modes").size(12),
        text("to create obstacles and walls").size(12),
        text("Play/Pause to run simulation").size(12),
        text("(Works best in CPU mode)").size(11),
    ]
    .spacing(3);

    // Color legend
    let legend = column![
        text("Color Legend").size(16),
        row![
            container(text("").size(12))
                .width(20)
                .height(20)
                .style(|_theme| container::Style {
                    background: Some(iced::Color::from_rgb(0.0, 1.0, 1.0).into()),
                    ..Default::default()
                }),
            text(" Negative pressure (-)").size(12),
        ]
        .spacing(5),
        row![
            container(text("").size(12))
                .width(20)
                .height(20)
                .style(|_theme| container::Style {
                    background: Some(iced::Color::from_rgb(0.05, 0.05, 0.05).into()),
                    ..Default::default()
                }),
            text(" Zero pressure").size(12),
        ]
        .spacing(5),
        row![
            container(text("").size(12))
                .width(20)
                .height(20)
                .style(|_theme| container::Style {
                    background: Some(iced::Color::from_rgb(1.0, 1.0, 0.0).into()),
                    ..Default::default()
                }),
            text(" Positive pressure (+)").size(12),
        ]
        .spacing(5),
        row![
            container(text("").size(12))
                .width(20)
                .height(20)
                .style(|_theme| container::Style {
                    background: Some(iced::Color::from_rgb(0.2, 0.0, 0.3).into()),
                    ..Default::default()
                }),
            text(" Absorber (purple)").size(12),
        ]
        .spacing(5),
        row![
            container(text("").size(12))
                .width(20)
                .height(20)
                .style(|_theme| container::Style {
                    background: Some(iced::Color::from_rgb(0.7, 0.7, 0.7).into()),
                    ..Default::default()
                }),
            text(" Reflector (gray)").size(12),
        ]
        .spacing(5),
    ]
    .spacing(5);

    // Main controls column
    column![
        text("WaveSim Controls").size(20),
        vertical_space().height(10),
        control_buttons,
        vertical_space().height(15),
        speed_section,
        vertical_space().height(15),
        cell_size_section,
        vertical_space().height(15),
        grid_section,
        vertical_space().height(15),
        impulse_section,
        vertical_space().height(15),
        backend_section,
        vertical_space().height(15),
        draw_mode_section,
        vertical_space().height(15),
        sim_mode_section,
        vertical_space().height(15),
        stats_btn,
        stats_panel,
        vertical_space().height(15),
        instructions,
        vertical_space().height(15),
        legend,
    ]
    .spacing(5)
    .width(Length::Fill)
    .into()
}
