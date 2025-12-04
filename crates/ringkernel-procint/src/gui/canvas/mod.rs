//! Canvas rendering for process visualization.
//!
//! Provides force-directed DFG and timeline views.

mod dfg_canvas;
mod timeline_canvas;
mod token_animation;

pub use dfg_canvas::*;
pub use timeline_canvas::*;
pub use token_animation::*;

use eframe::egui::{Pos2, Vec2};

/// Camera for canvas panning and zooming.
#[derive(Debug, Clone)]
pub struct Camera {
    /// Pan offset.
    pub offset: Vec2,
    /// Zoom level.
    pub zoom: f32,
    /// Target zoom (for smooth animation).
    target_zoom: f32,
    /// Target offset.
    target_offset: Vec2,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            offset: Vec2::ZERO,
            zoom: 1.0,
            target_zoom: 1.0,
            target_offset: Vec2::ZERO,
        }
    }
}

impl Camera {
    /// Transform world position to screen position.
    pub fn world_to_screen(&self, pos: Pos2, canvas_center: Pos2) -> Pos2 {
        let transformed = (pos.to_vec2() + self.offset) * self.zoom;
        canvas_center + transformed
    }

    /// Transform screen position to world position.
    pub fn screen_to_world(&self, pos: Pos2, canvas_center: Pos2) -> Pos2 {
        let relative = pos - canvas_center;
        Pos2::ZERO + (relative / self.zoom) - self.offset
    }

    /// Handle zoom input.
    pub fn zoom_by(&mut self, delta: f32, around: Pos2, canvas_center: Pos2) {
        let world_pos = self.screen_to_world(around, canvas_center);
        self.target_zoom = (self.target_zoom * (1.0 + delta * 0.1)).clamp(0.1, 5.0);

        // Adjust offset to zoom around cursor
        let new_screen = self.world_to_screen(world_pos, canvas_center);
        self.target_offset += (around - new_screen) / self.target_zoom;
    }

    /// Handle pan input.
    pub fn pan_by(&mut self, delta: Vec2) {
        self.target_offset += delta / self.zoom;
    }

    /// Reset camera.
    pub fn reset(&mut self) {
        self.target_zoom = 1.0;
        self.target_offset = Vec2::ZERO;
    }

    /// Update animation (call each frame).
    pub fn update(&mut self, dt: f32) {
        let lerp_speed = 8.0 * dt;
        self.zoom += (self.target_zoom - self.zoom) * lerp_speed;
        self.offset += (self.target_offset - self.offset) * lerp_speed;
    }
}

/// Position in the force-directed layout.
#[derive(Debug, Clone, Copy)]
pub struct NodePosition {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
}

impl Default for NodePosition {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
        }
    }
}

impl NodePosition {
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            vx: 0.0,
            vy: 0.0,
        }
    }

    pub fn pos(&self) -> Pos2 {
        Pos2::new(self.x, self.y)
    }
}
