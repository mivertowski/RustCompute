//! 3D camera system for visualization.
//!
//! Provides orbital camera controls similar to 3D modeling software.

use glam::{Mat4, Vec3};

/// 3D camera with perspective projection.
#[derive(Debug, Clone)]
pub struct Camera3D {
    /// Camera position in world space
    position: Vec3,
    /// Point the camera is looking at
    target: Vec3,
    /// Up vector
    up: Vec3,
    /// Field of view in radians
    fov: f32,
    /// Aspect ratio (width / height)
    aspect: f32,
    /// Near clipping plane
    near: f32,
    /// Far clipping plane
    far: f32,
}

impl Default for Camera3D {
    fn default() -> Self {
        Self {
            position: Vec3::new(5.0, 3.0, 5.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov: std::f32::consts::FRAC_PI_4, // 45 degrees
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl Camera3D {
    /// Create a new camera looking at the origin.
    pub fn new(position: Vec3, target: Vec3) -> Self {
        Self {
            position,
            target,
            ..Default::default()
        }
    }

    /// Create a camera positioned for a simulation grid.
    pub fn for_grid(grid_size: (f32, f32, f32)) -> Self {
        let center = Vec3::new(grid_size.0 / 2.0, grid_size.1 / 2.0, grid_size.2 / 2.0);
        let distance = (grid_size.0.max(grid_size.2) * 1.5).max(5.0);

        Self {
            position: center + Vec3::new(distance, distance * 0.7, distance),
            target: center,
            ..Default::default()
        }
    }

    /// Set the aspect ratio.
    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }

    /// Set the field of view in degrees.
    pub fn set_fov_degrees(&mut self, degrees: f32) {
        self.fov = degrees.to_radians();
    }

    /// Get the camera position.
    pub fn position(&self) -> Vec3 {
        self.position
    }

    /// Get the target position.
    pub fn target(&self) -> Vec3 {
        self.target
    }

    /// Set the camera position.
    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }

    /// Set the target position.
    pub fn set_target(&mut self, target: Vec3) {
        self.target = target;
    }

    /// Move the camera and target by a delta.
    pub fn translate(&mut self, delta: Vec3) {
        self.position += delta;
        self.target += delta;
    }

    /// Get the view matrix.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    /// Get the projection matrix.
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far)
    }

    /// Get the combined view-projection matrix.
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// Get the forward direction (normalized).
    pub fn forward(&self) -> Vec3 {
        (self.target - self.position).normalize()
    }

    /// Get the right direction (normalized).
    pub fn right(&self) -> Vec3 {
        self.forward().cross(self.up).normalize()
    }

    /// Get distance to target.
    pub fn distance_to_target(&self) -> f32 {
        (self.position - self.target).length()
    }
}

/// Controller for orbital camera movement.
pub struct CameraController {
    /// Rotation speed (radians per pixel)
    pub rotate_speed: f32,
    /// Pan speed (units per pixel)
    pub pan_speed: f32,
    /// Zoom speed (multiplier per scroll unit)
    pub zoom_speed: f32,
    /// Minimum zoom distance
    pub min_distance: f32,
    /// Maximum zoom distance
    pub max_distance: f32,
    /// Current orbital angles (theta, phi)
    theta: f32,
    phi: f32,
    /// Current distance from target
    distance: f32,
    /// Mouse state
    is_rotating: bool,
    is_panning: bool,
    last_mouse: (f32, f32),
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            rotate_speed: 0.005,
            pan_speed: 0.01,
            zoom_speed: 0.1,
            min_distance: 0.5,
            max_distance: 500.0,
            theta: std::f32::consts::FRAC_PI_4,
            phi: std::f32::consts::FRAC_PI_4,
            distance: 10.0,
            is_rotating: false,
            is_panning: false,
            last_mouse: (0.0, 0.0),
        }
    }
}

impl CameraController {
    /// Create a new camera controller.
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize the controller from an existing camera.
    pub fn from_camera(camera: &Camera3D) -> Self {
        let dir = camera.position - camera.target;
        let distance = dir.length();

        let theta = dir.x.atan2(dir.z);
        let phi = (dir.y / distance).asin();

        Self {
            distance,
            theta,
            phi,
            ..Default::default()
        }
    }

    /// Handle mouse button press.
    pub fn on_mouse_down(&mut self, button: MouseButton, x: f32, y: f32) {
        self.last_mouse = (x, y);
        match button {
            MouseButton::Left => self.is_rotating = true,
            MouseButton::Middle => self.is_panning = true,
            MouseButton::Right => self.is_panning = true,
        }
    }

    /// Handle mouse button release.
    pub fn on_mouse_up(&mut self, button: MouseButton) {
        match button {
            MouseButton::Left => self.is_rotating = false,
            MouseButton::Middle => self.is_panning = false,
            MouseButton::Right => self.is_panning = false,
        }
    }

    /// Handle mouse movement.
    pub fn on_mouse_move(&mut self, x: f32, y: f32, camera: &mut Camera3D) {
        let dx = x - self.last_mouse.0;
        let dy = y - self.last_mouse.1;
        self.last_mouse = (x, y);

        if self.is_rotating {
            self.theta -= dx * self.rotate_speed;
            self.phi = (self.phi + dy * self.rotate_speed).clamp(
                -std::f32::consts::FRAC_PI_2 + 0.01,
                std::f32::consts::FRAC_PI_2 - 0.01,
            );
            self.update_camera(camera);
        }

        if self.is_panning {
            let right = camera.right();
            let up = camera.up;
            let pan = (right * -dx + up * dy) * self.pan_speed * self.distance * 0.01;
            camera.translate(pan);
        }
    }

    /// Handle scroll/zoom.
    pub fn on_scroll(&mut self, delta: f32, camera: &mut Camera3D) {
        self.distance *= 1.0 - delta * self.zoom_speed;
        self.distance = self.distance.clamp(self.min_distance, self.max_distance);
        self.update_camera(camera);
    }

    /// Update camera position from orbital parameters.
    pub fn update_camera(&self, camera: &mut Camera3D) {
        let x = self.distance * self.phi.cos() * self.theta.sin();
        let y = self.distance * self.phi.sin();
        let z = self.distance * self.phi.cos() * self.theta.cos();

        camera.position = camera.target + Vec3::new(x, y, z);
    }

    /// Reset to default view.
    pub fn reset(&mut self, camera: &mut Camera3D, grid_size: (f32, f32, f32)) {
        let center = Vec3::new(grid_size.0 / 2.0, grid_size.1 / 2.0, grid_size.2 / 2.0);
        camera.target = center;

        self.theta = std::f32::consts::FRAC_PI_4;
        self.phi = std::f32::consts::FRAC_PI_4 * 0.5;
        self.distance = (grid_size.0.max(grid_size.2) * 1.5).max(5.0);

        self.update_camera(camera);
    }

    /// Set orbital angles directly.
    pub fn set_orbit(&mut self, theta: f32, phi: f32, camera: &mut Camera3D) {
        self.theta = theta;
        self.phi = phi.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
        self.update_camera(camera);
    }

    /// Set distance to target.
    pub fn set_distance(&mut self, distance: f32, camera: &mut Camera3D) {
        self.distance = distance.clamp(self.min_distance, self.max_distance);
        self.update_camera(camera);
    }

    /// Get current orbital parameters.
    pub fn orbit_params(&self) -> (f32, f32, f32) {
        (self.theta, self.phi, self.distance)
    }
}

/// Mouse button identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Middle,
    Right,
}

/// First-person camera controller for immersive view.
pub struct FirstPersonController {
    /// Movement speed (units per second)
    pub move_speed: f32,
    /// Look sensitivity
    pub sensitivity: f32,
    /// Current pitch (up/down rotation)
    pitch: f32,
    /// Current yaw (left/right rotation)
    yaw: f32,
    /// Movement input state
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
}

impl Default for FirstPersonController {
    fn default() -> Self {
        Self {
            move_speed: 5.0,
            sensitivity: 0.003,
            pitch: 0.0,
            yaw: 0.0,
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
        }
    }
}

impl FirstPersonController {
    pub fn new() -> Self {
        Self::default()
    }

    /// Handle key press.
    pub fn on_key_down(&mut self, key: Key) {
        match key {
            Key::W => self.forward = true,
            Key::S => self.backward = true,
            Key::A => self.left = true,
            Key::D => self.right = true,
            Key::Space => self.up = true,
            Key::Shift => self.down = true,
            _ => {}
        }
    }

    /// Handle key release.
    pub fn on_key_up(&mut self, key: Key) {
        match key {
            Key::W => self.forward = false,
            Key::S => self.backward = false,
            Key::A => self.left = false,
            Key::D => self.right = false,
            Key::Space => self.up = false,
            Key::Shift => self.down = false,
            _ => {}
        }
    }

    /// Handle mouse movement for look.
    pub fn on_mouse_move(&mut self, dx: f32, dy: f32, camera: &mut Camera3D) {
        self.yaw -= dx * self.sensitivity;
        self.pitch = (self.pitch - dy * self.sensitivity).clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );

        self.update_camera_direction(camera);
    }

    /// Update camera each frame.
    pub fn update(&self, dt: f32, camera: &mut Camera3D) {
        let forward = camera.forward();
        let right = camera.right();

        let mut movement = Vec3::ZERO;

        if self.forward {
            movement += forward;
        }
        if self.backward {
            movement -= forward;
        }
        if self.right {
            movement += right;
        }
        if self.left {
            movement -= right;
        }
        if self.up {
            movement += Vec3::Y;
        }
        if self.down {
            movement -= Vec3::Y;
        }

        if movement.length_squared() > 0.0 {
            movement = movement.normalize() * self.move_speed * dt;
            camera.position += movement;
            camera.target += movement;
        }
    }

    fn update_camera_direction(&self, camera: &mut Camera3D) {
        let direction = Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize();

        camera.target = camera.position + direction;
    }
}

/// Key identifiers for first-person controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    W,
    A,
    S,
    D,
    Space,
    Shift,
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_creation() {
        let camera = Camera3D::default();
        assert!(camera.position.length() > 0.0);
    }

    #[test]
    fn test_camera_matrices() {
        let camera = Camera3D::default();

        let view = camera.view_matrix();
        let proj = camera.projection_matrix();
        let view_proj = camera.view_projection_matrix();

        // View-projection should be the product
        let expected = proj * view;
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (view_proj.col(i)[j] - expected.col(i)[j]).abs() < 0.001,
                    "Mismatch at [{}, {}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_camera_controller() {
        let mut camera = Camera3D::default();
        let mut controller = CameraController::from_camera(&camera);

        let initial_distance = camera.distance_to_target();

        // Zoom in
        controller.on_scroll(1.0, &mut camera);
        assert!(camera.distance_to_target() < initial_distance);
    }

    #[test]
    fn test_for_grid() {
        let camera = Camera3D::for_grid((10.0, 5.0, 10.0));

        // Camera should be looking at approximately the center
        let center = Vec3::new(5.0, 2.5, 5.0);
        let diff = (camera.target - center).length();
        assert!(diff < 0.1);
    }
}
