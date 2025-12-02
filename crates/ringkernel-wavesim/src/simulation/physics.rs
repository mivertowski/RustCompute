//! Acoustic simulation parameters and physics calculations.

/// Parameters for the acoustic wave simulation.
///
/// Uses the FDTD (Finite-Difference Time-Domain) method for solving
/// the 2D wave equation.
#[derive(Debug, Clone)]
pub struct AcousticParams {
    /// Speed of sound in m/s (adjustable via UI slider).
    pub speed_of_sound: f32,

    /// Cell size (spatial step) in meters.
    pub cell_size: f32,

    /// Time step in seconds (computed for numerical stability).
    pub time_step: f32,

    /// Damping factor (energy loss per step, 0-1).
    pub damping: f32,
}

impl Default for AcousticParams {
    fn default() -> Self {
        Self::new(343.0, 1.0)
    }
}

impl AcousticParams {
    /// Create new acoustic parameters.
    ///
    /// The time step is automatically computed to satisfy the CFL condition
    /// for numerical stability: dt <= dx / (c * sqrt(2))
    ///
    /// # Arguments
    /// * `speed_of_sound` - Speed of sound in m/s (343 m/s in air at 20°C)
    /// * `cell_size` - Spatial step size in meters
    pub fn new(speed_of_sound: f32, cell_size: f32) -> Self {
        // CFL condition for 2D wave equation: c * dt / dx <= 1/sqrt(2)
        // We use a safety factor of 1.5 instead of sqrt(2) ~= 1.414
        let dt = cell_size / (speed_of_sound * 1.5);

        Self {
            speed_of_sound,
            cell_size,
            time_step: dt,
            damping: 0.001,
        }
    }

    /// Create parameters with custom damping.
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping.clamp(0.0, 1.0);
        self
    }

    /// Compute the Courant number (c * dt / dx).
    ///
    /// For numerical stability in 2D, this should be <= 1/sqrt(2) ~= 0.707
    pub fn courant_number(&self) -> f32 {
        self.speed_of_sound * self.time_step / self.cell_size
    }

    /// Check if the parameters satisfy the CFL stability condition.
    pub fn is_stable(&self) -> bool {
        self.courant_number() <= 1.0 / std::f32::consts::SQRT_2
    }

    /// Update speed of sound and recompute time step for stability.
    pub fn set_speed_of_sound(&mut self, speed: f32) {
        self.speed_of_sound = speed.max(1.0); // Minimum speed
        self.time_step = self.cell_size / (self.speed_of_sound * 1.5);
    }

    /// Update cell size and recompute time step for stability.
    pub fn set_cell_size(&mut self, size: f32) {
        self.cell_size = size.max(0.001); // Minimum 1mm
        self.time_step = self.cell_size / (self.speed_of_sound * 1.5);
    }

    /// Get the wavelength for a given frequency.
    pub fn wavelength(&self, frequency: f32) -> f32 {
        self.speed_of_sound / frequency
    }

    /// Get the number of cells per wavelength for a given frequency.
    pub fn cells_per_wavelength(&self, frequency: f32) -> f32 {
        self.wavelength(frequency) / self.cell_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = AcousticParams::default();
        assert_eq!(params.speed_of_sound, 343.0);
        assert_eq!(params.cell_size, 1.0);
    }

    #[test]
    fn test_stability() {
        let params = AcousticParams::new(343.0, 1.0);
        assert!(params.is_stable(), "Default parameters should be stable");
        assert!(
            params.courant_number() <= 1.0 / std::f32::consts::SQRT_2,
            "Courant number should satisfy CFL condition"
        );
    }

    #[test]
    fn test_speed_change() {
        let mut params = AcousticParams::new(343.0, 1.0);
        params.set_speed_of_sound(1000.0);
        assert_eq!(params.speed_of_sound, 1000.0);
        assert!(params.is_stable(), "Should remain stable after speed change");
    }

    #[test]
    fn test_wavelength() {
        let params = AcousticParams::new(343.0, 1.0);
        // At 343 Hz, wavelength should be ~1 meter
        let wavelength = params.wavelength(343.0);
        assert!((wavelength - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_courant_number() {
        let params = AcousticParams::new(343.0, 1.0);
        let c = params.courant_number();
        // With safety factor of 1.5, courant number should be 1/1.5 ~= 0.667
        assert!((c - 1.0 / 1.5).abs() < 0.01);
    }
}
