//! Realistic 3D acoustic physics simulation parameters.
//!
//! Implements accurate sound propagation models including:
//! - Temperature-dependent speed of sound
//! - Humidity-dependent absorption
//! - Frequency-dependent atmospheric damping (ISO 9613-1)
//! - Multiple propagation media (air, water, metal)


/// Physical constants for acoustic simulations.
pub mod constants {
    /// Speed of sound at 0°C in dry air (m/s)
    pub const SPEED_OF_SOUND_0C: f32 = 331.3;

    /// Standard atmospheric pressure (Pa)
    pub const STANDARD_PRESSURE: f32 = 101325.0;

    /// Reference temperature for calculations (°C)
    pub const REFERENCE_TEMP_C: f32 = 20.0;

    /// Molar mass of dry air (kg/mol)
    pub const MOLAR_MASS_AIR: f32 = 0.02897;

    /// Molar mass of water vapor (kg/mol)
    pub const MOLAR_MASS_WATER: f32 = 0.01802;

    /// Universal gas constant (J/(mol·K))
    pub const GAS_CONSTANT: f32 = 8.314;

    /// Average ear spacing for binaural audio (m)
    pub const EAR_SPACING: f32 = 0.17;

    /// Speed of sound in water at 20°C (m/s)
    pub const SPEED_IN_WATER: f32 = 1481.0;

    /// Speed of sound in steel (m/s)
    pub const SPEED_IN_STEEL: f32 = 5960.0;

    /// Speed of sound in aluminum (m/s)
    pub const SPEED_IN_ALUMINUM: f32 = 6420.0;
}

/// Propagation medium types with distinct physical properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Medium {
    /// Air at given temperature and humidity
    Air,
    /// Fresh water
    Water,
    /// Steel (solid metal)
    Steel,
    /// Aluminum (lightweight metal)
    Aluminum,
    /// Custom medium with user-defined properties
    Custom,
}

impl Default for Medium {
    fn default() -> Self {
        Medium::Air
    }
}

/// Properties of a propagation medium.
#[derive(Debug, Clone)]
pub struct MediumProperties {
    /// Speed of sound (m/s)
    pub speed_of_sound: f32,
    /// Density (kg/m³)
    pub density: f32,
    /// Base damping coefficient (linear absorption per meter)
    pub base_damping: f32,
    /// Whether frequency-dependent absorption applies
    pub frequency_dependent: bool,
    /// Characteristic impedance (Pa·s/m)
    pub impedance: f32,
}

impl MediumProperties {
    /// Create air properties at given temperature and humidity.
    pub fn air(temperature_c: f32, humidity_percent: f32) -> Self {
        let speed = speed_of_sound_in_air(temperature_c, humidity_percent);
        // Air density varies with temperature: ρ = P·M / (R·T)
        let temp_k = temperature_c + 273.15;
        let density = constants::STANDARD_PRESSURE * constants::MOLAR_MASS_AIR
            / (constants::GAS_CONSTANT * temp_k);

        Self {
            speed_of_sound: speed,
            density,
            base_damping: 0.0001, // Very low base damping, frequency-dependent takes over
            frequency_dependent: true,
            impedance: density * speed,
        }
    }

    /// Create water properties at given temperature.
    pub fn water(temperature_c: f32) -> Self {
        // Speed in freshwater varies with temperature (simplified Bilaniuk-Wong formula)
        // At 20°C, speed should be ~1481 m/s
        // c ≈ 1402.7 + 5.0 * T - 0.055 * T² + 0.00022 * T³
        let t = temperature_c;
        let speed = 1402.7 + 5.0 * t - 0.055 * t * t + 0.00022 * t * t * t;
        let density = 998.0 - 0.05 * (temperature_c - 20.0); // Approximate

        Self {
            speed_of_sound: speed,
            density,
            base_damping: 0.00002, // Water has very low absorption
            frequency_dependent: true,
            impedance: density * speed,
        }
    }

    /// Create steel properties.
    pub fn steel() -> Self {
        Self {
            speed_of_sound: constants::SPEED_IN_STEEL,
            density: 7850.0,
            base_damping: 0.00001, // Metals have minimal damping
            frequency_dependent: false,
            impedance: 7850.0 * constants::SPEED_IN_STEEL,
        }
    }

    /// Create aluminum properties.
    pub fn aluminum() -> Self {
        Self {
            speed_of_sound: constants::SPEED_IN_ALUMINUM,
            density: 2700.0,
            base_damping: 0.000015,
            frequency_dependent: false,
            impedance: 2700.0 * constants::SPEED_IN_ALUMINUM,
        }
    }

    /// Create custom medium with specified properties.
    pub fn custom(speed: f32, density: f32, damping: f32) -> Self {
        Self {
            speed_of_sound: speed,
            density,
            base_damping: damping,
            frequency_dependent: false,
            impedance: density * speed,
        }
    }
}

/// Environmental conditions affecting sound propagation.
#[derive(Debug, Clone)]
pub struct Environment {
    /// Temperature in Celsius
    pub temperature_c: f32,
    /// Relative humidity (0-100%)
    pub humidity_percent: f32,
    /// Atmospheric pressure in Pascals
    pub pressure_pa: f32,
    /// Current medium type
    pub medium: Medium,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            temperature_c: 20.0,
            humidity_percent: 50.0,
            pressure_pa: constants::STANDARD_PRESSURE,
            medium: Medium::Air,
        }
    }
}

impl Environment {
    /// Create a new environment with air at standard conditions.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set temperature in Celsius.
    pub fn with_temperature(mut self, temp_c: f32) -> Self {
        self.temperature_c = temp_c.clamp(-40.0, 60.0);
        self
    }

    /// Set relative humidity (0-100%).
    pub fn with_humidity(mut self, humidity: f32) -> Self {
        self.humidity_percent = humidity.clamp(0.0, 100.0);
        self
    }

    /// Set the propagation medium.
    pub fn with_medium(mut self, medium: Medium) -> Self {
        self.medium = medium;
        self
    }

    /// Get medium properties for current environment.
    pub fn medium_properties(&self) -> MediumProperties {
        match self.medium {
            Medium::Air => MediumProperties::air(self.temperature_c, self.humidity_percent),
            Medium::Water => MediumProperties::water(self.temperature_c),
            Medium::Steel => MediumProperties::steel(),
            Medium::Aluminum => MediumProperties::aluminum(),
            Medium::Custom => MediumProperties::custom(343.0, 1.2, 0.001),
        }
    }

    /// Get the speed of sound for current conditions.
    pub fn speed_of_sound(&self) -> f32 {
        self.medium_properties().speed_of_sound
    }
}

/// Calculate speed of sound in air as a function of temperature and humidity.
///
/// Uses the formula from Cramer (J. Acoust. Soc. Am., 1993):
/// c = 331.3 * sqrt(1 + T/273.15) * (1 + humidity_correction)
///
/// The humidity effect is small (~0.5% at 100% humidity) but measurable.
pub fn speed_of_sound_in_air(temperature_c: f32, humidity_percent: f32) -> f32 {
    let temp_k = temperature_c + 273.15;

    // Base speed from temperature (Laplace's formula)
    let base_speed = constants::SPEED_OF_SOUND_0C * (temp_k / 273.15).sqrt();

    // Humidity correction (simplified model)
    // Humid air is less dense than dry air, so sound travels faster
    // Effect is approximately 0.1-0.6 m/s per 10% humidity at typical temperatures
    let saturation_pressure = saturation_vapor_pressure(temperature_c);
    let vapor_pressure = (humidity_percent / 100.0) * saturation_pressure;
    let molar_fraction = vapor_pressure / constants::STANDARD_PRESSURE;

    // The correction factor accounts for the lower molecular weight of water vapor
    // Dry air: ~29 g/mol, Water vapor: ~18 g/mol
    let humidity_correction = 0.14 * molar_fraction;

    base_speed * (1.0 + humidity_correction)
}

/// Calculate saturation vapor pressure using the Buck equation.
///
/// Valid for -40°C to 50°C range.
pub fn saturation_vapor_pressure(temperature_c: f32) -> f32 {
    // Buck equation (enhanced accuracy)
    let t = temperature_c;
    611.21 * ((18.678 - t / 234.5) * (t / (257.14 + t))).exp()
}

/// Atmospheric absorption coefficient calculation based on ISO 9613-1.
///
/// Returns the absorption coefficient in dB/m for a given frequency.
/// This model accounts for:
/// - Molecular relaxation of oxygen and nitrogen
/// - Classical viscosity and thermal conductivity losses
/// - Humidity effects on relaxation frequencies
#[derive(Debug, Clone)]
pub struct AtmosphericAbsorption {
    /// Temperature in Celsius
    temperature_c: f32,
    /// Relative humidity (0-100%)
    humidity_percent: f32,
    /// Atmospheric pressure in Pascals
    pressure_pa: f32,
    /// Precomputed relaxation frequency for oxygen
    fr_o: f32,
    /// Precomputed relaxation frequency for nitrogen
    fr_n: f32,
}

impl AtmosphericAbsorption {
    /// Create a new absorption calculator for given conditions.
    pub fn new(temperature_c: f32, humidity_percent: f32, pressure_pa: f32) -> Self {
        let temp_k = temperature_c + 273.15;
        let temp_ratio = temp_k / 293.15; // Reference temp: 20°C

        // Calculate molar concentration of water vapor
        let psat = saturation_vapor_pressure(temperature_c);
        let h = humidity_percent / 100.0 * psat / pressure_pa;

        // Relaxation frequency for oxygen (ISO 9613-1, Eq. 3)
        let fr_o = (pressure_pa / constants::STANDARD_PRESSURE)
            * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));

        // Relaxation frequency for nitrogen (ISO 9613-1, Eq. 4)
        let fr_n = (pressure_pa / constants::STANDARD_PRESSURE)
            * temp_ratio.powf(-0.5)
            * (9.0 + 280.0 * h * (-4.17 * (temp_ratio.powf(-1.0 / 3.0) - 1.0)).exp());

        Self {
            temperature_c,
            humidity_percent,
            pressure_pa,
            fr_o,
            fr_n,
        }
    }

    /// Calculate absorption coefficient in dB/m for a given frequency.
    ///
    /// Based on ISO 9613-1:1993 standard.
    pub fn coefficient_db_per_meter(&self, frequency_hz: f32) -> f32 {
        let temp_k = self.temperature_c + 273.15;
        let temp_ratio = temp_k / 293.15;
        let pressure_ratio = self.pressure_pa / constants::STANDARD_PRESSURE;

        let f = frequency_hz;
        let f_sq = f * f;

        // Classical absorption (viscosity + thermal conductivity)
        let alpha_classic = 1.84e-11 * pressure_ratio.recip() * temp_ratio.sqrt() * f_sq;

        // Vibrational relaxation absorption for oxygen
        let alpha_o = 0.01275
            * (-2239.1 / temp_k).exp()
            * (self.fr_o + f_sq / self.fr_o).recip()
            * f_sq;

        // Vibrational relaxation absorption for nitrogen
        let alpha_n = 0.1068
            * (-3352.0 / temp_k).exp()
            * (self.fr_n + f_sq / self.fr_n).recip()
            * f_sq;

        // Total absorption in dB/m (convert from Np/m to dB/m: multiply by 8.686)
        8.686 * (alpha_classic + temp_ratio.powf(-2.5) * (alpha_o + alpha_n))
    }

    /// Calculate absorption coefficient as a linear factor per meter.
    ///
    /// Returns a multiplier (0-1) representing energy remaining after 1 meter.
    pub fn coefficient_linear(&self, frequency_hz: f32) -> f32 {
        let db_per_m = self.coefficient_db_per_meter(frequency_hz);
        10.0_f32.powf(-db_per_m / 20.0)
    }

    /// Calculate absorption for a specific distance.
    ///
    /// Returns the amplitude multiplier after traveling the given distance.
    pub fn absorption_at_distance(&self, frequency_hz: f32, distance_m: f32) -> f32 {
        let db_per_m = self.coefficient_db_per_meter(frequency_hz);
        10.0_f32.powf(-db_per_m * distance_m / 20.0)
    }
}

/// Multi-band damping coefficients for efficient GPU computation.
///
/// Pre-computes absorption for multiple frequency bands to avoid
/// per-sample FFT computations in the main FDTD loop.
#[derive(Debug, Clone)]
pub struct MultiBandDamping {
    /// Center frequencies for each band (Hz)
    pub center_frequencies: Vec<f32>,
    /// Damping coefficient per band (linear multiplier per time step)
    pub damping_coefficients: Vec<f32>,
    /// Number of frequency bands
    pub num_bands: usize,
}

impl MultiBandDamping {
    /// Create multi-band damping from environment conditions.
    ///
    /// Uses octave bands from 31.5 Hz to 16 kHz (standard ISO bands).
    pub fn from_environment(env: &Environment, time_step: f32, _cell_size: f32) -> Self {
        let absorption = AtmosphericAbsorption::new(
            env.temperature_c,
            env.humidity_percent,
            env.pressure_pa,
        );

        // ISO octave band center frequencies
        let center_frequencies: Vec<f32> = vec![
            31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
        ];

        let props = env.medium_properties();

        // Convert dB/m absorption to per-step damping
        // Distance traveled per step = speed * dt
        let distance_per_step = props.speed_of_sound * time_step;

        let damping_coefficients: Vec<f32> = center_frequencies
            .iter()
            .map(|&freq| {
                if props.frequency_dependent {
                    // Frequency-dependent absorption
                    let db_per_m = absorption.coefficient_db_per_meter(freq);
                    let db_per_step = db_per_m * distance_per_step;
                    // Convert dB to linear amplitude factor
                    10.0_f32.powf(-db_per_step / 20.0)
                } else {
                    // Constant damping for metals etc.
                    1.0 - props.base_damping * distance_per_step
                }
            })
            .collect();

        Self {
            num_bands: center_frequencies.len(),
            center_frequencies,
            damping_coefficients,
        }
    }

    /// Create simplified single-band damping.
    pub fn single_band(damping_factor: f32) -> Self {
        Self {
            center_frequencies: vec![1000.0],
            damping_coefficients: vec![1.0 - damping_factor],
            num_bands: 1,
        }
    }

    /// Get the effective damping for a dominant frequency.
    pub fn damping_for_frequency(&self, frequency: f32) -> f32 {
        // Find nearest band
        let mut min_diff = f32::MAX;
        let mut best_idx = 0;
        for (i, &center) in self.center_frequencies.iter().enumerate() {
            let diff = (frequency - center).abs();
            if diff < min_diff {
                min_diff = diff;
                best_idx = i;
            }
        }
        self.damping_coefficients[best_idx]
    }

    /// Get average damping across all bands (for simple simulations).
    pub fn average_damping(&self) -> f32 {
        self.damping_coefficients.iter().sum::<f32>() / self.num_bands as f32
    }
}

/// Parameters for the 3D acoustic wave simulation.
///
/// Uses the FDTD (Finite-Difference Time-Domain) method for solving
/// the 3D wave equation with realistic physics.
#[derive(Debug, Clone)]
pub struct AcousticParams3D {
    /// Environment (temperature, humidity, medium)
    pub environment: Environment,
    /// Medium properties derived from environment
    pub medium: MediumProperties,
    /// Cell size (spatial step) in meters (uniform in all dimensions)
    pub cell_size: f32,
    /// Time step in seconds (computed for numerical stability)
    pub time_step: f32,
    /// Courant number squared (c² = (speed * dt / dx)²)
    pub c_squared: f32,
    /// Multi-band damping coefficients
    pub damping: MultiBandDamping,
    /// Simple average damping factor for basic FDTD (0-1)
    pub simple_damping: f32,
}

impl Default for AcousticParams3D {
    fn default() -> Self {
        Self::new(Environment::default(), 0.1) // 10cm cell size
    }
}

impl AcousticParams3D {
    /// Create new 3D acoustic parameters from environment settings.
    ///
    /// The time step is automatically computed to satisfy the CFL condition
    /// for numerical stability in 3D: dt <= dx / (c * sqrt(3))
    ///
    /// # Arguments
    /// * `environment` - Environmental conditions (temperature, humidity, medium)
    /// * `cell_size` - Spatial step size in meters (same for x, y, z)
    pub fn new(environment: Environment, cell_size: f32) -> Self {
        let medium = environment.medium_properties();
        let speed = medium.speed_of_sound;

        // CFL condition for 3D wave equation: c * dt / dx <= 1/sqrt(3)
        // We use a safety factor of 2.0 instead of sqrt(3) ~= 1.732
        let time_step = cell_size / (speed * 2.0);

        // Courant number squared for FDTD update
        let courant = speed * time_step / cell_size;
        let c_squared = courant * courant;

        // Compute frequency-dependent damping
        let damping = MultiBandDamping::from_environment(&environment, time_step, cell_size);

        // Simple damping for basic FDTD
        let simple_damping = 1.0 - damping.average_damping().max(medium.base_damping);

        Self {
            environment,
            medium,
            cell_size,
            time_step,
            c_squared,
            damping,
            simple_damping: simple_damping.clamp(0.99, 0.9999),
        }
    }

    /// Create parameters for a specific medium at standard conditions.
    pub fn for_medium(medium: Medium, cell_size: f32) -> Self {
        Self::new(Environment::default().with_medium(medium), cell_size)
    }

    /// Update environment and recompute derived parameters.
    pub fn set_environment(&mut self, env: Environment) {
        *self = Self::new(env, self.cell_size);
    }

    /// Update cell size and recompute parameters.
    pub fn set_cell_size(&mut self, size: f32) {
        *self = Self::new(self.environment.clone(), size);
    }

    /// Compute the Courant number (c * dt / dx).
    ///
    /// For numerical stability in 3D, this should be <= 1/sqrt(3) ~= 0.577
    pub fn courant_number(&self) -> f32 {
        self.c_squared.sqrt()
    }

    /// Check if the parameters satisfy the CFL stability condition.
    pub fn is_stable(&self) -> bool {
        self.courant_number() <= 1.0 / 3.0_f32.sqrt()
    }

    /// Get the wavelength for a given frequency.
    pub fn wavelength(&self, frequency: f32) -> f32 {
        self.medium.speed_of_sound / frequency
    }

    /// Get the number of cells per wavelength for a given frequency.
    ///
    /// For accurate simulation, this should be >= 10 (ideally 20+).
    pub fn cells_per_wavelength(&self, frequency: f32) -> f32 {
        self.wavelength(frequency) / self.cell_size
    }

    /// Get recommended maximum frequency for accurate simulation.
    ///
    /// Based on the Nyquist-like criterion of ~10 cells per wavelength.
    pub fn max_accurate_frequency(&self) -> f32 {
        self.medium.speed_of_sound / (10.0 * self.cell_size)
    }
}

/// 3D position in the simulation space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Position3D {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Distance to another position.
    pub fn distance_to(&self, other: &Position3D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Convert to grid indices.
    pub fn to_grid_indices(&self, cell_size: f32) -> (usize, usize, usize) {
        (
            (self.x / cell_size).round() as usize,
            (self.y / cell_size).round() as usize,
            (self.z / cell_size).round() as usize,
        )
    }
}

/// Orientation in 3D space (for the virtual head).
#[derive(Debug, Clone, Copy)]
pub struct Orientation3D {
    /// Yaw angle in radians (rotation around Y axis)
    pub yaw: f32,
    /// Pitch angle in radians (rotation around X axis)
    pub pitch: f32,
    /// Roll angle in radians (rotation around Z axis)
    pub roll: f32,
}

impl Default for Orientation3D {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
        }
    }
}

impl Orientation3D {
    pub fn new(yaw: f32, pitch: f32, roll: f32) -> Self {
        Self { yaw, pitch, roll }
    }

    /// Get the forward direction vector.
    pub fn forward(&self) -> (f32, f32, f32) {
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();

        (sy * cp, -sp, cy * cp)
    }

    /// Get the right direction vector (for ear positions).
    pub fn right(&self) -> (f32, f32, f32) {
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        let cr = self.roll.cos();
        let sr = self.roll.sin();

        (cy * cr, sr, -sy * cr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speed_of_sound_temperature() {
        // At 0°C, speed should be ~331.3 m/s
        let speed_0c = speed_of_sound_in_air(0.0, 0.0);
        assert!((speed_0c - 331.3).abs() < 1.0);

        // At 20°C, speed should be ~343 m/s
        let speed_20c = speed_of_sound_in_air(20.0, 0.0);
        assert!((speed_20c - 343.0).abs() < 2.0);

        // Higher temperature = faster sound
        assert!(speed_20c > speed_0c);
    }

    #[test]
    fn test_speed_humidity_effect() {
        // Humidity should increase speed slightly
        let speed_dry = speed_of_sound_in_air(20.0, 0.0);
        let speed_humid = speed_of_sound_in_air(20.0, 100.0);

        // Effect should be small but positive
        assert!(speed_humid > speed_dry);
        assert!(speed_humid - speed_dry < 2.0); // Less than 2 m/s difference
    }

    #[test]
    fn test_atmospheric_absorption() {
        let absorption = AtmosphericAbsorption::new(20.0, 50.0, constants::STANDARD_PRESSURE);

        // Low frequencies should have minimal absorption
        let low_freq = absorption.coefficient_db_per_meter(100.0);
        assert!(low_freq < 0.01);

        // High frequencies should have more absorption
        let high_freq = absorption.coefficient_db_per_meter(8000.0);
        assert!(high_freq > low_freq);

        // Very high frequencies have significant absorption
        let very_high = absorption.coefficient_db_per_meter(16000.0);
        assert!(very_high > high_freq);
    }

    #[test]
    fn test_medium_properties() {
        let air = MediumProperties::air(20.0, 50.0);
        assert!((air.speed_of_sound - 343.0).abs() < 5.0);

        let water = MediumProperties::water(20.0);
        assert!((water.speed_of_sound - 1481.0).abs() < 10.0);

        let steel = MediumProperties::steel();
        assert_eq!(steel.speed_of_sound, constants::SPEED_IN_STEEL);
    }

    #[test]
    fn test_cfl_stability() {
        let params = AcousticParams3D::default();
        assert!(params.is_stable());
        assert!(params.courant_number() <= 1.0 / 3.0_f32.sqrt());
    }

    #[test]
    fn test_multiband_damping() {
        let env = Environment::default();
        let damping = MultiBandDamping::from_environment(&env, 0.0001, 0.01);

        // Should have multiple bands
        assert_eq!(damping.num_bands, 10);

        // Higher frequencies should have more damping (lower coefficient)
        let low_damping = damping.damping_for_frequency(100.0);
        let high_damping = damping.damping_for_frequency(8000.0);
        assert!(low_damping > high_damping);
    }

    #[test]
    fn test_position_distance() {
        let p1 = Position3D::new(0.0, 0.0, 0.0);
        let p2 = Position3D::new(3.0, 4.0, 0.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 0.001);
    }
}
