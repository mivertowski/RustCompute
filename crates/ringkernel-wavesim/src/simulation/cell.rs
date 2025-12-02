//! Cell state for a single grid element in the acoustic simulation.

use super::AcousticParams;

/// Direction to a neighboring cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    North,
    South,
    East,
    West,
}

impl Direction {
    /// All four cardinal directions.
    pub const ALL: [Direction; 4] = [
        Direction::North,
        Direction::South,
        Direction::East,
        Direction::West,
    ];

    /// Get the opposite direction.
    pub fn opposite(self) -> Direction {
        match self {
            Direction::North => Direction::South,
            Direction::South => Direction::North,
            Direction::East => Direction::West,
            Direction::West => Direction::East,
        }
    }
}

/// State for a single cell in the acoustic grid.
///
/// Each cell maintains pressure values and communicates with 4 neighbors
/// (North, South, East, West) to propagate acoustic waves.
#[derive(Debug, Clone)]
pub struct CellState {
    /// X coordinate in the grid.
    pub x: u32,
    /// Y coordinate in the grid.
    pub y: u32,

    /// Current pressure value.
    pub pressure: f32,
    /// Previous pressure value (for time-stepping).
    pub pressure_prev: f32,

    /// Neighbor pressures (received via neighbor exchange).
    pub p_north: f32,
    pub p_south: f32,
    pub p_east: f32,
    pub p_west: f32,

    /// True if this cell is on the grid boundary.
    pub is_boundary: bool,
    /// Reflection coefficient (0 = full absorb, 1 = full reflect).
    pub reflection_coeff: f32,
}

impl CellState {
    /// Create a new cell at the given grid position.
    pub fn new(x: u32, y: u32) -> Self {
        Self {
            x,
            y,
            pressure: 0.0,
            pressure_prev: 0.0,
            p_north: 0.0,
            p_south: 0.0,
            p_east: 0.0,
            p_west: 0.0,
            is_boundary: false,
            reflection_coeff: 1.0,
        }
    }

    /// Get the kernel ID for this cell.
    pub fn kernel_id(&self) -> String {
        format!("cell_{}_{}", self.x, self.y)
    }

    /// Get the kernel ID of a neighbor in the given direction.
    ///
    /// Returns None if the neighbor would be outside the grid bounds.
    pub fn neighbor_id(&self, dir: Direction, width: u32, height: u32) -> Option<String> {
        let (nx, ny) = match dir {
            Direction::North if self.y > 0 => (self.x, self.y - 1),
            Direction::South if self.y < height - 1 => (self.x, self.y + 1),
            Direction::West if self.x > 0 => (self.x - 1, self.y),
            Direction::East if self.x < width - 1 => (self.x + 1, self.y),
            _ => return None,
        };
        Some(format!("cell_{}_{}", nx, ny))
    }

    /// Compute the next pressure value using FDTD scheme.
    ///
    /// Uses the 2D wave equation: ∂²p/∂t² = c²∇²p
    ///
    /// Discretized as:
    /// p[n+1] = 2*p[n] - p[n-1] + c²*(p_N + p_S + p_E + p_W - 4*p)
    pub fn step(&mut self, params: &AcousticParams) {
        let c = params.courant_number();
        let c2 = c * c;

        // 2D Laplacian approximation
        let laplacian =
            self.p_north + self.p_south + self.p_east + self.p_west - 4.0 * self.pressure;

        // FDTD time-stepping
        let p_new = 2.0 * self.pressure - self.pressure_prev + c2 * laplacian;

        // Apply damping to simulate energy loss
        let p_damped = p_new * (1.0 - params.damping);

        // Apply boundary conditions
        let p_final = if self.is_boundary {
            p_damped * self.reflection_coeff
        } else {
            p_damped
        };

        // Update state
        self.pressure_prev = self.pressure;
        self.pressure = p_final;
    }

    /// Inject an impulse (Dirac delta) at this cell.
    pub fn inject_impulse(&mut self, amplitude: f32) {
        self.pressure += amplitude;
    }

    /// Reset the cell to initial state.
    pub fn reset(&mut self) {
        self.pressure = 0.0;
        self.pressure_prev = 0.0;
        self.p_north = 0.0;
        self.p_south = 0.0;
        self.p_east = 0.0;
        self.p_west = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_state_new() {
        let cell = CellState::new(5, 3);
        assert_eq!(cell.x, 5);
        assert_eq!(cell.y, 3);
        assert_eq!(cell.pressure, 0.0);
        assert!(!cell.is_boundary);
    }

    #[test]
    fn test_kernel_id() {
        let cell = CellState::new(5, 3);
        assert_eq!(cell.kernel_id(), "cell_5_3");
    }

    #[test]
    fn test_neighbor_id() {
        let cell = CellState::new(5, 5);
        assert_eq!(
            cell.neighbor_id(Direction::North, 10, 10),
            Some("cell_5_4".to_string())
        );
        assert_eq!(
            cell.neighbor_id(Direction::South, 10, 10),
            Some("cell_5_6".to_string())
        );
        assert_eq!(
            cell.neighbor_id(Direction::East, 10, 10),
            Some("cell_6_5".to_string())
        );
        assert_eq!(
            cell.neighbor_id(Direction::West, 10, 10),
            Some("cell_4_5".to_string())
        );
    }

    #[test]
    fn test_neighbor_id_at_boundary() {
        let cell = CellState::new(0, 0);
        assert_eq!(cell.neighbor_id(Direction::North, 10, 10), None);
        assert_eq!(cell.neighbor_id(Direction::West, 10, 10), None);
        assert!(cell.neighbor_id(Direction::South, 10, 10).is_some());
        assert!(cell.neighbor_id(Direction::East, 10, 10).is_some());
    }

    #[test]
    fn test_impulse_injection() {
        let mut cell = CellState::new(0, 0);
        cell.inject_impulse(0.5);
        assert_eq!(cell.pressure, 0.5);
        cell.inject_impulse(0.3);
        assert_eq!(cell.pressure, 0.8);
    }

    #[test]
    fn test_reset() {
        let mut cell = CellState::new(0, 0);
        cell.pressure = 1.0;
        cell.pressure_prev = 0.5;
        cell.p_north = 0.2;
        cell.reset();
        assert_eq!(cell.pressure, 0.0);
        assert_eq!(cell.pressure_prev, 0.0);
        assert_eq!(cell.p_north, 0.0);
    }
}
