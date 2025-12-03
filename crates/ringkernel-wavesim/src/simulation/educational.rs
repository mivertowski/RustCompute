//! Educational simulation modes demonstrating the evolution of parallel computing.
//!
//! These modes provide visual demonstrations of how computing paradigms have evolved:
//!
//! 1. **CellByCell (1950s Sequential)**: Processes one cell at a time, showing how early
//!    computers executed instructions sequentially.
//!
//! 2. **RowByRow (1970s Vectorization)**: Processes entire rows at once, demonstrating
//!    the vectorization paradigm introduced by machines like the Cray-1.
//!
//! 3. **ChaoticParallel (1990s Naive Parallelism)**: Multiple cells processed without
//!    coordination, showing race conditions and data inconsistency issues.
//!
//! 4. **SynchronizedParallel (2000s Barrier Sync)**: Parallel execution with barriers,
//!    demonstrating the dominant parallel computing model of the 2000s.
//!
//! 5. **ActorBased (Modern)**: Tile-based actors with Hybrid Logical Clocks for
//!    causal consistency - the RingKernel approach.

use rand::prelude::*;
use rand::rngs::ThreadRng;

/// Educational simulation modes demonstrating computing evolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimulationMode {
    /// Standard full-speed mode (no visualization of processing).
    #[default]
    Standard,

    /// Sequential cell-by-cell processing (1950s era).
    /// Processes one cell per frame with visual indicator.
    CellByCell,

    /// Row-by-row vectorized processing (1970s era).
    /// Processes one row per frame.
    RowByRow,

    /// Chaotic parallel without synchronization (1990s era).
    /// Shows data races and inconsistent results.
    ChaoticParallel,

    /// Synchronized parallel with barriers (2000s era).
    /// Shows proper synchronization but with overhead.
    SynchronizedParallel,

    /// Actor/tile-based with HLC (Modern approach).
    /// Demonstrates causal consistency with message passing.
    ActorBased,
}

impl std::fmt::Display for SimulationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimulationMode::Standard => write!(f, "Standard"),
            SimulationMode::CellByCell => write!(f, "Cell-by-Cell (1950s)"),
            SimulationMode::RowByRow => write!(f, "Row-by-Row (1970s)"),
            SimulationMode::ChaoticParallel => write!(f, "Chaotic (1990s)"),
            SimulationMode::SynchronizedParallel => write!(f, "Barrier Sync (2000s)"),
            SimulationMode::ActorBased => write!(f, "Actor/HLC (Modern)"),
        }
    }
}

impl SimulationMode {
    /// Get all available modes for the UI.
    pub fn all() -> &'static [SimulationMode] {
        &[
            SimulationMode::Standard,
            SimulationMode::CellByCell,
            SimulationMode::RowByRow,
            SimulationMode::ChaoticParallel,
            SimulationMode::SynchronizedParallel,
            SimulationMode::ActorBased,
        ]
    }

    /// Get a description of the mode for educational purposes.
    pub fn description(&self) -> &'static str {
        match self {
            SimulationMode::Standard =>
                "Full-speed parallel simulation",
            SimulationMode::CellByCell =>
                "1950s: Sequential processing, one cell at a time",
            SimulationMode::RowByRow =>
                "1970s: Vector processing, entire rows at once (Cray-style)",
            SimulationMode::ChaoticParallel =>
                "1990s: Parallel without sync - watch for glitches!",
            SimulationMode::SynchronizedParallel =>
                "2000s: Parallel with barriers - safe but slow",
            SimulationMode::ActorBased =>
                "Modern: Actor model with HLC for causal ordering",
        }
    }

    /// Whether this mode shows processing indicators.
    pub fn shows_processing(&self) -> bool {
        matches!(
            self,
            SimulationMode::CellByCell
                | SimulationMode::RowByRow
                | SimulationMode::ChaoticParallel
                | SimulationMode::SynchronizedParallel
                | SimulationMode::ActorBased
        )
    }
}

/// Result of a single frame of educational processing.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Whether the full simulation step is complete (buffers should be swapped).
    pub step_complete: bool,
    /// Whether buffers should be swapped after this frame.
    pub should_swap: bool,
}

impl StepResult {
    fn in_progress() -> Self {
        Self {
            step_complete: false,
            should_swap: false,
        }
    }

    fn complete() -> Self {
        Self {
            step_complete: true,
            should_swap: true,
        }
    }
}

/// State for tracking incremental processing in educational modes.
#[derive(Debug, Clone)]
pub struct ProcessingState {
    /// Current cell being processed (for CellByCell mode).
    pub current_cell: Option<(u32, u32)>,

    /// Current row being processed (for RowByRow mode).
    pub current_row: Option<u32>,

    /// Set of cells currently being processed (for parallel modes).
    pub active_cells: Vec<(u32, u32)>,

    /// Cells processed in the current step (for visualization).
    pub just_processed: Vec<(u32, u32)>,

    /// Current tile being processed (for ActorBased mode).
    pub active_tiles: Vec<(u32, u32)>,

    /// Whether processing is complete for the current simulation step.
    pub step_complete: bool,

    /// Logical clock value (for ActorBased mode).
    pub hlc_tick: u64,

    /// Messages in flight (for ActorBased visualization).
    pub messages_in_flight: Vec<TileMessage>,
}

/// A message between tiles (for ActorBased visualization).
#[derive(Debug, Clone)]
pub struct TileMessage {
    /// Source tile.
    pub from: (u32, u32),
    /// Destination tile.
    pub to: (u32, u32),
    /// HLC timestamp.
    pub timestamp: u64,
}

impl Default for ProcessingState {
    fn default() -> Self {
        Self {
            current_cell: None,
            current_row: None,
            active_cells: Vec::new(),
            just_processed: Vec::new(),
            active_tiles: Vec::new(),
            step_complete: true,
            hlc_tick: 0,
            messages_in_flight: Vec::new(),
        }
    }
}

impl ProcessingState {
    /// Reset state for a new step.
    pub fn begin_step(&mut self) {
        self.current_cell = None;
        self.current_row = None;
        self.active_cells.clear();
        self.just_processed.clear();
        self.active_tiles.clear();
        self.step_complete = false;
    }

    /// Mark step as complete.
    pub fn complete_step(&mut self) {
        self.step_complete = true;
        self.hlc_tick += 1;
    }
}

/// Educational step processor for SimulationGrid.
pub struct EducationalProcessor {
    /// Processing state for visualization.
    pub state: ProcessingState,

    /// Simulation mode.
    pub mode: SimulationMode,

    /// Cells per frame for parallel modes.
    pub cells_per_frame: usize,

    /// Random number generator for chaotic mode.
    rng: ThreadRng,

    /// Internal position tracking for incremental processing.
    cell_x: u32,
    cell_y: u32,

    /// Tile size for actor-based mode.
    tile_size: u32,
}

impl Default for EducationalProcessor {
    fn default() -> Self {
        Self {
            state: ProcessingState::default(),
            mode: SimulationMode::Standard,
            cells_per_frame: 16,
            rng: thread_rng(),
            cell_x: 1,
            cell_y: 1,
            tile_size: 16,
        }
    }
}

impl EducationalProcessor {
    /// Create a new processor with the given mode.
    pub fn new(mode: SimulationMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    /// Set the simulation mode.
    pub fn set_mode(&mut self, mode: SimulationMode) {
        self.mode = mode;
        self.reset();
    }

    /// Reset processing state.
    pub fn reset(&mut self) {
        self.state = ProcessingState::default();
        self.cell_x = 1;
        self.cell_y = 1;
    }

    /// Perform one frame of educational processing on the grid.
    /// Returns StepResult indicating whether step is complete and if buffers should be swapped.
    pub fn step_frame(
        &mut self,
        pressure: &mut [f32],
        pressure_prev: &mut [f32],
        width: usize,
        height: usize,
        c2: f32,
        damping: f32,
    ) -> StepResult {
        match self.mode {
            SimulationMode::Standard => {
                self.step_standard(pressure, pressure_prev, width, height, c2, damping)
            }
            SimulationMode::CellByCell => {
                self.step_cell_by_cell(pressure, pressure_prev, width, height, c2, damping)
            }
            SimulationMode::RowByRow => {
                self.step_row_by_row(pressure, pressure_prev, width, height, c2, damping)
            }
            SimulationMode::ChaoticParallel => {
                self.step_chaotic_parallel(pressure, pressure_prev, width, height, c2, damping)
            }
            SimulationMode::SynchronizedParallel => {
                self.step_synchronized_parallel(pressure, pressure_prev, width, height, c2, damping)
            }
            SimulationMode::ActorBased => {
                self.step_actor_based(pressure, pressure_prev, width, height, c2, damping)
            }
        }
    }

    /// Standard full-speed processing.
    fn step_standard(
        &mut self,
        pressure: &[f32],
        pressure_prev: &mut [f32],
        width: usize,
        height: usize,
        c2: f32,
        damping: f32,
    ) -> StepResult {
        // Process all interior cells
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = y * width + x;
                let p_curr = pressure[idx];
                let p_prev = pressure_prev[idx];
                let p_north = pressure[idx - width];
                let p_south = pressure[idx + width];
                let p_west = pressure[idx - 1];
                let p_east = pressure[idx + 1];

                let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
                let p_new = 2.0 * p_curr - p_prev + c2 * laplacian;
                pressure_prev[idx] = p_new * damping;
            }
        }

        self.state.complete_step();
        StepResult::complete()
    }

    /// Cell-by-cell sequential processing (1950s).
    /// Processes a small number of cells per frame for visualization.
    fn step_cell_by_cell(
        &mut self,
        pressure: &[f32],
        pressure_prev: &mut [f32],
        width: usize,
        height: usize,
        c2: f32,
        damping: f32,
    ) -> StepResult {
        self.state.just_processed.clear();

        let cells_this_frame = self.cells_per_frame.min(32);

        for _ in 0..cells_this_frame {
            if self.cell_y >= height as u32 - 1 {
                // Done with this step
                self.cell_x = 1;
                self.cell_y = 1;
                self.state.complete_step();
                return StepResult::complete();
            }

            let x = self.cell_x as usize;
            let y = self.cell_y as usize;

            // Process this cell
            let idx = y * width + x;
            let p_curr = pressure[idx];
            let p_prev = pressure_prev[idx];
            let p_north = pressure[idx - width];
            let p_south = pressure[idx + width];
            let p_west = pressure[idx - 1];
            let p_east = pressure[idx + 1];

            let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
            let p_new = 2.0 * p_curr - p_prev + c2 * laplacian;
            pressure_prev[idx] = p_new * damping;

            self.state.just_processed.push((self.cell_x, self.cell_y));
            self.state.current_cell = Some((self.cell_x, self.cell_y));

            // Advance to next cell
            self.cell_x += 1;
            if self.cell_x >= width as u32 - 1 {
                self.cell_x = 1;
                self.cell_y += 1;
            }
        }

        StepResult::in_progress()
    }

    /// Row-by-row vectorized processing (1970s).
    fn step_row_by_row(
        &mut self,
        pressure: &[f32],
        pressure_prev: &mut [f32],
        width: usize,
        height: usize,
        c2: f32,
        damping: f32,
    ) -> StepResult {
        self.state.just_processed.clear();

        if self.cell_y >= height as u32 - 1 {
            // Done with this step
            self.cell_y = 1;
            self.state.current_row = None;
            self.state.complete_step();
            return StepResult::complete();
        }

        let y = self.cell_y as usize;
        self.state.current_row = Some(self.cell_y);

        // Process entire row at once (vectorized style)
        for x in 1..width - 1 {
            let idx = y * width + x;
            let p_curr = pressure[idx];
            let p_prev = pressure_prev[idx];
            let p_north = pressure[idx - width];
            let p_south = pressure[idx + width];
            let p_west = pressure[idx - 1];
            let p_east = pressure[idx + 1];

            let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
            let p_new = 2.0 * p_curr - p_prev + c2 * laplacian;
            pressure_prev[idx] = p_new * damping;

            self.state.just_processed.push((x as u32, self.cell_y));
        }

        self.cell_y += 1;
        StepResult::in_progress()
    }

    /// Chaotic parallel processing without synchronization (1990s).
    /// Deliberately shows data races and inconsistent results.
    fn step_chaotic_parallel(
        &mut self,
        pressure: &mut [f32],
        pressure_prev: &mut [f32],
        width: usize,
        height: usize,
        c2: f32,
        damping: f32,
    ) -> StepResult {
        self.state.just_processed.clear();
        self.state.active_cells.clear();

        // Pick random cells to process (simulating unsynchronized parallel access)
        let num_cells = (width * height / 16).min(64);

        for _ in 0..num_cells {
            let x = self.rng.gen_range(1..width - 1);
            let y = self.rng.gen_range(1..height - 1);

            self.state.active_cells.push((x as u32, y as u32));

            // Process this cell - note: reading from already-modified neighbors causes chaos
            let idx = y * width + x;
            let p_curr = pressure[idx];
            let p_prev = pressure_prev[idx];

            // Sometimes read from pressure_prev (old data), sometimes from pressure (in-progress)
            // This simulates race conditions
            let (p_north, p_south, p_east, p_west) = if self.rng.gen_bool(0.5) {
                // Read from "current" buffer (race condition!)
                (
                    pressure[idx - width],
                    pressure[idx + width],
                    pressure[idx + 1],
                    pressure[idx - 1],
                )
            } else {
                // Read from "previous" buffer (also partially updated - chaos!)
                (
                    pressure_prev[idx - width],
                    pressure_prev[idx + width],
                    pressure_prev[idx + 1],
                    pressure_prev[idx - 1],
                )
            };

            let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
            let p_new = 2.0 * p_curr - p_prev + c2 * laplacian;

            // Sometimes write to wrong buffer too!
            if self.rng.gen_bool(0.3) {
                pressure[idx] = p_new * damping;  // Wrong buffer!
            } else {
                pressure_prev[idx] = p_new * damping;
            }
        }

        self.state.just_processed = self.state.active_cells.clone();

        // After enough frames, complete the step (but with inconsistent data!)
        self.cell_y += 1;
        if self.cell_y >= 8 {
            self.cell_y = 0;
            self.state.complete_step();
            return StepResult::complete();
        }

        StepResult::in_progress()
    }

    /// Synchronized parallel with barriers (2000s).
    fn step_synchronized_parallel(
        &mut self,
        pressure: &[f32],
        pressure_prev: &mut [f32],
        width: usize,
        height: usize,
        c2: f32,
        damping: f32,
    ) -> StepResult {
        self.state.just_processed.clear();
        self.state.active_cells.clear();

        // Process in wavefronts for correct dependencies
        // A wavefront is all cells where x + y = constant
        let max_wavefront = (width + height - 4) as u32;

        if self.cell_y > max_wavefront {
            self.cell_y = 0;
            self.state.complete_step();
            return StepResult::complete();
        }

        let wavefront = self.cell_y;

        // Process all cells in this wavefront (they're independent!)
        for x in 1..width - 1 {
            let y_calc = wavefront as i32 - x as i32 + 1;
            if y_calc >= 1 && y_calc < (height - 1) as i32 {
                let y = y_calc as usize;
                let idx = y * width + x;
                let p_curr = pressure[idx];
                let p_prev = pressure_prev[idx];
                let p_north = pressure[idx - width];
                let p_south = pressure[idx + width];
                let p_west = pressure[idx - 1];
                let p_east = pressure[idx + 1];

                let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
                let p_new = 2.0 * p_curr - p_prev + c2 * laplacian;
                pressure_prev[idx] = p_new * damping;

                self.state.active_cells.push((x as u32, y as u32));
            }
        }

        self.state.just_processed = self.state.active_cells.clone();
        self.cell_y += 1;  // Move to next wavefront
        StepResult::in_progress()
    }

    /// Actor/tile-based processing with HLC (Modern).
    fn step_actor_based(
        &mut self,
        pressure: &[f32],
        pressure_prev: &mut [f32],
        width: usize,
        height: usize,
        c2: f32,
        damping: f32,
    ) -> StepResult {
        self.state.just_processed.clear();
        self.state.active_tiles.clear();
        self.state.messages_in_flight.clear();

        let tile_size = self.tile_size as usize;
        // Interior region is [1, width-2] x [1, height-2]
        let interior_width = width - 2;
        let interior_height = height - 2;
        // Use ceiling division to ensure all cells are covered
        let tiles_x = (interior_width + tile_size - 1) / tile_size;
        let tiles_y = (interior_height + tile_size - 1) / tile_size;

        let current_tile = self.cell_y as usize;
        let total_tiles = tiles_x * tiles_y;

        if current_tile >= total_tiles {
            self.cell_y = 0;
            self.state.hlc_tick += 1;
            self.state.complete_step();
            return StepResult::complete();
        }

        // Process one tile at a time with HLC-ordered messages
        let tile_x = current_tile % tiles_x;
        let tile_y = current_tile / tiles_x;

        self.state.active_tiles.push((tile_x as u32, tile_y as u32));

        // Create visual messages to/from neighbors
        if tile_x > 0 {
            self.state.messages_in_flight.push(TileMessage {
                from: ((tile_x - 1) as u32, tile_y as u32),
                to: (tile_x as u32, tile_y as u32),
                timestamp: self.state.hlc_tick,
            });
        }
        if tile_y > 0 {
            self.state.messages_in_flight.push(TileMessage {
                from: (tile_x as u32, (tile_y - 1) as u32),
                to: (tile_x as u32, tile_y as u32),
                timestamp: self.state.hlc_tick,
            });
        }

        // Process all cells in this tile
        // Tile starts at offset 1 (skip boundary) + tile_x * tile_size
        let start_x = 1 + tile_x * tile_size;
        let start_y = 1 + tile_y * tile_size;
        // Clamp to interior region (exclude boundary cells)
        let end_x = (start_x + tile_size).min(width - 1);
        let end_y = (start_y + tile_size).min(height - 1);

        for y in start_y..end_y {
            for x in start_x..end_x {
                let idx = y * width + x;
                let p_curr = pressure[idx];
                let p_prev = pressure_prev[idx];
                let p_north = pressure[idx - width];
                let p_south = pressure[idx + width];
                let p_west = pressure[idx - 1];
                let p_east = pressure[idx + 1];

                let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
                let p_new = 2.0 * p_curr - p_prev + c2 * laplacian;
                pressure_prev[idx] = p_new * damping;

                self.state.just_processed.push((x as u32, y as u32));
            }
        }

        self.cell_y += 1;
        StepResult::in_progress()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_display() {
        assert_eq!(format!("{}", SimulationMode::CellByCell), "Cell-by-Cell (1950s)");
        assert_eq!(format!("{}", SimulationMode::RowByRow), "Row-by-Row (1970s)");
        assert_eq!(format!("{}", SimulationMode::ChaoticParallel), "Chaotic (1990s)");
    }

    #[test]
    fn test_mode_all() {
        let modes = SimulationMode::all();
        assert_eq!(modes.len(), 6);
    }

    #[test]
    fn test_processing_state_reset() {
        let mut state = ProcessingState::default();
        state.current_cell = Some((5, 5));
        state.step_complete = false;
        state.begin_step();
        assert!(state.current_cell.is_none());
        assert!(!state.step_complete);
    }

    #[test]
    fn test_cell_by_cell_processing() {
        let mut processor = EducationalProcessor::new(SimulationMode::CellByCell);
        let width = 8;
        let height = 8;
        let mut pressure = vec![0.0; width * height];
        let mut pressure_prev = vec![0.0; width * height];

        // Set initial impulse
        pressure[3 * width + 3] = 1.0;

        let c2 = 0.25;
        let damping = 0.99;

        // Run until step completes
        let mut iterations = 0;
        loop {
            let result = processor.step_frame(&mut pressure, &mut pressure_prev, width, height, c2, damping);
            if result.step_complete {
                if result.should_swap {
                    std::mem::swap(&mut pressure, &mut pressure_prev);
                }
                break;
            }
            iterations += 1;
            assert!(iterations < 100, "Should complete within reasonable iterations");
        }

        // Should have processed all interior cells
        assert!(processor.state.step_complete);
    }

    #[test]
    fn test_row_by_row_processing() {
        let mut processor = EducationalProcessor::new(SimulationMode::RowByRow);
        let width = 8;
        let height = 8;
        let mut pressure = vec![0.0; width * height];
        let mut pressure_prev = vec![0.0; width * height];

        pressure[3 * width + 3] = 1.0;

        let c2 = 0.25;
        let damping = 0.99;

        // First frame should process row 1
        let result = processor.step_frame(&mut pressure, &mut pressure_prev, width, height, c2, damping);
        assert!(!result.step_complete);
        assert_eq!(processor.state.current_row, Some(1));

        // Continue until complete
        let mut iterations = 0;
        loop {
            let result = processor.step_frame(&mut pressure, &mut pressure_prev, width, height, c2, damping);
            if result.step_complete {
                break;
            }
            iterations += 1;
            assert!(iterations < 20);
        }
    }

    #[test]
    fn test_actor_based_processing() {
        let mut processor = EducationalProcessor::new(SimulationMode::ActorBased);
        processor.tile_size = 4; // Small tiles for testing

        let width = 16;
        let height = 16;
        let mut pressure = vec![0.0; width * height];
        let mut pressure_prev = vec![0.0; width * height];

        pressure[8 * width + 8] = 1.0;

        let c2 = 0.25;
        let damping = 0.99;

        // First frame should process first tile
        let result = processor.step_frame(&mut pressure, &mut pressure_prev, width, height, c2, damping);
        assert!(!result.step_complete);
        assert!(!processor.state.active_tiles.is_empty());

        // Continue until complete
        let mut iterations = 0;
        loop {
            let result = processor.step_frame(&mut pressure, &mut pressure_prev, width, height, c2, damping);
            if result.step_complete {
                break;
            }
            iterations += 1;
            assert!(iterations < 50);
        }
    }
}
