# WaveSim Educational Simulation Modes

WaveSim includes animated simulation modes that visually demonstrate the evolution of parallel computing paradigms. These modes are designed for teaching and presentations about the history and concepts of parallel computing.

## Overview

The educational modes slow down simulation to show how different computing paradigms process data. Each mode visualizes its processing pattern with colored overlays on the wave simulation grid.

## Available Modes

### 1. Cell-by-Cell (1950s - Sequential Computing)

**Era**: Early electronic computers (ENIAC, UNIVAC)

**Behavior**: Processes one cell at a time, left-to-right, top-to-bottom with visible delay between cells.

**Visual**: Green highlight on the currently processing cell.

**Lesson**: "This is how early computers worked - one operation at a time. Fast for its era, but fundamentally limited by serial execution."

### 2. Row-by-Row (1970s - Vector Processing)

**Era**: Cray-1, early vector supercomputers

**Behavior**: Processes entire rows at once, sweeping from top to bottom.

**Visual**: Yellow highlight on the current row being processed.

**Lesson**: "Vector processors and SIMD - process multiple data elements with one instruction. A major leap in scientific computing."

### 3. Chaotic Parallel (1990s - Naive Parallelism)

**Era**: Early multi-processor systems without proper synchronization

**Behavior**: All cells update simultaneously but with random delays. Cells finish in unpredictable order.

**Visual**: Random cells flash green as they complete in arbitrary order.

**Lesson**: "Parallel without coordination = chaos and race conditions. More processors doesn't mean correct results."

### 4. Synchronized Parallel (2000s - Barrier Synchronization)

**Era**: MPI, OpenMP, early GPGPU

**Behavior**: All cells compute in parallel, then synchronize before the next step. Shows clean wavefronts but with visible "pause" at sync points.

**Visual**: All interior cells flash green simultaneously, then pause.

**Lesson**: "Barriers ensure correctness but create bottlenecks. The slowest thread determines overall performance."

### 5. Actor-Based with HLC (Modern - Causal Consistency)

**Era**: Modern distributed systems, RingKernel actor model

**Behavior**: Tiles process independently with temporal ordering via Hybrid Logical Clocks. Smooth, correct simulation with visible tile boundaries.

**Visual**: Cyan tiles with cyan borders highlight active tiles.

**Lesson**: "Actors + causal ordering = parallelism without chaos. Independent units coordinate through message passing and logical timestamps."

### 6. Standard (Production)

**Era**: Modern optimized code

**Behavior**: Full-speed parallel processing with no visualization delays.

**Visual**: Normal wave simulation display.

**Use**: Production simulation, benchmarking.

## Visual Indicators

| Indicator | Color | Mode(s) | Meaning |
|-----------|-------|---------|---------|
| Cell highlight | Green | CellByCell, ChaoticParallel, SynchronizedParallel | Cell being processed |
| Row highlight | Yellow | RowByRow | Current row being processed |
| Tile highlight | Cyan (fill) | ActorBased | Active tile region |
| Tile border | Cyan (stroke) | ActorBased | Tile boundary |

## Configuration

The educational processor can be configured programmatically:

```rust
use ringkernel_wavesim::simulation::{SimulationMode, EducationalProcessor};

// Create processor with specific mode
let mut processor = EducationalProcessor::new(SimulationMode::CellByCell);

// Adjust animation speed
processor.cells_per_frame = 16;   // Cells processed per GUI frame
processor.rows_per_frame = 2;     // Rows per frame (RowByRow)
processor.tiles_per_frame = 4;    // Tiles per frame (ActorBased)
```

## GUI Usage

1. Launch WaveSim: `cargo run -p ringkernel-wavesim --bin wavesim --release`
2. Select a mode from the "Simulation Mode" dropdown
3. Click "Play" to start the simulation
4. Observe the visual patterns showing how each paradigm processes data

## Teaching Tips

1. **Start with Cell-by-Cell**: Show how painfully slow sequential processing is
2. **Progress to Row-by-Row**: Demonstrate the power of vectorization
3. **Show Chaotic Parallel**: Illustrate why synchronization matters (artifacts visible)
4. **Demonstrate Synchronized**: Show correctness with barriers
5. **End with Actor-Based**: Modern approach balancing parallelism and correctness

## Technical Implementation

The educational modes are implemented in `crates/ringkernel-wavesim/src/simulation/educational.rs`:

- `SimulationMode` enum: Defines all available modes
- `EducationalProcessor`: Manages incremental simulation state
- `ProcessingState`: Tracks which cells/rows/tiles are being processed
- `StepResult`: Indicates when a simulation step completes

Each mode implements incremental FDTD wave equation updates, processing only a subset of cells per animation frame to create the visual effect.
