# Figures for RingKernel Paper

This directory contains figures for the technical paper. Most figures are generated
inline using TikZ/PGFPlots in the LaTeX source. Additional figures to be added:

## Required Figures

### Architecture Diagram (Figure 1)
- **File**: `architecture.pdf` or generated via TikZ
- **Content**: High-level RingKernel architecture showing H2K/K2H/K2K channels
- **Status**: Generated inline via TikZ in `04-system-design.tex`

### Latency Comparison (Figure 2)
- **File**: Generated via PGFPlots
- **Content**: Log-log plot of command latency vs steps per command
- **Status**: Generated inline in `06-evaluation.tex`

### Mixed Workload Breakdown (Figure 3)
- **File**: Generated via PGFPlots
- **Content**: Stacked bar chart of compute vs command time
- **Status**: Generated inline in `06-evaluation.tex`

## Optional Figures (for camera-ready)

### Message Envelope Format
- **File**: `envelope-format.pdf`
- **Content**: Visual representation of 256-byte message header
- **Tool**: Draw.io, Figma, or TikZ

### ControlBlock Structure
- **File**: `controlblock.pdf`
- **Content**: Memory layout of 128-byte ControlBlock
- **Tool**: Draw.io, Figma, or TikZ

### K2K Routing Table
- **File**: `k2k-routing.pdf`
- **Content**: Diagram of kernel-to-kernel message routing
- **Tool**: Draw.io, Figma, or TikZ

### GPU Memory Hierarchy
- **File**: `memory-hierarchy.pdf`
- **Content**: GPU memory hierarchy with RingKernel annotations
- **Tool**: Draw.io, Figma, or TikZ

## Export Guidelines

For camera-ready submission:
- Export vector formats (PDF, EPS) at 300+ DPI
- Ensure fonts are embedded
- Use consistent color scheme (blue for host, green for GPU)
- Keep text readable at column width (~3.5 inches)

## Color Palette

Suggested consistent colors:
- Host/CPU: `#3498db` (blue)
- GPU/Device: `#27ae60` (green)
- Mapped Memory: `#f1c40f` (yellow)
- K2K: `#e67e22` (orange)
- Errors: `#e74c3c` (red)
