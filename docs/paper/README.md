# RingKernel: A GPU-Native Persistent Actor Model

Technical paper describing the design, implementation, and evaluation of RingKernel.

## Abstract

The actor model has become foundational for building concurrent and distributed systems.
However, existing implementations target CPU architectures, leaving GPU parallelism
largely unexplored. RingKernel extends the actor model to GPU computing, treating GPU
thread blocks as persistent actors with lock-free message passing and Hybrid Logical
Clocks for causal ordering.

Key contributions:
- Formalization of GPU actor semantics with H2K/K2H/K2K communication channels
- 128-byte ControlBlock for GPU-resident actor lifecycle management
- Hybrid Logical Clocks on GPU for causal ordering
- Rust-to-CUDA transpiler for actor kernel generation
- Evaluation showing 11,327× lower command latency vs traditional GPU programming

## Building the Paper

### Prerequisites

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- pdflatex and bibtex
- Optional: latexmk for continuous builds

### Build Commands

```bash
# Build PDF
make

# Quick build (skip bibliography)
make quick

# Continuous rebuild on file changes
make watch

# Clean build artifacts
make clean

# Create arxiv submission tarball
make arxiv

# Word count
make wordcount
```

## Directory Structure

```
paper/
├── main.tex              # Main document
├── references.bib        # Bibliography
├── Makefile             # Build system
├── README.md            # This file
├── sections/
│   ├── 00-abstract.tex
│   ├── 01-introduction.tex
│   ├── 02-background.tex
│   ├── 03-related-work.tex
│   ├── 04-system-design.tex
│   ├── 05-implementation.tex
│   ├── 06-evaluation.tex
│   ├── 07-discussion.tex
│   ├── 08-conclusion.tex
│   └── 09-appendix.tex
└── figures/
    └── README.md        # Figure guidelines
```

## Target Venues

This paper is suitable for submission to:

### Systems Conferences
- **ASPLOS** - Architectural Support for Programming Languages and Operating Systems
- **EuroSys** - European Conference on Computer Systems
- **OSDI/SOSP** - Operating Systems Design and Implementation
- **ATC** - USENIX Annual Technical Conference

### Programming Languages
- **PLDI** - Programming Language Design and Implementation
- **OOPSLA** - Object-Oriented Programming, Systems, Languages & Applications
- **PPoPP** - Principles and Practice of Parallel Programming

### GPU/HPC
- **SC** - Supercomputing Conference
- **ICS** - International Conference on Supercomputing
- **CGO** - Code Generation and Optimization

### Preprint
- **arXiv** - cs.DC (Distributed Computing), cs.PL (Programming Languages)

## Paper Metadata

- **Title**: RingKernel: A GPU-Native Persistent Actor Model for High-Performance Concurrent Computing
- **Keywords**: Actor Model, GPU Computing, Persistent Kernels, Message Passing, CUDA, Hybrid Logical Clocks, Lock-Free Algorithms
- **ACM CCS Concepts**:
  - Software and its engineering → Concurrent programming languages
  - Computer systems organization → Heterogeneous (hybrid) systems
  - Software and its engineering → Message passing

## Reproducing Results

The evaluation benchmarks can be reproduced using the RingKernel repository:

```bash
# Clone repository
git clone https://github.com/mivertowski/RustCompute
cd RustCompute

# Build with CUDA support
cargo build --release --features cuda

# Run throughput benchmark
cargo run -p ringkernel-wavesim3d --bin wavesim3d-benchmark --release --features cuda-codegen

# Run latency benchmark
cargo run -p ringkernel-wavesim3d --bin interactive-benchmark --release --features cuda-codegen
```

Hardware requirements:
- NVIDIA GPU with Compute Capability 6.0+ (Pascal or newer)
- CUDA 12.0+
- 8GB+ GPU memory recommended

## Contributing

To contribute to the paper:

1. Edit the appropriate section file in `sections/`
2. Add references to `references.bib`
3. Run `make check` to verify no TODOs or undefined references
4. Run `make` to verify the paper builds
5. Submit a pull request

## License

The paper content is licensed under CC BY 4.0.
The RingKernel software is licensed under Apache 2.0 / MIT dual license.

## Citation

If you use RingKernel in your research, please cite:

```bibtex
@inproceedings{ringkernel2024,
  author    = {Ivertowski, Michael},
  title     = {RingKernel: A GPU-Native Persistent Actor Model for
               High-Performance Concurrent Computing},
  booktitle = {Proceedings of [Conference]},
  year      = {2024},
  note      = {Preprint: https://arxiv.org/abs/XXXX.XXXXX}
}
```
