# RingKernel Documentation

> GPU-Native Persistent Actor Model Framework for Rust

## Overview

RingKernel enables GPU-accelerated actor systems with persistent kernels, lock-free message passing, and hybrid logical clocks (HLC) for causal ordering. This documentation provides comprehensive coverage of architecture, specifications, and roadmap.

## Documents

### Strategy & Roadmap

| Document | Description |
|----------|-------------|
| [**ROADMAP.md**](../ROADMAP.md) | Master roadmap with phases, milestones, and timeline |

### Technical Specifications

| Document | Description |
|----------|-------------|
| [**ARCHITECTURE_ANALYSIS.md**](ARCHITECTURE_ANALYSIS.md) | Current state analysis of all backends and subsystems |
| [**PERSISTENT_KERNEL_SPEC.md**](PERSISTENT_KERNEL_SPEC.md) | Backend-agnostic persistent kernel specification |

### Feature Plans

| Document | Description |
|----------|-------------|
| [**ENTERPRISE_FEATURES.md**](ENTERPRISE_FEATURES.md) | Enterprise-grade features: resilience, security, compliance |
| [**DEVELOPER_EXPERIENCE.md**](DEVELOPER_EXPERIENCE.md) | Tooling, testing, and developer productivity |

### Implementation & Testing

| Document | Description |
|----------|-------------|
| [**IMPLEMENTATION_PLAN.md**](IMPLEMENTATION_PLAN.md) | Phased implementation with sprints, tasks, and deliverables |
| [**TESTING_STRATEGY.md**](TESTING_STRATEGY.md) | Comprehensive testing strategy across all backends |
| [**MILESTONE_CHECKLIST.md**](MILESTONE_CHECKLIST.md) | Trackable milestones with acceptance criteria |
| [**DEPENDENCY_GRAPH.md**](DEPENDENCY_GRAPH.md) | Implementation dependencies and critical paths |

## Quick Navigation

### By Topic

**Getting Started**
- [CLAUDE.md](../CLAUDE.md) - Build commands and project overview
- [ROADMAP.md](../ROADMAP.md) - Project direction and priorities

**Architecture**
- [Architecture Analysis](ARCHITECTURE_ANALYSIS.md) - Current implementation status
- [Persistent Kernel Spec](PERSISTENT_KERNEL_SPEC.md) - Core abstractions and protocols

**Enterprise**
- [Enterprise Features](ENTERPRISE_FEATURES.md) - Fault tolerance, security, compliance

**Developer Experience**
- [DX Roadmap](DEVELOPER_EXPERIENCE.md) - CLI, IDE, testing, documentation

**Implementation & Testing**
- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Sprints, tasks, deliverables
- [Testing Strategy](TESTING_STRATEGY.md) - Test pyramid, coverage, CI/CD
- [Milestone Checklist](MILESTONE_CHECKLIST.md) - Progress tracking
- [Dependency Graph](DEPENDENCY_GRAPH.md) - Critical paths and parallelization

### By Backend

| Backend | Status | Key Documents |
|---------|--------|---------------|
| **CUDA** | ✅ Complete | [Architecture](ARCHITECTURE_ANALYSIS.md#cuda-backend-analysis) |
| **WebGPU** | ⚠️ Limited | [Architecture](ARCHITECTURE_ANALYSIS.md#webgpu-backend-analysis) |
| **Metal** | 🔧 Scaffold | [Roadmap](../ROADMAP.md#11-metal-backend-implementation) |
| **CPU** | ✅ Complete | [Architecture](ARCHITECTURE_ANALYSIS.md#cpu-backend-analysis) |

## Key Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Command Injection Latency | 0.03µs (CUDA) | <0.1µs (all backends) |
| Backend Coverage | 2/3 production-ready | 3/3 |
| Test Count | 825+ | 1000+ |
| Speedup vs Traditional | 11,327x | >10,000x |

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to documentation and implementation.

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01 | 1.0 | Initial documentation suite |
