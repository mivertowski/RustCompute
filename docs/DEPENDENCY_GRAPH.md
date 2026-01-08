# Dependency Graph

> Implementation Dependencies and Critical Paths

## Overview

This document visualizes the dependencies between implementation milestones, identifies critical paths, and helps with parallel work planning.

---

## Phase Dependencies

### Phase 1: Foundation Completion

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    PHASE 1                              │
                    └─────────────────────────────────────────────────────────┘

   Week 1-4              Week 5-8              Week 9-10            Week 11-12
   ─────────             ────────              ─────────            ──────────

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  1.1 Metal   │───▶│  1.2 Metal   │───▶│  1.3 Metal   │
│    Core      │    │  Persistent  │    │     K2K      │
└──────────────┘    └──────────────┘    └──────────────┘

                                                            ┌──────────────┐
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │  1.5 Eco     │
│  1.4 WebGPU  │───▶│  1.4 WebGPU  │───▶│  1.4 WebGPU  │    │  Streaming   │
│   Batching   │    │   Atomics    │    │  Subgroups   │    └──────────────┘
└──────────────┘    └──────────────┘    └──────────────┘           │
                                                                    │
                                                                    │ (parallel)
                                                                    ▼
```

**Critical Path**: 1.1 → 1.2 → 1.3 (Metal backend chain)

**Parallel Tracks**:
- Metal backend (1.1 → 1.2 → 1.3)
- WebGPU optimization (1.4)
- Ecosystem streaming (1.5)

---

### Phase 2: Unified Code Generation

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    PHASE 2                              │
                    └─────────────────────────────────────────────────────────┘

        Week 1-4                    Week 5-8                    Week 9-12
        ────────                    ────────                    ─────────

                                ┌──────────────┐
                           ┌───▶│  2.2 CUDA    │────────┐
                           │    │   Lowering   │        │
                           │    └──────────────┘        │
    ┌──────────────┐       │                            │    ┌──────────────┐
    │   2.1 IR     │───────┼───▶┌──────────────┐        ├───▶│ 2.5 Multi-   │
    │  Foundation  │       │    │  2.3 WGSL    │────────┤    │   Backend    │
    └──────────────┘       │    │   Lowering   │        │    │ Proc Macros  │
                           │    └──────────────┘        │    └──────────────┘
                           │                            │
                           │    ┌──────────────┐        │
                           └───▶│  2.4 MSL     │────────┘
                                │   Lowering   │
                                └──────────────┘
                                       │
                                       │ (requires Phase 1.2)
                                       ▼
                              ┌──────────────────┐
                              │ Metal backend    │
                              │ for testing      │
                              └──────────────────┘
```

**Critical Path**: 2.1 → 2.4 → 2.5 (IR to MSL to proc macros)

**Dependencies**:
- 2.2, 2.3, 2.4 all depend on 2.1
- 2.5 depends on all of 2.2, 2.3, 2.4
- 2.4 needs Phase 1.2 (Metal persistent) for testing

---

### Phase 3: Enterprise Features

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    PHASE 3                              │
                    └─────────────────────────────────────────────────────────┘

        Week 1-4                    Week 5-8                    Week 9-12
        ────────                    ────────                    ─────────

    ┌──────────────┐
    │     3.1      │
    │ Checkpointing│ ─────────────────────────────────────────────────────────▶
    └──────────────┘

                            ┌──────────────┐
                       ┌───▶│   3.2.1      │
                       │    │ Topology     │
                       │    └──────────────┘
    ┌──────────────┐   │           │
    │   3.2 Multi  │───┤           ▼
    │     GPU      │   │    ┌──────────────┐    ┌──────────────┐
    └──────────────┘   │    │   3.2.2      │───▶│   3.2.3      │
                       │    │ Cross-GPU    │    │   Kernel     │
                       │    │    K2K       │    │  Migration   │
                       └───▶└──────────────┘    └──────────────┘

                                                ┌──────────────┐
    ┌──────────────┐                            │    3.4       │
    │     3.3      │──────────────────────────▶ │ Resilience   │
    │ Observability│                            │              │
    └──────────────┘                            └──────────────┘
```

**Critical Path**: 3.2.1 → 3.2.2 → 3.2.3 (Multi-GPU chain)

**Parallel Tracks**:
- Checkpointing (3.1)
- Multi-GPU (3.2)
- Observability (3.3) → Resilience (3.4)

---

### Phase 4: Ecosystem Expansion

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    PHASE 4                              │
                    └─────────────────────────────────────────────────────────┘

        Week 1-4                    Week 5-8                    Week 9-12
        ────────                    ────────                    ─────────

    ┌──────────────┐    ┌──────────────┐
    │  4.1 Arrow   │───▶│  4.1 Polars  │
    └──────────────┘    └──────────────┘
           │                   │
           │                   │
           ▼                   ▼
    ┌──────────────┐    ┌──────────────┐
    │  4.1 Candle  │    │ 4.1 DataFus  │
    └──────────────┘    └──────────────┘


    ┌──────────────┐    ┌──────────────┐
    │   4.2 CLI    │───▶│ 4.2 VSCode   │──────────────────────────┐
    │    Core      │    │  Extension   │                          │
    └──────────────┘    └──────────────┘                          │
                                                                   │
                                                                   ▼
                                                          ┌──────────────┐
                                                          │    4.3       │
                                                          │Documentation │
                                                          │              │
                                                          └──────────────┘
```

**Dependencies**:
- 4.1 depends on Phase 2 (code generation)
- 4.2 depends on Phase 2 (code generation)
- 4.3 depends on all previous phases (for accuracy)

---

## Cross-Phase Dependencies

```
                        PHASE 1              PHASE 2              PHASE 3              PHASE 4
                        ───────              ───────              ───────              ───────

                    ┌──────────────┐
                    │  1.1 Metal   │
                    │    Core      │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐                         ┌──────────────┐
                    │  1.2 Metal   │────────────────────────▶│     3.1      │
                    │  Persistent  │                         │ Checkpointing│
                    └──────┬───────┘                         └──────────────┘
                           │
                           ├──────────────────────────┐
                           │                          │
                           ▼                          ▼
                    ┌──────────────┐           ┌──────────────┐
                    │  1.3 Metal   │           │  2.4 MSL     │
                    │     K2K      │           │   Lowering   │────────┐
                    └──────────────┘           └──────────────┘        │
                                                      │                │
                                                      │                │
    ┌──────────────┐                                  │                │
    │   2.1 IR     │───────────────────────────────── │ ───────────────┤
    │  Foundation  │                                  │                │
    └──────┬───────┘                                  │                │
           │                                          │                │
           ├───────────────────────────┐              │                │
           │                           │              │                │
           ▼                           ▼              ▼                │
    ┌──────────────┐           ┌──────────────┐   ┌──────────────┐     │
    │  2.2 CUDA    │           │  2.3 WGSL    │   │ 2.5 Multi-   │◀────┘
    │   Lowering   │           │   Lowering   │   │   Backend    │
    └──────────────┘           └──────────────┘   └──────┬───────┘
                                                         │
                                                         │
                               ┌──────────────────────── │ ────────────────┐
                               │                         │                 │
                               ▼                         ▼                 ▼
                        ┌──────────────┐          ┌──────────────┐  ┌──────────────┐
                        │  4.1 Data    │          │   4.2 CLI    │  │    4.3       │
                        │  Processing  │          │   Tooling    │  │Documentation │
                        └──────────────┘          └──────────────┘  └──────────────┘
```

---

## Critical Paths Analysis

### Longest Critical Path
```
1.1 → 1.2 → 2.4 → 2.5 → 4.2 → 4.3
(4 weeks) (4 weeks) (4 weeks) (2 weeks) (4 weeks) (4 weeks) = 22 weeks
```

### Parallel Work Opportunities

| Time Period | Track A | Track B | Track C |
|-------------|---------|---------|---------|
| Q1 Wk 1-4 | 1.1 Metal Core | 1.4 WebGPU | 1.5 Streaming |
| Q1 Wk 5-8 | 1.2 Metal Persistent | 1.4 WebGPU cont. | 1.5 Streaming |
| Q1 Wk 9-12 | 1.3 Metal K2K | 2.1 IR Foundation | - |
| Q2 Wk 1-4 | 2.2 CUDA Lowering | 2.3 WGSL Lowering | 3.3 Observability |
| Q2 Wk 5-8 | 2.2 cont. | 2.3 cont. | 2.4 MSL Lowering |
| Q2 Wk 9-12 | 2.5 Multi-Backend | 3.1 Checkpointing | 3.4 Resilience |
| Q3 Wk 1-4 | 3.2 Multi-GPU | 4.1 Arrow | - |
| Q3 Wk 5-8 | 3.2 cont. | 4.1 Polars/Candle | 4.2 CLI Core |
| Q3 Wk 9-12 | 3.2 cont. | 4.2 VSCode | - |
| Q4 Wk 1-12 | 4.3 Documentation | 4.1 DataFusion | Stabilization |

---

## Dependency Matrix

### Phase 1 Dependencies

| Milestone | Depends On | Blocks |
|-----------|------------|--------|
| 1.1 Metal Core | None | 1.2 |
| 1.2 Metal Persistent | 1.1 | 1.3, 2.4, 3.1 |
| 1.3 Metal K2K | 1.2 | None |
| 1.4 WebGPU Opt | None | None |
| 1.5 Streaming | None | None |

### Phase 2 Dependencies

| Milestone | Depends On | Blocks |
|-----------|------------|--------|
| 2.1 IR Foundation | None | 2.2, 2.3, 2.4 |
| 2.2 CUDA Lowering | 2.1 | 2.5 |
| 2.3 WGSL Lowering | 2.1 | 2.5 |
| 2.4 MSL Lowering | 2.1, 1.2 | 2.5 |
| 2.5 Multi-Backend | 2.2, 2.3, 2.4 | 4.1, 4.2 |

### Phase 3 Dependencies

| Milestone | Depends On | Blocks |
|-----------|------------|--------|
| 3.1 Checkpointing | 1.2 | None |
| 3.2 Multi-GPU | 1.2 | None |
| 3.3 Observability | None | 3.4 |
| 3.4 Resilience | 3.3 | None |

### Phase 4 Dependencies

| Milestone | Depends On | Blocks |
|-----------|------------|--------|
| 4.1 Data Processing | 2.5 | 4.3 |
| 4.2 CLI Tooling | 2.5 | 4.3 |
| 4.3 Documentation | All | None |

---

## Resource Allocation by Phase

### Phase 1 Resource Plan

```
         Engineer A        Engineer B        Engineer C
         ──────────        ──────────        ──────────
Week 1   [1.1 Metal Core]  [1.4 WebGPU]     [1.5 Streaming]
Week 2   [1.1 Metal Core]  [1.4 WebGPU]     [1.5 Streaming]
Week 3   [1.1 Metal Core]  [1.4 WebGPU]     [1.5 Streaming]
Week 4   [1.1 Metal Core]  [1.4 WebGPU]     [1.5 Streaming]
Week 5   [1.2 Persistent]  [1.4 WebGPU]     [Review/Test]
Week 6   [1.2 Persistent]  [1.4 WebGPU]     [Review/Test]
Week 7   [1.2 Persistent]  [1.4 WebGPU]     [2.1 IR Start]
Week 8   [1.2 Persistent]  [1.4 WebGPU]     [2.1 IR Start]
Week 9   [1.3 Metal K2K]   [2.1 IR Found]   [2.1 IR Found]
Week 10  [1.3 Metal K2K]   [2.1 IR Found]   [2.1 IR Found]
Week 11  [Integration]     [2.1 IR Found]   [2.1 IR Found]
Week 12  [Integration]     [2.1 IR Found]   [2.1 IR Found]
```

---

## Milestone Ordering Rules

### Must Complete Before

| To Start This | Must Complete These First |
|---------------|---------------------------|
| 1.2 Metal Persistent | 1.1 Metal Core |
| 1.3 Metal K2K | 1.2 Metal Persistent |
| 2.2 CUDA Lowering | 2.1 IR Foundation |
| 2.3 WGSL Lowering | 2.1 IR Foundation |
| 2.4 MSL Lowering | 2.1 IR Foundation, 1.2 Metal Persistent |
| 2.5 Multi-Backend | 2.2, 2.3, 2.4 |
| 3.1 Checkpointing | 1.2 Metal Persistent (or CUDA equivalent) |
| 3.2 Multi-GPU | 1.2 Metal Persistent (or CUDA equivalent) |
| 3.4 Resilience | 3.3 Observability |
| 4.1 Data Processing | 2.5 Multi-Backend |
| 4.2 CLI Tooling | 2.5 Multi-Backend |
| 4.3 Documentation | All previous milestones |

### Can Run In Parallel

| Parallel Group | Milestones |
|----------------|------------|
| Phase 1 Start | 1.1, 1.4, 1.5 |
| Phase 2 Lowering | 2.2, 2.3, 2.4 (after 2.1) |
| Phase 3 Features | 3.1, 3.2, 3.3 |
| Phase 4 Ecosystem | 4.1, 4.2 |

---

## Risk Dependencies

### High-Risk Dependencies

| Dependency | Risk | Mitigation |
|------------|------|------------|
| 1.2 → 2.4 | Metal persistent needed for MSL testing | Early prototype Metal persistent |
| 2.1 → 2.2/2.3/2.4 | IR design may need iteration | MVP IR first, extend later |
| Phase 2 → Phase 4 | Codegen changes may break integrations | Stable API freeze for Phase 4 |

### Schedule Buffer Recommendations

| Phase | Buffer | Reason |
|-------|--------|--------|
| Phase 1 | +2 weeks | Metal API uncertainty |
| Phase 2 | +3 weeks | IR design complexity |
| Phase 3 | +1 week | Multi-GPU testing time |
| Phase 4 | +2 weeks | Documentation thoroughness |

---

## Quick Reference

### Start Here (No Dependencies)
- 1.1 Metal Core
- 1.4 WebGPU Optimization
- 1.5 Ecosystem Streaming
- 2.1 IR Foundation (after 1.1 complete)
- 3.3 Observability

### End Points (Nothing Depends On These)
- 1.3 Metal K2K
- 3.1 Checkpointing
- 3.2 Multi-GPU
- 3.4 Resilience
- 4.3 Documentation

### Blocking Critical Path Items
- 1.1 Metal Core (blocks 1.2)
- 1.2 Metal Persistent (blocks 1.3, 2.4, 3.1, 3.2)
- 2.1 IR Foundation (blocks 2.2, 2.3, 2.4)
- 2.5 Multi-Backend (blocks 4.1, 4.2)
