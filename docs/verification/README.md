RingKernel Formal Verification (TLA+)
======================================

This directory contains TLA+ models of RingKernel's core protocols.
The models cover safety properties required by the v1.1 specification
(section 5.2). TLC (the TLA+ model checker) can exhaustively explore
small bounded configurations of each model to check that the stated
invariants hold on every reachable state.

These specs are intended to be informally reviewed by reading, and
then machine-checked on dedicated hardware during the v1.1 hardware
phase. TLC is not required to be run as part of normal development.

Specifications
--------------

| File                    | Protocol                                  | Lines |
|-------------------------|-------------------------------------------|-------|
| hlc.tla                 | Hybrid Logical Clock monotonicity         | ~120  |
| k2k_delivery.tla        | Single-GPU kernel-to-kernel messaging     | ~140  |
| migration.tla           | 3-phase actor migration state machine     | ~180  |
| multi_gpu_k2k.tla       | Cross-GPU K2K with NVLink P2P / fallback  | ~170  |
| tenant_isolation.tla    | Tenant routing isolation                  | ~130  |
| actor_lifecycle.tla     | Actor create/destroy/restart              | ~150  |

Each `.tla` file has a matching `.cfg` that tells TLC which constants
and invariants to check.

Invariants vs. Liveness
-----------------------

The models focus on safety (bad things never happen). Liveness
properties (good things eventually happen) are limited to temporal
properties checkable with finite fairness (e.g., migrations eventually
complete once started). Unbounded liveness (e.g., every message is
eventually delivered) is beyond TLC's bounded-model capabilities and
is not asserted here.

Running TLC
-----------

TLC is part of the TLA+ Tools distribution. Three common ways to run:

1) Native install (Java 11+):

       # Download tla2tools.jar from https://github.com/tlaplus/tlaplus/releases
       java -XX:+UseParallelGC -jar tla2tools.jar -config hlc.cfg hlc.tla

2) Docker image `tlaplus/tlaplus`:

       docker run --rm -v "$PWD:/spec" -w /spec tlaplus/tlaplus \
           java -jar /opt/TLA+Tools/tla2tools.jar -config hlc.cfg hlc.tla

3) Via the provided helper script `tlc.sh`, which runs every spec in
   this directory:

       ./tlc.sh                       # runs all specs
       ./tlc.sh hlc                   # runs only hlc.tla
       TLC_CMD="docker run ..." ./tlc.sh

State Space Sizing
------------------

Each model is configured with small constants (2-3 nodes, <= 10
events, <= 4 messages) so TLC completes in seconds. Larger
configurations can be explored by editing the `.cfg` files, but state
space grows exponentially; consult `docs/verification/tlc.sh` for
sensible defaults.

Correspondence to the Implementation
------------------------------------

| TLA+ concept          | Rust implementation                             |
|-----------------------|-------------------------------------------------|
| clocks[node]          | `HlcClock` in `crates/ringkernel-core/src/hlc.rs` |
| Send / Deliver        | `K2K` trait in `crates/ringkernel-core/src/k2k.rs` |
| MigrationState        | `MigrationState` enum in `multi_gpu.rs`         |
| routing_table         | `ControlBlock` + `RingContext` routing          |
| tenant_id             | `TenantId` in `core/src/multi_tenancy.rs`       |
| actor_states          | `ActorState` in the actor runtime module        |

The TLA+ models intentionally abstract over low-level details
(buffer sizes, exact timings, GPU block geometry) so that the
invariants remain protocol-level.
