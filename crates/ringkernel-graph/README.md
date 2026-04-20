# ringkernel-graph

Graph algorithm primitives for the RingKernel workspace.

## Overview

`ringkernel-graph` provides sparse-graph data structures and parallel algorithms
intended for GPU acceleration through `ringkernel-core`. The algorithms are
implemented against CPU reference paths today, with CUDA kernels gated behind
the `cuda` feature.

Included:

- **CSR matrix** (`CsrMatrix`, `CsrMatrixBuilder`) — compressed sparse row
  storage with builder-style construction from edge lists.
- **BFS** (`bfs_parallel`, `bfs_sequential`) — parallel and sequential
  breadth-first search with multiple source support.
- **SCC** (`scc_kosaraju`, `scc_tarjan`) — strongly connected components via
  Kosaraju or Tarjan's algorithm.
- **Union-Find** (`union_find_parallel`, `union_find_sequential`,
  `UnionFind`) — disjoint-set structure with path compression.
- **SpMV** (`spmv`, `spmv_parallel`) — sparse matrix/vector multiplication.

All primitives are designed to be zero-copy friendly and compatible with the
persistent-kernel runtime in `ringkernel-core`.

## Relationship to `ringkernel-core`

This crate depends on `ringkernel-core` for shared error handling, tracing, and
runtime integration. It does **not** depend on a specific backend crate; the
CPU path is always available, and CUDA kernels are pulled in via the `cuda`
feature.

## Feature flags

| Flag | Default | Description                                               |
|------|---------|-----------------------------------------------------------|
| `cuda` | off   | Enable CUDA kernels via `cudarc` (requires a CUDA install) |

The default build exposes only the CPU algorithms, so the crate is usable on
machines without a GPU.

## Example

```rust
use ringkernel_graph::{bfs_parallel, CsrMatrix, NodeId};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a simple chain graph: 0 -> 1 -> 2
    let matrix = CsrMatrix::from_edges(3, &[(0, 1), (1, 2)]);

    // Run BFS starting from node 0
    let distances = bfs_parallel(&matrix, &[NodeId(0)]).await?;

    assert_eq!(distances[0].0, 0); // distance to self
    assert_eq!(distances[1].0, 1);
    assert_eq!(distances[2].0, 2);
    Ok(())
}
```

See the crate-level rustdoc for additional examples and the full API.

## Workspace context

For an overview of how this crate fits into the wider RingKernel workspace, see
[`docs/02-crate-structure.md`](../../docs/02-crate-structure.md).

## Status

v1.1.0 — part of the RingKernel 1.1 release. API is considered stable within
the 1.x series; breaking changes will follow semver.

## License

Apache-2.0. See the repository-level `LICENSE` file.
