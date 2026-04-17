//! NVLink / P2P topology detection.
//!
//! Multi-GPU actor placement needs to know which GPUs are directly connected
//! (and with what bandwidth) vs. only reachable over PCIe. This module probes
//! the local system at runtime and exposes a small adjacency-matrix API.
//!
//! # Detection strategy
//!
//! 1. If the `multi-gpu` feature is enabled, use [nvml-wrapper] to query NVLink
//!    state and link version per device, then pair devices via their PCI bus
//!    ID (`nvmlDeviceGetNvLinkState` + `nvmlDeviceGetNvLinkRemotePciInfo_v2`).
//!    This is the only way to get the *NVLink-specific* bandwidth — CUDA's
//!    driver-level `cuDeviceGetP2PAttribute` tells us P2P is possible but does
//!    not distinguish NVLink from PCIe.
//! 2. Fall back to CUDA driver probing via [`cuDeviceCanAccessPeer`] and
//!    [`cuDeviceGetP2PAttribute`]. When these are the only signal available,
//!    any P2P-capable pair is assumed to be NVLink-connected at a nominal
//!    fallback bandwidth (see [`FALLBACK_P2P_BANDWIDTH_GBPS`]).
//! 3. When no CUDA device is present (no GPU on this host, CI, etc.) we
//!    return a degenerate single-node topology — useful for unit tests and
//!    for code paths that want to reason about "the only local GPU".
//!
//! # API shape
//!
//! [`NvlinkTopology`] stores a dense adjacency matrix. Entries are bandwidth
//! in GB/s (bidirectional), with `0` meaning "no direct link". A self-loop is
//! modelled as the GPU's own memory bandwidth placeholder (also `0` in the
//! adjacency matrix; distance `Some(vec![i])`).
//!
//! Shortest-path distances are computed via BFS on the unweighted graph (one
//! hop per direct link) — this matches the spec's "shortest hop count" model
//! and is cheap for the small number of GPUs in a single node (typically 2-8).
//!
//! [nvml-wrapper]: https://docs.rs/nvml-wrapper

#![allow(dead_code)]

use std::collections::VecDeque;
use std::fmt;

use ringkernel_core::error::{Result, RingKernelError};

/// Fallback per-link bandwidth (GB/s) when NVML is not available but CUDA P2P
/// is supported between two GPUs. Conservative — approximates NVLink 2.0 which
/// is common on V100 / A100 / H100 hardware.
pub const FALLBACK_P2P_BANDWIDTH_GBPS: u32 = 25;

/// Bandwidth (GB/s) per NVLink *generation*. These are the advertised per-link
/// bidirectional bandwidths. Real per-GPU aggregate bandwidth depends on how
/// many links exist between a given pair — we scale by link count in
/// [`NvlinkTopology::probe`].
const NVLINK_GEN_BANDWIDTH_GBPS: &[(u32, u32)] = &[
    (1, 20), // NVLink 1.0: Pascal (P100)
    (2, 25), // NVLink 2.0: Volta (V100)
    (3, 25), // NVLink 3.0: Ampere (A100)
    (4, 25), // NVLink 4.0: Hopper (H100)
    (5, 50), // NVLink 5.0: Blackwell (B100/B200) — per-link bidirectional
];

/// Look up the per-link bandwidth for a given NVLink generation.
///
/// Unknown versions fall back to [`FALLBACK_P2P_BANDWIDTH_GBPS`].
fn nvlink_gen_bandwidth(version: u32) -> u32 {
    NVLINK_GEN_BANDWIDTH_GBPS
        .iter()
        .find_map(|(v, bw)| if *v == version { Some(*bw) } else { None })
        .unwrap_or(FALLBACK_P2P_BANDWIDTH_GBPS)
}

/// NVLink / P2P topology for the GPUs visible to this process.
///
/// The adjacency matrix stores bidirectional bandwidth in GB/s, with `0`
/// meaning "no direct link". All matrices are square with `gpu_count` rows and
/// columns; the diagonal is always `0` (no self-link).
#[derive(Clone)]
pub struct NvlinkTopology {
    /// Number of GPUs represented in this topology.
    gpu_count: u32,
    /// Adjacency matrix: `direct_link[i][j]` = bandwidth (GB/s) or `0` when
    /// there is no direct link between GPU `i` and GPU `j`. Always symmetric.
    pub direct_link: Vec<Vec<u32>>,
    /// Shortest hop count between GPUs: `hop_count[i][j]` = number of direct
    /// links on the shortest path from `i` to `j`, or `u32::MAX` when `j` is
    /// unreachable from `i`. The diagonal is `0`.
    pub hop_count: Vec<Vec<u32>>,
}

impl NvlinkTopology {
    /// Build a topology from a bandwidth adjacency matrix.
    ///
    /// This is the low-level constructor — real callers should use
    /// [`Self::probe`]. The matrix must be square; if the caller passes a
    /// non-symmetric matrix we symmetrise it by taking the max of
    /// `matrix[i][j]` and `matrix[j][i]` (so either side claiming a link is
    /// enough). Self-loops on the diagonal are forced to `0`.
    pub fn from_adjacency(mut matrix: Vec<Vec<u32>>) -> Result<Self> {
        let n = matrix.len();
        for (i, row) in matrix.iter().enumerate() {
            if row.len() != n {
                return Err(RingKernelError::BackendError(format!(
                    "topology matrix is not square: row {} has {} cols, expected {}",
                    i,
                    row.len(),
                    n
                )));
            }
        }
        for i in 0..n {
            matrix[i][i] = 0;
            for j in (i + 1)..n {
                let max = matrix[i][j].max(matrix[j][i]);
                matrix[i][j] = max;
                matrix[j][i] = max;
            }
        }
        let hop_count = compute_hop_count(&matrix);
        Ok(Self {
            gpu_count: n as u32,
            direct_link: matrix,
            hop_count,
        })
    }

    /// Build a degenerate single-GPU topology — useful as a default / for CI.
    pub fn single_gpu() -> Self {
        Self {
            gpu_count: 1,
            direct_link: vec![vec![0]],
            hop_count: vec![vec![0]],
        }
    }

    /// Build a topology representing `n` disconnected GPUs (no NVLink between
    /// any pair). `hop_count[i][j]` is `u32::MAX` off-diagonal.
    pub fn disconnected(n: u32) -> Self {
        let n_usize = n as usize;
        let mut direct_link = vec![vec![0u32; n_usize]; n_usize];
        let hop_count = compute_hop_count(&direct_link);
        // ensure diagonal is zero
        for i in 0..n_usize {
            direct_link[i][i] = 0;
        }
        Self {
            gpu_count: n,
            direct_link,
            hop_count,
        }
    }

    /// Probe the local system for an NVLink / P2P topology.
    ///
    /// Returns a single-GPU topology when:
    /// - no CUDA-capable GPU is visible, or
    /// - CUDA runtime errors out (the spec demands we degrade gracefully).
    ///
    /// When the `multi-gpu` feature is enabled we consult NVML for real
    /// per-link bandwidth; otherwise we fall back to CUDA's P2P-only detection.
    pub fn probe() -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            probe_impl()
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self::single_gpu())
        }
    }

    /// Number of GPUs covered by this topology.
    pub fn gpu_count(&self) -> u32 {
        self.gpu_count
    }

    /// Direct-link bandwidth (GB/s) between two GPUs. Returns `0` when there
    /// is no direct link, or when either index is out of range.
    pub fn bandwidth(&self, from: u32, to: u32) -> u32 {
        if from >= self.gpu_count || to >= self.gpu_count {
            return 0;
        }
        self.direct_link[from as usize][to as usize]
    }

    /// Returns `true` when GPUs `from` and `to` share a direct NVLink / P2P
    /// link (bandwidth > 0). Self-checks (`from == to`) return `false` — a
    /// GPU is not "linked" to itself.
    pub fn direct_link_exists(&self, from: u32, to: u32) -> bool {
        if from == to {
            return false;
        }
        self.bandwidth(from, to) > 0
    }

    /// GPUs reachable from `gpu` over a *direct* link with at least
    /// `min_bandwidth_gbps` GB/s. Ordered by descending bandwidth then
    /// ascending GPU id, which makes placement decisions deterministic.
    ///
    /// Returns an empty vector when `gpu` is out of range.
    pub fn co_located_candidates(&self, gpu: u32, min_bandwidth_gbps: u32) -> Vec<u32> {
        if gpu >= self.gpu_count {
            return Vec::new();
        }
        let row = &self.direct_link[gpu as usize];
        let mut candidates: Vec<(u32, u32)> = row
            .iter()
            .enumerate()
            .filter_map(|(j, bw)| {
                let j = j as u32;
                if j != gpu && *bw >= min_bandwidth_gbps && *bw > 0 {
                    Some((j, *bw))
                } else {
                    None
                }
            })
            .collect();
        candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        candidates.into_iter().map(|(id, _)| id).collect()
    }

    /// Shortest (fewest-hop) path of GPU indices from `from` to `to`,
    /// inclusive. Returns `None` when the two GPUs are in different connected
    /// components, or when either index is out of range.
    ///
    /// A trivial path `from == to` returns `Some(vec![from])`.
    pub fn shortest_path(&self, from: u32, to: u32) -> Option<Vec<u32>> {
        if from >= self.gpu_count || to >= self.gpu_count {
            return None;
        }
        if from == to {
            return Some(vec![from]);
        }
        let n = self.gpu_count as usize;
        let mut parent = vec![u32::MAX; n];
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();
        queue.push_back(from as usize);
        visited[from as usize] = true;
        while let Some(u) = queue.pop_front() {
            if u == to as usize {
                // reconstruct
                let mut path = vec![u as u32];
                let mut cur = u as u32;
                while cur != from {
                    cur = parent[cur as usize];
                    path.push(cur);
                }
                path.reverse();
                return Some(path);
            }
            for v in 0..n {
                if !visited[v] && self.direct_link[u][v] > 0 {
                    visited[v] = true;
                    parent[v] = u as u32;
                    queue.push_back(v);
                }
            }
        }
        None
    }

    /// Hop count between two GPUs, or `None` when unreachable or out of range.
    pub fn hops(&self, from: u32, to: u32) -> Option<u32> {
        if from >= self.gpu_count || to >= self.gpu_count {
            return None;
        }
        let h = self.hop_count[from as usize][to as usize];
        if h == u32::MAX {
            None
        } else {
            Some(h)
        }
    }
}

impl fmt::Debug for NvlinkTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "NvlinkTopology {{ gpu_count: {}", self.gpu_count)?;
        writeln!(f, "  direct_link (GB/s):")?;
        write!(f, "         ")?;
        for j in 0..self.gpu_count {
            write!(f, " gpu{:<3}", j)?;
        }
        writeln!(f)?;
        for i in 0..self.gpu_count {
            write!(f, "    gpu{:<3}", i)?;
            for j in 0..self.gpu_count {
                let bw = self.direct_link[i as usize][j as usize];
                if bw == 0 {
                    write!(f, "     -")?;
                } else {
                    write!(f, "  {:>4}", bw)?;
                }
            }
            writeln!(f)?;
        }
        writeln!(f, "  hop_count:")?;
        write!(f, "         ")?;
        for j in 0..self.gpu_count {
            write!(f, " gpu{:<3}", j)?;
        }
        writeln!(f)?;
        for i in 0..self.gpu_count {
            write!(f, "    gpu{:<3}", i)?;
            for j in 0..self.gpu_count {
                let h = self.hop_count[i as usize][j as usize];
                if h == u32::MAX {
                    write!(f, "     ~")?; // unreachable
                } else {
                    write!(f, "  {:>4}", h)?;
                }
            }
            writeln!(f)?;
        }
        write!(f, "}}")
    }
}

/// BFS over the unweighted graph implied by `direct_link` to produce the
/// hop-count matrix. Any entry > 0 counts as one hop; `0` means no link.
fn compute_hop_count(direct_link: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let n = direct_link.len();
    let mut hops = vec![vec![u32::MAX; n]; n];
    for (src, row) in hops.iter_mut().enumerate().take(n) {
        row[src] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(src);
        while let Some(u) = queue.pop_front() {
            let dist_u = row[u];
            for v in 0..n {
                if u != v && direct_link[u][v] > 0 && row[v] == u32::MAX {
                    row[v] = dist_u + 1;
                    queue.push_back(v);
                }
            }
        }
    }
    hops
}

// ---------- probe implementation ----------

#[cfg(feature = "cuda")]
fn probe_impl() -> Result<NvlinkTopology> {
    // Step 1: ask CUDA how many devices are visible. We guard against cudarc
    // panicking when the driver isn't installed (same pattern used by
    // `crate::is_cuda_available`).
    let device_count =
        match std::panic::catch_unwind(|| cudarc::driver::CudaContext::device_count().unwrap_or(0))
        {
            Ok(count) => count,
            Err(_) => {
                tracing::debug!("CUDA driver unavailable — returning single-GPU topology");
                return Ok(NvlinkTopology::single_gpu());
            }
        };

    if device_count == 0 {
        tracing::debug!("no CUDA devices visible — returning single-GPU topology");
        return Ok(NvlinkTopology::single_gpu());
    }
    let n = device_count as usize;

    // Step 2: P2P-capable adjacency from the CUDA driver. This is the
    // always-available fallback — it tells us which pairs *can* talk P2P but
    // not whether that link is NVLink vs PCIe.
    //
    // `mut` is load-bearing when the `multi-gpu` feature upgrades this matrix
    // below; keep it unconditionally so the code compiles cleanly either way.
    #[cfg_attr(not(feature = "multi-gpu"), allow(unused_mut))]
    let mut matrix = p2p_matrix_from_cuda(n)?;

    // Step 3: if NVML is available, upgrade any NVLink-connected pair with its
    // real per-link bandwidth. This is gated behind the `multi-gpu` feature
    // because nvml-wrapper pulls in libnvidia-ml at link time.
    #[cfg(feature = "multi-gpu")]
    {
        match enrich_with_nvml(n, &mut matrix) {
            Ok(()) => tracing::debug!("NVLink topology enriched via NVML"),
            Err(e) => tracing::warn!("NVML probe failed ({e}); using CUDA P2P fallback bandwidths"),
        }
    }

    NvlinkTopology::from_adjacency(matrix)
}

/// Build an N×N P2P-capability matrix from the CUDA driver API.
/// Entries are set to [`FALLBACK_P2P_BANDWIDTH_GBPS`] for any pair that
/// reports `cuDeviceCanAccessPeer = 1` (and is non-self).
#[cfg(feature = "cuda")]
fn p2p_matrix_from_cuda(n: usize) -> Result<Vec<Vec<u32>>> {
    use cudarc::driver::sys as cuda_sys;

    let mut matrix = vec![vec![0u32; n]; n];

    // Resolve each ordinal to a CUdevice handle up front.
    let mut devices: Vec<cuda_sys::CUdevice> = Vec::with_capacity(n);
    for ordinal in 0..n {
        let mut dev: cuda_sys::CUdevice = 0;
        // SAFETY: `cuDeviceGet` writes a single `CUdevice` into `dev` on
        // success; on failure we propagate and discard the uninitialised
        // value.
        let rc = unsafe { cuda_sys::cuDeviceGet(&mut dev, ordinal as i32) };
        if rc != cuda_sys::cudaError_enum::CUDA_SUCCESS {
            return Err(RingKernelError::BackendError(format!(
                "cuDeviceGet({ordinal}) failed: {rc:?}"
            )));
        }
        devices.push(dev);
    }

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut can_access: i32 = 0;
            // SAFETY: both device handles come from `cuDeviceGet` above and
            // are valid for the lifetime of this function. `can_access` is a
            // stack-allocated `i32` we own.
            let rc =
                unsafe { cuda_sys::cuDeviceCanAccessPeer(&mut can_access, devices[i], devices[j]) };
            if rc == cuda_sys::cudaError_enum::CUDA_SUCCESS && can_access == 1 {
                matrix[i][j] = FALLBACK_P2P_BANDWIDTH_GBPS;
            }
        }
    }
    Ok(matrix)
}

/// Walk every device's NvLinks via NVML and upgrade the fallback matrix to
/// reflect real per-link bandwidth (scaled by active link count).
#[cfg(all(feature = "cuda", feature = "multi-gpu"))]
fn enrich_with_nvml(n: usize, matrix: &mut [Vec<u32>]) -> Result<()> {
    use nvml_wrapper::Nvml;

    let nvml = Nvml::init()
        .map_err(|e| RingKernelError::BackendError(format!("Nvml::init failed: {e}")))?;

    // Capture each device's PCI bus id (domain:bus:device) — we'll match
    // against remote NvLink endpoints by PCI id.
    struct Slot {
        ordinal: u32,
        domain: u32,
        bus: u32,
        device: u32,
    }

    let mut slots: Vec<Slot> = Vec::with_capacity(n);
    for ordinal in 0..n {
        let nvml_dev = nvml.device_by_index(ordinal as u32).map_err(|e| {
            RingKernelError::BackendError(format!(
                "nvmlDeviceGetHandleByIndex({ordinal}) failed: {e}"
            ))
        })?;
        let pci = nvml_dev.pci_info().map_err(|e| {
            RingKernelError::BackendError(format!("nvmlDeviceGetPciInfo_v3({ordinal}) failed: {e}"))
        })?;
        slots.push(Slot {
            ordinal: ordinal as u32,
            domain: pci.domain,
            bus: pci.bus,
            device: pci.device,
        });
    }

    // Per pair we accumulate (active_links, max_version).
    let mut link_agg: Vec<Vec<(u32, u32)>> = vec![vec![(0u32, 0u32); n]; n];

    for slot in &slots {
        let device = match nvml.device_by_index(slot.ordinal) {
            Ok(d) => d,
            Err(_) => continue,
        };
        // NVIDIA doesn't publish a "how many NvLinks does this GPU have"
        // call — the convention is to probe 0..18 and stop on error. Hopper
        // has up to 18 links; we cap the loop accordingly.
        for link in 0..18u32 {
            let link_handle = device.link_wrapper_for(link);
            let active = match link_handle.is_active() {
                Ok(flag) => flag,
                Err(_) => break,
            };
            if !active {
                continue;
            }
            let version = link_handle.version().unwrap_or(0);
            let remote = match link_handle.remote_pci_info() {
                Ok(info) => info,
                Err(_) => continue,
            };
            if let Some(peer) = slots.iter().find(|s| {
                s.domain == remote.domain && s.bus == remote.bus && s.device == remote.device
            }) {
                let i = slot.ordinal as usize;
                let j = peer.ordinal as usize;
                if i == j {
                    continue;
                }
                let entry = &mut link_agg[i][j];
                entry.0 += 1;
                if version > entry.1 {
                    entry.1 = version;
                }
            }
        }
    }

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let (links, version) = link_agg[i][j];
            if links > 0 {
                let bw = links * nvlink_gen_bandwidth(version);
                matrix[i][j] = bw;
            }
        }
    }
    Ok(())
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert the matrix is symmetric.
    fn assert_symmetric(topo: &NvlinkTopology) {
        for i in 0..topo.gpu_count() {
            for j in 0..topo.gpu_count() {
                assert_eq!(
                    topo.bandwidth(i, j),
                    topo.bandwidth(j, i),
                    "matrix not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn single_gpu_topology_has_no_links() {
        let t = NvlinkTopology::single_gpu();
        assert_eq!(t.gpu_count(), 1);
        assert_eq!(t.bandwidth(0, 0), 0);
        assert!(!t.direct_link_exists(0, 0));
        assert_eq!(t.co_located_candidates(0, 1), Vec::<u32>::new());
        assert_eq!(t.shortest_path(0, 0), Some(vec![0]));
        assert_eq!(t.hops(0, 0), Some(0));
    }

    #[test]
    fn two_gpu_linear_topology_is_symmetric() {
        // gpu0 <-> gpu1 @ 25 GB/s
        let m = vec![vec![0, 25], vec![25, 0]];
        let t = NvlinkTopology::from_adjacency(m).expect("valid");
        assert_symmetric(&t);
        assert_eq!(t.gpu_count(), 2);
        assert!(t.direct_link_exists(0, 1));
        assert!(t.direct_link_exists(1, 0));
        assert_eq!(t.bandwidth(0, 1), 25);
        assert_eq!(t.hops(0, 1), Some(1));
        assert_eq!(t.shortest_path(0, 1), Some(vec![0, 1]));
        assert_eq!(t.co_located_candidates(0, 20), vec![1]);
        // min-bandwidth filter excludes below-threshold links
        assert_eq!(t.co_located_candidates(0, 50), Vec::<u32>::new());
    }

    #[test]
    fn four_gpu_ring_has_two_hop_diameter() {
        // 0 — 1
        // |   |
        // 3 — 2
        let m = vec![
            vec![0, 100, 0, 100],
            vec![100, 0, 100, 0],
            vec![0, 100, 0, 100],
            vec![100, 0, 100, 0],
        ];
        let t = NvlinkTopology::from_adjacency(m).expect("valid");
        assert_symmetric(&t);
        assert_eq!(t.hops(0, 2), Some(2));
        assert_eq!(t.hops(1, 3), Some(2));
        assert_eq!(t.hops(0, 1), Some(1));
        // direct neighbours are the co-located candidates
        let mut neighbours = t.co_located_candidates(0, 1);
        neighbours.sort();
        assert_eq!(neighbours, vec![1, 3]);
        // shortest path across the ring is exactly 3 nodes long
        let path = t.shortest_path(0, 2).expect("reachable");
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], 0);
        assert_eq!(path[2], 2);
    }

    #[test]
    fn eight_gpu_hypercube_has_log_n_diameter() {
        // 3-dimensional hypercube: vertices labelled 0..8; connect vertices
        // whose bitwise XOR is a single bit (hamming distance 1).
        let n: usize = 8;
        let mut m = vec![vec![0u32; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let x = i ^ j;
                    if x.count_ones() == 1 {
                        m[i][j] = 50;
                    }
                }
            }
        }
        let t = NvlinkTopology::from_adjacency(m).expect("valid");
        assert_symmetric(&t);
        // every vertex has exactly 3 neighbours
        for g in 0..8u32 {
            let neigh = t.co_located_candidates(g, 1);
            assert_eq!(neigh.len(), 3, "gpu {g} should have 3 direct links");
        }
        // diameter of a 3-cube is 3 (opposite corners)
        assert_eq!(t.hops(0, 7), Some(3));
        let path = t.shortest_path(0, 7).expect("reachable");
        assert_eq!(path.len(), 4);
    }

    #[test]
    fn disconnected_pairs_report_unreachable() {
        // Two 2-GPU clusters that cannot see each other:
        //   cluster A: 0 <-> 1
        //   cluster B: 2 <-> 3
        let m = vec![
            vec![0, 100, 0, 0],
            vec![100, 0, 0, 0],
            vec![0, 0, 0, 100],
            vec![0, 0, 100, 0],
        ];
        let t = NvlinkTopology::from_adjacency(m).expect("valid");
        assert!(t.direct_link_exists(0, 1));
        assert!(t.direct_link_exists(2, 3));
        assert!(!t.direct_link_exists(0, 2));
        assert_eq!(t.hops(0, 2), None);
        assert_eq!(t.hops(1, 3), None);
        assert_eq!(t.shortest_path(0, 3), None);
        assert_eq!(t.co_located_candidates(0, 1), vec![1]);
        assert_eq!(t.co_located_candidates(2, 1), vec![3]);
    }

    #[test]
    fn bandwidth_queries_handle_out_of_range() {
        let t = NvlinkTopology::single_gpu();
        assert_eq!(t.bandwidth(0, 99), 0);
        assert_eq!(t.bandwidth(99, 0), 0);
        assert!(!t.direct_link_exists(0, 99));
        assert_eq!(t.co_located_candidates(99, 1), Vec::<u32>::new());
        assert_eq!(t.shortest_path(0, 99), None);
        assert_eq!(t.hops(99, 0), None);
    }

    #[test]
    fn from_adjacency_rejects_non_square() {
        // 3 rows, mixed column counts
        let m = vec![vec![0, 0, 0], vec![0, 0], vec![0, 0, 0]];
        let err = NvlinkTopology::from_adjacency(m).expect_err("must reject");
        match err {
            RingKernelError::BackendError(msg) => assert!(msg.contains("not square")),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn from_adjacency_symmetrises_input() {
        // Upper-triangular input: take max(i,j) and (j,i) — the matrix should
        // become symmetric after construction.
        let m = vec![vec![0, 30, 0], vec![10, 0, 40], vec![0, 0, 0]];
        let t = NvlinkTopology::from_adjacency(m).expect("valid");
        assert_symmetric(&t);
        // max(30, 10) = 30
        assert_eq!(t.bandwidth(0, 1), 30);
        assert_eq!(t.bandwidth(1, 0), 30);
        // max(40, 0) = 40
        assert_eq!(t.bandwidth(1, 2), 40);
        assert_eq!(t.bandwidth(2, 1), 40);
    }

    #[test]
    fn from_adjacency_zeroes_diagonal() {
        // Caller accidentally set a self-loop — we should drop it.
        let m = vec![vec![99, 25], vec![25, 77]];
        let t = NvlinkTopology::from_adjacency(m).expect("valid");
        assert_eq!(t.bandwidth(0, 0), 0);
        assert_eq!(t.bandwidth(1, 1), 0);
        assert_eq!(t.bandwidth(0, 1), 25);
    }

    #[test]
    fn co_located_candidates_ordered_by_bandwidth_desc() {
        // gpu 0 has three neighbours with different bandwidths — the highest
        // must come first.
        let m = vec![
            vec![0, 10, 50, 30],
            vec![10, 0, 0, 0],
            vec![50, 0, 0, 0],
            vec![30, 0, 0, 0],
        ];
        let t = NvlinkTopology::from_adjacency(m).expect("valid");
        let ranked = t.co_located_candidates(0, 1);
        assert_eq!(ranked, vec![2, 3, 1]);
        // min_bandwidth of 25 keeps only gpus 2 and 3
        let filtered = t.co_located_candidates(0, 25);
        assert_eq!(filtered, vec![2, 3]);
        // min_bandwidth above any value returns nothing
        assert_eq!(t.co_located_candidates(0, 100), Vec::<u32>::new());
    }

    #[test]
    fn shortest_path_multi_hop_is_optimal() {
        // Linear chain 0-1-2-3-4 — shortest path from 0 to 4 must traverse
        // every node, with hop count 4.
        let mut m = vec![vec![0u32; 5]; 5];
        for i in 0..4 {
            m[i][i + 1] = 40;
            m[i + 1][i] = 40;
        }
        let t = NvlinkTopology::from_adjacency(m).expect("valid");
        assert_eq!(t.hops(0, 4), Some(4));
        assert_eq!(t.shortest_path(0, 4), Some(vec![0, 1, 2, 3, 4]));
        // Middle-out check: hops(1, 3) must be 2
        assert_eq!(t.hops(1, 3), Some(2));
    }

    #[test]
    fn disconnected_constructor_produces_unreachable_graph() {
        let t = NvlinkTopology::disconnected(3);
        assert_eq!(t.gpu_count(), 3);
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(t.hops(i, j), Some(0));
                } else {
                    assert_eq!(t.hops(i, j), None);
                    assert_eq!(t.bandwidth(i, j), 0);
                }
            }
        }
    }

    #[test]
    fn debug_output_renders_matrix() {
        let m = vec![vec![0, 25], vec![25, 0]];
        let t = NvlinkTopology::from_adjacency(m).expect("valid");
        let s = format!("{t:?}");
        assert!(s.contains("gpu_count: 2"));
        assert!(s.contains("25"), "bandwidth must appear in debug: {s}");
        assert!(s.contains("hop_count"));
    }

    #[test]
    fn nvlink_gen_bandwidth_known_generations() {
        assert_eq!(nvlink_gen_bandwidth(1), 20);
        assert_eq!(nvlink_gen_bandwidth(2), 25);
        assert_eq!(nvlink_gen_bandwidth(3), 25);
        assert_eq!(nvlink_gen_bandwidth(4), 25);
        assert_eq!(nvlink_gen_bandwidth(5), 50);
        // unknown -> fallback
        assert_eq!(nvlink_gen_bandwidth(99), FALLBACK_P2P_BANDWIDTH_GBPS);
    }

    /// Smoke test against whatever hardware is present. Ignored by default
    /// because CI GPUs may not have NVLink (or even more than one GPU).
    #[test]
    #[ignore = "requires CUDA hardware"]
    fn probe_real_hardware() {
        let topo = NvlinkTopology::probe().expect("probe must not fail");
        // probe() must always return *at least* one GPU
        assert!(topo.gpu_count() >= 1);
        // diagonal always zero
        for i in 0..topo.gpu_count() {
            assert_eq!(topo.bandwidth(i, i), 0);
        }
        // matrix must be symmetric
        for i in 0..topo.gpu_count() {
            for j in 0..topo.gpu_count() {
                assert_eq!(topo.bandwidth(i, j), topo.bandwidth(j, i));
            }
        }
        eprintln!("{topo:?}");
    }
}
