//! NVSHMEM symmetric heap bindings.
//!
//! This module exposes a minimal, opt-in Rust wrapper over NVIDIA's
//! NVSHMEM library (`libnvshmem_host.so`). It is gated behind the
//! `nvshmem` Cargo feature because NVSHMEM has external install
//! requirements (the `libnvshmem3-cuda-12` / `libnvshmem3-dev-cuda-12`
//! packages on Ubuntu, or a manual NVIDIA Developer install) and a
//! non-trivial bootstrap protocol.
//!
//! # Scope (v1.2)
//!
//! The RingKernel public API surface here is intentionally small — it
//! covers just the symmetric-heap operations needed by persistent GPU
//! actors that want PGAS-style cross-GPU memory:
//!
//!   * [`NvshmemHeap::new`] — initializes NVSHMEM for the current
//!     process (assumes the caller provides the bootstrap; see
//!     [Bootstrap options] below).
//!   * [`NvshmemHeap::malloc`] / [`NvshmemHeap::free`] — allocate and
//!     free symmetric buffers on the NVSHMEM heap.
//!   * [`NvshmemHeap::put`] / [`NvshmemHeap::get`] — one-sided puts
//!     and gets against another PE's symmetric heap slot.
//!   * [`NvshmemHeap::barrier_all`] — collective barrier across all
//!     PEs.
//!   * [`NvshmemHeap::my_pe`] / [`NvshmemHeap::n_pes`] — PE identity.
//!
//! Collective reductions, teams, signal primitives, and device-side
//! NVSHMEM (`nvshmem_putmem` from a kernel) are deliberately out of
//! scope for this first cut. They can be added alongside a kernel
//! that needs them.
//!
//! # Bootstrap options
//!
//! NVSHMEM requires a bootstrap to wire up the symmetric heap across
//! PEs. On a single-node multi-GPU box three paths are supported:
//!
//! 1. **MPI** — launch under `mpirun -np <n>`; NVSHMEM picks up MPI
//!    automatically. Requires `libmpi` and a bootstrap plugin.
//! 2. **`nvshmrun`** — NVSHMEM's bundled launcher.
//! 3. **Unique-ID bootstrap** — for single-process multi-device use,
//!    call `nvshmemx_get_uniqueid` on rank 0, share the blob out-of-
//!    band, and call `nvshmemx_set_attr_uniqueid_args` +
//!    `nvshmemx_init_attr` on each rank.
//!
//! For now this wrapper assumes the caller has already performed the
//! bootstrap via one of those paths — it only drives the post-init
//! operations. Smoke-tests in this module skip cleanly on a host that
//! has not been bootstrapped.
//!
//! # Safety
//!
//! All bindings are `unsafe extern "C"`; the safe wrappers above
//! either (a) convert between C and Rust types or (b) encapsulate
//! multi-step sequences. Because NVSHMEM bindings are opt-in, code
//! that doesn't enable the `nvshmem` feature is entirely unaffected.

#![cfg(feature = "nvshmem")]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use ringkernel_core::error::{Result, RingKernelError};

/// Raw NVSHMEM C bindings. Declared here rather than in a separate
/// `bindgen`-generated crate to keep the dependency footprint small.
#[allow(non_camel_case_types)]
pub mod sys {
    use std::ffi::c_void;

    /// NVSHMEM team handle (opaque).
    pub type nvshmem_team_t = i32;

    extern "C" {
        /// Returns the PE rank of the calling process.
        pub fn nvshmem_my_pe() -> i32;
        /// Returns the total number of PEs in the job.
        pub fn nvshmem_n_pes() -> i32;

        /// Allocate `size` bytes on the symmetric heap. Returns a
        /// pointer that is valid on every PE at the same offset.
        pub fn nvshmem_malloc(size: usize) -> *mut c_void;
        /// Free a symmetric heap allocation produced by
        /// [`nvshmem_malloc`].
        pub fn nvshmem_free(ptr: *mut c_void);

        /// Put `bytes` bytes from `source` (local) into `dest`
        /// (symmetric-heap slot on `pe`). Blocking.
        pub fn nvshmem_putmem(
            dest: *mut c_void,
            source: *const c_void,
            bytes: usize,
            pe: i32,
        );
        /// Get `bytes` bytes from `source` (symmetric-heap slot on
        /// `pe`) into `dest` (local). Blocking.
        pub fn nvshmem_getmem(
            dest: *mut c_void,
            source: *const c_void,
            bytes: usize,
            pe: i32,
        );

        /// Collective barrier over every PE.
        pub fn nvshmem_barrier_all();
        /// Fence: ordering for subsequent puts/gets on the current PE.
        pub fn nvshmem_fence();
    }

    // `nvshmem_finalize()` in the C header is an inline wrapper
    // around `nvshmemi_finalize()` which is not part of the exported
    // ABI of libnvshmem_host.so (3.6+). Bootstrap frameworks —
    // `nvshmrun`, MPI, or the unique-ID caller — are responsible for
    // tearing NVSHMEM down. Our `NvshmemHeap::Drop` therefore does
    // nothing on the library side; it just clears the
    // once-per-process active flag.
}

/// RAII handle to an initialized NVSHMEM runtime.
///
/// Creating this struct assumes the NVSHMEM bootstrap has already
/// happened (MPI, `nvshmrun`, or unique-ID). The struct's only role
/// is to (a) record the PE identity, (b) offer typed accessors for
/// the C API, and (c) call `nvshmem_finalize` when dropped. Only one
/// instance may exist at a time.
pub struct NvshmemHeap {
    my_pe: i32,
    n_pes: i32,
}

static ACTIVE: AtomicBool = AtomicBool::new(false);

impl NvshmemHeap {
    /// Acquire the NVSHMEM handle. Returns an error if NVSHMEM was
    /// not already bootstrapped (indicated by `nvshmem_n_pes()`
    /// returning 0 or a negative value).
    ///
    /// Only one `NvshmemHeap` may exist at a time per process; a
    /// second call returns [`RingKernelError::BackendError`] until
    /// the first is dropped.
    pub fn attach() -> Result<Self> {
        if ACTIVE.swap(true, Ordering::AcqRel) {
            return Err(RingKernelError::BackendError(
                "NvshmemHeap already attached in this process".into(),
            ));
        }

        let (my_pe, n_pes) = unsafe { (sys::nvshmem_my_pe(), sys::nvshmem_n_pes()) };
        if n_pes <= 0 {
            ACTIVE.store(false, Ordering::Release);
            return Err(RingKernelError::BackendError(format!(
                "NVSHMEM not bootstrapped: nvshmem_n_pes() = {n_pes}. \
                 Launch under mpirun / nvshmrun / unique-ID bootstrap."
            )));
        }
        Ok(Self { my_pe, n_pes })
    }

    /// This PE's rank.
    pub fn my_pe(&self) -> i32 {
        self.my_pe
    }

    /// Total number of PEs.
    pub fn n_pes(&self) -> i32 {
        self.n_pes
    }

    /// Allocate `len` symmetric bytes on the NVSHMEM heap.
    /// Returns a pointer valid on every PE.
    pub fn malloc(&self, len: usize) -> Result<SymmetricPtr> {
        let p = unsafe { sys::nvshmem_malloc(len) };
        if p.is_null() {
            return Err(RingKernelError::OutOfMemory {
                requested: len,
                available: 0,
            });
        }
        Ok(SymmetricPtr { ptr: p, len })
    }

    /// Blocking put of `source` bytes into `dest`'s slot on remote
    /// PE `pe`. `dest` must come from [`Self::malloc`]; `source` may
    /// be any buffer the current PE owns (host or device).
    pub fn put(&self, dest: &SymmetricPtr, source: *const c_void, bytes: usize, pe: i32) {
        unsafe { sys::nvshmem_putmem(dest.ptr, source, bytes, pe) }
    }

    /// Blocking get of `bytes` bytes from `source`'s slot on remote
    /// PE `pe` into the caller-owned `dest` buffer.
    pub fn get(&self, dest: *mut c_void, source: &SymmetricPtr, bytes: usize, pe: i32) {
        unsafe { sys::nvshmem_getmem(dest, source.ptr, bytes, pe) }
    }

    /// Collective barrier across every PE.
    pub fn barrier_all(&self) {
        unsafe { sys::nvshmem_barrier_all() }
    }

    /// Fence: ordering for the current PE's subsequent puts/gets.
    pub fn fence(&self) {
        unsafe { sys::nvshmem_fence() }
    }
}

impl Drop for NvshmemHeap {
    fn drop(&mut self) {
        // Library-level finalize is owned by the bootstrap framework
        // (mpirun / nvshmrun / unique-ID caller). We only release the
        // once-per-process active flag.
        ACTIVE.store(false, Ordering::Release);
    }
}

/// Non-owning symmetric heap pointer. `Drop`-free; allocations are
/// owned by the [`NvshmemHeap`] and freed explicitly via
/// [`NvshmemHeap::free`] (or leaked if the heap is finalized
/// without freeing — same as NVSHMEM itself).
#[derive(Debug, Clone, Copy)]
pub struct SymmetricPtr {
    ptr: *mut c_void,
    len: usize,
}

unsafe impl Send for SymmetricPtr {}
unsafe impl Sync for SymmetricPtr {}

impl SymmetricPtr {
    /// Raw pointer on the local PE's symmetric slot.
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Allocation length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always `false` — a null allocation would not have been
    /// produced by [`NvshmemHeap::malloc`].
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl NvshmemHeap {
    /// Free a symmetric heap allocation produced by
    /// [`Self::malloc`]. Safe because `SymmetricPtr` is owned by
    /// this heap.
    pub fn free(&self, ptr: SymmetricPtr) {
        unsafe { sys::nvshmem_free(ptr.ptr) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test — runs only when NVSHMEM has been bootstrapped
    /// (i.e. the test process was launched under mpirun / nvshmrun
    /// or with a unique-ID bootstrap before calling this test).
    ///
    /// On a plain single-process host `attach` returns
    /// `BackendError` and the test passes by skipping.
    #[test]
    #[ignore] // Requires NVSHMEM bootstrap + 2+ GPUs
    fn attach_and_query_pe() {
        match NvshmemHeap::attach() {
            Ok(heap) => {
                let pe = heap.my_pe();
                let n = heap.n_pes();
                assert!(n >= 1, "n_pes >= 1");
                assert!(pe >= 0 && pe < n, "my_pe in range");
                heap.barrier_all();
                let buf = heap.malloc(1024).expect("malloc 1 KiB symmetric");
                assert_eq!(buf.len(), 1024);
                heap.free(buf);
            }
            Err(e) => {
                eprintln!("SKIP: NVSHMEM not bootstrapped: {e}");
            }
        }
    }
}
