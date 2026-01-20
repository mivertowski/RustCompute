//! CUDA event wrappers for GPU-side timing.
//!
//! This module provides safe wrappers around CUDA events for precise GPU timing
//! without CPU overhead. Events are recorded on GPU streams and timing is
//! measured entirely on the GPU.
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_cuda::profiling::{CudaEvent, GpuTimer};
//!
//! let mut timer = GpuTimer::new()?;
//! timer.start(stream)?;
//! // ... kernel execution ...
//! timer.stop(stream)?;
//! timer.synchronize()?;
//! println!("Elapsed: {:.3} ms", timer.elapsed_ms()?);
//! ```

use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

use cudarc::driver::result as cuda_result;
use cudarc::driver::sys as cuda_sys;

use ringkernel_core::error::{Result, RingKernelError};

/// Flags for CUDA event creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaEventFlags {
    /// Default event creation (includes timing).
    Default,
    /// Event with blocking synchronization (reduces CPU spin).
    BlockingSync,
    /// Event without timing support (faster, less overhead).
    DisableTiming,
    /// Event that can be used for interprocess communication.
    Interprocess,
}

impl CudaEventFlags {
    /// Convert to CUDA event flags.
    fn to_cuda_flags(self) -> cuda_sys::CUevent_flags {
        match self {
            CudaEventFlags::Default => cuda_sys::CUevent_flags::CU_EVENT_DEFAULT,
            CudaEventFlags::BlockingSync => cuda_sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC,
            CudaEventFlags::DisableTiming => cuda_sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
            CudaEventFlags::Interprocess => cuda_sys::CUevent_flags::CU_EVENT_INTERPROCESS,
        }
    }
}

/// A CUDA event for GPU-side synchronization and timing.
///
/// Events can be recorded on a stream and later synchronized or queried for
/// timing information. They enable precise GPU-side timing without CPU overhead.
pub struct CudaEvent {
    event: cuda_sys::CUevent,
    timing_enabled: bool,
}

impl CudaEvent {
    /// Create a new CUDA event with default flags (timing enabled).
    pub fn new() -> Result<Self> {
        Self::with_flags(CudaEventFlags::Default)
    }

    /// Create a new CUDA event with specific flags.
    pub fn with_flags(flags: CudaEventFlags) -> Result<Self> {
        let timing_enabled = flags != CudaEventFlags::DisableTiming;
        let event = cuda_result::event::create(flags.to_cuda_flags()).map_err(|e| {
            RingKernelError::BackendError(format!("Failed to create CUDA event: {:?}", e))
        })?;

        Ok(Self {
            event,
            timing_enabled,
        })
    }

    /// Record this event on a stream.
    ///
    /// The event captures the current state of the stream. All work submitted
    /// to the stream before this call will complete before the event.
    ///
    /// # Safety
    ///
    /// The stream must be valid and belong to the current CUDA context.
    pub unsafe fn record(&self, stream: cuda_sys::CUstream) -> Result<()> {
        cuda_result::event::record(self.event, stream).map_err(|e| {
            RingKernelError::BackendError(format!("Failed to record CUDA event: {:?}", e))
        })
    }

    /// Record this event on the default (null) stream.
    ///
    /// # Safety
    ///
    /// A CUDA context must be active on the current thread.
    pub unsafe fn record_default(&self) -> Result<()> {
        self.record(ptr::null_mut())
    }

    /// Synchronize (wait) for this event to complete.
    ///
    /// Blocks the calling thread until all work captured by this event is complete.
    ///
    /// # Safety
    ///
    /// The event must have been recorded on a stream.
    pub unsafe fn synchronize(&self) -> Result<()> {
        cuda_result::event::synchronize(self.event).map_err(|e| {
            RingKernelError::BackendError(format!("Failed to synchronize CUDA event: {:?}", e))
        })
    }

    /// Query whether this event has completed (non-blocking).
    ///
    /// Returns `true` if all work captured by this event is complete,
    /// `false` if work is still in progress.
    ///
    /// # Safety
    ///
    /// The event must have been recorded on a stream.
    pub unsafe fn query(&self) -> Result<bool> {
        match cuda_result::event::query(self.event) {
            Ok(()) => Ok(true),
            Err(e) if e.0 == cuda_sys::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            Err(e) => Err(RingKernelError::BackendError(format!(
                "Failed to query CUDA event: {:?}",
                e
            ))),
        }
    }

    /// Check if timing is enabled for this event.
    pub fn timing_enabled(&self) -> bool {
        self.timing_enabled
    }

    /// Get the raw CUDA event handle.
    ///
    /// # Safety
    ///
    /// The caller must ensure the event is not destroyed while the handle is in use.
    pub unsafe fn raw(&self) -> cuda_sys::CUevent {
        self.event
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        // Safety: We own this event and it's being destroyed
        unsafe {
            let _ = cuda_result::event::destroy(self.event);
        }
    }
}

// CudaEvent is Send+Sync because CUDA events can be used from any thread
// once created (within the same context)
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

/// A GPU timer using CUDA events for precise timing.
///
/// This timer uses two CUDA events (start and stop) to measure GPU execution
/// time. The timing is performed entirely on the GPU, providing accurate
/// measurements without CPU overhead.
///
/// # Example
///
/// ```ignore
/// let mut timer = GpuTimer::new()?;
/// unsafe {
///     timer.start(stream)?;
///     // ... kernel launch ...
///     timer.stop(stream)?;
///     timer.synchronize()?;
/// }
/// println!("Elapsed: {:.3} ms", timer.elapsed_ms()?);
/// ```
pub struct GpuTimer {
    start: CudaEvent,
    stop: CudaEvent,
    started: bool,
    stopped: bool,
}

impl GpuTimer {
    /// Create a new GPU timer.
    pub fn new() -> Result<Self> {
        Ok(Self {
            start: CudaEvent::new()?,
            stop: CudaEvent::new()?,
            started: false,
            stopped: false,
        })
    }

    /// Start the timer by recording the start event.
    ///
    /// # Safety
    ///
    /// The stream must be valid and belong to the current CUDA context.
    pub unsafe fn start(&mut self, stream: cuda_sys::CUstream) -> Result<()> {
        self.start.record(stream)?;
        self.started = true;
        self.stopped = false;
        Ok(())
    }

    /// Start the timer on the default (null) stream.
    ///
    /// # Safety
    ///
    /// A CUDA context must be active on the current thread.
    pub unsafe fn start_default(&mut self) -> Result<()> {
        self.start(ptr::null_mut())
    }

    /// Stop the timer by recording the stop event.
    ///
    /// # Safety
    ///
    /// The stream must be valid and belong to the current CUDA context.
    /// `start()` must have been called before this.
    pub unsafe fn stop(&mut self, stream: cuda_sys::CUstream) -> Result<()> {
        if !self.started {
            return Err(RingKernelError::InvalidState {
                expected: "started".to_string(),
                actual: "not started".to_string(),
            });
        }
        self.stop.record(stream)?;
        self.stopped = true;
        Ok(())
    }

    /// Stop the timer on the default (null) stream.
    ///
    /// # Safety
    ///
    /// A CUDA context must be active on the current thread.
    /// `start()` must have been called before this.
    pub unsafe fn stop_default(&mut self) -> Result<()> {
        self.stop(ptr::null_mut())
    }

    /// Synchronize (wait) for the timer to complete.
    ///
    /// Blocks until the stop event has completed.
    ///
    /// # Safety
    ///
    /// `start()` and `stop()` must have been called before this.
    pub unsafe fn synchronize(&self) -> Result<()> {
        if !self.stopped {
            return Err(RingKernelError::InvalidState {
                expected: "stopped".to_string(),
                actual: "not stopped".to_string(),
            });
        }
        self.stop.synchronize()
    }

    /// Check if the timer measurement is complete (non-blocking).
    ///
    /// Returns `true` if both start and stop events have completed.
    ///
    /// # Safety
    ///
    /// `start()` and `stop()` must have been called before this.
    pub unsafe fn is_complete(&self) -> Result<bool> {
        if !self.stopped {
            return Ok(false);
        }
        self.stop.query()
    }

    /// Get the elapsed time in milliseconds.
    ///
    /// This blocks until the stop event is complete, then returns the elapsed
    /// time between start and stop events.
    ///
    /// # Safety
    ///
    /// `start()` and `stop()` must have been called before this.
    pub unsafe fn elapsed_ms(&self) -> Result<f32> {
        if !self.stopped {
            return Err(RingKernelError::InvalidState {
                expected: "stopped".to_string(),
                actual: "not stopped".to_string(),
            });
        }

        // Synchronize to ensure events are complete
        self.stop.synchronize()?;

        cuda_result::event::elapsed(self.start.event, self.stop.event).map_err(|e| {
            RingKernelError::BackendError(format!("Failed to get elapsed time: {:?}", e))
        })
    }

    /// Reset the timer for reuse.
    pub fn reset(&mut self) {
        self.started = false;
        self.stopped = false;
    }

    /// Check if the timer has been started.
    pub fn is_started(&self) -> bool {
        self.started
    }

    /// Check if the timer has been stopped.
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }
}

/// A pool of reusable GPU timers.
///
/// This pool maintains a collection of `GpuTimer` instances that can be
/// borrowed and returned, reducing the overhead of creating new timers.
/// Uses interior mutability to allow multiple handles to be acquired.
pub struct GpuTimerPool {
    timers: parking_lot::RwLock<Vec<GpuTimer>>,
    available: parking_lot::Mutex<Vec<usize>>,
    next_id: AtomicUsize,
}

impl GpuTimerPool {
    /// Create a new timer pool with the specified initial capacity.
    pub fn new(capacity: usize) -> Result<Self> {
        let mut timers = Vec::with_capacity(capacity);
        let mut available = Vec::with_capacity(capacity);

        for i in 0..capacity {
            timers.push(GpuTimer::new()?);
            available.push(i);
        }

        Ok(Self {
            timers: parking_lot::RwLock::new(timers),
            available: parking_lot::Mutex::new(available),
            next_id: AtomicUsize::new(capacity),
        })
    }

    /// Acquire a timer from the pool.
    ///
    /// Returns a timer index that can be used to access the timer.
    /// Call `return_timer` when done.
    pub fn acquire(&self) -> Result<usize> {
        let mut available = self.available.lock();

        if let Some(idx) = available.pop() {
            let mut timers = self.timers.write();
            timers[idx].reset();
            Ok(idx)
        } else {
            let idx = self.next_id.fetch_add(1, Ordering::Relaxed);
            let mut timers = self.timers.write();
            timers.push(GpuTimer::new()?);
            Ok(idx)
        }
    }

    /// Get a reference to a timer by index.
    pub fn get(&self, index: usize) -> Option<parking_lot::MappedRwLockReadGuard<'_, GpuTimer>> {
        let timers = self.timers.read();
        if index < timers.len() {
            Some(parking_lot::RwLockReadGuard::map(timers, |t| &t[index]))
        } else {
            None
        }
    }

    /// Get a mutable reference to a timer by index.
    pub fn get_mut(
        &self,
        index: usize,
    ) -> Option<parking_lot::MappedRwLockWriteGuard<'_, GpuTimer>> {
        let timers = self.timers.write();
        if index < timers.len() {
            Some(parking_lot::RwLockWriteGuard::map(timers, |t| {
                &mut t[index]
            }))
        } else {
            None
        }
    }

    /// Return a timer to the pool.
    pub fn return_timer(&self, index: usize) {
        let timers = self.timers.read();
        if index < timers.len() {
            drop(timers);
            let mut available = self.available.lock();
            available.push(index);
        }
    }

    /// Get the number of available timers in the pool.
    pub fn available_count(&self) -> usize {
        self.available.lock().len()
    }

    /// Get the total number of timers in the pool.
    pub fn total_count(&self) -> usize {
        self.timers.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_event_flags() {
        assert_eq!(
            CudaEventFlags::Default.to_cuda_flags(),
            cuda_sys::CUevent_flags::CU_EVENT_DEFAULT
        );
        assert_eq!(
            CudaEventFlags::BlockingSync.to_cuda_flags(),
            cuda_sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC
        );
        assert_eq!(
            CudaEventFlags::DisableTiming.to_cuda_flags(),
            cuda_sys::CUevent_flags::CU_EVENT_DISABLE_TIMING
        );
        assert_eq!(
            CudaEventFlags::Interprocess.to_cuda_flags(),
            cuda_sys::CUevent_flags::CU_EVENT_INTERPROCESS
        );
    }

    #[test]
    fn test_cuda_event_timing_enabled() {
        // Can't actually create events without CUDA, but we can test the flag logic
        assert!(CudaEventFlags::Default != CudaEventFlags::DisableTiming);
        assert!(CudaEventFlags::BlockingSync != CudaEventFlags::DisableTiming);
    }

    #[test]
    fn test_gpu_timer_state() {
        // Test state machine without actual CUDA
        let timer_result = GpuTimer::new();
        // Skip if no CUDA available
        if timer_result.is_err() {
            return;
        }

        let timer = timer_result.unwrap();
        assert!(!timer.is_started());
        assert!(!timer.is_stopped());
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cuda_event_creation() {
        let event = CudaEvent::new().expect("Failed to create event");
        assert!(event.timing_enabled());
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cuda_event_no_timing() {
        let event =
            CudaEvent::with_flags(CudaEventFlags::DisableTiming).expect("Failed to create event");
        assert!(!event.timing_enabled());
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_gpu_timer_basic() {
        let mut timer = GpuTimer::new().expect("Failed to create timer");

        unsafe {
            timer.start_default().expect("Failed to start timer");
            assert!(timer.is_started());
            assert!(!timer.is_stopped());

            timer.stop_default().expect("Failed to stop timer");
            assert!(timer.is_started());
            assert!(timer.is_stopped());

            let elapsed = timer.elapsed_ms().expect("Failed to get elapsed time");
            assert!(elapsed >= 0.0);
        }
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_gpu_timer_pool() {
        let pool = GpuTimerPool::new(4).expect("Failed to create pool");
        assert_eq!(pool.available_count(), 4);
        assert_eq!(pool.total_count(), 4);

        let idx1 = pool.acquire().expect("Failed to acquire timer");
        assert_eq!(pool.available_count(), 3);

        let idx2 = pool.acquire().expect("Failed to acquire timer");
        assert_eq!(pool.available_count(), 2);

        // Return timers
        pool.return_timer(idx1);
        pool.return_timer(idx2);

        assert_eq!(pool.available_count(), 4);
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_gpu_timer_pool_expansion() {
        let pool = GpuTimerPool::new(2).expect("Failed to create pool");

        let _idx1 = pool.acquire().expect("Failed to acquire");
        let _idx2 = pool.acquire().expect("Failed to acquire");
        let _idx3 = pool.acquire().expect("Failed to acquire"); // Should expand pool

        assert_eq!(pool.total_count(), 3);
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_gpu_timer_pool_get() {
        let pool = GpuTimerPool::new(2).expect("Failed to create pool");

        let idx = pool.acquire().expect("Failed to acquire");

        // Get immutable reference
        {
            let timer = pool.get(idx).expect("Timer not found");
            assert!(!timer.is_started());
        }

        // Get mutable reference
        {
            let timer = pool.get_mut(idx).expect("Timer not found");
            // Can access the timer
            assert!(!timer.is_stopped());
        }

        pool.return_timer(idx);
    }
}
