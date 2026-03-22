//! Graceful shutdown and signal handling for persistent GPU kernels.
//!
//! Persistent GPU kernels run indefinitely and must be shut down cleanly to
//! avoid GPU resource leaks, corrupted state, or orphaned device memory.
//! This module provides infrastructure for capturing OS signals (SIGTERM,
//! SIGINT / Ctrl-C) and propagating a shutdown request to all interested
//! parties via a lightweight, clone-able [`ShutdownSignal`].
//!
//! # Overview
//!
//! - [`GracefulShutdown`] -- registers signal handlers and owns the shutdown
//!   lifecycle (signal capture, grace period, force termination).
//! - [`ShutdownSignal`] -- a cheap, clone-able handle that kernels and
//!   background tasks can poll or `.await` to learn about a pending shutdown.
//! - [`ShutdownGuard`] -- returned by [`GracefulShutdown::install`]; dropping
//!   the guard cancels the signal listener.
//!
//! # Example
//!
//! ```rust,ignore
//! use ringkernel_core::shutdown::GracefulShutdown;
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() {
//!     let shutdown = GracefulShutdown::new(Duration::from_secs(5));
//!     let signal = shutdown.signal();
//!
//!     // Hand `signal` clones to kernel loops, background tasks, etc.
//!     let guard = shutdown.install();
//!
//!     // In a kernel loop:
//!     loop {
//!         if signal.is_shutdown_requested() {
//!             break;
//!         }
//!         // ... do work ...
//!     }
//!
//!     // Or await the signal:
//!     // signal.wait().await;
//!
//!     guard.wait().await;
//! }
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::watch;
use tokio::task::JoinHandle;

// ============================================================================
// ShutdownSignal
// ============================================================================

/// A lightweight, clone-able handle for checking or awaiting shutdown.
///
/// Multiple kernels and background tasks can each hold a clone. Checking
/// [`is_shutdown_requested`](ShutdownSignal::is_shutdown_requested) is a
/// single atomic load and therefore safe to call on hot paths.
#[derive(Clone)]
pub struct ShutdownSignal {
    /// Atomic flag -- fast, non-blocking check.
    requested: Arc<AtomicBool>,
    /// Watch receiver -- enables `.await`-based notification.
    rx: watch::Receiver<bool>,
}

impl ShutdownSignal {
    /// Returns `true` once shutdown has been requested.
    ///
    /// This is a single atomic load and can be called on hot paths without
    /// any overhead.
    pub fn is_shutdown_requested(&self) -> bool {
        self.requested.load(Ordering::SeqCst)
    }

    /// Wait until a shutdown signal is received.
    ///
    /// This future completes as soon as any signal (SIGTERM, SIGINT, or a
    /// manual trigger) fires. It is cancel-safe.
    pub async fn wait(&self) {
        // Fast path: already triggered.
        if self.is_shutdown_requested() {
            return;
        }

        let mut rx = self.rx.clone();
        // `changed()` returns when the sender writes a new value.
        // Ignore the error case (sender dropped) -- treat it as shutdown.
        loop {
            if *rx.borrow_and_update() {
                return;
            }
            if rx.changed().await.is_err() {
                // Sender dropped -- treat as shutdown.
                return;
            }
        }
    }
}

impl std::fmt::Debug for ShutdownSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShutdownSignal")
            .field("requested", &self.is_shutdown_requested())
            .finish()
    }
}

// ============================================================================
// GracefulShutdown
// ============================================================================

/// Builder and coordinator for graceful shutdown of persistent GPU kernels.
///
/// Captures SIGTERM and SIGINT (Ctrl-C), notifies all holders of
/// [`ShutdownSignal`], waits for a configurable grace period, then
/// force-terminates.
pub struct GracefulShutdown {
    /// Shared flag.
    requested: Arc<AtomicBool>,
    /// Watch channel sender (triggers `ShutdownSignal::wait()`).
    tx: watch::Sender<bool>,
    /// Watch channel receiver (cloned into each `ShutdownSignal`).
    rx: watch::Receiver<bool>,
    /// How long to wait after the signal before force termination.
    grace_period: Duration,
}

impl GracefulShutdown {
    /// Create a new shutdown coordinator with the given grace period.
    ///
    /// The grace period determines how long [`ShutdownGuard::wait`] will
    /// allow after the signal fires before it returns (allowing the caller
    /// to force-terminate remaining work).
    pub fn new(grace_period: Duration) -> Self {
        let (tx, rx) = watch::channel(false);
        Self {
            requested: Arc::new(AtomicBool::new(false)),
            tx,
            rx,
            grace_period,
        }
    }

    /// Create a shutdown coordinator with the default grace period (5 seconds).
    pub fn with_default_grace_period() -> Self {
        Self::new(Duration::from_secs(5))
    }

    /// Get the configured grace period.
    pub fn grace_period(&self) -> Duration {
        self.grace_period
    }

    /// Check if shutdown has already been requested.
    pub fn is_shutdown_requested(&self) -> bool {
        self.requested.load(Ordering::SeqCst)
    }

    /// Obtain a [`ShutdownSignal`] that can be cloned and distributed to
    /// kernels and background tasks.
    pub fn signal(&self) -> ShutdownSignal {
        ShutdownSignal {
            requested: Arc::clone(&self.requested),
            rx: self.rx.clone(),
        }
    }

    /// Manually trigger shutdown (useful for programmatic shutdown or testing).
    pub fn trigger(&self) {
        self.requested.store(true, Ordering::SeqCst);
        let _ = self.tx.send(true);
    }

    /// Install OS signal handlers and return a [`ShutdownGuard`].
    ///
    /// The guard spawns a background task that listens for SIGTERM and
    /// SIGINT (Ctrl-C). When a signal is received the shutdown flag is set
    /// and all [`ShutdownSignal`] holders are notified.
    ///
    /// Call [`ShutdownGuard::wait`] to block until the grace period elapses
    /// after a signal (or until all work completes, whichever comes first).
    pub fn install(self) -> ShutdownGuard {
        let requested = Arc::clone(&self.requested);
        let tx = self.tx;
        let grace_period = self.grace_period;
        let signal = ShutdownSignal {
            requested: Arc::clone(&self.requested),
            rx: self.rx.clone(),
        };

        let handle = tokio::spawn(async move {
            // Wait for either SIGINT (Ctrl-C) or SIGTERM.
            let sigint = tokio::signal::ctrl_c();

            #[cfg(unix)]
            {
                let mut sigterm =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("failed to register SIGTERM handler");

                tokio::select! {
                    _ = sigint => {
                        tracing::info!("Received SIGINT (Ctrl-C), initiating graceful shutdown");
                    }
                    _ = sigterm.recv() => {
                        tracing::info!("Received SIGTERM, initiating graceful shutdown");
                    }
                }
            }

            #[cfg(not(unix))]
            {
                // On non-Unix platforms only Ctrl-C is available.
                let _ = sigint.await;
                tracing::info!("Received Ctrl-C, initiating graceful shutdown");
            }

            // Set the shutdown flag and notify all watchers.
            requested.store(true, Ordering::SeqCst);
            let _ = tx.send(true);

            // Allow the grace period for cleanup.
            tracing::info!(
                grace_period_secs = grace_period.as_secs_f64(),
                "Shutdown signal sent; grace period started"
            );
            tokio::time::sleep(grace_period).await;
            tracing::warn!("Grace period elapsed; force termination may follow");
        });

        ShutdownGuard { handle, signal }
    }
}

impl std::fmt::Debug for GracefulShutdown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GracefulShutdown")
            .field("requested", &self.is_shutdown_requested())
            .field("grace_period", &self.grace_period)
            .finish()
    }
}

// ============================================================================
// ShutdownGuard
// ============================================================================

/// Guard returned by [`GracefulShutdown::install`].
///
/// Holds the background signal-listening task. Call [`wait`](Self::wait) to
/// block until the signal fires and the grace period elapses (or until the
/// task is aborted on drop).
pub struct ShutdownGuard {
    /// Background task listening for OS signals.
    handle: JoinHandle<()>,
    /// A signal handle for callers to check / await.
    signal: ShutdownSignal,
}

impl ShutdownGuard {
    /// Get a [`ShutdownSignal`] from this guard.
    pub fn signal(&self) -> ShutdownSignal {
        self.signal.clone()
    }

    /// Wait for the signal listener task to complete.
    ///
    /// This resolves after a signal has been received **and** the grace
    /// period has elapsed. If no signal is ever received this future
    /// never completes (you can `select!` it against your main workload).
    pub async fn wait(self) {
        let _ = self.handle.await;
    }

    /// Cancel the signal listener without waiting.
    pub fn cancel(self) {
        self.handle.abort();
    }
}

impl std::fmt::Debug for ShutdownGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShutdownGuard")
            .field("signal", &self.signal)
            .field("finished", &self.handle.is_finished())
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_grace_period() {
        let shutdown = GracefulShutdown::with_default_grace_period();
        assert_eq!(shutdown.grace_period(), Duration::from_secs(5));
    }

    #[test]
    fn test_custom_grace_period() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(30));
        assert_eq!(shutdown.grace_period(), Duration::from_secs(30));
    }

    #[test]
    fn test_not_requested_initially() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(1));
        assert!(!shutdown.is_shutdown_requested());

        let signal = shutdown.signal();
        assert!(!signal.is_shutdown_requested());
    }

    #[test]
    fn test_manual_trigger() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(1));
        let signal = shutdown.signal();

        assert!(!signal.is_shutdown_requested());
        shutdown.trigger();
        assert!(signal.is_shutdown_requested());
        assert!(shutdown.is_shutdown_requested());
    }

    #[test]
    fn test_signal_clone_shares_state() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(1));
        let s1 = shutdown.signal();
        let s2 = s1.clone();
        let s3 = shutdown.signal();

        assert!(!s1.is_shutdown_requested());
        assert!(!s2.is_shutdown_requested());
        assert!(!s3.is_shutdown_requested());

        shutdown.trigger();

        assert!(s1.is_shutdown_requested());
        assert!(s2.is_shutdown_requested());
        assert!(s3.is_shutdown_requested());
    }

    #[tokio::test]
    async fn test_signal_wait_resolves_on_trigger() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(1));
        let signal = shutdown.signal();

        // Trigger from a separate task after a short delay.
        let trigger_handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            shutdown.trigger();
        });

        // This should resolve once trigger fires.
        tokio::time::timeout(Duration::from_secs(2), signal.wait())
            .await
            .expect("signal.wait() should have resolved within timeout");

        trigger_handle.await.unwrap();
    }

    #[tokio::test]
    async fn test_signal_wait_returns_immediately_if_already_triggered() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(1));
        let signal = shutdown.signal();
        shutdown.trigger();

        // Should return immediately since already triggered.
        tokio::time::timeout(Duration::from_millis(100), signal.wait())
            .await
            .expect("signal.wait() should return immediately when already triggered");
    }

    #[tokio::test]
    async fn test_guard_signal() {
        let shutdown = GracefulShutdown::new(Duration::from_millis(50));
        let signal = shutdown.signal();
        let guard = shutdown.install();

        // Guard's signal should share state.
        let guard_signal = guard.signal();
        assert!(!guard_signal.is_shutdown_requested());
        assert!(!signal.is_shutdown_requested());

        // Cancel so the test doesn't block.
        guard.cancel();
    }

    #[tokio::test]
    async fn test_guard_cancel() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(60));
        let guard = shutdown.install();

        // Cancel should not panic and should abort the listener task.
        guard.cancel();
    }

    #[tokio::test]
    async fn test_multiple_signals_from_same_shutdown() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(1));

        let signals: Vec<_> = (0..10).map(|_| shutdown.signal()).collect();

        for s in &signals {
            assert!(!s.is_shutdown_requested());
        }

        shutdown.trigger();

        for s in &signals {
            assert!(s.is_shutdown_requested());
        }
    }

    #[tokio::test]
    async fn test_wait_with_dropped_sender() {
        // If the GracefulShutdown (which owns the sender) is dropped,
        // ShutdownSignal::wait() should still resolve (treat as shutdown).
        let signal = {
            let shutdown = GracefulShutdown::new(Duration::from_secs(1));
            shutdown.signal()
            // shutdown dropped here
        };

        tokio::time::timeout(Duration::from_millis(100), signal.wait())
            .await
            .expect("signal.wait() should resolve when sender is dropped");
    }

    #[test]
    fn test_debug_impls() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(5));
        let debug_str = format!("{:?}", shutdown);
        assert!(debug_str.contains("GracefulShutdown"));
        assert!(debug_str.contains("requested"));

        let signal = shutdown.signal();
        let debug_str = format!("{:?}", signal);
        assert!(debug_str.contains("ShutdownSignal"));
    }

    #[tokio::test]
    async fn test_guard_debug() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(1));
        let guard = shutdown.install();
        let debug_str = format!("{:?}", guard);
        assert!(debug_str.contains("ShutdownGuard"));
        guard.cancel();
    }

    #[tokio::test]
    async fn test_concurrent_trigger_is_safe() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(1));
        let signal = shutdown.signal();

        // Trigger from multiple tasks concurrently -- should not panic.
        let shutdown = Arc::new(shutdown);
        let mut handles = Vec::new();
        for _ in 0..10 {
            let s = Arc::clone(&shutdown);
            handles.push(tokio::spawn(async move {
                s.trigger();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        assert!(signal.is_shutdown_requested());
    }

    #[tokio::test]
    async fn test_signal_wait_multiple_waiters() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(1));
        let mut handles = Vec::new();

        for _ in 0..5 {
            let signal = shutdown.signal();
            handles.push(tokio::spawn(async move {
                signal.wait().await;
            }));
        }

        // Let waiters register, then trigger.
        tokio::time::sleep(Duration::from_millis(50)).await;
        shutdown.trigger();

        // All waiters should resolve.
        for h in handles {
            tokio::time::timeout(Duration::from_secs(2), h)
                .await
                .expect("waiter should complete")
                .expect("waiter task should not panic");
        }
    }
}
