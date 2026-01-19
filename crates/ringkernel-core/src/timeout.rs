//! Operation-level timeouts and deadline management.
//!
//! This module provides utilities for adding timeouts to async operations,
//! deadline propagation through contexts, and cancellation handling.
//!
//! # Example
//!
//! ```rust,ignore
//! use ringkernel_core::timeout::{TimeoutLayer, Deadline, with_timeout};
//! use std::time::Duration;
//!
//! // Simple timeout wrapper
//! let result = with_timeout(Duration::from_secs(5), async {
//!     // Long-running operation
//!     kernel.send(message).await
//! }).await?;
//!
//! // Deadline propagation
//! let deadline = Deadline::from_duration(Duration::from_secs(30));
//! let ctx = OperationContext::new().with_deadline(deadline);
//!
//! // Check remaining time
//! if let Some(remaining) = ctx.remaining_time() {
//!     if remaining < Duration::from_millis(100) {
//!         return Err("Not enough time remaining");
//!     }
//! }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use pin_project_lite::pin_project;

// ============================================================================
// DEADLINE
// ============================================================================

/// A deadline for an operation.
#[derive(Debug, Clone, Copy)]
pub struct Deadline {
    /// When the deadline expires.
    expires_at: Instant,
}

impl Deadline {
    /// Create a deadline from a duration from now.
    pub fn from_duration(duration: Duration) -> Self {
        Self {
            expires_at: Instant::now() + duration,
        }
    }

    /// Create a deadline at a specific instant.
    pub fn at(instant: Instant) -> Self {
        Self {
            expires_at: instant,
        }
    }

    /// Check if the deadline has passed.
    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.expires_at
    }

    /// Get the remaining time until deadline.
    pub fn remaining(&self) -> Option<Duration> {
        let now = Instant::now();
        if now >= self.expires_at {
            None
        } else {
            Some(self.expires_at - now)
        }
    }

    /// Get the deadline instant.
    pub fn expires_at(&self) -> Instant {
        self.expires_at
    }

    /// Create a child deadline that is the minimum of this and another duration.
    pub fn with_timeout(&self, timeout: Duration) -> Self {
        let timeout_deadline = Instant::now() + timeout;
        Self {
            expires_at: self.expires_at.min(timeout_deadline),
        }
    }
}

// ============================================================================
// OPERATION CONTEXT
// ============================================================================

/// Context for tracking operation state including deadlines and cancellation.
#[derive(Debug, Clone)]
pub struct OperationContext {
    /// Operation name/ID for tracing.
    pub name: Option<String>,
    /// Deadline for the operation.
    pub deadline: Option<Deadline>,
    /// Cancellation flag.
    cancelled: Arc<AtomicBool>,
    /// Parent context (for propagation).
    parent: Option<Arc<OperationContext>>,
    /// Creation time.
    created_at: Instant,
}

impl OperationContext {
    /// Create a new operation context.
    pub fn new() -> Self {
        Self {
            name: None,
            deadline: None,
            cancelled: Arc::new(AtomicBool::new(false)),
            parent: None,
            created_at: Instant::now(),
        }
    }

    /// Set the operation name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set a deadline.
    pub fn with_deadline(mut self, deadline: Deadline) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Set a timeout (creates deadline from duration).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.deadline = Some(Deadline::from_duration(timeout));
        self
    }

    /// Set parent context.
    pub fn with_parent(mut self, parent: Arc<OperationContext>) -> Self {
        // Inherit deadline from parent if not set
        if self.deadline.is_none() {
            self.deadline = parent.deadline;
        } else if let (Some(parent_deadline), Some(ref my_deadline)) =
            (parent.deadline, &self.deadline)
        {
            // Use earlier deadline
            if parent_deadline.expires_at < my_deadline.expires_at {
                self.deadline = Some(parent_deadline);
            }
        }
        self.parent = Some(parent);
        self
    }

    /// Create a child context.
    pub fn child(&self) -> OperationContext {
        OperationContext::new().with_parent(Arc::new(self.clone()))
    }

    /// Create a child context with additional timeout.
    pub fn child_with_timeout(&self, timeout: Duration) -> OperationContext {
        let deadline = match self.deadline {
            Some(d) => d.with_timeout(timeout),
            None => Deadline::from_duration(timeout),
        };
        OperationContext::new()
            .with_deadline(deadline)
            .with_parent(Arc::new(self.clone()))
    }

    /// Check if operation is cancelled.
    pub fn is_cancelled(&self) -> bool {
        if self.cancelled.load(Ordering::Relaxed) {
            return true;
        }
        if let Some(ref parent) = self.parent {
            return parent.is_cancelled();
        }
        false
    }

    /// Cancel the operation.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Check if deadline has expired.
    pub fn is_expired(&self) -> bool {
        self.deadline.map(|d| d.is_expired()).unwrap_or(false)
    }

    /// Get remaining time until deadline.
    pub fn remaining_time(&self) -> Option<Duration> {
        self.deadline.and_then(|d| d.remaining())
    }

    /// Check if operation should continue (not cancelled, not expired).
    pub fn should_continue(&self) -> bool {
        !self.is_cancelled() && !self.is_expired()
    }

    /// Get elapsed time since context creation.
    pub fn elapsed(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get the cancellation token for sharing.
    pub fn cancellation_token(&self) -> CancellationToken {
        CancellationToken {
            flag: self.cancelled.clone(),
        }
    }
}

impl Default for OperationContext {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CANCELLATION TOKEN
// ============================================================================

/// A token that can be used to cancel an operation.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    flag: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a new cancellation token.
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Check if cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    /// Cancel.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TIMEOUT FUTURE
// ============================================================================

/// Error returned when an operation times out.
#[derive(Debug, Clone)]
pub struct TimeoutError {
    /// Operation name (if set).
    pub operation: Option<String>,
    /// Configured timeout duration.
    pub timeout: Duration,
    /// Elapsed time.
    pub elapsed: Duration,
}

impl std::fmt::Display for TimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.operation {
            Some(name) => write!(
                f,
                "Operation '{}' timed out after {:?} (limit: {:?})",
                name, self.elapsed, self.timeout
            ),
            None => write!(
                f,
                "Operation timed out after {:?} (limit: {:?})",
                self.elapsed, self.timeout
            ),
        }
    }
}

impl std::error::Error for TimeoutError {}

pin_project! {
    /// Future that wraps another future with a timeout.
    pub struct Timeout<F> {
        #[pin]
        inner: F,
        deadline: Deadline,
        started_at: Instant,
        operation_name: Option<String>,
    }
}

impl<F: Future> Future for Timeout<F> {
    type Output = Result<F::Output, TimeoutError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();

        // Check timeout first
        if this.deadline.is_expired() {
            return Poll::Ready(Err(TimeoutError {
                operation: this.operation_name.clone(),
                timeout: this
                    .deadline
                    .expires_at()
                    .saturating_duration_since(*this.started_at),
                elapsed: this.started_at.elapsed(),
            }));
        }

        // Poll the inner future
        match this.inner.poll(cx) {
            Poll::Ready(value) => Poll::Ready(Ok(value)),
            Poll::Pending => {
                // Schedule a wakeup at deadline
                // Note: In a real implementation, you'd use tokio::time::timeout_at
                // For now, we just return Pending and rely on the executor
                Poll::Pending
            }
        }
    }
}

/// Wrap a future with a timeout.
pub fn timeout<F: Future>(duration: Duration, future: F) -> Timeout<F> {
    Timeout {
        inner: future,
        deadline: Deadline::from_duration(duration),
        started_at: Instant::now(),
        operation_name: None,
    }
}

/// Wrap a future with a timeout and operation name.
pub fn timeout_named<F: Future>(
    name: impl Into<String>,
    duration: Duration,
    future: F,
) -> Timeout<F> {
    Timeout {
        inner: future,
        deadline: Deadline::from_duration(duration),
        started_at: Instant::now(),
        operation_name: Some(name.into()),
    }
}

/// Execute an async operation with timeout using tokio.
pub async fn with_timeout<F, T>(duration: Duration, future: F) -> Result<T, TimeoutError>
where
    F: Future<Output = T>,
{
    let started_at = Instant::now();
    match tokio::time::timeout(duration, future).await {
        Ok(result) => Ok(result),
        Err(_) => Err(TimeoutError {
            operation: None,
            timeout: duration,
            elapsed: started_at.elapsed(),
        }),
    }
}

/// Execute an async operation with timeout and operation name.
pub async fn with_timeout_named<F, T>(
    name: impl Into<String>,
    duration: Duration,
    future: F,
) -> Result<T, TimeoutError>
where
    F: Future<Output = T>,
{
    let name = name.into();
    let started_at = Instant::now();
    match tokio::time::timeout(duration, future).await {
        Ok(result) => Ok(result),
        Err(_) => Err(TimeoutError {
            operation: Some(name),
            timeout: duration,
            elapsed: started_at.elapsed(),
        }),
    }
}

// ============================================================================
// TIMEOUT STATISTICS
// ============================================================================

/// Statistics for timeout tracking.
#[derive(Debug, Default)]
pub struct TimeoutStats {
    /// Total operations.
    pub total_operations: AtomicU64,
    /// Successful completions.
    pub completed: AtomicU64,
    /// Timeouts.
    pub timeouts: AtomicU64,
    /// Cancellations.
    pub cancellations: AtomicU64,
}

impl TimeoutStats {
    /// Create new stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completion.
    pub fn record_completed(&self) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        self.completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a timeout.
    pub fn record_timeout(&self) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        self.timeouts.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cancellation.
    pub fn record_cancellation(&self) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        self.cancellations.fetch_add(1, Ordering::Relaxed);
    }

    /// Get timeout rate.
    pub fn timeout_rate(&self) -> f64 {
        let total = self.total_operations.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let timeouts = self.timeouts.load(Ordering::Relaxed);
        timeouts as f64 / total as f64
    }

    /// Get snapshot of stats.
    pub fn snapshot(&self) -> TimeoutStatsSnapshot {
        TimeoutStatsSnapshot {
            total_operations: self.total_operations.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            timeouts: self.timeouts.load(Ordering::Relaxed),
            cancellations: self.cancellations.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of timeout statistics.
#[derive(Debug, Clone)]
pub struct TimeoutStatsSnapshot {
    /// Total operations.
    pub total_operations: u64,
    /// Successful completions.
    pub completed: u64,
    /// Timeouts.
    pub timeouts: u64,
    /// Cancellations.
    pub cancellations: u64,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deadline() {
        let deadline = Deadline::from_duration(Duration::from_secs(10));
        assert!(!deadline.is_expired());
        assert!(deadline.remaining().is_some());

        let expired = Deadline::from_duration(Duration::from_nanos(1));
        std::thread::sleep(Duration::from_millis(1));
        assert!(expired.is_expired());
        assert!(expired.remaining().is_none());
    }

    #[test]
    fn test_deadline_with_timeout() {
        let deadline = Deadline::from_duration(Duration::from_secs(60));
        let shorter = deadline.with_timeout(Duration::from_secs(5));

        // shorter should have earlier expiration
        assert!(shorter.expires_at() < deadline.expires_at());
    }

    #[test]
    fn test_operation_context() {
        let ctx = OperationContext::new()
            .with_name("test_op")
            .with_timeout(Duration::from_secs(30));

        assert!(!ctx.is_cancelled());
        assert!(!ctx.is_expired());
        assert!(ctx.should_continue());
        assert!(ctx.remaining_time().is_some());
    }

    #[test]
    fn test_operation_context_cancellation() {
        let ctx = OperationContext::new();
        assert!(!ctx.is_cancelled());

        ctx.cancel();
        assert!(ctx.is_cancelled());
        assert!(!ctx.should_continue());
    }

    #[test]
    fn test_operation_context_parent() {
        let parent = OperationContext::new().with_timeout(Duration::from_secs(30));

        let child = parent.child();

        // Child inherits parent deadline
        assert!(child.deadline.is_some());

        // Parent cancellation propagates to child
        parent.cancel();
        assert!(child.is_cancelled());
    }

    #[test]
    fn test_cancellation_token() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());

        token.cancel();
        assert!(token.is_cancelled());

        // Clone shares state
        let token2 = token.clone();
        assert!(token2.is_cancelled());
    }

    #[test]
    fn test_timeout_error_display() {
        let error = TimeoutError {
            operation: Some("send_message".to_string()),
            timeout: Duration::from_secs(5),
            elapsed: Duration::from_secs(5),
        };

        let display = format!("{}", error);
        assert!(display.contains("send_message"));
        assert!(display.contains("timed out"));
    }

    #[test]
    fn test_timeout_stats() {
        let stats = TimeoutStats::new();

        stats.record_completed();
        stats.record_completed();
        stats.record_timeout();
        stats.record_cancellation();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_operations, 4);
        assert_eq!(snapshot.completed, 2);
        assert_eq!(snapshot.timeouts, 1);
        assert_eq!(snapshot.cancellations, 1);
        assert!((stats.timeout_rate() - 0.25).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_with_timeout_success() {
        let result = with_timeout(Duration::from_secs(5), async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            42
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_with_timeout_failure() {
        let result = with_timeout(Duration::from_millis(10), async {
            tokio::time::sleep(Duration::from_secs(60)).await;
            42
        })
        .await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.elapsed >= Duration::from_millis(10));
    }

    #[tokio::test]
    async fn test_with_timeout_named() {
        let result = with_timeout_named("test_operation", Duration::from_millis(10), async {
            tokio::time::sleep(Duration::from_secs(60)).await;
            42
        })
        .await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(error.operation, Some("test_operation".to_string()));
    }
}
