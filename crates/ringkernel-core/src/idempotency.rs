//! Message Idempotency & Deduplication — FR-006
//!
//! Deduplication layer for K2K messages ensuring exactly-once semantics:
//! - Each message carries an idempotency key (derived from MessageId)
//! - Receiver maintains a bounded dedup cache with configurable TTL
//! - Duplicate messages are silently dropped (logged, not processed)
//!
//! # Usage
//!
//! ```ignore
//! let mut dedup = DeduplicationCache::new(DeduplicationConfig::default());
//!
//! // First time: not a duplicate
//! assert!(!dedup.is_duplicate(msg_id_1));
//!
//! // Second time: duplicate detected
//! assert!(dedup.is_duplicate(msg_id_1));
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Configuration for the deduplication cache.
#[derive(Debug, Clone)]
pub struct DeduplicationConfig {
    /// Maximum entries in the cache.
    pub max_entries: usize,
    /// Time-to-live for cache entries.
    pub ttl: Duration,
    /// Whether to log duplicate detections.
    pub log_duplicates: bool,
}

impl Default for DeduplicationConfig {
    fn default() -> Self {
        Self {
            max_entries: 100_000,
            ttl: Duration::from_secs(300), // 5 minutes
            log_duplicates: false,
        }
    }
}

/// Deduplication cache for K2K message processing.
///
/// Tracks recently-seen message IDs to prevent duplicate processing.
/// Uses a bounded LRU-style eviction with TTL expiry.
pub struct DeduplicationCache {
    /// Map from idempotency key to insertion time.
    seen: HashMap<u64, Instant>,
    /// Insertion-order queue for TTL eviction.
    order: VecDeque<(u64, Instant)>,
    /// Configuration.
    config: DeduplicationConfig,
    /// Metrics.
    duplicates_detected: u64,
    total_checked: u64,
}

impl DeduplicationCache {
    /// Create a new deduplication cache.
    pub fn new(config: DeduplicationConfig) -> Self {
        Self {
            seen: HashMap::with_capacity(config.max_entries),
            order: VecDeque::with_capacity(config.max_entries),
            config,
            duplicates_detected: 0,
            total_checked: 0,
        }
    }

    /// Check if a message ID is a duplicate. If not, records it.
    ///
    /// Returns `true` if this message was already seen (duplicate).
    /// Returns `false` if this is the first time (not duplicate, now recorded).
    pub fn is_duplicate(&mut self, idempotency_key: u64) -> bool {
        self.total_checked += 1;
        self.expire_old();

        if self.seen.contains_key(&idempotency_key) {
            self.duplicates_detected += 1;
            if self.config.log_duplicates {
                tracing::debug!(key = idempotency_key, "Duplicate message detected");
            }
            return true;
        }

        // Record the new key
        let now = Instant::now();
        self.seen.insert(idempotency_key, now);
        self.order.push_back((idempotency_key, now));

        // Evict if over capacity
        while self.seen.len() > self.config.max_entries {
            if let Some((old_key, _)) = self.order.pop_front() {
                self.seen.remove(&old_key);
            }
        }

        false
    }

    /// Check if a key was seen without recording it.
    pub fn was_seen(&self, idempotency_key: u64) -> bool {
        self.seen.contains_key(&idempotency_key)
    }

    /// Expire entries older than TTL.
    fn expire_old(&mut self) {
        let cutoff = Instant::now() - self.config.ttl;

        while let Some(&(key, time)) = self.order.front() {
            if time < cutoff {
                self.order.pop_front();
                self.seen.remove(&key);
            } else {
                break; // Remaining entries are newer
            }
        }
    }

    /// Number of entries in the cache.
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.seen.clear();
        self.order.clear();
    }

    /// Get deduplication metrics.
    pub fn metrics(&self) -> DeduplicationMetrics {
        DeduplicationMetrics {
            cache_size: self.seen.len() as u64,
            total_checked: self.total_checked,
            duplicates_detected: self.duplicates_detected,
            dedup_rate: if self.total_checked > 0 {
                self.duplicates_detected as f64 / self.total_checked as f64
            } else {
                0.0
            },
        }
    }
}

/// Deduplication metrics.
#[derive(Debug, Clone)]
pub struct DeduplicationMetrics {
    /// Current cache size.
    pub cache_size: u64,
    /// Total messages checked.
    pub total_checked: u64,
    /// Duplicates detected.
    pub duplicates_detected: u64,
    /// Deduplication rate (duplicates / total).
    pub dedup_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_dedup() {
        let mut cache = DeduplicationCache::new(DeduplicationConfig::default());

        assert!(!cache.is_duplicate(1)); // First time
        assert!(cache.is_duplicate(1));  // Duplicate
        assert!(!cache.is_duplicate(2)); // New message
        assert!(cache.is_duplicate(2));  // Duplicate
    }

    #[test]
    fn test_capacity_eviction() {
        let mut cache = DeduplicationCache::new(DeduplicationConfig {
            max_entries: 3,
            ..Default::default()
        });

        cache.is_duplicate(1);
        cache.is_duplicate(2);
        cache.is_duplicate(3);
        cache.is_duplicate(4); // Evicts 1

        assert!(!cache.was_seen(1)); // Evicted
        assert!(cache.was_seen(2));
        assert!(cache.was_seen(3));
        assert!(cache.was_seen(4));
    }

    #[test]
    fn test_metrics() {
        let mut cache = DeduplicationCache::new(DeduplicationConfig::default());

        cache.is_duplicate(1);
        cache.is_duplicate(2);
        cache.is_duplicate(1); // Duplicate

        let m = cache.metrics();
        assert_eq!(m.total_checked, 3);
        assert_eq!(m.duplicates_detected, 1);
        assert!((m.dedup_rate - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_clear() {
        let mut cache = DeduplicationCache::new(DeduplicationConfig::default());
        cache.is_duplicate(1);
        cache.is_duplicate(2);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(!cache.is_duplicate(1)); // No longer seen
    }

    #[test]
    fn test_ttl_expiry() {
        let mut cache = DeduplicationCache::new(DeduplicationConfig {
            ttl: Duration::from_millis(1), // Very short TTL for testing
            ..Default::default()
        });

        cache.is_duplicate(1);
        std::thread::sleep(Duration::from_millis(5));

        // After TTL, message should not be considered duplicate
        assert!(!cache.is_duplicate(1));
    }
}
