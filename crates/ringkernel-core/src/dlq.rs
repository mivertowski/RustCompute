//! Dead Letter Queue — FR-004
//!
//! Persistent DLQ for failed/undeliverable messages:
//! - Messages that fail processing N times → routed to DLQ
//! - DLQ stored in memory (can be persisted via checkpoint)
//! - Replay API: re-inject messages to specified destination
//! - Per-actor DLQ configuration (max retries, TTL)
//! - DLQ metrics: depth, ingress rate, age distribution

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::message::MessageEnvelope;
use crate::runtime::KernelId;

/// Reason a message was sent to the dead letter queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeadLetterReason {
    /// Message processing failed after max retries.
    MaxRetriesExceeded {
        /// Number of retries attempted.
        retries: u32,
        /// Maximum retries allowed.
        max: u32,
    },
    /// Destination actor not found.
    ActorNotFound {
        /// Name of the actor that was not found.
        actor_name: String,
    },
    /// Queue was full and overflow policy is DropToDlq.
    QueueFull {
        /// Capacity of the full queue.
        queue_capacity: usize,
    },
    /// Message TTL expired before delivery.
    TtlExpired {
        /// Age of the message when it expired.
        age: Duration,
    },
    /// Actor was destroyed while message was in-flight.
    ActorDestroyed {
        /// ID of the destroyed actor.
        actor_id: String,
    },
    /// Explicit rejection by the actor's message handler.
    Rejected {
        /// Reason for rejection.
        reason: String,
    },
}

impl std::fmt::Display for DeadLetterReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaxRetriesExceeded { retries, max } => {
                write!(f, "max retries exceeded ({}/{})", retries, max)
            }
            Self::ActorNotFound { actor_name } => write!(f, "actor not found: {}", actor_name),
            Self::QueueFull { queue_capacity } => {
                write!(f, "queue full (capacity {})", queue_capacity)
            }
            Self::TtlExpired { age } => write!(f, "TTL expired (age {:?})", age),
            Self::ActorDestroyed { actor_id } => write!(f, "actor destroyed: {}", actor_id),
            Self::Rejected { reason } => write!(f, "rejected: {}", reason),
        }
    }
}

/// A message in the dead letter queue with metadata.
#[derive(Debug, Clone)]
pub struct DeadLetter {
    /// The original message.
    pub envelope: MessageEnvelope,
    /// Why it was sent to the DLQ.
    pub reason: DeadLetterReason,
    /// Original destination.
    pub destination: KernelId,
    /// When the message arrived in the DLQ.
    pub arrived_at: Instant,
    /// Number of delivery attempts before DLQ.
    pub attempts: u32,
    /// Unique DLQ sequence number.
    pub sequence: u64,
}

/// Configuration for dead letter queue behavior.
#[derive(Debug, Clone)]
pub struct DlqConfig {
    /// Maximum number of messages in the DLQ.
    pub max_size: usize,
    /// Maximum age of a message in the DLQ before automatic expiry.
    pub max_age: Duration,
    /// Maximum delivery retries before routing to DLQ.
    pub max_retries: u32,
    /// Whether to log each DLQ entry.
    pub log_entries: bool,
}

impl Default for DlqConfig {
    fn default() -> Self {
        Self {
            max_size: 10_000,
            max_age: Duration::from_secs(3600), // 1 hour
            max_retries: 3,
            log_entries: true,
        }
    }
}

/// Dead letter queue.
pub struct DeadLetterQueue {
    /// Queued dead letters.
    letters: VecDeque<DeadLetter>,
    /// Configuration.
    config: DlqConfig,
    /// Sequence counter.
    next_sequence: u64,
    /// Metrics.
    total_received: u64,
    total_replayed: u64,
    total_expired: u64,
}

impl DeadLetterQueue {
    /// Create a new DLQ with the given configuration.
    pub fn new(config: DlqConfig) -> Self {
        Self {
            letters: VecDeque::new(),
            config,
            next_sequence: 0,
            total_received: 0,
            total_replayed: 0,
            total_expired: 0,
        }
    }

    /// Route a message to the DLQ.
    pub fn enqueue(
        &mut self,
        envelope: MessageEnvelope,
        reason: DeadLetterReason,
        destination: KernelId,
        attempts: u32,
    ) {
        let seq = self.next_sequence;
        self.next_sequence += 1;
        self.total_received += 1;

        if self.config.log_entries {
            tracing::warn!(
                sequence = seq,
                destination = %destination,
                reason = %reason,
                attempts = attempts,
                "Message routed to dead letter queue"
            );
        }

        let letter = DeadLetter {
            envelope,
            reason,
            destination,
            arrived_at: Instant::now(),
            attempts,
            sequence: seq,
        };

        self.letters.push_back(letter);

        // Evict oldest if over capacity
        while self.letters.len() > self.config.max_size {
            self.letters.pop_front();
            self.total_expired += 1;
        }
    }

    /// Replay messages from the DLQ matching a filter.
    ///
    /// Returns messages removed from the DLQ for re-delivery.
    pub fn replay<F>(&mut self, filter: F) -> Vec<DeadLetter>
    where
        F: Fn(&DeadLetter) -> bool,
    {
        let mut replayed = Vec::new();
        let mut remaining = VecDeque::new();

        for letter in self.letters.drain(..) {
            if filter(&letter) {
                self.total_replayed += 1;
                replayed.push(letter);
            } else {
                remaining.push_back(letter);
            }
        }

        self.letters = remaining;
        replayed
    }

    /// Replay all messages destined for a specific actor.
    pub fn replay_for(&mut self, destination: &KernelId) -> Vec<DeadLetter> {
        let dest = destination.clone();
        self.replay(move |letter| letter.destination == dest)
    }

    /// Expire messages older than max_age.
    pub fn expire_old(&mut self) -> u64 {
        let max_age = self.config.max_age;
        let before = self.letters.len();

        self.letters
            .retain(|letter| letter.arrived_at.elapsed() < max_age);

        let expired = (before - self.letters.len()) as u64;
        self.total_expired += expired;
        expired
    }

    /// Browse the DLQ contents (non-destructive).
    pub fn browse(&self, limit: usize) -> Vec<&DeadLetter> {
        self.letters.iter().take(limit).collect()
    }

    /// Number of messages in the DLQ.
    pub fn len(&self) -> usize {
        self.letters.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.letters.is_empty()
    }

    /// Clear the DLQ.
    pub fn clear(&mut self) {
        self.letters.clear();
    }

    /// Get DLQ metrics.
    pub fn metrics(&self) -> DlqMetrics {
        let oldest_age = self
            .letters
            .front()
            .map(|l| l.arrived_at.elapsed())
            .unwrap_or_default();

        DlqMetrics {
            depth: self.letters.len() as u64,
            total_received: self.total_received,
            total_replayed: self.total_replayed,
            total_expired: self.total_expired,
            oldest_age,
        }
    }

    /// Get the DLQ configuration.
    pub fn config(&self) -> &DlqConfig {
        &self.config
    }
}

/// DLQ metrics snapshot.
#[derive(Debug, Clone)]
pub struct DlqMetrics {
    /// Current queue depth.
    pub depth: u64,
    /// Total messages received by DLQ (lifetime).
    pub total_received: u64,
    /// Total messages successfully replayed.
    pub total_replayed: u64,
    /// Total messages expired.
    pub total_expired: u64,
    /// Age of the oldest message in the DLQ.
    pub oldest_age: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hlc::HlcTimestamp;
    use crate::message::MessageHeader;

    fn test_envelope() -> MessageEnvelope {
        MessageEnvelope {
            header: MessageHeader::new(1, 0, 1, 64, HlcTimestamp::now(1)),
            payload: vec![42u8; 64],
        }
    }

    #[test]
    fn test_dlq_enqueue_and_browse() {
        let mut dlq = DeadLetterQueue::new(DlqConfig {
            log_entries: false,
            ..Default::default()
        });

        dlq.enqueue(
            test_envelope(),
            DeadLetterReason::MaxRetriesExceeded { retries: 3, max: 3 },
            KernelId::new("actor_a"),
            3,
        );

        assert_eq!(dlq.len(), 1);
        let letters = dlq.browse(10);
        assert_eq!(letters.len(), 1);
        assert_eq!(letters[0].destination, KernelId::new("actor_a"));
    }

    #[test]
    fn test_dlq_replay() {
        let mut dlq = DeadLetterQueue::new(DlqConfig {
            log_entries: false,
            ..Default::default()
        });

        dlq.enqueue(
            test_envelope(),
            DeadLetterReason::QueueFull {
                queue_capacity: 256,
            },
            KernelId::new("a"),
            1,
        );
        dlq.enqueue(
            test_envelope(),
            DeadLetterReason::QueueFull {
                queue_capacity: 256,
            },
            KernelId::new("b"),
            1,
        );
        dlq.enqueue(
            test_envelope(),
            DeadLetterReason::QueueFull {
                queue_capacity: 256,
            },
            KernelId::new("a"),
            1,
        );

        // Replay only actor "a" messages
        let replayed = dlq.replay_for(&KernelId::new("a"));
        assert_eq!(replayed.len(), 2);
        assert_eq!(dlq.len(), 1); // Only "b" remains
    }

    #[test]
    fn test_dlq_max_size() {
        let mut dlq = DeadLetterQueue::new(DlqConfig {
            max_size: 3,
            log_entries: false,
            ..Default::default()
        });

        for i in 0..5 {
            dlq.enqueue(
                test_envelope(),
                DeadLetterReason::ActorNotFound {
                    actor_name: format!("a{}", i),
                },
                KernelId::new(&format!("k{}", i)),
                1,
            );
        }

        assert_eq!(dlq.len(), 3); // Oldest 2 evicted
        let metrics = dlq.metrics();
        assert_eq!(metrics.total_received, 5);
        assert_eq!(metrics.total_expired, 2);
    }

    #[test]
    fn test_dlq_replay_with_filter() {
        let mut dlq = DeadLetterQueue::new(DlqConfig {
            log_entries: false,
            ..Default::default()
        });

        dlq.enqueue(
            test_envelope(),
            DeadLetterReason::MaxRetriesExceeded { retries: 3, max: 3 },
            KernelId::new("a"),
            3,
        );
        dlq.enqueue(
            test_envelope(),
            DeadLetterReason::QueueFull {
                queue_capacity: 256,
            },
            KernelId::new("a"),
            1,
        );

        // Replay only queue-full messages
        let replayed = dlq.replay(|l| matches!(l.reason, DeadLetterReason::QueueFull { .. }));
        assert_eq!(replayed.len(), 1);
        assert_eq!(dlq.len(), 1);
    }

    #[test]
    fn test_dlq_metrics() {
        let mut dlq = DeadLetterQueue::new(DlqConfig {
            log_entries: false,
            ..Default::default()
        });

        dlq.enqueue(
            test_envelope(),
            DeadLetterReason::Rejected {
                reason: "test".into(),
            },
            KernelId::new("a"),
            1,
        );
        dlq.enqueue(
            test_envelope(),
            DeadLetterReason::Rejected {
                reason: "test".into(),
            },
            KernelId::new("b"),
            1,
        );
        dlq.replay_for(&KernelId::new("a"));

        let m = dlq.metrics();
        assert_eq!(m.total_received, 2);
        assert_eq!(m.total_replayed, 1);
        assert_eq!(m.depth, 1);
    }

    #[test]
    fn test_dead_letter_reason_display() {
        let r = DeadLetterReason::MaxRetriesExceeded { retries: 3, max: 3 };
        assert!(format!("{}", r).contains("3/3"));

        let r = DeadLetterReason::ActorNotFound {
            actor_name: "test".into(),
        };
        assert!(format!("{}", r).contains("test"));
    }
}
