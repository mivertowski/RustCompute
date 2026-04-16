//! Streaming Integrations — FR-011
//!
//! External event stream consumption bridges for Kafka, Redis Streams, and NATS.
//! Each provider implements the `StreamConsumer` trait, which bridges external
//! events into K2K messages for GPU actor processing.
//!
//! # Architecture
//!
//! ```text
//! External Stream          Bridge              GPU Actors
//! ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐
//! │ Kafka Topic │──►│ KafkaConsumer│──►│ K2K → Actor Queue │
//! │ Redis XREAD │──►│ RedisConsumer│──►│ K2K → Actor Queue │
//! │ NATS Subject│──►│ NatsConsumer │──►│ K2K → Actor Queue │
//! └─────────────┘   └──────────────┘   └──────────────────┘
//!                   Backpressure: pause consumption when actors overloaded
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A message consumed from an external stream.
#[derive(Debug, Clone)]
pub struct StreamMessage {
    /// Unique offset/ID within the stream (for commit tracking).
    pub offset: StreamOffset,
    /// Message key (for partitioning).
    pub key: Option<Vec<u8>>,
    /// Message payload.
    pub payload: Vec<u8>,
    /// Timestamp from the source stream.
    pub timestamp: Option<u64>,
    /// Headers/metadata.
    pub headers: HashMap<String, String>,
    /// Source topic/subject/stream name.
    pub source: String,
}

/// Offset within a stream (provider-specific).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StreamOffset {
    /// Kafka-style: topic + partition + offset.
    Kafka { topic: String, partition: i32, offset: i64 },
    /// Redis-style: stream name + message ID.
    Redis { stream: String, id: String },
    /// NATS-style: subject + sequence.
    Nats { subject: String, sequence: u64 },
    /// Generic sequence number.
    Sequence(u64),
}

/// Configuration for a stream consumer.
#[derive(Debug, Clone)]
pub struct StreamConsumerConfig {
    /// Consumer group ID (for parallel consumption).
    pub group_id: String,
    /// Maximum messages to buffer before applying backpressure.
    pub max_buffer_size: usize,
    /// Poll interval when no messages available.
    pub poll_interval: Duration,
    /// Commit strategy.
    pub commit_strategy: CommitStrategy,
    /// Maximum messages per poll.
    pub max_poll_messages: usize,
    /// Connection timeout.
    pub connect_timeout: Duration,
}

impl Default for StreamConsumerConfig {
    fn default() -> Self {
        Self {
            group_id: "ringkernel-default".to_string(),
            max_buffer_size: 10_000,
            poll_interval: Duration::from_millis(100),
            commit_strategy: CommitStrategy::AfterProcess,
            max_poll_messages: 500,
            connect_timeout: Duration::from_secs(10),
        }
    }
}

/// When to commit offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitStrategy {
    /// Commit after each message is processed (at-least-once, slow).
    AfterProcess,
    /// Commit after each batch poll (at-least-once, faster).
    AfterPoll,
    /// Manual commit (caller controls).
    Manual,
}

/// Trait for stream consumer implementations.
///
/// Each provider (Kafka, Redis, NATS) implements this trait.
/// The runtime bridges consumed messages into K2K messages.
pub trait StreamConsumer: Send + Sync {
    /// Connect to the stream source.
    fn connect(&mut self) -> Result<(), StreamError>;

    /// Subscribe to topics/subjects/streams.
    fn subscribe(&mut self, sources: &[&str]) -> Result<(), StreamError>;

    /// Poll for new messages (non-blocking).
    ///
    /// Returns up to `max` messages, or empty if none available.
    fn poll(&mut self, max: usize) -> Result<Vec<StreamMessage>, StreamError>;

    /// Commit offset (acknowledge processing).
    fn commit(&mut self, offset: &StreamOffset) -> Result<(), StreamError>;

    /// Disconnect from the stream source.
    fn disconnect(&mut self) -> Result<(), StreamError>;

    /// Check if connected.
    fn is_connected(&self) -> bool;

    /// Get consumer metrics.
    fn metrics(&self) -> StreamConsumerMetrics;
}

/// Metrics for a stream consumer.
#[derive(Debug, Clone, Default)]
pub struct StreamConsumerMetrics {
    /// Total messages consumed.
    pub messages_consumed: u64,
    /// Total messages committed.
    pub messages_committed: u64,
    /// Total errors.
    pub errors: u64,
    /// Current lag (messages behind head).
    pub lag: u64,
    /// Messages per second (recent).
    pub throughput: f64,
    /// Connected since.
    pub connected_since: Option<Instant>,
}

/// Errors from stream operations.
#[derive(Debug, Clone)]
pub enum StreamError {
    /// Connection failed.
    ConnectionFailed(String),
    /// Subscribe failed.
    SubscribeFailed(String),
    /// Poll failed.
    PollFailed(String),
    /// Commit failed.
    CommitFailed(String),
    /// Timeout.
    Timeout(Duration),
    /// Provider-specific error.
    ProviderError(String),
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
            Self::SubscribeFailed(msg) => write!(f, "Subscribe failed: {}", msg),
            Self::PollFailed(msg) => write!(f, "Poll failed: {}", msg),
            Self::CommitFailed(msg) => write!(f, "Commit failed: {}", msg),
            Self::Timeout(d) => write!(f, "Timeout after {:?}", d),
            Self::ProviderError(msg) => write!(f, "Provider error: {}", msg),
        }
    }
}

impl std::error::Error for StreamError {}

// ============================================================================
// In-Memory Stream Consumer (for testing)
// ============================================================================

/// In-memory stream consumer for testing the bridge without external dependencies.
pub struct InMemoryConsumer {
    messages: Vec<StreamMessage>,
    position: usize,
    committed: Vec<StreamOffset>,
    connected: bool,
    subscriptions: Vec<String>,
    metrics: StreamConsumerMetrics,
}

impl InMemoryConsumer {
    /// Create a new in-memory consumer with pre-loaded messages.
    pub fn new(messages: Vec<StreamMessage>) -> Self {
        Self {
            messages,
            position: 0,
            committed: Vec::new(),
            connected: false,
            subscriptions: Vec::new(),
            metrics: StreamConsumerMetrics::default(),
        }
    }

    /// Create an empty consumer.
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    /// Add a message to the stream.
    pub fn produce(&mut self, source: &str, key: Option<&[u8]>, payload: &[u8]) {
        let seq = self.messages.len() as u64;
        self.messages.push(StreamMessage {
            offset: StreamOffset::Sequence(seq),
            key: key.map(|k| k.to_vec()),
            payload: payload.to_vec(),
            timestamp: Some(seq),
            headers: HashMap::new(),
            source: source.to_string(),
        });
    }

    /// Get committed offsets.
    pub fn committed_offsets(&self) -> &[StreamOffset] {
        &self.committed
    }
}

impl StreamConsumer for InMemoryConsumer {
    fn connect(&mut self) -> Result<(), StreamError> {
        self.connected = true;
        self.metrics.connected_since = Some(Instant::now());
        Ok(())
    }

    fn subscribe(&mut self, sources: &[&str]) -> Result<(), StreamError> {
        self.subscriptions = sources.iter().map(|s| s.to_string()).collect();
        Ok(())
    }

    fn poll(&mut self, max: usize) -> Result<Vec<StreamMessage>, StreamError> {
        if !self.connected {
            return Err(StreamError::ConnectionFailed("Not connected".into()));
        }

        let end = (self.position + max).min(self.messages.len());
        let batch: Vec<StreamMessage> = self.messages[self.position..end]
            .iter()
            .filter(|m| self.subscriptions.is_empty() || self.subscriptions.contains(&m.source))
            .cloned()
            .collect();

        self.position = end;
        self.metrics.messages_consumed += batch.len() as u64;
        Ok(batch)
    }

    fn commit(&mut self, offset: &StreamOffset) -> Result<(), StreamError> {
        self.committed.push(offset.clone());
        self.metrics.messages_committed += 1;
        Ok(())
    }

    fn disconnect(&mut self) -> Result<(), StreamError> {
        self.connected = false;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    fn metrics(&self) -> StreamConsumerMetrics {
        self.metrics.clone()
    }
}

// ============================================================================
// Kafka Consumer Stub (requires rdkafka dependency)
// ============================================================================

/// Kafka consumer configuration.
#[derive(Debug, Clone)]
pub struct KafkaConfig {
    /// Broker addresses (e.g., "localhost:9092").
    pub brokers: String,
    /// Consumer group ID.
    pub group_id: String,
    /// Auto offset reset policy.
    pub auto_offset_reset: String,
    /// Enable auto-commit.
    pub enable_auto_commit: bool,
}

impl Default for KafkaConfig {
    fn default() -> Self {
        Self {
            brokers: "localhost:9092".to_string(),
            group_id: "ringkernel".to_string(),
            auto_offset_reset: "earliest".to_string(),
            enable_auto_commit: false,
        }
    }
}

/// NATS consumer configuration.
#[derive(Debug, Clone)]
pub struct NatsConfig {
    /// NATS server URL.
    pub url: String,
    /// NATS credentials (optional).
    pub credentials: Option<String>,
}

impl Default for NatsConfig {
    fn default() -> Self {
        Self {
            url: "nats://localhost:4222".to_string(),
            credentials: None,
        }
    }
}

/// Redis Streams consumer configuration.
#[derive(Debug, Clone)]
pub struct RedisStreamConfig {
    /// Redis URL.
    pub url: String,
    /// Consumer group name.
    pub group: String,
    /// Consumer name within group.
    pub consumer: String,
}

impl Default for RedisStreamConfig {
    fn default() -> Self {
        Self {
            url: "redis://localhost:6379".to_string(),
            group: "ringkernel".to_string(),
            consumer: "worker-1".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_consumer_basic() {
        let mut consumer = InMemoryConsumer::empty();
        consumer.produce("events", None, b"hello");
        consumer.produce("events", None, b"world");

        consumer.connect().unwrap();
        consumer.subscribe(&["events"]).unwrap();

        let msgs = consumer.poll(10).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].payload, b"hello");
        assert_eq!(msgs[1].payload, b"world");
    }

    #[test]
    fn test_in_memory_consumer_batching() {
        let mut consumer = InMemoryConsumer::empty();
        for i in 0..10 {
            consumer.produce("topic", None, format!("msg-{}", i).as_bytes());
        }

        consumer.connect().unwrap();
        consumer.subscribe(&["topic"]).unwrap();

        let batch1 = consumer.poll(3).unwrap();
        assert_eq!(batch1.len(), 3);

        let batch2 = consumer.poll(3).unwrap();
        assert_eq!(batch2.len(), 3);

        let batch3 = consumer.poll(100).unwrap();
        assert_eq!(batch3.len(), 4); // Remaining
    }

    #[test]
    fn test_in_memory_consumer_commit() {
        let mut consumer = InMemoryConsumer::empty();
        consumer.produce("t", None, b"data");
        consumer.connect().unwrap();
        consumer.subscribe(&["t"]).unwrap();

        let msgs = consumer.poll(1).unwrap();
        consumer.commit(&msgs[0].offset).unwrap();

        assert_eq!(consumer.committed_offsets().len(), 1);
    }

    #[test]
    fn test_in_memory_consumer_not_connected() {
        let mut consumer = InMemoryConsumer::empty();
        let result = consumer.poll(1);
        assert!(result.is_err());
    }

    #[test]
    fn test_stream_consumer_metrics() {
        let mut consumer = InMemoryConsumer::empty();
        consumer.produce("t", None, b"a");
        consumer.produce("t", None, b"b");

        consumer.connect().unwrap();
        consumer.subscribe(&["t"]).unwrap();
        consumer.poll(10).unwrap();

        let m = consumer.metrics();
        assert_eq!(m.messages_consumed, 2);
        assert!(m.connected_since.is_some());
    }

    #[test]
    fn test_kafka_config_defaults() {
        let config = KafkaConfig::default();
        assert_eq!(config.brokers, "localhost:9092");
        assert!(!config.enable_auto_commit);
    }

    #[test]
    fn test_nats_config_defaults() {
        let config = NatsConfig::default();
        assert_eq!(config.url, "nats://localhost:4222");
    }

    #[test]
    fn test_redis_config_defaults() {
        let config = RedisStreamConfig::default();
        assert_eq!(config.group, "ringkernel");
    }

    #[test]
    fn test_stream_offset_equality() {
        let a = StreamOffset::Sequence(42);
        let b = StreamOffset::Sequence(42);
        assert_eq!(a, b);
    }
}
