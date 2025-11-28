//! Topic-based publish/subscribe messaging.
//!
//! This module provides a topic-based pub/sub system for kernels to
//! communicate through named topics without direct knowledge of each other.

use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;

use crate::error::Result;
use crate::hlc::HlcTimestamp;
use crate::message::MessageEnvelope;
use crate::runtime::KernelId;

/// A topic name for pub/sub messaging.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Topic(pub String);

impl Topic {
    /// Create a new topic.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get the topic name.
    pub fn name(&self) -> &str {
        &self.0
    }

    /// Check if this is a wildcard pattern.
    pub fn is_pattern(&self) -> bool {
        self.0.contains('*') || self.0.contains('#')
    }

    /// Check if a topic matches this pattern.
    pub fn matches(&self, other: &Topic) -> bool {
        if !self.is_pattern() {
            return self.0 == other.0;
        }

        // Simple wildcard matching
        // * matches one level, # matches multiple levels
        let pattern_parts: Vec<&str> = self.0.split('/').collect();
        let topic_parts: Vec<&str> = other.0.split('/').collect();

        let mut p_idx = 0;
        let mut t_idx = 0;

        while p_idx < pattern_parts.len() && t_idx < topic_parts.len() {
            match pattern_parts[p_idx] {
                "#" => return true, // # matches everything remaining
                "*" => {
                    // * matches exactly one level
                    p_idx += 1;
                    t_idx += 1;
                }
                part if part == topic_parts[t_idx] => {
                    p_idx += 1;
                    t_idx += 1;
                }
                _ => return false,
            }
        }

        p_idx == pattern_parts.len() && t_idx == topic_parts.len()
    }
}

impl std::fmt::Display for Topic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for Topic {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for Topic {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Configuration for the pub/sub broker.
#[derive(Debug, Clone)]
pub struct PubSubConfig {
    /// Maximum subscribers per topic.
    pub max_subscribers_per_topic: usize,
    /// Channel buffer size for each subscription.
    pub channel_buffer_size: usize,
    /// Maximum retained messages per topic.
    pub max_retained_messages: usize,
    /// Enable message persistence.
    pub enable_persistence: bool,
    /// Default QoS level.
    pub default_qos: QoS,
}

impl Default for PubSubConfig {
    fn default() -> Self {
        Self {
            max_subscribers_per_topic: 1000,
            channel_buffer_size: 256,
            max_retained_messages: 100,
            enable_persistence: false,
            default_qos: QoS::AtMostOnce,
        }
    }
}

/// Quality of Service level for message delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QoS {
    /// Fire and forget - no delivery guarantee.
    #[default]
    AtMostOnce,
    /// Deliver at least once (may duplicate).
    AtLeastOnce,
    /// Deliver exactly once.
    ExactlyOnce,
}

/// A published message on a topic.
#[derive(Debug, Clone)]
pub struct Publication {
    /// Topic the message was published to.
    pub topic: Topic,
    /// Publisher kernel ID.
    pub publisher: KernelId,
    /// The message envelope.
    pub envelope: MessageEnvelope,
    /// Publication timestamp.
    pub timestamp: HlcTimestamp,
    /// QoS level.
    pub qos: QoS,
    /// Sequence number (for ordering).
    pub sequence: u64,
    /// Whether this is a retained message.
    pub retained: bool,
}

impl Publication {
    /// Create a new publication.
    pub fn new(
        topic: Topic,
        publisher: KernelId,
        envelope: MessageEnvelope,
        timestamp: HlcTimestamp,
    ) -> Self {
        Self {
            topic,
            publisher,
            envelope,
            timestamp,
            qos: QoS::default(),
            sequence: 0,
            retained: false,
        }
    }

    /// Set QoS level.
    pub fn with_qos(mut self, qos: QoS) -> Self {
        self.qos = qos;
        self
    }

    /// Mark as retained.
    pub fn with_retained(mut self, retained: bool) -> Self {
        self.retained = retained;
        self
    }
}

/// A subscription to a topic.
pub struct Subscription {
    /// Subscription ID.
    pub id: u64,
    /// Topic pattern (may include wildcards).
    pub pattern: Topic,
    /// Subscriber kernel ID.
    pub subscriber: KernelId,
    /// Message receiver.
    receiver: broadcast::Receiver<Publication>,
    /// Reference to broker for unsubscribe.
    broker: Arc<PubSubBroker>,
}

impl Subscription {
    /// Receive the next publication.
    pub async fn receive(&mut self) -> Option<Publication> {
        loop {
            match self.receiver.recv().await {
                Ok(pub_msg) => {
                    if self.pattern.matches(&pub_msg.topic) {
                        return Some(pub_msg);
                    }
                }
                Err(broadcast::error::RecvError::Closed) => return None,
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
            }
        }
    }

    /// Try to receive a publication (non-blocking).
    pub fn try_receive(&mut self) -> Option<Publication> {
        loop {
            match self.receiver.try_recv() {
                Ok(pub_msg) => {
                    if self.pattern.matches(&pub_msg.topic) {
                        return Some(pub_msg);
                    }
                }
                Err(_) => return None,
            }
        }
    }

    /// Unsubscribe from the topic.
    pub fn unsubscribe(self) {
        self.broker.unsubscribe(self.id);
    }
}

/// Topic info and statistics.
#[derive(Debug, Clone)]
pub struct TopicInfo {
    /// Topic name.
    pub topic: Topic,
    /// Number of subscribers.
    pub subscriber_count: usize,
    /// Total messages published.
    pub messages_published: u64,
    /// Retained message count.
    pub retained_count: usize,
}

/// Pub/sub message broker.
pub struct PubSubBroker {
    /// Configuration.
    config: PubSubConfig,
    /// Broadcast sender for all publications.
    sender: broadcast::Sender<Publication>,
    /// Subscriptions by ID.
    subscriptions: RwLock<HashMap<u64, SubscriptionInfo>>,
    /// Subscription counter.
    subscription_counter: AtomicU64,
    /// Topic statistics.
    topic_stats: RwLock<HashMap<Topic, TopicStats>>,
    /// Retained messages per topic.
    retained: RwLock<HashMap<Topic, Vec<Publication>>>,
    /// Global message sequence.
    sequence: AtomicU64,
}

/// Internal subscription info.
struct SubscriptionInfo {
    pattern: Topic,
    #[allow(dead_code)]
    subscriber: KernelId,
}

/// Topic statistics.
#[derive(Debug, Clone, Default)]
struct TopicStats {
    subscribers: HashSet<u64>,
    messages_published: u64,
}

impl PubSubBroker {
    /// Create a new pub/sub broker.
    pub fn new(config: PubSubConfig) -> Arc<Self> {
        let (sender, _) = broadcast::channel(config.channel_buffer_size);

        Arc::new(Self {
            config,
            sender,
            subscriptions: RwLock::new(HashMap::new()),
            subscription_counter: AtomicU64::new(0),
            topic_stats: RwLock::new(HashMap::new()),
            retained: RwLock::new(HashMap::new()),
            sequence: AtomicU64::new(0),
        })
    }

    /// Subscribe to a topic.
    pub fn subscribe(self: &Arc<Self>, subscriber: KernelId, pattern: Topic) -> Subscription {
        let id = self.subscription_counter.fetch_add(1, Ordering::Relaxed);

        // Store subscription info
        self.subscriptions.write().insert(
            id,
            SubscriptionInfo {
                pattern: pattern.clone(),
                subscriber: subscriber.clone(),
            },
        );

        // Update topic stats
        let mut stats = self.topic_stats.write();
        stats
            .entry(pattern.clone())
            .or_default()
            .subscribers
            .insert(id);

        Subscription {
            id,
            pattern,
            subscriber,
            receiver: self.sender.subscribe(),
            broker: Arc::clone(self),
        }
    }

    /// Unsubscribe by subscription ID.
    pub fn unsubscribe(&self, subscription_id: u64) {
        let info = self.subscriptions.write().remove(&subscription_id);

        if let Some(info) = info {
            let mut stats = self.topic_stats.write();
            if let Some(topic_stats) = stats.get_mut(&info.pattern) {
                topic_stats.subscribers.remove(&subscription_id);
            }
        }
    }

    /// Publish a message to a topic.
    pub fn publish(
        &self,
        topic: Topic,
        publisher: KernelId,
        envelope: MessageEnvelope,
        timestamp: HlcTimestamp,
    ) -> Result<u64> {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);

        let mut publication = Publication::new(topic.clone(), publisher, envelope, timestamp);
        publication.sequence = sequence;

        // Update stats
        {
            let mut stats = self.topic_stats.write();
            let topic_stats = stats.entry(topic.clone()).or_default();
            topic_stats.messages_published += 1;
        }

        // Handle retained message
        if publication.retained {
            let mut retained = self.retained.write();
            let retained_list = retained.entry(topic).or_default();
            retained_list.push(publication.clone());

            // Trim to max retained
            if retained_list.len() > self.config.max_retained_messages {
                retained_list.remove(0);
            }
        }

        // Broadcast to all subscribers
        // Note: subscribers filter by pattern in their receive
        let _ = self.sender.send(publication);

        Ok(sequence)
    }

    /// Publish with QoS setting.
    pub fn publish_qos(
        &self,
        topic: Topic,
        publisher: KernelId,
        envelope: MessageEnvelope,
        timestamp: HlcTimestamp,
        qos: QoS,
    ) -> Result<u64> {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);

        let mut publication = Publication::new(topic.clone(), publisher, envelope, timestamp);
        publication.sequence = sequence;
        publication.qos = qos;

        // Update stats
        {
            let mut stats = self.topic_stats.write();
            let topic_stats = stats.entry(topic).or_default();
            topic_stats.messages_published += 1;
        }

        let _ = self.sender.send(publication);
        Ok(sequence)
    }

    /// Publish a retained message.
    pub fn publish_retained(
        &self,
        topic: Topic,
        publisher: KernelId,
        envelope: MessageEnvelope,
        timestamp: HlcTimestamp,
    ) -> Result<u64> {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);

        let mut publication = Publication::new(topic.clone(), publisher, envelope, timestamp);
        publication.sequence = sequence;
        publication.retained = true;

        // Store retained message
        {
            let mut retained = self.retained.write();
            let retained_list = retained.entry(topic.clone()).or_default();
            retained_list.push(publication.clone());

            if retained_list.len() > self.config.max_retained_messages {
                retained_list.remove(0);
            }
        }

        // Update stats
        {
            let mut stats = self.topic_stats.write();
            let topic_stats = stats.entry(topic).or_default();
            topic_stats.messages_published += 1;
        }

        let _ = self.sender.send(publication);
        Ok(sequence)
    }

    /// Get retained messages for a topic.
    pub fn get_retained(&self, topic: &Topic) -> Vec<Publication> {
        self.retained.read().get(topic).cloned().unwrap_or_default()
    }

    /// Clear retained messages for a topic.
    pub fn clear_retained(&self, topic: &Topic) {
        self.retained.write().remove(topic);
    }

    /// Get topic information.
    pub fn topic_info(&self, topic: &Topic) -> Option<TopicInfo> {
        let stats = self.topic_stats.read();
        let topic_stats = stats.get(topic)?;

        let retained_count = self
            .retained
            .read()
            .get(topic)
            .map(|v| v.len())
            .unwrap_or(0);

        Some(TopicInfo {
            topic: topic.clone(),
            subscriber_count: topic_stats.subscribers.len(),
            messages_published: topic_stats.messages_published,
            retained_count,
        })
    }

    /// List all topics with subscribers.
    pub fn list_topics(&self) -> Vec<Topic> {
        self.topic_stats
            .read()
            .iter()
            .filter(|(_, stats)| !stats.subscribers.is_empty())
            .map(|(topic, _)| topic.clone())
            .collect()
    }

    /// Get broker statistics.
    pub fn stats(&self) -> PubSubStats {
        let stats = self.topic_stats.read();
        let total_subscribers: usize = stats.values().map(|s| s.subscribers.len()).sum();
        let total_messages: u64 = stats.values().map(|s| s.messages_published).sum();
        let retained_count: usize = self.retained.read().values().map(|v| v.len()).sum();

        PubSubStats {
            topic_count: stats.len(),
            total_subscribers,
            total_messages_published: total_messages,
            retained_message_count: retained_count,
        }
    }
}

/// Pub/sub broker statistics.
#[derive(Debug, Clone, Default)]
pub struct PubSubStats {
    /// Number of topics with activity.
    pub topic_count: usize,
    /// Total number of subscriptions.
    pub total_subscribers: usize,
    /// Total messages published.
    pub total_messages_published: u64,
    /// Total retained messages.
    pub retained_message_count: usize,
}

/// Builder for creating pub/sub infrastructure.
pub struct PubSubBuilder {
    config: PubSubConfig,
}

impl PubSubBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: PubSubConfig::default(),
        }
    }

    /// Set maximum subscribers per topic.
    pub fn max_subscribers_per_topic(mut self, count: usize) -> Self {
        self.config.max_subscribers_per_topic = count;
        self
    }

    /// Set channel buffer size.
    pub fn channel_buffer_size(mut self, size: usize) -> Self {
        self.config.channel_buffer_size = size;
        self
    }

    /// Set maximum retained messages.
    pub fn max_retained_messages(mut self, count: usize) -> Self {
        self.config.max_retained_messages = count;
        self
    }

    /// Enable message persistence.
    pub fn enable_persistence(mut self, enable: bool) -> Self {
        self.config.enable_persistence = enable;
        self
    }

    /// Set default QoS.
    pub fn default_qos(mut self, qos: QoS) -> Self {
        self.config.default_qos = qos;
        self
    }

    /// Build the pub/sub broker.
    pub fn build(self) -> Arc<PubSubBroker> {
        PubSubBroker::new(self.config)
    }
}

impl Default for PubSubBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_matching() {
        let pattern = Topic::new("sensors/*/temperature");
        let topic1 = Topic::new("sensors/kitchen/temperature");
        let topic2 = Topic::new("sensors/living_room/temperature");
        let topic3 = Topic::new("sensors/kitchen/humidity");

        assert!(pattern.matches(&topic1));
        assert!(pattern.matches(&topic2));
        assert!(!pattern.matches(&topic3));
    }

    #[test]
    fn test_topic_wildcard_hash() {
        let pattern = Topic::new("sensors/#");
        let topic1 = Topic::new("sensors/kitchen/temperature");
        let topic2 = Topic::new("sensors/a/b/c/d");

        assert!(pattern.matches(&topic1));
        assert!(pattern.matches(&topic2));
    }

    #[test]
    fn test_topic_exact_match() {
        let pattern = Topic::new("sensors/kitchen/temperature");
        let topic1 = Topic::new("sensors/kitchen/temperature");
        let topic2 = Topic::new("sensors/kitchen/humidity");

        assert!(pattern.matches(&topic1));
        assert!(!pattern.matches(&topic2));
    }

    #[tokio::test]
    async fn test_pubsub_broker() {
        let broker = PubSubBuilder::new().build();

        let publisher = KernelId::new("publisher");
        let subscriber = KernelId::new("subscriber");
        let topic = Topic::new("test/topic");

        let mut subscription = broker.subscribe(subscriber, topic.clone());

        // Publish a message
        let envelope = MessageEnvelope::empty(1, 2, HlcTimestamp::now(1));
        let timestamp = HlcTimestamp::now(1);

        broker
            .publish(topic.clone(), publisher.clone(), envelope, timestamp)
            .unwrap();

        // Small delay for broadcast
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Receive the message
        let received = subscription.try_receive();
        assert!(received.is_some());
        assert_eq!(received.unwrap().publisher, publisher);
    }

    #[test]
    fn test_pubsub_stats() {
        let broker = PubSubBuilder::new().build();

        let topic = Topic::new("test");
        let kernel = KernelId::new("kernel");

        let _sub = broker.subscribe(kernel.clone(), topic.clone());

        let stats = broker.stats();
        assert_eq!(stats.topic_count, 1);
        assert_eq!(stats.total_subscribers, 1);
    }
}
