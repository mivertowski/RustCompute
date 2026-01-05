//! Fuzz target for message queue operations.
//!
//! Tests the lock-free message queue with random operation sequences
//! to find race conditions or crashes.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use ringkernel_core::MessageQueue;
use std::sync::Arc;

/// Operations that can be performed on the message queue.
#[derive(Debug, Arbitrary)]
enum QueueOp {
    /// Push a message to the queue.
    Push { data: Vec<u8> },
    /// Try to pop a message from the queue.
    TryPop,
    /// Check if the queue is empty.
    IsEmpty,
    /// Check if the queue is full.
    IsFull,
    /// Get the queue length.
    Len,
    /// Get the queue capacity.
    Capacity,
    /// Clear the queue.
    Clear,
}

/// Fuzz input: queue configuration and operation sequence.
#[derive(Debug, Arbitrary)]
struct FuzzInput {
    /// Queue capacity (will be rounded to power of 2).
    capacity_log2: u8,
    /// Operations to perform.
    ops: Vec<QueueOp>,
}

fuzz_target!(|input: FuzzInput| {
    // Limit capacity to reasonable range (16 to 4096)
    let capacity_log2 = (input.capacity_log2 % 9).max(4); // 4-12 -> 16 to 4096
    let capacity = 1 << capacity_log2;

    // Limit operations to prevent timeout
    if input.ops.len() > 1000 {
        return;
    }

    // Create queue
    let queue: MessageQueue<Vec<u8>> = MessageQueue::new(capacity);

    // Execute operations
    for op in &input.ops {
        match op {
            QueueOp::Push { data } => {
                // Limit message size
                let data = if data.len() > 1024 {
                    data[..1024].to_vec()
                } else {
                    data.clone()
                };
                let _ = queue.try_push(data);
            }
            QueueOp::TryPop => {
                let _ = queue.try_pop();
            }
            QueueOp::IsEmpty => {
                let _ = queue.is_empty();
            }
            QueueOp::IsFull => {
                let _ = queue.is_full();
            }
            QueueOp::Len => {
                let _ = queue.len();
            }
            QueueOp::Capacity => {
                let _ = queue.capacity();
            }
            QueueOp::Clear => {
                queue.clear();
            }
        }
    }

    // Final consistency check
    let len = queue.len();
    let capacity = queue.capacity();
    assert!(len <= capacity, "Queue length {} exceeds capacity {}", len, capacity);
});
