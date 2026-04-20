//! Two-thread SPSC throughput benchmark.
//!
//! Measures sustained throughput with a dedicated producer thread
//! and a dedicated consumer thread — the actual concurrent shape
//! the persistent-actor runtime uses for H2K / K2H queues. The
//! single-threaded `sustained_throughput.rs` test measures
//! operation latency (enqueue+dequeue round-trip in one thread) and
//! does NOT observe cache-coherence costs between separate cores.
//!
//! This test is the baseline for cache-line-padding optimizations
//! (see [`ringkernel_core::queue::CachePadded`]). Expected shape:
//!
//!   - Unpadded (adjacent head/tail on one cache line): throughput
//!     is dominated by cross-core ping-pong; typical numbers
//!     5-15 Mops/s on modern x86 depending on HBM speed.
//!   - Padded (head, tail, and stats each on their own 128-byte
//!     line): throughput is bounded by atomic latency + message
//!     copy bandwidth; typically 2-5x faster on the same hardware.
//!
//! This is `#[ignore]` by default because it's a performance
//! measurement, not a correctness check — a slow run doesn't mean
//! the code is broken. Run explicitly with:
//!
//!     cargo test -p ringkernel-core --release --test \
//!         spsc_two_thread_throughput -- --ignored --nocapture

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ringkernel_core::hlc::HlcTimestamp;
use ringkernel_core::message::{MessageEnvelope, MessageHeader};
use ringkernel_core::queue::{MessageQueue, SpscQueue};

const QUEUE_CAPACITY: usize = 4096;
const PAYLOAD_BYTES: usize = 256;

fn run(duration: Duration) -> (u64, u64) {
    let queue = Arc::new(SpscQueue::new(QUEUE_CAPACITY));
    let stop = Arc::new(AtomicBool::new(false));

    // Use an empty-payload envelope so the benchmark measures queue
    // overhead (atomic coherence, memcpy of the fixed-size header),
    // not payload heap allocation. Callers that care about payload
    // copy cost should combine this number with `memcpy` bandwidth.
    let envelope_template = MessageEnvelope {
        header: MessageHeader::new(1, 0, 1, 0, HlcTimestamp::now(1)),
        payload: Vec::new(),
        ..Default::default()
    };

    let producer_queue = Arc::clone(&queue);
    let producer_stop = Arc::clone(&stop);
    let producer_env = envelope_template.clone();
    let producer = thread::spawn(move || {
        let mut ops: u64 = 0;
        while !producer_stop.load(Ordering::Relaxed) {
            if producer_queue.try_enqueue(producer_env.clone()).is_ok() {
                ops += 1;
            } else {
                // Queue full — spin to let consumer catch up.
                std::hint::spin_loop();
            }
        }
        ops
    });

    let consumer_queue = Arc::clone(&queue);
    let consumer_stop = Arc::clone(&stop);
    let consumer = thread::spawn(move || {
        let mut ops: u64 = 0;
        while !consumer_stop.load(Ordering::Relaxed) || !consumer_queue.is_empty() {
            if consumer_queue.try_dequeue().is_ok() {
                ops += 1;
            } else {
                std::hint::spin_loop();
            }
        }
        ops
    });

    thread::sleep(duration);
    stop.store(true, Ordering::Relaxed);

    let p_ops = producer.join().expect("producer joined");
    let c_ops = consumer.join().expect("consumer joined");
    (p_ops, c_ops)
}

#[test]
#[ignore] // Perf measurement — takes seconds; not a correctness test.
fn spsc_two_thread_10s() {
    let duration = Duration::from_secs(10);
    println!();
    println!("══════════════════════════════════════════════════════════════════");
    println!(
        "  SPSC Two-Thread Throughput ({}-second run)",
        duration.as_secs()
    );
    println!(
        "  Queue capacity: {}, payload: {} B",
        QUEUE_CAPACITY, PAYLOAD_BYTES
    );
    println!("══════════════════════════════════════════════════════════════════");

    // Warmup
    let _ = run(Duration::from_secs(1));

    let t0 = Instant::now();
    let (produced, consumed) = run(duration);
    let elapsed = t0.elapsed().as_secs_f64();

    let p_rate = produced as f64 / elapsed;
    let c_rate = consumed as f64 / elapsed;

    println!();
    println!("  Produced:   {produced} ({:.2} Mmsg/s)", p_rate / 1e6);
    println!("  Consumed:   {consumed} ({:.2} Mmsg/s)", c_rate / 1e6);
    println!("  Elapsed:    {elapsed:.2} s");
    println!();

    // Sanity: at least some throughput, and producer / consumer
    // shouldn't diverge by more than the queue capacity (since the
    // consumer drains on the way out).
    assert!(produced > 1_000, "producer should make progress");
    assert!(consumed > 1_000, "consumer should make progress");
    assert!(
        produced.saturating_sub(consumed) <= QUEUE_CAPACITY as u64,
        "producer/consumer delta should be ≤ queue capacity"
    );
}
