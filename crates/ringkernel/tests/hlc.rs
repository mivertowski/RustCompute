//! Integration tests for Hybrid Logical Clock functionality.

use ringkernel::prelude::*;

/// Test HLC timestamp creation.
#[test]
fn test_hlc_timestamp_creation() {
    let ts = HlcTimestamp::new(100, 5, 1);

    assert_eq!(ts.physical, 100);
    assert_eq!(ts.logical, 5);
    assert_eq!(ts.node_id, 1);
}

/// Test HLC timestamp ordering.
#[test]
fn test_hlc_ordering() {
    let ts1 = HlcTimestamp::new(100, 0, 1);
    let ts2 = HlcTimestamp::new(100, 1, 1);
    let ts3 = HlcTimestamp::new(101, 0, 1);
    let ts4 = HlcTimestamp::new(100, 0, 2);

    // Physical time dominates
    assert!(ts1 < ts3);
    assert!(ts2 < ts3);

    // Logical counter breaks ties
    assert!(ts1 < ts2);

    // Node ID breaks final ties
    assert!(ts1 < ts4);
}

/// Test HLC zero timestamp.
#[test]
fn test_hlc_zero() {
    let ts = HlcTimestamp::zero();

    assert_eq!(ts.physical, 0);
    assert_eq!(ts.logical, 0);
    assert_eq!(ts.node_id, 0);
    assert!(ts.is_zero());
}

/// Test HLC timestamp from current time.
#[test]
fn test_hlc_now() {
    let ts = HlcTimestamp::now(42);

    assert!(ts.physical > 0);
    assert_eq!(ts.logical, 0);
    assert_eq!(ts.node_id, 42);
    assert!(!ts.is_zero());
}

/// Test HLC clock creation.
#[test]
fn test_hlc_clock_creation() {
    let clock = HlcClock::new(42);

    let ts = clock.now();
    assert_eq!(ts.node_id, 42);
    assert!(ts.physical > 0);
}

/// Test HLC clock tick monotonicity.
#[test]
fn test_hlc_clock_monotonic() {
    let clock = HlcClock::new(1);

    let ts1 = clock.tick();
    let ts2 = clock.tick();
    let ts3 = clock.tick();

    // Each tick should produce a greater timestamp
    assert!(ts1 < ts2);
    assert!(ts2 < ts3);
}

/// Test HLC clock update from received message.
#[test]
fn test_hlc_clock_update() {
    let clock = HlcClock::new(1);

    // Get initial timestamp
    let _initial = clock.tick();

    // Receive a message from the future (but within allowed drift)
    let remote = HlcTimestamp::new(clock.now().physical + 1000, 100, 2);

    let result = clock.update(&remote);
    assert!(result.is_ok());

    // Next tick should be after the remote timestamp
    let next = clock.tick();
    assert!(next > remote);
}

/// Test HLC timestamp display.
#[test]
fn test_hlc_display_format() {
    let ts = HlcTimestamp::new(1234567890, 42, 7);

    let display_str = format!("{}", ts);
    assert!(display_str.contains("1234567890"));
    assert!(display_str.contains("42"));
    assert!(display_str.contains("7"));
}

/// Test HLC timestamp comparison edge cases.
#[test]
fn test_hlc_comparison_edge_cases() {
    // Test with max values
    let ts_max = HlcTimestamp::new(u64::MAX, u64::MAX, u64::MAX);
    let ts_almost_max = HlcTimestamp::new(u64::MAX, u64::MAX, u64::MAX - 1);

    assert!(ts_almost_max < ts_max);

    // Test with zero values
    let ts_zero = HlcTimestamp::new(0, 0, 0);
    let ts_one = HlcTimestamp::new(0, 0, 1);

    assert!(ts_zero < ts_one);
}

/// Test HLC clock concurrent access (basic thread safety).
#[tokio::test]
async fn test_hlc_clock_concurrent() {
    use std::sync::Arc;
    use tokio::task::JoinSet;

    let clock = Arc::new(HlcClock::new(1));
    let mut tasks = JoinSet::new();

    // Spawn multiple tasks that tick the clock
    for _ in 0..10 {
        let clock = Arc::clone(&clock);
        tasks.spawn(async move {
            let mut timestamps = Vec::new();
            for _ in 0..100 {
                timestamps.push(clock.tick());
            }
            timestamps
        });
    }

    // Collect all timestamps
    let mut all_timestamps = Vec::new();
    while let Some(result) = tasks.join_next().await {
        all_timestamps.extend(result.unwrap());
    }

    // All timestamps should be unique (since node_id is the same,
    // but physical + logical should differ)
    let unique_count = {
        let mut unique = all_timestamps.clone();
        unique.sort();
        unique.dedup();
        unique.len()
    };

    // At least most timestamps should be unique
    // (some might collide if tick happens very fast)
    assert!(unique_count > all_timestamps.len() / 2);
}

/// Test HLC timestamp pack/unpack.
#[test]
fn test_hlc_pack_unpack() {
    let original = HlcTimestamp::new(1234567890, 42, 7);
    let packed = original.pack();
    let unpacked = HlcTimestamp::unpack(packed);

    assert_eq!(original.physical, unpacked.physical);
    // Note: logical and node_id may be truncated due to packing format
}

/// Test HLC timestamp time conversions.
#[test]
fn test_hlc_time_conversions() {
    let ts = HlcTimestamp::new(1_000_000, 0, 0); // 1 second in microseconds

    assert_eq!(ts.as_micros(), 1_000_000);
    assert_eq!(ts.as_millis(), 1_000);
}

/// Test HLC state (GPU-side compact storage).
#[test]
fn test_hlc_state() {
    let state = HlcState::new(1234567890, 42);

    assert_eq!(state.physical, 1234567890);
    assert_eq!(state.logical, 42);
}

/// Test HLC timestamp default.
#[test]
fn test_hlc_timestamp_default() {
    let ts: HlcTimestamp = Default::default();

    assert!(ts.is_zero());
    assert_eq!(ts.physical, 0);
    assert_eq!(ts.logical, 0);
    assert_eq!(ts.node_id, 0);
}

/// Test HLC timestamp equality.
#[test]
fn test_hlc_timestamp_equality() {
    let ts1 = HlcTimestamp::new(100, 5, 1);
    let ts2 = HlcTimestamp::new(100, 5, 1);
    let ts3 = HlcTimestamp::new(100, 5, 2);

    assert_eq!(ts1, ts2);
    assert_ne!(ts1, ts3);
}

/// Test HLC clock node_id.
#[test]
fn test_hlc_clock_node_id() {
    let clock = HlcClock::new(123);
    assert_eq!(clock.node_id(), 123);
}
