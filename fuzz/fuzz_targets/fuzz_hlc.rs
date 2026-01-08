//! Fuzz target for Hybrid Logical Clock (HLC) operations.
//!
//! Tests the HLC implementation with random operation sequences
//! to verify clock ordering invariants.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use ringkernel_core::{HlcClock, HlcTimestamp};

/// Operations that can be performed on the HLC.
#[derive(Debug, Arbitrary)]
enum HlcOp {
    /// Generate a new timestamp (tick).
    Tick,
    /// Update clock with a received timestamp.
    Update { physical: u64, logical: u16 },
    /// Compare two timestamps.
    Compare { physical1: u64, logical1: u16, physical2: u64, logical2: u16 },
    /// Get current time.
    Now,
}

/// Fuzz input: node configuration and operation sequence.
#[derive(Debug, Arbitrary)]
struct FuzzInput {
    /// Node ID for the clock.
    node_id: u64,
    /// Operations to perform.
    ops: Vec<HlcOp>,
}

fuzz_target!(|input: FuzzInput| {
    // Limit operations
    if input.ops.len() > 500 {
        return;
    }

    // Create HLC clock
    let clock = HlcClock::new(input.node_id);

    // Track timestamps for ordering verification
    let mut prev_timestamp: Option<HlcTimestamp> = None;

    for op in &input.ops {
        match op {
            HlcOp::Tick => {
                let ts = clock.tick();

                // Verify monotonicity: each tick should produce a strictly greater timestamp
                if let Some(prev) = prev_timestamp {
                    assert!(
                        ts > prev,
                        "HLC tick produced non-monotonic timestamp: {:?} not > {:?}",
                        ts,
                        prev
                    );
                }
                prev_timestamp = Some(ts);
            }
            HlcOp::Update { physical, logical } => {
                // Create a timestamp to update from
                let received = HlcTimestamp::new(*physical, *logical);
                let ts = clock.update(received);

                // Verify that update produces a timestamp >= received
                assert!(
                    ts >= received,
                    "HLC update produced timestamp < received: {:?} < {:?}",
                    ts,
                    received
                );

                // Verify monotonicity
                if let Some(prev) = prev_timestamp {
                    assert!(
                        ts > prev,
                        "HLC update produced non-monotonic timestamp: {:?} not > {:?}",
                        ts,
                        prev
                    );
                }
                prev_timestamp = Some(ts);
            }
            HlcOp::Compare { physical1, logical1, physical2, logical2 } => {
                let ts1 = HlcTimestamp::new(*physical1, *logical1);
                let ts2 = HlcTimestamp::new(*physical2, *logical2);

                // Verify comparison is consistent
                let cmp1 = ts1.cmp(&ts2);
                let cmp2 = ts2.cmp(&ts1);

                // Ensure anti-symmetry
                match cmp1 {
                    std::cmp::Ordering::Less => {
                        assert_eq!(cmp2, std::cmp::Ordering::Greater);
                    }
                    std::cmp::Ordering::Greater => {
                        assert_eq!(cmp2, std::cmp::Ordering::Less);
                    }
                    std::cmp::Ordering::Equal => {
                        assert_eq!(cmp2, std::cmp::Ordering::Equal);
                        assert_eq!(ts1, ts2);
                    }
                }

                // Verify transitivity with a third timestamp if both are not equal
                if ts1 != ts2 {
                    let ts3 = HlcTimestamp::new(
                        (*physical1).wrapping_add(*physical2) / 2,
                        (*logical1).wrapping_add(*logical2) / 2,
                    );
                    let cmp13 = ts1.cmp(&ts3);
                    let cmp32 = ts3.cmp(&ts2);

                    // If ts1 < ts3 < ts2 then ts1 < ts2 (transitivity)
                    if cmp13 == std::cmp::Ordering::Less && cmp32 == std::cmp::Ordering::Less {
                        assert_eq!(cmp1, std::cmp::Ordering::Less);
                    }
                    if cmp13 == std::cmp::Ordering::Greater && cmp32 == std::cmp::Ordering::Greater {
                        assert_eq!(cmp1, std::cmp::Ordering::Greater);
                    }
                }
            }
            HlcOp::Now => {
                let ts = clock.now();

                // Verify that now() returns a timestamp that respects ordering
                if let Some(prev) = prev_timestamp {
                    // now() should return >= previous tick (though may not be strictly greater)
                    assert!(
                        ts >= prev,
                        "HLC now() returned timestamp < previous: {:?} < {:?}",
                        ts,
                        prev
                    );
                }
                // Don't update prev_timestamp since now() doesn't advance the clock
            }
        }
    }

    // Final verification: create many ticks and ensure strict ordering
    let mut timestamps = Vec::new();
    for _ in 0..10 {
        timestamps.push(clock.tick());
    }

    for window in timestamps.windows(2) {
        assert!(
            window[1] > window[0],
            "Final tick sequence not strictly ordered: {:?} not > {:?}",
            window[1],
            window[0]
        );
    }
});
