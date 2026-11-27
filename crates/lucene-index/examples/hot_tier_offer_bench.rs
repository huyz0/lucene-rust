//! What the hot tier's heat tracking costs a row that is not hot: nanoseconds
//! per offered update over N distinct rows, and the tracker's memory as the
//! rows grow. Every offer goes through (no row is updated twice), as a bulk
//! update does.
//!
//! ```text
//! cargo run -p lucene-index --release --example hot_tier_offer_bench
//! ```
#![allow(clippy::arithmetic_side_effects)] // A measurement harness's own sizes.

use std::convert::Infallible;
use std::time::Instant;

use lucene_index::buffered_updates::Term;
use lucene_index::hot_tier::{HeldOp, HotTier, HotTierConfig, Offer};
use lucene_index::index_writer::{ExplicitDocument, ExplicitFields};

fn rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

fn main() {
    // `rate`: offers a second (0: as fast as the loop goes). The engine applies
    // at most about 190k bulk updates a second on this box.
    let rate: f64 = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let sizes: Vec<u64> = if rate > 0.0 {
        vec![1_000_000, 4_000_000]
    } else {
        vec![100_000, 1_000_000, 10_000_000, 50_000_000]
    };
    for distinct in sizes {
        let tier = HotTier::new(HotTierConfig::with_budget(64 << 20));
        let rss0 = rss_kb();
        let mut apply = |_op: HeldOp| -> Result<(), Infallible> { Ok(()) };
        // Pre-build the terms so only the offer is timed.
        let chunk = 1_000_000u64;
        let (mut total_ns, mut through, mut held) = (0u128, 0u64, 0u64);
        let mut start = 0u64;
        while start < distinct {
            let end = (start + chunk).min(distinct);
            let ops: Vec<HeldOp> = (start..end)
                .map(|i| {
                    HeldOp::new(
                        None,
                        Term::new(
                            "_id",
                            format!("d{}", i.wrapping_mul(2_654_435_761)).into_bytes(),
                        ),
                        vec![ExplicitDocument {
                            stored: Vec::new(),
                            fields: ExplicitFields::default(),
                        }],
                        None,
                        Vec::new(),
                    )
                })
                .collect();
            let mut ops = ops.into_iter();
            loop {
                let group: Vec<HeldOp> = ops.by_ref().take(1000).collect();
                if group.is_empty() {
                    break;
                }
                let t = Instant::now();
                for op in group {
                    match tier.offer(op, &mut apply) {
                        Ok(Offer::Through { .. }) => through += 1,
                        Ok(Offer::Held { .. }) => held += 1,
                        Err(never) => match never {},
                    }
                }
                total_ns += t.elapsed().as_nanos();
                if rate > 0.0 {
                    std::thread::sleep(std::time::Duration::from_secs_f64(1000.0 / rate));
                }
            }
            start = end;
        }
        let st = tier.stats();
        println!(
            "{{\"distinct_rows\":{distinct},\"ns_per_offer\":{:.1},\"held_by_mistake\":{held},\"through\":{through},\"tracker_bytes\":{},\"held_rows\":{},\"rss_delta_mb\":{}}}",
            total_ns as f64 / distinct as f64,
            st.sketch_bytes,
            st.held_rows,
            (rss_kb() as i64 - rss0 as i64) / 1024
        );
        let mut drain_apply = |_op: HeldOp| -> Result<(), Infallible> { Ok(()) };
        let _ = tier.drain(&mut drain_apply);
    }
}
