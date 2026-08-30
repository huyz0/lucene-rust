//! Alternating A/B measurement of `IndexedDISI`'s **block jump table**: the
//! same bytes, the same cursor, read once with the table declared and once
//! with it declared absent (`jumpTableEntryCount = 0`, which is what every
//! region this port wrote before `c39-codecs-readpath` carried).
//!
//! Both arms run in one process from one build, so this is the cleanest shape
//! the A/B method has -- there is no second binary and no rebuild between
//! arms, only the `jump_table_entry_count` argument. Criterion is not used:
//! it reported 83/91/129 µs for identical code on this machine (see
//! `docs/sweep/m2/c24-arith-codecs.md`), so every figure here is a
//! **min of N alternating repetitions**, which is the statistic that survives
//! a noisy host.
//!
//! Run: `cargo run --release -p lucene-codecs --example disi_jump_table`
// Benchmark support code opts out of the arithmetic gate at the file
// boundary, as the fixture writers do. See `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::indexed_disi::{self, DisiCursor, DEFAULT_DENSE_RANK_POWER};
use std::hint::black_box;
use std::time::Instant;

/// One cold seek per target: a fresh cursor, then a single `advance_exact`.
/// That is `Lucene90DocValuesProducer`'s single-lookup shape and the only one
/// where the jump table can pay -- a forward scan reads each block header once
/// no matter what.
fn cold_seeks(region: &[u8], jumps: i16, targets: &[i32]) -> u64 {
    let mut found = 0u64;
    for &target in targets {
        let mut cursor = DisiCursor::new(black_box(region), DEFAULT_DENSE_RANK_POWER, jumps);
        if cursor.advance_exact(target).unwrap().is_some() {
            found += 1;
        }
    }
    found
}

/// The other shape: one cursor walked forward over every present doc. The
/// jump table is never consulted here (each step is at most one block ahead),
/// so the two arms differ only by `advance_block`'s new guard -- which makes
/// this the regression check for the scan path c2 optimised.
fn forward_scan(region: &[u8], jumps: i16, docs: &[i32]) -> u64 {
    let mut cursor = DisiCursor::new(black_box(region), DEFAULT_DENSE_RANK_POWER, jumps);
    let mut sum = 0u64;
    for &doc in docs {
        if let Some(ord) = cursor.advance_exact(doc).unwrap() {
            sum = sum.wrapping_add(ord as u64);
        }
    }
    sum
}

fn run(label: &str, max_doc: i32, step: i32, reps: usize) {
    let docs: Vec<i32> = (0..max_doc).step_by(step as usize).collect();
    let (region, jumps) =
        indexed_disi::write_with_dense_rank_power(&docs, DEFAULT_DENSE_RANK_POWER);

    // Targets spread over the whole doc-id range, so the average seek crosses
    // several 65 536-doc blocks. A cursor without the table walks one header
    // per block it passes.
    let targets: Vec<i32> = (0..2000)
        .map(|i| ((i as i64 * 7919) % max_doc as i64) as i32)
        .collect();

    let mut with = u128::MAX;
    let mut without = u128::MAX;
    let mut checksum = 0u64;
    for _ in 0..reps {
        let t = Instant::now();
        checksum ^= cold_seeks(&region, jumps, &targets);
        with = with.min(t.elapsed().as_nanos());

        let t = Instant::now();
        checksum ^= cold_seeks(&region, 0, &targets);
        without = without.min(t.elapsed().as_nanos());
    }

    let mut fwd_with = u128::MAX;
    let mut fwd_without = u128::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        checksum ^= forward_scan(&region, jumps, &docs);
        fwd_with = fwd_with.min(t.elapsed().as_nanos());

        let t = Instant::now();
        checksum ^= forward_scan(&region, 0, &docs);
        fwd_without = fwd_without.min(t.elapsed().as_nanos());
    }

    let blocks = (max_doc as f64 / 65536.0).ceil();
    println!(
        "{label:<28} blocks={blocks:>5.0} region={:>9} B jumps={jumps:>3}  \
         with={:>8.1} µs  without={:>8.1} µs  {:.2}x  (checksum {checksum})",
        region.len(),
        with as f64 / 1000.0,
        without as f64 / 1000.0,
        without as f64 / with as f64,
    );
    println!(
        "{:<28} forward scan of {} present docs: with={:>8.1} µs  without={:>8.1} µs  {:.3}x",
        "",
        docs.len(),
        fwd_with as f64 / 1000.0,
        fwd_without as f64 / 1000.0,
        fwd_without as f64 / fwd_with as f64,
    );
}

fn main() {
    let reps = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(40);
    println!("2000 cold seeks per arm, min of {reps} alternating repetitions\n");
    run("1M docs, 1 in 10", 1_000_000, 10, reps);
    run("4M docs, 1 in 10", 4_000_000, 10, reps);
    run("16M docs, 1 in 40", 16_000_000, 40, reps);
}
