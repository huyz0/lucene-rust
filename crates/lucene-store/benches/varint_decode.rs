//! Microbenchmarks for `DataInput`'s vint/vlong/group-varint decoders --
//! these run once per value in every postings/doc-values/stored-fields
//! block that hasn't been ported to a bulk-decode path yet (see
//! `docs/parity.md`), so their per-value cost sets a floor under nearly
//! every read-path decode loop in this port.
//!
//! Input bytes are the same `fixtures/data/*.bin` files the differential
//! tests (`crates/lucene-store/tests/java_fixtures.rs`) decode and verify
//! against real Java Lucene output -- reusing them here means the
//! benchmark exercises the same byte shapes (value magnitude distribution,
//! continuation-byte counts) a real segment produces, not synthetic random
//! data.
//!
//! Run with: `cargo bench -p lucene-store`
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_store::{DataInput, SliceInput};

fn fixture_bytes(name: &str) -> Vec<u8> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/data/");
    std::fs::read(format!("{dir}{name}.bin")).expect("run fixtures generator first")
}

fn bench_read_vint(c: &mut Criterion) {
    let bin = fixture_bytes("vint");
    c.bench_function("data_input/read_vint_block", |b| {
        b.iter(|| {
            let mut input = SliceInput::new(&bin);
            let mut sum: i64 = 0;
            while input.remaining() > 0 {
                sum = sum.wrapping_add(input.read_vint().unwrap() as i64);
            }
            black_box(sum)
        })
    });
}

fn bench_read_vlong(c: &mut Criterion) {
    let bin = fixture_bytes("vlong");
    c.bench_function("data_input/read_vlong_block", |b| {
        b.iter(|| {
            let mut input = SliceInput::new(&bin);
            let mut sum: i64 = 0;
            while input.remaining() > 0 {
                sum = sum.wrapping_add(input.read_vlong().unwrap());
            }
            black_box(sum)
        })
    });
}

fn bench_read_zlong(c: &mut Criterion) {
    let bin = fixture_bytes("zlong");
    c.bench_function("data_input/read_zlong_block", |b| {
        b.iter(|| {
            let mut input = SliceInput::new(&bin);
            let mut sum: i64 = 0;
            while input.remaining() > 0 {
                sum = sum.wrapping_add(input.read_zlong().unwrap());
            }
            black_box(sum)
        })
    });
}

fn bench_read_group_vints(c: &mut Criterion) {
    let bin = fixture_bytes("group_vint");
    // Same value count as the differential test's expected file: read once
    // up front just to size the destination buffer.
    let expected_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/data/");
    let count = std::fs::read_to_string(format!("{expected_dir}group_vint.expected"))
        .expect("run fixtures generator first")
        .lines()
        .count();
    let mut dst = vec![0u64; count];
    c.bench_function("data_input/read_group_vints_block", |b| {
        b.iter(|| {
            let mut input = SliceInput::new(&bin);
            input.read_group_vints(&mut dst).unwrap();
            black_box(&dst);
        })
    });
}

/// `DataInput::read_i64s` (Java `DataInput.readLongs`) is how a `.liv`
/// live-docs bitset, a BKD leaf's packed words and stored-fields' bitsets all
/// come off disk -- always a whole run of words at once, never one. 2048
/// words is a 16384-doc live-docs bitset, the same block granularity the
/// `lucene-util` benches use.
fn bench_read_i64s(c: &mut Criterion) {
    let words: Vec<u8> = (0..2048u64)
        .flat_map(|i| (i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).to_le_bytes())
        .collect();
    let mut dst = vec![0i64; 2048];
    c.bench_function("data_input/read_i64s_block", |b| {
        b.iter(|| {
            let mut input = SliceInput::new(&words);
            input.read_i64s(&mut dst).unwrap();
            black_box(&dst);
        })
    });
}

criterion_group!(
    benches,
    bench_read_vint,
    bench_read_vlong,
    bench_read_zlong,
    bench_read_group_vints,
    bench_read_i64s
);
criterion_main!(benches);
