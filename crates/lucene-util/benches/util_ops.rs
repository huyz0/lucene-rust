//! Microbenchmarks for `lucene-util` primitives that sit underneath every
//! decode path in this port: zigzag encode/decode (used by every `zlong`
//! field: norms deltas, some doc-values encodings) and `FixedBitSet`
//! get/cardinality (the in-memory shape `.liv` live-docs bytes decode into,
//! consulted once per candidate doc during collection).
//!
//! `FixedBitSet` sizes use 16384 bits -- the same per-block granularity
//! Lucene's doc-values/postings formats use (`DocValuesConsumer`'s NUMERIC
//! block size, `Lucene90DocValuesFormat`), matching a realistic leaf/block
//! rather than a toy handful of bits.
//!
//! Run with: `cargo bench -p lucene-util`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::zigzag;

const BLOCK_SIZE: usize = 16384;

fn bench_zigzag_roundtrip(c: &mut Criterion) {
    let values: Vec<i64> = (0..BLOCK_SIZE as i64)
        .map(|i| if i % 2 == 0 { i * 31 } else { -(i * 17) })
        .collect();
    c.bench_function("zigzag/encode_decode_block", |b| {
        b.iter(|| {
            let mut sum: i64 = 0;
            for &v in &values {
                let enc = zigzag::encode(black_box(v));
                sum = sum.wrapping_add(zigzag::decode(enc));
            }
            black_box(sum)
        })
    });
}

fn realistic_bitset() -> FixedBitSet {
    // Every third doc "deleted" -- roughly matches the live-docs fixture's
    // deletion density (2 of 5 docs deleted; scaled up here to a realistic
    // block size for a meaningful measurement).
    let mut bs = FixedBitSet::new(BLOCK_SIZE);
    for i in 0..BLOCK_SIZE {
        if i % 3 != 0 {
            bs.set(i);
        }
    }
    bs
}

fn bench_fixed_bit_set_get(c: &mut Criterion) {
    let bs = realistic_bitset();
    c.bench_function("fixed_bit_set/get_all_block", |b| {
        b.iter(|| {
            let mut count = 0usize;
            for i in 0..BLOCK_SIZE {
                if bs.get(black_box(i)) {
                    count += 1;
                }
            }
            black_box(count)
        })
    });
}

fn bench_fixed_bit_set_cardinality(c: &mut Criterion) {
    let bs = realistic_bitset();
    c.bench_function("fixed_bit_set/cardinality_block", |b| {
        b.iter(|| black_box(bs.cardinality()))
    });
}

criterion_group!(
    benches,
    bench_zigzag_roundtrip,
    bench_fixed_bit_set_get,
    bench_fixed_bit_set_cardinality
);
criterion_main!(benches);
