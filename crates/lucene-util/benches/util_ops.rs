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
use lucene_util::quantization;
use lucene_util::vector_util;
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

/// Byte-vector similarity kernels (int8 / uint8 quantized HNSW scoring),
/// SIMD dispatch vs the scalar specification, at a typical 768-dim vector.
fn bench_byte_vector_kernels(c: &mut Criterion) {
    let a: Vec<u8> = (0..768u32).map(|i| ((i * 37 + 11) % 251) as u8).collect();
    let b: Vec<u8> = (0..768u32).map(|i| ((i * 91 + 7) % 241) as u8).collect();
    let (ua, ub) = (&a, &b);
    c.bench_function("byte_vectors/dot_i8", |bn| {
        bn.iter(|| black_box(vector_util::dot_product_i8(black_box(&a), black_box(&b))))
    });
    c.bench_function("byte_vectors/dot_i8_scalar", |bn| {
        bn.iter(|| {
            black_box(vector_util::dot_product_i8_scalar(
                black_box(&a),
                black_box(&b),
            ))
        })
    });
    c.bench_function("byte_vectors/square_i8", |bn| {
        bn.iter(|| {
            black_box(vector_util::square_distance_i8(
                black_box(&a),
                black_box(&b),
            ))
        })
    });
    c.bench_function("byte_vectors/square_i8_scalar", |bn| {
        bn.iter(|| {
            black_box(vector_util::square_distance_i8_scalar(
                black_box(&a),
                black_box(&b),
            ))
        })
    });
    c.bench_function("byte_vectors/square_u8", |bn| {
        bn.iter(|| {
            black_box(vector_util::uint8_square_distance(
                black_box(ua),
                black_box(ub),
            ))
        })
    });
    c.bench_function("byte_vectors/square_u8_scalar", |bn| {
        bn.iter(|| {
            black_box(vector_util::uint8_square_distance_scalar(
                black_box(ua),
                black_box(ub),
            ))
        })
    });
}

/// `OffHeapScalarQuantizedVectorValues`' uint4 nibble unpack (Lucene 10.6's
/// `VectorUtil.int4Unpack`, #16631) of a 768-dim `PackedNibble` vector: the
/// per-vector decode a 4-bit quantized merge and exact search pay.
fn bench_unpack_nibbles(c: &mut Criterion) {
    let packed: Vec<u8> = (0..384u32).map(|i| ((i * 37 + 11) % 256) as u8).collect();
    let mut unpacked = vec![0u8; 768];
    c.bench_function("quantization/unpack_nibbles_768", |b| {
        b.iter(|| {
            quantization::unpack_nibbles(black_box(&packed), &mut unpacked);
            black_box(&unpacked);
        })
    });
}

/// Range combinators over a 1M-bit set at an unaligned offset: `apply_mask`
/// (live docs applied to a doc-id slice) and `for_each_in_range`.
fn bench_fixed_bit_set_ranges(c: &mut Criterion) {
    const N: usize = 1 << 20;
    let mut live = FixedBitSet::new(N);
    for i in 0..N {
        if i % 13 != 0 {
            live.set(i);
        }
    }
    let mut dest = FixedBitSet::new(N / 2);
    for i in 0..dest.len() {
        if i % 3 == 0 {
            dest.set(i);
        }
    }
    c.bench_function("fixed_bit_set/apply_mask_512k", |b| {
        b.iter(|| {
            let mut d = dest.clone();
            live.apply_mask(&mut d, 1001).unwrap();
            black_box(d)
        })
    });
    c.bench_function("fixed_bit_set/for_each_in_range_512k", |b| {
        b.iter(|| {
            let mut sum = 0i64;
            dest.for_each_in_range(17, dest.len() - 5, 0, |d| sum += i64::from(d));
            black_box(sum)
        })
    });
}

criterion_group!(
    benches,
    bench_fixed_bit_set_ranges,
    bench_byte_vector_kernels,
    bench_unpack_nibbles,
    bench_zigzag_roundtrip,
    bench_fixed_bit_set_get,
    bench_fixed_bit_set_cardinality
);
criterion_main!(benches);
