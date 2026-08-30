//! Microbenchmarks for the LZ4 block codec (`org.apache.lucene.util.compress.LZ4`).
//!
//! Three things this guards, all of which the M2 sweep of
//! `crates/lucene-codecs/src/lz4.rs` changed:
//!
//! 1. **Hash-table reuse.** Java's compressors own one `HashTable` for their
//!    whole lifetime and never clear it between inputs; a table allocated per
//!    call shows up here as `compress/fresh_table` vs `compress/reused_table`.
//!    Stored fields compresses one unit per sub-block, so this is a per-8kB
//!    cost, not a per-segment one.
//! 2. **Match copying.** `decompress` copies each match with bulk
//!    `copy_within` runs rather than a bounds-checked byte loop; the
//!    `decompress/*` cases cover a run-length-heavy payload (short match
//!    distances, the worst case for that split) and an ordinary text one.
//! 3. **`HighCompressionHashTable` vs `FastCompressionHashTable`** — the
//!    ratio/speed trade Lucene makes between `CompressionMode.FAST` and the
//!    block-tree terms writer's suffix compression.
//!
//! Run with: `cargo bench -p lucene-codecs --bench lz4_codec`
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use lucene_codecs::lz4::{self, FastCompressionHashTable, HighCompressionHashTable};
use lucene_store::data_input::SliceInput;

/// ~64kB of English-ish text: compressible, but not degenerately so, and big
/// enough to be a realistic BEST_SPEED sub-block-sized workload.
fn text_payload() -> Vec<u8> {
    let mut out = Vec::new();
    let words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "lucene",
        "segment",
        "posting",
        "list",
        "term",
        "frequency",
        "document",
    ];
    let mut i = 0usize;
    while out.len() < 64 * 1024 {
        out.extend_from_slice(words[i % words.len()].as_bytes());
        out.push(b' ');
        i = i.wrapping_mul(31).wrapping_add(7);
    }
    out
}

/// Highly repetitive: long matches at short distances, which is the payload
/// shape where `decompress`'s overlapping-copy path dominates.
fn run_length_payload() -> Vec<u8> {
    let mut out = Vec::new();
    while out.len() < 64 * 1024 {
        out.extend_from_slice(b"aaaaaaaaaaaaaaaabbbbbbbbbbbbbbbb");
    }
    out
}

fn bench_compress(c: &mut Criterion) {
    let payload = text_payload();
    let mut group = c.benchmark_group("lz4/compress");
    group.throughput(Throughput::Bytes(payload.len() as u64));

    let mut ht = FastCompressionHashTable::new();
    group.bench_function("reused_table", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(payload.len());
            lz4::compress_into(black_box(&payload), &mut out, &mut ht);
            black_box(out.len())
        })
    });

    group.bench_function("fresh_table", |b| {
        b.iter(|| black_box(lz4::compress(black_box(&payload)).len()))
    });

    let mut hc = HighCompressionHashTable::new();
    group.bench_function("high_compression", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(payload.len());
            lz4::compress_into(black_box(&payload), &mut out, &mut hc);
            black_box(out.len())
        })
    });

    group.finish();
}

fn bench_compress_with_dictionary(c: &mut Criterion) {
    // The stored-fields BEST_SPEED shape: a dictionary about half a
    // sub-block, then a sub-block compressed against it.
    let payload = text_payload();
    let dict_len = payload.len() / 20;
    let block_len = (payload.len() - dict_len).div_ceil(10);
    let buffer: Vec<u8> = payload[..dict_len + block_len].to_vec();

    let mut group = c.benchmark_group("lz4/compress_with_dictionary");
    group.throughput(Throughput::Bytes(block_len as u64));
    let mut ht = FastCompressionHashTable::new();
    group.bench_function("sub_block", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(block_len);
            lz4::compress_with_dictionary(
                black_box(&buffer),
                0,
                dict_len,
                block_len,
                &mut out,
                &mut ht,
            );
            black_box(out.len())
        })
    });
    group.finish();
}

fn bench_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz4/decompress");
    for (name, payload) in [
        ("text", text_payload()),
        ("run_length", run_length_payload()),
    ] {
        let compressed = lz4::compress(&payload);
        group.throughput(Throughput::Bytes(payload.len() as u64));
        group.bench_function(name, |b| {
            let mut dest = vec![0u8; payload.len()];
            b.iter(|| {
                let mut input = SliceInput::new(black_box(&compressed));
                lz4::decompress(&mut input, payload.len(), &mut dest, 0).unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_compress,
    bench_compress_with_dictionary,
    bench_decompress
);
criterion_main!(benches);
