//! Microbenchmark for the innermost decode kernel in the engine:
//! `ForUtil.decode` / `PForUtil.decode` (`for_util::for_decode`/`pfor_decode`),
//! which unpacks one 256-value block of doc deltas or freqs.
//!
//! This exists to be compared against Lucene's own number for the same
//! operation, not read in isolation. The Java counterpart is
//! `benchmarks/micro/src/org/apache/lucene/codecs/lucene104/ForUtilMicro.java`,
//! which drives `PostingIndexInput.decode` -- the wrapper Lucene made public
//! *specifically* so posting decode can be benchmarked from outside the jar --
//! with `--add-modules jdk.incubator.vector` so the Panama
//! `MemorySegmentPostingDecodingUtil` path is live. `scripts/bench-micro.sh`
//! runs both and joins them.
//!
//! Both sides decode the same block shapes: 256 values drawn deterministically
//! from `[0, 2^bits)` so the packing is exact and no `PForUtil` exception
//! patching is involved. Timings are per 256-value block; divide by 256 for
//! per-value.
//!
//! This bench covers `bitsPerValue` `1..=32`; the cross-engine one stops at 31.
//! Lucene's `ForUtil.decodeSlow` indexes `MASKS32`, declared `new int[32]`, so
//! 32 bits per value throws there and has no Java number to be divided by. It
//! is kept here because this port's decoder does accept it, and an untested
//! branch in the innermost decode loop is worse than an unpaired one.
//!
//! Run with: `cargo bench -p lucene-codecs --bench for_util_decode`

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use lucene_codecs::for_util::{self, ForUtil, BLOCK_SIZE};
use lucene_store::data_input::SliceInput;

/// Deterministic values in `[0, 2^bits)`. A xorshift rather than a counter:
/// consecutive-integer input would let a bit-packer's masks fold into
/// constants in a way real doc deltas never do.
fn block_for(bits: u32) -> [u32; BLOCK_SIZE] {
    let mut out = [0u32; BLOCK_SIZE];
    let mut state: u32 = 0x9E37_79B9 ^ bits;
    let mask: u32 = if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    for slot in out.iter_mut() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *slot = state & mask;
    }
    out
}

fn encoded(bits: u32) -> Vec<u8> {
    let values = block_for(bits);
    let mut buf = Vec::new();
    for_util::for_encode(&values, bits, &mut buf);
    buf
}

fn bench_for_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("for_util/for_decode");
    group.throughput(Throughput::Elements(BLOCK_SIZE as u64));
    for bits in 1..=32u32 {
        let bytes = encoded(bits);
        // Guard the fixture itself: a decode benchmark over bytes that do not
        // round-trip would measure the wrong work and still look fast.
        {
            let mut out = [0u32; BLOCK_SIZE];
            let mut r = SliceInput::new(&bytes);
            ForUtil::new()
                .decode(bits, &mut r, &mut out)
                .expect("decode");
            assert_eq!(out, block_for(bits), "round-trip failed at bits={bits}");
        }
        group.bench_function(format!("bits{bits:02}"), |b| {
            let mut out = [0u32; BLOCK_SIZE];
            // Held across iterations, as a real caller decoding a posting list
            // block after block does -- and as Lucene's own `ForUtil` instance
            // is.
            let mut fu = ForUtil::new();
            b.iter(|| {
                let mut r = SliceInput::new(black_box(&bytes));
                fu.decode(black_box(bits), &mut r, &mut out).unwrap();
                black_box(&out[0]);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_for_decode);
criterion_main!(benches);
