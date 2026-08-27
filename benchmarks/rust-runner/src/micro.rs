//! Rust side of the component microbenchmarks, matching the Java harnesses
//! under `benchmarks/micro/java/` measurement-for-measurement: same generated
//! inputs, same warmup-then-measure protocol, same `case<TAB>ns_per_op<TAB>ops`
//! TSV on stdout. `scripts/bench-micro.sh` runs both and joins them.
//!
//! Deliberately a plain timed loop rather than `criterion`. The crate's
//! criterion benches stay for tracking Rust-vs-Rust regressions, but criterion's
//! estimate is not the same statistic as a Java timed loop's mean, and this
//! harness exists to produce a number that can be divided by Java's.
//!
//! Built out of `bench-runner`, so it inherits that crate's release profile
//! (fat LTO, one codegen unit) -- the configuration the shipped read path is
//! actually measured in. Measuring the kernel under a different profile than
//! the engine uses would report a speed nothing else can reach.

use std::hint::black_box;
use std::time::{Duration, Instant};

use lucene_codecs::for_util::{self, ForUtil, BLOCK_SIZE};
use lucene_store::data_input::SliceInput;

/// Deterministic values in `[0, 2^bits)`, bit for bit identical to
/// `ForUtilMicro.blockFor` on the Java side. Both harnesses must decode the
/// same bytes or the comparison is between two different workloads.
fn block_for(bits: u32) -> [u32; BLOCK_SIZE] {
    let mut out = [0u32; BLOCK_SIZE];
    let mut state: u32 = 0x9E37_79B9 ^ bits;
    let mask: u32 = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
    for slot in out.iter_mut() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *slot = state & mask;
    }
    out
}

/// Run `op` in batches until `budget` elapses; returns (elapsed, ops).
/// Batched so the clock read is amortized, exactly as the Java loop does.
fn timed_loop(budget: Duration, mut op: impl FnMut()) -> (Duration, u64) {
    const BATCH: u64 = 1024;
    let start = Instant::now();
    let mut ops = 0u64;
    loop {
        for _ in 0..BATCH {
            op();
        }
        ops += BATCH;
        if start.elapsed() >= budget {
            return (start.elapsed(), ops);
        }
    }
}

fn bench_for_decode(warmup: Duration, measure: Duration) {
    // Lucene's `ForUtil` supports `bitsPerValue` 1..=31 only: `decodeSlow`
    // indexes `MASKS32`, which is `new int[32]`, so `bitsPerValue == 32`
    // throws `ArrayIndexOutOfBoundsException` there. This port's `mask32`
    // saturates instead and decodes 32 happily -- being more permissive on the
    // read side is harmless, but there is nothing on the Java side to compare
    // against, so the shared range stops at 31. See `docs/sweep/findings.md`.
    for bits in 1..=31u32 {
        let values = block_for(bits);
        let mut bytes = Vec::new();
        for_util::for_encode(&values, bits, &mut bytes);

        let mut decoded = [0u32; BLOCK_SIZE];
        // One decoder held across every iteration, mirroring the Java harness's
        // single `new ForUtil()` per case. Constructing one per call would
        // charge this side a scratch-buffer zero-fill Lucene never pays, and
        // measure a workload the engine does not run.
        let mut fu = ForUtil::new();

        // Guard the fixture: a decode benchmark over bytes that do not
        // round-trip measures the wrong work and still looks fast.
        {
            let mut r = SliceInput::new(&bytes);
            fu.decode(bits, &mut r, &mut decoded).expect("decode");
            assert_eq!(decoded, values, "round-trip failed at bits={bits}");
        }

        let mut run = |budget| {
            timed_loop(budget, || {
                let mut r = SliceInput::new(black_box(&bytes));
                fu.decode(black_box(bits), &mut r, &mut decoded).unwrap();
                black_box(&decoded[0]);
            })
        };
        run(warmup);
        let (elapsed, ops) = run(measure);
        println!(
            "bits{bits:02}\t{:.3}\t{ops}",
            elapsed.as_nanos() as f64 / ops as f64
        );
    }
}

fn main() {
    let ms = |name: &str, default: u64| -> Duration {
        Duration::from_millis(
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default),
        )
    };
    let warmup = ms("MICRO_WARMUP_MS", 1500);
    let measure = ms("MICRO_MEASURE_MS", 2000);

    let which = std::env::args().nth(1).unwrap_or_else(|| "for_decode".into());
    match which.as_str() {
        "for_decode" => bench_for_decode(warmup, measure),
        other => {
            eprintln!("micro: unknown benchmark {other:?}");
            std::process::exit(2);
        }
    }
}
