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

use lucene_codecs::direct_reader;
use lucene_codecs::for_util::{self, ForUtil, BLOCK_SIZE};
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::data_input::SliceInput;
use lucene_store::MmapDirectory;

/// Deterministic values in `[0, 2^bits)`, bit for bit identical to
/// `ForUtilMicro.blockFor` on the Java side. Both harnesses must decode the
/// same bytes or the comparison is between two different workloads.
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

/// Run `op` in adaptively-sized batches until `budget` elapses; returns
/// (elapsed, ops).
///
/// Batched so the clock read is amortized, but the batch **grows from one**
/// rather than being fixed. A fixed batch must be sized for the cheapest case,
/// and this harness spans nanoseconds (`for_decode`) to hundreds of
/// milliseconds (`reader_open`): a hard-coded 1024 meant one batch of reader
/// opens ran for twenty minutes past its budget before the clock was consulted
/// at all. Doubling until a batch takes a measurable slice of the budget keeps
/// the clock overhead negligible for fast operations without overshooting slow
/// ones.
fn timed_loop(budget: Duration, mut op: impl FnMut()) -> (Duration, u64) {
    let start = Instant::now();
    let mut ops = 0u64;
    let mut batch = 1u64;
    loop {
        let batch_start = Instant::now();
        for _ in 0..batch {
            op();
        }
        ops += batch;
        let elapsed = start.elapsed();
        if elapsed >= budget {
            return (elapsed, ops);
        }
        // Grow only while a whole batch is still short next to the budget, so
        // the clock is read a bounded number of times either way.
        if batch_start.elapsed() * 64 < budget {
            batch = batch.saturating_mul(2);
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
        // `for_encode` packs in place and consumes its input, as
        // `ForUtil.encode(int[], ...)` does -- so the fixture is encoded from a
        // scratch copy and `values` stays the pristine expectation the
        // round-trip guard below compares against.
        let mut scratch = values;
        for_util::for_encode(&mut scratch, bits, &mut bytes);

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

/// Walk a whole posting list with `next_doc()`, the operation M1 is actually
/// about. The Java counterpart is `PostingsIterMicro`, which drives
/// `Lucene104PostingsReader`'s `BlockPostingsEnum` through the public
/// `TermsEnum.postings()` API over the same index directory and the same terms,
/// so both sides walk identical on-disk bytes.
///
/// Reports ns per *document* rather than per block: block counts differ between
/// terms and a per-block number would not be comparable across the cases.
fn bench_postings_iter(warmup: Duration, measure: Duration, index: &str) {
    let dir = MmapDirectory::open(index.to_string());
    let reader = DirectoryReader::open(&dir).expect("open index");
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();

    // Zipf-ranked vocabulary, so this spans three orders of magnitude of
    // posting-list length: t0 is the most frequent term, t2s is well down the
    // tail. A single term would measure one block-encoding shape.
    for term in ["t0", "t1", "tz", "t2s"] {
        let mut total_docs = 0u64;
        let mut run = |budget: Duration| {
            timed_loop(budget, || {
                let mut n = 0u64;
                for seg in segments.iter() {
                    let Some(field) = seg.fields.field("body") else {
                        continue;
                    };
                    let Some(doc_in) = seg.doc_in else { continue };
                    let Ok(Some(mut cursor)) = field.lazy_postings(term.as_bytes(), doc_in) else {
                        continue;
                    };
                    loop {
                        let doc = cursor.next_doc().expect("next_doc");
                        if doc == i32::MAX {
                            break;
                        }
                        n += 1;
                        black_box(doc);
                    }
                }
                total_docs = n;
            })
        };
        run(warmup);
        let (elapsed, iters) = run(measure);
        if total_docs == 0 {
            eprintln!("micro: term {term:?} has no postings in this index; skipping");
            continue;
        }
        println!(
            "{term}\t{:.3}\t{}",
            elapsed.as_nanos() as f64 / (iters * total_docs) as f64,
            iters * total_docs
        );
    }
}

/// `DirectReader.get(index)` -- the per-value read behind doc values and
/// monotonic sequences. The Java counterpart is `DirectReaderMicro`, driving
/// `DirectReader.getInstance(RandomAccessInput, bitsPerValue)` over the same
/// bit-packed bytes.
///
/// Reads a fixed stride through a 1 MiB packed array rather than sequentially:
/// this primitive exists to serve random per-document lookups, and a
/// sequential sweep would measure the prefetcher instead.
fn bench_direct_reader(warmup: Duration, measure: Duration) {
    // Every width `DirectWriter` supports.
    for bits in [1u8, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64] {
        let count = 1 << 17;
        let mask: i64 = if bits >= 64 { -1 } else { (1i64 << bits) - 1 };
        let mut state: u64 = 0x243F_6A88_85A3_08D3 ^ bits as u64;
        let values: Vec<i64> = (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as i64) & mask
            })
            .collect();
        let mut packed = direct_reader::encode(&values, bits);
        // `DirectWriter` pads its output so a reader may always load a whole
        // word; without it the tail elements take the slow path and the
        // measurement is of the padding, not the read.
        packed.resize(packed.len() + 8, 0);

        // Guard the fixture, as the other cases do.
        for i in [0usize, 1, count / 2, count - 1] {
            assert_eq!(
                direct_reader::get(&packed, bits, i as i64).unwrap(),
                values[i],
                "round-trip failed at bits={bits} i={i}"
            );
        }

        // An odd stride, so consecutive reads land in different cache lines
        // without ever repeating a value.
        const STRIDE: usize = 4099;
        let run = |budget| {
            let mut i = 0usize;
            timed_loop(budget, || {
                i = (i + STRIDE) & (count - 1);
                black_box(direct_reader::get(black_box(&packed), bits, i as i64).unwrap());
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

/// Opening a reader: `DirectoryReader::open` plus `open_segments`, against
/// Java's `DirectoryReader.open`.
///
/// This is not a query benchmark, and it is here because of an architectural
/// difference the query benchmarks cannot see. `blocktree::FieldTerms` holds
/// `Vec<(Vec<u8>, TermStats, TermMetadata)>` -- **every term in the field**,
/// each with its own allocation -- built when the segment is opened. Lucene's
/// `SegmentTermsEnum` holds none of that: it walks the `.tip` FST to a block,
/// scans that block's suffix bytes in place, and decodes metadata only for the
/// term actually sought. So this port pays O(vocabulary) time and memory per
/// open where Lucene pays O(1), and a search benchmark never shows it because
/// the reader is opened once, outside the timed region.
///
/// It matters anyway: a search engine reopens readers on every refresh.
fn bench_reader_open(warmup: Duration, measure: Duration, index: &str) {
    let dir = MmapDirectory::open(index.to_string());
    let run = |budget| {
        timed_loop(budget, || {
            let reader = DirectoryReader::open(&dir).expect("open index");
            let opened = reader.open_segments().expect("open segments");
            black_box(opened.as_open_segments().len());
        })
    };
    run(warmup);
    let (elapsed, ops) = run(measure);
    println!("open\t{:.3}\t{ops}", elapsed.as_nanos() as f64 / ops as f64);
}

/// Fetching stored fields for a document -- `StoredFields.document(docId)`,
/// which every real search does once per returned hit and which this project
/// had never compared against Lucene.
///
/// Reads a fixed odd stride through the segment so consecutive fetches land in
/// different compressed blocks, which is what a top-k result set looks like.
/// Sequential fetching would measure the block cache instead.
fn bench_stored_fields(warmup: Duration, measure: Duration, index: &str) {
    let dir = MmapDirectory::open(index.to_string());
    let reader = DirectoryReader::open(&dir).expect("open index");
    let seg = &reader.segment_readers()[0];
    let name = &seg.segment_name;

    let read = |ext: &str| -> Vec<u8> {
        let path = format!("{index}/{name}{ext}");
        std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
    };
    let (fdt, fdx, fdm) = (read(".fdt"), read(".fdx"), read(".fdm"));
    let sr = lucene_codecs::stored_fields::open(&fdt, &fdx, &fdm, &seg.segment_id(), "")
        .expect("open stored fields");

    let max_doc = seg.max_doc;
    const STRIDE: i32 = 4099;
    // Guard the fixture: a benchmark over a reader that returns nothing would
    // look extremely fast. `GenCorpus` currently indexes every field
    // `Store.NO`, so the M1 corpus has a 66 KB `.fdt` for 5M documents and
    // cannot exercise this at all -- see `docs/sweep/findings.md`.
    if sr.document(0).expect("document 0").fields.is_empty() {
        eprintln!(
            "micro: this index stores no fields, so stored-field retrieval cannot be \
             measured against it -- regenerate the corpus with a stored field first"
        );
        return;
    }

    let mut doc = 0i32;
    let mut run = |budget| {
        timed_loop(budget, || {
            doc = (doc + STRIDE) % max_doc;
            black_box(sr.document(doc).expect("document").fields.len());
        })
    };
    run(warmup);
    let (elapsed, ops) = run(measure);
    println!(
        "document\t{:.3}\t{ops}",
        elapsed.as_nanos() as f64 / ops as f64
    );
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

    let which = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "for_decode".into());
    match which.as_str() {
        "for_decode" => bench_for_decode(warmup, measure),
        "direct_reader" => bench_direct_reader(warmup, measure),
        "stored_fields" => {
            let index = std::env::args()
                .nth(2)
                .expect("stored_fields needs an index directory");
            bench_stored_fields(warmup, measure, &index);
        }
        "reader_open" => {
            let index = std::env::args()
                .nth(2)
                .expect("reader_open needs an index directory");
            bench_reader_open(warmup, measure, &index);
        }
        "postings_iter" => {
            let index = std::env::args()
                .nth(2)
                .expect("postings_iter needs an index directory");
            bench_postings_iter(warmup, measure, &index);
        }
        other => {
            eprintln!("micro: unknown benchmark {other:?}");
            std::process::exit(2);
        }
    }
}
