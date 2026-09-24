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
        // Shaped as Java's `walk`: a reader built once (`getInstance`), and
        // 4096 reads per clock check summed into a local, so the sum stays in
        // a register. A per-read `black_box`, or a sink captured by the
        // closure (which the clock calls keep in memory), put a store-forward
        // chain under every read: a flat 1.3 ns floor across all fourteen
        // widths, twice Java's whole one-bit read.
        const INNER: u64 = 4096;
        // The walk is compiled per width (`with_width`), as each of the Java
        // side's per-width JVMs sees one `DirectPackedReaderNN` class.
        struct Walk {
            budget: Duration,
            mask: usize,
        }
        impl direct_reader::WidthVisitor for Walk {
            type Output = (Duration, u64);
            fn visit<const B: u32>(self, reader: direct_reader::FixedWidthReader<'_, B>) -> Self::Output {
                let mut i = 0usize;
                let (elapsed, calls) = timed_loop(self.budget, || {
                    // Locals, not the captures: through a reference they stay
                    // in memory, and the store-to-load chain is the floor.
                    let (mut j, mut sink) = (i, 0i64);
                    for _ in 0..INNER {
                        j = (j + STRIDE) & self.mask;
                        sink = sink.wrapping_add(reader.get(j as i64).unwrap());
                    }
                    i = j;
                    black_box(sink);
                });
                (elapsed, calls * INNER)
            }
        }
        let run = |budget| {
            let reader = direct_reader::DirectReader::new(black_box(&packed[..]), bits).unwrap();
            reader.with_width(Walk {
                budget,
                mask: count - 1,
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

// ---------------------------------------------------------------------------
// The per-area sweep. Every case below has a same-named case in
// `benchmarks/micro/java/SweepMicro.java`, over the same generated input or the
// same corpus directory. Each timed call does a *batch* of work and the
// reported figure is ns per unit of that work, on both sides.
// ---------------------------------------------------------------------------

mod counting_alloc {
    //! A counting global allocator, so memory cases can report the heap an
    //! operation leaves resident -- the Rust counterpart of Java's used-heap
    //! delta after GC. The mmap'd index files are outside both figures.
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicIsize, Ordering};

    pub struct Counting;
    pub static LIVE: AtomicIsize = AtomicIsize::new(0);

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            LIVE.fetch_add(l.size() as isize, Ordering::Relaxed);
            unsafe { System.alloc(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            LIVE.fetch_sub(l.size() as isize, Ordering::Relaxed);
            unsafe { System.dealloc(p, l) }
        }
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            LIVE.fetch_add(l.size() as isize, Ordering::Relaxed);
            unsafe { System.alloc_zeroed(l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
            LIVE.fetch_add(new as isize - l.size() as isize, Ordering::Relaxed);
            unsafe { System.realloc(p, l, new) }
        }
    }

    pub fn live() -> isize {
        LIVE.load(Ordering::Relaxed)
    }
}

#[global_allocator]
static ALLOC: counting_alloc::Counting = counting_alloc::Counting;

/// Warmup, then measure; prints `name<TAB>ns_per_unit<TAB>units`. `op` does one
/// batch and returns how many units of work it did.
fn measure(name: &str, warmup: Duration, budget: Duration, mut op: impl FnMut() -> u64) {
    // `MICRO_CASE=<name>` runs one case alone, so a profile of it is not
    // averaged with its siblings'.
    if std::env::var("MICRO_CASE").is_ok_and(|only| only != name) {
        return;
    }
    let mut run = |b: Duration| {
        let start = Instant::now();
        let mut units = 0u64;
        loop {
            units += op();
            let e = start.elapsed();
            if e >= b {
                return (e, units);
            }
        }
    };
    run(warmup);
    let (elapsed, units) = run(budget);
    println!(
        "{name}\t{:.3}\t{units}",
        elapsed.as_nanos() as f64 / units as f64
    );
}

/// xorshift64, identical to `SweepMicro.Rng`.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// `PForUtilMicro.block`, bit for bit: 256 values under `2^bits`, three of them
/// patched with 8 extra bits so every block takes the exception path.
fn pfor_block(bits: u32) -> [u32; BLOCK_SIZE] {
    let mut out = [0u32; BLOCK_SIZE];
    let mut state: u32 = 0x51ED_270B ^ bits;
    let mask = (1u32 << bits) - 1;
    for slot in out.iter_mut() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *slot = state & mask;
    }
    for slot in [17usize, 101, 230] {
        out[slot] = (out[slot] & mask) | (0xA5 << bits);
    }
    out
}

fn bench_pfor_decode(w: Duration, m: Duration) {
    for bits in 1..=23u32 {
        let values = pfor_block(bits);
        let mut bytes = Vec::new();
        let mut scratch = values;
        for_util::pfor_encode(&mut scratch, &mut bytes);
        let mut fu = ForUtil::new();
        let mut decoded = [0u32; BLOCK_SIZE];
        fu.pfor_decode(&mut SliceInput::new(&bytes), &mut decoded)
            .unwrap();
        assert_eq!(decoded, values, "pfor round-trip failed at bits={bits}");
        measure(&format!("bits{bits:02}"), w, m, || {
            for _ in 0..1024 {
                let mut r = SliceInput::new(black_box(&bytes));
                fu.pfor_decode(&mut r, &mut decoded).unwrap();
                black_box(&decoded[0]);
            }
            1024
        });
    }
}

fn bench_vint(w: Duration, m: Duration) {
    use lucene_store::data_input::DataInput;
    use lucene_store::data_output::DataOutput;
    const N: usize = 1 << 20;
    let mut r = Rng(0x1234_5678_9ABC_DEF1);
    let ints: Vec<i32> = (0..N)
        .map(|_| {
            let x = r.next();
            (((x >> 33) as u32) >> ((x & 0xFFFF) % 31)) as i32
        })
        .collect();
    let mut r = Rng(0x0FED_CBA9_8765_4321);
    let longs: Vec<i64> = (0..N)
        .map(|_| {
            let x = r.next();
            ((x >> 1) >> ((x & 0xFFFF) % 63)) as i64
        })
        .collect();
    let mut vint = Vec::new();
    for &x in &ints {
        vint.write_vint(x);
    }
    let mut vlong = Vec::new();
    for &x in &longs {
        vlong.write_vlong(x);
    }
    let mut group = Vec::new();
    for g in ints.chunks(128) {
        let g: Vec<u32> = g.iter().map(|&x| x as u32).collect();
        group.write_group_vints(&g);
    }
    {
        let mut inp = SliceInput::new(&vint);
        for (i, &x) in ints.iter().enumerate() {
            assert_eq!(inp.read_vint().unwrap(), x, "vint {i}");
        }
        let mut inp = SliceInput::new(&group);
        let mut dst = [0u64; 128];
        for g in ints.chunks(128) {
            inp.read_group_vints(&mut dst).unwrap();
            for (a, b) in g.iter().zip(&dst) {
                assert_eq!(*a as u32 as u64, *b);
            }
        }
    }
    measure("vint", w, m, || {
        let mut inp = SliceInput::new(black_box(&vint));
        let mut acc = 0i32;
        for _ in 0..N {
            acc = acc.wrapping_add(inp.read_vint().unwrap());
        }
        black_box(acc);
        N as u64
    });
    measure("vlong", w, m, || {
        let mut inp = SliceInput::new(black_box(&vlong));
        let mut acc = 0i64;
        for _ in 0..N {
            acc = acc.wrapping_add(inp.read_vlong().unwrap());
        }
        black_box(acc);
        N as u64
    });
    let mut dst = [0u64; 128];
    measure("group_vint", w, m, || {
        let mut inp = SliceInput::new(black_box(&group));
        for _ in 0..N / 128 {
            inp.read_group_vints(&mut dst).unwrap();
            black_box(dst[127]);
        }
        N as u64
    });
}

fn random_bits(
    num_bits: usize,
    seed: u64,
    density_pct: u64,
) -> lucene_util::fixed_bit_set::FixedBitSet {
    let mut b = lucene_util::fixed_bit_set::FixedBitSet::new(num_bits);
    let mut r = Rng(seed);
    for i in 0..num_bits {
        if r.next() % 100 < density_pct {
            b.set(i);
        }
    }
    b
}

fn bench_bitset(w: Duration, m: Duration) {
    use lucene_util::fixed_bit_set::FixedBitSet;
    const NUM_BITS: usize = 1 << 22;
    let a = random_bits(NUM_BITS, 0x1111_2222_3333_4444, 10);
    let b = random_bits(NUM_BITS, 0x5555_6666_7777_8888, 10);
    let set_bits = a.cardinality() as u64;
    measure("cardinality", w, m, || {
        black_box(black_box(&a).cardinality());
        (NUM_BITS / 64) as u64
    });
    measure("next_set_bit", w, m, || {
        let a = black_box(&a);
        let mut n = 0u64;
        let mut i = a.next_set_bit(0);
        while let Some(b) = i {
            n += 1;
            i = a.next_set_bit(b + 1);
        }
        black_box(n);
        set_bits
    });
    measure("intersection_count", w, m, || {
        black_box(FixedBitSet::intersection_count(
            black_box(&a),
            black_box(&b),
        ));
        (NUM_BITS / 64) as u64
    });
    let mut c = a.clone();
    measure("or", w, m, || {
        c.or(black_box(&b));
        (NUM_BITS / 64) as u64
    });
    let mut r = Rng(0x9999_AAAA_BBBB_CCCC);
    let probes: Vec<usize> = (0..1 << 16)
        .map(|_| (r.next() % NUM_BITS as u64) as usize)
        .collect();
    measure("get_random", w, m, || {
        let a = black_box(&a);
        let n = probes.iter().filter(|&&p| a.get(p)).count();
        black_box(n);
        probes.len() as u64
    });
}

fn text_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed);
    let mut s = String::with_capacity(len + 16);
    while s.len() < len {
        if !s.is_empty() {
            s.push(' ');
        }
        s.push('t');
        s.push_str(&lucene_util::base36::to_base36((r.next() % 2000) as i64));
    }
    s.truncate(len);
    s.into_bytes()
}

fn bench_lz4(w: Duration, m: Duration) {
    use lucene_codecs::lz4;
    for len in [16 * 1024usize, 60 * 1024] {
        let src = text_bytes(len, 0xABCD_EF01_2345_6789 ^ len as u64);
        let sz = format!("{}k", len / 1024);
        let mut compressed = Vec::new();
        lz4::compress_into(
            &src,
            &mut compressed,
            &mut lz4::FastCompressionHashTable::new(),
        );
        let mut dst = vec![0u8; len];
        lz4::decompress_slice(&mut SliceInput::new(&compressed), len, &mut dst, 0).unwrap();
        assert_eq!(src, dst, "lz4 round trip");
        measure(&format!("decompress_{sz}"), w, m, || {
            let mut inp = SliceInput::new(black_box(&compressed));
            lz4::decompress_slice(&mut inp, len, &mut dst, 0).unwrap();
            black_box(dst[len - 1]);
            len as u64
        });
        let mut fast = lz4::FastCompressionHashTable::new();
        let mut out = Vec::with_capacity(len);
        measure(&format!("compress_fast_{sz}"), w, m, || {
            out.clear();
            lz4::compress_into(black_box(&src), &mut out, &mut fast);
            black_box(out.len());
            len as u64
        });
        let mut high = lz4::HighCompressionHashTable::new();
        measure(&format!("compress_high_{sz}"), w, m, || {
            out.clear();
            lz4::compress_into(black_box(&src), &mut out, &mut high);
            black_box(out.len());
            len as u64
        });
    }
}

fn bench_direct_monotonic(w: Duration, m: Duration) {
    use lucene_codecs::direct_monotonic;
    const N: usize = 1 << 20;
    let block_shift = 16;
    let mut r = Rng(0x7777_1234_ABCD_0001);
    let mut acc = 0i64;
    let values: Vec<i64> = (0..N)
        .map(|_| {
            acc += (r.next() % 1000) as i64;
            acc
        })
        .collect();
    let (meta_bytes, mut data) = direct_monotonic::write(&values, block_shift);
    data.resize(data.len() + 8, 0);
    let meta =
        direct_monotonic::load_meta(&mut SliceInput::new(&meta_bytes), N as i64, block_shift)
            .unwrap();
    for i in [0usize, 1, N / 2, N - 1] {
        assert_eq!(
            direct_monotonic::get(&data, &meta, i as i64).unwrap(),
            values[i]
        );
    }
    const STRIDE: usize = 4099;
    measure("get_random", w, m, || {
        let mut i = 0usize;
        let mut s = 0i64;
        for _ in 0..4096 {
            i = (i + STRIDE) & (N - 1);
            s = s.wrapping_add(direct_monotonic::get(black_box(&data), &meta, i as i64).unwrap());
        }
        black_box(s);
        4096
    });
    measure("get_seq", w, m, || {
        let mut s = 0i64;
        for i in 0..N {
            s = s.wrapping_add(direct_monotonic::get(black_box(&data), &meta, i as i64).unwrap());
        }
        black_box(s);
        N as u64
    });
}

fn bench_checksum(w: Duration, m: Duration) {
    let len = 16usize << 20;
    let data = text_bytes(len, 0x5151_5151_5151_5151);
    measure("crc32_16m", w, m, || {
        black_box(crc32fast::hash(black_box(&data)));
        len as u64
    });
}

fn analysis_docs() -> Vec<String> {
    let mut r = Rng(0x2468_ACE0_1357_9BDF);
    let mut docs = Vec::new();
    for _ in 0..2000 {
        let words = 40 + (r.next() % 120) as usize;
        let mut s = String::new();
        for wi in 0..words {
            if wi > 0 {
                s.push(' ');
            }
            let x = r.next();
            let a = x % 50000;
            let b = (x >> 20) % 50000;
            let mut word = format!("t{}", lucene_util::base36::to_base36(a.min(b) as i64));
            if x & 7 == 0 {
                word = word.to_uppercase();
            }
            s.push_str(&word);
            if x & 31 == 1 {
                s.push(',');
            }
            if x & 63 == 2 {
                s.push('.');
            }
        }
        docs.push(s);
    }
    docs
}

fn bench_analysis(w: Duration, m: Duration) {
    let docs = analysis_docs();
    let analyzer = lucene_analysis::Analyzer::standard(None);
    measure("standard", w, m, || {
        let mut tokens = 0u64;
        for text in &docs {
            for t in analyzer.analyze(black_box(text)) {
                tokens += 1;
                black_box(t.term.len());
            }
        }
        tokens
    });
}

/// `SweepMicro.floatVectors`, bit for bit.
fn float_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut r = Rng(seed);
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| ((r.next() >> 40) as f64 / (1u64 << 24) as f64) as f32 - 0.5)
                .collect()
        })
        .collect()
}

fn byte_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut r = Rng(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| (r.next() >> 56) as u8).collect())
        .collect()
}

/// The similarity kernels vector search spends its time in -- see
/// `SweepMicro.vectors`.
fn bench_vectors(w: Duration, m: Duration) {
    use lucene_codecs::vectors;
    for dim in [128usize, 768] {
        let docs = float_vectors(1024, dim, 0xF00D + dim as u64);
        let q = &float_vectors(1, dim, 0xBEEF + dim as u64)[0];
        measure(&format!("dot_f32_{dim}"), w, m, || {
            let mut s = 0.0f32;
            for d in &docs {
                s += vectors::dot_product(black_box(q), d);
            }
            black_box(s);
            docs.len() as u64
        });
        measure(&format!("l2_f32_{dim}"), w, m, || {
            let mut s = 0.0f32;
            for d in &docs {
                s += vectors::square_distance(black_box(q), d);
            }
            black_box(s);
            docs.len() as u64
        });
        measure(&format!("cos_f32_{dim}"), w, m, || {
            let mut s = 0.0f32;
            for d in &docs {
                s += vectors::cosine(black_box(q), d);
            }
            black_box(s);
            docs.len() as u64
        });
        let bdocs = byte_vectors(1024, dim, 0xB17E + dim as u64);
        let bq = &byte_vectors(1, dim, 0xB0B + dim as u64)[0];
        measure(&format!("dot_u8_{dim}"), w, m, || {
            let mut s = 0i64;
            for d in &bdocs {
                s += vectors::dot_product_bytes(black_box(bq), d) as i64;
            }
            black_box(s);
            bdocs.len() as u64
        });
    }
}

/// Opens the corpus and hands the first segment's pieces to `f`.
fn with_segment(
    index: &str,
    f: impl FnOnce(
        &lucene_search::directory_reader::SegmentReader,
        &lucene_search::multi_segment::OpenSegment<'_>,
    ),
) {
    let dir = MmapDirectory::open(index.to_string());
    let reader = DirectoryReader::open(&dir).expect("open index");
    let opened = reader.open_segments().expect("open segments");
    let segs = opened.as_open_segments();
    f(&reader.segment_readers()[0], &segs[0]);
}

fn bench_postings_adv(w: Duration, m: Duration, index: &str) {
    with_segment(index, |_, seg| {
        let field = seg.fields.field("body").expect("body");
        let doc_in = seg.doc_in.expect("doc_in");
        for term in ["t0", "t1", "tz"] {
            // Sought once, outside the timed loop, as Java's `te.seekExact`
            // is; each iteration opens a fresh enum from the term state, as
            // its `te.postings(null, NONE)` does.
            let seeked = field
                .seek_term_state(term.as_bytes())
                .unwrap()
                .expect("term");
            for gap in [8i32, 64, 1024] {
                measure(&format!("{term}_gap{gap}"), w, m, || {
                    let mut c = field
                        .lazy_postings_for(
                            &seeked,
                            doc_in,
                            lucene_codecs::postings::PostingsFlags::DocsOnly,
                        )
                        .unwrap();
                    let mut n = 0u64;
                    let mut doc = c.advance(0).unwrap();
                    while doc != lucene_codecs::postings::NO_MORE_DOCS {
                        n += 1;
                        doc = c.advance(doc + gap).unwrap();
                    }
                    black_box(n);
                    n
                });
            }
        }
    });
}

fn bench_postings_freq(w: Duration, m: Duration, index: &str) {
    with_segment(index, |_, seg| {
        let field = seg.fields.field("body").expect("body");
        let doc_in = seg.doc_in.expect("doc_in");
        for term in ["t0", "t1", "tz", "t2s"] {
            measure(term, w, m, || {
                let mut c = field
                    .lazy_postings_with_flags(
                        term.as_bytes(),
                        doc_in,
                        lucene_codecs::postings::PostingsFlags::Freqs,
                    )
                    .unwrap()
                    .expect("term");
                let (mut n, mut f) = (0u64, 0u64);
                while c.next_doc().unwrap() != lucene_codecs::postings::NO_MORE_DOCS {
                    n += 1;
                    f += c.freq().unwrap_or(1) as u64;
                }
                black_box(f);
                n
            });
        }
    });
}

/// Every position of every document, through the lazy `PositionsCursor` the
/// phrase path reads with -- `PostingsEnum.POSITIONS` on the Java side.
fn bench_positions(w: Duration, m: Duration, index: &str) {
    with_segment(index, |_, seg| {
        let field = seg.fields.field("body").expect("body");
        let doc_in = seg.doc_in.expect("doc_in");
        let pos_in = seg.pos_in.expect("pos_in");
        for term in ["t1", "tz", "t2s"] {
            measure(term, w, m, || {
                let mut c = field
                    .lazy_positions(term.as_bytes(), doc_in, pos_in)
                    .unwrap()
                    .expect("term");
                let (mut n, mut s) = (0u64, 0i64);
                while c.next_doc().unwrap() != lucene_codecs::postings::NO_MORE_DOCS {
                    let f = c.freq();
                    for _ in 0..f {
                        s = s.wrapping_add(c.next_position().unwrap() as i64);
                    }
                    n += f as u64;
                }
                black_box(s);
                n
            });
        }
    });
}

fn bench_term_seek(w: Duration, m: Duration, index: &str) {
    with_segment(index, |_, seg| {
        let field = seg.fields.field("body").expect("body");
        let mut terms: Vec<Vec<u8>> = Vec::new();
        let mut it = field.iter();
        let mut n = 0usize;
        while let Some((t, _)) = it.next() {
            if n % 97 == 0 {
                terms.push(t.to_vec());
            }
            n += 1;
        }
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for i in (1..terms.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let j = (state % (i as u64 + 1)) as usize;
            terms.swap(i, j);
        }
        terms.truncate(2000);
        let misses: Vec<Vec<u8>> = terms
            .iter()
            .map(|t| {
                let mut v = t.clone();
                v.push(b'~');
                v
            })
            .collect();
        for (name, targets) in [("seek_hit", &terms), ("seek_miss", &misses)] {
            measure(name, w, m, || {
                let mut acc = 0i64;
                for t in targets {
                    acc += field
                        .seek_exact(black_box(t))
                        .map_or(0, |s| s.doc_freq as i64);
                }
                black_box(acc);
                targets.len() as u64
            });
        }
        measure("next_all", w, m, || {
            let mut it = field.iter();
            let mut ops = 0u64;
            let mut acc = 0u64;
            while let Some((t, s)) = it.next() {
                acc += t.len() as u64 + s.doc_freq as u64;
                ops += 1;
            }
            black_box(acc);
            ops
        });
    });
}

fn bench_doc_values(w: Duration, m: Duration, index: &str) {
    use lucene_codecs::doc_values::{self, SortedSetKind};
    with_segment(index, |r, _| {
        let max_doc = r.max_doc;
        let num = r.field_infos().field_by_name("num").expect("num").number;
        let kw = r
            .field_infos()
            .field_by_name("keyword")
            .expect("keyword")
            .number;
        let cat = r.field_infos().field_by_name("cat").expect("cat").number;
        let (meta, data) = r.doc_values_for_field(num).expect("dv");
        let num_entry = meta.numeric_entry(num).expect("num entry");
        // The `_seq` cases use each engine's sequential API: Java's `nextDoc`
        // loop, and `for_each_value` here, which is what this port's range and
        // sort consumers call. `numeric_stride37` is the random-access one.
        measure("numeric_seq", w, m, || {
            let mut v = doc_values::NumericReader::new(data, num_entry);
            let mut s = 0i64;
            v.for_each_value(0, max_doc, |_, x| s = s.wrapping_add(x))
                .unwrap();
            black_box(s);
            max_doc as u64
        });
        measure("numeric_stride37", w, m, || {
            let mut v = doc_values::NumericReader::new(data, num_entry);
            let (mut s, mut n) = (0i64, 0u64);
            let mut d = 0;
            while d < max_doc {
                if let Some(x) = v.value(d).unwrap() {
                    s = s.wrapping_add(x);
                }
                n += 1;
                d += 37;
            }
            black_box(s);
            n
        });
        let (meta, data) = r.doc_values_for_field(kw).expect("dv");
        let kw_entry = meta.sorted_entry(kw).expect("keyword sorted entry");
        measure("sorted_ord_seq", w, m, || {
            let mut v = doc_values::NumericReader::new(data, &kw_entry.ords);
            let mut s = 0i64;
            v.for_each_value(0, max_doc, |_, x| s = s.wrapping_add(x))
                .unwrap();
            black_box(s);
            max_doc as u64
        });
        let (meta, data) = r.doc_values_for_field(cat).expect("dv");
        let cat_entry = meta.sorted_set_entry(cat).expect("cat entry");
        measure("sorted_set_seq", w, m, || {
            let mut s = 0i64;
            match &cat_entry.kind {
                SortedSetKind::Single(e) => {
                    let mut v = doc_values::NumericReader::new(data, &e.ords);
                    v.for_each_value(0, max_doc, |_, x| s = s.wrapping_add(x))
                        .unwrap();
                }
                SortedSetKind::Multi { ords, .. } => {
                    for d in 0..max_doc {
                        for x in doc_values::sorted_numeric_values(data, ords, d).unwrap() {
                            s = s.wrapping_add(x);
                        }
                    }
                }
            }
            black_box(s);
            max_doc as u64
        });
    });
}

fn bench_norms(w: Duration, m: Duration, index: &str) {
    with_segment(index, |r, _| {
        let max_doc = r.max_doc;
        let body = r.field_infos().field_by_name("body").expect("body").number;
        let entry = r.norms_entry(body).expect("norms entry");
        let data = r.norms_data().expect("norms data");
        measure("body_seq", w, m, || {
            let mut r = lucene_codecs::norms::NormsReader::new(data, entry);
            let mut s = 0i64;
            for d in 0..max_doc {
                if let Some(x) = r.value(d).unwrap() {
                    s = s.wrapping_add(x);
                }
            }
            black_box(s);
            max_doc as u64
        });
        measure("body_stride37", w, m, || {
            let mut r = lucene_codecs::norms::NormsReader::new(data, entry);
            let (mut s, mut n) = (0i64, 0u64);
            let mut d = 0;
            while d < max_doc {
                if let Some(x) = r.value(d).unwrap() {
                    s = s.wrapping_add(x);
                }
                n += 1;
                d += 37;
            }
            black_box(s);
            n
        });
    });
}

struct RangeCounter {
    lo: [u8; 8],
    hi: [u8; 8],
    count: u64,
}

impl lucene_codecs::points::IntersectVisitor for RangeCounter {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> lucene_codecs::points::Relation {
        use lucene_codecs::points::Relation;
        if min[..8] > self.hi[..] || max[..8] < self.lo[..] {
            return Relation::CellOutsideQuery;
        }
        if min[..8] >= self.lo[..] && max[..8] <= self.hi[..] {
            return Relation::CellInsideQuery;
        }
        Relation::CellCrossesQuery
    }
    fn visit(&mut self, _doc: i32) {
        self.count += 1;
    }
    fn visit_with_value(&mut self, _doc: i32, v: &[u8]) {
        if v[..8] >= self.lo[..] && v[..8] <= self.hi[..] {
            self.count += 1;
        }
    }
}

fn bench_points(w: Duration, m: Duration, index: &str) {
    with_segment(index, |r, _| {
        let (kdm, kdi, kdd) = r.points_files().expect("points");
        let pr =
            lucene_codecs::points::open(kdm, kdi, kdd, &r.segment_id(), "").expect("open points");
        let num = r.field_infos().field_by_name("num").expect("num").number;
        for (lo, hi) in [(0i64, 1000i64), (0, 100_000), (250_000, 750_000)] {
            let (plo, phi) = (
                lucene_search::points_query::pack_i64(lo),
                lucene_search::points_query::pack_i64(hi),
            );
            measure(&format!("range_{lo}_{hi}"), w, m, || {
                let mut c = RangeCounter {
                    lo: plo[..8].try_into().unwrap(),
                    hi: phi[..8].try_into().unwrap(),
                    count: 0,
                };
                pr.intersect(num, &mut c).unwrap();
                black_box(c.count);
                1
            });
        }
    });
}

/// Heap an open reader keeps resident -- see `SweepMicro.memory`.
fn bench_memory(index: &str) {
    let dir = MmapDirectory::open(index.to_string());
    let before = counting_alloc::live();
    let reader = DirectoryReader::open(&dir).expect("open index");
    let opened = reader.open_segments().expect("open segments");
    let after = counting_alloc::live();
    println!("open_heap_bytes\t{}\t1", after - before);
    let segs = opened.as_open_segments();
    for seg in &segs {
        for f in ["body", "title", "keyword"] {
            let (Some(field), Some(doc_in)) = (seg.fields.field(f), seg.doc_in) else {
                continue;
            };
            if let Ok(Some(mut c)) = field.lazy_postings_with_flags(
                b"t0",
                doc_in,
                lucene_codecs::postings::PostingsFlags::Freqs,
            ) {
                let mut s = 0u64;
                while c.next_doc().unwrap() != lucene_codecs::postings::NO_MORE_DOCS {
                    s += c.freq().unwrap_or(1) as u64;
                }
                black_box(s);
            }
        }
    }
    let touched = counting_alloc::live();
    println!("after_query_heap_bytes\t{}\t1", touched - before);
    drop(segs);
    drop(opened);
    drop(reader);
}

/// Term-dictionary write throughput, against `TermDictWriteMicro.java`: one
/// singleton `IndexOptions::Docs` field written by `postings_writer`, so the
/// work is the block-tree writer and `encodeTerm` (no `.doc` bytes). Same
/// generated terms as the Java side; for these inputs both engines write
/// identical bytes (`crates/lucene-codecs/tests/blocktree_writer_identity.rs`).
fn bench_term_dict_write(warmup: Duration, measure: Duration) {
    use lucene_codecs::field_infos::IndexOptions;
    use lucene_codecs::postings_writer::{self, FieldPostingsInput, TermPostings};

    fn id_terms(n: usize) -> Vec<Vec<u8>> {
        (0..n).map(|i| format!("{i:08}").into_bytes()).collect()
    }
    fn word_terms(n: usize) -> Vec<Vec<u8>> {
        let mut s: u64 = 42;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut set = std::collections::BTreeSet::new();
        while set.len() < n {
            let len = 4 + (next() % 9) as usize;
            let word: Vec<u8> = (0..len).map(|_| b'a' + (next() % 26) as u8).collect();
            set.insert(word);
        }
        set.into_iter().collect()
    }

    for (name, terms) in [("ids_1m", id_terms(1_000_000)), ("words_200k", word_terms(200_000))] {
        let postings: Vec<TermPostings> = terms
            .into_iter()
            .enumerate()
            .map(|(doc, term)| TermPostings {
                term,
                docs: vec![(doc as i32, 1)],
                ..TermPostings::default()
            })
            .collect();
        let n = postings.len();
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::Docs,
            doc_count: n as i32,
            has_payloads: false,
            terms: &postings,
        };
        let id = [7u8; 16];
        let write = || {
            postings_writer::write_fields(std::slice::from_ref(black_box(&input)), &id, "")
                .expect("write")
        };
        let out = write();
        let bytes = out.doc.len() + out.psm.len() + out.tim.len() + out.tip.len() + out.tmd.len();
        eprintln!("{name}: {n} terms, {bytes} bytes written");
        let run = |budget| {
            timed_loop(budget, || {
                black_box(write());
            })
        };
        run(warmup);
        let (elapsed, ops) = run(measure);
        let units = ops * n as u64;
        println!(
            "{name}\t{:.3}\t{units}",
            elapsed.as_nanos() as f64 / units as f64
        );
    }
}

/// Doc-values merge throughput, against `DvMergeMicro.java`: four segments of
/// documents carrying five sparse doc-values columns (every type, each missing
/// on a different stride, as `write_sparse_doc_values_fixture`), merged into
/// one. Each timed merge starts from a fresh, untimed copy of the unmerged
/// segments, and includes opening the writer -- as the Java side's does.
fn bench_dv_merge(warmup: Duration, measure: Duration) {
    use lucene_codecs::field_infos::{
        DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };
    use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::merge_policy::MergePolicyConfig;
    use lucene_index::segment_info::LuceneVersion;
    use lucene_store::FsDirectory;

    const DOCS: usize = 200_000;
    const SEGMENTS: usize = 4;
    let version = LuceneVersion { major: 10, minor: 5, bugfix: 0 };
    let field = |name: &str, number: i32, dv: DocValuesType| FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: true,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::None,
        doc_values_type: dv,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Byte,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    };
    let fields = || {
        vec![
            field("num", 0, DocValuesType::Numeric),
            field("bin", 1, DocValuesType::Binary),
            field("sorted", 2, DocValuesType::Sorted),
            field("snum", 3, DocValuesType::SortedNumeric),
            field("sset", 4, DocValuesType::SortedSet),
        ]
    };
    let document = |i: usize| {
        let mut fields = Vec::new();
        let mut add = |field_number, value| fields.push(StoredField { field_number, value });
        if !i.is_multiple_of(3) {
            add(0, FieldValue::Long(7 * i as i64 - 1000));
        }
        if !i.is_multiple_of(5) {
            add(1, FieldValue::Binary(format!("b{i}").into_bytes()));
        }
        if !i.is_multiple_of(7) {
            add(2, FieldValue::String(format!("s{}", i % 50)));
        }
        if !i.is_multiple_of(11) {
            for v in [(i % 13) as i64, i as i64, -(i as i64)] {
                add(3, FieldValue::Long(v));
            }
        }
        if !i.is_multiple_of(13) {
            add(4, FieldValue::String(format!("t{}", i % 17)));
            add(4, FieldValue::String(format!("t{}", i % 19)));
        }
        Document { fields }
    };
    let configure = |w: &mut IndexWriter<'_>| {
        w.set_doc_values_field(Some("num")).unwrap();
        for name in ["bin", "sorted", "snum", "sset"] {
            w.add_doc_values_field(name).unwrap();
        }
    };

    let root = std::env::temp_dir().join(format!("dv-merge-micro-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let source = root.join("source");
    std::fs::create_dir_all(&source).unwrap();
    {
        let dir = FsDirectory::open(&source);
        let mut w = IndexWriter::open(&dir, fields(), "Lucene104", version).unwrap();
        w.set_max_buffered_docs((DOCS / SEGMENTS) as i32).unwrap();
        // Flush on document count only, as the Java side does.
        w.set_ram_buffer_size_mb(4096.0).unwrap();
        configure(&mut w);
        for i in 0..DOCS {
            w.add_document(document(i)).unwrap();
        }
        w.commit().unwrap();
        assert_eq!(w.segment_infos().segments.len(), SEGMENTS);
    }
    let work = root.join("work");
    let merge_once = || -> Duration {
        let _ = std::fs::remove_dir_all(&work);
        std::fs::create_dir_all(&work).unwrap();
        for f in std::fs::read_dir(&source).unwrap() {
            let f = f.unwrap();
            std::fs::copy(f.path(), work.join(f.file_name())).unwrap();
        }
        let start = Instant::now();
        let dir = FsDirectory::open(&work);
        let mut w = IndexWriter::open(&dir, fields(), "Lucene104", version).unwrap();
        configure(&mut w);
        w.set_merge_policy(Some(MergePolicyConfig {
            max_merge_at_once: 10,
            segments_per_tier: 2,
            max_merged_segment_size: u64::MAX / 4,
            floor_segment_size: 1 << 30,
            ..MergePolicyConfig::default()
        }));
        w.commit().unwrap();
        let elapsed = start.elapsed();
        assert_eq!(w.segment_infos().segments.len(), 1, "the merge ran");
        elapsed
    };
    let warm_end = Instant::now() + warmup;
    while Instant::now() < warm_end {
        merge_once();
    }
    let mut total = Duration::ZERO;
    let mut docs = 0u64;
    let measure_end = Instant::now() + measure;
    loop {
        total += merge_once();
        docs += DOCS as u64;
        if Instant::now() >= measure_end {
            break;
        }
    }
    println!(
        "sparse_5_types\t{:.3}\t{docs}",
        total.as_nanos() as f64 / docs as f64
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// Points write throughput, against `PointsWriteMicro.java`: documents
/// carrying the four point fields of `write_points_segment_fixture` and
/// nothing else. `flush` indexes 200 000 of them into one segment and
/// commits; `merge` merges four 50 000-document segments into one, each run
/// from a fresh, untimed copy (writer open included, as on the Java side).
fn bench_points_write(warmup: Duration, measure: Duration) {
    use lucene_codecs::field_infos::{
        DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };
    use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::merge_policy::MergePolicyConfig;
    use lucene_index::segment_info::LuceneVersion;
    use lucene_store::FsDirectory;

    const DOCS: usize = 200_000;
    const SEGMENTS: usize = 4;
    let version = LuceneVersion { major: 10, minor: 5, bugfix: 0 };
    let point = |name: &str, number: i32, dims: i32, bytes: i32| FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: true,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::None,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: dims,
        point_index_dimension_count: dims,
        point_num_bytes: bytes,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Byte,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    };
    let fields = || {
        vec![
            point("lp", 0, 1, 8),
            point("ip", 1, 1, 4),
            point("dp", 2, 1, 8),
            point("xy", 3, 2, 4),
        ]
    };
    let pack = |values: &[i32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| ((v as u32) ^ 0x8000_0000).to_be_bytes())
            .collect()
    };
    let document = |i: usize| {
        let mut fields = Vec::new();
        let mut add = |field_number, value| fields.push(StoredField { field_number, value });
        add(0, FieldValue::Long(7 * i as i64 - 1000));
        if i.is_multiple_of(4) {
            add(0, FieldValue::Long(-(i as i64)));
        }
        if !i.is_multiple_of(3) {
            add(1, FieldValue::Int((i % 1000) as i32));
        }
        add(2, FieldValue::Double(i as f64 / 8.0 - 100.0));
        add(3, FieldValue::Binary(pack(&[(i % 97) as i32, (i % 89) as i32 - 44])));
        Document { fields }
    };
    let index = |path: &std::path::Path, per_segment: usize| {
        std::fs::create_dir_all(path).unwrap();
        let dir = FsDirectory::open(path);
        let mut w = IndexWriter::open(&dir, fields(), "Lucene104", version).unwrap();
        w.set_max_buffered_docs(per_segment as i32).unwrap();
        w.set_ram_buffer_size_mb(4096.0).unwrap();
        for name in ["lp", "ip", "dp", "xy"] {
            w.add_points_field(name).unwrap();
        }
        for i in 0..DOCS {
            w.add_document(document(i)).unwrap();
        }
        w.commit().unwrap();
        w.segment_infos().segments.len()
    };
    let run = |name: &str, mut once: Box<dyn FnMut() -> Duration + '_>| {
        let warm_end = Instant::now() + warmup;
        while Instant::now() < warm_end {
            once();
        }
        let mut total = Duration::ZERO;
        let mut docs = 0u64;
        let end = Instant::now() + measure;
        loop {
            total += once();
            docs += DOCS as u64;
            if Instant::now() >= end {
                break;
            }
        }
        println!("{name}\t{:.3}\t{docs}", total.as_nanos() as f64 / docs as f64);
    };

    let root = std::env::temp_dir().join(format!("points-write-micro-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let flush_dir = root.join("flush");
    run(
        "flush",
        Box::new(|| {
            let _ = std::fs::remove_dir_all(&flush_dir);
            let start = Instant::now();
            assert_eq!(index(&flush_dir, DOCS), 1);
            start.elapsed()
        }),
    );

    let source = root.join("source");
    assert_eq!(index(&source, DOCS / SEGMENTS), SEGMENTS);
    let work = root.join("work");
    run(
        "merge",
        Box::new(|| {
            let _ = std::fs::remove_dir_all(&work);
            std::fs::create_dir_all(&work).unwrap();
            for f in std::fs::read_dir(&source).unwrap() {
                let f = f.unwrap();
                std::fs::copy(f.path(), work.join(f.file_name())).unwrap();
            }
            let start = Instant::now();
            let dir = FsDirectory::open(&work);
            let mut w = IndexWriter::open(&dir, fields(), "Lucene104", version).unwrap();
            for name in ["lp", "ip", "dp", "xy"] {
                w.add_points_field(name).unwrap();
            }
            w.set_merge_policy(Some(MergePolicyConfig {
                max_merge_at_once: 10,
                segments_per_tier: 2,
                max_merged_segment_size: u64::MAX / 4,
                floor_segment_size: 1 << 30,
                ..MergePolicyConfig::default()
            }));
            w.commit().unwrap();
            let elapsed = start.elapsed();
            assert_eq!(w.segment_infos().segments.len(), 1, "the merge ran");
            elapsed
        }),
    );
    let _ = std::fs::remove_dir_all(&root);
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
        "vint" => bench_vint(warmup, measure),
        "pfor_decode" => bench_pfor_decode(warmup, measure),
        "bitset" => bench_bitset(warmup, measure),
        "lz4" => bench_lz4(warmup, measure),
        "direct_monotonic" => bench_direct_monotonic(warmup, measure),
        "checksum" => bench_checksum(warmup, measure),
        "analysis" => bench_analysis(warmup, measure),
        "vectors" => bench_vectors(warmup, measure),
        "term_dict_write" => bench_term_dict_write(warmup, measure),
        "dv_merge" => bench_dv_merge(warmup, measure),
        "points_write" => bench_points_write(warmup, measure),
        corpus @ ("postings_adv" | "postings_freq" | "positions" | "term_seek" | "doc_values"
        | "norms" | "points" | "memory") => {
            let index = std::env::args()
                .nth(2)
                .unwrap_or_else(|| panic!("{corpus} needs an index directory"));
            match corpus {
                "postings_adv" => bench_postings_adv(warmup, measure, &index),
                "postings_freq" => bench_postings_freq(warmup, measure, &index),
                "positions" => bench_positions(warmup, measure, &index),
                "term_seek" => bench_term_seek(warmup, measure, &index),
                "doc_values" => bench_doc_values(warmup, measure, &index),
                "norms" => bench_norms(warmup, measure, &index),
                "points" => bench_points(warmup, measure, &index),
                _ => bench_memory(&index),
            }
        }
        other => {
            eprintln!("micro: unknown benchmark {other:?}");
            std::process::exit(2);
        }
    }
}
