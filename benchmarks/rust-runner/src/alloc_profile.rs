//! Heap allocations and peak live heap along the OpenSearch plugin's hot
//! paths, through the same C ABI the plugin calls.
//!
//! A counting global allocator records every allocation, the bytes asked for
//! and the live-heap high-water mark; each phase reports them per operation:
//!
//! - `index`: `ffi_engine_writer_apply`, one JVM-inverted document per call
//!   (stored `_id` and `_source`, inverted `_id` and `body`, `_seq_no` doc
//!   value and point, `_version` doc value), the blob built outside the
//!   counted region;
//! - `commit`: the flush of everything buffered;
//! - `force_merge`: the segments merged into one;
//! - `open`: `ffi_open_jvm_reader` over the merged commit;
//! - `search/<shape>`: `ffi_jvm_reader_search`, top 10;
//! - `fetch`: `ffi_jvm_reader_document_alloc` for each hit of a search.
//!
//! Background threads (parallel flush and merge stages) are counted too: the
//! counters are process-wide. Run with
//! `cargo run --release --bin alloc-profile -- [docs]`; the index goes to a
//! fresh directory under `$TMPDIR`.
//!
//! The numbers are what the allocator sees, not RSS: `index-bench` reports
//! that.

use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::c_char;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use lucene_ffi as _;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::FsDirectory;

struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);

fn grew(by: u64) {
    let live = LIVE.fetch_add(by, Relaxed) + by;
    PEAK.fetch_max(live, Relaxed);
}

// SAFETY: every method forwards to `System` with the caller's own layout and
// pointer; the counters never touch the memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size() as u64, Relaxed);
        grew(layout.size() as u64);
        // SAFETY: forwarded.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size() as u64, Relaxed);
        grew(layout.size() as u64);
        // SAFETY: forwarded.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as u64, Relaxed);
        // SAFETY: forwarded.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(new_size as u64, Relaxed);
        if new_size >= layout.size() {
            grew((new_size - layout.size()) as u64);
        } else {
            LIVE.fetch_sub((layout.size() - new_size) as u64, Relaxed);
        }
        // SAFETY: forwarded.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The counters at the start of a phase.
struct Phase {
    name: String,
    allocs: u64,
    bytes: u64,
    live: u64,
    start: std::time::Instant,
}

fn begin(name: impl Into<String>) -> Phase {
    let live = LIVE.load(Relaxed);
    PEAK.store(live, Relaxed);
    Phase {
        name: name.into(),
        allocs: ALLOCS.load(Relaxed),
        bytes: BYTES.load(Relaxed),
        live,
        start: std::time::Instant::now(),
    }
}

/// One TSV row: phase, ops, allocations per op, bytes per op, live-heap
/// growth over the phase, peak above the phase's starting live heap, ms.
fn end(p: Phase, ops: u64) {
    let ms = p.start.elapsed().as_secs_f64() * 1e3;
    let allocs = ALLOCS.load(Relaxed) - p.allocs;
    let bytes = BYTES.load(Relaxed) - p.bytes;
    let live = LIVE.load(Relaxed) as i64 - p.live as i64;
    let peak = PEAK.load(Relaxed) - p.live;
    let ops = ops.max(1);
    println!(
        "{}\t{}\t{:.1}\t{:.0}\t{}\t{}\t{:.1}",
        p.name,
        ops,
        allocs as f64 / ops as f64,
        bytes as f64 / ops as f64,
        live / 1024,
        peak / 1024,
        ms
    );
}

extern "C" {
    fn ffi_engine_writer_open(
        path: *const c_char,
        path_len: usize,
        ram_buffer_mb: f64,
        fault_injection: u8,
        max_docs: i32,
        out_handle: *mut u64,
    ) -> i32;
    fn ffi_engine_writer_register_field(
        handle: u64,
        spec: *const u8,
        spec_len: usize,
        out_number: *mut i32,
    ) -> i32;
    fn ffi_engine_writer_apply(handle: u64, op: *const u8, op_len: usize) -> i32;
    fn ffi_engine_writer_commit(
        handle: u64,
        user_data: *const u8,
        user_data_len: usize,
        out_generation: *mut i64,
    ) -> i32;
    fn ffi_engine_writer_force_merge(
        handle: u64,
        max_segments: i32,
        only_deletes: u8,
        out_generation: *mut i64,
    ) -> i32;
    fn ffi_engine_writer_close(handle: u64) -> i32;
    fn ffi_engine_writer_stats(handle: u64, out: *mut i64, n: usize) -> i32;
    fn ffi_open_jvm_reader(
        path: *const c_char,
        path_len: usize,
        infos: *const u8,
        infos_len: usize,
        generation: i64,
        previous: u64,
        expected_max_docs: *const i32,
        segment_count: usize,
        live_words: *const u64,
        live_word_counts: *const usize,
        out_handle: *mut u64,
    ) -> i32;
    fn ffi_jvm_reader_search(
        handle: u64,
        query: *const u8,
        query_len: usize,
        top_n: usize,
        count_limit: i64,
        out_docs: *mut i32,
        out_scores: *mut f32,
        buf_len: usize,
        out_hit_count: *mut usize,
        out_total: *mut i64,
        out_total_is_lower_bound: *mut bool,
    ) -> i32;
    fn ffi_jvm_reader_document_alloc(
        handle: u64,
        segment: i32,
        doc: i32,
        out_ptr: *mut *mut u8,
        out_len: *mut usize,
    ) -> i32;
    fn ffi_jvm_free_bytes(ptr: *mut u8, len: usize);
    fn ffi_close_jvm_reader(handle: u64) -> i32;
}

/// A little-endian blob, as the plugin's encoders build them.
#[derive(Default)]
struct Blob(Vec<u8>);

impl Blob {
    fn u8(&mut self, v: u8) -> &mut Self {
        self.0.push(v);
        self
    }
    fn i32(&mut self, v: i32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn i64(&mut self, v: i64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn bytes(&mut self, b: &[u8]) -> &mut Self {
        self.i32(b.len() as i32);
        self.0.extend_from_slice(b);
        self
    }
}

/// `ffi_engine_writer_register_field`'s spec.
fn field(
    name: &str,
    index_options: u8,
    omit_norms: u8,
    dv: u8,
    point_bytes: i32,
    soft: u8,
) -> Vec<u8> {
    let mut b = Blob::default();
    b.bytes(name.as_bytes())
        .u8(index_options)
        .u8(omit_norms)
        .u8(0)
        .u8(0)
        .u8(dv)
        .u8(0)
        .i32(i32::from(point_bytes > 0))
        .i32(i32::from(point_bytes > 0))
        .i32(point_bytes)
        .i32(0)
        .u8(soft);
    b.0
}

fn check(rc: i32, what: &str) {
    assert_eq!(rc, 0, "{what} failed with status {rc}");
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Skewed towards small values, so a few terms are frequent.
    fn skewed(&mut self, n: u64) -> u64 {
        let a = self.next() % n;
        self.next() % (a + 1)
    }
}

struct Fields {
    id: i32,
    body: i32,
    seq: i32,
    version: i32,
    source: i32,
}

/// One add operation: an OpenSearch-shaped document, inverted as the JVM
/// inverts it (one entry per distinct term, its positions ascending).
fn add_op(f: &Fields, i: usize, rng: &mut Rng, vocab: &[String], words: usize) -> Vec<u8> {
    let id = format!("{i:010}");
    let mut terms: std::collections::BTreeMap<&str, Vec<i32>> = Default::default();
    let mut source = String::from("{\"body\":\"");
    for p in 0..words {
        let w = &vocab[rng.skewed(vocab.len() as u64) as usize];
        terms.entry(w).or_default().push(p as i32);
        if p > 0 {
            source.push(' ');
        }
        source.push_str(w);
    }
    source.push_str("\"}");
    let mut b = Blob::default();
    b.u8(0).i32(1);
    // stored: _id (binary), _source (binary)
    b.i32(2);
    b.i32(f.id).u8(1).bytes(id.as_bytes());
    b.i32(f.source).u8(1).bytes(source.as_bytes());
    // inverted: _id (docs only, no norms), body (positions, a norm)
    b.i32(2);
    b.i32(f.id).u8(0).i32(1).bytes(id.as_bytes()).i32(1).u8(0);
    b.i32(f.body)
        .u8(1)
        .i64(words as i64)
        .i32(terms.len() as i32);
    for (t, positions) in &terms {
        b.bytes(t.as_bytes()).i32(positions.len() as i32).u8(1);
        for &p in positions {
            b.i32(p);
        }
    }
    // doc values: _seq_no, _version
    b.i32(2);
    b.i32(f.seq).u8(0).i64(i as i64);
    b.i32(f.version).u8(0).i64(1);
    // points: _seq_no
    b.i32(1);
    b.i32(f.seq).bytes(&((i as i64) ^ i64::MIN).to_be_bytes());
    b.0
}

fn term_query(field: &str, term: &str) -> Vec<u8> {
    let mut b = Blob::default();
    b.u8(0).bytes(field.as_bytes()).bytes(term.as_bytes());
    b.0
}

/// A flat boolean of term clauses, `(occur, term)` each.
fn boolean_query(field: &str, clauses: &[(u8, &str)], msm: i32) -> Vec<u8> {
    let mut b = Blob::default();
    b.u8(1).i32(msm).i32(clauses.len() as i32);
    for (occur, term) in clauses {
        b.u8(*occur)
            .u8(0)
            .i32(-1)
            .i32(0)
            .bytes(field.as_bytes())
            .bytes(term.as_bytes());
    }
    b.0
}

fn phrase_query(field: &str, terms: &[&str]) -> Vec<u8> {
    let mut b = Blob::default();
    b.u8(2)
        .u8(7)
        .bytes(field.as_bytes())
        .i32(0)
        .i32(terms.len() as i32);
    for (i, t) in terms.iter().enumerate() {
        b.i32(i as i32).bytes(t.as_bytes());
    }
    b.0
}

fn main() {
    let docs: usize = std::env::args()
        .nth(1)
        .map(|a| a.parse().expect("docs"))
        .unwrap_or(20_000);
    let words = 120;
    let vocab: Vec<String> = (0..30_000).map(|i| format!("w{i}")).collect();
    let root = std::env::temp_dir().join(format!("alloc-profile-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.to_str().unwrap().to_string();
    {
        let dir = FsDirectory::open(&root);
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
        w.commit().unwrap();
    }
    println!("phase\tops\tallocs/op\tbytes/op\tlive_kb\tpeak_kb\tms");

    let mut h = 0u64;
    // SAFETY: live string and out-pointer.
    check(
        unsafe {
            ffi_engine_writer_open(path.as_ptr().cast(), path.len(), 64.0, 0, i32::MAX, &mut h)
        },
        "open writer",
    );
    let register = |spec: Vec<u8>| {
        let mut n = -1;
        // SAFETY: live slice and out-pointer.
        check(
            unsafe { ffi_engine_writer_register_field(h, spec.as_ptr(), spec.len(), &mut n) },
            "register",
        );
        n
    };
    let f = Fields {
        id: register(field("_id", 1, 1, 0, 0, 0)),
        body: register(field("body", 3, 0, 0, 0, 0)),
        seq: register(field("_seq_no", 0, 0, 1, 8, 0)),
        version: register(field("_version", 0, 0, 1, 0, 0)),
        source: register(field("_source", 0, 0, 0, 0, 0)),
    };
    register(field("__soft_deletes", 0, 0, 1, 0, 1));

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let ops: Vec<Vec<u8>> = (0..docs)
        .map(|i| add_op(&f, i, &mut rng, &vocab, words))
        .collect();
    let segments = 4;
    let per_commit = docs.div_ceil(segments);
    let mut generation = 0i64;
    let empty_user_data = 0i32.to_le_bytes();
    let mut index_phase = begin("index");
    let mut commit_total = (0u64, 0u64, 0u64, 0f64);
    // The writer's own RAM accounting (what triggers a flush) against what
    // the buffered documents really hold on the heap, at each commit.
    let mut buffer_start = LIVE.load(Relaxed);
    let mut ram = (0i64, 0i64);
    for (i, op) in ops.iter().enumerate() {
        // SAFETY: live slice.
        check(
            unsafe { ffi_engine_writer_apply(h, op.as_ptr(), op.len()) },
            "apply",
        );
        if (i + 1) % per_commit == 0 || i + 1 == docs {
            let mut stats = [0i64; 5];
            // SAFETY: `stats` holds the five values.
            check(
                unsafe { ffi_engine_writer_stats(h, stats.as_mut_ptr(), 5) },
                "stats",
            );
            ram.0 += stats[0];
            ram.1 += LIVE.load(Relaxed) as i64 - buffer_start as i64;
            // The commit is its own phase; take it out of `index`.
            let a = ALLOCS.load(Relaxed);
            let by = BYTES.load(Relaxed);
            let t = std::time::Instant::now();
            let live_before = LIVE.load(Relaxed);
            let peak_before = PEAK.load(Relaxed);
            PEAK.store(live_before, Relaxed);
            // SAFETY: live slice and out-pointer.
            check(
                unsafe {
                    ffi_engine_writer_commit(
                        h,
                        empty_user_data.as_ptr(),
                        empty_user_data.len(),
                        &mut generation,
                    )
                },
                "commit",
            );
            commit_total.0 += ALLOCS.load(Relaxed) - a;
            commit_total.1 += BYTES.load(Relaxed) - by;
            commit_total.2 = commit_total.2.max(PEAK.load(Relaxed) - live_before);
            commit_total.3 += t.elapsed().as_secs_f64() * 1e3;
            PEAK.store(peak_before.max(LIVE.load(Relaxed)), Relaxed);
            index_phase.allocs += ALLOCS.load(Relaxed) - a;
            index_phase.bytes += BYTES.load(Relaxed) - by;
            index_phase.start += t.elapsed();
            buffer_start = LIVE.load(Relaxed);
        }
    }
    end(index_phase, docs as u64);
    println!(
        "commit\t{segments}\t{:.1}\t{:.0}\t-\t{}\t{:.1}",
        commit_total.0 as f64 / segments as f64,
        commit_total.1 as f64 / segments as f64,
        commit_total.2 / 1024,
        commit_total.3
    );
    eprintln!(
        "buffer: writer accounts {} KB, heap holds {} KB (per flush, mean)",
        ram.0 / 1024 / segments as i64,
        ram.1 / 1024 / segments as i64
    );
    drop(ops);

    let p = begin("force_merge");
    // SAFETY: live out-pointer.
    check(
        unsafe { ffi_engine_writer_force_merge(h, 1, 0, &mut generation) },
        "force merge",
    );
    end(p, 1);
    ffi_engine_writer_close_checked(h);

    // The reader, as the plugin opens it: the commit's segments_N bytes.
    let files: Vec<String> = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    let generation = lucene_store::directory::last_commit_generation(&files).unwrap();
    let infos =
        std::fs::read(root.join(lucene_store::directory::segments_file_name(generation).unwrap()))
            .unwrap();
    let max_docs: Vec<i32> = {
        let dir = FsDirectory::open(&root);
        let reader = DirectoryReader::open(&dir).unwrap();
        reader.segment_readers().iter().map(|s| s.max_doc).collect()
    };
    let mut r = 0u64;
    let p = begin("open");
    // SAFETY: live slices and out-pointer; null live docs (no deletions).
    check(
        unsafe {
            ffi_open_jvm_reader(
                path.as_ptr().cast(),
                path.len(),
                infos.as_ptr(),
                infos.len(),
                generation,
                0,
                max_docs.as_ptr(),
                max_docs.len(),
                std::ptr::null(),
                std::ptr::null(),
                &mut r,
            )
        },
        "open reader",
    );
    end(p, 1);

    let frequent = vocab[0].as_str();
    let common = vocab[3].as_str();
    let mid = vocab[40].as_str();
    let rare = vocab[2000].as_str();
    let shapes: Vec<(&str, Vec<u8>)> = vec![
        ("term_frequent", term_query("body", frequent)),
        ("term_rare", term_query("body", rare)),
        ("id_lookup", term_query("_id", "0000001234")),
        ("or2", boolean_query("body", &[(2, common), (2, mid)], 0)),
        ("and2", boolean_query("body", &[(0, common), (0, mid)], 0)),
        (
            "filter_or",
            boolean_query("body", &[(1, frequent), (2, common), (2, mid)], 0),
        ),
        ("phrase2", phrase_query("body", &[frequent, common])),
    ];
    let iters = 200u64;
    let mut out_docs = [0i32; 10];
    let mut out_scores = [0f32; 10];
    let mut fetched = Vec::new();
    for (name, q) in &shapes {
        let run = |out_docs: &mut [i32; 10], out_scores: &mut [f32; 10]| {
            let (mut n, mut total, mut lower) = (0usize, 0i64, false);
            // SAFETY: live slice and buffers of 10.
            check(
                unsafe {
                    ffi_jvm_reader_search(
                        r,
                        q.as_ptr(),
                        q.len(),
                        10,
                        1000,
                        out_docs.as_mut_ptr(),
                        out_scores.as_mut_ptr(),
                        10,
                        &mut n,
                        &mut total,
                        &mut lower,
                    )
                },
                name,
            );
            n
        };
        // Warm the reader's lazily built per-field state first.
        let n = run(&mut out_docs, &mut out_scores);
        fetched.extend_from_slice(&out_docs[..n]);
        let p = begin(format!("search/{name}"));
        for _ in 0..iters {
            run(&mut out_docs, &mut out_scores);
        }
        end(p, iters);
    }

    // Stored fields of the hits, as the fetch phase asks for them.
    let p = begin("fetch");
    let mut fetches = 0u64;
    for _ in 0..20 {
        for &doc in &fetched {
            let (mut ptr, mut len) = (std::ptr::null_mut(), 0usize);
            // SAFETY: live out-pointers; the buffer goes back to the library.
            unsafe {
                check(
                    ffi_jvm_reader_document_alloc(r, 0, doc, &mut ptr, &mut len),
                    "fetch",
                );
                ffi_jvm_free_bytes(ptr, len);
            }
            fetches += 1;
        }
    }
    end(p, fetches);
    // SAFETY: a handle this process opened.
    unsafe { ffi_close_jvm_reader(r) };
    let _ = std::fs::remove_dir_all(&root);
}

fn ffi_engine_writer_close_checked(h: u64) {
    // SAFETY: a handle this process opened.
    check(unsafe { ffi_engine_writer_close(h) }, "close writer");
}
