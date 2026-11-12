//! Time and heap allocations per call of the plugin's aggregation, sorted
//! search and fetch entry points, through the C ABI the plugin calls, over
//! the benchmark corpus (`scripts/bench-corpus.sh`: `keyword` SORTED, `cat`
//! SORTED_SET, `num` a LongPoint with NUMERIC doc values, `title` and `id`
//! stored).
//!
//! `sweep <index dir> [case-substring]`; `SWEEP_MS` sets each case's
//! measuring time (default 1500 ms, after a 300 ms warmup). Each row: case,
//! calls, microseconds per call, allocations per call, bytes allocated per
//! call, peak live heap above the start of the case (KiB), and a digest of
//! the output so two builds can be checked for equal results.

use std::alloc::{GlobalAlloc, Layout};
use std::ffi::c_char;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use lucene_ffi as _;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::FsDirectory;
use mimalloc::MiMalloc;

struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);

/// `SWEEP_BIG`: allocations of at least this many bytes print a backtrace
/// (to stderr), to find where the bytes go.
static BIG: AtomicU64 = AtomicU64::new(u64::MAX);
thread_local! {
    static IN_TRACE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn traced(size: usize) {
    if size as u64 >= BIG.load(Relaxed) {
        IN_TRACE.with(|t| {
            if !t.get() {
                t.set(true);
                eprintln!(
                    "--- {size} bytes\n{}",
                    std::backtrace::Backtrace::force_capture()
                );
                t.set(false);
            }
        });
    }
}

fn grew(by: u64) {
    let live = LIVE.fetch_add(by, Relaxed) + by;
    PEAK.fetch_max(live, Relaxed);
}

// SAFETY: every method forwards to mimalloc with the caller's own layout and
// pointer; the counters never touch the memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        traced(layout.size());
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size() as u64, Relaxed);
        grew(layout.size() as u64);
        // SAFETY: forwarded.
        unsafe { MiMalloc.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        traced(layout.size());
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size() as u64, Relaxed);
        grew(layout.size() as u64);
        // SAFETY: forwarded.
        unsafe { MiMalloc.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as u64, Relaxed);
        // SAFETY: forwarded.
        unsafe { MiMalloc.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        traced(new_size);
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(new_size as u64, Relaxed);
        if new_size >= layout.size() {
            grew((new_size - layout.size()) as u64);
        } else {
            LIVE.fetch_sub((layout.size() - new_size) as u64, Relaxed);
        }
        // SAFETY: forwarded.
        unsafe { MiMalloc.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

extern "C" {
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
    fn ffi_jvm_reader_search_sorted_alloc(
        handle: u64,
        query: *const u8,
        query_len: usize,
        sort: *const u8,
        sort_len: usize,
        top_n: usize,
        count_limit: i64,
        out_docs: *mut i32,
        docs_cap: usize,
        out_values: *mut i64,
        values_cap: usize,
        out_counts: *mut i64,
        out_terms: *mut *mut u8,
        out_terms_len: *mut usize,
    ) -> i32;
    fn ffi_jvm_reader_aggregate_tree_alloc(
        handle: u64,
        query: *const u8,
        query_len: usize,
        tree: *const u8,
        tree_len: usize,
        out_ptr: *mut *mut u8,
        out_len: *mut usize,
    ) -> i32;
    fn ffi_jvm_reader_document_alloc(
        handle: u64,
        segment: i32,
        doc: i32,
        out_ptr: *mut *mut u8,
        out_len: *mut usize,
    ) -> i32;
    fn ffi_jvm_reader_doc_values_alloc(
        handle: u64,
        segment: i32,
        doc: i32,
        fields: *const i32,
        count: usize,
        out_ptr: *mut *mut u8,
        out_len: *mut usize,
    ) -> i32;
    fn ffi_jvm_free_bytes(ptr: *mut u8, len: usize);
    fn ffi_jvm_last_error(out_ptr: *mut *mut u8, out_len: *mut usize) -> i32;
}

fn check(status: i32, what: &str) {
    if status != 0 {
        let (mut ptr, mut len) = (std::ptr::null_mut(), 0usize);
        // SAFETY: live out-pointers; the message stays the library's.
        let msg = unsafe {
            ffi_jvm_last_error(&mut ptr, &mut len);
            if ptr.is_null() {
                String::new()
            } else {
                String::from_utf8_lossy(std::slice::from_raw_parts(ptr, len)).into_owned()
            }
        };
        panic!("{what}: status {status}: {msg}");
    }
}

/// A little-endian blob, as the plugin's encoders build them.
#[derive(Default, Clone)]
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
    fn f64(&mut self, v: f64) -> &mut Self {
        self.0.extend_from_slice(&v.to_bits().to_le_bytes());
        self
    }
    fn bytes(&mut self, b: &[u8]) -> &mut Self {
        self.i32(b.len() as i32);
        self.0.extend_from_slice(b);
        self
    }
    fn s(&mut self, s: &str) -> &mut Self {
        self.bytes(s.as_bytes())
    }
}

fn term_query(field: &str, term: &str) -> Vec<u8> {
    let mut b = Blob::default();
    b.u8(0).s(field).s(term);
    b.0
}

fn match_all() -> Vec<u8> {
    vec![2, 5]
}

/// `#term(body) #num:[lo, hi]`, a filter-only boolean.
fn term_and_range(term: &str, lo: i64, hi: i64) -> Vec<u8> {
    let mut b = Blob::default();
    b.u8(2).u8(1).i32(0).i32(2);
    b.u8(1).u8(0).s("body").s(term);
    b.u8(1).u8(11).s("num").i64(lo).i64(hi);
    b.0
}

// Node tags of jvm_aggs.
const METRIC: u8 = 0;
const CARDINALITY: u8 = 1;
const TERMS: u8 = 2;
const HISTOGRAM: u8 = 3;
const DATE_HISTOGRAM: u8 = 4;
const RANGE: u8 = 5;
const FILTERS: u8 = 6;
const NUMERIC_TERMS: u8 = 8;
const COMPOSITE: u8 = 10;
const PERCENTILES: u8 = 11;
const TOP_HITS: u8 = 12;

fn stats(b: &mut Blob, field: &str) {
    b.u8(METRIC).u8(0).u8(0).u8(31).s(field);
}

fn terms(b: &mut Blob, field: &str, size: i32) {
    b.u8(TERMS).s(field).i32(size).u8(0).u8(0);
}

fn tree(nodes: impl FnOnce(&mut Blob)) -> Vec<u8> {
    let mut b = Blob::default();
    nodes(&mut b);
    b.u8(0).i32(0);
    b.0
}

fn agg_cases() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        (
            "terms_keyword",
            tree(|b| {
                b.u8(1);
                terms(b, "keyword", 25);
                b.u8(0);
            }),
        ),
        (
            "terms_cat",
            tree(|b| {
                b.u8(1);
                terms(b, "cat", 25);
                b.u8(0);
            }),
        ),
        (
            "terms_cat>stats",
            tree(|b| {
                b.u8(1);
                terms(b, "cat", 25);
                b.u8(1);
                stats(b, "num");
            }),
        ),
        (
            "terms_keyword>terms_cat",
            tree(|b| {
                b.u8(1);
                terms(b, "keyword", 10);
                b.u8(1);
                terms(b, "cat", 5);
                b.u8(0);
            }),
        ),
        (
            "stats",
            tree(|b| {
                b.u8(1);
                stats(b, "num");
            }),
        ),
        (
            "histogram>stats",
            tree(|b| {
                b.u8(1)
                    .u8(HISTOGRAM)
                    .s("num")
                    .u8(0)
                    .f64(10_000.0)
                    .f64(0.0)
                    .u8(0);
                b.u8(1);
                stats(b, "num");
            }),
        ),
        (
            "date_histogram_fixed",
            tree(|b| {
                // `num` read as epoch millis: 1000 buckets of 1 s.
                b.u8(1)
                    .u8(DATE_HISTOGRAM)
                    .s("num")
                    .u8(8)
                    .i64(1000)
                    .i64(0)
                    .i64(0)
                    .u8(0);
                b.u8(0);
            }),
        ),
        (
            "date_histogram>terms_cat",
            tree(|b| {
                b.u8(1)
                    .u8(DATE_HISTOGRAM)
                    .s("num")
                    .u8(8)
                    .i64(100_000)
                    .i64(0)
                    .i64(0)
                    .u8(0);
                b.u8(1);
                terms(b, "cat", 5);
                b.u8(0);
            }),
        ),
        (
            "range",
            tree(|b| {
                b.u8(1).u8(RANGE).s("num").u8(0).i32(4);
                for (lo, hi) in [
                    (f64::NEG_INFINITY, 100_000.0),
                    (100_000.0, 500_000.0),
                    (500_000.0, 900_000.0),
                    (900_000.0, f64::INFINITY),
                ] {
                    b.f64(lo).f64(hi);
                }
                b.u8(1);
                stats(b, "num");
            }),
        ),
        (
            "numeric_terms",
            tree(|b| {
                b.u8(1).u8(NUMERIC_TERMS).s("num").u8(0).i32(25).u8(0);
                b.u8(0);
            }),
        ),
        (
            "cardinality_keyword",
            tree(|b| {
                b.u8(1).u8(CARDINALITY).u8(0).u8(14).s("keyword");
            }),
        ),
        (
            "cardinality_num",
            tree(|b| {
                b.u8(1).u8(CARDINALITY).u8(1).u8(14).s("num");
            }),
        ),
        (
            "percentiles",
            tree(|b| {
                b.u8(1).u8(PERCENTILES).u8(0).f64(100.0).s("num");
            }),
        ),
        (
            "terms_cat>percentiles",
            tree(|b| {
                b.u8(1);
                terms(b, "cat", 10);
                b.u8(1).u8(PERCENTILES).u8(0).f64(100.0).s("num");
            }),
        ),
        (
            "terms_cat>top_hits",
            tree(|b| {
                b.u8(1);
                terms(b, "cat", 10);
                b.u8(1).u8(TOP_HITS).i32(3).u8(0);
            }),
        ),
        (
            "filters>stats",
            tree(|b| {
                b.u8(1).u8(FILTERS).i32(3);
                for t in ["t1", "t5", "t20"] {
                    b.bytes(&term_query("body", t));
                }
                b.u8(1).u8(1);
                stats(b, "num");
            }),
        ),
        (
            "composite_cat_keyword",
            tree(|b| {
                b.u8(1).u8(COMPOSITE).i32(100).u8(2);
                b.s("cat").u8(0).u8(0).u8(0);
                b.s("keyword").u8(0).u8(0).u8(0);
                b.u8(0).i32(0);
                b.u8(0);
            }),
        ),
    ]
}

const SORT_LONG: u8 = 2;
const SORT_STRING: u8 = 6;

fn sort_blob(keys: &[(u8, bool, &str)]) -> Vec<u8> {
    let mut b = Blob::default();
    b.u8(keys.len() as u8);
    for &(ty, rev, field) in keys {
        b.u8(ty).u8(u8::from(rev)).s(field);
        b.i64(if ty == SORT_STRING { 1 } else { i64::MAX });
    }
    b.u8(0).u8(0).i32(0);
    b.0
}

struct Row {
    calls: u64,
    us: f64,
    allocs: f64,
    bytes: f64,
    peak_kib: u64,
}

fn measure(budget: Duration, mut f: impl FnMut() -> u64) -> (Row, u64) {
    let warm = Instant::now();
    let mut digest = 0u64;
    while warm.elapsed() < Duration::from_millis(300) {
        digest = f();
    }
    let live = LIVE.load(Relaxed);
    PEAK.store(live, Relaxed);
    let (a0, b0) = (ALLOCS.load(Relaxed), BYTES.load(Relaxed));
    let start = Instant::now();
    let mut calls = 0u64;
    while start.elapsed() < budget || calls < 3 {
        let d = f();
        assert_eq!(d, digest, "nondeterministic output");
        calls += 1;
    }
    let el = start.elapsed().as_secs_f64();
    let n = calls as f64;
    (
        Row {
            calls,
            us: el * 1e6 / n,
            allocs: (ALLOCS.load(Relaxed) - a0) as f64 / n,
            bytes: (BYTES.load(Relaxed) - b0) as f64 / n,
            peak_kib: (PEAK.load(Relaxed).saturating_sub(live)) / 1024,
        },
        digest,
    )
}

fn hash(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn main() {
    let dir = std::env::args().nth(1).expect("sweep <index dir> [filter]");
    let only = std::env::args().nth(2);
    let budget = Duration::from_millis(
        std::env::var("SWEEP_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1500),
    );
    let root = std::path::PathBuf::from(&dir);
    let files: Vec<String> = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    let generation = lucene_store::directory::last_commit_generation(&files).unwrap();
    let infos =
        std::fs::read(root.join(lucene_store::directory::segments_file_name(generation).unwrap()))
            .unwrap();
    let (max_docs, field_numbers) = {
        let reader = DirectoryReader::open(&FsDirectory::open(&root)).unwrap();
        let readers = reader.segment_readers();
        let max: Vec<i32> = readers.iter().map(|s| s.max_doc).collect();
        let infos = readers[0].field_infos();
        let num = |n: &str| infos.field_by_name(n).unwrap().number;
        (max, (num("cat"), num("keyword")))
    };
    let mut r = 0u64;
    // SAFETY: live slices and out-pointer; null live docs (no deletions).
    check(
        unsafe {
            ffi_open_jvm_reader(
                dir.as_ptr().cast(),
                dir.len(),
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
    if let Some(b) = std::env::var("SWEEP_BIG").ok().and_then(|v| v.parse().ok()) {
        BIG.store(b, Relaxed);
    }
    let wanted = |name: &str| only.as_deref().is_none_or(|f| name.contains(f));
    println!("case\tcalls\tus\tallocs\tbytes\tpeak_kib\tdigest");
    let print = |name: &str, (row, d): (Row, u64)| {
        println!(
            "{name}\t{}\t{:.2}\t{:.1}\t{:.0}\t{}\t{:016x}",
            row.calls, row.us, row.allocs, row.bytes, row.peak_kib, d
        );
    };

    let queries: Vec<(&str, Vec<u8>)> = vec![
        ("all", match_all()),
        ("t0", term_query("body", "t0")),
        ("t5", term_query("body", "t5")),
        ("t1k_range", term_and_range("t1k", 0, 500_000)),
    ];
    for (agg, tree) in agg_cases() {
        for (qn, q) in &queries {
            let name = format!("agg/{agg}/{qn}");
            if !wanted(&name) {
                continue;
            }
            let run = || {
                let (mut ptr, mut len) = (std::ptr::null_mut(), 0usize);
                // SAFETY: live slices and out-pointers; the buffer goes back.
                unsafe {
                    check(
                        ffi_jvm_reader_aggregate_tree_alloc(
                            r,
                            q.as_ptr(),
                            q.len(),
                            tree.as_ptr(),
                            tree.len(),
                            &mut ptr,
                            &mut len,
                        ),
                        &name,
                    );
                    let h = hash(std::slice::from_raw_parts(ptr, len));
                    ffi_jvm_free_bytes(ptr, len);
                    h
                }
            };
            print(&name, measure(budget, run));
        }
    }

    let sorts: Vec<(&str, Vec<u8>, usize)> = vec![
        ("num_asc", sort_blob(&[(SORT_LONG, false, "num")]), 1),
        ("num_desc", sort_blob(&[(SORT_LONG, true, "num")]), 1),
        (
            "keyword_asc",
            sort_blob(&[(SORT_STRING, false, "keyword")]),
            1,
        ),
        (
            "keyword_num",
            sort_blob(&[(SORT_STRING, false, "keyword"), (SORT_LONG, true, "num")]),
            2,
        ),
    ];
    for (sn, sort, keys) in &sorts {
        for (qn, q) in &queries {
            for top in [10usize, 100] {
                let name = format!("sort/{sn}/{qn}/top{top}");
                if !wanted(&name) {
                    continue;
                }
                let mut docs = vec![0i32; top];
                let mut values = vec![0i64; top * keys];
                let mut counts = [0i64; 4];
                let run = || {
                    let (mut tp, mut tl) = (std::ptr::null_mut(), 0usize);
                    // SAFETY: buffers of `top` hits and `top * keys` values.
                    unsafe {
                        check(
                            ffi_jvm_reader_search_sorted_alloc(
                                r,
                                q.as_ptr(),
                                q.len(),
                                sort.as_ptr(),
                                sort.len(),
                                top,
                                1000,
                                docs.as_mut_ptr(),
                                top,
                                values.as_mut_ptr(),
                                values.len(),
                                counts.as_mut_ptr(),
                                &mut tp,
                                &mut tl,
                            ),
                            &name,
                        );
                        let mut h = if tl > 0 {
                            hash(std::slice::from_raw_parts(tp, tl))
                        } else {
                            0
                        };
                        if !tp.is_null() {
                            ffi_jvm_free_bytes(tp, tl);
                        }
                        for d in &docs {
                            h = h.wrapping_mul(31).wrapping_add(*d as u64);
                        }
                        for v in &values {
                            h = h.wrapping_mul(31).wrapping_add(*v as u64);
                        }
                        h.wrapping_add(counts[0] as u64)
                    }
                };
                print(&name, measure(budget, run));
            }
        }
    }

    // Fetch: the top 100 of a scored search, as the fetch phase asks for
    // them: one call per hit (stored fields), and the doc values of `cat`
    // and `keyword` per hit (docvalue_fields).
    let mut hits = vec![0i32; 100];
    let mut scores = vec![0f32; 100];
    let (mut n, mut total, mut lower) = (0usize, 0i64, false);
    let q = term_query("body", "t5");
    // SAFETY: buffers of 100.
    check(
        unsafe {
            ffi_jvm_reader_search(
                r,
                q.as_ptr(),
                q.len(),
                100,
                1000,
                hits.as_mut_ptr(),
                scores.as_mut_ptr(),
                100,
                &mut n,
                &mut total,
                &mut lower,
            )
        },
        "search",
    );
    hits.truncate(n);
    // Global docs to (segment, doc).
    let mut bases = vec![0i32];
    for m in &max_docs {
        bases.push(bases.last().unwrap() + m);
    }
    let local: Vec<(i32, i32)> = hits
        .iter()
        .map(|&d| {
            let s = bases.partition_point(|&b| b <= d) - 1;
            (s as i32, d - bases[s])
        })
        .collect();
    let mut sorted_local = local.clone();
    sorted_local.sort_unstable();
    for (label, docs) in [("score_order", &local), ("doc_order", &sorted_local)] {
        let name = format!("fetch/stored/{label}/100");
        if wanted(&name) {
            let run = || {
                let mut h = 0u64;
                for &(s, d) in docs {
                    let (mut ptr, mut len) = (std::ptr::null_mut(), 0usize);
                    // SAFETY: live out-pointers; the buffer goes back.
                    unsafe {
                        check(
                            ffi_jvm_reader_document_alloc(r, s, d, &mut ptr, &mut len),
                            "fetch",
                        );
                        h = h.wrapping_mul(31) ^ hash(std::slice::from_raw_parts(ptr, len));
                        ffi_jvm_free_bytes(ptr, len);
                    }
                }
                h
            };
            print(&name, measure(budget, run));
        }
        let name = format!("fetch/doc_values/{label}/100");
        if wanted(&name) {
            let fields = [field_numbers.0, field_numbers.1];
            let run = || {
                let mut h = 0u64;
                for &(s, d) in docs {
                    let (mut ptr, mut len) = (std::ptr::null_mut(), 0usize);
                    // SAFETY: live slices and out-pointers; the buffer goes back.
                    unsafe {
                        check(
                            ffi_jvm_reader_doc_values_alloc(
                                r,
                                s,
                                d,
                                fields.as_ptr(),
                                fields.len(),
                                &mut ptr,
                                &mut len,
                            ),
                            "doc values",
                        );
                        h = h.wrapping_mul(31) ^ hash(std::slice::from_raw_parts(ptr, len));
                        ffi_jvm_free_bytes(ptr, len);
                    }
                }
                h
            };
            print(&name, measure(budget, run));
        }
    }
}
