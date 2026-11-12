//! Peak native heap and time per call of aggregations that make many
//! buckets, through the C ABI the plugin calls, over an index with a unique
//! `id` keyword, a 1,000-value `host` keyword, a long `n` and a date `ts`
//! (`opensearch-plugin/e2e/memory_bounds.py`'s corpus).
//!
//! `sort/...` cases sort the top 100 hits; with `SORT_STREAM=1` the node's
//! off-heap bound is set so small that no sort column is decoded, so they
//! read the doc values per query instead of holding decoded columns (the
//! `columns_kib` column says what the decoded columns hold).
//!
//! `agg-memory <index dir> [case-substring]`; `SWEEP_MS` sets each case's
//! measuring time (default 1500 ms). Each row: case, calls, microseconds
//! per call, allocations per call, bytes allocated per call, peak live heap
//! above the start of the case (KiB), and a digest of the output.

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

const SORT_LONG: u8 = 2;
const SORT_STRING: u8 = 6;

fn sort_blob(ty: u8, reverse: bool, field: &str) -> Vec<u8> {
    let mut b = Blob::default();
    b.u8(1).u8(ty).u8(u8::from(reverse)).s(field);
    b.i64(if ty == SORT_STRING { 1 } else { i64::MAX });
    b.u8(0).u8(0).i32(0);
    b.0
}

fn match_all() -> Vec<u8> {
    vec![2, 5]
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

const METRIC: u8 = 0;
const TERMS: u8 = 2;
const DATE_HISTOGRAM: u8 = 4;
const NUMERIC_TERMS: u8 = 8;
const NEED_COUNT_SUM: u8 = 3;
const NEED_MAX: u8 = 16;
const NEED_ALL: u8 = 31;

fn metric(b: &mut Blob, needs: u8, field: &str) {
    b.u8(METRIC).u8(0).u8(0).u8(needs).s(field);
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

fn date_histogram(b: &mut Blob, field: &str, interval_ms: i64) {
    b.u8(DATE_HISTOGRAM)
        .s(field)
        .u8(8)
        .i64(interval_ms)
        .i64(0)
        .i64(0)
        .u8(0);
}

fn cases() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("terms_id", tree(|b| {
            b.u8(1);
            terms(b, "id", 60_000);
            b.u8(0);
        })),
        ("terms_id>avg", tree(|b| {
            b.u8(1);
            terms(b, "id", 60_000);
            b.u8(1);
            metric(b, NEED_COUNT_SUM, "n");
        })),
        ("terms_id>stats", tree(|b| {
            b.u8(1);
            terms(b, "id", 60_000);
            b.u8(1);
            metric(b, NEED_ALL, "n");
        })),
        ("numeric_terms_n>avg", tree(|b| {
            b.u8(1).u8(NUMERIC_TERMS).s("n").u8(0).i32(60_000).u8(0);
            b.u8(1);
            metric(b, NEED_COUNT_SUM, "n");
        })),
        ("date_histogram_7ms>max", tree(|b| {
            b.u8(1);
            date_histogram(b, "ts", 7);
            b.u8(1);
            metric(b, NEED_MAX, "n");
        })),
        ("date_histogram_1s>avg", tree(|b| {
            b.u8(1);
            date_histogram(b, "ts", 1000);
            b.u8(1);
            metric(b, NEED_COUNT_SUM, "n");
        })),
        ("terms_host>avg", tree(|b| {
            b.u8(1);
            terms(b, "host", 10);
            b.u8(1);
            metric(b, NEED_COUNT_SUM, "n");
        })),
        ("terms_host>terms_id", tree(|b| {
            b.u8(1);
            terms(b, "host", 1000);
            b.u8(1);
            terms(b, "id", 50);
            b.u8(0);
        })),
    ]
}

fn main() {
    let dir = std::env::args().nth(1).expect("agg-memory <index dir> [filter]");
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
    let max_docs: Vec<i32> = DirectoryReader::open(&FsDirectory::open(&root))
        .unwrap()
        .segment_readers()
        .iter()
        .map(|s| s.max_doc)
        .collect();
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
    if std::env::var("SORT_STREAM").is_ok_and(|v| v == "1") {
        lucene_util::memory::set_process_limit(8);
    }
    println!("case\tcalls\tus\tallocs\tbytes\tpeak_kib\tdigest\tcolumns_kib");
    let q = match_all();
    let sorts: Vec<(&str, Vec<u8>, Vec<u8>)> = vec![
        ("sort/ts_desc/all", sort_blob(SORT_LONG, true, "ts"), match_all()),
        ("sort/n/host", sort_blob(SORT_LONG, false, "n"), term_query("host", "h7")),
        ("sort/id/all", sort_blob(SORT_STRING, false, "id"), match_all()),
    ];
    for (name, sort, query) in &sorts {
        if only.as_deref().is_some_and(|f| !name.contains(f)) {
            continue;
        }
        let top = 100usize;
        let mut docs = vec![0i32; top];
        let mut values = vec![0i64; top];
        let mut counts = [0i64; 6];
        let run = || {
            let (mut tp, mut tl) = (std::ptr::null_mut(), 0usize);
            // SAFETY: buffers of `top` hits and values, six count slots.
            unsafe {
                check(
                    ffi_jvm_reader_search_sorted_alloc(
                        r,
                        query.as_ptr(),
                        query.len(),
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
                    name,
                );
                let mut h = if tl > 0 { hash(std::slice::from_raw_parts(tp, tl)) } else { 0 };
                if !tp.is_null() {
                    ffi_jvm_free_bytes(tp, tl);
                }
                for d in &docs {
                    h = h.rotate_left(7) ^ *d as u64;
                }
                h
            }
        };
        let (row, d) = measure(budget, run);
        println!(
            "{name}\t{}\t{:.0}\t{:.1}\t{:.0}\t{}\t{:016x}\t{}",
            row.calls,
            row.us,
            row.allocs,
            row.bytes,
            row.peak_kib,
            d,
            lucene_search::sort_column_bytes() >> 10
        );
    }
    for (name, tree) in cases() {
        if only.as_deref().is_some_and(|f| !name.contains(f)) {
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
                    name,
                );
                let h = hash(std::slice::from_raw_parts(ptr, len));
                ffi_jvm_free_bytes(ptr, len);
                h
            }
        };
        let (row, d) = measure(budget, run);
        println!(
            "{name}\t{}\t{:.0}\t{:.1}\t{:.0}\t{}\t{:016x}",
            row.calls, row.us, row.allocs, row.bytes, row.peak_kib, d
        );
    }
}
