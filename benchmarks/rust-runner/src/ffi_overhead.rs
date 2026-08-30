//! T1.5: what does crossing the C ABI actually cost per search call?
//!
//! `PLAN.md` §2 Phase 4 budgets <1µs of FFI overhead per search. This measures
//! the C-ABI half of that -- handle lookup, argument marshalling, result buffer
//! construction. The JNI/FFM half is confirmed end-to-end in M2.
//!
//! The measurement is decomposed rather than blended, because
//! `ffi_search_term_query_multi_segment` calls `open_segments()` inside its own
//! per-call guard. Attributing that setup cost to "FFI overhead" would be
//! wrong; it is a separate, and much larger, finding.
//!
//!   A. direct, segments hoisted    -- what a well-written in-process caller does
//!   B. direct, open_segments/call  -- the work the FFI path actually repeats
//!   C. through the C ABI           -- B plus the boundary
//!
//! So: boundary cost = C - B, and per-call setup cost = B - A.
//!
//! D adds the question a single-threaded measurement cannot answer: does the
//! boundary let a multi-threaded JVM caller actually run searches in
//! parallel? Every FFI query holds its registry guard for the whole search
//! (the handle lookup borrows the segment/reader out of the guard), so this
//! is a property of `registry.rs`'s lock choice, not of the search code. It
//! reports the speedup of T threads over one -- ~1x means the boundary is
//! serializing every caller thread (which a `Mutex` registry did), ~Tx means
//! it is not.

// Link the cdylib's rlib so its #[no_mangle] symbols resolve: the extern "C"
// declarations below create no Rust-level dependency on their own.
use lucene_ffi as _;

use std::ffi::c_char;
use std::time::Instant;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::query::TermQuery;
use lucene_search::search_term_query_multi_segment;
use lucene_store::MmapDirectory;

// Declared rather than called through Rust module paths: lucene-ffi's modules
// are private and its surface is the C ABI. Going through the exported symbols
// is also the more faithful measurement -- this is the same linkage a JNI or
// FFM caller resolves.
extern "C" {
    fn ffi_open_directory_reader(path: *const c_char, path_len: usize, out: *mut u64) -> i32;
    fn ffi_search_term_query_multi_segment(
        reader_handle: u64,
        field: *const c_char,
        field_len: usize,
        term: *const u8,
        term_len: usize,
        top_n: usize,
        out: *mut u64,
    ) -> i32;
    fn ffi_close_scored_results(handle: u64) -> i32;
}

const TOP_N: usize = 50;
const ITERS: usize = 2000;
/// Default fan-out for section D. Overridable with `FFI_OVERHEAD_THREADS`,
/// because how much a registry-contention change is worth depends on how many
/// caller threads are actually hitting the boundary at once -- at 4 threads on
/// a 20-core box the locks are barely contended, and the interesting shape is
/// at fan-outs at or above the core count. Added by the M2 sweep batch
/// `c13-ffi-surface` while measuring the sharded results registries.
const DEFAULT_THREADS: usize = 4;
// D runs far more iterations than A-C: at ~0.4us/call, 2000 calls is under a
// millisecond of work, which thread spawn/join alone would dominate.
const CONCURRENT_ITERS: usize = 400_000;

fn main() {
    let threads: usize = std::env::var("FFI_OVERHEAD_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|t| *t > 0)
        .unwrap_or(DEFAULT_THREADS);
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: ffi-overhead <index-dir> <field> <term>");
        std::process::exit(2);
    }
    let (dir_path, field, term) = (&a[1], &a[2], &a[3]);

    let dir = MmapDirectory::open(dir_path.clone());
    let reader = DirectoryReader::open(&dir).expect("open index");
    let query = TermQuery {
        field: field.clone(),
        term: term.clone().into_bytes(),
    };

    // A: segments opened once, outside the loop.
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();
    let norms: Vec<Option<&FieldNorms<'_>>> = vec![None; segments.len()];
    let t = Instant::now();
    for _ in 0..ITERS {
        std::hint::black_box(
            search_term_query_multi_segment(&segments, &query, &norms, TOP_N).expect("direct"),
        );
    }
    let a_ns = t.elapsed().as_nanos() as f64 / ITERS as f64;
    drop(segments);
    drop(opened);

    // B: segments reopened every call, matching what the FFI path does.
    let t = Instant::now();
    for _ in 0..ITERS {
        let opened = reader.open_segments().expect("open segments");
        let segments = opened.as_open_segments();
        let norms: Vec<Option<&FieldNorms<'_>>> = vec![None; segments.len()];
        std::hint::black_box(
            search_term_query_multi_segment(&segments, &query, &norms, TOP_N).expect("direct"),
        );
    }
    let b_ns = t.elapsed().as_nanos() as f64 / ITERS as f64;

    // C: the same query through the exported C ABI.
    let mut handle: u64 = 0;
    let rc = unsafe {
        ffi_open_directory_reader(
            dir_path.as_ptr() as *const c_char,
            dir_path.len(),
            &mut handle,
        )
    };
    assert_eq!(rc, 0, "ffi_open_directory_reader failed: {rc}");

    let t = Instant::now();
    for _ in 0..ITERS {
        let mut results: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_multi_segment(
                handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                TOP_N,
                &mut results,
            )
        };
        assert_eq!(rc, 0, "ffi search failed: {rc}");
        unsafe { ffi_close_scored_results(results) };
    }
    let c_ns = t.elapsed().as_nanos() as f64 / ITERS as f64;

    println!("iterations              : {ITERS}");
    println!("A direct, hoisted       : {:>10.0} ns/call", a_ns);
    println!("B direct, reopen/call   : {:>10.0} ns/call", b_ns);
    println!("C through the C ABI     : {:>10.0} ns/call", c_ns);
    println!();
    println!(
        "FFI boundary  (C - B)   : {:>10.0} ns/call   [budget: <1000 ns]",
        c_ns - b_ns
    );
    println!(
        "per-call setup (B - A)  : {:>10.0} ns/call   (open_segments repeated per search)",
        b_ns - a_ns
    );

    // D: the same C-ABI call from THREADS threads at once. `handle` is a
    // plain `u64` (see the ffi-safety skill's opaque-handle rule), so every
    // thread uses the same reader handle -- exactly what a JVM caller with a
    // shared reader does.
    // Single-threaded baseline at the same iteration count, so the speedup
    // compares like with like (C's ITERS is far smaller).
    let t = Instant::now();
    for _ in 0..CONCURRENT_ITERS {
        let mut results: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_multi_segment(
                handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                TOP_N,
                &mut results,
            )
        };
        assert_eq!(rc, 0, "ffi search failed: {rc}");
        unsafe { ffi_close_scored_results(results) };
    }
    let serial_wall_ns = t.elapsed().as_nanos() as f64;

    // Truncating division: the threads together run `per_thread * threads`
    // calls, which is `CONCURRENT_ITERS` only when `threads` divides it. Both
    // the per-call figure and the speedup below divide by the *actual* count,
    // so an odd `FFI_OVERHEAD_THREADS` cannot silently overstate either.
    let per_thread = CONCURRENT_ITERS / threads;
    let concurrent_calls = per_thread * threads;
    let t = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                for _ in 0..per_thread {
                    let mut results: u64 = 0;
                    let rc = unsafe {
                        ffi_search_term_query_multi_segment(
                            handle,
                            field.as_ptr() as *const c_char,
                            field.len(),
                            term.as_ptr(),
                            term.len(),
                            TOP_N,
                            &mut results,
                        )
                    };
                    assert_eq!(rc, 0, "ffi search failed: {rc}");
                    unsafe { ffi_close_scored_results(results) };
                }
            });
        }
    });
    let d_wall_ns = t.elapsed().as_nanos() as f64;
    println!();
    println!("D iterations            : {CONCURRENT_ITERS}");
    println!(
        "D 1 thread,  C ABI      : {:>10.0} ns/call wall",
        serial_wall_ns / CONCURRENT_ITERS as f64
    );
    println!(
        "D {threads} threads, C ABI      : {:>10.0} ns/call wall",
        d_wall_ns / concurrent_calls as f64
    );
    println!(
        "concurrency speedup     : {:>10.2}x         [1.00x = the boundary serializes callers]",
        (serial_wall_ns / CONCURRENT_ITERS as f64) / (d_wall_ns / concurrent_calls as f64)
    );
}
