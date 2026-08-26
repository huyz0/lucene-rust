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

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: ffi-overhead <index-dir> <field> <term>");
        std::process::exit(2);
    }
    let (dir_path, field, term) = (&a[1], &a[2], &a[3]);

    let dir = MmapDirectory::open(dir_path.clone());
    let reader = DirectoryReader::open(&dir).expect("open index");
    let query = TermQuery { field: field.clone(), term: term.clone().into_bytes() };

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
    println!("FFI boundary  (C - B)   : {:>10.0} ns/call   [budget: <1000 ns]", c_ns - b_ns);
    println!("per-call setup (B - A)  : {:>10.0} ns/call   (open_segments repeated per search)", b_ns - a_ns);
}
