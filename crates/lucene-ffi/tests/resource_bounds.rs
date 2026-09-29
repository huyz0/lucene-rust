//! **Nothing the C ABI hands out outlives its release**, measured in bytes.
//!
//! The time-based soak's replacement for the native side: every entry point
//! the JVM calls in a loop -- open and close a reader, reopen one from the
//! previous, search, take an error message and free it -- runs thousands of
//! times, and the heap must not grow. A leak of one handle, one result buffer
//! or one error string per call shows as bytes per iteration here, not as RSS
//! creeping up over days.
//!
//! One test in its own binary on purpose: a counting global allocator sees the
//! whole process, and the descriptor and thread counts are process-wide too,
//! so nothing may run beside it. The entry points are declared as the JVM
//! sees them, the way the fuzz targets do.
//!
//! What it cannot catch: memory mapped rather than allocated (the engine
//! writer's refresh-loop test counts mappings), and growth bounded by a cache
//! that fills during the warm-up and then stays full -- by design.
#![cfg(target_os = "linux")]
#![allow(clippy::arithmetic_side_effects)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::os::raw::c_char;
use std::sync::atomic::{AtomicIsize, Ordering};

// Link the library: its `#[no_mangle]` entry points are what is measured.
extern crate lucene_ffi;

/// The system allocator, counting the bytes currently allocated.
struct Counting;

static LIVE: AtomicIsize = AtomicIsize::new(0);

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged, so `System`'s guarantees are this allocator's; the counter is
// bookkeeping only.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract for `alloc`, passed through.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract for `alloc_zeroed`, passed through.
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract for `dealloc`, passed through.
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller's contract for `realloc`, passed through.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            LIVE.fetch_add(
                new_size as isize - layout.size() as isize,
                Ordering::Relaxed,
            );
        }
        p
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
    fn ffi_close_jvm_reader(handle: u64) -> i32;
    fn ffi_jvm_last_error(out_ptr: *mut *mut u8, out_len: *mut usize) -> i32;
    fn ffi_jvm_free_bytes(ptr: *mut u8, len: usize);
    fn ffi_open_directory_reader(path: *const c_char, path_len: usize, out: *mut u64) -> i32;
    fn ffi_close_directory_reader(handle: u64) -> i32;
    fn ffi_search_term_query_multi_segment(
        reader_handle: u64,
        field: *const c_char,
        field_len: usize,
        term: *const u8,
        term_len: usize,
        top_n: usize,
        out_scored_results_handle: *mut u64,
    ) -> i32;
    fn ffi_close_scored_results(handle: u64) -> i32;
}

/// A real Java-written two-segment index (`GenMultiSegmentScoring`): four
/// documents per segment, `body` holding "fox" in four of them.
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/data/multi_segment_scoring_index"
);
const MAX_DOCS: [i32; 2] = [4, 4];

/// Iterations per measured window.
const N: usize = 2000;
/// Growth a window may show without being a leak: allocator and thread-local
/// noise, far below one leaked byte per iteration.
const SLACK: isize = 1024;

fn open_fds() -> usize {
    std::fs::read_dir("/proc/self/fd").unwrap().count()
}

fn threads() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|l| l.strip_prefix("Threads:"))
        .and_then(|v| v.trim().parse().ok())
        .unwrap()
}

/// Runs `op` for a warm-up, then two windows of [`N`] calls; neither window
/// may grow the heap past [`SLACK`] or change the descriptor or thread count.
fn steady(name: &str, mut op: impl FnMut()) {
    for _ in 0..N / 4 {
        op();
    }
    let (fds, threads0) = (open_fds(), threads());
    for window in 0..2 {
        let before = LIVE.load(Ordering::Relaxed);
        for _ in 0..N {
            op();
        }
        let grown = LIVE.load(Ordering::Relaxed) - before;
        eprintln!("{name}: window {window}: {grown:+} bytes over {N} calls");
        assert!(
            grown <= SLACK,
            "{name}: the heap grew {grown} bytes over {N} calls (window {window})"
        );
    }
    assert_eq!(open_fds(), fds, "{name}: descriptors leaked");
    assert_eq!(threads(), threads0, "{name}: threads leaked");
}

fn open_jvm(previous: u64) -> u64 {
    let infos = std::fs::read(format!("{FIXTURE}/segments_2")).unwrap();
    let mut handle = 0;
    // SAFETY: live slices of the stated lengths, a live out-pointer, no
    // live-docs words (null counts).
    let rc = unsafe {
        ffi_open_jvm_reader(
            FIXTURE.as_ptr().cast(),
            FIXTURE.len(),
            infos.as_ptr(),
            infos.len(),
            2,
            previous,
            MAX_DOCS.as_ptr(),
            MAX_DOCS.len(),
            std::ptr::null(),
            std::ptr::null(),
            &mut handle,
        )
    };
    assert_eq!(rc, 0);
    handle
}

/// `QUERY_TERM` over `body:fox`, as the JVM encodes it.
fn fox() -> Vec<u8> {
    let mut b = vec![0u8];
    for part in [&b"body"[..], &b"fox"[..]] {
        b.extend_from_slice(&(part.len() as i32).to_le_bytes());
        b.extend_from_slice(part);
    }
    b
}

/// `(status, total hits)` of `query` on `handle`.
fn search(handle: u64, query: &[u8]) -> (i32, i64) {
    let (mut docs, mut scores) = ([0i32; 10], [0f32; 10]);
    let (mut n, mut total, mut lower) = (0usize, 0i64, false);
    // SAFETY: live buffers of 10 and live out-pointers.
    let rc = unsafe {
        ffi_jvm_reader_search(
            handle,
            query.as_ptr(),
            query.len(),
            10,
            i64::MAX,
            docs.as_mut_ptr(),
            scores.as_mut_ptr(),
            10,
            &mut n,
            &mut total,
            &mut lower,
        )
    };
    (rc, total)
}

#[test]
fn every_native_resource_is_released_by_its_close() {
    let query = fox();

    steady("jvm reader open, search, close", || {
        let h = open_jvm(0);
        assert_eq!(search(h, &query), (0, 4));
        // SAFETY: closing a handle this closure opened.
        assert_eq!(unsafe { ffi_close_jvm_reader(h) }, 0);
    });

    // The NRT chain: each reader opened from the previous one, which then
    // closes -- the shared segments must go with the last handle naming them.
    let mut current = open_jvm(0);
    steady("jvm reader reopen from the previous", || {
        let next = open_jvm(current);
        assert_eq!(search(next, &query), (0, 4));
        // SAFETY: closing the handle this chain opened last round.
        assert_eq!(unsafe { ffi_close_jvm_reader(current) }, 0);
        current = next;
    });
    // SAFETY: the chain's last handle.
    assert_eq!(unsafe { ffi_close_jvm_reader(current) }, 0);

    // Error paths: a closed handle and a malformed query, each leaving a
    // message the JVM takes as an owned buffer and frees.
    let closed = open_jvm(0);
    // SAFETY: closing a handle opened just above.
    assert_eq!(unsafe { ffi_close_jvm_reader(closed) }, 0);
    let live = open_jvm(0);
    steady("errors and their messages", || {
        for (handle, q) in [(closed, &query[..]), (live, &query[..3])] {
            assert_ne!(search(handle, q).0, 0);
            let (mut ptr, mut len) = (std::ptr::null_mut(), 0usize);
            // SAFETY: live out-pointers; the buffer is freed right after.
            assert_eq!(unsafe { ffi_jvm_last_error(&mut ptr, &mut len) }, 0);
            assert!(len > 0);
            // SAFETY: exactly the pointer and length `ffi_jvm_last_error` gave.
            unsafe { ffi_jvm_free_bytes(ptr, len) };
        }
    });
    // SAFETY: closing a handle opened above.
    assert_eq!(unsafe { ffi_close_jvm_reader(live) }, 0);

    // The older reader-handle surface: open, search into a results handle,
    // close both.
    steady("directory reader and scored results", || {
        let mut reader = 0;
        // SAFETY: a live path of the stated length and a live out-pointer.
        let rc = unsafe {
            ffi_open_directory_reader(FIXTURE.as_ptr().cast(), FIXTURE.len(), &mut reader)
        };
        assert_eq!(rc, 0);
        let mut results = 0;
        // SAFETY: live field and term slices and a live out-pointer.
        let rc = unsafe {
            ffi_search_term_query_multi_segment(
                reader,
                b"body".as_ptr().cast(),
                4,
                b"fox".as_ptr(),
                3,
                10,
                &mut results,
            )
        };
        assert_eq!(rc, 0);
        // SAFETY: closing handles this closure opened.
        unsafe {
            assert_eq!(ffi_close_scored_results(results), 0);
            assert_eq!(ffi_close_directory_reader(reader), 0);
        }
    });
}
