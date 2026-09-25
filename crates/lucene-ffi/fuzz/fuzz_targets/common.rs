//! Shared by every target: the C ABI as a JVM caller sees it, and one
//! reader over a real Java-written two-segment index, opened once.
#![allow(dead_code)]

use std::os::raw::c_char;
use std::sync::OnceLock;

// Link the library: its `#[no_mangle]` entry points are what is fuzzed, not
// its Rust API.
extern crate lucene_ffi;

/// `FfiStatus::Panic`. A caught panic is still a finding: every input must be
/// answered with a status the caller can act on, not with an unwind that
/// `catch_unwind` happened to stop.
pub const PANIC: i32 = 9;

extern "C" {
    pub fn ffi_open_jvm_reader(
        path: *const c_char,
        path_len: usize,
        infos: *const u8,
        infos_len: usize,
        generation: i64,
        previous: u64,
        expected_max_docs: *const i32,
        segment_count: usize,
        out_handle: *mut u64,
    ) -> i32;
    pub fn ffi_jvm_reader_set_live_docs(
        handle: u64,
        segment: usize,
        words: *const u64,
        words_len: usize,
    ) -> i32;
    pub fn ffi_jvm_reader_search(
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
    pub fn ffi_close_jvm_reader(handle: u64) -> i32;
    pub fn ffi_open_directory_reader(path: *const c_char, path_len: usize, out: *mut u64) -> i32;
    pub fn ffi_search_boolean_query_multi_segment(
        reader_handle: u64,
        clause_occurs: *const u8,
        clause_kinds: *const u8,
        clause_fields: *const *const c_char,
        clause_field_lens: *const usize,
        clause_terms: *const *const u8,
        clause_term_lens: *const usize,
        clause_parents: *const i32,
        clause_params: *const i32,
        clause_count: usize,
        minimum_should_match: i32,
        top_n: usize,
        out_scored_results_handle: *mut u64,
    ) -> i32;
    pub fn ffi_close_scored_results(handle: u64) -> i32;
}

pub const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../fixtures/data/multi_segment_scoring_index"
);

pub fn infos() -> &'static [u8] {
    static INFOS: OnceLock<Vec<u8>> = OnceLock::new();
    INFOS.get_or_init(|| std::fs::read(format!("{FIXTURE}/segments_2")).expect("segments_2"))
}

/// A fresh JVM reader over the fixture.
pub fn open_jvm_reader() -> u64 {
    let docs = [4i32, 4];
    let mut h = 0u64;
    let rc = unsafe {
        ffi_open_jvm_reader(
            FIXTURE.as_ptr().cast(),
            FIXTURE.len(),
            infos().as_ptr(),
            infos().len(),
            2,
            0,
            docs.as_ptr(),
            docs.len(),
            &mut h,
        )
    };
    assert_eq!(rc, 0, "the fixture must open");
    h
}

/// One JVM reader shared by every iteration (opening per input would make
/// the targets measure the open, not the input).
pub fn shared_jvm_reader() -> u64 {
    static H: OnceLock<u64> = OnceLock::new();
    *H.get_or_init(open_jvm_reader)
}

/// Runs `blob` with the given sizing; returns the status.
pub fn search(handle: u64, blob: &[u8], top_n: usize, count_limit: i64) -> i32 {
    let mut docs = vec![0i32; top_n];
    let mut scores = vec![0f32; top_n];
    let (mut n, mut total, mut lower) = (0usize, 0i64, false);
    let rc = unsafe {
        ffi_jvm_reader_search(
            handle,
            blob.as_ptr(),
            blob.len(),
            top_n,
            count_limit,
            docs.as_mut_ptr(),
            scores.as_mut_ptr(),
            top_n,
            &mut n,
            &mut total,
            &mut lower,
        )
    };
    if rc == 0 {
        assert!(n <= top_n, "more hits than asked for");
        assert!(docs[..n].iter().all(|&d| (0..8).contains(&d)), "a doc id outside the index");
        assert!(total == -1 || total >= n as i64, "fewer total hits than returned hits");
        assert!(total <= 8, "more total hits than documents");
    }
    rc
}
