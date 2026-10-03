//! `ffi_jvm_reader_search_sorted` with an arbitrary query blob and sort blob
//! against a real index: the sort decoder's bounds checks (every key type,
//! the geo-distance keys' origins and comparators included), `search_after`,
//! the options, slices and index-sort flags, and the sorted search paths
//! the decoded sort can reach.
#![no_main]
mod common;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let top_n = 1 + usize::from(data[0] % 12);
    let count_limit = i64::from(data[1] % 12) - 1;
    let rest = &data[3..];
    // The first `data[2]` bytes (at most all of them) are the query blob.
    let split = usize::from(data[2]).min(rest.len());
    let (query, sort) = rest.split_at(split);
    let rc = common::search_sorted(common::shared_jvm_reader(), query, sort, top_n, count_limit);
    assert_ne!(rc, common::PANIC, "a query or sort blob panicked");
});
