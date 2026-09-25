//! `ffi_jvm_reader_search` with an arbitrary query blob and sizing, against
//! a real index: the decoder's bounds checks, the clause-tree rules, and every
//! search and count path the decoded query can reach.
#![no_main]
mod common;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let top_n = usize::from(data[0] % 12);
    let count_limit = i64::from(data[1] % 12) - 1;
    let rc = common::search(common::shared_jvm_reader(), &data[2..], top_n, count_limit);
    assert_ne!(rc, common::PANIC, "a query blob panicked");
});
