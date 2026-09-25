//! The occur-tagged clause-array format through the plain C ABI
//! (`ffi_search_boolean_query_multi_segment`): arbitrary occurs, kinds,
//! parents and params -- the arrays a C or JNI caller builds by hand, where
//! the JVM reader's blob decoder is not in the way.
#![no_main]
mod common;

use std::os::raw::c_char;
use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;

const FIELDS: [&str; 3] = ["body", "id", ""];
const TERMS: [&str; 4] = ["fox", "dog", "the", "zzz"];

fuzz_target!(|data: &[u8]| {
    static READER: OnceLock<u64> = OnceLock::new();
    let reader = *READER.get_or_init(|| {
        let mut h = 0u64;
        let rc = unsafe { common::ffi_open_directory_reader(common::FIXTURE.as_ptr().cast(), common::FIXTURE.len(), &mut h) };
        assert_eq!(rc, 0);
        h
    });
    // Six bytes per clause: occur, kind, parent, param, field, term.
    let clauses: Vec<&[u8]> = data.chunks_exact(6).take(40).collect();
    let n = clauses.len();
    let occurs: Vec<u8> = clauses.iter().map(|c| c[0] % 5).collect();
    let kinds: Vec<u8> = clauses.iter().map(|c| c[1] % 5).collect();
    let parents: Vec<i32> = clauses.iter().map(|c| i32::from(c[2] as i8)).collect();
    let params: Vec<i32> = clauses
        .iter()
        .map(|c| match c[3] % 4 {
            0 => 0,
            1 => i32::from(c[3] >> 2),
            2 => 1.5f32.to_bits() as i32,
            _ => f32::NAN.to_bits() as i32,
        })
        .collect();
    let fields: Vec<&str> = clauses.iter().map(|c| FIELDS[usize::from(c[4]) % FIELDS.len()]).collect();
    let terms: Vec<&str> = clauses.iter().map(|c| TERMS[usize::from(c[5]) % TERMS.len()]).collect();
    let field_ptrs: Vec<*const c_char> = fields.iter().map(|f| f.as_ptr().cast()).collect();
    let field_lens: Vec<usize> = fields.iter().map(|f| f.len()).collect();
    let term_ptrs: Vec<*const u8> = terms.iter().map(|t| t.as_ptr()).collect();
    let term_lens: Vec<usize> = terms.iter().map(|t| t.len()).collect();
    let msm = data.first().map_or(0, |b| i32::from(*b % 4));
    let mut results = 0u64;
    let rc = unsafe {
        common::ffi_search_boolean_query_multi_segment(
            reader,
            occurs.as_ptr(),
            kinds.as_ptr(),
            field_ptrs.as_ptr(),
            field_lens.as_ptr(),
            term_ptrs.as_ptr(),
            term_lens.as_ptr(),
            parents.as_ptr(),
            params.as_ptr(),
            n,
            msm,
            10,
            &mut results,
        )
    };
    assert_ne!(rc, common::PANIC, "clause arrays panicked");
    if rc == 0 {
        unsafe { common::ffi_close_scored_results(results) };
    }
});
