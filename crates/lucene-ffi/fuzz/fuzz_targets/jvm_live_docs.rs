//! `ffi_jvm_reader_set_live_docs` with arbitrary words and segment, then a
//! search: the live-docs validation, and that whatever it accepts is safe to
//! search and count under.
#![no_main]
mod common;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let h = common::open_jvm_reader();
    let segment = usize::from(data[0] % 3);
    let words: Vec<u64> = data[1..]
        .chunks(8)
        .map(|c| {
            let mut b = [0u8; 8];
            b[..c.len()].copy_from_slice(c);
            u64::from_le_bytes(b)
        })
        .collect();
    let rc = unsafe { common::ffi_jvm_reader_set_live_docs(h, segment, words.as_ptr(), words.len()) };
    assert_ne!(rc, common::PANIC, "live docs panicked");
    // body:fox OR body:dog, then the same as a conjunction.
    let or = [
        1u8, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 255, 255, 255, 255, 0, 0, 0, 0, 4, 0, 0, 0, b'b', b'o', b'd', b'y', 3, 0,
        0, 0, b'f', b'o', b'x', 2, 0, 255, 255, 255, 255, 0, 0, 0, 0, 4, 0, 0, 0, b'b', b'o', b'd', b'y', 3, 0, 0,
        0, b'd', b'o', b'g',
    ];
    let mut and = or;
    and[9] = 0;
    and[34] = 0;
    for blob in [&or[..], &and[..]] {
        for (top_n, limit) in [(0, 3), (2, 1), (8, i64::MAX)] {
            assert_ne!(common::search(h, blob, top_n, limit), common::PANIC);
        }
    }
    assert_eq!(unsafe { common::ffi_close_jvm_reader(h) }, 0);
});
