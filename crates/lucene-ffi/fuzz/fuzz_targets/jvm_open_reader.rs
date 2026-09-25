//! `ffi_open_jvm_reader` with arbitrary `SegmentInfos` bytes, generation and
//! expected segment sizes: the `segments_N` parser and the reader open, on
//! input a JVM caller controls. The seed corpus should start from the real
//! `segments_2` (see README.md) so mutations reach past the header.
#![no_main]
mod common;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let generation = i64::from(data[0] % 4);
    let docs: Vec<i32> = (0..usize::from(data[1] % 4)).map(|i| 4 + i32::from(data[2] % 2) * i as i32).collect();
    let infos = &data[3..];
    let mut h = 0u64;
    let rc = unsafe {
        common::ffi_open_jvm_reader(
            common::FIXTURE.as_ptr().cast(),
            common::FIXTURE.len(),
            infos.as_ptr(),
            infos.len(),
            generation,
            0,
            docs.as_ptr(),
            docs.len(),
            std::ptr::null(),
            std::ptr::null(),
            &mut h,
        )
    };
    assert_ne!(rc, common::PANIC, "SegmentInfos bytes panicked");
    if rc == 0 {
        // Whatever opened must search and close cleanly.
        let blob = [0u8, 4, 0, 0, 0, b'b', b'o', b'd', b'y', 3, 0, 0, 0, b'f', b'o', b'x'];
        assert_ne!(common::search(h, &blob, 5, 10), common::PANIC);
        assert_eq!(unsafe { common::ffi_close_jvm_reader(h) }, 0);
    }
});
