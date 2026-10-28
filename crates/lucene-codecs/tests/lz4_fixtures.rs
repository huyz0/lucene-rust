//! `LZ4.compress` against Java (`GenLz4FastTableReuse`): one
//! [`FastCompressionHashTable`] reused across inputs whose sizes cross the
//! 64 KiB offset-width line both ways must emit Java's bytes for every one,
//! which holds only if the table is reset exactly when Java reallocates it.

// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::lz4::{self, FastCompressionHashTable};

const WORDS: [&str; 16] = [
    "lucene", "segment", "posting", "rust", "codec", "merge", "vector", "term", "field", "doc",
    "values", "index", "reader", "writer", "block", "skip",
];

/// `GenLz4FastTableReuse`'s input for `size`, continuing `seed`.
fn input(seed: &mut u64, size: usize) -> Vec<u8> {
    let mut s = String::new();
    while s.len() < size {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        s.push_str(WORDS[((*seed >> 33) % WORDS.len() as u64) as usize]);
        s.push(' ');
        if (*seed >> 20) & 7 == 0 {
            s.push_str(&(*seed & 0xFFFF).to_string());
            s.push(' ');
        }
    }
    s.truncate(size);
    s.into_bytes()
}

#[test]
fn a_reused_fast_table_compresses_as_java_across_offset_widths() {
    let bin = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/lz4_fast_table_reuse.bin"
    ))
    .unwrap();
    let mut ht = FastCompressionHashTable::new();
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut at = 0;
    let mut cases = 0;
    while at < bin.len() {
        let len = u32::from_be_bytes(bin[at..at + 4].try_into().unwrap()) as usize;
        let clen = u32::from_be_bytes(bin[at + 4..at + 8].try_into().unwrap()) as usize;
        let java = &bin[at + 8..at + 8 + clen];
        at += 8 + clen;
        let data = input(&mut seed, len);
        let mut ours = Vec::new();
        lz4::compress_into(&data, &mut ours, &mut ht);
        assert_eq!(ours.len(), java.len(), "case {cases} ({len} bytes): length");
        assert!(
            ours == java,
            "case {cases} ({len} bytes) differs from Java's"
        );
        cases += 1;
    }
    assert_eq!(cases, 7);
}
