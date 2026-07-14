//! Writes an FST built by this port's own `fst::build_fst`/`fst::write_fst`
//! (a from-scratch, simplified construction -- no suffix sharing/
//! minimization, no output pushing, no fixed-length-arc compaction; see
//! `crates/lucene-codecs/src/fst.rs`'s module doc) to disk in the exact byte
//! layout `Fst::read`/real Lucene's `FST.read(Path, Outputs)` both parse.
//!
//! Reverse-direction differential test (Rust writes, Java reads), same
//! division of labor as the other `write_*_fixture.rs` examples:
//! `fixtures/src/VerifyFst.java` opens the result through real Lucene's
//! `FST.read(Path, ByteSequenceOutputs)` and looks up every key via real
//! `Util.get(FST, BytesRef)`.
//!
//! Uses the exact same 7-key set as the read-side fixture
//! (`fixtures/src/GenFst.java`) -- `app`/`apple`/`application`,
//! `banana`/`band`/`bandana`, `z` -- sharing prefixes/suffixes, plus the same
//! deliberately-absent keys, so this write-side fixture exercises the same
//! shape of FST that the read-side fixture already validates, just built by
//! this port's writer instead of real `FSTCompiler`.
//!
//! Also writes a second, larger fixture (`large/`, 200 keys) forcing
//! multi-byte `vlong` node-address targets -- the same shape
//! `build_fst_many_keys_forces_multi_byte_vlong_targets` stress-tests via
//! self-round-trip in `fst.rs`'s own unit tests, but never previously run
//! through a real Lucene reader, since the small 7-key set here is too
//! small to reach that encoding path.
//!
//! Run: `cargo run -p lucene-codecs --example write_fst_fixture -- <dir>`

use lucene_codecs::fst::{build_fst, write_fst};
use std::io::Write;

fn write_fixture(out_dir: &str, entries: &[(Vec<u8>, Vec<u8>)], absent_keys: &[&[u8]]) {
    std::fs::create_dir_all(out_dir).unwrap();

    let fst = build_fst(entries).expect("build_fst over sorted entries");
    let bytes = write_fst(&fst);
    std::fs::write(format!("{out_dir}/fst.bin"), &bytes).unwrap();

    let mut manifest = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    writeln!(manifest, "num_present={}", entries.len()).unwrap();
    for (i, (key, output)) in entries.iter().enumerate() {
        writeln!(manifest, "present.{i}.key_hex={}", hex(key)).unwrap();
        writeln!(manifest, "present.{i}.output_hex={}", hex(output)).unwrap();
    }
    writeln!(manifest, "num_absent={}", absent_keys.len()).unwrap();
    for (i, key) in absent_keys.iter().enumerate() {
        writeln!(manifest, "absent.{i}.key_hex={}", hex(key)).unwrap();
    }

    println!("wrote FST fixture to {out_dir}");
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_fst_fixture <output-dir>");

    let small_entries: Vec<(Vec<u8>, Vec<u8>)> = vec![
        (b"app".to_vec(), b"1".to_vec()),
        (b"apple".to_vec(), b"2".to_vec()),
        (b"application".to_vec(), b"3".to_vec()),
        (b"banana".to_vec(), b"4".to_vec()),
        (b"band".to_vec(), b"5".to_vec()),
        (b"bandana".to_vec(), b"6".to_vec()),
        (b"z".to_vec(), b"26".to_vec()),
    ];
    let small_absent: Vec<&[u8]> = vec![
        b"",
        b"a",
        b"appl",
        b"apples",
        b"ban",
        b"bandanas",
        b"cat",
        b"zz",
    ];
    write_fixture(&out_dir, &small_entries, &small_absent);

    let large_entries: Vec<(Vec<u8>, Vec<u8>)> = (0u16..200)
        .map(|i| {
            let key = format!("key{i:04}").into_bytes();
            let output = i.to_le_bytes().to_vec();
            (key, output)
        })
        .collect();
    let large_absent: Vec<&[u8]> = vec![b"key9999", b"key0200", b""];
    write_fixture(&format!("{out_dir}/large"), &large_entries, &large_absent);
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
