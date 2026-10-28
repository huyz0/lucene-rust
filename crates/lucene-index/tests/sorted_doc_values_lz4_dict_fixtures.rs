//! The doc-values terms dictionary's LZ4 blocks against Java
//! (`GenSortedDocValuesLz4Dict`): Lucene compresses each 64-term block with
//! `LZ4.compressWithDictionary`, the block's first term as the preset
//! dictionary and one hash table per field, so a term repeating bytes of the
//! first one becomes a back-reference into it. The same documents written
//! through [`IndexWriter`] must produce the same `.dvm`/`.dvd`/`.dvs` bytes,
//! the segment id aside. Terms short enough to stay below LZ4's minimum match
//! (every other doc-values fixture) cannot tell the two compressions apart.

// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_index::document::{Document, SortedDocValuesField, SortedSetDocValuesField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_index::segment_infos;
use lucene_store::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

/// `GenSortedDocValuesLz4Dict.term`.
fn term(i: usize) -> String {
    format!("{i:05}-commonmiddle-{}-commonmiddle-tail{}", i % 7, i % 3)
}

/// `ours` with its segment id replaced by `id` and its footer re-signed.
fn with_segment_id(ours: &[u8], our_id: &[u8; 16], id: &[u8; 16]) -> Vec<u8> {
    let mut bytes = ours.to_vec();
    let at = bytes[..64]
        .windows(16)
        .position(|w| w == our_id)
        .expect("the index header carries the segment id");
    bytes[at..at + 16].copy_from_slice(id);
    let n = bytes.len();
    let crc = u64::from(crc32fast::hash(&bytes[..n - 8]));
    bytes[n - 8..].copy_from_slice(&crc.to_be_bytes());
    bytes
}

#[test]
fn terms_dict_blocks_compress_against_their_first_term_as_java_does() {
    let tmp = TempDir::new("sorted-dv-lz4-dict");
    let dir = FsDirectory::open(&tmp);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
    for i in 0..300 {
        let mut d = Document::new();
        d.add(SortedDocValuesField::new("sorted", term(i)));
        d.add(SortedSetDocValuesField::new("set", term(i)));
        d.add(SortedSetDocValuesField::new("set", term(i + 1000)));
        w.add_fields_document(&d).unwrap();
    }
    w.commit().unwrap();

    let java = FsDirectory::open(format!(
        "{}/../../fixtures/data/sorted_dv_lz4_dict_index",
        env!("CARGO_MANIFEST_DIR")
    ));
    let ours = segment_infos::read_latest(&dir).unwrap();
    let theirs = segment_infos::read_latest(&java).unwrap();
    assert_eq!(ours.segments.len(), 1);
    let (a, b) = (&ours.segments[0], &theirs.segments[0]);
    for ext in ["dvm", "dvd", "dvs"] {
        let name = format!("{}_Lucene90_0.{ext}", a.segment_name);
        let mine = dir.open(&name).unwrap();
        let java_bytes = java.open(&name).unwrap();
        let mine = with_segment_id(&mine, &a.segment_id, &b.segment_id);
        assert_eq!(mine.len(), java_bytes.len(), "{name} length");
        assert!(mine == java_bytes[..], "{name} differs from Java's");
    }
}
