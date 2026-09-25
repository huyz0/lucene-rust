//! Byte-identity differential test for the term-dictionary write path:
//! `fixtures/data/blocktree_byte_identity_index/` is a real Lucene 10.5.0
//! segment (`GenBlockTreeByteIdentity`), and this port's writer, handed the
//! same terms, postings, segment id and suffix, must produce its `.tim`,
//! `.tip`, `.tmd`, `.doc` and `.psm` byte for byte.
//!
//! `VerifyIndex` already proves real Lucene *reads* what this port writes and
//! cuts the same blocks. That leaves every choice Lucene would accept either
//! way unchecked: a trie node's child-label strategy, the byte width of a
//! child pointer, the singleton run-length form of a block's stats, the
//! suffix-length shortcut. This test pins all of them.
//!
//! The port deviates from Java in exactly two places (suffix compression and
//! `encodeTerm`'s zigzag singleton branch; see `blocktree_writer`'s module
//! doc), and the generator builds a term set on which Java takes neither --
//! so any difference here is a defect, not a documented deviation.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{self, IndexOptions};
use lucene_codecs::postings_writer::{self, FieldPostingsInput, TermPostings};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_byte_identity_index/"
    )
    .to_string()
}

fn manifest() -> Vec<(String, String)> {
    std::fs::read_to_string(format!("{}manifest.properties", dir()))
        .expect("run the fixtures generator first (GenBlockTreeByteIdentity)")
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn get<'m>(m: &'m [(String, String)], key: &str) -> &'m str {
    m.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| panic!("manifest key {key} missing"))
}

fn raw(m: &[(String, String)], ext: &str) -> Vec<u8> {
    let name = get(m, &format!("{ext}_file_name"));
    std::fs::read(format!("{}{name}.raw", dir())).unwrap_or_else(|_| panic!("missing {name}.raw"))
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

/// The first offset at which `got` and `want` differ, with a little context,
/// so a failure names the byte rather than dumping two 100 KB arrays.
fn assert_same_bytes(file: &str, got: &[u8], want: &[u8]) {
    if got == want {
        return;
    }
    let at = got
        .iter()
        .zip(want)
        .position(|(a, b)| a != b)
        .unwrap_or(got.len().min(want.len()));
    let window = |b: &[u8]| b[at.saturating_sub(8)..(at + 8).min(b.len())].to_vec();
    panic!(
        "{file}: first difference at byte {at} of {} (Lucene wrote {}): port {:02x?} vs Lucene {:02x?}",
        got.len(),
        want.len(),
        window(got),
        window(want)
    );
}

#[test]
fn term_dictionary_and_postings_are_byte_identical_to_lucene() {
    let m = manifest();
    let id = id_from_hex(get(&m, "id_hex"));
    let suffix = get(&m, "segment_suffix");
    let infos = field_infos::parse(&raw(&m, "fnm"), &id, "").expect("parse .fnm");
    let field = infos
        .fields
        .iter()
        .find(|f| f.name == "t")
        .expect("field t");
    assert_eq!(field.index_options, IndexOptions::Docs);

    let text = std::fs::read_to_string(format!("{}terms.txt", dir())).expect("terms.txt");
    let terms: Vec<TermPostings> = text
        .lines()
        .map(|t| TermPostings {
            term: t.as_bytes().to_vec(),
            docs: vec![(0, 1), (1, 1)],
            ..Default::default()
        })
        .collect();
    let num_terms: usize = get(&m, "num_terms").parse().unwrap();
    assert_eq!(terms.len(), num_terms);
    assert!(
        num_terms > 10_000,
        "the fixture must be large enough to nest"
    );

    let input = FieldPostingsInput {
        field_number: field.number,
        index_options: IndexOptions::Docs,
        doc_count: 2,
        has_payloads: false,
        terms: &terms,
    };
    let out = postings_writer::write_fields(&[input], &id, suffix).expect("write");

    assert_same_bytes("doc", &out.doc, &raw(&m, "doc"));
    assert_same_bytes("tim", &out.tim, &raw(&m, "tim"));
    assert_same_bytes("tip", &out.tip, &raw(&m, "tip"));
    assert_same_bytes("tmd", &out.tmd, &raw(&m, "tmd"));
    assert_same_bytes("psm", &out.psm, &raw(&m, "psm"));
}
