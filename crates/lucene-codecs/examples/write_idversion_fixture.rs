//! Writes two `IDVersion` postings groups (lucene-sandbox's
//! `IDVersionPostingsFormat`) plus their expected contents to the directory
//! given as the first CLI argument.
//!
//! Reverse-direction fixture (Rust writes, Java reads): the counterpart
//! `fixtures/src/VerifyIdVersion.java` opens each group through real Lucene's
//! own `IDVersionPostingsFormat.fieldsProducer`, constructing a matching
//! `SegmentInfo`/`FieldInfos` in Java (no `.si`/`.fnm` from Rust), the same
//! pattern as `write_term_vectors_fixture.rs`.
//!
//! `_0_IDVersion_0.tiv/.tipv` holds field `id` at the default block sizes
//! (25/48), with one document in seven deleted at "flush" (the writer drops
//! those terms); `_0_IDVersion_1.tiv/.tipv` holds field `id2` at 4/9, which
//! nests prefix blocks several levels deep and floors most of them. Ids mix
//! a sequential, prefix-sharing family with scattered base-36 ones, and
//! versions span `MIN_VERSION`..=`MAX_VERSION` (both ends included), so the
//! first-term `vLong` and the `zLong` deltas between neighbours both reach
//! their widest encodings, and the FST's `Long.MAX_VALUE - maxVersion`
//! outputs reach zero and near-`Long.MAX_VALUE`.
//!
//! Each group's `expected.<suffix>.tsv` lists `term-hex\tdoc\tversion` for
//! every term that must be present, and `deleted.<suffix>.tsv` the terms that
//! must not be.
//!
//! Run: `cargo run -p lucene-codecs --example write_idversion_fixture -- <dir>`
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use lucene_codecs::field_infos::IndexOptions;
use lucene_codecs::idversion::{
    write_fields, IdVersionFieldInput, IdVersionPostingsFormat, MAX_VERSION, MIN_VERSION,
};
use lucene_codecs::postings_writer::TermPostings;

const SEGMENT_ID: [u8; 16] = *b"rustidversion001";
const MAX_DOC: i32 = 20_000;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn base36(mut v: u64) -> String {
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut out = Vec::new();
    loop {
        out.push(DIGITS[(v % 36) as usize]);
        v /= 36;
        if v == 0 {
            break;
        }
    }
    out.reverse();
    String::from_utf8(out).unwrap()
}

/// `MAX_DOC` distinct ids, each with its document and version, in term order.
fn corpus(salt: u64) -> BTreeMap<Vec<u8>, (i32, i64)> {
    let mut state = 0x2545_F491_4F6C_DD1D_u64 ^ salt;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut ids = BTreeMap::new();
    let mut doc = 0;
    while doc < MAX_DOC {
        let id = if doc % 3 == 0 {
            // Sequential: long shared prefixes, dense leaf blocks.
            format!("u{:06}", doc).into_bytes()
        } else {
            base36(next() >> 20).into_bytes()
        };
        let version = match doc % 997 {
            0 => MIN_VERSION,
            1 => MAX_VERSION,
            2 => MAX_VERSION - 1,
            _ => (next() >> 2) as i64 % (MAX_VERSION / (1 + (doc as i64 % 5) * 1_000_000)),
        };
        if ids.insert(id, (doc, version)).is_none() {
            doc += 1;
        }
    }
    ids
}

fn write_group(
    out: &Path,
    suffix: &str,
    field_number: i32,
    format: &IdVersionPostingsFormat,
    salt: u64,
    deleted: &dyn Fn(i32) -> bool,
) {
    let ids = corpus(salt);
    let terms: Vec<TermPostings> = ids
        .iter()
        .map(|(term, &(doc, version))| TermPostings {
            term: term.clone(),
            docs: vec![(doc, 1)],
            positions: vec![vec![0]],
            offsets: vec![],
            payload_bytes: version.to_be_bytes().to_vec(),
            payload_lengths: vec![8],
        })
        .collect();
    let input = IdVersionFieldInput {
        field_number,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        store_term_vectors: false,
        has_payloads: true,
        terms: &terms,
    };
    let live = |doc: i32| !deleted(doc);
    let files = write_fields(&[input], Some(&live), format, &SEGMENT_ID, suffix).unwrap();
    std::fs::write(out.join(format!("_0_{suffix}.tiv")), files.tiv).unwrap();
    std::fs::write(out.join(format!("_0_{suffix}.tipv")), files.tipv).unwrap();

    let (mut expected, mut gone) = (String::new(), String::new());
    for (term, &(doc, version)) in &ids {
        if deleted(doc) {
            writeln!(gone, "{}", hex(term)).unwrap();
        } else {
            writeln!(expected, "{}\t{doc}\t{version}", hex(term)).unwrap();
        }
    }
    std::fs::write(out.join(format!("expected.{suffix}.tsv")), expected).unwrap();
    std::fs::write(out.join(format!("deleted.{suffix}.tsv")), gone).unwrap();
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_idversion_fixture <output-dir>");
    let out = Path::new(&out_dir);
    std::fs::create_dir_all(out).unwrap();

    write_group(
        out,
        "IDVersion_0",
        0,
        &IdVersionPostingsFormat::default(),
        1,
        &|doc| doc % 7 == 3,
    );
    write_group(
        out,
        "IDVersion_1",
        1,
        &IdVersionPostingsFormat::new(4, 9).unwrap(),
        2,
        &|_| false,
    );
    std::fs::write(
        out.join("manifest.properties"),
        format!("id_hex={}\nmax_doc={MAX_DOC}\n", hex(&SEGMENT_ID)),
    )
    .unwrap();
    println!("wrote two IDVersion groups to {}", out.display());
}
