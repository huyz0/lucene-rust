//! Writes two `BloomFilter` postings groups (lucene-codecs'
//! `BloomFilteringPostingsFormat`) plus their expected contents to the
//! directory given as the first CLI argument.
//!
//! Reverse-direction fixture (Rust writes, Java reads): the counterpart
//! `fixtures/src/VerifyCodecsBloom.java` opens each group through real
//! Lucene's own `BloomFilteringPostingsFormat.fieldsProducer`, constructing a
//! matching `SegmentInfo`/`FieldInfos` in Java (no `.si`/`.fnm` from Rust),
//! the same pattern as `write_idversion_fixture.rs`.
//!
//! `_0_BloomFilter_0.*` holds field `id` (docs only, one id per document)
//! over the default `Lucene104PostingsFormat`; `_0_BloomFilter_1.*` holds
//! `tag` (docs and freqs, a few hundred terms over many documents) and
//! `body` (positions, far more distinct terms than the filter is sized for,
//! so its filter saturates and is left out of the `.blm`) over
//! `Lucene104PostingsFormat(4, 9)`.
//!
//! Each group's `blooms.<suffix>.tsv` lists `field\tFuzzySet.toString()` for
//! every filter written, `terms.<suffix>.tsv` lists
//! `field\tterm-hex\tdocs` (comma-separated) for every term, and
//! `probes.<suffix>.tsv` lists `field\tterm-hex\tMAYBE|NO|-` for absent terms
//! as this port's filter answers them.
//!
//! Run: `cargo run -p lucene-codecs --example write_bloom_fixture -- <dir>`
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use lucene_codecs::bloom::{
    BloomFilteredFieldsProducer, BloomFilteringPostingsFormat, ContainsResult,
};
use lucene_codecs::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use lucene_codecs::per_field_postings::Lucene104PostingsFormat;
use lucene_codecs::postings_writer::{FieldPostingsInput, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};

const SEGMENT_ID: [u8; 16] = *b"rustbloomfilter1";
const MAX_DOC: i32 = 20_000;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn word(rng: &mut Rng) -> Vec<u8> {
    let len = 2 + rng.below(9);
    (0..len).map(|_| b'a' + rng.below(26) as u8).collect()
}

/// term -> doc -> positions.
type Postings = BTreeMap<Vec<u8>, BTreeMap<i32, Vec<i32>>>;

fn corpus() -> (Postings, Postings, Postings) {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut id, mut tag, mut body) = (Postings::new(), Postings::new(), Postings::new());
    for doc in 0..MAX_DOC {
        let key = format!("{:08x}{}", doc * 7919, doc % 13).into_bytes();
        id.entry(key).or_default().insert(doc, vec![]);
        for _ in 0..1 + rng.below(3) {
            let t = format!("t{}", rng.below(300)).into_bytes();
            tag.entry(t).or_default().entry(doc).or_default();
        }
        // Mostly fresh words, so `body` ends up with several times more
        // distinct terms than documents and its filter saturates.
        for pos in 0..12 {
            let t = if pos % 4 == 0 {
                format!("w{}", rng.below(50)).into_bytes()
            } else {
                word(&mut rng)
            };
            body.entry(t).or_default().entry(doc).or_default().push(pos);
        }
    }
    (id, tag, body)
}

fn term_postings(p: &Postings, positions: bool) -> Vec<TermPostings> {
    p.iter()
        .map(|(term, docs)| TermPostings {
            term: term.clone(),
            docs: docs
                .iter()
                .map(|(&d, pos)| (d, pos.len().max(1) as i32))
                .collect(),
            positions: if positions {
                docs.values().cloned().collect()
            } else {
                vec![]
            },
            ..TermPostings::default()
        })
        .collect()
}

fn field_infos() -> FieldInfos {
    FieldInfos::new(vec![
        FieldInfo::new("id", 0).with_index_options(IndexOptions::Docs),
        FieldInfo::new("tag", 1).with_index_options(IndexOptions::DocsAndFreqs),
        FieldInfo::new("body", 2).with_index_options(IndexOptions::DocsAndFreqsAndPositions),
    ])
    .unwrap()
}

fn write_group(
    out: &Path,
    suffix: &str,
    delegate: Lucene104PostingsFormat,
    fields: &[(&str, i32, IndexOptions, &Postings)],
) {
    let fis = field_infos();
    let built: Vec<Vec<TermPostings>> = fields
        .iter()
        .map(|(_, _, opts, p)| term_postings(p, opts.subsumes_positions()))
        .collect();
    let inputs: Vec<FieldPostingsInput<'_>> = fields
        .iter()
        .zip(&built)
        .map(|((_, number, opts, p), terms)| {
            let docs: std::collections::BTreeSet<i32> =
                p.values().flat_map(|d| d.keys().copied()).collect();
            FieldPostingsInput {
                field_number: *number,
                index_options: *opts,
                doc_count: docs.len() as i32,
                has_payloads: false,
                terms,
            }
        })
        .collect();
    let state = SegmentWriteState {
        max_doc: MAX_DOC,
        segment_id: &SEGMENT_ID,
        segment_suffix: suffix,
        field_infos: &fis,
    };
    let written = BloomFilteringPostingsFormat::new(delegate)
        .write(&inputs, &[], &state)
        .unwrap();
    let d = &written.delegate;
    for (ext, bytes) in [
        ("blm", &written.blm),
        ("doc", &d.doc),
        ("psm", &d.psm),
        ("pos", &d.pos),
        ("pay", &d.pay),
        ("tim", &d.tim),
        ("tip", &d.tip),
        ("tmd", &d.tmd),
    ] {
        if !bytes.is_empty() {
            std::fs::write(out.join(format!("_0_{suffix}.{ext}")), bytes).unwrap();
        }
    }

    // Read the group back to record what this port's filters answer.
    let read = SegmentReadState {
        max_doc: MAX_DOC,
        segment_id: &SEGMENT_ID,
        segment_suffix: suffix,
        field_infos: &fis,
    };
    let producer =
        BloomFilteredFieldsProducer::open(&written.blm, &d.tim, &d.tip, &d.tmd, &read).unwrap();
    let (mut blooms, mut terms, mut probes) = (String::new(), String::new(), String::new());
    let mut rng = Rng(0x1234_5678 ^ suffix.len() as u64);
    for (name, _, _, p) in fields {
        let filter = producer.bloom(name);
        if *name == "body" {
            assert!(filter.is_none(), "body's filter must saturate");
        } else {
            writeln!(blooms, "{name}\t{}", filter.expect("unsaturated filter")).unwrap();
        }
        for (term, docs) in p.iter() {
            let docs: Vec<String> = docs.keys().map(i32::to_string).collect();
            writeln!(terms, "{name}\t{}\t{}", hex(term), docs.join(",")).unwrap();
        }
        let mut absent = 0;
        while absent < 3000 {
            let t = if absent % 2 == 0 {
                word(&mut rng)
            } else {
                format!("z{}", rng.next()).into_bytes()
            };
            if p.contains_key(&t) {
                continue;
            }
            let answer = filter.map_or("-", |f| match f.contains(&t) {
                ContainsResult::No => "NO",
                ContainsResult::Maybe => "MAYBE",
            });
            writeln!(probes, "{name}\t{}\t{answer}", hex(&t)).unwrap();
            absent += 1;
        }
    }
    std::fs::write(out.join(format!("blooms.{suffix}.tsv")), blooms).unwrap();
    std::fs::write(out.join(format!("terms.{suffix}.tsv")), terms).unwrap();
    std::fs::write(out.join(format!("probes.{suffix}.tsv")), probes).unwrap();
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_bloom_fixture <output-dir>");
    let out = Path::new(&out_dir);
    std::fs::create_dir_all(out).unwrap();
    let (id, tag, body) = corpus();
    write_group(
        out,
        "BloomFilter_0",
        Lucene104PostingsFormat::default(),
        &[("id", 0, IndexOptions::Docs, &id)],
    );
    write_group(
        out,
        "BloomFilter_1",
        Lucene104PostingsFormat::new(4, 9).unwrap(),
        &[
            ("tag", 1, IndexOptions::DocsAndFreqs, &tag),
            ("body", 2, IndexOptions::DocsAndFreqsAndPositions, &body),
        ],
    );
    std::fs::write(
        out.join("manifest.properties"),
        format!("id_hex={}\nmax_doc={MAX_DOC}\n", hex(&SEGMENT_ID)),
    )
    .unwrap();
    println!("wrote two BloomFilter groups to {}", out.display());
}
