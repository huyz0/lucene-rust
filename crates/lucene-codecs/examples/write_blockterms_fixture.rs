//! Writes four block terms groups (lucene-codecs' `blockterms` package) --
//! `BlockTerms_0` over `FixedGapTermsIndexWriter(32)` (the default),
//! `BlockTerms_1` over `FixedGapTermsIndexWriter(3)`, `BlockTerms_2` over
//! `VariableGapTermsIndexWriter(EveryNTermSelector(5))` and `BlockTerms_3`
//! over `VariableGapTermsIndexWriter(EveryNOrDocFreqTermSelector(30, 9))` --
//! plus their expected contents, to the directory given as the first CLI
//! argument.
//!
//! Reverse-direction fixture (Rust writes, Java reads): the counterpart
//! `fixtures/src/VerifyCodecsBlockTerms.java` opens each group through real
//! Lucene's own `BlockTermsReader` over its `FixedGapTermsIndexReader` or
//! `VariableGapTermsIndexReader` and a `Lucene104PostingsReader` (Lucene
//! ships no postings format over this package), constructing a matching
//! `SegmentInfo`/`FieldInfos` in Java (no `.si`/`.fnm` from Rust), the same
//! pattern as `write_uniformsplit_fixture.rs`.
//!
//! The groups hold the same three fields: `id` (docs only, one id per
//! document), `tag` (docs and freqs, with the empty term) and `body`
//! (positions, offsets and payloads).
//!
//! `terms.tsv` lists `field<TAB>term-hex<TAB>docFreq<TAB>totalTermFreq` then
//! one `doc:freq[:pos.pos:start-end.start-end:payload-hex|-]` per document,
//! every field's terms in order (a term's ord is its line within its field).
//!
//! Run: `cargo run -p lucene-codecs --example write_blockterms_fixture -- <dir>`
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use lucene_codecs::blockterms::{
    self, BlockTermsReader, EveryNOrDocFreqTermSelector, EveryNTermSelector,
    FixedGapTermsIndexReader, FixedGapTermsIndexWriter, TermsIndexReader,
    VariableGapTermsIndexReader, VariableGapTermsIndexWriter,
};
use lucene_codecs::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use lucene_codecs::postings_writer::{FieldPostingsInput, PostingsFiles, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};

const SEGMENT_ID: [u8; 16] = *b"rustblocktermsid";
const MAX_DOC: i32 = 6_000;

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

/// One occurrence: position, offsets, payload.
type Occ = (i32, i32, i32, Vec<u8>);
/// term -> doc -> occurrences (empty for fields without positions, one
/// placeholder per occurrence for `tag`'s freqs).
type Postings = BTreeMap<Vec<u8>, BTreeMap<i32, Vec<Occ>>>;

fn corpus() -> (Postings, Postings, Postings) {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let (mut id, mut tag, mut body) = (Postings::new(), Postings::new(), Postings::new());
    for doc in 0..MAX_DOC {
        let key = format!("id{:x}_{}", doc * 7919, doc % 7).into_bytes();
        id.entry(key).or_default().insert(doc, vec![]);
        if doc % 97 == 0 {
            tag.entry(Vec::new())
                .or_default()
                .entry(doc)
                .or_default()
                .push((0, 0, 0, vec![]));
        }
        for _ in 0..1 + rng.below(4) {
            let t = format!("t{}", rng.below(40)).into_bytes();
            tag.entry(t)
                .or_default()
                .entry(doc)
                .or_default()
                .push((0, 0, 0, vec![]));
        }
        let mut offset = 0;
        for pos in 0..8 {
            let t = if rng.below(3) == 0 {
                format!("w{}", rng.below(30)).into_bytes()
            } else {
                let len = 2 + rng.below(5);
                (0..len).map(|_| b'a' + rng.below(6) as u8).collect()
            };
            let payload = if rng.below(4) == 0 {
                vec![]
            } else {
                (0..1 + rng.below(3))
                    .map(|_| rng.below(256) as u8)
                    .collect()
            };
            let end = offset + t.len() as i32;
            body.entry(t)
                .or_default()
                .entry(doc)
                .or_default()
                .push((pos, offset, end, payload));
            offset = end + 1;
        }
    }
    (id, tag, body)
}

fn term_postings(p: &Postings, io: IndexOptions) -> Vec<TermPostings> {
    p.iter()
        .map(|(term, docs)| {
            let mut tp = TermPostings {
                term: term.clone(),
                docs: docs
                    .iter()
                    .map(|(&d, occ)| (d, occ.len().max(1) as i32))
                    .collect(),
                ..TermPostings::default()
            };
            if io.subsumes_positions() {
                tp.positions = docs
                    .values()
                    .map(|o| o.iter().map(|x| x.0).collect())
                    .collect();
                tp.offsets = docs
                    .values()
                    .map(|o| o.iter().map(|x| (x.1, x.2)).collect())
                    .collect();
                for o in docs.values().flatten() {
                    tp.payload_lengths.push(o.3.len() as u32);
                    tp.payload_bytes.extend_from_slice(&o.3);
                }
            }
            tp
        })
        .collect()
}

fn field_infos() -> FieldInfos {
    FieldInfos::new(vec![
        FieldInfo::new("id", 0).with_index_options(IndexOptions::Docs),
        FieldInfo::new("tag", 1).with_index_options(IndexOptions::DocsAndFreqs),
        FieldInfo::new("body", 2)
            .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
            .with_store_payloads(true),
    ])
    .unwrap()
}

/// `VerifyCodecsBlockTerms.termLine`'s line for one term.
fn term_line(name: &str, term: &[u8], docs: &BTreeMap<i32, Vec<Occ>>, io: IndexOptions) -> String {
    let df = docs.len();
    let ttf: usize = if io == IndexOptions::Docs {
        df
    } else {
        docs.values().map(|o| o.len().max(1)).sum()
    };
    let mut s = format!("{name}\t{}\t{df}\t{ttf}", hex(term));
    for (doc, occ) in docs {
        write!(s, "\t{doc}:{}", occ.len().max(1)).unwrap();
        if io.subsumes_positions() {
            let pos: Vec<String> = occ.iter().map(|o| o.0.to_string()).collect();
            let offs: Vec<String> = occ.iter().map(|o| format!("{}-{}", o.1, o.2)).collect();
            let pays: Vec<String> = occ
                .iter()
                .map(|o| {
                    if o.3.is_empty() {
                        "-".into()
                    } else {
                        hex(&o.3)
                    }
                })
                .collect();
            write!(
                s,
                ":{}:{}:{}",
                pos.join("."),
                offs.join("."),
                pays.join(".")
            )
            .unwrap();
        }
    }
    s
}

/// The files of one group, by extension.
type Files = Vec<(&'static str, Vec<u8>)>;

fn files(tib: Vec<u8>, index: Files, p: PostingsFiles) -> Files {
    let mut out = vec![("tib", tib)];
    out.extend(index);
    out.extend([
        ("doc", p.doc),
        ("psm", p.psm),
        ("pos", p.pos),
        ("pay", p.pay),
    ]);
    out
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_blockterms_fixture <output-dir>");
    let out = Path::new(&out_dir);
    std::fs::create_dir_all(out).unwrap();
    let (id, tag, body) = corpus();
    let fields: [(&str, i32, IndexOptions, bool, &Postings); 3] = [
        (
            "body",
            2,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            true,
            &body,
        ),
        ("id", 0, IndexOptions::Docs, false, &id),
        ("tag", 1, IndexOptions::DocsAndFreqs, false, &tag),
    ];
    let fis = field_infos();
    let built: Vec<Vec<TermPostings>> = fields
        .iter()
        .map(|(_, _, io, _, p)| term_postings(p, *io))
        .collect();
    let inputs: Vec<FieldPostingsInput<'_>> = fields
        .iter()
        .zip(&built)
        .map(|((_, number, io, payloads, p), terms)| {
            let docs: std::collections::BTreeSet<i32> =
                p.values().flat_map(|d| d.keys().copied()).collect();
            FieldPostingsInput {
                field_number: *number,
                index_options: *io,
                doc_count: docs.len() as i32,
                has_payloads: *payloads,
                terms,
            }
        })
        .collect();
    let mut terms_tsv = String::new();
    for (name, _, io, _, p) in &fields {
        for (term, docs) in p.iter() {
            terms_tsv.push_str(&term_line(name, term, docs, *io));
            terms_tsv.push('\n');
        }
    }

    for suffix in [
        "BlockTerms_0",
        "BlockTerms_1",
        "BlockTerms_2",
        "BlockTerms_3",
    ] {
        let write = SegmentWriteState {
            max_doc: MAX_DOC,
            segment_id: &SEGMENT_ID,
            segment_suffix: suffix,
            field_infos: &fis,
        };
        let read = SegmentReadState {
            max_doc: MAX_DOC,
            segment_id: &SEGMENT_ID,
            segment_suffix: suffix,
            field_infos: &fis,
        };
        let (written, index): (Files, TermsIndexReader) = match suffix {
            "BlockTerms_0" | "BlockTerms_1" => {
                let interval = if suffix == "BlockTerms_0" { 32 } else { 3 };
                let w = FixedGapTermsIndexWriter::new(&write, interval).unwrap();
                let o = blockterms::write(w, &inputs, &[], &write).unwrap();
                let index = FixedGapTermsIndexReader::open(&o.index, &read).unwrap();
                (
                    files(o.tib, vec![("tii", o.index)], o.postings),
                    index.into(),
                )
            }
            "BlockTerms_2" => {
                let w = VariableGapTermsIndexWriter::new(&write, EveryNTermSelector::new(5));
                let o = blockterms::write(w, &inputs, &[], &write).unwrap();
                let index =
                    VariableGapTermsIndexReader::open(&o.index.tmv, &o.index.tiv, &read).unwrap();
                let idx = vec![("tiv", o.index.tiv), ("tmv", o.index.tmv)];
                (files(o.tib, idx, o.postings), index.into())
            }
            _ => {
                let selector = EveryNOrDocFreqTermSelector::new(30, 9);
                let w = VariableGapTermsIndexWriter::new(&write, selector);
                let o = blockterms::write(w, &inputs, &[], &write).unwrap();
                let index =
                    VariableGapTermsIndexReader::open(&o.index.tmv, &o.index.tiv, &read).unwrap();
                let idx = vec![("tiv", o.index.tiv), ("tmv", o.index.tmv)];
                (files(o.tib, idx, o.postings), index.into())
            }
        };
        // This port reads its own bytes back before Java is asked to.
        let reader = BlockTermsReader::open(written[0].1.clone(), index, &read).unwrap();
        reader.check_integrity().unwrap();
        for (name, _, _, _, p) in &fields {
            let mut e = reader.terms(name).unwrap().iterator();
            for term in p.keys() {
                assert_eq!(e.next().unwrap(), Some(term.as_slice()), "{suffix} {name}");
            }
            assert_eq!(e.next().unwrap(), None);
        }
        for (ext, bytes) in written {
            std::fs::write(out.join(format!("_0_{suffix}.{ext}")), bytes).unwrap();
        }
    }
    std::fs::write(out.join("terms.tsv"), &terms_tsv).unwrap();
    std::fs::write(
        out.join("manifest.properties"),
        format!("id_hex={}\nmax_doc={MAX_DOC}\n", hex(&SEGMENT_ID)),
    )
    .unwrap();
    println!("wrote four block terms groups to {}", out.display());
}
