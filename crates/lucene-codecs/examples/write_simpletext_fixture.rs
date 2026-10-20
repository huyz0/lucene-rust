//! Writes a SimpleText postings file (lucene-codecs' `simpletext` package),
//! `_0_SimpleText_0.pst`, plus its expected contents, to the directory given
//! as the first CLI argument.
//!
//! Reverse-direction fixture (Rust writes, Java reads): the counterpart
//! `fixtures/src/VerifyCodecsSimpleText.java` opens the file through real
//! Lucene's own `SimpleTextCodec().postingsFormat()` over a hand-built
//! `SegmentInfo`/`FieldInfos` (no `.si`/`.fnm` from Rust), walks it against
//! `terms.tsv`, then has Lucene's own writer re-write it from its own reader
//! with the norms of `norms.tsv` and requires the two files to be equal byte
//! for byte -- skip lists and their impacts included.
//!
//! The fields: `id` (docs only, no norms, with the empty term), `bin`
//! (docs only, binary terms holding `\n`, `\` and `0xff`), `tag` (freqs) and
//! `body` (positions, offsets and payloads that hold `\n` and `\`), the last
//! two with norms; enough documents that `body`'s common terms skip over
//! three levels.
//!
//! `terms.tsv` lists `field<TAB>term-hex<TAB>docFreq<TAB>totalTermFreq` then
//! one `doc:freq[:pos.pos:start-end.start-end:payload-hex|-]` per document;
//! `norms.tsv` lists `field` then each document's norm.
//!
//! Run: `cargo run -p lucene-codecs --example write_simpletext_fixture -- <dir>`
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use lucene_codecs::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use lucene_codecs::postings_writer::{FieldNorms, FieldPostingsInput, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_codecs::simpletext::fields_reader::SimpleTextFieldsReader;
use lucene_codecs::simpletext::fields_writer::{postings_file_name, write as write_pst};
use lucene_codecs::simpletext::{ALL, NO_MORE_DOCS};

const SEGMENT_ID: [u8; 16] = *b"rustsimpletextid";
const MAX_DOC: i32 = 3_000;
const SUFFIX: &str = "SimpleText_0";

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
/// term -> doc -> occurrences (one placeholder per occurrence for `tag`,
/// none for docs-only fields).
type Postings = BTreeMap<Vec<u8>, BTreeMap<i32, Vec<Occ>>>;

struct Corpus {
    id: Postings,
    bin: Postings,
    tag: Postings,
    body: Postings,
    tag_norms: Vec<i64>,
    body_norms: Vec<i64>,
}

fn corpus() -> Corpus {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut c = Corpus {
        id: Postings::new(),
        bin: Postings::new(),
        tag: Postings::new(),
        body: Postings::new(),
        tag_norms: Vec::new(),
        body_norms: Vec::new(),
    };
    let alphabet = [b'a', b'\n', b'\\', 0xff, b'z'];
    for doc in 0..MAX_DOC {
        let key = format!("id{:x}", doc * 7919).into_bytes();
        c.id.entry(key).or_default().insert(doc, vec![]);
        if doc % 211 == 5 {
            c.id.entry(Vec::new()).or_default().insert(doc, vec![]);
        }
        let len = 1 + rng.below(3) as usize;
        let bin: Vec<u8> = (0..len)
            .map(|_| alphabet[rng.below(alphabet.len() as u64) as usize])
            .collect();
        c.bin.entry(bin).or_default().insert(doc, vec![]);
        let mut tag_len = 0;
        for _ in 0..1 + rng.below(4) {
            let bound = 1 + rng.below(60);
            let t = format!("t{}", rng.below(bound)).into_bytes();
            c.tag
                .entry(t)
                .or_default()
                .entry(doc)
                .or_default()
                .push((0, 0, 0, vec![]));
            tag_len += 1;
        }
        c.tag_norms.push(tag_len);
        let mut offset = 0;
        let words = 2 + rng.below(10) as i32;
        for pos in 0..words {
            let bound = 1 + rng.below(400);
            let t = format!("w{}", rng.below(bound)).into_bytes();
            let payload = match rng.below(4) {
                0 => vec![],
                1 => vec![b'\n', b'\\'],
                _ => (0..1 + rng.below(3))
                    .map(|_| rng.below(256) as u8)
                    .collect(),
            };
            let end = offset + t.len() as i32;
            c.body
                .entry(t)
                .or_default()
                .entry(doc)
                .or_default()
                .push((pos, offset, end, payload));
            offset = end + 1;
        }
        // A norm that varies with the length, and sometimes not, so that the
        // competitive impacts keep more than one pair.
        c.body_norms
            .push(i64::from(words) + if rng.below(5) == 0 { 40 } else { 0 });
    }
    c
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
        FieldInfo::new("id", 0)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
        FieldInfo::new("bin", 1)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
        FieldInfo::new("tag", 2).with_index_options(IndexOptions::DocsAndFreqs),
        FieldInfo::new("body", 3)
            .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
            .with_store_payloads(true),
    ])
    .unwrap()
}

/// `VerifyCodecsSimpleText.termLine`'s line for one term.
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

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_simpletext_fixture <output-dir>");
    let out = Path::new(&out_dir);
    std::fs::create_dir_all(out).unwrap();
    let c = corpus();
    // Name order, as a flush hands them to the writer.
    let fields: [(&str, i32, IndexOptions, bool, &Postings); 4] = [
        ("bin", 1, IndexOptions::Docs, false, &c.bin),
        (
            "body",
            3,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            true,
            &c.body,
        ),
        ("id", 0, IndexOptions::Docs, false, &c.id),
        ("tag", 2, IndexOptions::DocsAndFreqs, false, &c.tag),
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
            let docs: BTreeSet<i32> = p.values().flat_map(|d| d.keys().copied()).collect();
            FieldPostingsInput {
                field_number: *number,
                index_options: *io,
                doc_count: docs.len() as i32,
                has_payloads: *payloads,
                terms,
            }
        })
        .collect();
    let norms = [
        FieldNorms {
            field_number: 2,
            values: &c.tag_norms,
        },
        FieldNorms {
            field_number: 3,
            values: &c.body_norms,
        },
    ];
    let write = SegmentWriteState {
        max_doc: MAX_DOC,
        segment_id: &SEGMENT_ID,
        segment_suffix: SUFFIX,
        field_infos: &fis,
    };
    let pst = write_pst(&inputs, &norms, &write).unwrap();

    // This port reads its own bytes back before Java is asked to.
    let read = SegmentReadState {
        max_doc: MAX_DOC,
        segment_id: &SEGMENT_ID,
        segment_suffix: SUFFIX,
        field_infos: &fis,
    };
    let reader = SimpleTextFieldsReader::open(pst.clone(), &read).unwrap();
    let mut terms_tsv = String::new();
    for (name, _, io, _, p) in &fields {
        let terms = reader.terms(name).unwrap().unwrap();
        let mut e = terms.iterator();
        for (term, docs) in p.iter() {
            assert_eq!(e.next().unwrap(), Some(term.as_slice()), "{name}");
            let mut pe = e.postings(ALL).unwrap();
            for doc in docs.keys() {
                assert_eq!(pe.next_doc().unwrap(), *doc);
            }
            assert_eq!(pe.next_doc().unwrap(), NO_MORE_DOCS);
            terms_tsv.push_str(&term_line(name, term, docs, *io));
            terms_tsv.push('\n');
        }
        assert_eq!(e.next().unwrap(), None);
    }
    let mut norms_tsv = String::new();
    for (name, values) in [("tag", &c.tag_norms), ("body", &c.body_norms)] {
        norms_tsv.push_str(name);
        for v in values.iter() {
            write!(norms_tsv, "\t{v}").unwrap();
        }
        norms_tsv.push('\n');
    }
    std::fs::write(out.join(postings_file_name("_0", SUFFIX)), pst).unwrap();
    std::fs::write(out.join("terms.tsv"), &terms_tsv).unwrap();
    std::fs::write(out.join("norms.tsv"), &norms_tsv).unwrap();
    std::fs::write(
        out.join("manifest.properties"),
        format!("id_hex={}\nmax_doc={MAX_DOC}\n", hex(&SEGMENT_ID)),
    )
    .unwrap();
    println!("wrote a SimpleText postings file to {}", out.display());
}
