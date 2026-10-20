//! Writes three `STUniformSplitPostingsFormat` groups (lucene-codecs'
//! `uniformsplit.sharedterms` package) -- `SharedTermsUniformSplit_0` with
//! the default 32/3 block lines, `SharedTermsUniformSplit_1` with 8/2, and
//! `SharedTermsUniformSplit_2` with 16/3 through an XOR block encoder (every
//! byte XOR 0x5a, then the byte sum of the plain bytes) -- plus their
//! expected contents, to the directory given as the first CLI argument.
//!
//! Reverse-direction fixture (Rust writes, Java reads): the counterpart
//! `fixtures/src/VerifyCodecsSharedTerms.java` opens each group through real
//! Lucene's own `STUniformSplitPostingsFormat.fieldsProducer` (with the same
//! XOR codec for the third), constructing a matching
//! `SegmentInfo`/`FieldInfos` in Java (no `.si`/`.fnm` from Rust), the same
//! pattern as `write_uniformsplit_fixture.rs`.
//!
//! The groups hold the same three fields, whose terms share one dictionary:
//! `id` (docs only, one id per document), `tag` (docs and freqs, with the
//! empty term) and `body` (positions, offsets and payloads; a quarter of its
//! terms are `tag`'s, so many lines carry both fields' states).
//!
//! `terms.<suffix>.tsv` lists `field<TAB>term-hex<TAB>docFreq<TAB>totalTermFreq`
//! then one `doc:freq[:pos.pos:start-end.start-end:payload-hex|-]` per
//! document; `intersect.<suffix>.tsv` lists `field<TAB>pattern<TAB>start|-` then
//! `term-hex:docFreq:totalTermFreq` for every term this port's own reader
//! enumerates for `Terms.intersect`.
//!
//! Run: `cargo run -p lucene-codecs --example write_sharedterms_fixture -- <dir>`
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use std::sync::Arc;

use lucene_codecs::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use lucene_codecs::postings_writer::{FieldPostingsInput, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_codecs::uniformsplit::sharedterms::StUniformSplitPostingsFormat;
use lucene_codecs::uniformsplit::{self, BlockDecoder, BlockEncoder};
use lucene_util::automaton::{operations, CompiledAutomaton, RegExp};

const SEGMENT_ID: [u8; 16] = *b"rustsharedterms!";
const MAX_DOC: i32 = 6_000;
const PATTERNS: [&str; 6] = [
    "[a-c].*",
    ".*ab.*",
    "t[0-3].?",
    "w1.*",
    "(ab|cd)e.*",
    "id1[0-9a-f]*_3",
];
const STARTS: [Option<&str>; 6] = [
    None,
    Some("b"),
    Some("w1"),
    Some("id5"),
    Some("ca"),
    Some("zz"),
];

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// `VerifyCodecsSharedTerms`' `XorEncoder`.
struct XorEncoder;

impl BlockEncoder for XorEncoder {
    fn encode(&self, bytes: &[u8]) -> uniformsplit::Result<Vec<u8>> {
        let mut out: Vec<u8> = bytes.iter().map(|b| b ^ 0x5a).collect();
        out.push(bytes.iter().fold(0u8, |s, b| s.wrapping_add(*b)));
        Ok(out)
    }
}

/// `VerifyCodecsSharedTerms`' `XorDecoder`.
struct XorDecoder;

impl BlockDecoder for XorDecoder {
    fn decode(&self, bytes: &[u8]) -> uniformsplit::Result<Vec<u8>> {
        let (&sum, body) = bytes.split_last().expect("an encoded block");
        let out: Vec<u8> = body.iter().map(|b| b ^ 0x5a).collect();
        assert_eq!(out.iter().fold(0u8, |s, b| s.wrapping_add(*b)), sum);
        Ok(out)
    }
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
            let t = if rng.below(4) == 0 {
                // One of `tag`'s terms: a line with both fields' states.
                format!("t{}", rng.below(40)).into_bytes()
            } else if rng.below(3) == 0 {
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

/// `GenCodecsSharedTerms.dumpPostings`' line for one term.
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

fn compile(pattern: &str) -> CompiledAutomaton {
    let a = RegExp::with_flags(pattern, RegExp::ALL, 0)
        .unwrap()
        .to_automaton()
        .unwrap();
    let a = operations::determinize(&a, operations::DEFAULT_DETERMINIZE_WORK_LIMIT).unwrap();
    CompiledAutomaton::with_options(&a, false, true, false).unwrap()
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_sharedterms_fixture <output-dir>");
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
    let compiled: Vec<(&str, CompiledAutomaton)> =
        PATTERNS.iter().map(|p| (*p, compile(p))).collect();

    for (suffix, format) in [
        (
            "SharedTermsUniformSplit_0",
            StUniformSplitPostingsFormat::default(),
        ),
        (
            "SharedTermsUniformSplit_1",
            StUniformSplitPostingsFormat::new(8, 2, None, None, true).unwrap(),
        ),
        (
            "SharedTermsUniformSplit_2",
            StUniformSplitPostingsFormat::new(
                16,
                3,
                Some(Arc::new(XorEncoder)),
                Some(Arc::new(XorDecoder)),
                false,
            )
            .unwrap(),
        ),
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
        let written = format.write(&inputs, &[], &write).unwrap();
        let reader = format
            .open(written.ustb.clone(), &written.ustd, &read)
            .unwrap();
        let mut intersect = String::new();
        for (name, ..) in &fields {
            let terms = reader.terms(name).unwrap();
            for (pattern, c) in &compiled {
                for start in STARTS {
                    let mut e = terms.intersect(c, start.map(str::as_bytes)).unwrap();
                    write!(intersect, "{name}\t{pattern}\t{}", start.unwrap_or("-")).unwrap();
                    while let Some(t) = e.next().unwrap() {
                        let t = hex(t);
                        let (df, ttf) = (e.doc_freq().unwrap(), e.total_term_freq().unwrap());
                        write!(intersect, "\t{t}:{df}:{ttf}").unwrap();
                    }
                    intersect.push('\n');
                }
            }
        }
        let p = written.postings;
        for (ext, bytes) in [
            ("stustb", written.ustb),
            ("stustd", written.ustd),
            ("doc", p.doc),
            ("psm", p.psm),
            ("pos", p.pos),
            ("pay", p.pay),
        ] {
            std::fs::write(out.join(format!("_0_{suffix}.{ext}")), bytes).unwrap();
        }
        std::fs::write(out.join(format!("terms.{suffix}.tsv")), &terms_tsv).unwrap();
        std::fs::write(out.join(format!("intersect.{suffix}.tsv")), intersect).unwrap();
    }
    std::fs::write(
        out.join("manifest.properties"),
        format!("id_hex={}\nmax_doc={MAX_DOC}\n", hex(&SEGMENT_ID)),
    )
    .unwrap();
    println!(
        "wrote three SharedTermsUniformSplit groups to {}",
        out.display()
    );
}
