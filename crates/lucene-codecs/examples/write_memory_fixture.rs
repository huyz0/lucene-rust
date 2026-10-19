//! Writes an `FSTPostingsFormat` ("FST50") group and a
//! `DirectPostingsFormat` ("Direct") group (lucene-codecs' `memory`
//! package), plus their expected contents, to the directory given as the
//! first CLI argument.
//!
//! Reverse-direction fixture (Rust writes, Java reads): the counterpart
//! `fixtures/src/VerifyCodecsMemory.java` opens each group through real
//! Lucene's own `FSTPostingsFormat.fieldsProducer` and
//! `DirectPostingsFormat.fieldsProducer`, constructing a matching
//! `SegmentInfo`/`FieldInfos` in Java (no `.si`/`.fnm` from Rust), the same
//! pattern as `write_bloom_fixture.rs`.
//!
//! Both groups hold the same three fields: `id` (docs only, one id per
//! document), `tag` (docs and freqs) and `body` (positions, offsets and
//! payloads, a few words in most documents so `Direct` keeps both its
//! low- and high-frequency term shapes).
//!
//! `terms.<suffix>.tsv` lists `field\tterm-hex\tdocFreq\ttotalTermFreq`
//! then one `doc:freq[:pos.pos:start-end.start-end:payload-hex|-]` per
//! document, as `GenCodecsMemory` dumps them; `intersect.<suffix>.tsv` lists
//! `field\tpattern\tstart|-` then `term-hex:docFreq:totalTermFreq` for every
//! term this port's own reader enumerates for `Terms.intersect`.
//!
//! Run: `cargo run -p lucene-codecs --example write_memory_fixture -- <dir>`
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use lucene_codecs::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use lucene_codecs::memory::{DirectFiles, DirectPostingsFormat, FstPostingsFormat};
use lucene_codecs::postings_writer::{FieldPostingsInput, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_util::automaton::{operations, CompiledAutomaton, RegExp};

const SEGMENT_ID: [u8; 16] = *b"rustmemorycodecs";
const MAX_DOC: i32 = 6_000;
const PATTERNS: [&str; 5] = ["[a-c].*", ".*ab.*", "t[0-3].?", "w1.*", "(ab|cd)e.*"];
const STARTS: [Option<&str>; 3] = [None, Some("b"), Some("w1")];

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

/// `GenCodecsMemory.dumpPostings`' line for one term.
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

/// Whether the automaton has a live state after every byte of `start`
/// (`DirectIntersectTermsEnum` requires it).
fn live_prefix(compiled: &CompiledAutomaton, start: &[u8]) -> bool {
    let run = compiled.get_byte_runnable().unwrap();
    let mut state = 0;
    for &b in start {
        state = run.step(state, i32::from(b));
        if state == -1 {
            return false;
        }
    }
    true
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_memory_fixture <output-dir>");
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

    for suffix in ["FST50_0", "Direct_0"] {
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
        let mut intersect = String::new();
        let files: Vec<(&str, Vec<u8>)> = if suffix.starts_with("FST") {
            let written = FstPostingsFormat.write(&inputs, &[], &write).unwrap();
            let reader = FstPostingsFormat.open(&written.tfp, &read).unwrap();
            for (name, ..) in &fields {
                let terms = reader.terms(name).unwrap();
                for (pattern, c) in &compiled {
                    for start in STARTS {
                        let mut e = terms.intersect(c, start.map(str::as_bytes)).unwrap();
                        write!(intersect, "{name}\t{pattern}\t{}", start.unwrap_or("-")).unwrap();
                        while let Some(t) = e.next().unwrap() {
                            let t = hex(t);
                            write!(intersect, "\t{t}:{}:{}", e.doc_freq(), e.total_term_freq())
                                .unwrap();
                        }
                        intersect.push('\n');
                    }
                }
            }
            let p = written.postings;
            vec![
                ("tfp", written.tfp),
                ("doc", p.doc),
                ("psm", p.psm),
                ("pos", p.pos),
                ("pay", p.pay),
            ]
        } else {
            let w = DirectPostingsFormat::default()
                .write(&inputs, &[], &write)
                .unwrap();
            let direct = DirectPostingsFormat::default()
                .open(
                    &DirectFiles {
                        tim: &w.tim,
                        tip: &w.tip,
                        tmd: &w.tmd,
                        doc: &w.doc,
                        pos: Some(&w.pos),
                        pay: Some(&w.pay),
                    },
                    &read,
                )
                .unwrap();
            for (name, ..) in &fields {
                let terms = direct.terms(name).unwrap();
                for (pattern, c) in &compiled {
                    for start in STARTS {
                        if start.is_some_and(|s| !live_prefix(c, s.as_bytes())) {
                            continue;
                        }
                        let mut e = terms.intersect(c, start.map(str::as_bytes)).unwrap();
                        write!(intersect, "{name}\t{pattern}\t{}", start.unwrap_or("-")).unwrap();
                        while let Some(t) = e.next() {
                            let t = hex(t);
                            write!(intersect, "\t{t}:{}:{}", e.doc_freq(), e.total_term_freq())
                                .unwrap();
                        }
                        intersect.push('\n');
                    }
                }
            }
            vec![
                ("tim", w.tim),
                ("tip", w.tip),
                ("tmd", w.tmd),
                ("doc", w.doc),
                ("psm", w.psm),
                ("pos", w.pos),
                ("pay", w.pay),
            ]
        };
        for (ext, bytes) in files {
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
    println!("wrote an FST50 and a Direct group to {}", out.display());
}
