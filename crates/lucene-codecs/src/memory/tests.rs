//! Unit tests for the memory postings formats: a synthetic segment written
//! through each format and read back against the model it was built from,
//! plus the boundaries the Java fixtures do not reach (Java's own
//! `TestDirectPostingsFormat`/`TestFSTPostingsFormat` run the shared
//! `BasePostingsFormatTestCase` over random indexes; these exercise the
//! same reader surface over one deterministic one).
// Test code opts out of the arithmetic gate at the file boundary: see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_util::automaton::{operations, AutomatonType, CompiledAutomaton, RegExp};

use super::*;
use crate::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use crate::fst_compiler::{FstOutputs, ReverseReader};
use crate::postings::{DocInput, PostingsFlags, NO_MORE_DOCS};
use crate::postings_writer::{FieldPostingsInput, TermPostings};
use crate::segment_state::{SegmentReadState, SegmentWriteState};

const ID: [u8; 16] = *b"memoryunittests!";
const MAX_DOC: i32 = 300;

/// `(name, number, IndexOptions, payloads)`.
const FIELDS: [(&str, i32, IndexOptions, bool); 6] = [
    ("d", 0, IndexOptions::Docs, false),
    ("f", 1, IndexOptions::DocsAndFreqs, false),
    ("p", 2, IndexOptions::DocsAndFreqsAndPositions, false),
    ("pp", 3, IndexOptions::DocsAndFreqsAndPositions, true),
    (
        "o",
        4,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        false,
    ),
    (
        "op",
        5,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        true,
    ),
];

/// One occurrence: position, start and end offsets, payload.
type Occ = (i32, i32, i32, Vec<u8>);

/// A term of the model: its bytes and `(doc, occurrences)` per document.
#[derive(Debug, Clone)]
struct Term {
    bytes: Vec<u8>,
    docs: Vec<(i32, Vec<Occ>)>,
}

impl Term {
    fn freq(&self, i: usize, io: IndexOptions) -> i32 {
        if io == IndexOptions::Docs {
            1
        } else {
            self.docs[i].1.len() as i32
        }
    }

    fn total_term_freq(&self, io: IndexOptions) -> i64 {
        (0..self.docs.len())
            .map(|i| i64::from(self.freq(i, io)))
            .sum()
    }
}

/// The terms every field holds (the empty term only in even fields): runs
/// sharing long prefixes (skips), a dense label range (direct addressing)
/// and a sparse one (binary-searched arcs).
fn term_bytes(number: i32) -> Vec<Vec<u8>> {
    let mut t: Vec<Vec<u8>> = vec![b"a".to_vec(), b"b".to_vec(), b"zzz".to_vec()];
    if number % 2 == 0 {
        t.push(Vec::new());
    }
    t.extend((0..40).map(|i| format!("ab{i:02}").into_bytes()));
    t.extend((0..12).map(|i| format!("abc{i}x").into_bytes()));
    t.extend((0..10).map(|i| format!("m{}", "n".repeat(i)).into_bytes()));
    t.extend([0x05u8, 0x31, 0x52, 0x77, 0x90, 0xa3, 0xc8, 0xee].map(|b| vec![b'q', b, b'z']));
    t.push(vec![0xff, 0x01]);
    t.sort();
    t.dedup();
    t
}

fn model(number: i32) -> Vec<Term> {
    term_bytes(number)
        .into_iter()
        .enumerate()
        .map(|(i, bytes)| {
            // Steps 1..79: 300 documents down to 4.
            let step = 1 + (i % 7) * 13;
            let docs = ((i % 5) as i32..MAX_DOC)
                .step_by(step)
                .map(|doc| {
                    let freq = 1 + (doc as usize + i) % 3;
                    let occ = (0..freq as i32)
                        .map(|k| {
                            let pos = k * 3 + doc % 2;
                            let payload_len = (doc as usize + k as usize + i) % 3;
                            let payload =
                                [(doc % 251) as u8, k as u8, i as u8][..payload_len].to_vec();
                            (pos, pos * 4, pos * 4 + 2 + k, payload)
                        })
                        .collect();
                    (doc, occ)
                })
                .collect();
            Term { bytes, docs }
        })
        .collect()
}

fn field_infos() -> FieldInfos {
    FieldInfos::new(
        FIELDS
            .iter()
            .map(|&(name, number, io, payloads)| {
                FieldInfo::new(name, number)
                    .with_index_options(io)
                    .with_store_payloads(payloads)
            })
            .collect(),
    )
    .unwrap()
}

fn term_postings(terms: &[Term], io: IndexOptions, payloads: bool) -> Vec<TermPostings> {
    terms
        .iter()
        .map(|t| {
            let mut tp = TermPostings {
                term: t.bytes.clone(),
                docs: (0..t.docs.len())
                    .map(|i| (t.docs[i].0, t.freq(i, io)))
                    .collect(),
                ..TermPostings::default()
            };
            if io.subsumes_positions() {
                tp.positions = t
                    .docs
                    .iter()
                    .map(|(_, o)| o.iter().map(|x| x.0).collect())
                    .collect();
            }
            if io.subsumes_offsets() {
                tp.offsets = t
                    .docs
                    .iter()
                    .map(|(_, o)| o.iter().map(|x| (x.1, x.2)).collect())
                    .collect();
            }
            if payloads {
                for o in t.docs.iter().flat_map(|(_, o)| o) {
                    tp.payload_lengths.push(o.3.len() as u32);
                    tp.payload_bytes.extend_from_slice(&o.3);
                }
            }
            tp
        })
        .collect()
}

/// The six fields' models and their writer inputs.
struct Corpus {
    fis: FieldInfos,
    models: Vec<Vec<Term>>,
    built: Vec<Vec<TermPostings>>,
}

impl Corpus {
    fn new() -> Corpus {
        let models: Vec<Vec<Term>> = FIELDS.iter().map(|f| model(f.1)).collect();
        let built = FIELDS
            .iter()
            .zip(&models)
            .map(|(f, m)| term_postings(m, f.2, f.3))
            .collect();
        Corpus {
            fis: field_infos(),
            models,
            built,
        }
    }

    fn inputs(&self) -> Vec<FieldPostingsInput<'_>> {
        FIELDS
            .iter()
            .zip(&self.built)
            .map(|(&(_, number, io, payloads), terms)| FieldPostingsInput {
                field_number: number,
                index_options: io,
                doc_count: MAX_DOC,
                has_payloads: payloads,
                terms,
            })
            .collect()
    }

    fn write_state<'a>(&'a self, suffix: &'a str) -> SegmentWriteState<'a> {
        SegmentWriteState {
            max_doc: MAX_DOC,
            segment_id: &ID,
            segment_suffix: suffix,
            field_infos: &self.fis,
        }
    }

    fn read_state<'a>(&'a self, suffix: &'a str) -> SegmentReadState<'a> {
        SegmentReadState {
            max_doc: MAX_DOC,
            segment_id: &ID,
            segment_suffix: suffix,
            field_infos: &self.fis,
        }
    }
}

fn compile(pattern: &str) -> CompiledAutomaton {
    let a = RegExp::with_flags(pattern, RegExp::ALL, 0)
        .unwrap()
        .to_automaton()
        .unwrap();
    let a = operations::determinize(&a, operations::DEFAULT_DETERMINIZE_WORK_LIMIT).unwrap();
    CompiledAutomaton::with_options(&a, false, true, false).unwrap()
}

/// The automaton's state after `bytes`, `-1` once dead.
fn run(c: &CompiledAutomaton, bytes: &[u8]) -> i32 {
    let r = c.get_byte_runnable().unwrap();
    bytes
        .iter()
        .fold(0, |s, &b| if s < 0 { s } else { r.step(s, i32::from(b)) })
}

fn accepts(c: &CompiledAutomaton, bytes: &[u8]) -> bool {
    let s = run(c, bytes);
    s >= 0 && c.get_byte_runnable().unwrap().is_accept(s)
}

/// What `intersect(c, start)` must return: the accepted terms after
/// `start`.
fn expected_intersect<'t>(
    terms: &'t [Term],
    c: &CompiledAutomaton,
    start: Option<&[u8]>,
) -> Vec<&'t [u8]> {
    terms
        .iter()
        .map(|t| t.bytes.as_slice())
        .filter(|t| start.is_none_or(|s| *t > s) && accepts(c, t))
        .collect()
}

const PATTERNS: [&str; 11] = [
    "(ab1.*)?", "ab.*", "ab[0-2].", "abc.*x", "m+", "(a|b)", "a.*", "[^a].*", ".*1.*", ".*x", "q.z",
];
const STARTS: [Option<&[u8]>; 9] = [
    None,
    Some(b""),
    Some(b"a"),
    Some(b"ab"),
    Some(b"ab1"),
    Some(b"abc"),
    Some(b"abc3x"),
    Some(b"m"),
    Some(b"zzzz"),
];

/// Seek targets around every term: itself, a byte after, cut short, and
/// its last byte bumped.
fn targets(terms: &[Term]) -> Vec<Vec<u8>> {
    let mut out = vec![
        Vec::new(),
        vec![0],
        vec![0xff, 0xff],
        b"q\x00".to_vec(),
        b"q\x06".to_vec(),
        b"q\xff".to_vec(),
    ];
    for t in terms {
        let b = &t.bytes;
        out.push(b.clone());
        let mut longer = b.clone();
        longer.push(0);
        out.push(longer);
        if let Some((&last, head)) = b.split_last() {
            out.push(head.to_vec());
            let mut bumped = head.to_vec();
            bumped.push(last.wrapping_add(1));
            out.push(bumped);
        }
    }
    // Back and forth, so the enums seek backwards as well as forwards.
    let n = out.len();
    (0..n).map(|i| out[(i * 37) % n].clone()).collect()
}

fn ceil<'t>(terms: &'t [Term], target: &[u8]) -> Option<&'t [u8]> {
    terms
        .iter()
        .map(|t| t.bytes.as_slice())
        .find(|t| *t >= target)
}

#[test]
fn names() {
    assert_eq!(FstPostingsFormat.name(), "FST50");
    assert_eq!(DirectPostingsFormat::default().name(), "Direct");
}

// ---- FSTTermOutputs ----

fn data(bytes: Option<&[u8]>, doc_freq: i32, total_term_freq: i64) -> TermData {
    TermData {
        bytes: bytes.map(<[u8]>::to_vec),
        doc_freq,
        total_term_freq,
    }
}

#[test]
fn term_outputs_share_only_whole_outputs() {
    type O = FstTermOutputs<true>;
    let a = data(Some(b"x"), 3, 5);
    let b = data(Some(b"y"), 3, 5);
    let none = TermData::NO_OUTPUT;
    assert_eq!(O::no_output(), none);
    assert_eq!(O::common(&a, &a), a);
    assert_eq!(O::common(&a, &b), none);
    assert_eq!(O::common(&a, &none), none);
    assert_eq!(O::common(&none, &a), none);

    assert_eq!(O::subtract(&a, &none), a);
    assert_eq!(O::subtract(&a, &a), none);
    assert_eq!(O::subtract(&a, &b), a);

    assert_eq!(O::add(&none, &a), a);
    assert_eq!(O::add(&a, &none), a);
    // A prefix with no bytes and no statistics adds nothing.
    assert_eq!(O::add(&a, &data(None, 0, 7)), a);
    assert_eq!(O::add(&a, &b), b);
}

fn round_trip<O: FstOutputs<Value = TermData>>(values: &[TermData]) {
    let mut buf = Vec::new();
    for v in values {
        O::write(v, &mut buf);
    }
    // The FST's bytes are read backwards from the last.
    buf.reverse();
    let mut r = ReverseReader::new(&buf);
    r.set_position(buf.len() as i64 - 1);
    for v in values {
        let at = r.position();
        let got = O::read(&mut r).unwrap();
        // `writeBytes(b)` with an empty array is `NO bytes`.
        let want_bytes = v.bytes.clone().filter(|b| !b.is_empty());
        assert_eq!(got.bytes, want_bytes, "{v:?}");
        assert_eq!(got.doc_freq, v.doc_freq, "{v:?}");
        let end = r.position();
        r.set_position(at);
        O::skip_output(&mut r).unwrap();
        assert_eq!(r.position(), end, "skip_output of {v:?}");
    }
    assert_eq!(r.position(), -1);
}

#[test]
fn term_outputs_round_trip_with_and_without_positions() {
    let long = vec![7u8; 40];
    let values = [
        data(Some(b"abc"), 4, 4),
        data(Some(b"abc"), 4, 9),
        data(Some(&long), 1, 1),
        data(Some(&long), 2, 300),
        data(Some(b""), 0, -1),
        data(None, 5, 6),
        data(None, 0, -1),
    ];
    round_trip::<FstTermOutputs<true>>(&values);
    let mut r = Vec::new();
    FstTermOutputs::<true>::write(&values[1], &mut r);
    r.reverse();
    let mut rr = ReverseReader::new(&r);
    rr.set_position(r.len() as i64 - 1);
    assert_eq!(
        FstTermOutputs::<true>::read(&mut rr)
            .unwrap()
            .total_term_freq,
        9
    );

    let docs_only: Vec<TermData> = values
        .iter()
        .map(|v| data(v.bytes.as_deref(), v.doc_freq, -1))
        .collect();
    round_trip::<FstTermOutputs<false>>(&docs_only);
}

#[test]
fn term_outputs_refuse_a_negative_metadata_length() {
    // Flags: bytes present, inline length 0, so a vint length follows: -1.
    let mut buf = vec![0x01u8, 0xff, 0xff, 0xff, 0xff, 0x0f];
    buf.reverse();
    for read in [
        |r: &mut ReverseReader<'_>| FstTermOutputs::<true>::read(r).map(|_| ()),
        |r: &mut ReverseReader<'_>| FstTermOutputs::<true>::skip_output(r),
    ] {
        let mut r = ReverseReader::new(&buf);
        r.set_position(buf.len() as i64 - 1);
        assert!(read(&mut r).is_err());
    }
}

// ---- FSTPostingsFormat ----

#[test]
fn fst_reader_walks_seeks_and_intersects_the_model() {
    let corpus = Corpus::new();
    let suffix = "FST50_0";
    let written = FstPostingsFormat
        .write(&corpus.inputs(), &[], &corpus.write_state(suffix))
        .unwrap();
    let reader = FstPostingsFormat
        .open(&written.tfp, &corpus.read_state(suffix))
        .unwrap();
    assert_eq!(reader.size(), FIELDS.len());
    assert_eq!(
        reader.to_string(),
        format!(
            "FSTTermsReader(fields={},delegate=Lucene104PostingsReader)",
            FIELDS.len()
        )
    );
    let doc_in = DocInput::open(&written.postings.doc, &ID, suffix).unwrap();
    for ((name, number, io, payloads), terms) in FIELDS.iter().zip(&corpus.models) {
        let field = reader.terms(name).unwrap();
        assert_eq!(field.field_info().number, *number);
        assert_eq!(field.size(), terms.len() as i64);
        assert_eq!(field.has_payloads(), *payloads);
        assert_eq!(field.has_offsets(), io.subsumes_offsets());
        assert!(field.to_string().starts_with("FSTTerms(terms="));

        // Walk: every term, its statistics and documents.
        let mut walk = field.iter();
        for t in terms {
            assert_eq!(walk.next().unwrap(), Some(t.bytes.as_slice()), "{name}");
            assert_eq!(walk.term(), Some(t.bytes.as_slice()));
            assert_eq!(walk.doc_freq(), t.docs.len() as i32);
            assert_eq!(walk.total_term_freq(), t.total_term_freq(*io), "{name}");
            assert_eq!(walk.term_data().doc_freq, t.docs.len() as i32);
            let p = walk.postings(&doc_in, PostingsFlags::Freqs).unwrap();
            assert_eq!(p.docs, t.docs.iter().map(|d| d.0).collect::<Vec<_>>());
        }
        assert_eq!(walk.next().unwrap(), None);
        assert!(matches!(walk.ord(), Err(Error::Unsupported(_))));

        // Seeks, forwards and backwards on one enum.
        let mut seek = field.iter();
        for target in targets(terms) {
            let want = ceil(terms, &target);
            let status = seek.seek_ceil(&target).unwrap();
            match want {
                None => assert_eq!(status, SeekStatus::End, "{name} {target:?}"),
                Some(w) => {
                    let found = if w == target.as_slice() {
                        SeekStatus::Found
                    } else {
                        SeekStatus::NotFound
                    };
                    assert_eq!(status, found, "{name} {target:?}");
                    assert_eq!(seek.term(), Some(w));
                    let df = terms.iter().find(|t| t.bytes == w).unwrap().docs.len() as i32;
                    assert_eq!(seek.doc_freq(), df);
                }
            }
            let exact = seek.seek_exact(&target).unwrap();
            assert_eq!(exact, want == Some(target.as_slice()), "{name} {target:?}");
            if exact {
                assert_eq!(seek.term(), Some(target.as_slice()));
                assert!(seek.term_metadata().is_ok());
            }
        }

        // Intersections.
        for pattern in PATTERNS {
            let c = compile(pattern);
            for start in STARTS {
                let mut e = field.intersect(&c, start).unwrap();
                let mut got = Vec::new();
                while let Some(t) = e.next().unwrap() {
                    let t = t.to_vec();
                    let model = terms.iter().find(|m| m.bytes == t).unwrap();
                    assert_eq!(e.doc_freq(), model.docs.len() as i32);
                    assert_eq!(e.total_term_freq(), model.total_term_freq(*io));
                    assert_eq!(e.term_metadata().unwrap(), e.term_metadata().unwrap());
                    let p = e.postings(&doc_in, PostingsFlags::DocsOnly).unwrap();
                    assert_eq!(p.docs.len(), model.docs.len());
                    got.push(t);
                }
                let want: Vec<Vec<u8>> = expected_intersect(terms, &c, start)
                    .into_iter()
                    .map(<[u8]>::to_vec)
                    .collect();
                assert_eq!(got, want, "{name} {pattern} {start:?}");
            }
        }
        // `seekCeil` on a fresh intersect enum (Java calls it only from the
        // constructor, at the root).
        let c = compile("ab.*");
        let mut e = field.intersect(&c, None).unwrap();
        assert_eq!(e.seek_ceil(b"ab05").unwrap(), SeekStatus::Found);
        assert_eq!(e.term(), Some(&b"ab05"[..]));
        assert!(field
            .intersect(&c, None)
            .unwrap()
            .seek_exact(b"ab10")
            .unwrap());
        assert!(!field
            .intersect(&c, None)
            .unwrap()
            .seek_exact(b"ab100")
            .unwrap());
    }
    assert!(reader.terms("nope").is_none());
    assert_eq!(reader.field_names().count(), FIELDS.len());
    assert!(
        format!("{:?}", FstTermsWriter::new(&corpus.write_state(suffix)))
            .contains("FstTermsWriter")
    );
}

#[test]
fn fst_intersect_refuses_a_non_normal_automaton() {
    let corpus = Corpus::new();
    let suffix = "FST50_0";
    let written = FstPostingsFormat
        .write(&corpus.inputs(), &[], &corpus.write_state(suffix))
        .unwrap();
    let reader = FstPostingsFormat
        .open(&written.tfp, &corpus.read_state(suffix))
        .unwrap();
    let all = compile(".*");
    assert_ne!(all.automaton_type, AutomatonType::NORMAL);
    assert!(matches!(
        reader.terms("f").unwrap().intersect(&all, None),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn fst_reader_refuses_damaged_and_mismatched_dictionaries() {
    let corpus = Corpus::new();
    let suffix = "FST50_0";
    let written = FstPostingsFormat
        .write(&corpus.inputs(), &[], &corpus.write_state(suffix))
        .unwrap();
    let state = corpus.read_state(suffix);
    // Too short for anything.
    assert!(FstPostingsFormat.open(&written.tfp[..10], &state).is_err());
    // A flipped byte in the body: the checksum.
    let mut bad = written.tfp.clone();
    let mid = bad.len() / 2;
    bad[mid] ^= 0x55;
    assert!(FstPostingsFormat.open(&bad, &state).is_err());
    // Another segment's id.
    let other = SegmentReadState {
        segment_id: b"someothersegment",
        ..state
    };
    assert!(FstPostingsFormat.open(&written.tfp, &other).is_err());
    // Field summaries `checkFieldSummary` refuses: more documents than
    // the segment has, and fewer postings than documents.
    let small = SegmentReadState {
        max_doc: 10,
        ..state
    };
    assert!(FstPostingsFormat.open(&written.tfp, &small).is_err());
    let few = [FieldPostingsInput {
        field_number: 1,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 250,
        has_payloads: false,
        terms: &corpus.built[1][6..7],
    }];
    let w = FstPostingsFormat
        .write(&few, &[], &corpus.write_state(suffix))
        .unwrap();
    assert!(FstPostingsFormat.open(&w.tfp, &state).is_err());
    // A writer error surfaces: terms out of order.
    let mut unsorted = corpus.built[1].clone();
    unsorted.swap(0, 1);
    let input = [FieldPostingsInput {
        field_number: 1,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: MAX_DOC,
        has_payloads: false,
        terms: &unsorted,
    }];
    assert!(FstPostingsFormat
        .write(&input, &[], &corpus.write_state(suffix))
        .is_err());
}

// ---- DirectPostingsFormat ----

struct DirectGroup {
    out: crate::postings_writer::Output,
}

impl DirectGroup {
    fn write(corpus: &Corpus) -> DirectGroup {
        let out = DirectPostingsFormat::default()
            .write(&corpus.inputs(), &[], &corpus.write_state("Direct_0"))
            .unwrap();
        DirectGroup { out }
    }

    fn files(&self) -> DirectFiles<'_> {
        DirectFiles {
            tim: &self.out.tim,
            tip: &self.out.tip,
            tmd: &self.out.tmd,
            doc: &self.out.doc,
            pos: Some(&self.out.pos),
            pay: Some(&self.out.pay),
        }
    }
}

/// Every document, freq, position, offset and payload of `p`, against the
/// model term.
fn check_postings(
    mut p: DirectPostingsEnum<'_>,
    t: &Term,
    io: IndexOptions,
    payloads: bool,
    positions: bool,
    what: &str,
) {
    assert_eq!(p.doc_id(), -1, "{what}");
    assert!(p.cost() > 0);
    for (i, (doc, occ)) in t.docs.iter().enumerate() {
        assert_eq!(p.next_doc(), *doc, "{what}");
        assert_eq!(p.doc_id(), *doc, "{what}");
        assert_eq!(p.freq(), t.freq(i, io), "{what} doc {doc}");
        if positions && io.subsumes_positions() {
            for o in occ {
                assert_eq!(p.next_position(), o.0, "{what} doc {doc}");
                if io.subsumes_offsets() {
                    assert_eq!((p.start_offset(), p.end_offset()), (o.1, o.2), "{what}");
                } else {
                    assert_eq!((p.start_offset(), p.end_offset()), (-1, -1), "{what}");
                }
                let want = (payloads && !o.3.is_empty()).then_some(o.3.as_slice());
                assert_eq!(p.payload(), want, "{what} doc {doc}");
            }
        } else {
            assert_eq!(p.next_position(), -1);
            assert_eq!((p.start_offset(), p.end_offset()), (-1, -1));
            assert_eq!(p.payload(), None);
        }
    }
    assert_eq!(p.next_doc(), NO_MORE_DOCS, "{what}");
    assert_eq!(p.doc_id(), NO_MORE_DOCS, "{what}");
}

/// `advance` from a fresh enum and along one, against the model.
fn check_advance(field: &DirectField, ord: i64, t: &Term, io: IndexOptions, what: &str) {
    let first_at = |target: i32| t.docs.iter().position(|d| d.0 >= target);
    let mut e = field.iter();
    e.seek_exact_ord(ord).unwrap();
    for target in [0, 1, 5, 77, 150, 299, 300, 1000] {
        let mut p = e.postings(true).unwrap();
        let got = p.advance(target);
        match first_at(target) {
            Some(i) => {
                assert_eq!(got, t.docs[i].0, "{what} advance({target})");
                assert_eq!(p.freq(), t.freq(i, io));
                if io.subsumes_positions() {
                    assert_eq!(p.next_position(), t.docs[i].1[0].0, "{what}");
                }
            }
            None => assert_eq!(got, NO_MORE_DOCS, "{what} advance({target})"),
        }
    }
    for (positions, delta) in [(true, 1), (false, 7), (true, 40)] {
        let mut p = e.postings(positions).unwrap();
        let mut doc = -1;
        loop {
            let target = doc + delta;
            doc = p.advance(target);
            match first_at(target) {
                Some(i) => assert_eq!(doc, t.docs[i].0, "{what} advance({target}) by {delta}"),
                None => {
                    assert_eq!(doc, NO_MORE_DOCS, "{what}");
                    break;
                }
            }
        }
    }
}

fn check_direct_fields(corpus: &Corpus, fields: &DirectFields) {
    assert_eq!(fields.size(), FIELDS.len());
    assert_eq!(fields.field_names().count(), FIELDS.len());
    assert_eq!(
        fields.to_string(),
        format!("DirectFields(fields={})", FIELDS.len())
    );
    assert!(fields.terms("nope").is_none());
    for ((name, _, io, payloads), terms) in FIELDS.iter().zip(&corpus.models) {
        let field = fields.terms(name).unwrap();
        assert_eq!(field.size(), terms.len() as i64);
        assert_eq!(field.has_freqs(), *io != IndexOptions::Docs);
        assert_eq!(field.has_positions(), io.subsumes_positions());
        assert_eq!(field.has_offsets(), io.subsumes_offsets());
        assert_eq!(field.has_payloads(), *payloads);
        assert_eq!(field.doc_count(), MAX_DOC);
        let docs: i64 = terms.iter().map(|t| t.docs.len() as i64).sum();
        assert_eq!(field.sum_doc_freq(), docs);
        if *io != IndexOptions::Docs {
            let ttf: i64 = terms.iter().map(|t| t.total_term_freq(*io)).sum();
            assert_eq!(field.sum_total_term_freq(), ttf);
        }
        assert!(field.to_string().starts_with("DirectTerms(terms="));

        let mut walk = field.iter();
        assert_eq!(walk.term(), None);
        assert_eq!(walk.doc_freq(), 0);
        assert_eq!(walk.total_term_freq(), 0);
        assert!(walk.postings(true).is_none());
        for (ord, t) in terms.iter().enumerate() {
            let what = format!("{name} {:?}", String::from_utf8_lossy(&t.bytes));
            assert_eq!(walk.next(), Some(t.bytes.as_slice()), "{what}");
            assert_eq!(walk.ord(), ord as i64);
            assert_eq!(walk.doc_freq(), t.docs.len() as i32, "{what}");
            assert_eq!(walk.total_term_freq(), t.total_term_freq(*io), "{what}");
            check_postings(walk.postings(true).unwrap(), t, *io, *payloads, true, &what);
            check_postings(
                walk.postings(false).unwrap(),
                t,
                *io,
                *payloads,
                false,
                &what,
            );
            check_advance(field, ord as i64, t, *io, &what);
        }
        assert_eq!(walk.next(), None);
        assert_eq!(walk.next(), None);
        assert!(walk.seek_exact_ord(-1).is_err());
        assert!(walk.seek_exact_ord(terms.len() as i64).is_err());

        let mut seek = field.iter();
        for target in targets(terms) {
            let want = ceil(terms, &target);
            let before = seek.term();
            let status = seek.seek_ceil(&target);
            match want {
                None => {
                    assert_eq!(status, SeekStatus::End, "{name} {target:?}");
                    assert_eq!(seek.term(), before, "End leaves the enum where it was");
                }
                Some(w) => {
                    let found = if w == target.as_slice() {
                        SeekStatus::Found
                    } else {
                        SeekStatus::NotFound
                    };
                    assert_eq!(status, found, "{name} {target:?}");
                    assert_eq!(seek.term(), Some(w));
                }
            }
            assert_eq!(seek.seek_exact(&target), want == Some(target.as_slice()));
        }
    }
}

#[test]
fn direct_reader_matches_the_model() {
    let corpus = Corpus::new();
    let group = DirectGroup::write(&corpus);
    let state = corpus.read_state("Direct_0");
    for format in [
        DirectPostingsFormat::default(),
        DirectPostingsFormat::new(2, 1),
        DirectPostingsFormat::new(1000, 1000),
    ] {
        let fields = format.open(&group.files(), &state).unwrap();
        check_direct_fields(&corpus, &fields);
    }
}

fn check_direct_intersect(corpus: &Corpus, fields: &DirectFields) {
    for ((name, _, io, _), terms) in FIELDS.iter().zip(&corpus.models) {
        let field = fields.terms(name).unwrap();
        for pattern in PATTERNS {
            let c = compile(pattern);
            for start in STARTS {
                let live = start.is_none_or(|s| run(&c, s) >= 0);
                let result = field.intersect(&c, start);
                // Java assumes the automaton accepts the start term's
                // prefix; this port refuses a start it cannot walk.
                let Ok(mut e) =
                    result.map_err(|e| assert!(!live && matches!(e, Error::Unsupported(_))))
                else {
                    continue;
                };
                assert_eq!(e.term(), None);
                assert_eq!(e.doc_freq(), 0);
                assert_eq!(e.total_term_freq(), 0);
                assert!(e.postings(false).is_none());
                let mut got = Vec::new();
                while let Some(t) = e.next() {
                    let model = terms.iter().find(|m| m.bytes == t).unwrap();
                    assert_eq!(e.term(), Some(t));
                    assert_eq!(e.doc_freq(), model.docs.len() as i32);
                    assert_eq!(e.total_term_freq(), model.total_term_freq(*io));
                    assert_eq!(terms[e.ord() as usize].bytes, t);
                    assert_eq!(e.postings(false).unwrap().next_doc(), model.docs[0].0);
                    got.push(t);
                }
                assert_eq!(
                    got,
                    expected_intersect(terms, &c, start),
                    "{name} {pattern} {start:?}"
                );
                assert!(matches!(e.seek_ceil(b"a"), Err(Error::Unsupported(_))));
            }
        }
        assert!(matches!(
            field.intersect(&compile(".*"), None),
            Err(Error::Unsupported(_))
        ));
        assert!(
            format!("{:?}", field.intersect(&compile("ab.*"), None).unwrap()).contains("term_ord")
        );
    }
}

#[test]
fn direct_intersect_matches_a_brute_force_scan() {
    let corpus = Corpus::new();
    let group = DirectGroup::write(&corpus);
    let state = corpus.read_state("Direct_0");
    for format in [
        DirectPostingsFormat::default(),
        DirectPostingsFormat::new(2, 32),
        DirectPostingsFormat::new(1, 32),
    ] {
        let fields = format.open(&group.files(), &state).unwrap();
        check_direct_intersect(&corpus, &fields);
    }
}

#[test]
fn direct_intersect_refuses_a_dead_start_term() {
    let corpus = Corpus::new();
    let group = DirectGroup::write(&corpus);
    let fields = DirectPostingsFormat::new(2, 32)
        .open(&group.files(), &corpus.read_state("Direct_0"))
        .unwrap();
    // "ab" has skips; the automaton dies on 'b'.
    let c = compile("a[0-9].*");
    assert!(matches!(
        fields.terms("f").unwrap().intersect(&c, Some(b"ab05")),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn direct_reader_refuses_damaged_or_missing_files() {
    let corpus = Corpus::new();
    let group = DirectGroup::write(&corpus);
    let state = corpus.read_state("Direct_0");
    let format = DirectPostingsFormat::default();
    let files = group.files();
    // A field indexes positions, but no `.pos`.
    assert!(format
        .open(&DirectFiles { pos: None, ..files }, &state)
        .is_err());
    for which in 0..6 {
        let mut bufs = [
            group.out.tim.clone(),
            group.out.tip.clone(),
            group.out.tmd.clone(),
            group.out.doc.clone(),
            group.out.pos.clone(),
            group.out.pay.clone(),
        ];
        // Truncated: no footer.
        bufs[which].truncate(4);
        let damaged = DirectFiles {
            tim: &bufs[0],
            tip: &bufs[1],
            tmd: &bufs[2],
            doc: &bufs[3],
            pos: Some(&bufs[4]),
            pay: Some(&bufs[5]),
        };
        assert!(format.open(&damaged, &state).is_err(), "file {which}");
    }
    // A flipped byte in the `.doc` body: the checksum.
    let mut doc = group.out.doc.clone();
    let mid = doc.len() / 2;
    doc[mid] ^= 0x55;
    assert!(format
        .open(&DirectFiles { doc: &doc, ..files }, &state)
        .is_err());
    // A writer error surfaces.
    let mut unsorted = corpus.built[0].clone();
    unsorted.swap(1, 2);
    let input = [FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::Docs,
        doc_count: MAX_DOC,
        has_payloads: false,
        terms: &unsorted,
    }];
    assert!(format
        .write(&input, &[], &corpus.write_state("Direct_0"))
        .is_err());
}

#[test]
fn direct_reader_without_positions_needs_no_pos_file() {
    let corpus = Corpus::new();
    let inputs: Vec<_> = corpus.inputs().into_iter().take(2).collect();
    let fis = FieldInfos::new(vec![
        FieldInfo::new("d", 0).with_index_options(IndexOptions::Docs),
        FieldInfo::new("f", 1).with_index_options(IndexOptions::DocsAndFreqs),
    ])
    .unwrap();
    let w = DirectPostingsFormat::default()
        .write(
            &inputs,
            &[],
            &SegmentWriteState {
                max_doc: MAX_DOC,
                segment_id: &ID,
                segment_suffix: "Direct_0",
                field_infos: &fis,
            },
        )
        .unwrap();
    let fields = DirectPostingsFormat::default()
        .open(
            &DirectFiles {
                tim: &w.tim,
                tip: &w.tip,
                tmd: &w.tmd,
                doc: &w.doc,
                pos: None,
                pay: None,
            },
            &SegmentReadState {
                max_doc: MAX_DOC,
                segment_id: &ID,
                segment_suffix: "Direct_0",
                field_infos: &fis,
            },
        )
        .unwrap();
    assert_eq!(fields.size(), 2);
    let f = fields.terms("f").unwrap();
    let mut e = f.iter();
    assert!(e.seek_exact(b"ab00"));
    let mut p = e.postings(true).unwrap();
    assert_eq!(p.next_position(), -1);
}
