//! Unit tests for the SimpleText postings: the line primitives' boundaries,
//! the writer's input checks, and what the reader and the skip reader do
//! with a file that is not what the writer would write. Format fidelity is
//! `tests/codecs_simpletext_fixtures.rs`' job.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use super::fields_reader::{SimpleTextFieldsReader, SimpleTextImpactsEnum, SimpleTextPostings};
use super::fields_writer::{postings_file_name, write};
use super::skip::{self, Impacts, SimpleTextSkipReader};
use super::util::{self, parse_int, parse_long};
use super::*;
use crate::field_infos::{FieldInfo, FieldInfos};
use crate::postings_writer::{FieldNorms, FieldPostingsInput, TermPostings};
use crate::segment_state::{SegmentReadState, SegmentWriteState};

const ID: [u8; 16] = *b"simpletext-tests";

fn infos() -> FieldInfos {
    FieldInfos::new(vec![
        FieldInfo::new("f", 0)
            .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
            .with_store_payloads(true),
        FieldInfo::new("g", 1)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
        FieldInfo::new("h", 2).with_index_options(IndexOptions::DocsAndFreqsAndPositions),
    ])
    .unwrap()
}

fn write_state(fis: &FieldInfos, max_doc: i32) -> SegmentWriteState<'_> {
    SegmentWriteState {
        max_doc,
        segment_id: &ID,
        segment_suffix: "",
        field_infos: fis,
    }
}

fn read_state(fis: &FieldInfos, max_doc: i32) -> SegmentReadState<'_> {
    SegmentReadState {
        max_doc,
        segment_id: &ID,
        segment_suffix: "",
        field_infos: fis,
    }
}

/// `n` documents `0, step, 2*step, ...`, each with `freq` positions, offsets
/// and (for odd docs) a payload holding the escape bytes.
fn positions_term(term: &[u8], n: i32, step: i32, freq: i32) -> TermPostings {
    let mut tp = TermPostings {
        term: term.to_vec(),
        ..TermPostings::default()
    };
    for i in 0..n {
        let doc = i * step;
        tp.docs.push((doc, freq));
        tp.positions.push((0..freq).map(|p| p * 2).collect());
        tp.offsets
            .push((0..freq).map(|p| (p * 3, p * 3 + 2)).collect());
        for _ in 0..freq {
            let payload: &[u8] = if doc % 2 == 1 { b"\n\\" } else { b"" };
            tp.payload_lengths.push(payload.len() as u32);
            tp.payload_bytes.extend_from_slice(payload);
        }
    }
    tp
}

fn docs_term(term: &[u8], docs: impl IntoIterator<Item = i32>) -> TermPostings {
    TermPostings {
        term: term.to_vec(),
        docs: docs.into_iter().map(|d| (d, 1)).collect(),
        ..TermPostings::default()
    }
}

fn input<'a>(number: i32, io: IndexOptions, terms: &'a [TermPostings]) -> FieldPostingsInput<'a> {
    FieldPostingsInput {
        field_number: number,
        index_options: io,
        doc_count: 0,
        has_payloads: number == 0,
        terms,
    }
}

const MAX_DOC: i32 = 1000;

/// A file with `f` (`a`: 40 docs, skip lists over two levels; `b`: 3 docs),
/// `g` (`x`: 10 docs, a one-level skip list), and `f`'s norms.
fn sample() -> Vec<u8> {
    let fis = infos();
    let f = [
        positions_term(b"a", 40, 3, 2),
        positions_term(b"b", 3, 5, 1),
    ];
    let g = [docs_term(b"x", 0..10)];
    let norms: Vec<i64> = (0..MAX_DOC as i64).map(|d| 1 + d % 5).collect();
    write(
        &[
            input(0, IndexOptions::DocsAndFreqsAndPositionsAndOffsets, &f),
            input(1, IndexOptions::Docs, &g),
        ],
        &[
            FieldNorms {
                field_number: 0,
                values: &norms,
            },
            FieldNorms {
                field_number: 1,
                values: &norms,
            },
        ],
        &write_state(&fis, MAX_DOC),
    )
    .unwrap()
}

fn open(bytes: Vec<u8>) -> Result<SimpleTextFieldsReader> {
    let fis = infos();
    SimpleTextFieldsReader::open(bytes, &read_state(&fis, MAX_DOC))
}

/// `body` + `END` + a checksum line that matches.
fn file(body: &str) -> Vec<u8> {
    let mut out = body.as_bytes().to_vec();
    out.extend_from_slice(b"END\n");
    util::write_checksum(&mut out);
    out
}

/// `bytes` with its checksum line recomputed after `edit`.
fn refooter(bytes: &[u8], edit: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let at = bytes.len() - b"checksum 00000000000000000000\n".len();
    let mut body = bytes[..at].to_vec();
    edit(&mut body);
    util::write_checksum(&mut body);
    body
}

fn replace(body: &mut Vec<u8>, from: &[u8], to: &[u8]) {
    let at = body
        .windows(from.len())
        .position(|w| w == from)
        .unwrap_or_else(|| panic!("{:?} not in the file", String::from_utf8_lossy(from)));
    body.splice(at..at + from.len(), to.iter().copied());
}

fn replace_all(body: &mut Vec<u8>, from: &[u8], to: &[u8]) {
    let mut out = Vec::new();
    let mut i = 0;
    while i < body.len() {
        if body[i..].starts_with(from) {
            out.extend_from_slice(to);
            i += from.len();
        } else {
            out.push(body[i]);
            i += 1;
        }
    }
    *body = out;
}

fn err<T: std::fmt::Debug>(r: Result<T>) -> String {
    r.unwrap_err().to_string()
}

// --- util ---------------------------------------------------------------

#[test]
fn escapes_round_trip() {
    let mut out = Vec::new();
    util::write(&mut out, b"a\nb\\c");
    util::write_newline(&mut out);
    assert_eq!(out, b"a\\\nb\\\\c\n");
    let (mut pos, mut scratch) = (0, Vec::new());
    util::read_line(&out, &mut pos, &mut scratch).unwrap();
    assert_eq!((scratch.as_slice(), pos), (&b"a\nb\\c"[..], out.len()));
    // No newline; an escape with nothing behind it.
    for bad in [&b"abc"[..], b"ab\\"] {
        let mut pos = 0;
        assert!(util::read_line(bad, &mut pos, &mut scratch).is_err());
    }
}

#[test]
fn check_footer_paths() {
    let good = file("");
    let mut pos = 4;
    util::check_footer(&good, 0, &mut pos).unwrap();
    assert_eq!(pos, good.len());
    let mut pos = 4;
    assert!(err(util::check_footer(&good, 5, &mut pos)).contains("outside"));
    let mut pos = 0;
    assert!(err(util::check_footer(&good, 0, &mut pos)).contains("expected checksum line"));
    let mut pos = 4;
    assert!(err(util::check_footer(&good, 1, &mut pos)).contains("checksum failure"));
    let mut longer = good.clone();
    longer.push(b'x');
    let mut pos = 4;
    assert!(err(util::check_footer(&longer, 0, &mut pos)).contains("Unexpected stuff"));
}

#[test]
fn from_bytes_ref_string_shapes() {
    assert_eq!(
        util::from_bytes_ref_string("[61 62 ff]").unwrap(),
        vec![0x61, 0x62, 0xff]
    );
    // `(byte) Integer.parseInt(part, 16)` keeps the low byte.
    assert_eq!(util::from_bytes_ref_string("[1ff]").unwrap(), vec![0xff]);
    assert!(util::from_bytes_ref_string("[]").unwrap().is_empty());
    // `String.split(" ")` drops trailing empty strings, not leading ones.
    assert_eq!(util::from_bytes_ref_string("[61 ]").unwrap(), vec![0x61]);
    assert_eq!(util::from_bytes_ref_string("[61  ]").unwrap(), vec![0x61]);
    assert!(util::from_bytes_ref_string("[ ]").unwrap().is_empty());
    for bad in ["61", "[61", "61]", "[zz]", "[61  62]", "[ 61]"] {
        assert!(matches!(
            util::from_bytes_ref_string(bad),
            Err(Error::IllegalArgument(_))
        ));
    }
}

#[test]
fn parse_int_bounds() {
    for (s, v) in [
        ("0", 0),
        ("-0", 0),
        ("17", 17),
        ("2147483647", i32::MAX),
        ("-2147483648", i32::MIN),
    ] {
        assert_eq!(parse_int(s.as_bytes()).unwrap(), v, "{s}");
    }
    for s in [
        "",
        "-",
        "+1",
        "1a",
        " 1",
        "2147483648",
        "-2147483649",
        "99999999999999999999",
    ] {
        assert!(
            err(parse_int(s.as_bytes())).contains("For input string"),
            "{s}"
        );
    }
    assert!(parse_int(&[0xd9, 0xa1]).is_err());
}

#[test]
fn parse_long_bounds() {
    assert_eq!(parse_long(b"+5").unwrap(), 5);
    assert_eq!(parse_long(b"-9223372036854775808").unwrap(), i64::MIN);
    assert_eq!(parse_long(b"9223372036854775807").unwrap(), i64::MAX);
    for s in [
        "",
        "+",
        "9223372036854775808",
        "-9223372036854775809",
        "1.0",
    ] {
        assert!(parse_long(s.as_bytes()).is_err(), "{s}");
    }
}

// --- mod ----------------------------------------------------------------

#[test]
fn helpers() {
    assert!(!subsumes_freqs(IndexOptions::None));
    assert!(!subsumes_freqs(IndexOptions::Docs));
    for io in [
        IndexOptions::DocsAndFreqs,
        IndexOptions::DocsAndCustomFreqs,
        IndexOptions::DocsAndFreqsAndPositions,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
    ] {
        assert!(subsumes_freqs(io), "{io:?}");
    }
    assert!(feature_requested(ALL, PAYLOADS));
    assert!(feature_requested(POSITIONS, FREQS));
    assert!(!feature_requested(FREQS, POSITIONS));
    assert!(!feature_requested(NONE, FREQS));
    assert_eq!(postings_file_name("_3", ""), "_3.pst");
    assert_eq!(
        postings_file_name("_3", "SimpleText_0"),
        "_3_SimpleText_0.pst"
    );
    assert!(file_pointer(b"abc", 3, "x").is_ok());
    assert!(err(file_pointer(b"abc", 4, "x")).contains("outside the file"));
    assert!(file_pointer(b"abc", -1, "x").is_err());
    assert_eq!(Error::Unsupported("ord").to_string(), "ord");
    assert_eq!(Error::IllegalArgument("bad".into()).to_string(), "bad");
}

// --- writer -------------------------------------------------------------

#[test]
fn writer_refuses_inconsistent_input() {
    let fis = infos();
    let state = write_state(&fis, MAX_DOC);
    let io = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
    let run = |terms: &[TermPostings], number: i32| -> String {
        let r = write(&[input(number, io, terms)], &[], &state);
        match r {
            Err(Error::IllegalArgument(m)) => m,
            other => panic!("{other:?}"),
        }
    };
    let good = positions_term(b"t", 2, 1, 2);
    assert!(run(std::slice::from_ref(&good), 7).contains("unknown field number"));

    let mut t = good.clone();
    t.positions.pop();
    assert!(run(&[t], 0).contains("positions lists"));
    let mut t = good.clone();
    t.offsets.pop();
    assert!(run(&[t], 0).contains("offsets lists"));
    let mut t = good.clone();
    t.positions[1].pop();
    assert!(run(&[t], 0).contains("freq 2"));
    let mut t = good.clone();
    t.offsets[0].pop();
    assert!(run(&[t], 0).contains("freq 2"));
    let mut t = good.clone();
    t.payload_lengths.truncate(1);
    assert!(run(&[t], 0).contains("missing payload length"));
    let mut t = good;
    t.payload_lengths[3] = 9;
    assert!(run(&[t], 0).contains("payload past its run"));
}

#[test]
fn writer_skips_empty_terms_and_fields() {
    let fis = infos();
    let empty = [docs_term(b"z", [])];
    let out = write(
        &[
            input(1, IndexOptions::Docs, &empty),
            input(1, IndexOptions::Docs, &[]),
        ],
        &[],
        &write_state(&fis, MAX_DOC),
    )
    .unwrap();
    assert_eq!(out, file(""));
}

// --- reader -------------------------------------------------------------

#[test]
fn reader_reads_its_writer() {
    let r = open(sample()).unwrap();
    r.check_integrity().unwrap();
    assert_eq!(r.field_names().collect::<Vec<_>>(), ["f", "g"]);
    assert_eq!(r.to_string(), "SimpleTextFieldsReader(fields=2)");
    assert!(format!("{r:?}").contains("SimpleTextFieldsReader"));
    let f = r.terms("f").unwrap().unwrap();
    // Loaded once, then cached.
    assert!(std::ptr::eq(f, r.terms("f").unwrap().unwrap()));
    assert_eq!((f.size(), f.doc_count(), f.sum_doc_freq()), (2, 42, 43));
    assert_eq!(f.sum_total_term_freq(), 83);
    assert!(f.has_freqs() && f.has_positions() && f.has_offsets() && f.has_payloads());
    assert_eq!(
        f.to_string(),
        "SimpleTextTerms(terms=2,postings=43,positions=83,docs=42)"
    );
    assert_eq!(format!("{f:?}"), f.to_string());

    let mut te = f.iterator();
    assert_eq!(te.term(), b"");
    assert!(te.seek_exact(b"b").unwrap());
    assert!(format!("{te:?}").contains("doc_freq: 3"));
    assert_eq!((te.doc_freq(), te.total_term_freq()), (3, 3));
    assert!(matches!(te.ord(), Err(Error::Unsupported(_))));
    assert!(matches!(te.seek_exact_ord(0), Err(Error::Unsupported(_))));
    assert!(!te.seek_exact(b"ab").unwrap());
    assert_eq!(te.seek_ceil(b"ab").unwrap(), SeekStatus::NotFound);
    assert_eq!(te.term(), b"b");
    assert_eq!(te.seek_ceil(b"a").unwrap(), SeekStatus::Found);
    assert_eq!(te.seek_ceil(b"c").unwrap(), SeekStatus::End);

    // Positions, offsets and payloads of `b`'s second document (doc 5).
    assert!(te.seek_exact(b"b").unwrap());
    let mut pe = te.postings(ALL).unwrap();
    assert_eq!(pe.cost(), 3);
    assert_eq!(pe.advance(1).unwrap(), 5);
    assert_eq!(pe.freq(), 1);
    assert_eq!(pe.next_position().unwrap(), 0);
    assert_eq!((pe.start_offset(), pe.end_offset()), (0, 2));
    assert_eq!(pe.payload(), Some(&b"\n\\"[..]));
    assert_eq!(pe.next_doc().unwrap(), 10);
    pe.next_position().unwrap();
    assert_eq!(pe.payload(), None);
    assert_eq!(pe.next_doc().unwrap(), NO_MORE_DOCS);
    assert_eq!(pe.next_doc().unwrap(), NO_MORE_DOCS);

    // A docs enum on a positions field: Java's -1s.
    let mut de = te.postings(FREQS).unwrap();
    assert!(matches!(de, SimpleTextPostings::Docs(_)));
    assert_eq!(de.next_doc().unwrap(), 0);
    assert_eq!(de.freq(), 1);
    assert_eq!(de.next_position().unwrap(), -1);
    assert_eq!((de.start_offset(), de.end_offset()), (-1, -1));
    assert_eq!(de.payload(), None);

    let g = r.terms("g").unwrap().unwrap();
    assert!(!g.has_freqs() && !g.has_positions() && !g.has_offsets() && !g.has_payloads());
    let mut te = g.iterator();
    assert_eq!(te.next().unwrap(), Some(&b"x"[..]));
    assert_eq!((te.doc_freq(), te.total_term_freq()), (10, 10));
    assert_eq!(te.next().unwrap(), None);
    // DOCS: freq is always 1, positions are refused.
    assert!(te.seek_exact(b"x").unwrap());
    let mut de = te.postings(ALL).unwrap();
    assert!(matches!(de, SimpleTextPostings::Docs(_)));
    assert_eq!(de.advance(9).unwrap(), 9);
    assert_eq!(de.freq(), 1);
    assert_eq!(de.next_doc().unwrap(), NO_MORE_DOCS);
    assert_eq!(de.next_doc().unwrap(), NO_MORE_DOCS);
    assert!(r.terms("h").unwrap().is_none());
}

#[test]
fn impacts_slow_and_skip() {
    let r = open(sample()).unwrap();
    let f = r.terms("f").unwrap().unwrap();
    let mut te = f.iterator();
    assert!(te.seek_exact(b"b").unwrap());
    let mut ie = te.impacts(FREQS).unwrap();
    assert!(matches!(ie, SimpleTextImpactsEnum::Slow(_)));
    ie.advance_shallow(100).unwrap();
    let imp = ie.impacts().unwrap();
    assert_eq!(imp.num_levels(), 1);
    assert_eq!(imp.doc_id_up_to(0), Some(NO_MORE_DOCS));
    assert_eq!(imp.impacts(0).unwrap().len(), 1);
    assert_eq!(imp.impacts(0).unwrap()[0].freq, i32::MAX);
    assert_eq!(
        (imp.doc_id_up_to(1), imp.impacts(1).is_none()),
        (None, true)
    );

    // `a`: 40 docs 0, 3, ..., 117; blocks of 8, two levels.
    assert!(te.seek_exact(b"a").unwrap());
    for flags in [FREQS, ALL] {
        let mut ie = te.impacts(flags).unwrap();
        assert!(matches!(ie, SimpleTextImpactsEnum::Skip(_)));
        assert_eq!(ie.postings().advance(30).unwrap(), 30);
        let imp = ie.impacts().unwrap();
        assert_eq!(imp.num_levels(), 2);
        // Level 0's block holds docs 24..45; level 1 docs 0..69.
        assert_eq!(imp.doc_id_up_to(0), Some(45));
        assert_eq!(imp.doc_id_up_to(1), Some(69));
        assert!(imp.impacts(1).unwrap().iter().all(|i| i.freq == 2));
        assert_eq!(imp.doc_id_up_to(2), None);
        assert!(imp.impacts(2).is_none());
        // Past the last block: one dummy level.
        ie.advance_shallow(200).unwrap();
        let imp = ie.impacts().unwrap();
        assert_eq!(imp.num_levels(), 1);
        assert_eq!(imp.impacts(0).unwrap()[0].freq, i32::MAX);
        assert_eq!(ie.postings().advance(115).unwrap(), 117);
        assert_eq!(ie.postings().next_doc().unwrap(), NO_MORE_DOCS);
    }

    let g = r.terms("g").unwrap().unwrap();
    let mut te = g.iterator();
    assert!(te.seek_exact(b"x").unwrap());
    let mut ie = te.impacts(NONE).unwrap();
    // Before the first document: the skip reader's initial, unread level.
    let imp = ie.impacts().unwrap();
    assert!(matches!(imp, Impacts::Skip(_)));
    assert_eq!(imp.doc_id_up_to(0), Some(0));
    assert_eq!(ie.postings().advance(1).unwrap(), 1);
    assert_eq!(ie.impacts().unwrap().doc_id_up_to(0), Some(7));
}

#[test]
fn empty_field_is_the_empty_enum() {
    let r = open(file("field g\n")).unwrap();
    let g = r.terms("g").unwrap().unwrap();
    assert_eq!((g.size(), g.doc_count(), g.sum_doc_freq()), (0, 0, 0));
    let mut te = g.iterator();
    assert_eq!(te.next().unwrap(), None);
    assert!(!te.seek_exact(b"x").unwrap());
    assert_eq!(te.seek_ceil(b"x").unwrap(), SeekStatus::End);
    assert_eq!(te.term(), b"");
    // `TermsEnum.EMPTY.postings`/`impacts` throw `IllegalStateException`.
    assert!(matches!(te.postings(ALL), Err(Error::IllegalState(_))));
    assert!(matches!(te.impacts(ALL), Err(Error::IllegalState(_))));
}

#[test]
fn field_names_in_utf16_order() {
    // UTF-8 puts U+FFFD (EF ..) before U+10000 (F0 ..); UTF-16, like Java's
    // `TreeMap<String, ...>`, puts U+10000's surrogate D800 first.
    let r = open(file("field \u{10000}\nfield \u{FFFD}\nfield a\n")).unwrap();
    assert_eq!(
        r.field_names().collect::<Vec<_>>(),
        ["a", "\u{10000}", "\u{FFFD}"]
    );
}

#[test]
fn reader_refuses_damaged_files() {
    // No END, and no checksum line.
    assert!(matches!(
        open(b"field g\n  term x\n".to_vec()),
        Err(Error::Store(lucene_store::Error::Eof { .. }))
    ));
    let terms_err = |body: &str, field: &str| -> String {
        let r = open(file(body)).unwrap();
        err(r.terms(field))
    };
    // A field `FieldInfos` lacks.
    assert!(terms_err("field q\n", "q").contains("not in FieldInfos"));
    // A doc outside maxDoc, a negative one, one that is not a number.
    assert!(terms_err("field g\n  term x\n    doc 1000\n", "g").contains("out of bounds"));
    assert!(terms_err("field g\n  term x\n    doc -1\n", "g").contains("out of bounds"));
    assert!(terms_err("field g\n  term x\n    doc 1x\n", "g").contains("For input string"));
    // freq 0 and less: totalTermFreq goes negative.
    let neg = "field h\n  term x\n    doc 1\n      freq -1\n";
    let fis = infos();
    let r = SimpleTextFieldsReader::open(file(neg), &read_state(&fis, MAX_DOC)).unwrap();
    assert!(err(r.terms("h")).contains("negative totalTermFreq"));
    let neg2 = "field h\n  term x\n    doc 1\n      freq -1\n  term y\n    doc 1\n";
    let r = SimpleTextFieldsReader::open(file(neg2), &read_state(&fis, MAX_DOC)).unwrap();
    assert!(err(r.terms("h")).contains("negative totalTermFreq"));
    // A freq that is not a number.
    assert!(terms_err("field h\n  term x\n    doc 1\n      freq z\n", "h").contains("For input"));
    // Terms out of order: the FST refuses them.
    assert!(matches!(
        open(file("field g\n  term b\n    doc 1\n  term a\n    doc 2\n"))
            .unwrap()
            .terms("g"),
        Err(Error::Compile(_))
    ));
}

#[test]
fn postings_refuse_damaged_lines() {
    // `pos` expected, a `doc` line found (Java asserts).
    let body = "field h\n  term x\n    doc 1\n      freq 2\n      pos 4\n    doc 2\n      freq 1\n      pos 1\n";
    let r = open(file(body)).unwrap();
    let h = r.terms("h").unwrap().unwrap();
    let mut te = h.iterator();
    assert!(te.seek_exact(b"x").unwrap());
    let mut pe = te.postings(POSITIONS).unwrap();
    assert_eq!(pe.next_doc().unwrap(), 1);
    assert_eq!(pe.next_position().unwrap(), 4);
    assert!(err(pe.next_position()).contains("expected pos"));
}

// --- skip reader ----------------------------------------------------------

/// `sample()` with `edit` applied to its body and a fresh checksum; the
/// error `a`'s postings or impacts then fail with.
fn skip_err(edit: impl FnOnce(&mut Vec<u8>)) -> String {
    let bytes = refooter(&sample(), edit);
    let r = match open(bytes) {
        Ok(r) => r,
        Err(e) => return e.to_string(),
    };
    let f = match r.terms("f") {
        Ok(f) => f.unwrap(),
        Err(e) => return e.to_string(),
    };
    let mut te = f.iterator();
    assert!(te.seek_exact(b"a").unwrap());
    let mut ie = match te.impacts(FREQS) {
        Ok(ie) => ie,
        Err(e) => return e.to_string(),
    };
    let mut target = 1;
    loop {
        if let Err(e) = ie.advance_shallow(target) {
            return e.to_string();
        }
        match ie.postings().advance(target) {
            Err(e) => return e.to_string(),
            Ok(NO_MORE_DOCS) => panic!("no error"),
            Ok(_) => target += 7,
        }
    }
}

#[test]
fn skip_reader_refuses_damaged_lists() {
    // A level length that does not parse, that runs past the file, and a
    // line too short to hold its prefix.
    assert!(skip_err(|b| replace(b, b"levelLength ", b"levelLength x")).contains("For input"));
    assert!(
        skip_err(|b| replace(b, b"levelLength ", b"levelLength 9999999")).contains("level length")
    );
    assert!(
        skip_err(|b| replace(b, b"      levelLength ", b"l")).contains("shorter than its prefix")
    );
    // A child pointer past the file.
    assert!(
        skip_err(|b| replace(b, b"childPointer ", b"childPointer 9999999"))
            .contains("child pointer")
    );
    // Every skipDocFP past the file.
    assert!(skip_err(|b| replace_all(b, b"skipDocFP ", b"skipDocFP 9999")).contains("skipDocFP"));
    // A norm that does not parse.
    assert!(skip_err(|b| replace(b, b"norm ", b"norm x")).contains("For input"));
}

#[test]
fn skip_entry_running_into_end_checks_the_footer() {
    // `g`'s `x` is the last term: drop its one entry's `impactsEnd`, so
    // reading the entry runs into END and checks a footer whose checksum
    // covers the whole file, not the entry.
    let bytes = refooter(&sample(), |b| {
        let at = b.windows(10).rposition(|w| w == b"impactsEnd").unwrap();
        let start = b[..at].iter().rposition(|&c| c == b'\n').unwrap() + 1;
        let end = at + b[at..].iter().position(|&c| c == b'\n').unwrap() + 1;
        b.drain(start..end);
    });
    let r = open(bytes).unwrap();
    let mut te = r.terms("g").unwrap().unwrap().iterator();
    assert!(te.seek_exact(b"x").unwrap());
    let mut pe = te.postings(NONE).unwrap();
    assert!(err(pe.advance(3)).contains("checksum failure"));
}

#[test]
fn skip_reader_without_a_list() {
    let data = sample();
    let mut r = SimpleTextSkipReader::new(&data);
    r.reset(0, 3).unwrap();
    assert!(!r.has_skip_list());
    assert_eq!(r.next_skip_doc(), NO_MORE_DOCS);
    assert_eq!(r.next_skip_doc_fp(), -1);
    r.skip_to(5).unwrap();
    assert!(err(r.reset(i64::MAX, 9)).contains("skip pointer"));
    assert!(err(r.reset(data.len() as i64 + 1, 9)).contains("skip pointer"));
    assert_eq!(skip::BLOCK_SIZE, 8);
}
