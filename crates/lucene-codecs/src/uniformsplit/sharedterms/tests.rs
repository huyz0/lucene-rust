//! Unit tests of the shared-terms dictionary: fields with overlapping terms
//! written into one group and read back through each field's enum, the
//! intersect and the merging reader; the line serializer's boundaries; the
//! union metadata; the writer's and reader's refusals. Format fidelity is
//! `tests/codecs_sharedterms_fixtures.rs`' job, against Java's bytes.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_store::data_output::DataOutput;
use lucene_util::automaton::{operations, CompiledAutomaton, RegExp};

use super::super::field_metadata::FieldMetadata;
use super::super::{BlockHeader, SeekStatus, UniformSplitOutput};
use super::*;
use crate::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use crate::postings::{DocInput, PostingsFlags};
use crate::postings_writer::TermPostings;

const ID: [u8; 16] = *b"sharedtermstests";
const SUFFIX: &str = "SharedTermsUniformSplit_0";

/// `a` (docs), `b` (freqs), `c` (docs), numbered 0, 1, 2.
fn infos() -> FieldInfos {
    FieldInfos::new(vec![
        FieldInfo::new("a", 0).with_index_options(IndexOptions::Docs),
        FieldInfo::new("b", 1).with_index_options(IndexOptions::DocsAndFreqs),
        FieldInfo::new("c", 2).with_index_options(IndexOptions::Docs),
    ])
    .unwrap()
}

fn tp(term: &str, docs: &[i32]) -> TermPostings {
    TermPostings {
        term: term.as_bytes().to_vec(),
        docs: docs.iter().map(|&d| (d, 1 + d % 3)).collect(),
        ..TermPostings::default()
    }
}

fn write_state(fis: &FieldInfos) -> SegmentWriteState<'_> {
    SegmentWriteState {
        max_doc: 10,
        segment_id: &ID,
        segment_suffix: SUFFIX,
        field_infos: fis,
    }
}

fn read_state(fis: &FieldInfos) -> SegmentReadState<'_> {
    SegmentReadState {
        max_doc: 10,
        segment_id: &ID,
        segment_suffix: SUFFIX,
        field_infos: fis,
    }
}

fn input(number: i32, terms: &[TermPostings]) -> FieldPostingsInput<'_> {
    FieldPostingsInput {
        field_number: number,
        index_options: if number == 1 {
            IndexOptions::DocsAndFreqs
        } else {
            IndexOptions::Docs
        },
        // The distinct documents of the field.
        doc_count: terms
            .iter()
            .flat_map(|t| t.docs.iter().map(|d| d.0))
            .collect::<std::collections::BTreeSet<_>>()
            .len() as i32,
        has_payloads: false,
        terms,
    }
}

/// `a`: x y; `b`: y z; `c`: w -- the lines w (c), x (a), y (a, b), z (b).
fn fields() -> [Vec<TermPostings>; 3] {
    [
        vec![tp("x", &[1, 2]), tp("y", &[3])],
        vec![tp("y", &[4, 5, 6]), tp("z", &[7])],
        vec![tp("w", &[0, 8])],
    ]
}

fn write(format: &StUniformSplitPostingsFormat, fis: &FieldInfos) -> UniformSplitOutput {
    let [a, b, c] = fields();
    let inputs = [input(0, &a), input(1, &b), input(2, &c)];
    format.write(&inputs, &[], &write_state(fis)).unwrap()
}

fn terms_of(e: &mut StBlockReader<'_>) -> Vec<String> {
    let mut out = Vec::new();
    while let Some(t) = e.next().unwrap() {
        out.push(String::from_utf8(t.to_vec()).unwrap());
    }
    out
}

#[test]
fn each_field_reads_its_own_lines() {
    let fis = infos();
    let format = StUniformSplitPostingsFormat::default();
    assert_eq!(format.settings().target_num_block_lines(), 32);
    let out = write(&format, &fis);
    let reader = format.open(out.ustb, &out.ustd, &read_state(&fis)).unwrap();
    reader.check_integrity().unwrap();
    assert_eq!(reader.size(), 3);
    assert_eq!(reader.version(), VERSION_CURRENT);
    assert_eq!(reader.field_names().collect::<Vec<_>>(), ["a", "b", "c"]);
    assert!(format!("{reader:?}").contains("\"b\""));
    let doc_in = DocInput::open(&out.postings.doc, &ID, SUFFIX).unwrap();

    let a = reader.terms("a").unwrap();
    assert!(format!("{a:?}").contains("\"a\""));
    assert_eq!(terms_of(&mut a.iter()), ["x", "y"]);
    assert_eq!(a.min().unwrap().as_deref(), Some(&b"x"[..]));
    assert_eq!(a.max(), b"y");
    assert_eq!((a.size(), a.sum_doc_freq(), a.doc_count()), (2, 3, 3));
    assert_eq!(a.sum_total_term_freq(), 3);
    assert!(!a.has_freqs() && !a.has_positions() && !a.has_offsets() && !a.has_payloads());
    // The union spans every field: from c's first term's block to z.
    assert_eq!(a.union_metadata().last_term(), b"z");
    assert_eq!(a.union_metadata().field_info().number, -1);

    let b = reader.terms("b").unwrap();
    assert_eq!(terms_of(&mut b.iter()), ["y", "z"]);
    assert!(b.has_freqs());
    assert_eq!(b.sum_total_term_freq(), 1 + 2 + 3 + 2);
    let mut e = b.iter();
    assert!(e.seek_exact(b"y").unwrap());
    assert_eq!(e.doc_freq().unwrap(), 3);
    assert_eq!(e.total_term_freq().unwrap(), 6);
    let state = e.term_state().unwrap();
    assert_eq!(
        e.postings(&doc_in, PostingsFlags::Freqs).unwrap().docs,
        [4, 5, 6]
    );
    assert_eq!(
        b.postings(&state, &doc_in, PostingsFlags::Freqs)
            .unwrap()
            .freqs,
        [2, 3, 1]
    );
    // y is a's too, with its own state on the same line.
    let mut e = a.iter();
    assert!(e.seek_exact(b"y").unwrap());
    assert_eq!(
        e.postings(&doc_in, PostingsFlags::DocsOnly).unwrap().docs,
        [3]
    );

    // x is a line of a only: b does not have it, and an enum left on it
    // has no state for b.
    let mut e = b.iter();
    assert!(!e.seek_exact(b"x").unwrap());
    assert!(matches!(e.doc_freq(), Err(Error::Unsupported(_))));
    assert!(!e.seek_exact(b"q").unwrap());
    // seekCeil onto another field's line moves on to the field's next.
    assert_eq!(e.seek_ceil(b"w").unwrap(), SeekStatus::NotFound);
    assert_eq!(e.term(), Some(&b"y"[..]));
    assert_eq!(e.seek_ceil(b"y").unwrap(), SeekStatus::Found);
    assert_eq!(e.seek_ceil(b"zz").unwrap(), SeekStatus::End);
    let mut e = a.iter();
    assert_eq!(e.seek_ceil(b"").unwrap(), SeekStatus::NotFound);
    assert_eq!(e.term(), Some(&b"x"[..]));
    // Past a's last term, z is b's: a's enum ends there.
    assert_eq!(e.seek_ceil(b"yy").unwrap(), SeekStatus::End);
    let mut e = a.iter();
    assert!(e.seek_exact(b"y").unwrap());
    assert_eq!(e.next().unwrap(), None);
    // seekExact(term, state), then next.
    let mut e = b.iter();
    e.seek_exact_state(b"y", &state);
    assert_eq!(e.term(), Some(&b"y"[..]));
    assert_eq!(e.next().unwrap(), Some(&b"z"[..]));

    let c = reader.terms("c").unwrap();
    assert_eq!(terms_of(&mut c.iter()), ["w"]);
}

fn compile(pattern: &str) -> CompiledAutomaton {
    let a = RegExp::with_flags(pattern, RegExp::ALL, 0)
        .unwrap()
        .to_automaton()
        .unwrap();
    let a = operations::determinize(&a, operations::DEFAULT_DETERMINIZE_WORK_LIMIT).unwrap();
    CompiledAutomaton::with_options(&a, false, true, false).unwrap()
}

#[test]
fn intersect_skips_other_fields_lines() {
    let fis = infos();
    let format = StUniformSplitPostingsFormat::new(4, 1, None, None, false).unwrap();
    let out = write(&format, &fis);
    let reader = format.open(out.ustb, &out.ustd, &read_state(&fis)).unwrap();
    let doc_in = DocInput::open(&out.postings.doc, &ID, SUFFIX).unwrap();
    let compiled = compile("[w-z]");
    let b = reader.terms("b").unwrap();
    let mut e = b.intersect(&compiled, None).unwrap();
    assert!(format!("{e:?}").contains("IntersectBlockReader"));
    assert_eq!(e.next().unwrap(), Some(&b"y"[..]));
    assert_eq!(e.term(), Some(&b"y"[..]));
    assert_eq!(
        (e.doc_freq().unwrap(), e.total_term_freq().unwrap()),
        (3, 6)
    );
    assert_eq!(e.term_state().unwrap().doc_freq, 3);
    assert_eq!(
        e.postings(&doc_in, PostingsFlags::DocsOnly).unwrap().docs,
        [4, 5, 6]
    );
    assert_eq!(e.next().unwrap(), Some(&b"z"[..]));
    assert_eq!(e.next().unwrap(), None);
    assert!(e.seek_ceil(b"y").is_err());
    let c = reader.terms("c").unwrap();
    let mut e = c.intersect(&compiled, Some(b"w")).unwrap();
    assert_eq!(e.next().unwrap(), None);
}

#[test]
fn merging_reader_walks_every_line() {
    let fis = infos();
    let format = StUniformSplitPostingsFormat::default();
    let out = write(&format, &fis);
    let reader = format.open(out.ustb, &out.ustd, &read_state(&fis)).unwrap();
    let doc_in = DocInput::open(&out.postings.doc, &ID, SUFFIX).unwrap();
    let mut merging = reader.terms("c").unwrap().merging_block_reader();
    assert!(format!("{merging:?}").contains("BlockReader"));
    let mut states = Vec::new();
    let mut lines = Vec::new();
    while let Some(term) = merging.next().unwrap() {
        let term = String::from_utf8(term.to_vec()).unwrap();
        assert_eq!(merging.term(), Some(term.as_bytes()));
        merging.read_field_term_states(&mut states).unwrap();
        let numbers: Vec<i32> = states.iter().map(|(n, _)| *n).collect();
        lines.push((term, numbers));
    }
    let want = [
        ("w", vec![2]),
        ("x", vec![0]),
        ("y", vec![0, 1]),
        ("z", vec![1]),
    ];
    let want: Vec<(String, Vec<i32>)> = want
        .iter()
        .map(|(t, n)| (t.to_string(), n.clone()))
        .collect();
    assert_eq!(lines, want);
    // The last line's state reads b's postings.
    let (number, state) = states[0];
    assert_eq!(number, 1);
    let p = merging
        .postings(number, &state, &doc_in, PostingsFlags::DocsOnly)
        .unwrap();
    assert_eq!(p.docs, [7]);
    assert!(merging
        .postings(9, &state, &doc_in, PostingsFlags::DocsOnly)
        .is_err());
    assert!(merging.seek_ceil(b"x").is_err());
    assert!(merging.seek_exact(b"x").is_err());
}

#[test]
fn a_writer_without_terms_writes_an_empty_group() {
    let fis = infos();
    let state = write_state(&fis);
    let writer = StUniformSplitTermsWriter::new(&state, 32, 3, None, &[(0, 0), (1, 0)]).unwrap();
    assert!(format!("{writer:?}").contains("fields: 2"));
    let (stustb, stustd) = writer.finish().unwrap();
    let reader = StUniformSplitPostingsFormat::default()
        .open(stustb, &stustd, &read_state(&fis))
        .unwrap();
    assert_eq!(reader.size(), 0);
    assert!(reader.terms("a").is_none());
    reader.check_integrity().unwrap();
}

#[test]
fn the_writer_refuses_bad_input() {
    let fis = infos();
    let state = write_state(&fis);
    let format = StUniformSplitPostingsFormat::default();
    let [a, b, _] = fields();
    // A field number twice, or unknown.
    let err = format
        .write(&[input(0, &a), input(0, &b)], &[], &state)
        .unwrap_err();
    assert!(matches!(err, Error::IllegalArgument(_)), "{err}");
    let err = format.write(&[input(7, &a)], &[], &state).unwrap_err();
    assert!(matches!(err, Error::IllegalArgument(_)), "{err}");
    // A field without terms, and unsorted terms (named at the field's index).
    let err = format.write(&[input(0, &[])], &[], &state).unwrap_err();
    assert!(matches!(
        err,
        Error::Write(postings_writer::Error::EmptyTerms)
    ));
    let unsorted = [tp("a", &[1]), tp("c", &[1]), tp("b", &[2])];
    let err = format
        .write(&[input(0, &unsorted)], &[], &state)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Write(postings_writer::Error::TermsNotSorted(2))
    ));
    // Settings and codecs the parent format refuses.
    assert!(StUniformSplitPostingsFormat::new(0, 0, None, None, false).is_err());
    assert!(StUniformSplitTermsWriter::new(&state, 4, 4, None, &[]).is_err());
    assert!(StUniformSplitTermsWriter::new(&state, 32, 3, None, &[(5, 1)]).is_err());
}

/// Feeds `inputs` (one-term fields, in the order given) through the
/// postings writer into a writer of fields `a` and `b`.
fn feed(inputs: &[FieldPostingsInput<'_>]) -> Result<()> {
    let fis = infos();
    let state = write_state(&fis);
    let mut writer = StUniformSplitTermsWriter::new(&state, 32, 3, None, &[(0, 10), (1, 10)])?;
    postings_writer::write_postings(inputs, &[], None, None, &ID, SUFFIX, |f| {
        writer.add_term_states(f)
    })?;
    writer.finish().map(|_| ())
}

#[test]
fn term_states_must_come_by_term_then_field() {
    let (x, y) = ([tp("x", &[1])], [tp("y", &[1])]);
    assert!(feed(&[input(0, &x), input(1, &x), input(0, &y)]).is_ok());
    // A term before the line's.
    let err = feed(&[input(0, &y), input(1, &x)]).unwrap_err();
    assert!(matches!(err, Error::IllegalArgument(_)), "{err}");
    // The same term for a field numbered lower, or the same field again.
    assert!(feed(&[input(1, &x), input(0, &x)]).is_err());
    assert!(feed(&[input(0, &x), input(0, &x)]).is_err());
    // A field the writer was not given.
    assert!(feed(&[input(2, &x)]).is_err());
}

#[test]
fn the_union_spans_every_field() {
    let mut builder = UnionFieldMetadataBuilder::default();
    let err = builder.build().unwrap_err();
    assert!(matches!(err, Error::IllegalArgument(_)));
    let field = |number: i32, first: i64, last: i64, term: &str| {
        let mut m = FieldMetadata::new(FieldInfo::new(format!("f{number}"), number), 1);
        m.dictionary_start_fp = 40;
        m.first_block_start_fp = first;
        m.last_block_start_fp = last;
        m.last_term = term.as_bytes().to_vec();
        m
    };
    builder.add_field_metadata(&field(0, 70, 90, "m")).unwrap();
    builder.add_field_metadata(&field(1, 60, 80, "z")).unwrap();
    builder.add_field_metadata(&field(2, 75, 85, "k")).unwrap();
    let union = builder.build().unwrap();
    assert_eq!(union.dictionary_start_fp(), 40);
    assert_eq!(union.first_block_start_fp(), 60);
    assert_eq!(union.last_block_start_fp(), 90);
    assert_eq!(union.last_term(), b"z");
    assert_eq!(union.num_terms(), 0);
    // Fields over different dictionaries.
    let mut other = field(3, 1, 2, "a");
    other.dictionary_start_fp = 41;
    assert!(builder.add_field_metadata(&other).is_err());
}

fn header() -> BlockHeader {
    BlockHeader::new(1, 0, 0, 0, 0, 0)
}

#[test]
fn line_term_states_refuse_unknown_fields_and_truncation() {
    let fis = FieldInfos::new(vec![
        FieldInfo::new("a", 0).with_index_options(IndexOptions::Docs),
        FieldInfo::new("b", 3).with_index_options(IndexOptions::Docs),
    ])
    .unwrap();
    let shared = SharedFieldInfos::new(&fis);
    let read =
        |buf: &[u8], field: i32| shared.read_term_state_for_field(field, buf, &mut 0, &header());
    let all = |buf: &[u8]| {
        let mut out = Vec::new();
        shared
            .read_field_term_states(buf, &mut 0, &header(), &mut out)
            .map(|()| out)
    };
    // One field, number 3: docFreq 1, singleton doc 5.
    let mut one = Vec::new();
    one.write_zint(-3);
    one.write_vint(1);
    one.write_vint(5);
    let state = read(&one, 3).unwrap().unwrap();
    assert_eq!((state.doc_freq, state.meta.singleton_doc_id), (1, 5));
    assert_eq!(read(&one, 0).unwrap(), None);
    assert_eq!(all(&one).unwrap(), [(3, state)]);
    // Field 7 is unknown to the segment.
    let mut unknown = Vec::new();
    unknown.write_zint(-7);
    assert!(read(&unknown, 7).is_err());
    assert!(all(&unknown).is_err());
    // `i32::MIN` negates to itself: no field's.
    let mut min = Vec::new();
    min.write_zint(i32::MIN);
    assert_eq!(read(&min, 0).unwrap(), None);
    // Two fields, 0 then 3, each a singleton.
    let mut two = Vec::new();
    two.write_zint(2);
    two.write_vint(0);
    two.write_vint(3);
    for doc in [8, 9] {
        two.write_vint(1);
        two.write_vint(doc);
    }
    assert_eq!(read(&two, 3).unwrap().unwrap().meta.singleton_doc_id, 9);
    assert_eq!(read(&two, 0).unwrap().unwrap().meta.singleton_doc_id, 8);
    // 1 is not listed (0 < 1 < 3), nor is 9 (after both).
    assert_eq!(read(&two, 1).unwrap(), None);
    assert_eq!(read(&two, 9).unwrap(), None);
    let states = all(&two).unwrap();
    assert_eq!(states.iter().map(|(n, _)| *n).collect::<Vec<_>>(), [0, 3]);
    // A listed field the segment lacks, before the one asked for.
    let mut listed = Vec::new();
    listed.write_zint(2);
    listed.write_vint(7);
    listed.write_vint(9);
    assert!(read(&listed, 9).is_err());
    assert!(all(&listed).is_err());
    // A count past the line's end.
    let mut truncated = Vec::new();
    truncated.write_zint(1_000_000);
    truncated.write_vint(0);
    assert!(read(&truncated, 3).is_err());
    assert!(all(&truncated).is_err());
    assert!(read(&[], 0).is_err());
}
