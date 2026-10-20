//! Unit tests of the block terms dictionary: the writer's blocks read back
//! over both terms indexes and three index settings (every seek shape
//! against a sorted scan, ascending seeks that stay in a block, ords,
//! forced term states, postings), the index writers' and selectors' own
//! rules, and the readers' refusals of bytes Java would refuse or misread.
//! Format fidelity is `tests/codecs_blockterms_fixtures.rs`' job, against
//! Java's bytes.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_store::codec_util::{self, FOOTER_LENGTH};
use lucene_store::data_output::DataOutput;

use super::*;
use crate::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use crate::postings::{DocInput, PostingsFlags};
use crate::postings_writer::{FieldPostingsInput, TermPostings};
use crate::segment_state::{SegmentReadState, SegmentWriteState};

const ID: [u8; 16] = *b"blocktermsidssss";
const SUFFIX: &str = "BlockTerms_0";

/// The terms index a test group is written with.
#[derive(Debug, Clone, Copy)]
enum Kind {
    Fixed(i32),
    EveryN(i32),
    DocFreq(i32, i32),
}

const KINDS: [Kind; 4] = [
    Kind::Fixed(4),
    Kind::Fixed(32),
    Kind::EveryN(3),
    Kind::DocFreq(3, 7),
];

/// A written group: the `.tib`, the index files and the postings.
struct Group {
    tib: Vec<u8>,
    tii: Vec<u8>,
    vg: VariableGapFiles,
    doc: Vec<u8>,
    fis: FieldInfos,
    max_doc: i32,
}

fn infos() -> FieldInfos {
    FieldInfos::new(vec![
        FieldInfo::new("f", 0).with_index_options(IndexOptions::Docs),
        FieldInfo::new("g", 1).with_index_options(IndexOptions::DocsAndFreqs),
    ])
    .unwrap()
}

fn write_state(fis: &FieldInfos, max_doc: i32) -> SegmentWriteState<'_> {
    SegmentWriteState {
        max_doc,
        segment_id: &ID,
        segment_suffix: SUFFIX,
        field_infos: fis,
    }
}

fn read_state(fis: &FieldInfos, max_doc: i32) -> SegmentReadState<'_> {
    SegmentReadState {
        max_doc,
        segment_id: &ID,
        segment_suffix: SUFFIX,
        field_infos: fis,
    }
}

/// Term `i` of `terms` in documents `0..=i % 3`.
fn postings_of(terms: &[Vec<u8>]) -> Vec<TermPostings> {
    terms
        .iter()
        .enumerate()
        .map(|(i, t)| TermPostings {
            term: t.clone(),
            docs: (0..=(i % 3) as i32).map(|d| (d, 1)).collect(),
            ..TermPostings::default()
        })
        .collect()
}

/// Field `f` (DOCS) holding `terms` and, when `with_g`, field `g` (freqs)
/// holding `g` (one document) and `h` (three).
fn write(kind: Kind, terms: &[Vec<u8>], with_g: bool) -> Result<Group> {
    let fis = infos();
    let max_doc = 3;
    let f = postings_of(terms);
    let g = vec![
        TermPostings {
            term: b"g".to_vec(),
            docs: vec![(0, 3)],
            ..TermPostings::default()
        },
        TermPostings {
            term: b"h".to_vec(),
            docs: vec![(0, 1), (1, 2), (2, 5)],
            ..TermPostings::default()
        },
    ];
    let mut inputs = Vec::new();
    if !terms.is_empty() {
        inputs.push(FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::Docs,
            doc_count: terms.len().min(3) as i32,
            has_payloads: false,
            terms: &f,
        });
    }
    if with_g {
        inputs.push(FieldPostingsInput {
            field_number: 1,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 3,
            has_payloads: false,
            terms: &g,
        });
    }
    let state = write_state(&fis, max_doc);
    let (tib, tii, vg, doc) = match kind {
        Kind::Fixed(interval) => {
            let out = super::write(
                FixedGapTermsIndexWriter::new(&state, interval)?,
                &inputs,
                &[],
                &state,
            )?;
            (
                out.tib,
                out.index,
                VariableGapFiles::default(),
                out.postings.doc,
            )
        }
        Kind::EveryN(n) => {
            let w = VariableGapTermsIndexWriter::new(&state, EveryNTermSelector::new(n));
            let out = super::write(w, &inputs, &[], &state)?;
            (out.tib, Vec::new(), out.index, out.postings.doc)
        }
        Kind::DocFreq(df, n) => {
            let selector = EveryNOrDocFreqTermSelector::new(df, n);
            let w = VariableGapTermsIndexWriter::new(&state, selector);
            let out = super::write(w, &inputs, &[], &state)?;
            (out.tib, Vec::new(), out.index, out.postings.doc)
        }
    };
    Ok(Group {
        tib,
        tii,
        vg,
        doc,
        fis,
        max_doc,
    })
}

fn index_of(kind: Kind, group: &Group) -> Result<TermsIndexReader> {
    let state = read_state(&group.fis, group.max_doc);
    Ok(match kind {
        Kind::Fixed(_) => FixedGapTermsIndexReader::open(&group.tii, &state)?.into(),
        _ => VariableGapTermsIndexReader::open(&group.vg.tmv, &group.vg.tiv, &state)?.into(),
    })
}

fn open(kind: Kind, group: &Group) -> Result<BlockTermsReader> {
    let index = index_of(kind, group)?;
    BlockTermsReader::open(
        group.tib.clone(),
        index,
        &read_state(&group.fis, group.max_doc),
    )
}

/// The empty term, runs sharing long prefixes, terms that extend each
/// other, and a few high bytes -- sorted, distinct.
fn corpus() -> Vec<Vec<u8>> {
    let mut terms: Vec<Vec<u8>> = vec![Vec::new()];
    for i in 0..300 {
        terms.push(format!("a{:x}", i * 7919 % 4096).into_bytes());
    }
    for i in 0..60 {
        terms.push(format!("bcdefghijklmnop{i:03}x").into_bytes());
    }
    for n in 1..30 {
        terms.push(vec![b'c'; n]);
    }
    terms.push(vec![b'd']);
    terms.push(vec![0xff, 0x00, 0x7f]);
    terms.push(vec![0xff; 300]);
    terms.sort();
    terms.dedup();
    terms
}

/// Targets around every term: itself, one byte more, one byte less, a
/// prefix, and a few that sort before or after everything.
fn targets(terms: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut out = vec![
        Vec::new(),
        vec![0],
        b"0".to_vec(),
        vec![0xff; 301],
        b"zz".to_vec(),
    ];
    for t in terms {
        out.push(t.clone());
        let mut more = t.clone();
        more.push(0);
        out.push(more);
        let mut less = t.clone();
        if let Some(last) = less.last_mut() {
            if *last > 0 {
                *last -= 1;
                less.push(0xff);
                out.push(less);
            }
        }
        if t.len() > 1 {
            out.push(t[..t.len() / 2].to_vec());
        }
    }
    out
}

/// What `seekCeil(target)` must answer: the status and the term landed on.
fn expected_ceil(terms: &[Vec<u8>], target: &[u8]) -> (SeekStatus, Option<Vec<u8>>) {
    match terms.binary_search_by(|t| t.as_slice().cmp(target)) {
        Ok(i) => (SeekStatus::Found, Some(terms[i].clone())),
        Err(i) if i == terms.len() => (SeekStatus::End, None),
        Err(i) => (SeekStatus::NotFound, Some(terms[i].clone())),
    }
}

#[test]
fn every_seek_lands_where_a_sorted_scan_does() {
    let terms = corpus();
    for kind in KINDS {
        let group = write(kind, &terms, true).unwrap();
        let reader = open(kind, &group).unwrap();
        let fr = reader.terms("f").unwrap();
        assert_eq!(fr.size(), terms.len() as i64);
        for target in targets(&terms) {
            // A fresh enum, and one reused from the seek before.
            let mut fresh = fr.iterator();
            let (status, term) = expected_ceil(&terms, &target);
            assert_eq!(
                fresh.seek_ceil(&target).unwrap(),
                status,
                "{kind:?} {target:?}"
            );
            if let Some(term) = &term {
                assert_eq!(fresh.term(), term.as_slice());
                let i = terms.binary_search(term).unwrap();
                assert_eq!(fresh.doc_freq().unwrap(), (i % 3) as i32 + 1);
                assert_eq!(fresh.total_term_freq().unwrap(), (i % 3) as i64 + 1);
                if let Kind::Fixed(_) = kind {
                    assert_eq!(fresh.ord().unwrap(), i as i64);
                }
                // The enum goes on from there.
                let next = fresh.next().unwrap().map(<[u8]>::to_vec);
                assert_eq!(next.as_ref(), terms.get(i + 1));
            } else {
                assert_eq!(fresh.next().unwrap(), None);
            }
            assert_eq!(
                fr.iterator().seek_exact(&target).unwrap(),
                status == SeekStatus::Found
            );
        }
    }
}

#[test]
fn ascending_seeks_reuse_the_current_block() {
    let terms = corpus();
    for kind in KINDS {
        let group = write(kind, &terms, false).unwrap();
        let reader = open(kind, &group).unwrap();
        let fr = reader.terms("f").unwrap();
        let mut e = fr.iterator();
        for target in {
            let mut t = targets(&terms);
            t.sort();
            t
        } {
            let (status, term) = expected_ceil(&terms, &target);
            assert_eq!(e.seek_ceil(&target).unwrap(), status, "{kind:?} {target:?}");
            if let Some(term) = term {
                assert_eq!(e.term(), term.as_slice());
                if let Kind::Fixed(_) = kind {
                    let i = terms.binary_search(&term).unwrap();
                    assert_eq!(e.ord().unwrap(), i as i64);
                }
                // The same target again is answered from the cache.
                if status == SeekStatus::Found {
                    assert_eq!(e.seek_ceil(&target).unwrap(), SeekStatus::Found);
                }
            }
        }
        // Every term, one after the other, then a step past the last.
        let mut e = fr.iterator();
        for t in &terms {
            assert_eq!(e.seek_ceil(t).unwrap(), SeekStatus::Found);
        }
        assert_eq!(e.seek_ceil(&[0xff; 301]).unwrap(), SeekStatus::End);
        assert_eq!(e.seek_ceil(&[0xff; 302]).unwrap(), SeekStatus::End);
    }
}

#[test]
fn next_walks_every_term_and_stays_at_the_end() {
    let terms = corpus();
    for kind in KINDS {
        let group = write(kind, &terms, true).unwrap();
        let reader = open(kind, &group).unwrap();
        let fr = reader.terms("f").unwrap();
        let mut e = fr.iterator();
        let mut seen = Vec::new();
        while let Some(t) = e.next().unwrap() {
            seen.push(t.to_vec());
            let i = seen.len() - 1;
            if i % 5 == 0 {
                assert_eq!(e.doc_freq().unwrap(), (i % 3) as i32 + 1);
            }
        }
        assert_eq!(seen, terms);
        assert_eq!(e.next().unwrap(), None);
        assert_eq!(e.next().unwrap(), None);
    }
}

#[test]
fn ords_seek_every_term_over_the_fixed_gap_index() {
    let terms = corpus();
    for interval in [1, 4, 32] {
        let kind = Kind::Fixed(interval);
        let group = write(kind, &terms, false).unwrap();
        let reader = open(kind, &group).unwrap();
        let fr = reader.terms("f").unwrap();
        let mut e = fr.iterator();
        for (ord, term) in terms.iter().enumerate().rev() {
            e.seek_exact_ord(ord as i64).unwrap();
            assert_eq!(e.term(), term.as_slice());
            assert_eq!(e.ord().unwrap(), ord as i64);
            assert_eq!(e.doc_freq().unwrap(), (ord % 3) as i32 + 1);
            let next = e.next().unwrap().map(<[u8]>::to_vec);
            assert_eq!(next.as_ref(), terms.get(ord + 1));
        }
        assert!(matches!(
            e.seek_exact_ord(terms.len() as i64),
            Err(Error::IllegalArgument(_))
        ));
        assert!(matches!(
            e.seek_exact_ord(-1),
            Err(Error::IllegalArgument(_))
        ));
    }
}

#[test]
fn the_variable_gap_index_has_no_ords() {
    let terms = corpus();
    let kind = Kind::EveryN(3);
    let group = write(kind, &terms, false).unwrap();
    let reader = open(kind, &group).unwrap();
    let mut e = reader.terms("f").unwrap().iterator();
    assert!(matches!(e.ord(), Err(Error::Unsupported(_))));
    assert!(matches!(e.seek_exact_ord(0), Err(Error::Unsupported(_))));
}

#[test]
fn a_term_state_seeks_without_reading_until_next() {
    let terms = corpus();
    for kind in KINDS {
        let group = write(kind, &terms, false).unwrap();
        let reader = open(kind, &group).unwrap();
        let fr = reader.terms("f").unwrap();
        let mut states = Vec::new();
        let mut e = fr.iterator();
        while e.next().unwrap().is_some() {
            states.push((e.term().to_vec(), e.term_state().unwrap()));
        }
        for (i, (term, state)) in states.iter().enumerate().step_by(7) {
            let mut e = fr.iterator();
            e.seek_exact_state(term, state);
            assert_eq!(e.term(), term.as_slice());
            // The state is the one handed in, until a `next()`.
            assert_eq!(e.doc_freq().unwrap(), state.doc_freq);
            assert_eq!(e.term_state().unwrap(), *state);
            let next = e.next().unwrap().map(<[u8]>::to_vec);
            assert_eq!(next.as_ref(), terms.get(i + 1));
            if i + 1 < terms.len() {
                if let Kind::Fixed(_) = kind {
                    assert_eq!(e.ord().unwrap(), i as i64 + 1);
                }
                assert_eq!(e.doc_freq().unwrap(), ((i + 1) % 3) as i32 + 1);
            }
        }
    }
}

#[test]
fn postings_decode_from_the_term_metadata() {
    for kind in KINDS {
        let group = write(kind, &[b"x".to_vec(), b"y".to_vec()], true).unwrap();
        let reader = open(kind, &group).unwrap();
        let doc_in = DocInput::open(&group.doc, &ID, SUFFIX).unwrap();
        let fr = reader.terms("g").unwrap();
        assert!(fr.has_freqs());
        assert!(!fr.has_positions() && !fr.has_offsets() && !fr.has_payloads());
        assert_eq!(fr.info().name, "g");
        assert_eq!(
            (fr.sum_doc_freq(), fr.sum_total_term_freq(), fr.doc_count()),
            (4, 11, 3)
        );
        let mut e = fr.iterator();
        assert!(matches!(
            e.postings(&doc_in, PostingsFlags::FreqsNoImpacts),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(e.seek_ceil(b"g").unwrap(), SeekStatus::Found);
        let p = e.postings(&doc_in, PostingsFlags::FreqsNoImpacts).unwrap();
        assert_eq!((p.docs, p.freqs), (vec![0], vec![3]));
        assert_eq!(e.seek_ceil(b"h").unwrap(), SeekStatus::Found);
        assert_eq!(e.total_term_freq().unwrap(), 8);
        let p = e.postings(&doc_in, PostingsFlags::FreqsNoImpacts).unwrap();
        assert_eq!((p.docs, p.freqs), (vec![0, 1, 2], vec![1, 2, 5]));

        let f = reader.terms("f").unwrap();
        assert!(!f.has_freqs());
        assert_eq!((f.sum_doc_freq(), f.sum_total_term_freq()), (3, 3));
        assert_eq!(reader.size(), 2);
        assert_eq!(reader.field_names().collect::<Vec<_>>(), ["f", "g"]);
        assert!(reader.terms("nope").is_none());
        reader.check_integrity().unwrap();
        assert!(reader.to_string().starts_with("BlockTermsReader(index="));
    }
}

#[test]
fn a_field_without_index_terms_only_iterates() {
    // The `.tib` holds `f` and `g`; the index was written for `g` alone.
    for kind in KINDS {
        let both = write(kind, &corpus(), true).unwrap();
        let only_g = write(kind, &[], true).unwrap();
        let index = index_of(kind, &only_g).unwrap();
        let reader =
            BlockTermsReader::open(both.tib.clone(), index, &read_state(&both.fis, 3)).unwrap();
        let mut e = reader.terms("f").unwrap().iterator();
        assert!(matches!(
            e.seek_ceil(b"a"),
            Err(Error::IllegalState("terms index was not loaded"))
        ));
        assert!(matches!(e.seek_exact_ord(0), Err(Error::IllegalState(_))));
        assert_eq!(e.next().unwrap(), Some(&b""[..]));
    }
}

#[test]
fn the_index_readers_list_their_fields() {
    let group = write(Kind::Fixed(4), &corpus(), true).unwrap();
    let state = read_state(&group.fis, 3);
    let fixed = FixedGapTermsIndexReader::open(&group.tii, &state).unwrap();
    assert_eq!(fixed.field_names().collect::<Vec<_>>(), ["f", "g"]);
    assert_eq!(fixed.index_interval(), 4);
    assert!(fixed.field_enum("nope").is_none());
    let mut e = fixed.field_enum("f").unwrap();
    assert!(matches!(e.seek_ord(-1), Err(Error::IllegalArgument(_))));
    assert_eq!(e.seek_ord(5).unwrap(), e.seek(e.term()).unwrap());
    assert_eq!(e.ord(), 4);

    let group = write(Kind::EveryN(3), &corpus(), true).unwrap();
    let vg = VariableGapTermsIndexReader::open(&group.vg.tmv, &group.vg.tiv, &state).unwrap();
    assert_eq!(vg.field_names().collect::<Vec<_>>(), ["f", "g"]);
    assert!(!vg.supports_ord());
    let mut e = vg.field_enum("g").unwrap();
    assert_eq!(e.term(), None);
    assert!(e.next().unwrap().is_some());
    assert_eq!(e.term(), Some(&b""[..]));
    while e.next().unwrap().is_some() {}
    assert_eq!(e.term(), None);
}

#[test]
fn selectors_pick_every_nth_term_or_a_frequent_one() {
    let stats = |doc_freq| TermStats {
        doc_freq,
        total_term_freq: -1,
    };
    let info = FieldInfo::new("f", 0);
    let mut every = EveryNTermSelector::new(3);
    let picks: Vec<bool> = (0..7)
        .map(|_| every.is_index_term(b"t", stats(1)))
        .collect();
    assert_eq!(picks, [true, false, false, true, false, false, true]);
    every.new_field(&info);
    assert!(every.is_index_term(b"t", stats(1)));

    let mut df = EveryNOrDocFreqTermSelector::new(5, 3);
    let picks: Vec<bool> = [1, 1, 9, 1, 1, 1]
        .iter()
        .map(|&d| df.is_index_term(b"t", stats(d)))
        .collect();
    assert_eq!(picks, [true, false, true, false, false, true]);
    assert!(!df.is_index_term(b"t", stats(1)));
    df.new_field(&info);
    assert!(df.is_index_term(b"t", stats(1)));
}

#[test]
fn index_writers_refuse_misuse() {
    let fis = infos();
    let state = write_state(&fis, 3);
    let stats = TermStats {
        doc_freq: 1,
        total_term_freq: -1,
    };
    assert!(matches!(
        FixedGapTermsIndexWriter::new(&state, 0),
        Err(Error::IllegalArgument(_))
    ));
    let mut fixed = FixedGapTermsIndexWriter::new(&state, 2).unwrap();
    assert!(!fixed.check_index_term(b"a", stats));
    assert!(matches!(
        fixed.add(b"a", stats, 0),
        Err(Error::IllegalState(_))
    ));
    assert!(matches!(fixed.finish(0), Err(Error::IllegalState(_))));
    fixed.add_field(fis.field_by_number(0).unwrap(), 0).unwrap();
    assert!(fixed.check_index_term(b"b", stats));
    fixed.add(b"b", stats, 0).unwrap();
    assert!(!fixed.check_index_term(b"c", stats));
    // The term before the index term again, and one its predecessor
    // extends.
    assert!(matches!(
        fixed.add(b"c", stats, 0),
        Err(Error::IllegalArgument(_))
    ));
    fixed.add(b"d", stats, 0).unwrap();
    assert!(matches!(
        fixed.add(b"", stats, 0),
        Err(Error::IllegalArgument(_))
    ));

    let mut vg = VariableGapTermsIndexWriter::new(&state, EveryNTermSelector::new(2));
    assert!(format!("{vg:?}").starts_with("VariableGapTermsIndexWriter"));
    assert!(!vg.check_index_term(b"a", stats));
    assert!(matches!(
        vg.add(b"a", stats, 0),
        Err(Error::IllegalState(_))
    ));
    assert!(matches!(vg.finish(0), Err(Error::IllegalState(_))));
    vg.add_field(fis.field_by_number(0).unwrap(), 0).unwrap();
    assert!(vg.check_index_term(b"b", stats));
    vg.add(b"b", stats, 0).unwrap();
    // The FST refuses an input out of order.
    assert!(vg.add(b"a", stats, 0).is_err());

    // A field number the segment lacks.
    let other = FieldInfos::new(vec![FieldInfo::new("z", 9)]).unwrap();
    let terms = postings_of(&[b"a".to_vec()]);
    let inputs = [FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::Docs,
        doc_count: 1,
        has_payloads: false,
        terms: &terms,
    }];
    let state = write_state(&other, 3);
    let w = FixedGapTermsIndexWriter::new(&state, 4).unwrap();
    assert!(super::write(w, &inputs, &[], &state).is_err());
}

#[test]
fn prefix_lengths_cut_at_the_first_difference() {
    assert_eq!(variable_gap::indexed_term_prefix_length(b"abc", b"abd"), 3);
    assert_eq!(variable_gap::indexed_term_prefix_length(b"ab", b"abcd"), 3);
    assert_eq!(variable_gap::indexed_term_prefix_length(b"", b"x"), 1);
    assert_eq!(variable_gap::indexed_term_prefix_length(b"abc", b"abc"), 3);
}

/// `bytes` with its footer rewritten over the bytes before it.
fn refooter(mut bytes: Vec<u8>) -> Vec<u8> {
    bytes.truncate(bytes.len() - FOOTER_LENGTH);
    codec_util::write_footer(&mut bytes);
    bytes
}

/// `file` with everything from its directory on replaced by `dir`, its
/// offset and a footer.
fn with_dir(file: &[u8], dir: &[u8]) -> Vec<u8> {
    let at = file.len() - FOOTER_LENGTH - 8;
    let start = i64::from_le_bytes(file[at..at + 8].try_into().unwrap()) as usize;
    let mut out = file[..start].to_vec();
    out.extend_from_slice(dir);
    out.write_i64(start as i64);
    codec_util::write_footer(&mut out);
    out
}

/// One `.tib` directory entry: `(number, numTerms, termsStart, sumTTF,
/// sumDocFreq, docCount)`, `sumTTF` only when given.
type DirEntry = (i32, i64, i64, Option<i64>, i64, i32);

/// A `.tib` directory of `count` and `fields`.
fn tib_dir(count: i32, fields: &[DirEntry]) -> Vec<u8> {
    let mut dir = Vec::new();
    dir.write_vint(count);
    for &(number, num_terms, start, ttf, df, docs) in fields {
        dir.write_vint(number);
        dir.write_vlong(num_terms);
        dir.write_vlong(start);
        if let Some(ttf) = ttf {
            dir.write_vlong(ttf);
        }
        dir.write_vlong(df);
        dir.write_vint(docs);
    }
    dir
}

fn assert_corrupt<T: std::fmt::Debug>(r: Result<T>) {
    match r {
        Err(Error::Store(_)) | Err(Error::Fst(_)) => {}
        other => panic!("expected Corrupted, got {other:?}"),
    }
}

#[test]
fn the_terms_reader_refuses_a_bad_directory() {
    let kind = Kind::Fixed(4);
    let group = write(kind, &corpus(), false).unwrap();
    let open_tib = |tib: Vec<u8>| {
        BlockTermsReader::open(
            tib,
            index_of(kind, &group).unwrap(),
            &read_state(&group.fis, 3),
        )
    };
    // The real directory, re-sealed, still opens: `f`'s blocks start right
    // after the two headers and the postings block size.
    let start = (codec_util::index_header_length(CODEC_NAME, SUFFIX)
        + codec_util::index_header_length(POSTINGS_TERMS_CODEC, SUFFIX)
        + 2) as i64;
    let good = (0, corpus().len() as i64, start, None, 1000, 3);
    open_tib(with_dir(&group.tib, &tib_dir(1, &[good]))).unwrap();
    for dir in [
        tib_dir(-1, &[]),
        tib_dir(1, &[(good.0, 0, start, None, 1000, 3)]),
        tib_dir(1, &[(7, good.1, start, None, 1000, 3)]),
        tib_dir(1, &[(good.0, good.1, start, None, 1000, 4)]),
        tib_dir(1, &[(good.0, good.1, start, None, 1000, -1)]),
        tib_dir(1, &[(good.0, good.1, start, None, 2, 3)]),
        tib_dir(1, &[(good.0, good.1, 1 << 40, None, 1000, 3)]),
        tib_dir(2, &[good, good]),
        tib_dir(1, &[(1, 2, start, Some(2), 3, 3)]),
        tib_dir(2, &[good]),
    ] {
        assert_corrupt(open_tib(with_dir(&group.tib, &dir)));
    }
    // A directory offset past the file, and a file too short for one.
    let mut tib = group.tib.clone();
    let at = tib.len() - FOOTER_LENGTH - 8;
    tib[at..at + 8].copy_from_slice(&(1i64 << 40).to_le_bytes());
    assert_corrupt(open_tib(refooter(tib)));
    assert_corrupt(open_tib(group.tib[..FOOTER_LENGTH].to_vec()));
    // A different postings block size.
    let mut tib = group.tib.clone();
    let header = codec_util::index_header_length(CODEC_NAME, SUFFIX)
        + codec_util::index_header_length(POSTINGS_TERMS_CODEC, SUFFIX);
    tib[header] = 64;
    assert!(matches!(open_tib(tib), Err(Error::Unsupported(_))));
    // A damaged byte is found by `checkIntegrity`, not at open.
    let mut tib = group.tib.clone();
    tib[header + 3] ^= 1;
    let reader = open_tib(tib).unwrap();
    assert_corrupt(reader.check_integrity());
    let reader = BlockTermsReader {
        tib: vec![0u8; 3].into(),
        fields: Default::default(),
    };
    assert_corrupt(reader.check_integrity());
}

/// `.tii`'s one directory entry: `(number, numIndexTerms, termsStart,
/// indexStart, packedIndexStart, packedOffsetsStart)`.
type TiiEntry = (i32, i32, i64, i64, i64, i64);

/// The `.tii`'s first directory entry.
fn tii_entry(tii: &[u8]) -> TiiEntry {
    use lucene_store::data_input::{DataInput, SliceInput};
    let at = tii.len() - FOOTER_LENGTH - 8;
    let start = i64::from_le_bytes(tii[at..at + 8].try_into().unwrap()) as usize;
    let mut input = SliceInput::new(tii);
    input.seek(start).unwrap();
    assert_eq!(input.read_vint().unwrap(), 1);
    (
        input.read_vint().unwrap(),
        input.read_vint().unwrap(),
        input.read_vlong().unwrap(),
        input.read_vlong().unwrap(),
        input.read_vlong().unwrap(),
        input.read_vlong().unwrap(),
    )
}

/// A `.tii` directory of `count` and `entries`.
fn tii_dir(count: i32, entries: &[TiiEntry]) -> Vec<u8> {
    let mut dir = Vec::new();
    dir.write_vint(count);
    for &(number, n, terms, index, packed, offsets) in entries {
        dir.write_vint(number);
        dir.write_vint(n);
        dir.write_vlong(terms);
        dir.write_vlong(index);
        dir.write_vlong(packed);
        dir.write_vlong(offsets);
    }
    dir
}

#[test]
fn the_fixed_gap_index_refuses_a_bad_directory() {
    let group = write(Kind::Fixed(4), &corpus(), false).unwrap();
    let state = read_state(&group.fis, 3);
    let open = |tii: &[u8]| FixedGapTermsIndexReader::open(tii, &state);
    let e = tii_entry(&group.tii);
    open(&with_dir(&group.tii, &tii_dir(1, &[e]))).unwrap();
    let (number, n, terms, index, packed, offsets) = e;
    for dir in [
        tii_dir(-1, &[]),
        tii_dir(1, &[(number, 0, terms, index, packed, offsets)]),
        tii_dir(1, &[(7, n, terms, index, packed, offsets)]),
        tii_dir(1, &[(number, n, terms, packed + 1, packed, offsets)]),
        // More index terms than prefix bytes.
        tii_dir(
            1,
            &[(
                number,
                (packed - index) as i32 + 2,
                terms,
                index,
                packed,
                offsets,
            )],
        ),
        // A prefix ending past the field's prefix bytes.
        tii_dir(1, &[(number, n, terms, index + 1, packed, offsets)]),
        tii_dir(2, &[e, e]),
    ] {
        assert_corrupt(open(&with_dir(&group.tii, &dir)));
    }
    // An interval of 0, re-sealed.
    let mut tii = group.tii.clone();
    tii[codec_util::index_header_length(fixed_gap::CODEC_NAME, SUFFIX)] = 0;
    assert_corrupt(open(&refooter(tii)));
}

#[test]
fn the_variable_gap_index_refuses_a_bad_directory() {
    let group = write(Kind::EveryN(3), &corpus(), false).unwrap();
    let state = read_state(&group.fis, 3);
    let (tmv, tiv) = (&group.vg.tmv, &group.vg.tiv);
    let reader = VariableGapTermsIndexReader::open(tmv, tiv, &state).unwrap();
    assert!(format!("{:?}", reader.field_enum("f").unwrap()).contains("VariableGapIndexEnum"));
    // Each file's header is checked: the meta file is not an index file.
    assert!(VariableGapTermsIndexReader::open(tmv, tmv, &state).is_err());
    let meta_header = codec_util::index_header_length(variable_gap::META_CODEC_NAME, SUFFIX);
    let index_header = codec_util::index_header_length(variable_gap::CODEC_NAME, SUFFIX);
    // A field's `.tiv` offset off by one ("Gap in FST").
    let mut bad = tmv.clone();
    bad[meta_header + 4] += 1;
    assert_corrupt(VariableGapTermsIndexReader::open(
        &refooter(bad),
        tiv,
        &state,
    ));
    // The same field twice, its FST twice in the `.tiv`.
    let body = &tiv[index_header..tiv.len() - FOOTER_LENGTH];
    let rest = &tmv[meta_header + 4 + 1..tmv.len() - FOOTER_LENGTH - 4];
    let mut meta = tmv[..meta_header].to_vec();
    for offset in [index_header, index_header + body.len()] {
        meta.write_i32(0);
        meta.write_vlong(offset as i64);
        meta.extend_from_slice(rest);
    }
    meta.write_i32(-1);
    codec_util::write_footer(&mut meta);
    let mut index = tiv[..index_header].to_vec();
    index.extend_from_slice(body);
    index.extend_from_slice(body);
    codec_util::write_footer(&mut index);
    assert_corrupt(VariableGapTermsIndexReader::open(&meta, &index, &state));
}

/// One block: `count`, `prefix`, the suffixes, then the stats and
/// metadata blobs as given.
fn block(count: i32, prefix: i32, suffixes: &[&[u8]], freqs: &[u8], meta: &[u8]) -> Vec<u8> {
    let mut blob = Vec::new();
    for s in suffixes {
        blob.write_vint(s.len() as i32);
        blob.extend_from_slice(s);
    }
    let mut out = Vec::new();
    out.write_vint(count);
    out.write_vint(prefix);
    for b in [&blob[..], freqs, meta] {
        out.write_vint(b.len() as i32);
        out.extend_from_slice(b);
    }
    out
}

/// A reader over `field` (field `f`'s blocks and end marker) with
/// `num_terms` terms, its fixed-gap index holding one index term (the
/// empty prefix of `b`) at the field's start.
fn handmade(field: &[u8], num_terms: i64) -> BlockTermsReader {
    let group = write(Kind::Fixed(1000), &[b"b".to_vec(), b"c".to_vec()], false).unwrap();
    let start = codec_util::index_header_length(CODEC_NAME, SUFFIX)
        + codec_util::index_header_length(POSTINGS_TERMS_CODEC, SUFFIX)
        + 2; // vInt(128), the postings block size
    let mut tib = group.tib[..start].to_vec();
    tib.extend_from_slice(field);
    let dir_start = tib.len();
    tib.extend_from_slice(&tib_dir(1, &[(0, num_terms, start as i64, None, 3, 3)]));
    tib.write_i64(dir_start as i64);
    codec_util::write_footer(&mut tib);
    let index = index_of(Kind::Fixed(1000), &group).unwrap();
    BlockTermsReader::open(tib, index, &read_state(&group.fis, 3)).unwrap()
}

#[test]
fn malformed_blocks_are_errors() {
    let end = vec![0u8];
    // A negative term count.
    let mut negative = Vec::new();
    negative.write_vint(-1);
    let r = handmade(&negative, 1);
    assert_corrupt(r.terms("f").unwrap().iterator().seek_ceil(b"b"));
    // The index points at the end marker.
    let r = handmade(&end, 1);
    assert_corrupt(r.terms("f").unwrap().iterator().seek_ceil(b"b"));
    assert_corrupt(r.terms("f").unwrap().iterator().seek_exact_ord(0));
    // A block prefix longer than the index term and than the term before.
    let r = handmade(
        &[block(1, 5, &[b"x"], &[1], &[0, 0]), end.clone()].concat(),
        1,
    );
    assert_corrupt(r.terms("f").unwrap().iterator().seek_ceil(b"bzzzzzz"));
    assert_corrupt(r.terms("f").unwrap().iterator().next().map(|t| t.is_some()));
    // Metadata that does not decode.
    let r = handmade(&[block(1, 0, &[b"b"], &[1], &[]), end.clone()].concat(), 1);
    let mut e = r.terms("f").unwrap().iterator();
    assert!(e.seek_exact(b"b").unwrap());
    assert!(e.doc_freq().is_err());
    // More terms in the directory than in the blocks.
    let r = handmade(
        &[block(1, 0, &[b"b"], &[1], &[0, 0]), end.clone()].concat(),
        3,
    );
    assert_corrupt(r.terms("f").unwrap().iterator().seek_exact_ord(2));
    // A term state at the end marker, or past its block.
    // `docStartFP` 0 (`<< 1`), singleton document 0.
    let one = block(1, 0, &[b"b"], &[1], &[0, 0]);
    let r = handmade(&[one.clone(), end].concat(), 1);
    let fr = r.terms("f").unwrap();
    assert!(format!("{fr:?}").contains("BlockTermsFieldReader"));
    let mut e = fr.iterator();
    assert!(e.seek_exact(b"b").unwrap());
    let state = e.term_state().unwrap();
    let at_end = BlockTermState {
        block_file_pointer: state.block_file_pointer + one.len(),
        ..state
    };
    let mut e = fr.iterator();
    e.seek_exact_state(b"b", &at_end);
    assert_corrupt(e.next().map(|t| t.is_some()));
    let past = BlockTermState {
        term_block_ord: 5,
        ..state
    };
    let mut e = fr.iterator();
    e.seek_exact_state(b"b", &past);
    assert_corrupt(e.next().map(|t| t.is_some()));
}

#[test]
fn the_writer_and_a_wrong_header_are_reported() {
    let fis = infos();
    let state = write_state(&fis, 3);
    let w = BlockTermsWriter::new(FixedGapTermsIndexWriter::new(&state, 4).unwrap(), &state);
    assert!(format!("{w:?}").contains("BlockTermsWriter"));
    let group = write(Kind::Fixed(4), &corpus(), false).unwrap();
    let index = index_of(Kind::Fixed(4), &group).unwrap();
    // The `.tii` is not a `.tib`.
    assert!(BlockTermsReader::open(group.tii.clone(), index, &read_state(&group.fis, 3)).is_err());
}
