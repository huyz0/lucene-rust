//! Unit tests of the uniform split terms dictionary: the writer's blocks
//! read back through both enums under several block settings and an
//! encoder (every seek shape against a sorted scan, forced term states,
//! intersections against the automaton run on every term), and the
//! reader's refusals. Format fidelity is
//! `tests/codecs_uniformsplit_fixtures.rs`' job, against Java's bytes.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_store::codec_util;
use lucene_store::data_output::DataOutput;
use lucene_util::automaton::{operations, AutomatonType, CompiledAutomaton, RegExp};

use super::block_line::{num_bits_to_encode, write_line};
use super::field_metadata::FieldMetadata;
use super::*;
use crate::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use crate::postings::{DocInput, PostingsFlags};
use crate::postings_writer::{FieldPostingsInput, TermPostings};
use crate::segment_state::{SegmentReadState, SegmentWriteState};

const ID: [u8; 16] = *b"uniformsplitidss";
const SUFFIX: &str = "UniformSplit_0";

/// Every byte XOR 0x5a, then the byte sum of the plain bytes.
struct Xor;

impl BlockEncoder for Xor {
    fn encode(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        let mut out: Vec<u8> = bytes.iter().map(|b| b ^ 0x5a).collect();
        out.push(bytes.iter().fold(0u8, |s, b| s.wrapping_add(*b)));
        Ok(out)
    }
}

impl BlockDecoder for Xor {
    fn decode(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        let (&sum, body) = bytes
            .split_last()
            .ok_or_else(|| Error::IllegalArgument("empty".into()))?;
        let out: Vec<u8> = body.iter().map(|b| b ^ 0x5a).collect();
        if out.iter().fold(0u8, |s, b| s.wrapping_add(*b)) != sum {
            return Err(Error::IllegalArgument("sum".into()));
        }
        Ok(out)
    }
}

/// An encoder that always fails.
struct Failing;

impl BlockEncoder for Failing {
    fn encode(&self, _: &[u8]) -> Result<Vec<u8>> {
        Err(Error::IllegalArgument("refused".into()))
    }
}

impl BlockDecoder for Failing {
    fn decode(&self, _: &[u8]) -> Result<Vec<u8>> {
        Err(Error::IllegalArgument("refused".into()))
    }
}

fn xor_format(target: i32, delta: i32) -> UniformSplitPostingsFormat {
    let xor = Arc::new(Xor);
    UniformSplitPostingsFormat::new(target, delta, Some(xor.clone()), Some(xor), false).unwrap()
}

fn infos() -> FieldInfos {
    FieldInfos::new(vec![
        FieldInfo::new("f", 0).with_index_options(IndexOptions::Docs),
        FieldInfo::new("g", 1).with_index_options(IndexOptions::DocsAndFreqs),
    ])
    .unwrap()
}

/// Field `f` (DOCS) holding `terms`, term `i` in document `i`, and when
/// `with_g` field `g` (freqs) holding `g`/`h` (`h` in three documents).
fn write(
    format: &UniformSplitPostingsFormat,
    terms: &[Vec<u8>],
    with_g: bool,
) -> Result<(UniformSplitOutput, FieldInfos, i32)> {
    let fis = infos();
    let max_doc = terms.len().max(3) as i32;
    let f: Vec<TermPostings> = terms
        .iter()
        .enumerate()
        .map(|(i, t)| TermPostings {
            term: t.clone(),
            docs: vec![(i as i32, 1)],
            ..TermPostings::default()
        })
        .collect();
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
            doc_count: terms.len() as i32,
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
    let state = SegmentWriteState {
        max_doc,
        segment_id: &ID,
        segment_suffix: SUFFIX,
        field_infos: &fis,
    };
    let out = format.write(&inputs, &[], &state)?;
    Ok((out, fis, max_doc))
}

fn read_state(fis: &FieldInfos, max_doc: i32) -> SegmentReadState<'_> {
    SegmentReadState {
        max_doc,
        segment_id: &ID,
        segment_suffix: SUFFIX,
        field_infos: fis,
    }
}

fn open(
    format: &UniformSplitPostingsFormat,
    ustb: Vec<u8>,
    ustd: &[u8],
    fis: &FieldInfos,
    max_doc: i32,
) -> Result<UniformSplitTermsReader> {
    format.open(ustb, ustd, &read_state(fis, max_doc))
}

/// The empty term, terms sharing long prefixes, chains of one byte and a
/// few high bytes -- sorted, distinct.
fn corpus() -> Vec<Vec<u8>> {
    let mut terms: Vec<Vec<u8>> = vec![Vec::new()];
    for i in 0..700 {
        terms.push(format!("a{:x}", i * 7919 % 4096).into_bytes());
    }
    for i in 0..200 {
        terms.push(format!("bcdefghijklmnop{i:03}x").into_bytes());
    }
    for n in 1..40 {
        terms.push(vec![b'c'; n]);
    }
    terms.push(vec![b'd']);
    terms.push(vec![0xff, 0x00, 0x7f]);
    terms.push(vec![0xff; 300]);
    terms.sort();
    terms.dedup();
    terms
}

/// Seek targets: every term, just past it, its prefix, and a few outside.
fn probes(terms: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for (i, t) in terms.iter().enumerate() {
        if i % 3 == 0 {
            out.push(t.clone());
            let mut longer = t.clone();
            longer.push(0);
            out.push(longer);
            if !t.is_empty() {
                out.push(t[..t.len() - 1].to_vec());
                let mut bumped = t.clone();
                let last = bumped.len() - 1;
                bumped[last] = bumped[last].wrapping_add(1);
                out.push(bumped);
            }
        }
    }
    out.extend([
        b"0".to_vec(),
        b"zzz".to_vec(),
        vec![0xff; 301],
        vec![0xff, 0x01],
        b"bcdefghijklmnop".to_vec(),
    ]);
    // A fixed shuffle, so one enum seeks back and forth.
    let n = out.len();
    for i in 0..n {
        out.swap(i, (i * 7919 + 13) % n);
    }
    out
}

fn check_roundtrip(format: &UniformSplitPostingsFormat, terms: &[Vec<u8>]) {
    let (out, fis, max_doc) = write(format, terms, true).unwrap();
    let reader = open(format, out.ustb.clone(), &out.ustd, &fis, max_doc).unwrap();
    reader.check_integrity().unwrap();
    assert_eq!(reader.size(), 2);
    assert_eq!(reader.field_names().collect::<Vec<_>>(), ["f", "g"]);
    assert_eq!(reader.version(), VERSION_CURRENT);
    assert!(format!("{reader:?}").contains("UniformSplitTermsReader"));
    let f = reader.terms("f").unwrap();
    assert!(format!("{f:?}").contains("\"f\""));
    assert_eq!(f.size(), terms.len() as i64);
    assert_eq!(f.sum_doc_freq(), terms.len() as i64);
    assert_eq!(f.sum_total_term_freq(), terms.len() as i64);
    assert_eq!(f.doc_count(), terms.len() as i32);
    assert!(!f.has_freqs() && !f.has_positions() && !f.has_offsets() && !f.has_payloads());
    assert_eq!(
        f.min().unwrap().as_deref(),
        terms.first().map(Vec::as_slice)
    );
    assert_eq!(f.max(), terms.last().unwrap().as_slice());
    assert_eq!(f.field_info().name, "f");
    assert_eq!(f.metadata().num_terms(), terms.len() as i64);
    assert!(f.metadata().dictionary_start_fp() >= 0);
    assert!(f.metadata().first_block_start_fp() <= f.metadata().last_block_start_fp());
    assert_eq!(f.metadata().last_term(), f.max());
    assert_eq!(f.metadata().doc_count(), f.doc_count());
    assert_eq!(f.metadata().sum_doc_freq(), f.sum_doc_freq());
    assert_eq!(f.metadata().sum_total_term_freq(), f.sum_total_term_freq());
    assert_eq!(f.metadata().field_info().number, 0);
    assert!(!f.dictionary().fst().save().is_empty());
    assert!(reader.terms("nope").is_none());

    // Every term, its state and its postings.
    let doc_in = DocInput::open(&out.postings.doc, &ID, SUFFIX).unwrap();
    let mut walk = f.iter();
    assert!(format!("{walk:?}").contains("BlockReader"));
    for (i, t) in terms.iter().enumerate() {
        assert_eq!(walk.next().unwrap(), Some(t.as_slice()), "{i}");
        assert_eq!(walk.doc_freq().unwrap(), 1);
        assert_eq!(walk.total_term_freq().unwrap(), 1);
        let p = walk.postings(&doc_in, PostingsFlags::Freqs).unwrap();
        assert_eq!(p.docs, [i as i32]);
    }
    assert_eq!(walk.next().unwrap(), None);
    // Past the end, Java starts over.
    assert_eq!(walk.next().unwrap(), Some(terms[0].as_slice()));

    // Seeks against a sorted scan, on one reused enum.
    let mut e = f.iter();
    for target in probes(terms) {
        let at = terms.partition_point(|t| t.as_slice() < target.as_slice());
        let status = e.seek_ceil(&target).unwrap();
        match terms.get(at) {
            None => assert_eq!(status, SeekStatus::End, "{target:x?}"),
            Some(t) => {
                let want = if *t == target {
                    SeekStatus::Found
                } else {
                    SeekStatus::NotFound
                };
                assert_eq!(status, want, "{target:x?}");
                assert_eq!(e.term(), Some(t.as_slice()));
                assert_eq!(e.doc_freq().unwrap(), 1);
                assert_eq!(
                    e.next().unwrap(),
                    terms.get(at + 1).map(Vec::as_slice),
                    "{target:x?}"
                );
            }
        }
        if status == SeekStatus::End {
            e = f.iter();
        }
        let present = terms.binary_search(&target).is_ok();
        assert_eq!(f.iter().seek_exact(&target).unwrap(), present);
        let mut reused = f.iter();
        assert_eq!(reused.seek_exact(&target).unwrap(), present);
        // Seeking the current term again is a no-op.
        if present {
            assert!(reused.seek_exact(&target).unwrap());
            assert_eq!(reused.seek_ceil(&target).unwrap(), SeekStatus::Found);
        }
    }

    // Term states move between enums.
    for i in (0..terms.len()).step_by(17) {
        let mut a = f.iter();
        assert!(a.seek_exact(&terms[i]).unwrap());
        let state = a.term_state().unwrap();
        assert_eq!(state.term_block_ord, a.term_state().unwrap().term_block_ord);
        let mut b = f.iter();
        b.seek_exact_state(&terms[i], &state);
        assert_eq!(b.term(), Some(terms[i].as_slice()));
        assert_eq!(b.term_state().unwrap(), state);
        assert_eq!(b.next().unwrap(), terms.get(i + 1).map(Vec::as_slice));
        // On its own, positioned enum too, and on one positioned elsewhere.
        a.seek_exact_state(&terms[i], &state);
        assert_eq!(a.next().unwrap(), terms.get(i + 1).map(Vec::as_slice));
        let mut c = f.iter();
        c.seek_ceil(terms.last().unwrap()).unwrap();
        c.seek_exact_state(&terms[i], &state);
        assert_eq!(c.next().unwrap(), terms.get(i + 1).map(Vec::as_slice));
    }

    // Field `g`: freqs, a singleton and a three-document term.
    let g = reader.terms("g").unwrap();
    assert!(g.has_freqs());
    assert_eq!(g.sum_total_term_freq(), 11);
    let mut e = g.iter();
    assert!(e.seek_exact(b"h").unwrap());
    assert_eq!(e.total_term_freq().unwrap(), 8);
    let p = e.postings(&doc_in, PostingsFlags::Freqs).unwrap();
    assert_eq!(p.docs, [0, 1, 2]);
    assert_eq!(p.freqs, [1, 2, 5]);
    assert!(e.seek_exact(b"g").unwrap());
    let p = e.postings(&doc_in, PostingsFlags::Freqs).unwrap();
    assert_eq!((p.docs, p.freqs), (vec![0], vec![3]));

    // Intersections against the automaton run on every term.
    for pattern in [
        "a.*",
        ".*1.*",
        "a[0-3].?",
        "c{3,5}",
        "bcdefghijklmnop1[0-4].x",
        ".*",
        "d",
        "a[^0-9]*",
        "c*",
    ] {
        let compiled = compile(pattern);
        if compiled.automaton_type != AutomatonType::NORMAL {
            assert!(matches!(
                f.intersect(&compiled, None),
                Err(Error::Unsupported(_))
            ));
            continue;
        }
        let run = compiled.get_byte_runnable().unwrap();
        for start in [None, Some(&b"a8"[..]), Some(b"c"), Some(b"zz"), Some(b"")] {
            let want: Vec<&[u8]> = terms
                .iter()
                .map(Vec::as_slice)
                .filter(|t| start.is_none_or(|s| *t > s) && run.run(t))
                .collect();
            let mut e = f.intersect(&compiled, start).unwrap();
            let mut got: Vec<Vec<u8>> = Vec::new();
            while let Some(t) = e.next().unwrap().map(<[u8]>::to_vec) {
                assert_eq!(e.term(), Some(t.as_slice()));
                got.push(t);
                assert_eq!(e.doc_freq().unwrap(), 1);
                assert_eq!(e.total_term_freq().unwrap(), 1);
                assert_eq!(e.term_state().unwrap().doc_freq, 1);
            }
            assert_eq!(got, want, "{pattern} {start:?}");
            assert!(format!("{e:?}").contains("IntersectBlockReader"));
        }
    }
}

fn compile(pattern: &str) -> CompiledAutomaton {
    let a = RegExp::with_flags(pattern, RegExp::ALL, 0)
        .unwrap()
        .to_automaton()
        .unwrap();
    let a = operations::determinize(&a, operations::DEFAULT_DETERMINIZE_WORK_LIMIT).unwrap();
    let finite = operations::is_finite(&a).unwrap();
    CompiledAutomaton::with_options(&a, finite, true, false).unwrap()
}

#[test]
fn round_trips_under_every_setting() {
    let terms = corpus();
    check_roundtrip(&UniformSplitPostingsFormat::default(), &terms);
    check_roundtrip(
        &UniformSplitPostingsFormat::new(8, 2, None, None, true).unwrap(),
        &terms,
    );
    check_roundtrip(
        &UniformSplitPostingsFormat::new(3, 0, None, None, true).unwrap(),
        &terms,
    );
    check_roundtrip(
        &UniformSplitPostingsFormat::new(990, 10, None, None, true).unwrap(),
        &terms,
    );
    check_roundtrip(&xor_format(16, 3), &terms);
    // One term, and the empty term alone.
    check_roundtrip(&xor_format(4, 1), &[b"x".to_vec()]);
    check_roundtrip(&UniformSplitPostingsFormat::default(), &[Vec::new()]);
}

#[test]
fn postings_go_through_an_intersect_enum() {
    let format = UniformSplitPostingsFormat::default();
    let terms = corpus();
    let (out, fis, max_doc) = write(&format, &terms, true).unwrap();
    let reader = open(&format, out.ustb.clone(), &out.ustd, &fis, max_doc).unwrap();
    let doc_in = DocInput::open(&out.postings.doc, &ID, SUFFIX).unwrap();
    let g = reader.terms("g").unwrap();
    assert!(format!("{:?}", g.dictionary().browser()).contains("Browser"));
    let compiled = compile("[h-i]");
    assert_eq!(compiled.automaton_type, AutomatonType::NORMAL);
    let mut e = g.intersect(&compiled, None).unwrap();
    assert_eq!(e.next().unwrap(), Some(&b"h"[..]));
    assert_eq!(
        e.postings(&doc_in, PostingsFlags::Freqs).unwrap().freqs,
        [1, 2, 5]
    );
    assert_eq!(e.next().unwrap(), None);
    assert!(e.seek_ceil(b"h").is_err());
    // Not NORMAL: refused.
    let all = compile(".*");
    let any = CompiledAutomaton::with_options(
        &RegExp::with_flags(".*", RegExp::ALL, 0)
            .unwrap()
            .to_automaton()
            .unwrap(),
        false,
        true,
        false,
    )
    .unwrap();
    for c in [&all, &any] {
        if c.automaton_type != AutomatonType::NORMAL {
            assert!(matches!(g.intersect(c, None), Err(Error::Unsupported(_))));
        }
    }
    let mut e = g.iter();
    assert!(matches!(e.doc_freq(), Err(Error::Unsupported(_))));
    assert!(e.ord().is_err());
    assert!(e.seek_exact_ord(0).is_err());
}

#[test]
fn settings_are_validated() {
    for (target, delta) in [(0, 0), (-1, 0), (10, -1), (10, 10), (900, 101)] {
        assert!(matches!(
            UniformSplitPostingsFormat::new(target, delta, None, None, false),
            Err(Error::IllegalArgument(_))
        ));
    }
    assert!(validate_settings(999, 1).is_ok());
    let xor = Arc::new(Xor);
    assert!(UniformSplitPostingsFormat::new(32, 3, Some(xor.clone()), None, false).is_err());
    assert!(UniformSplitPostingsFormat::new(32, 3, None, Some(xor), false).is_err());
    let format = UniformSplitPostingsFormat::default();
    assert_eq!(
        (
            format.target_num_block_lines(),
            format.delta_num_lines(),
            format.dictionary_on_heap()
        ),
        (
            DEFAULT_TARGET_NUM_BLOCK_LINES,
            DEFAULT_DELTA_NUM_LINES,
            false
        )
    );
    assert!(format!("{format:?}").contains("UniformSplitPostingsFormat"));
}

#[test]
fn writer_refuses_what_java_cannot_write() {
    // 2/1: the field's last two lines, the first with the smaller
    // mdpLength, would be split into an empty block (Java fails too).
    let format = UniformSplitPostingsFormat::new(2, 1, None, None, false).unwrap();
    assert!(matches!(
        write(&format, &[b"a".to_vec(), b"ab".to_vec()], false),
        Err(Error::IllegalArgument(_))
    ));
    assert!(write(&format, &[b"a".to_vec(), b"b".to_vec()], false).is_ok());
    // 3/2: two lines left over put the split's scan start at -2, Java's
    // `IndexOutOfBoundsException`.
    let format = UniformSplitPostingsFormat::new(3, 2, None, None, false).unwrap();
    let err = write(&format, &[b"a".to_vec(), b"b".to_vec()], false).unwrap_err();
    assert!(err.to_string().contains("cannot be split"), "{err}");
    // An encoder's failure, for a block, a dictionary and the metadata.
    let failing = Arc::new(Failing);
    let format =
        UniformSplitPostingsFormat::new(32, 3, Some(failing.clone()), Some(failing), false)
            .unwrap();
    assert!(write(&format, &[b"a".to_vec()], false).is_err());
    assert!(write(&format, &[], false).is_err());
    // A field number the segment lacks.
    let fis = infos();
    let state = SegmentWriteState {
        max_doc: 1,
        segment_id: &ID,
        segment_suffix: SUFFIX,
        field_infos: &fis,
    };
    let t = vec![TermPostings {
        term: b"a".to_vec(),
        docs: vec![(0, 1)],
        ..TermPostings::default()
    }];
    let inputs = [FieldPostingsInput {
        field_number: 7,
        index_options: IndexOptions::Docs,
        doc_count: 1,
        has_payloads: false,
        terms: &t,
    }];
    assert!(UniformSplitPostingsFormat::default()
        .write(&inputs, &[], &state)
        .is_err());
}

/// Rewrites the `.ustb` footer's checksum after an edit (only
/// `check_integrity` reads it).
fn refooter(mut ustb: Vec<u8>) -> Vec<u8> {
    let end = ustb.len() - codec_util::FOOTER_LENGTH;
    ustb.truncate(end);
    codec_util::write_footer(&mut ustb);
    ustb
}

/// A `.ustb` whose fields metadata is `metadata` (`num_fields` entries).
fn with_metadata(ustb: &[u8], num_fields: i32, metadata: &[u8]) -> Vec<u8> {
    let at = ustb.len() - codec_util::FOOTER_LENGTH - 8;
    let start = i64::from_le_bytes(ustb[at..at + 8].try_into().unwrap()) as usize;
    let mut out = ustb[..start].to_vec();
    out.write_vint(num_fields);
    out.write_bytes(metadata);
    out.write_i64(start as i64);
    codec_util::write_footer(&mut out);
    out
}

#[test]
fn reader_refuses_damaged_files() {
    let format = UniformSplitPostingsFormat::default();
    let terms = corpus();
    let (out, fis, max_doc) = write(&format, &terms, true).unwrap();
    let ustb = out.ustb.clone();
    let ustd = out.ustd.clone();
    let try_open = |b: Vec<u8>, d: &[u8]| open(&format, b, d, &fis, max_doc);
    assert!(try_open(ustb.clone(), &ustd).is_ok());
    // Too short for anything.
    assert!(try_open(ustb[..10].to_vec(), &ustd).is_err());
    assert!(try_open(ustb.clone(), &ustd[..10]).is_err());
    // A fields offset past its own position.
    let mut bad = ustb.clone();
    let at = bad.len() - codec_util::FOOTER_LENGTH - 8;
    bad[at..at + 8].copy_from_slice(&(at as i64 + 1).to_le_bytes());
    assert!(try_open(refooter(bad), &ustd).is_err());
    // A negative field count.
    assert!(try_open(with_metadata(&ustb, -1, &[]), &ustd).is_err());
    // No fields at all opens empty.
    let empty = try_open(with_metadata(&ustb, 0, &[]), &ustd).unwrap();
    assert_eq!(empty.size(), 0);
    // A postings block size other than 256.
    let mut bad = ustb.clone();
    // Past this format's index header and the postings terms header: magic,
    // codec name, version, id, suffix.
    let mut pos = codec_util::index_header_length(NAME, SUFFIX);
    pos += 4 + 1 + usize::from(bad[pos + 4]) + 4 + 16 + 1 + SUFFIX.len();
    // vInt(256) is two bytes: 0x80 0x02.
    assert_eq!(&bad[pos..pos + 2], [0x80, 0x02]);
    bad[pos + 1] = 0x03;
    assert!(matches!(
        try_open(refooter(bad), &ustd),
        Err(Error::Unsupported(_))
    ));

    // Field metadata Java refuses.
    let reader = try_open(ustb.clone(), &ustd).unwrap();
    let good = reader.terms("f").unwrap().metadata().clone();
    let edits: [&dyn Fn(&mut FieldMetadata); 7] = [
        &|m| m.num_terms = 0,
        &|m| m.doc_count = -1,
        &|m| m.doc_count = max_doc + 1,
        &|m| m.sum_doc_freq = 0,
        &|m| m.dictionary_start_fp = 1 << 40,
        &|m| m.dictionary_start_fp = 3,
        &|m| m.field_info.number = 9,
    ];
    for edit in edits {
        let mut m = good.clone();
        edit(&mut m);
        let mut buf = Vec::new();
        m.write(&mut buf);
        assert!(try_open(with_metadata(&ustb, 1, &buf), &ustd).is_err());
    }
    // `g` has freqs: a sumTotalTermFreq delta that overflows.
    let mut m = reader.terms("g").unwrap().metadata().clone();
    m.sum_total_term_freq = m.sum_doc_freq.wrapping_add(i64::MAX);
    let mut buf = Vec::new();
    m.write(&mut buf);
    assert!(try_open(with_metadata(&ustb, 1, &buf), &ustd).is_err());
    // A negative last term length, and one longer than the file.
    let mut buf = Vec::new();
    good.write(&mut buf);
    let len_at = buf.len() - good.last_term.len() - 1;
    let mut neg = buf[..len_at].to_vec();
    neg.write_vint(-1);
    assert!(try_open(with_metadata(&ustb, 1, &neg), &ustd).is_err());
    let mut long = buf[..len_at].to_vec();
    long.write_vint(1 << 20);
    assert!(try_open(with_metadata(&ustb, 1, &long), &ustd).is_err());
    // The same field twice: the later one wins, as `HashMap.put`.
    let mut twice = buf.clone();
    twice.extend_from_slice(&buf);
    assert_eq!(
        try_open(with_metadata(&ustb, 2, &twice), &ustd)
            .unwrap()
            .size(),
        1
    );

    // A .ustd of another version than the .ustb.
    let mut v0 = ustd.clone();
    // The version is the int right after the magic and the codec name.
    let at = 4 + 1 + NAME.len();
    v0[at + 3] = 0;
    let end = v0.len() - codec_util::FOOTER_LENGTH;
    v0.truncate(end);
    codec_util::write_footer(&mut v0);
    assert!(try_open(ustb.clone(), &v0).is_err());

    // An encoded group: its metadata's length, and a decoder that fails.
    let xor = xor_format(16, 3);
    let (out, ..) = write(&xor, &terms, true).unwrap();
    assert!(open(&xor, out.ustb.clone(), &out.ustd, &fis, max_doc).is_ok());
    let at = out.ustb.len() - codec_util::FOOTER_LENGTH - 8;
    let start = i64::from_le_bytes(out.ustb[at..at + 8].try_into().unwrap()) as usize;
    let mut neg = out.ustb[..start].to_vec();
    neg.write_vint(2);
    // vLong -1: nine bytes, the last holding eight bits.
    neg.extend_from_slice(&[0xff; 9]);
    neg.write_i64(start as i64);
    codec_util::write_footer(&mut neg);
    assert!(open(&xor, neg, &out.ustd, &fis, max_doc).is_err());
    let failing = Arc::new(Failing);
    let refusing =
        UniformSplitPostingsFormat::new(16, 3, Some(failing.clone()), Some(failing), false)
            .unwrap();
    assert!(open(&refusing, out.ustb.clone(), &out.ustd, &fis, max_doc).is_err());
}

/// Where `f`'s first block's bytes start, after their vInt length.
fn first_block(reader: &UniformSplitTermsReader, ustb: &[u8]) -> usize {
    let mut pos = reader.terms("f").unwrap().metadata().first_block_start_fp() as usize;
    read_vint(ustb, &mut pos).unwrap();
    pos
}

#[test]
fn reader_refuses_damaged_blocks() {
    let format = UniformSplitPostingsFormat::new(8, 2, None, None, false).unwrap();
    let terms = corpus();
    let (out, fis, max_doc) = write(&format, &terms, false).unwrap();
    let reader = open(&format, out.ustb.clone(), &out.ustd, &fis, max_doc).unwrap();
    let body = first_block(&reader, &out.ustb);
    let damaged = |edit: &dyn Fn(&mut Vec<u8>)| {
        let mut ustb = out.ustb.clone();
        edit(&mut ustb);
        open(&format, ustb, &out.ustd, &fis, max_doc).unwrap()
    };
    // A block of zero lines.
    let r = damaged(&|b| b[body] = 0);
    let f = r.terms("f").unwrap();
    assert!(f.iter().next().is_err());
    assert!(f.iter().seek_ceil(b"a").is_err());
    assert!(f.iter().seek_exact(b"").is_err());
    // A middle line offset past the block: the middle line cannot be read.
    // linesCount, three vLong bases, termStatesBaseOffset, then
    // middleLineOffset, widened to i32::MAX: every later byte moves by 4,
    // so the fields metadata is rewritten after the moved blocks.
    let mut pos = body;
    for _ in 0..5 {
        read_vlong(&out.ustb, &mut pos).unwrap();
    }
    let mut end = pos;
    read_vint(&out.ustb, &mut end).unwrap();
    let at = out.ustb.len() - codec_util::FOOTER_LENGTH - 8;
    let start = i64::from_le_bytes(out.ustb[at..at + 8].try_into().unwrap()) as usize;
    let fp = reader.terms("f").unwrap().metadata().first_block_start_fp() as usize;
    let mut moved = out.ustb[..fp].to_vec();
    let mut len_at = fp;
    let block_len = read_vint(&out.ustb, &mut len_at).unwrap();
    moved.write_vint(block_len + 4);
    assert_eq!(moved.len(), body, "the block length keeps its width");
    moved.extend_from_slice(&out.ustb[body..pos]);
    moved.write_vint(i32::MAX);
    moved.extend_from_slice(&out.ustb[end..start]);
    let new_start = moved.len();
    let mut m = reader.terms("f").unwrap().metadata().clone();
    m.last_block_start_fp += (new_start - start) as i64;
    moved.write_vint(1);
    m.write(&mut moved);
    moved.write_i64(new_start as i64);
    codec_util::write_footer(&mut moved);
    let r = open(&format, moved, &out.ustd, &fis, max_doc).unwrap();
    assert!(r.terms("f").unwrap().iter().seek_ceil(b"").is_err());
    // A first block pointer past the file.
    let mut bad = reader.terms("f").unwrap().metadata().clone();
    bad.first_block_start_fp = 1 << 40;
    let mut buf = Vec::new();
    bad.write(&mut buf);
    let r = open(
        &format,
        with_metadata(&out.ustb, 1, &buf),
        &out.ustd,
        &fis,
        max_doc,
    )
    .unwrap();
    assert!(r.terms("f").unwrap().iter().next().is_err());
    // A first block after the last: no first block.
    let mut bad = reader.terms("f").unwrap().metadata().clone();
    bad.first_block_start_fp = bad.last_block_start_fp + 1;
    let mut buf = Vec::new();
    bad.write(&mut buf);
    let r = open(
        &format,
        with_metadata(&out.ustb, 1, &buf),
        &out.ustd,
        &fis,
        max_doc,
    )
    .unwrap();
    let f = r.terms("f").unwrap();
    assert!(f.iter().next().is_err());
    // A forced state pointing past the last block.
    let mut e = reader.terms("f").unwrap().iter();
    assert!(e.seek_exact(b"d").unwrap());
    let mut state = e.term_state().unwrap();
    state.block_file_pointer = reader.terms("f").unwrap().metadata().last_block_start_fp() + 1;
    let mut b = reader.terms("f").unwrap().iter();
    b.seek_exact_state(b"d", &state);
    assert!(b.next().is_err());
    // A term state offset past its block.
    let r = damaged(&|b| {
        let mut pos = body;
        for _ in 0..4 {
            read_vlong(b, &mut pos).unwrap();
        }
        b[pos] = 0x7f;
    });
    let f = r.terms("f").unwrap();
    let mut e = f.iter();
    if e.next().is_ok() {
        assert!(e.doc_freq().is_err());
    }
}

#[test]
fn intersect_refuses_a_dictionary_that_seeks_back() {
    // A field whose dictionary maps every key to the first block, so the
    // intersect enum's seek from a later block goes backwards.
    let format = UniformSplitPostingsFormat::new(4, 1, None, None, false).unwrap();
    let terms: Vec<Vec<u8>> = (0..200).map(|i| format!("k{i:03}").into_bytes()).collect();
    let (out, fis, max_doc) = write(&format, &terms, false).unwrap();
    let reader = open(&format, out.ustb.clone(), &out.ustd, &fis, max_doc).unwrap();
    let f = reader.terms("f").unwrap();
    let first = f.metadata().first_block_start_fp();
    let mut builder = super::dictionary::FstDictionaryBuilder::new();
    builder.add(b"k", first).unwrap();
    let mut ustd = out.ustd[..f.metadata().dictionary_start_fp() as usize].to_vec();
    builder.build().unwrap().write(&mut ustd, None).unwrap();
    codec_util::write_footer(&mut ustd);
    let r = open(&format, out.ustb.clone(), &ustd, &fis, max_doc).unwrap();
    let f = r.terms("f").unwrap();
    // A pattern that rejects runs of lines and then skips ahead.
    let compiled = compile("k1[5-9]0");
    let mut e = f.intersect(&compiled, Some(b"k1")).unwrap();
    let mut res = Vec::new();
    loop {
        match e.next() {
            Ok(Some(t)) => res.push(t.to_vec()),
            Ok(None) => break,
            Err(_) => {
                res.clear();
                break;
            }
        }
    }
    // Either refused, or (when no seek was needed) right.
    if !res.is_empty() {
        assert_eq!(res.len(), 5);
    }
}

#[test]
fn lines_and_headers_refuse_corruption() {
    // A negative term state offset.
    let mut buf = Vec::new();
    buf.write_vint(-1);
    let mut reader = BlockLineReader::default();
    assert!(reader.read_line(&buf, &mut 0, true).is_err());
    // A shared prefix longer than the line before.
    let mut buf = Vec::new();
    write_line(&mut buf, b"ab", 1, None, 0, true);
    let mut bad = Vec::new();
    bad.write_vint(0);
    bad.write_vlong((1 << 2) | 3); // mdp - 1 = 3 > 2
    bad.push(b'x');
    let mut pos = 0;
    reader.read_line(&buf, &mut pos, true).unwrap();
    assert!(reader.read_line(&bad, &mut 0, false).is_err());
    // The empty term after another.
    let mut buf = Vec::new();
    write_line(&mut buf, b"", 0, Some(b"ab"), 3, false);
    assert_eq!(reader.read_line(&buf, &mut 0, false).unwrap(), (3, 0));
    assert!(reader.current_term.is_empty());
    // Header refusals.
    for (lines, base_offset, middle) in [(0, 0, 0), (1001, 0, 0), (2, -1, 0), (2, 0, -1)] {
        let mut buf = Vec::new();
        buf.write_vint(lines);
        buf.write_vlong(0);
        buf.write_vlong(0);
        buf.write_vlong(0);
        buf.write_vint(base_offset);
        buf.write_vint(middle);
        assert!(BlockHeader::read(&buf, &mut 0).is_err());
    }
    // A negative file pointer in a term state.
    let mut buf = Vec::new();
    buf.write_vint(2);
    // vLong -1: nine bytes, the last holding eight bits.
    buf.extend_from_slice(&[0xff; 9]);
    assert!(DeltaBaseTermStateSerializer::read_term_state(
        0,
        0,
        0,
        &buf,
        &mut 0,
        IndexOptions::Docs,
        false
    )
    .is_err());
    assert_eq!(num_bits_to_encode(0), 0);
    assert_eq!(num_bits_to_encode(255), 8);
    assert_eq!(num_bits_to_encode(-1), 32);
    assert_eq!(TermBytes::compute_mdp_length(None, b""), 0);
    assert_eq!(TermBytes::compute_mdp_length(Some(b"ab"), b"abc"), 3);
    assert_eq!(TermBytes::suffix_offset(0), 0);
    // A slice past the end, and a negative length.
    assert!(read_slice(&[1, 2], &mut 1, 2).is_err());
    assert!(length(-1).is_err());
}

#[test]
fn term_states_round_trip_with_positions_and_payloads() {
    let mut s = DeltaBaseTermStateSerializer::default();
    let meta = crate::postings::TermMetadata {
        doc_start_fp: 100,
        singleton_doc_id: -1,
        pos_start_fp: 200,
        pay_start_fp: 300,
        last_pos_block_offset: 7,
        skip_offset: -1,
    };
    let mut buf = Vec::new();
    let io = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
    s.write_term_state(&mut buf, io, true, 3, 400, &meta);
    let mut more = meta;
    more.doc_start_fp = 150;
    more.last_pos_block_offset = -1;
    s.write_term_state(&mut buf, io, true, 3, 9, &more);
    let mut pos = 0;
    let a = DeltaBaseTermStateSerializer::read_term_state(
        s.base_doc_start_fp,
        s.base_pos_start_fp,
        s.base_pay_start_fp,
        &buf,
        &mut pos,
        io,
        true,
    )
    .unwrap();
    assert_eq!((a.doc_freq, a.total_term_freq, a.meta), (3, 400, meta));
    let b = DeltaBaseTermStateSerializer::read_term_state(
        s.base_doc_start_fp,
        s.base_pos_start_fp,
        s.base_pay_start_fp,
        &buf,
        &mut pos,
        io,
        true,
    )
    .unwrap();
    assert_eq!(b.meta.doc_start_fp, 150);
    assert_eq!(b.meta.last_pos_block_offset, -1);
    assert_eq!(pos, buf.len());
    s.reset_base_start_fp();
    assert_eq!(s, DeltaBaseTermStateSerializer::default());
}
