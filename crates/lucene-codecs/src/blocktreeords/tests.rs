//! Unit tests of the ords block tree: the outputs algebra, the writer's
//! blocks read back through both enums (every ord, every seek shape, floor
//! blocks and deep trees), and the reader's refusals. Format fidelity is
//! `tests/codecs_blocktreeords_fixtures.rs`' job, against Java's bytes.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_util::automaton::{operations, CompiledAutomaton, RegExp};

use super::*;
use crate::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use crate::fst_compiler::FstOutputs;
use crate::postings::PostingsFlags;
use crate::postings_writer::{FieldPostingsInput, TermPostings};
use crate::segment_state::{SegmentReadState, SegmentWriteState};

const ID: [u8; 16] = *b"ordsblocktreeids";
const SUFFIX: &str = "BlockTreeOrds_0";

fn infos() -> FieldInfos {
    FieldInfos::new(vec![
        FieldInfo::new("f", 0).with_index_options(IndexOptions::Docs),
        FieldInfo::new("g", 1).with_index_options(IndexOptions::DocsAndFreqs),
    ])
    .unwrap()
}

/// Field `f` (DOCS) holding `terms`, term `i` in document `i`, and when
/// `with_g` field `g` (freqs) holding `[b"g"]`, written with `format`.
fn write(
    format: BlockTreeOrdsPostingsFormat,
    terms: &[Vec<u8>],
    with_g: bool,
) -> (BlockTreeOrdsOutput, FieldInfos, i32) {
    let fis = infos();
    let max_doc = terms.len().max(1) as i32;
    let f: Vec<TermPostings> = terms
        .iter()
        .enumerate()
        .map(|(i, t)| TermPostings {
            term: t.clone(),
            docs: vec![(i as i32, 1)],
            ..TermPostings::default()
        })
        .collect();
    let g = vec![TermPostings {
        term: b"g".to_vec(),
        docs: vec![(0, 3)],
        ..TermPostings::default()
    }];
    let mut inputs = vec![FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::Docs,
        doc_count: terms.len() as i32,
        has_payloads: false,
        terms: &f,
    }];
    if with_g {
        inputs.push(FieldPostingsInput {
            field_number: 1,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 1,
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
    let out = format.write(&inputs, &[], &state).unwrap();
    (out, fis, max_doc)
}

fn open(
    out: &BlockTreeOrdsOutput,
    fis: &FieldInfos,
    max_doc: i32,
) -> Result<OrdsBlockTreeTermsReader> {
    let state = SegmentReadState {
        max_doc,
        segment_id: &ID,
        segment_suffix: SUFFIX,
        field_infos: fis,
    };
    BlockTreeOrdsPostingsFormat::default().open(out.tio.clone(), &out.tipo, &state)
}

/// Terms sharing long prefixes (floor blocks under `a`/`b`), a few short
/// ones, and a deep chain -- sorted, distinct.
fn corpus() -> Vec<Vec<u8>> {
    let mut terms: Vec<Vec<u8>> = Vec::new();
    for i in 0..1500 {
        terms.push(format!("a{:x}", i * 7919 % 4096).into_bytes());
    }
    for i in 0..300 {
        terms.push(format!("b{i:03}x").into_bytes());
    }
    for n in 1..40 {
        terms.push(vec![b'c'; n]);
    }
    terms.push(vec![b'd']);
    terms.push(vec![0xff, 0x00, 0x7f]);
    terms.sort();
    terms.dedup();
    terms
}

#[test]
fn outputs_algebra() {
    let a = Output::new(vec![1, 2, 3], 5, 10);
    let b = Output::new(vec![1, 2, 9], 7, 4);
    let c = FstOrdsOutputs::common(&a, &b);
    assert_eq!(c, Output::new(vec![1, 2], 5, 4));
    let rest = FstOrdsOutputs::subtract(&a, &c);
    assert_eq!(rest, Output::new(vec![3], 0, 6));
    assert_eq!(FstOrdsOutputs::add(&c, &rest), a);
    assert_eq!(FstOrdsOutputs::subtract(&a, &Output::NO_OUTPUT), a);
    assert_eq!(FstOrdsOutputs::add(&Output::NO_OUTPUT, &a), a);
    assert_eq!(FstOrdsOutputs::add(&a, &Output::NO_OUTPUT), a);
    assert_eq!(FstOrdsOutputs::no_output(), Output::NO_OUTPUT);
    assert_eq!(Output::new(vec![], 3, i64::MAX - 9).last_ord(), 9);
    let mut buf = Vec::new();
    FstOrdsOutputs::write(&a, &mut buf);
    assert_eq!(buf, [3, 1, 2, 3, 5, 10]);
}

#[test]
fn format_settings() {
    let f = BlockTreeOrdsPostingsFormat::new(10, 20).unwrap();
    assert_eq!((f.min_terms_in_block(), f.max_terms_in_block()), (10, 20));
    assert_eq!(f.name(), "BlockTreeOrds(blocksize=128)");
    assert!(matches!(
        BlockTreeOrdsPostingsFormat::new(1, 20),
        Err(Error::IllegalArgument(_))
    ));
    assert!(BlockTreeOrdsPostingsFormat::new(30, 40).is_err());
    let d = BlockTreeOrdsPostingsFormat::default();
    assert_eq!((d.min_terms_in_block(), d.max_terms_in_block()), (25, 48));
}

/// Every enum operation against the sorted term list, for one block size.
fn check_enum(format: BlockTreeOrdsPostingsFormat) {
    let terms = corpus();
    let (out, fis, max_doc) = write(format, &terms, true);
    let reader = open(&out, &fis, max_doc).unwrap();
    reader.check_integrity().unwrap();
    assert_eq!(reader.size(), 2);
    assert_eq!(reader.field_names().collect::<Vec<_>>(), ["f", "g"]);
    assert!(reader.terms("missing").is_none());
    let fr = reader.terms("f").unwrap();
    assert_eq!(fr.size(), terms.len() as i64);
    assert_eq!(fr.min(), terms[0].as_slice());
    assert_eq!(fr.max(), terms[terms.len() - 1].as_slice());
    assert!(!fr.has_freqs() && !fr.has_positions() && !fr.has_offsets());
    assert!(!fr.has_payloads());
    assert_eq!(fr.root_code().start_ord, 0);
    assert!(fr.root_block_fp() > 0 && fr.index_start_fp() > 0);
    assert!(format!("{fr:?}").contains("OrdsFieldReader"));

    // `next()` over everything, ords in order; then past the end.
    let mut e = fr.iter();
    assert_eq!(e.ord(), -1);
    for (i, t) in terms.iter().enumerate() {
        assert_eq!(e.next().unwrap(), Some(t.as_slice()));
        assert_eq!(e.ord(), i as i64);
        assert_eq!(e.doc_freq().unwrap(), 1);
        assert_eq!(e.total_term_freq().unwrap(), 1);
    }
    assert_eq!(e.next().unwrap(), None);
    assert!(format!("{e:?}").contains("OrdsSegmentTermsEnum"));
    // After the end, Java's enum re-seeks the empty term (asserting it
    // exists): what follows is assertion territory, but never a panic.
    let _ = e.next();

    // `seekExact(ord)` for every ord, forwards then backwards on one enum.
    let mut e = fr.iter();
    for (i, t) in terms.iter().enumerate() {
        e.seek_exact_ord(i as i64).unwrap();
        assert_eq!(e.term(), t.as_slice(), "ord {i}");
        assert_eq!(e.ord(), i as i64);
    }
    for (i, t) in terms.iter().enumerate().rev().step_by(7) {
        e.seek_exact_ord(i as i64).unwrap();
        assert_eq!(e.term(), t.as_slice(), "ord {i}");
        let next = e.next().unwrap().map(<[u8]>::to_vec);
        assert_eq!(next.as_ref(), terms.get(i + 1), "after ord {i}");
    }
    assert!(matches!(
        e.seek_exact_ord(terms.len() as i64),
        Err(Error::IllegalArgument(_))
    ));

    // `seekExact`/`seekCeil` on present and absent targets, reused enum,
    // jumping back and forth.
    let mut probes: Vec<Vec<u8>> = Vec::new();
    for (i, t) in terms.iter().enumerate().step_by(13) {
        probes.push(t.clone());
        let mut longer = t.clone();
        longer.push(b'!');
        probes.push(longer);
        let mut shorter = t.clone();
        shorter.pop();
        probes.push(shorter);
        let mut bumped = t.clone();
        *bumped.last_mut().unwrap() ^= 0x01;
        probes.push(bumped);
        if i % 2 == 0 {
            probes.push(b"a".to_vec());
        }
    }
    probes.extend([vec![], b"zzz".to_vec(), vec![0xff; 4], b"b150".to_vec()]);
    let mut seek = fr.iter();
    let mut ceil = fr.iter();
    for (k, p) in probes.iter().enumerate() {
        let want = terms.binary_search(p);
        assert_eq!(seek.seek_exact(p).unwrap(), want.is_ok(), "{p:?}");
        if let Ok(i) = want {
            // Again, on the same term: answered without a scan.
            assert!(seek.seek_exact(p).unwrap());
            assert_eq!(seek.ord(), i as i64);
        }
        let status = ceil.seek_ceil(p).unwrap();
        match want {
            Ok(i) => {
                assert_eq!(status, SeekStatus::Found, "{p:?}");
                assert_eq!(ceil.ord(), i as i64);
            }
            Err(i) if i == terms.len() => {
                assert_eq!(status, SeekStatus::End, "{p:?}");
                ceil = fr.iter();
                continue;
            }
            Err(i) => {
                assert_eq!(status, SeekStatus::NotFound, "{p:?}");
                assert_eq!(ceil.term(), terms[i].as_slice(), "{p:?}");
                assert_eq!(ceil.ord(), i as i64, "{p:?}");
            }
        }
        if k % 3 == 0 {
            let i = want.unwrap_or_else(|i| i);
            let next = ceil.next().unwrap().map(<[u8]>::to_vec);
            assert_eq!(next.as_ref(), terms.get(i + 1), "{p:?}");
        }
    }

    // Term states move between enums; postings through the state.
    let t = &terms[terms.len() / 3];
    let mut a = fr.iter();
    assert!(a.seek_exact(t).unwrap());
    let state = a.term_state().unwrap();
    assert_eq!(state.doc_freq, 1);
    let meta = a.term_metadata().unwrap();
    assert_eq!(meta, state.meta);
    let mut b = fr.iter();
    b.next().unwrap();
    b.seek_exact_state(t, state);
    assert_eq!(b.term(), t.as_slice());
    assert_eq!(b.ord(), state.ord);
    let doc = out.postings.doc.clone();
    let doc_in = crate::postings::DocInput::open(&doc, &ID, SUFFIX).unwrap();
    let p = b.postings(&doc_in, PostingsFlags::Freqs).unwrap();
    assert_eq!(p.docs, vec![terms.len() as i32 / 3]);

    // The freqs field: its one term.
    let g = reader.terms("g").unwrap();
    assert!(g.has_freqs());
    let mut e = g.iter();
    assert_eq!(e.next().unwrap(), Some(&b"g"[..]));
    assert_eq!(e.total_term_freq().unwrap(), 3);
    assert_eq!(
        g.to_string(),
        "OrdsBlockTreeTerms(terms=1,postings=1,positions=3,docs=1)"
    );
}

#[test]
fn enum_over_default_blocks() {
    check_enum(BlockTreeOrdsPostingsFormat::default());
}

#[test]
fn enum_over_small_blocks() {
    check_enum(BlockTreeOrdsPostingsFormat::new(2, 3).unwrap());
}

fn compile(pattern: &str) -> CompiledAutomaton {
    let a = RegExp::with_flags(pattern, RegExp::ALL, 0)
        .unwrap()
        .to_automaton()
        .unwrap();
    let a = operations::determinize(&a, operations::DEFAULT_DETERMINIZE_WORK_LIMIT).unwrap();
    CompiledAutomaton::with_options(&a, false, true, false).unwrap()
}

fn check_intersect(format: BlockTreeOrdsPostingsFormat) {
    let terms = corpus();
    let (out, fis, max_doc) = write(format, &terms, false);
    let reader = open(&out, &fis, max_doc).unwrap();
    let fr = reader.terms("f").unwrap();
    let doc = out.postings.doc.clone();
    let doc_in = crate::postings::DocInput::open(&doc, &ID, SUFFIX).unwrap();
    let mut matched = 0;
    for pattern in [
        "a.*",
        "a[1-3].*",
        ".*9",
        ".*0x",
        "b1[0-4].?x",
        "c{3,9}",
        "[a-c][0-9a-f]",
        "a[0-7].*f",
        "c*",
        ".*",
        "(a|b)[0-9]*",
        "[ab].*[05]",
        "d|cc",
        "x.*|a",
    ] {
        let compiled = compile(pattern);
        if compiled.automaton_type != lucene_util::automaton::AutomatonType::NORMAL {
            assert!(matches!(
                fr.intersect(&compiled, None),
                Err(Error::Unsupported(_))
            ));
            continue;
        }
        let run = compiled.get_byte_runnable().unwrap();
        for start in [
            None,
            Some(&b"a"[..]),
            Some(b"a8"),
            Some(b"b2"),
            Some(b"ccc"),
        ] {
            if start.is_some_and(|s| {
                let mut st = 0;
                s.iter().any(|&b| {
                    st = run.step(st, i32::from(b));
                    st == -1
                })
            }) {
                continue;
            }
            let mut e = fr.intersect(&compiled, start).unwrap();
            let want: Vec<&Vec<u8>> = terms
                .iter()
                .filter(|t| start.is_none_or(|s| t.as_slice() > s) && run.run(t))
                .collect();
            let mut got = Vec::new();
            while let Some(t) = e.next().unwrap() {
                got.push(t.to_vec());
                assert_eq!(e.doc_freq().unwrap(), 1);
                assert_eq!(e.total_term_freq().unwrap(), 1);
            }
            assert_eq!(got.iter().collect::<Vec<_>>(), want, "{pattern} {start:?}");
            matched += got.len();
        }
    }
    assert!(matched > 3000, "{matched}");
    // Metadata, postings and the unsupported calls.
    let compiled = compile("b.*");
    let mut e = fr.intersect(&compiled, None).unwrap();
    assert_eq!(e.next().unwrap(), Some(&b"b000x"[..]));
    assert_eq!(e.term(), b"b000x");
    let meta = e.term_metadata().unwrap();
    assert_eq!(e.term_state().unwrap().meta, meta);
    let p = e.postings(&doc_in, PostingsFlags::Freqs).unwrap();
    assert_eq!(p.docs.len(), 1);
    assert!(matches!(e.ord(), Err(Error::Unsupported(_))));
    assert!(matches!(e.seek(), Err(Error::Unsupported(_))));
    assert!(format!("{e:?}").contains("OrdsIntersectTermsEnum"));
}

#[test]
fn intersect_over_default_blocks() {
    check_intersect(BlockTreeOrdsPostingsFormat::default());
}

#[test]
fn intersect_over_small_blocks() {
    check_intersect(BlockTreeOrdsPostingsFormat::new(2, 3).unwrap());
}

#[test]
fn empty_and_single_term_fields() {
    let (out, fis, max_doc) = write(BlockTreeOrdsPostingsFormat::default(), &[vec![]], false);
    let reader = open(&out, &fis, max_doc).unwrap();
    let fr = reader.terms("f").unwrap();
    let mut e = fr.iter();
    assert!(e.seek_exact(b"").unwrap());
    assert_eq!(e.seek_ceil(b"x").unwrap(), SeekStatus::End);
    let mut e = fr.iter();
    e.seek_exact_ord(0).unwrap();
    assert_eq!(e.term(), b"");
    assert_eq!(e.next().unwrap(), None);
    assert_eq!(e.next().unwrap(), None, "the empty term is the only one");
}

#[test]
fn reader_refusals() {
    let terms = corpus();
    let (out, fis, max_doc) = write(BlockTreeOrdsPostingsFormat::default(), &terms, true);
    assert!(open(&out, &fis, max_doc).is_ok());
    assert_eq!(
        open(&out, &fis, max_doc).unwrap().to_string(),
        "OrdsBlockTreeTermsReader(fields=2,delegate=Lucene104PostingsReader)"
    );
    // docCount above maxDoc.
    assert!(open(&out, &fis, 3).is_err());
    // A field number the FieldInfos lack.
    let only_g = FieldInfos::new(vec![
        FieldInfo::new("g", 1).with_index_options(IndexOptions::DocsAndFreqs)
    ])
    .unwrap();
    assert!(open(&out, &only_g, max_doc).is_err());
    // Truncated or damaged files.
    let mut short = out.clone();
    short.tio.truncate(10);
    assert!(open(&short, &fis, max_doc).is_err());
    let mut short = out.clone();
    short.tipo.truncate(10);
    assert!(open(&short, &fis, max_doc).is_err());
    let mut flipped = out.clone();
    let at = flipped.tipo.len() / 2;
    flipped.tipo[at] ^= 0x40;
    assert!(open(&flipped, &fis, max_doc).is_err());
    let mut flipped = out.clone();
    let at = flipped.tio.len() / 2;
    flipped.tio[at] ^= 0x40;
    let r = open(&flipped, &fis, max_doc).unwrap();
    assert!(r.check_integrity().is_err());
    // The postings block size after the postings header.
    let header = codec_util_header_len(TERMS_CODEC_NAME)
        + codec_util_header_len(super::reader::POSTINGS_TERMS_CODEC);
    let mut bad = out.clone();
    bad.tio[header] = 64;
    assert!(matches!(
        open(&bad, &fis, max_doc),
        Err(Error::Unsupported(_))
    ));
    // The index's version differs from the terms file's.
    let mut bad = out.clone();
    let at = 4 + 1 + TERMS_INDEX_CODEC_NAME.len() + 3;
    bad.tipo[at] = 2;
    assert!(open(&bad, &fis, max_doc).is_err());
}

/// An index header's length: magic, name, version, id, suffix.
fn codec_util_header_len(name: &str) -> usize {
    4 + 1 + name.len() + 4 + ID.len() + 1 + SUFFIX.len()
}

#[test]
fn helpers_refuse_short_input() {
    let mut pos = 0;
    // SENTINEL-OK: this `read_byte` returns a `Result`, not a `-1`.
    assert!(read_byte(&[], &mut pos).is_err());
    assert!(length(-1).is_err());
    assert_eq!(ushr32(-2, 1), i32::MAX);
    assert_eq!(ushr64(-2, 1), i64::MAX);
    let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
    // entCount 0.
    assert!(read_block(&[0, 0, 0, 0], 0, &mut a, &mut b, &mut c).is_err());
    // A suffix blob past the end.
    assert!(read_block(&[2, 40, 1], 0, &mut a, &mut b, &mut c).is_err());
    assert!(read_block(&[2], -1, &mut a, &mut b, &mut c).is_err());
}

/// One `.tio` field-directory entry and its `.tipo` index start.
#[derive(Clone)]
struct DirEntry {
    field: i32,
    num_terms: i64,
    root_code: Vec<u8>,
    sum_total_term_freq: Option<i64>,
    sum_doc_freq: i64,
    doc_count: i32,
    min_term: Vec<u8>,
    max_term: Vec<u8>,
    index_start_fp: i64,
}

/// The directory start of a file this writer wrote (before its footer).
fn dir_start(file: &[u8]) -> usize {
    use lucene_store::data_input::{DataInput, SliceInput};
    let at = file.len() - lucene_store::codec_util::FOOTER_LENGTH - 8;
    SliceInput::new(&file[at..]).read_i64().unwrap() as usize
}

/// `out`'s field directory, decoded as the writer laid it out.
fn read_dir(out: &BlockTreeOrdsOutput, fis: &FieldInfos) -> Vec<DirEntry> {
    use lucene_store::data_input::{DataInput, SliceInput};
    let mut d = SliceInput::new(&out.tio[dir_start(&out.tio)..]);
    let mut x = SliceInput::new(&out.tipo[dir_start(&out.tipo)..]);
    let bytes = |d: &mut SliceInput<'_>| {
        let mut b = vec![0; d.read_vint().unwrap() as usize];
        d.read_bytes(&mut b).unwrap();
        b
    };
    (0..d.read_vint().unwrap())
        .map(|_| {
            let field = d.read_vint().unwrap();
            let num_terms = d.read_vlong().unwrap();
            let root_code = bytes(&mut d);
            let docs = fis.field_by_number(field).unwrap().index_options == IndexOptions::Docs;
            let sum_total_term_freq = (!docs).then(|| d.read_vlong().unwrap());
            DirEntry {
                field,
                num_terms,
                root_code,
                sum_total_term_freq,
                sum_doc_freq: d.read_vlong().unwrap(),
                doc_count: d.read_vint().unwrap(),
                min_term: bytes(&mut d),
                max_term: bytes(&mut d),
                index_start_fp: x.read_vlong().unwrap(),
            }
        })
        .collect()
}

/// `out` with its directories replaced by `entries` (`num_fields` entries
/// announced) and fresh footers.
fn with_dir(
    out: &BlockTreeOrdsOutput,
    num_fields: i32,
    entries: &[DirEntry],
) -> BlockTreeOrdsOutput {
    use lucene_store::data_output::DataOutput;
    let (tio_dir, tipo_dir) = (dir_start(&out.tio), dir_start(&out.tipo));
    let mut tio = out.tio[..tio_dir].to_vec();
    let mut tipo = out.tipo[..tipo_dir].to_vec();
    tio.write_vint(num_fields);
    for e in entries {
        tio.write_vint(e.field);
        tio.write_vlong(e.num_terms);
        tio.write_vint(e.root_code.len() as i32);
        tio.write_bytes(&e.root_code);
        if let Some(t) = e.sum_total_term_freq {
            tio.write_vlong(t);
        }
        tio.write_vlong(e.sum_doc_freq);
        tio.write_vint(e.doc_count);
        tio.write_vint(e.min_term.len() as i32);
        tio.write_bytes(&e.min_term);
        tio.write_vint(e.max_term.len() as i32);
        tio.write_bytes(&e.max_term);
        tipo.write_vlong(e.index_start_fp);
    }
    tio.write_i64(tio_dir as i64);
    lucene_store::codec_util::write_footer(&mut tio);
    tipo.write_i64(tipo_dir as i64);
    lucene_store::codec_util::write_footer(&mut tipo);
    BlockTreeOrdsOutput {
        tio,
        tipo,
        ..out.clone()
    }
}

#[test]
fn reader_refuses_bad_field_directories() {
    let (out, fis, max_doc) = write(BlockTreeOrdsPostingsFormat::default(), &corpus(), true);
    let dir = read_dir(&out, &fis);
    assert_eq!(dir.len(), 2);
    // The directory rewritten as is opens: the helpers are faithful.
    assert!(open(&with_dir(&out, 2, &dir), &fis, max_doc).is_ok());
    let refused = |num_fields: i32, entries: &[DirEntry]| {
        open(&with_dir(&out, num_fields, entries), &fis, max_doc)
            .unwrap_err()
            .to_string()
    };
    assert!(refused(-1, &[]).contains("invalid numFields: -1"));
    // The same field twice.
    let twice = [dir[0].clone(), dir[0].clone()];
    assert!(refused(2, &twice).contains("duplicate field: f"));
    // sumDocFreq under docCount (`g` has frequencies, so both are stored).
    let mut g = dir[1].clone();
    g.sum_doc_freq = 0;
    g.sum_total_term_freq = Some(0);
    assert!(refused(1, &[g]).contains("invalid sumDocFreq"));
    // sumTotalTermFreq under sumDocFreq.
    let mut g = dir[1].clone();
    g.sum_total_term_freq = Some(g.sum_doc_freq - 1);
    assert!(refused(1, &[g]).contains("invalid sumTotalTermFreq"));
    // An index start past the end of the `.tipo`.
    let mut f = dir[0].clone();
    f.index_start_fp = 1 << 40;
    assert!(refused(1, &[f]).contains("invalid indexStartFP"));
    // A min term announced longer than what is left of the file.
    let mut f = dir[0].clone();
    f.min_term = Vec::new();
    let mut bad = with_dir(&out, 1, &[f]);
    let at = dir_start(&bad.tio);
    // field, numTerms, root code, sumDocFreq, docCount: then minTerm's
    // length, rewritten in place as a large vint.
    let mut d = lucene_store::data_input::SliceInput::new(&bad.tio[at..]);
    {
        use lucene_store::data_input::DataInput;
        d.read_vint().unwrap();
        d.read_vint().unwrap();
        d.read_vlong().unwrap();
        let n = d.read_vint().unwrap() as usize;
        d.seek(d.position() + n).unwrap();
        d.read_vlong().unwrap();
        d.read_vint().unwrap();
    }
    let len_at = at + d.position();
    assert_eq!(bad.tio[len_at], 0);
    bad.tio[len_at] = 0x7f;
    let state = SegmentReadState {
        max_doc,
        segment_id: &ID,
        segment_suffix: SUFFIX,
        field_infos: &fis,
    };
    let err = BlockTreeOrdsPostingsFormat::default()
        .open(bad.tio.clone(), &bad.tipo, &state)
        .unwrap_err();
    assert!(err.to_string().contains("past the end"), "{err}");
}
