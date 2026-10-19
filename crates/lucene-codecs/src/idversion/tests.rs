//! Unit tests of the idversion port: the checks and messages the
//! Java-fixture tests (`tests/sandbox_idversion_fixtures.rs`) never reach.

// Test-support arithmetic over small in-memory values; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::blocktree::SeekStatus;
use crate::field_infos::IndexOptions;
use crate::fst::{InputType, Pair};
use crate::fst_compiler::FstCompilerBuilder;
use crate::postings_writer::TermPostings;
use lucene_store::codec_util::{self, FOOTER_LENGTH, ID_LENGTH};
use lucene_store::data_output::DataOutput;

const ID: [u8; ID_LENGTH] = [7; ID_LENGTH];

fn posting(term: &str, docs: &[(i32, i64)]) -> TermPostings {
    TermPostings {
        term: term.as_bytes().to_vec(),
        docs: docs.iter().map(|&(d, _)| (d, 1)).collect(),
        positions: docs.iter().map(|_| vec![0]).collect(),
        offsets: vec![],
        payload_bytes: docs.iter().flat_map(|&(_, v)| v.to_be_bytes()).collect(),
        payload_lengths: docs.iter().map(|_| 8).collect(),
    }
}

fn input(terms: &[TermPostings]) -> IdVersionFieldInput<'_, TermPostings> {
    IdVersionFieldInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        store_term_vectors: false,
        has_payloads: true,
        terms,
    }
}

fn write(terms: &[TermPostings]) -> Result<IdVersionOutput, Error> {
    write_fields(
        &[input(terms)],
        None,
        &IdVersionPostingsFormat::default(),
        &ID,
        "IDVersion_0",
    )
}

fn message(r: Result<IdVersionOutput, Error>) -> String {
    match r {
        Err(Error::IllegalArgument(m)) => m,
        other => panic!("expected IllegalArgument, got {other:?}"),
    }
}

fn names(n: i32) -> Option<String> {
    (n == 0).then(|| "id".to_string())
}

fn open(out: &IdVersionOutput) -> Result<VersionBlockTreeTermsReader, Error> {
    VersionBlockTreeTermsReader::open(out.tiv.clone(), &out.tipv, &ID, "IDVersion_0", 100, &names)
}

/// Many terms with shared prefixes, so blocks nest and floor.
fn many() -> Vec<TermPostings> {
    let mut terms = Vec::new();
    let mut doc = 0;
    for a in b'a'..=b'c' {
        for b in b'a'..=b'h' {
            for c in ["", "x", "y"] {
                let t = format!("{}{}{c}", a as char, b as char);
                terms.push(posting(&t, &[(doc, i64::from(doc) * 3)]));
                doc += 1;
            }
        }
    }
    terms
}

fn open_many() -> (IdVersionOutput, usize) {
    let terms = many();
    let n = terms.len();
    let out = write_fields(
        &[input(&terms)],
        None,
        &IdVersionPostingsFormat::new(2, 3).unwrap(),
        &ID,
        "IDVersion_0",
    )
    .unwrap();
    (out, n)
}

#[test]
fn the_format_validates_its_block_sizes() {
    let f = IdVersionPostingsFormat::default();
    assert_eq!((f.min_terms_in_block(), f.max_terms_in_block()), (25, 48));
    assert!(IdVersionPostingsFormat::new(1, 3).is_err());
    assert!(IdVersionPostingsFormat::new(10, 12).is_err());
    assert_eq!(
        IdVersionPostingsFormat::new(4, 9)
            .unwrap()
            .max_terms_in_block(),
        9
    );
}

#[test]
fn versions_convert_to_and_from_payloads() {
    assert_eq!(
        bytes_to_long(&long_to_bytes(MAX_VERSION).unwrap()),
        MAX_VERSION
    );
    assert_eq!(long_to_bytes(1).unwrap(), [0, 0, 0, 0, 0, 0, 0, 1]);
    for bad in [-1, MAX_VERSION + 1] {
        let Err(Error::IllegalArgument(m)) = long_to_bytes(bad) else {
            panic!("{bad} accepted");
        };
        assert!(
            m.starts_with("version must be >= MIN_VERSION=0 and <= MAX_VERSION="),
            "{m}"
        );
    }
}

#[test]
fn the_single_enums_hold_one_document() {
    let mut d = SingleDocsEnum::new(5);
    assert_eq!(
        (d.doc_id(), d.cost(), d.freq(), d.next_position()),
        (-1, 1, 1, -1)
    );
    assert_eq!(d.next_doc(), 5);
    assert_eq!(d.next_doc(), NO_MORE_DOCS);
    let mut d = SingleDocsEnum::new(5);
    assert_eq!(d.advance(5), 5);
    assert_eq!(d.advance(5), NO_MORE_DOCS);
    assert_eq!(SingleDocsEnum::new(5).advance(6), NO_MORE_DOCS);

    let mut p = SinglePostingsEnum::new(3, 9);
    assert_eq!((p.doc_id(), p.cost(), p.freq()), (-1, 1, 1));
    assert_eq!((p.start_offset(), p.end_offset()), (-1, -1));
    assert_eq!(p.next_doc(), 3);
    assert_eq!(p.next_position(), 0);
    assert_eq!(bytes_to_long(&p.payload()), 9);
    assert_eq!(p.next_doc(), NO_MORE_DOCS);
    let mut p = SinglePostingsEnum::new(3, 9);
    assert_eq!(p.advance(2), 3);
    assert_eq!(p.advance(4), NO_MORE_DOCS);
}

#[test]
fn the_writer_refuses_fields_and_postings_as_java_does() {
    let ok = [posting("a", &[(0, 1)])];
    let mut i = input(&ok);
    i.index_options = IndexOptions::DocsAndFreqs;
    let r = write_fields(&[i], None, &IdVersionPostingsFormat::default(), &ID, "");
    assert_eq!(
        message(r),
        "field must be index using IndexOptions.DOCS_AND_FREQS_AND_POSITIONS"
    );
    let mut i = input(&ok);
    i.store_term_vectors = true;
    let r = write_fields(&[i], None, &IdVersionPostingsFormat::default(), &ID, "");
    assert_eq!(
        message(r),
        "field cannot index term vectors: CheckIndex will report this as index corruption"
    );
    let mut i = input(&ok);
    i.has_payloads = false;
    let r = write_fields(&[i], None, &IdVersionPostingsFormat::default(), &ID, "");
    assert_eq!(message(r), "token doens't have a payload");

    assert_eq!(
        message(write(&[posting("a", &[(0, 1), (3, 2)])])),
        "term appears in more than one document: 0 and 3"
    );
    let mut twice = posting("a", &[(0, 1)]);
    twice.docs[0].1 = 2;
    assert_eq!(
        message(write(&[twice])),
        "term appears more than once in the document"
    );
    let mut short = posting("a", &[(0, 1)]);
    short.payload_lengths = vec![4];
    short.payload_bytes.truncate(4);
    assert_eq!(message(write(&[short])), "payload.length != 8 (got 4)");
    let mut cut = posting("a", &[(0, 1)]);
    cut.payload_bytes.truncate(4);
    assert_eq!(message(write(&[cut])), "payload run is short");
    let mut none = posting("a", &[(0, 1)]);
    none.positions = vec![vec![]];
    assert_eq!(message(write(&[none])), "missing addPosition");
    assert_eq!(
        message(write(&[posting("a", &[(0, -1)])])),
        "version must be >= MIN_VERSION=0 (got: -1; payload=[ff ff ff ff ff ff ff ff])"
    );
    assert_eq!(
        message(write(&[posting("a", &[(0, MAX_VERSION + 1)])])),
        format!(
            "version must be <= MAX_VERSION={MAX_VERSION} (got: {}; payload=[40 0 0 0 0 0 0 0])",
            MAX_VERSION + 1
        )
    );
}

#[test]
fn a_term_whose_documents_are_all_deleted_is_not_written() {
    let terms = [posting("a", &[(0, 1)]), posting("b", &[(1, 2), (2, 3)])];
    let live = |d: i32| d != 1 && d != 0;
    let out = write_fields(
        &[input(&terms)],
        Some(&live),
        &IdVersionPostingsFormat::default(),
        &ID,
        "IDVersion_0",
    )
    .unwrap();
    let reader = open(&out).unwrap();
    let fr = reader.terms("id").unwrap();
    assert_eq!((fr.size(), fr.min(), fr.max()), (1, &b"b"[..], &b"b"[..]));
    let mut te = reader.iterator("id").unwrap();
    assert_eq!(te.next().unwrap(), Some(&b"b"[..]));
    assert_eq!(te.term_state().unwrap().id_version, 3);

    // A field with no live term is left out of the directory.
    let all_dead = |_: i32| false;
    let out = write_fields(
        &[input(&terms)],
        Some(&all_dead),
        &IdVersionPostingsFormat::default(),
        &ID,
        "IDVersion_0",
    )
    .unwrap();
    let reader = open(&out).unwrap();
    assert_eq!(reader.size(), 0);
    assert!(reader.iterator("id").is_none());
}

#[test]
fn the_reader_exposes_the_field_and_its_statistics() {
    let (out, n) = open_many();
    let reader = open(&out).unwrap();
    reader.check_integrity().unwrap();
    let fr = reader.terms("id").unwrap();
    assert_eq!(fr.size(), n as i64);
    assert_eq!(fr.sum_doc_freq(), n as i64);
    assert_eq!(fr.sum_total_term_freq(), n as i64);
    assert_eq!(fr.doc_count(), n as i32);
    assert_eq!((fr.min(), fr.max()), (&b"aa"[..], &b"chy"[..]));
    assert_eq!(fr.field_number(), 0);
    assert!(fr.index_start_fp() > 0);
    assert!(fr.root_block_fp() > 0);
    assert!(!fr.root_code().first.is_empty());
    assert!(reader.terms("other").is_none());
}

#[test]
fn seeks_and_walks_agree_with_the_terms_written() {
    let (out, _) = open_many();
    let reader = open(&out).unwrap();
    let all: Vec<Vec<u8>> = many().into_iter().map(|t| t.term).collect();
    let mut te = reader.iterator("id").unwrap();
    for t in &all {
        assert_eq!(te.next().unwrap(), Some(&t[..]));
    }
    assert_eq!(te.next().unwrap(), None);

    for (i, t) in all.iter().enumerate() {
        let mut te = reader.iterator("id").unwrap();
        let v = i as i64 * 3;
        assert!(te.try_seek_exact(t).unwrap());
        assert_eq!(te.version(), v);
        assert!(te.seek_exact_with_version(t, v).unwrap());
        assert!(!te.seek_exact_with_version(t, v + 1).unwrap());
        assert_eq!(te.try_seek_ceil(t).unwrap(), SeekStatus::Found);
        let mut more = t.clone();
        more.push(b'0');
        let status = te.try_seek_ceil(&more).unwrap();
        match all.get(i + 1) {
            Some(next) => {
                assert_eq!(status, SeekStatus::NotFound);
                assert_eq!(te.term(), &next[..]);
            }
            None => assert_eq!(status, SeekStatus::End),
        }
    }
    let mut te = reader.iterator("id").unwrap();
    assert!(!te.try_seek_exact(b"zz").unwrap());
    assert!(!te.try_seek_exact(b"ab0").unwrap());
    assert!(!te.try_seek_exact(b"").unwrap());
    assert_eq!(te.try_seek_ceil(b"").unwrap(), SeekStatus::NotFound);
    assert_eq!(te.term(), b"aa");
    assert_eq!(te.try_seek_ceil(b"zz").unwrap(), SeekStatus::End);
}

#[test]
fn a_term_state_repositions_the_enum() {
    let (out, _) = open_many();
    let reader = open(&out).unwrap();
    let mut te = reader.iterator("id").unwrap();
    assert!(te.try_seek_exact(b"bcx").unwrap());
    let state = te.term_state().unwrap();
    assert_eq!((state.doc_freq, state.total_term_freq), (1, 1));
    let mut other = reader.iterator("id").unwrap();
    other.seek_exact_state(b"bcx", state);
    assert_eq!(other.term(), b"bcx");
    let IdVersionPostings::Docs(mut docs) = other.postings(false).unwrap() else {
        panic!("docs only were asked for");
    };
    assert_eq!(docs.next_doc(), state.doc_id);
    // Already there: a no-op.
    te.seek_exact_state(b"bcx", state);
    assert_eq!(te.version(), state.id_version);
    // `next` from a state seeks to it first -- on an enum that has read a
    // block; a fresh one (as `other`) starts from the first term, as in Java.
    let mut third = reader.iterator("id").unwrap();
    assert!(third.try_seek_exact(b"aby").unwrap());
    let aby = third.term_state().unwrap();
    te.seek_exact_state(b"aby", aby);
    assert_eq!(te.next().unwrap(), Some(&b"ac"[..]));
    assert_eq!(other.next().unwrap(), Some(&b"aa"[..]));
}

#[test]
fn the_reader_refuses_corrupt_files_as_java_does() {
    let out = write(&[posting("a", &[(0, 1)])]).unwrap();
    let err = |r: Result<VersionBlockTreeTermsReader, Error>| match r {
        Err(e) => e.to_string(),
        Ok(_) => panic!("opened"),
    };
    // Too many terms for the segment.
    let r = VersionBlockTreeTermsReader::open(
        out.tiv.clone(),
        &out.tipv,
        &ID,
        "IDVersion_0",
        0,
        &names,
    );
    assert!(err(r).contains("invalid docCount: 1 maxDoc: 0"));
    // An unknown field number.
    let r = VersionBlockTreeTermsReader::open(
        out.tiv.clone(),
        &out.tipv,
        &ID,
        "IDVersion_0",
        10,
        &|_| None,
    );
    assert!(err(r).contains("invalid field number: 0"));
    // Two field numbers with one name.
    let terms = [posting("a", &[(0, 1)])];
    let mut second = input(&terms);
    second.field_number = 1;
    let two = write_fields(
        &[input(&terms), second],
        None,
        &IdVersionPostingsFormat::default(),
        &ID,
        "IDVersion_0",
    )
    .unwrap();
    let r = VersionBlockTreeTermsReader::open(
        two.tiv.clone(),
        &two.tipv,
        &ID,
        "IDVersion_0",
        10,
        &|_| Some("id".to_string()),
    );
    assert!(err(r).contains("duplicate field: id"));
    // A wrong suffix, and a broken index checksum.
    let r = VersionBlockTreeTermsReader::open(out.tiv.clone(), &out.tipv, &ID, "x", 10, &names);
    assert!(err(r).contains("suffix"));
    let mut tipv = out.tipv.clone();
    let at = tipv.len() - 20;
    tipv[at] ^= 1;
    assert!(open(&IdVersionOutput {
        tiv: out.tiv.clone(),
        tipv
    })
    .is_err());
    // A broken `.tiv` checksum shows on `checkIntegrity`.
    let mut tiv = out.tiv.clone();
    let at = tiv.len() - 30;
    tiv[at] ^= 1;
    let reader = VersionBlockTreeTermsReader::open(tiv, &out.tipv, &ID, "IDVersion_0", 10, &names);
    if let Ok(reader) = reader {
        assert!(reader.check_integrity().is_err());
    }
    // A short file.
    assert!(open(&IdVersionOutput {
        tiv: out.tiv[..10].to_vec(),
        tipv: out.tipv.clone()
    })
    .is_err());
}

/// A `.tiv` of the two headers and `dir` as its field directory.
fn tiv_with_dir(postings_codec: &str, dir: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    codec_util::write_index_header(&mut out, TERMS_CODEC_NAME, 1, &ID, "IDVersion_0");
    codec_util::write_index_header(&mut out, postings_codec, 1, &ID, "IDVersion_0");
    let start = out.len() as i64;
    out.extend_from_slice(dir);
    out.extend_from_slice(&start.to_le_bytes());
    codec_util::write_footer(&mut out);
    out
}

/// A `.tipv` of its header, `body`, then `dir` as its field directory.
fn tipv_with(body: &[u8], dir: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    codec_util::write_index_header(&mut out, TERMS_INDEX_CODEC_NAME, 1, &ID, "IDVersion_0");
    out.extend_from_slice(body);
    let start = out.len() as i64;
    out.extend_from_slice(dir);
    out.extend_from_slice(&start.to_le_bytes());
    codec_util::write_footer(&mut out);
    out
}

fn open_files(tiv: Vec<u8>, tipv: &[u8]) -> String {
    match VersionBlockTreeTermsReader::open(tiv, tipv, &ID, "IDVersion_0", 10, &names) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("opened"),
    }
}

/// One field record: number 0, one term, `code` as the root code, min and
/// max `a`.
fn one_field(code_len: i32) -> Vec<u8> {
    let mut dir = Vec::new();
    dir.write_vint(1);
    dir.write_vint(0);
    dir.write_vlong(1);
    dir.write_vint(code_len);
    dir.write_vlong(4 << OUTPUT_FLAGS_NUM_BITS);
    dir.write_vlong(0);
    for _ in 0..2 {
        dir.write_vint(1);
        dir.push(b'a');
    }
    dir
}

#[test]
fn a_corrupt_field_directory_is_an_error() {
    let empty_tipv = tipv_with(&[], &[]);
    let mut dir = Vec::new();
    dir.write_vint(-1);
    let tiv = tiv_with_dir(POSTINGS_TERMS_CODEC, &dir);
    assert!(open_files(tiv, &empty_tipv).contains("invalid numFields: -1"));
    let tiv = tiv_with_dir("NotIDVersion", &[0]);
    assert!(open_files(tiv, &empty_tipv).contains("NotIDVersion"));
    let tiv = tiv_with_dir(POSTINGS_TERMS_CODEC, &one_field(100));
    assert!(open_files(tiv, &empty_tipv).contains("length 100 past the end"));

    // A terms index without the empty input's output.
    let mut compiler = FstCompilerBuilder::new(InputType::Byte1).build::<IndexOutputs>();
    compiler
        .add_bytes(
            b"a",
            Pair {
                first: vec![1],
                second: 1,
            },
        )
        .unwrap();
    let fst = compiler.compile().unwrap().save::<IndexOutputs>();
    let header_len = tipv_with(&[], &[]).len() - 8 - FOOTER_LENGTH;
    let mut index_dir = Vec::new();
    index_dir.write_vlong(header_len as i64);
    let tiv = tiv_with_dir(POSTINGS_TERMS_CODEC, &one_field(1));
    let tipv = tipv_with(&fst, &index_dir);
    assert!(open_files(tiv.clone(), &tipv).contains("terms index has no root output"));
    let mut index_dir = Vec::new();
    index_dir.write_vlong(1 << 40);
    let tipv = tipv_with(&fst, &index_dir);
    assert!(open_files(tiv, &tipv).contains("invalid indexStartFP"));
}

#[test]
fn a_corrupt_block_is_an_error_not_a_panic() {
    let out = write(&[posting("a", &[(0, 1)])]).unwrap();
    let fp = {
        let reader = open(&out).unwrap();
        reader.terms("id").unwrap().root_block_fp() as usize
    };
    // The entry's suffix runs past the block's suffix bytes.
    let mut tiv = out.tiv.clone();
    tiv[fp + 2] = 5;
    let reader =
        VersionBlockTreeTermsReader::open(tiv, &out.tipv, &ID, "IDVersion_0", 10, &names).unwrap();
    let mut te = reader.iterator("id").unwrap();
    assert!(te
        .next()
        .unwrap_err()
        .to_string()
        .contains("suffix past the end"));
    // The block's suffix bytes run past the file.
    let mut tiv = out.tiv.clone();
    tiv[fp + 1] = 0xfe;
    tiv[fp + 2] = 0x7f;
    let reader =
        VersionBlockTreeTermsReader::open(tiv, &out.tipv, &ID, "IDVersion_0", 10, &names).unwrap();
    let mut te = reader.iterator("id").unwrap();
    assert!(te
        .try_seek_exact(b"a")
        .unwrap_err()
        .to_string()
        .contains("past the end"));
}

/// A field record with the given number, term count and root-code length
/// (the code's bytes are `code_len` zeros).
fn field_record(field: i32, num_terms: i64, code_len: i32) -> Vec<u8> {
    let mut dir = Vec::new();
    dir.write_vint(field);
    dir.write_vlong(num_terms);
    dir.write_vint(code_len);
    dir.extend(std::iter::repeat_n(
        0u8,
        usize::try_from(code_len).unwrap_or(0),
    ));
    dir.write_vlong(0);
    dir
}

#[test]
fn a_field_record_is_checked_as_java_checks_it() {
    let empty_tipv = tipv_with(&[], &[]);
    let with = |records: &[Vec<u8>]| {
        let mut dir = Vec::new();
        dir.write_vint(records.len() as i32);
        for r in records {
            dir.extend_from_slice(r);
        }
        tiv_with_dir(POSTINGS_TERMS_CODEC, &dir)
    };
    assert!(
        open_files(with(&[field_record(7, 1, 1)]), &empty_tipv).contains("invalid field number: 7")
    );
    assert!(open_files(with(&[field_record(0, 1, -2)]), &empty_tipv).contains("negative length -2"));
    let mut too_many = field_record(0, 11, 1);
    for _ in 0..2 {
        too_many.write_vint(1);
        too_many.push(b'a');
    }
    assert!(open_files(with(&[too_many]), &empty_tipv).contains("invalid docCount: 11 maxDoc: 10"));

    // A directory offset past the directory itself, and a file too short to
    // hold one.
    let mut tiv = tiv_with_dir(POSTINGS_TERMS_CODEC, &[0]);
    let at = tiv.len() - FOOTER_LENGTH - 8;
    tiv[at..at + 8].copy_from_slice(&(1i64 << 40).to_le_bytes());
    assert!(open_files(tiv, &empty_tipv).contains("invalid directory offset"));
    let short = [0u8; 4];
    let mut input = lucene_store::data_input::SliceInput::new(&short);
    let err = super::reader::seek_dir(&mut input, short.len()).unwrap_err();
    assert!(err.to_string().contains("file too short"), "{err}");
}

#[test]
fn the_reader_lists_its_fields() {
    let (out, _) = open_many();
    let reader = open(&out).unwrap();
    assert_eq!(reader.field_names().collect::<Vec<_>>(), vec!["id"]);
    assert_eq!(reader.size(), 1);
}
