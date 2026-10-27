//! The test framework's six `CompressingCodec`s installed in
//! [`lucene_index::index_writer::IndexWriter`] (`CompressingCodec::installed`):
//! for each of the fixture's cases, flushed and merged, every segment records
//! the codec's name and no `Lucene90StoredFieldsFormat.mode`, its stored
//! fields and term vectors sit under the codec's segment suffix with its data
//! codec name, and every stored field and term vector reads back, deleted
//! documents included until a merge drops them. `VerifyCompressingCodecs`
//! (verify-write-path) has Lucene write the same documents with the same
//! codecs and compares the stored-fields bytes.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

#[path = "../examples/write_compressing_codec_fixture.rs"]
mod fixture;

use std::collections::BTreeSet;

use lucene_codecs::stored_fields::FieldValue;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

use fixture::{big, body, deleted, CASES, PER_SEGMENT, SEGMENTS};

const TOTAL: usize = PER_SEGMENT * SEGMENTS;

/// `body(i)`'s tokens, in term order, with their positions and offsets.
fn expected_vectors(i: usize) -> Vec<(String, i32, i32, i32)> {
    let text = body(i);
    let mut start = 0;
    let mut tokens: Vec<(String, i32, i32, i32)> = text
        .split(' ')
        .enumerate()
        .map(|(position, token)| {
            let token_start = start;
            start += token.len() + 1;
            (
                token.to_string(),
                position as i32,
                token_start as i32,
                (token_start + token.len()) as i32,
            )
        })
        .collect();
    tokens.sort();
    tokens
}

fn read_back(dir: &dyn Directory, merged: bool, suffix: &str, data_codec: &str) {
    let reader = DirectoryReader::open(dir).unwrap();
    let mut live = BTreeSet::new();
    let mut seen = 0;
    for segment in reader.segment_readers() {
        let tv = segment
            .term_vectors_reader()
            .unwrap()
            .expect("term vectors");
        assert_eq!(tv.data_codec(), data_codec);
        for doc in 0..segment.max_doc {
            let stored = segment.stored_document(doc).unwrap().unwrap();
            let values: Vec<&FieldValue> = stored.fields.iter().map(|f| &f.value).collect();
            let FieldValue::String(id) = values[0] else {
                panic!("id")
            };
            let i: usize = id[1..].parse().unwrap();
            assert_eq!(values[1], &FieldValue::String(body(i)));
            assert_eq!(values[2], &FieldValue::Int(i as i32));
            assert_eq!(
                values.get(3).copied(),
                big(i).map(FieldValue::String).as_ref()
            );
            let vectors = tv.document(doc).unwrap().expect("a term vector");
            let terms: Vec<(String, i32, i32, i32)> = vectors.fields[0]
                .terms
                .iter()
                .map(|t| {
                    assert_eq!(t.freq, 1);
                    (
                        String::from_utf8(t.term.clone()).unwrap(),
                        t.positions.as_ref().unwrap()[0],
                        t.start_offsets.as_ref().unwrap()[0],
                        t.end_offsets.as_ref().unwrap()[0],
                    )
                })
                .collect();
            assert_eq!(terms, expected_vectors(i), "d{i}");
            if segment
                .live_docs()
                .is_none_or(|bits| bits.get(doc as usize))
            {
                live.insert(i);
            }
            seen += 1;
        }
        for ext in ["fdt", "fdx", "fdm", "tvd", "tvx", "tvm"] {
            let name =
                lucene_index::index_writer::segment_file_name(&segment.segment_name, suffix, ext);
            assert!(dir.file_length(&name).is_ok(), "{name}");
        }
    }
    let expected: BTreeSet<usize> = (0..TOTAL).filter(|&i| !deleted(i)).collect();
    assert_eq!(live, expected);
    assert_eq!(seen, if merged { expected.len() } else { TOTAL });
}

#[test]
fn every_case_writes_its_format_and_reads_back_flushed_and_merged() {
    for case in &CASES {
        let codec = fixture::codec(case);
        let installed = codec.installed().unwrap();
        let format = installed.format();
        let tmp = TempDir::new("compressing-codec-write-path");
        for (sub, merge) in [("flushed", false), ("merged", true)] {
            let path = tmp.path().join(sub);
            std::fs::create_dir_all(&path).unwrap();
            let dir = FsDirectory::open(&path);
            fixture::write(&dir, case, merge);
            let infos = lucene_index::segment_infos::read_latest(&dir).unwrap();
            for sci in &infos.segments {
                assert_eq!(sci.codec_name, format.format_name, "{}", case.name);
            }
            read_back(&dir, merge, format.segment_suffix, format.format_name);
        }
    }
}
