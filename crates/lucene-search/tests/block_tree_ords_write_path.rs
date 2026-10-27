//! `BlockTreeOrdsPostingsFormat` through [`IndexWriter`]: fields routed to it
//! are flushed as `_BlockTreeOrds_0.tio`/`.tipo` beside their postings, a
//! buffered delete by a routed field's term finds its documents, a merge
//! reads and rewrites the group, and [`DirectoryReader`] answers term and
//! phrase queries over it exactly as over the same documents on the default
//! format.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::per_field_postings::Lucene104PostingsFormat;
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::{search_phrase_query, search_term_query, PhraseQuery, TermQuery, VecCollector};
use lucene_store::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

const PER_SEGMENT: usize = 200;
const SEGMENTS: usize = 3;

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    }
}

fn write(dir: &FsDirectory, ords: bool, merge: bool) {
    let fields = vec![
        FieldInfo::new("id", 0),
        FieldInfo::new("body", 1).with_index_options(IndexOptions::DocsAndFreqsAndPositions),
        FieldInfo::new("tag", 2).with_index_options(IndexOptions::Docs),
        FieldInfo::new("key", 3).with_index_options(IndexOptions::DocsAndFreqs),
    ];
    let mut writer = IndexWriter::open(dir, fields, "Lucene104", version()).unwrap();
    for name in ["body", "tag", "key"] {
        writer.add_postings_field(name).unwrap();
    }
    if ords {
        // Small blocks, so the dictionary has floor and sub-blocks.
        let format = Lucene104PostingsFormat::block_tree_ords(4, 8).unwrap();
        writer.set_postings_format_for_field("body", format);
        writer.set_postings_format_for_field("tag", format);
    }
    for i in 0..PER_SEGMENT * SEGMENTS {
        let text = |field_number: i32, s: String| StoredField {
            field_number,
            value: FieldValue::String(s),
        };
        writer
            .add_document(Document {
                fields: vec![
                    text(0, format!("doc{i}")),
                    text(1, format!("shared a{} b{} a{}", i % 13, i % 29, i % 5)),
                    text(2, format!("t{}", i % 7)),
                    text(3, format!("k{}", i % 50)),
                ],
            })
            .unwrap();
        if (i + 1) % PER_SEGMENT == 0 {
            writer.commit().unwrap();
        }
    }
    writer
        .delete_documents_by_term(&[Term::new("tag", "t3")])
        .unwrap();
    writer.commit().unwrap();
    if merge {
        writer.force_merge(1).unwrap();
        writer.commit().unwrap();
    }
}

/// Every query's live hits, as (segment, doc) pairs.
fn hits(dir: &FsDirectory) -> Vec<(String, Vec<(usize, i32)>)> {
    let reader = DirectoryReader::open(dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let mut out = Vec::new();
    let mut terms: Vec<(&str, String)> = vec![("body", "shared".into()), ("body", "zzz".into())];
    terms.extend((0..13).map(|i| ("body", format!("a{i}"))));
    terms.extend((0..29).map(|i| ("body", format!("b{i}"))));
    terms.extend((0..7).map(|i| ("tag", format!("t{i}"))));
    terms.extend((0..50).map(|i| ("key", format!("k{i}"))));
    for (field, term) in &terms {
        let mut docs = Vec::new();
        for (index, seg) in opened.as_open_segments().iter().enumerate() {
            let mut collector = VecCollector::default();
            search_term_query(
                seg.fields,
                seg.doc_in,
                seg.live_docs,
                &TermQuery::new(*field, term.as_bytes()),
                &mut collector,
            )
            .unwrap();
            docs.extend(collector.docs.iter().map(|&d| (index, d)));
        }
        out.push((format!("{field}:{term}"), docs));
    }
    for (words, slop) in [(["shared", "a3"], 0), (["a2", "b7"], 0), (["a1", "a1"], 2)] {
        let mut docs = Vec::new();
        for (index, seg) in opened.as_open_segments().iter().enumerate() {
            let mut collector = VecCollector::default();
            search_phrase_query(
                seg.fields,
                seg.doc_in,
                seg.pos_in,
                seg.pay_in,
                seg.live_docs,
                &PhraseQuery::new("body", words.iter().copied()).with_slop(slop),
                &mut collector,
            )
            .unwrap();
            docs.extend(collector.docs.iter().map(|&d| (index, d)));
        }
        out.push((format!("body:\"{words:?}\"~{slop}"), docs));
    }
    out
}

fn check(merge: bool) {
    let default_path = TempDir::new("ords-write-path-default");
    let ords_path = TempDir::new("ords-write-path-ords");
    let (default_dir, ords_dir) = (
        FsDirectory::open(&default_path),
        FsDirectory::open(&ords_path),
    );
    write(&default_dir, false, merge);
    write(&ords_dir, true, merge);

    let files = ords_dir.list_all().unwrap();
    for ext in ["tio", "tipo", "doc", "pos", "psm"] {
        assert!(
            files
                .iter()
                .any(|f| f.ends_with(&format!("_BlockTreeOrds_0.{ext}"))),
            "no BlockTreeOrds .{ext} in {files:?}"
        );
    }
    assert!(files.iter().any(|f| f.ends_with("_Lucene104_0.tmd")));
    let results = lucene_index::check_index::check_directory(&ords_dir).unwrap();
    for result in &results {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
    let names: Vec<&str> = results
        .iter()
        .flat_map(|r| r.checks.iter())
        .map(|c| c.name.as_str())
        .collect();
    for field in ["body", "tag", "key"] {
        let wanted = format!("postings.field_summary:{field}");
        assert!(names.contains(&wanted.as_str()), "{wanted} did not run");
    }

    let (expected, actual) = (hits(&default_dir), hits(&ords_dir));
    assert_eq!(expected.len(), actual.len());
    for ((query, want), (_, got)) in expected.iter().zip(&actual) {
        assert_eq!(want, got, "{query}");
    }
    // The delete reached the routed field's documents.
    let deleted = actual.iter().find(|(q, _)| q == "tag:t3").unwrap();
    assert!(deleted.1.is_empty(), "{deleted:?}");
    assert!(!actual
        .iter()
        .find(|(q, _)| q == "tag:t4")
        .unwrap()
        .1
        .is_empty());
}

#[test]
fn flushed_segments_read_back_like_the_default_format() {
    check(false);
}

#[test]
fn a_merge_rewrites_the_group_and_reads_back() {
    check(true);
}
