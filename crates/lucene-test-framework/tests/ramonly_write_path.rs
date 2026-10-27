//! `RAMOnlyPostingsFormat` through [`IndexWriter`]: a field routed to
//! [`RamOnlyPostingsFormat::spi`] is flushed as only `_<seg>_RAMOnly_0.id`
//! (no `.doc`/`.pos`/`.psm` of its own), its postings kept in the format's
//! slot; buffered deletes by a routed term find their documents, a merge
//! reads every source slot and writes a new one, CheckIndex passes over the
//! re-encoded group, and [`DirectoryReader`] answers term and phrase queries
//! over it exactly as over the same documents on the default format. A
//! docs-only or offsets field routed to it fails, as Java's does.
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
use lucene_test_framework::codecs::ramonly::RamOnlyPostingsFormat;
use lucene_test_framework::codecs::register_postings_formats;
use lucene_util::test_support::TempDir;

const PER_SEGMENT: usize = 150;
const SEGMENTS: usize = 3;

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    }
}

fn ram_only() -> Lucene104PostingsFormat {
    register_postings_formats();
    Lucene104PostingsFormat::dictionary(RamOnlyPostingsFormat::spi())
}

fn writer(dir: &FsDirectory, tag_options: IndexOptions) -> IndexWriter<'_> {
    let fields = vec![
        FieldInfo::new("id", 0),
        FieldInfo::new("body", 1).with_index_options(IndexOptions::DocsAndFreqsAndPositions),
        FieldInfo::new("tag", 2).with_index_options(tag_options),
    ];
    let mut writer = IndexWriter::open(dir, fields, "Lucene104", version()).unwrap();
    for name in ["body", "tag"] {
        writer.add_postings_field(name).unwrap();
    }
    writer
}

fn doc(i: usize) -> Document {
    let text = |field_number: i32, s: String| StoredField {
        field_number,
        value: FieldValue::String(s),
    };
    // A supplementary and a high-BMP letter order differently in UTF-16
    // (RAMOnly's map) and UTF-8 (the block tree).
    let odd = if i.is_multiple_of(11) {
        " \u{1D400}x \u{FB00}y"
    } else {
        ""
    };
    Document {
        fields: vec![
            text(0, format!("doc{i}")),
            text(1, format!("shared a{} b{} a{}{odd}", i % 13, i % 29, i % 5)),
            text(2, format!("t{}", i % 7)),
        ],
    }
}

fn write(dir: &FsDirectory, routed: bool, merge: bool) {
    let mut writer = writer(dir, IndexOptions::Docs);
    if routed {
        writer.set_postings_format_for_field("body", ram_only());
    }
    for i in 0..PER_SEGMENT * SEGMENTS {
        writer.add_document(doc(i)).unwrap();
        if (i + 1) % PER_SEGMENT == 0 {
            writer.commit().unwrap();
        }
    }
    writer
        .delete_documents_by_term(&[Term::new("tag", "t3"), Term::new("body", "b28")])
        .unwrap();
    writer.commit().unwrap();
    if merge {
        writer.force_merge(1).unwrap();
        writer.commit().unwrap();
    }
}

/// Every query's live hits, as their documents' sorted stored ids.
fn hits(dir: &FsDirectory) -> Vec<(String, Vec<String>)> {
    let reader = DirectoryReader::open(dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let readers = reader.segment_readers();
    let id = |index: usize, doc: i32| {
        let stored = readers[index].stored_document(doc).unwrap().unwrap();
        let FieldValue::String(id) = &stored.fields[0].value else {
            panic!("an id");
        };
        id.clone()
    };
    let mut out = Vec::new();
    let mut terms: Vec<(&str, String)> = vec![
        ("body", "shared".into()),
        ("body", "zzz".into()),
        ("body", "\u{1D400}x".into()),
        ("body", "\u{FB00}y".into()),
    ];
    terms.extend((0..13).map(|i| ("body", format!("a{i}"))));
    terms.extend((0..29).map(|i| ("body", format!("b{i}"))));
    terms.extend((0..7).map(|i| ("tag", format!("t{i}"))));
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
            docs.extend(collector.docs.iter().map(|&d| id(index, d)));
        }
        docs.sort();
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
            docs.extend(collector.docs.iter().map(|&d| id(index, d)));
        }
        docs.sort();
        out.push((format!("body:\"{words:?}\"~{slop}"), docs));
    }
    out
}

fn check(merge: bool) {
    let default_path = TempDir::new("ramonly-write-path-default");
    let routed_path = TempDir::new("ramonly-write-path-routed");
    let (default_dir, routed_dir) = (
        FsDirectory::open(&default_path),
        FsDirectory::open(&routed_path),
    );
    write(&default_dir, false, merge);
    write(&routed_dir, true, merge);

    let files = routed_dir.list_all().unwrap();
    let ids: Vec<&String> = files
        .iter()
        .filter(|f| f.ends_with("_RAMOnly_0.id"))
        .collect();
    assert_eq!(ids.len(), if merge { 1 } else { SEGMENTS }, "{files:?}");
    for ext in ["doc", "pos", "pay", "psm", "tim", "tip", "tmd"] {
        assert!(
            !files
                .iter()
                .any(|f| f.ends_with(&format!("_RAMOnly_0.{ext}"))),
            "a RAMOnly .{ext} in {files:?}"
        );
    }
    assert!(files.iter().any(|f| f.ends_with("_Lucene104_0.tmd")));

    let results = lucene_index::check_index::check_directory(&routed_dir).unwrap();
    for result in &results {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
    let names: Vec<&str> = results
        .iter()
        .flat_map(|r| r.checks.iter())
        .map(|c| c.name.as_str())
        .collect();
    assert!(names.contains(&"postings.field_summary:body"));

    let (expected, actual) = (hits(&default_dir), hits(&routed_dir));
    assert_eq!(expected, actual);
    for deleted in ["tag:t3", "body:b28"] {
        let hits = &actual.iter().find(|(q, _)| q == deleted).unwrap().1;
        assert!(hits.is_empty(), "{deleted}: {hits:?}");
    }
    for odd in ["body:\u{1D400}x", "body:\u{FB00}y"] {
        let hits = &actual.iter().find(|(q, _)| q == odd).unwrap().1;
        assert!(!hits.is_empty(), "{odd}");
    }
}

#[test]
fn flushed_segments_read_back_like_the_default_format() {
    check(false);
}

#[test]
fn a_merge_reads_every_slot_and_writes_its_own() {
    check(true);
}

#[test]
fn docs_only_and_offsets_fields_are_refused() {
    for options in [
        IndexOptions::Docs,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
    ] {
        let path = TempDir::new("ramonly-write-path-refused");
        let dir = FsDirectory::open(&path);
        let mut writer = writer(&dir, options);
        writer.set_postings_format_for_field("tag", ram_only());
        writer.add_document(doc(0)).unwrap();
        assert!(writer.commit().is_err(), "{options:?}");
    }
}
