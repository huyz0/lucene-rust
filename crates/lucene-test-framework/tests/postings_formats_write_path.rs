//! The test framework's terms-dictionary postings formats through
//! [`IndexWriter`]: `LuceneFixedGap`, `LuceneVarGapFixedInterval`,
//! `LuceneVarGapDocFreqInterval`, `UniformSplitRot13`, `STUniformSplitRot13`
//! `TestBloomFilteredLucenePostings` and `MockRandom` (seeds covering its
//! dictionaries). Fields routed to one are flushed as
//! `_<Format>_0.<dictionary>` beside their postings, a buffered delete by a
//! routed field's term finds its documents, a merge reads and rewrites the
//! group, CheckIndex passes, and [`DirectoryReader`] answers term and phrase
//! queries over it exactly as over the same documents on the default format.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::per_field_postings::Lucene104PostingsFormat;
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_codecs::terms_dictionary::TermsDictionaryFormat;
use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::{search_phrase_query, search_term_query, PhraseQuery, TermQuery, VecCollector};
use lucene_store::{Directory, FsDirectory};
use lucene_test_framework::codecs::bloom::TestBloomFilteredLucenePostings;
use lucene_test_framework::codecs::mockrandom::MockRandomPostingsFormat;
use lucene_test_framework::codecs::rot13::{
    StUniformSplitRot13PostingsFormat, UniformSplitRot13PostingsFormat,
};
use lucene_test_framework::codecs::{blockterms, register_postings_formats};
use lucene_util::java_random::JavaRandom;
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

fn write(dir: &FsDirectory, routed: Option<Lucene104PostingsFormat>, merge: bool) {
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
    if let Some(format) = routed {
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

/// Every query's live hits, as their documents' sorted stored ids: a merge
/// may order its sources by size, which differs between formats.
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

/// Returns the routed index's file names.
fn check(
    routed: Lucene104PostingsFormat,
    dictionary: &dyn TermsDictionaryFormat,
    merge: bool,
) -> Vec<String> {
    let default_path = TempDir::new("tf-dictionary-write-path-default");
    let ords_path = TempDir::new("tf-dictionary-write-path-routed");
    let (default_dir, ords_dir) = (
        FsDirectory::open(&default_path),
        FsDirectory::open(&ords_path),
    );
    write(&default_dir, None, merge);
    write(&ords_dir, Some(routed), merge);

    let files = ords_dir.list_all().unwrap();
    let name = dictionary.name();
    // The marker: a format may leave its other dictionary files unwritten
    // (MockRandom's seed picks its dictionary).
    let marker = dictionary.extensions()[0];
    for ext in [marker, "doc", "pos", "psm"] {
        assert!(
            files
                .iter()
                .any(|f| f.ends_with(&format!("_{name}_0.{ext}"))),
            "no {name} .{ext} in {files:?}"
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
    files
}

fn formats() -> Vec<&'static dyn TermsDictionaryFormat> {
    register_postings_formats();
    vec![
        &blockterms::FIXED_GAP,
        &blockterms::VAR_GAP_FIXED_INTERVAL,
        &blockterms::VAR_GAP_DOC_FREQ_INTERVAL,
        UniformSplitRot13PostingsFormat::default()
            .dictionary_format()
            .unwrap(),
        StUniformSplitRot13PostingsFormat
            .dictionary_format()
            .unwrap(),
        TestBloomFilteredLucenePostings::dictionary_format(),
    ]
}

/// `MockRandomPostingsFormat(new Random(seed))` for seeds whose flushes and
/// merge between them select every dictionary kind.
fn mock_randoms() -> Vec<&'static dyn TermsDictionaryFormat> {
    register_postings_formats();
    (0..6)
        .map(|seed| {
            MockRandomPostingsFormat::new(&mut JavaRandom::new(seed)).into_dictionary_format()
        })
        .collect()
}

fn check_all(merge: bool) {
    for dictionary in formats() {
        check(
            Lucene104PostingsFormat::dictionary(dictionary),
            dictionary,
            merge,
        );
    }
    let mut written = Vec::new();
    for dictionary in mock_randoms() {
        written.extend(check(
            Lucene104PostingsFormat::dictionary(dictionary),
            dictionary,
            merge,
        ));
    }
    // Flushed segments keep every seed's choice; a merge reads them all as
    // its sources and leaves only its own.
    let kinds: &[&str] = if merge {
        &[]
    } else {
        &["tfp", "tim", "tii", "tmv", "tio"]
    };
    for ext in kinds {
        assert!(
            written
                .iter()
                .any(|f| f.ends_with(&format!("_MockRandom_0.{ext}"))),
            "no MockRandom seed chose .{ext}: {written:?}"
        );
    }
}

#[test]
fn flushed_segments_read_back_like_the_default_format() {
    check_all(false);
}

#[test]
fn a_merge_rewrites_the_group_and_reads_back() {
    check_all(true);
}
