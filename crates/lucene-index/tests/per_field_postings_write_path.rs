//! `PerFieldPostingsFormat` through [`IndexWriter`]: fields routed to a
//! second `Lucene104PostingsFormat` with
//! [`IndexWriter::set_postings_format_for_field`] get their own
//! `_Lucene104_1` files and `.fnm` suffix; a buffered delete by a routed
//! field's term finds its documents; a merge reads both formats of every
//! source and writes both again; and this port's `check_index` walks every
//! field of both. `scripts/verify-write-path.sh`'s `per-field-formats` case
//! has real Lucene read the same shape.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    self, DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::per_field_postings::{self, Lucene104PostingsFormat};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::check_index;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::{self, LuceneVersion};
use lucene_index::segment_infos;
use lucene_store::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

const PER_SEGMENT: usize = 300;
const SEGMENTS: usize = 3;

fn field(name: &str, number: i32, index_options: IndexOptions) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Byte,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

/// The same routing as an `IndexWriterConfig.setCodec` codec.
#[derive(Debug)]
struct Codec;

impl lucene_codecs::codec::Lucene104Codec for Codec {
    fn postings_format_for_field(&self, field: &str) -> Lucene104PostingsFormat {
        if field == "tag" || field == "key" {
            Lucene104PostingsFormat::new(10, 20).unwrap()
        } else {
            Lucene104PostingsFormat::default()
        }
    }
}

/// A codec of another name is refused: this writer writes `Lucene104`.
#[derive(Debug)]
struct Foreign;

impl lucene_codecs::codec::Lucene104Codec for Foreign {
    fn name(&self) -> &str {
        "Asserting"
    }
}

fn write(dir: &FsDirectory, merge: bool) {
    write_with(dir, merge, false);
}

fn write_with(dir: &FsDirectory, merge: bool, codec: bool) {
    let fields = vec![
        field("id", 0, IndexOptions::None),
        field("body", 1, IndexOptions::DocsAndFreqsAndPositions),
        field("tag", 2, IndexOptions::Docs),
        field("key", 3, IndexOptions::DocsAndFreqs),
    ];
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut writer = IndexWriter::open(dir, fields, "Lucene104", version).unwrap();
    for name in ["body", "tag", "key"] {
        writer.add_postings_field(name).unwrap();
    }
    assert!(writer
        .set_codec(Some(std::sync::Arc::new(Foreign)))
        .is_err());
    if codec {
        writer.set_codec(Some(std::sync::Arc::new(Codec))).unwrap();
    } else {
        let small = Lucene104PostingsFormat::new(10, 20).unwrap();
        writer.set_postings_format_for_field("tag", small);
        writer.set_postings_format_for_field("key", small);
        // Re-routing replaces, never duplicates.
        writer.set_postings_format_for_field("key", small);
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
                    text(1, format!("shared a{}", i % 13)),
                    text(2, format!("t{}", i % 7)),
                    text(3, format!("k{}", i % 50)),
                ],
            })
            .unwrap();
        if (i + 1) % PER_SEGMENT == 0 {
            if i + 1 == PER_SEGMENT * SEGMENTS {
                writer.set_use_compound_file(true);
            }
            writer.commit().unwrap();
        }
    }
    writer
        .delete_documents_by_term(&[Term {
            field: "tag".to_string(),
            bytes: b"t3".to_vec(),
        }])
        .unwrap();
    writer.commit().unwrap();
    if merge {
        writer.force_merge(1).unwrap();
        writer.commit().unwrap();
    }
}

fn expected_live() -> usize {
    (0..PER_SEGMENT * SEGMENTS).filter(|i| i % 7 != 3).count()
}

/// Every segment: two postings groups, the `.fnm` routing each field to its
/// own, the delete applied, and `check_index` clean with the postings checks
/// of every field actually run.
fn assert_index(path: &TempDir, segments: usize) {
    let dir = FsDirectory::open(path);
    let infos = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(infos.segments.len(), segments);
    let mut live = 0;
    for sci in &infos.segments {
        let si = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si, &sci.segment_id).unwrap();
        live += (si.doc_count - sci.del_count) as usize;
    }
    assert_eq!(live, expected_live());

    // A loose segment's files and `.fnm`.
    let loose = &infos.segments[0];
    let files: Vec<String> = dir.list_all().unwrap();
    let own: Vec<String> = files
        .iter()
        .filter(|f| {
            f.starts_with(&format!("{}_", loose.segment_name))
                || f.starts_with(&format!("{}.", loose.segment_name))
        })
        .cloned()
        .collect();
    assert_eq!(
        per_field_postings::group_suffixes(&own, &loose.segment_name),
        ["Lucene104_0", "Lucene104_1"]
    );
    let fnm = dir.open(&format!("{}.fnm", loose.segment_name)).unwrap();
    let fnm = field_infos::parse(&fnm, &loose.segment_id, "").unwrap();
    for (name, suffix) in [
        ("body", Some("0")),
        ("tag", Some("1")),
        ("key", Some("1")),
        ("id", None),
    ] {
        let f = fnm.fields.iter().find(|f| f.name == name).unwrap();
        let attr = f
            .attributes
            .iter()
            .find(|(k, _)| k == per_field_postings::PER_FIELD_SUFFIX_KEY)
            .map(|(_, v)| v.as_str());
        assert_eq!(attr, suffix, "{name}");
    }

    let results = check_index::check_directory(&dir).unwrap();
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
}

#[test]
fn flushed_segments_route_fields_to_their_postings_format() {
    let path = TempDir::new("per-field-flushed");
    write(&FsDirectory::open(&path), false);
    assert_index(&path, SEGMENTS);
}

#[test]
fn a_merge_reads_and_writes_both_postings_formats() {
    let path = TempDir::new("per-field-merged");
    write(&FsDirectory::open(&path), true);
    assert_index(&path, 1);
}

#[test]
fn a_codec_routes_postings_as_the_per_field_setters_do() {
    let path = TempDir::new("per-field-codec");
    write_with(&FsDirectory::open(&path), true, true);
    assert_index(&path, 1);
}

/// Ten documents carrying `tag:x` (routed to the second format) and, when
/// `body`, `body:x`; an eleventh carrying `body:y` only when `last_body`;
/// then `tag:x` deleted and the segment merged.
fn write_pruned(dir: &FsDirectory, last_body: bool) {
    let fields = vec![
        field("id", 0, IndexOptions::None),
        field("body", 1, IndexOptions::DocsAndFreqsAndPositions),
        field("tag", 2, IndexOptions::Docs),
    ];
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut writer = IndexWriter::open(dir, fields, "Lucene104", version).unwrap();
    for name in ["body", "tag"] {
        writer.add_postings_field(name).unwrap();
    }
    writer.set_postings_format_for_field("tag", Lucene104PostingsFormat::new(10, 20).unwrap());
    let text = |field_number: i32, s: &str| StoredField {
        field_number,
        value: FieldValue::String(s.to_string()),
    };
    for i in 0..10 {
        writer
            .add_document(Document {
                fields: vec![text(0, &format!("doc{i}")), text(1, "x"), text(2, "x")],
            })
            .unwrap();
    }
    let mut last = vec![text(0, "doc10")];
    if last_body {
        last.push(text(1, "y"));
    }
    writer.add_document(Document { fields: last }).unwrap();
    writer
        .delete_documents_by_term(&[Term {
            field: "tag".to_string(),
            bytes: b"x".to_vec(),
        }])
        .unwrap();
    writer.commit().unwrap();
    writer.force_merge(1).unwrap();
    writer.commit().unwrap();
}

/// The merged segment still claims both fields indexed, routes each to its
/// format, writes each format's files -- the pruned ones with no field in
/// them -- and passes `check_index`.
fn assert_pruned(path: &TempDir, groups_with_terms: &[&str]) {
    let dir = FsDirectory::open(path);
    let infos = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(infos.segments.len(), 1);
    let sci = &infos.segments[0];
    let files = dir.list_all().unwrap();
    for suffix in ["Lucene104_0", "Lucene104_1"] {
        for ext in ["doc", "psm", "tim", "tip", "tmd"] {
            let name = format!("{}_{suffix}.{ext}", sci.segment_name);
            assert!(files.contains(&name), "{name} missing: {files:?}");
        }
    }
    let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
    let fis = field_infos::parse(&fnm, &sci.segment_id, "").unwrap();
    for name in ["body", "tag"] {
        let f = fis.fields.iter().find(|f| f.name == name).unwrap();
        assert_ne!(f.index_options, IndexOptions::None, "{name}");
        assert!(
            f.attributes
                .iter()
                .any(|(k, _)| k == per_field_postings::PER_FIELD_SUFFIX_KEY),
            "{name} lost its routing"
        );
    }
    let results = check_index::check_directory(&dir).unwrap();
    for result in &results {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
    let names: Vec<&str> = results
        .iter()
        .flat_map(|r| r.checks.iter())
        .map(|c| c.name.as_str())
        .collect();
    for field in ["body", "tag"] {
        let wanted = format!("postings.field_summary:{field}");
        assert_eq!(
            names.contains(&wanted.as_str()),
            groups_with_terms.contains(&field),
            "{wanted}"
        );
    }
}

/// `BasePointsFormatTestCase.testAllPointDocsDeletedInSegment`'s shape: a
/// merge whose every term of a field belonged to deleted documents. Java's
/// `PerFieldPostingsFormat` still routes the field (its sources list it) and
/// writes the group's files with no field in them; this port dropped the
/// field, leaving a `.fnm` claiming postings with no `.tim`.
#[test]
fn a_merge_keeps_a_field_whose_every_term_was_deleted() {
    let path = TempDir::new("per-field-pruned");
    write_pruned(&FsDirectory::open(&path), true);
    assert_pruned(&path, &["body"]);
}

/// The same with no live term in any field: both groups write empty files.
#[test]
fn a_merge_writes_empty_groups_when_every_term_was_deleted() {
    let path = TempDir::new("per-field-all-pruned");
    write_pruned(&FsDirectory::open(&path), false);
    assert_pruned(&path, &[]);
}
