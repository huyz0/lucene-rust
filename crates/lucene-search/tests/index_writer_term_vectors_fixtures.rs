//! Required end-to-end proof that a document added through this port's own
//! `IndexWriter::add_document` + `IndexWriter::commit()` -- not a hand-built
//! fixture -- ends up with genuinely readable term vectors, once
//! `IndexWriter::set_term_vector_field` has opted a field into real term
//! vectors (`crates/lucene-index/src/index_writer.rs`'s new wiring of
//! `lucene_codecs::term_vectors::write_best_speed` into `commit()`).
//!
//! Lives in `lucene-search` (not `lucene-index`) for the same reason
//! `index_writer_postings_fixtures.rs` does: `lucene-index` must not depend
//! on `lucene-search` (strictly downward dependency graph, see the
//! `architecture` skill), but `lucene-search` already depends on
//! `lucene-index`, so this is the natural home for an
//! IndexWriter-then-read round trip -- and lets this test go through
//! `lucene_search::term_vector_for_doc`, the existing query-facing read API
//! over decoded term vectors, rather than the raw codec reader directly.
//!
//! Scope proven here matches `term_vectors.rs::write_best_speed`'s own
//! documented scope exactly: one field opted into term vectors at a time,
//! positions only (no offsets/payloads), single chunk.

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_codecs::term_vectors::{self as tv};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::{self, LuceneVersion};
use lucene_search::term_vectors_query::term_vector_for_doc;
use lucene_store::directory::{Directory, FsDirectory};

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
    }
}

fn field_info(
    number: i32,
    name: &str,
    index_options: IndexOptions,
    store_term_vectors: bool,
) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: Vec::new(),
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Float32,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

fn doc(id: &str, body: &str) -> Document {
    Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String(body.to_string()),
            },
        ],
    }
}

fn tempdir(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "lucene-rust-index-writer-tv-fixture-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// The critical end-to-end proof this task requires: documents added via
/// `IndexWriter::add_document` (not a hand-built fixture), flushed by a real
/// `IndexWriter::commit()` with `set_term_vector_field(Some("body"))`
/// configured, opened back through the existing, unmodified
/// `term_vectors::open`/`TermVectorsReader::document` read side, and queried
/// via the existing, unmodified `lucene_search::term_vector_for_doc` --
/// asserting the exact per-document term/freq/position data a real
/// `IndexReader.getTermVector` would return.
#[test]
fn documents_added_via_index_writer_have_readable_term_vectors() {
    let tmp = tempdir("readable");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        field_info(0, "id", IndexOptions::None, false),
        field_info(1, "body", IndexOptions::DocsAndFreqs, true),
    ];
    let mut writer = IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();

    writer.add_document(doc("a", "the quick fox"));
    writer.add_document(doc("b", "the lazy fox"));
    writer.add_document(doc("c", "the fox runs"));
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1);
    let sci = &sis.segments[0];

    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    for ext in ["tvd", "tvx", "tvm"] {
        let name = format!("{}.{ext}", sci.segment_name);
        assert!(si.files.contains(&name), "missing {name} in .si files");
        assert!(
            dir.list_all().unwrap().contains(&name),
            "missing {name} on disk"
        );
    }

    let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
    let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
    let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
    let reader = tv::open(&tvd, &tvx, &tvm, &sci.segment_id, "")
        .expect("term_vectors::open on IndexWriter-produced .tvd/.tvx/.tvm");
    assert_eq!(reader.max_doc(), 3);

    let field_infos = lucene_codecs::field_infos::FieldInfos {
        fields: fields.clone(),
    };

    let field0 = term_vector_for_doc(&reader, &field_infos, 0, "body")
        .unwrap()
        .expect("doc 0 has a term vector for 'body'");
    assert!(field0.has_positions);
    assert!(!field0.has_offsets);
    let mut terms0: Vec<(String, i32)> = field0
        .terms
        .iter()
        .map(|t| (String::from_utf8(t.term.clone()).unwrap(), t.freq))
        .collect();
    terms0.sort();
    assert_eq!(
        terms0,
        vec![
            ("fox".to_string(), 1),
            ("quick".to_string(), 1),
            ("the".to_string(), 1),
        ]
    );
    let the_term = field0.terms.iter().find(|t| t.term == b"the").unwrap();
    assert_eq!(the_term.positions, Some(vec![0]));

    let field1 = term_vector_for_doc(&reader, &field_infos, 1, "body")
        .unwrap()
        .expect("doc 1 has a term vector for 'body'");
    let mut terms1: Vec<String> = field1
        .terms
        .iter()
        .map(|t| String::from_utf8(t.term.clone()).unwrap())
        .collect();
    terms1.sort();
    assert_eq!(terms1, vec!["fox", "lazy", "the"]);

    // A field never opted into term vectors ("id") has none.
    assert!(term_vector_for_doc(&reader, &field_infos, 0, "id")
        .unwrap()
        .is_none());
}

/// Backward compatibility: a writer that never calls
/// `set_term_vector_field` must produce exactly the same on-disk shape as
/// before this feature existed -- no `.tvd`/`.tvx`/`.tvm` files at all.
#[test]
fn commit_with_no_term_vector_field_configured_writes_no_term_vector_files() {
    let tmp = tempdir("no-tv-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        field_info(0, "id", IndexOptions::None, false),
        field_info(1, "body", IndexOptions::DocsAndFreqs, true),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc("a", "the quick fox"));
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let files = dir.list_all().unwrap();
    for ext in ["tvd", "tvx", "tvm"] {
        assert!(!files.contains(&format!("{}.{ext}", sci.segment_name)));
    }
}

/// A field opted into both real postings (`set_postings_field`) and real
/// term vectors (`set_term_vector_field`) in the same commit -- real
/// Lucene's ordinary case of a field being both indexed and having term
/// vectors stored. Both write-side passes must coexist correctly in one
/// `commit()`: the segment's `.si` must list all seven files
/// (`.doc`/`.tim`/`.tip`/`.tmd`/`.tvd`/`.tvx`/`.tvm`), and both are
/// independently readable back with the expected content.
#[test]
fn a_field_with_both_postings_and_term_vectors_configured_together_produces_both_correctly() {
    let tmp = tempdir("postings-and-tv");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        field_info(0, "id", IndexOptions::None, false),
        field_info(1, "body", IndexOptions::DocsAndFreqs, true),
    ];
    let mut writer = IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();

    writer.add_document(doc("a", "the quick fox"));
    writer.add_document(doc("b", "the lazy fox"));
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1);
    let sci = &sis.segments[0];

    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    for ext in ["doc", "tim", "tip", "tmd", "tvd", "tvx", "tvm"] {
        let name = format!("{}.{ext}", sci.segment_name);
        assert!(si.files.contains(&name), "missing {name} in .si files");
        assert!(
            dir.list_all().unwrap().contains(&name),
            "missing {name} on disk"
        );
    }

    // Postings side: readable via blocktree exactly as before.
    let tim = dir.open(&format!("{}.tim", sci.segment_name)).unwrap();
    let tip = dir.open(&format!("{}.tip", sci.segment_name)).unwrap();
    let tmd = dir.open(&format!("{}.tmd", sci.segment_name)).unwrap();
    let field_infos_struct = lucene_codecs::field_infos::FieldInfos {
        fields: fields.clone(),
    };
    let block_fields = lucene_codecs::blocktree::open(
        &tim,
        &tip,
        &tmd,
        &field_infos_struct,
        &sci.segment_id,
        "",
        2,
    )
    .expect("blocktree::open on IndexWriter-produced .tim/.tip/.tmd");
    let field = block_fields.field("body").unwrap();
    let doc_bytes = dir.open(&format!("{}.doc", sci.segment_name)).unwrap();
    let doc_in = lucene_codecs::postings::DocInput::open(&doc_bytes, &sci.segment_id, "")
        .expect("open .doc");
    let postings = field.postings(b"fox", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(postings.docs, vec![0, 1]);

    // Term-vector side: readable via term_vectors::open.
    let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
    let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
    let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
    let reader = tv::open(&tvd, &tvx, &tvm, &sci.segment_id, "")
        .expect("term_vectors::open on IndexWriter-produced .tvd/.tvx/.tvm");
    let field0 = term_vector_for_doc(&reader, &field_infos_struct, 0, "body")
        .unwrap()
        .expect("doc 0 has a term vector for 'body'");
    let mut terms0: Vec<String> = field0
        .terms
        .iter()
        .map(|t| String::from_utf8(t.term.clone()).unwrap())
        .collect();
    terms0.sort();
    assert_eq!(terms0, vec!["fox", "quick", "the"]);
}

/// Multi-field term vectors in one commit: `IndexWriter::set_term_vector_field`
/// together with `IndexWriter::add_term_vector_field` opts in `title` and
/// `body` at once, so a single `term_vectors::write_best_speed` call (which
/// already accepts multiple `TermVectorField` entries per
/// `TermVectorsDocument`) must produce one `.tvd`/`.tvx`/`.tvm` file set where
/// each doc's term vector correctly carries both fields' independent term
/// data -- not just one, and not a mix of the two.
#[test]
fn two_distinct_term_vector_fields_in_one_commit_are_both_readable() {
    let tmp = tempdir("two-tv-fields");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        field_info(0, "id", IndexOptions::None, false),
        field_info(1, "title", IndexOptions::DocsAndFreqs, true),
        field_info(2, "body", IndexOptions::DocsAndFreqs, true),
    ];
    let mut writer = IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();
    writer.set_term_vector_field(Some("title")).unwrap();
    writer.add_term_vector_field("body").unwrap();

    let two_field_doc = |id: &str, title: &str, body: &str| Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String(title.to_string()),
            },
            StoredField {
                field_number: 2,
                value: FieldValue::String(body.to_string()),
            },
        ],
    };

    writer.add_document(two_field_doc("a", "space exploration", "the quick fox"));
    writer.add_document(two_field_doc("b", "deep sea diving", "the lazy fox"));
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1);
    let sci = &sis.segments[0];

    let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
    let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
    let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
    let reader = tv::open(&tvd, &tvx, &tvm, &sci.segment_id, "")
        .expect("term_vectors::open on IndexWriter-produced multi-field .tvd/.tvx/.tvm");
    assert_eq!(reader.max_doc(), 2);

    let field_infos = lucene_codecs::field_infos::FieldInfos {
        fields: fields.clone(),
    };

    let sorted_terms = |field: &lucene_codecs::term_vectors::TermVectorField| {
        let mut terms: Vec<String> = field
            .terms
            .iter()
            .map(|t| String::from_utf8(t.term.clone()).unwrap())
            .collect();
        terms.sort();
        terms
    };

    let doc0_title = term_vector_for_doc(&reader, &field_infos, 0, "title")
        .unwrap()
        .expect("doc 0 has a term vector for 'title'");
    assert_eq!(sorted_terms(&doc0_title), vec!["exploration", "space"]);
    let doc0_body = term_vector_for_doc(&reader, &field_infos, 0, "body")
        .unwrap()
        .expect("doc 0 has a term vector for 'body'");
    assert_eq!(sorted_terms(&doc0_body), vec!["fox", "quick", "the"]);

    let doc1_title = term_vector_for_doc(&reader, &field_infos, 1, "title")
        .unwrap()
        .expect("doc 1 has a term vector for 'title'");
    assert_eq!(sorted_terms(&doc1_title), vec!["deep", "diving", "sea"]);
    let doc1_body = term_vector_for_doc(&reader, &field_infos, 1, "body")
        .unwrap()
        .expect("doc 1 has a term vector for 'body'");
    assert_eq!(sorted_terms(&doc1_body), vec!["fox", "lazy", "the"]);

    // Cross-contamination check: `title`'s vocabulary must never leak into
    // `body`'s term vector or vice versa.
    assert!(!doc0_body
        .terms
        .iter()
        .any(|t| t.term == b"space" || t.term == b"exploration"));
    assert!(!doc0_title
        .terms
        .iter()
        .any(|t| t.term == b"fox" || t.term == b"quick"));
}
