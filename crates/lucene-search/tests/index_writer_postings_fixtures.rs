//! Required end-to-end proof that a document added through this port's own
//! `IndexWriter::add_document` + `IndexWriter::commit()` -- not a hand-built
//! fixture -- is genuinely searchable by term query, once
//! `IndexWriter::set_postings_field` has opted a field into real postings
//! (`crates/lucene-index/src/index_writer.rs`'s new wiring of
//! `lucene_codecs::postings_writer::write_single_field` into `commit()`).
//!
//! Lives in `lucene-search` (not `lucene-index`) for the same reason
//! `postings_writer_round_trip.rs` does: `lucene-index` must not depend on
//! `lucene-search` (strictly downward dependency graph, see the
//! `architecture` skill), but `lucene-search` already depends on
//! `lucene-index`, so this is the natural home for an
//! IndexWriter-then-query round trip.
//!
//! Scope proven here matches `postings_writer.rs`'s own documented scope
//! exactly: one field indexed with postings at a time, one `.tim` block per
//! commit (`docFreq < 256`), term-frequency only (no positions/phrase
//! queries). This does **not** prove multi-field or multi-block indexing --
//! neither exists yet.

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::postings::DocInput;
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::{self, LuceneVersion};
use lucene_search::{search_term_query, TermQuery, VecCollector};
use lucene_store::directory::{Directory, FsDirectory};

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
    }
}

fn field_info(number: i32, name: &str, index_options: IndexOptions) -> FieldInfo {
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
        "lucene-rust-index-writer-postings-fixture-{tag}-{}-{}",
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
/// `IndexWriter::commit()`, opened back through the existing, unmodified
/// `blocktree::open`/`postings::DocInput` read side, and queried via the
/// existing, unmodified `lucene_search::search_term_query` -- asserting the
/// exact doc IDs a real `IndexSearcher` would return for several distinct
/// terms across multiple documents.
#[test]
fn documents_added_via_index_writer_are_searchable_by_term_query() {
    let tmp = tempdir("searchable");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        field_info(0, "id", IndexOptions::None),
        field_info(1, "body", IndexOptions::DocsAndFreqs),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();

    writer.add_document(doc("a", "the quick fox jumps"));
    writer.add_document(doc("b", "the lazy fox sleeps"));
    writer.add_document(doc("c", "the fox and the hound"));
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1);
    let sci = &sis.segments[0];

    let tim = dir.open(&format!("{}.tim", sci.segment_name)).unwrap();
    let tip = dir.open(&format!("{}.tip", sci.segment_name)).unwrap();
    let tmd = dir.open(&format!("{}.tmd", sci.segment_name)).unwrap();
    let doc_bytes = dir.open(&format!("{}.doc", sci.segment_name)).unwrap();
    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();

    let field_infos = lucene_codecs::field_infos::FieldInfos {
        fields: vec![
            field_info(0, "id", IndexOptions::None),
            field_info(1, "body", IndexOptions::DocsAndFreqs),
        ],
    };
    let fields = lucene_codecs::blocktree::open(
        &tim,
        &tip,
        &tmd,
        &field_infos,
        &sci.segment_id,
        "",
        si.doc_count,
    )
    .expect("blocktree::open on IndexWriter-produced .tim/.tip/.tmd");
    let doc_in = DocInput::open(&doc_bytes, &sci.segment_id, "").expect("open .doc");

    let case = |term: &str, expected: &[i32]| {
        let mut collector = VecCollector::default();
        let query = TermQuery::new("body", term.as_bytes());
        search_term_query(&fields, Some(&doc_in), None, &query, &mut collector)
            .unwrap_or_else(|e| panic!("search_term_query({term:?}) failed: {e}"));
        assert_eq!(collector.docs, expected, "term {term:?}");
    };

    // "the" occurs in all 3 docs (doc 2 twice, but term-freq isn't asserted
    // here -- that's already covered at the encoder level in
    // `postings_writer.rs`'s own tests).
    case("the", &[0, 1, 2]);
    // "fox" also occurs in all 3.
    case("fox", &[0, 1, 2]);
    // Singleton terms (docFreq == 1), one per doc.
    case("quick", &[0]);
    case("lazy", &[1]);
    case("hound", &[2]);
    // A term that was never indexed at all.
    case("nonexistent", &[]);
}

/// The documented `docFreq >= 256` boundary as seen from `IndexWriter`
/// itself: a term shared by 256 pending docs must make `commit()` fail
/// rather than silently write a wrong/truncated segment -- this writer has
/// no multi-block `.tim` support (see `postings_writer.rs`'s own scope note).
#[test]
fn commit_rejects_a_term_at_the_256_doc_freq_boundary() {
    let tmp = tempdir("docfreq-boundary");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        field_info(0, "id", IndexOptions::None),
        field_info(1, "body", IndexOptions::DocsAndFreqs),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();

    for i in 0..256 {
        writer.add_document(doc(&i.to_string(), "shared"));
    }
    let result = writer.commit();
    assert!(
        result.is_err(),
        "expected commit() to reject a docFreq >= 256 term rather than \
         silently write wrong/truncated postings"
    );
}
