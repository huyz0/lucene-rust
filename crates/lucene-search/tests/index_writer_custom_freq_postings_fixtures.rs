//! Required end-to-end proof for task #212: documents added through
//! `IndexWriter::add_document_with_custom_freq_terms` (not analyzed text) and
//! flushed by a real `IndexWriter::commit()` produce real
//! `IndexOptions::DocsAndCustomFreqs` postings whose freq value is exactly the
//! caller's explicit `custom_freq` -- not a derived term-occurrence count --
//! and that this is genuinely what BM25 scoring sees via
//! `lucene_search::search_term_query_scored`.
//!
//! Lives in `lucene-search` (not `lucene-index`) for the same reason
//! `index_writer_postings_fixtures.rs` does: `lucene-index` must not depend
//! on `lucene-search` (strictly downward dependency graph, see the
//! `architecture` skill), but `lucene-search` already depends on
//! `lucene-index`, so this is the natural home for an
//! IndexWriter-then-query round trip.
//!
//! The critical, deliberately-engineered proof point: each doc supplies
//! exactly one occurrence of the term "score" in its custom-freq term list
//! (so a literal occurrence count would always read back as `1` for every
//! doc), yet each doc's `custom_freq` value is different (5, 50, 1) --
//! forcing scores to differ in a way only the explicit custom freq value can
//! explain, never a fallback to counting occurrences.

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::postings::DocInput;
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::{self, LuceneVersion};
use lucene_search::collector::TopDocsCollector;
use lucene_search::{search_term_query_scored, similarity, TermQuery};
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

fn doc(id: &str) -> Document {
    Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(id.to_string()),
        }],
    }
}

fn tempdir(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "lucene-rust-index-writer-custom-freq-postings-fixture-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn custom_freq_postings_round_trip_and_drive_real_bm25_scoring() {
    let tmp = tempdir("custom-freq-scored");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        field_info(0, "id", IndexOptions::None),
        field_info(1, "score", IndexOptions::DocsAndCustomFreqs),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer
        .set_custom_freq_postings_field(Some("score"))
        .unwrap();

    // Every doc supplies exactly ONE occurrence of the term "score" in its
    // custom-freq term list -- a literal occurrence count would read back as
    // `1` for every single doc here. The custom_freq values below (5, 50, 1)
    // are chosen to be different from that occurrence count (except doc 2,
    // included to prove the "coincidentally equal to 1" case round-trips
    // too), so any score difference across docs can only be explained by the
    // real custom_freq value actually reaching the scorer, not a silent
    // fallback to counting.
    writer.add_document_with_custom_freq_terms(doc("a"), vec![("score".to_string(), 5)]);
    writer.add_document_with_custom_freq_terms(doc("b"), vec![("score".to_string(), 50)]);
    writer.add_document_with_custom_freq_terms(doc("c"), vec![("score".to_string(), 1)]);
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
            field_info(1, "score", IndexOptions::DocsAndCustomFreqs),
        ],
    };
    let opened_fields = lucene_codecs::blocktree::open(
        &tim,
        &tip,
        &tmd,
        &field_infos,
        &sci.segment_id,
        "",
        si.doc_count,
    )
    .expect("blocktree::open on IndexWriter-produced DocsAndCustomFreqs .tim/.tip/.tmd");
    let doc_in = DocInput::open(&doc_bytes, &sci.segment_id, "").expect("open .doc");

    // First, the raw postings-level proof: the freq value read back for each
    // doc is exactly its custom_freq, not `1` (the real occurrence count of
    // "score" in every doc's term list).
    let field = opened_fields.field("score").unwrap();
    let postings = field.postings(b"score", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(postings.docs, vec![0, 1, 2]);
    assert_eq!(
        postings.freqs,
        vec![5, 50, 1],
        "freqs must be the explicit custom_freq values, not the literal \
         occurrence count (which would be [1, 1, 1] for every doc here)"
    );

    // Second, the scoring-level proof: run a real TermQuery through
    // `search_term_query_scored` (this port's BM25 scorer) with no norms
    // opened (`UNNORMED_FIELD_LENGTH` for both field_length/avg_field_length,
    // matching every existing `DocsAndFreqs` scored-query test in this
    // crate), and confirm every doc's score matches the BM25 formula
    // evaluated directly against its expected custom_freq -- and that no two
    // docs get the same score, since their custom_freq values differ.
    let query = TermQuery::new("score", b"score");
    let mut collector = TopDocsCollector::new(3);
    search_term_query_scored(
        &opened_fields,
        Some(&doc_in),
        None,
        &query,
        None,
        &mut collector,
    )
    .unwrap();
    let hits = collector.top_docs();
    assert_eq!(hits.len(), 3);

    let doc_freq = 3i64; // "score" occurs in all 3 docs
    let doc_count = 3i64;
    let expected_score = |custom_freq: i32| {
        similarity::score(
            doc_freq,
            doc_count,
            custom_freq as f32,
            similarity::UNNORMED_FIELD_LENGTH,
            similarity::UNNORMED_FIELD_LENGTH,
        )
    };
    let expected: std::collections::HashMap<i32, f32> = [
        (0, expected_score(5)),
        (1, expected_score(50)),
        (2, expected_score(1)),
    ]
    .into_iter()
    .collect();

    for hit in hits {
        let want = expected[&hit.doc_id];
        assert!(
            (hit.score - want).abs() < 1e-6,
            "doc {} scored {} but expected {want} from its own custom_freq \
             (proves scoring used the explicit custom_freq value, not a \
             fallback occurrence count of 1 for every doc)",
            hit.doc_id,
            hit.score
        );
    }

    // A higher custom_freq strictly increases the BM25 score at fixed
    // doc_freq/doc_count/field_length (monotonic tfNorm) -- doc 1
    // (custom_freq 50) must outrank doc 0 (custom_freq 5), which must
    // outrank doc 2 (custom_freq 1). This is exactly the ranking a real
    // TopDocsCollector would produce, confirming the custom freq drives
    // real relevance ranking, not just an opaque stored number.
    assert_eq!(hits[0].doc_id, 1);
    assert_eq!(hits[1].doc_id, 0);
    assert_eq!(hits[2].doc_id, 2);
}

#[test]
fn custom_freq_postings_field_rejects_a_zero_custom_freq() {
    // The codec layer's existing "freq < 1" rejection applies unchanged to
    // DocsAndCustomFreqs -- an IndexWriter caller can't silently commit a
    // meaningless zero/negative custom score.
    let tmp = tempdir("custom-freq-rejects-zero");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        field_info(0, "id", IndexOptions::None),
        field_info(1, "score", IndexOptions::DocsAndCustomFreqs),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer
        .set_custom_freq_postings_field(Some("score"))
        .unwrap();

    writer.add_document_with_custom_freq_terms(doc("a"), vec![("score".to_string(), 0)]);
    assert!(writer.commit().is_err());
}
