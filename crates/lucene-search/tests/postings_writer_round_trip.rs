//! Required end-to-end proof for the postings write side
//! (`lucene_codecs::postings_writer::write_single_field`): hand-built
//! single-field postings, written by this port's own new writer, opened by
//! the *existing, unmodified* `lucene_codecs::blocktree::open` +
//! `lucene_codecs::postings::DocInput`, and queried via the *existing,
//! unmodified* `lucene_search::search_term_query` — proving real doc IDs
//! come back correctly through the whole read stack, not just that the
//! written bytes decode in isolation (see
//! `lucene-codecs/src/postings_writer.rs`'s own unit tests for the
//! byte/decode-level checks; this file is the query-layer capstone the task
//! specifically requires).
//!
//! Lives in `lucene-search` rather than `lucene-codecs` because
//! `lucene-codecs` must not depend on `lucene-search` (strictly downward
//! dependency graph, see the `architecture` skill) — `lucene-search`
//! already depends on `lucene-codecs`, so this is the natural home for a
//! writer-then-query round trip.

use lucene_codecs::blocktree;
use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::postings::{DocInput, PosInput};
use lucene_codecs::postings_writer::{write_single_field, FieldPostingsInput, TermPostings};
use lucene_search::{search_phrase_query, search_term_query, PhraseQuery, TermQuery, VecCollector};

const SEG_ID: [u8; 16] = [42u8; 16];
const SUFFIX: &str = "";

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

/// Writes a single field ("body") with a mix of singleton and multi-doc
/// terms, opens it back through the real read side, and runs
/// `search_term_query` for each term plus a missing term — asserting the
/// exact doc ID sets a real `IndexSearcher` would return.
#[test]
fn term_query_finds_correct_docs_over_freshly_written_postings() {
    let terms = vec![
        TermPostings {
            term: b"fox".to_vec(),
            docs: vec![(1, 2), (4, 1), (7, 3)],
            ..Default::default()
        },
        TermPostings {
            term: b"quick".to_vec(),
            docs: vec![(4, 1)], // docFreq == 1: pulsed into the term dict, no .doc bytes
            ..Default::default()
        },
        TermPostings {
            term: b"the".to_vec(),
            docs: vec![(0, 1), (1, 1), (4, 2), (7, 1)],
            ..Default::default()
        },
    ];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 8,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).expect("write_single_field");

    let field_infos = FieldInfos {
        fields: vec![field_info(0, "body", IndexOptions::DocsAndFreqs)],
    };
    let fields = blocktree::open(
        &output.tim,
        &output.tip,
        &output.tmd,
        &field_infos,
        &SEG_ID,
        SUFFIX,
        8,
    )
    .expect("blocktree::open on freshly written .tim/.tip/.tmd");
    let doc_in = DocInput::open(&output.doc, &SEG_ID, SUFFIX).expect("DocInput::open");

    let case = |term: &str, expected: &[i32]| {
        let mut collector = VecCollector::default();
        let query = TermQuery::new("body", term.as_bytes());
        search_term_query(&fields, Some(&doc_in), None, &query, &mut collector)
            .unwrap_or_else(|e| panic!("search_term_query({term:?}) failed: {e}"));
        assert_eq!(collector.docs, expected, "term {term:?}");
    };

    case("fox", &[1, 4, 7]);
    case("quick", &[4]);
    case("the", &[0, 1, 4, 7]);
    case("missing", &[]);
}

/// Same as above but with a `.liv` (live-docs) style filter applied via
/// `search_term_query`'s `live_docs` parameter, confirming the writer's
/// output composes correctly with deletion filtering exactly like a real
/// segment would (doc 4 marked deleted, so every term's result excludes it
/// even though it's in that term's raw postings list).
#[test]
fn term_query_respects_live_docs_filter() {
    let terms = vec![TermPostings {
        term: b"fox".to_vec(),
        docs: vec![(1, 1), (4, 1), (7, 1)],
        ..Default::default()
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 3,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).expect("write_single_field");

    let field_infos = FieldInfos {
        fields: vec![field_info(0, "body", IndexOptions::DocsAndFreqs)],
    };
    let fields = blocktree::open(
        &output.tim,
        &output.tip,
        &output.tmd,
        &field_infos,
        &SEG_ID,
        SUFFIX,
        8,
    )
    .unwrap();
    let doc_in = DocInput::open(&output.doc, &SEG_ID, SUFFIX).unwrap();

    let mut live_docs = lucene_util::FixedBitSet::new(8);
    for d in [0, 1, 2, 3, 5, 6, 7] {
        live_docs.set(d);
    }
    // doc 4 left unset (deleted).

    let mut collector = VecCollector::default();
    let query = TermQuery::new("body", b"fox".as_slice());
    search_term_query(
        &fields,
        Some(&doc_in),
        Some(&live_docs),
        &query,
        &mut collector,
    )
    .unwrap();
    assert_eq!(collector.docs, vec![1, 7]);
}

/// The critical end-to-end proof for the positions write-side: writes a
/// multi-term field (`"fox"`/`"jumps"`/`"quick"`, three terms so the `.tim`
/// suffix/stats/metadata threading gets exercised, not just a single-term
/// edge case) with `IndexOptions::DocsAndFreqsAndPositions`, where two docs
/// share every term but only one doc has them in an exactly adjacent
/// "quick fox" pattern -- then runs the *existing, unmodified*
/// `lucene_search::search_phrase_query` against the freshly written
/// `.doc`/`.pos`/`.tim`/`.tip`/`.tmd` bytes and asserts:
/// - doc 0 ("quick fox jumps": quick@0, fox@1, jumps@2) matches the
///   `["quick", "fox"]` phrase (exact adjacency) and also `["fox", "jumps"]`.
/// - doc 1 ("quick jumps fox": quick@0, jumps@1, fox@2) does **not** match
///   `["quick", "fox"]` (both terms are present in the doc, and a plain
///   `TermQuery` conjunction would wrongly include it, but they're 2 apart,
///   not adjacent) -- the required negative case proving this isn't just a
///   doc-ID conjunction wearing a phrase-query hat.
#[test]
fn phrase_query_finds_correct_docs_over_freshly_written_positions() {
    let terms = vec![
        TermPostings {
            term: b"fox".to_vec(),
            docs: vec![(0, 1), (1, 1)],
            positions: vec![vec![1], vec![2]],
        },
        TermPostings {
            term: b"jumps".to_vec(),
            docs: vec![(0, 1), (1, 1)],
            positions: vec![vec![2], vec![1]],
        },
        TermPostings {
            term: b"quick".to_vec(),
            docs: vec![(0, 1), (1, 1)],
            positions: vec![vec![0], vec![0]],
        },
    ];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: 2,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).expect("write_single_field");

    let field_infos = FieldInfos {
        fields: vec![field_info(
            0,
            "body",
            IndexOptions::DocsAndFreqsAndPositions,
        )],
    };
    let fields = blocktree::open(
        &output.tim,
        &output.tip,
        &output.tmd,
        &field_infos,
        &SEG_ID,
        SUFFIX,
        2,
    )
    .expect("blocktree::open on freshly written .tim/.tip/.tmd");
    let doc_in = DocInput::open(&output.doc, &SEG_ID, SUFFIX).expect("DocInput::open");
    let pos_in = PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("PosInput::open");

    let case = |terms: &[&[u8]], expected: &[i32]| {
        let mut collector = VecCollector::default();
        let query = PhraseQuery::new("body", terms.iter().map(|t| t.to_vec()));
        search_phrase_query(
            &fields,
            Some(&doc_in),
            Some(&pos_in),
            None,
            None,
            &query,
            &mut collector,
        )
        .unwrap_or_else(|e| panic!("search_phrase_query({terms:?}) failed: {e}"));
        assert_eq!(collector.docs, expected, "phrase {terms:?}");
    };

    case(&[b"quick", b"fox"], &[0]);
    case(&[b"fox", b"jumps"], &[0]);
    // doc 1 is "quick jumps fox" (quick@0, jumps@1, fox@2): this exact
    // 3-term order matches doc 1, not doc 0 -- the mirror-image negative
    // case showing the writer's positions distinguish more than just doc-ID
    // conjunction (both docs contain all three terms).
    case(&[b"quick", b"jumps", b"fox"], &[1]);
}
