use std::sync::Arc;

use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_index::document::{
    FieldToken, FieldTokens, FieldType, IndexOptions, KeywordField, KnnFloatVectorField,
    SortedNumericDocValuesField, SortedSetDocValuesField, Store, StringField, TextField,
    VectorSimilarityFunction,
};

use super::*;
use crate::query::{Clause, TermQuery};
use crate::similarities::ClassicSimilarity;

fn std_analyzer() -> Analyzer {
    Analyzer::new(StandardAnalyzer::new())
}

fn term(field: &str, text: &str) -> Clause {
    Clause::Term(TermQuery::new(field, text.as_bytes().to_vec()))
}

fn token(text: &str, inc: i32, start: i32, end: i32) -> FieldToken {
    FieldToken {
        position_increment: inc,
        ..FieldToken::new(text.as_bytes().to_vec(), start, end)
    }
}

fn stream(tokens: Vec<FieldToken>, final_offset: i32) -> FieldTokens {
    FieldTokens {
        tokens,
        final_position_increment: 0,
        final_offset,
        end_attributes: None,
    }
}

#[test]
fn values_of_one_field_continue_after_the_gaps() {
    let mut mi = MemoryIndex::with_options(true, false);
    mi.add_field_tokens("f", &stream(vec![token("a", 1, 0, 1)], 1), 10, 5)
        .unwrap();
    mi.add_field_tokens(
        "f",
        &stream(vec![token("b", 1, 0, 1), token("c", 0, 0, 1)], 3),
        10,
        5,
    )
    .unwrap();
    // `a` at 0; `b` at 0 + 10 + 1, offsets shifted by 1 + 5; `c` stacked on it
    assert_eq!(
        mi.to_string_debug(),
        "f:\n\t'[61]':1: [(0, 0, 1)]\n\t'[62]':1: [(11, 6, 7)]\n\t'[63]':1: [(11, 6, 7)]\n\
         \tterms=3, positions=3\n\nfields=1, terms=3, positions=3"
    );
    let info = &mi.fields["f"];
    assert_eq!(
        (
            info.num_tokens,
            info.num_overlap_tokens,
            info.last_position,
            info.last_offset
        ),
        (3, 1, 11, 9)
    );
}

#[test]
fn an_empty_value_leaves_the_gap_rule_untouched() {
    let mut mi = MemoryIndex::new();
    // no tokens yet: the next value still starts at -1, without a gap
    mi.add_field_tokens("f", &stream(vec![], 7), 100, 100)
        .unwrap();
    mi.add_field_tokens("f", &stream(vec![token("a", 1, 0, 1)], 1), 100, 100)
        .unwrap();
    assert!(mi.to_string_debug().contains("'[61]':1: [(0)]"));
    assert_eq!(mi.fields["f"].last_offset, 1);
}

#[test]
fn a_binary_term_takes_the_next_position_without_offsets_or_payload() {
    let mut mi = MemoryIndex::with_options(true, true);
    let field = StringField::from_bytes("id", vec![0xff, 0x01], Store::No);
    mi.add_field(&field, None).unwrap();
    mi.add_field(&field, None).unwrap();
    assert!(mi
        .to_string_debug()
        .contains("'[ff 1]':2: [(0, 0, 0), (1, 0, 0)]"));
    assert_eq!(mi.search(&term("id", "\u{1}")).unwrap(), 0.0);
}

#[test]
fn payloads_print_and_index_only_when_non_empty() {
    let mut mi = MemoryIndex::with_options(false, true);
    let mut a = token("a", 1, 0, 1);
    a.payload = Some(vec![1, 0xab]);
    let mut b = token("a", 1, 2, 3);
    b.payload = Some(Vec::new());
    mi.add_field_tokens("f", &stream(vec![a, b], 3), 0, 1)
        .unwrap();
    assert!(mi
        .to_string_debug()
        .contains("'[61]':2: [(0, [1 ab]), (1)]"));
    assert!(mi.search(&term("f", "a")).unwrap() > 0.0);
}

#[test]
fn a_position_the_index_cannot_hold_is_refused_at_build() {
    let mut mi = MemoryIndex::new();
    // a first increment of 0 puts the token at -1, which Java's reader
    // serves and a segment cannot hold
    mi.add_field_tokens("f", &stream(vec![token("a", 0, 0, 1)], 1), 0, 1)
        .unwrap();
    let e = mi.freeze().unwrap_err();
    assert!(e.to_string().contains("negative or goes backwards"), "{e}");
    let mut mi = MemoryIndex::new();
    mi.add_field_tokens(
        "f",
        &stream(vec![token("a", 3, 0, 1), token("a", -2, 0, 1)], 1),
        0,
        1,
    )
    .unwrap();
    assert!(mi.freeze().is_err());
}

#[test]
fn freezing_refuses_fields_and_similarities_and_reset_thaws() {
    let mut mi = MemoryIndex::new();
    mi.add_field_text("body", "a b a", &std_analyzer()).unwrap();
    mi.freeze().unwrap();
    mi.freeze().unwrap();
    let first = mi.create_reader().unwrap();
    assert!(Arc::ptr_eq(&first, &mi.create_reader().unwrap()));
    assert_eq!(
        mi.add_field_text("body", "c", &std_analyzer())
            .unwrap_err()
            .to_string(),
        "Cannot call addField() when MemoryIndex is frozen"
    );
    assert_eq!(
        mi.set_similarity(Arc::new(ClassicSimilarity::default()))
            .unwrap_err()
            .to_string(),
        "Cannot set Similarity when MemoryIndex is frozen"
    );
    let bm25 = mi.search(&term("body", "a")).unwrap();
    mi.reset();
    assert_eq!(mi.to_string_debug(), "\nfields=0, terms=0, positions=0");
    mi.set_similarity(Arc::new(ClassicSimilarity::default()))
        .unwrap();
    mi.add_field_text("body", "a b a", &std_analyzer()).unwrap();
    let classic = mi.search(&term("body", "a")).unwrap();
    assert_ne!(bm25, classic);
    // the same similarity again changes nothing; a new one drops the norms
    let sim: Arc<dyn Similarity> = Arc::new(ClassicSimilarity::default());
    mi.set_similarity(Arc::clone(&sim)).unwrap();
    mi.fields.get_mut("body").unwrap().norm(sim.as_ref());
    mi.set_similarity(Arc::clone(&sim)).unwrap();
    assert!(mi.fields["body"].norm.is_some());
    mi.set_similarity(Arc::new(Bm25Similarity::default()))
        .unwrap();
    assert!(mi.fields["body"].norm.is_none());
    assert_eq!(mi.search(&term("body", "a")).unwrap(), bm25);
}

#[test]
fn a_query_that_matches_nothing_scores_zero() {
    let mut mi = MemoryIndex::default();
    mi.add_field_text("body", "a", &std_analyzer()).unwrap();
    assert_eq!(mi.search(&term("body", "zzz")).unwrap(), 0.0);
    assert_eq!(
        mi.search(&Clause::MatchNoDocs(crate::query::MatchNoDocsQuery::new()))
            .unwrap(),
        0.0
    );
    let mut both = crate::query::BooleanQuery::new();
    both.should.push(term("body", "a"));
    assert!(mi.search(&Clause::Boolean(Box::new(both))).unwrap() > 0.0);
    let reader = mi.create_reader().unwrap();
    assert_eq!(reader.directory_reader().unwrap().max_doc(), 1);
}

#[test]
fn doc_values_points_and_stored_values_are_kept() {
    let mut mi = MemoryIndex::new();
    mi.add_field(&SortedNumericDocValuesField::new("sn", 3), None)
        .unwrap();
    mi.add_field(&SortedNumericDocValuesField::new("sn", 1), None)
        .unwrap();
    mi.add_field(&SortedSetDocValuesField::new("ss", b"b".to_vec()), None)
        .unwrap();
    mi.add_field(&SortedSetDocValuesField::new("ss", b"b".to_vec()), None)
        .unwrap();
    mi.add_field(&KeywordField::new("k", "v", Store::Yes), None)
        .unwrap();
    assert_eq!(mi.fields["sn"].numeric_dv, vec![3, 1]);
    assert_eq!(mi.fields["ss"].sorted_set_dv.len(), 1);
    assert_eq!(mi.fields["k"].stored.len(), 1);
    mi.freeze().unwrap();
    assert!(mi.search(&term("k", "v")).unwrap() > 0.0);
}

#[test]
fn a_point_of_another_shape_is_refused() {
    let mut mi = MemoryIndex::new();
    mi.add_field(
        &lucene_index::document::IntPoint::new("p", &[1]).unwrap(),
        None,
    )
    .unwrap();
    let e = mi
        .add_field(
            &lucene_index::document::IntPoint::new("p", &[1, 2]).unwrap(),
            None,
        )
        .unwrap_err();
    assert_eq!(
        e.to_string(),
        "cannot change point dimension count from 1 to 2 for field=\"p\""
    );
}

#[test]
fn fields_the_port_cannot_take_are_refused() {
    let mut mi = MemoryIndex::new();
    let v =
        KnnFloatVectorField::new("v", vec![1.0, 2.0], VectorSimilarityFunction::Euclidean).unwrap();
    assert!(mi.add_field(&v, None).is_err());
    // a tokenized field without an analyzer: Java's NullPointerException
    let t = TextField::new("t", "a b", Store::No);
    assert!(mi
        .add_field(&t, None)
        .unwrap_err()
        .to_string()
        .starts_with("java.lang.NullPointerException"));
    // an indexed binary-invertable field without a value
    let mut ft = FieldType::new();
    ft.set_index_options(IndexOptions::Docs).unwrap();
    ft.set_tokenized(false).unwrap();
    #[derive(Debug)]
    struct NoValue(FieldType);
    impl lucene_index::document::IndexableField for NoValue {
        fn name(&self) -> &str {
            "nv"
        }
        fn field_type(&self) -> &FieldType {
            &self.0
        }
        fn invertable_type(&self) -> lucene_index::document::InvertableType {
            lucene_index::document::InvertableType::Binary
        }
        fn token_stream(
            &self,
            _: &Analyzer,
        ) -> lucene_index::document::Result<Option<FieldTokens>> {
            Ok(None)
        }
    }
    assert_eq!(
        mi.add_field(&NoValue(ft), None).unwrap_err().to_string(),
        "Indexed field must provide a TokenStream or a binary value"
    );
}

#[test]
fn from_document_adds_every_field_with_the_analyzer() {
    let mut doc = lucene_index::document::Document::new();
    doc.add(TextField::new("body", "Quick fox", Store::No));
    doc.add(StringField::new("id", "1", Store::Yes));
    let mut mi = MemoryIndex::from_document(&doc, &std_analyzer(), true, false).unwrap();
    assert!(mi.search(&term("body", "quick")).unwrap() > 0.0);
    assert!(mi.search(&term("id", "1")).unwrap() > 0.0);
    let tokens = MemoryIndex::keyword_token_stream(&["ab", "c"]);
    assert_eq!(
        tokens
            .tokens
            .iter()
            .map(|t| (t.start_offset, t.end_offset))
            .collect::<Vec<_>>(),
        vec![(0, 2), (3, 4)]
    );
}

/// Whitespace, then `DelimitedPayloadTokenFilter('|', IdentityEncoder)`.
struct Payloads;

impl lucene_analysis::AnalyzerDefinition for Payloads {
    fn create_components(
        &self,
        _field: &str,
    ) -> std::result::Result<lucene_analysis::TokenStreamComponents, lucene_analysis::AnalysisError>
    {
        Ok(lucene_analysis::TokenStreamComponents::new(
            lucene_analysis::payloads::DelimitedPayloadTokenFilter::new(
                lucene_analysis::util::WhitespaceTokenizer::new(),
                u16::from(b'|'),
                lucene_analysis::payloads::IdentityEncoder,
            ),
        ))
    }
}

/// The in-memory form and the segment answer every query of the fixture
/// corpus alike, document by document, bit for bit -- under offsets,
/// payloads and a non-default similarity.
#[test]
fn the_in_memory_form_and_the_segment_score_alike() {
    let corpus = |name: &str| -> Vec<String> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/corpus")
            .join(name);
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect()
    };
    let docs = corpus("memory-index-docs.txt");
    let queries = corpus("memory-index-queries.txt");
    let configs: [(Analyzer, bool, bool, Arc<dyn Similarity>); 3] = [
        (
            std_analyzer(),
            false,
            false,
            Arc::new(Bm25Similarity::default()),
        ),
        (
            std_analyzer(),
            true,
            false,
            Arc::new(ClassicSimilarity::default()),
        ),
        (
            Analyzer::new(Payloads),
            true,
            true,
            Arc::new(Bm25Similarity::default()),
        ),
    ];
    let mut compared = 0;
    for (analyzer, offsets, payloads, sim) in configs {
        let analyzer = Arc::new(analyzer);
        let parser = crate::queryparser::classic::QueryParser::new("body", Arc::clone(&analyzer));
        let clauses: Vec<Clause> = queries
            .iter()
            .map(|q| parser.parse(q).unwrap().to_clause(1))
            .collect();
        for line in &docs {
            let mut mi = MemoryIndex::with_options(offsets, payloads);
            mi.set_similarity(Arc::clone(&sim)).unwrap();
            for pair in line.split('\t') {
                let (field, value) = pair.split_once('=').unwrap();
                mi.add_field_text(field, value, &analyzer).unwrap();
            }
            mi.freeze().unwrap();
            let reader = mi.create_reader().unwrap();
            for (q, clause) in queries.iter().zip(&clauses) {
                let fast = reader.search(clause).unwrap();
                let segment = match reader::as_boolean(clause) {
                    Some(bq) => reader.search_segment(&bq).unwrap(),
                    None => 0.0,
                };
                assert_eq!(fast.to_bits(), segment.to_bits(), "{line:?} {q:?}");
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 3 * docs.len() * queries.len());
}
