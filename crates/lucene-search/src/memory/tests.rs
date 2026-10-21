use std::sync::Arc;

use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_codecs::term_vectors::{TermVectorField, TermVectorTerm};
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
                let segment = match reader::as_boolean(&clause.clone().rewrite()) {
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

// ---------------------------------------------------------------------------
// The in-heap leaf against the codec form
// ---------------------------------------------------------------------------

fn corpus(name: &str) -> Vec<String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/corpus")
        .join(name);
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
}

/// A matches tree as text: each field's iterator, sub-matches included.
fn render_matches(m: &crate::Result<Option<crate::matches::BoxMatches>>) -> String {
    use crate::matches::{clause_name, BoxMatchesIterator};
    fn iter(mut it: BoxMatchesIterator, out: &mut String) {
        while it.next().unwrap() {
            out.push_str(&format!(
                "[{} {} {} {} {}",
                it.start_position(),
                it.end_position(),
                it.start_offset(),
                it.end_offset(),
                clause_name(it.query())
            ));
            if let Some(sub) = it.sub_matches().unwrap() {
                iter(sub, out);
            }
            out.push(']');
        }
    }
    match m {
        Err(e) => format!("error {e}"),
        Ok(None) => "none".into(),
        Ok(Some(m)) if m.is_match_with_no_terms() => "noterms".into(),
        Ok(Some(m)) => {
            let mut out = String::new();
            for field in m.fields() {
                out.push_str(&field);
                out.push('|');
                if let Some(it) = m.get_matches(&field).unwrap() {
                    iter(it, &mut out);
                }
                out.push(';');
            }
            out
        }
    }
}

/// The clauses beyond the corpus's: every kind the in-heap leaf answers,
/// and some it hands to the codec form.
fn extra_clauses() -> Vec<Clause> {
    use crate::query::{
        BoostQuery, ConstantScoreQuery, DisjunctionMaxQuery, FieldExistsQuery, FuzzyQuery,
        MatchAllDocsQuery, MultiPhraseQuery, PrefixQuery, RegexpQuery, SpanQuery, TermInSetQuery,
        WildcardQuery,
    };
    let mut sloppy = MultiPhraseQuery::new("body", [vec!["quick", "a"], vec!["fox", "b"]]);
    sloppy.slop = 2;
    vec![
        PrefixQuery::new("body", "qu").into(),
        PrefixQuery::new("absent", "qu").into(),
        WildcardQuery::new("body", "f?x").into(),
        WildcardQuery::new("title", "*a*").into(),
        RegexpQuery::new("body", "[a-f].*").into(),
        RegexpQuery::new("absent", "[").into(),
        TermInSetQuery::new("body", ["fox", "a", "zzz", "b"]).into(),
        FuzzyQuery::new("body", "quack").into(),
        FuzzyQuery::new("absent", "quack").into(),
        MultiPhraseQuery::new("body", [vec!["quick", "a"], vec!["brown", "b"]]).into(),
        sloppy.into(),
        SpanQuery::SpanNear {
            clauses: vec![
                SpanQuery::span_term("body", "quick"),
                SpanQuery::span_term("body", "fox"),
            ],
            slop: 2,
            in_order: true,
        }
        .into(),
        SpanQuery::SpanOr {
            clauses: vec![
                SpanQuery::span_term("body", "a"),
                SpanQuery::span_term("title", "alaska"),
            ],
        }
        .into(),
        MatchAllDocsQuery::new(1).into(),
        Clause::Exists(FieldExistsQuery::new("body")),
        DisjunctionMaxQuery::new([term("body", "a"), term("body", "fox")], 0.1).into(),
        ConstantScoreQuery::new(term("body", "b"), 2.0).into(),
        BoostQuery::new(term("title", "a"), 3.0).into(),
    ]
}

/// `Weight.matches` from the in-heap arrays answers as the codec form of
/// the same document does, for every clause of the corpus and the extras,
/// with and without offsets and payloads.
#[test]
fn in_heap_matches_duel_the_codec_form() {
    let docs = corpus("memory-index-docs.txt");
    let queries = corpus("memory-index-queries.txt");
    let mut compared = 0;
    for (offsets, payloads) in [(false, false), (true, false), (true, true)] {
        let analyzer = Arc::new(if payloads {
            Analyzer::new(Payloads)
        } else {
            std_analyzer()
        });
        let parser = crate::queryparser::classic::QueryParser::new("body", Arc::clone(&analyzer));
        let mut clauses: Vec<Clause> = queries
            .iter()
            .map(|q| parser.parse(q).unwrap().to_clause(1).rewrite())
            .collect();
        clauses.extend(extra_clauses());
        for line in &docs {
            let mut mi = MemoryIndex::with_options(offsets, payloads);
            for pair in line.split('\t') {
                let (field, value) = pair.split_once('=').unwrap();
                mi.add_field_text(field, value, &analyzer).unwrap();
            }
            let reader = mi.create_reader().unwrap();
            for clause in &clauses {
                let fast = render_matches(&reader.matches(clause, 0));
                let codec = render_matches(
                    &reader
                        .with_searcher(|s| {
                            crate::matches::leaf_matches(&s.segments()[0], clause, 0)
                        })
                        .unwrap(),
                );
                assert_eq!(fast, codec, "{line:?} {clause:?}");
                compared += 1;
            }
        }
    }
    assert!(compared > 3 * 16 * 56);
}

#[test]
fn in_heap_matches_refuse_another_document() {
    use crate::matches::MatchesLeaf;
    let mut mi = MemoryIndex::new();
    mi.add_field_text("body", "a b", &std_analyzer()).unwrap();
    let reader = mi.create_reader().unwrap();
    assert!(matches!(
        reader.matches(&term("body", "a"), 1),
        Err(crate::Error::IllegalArgument(_))
    ));
    assert!(reader.occurrences("body", b"a", 1).unwrap().is_none());
    assert!(reader.reads_positions());
    assert!(reader.field_index_options("absent").is_none());
    assert!(reader.expanded_terms(&term("body", "a")).unwrap().is_none());
    assert!(MatchesLeaf::codec_matches(&*reader, &term("body", "a"), 0)
        .unwrap()
        .is_some());
}

/// A document with every kind of value a memory index holds.
fn rich_index() -> Arc<MemoryIndexReader> {
    use lucene_index::document::{
        BinaryDocValuesField, Field, IntPoint, NumericDocValuesField, SortedDocValuesField,
        StoredValue,
    };
    let mut mi = MemoryIndex::with_options(true, true);
    let payloads = Analyzer::new(Payloads);
    mi.add_field_text("body", "a|x b a|yz c b a", &payloads)
        .unwrap();
    mi.add_field_text("plain", "one two one", &std_analyzer())
        .unwrap();
    mi.add_field(&NumericDocValuesField::new("n", 7), None)
        .unwrap();
    mi.add_field(&SortedNumericDocValuesField::new("sn", 3), None)
        .unwrap();
    mi.add_field(&SortedNumericDocValuesField::new("sn", -1), None)
        .unwrap();
    mi.add_field(&BinaryDocValuesField::new("bin", b"bv".to_vec()), None)
        .unwrap();
    mi.add_field(&SortedDocValuesField::new("sd", b"s".to_vec()), None)
        .unwrap();
    mi.add_field(&SortedSetDocValuesField::new("ss", b"z".to_vec()), None)
        .unwrap();
    mi.add_field(&SortedSetDocValuesField::new("ss", b"m".to_vec()), None)
        .unwrap();
    mi.add_field(&IntPoint::new("p1", &[5]).unwrap(), None)
        .unwrap();
    mi.add_field(&IntPoint::new("p1", &[-3]).unwrap(), None)
        .unwrap();
    mi.add_field(&IntPoint::new("p2", &[1, 9]).unwrap(), None)
        .unwrap();
    mi.add_field(&IntPoint::new("p2", &[4, 2]).unwrap(), None)
        .unwrap();
    mi.add_field(&KeywordField::new("k", "v", Store::Yes), None)
        .unwrap();
    for (name, value) in [
        ("s_str", StoredValue::String("str".into())),
        ("s_bin", StoredValue::Binary(vec![1, 2])),
        ("s_int", StoredValue::Int(-4)),
        ("s_long", StoredValue::Long(1 << 40)),
        ("s_float", StoredValue::Float(1.5)),
        ("s_double", StoredValue::Double(-2.25)),
    ] {
        mi.add_field(&Field::stored(name, value), None).unwrap();
    }
    mi.create_reader().unwrap()
}

/// Records what an intersection visits, answering one relation.
#[derive(Default)]
struct Visits {
    inside: bool,
    outside: bool,
    seen: Vec<(i32, Vec<u8>)>,
}

impl crate::reader::IntersectVisitor for Visits {
    fn compare(&mut self, _min: &[u8], _max: &[u8]) -> crate::reader::Relation {
        if self.outside {
            crate::reader::Relation::CellOutsideQuery
        } else if self.inside {
            crate::reader::Relation::CellInsideQuery
        } else {
            crate::reader::Relation::CellCrossesQuery
        }
    }
    fn visit(&mut self, doc_id: i32) {
        self.seen.push((doc_id, Vec::new()));
    }
    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        self.seen.push((doc_id, packed_value.to_vec()));
    }
}

/// Records the stored fields visited, skipping or stopping on request.
struct Stored {
    skip: i32,
    stop: i32,
    seen: Vec<String>,
}

type StoredResult<T> = lucene_codecs::stored_fields::Result<T>;

impl lucene_codecs::stored_fields::StoredFieldVisitor for Stored {
    fn needs_field(&mut self, n: i32) -> StoredResult<lucene_codecs::stored_fields::VisitStatus> {
        use lucene_codecs::stored_fields::VisitStatus;
        Ok(if n == self.stop {
            VisitStatus::Stop
        } else if n == self.skip {
            VisitStatus::No
        } else {
            VisitStatus::Yes
        })
    }
    fn string_field(&mut self, n: i32, v: &str) -> StoredResult<()> {
        self.seen.push(format!("{n}:s:{v}"));
        Ok(())
    }
    fn binary_field(&mut self, n: i32, v: &[u8]) -> StoredResult<()> {
        self.seen.push(format!("{n}:b:{v:?}"));
        Ok(())
    }
    fn int_field(&mut self, n: i32, v: i32) -> StoredResult<()> {
        self.seen.push(format!("{n}:i:{v}"));
        Ok(())
    }
    fn long_field(&mut self, n: i32, v: i64) -> StoredResult<()> {
        self.seen.push(format!("{n}:l:{v}"));
        Ok(())
    }
    fn float_field(&mut self, n: i32, v: f32) -> StoredResult<()> {
        self.seen.push(format!("{n}:f:{v}"));
        Ok(())
    }
    fn double_field(&mut self, n: i32, v: f64) -> StoredResult<()> {
        self.seen.push(format!("{n}:d:{v}"));
        Ok(())
    }
}

/// Everything a [`crate::reader::LeafReader`] says of one field, as text.
fn describe_leaf(r: &dyn crate::reader::LeafReader, field: &str) -> Vec<String> {
    use crate::reader::{PostingsFlags, NO_MORE_DOCS};
    let mut out = Vec::new();
    match r.terms(field).unwrap() {
        None => out.push("no terms".into()),
        Some(t) => {
            out.push(format!(
                "size {} sttf {} sdf {} dc {} freqs {} offs {} pos {} pay {}",
                t.size(),
                t.sum_total_term_freq(),
                t.sum_doc_freq(),
                t.doc_count(),
                t.has_freqs(),
                t.has_offsets(),
                t.has_positions(),
                t.has_payloads()
            ));
            let mut te = t.iterator().unwrap();
            while let Some(term) = te.next().unwrap() {
                let term = term.to_vec();
                out.push(format!(
                    "{term:?} df {} ttf {}",
                    te.doc_freq().unwrap(),
                    te.total_term_freq().unwrap()
                ));
                for flags in [
                    PostingsFlags::None,
                    PostingsFlags::Freqs,
                    PostingsFlags::Positions,
                    PostingsFlags::Offsets,
                    PostingsFlags::Payloads,
                ] {
                    let mut p = te.postings(flags).unwrap();
                    let mut s = format!("{flags:?}");
                    while p.next_doc().unwrap() != NO_MORE_DOCS {
                        s.push_str(&format!(" doc {} freq {}", p.doc_id(), p.freq()));
                        if flags.wants_positions() {
                            for _ in 0..p.freq() {
                                s.push_str(&format!(
                                    " {}@{}-{}{:?}",
                                    p.next_position().unwrap(),
                                    p.start_offset(),
                                    p.end_offset(),
                                    p.payload()
                                ));
                            }
                        }
                    }
                    out.push(s);
                }
            }
            for target in [&b""[..], b"a", b"aa", b"b", b"one", b"zzz"] {
                let mut te = t.iterator().unwrap();
                let status = te.try_seek_ceil(target).unwrap();
                out.push(format!("ceil {target:?} {status:?} {:?}", te.term()));
                let found = te.try_seek_exact(target).unwrap();
                out.push(format!("exact {target:?} {found}"));
            }
        }
    }
    if let Some(mut n) = r.norm_values(field).unwrap() {
        out.push(format!(
            "norm {} {}",
            n.advance_exact(0).unwrap(),
            n.long_value()
        ));
    }
    if let Some(mut v) = r.numeric_doc_values(field).unwrap() {
        out.push(format!(
            "numeric {} {}",
            v.next_doc().unwrap(),
            v.long_value()
        ));
        out.push(format!("then {} cost {}", v.next_doc().unwrap(), v.cost()));
    }
    if let Some(mut v) = r.binary_doc_values(field).unwrap() {
        out.push(format!(
            "binary {} {:?}",
            v.advance(0).unwrap(),
            v.binary_value()
        ));
    }
    if let Some(mut v) = r.sorted_doc_values(field).unwrap() {
        out.push(format!(
            "sorted {} {} {} {:?}",
            v.advance_exact(0).unwrap(),
            v.ord_value(),
            v.value_count(),
            v.lookup_ord(0).unwrap()
        ));
    }
    if let Some(mut v) = r.sorted_numeric_doc_values(field).unwrap() {
        for _ in 0..2 {
            assert!(v.advance_exact(0).unwrap());
            let values: Vec<i64> = (0..v.doc_value_count())
                .map(|_| v.next_value().unwrap())
                .collect();
            out.push(format!("sorted numeric {values:?}"));
        }
    }
    if let Some(mut v) = r.sorted_set_doc_values(field).unwrap() {
        for _ in 0..2 {
            assert!(v.advance_exact(0).unwrap());
            let ords: Vec<i64> = (0..v.doc_value_count())
                .map(|_| v.next_ord().unwrap())
                .collect();
            out.push(format!("sorted set {ords:?} of {}", v.value_count()));
        }
        for o in 0..v.value_count() {
            out.push(format!("ord {o} {:?}", v.lookup_ord(o).unwrap()));
        }
    }
    if let Some(p) = r.point_values(field).unwrap() {
        out.push(format!(
            "points {:?} {:?} {} {} {} {} {}",
            p.min_packed_value(),
            p.max_packed_value(),
            p.num_dimensions(),
            p.num_index_dimensions(),
            p.bytes_per_dimension(),
            p.size(),
            p.doc_count()
        ));
        let mut v = Visits {
            inside: true,
            ..Visits::default()
        };
        p.intersect(&mut v).unwrap();
        out.push(format!("inside {}", v.seen.len()));
    }
    out
}

/// The in-heap `LeafReader` reads as the segment of the same document does,
/// field by field.
#[test]
fn leaf_duels_the_segment() {
    use crate::reader::LeafReader;
    let reader = rich_index();
    let segment = &reader.directory_reader().unwrap().segment_readers()[0];
    let fields = [
        "body", "plain", "n", "sn", "bin", "sd", "ss", "p1", "p2", "k", "s_str", "absent",
    ];
    for field in fields {
        assert_eq!(
            describe_leaf(&*reader, field),
            describe_leaf(segment, field),
            "{field}"
        );
    }
    let names = |r: &dyn LeafReader| -> Vec<(String, i32)> {
        r.field_infos()
            .fields
            .iter()
            .map(|f| (f.name.clone(), f.number))
            .collect()
    };
    assert_eq!(names(&*reader), names(segment));
    let stored = |r: &dyn LeafReader, skip, stop| {
        let mut v = Stored {
            skip,
            stop,
            seen: Vec::new(),
        };
        r.document(0, &mut v).unwrap();
        v.seen
    };
    assert_eq!(stored(&*reader, -1, -1), stored(segment, -1, -1));
    assert_eq!(stored(&*reader, -1, -1).len(), 7);
    // a skipped field is not visited, and STOP ends the walk
    let number = |name: &str| {
        reader
            .field_infos()
            .fields
            .iter()
            .find(|f| f.name == name)
            .unwrap()
            .number
    };
    assert_eq!(stored(&*reader, number("s_int"), -1).len(), 6);
    assert!(stored(&*reader, -1, number("k")).len() < 7);
}

#[test]
fn the_leaf_reader_edges() {
    use crate::reader::{IndexReader, LeafReader, PostingsFlags};
    let reader = rich_index();
    assert_eq!((reader.max_doc(), reader.num_docs()), (1, 1));
    assert_eq!(reader.leaves().len(), 1);
    assert!(reader.reader_cache_helper().is_none());
    assert!(reader.core_cache_helper().is_none());
    assert!(reader.live_docs().is_none());
    assert!(reader.index_sort().is_none());
    assert!(reader.term_vectors(0).unwrap().is_none());
    assert!(reader.float_vector_values("body").unwrap().is_none());
    assert!(reader.byte_vector_values("body").unwrap().is_none());
    // a doc-values field read as another type has none
    assert!(reader.numeric_doc_values("sn").unwrap().is_none());
    assert!(reader.sorted_set_doc_values("sd").unwrap().is_none());
    assert!(reader.point_values("body").unwrap().is_none());
    assert!(reader.point_values("absent").unwrap().is_none());

    let terms = reader.terms("body").unwrap().unwrap();
    let mut te = terms.iterator().unwrap();
    // not yet positioned
    assert!(te.term().is_none());
    assert!(te.doc_freq().is_err());
    assert!(te.ord().is_err());
    assert!(te.postings(PostingsFlags::Freqs).is_err());
    te.seek_exact_ord(1).unwrap();
    assert_eq!((te.term(), te.ord().unwrap()), (Some(&b"b"[..]), 1));
    assert!(te.seek_exact_ord(3).is_err());
    assert!(te.seek_exact_ord(-1).is_err());
    // a failed seek leaves it unpositioned; past the end stays at the end
    assert!(!te.try_seek_exact(b"q").unwrap());
    assert!(te.total_term_freq().is_err());
    while te.next().unwrap().is_some() {}
    assert!(te.next().unwrap().is_none());

    // a sorted-set ordinal out of range; one value too many
    let mut ss = reader.sorted_set_doc_values("ss").unwrap().unwrap();
    assert!(ss.lookup_ord(2).is_err());
    assert!(ss.lookup_ord(-1).is_err());
    let mut sn = reader.sorted_numeric_doc_values("sn").unwrap().unwrap();
    assert!(sn.advance_exact(0).unwrap());
    sn.next_value().unwrap();
    sn.next_value().unwrap();
    assert!(sn.next_value().is_err());
    assert!(!sn.advance_exact(1).unwrap());

    // points: outside visits nothing, crossing visits the values, and the
    // estimate follows the relation
    for (name, dims) in [("p1", 1), ("p2", 2)] {
        let p = reader.point_values(name).unwrap().unwrap();
        assert_eq!(p.num_dimensions(), dims);
        let mut out = Visits {
            outside: true,
            ..Visits::default()
        };
        p.intersect(&mut out).unwrap();
        assert!(out.seen.is_empty());
        assert_eq!(p.estimate_point_count(&mut out).unwrap(), 0);
        let mut cross = Visits::default();
        p.intersect(&mut cross).unwrap();
        assert_eq!(cross.seen.len(), 2);
        let width = 4 * usize::try_from(dims).unwrap();
        assert!(cross.seen.iter().all(|(d, v)| *d == 0 && v.len() == width));
        assert_eq!(p.estimate_point_count(&mut cross).unwrap(), 1);
        let mut inside = Visits {
            inside: true,
            ..Visits::default()
        };
        assert_eq!(p.estimate_point_count(&mut inside).unwrap(), 2);
    }
    // one dimension: ascending; more: each dimension's own bounds
    let p1 = reader.point_values("p1").unwrap().unwrap();
    let mut cross = Visits::default();
    p1.intersect(&mut cross).unwrap();
    assert!(cross.seen[0].1 < cross.seen[1].1);
    let p2 = reader.point_values("p2").unwrap().unwrap();
    let enc = |v: i32| (v.cast_unsigned() ^ 0x8000_0000).to_be_bytes();
    assert_eq!(p2.min_packed_value(), [enc(1), enc(2)].concat());
    assert_eq!(p2.max_packed_value(), [enc(4), enc(9)].concat());
}

// ---------------------------------------------------------------------------
// The reader off a term vector
// ---------------------------------------------------------------------------

fn tv_term(term: &[u8], positions: &[i32], offsets: &[(i32, i32)]) -> TermVectorTerm {
    TermVectorTerm {
        term: term.to_vec(),
        freq: i32::try_from(positions.len()).unwrap(),
        positions: Some(positions.to_vec()),
        start_offsets: Some(offsets.iter().map(|o| o.0).collect()),
        end_offsets: Some(offsets.iter().map(|o| o.1).collect()),
        payloads: None,
    }
}

fn tv_field(terms: Vec<TermVectorTerm>) -> TermVectorField {
    TermVectorField {
        field_number: 3,
        has_positions: true,
        has_offsets: true,
        has_payloads: false,
        terms,
    }
}

/// The reader the stream replay builds, as text: `None` when the stream
/// refuses the vector.
fn replayed(vector: &TermVectorField) -> Option<String> {
    use lucene_analysis::TokenStream;
    let mut ts = crate::highlight::TokenStreamFromTermVector::new(vector, -1).ok()?;
    ts.reset().ok()?;
    let mut tokens = Vec::new();
    while ts.increment_token().ok()? {
        let a = ts.attributes();
        tokens.push(token(
            std::str::from_utf8(a.term_bytes()).ok()?,
            a.position_increment(),
            a.start_offset(),
            a.end_offset(),
        ));
    }
    ts.end().ok()?;
    let final_offset = ts.attributes().end_offset();
    let mut mi = MemoryIndex::with_options(true, false);
    mi.add_field_tokens("tv", &stream(tokens, final_offset), 0, 1)
        .ok()?;
    let r = mi.create_reader().ok()?;
    Some(format!("{:?} {:?}", r.doc, r.field_infos))
}

fn direct(vector: &TermVectorField) -> Option<String> {
    MemoryIndex::vector_reader("tv", vector)
        .unwrap()
        .map(|r| format!("{:?} {:?}", r.doc, r.field_infos))
}

/// The reader built off a term vector is the one its replay as a token
/// stream builds -- terms, postings, statistics and norm -- and every case
/// it does not cover is left to the replay.
#[test]
fn a_vector_reader_is_the_replayed_reader() {
    let long_span = (0, i32::from(i16::MAX) + 100);
    let plain = [
        // overlapping tokens (two terms at one position), gaps, repeats
        vec![
            tv_term(b"brown", &[1, 5], &[(4, 9), (30, 35)]),
            tv_term(b"fast", &[1], &[(4, 9)]),
            tv_term(b"quick", &[0, 0, 7], &[(0, 3), (0, 5), (40, 45)]),
        ],
        vec![tv_term("ölmühle".as_bytes(), &[2], &[(1, 8)])],
        // a span longer than Short.MAX_VALUE is cut
        vec![tv_term(b"x", &[0], &[long_span])],
    ];
    for terms in plain {
        let v = tv_field(terms);
        assert!(direct(&v).is_some(), "{v:?}");
        assert_eq!(direct(&v), replayed(&v), "{v:?}");
    }
    let one = || tv_field(vec![tv_term(b"a", &[0], &[(0, 1)])]);
    let mut no_positions = one();
    no_positions.has_positions = false;
    no_positions.terms[0].positions = None;
    let mut no_offsets = one();
    no_offsets.has_offsets = false;
    let mut zero_freq = one();
    zero_freq.terms[0].freq = 0;
    let mut short = one();
    short.terms[0].freq = 2;
    let refused = [
        no_positions,
        no_offsets,
        zero_freq,
        short,
        tv_field(Vec::new()),
        tv_field(vec![tv_term(b"a", &[-1], &[(0, 1)])]),
        tv_field(vec![tv_term(b"a", &[0], &[(-1, 1)])]),
        tv_field(vec![tv_term(b"a", &[0], &[(5, 1)])]),
        tv_field(vec![tv_term(b"a", &[3, 1], &[(0, 1), (2, 3)])]),
        tv_field(vec![tv_term(b"a", &[1, 1], &[(4, 5), (2, 3)])]),
        tv_field(vec![tv_term(&[0xff, 0xfe], &[0], &[(0, 1)])]),
        tv_field(vec![tv_term(&[b'a'; 40_000], &[0], &[(0, 1)])]),
    ];
    for v in refused {
        assert!(
            direct(&v).is_none(),
            "{:?}",
            v.terms.first().map(|t| t.freq)
        );
    }
}
