//! MemoryIndex against the same documents indexed as a segment, under the
//! classic query parser's queries.

use std::sync::Arc;

use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_queryparser::classic::QueryParser;
use lucene_search::memory::*;
use lucene_search::query::*;
use lucene_search::similarities::{Bm25Similarity, ClassicSimilarity, Similarity};

fn term(field: &str, text: &str) -> Clause {
    Clause::Term(TermQuery::new(field, text.as_bytes().to_vec()))
}

/// The query as the boolean a searcher runs: `None` for one that matches
/// nothing.
fn as_boolean(query: &Clause) -> Option<BooleanQuery> {
    match query {
        Clause::MatchNoDocs(_) => None,
        Clause::Boolean(b) => Some((**b).clone()),
        other => {
            let mut b = BooleanQuery::new();
            b.must.push(other.clone());
            Some(b)
        }
    }
}

fn std_analyzer() -> Analyzer {
    Analyzer::new(StandardAnalyzer::new())
}

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
        let parser = QueryParser::new("body", Arc::clone(&analyzer));
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
                let segment = match as_boolean(&clause.clone().rewrite()) {
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
fn render_matches(m: &lucene_search::Result<Option<lucene_search::matches::BoxMatches>>) -> String {
    use lucene_search::matches::{clause_name, BoxMatchesIterator};
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
    use lucene_search::query::{
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
        let parser = QueryParser::new("body", Arc::clone(&analyzer));
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
                            lucene_search::matches::leaf_matches(&s.segments()[0], clause, 0)
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
