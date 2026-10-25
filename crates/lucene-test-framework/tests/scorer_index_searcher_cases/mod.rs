#![allow(dead_code)]
use lucene_analysis::standard::StandardAnalyzer;
use lucene_analysis::Analyzer;
use lucene_index::document::{Document, Store, StringField, TextField};
use lucene_search::collector::{ScoreMode, ScoringCollector, TotalHitsRelation};
use lucene_search::collectors::CollectorManager;
use lucene_search::extended_query::{BlendedRewrite, BlendedTermQuery};
use lucene_search::query::{
    BooleanQuery, BoostQuery, Clause, ConstantScoreQuery, DisjunctionMaxQuery, MatchAllDocsQuery,
    PhraseQuery, TermQuery,
};
use lucene_search::top_docs::TopDocs;
use lucene_search::Result;
use lucene_test_framework::search::test_index::TestIndex;
use std::sync::Arc;

pub const TEXT: [&str; 12] = [
    "a b a", "b c", "a c b", "a a c", "d b", "a b c", "a c", "b d", "c d", "a a b", "a d c",
    "a b d",
];
pub fn index(repeats: usize) -> TestIndex {
    let segments: Vec<Vec<Document>> = (0..3)
        .map(|segment| {
            let mut docs = Vec::new();
            for _ in 0..repeats {
                for (i, text) in TEXT.iter().enumerate().skip(segment * 4).take(4) {
                    let mut doc = Document::new();
                    doc.add(StringField::new("id", i.to_string(), Store::No));
                    doc.add(TextField::new("body", *text, Store::No));
                    docs.push(doc);
                }
            }
            docs
        })
        .collect();
    TestIndex::build(
        Arc::new(Analyzer::new(StandardAnalyzer::default())),
        &segments,
        &[("id", "1"), ("id", "6"), ("id", "10")],
        &["body"],
    )
}
pub fn term(value: &str) -> Clause {
    TermQuery::new("body", value).into()
}
pub fn top_level(clause: Clause) -> BooleanQuery {
    match clause {
        Clause::Boolean(b) => *b,
        q => BooleanQuery {
            must: vec![q],
            ..BooleanQuery::default()
        },
    }
}
pub fn clauses() -> Vec<Clause> {
    let phrase: Clause = PhraseQuery::new("body", [b"a".to_vec(), b"b".to_vec()]).into();
    vec![
        term("a"),
        BooleanQuery {
            should: vec![term("a"), term("b")],
            ..BooleanQuery::default()
        }
        .into(),
        BooleanQuery {
            must: vec![term("a"), term("b")],
            ..BooleanQuery::default()
        }
        .into(),
        BooleanQuery {
            must: vec![term("a")],
            should: vec![term("b")],
            ..BooleanQuery::default()
        }
        .into(),
        BooleanQuery {
            must: vec![term("a")],
            must_not: vec![term("d")],
            ..BooleanQuery::default()
        }
        .into(),
        phrase.clone(),
        BooleanQuery {
            must: vec![term("a")],
            filter: vec![phrase.clone()],
            ..BooleanQuery::default()
        }
        .into(),
        ConstantScoreQuery::new(phrase, 1.0).into(),
        MatchAllDocsQuery::new(0).into(),
        term("absent"),
    ]
}
pub fn threshold_clauses() -> Vec<(&'static str, Clause, usize)> {
    vec![
        ("direct", term("a"), 6),
        (
            "dismax",
            DisjunctionMaxQuery::new([term("a"), term("b")], 0.0).into(),
            8,
        ),
        (
            "dismax_tie",
            DisjunctionMaxQuery::new([term("a"), term("b")], 0.5).into(),
            8,
        ),
        (
            "dismax_singleton",
            DisjunctionMaxQuery::new([term("a"), term("absent")], 0.0).into(),
            6,
        ),
        (
            "dismax_singleton_tie",
            DisjunctionMaxQuery::new([term("a"), term("absent")], 0.5).into(),
            6,
        ),
        (
            "blended",
            BlendedTermQuery::new(
                [("body", b"a".to_vec(), 1.0), ("body", b"b".to_vec(), 1.0)],
                BlendedRewrite::DisjunctionMax(0.0),
            )
            .unwrap()
            .into(),
            8,
        ),
        (
            "single_phrase",
            PhraseQuery::new("body", [b"a".to_vec()]).into(),
            6,
        ),
        (
            "required_filter",
            BooleanQuery {
                must: vec![term("a")],
                filter: vec![term("b")],
                ..BooleanQuery::default()
            }
            .into(),
            5,
        ),
        (
            "required_exclusion",
            BooleanQuery {
                must: vec![term("a")],
                must_not: vec![term("d")],
                ..BooleanQuery::default()
            }
            .into(),
            5,
        ),
    ]
}
pub fn constant_threshold_clauses() -> Vec<(&'static str, Clause)> {
    let constant = || -> Clause { ConstantScoreQuery::new(term("a"), 1.0).into() };
    vec![
        ("constant_term", constant()),
        (
            "boosted_constant_term",
            BoostQuery::new(constant(), 2.0).into(),
        ),
        (
            "nested_boosted_constant_term",
            BoostQuery::new(
                ConstantScoreQuery::new(BoostQuery::new(constant(), 3.0), 1.0),
                2.0,
            )
            .into(),
        ),
        (
            "constant_required_filter",
            BooleanQuery {
                must: vec![constant()],
                filter: vec![term("b")],
                ..BooleanQuery::default()
            }
            .into(),
        ),
        (
            "constant_required_exclusion",
            BooleanQuery {
                must: vec![constant()],
                must_not: vec![term("d")],
                ..BooleanQuery::default()
            }
            .into(),
        ),
        (
            "constant_disjunction",
            BooleanQuery {
                should: vec![constant(), ConstantScoreQuery::new(term("b"), 1.0).into()],
                ..BooleanQuery::default()
            }
            .into(),
        ),
        (
            "constant_dismax",
            DisjunctionMaxQuery::new(
                [constant(), ConstantScoreQuery::new(term("b"), 1.0).into()],
                0.0,
            )
            .into(),
        ),
        (
            "constant_dismax_tie",
            DisjunctionMaxQuery::new(
                [constant(), ConstantScoreQuery::new(term("b"), 1.0).into()],
                0.5,
            )
            .into(),
        ),
        (
            "constant_dismax_singleton",
            DisjunctionMaxQuery::new([constant(), term("absent")], 0.0).into(),
        ),
        (
            "constant_dismax_singleton_tie",
            DisjunctionMaxQuery::new([constant(), term("absent")], 0.5).into(),
        ),
        (
            "constant_dismax_mixed",
            DisjunctionMaxQuery::new([constant(), term("b")], 0.0).into(),
        ),
    ]
}
#[derive(Debug)]
pub struct Hits {
    pub mode: ScoreMode,
    pub hits: Vec<(i32, u32)>,
}
impl Hits {
    pub fn new(mode: ScoreMode) -> Self {
        Self {
            mode,
            hits: Vec::new(),
        }
    }
}
impl ScoringCollector for Hits {
    fn collect(&mut self, doc: i32, score: f32) {
        self.hits.push((
            doc,
            if self.mode.needs_scores() {
                score.to_bits()
            } else {
                0
            },
        ));
    }
    fn score_mode(&self) -> ScoreMode {
        self.mode
    }
}
pub fn hits_text(hits: &[(i32, u32)]) -> String {
    hits.iter()
        .map(|(doc, bits)| format!("{doc}:{bits:08x}"))
        .collect::<Vec<_>>()
        .join(",")
}
pub fn top_text(top: &TopDocs) -> String {
    let relation = match top.total_hits.relation {
        TotalHitsRelation::EqualTo => "EQUAL_TO",
        TotalHitsRelation::GreaterThanOrEqualTo => "GREATER_THAN_OR_EQUAL_TO",
    };
    let hits: Vec<_> = top
        .score_docs
        .iter()
        .map(|s| (s.doc, s.score.to_bits()))
        .collect();
    format!("{}:{relation}:{}", top.total_hits.value, hits_text(&hits))
}
pub struct Manager;
impl CollectorManager for Manager {
    type Collector = Hits;
    type Output = Vec<(i32, u32)>;
    fn new_collector(&self) -> Result<Hits> {
        Ok(Hits::new(ScoreMode::Complete))
    }
    fn reduce(&self, collectors: Vec<Hits>) -> Result<Self::Output> {
        let mut hits: Vec<_> = collectors.into_iter().flat_map(|c| c.hits).collect();
        hits.sort_by_key(|h| h.0);
        Ok(hits)
    }
}
