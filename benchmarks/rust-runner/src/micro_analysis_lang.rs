//! M11 part 3's pair (`benchmarks/micro/java/AnalysisLangMicro.java` is the
//! Java twin): each language analyzer over its own language's lines of
//! `fixtures/corpus/analysis-lang.txt` (repeated 200 times as documents; the
//! whole multilingual corpus would mostly measure a stemmer missing on
//! foreign words), and synonym filters over
//! `fixtures/corpus/analysis-synonym.txt` repeated 50 times.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use lucene_analysis::core_analysis::FlattenGraphFilter;
use lucene_analysis::lang::{ar, de, el, es, fr, hi, pt, ru};
use lucene_analysis::synonym::{SolrSynonymParser, SynonymFilter, SynonymGraphFilter};
use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerDefinition, LowerCaseFilter, StandardTokenizer, TokenStream,
    TokenStreamComponents,
};

use super::{consume_stream, measure};

type Sink = Result<TokenStreamComponents, AnalysisError>;

struct Chain(Box<dyn Fn() -> Sink + Send + Sync>);

impl AnalyzerDefinition for Chain {
    fn create_components(&self, _field: &str) -> Sink {
        (self.0)()
    }
}

fn chain(f: impl Fn() -> Sink + Send + Sync + 'static) -> Analyzer {
    Analyzer::new(Chain(Box::new(f)))
}

fn comps(sink: impl TokenStream + 'static) -> Sink {
    Ok(TokenStreamComponents::new(sink))
}

fn docs(file: &str) -> Vec<String> {
    let text = std::fs::read_to_string(format!("fixtures/corpus/{file}")).unwrap();
    let lines: Vec<String> = text.lines().map(str::to_string).collect();
    (0..50).flat_map(|_| lines.clone()).collect()
}

/// The corpus's lines numbered `nums` (1-based), repeated 200 times.
fn lines(nums: &[usize]) -> Vec<String> {
    let text = std::fs::read_to_string("fixtures/corpus/analysis-lang.txt").unwrap();
    let all: Vec<&str> = text.lines().collect();
    (0..200)
        .flat_map(|_| nums.iter().map(|&n| all[n - 1].to_string()))
        .collect()
}

fn run(name: &str, a: &Analyzer, docs: &[String], w: Duration, m: Duration) {
    measure(name, w, m, || {
        let mut tokens = 0u64;
        for text in docs {
            let mut ts = a.token_stream("body", black_box(text)).unwrap();
            tokens += consume_stream(&mut ts);
        }
        tokens
    });
}

pub(super) fn bench_analysis_lang(w: Duration, mt: Duration) {
    let (de_l, fr_l, es_l, ar_l) = (lines(&[1, 2]), lines(&[3, 4]), lines(&[5]), lines(&[28]));
    let (hi_l, el_l, ru_l, pt_l) = (lines(&[31]), lines(&[36]), lines(&[16]), lines(&[7]));
    let syn = docs("analysis-synonym.txt");
    let rules = std::fs::read_to_string("fixtures/corpus/synonyms-solr.txt").unwrap();
    let parser_analyzer = chain(|| comps(LowerCaseFilter::new(WhitespaceTokenizer::new())));
    let mut p = SolrSynonymParser::new(true, true, &parser_analyzer);
    p.parse(&rules).unwrap();
    let map = Arc::new(p.build().unwrap());
    let (m1, m2, m3) = (Arc::clone(&map), Arc::clone(&map), map);
    let cases: Vec<(&str, Analyzer, &Vec<String>)> = vec![
        (
            "german",
            Analyzer::new(de::GermanAnalyzer::default()),
            &de_l,
        ),
        (
            "french",
            Analyzer::new(fr::FrenchAnalyzer::default()),
            &fr_l,
        ),
        (
            "spanish",
            Analyzer::new(es::SpanishAnalyzer::default()),
            &es_l,
        ),
        (
            "arabic",
            Analyzer::new(ar::ArabicAnalyzer::default()),
            &ar_l,
        ),
        ("hindi", Analyzer::new(hi::HindiAnalyzer::default()), &hi_l),
        ("greek", Analyzer::new(el::GreekAnalyzer::default()), &el_l),
        (
            "russian_light",
            chain(|| {
                comps(ru::RussianLightStemFilter::new(LowerCaseFilter::new(
                    StandardTokenizer::new(),
                )))
            }),
            &ru_l,
        ),
        (
            "portuguese_rslp",
            chain(|| {
                comps(pt::PortugueseStemFilter::new(LowerCaseFilter::new(
                    StandardTokenizer::new(),
                )))
            }),
            &pt_l,
        ),
        (
            "synonym_graph",
            chain(move || {
                comps(SynonymGraphFilter::new(
                    LowerCaseFilter::new(WhitespaceTokenizer::new()),
                    Arc::clone(&m1),
                    true,
                ))
            }),
            &syn,
        ),
        (
            "synonym_graph_flatten",
            chain(move || {
                comps(FlattenGraphFilter::new(SynonymGraphFilter::new(
                    LowerCaseFilter::new(WhitespaceTokenizer::new()),
                    Arc::clone(&m2),
                    true,
                )))
            }),
            &syn,
        ),
        (
            "synonym_legacy",
            chain(move || {
                comps(SynonymFilter::new(
                    LowerCaseFilter::new(WhitespaceTokenizer::new()),
                    Arc::clone(&m3),
                    true,
                ))
            }),
            &syn,
        ),
    ];
    // `MICRO_CASE=<name>` runs one case (for profiling).
    let only = std::env::var("MICRO_CASE").ok();
    for (name, a, corpus) in &cases {
        if only.as_deref().is_some_and(|o| o != *name) {
            continue;
        }
        run(name, a, corpus, w, mt);
    }
}
