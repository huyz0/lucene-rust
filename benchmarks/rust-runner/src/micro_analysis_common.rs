//! M11's analysis-common pair (`benchmarks/micro/java/AnalysisCommonMicro.java`
//! is the Java twin): one analyzer per ported package over `SweepMicro`'s
//! documents, each token's term bytes read as `IndexingChain` reads them.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use lucene_analysis::charfilter::HTMLStripCharFilter;
use lucene_analysis::cjk::CJKAnalyzer;
use lucene_analysis::core_analysis::{SimpleAnalyzer, WhitespaceAnalyzer};
use lucene_analysis::email::UAX29URLEmailAnalyzer;
use lucene_analysis::en::{EnglishAnalyzer, KStemFilter};
use lucene_analysis::miscellaneous::{self as m, AsciiFoldingTokenFilter, WordDelimiterGraphFilter};
use lucene_analysis::ngram::NGramTokenizer;
use lucene_analysis::pattern::PatternTokenizer;
use lucene_analysis::reader::CharReader;
use lucene_analysis::shingle::ShingleFilter;
use lucene_analysis::util::{JavaPattern, WhitespaceTokenizer};
use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerDefinition, LowerCaseFilter, StandardTokenizer,
    TokenStream, TokenStreamComponents,
};

use super::{analysis_docs, consume_stream, measure, multilingual_docs};

type Sink = Result<TokenStreamComponents, AnalysisError>;

/// `AnalysisCommonMicro.chain`.
struct Chain {
    components: Box<dyn Fn() -> Sink + Send + Sync>,
    char_filter: bool,
}

impl AnalyzerDefinition for Chain {
    fn create_components(&self, _field: &str) -> Sink {
        (self.components)()
    }

    fn init_reader(&self, _field: &str, reader: Box<dyn CharReader>) -> Box<dyn CharReader> {
        if self.char_filter {
            Box::new(HTMLStripCharFilter::new(reader))
        } else {
            reader
        }
    }
}

fn chain(char_filter: bool, f: impl Fn() -> Sink + Send + Sync + 'static) -> Analyzer {
    Analyzer::new(Chain { components: Box::new(f), char_filter })
}

fn comps(sink: impl TokenStream + 'static) -> Sink {
    Ok(TokenStreamComponents::new(sink))
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

pub(super) fn bench_analysis_common(w: Duration, mt: Duration) {
    let docs = analysis_docs();
    let multi = multilingual_docs();
    let html: Vec<String> = docs.iter().map(|d| format!("<p>{d}</p>")).collect();
    let wdgf = m::GENERATE_WORD_PARTS
        | m::GENERATE_NUMBER_PARTS
        | m::SPLIT_ON_CASE_CHANGE
        | m::SPLIT_ON_NUMERICS
        | m::STEM_ENGLISH_POSSESSIVE;
    let pattern = Arc::new(JavaPattern::compile("[ ,.]+").unwrap());
    let cases: Vec<(&str, Analyzer, &Vec<String>)> = vec![
        ("whitespace", Analyzer::new(WhitespaceAnalyzer::default()), &docs),
        ("simple", Analyzer::new(SimpleAnalyzer), &docs),
        ("english", Analyzer::new(EnglishAnalyzer::default()), &docs),
        (
            "wdgf",
            chain(false, move || comps(WordDelimiterGraphFilter::new(WhitespaceTokenizer::new(), wdgf, None)?)),
            &docs,
        ),
        ("ngram_2_3", chain(false, || comps(NGramTokenizer::new(2, 3)?)), &docs),
        ("shingle", chain(false, || comps(ShingleFilter::new(StandardTokenizer::new(), 2, 2)?)), &docs),
        (
            "kstem",
            chain(false, || comps(KStemFilter::new(LowerCaseFilter::new(StandardTokenizer::new())))),
            &docs,
        ),
        (
            "pattern",
            chain(false, move || comps(PatternTokenizer::new(&pattern, -1)?)),
            &docs,
        ),
        ("html_strip", chain(true, || comps(WhitespaceTokenizer::new())), &html),
        (
            "ascii_folding_multilingual",
            chain(false, || comps(AsciiFoldingTokenFilter::new(StandardTokenizer::new(), false))),
            &multi,
        ),
        ("cjk_multilingual", Analyzer::new(CJKAnalyzer::default()), &multi),
        ("uax29_url_email_multilingual", Analyzer::new(UAX29URLEmailAnalyzer::default()), &multi),
    ];
    for (name, a, corpus) in &cases {
        run(name, a, corpus, w, mt);
    }
}
