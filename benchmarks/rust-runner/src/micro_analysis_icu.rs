//! M12 T12.4's pairs (`benchmarks/micro/java/AnalysisIcuMicro.java` is the
//! Java twin): analysis-icu over `fixtures/corpus/analysis-icu.txt` and the
//! normalization fixture's stress strings, one document per line.

use std::hint::black_box;
use std::time::Duration;

use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerDefinition, CharReader, TokenStream, TokenStreamComponents,
};
use lucene_analysis_icu::{
    Collator, ICUCollationKeyAnalyzer, ICUFoldingFilter, ICUNormalizer2CharFilter,
    ICUNormalizer2Filter, ICUTokenizer, ICUTransformFilter, Transliterator,
};

use super::{consume_stream, measure};

type Sink = Result<TokenStreamComponents, AnalysisError>;
type Wrap = Box<dyn Fn(Box<dyn CharReader>) -> Box<dyn CharReader> + Send + Sync>;

struct Chain(Box<dyn Fn() -> Sink + Send + Sync>, Option<Wrap>);

impl AnalyzerDefinition for Chain {
    fn create_components(&self, _field: &str) -> Sink {
        (self.0)()
    }

    fn init_reader(&self, _field: &str, reader: Box<dyn CharReader>) -> Box<dyn CharReader> {
        match &self.1 {
            Some(w) => w(reader),
            None => reader,
        }
    }
}

fn chain(f: impl Fn() -> Sink + Send + Sync + 'static) -> Analyzer {
    Analyzer::new(Chain(Box::new(f), None))
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

/// `AnalysisIcuMicro.docs`.
fn docs() -> Vec<String> {
    let mut docs = Vec::new();
    for (file, skip_escaped) in [
        ("fixtures/corpus/analysis-icu.txt", false),
        ("fixtures/data/analysis_icu/norm_strings.txt", true),
    ] {
        let text = std::fs::read_to_string(file).unwrap_or_else(|e| panic!("{file}: {e}"));
        docs.extend(
            text.split('\n')
                .filter(|l| !l.is_empty() && !(skip_escaped && l.contains('\\')))
                .map(str::to_string),
        );
    }
    docs
}

pub(super) fn bench_analysis_icu(w: Duration, m: Duration) {
    let docs = docs();
    run(
        "icu_normalizer_nfkc_cf",
        &chain(|| comps(ICUNormalizer2Filter::new(WhitespaceTokenizer::new()))),
        &docs,
        w,
        m,
    );
    run(
        "icu_folding",
        &chain(|| comps(ICUFoldingFilter::new(WhitespaceTokenizer::new()))),
        &docs,
        w,
        m,
    );
    let cf: Wrap = Box::new(|r| Box::new(ICUNormalizer2CharFilter::new(r)));
    run(
        "icu_normalizer_charfilter",
        &Analyzer::new(Chain(
            Box::new(|| comps(WhitespaceTokenizer::new())),
            Some(cf),
        )),
        &docs,
        w,
        m,
    );
    run(
        "icu_tokenizer",
        &chain(|| comps(ICUTokenizer::new())),
        &docs,
        w,
        m,
    );
    for (name, id) in [
        ("icu_transform_any_latin", "Any-Latin"),
        ("icu_transform_trad_simp", "Traditional-Simplified"),
    ] {
        let t = Transliterator::get_instance(id, 0).unwrap();
        run(
            name,
            &chain(move || comps(ICUTransformFilter::new(WhitespaceTokenizer::new(), t.clone()))),
            &docs,
            w,
            m,
        );
    }
    let root = Collator::root().unwrap();
    run(
        "icu_collation_key",
        &Analyzer::new(ICUCollationKeyAnalyzer::new(root)),
        &docs,
        w,
        m,
    );
    let mut identical = Collator::get_instance("de@collation=phonebook").unwrap();
    identical
        .set_strength(lucene_analysis_icu::icu4j::coll::collator::IDENTICAL)
        .unwrap();
    identical.set_alternate_handling_shifted(true);
    run(
        "icu_collation_key_phonebook_identical",
        &Analyzer::new(ICUCollationKeyAnalyzer::new(identical)),
        &docs,
        w,
        m,
    );
}
