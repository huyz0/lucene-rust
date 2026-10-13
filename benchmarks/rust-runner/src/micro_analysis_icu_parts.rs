//! M12's ICU gaps taken apart (`benchmarks/micro/java/AnalysisIcuPartsMicro.java`
//! is the Java twin): for each `--bench analysis_icu` case below Lucene, the
//! tokenizer alone and the ICU kernel alone over the same inputs, so that
//! the chain's ratio splits into the part each engine spends where. Units
//! are the chain's: tokens (a term, or a document for the collation key),
//! and break boundaries for the break iterator.

use std::hint::black_box;
use std::time::Duration;

use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerDefinition, KeywordTokenizer, TokenStream,
    TokenStreamComponents,
};
use lucene_analysis_icu::icu4j::coll::collator::KeyScratch;
use lucene_analysis_icu::icu4j::normalizer2::QuickCheck;
use lucene_analysis_icu::icu4j::rbbi::RuleBasedBreakIterator;
use lucene_analysis_icu::icu4j::translit::Position;
use lucene_analysis_icu::{Collator, Normalizer2, Transliterator};

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

fn run_chain(name: &str, a: &Analyzer, docs: &[String], w: Duration, m: Duration) {
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

/// The whitespace tokenizer's terms of every document, once.
fn terms(docs: &[String]) -> Vec<String> {
    let a = chain(|| comps(WhitespaceTokenizer::new()));
    let mut out = Vec::new();
    for text in docs {
        let mut ts = a.token_stream("body", text).unwrap();
        ts.reset().unwrap();
        while ts.increment_token().unwrap() {
            out.push(ts.attributes().term().to_string());
        }
        ts.end().unwrap();
        ts.close().unwrap();
    }
    out
}

fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

pub(super) fn bench_analysis_icu_parts(w: Duration, m: Duration) {
    let docs = docs();
    let terms = terms(&docs);
    let terms16: Vec<Vec<u16>> = terms.iter().map(|t| utf16(t)).collect();

    // The tokenizers alone.
    run_chain(
        "whitespace_only",
        &chain(|| comps(WhitespaceTokenizer::new())),
        &docs,
        w,
        m,
    );
    run_chain(
        "keyword_only",
        &chain(|| comps(KeywordTokenizer::new())),
        &docs,
        w,
        m,
    );

    // icu_folding's kernel: utr30 per term, as Java's filter calls it
    // (quickCheck, then normalize unless YES), over UTF-16 terms.
    let fold = Normalizer2::utr30();
    let mut buf = Vec::new();
    measure("fold_kernel_utf16", w, m, || {
        for t in &terms16 {
            if fold.quick_check(black_box(t)) != QuickCheck::Yes {
                buf.clear();
                fold.normalize_to(t, &mut buf);
                black_box(&buf);
            }
        }
        terms16.len() as u64
    });
    // The same through the Rust filter's own path from a UTF-8 term.
    let ascii = fold.ascii_map().copied();
    let mut units = Vec::new();
    measure("fold_kernel_from_utf8", w, m, || {
        for t in &terms {
            let t = black_box(t.as_str());
            if fold.quick_yes_utf8(t) {
                continue;
            }
            if t.is_ascii() {
                if let Some(map) = &ascii {
                    black_box(t.bytes().any(|b| map[usize::from(b)] != b));
                    continue;
                }
            }
            units.clear();
            units.extend(t.encode_utf16());
            fold.normalize_to(&units, &mut buf);
            black_box(buf != units);
        }
        terms.len() as u64
    });

    // icu_transform_trad_simp's kernel: filteredTransliterate per term.
    let ts = Transliterator::get_instance("Traditional-Simplified", 0).unwrap();
    let mut text = Vec::new();
    measure("trad_simp_kernel_utf16", w, m, || {
        for t in &terms16 {
            text.clear();
            text.extend_from_slice(black_box(t));
            let len = text.len() as i32;
            let mut pos = Position {
                context_start: 0,
                context_limit: len,
                start: 0,
                limit: len,
            };
            ts.filtered_transliterate(&mut text, &mut pos).unwrap();
            black_box(&text);
        }
        terms16.len() as u64
    });

    // icu_collation_key's kernel: the root key of each whole document.
    let root = Collator::root().unwrap();
    let mut scratch = KeyScratch::default();
    let mut key = Vec::new();
    measure("coll_key_kernel_utf8", w, m, || {
        for d in &docs {
            key.clear();
            root.raw_collation_key_into(black_box(d), &mut scratch, &mut key);
            black_box(&key);
        }
        docs.len() as u64
    });
    let docs16: Vec<Vec<u16>> = docs.iter().map(|d| utf16(d)).collect();
    measure("coll_key_kernel_utf16", w, m, || {
        for d in &docs16 {
            black_box(root.raw_collation_key_utf16(black_box(d)));
        }
        docs16.len() as u64
    });

    // icu_tokenizer's kernel: Lucene's Default.brk over each document.
    let mut bi = RuleBasedBreakIterator::from_compiled_rules(include_bytes!(
        "../../../crates/lucene-analysis-icu/src/resources/Default.brk"
    ))
    .unwrap();
    measure("default_brk_kernel", w, m, || {
        let mut n = 0u64;
        for d in &docs16 {
            bi.set_text(black_box(d));
            let mut b = bi.first();
            while b >= 0 {
                n += 1;
                b = bi.next();
            }
        }
        n
    });
}
