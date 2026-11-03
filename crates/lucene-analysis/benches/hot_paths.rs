//! Per-token cost of the analysis chains an index spends its time in: the
//! tokenizers, the common filters, and the chains OpenSearch builds from
//! them, each over the same generated English-like and multilingual text.
//!
//! Run with: `cargo bench -p lucene-analysis --bench hot_paths`
#![allow(clippy::arithmetic_side_effects)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use lucene_analysis::factory::CustomAnalyzer;
use lucene_analysis::{Analyzer, StandardAnalyzer, TokenStream};

/// English words with the shapes the filters branch on: plurals, `-ing`,
/// `-ed`, `-ly`, `-ation`, possessives, capitals, digits, hyphens, accents.
const WORDS: &[&str] = &[
    "the",
    "of",
    "and",
    "to",
    "in",
    "is",
    "that",
    "for",
    "it",
    "as",
    "was",
    "with",
    "be",
    "by",
    "on",
    "not",
    "he",
    "this",
    "are",
    "or",
    "his",
    "from",
    "at",
    "which",
    "but",
    "have",
    "an",
    "had",
    "they",
    "you",
    "were",
    "their",
    "one",
    "all",
    "we",
    "can",
    "her",
    "has",
    "there",
    "been",
    "if",
    "more",
    "when",
    "will",
    "would",
    "who",
    "so",
    "no",
    "Search",
    "Engine",
    "indexing",
    "indexed",
    "indexes",
    "documents",
    "document's",
    "queries",
    "querying",
    "analysis",
    "analyzers",
    "tokenization",
    "relational",
    "conditional",
    "rationalization",
    "happily",
    "running",
    "runner",
    "ran",
    "connections",
    "connected",
    "connecting",
    "generalizations",
    "oscillators",
    "hopefulness",
    "effective",
    "electricity",
    "caresses",
    "ponies",
    "cats",
    "agreed",
    "plastered",
    "motoring",
    "sing",
    "conflated",
    "troubled",
    "sized",
    "hopping",
    "tanned",
    "falling",
    "hissing",
    "fizzed",
    "failing",
    "filing",
    "happy",
    "sky",
    "OpenSearch",
    "Lucene",
    "Wi-Fi",
    "e-mail",
    "PowerShot",
    "SD500",
    "2024",
    "3.14",
    "1,000",
    "naïve",
    "café",
    "résumé",
    "Zürich",
    "Straße",
    "über",
    "jalapeño",
    "façade",
    "London",
    "Paris",
    "John's",
    "company's",
    "users'",
    "isn't",
    "don't",
    "it's",
    "O'Neil",
    "configuration",
    "performance",
    "benchmark",
    "microbenchmarks",
    "throughput",
    "latency",
    "allocation",
    "allocations",
    "memory",
    "buffer",
    "buffers",
    "hashing",
    "hashes",
    "segments",
    "merging",
    "merged",
    "postings",
    "positions",
    "offsets",
    "payloads",
    "frequency",
    "frequencies",
    "norms",
    "scoring",
    "scored",
    "similarity",
    "BM25",
];

const MULTI: &[&str] = &[
    "日本語",
    "テキスト",
    "ひらがな",
    "한국어",
    "ภาษาไทย",
    "Москва",
    "naïve",
    "😀",
    "3.14",
    "e-mail",
    "user@example.com",
    "中国人",
    "العربية",
    "Straße",
    "1,000",
    "Ελληνικά",
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// 500 documents of 40-160 words, Zipf-ish (the minimum of two draws), with
/// commas and full stops.
fn english_docs() -> Vec<String> {
    let mut r = Rng(0x2468_ACE0_1357_9BDF);
    (0..500)
        .map(|_| {
            let n = 40 + (r.next() % 120) as usize;
            let mut s = String::new();
            for i in 0..n {
                if i > 0 {
                    s.push(' ');
                }
                let x = r.next();
                let a = (x % WORDS.len() as u64) as usize;
                let b = ((x >> 20) % WORDS.len() as u64) as usize;
                s.push_str(WORDS[a.min(b)]);
                match x >> 58 {
                    0..=2 => s.push(','),
                    3 => s.push('.'),
                    _ => {}
                }
            }
            s
        })
        .collect()
}

fn multilingual_docs() -> Vec<String> {
    let mut r = Rng(0x1357_9BDF_2468_ACE0);
    (0..250)
        .map(|_| {
            let n = 40 + (r.next() % 120) as usize;
            let mut s = String::new();
            for i in 0..n {
                if i > 0 {
                    s.push(' ');
                }
                let x = r.next();
                if x & 1 == 0 {
                    s.push_str(MULTI[((x >> 8) % MULTI.len() as u64) as usize]);
                } else {
                    s.push_str(WORDS[((x >> 8) % WORDS.len() as u64) as usize]);
                }
            }
            s
        })
        .collect()
}

/// Drains a stream as `IndexingChain` does: every token's term bytes.
fn consume(ts: &mut dyn TokenStream) -> u64 {
    let mut n = 0u64;
    ts.reset().unwrap();
    while ts.increment_token().unwrap() {
        n += ts.attributes().term_bytes().len() as u64;
    }
    ts.end().unwrap();
    ts.close().unwrap();
    n
}

fn custom(
    char_filters: &[&str],
    tokenizer: (&str, &[&str]),
    filters: &[(&str, &[&str])],
) -> Analyzer {
    let mut b = CustomAnalyzer::builder();
    for cf in char_filters {
        b = b.add_char_filter(cf, &[]).unwrap();
    }
    b = b.with_tokenizer(tokenizer.0, tokenizer.1).unwrap();
    for (name, params) in filters {
        b = b.add_token_filter(name, params).unwrap();
    }
    b.build().unwrap().into_analyzer()
}

fn bench_chains(c: &mut Criterion) {
    let en = english_docs();
    let multi = multilingual_docs();
    let html: Vec<String> = en
        .iter()
        .map(|d| format!("<p>{d} &amp; <b>more</b></p>"))
        .collect();
    let std_tok: (&str, &[&str]) = ("standard", &[]);
    let ws: (&str, &[&str]) = ("whitespace", &[]);
    let lc: (&str, &[&str]) = ("lowercase", &[]);
    let cases: Vec<(&str, Analyzer, &Vec<String>)> = vec![
        (
            "standard_analyzer",
            Analyzer::new(StandardAnalyzer::new()),
            &en,
        ),
        (
            "standard_analyzer_multi",
            Analyzer::new(StandardAnalyzer::new()),
            &multi,
        ),
        ("tok_standard", custom(&[], std_tok, &[]), &en),
        ("tok_whitespace", custom(&[], ws, &[]), &en),
        ("tok_keyword", custom(&[], ("keyword", &[]), &[]), &en),
        ("tok_letter", custom(&[], ("letter", &[]), &[]), &en),
        ("tok_classic", custom(&[], ("classic", &[]), &[]), &en),
        ("tok_uax29", custom(&[], ("uax29URLEmail", &[]), &[]), &en),
        (
            "tok_pattern",
            custom(&[], ("pattern", &["pattern", "[ ,.]+"]), &[]),
            &en,
        ),
        (
            "tok_edge_ngram",
            custom(
                &[],
                ("edgeNGram", &["minGramSize", "1", "maxGramSize", "10"]),
                &[],
            ),
            &en,
        ),
        ("std_lowercase", custom(&[], std_tok, &[lc]), &en),
        ("std_lowercase_multi", custom(&[], std_tok, &[lc]), &multi),
        (
            "std_lc_stop",
            custom(&[], std_tok, &[lc, ("stop", &[])]),
            &en,
        ),
        (
            "std_lc_porter",
            custom(&[], std_tok, &[lc, ("porterStem", &[])]),
            &en,
        ),
        (
            "std_lc_kstem",
            custom(&[], std_tok, &[lc, ("kStem", &[])]),
            &en,
        ),
        (
            "std_lc_snowball",
            custom(
                &[],
                std_tok,
                &[lc, ("snowballPorter", &["language", "English"])],
            ),
            &en,
        ),
        (
            "std_possessive",
            custom(&[], std_tok, &[("englishPossessive", &[])]),
            &en,
        ),
        (
            "std_ascii_folding",
            custom(&[], std_tok, &[("asciiFolding", &[])]),
            &multi,
        ),
        (
            "std_lc_edge_ngram",
            custom(
                &[],
                std_tok,
                &[
                    lc,
                    ("edgeNGram", &["minGramSize", "1", "maxGramSize", "10"]),
                ],
            ),
            &en,
        ),
        (
            "std_lc_shingle",
            custom(&[], std_tok, &[lc, ("shingle", &[])]),
            &en,
        ),
        (
            "std_lc_unique",
            custom(&[], std_tok, &[lc, ("removeDuplicates", &[])]),
            &en,
        ),
        (
            "std_lc_length",
            custom(&[], std_tok, &[lc, ("length", &["min", "2", "max", "20"])]),
            &en,
        ),
        (
            "std_lc_trim",
            custom(&[], std_tok, &[lc, ("trim", &[])]),
            &en,
        ),
        (
            "std_lc_reverse",
            custom(&[], std_tok, &[lc, ("reverseString", &[])]),
            &en,
        ),
        (
            "ws_wdgf",
            custom(&[], ws, &[("wordDelimiterGraph", &[])]),
            &en,
        ),
        (
            "ws_wdgf_lc",
            custom(&[], ws, &[("wordDelimiterGraph", &[]), lc]),
            &en,
        ),
        (
            "ws_lc_ngram",
            custom(
                &[],
                ws,
                &[lc, ("nGram", &["minGramSize", "2", "maxGramSize", "3"])],
            ),
            &en,
        ),
        ("html_std_lc", custom(&["htmlStrip"], std_tok, &[lc]), &html),
        (
            "english_chain",
            custom(
                &[],
                std_tok,
                &[
                    ("englishPossessive", &[]),
                    lc,
                    ("stop", &[]),
                    ("porterStem", &[]),
                ],
            ),
            &en,
        ),
    ];
    let mut g = c.benchmark_group("analysis");
    for (name, a, docs) in &cases {
        let bytes: usize = docs.iter().map(String::len).sum();
        g.throughput(Throughput::Bytes(bytes as u64));
        g.bench_function(*name, |b| {
            b.iter(|| {
                let mut n = 0u64;
                for d in docs.iter() {
                    let mut ts = a.token_stream("body", black_box(d)).unwrap();
                    n += consume(&mut ts);
                }
                n
            })
        });
    }
    g.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20).warm_up_time(std::time::Duration::from_millis(500)).measurement_time(std::time::Duration::from_secs(2));
    targets = bench_chains
}
criterion_main!(benches);
