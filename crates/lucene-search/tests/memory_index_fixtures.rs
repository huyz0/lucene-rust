//! M13 T13.5: `MemoryIndex` against Lucene 10.5.0
//! (`fixtures/src/GenMemoryIndex.java`): every document of
//! `fixtures/corpus/memory-index-docs.txt` under eight configurations
//! (analyzers, similarities, offsets, payloads, gaps), its `toStringDebug()`
//! and the score of every query of `memory-index-queries.txt` bit for bit;
//! then the generator's named scenarios (refusals, points, doc values, field-exists,
//! keyword streams, reset).

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use lucene_analysis::core_analysis::WhitespaceAnalyzer;
use lucene_analysis::payloads::{DelimitedPayloadTokenFilter, IdentityEncoder};
use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerDefinition, StandardAnalyzer, TokenStream,
    TokenStreamComponents,
};
use lucene_index::document::{
    BinaryDocValuesField, LongPoint, NumericDocValuesField, Store, StringField,
};
use lucene_search::memory::MemoryIndex;
use lucene_search::query::{Clause, FieldExistsQuery, MatchAllDocsQuery, PointsRangeQuery};
use lucene_search::queryparser::classic::QueryParser;
use lucene_search::similarities::{
    Bm25Similarity, BooleanSimilarity, ClassicSimilarity, Similarity,
};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// `AnalysisRows.esc` undone.
fn unesc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('u') => {
                let h: String = it.by_ref().take(4).collect();
                out.push(char::from_u32(u32::from_str_radix(&h, 16).unwrap()).unwrap());
            }
            Some(o) => out.push(o),
            None => out.push('\\'),
        }
    }
    out
}

/// `GenMemoryIndex.gapped()`: `StandardAnalyzer` with gaps 100 and 10.
struct Gapped(StandardAnalyzer);

impl AnalyzerDefinition for Gapped {
    fn create_components(&self, field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        self.0.create_components(field)
    }

    fn normalize(&self, field: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        self.0.normalize(field, input)
    }

    fn position_increment_gap(&self, _field: &str) -> i32 {
        100
    }

    fn offset_gap(&self, _field: &str) -> i32 {
        10
    }
}

/// `GenMemoryIndex.payloads()`: whitespace, then
/// `DelimitedPayloadTokenFilter('|', IdentityEncoder)`.
struct Payloads;

impl AnalyzerDefinition for Payloads {
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(
            DelimitedPayloadTokenFilter::new(
                WhitespaceTokenizer::new(),
                u16::from(b'|'),
                IdentityEncoder,
            ),
        ))
    }
}

struct Config {
    name: &'static str,
    analyzer: fn() -> Analyzer,
    similarity: fn() -> Arc<dyn Similarity>,
    offsets: bool,
    payloads: bool,
}

fn standard() -> Analyzer {
    Analyzer::new(StandardAnalyzer::new())
}

fn whitespace() -> Analyzer {
    Analyzer::new(WhitespaceAnalyzer::default())
}

fn bm25() -> Arc<dyn Similarity> {
    Arc::new(Bm25Similarity::default())
}

const CONFIGS: &[Config] = &[
    Config {
        name: "std",
        analyzer: standard,
        similarity: bm25,
        offsets: false,
        payloads: false,
    },
    Config {
        name: "std_offsets",
        analyzer: standard,
        similarity: bm25,
        offsets: true,
        payloads: false,
    },
    Config {
        name: "tuned",
        analyzer: standard,
        similarity: || Arc::new(Bm25Similarity::new(1.5, 0.5, true).unwrap()),
        offsets: false,
        payloads: false,
    },
    Config {
        name: "classic",
        analyzer: standard,
        similarity: || Arc::new(ClassicSimilarity::default()),
        offsets: true,
        payloads: false,
    },
    Config {
        name: "boolean",
        analyzer: standard,
        similarity: || Arc::new(BooleanSimilarity),
        offsets: false,
        payloads: false,
    },
    Config {
        name: "ws",
        analyzer: whitespace,
        similarity: bm25,
        offsets: true,
        payloads: true,
    },
    Config {
        name: "gaps",
        analyzer: || Analyzer::new(Gapped(StandardAnalyzer::new())),
        similarity: bm25,
        offsets: true,
        payloads: false,
    },
    Config {
        name: "payloads",
        analyzer: || Analyzer::new(Payloads),
        similarity: bm25,
        offsets: true,
        payloads: true,
    },
];

fn corpus(name: &str) -> Vec<String> {
    fs::read_to_string(dir().join("corpus").join(name))
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
}

fn bits(f: f32) -> String {
    format!("{:x}", f.to_bits())
}

fn error(e: lucene_search::Error) -> String {
    let class = match &e {
        lucene_search::Error::IllegalState(m) if m.starts_with("java.") => {
            return format!("X\t{m}");
        }
        lucene_search::Error::IllegalState(_) => "java.lang.IllegalStateException",
        _ => "java.lang.IllegalArgumentException",
    };
    let message = match &e {
        lucene_search::Error::IllegalArgument(m) | lucene_search::Error::IllegalState(m) => {
            m.clone()
        }
        other => other.to_string(),
    };
    format!("X\t{class}: {message}")
}

#[test]
fn memory_index_matches_lucene() {
    let docs = corpus("memory-index-docs.txt");
    let queries = corpus("memory-index-queries.txt");
    let mut bad = Vec::new();
    let mut total = 0;
    for c in CONFIGS {
        let analyzer = Arc::new((c.analyzer)());
        let parser = QueryParser::new("body", Arc::clone(&analyzer));
        let parsed: Vec<Clause> = queries
            .iter()
            .map(|q| parser.parse(q).unwrap().to_clause(1))
            .collect();
        let want =
            fs::read_to_string(dir().join(format!("data/memory_index/{}.tsv", c.name))).unwrap();
        let mut got = String::new();
        for (d, line) in docs.iter().enumerate() {
            let mut mi = MemoryIndex::with_options(c.offsets, c.payloads);
            mi.set_similarity((c.similarity)()).unwrap();
            for pair in line.split('\t') {
                let (field, value) = pair.split_once('=').unwrap();
                mi.add_field_text(field, value, &analyzer).unwrap();
            }
            got.push_str(&format!("D\t{d}\t{}\n", esc(&mi.to_string_debug())));
            mi.freeze().unwrap();
            for (q, clause) in queries.iter().zip(&parsed) {
                let score = match mi.search(clause) {
                    Ok(s) => bits(s),
                    Err(e) => error(e),
                };
                got.push_str(&format!("S\t{d}\t{}\t{score}\n", esc(q)));
            }
        }
        for (w, g) in want.lines().zip(got.lines()) {
            total += 1;
            if w != g {
                bad.push(format!(
                    "{}\n  want {}\n  got  {}",
                    c.name,
                    unesc(w),
                    unesc(g)
                ));
            }
        }
        assert_eq!(want.lines().count(), got.lines().count(), "{}", c.name);
    }
    assert!(total > 5000, "only {total} rows");
    let shown: Vec<&str> = bad.iter().take(30).map(String::as_str).collect();
    assert!(
        bad.is_empty(),
        "{} of {total} rows differ:\n{}",
        bad.len(),
        shown.join("\n")
    );
}

/// `AnalysisRows.esc`.
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn scenario(name: &str) -> Result<f32, lucene_search::Error> {
    let std = || Analyzer::new(StandardAnalyzer::new());
    let parse = |field: &str, a: Analyzer, q: &str| {
        QueryParser::new(field, Arc::new(a))
            .parse(q)
            .unwrap()
            .to_clause(1)
    };
    match name {
        "empty_match_all" => {
            MemoryIndex::new().search(&Clause::MatchAllDocs(MatchAllDocsQuery::new(1)))
        }
        "frozen_add" => {
            let mut mi = MemoryIndex::new();
            mi.add_field_text("body", "a b", &std())?;
            mi.freeze()?;
            mi.add_field_text("body", "c", &std())?;
            Ok(0.0)
        }
        "frozen_similarity" => {
            let mut mi = MemoryIndex::new();
            mi.freeze()?;
            mi.set_similarity(Arc::new(ClassicSimilarity::default()))?;
            Ok(0.0)
        }
        "numeric_twice" => {
            let mut mi = MemoryIndex::new();
            mi.add_field(&NumericDocValuesField::new("n", 1), None)?;
            mi.add_field(&NumericDocValuesField::new("n", 2), None)?;
            Ok(0.0)
        }
        "binary_twice" => {
            let mut mi = MemoryIndex::new();
            mi.add_field(&BinaryDocValuesField::new("b", b"x".to_vec()), None)?;
            mi.add_field(&BinaryDocValuesField::new("b", b"y".to_vec()), None)?;
            Ok(0.0)
        }
        "doc_values_type_change" => {
            let mut mi = MemoryIndex::new();
            mi.add_field(&NumericDocValuesField::new("d", 1), None)?;
            mi.add_field(&BinaryDocValuesField::new("d", b"y".to_vec()), None)?;
            Ok(0.0)
        }
        "point_in_range" | "point_out_of_range" => {
            let mut mi = MemoryIndex::new();
            mi.add_field(&LongPoint::new("p", &[5]).unwrap(), None)?;
            let (min, max) = if name == "point_in_range" {
                mi.add_field(&LongPoint::new("p", &[50]).unwrap(), None)?;
                (1, 10)
            } else {
                (6, 10)
            };
            mi.search(&Clause::PointsRange(PointsRangeQuery {
                field: "p".into(),
                min,
                max,
            }))
        }
        "string_field" => {
            let mut mi = MemoryIndex::new();
            mi.add_field(&StringField::new("id", "Ab-1", Store::No), None)?;
            mi.add_field_text("body", "some text", &std())?;
            mi.search(&parse("id", whitespace(), "Ab-1 OR body:text"))
        }
        "same_field_twice" => {
            let mut mi = MemoryIndex::new();
            mi.add_field_text("body", "the quick", &std())?;
            mi.add_field_text("body", "brown fox", &std())?;
            mi.search(&parse("body", std(), "\"quick brown\""))
        }
        "keyword_stream" => {
            let mut mi = MemoryIndex::with_options(true, false);
            let tokens = MemoryIndex::keyword_token_stream(&["Red Fox", "blue", "Red Fox"]);
            mi.add_field_tokens("tags", &tokens, 0, 1)?;
            mi.search(&parse("tags", whitespace(), "\"Red Fox\" OR blue"))
        }
        "exists_text" | "exists_missing" => {
            let mut mi = MemoryIndex::new();
            mi.add_field_text("body", "a b", &std())?;
            let field = if name == "exists_text" {
                "body"
            } else {
                "title"
            };
            mi.search(&Clause::Exists(FieldExistsQuery::new(field)))
        }
        "exists_doc_values" => {
            let mut mi = MemoryIndex::new();
            mi.add_field(&NumericDocValuesField::new("n", 7), None)?;
            mi.search(&Clause::Exists(FieldExistsQuery::new("n")))
        }
        "text_beside_doc_values" => {
            let mut mi = MemoryIndex::with_options(true, false);
            mi.add_field(&NumericDocValuesField::new("n", 7), None)?;
            mi.add_field_text("body", "the quick brown fox", &std())?;
            mi.search(&parse("body", std(), "\"quick brown\" fox"))
        }
        "payloads_beside_doc_values" => {
            let mut mi = MemoryIndex::with_options(true, true);
            mi.add_field(&NumericDocValuesField::new("n", 7), None)?;
            mi.add_field_text("body", "quick|5 brown|x fox", &Analyzer::new(Payloads))?;
            mi.search(&parse("body", whitespace(), "\"quick brown\" fox"))
        }
        "similarity_after_reset" => {
            let mut mi = MemoryIndex::new();
            mi.set_similarity(Arc::new(ClassicSimilarity::default()))?;
            mi.add_field_text("body", "a a b", &std())?;
            mi.reset();
            mi.add_field_text("body", "a a b", &std())?;
            mi.search(&parse("body", std(), "a"))
        }
        other => panic!("unknown scenario {other}"),
    }
}

#[test]
fn memory_index_scenarios_match_lucene() {
    let want = fs::read_to_string(dir().join("data/memory_index/scenarios.tsv")).unwrap();
    let mut bad = Vec::new();
    for line in want.lines() {
        let (name, expected) = line.split_once('\t').unwrap();
        let got = match scenario(name) {
            Ok(s) => bits(s),
            Err(e) => error(e),
        };
        if unesc(expected) != unesc(&got) {
            bad.push(format!("{name}\n  want {}\n  got  {got}", unesc(expected)));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
