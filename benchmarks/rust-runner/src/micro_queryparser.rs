//! M13 T13.1's pairs (`benchmarks/micro/java/QueryParserMicro.java` is the
//! Java twin): the classic `QueryParser` over
//! `fixtures/corpus/queryparser-classic.txt` and over
//! `fixtures/data/queryparser_classic/fragments.txt` (most of which are
//! syntax errors), and `MultiFieldQueryParser` over the corpus;
//! `ComplexPhraseQueryParser` over its corpus, each query parsed and rewritten
//! against `fixtures/data/queryparser_complexphrase/index`.

use std::hint::black_box;
use std::time::Duration;

use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_search::queryparser::classic::QueryParser;
use lucene_search::queryparser::simple::SimpleQueryParser;

use super::measure;

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
                let unit = u32::from_str_radix(&h, 16).unwrap();
                out.push(char::from_u32(unit).unwrap_or('\u{fffd}'));
            }
            Some(o) => out.push(o),
            None => out.push('\\'),
        }
    }
    out
}

fn run(name: &str, p: &QueryParser, queries: &[String], w: Duration, m: Duration) {
    measure(name, w, m, || {
        for q in queries {
            match p.parse(black_box(q)) {
                Ok(parsed) => {
                    black_box(&parsed);
                }
                Err(e) => {
                    black_box(e.message.len());
                }
            }
        }
        queries.len() as u64
    });
}

pub fn bench_queryparser(w: Duration, m: Duration) {
    let corpus: Vec<String> = std::fs::read_to_string("fixtures/corpus/queryparser-classic.txt")
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    let fragments: Vec<String> =
        std::fs::read_to_string("fixtures/data/queryparser_classic/fragments.txt")
            .unwrap()
            .lines()
            .map(unesc)
            .collect();
    let standard = || Analyzer::new(StandardAnalyzer::new());
    run(
        "qp_classic_corpus",
        &QueryParser::new("f", standard()),
        &corpus,
        w,
        m,
    );
    run(
        "qp_classic_fragments",
        &QueryParser::new("f", standard()),
        &fragments,
        w,
        m,
    );
    let mut simple = corpus.clone();
    simple.extend(
        std::fs::read_to_string("fixtures/corpus/queryparser-simple.txt")
            .unwrap()
            .lines()
            .map(str::to_string),
    );
    let sp = SimpleQueryParser::new(standard(), "f");
    measure("qp_simple_corpus", w, m, || {
        for q in &simple {
            match sp.parse(black_box(q)) {
                Ok(parsed) => {
                    black_box(&parsed);
                }
                Err(e) => {
                    black_box(e.message.len());
                }
            }
        }
        simple.len() as u64
    });
    let surround: Vec<String> = std::fs::read_to_string("fixtures/corpus/queryparser-surround.txt")
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    measure("qp_surround_corpus", w, m, || {
        for q in &surround {
            match lucene_search::queryparser::surround::parser::parse(black_box(q)) {
                Ok(parsed) => {
                    black_box(parsed.to_string().len());
                }
                Err(e) => {
                    black_box(e.message.len());
                }
            }
        }
        surround.len() as u64
    });
    run(
        "qp_multi_field_corpus",
        &QueryParser::multi_field(&["title", "body"], standard()),
        &corpus,
        w,
        m,
    );
    let complex: Vec<String> =
        std::fs::read_to_string("fixtures/corpus/queryparser-complexphrase.txt")
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
    let reader = lucene_search::directory_reader::DirectoryReader::open(
        &lucene_store::FsDirectory::open("fixtures/data/queryparser_complexphrase/index"),
    )
    .unwrap();
    let max_doc = reader.max_doc();
    let p = QueryParser::complex_phrase("body", standard());
    measure("qp_complexphrase_corpus", w, m, || {
        for q in &complex {
            let rewritten = p
                .parse(black_box(q))
                .and_then(|parsed| parsed.rewrite_complex_phrases(&reader))
                .map(|r| r.to_clause(max_doc).rewrite());
            match rewritten {
                Ok(c) => {
                    black_box(&c);
                }
                Err(e) => {
                    black_box(e.message.len());
                }
            }
        }
        complex.len() as u64
    });
}
