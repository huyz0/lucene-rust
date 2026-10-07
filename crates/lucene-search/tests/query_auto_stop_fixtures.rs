//! M11 T11.6: `QueryAutoStopWordAnalyzer` against
//! `fixtures/src/GenQueryAutoStop.java`: the same constructors over the same
//! Java-written index (three segments, deletions) give the same stop words
//! per field and the same tokens.

use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::{Analyzer, AnalyzerDefinition, TokenStream, TokenStreamComponents};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::query_auto_stop_word_analyzer::QueryAutoStopWordAnalyzer;
use lucene_store::FsDirectory;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/query_auto_stop"
    )
    .to_string()
}

/// `WhitespaceAnalyzer`.
struct Whitespace;

impl AnalyzerDefinition for Whitespace {
    fn create_components(
        &self,
        _field: &str,
    ) -> Result<TokenStreamComponents, lucene_analysis::AnalysisError> {
        Ok(TokenStreamComponents::new(WhitespaceTokenizer::new()))
    }
}

/// The generator's `LINES`.
const LINES: [&str; 4] = [
    "the lucene index and the rust search",
    "a query of stop words",
    "日本 écrit",
    "",
];
const FIELDS_ASKED: [&str; 5] = ["body", "title", "id", "note", "missing"];

fn record(name: &str, a: QueryAutoStopWordAnalyzer) -> String {
    let mut o = format!("#case\t{name}\n");
    for f in FIELDS_ASKED {
        o += &format!("stop\t{f}\t{}\n", a.stop_words(f).join(" "));
    }
    let all: Vec<String> = a
        .all_stop_words()
        .into_iter()
        .map(|(f, w)| format!("{f}:{w}"))
        .collect();
    o += &format!("all\t{}\n", all.join(" "));
    let a = a.into_analyzer();
    for f in ["body", "title", "missing"] {
        for (ln, line) in LINES.iter().enumerate() {
            let mut ts = a.token_stream(f, line).unwrap();
            ts.reset().unwrap();
            let mut toks = Vec::new();
            while ts.increment_token().unwrap() {
                let at = ts.attributes();
                toks.push(format!("{}:{}", at.term(), at.position_increment()));
            }
            ts.end().unwrap();
            ts.close().unwrap();
            o += &format!("tokens\t{f}\t{ln}\t{}\n", toks.join(" "));
        }
    }
    o
}

#[test]
fn stop_words_and_tokens_match_lucene() {
    let reader = DirectoryReader::open(&FsDirectory::open(format!("{}/index", dir()))).unwrap();
    let ws = || Analyzer::new(Whitespace);
    let mut out = String::new();
    out += &record(
        "default",
        QueryAutoStopWordAnalyzer::new(ws(), &reader).unwrap(),
    );
    out += &record(
        "maxDocFreq=10",
        QueryAutoStopWordAnalyzer::with_max_doc_freq(ws(), &reader, 10).unwrap(),
    );
    out += &record(
        "percent=0.1",
        QueryAutoStopWordAnalyzer::with_max_percent_docs(ws(), &reader, 0.1).unwrap(),
    );
    out += &record(
        "fields=body,missing percent=0.25",
        QueryAutoStopWordAnalyzer::for_fields_percent(ws(), &reader, &["body", "missing"], 0.25)
            .unwrap(),
    );
    out += &record(
        "fields=title,id maxDocFreq=3",
        QueryAutoStopWordAnalyzer::for_fields(ws(), &reader, &["title", "id"], 3).unwrap(),
    );
    out += &record(
        "maxDocFreq=1000",
        QueryAutoStopWordAnalyzer::with_max_doc_freq(ws(), &reader, 1000).unwrap(),
    );
    let expected = std::fs::read_to_string(format!("{}/cases.txt", dir())).unwrap();
    for (i, (a, e)) in out.lines().zip(expected.lines()).enumerate() {
        assert_eq!(a, e, "cases.txt line {}", i + 1);
    }
    assert_eq!(out.lines().count(), expected.lines().count());
}
