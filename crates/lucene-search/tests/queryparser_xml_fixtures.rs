//! M13 T13.1: `CoreParser` against Lucene 10.5.0 (`fixtures/src/GenQueryParserXml.java`). Each
//! XML document of `fixtures/corpus/queryparser-xml.txt` parses to the query Java's `CoreParser`
//! makes, with the same `toString`, class tree and matching documents over a ten-document
//! Java-written index, or fails with the same exception class and message (a syntax error by its
//! class alone).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause};
use lucene_search::queryparser::xml::{CoreParser, PARSER_EXCEPTION};
use lucene_search::queryparser::ParsedQuery;
use lucene_store::FsDirectory;

fn fixtures() -> PathBuf {
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

#[test]
fn core_parser_matches_lucene() {
    let dir = fixtures().join("data/queryparser_xml");
    let reader = DirectoryReader::open(&FsDirectory::open(dir.join("index"))).unwrap();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let mut owned: Vec<HashMap<String, FieldNorms<'_>>> = Vec::new();
    for field in ["body", "title"] {
        for (i, n) in reader.field_norms(field).into_iter().enumerate() {
            if owned.len() <= i {
                owned.push(HashMap::new());
            }
            if let Some(n) = n {
                owned[i].insert(field.to_string(), n);
            }
        }
    }
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let analyzer: std::sync::Arc<Analyzer> =
        std::sync::Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let parser = CoreParser::new(Some("body"), analyzer);

    let corpus = fs::read_to_string(fixtures().join("corpus/queryparser-xml.txt")).unwrap();
    let corpus: Vec<&str> = corpus.lines().collect();
    let tsv = fs::read_to_string(dir.join("xml.tsv")).unwrap();
    let rows: Vec<&str> = tsv.lines().collect();
    assert_eq!(rows.len(), corpus.len(), "one row per corpus line");

    let mut bad = Vec::new();
    for (xml, row) in corpus.iter().zip(&rows) {
        let cols: Vec<String> = row.split('\t').map(unesc).collect();
        assert_eq!(cols[0], *xml, "row for the wrong document");
        let want: Vec<String> = cols[1..].to_vec();
        let got = match parser.parse(xml) {
            Err(e)
                if e.class == PARSER_EXCEPTION
                    && e.message.starts_with("Error parsing XML stream") =>
            {
                vec!["X".into(), format!("{PARSER_EXCEPTION}: (syntax)")]
            }
            Err(e) => vec!["X".into(), format!("{}: {}", e.class, e.message)],
            Ok(q) => {
                let mut out = vec!["Q".into(), q.to_string("body"), q.describe("f")];
                out.extend(hits_or_error(&q, &reader, &searcher));
                out
            }
        };
        if got != want {
            bad.push(format!("{xml}\n  want {want:?}\n  got  {got:?}"));
        }
    }
    let shown: Vec<&str> = bad.iter().take(30).map(String::as_str).collect();
    assert!(
        bad.is_empty(),
        "{} of {} differ:\n{}",
        bad.len(),
        corpus.len(),
        shown.join("\n")
    );
}

/// The matching documents of `q` as `[1,7]` or `-`, or the `X` columns of the
/// exception its rewrite throws.
fn hits_or_error(
    q: &ParsedQuery,
    reader: &DirectoryReader,
    searcher: &IndexSearcher<'_, '_>,
) -> Vec<String> {
    let max_doc = reader.max_doc();
    let rewritten = ParsedQuery::from_clause(q.to_clause(max_doc).rewrite());
    let clause = rewritten.to_clause(max_doc);
    let bq = match clause {
        Clause::MatchNoDocs(_) => return vec!["-".into()],
        Clause::Boolean(b) => *b,
        other => {
            let mut b = BooleanQuery::new();
            b.must.push(other);
            b
        }
    };
    let hits = match searcher.search(&bq, 100) {
        Ok(h) => h,
        // a point range over a field of another shape: `checkValidPointValues`
        Err(lucene_search::Error::IllegalArgument(m)) => {
            return vec![
                "X".into(),
                format!("java.lang.IllegalArgumentException: {m}"),
            ];
        }
        Err(e) => panic!("unexpected search error: {e}"),
    };
    let mut ids: Vec<i32> = hits.score_docs.iter().map(|h| h.doc).collect();
    ids.sort_unstable();
    if ids.is_empty() {
        vec!["-".into()]
    } else {
        vec![format!(
            "[{}]",
            ids.iter().map(i32::to_string).collect::<Vec<_>>().join(",")
        )]
    }
}
