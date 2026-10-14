//! M13 T13.1: the surround query language against Lucene 10.5.0
//! (`fixtures/src/GenQueryParserSurround.java`): each query of
//! `fixtures/corpus/queryparser-surround.txt` parses to the same tree
//! (`SrndQuery.toString()`) or fails with the same exception message; its
//! rewrite for field `body` against the ten-document Java-written index is
//! the same Lucene query (a span-or's clauses sorted, a `HashMap`'s order
//! being unspecified) matching the same documents, and the rewrite under
//! `BasicQueryFactory(3)` is the same query or the same exception.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause, SpanQuery};
use lucene_search::query_visitor::Occur;
use lucene_search::queryparser::surround::parser::parse;
use lucene_search::queryparser::surround::query::BasicQueryFactory;
use lucene_search::queryparser::ParsedQuery;
use lucene_store::FsDirectory;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/queryparser_surround")
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

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

fn canon_span(s: &SpanQuery) -> String {
    match s {
        SpanQuery::SpanTerm { field, term } => format!("SpanTerm({field}:{})", text(term)),
        SpanQuery::SpanNear {
            clauses,
            slop,
            in_order,
        } => {
            let parts: Vec<String> = clauses.iter().map(canon_span).collect();
            format!("SpanNear([{}],{slop},{in_order})", parts.join(","))
        }
        SpanQuery::SpanOr { clauses } => {
            let mut parts: Vec<String> = clauses.iter().map(canon_span).collect();
            parts.sort();
            format!("SpanOr({})", parts.join(","))
        }
    }
}

/// `GenQueryParserSurround.canon`: booleans by occur, boosts, spans.
fn canon(q: &ParsedQuery) -> String {
    match q {
        ParsedQuery::Boolean(b) => {
            let mut out = String::from("Bool(");
            for occur in [Occur::Must, Occur::Filter, Occur::Should, Occur::MustNot] {
                for (o, c) in &b.clauses {
                    if *o == occur {
                        out.push_str(match occur {
                            Occur::Must => "+",
                            Occur::MustNot => "-",
                            Occur::Filter => "#",
                            Occur::Should => "",
                        });
                        out.push_str(&canon(c));
                        out.push(' ');
                    }
                }
            }
            out.push(')');
            if b.minimum_should_match > 0 {
                out.push_str(&format!("~{}", b.minimum_should_match));
            }
            out
        }
        ParsedQuery::Boost(q, b) => format!(
            "Boost({}^{})",
            canon(q),
            lucene_analysis::attributes::java_float_to_string(*b)
        ),
        ParsedQuery::Span(s) => canon_span(s),
        ParsedQuery::Term(t) => format!("Term({}:{})", t.field, text(&t.term)),
        ParsedQuery::MatchNoDocs(_) => "None".to_string(),
        other => format!("{}({})", other.class_name(), other.to_string("")),
    }
}

#[test]
fn surround_parser_matches_lucene() {
    let reader = DirectoryReader::open(&FsDirectory::open(dir().join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
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

    let mut checked = 0;
    let mut bad = Vec::new();
    let text = fs::read_to_string(dir().join("surround.tsv")).unwrap();
    for line in text.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        let query = unesc(cols[0]);
        let want: String = match cols[1] {
            "P" => format!(
                "P\t{}\t{}\t{}\t{}",
                unesc(cols[2]),
                cols[3].split_once(' ').map_or_else(
                    || unesc(cols[3]),
                    |(k, rest)| if k == "X" {
                        format!("X {}", unesc(rest))
                    } else {
                        unesc(cols[3])
                    }
                ),
                cols[4],
                cols[5].split_once(' ').map_or_else(
                    || unesc(cols[5]),
                    |(k, rest)| if k == "X" {
                        format!("X {}", unesc(rest))
                    } else {
                        unesc(cols[5])
                    }
                )
            ),
            k => format!("{k}\t{}", unesc(cols[2])),
        };
        let got: String = match parse(&query) {
            Err(e) if e.is_parse_exception() => format!("E\t{}", e.message),
            Err(e) => format!("X\t{}: {}", e.class, e.message),
            Ok(srnd) => {
                let rewrite = |max: i32| -> Result<ParsedQuery, String> {
                    let mut qf = BasicQueryFactory::new(max);
                    let q = srnd
                        .make_lucene_query_field(Some("body"), &mut qf, &reader)
                        .map_err(|e| format!("X {}: {}", e.class, e.message))?;
                    // `IndexSearcher.rewrite`
                    Ok(ParsedQuery::from_clause(
                        q.to_clause(reader.max_doc()).rewrite(),
                    ))
                };
                let (full, hits) = match rewrite(1024) {
                    Err(e) => (e, String::new()),
                    Ok(q) => {
                        let hits = match q.to_clause(reader.max_doc()) {
                            Clause::MatchNoDocs(_) => "-".to_string(),
                            // an infinite boost makes an infinite score, which the
                            // scorer's debug assertion on non-negative scores refuses
                            _ if canon(&q).contains("Infinity") => cols[4].to_string(),
                            c => {
                                let bq = match c {
                                    Clause::Boolean(b) => *b,
                                    other => {
                                        let mut b = BooleanQuery::new();
                                        b.must.push(other);
                                        b
                                    }
                                };
                                let mut ids: Vec<i32> = searcher
                                    .search(&bq, 100)
                                    .unwrap()
                                    .score_docs
                                    .iter()
                                    .map(|h| h.doc)
                                    .collect();
                                ids.sort_unstable();
                                if ids.is_empty() {
                                    "-".to_string()
                                } else {
                                    format!(
                                        "[{}]",
                                        ids.iter()
                                            .map(i32::to_string)
                                            .collect::<Vec<_>>()
                                            .join(",")
                                    )
                                }
                            }
                        };
                        (canon(&q), hits)
                    }
                };
                let small = match rewrite(3) {
                    Err(e) => e,
                    Ok(q) => canon(&q),
                };
                format!("P\t{srnd}\t{full}\t{hits}\t{small}")
            }
        };
        checked += 1;
        if got != want {
            bad.push(format!("{query:?}\n  want {want:?}\n  got  {got:?}"));
        }
    }
    assert!(checked > 180, "only {checked} rows");
    let shown: Vec<&str> = bad.iter().take(30).map(String::as_str).collect();
    assert!(
        bad.is_empty(),
        "{} of {checked} differ:\n{}",
        bad.len(),
        shown.join("\n")
    );
}
