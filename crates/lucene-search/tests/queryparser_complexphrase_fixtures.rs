//! M13 T13.1: `ComplexPhraseQueryParser` against Lucene 10.5.0
//! (`fixtures/src/GenQueryParserComplexPhrase.java`), in order and not: each
//! query of `fixtures/corpus/queryparser-complexphrase.txt` parses to the
//! same tree or fails with the same exception, and against the twelve-document,
//! two-segment Java-written index rewrites to the same query (a boolean's clauses sorted
//! within each occur, a span-or's in their order) matching the same documents.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause};
use lucene_search::query_visitor::Occur;
use lucene_search::queryparser::classic::QueryParser;
use lucene_search::queryparser::ParsedQuery;
use lucene_search::spans::SpanNode;
use lucene_store::FsDirectory;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/queryparser_complexphrase")
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

fn canon_span(s: &SpanNode) -> String {
    match s {
        SpanNode::Term { field, term } => format!("SpanTerm({field}:{})", text(term)),
        SpanNode::Near {
            clauses,
            slop,
            in_order,
        } => {
            let parts: Vec<String> = clauses.iter().map(canon_span).collect();
            format!("SpanNear([{}],{slop},{in_order})", parts.join(","))
        }
        SpanNode::Or { clauses } => {
            let parts: Vec<String> = clauses.iter().map(canon_span).collect();
            format!("SpanOr({})", parts.join(","))
        }
        SpanNode::Not {
            include, exclude, ..
        } => format!("SpanNot({},{})", canon_span(include), canon_span(exclude)),
        other => format!("{other}"),
    }
}

/// `GenQueryParserComplexPhrase.canon`.
fn canon(q: &ParsedQuery) -> String {
    match q {
        ParsedQuery::Boolean(b) => {
            let mut out = String::from("Bool(");
            for occur in [Occur::Must, Occur::Filter, Occur::Should, Occur::MustNot] {
                let mut parts: Vec<String> = b
                    .clauses
                    .iter()
                    .filter(|(o, _)| *o == occur)
                    .map(|(_, c)| {
                        let prefix = match occur {
                            Occur::Must => "+",
                            Occur::MustNot => "-",
                            Occur::Filter => "#",
                            Occur::Should => "",
                        };
                        format!("{prefix}{}", canon(c))
                    })
                    .collect();
                parts.sort();
                for p in parts {
                    out.push_str(&p);
                    out.push(' ');
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
        ParsedQuery::SpanNode(s) => canon_span(s),
        ParsedQuery::Term(t) => format!("Term({}:{})", t.field, text(&t.term)),
        ParsedQuery::MatchNoDocs(_) => "None".to_string(),
        other => format!("{}({})", other.class_name(), other.to_string("")),
    }
}

#[test]
fn complex_phrase_parser_matches_lucene() {
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
    let analyzer = std::sync::Arc::new(lucene_analysis::Analyzer::new(
        lucene_analysis::StandardAnalyzer::new(),
    ));

    let mut checked = 0;
    let mut bad = Vec::new();
    for (config, in_order) in [("in_order", true), ("unordered", false)] {
        let mut p = QueryParser::complex_phrase("body", std::sync::Arc::clone(&analyzer));
        p.set_in_order(in_order);
        let tsv = fs::read_to_string(dir().join(format!("{config}.tsv"))).unwrap();
        for line in tsv.lines() {
            let cols: Vec<&str> = line.split('\t').collect();
            let query = unesc(cols[0]);
            let want = match cols[1] {
                "Q" => format!(
                    "Q\t{}\t{}\t{}\t{}",
                    unesc(cols[2]),
                    unesc(cols[3]),
                    unesc(cols[4]),
                    cols.get(5).copied().unwrap_or("")
                ),
                k => format!("{k}\t{}", unesc(cols[2])),
            };
            let got = match p.parse(&query) {
                Err(e) if e.is_parse_exception() => format!("E\t{}", e.message),
                Err(e) => format!("X\t{}: {}", e.class, e.message),
                Ok(q) => {
                    let head = format!("Q\t{}\t{}", q.to_string("body"), q.describe("f"));
                    match q.rewrite_complex_phrases(&reader) {
                        Err(e) => format!("{head}\tX {}: {}\t", e.class, e.message),
                        Ok(r) => {
                            // `IndexSearcher.rewrite` of what is left
                            let r =
                                ParsedQuery::from_clause(r.to_clause(reader.max_doc()).rewrite());
                            let hits = match r.to_clause(reader.max_doc()) {
                                Clause::MatchNoDocs(_) => "-".to_string(),
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
                            format!("{head}\t{}\t{hits}", canon(&r))
                        }
                    }
                }
            };
            checked += 1;
            if got != want {
                bad.push(format!(
                    "{config} {query:?}\n  want {want:?}\n  got  {got:?}"
                ));
            }
        }
    }
    assert!(checked > 120, "only {checked} rows");
    let shown: Vec<&str> = bad.iter().take(30).map(String::as_str).collect();
    assert!(
        bad.is_empty(),
        "{} of {checked} differ:\n{}",
        bad.len(),
        shown.join("\n")
    );
}

/// Contents no query string makes (the analyzer here makes no synonyms, the
/// parser no prefix length), rewritten as `ComplexPhraseQuery.rewrite`
/// would against the same index.
#[test]
fn contents_the_corpus_cannot_make_rewrite_as_lucene_would() {
    use lucene_search::extended_query::SynonymQuery;
    use lucene_search::query::{FuzzyQuery, TermQuery};
    use lucene_search::queryparser::complex_phrase::ComplexPhraseQuery;
    use lucene_search::queryparser::ParsedBoolean;

    let reader = DirectoryReader::open(&FsDirectory::open(dir().join("index"))).unwrap();
    let phrase = |contents: Option<ParsedQuery>| ComplexPhraseQuery {
        field: "body".into(),
        text: "x".into(),
        slop: 0,
        in_order: true,
        contents,
    };
    let should = |qs: Vec<ParsedQuery>| {
        Some(ParsedQuery::Boolean(ParsedBoolean {
            clauses: qs.into_iter().map(|q| (Occur::Should, q)).collect(),
            minimum_should_match: 0,
        }))
    };
    let fox = || ParsedQuery::Term(TermQuery::new("body", b"fox".to_vec()));
    let rewrite = |c: Option<ParsedQuery>| match phrase(c).rewrite(&reader) {
        Ok(q) => canon(&q),
        Err(e) => format!("{}: {}", e.class, e.message),
    };

    assert_eq!(
        rewrite(None),
        "RuntimeException: ComplexPhraseQuery: the phrase was never parsed"
    );
    assert_eq!(
        rewrite(Some(ParsedQuery::MatchAll)),
        "IllegalArgumentException: Unknown query type \"org.apache.lucene.search.MatchAllDocsQuery\" found in phrase query string \"x\""
    );
    assert_eq!(
        rewrite(should(vec![ParsedQuery::MatchAll, fox()])),
        "IllegalArgumentException: Unknown query type \"org.apache.lucene.search.MatchAllDocsQuery\" found in phrase query string \"x\""
    );
    // a synonym clause becomes the span-or of its terms
    let synonym = SynonymQuery::new(
        "body",
        vec![(b"quick".to_vec(), 1.0), (b"lazy".to_vec(), 1.0)],
    )
    .unwrap();
    assert_eq!(
        rewrite(should(vec![ParsedQuery::Synonym(synonym), fox()])),
        "SpanNear([SpanOr(SpanTerm(body:lazy),SpanTerm(body:quick)),SpanTerm(body:fox)],0,true)"
    );
    // a fuzzy term's prefix must match exactly: "qu" + "ikc" is one
    // transposition from "quick"; nothing starts with "qx"
    let fuzzy = |t: &str| {
        ParsedQuery::Fuzzy(
            FuzzyQuery::new("body", t.as_bytes().to_vec())
                .with_max_edits(1)
                .with_prefix_length(2),
        )
    };
    assert_eq!(
        rewrite(should(vec![fuzzy("quikc"), fox()])),
        "SpanNear([SpanTerm(body:quick),SpanTerm(body:fox)],0,true)"
    );
    assert_eq!(
        rewrite(should(vec![fuzzy("qxick"), fox()])),
        "SpanNear([SpanTerm(body:Dummy clause because no terms found - must match nothing),SpanTerm(body:fox)],0,true)"
    );
    // a lone fuzzy term is the fuzzy query's own rewrite
    assert_eq!(rewrite(Some(fuzzy("quikc"))), "Boost(Term(body:quick)^0.8)");
}
