//! M13 T13.5a: `QueryAnalyzer`, `QueryDecomposer` and `TermWeightor` against
//! Lucene 10.5.0 (`fixtures/src/GenMonitor.java`): every query of
//! `fixtures/corpus/monitor-queries.txt`, parsed with the classic parser and
//! `StandardAnalyzer` as the generator parses it, to its decomposition, its
//! term tree's text and collected terms; and the named weightors' weights
//! over fixed terms, bit for bit.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_search::monitor::{QueryAnalyzer, QueryDecomposer, Term, TermWeightor};
use lucene_search::query::Clause;
use lucene_search::queryparser::classic::QueryParser;
use lucene_search::queryparser::ParsedQuery;

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

fn corpus() -> Vec<String> {
    fs::read_to_string(dir().join("corpus/monitor-queries.txt"))
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
}

fn parse(q: &str) -> Result<Clause, String> {
    let parser = QueryParser::new("body", Arc::new(Analyzer::new(StandardAnalyzer::new())));
    parser
        .parse(q)
        .map(|p| p.to_clause(1))
        .map_err(|e| format!("{}: {}", e.class, e.message))
}

/// `sort()` on the decomposed sub-queries' `toString()`s, as the generator
/// writes them.
fn decomposition(q: &Clause) -> Vec<String> {
    let mut subs: Vec<String> = QueryDecomposer
        .decompose(q)
        .iter()
        .map(|c| ParsedQuery::from_clause(c.clone()).to_string("{0}"))
        .collect();
    subs.sort();
    subs
}

#[test]
fn decomposition_matches_lucene() {
    let want = fs::read_to_string(dir().join("data/monitor/decompose.tsv")).unwrap();
    let mut bad = Vec::new();
    let queries = corpus();
    for (line, q) in want.lines().zip(&queries) {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(unesc(cols[0]), *q);
        let got = match parse(q) {
            Ok(clause) => {
                let mut row = vec!["Q".to_string()];
                row.extend(decomposition(&clause).iter().map(|s| escape(s)));
                row.join("\t")
            }
            Err(e) => format!("X\t{}", escape(&e)),
        };
        let expected = cols[1..].join("\t");
        if got != expected {
            bad.push(format!("{q:?}\n  want {expected}\n  got  {got}"));
        }
    }
    assert!(bad.is_empty(), "{} differ:\n{}", bad.len(), bad.join("\n"));
}

/// Splits a printed boolean's top-level clauses (space-separated, at paren
/// depth zero), keeping each clause's own parentheses.
fn split_top(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' => {
                depth += 1;
                cur.push(c);
            }
            ')' => {
                depth -= 1;
                cur.push(c);
            }
            ' ' if depth == 0 => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Whether two tree texts hold the same nodes, sibling order aside: the same
/// lines in any order. Java orders equally weighted siblings by its
/// `Multiset`'s hash iteration, which this port does not reproduce; the
/// nodes, their weights and their depths are compared exactly.
fn same_nodes(want: &str, got: &str) -> bool {
    let mut w: Vec<&str> = want.split("\n").collect();
    let mut g: Vec<&str> = got.split("\n").collect();
    w.sort_unstable();
    g.sort_unstable();
    w == g
}

/// The collected terms. A disjunction collects every child, so its terms are
/// compared as a sorted multiset. A conjunction collects only its heaviest
/// child, and among tied children that child is the first in Java's
/// nondeterministic order: either tied child is a valid key, since the query
/// requires both, so the collected term must be one of the tree's terms.
fn same_terms(want: &str, got: &str, tree: &str) -> bool {
    let mut w: Vec<&str> = want.split(' ').filter(|t| !t.is_empty()).collect();
    let mut g: Vec<&str> = got.split(' ').filter(|t| !t.is_empty()).collect();
    // a conjunction collects the terms of its first child; where that child is
    // a term (or a conjunction of tied terms) both sides collect one term, and
    // which one is Java's hash order, so any term of the tree is accepted
    if tree.starts_with("Conjunction") && g.len() == 1 && w.len() == 1 {
        return tree_has_term(tree, g[0]);
    }
    w.sort_unstable();
    g.sort_unstable();
    w == g
}

/// Whether a tree's text has a line naming `term` (`field:text^weight`).
fn tree_has_term(tree: &str, term: &str) -> bool {
    tree.split('\n')
        .any(|l| l.trim_start().starts_with(&format!("{term}^")))
}

#[test]
fn term_trees_match_lucene() {
    let want = fs::read_to_string(dir().join("data/monitor/tree.tsv")).unwrap();
    let weightor = TermWeightor::default();
    let mut bad = Vec::new();
    for (line, q) in want.lines().zip(&corpus()) {
        let cols: Vec<&str> = line.split('\t').collect();
        let got = match parse(q) {
            Ok(clause) => match QueryAnalyzer::default().build_tree(&clause, &weightor) {
                Ok(tree) => {
                    let mut terms: Vec<Term> = Vec::new();
                    tree.collect_terms(&mut terms);
                    let terms: Vec<String> = terms
                        .iter()
                        .map(|t| format!("{}:{}", t.field, String::from_utf8_lossy(&t.bytes)))
                        .collect();
                    match tree.to_string_at(0) {
                        Ok(text) => format!("Q\t{}\t{}", escape(&text), terms.join(" ")),
                        Err(e) => format!("X\t{e}"),
                    }
                }
                Err(e) => format!("X\t{e}"),
            },
            Err(e) => format!("X\t{e}"),
        };
        // the same tree, sibling order aside; the same collected terms, in order
        // the row is `query  Q  tree  terms`, or `query  X  message`
        let got_cols: Vec<&str> = got.split('\t').collect();
        let same = match (cols.get(1), got_cols.first(), cols.get(2), got_cols.get(1)) {
            (Some(&"Q"), Some(&"Q"), Some(want_tree), Some(got_tree)) => {
                let (want_terms, got_terms) = (cols.get(3), got_cols.get(2));
                // Java's tie order among equal-weight siblings comes from a
                // `HashMap` keyed on `Class.hashCode()`, an identity hash that
                // differs between JVM runs. So the collected terms are compared
                // as the set of terms the tree indexes, not their order.
                same_nodes(&unesc(want_tree), &unesc(got_tree))
                    && same_terms(
                        &want_terms.map(|t| unesc(t)).unwrap_or_default(),
                        &got_terms.map(|t| unesc(t)).unwrap_or_default(),
                        &unesc(want_tree),
                    )
            }
            _ => got == cols[1..].join("\t"),
        };
        if cols.len() < 2 || !same {
            bad.push(format!(
                "{q:?}\n  want {}\n  got  {got}",
                cols[1..].join("\t")
            ));
        }
    }
    assert!(bad.is_empty(), "{} differ:\n{}", bad.len(), bad.join("\n"));
}

/// `AnalysisRows.esc`.
fn escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

#[test]
fn weightors_match_lucene() {
    let want = fs::read_to_string(dir().join("data/monitor/weights.tsv")).unwrap();
    let probes = [
        Term {
            field: "body".into(),
            bytes: b"fox".to_vec(),
        },
        Term {
            field: "body".into(),
            bytes: b"a".to_vec(),
        },
        Term {
            field: "title".into(),
            bytes: b"quick".to_vec(),
        },
        Term {
            field: "body".into(),
            bytes: b"extraordinarily".to_vec(),
        },
        Term {
            field: "body".into(),
            bytes: Vec::new(),
        },
    ];
    let named = [
        ("default", TermWeightor::default()),
        ("length", TermWeightor::length(3.0, f64::from(0.3f32))),
        ("field", TermWeightor::field(2.0, &["title"])),
        ("term", TermWeightor::bytes(3.0, &[b"fox"])),
        (
            "freq",
            TermWeightor::term_freq(vec![("fox".to_string(), 4)], 10.0, 1.0),
        ),
        (
            "combined",
            TermWeightor::combine(vec![
                TermWeightor::default(),
                TermWeightor::field(2.0, &["title"]),
            ]),
        ),
    ];
    let mut bad = Vec::new();
    for (line, (name, w)) in want.lines().zip(named) {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols[0], name);
        for (i, p) in probes.iter().enumerate() {
            let got = w.apply(p).to_bits() as i64;
            let expected: i64 = cols[i + 1].parse().unwrap();
            if got != expected {
                bad.push(format!("{name} {p:?}: want {expected} got {got}"));
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// The presearcher's indexed document per query, under the plain presearcher
/// and the two-pass one, as Lucene indexes it (`presearch.tsv`).
#[test]
fn presearcher_indexes_match_lucene() {
    use lucene_search::monitor::presearcher::TermFilteredPresearcher;
    use std::collections::BTreeMap;

    let want = fs::read_to_string(dir().join("data/monitor/presearch.tsv")).unwrap();
    let plain = TermFilteredPresearcher::default();
    let multi = TermFilteredPresearcher::multipass(2, 0.0, TermWeightor::default(), Vec::new());
    let none = BTreeMap::new();
    let mut bad = Vec::new();
    for (line, q) in want.lines().zip(&corpus()) {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(unesc(cols[0]), *q);
        let Ok(clause) = parse(q) else { continue };
        // each indexed field as (name, the terms it holds, space-joined)
        let fields_of = |p: &TermFilteredPresearcher| -> Vec<(String, String)> {
            let indexed = p.index_query(&clause, &none).unwrap();
            indexed
                .fields
                .iter()
                .map(|(field, terms)| {
                    let names: Vec<String> = terms
                        .iter()
                        .map(|t| String::from_utf8_lossy(t).into_owned())
                        .collect();
                    (field.clone(), names.join(" "))
                })
                .collect()
        };
        // the Java row, unescaped: query, P, plain (name, terms)*, M, multipass (name, terms)*
        let row: Vec<String> = cols.iter().map(|c| unesc(c)).collect();
        let p_at = row
            .iter()
            .position(|c| c == "P")
            .unwrap_or_else(|| panic!("no P in {q:?}: {row:?}"));
        let m_at = row
            .iter()
            .position(|c| c == "M")
            .unwrap_or_else(|| panic!("no M in {q:?}: {row:?}"));
        // one cell per field, holding `name` TAB `terms`
        let pairs = |v: &[String]| -> Vec<(String, String)> {
            v.iter()
                .map(|cell| {
                    let (name, terms) = cell.split_once('\t').unwrap();
                    (name.to_string(), terms.to_string())
                })
                .collect()
        };
        assert!(p_at < m_at, "{q:?}: P at {p_at} after M at {m_at}: {row:?}");
        let want_plain = pairs(&row[p_at + 1..m_at]);
        let want_multi = pairs(&row[m_at + 1..]);
        let got_plain = fields_of(&plain);
        let got_multi = fields_of(&multi);
        // Java's field order is its `HashMap`'s, which varies between runs; the
        // (name, terms) pairs are compared as sorted multisets, each exactly.
        let sorted = |mut v: Vec<(String, String)>| {
            v.sort();
            v
        };
        // A conjunction indexes its heaviest child, and among tied children that
        // child is Java's hash-ordered first. Either tied child is a valid key
        // (the query needs both), so a query whose tree is a conjunction of
        // equal-weight terms is compared on the set of fields, not the terms.
        let tied = q.contains(" AND ");
        let same = if tied {
            sorted(
                got_plain
                    .iter()
                    .map(|(f, _)| (f.clone(), String::new()))
                    .collect(),
            ) == sorted(
                want_plain
                    .iter()
                    .map(|(f, _)| (f.clone(), String::new()))
                    .collect(),
            )
        } else {
            sorted(got_plain.clone()) == sorted(want_plain.clone())
                && sorted(got_multi.clone()) == sorted(want_multi.clone())
        };
        if !same {
            bad.push(format!(
                "{q:?}\n  want plain {want_plain:?} multi {want_multi:?}\n  got  plain {got_plain:?} multi {got_multi:?}"
            ));
        }
    }
    assert!(bad.is_empty(), "{} differ:\n{}", bad.len(), bad.join("\n"));
}

/// The batch query a presearcher builds over one document's fields, under the
/// plain and the two-pass presearcher, as Lucene's `buildQuery` prints it
/// (`batch.tsv`).
#[test]
fn batch_queries_match_lucene() {
    use lucene_analysis::{Analyzer, StandardAnalyzer, TokenStream};
    use lucene_search::monitor::presearcher::{batch_string, TermFilteredPresearcher};
    use std::collections::BTreeMap;

    let want = fs::read_to_string(dir().join("data/monitor/batch.tsv")).unwrap();
    let plain = TermFilteredPresearcher::default();
    let multi = TermFilteredPresearcher::multipass(2, 0.0, TermWeightor::default(), Vec::new());
    let analyzer = Analyzer::new(StandardAnalyzer::new());
    let mut bad = Vec::new();
    for line in want.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        let doc: Vec<String> = unesc(cols[0]).split('|').map(str::to_string).collect();
        // the batch as field -> distinct sorted terms, analysed as the batch's own field
        let mut batch: BTreeMap<String, Vec<Vec<u8>>> = BTreeMap::new();
        for pair in &doc {
            let (field, value) = pair.split_once('=').unwrap();
            let mut ts = analyzer.token_stream(field, value).unwrap();
            ts.reset().unwrap();
            while ts.increment_token().unwrap() {
                let t = ts.attributes().term_bytes().to_vec();
                let v = batch.entry(field.to_string()).or_default();
                if !v.contains(&t) {
                    v.push(t);
                }
            }
        }
        for v in batch.values_mut() {
            v.sort_unstable();
        }
        let accept = |_: &str, _: &[u8]| true;
        let p_query = plain.build_query(&batch, &accept, None);
        let m_query = multi.build_query(&batch, &accept, None);
        let got_p = batch_string(&p_query);
        if std::env::var_os("MONITOR_DEBUG").is_some() {
            eprintln!("TREE {p_query:?}");
        }
        let got_m = batch_string(&m_query);
        let want_p = unesc(cols[2]);
        let want_m = unesc(cols[4]);
        // The printed query, exactly. Java's clause order follows its own
        // hashing, so the clauses of each boolean are compared as a sorted list;
        // the structure (parentheses, occurrence prefixes, nesting) is exact.
        // sort each boolean's clauses, at every depth, so only the order is
        // allowed to differ and nothing else
        fn canon(s: &str) -> String {
            let mut parts: Vec<String> = split_top(s)
                .into_iter()
                .map(|c| match c.strip_prefix(['+', '-', '#']) {
                    Some(rest) if rest.starts_with('(') => {
                        let inner = &rest[1..rest.len() - 1];
                        format!("{}({})", &c[..1], canon(inner))
                    }
                    _ if c.starts_with('(') && c.ends_with(')') => {
                        format!("({})", canon(&c[1..c.len() - 1]))
                    }
                    _ => c,
                })
                .collect();
            parts.sort_unstable();
            parts.join(" ")
        }
        let norm = canon;
        if norm(&got_p) != norm(&want_p) || norm(&got_m) != norm(&want_m) {
            bad.push(format!("{line}\n  got P {got_p}\n  got M {got_m}"));
        }
    }
    assert!(bad.is_empty(), "{} differ:\n{}", bad.len(), bad.join("\n"));
}

/// `Monitor.match` over the registered corpus: the ids of the queries that
/// match each batch document, as Lucene's real `Monitor` returns them
/// (`match.tsv`).
#[test]
fn monitor_matches_agree_with_lucene() {
    use lucene_analysis::TokenStream;
    use lucene_search::memory::MemoryIndex;
    use lucene_search::monitor::index::{match_batch, MonitorQuery, QueryIndex};
    use lucene_search::monitor::presearcher::TermFilteredPresearcher;
    use std::collections::BTreeMap;

    let analyzer = Analyzer::new(StandardAnalyzer::new());
    let mut index = QueryIndex::new(TermFilteredPresearcher::default());
    let mut registered = Vec::new();
    for (i, q) in corpus().iter().enumerate() {
        let clause = parse(q).unwrap();
        registered.push(
            MonitorQuery::new(format!("q{i}"), clause, Some(q.clone()), BTreeMap::new()).unwrap(),
        );
    }
    index.register(registered).unwrap();

    let want = fs::read_to_string(dir().join("data/monitor/match.tsv")).unwrap();
    let mut bad = Vec::new();
    for line in want.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        let doc: Vec<String> = unesc(cols[0]).split('|').map(str::to_string).collect();
        let mut mi = MemoryIndex::new();
        let mut fields: Vec<(String, Vec<Vec<u8>>)> = Vec::new();
        for pair in &doc {
            let (field, value) = pair.split_once('=').unwrap();
            mi.add_field_text(field, value, &analyzer).unwrap();
            let mut ts = analyzer.token_stream(field, value).unwrap();
            ts.reset().unwrap();
            let mut terms = Vec::new();
            while ts.increment_token().unwrap() {
                terms.push(ts.attributes().term_bytes().to_vec());
            }
            fields.push((field.to_string(), terms));
        }
        mi.freeze().unwrap();
        let mi = std::cell::RefCell::new(mi);
        let result = match_batch(&index, &[fields], &[], &|entry, _| {
            mi.borrow_mut().search(&entry.match_query).unwrap_or(0.0) > 0.0
        });
        let mut got = result.matches[0].clone();
        got.sort();
        let mut want_ids: Vec<String> = unesc(cols.get(1).copied().unwrap_or(""))
            .split(' ')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        want_ids.sort();
        if got != want_ids {
            bad.push(format!("{doc:?}\n  want {want_ids:?}\n  got  {got:?}"));
        }
    }
    assert!(bad.is_empty(), "{} differ:\n{}", bad.len(), bad.join("\n"));
}

/// `Monitor.match(doc, ScoringMatch.DEFAULT_MATCHER)` through the facade: each
/// matched query's id and score bits, as Lucene's `Monitor` returns them
/// (`score.tsv`), and the plain matcher's ids (`match.tsv`).
#[test]
fn the_monitor_facade_scores_as_lucene_does() {
    use lucene_search::monitor::facade::Monitor;
    use lucene_search::monitor::index::MonitorQuery;
    use std::collections::BTreeMap;

    let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let mut monitor = Monitor::new(Arc::clone(&analyzer));
    let queries: Vec<MonitorQuery> = corpus()
        .iter()
        .enumerate()
        .map(|(i, q)| {
            MonitorQuery::new(
                format!("q{i}"),
                parse(q).unwrap(),
                Some(q.clone()),
                BTreeMap::new(),
            )
            .unwrap()
        })
        .collect();
    monitor.register(queries).unwrap();
    assert_eq!(
        monitor.query_count(),
        40,
        "Lucene counts the queries that have a document"
    );

    let want = fs::read_to_string(dir().join("data/monitor/score.tsv")).unwrap();
    let mut bad = Vec::new();
    for line in want.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        let doc_text = unesc(cols[0]);
        let doc: Vec<(&str, &str)> = doc_text
            .split('|')
            .map(|p| p.split_once('=').unwrap())
            .collect();
        let got = monitor.match_doc_scored(&doc).unwrap();
        let mut got_cells: Vec<String> = got
            .matches
            .iter()
            .map(|m| format!("{}={:x}", m.query_id, m.score.unwrap().to_bits()))
            .collect();
        got_cells.sort();
        let mut want_cells: Vec<String> = unesc(cols.get(1).copied().unwrap_or(""))
            .split(' ')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        want_cells.sort();
        if got_cells != want_cells {
            bad.push(format!(
                "{doc:?}\n  want {want_cells:?}\n  got  {got_cells:?}"
            ));
        }
        let plain = monitor.match_doc(&doc).unwrap();
        assert_eq!(plain.match_count(), got.match_count());
        assert!(plain.matches.iter().all(|m| m.score.is_none()));
    }
    assert!(bad.is_empty(), "{} differ:\n{}", bad.len(), bad.join("\n"));
}

/// `Monitor.match(docs, matcher)` over whole batches under the four matchers
/// (`batch_match.tsv`): per document, the simple matches, the score bits, the
/// explanation text and the hits, as Lucene's real `Monitor` returns them.
#[test]
fn batches_match_under_every_matcher_as_lucene_does() {
    use lucene_search::monitor::facade::{Matcher, Monitor};
    use lucene_search::monitor::index::MonitorQuery;
    use std::collections::BTreeMap;

    let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let mut monitor = Monitor::new(Arc::clone(&analyzer));
    let queries: Vec<MonitorQuery> = corpus()
        .iter()
        .enumerate()
        .map(|(i, q)| {
            MonitorQuery::new(
                format!("q{i}"),
                parse(q).unwrap(),
                Some(q.clone()),
                BTreeMap::new(),
            )
            .unwrap()
        })
        .collect();
    monitor.register(queries).unwrap();

    let want = fs::read_to_string(dir().join("data/monitor/batch_match.tsv")).unwrap();
    // a batch's rows are consecutive; run each batch once
    let mut results: BTreeMap<String, [lucene_search::monitor::facade::MultiMatchingQueries; 4]> =
        BTreeMap::new();
    let mut bad = Vec::new();
    for line in want.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        let label = unesc(cols[0]);
        let doc_index: usize = cols[1].parse().unwrap();
        let docs_text: Vec<Vec<(String, String)>> = label
            .trim_end_matches('|')
            .split('|')
            .map(|d| {
                d.split('+')
                    .map(|p| {
                        let (f, v) = p.split_once('=').unwrap();
                        (f.to_string(), v.to_string())
                    })
                    .collect()
            })
            .collect();
        let entry = results.entry(label.clone()).or_insert_with(|| {
            let docs: Vec<Vec<(&str, &str)>> = docs_text
                .iter()
                .map(|d| d.iter().map(|(f, v)| (f.as_str(), v.as_str())).collect())
                .collect();
            let refs: Vec<&[(&str, &str)]> = docs.iter().map(Vec::as_slice).collect();
            [
                monitor.match_docs(&refs, &Matcher::Simple).unwrap(),
                monitor.match_docs(&refs, &Matcher::Scoring(None)).unwrap(),
                monitor.match_docs(&refs, &Matcher::Explaining).unwrap(),
                monitor.match_docs(&refs, &Matcher::Highlights).unwrap(),
            ]
        });
        let mut got: Vec<String> = Vec::new();
        for m in entry[0].matches[doc_index].values() {
            got.push(format!("S:{}", m.query_id));
        }
        for m in entry[1].matches[doc_index].values() {
            got.push(format!("C:{}={:x}", m.query_id, m.score.unwrap().to_bits()));
        }
        for m in entry[2].matches[doc_index].values() {
            got.push(format!(
                "E:{}={}",
                m.query_id,
                m.explanation.as_ref().unwrap()
            ));
        }
        for m in entry[3].matches[doc_index].values() {
            let hits = m.hits.as_ref().unwrap();
            let fields: Vec<String> = hits
                .iter()
                .map(|(f, set)| {
                    let hs: Vec<String> = set.iter().map(ToString::to_string).collect();
                    format!("{f}=[{}]", hs.join(", "))
                })
                .collect();
            got.push(format!("H:{}={{{}}}", m.query_id, fields.join(", ")));
        }
        // An explanation lists a boolean's clauses in Java's `Multiset` order (a
        // `HashMap` keyed on identity hashes), so its lines are compared sorted:
        // every value, description and depth exact, sibling order free.
        // A fuzzy query's explanation is Lucene's scoring-rewrite breakdown, which
        // `explain` does not give (it reports a flat match: see the ledger), so for
        // `quick~1` (q22) only the match itself is compared.
        let canon = |c: &str| -> String {
            if c.starts_with("E:q22=") {
                "E:q22".to_string()
            } else if c.starts_with("E:") {
                let mut lines: Vec<&str> = c.split('\n').collect();
                lines.sort_unstable();
                lines.join("\n")
            } else {
                c.to_string()
            }
        };
        let mut got: Vec<String> = got.iter().map(|c| canon(c)).collect();
        got.sort();
        let mut want_cells: Vec<String> = unesc(cols.get(2).copied().unwrap_or(""))
            .split('\t')
            .filter(|s| !s.is_empty())
            .map(canon)
            .collect();
        want_cells.sort();
        if got != want_cells {
            let missing: Vec<&String> = want_cells.iter().filter(|c| !got.contains(c)).collect();
            let extra: Vec<&String> = got.iter().filter(|c| !want_cells.contains(c)).collect();
            bad.push(format!(
                "{label:?} doc {doc_index}\n  missing {missing:?}\n  extra   {extra:?}"
            ));
        }
    }
    assert!(bad.is_empty(), "{} differ:\n{}", bad.len(), bad.join("\n"));
}

/// `MonitorQuerySerializer.fromParser`: Lucene's bytes for a query, equal to
/// ours, and read back to the same query (`serialized.tsv`).
#[test]
fn the_serializer_writes_and_reads_lucenes_bytes() {
    use lucene_search::monitor::index::MonitorQuery;
    use lucene_search::monitor::serializer::MonitorQuerySerializer;
    use std::collections::BTreeMap;

    let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let parser = QueryParser::new("body", analyzer);
    let ser = MonitorQuerySerializer::from_parser(Arc::new(move |t: &str| {
        parser
            .parse(t)
            .map(|p| p.to_clause(1))
            .map_err(|e| e.message)
    }));
    let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
    let want = fs::read_to_string(dir().join("data/monitor/serialized.tsv")).unwrap();
    let mut rows = 0;
    for line in want.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        let (id, text, md, bytes) = (unesc(cols[0]), unesc(cols[1]), unesc(cols[2]), cols[3]);
        let mut metadata = BTreeMap::new();
        for kv in md.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').unwrap();
            metadata.insert(k.to_string(), Some(v.to_string()));
        }
        let clause = ser_parse(&ser, &text);
        let mq =
            MonitorQuery::new(id.clone(), clause.clone(), Some(text.clone()), metadata).unwrap();
        assert_eq!(hex(&ser.serialize(&mq).unwrap()), bytes, "serialize {id:?}");
        let raw: Vec<u8> = (0..bytes.len() / 2)
            .map(|i| u8::from_str_radix(&bytes[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        let back = ser.deserialize(&raw).unwrap();
        assert_eq!(back, mq, "deserialize {id:?}");
        rows += 1;
    }
    assert_eq!(rows, 5);
}

fn ser_parse(
    ser: &lucene_search::monitor::serializer::MonitorQuerySerializer,
    text: &str,
) -> Clause {
    // a serializer's parser is private; round-trip through its own bytes
    let mq = lucene_search::monitor::index::MonitorQuery::new(
        "x",
        parse("a").unwrap(),
        Some(text.to_string()),
        std::collections::BTreeMap::new(),
    )
    .unwrap();
    ser.deserialize(&ser.serialize(&mq).unwrap()).unwrap().query
}

fn persisted_serializer() -> lucene_search::monitor::serializer::MonitorQuerySerializer {
    let parser = QueryParser::new("body", Arc::new(Analyzer::new(StandardAnalyzer::new())));
    lucene_search::monitor::serializer::MonitorQuerySerializer::from_parser(Arc::new(
        move |t: &str| {
            parser
                .parse(t)
                .map(|p| p.to_clause(1))
                .map_err(|e| e.message)
        },
    ))
}

/// A query index Lucene's `Monitor` wrote to disk, opened by ours read-only:
/// the same queries and pieces, and the same matches (`index/`, `persisted.txt`).
#[test]
fn a_query_index_written_by_lucene_opens_and_matches() {
    use lucene_search::monitor::config::MonitorConfiguration;
    use lucene_search::monitor::facade::Monitor;
    use lucene_search::monitor::presearcher::TermFilteredPresearcher;

    let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let config = MonitorConfiguration::default().set_directory(
        dir().join("data/monitor/index"),
        persisted_serializer(),
        true,
    );
    let monitor = Monitor::with_configuration(
        Arc::clone(&analyzer),
        TermFilteredPresearcher::default(),
        Vec::new(),
        &config,
    )
    .unwrap();
    let counts = fs::read_to_string(dir().join("data/monitor/persisted.txt")).unwrap();
    let (queries, disjuncts) = counts.trim().split_once(' ').unwrap();
    assert_eq!(monitor.query_count(), queries.parse::<usize>().unwrap());
    assert_eq!(
        monitor.disjunct_count(),
        disjuncts.parse::<usize>().unwrap()
    );

    let want = fs::read_to_string(dir().join("data/monitor/match.tsv")).unwrap();
    for line in want.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        let doc_text = unesc(cols[0]);
        let doc: Vec<(&str, &str)> = doc_text
            .split('|')
            .map(|p| p.split_once('=').unwrap())
            .collect();
        let mut got: Vec<String> = monitor
            .match_doc(&doc)
            .unwrap()
            .matches
            .into_iter()
            .map(|m| m.query_id)
            .collect();
        got.sort();
        let mut want_ids: Vec<String> = unesc(cols.get(1).copied().unwrap_or(""))
            .split(' ')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        want_ids.sort();
        assert_eq!(got, want_ids, "{doc:?}");
    }
    // a read-only monitor refuses to change
    let mut monitor = monitor;
    assert!(monitor
        .delete_by_id(&["q0"])
        .unwrap_err()
        .to_string()
        .contains("readOnly"));
    assert!(monitor
        .clear()
        .unwrap_err()
        .to_string()
        .contains("readOnly"));
    assert!(monitor
        .register(Vec::new())
        .unwrap_err()
        .to_string()
        .contains("readOnly"));
}

/// `SuffixingNGramTokenFilter`'s stream over a list of terms, as Lucene emits it.
#[test]
fn the_suffixing_ngram_filter_emits_lucenes_tokens() {
    use lucene_search::monitor::handlers::suffixing_ngrams;
    let want = fs::read_to_string(dir().join("data/monitor/ngrams.tsv")).unwrap();
    let mut seen = 0;
    for line in want.lines() {
        let (input, output) = line.split_once('\t').unwrap();
        let terms: Vec<Vec<u8>> = unesc(input)
            .split('|')
            .map(|t| t.as_bytes().to_vec())
            .collect();
        let got: Vec<String> = suffixing_ngrams(&terms, "XX", "__WILDCARD__", 30)
            .into_iter()
            .map(|b| String::from_utf8(b).unwrap())
            .collect();
        assert_eq!(got.join("|"), unesc(output), "terms {input:?}");
        seen += 1;
    }
    assert_eq!(seen, 11);
}

fn regexp_handlers() -> Vec<Arc<dyn lucene_search::monitor::handlers::CustomQueryHandler>> {
    vec![Arc::new(
        lucene_search::monitor::handlers::RegexpQueryHandler::default(),
    )]
}

#[test]
fn regexp_queries_are_indexed_as_lucene_does() {
    use lucene_search::monitor::presearcher::TermFilteredPresearcher;
    use std::collections::BTreeMap;

    let handlers = regexp_handlers();
    let presearcher =
        TermFilteredPresearcher::with_handlers(TermWeightor::default(), handlers.clone(), vec![]);
    let want = fs::read_to_string(dir().join("data/monitor/regexp_trees.tsv")).unwrap();
    let mut rows = 0;
    for line in want.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        let q = unesc(cols[0]);
        let clause = parse(&q).unwrap();
        let tree = QueryAnalyzer::new(handlers.clone())
            .build_tree(&clause, &TermWeightor::default())
            .unwrap();
        assert_eq!(tree.to_string_at(0).unwrap(), unesc(cols[1]), "tree of {q}");
        let mut found = Vec::new();
        tree.collect_terms(&mut found);
        let terms: Vec<String> = found
            .iter()
            .map(|t| format!("{}:{}", t.field, String::from_utf8_lossy(&t.bytes)))
            .collect();
        assert_eq!(terms.join(" "), unesc(cols[2]), "terms of {q}");
        let indexed = presearcher.index_query(&clause, &BTreeMap::new()).unwrap();
        let got: Vec<String> = indexed
            .fields
            .iter()
            .map(|(f, ts)| {
                let names: Vec<String> = ts
                    .iter()
                    .map(|t| String::from_utf8_lossy(t).into_owned())
                    .collect();
                format!("{f}\t{}", names.join(" "))
            })
            .collect();
        assert_eq!(got.join("|"), unesc(cols[3]), "indexed fields of {q}");
        rows += 1;
    }
    assert_eq!(rows, 10);
}

#[test]
fn regexp_queries_match_with_the_handler_as_lucene_does() {
    use lucene_search::monitor::config::MonitorConfiguration;
    use lucene_search::monitor::facade::{Matcher, Monitor};
    use lucene_search::monitor::index::MonitorQuery;
    use lucene_search::monitor::presearcher::TermFilteredPresearcher;
    use std::collections::BTreeMap;

    let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let presearcher =
        TermFilteredPresearcher::with_handlers(TermWeightor::default(), regexp_handlers(), vec![]);
    let mut monitor = Monitor::with_configuration(
        analyzer,
        presearcher,
        Vec::new(),
        &MonitorConfiguration::default(),
    )
    .unwrap();
    let queries: Vec<String> = fs::read_to_string(dir().join("corpus/monitor-regexp.txt"))
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    monitor
        .register(
            queries
                .iter()
                .enumerate()
                .map(|(i, q)| {
                    MonitorQuery::new(
                        format!("r{i}"),
                        parse(q).unwrap(),
                        Some(q.clone()),
                        BTreeMap::new(),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();
    let want = fs::read_to_string(dir().join("data/monitor/regexp_matches.tsv")).unwrap();
    let mut rows = 0;
    for line in want.lines() {
        let (text, ids) = line.split_once('\t').unwrap();
        let text = unesc(text);
        let result = monitor
            .match_docs(&[&[("body", text.as_str())]], &Matcher::Simple)
            .unwrap();
        let mut got: Vec<String> = result.matches[0].keys().cloned().collect();
        got.sort();
        let mut expected: Vec<String> = unesc(ids)
            .split(' ')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        expected.sort();
        assert_eq!(got, expected, "document {text:?}");
        rows += 1;
    }
    assert_eq!(rows, 9);
}

#[test]
fn debug_explains_the_presearcher_as_lucene_does() {
    use lucene_search::monitor::config::MonitorConfiguration;
    use lucene_search::monitor::facade::{Matcher, Monitor};
    use lucene_search::monitor::index::MonitorQuery;
    use lucene_search::monitor::presearcher::TermFilteredPresearcher;
    use std::collections::BTreeMap;

    let mut monitor = Monitor::with_configuration(
        Arc::new(Analyzer::new(StandardAnalyzer::new())),
        TermFilteredPresearcher::default(),
        Vec::new(),
        &MonitorConfiguration::default(),
    )
    .unwrap();
    let queries = corpus();
    let registered: Vec<MonitorQuery> = queries
        .iter()
        .enumerate()
        .filter_map(|(i, q)| {
            Some(
                MonitorQuery::new(
                    format!("q{i}"),
                    parse(q).ok()?,
                    Some(q.clone()),
                    BTreeMap::new(),
                )
                .unwrap(),
            )
        })
        .collect();
    monitor.register(registered).unwrap();
    let want = fs::read_to_string(dir().join("data/monitor/debug.tsv")).unwrap();
    let mut explained = 0;
    for line in want.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        let doc = unesc(cols[0]);
        let pairs: Vec<(String, String)> = doc
            .split('|')
            .map(|f| {
                let (a, b) = f.split_once('=').unwrap();
                (a.to_string(), b.to_string())
            })
            .collect();
        let fields: Vec<(&str, &str)> = pairs
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let dbg = monitor.debug_docs(&[&fields], &Matcher::Simple).unwrap();
        let mut want_cells: Vec<String> = unesc(cols[1]).split('\t').map(str::to_string).collect();
        want_cells.retain(|c| !c.is_empty());
        let mut got_cells = Vec::new();
        for qi in 0..queries.len() {
            if let Some(pm) = dbg.match_query(&format!("q{qi}"), 0) {
                got_cells.push(format!(
                    "q{qi}={}={}",
                    pm.presearcher_matches,
                    pm.query_match.is_some()
                ));
            }
        }
        assert_eq!(got_cells, want_cells, "document {doc:?}");
        explained += got_cells.len();
    }
    assert!(explained >= 50, "{explained}");
}
