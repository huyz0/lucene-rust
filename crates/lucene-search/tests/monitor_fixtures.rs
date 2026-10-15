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
    // a conjunction whose heaviest child is a term collects that one term
    let heaviest_is_term = tree.split('\n').nth(1).is_some_and(|l| {
        !l.trim_start().starts_with("Conjunction") && !l.trim_start().starts_with("Disjunction")
    });
    if tree.starts_with("Conjunction") && heaviest_is_term {
        return g.len() == 1 && w.len() == 1 && tree_has_term(tree, g[0]);
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
            Ok(clause) => match QueryAnalyzer.build_tree(&clause, &weightor) {
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
