#![allow(clippy::arithmetic_side_effects)]
//! **The Matches API against real Lucene.**
//!
//! `fixtures/src/GenMatches.java` writes two segments (the first with
//! deletions) with a positions-and-offsets field, a positions-only field, a
//! freqs-only field and a long point, and records `Weight.matches(ctx, doc)`
//! for 33 queries over every document, deleted ones included: which
//! documents match, each field's matches in iterator order as `[start
//! position, end position, start offset, end offset]` with the class of the
//! query each came from, and `MATCH_WITH_NO_TERMS`. The queries cover terms,
//! exact and sloppy phrases (reordered, repeated terms), a one-term phrase,
//! prefix, wildcard, regexp and term-set expansions, booleans with required,
//! prohibited, filter and minimum-should-match clauses, dis-max, constant
//! score, boost, match-all, points ranges, fields without offsets and without
//! positions. Named clauses are checked through `findNamedMatches`.

mod m7support;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::matches::{self, clause_name, find_named_matches, named_matches, Matches};
use lucene_store::FsDirectory;
use m7support::{fixture, Grammar, Manifest};

const GRAMMAR: Grammar = Grammar {
    text: "body",
    range: "r",
};

fn render(m: &dyn Matches) -> String {
    if m.is_match_with_no_terms() {
        return "noterms".to_string();
    }
    let mut out = Vec::new();
    for field in m.fields() {
        let mut s = format!("{field}|");
        let mut first = true;
        if let Some(mut it) = m.get_matches(&field).unwrap() {
            while it.next().unwrap() {
                if !first {
                    s.push(',');
                }
                first = false;
                s.push_str(&format!(
                    "{}:{}:{}:{}:{}",
                    it.start_position(),
                    it.end_position(),
                    it.start_offset(),
                    it.end_offset(),
                    clause_name(it.query())
                ));
                assert!(it.sub_matches().unwrap().is_none());
            }
        }
        out.push(s);
    }
    out.join(";")
}

#[test]
fn matches_match_real_lucene() {
    let dir = fixture("matches_index");
    let m = Manifest::load(&format!("{dir}/manifest.properties"));
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).expect("open reader");
    let mut opened = reader.open_segments().expect("open postings");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let max_doc = searcher.max_doc();
    assert!(segments[0].live_docs.is_some(), "deletions in the fixture");

    let queries: usize = m.get("query_count").parse().unwrap();
    let mut compared = 0usize;
    let mut failures = Vec::new();
    for q in 0..queries {
        let text = m.get(&format!("q.{q}"));
        let clause = GRAMMAR.clause(text);
        let want_docs: Vec<i32> = m
            .get(&format!("docs.{q}"))
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().unwrap())
            .collect();
        let mut got_docs = Vec::new();
        for doc in 0..max_doc {
            let Some(got) = matches::matches(&searcher, &clause, doc).unwrap() else {
                continue;
            };
            got_docs.push(doc);
            let Some(want) = m.opt(&format!("m.{q}.{doc}")) else {
                continue;
            };
            compared += 1;
            let got = render(got.as_ref());
            if got != want {
                failures.push(format!("{text} doc {doc}:\n  got  {got}\n  want {want}"));
            }
        }
        if got_docs != want_docs {
            failures.push(format!(
                "{text}: matching docs {} vs Lucene's {}",
                got_docs.len(),
                want_docs.len()
            ));
        }
    }
    assert!(compared > 2000, "compared {compared}");
    assert!(
        failures.is_empty(),
        "{} disagreements:\n{}",
        failures.len(),
        failures
            .iter()
            .take(12)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );

    // Named clauses: the names `findNamedMatches` reports per document.
    let named: Vec<(String, lucene_search::Clause)> = ["a", "b", "c", "d"]
        .iter()
        .map(|n| {
            (
                n.to_string(),
                GRAMMAR.clause(m.get(&format!("named.q.{n}"))),
            )
        })
        .collect();
    let mut named_docs = 0;
    for doc in 0..max_doc {
        let leaf = searcher.segment_of(doc).unwrap();
        let seg = &searcher.segments()[leaf];
        let found = named_matches(seg, &named, doc - seg.doc_base).unwrap();
        let got: Vec<String> = if found.is_empty() {
            Vec::new()
        } else {
            let tree = matches::from_sub_matches(
                found
                    .into_iter()
                    .map(|n| Box::new(n) as matches::BoxMatches)
                    .collect(),
            )
            .unwrap();
            let mut names: Vec<String> = find_named_matches(tree.as_ref())
                .iter()
                .map(|n| n.name().to_string())
                .collect();
            names.sort();
            names
        };
        let want: Vec<String> = m
            .opt(&format!("named.{doc}"))
            .map(|s| s.split(',').map(str::to_string).collect())
            .unwrap_or_default();
        assert_eq!(got, want, "named, doc {doc}");
        named_docs += usize::from(!want.is_empty());
    }
    assert!(named_docs > 50);

    // What is not ported says so.
    let fuzzy = lucene_search::Clause::Fuzzy(lucene_search::FuzzyQuery::new("body", "w1"));
    assert!(matches::matches(&searcher, &fuzzy, 1).is_err());
    assert!(matches::matches(&searcher, &GRAMMAR.clause("(t w0)"), max_doc).is_err());
    let no_positions = GRAMMAR.clause("(pf tag 0 w1 w2)");
    let err = (0..max_doc).find_map(|d| matches::matches(&searcher, &no_positions, d).err());
    assert!(err.is_some(), "a phrase over a field without positions");
}
