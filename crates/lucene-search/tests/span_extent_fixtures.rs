//! **The span *extents* real Lucene's `Spans` iterator emits, not just the hit
//! set it produces.**
//!
//! `NearSpansOrdered` and `NearSpansUnordered` are iterators whose sub-span
//! cursors only ever move forward, so the arrangements they visit are a strict
//! subset of "every combination of one span per clause that fits the slop
//! budget". `NearSpansOrdered`'s own class doc says so -- "the formed spans
//! only contains minimum slop matches" -- and `NearSpansUnordered`'s
//! `endPosition()` is a *running* `maxEndPosition` that is never recomputed
//! when the span that set it moves on.
//!
//! This port enumerated the cartesian product until `c43-final-cleanup`, which
//! reported extents Lucene never produces. No hit set moved, because
//! `span_doc_ids` only asks whether the span list is non-empty -- which is
//! exactly why it survived three batches. What it moves is the extents, and a
//! nested `SpanNear`-of-`SpanNear` *consumes* its inner clause's extents, so
//! the `nested_*` cases here are hit-set cases as well.
//!
//! Ground truth: `fixtures/src/AppendSpanExtentManifest.java`, which walks real
//! `SpanWeight.getSpans(ctx, Postings.POSITIONS)` to `NO_MORE_POSITIONS` and
//! records every `(startPosition(), endPosition())` pair, per leaf, per
//! document. It is appended to the committed `multi_segment_scoring_index`
//! without regenerating it -- that fixture is the only committed Java-written
//! index whose position lists are rich enough to separate a walk from a
//! product (`GenMultiSegmentScoring.longBody` puts up to twenty occurrences of
//! one term in a document; `blocktree_index`'s `pos` field has two, where the
//! two algorithms agree on every query).
//!
//! The query is parsed from the same S-expression the Java recorded, by a twin
//! of the Java parser, so the recorded query and the tested query cannot drift.

#![allow(clippy::arithmetic_side_effects)] // Test arithmetic is not read off disk.

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::{span_doc_extents, SpanQuery};
use lucene_store::FsDirectory;

fn fixture_dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/multi_segment_scoring_index"
    )
    .to_string()
}

struct Manifest(Vec<(String, String)>);

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}/manifest.properties", fixture_dir()))
            .expect("run scripts/gen-fixtures.sh --append-only first");
        Manifest(
            text.lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        )
    }

    fn get(&self, key: &str) -> &str {
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }
}

/// `doc:start-end,start-end;doc:...` as Java rendered it, in the order the
/// `Spans` walk emitted the pairs.
fn parse_leaf(raw: &str) -> Vec<(i32, Vec<(i32, i32)>)> {
    if raw.is_empty() {
        return Vec::new();
    }
    raw.split(';')
        .map(|per_doc| {
            let (doc, spans) = per_doc.split_once(':').expect("doc:spans");
            let spans = spans
                .split(',')
                .map(|pair| {
                    let (start, end) = pair.split_once('-').expect("start-end");
                    (start.parse().unwrap(), end.parse().unwrap())
                })
                .collect();
            (doc.parse().unwrap(), spans)
        })
        .collect()
}

/// Twin of `AppendSpanExtentManifest.Parser`: `t(field,term)`,
/// `n(slop,inOrder,child,...)`, `o(child,...)`.
struct SpanParser<'a> {
    src: &'a [u8],
    at: usize,
}

impl<'a> SpanParser<'a> {
    fn parse(source: &'a str) -> SpanQuery {
        let mut p = SpanParser {
            src: source.as_bytes(),
            at: 0,
        };
        let q = p.expr();
        assert_eq!(p.at, p.src.len(), "trailing input in {source}");
        q
    }

    fn expr(&mut self) -> SpanQuery {
        let kind = self.src[self.at];
        self.at += 1;
        assert_eq!(self.src[self.at], b'(', "expected ( at {}", self.at);
        self.at += 1;
        let result = match kind {
            b't' => {
                let field = self.word();
                self.eat(b',');
                let term = self.word();
                SpanQuery::span_term(field, term.into_bytes())
            }
            b'n' => {
                let slop: u32 = self.word().parse().unwrap();
                self.eat(b',');
                let in_order: bool = self.word().parse().unwrap();
                let mut clauses = Vec::new();
                while self.src[self.at] == b',' {
                    self.at += 1;
                    clauses.push(self.expr());
                }
                SpanQuery::span_near(clauses, slop, in_order)
            }
            b'o' => {
                let mut clauses = vec![self.expr()];
                while self.src[self.at] == b',' {
                    self.at += 1;
                    clauses.push(self.expr());
                }
                SpanQuery::span_or(clauses)
            }
            other => panic!("unknown node kind {}", other as char),
        };
        assert_eq!(self.src[self.at], b')', "expected ) at {}", self.at);
        self.at += 1;
        result
    }

    fn word(&mut self) -> String {
        let start = self.at;
        while self.at < self.src.len()
            && (self.src[self.at].is_ascii_alphanumeric() || self.src[self.at] == b'_')
        {
            self.at += 1;
        }
        assert!(self.at > start, "expected a word at {start}");
        String::from_utf8(self.src[start..self.at].to_vec()).unwrap()
    }

    fn eat(&mut self, byte: u8) {
        assert_eq!(
            self.src[self.at], byte,
            "expected {} at {}",
            byte as char, self.at
        );
        self.at += 1;
    }
}

/// Every recorded case's whole extent sequence, per leaf, per document.
#[test]
fn span_extents_match_real_lucenes_spans_walk() {
    let m = Manifest::load();
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open reader");
    let leaf_count: usize = m.get("spanextent.leaf_count").parse().unwrap();
    assert_eq!(
        reader.segment_readers().len(),
        leaf_count,
        "the manifest was recorded over a different number of leaves"
    );
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();

    let cases: Vec<&str> = m.get("spanextent.cases").split(',').collect();
    let cases_len = cases.len();
    assert!(cases.len() >= 23, "the recorded case matrix shrank");

    let mut compared = 0usize;
    for case in cases {
        let source = m.get(&format!("spanextent.{case}.source")).to_string();
        let query = SpanParser::parse(&source);
        for (ord, seg) in segments.iter().enumerate() {
            let expected = parse_leaf(m.get(&format!("spanextent.{case}.leaf{ord}")));
            // Java's walk emits `(start, end)` pairs that are non-decreasing in
            // both components, so this port's sort-and-dedup can only remove
            // consecutive repeats. Assert the sequence really is ordered, or
            // the comparison below would be hiding a reordering.
            for (doc, spans) in &expected {
                for pair in spans.windows(2) {
                    assert!(
                        pair[0] < pair[1],
                        "{case} leaf{ord} doc {doc}: Lucene's own sequence is not strictly \
                         increasing at {:?} -> {:?}",
                        pair[0],
                        pair[1]
                    );
                }
            }
            let got = span_doc_extents(
                seg.fields,
                seg.doc_in,
                seg.pos_in,
                seg.pay_in,
                seg.live_docs,
                &query,
            )
            .unwrap_or_else(|e| panic!("{case} leaf{ord}: {e}"));
            assert_eq!(
                got, expected,
                "{case} leaf{ord}: `{source}` -- this port's span extents differ from real \
                 Lucene's `Spans` walk"
            );
            compared += 1;
        }
    }
    assert_eq!(
        compared,
        cases_len * leaf_count,
        "every case must be compared against every leaf"
    );
}

/// The extents are not decoration: a nested `SpanNear` consumes its inner
/// clause's extents, so an extent Lucene never produces becomes a **document**
/// this port returns and Lucene does not.
///
/// This asserts the hit sets separately from the extents above, through the
/// same public entry point a caller uses, so the consequence is pinned even if
/// the extent comparison is ever loosened.
#[test]
fn nested_span_near_hit_sets_match_real_lucene() {
    let m = Manifest::load();
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open reader");
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();

    let mut any_hits = false;
    for case in m.get("spanextent.cases").split(',') {
        if !case.starts_with("nested_") {
            continue;
        }
        let source = m.get(&format!("spanextent.{case}.source")).to_string();
        let query = SpanParser::parse(&source);
        for (ord, seg) in segments.iter().enumerate() {
            let expected: Vec<i32> = parse_leaf(m.get(&format!("spanextent.{case}.leaf{ord}")))
                .into_iter()
                .map(|(doc, _)| doc)
                .collect();
            let mut collector = lucene_search::VecCollector::default();
            lucene_search::search_span_query(
                seg.fields,
                seg.doc_in,
                seg.pos_in,
                seg.pay_in,
                seg.live_docs,
                &query,
                &mut collector,
            )
            .unwrap_or_else(|e| panic!("{case} leaf{ord}: {e}"));
            let mut got = collector.docs.clone();
            got.sort_unstable();
            assert_eq!(got, expected, "{case} leaf{ord}: `{source}`");
            any_hits |= !expected.is_empty();
        }
    }
    assert!(any_hits, "every nested case matched nothing -- vacuous");
}
