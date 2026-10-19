//! M14.f, differentially against Lucene 10.5.0:
//! `fixtures/src/GenSandboxPayloads.java`.
//!
//! lucene-sandbox's `PayloadSpanUtil.getPayloadsForQuery` over a
//! two-segment index with deletions: term, phrase, multi-phrase, boolean,
//! disjunction-max and span queries (a wrapped prefix among them), a query
//! it ignores, and the queries Java refuses -- every payload, in Java's
//! order (`payloads.tsv`).

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::{
    ExtendedQuery, MultiTermQuery, MultiTermSource, RewriteMethod,
};
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{
    BooleanQuery, Clause, DisjunctionMaxQuery, MultiPhraseQuery, PhraseQuery, PrefixQuery,
    TermQuery, WildcardQuery,
};
use lucene_search::sandbox::payloads::PayloadSpanUtil;
use lucene_search::spans::SpanNode;
use lucene_search::Error;
use lucene_store::FsDirectory;

fn data() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/sandbox_payloads")
}

/// `GenSandboxPayloads.query()`/`span()` over the spec's tokens.
struct Spec<'t> {
    toks: Vec<&'t str>,
    at: usize,
}

impl<'t> Spec<'t> {
    fn next(&mut self) -> &'t str {
        self.at += 1;
        self.toks[self.at - 1]
    }
    fn int(&mut self) -> i64 {
        self.next().parse().unwrap()
    }
    fn n(&mut self) -> usize {
        self.next().parse().unwrap()
    }

    fn query(&mut self) -> Clause {
        match self.next() {
            "T" => {
                let f = self.next();
                Clause::Term(TermQuery::new(f, self.next()))
            }
            "P" => {
                let slop = self.int() as u32;
                let n = self.n();
                Clause::Phrase(PhraseQuery {
                    field: "body".into(),
                    terms: (0..n).map(|_| self.next().as_bytes().to_vec()).collect(),
                    slop,
                    positions: Vec::new(),
                })
            }
            "MP" => {
                let slop = self.int() as u32;
                let n = self.n();
                let mut term_arrays = Vec::new();
                let mut positions = Vec::new();
                for _ in 0..n {
                    positions.push(self.int() as i32);
                    let k = self.n();
                    term_arrays.push((0..k).map(|_| self.next().as_bytes().to_vec()).collect());
                }
                Clause::MultiPhrase(MultiPhraseQuery {
                    field: "body".into(),
                    term_arrays,
                    slop,
                    positions,
                })
            }
            "B" => {
                let (m, s, n) = (self.n(), self.n(), self.n());
                let must = (0..m).map(|_| self.query()).collect();
                let should = (0..s).map(|_| self.query()).collect();
                let must_not = (0..n).map(|_| self.query()).collect();
                Clause::Boolean(Box::new(BooleanQuery {
                    must,
                    should,
                    must_not,
                    ..Default::default()
                }))
            }
            "BF" => {
                let (m, f, s) = (self.n(), self.n(), self.n());
                let must = (0..m).map(|_| self.query()).collect();
                let filter = (0..f).map(|_| self.query()).collect();
                let should = (0..s).map(|_| self.query()).collect();
                Clause::Boolean(Box::new(BooleanQuery {
                    must,
                    filter,
                    should,
                    ..Default::default()
                }))
            }
            "D" => {
                let n = self.n();
                let qs: Vec<Clause> = (0..n).map(|_| self.query()).collect();
                Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(qs, 0.1)))
            }
            "W" => {
                let f = self.next();
                Clause::Wildcard(WildcardQuery::new(f, self.next()))
            }
            _ => {
                self.at -= 1;
                Clause::Extended(Box::new(ExtendedQuery::Span(self.span())))
            }
        }
    }

    fn span(&mut self) -> SpanNode {
        match self.next() {
            "ST" => SpanNode::term("body", self.next()),
            "SN" => {
                let slop = self.int() as i32;
                let in_order: bool = self.next().parse().unwrap();
                let n = self.n();
                let cs = (0..n).map(|_| self.span()).collect();
                SpanNode::near(cs, slop, in_order).unwrap()
            }
            "SO" => {
                let n = self.n();
                SpanNode::or((0..n).map(|_| self.span()).collect()).unwrap()
            }
            "SP" => SpanNode::multi_term(MultiTermQuery::new(
                MultiTermSource::Prefix(PrefixQuery::new("body", self.next())),
                RewriteMethod::ScoringBoolean,
            )),
            other => panic!("{other}"),
        }
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn payloads_for_query_are_lucenes() {
    let text = std::fs::read_to_string(data().join("payloads.tsv")).unwrap();
    let reader = DirectoryReader::open(&FsDirectory::open(data().join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let norms: Vec<SegmentNorms<'_, '_>> = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let util = PayloadSpanUtil::new(&searcher);
    let mut bad = 0;
    for line in text.lines() {
        let (spec, want) = line.split_once('\t').unwrap();
        let mut s = Spec {
            toks: spec.split(' ').collect(),
            at: 0,
        };
        let q = s.query();
        assert_eq!(s.at, s.toks.len(), "{spec}");
        let got = match util.get_payloads_for_query(&q) {
            Ok(ps) => format!(
                "{}\t{}",
                ps.len(),
                ps.iter().map(|p| hex(p)).collect::<Vec<_>>().join(",")
            ),
            Err(Error::IllegalArgument(m)) => format!("IllegalArgumentException: {m}"),
            Err(Error::IllegalState(m)) => format!("IllegalStateException: {m}"),
            Err(e) => format!("{e:?}"),
        };
        if got != want {
            bad += 1;
            eprintln!("{spec}\n  java {want}\n  rust {got}");
        }
    }
    assert_eq!(bad, 0, "{bad} of {} lines differ", text.lines().count());
}
