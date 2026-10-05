#![allow(clippy::arithmetic_side_effects)]
//! Every composite scorer against a brute-force evaluation of the same tree.
//!
//! The leaves are synthetic ([`Fake`]): a set of matching documents with
//! scores, optionally behind a two-phase approximation that also stops on
//! documents that do not match, and with block maxima over blocks of eight
//! documents so the pruning paths have real bounds to skip on. Scores are
//! multiples of 1/8, so every sum is exact in `f32` and `f64` alike and the
//! comparison can be exact.
//!
//! Real postings run through the same composites in
//! `tests/mixed_boolean_fixtures.rs`, against Lucene; what this adds is what
//! that corpus cannot reach -- two-phase clauses (no leaf in this crate is
//! two-phase yet) and thousands of random tree shapes.

// Turns the batched scoring paths off (`super::batches_on`), for a test
// comparing them with the document-at-a-time ones.
thread_local! {
    pub(crate) static BATCHES_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

use super::build::{compose, Child};
use super::disjunction::{Combine, DisjunctionScorer};
use super::leaf::{AllDocs, ConstantScorer, DocList};
use super::{score_segment, BoxScorer, Bulk, Mode, Scorer, NO_MORE_DOCS};
use crate::collector::{ScoreDoc, ScoringCollector, TopDocsCollector};
use crate::Result;

/// A small deterministic generator (SplitMix64), so a failure reproduces.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.below(100) < pct
    }
}

const MAX_DOC: i32 = 600;
const BLOCK: i32 = 8;

/// A synthetic leaf. See the module doc.
#[derive(Clone)]
struct Leaf {
    /// Ascending (doc, score) matches.
    hits: Vec<(i32, f32)>,
    /// Ascending approximation: `hits`' docs plus, when two-phase, extras.
    approx: Vec<i32>,
    two_phase: bool,
    /// Per block of [`BLOCK`] documents, the best score in it.
    block_max: Vec<f32>,
}

impl Leaf {
    fn new(hits: Vec<(i32, f32)>, approx: Vec<i32>, two_phase: bool) -> Self {
        let mut block_max = vec![0.0f32; (MAX_DOC / BLOCK + 1) as usize];
        for &(d, s) in &hits {
            let b = &mut block_max[(d / BLOCK) as usize];
            *b = b.max(s);
        }
        Leaf {
            hits,
            approx,
            two_phase,
            block_max,
        }
    }

    fn random(rng: &mut Rng) -> Self {
        let density = 1 + rng.below(60);
        let two_phase = rng.chance(30);
        let mut hits = Vec::new();
        let mut approx = Vec::new();
        for d in 0..MAX_DOC {
            if rng.below(100) < density {
                // 1/8 .. 32/8
                hits.push((d, (1 + rng.below(32)) as f32 / 8.0));
                approx.push(d);
            } else if two_phase && rng.chance(20) {
                approx.push(d);
            }
        }
        Leaf::new(hits, approx, two_phase)
    }

    fn score_of(&self, doc: i32) -> Option<f32> {
        self.hits
            .binary_search_by_key(&doc, |h| h.0)
            .ok()
            .map(|i| self.hits[i].1)
    }
}

struct Fake {
    leaf: Leaf,
    at: usize,
    doc: i32,
    shallow: i32,
}

impl Fake {
    fn new(leaf: Leaf) -> Self {
        Fake {
            leaf,
            at: 0,
            doc: -1,
            shallow: 0,
        }
    }
}

impl Scorer for Fake {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.advance(self.doc + 1)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        assert!(target > self.doc, "advance({target}) from {}", self.doc);
        while self.at < self.leaf.approx.len() && self.leaf.approx[self.at] < target {
            self.at += 1;
        }
        self.doc = self
            .leaf
            .approx
            .get(self.at)
            .copied()
            .unwrap_or(NO_MORE_DOCS);
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        self.leaf.approx.len() as i64
    }
    fn two_phase(&self) -> bool {
        self.leaf.two_phase
    }
    fn matches(&mut self) -> Result<bool> {
        Ok(self.leaf.score_of(self.doc).is_some())
    }
    fn match_cost(&self) -> f32 {
        if self.leaf.two_phase {
            3.0
        } else {
            0.0
        }
    }
    fn score(&mut self) -> Result<f32> {
        Ok(self
            .leaf
            .score_of(self.doc)
            .expect("score() only on a matching document"))
    }
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        self.shallow = target.max(0);
        if self.shallow >= MAX_DOC {
            return Ok(NO_MORE_DOCS);
        }
        Ok((self.shallow / BLOCK + 1) * BLOCK - 1)
    }
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        // Whole blocks: a bound, as impacts are, not an exact maximum.
        let first = (self.shallow / BLOCK).min(self.leaf.block_max.len() as i32) as usize;
        let last = (up_to.min(MAX_DOC - 1) / BLOCK) as usize;
        Ok(self
            .leaf
            .block_max
            .get(
                first
                    ..=last
                        .max(first)
                        .min(self.leaf.block_max.len().saturating_sub(1)),
            )
            .unwrap_or(&[])
            .iter()
            .copied()
            .fold(0.0, f32::max))
    }
}

/// A query tree over [`Leaf`]s.
#[derive(Clone)]
enum Node {
    Leaf(Leaf),
    Bool {
        must: Vec<Node>,
        filter: Vec<Node>,
        should: Vec<Node>,
        must_not: Vec<Node>,
        msm: usize,
    },
    Const(f32, Box<Node>),
    DisMax(f32, Vec<Node>),
    All,
}

fn random_node(rng: &mut Rng, depth: u32) -> Node {
    if depth == 0 || rng.chance(35) {
        return if rng.chance(3) {
            Node::All
        } else {
            Node::Leaf(Leaf::random(rng))
        };
    }
    match rng.below(10) {
        0 => Node::Const(
            (1 + rng.below(8)) as f32 / 4.0,
            Box::new(random_node(rng, depth - 1)),
        ),
        1 => {
            // Up to seven: `DisiQueue` scans four or fewer and heaps more.
            let n = 2 + rng.below(6) as usize;
            let tie = if rng.chance(50) { 0.0 } else { 0.5 };
            Node::DisMax(tie, (0..n).map(|_| random_node(rng, depth - 1)).collect())
        }
        _ => {
            let mut v = |max: u64| -> Vec<Node> {
                (0..rng.below(max + 1))
                    .map(|_| random_node(rng, depth - 1))
                    .collect::<Vec<_>>()
            };
            let (must, filter, should, must_not) = (v(2), v(2), v(4), v(2));
            let msm = if should.is_empty() || rng.chance(50) {
                0
            } else {
                rng.below(should.len() as u64 + 1) as usize
            };
            Node::Bool {
                must,
                filter,
                should,
                must_not,
                msm,
            }
        }
    }
}

/// The scorer tree for `node`, as `build` would compose it.
fn build(node: &Node, mode: Mode, top_level: bool) -> Option<BoxScorer<'static>> {
    match node {
        Node::Leaf(l) => {
            if l.approx.is_empty() {
                None
            } else {
                Some(Box::new(Fake::new(l.clone())))
            }
        }
        Node::All => Some(Box::new(ConstantScorer::new(
            Box::new(AllDocs::new(MAX_DOC)),
            1.0,
            mode == Mode::TopScores,
        ))),
        Node::Const(score, inner) => {
            let inner = build(inner, Mode::NoScores, false)?;
            if !mode.needs_scores() {
                return Some(inner);
            }
            Some(Box::new(ConstantScorer::new(
                inner,
                *score,
                mode == Mode::TopScores,
            )))
        }
        Node::DisMax(tie, nodes) => {
            let mut subs: Vec<BoxScorer<'static>> =
                nodes.iter().filter_map(|n| build(n, mode, false)).collect();
            match subs.len() {
                0 => None,
                1 => subs.pop(),
                _ => {
                    let d = DisjunctionScorer::new(subs, Combine::Max(*tie), mode.needs_scores());
                    Some(Box::new(if mode == Mode::TopScores {
                        d.with_block_propagator().unwrap()
                    } else {
                        d
                    }))
                }
            }
        }
        Node::Bool {
            must,
            filter,
            should,
            must_not,
            msm,
        } => {
            let child_top = top_level && must.len() + should.len() == 1;
            let must: Option<Vec<_>> = must
                .iter()
                .map(|n| build(n, mode, child_top).map(Child::Scorer))
                .collect();
            let filter: Option<Vec<_>> = filter
                .iter()
                .map(|n| build(n, Mode::NoScores, false).map(Child::Scorer))
                .collect();
            let should = should
                .iter()
                .filter_map(|n| build(n, mode, child_top).map(Child::Scorer))
                .collect();
            let must_not = must_not
                .iter()
                .filter_map(|n| build(n, Mode::NoScores, false).map(Child::Scorer))
                .collect();
            compose(must?, filter?, should, must_not, *msm, mode, top_level).unwrap()
        }
    }
}

/// Brute force: does `node` match `doc`, and with what score.
fn eval(node: &Node, doc: i32) -> Option<f64> {
    match node {
        Node::Leaf(l) => l.score_of(doc).map(f64::from),
        Node::All => Some(1.0),
        Node::Const(s, inner) => eval(inner, doc).map(|_| f64::from(*s)),
        Node::DisMax(tie, nodes) => {
            let scores: Vec<f64> = nodes.iter().filter_map(|n| eval(n, doc)).collect();
            if scores.is_empty() {
                return None;
            }
            let max = scores.iter().copied().fold(0.0, f64::max);
            let sum: f64 = scores.iter().sum();
            Some(max + (sum - max) * f64::from(*tie))
        }
        Node::Bool {
            must,
            filter,
            should,
            must_not,
            msm,
        } => {
            if must.is_empty() && filter.is_empty() && should.is_empty() {
                return None;
            }
            let mut score = 0.0;
            for n in must {
                score += eval(n, doc)?;
            }
            for n in filter {
                eval(n, doc)?;
            }
            if must_not.iter().any(|n| eval(n, doc).is_some()) {
                return None;
            }
            let matched: Vec<f64> = should.iter().filter_map(|n| eval(n, doc)).collect();
            let needed = if must.is_empty() && filter.is_empty() {
                (*msm).max(1)
            } else {
                *msm
            };
            if matched.len() < needed {
                return None;
            }
            Some(score + matched.iter().sum::<f64>())
        }
    }
}

/// Every hit, as `(doc, score)`.
struct All(Vec<(i32, f32)>);

impl ScoringCollector for All {
    fn collect(&mut self, doc_id: i32, score: f32) {
        self.0.push((doc_id, score));
    }
}

fn expected(node: &Node, scored: bool) -> Vec<(i32, f32)> {
    (0..MAX_DOC)
        .filter_map(|d| eval(node, d).map(|s| (d, if scored { s as f32 } else { 0.0 })))
        .collect()
}

fn top_k(mut hits: Vec<(i32, f32)>, k: usize) -> Vec<(i32, f32)> {
    hits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    hits.truncate(k);
    hits
}

fn pairs(docs: &[ScoreDoc]) -> Vec<(i32, f32)> {
    docs.iter().map(|h| (h.doc_id, h.score)).collect()
}

#[test]
fn random_trees_match_brute_force_in_every_mode() {
    let mut rng = Rng(0x5eed);
    let mut pruned_runs = 0;
    for case in 0..1500 {
        let node = random_node(&mut rng, 3);
        let top_level = node_is_bool(&node);
        let label = format!("case {case}");

        // COMPLETE: every match, with its score.
        let mut all = All(Vec::new());
        if let Some(s) = build(&node, Mode::Complete, top_level) {
            score_segment(&mut Bulk::scorer(s), Mode::Complete, None, &mut all).unwrap();
        }
        assert_eq!(all.0, expected(&node, true), "{label}: complete");

        // COMPLETE_NO_SCORES: every match.
        let mut none = All(Vec::new());
        if let Some(s) = build(&node, Mode::NoScores, top_level) {
            score_segment(&mut Bulk::scorer(s), Mode::NoScores, None, &mut none).unwrap();
        }
        assert_eq!(none.0, expected(&node, false), "{label}: no scores");

        // TOP_SCORES with a small threshold, so pruning starts early.
        let k = 1 + rng.below(12) as usize;
        let mut top = TopDocsCollector::with_total_hits_threshold(k, k as u64);
        if let Some(s) = build(&node, Mode::TopScores, top_level) {
            score_segment(&mut Bulk::scorer(s), Mode::TopScores, None, &mut top).unwrap();
        }
        let want = top_k(expected(&node, true), k);
        assert_eq!(pairs(top.top_docs()), want, "{label}: top {k}");
        if top.total_hits().value < expected(&node, true).len() as u64 {
            pruned_runs += 1;
        }
    }
    assert!(
        pruned_runs > 100,
        "pruning must actually skip documents in a good share of cases, got {pruned_runs}"
    );
}

/// `MaxScoreBulkScorer` over arbitrary scorers (`Bulk::ScorerDisjunction`),
/// two-phase ones included, with and without a filter: every surviving hit
/// and score must be brute force's.
#[test]
fn scorer_disjunctions_match_brute_force() {
    use super::bulk::ScorerLeg;
    use crate::bulk_scorer::MaxScore;
    let mut rng = Rng(0xd15c);
    let mut two_phase_runs = 0;
    for case in 0..600 {
        let should: Vec<Node> = (0..2 + rng.below(5))
            .map(|_| random_node(&mut rng, 2))
            .collect();
        let filter: Vec<Node> = if rng.chance(40) {
            vec![random_node(&mut rng, 1)]
        } else {
            Vec::new()
        };
        let node = Node::Bool {
            must: Vec::new(),
            filter: filter.clone(),
            should: should.clone(),
            must_not: Vec::new(),
            msm: if filter.is_empty() { 0 } else { 1 },
        };
        let k = 1 + rng.below(10) as usize;
        let mut legs: Vec<ScorerLeg<'static>> = Vec::new();
        for child in &should {
            if let Some(s) = build(child, Mode::TopScores, false) {
                if s.two_phase() {
                    two_phase_runs += 1;
                }
                legs.push(ScorerLeg::new(s));
            }
        }
        let filter_scorer = match filter.first() {
            Some(f) => match build(f, Mode::NoScores, false) {
                Some(s) => Some(s),
                None => continue,
            },
            None => None,
        };
        if legs.is_empty() {
            continue;
        }
        let state = MaxScore::new(&mut legs);
        let mut bulk = Bulk::ScorerDisjunction(legs, filter_scorer, state);
        let mut top = TopDocsCollector::with_total_hits_threshold(k, k as u64);
        score_segment(&mut bulk, Mode::TopScores, None, &mut top).unwrap();
        assert_eq!(
            pairs(top.top_docs()),
            top_k(expected(&node, true), k),
            "case {case}: top {k}"
        );
    }
    assert!(two_phase_runs > 50, "two-phase clauses: {two_phase_runs}");
}

/// The query cache: a non-scoring clause used again and again is cached per
/// segment once the policy says so (a phrase after five uses), and every run
/// (and a term set's rewritten disjunction under a scoring clause) -- uncached,
/// the one that builds the entry, the ones served from it --
/// returns the same hits and score bits, deletions included (the cached set
/// is the core's, live docs are applied after).
#[test]
fn cached_clauses_search_exactly_like_uncached_ones() {
    use crate::directory_reader::DirectoryReader;
    use crate::field_norms::FieldNorms;
    use crate::query::{BooleanQuery, Clause, PhraseQuery, TermQuery};
    use std::collections::HashMap;
    let dir = lucene_store::FsDirectory::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/mixed_boolean_scoring_index"
    ));
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    assert!(segments.iter().any(|s| s.live_docs.is_some()));
    let owned: Vec<HashMap<String, FieldNorms<'_>>> = reader
        .field_norms("body")
        .into_iter()
        .map(|n| n.into_iter().map(|n| ("body".to_string(), n)).collect())
        .collect();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let term = |w: &str| Clause::Term(TermQuery::new("body", w.as_bytes().to_vec()));
    let phrase = |a: &str, b: &str| Clause::Phrase(PhraseQuery::new("body", [a, b]));
    let mut excluded = BooleanQuery::new();
    excluded.must.push(term("w3"));
    excluded.must_not.push(phrase("w0", "w1"));
    let mut filtered = BooleanQuery::new();
    filtered.filter.push(phrase("w0", "w2"));
    filtered.should.push(term("w1"));
    filtered.should.push(term("w4"));
    // A term set as a scoring `MUST`: its rewrite's `BooleanQuery` of terms
    // is built without scores inside `ConstantScoreQuery`, so it is cached
    // too (after four uses, a compound query) while the clause still scores.
    let mut term_set = BooleanQuery::new();
    term_set
        .must
        .push(Clause::TermInSet(crate::query::TermInSetQuery::new(
            "body",
            ["w3", "w5", "w7"],
        )));
    term_set.should.push(term("w0"));
    for q in [&excluded, &filtered, &term_set] {
        let run = || {
            crate::multi_segment::search_boolean_query_multi_segment_maxscore_counting(
                &segments,
                q,
                &norms,
                10,
                u64::MAX,
            )
            .unwrap()
        };
        let (first, first_total) = run();
        let bits = |h: &[crate::ScoreDoc]| -> Vec<(i32, u32)> {
            h.iter().map(|d| (d.doc_id, d.score.to_bits())).collect()
        };
        for i in 0..7 {
            let (hits, total) = run();
            assert_eq!(bits(&hits), bits(&first), "run {i} of {q:?}");
            assert_eq!(total.value, first_total.value, "run {i} of {q:?}");
        }
    }
    for r in reader.segment_readers() {
        let (entries, bytes) = r.query_cache().stats();
        assert_eq!(
            entries, 3,
            "both phrases and the term set cached in {}",
            r.segment_name
        );
        assert!(bytes > 0);
    }
}

/// A cached term set as a scoring `MUST` beside one optional term
/// (`Bulk::ReqBitsOpt`): with a count threshold small enough that the
/// collector's threshold passes the set's constant score, the walk switches
/// from the set's bits to the term's postings. Every run -- the uncached
/// first ones through `ReqOptSumScorer`, the cached ones through the
/// specialised bulk scorer -- returns the same hits and score bits, whatever
/// the threshold and the number of hits, deletions included.
#[test]
fn a_cached_required_set_with_an_optional_term_scores_like_req_opt_sum() {
    use crate::directory_reader::DirectoryReader;
    use crate::field_norms::FieldNorms;
    use crate::query::{BooleanQuery, Clause, TermInSetQuery, TermQuery};
    use std::collections::HashMap;
    let dir = lucene_store::FsDirectory::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/mixed_boolean_scoring_index"
    ));
    let bits = |h: &[crate::ScoreDoc]| -> Vec<(i32, u32)> {
        h.iter().map(|d| (d.doc_id, d.score.to_bits())).collect()
    };
    crate::bulk_scorer::test_only_req_bits_opt_phases::take();
    // Each with and without a `MUST_NOT` (which wraps the bulk scorer in
    // `ReqExcl`, re-entering it window by window).
    for (set, opt, not) in [
        (["w3", "w5", "w7"], "w0", None),
        (["w1", "w2", "w9"], "w4", None),
        (["w3", "w5", "w7"], "w0", Some("w8")),
        (["w1", "w2", "w9"], "w4", Some("w6")),
    ] {
        for (top_n, threshold) in [(10, 10u64), (3, 0), (5, 50), (20, 1000), (1, 1)] {
            // A fresh reader: every combination starts uncached.
            let reader = DirectoryReader::open(&dir).unwrap();
            let opened = reader.open_segments().unwrap();
            let segments = opened.as_open_segments();
            assert!(segments.iter().any(|s| s.live_docs.is_some()));
            let owned: Vec<HashMap<String, FieldNorms<'_>>> = reader
                .field_norms("body")
                .into_iter()
                .map(|n| n.into_iter().map(|n| ("body".to_string(), n)).collect())
                .collect();
            let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> =
                owned.iter().map(Some).collect();
            let mut q = BooleanQuery::new();
            q.must
                .push(Clause::TermInSet(TermInSetQuery::new("body", set)));
            q.should.push(Clause::Term(TermQuery::new(
                "body",
                opt.as_bytes().to_vec(),
            )));
            let run = || {
                crate::multi_segment::search_boolean_query_multi_segment_maxscore_counting(
                    &segments, &q, &norms, top_n, threshold,
                )
                .unwrap()
                .0
            };
            let first = run();
            assert!(!first.is_empty());
            for i in 0..7 {
                assert_eq!(
                    bits(&run()),
                    bits(&first),
                    "run {i}, {set:?} + {opt} - {not:?}, top {top_n}, threshold {threshold}"
                );
            }
            for r in reader.segment_readers() {
                assert_eq!(
                    r.query_cache().stats().0,
                    1,
                    "the set cached in {}",
                    r.segment_name
                );
            }
        }
    }
    // The specialised scorer ran, in both of its phases: the cached runs
    // above did not all fall back to the generic `ReqOptSumScorer`.
    let [required, optional] = crate::bulk_scorer::test_only_req_bits_opt_phases::take();
    assert!(
        required > 0 && optional > 0,
        "phases reached: {required}, {optional}"
    );
}

/// A lone term set, top hits with a small count threshold: once its
/// rewritten disjunction is cached, the bulk scorer walks the cached bit set
/// word by word and stops as soon as the threshold passes the constant score.
/// Every run -- streamed-free uncached ones, the one that builds the entry,
/// the cached ones -- returns the same hits and count relation, deletions
/// included; and a count with no threshold walks the whole set.
#[test]
fn a_cached_term_set_is_walked_as_a_bit_set() {
    use crate::directory_reader::DirectoryReader;
    use crate::field_norms::FieldNorms;
    use crate::query::{BooleanQuery, Clause, TermInSetQuery};
    use std::collections::HashMap;
    let dir = lucene_store::FsDirectory::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/mixed_boolean_scoring_index"
    ));
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    assert!(segments.iter().any(|s| s.live_docs.is_some()));
    let owned: Vec<HashMap<String, FieldNorms<'_>>> = reader
        .field_norms("body")
        .into_iter()
        .map(|n| n.into_iter().map(|n| ("body".to_string(), n)).collect())
        .collect();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let mut q = BooleanQuery::new();
    q.must.push(Clause::TermInSet(TermInSetQuery::new(
        "body",
        ["w2", "w4", "w6"],
    )));
    let bits = |h: &[crate::ScoreDoc]| -> Vec<(i32, u32)> {
        h.iter().map(|d| (d.doc_id, d.score.to_bits())).collect()
    };
    // Thresholds the cached walk reaches mid-segment, in a later segment, and
    // never: its popcount count must agree with the first, uncached run.
    for (top_n, threshold) in [
        (10, 20u64),
        (5, 0),
        (7, u64::MAX),
        (3, 77),
        (1, 200),
        (2, 100_000),
    ] {
        let run = || {
            crate::multi_segment::search_boolean_query_multi_segment_maxscore_counting(
                &segments, &q, &norms, top_n, threshold,
            )
            .unwrap()
        };
        let (first, first_total) = run();
        assert_eq!(first.len(), top_n);
        for i in 0..6 {
            let (hits, total) = run();
            assert_eq!(bits(&hits), bits(&first), "run {i}, top {top_n}");
            assert_eq!(
                (total.value, total.relation),
                (first_total.value, first_total.relation),
                "run {i}, top {top_n}"
            );
        }
        if threshold == u64::MAX {
            // Counted exactly: the cached walk visits every live match.
            let exact = crate::multi_segment::search_boolean_query_multi_segment_maxscore_counting(
                &segments,
                &q,
                &norms,
                1,
                u64::MAX,
            )
            .unwrap()
            .1;
            assert_eq!(exact.value, first_total.value);
        }
    }
    for r in reader.segment_readers() {
        assert_eq!(
            r.query_cache().stats().0,
            1,
            "the term set cached in {}",
            r.segment_name
        );
    }
}

/// `StreamedTerms` matches exactly the union of every term it was fed:
/// blended (16 iterators, the rest in the set) and plain (all in the set),
/// sparse and dense, and through the collected fallback when the segment
/// gives it no `maxDoc` -- whichever terms the priority queue keeps.
#[test]
fn streamed_terms_match_the_union_of_their_postings() {
    use crate::directory_reader::DirectoryReader;
    use crate::query::{Clause, PrefixQuery};
    let dir = lucene_store::FsDirectory::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/mixed_boolean_scoring_index"
    ));
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let seg = &opened.as_open_segments()[0];
    let ctx = super::build::LeafContext {
        fields: seg.fields,
        doc_in: seg.doc_in,
        pos_in: seg.pos_in,
        pay_in: seg.pay_in,
        live_docs: None,
        points: None,
        norms: None,
        global: None,
        max_doc: seg.max_doc,
        cache: None,
        reader: None,
        similarity: None,
    };
    let no_max_doc = super::build::LeafContext {
        max_doc: None,
        ..ctx
    };
    let field = ctx.fields.field("body").unwrap();
    let mut streamed_some = false;
    for prefix in ["w", "w1", "w2", "w12"] {
        let clause = Clause::Prefix(PrefixQuery::new("body", prefix));
        let terms = crate::expanded_terms(ctx.fields, &clause)
            .unwrap()
            .unwrap()
            .1;
        // Reversed as well: the queue then sees its highest `docFreq`s last
        // and has to evict.
        let orders: [Vec<_>; 2] = [terms.clone(), terms.iter().rev().cloned().collect()];
        let mut want: Vec<i32> = Vec::new();
        for (_, t) in &terms {
            let mut c = field
                .lazy_postings_for(
                    t,
                    ctx.doc_in.unwrap(),
                    lucene_codecs::postings::PostingsFlags::DocsOnly,
                )
                .unwrap();
            let mut d = c.next_doc().unwrap();
            while d != NO_MORE_DOCS {
                want.push(d);
                d = c.next_doc().unwrap();
            }
        }
        want.sort_unstable();
        want.dedup();
        streamed_some |= terms.len() > super::multi_term::BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD;
        // A `maxDoc` large enough that the union stays a sparse list, and one
        // whose `maxDoc >> 7` falls inside the terms' summed `docFreq`, so the
        // list upgrades to the bit set part way through.
        let sum_df: i32 = terms.iter().map(|(_, t)| t.stats.doc_freq).sum();
        let sparse = super::build::LeafContext {
            max_doc: Some(1 << 24),
            ..ctx
        };
        let upgrading = super::build::LeafContext {
            max_doc: Some((128 * (sum_df / 2)).max(ctx.max_doc.unwrap())),
            ..ctx
        };
        for c in [&ctx, &no_max_doc, &sparse, &upgrading] {
            for blended in [true, false] {
                for order in &orders {
                    let mut stream = super::multi_term::StreamedTerms::new(c, "body", blended);
                    for (term, seeked) in order.clone() {
                        stream.push(term, seeked).unwrap();
                    }
                    let mut got = Vec::new();
                    if let Some(mut s) = stream.finish(c, "body", 1.0, Mode::Complete).unwrap() {
                        let mut d = s.next_doc().unwrap();
                        while d != NO_MORE_DOCS {
                            assert_eq!(s.score().unwrap(), 1.0, "constant-scored");
                            got.push(d);
                            d = s.next_doc().unwrap();
                        }
                    }
                    assert_eq!(got, want, "prefix {prefix} blended {blended}");
                }
            }
        }
    }
    assert!(
        streamed_some,
        "some prefix must expand past the boolean rewrite"
    );

    let clause = Clause::Prefix(PrefixQuery::new("body", "w"));
    let terms = crate::expanded_terms(ctx.fields, &clause)
        .unwrap()
        .unwrap()
        .1;
    let first_doc = |c: &super::build::LeafContext<'_>, field: &str, blended: bool, n: usize| {
        let mut stream = super::multi_term::StreamedTerms::new(c, field, blended);
        for (term, seeked) in terms.iter().take(n).cloned() {
            stream.push(term, seeked).unwrap();
        }
        stream
            .finish(c, field, 1.0, Mode::Complete)
            .unwrap()
            .map(|mut s| s.next_doc().unwrap())
    };
    // Every document past a `maxDoc` of 1: the plain union holds document 0
    // at most, and nothing if no term has it.
    let tiny = super::build::LeafContext {
        max_doc: Some(1),
        ..ctx
    };
    assert_eq!(
        first_doc(&tiny, "body", false, terms.len()),
        first_doc(&ctx, "body", false, terms.len()).filter(|&d| d == 0)
    );
    // No terms, a field this segment lacks, no `.doc` input: nothing.
    assert!(first_doc(&ctx, "body", true, 0).is_none());
    assert!(first_doc(&ctx, "nosuchfield", true, 1).is_none());
    let no_docs = super::build::LeafContext {
        doc_in: None,
        ..ctx
    };
    assert!(first_doc(&no_docs, "body", true, 1).is_none());
    // A clause on a field the segment lacks matches nothing in it.
    for absent in [
        Clause::Regexp(crate::query::RegexpQuery::new("nosuchfield", "w.*")),
        Clause::TermInSet(crate::query::TermInSetQuery::new("nosuchfield", ["w1"])),
    ] {
        assert!(matches!(
            super::multi_term::multi_term(&ctx, &absent, 1.0, Mode::Complete).unwrap(),
            Some(None)
        ));
    }
    // `advance` over a blended union steps its set as well as its terms, and
    // lands where stepping one document at a time does.
    let union = || {
        let mut stream = super::multi_term::StreamedTerms::new(&ctx, "body", true);
        for (term, seeked) in terms.iter().cloned() {
            stream.push(term, seeked).unwrap();
        }
        stream
            .finish(&ctx, "body", 1.0, Mode::Complete)
            .unwrap()
            .unwrap()
    };
    let mut all = Vec::new();
    let mut s = union();
    let mut d = s.next_doc().unwrap();
    while d != NO_MORE_DOCS {
        all.push(d);
        d = s.next_doc().unwrap();
    }
    let mut s = union();
    let mut d = s.next_doc().unwrap();
    while d != NO_MORE_DOCS {
        let target = d + 3;
        d = s.advance(target).unwrap();
        let want = all
            .iter()
            .copied()
            .find(|&x| x >= target)
            .unwrap_or(NO_MORE_DOCS);
        assert_eq!(d, want, "advance({target})");
    }
}

/// A cursor reset onto term after term (`TermsEnum.postings(reuse, ...)`)
/// reads every term's postings exactly as a fresh cursor does, in either
/// order -- no block, bit set or position left over from the term before --
/// and refuses a pulsed term, which has no `.doc` bytes to reset onto.
#[test]
fn a_reused_postings_cursor_reads_each_term_like_a_fresh_one() {
    use crate::directory_reader::DirectoryReader;
    use crate::query::{Clause, PrefixQuery};
    use lucene_codecs::postings::PostingsFlags;
    let dir = lucene_store::FsDirectory::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/mixed_boolean_scoring_index"
    ));
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let seg = &opened.as_open_segments()[0];
    let doc_in = seg.doc_in.unwrap();
    let field = seg.fields.field("body").unwrap();
    let clause = Clause::Prefix(PrefixQuery::new("body", "w"));
    let terms = crate::expanded_terms(seg.fields, &clause)
        .unwrap()
        .unwrap()
        .1;
    let drain = |c: &mut lucene_codecs::postings::LazyDocsCursor<'_>| {
        let mut docs = Vec::new();
        let mut d = c.next_doc().unwrap();
        while d != NO_MORE_DOCS {
            docs.push(d);
            d = c.next_doc().unwrap();
        }
        docs
    };
    let mut checked = 0;
    for order in [false, true] {
        let mut reuse = None;
        let list: Vec<_> = if order {
            terms.iter().rev().collect()
        } else {
            terms.iter().collect()
        };
        for (_, t) in list {
            if t.stats.doc_freq <= 1 {
                continue;
            }
            let mut fresh = field
                .lazy_postings_for(t, doc_in, PostingsFlags::DocsOnly)
                .unwrap();
            let want = drain(&mut fresh);
            let reused = field
                .reuse_postings_for(t, doc_in, PostingsFlags::DocsOnly, &mut reuse)
                .unwrap();
            assert_eq!(drain(reused), want);
            checked += 1;
        }
    }
    assert!(
        checked > 2,
        "the prefix must reach several multi-document terms"
    );
    if let Some((_, pulsed)) = terms.iter().find(|(_, t)| t.stats.doc_freq == 1) {
        let mut empty = None;
        assert!(field
            .reuse_postings_for(pulsed, doc_in, PostingsFlags::DocsOnly, &mut empty)
            .is_err());
    }
}

/// Every term of `fields` in the index at `path` (relative to the fixtures
/// directory), checked against [`FieldTerms::tail_only_first_doc`]: a term
/// with `1 < docFreq < block` gets the first document its cursor lands on,
/// every other term no answer. Returns how many were answered, how many of
/// those had fewer than four documents (plain vints rather than a group
/// varint), how many were in `docs_only`, and how many terms filled at
/// least one full block (and so had to get no answer).
///
/// [`FieldTerms::tail_only_first_doc`]: lucene_codecs::blocktree::FieldTerms::tail_only_first_doc
fn check_tail_only_first_doc(
    path: &str,
    fields: &[&str],
    docs_only: &str,
    block: i32,
) -> (usize, usize, usize, usize) {
    use crate::directory_reader::DirectoryReader;
    use lucene_codecs::postings::PostingsFlags;
    let dir = lucene_store::FsDirectory::open(format!(
        "{}/../../fixtures/data/{path}",
        env!("CARGO_MANIFEST_DIR")
    ));
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let (mut answered, mut few, mut only, mut full) = (0, 0, 0, 0);
    for seg in opened.as_open_segments() {
        let doc_in = seg.doc_in.unwrap();
        for &name in fields {
            let Some(field) = seg.fields.field(name) else {
                continue;
            };
            let mut it = field.iter();
            while it.try_next_term().unwrap().is_some() {
                let t = it.try_seeked_term().unwrap().unwrap();
                let got = field.tail_only_first_doc(&t, doc_in).unwrap();
                if t.stats.doc_freq <= 1 || t.stats.doc_freq >= block {
                    assert_eq!(got, None, "{path} {name} df={}", t.stats.doc_freq);
                    full += usize::from(t.stats.doc_freq >= block);
                    continue;
                }
                let mut c = field
                    .lazy_postings_for(&t, doc_in, PostingsFlags::DocsOnly)
                    .unwrap();
                assert_eq!(got, Some(c.next_doc().unwrap()), "{path} {name}");
                answered += 1;
                few += usize::from(t.stats.doc_freq < 4);
                only += usize::from(name == docs_only);
            }
        }
    }
    (answered, few, only, full)
}

/// A tail-only term's first document, read from its first delta alone, is
/// the one its cursor lands on first -- for a field with frequencies (the
/// delta's low bit is the freq flag) and a docs-only one, with fewer than four
/// documents (plain vints) and more (group varints) -- and every other term
/// gets no answer.
#[test]
fn a_tail_only_terms_first_doc_matches_its_cursor() {
    let (answered, few, docs_only, _) =
        check_tail_only_first_doc("m7_queries_index", &["body", "title", "tag"], "tag", 256);
    assert!(
        answered > 0 && few > 0 && docs_only > 0,
        "{answered} {few} {docs_only}"
    );
}

/// The same over indices the 9.12-10.3 postings formats wrote (`Lucene912`,
/// `Lucene101`, blocks of 128 documents), each by that release's own jars: a
/// term of 128 to 255 documents fills a block there, so it must get no
/// answer -- the walk would otherwise skip it on a wrong first document.
#[test]
fn a_tail_only_terms_first_doc_matches_its_cursor_on_older_formats() {
    let (mut answered, mut few, mut docs_only, mut full) = (0, 0, 0, 0);
    for version in ["9.12.2", "10.0.0", "10.2.2"] {
        for root in ["bwc", "bwc-big"] {
            let path = format!("{root}/{version}");
            if !std::path::Path::new(&format!(
                "{}/../../fixtures/data/{path}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .exists()
            {
                continue;
            }
            let (a, f, d, b) =
                check_tail_only_first_doc(&path, &["body", "docs", "freqs", "f", "d"], "docs", 128);
            answered += a;
            few += f;
            docs_only += d;
            full += b;
        }
    }
    assert!(
        answered > 0 && few > 0 && docs_only > 0 && full > 0,
        "{answered} {few} {docs_only} {full}"
    );
}

/// A sink that breaks ends [`super::extended::visit_terms`]'s walk at that
/// term, on the iterator path (a prefix) and the term-range loop alike.
#[test]
fn visit_terms_stops_where_the_sink_breaks() {
    use crate::directory_reader::DirectoryReader;
    use crate::extended_query::{MultiTermSource, TermRangeQuery};
    use crate::query::PrefixQuery;
    use std::ops::ControlFlow;
    let dir = lucene_store::FsDirectory::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/m7_queries_index"
    ));
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let seg = &opened.as_open_segments()[0];
    for source in [
        MultiTermSource::Prefix(PrefixQuery::new("body", Vec::new())),
        MultiTermSource::TermRange(TermRangeQuery::new("body", None, None, true, true)),
    ] {
        let mut all = 0usize;
        super::extended::visit_terms(seg.fields, &source, None, &mut |_, _| {
            all += 1;
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
        assert!(all > 3, "{all}");
        let mut seen = 0usize;
        super::extended::visit_terms(seg.fields, &source, None, &mut |_, _| {
            seen += 1;
            Ok(if seen == 3 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            })
        })
        .unwrap();
        assert_eq!(seen, 3);
    }
}

/// The multi-term builder's edges, and its union scorer driven directly.
#[test]
fn multi_term_edges_and_the_term_union() {
    use crate::directory_reader::DirectoryReader;
    use crate::query::{Clause, PrefixQuery, TermQuery};
    let dir = lucene_store::FsDirectory::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/mixed_boolean_scoring_index"
    ));
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let seg = &opened.as_open_segments()[0];
    let ctx = super::build::LeafContext {
        fields: seg.fields,
        doc_in: seg.doc_in,
        pos_in: seg.pos_in,
        pay_in: seg.pay_in,
        live_docs: None,
        points: None,
        norms: None,
        global: None,
        max_doc: seg.max_doc,
        cache: None,
        reader: None,
        similarity: None,
    };
    fn mt<'a>(c: &Clause, ctx: &super::build::LeafContext<'a>) -> Option<Option<BoxScorer<'a>>> {
        super::multi_term::multi_term(ctx, c, 1.0, Mode::Complete).unwrap()
    }
    let term = Clause::Term(TermQuery::new("body", "w0"));
    assert!(mt(&term, &ctx).is_none(), "not the multi-term family");
    let prefix = Clause::Prefix(PrefixQuery::new("body", "w1"));
    let no_docs = super::build::LeafContext {
        doc_in: None,
        ..ctx
    };
    assert!(mt(&prefix, &no_docs).is_none(), "no .doc input");
    let absent = Clause::Prefix(PrefixQuery::new("nosuchfield", "w"));
    assert!(matches!(mt(&absent, &ctx), Some(None)), "absent field");
    let none = Clause::Prefix(PrefixQuery::new("body", "zzz"));
    assert!(matches!(mt(&none, &ctx), Some(None)), "no matching term");
    // The union itself: advance, next, a 0 score and bound.
    let Some(Some(mut s)) = mt(&prefix, &ctx) else {
        panic!("w1 has terms")
    };
    let first = s.next_doc().unwrap();
    let later = s.advance(first + 100).unwrap();
    assert!(later >= first + 100);
    assert_eq!(s.score().unwrap(), 1.0, "constant-scored");
    let terms = crate::expanded_terms(ctx.fields, &prefix)
        .unwrap()
        .unwrap()
        .1;
    let field = ctx.fields.field("body").unwrap();
    let legs = terms
        .iter()
        .map(|(_, t)| {
            let c = field
                .lazy_postings_for(
                    t,
                    ctx.doc_in.unwrap(),
                    lucene_codecs::postings::PostingsFlags::DocsOnly,
                )
                .unwrap();
            crate::bulk_scorer::TermLeg::filter(c, t.stats.doc_freq as i64)
        })
        .collect();
    let mut u = super::multi_term::TermUnion::new(legs, None);
    assert_eq!(u.next_doc().unwrap(), first);
    assert_eq!(u.advance(first + 100).unwrap(), later);
    assert_eq!(u.score().unwrap(), 0.0);
    assert_eq!(u.max_score(NO_MORE_DOCS).unwrap(), 0.0);
    assert!(u.cost() > 0);
}

/// A filter that stops answering by membership mid-way falls back to the
/// leapfrog, and a leg conjunction takes a threshold on its one scoring leg.
#[test]
fn conjunction_membership_fallback_and_leg_thresholds() {
    use super::conjunction::ConjunctionScorer;
    use std::cell::Cell;
    struct Flaky {
        inner: DocList,
        probes: Cell<u32>,
    }
    impl Scorer for Flaky {
        fn doc_id(&self) -> i32 {
            self.inner.doc_id()
        }
        fn next_doc(&mut self) -> crate::Result<i32> {
            self.inner.next_doc()
        }
        fn advance(&mut self, t: i32) -> crate::Result<i32> {
            self.inner.advance(t)
        }
        fn cost(&self) -> i64 {
            100
        }
        fn score(&mut self) -> crate::Result<f32> {
            Ok(0.0)
        }
        fn max_score(&mut self, _: i32) -> crate::Result<f32> {
            Ok(0.0)
        }
        // Random access for the constructor's probe, then never again.
        fn contains(&self, _doc: i32) -> Option<bool> {
            self.probes.set(self.probes.get() + 1);
            (self.probes.get() == 1).then_some(true)
        }
    }
    let filter = Flaky {
        inner: DocList::new(vec![2, 4, 6, 8], Vec::new()),
        probes: Cell::new(0),
    };
    let lead = DocList::new(vec![1, 2, 3, 6, 7, 9], vec![1.0; 6]);
    let mut c = ConjunctionScorer::new(vec![Box::new(filter)], vec![Box::new(lead)]);
    let mut got = Vec::new();
    let mut d = c.next_doc().unwrap();
    while d != NO_MORE_DOCS {
        got.push(d);
        d = c.next_doc().unwrap();
    }
    assert_eq!(got, [2, 6]);

    // `LegConjunctionScorer`: one scoring leg takes the threshold.
    use crate::directory_reader::DirectoryReader;
    use crate::query::{BooleanQuery, Clause, TermQuery};
    let dir = lucene_store::FsDirectory::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/mixed_boolean_scoring_index"
    ));
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let seg = &opened.as_open_segments()[0];
    let ctx = super::build::LeafContext {
        fields: seg.fields,
        doc_in: seg.doc_in,
        pos_in: seg.pos_in,
        pay_in: seg.pay_in,
        live_docs: None,
        points: None,
        norms: None,
        global: None,
        max_doc: seg.max_doc,
        cache: None,
        reader: None,
        similarity: None,
    };
    let mut b = BooleanQuery::new();
    b.must.push(Clause::Term(TermQuery::new("body", "w0")));
    b.filter.push(Clause::Term(TermQuery::new("body", "w1")));
    let clause = Clause::Boolean(Box::new(b));
    let mut s = super::build::build(&ctx, &clause, 1.0, Mode::TopScores, false)
        .unwrap()
        .unwrap();
    let first = s.next_doc().unwrap();
    s.set_min_competitive_score(f32::MAX).unwrap();
    assert!(
        s.next_doc().unwrap() > first,
        "nothing competes past the threshold"
    );
}

/// A phrase on a segment without norms scores every document at the
/// unnormed length, like a term does.
#[test]
fn a_phrase_without_norms_scores_unnormed() {
    use crate::directory_reader::DirectoryReader;
    use crate::query::{Clause, PhraseQuery};
    let dir = lucene_store::FsDirectory::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/mixed_boolean_scoring_index"
    ));
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let seg = &opened.as_open_segments()[0];
    let ctx = super::build::LeafContext {
        fields: seg.fields,
        doc_in: seg.doc_in,
        pos_in: seg.pos_in,
        pay_in: seg.pay_in,
        live_docs: None,
        points: None,
        norms: None,
        global: None,
        max_doc: seg.max_doc,
        cache: None,
        reader: None,
        similarity: None,
    };
    let phrase = Clause::Phrase(PhraseQuery::new("body", ["w0", "w1"]));
    let s = super::build::build(&ctx, &phrase, 1.0, Mode::Complete, true)
        .unwrap()
        .unwrap();
    let mut all = All(Vec::new());
    score_segment(&mut Bulk::scorer(s), Mode::Complete, None, &mut all).unwrap();
    assert!(!all.0.is_empty());
    assert!(all
        .0
        .iter()
        .all(|&(_, score)| score > 0.0 && score.is_finite()));
}

/// A match-all without a maxDoc of its own (as the JVM decodes it) on a
/// segment that supplies none is an error, not a walk to `i32::MAX`.
#[test]
fn a_match_all_needs_some_max_doc() {
    use super::build::LeafContext;
    use crate::query::{Clause, MatchAllDocsQuery};
    let fields = lucene_codecs::blocktree::BlockTreeFields::default();
    let ctx = LeafContext {
        fields: &fields,
        doc_in: None,
        pos_in: None,
        pay_in: None,
        live_docs: None,
        points: None,
        norms: None,
        global: None,
        max_doc: None,
        cache: None,
        reader: None,
        similarity: None,
    };
    let unknown = Clause::MatchAllDocs(MatchAllDocsQuery::new(i32::MAX));
    assert!(matches!(
        super::build::build(&ctx, &unknown, 1.0, Mode::TopScores, true),
        Err(crate::Error::MatchAllWithoutMaxDoc)
    ));
    let known = Clause::MatchAllDocs(MatchAllDocsQuery::new(3));
    let mut all = All(Vec::new());
    let s = super::build::build(&ctx, &known, 1.0, Mode::Complete, true)
        .unwrap()
        .unwrap();
    score_segment(&mut Bulk::scorer(s), Mode::Complete, None, &mut all).unwrap();
    assert_eq!(all.0.len(), 3);
}

fn node_is_bool(node: &Node) -> bool {
    matches!(node, Node::Bool { .. })
}

#[test]
fn deleted_documents_are_never_collected() {
    let mut rng = Rng(7);
    let mut live = lucene_util::fixed_bit_set::FixedBitSet::new(MAX_DOC as usize);
    for d in 0..MAX_DOC {
        if d % 3 != 0 {
            // FBS: `d < MAX_DOC`, the bitset's size.
            live.set(d as usize);
        }
    }
    for _ in 0..200 {
        let node = random_node(&mut rng, 2);
        let mut all = All(Vec::new());
        if let Some(s) = build(&node, Mode::Complete, true) {
            score_segment(&mut Bulk::scorer(s), Mode::Complete, Some(&live), &mut all).unwrap();
        }
        let want: Vec<(i32, f32)> = expected(&node, true)
            .into_iter()
            .filter(|h| h.0 % 3 != 0)
            .collect();
        assert_eq!(all.0, want);
    }
}

/// A shared collector that already holds a threshold from an earlier segment
/// hands it to the next segment's scorer before its first document.
#[test]
fn a_threshold_from_an_earlier_segment_prunes_from_the_first_document() {
    let leaf = Leaf::new(
        (0..MAX_DOC).map(|d| (d, 1.0)).collect(),
        (0..MAX_DOC).collect(),
        false,
    );
    let node = Node::Const(1.0, Box::new(Node::Leaf(leaf)));
    let mut top = TopDocsCollector::with_total_hits_threshold(2, 2);
    for d in 0..3 {
        top.collect(10_000 + d, 5.0);
    }
    let s = build(&node, Mode::TopScores, true).unwrap();
    score_segment(&mut Bulk::scorer(s), Mode::TopScores, None, &mut top).unwrap();
    assert_eq!(
        top.total_hits().value,
        3,
        "a constant 1.0 cannot beat 5.0: nothing in this segment is visited"
    );
}

#[test]
fn doc_list_iterates_advances_and_scores() {
    let mut l = DocList::new(vec![2, 5, 9], vec![0.5, 1.5, 1.0]);
    assert_eq!(l.doc_id(), -1);
    assert_eq!(l.cost(), 3);
    assert_eq!(l.max_score(NO_MORE_DOCS).unwrap(), 1.5);
    assert_eq!(l.next_doc().unwrap(), 2);
    assert_eq!(l.score().unwrap(), 0.5);
    assert_eq!(l.advance(6).unwrap(), 9);
    assert_eq!(l.score().unwrap(), 1.0);
    assert_eq!(l.advance(10).unwrap(), NO_MORE_DOCS);
    assert_eq!(l.next_doc().unwrap(), NO_MORE_DOCS);
    let mut unscored = DocList::new(vec![1], Vec::new());
    assert_eq!(unscored.next_doc().unwrap(), 1);
    assert_eq!(unscored.score().unwrap(), 0.0);
}

#[test]
fn all_docs_is_the_whole_range() {
    let mut a = AllDocs::new(3);
    assert_eq!(a.cost(), 3);
    assert_eq!(a.next_doc().unwrap(), 0);
    assert_eq!(a.advance(2).unwrap(), 2);
    assert_eq!(a.next_doc().unwrap(), NO_MORE_DOCS);
    assert_eq!(a.score().unwrap(), 0.0);
    assert_eq!(a.max_score(0).unwrap(), 0.0);
}

#[test]
fn wand_scaling_matches_lucene() {
    use super::wand::{scale_max_score, scaling_factor};
    // WANDScorer.scalingFactor: FLOAT_MANTISSA_BITS - 1 - getExponent.
    assert_eq!(scaling_factor(1.0), 23);
    assert_eq!(scaling_factor(3.0), 22);
    assert_eq!(scaling_factor(0.25), 25);
    assert_eq!(scaling_factor(0.0), scaling_factor(f32::from_bits(1)) + 1);
    assert_eq!(scaling_factor(f32::INFINITY), scaling_factor(f32::MAX) - 1);
    // Rounds up, and saturates at 2^24 - 1.
    assert_eq!(scale_max_score(1.0, 23), 1 << 23);
    assert_eq!(scale_max_score(1.5, 0), 2);
    assert_eq!(scale_max_score(4.0, 23), (1 << 24) - 1);
}

// ---- the real postings: which bulk scorer each fixture query reaches -------

mod fixture {
    use std::collections::{BTreeSet, HashMap};

    use crate::bulk_scorer::test_only_req_opt_paths;
    use crate::directory_reader::DirectoryReader;
    use crate::exec::{bulk_boolean, LeafContext, Mode};
    use crate::field_norms::FieldNorms;
    use crate::query::{BoostQuery, ConstantScoreQuery, DisjunctionMaxQuery};
    use crate::{BooleanQuery, Clause, TermQuery};

    fn dir() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/mixed_boolean_scoring_index"
        )
        .to_string()
    }

    /// `GenMixedBooleanScoring`'s grammar, as `tests/mixed_boolean_fixtures.rs`
    /// parses it.
    pub(super) fn parse(text: &str) -> BooleanQuery {
        let toks: Vec<String> = text
            .replace('(', " ( ")
            .replace(')', " ) ")
            .split_whitespace()
            .map(str::to_string)
            .collect();
        let mut at = 0;
        let clause = node(&toks, &mut at);
        match clause {
            Clause::Boolean(b) => *b,
            other => BooleanQuery {
                must: vec![other],
                ..Default::default()
            },
        }
    }

    fn node(t: &[String], at: &mut usize) -> Clause {
        {
            use crate::query::{PrefixQuery, RegexpQuery, TermInSetQuery, WildcardQuery};
            let op = t[*at + 1].as_str();
            if op == "r" {
                let min: i64 = t[*at + 2].parse().unwrap();
                let max: i64 = t[*at + 3].parse().unwrap();
                *at += 5;
                return Clause::PointsRange(crate::query::PointsRangeQuery::new("n", min, max));
            }
            if matches!(op, "pre" | "wc" | "re" | "ts") {
                *at += 2;
                let mut words = Vec::new();
                while t[*at] != ")" {
                    words.push(t[*at].clone());
                    *at += 1;
                }
                *at += 1;
                let w = words[0].clone();
                return match op {
                    "pre" => Clause::Prefix(PrefixQuery::new("body", w.into_bytes())),
                    "wc" => Clause::Wildcard(WildcardQuery::new("body", w.into_bytes())),
                    "re" => Clause::Regexp(RegexpQuery::new("body", w)),
                    _ => Clause::TermInSet(TermInSetQuery::new(
                        "body",
                        words.iter().map(|w| w.clone().into_bytes()),
                    )),
                };
            }
        }
        if t[*at + 1] == "p" || t[*at + 1] == "ps" {
            let slop: u32 = if t[*at + 1] == "ps" {
                *at += 1;
                t[*at + 1].parse().unwrap()
            } else {
                0
            };
            *at += 2;
            let mut words = Vec::new();
            while t[*at] != ")" {
                words.push(t[*at].clone());
                *at += 1;
            }
            *at += 1;
            return Clause::Phrase(crate::query::PhraseQuery::new("body", words).with_slop(slop));
        }
        let mut next = || {
            *at += 1;
            t[*at - 1].clone()
        };
        assert_eq!(next(), "(");
        let op = next();
        let q = match op.as_str() {
            "t" => Clause::Term(TermQuery::new("body", next().into_bytes())),
            "boost" => {
                let f: f32 = next().parse().unwrap();
                BoostQuery::new(node(t, at), f).into()
            }
            "const" => ConstantScoreQuery::new(node(t, at), 1.0).into(),
            "dismax" => {
                let tie: f32 = next().parse().unwrap();
                let mut ds = Vec::new();
                while t[*at] == "(" {
                    ds.push(node(t, at));
                }
                DisjunctionMaxQuery::new(ds, tie).into()
            }
            "b" => {
                let mut b = BooleanQuery::new();
                b.minimum_should_match = next().parse().unwrap();
                while t[*at] == "(" {
                    *at += 1;
                    let occur = t[*at].clone();
                    *at += 1;
                    let c = node(t, at);
                    match occur.as_str() {
                        "+" => b.must.push(c),
                        "#" => b.filter.push(c),
                        "?" => b.should.push(c),
                        _ => b.must_not.push(c),
                    }
                    assert_eq!(t[*at], ")");
                    *at += 1;
                }
                Clause::Boolean(Box::new(b))
            }
            other => panic!("unknown op {other}"),
        };
        assert_eq!(t[*at], ")");
        *at += 1;
        q
    }

    fn manifest() -> HashMap<String, String> {
        std::fs::read_to_string(format!("{}/manifest.properties", dir()))
            .expect("fixture manifest")
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    /// Every bulk scorer the dispatch can choose, and every `ReqOptBulk` path,
    /// is reached by some fixture query -- so the Lucene differential test in
    /// `tests/mixed_boolean_fixtures.rs` covers each of them, not only the
    /// ones a query mix happens to reach. Also runs every query in
    /// `COMPLETE_NO_SCORES` and checks its count against Lucene's.
    #[test]
    fn the_fixture_queries_reach_every_bulk_scorer_and_count_like_lucene() {
        let m = manifest();
        let reader = DirectoryReader::open(&lucene_store::FsDirectory::open(dir())).unwrap();
        let mut opened = reader.open_segments().unwrap();
        opened.open_points().unwrap();
        let segments = opened.as_open_segments();
        let owned: Vec<HashMap<String, FieldNorms<'_>>> = reader
            .field_norms("body")
            .into_iter()
            .map(|n| n.into_iter().map(|n| ("body".to_string(), n)).collect())
            .collect();
        let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
        let count: usize = m["query_count"].parse().unwrap();
        let mut kinds = BTreeSet::new();
        test_only_req_opt_paths::take();
        for i in 0..count {
            let q = parse(&m[&format!("query.{i}")]);
            let global = crate::multi_segment::global_boolean_stats(&segments, &q).unwrap();
            // The pruned search, as the Lucene differential runs it.
            crate::multi_segment::search_boolean_query_multi_segment_maxscore_counting(
                &segments, &q, &norms, 10, 100,
            )
            .unwrap();
            let mut total = 0u64;
            for (s, seg) in segments.iter().enumerate() {
                let ctx = LeafContext {
                    fields: seg.fields,
                    doc_in: seg.doc_in,
                    pos_in: seg.pos_in,
                    pay_in: seg.pay_in,
                    live_docs: seg.live_docs,
                    points: seg.points,
                    norms: norms[s],
                    global: Some(&global),
                    max_doc: None,
                    cache: None,
                    reader: None,
                    similarity: None,
                };
                for mode in [Mode::TopScores, Mode::Complete] {
                    if let Some(b) = bulk_boolean(&ctx, &q, 1.0, mode).unwrap() {
                        kinds.insert(b.kind());
                    }
                }
                if let Some(mut b) = bulk_boolean(&ctx, &q, 1.0, Mode::NoScores).unwrap() {
                    kinds.insert(b.kind());
                    let mut c = Count(0);
                    crate::exec::score_segment(&mut b, Mode::NoScores, seg.live_docs, &mut c)
                        .unwrap();
                    total += c.0;
                }
            }
            assert_eq!(
                total.to_string(),
                m[&format!("query.{i}.total")],
                "{}: COMPLETE_NO_SCORES count",
                m[&format!("query.{i}")]
            );
        }
        let all: BTreeSet<&str> = [
            "scorer",
            "term",
            "conjunction",
            "filtered_term",
            "scorer_conjunction",
            "disjunction",
            "filtered_disjunction",
            "min_should_match",
            "scorer_disjunction",
            "req_opt",
            // A one-phase nested boolean beside one optional term, scored
            // exhaustively (`COMPLETE`: no threshold to skip on).
            "req_scorer_opt",
            "req_excl",
            "dismax",
            "union",
        ]
        .into_iter()
        .collect();
        assert_eq!(kinds, all, "bulk scorers the fixture reaches");
        let paths = test_only_req_opt_paths::take();
        // A window per lead block: never the 64K-id march through the whole
        // `int` range once a skipped window leaves the lead in its last block.
        let windows = test_only_req_opt_paths::take_windows();
        assert!(windows < 30_000, "ReqOptBulk windows: {windows}");
        assert!(
            paths.iter().all(|&n| n > 0),
            "every ReqOptBulk path must run (required-led, optional-led single, \
             optional-led window, filtered MaxScore): {paths:?}"
        );
    }

    /// `FilterConjunction` taking over legs that are out of step -- a
    /// non-lead leg already *past* the lead's document, which is how the
    /// batch path can leave them -- must still report only documents every
    /// leg matches. (R1 review: it used to treat a leg ahead as a match.)
    #[test]
    fn a_filter_conjunction_taking_over_out_of_step_legs_reports_only_matches() {
        use crate::bulk_scorer::{BulkLeg, FilterConjunction, TermLeg};
        use crate::exec::build::{term_leg, TermForm};
        use crate::exec::Scorer as _;

        let reader = DirectoryReader::open(&lucene_store::FsDirectory::open(dir())).unwrap();
        let opened = reader.open_segments().unwrap();
        let seg = &opened.as_open_segments()[0];
        let ctx = LeafContext {
            fields: seg.fields,
            doc_in: seg.doc_in,
            pos_in: None,
            pay_in: None,
            live_docs: None,
            points: None,
            norms: None,
            global: None,
            max_doc: None,
            cache: None,
            reader: None,
            similarity: None,
        };
        let leg = |t: &str| -> TermLeg<'_> {
            match term_leg(
                &ctx,
                &Clause::Term(TermQuery::new("body", t)),
                1.0,
                Mode::NoScores,
            )
            .unwrap()
            {
                TermForm::Leg(l) => *l,
                _ => panic!("{t} is a term leg"),
            }
        };
        let docs_of = |t: &str| -> Vec<i32> {
            let mut l = leg(t);
            let mut v = Vec::new();
            while BulkLeg::next_doc(&mut l).unwrap() != super::NO_MORE_DOCS {
                v.push(BulkLeg::doc_id(&l));
            }
            v
        };
        let (a, b) = (docs_of("w8"), docs_of("w9"));
        let both: Vec<i32> = a
            .iter()
            .copied()
            .filter(|d| b.binary_search(d).is_ok())
            .collect();
        assert!(both.len() > 5, "the fixture's w8 and w9 overlap");
        let mut checked = 0;
        // The batch path's state: the lead has moved on from `prev` (the last
        // document of a batch) to `lead_at`, and the other leg was advanced
        // to `prev` -- landing on its first document at or after it, which
        // can be past `lead_at`. Every such start where it is past is tried.
        for w in a.windows(2).take(2000) {
            let (prev, lead_at) = (w[0], w[1]);
            let Some(&landed) = b.iter().find(|&&d| d >= prev) else {
                continue;
            };
            if landed <= lead_at {
                continue;
            }
            let mut legs = vec![leg("w8"), leg("w9")];
            BulkLeg::advance(&mut legs[0], lead_at).unwrap();
            BulkLeg::advance(&mut legs[1], prev).unwrap();
            let mut f = FilterConjunction { legs: &mut legs };
            let got = f.align(lead_at).unwrap();
            let want = both
                .iter()
                .copied()
                .find(|&d| d >= lead_at)
                .unwrap_or(super::NO_MORE_DOCS);
            assert_eq!(got, want, "lead at {lead_at}, other leg ahead at {landed}");
            let next = f.next_doc().unwrap();
            let want_next = both
                .iter()
                .copied()
                .find(|&d| d > want)
                .unwrap_or(super::NO_MORE_DOCS);
            assert_eq!(next, want_next, "after {want}");
            checked += 1;
        }
        assert!(checked > 50, "enough out-of-step starts: {checked}");
    }

    /// A scored dismax of terms (`TermDisMaxScorer`) answers every question
    /// the dismax `DisjunctionScorer` over the same terms answers: the
    /// documents a walk of `next_doc`s and `advance`s lands on, their
    /// scores and run ends, the block bounds (`advance_shallow`,
    /// `max_score`), with a threshold handed down part way (which a
    /// tie-breaker of 0 passes to the terms), and a batch at a time the
    /// documents and scores the walk gives -- with ties of 0 and not, and a
    /// term the segment lacks.
    #[test]
    fn a_term_dismax_answers_what_the_dismax_disjunction_answers() {
        use super::super::disjunction::{Combine, DisjunctionScorer};
        use super::super::{BoxScorer, NO_MORE_DOCS};
        use crate::query::DisjunctionMaxQuery;
        let reader = DirectoryReader::open(&lucene_store::FsDirectory::open(dir())).unwrap();
        let opened = reader.open_segments().unwrap();
        let seg = &opened.as_open_segments()[0];
        let owned = reader.field_norms("body");
        let norms: HashMap<String, FieldNorms<'_>> = owned
            .into_iter()
            .next()
            .flatten()
            .map(|n| ("body".to_string(), n))
            .into_iter()
            .collect();
        let ctx = LeafContext {
            fields: seg.fields,
            doc_in: seg.doc_in,
            pos_in: None,
            pay_in: None,
            live_docs: seg.live_docs,
            points: None,
            norms: Some(&norms),
            global: None,
            max_doc: None,
            cache: None,
            reader: None,
            similarity: None,
        };
        let mut checked = 0usize;
        for tie in [0.0f32, 0.3] {
            for terms in [
                &["w0", "w1"][..],
                &["w2", "w5", "w7"],
                &["w1", "nosuchterm", "w3"],
            ] {
                let d = DisjunctionMaxQuery {
                    disjuncts: terms
                        .iter()
                        .map(|t| Clause::Term(TermQuery::new("body", *t)))
                        .collect(),
                    tie_breaker: tie,
                };
                let clause = Clause::DisjunctionMax(Box::new(d.clone()));
                let fast = || {
                    super::super::build::build(&ctx, &clause, 1.0, Mode::TopScores, false)
                        .unwrap()
                        .unwrap()
                };
                let slow = || -> BoxScorer<'_> {
                    let subs: Vec<BoxScorer<'_>> = d
                        .disjuncts
                        .iter()
                        .filter_map(|c| {
                            super::super::build::build(&ctx, c, 1.0, Mode::TopScores, false)
                                .unwrap()
                        })
                        .collect();
                    Box::new(
                        DisjunctionScorer::new(subs, Combine::Max(tie), true)
                            .with_block_propagator()
                            .unwrap(),
                    )
                };
                // A walk: steps and jumps, bounds read on the way, and a
                // threshold from half way.
                let (mut a, mut b) = (fast(), slow());
                assert_eq!(a.cost(), b.cost());
                assert!(!a.two_phase());
                let mut step = 0i32;
                loop {
                    let (da, db) = if step % 3 == 2 {
                        let target = a.doc_id().saturating_add(1 + step % 97);
                        (a.advance(target).unwrap(), b.advance(target).unwrap())
                    } else {
                        (a.next_doc().unwrap(), b.next_doc().unwrap())
                    };
                    assert_eq!(da, db, "{terms:?} tie {tie} step {step}");
                    if da == NO_MORE_DOCS {
                        break;
                    }
                    assert_eq!(a.doc_id(), b.doc_id());
                    assert_eq!(
                        a.score().unwrap().to_bits(),
                        b.score().unwrap().to_bits(),
                        "{terms:?} tie {tie} doc {da}"
                    );
                    assert_eq!(a.doc_id_run_end(), b.doc_id_run_end());
                    if step % 11 == 0 {
                        let target = da.saturating_add(step % 300);
                        assert_eq!(
                            a.advance_shallow(target).unwrap(),
                            b.advance_shallow(target).unwrap()
                        );
                        let up_to = target.saturating_add(500);
                        assert_eq!(
                            a.max_score(up_to).unwrap().to_bits(),
                            b.max_score(up_to).unwrap().to_bits()
                        );
                    }
                    if step == 200 {
                        a.set_min_competitive_score(1.0).unwrap();
                        b.set_min_competitive_score(1.0).unwrap();
                    }
                    step += 1;
                    checked += 1;
                }
                // Batches: what the per-document walk collects, live ones only.
                let (mut a, mut b) = (fast(), slow());
                a.next_doc().unwrap();
                let mut batched = Vec::new();
                let mut buf = crate::bulk_scorer::DocScores::default();
                let mut up_to = 700;
                while a.doc_id() != NO_MORE_DOCS {
                    a.next_docs_and_scores(up_to, seg.live_docs, &mut buf)
                        .unwrap();
                    if buf.docs.is_empty() {
                        up_to = up_to.saturating_add(9_001);
                        continue;
                    }
                    batched.extend(
                        buf.docs
                            .iter()
                            .zip(&buf.scores)
                            .map(|(&d, &s)| (d, s.to_bits())),
                    );
                }
                let mut walked = Vec::new();
                while b.next_doc().unwrap() != NO_MORE_DOCS {
                    let doc = b.doc_id();
                    if seg.live_docs.is_none_or(|l| l.get_doc(doc)) {
                        walked.push((doc, b.score().unwrap().to_bits()));
                    }
                }
                assert_eq!(batched, walked, "{terms:?} tie {tie} batched");
                // Past the end: an empty batch.
                a.next_docs_and_scores(NO_MORE_DOCS, None, &mut buf)
                    .unwrap();
                assert!(buf.docs.is_empty());
            }
        }
        assert!(checked > 1000, "{checked}");
    }

    struct Count(u64);

    impl crate::collector::ScoringCollector for Count {
        fn collect(&mut self, _doc: i32, _score: f32) {
            self.0 += 1;
        }
        fn score_mode(&self) -> crate::collector::ScoreMode {
            crate::collector::ScoreMode::CompleteNoScores
        }
    }
}

/// `Mode::of` for each collector score mode, and the `Scorer` trait's
/// defaults -- what a leaf with no block maxima, no two-phase check and no
/// runs longer than one document answers.
#[test]
fn modes_follow_the_collector_and_scorer_defaults_are_conservative() {
    use crate::collector::ScoreMode;

    struct With(ScoreMode);
    impl ScoringCollector for With {
        fn collect(&mut self, _doc: i32, _score: f32) {}
        fn score_mode(&self) -> ScoreMode {
            self.0
        }
    }
    assert_eq!(Mode::of(&With(ScoreMode::CompleteNoScores)), Mode::NoScores);
    assert_eq!(Mode::of(&With(ScoreMode::Complete)), Mode::Complete);
    assert_eq!(Mode::of(&With(ScoreMode::TopScores)), Mode::TopScores);
    assert!(Mode::TopScores.needs_scores() && !Mode::NoScores.needs_scores());

    let mut all = AllDocs::new(4);
    assert!(!all.two_phase());
    assert!(all.matches().unwrap());
    assert_eq!(all.match_cost(), 0.0);
    assert_eq!(all.advance_shallow(0).unwrap(), NO_MORE_DOCS);
    all.set_min_competitive_score(1.0).unwrap();
    assert_eq!(super::exact_advance(&mut all, 2).unwrap(), 2);
    assert_eq!(all.doc_id_run_end(), 4, "match-all runs to maxDoc");
}

#[test]
fn below_stops_a_scorer_at_its_end() {
    let docs = [2, 5, 9, 14, 20];
    let leaf = || {
        Leaf::new(
            docs.iter().map(|&d| (d, 1.0)).collect(),
            docs.to_vec(),
            false,
        )
    };
    let collect = |mut s: super::Below<'_>| {
        let mut out = Vec::new();
        let mut d = s.next_doc().unwrap();
        while d != NO_MORE_DOCS {
            assert_eq!(s.score().unwrap(), 1.0);
            out.push(d);
            d = s.next_doc().unwrap();
        }
        assert_eq!(s.doc_id(), NO_MORE_DOCS);
        // Once past the end it stays there.
        assert_eq!(s.next_doc().unwrap(), NO_MORE_DOCS);
        out
    };
    // Documents below 14 only: 14 itself is past the end.
    assert_eq!(
        collect(super::Below::new(Box::new(Fake::new(leaf())), 14)),
        [2, 5, 9]
    );
    assert_eq!(
        collect(super::Below::new(Box::new(Fake::new(leaf())), 15)),
        [2, 5, 9, 14]
    );
    assert!(collect(super::Below::new(Box::new(Fake::new(leaf())), 0)).is_empty());
    // advance: to a match below the end, then to a target past it.
    let mut s = super::Below::new(Box::new(Fake::new(leaf())), 14);
    assert_eq!(s.advance(6).unwrap(), 9);
    assert_eq!(s.doc_id(), 9);
    assert_eq!(s.advance(14).unwrap(), NO_MORE_DOCS);
    // advance landing past the end.
    let mut s = super::Below::new(Box::new(Fake::new(leaf())), 12);
    assert_eq!(s.advance(10).unwrap(), NO_MORE_DOCS);
    assert_eq!(s.advance(11).unwrap(), NO_MORE_DOCS);
    // The rest delegates.
    let mut s = super::Below::new(Box::new(Fake::new(leaf())), 12);
    assert_eq!(s.cost(), 5);
    assert!(!s.two_phase());
    assert_eq!(s.match_cost(), 0.0);
    assert!(s.doc_id_run_end() <= 12);
    assert_eq!(s.contains(13), None);
    assert_eq!(s.next_doc().unwrap(), 2);
    assert!(s.matches().unwrap());
    assert!(s.advance_shallow(0).unwrap() >= 0);
    assert!(s.max_score(11).unwrap() >= 1.0);
    s.set_min_competitive_score(0.5).unwrap();
}

/// The popcount the cached bit-set walk counts losing hits with: range ends
/// inside and on word boundaries, deleted documents, and the document that
/// reaches the count.
#[test]
fn count_set_bits_counts_live_bits_until_the_need() {
    use super::bulk::count_set_bits;
    use lucene_util::fixed_bit_set::FixedBitSet;
    let mut bits = FixedBitSet::new(200);
    let set: Vec<usize> = (0..200).filter(|d| d % 3 == 0).collect();
    for &d in &set {
        // FBS: `set` holds documents below 200, the set's length.
        bits.set(d);
    }
    let words = bits.words();
    let naive = |from: usize, end: usize, live: Option<&FixedBitSet>| -> Vec<usize> {
        set.iter()
            .copied()
            .filter(|&d| d >= from && d < end && live.is_none_or(|l| l.get(d)))
            .collect()
    };
    let mut live = FixedBitSet::new(200);
    for d in 0..200 {
        if d % 7 != 0 {
            // FBS: `d < 200`, the set's length.
            live.set(d);
        }
    }
    for l in [None, Some(&live)] {
        for (from, end) in [(0, 200), (1, 64), (63, 129), (64, 128), (100, 101), (5, 5)] {
            let all = naive(from, end, l);
            assert_eq!(
                count_set_bits(words, l, from, end, u64::MAX),
                (all.len() as u64, None)
            );
            for need in 1..=all.len() {
                assert_eq!(
                    count_set_bits(words, l, from, end, need as u64),
                    (need as u64, Some(all[need - 1])),
                    "{from}..{end} need {need}"
                );
            }
        }
    }
}

/// A field indexed without frequencies but with norms, against Lucene
/// (`fixtures/src/GenDocsOnlyNorms.java`): a term's top 10 under three
/// total-hits thresholds, through the term path and, boosted, through the
/// scorer tree. Lucene bounds such a term by the impact `(freq 1, norm 1)`, so
/// a full queue whose threshold passes the one-token score ends the scan: the
/// totals show where, and the skip counter that it was a skip, not the end of
/// the postings.
#[test]
fn a_docs_only_field_with_norms_ends_its_scan_as_lucene_does() {
    use crate::directory_reader::DirectoryReader;
    use crate::field_norms::FieldNorms;
    use crate::query::{BooleanQuery, BoostQuery, Clause, TermQuery};
    use std::collections::HashMap;
    let base = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/docs_only_norms_index"
    );
    let manifest: HashMap<String, String> =
        std::fs::read_to_string(format!("{base}/manifest.properties"))
            .expect("run scripts/gen-fixtures.sh --only GenDocsOnlyNorms")
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
    let get = |k: String| manifest.get(&k).unwrap_or_else(|| panic!("{k}")).clone();
    let reader = DirectoryReader::open(&lucene_store::FsDirectory::open(base)).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms("docs_only");
    assert!(
        owned.iter().all(Option::is_some),
        "the field keeps its norms"
    );
    let term_norms: Vec<Option<&FieldNorms<'_>>> = owned.iter().map(Option::as_ref).collect();
    let maps: Vec<HashMap<String, FieldNorms<'_>>> = reader
        .field_norms("docs_only")
        .into_iter()
        .map(|n| {
            n.into_iter()
                .map(|n| ("docs_only".to_string(), n))
                .collect()
        })
        .collect();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = maps.iter().map(Some).collect();
    let runs: usize = get("run_count".into()).parse().unwrap();
    assert_eq!(runs, 18);
    let mut ended_early = 0;
    for r in 0..runs {
        let k = |f: &str| get(format!("run.{r}.{f}"));
        let term = k("term");
        let boost = f32::from_bits(k("boost").parse::<i32>().unwrap() as u32);
        let threshold: u64 = match k("threshold").as_str() {
            "max" => u64::MAX,
            n => n.parse().unwrap(),
        };
        let want: Vec<(i32, u32)> = k("hits")
            .split(',')
            .map(|h| {
                let (d, s) = h.split_once(':').unwrap();
                (d.parse().unwrap(), s.parse::<i32>().unwrap() as u32)
            })
            .collect();
        let want_total: u64 = k("total").parse().unwrap();
        let want_gte = k("relation") == "gte";
        let q = TermQuery::new("docs_only", term.as_bytes().to_vec());
        crate::test_only_maxscore_block_skip_counter::reset();
        let (hits, total) = if boost == 1.0 {
            crate::multi_segment::search_term_query_multi_segment_counting(
                &segments,
                &q,
                &term_norms,
                10,
                threshold,
            )
            .unwrap()
        } else {
            let mut b = BooleanQuery::new();
            b.must.push(Clause::Boost(Box::new(BoostQuery::new(
                Clause::Term(q),
                boost,
            ))));
            crate::multi_segment::search_boolean_query_multi_segment_maxscore_counting(
                &segments, &b, &norms, 10, threshold,
            )
            .unwrap()
        };
        let got: Vec<(i32, u32)> = hits.iter().map(|h| (h.doc_id, h.score.to_bits())).collect();
        assert_eq!(
            got, want,
            "run {r}: docs_only:{term}^{boost} threshold {threshold}"
        );
        assert_eq!(
            (
                total.value,
                total.relation == crate::collector::TotalHitsRelation::GreaterThanOrEqualTo
            ),
            (want_total, want_gte),
            "run {r}: docs_only:{term}^{boost} threshold {threshold}"
        );
        if want_gte && crate::test_only_maxscore_block_skip_counter::count() > 0 {
            ended_early += 1;
        }
    }
    // `kw:a` and `kw:b` at both finite thresholds, boosted or not.
    assert_eq!(ended_early, 8, "the scans a full queue ends");
}

/// Pruning under a similarity other than BM25 is sound: on an index whose
/// terms span many full postings blocks (so `MaxScoreCache`'s per-level
/// bounds, not just the global one, decide what is skipped), the top 10 a
/// pruned search keeps are the first 10 of an unpruned one -- a queue too
/// large to fill never publishes a threshold -- and blocks really are skipped.
#[test]
fn similarity_bounds_prune_soundly() {
    use crate::directory_reader::DirectoryReader;
    use crate::multi_segment::search_boolean_query_multi_segment_with_similarity as search;
    use crate::query::{BooleanQuery, Clause, PhraseQuery, TermQuery};
    use crate::similarities::*;
    use std::sync::Arc;
    let base = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/mixed_boolean_scoring_index"
    );
    let reader = DirectoryReader::open(&lucene_store::FsDirectory::open(base)).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<_> = owned.iter().map(Some).collect();
    let t = |w: &str| Clause::Term(TermQuery::new("body", w));
    let queries = [
        BooleanQuery::new().with_must([t("w0")]),
        BooleanQuery::new().with_must([t("w3")]),
        BooleanQuery::new().with_should([t("w0"), t("w1"), t("w2")]),
        BooleanQuery::new()
            .with_must([t("w0")])
            .with_should([t("w3")]),
        BooleanQuery::new().with_must([t("w1"), t("w2")]),
        BooleanQuery::new().with_must([Clause::Phrase(PhraseQuery::new("body", ["w0", "w1"]))]),
        BooleanQuery::new().with_should([
            Clause::Phrase(PhraseQuery::new("body", ["w1", "w0"]).with_slop(2)),
            t("w4"),
        ]),
    ];
    let sims: [Arc<dyn Similarity>; 4] = [
        Arc::new(ClassicSimilarity::default()),
        Arc::new(
            DfrSimilarity::new(BasicModel::G, AfterEffect::L, Normalization::H1_DEFAULT).unwrap(),
        ),
        Arc::new(LmDirichletSimilarity::default()),
        Arc::new(Bm25Similarity::new(2.0, 0.3, true).unwrap()),
    ];
    crate::test_only_maxscore_block_skip_counter::reset();
    for sim in &sims {
        for q in &queries {
            let pruned = search(&segments, q, &norms, 10, sim.as_ref()).unwrap();
            let all = search(&segments, q, &norms, 30_000, sim.as_ref()).unwrap();
            assert_eq!(pruned.len(), 10, "{sim:?} {q:?}");
            assert_eq!(pruned, all[..10], "{sim:?} {q:?}");
        }
    }
    assert!(
        crate::test_only_maxscore_block_skip_counter::count() > 0,
        "the bounds skipped nothing: the test proves nothing about them"
    );
}

impl super::disjunction::Clause for Fake {
    fn scorer(&mut self) -> &mut dyn Scorer {
        self
    }
}

/// `DisjunctionScoreBlockBoundaryPropagator`: clauses ordered by their
/// global maximum, block bounds from the lead clause up, the stronger
/// clauses' next documents ending a block early, and the lead moving past
/// every clause the minimum competitive score has outgrown.
#[test]
fn block_boundary_propagator_follows_the_lead_clause() {
    use super::disjunction::BlockBoundaryPropagator;
    let leaf = |hits: &[(i32, f32)]| {
        Fake::new(Leaf::new(
            hits.to_vec(),
            hits.iter().map(|h| h.0).collect(),
            false,
        ))
    };
    // Given strongest first; the propagator orders them weakest first.
    let mut subs = vec![
        leaf(&[(30, 3.0)]),
        leaf(&[(2, 2.0), (50, 2.0)]),
        leaf(&[(0, 1.0), (20, 1.0), (40, 1.0)]),
    ];
    let mut p = BlockBoundaryPropagator::new(&mut subs).unwrap();
    assert_eq!(p.advance_shallow(&mut subs, 0).unwrap(), BLOCK - 1);
    subs[0].next_doc().unwrap();
    subs[1].next_doc().unwrap();
    // The strongest clause is on 30: the block ends before it.
    assert_eq!(p.advance_shallow(&mut subs, 24).unwrap(), 29);
    assert_eq!(p.advance_shallow(&mut subs, 10).unwrap(), 15);

    // Above the weakest clause's maximum: the middle one leads, the weakest
    // is only propagated to.
    p.set_min_competitive_score(1.5);
    subs[2].shallow = 0;
    assert_eq!(p.advance_shallow(&mut subs, 24).unwrap(), 29);
    assert_eq!(subs[2].shallow, 24);
    subs[1].next_doc().unwrap();
    // The lead advances shallowly to its own document when that is beyond.
    assert_eq!(p.advance_shallow(&mut subs, 24).unwrap(), 29);
    assert_eq!(subs[1].shallow, 50);

    // Above both: the strongest leads alone, and never past the last clause.
    p.set_min_competitive_score(2.5);
    p.set_min_competitive_score(10.0);
    assert_eq!(p.advance_shallow(&mut subs, 24).unwrap(), 31);
    assert_eq!(subs[0].shallow, 30);
}

/// `IndexOrDocValuesQuery` in the tree: run alone it takes the points side
/// (`bulkScorer`), behind a selective term the doc-values side
/// (`get(leadCost)` with the term's cost), and either way it matches what
/// each side matches alone.
#[test]
fn index_or_doc_values_picks_its_side_by_the_lead_cost() {
    use super::extended::IODV_PLANS;
    use crate::directory_reader::DirectoryReader;
    use crate::doc_value_query::IndexOrDocValuesPlan;
    use crate::extended_query::{
        IndexOrDocValuesQuery, NumericDocValuesRangeQuery, PointRangeQuery,
    };
    use crate::index_searcher::IndexSearcher;
    use crate::query::{BooleanQuery, Clause, TermQuery};
    let dir = lucene_store::FsDirectory::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/m7_queries_index"
    ));
    let reader = DirectoryReader::open(&dir).unwrap();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let points = || Clause::from(PointRangeQuery::long_range("num", &[-1000], &[1000]).unwrap());
    let dv = || Clause::from(NumericDocValuesRangeQuery::new("num", -1000, 1000));
    let iodv = || Clause::from(IndexOrDocValuesQuery::new(points(), dv()));
    let hits = |filter: Clause, lead: bool| {
        let mut q = BooleanQuery::new();
        if lead {
            // Five documents' ids: a lead cost of 5.
            let mut ids = BooleanQuery::new();
            for id in ["1", "2", "3", "4", "5"] {
                ids.should
                    .push(Clause::Term(TermQuery::new("id", id.as_bytes().to_vec())));
            }
            q.must.push(Clause::Boolean(Box::new(ids)));
        }
        q.filter.push(filter);
        IODV_PLANS.with(|p| p.borrow_mut().clear());
        let top = searcher.search(&q, 1000).unwrap();
        let plans = IODV_PLANS.with(|p| p.borrow().clone());
        let docs: Vec<(i32, u32)> = top
            .score_docs
            .iter()
            .map(|h| (h.doc, h.score.to_bits()))
            .collect();
        (docs, plans)
    };
    for lead in [false, true] {
        let (by_points, none) = hits(points(), lead);
        assert!(none.is_empty());
        let (by_dv, _) = hits(dv(), lead);
        let (either, plans) = hits(iodv(), lead);
        assert!(!either.is_empty());
        assert_eq!(either, by_points);
        assert_eq!(either, by_dv);
        let want = if lead {
            IndexOrDocValuesPlan::DocValues
        } else {
            IndexOrDocValuesPlan::Index
        };
        assert!(!plans.is_empty());
        assert!(plans.iter().all(|&p| p == want), "lead {lead}: {plans:?}");
    }
    // A segment without the field's doc values (or points): no scorer, as
    // Java's `null` supplier.
    let no_dv = Clause::from(IndexOrDocValuesQuery::new(
        points(),
        NumericDocValuesRangeQuery::new("nope", 0, 100),
    ));
    assert!(hits(no_dv, true).0.is_empty());
    let no_points = Clause::from(IndexOrDocValuesQuery::new(
        PointRangeQuery::long_range("nope", &[0], &[100]).unwrap(),
        dv(),
    ));
    assert!(hits(no_points, true).0.is_empty());
    assert!(!hits(no_points_optional(), false).0.is_empty());

    fn no_points_optional() -> Clause {
        let mut q = BooleanQuery::new();
        q.should.push(Clause::from(IndexOrDocValuesQuery::new(
            PointRangeQuery::long_range("nope", &[0], &[100]).unwrap(),
            NumericDocValuesRangeQuery::new("num", 0, 100),
        )));
        q.should
            .push(Clause::Term(TermQuery::new("body", b"t7".to_vec())));
        Clause::Boolean(Box::new(q))
    }
}

#[test]
fn a_scorer_confirms_a_batch_one_document_at_a_time_by_default() {
    let mut s = AllDocs::new(10);
    assert!(!s.batch_matches() && !s.prefers_batches() && !s.constant_scores());
    let mut keep = [false; 3];
    s.matches_batch(&[2, 5, 9], &mut keep).unwrap();
    assert_eq!(keep, [true; 3]);
    assert_eq!(s.doc_id(), 9);
}

/// A block join scoring its children (unbounded maximum) as the one `MUST`
/// beside one optional term takes `Bulk::ReqScorerOpt` in `TOP_SCORES`, and
/// returns Lucene's hits and score bits (`GenBlockJoin`'s
/// recorded searches); with `ScoreMode::None` the join's maximum is its
/// child's, a threshold can skip on it, and top hits keep the generic
/// `ReqOptSumScorer`.
#[test]
fn a_scoring_block_join_beside_an_optional_term_scores_like_lucene() {
    use crate::directory_reader::DirectoryReader;
    use crate::field_norms::FieldNorms;
    use crate::index_searcher::{IndexSearcher, SegmentNorms};
    use crate::join::{QueryBitSetProducer, ScoreMode, ToParentBlockJoinQuery};
    use crate::query::{BooleanQuery, Clause, TermQuery};
    use std::collections::HashMap;
    use std::sync::Arc;
    let data =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/block_join");
    let text = std::fs::read_to_string(data.join("searches.tsv")).unwrap();
    let reader =
        DirectoryReader::open(&lucene_store::FsDirectory::open(data.join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned
        .iter()
        .map(|m: &HashMap<String, FieldNorms<'_>>| Some(m))
        .collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let term = |f: &str, v: &str| Clause::Term(TermQuery::new(f, v.as_bytes().to_vec()));
    let query = |mode: ScoreMode, word: &str| {
        let parents = Arc::new(QueryBitSetProducer::new(BooleanQuery {
            must: vec![term("type", "parent")],
            ..Default::default()
        }));
        let children = BooleanQuery {
            filter: vec![term("type", "child")],
            ..Default::default()
        };
        let inner = BooleanQuery {
            must: vec![ToParentBlockJoinQuery::new(children, parents, mode).into()],
            should: vec![term("body", word)],
            ..Default::default()
        };
        BooleanQuery {
            must: vec![Clause::Boolean(Box::new(inner))],
            ..Default::default()
        }
    };
    let hits = |q: &BooleanQuery, n: usize| -> Vec<(i32, u32)> {
        searcher
            .search(q, n)
            .unwrap()
            .score_docs
            .iter()
            .map(|h| (h.doc, h.score.to_bits()))
            .collect()
    };
    for (mode, name, word) in [
        (ScoreMode::Max, "Max", "green"),
        (ScoreMode::Total, "Total", "rare"),
    ] {
        for (kind, n) in [("all", 100_000), ("top", 10)] {
            let spec = format!(
                "bool(must:tp({name},P0,bool(filter:t(type,child))),should:t(body,{word}))"
            );
            let want: Vec<(i32, u32)> = text
                .lines()
                .find_map(|l| l.strip_prefix(&format!("{kind}\t{spec}\t")))
                .unwrap_or_else(|| panic!("no fixture line for {kind} {spec}"))
                .split(' ')
                .filter(|s| *s != "-")
                .map(|h| {
                    let (d, b) = h.split_once(':').unwrap();
                    (d.parse().unwrap(), u32::from_str_radix(b, 16).unwrap())
                })
                .collect();
            super::bulk::test_only_req_scorer_opt::take();
            assert_eq!(hits(&query(mode, word), n), want, "{kind} {spec}");
            assert!(
                super::bulk::test_only_req_scorer_opt::take() > 0,
                "{kind} {spec}: the path ran"
            );
        }
    }
    // No unbounded maximum: top hits (any `n`) through `ReqOptSumScorer`.
    let q = query(ScoreMode::None, "green");
    super::bulk::test_only_req_scorer_opt::take();
    let top = hits(&q, 10);
    let all = hits(&q, 100_000);
    assert_eq!(super::bulk::test_only_req_scorer_opt::take(), 0);
    assert_eq!(top[..], all[..top.len()]);
}
