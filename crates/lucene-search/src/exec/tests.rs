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
                _ => Some(Box::new(DisjunctionScorer::new(
                    subs,
                    Combine::Max(*tie),
                    mode.needs_scores(),
                ))),
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
        let opened = reader.open_segments().unwrap();
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
                    points: None,
                    norms: norms[s],
                    global: Some(&global),
                    max_doc: None,
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
            "scorer_conjunction",
            "disjunction",
            "filtered_disjunction",
            "scorer_disjunction",
            "req_opt",
            "req_excl",
            "dismax",
        ]
        .into_iter()
        .collect();
        assert_eq!(kinds, all, "bulk scorers the fixture reaches");
        let paths = test_only_req_opt_paths::take();
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
    assert_eq!(
        all.doc_id_run_end(),
        3,
        "one document, unless a scorer knows better"
    );
}
