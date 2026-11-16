//! Profile mode: the scorer tree of a query, built and driven with a timer around every
//! clause, as OpenSearch's `ProfileWeight` and `ProfileScorer` time a Lucene tree.
//!
//! A profile request runs the query a second time, with a
//! [`Session`] installed on the thread ([`Guard`]). The scorer builder then
//!
//! * builds every clause as its own scorer ([`active`] turns off the bulk shapes that fuse
//!   term legs and nested booleans into one loop), and
//! * wraps each of them in a [`ProfileScorer`], which counts and times the calls the
//!   stock profile counts: `build_scorer`, `next_doc`, `advance`, `match`, `score`,
//!   `shallow_advance`, `compute_max_score` and `set_min_competitive_score`.
//!
//! Timers keep stock's sampling (`org.opensearch.search.profile.Timer`): every call is
//! counted, the first 256 are timed, then one in two up to 512, and so on to one in 1024,
//! so a clause stepped a million times costs a few thousand clock reads, not a million.
//!
//! A clause is identified by its [`Clause`] path from the root (the index of each step
//! among a clause's children, [`children`]), which the host matches against the query tree
//! it encoded. A clause a builder derives (a rewrite's copy, a clause built inside an
//! extended query) is not found under the one being built and is left unwrapped: its time
//! is its parent's.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::OnceLock;
use std::time::Instant;

use lucene_util::fixed_bit_set::FixedBitSet;

use super::{BoxScorer, Scorer};
use crate::query::Clause;
use crate::Result;

/// The timers a clause has, in the order [`Kind`] names them. (Stock's ninth, `create_weight`,
/// is the host's: nothing here builds a `Weight`.)
pub const TIMERS: usize = 8;

/// What a timer times; the discriminant is its index in [`NodeTimers`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `ScorerSupplier` creation and `get`: building the clause's scorer for one segment.
    BuildScorer = 0,
    /// `DocIdSetIterator.nextDoc` of the approximation.
    NextDoc = 1,
    /// `DocIdSetIterator.advance` of the approximation.
    Advance = 2,
    /// `TwoPhaseIterator.matches`.
    Match = 3,
    /// `Scorer.score`.
    Score = 4,
    /// `Scorer.advanceShallow`.
    ShallowAdvance = 5,
    /// `Scorer.getMaxScore`.
    ComputeMaxScore = 6,
    /// `Scorable.setMinCompetitiveScore`.
    SetMinCompetitiveScore = 7,
}

/// Nanoseconds since the first call of the process, plus one, so that no reading is `0`
/// (stock's `Timer` takes `start == 0` for "not running").
fn now() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let epoch = EPOCH.get_or_init(Instant::now);
    u64::try_from(epoch.elapsed().as_nanos()).unwrap_or(u64::MAX - 1) + 1
}

/// Nanoseconds on the clock [`Timer`]s read, for a host to relate the starts it is given to
/// its own clock.
pub fn clock() -> u64 {
    now()
}

/// `org.opensearch.search.profile.Timer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timer {
    timing: u64,
    count: u64,
    last_count: u64,
    start: u64,
    earliest_start: u64,
    timing_this_call: bool,
}

impl Timer {
    #[inline]
    fn start(&mut self) {
        self.timing_this_call = (self.count - self.last_count) >= (self.last_count >> 8).min(1024);
        if self.timing_this_call {
            self.start = now();
            if self.count == 0 {
                self.earliest_start = self.start;
            }
        }
        self.count += 1;
    }

    #[inline]
    fn stop(&mut self) {
        if self.timing_this_call {
            self.timing += (self.count - self.last_count) * now().saturating_sub(self.start).max(1);
            self.last_count = self.count;
            self.start = 0;
        }
    }

    /// How many times the timer ran.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// The clock reading the timer first started at ([`clock`]), `0` when it never did.
    pub fn earliest_start(&self) -> u64 {
        self.earliest_start
    }

    /// The time the timed calls took, extended to the calls that were only counted
    /// (`Timer.getApproximateTiming`).
    pub fn nanos(&self) -> u64 {
        let mut timing = self.timing;
        if self.count > self.last_count && self.last_count > 0 {
            timing += (self.count - self.last_count) * timing / self.last_count;
        }
        timing
    }
}

/// One clause's timers over one segment.
pub type NodeTimers = Rc<RefCell<[Timer; TIMERS]>>;

/// A clause being built: its path, and the path of each clause under it by address.
struct Frame {
    path: Vec<u32>,
    below: HashMap<usize, Vec<u32>>,
}

/// The name of `clause`'s variant: the head of its `Debug` form.
fn kind_of(clause: &Clause) -> String {
    use std::fmt::Write;
    struct Head(String);
    impl Write for Head {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            for c in s.chars() {
                if !(c.is_alphanumeric() || c == '_') {
                    return Err(std::fmt::Error);
                }
                self.0.push(c);
            }
            Ok(())
        }
    }
    let mut head = Head(String::new());
    // The error only stops the formatting once the name is out.
    let _ = write!(head, "{clause:?}");
    head.0
}

fn address(clause: &Clause) -> usize {
    std::ptr::from_ref(clause) as usize
}

/// The path of every clause under `clause`, by address.
fn descendants(clause: &Clause) -> HashMap<usize, Vec<u32>> {
    fn walk(clause: &Clause, path: &mut Vec<u32>, out: &mut HashMap<usize, Vec<u32>>) {
        for (i, child) in children(clause).into_iter().enumerate() {
            path.push(u32::try_from(i).unwrap_or(u32::MAX));
            out.insert(address(child), path.clone());
            walk(child, path, out);
            path.pop();
        }
    }
    let mut out = HashMap::new();
    walk(clause, &mut Vec::new(), &mut out);
    out
}

/// What a profiled run records: per segment and clause path, the clause's timers.
#[derive(Default)]
pub struct Session {
    segment: Cell<usize>,
    stack: RefCell<Vec<Frame>>,
    nodes: RefCell<HashMap<(usize, Vec<u32>), NodeTimers>>,
    order: RefCell<Vec<(usize, Vec<u32>)>>,
    kinds: RefCell<HashMap<Vec<u32>, String>>,
}

/// One clause's timers over one segment, as a profiled run reports them.
#[derive(Debug, Clone)]
pub struct NodeReport {
    /// The segment's index in the reader.
    pub segment: usize,
    /// The clause: the index of each step down from the root among a clause's [`children`].
    pub path: Vec<u32>,
    /// What the clause is: its [`Clause`] variant's name (`Term`, `Boolean`, `Boost`, ...).
    pub kind: String,
    /// The timers, by [`Kind`].
    pub timers: [Timer; TIMERS],
}

impl Session {
    /// An empty session.
    pub fn new() -> Rc<Self> {
        Rc::new(Self::default())
    }

    /// The segment the scorers built from now on belong to.
    pub fn set_segment(&self, segment: usize) {
        self.segment.set(segment);
    }

    /// Everything recorded, in the order clauses were first built.
    pub fn report(&self) -> Vec<NodeReport> {
        let nodes = self.nodes.borrow();
        let kinds = self.kinds.borrow();
        self.order
            .borrow()
            .iter()
            .filter_map(|key| {
                nodes.get(key).map(|t| NodeReport {
                    segment: key.0,
                    path: key.1.clone(),
                    kind: kinds.get(&key.1).cloned().unwrap_or_default(),
                    timers: *t.borrow(),
                })
            })
            .collect()
    }

    fn timers(&self, path: &[u32], clause: &Clause) -> NodeTimers {
        let key = (self.segment.get(), path.to_vec());
        let mut nodes = self.nodes.borrow_mut();
        if let Some(existing) = nodes.get(&key) {
            return Rc::clone(existing);
        }
        self.kinds
            .borrow_mut()
            .entry(path.to_vec())
            .or_insert_with(|| kind_of(clause));
        let fresh: NodeTimers = Rc::default();
        nodes.insert(key.clone(), Rc::clone(&fresh));
        self.order.borrow_mut().push(key);
        fresh
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Rc<Session>>> = const { RefCell::new(None) };
}

/// Installs a [`Session`] on this thread until dropped.
pub struct Guard {
    previous: Option<Rc<Session>>,
}

impl Guard {
    /// `session` records the scorer trees built on this thread while the guard lives.
    pub fn install(session: &Rc<Session>) -> Self {
        let previous = CURRENT.with(|c| c.borrow_mut().replace(Rc::clone(session)));
        Self { previous }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        CURRENT.with(|c| *c.borrow_mut() = previous);
    }
}

fn installed() -> Option<Rc<Session>> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Whether a [`Session`] is installed on this thread.
pub(crate) fn active() -> bool {
    CURRENT.with(|c| c.borrow().is_some())
}

/// The clauses a clause is built from, in the order a path's steps index them: a boolean's
/// `MUST`, `FILTER`, `SHOULD` and `MUST_NOT` clauses (in that order, as the host's tree
/// lists them), a dismax's disjuncts, a boost's or constant score's inner clause.
pub fn children(clause: &Clause) -> Vec<&Clause> {
    match clause {
        Clause::Boolean(b) => b
            .must
            .iter()
            .chain(&b.filter)
            .chain(&b.should)
            .chain(&b.must_not)
            .collect(),
        Clause::DisjunctionMax(d) => d.disjuncts.iter().collect(),
        Clause::ConstantScore(c) => vec![c.inner.as_ref()],
        Clause::Boost(b) => vec![b.inner.as_ref()],
        _ => Vec::new(),
    }
}

/// Builds `clause`'s scorer with `build`, timed as the clause's `build_scorer`, and wraps it in a
/// [`ProfileScorer`] when a session is installed; otherwise just `build()`.
pub(crate) fn profiled<'a>(
    clause: &Clause,
    build: impl FnOnce() -> Result<Option<BoxScorer<'a>>>,
) -> Result<Option<BoxScorer<'a>>> {
    let Some(session) = installed() else {
        return build();
    };
    let Some(path) = enter(&session, clause) else {
        return build();
    };
    let timers = session.timers(&path, clause);
    timers.borrow_mut()[Kind::BuildScorer as usize].start();
    let built = build();
    timers.borrow_mut()[Kind::BuildScorer as usize].stop();
    session.stack.borrow_mut().pop();
    Ok(built?.map(|inner| -> BoxScorer<'a> { Box::new(ProfileScorer { inner, timers }) }))
}

/// Starts building `clause` (pushing it), returning its path, or `None` when it is not one of the
/// clauses being profiled: the root of a session is the first clause built with nothing building.
fn enter(session: &Session, clause: &Clause) -> Option<Vec<u32>> {
    let mut stack = session.stack.borrow_mut();
    let path = match stack.last() {
        None => Vec::new(),
        Some(parent) => {
            let mut path = parent.path.clone();
            path.extend_from_slice(parent.below.get(&address(clause))?);
            path
        }
    };
    stack.push(Frame {
        path: path.clone(),
        below: descendants(clause),
    });
    Some(path)
}

/// Wraps a scorer of `clause` that did not go through [`profiled`] (a term leg or a cached set a
/// boolean built itself): its path is found under the clause being built, and building it took
/// no time worth a timer of its own.
pub(crate) fn wrap_child<'a>(clause: &Clause, scorer: BoxScorer<'a>) -> BoxScorer<'a> {
    let Some(session) = installed() else {
        return scorer;
    };
    let path = {
        let stack = session.stack.borrow();
        let Some(parent) = stack.last() else {
            return scorer;
        };
        let Some(rest) = parent.below.get(&address(clause)) else {
            return scorer;
        };
        let mut path = parent.path.clone();
        path.extend_from_slice(rest);
        path
    };
    let timers = session.timers(&path, clause);
    {
        let mut t = timers.borrow_mut();
        t[Kind::BuildScorer as usize].start();
        t[Kind::BuildScorer as usize].stop();
    }
    Box::new(ProfileScorer {
        inner: scorer,
        timers,
    })
}

/// `ProfileScorer`: the wrapped scorer, with a timer around the calls stock times. What it does
/// not forward are the bulk shortcuts ([`Scorer::next_docs_and_scores`], constant ranges,
/// random access): a caller then steps it a document at a time through the methods that are
/// timed, as `DefaultBulkScorer` steps a stock `ProfileScorer`.
pub struct ProfileScorer<'a> {
    inner: BoxScorer<'a>,
    timers: NodeTimers,
}

impl ProfileScorer<'_> {
    #[inline]
    fn timed<T>(&mut self, kind: Kind, f: impl FnOnce(&mut Self) -> T) -> T {
        self.timers.borrow_mut()[kind as usize].start();
        let out = f(self);
        self.timers.borrow_mut()[kind as usize].stop();
        out
    }
}

impl Scorer for ProfileScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.inner.doc_id()
    }

    fn next_doc(&mut self) -> Result<i32> {
        self.timed(Kind::NextDoc, |s| s.inner.next_doc())
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        self.timed(Kind::Advance, |s| s.inner.advance(target))
    }

    fn cost(&self) -> i64 {
        self.inner.cost()
    }

    fn two_phase(&self) -> bool {
        self.inner.two_phase()
    }

    fn matches(&mut self) -> Result<bool> {
        // Lucene calls `matches` only on a scorer with a two-phase iterator.
        if !self.inner.two_phase() {
            return self.inner.matches();
        }
        self.timed(Kind::Match, |s| s.inner.matches())
    }

    fn match_cost(&self) -> f32 {
        self.inner.match_cost()
    }

    fn score(&mut self) -> Result<f32> {
        self.timed(Kind::Score, |s| s.inner.score())
    }

    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        self.timed(Kind::ShallowAdvance, |s| s.inner.advance_shallow(target))
    }

    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        self.timed(Kind::ComputeMaxScore, |s| s.inner.max_score(up_to))
    }

    fn max_score_unbounded(&self) -> bool {
        self.inner.max_score_unbounded()
    }

    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        self.timed(Kind::SetMinCompetitiveScore, |s| {
            s.inner.set_min_competitive_score(min)
        })
    }

    fn doc_id_run_end(&self) -> i32 {
        self.inner.doc_id_run_end()
    }

    fn two_phase_run_end(&self) -> i32 {
        self.inner.two_phase_run_end()
    }

    fn indri_boost(&self) -> Option<f32> {
        self.inner.indri_boost()
    }

    fn smoothing_score(&mut self, doc: i32) -> Result<f32> {
        self.inner.smoothing_score(doc)
    }

    fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        live_docs: Option<&FixedBitSet>,
        out: &mut crate::bulk_scorer::DocScores,
    ) -> Result<()> {
        super::docs_and_scores_one_by_one(self, up_to, live_docs, out)
    }
}

/// `run()`, except in a profiled run, where it is `None`: the shapes that fuse clauses into one
/// iterator (a run of sort-prefix filters) would leave clauses without a scorer to time.
pub(crate) fn unless_profiled<T>(run: impl FnOnce() -> Result<Option<T>>) -> Result<Option<T>> {
    if active() {
        Ok(None)
    } else {
        run()
    }
}

/// The root scorer of a profiled run over one segment: `clause`'s tree, wrapped, as the default
/// bulk scorer steps it.
pub fn root<'a>(
    ctx: &super::LeafContext<'a>,
    clause: &Clause,
    mode: super::Mode,
) -> Result<Option<super::Bulk<'a>>> {
    Ok(super::build::build(ctx, clause, 1.0, mode, true)?.map(super::Bulk::scorer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{BoostQuery, ConstantScoreQuery, DisjunctionMaxQuery, TermQuery};

    fn term(t: &str) -> Clause {
        Clause::Term(TermQuery {
            field: "f".into(),
            term: t.as_bytes().to_vec(),
            doc_freq: None,
        })
    }

    #[test]
    fn a_timer_times_every_call_at_first_then_samples() {
        let mut t = Timer::default();
        assert_eq!((t.count(), t.nanos(), t.earliest_start()), (0, 0, 0));
        for _ in 0..300 {
            t.start();
            t.stop();
        }
        assert_eq!(t.count(), 300);
        assert!(t.earliest_start() > 0);
        assert!(t.nanos() > 0);
        for _ in 0..5000 {
            t.start();
            t.stop();
        }
        assert_eq!(t.count(), 5300);
        assert!(t.last_count <= t.count);
    }

    #[test]
    fn the_clock_never_reads_zero_and_does_not_go_back() {
        let a = clock();
        let b = clock();
        assert!(a > 0 && b >= a);
    }

    #[test]
    fn kinds_are_the_variant_names() {
        assert_eq!(kind_of(&term("a")), "Term");
        let b = Clause::Boost(Box::new(BoostQuery::new(term("a"), 2.0)));
        assert_eq!(kind_of(&b), "Boost");
        let c = Clause::ConstantScore(Box::new(ConstantScoreQuery::new(term("a"), 1.0)));
        assert_eq!(kind_of(&c), "ConstantScore");
    }

    #[test]
    fn children_follow_the_paths_steps() {
        assert!(children(&term("a")).is_empty());
        let dm = Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(
            vec![term("a"), term("b")],
            0.0,
        )));
        assert_eq!(children(&dm).len(), 2);
        let cs = Clause::ConstantScore(Box::new(ConstantScoreQuery::new(dm.clone(), 1.0)));
        assert_eq!(children(&cs).len(), 1);
        let boost = Clause::Boost(Box::new(BoostQuery::new(cs.clone(), 2.0)));
        assert_eq!(children(&boost).len(), 1);
        let paths = descendants(&boost);
        // boost -> constant score (0) -> dismax (0.0) -> its two disjuncts (0.0.0, 0.0.1)
        assert_eq!(paths.len(), 4);
        let mut found: Vec<_> = paths.values().cloned().collect();
        found.sort();
        assert_eq!(
            found,
            vec![vec![0], vec![0, 0], vec![0, 0, 0], vec![0, 0, 1]]
        );
    }

    #[test]
    fn a_session_installs_per_thread_and_nests() {
        assert!(!active());
        let outer = Session::new();
        let inner = Session::new();
        {
            let _a = Guard::install(&outer);
            assert!(active());
            {
                let _b = Guard::install(&inner);
                assert!(Rc::ptr_eq(&installed().unwrap(), &inner));
            }
            assert!(Rc::ptr_eq(&installed().unwrap(), &outer));
        }
        assert!(!active());
        assert!(installed().is_none());
    }

    #[test]
    fn nothing_is_recorded_without_a_session() {
        let clause = term("a");
        let built = profiled(&clause, || Ok(None)).unwrap();
        assert!(built.is_none());
        let ran = unless_profiled(|| Ok(Some(7))).unwrap();
        assert_eq!(ran, Some(7));
    }

    #[test]
    fn a_profiled_build_records_the_root_and_hides_fused_shapes() {
        let session = Session::new();
        session.set_segment(3);
        let _guard = Guard::install(&session);
        let clause = term("a");
        let built = profiled(&clause, || Ok(None)).unwrap();
        assert!(built.is_none());
        let none: Result<Option<i32>> = unless_profiled(|| Ok(Some(1)));
        assert!(none.unwrap().is_none());
        let report = session.report();
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].segment, 3);
        assert_eq!(report[0].kind, "Term");
        assert!(report[0].path.is_empty());
        assert_eq!(report[0].timers[Kind::BuildScorer as usize].count(), 1);
    }
}
