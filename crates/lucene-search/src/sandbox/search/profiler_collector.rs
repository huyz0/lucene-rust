//! The collector profilers: `ProfilerCollector` wraps a collector
//! (`ProfilerCollectorWrapper`), timing every call that reaches it --
//! `scoreMode()` and each `collect(doc)`, at least one nanosecond a call --
//! and reports a `ProfilerCollectorResult` tree: the wrapped collector's
//! name (its type's simple name, `deriveCollectorName`), a reason, the time,
//! and the results of the profilers given as its children.
//! `ProfilerCollectorManager` makes one profiled collector per slice and
//! reduces them to one result without children: their name and reason, and
//! the sum of their times.
//!
//! Java's wrapper is a `FilterCollector` whose leaf collector forwards only
//! `collect(doc)`, `setScorer` and `finish()`: a bulk path (`collectRange`, a
//! `DocIdStream`, the competitive iterator) falls back to one timed
//! `collect` per document, while the scorer still sees the wrapped
//! collector's minimum competitive score. Here the wrapper forwards the
//! threshold questions ([`ScoringCollector::min_competitive_score`],
//! [`ScoringCollector::pruning_threshold`],
//! [`ScoringCollector::constant_score_hits_needed`]) and leaves the bulk
//! ones (`collect_many`, `add_hits`, `count_losing_hits`) at their
//! defaults, which hand every document to [`ScoringCollector::collect`]:
//! the same documents, scores and totals as the collector unwrapped.
//!
//! Differences: the times are this process's and are not Lucene's to
//! compare; a child is a [`ProfilerHandle`] (the profiler's name, reason and
//! shared clock) rather than the profiler itself, which a parent cannot hold
//! while the child is collecting elsewhere; the derived name is the Rust
//! type's (`TopDocsCollector` for Java's `TopScoreDocCollector`); a manager
//! reducing no collectors names an empty name and reason where Java's are
//! `null`; `setWeight`, `finish()` and the `getLeafCollector` timing have no
//! counterpart (no `Weight`, and leaves are not opened through a
//! collector).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::collector::{ScoreMode, ScoringCollector};
use crate::collectors::CollectorManager;
use crate::Result;

/// `ProfilerCollectorResult`: a profiled collector's name, reason, time in
/// nanoseconds and children.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfilerCollectorResult {
    pub name: String,
    pub reason: String,
    pub time: u64,
    pub children: Vec<ProfilerCollectorResult>,
}

impl ProfilerCollectorResult {
    /// `getProfiledChildren()`.
    pub fn profiled_children(&self) -> &[ProfilerCollectorResult] {
        &self.children
    }
}

/// What a parent profiler reads of a child (`ProfilerCollector` in Java's
/// `children` list): its name, reason, clock and own children.
#[derive(Debug, Clone)]
pub struct ProfilerHandle {
    name: String,
    reason: String,
    time: Arc<AtomicU64>,
    children: Vec<ProfilerHandle>,
}

impl ProfilerHandle {
    /// `doGetCollectorTree(collector)`.
    pub fn profile_result(&self) -> ProfilerCollectorResult {
        ProfilerCollectorResult {
            name: self.name.clone(),
            reason: self.reason.clone(),
            time: self.time.load(Ordering::Relaxed),
            children: self.children.iter().map(Self::profile_result).collect(),
        }
    }
}

/// `deriveCollectorName(c)`: the type's simple name, its path and generic
/// arguments dropped.
fn simple_name<C>() -> String {
    let full = std::any::type_name::<C>();
    let base = full.split('<').next().unwrap_or(full);
    base.rsplit("::").next().unwrap_or(base).to_string()
}

/// `ProfilerCollector(collector, reason, children)`.
#[derive(Debug)]
pub struct ProfilerCollector<C> {
    collector: C,
    handle: ProfilerHandle,
}

impl<C: ScoringCollector> ProfilerCollector<C> {
    /// `new ProfilerCollector(collector, reason, children)`, named after
    /// the collector's type.
    pub fn new(collector: C, reason: impl Into<String>, children: Vec<ProfilerHandle>) -> Self {
        Self::with_name(collector, simple_name::<C>(), reason, children)
    }

    /// A profiler whose `deriveCollectorName` is overridden to `name`.
    pub fn with_name(
        collector: C,
        name: impl Into<String>,
        reason: impl Into<String>,
        children: Vec<ProfilerHandle>,
    ) -> Self {
        Self {
            collector,
            handle: ProfilerHandle {
                name: name.into(),
                reason: reason.into(),
                time: Arc::new(AtomicU64::new(0)),
                children,
            },
        }
    }

    /// `getTime()`: nanoseconds spent in the wrapped collector.
    pub fn time(&self) -> u64 {
        self.handle.time.load(Ordering::Relaxed)
    }

    /// `getReason()`.
    pub fn reason(&self) -> &str {
        &self.handle.reason
    }

    /// `getName()`.
    pub fn name(&self) -> &str {
        &self.handle.name
    }

    /// This profiler as a parent's child.
    pub fn handle(&self) -> ProfilerHandle {
        self.handle.clone()
    }

    /// `getProfileResult()`.
    pub fn profile_result(&self) -> ProfilerCollectorResult {
        self.handle.profile_result()
    }

    /// The wrapped collector.
    pub fn collector(&self) -> &C {
        &self.collector
    }

    /// The wrapped collector, the profiler dropped.
    pub fn into_collector(self) -> C {
        self.collector
    }

    /// `time += Math.max(1, System.nanoTime() - start)`.
    fn charge(&self, start: Instant) {
        let spent = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.handle.time.fetch_add(spent.max(1), Ordering::Relaxed);
    }
}

impl<C: ScoringCollector> ScoringCollector for ProfilerCollector<C> {
    fn collect(&mut self, doc_id: i32, score: f32) {
        let start = Instant::now();
        self.collector.collect(doc_id, score);
        self.charge(start);
    }

    fn score_mode(&self) -> ScoreMode {
        let start = Instant::now();
        let mode = self.collector.score_mode();
        self.charge(start);
        mode
    }

    fn min_competitive_score(&self) -> Option<f32> {
        self.collector.min_competitive_score()
    }

    fn pruning_threshold(&self) -> Option<f32> {
        self.collector.pruning_threshold()
    }

    fn constant_score_hits_needed(&self) -> Option<u64> {
        self.collector.constant_score_hits_needed()
    }
}

/// `ProfilerCollectorManager(reason)`, its `createCollector()` the given
/// function.
pub struct ProfilerCollectorManager<F> {
    reason: String,
    create_collector: F,
}

impl<F> std::fmt::Debug for ProfilerCollectorManager<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ProfilerCollectorManager(reason={})", self.reason)
    }
}

impl<F, C> ProfilerCollectorManager<F>
where
    F: Fn() -> Result<C> + Sync,
    C: ScoringCollector + Send,
{
    /// `ProfilerCollectorManager(reason)` with `createCollector()`.
    pub fn new(reason: impl Into<String>, create_collector: F) -> Self {
        Self {
            reason: reason.into(),
            create_collector,
        }
    }
}

impl<F, C> CollectorManager for ProfilerCollectorManager<F>
where
    F: Fn() -> Result<C> + Sync,
    C: ScoringCollector + Send,
{
    type Collector = ProfilerCollector<C>;
    type Output = ProfilerCollectorResult;

    /// `newCollector()`: `new ProfilerCollector(createCollector(), reason,
    /// List.of())`.
    fn new_collector(&self) -> Result<Self::Collector> {
        Ok(ProfilerCollector::new(
            (self.create_collector)()?,
            self.reason.clone(),
            Vec::new(),
        ))
    }

    /// `reduce(collectors)`: the last collector's name and reason (Java
    /// asserts they all agree), the sum of their times, no children.
    fn reduce(&self, collectors: Vec<Self::Collector>) -> Result<ProfilerCollectorResult> {
        let mut out = ProfilerCollectorResult {
            name: String::new(),
            reason: String::new(),
            time: 0,
            children: Vec::new(),
        };
        for c in &collectors {
            out.name = c.name().to_string();
            out.reason = c.reason().to_string();
            out.time = out.time.saturating_add(c.profile_result().time);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::TopDocsCollector;
    use crate::sandbox::search::LargeNumHitsTopDocsCollector;

    #[test]
    fn names_are_the_simple_type_name() {
        assert_eq!(simple_name::<TopDocsCollector>(), "TopDocsCollector");
        assert_eq!(simple_name::<Vec<u8>>(), "Vec");
        let p = ProfilerCollector::with_name(TopDocsCollector::new(3), "X", "why", Vec::new());
        assert_eq!((p.name(), p.reason()), ("X", "why"));
    }

    #[test]
    fn times_every_call_and_passes_results_through() {
        let mut p = ProfilerCollector::new(
            LargeNumHitsTopDocsCollector::new(2).unwrap(),
            "large",
            Vec::new(),
        );
        assert_eq!(p.time(), 0);
        assert_eq!(p.score_mode(), ScoreMode::Complete);
        let after_mode = p.time();
        assert!(after_mode >= 1);
        for (d, s) in [(0, 1.0), (1, 2.0), (2, 3.0)] {
            p.collect(d, s);
        }
        // At least a nanosecond for each of the three calls.
        assert!(p.time() >= after_mode + 3);
        assert_eq!(p.min_competitive_score(), None);
        assert_eq!(p.pruning_threshold(), None);
        assert_eq!(p.constant_score_hits_needed(), None);
        // The bulk paths are not forwarded: the caller collects one by one.
        assert!(!p.add_hits(0));
        assert!(!p.count_losing_hits(1, 0.5));
        let r = p.profile_result();
        assert_eq!(r.name, "LargeNumHitsTopDocsCollector");
        assert_eq!(r.reason, "large");
        assert_eq!(r.time, p.time());
        assert!(r.profiled_children().is_empty());
        let mut inner = p.into_collector();
        let top: Vec<i32> = inner.top_docs().score_docs.iter().map(|s| s.doc).collect();
        assert_eq!(top, vec![2, 1]);
    }

    #[test]
    fn a_parent_reports_its_children() {
        let mut child = ProfilerCollector::new(TopDocsCollector::new(1), "child", Vec::new());
        let grand = ProfilerCollector::new(TopDocsCollector::new(1), "grand", Vec::new());
        let child_handle = ProfilerHandle {
            children: vec![grand.handle()],
            ..child.handle()
        };
        let parent = ProfilerCollector::new(TopDocsCollector::new(1), "parent", vec![child_handle]);
        child.collect(0, 1.0);
        let r = parent.profile_result();
        assert_eq!(r.children.len(), 1);
        // The child's clock is shared: its collecting shows in the tree.
        assert!(r.children[0].time >= 1);
        assert_eq!(r.children[0].reason, "child");
        assert_eq!(r.children[0].children[0].reason, "grand");
        assert_eq!(r.children[0].children[0].time, 0);
        assert!(!format!("{:?}", parent.collector()).is_empty());
    }

    #[test]
    fn the_manager_sums_its_collectors() {
        let m = ProfilerCollectorManager::new("top", || Ok(TopDocsCollector::new(2)));
        assert_eq!(format!("{m:?}"), "ProfilerCollectorManager(reason=top)");
        let empty = m.reduce(Vec::new()).unwrap();
        assert_eq!((empty.name.as_str(), empty.time), ("", 0));
        let mut a = m.new_collector().unwrap();
        let mut b = m.new_collector().unwrap();
        a.collect(0, 1.0);
        b.collect(1, 1.0);
        let total = a.time() + b.time();
        let r = m.reduce(vec![a, b]).unwrap();
        assert_eq!(r.name, "TopDocsCollector");
        assert_eq!(r.reason, "top");
        assert_eq!(r.time, total);
        assert!(r.children.is_empty());
        let failing = ProfilerCollectorManager::new("x", || {
            Err::<TopDocsCollector, _>(crate::Error::IllegalState("no".into()))
        });
        assert!(failing.new_collector().is_err());
    }
}
