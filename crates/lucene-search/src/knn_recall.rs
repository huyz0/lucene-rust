//! Recall-targeted KNN search: the search depth (an HNSW search's beam, an
//! IVF search's probes and rerank cap) tuned at run time to reach a target
//! recall, measured on a sample of the live queries.
//!
//! A [`RecallTuner`] holds one field's state: a level on a ladder of
//! search efforts from 1/4 to 64, each step `sqrt(2)` deeper, level
//! [`BASE_LEVEL`] the search as asked. An effort scales an HNSW search's
//! candidates (never below `k`) and an IVF search's probes and rerank cap
//! ([`KnnFloatVectorQuery::search_effort`]). Every query searches at the
//! current level. A sampled query is also searched exactly, every segment
//! scanned with its index unused, and again at the [`NEIGHBOURS`] levels
//! on either side; each recall (the share of the exact top `k` the search
//! found) joins a decaying average per level. A level *reaches* the target
//! when its mean recall less [`CONFIDENCE`] standard errors does: a level
//! picked on a lucky run of samples would otherwise miss the target on the
//! queries that follow. The level then moves:
//!
//! - up, when the current level does not reach the target;
//! - down, when the level below does.
//!
//! A level needs [`MIN_SAMPLES`] samples (decayed weight) before it moves
//! anything. The averages decay by [`DECAY`] per sample, so the level
//! follows a change in the data or in the queries within about a hundred
//! samples.
//!
//! **Sampling.** The first [`WARMUP_SAMPLES`] samples are one query in ten
//! (or the sample rate, if higher), so the level settles quickly, while
//! the checks take no longer than the searches ([`WARMUP_BUDGET`]). After
//! that a query is sampled at the sample rate, and only while the checks
//! have taken at most [`RecallTarget::budget`] of the time the tuned
//! searches took: an exact scan costs what reading every vector costs, so
//! on a large index the budget, not the rate, sets how often it runs.
//!
//! **Where the check runs.** [`knn_top_hits_tuned`] checks a sampled query
//! before it returns. A caller that must not add the check to the query's
//! latency runs the search with [`RecallTuner::begin`] and
//! [`knn_top_hits_at_effort`], and calls [`check_sample`] later (on another
//! thread, over the same reader) with the hits it returned.
//!
//! Nothing is tuned unless a caller asks for it: the plain searches in
//! [`crate::knn_search`] never scale their depth.
//!
//! [`KnnFloatVectorQuery::search_effort`]: crate::vector_query::KnnFloatVectorQuery::search_effort

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::collector::ScoreDoc;
use crate::join::BitSetProducer;
use crate::knn_search::{top_hits, BufferedVectors, Depth, KnnTarget};
use crate::multi_segment::OpenSegment;
use crate::query::Clause;
use crate::{Error, Result};

/// The levels of the effort ladder.
pub const LEVELS: usize = 17;

/// The level searching as asked (effort `1.0`).
pub const BASE_LEVEL: usize = 4;

/// The decayed samples a level needs before its average moves the level.
pub const MIN_SAMPLES: f64 = 32.0;

/// Each sample's decay of every level's average.
pub const DECAY: f64 = 0.99;

/// The samples taken at one query in ten (or the sample rate, if higher),
/// within [`WARMUP_BUDGET`], before the sample rate and budget apply.
pub const WARMUP_SAMPLES: u64 = 64;

/// The checks' share of the search time while warming up: as long as the
/// searches took.
pub const WARMUP_BUDGET: f32 = 1.0;

/// The standard errors a level's mean recall must clear the target by:
/// a level reaches the target when its mean less this many standard errors
/// does.
pub const CONFIDENCE: f64 = 1.0;

/// The levels each side of the searched one that a sample also searches.
pub const NEIGHBOURS: usize = 2;

/// The default [`RecallTarget::budget`].
pub const DEFAULT_BUDGET: f32 = 0.05;

/// Level `level`'s search effort: `2^((level - BASE_LEVEL) / 2)`.
pub fn effort(level: usize) -> f32 {
    let steps = level.min(LEVELS - 1) as i32 - BASE_LEVEL as i32;
    2f32.powf(steps as f32 / 2.0)
}

/// What a [`RecallTuner`] aims for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecallTarget {
    /// The mean recall at `k` to reach, in `(0, 1]`.
    pub recall: f32,
    /// The most queries checked against an exact search, as a share of
    /// them, in `(0, 1]`.
    pub sample_rate: f32,
    /// After warm-up, the most time the checks may take, as a share of the
    /// time the tuned searches took, at least `0` ([`DEFAULT_BUDGET`]).
    pub budget: f32,
}

impl RecallTarget {
    /// The target with the [`DEFAULT_BUDGET`], checked.
    ///
    /// # Errors
    /// A recall or sample rate outside `(0, 1]`.
    pub fn new(recall: f32, sample_rate: f32) -> Result<Self> {
        let unit = |v: f32| v > 0.0 && v <= 1.0;
        if !unit(recall) {
            return Err(Error::InvalidKnnQuery(format!(
                "target recall must be in (0, 1]; got {recall}"
            )));
        }
        if !unit(sample_rate) {
            return Err(Error::InvalidKnnQuery(format!(
                "recall sample rate must be in (0, 1]; got {sample_rate}"
            )));
        }
        Ok(Self {
            recall,
            sample_rate,
            budget: DEFAULT_BUDGET,
        })
    }

    /// This target with another budget.
    ///
    /// # Errors
    /// A budget below zero or not finite.
    pub fn with_budget(mut self, budget: f32) -> Result<Self> {
        if !(budget >= 0.0 && budget.is_finite()) {
            return Err(Error::InvalidKnnQuery(format!(
                "recall check budget must be at least 0; got {budget}"
            )));
        }
        self.budget = budget;
        Ok(self)
    }

    /// One query in this many is sampled.
    fn every(&self) -> u64 {
        (1.0 / f64::from(self.sample_rate)).round().max(1.0) as u64
    }
}

/// A level's decaying recall average.
#[derive(Debug, Clone, Copy, Default)]
struct Level {
    /// Sum of the samples' recalls, decayed.
    recall: f64,
    /// Sum of their squares, decayed.
    squares: f64,
    /// The samples, decayed.
    weight: f64,
}

impl Level {
    fn mean(&self) -> Option<f64> {
        (self.weight >= MIN_SAMPLES).then(|| self.recall / self.weight)
    }

    /// The mean less [`CONFIDENCE`] standard errors: what the level is
    /// taken to reach. The variance is at least half a coin flip's at the
    /// mean (`m (1 - m) / 2`): a query's recall is mostly all or nothing
    /// at a high mean, and the rare hard query a few dozen samples miss
    /// would otherwise make the error look smaller than it is.
    fn lower_bound(&self) -> Option<f64> {
        let mean = self.mean()?;
        let observed = self.squares / self.weight - mean * mean;
        let variance = observed.max(mean * (1.0 - mean) / 2.0).max(0.0);
        Some(mean - CONFIDENCE * (variance / self.weight).sqrt())
    }
}

/// What a [`RecallTuner`] has measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecallStats {
    pub target: RecallTarget,
    /// The level searches run at.
    pub level: usize,
    /// That level's search effort.
    pub effort: f32,
    /// The mean recall measured at that level, once it has
    /// [`MIN_SAMPLES`].
    pub recall: Option<f64>,
    /// That mean less [`CONFIDENCE`] standard errors.
    pub recall_lower_bound: Option<f64>,
    /// The tuned searches run.
    pub queries: u64,
    /// The searches checked against an exact one.
    pub samples: u64,
    /// The time the tuned searches took, in nanoseconds.
    pub search_nanos: u64,
    /// The time the checks took, in nanoseconds.
    pub check_nanos: u64,
}

/// One search's place in a [`RecallTuner`]: the effort to search at, and
/// whether to check it ([`check_sample`]).
#[derive(Debug, Clone, Copy)]
pub struct TunedSearch {
    /// The level searched at.
    pub level: usize,
    /// [`effort`] of that level.
    pub effort: f32,
    /// Whether the search is to be checked.
    pub sampled: bool,
    started: Instant,
}

/// One field's recall-targeted search state ([module doc](self)): shared
/// by every search of the field, from any thread.
#[derive(Debug)]
pub struct RecallTuner {
    target: RecallTarget,
    level: AtomicUsize,
    queries: AtomicU64,
    /// The query count at which the next sample is due; `u64::MAX` while a
    /// claimed sample is being checked.
    next_sample: AtomicU64,
    search_nanos: AtomicU64,
    check_nanos: AtomicU64,
    state: Mutex<State>,
}

#[derive(Debug)]
struct State {
    levels: [Level; LEVELS],
    samples: u64,
}

fn nanos(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

impl RecallTuner {
    /// A tuner starting at [`BASE_LEVEL`].
    pub fn new(target: RecallTarget) -> Self {
        Self {
            target,
            level: AtomicUsize::new(BASE_LEVEL),
            queries: AtomicU64::new(0),
            next_sample: AtomicU64::new(0),
            search_nanos: AtomicU64::new(0),
            check_nanos: AtomicU64::new(0),
            state: Mutex::new(State {
                levels: [Level::default(); LEVELS],
                samples: 0,
            }),
        }
    }

    pub fn target(&self) -> RecallTarget {
        self.target
    }

    /// The level searches run at now.
    pub fn level(&self) -> usize {
        self.level.load(Ordering::Relaxed)
    }

    pub fn stats(&self) -> RecallStats {
        let level = self.level();
        let state = self.lock();
        RecallStats {
            target: self.target,
            level,
            effort: effort(level),
            recall: state.levels[level].mean(),
            recall_lower_bound: state.levels[level].lower_bound(),
            queries: self.queries.load(Ordering::Relaxed),
            samples: state.samples,
            search_nanos: self.search_nanos.load(Ordering::Relaxed),
            check_nanos: self.check_nanos.load(Ordering::Relaxed),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        // A panic while holding the lock leaves averages that are still
        // numbers: keep using them.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Starts a tuned search: the effort to search at, and whether the
    /// search is sampled. Pass the result to [`Self::searched`] once the
    /// search is done, and a sampled one to [`check_sample`].
    pub fn begin(&self) -> TunedSearch {
        let started = Instant::now();
        let n = self.queries.fetch_add(1, Ordering::Relaxed);
        let level = self.level();
        let due = self.next_sample.load(Ordering::Relaxed);
        let sampled = n >= due
            && self.within_budget()
            // Claimed until the check records and sets the next one.
            && self
                .next_sample
                .compare_exchange(due, u64::MAX, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok();
        TunedSearch {
            level,
            effort: effort(level),
            sampled,
            started,
        }
    }

    /// Records a search started with [`Self::begin`] as done.
    pub fn searched(&self, search: &TunedSearch) {
        self.search_nanos
            .fetch_add(nanos(search.started), Ordering::Relaxed);
    }

    /// Whether the checks so far are within the budget ([`WARMUP_BUDGET`]
    /// while warming up, when it is larger).
    fn within_budget(&self) -> bool {
        let checks = self.check_nanos.load(Ordering::Relaxed) as f64;
        let searches = self.search_nanos.load(Ordering::Relaxed) as f64;
        let budget = if self.warming_up() {
            self.target.budget.max(WARMUP_BUDGET)
        } else {
            self.target.budget
        };
        checks <= f64::from(budget) * searches
    }

    fn warming_up(&self) -> bool {
        self.lock().samples < WARMUP_SAMPLES
    }

    /// Gives back a sample claimed by [`Self::begin`] that was not checked:
    /// the next search is due.
    pub fn release(&self) {
        let _ = self.next_sample.compare_exchange(
            u64::MAX,
            self.queries.load(Ordering::Relaxed),
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }

    /// Records one sample: `recalls` measured at their levels around
    /// `level`, the level the query searched at. Moves the level.
    // ARITH: levels are below `LEVELS`; the averages are finite.
    #[allow(clippy::arithmetic_side_effects)]
    fn record(&self, level: usize, recalls: &[(usize, f64)]) {
        let mut state = self.lock();
        for l in &mut state.levels {
            l.recall *= DECAY;
            l.squares *= DECAY;
            l.weight *= DECAY;
        }
        for &(l, r) in recalls {
            if let Some(stats) = state.levels.get_mut(l) {
                stats.recall += r;
                stats.squares += r * r;
                stats.weight += 1.0;
            }
        }
        state.samples += 1;
        let every = self.target.every();
        let gap = if state.samples < WARMUP_SAMPLES {
            every.min(10)
        } else {
            every
        };
        let queries = self.queries.load(Ordering::Relaxed);
        self.next_sample
            .store(queries.saturating_add(gap - 1), Ordering::Relaxed);
        // Another sample may have moved the level since this query read it:
        // only the level this one measured around is moved from.
        if self.level() != level {
            return;
        }
        let target = f64::from(self.target.recall);
        let bound = |l: usize| state.levels[l].lower_bound();
        let below = |l: usize| bound(l).is_some_and(|b| b < target);
        let reaches = |l: usize| bound(l).is_some_and(|b| b >= target);
        let to = if below(level) && level + 1 < LEVELS {
            level + 1
        } else if level > 0 && reaches(level - 1) {
            level - 1
        } else {
            level
        };
        self.level.store(to, Ordering::Relaxed);
    }
}

/// The share of `exact`'s first `k` hits that `found` holds, `1.0` when
/// there are none.
fn recall_at(found: &[ScoreDoc], exact: &[ScoreDoc], k: usize) -> f64 {
    let truth = &exact[..exact.len().min(k)];
    if truth.is_empty() {
        return 1.0;
    }
    let mut ids: Vec<i32> = truth.iter().map(|h| h.doc_id).collect();
    ids.sort_unstable();
    let hits = found
        .iter()
        .take(k)
        .filter(|h| ids.binary_search(&h.doc_id).is_ok())
        .count();
    hits as f64 / truth.len() as f64
}

/// [`crate::knn_search::knn_top_hits_buffered`] (or, with `nested`,
/// [`crate::knn_search::knn_top_hits_nested_with`]) at the depth `tuner`
/// has tuned for its target recall, a sampled query checked before it
/// returns ([module doc](self)).
///
/// # Errors
/// As [`crate::knn_search::knn_top_hits`].
#[allow(clippy::too_many_arguments)]
pub fn knn_top_hits_tuned(
    segments: &[OpenSegment<'_>],
    field: &str,
    target: &KnnTarget,
    k: usize,
    candidates: usize,
    filter: Option<&Clause>,
    nested: Option<(&dyn BitSetProducer, bool)>,
    ivf_nprobes: usize,
    buffers: &[BufferedVectors<'_>],
    tuner: &RecallTuner,
) -> Result<Vec<ScoreDoc>> {
    let search = tuner.begin();
    let hits = knn_top_hits_at_effort(
        segments,
        field,
        target,
        k,
        candidates,
        filter,
        nested,
        ivf_nprobes,
        buffers,
        None,
        search.effort,
    );
    tuner.searched(&search);
    let hits = match hits {
        Ok(hits) => hits,
        Err(e) => {
            if search.sampled {
                tuner.release();
            }
            return Err(e);
        }
    };
    check_sample(
        segments,
        field,
        target,
        k,
        candidates,
        filter,
        nested,
        ivf_nprobes,
        buffers,
        None,
        tuner,
        &search,
        &hits,
    )?;
    Ok(hits)
}

/// Checks a sampled search ([`TunedSearch::sampled`]) that returned `hits`
/// against an exact one over the same `segments` and arguments, searches
/// again at the [`NEIGHBOURS`] levels on either side, and records the recalls in `tuner`.
/// An unsampled search is not checked. Neither is a search over `buffers`
/// nor an expanded nested one: there is no exact search to compare them
/// with here, and their sample is given back. The levels searched again
/// keep the search's `exploration_slack`.
///
/// # Errors
/// As [`crate::knn_search::knn_top_hits`].
#[allow(clippy::too_many_arguments)]
pub fn check_sample(
    segments: &[OpenSegment<'_>],
    field: &str,
    target: &KnnTarget,
    k: usize,
    candidates: usize,
    filter: Option<&Clause>,
    nested: Option<(&dyn BitSetProducer, bool)>,
    ivf_nprobes: usize,
    buffers: &[BufferedVectors<'_>],
    exploration_slack: Option<f32>,
    tuner: &RecallTuner,
    search: &TunedSearch,
    hits: &[ScoreDoc],
) -> Result<()> {
    if !search.sampled {
        return Ok(());
    }
    if !buffers.is_empty() || matches!(nested, Some((_, true))) {
        tuner.release();
        return Ok(());
    }
    let started = Instant::now();
    let run = |depth: Depth| {
        top_hits(
            segments,
            field,
            target,
            k,
            candidates,
            filter,
            nested,
            ivf_nprobes,
            buffers,
            depth,
        )
    };
    let checked = (|| -> Result<Vec<(usize, f64)>> {
        let exact = run(Depth {
            effort: 1.0,
            exact: true,
            exploration_slack: None,
        })?;
        let at = |level: usize| {
            run(Depth {
                effort: effort(level),
                exact: false,
                exploration_slack,
            })
        };
        let level = search.level;
        let mut recalls = Vec::with_capacity(2 * NEIGHBOURS + 1); // alloc-ok: a few levels
        recalls.push((level, recall_at(hits, &exact, k)));
        let lowest = level.saturating_sub(NEIGHBOURS);
        let highest = level.saturating_add(NEIGHBOURS).min(LEVELS - 1);
        for l in (lowest..=highest).filter(|&l| l != level) {
            recalls.push((l, recall_at(&at(l)?, &exact, k)));
        }
        Ok(recalls)
    })();
    tuner
        .check_nanos
        .fetch_add(nanos(started), Ordering::Relaxed);
    match checked {
        Ok(recalls) => {
            tuner.record(search.level, &recalls);
            Ok(())
        }
        Err(e) => {
            tuner.release();
            Err(e)
        }
    }
}

/// [`crate::knn_search::knn_top_hits_buffered`] (or, with `nested`,
/// [`crate::knn_search::knn_top_hits_nested_with`]) at a fixed search
/// effort: what a tuned search runs at its level ([module doc](self)),
/// with the graph walks' optional `exploration_slack`.
///
/// # Errors
/// As [`crate::knn_search::knn_top_hits`].
#[allow(clippy::too_many_arguments)]
pub fn knn_top_hits_at_effort(
    segments: &[OpenSegment<'_>],
    field: &str,
    target: &KnnTarget,
    k: usize,
    candidates: usize,
    filter: Option<&Clause>,
    nested: Option<(&dyn BitSetProducer, bool)>,
    ivf_nprobes: usize,
    buffers: &[BufferedVectors<'_>],
    exploration_slack: Option<f32>,
    effort: f32,
) -> Result<Vec<ScoreDoc>> {
    top_hits(
        segments,
        field,
        target,
        k,
        candidates,
        filter,
        nested,
        ivf_nprobes,
        buffers,
        Depth {
            effort,
            exact: false,
            exploration_slack,
        },
    )
}

/// The exact top `k` of [`crate::knn_search::knn_top_hits`] (or, with
/// `nested`, [`crate::knn_search::knn_top_hits_nested_with`]): every
/// segment scanned with its vector index unused, what [`check_sample`]
/// measures recall against.
///
/// # Errors
/// As [`crate::knn_search::knn_top_hits`].
pub fn knn_top_hits_exact(
    segments: &[OpenSegment<'_>],
    field: &str,
    target: &KnnTarget,
    k: usize,
    filter: Option<&Clause>,
    nested: Option<(&dyn BitSetProducer, bool)>,
) -> Result<Vec<ScoreDoc>> {
    top_hits(
        segments,
        field,
        target,
        k,
        k,
        filter,
        nested,
        0,
        &[],
        Depth {
            effort: 1.0,
            exact: true,
            exploration_slack: None,
        },
    )
}

#[cfg(test)]
mod tests;
