//! `LuceneTestCase`'s writer configuration: `newIndexWriterConfig`,
//! `newMergePolicy` and its siblings (`newLogMergePolicy`,
//! `newTieredMergePolicy`, `newAlcoholicMergePolicy`, `configureRandom`) and
//! `maybeChangeLiveIndexWriterConfig`.
//!
//! Java's helpers draw from two streams: the `Random r` they are handed and,
//! here and there, the test's own `random()`. [`Draws`] carries both; built
//! with [`Draws::new`] they are one stream (Java's `r == random()`, what
//! `newIndexWriterConfig()` passes), with [`Draws::with_context`] two, so a
//! caller holding Java's `r` gets every value Java draws from it.
//!
//! Differs:
//!
//! - The writer is configured after it is opened ([`RandomIndexWriterConfig::apply`]):
//!   this port's `IndexWriter` takes its settings by setter, not from an
//!   `IndexWriterConfig`.
//! - The merge scheduler is drawn ([`MergeSchedulerChoice`]) but only a
//!   `ConcurrentIndexWriter` takes one ([`MergeSchedulerChoice::build`]);
//!   the single-threaded writer merges in line. The always-stalling
//!   `maybeStall` override has no seam and is only recorded.
//! - `setMergedSegmentWarmer(SimpleMergedSegmentWarmer)`,
//!   `setIndexWriterEventListener` and `setCheckPendingFlushUpdate` are
//!   drawn and recorded: the warmer and the listener are `not-needed` in
//!   lucene-core, and the single-threaded writer has no pending-flush check.
//! - `classEnvRule.similarity` (`RandomSimilarity`) is not installed: the
//!   writer keeps its default similarity.
//! - The merge policy is a [`LiveMergePolicy`] the writer and the caller
//!   share, so a live change reaches the policy the writer is using, as
//!   Java's `LiveIndexWriterConfig` does.

use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard};

use lucene_analysis::Analyzer;
use lucene_index::index_writer::{
    IndexWriter, DEFAULT_RAM_BUFFER_SIZE_MB, DISABLE_AUTO_FLUSH, DISABLE_AUTO_FLUSH_MB,
};
use lucene_index::merge_policy::api::{
    CompoundFileSettings, MergeContext, MergePolicy, MergeSegment, MergeSpecification,
    MergeTrigger, NoMergePolicy, Result, TieredMergePolicy,
};
use lucene_index::merge_policy::log::{LogByteSizeMergePolicy, LogDocMergePolicy, LogMergePolicy};
use lucene_index::merge_scheduler::{
    ConcurrentMergeScheduler, MergeScheduler, SerialMergeScheduler,
};
use lucene_util::java_random::JavaRandom;
use std::collections::HashMap;

use super::merge_policies::{AlcoholicMergePolicy, MockRandomMergePolicy};
use crate::util::lucene_test_case;
use crate::util::test_util;

/// Java's two streams: the `Random r` a helper is handed, and the test's
/// own `random()` (the "context").
pub struct Draws<'a> {
    r: &'a mut JavaRandom,
    ctx: Option<&'a mut JavaRandom>,
    nightly: bool,
    multiplier: i32,
}

impl<'a> Draws<'a> {
    /// `r` is also `random()`; the knobs are the environment's.
    pub fn new(r: &'a mut JavaRandom) -> Self {
        let (nightly, multiplier) = lucene_test_case::knobs();
        Self {
            r,
            ctx: None,
            nightly,
            multiplier,
        }
    }

    /// `r`, with `ctx` as `random()`.
    pub fn with_context(r: &'a mut JavaRandom, ctx: &'a mut JavaRandom) -> Self {
        Self {
            ctx: Some(ctx),
            ..Self::new(r)
        }
    }

    /// `TEST_NIGHTLY` and `RANDOM_MULTIPLIER` for this caller.
    pub fn knobs(mut self, nightly: bool, multiplier: i32) -> Self {
        self.nightly = nightly;
        self.multiplier = multiplier;
        self
    }

    /// `r`.
    pub fn r(&mut self) -> &mut JavaRandom {
        self.r
    }

    /// `random()`.
    pub fn ctx(&mut self) -> &mut JavaRandom {
        match self.ctx.as_deref_mut() {
            Some(c) => c,
            None => self.r,
        }
    }

    /// `rarely(r)`.
    pub fn rarely(&mut self) -> bool {
        lucene_test_case::rarely(self.r, self.nightly, self.multiplier)
    }

    /// `rarely()`, on `random()`.
    pub fn ctx_rarely(&mut self) -> bool {
        let (nightly, multiplier) = (self.nightly, self.multiplier);
        lucene_test_case::rarely(self.ctx(), nightly, multiplier)
    }

    /// `atLeast(r, i)`.
    pub fn at_least(&mut self, i: i32) -> i32 {
        lucene_test_case::at_least(self.r, self.multiplier, i)
    }

    /// `TestUtil.nextInt(r, start, end)`.
    pub fn next_int(&mut self, start: i32, end: i32) -> i32 {
        test_util::next_int(self.r, start, end)
    }
}

/// `(long) (mb * 1024 * 1024)`, saturating at the top as Java's setters do.
fn mb_to_bytes(mb: f64) -> u64 {
    (mb * 1024.0 * 1024.0) as u64
}

/// The policies `newMergePolicy` chooses from, as one type: what a live
/// change (`instanceof LogMergePolicy` / `TieredMergePolicy`) inspects.
#[derive(Debug)]
pub enum PolicyKind {
    /// `newLogMergePolicy` (a `LogDocMergePolicy` or a
    /// `LogByteSizeMergePolicy`) or `newAlcoholicMergePolicy` (a
    /// `LogMergePolicy` subclass in Java too).
    Log(LogMergePolicy),
    /// `newTieredMergePolicy`.
    Tiered(TieredMergePolicy),
    /// `new MockRandomMergePolicy(r)`.
    MockRandom(MockRandomMergePolicy),
    /// `NoMergePolicy.INSTANCE`.
    NoMerge(NoMergePolicy),
    /// Any other policy a test installs.
    Other(Box<dyn MergePolicy>),
}

impl PolicyKind {
    /// The policy itself.
    pub fn policy(&self) -> &dyn MergePolicy {
        match self {
            PolicyKind::Log(p) => p,
            PolicyKind::Tiered(p) => p,
            PolicyKind::MockRandom(p) => p,
            PolicyKind::NoMerge(p) => p,
            PolicyKind::Other(p) => p.as_ref(),
        }
    }

    /// The policy itself, to change.
    pub fn policy_mut(&mut self) -> &mut dyn MergePolicy {
        match self {
            PolicyKind::Log(p) => p,
            PolicyKind::Tiered(p) => p,
            PolicyKind::MockRandom(p) => p,
            PolicyKind::NoMerge(p) => p,
            PolicyKind::Other(p) => p.as_mut(),
        }
    }
}

macro_rules! delegate_merge_policy {
    ($t:ty, |$s:ident| $inner:expr, |$m:ident| $inner_mut:expr) => {
        impl MergePolicy for $t {
            fn find_merges(
                &self,
                trigger: MergeTrigger,
                infos: &[MergeSegment],
                ctx: &dyn MergeContext,
            ) -> Result<Option<MergeSpecification>> {
                let $s = self;
                $inner.find_merges(trigger, infos, ctx)
            }
            fn find_forced_merges(
                &self,
                infos: &[MergeSegment],
                max_segment_count: i32,
                segments_to_merge: &HashMap<String, bool>,
                ctx: &dyn MergeContext,
            ) -> Result<Option<MergeSpecification>> {
                let $s = self;
                $inner.find_forced_merges(infos, max_segment_count, segments_to_merge, ctx)
            }
            fn find_forced_deletes_merges(
                &self,
                infos: &[MergeSegment],
                ctx: &dyn MergeContext,
            ) -> Result<Option<MergeSpecification>> {
                let $s = self;
                $inner.find_forced_deletes_merges(infos, ctx)
            }
            fn find_merges_for_readers(&self, max_docs: &[i64]) -> Option<MergeSpecification> {
                let $s = self;
                $inner.find_merges_for_readers(max_docs)
            }
            fn find_full_flush_merges(
                &self,
                trigger: MergeTrigger,
                infos: &[MergeSegment],
                ctx: &dyn MergeContext,
            ) -> Result<Option<MergeSpecification>> {
                let $s = self;
                $inner.find_full_flush_merges(trigger, infos, ctx)
            }
            fn use_compound_file(
                &self,
                infos: &[MergeSegment],
                merged: &MergeSegment,
                ctx: &dyn MergeContext,
            ) -> Result<bool> {
                let $s = self;
                $inner.use_compound_file(infos, merged, ctx)
            }
            fn size(&self, info: &MergeSegment, ctx: &dyn MergeContext) -> Result<i64> {
                let $s = self;
                $inner.size(info, ctx)
            }
            fn max_full_flush_merge_size(&self) -> i64 {
                let $s = self;
                $inner.max_full_flush_merge_size()
            }
            fn keep_fully_deleted_segment(&self) -> bool {
                let $s = self;
                $inner.keep_fully_deleted_segment()
            }
            fn num_deletes_to_merge(&self, info: &MergeSegment, del_count: i32) -> i32 {
                let $s = self;
                $inner.num_deletes_to_merge(info, del_count)
            }
            fn compound_file_settings(&self) -> CompoundFileSettings {
                let $s = self;
                $inner.compound_file_settings()
            }
            fn compound_file_settings_mut(&mut self) -> &mut CompoundFileSettings {
                let $m = self;
                $inner_mut.compound_file_settings_mut()
            }
            fn is_merged(
                &self,
                infos: &[MergeSegment],
                info: &MergeSegment,
                ctx: &dyn MergeContext,
            ) -> Result<bool> {
                let $s = self;
                $inner.is_merged(infos, info, ctx)
            }
        }
    };
}

delegate_merge_policy!(PolicyKind, |s| s.policy(), |m| m.policy_mut());

/// The merge policy a writer and its test share: Java's
/// `LiveIndexWriterConfig.getMergePolicy()`, which the test may change
/// while the writer runs.
#[derive(Debug, Clone)]
pub struct LiveMergePolicy(Arc<RwLock<PolicyKind>>);

impl LiveMergePolicy {
    /// A policy to share.
    pub fn new(kind: PolicyKind) -> Self {
        Self(Arc::new(RwLock::new(kind)))
    }

    /// The policy as it is now.
    pub fn kind(&self) -> RwLockReadGuard<'_, PolicyKind> {
        self.0.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Changes the policy in place, for the writer's next decision.
    pub fn change<R>(&self, f: impl FnOnce(&mut PolicyKind) -> R) -> R {
        f(&mut self.0.write().unwrap_or_else(PoisonError::into_inner))
    }
}

delegate_merge_policy!(LiveMergePolicy, |s| s.kind(), |m| {
    // Only reachable while nothing shares the policy yet: a shared one is
    // changed through `change`.
    Arc::get_mut(&mut m.0)
        .expect("a LiveMergePolicy shared with a writer is changed through LiveMergePolicy::change")
        .get_mut()
        .unwrap_or_else(PoisonError::into_inner)
});

/// `configureRandom(r, mergePolicy)`: a random `noCFSRatio` and, rarely, a
/// small `maxCFSSegmentSizeMB`.
pub fn configure_random(d: &mut Draws<'_>, policy: &mut dyn MergePolicy) {
    let settings = policy.compound_file_settings_mut();
    let ratio = if d.r().next_boolean() {
        0.1 + d.r().next_double() * 0.8
    } else if d.r().next_boolean() {
        1.0
    } else {
        0.0
    };
    settings.set_no_cfs_ratio(ratio).expect("a ratio in 0..=1");
    let max_cfs = if d.rarely() {
        0.2 + d.r().next_double() * 2.0
    } else {
        f64::INFINITY
    };
    settings
        .set_max_cfs_segment_size_mb(max_cfs)
        .expect("a positive size");
}

/// `newLogMergePolicy(r)`: a `LogDocMergePolicy` or a
/// `LogByteSizeMergePolicy` with random settings.
pub fn new_log_merge_policy(d: &mut Draws<'_>) -> LogMergePolicy {
    let mut logmp: LogMergePolicy = if d.r().next_boolean() {
        (*LogDocMergePolicy::new()).clone()
    } else {
        (*LogByteSizeMergePolicy::new()).clone()
    };
    logmp.calibrate_size_by_deletes = d.r().next_boolean();
    let concurrency = test_util::next_int(d.ctx(), 1, 16);
    logmp
        .set_target_search_concurrency(concurrency)
        .expect("a concurrency of at least 1");
    let factor = if d.rarely() {
        d.next_int(2, 9)
    } else {
        d.next_int(10, 50)
    };
    logmp
        .set_merge_factor(factor)
        .expect("a factor of at least 2");
    configure_random(d, &mut logmp);
    logmp
}

/// `newTieredMergePolicy(r)`: a `TieredMergePolicy` with random settings.
pub fn new_tiered_merge_policy(d: &mut Draws<'_>) -> TieredMergePolicy {
    let mut tmp = TieredMergePolicy::default();
    tmp.config.max_merge_at_once = if d.rarely() {
        d.next_int(2, 9)
    } else {
        d.next_int(10, 50)
    } as usize;
    let max_merged_mb = if d.rarely() {
        0.2 + d.r().next_double() * 2.0
    } else {
        10.0 + d.r().next_double() * 100.0
    };
    tmp.config.max_merged_segment_size = mb_to_bytes(max_merged_mb);
    tmp.config.floor_segment_size = mb_to_bytes(0.2 + d.r().next_double() * 2.0);
    tmp.config.force_merge_deletes_pct_allowed = d.r().next_double() * 30.0;
    tmp.config.segments_per_tier = if d.rarely() {
        d.next_int(2, 20)
    } else {
        d.next_int(10, 50)
    } as usize;
    tmp.config.target_search_concurrency = if d.rarely() {
        d.next_int(10, 50)
    } else {
        d.next_int(2, 20)
    } as usize;
    configure_random(d, &mut tmp);
    tmp.config.deletes_pct_allowed = 20.0 + d.ctx().next_double() * 30.0;
    tmp
}

/// `newAlcoholicMergePolicy(r, tz)`, with the zone as its offset from UTC in
/// milliseconds: the policy gets `new Random(r.nextLong())`.
pub fn new_alcoholic_merge_policy(d: &mut Draws<'_>, utc_offset_millis: i64) -> LogMergePolicy {
    let own = JavaRandom::new(d.r().next_long());
    AlcoholicMergePolicy::new(utc_offset_millis, own)
}

/// `newMergePolicy(r, includeMockMP)`: rarely a `MockRandomMergePolicy`,
/// else half the time a tiered policy, rarely an alcoholic one, otherwise a
/// log policy.
pub fn new_merge_policy(
    d: &mut Draws<'_>,
    include_mock_mp: bool,
    utc_offset_millis: i64,
) -> PolicyKind {
    if include_mock_mp && d.rarely() {
        PolicyKind::MockRandom(MockRandomMergePolicy::new(d.r()))
    } else if d.r().next_boolean() {
        PolicyKind::Tiered(new_tiered_merge_policy(d))
    } else if d.rarely() {
        PolicyKind::Log(new_alcoholic_merge_policy(d, utc_offset_millis))
    } else {
        PolicyKind::Log(new_log_merge_policy(d))
    }
}

/// The merge scheduler `newIndexWriterConfig` chose.
#[derive(Debug, Clone, PartialEq)]
pub enum MergeSchedulerChoice {
    /// `new SerialMergeScheduler()` (rarely).
    Serial,
    /// The second `rarely`: a test CMS (`always_stall`: the one whose
    /// `maybeStall` always returns true) with random limits.
    Concurrent {
        always_stall: bool,
        max_merge_count: i32,
        max_thread_count: i32,
        auto_io_throttle: bool,
        force_merge_mb_per_sec: f64,
    },
    /// The usual case: a CMS (`test`: Java's `TestConcurrentMergeScheduler`)
    /// running one merge at a time with up to three pending.
    Default { test: bool },
}

impl MergeSchedulerChoice {
    /// The scheduler, for a writer that takes one.
    pub fn build(&self) -> Arc<dyn MergeScheduler> {
        match self {
            MergeSchedulerChoice::Serial => Arc::new(SerialMergeScheduler::new()),
            MergeSchedulerChoice::Concurrent {
                max_merge_count,
                max_thread_count,
                auto_io_throttle,
                force_merge_mb_per_sec,
                ..
            } => {
                let cms = ConcurrentMergeScheduler::new();
                cms.set_max_merges_and_threads(*max_merge_count, *max_thread_count)
                    .expect("Java's own limits");
                if !auto_io_throttle {
                    cms.disable_auto_io_throttle();
                }
                cms.set_force_merge_mb_per_sec(*force_merge_mb_per_sec);
                Arc::new(cms)
            }
            MergeSchedulerChoice::Default { .. } => {
                let cms = ConcurrentMergeScheduler::new();
                cms.set_max_merges_and_threads(3, 1)
                    .expect("Java's own limits");
                Arc::new(cms)
            }
        }
    }
}

/// `LuceneTestCase.LiveIWCFlushMode`: which flush trigger a live change
/// may switch to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveFlushMode {
    /// `BY_RAM`.
    ByRam,
    /// `BY_DOCS`.
    ByDocs,
    /// `EITHER`.
    Either,
}

impl LiveFlushMode {
    /// `TestRuleSetupAndRestoreClassEnv`'s draw: `random().nextInt(3)`.
    pub fn random(random: &mut JavaRandom) -> Self {
        match random.next_int_bounded(3) {
            0 => LiveFlushMode::ByRam,
            1 => LiveFlushMode::ByDocs,
            _ => LiveFlushMode::Either,
        }
    }
}

/// `IndexWriterConfig` as `newIndexWriterConfig(r, a)` leaves it.
pub struct RandomIndexWriterConfig {
    /// The analyzer.
    pub analyzer: Arc<Analyzer>,
    /// `setMergeScheduler`.
    pub merge_scheduler: MergeSchedulerChoice,
    /// `setMaxBufferedDocs`, when drawn (else the default, disabled).
    pub max_buffered_docs: Option<i32>,
    /// `setRAMBufferSizeMB`: the default until a live change sets it
    /// ([`DISABLE_AUTO_FLUSH_MB`] when flushing by document count).
    pub ram_buffer_size_mb: f64,
    /// `setMergePolicy(newMergePolicy(r))`.
    pub merge_policy: LiveMergePolicy,
    /// `setMergedSegmentWarmer(new SimpleMergedSegmentWarmer(...))`, rarely.
    pub merged_segment_warmer: bool,
    /// `setUseCompoundFile`.
    pub use_compound_file: bool,
    /// `setReaderPooling`.
    pub reader_pooling: bool,
    /// `setCheckPendingFlushUpdate` (true unless rarely turned off).
    pub check_pending_flush_update: bool,
    /// `setIndexWriterEventListener(new MockIndexWriterEventListener())`,
    /// rarely.
    pub event_listener: bool,
    /// `setMaxFullFlushMergeWaitMillis`, the last of its two draws.
    pub max_full_flush_merge_wait_millis: i64,
    /// `setCommitOnClose` (Java's default, `true`).
    pub commit_on_close: bool,
}

/// `newIndexWriterConfig(r, a)`: a configuration with random merge
/// scheduler, buffering, merge policy and compound-file settings.
pub fn new_index_writer_config(
    d: &mut Draws<'_>,
    analyzer: Arc<Analyzer>,
    utc_offset_millis: i64,
) -> RandomIndexWriterConfig {
    let merge_scheduler = if d.rarely() {
        MergeSchedulerChoice::Serial
    } else if d.rarely() {
        let always_stall = !d.r().next_boolean();
        let max_thread_count = d.next_int(1, 4);
        let max_merge_count = d.next_int(max_thread_count, max_thread_count + 4);
        let auto_io_throttle = !d.ctx().next_boolean();
        let force_merge_mb_per_sec = 10.0 + 10.0 * d.ctx().next_double();
        MergeSchedulerChoice::Concurrent {
            always_stall,
            max_merge_count,
            max_thread_count,
            auto_io_throttle,
            force_merge_mb_per_sec,
        }
    } else {
        MergeSchedulerChoice::Default {
            test: d.ctx().next_boolean(),
        }
    };
    let max_buffered_docs = if d.r().next_boolean() {
        Some(if d.rarely() {
            // crazy value
            d.next_int(2, 15)
        } else {
            // reasonable value
            d.next_int(16, 1000)
        })
    } else {
        None
    };
    let merge_policy = LiveMergePolicy::new(new_merge_policy(d, true, utc_offset_millis));
    let merged_segment_warmer = d.rarely();
    let use_compound_file = d.r().next_boolean();
    let reader_pooling = d.r().next_boolean();
    let check_pending_flush_update = !d.rarely();
    let event_listener = d.rarely();
    // Java sets it twice; the first draw is overwritten.
    let _ = d.r().next_int_bounded(3);
    let max_full_flush_merge_wait_millis = if d.ctx_rarely() {
        d.at_least(1000)
    } else {
        d.at_least(200)
    };
    RandomIndexWriterConfig {
        analyzer,
        merge_scheduler,
        max_buffered_docs,
        ram_buffer_size_mb: DEFAULT_RAM_BUFFER_SIZE_MB,
        merge_policy,
        merged_segment_warmer,
        use_compound_file,
        reader_pooling,
        check_pending_flush_update,
        event_listener,
        max_full_flush_merge_wait_millis: i64::from(max_full_flush_merge_wait_millis),
        commit_on_close: true,
    }
}

impl RandomIndexWriterConfig {
    /// Sets this configuration on `w`, as `new IndexWriter(dir, conf)` reads
    /// it.
    ///
    /// # Errors
    /// What the writer's setters refuse.
    pub fn apply(&self, w: &mut IndexWriter<'_>) -> lucene_index::index_writer::Result<()> {
        w.set_analyzer(Some(Arc::clone(&self.analyzer)));
        // In the order the setters allow: the enabled trigger first.
        if let Some(docs) = self.max_buffered_docs {
            w.set_max_buffered_docs(docs)?;
            w.set_ram_buffer_size_mb(self.ram_buffer_size_mb)?;
        } else {
            w.set_ram_buffer_size_mb(self.ram_buffer_size_mb)?;
            w.set_max_buffered_docs(DISABLE_AUTO_FLUSH)?;
        }
        w.set_pluggable_merge_policy(Some(Arc::new(self.merge_policy.clone())));
        w.set_use_compound_file(self.use_compound_file);
        w.set_reader_pooling(self.reader_pooling);
        w.set_max_full_flush_merge_wait_millis(self.max_full_flush_merge_wait_millis);
        w.set_commit_on_close(self.commit_on_close);
        Ok(())
    }

    /// Whether the policy is `NoMergePolicy`.
    pub fn is_no_merge_policy(&self) -> bool {
        matches!(*self.merge_policy.kind(), PolicyKind::NoMerge(_))
    }

    /// Whether the policy is a `TieredMergePolicy`.
    pub fn is_tiered_merge_policy(&self) -> bool {
        matches!(*self.merge_policy.kind(), PolicyKind::Tiered(_))
    }
}

/// `maybeChangeLiveIndexWriterConfig(r, c)`: now and then, new flush
/// triggers, warmer, compound-file setting, scheduler limits or merge
/// policy settings, applied to `w` (whose configuration `c` is).
///
/// # Errors
/// What the writer's setters refuse.
pub fn maybe_change_live_index_writer_config(
    d: &mut Draws<'_>,
    flush_mode: LiveFlushMode,
    c: &mut RandomIndexWriterConfig,
    w: &mut IndexWriter<'_>,
) -> lucene_index::index_writer::Result<()> {
    if d.rarely() {
        // change flush parameters, in the order the setters allow
        let flush_by_ram = match flush_mode {
            LiveFlushMode::ByRam => true,
            LiveFlushMode::ByDocs => false,
            LiveFlushMode::Either => d.r().next_boolean(),
        };
        if flush_by_ram {
            c.ram_buffer_size_mb = f64::from(d.next_int(1, 10));
            w.set_ram_buffer_size_mb(c.ram_buffer_size_mb)?;
            w.set_max_buffered_docs(DISABLE_AUTO_FLUSH)?;
            c.max_buffered_docs = None;
        } else {
            let docs = if d.rarely() {
                // crazy value
                d.next_int(2, 15)
            } else {
                // reasonable value
                d.next_int(16, 1000)
            };
            w.set_max_buffered_docs(docs)?;
            w.set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB)?;
            c.max_buffered_docs = Some(docs);
            c.ram_buffer_size_mb = DISABLE_AUTO_FLUSH_MB;
        }
    }
    if d.rarely() {
        // change warmer parameters (the warmer is always none or simple)
        c.merged_segment_warmer = d.r().next_boolean();
    }
    if d.rarely() {
        // change CFS flush parameters
        c.use_compound_file = d.r().next_boolean();
        w.set_use_compound_file(c.use_compound_file);
    }
    if d.rarely() {
        // change CMS merge parameters
        if !matches!(c.merge_scheduler, MergeSchedulerChoice::Serial) {
            let max_thread_count = d.next_int(1, 4);
            let max_merge_count = d.next_int(max_thread_count, max_thread_count + 4);
            let auto_io_throttle = d.ctx().next_boolean();
            let always_stall = matches!(
                c.merge_scheduler,
                MergeSchedulerChoice::Concurrent {
                    always_stall: true,
                    ..
                }
            );
            let force_merge_mb_per_sec = match c.merge_scheduler {
                MergeSchedulerChoice::Concurrent {
                    force_merge_mb_per_sec,
                    ..
                } => force_merge_mb_per_sec,
                _ => f64::INFINITY,
            };
            c.merge_scheduler = MergeSchedulerChoice::Concurrent {
                always_stall,
                max_merge_count,
                max_thread_count,
                auto_io_throttle,
                force_merge_mb_per_sec,
            };
        }
    }
    if d.rarely() {
        c.merge_policy.change(|kind| {
            configure_random(d, kind.policy_mut());
            match kind {
                PolicyKind::Log(logmp) => {
                    logmp.calibrate_size_by_deletes = d.r().next_boolean();
                    let factor = if d.rarely() {
                        d.next_int(2, 9)
                    } else {
                        d.next_int(10, 50)
                    };
                    logmp
                        .set_merge_factor(factor)
                        .expect("a factor of at least 2");
                }
                PolicyKind::Tiered(tmp) => {
                    tmp.config.max_merge_at_once = if d.rarely() {
                        d.next_int(2, 9)
                    } else {
                        d.next_int(10, 50)
                    } as usize;
                    let max_merged_mb = if d.rarely() {
                        0.2 + d.r().next_double() * 2.0
                    } else {
                        d.r().next_double() * 100.0
                    };
                    tmp.config.max_merged_segment_size = mb_to_bytes(max_merged_mb);
                    tmp.config.floor_segment_size = mb_to_bytes(0.2 + d.r().next_double() * 2.0);
                    tmp.config.force_merge_deletes_pct_allowed = d.r().next_double() * 30.0;
                    tmp.config.segments_per_tier = if d.rarely() {
                        d.next_int(2, 20)
                    } else {
                        d.next_int(10, 50)
                    } as usize;
                    configure_random(d, tmp);
                    tmp.config.deletes_pct_allowed = 20.0 + d.ctx().next_double() * 30.0;
                }
                _ => {}
            }
        });
    }
    Ok(())
}
