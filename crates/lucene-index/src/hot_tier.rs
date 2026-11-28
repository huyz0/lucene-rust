//! A bounded write-behind tier for rows that keep being updated.
//!
//! An update of a document that was updated a moment ago is, today, a full
//! soft update: the new version is inverted and buffered, the older one stays
//! as a dead version until a merge drops it. When one row changes many times
//! between two refreshes only the last version is ever searched, so the others
//! cost indexing CPU, buffer memory, flush bytes and merge work for nothing.
//!
//! The tier holds the **latest operation** of each hot row and applies it when
//! the buffers are next cut (a flush, a commit, a snapshot, a live view) or
//! when a drain is asked for. An operation that a newer one replaces before
//! that is dropped without ever being inverted.
//!
//! - **Heat.** A row is hot once it is updated faster than
//!   [`HotTierConfig::min_updates_per_second`] (by default, more than once a second). Heat lives in a counting sketch
//!   (two rows of bytes, the smaller count counts), not per row. The sketch
//!   is cut into stripes that own no memory until an update reaches them,
//!   grow fourfold while their cells fill (up to an eighth of the budget in
//!   all) and give their cells back once every count has decayed to zero,
//!   so an idle shard holds nothing. A row
//!   updated once -- a bulk update of any size -- or a few times a second
//!   costs two byte increments and goes straight to the writer as before.
//! - **Memory.** [`HotTierConfig::max_bytes`] bounds the sketch plus every held
//!   operation, counted until the operation has been applied (an operation
//!   being spilled still counts). When a hot row does not fit, the least
//!   heated sampled rows are applied to make room, and if that does not make
//!   room either the new operation is written through. The tier never holds
//!   more than the budget; what it cannot hold goes through the normal path
//!   at the normal cost.
//! - **Order.** A shard's lock is held while the operations it holds are applied, so an
//!   older operation never lands after a newer one for the same row. A plain
//!   add of a row (an index right after a delete, which an engine adds rather
//!   than updates) applies that row's held operation first
//!   ([`HotTier::settle`]); a reader is not current while any operation is held.
//!
//! - **Errors.** `apply` failing (a document limit, an I/O error) fails the
//!   cut that drains, as a failing write fails; the failed operation is not
//!   retried. Callers treat a writer error as fatal, as the engine does.
//! - **Lock order.** `apply` runs under a shard lock, so it may take the writer's
//!   own locks; nothing holding a writer lock may take a shard lock.
//! - **Memory.** Held bytes are walked from the operation (values, field
//!   buffers, the map's slot and key copies) and checked against a counting
//!   allocator in `tests/hot_tier_memory.rs`: the estimate is 0 to 10% above
//!   what the allocator holds.
//! - **History.** An operation a newer one replaces leaves a [`HistoryStub`]
//!   (a no-op tombstone with its sequence number) when the caller configures
//!   one, so the index's history has no gap; without a stub the replaced
//!   operation is applied whole.
//!
//! An update is searchable at the next cut (refresh, flush, commit, live
//! view), as a refresh makes any update searchable; the versions that a newer
//! one replaced within a cut do not exist in the segments.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use lucene_codecs::stored_fields::{FieldValue, StoredField};

use crate::buffered_updates::{DocValuesUpdate, Term};
use crate::index_writer::{DocumentVector, ExplicitDocument};

const SHARDS: usize = 64;

/// The memory every tier of the process shares: a node's hot tiers together
/// stay within [`set_node_limit`], and each tier holding anything gets an equal
/// share of it (the limit over the tiers now holding memory), so many busy
/// shards cannot take more than one does and an idle shard takes nothing.
struct NodePool {
    /// `0`: no node-wide limit.
    limit: AtomicUsize,
    used: AtomicUsize,
    /// Tiers holding at least a byte.
    active: AtomicUsize,
}

static NODE: NodePool = NodePool {
    limit: AtomicUsize::new(0),
    used: AtomicUsize::new(0),
    active: AtomicUsize::new(0),
};

/// Caps the memory all hot tiers of the process hold together (`0` lifts the
/// cap), shared fairly: see [`HotTierConfig::max_bytes`] for the per-shard bound
/// that applies as well. A tier already over its new share keeps what it holds
/// until its next drain and admits nothing more.
pub fn set_node_limit(bytes: usize) {
    NODE.limit.store(bytes, Ordering::Release);
}

/// What every hot tier of the process holds now (sketches and held operations).
pub fn node_bytes() -> usize {
    NODE.used.load(Ordering::Acquire)
}
/// The heat sketch is cut into this many independently locked stripes.
const STRIPES: usize = 64;
/// Slots per row of a stripe's sketch when it is first needed (two byte rows
/// of this many cells: 512 bytes).
const STRIPE_START_SLOTS: usize = 256;
/// The most slots per row a stripe grows to (16 KiB a stripe, 1 MiB in all,
/// as 1M slots a row were before the sketch grew with the load).
const STRIPE_MAX_SLOTS: usize = 1 << 14;
/// How many offers a failed search for a row to spill silences the next ones.
const NO_SPILL_TICKS: u64 = 256;

/// What the tier is allowed to do.
#[derive(Debug, Clone)]
pub struct HotTierConfig {
    /// The hard bound on the tier's memory: the heat sketch and every held
    /// operation, counted from admission until the operation has been applied.
    pub max_bytes: usize,
    /// A row is held while it is updated more often than this many times a
    /// second ([`HotTierConfig::DEFAULT_MIN_UPDATES_PER_SECOND`]).
    pub min_updates_per_second: f64,
    /// Every period the sketch's counts are halved.
    pub age_every: Duration,
    /// How many held rows an admission samples for the one to spill.
    pub evict_sample: usize,
    /// What the caller writes in place of an operation the tier drops, so the
    /// sequence numbers it carried are not missing from the index's history.
    pub history: Option<HistoryStub>,
}

/// A small no-op document, built from a dropped operation, that keeps its
/// sequence number in the index (as a no-op tombstone does, hidden by the
/// soft-deletes field) so that history readers see no gap.
#[derive(Debug, Clone, Default)]
pub struct HistoryStub {
    /// Fields whose values are copied from the dropped operation's document
    /// (doc values and points): its sequence number and primary term.
    pub copy: Vec<String>,
    /// Doc-values fields every stub carries, with their value (the version,
    /// the tombstone marker).
    pub constants: Vec<(String, i64)>,
    /// A stored field every stub carries, as a no-op's reason.
    pub source: Option<(String, Vec<u8>)>,
}

impl HotTierConfig {
    /// A row updated more than once a second is hot.
    pub const DEFAULT_MIN_UPDATES_PER_SECOND: f64 = 1.0;
    /// The lowest rate that tells a hot row from a cold one: a counter of 2
    /// is the least that separates a second update from the first.
    pub const MIN_UPDATES_PER_SECOND: f64 = 0.5;
    /// The highest rate a threshold may name (a count of 200 in the sketch).
    pub const MAX_UPDATES_PER_SECOND: f64 = 100.0;

    /// A tier of at most `max_bytes`, holding rows updated more than once a second.
    pub fn with_budget(max_bytes: usize) -> HotTierConfig {
        HotTierConfig {
            max_bytes,
            min_updates_per_second: Self::DEFAULT_MIN_UPDATES_PER_SECOND,
            age_every: Duration::from_secs(1),
            evict_sample: 8,
            history: None,
        }
    }

    /// Keeps the history of dropped operations: see [`HistoryStub`].
    pub fn with_history(mut self, stub: HistoryStub) -> HotTierConfig {
        self.history = Some(stub);
        self
    }

    /// Sets the threshold; an error names a rate outside
    /// [`Self::MIN_UPDATES_PER_SECOND`]..=[`Self::MAX_UPDATES_PER_SECOND`].
    pub fn with_min_updates_per_second(mut self, rate: f64) -> Result<HotTierConfig, String> {
        if !(Self::MIN_UPDATES_PER_SECOND..=Self::MAX_UPDATES_PER_SECOND).contains(&rate) {
            return Err(format!(
                "the hot tier's update rate must be between {} and {} updates a second, got {rate}",
                Self::MIN_UPDATES_PER_SECOND,
                Self::MAX_UPDATES_PER_SECOND
            ));
        }
        self.min_updates_per_second = rate;
        Ok(self)
    }

    /// The sketch count from which a row is held. The counts are halved
    /// every `age_every`, so a row updated `r` times a second swings between
    /// `r * age` and `2 * r * age` (the count just before a halving is twice
    /// the one just after it, plus the updates in between): a count above
    /// `2 * rate * age` is reached only by a row faster than `rate`. At the
    /// default, 1 a second and a 1 s period, that is 3.
    pub fn admit_at(&self) -> u8 {
        let per_period = self.min_updates_per_second * self.age_every.as_secs_f64();
        let count = (2.0 * per_period).floor() + 1.0;
        count.clamp(2.0, 200.0) as u8
    }
}

/// One operation a row's latest update is made of: what
/// `soft_update_explicit_documents_with_vectors_in` takes.
pub struct HeldOp {
    pub partition: Option<Arc<str>>,
    pub term: Term,
    pub docs: Vec<ExplicitDocument>,
    pub vectors: Option<Vec<Vec<DocumentVector>>>,
    pub soft_deletes: Vec<DocValuesUpdate>,
    bytes: usize,
    heat: u8,
}

impl HeldOp {
    /// Wraps an operation; the memory it holds is measured here.
    pub fn new(
        partition: Option<Arc<str>>,
        term: Term,
        docs: Vec<ExplicitDocument>,
        vectors: Option<Vec<Vec<DocumentVector>>>,
        soft_deletes: Vec<DocValuesUpdate>,
    ) -> HeldOp {
        HeldOp {
            partition,
            term,
            docs,
            vectors,
            soft_deletes,
            bytes: 0,
            heat: 0,
        }
    }

    /// Measures the operation, once, when the tier is about to keep it: an
    /// operation that goes straight through is never walked.
    fn measure(&mut self) {
        if self.bytes == 0 {
            self.bytes = op_bytes(
                &self.term,
                &self.docs,
                self.vectors.as_deref(),
                &self.soft_deletes,
            );
        }
    }

    /// The memory this operation takes while held (0 until the tier has
    /// kept or refused it).
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

fn op_bytes(
    term: &Term,
    docs: &[ExplicitDocument],
    vectors: Option<&[Vec<DocumentVector>]>,
    soft: &[DocValuesUpdate],
) -> usize {
    use std::mem::size_of;
    // The map's slot (the shard map grows by doubling, so about half again as
    // many slots as rows), the key (a copy of the term's bytes) and the
    // operation's own fields.
    let slot = size_of::<Vec<u8>>()
        .saturating_add(size_of::<HeldOp>())
        .saturating_add(1);
    let mut n = slot
        .saturating_mul(3)
        .saturating_div(2)
        .saturating_add(term.bytes.capacity().saturating_mul(2))
        .saturating_add(term.field.capacity());
    // The soft-deletes updates, each with its own copy of the term.
    for u in soft {
        n = n
            .saturating_add(size_of::<DocValuesUpdate>())
            .saturating_add(term.bytes.capacity())
            .saturating_add(term.field.capacity())
            .saturating_add(update_field_bytes(u));
    }
    n = n.saturating_add(docs.len().saturating_mul(size_of::<ExplicitDocument>()));
    for d in docs {
        n = n
            .saturating_add(d.fields.ram_bytes())
            .saturating_add(
                d.fields
                    .inverted
                    .len()
                    .saturating_mul(size_of::<crate::index_writer::InvertedField>()),
            )
            .saturating_add(
                d.fields
                    .doc_values
                    .len()
                    .saturating_add(d.fields.points.len())
                    .saturating_mul(size_of::<StoredField>()),
            )
            .saturating_add(d.stored.capacity().saturating_mul(size_of::<StoredField>()));
        for f in &d.stored {
            n = n.saturating_add(match &f.value {
                FieldValue::String(s) => s.capacity(),
                FieldValue::Shared(s) => s.len(),
                FieldValue::Binary(b) => b.capacity(),
                _ => 0,
            });
        }
    }
    if let Some(v) = vectors {
        for dv in v.iter().flatten() {
            let bytes = match &dv.value {
                crate::index_writer::VectorValue::Float32(v) => v.len().saturating_mul(4),
                crate::index_writer::VectorValue::Byte(v) => v.len(),
            };
            n = n
                .saturating_add(size_of::<DocumentVector>())
                .saturating_add(bytes);
        }
    }
    // Allocator rounding and what the walk does not see.
    n.saturating_add(n / 32)
}

fn update_field_bytes(u: &DocValuesUpdate) -> usize {
    match u {
        DocValuesUpdate::Numeric { field, .. } => field.capacity(),
        _ => 0,
    }
}

/// What [`HotTier::offer`] decided.
pub enum Offer {
    /// The tier holds the operation. `dropped` is the older held operation of
    /// the row it replaced: never applied, so the caller owes its history.
    Held { dropped: Option<HeldOp> },
    /// Not held: apply it now, as without the tier. `dropped` is as above.
    Through { op: HeldOp, dropped: Option<HeldOp> },
}

/// Counters, read with [`HotTier::stats`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HotTierStats {
    pub offered: u64,
    /// Went to the writer at once: not hot, or no room.
    pub written_through: u64,
    /// Held as a row's first held operation.
    pub admitted: u64,
    /// Held in place of an older held operation, which was never applied.
    pub coalesced: u64,
    /// Applied to make room for a hotter row.
    pub spilled: u64,
    /// Applied by a drain.
    pub drained: u64,
    pub drains: u64,
    /// Wanted a place, found none after sampling.
    pub refused_full: u64,
    pub bytes: usize,
    pub peak_bytes: usize,
    pub held_rows: usize,
    pub sketch_bytes: usize,
}

/// One stripe of the heat sketch: the cells, and how many of the first row's
/// are in use.
#[derive(Default)]
struct Stripe {
    cells: Vec<u8>,
    used: usize,
}

struct Shard {
    rows: HashMap<Vec<u8>, HeldOp>,
}

/// See the module documentation.
pub struct HotTier {
    cfg: HotTierConfig,
    /// The heat sketch, by stripe: two byte rows of a power of two of cells,
    /// empty until a stripe sees an update, grown while its cells fill, and
    /// freed once its counts have all decayed to zero.
    sketch: Vec<Mutex<Stripe>>,
    sketch_bytes: AtomicUsize,
    max_slots: usize,
    shards: Vec<Mutex<Shard>>,
    bytes: AtomicUsize,
    peak: AtomicUsize,
    rows: AtomicUsize,
    held_docs: AtomicUsize,
    ticks: AtomicU64,
    aged_at: Mutex<Instant>,
    offered: AtomicU64,
    through: AtomicU64,
    admitted: AtomicU64,
    coalesced: AtomicU64,
    spilled: AtomicU64,
    drained: AtomicU64,
    drains: AtomicU64,
    refused: AtomicU64,
    sample_at: AtomicUsize,
    no_spill_until: AtomicU64,
    admit: u8,
    /// The field of the terms the held rows are keyed by (the first one offered).
    term_field: std::sync::OnceLock<String>,
}

/// Takes `n` bytes of the node's pool, when `limit` (0: none) leaves room.
fn node_take(n: usize, limit: usize) -> bool {
    if limit == 0 {
        NODE.used.fetch_add(n, Ordering::AcqRel);
        return true;
    }
    NODE.used
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |u| {
            (u.saturating_add(n) <= limit).then_some(u.saturating_add(n))
        })
        .is_ok()
}

impl Drop for HotTier {
    /// A closed shard gives back what it still holds.
    fn drop(&mut self) {
        let held = self.bytes.load(Ordering::Acquire);
        self.release(held);
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn hash(bytes: &[u8]) -> u64 {
    // FNV-1a, then a finaliser so the two sketch rows draw on different bits.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h
}

impl HotTier {
    /// A tier within `cfg.max_bytes`. The sketch takes at most an eighth of
    /// the budget (two byte rows, a power of two each, 1 MiB at most).
    pub fn new(cfg: HotTierConfig) -> HotTier {
        let admit = cfg.admit_at();
        Self::build(cfg, admit)
    }

    fn build(cfg: HotTierConfig, admit: u8) -> HotTier {
        // At most an eighth of the budget, even when every stripe is full.
        let mut max_slots = STRIPE_MAX_SLOTS;
        while max_slots > 4
            && max_slots.saturating_mul(2).saturating_mul(STRIPES) > cfg.max_bytes / 8
        {
            max_slots /= 2;
        }
        let shards = (0..SHARDS)
            .map(|_| {
                Mutex::new(Shard {
                    rows: HashMap::new(),
                })
            })
            .collect();
        let sketch = (0..STRIPES).map(|_| Mutex::default()).collect();
        HotTier {
            admit,
            cfg,
            sketch,
            sketch_bytes: AtomicUsize::new(0),
            max_slots,
            shards,
            bytes: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            rows: AtomicUsize::new(0),
            held_docs: AtomicUsize::new(0),
            ticks: AtomicU64::new(0),
            aged_at: Mutex::new(Instant::now()),
            offered: AtomicU64::new(0),
            through: AtomicU64::new(0),
            admitted: AtomicU64::new(0),
            coalesced: AtomicU64::new(0),
            spilled: AtomicU64::new(0),
            drained: AtomicU64::new(0),
            drains: AtomicU64::new(0),
            refused: AtomicU64::new(0),
            sample_at: AtomicUsize::new(0),
            no_spill_until: AtomicU64::new(0),
            term_field: std::sync::OnceLock::new(),
        }
    }

    /// Counts one update of the row hashed `h` and returns its heat. A
    /// stripe's cells are allocated by its first update; when the budget has
    /// no room for them the row counts as cold (0).
    fn bump(&self, h: u64) -> u8 {
        let mut st = lock(&self.sketch[((h >> 48) as usize) % STRIPES]);
        if st.cells.is_empty() {
            let bytes = STRIPE_START_SLOTS.min(self.max_slots).saturating_mul(2);
            if !self.reserve(bytes) {
                return 0;
            }
            self.sketch_bytes.fetch_add(bytes, Ordering::AcqRel);
            st.cells = vec![0u8; bytes];
            st.used = 0;
        }
        // Over a quarter of the first row in use: the stripe is shared by too
        // many rows to tell them apart, so it grows fourfold (when the budget
        // allows) and starts counting again.
        let have = st.cells.len();
        if st.used.saturating_mul(4) > have / 2 && have / 2 < self.max_slots {
            let want = have.saturating_mul(4).min(self.max_slots.saturating_mul(2));
            let extra = want.saturating_sub(have);
            if self.reserve(extra) {
                self.sketch_bytes.fetch_add(extra, Ordering::AcqRel);
                st.cells = vec![0u8; want];
                st.used = 0;
            }
        }
        let n = st.cells.len() / 2;
        let mask = n.saturating_sub(1);
        let a = (h as usize) & mask;
        let b = n.saturating_add(((h >> 32) as usize) & mask);
        if st.cells[a] == 0 {
            st.used = st.used.saturating_add(1);
        }
        let mut inc = |i: usize| -> u8 {
            let v = st.cells[i];
            if v < u8::MAX {
                st.cells[i] = v.saturating_add(1);
            }
            st.cells[i]
        };
        let (x, y) = (inc(a), inc(b));
        x.min(y)
    }

    fn maybe_age(&self) {
        if self.ticks.fetch_add(1, Ordering::Relaxed) & 1023 != 0 {
            return;
        }
        self.age();
    }

    /// When a period has passed, halves every count (see [`Self::halve`]).
    fn age(&self) {
        let mut at = lock(&self.aged_at);
        if at.elapsed() < self.cfg.age_every {
            return;
        }
        *at = Instant::now();
        drop(at);
        self.halve();
    }

    /// Halves every count; a stripe whose counts are all gone gives its cells
    /// back.
    fn halve(&self) {
        for stripe in &self.sketch {
            let mut st = lock(stripe);
            if st.cells.is_empty() {
                continue;
            }
            let half = st.cells.len() / 2;
            let (mut used, mut first) = (0usize, 0usize);
            for (i, c) in st.cells.iter_mut().enumerate() {
                *c >>= 1;
                if *c != 0 {
                    used = used.saturating_add(1);
                    first = first.saturating_add(usize::from(i < half));
                }
            }
            st.used = first;
            if used == 0 {
                let have = st.cells.len();
                self.release(have);
                self.sketch_bytes.fetch_sub(have, Ordering::AcqRel);
                st.cells = Vec::new();
            }
        }
    }

    fn docs_up(&self, n: usize) {
        self.held_docs.fetch_add(n, Ordering::AcqRel);
    }

    fn docs_down(&self, n: usize) {
        let _ = self
            .held_docs
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |d| {
                Some(d.saturating_sub(n))
            });
    }

    /// How many documents the held operations will add when applied.
    pub fn held_docs(&self) -> usize {
        self.held_docs.load(Ordering::Acquire)
    }

    fn rows_down(&self) {
        let _ = self
            .rows
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |r| {
                Some(r.saturating_sub(1))
            });
    }

    fn release(&self, n: usize) {
        if let Ok(prev) = self
            .bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |b| {
                Some(b.saturating_sub(n))
            })
        {
            let freed = n.min(prev);
            NODE.used.fetch_sub(freed, Ordering::AcqRel);
            if freed > 0 && prev == freed {
                NODE.active.fetch_sub(1, Ordering::AcqRel);
            }
        }
    }

    /// Whether `n` more bytes fit, reserving them when they do: within this
    /// tier's budget, within its equal share of the node's, and with room left
    /// in the node's.
    fn reserve(&self, n: usize) -> bool {
        let mut cur = self.bytes.load(Ordering::Acquire);
        loop {
            let want = cur.saturating_add(n);
            if want > self.cfg.max_bytes {
                return false;
            }
            let limit = NODE.limit.load(Ordering::Acquire);
            if limit > 0 {
                let tiers = NODE
                    .active
                    .load(Ordering::Acquire)
                    .saturating_add(usize::from(cur == 0))
                    .max(1);
                if want > limit.checked_div(tiers).unwrap_or(limit) {
                    return false;
                }
            }
            if !node_take(n, limit) {
                return false;
            }
            match self
                .bytes
                .compare_exchange_weak(cur, want, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    if cur == 0 {
                        NODE.active.fetch_add(1, Ordering::AcqRel);
                    }
                    self.peak.fetch_max(want, Ordering::Relaxed);
                    return true;
                }
                Err(c) => {
                    NODE.used.fetch_sub(n, Ordering::AcqRel);
                    cur = c;
                }
            }
        }
    }

    /// Offers the latest operation of its row. `apply` runs an operation on
    /// the writer; it is called, in order, for rows spilled to make room.
    ///
    /// `Offer::Through` hands the operation back to be applied at once.
    pub fn offer<E>(
        &self,
        mut op: HeldOp,
        apply: &mut dyn FnMut(HeldOp) -> Result<(), E>,
    ) -> Result<Offer, E> {
        self.offered.fetch_add(1, Ordering::Relaxed);
        // Rows are keyed by the bytes of one field's terms: an update by any
        // other field's term is not held.
        if self.term_field.get_or_init(|| op.term.field.clone()) != &op.term.field {
            self.through.fetch_add(1, Ordering::Relaxed);
            return Ok(Offer::Through { op, dropped: None });
        }
        self.maybe_age();
        let h = hash(&op.term.bytes);
        let heat = self.bump(h);
        op.heat = heat;
        let shard = (h as usize) % SHARDS;
        // The older held operation of this row, when a larger version of it
        // found no room beside the others and must look for a place.
        let mut carried: Option<HeldOp> = None;
        // An older held operation of this row is replaced.
        if self.rows.load(Ordering::Acquire) > 0 {
            let mut s = lock(&self.shards[shard]);
            if let Some(old) = s.rows.get_mut(op.term.bytes.as_slice()) {
                op.measure();
                let grow = op.bytes.saturating_sub(old.bytes);
                if grow == 0 || self.reserve(grow) {
                    let shrink = old.bytes.saturating_sub(op.bytes);
                    if shrink > 0 {
                        self.release(shrink);
                    }
                    self.docs_up(op.docs.len());
                    let dropped = std::mem::replace(old, op);
                    self.docs_down(dropped.docs.len());
                    self.coalesced.fetch_add(1, Ordering::Relaxed);
                    return Ok(Offer::Held {
                        dropped: Some(dropped),
                    });
                }
                // No room beside the others for the larger version: the older
                // one is dropped (this one replaces it) and this one looks for
                // a place as a row of its own, spilling a cooler row if it must.
                carried = s.rows.remove(op.term.bytes.as_slice());
                if let Some(old) = &carried {
                    self.release(old.bytes);
                    self.docs_down(old.docs.len());
                    self.rows_down();
                }
            }
        }
        if heat < self.admit && carried.is_none() {
            self.through.fetch_add(1, Ordering::Relaxed);
            return Ok(Offer::Through { op, dropped: None });
        }
        // A hot row with no held operation: find it a place.
        op.measure();
        if op.bytes
            > self
                .cfg
                .max_bytes
                .saturating_sub(self.sketch_bytes.load(Ordering::Relaxed))
        {
            self.refused.fetch_add(1, Ordering::Relaxed);
            self.through.fetch_add(1, Ordering::Relaxed);
            return Ok(Offer::Through {
                op,
                dropped: carried,
            });
        }
        // After a failed search for a row to spill, the next hot rows skip
        // the search for a while: a full tier of equally hot rows would
        // otherwise pay it on every update.
        // A row far hotter than the threshold always searches: rows that only
        // just made it in must not keep it out.
        let urgent = heat >= self.admit.saturating_mul(4).max(16);
        let tick = self.ticks.load(Ordering::Relaxed);
        let mut tries = 0usize;
        while !self.reserve(op.bytes) {
            tries = tries.saturating_add(1);
            if tries > 4
                || (!urgent && tick < self.no_spill_until.load(Ordering::Relaxed))
                || !self.spill_one(heat, apply)?
            {
                if tries <= 4 && !urgent {
                    self.no_spill_until
                        .store(tick.saturating_add(NO_SPILL_TICKS), Ordering::Relaxed);
                }
                self.refused.fetch_add(1, Ordering::Relaxed);
                self.through.fetch_add(1, Ordering::Relaxed);
                return Ok(Offer::Through {
                    op,
                    dropped: carried,
                });
            }
        }
        let key = op.term.bytes.clone();
        self.docs_up(op.docs.len());
        let mut s = lock(&self.shards[shard]);
        let mut dropped = s.rows.insert(key, op);
        drop(s);
        if let Some(old) = &dropped {
            self.docs_down(old.docs.len());
            // Raced with another thread's insert of this row: the older one
            // was replaced, and its bytes are given back.
            self.release(old.bytes);
            self.coalesced.fetch_add(1, Ordering::Relaxed);
        } else {
            self.rows.fetch_add(1, Ordering::AcqRel);
            self.admitted.fetch_add(1, Ordering::Relaxed);
        }
        match (carried, dropped.is_some()) {
            (Some(old), false) => dropped = Some(old),
            // Two older versions (a race the engine's per-id lock rules
            // out): the second is applied, so no sequence number is lost.
            (Some(old), true) => apply(old)?,
            (None, _) => {}
        }
        Ok(Offer::Held { dropped })
    }

    /// Applies the least heated of a few sampled held rows, if it is cooler
    /// than `than`; whether room was made.
    fn spill_one<E>(
        &self,
        than: u8,
        apply: &mut dyn FnMut(HeldOp) -> Result<(), E>,
    ) -> Result<bool, E> {
        let start = self.sample_at.fetch_add(1, Ordering::Relaxed);
        let mut best: Option<(usize, Vec<u8>, u8)> = None;
        let mut seen = 0usize;
        for i in 0..SHARDS {
            let sh = start.wrapping_add(i) % SHARDS;
            let s = lock(&self.shards[sh]);
            for (k, v) in s.rows.iter().take(self.cfg.evict_sample.max(1)) {
                seen = seen.saturating_add(1);
                if best.as_ref().is_none_or(|b| v.heat < b.2) {
                    best = Some((sh, k.clone(), v.heat));
                }
            }
            if seen >= self.cfg.evict_sample.max(1) {
                break;
            }
        }
        let Some((sh, key, heat)) = best else {
            return Ok(false);
        };
        if heat >= than {
            return Ok(false);
        }
        let mut s = lock(&self.shards[sh]);
        let Some(op) = s.rows.remove(&key) else {
            return Ok(true);
        };
        self.rows_down();
        let (bytes, docs) = (op.bytes, op.docs.len());
        // Applied with the shard held, counted until it is applied.
        let r = apply(op);
        self.release(bytes);
        self.docs_down(docs);
        r?;
        self.spilled.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    /// The field of the terms the held rows are keyed by; `None` before any
    /// update was offered.
    pub fn term_field(&self) -> Option<&str> {
        self.term_field.get().map(String::as_str)
    }

    /// Applies the held operation of the row keyed `key`, if there is one:
    /// what a plain add of that row does first, so the add lands after the
    /// held update that came before it and no delete of that update reaches
    /// the added document. Returns whether an operation was applied.
    pub fn settle<E>(
        &self,
        key: &[u8],
        apply: &mut dyn FnMut(HeldOp) -> Result<(), E>,
    ) -> Result<bool, E> {
        if self.rows.load(Ordering::Acquire) == 0 {
            return Ok(false);
        }
        let mut s = lock(&self.shards[(hash(key) as usize) % SHARDS]);
        let Some(op) = s.rows.remove(key) else {
            return Ok(false);
        };
        self.rows_down();
        let (bytes, docs) = (op.bytes, op.docs.len());
        // Applied with the shard held, counted until it is applied.
        let r = apply(op);
        self.release(bytes);
        self.docs_down(docs);
        r?;
        self.drained.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    /// Applies every held operation.
    pub fn drain<E>(&self, apply: &mut dyn FnMut(HeldOp) -> Result<(), E>) -> Result<usize, E> {
        // A tier with nothing held and no sketch has nothing to apply or age:
        // a refresh of an idle shard costs two loads.
        if self.rows.load(Ordering::Acquire) == 0 && self.sketch_bytes.load(Ordering::Acquire) == 0
        {
            return Ok(0);
        }
        self.drains.fetch_add(1, Ordering::Relaxed);
        // A cut is when an idle tier lets go of the sketch it no longer needs.
        self.age();
        let mut n = 0usize;
        for shard in &self.shards {
            let mut s = lock(shard);
            let keys: Vec<Vec<u8>> = s.rows.keys().cloned().collect();
            for k in keys {
                if let Some(op) = s.rows.remove(&k) {
                    self.rows_down();
                    let (bytes, docs) = (op.bytes, op.docs.len());
                    let r = apply(op);
                    self.release(bytes);
                    self.docs_down(docs);
                    r?;
                    n = n.saturating_add(1);
                }
            }
        }
        self.drained.fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
    }

    /// What to write in place of a dropped operation, when configured.
    pub fn history(&self) -> Option<&HistoryStub> {
        self.cfg.history.as_ref()
    }

    /// Whether any operation is held.
    pub fn is_empty(&self) -> bool {
        self.rows.load(Ordering::Acquire) == 0
    }

    /// The memory the tier holds now, sketch included.
    pub fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Acquire)
    }

    pub fn stats(&self) -> HotTierStats {
        HotTierStats {
            offered: self.offered.load(Ordering::Relaxed),
            written_through: self.through.load(Ordering::Relaxed),
            admitted: self.admitted.load(Ordering::Relaxed),
            coalesced: self.coalesced.load(Ordering::Relaxed),
            spilled: self.spilled.load(Ordering::Relaxed),
            drained: self.drained.load(Ordering::Relaxed),
            drains: self.drains.load(Ordering::Relaxed),
            refused_full: self.refused.load(Ordering::Relaxed),
            bytes: self.bytes(),
            peak_bytes: self.peak.load(Ordering::Relaxed),
            held_rows: self.rows.load(Ordering::Relaxed),
            sketch_bytes: self.sketch_bytes.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    // Test arithmetic on small constants; see docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use std::collections::HashMap;
    use std::convert::Infallible;

    use lucene_codecs::stored_fields::{FieldValue, StoredField};

    use super::*;
    use crate::index_writer::ExplicitFields;

    fn op(id: u32, version: u8, size: usize) -> HeldOp {
        let term = Term::new("_id", format!("d{id}").into_bytes());
        let doc = ExplicitDocument {
            stored: vec![StoredField {
                field_number: 0,
                value: FieldValue::Binary(vec![version; size]),
            }],
            fields: ExplicitFields::default(),
        };
        HeldOp::new(None, term, vec![doc], None, Vec::new())
    }

    fn version_of(op: &HeldOp) -> u8 {
        match &op.docs[0].stored[0].value {
            FieldValue::Binary(b) => b[0],
            _ => unreachable!(),
        }
    }

    type Applied = HashMap<Vec<u8>, Vec<u8>>;

    fn run(
        tier: &HotTier,
        applied: &mut Applied,
        id: u32,
        version: u8,
        size: usize,
    ) -> Option<HeldOp> {
        let mut apply = |held: HeldOp| -> Result<(), Infallible> {
            applied
                .entry(held.term.bytes.clone())
                .or_default()
                .push(version_of(&held));
            Ok(())
        };
        match tier.offer(op(id, version, size), &mut apply) {
            Ok(Offer::Held { .. }) => None,
            Ok(Offer::Through { op, .. }) => Some(op),
            Err(never) => match never {},
        }
    }

    fn drain(tier: &HotTier, applied: &mut Applied) -> usize {
        let mut apply = |held: HeldOp| -> Result<(), Infallible> {
            applied
                .entry(held.term.bytes.clone())
                .or_default()
                .push(version_of(&held));
            Ok(())
        };
        match tier.drain(&mut apply) {
            Ok(n) => n,
            Err(never) => match never {},
        }
    }

    fn cfg(max_bytes: usize) -> HotTierConfig {
        HotTierConfig {
            max_bytes,
            min_updates_per_second: 1.0,
            age_every: Duration::from_secs(3600),
            evict_sample: 8,
            history: None,
        }
    }

    #[test]
    fn a_row_is_held_from_its_third_update_and_coalesced() {
        let tier = HotTier::build(cfg(1 << 22), 3);
        let mut applied = Applied::new();
        // The first two updates go through: not hot yet.
        assert!(run(&tier, &mut applied, 1, 1, 100).is_some());
        assert!(run(&tier, &mut applied, 1, 2, 100).is_some());
        // The third and later are held; each replaces the one before.
        for v in 3..=9 {
            assert!(run(&tier, &mut applied, 1, v, 100).is_none());
        }
        let st = tier.stats();
        assert_eq!((st.admitted, st.coalesced, st.held_rows), (1, 6, 1));
        assert_eq!(st.written_through, 2);
        assert_eq!(drain(&tier, &mut applied), 1);
        // Only the last held version was ever applied.
        assert_eq!(tier.stats().held_rows, 0);
        assert!(tier.is_empty());
        assert_eq!(tier.bytes(), tier.stats().sketch_bytes);
        assert_eq!(applied.len(), 1);
        assert_eq!(applied.values().next().unwrap(), &vec![9u8]);
    }

    /// A plain add of a row applies the row's held update first, once.
    #[test]
    fn settling_a_row_applies_its_held_update_and_only_that() {
        let tier = HotTier::build(cfg(1 << 22), 3);
        let mut applied = Applied::new();
        let key = |id: u32| format!("d{id}").into_bytes();
        let settle = |tier: &HotTier, applied: &mut Applied, id: u32| {
            let mut apply = |held: HeldOp| -> Result<(), Infallible> {
                applied
                    .entry(held.term.bytes.clone())
                    .or_default()
                    .push(version_of(&held));
                Ok(())
            };
            match tier.settle(&key(id), &mut apply) {
                Ok(done) => done,
                Err(never) => match never {},
            }
        };
        assert_eq!(tier.term_field(), None);
        // Nothing held: nothing to apply.
        assert!(!settle(&tier, &mut applied, 1));
        for v in 1..=5 {
            run(&tier, &mut applied, 1, v, 100);
            run(&tier, &mut applied, 2, v, 100);
        }
        assert_eq!(tier.term_field(), Some("_id"));
        applied.clear();
        let held = tier.bytes();
        assert!(settle(&tier, &mut applied, 1));
        // Row 1's latest version, applied; row 2 is still held.
        assert_eq!(applied.get(&key(1)), Some(&vec![5u8]));
        assert!(!applied.contains_key(&key(2)));
        assert_eq!(tier.stats().held_rows, 1);
        assert!(tier.bytes() < held);
        // A second settle of the same row has nothing left, and a row never
        // held is not found.
        assert!(!settle(&tier, &mut applied, 1));
        assert!(!settle(&tier, &mut applied, 99));
        assert_eq!(drain(&tier, &mut applied), 1);
        assert_eq!(tier.bytes(), tier.stats().sketch_bytes);
    }

    #[test]
    fn a_failed_settle_reports_the_error() {
        let tier = HotTier::build(cfg(1 << 22), 3);
        let mut applied = Applied::new();
        for v in 1..=4 {
            run(&tier, &mut applied, 1, v, 100);
        }
        let mut fail = |_: HeldOp| -> Result<(), &'static str> { Err("disk full") };
        assert_eq!(tier.settle(b"d1", &mut fail), Err("disk full"));
        assert!(tier.is_empty());
    }

    /// Rows are keyed by the bytes of one field's terms: an update by another
    /// field's term goes straight through.
    #[test]
    fn an_update_by_another_terms_field_is_not_held() {
        let tier = HotTier::build(cfg(1 << 22), 3);
        let mut applied = Applied::new();
        run(&tier, &mut applied, 1, 1, 100);
        for _ in 0..6 {
            let mut apply = |_: HeldOp| -> Result<(), Infallible> { Ok(()) };
            let mut o = op(1, 2, 100);
            o.term = Term::new("other", b"d1".to_vec());
            assert!(matches!(
                tier.offer(o, &mut apply),
                Ok(Offer::Through { dropped: None, .. })
            ));
        }
        assert!(tier.is_empty());
    }

    #[test]
    fn rows_updated_once_are_never_held() {
        let tier = HotTier::build(cfg(1 << 22), 3);
        let mut applied = Applied::new();
        let mut held = 0;
        for id in 0..50_000u32 {
            if run(&tier, &mut applied, id, 1, 200).is_none() {
                held += 1;
            }
        }
        // The sketch may take a row for hot when two others share both its
        // cells: a few in ten thousand, held until the next drain.
        let st = tier.stats();
        assert!(held < 50, "{held} of 50000 rows held");
        assert_eq!(st.written_through, 50_000 - held);
        assert_eq!(drain(&tier, &mut applied) as u64, held);
        assert_eq!(tier.bytes(), st.sketch_bytes);
    }

    #[test]
    fn the_budget_is_never_exceeded_and_the_last_version_always_wins() {
        // Room for about 20 rows of 1 KiB beside the sketch.
        let budget = 20 * 1300 + 4096;
        let tier = HotTier::build(cfg(budget), 3);
        let mut applied = Applied::new();
        let mut latest: HashMap<u32, u8> = HashMap::new();
        let mut through = Vec::new();
        for round in 1..=20u8 {
            for id in 0..100u32 {
                latest.insert(id, round);
                if let Some(op) = run(&tier, &mut applied, id, round, 1024) {
                    through.push((op.term.bytes.clone(), version_of(&op)));
                }
                assert!(tier.bytes() <= budget, "{} > {budget}", tier.bytes());
            }
        }
        // Rows written through are applied by the caller, in order, as the
        // writer does; spilled ones were applied by the tier.
        for (k, v) in through {
            applied.entry(k).or_default().push(v);
        }
        drain(&tier, &mut applied);
        let st = tier.stats();
        assert!(st.peak_bytes <= budget, "peak {} > {budget}", st.peak_bytes);
        assert!(st.spilled > 0 || st.refused_full > 0);
        for (id, round) in latest {
            let seen = &applied[format!("d{id}").as_bytes()];
            assert_eq!(*seen.last().unwrap(), round, "row {id}: {seen:?}");
        }
    }

    #[test]
    fn a_row_larger_than_the_budget_goes_through() {
        let tier = HotTier::build(cfg(8192), 3);
        let mut applied = Applied::new();
        for v in 1..=5 {
            assert!(run(&tier, &mut applied, 1, v, 1 << 20).is_some());
        }
        assert_eq!(tier.stats().held_rows, 0);
    }

    #[test]
    fn a_larger_version_that_does_not_fit_replaces_and_goes_through() {
        let tier = HotTier::build(cfg(16 * 1024), 3);
        let mut applied = Applied::new();
        for v in 1..=3 {
            let _ = run(&tier, &mut applied, 1, v, 100);
        }
        assert_eq!(tier.stats().held_rows, 1);
        // The next version is too big to hold: it goes through and the
        // smaller held one, which it replaces, is dropped.
        assert!(run(&tier, &mut applied, 1, 4, 64 * 1024).is_some());
        assert_eq!(tier.stats().held_rows, 0);
        assert_eq!(tier.bytes(), tier.stats().sketch_bytes);
        assert_eq!(drain(&tier, &mut applied), 0);
    }

    #[test]
    fn ageing_forgets_old_heat() {
        let mut c = cfg(1 << 22);
        c.age_every = Duration::ZERO;
        let tier = HotTier::build(c, 3);
        let mut applied = Applied::new();
        assert!(run(&tier, &mut applied, 1, 1, 100).is_some());
        assert!(run(&tier, &mut applied, 1, 2, 100).is_some());
        // 4096 other offers: ageing halves the counts at least 3 times, so
        // this row's two updates are forgotten and it is cold again.
        for id in 1000..5096u32 {
            let _ = run(&tier, &mut applied, id, 1, 10);
        }
        assert!(run(&tier, &mut applied, 1, 3, 100).is_some());
    }

    #[test]
    fn the_threshold_is_a_rate_with_a_default_of_more_than_one_a_second() {
        let c = HotTierConfig::with_budget(1 << 20);
        assert_eq!(c.min_updates_per_second, 1.0);
        assert_eq!(c.admit_at(), 3);
        let at = |rate: f64| {
            HotTierConfig::with_budget(1 << 20)
                .with_min_updates_per_second(rate)
                .unwrap()
                .admit_at()
        };
        // 0.5 a second is the lowest: a second update within the period.
        assert_eq!(
            (at(0.5), at(1.0), at(2.0), at(10.0), at(100.0)),
            (2, 3, 5, 21, 200)
        );
        // A shorter period needs fewer counts for the same rate.
        let mut half = HotTierConfig::with_budget(1 << 20)
            .with_min_updates_per_second(4.0)
            .unwrap();
        half.age_every = Duration::from_millis(500);
        assert_eq!(half.admit_at(), 5);
        for bad in [0.0, 0.49, -1.0, 100.5, f64::NAN, f64::INFINITY] {
            let e = HotTierConfig::with_budget(1)
                .with_min_updates_per_second(bad)
                .unwrap_err();
            assert!(e.contains("between 0.5 and 100"), "{e}");
        }
    }

    #[test]
    fn a_row_faster_than_the_rate_is_held_and_one_slower_is_not() {
        // Row A is updated 6 times and row B once per period, three
        // periods running (the sketch is halved by hand between them).
        let mut c = cfg(1 << 22);
        c.age_every = Duration::from_secs(1);
        let tier = HotTier::new(c);
        let mut applied = Applied::new();
        let (mut a_held, mut b_held) = (0, 0);
        for _period in 0..3 {
            for v in 0..6u8 {
                if run(&tier, &mut applied, 1, v, 100).is_none() {
                    a_held += 1;
                }
            }
            for v in 0..1u8 {
                if run(&tier, &mut applied, 2, v, 100).is_none() {
                    b_held += 1;
                }
            }
            drain(&tier, &mut applied);
            tier.halve();
        }
        assert!(a_held >= 9, "{a_held}");
        assert_eq!(b_held, 0);
    }

    #[test]
    fn the_sketch_is_counted_and_shrinks_with_a_small_budget() {
        let big = HotTier::build(cfg(64 << 20), 3);
        assert_eq!(big.stats().sketch_bytes, 0);
        assert_eq!(big.max_slots * 2 * STRIPES, 2 << 20);
        let small = HotTier::build(cfg(64 << 10), 3);
        assert!(small.max_slots * 2 * STRIPES <= (64 << 10) / 8);
        // Every stripe the small tier uses stays within its budget.
        let mut applied = Applied::new();
        for id in 0..20_000u32 {
            let _ = run(&small, &mut applied, id, 1, 8);
            assert!(small.bytes() <= 64 << 10);
        }
        assert!(small.stats().sketch_bytes <= (64 << 10) / 8);
        drain(&small, &mut applied);
        assert_eq!(small.bytes(), small.stats().sketch_bytes);
    }

    #[test]
    fn an_idle_tier_holds_nothing_and_a_busy_one_grows_its_sketch() {
        let tier = HotTier::build(cfg(8 << 20), 3);
        assert_eq!((tier.bytes(), tier.stats().sketch_bytes), (0, 0));
        let mut applied = Applied::new();
        // A burst of distinct rows: the stripes start small and grow with it.
        for id in 0..200_000u32 {
            let _ = run(&tier, &mut applied, id, 1, 8);
        }
        let busy = tier.stats().sketch_bytes;
        assert!(busy > STRIPES * STRIPE_START_SLOTS * 2, "{busy}");
        assert!(busy <= (8 << 20) / 8, "{busy}");
        drain(&tier, &mut applied);
        assert_eq!(tier.bytes(), tier.stats().sketch_bytes);
        // Counts of at most 255 halve to nothing in eight periods, and the
        // sketch is freed with them.
        for _ in 0..9 {
            tier.halve();
        }
        assert_eq!((tier.bytes(), tier.stats().sketch_bytes), (0, 0));
        // And it is reallocated by the next update.
        let _ = run(&tier, &mut applied, 1, 1, 8);
        assert!(tier.stats().sketch_bytes > 0);
    }

    #[test]
    fn a_hot_row_whose_larger_version_does_not_fit_keeps_its_place() {
        // Room for two 1 KiB rows beside the sketch; a third row is held,
        // cooler than the first, which then grows.
        let tier = HotTier::build(cfg(24 * 1024), 3);
        let mut applied = Applied::new();
        for round in 0..8u8 {
            for id in 0..3u32 {
                let _ = run(&tier, &mut applied, id, round, 1024);
            }
        }
        for round in 8..20u8 {
            let _ = run(&tier, &mut applied, 0, round, 1024);
        }
        let before = tier.stats().held_rows;
        // A version of row 0 twice the size: it must still be held (a cooler
        // row is spilled for it) and not written through.
        let through = run(&tier, &mut applied, 0, 99, 2048);
        assert!(through.is_none(), "a hot row's larger version was refused");
        assert!(tier.stats().held_rows <= before);
        assert!(tier.bytes() <= 24 * 1024);
    }

    #[test]
    fn threads_offering_the_same_rows_leave_the_accounting_exact() {
        let tier = std::sync::Arc::new(HotTier::build(cfg(1 << 20), 2));
        let handles: Vec<_> = (0..4u8)
            .map(|t| {
                let tier = tier.clone();
                std::thread::spawn(move || {
                    let mut sink = |_: HeldOp| -> Result<(), Infallible> { Ok(()) };
                    for round in 0..300u32 {
                        for id in 0..40u32 {
                            let _ = tier.offer(op(id, t.wrapping_add(round as u8), 512), &mut sink);
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(tier.bytes() <= 1 << 20);
        let mut applied = Applied::new();
        drain(&tier, &mut applied);
        assert!(tier.is_empty());
        assert_eq!(tier.bytes(), tier.stats().sketch_bytes);
    }
}
