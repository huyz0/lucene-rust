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
//!   [`HotTierConfig::min_updates_per_second`] (by default, more than once a second). Heat lives in a fixed counting sketch
//!   (two rows of bytes, the smaller count counts), not per row, so a row
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
//!   older operation never lands after a newer one for the same row.
//!
//! - **Errors.** `apply` failing (a document limit, an I/O error) fails the
//!   cut that drains, as a failing write fails; the failed operation is not
//!   retried. Callers treat a writer error as fatal, as the engine does.
//! - **Lock order.** `apply` runs under a shard lock, so it may take the writer's
//!   own locks; nothing holding a writer lock may take a shard lock.
//! - **Memory is an estimate.** Held bytes are the operation's stored values
//!   and field buffers as measured, without allocator or map overhead.
//!
//! The tier changes when an update becomes searchable (at the next cut, not
//! at its acknowledgement) and which intermediate versions exist in the
//! segments; callers whose history must hold every sequence number (soft
//! deletes retention) keep the tier off for rows that history covers.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use lucene_codecs::stored_fields::FieldValue;

use crate::buffered_updates::{DocValuesUpdate, Term};
use crate::index_writer::{DocumentVector, ExplicitDocument};

const SHARDS: usize = 64;
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
        }
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
    // A map slot, the key, the op's own fields.
    let mut n = 160usize
        .saturating_add(term.bytes.capacity())
        .saturating_add(term.field.capacity())
        .saturating_add(soft.len().saturating_mul(96));
    for d in docs {
        n = n
            .saturating_add(std::mem::size_of::<ExplicitDocument>())
            .saturating_add(d.fields.ram_bytes());
        for f in &d.stored {
            let value = match &f.value {
                FieldValue::String(s) => s.capacity(),
                FieldValue::Shared(s) => s.len(),
                FieldValue::Binary(b) => b.capacity(),
                _ => 0,
            };
            n = n.saturating_add(48).saturating_add(value);
        }
    }
    if let Some(v) = vectors {
        for dv in v.iter().flatten() {
            let bytes = match &dv.value {
                crate::index_writer::VectorValue::Float32(v) => v.len().saturating_mul(4),
                crate::index_writer::VectorValue::Byte(v) => v.len(),
            };
            n = n.saturating_add(64).saturating_add(bytes);
        }
    }
    n
}

/// What [`HotTier::offer`] decided.
pub enum Offer {
    /// The tier holds the operation (any older one for the row is dropped).
    Held,
    /// Not held: apply it now, as without the tier.
    Through(HeldOp),
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

struct Shard {
    rows: HashMap<Vec<u8>, HeldOp>,
}

/// See the module documentation.
pub struct HotTier {
    cfg: HotTierConfig,
    sketch: Vec<AtomicU8>,
    mask: usize,
    shards: Vec<Mutex<Shard>>,
    bytes: AtomicUsize,
    peak: AtomicUsize,
    rows: AtomicUsize,
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
        let mut slots = 1usize << 20;
        while slots > 64 && slots.saturating_mul(2) > cfg.max_bytes / 8 {
            slots /= 2;
        }
        let sketch = (0..slots.saturating_mul(2))
            .map(|_| AtomicU8::new(0))
            .collect();
        let shards = (0..SHARDS)
            .map(|_| {
                Mutex::new(Shard {
                    rows: HashMap::new(),
                })
            })
            .collect();
        let sketch_bytes = slots.saturating_mul(2);
        HotTier {
            admit,
            cfg,
            sketch,
            mask: slots.saturating_sub(1),
            shards,
            bytes: AtomicUsize::new(sketch_bytes),
            peak: AtomicUsize::new(sketch_bytes),
            rows: AtomicUsize::new(0),
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
        }
    }

    /// Counts one update of the row hashed `h` and returns its heat.
    fn bump(&self, h: u64) -> u8 {
        let n = self.mask.saturating_add(1);
        let a = (h as usize) & self.mask;
        let b = n.saturating_add(((h >> 32) as usize) & self.mask);
        let inc = |i: usize| -> u8 {
            let cell = &self.sketch[i];
            let v = cell.load(Ordering::Relaxed);
            if v < u8::MAX {
                cell.store(v.saturating_add(1), Ordering::Relaxed);
                v.saturating_add(1)
            } else {
                v
            }
        };
        inc(a).min(inc(b))
    }

    fn maybe_age(&self) {
        if self.ticks.fetch_add(1, Ordering::Relaxed) & 1023 != 0 {
            return;
        }
        let mut at = lock(&self.aged_at);
        if at.elapsed() >= self.cfg.age_every {
            *at = Instant::now();
            drop(at);
            for c in &self.sketch {
                let v = c.load(Ordering::Relaxed);
                if v != 0 {
                    c.store(v >> 1, Ordering::Relaxed);
                }
            }
        }
    }

    fn rows_down(&self) {
        let _ = self
            .rows
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |r| {
                Some(r.saturating_sub(1))
            });
    }

    fn release(&self, n: usize) {
        let _ = self
            .bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |b| {
                Some(b.saturating_sub(n))
            });
    }

    /// Whether `n` more bytes fit, reserving them when they do.
    fn reserve(&self, n: usize) -> bool {
        let mut cur = self.bytes.load(Ordering::Acquire);
        loop {
            if cur.saturating_add(n) > self.cfg.max_bytes {
                return false;
            }
            match self.bytes.compare_exchange_weak(
                cur,
                cur.saturating_add(n),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.peak
                        .fetch_max(cur.saturating_add(n), Ordering::Relaxed);
                    return true;
                }
                Err(c) => cur = c,
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
        self.maybe_age();
        let h = hash(&op.term.bytes);
        let heat = self.bump(h);
        op.heat = heat;
        let shard = (h as usize) % SHARDS;
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
                    *old = op;
                    self.coalesced.fetch_add(1, Ordering::Relaxed);
                    return Ok(Offer::Held);
                }
                // No room for the larger version: the older one is dropped
                // (this one replaces it) and this one goes through.
                if let Some(old) = s.rows.remove(op.term.bytes.as_slice()) {
                    self.release(old.bytes);
                    self.rows_down();
                }
                self.refused.fetch_add(1, Ordering::Relaxed);
                self.through.fetch_add(1, Ordering::Relaxed);
                return Ok(Offer::Through(op));
            }
        }
        if heat < self.admit {
            self.through.fetch_add(1, Ordering::Relaxed);
            return Ok(Offer::Through(op));
        }
        // A hot row with no held operation: find it a place.
        op.measure();
        if op.bytes > self.cfg.max_bytes.saturating_sub(self.sketch.len()) {
            self.refused.fetch_add(1, Ordering::Relaxed);
            self.through.fetch_add(1, Ordering::Relaxed);
            return Ok(Offer::Through(op));
        }
        // After a failed search for a row to spill, the next hot rows skip
        // the search for a while: a full tier of equally hot rows would
        // otherwise pay it on every update.
        let tick = self.ticks.load(Ordering::Relaxed);
        let mut tries = 0usize;
        while !self.reserve(op.bytes) {
            tries = tries.saturating_add(1);
            if tries > 4
                || tick < self.no_spill_until.load(Ordering::Relaxed)
                || !self.spill_one(heat, apply)?
            {
                if tries <= 4 {
                    self.no_spill_until
                        .store(tick.saturating_add(NO_SPILL_TICKS), Ordering::Relaxed);
                }
                self.refused.fetch_add(1, Ordering::Relaxed);
                self.through.fetch_add(1, Ordering::Relaxed);
                return Ok(Offer::Through(op));
            }
        }
        let key = op.term.bytes.clone();
        let mut s = lock(&self.shards[shard]);
        if let Some(old) = s.rows.insert(key, op) {
            // Raced with another thread's insert of this row: the older one
            // was replaced, and its bytes are given back.
            drop(s);
            self.release(old.bytes);
            self.coalesced.fetch_add(1, Ordering::Relaxed);
        } else {
            self.rows.fetch_add(1, Ordering::AcqRel);
            self.admitted.fetch_add(1, Ordering::Relaxed);
        }
        Ok(Offer::Held)
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
        let bytes = op.bytes;
        // Applied with the shard held, counted until it is applied.
        let r = apply(op);
        self.release(bytes);
        r?;
        self.spilled.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    /// Applies every held operation.
    pub fn drain<E>(&self, apply: &mut dyn FnMut(HeldOp) -> Result<(), E>) -> Result<usize, E> {
        self.drains.fetch_add(1, Ordering::Relaxed);
        let mut n = 0usize;
        for shard in &self.shards {
            let mut s = lock(shard);
            let keys: Vec<Vec<u8>> = s.rows.keys().cloned().collect();
            for k in keys {
                if let Some(op) = s.rows.remove(&k) {
                    self.rows_down();
                    let bytes = op.bytes;
                    let r = apply(op);
                    self.release(bytes);
                    r?;
                    n = n.saturating_add(1);
                }
            }
        }
        self.drained.fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
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
            sketch_bytes: self.sketch.len(),
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
            Ok(Offer::Held) => None,
            Ok(Offer::Through(op)) => Some(op),
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
            for c in &tier.sketch {
                c.store(c.load(Ordering::Relaxed) >> 1, Ordering::Relaxed);
            }
        }
        assert!(a_held >= 9, "{a_held}");
        assert_eq!(b_held, 0);
    }

    #[test]
    fn the_sketch_is_counted_and_shrinks_with_a_small_budget() {
        let big = HotTier::build(cfg(64 << 20), 3);
        assert_eq!(big.stats().sketch_bytes, 2 << 20);
        let small = HotTier::build(cfg(64 << 10), 3);
        assert!(small.stats().sketch_bytes <= (64 << 10) / 4);
        assert_eq!(small.bytes(), small.stats().sketch_bytes);
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
