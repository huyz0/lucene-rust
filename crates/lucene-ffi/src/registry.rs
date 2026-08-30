//! Process-wide handle registries. A JNI caller has no way to hand this
//! crate a Rust reference across calls (see the `ffi-safety` skill), so
//! every opened `Directory`/segment/result set lives in one of these global
//! [`SlotMap`]s, guarded by an `RwLock` (JNI callers may call from more than
//! one JVM thread) behind a `u64` handle the caller carries between calls.
//!
//! **Why `RwLock`, not `Mutex`** (changed in the M2 sweep): a query holds
//! its registry guard for the *whole* search -- `SlotMap::get` hands back a
//! `&SegmentHandle`/`&DirectoryReaderHandle` borrowed from the guard, and
//! the term dictionary, postings bytes and live docs are read straight out
//! of it while the collector runs. Under a `Mutex` that made every search
//! from every JVM thread mutually exclusive: an N-core node ran one query at
//! a time, no matter how many segments or threads it had, which also made
//! `ffi_search_*_multi_segment_concurrent`'s rayon fan-out pointless (the
//! caller's threads queued outside the boundary instead). Reads are the
//! overwhelming majority of registry traffic and are genuinely shared
//! (`&SlotMap` -> `&T`), so [`read_recovering`] takes a shared guard and
//! only open/close/mutate paths ([`lock_recovering`], `insert_checked`/
//! `remove`/`get_mut`) take the exclusive one. Handle validation is
//! unchanged -- the generation and registry tags still gate every access,
//! and a handle cannot be closed *while* another thread is using it because
//! `ffi_close_*` needs the write guard the in-flight read guard holds off.
//!
//! Measured with `benchmarks/rust-runner/src/ffi_overhead.rs`'s section D
//! (`ffi_search_term_query_multi_segment` against
//! `fixtures/data/blocktree_index`, 400k calls, 4 threads), building the
//! same binary twice with only `read_recovering`'s guard kind changed and
//! running the two alternately:
//!
//! | | 4-thread wall | vs 1 thread |
//! |---|---|---|
//! | exclusive guard for every lookup | 2164 ns/call | 0.29x |
//! | shared guard for lookups | 347 ns/call | 1.17x |
//!
//! Adding threads made the exclusive version 3.5x *slower*; the shared one
//! gets faster. 6.2x the four-thread throughput for the same work.
//!
//! Those figures are from an otherwise-idle machine. Repeated later at load
//! average ~2 (other builds running), the same paired A/B gives 2712-2799
//! ns/call vs 976-1075 ns/call -- 2.7x rather than 6.2x, because fewer cores
//! are actually free for the four threads to spread over. The direction and
//! the order of magnitude are robust; the exact ceiling is a property of how
//! much of the machine the caller owns.
//!
//! That change left one ceiling behind, which the follow-up batch
//! `c13-ffi-surface` then removed. Even at its best the `RwLock` measurement
//! was 1.17x rather than ~4x, because each call still took an *exclusive*
//! guard twice on the results registries -- once to insert the results handle
//! the query produced, once for the caller's `ffi_close_*` -- and this
//! fixture's query only takes ~0.4us, so those two acquisitions dominated
//! what was left.
//!
//! **Sharded results registries** (c13): the six results registries are now
//! [`Sharded`] -- [`crate::handle::SHARDS`] independent
//! `RwLock<SlotMap<T>>`s each, with the issuing shard recorded in four bits
//! of the handle itself (see `handle.rs`'s module doc). An insert goes to the
//! calling thread's own sticky shard, so N threads inserting concurrently
//! take N *different* locks; a lookup or close reads the shard straight out
//! of the handle and touches only that one. Nothing about handle validation
//! changes: the tag, the shard and the generation must all match the slot's
//! current occupant, and a handle carrying a wrong shard id simply misses
//! (the shard field is masked, so every `u64` names a real shard -- there is
//! no out-of-range panic to reach).
//!
//! Measured with the same paired A/B methodology b15 used and the same
//! section D of `benchmarks/rust-runner/src/ffi_overhead.rs` (400k
//! `ffi_search_term_query_multi_segment` calls against
//! `fixtures/data/blocktree_index`): the same binary built twice with one
//! line changed -- [`Sharded::my_shard`]'s round-robin modulus, `% SHARDS`
//! against `% 1`, so the "before" build puts every insert on one shard and is
//! otherwise byte-identical -- then run alternately. Three rounds per fan-out
//! on a 20-core machine at load average ~5:
//!
//! | caller threads | unsharded (ns/call wall) | sharded | speedup |
//! |---|---|---|---|
//! | 1 | 1054 / 1059 / 1119 | 1048 / 1056 / 1055 | 1.00x |
//! | 4 | 571 / 695 | 558 / 620 | ~1.1x |
//! | 16 | 487 / 494 / 490 | 235 / 232 / 245 | **2.08x** |
//! | 32 | 494 / 480 | 222 / 226 | **2.19x** |
//!
//! Read the 1-thread row first: the shard lookup is a shift and a mask, so it
//! costs nothing, and the "FFI boundary (C - B)" line is unchanged too
//! (560-589 ns against 579-593 ns, within run-to-run noise).
//!
//! **The fan-out is the whole story, and it is why b15's 1.17x looked like a
//! ceiling.** Section D's default fan-out is 4, and four threads on a 20-core
//! box barely contend for anything -- which is why the same change is worth
//! only ~10% there. At 16 and 32 threads, the fan-outs an OpenSearch node's
//! search thread pool actually runs at (it is sized to the core count), the
//! single results-registry lock is the binding constraint and removing it
//! doubles throughput. Expressed as scaling rather than as a ratio: over its
//! own 1-thread baseline the unsharded build reaches 2.15x on 16 threads and
//! the sharded one 4.46x.
//!
//! The bench takes `FFI_OVERHEAD_THREADS` to reach those fan-outs; the
//! default stays 4 so the number stays comparable with b15's.
//!
//! The non-results registries (`directories`/`segments`/`directory_readers`/
//! `writers`/`vectors`) are deliberately **not** sharded: they are written
//! only by open/close, which happens once per reader lifetime rather than
//! twice per query, so sharding them would add a shard-selection branch to
//! the hot *read* path (every search validates a segment or reader handle)
//! to relieve contention that is not there.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{OnceLock, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use lucene_codecs::blocktree::BlockTreeFields;
use lucene_codecs::doc_values::DocValuesMeta;
use lucene_codecs::field_infos::FieldInfos;
use lucene_codecs::norms::Norms;
use lucene_search::ScoreDoc;
use lucene_store::directory::FsDirectory;
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::handle::{shard_of, RegistryTag, SlotMap, SHARDS};

/// Takes `lock`'s **exclusive** guard -- the one `insert_checked`/`remove`/
/// `get_mut` need -- recovering the inner value if a previous holder of this
/// same lock panicked instead of propagating the poison as a second panic.
///
/// **Why this is sound for these three registries specifically**: every
/// mutation [`SlotMap`] performs (`insert`/`remove`) is a single, non-panicking
/// sequence of `Vec`/field writes with no possibility of observing a torn
/// write from within this crate (`insert`/`remove` never call into
/// arbitrary/foreign code that could panic mid-mutation -- see `handle.rs`).
/// A panic that poisons one of these mutexes therefore always happens while
/// the guard is held read-only (e.g. mid-query, borrowing a `&SegmentHandle`
/// while decoding adversarial bytes) or entirely outside any `SlotMap`
/// method body, never mid-`insert`/mid-`remove` -- so the slotmap itself is
/// never left in a half-written state, only "the operation using its
/// borrowed contents failed." Recovering it is safe: every subsequent access
/// still goes through the same generation-tag (and, since the handle-tag
/// fix, registry-tag) validation, so a wrong/stale/cross-registry handle is
/// still rejected as before -- recovery only prevents *this* mutex from
/// wedging every future call into a permanent [`crate::error::FfiStatus::Panic`]
/// (defeating `catch_unwind`'s purpose of isolating one bad call), it does
/// not weaken any handle-validation guarantee.
pub(crate) fn lock_recovering<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

/// Takes `lock`'s **shared** guard, for the read-only handle lookups that
/// make up almost every call in this crate (see this module's doc comment
/// for why searches must not serialize on each other). Same
/// poison-recovering behavior as [`lock_recovering`], for the same reasons.
pub(crate) fn read_recovering<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

/// A registry split into [`SHARDS`] independent `RwLock<SlotMap<T>>`s, with
/// the issuing shard packed into every handle it hands out -- see this
/// module's doc comment for the measurement that motivated it, and
/// `handle.rs`'s for the bit layout.
///
/// **Insert picks the calling thread's own shard.** Each OS thread claims one
/// shard the first time it inserts anything anywhere (a single
/// `AtomicUsize::fetch_add` round-robin, then cached in a `Cell` thread-local),
/// so a fan-out of N <= [`SHARDS`] JVM threads inserting concurrently
/// contends on N *different* locks instead of one. It is a hint, never a
/// correctness input: which shard a handle came from is read back out of the
/// handle, so a handle created on one thread and closed on another finds its
/// slot exactly as before.
///
/// **Lookup and close take only the owning shard's lock**, read out of the
/// handle with a shift and a mask. The shard field is masked to its four
/// bits, so any `u64` -- including a garbage or hostile one -- names a real
/// shard and is then rejected by that shard's ordinary tag/generation check;
/// there is no index-out-of-range path to reach.
pub(crate) struct Sharded<T> {
    shards: Vec<RwLock<SlotMap<T>>>,
}

/// Round-robin source for [`Sharded::my_shard`]; one counter for the whole
/// process, read once per thread rather than once per insert.
static NEXT_SHARD: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// This thread's sticky shard index, `usize::MAX` until claimed.
    static MY_SHARD: Cell<usize> = const { Cell::new(usize::MAX) };
}

impl<T> Sharded<T> {
    fn new(tag: RegistryTag) -> Self {
        Sharded {
            shards: (0..SHARDS)
                .map(|i| RwLock::new(SlotMap::new_shard(tag, i)))
                .collect(),
        }
    }

    /// The shard this thread inserts into. Claimed once per thread and then
    /// reused, so the common "one JVM worker thread runs many queries" shape
    /// keeps hitting one uncontended lock.
    fn my_shard() -> usize {
        MY_SHARD.with(|cell| {
            let current = cell.get();
            if current != usize::MAX {
                return current;
            }
            let claimed = NEXT_SHARD.fetch_add(1, Ordering::Relaxed) % SHARDS;
            cell.set(claimed);
            claimed
        })
    }

    /// [`SlotMap::insert_checked`] on this thread's own shard. Returns the
    /// packed handle, which carries that shard's id.
    pub(crate) fn insert_checked(&self, value: T) -> Result<u64, crate::error::FfiStatus> {
        lock_recovering(&self.shards[Self::my_shard()]).insert_checked(value)
    }

    /// The **shared** guard of the shard `handle` was issued by -- the
    /// read-only lookup path. Call `.get(handle)` on the result.
    pub(crate) fn read(&self, handle: u64) -> RwLockReadGuard<'_, SlotMap<T>> {
        read_recovering(&self.shards[shard_of(handle)])
    }

    /// The **exclusive** guard of the shard `handle` was issued by -- the
    /// close path. Call `.remove(handle)` on the result.
    pub(crate) fn write(&self, handle: u64) -> RwLockWriteGuard<'_, SlotMap<T>> {
        lock_recovering(&self.shards[shard_of(handle)])
    }
}

/// One opened segment's decoded term dictionary plus the raw bytes of
/// whichever postings files were opened alongside it. `postings::DocInput`/
/// `PosInput`/`PayInput` all borrow from a byte slice (see
/// `lucene-codecs/src/postings.rs`), so this struct owns the bytes and a
/// fresh `DocInput::open(&self.doc, ...)` etc. is (cheaply -- header/footer
/// checks only) reconstructed per query call rather than stored as a
/// self-referential field.
pub struct SegmentHandle {
    pub fields: BlockTreeFields,
    pub doc_bytes: Option<Vec<u8>>,
    pub pos_bytes: Option<Vec<u8>>,
    /// This segment's whole `.pay` (payloads/offsets) file, opened by
    /// `ffi_open_segment`'s optional `pay_name` parameter (M2 sweep batch
    /// `c13-ffi-surface`, closing b15's recorded deferral). `None` when the
    /// caller opened no `.pay`, which is correct for a field whose
    /// `IndexOptions` carry neither payloads nor offsets, and which
    /// `search_phrase_query` surfaces as a clean error for a field that needs
    /// one. Owned bytes for the same self-referential reason `doc_bytes`/
    /// `pos_bytes` are (see this struct's own doc comment).
    pub pay_bytes: Option<Vec<u8>>,
    pub segment_id: [u8; 16],
    pub segment_suffix: String,
    pub max_doc: i32,
    /// This segment's parsed `.fnm` (task #30) -- kept around (rather than
    /// dropped once `blocktree::open` has consumed it) so a scored query can
    /// map a field *name* (the only thing a caller passes over the C ABI) to
    /// the field *number* `norms`/`NormsEntry` are keyed by. `BlockTreeFields`
    /// has no such name->number mapping of its own (see `blocktree.rs`), so
    /// this is the only place left to look it up from.
    pub field_infos: FieldInfos,
    /// This segment's whole `.nvd` (norms data) file, opened by
    /// `ffi_open_segment`'s optional `nvd_name`/`nvm_name` parameters (task
    /// #30) -- `None` when the caller opened the segment without norms
    /// (every scored query then falls back to
    /// `lucene_search::similarity::UNNORMED_FIELD_LENGTH`, the same
    /// documented approximation `FieldNorms`'s absence already means
    /// elsewhere in this port).
    pub norms_data: Option<Vec<u8>>,
    /// This segment's parsed `.nvm` (norms metadata) -- one [`Norms`] entry
    /// per field that has norms, looked up by field number via
    /// `field_infos` above. `Some` iff `norms_data` is `Some` (both come from
    /// the same `nvd_name`/`nvm_name is-null` check in `ffi_open_segment`).
    pub norms: Option<Norms>,
    /// This segment's whole `.dvd` (doc-values data) file, opened by
    /// `ffi_open_segment`'s optional `dvm_name`/`dvd_name`/`dv_suffix`
    /// parameters (task #40) -- `None` when the caller opened the segment
    /// without doc values, in which case `ffi_sort_by_doc_value`/
    /// `ffi_sort_by_multi_valued_doc_value` return [`crate::error::FfiStatus::InvalidArgument`]
    /// (there is nothing to sort by, unlike norms' "fall back to a constant"
    /// story -- a sort with no values for its field has no sensible
    /// fallback).
    pub dv_data: Option<Vec<u8>>,
    /// This segment's parsed `.dvm` (doc-values metadata) -- one entry per
    /// doc-values field, looked up by field number via `field_infos` above,
    /// same pattern as `norms`. `Some` iff `dv_data` is `Some`.
    ///
    /// **This is the *base* column.** A field that has taken an
    /// `IndexWriter.updateNumericDocValue`/`updateBinaryDocValue` is no
    /// longer served from here -- see [`Self::dv_generations`] and
    /// [`Self::doc_values_for_field`], which is what every read path must go
    /// through.
    pub dv_meta: Option<DocValuesMeta>,
    /// Per-field doc-values **update generations**, attached by
    /// [`crate::segment::ffi_segment_add_doc_values_generation`] -- Java's
    /// `SegmentDocValuesProducer.dvProducersByField`.
    ///
    /// A doc-values update rewrites one field's whole column into a
    /// generation-suffixed `.dvm`/`.dvd` pair and leaves the base pair
    /// untouched, so after `updateNumericDocValue` the base column still
    /// holds that field's *superseded* values. Reading the base pair for a
    /// field that has a generation is not a missing feature, it is a wrong
    /// answer: a sort or a facet comes back with the pre-update values and
    /// nothing structural notices.
    pub dv_generations: Vec<DocValuesGenerationColumn>,
    /// This segment's whole `.kdm`/`.kdi`/`.kdd` (BKD points meta/index/data)
    /// files, opened together by `ffi_open_segment`'s optional
    /// `kdm_name`/`kdi_name`/`kdd_name` parameters (Points range query FFI
    /// exposure) -- `None` when the caller opened the segment without points
    /// data, in which case `ffi_search_points_range` returns
    /// [`crate::error::FfiStatus::InvalidArgument`] (there is nothing to
    /// search, same "no sensible fallback" reasoning `dv_data`'s own doc
    /// comment gives). Stored as owned bytes rather than an already-open
    /// `lucene_codecs::points::PointsReader` for the same self-referential
    /// reason `doc_bytes`/`pos_bytes` are raw bytes, not an already-open
    /// `DocInput`/`PosInput` -- see this struct's own doc comment; a fresh
    /// `PointsReader::open` is reconstructed per query call (cheap -- header/
    /// footer checks only, `PointsReader::open` does no per-point decoding
    /// up front).
    pub points_data: Option<(Vec<u8>, Vec<u8>, Vec<u8>)>,
    /// This segment's decoded `.liv` (live docs / deletions) bitset, attached
    /// by [`crate::segment::ffi_segment_set_live_docs`] -- bit `d` set means
    /// doc `d` is live. `None` means "this segment has no deletions"
    /// (`SegmentCommitInfo.getDelGen() == -1`), which is also what every
    /// `lucene_search` function documents `live_docs: None` to mean, so an
    /// undeleted segment costs nothing.
    ///
    /// **Not a parameter of [`crate::segment::ffi_open_segment`]**: unlike
    /// every other optional file that function opens, a `.liv`'s name is
    /// *generation-suffixed* (`_0_1.liv`) and needs two extra scalars
    /// (`del_gen`, `del_count`) that `live_docs::parse` validates the file
    /// against, and it is the one segment file that changes without the
    /// segment itself being rewritten. Attaching it through its own
    /// additive call keeps `ffi_open_segment`'s already-30-parameter C
    /// signature stable for existing callers and lets a caller refresh
    /// deletions on an open segment handle without reopening its term
    /// dictionary and postings.
    pub live_docs: Option<FixedBitSet>,
}

/// One field's doc-values **update generation**: the rewritten `.dvm`/`.dvd`
/// pair `FieldInfo.docValuesGen` points at, holding that field's current
/// column. Java's per-field `SegmentDocValuesProducer` entry.
pub struct DocValuesGenerationColumn {
    pub field_number: i32,
    pub meta: DocValuesMeta,
    pub data: Vec<u8>,
}

impl SegmentHandle {
    /// The `(meta, data)` pair holding **`field_number`'s current column** --
    /// `SegmentDocValuesProducer.dvProducersByField`. A field that has taken
    /// a doc-values update is served from the generation attached by
    /// [`crate::segment::ffi_segment_add_doc_values_generation`]; every other
    /// field falls through to the base `.dvm`/`.dvd`.
    ///
    /// **Every doc-values read on this handle must go through here**, never
    /// through `dv_meta`/`dv_data` directly: those are the base pair, and for
    /// an updated field the base pair is the superseded column. The pair is
    /// returned *together* so an entry can never be decoded against the wrong
    /// file -- which is the shape of the bug this replaced (the entry was
    /// looked up in one place and the bytes fetched in another).
    ///
    /// `None` means the segment was opened without doc values and has no
    /// generation for this field either, i.e. there is nothing to read.
    pub fn doc_values_for_field(&self, field_number: i32) -> Option<(&DocValuesMeta, &[u8])> {
        if let Some(generation) = self
            .dv_generations
            .iter()
            .find(|g| g.field_number == field_number)
        {
            return Some((&generation.meta, &generation.data));
        }
        match (self.dv_meta.as_ref(), self.dv_data.as_deref()) {
            (Some(meta), Some(data)) => Some((meta, data)),
            _ => None,
        }
    }
}

/// A completed unscored query's collected, ascending, live doc IDs -- read
/// back via `ffi_results_len`/`ffi_results_copy`, then released via
/// `ffi_close_results`.
pub struct ResultsHandle {
    pub docs: Vec<i32>,
}

/// A completed *scored* query's `(doc_id, score)` hits, kept in `TopDocsCollector`
/// order (best-first, ties broken by lower doc ID -- see `collector.rs`'s
/// `rank_order`) -- read back via `ffi_scored_results_len`/`ffi_scored_results_copy`,
/// then released via `ffi_close_scored_results`.
///
/// **Why a new registry/handle type instead of widening `ResultsHandle` with an
/// optional `Vec<f32>`**: `ResultsHandle` is a public, already-shipped shape read by
/// `ffi_results_len`/`ffi_results_copy`'s existing (unscored) contract -- adding an
/// optional scores field there would force every existing caller of the unscored
/// path to reason about a field that's always `None` for them, and would let a
/// caller accidentally call `ffi_results_copy` against a handle that was actually
/// populated by a scored query (or vice versa) since both would share one handle
/// type and one registry tag. A separate `ScoredResultsHandle`/`RegistryTag::
/// ScoredResults` keeps the two result shapes as distinct as the two collector
/// traits they come from (`Collector` vs `ScoringCollector`, see `collector.rs`'s
/// module doc for that same non-breaking-addition reasoning) -- a results handle
/// from the wrong search flavor is rejected by the registry-tag check before it
/// can be misread, exactly like a directory handle passed to a segment call is.
pub struct ScoredResultsHandle {
    pub hits: Vec<ScoreDoc>,
}

/// A completed doc-value sort's `(doc_id, value)` pairs (task #40, wrapping
/// `lucene_search::sort_by_numeric_doc_value`/`sort_by_multi_valued_doc_value`)
/// -- ascending by value, ties broken by ascending doc ID (see those
/// functions' own doc comments) -- read back via
/// `ffi_sorted_results_len`/`ffi_sorted_results_copy`, then released via
/// `ffi_close_sorted_results`.
///
/// **Why a new registry/handle type instead of reusing `ScoredResultsHandle`**:
/// a sort result's second element is the actual doc-value used for
/// ordering (an arbitrary `i64` -- a raw NUMERIC value, or a SORTED_NUMERIC/
/// SORTED_SET reduced value/ordinal), not a BM25 `f32` score -- a different
/// wire type (`i64` vs `f32`), a different scale/meaning a caller must not
/// confuse with a relevance score, and a different collector-less code path
/// (a plain sort over an already-known candidate set, not a
/// `TopDocsCollector` scored search, see `lucene-search`'s
/// `doc_value_query.rs` module doc for that design rationale). Keeping this
/// as its own registry/tag means a scored-results handle can never be
/// accidentally passed to `ffi_sorted_results_copy` (or vice versa) and
/// misread as the wrong element type -- exactly the same reasoning
/// `ScoredResultsHandle`'s own doc comment gives for not widening
/// `ResultsHandle`.
pub struct SortedResultsHandle {
    pub pairs: Vec<(i32, i64)>,
}

/// An opened `DirectoryReader` (task #51, wrapping task #45's
/// `lucene_search::directory_reader::DirectoryReader`): every segment listed
/// in a commit's `segments_N`, already opened with each segment's `doc_base`
/// computed -- read back via [`crate::directory_reader::ffi_search_term_query_multi_segment`]/
/// [`crate::directory_reader::ffi_search_boolean_query_multi_segment`] (task #41's
/// multi-segment fan-out/merge), released via
/// [`crate::directory_reader::ffi_close_directory_reader`].
///
/// **Why its own registry/tag instead of reusing `Directory`**: a
/// `DirectoryReader` owns a whole tree of already-opened, already-decoded
/// segment readers (term dictionaries, postings byte buffers, live docs) --
/// a fundamentally different lifetime and size class from `Directory`'s bare
/// `FsDirectory` (a filesystem root with no segment state at all) or
/// `Segment`'s single already-opened segment. Folding it into either would
/// let a directory/segment handle be silently accepted where a
/// `DirectoryReader` handle is expected (or vice versa) since they'd share
/// one registry tag -- exactly the cross-registry confusion `RegistryTag`
/// exists to rule out (see `handle.rs`'s module doc).
///
/// **No norms/doc-values plumbing**: task #45's `DirectoryReader` has no
/// `.nvm`/`.nvd`/`.dvm`/`.dvd` support at all (see that module's doc
/// comment) -- every multi-segment scored query built on this handle always
/// passes `norms: None` per segment, the same documented
/// `UNNORMED_FIELD_LENGTH` fallback `lucene_search`'s own scored functions
/// already use for a bare `None`, not a new gap introduced by this handle.
pub struct DirectoryReaderHandle {
    pub reader: lucene_search::directory_reader::DirectoryReader,
}

/// A completed SortedSet facet count's resolved `(ord, label, count)` triples
/// (Faceted search FFI exposure, wrapping `lucene_search::facets::facet_counts`/
/// `resolve_labels`/`top_n_facets`) -- read back via
/// `results_facets.rs`'s `ffi_facet_results_len`/`ffi_facet_results_copy`/
/// `ffi_facet_result_label`, then released via `ffi_close_facet_results`.
///
/// **Why a new registry/handle type instead of reusing `SortedResultsHandle`**:
/// a facet result's element is `(ord, label, count)` -- it carries a resolved
/// string label alongside a `u64` count, not a `(doc_id, value)` `i64` pair --
/// a different wire shape needing its own string-accessor
/// (`ffi_facet_result_label`) that a sorted-results handle has no equivalent
/// of. Keeping this as its own registry/tag means a sorted-results handle can
/// never be accidentally passed to a facet-results accessor (or vice versa)
/// and misread as the wrong element type -- exactly the same reasoning
/// `SortedResultsHandle`'s own doc comment gives for not widening
/// `ScoredResultsHandle`.
///
/// **NUMERIC range facet counts have no equivalent handle**: `ffi_range_facet_counts`
/// (also in `facets.rs`) writes counts directly into a caller-allocated
/// buffer instead -- every range's label is caller-supplied input, not
/// resolved from the index, so the caller already owns every label string
/// and there is nothing new to hand back behind a handle. See `facets.rs`'s
/// module doc for the full rationale.
pub struct FacetResultsHandle {
    pub facets: Vec<lucene_search::facets::FacetCount>,
}

/// A completed highlight fragment assembly's [`lucene_search::highlighter::Fragment`]s
/// (`highlighter.rs`'s `ffi_assemble_fragments`, wrapping
/// `lucene_search::highlighter::assemble_fragments`) -- read back via
/// `results_fragments.rs`'s `ffi_fragment_results_len`/`ffi_fragment_result_text`/
/// `ffi_fragment_result_matched_terms_len`/`ffi_fragment_result_matched_term`,
/// then released via `ffi_close_fragment_results`.
///
/// **Why a new registry/handle type instead of reusing `FacetResultsHandle`**:
/// a fragment carries a highlighted `text` string *and* a variable-length list
/// of `matched_terms` strings per element -- a two-level variable-length shape
/// none of this crate's existing handles have (`FacetResultsHandle`'s element
/// has exactly one string field, `label`). Keeping this as its own
/// registry/tag means a facet- or sorted-results handle can never be
/// accidentally passed to a fragment accessor (or vice versa) and misread --
/// the same reasoning every other handle type in this file already gives for
/// not widening an existing one.
pub struct FragmentResultsHandle {
    pub fragments: Vec<lucene_search::highlighter::Fragment>,
}

/// One node of a flattened `lucene_search::explain::Explanation` tree (Query
/// explain FFI exposure, `explain.rs`'s `ffi_explain_term_query`/
/// `ffi_explain_phrase_query`/`ffi_explain_boolean_query`, wrapping
/// `lucene_search::explain::explain_clause`) -- see `explain.rs`'s module doc
/// for the full flattening scheme (depth-first pre-order, root always index
/// `0`, each node's `children` a list of *indices into the same flat `Vec`*
/// rather than nested owned `Explanation`s).
pub struct ExplainNode {
    pub matched: bool,
    pub value: f32,
    pub description: String,
    pub children: Vec<usize>,
}

/// A completed explain call's flattened [`ExplainNode`] tree -- read back via
/// `results_explain.rs`'s `ffi_explain_node_value`/`ffi_explain_node_matched`/
/// `ffi_explain_node_description`/`ffi_explain_node_child_count`/
/// `ffi_explain_node_child_at`, then released via `ffi_close_explain_results`.
///
/// **Why a new registry/handle type instead of reusing `FragmentResultsHandle`**:
/// an explain node's shape (a `bool` + an `f32` + a `String` + a list of
/// *sibling node indices*, forming a recursive tree) has no correspondence to
/// a fragment's flat `text` + `matched_terms` list -- see this crate's other
/// handle doc comments for why each genuinely distinct result shape gets its
/// own registry/tag rather than widening an existing one (a fragment-results
/// handle must never be accidentally accepted by an explain accessor, or vice
/// versa).
pub struct ExplainResultsHandle {
    pub nodes: Vec<ExplainNode>,
}

/// An opened `lucene_index::index_writer::IndexWriter` (IndexWriter
/// commit/merge-policy FFI exposure, `writer.rs`'s `ffi_open_writer`) --
/// read back/mutated in place via `writer.rs`'s `ffi_writer_add_document`/
/// `ffi_writer_commit`/`ffi_writer_prepare_commit`/`ffi_writer_finish_commit`/
/// `ffi_writer_rollback`/`ffi_writer_set_merge_policy`, released via
/// `ffi_close_writer`.
///
/// **Why this owns a boxed `FsDirectory` alongside the writer, and why that
/// is sound**: [`lucene_index::index_writer::IndexWriter`] is generic over a
/// borrowed `&'d dyn Directory` -- there is no owned, 'static-lifetime
/// `IndexWriter` type this crate could otherwise put behind a plain handle
/// the way [`SegmentHandle`] holds owned bytes. This struct instead owns the
/// `FsDirectory` on the heap (`Box<FsDirectory>`, a stable address that does
/// not move even if this whole `WriterHandle` value is moved -- only the
/// `Box` pointer itself would move, not its heap allocation) and constructs
/// the writer against a reference into that same allocation, lifetime-erased
/// to `'static` via one contained `unsafe` block in `writer.rs` (see that
/// module's `open_writer_handle` for the exact `# Safety` argument). This is
/// sound as long as `dir` outlives every use of `writer`, which struct field
/// declaration order guarantees here: Rust drops a struct's fields in
/// declaration order, so `writer` (declared first, holding the borrow) is
/// always dropped before `dir` (declared last, owning the allocation the
/// borrow points into) -- the borrow is never live past its referent's
/// lifetime.
pub struct WriterHandle {
    pub writer: lucene_index::index_writer::IndexWriter<'static>,
    // Never read directly (the whole point of this field is to keep the
    // heap allocation `writer`'s borrow points into alive -- see this
    // struct's own doc comment), hence the `allow`.
    #[allow(dead_code)]
    pub dir: Box<FsDirectory>,
}

// SAFETY: `WriterHandle` is `!Send`/`!Sync` only because it holds a bare
// `&'static dyn Directory` trait object, which carries no `Send`/`Sync`
// bound the compiler can see -- not because anything in it is actually
// thread-hostile. What makes these impls sound:
//
// - `IndexWriter` (`lucene-index/src/index_writer.rs`) is a plain aggregate
//   of `Vec`s, `String`s, `SegmentInfos` and `Option`s. `lucene-index` is
//   `#![forbid(unsafe_code)]`, so it contains no `Cell`/`RefCell`/
//   `UnsafeCell`/raw pointer and has **no interior mutability at all**: every
//   mutation needs `&mut self`, which only the write guard can hand out.
//   Concurrent shared `&IndexWriter` reads are therefore data-race-free.
// - The `dyn Directory` it borrows is always the `Box<FsDirectory>` in this
//   same struct (see `open_writer_handle`), and `FsDirectory` is
//   file-path/file-handle state with no thread affinity -- nothing here
//   depends on thread-local state or a non-thread-safe primitive.
// - Aliasing/lifetime safety of the erased borrow is a separate argument,
//   given in full in this struct's own doc comment (stable `Box` address +
//   field declaration order).
//
// NOTE: this used to say "at most one thread ever touches a given
// `WriterHandle` at a time", which was true while the registries were
// `Mutex`es. It is **no longer true** -- `read_recovering(writers())` hands
// `&WriterHandle` to arbitrarily many threads at once (see
// `writer.rs`'s read-only accessors). The impls stay sound for the reason
// above (no interior mutability), not for that one.
unsafe impl Send for WriterHandle {}
// SAFETY: see the `Send` impl above. `RwLock<T>` requires `T: Send + Sync`
// to be `Sync` itself, which is why both impls are needed here; the
// "concurrent shared `&WriterHandle` is data-race-free" half of that
// argument is exactly what the shared read guard relies on.
unsafe impl Sync for WriterHandle {}

/// One segment's opened vector files: the `Lucene99FlatVectorsFormat`
/// `.vemf`/`.vec` pair and, optionally, the `Lucene99HnswVectorsFormat`
/// `.vem`/`.vex` graph -- see `vectors.rs`'s module doc.
///
/// Owns the bytes rather than an already-open `FlatVectorsReader`/
/// `HnswVectorsReader` for exactly the reason [`SegmentHandle`] owns its
/// postings bytes: both readers borrow from the buffer they were opened
/// over, so storing one here would make this struct self-referential. A
/// fresh reader is reconstructed per search call -- header/footer checks and
/// a field-entry walk, no vector data touched.
///
/// **Why its own registry/tag rather than fields on [`SegmentHandle`]**:
/// [`crate::segment::ffi_open_segment`] requires a term dictionary, and a
/// segment can carry vectors with no postings at all. See
/// [`crate::handle::RegistryTag::Vectors`].
pub struct VectorsHandle {
    /// This segment's `.fnm`, for the field-name -> field-number mapping the
    /// vector formats key their entries by -- the same role it plays in
    /// [`SegmentHandle::field_infos`].
    pub field_infos: FieldInfos,
    /// The whole `.vemf` (flat vectors metadata).
    pub vemf: Vec<u8>,
    /// The whole `.vec` (flat vectors data).
    pub vec: Vec<u8>,
    /// The whole `.vem`/`.vex` (HNSW graph metadata/index), or `None` when
    /// the caller opened no graph -- in which case every search takes
    /// `hnsw_vectors::search`'s exhaustive branch, which is exact.
    pub hnsw: Option<(Vec<u8>, Vec<u8>)>,
    pub segment_id: [u8; 16],
    /// The per-field codec suffix in the vector files' index headers.
    pub suffix: String,
    pub max_doc: i32,
    /// This segment's `.liv`, attached by
    /// [`crate::vectors::ffi_vectors_set_live_docs`]; bit `d` set means doc
    /// `d` is live. Same meaning as [`SegmentHandle::live_docs`].
    pub live_docs: Option<FixedBitSet>,
}

pub fn vectors() -> &'static RwLock<SlotMap<VectorsHandle>> {
    static REGISTRY: OnceLock<RwLock<SlotMap<VectorsHandle>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(SlotMap::new(RegistryTag::Vectors)))
}

pub fn directories() -> &'static RwLock<SlotMap<FsDirectory>> {
    static REGISTRY: OnceLock<RwLock<SlotMap<FsDirectory>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(SlotMap::new(RegistryTag::Directory)))
}

pub fn segments() -> &'static RwLock<SlotMap<SegmentHandle>> {
    static REGISTRY: OnceLock<RwLock<SlotMap<SegmentHandle>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(SlotMap::new(RegistryTag::Segment)))
}

pub fn results() -> &'static Sharded<ResultsHandle> {
    static REGISTRY: OnceLock<Sharded<ResultsHandle>> = OnceLock::new();
    REGISTRY.get_or_init(|| Sharded::new(RegistryTag::Results))
}

pub fn scored_results() -> &'static Sharded<ScoredResultsHandle> {
    static REGISTRY: OnceLock<Sharded<ScoredResultsHandle>> = OnceLock::new();
    REGISTRY.get_or_init(|| Sharded::new(RegistryTag::ScoredResults))
}

pub fn sorted_results() -> &'static Sharded<SortedResultsHandle> {
    static REGISTRY: OnceLock<Sharded<SortedResultsHandle>> = OnceLock::new();
    REGISTRY.get_or_init(|| Sharded::new(RegistryTag::SortedResults))
}

pub fn directory_readers() -> &'static RwLock<SlotMap<DirectoryReaderHandle>> {
    static REGISTRY: OnceLock<RwLock<SlotMap<DirectoryReaderHandle>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(SlotMap::new(RegistryTag::DirectoryReader)))
}

pub fn facet_results() -> &'static Sharded<FacetResultsHandle> {
    static REGISTRY: OnceLock<Sharded<FacetResultsHandle>> = OnceLock::new();
    REGISTRY.get_or_init(|| Sharded::new(RegistryTag::FacetResults))
}

pub fn fragment_results() -> &'static Sharded<FragmentResultsHandle> {
    static REGISTRY: OnceLock<Sharded<FragmentResultsHandle>> = OnceLock::new();
    REGISTRY.get_or_init(|| Sharded::new(RegistryTag::FragmentResults))
}

pub fn explain_results() -> &'static Sharded<ExplainResultsHandle> {
    static REGISTRY: OnceLock<Sharded<ExplainResultsHandle>> = OnceLock::new();
    REGISTRY.get_or_init(|| Sharded::new(RegistryTag::ExplainResults))
}

pub fn writers() -> &'static RwLock<SlotMap<WriterHandle>> {
    static REGISTRY: OnceLock<RwLock<SlotMap<WriterHandle>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(SlotMap::new(RegistryTag::Writer)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`lock_recovering`] must hand back the inner value after a writer
    /// panicked holding it, rather than propagating the poison as a second
    /// panic -- otherwise one bad call would wedge every future call on that
    /// registry, defeating `catch_unwind`'s isolation.
    #[test]
    fn lock_recovering_recovers_a_poisoned_write_guard() {
        let lock = RwLock::new(7i32);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut guard = lock.write().unwrap();
            *guard = 9;
            panic!("poison it");
        }));
        assert!(lock.is_poisoned());
        assert_eq!(*lock_recovering(&lock), 9);
        assert_eq!(*read_recovering(&lock), 9);
    }

    /// The point of the `RwLock`: two threads can hold read guards on the
    /// same registry at once. Under the old `Mutex` this would deadlock the
    /// test (each thread waiting for the other to release), which is exactly
    /// what every concurrent FFI search used to do to every other one.
    #[test]
    fn two_readers_hold_the_same_registry_concurrently() {
        static LOCK: OnceLock<RwLock<i32>> = OnceLock::new();
        let lock = LOCK.get_or_init(|| RwLock::new(5));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let b2 = std::sync::Arc::clone(&barrier);
        let other = std::thread::spawn(move || {
            let guard = read_recovering(lock);
            // Only proceeds if the *other* thread also got its read guard.
            b2.wait();
            *guard
        });
        let guard = read_recovering(lock);
        barrier.wait();
        assert_eq!(*guard, 5);
        assert_eq!(other.join().unwrap(), 5);
    }

    /// Every registry is its own lock, so opening a results handle from
    /// inside a query that is holding the segments registry cannot deadlock.
    #[test]
    fn each_registry_is_an_independent_lock() {
        let segments_guard = read_recovering(segments());
        let handle = results()
            .insert_checked(ResultsHandle { docs: vec![1, 2] })
            .unwrap();
        assert_eq!(results().read(handle).get(handle).unwrap().docs, vec![1, 2]);
        results().write(handle).remove(handle).unwrap();
        drop(segments_guard);
    }
    /// A sharded registry's handles must round-trip through *any* thread: the
    /// shard is read back out of the handle, so a results handle created on
    /// one thread and closed on another finds its slot.
    #[test]
    fn a_sharded_handle_created_on_one_thread_is_readable_on_another() {
        let handle = std::thread::spawn(|| {
            results()
                .insert_checked(ResultsHandle {
                    docs: vec![7, 8, 9],
                })
                .unwrap()
        })
        .join()
        .unwrap();
        assert_eq!(
            results().read(handle).get(handle).unwrap().docs,
            vec![7, 8, 9]
        );
        assert!(results().write(handle).remove(handle).is_some());
        assert!(results().read(handle).get(handle).is_none());
    }

    /// Inserts from many threads must land on more than one shard -- that is
    /// the entire point. Asserted on the handles' own shard field, so this
    /// checks the property the measurement depends on, not an implementation
    /// detail of the counter.
    #[test]
    fn concurrent_inserts_spread_across_shards() {
        let handles: Vec<u64> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..8)
                .map(|i| {
                    scope.spawn(move || {
                        results()
                            .insert_checked(ResultsHandle { docs: vec![i] })
                            .unwrap()
                    })
                })
                .collect();
            workers.into_iter().map(|w| w.join().unwrap()).collect()
        });
        let mut shards: Vec<usize> = handles.iter().copied().map(shard_of).collect();
        shards.sort_unstable();
        shards.dedup();
        assert!(
            shards.len() > 1,
            "8 threads must not all land on one shard: {shards:?}"
        );
        for h in handles {
            assert!(results().write(h).remove(h).is_some());
        }
    }

    /// A handle whose shard bits have been tampered with must miss, exactly
    /// as a tampered generation or tag does -- never index out of range, and
    /// never alias a live entry in another shard.
    #[test]
    fn a_handle_with_the_wrong_shard_bits_is_rejected() {
        let handle = results()
            .insert_checked(ResultsHandle { docs: vec![1] })
            .unwrap();
        let real_shard = shard_of(handle);
        for candidate in 0..SHARDS {
            if candidate == real_shard {
                continue;
            }
            // Rewrite only the shard field.
            let tampered = (handle & !(0xF << 52)) | ((candidate as u64) << 52);
            assert!(
                results().read(tampered).get(tampered).is_none(),
                "shard {candidate} accepted a handle issued by shard {real_shard}"
            );
        }
        // Every possible shard field names a real shard, so even a fully
        // garbage handle is a miss rather than a panic.
        assert!(results().read(u64::MAX).get(u64::MAX).is_none());
        assert!(results().write(handle).remove(handle).is_some());
    }

    /// Two different sharded registries stay independent: a handle from one
    /// is rejected by the other on its registry tag, shard bits or not.
    #[test]
    fn sharded_registries_do_not_accept_each_other_s_handles() {
        let r = results()
            .insert_checked(ResultsHandle { docs: vec![1] })
            .unwrap();
        let s = scored_results()
            .insert_checked(ScoredResultsHandle { hits: Vec::new() })
            .unwrap();
        assert!(scored_results().read(r).get(r).is_none());
        assert!(results().read(s).get(s).is_none());
        assert!(results().write(r).remove(r).is_some());
        assert!(scored_results().write(s).remove(s).is_some());
    }
}
