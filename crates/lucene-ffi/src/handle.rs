//! A minimal generation-tagged slotmap for opaque `u64` handles crossing the
//! FFI boundary — see the `ffi-safety` skill's "opaque handles only" rule.
//! No Rust pointer or reference is ever handed to a caller; every exported
//! `lucene-ffi` function that "returns a handle" really returns one of these
//! packed `u64`s, and every function that "takes a handle" looks it up here
//! first, so a stale/unknown/closed handle is a lookup miss (an error code),
//! never a dereference of freed memory.
//!
//! **Why hand-rolled instead of reusing a crate**: this is FFI-specific
//! plumbing (pack/unpack into a single `u64` a JNI `long` can carry, not a
//! general in-process slotmap need any other crate in this workspace has —
//! `lucene-util`/`lucene-store` have no analogous "many opaque handles,
//! caller-driven open/close lifecycle" requirement), so it lives here rather
//! than in `lucene-util`.
//!
//! **Encoding**: a handle packs a 24-bit slot index in the low bits, a
//! 28-bit generation counter above it, a 4-bit *shard* id above that, and an
//! 8-bit registry-type tag in the top bits:
//! `tag << 56 | shard << 52 | generation << 24 | index`. The tag
//! identifies which of this crate's three [`crate::registry`] instances
//! (`Directory`/`Segment`/`Results`, see [`RegistryTag`]) the handle was
//! issued from, so a handle from the wrong registry is rejected by
//! [`SlotMap::get`]/[`SlotMap::remove`] on a tag mismatch *before* any
//! index/generation lookup happens — two handles from different registries
//! can otherwise carry identical `(index, generation)` bit patterns (both
//! starting at index 0, generation 1), and without the tag nothing would
//! stop a directory handle from being looked up directly in the segment
//! registry. Every `insert` into a freed slot bumps that slot's generation,
//! so a handle captured before a `remove`/reuse cycle carries the *old*
//! generation and fails the generation check — it can never silently alias
//! the new occupant. Generation 0 is never issued (every slot starts at
//! generation 1), so the all-zero handle `0` is guaranteed invalid too, a
//! convenient sentinel for "no handle" on the C side.
//!
//! **Why a shard field** (added in the M2 sweep batch `c13-ffi-surface`): the
//! six *results* registries (see [`crate::registry`]) are written twice per
//! search -- once to insert the results handle the query produced, once when
//! the caller closes it -- so they are the one registry kind whose exclusive
//! sections are on the hot path. b15 measured that pair as the remaining
//! concurrency ceiling after the `Mutex` -> `RwLock` change (1.17x on four
//! threads, not ~4x). [`crate::registry::Sharded`] splits each of those
//! registries into [`SHARDS`] independent `RwLock<SlotMap<T>>`s and records
//! *which* one issued a handle in these four bits, so a lookup or a close
//! goes straight to the owning shard without consulting (or locking) any
//! other. Four bits is why the generation field is 28 bits rather than 32:
//! 2^28 reuses of one slot before the counter wraps, against a 2^24 cap on
//! simultaneously-open handles per shard, is still an enormous margin, and
//! the index field -- the one that must not truncate (see [`MAX_SLOTS`]) --
//! is untouched.

use crate::error::FfiStatus;

const INDEX_BITS: u32 = 24;
const GENERATION_BITS: u32 = 28;
const SHARD_BITS: u32 = 4;
const INDEX_MASK: u64 = (1 << INDEX_BITS) - 1;
const GENERATION_MASK: u64 = (1 << GENERATION_BITS) - 1;
const SHARD_MASK: u64 = (1 << SHARD_BITS) - 1;

/// How many independent `RwLock<SlotMap<T>>`s a sharded registry is split
/// into -- see this module's doc comment and [`crate::registry::Sharded`].
/// Must be a power of two and fit in [`SHARD_BITS`] bits.
pub const SHARDS: usize = 1 << SHARD_BITS;

/// Identifies which process-wide registry (see [`crate::registry`]) a
/// packed handle was issued from. Encoded in the top 8 bits of every handle
/// so a handle from one registry can never be silently accepted by another
/// registry's [`SlotMap::get`]/[`SlotMap::remove`] — see this module's doc
/// comment.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistryTag {
    Directory = 1,
    Segment = 2,
    Results = 3,
    /// `(doc_id, score)` pairs from a scored query (task #30), kept in a
    /// registry separate from the unscored [`RegistryTag::Results`] rather
    /// than widening `ResultsHandle` itself — see `registry.rs`'s
    /// `ScoredResultsHandle` doc comment for why.
    ScoredResults = 4,
    /// `(doc_id, value)` pairs from a doc-value sort (task #40), kept in yet
    /// another registry separate from both `Results` and `ScoredResults` --
    /// see `registry.rs`'s `SortedResultsHandle` doc comment for why.
    SortedResults = 5,
    /// An opened `lucene_search::directory_reader::DirectoryReader` (task
    /// #51), one per open commit -- kept in its own registry (not folded
    /// into `Directory` or `Segment`) since it owns every segment already
    /// opened under it, not a single filesystem root or a single segment's
    /// files -- see `registry.rs`'s `DirectoryReaderHandle` doc comment.
    DirectoryReader = 6,
    /// Resolved `(ord, label, count)` triples from a SortedSet facet count
    /// (`facets.rs`'s `ffi_facet_counts_sorted_set`, wrapping
    /// `lucene_search::facets::facet_counts`/`resolve_labels`/`top_n_facets`)
    /// -- kept in its own registry rather than folded into `SortedResults`
    /// since a facet result's element also carries a resolved string label,
    /// not just a `(doc_id, value)` pair -- see `registry.rs`'s
    /// `FacetResultsHandle` doc comment.
    FacetResults = 7,
    /// Assembled highlight fragments (`highlighter.rs`'s
    /// `ffi_assemble_fragments`, wrapping
    /// `lucene_search::highlighter::assemble_fragments`) -- kept in its own
    /// registry rather than folded into any existing one since a fragment's
    /// element (`text` plus a variable-length `matched_terms` list) has no
    /// resemblance to any of this crate's other result shapes -- see
    /// `registry.rs`'s `FragmentResultsHandle` doc comment.
    FragmentResults = 8,
    /// A flattened `lucene_search::explain::Explanation` tree (`explain.rs`'s
    /// `ffi_explain_term_query`/`ffi_explain_phrase_query`/
    /// `ffi_explain_boolean_query`, wrapping
    /// `lucene_search::explain::explain_clause`) -- kept in its own registry
    /// rather than folded into any existing one since its element (one flattened
    /// tree node, carrying a `value`/`matched`/`description` plus a
    /// variable-length list of *child node indices*, not child values) has no
    /// resemblance to any of this crate's other result shapes -- see
    /// `registry.rs`'s `ExplainResultsHandle` doc comment.
    ExplainResults = 9,
    /// An opened `lucene_index::index_writer::IndexWriter` (IndexWriter
    /// commit/merge-policy FFI exposure, `writer.rs`'s `ffi_open_writer`) --
    /// kept in its own registry since it owns not just a filesystem root
    /// (unlike `Directory`) but a live, mutable, stateful writer session
    /// (buffered documents, a committed `SegmentInfos`, an optional prepared
    /// commit) -- see `registry.rs`'s `WriterHandle` doc comment for why this
    /// handle's value is not a plain `Directory`/`SegmentHandle`-shaped
    /// struct.
    Writer = 10,
    /// An opened per-segment vector reader (`vectors.rs`'s
    /// `ffi_open_vectors`): a `.vemf`/`.vec` flat vector store plus,
    /// optionally, its `.vem`/`.vex` HNSW graph -- kept in its own registry
    /// rather than folded into `Segment` because a vector field lives in a
    /// segment that need not have a term dictionary at all (real Lucene's
    /// `KnnVectorsReader` is a separate per-segment reader from
    /// `FieldsProducer`, and `ffi_open_segment` requires `.tim`/`.tip`/`.tmd`
    /// that a vectors-only segment does not have) -- see `registry.rs`'s
    /// `VectorsHandle` doc comment.
    Vectors = 11,
}

/// The largest slot index the 24-bit index field can represent, and so the
/// hard cap on simultaneously-open handles in any one registry (any one
/// *shard* of one, for a sharded registry -- see [`crate::registry::Sharded`]). Reaching it
/// means a caller leaked ~16.7M handles of one kind without closing them;
/// [`SlotMap::insert`] reports that as `None` rather than letting `pack`
/// truncate the index and hand back a handle that silently aliases a
/// *different* live entry in the same registry (a type-confusion bug the
/// generation tag cannot catch, since the aliased slot's generation is
/// whatever it happens to be).
pub const MAX_SLOTS: usize = INDEX_MASK as usize + 1;

fn pack(tag: RegistryTag, shard: u8, index: u32, generation: u32) -> u64 {
    debug_assert!(index as u64 <= INDEX_MASK);
    debug_assert!(shard as u64 <= SHARD_MASK);
    debug_assert!(generation as u64 <= GENERATION_MASK);
    ((tag as u64) << (SHARD_BITS + GENERATION_BITS + INDEX_BITS))
        | ((shard as u64 & SHARD_MASK) << (GENERATION_BITS + INDEX_BITS))
        | ((generation as u64 & GENERATION_MASK) << INDEX_BITS)
        | (index as u64 & INDEX_MASK)
}

fn unpack(handle: u64) -> (u8, u8, u32, u32) {
    let index = (handle & INDEX_MASK) as u32;
    let generation = ((handle >> INDEX_BITS) & GENERATION_MASK) as u32;
    let shard = ((handle >> (GENERATION_BITS + INDEX_BITS)) & SHARD_MASK) as u8;
    let tag = (handle >> (SHARD_BITS + GENERATION_BITS + INDEX_BITS)) as u8;
    (tag, shard, index, generation)
}

/// The shard id packed into `handle`, without validating anything else about
/// it -- [`crate::registry::Sharded`] uses this to pick which of its
/// [`SHARDS`] locks to take *before* the handle is validated (the shard field
/// is masked to [`SHARD_BITS`] bits, so every possible `u64` names a real
/// shard, and the named shard's own tag/generation check then applies -- see
/// [`crate::registry::Sharded`] on why that makes the field routing rather
/// than authentication).
pub fn shard_of(handle: u64) -> usize {
    ((handle >> (GENERATION_BITS + INDEX_BITS)) & SHARD_MASK) as usize
}

/// The next generation for a slot being reused. Wraps within
/// [`GENERATION_BITS`] (never into the shard field above it) and never
/// yields `0`, so the all-zero handle stays permanently invalid.
fn next_generation(current: u32) -> u32 {
    let next = (current as u64 + 1) & GENERATION_MASK;
    if next == 0 {
        1
    } else {
        next as u32
    }
}

struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

/// A generation-tagged slotmap: `insert` hands back an opaque `u64`,
/// `get`/`remove` only succeed for a handle whose registry tag *and*
/// generation both match the slot's current occupant.
pub struct SlotMap<T> {
    tag: RegistryTag,
    /// Which shard of a [`crate::registry::Sharded`] registry this map is,
    /// stamped into every handle it issues so a later lookup/close goes
    /// straight back here. Always `0` for an unsharded registry.
    shard: u8,
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
    /// Always [`MAX_SLOTS`] in production; a field rather than a hardcoded
    /// constant purely so `SlotMap::set_max_slots_for_test` can drive the
    /// exhaustion path without allocating 16.7M slots in a unit test.
    max_slots: usize,
}

impl<T> SlotMap<T> {
    pub fn new(tag: RegistryTag) -> Self {
        Self::new_shard(tag, 0)
    }

    /// [`SlotMap::new`] for shard `shard` of a sharded registry -- see
    /// [`crate::registry::Sharded`]. `shard` must be `< SHARDS`.
    pub fn new_shard(tag: RegistryTag, shard: usize) -> Self {
        assert!(shard < SHARDS, "shard id {shard} out of range");
        Self {
            tag,
            shard: shard as u8,
            slots: Vec::new(),
            free: Vec::new(),
            max_slots: MAX_SLOTS,
        }
    }

    /// Lowers this map's slot cap so a test can reach the exhaustion branch
    /// of [`SlotMap::insert`] without allocating [`MAX_SLOTS`] slots.
    #[cfg(test)]
    fn set_max_slots_for_test(&mut self, max_slots: usize) {
        self.max_slots = max_slots;
    }

    /// Stores `value` and returns its packed handle, or `None` when this
    /// registry already holds [`MAX_SLOTS`] entries -- see that constant's
    /// doc comment for why exhaustion must be an error rather than a
    /// silently truncated (aliasing) index.
    pub fn insert(&mut self, value: T) -> Option<u64> {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.generation = next_generation(slot.generation);
            slot.value = Some(value);
            Some(pack(self.tag, self.shard, index, slot.generation))
        } else {
            if self.slots.len() >= self.max_slots {
                return None;
            }
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 1,
                value: Some(value),
            });
            Some(pack(self.tag, self.shard, index, 1))
        }
    }

    fn slot(&self, handle: u64) -> Option<&Slot<T>> {
        let (tag, shard, index, generation) = unpack(handle);
        if tag != self.tag as u8 || shard != self.shard {
            return None;
        }
        let slot = self.slots.get(index as usize)?;
        (slot.generation == generation).then_some(slot)
    }

    pub fn get(&self, handle: u64) -> Option<&T> {
        self.slot(handle)?.value.as_ref()
    }

    /// In-place mutation accessor. Originally test-only (used to fabricate
    /// an otherwise unreachable corrupted state, e.g. a `SegmentHandle`
    /// whose `.doc` bytes fail to reopen, for `query.rs`'s decode-error-path
    /// tests) -- every handle type before `writer.rs`'s `WriterHandle` was
    /// immutable once inserted, so production code never needed this.
    /// `writer.rs`'s `IndexWriter`-backed handle is genuinely stateful
    /// (`add_document`/`commit`/etc. all take `&mut self`), so this is now a
    /// real part of the crate's handle API surface, not test-only.
    pub fn get_mut(&mut self, handle: u64) -> Option<&mut T> {
        let (tag, shard, index, generation) = unpack(handle);
        if tag != self.tag as u8 || shard != self.shard {
            return None;
        }
        let slot = self.slots.get_mut(index as usize)?;
        if slot.generation == generation {
            slot.value.as_mut()
        } else {
            None
        }
    }

    /// [`SlotMap::insert`] with registry exhaustion already mapped to
    /// [`FfiStatus::HandleLimit`] plus a last-error message -- the form every
    /// exported function uses, so "too many open handles" is a status code a
    /// JNI caller can branch on rather than a truncated, aliasing handle.
    pub fn insert_checked(&mut self, value: T) -> Result<u64, FfiStatus> {
        self.insert(value).ok_or_else(|| {
            crate::error::set_last_error(format!(
                "handle registry exhausted: {MAX_SLOTS} handles of this kind are already open \
                 (are handles being leaked instead of closed?)"
            ));
            FfiStatus::HandleLimit
        })
    }

    /// Removes and returns the handle's value, freeing its slot for reuse
    /// (with the generation bumped on the *next* `insert`, not here — a
    /// concurrent removed-but-not-yet-reused handle still fails `get`
    /// because `value` is `None`).
    pub fn remove(&mut self, handle: u64) -> Option<T> {
        let (tag, shard, index, generation) = unpack(handle);
        if tag != self.tag as u8 || shard != self.shard {
            return None;
        }
        let slot = self.slots.get_mut(index as usize)?;
        if slot.generation != generation {
            return None;
        }
        let value = slot.value.take()?;
        self.free.push(index);
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_then_get_roundtrips() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::Segment);
        let h = map.insert(42).unwrap();
        assert_eq!(map.get(h), Some(&42));
    }

    #[test]
    fn remove_returns_value_and_invalidates_handle() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::Segment);
        let h = map.insert(7).unwrap();
        assert_eq!(map.remove(h), Some(7));
        assert_eq!(map.get(h), None);
        assert_eq!(map.remove(h), None);
    }

    #[test]
    fn stale_handle_does_not_alias_reused_slot() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::Segment);
        let h1 = map.insert(1).unwrap();
        map.remove(h1).unwrap();
        let h2 = map.insert(2).unwrap();
        // Same slot index reused, but a different generation.
        assert_eq!(map.get(h1), None);
        assert_eq!(map.get(h2), Some(&2));
        assert_ne!(h1, h2);
    }

    #[test]
    fn unknown_handle_out_of_range_is_none() {
        let map: SlotMap<i32> = SlotMap::new(RegistryTag::Segment);
        assert_eq!(map.get(999), None);
    }

    #[test]
    fn remove_with_stale_generation_on_a_reused_slot_is_none() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::Segment);
        let h1 = map.insert(1).unwrap();
        map.remove(h1).unwrap();
        let h2 = map.insert(2).unwrap();
        // `h1`'s index was reused by `h2`'s insert, but `h1`'s generation is
        // now stale -- `remove(h1)` must hit the generation-mismatch branch,
        // not silently remove `h2`'s value.
        assert_eq!(map.remove(h1), None);
        assert_eq!(map.get(h2), Some(&2));
    }

    #[test]
    fn fresh_map_of_any_tag_rejects_handle_zero() {
        let map: SlotMap<i32> = SlotMap::new(RegistryTag::Directory);
        assert_eq!(map.get(0), None);
    }

    #[test]
    fn zero_handle_is_never_valid_for_a_fresh_map() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::Segment);
        let h = map.insert(5).unwrap();
        assert_ne!(h, 0);
        assert_eq!(map.get(0), None);
    }

    #[test]
    fn multiple_inserts_get_distinct_handles() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::Segment);
        let h1 = map.insert(1).unwrap();
        let h2 = map.insert(2).unwrap();
        assert_ne!(h1, h2);
        assert_eq!(map.get(h1), Some(&1));
        assert_eq!(map.get(h2), Some(&2));
    }

    #[test]
    fn insert_reports_exhaustion_instead_of_truncating_the_index() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::Results);
        map.set_max_slots_for_test(2);
        let h1 = map.insert(1).unwrap();
        let h2 = map.insert(2).unwrap();
        // Third insert has no free slot and no room to grow: it must report
        // `None` rather than packing a truncated index that would alias an
        // existing live entry.
        assert_eq!(map.insert(3), None);
        // The two live handles are untouched by the refused insert.
        assert_eq!(map.get(h1), Some(&1));
        assert_eq!(map.get(h2), Some(&2));
        // Closing one frees its slot, so the next insert succeeds again.
        assert_eq!(map.remove(h1), Some(1));
        let h3 = map.insert(3).unwrap();
        assert_eq!(map.get(h3), Some(&3));
        assert_ne!(h3, h1, "reused slot must carry a bumped generation");
    }

    #[test]
    fn insert_checked_maps_exhaustion_to_a_status_code_and_message() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::Results);
        map.set_max_slots_for_test(1);
        map.insert_checked(1).unwrap();
        assert_eq!(map.insert_checked(2), Err(FfiStatus::HandleLimit));
    }

    #[test]
    fn max_slots_matches_the_index_field_width() {
        // The whole point of the cap: one more than this and `pack` would
        // truncate the index into another slot's bit pattern.
        assert_eq!(MAX_SLOTS as u64, INDEX_MASK + 1);
    }

    #[test]
    fn identical_index_generation_from_a_different_tag_is_rejected() {
        // Two same-shaped maps of different tags, each inserting once, produce
        // handles with the same (index, generation) bit pattern but different
        // tags -- a handle from one must never be accepted by the other's
        // `get`/`remove` (this is what stops a directory handle from being
        // silently accepted by the segment registry).
        let mut segment_map: SlotMap<i32> = SlotMap::new(RegistryTag::Segment);
        let mut directory_map: SlotMap<i32> = SlotMap::new(RegistryTag::Directory);
        let seg_handle = segment_map.insert(1).unwrap();
        let dir_handle = directory_map.insert(2).unwrap();
        // Same index (0) and generation (1) bit pattern, different tag bits.
        assert_ne!(seg_handle, dir_handle);
        assert_eq!(directory_map.get(seg_handle), None);
        assert_eq!(segment_map.get(dir_handle), None);
        assert_eq!(directory_map.remove(seg_handle), None);
        assert_eq!(segment_map.remove(dir_handle), None);
        // Legitimate access to each map with its own handle still works.
        assert_eq!(segment_map.get(seg_handle), Some(&1));
        assert_eq!(directory_map.get(dir_handle), Some(&2));
    }
    #[test]
    fn a_handle_carries_the_shard_that_issued_it() {
        for shard in 0..SHARDS {
            let mut map: SlotMap<i32> = SlotMap::new_shard(RegistryTag::Results, shard);
            let h = map.insert(shard as i32).unwrap();
            assert_eq!(shard_of(h), shard);
            assert_eq!(map.get(h), Some(&(shard as i32)));
        }
    }

    #[test]
    fn a_handle_from_another_shard_of_the_same_registry_is_rejected() {
        let mut a: SlotMap<i32> = SlotMap::new_shard(RegistryTag::Results, 0);
        let mut b: SlotMap<i32> = SlotMap::new_shard(RegistryTag::Results, 1);
        let ha = a.insert(1).unwrap();
        let hb = b.insert(2).unwrap();
        // Same tag, same (index, generation), different shard.
        assert_ne!(ha, hb);
        assert_eq!(b.get(ha), None);
        assert_eq!(a.get(hb), None);
        assert_eq!(b.remove(ha), None);
        assert_eq!(a.remove(hb), None);
        assert_eq!(a.get_mut(hb), None);
        assert_eq!(a.get(ha), Some(&1));
        assert_eq!(b.get(hb), Some(&2));
    }

    /// The generation field is four bits narrower than it was, so its wrap
    /// must stay inside it: a generation that overflowed into the shard bits
    /// would make a stale handle alias a *different shard's* live entry.
    #[test]
    fn a_wrapping_generation_never_bleeds_into_the_shard_field() {
        assert_eq!(next_generation(1), 2);
        assert_eq!(
            next_generation(GENERATION_MASK as u32 - 1),
            GENERATION_MASK as u32
        );
        // The wrap: never 0 (so the all-zero handle stays invalid), never
        // wider than the field.
        let wrapped = next_generation(GENERATION_MASK as u32);
        assert_eq!(wrapped, 1);
        assert!((wrapped as u64) <= GENERATION_MASK);
        // And a packed handle at the widest generation still reports its own
        // shard, not a neighbour's.
        let packed = pack(RegistryTag::Results, 5, 3, GENERATION_MASK as u32);
        assert_eq!(shard_of(packed), 5);
        let (tag, shard, index, generation) = unpack(packed);
        assert_eq!(
            (tag, shard, index, generation),
            (RegistryTag::Results as u8, 5, 3, GENERATION_MASK as u32)
        );
    }

    #[test]
    fn the_bit_fields_tile_a_u64_exactly() {
        assert_eq!(INDEX_BITS + GENERATION_BITS + SHARD_BITS + 8, 64);
        assert_eq!(SHARDS, (SHARD_MASK + 1) as usize);
    }

    #[test]
    #[should_panic(expected = "shard id")]
    fn a_shard_id_past_the_field_width_is_a_programming_error() {
        let _: SlotMap<i32> = SlotMap::new_shard(RegistryTag::Results, SHARDS);
    }
}
