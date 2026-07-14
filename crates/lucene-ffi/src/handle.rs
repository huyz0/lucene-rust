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
//! 32-bit generation counter in the middle bits, and an 8-bit registry-type
//! tag in the top bits: `tag << 56 | generation << 24 | index`. The tag
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

const INDEX_BITS: u32 = 24;
const GENERATION_BITS: u32 = 32;
const INDEX_MASK: u64 = (1 << INDEX_BITS) - 1;
const GENERATION_MASK: u64 = (1 << GENERATION_BITS) - 1;

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
}

fn pack(tag: RegistryTag, index: u32, generation: u32) -> u64 {
    debug_assert!(index as u64 <= INDEX_MASK);
    ((tag as u64) << (GENERATION_BITS + INDEX_BITS))
        | ((generation as u64) << INDEX_BITS)
        | (index as u64 & INDEX_MASK)
}

fn unpack(handle: u64) -> (u8, u32, u32) {
    let index = (handle & INDEX_MASK) as u32;
    let generation = ((handle >> INDEX_BITS) & GENERATION_MASK) as u32;
    let tag = (handle >> (GENERATION_BITS + INDEX_BITS)) as u8;
    (tag, index, generation)
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
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
}

impl<T> SlotMap<T> {
    pub fn new(tag: RegistryTag) -> Self {
        Self {
            tag,
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

    pub fn insert(&mut self, value: T) -> u64 {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.generation = slot.generation.wrapping_add(1).max(1);
            slot.value = Some(value);
            pack(self.tag, index, slot.generation)
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 1,
                value: Some(value),
            });
            pack(self.tag, index, 1)
        }
    }

    fn slot(&self, handle: u64) -> Option<&Slot<T>> {
        let (tag, index, generation) = unpack(handle);
        if tag != self.tag as u8 {
            return None;
        }
        let slot = self.slots.get(index as usize)?;
        (slot.generation == generation).then_some(slot)
    }

    pub fn get(&self, handle: u64) -> Option<&T> {
        self.slot(handle)?.value.as_ref()
    }

    /// Test-only in-place mutation accessor, used to fabricate an otherwise
    /// unreachable corrupted state (e.g. a `SegmentHandle` whose `.doc`
    /// bytes fail to reopen) for `query.rs`'s decode-error-path tests --
    /// production code never needs to mutate a handle's value in place, so
    /// this is `#[cfg(test)]`-only rather than a real part of the crate's
    /// handle API surface.
    #[cfg(test)]
    pub fn get_mut(&mut self, handle: u64) -> Option<&mut T> {
        let (tag, index, generation) = unpack(handle);
        if tag != self.tag as u8 {
            return None;
        }
        let slot = self.slots.get_mut(index as usize)?;
        if slot.generation == generation {
            slot.value.as_mut()
        } else {
            None
        }
    }

    /// Removes and returns the handle's value, freeing its slot for reuse
    /// (with the generation bumped on the *next* `insert`, not here — a
    /// concurrent removed-but-not-yet-reused handle still fails `get`
    /// because `value` is `None`).
    pub fn remove(&mut self, handle: u64) -> Option<T> {
        let (tag, index, generation) = unpack(handle);
        if tag != self.tag as u8 {
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
        let h = map.insert(42);
        assert_eq!(map.get(h), Some(&42));
    }

    #[test]
    fn remove_returns_value_and_invalidates_handle() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::Segment);
        let h = map.insert(7);
        assert_eq!(map.remove(h), Some(7));
        assert_eq!(map.get(h), None);
        assert_eq!(map.remove(h), None);
    }

    #[test]
    fn stale_handle_does_not_alias_reused_slot() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::Segment);
        let h1 = map.insert(1);
        map.remove(h1).unwrap();
        let h2 = map.insert(2);
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
        let h1 = map.insert(1);
        map.remove(h1).unwrap();
        let h2 = map.insert(2);
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
        let h = map.insert(5);
        assert_ne!(h, 0);
        assert_eq!(map.get(0), None);
    }

    #[test]
    fn multiple_inserts_get_distinct_handles() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::Segment);
        let h1 = map.insert(1);
        let h2 = map.insert(2);
        assert_ne!(h1, h2);
        assert_eq!(map.get(h1), Some(&1));
        assert_eq!(map.get(h2), Some(&2));
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
        let seg_handle = segment_map.insert(1);
        let dir_handle = directory_map.insert(2);
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
}
