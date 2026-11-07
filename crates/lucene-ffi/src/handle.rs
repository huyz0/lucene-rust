//! A generation-tagged slotmap for the opaque `u64` handles that cross the
//! FFI boundary. No pointer is ever handed to a caller: an exported function
//! returns a packed handle and looks every handle it takes up here first, so
//! a stale, unknown or closed handle is a lookup miss, never a dereference of
//! freed memory.
//!
//! A handle packs `tag << 56 | generation << 24 | index`: a 24-bit slot index,
//! a 32-bit generation, and an 8-bit [`RegistryTag`] naming the registry that
//! issued it, so a handle is refused by any other registry even when its index
//! and generation match. Reusing a slot bumps its generation, so a handle kept
//! past a close never aliases the slot's next occupant. Generation 0 is never
//! issued, so the handle `0` is always invalid.

use crate::error::FfiStatus;

const INDEX_BITS: u32 = 24;
const GENERATION_BITS: u32 = 32;
const INDEX_MASK: u64 = (1 << INDEX_BITS) - 1;
const GENERATION_MASK: u64 = (1 << GENERATION_BITS) - 1;

/// The registry a handle was issued from, packed into its top 8 bits.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistryTag {
    /// A reader over a JVM reader's segments (`jvm_reader.rs`).
    JvmReader = 12,
    /// A shard's index writer (`engine_writer.rs`).
    EngineWriter = 13,
    ZstdDictionary = 14,
    /// A zstd term-block dictionary (`zstd_terms.rs`).
    ZstdTermsDictionary = 15,
}

/// The cap on simultaneously open handles in one registry: the 24-bit index
/// field's range. [`SlotMap::insert`] refuses past it rather than truncating
/// an index into another live slot's.
pub const MAX_SLOTS: usize = INDEX_MASK as usize + 1;

fn pack(tag: RegistryTag, index: u32, generation: u32) -> u64 {
    debug_assert!(index as u64 <= INDEX_MASK);
    ((tag as u64) << (GENERATION_BITS + INDEX_BITS))
        | ((generation as u64 & GENERATION_MASK) << INDEX_BITS)
        | (index as u64 & INDEX_MASK)
}

fn unpack(handle: u64) -> (u8, u32, u32) {
    let index = (handle & INDEX_MASK) as u32;
    let generation = ((handle >> INDEX_BITS) & GENERATION_MASK) as u32;
    let tag = (handle >> (GENERATION_BITS + INDEX_BITS)) as u8;
    (tag, index, generation)
}

/// The next generation for a reused slot; wraps without ever yielding `0`.
fn next_generation(current: u32) -> u32 {
    match current.wrapping_add(1) {
        0 => 1,
        next => next,
    }
}

struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

/// A generation-tagged slotmap: `insert` returns an opaque `u64`; `get` and
/// `remove` succeed only for a handle whose tag and generation match the
/// slot's current occupant.
pub struct SlotMap<T> {
    tag: RegistryTag,
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
    /// [`MAX_SLOTS`], lowered by tests to reach the exhaustion path.
    max_slots: usize,
}

impl<T> SlotMap<T> {
    pub fn new(tag: RegistryTag) -> Self {
        Self {
            tag,
            slots: Vec::new(),
            free: Vec::new(),
            max_slots: MAX_SLOTS,
        }
    }

    #[cfg(test)]
    fn set_max_slots_for_test(&mut self, max_slots: usize) {
        self.max_slots = max_slots;
    }

    /// Stores `value` and returns its handle, or `None` when the registry
    /// already holds [`MAX_SLOTS`] entries.
    pub fn insert(&mut self, value: T) -> Option<u64> {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.generation = next_generation(slot.generation);
            slot.value = Some(value);
            Some(pack(self.tag, index, slot.generation))
        } else {
            if self.slots.len() >= self.max_slots {
                return None;
            }
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 1,
                value: Some(value),
            });
            Some(pack(self.tag, index, 1))
        }
    }

    fn slot_index(&self, handle: u64) -> Option<usize> {
        let (tag, index, generation) = unpack(handle);
        if tag != self.tag as u8 {
            return None;
        }
        let slot = self.slots.get(index as usize)?;
        (slot.generation == generation).then_some(index as usize)
    }

    pub fn get(&self, handle: u64) -> Option<&T> {
        self.slots[self.slot_index(handle)?].value.as_ref()
    }

    /// [`SlotMap::insert`] with exhaustion mapped to
    /// [`FfiStatus::HandleLimit`] and a last-error message.
    pub fn insert_checked(&mut self, value: T) -> Result<u64, FfiStatus> {
        self.insert(value).ok_or_else(|| {
            crate::error::set_last_error(format!(
                "handle registry exhausted: {MAX_SLOTS} handles of this kind are already open \
                 (are handles being leaked instead of closed?)"
            ));
            FfiStatus::HandleLimit
        })
    }

    /// Every live value, in slot order.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().filter_map(|s| s.value.as_ref())
    }

    /// Removes and returns the handle's value, freeing its slot for reuse.
    pub fn remove(&mut self, handle: u64) -> Option<T> {
        let index = self.slot_index(handle)?;
        let value = self.slots[index].value.take()?;
        self.free.push(index as u32);
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_then_get_roundtrips() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::JvmReader);
        let h = map.insert(42).unwrap();
        assert_eq!(map.get(h), Some(&42));
    }

    #[test]
    fn remove_returns_value_and_invalidates_handle() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::JvmReader);
        let h = map.insert(7).unwrap();
        assert_eq!(map.remove(h), Some(7));
        assert_eq!(map.get(h), None);
        assert_eq!(map.remove(h), None);
    }

    #[test]
    fn stale_handle_does_not_alias_reused_slot() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::JvmReader);
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
        let map: SlotMap<i32> = SlotMap::new(RegistryTag::JvmReader);
        assert_eq!(map.get(999), None);
    }

    #[test]
    fn remove_with_stale_generation_on_a_reused_slot_is_none() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::JvmReader);
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
        let map: SlotMap<i32> = SlotMap::new(RegistryTag::EngineWriter);
        assert_eq!(map.get(0), None);
    }

    #[test]
    fn zero_handle_is_never_valid_for_a_fresh_map() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::JvmReader);
        let h = map.insert(5).unwrap();
        assert_ne!(h, 0);
        assert_eq!(map.get(0), None);
    }

    #[test]
    fn multiple_inserts_get_distinct_handles() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::JvmReader);
        let h1 = map.insert(1).unwrap();
        let h2 = map.insert(2).unwrap();
        assert_ne!(h1, h2);
        assert_eq!(map.get(h1), Some(&1));
        assert_eq!(map.get(h2), Some(&2));
    }

    #[test]
    fn insert_reports_exhaustion_instead_of_truncating_the_index() {
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::JvmReader);
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
        let mut map: SlotMap<i32> = SlotMap::new(RegistryTag::JvmReader);
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
        // `get`/`remove`.
        let mut reader_map: SlotMap<i32> = SlotMap::new(RegistryTag::JvmReader);
        let mut writer_map: SlotMap<i32> = SlotMap::new(RegistryTag::EngineWriter);
        let reader_handle = reader_map.insert(1).unwrap();
        let writer_handle = writer_map.insert(2).unwrap();
        // Same index (0) and generation (1) bit pattern, different tag bits.
        assert_ne!(reader_handle, writer_handle);
        assert_eq!(writer_map.get(reader_handle), None);
        assert_eq!(reader_map.get(writer_handle), None);
        assert_eq!(writer_map.remove(reader_handle), None);
        assert_eq!(reader_map.remove(writer_handle), None);
        // Legitimate access to each map with its own handle still works.
        assert_eq!(reader_map.get(reader_handle), Some(&1));
        assert_eq!(writer_map.get(writer_handle), Some(&2));
    }

    #[test]
    fn a_wrapping_generation_skips_zero_and_stays_in_its_field() {
        assert_eq!(next_generation(1), 2);
        assert_eq!(next_generation(GENERATION_MASK as u32), 1);
        let packed = pack(RegistryTag::JvmReader, 3, GENERATION_MASK as u32);
        assert_eq!(
            unpack(packed),
            (RegistryTag::JvmReader as u8, 3, GENERATION_MASK as u32)
        );
    }

    #[test]
    fn the_bit_fields_tile_a_u64_exactly() {
        assert_eq!(INDEX_BITS + GENERATION_BITS + 8, 64);
    }
}
