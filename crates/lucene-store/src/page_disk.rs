//! The page cache's local disk tier: fixed-size slots in a few large slab
//! files, with CLOCK eviction.
//!
//! Memory per cached page is one slot record (16 bytes), plus 4 bytes per
//! page of each file the tier has seen (a dense page-to-slot table). There
//! is no file per page: slot `s` lives at byte `(s % SLOTS_PER_SLAB) *
//! slot_len` of slab file `s / SLOTS_PER_SLAB`, so a 2 TB tier of 256 KiB
//! pages is a few thousand files and about 130 MB of RAM.
//!
//! A slot is published only after its bytes are written, and a read checks
//! afterwards that the slot still holds the same page (its generation did
//! not change): a slot evicted and reused while it was being read is a
//! miss, never another page's bytes.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::error::{Error, Result};
use crate::lock::lock_ignoring_poison;

/// Slots per slab file.
const SLOTS_PER_SLAB: u32 = 4096;
/// No slot.
const NONE: u32 = u32::MAX;

/// One slot's record.
#[derive(Clone, Copy, Default)]
struct Slot {
    /// The owning file's id, or `NONE` when free.
    file: u32,
    /// The page of that file.
    page: u32,
    /// Bytes stored.
    len: u32,
    /// Bumped on every reuse.
    generation: u16,
    /// CLOCK's reference bit (bit 0), pinned (bit 1), being written (bit 2).
    flags: u8,
}

const REFERENCED: u8 = 1;
const PINNED: u8 = 2;
const WRITING: u8 = 4;

/// A file the tier has seen: its page-to-slot table.
struct FileSlots {
    pages: Vec<u32>,
    pinned: bool,
}

#[derive(Default)]
struct State {
    ids: HashMap<Arc<str>, u32>,
    files: HashMap<u32, FileSlots>,
    next_id: u32,
    slots: Vec<Slot>,
    free: Vec<u32>,
    hand: usize,
    used: u64,
}

pub(crate) struct DiskTier {
    root: PathBuf,
    slot_len: u32,
    capacity: u32,
    state: Mutex<State>,
    slabs: Mutex<Vec<Arc<File>>>,
}

impl DiskTier {
    /// A tier under `root` (emptied) of at most `budget` bytes, in slots
    /// big enough for pages of `page_size` plus a decoder's per-unit
    /// overhead.
    pub(crate) fn open(root: &Path, budget: u64, page_size: usize) -> Result<DiskTier> {
        match std::fs::remove_dir_all(root) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(Error::Io(e)),
        }
        std::fs::create_dir_all(root)?;
        let extra = (page_size / 64).max(4096);
        let slot_len = page_size
            .checked_add(extra)
            .map(|n| n.div_ceil(4096).saturating_mul(4096))
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| {
                Error::IllegalArgument("page_size too large for the disk tier".into())
            })?;
        let capacity = budget
            .checked_div(u64::from(slot_len))
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(NONE.saturating_sub(1));
        Ok(DiskTier {
            root: root.to_path_buf(),
            slot_len,
            capacity,
            state: Mutex::new(State::default()),
            slabs: Mutex::new(Vec::new()),
        })
    }

    pub(crate) fn used(&self) -> u64 {
        lock_ignoring_poison(&self.state).used
    }

    /// The slab holding `slot`, created on first use.
    fn slab(&self, slot: u32) -> std::io::Result<(Arc<File>, u64)> {
        let index = (slot / SLOTS_PER_SLAB) as usize;
        let offset = u64::from(slot % SLOTS_PER_SLAB).saturating_mul(u64::from(self.slot_len));
        let mut slabs = lock_ignoring_poison(&self.slabs);
        while slabs.len() <= index {
            let path = self.root.join(format!("slab-{}", slabs.len()));
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)?;
            slabs.push(Arc::new(file));
        }
        Ok((Arc::clone(&slabs[index]), offset))
    }

    fn id_of(st: &mut State, key: &Arc<str>) -> u32 {
        if let Some(id) = st.ids.get(key) {
            return *id;
        }
        let id = st.next_id;
        st.next_id = st.next_id.saturating_add(1);
        st.ids.insert(Arc::clone(key), id);
        st.files.insert(
            id,
            FileSlots {
                pages: Vec::new(),
                pinned: false,
            },
        );
        id
    }

    fn slot_of(st: &State, key: &Arc<str>, page: u64) -> Option<u32> {
        let id = st.ids.get(key)?;
        let p = usize::try_from(page).ok()?;
        let s = *st.files.get(id)?.pages.get(p)?;
        (s != NONE).then_some(s)
    }

    /// Whether page `page` of `key` is cached.
    pub(crate) fn contains(&self, key: &Arc<str>, page: u64) -> bool {
        Self::slot_of(&lock_ignoring_poison(&self.state), key, page).is_some()
    }

    pub(crate) fn get(&self, key: &Arc<str>, page: u64) -> Option<Arc<[u8]>> {
        let (slot, generation, len) = {
            let mut st = lock_ignoring_poison(&self.state);
            let slot = Self::slot_of(&st, key, page)?;
            let rec = st.slots.get_mut(slot as usize)?;
            rec.flags |= REFERENCED;
            (slot, rec.generation, rec.len)
        };
        let mut buf = vec![0u8; len as usize];
        let read = self
            .slab(slot)
            .and_then(|(file, offset)| read_exact_at(&file, &mut buf, offset));
        let mut st = lock_ignoring_poison(&self.state);
        let same = st
            .slots
            .get(slot as usize)
            .is_some_and(|r| r.generation == generation && r.flags & WRITING == 0);
        if read.is_err() && same {
            // An unreadable slot is dropped, so the page is fetched and
            // cached again rather than failing every later read.
            Self::release(&mut st, slot);
            st.free.push(slot);
        }
        (read.is_ok() && same).then(|| Arc::from(buf))
    }

    pub(crate) fn put(&self, key: &Arc<str>, page: u64, bytes: &[u8]) {
        let Ok(len) = u32::try_from(bytes.len()) else {
            return;
        };
        let Ok(page32) = u32::try_from(page) else {
            return;
        };
        if len > self.slot_len {
            return;
        }
        let (slot, id, generation) = {
            let mut st = lock_ignoring_poison(&self.state);
            if Self::slot_of(&st, key, page).is_some() {
                return;
            }
            let id = Self::id_of(&mut st, key);
            let pinned = st.files.get(&id).is_some_and(|f| f.pinned);
            let Some(slot) = self.allocate(&mut st, pinned) else {
                return;
            };
            let rec = &mut st.slots[slot as usize];
            rec.file = id;
            rec.page = page32;
            rec.len = len;
            rec.generation = rec.generation.wrapping_add(1);
            rec.flags = WRITING | if pinned { PINNED } else { 0 };
            (slot, id, rec.generation)
        };
        let written = self
            .slab(slot)
            .and_then(|(file, offset)| write_all_at(&file, bytes, offset));
        let mut st = lock_ignoring_poison(&self.state);
        let alive = st.ids.get(key) == Some(&id)
            && st
                .slots
                .get(slot as usize)
                .is_some_and(|r| r.generation == generation);
        if written.is_err() || !alive {
            if st
                .slots
                .get(slot as usize)
                .is_some_and(|r| r.generation == generation)
            {
                st.slots[slot as usize].file = NONE;
                st.slots[slot as usize].flags = 0;
                st.free.push(slot);
            }
            return;
        }
        let p = page as usize;
        if Self::slot_of(&st, key, page).is_some() {
            // Another put of the same page published first: keeping this
            // slot too would leave it unreachable (and, pinned, never freed).
            st.slots[slot as usize].file = NONE;
            st.slots[slot as usize].flags = 0;
            st.free.push(slot);
            return;
        }
        // The file may have been pinned while this page was written.
        let pinned = st.files.get(&id).is_some_and(|f| f.pinned);
        let rec = &mut st.slots[slot as usize];
        rec.flags &= !WRITING;
        if pinned {
            rec.flags |= PINNED;
        }
        st.used = st.used.saturating_add(u64::from(len));
        if let Some(f) = st.files.get_mut(&id) {
            if f.pages.len() <= p {
                f.pages.resize(p.saturating_add(1), NONE);
            }
            f.pages[p] = slot;
        }
    }

    /// A free slot: an unused one, a new one under the budget, or one
    /// CLOCK evicts; for a pinned page, a new one past the budget when every
    /// slot is pinned or being written. `None` for any other page then.
    fn allocate(&self, st: &mut State, pinned: bool) -> Option<u32> {
        if let Some(s) = st.free.pop() {
            return Some(s);
        }
        let n = u32::try_from(st.slots.len()).unwrap_or(NONE);
        if n < self.capacity {
            st.slots.push(Slot {
                file: NONE,
                ..Slot::default()
            });
            return Some(n);
        }
        // Two sweeps: the first may only clear reference bits.
        let total = st.slots.len();
        for _ in 0..total.saturating_mul(2) {
            let s = st.hand;
            st.hand = st.hand.saturating_add(1);
            if st.hand >= total {
                st.hand = 0;
            }
            let rec = st.slots[s];
            if rec.flags & (PINNED | WRITING) != 0 {
                continue;
            }
            if rec.flags & REFERENCED != 0 {
                st.slots[s].flags &= !REFERENCED;
                continue;
            }
            Self::release(st, s as u32);
            return Some(s as u32);
        }
        // Every slot pinned or being written: a pinned page still gets one,
        // past the budget, as hot files must stay local.
        if pinned && n < NONE {
            st.slots.push(Slot {
                file: NONE,
                ..Slot::default()
            });
            return Some(n);
        }
        None
    }

    /// Unmaps slot `s` from its owner (the caller reuses or frees it).
    fn release(st: &mut State, s: u32) {
        let rec = st.slots[s as usize];
        if rec.file != NONE {
            if let Some(f) = st.files.get_mut(&rec.file) {
                if let Some(e) = f.pages.get_mut(rec.page as usize) {
                    if *e == s {
                        *e = NONE;
                    }
                }
            }
            st.used = st.used.saturating_sub(u64::from(rec.len));
        }
        st.slots[s as usize].file = NONE;
        st.slots[s as usize].flags = 0;
        st.slots[s as usize].generation = rec.generation.wrapping_add(1);
    }

    pub(crate) fn pin(&self, key: &str) {
        let key: Arc<str> = Arc::from(key);
        let mut st = lock_ignoring_poison(&self.state);
        let id = Self::id_of(&mut st, &key);
        let slots: Vec<u32> = match st.files.get_mut(&id) {
            Some(f) => {
                f.pinned = true;
                f.pages.iter().copied().filter(|s| *s != NONE).collect()
            }
            None => Vec::new(),
        };
        for s in slots {
            st.slots[s as usize].flags |= PINNED;
        }
    }

    pub(crate) fn forget_prefix(&self, prefix: &str) {
        let mut st = lock_ignoring_poison(&self.state);
        let gone: Vec<Arc<str>> = st
            .ids
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        for k in gone {
            let Some(id) = st.ids.remove(&k) else {
                continue;
            };
            let Some(f) = st.files.remove(&id) else {
                continue;
            };
            for s in f.pages.into_iter().filter(|s| *s != NONE) {
                if st.slots.get(s as usize).is_some_and(|r| r.file == id) {
                    Self::release(&mut st, s);
                    st.free.push(s);
                }
            }
        }
    }

    /// RAM held by the tier's records, for tests and reporting.
    pub(crate) fn metadata_bytes(&self) -> usize {
        let st = lock_ignoring_poison(&self.state);
        let pages: usize = st.files.values().map(|f| f.pages.capacity()).sum();
        st.slots
            .capacity()
            .saturating_mul(std::mem::size_of::<Slot>())
            .saturating_add(pages.saturating_mul(4))
            .saturating_add(st.free.capacity().saturating_mul(4))
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(unix)]
fn write_all_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    while !buf.is_empty() {
        let n = std::os::windows::fs::FileExt::seek_read(file, buf, offset)?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[n..];
        offset = offset.saturating_add(n as u64);
    }
    Ok(())
}

#[cfg(windows)]
fn write_all_at(file: &File, mut buf: &[u8], mut offset: u64) -> std::io::Result<()> {
    while !buf.is_empty() {
        let n = std::os::windows::fs::FileExt::seek_write(file, buf, offset)?;
        buf = &buf[n..];
        offset = offset.saturating_add(n as u64);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;

    fn key(s: &str) -> Arc<str> {
        Arc::from(s)
    }

    #[test]
    fn pages_round_trip_through_slots_and_clock_spares_referenced_and_pinned_ones() {
        let tmp = lucene_util::test_support::TempDir::new("page-disk-clock");
        // Slots of 8 KiB (4 KiB page plus overhead), three of them.
        let d = DiskTier::open(&tmp.path().join("t"), 3 * 8192, 4096).unwrap();
        assert_eq!(d.slot_len, 8192);
        let (a, b) = (key("a"), key("b"));
        d.put(&a, 0, &[1; 100]);
        d.pin("a");
        d.put(&a, 1, &[2; 4096]);
        d.put(&b, 0, &[3; 10]);
        assert_eq!(d.used(), 4206);
        assert_eq!(&d.get(&a, 1).unwrap()[..], &[2; 4096][..]);
        // Full: CLOCK skips the pinned pages of `a` and evicts `b`'s page.
        d.put(&b, 1, &[4; 10]);
        assert!(d.contains(&a, 0) && d.contains(&a, 1));
        assert!(!d.contains(&b, 0) && d.contains(&b, 1));
        // Only pinned pages and one free: a put of a pinned file still fits
        // by evicting `b`'s last page, then nothing more fits.
        d.put(&a, 2, &[5; 10]);
        assert!(d.contains(&a, 2) && !d.contains(&b, 1));
        d.put(&b, 2, &[6; 10]);
        assert!(!d.contains(&b, 2), "every slot pinned");
        // A pinned page still gets a slot, past the budget.
        d.put(&a, 3, &[8; 10]);
        assert!(d.contains(&a, 3));
        // A second put of a cached page is a no-op; an oversized one too.
        d.put(&a, 0, &[9; 100]);
        assert_eq!(&d.get(&a, 0).unwrap()[..], &[1; 100][..]);
        d.put(&b, 3, &vec![0; 9000]);
        assert!(!d.contains(&b, 3));
        assert!(d.get(&b, 99).is_none() && d.get(&key("z"), 0).is_none());
        d.forget_prefix("a");
        assert_eq!(d.used(), 0);
        d.put(&b, 4, &[7; 10]);
        assert_eq!(&d.get(&b, 4).unwrap()[..], &[7; 10][..]);
        assert!(d.metadata_bytes() > 0);
    }

    #[test]
    fn many_pages_cost_a_few_bytes_each_and_few_files() {
        let tmp = lucene_util::test_support::TempDir::new("page-disk-many");
        let d = DiskTier::open(&tmp.path().join("t"), 20_000 * 8192, 4096).unwrap();
        let k = key("f");
        for p in 0..10_000u64 {
            d.put(&k, p, &[p as u8; 16]);
        }
        // 16-byte slots and 4-byte page entries, plus vector growth slack.
        assert!(d.metadata_bytes() < 10_000 * 40, "{}", d.metadata_bytes());
        assert_eq!(std::fs::read_dir(tmp.path().join("t")).unwrap().count(), 3);
        assert_eq!(&d.get(&k, 9_999).unwrap()[..], &[(9_999u64) as u8; 16][..]);
    }

    #[test]
    fn a_failed_write_frees_its_slot_and_a_forgotten_file_is_not_published() {
        let tmp = lucene_util::test_support::TempDir::new("page-disk-fail");
        let root = tmp.path().join("t");
        let d = DiskTier::open(&root, 4 * 8192, 4096).unwrap();
        let k = key("f");
        d.put(&k, 0, &[1; 10]);
        // Slab files gone and their directory replaced by a file: new slabs
        // cannot be created.
        std::fs::remove_dir_all(&root).unwrap();
        std::fs::write(&root, b"x").unwrap();
        lock_ignoring_poison(&d.slabs).clear();
        d.put(&k, 1, &[2; 10]);
        assert!(!d.contains(&k, 1));
        assert!(d.get(&k, 0).is_none(), "unreadable slab is a miss");
        assert!(!d.contains(&k, 0), "and dropped, so it is cached again");
        assert_eq!(d.used(), 0);
        assert!(DiskTier::open(&root.join("sub"), 1, 4096).is_err());
        assert!(DiskTier::open(&tmp.path().join("u"), 1, usize::MAX).is_err());
    }
}
