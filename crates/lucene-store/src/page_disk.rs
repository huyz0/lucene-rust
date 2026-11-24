//! The page cache's local disk tier: slots of a few sizes in large slab
//! files, with CLOCK eviction.
//!
//! A page goes in the smallest size class that holds it: 4 KiB, doubling
//! up to the full slot (a page plus a decoder's overhead), so a 100-byte
//! file takes 4 KiB, not a whole page's slot. The budget counts the slots
//! in use; a freed slot's disk space is returned to the filesystem (hole
//! punching on Linux) and its position reused by the next page of its class.
//!
//! Memory per cached page is one slot record (16 bytes) and its place in
//! the CLOCK ring (4 bytes), plus 4 bytes per page of each file the tier has
//! seen (a dense page-to-slot table). There is no file per page: slot `i` of
//! class `c` lives at byte `(i % SLOTS_PER_SLAB) * len(c)` of that class's
//! slab file `i / SLOTS_PER_SLAB`.
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
/// A slot id is its class in the top bits and its index below.
const CLASS_SHIFT: u32 = 28;
const INDEX_MASK: u32 = (1 << CLASS_SHIFT) - 1;
/// The smallest slot.
const MIN_SLOT: u32 = 4096;

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
    /// Reads since the page was stored, up to `u8::MAX`.
    hits: u8,
}

const REFERENCED: u8 = 1;
const PINNED: u8 = 2;
const WRITING: u8 = 4;

/// A cached page, as listed by [`DiskTier::resident`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resident {
    /// The file's cache key.
    pub key: Arc<str>,
    /// The page of that file.
    pub page: u64,
    /// Bytes stored.
    pub len: u32,
    /// Reads since it was stored, up to `u8::MAX`.
    pub hits: u8,
}

/// A file the tier has seen: its page-to-slot table.
struct FileSlots {
    pages: Vec<u32>,
    pinned: bool,
}

/// The slots of one size.
#[derive(Default)]
struct Class {
    slots: Vec<Slot>,
    /// Indexes of free slots (their disk space already released).
    free: Vec<u32>,
}

#[derive(Default)]
struct State {
    ids: HashMap<Arc<str>, u32>,
    files: HashMap<u32, FileSlots>,
    next_id: u32,
    classes: Vec<Class>,
    /// Every slot ever made, in order: what the CLOCK hand walks.
    ring: Vec<u32>,
    hand: usize,
    /// Bytes stored in live slots.
    used: u64,
    /// Bytes of the slots in use (each its class's size): what the budget
    /// bounds.
    footprint: u64,
    /// The part of `footprint` in slots CLOCK cannot evict (pinned or
    /// being written).
    held: u64,
}

fn split(slot: u32) -> (usize, usize) {
    ((slot >> CLASS_SHIFT) as usize, (slot & INDEX_MASK) as usize)
}

fn join(class: usize, index: usize) -> Option<u32> {
    let c = u32::try_from(class)
        .ok()
        .filter(|c| *c < 1 << (u32::BITS - CLASS_SHIFT))?;
    let i = u32::try_from(index).ok().filter(|i| *i <= INDEX_MASK)?;
    let id = (c << CLASS_SHIFT) | i;
    (id != NONE).then_some(id)
}

impl State {
    fn slot(&self, slot: u32) -> Option<&Slot> {
        let (c, i) = split(slot);
        self.classes.get(c)?.slots.get(i)
    }

    fn slot_mut(&mut self, slot: u32) -> Option<&mut Slot> {
        let (c, i) = split(slot);
        self.classes.get_mut(c)?.slots.get_mut(i)
    }
}

pub(crate) struct DiskTier {
    root: PathBuf,
    /// Each class's slot length, ascending; the last holds a whole page.
    class_len: Vec<u32>,
    budget: u64,
    state: Mutex<State>,
    slabs: Mutex<Vec<Vec<Arc<File>>>>,
}

impl DiskTier {
    /// A tier under `root` (emptied) of at most `budget` bytes, its largest
    /// slots big enough for pages of `page_size` plus a decoder's per-unit
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
        let mut class_len = Vec::new();
        let mut len = MIN_SLOT;
        while len < slot_len {
            class_len.push(len);
            len = len.saturating_mul(2);
        }
        class_len.push(slot_len);
        // A slot id keeps its class in the bits above `CLASS_SHIFT`: a
        // class past them would alias another's slots.
        if class_len.len() > 1 << (u32::BITS - CLASS_SHIFT) {
            return Err(Error::IllegalArgument(
                "page_size too large for the disk tier".into(),
            ));
        }
        let classes = (0..class_len.len()).map(|_| Class::default()).collect();
        Ok(DiskTier {
            root: root.to_path_buf(),
            slabs: Mutex::new(vec![Vec::new(); class_len.len()]),
            class_len,
            budget,
            state: Mutex::new(State {
                classes,
                ..State::default()
            }),
        })
    }

    /// Bytes of pages held.
    pub(crate) fn used(&self) -> u64 {
        lock_ignoring_poison(&self.state).used
    }

    /// Bytes of the slots holding them, which the budget bounds.
    pub(crate) fn footprint(&self) -> u64 {
        lock_ignoring_poison(&self.state).footprint
    }

    /// Where `slot` lives: its slab file (created on first use) and offset.
    fn slab(&self, slot: u32) -> std::io::Result<(Arc<File>, u64)> {
        let (class, index) = split(slot);
        let len = u64::from(self.class_len[class]);
        let n = (index as u32 / SLOTS_PER_SLAB) as usize;
        let offset = u64::from(index as u32 % SLOTS_PER_SLAB).saturating_mul(len);
        let mut slabs = lock_ignoring_poison(&self.slabs);
        let files = &mut slabs[class];
        while files.len() <= n {
            let path = self.root.join(format!("slab-{class}-{}", files.len()));
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)?;
            files.push(Arc::new(file));
        }
        Ok((Arc::clone(&files[n]), offset))
    }

    /// Gives `slot`'s disk space back to the filesystem.
    fn discard(&self, slot: u32) {
        let (class, _) = split(slot);
        let len = u64::from(self.class_len[class]);
        if let Ok((file, offset)) = self.slab(slot) {
            punch_hole(&file, offset, len);
        }
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
        self.read(key, page, true)
    }

    /// The bytes of a cached page, without counting a read or sparing the
    /// page from eviction: for a page handed to another node.
    pub(crate) fn peek(&self, key: &str, page: u64) -> Option<Arc<[u8]>> {
        let key: Arc<str> = {
            let st = lock_ignoring_poison(&self.state);
            Arc::clone(st.ids.get_key_value(key)?.0)
        };
        self.read(&key, page, false)
    }

    /// For each of `prefixes`, up to `limit` cached pages of files whose key
    /// starts with it, the most read first (ties by file and page); one
    /// pass over the tier under one hold of its lock.
    pub(crate) fn resident(&self, prefixes: &[&str], limit: usize) -> Vec<Vec<Resident>> {
        let st = lock_ignoring_poison(&self.state);
        let mut out: Vec<Vec<Resident>> = prefixes.iter().map(|_| Vec::new()).collect();
        for (key, id) in &st.ids {
            let Some(at) = prefixes.iter().position(|p| key.starts_with(p)) else {
                continue;
            };
            let Some(file) = st.files.get(id) else {
                continue;
            };
            for (page, &slot) in file.pages.iter().enumerate() {
                if slot == NONE {
                    continue;
                }
                let Some(rec) = st
                    .slot(slot)
                    .filter(|r| r.file == *id && r.flags & WRITING == 0)
                else {
                    continue;
                };
                out[at].push(Resident {
                    key: Arc::clone(key),
                    page: page as u64,
                    len: rec.len,
                    hits: rec.hits,
                });
            }
        }
        for pages in &mut out {
            pages.sort_by(|a, b| {
                b.hits
                    .cmp(&a.hits)
                    .then_with(|| a.key.cmp(&b.key))
                    .then(a.page.cmp(&b.page))
            });
            pages.truncate(limit);
        }
        out
    }

    fn read(&self, key: &Arc<str>, page: u64, count: bool) -> Option<Arc<[u8]>> {
        let (slot, generation, len) = {
            let mut st = lock_ignoring_poison(&self.state);
            let slot = Self::slot_of(&st, key, page)?;
            let rec = st.slot_mut(slot)?;
            if count {
                rec.flags |= REFERENCED;
                rec.hits = rec.hits.saturating_add(1);
            }
            (slot, rec.generation, rec.len)
        };
        let mut buf = vec![0u8; len as usize];
        let read = self
            .slab(slot)
            .and_then(|(file, offset)| read_exact_at(&file, &mut buf, offset));
        let mut st = lock_ignoring_poison(&self.state);
        let same = st
            .slot(slot)
            .is_some_and(|r| r.generation == generation && r.flags & WRITING == 0);
        if read.is_err() && same {
            // An unreadable slot is dropped, so the page is fetched and
            // cached again rather than failing every later read.
            self.free(&mut st, slot);
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
        let Some(class) = self.class_len.iter().position(|&c| len <= c) else {
            return;
        };
        let (slot, id, generation) = {
            let mut st = lock_ignoring_poison(&self.state);
            if Self::slot_of(&st, key, page).is_some() {
                return;
            }
            let id = Self::id_of(&mut st, key);
            let pinned = st.files.get(&id).is_some_and(|f| f.pinned);
            let Some(slot) = self.allocate(&mut st, class, pinned) else {
                return;
            };
            let Some(rec) = st.slot_mut(slot) else {
                return;
            };
            rec.file = id;
            rec.page = page32;
            rec.len = len;
            rec.generation = rec.generation.wrapping_add(1);
            rec.flags = WRITING | if pinned { PINNED } else { 0 };
            rec.hits = 0;
            let generation = rec.generation;
            st.held = st.held.saturating_add(u64::from(self.class_len[class]));
            (slot, id, generation)
        };
        let written = self
            .slab(slot)
            .and_then(|(file, offset)| write_all_at(&file, bytes, offset));
        let mut st = lock_ignoring_poison(&self.state);
        let ours = st.slot(slot).is_some_and(|r| r.generation == generation);
        let alive = ours && st.ids.get(key) == Some(&id);
        // Not published when the write failed, the file was forgotten
        // meanwhile, or another put of the same page published first
        // (keeping this slot too would leave it unreachable).
        if written.is_err() || !alive || Self::slot_of(&st, key, page).is_some() {
            if ours {
                self.free(&mut st, slot);
            }
            return;
        }
        // The file may have been pinned while this page was written.
        let pinned = st.files.get(&id).is_some_and(|f| f.pinned);
        let released = st.slot_mut(slot).is_some_and(|rec| {
            rec.flags &= !WRITING;
            if pinned {
                rec.flags |= PINNED;
            }
            rec.flags & PINNED == 0
        });
        if released {
            st.held = st.held.saturating_sub(u64::from(self.class_len[class]));
        }
        st.used = st.used.saturating_add(u64::from(len));
        let p = page as usize;
        if let Some(f) = st.files.get_mut(&id) {
            if f.pages.len() <= p {
                f.pages.resize(p.saturating_add(1), NONE);
            }
            f.pages[p] = slot;
        }
    }

    /// A slot of `class`: room is made under the budget by CLOCK, then a
    /// free slot of the class is reused or a new one made. A pinned page
    /// gets one past the budget when nothing can be evicted; any other page
    /// then gets `None`.
    fn allocate(&self, st: &mut State, class: usize, pinned: bool) -> Option<u32> {
        let need = u64::from(self.class_len[class]);
        let mut evicted = Vec::new();
        if st.footprint.saturating_add(need) > self.budget {
            // Nothing is evicted for a page that would not fit even then.
            let can_fit = st.held.saturating_add(need) <= self.budget;
            if !pinned && !can_fit {
                return None;
            }
            if !self.evict(st, need, &mut evicted) && !pinned {
                self.discard_all(&evicted);
                return None;
            }
        }
        let slot = match st.classes[class].free.pop() {
            Some(i) => join(class, i as usize)?,
            None => {
                let index = st.classes[class].slots.len();
                let slot = join(class, index)?;
                st.classes[class].slots.push(Slot {
                    file: NONE,
                    ..Slot::default()
                });
                st.ring.push(slot);
                slot
            }
        };
        st.footprint = st.footprint.saturating_add(need);
        // The evicted slots not reused at once give their space back; the
        // one reused is about to be overwritten.
        evicted.retain(|&e| e != slot);
        self.discard_all(&evicted);
        Some(slot)
    }

    fn discard_all(&self, slots: &[u32]) {
        for &s in slots {
            self.discard(s);
        }
    }

    /// Evicts unpinned, unreferenced pages until `need` more bytes fit
    /// under the budget, their slots added to `evicted` with their disk
    /// space not yet released; `false` when they cannot be made to fit.
    fn evict(&self, st: &mut State, need: u64, evicted: &mut Vec<u32>) -> bool {
        let total = st.ring.len();
        // Two sweeps: the first may only clear reference bits.
        for _ in 0..total.saturating_mul(2) {
            if st.footprint.saturating_add(need) <= self.budget {
                return true;
            }
            let slot = st.ring[st.hand];
            st.hand = st.hand.saturating_add(1);
            if st.hand >= total {
                st.hand = 0;
            }
            let Some(rec) = st.slot_mut(slot) else {
                continue;
            };
            if rec.file == NONE || rec.flags & (PINNED | WRITING) != 0 {
                continue;
            }
            if rec.flags & REFERENCED != 0 {
                rec.flags &= !REFERENCED;
                continue;
            }
            self.unmap(st, slot);
            evicted.push(slot);
        }
        st.footprint.saturating_add(need) <= self.budget
    }

    /// Unmaps `slot` from its page and frees it, its disk space released.
    fn free(&self, st: &mut State, slot: u32) {
        if self.unmap(st, slot) {
            self.discard(slot);
        }
    }

    /// Unmaps `slot` from its page and puts it on its class's free list,
    /// its disk space kept; whether it was a slot.
    fn unmap(&self, st: &mut State, slot: u32) -> bool {
        let Some(rec) = st.slot(slot).copied() else {
            return false;
        };
        if rec.file != NONE && rec.flags & WRITING == 0 {
            if let Some(f) = st.files.get_mut(&rec.file) {
                if let Some(e) = f.pages.get_mut(rec.page as usize) {
                    if *e == slot {
                        *e = NONE;
                    }
                }
            }
            st.used = st.used.saturating_sub(u64::from(rec.len));
        }
        if let Some(r) = st.slot_mut(slot) {
            r.file = NONE;
            r.flags = 0;
            r.generation = rec.generation.wrapping_add(1);
        }
        let (class, index) = split(slot);
        let size = u64::from(self.class_len[class]);
        st.footprint = st.footprint.saturating_sub(size);
        if rec.flags & (PINNED | WRITING) != 0 {
            st.held = st.held.saturating_sub(size);
        }
        st.classes[class].free.push(index as u32);
        true
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
            let size = u64::from(self.class_len[split(s).0]);
            let newly_held = st.slot_mut(s).is_some_and(|r| {
                let was_held = r.flags & (PINNED | WRITING) != 0;
                r.flags |= PINNED;
                !was_held
            });
            if newly_held {
                st.held = st.held.saturating_add(size);
            }
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
                if st.slot(s).is_some_and(|r| r.file == id) {
                    self.free(&mut st, s);
                }
            }
        }
    }

    /// RAM held by the tier's records, for tests and reporting.
    pub(crate) fn metadata_bytes(&self) -> usize {
        let st = lock_ignoring_poison(&self.state);
        let pages: usize = st.files.values().map(|f| f.pages.capacity()).sum();
        let slots: usize = st
            .classes
            .iter()
            .map(|c| {
                c.slots
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Slot>())
                    .saturating_add(c.free.capacity().saturating_mul(4))
            })
            .sum();
        slots
            .saturating_add(pages.saturating_mul(4))
            .saturating_add(st.ring.capacity().saturating_mul(4))
    }
}

/// Releases `len` bytes at `offset` of `file` to the filesystem, keeping
/// its size. Best effort: where it is unsupported the space stays allocated
/// until the slot is reused.
#[cfg(target_os = "linux")]
fn punch_hole(file: &File, offset: u64, len: u64) {
    use std::os::fd::AsRawFd;
    let (Ok(offset), Ok(len)) = (libc::off_t::try_from(offset), libc::off_t::try_from(len)) else {
        return;
    };
    // SAFETY: `fallocate` on an open descriptor this tier owns; a failure
    // only leaves the space allocated.
    unsafe {
        libc::fallocate(
            file.as_raw_fd(),
            libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
            offset,
            len,
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn punch_hole(file: &File, offset: u64, len: u64) {
    let _ = (file, offset, len);
}

#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)?;
    drop_cached(file, offset, buf.len());
    Ok(())
}

#[cfg(unix)]
fn write_all_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, buf, offset)?;
    drop_cached(file, offset, buf.len());
    Ok(())
}

/// Asks the kernel not to keep a slot's bytes in its page cache: a page
/// read or written here goes on into the memory tier (the pager's budget),
/// so a second copy in the kernel's cache would hold the same bytes twice.
#[cfg(target_os = "linux")]
fn drop_cached(file: &File, offset: u64, len: usize) {
    use std::os::fd::AsRawFd;
    let (Ok(offset), Ok(len)) = (libc::off_t::try_from(offset), libc::off_t::try_from(len)) else {
        return;
    };
    // SAFETY: a hint on an open descriptor; it changes no file contents.
    unsafe { libc::posix_fadvise(file.as_raw_fd(), offset, len, libc::POSIX_FADV_DONTNEED) };
}

#[cfg(all(unix, not(target_os = "linux")))]
fn drop_cached(_file: &File, _offset: u64, _len: usize) {}

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
mod tests;
