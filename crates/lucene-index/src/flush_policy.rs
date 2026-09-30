//! Port of `org.apache.lucene.index.FlushPolicy` and
//! `FlushByRamOrCountsPolicy`: when an indexing buffer
//! (`DocumentsWriterPerThread`, a [`crate::concurrent_writer`] slot) is
//! marked for flushing, and when the buffered deletes are applied instead.
//!
//! Java's `FlushPolicy` is package-private and `FlushByRamOrCountsPolicy` its
//! one implementation (`IndexWriterConfig.setFlushPolicy` is package-private
//! too), so the trait here is the same seam and nothing more. The policy
//! decides against a [`FlushControl`] -- the part of
//! `DocumentsWriterFlushControl` it reads and writes: the RAM held by the
//! buffers not yet marked (`activeBytes`), the RAM held by buffered deletes
//! (`getDeleteBytesUsed`), each buffer's document count and RAM, and the
//! `flushPending`/`applyAllDeletes` flags. The writer owns the control under
//! one lock, as Java's is `synchronized`, and acts on the flags it leaves.

/// One indexing buffer as the flush control sees it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SlotState {
    /// `getNumDocsInRAM()`.
    pub num_docs: usize,
    /// `getLastCommittedBytesUsed()`.
    pub ram_bytes: usize,
    /// `isFlushPending()`.
    pub flush_pending: bool,
}

/// `DocumentsWriterFlushControl`, the part a [`FlushPolicy`] sees.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FlushControl {
    slots: Vec<SlotState>,
    active_bytes: usize,
    flush_bytes: usize,
    delete_bytes: usize,
    apply_all_deletes: bool,
}

impl FlushControl {
    /// A control over `slots` empty buffers.
    pub fn new(slots: usize) -> Self {
        Self {
            slots: vec![SlotState::default(); slots],
            ..Self::default()
        }
    }

    /// `activeBytes()`: RAM in the buffers not marked for flushing.
    pub fn active_bytes(&self) -> usize {
        self.active_bytes
    }

    /// `flushBytes()`: RAM in buffers marked for flushing or being flushed.
    pub fn flush_bytes(&self) -> usize {
        self.flush_bytes
    }

    /// `getDeleteBytesUsed()`.
    pub fn delete_bytes_used(&self) -> usize {
        self.delete_bytes
    }

    /// Sets what the buffered deletes hold.
    pub fn set_delete_bytes_used(&mut self, bytes: usize) {
        self.delete_bytes = bytes;
    }

    /// Every buffer's state.
    pub fn slots(&self) -> &[SlotState] {
        &self.slots
    }

    /// Buffer `slot` now holds `num_docs` documents in `ram_bytes`
    /// (`DocumentsWriterFlushControl.commitPerThreadBytes`): the change counts
    /// toward `activeBytes`, or toward `flushBytes` once it is marked.
    pub fn commit_slot_bytes(&mut self, slot: usize, num_docs: usize, ram_bytes: usize) {
        let Some(s) = self.slots.get_mut(slot) else {
            return;
        };
        let (total, before) = if s.flush_pending {
            (&mut self.flush_bytes, s.ram_bytes)
        } else {
            (&mut self.active_bytes, s.ram_bytes)
        };
        *total = total.saturating_sub(before).saturating_add(ram_bytes);
        s.num_docs = num_docs;
        s.ram_bytes = ram_bytes;
    }

    /// `setFlushPending(perThread)`: marks a buffer holding documents, moving
    /// its RAM from `activeBytes` to `flushBytes`.
    pub fn set_flush_pending(&mut self, slot: usize) {
        let Some(s) = self.slots.get_mut(slot) else {
            return;
        };
        if s.flush_pending || s.num_docs == 0 {
            return;
        }
        s.flush_pending = true;
        self.active_bytes = self.active_bytes.saturating_sub(s.ram_bytes);
        self.flush_bytes = self.flush_bytes.saturating_add(s.ram_bytes);
    }

    /// Buffer `slot` was taken for its segment (`checkoutForFlush`): its RAM
    /// now counts toward `flushBytes` until [`Self::flush_done`], and the slot
    /// starts again empty and unmarked.
    pub fn checkout_for_flush(&mut self, slot: usize) -> usize {
        let Some(s) = self.slots.get_mut(slot) else {
            return 0;
        };
        let bytes = s.ram_bytes;
        if !s.flush_pending {
            self.active_bytes = self.active_bytes.saturating_sub(bytes);
            self.flush_bytes = self.flush_bytes.saturating_add(bytes);
        }
        *s = SlotState::default();
        bytes
    }

    /// `doAfterFlush`: a checked-out buffer's RAM is released.
    pub fn flush_done(&mut self, bytes: usize) {
        self.flush_bytes = self.flush_bytes.saturating_sub(bytes);
    }

    /// `isFlushPending()` of buffer `slot`.
    pub fn is_flush_pending(&self, slot: usize) -> bool {
        self.slots.get(slot).is_some_and(|s| s.flush_pending)
    }

    /// `setApplyAllDeletes()`.
    pub fn set_apply_all_deletes(&mut self) {
        self.apply_all_deletes = true;
    }

    /// `getAndResetApplyAllDeletes()`.
    pub fn get_and_reset_apply_all_deletes(&mut self) -> bool {
        std::mem::take(&mut self.apply_all_deletes)
    }

    /// `findLargestNonPendingWriter()`: the unmarked buffer holding documents
    /// with the most RAM, the first of equals.
    pub fn find_largest_non_pending_writer(&self) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for (i, s) in self.slots.iter().enumerate() {
            if !s.flush_pending && s.num_docs > 0 && best.is_none_or(|(_, ram)| s.ram_bytes > ram) {
                best = Some((i, s.ram_bytes));
            }
        }
        best.map(|(i, _)| i)
    }

    /// `numFlushingDWPT`/`numQueuedFlushes`: buffers marked and not yet taken.
    pub fn num_pending(&self) -> usize {
        self.slots.iter().filter(|s| s.flush_pending).count()
    }
}

/// `FlushPolicy`: called after every change to a buffer (`slot` is the
/// buffer an add just went into) and after every buffered delete (`slot` is
/// `None`), under the control's lock.
pub trait FlushPolicy: Send + Sync + std::fmt::Debug {
    /// `onChange(control, perThread)`.
    fn on_change(&self, control: &mut FlushControl, slot: Option<usize>);
}

/// `FlushByRamOrCountsPolicy`: a buffer is marked once it holds
/// `maxBufferedDocs` documents; otherwise, with a RAM buffer, the largest
/// unmarked buffer is marked once buffers and deletes together reach it, and
/// deletes alone reaching it apply them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlushByRamOrCountsPolicy {
    /// `getMaxBufferedDocs()`, `None` for `DISABLE_AUTO_FLUSH`.
    pub max_buffered_docs: Option<usize>,
    /// `getRAMBufferSizeMB()` in bytes, `None` for `DISABLE_AUTO_FLUSH`.
    pub ram_buffer_bytes: Option<usize>,
}

impl FlushPolicy for FlushByRamOrCountsPolicy {
    fn on_change(&self, control: &mut FlushControl, slot: Option<usize>) {
        if let Some(i) = slot {
            if self
                .max_buffered_docs
                .is_some_and(|max| control.slots.get(i).is_some_and(|s| s.num_docs >= max))
            {
                control.set_flush_pending(i);
                return;
            }
        }
        let Some(limit) = self.ram_buffer_bytes else {
            return;
        };
        let active = control.active_bytes();
        let deletes = control.delete_bytes_used();
        if deletes >= limit && active >= limit && slot.is_some() {
            control.set_apply_all_deletes();
            mark_largest_writer_pending(control);
        } else if deletes >= limit {
            control.set_apply_all_deletes();
        } else if active.saturating_add(deletes) >= limit && slot.is_some() {
            mark_largest_writer_pending(control);
        }
    }
}

/// `markLargestWriterPending`.
fn mark_largest_writer_pending(control: &mut FlushControl) {
    if let Some(largest) = control.find_largest_non_pending_writer() {
        control.set_flush_pending(largest);
    }
}

#[cfg(test)]
mod tests;
