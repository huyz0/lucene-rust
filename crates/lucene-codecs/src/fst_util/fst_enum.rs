//! Port of `org.apache.lucene.util.fst.FSTEnum` and `BytesRefFSTEnum` over
//! a [`TypedFst`]: `next`, `seekCeil` and `seekExact` through any
//! [`FstOutputs`], every node encoding (list, binary search, direct
//! addressing, continuous) walked as Java walks it.
//!
//! `crate::fst::FstEnum` does the same over byte-sequence outputs only;
//! this one is what a term dictionary whose FST carries its own outputs
//! (`FSTTermsReader`'s `TermData`) iterates with. `seekFloor` has no caller
//! and is not ported.

use super::{Arc, Result, TypedFst, END_LABEL};
use crate::fst::{ARCS_FOR_BINARY_SEARCH, ARCS_FOR_DIRECT_ADDRESSING};
use crate::fst_compiler::{FstOutputs, FstReadError};

/// `BytesRefFSTEnum<T>`.
#[derive(Debug)]
pub struct BytesRefFstEnum<'f, O: FstOutputs> {
    fst: &'f TypedFst<O>,
    /// `arcs[0..=upto]`; `arcs[0]` is the FST's first arc.
    arcs: Vec<Arc<O::Value>>,
    /// `output[i]`: the output accumulated through `arcs[i]`.
    output: Vec<O::Value>,
    upto: usize,
    /// `current.bytes`, `offset` fixed at 1: the label taken at depth `i`
    /// is `current[i]`.
    current: Vec<u8>,
    target: Vec<u8>,
}

impl<'f, O: FstOutputs> BytesRefFstEnum<'f, O> {
    /// `new BytesRefFSTEnum<>(fst)`.
    pub fn new(fst: &'f TypedFst<O>) -> Self {
        BytesRefFstEnum {
            fst,
            arcs: vec![fst.first_arc()],
            output: vec![O::no_output()],
            upto: 0,
            current: vec![0; 10],
            target: Vec::new(),
        }
    }

    /// `next()`: the next term and its output, or `None` past the last.
    // A cursor that also seeks, as Java's `TermsEnum`, not an `Iterator`.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<(&[u8], &O::Value)>> {
        self.do_next()?;
        Ok(self.result())
    }

    /// `seekCeil(target)`: the smallest term `>= target`.
    pub fn seek_ceil(&mut self, target: &[u8]) -> Result<Option<(&[u8], &O::Value)>> {
        self.set_target(target);
        self.do_seek_ceil()?;
        Ok(self.result())
    }

    /// `seekExact(target)`: `target` and its output, or `None` when absent
    /// (the enum is then left on a prefix of it, as Java's).
    pub fn seek_exact(&mut self, target: &[u8]) -> Result<Option<(&[u8], &O::Value)>> {
        self.set_target(target);
        if self.do_seek_exact()? {
            Ok(self.result())
        } else {
            Ok(None)
        }
    }

    /// `current()`: the term and output the last call landed on.
    pub fn current(&self) -> Option<(&[u8], &O::Value)> {
        self.result()
    }

    fn set_target(&mut self, target: &[u8]) {
        self.target.clear();
        self.target.extend_from_slice(target);
    }

    /// `setResult()`.
    // ARITH: `upto >= 1` in the branch taken, and `current` holds `upto + 1`
    // bytes (grown by `incr`).
    #[allow(clippy::arithmetic_side_effects)]
    fn result(&self) -> Option<(&[u8], &O::Value)> {
        if self.upto == 0 {
            None
        } else {
            Some((&self.current[1..self.upto], &self.output[self.upto]))
        }
    }

    /// `getTargetLabel()`.
    // ARITH: `upto >= 1` whenever a target label is asked for (the seeks
    // start at depth 1).
    #[allow(clippy::arithmetic_side_effects)]
    fn target_label(&self) -> i32 {
        let at = self.upto - 1;
        if at == self.target.len() {
            END_LABEL
        } else {
            i32::from(self.target[at])
        }
    }

    /// `getCurrentLabel()`.
    fn current_label(&self) -> i32 {
        i32::from(self.current[self.upto])
    }

    /// `setCurrentLabel(label)`: the low byte, as Java's `(byte) label`
    /// stores `END_LABEL` as `0xFF`.
    fn set_current_label(&mut self, label: i32) {
        self.current[self.upto] = label as u8;
    }

    /// `getArc(idx)`.
    fn ensure_arc(&mut self, idx: usize) {
        while self.arcs.len() <= idx {
            self.arcs.push(Arc::new(O::no_output()));
        }
    }

    /// `incr()`.
    // ARITH: the depth of a term, bounded by its length.
    #[allow(clippy::arithmetic_side_effects)]
    fn incr(&mut self) {
        self.upto += 1;
        if self.current.len() <= self.upto {
            self.current.resize(self.upto + 1, 0);
        }
        self.ensure_arc(self.upto);
        while self.output.len() <= self.upto {
            self.output.push(O::no_output());
        }
    }

    /// `output[upto] = add(output[upto - 1], arcs[upto].output())`.
    // ARITH: called at depth `upto >= 1`.
    #[allow(clippy::arithmetic_side_effects)]
    fn accumulate(&mut self) {
        let sum = O::add(&self.output[self.upto - 1], self.arcs[self.upto].output());
        // Java's `output` starts with room for 10; depth 1 is reached
        // without `incr()` (`doNext`/`rewindPrefix` from the root).
        while self.output.len() <= self.upto {
            self.output.push(O::no_output());
        }
        self.output[self.upto] = sum;
    }

    /// `fst.readFirstTargetArc(arcs[from], arcs[from + 1], in)`.
    // ARITH: `from + 1` is a depth `ensure_arc` has made.
    #[allow(clippy::arithmetic_side_effects)]
    fn read_first_target(&mut self, from: usize) -> Result<()> {
        self.ensure_arc(from + 1);
        let (lo, hi) = self.arcs.split_at_mut(from + 1);
        let mut input = self.fst.bytes_reader();
        self.fst
            .read_first_target_arc(&lo[from], &mut hi[0], &mut input)
    }

    /// `fst.readNextArc(arcs[idx], in)`.
    fn read_next(&mut self, idx: usize) -> Result<()> {
        let mut input = self.fst.bytes_reader();
        self.fst.read_next_arc(&mut self.arcs[idx], &mut input)
    }

    /// `rewindPrefix()`.
    // ARITH: depths bounded by the current term's and the target's lengths.
    #[allow(clippy::arithmetic_side_effects)]
    fn rewind_prefix(&mut self) -> Result<()> {
        if self.upto == 0 {
            self.upto = 1;
            return self.read_first_target(0);
        }
        let current_limit = self.upto;
        self.upto = 1;
        while self.upto < current_limit && self.upto <= self.target.len() + 1 {
            let cmp = self.current_label() - self.target_label();
            if cmp < 0 {
                // Seek forward.
                break;
            } else if cmp > 0 {
                // Seek backwards: reset this arc to the first arc.
                return self.read_first_target(self.upto - 1);
            }
            self.upto += 1;
        }
        Ok(())
    }

    /// `doNext()`.
    // ARITH: `upto` steps down while positive.
    #[allow(clippy::arithmetic_side_effects)]
    fn do_next(&mut self) -> Result<()> {
        if self.upto == 0 {
            self.upto = 1;
            self.read_first_target(0)?;
        } else {
            while self.arcs[self.upto].is_last() {
                self.upto -= 1;
                if self.upto == 0 {
                    return Ok(());
                }
            }
            self.read_next(self.upto)?;
        }
        self.push_first()
    }

    /// The arc at `upto` matched the target label: accumulate its output
    /// and, unless it ends the term, descend to its first target arc.
    /// `true` while the seek goes on (Java returning the next arc).
    fn take_matched(&mut self, target_label: i32) -> Result<bool> {
        self.accumulate();
        if target_label == END_LABEL {
            return Ok(false);
        }
        let label = self.arcs[self.upto].label();
        self.set_current_label(label);
        self.incr();
        let from = self.upto.saturating_sub(1);
        self.read_first_target(from)?;
        Ok(true)
    }

    /// `doSeekCeil()`.
    fn do_seek_ceil(&mut self) -> Result<()> {
        self.rewind_prefix()?;
        loop {
            let target_label = self.target_label();
            let arc = &self.arcs[self.upto];
            let go_on = if arc.bytes_per_arc() != 0 && arc.label() != END_LABEL {
                match arc.node_flags() {
                    ARCS_FOR_DIRECT_ADDRESSING => self.seek_ceil_direct(target_label)?,
                    ARCS_FOR_BINARY_SEARCH => self.seek_ceil_packed(target_label)?,
                    _ => self.seek_ceil_continuous(target_label)?,
                }
            } else {
                self.seek_ceil_list(target_label)?
            };
            if !go_on {
                return Ok(());
            }
        }
    }

    /// `doSeekCeilArrayContinuous`.
    fn seek_ceil_continuous(&mut self, target_label: i32) -> Result<bool> {
        let arc = &self.arcs[self.upto];
        // Two `i32`s, so their difference fits an `i64`.
        let target_index = i64::from(target_label).wrapping_sub(i64::from(arc.first_label));
        if target_index >= i64::from(arc.num_arcs()) {
            self.rollback_to_last_fork_then_push()?;
            return Ok(false);
        }
        let mut input = self.fst.bytes_reader();
        if target_index < 0 {
            self.fst
                .read_arc_by_continuous(&mut self.arcs[self.upto], &mut input, 0)?;
            self.push_first()?;
            return Ok(false);
        }
        self.fst.read_arc_by_continuous(
            &mut self.arcs[self.upto],
            &mut input,
            target_index as i32,
        )?;
        self.take_matched(target_label)
    }

    /// `doSeekCeilArrayDirectAddressing`.
    fn seek_ceil_direct(&mut self, target_label: i32) -> Result<bool> {
        let arc = &self.arcs[self.upto];
        // Two `i32`s, so their difference fits an `i64`.
        let target_index = i64::from(target_label).wrapping_sub(i64::from(arc.first_label));
        if target_index >= i64::from(arc.num_arcs()) {
            self.rollback_to_last_fork_then_push()?;
            return Ok(false);
        }
        let mut input = self.fst.bytes_reader();
        if target_index < 0 {
            // The first arc is already past the target.
            self.fst
                .read_arc_by_direct_addressing(&mut self.arcs[self.upto], &mut input, 0)?;
            self.push_first()?;
            return Ok(false);
        }
        let idx = target_index as i32;
        if self.fst.bit_is_set(idx, arc, &mut input)? {
            self.fst
                .read_arc_by_direct_addressing(&mut self.arcs[self.upto], &mut input, idx)?;
            return self.take_matched(target_label);
        }
        let ceil = self
            .fst
            .bit_next_set(idx, &self.arcs[self.upto], &mut input)?;
        if ceil < 0 {
            return Err(FstReadError(
                "direct-addressing node without a ceiling arc".into(),
            ));
        }
        self.fst
            .read_arc_by_direct_addressing(&mut self.arcs[self.upto], &mut input, ceil)?;
        self.push_first()?;
        Ok(false)
    }

    /// `doSeekCeilArrayPacked`.
    // ARITH: `idx` is a binary search's insertion point, within the node.
    #[allow(clippy::arithmetic_side_effects)]
    fn seek_ceil_packed(&mut self, target_label: i32) -> Result<bool> {
        let idx = super::binary_search(self.fst, &self.arcs[self.upto], target_label)?;
        let mut input = self.fst.bytes_reader();
        if idx >= 0 {
            self.fst
                .read_arc_by_index(&mut self.arcs[self.upto], &mut input, idx)?;
            return self.take_matched(target_label);
        }
        let idx = -1 - idx;
        if idx == self.arcs[self.upto].num_arcs() {
            // Dead end.
            self.fst
                .read_arc_by_index(&mut self.arcs[self.upto], &mut input, idx - 1)?;
            self.upto -= 1;
            self.next_fork()?;
            Ok(false)
        } else {
            self.fst
                .read_arc_by_index(&mut self.arcs[self.upto], &mut input, idx)?;
            self.push_first()?;
            Ok(false)
        }
    }

    /// `doSeekCeilList`.
    // ARITH: `upto >= 1` at every seek step.
    #[allow(clippy::arithmetic_side_effects)]
    fn seek_ceil_list(&mut self, target_label: i32) -> Result<bool> {
        let arc = &self.arcs[self.upto];
        if arc.label() == target_label {
            self.take_matched(target_label)
        } else if arc.label() > target_label {
            self.push_first()?;
            Ok(false)
        } else if arc.is_last() {
            self.upto -= 1;
            self.next_fork()?;
            Ok(false)
        } else {
            self.read_next(self.upto)?;
            Ok(true)
        }
    }

    /// The dead-end walk shared by `doSeekCeilArrayPacked`,
    /// `doSeekCeilList` and `rollbackToLastForkThenPush`: up to the first
    /// ancestor with a next arc, then down its first path.
    // ARITH: `upto` steps down while positive.
    #[allow(clippy::arithmetic_side_effects)]
    fn next_fork(&mut self) -> Result<()> {
        loop {
            if self.upto == 0 {
                return Ok(());
            }
            if !self.arcs[self.upto].is_last() {
                self.read_next(self.upto)?;
                return self.push_first();
            }
            self.upto -= 1;
        }
    }

    /// `rollbackToLastForkThenPush()`.
    // ARITH: called at depth `upto >= 1`.
    #[allow(clippy::arithmetic_side_effects)]
    fn rollback_to_last_fork_then_push(&mut self) -> Result<()> {
        self.upto -= 1;
        self.next_fork()
    }

    /// `doSeekExact()`.
    // ARITH: `upto >= 1` after `rewind_prefix`.
    #[allow(clippy::arithmetic_side_effects)]
    fn do_seek_exact(&mut self) -> Result<bool> {
        self.rewind_prefix()?;
        let mut target_label = self.target_label();
        loop {
            self.ensure_arc(self.upto);
            let mut input = self.fst.bytes_reader();
            let found =
                self.fst
                    .find_target_arc(target_label, &self.arcs[self.upto - 1], &mut input)?;
            match found {
                None => {
                    // Short circuit: reset this arc to the first arc.
                    self.read_first_target(self.upto - 1)?;
                    return Ok(false);
                }
                Some(next) => self.arcs[self.upto] = next,
            }
            self.accumulate();
            if target_label == END_LABEL {
                return Ok(true);
            }
            self.set_current_label(target_label);
            self.incr();
            target_label = self.target_label();
        }
    }

    /// `pushFirst()`: from the arc at `upto`, the first path down to a
    /// term's end.
    fn push_first(&mut self) -> Result<()> {
        loop {
            self.accumulate();
            let label = self.arcs[self.upto].label();
            if label == END_LABEL {
                return Ok(());
            }
            self.set_current_label(label);
            self.incr();
            let from = self.upto.saturating_sub(1);
            self.read_first_target(from)?;
        }
    }
}
