//! Port of `org.apache.lucene.util.fst.FSTEnum` and `BytesRefFSTEnum` over
//! a [`TypedFst`]: `next`, `seekCeil` and `seekExact` through any
//! [`FstOutputs`], every node encoding (list, binary search, direct
//! addressing, continuous) walked as Java walks it.
//!
//! `crate::fst::FstEnum` does the same over byte-sequence outputs only;
//! this one is what a term dictionary whose FST carries its own outputs
//! (`FSTTermsReader`'s `TermData`) iterates with. `seekFloor` is what
//! `uniformsplit`'s `FSTDictionary.Browser.seekBlock` finds a block with.

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

    /// `seekFloor(target)`: the largest term `<= target`, or `None` when
    /// every term is greater.
    pub fn seek_floor(&mut self, target: &[u8]) -> Result<Option<(&[u8], &O::Value)>> {
        self.set_target(target);
        self.do_seek_floor()?;
        Ok(self.result())
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

    /// `doSeekFloor()`.
    fn do_seek_floor(&mut self) -> Result<()> {
        self.rewind_prefix()?;
        loop {
            let target_label = self.target_label();
            let arc = &self.arcs[self.upto];
            let go_on = if arc.bytes_per_arc() != 0 && arc.label() != END_LABEL {
                match arc.node_flags() {
                    ARCS_FOR_DIRECT_ADDRESSING => self.seek_floor_direct(target_label)?,
                    ARCS_FOR_BINARY_SEARCH => self.seek_floor_packed(target_label)?,
                    _ => self.seek_floor_continuous(target_label)?,
                }
            } else {
                self.seek_floor_list(target_label)?
            };
            if !go_on {
                return Ok(());
            }
        }
    }

    /// `doSeekFloorContinuous`.
    fn seek_floor_continuous(&mut self, target_label: i32) -> Result<bool> {
        let arc = &self.arcs[self.upto];
        // Two `i32`s, so their difference fits an `i64`.
        let target_index = i64::from(target_label).wrapping_sub(i64::from(arc.first_label));
        if target_index < 0 {
            // Before the first arc.
            self.backtrack_to_floor_arc(target_label)?;
            return Ok(false);
        }
        let mut input = self.fst.bytes_reader();
        if target_index >= i64::from(arc.num_arcs()) {
            // After the last arc.
            self.fst
                .read_last_arc_by_continuous(&mut self.arcs[self.upto], &mut input)?;
            self.push_last()?;
            return Ok(false);
        }
        self.fst.read_arc_by_continuous(
            &mut self.arcs[self.upto],
            &mut input,
            target_index as i32,
        )?;
        self.take_matched(target_label)
    }

    /// `doSeekFloorArrayDirectAddressing`.
    fn seek_floor_direct(&mut self, target_label: i32) -> Result<bool> {
        let arc = &self.arcs[self.upto];
        // Two `i32`s, so their difference fits an `i64`.
        let target_index = i64::from(target_label).wrapping_sub(i64::from(arc.first_label));
        if target_index < 0 {
            self.backtrack_to_floor_arc(target_label)?;
            return Ok(false);
        }
        let mut input = self.fst.bytes_reader();
        if target_index >= i64::from(arc.num_arcs()) {
            self.fst
                .read_last_arc_by_direct_addressing(&mut self.arcs[self.upto], &mut input)?;
            self.push_last()?;
            return Ok(false);
        }
        let idx = target_index as i32;
        if self.fst.bit_is_set(idx, arc, &mut input)? {
            self.fst
                .read_arc_by_direct_addressing(&mut self.arcs[self.upto], &mut input, idx)?;
            return self.take_matched(target_label);
        }
        // Scan backwards to the floor arc.
        let floor = self
            .fst
            .bit_previous_set(idx, &self.arcs[self.upto], &mut input)?;
        if floor < 0 {
            // Java asserts it: the first arc's bit is always set.
            return Err(FstReadError(
                "direct-addressing node without a floor arc".into(),
            ));
        }
        self.fst
            .read_arc_by_direct_addressing(&mut self.arcs[self.upto], &mut input, floor)?;
        self.push_last()?;
        Ok(false)
    }

    /// `doSeekFloorArrayPacked`.
    // ARITH: `-2 - idx` of a binary search's negative insertion point
    // (`idx <= -2` in that branch) is a node index `>= 0`.
    #[allow(clippy::arithmetic_side_effects)]
    fn seek_floor_packed(&mut self, target_label: i32) -> Result<bool> {
        let idx = super::binary_search(self.fst, &self.arcs[self.upto], target_label)?;
        let mut input = self.fst.bytes_reader();
        if idx >= 0 {
            self.fst
                .read_arc_by_index(&mut self.arcs[self.upto], &mut input, idx)?;
            return self.take_matched(target_label);
        }
        if idx == -1 {
            // Before the first arc.
            self.backtrack_to_floor_arc(target_label)?;
            return Ok(false);
        }
        // A floor arc: idx is -1 - (floor + 1).
        self.fst
            .read_arc_by_index(&mut self.arcs[self.upto], &mut input, -2 - idx)?;
        self.push_last()?;
        Ok(false)
    }

    /// `doSeekFloorList`.
    fn seek_floor_list(&mut self, target_label: i32) -> Result<bool> {
        let arc = &self.arcs[self.upto];
        if arc.label() == target_label {
            return self.take_matched(target_label);
        }
        if arc.label() > target_label {
            self.backtrack_scanning(target_label)?;
            return Ok(false);
        }
        if !arc.is_last() {
            let mut input = self.fst.bytes_reader();
            if self.fst.read_next_arc_label(arc, &mut input)? > target_label {
                self.push_last()?;
                return Ok(false);
            }
            // Keep scanning.
            self.read_next(self.upto)?;
            return Ok(true);
        }
        self.push_last()?;
        Ok(false)
    }

    /// `doSeekFloorList`'s backtrack: up the path to the first node whose
    /// first arc is before the target label there, then forward along it
    /// to the arc just before that label.
    // ARITH: `upto >= 1` on entry and the loop stops when it reaches 0.
    #[allow(clippy::arithmetic_side_effects)]
    fn backtrack_scanning(&mut self, mut target_label: i32) -> Result<()> {
        loop {
            self.read_first_target(self.upto - 1)?;
            if self.arcs[self.upto].label() < target_label {
                self.scan_to_floor(target_label)?;
                return self.push_last();
            }
            self.upto -= 1;
            if self.upto == 0 {
                return Ok(());
            }
            target_label = self.target_label();
        }
    }

    /// Forward from the arc at `upto` while the next arc's label is below
    /// `target_label`.
    fn scan_to_floor(&mut self, target_label: i32) -> Result<()> {
        let mut input = self.fst.bytes_reader();
        while !self.arcs[self.upto].is_last()
            && self
                .fst
                .read_next_arc_label(&self.arcs[self.upto], &mut input)?
                < target_label
        {
            self.read_next(self.upto)?;
        }
        Ok(())
    }

    /// `backtrackToFloorArc(arc, targetLabel, in)`.
    // ARITH: `upto >= 1` on entry and the loop stops when it reaches 0.
    #[allow(clippy::arithmetic_side_effects)]
    fn backtrack_to_floor_arc(&mut self, mut target_label: i32) -> Result<()> {
        loop {
            // First, walk backwards to a node whose first arc is before the
            // target label.
            self.read_first_target(self.upto - 1)?;
            let arc = &self.arcs[self.upto];
            if arc.label() < target_label {
                // Then, on this node, find the arc just before the label.
                if !arc.is_last() {
                    if arc.bytes_per_arc() != 0 && arc.label() != END_LABEL {
                        match arc.node_flags() {
                            ARCS_FOR_BINARY_SEARCH => {
                                self.find_next_floor_arc_binary_search(target_label)?
                            }
                            ARCS_FOR_DIRECT_ADDRESSING => {
                                self.find_next_floor_arc_direct(target_label)?
                            }
                            _ => self.find_next_floor_arc_continuous(target_label)?,
                        }
                    } else {
                        self.scan_to_floor(target_label)?;
                    }
                }
                return self.push_last();
            }
            self.upto -= 1;
            if self.upto == 0 {
                return Ok(());
            }
            target_label = self.target_label();
        }
    }

    /// `findNextFloorArcDirectAddressing`: the arc at `upto` is its node's
    /// first; move to the last one before `target_label`, if not it.
    fn find_next_floor_arc_direct(&mut self, target_label: i32) -> Result<()> {
        let arc = &self.arcs[self.upto];
        if arc.num_arcs() <= 1 {
            return Ok(());
        }
        let mut input = self.fst.bytes_reader();
        // Two `i32`s, so their difference fits an `i64`; >= 0 as the first
        // label is below the target's.
        let target_index = i64::from(target_label).wrapping_sub(i64::from(arc.first_label));
        if target_index >= i64::from(arc.num_arcs()) {
            // Beyond the last arc: take the last arc.
            return self
                .fst
                .read_last_arc_by_direct_addressing(&mut self.arcs[self.upto], &mut input);
        }
        // Take the preceding arc, even if the target is present.
        let floor = self
            .fst
            .bit_previous_set(target_index as i32, arc, &mut input)?;
        if floor > 0 {
            self.fst
                .read_arc_by_direct_addressing(&mut self.arcs[self.upto], &mut input, floor)?;
        }
        Ok(())
    }

    /// `findNextFloorArcContinuous`.
    // ARITH: `target_index >= 1` in the branch that subtracts (the first
    // label is below the target's and `target_index < numArcs`).
    #[allow(clippy::arithmetic_side_effects)]
    fn find_next_floor_arc_continuous(&mut self, target_label: i32) -> Result<()> {
        let arc = &self.arcs[self.upto];
        if arc.num_arcs() <= 1 {
            return Ok(());
        }
        let mut input = self.fst.bytes_reader();
        let target_index = i64::from(target_label).wrapping_sub(i64::from(arc.first_label));
        if target_index >= i64::from(arc.num_arcs()) {
            return self
                .fst
                .read_last_arc_by_continuous(&mut self.arcs[self.upto], &mut input);
        }
        self.fst.read_arc_by_continuous(
            &mut self.arcs[self.upto],
            &mut input,
            target_index as i32 - 1,
        )
    }

    /// `findNextFloorArcBinarySearch`.
    // ARITH: `idx - 1` for `idx > 1` and `-2 - idx` for `idx < -2`: node
    // indices >= 1.
    #[allow(clippy::arithmetic_side_effects)]
    fn find_next_floor_arc_binary_search(&mut self, target_label: i32) -> Result<()> {
        if self.arcs[self.upto].num_arcs() <= 1 {
            return Ok(());
        }
        let idx = super::binary_search(self.fst, &self.arcs[self.upto], target_label)?;
        let mut input = self.fst.bytes_reader();
        if idx > 1 {
            self.fst
                .read_arc_by_index(&mut self.arcs[self.upto], &mut input, idx - 1)?;
        } else if idx < -2 {
            self.fst
                .read_arc_by_index(&mut self.arcs[self.upto], &mut input, -2 - idx)?;
        }
        Ok(())
    }

    /// `pushLast()`: from the arc at `upto`, the last path down to a
    /// term's end.
    // ARITH: `upto >= 1` after `incr`.
    #[allow(clippy::arithmetic_side_effects)]
    fn push_last(&mut self) -> Result<()> {
        loop {
            let label = self.arcs[self.upto].label();
            self.set_current_label(label);
            self.accumulate();
            if label == END_LABEL {
                return Ok(());
            }
            self.incr();
            let mut input = self.fst.bytes_reader();
            let last = self
                .fst
                .read_last_target_arc(&self.arcs[self.upto - 1], &mut input)?;
            self.arcs[self.upto] = last;
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

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;
    use crate::fst::{InputType, PositiveIntOutputs};
    use crate::fst_compiler::FstCompilerBuilder;

    /// Keys with dense, sparse and contiguous label runs, so list,
    /// binary-search, direct-addressing and continuous nodes all appear.
    fn keys(seed: u64, n: usize) -> Vec<Vec<u8>> {
        let mut x = seed;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut keys = std::collections::BTreeSet::new();
        for i in 0..n {
            let len = 1 + (next() % 5) as usize;
            let k: Vec<u8> = (0..len)
                .map(|j| match (i + j) % 3 {
                    0 => b'a' + (next() % 26) as u8,
                    1 => (next() % 256) as u8,
                    _ => b'm' + (next() % 4) as u8,
                })
                .collect();
            keys.insert(k);
        }
        // A contiguous run of first labels.
        for b in b'0'..=b'9' {
            keys.insert(vec![b]);
            keys.insert(vec![b, b'x']);
        }
        keys.insert(Vec::new());
        keys.into_iter().collect()
    }

    fn build(keys: &[Vec<u8>], factor: f32) -> TypedFst<PositiveIntOutputs> {
        let mut c = FstCompilerBuilder::new(InputType::Byte1)
            .direct_addressing_max_oversizing_factor(factor)
            .build::<PositiveIntOutputs>();
        for (i, k) in keys.iter().enumerate() {
            c.add_bytes(k, i as i64 * 3).unwrap();
        }
        TypedFst::from_compiled(c.compile().unwrap())
    }

    #[test]
    fn seek_floor_matches_a_sorted_scan() {
        for (seed, factor) in [(7u64, 0.0f32), (11, 1.0), (13, 64.0), (17, -1.0)] {
            for skip_empty in [false, true] {
                let mut keys = keys(seed, 600);
                if skip_empty {
                    keys.remove(0);
                }
                let fst = build(&keys, factor);
                let mut e = BytesRefFstEnum::new(&fst);
                let mut probes: Vec<Vec<u8>> = keys.clone();
                for k in &keys {
                    let mut longer = k.clone();
                    longer.push(0);
                    probes.push(longer);
                    let mut bumped = k.clone();
                    if let Some(last) = bumped.last_mut() {
                        if *last > 0 {
                            *last -= 1;
                            probes.push(bumped.clone());
                        }
                        if let Some(l) = bumped.last_mut() {
                            *l = l.saturating_add(2);
                        }
                        probes.push(bumped);
                    }
                }
                probes.push(vec![0xff; 6]);
                probes.push(Vec::new());
                // In a scrambled order, so the enum rewinds as well as advances.
                let n = probes.len();
                for step in 0..n {
                    let probe = &probes[(step * 7919) % n];
                    let want = keys
                        .iter()
                        .enumerate()
                        .rev()
                        .find(|(_, k)| k.as_slice() <= probe.as_slice())
                        .map(|(i, k)| (k.clone(), i as i64 * 3));
                    let got = e.seek_floor(probe).unwrap().map(|(k, v)| (k.to_vec(), *v));
                    assert_eq!(got, want, "factor {factor} probe {probe:?}");
                }
            }
        }
    }
}
