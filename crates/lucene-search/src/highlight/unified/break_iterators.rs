//! `java.text.BreakIterator` as the unified highlighter uses it, and Lucene's
//! own break iterators over it: `WholeBreakIterator`, `LengthGoalBreakIterator`,
//! `CustomSeparatorBreakIterator` and `SplittingBreakIterator`.
//!
//! Offsets are `i32` UTF-16 code-unit indexes and "no boundary" is
//! [`DONE`] (`-1`), as in Java: the length-goal arithmetic compares and
//! subtracts offsets with `DONE` among them.

use super::super::word_break::{Kind, RuleBreakIterator};

/// `BreakIterator.DONE`.
pub const DONE: i32 = -1;

/// The part of `java.text.BreakIterator` the highlighter calls. An iterator
/// is stateful: `set_text` rewinds it.
pub trait BreakIterator {
    /// `setText(String)`.
    fn set_text(&mut self, text: &[u16]);
    /// `getText().getEndIndex()`.
    fn text_end(&self) -> i32;
    /// `current()`.
    fn current(&self) -> i32;
    /// `first()`.
    fn first(&mut self) -> i32;
    /// `last()`.
    fn last(&mut self) -> i32;
    /// `next()`: [`DONE`] past the last boundary.
    fn next(&mut self) -> i32;
    /// `previous()`: [`DONE`] before the first boundary.
    fn previous(&mut self) -> i32;
    /// `following(offset)`: [`DONE`] when `offset` is the end.
    fn following(&mut self, offset: i32) -> i32;
    /// `preceding(offset)`: [`DONE`] when `offset` is the start.
    fn preceding(&mut self, offset: i32) -> i32;
}

fn to_i32(v: usize) -> i32 {
    i32::try_from(v).unwrap_or(i32::MAX)
}

fn from_opt(v: Option<usize>) -> i32 {
    v.map_or(DONE, to_i32)
}

fn offset(v: i32) -> usize {
    assert!(v >= 0, "offset out of bounds");
    usize::try_from(v).unwrap_or(usize::MAX)
}

impl BreakIterator for RuleBreakIterator {
    fn set_text(&mut self, text: &[u16]) {
        RuleBreakIterator::set_text(self, text.to_vec());
    }
    fn text_end(&self) -> i32 {
        to_i32(self.text_len())
    }
    fn current(&self) -> i32 {
        to_i32(RuleBreakIterator::current(self))
    }
    fn first(&mut self) -> i32 {
        to_i32(RuleBreakIterator::first(self))
    }
    fn last(&mut self) -> i32 {
        to_i32(RuleBreakIterator::last(self))
    }
    fn next(&mut self) -> i32 {
        from_opt(RuleBreakIterator::next(self))
    }
    fn previous(&mut self) -> i32 {
        from_opt(RuleBreakIterator::previous(self))
    }
    fn following(&mut self, off: i32) -> i32 {
        from_opt(RuleBreakIterator::following(self, offset(off)))
    }
    fn preceding(&mut self, off: i32) -> i32 {
        from_opt(RuleBreakIterator::preceding(self, offset(off)))
    }
}

/// `BreakIterator.getSentenceInstance(Locale.ROOT)`, the highlighter's
/// default.
pub fn sentence_instance() -> Box<dyn BreakIterator> {
    Box::new(RuleBreakIterator::from_utf16(Kind::Sentence, Vec::new()))
}

/// `BreakIterator.getWordInstance(Locale.ROOT)`.
pub fn word_instance() -> Box<dyn BreakIterator> {
    Box::new(RuleBreakIterator::from_utf16(Kind::Word, Vec::new()))
}

// ---------------------------------------------------------------------------
// WholeBreakIterator
// ---------------------------------------------------------------------------

/// `WholeBreakIterator`: the whole text is one passage.
#[derive(Debug, Default, Clone)]
pub struct WholeBreakIterator {
    end: i32,
    current: i32,
}

impl WholeBreakIterator {
    /// `new WholeBreakIterator()`.
    pub fn new() -> Self {
        Self::default()
    }

    fn check(&self, pos: i32) {
        assert!(pos >= 0 && pos <= self.end, "offset out of bounds");
    }
}

impl BreakIterator for WholeBreakIterator {
    fn set_text(&mut self, text: &[u16]) {
        self.end = to_i32(text.len());
        self.current = 0;
    }
    fn text_end(&self) -> i32 {
        self.end
    }
    fn current(&self) -> i32 {
        self.current
    }
    fn first(&mut self) -> i32 {
        self.current = 0;
        self.current
    }
    fn last(&mut self) -> i32 {
        self.current = self.end;
        self.current
    }
    fn next(&mut self) -> i32 {
        if self.current == self.end {
            DONE
        } else {
            self.last()
        }
    }
    fn previous(&mut self) -> i32 {
        if self.current == 0 {
            DONE
        } else {
            self.first()
        }
    }
    fn following(&mut self, pos: i32) -> i32 {
        self.check(pos);
        if pos == self.end {
            self.current = self.end;
            DONE
        } else {
            self.last()
        }
    }
    fn preceding(&mut self, pos: i32) -> i32 {
        self.check(pos);
        if pos == 0 {
            self.current = 0;
            DONE
        } else {
            self.first()
        }
    }
}

// ---------------------------------------------------------------------------
// LengthGoalBreakIterator
// ---------------------------------------------------------------------------

/// `LengthGoalBreakIterator`: a passage near a length, built of whole
/// passages of the iterator it wraps.
pub struct LengthGoalBreakIterator {
    base: Box<dyn BreakIterator>,
    length_goal: i32,
    fragment_alignment: f32,
    is_minimum_length: bool,
    current_cache: i32,
}

impl LengthGoalBreakIterator {
    /// `createMinLength(baseIter, minLength, fragmentAlignment)`.
    ///
    /// # Panics
    /// When `fragment_alignment` is outside `[0, 1]`
    /// (`IllegalArgumentException`).
    pub fn create_min_length(
        base: Box<dyn BreakIterator>,
        min_length: i32,
        fragment_alignment: f32,
    ) -> Self {
        Self::new(base, min_length, fragment_alignment, true)
    }

    /// `createClosestToLength(baseIter, targetLength, fragmentAlignment)`.
    ///
    /// # Panics
    /// As [`Self::create_min_length`].
    pub fn create_closest_to_length(
        base: Box<dyn BreakIterator>,
        target_length: i32,
        fragment_alignment: f32,
    ) -> Self {
        Self::new(base, target_length, fragment_alignment, false)
    }

    fn new(
        base: Box<dyn BreakIterator>,
        length_goal: i32,
        fragment_alignment: f32,
        is_minimum_length: bool,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&fragment_alignment) && fragment_alignment.is_finite(),
            "fragmentAlignment must be >= zero and <= one"
        );
        let current_cache = base.current();
        LengthGoalBreakIterator {
            base,
            length_goal,
            fragment_alignment,
            is_minimum_length,
            current_cache,
        }
    }

    /// The private `following(matchEndIndex, targetIdx)`.
    fn following_target(&mut self, match_end_index: i32, target_idx: i32) -> i32 {
        if target_idx >= self.base.text_end() {
            if self.current_cache == self.base.last() {
                return DONE;
            }
            self.current_cache = self.base.last();
            return self.current_cache;
        }
        let after_idx = self.base.following(target_idx - 1);
        if after_idx == DONE {
            self.current_cache = self.base.last();
            return DONE;
        }
        if after_idx == target_idx {
            // right on the money
            self.current_cache = after_idx;
            return self.current_cache;
        }
        if self.is_minimum_length {
            // thus never undershoot
            self.current_cache = after_idx;
            return self.current_cache;
        }
        let before_idx = self.base.preceding(target_idx);
        if target_idx - before_idx < after_idx - target_idx && before_idx > match_end_index {
            self.current_cache = before_idx;
            return self.current_cache;
        }
        self.current_cache = after_idx;
        self.current_cache
    }

    /// `(int) (lengthGoal * factor)`, Java's float-to-int cast.
    fn scaled(&self, factor: f32) -> i32 {
        // `as` saturates, as Java's narrowing does
        (self.length_goal as f32 * factor) as i32
    }
}

impl BreakIterator for LengthGoalBreakIterator {
    fn set_text(&mut self, text: &[u16]) {
        self.base.set_text(text);
        self.current_cache = self.base.current();
    }
    fn text_end(&self) -> i32 {
        self.base.text_end()
    }
    fn current(&self) -> i32 {
        self.current_cache
    }
    fn first(&mut self) -> i32 {
        self.current_cache = self.base.first();
        self.current_cache
    }
    fn last(&mut self) -> i32 {
        self.current_cache = self.base.last();
        self.current_cache
    }
    fn next(&mut self) -> i32 {
        let cache = self.current_cache;
        self.following_target(cache, cache + self.length_goal)
    }
    fn previous(&mut self) -> i32 {
        // `assert false : "Not supported"`
        self.base.previous()
    }
    fn following(&mut self, match_end_index: i32) -> i32 {
        let target = (match_end_index + 1) + self.scaled(1.0 - self.fragment_alignment);
        self.following_target(match_end_index, target)
    }
    fn preceding(&mut self, match_start_index: i32) -> i32 {
        let target_idx = (match_start_index - 1) - self.scaled(self.fragment_alignment);
        if target_idx <= 0 {
            if self.current_cache == self.base.first() {
                return DONE;
            }
            self.current_cache = self.base.first();
            return self.current_cache;
        }
        let before_idx = self.base.preceding(target_idx + 1);
        if before_idx == DONE {
            self.current_cache = self.base.first();
            return DONE;
        }
        if before_idx == target_idx {
            // right on the money
            self.current_cache = before_idx;
            return self.current_cache;
        }
        if self.is_minimum_length {
            // thus never undershoot
            self.current_cache = before_idx;
            return self.current_cache;
        }
        let after_idx = self.base.following(target_idx - 1);
        if after_idx - target_idx < target_idx - before_idx && after_idx < match_start_index {
            self.current_cache = after_idx;
            return self.current_cache;
        }
        self.current_cache = before_idx;
        self.current_cache
    }
}

// ---------------------------------------------------------------------------
// CustomSeparatorBreakIterator
// ---------------------------------------------------------------------------

/// `CustomSeparatorBreakIterator`: a boundary after each occurrence of one
/// character.
#[derive(Debug, Clone)]
pub struct CustomSeparatorBreakIterator {
    separator: u16,
    text: Vec<u16>,
    /// The `CharacterIterator`'s index.
    index: i32,
    current: i32,
}

impl CustomSeparatorBreakIterator {
    /// `new CustomSeparatorBreakIterator(separator)`.
    pub fn new(separator: u16) -> Self {
        CustomSeparatorBreakIterator {
            separator,
            text: Vec::new(),
            index: 0,
            current: 0,
        }
    }

    fn end(&self) -> i32 {
        to_i32(self.text.len())
    }

    /// `CharacterIterator.next()`: `None` for `DONE`.
    fn step_forward(&mut self) -> Option<u16> {
        if self.index + 1 < self.end() {
            self.index += 1;
            Some(self.text[offset(self.index)])
        } else {
            self.index = self.end();
            None
        }
    }

    /// `CharacterIterator.previous()`: `None` for `DONE`.
    fn step_back(&mut self) -> Option<u16> {
        if self.index > 0 {
            self.index -= 1;
            Some(self.text[offset(self.index)])
        } else {
            None
        }
    }

    fn advance_forward(&mut self) -> i32 {
        while let Some(c) = self.step_forward() {
            if c == self.separator {
                self.current = self.index + 1;
                return self.current;
            }
        }
        self.current = self.index;
        self.current
    }

    fn advance_backward(&mut self) -> i32 {
        while let Some(c) = self.step_back() {
            if c == self.separator {
                self.current = self.index + 1;
                return self.current;
            }
        }
        self.current = self.index;
        self.current
    }

    fn check(&self, pos: i32) {
        assert!(pos >= 0 && pos <= self.end(), "offset out of bounds");
    }
}

impl BreakIterator for CustomSeparatorBreakIterator {
    fn set_text(&mut self, text: &[u16]) {
        self.text = text.to_vec();
        self.index = 0;
        self.current = 0;
    }
    fn text_end(&self) -> i32 {
        self.end()
    }
    fn current(&self) -> i32 {
        self.current
    }
    fn first(&mut self) -> i32 {
        self.index = 0;
        self.current = self.index;
        self.current
    }
    fn last(&mut self) -> i32 {
        self.index = self.end();
        self.current = self.index;
        self.current
    }
    fn next(&mut self) -> i32 {
        if self.index == self.end() {
            DONE
        } else {
            self.advance_forward()
        }
    }
    fn previous(&mut self) -> i32 {
        if self.index == 0 {
            DONE
        } else {
            self.advance_backward()
        }
    }
    fn following(&mut self, pos: i32) -> i32 {
        self.check(pos);
        if pos == self.end() {
            self.index = self.end();
            self.current = self.index;
            DONE
        } else {
            self.index = pos;
            self.current = self.index;
            self.advance_forward()
        }
    }
    fn preceding(&mut self, pos: i32) -> i32 {
        self.check(pos);
        if pos == 0 {
            self.index = 0;
            self.current = self.index;
            DONE
        } else {
            self.index = pos;
            self.current = self.index;
            self.advance_backward()
        }
    }
}

// ---------------------------------------------------------------------------
// SplittingBreakIterator
// ---------------------------------------------------------------------------

/// `SplittingBreakIterator`: the base iterator run over each slice of the
/// text between occurrences of one character, so a passage never crosses it.
pub struct SplittingBreakIterator {
    base: Box<dyn BreakIterator>,
    slice_char: u16,
    text: Vec<u16>,
    slice_start: i32,
    slice_end: i32,
    current: i32,
}

impl SplittingBreakIterator {
    /// `new SplittingBreakIterator(baseIter, sliceChar)`.
    pub fn new(base: Box<dyn BreakIterator>, slice_char: u16) -> Self {
        SplittingBreakIterator {
            base,
            slice_char,
            text: Vec::new(),
            slice_start: 0,
            slice_end: 0,
            current: 0,
        }
    }

    fn len(&self) -> i32 {
        to_i32(self.text.len())
    }

    /// `text.indexOf(sliceChar, from)`, `-1` when absent.
    // SENTINEL: `-1` = not found, `String.indexOf`'s contract; compared only.
    fn index_of(&self, from: i32) -> i32 {
        let from = offset(from.max(0));
        self.text
            .iter()
            .skip(from)
            .position(|&c| c == self.slice_char)
            .map_or(DONE, |p| to_i32(p + from))
    }

    /// `text.lastIndexOf(sliceChar, from)`, `-1` when absent.
    // SENTINEL: `-1` = not found, `String.lastIndexOf`'s contract; compared
    // only.
    fn last_index_of(&self, from: i32) -> i32 {
        if from < 0 {
            return DONE;
        }
        let end = offset(from).saturating_add(1).min(self.text.len());
        self.text[..end]
            .iter()
            .rposition(|&c| c == self.slice_char)
            .map_or(DONE, to_i32)
    }

    fn set_slice(&mut self) {
        let (s, e) = (offset(self.slice_start), offset(self.slice_end));
        let slice = self.text[s..e].to_vec();
        self.base.set_text(&slice);
    }
}

impl BreakIterator for SplittingBreakIterator {
    fn set_text(&mut self, text: &[u16]) {
        self.text = text.to_vec();
        self.first();
    }
    fn text_end(&self) -> i32 {
        self.len()
    }
    fn current(&self) -> i32 {
        self.current
    }
    fn first(&mut self) -> i32 {
        self.slice_start = 0;
        self.slice_end = self.index_of(0);
        if self.slice_end == DONE {
            self.slice_end = self.len();
        }
        if self.slice_start == self.slice_end {
            self.current = self.slice_start;
            return self.current;
        }
        self.set_slice();
        self.current = self.slice_start + self.base.current();
        self.current
    }
    fn last(&mut self) -> i32 {
        self.slice_end = self.len();
        self.slice_start = self.last_index_of(self.len() - 1);
        if self.slice_start == DONE {
            self.slice_start = 0;
        } else {
            self.slice_start += 1;
        }
        if self.slice_end == self.slice_start {
            self.current = self.slice_end;
            return self.current;
        }
        self.set_slice();
        self.current = self.slice_start + self.base.last();
        self.current
    }
    fn next(&mut self) -> i32 {
        let prev_current = self.current;
        self.current = if self.slice_start == self.slice_end {
            DONE
        } else {
            self.base.next()
        };
        if self.current != DONE {
            self.current += self.slice_start;
            return self.current;
        }
        if self.slice_end >= self.len() {
            self.current = prev_current;
            return DONE;
        }
        self.slice_start = self.slice_end + 1;
        self.slice_end = self.index_of(self.slice_start);
        if self.slice_end == DONE {
            self.slice_end = self.len();
        }
        if self.slice_start == self.slice_end {
            self.current = self.slice_start;
            return self.current;
        }
        self.set_slice();
        self.current = self.slice_start + self.base.current();
        self.current
    }
    fn previous(&mut self) -> i32 {
        let prev_current = self.current;
        self.current = if self.slice_start == self.slice_end {
            DONE
        } else {
            self.base.previous()
        };
        if self.current != DONE {
            self.current += self.slice_start;
            return self.current;
        }
        if self.slice_start == 0 {
            self.current = prev_current;
            return DONE;
        }
        self.slice_end = self.slice_start - 1;
        self.slice_start = self.last_index_of(self.slice_end - 1);
        if self.slice_start == DONE {
            self.slice_start = 0;
        } else {
            self.slice_start += 1;
        }
        if self.slice_start == self.slice_end {
            self.current = self.slice_start;
            return self.current;
        }
        self.set_slice();
        self.current = self.slice_start + self.base.last();
        self.current
    }
    fn following(&mut self, off: i32) -> i32 {
        if off + 1 < self.slice_start || off + 1 > self.slice_end {
            if off == self.len() {
                self.last();
                return DONE;
            }
            self.slice_start = self.last_index_of(off);
            if self.slice_start == DONE {
                self.slice_start = 0;
            } else {
                self.slice_start += 1;
            }
            self.slice_end = self.index_of((off + 1).max(self.slice_start));
            if self.slice_end == DONE {
                self.slice_end = self.len();
            }
            if self.slice_start != self.slice_end {
                self.set_slice();
            }
        }
        if self.slice_start == self.slice_end {
            self.current = off + 1;
        } else if off == self.slice_start - 1 {
            self.current = self.slice_start + self.base.first();
        } else {
            self.current = self.slice_start + self.base.following(off - self.slice_start);
        }
        self.current
    }
    fn preceding(&mut self, off: i32) -> i32 {
        if off - 1 < self.slice_start || off - 1 > self.slice_end {
            if off == 0 {
                self.first();
                return DONE;
            }
            self.slice_end = self.index_of(off);
            if self.slice_end == DONE {
                self.slice_end = self.len();
            }
            self.slice_start = self.last_index_of(off - 1);
            if self.slice_start == DONE {
                self.slice_start = 0;
            } else {
                self.slice_start = (self.slice_start + 1).min(self.slice_end);
            }
            if self.slice_start != self.slice_end {
                self.set_slice();
            }
        }
        if self.slice_start == self.slice_end {
            self.current = off - 1;
        } else if off == self.slice_end + 1 {
            self.current = self.slice_start + self.base.last();
        } else {
            self.current = self.slice_start + self.base.preceding(off - self.slice_start);
        }
        self.current
    }
}
