//! `java.text.CharacterIterator` as break iteration drives it (Lucene's
//! `CharArrayIterator` and `java.text.StringCharacterIterator` behave the
//! same over a slice: indices `0..len`, `DONE` = U+FFFF past either end),
//! with ICU4J's `impl.CharacterIteration` code point steps and the
//! `UCharacterIterator.getInstance(CharacterIterator)` wrapper's
//! `nextCodePoint` the dictionary matchers read through.

/// `CharacterIterator.DONE`.
pub const DONE: i32 = 0xffff;
/// `CharacterIteration.DONE32`.
pub const DONE32: i32 = 0x7fff_ffff;

/// A character iterator over UTF-16 units.
#[derive(Debug, Clone, Copy)]
pub struct CharIter<'a> {
    text: &'a [u16],
    index: usize,
}

impl<'a> CharIter<'a> {
    /// An iterator at index 0.
    pub fn new(text: &'a [u16]) -> Self {
        CharIter { text, index: 0 }
    }

    /// The text.
    pub fn text(&self) -> &'a [u16] {
        self.text
    }

    /// `getBeginIndex()`: always 0.
    pub fn begin(&self) -> usize {
        0
    }

    /// `getEndIndex()`.
    pub fn end(&self) -> usize {
        self.text.len()
    }

    /// `getIndex()`.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// `current()`.
    #[inline]
    pub fn current(&self) -> i32 {
        self.text.get(self.index).map_or(DONE, |&u| i32::from(u))
    }

    /// `setIndex(position)`; a position past the end is clamped (Java
    /// throws `IllegalArgumentException`, which no caller here provokes).
    #[inline]
    pub fn set_index(&mut self, position: usize) -> i32 {
        self.index = position.min(self.text.len());
        self.current()
    }

    /// `first()`.
    pub fn first(&mut self) -> i32 {
        self.set_index(0)
    }

    /// `next()` (Java's name: a `CharacterIterator`, not an `Iterator`).
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> i32 {
        if self.index.saturating_add(1) >= self.text.len() {
            self.index = self.text.len();
            DONE
        } else {
            self.index = self.index.saturating_add(1);
            self.current()
        }
    }

    /// `previous()`.
    #[inline]
    pub fn previous(&mut self) -> i32 {
        if self.index == 0 {
            DONE
        } else {
            self.index = self.index.saturating_sub(1);
            self.current()
        }
    }

    /// `CharacterIteration.next32(ci)`.
    pub fn next32(&mut self) -> i32 {
        let mut c = self.current();
        if (0xd800..=0xdbff).contains(&c) {
            c = self.next();
            if !(0xdc00..=0xdfff).contains(&c) {
                self.previous();
            }
        }
        c = self.next();
        if c >= 0xd800 {
            c = self.next_trail32(c);
        }
        if c >= 0x10000 && c != DONE32 {
            self.previous();
        }
        c
    }

    /// `CharacterIteration.nextTrail32(ci, lead)`.
    pub fn next_trail32(&mut self, lead: i32) -> i32 {
        if lead == DONE && self.index >= self.text.len() {
            return DONE32;
        }
        let mut ret = lead;
        if lead <= 0xdbff {
            let trail = self.next();
            if (0xdc00..=0xdfff).contains(&trail) {
                ret = crate::icu4j::utf16::to_code_point(lead, trail);
            } else {
                self.previous();
            }
        }
        ret
    }

    /// `CharacterIteration.previous32(ci)`.
    pub fn previous32(&mut self) -> i32 {
        if self.index == 0 {
            return DONE32;
        }
        let trail = self.previous();
        let mut ret = trail;
        if (0xdc00..=0xdfff).contains(&trail) && self.index > 0 {
            let lead = self.previous();
            if (0xd800..=0xdbff).contains(&lead) {
                ret = crate::icu4j::utf16::to_code_point(lead, trail);
            } else {
                self.next();
            }
        }
        ret
    }

    /// `CharacterIteration.current32(ci)`.
    pub fn current32(&mut self) -> i32 {
        let lead = self.current();
        if lead < 0xd800 {
            return lead;
        }
        if (0xd800..=0xdbff).contains(&lead) {
            let trail = self.next();
            self.previous();
            if (0xdc00..=0xdfff).contains(&trail) {
                return crate::icu4j::utf16::to_code_point(lead, trail);
            }
        } else if lead == DONE && self.index >= self.text.len() {
            return DONE32;
        }
        lead
    }

    /// `UCharacterIterator.getInstance(this).next()`: the unit at the index,
    /// then advance; -1 for `DONE` (also for a real U+FFFF, as ICU4J's
    /// `CharacterIteratorWrapper` reads it).
    // SENTINEL: `-1` = `UCharacterIterator.DONE`.
    fn wrapper_next(&mut self) -> i32 {
        let i = self.current();
        self.next();
        if i == DONE {
            -1
        } else {
            i
        }
    }

    /// `UCharacterIterator.nextCodePoint()` through the wrapper (shares this
    /// iterator's index).
    pub fn next_code_point(&mut self) -> i32 {
        let ch1 = self.wrapper_next();
        if (0xd800..=0xdbff).contains(&ch1) {
            let ch2 = self.wrapper_next();
            if (0xdc00..=0xdfff).contains(&ch2) {
                return crate::icu4j::utf16::to_code_point(ch1, ch2);
            } else if ch2 != -1 {
                // previous() through the wrapper
                self.previous();
            }
        }
        ch1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_semantics() {
        let t = crate::icu4j::utf16::units("a\u{10400}b");
        let mut it = CharIter::new(&t);
        assert_eq!(it.text().len(), 4);
        assert_eq!(it.begin(), 0);
        assert_eq!(it.end(), 4);
        assert_eq!(it.first(), 'a' as i32);
        assert_eq!(it.next32(), 0x10400);
        assert_eq!(it.index(), 1);
        assert_eq!(it.current32(), 0x10400);
        assert_eq!(it.next32(), 'b' as i32);
        assert_eq!(it.index(), 3);
        assert_eq!(it.next32(), DONE32);
        assert_eq!(it.index(), 4);
        assert_eq!(it.current32(), DONE32);
        assert_eq!(it.next(), DONE);
        assert_eq!(it.previous32(), 'b' as i32);
        assert_eq!(it.previous32(), 0x10400);
        assert_eq!(it.previous32(), 'a' as i32);
        assert_eq!(it.previous32(), DONE32);
        assert_eq!(it.previous(), DONE);
        it.set_index(0);
        assert_eq!(it.next_code_point(), 'a' as i32);
        assert_eq!(it.next_code_point(), 0x10400);
        assert_eq!(it.next_code_point(), 'b' as i32);
        assert_eq!(it.next_code_point(), -1);
        // Unpaired surrogates.
        let t = [0xd800u16, 0x41, 0xdc00, 0xd800];
        let mut it = CharIter::new(&t);
        assert_eq!(it.current32(), 0xd800);
        assert_eq!(it.next32(), 0x41);
        assert_eq!(it.next32(), 0xdc00);
        it.set_index(3);
        assert_eq!(it.current32(), 0xd800);
        assert_eq!(it.next_trail32(0xd800), 0xd800);
        it.set_index(3);
        assert_eq!(it.previous32(), 0xdc00);
        it.set_index(0);
        assert_eq!(it.next_code_point(), 0xd800);
        assert_eq!(it.index(), 1);
        it.set_index(3);
        assert_eq!(it.next_code_point(), 0xd800);
        assert_eq!(it.set_index(9), DONE);
    }
}
