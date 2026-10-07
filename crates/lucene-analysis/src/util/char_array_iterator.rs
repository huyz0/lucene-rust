//! `org.apache.lucene.analysis.util.CharArrayIterator`: a
//! `java.text.CharacterIterator` over a window of a `char[]`, which
//! `SegmentingTokenizerBase` hands its `BreakIterator`s.
//!
//! The break iterators here ([`super::sentence_break`]) take the window as a
//! slice instead, so the tokenizers do not go through this type; it is the
//! Java class's contract for callers that walk text character by character.
//! `newSentenceInstance`/`newWordInstance` differ only by the JRE-bug
//! workaround Java enables when `HAS_BUGGY_BREAKITERATORS` -- false on every
//! JDK Lucene 10 runs on, and so false here: both are [`CharArrayIterator::new`].

/// `CharacterIterator.DONE`.
pub const DONE: u16 = 0xFFFF;

/// `CharArrayIterator`: indices run `0..=length` relative to `start`
/// (`getBeginIndex()` is 0, `getEndIndex()` the window's length).
#[derive(Clone, Debug, Default)]
pub struct CharArrayIterator {
    array: Vec<u16>,
    start: usize,
    index: usize,
    length: usize,
    limit: usize,
}

impl CharArrayIterator {
    /// `newSentenceInstance()` / `newWordInstance()` (no JRE bug to work
    /// around, see the module docs).
    pub fn new() -> Self {
        Self::default()
    }

    /// `setText(array, start, length)`; `None` (Java's
    /// `ArrayIndexOutOfBoundsException` on first access) when the window
    /// does not fit the array.
    pub fn set_text(&mut self, array: &[u16], start: usize, length: usize) -> Option<()> {
        let limit = start.checked_add(length)?;
        if limit > array.len() {
            return None;
        }
        self.array.clear();
        self.array.extend_from_slice(array);
        self.start = start;
        self.index = start;
        self.length = length;
        self.limit = limit;
        Some(())
    }

    /// `getText()`.
    pub fn text(&self) -> &[u16] {
        &self.array
    }

    /// `getStart()`.
    pub fn start(&self) -> usize {
        self.start
    }

    /// `getLength()`.
    pub fn length(&self) -> usize {
        self.length
    }

    /// `current()`: the unit at the index, [`DONE`] at the end.
    pub fn current(&self) -> u16 {
        if self.index == self.limit {
            DONE
        } else {
            self.array[self.index]
        }
    }

    /// `first()`.
    pub fn first(&mut self) -> u16 {
        self.index = self.start;
        self.current()
    }

    /// `getBeginIndex()`.
    pub fn begin_index(&self) -> usize {
        0
    }

    /// `getEndIndex()`.
    pub fn end_index(&self) -> usize {
        self.length
    }

    /// `getIndex()`.
    pub fn index(&self) -> usize {
        self.index - self.start
    }

    /// `last()`.
    pub fn last(&mut self) -> u16 {
        self.index = if self.limit == self.start {
            self.limit
        } else {
            self.limit - 1
        };
        self.current()
    }

    /// `next()` (Java's `CharacterIterator.next`, not an `Iterator`).
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> u16 {
        self.index += 1;
        if self.index >= self.limit {
            self.index = self.limit;
            DONE
        } else {
            self.current()
        }
    }

    /// `previous()`.
    pub fn previous(&mut self) -> u16 {
        if self.index <= self.start {
            self.index = self.start;
            DONE
        } else {
            self.index -= 1;
            self.current()
        }
    }

    /// `setIndex(position)`: `Err` with Java's `IllegalArgumentException`
    /// message outside `0..=length`.
    pub fn set_index(&mut self, position: i64) -> Result<u16, String> {
        if position < 0 || position > self.length as i64 {
            return Err(format!("Illegal Position: {position}"));
        }
        self.index = self.start + position as usize;
        Ok(self.current())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walks_a_window_as_java_does() {
        let text: Vec<u16> = "xxabcxx".encode_utf16().collect();
        let mut ci = CharArrayIterator::new();
        assert_eq!(ci.current(), DONE);
        ci.set_text(&text, 2, 3).unwrap();
        assert_eq!((ci.start(), ci.length(), ci.text().len()), (2, 3, 7));
        assert_eq!((ci.begin_index(), ci.end_index(), ci.index()), (0, 3, 0));
        assert_eq!(ci.current(), u16::from(b'a'));
        assert_eq!(ci.next(), u16::from(b'b'));
        assert_eq!(ci.next(), u16::from(b'c'));
        assert_eq!(ci.next(), DONE);
        assert_eq!(ci.index(), 3);
        assert_eq!(ci.next(), DONE);
        assert_eq!(ci.previous(), u16::from(b'c'));
        assert_eq!(ci.first(), u16::from(b'a'));
        assert_eq!(ci.previous(), DONE);
        assert_eq!(ci.index(), 0);
        assert_eq!(ci.last(), u16::from(b'c'));
        assert_eq!(ci.set_index(3), Ok(DONE));
        assert_eq!(ci.set_index(1), Ok(u16::from(b'b')));
        assert_eq!(ci.set_index(4), Err("Illegal Position: 4".to_string()));
        assert_eq!(ci.set_index(-1), Err("Illegal Position: -1".to_string()));
        // An empty window.
        ci.set_text(&text, 7, 0).unwrap();
        assert_eq!((ci.last(), ci.first(), ci.index()), (DONE, DONE, 0));
        assert!(ci.set_text(&text, 5, 3).is_none());
        assert!(ci.set_text(&text, usize::MAX, 3).is_none());
        let copy = ci.clone();
        assert_eq!(copy.length(), 0);
    }
}
