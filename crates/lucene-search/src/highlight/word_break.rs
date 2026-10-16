//! `BreakIterator.getWordInstance(Locale.ROOT)` of the JDK -- the legacy
//! rule-based word iterator, which is not UAX #29's: `e-mail`, `foo_bar`,
//! `a.b` and `2.5%` are each one word, a run of white space is one segment,
//! and a number takes a trailing `%`. Lucene's `BreakIteratorShrinkingAdjuster`
//! (`matchhighlight`) places passage edges with it.
//!
//! A character is put in one of twenty-two classes ([`class_of`]); two small
//! deterministic automata read classes, [`NEXT`] forwards from a segment's
//! first character (the last accepting state it passes names the segment's
//! end) and [`PREVIOUS`] backwards from an offset (to find a boundary to start
//! the forward scan from). The iterator keeps the JDK's own state -- the
//! cursor and the last break found -- so that `following`, `preceding` and
//! `isBoundary` give its answers even where they are quirks (an offset inside
//! a surrogate pair). The classes and the automata were read off the JDK's
//! behaviour; the tests compare `first`/`next`, `isBoundary`, `following` and
//! `preceding` with the JDK's over thousands of random strings
//! (`wordfuzz.tsv`), asked of fresh and of reused iterators.
//!
//! Offsets are UTF-16 code units, as `String` indexes. `Character.getType`
//! is Unicode 16 here, the JDK 21 that generated the fixtures Unicode 15: a
//! character added since is a symbol to the JDK and may be a letter here.

use lucene_analysis::java_character as jc;

/// The number of character classes.
const CLASSES: usize = 22;

/// Class of a character no rule names: symbols, punctuation, unassigned.
const OTHER: usize = 20;

/// `BreakIterator.DONE`.
const DONE: i64 = -1;

/// What the end of the text reads as (`CharacterIterator.DONE`).
const END_CHAR: u32 = 0xFFFF;

/// Forward automaton: the next state for a class, in each state. Row 0 is
/// the stop state; row 1 is the start. Zero is "stop".
///
/// Classes: 0 control, 1 kana combining mark, 2 `.`, 3 ideograph, 4 letter
/// (or spacing mark), 5 danda, 6 post-number sign (`%`, `&`, cent, per
/// mille), 7 CR, 8 `"` and `'`, 9 pre-number sign (currency, `#`), 10 kana
/// punctuation, 11 other mark, 12 digit, 13 space, 14 hiragana, 15 katakana,
/// 16 line separator (LF, FF, U+2028, U+2029), 17 tab, 18 soft hyphen,
/// 19 `,` (and U+066B), 20 anything else, 21 dash or connector (`-`, `_`).
///
/// States: 3 is a word and 6 a number (`4` and `7` wait for a letter or a
/// digit, `5` follows a danda), 8 a run of white space, 2, 9 and 16 close a
/// segment at once or after combining marks, 12 .. 19 are runs of kana and
/// ideographs.
const NEXT: [[u8; CLASSES]; 20] = [
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        2, 12, 14, 15, 3, 16, 16, 9, 16, 14, 17, 2, 6, 8, 18, 19, 2, 8, 2, 16, 16, 16,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 3, 4, 0, 3, 5, 0, 0, 4, 0, 0, 3, 6, 0, 0, 0, 0, 0, 4, 0, 0, 4,
    ],
    [
        0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 6, 7, 0, 3, 0, 2, 0, 7, 0, 0, 6, 6, 0, 0, 0, 0, 0, 0, 7, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 8, 0, 0, 0, 0, 0, 9, 0, 0, 0, 8, 0, 8, 0, 0, 2, 8, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
    ],
    [
        0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 11, 10, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 16, 0, 13, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 12, 16, 0, 0, 11, 10, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 11, 16, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 10, 16, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0,
    ],
];

/// Whether a segment may end after the forward state.
const ENDS: [bool; 20] = [
    false, true, true, true, false, true, true, false, true, true, true, true, true, true, true,
    true, true, true, true, true,
];

/// Backward automaton: the same classes, read from an offset towards the
/// start; stop (0) when a boundary is certain.
const PREVIOUS: [[u8; CLASSES]; 15] = [
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        2, 3, 4, 5, 6, 7, 8, 9, 4, 2, 10, 3, 11, 9, 12, 13, 14, 9, 7, 8, 2, 7,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 3, 4, 5, 6, 7, 8, 0, 4, 2, 10, 3, 11, 9, 12, 13, 0, 9, 0, 8, 2, 7,
    ],
    [
        0, 3, 0, 0, 6, 0, 0, 0, 0, 0, 0, 3, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 3, 4, 0, 6, 0, 0, 0, 4, 0, 0, 3, 11, 0, 0, 0, 0, 0, 7, 0, 0, 7,
    ],
    [
        0, 3, 0, 0, 6, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 9, 0, 0, 0, 9, 0, 0, 0, 0,
    ],
    [
        0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 12, 13, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 3, 4, 0, 6, 7, 0, 0, 4, 2, 0, 3, 11, 0, 0, 0, 0, 0, 0, 8, 0, 0,
    ],
    [
        0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 3, 0, 0, 0, 0, 0, 9, 0, 0, 0, 3, 0, 9, 0, 0, 0, 9, 0, 0, 0, 0,
    ],
];

/// The class of one code point. `None` is a character the rules skip
/// (an invisible format character inside a segment).
fn class_of(cp: u32) -> Option<usize> {
    match cp {
        0xAD => return Some(18),
        0x2E => return Some(2),
        0x2C | 0x66B => return Some(19),
        0x22 | 0x27 => return Some(8),
        0x0D => return Some(7),
        0x09 => return Some(17),
        0x0A | 0x0C | 0x2028 | 0x2029 => return Some(16),
        0x964 | 0x965 => return Some(5),
        0x3099 | 0x309A => return Some(1),
        0x309B | 0x309C | 0x30FB | 0x30FC => return Some(10),
        0x3041..=0x3094 | 0x309D | 0x309E => return Some(14),
        0x30A1..=0x30FA | 0x30FD | 0x30FE => return Some(15),
        0x25 | 0x26 | 0xA2 | 0x66A | 0x2030 | 0x2031 => return Some(6),
        0x2027 => return Some(21),
        0x23 => return Some(9),
        _ => {}
    }
    let t = jc::get_type(cp);
    if t == jc::FORMAT {
        // the rules ignore a format character only in the BMP
        return if cp < 0x10000 { None } else { Some(0) };
    }
    Some(match t {
        jc::CONTROL => 0,
        _ if matches!(cp, 0x3005 | 0x4E00..=0x9FA5 | 0xF900..=0xFA2D) => 3,
        jc::UPPERCASE_LETTER
        | jc::LOWERCASE_LETTER
        | jc::TITLECASE_LETTER
        | jc::MODIFIER_LETTER
        | jc::OTHER_LETTER
        | jc::COMBINING_SPACING_MARK => 4,
        jc::NON_SPACING_MARK | jc::ENCLOSING_MARK => 11,
        jc::DECIMAL_DIGIT_NUMBER | jc::LETTER_NUMBER | jc::OTHER_NUMBER => 12,
        jc::SPACE_SEPARATOR => 13,
        jc::CURRENCY_SYMBOL => 9,
        jc::DASH_PUNCTUATION | jc::CONNECTOR_PUNCTUATION => 21,
        _ => OTHER,
    })
}

/// `BreakIterator.getWordInstance(Locale.ROOT)` over one text.
#[derive(Debug, Clone)]
pub struct WordBreakIterator {
    text: Vec<u16>,
    /// The `CharacterIterator`'s position.
    pos: usize,
    /// `cachedLastKnownBreak`.
    cached_last_known_break: i64,
}

impl WordBreakIterator {
    /// `setText(text)`.
    pub fn new(text: &str) -> Self {
        Self::from_utf16(text.encode_utf16().collect())
    }

    /// `setText` over UTF-16 code units.
    pub fn from_utf16(text: Vec<u16>) -> Self {
        WordBreakIterator {
            text,
            pos: 0,
            cached_last_known_break: DONE,
        }
    }

    /// The text's length in code units.
    pub fn text_len(&self) -> usize {
        self.text.len()
    }

    // --- the CharacterIterator over the text ---------------------------

    /// `CharacterIterator.current()`.
    fn current_unit(&self) -> u32 {
        self.text.get(self.pos).map_or(END_CHAR, |&u| u32::from(u))
    }

    /// `CharacterIterator.next()`.
    fn next_unit(&mut self) -> u32 {
        if self.pos + 1 < self.text.len() {
            self.pos += 1;
            u32::from(self.text[self.pos])
        } else {
            self.pos = self.text.len();
            END_CHAR
        }
    }

    /// `CharacterIterator.previous()`.
    fn previous_unit(&mut self) -> u32 {
        if self.pos > 0 {
            self.pos -= 1;
            u32::from(self.text[self.pos])
        } else {
            END_CHAR
        }
    }

    /// `getCurrent()`: the code point at the cursor.
    fn get_current(&mut self) -> u32 {
        let ch = self.current_unit();
        if (0xD800..0xDC00).contains(&ch) && self.pos < self.text.len() {
            let lo = self.next_unit();
            self.previous_unit();
            if (0xDC00..0xE000).contains(&lo) {
                return 0x10000 + ((ch - 0xD800) << 10) + (lo - 0xDC00);
            }
        }
        ch
    }

    /// `getCurrentCodePointCount()`.
    fn current_code_point_count(&mut self) -> usize {
        let ch = self.current_unit();
        if (0xD800..0xDC00).contains(&ch) && self.pos < self.text.len() {
            let lo = self.next_unit();
            self.previous_unit();
            if (0xDC00..0xE000).contains(&lo) {
                return 2;
            }
        }
        1
    }

    /// `getNext()`: moves past the current code point and reads the next.
    fn get_next(&mut self) -> u32 {
        let end = self.text.len();
        let mut i = self.pos;
        if i == end {
            return END_CHAR;
        }
        i += self.current_code_point_count();
        if i >= end {
            return END_CHAR;
        }
        self.pos = i;
        self.get_current()
    }

    /// `getNextIndex()`: the index after the current code point.
    fn get_next_index(&mut self) -> usize {
        (self.pos + self.current_code_point_count()).min(self.text.len())
    }

    /// `getPrevious()`: moves back a code point and reads it.
    fn get_previous(&mut self) -> u32 {
        let ch = self.previous_unit();
        if (0xDC00..0xE000).contains(&ch) && self.pos > 0 {
            let hi = self.previous_unit();
            if (0xD800..0xDC00).contains(&hi) {
                return 0x10000 + ((hi - 0xD800) << 10) + (ch - 0xDC00);
            }
            self.next_unit();
        }
        ch
    }

    // --- the two automata ------------------------------------------------

    /// `handleNext()`: the end of the segment at the cursor, `DONE` at the
    /// end of the text.
    fn handle_next(&mut self) -> i64 {
        if self.pos == self.text.len() {
            return DONE;
        }
        let mut result = self.get_next_index();
        let mut state = 1usize;
        let mut c = self.get_current();
        while c != END_CHAR && state != 0 {
            if let Some(class) = class_of(c) {
                state = usize::from(NEXT[state][class]);
            }
            if ENDS[state] {
                result = self.get_next_index();
            }
            c = self.get_next();
        }
        self.pos = result;
        i64::try_from(result).unwrap_or(i64::MAX)
    }

    /// `handlePrevious()`: a boundary at or before the cursor.
    fn handle_previous(&mut self) -> i64 {
        let mut state = 1usize;
        let mut category: Option<usize> = Some(0);
        let mut last_category: Option<usize> = Some(0);
        let mut c = self.get_current();
        while c != END_CHAR && state != 0 {
            last_category = category;
            category = class_of(c);
            if let Some(class) = category {
                state = usize::from(PREVIOUS[state][class]);
            }
            c = self.get_previous();
        }
        if c != END_CHAR {
            if last_category.is_some() {
                self.get_next();
                self.get_next();
            } else {
                self.get_next();
            }
        }
        i64::try_from(self.pos).unwrap_or(i64::MAX)
    }

    // --- BreakIterator -----------------------------------------------------

    fn check_offset(&self, offset: usize) {
        assert!(offset <= self.text.len(), "offset out of bounds");
    }

    fn index(&self) -> i64 {
        i64::try_from(self.pos).unwrap_or(i64::MAX)
    }

    fn set_index(&mut self, at: i64) {
        self.pos = usize::try_from(at).unwrap_or(0);
    }

    /// `first()`.
    pub fn first(&mut self) -> usize {
        self.pos = 0;
        0
    }

    /// `next()`: the next boundary, `None` for `DONE`.
    #[allow(clippy::should_implement_trait)] // `BreakIterator.next`, not an `Iterator`
    pub fn next(&mut self) -> Option<usize> {
        usize::try_from(self.handle_next()).ok()
    }

    /// `previous()`: the previous boundary, `None` for `DONE`.
    pub fn previous(&mut self) -> Option<usize> {
        if self.pos == 0 {
            return None;
        }
        let start = self.index();
        let mut last_result = self.cached_last_known_break;
        if last_result >= start || last_result <= DONE {
            self.get_previous();
            last_result = self.handle_previous();
        } else {
            self.set_index(last_result);
        }
        let mut next_result = last_result;
        while next_result != DONE && next_result < start {
            last_result = next_result;
            next_result = self.handle_next();
        }
        self.set_index(last_result);
        self.cached_last_known_break = last_result;
        usize::try_from(last_result).ok()
    }

    /// `following(offset)`: the first boundary after `offset`, `None` for
    /// `DONE`.
    pub fn following(&mut self, offset: usize) -> Option<usize> {
        self.check_offset(offset);
        self.pos = offset;
        if offset == 0 {
            self.cached_last_known_break = self.handle_next();
            return usize::try_from(self.cached_last_known_break).ok();
        }
        let off = self.index();
        let mut result = self.cached_last_known_break;
        if result >= off || result <= DONE {
            result = self.handle_previous();
        } else {
            self.set_index(result);
        }
        while result != DONE && result <= off {
            result = self.handle_next();
        }
        self.cached_last_known_break = result;
        usize::try_from(result).ok()
    }

    /// `preceding(offset)`: the last boundary before `offset`, `None` for
    /// `DONE`.
    pub fn preceding(&mut self, offset: usize) -> Option<usize> {
        self.check_offset(offset);
        self.pos = offset;
        self.previous()
    }

    /// `isBoundary(offset)`.
    pub fn is_boundary(&mut self, offset: usize) -> bool {
        self.check_offset(offset);
        if offset == 0 {
            return true;
        }
        self.following(offset - 1) == Some(offset)
    }

    /// Every boundary, `first()` then `next()` until `DONE`.
    pub fn boundaries(&mut self) -> Vec<usize> {
        let mut out = vec![self.first()];
        while let Some(b) = self.next() {
            out.push(b);
        }
        out
    }
}
