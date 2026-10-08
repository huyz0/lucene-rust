//! `com.ibm.icu.impl.UCaseProps`: Unicode case mapping and folding over
//! ICU's `ucase.icu` (data format `cAsE`, format version 4) -- the full
//! lower, upper and title mappings with their context conditions (final
//! sigma, Lithuanian dots, Turkic i, Armenian ech-yiwn) and full case
//! folding, what the case transliterators (`Any-Lower`, `Any-Upper`,
//! `Any-Title`, `Any-CaseFold`) call.
//!
//! A code point's 16-bit properties (from a `Trie2_16`) hold its case type,
//! a delta to its simple mapping, or an offset into the exceptions string,
//! whose slots hold simple mappings, deltas, closures and full mappings.
//! Not ported: the case closure and unfolding (case-insensitive sets), the
//! binary case properties.

use std::sync::OnceLock;

use crate::icu4j::binary::read_header;
use crate::icu4j::trie2::Trie2_32;
use crate::icu4j::utf16;
use crate::IcuError;

const DATA: &[u8] = include_bytes!("../resources/ucase.icu");
const FMT: u32 = 0x6341_5345; // "cAsE"
const IX_TRIE_SIZE: usize = 2;
const IX_EXC_LENGTH: usize = 3;
const IX_TOP: usize = 16;

pub const NONE: i32 = 0;
pub const LOWER: i32 = 1;
const TYPE_MASK: i32 = 3;
const EXCEPTION: i32 = 8;
const DOT_MASK: i32 = 0x60;
const SOFT_DOTTED: i32 = 0x20;
const ABOVE: i32 = 0x40;
const OTHER_ACCENT: i32 = 0x60;
const DELTA_SHIFT: i32 = 7;
const EXC_SHIFT: i32 = 4;
const EXC_LOWER: i32 = 0;
const EXC_FOLD: i32 = 1;
const EXC_UPPER: i32 = 2;
const EXC_TITLE: i32 = 3;
const EXC_DELTA: i32 = 4;
const EXC_FULL_MAPPINGS: i32 = 7;
const EXC_DOUBLE_SLOTS: i32 = 0x100;
const EXC_NO_SIMPLE_CASE_FOLDING: i32 = 0x200;
const EXC_DELTA_IS_NEGATIVE: i32 = 0x400;
const EXC_DOT_SHIFT: i32 = 7;
const EXC_CONDITIONAL_SPECIAL: i32 = 0x4000;
const EXC_CONDITIONAL_FOLD: i32 = 0x8000;
const FULL_LOWER: i32 = 0xf;

/// `MAX_STRING_LENGTH`: a mapping result at most this is a string length.
pub const MAX_STRING_LENGTH: i32 = 0x1f;

/// `UCaseProps.LOC_*`.
pub const LOC_ROOT: i32 = 1;
pub const LOC_TURKISH: i32 = 2;
pub const LOC_LITHUANIAN: i32 = 3;
pub const LOC_GREEK: i32 = 4;
pub const LOC_DUTCH: i32 = 5;
pub const LOC_ARMENIAN: i32 = 6;

/// `UCaseProps.ContextIterator`: the text around the code point being
/// mapped (`reset(dir)`: 1 forward from after it, -1 backward from before
/// it; `next_context()`, Java's `next()`: the next code point, or a negative value at the end).
pub trait ContextIterator {
    fn reset(&mut self, dir: i32);
    fn next_context(&mut self) -> i32;
}

/// `UCaseProps`.
#[derive(Debug)]
pub struct UCaseProps {
    trie: Trie2_32,
    exceptions: Vec<u16>,
}

/// `UCaseProps.INSTANCE`.
pub fn instance() -> &'static UCaseProps {
    static P: OnceLock<UCaseProps> = OnceLock::new();
    P.get_or_init(|| UCaseProps::load(DATA).expect("vendored ucase.icu loads"))
}

/// `getCaseLocale(language)`.
pub fn case_locale(language: &str) -> i32 {
    match language {
        "en" => LOC_ROOT,
        l if l.len() == 2 && l.as_bytes()[0] > b't' => LOC_ROOT,
        "tr" | "az" | "tur" | "aze" => LOC_TURKISH,
        "el" | "ell" => LOC_GREEK,
        "lt" | "lit" => LOC_LITHUANIAN,
        "nl" | "nld" => LOC_DUTCH,
        "hy" | "hye" => LOC_ARMENIAN,
        _ => LOC_ROOT,
    }
}

/// `flagsOffset[]`: the number of set bits in a byte.
#[inline]
fn flags_offset(flags: i32) -> i32 {
    (flags & 0xff).count_ones() as i32
}

/// The full-mapping result of a case function: a code point (`>
/// MAX_STRING_LENGTH`), a string length (`out` holds it), or `~c` (`< 0`,
/// unchanged).
pub type CaseResult = i32;

// ARITH: (the whole impl) offsets into the exceptions string come from the
// data, every read bounds-checked (`exc`); code point deltas stay within
// the code space for the vendored data.
#[allow(clippy::arithmetic_side_effects)]
impl UCaseProps {
    /// `readData(bytes)`.
    pub fn load(bytes: &[u8]) -> Result<UCaseProps, IcuError> {
        let (mut r, _) = read_header(bytes, FMT, |v| v[0] == 4)?;
        let count = usize::try_from(r.i32()?).unwrap_or(0);
        if count < IX_TOP {
            return Err(IcuError::new("indexes[0] too small in ucase.icu"));
        }
        let mut indexes = vec![0i32; count];
        indexes[0] = count as i32;
        for slot in indexes.iter_mut().skip(1) {
            *slot = r.i32()?;
        }
        let trie = Trie2_32::from_serialized_16(&mut r)?;
        let expected = usize::try_from(indexes[IX_TRIE_SIZE]).unwrap_or(0);
        let length = trie.serialized_length();
        if length > expected {
            return Err(IcuError::new("ucase.icu: not enough bytes for the trie"));
        }
        r.skip(expected - length)?;
        let exc_length = usize::try_from(indexes[IX_EXC_LENGTH]).unwrap_or(0);
        let exceptions = r.u16s(exc_length)?;
        Ok(UCaseProps { trie, exceptions })
    }

    #[inline]
    fn exc(&self, i: i32) -> i32 {
        usize::try_from(i)
            .ok()
            .and_then(|i| self.exceptions.get(i))
            .map_or(0, |&u| i32::from(u))
    }

    #[inline]
    fn props(&self, c: i32) -> i32 {
        self.trie.get(c)
    }

    fn has_slot(flags: i32, index: i32) -> bool {
        flags & (1 << index) != 0
    }

    fn slot_offset(flags: i32, index: i32) -> i32 {
        flags_offset(flags & ((1 << index) - 1))
    }

    /// `getSlotValueAndOffset`: the value and the offset of its last unit.
    fn slot_value_and_offset(&self, exc_word: i32, index: i32, exc_offset: i32) -> (i32, i32) {
        if exc_word & EXC_DOUBLE_SLOTS == 0 {
            let o = exc_offset + Self::slot_offset(exc_word, index);
            (self.exc(o), o)
        } else {
            let o = exc_offset + 2 * Self::slot_offset(exc_word, index);
            ((self.exc(o) << 16) | self.exc(o + 1), o + 1)
        }
    }

    fn slot_value(&self, exc_word: i32, index: i32, exc_offset: i32) -> i32 {
        self.slot_value_and_offset(exc_word, index, exc_offset).0
    }

    fn delta(props: i32) -> i32 {
        i32::from(props as i16) >> DELTA_SHIFT
    }

    fn is_upper_or_title(props: i32) -> bool {
        props & 2 != 0
    }

    /// `getType(c)`.
    pub fn get_type(&self, c: i32) -> i32 {
        self.props(c) & TYPE_MASK
    }

    /// `getTypeOrIgnorable(c)`.
    pub fn get_type_or_ignorable(&self, c: i32) -> i32 {
        self.props(c) & 7
    }

    /// `getDotType(c)`.
    fn dot_type(&self, c: i32) -> i32 {
        let props = self.props(c);
        if props & EXCEPTION == 0 {
            props & DOT_MASK
        } else {
            (self.exc(props >> EXC_SHIFT) >> EXC_DOT_SHIFT) & DOT_MASK
        }
    }

    /// `isCaseSensitive(c)`.
    pub fn is_case_sensitive(&self, c: i32) -> bool {
        let props = self.props(c);
        if props & EXCEPTION == 0 {
            props & 0x10 != 0
        } else {
            self.exc(props >> EXC_SHIFT) & 0x800 != 0
        }
    }

    fn append_exc(&self, out: &mut Vec<u16>, start: i32, len: i32) {
        for i in start..start + len {
            out.push(self.exc(i) as u16);
        }
    }

    fn is_followed_by_cased_letter(&self, iter: &mut dyn ContextIterator, dir: i32) -> bool {
        iter.reset(dir);
        loop {
            let c = iter.next_context();
            if c < 0 {
                return false;
            }
            let t = self.get_type_or_ignorable(c);
            if t & 4 != 0 {
                continue;
            }
            return t != NONE;
        }
    }

    fn is_preceded_by_soft_dotted(&self, iter: &mut dyn ContextIterator) -> bool {
        iter.reset(-1);
        loop {
            let c = iter.next_context();
            if c < 0 {
                return false;
            }
            let d = self.dot_type(c);
            if d == SOFT_DOTTED {
                return true;
            } else if d != OTHER_ACCENT {
                return false;
            }
        }
    }

    fn is_preceded_by_capital_i(&self, iter: &mut dyn ContextIterator) -> bool {
        iter.reset(-1);
        loop {
            let c = iter.next_context();
            if c < 0 {
                return false;
            }
            if c == 0x49 {
                return true;
            }
            if self.dot_type(c) != OTHER_ACCENT {
                return false;
            }
        }
    }

    fn is_followed_by_more_above(&self, iter: &mut dyn ContextIterator) -> bool {
        iter.reset(1);
        loop {
            let c = iter.next_context();
            if c < 0 {
                return false;
            }
            let d = self.dot_type(c);
            if d == ABOVE {
                return true;
            } else if d != OTHER_ACCENT {
                return false;
            }
        }
    }

    fn is_followed_by_dot_above(&self, iter: &mut dyn ContextIterator) -> bool {
        iter.reset(1);
        loop {
            let c = iter.next_context();
            if c < 0 {
                return false;
            }
            if c == 0x307 {
                return true;
            }
            if self.dot_type(c) != OTHER_ACCENT {
                return false;
            }
        }
    }

    /// `toFullLower(c, iter, out, caseLocale)`.
    pub fn to_full_lower(
        &self,
        c: i32,
        iter: &mut dyn ContextIterator,
        out: &mut Vec<u16>,
        loc: i32,
    ) -> CaseResult {
        let mut result = c;
        let props = self.props(c);
        if props & EXCEPTION == 0 {
            if Self::is_upper_or_title(props) {
                result = c + Self::delta(props);
            }
        } else {
            let mut exc_offset = props >> EXC_SHIFT;
            let exc_word = self.exc(exc_offset);
            exc_offset += 1;
            let exc_offset2 = exc_offset;
            if exc_word & EXC_CONDITIONAL_SPECIAL != 0 {
                if loc == LOC_LITHUANIAN
                    && (((c == 0x49 || c == 0x4a || c == 0x12e)
                        && self.is_followed_by_more_above(iter))
                        || c == 0xcc
                        || c == 0xcd
                        || c == 0x128)
                {
                    let (s, n): (&[u16], i32) = match c {
                        0x49 => (&[0x69, 0x307], 2),
                        0x4a => (&[0x6a, 0x307], 2),
                        0x12e => (&[0x12f, 0x307], 2),
                        0xcc => (&[0x69, 0x307, 0x300], 3),
                        0xcd => (&[0x69, 0x307, 0x301], 3),
                        _ => (&[0x69, 0x307, 0x303], 3),
                    };
                    out.extend_from_slice(s);
                    return n;
                } else if loc == LOC_TURKISH && c == 0x130 {
                    return 0x69;
                } else if loc == LOC_TURKISH && c == 0x307 && self.is_preceded_by_capital_i(iter) {
                    return 0;
                } else if loc == LOC_TURKISH && c == 0x49 && !self.is_followed_by_dot_above(iter) {
                    return 0x131;
                } else if c == 0x130 {
                    out.extend_from_slice(&[0x69, 0x307]);
                    return 2;
                } else if c == 0x3a3
                    && !self.is_followed_by_cased_letter(iter, 1)
                    && self.is_followed_by_cased_letter(iter, -1)
                {
                    return 0x3c2;
                }
            } else if Self::has_slot(exc_word, EXC_FULL_MAPPINGS) {
                let (value, o) =
                    self.slot_value_and_offset(exc_word, EXC_FULL_MAPPINGS, exc_offset);
                let full = value & FULL_LOWER;
                if full != 0 {
                    self.append_exc(out, o + 1, full);
                    return full;
                }
            }
            if Self::has_slot(exc_word, EXC_DELTA) && Self::is_upper_or_title(props) {
                let delta = self.slot_value(exc_word, EXC_DELTA, exc_offset2);
                return if exc_word & EXC_DELTA_IS_NEGATIVE == 0 {
                    c + delta
                } else {
                    c - delta
                };
            }
            if Self::has_slot(exc_word, EXC_LOWER) {
                result = self.slot_value(exc_word, EXC_LOWER, exc_offset2);
            }
        }
        if result == c {
            !result
        } else {
            result
        }
    }

    /// `toUpperOrTitle(c, iter, out, loc, upperNotTitle)`.
    fn to_upper_or_title(
        &self,
        c: i32,
        iter: &mut dyn ContextIterator,
        out: &mut Vec<u16>,
        loc: i32,
        upper_not_title: bool,
    ) -> CaseResult {
        let mut result = c;
        let props = self.props(c);
        if props & EXCEPTION == 0 {
            if props & TYPE_MASK == LOWER {
                result = c + Self::delta(props);
            }
        } else {
            let mut exc_offset = props >> EXC_SHIFT;
            let exc_word = self.exc(exc_offset);
            exc_offset += 1;
            let exc_offset2 = exc_offset;
            if exc_word & EXC_CONDITIONAL_SPECIAL != 0 {
                if loc == LOC_TURKISH && c == 0x69 {
                    return 0x130;
                } else if loc == LOC_LITHUANIAN
                    && c == 0x307
                    && self.is_preceded_by_soft_dotted(iter)
                {
                    return 0;
                } else if c == 0x587 {
                    // ech-yiwn: "ԵՎ"/"Եվ" in Armenian, else "ԵՒ"/"Եւ".
                    let s: [u16; 2] = if loc == LOC_ARMENIAN {
                        if upper_not_title {
                            [0x535, 0x54e]
                        } else {
                            [0x535, 0x57e]
                        }
                    } else if upper_not_title {
                        [0x535, 0x552]
                    } else {
                        [0x535, 0x582]
                    };
                    out.extend_from_slice(&s);
                    return 2;
                }
            } else if Self::has_slot(exc_word, EXC_FULL_MAPPINGS) {
                let (value, o) =
                    self.slot_value_and_offset(exc_word, EXC_FULL_MAPPINGS, exc_offset);
                let mut full = value & 0xffff;
                let mut off = o + 1;
                off += full & FULL_LOWER;
                full >>= 4;
                off += full & 0xf;
                full >>= 4;
                if upper_not_title {
                    full &= 0xf;
                } else {
                    off += full & 0xf;
                    full = (full >> 4) & 0xf;
                }
                if full != 0 {
                    self.append_exc(out, off, full);
                    return full;
                }
            }
            if Self::has_slot(exc_word, EXC_DELTA) && props & TYPE_MASK == LOWER {
                let delta = self.slot_value(exc_word, EXC_DELTA, exc_offset2);
                return if exc_word & EXC_DELTA_IS_NEGATIVE == 0 {
                    c + delta
                } else {
                    c - delta
                };
            }
            let index = if !upper_not_title && Self::has_slot(exc_word, EXC_TITLE) {
                EXC_TITLE
            } else if Self::has_slot(exc_word, EXC_UPPER) {
                EXC_UPPER
            } else {
                return !c;
            };
            result = self.slot_value(exc_word, index, exc_offset2);
        }
        if result == c {
            !result
        } else {
            result
        }
    }

    /// `toFullUpper(c, iter, out, caseLocale)`.
    pub fn to_full_upper(
        &self,
        c: i32,
        iter: &mut dyn ContextIterator,
        out: &mut Vec<u16>,
        loc: i32,
    ) -> CaseResult {
        self.to_upper_or_title(c, iter, out, loc, true)
    }

    /// `toFullTitle(c, iter, out, caseLocale)`.
    pub fn to_full_title(
        &self,
        c: i32,
        iter: &mut dyn ContextIterator,
        out: &mut Vec<u16>,
        loc: i32,
    ) -> CaseResult {
        self.to_upper_or_title(c, iter, out, loc, false)
    }

    /// `toFullFolding(c, out, options)` with `FOLD_CASE_DEFAULT` (`turkic`
    /// false) or `FOLD_CASE_EXCLUDE_SPECIAL_I` (true).
    pub fn to_full_folding(&self, c: i32, out: &mut Vec<u16>, turkic: bool) -> CaseResult {
        let mut result = c;
        let props = self.props(c);
        if props & EXCEPTION == 0 {
            if Self::is_upper_or_title(props) {
                result = c + Self::delta(props);
            }
        } else {
            let mut exc_offset = props >> EXC_SHIFT;
            let exc_word = self.exc(exc_offset);
            exc_offset += 1;
            let exc_offset2 = exc_offset;
            if exc_word & EXC_CONDITIONAL_FOLD != 0 {
                if !turkic {
                    if c == 0x49 {
                        return 0x69;
                    } else if c == 0x130 {
                        out.extend_from_slice(&[0x69, 0x307]);
                        return 2;
                    }
                } else if c == 0x49 {
                    return 0x131;
                } else if c == 0x130 {
                    return 0x69;
                }
            } else if Self::has_slot(exc_word, EXC_FULL_MAPPINGS) {
                let (value, o) =
                    self.slot_value_and_offset(exc_word, EXC_FULL_MAPPINGS, exc_offset);
                let full = value & 0xffff;
                let off = o + 1 + (full & FULL_LOWER);
                let full = (full >> 4) & 0xf;
                if full != 0 {
                    self.append_exc(out, off, full);
                    return full;
                }
            }
            if exc_word & EXC_NO_SIMPLE_CASE_FOLDING != 0 {
                return !c;
            }
            if Self::has_slot(exc_word, EXC_DELTA) && Self::is_upper_or_title(props) {
                let delta = self.slot_value(exc_word, EXC_DELTA, exc_offset2);
                return if exc_word & EXC_DELTA_IS_NEGATIVE == 0 {
                    c + delta
                } else {
                    c - delta
                };
            }
            let index = if Self::has_slot(exc_word, EXC_FOLD) {
                EXC_FOLD
            } else if Self::has_slot(exc_word, EXC_LOWER) {
                EXC_LOWER
            } else {
                return !c;
            };
            result = self.slot_value(exc_word, index, exc_offset2);
        }
        if result == c {
            !result
        } else {
            result
        }
    }
}

/// Appends a case function's result for `c` to `dest` (Java's callers:
/// the string, the mapped code point, or `c` itself when unchanged).
pub fn append_result(dest: &mut Vec<u16>, result: CaseResult, out: &[u16]) {
    if result < 0 {
        utf16::push_code_point(dest, !result);
    } else if result <= MAX_STRING_LENGTH {
        dest.extend_from_slice(out);
    } else {
        utf16::push_code_point(dest, result);
    }
}

#[cfg(test)]
// ARITH: (the whole module) test code building small inputs by hand.
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    struct NoContext;
    impl ContextIterator for NoContext {
        fn reset(&mut self, _dir: i32) {}
        fn next_context(&mut self) -> i32 {
            -1
        }
    }

    #[test]
    fn mappings() {
        let p = instance();
        let mut out = Vec::new();
        assert_eq!(
            p.to_full_lower(0x41, &mut NoContext, &mut out, LOC_ROOT),
            0x61
        );
        assert_eq!(
            p.to_full_lower(0x61, &mut NoContext, &mut out, LOC_ROOT),
            !0x61
        );
        assert_eq!(p.to_full_upper(0xdf, &mut NoContext, &mut out, LOC_ROOT), 2);
        assert_eq!(out, [0x53, 0x53]);
        out.clear();
        assert_eq!(
            p.to_full_title(0x1c6, &mut NoContext, &mut out, LOC_ROOT),
            0x1c5
        );
        assert_eq!(p.to_full_folding(0xdf, &mut out, false), 2);
        assert_eq!(out, [0x73, 0x73]);
        out.clear();
        assert_eq!(p.to_full_folding(0x130, &mut out, false), 2);
        assert_eq!(p.to_full_folding(0x49, &mut out, true), 0x131);
        assert_eq!(
            p.to_full_lower(0x130, &mut NoContext, &mut out, LOC_TURKISH),
            0x69
        );
        assert_eq!(
            p.to_full_upper(0x69, &mut NoContext, &mut out, LOC_TURKISH),
            0x130
        );
        assert_eq!(case_locale("tr"), LOC_TURKISH);
        assert_eq!(case_locale("zz"), LOC_ROOT);
        assert_eq!(case_locale("hye"), LOC_ARMENIAN);
        assert!(p.is_case_sensitive(0x41));
        assert_eq!(p.get_type(0x41), 2);
        let mut d = Vec::new();
        append_result(&mut d, !0x41, &[]);
        append_result(&mut d, 2, &[0x53, 0x53]);
        append_result(&mut d, 0x1f600, &[]);
        assert_eq!(d, [0x41, 0x53, 0x53, 0xd83d, 0xde00]);
        assert!(UCaseProps::load(&[0; 30]).is_err());
        // Too few indexes; a trie longer than its index says.
        let h = usize::from(u16::from_be_bytes([DATA[0], DATA[1]]));
        let mut b = DATA.to_vec();
        b[h..h + 4].copy_from_slice(&5i32.to_be_bytes());
        assert!(UCaseProps::load(&b).is_err());
        let mut b = DATA.to_vec();
        let at = h + 4 * IX_TRIE_SIZE;
        b[at..at + 4].copy_from_slice(&8i32.to_be_bytes());
        assert!(UCaseProps::load(&b).is_err());
    }

    /// The text around a code point: `before` (nearest last) and `after`.
    struct Around<'a> {
        before: &'a [i32],
        after: &'a [i32],
        dir: i32,
        i: usize,
    }

    impl ContextIterator for Around<'_> {
        fn reset(&mut self, dir: i32) {
            self.dir = dir;
            self.i = 0;
        }
        fn next_context(&mut self) -> i32 {
            let side = if self.dir > 0 {
                self.after
            } else {
                self.before
            };
            let c = side.get(self.i).copied().unwrap_or(-1);
            self.i += 1;
            c
        }
    }

    fn around<'a>(before: &'a [i32], after: &'a [i32]) -> Around<'a> {
        Around {
            before,
            after,
            dir: 0,
            i: 0,
        }
    }

    /// SpecialCasing.txt's conditional mappings, per case locale.
    #[test]
    fn conditional_mappings() {
        let p = instance();
        let lower = |c: i32, before: &[i32], after: &[i32], loc: i32| {
            let mut out = Vec::new();
            let r = p.to_full_lower(c, &mut around(before, after), &mut out, loc);
            let mut d = Vec::new();
            append_result(&mut d, r, &out);
            d
        };
        let upper = |c: i32, before: &[i32], after: &[i32], loc: i32| {
            let mut out = Vec::new();
            let r = p.to_full_upper(c, &mut around(before, after), &mut out, loc);
            let mut d = Vec::new();
            append_result(&mut d, r, &out);
            d
        };
        // Lithuanian: keep the dot of i before more accents above.
        assert_eq!(lower(0x49, &[], &[0x300], LOC_LITHUANIAN), [0x69, 0x307]);
        assert_eq!(lower(0x49, &[], &[0x41], LOC_LITHUANIAN), [0x69]);
        assert_eq!(
            lower(0x49, &[], &[0x334, 0x301], LOC_LITHUANIAN),
            [0x69, 0x307]
        );
        assert_eq!(lower(0x49, &[], &[], LOC_LITHUANIAN), [0x69]);
        assert_eq!(lower(0x4a, &[], &[0x301], LOC_LITHUANIAN), [0x6a, 0x307]);
        assert_eq!(lower(0x12e, &[], &[0x301], LOC_LITHUANIAN), [0x12f, 0x307]);
        assert_eq!(lower(0xcc, &[], &[], LOC_LITHUANIAN), [0x69, 0x307, 0x300]);
        assert_eq!(lower(0xcd, &[], &[], LOC_LITHUANIAN), [0x69, 0x307, 0x301]);
        assert_eq!(lower(0x128, &[], &[], LOC_LITHUANIAN), [0x69, 0x307, 0x303]);
        assert_eq!(
            upper(0x307, &[0x69], &[], LOC_LITHUANIAN),
            Vec::<u16>::new()
        );
        assert_eq!(
            upper(0x307, &[0x334, 0x69], &[], LOC_LITHUANIAN),
            Vec::<u16>::new()
        );
        assert_eq!(upper(0x307, &[0x61], &[], LOC_LITHUANIAN), [0x307]);
        assert_eq!(upper(0x307, &[], &[], LOC_LITHUANIAN), [0x307]);
        // Turkic: dotless and dotted i.
        assert_eq!(lower(0x307, &[0x49], &[], LOC_TURKISH), Vec::<u16>::new());
        assert_eq!(
            lower(0x307, &[0x334, 0x49], &[], LOC_TURKISH),
            Vec::<u16>::new()
        );
        assert_eq!(lower(0x307, &[0x41], &[], LOC_TURKISH), [0x307]);
        assert_eq!(lower(0x307, &[], &[], LOC_TURKISH), [0x307]);
        assert_eq!(lower(0x49, &[], &[0x41], LOC_TURKISH), [0x131]);
        assert_eq!(lower(0x49, &[], &[0x334, 0x307], LOC_TURKISH), [0x69]);
        assert_eq!(lower(0x49, &[], &[], LOC_TURKISH), [0x131]);
        assert_eq!(lower(0x130, &[], &[], LOC_ROOT), [0x69, 0x307]);
        // Final sigma: after a cased letter (skipping ignorables), not before one.
        assert_eq!(lower(0x3a3, &[0x391], &[], LOC_ROOT), [0x3c2]);
        assert_eq!(lower(0x3a3, &[0x27, 0x391], &[0x20], LOC_ROOT), [0x3c2]);
        assert_eq!(lower(0x3a3, &[0x391], &[0x391], LOC_ROOT), [0x3c3]);
        assert_eq!(lower(0x3a3, &[], &[], LOC_ROOT), [0x3c3]);
        // Armenian ech-yiwn.
        assert_eq!(upper(0x587, &[], &[], LOC_ARMENIAN), [0x535, 0x54e]);
        assert_eq!(upper(0x587, &[], &[], LOC_ROOT), [0x535, 0x552]);
        let mut out = Vec::new();
        assert_eq!(
            p.to_full_title(0x587, &mut NoContext, &mut out, LOC_ARMENIAN),
            2
        );
        assert_eq!(out, [0x535, 0x57e]);
        out.clear();
        assert_eq!(
            p.to_full_title(0x587, &mut NoContext, &mut out, LOC_ROOT),
            2
        );
        assert_eq!(out, [0x535, 0x582]);
        // Unconditional exceptions: full title of a ligature, simple ones.
        out.clear();
        assert_eq!(
            p.to_full_title(0xfb01, &mut NoContext, &mut out, LOC_ROOT),
            2
        );
        assert_eq!(out, [0x46, 0x69]);
        assert_eq!(lower(0x1c5, &[], &[], LOC_ROOT), [0x1c6]);
        assert_eq!(upper(0x1c5, &[], &[], LOC_ROOT), [0x1c4]);
        assert_eq!(upper(0x1c4, &[], &[], LOC_ROOT), [0x1c4]);
        assert_eq!(lower(0x212a, &[], &[], LOC_ROOT), [0x6b]);
        assert_eq!(upper(0x3c2, &[], &[], LOC_ROOT), [0x3a3]);
        assert_eq!(upper(0x1e9e, &[], &[], LOC_ROOT), [0x1e9e]);
        let mut out = Vec::new();
        assert_eq!(p.to_full_folding(0x3c2, &mut out, false), 0x3c3);
        assert_eq!(p.to_full_folding(0x130, &mut out, true), 0x69);
        assert_eq!(p.to_full_folding(0x49, &mut out, false), 0x69);
        assert_eq!(p.to_full_folding(0x1e9e, &mut out, false), 2);
        assert_eq!(p.to_full_folding(0x1c5, &mut Vec::new(), false), 0x1c6);
        assert_eq!(p.to_full_folding(0x61, &mut Vec::new(), false), !0x61);
        assert_eq!(p.to_full_folding(0x131, &mut Vec::new(), false), !0x131);
        for c in [0x41, 0x307, 0x69, 0x334, 0x3a3, 0x1c5] {
            let _ = p.get_type_or_ignorable(c);
            let _ = p.is_case_sensitive(c);
        }
        assert!(p.is_case_sensitive(0x1c5));
        for l in [
            "en", "uk", "az", "aze", "el", "ell", "lt", "lit", "nl", "nld", "hy", "de",
        ] {
            assert!(case_locale(l) >= LOC_ROOT);
        }
    }
}
