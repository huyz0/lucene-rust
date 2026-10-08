//! `org.apache.lucene.analysis.icu.segmentation.ScriptIterator`: splits text
//! into runs of one script, Common and Inherited joining their neighbours,
//! a code point whose Script_Extensions holds the run's script staying in
//! it, combining marks never starting a run -- and, with `combineCJ`, Han,
//! Hiragana and Katakana as one Japanese run (full-width digits as Latin).

use crate::icu4j::uprops;
use crate::icu4j::utf16;

/// `UScript.COMMON`.
pub const COMMON: i32 = 0;
/// `UScript.INHERITED`.
pub const INHERITED: i32 = 1;
/// `UScript.HAN`.
pub const HAN: i32 = 17;
/// `UScript.HANGUL`.
pub const HANGUL: i32 = 18;
/// `UScript.HIRAGANA`.
pub const HIRAGANA: i32 = 20;
/// `UScript.KATAKANA`.
pub const KATAKANA: i32 = 22;
/// `UScript.LATIN`.
pub const LATIN: i32 = 25;
/// `UScript.MYANMAR`.
pub const MYANMAR: i32 = 28;
/// `UScript.JAPANESE`.
pub const JAPANESE: i32 = 105;
/// `UScript.INVALID_CODE`.
pub const INVALID_CODE: i32 = -1;

/// `ScriptIterator`.
#[derive(Debug, Clone)]
pub struct ScriptIterator {
    start: usize,
    limit: usize,
    index: usize,
    script_start: usize,
    script_limit: usize,
    script_code: i32,
    combine_cj: bool,
}

impl ScriptIterator {
    /// `new ScriptIterator(combineCJ)`.
    pub fn new(combine_cj: bool) -> Self {
        ScriptIterator {
            start: 0,
            limit: 0,
            index: 0,
            script_start: 0,
            script_limit: 0,
            script_code: INVALID_CODE,
            combine_cj,
        }
    }

    /// `getScriptStart()`.
    pub fn script_start(&self) -> usize {
        self.script_start
    }

    /// `getScriptLimit()`.
    pub fn script_limit(&self) -> usize {
        self.script_limit
    }

    /// `getScriptCode()`.
    pub fn script_code(&self) -> i32 {
        self.script_code
    }

    /// `next()` over `text` (the same buffer `set_text` was given).
    pub fn next(&mut self, text: &[u16]) -> bool {
        if self.script_limit >= self.limit {
            return false;
        }
        self.script_code = COMMON;
        self.script_start = self.script_limit;
        while self.index < self.limit {
            let ch = char_at_absolute(text, self.start, self.limit, self.index);
            let sc = self.get_script(ch);
            if is_same_script(self.script_code, sc, ch) || is_combining_mark(ch) {
                self.index = self.index.saturating_add(if ch > 0xffff { 2 } else { 1 });
                if self.script_code <= INHERITED && sc > INHERITED {
                    self.script_code = sc;
                }
            } else {
                break;
            }
        }
        self.script_limit = self.index;
        true
    }

    /// `setText(text, start, length)`.
    pub fn set_text(&mut self, start: usize, length: usize) {
        self.start = start;
        self.index = start;
        self.limit = start.saturating_add(length);
        self.script_start = start;
        self.script_limit = start;
        self.script_code = INVALID_CODE;
    }

    /// `getScript(codepoint)`.
    fn get_script(&self, codepoint: i32) -> i32 {
        if (0..128).contains(&codepoint) {
            return uprops::script(codepoint);
        }
        let script = uprops::script(codepoint);
        if self.combine_cj {
            if script == HAN || script == HIRAGANA || script == KATAKANA {
                JAPANESE
            } else if (0xff10..=0xff19).contains(&codepoint) {
                LATIN
            } else {
                script
            }
        } else {
            script
        }
    }
}

/// `UTF16.charAt(char[] text, int start, int limit, int offset)` with the
/// offset already absolute: the code point at `offset`, a surrogate pair
/// only within `[start, limit)`.
pub(crate) fn char_at_absolute(text: &[u16], start: usize, limit: usize, offset: usize) -> i32 {
    let c = text.get(offset).map_or(0, |&u| i32::from(u));
    if utf16::is_lead(c) {
        let n = offset.saturating_add(1);
        if n < limit {
            if let Some(&t) = text.get(n) {
                if utf16::is_trail(i32::from(t)) {
                    return utf16::to_code_point(c, i32::from(t));
                }
            }
        }
    } else if utf16::is_trail(c) && offset > start {
        if let Some(&l) = text.get(offset.saturating_sub(1)) {
            if utf16::is_lead(i32::from(l)) {
                return utf16::to_code_point(i32::from(l), c);
            }
        }
    }
    c
}

/// `isSameScript(currentScript, script, codepoint)`.
fn is_same_script(current_script: i32, script: i32, codepoint: i32) -> bool {
    current_script == script
        || current_script <= INHERITED
        || script <= INHERITED
        || uprops::has_script(codepoint, current_script)
}

/// `isCombiningMark(codepoint)`.
fn is_combining_mark(codepoint: i32) -> bool {
    let t = uprops::char_type(codepoint);
    t == uprops::COMBINING_SPACING_MARK
        || t == uprops::NON_SPACING_MARK
        || t == uprops::ENCLOSING_MARK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs(s: &str, combine: bool) -> Vec<(usize, usize, i32)> {
        let t = utf16::units(s);
        let mut it = ScriptIterator::new(combine);
        it.set_text(0, t.len());
        let mut out = Vec::new();
        while it.next(&t) {
            out.push((it.script_start(), it.script_limit(), it.script_code()));
        }
        out
    }

    #[test]
    fn splits_by_script() {
        assert_eq!(runs("abc ддд", false), vec![(0, 4, LATIN), (4, 7, 8)]);
        assert_eq!(runs("漢字かな", true), vec![(0, 4, JAPANESE)]);
        assert_eq!(runs("漢字かな", false), vec![(0, 2, HAN), (2, 4, HIRAGANA)]);
        assert_eq!(runs("１２", true), vec![(0, 2, LATIN)]);
        assert_eq!(runs("1 \u{301}", false), vec![(0, 3, COMMON)]);
        assert_eq!(runs("", false), vec![]);
        assert_eq!(char_at_absolute(&[0xd800, 0xdc00], 0, 1, 0), 0xd800);
        assert_eq!(char_at_absolute(&[0xd800, 0xdc00], 0, 2, 1), 0x10000);
        assert_eq!(char_at_absolute(&[0xd800, 0xdc00], 1, 2, 1), 0xdc00);
    }
}
