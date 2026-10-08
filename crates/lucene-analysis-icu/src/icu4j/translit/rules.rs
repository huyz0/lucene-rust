//! The rule engine of `com.ibm.icu.text.RuleBasedTransliterator`:
//! `RuleBasedTransliterator.Data`, `TransliterationRuleSet`,
//! `TransliterationRule`, and the matchers and replacers a rule's
//! stand-in characters name -- `UnicodeSet`, `StringMatcher` (strings and
//! `(segments)`), `Quantifier` (`* + ?`), `StringReplacer` (output with a
//! cursor) and `FunctionReplacer` (`&Translit(...)`).
//!
//! A parsed rule is text in which private-use stand-ins (from
//! `variables_base`) name matchers and replacers. Java keeps a segment's
//! match on its `StringMatcher`; here it lives in a per-call [`MatchState`]
//! (the data is shared and immutable), reset where Java resets it. Only
//! non-incremental transliteration is ported (Lucene's and the
//! transliterators' own calls are all non-incremental), so the partial-match
//! paths reduce to their non-incremental branches.

use std::collections::HashMap;
use std::sync::Arc;

use crate::icu4j::translit::{
    char32_at, char_count, copy_text, replace_text, Position, Transliterator,
};
use crate::icu4j::unicode_set::UnicodeSet;
use crate::icu4j::utf16;
use crate::IcuError;

/// `UnicodeMatcher.U_MISMATCH`, `U_PARTIAL_MATCH`, `U_MATCH`.
pub const U_MISMATCH: i32 = 0;
pub const U_MATCH: i32 = 2;

/// `UnicodeMatcher.ETHER`.
const ETHER: i32 = 0xffff;

/// A matcher or replacer a stand-in names.
#[derive(Debug, Clone)]
pub enum Var {
    /// `UnicodeSet`.
    Set(UnicodeSet),
    /// `StringMatcher` (`segment` > 0 for a `(segment)`, also a replacer
    /// then: `$n`).
    Matcher { pattern: Vec<u16>, segment: usize },
    /// `Quantifier` over a `StringMatcher`.
    Quantifier {
        pattern: Vec<u16>,
        min: i32,
        max: i32,
    },
    /// `FunctionReplacer`: a replacer's output through a transliterator.
    Function {
        translit: Arc<Transliterator>,
        replacer: StringReplacer,
    },
    /// A segment's slot before its `)` is parsed (Java's `null`).
    Pending,
}

/// `RuleBasedTransliterator.Data`.
#[derive(Debug, Clone, Default)]
pub struct Data {
    pub rule_set: RuleSet,
    pub variable_names: HashMap<String, Vec<u16>>,
    pub variables: Vec<Var>,
    pub variables_base: u16,
}

/// Each segment matcher's last match (`matchStart`, `matchLimit`), by
/// variable index.
#[derive(Debug)]
pub struct MatchState {
    matches: Vec<(i32, i32)>,
}

impl MatchState {
    pub fn new(n: usize) -> Self {
        MatchState {
            matches: vec![(-1, -1); n],
        }
    }

    fn get(&self, i: Option<usize>) -> (i32, i32) {
        i.and_then(|i| self.matches.get(i))
            .copied()
            .unwrap_or((-1, -1))
    }

    fn set(&mut self, i: Option<usize>, v: (i32, i32)) {
        if let Some(slot) = i.and_then(|i| self.matches.get_mut(i)) {
            *slot = v;
        }
    }
}

// ARITH: (the whole impl) stand-in arithmetic on u16 values and text
// positions within the text's length (Java's int arithmetic).
#[allow(clippy::arithmetic_side_effects)]
impl Data {
    /// The variable index of a stand-in.
    #[inline]
    fn index_of(&self, c: i32) -> Option<usize> {
        let i = c - i32::from(self.variables_base);
        usize::try_from(i)
            .ok()
            .filter(|&i| i < self.variables.len())
    }

    /// `lookupMatcher(c)`: the matcher a stand-in names.
    pub fn lookup_matcher(&self, c: i32) -> Option<(usize, &Var)> {
        let i = self.index_of(c)?;
        match &self.variables[i] {
            v @ (Var::Set(_) | Var::Matcher { .. } | Var::Quantifier { .. }) => Some((i, v)),
            _ => None,
        }
    }

    /// `lookupReplacer(c)`: the replacer a stand-in names.
    fn lookup_replacer(&self, c: i32) -> Option<(usize, &Var)> {
        let i = self.index_of(c)?;
        match &self.variables[i] {
            v @ (Var::Matcher { .. } | Var::Function { .. }) => Some((i, v)),
            _ => None,
        }
    }

    /// `UnicodeMatcher.matches(text, offset, limit, false)` of variable
    /// `var` (index `idx` when it is one of the variables).
    pub fn var_matches(
        &self,
        idx: Option<usize>,
        var: &Var,
        text: &[u16],
        offset: &mut i32,
        limit: i32,
        state: &mut MatchState,
    ) -> i32 {
        match var {
            Var::Set(set) => set_matches(set, text, offset, limit),
            Var::Matcher { pattern, .. } => {
                self.string_matches(pattern, idx, text, offset, limit, state)
            }
            Var::Quantifier { pattern, min, max } => {
                let start = *offset;
                let mut count = 0;
                while count < *max {
                    let pos = *offset;
                    let m = self.string_matches(pattern, None, text, offset, limit, state);
                    if m == U_MATCH {
                        count += 1;
                        if pos == *offset {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                if count >= *min {
                    return U_MATCH;
                }
                *offset = start;
                U_MISMATCH
            }
            _ => U_MISMATCH,
        }
    }

    /// `StringMatcher.matches`.
    pub fn string_matches(
        &self,
        pattern: &[u16],
        idx: Option<usize>,
        text: &[u16],
        offset: &mut i32,
        limit: i32,
        state: &mut MatchState,
    ) -> i32 {
        let mut cursor = *offset;
        let at = |i: i32| -> i32 {
            usize::try_from(i)
                .ok()
                .and_then(|i| text.get(i))
                .map_or(-1, |&u| i32::from(u))
        };
        if limit < cursor {
            // Match in the reverse direction.
            for &key in pattern.iter().rev() {
                let key = i32::from(key);
                match self.lookup_matcher(key) {
                    None => {
                        if cursor > limit && key == at(cursor) {
                            cursor -= 1;
                        } else {
                            return U_MISMATCH;
                        }
                    }
                    Some((i, m)) => {
                        let m = self.var_matches(Some(i), m, text, &mut cursor, limit, state);
                        if m != U_MATCH {
                            return m;
                        }
                    }
                }
            }
            // Record the match the first time only (a quantified segment
            // in an ante context keeps its leftmost).
            if state.get(idx).0 < 0 {
                state.set(idx, (cursor + 1, *offset + 1));
            }
        } else {
            for &key in pattern {
                let key = i32::from(key);
                match self.lookup_matcher(key) {
                    None => {
                        if cursor < limit && key == at(cursor) {
                            cursor += 1;
                        } else {
                            return U_MISMATCH;
                        }
                    }
                    Some((i, m)) => {
                        let m = self.var_matches(Some(i), m, text, &mut cursor, limit, state);
                        if m != U_MATCH {
                            return m;
                        }
                    }
                }
            }
            state.set(idx, (*offset, cursor));
        }
        *offset = cursor;
        U_MATCH
    }

    /// `UnicodeMatcher.matchesIndexValue(v)` of a variable.
    fn var_matches_index_value(&self, var: &Var, v: i32) -> bool {
        match var {
            Var::Set(set) => set.matches_index_value(v),
            Var::Matcher { pattern, .. } => self.string_matches_index_value(pattern, v),
            Var::Quantifier { pattern, min, .. } => {
                *min == 0 || self.string_matches_index_value(pattern, v)
            }
            _ => false,
        }
    }

    fn string_matches_index_value(&self, pattern: &[u16], v: i32) -> bool {
        if pattern.is_empty() {
            return true;
        }
        let c = utf16::code_point_at(pattern, 0);
        match self.lookup_matcher(c) {
            None => (c & 0xff) == v,
            Some((_, m)) => self.var_matches_index_value(m, v),
        }
    }

    /// `UnicodeMatcher.addMatchSetTo(set)` of a variable.
    fn var_add_match_set_to(&self, var: &Var, to: &mut UnicodeSet) {
        match var {
            Var::Set(set) => to.add_all(set),
            Var::Matcher { pattern, .. } => self.string_add_match_set_to(pattern, to),
            Var::Quantifier { pattern, max, .. } if *max > 0 => {
                self.string_add_match_set_to(pattern, to);
            }
            _ => {}
        }
    }

    fn string_add_match_set_to(&self, pattern: &[u16], to: &mut UnicodeSet) {
        let mut i = 0;
        while i < pattern.len() {
            let ch = utf16::code_point_at(pattern, i);
            match self.lookup_matcher(ch) {
                None => to.add(ch as u32),
                Some((_, m)) => self.var_add_match_set_to(m, to),
            }
            i += char_count(ch);
        }
    }

    /// `UnicodeReplacer.replace(text, start, limit, cursor)` of a variable.
    #[allow(clippy::too_many_arguments)]
    fn var_replace(
        &self,
        idx: usize,
        var: &Var,
        text: &mut Vec<u16>,
        start: i32,
        limit: i32,
        cursor: &mut i32,
        state: &mut MatchState,
    ) -> Result<i32, IcuError> {
        match var {
            Var::Matcher { .. } => {
                // StringMatcher.replace: copy the segment's last match.
                let (ms, ml) = state.get(Some(idx));
                let mut out_len = 0;
                let dest = limit;
                if ms >= 0 && ms != ml {
                    copy_text(text, ms, ml, dest);
                    out_len = ml - ms;
                }
                replace_text(text, start, limit, &[]);
                Ok(out_len)
            }
            Var::Function { translit, replacer } => {
                let len = replacer.replace(self, text, start, limit, cursor, state)?;
                let limit = start + len;
                let limit = translit.transliterate_range(text, start, limit)?;
                Ok(limit - start)
            }
            _ => Ok(0),
        }
    }
}

/// `UnicodeSet.matches(text, offset, limit, false)`.
// ARITH: positions within the text.
#[allow(clippy::arithmetic_side_effects)]
fn set_matches(set: &UnicodeSet, text: &[u16], offset: &mut i32, limit: i32) -> i32 {
    if *offset == limit {
        return if set.contains(ETHER) {
            U_MATCH
        } else {
            U_MISMATCH
        };
    }
    if set.has_strings() {
        let forward = *offset < limit;
        let first = usize::try_from(*offset)
            .ok()
            .and_then(|i| text.get(i))
            .copied()
            .unwrap_or(0);
        let mut high_water = 0usize;
        for trial in set.strings() {
            let Some(&c) = (if forward { trial.first() } else { trial.last() }) else {
                continue;
            };
            if forward && c > first {
                break;
            }
            if c != first {
                continue;
            }
            let length = match_rest(text, *offset, limit, trial);
            if length == trial.len() {
                if length > high_water {
                    high_water = length;
                }
                if forward && length < high_water {
                    break;
                }
            }
        }
        if high_water != 0 {
            let hw = high_water as i32;
            *offset += if forward { hw } else { -hw };
            return U_MATCH;
        }
    }
    // UnicodeFilter.matches
    if *offset < limit {
        let c = char32_at(text, *offset);
        if set.contains(c) {
            *offset += char_count(c) as i32;
            return U_MATCH;
        }
    }
    if *offset > limit {
        let c = char32_at(text, *offset);
        if set.contains(c) {
            *offset -= 1;
            if *offset >= 0 {
                *offset -= char_count(char32_at(text, *offset)) as i32 - 1;
            }
            return U_MATCH;
        }
    }
    U_MISMATCH
}

/// `UnicodeSet.matchRest(text, start, limit, s)`.
// ARITH: positions within the text and the string.
#[allow(clippy::arithmetic_side_effects)]
fn match_rest(text: &[u16], start: i32, limit: i32, s: &[u16]) -> usize {
    let slen = s.len() as i32;
    let at = |i: i32| usize::try_from(i).ok().and_then(|i| text.get(i)).copied();
    let max_len;
    if start < limit {
        max_len = (limit - start).min(slen);
        for i in 1..max_len {
            if at(start + i) != Some(s[i as usize]) {
                return 0;
            }
        }
    } else {
        max_len = (start - limit).min(slen);
        let last = slen - 1;
        for i in 1..max_len {
            if at(start - i) != Some(s[(last - i) as usize]) {
                return 0;
            }
        }
    }
    max_len.max(0) as usize
}

/// `StringReplacer`.
#[derive(Debug, Clone)]
pub struct StringReplacer {
    pub output: Vec<u16>,
    pub cursor_pos: i32,
    pub has_cursor: bool,
}

// ARITH: (the whole impl) positions within the text (Java's int
// arithmetic); the temporary buffer lives past the text's end.
#[allow(clippy::arithmetic_side_effects)]
impl StringReplacer {
    /// `replace(text, start, limit, cursor)`.
    pub fn replace(
        &self,
        data: &Data,
        text: &mut Vec<u16>,
        start: i32,
        limit: i32,
        cursor: &mut i32,
        state: &mut MatchState,
    ) -> Result<i32, IcuError> {
        let output = &self.output;
        let complex = output
            .iter()
            .any(|&u| data.lookup_replacer(i32::from(u)).is_some());
        let out_len;
        let mut new_start = 0;
        if !complex {
            // Java takes this path from the second call on (`isComplex`
            // drops after a first call finds no replacer): the same text.
            replace_text(text, start, limit, output);
            out_len = output.len() as i32;
            new_start = self.cursor_pos;
        } else {
            let mut buf: Vec<u16> = Vec::new();
            let temp_start = text.len() as i32;
            let mut dest_start = temp_start;
            if start > 0 {
                let len = char_count(char32_at(text, start - 1)) as i32;
                copy_text(text, start - len, start, temp_start);
                dest_start += len;
            } else {
                replace_text(text, temp_start, temp_start, &[0xffff]);
                dest_start += 1;
            }
            let mut dest_limit = dest_start;
            let mut temp_extra = 0;
            let mut o = 0usize;
            while o < output.len() {
                if o as i32 == self.cursor_pos {
                    new_start = buf.len() as i32 + dest_limit - dest_start;
                }
                let c = utf16::code_point_at(output, o);
                let next = o + char_count(c);
                if next == output.len() {
                    temp_extra = char_count(char32_at(text, limit)) as i32;
                    copy_text(text, limit, limit + temp_extra, dest_limit);
                }
                match data.lookup_replacer(c) {
                    None => utf16::push_code_point(&mut buf, c),
                    Some((i, r)) => {
                        if !buf.is_empty() {
                            replace_text(text, dest_limit, dest_limit, &buf);
                            dest_limit += buf.len() as i32;
                            buf.clear();
                        }
                        let len =
                            data.var_replace(i, r, text, dest_limit, dest_limit, cursor, state)?;
                        dest_limit += len;
                    }
                }
                o = next;
            }
            if !buf.is_empty() {
                replace_text(text, dest_limit, dest_limit, &buf);
                dest_limit += buf.len() as i32;
            }
            if o as i32 == self.cursor_pos {
                new_start = dest_limit - dest_start;
            }
            out_len = dest_limit - dest_start;
            copy_text(text, dest_start, dest_limit, start);
            replace_text(
                text,
                temp_start + out_len,
                dest_limit + temp_extra + out_len,
                &[],
            );
            replace_text(text, start + out_len, limit + out_len, &[]);
        }
        if self.has_cursor {
            if self.cursor_pos < 0 {
                new_start = start;
                let mut n = self.cursor_pos;
                while n < 0 && new_start > 0 {
                    new_start -= char_count(char32_at(text, new_start - 1)) as i32;
                    n += 1;
                }
                new_start += n;
            } else if self.cursor_pos > output.len() as i32 {
                new_start = start + out_len;
                let mut n = self.cursor_pos - output.len() as i32;
                while n > 0 && new_start < text.len() as i32 {
                    new_start += char_count(char32_at(text, new_start)) as i32;
                    n -= 1;
                }
                new_start += n;
            } else {
                new_start += start;
            }
            *cursor = new_start;
        }
        Ok(out_len)
    }
}

/// `TransliterationRule`.
#[derive(Debug, Clone)]
pub struct Rule {
    ante_context: Option<Vec<u16>>,
    key: Option<Vec<u16>>,
    post_context: Option<Vec<u16>>,
    output: StringReplacer,
    pattern: Vec<u16>,
    segments: Vec<usize>,
    ante_context_length: usize,
    key_length: usize,
    flags: u8,
}

const ANCHOR_START: u8 = 1;
const ANCHOR_END: u8 = 2;

// ARITH: (the whole impl) lengths of the rule's pattern and positions in
// the text (Java's int arithmetic).
#[allow(clippy::arithmetic_side_effects)]
impl Rule {
    /// `TransliterationRule(input, anteContextPos, postContextPos, output,
    /// cursorPos, cursorOffset, segs, anchorStart, anchorEnd, data)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: Vec<u16>,
        ante_context_pos: i32,
        post_context_pos: i32,
        output: Vec<u16>,
        cursor_pos: i32,
        cursor_offset: i32,
        segments: Vec<usize>,
        anchor_start: bool,
        anchor_end: bool,
    ) -> Result<Rule, IcuError> {
        let len = input.len() as i32;
        let ante = if ante_context_pos < 0 {
            0
        } else {
            if ante_context_pos > len {
                return Err(IcuError::illegal_argument("Invalid ante context"));
            }
            ante_context_pos
        };
        let key_length = if post_context_pos < 0 {
            len - ante
        } else {
            if post_context_pos < ante || post_context_pos > len {
                return Err(IcuError::illegal_argument("Invalid post context"));
            }
            post_context_pos - ante
        };
        let out_len = output.len() as i32;
        let cursor_pos = if cursor_pos < 0 {
            out_len
        } else if cursor_pos > out_len {
            return Err(IcuError::illegal_argument("Invalid cursor position"));
        } else {
            cursor_pos
        };
        let (ante, key_length) = (ante as usize, key_length as usize);
        let post_len = input.len() - key_length - ante;
        Ok(Rule {
            ante_context: (ante > 0).then(|| input[..ante].to_vec()),
            key: (key_length > 0).then(|| input[ante..ante + key_length].to_vec()),
            post_context: (post_len > 0).then(|| input[ante + key_length..].to_vec()),
            output: StringReplacer {
                output,
                cursor_pos: cursor_pos + cursor_offset,
                has_cursor: true,
            },
            pattern: input,
            segments,
            ante_context_length: ante,
            key_length,
            flags: u8::from(anchor_start) * ANCHOR_START + u8::from(anchor_end) * ANCHOR_END,
        })
    }

    /// `getAnteContextLength()`.
    fn ante_context_length_with_anchor(&self) -> usize {
        self.ante_context_length + usize::from(self.flags & ANCHOR_START != 0)
    }

    /// `getIndexValue()`.
    // SENTINEL: `-1` = the key starts with a matcher, so the rule is indexed
    // under every byte its matcher can start with.
    fn index_value(&self, data: &Data) -> i32 {
        if self.ante_context_length == self.pattern.len() {
            return -1;
        }
        let c = utf16::code_point_at(&self.pattern, self.ante_context_length);
        if data.lookup_matcher(c).is_none() {
            c & 0xff
        } else {
            -1
        }
    }

    /// `matchesIndexValue(v)`.
    fn matches_index_value(&self, data: &Data, v: i32) -> bool {
        match self.key.as_ref().or(self.post_context.as_ref()) {
            Some(m) => data.string_matches_index_value(m, v),
            None => true,
        }
    }

    /// `masks(r2)`.
    fn masks(&self, r2: &Rule) -> bool {
        let len = self.pattern.len();
        let left = self.ante_context_length;
        let left2 = r2.ante_context_length;
        let right = len - left;
        let right2 = r2.pattern.len() - left2;
        if left == left2
            && right == right2
            && self.key_length <= r2.key_length
            && r2.pattern.get(..len) == Some(&self.pattern[..])
        {
            return self.flags == r2.flags
                || (self.flags & ANCHOR_START == 0 && self.flags & ANCHOR_END == 0)
                || (r2.flags & ANCHOR_START != 0 && r2.flags & ANCHOR_END != 0);
        }
        left <= left2
            && (right < right2 || (right == right2 && self.key_length <= r2.key_length))
            && r2.pattern.get(left2 - left..left2 - left + len) == Some(&self.pattern[..])
    }

    /// `matchAndReplace(text, pos, false)`.
    fn match_and_replace(
        &self,
        data: &Data,
        text: &mut Vec<u16>,
        pos: &mut Position,
        state: &mut MatchState,
    ) -> Result<i32, IcuError> {
        for &s in &self.segments {
            state.set(Some(s), (-1, -1));
        }
        let ante_limit = pos_before(text, pos.context_start);
        let mut o = pos_before(text, pos.start);
        if let Some(ante) = &self.ante_context {
            if data.string_matches(ante, None, text, &mut o, ante_limit, state) != U_MATCH {
                return Ok(U_MISMATCH);
            }
        }
        let min_o_text = pos_after(text, o);
        if self.flags & ANCHOR_START != 0 && o != ante_limit {
            return Ok(U_MISMATCH);
        }
        let mut cur = pos.start;
        if let Some(key) = &self.key {
            let m = data.string_matches(key, None, text, &mut cur, pos.limit, state);
            if m != U_MATCH {
                return Ok(m);
            }
        }
        let key_limit = cur;
        if let Some(post) = &self.post_context {
            let m = data.string_matches(post, None, text, &mut cur, pos.context_limit, state);
            if m != U_MATCH {
                return Ok(m);
            }
        }
        let mut o_text = cur;
        if self.flags & ANCHOR_END != 0 && o_text != pos.context_limit {
            return Ok(U_MISMATCH);
        }
        let mut new_start = 0;
        let new_length =
            self.output
                .replace(data, text, pos.start, key_limit, &mut new_start, state)?;
        let len_delta = new_length - (key_limit - pos.start);
        o_text += len_delta;
        pos.limit += len_delta;
        pos.context_limit += len_delta;
        pos.start = min_o_text.max(o_text.min(pos.limit).min(new_start));
        Ok(U_MATCH)
    }

    /// The source characters this rule's key can match, if every key
    /// element can match within `filter` (`addSourceTargetSet`'s source half).
    fn add_source_set(&self, data: &Data, filter: &UnicodeSet, source: &mut UnicodeSet) {
        let limit = self.ante_context_length + self.key_length;
        let mut temp_source = UnicodeSet::new();
        let mut i = self.ante_context_length;
        while i < limit {
            let ch = utf16::code_point_at(&self.pattern, i);
            i += char_count(ch);
            match data.lookup_matcher(ch) {
                None => {
                    if !filter.contains(ch) {
                        return;
                    }
                    temp_source.add(ch as u32);
                }
                Some((_, Var::Set(set))) => {
                    if !filter.contains_some(set) {
                        return;
                    }
                    temp_source.add_all(set);
                }
                Some((_, m)) => {
                    let mut temp = UnicodeSet::new();
                    data.var_add_match_set_to(m, &mut temp);
                    if !filter.contains_some(&temp) {
                        return;
                    }
                    temp_source.add_all(&temp);
                }
            }
        }
        source.add_all(&temp_source);
    }
}

/// `TransliterationRule.posBefore(str, pos)`.
// ARITH: positions within the text.
#[allow(clippy::arithmetic_side_effects)]
fn pos_before(text: &[u16], pos: i32) -> i32 {
    if pos > 0 {
        pos - char_count(char32_at(text, pos - 1)) as i32
    } else {
        pos - 1
    }
}

/// `TransliterationRule.posAfter(str, pos)`.
// ARITH: positions within the text.
#[allow(clippy::arithmetic_side_effects)]
fn pos_after(text: &[u16], pos: i32) -> i32 {
    if pos >= 0 && (pos as usize) < text.len() {
        pos + char_count(char32_at(text, pos)) as i32
    } else {
        pos + 1
    }
}

/// `TransliterationRuleSet`.
#[derive(Debug, Clone, Default)]
pub struct RuleSet {
    rule_vector: Vec<Rule>,
    max_context_length: usize,
    /// The rules for each low byte of the first key character (`index`,
    /// `rules`), filled by `freeze`.
    index: Vec<usize>,
    rules: Vec<usize>,
}

// ARITH: (the whole impl) counts of rules and positions in the text.
#[allow(clippy::arithmetic_side_effects)]
impl RuleSet {
    /// `getMaximumContextLength()`.
    pub fn maximum_context_length(&self) -> usize {
        self.max_context_length
    }

    /// `addRule(rule)`.
    pub fn add_rule(&mut self, rule: Rule) {
        let len = rule.ante_context_length_with_anchor();
        if len > self.max_context_length {
            self.max_context_length = len;
        }
        self.rule_vector.push(rule);
    }

    /// `freeze()`: index the rules by their first key character's low byte
    /// and refuse a rule that masks a later one.
    pub fn freeze(&mut self, data: &Data) -> Result<(), IcuError> {
        let n = self.rule_vector.len();
        let index_value: Vec<i32> = self
            .rule_vector
            .iter()
            .map(|r| r.index_value(data))
            .collect();
        let mut index = Vec::with_capacity(257);
        let mut v: Vec<usize> = Vec::with_capacity(2 * n);
        for x in 0..256 {
            index.push(v.len());
            for (j, &value) in index_value.iter().enumerate() {
                if value >= 0 {
                    if value == x {
                        v.push(j);
                    }
                } else if self.rule_vector[j].matches_index_value(data, x) {
                    v.push(j);
                }
            }
        }
        index.push(v.len());
        let mut errors = Vec::new();
        for x in 0..256 {
            let (a, b) = (index[x], index[x + 1]);
            for j in a..b.saturating_sub(1) {
                let r1 = &self.rule_vector[v[j]];
                for &k in &v[j + 1..b] {
                    if r1.masks(&self.rule_vector[k]) {
                        errors.push(format!("Rule {} masks {}", v[j], k));
                    }
                }
            }
        }
        self.index = index;
        self.rules = v;
        if !errors.is_empty() {
            return Err(IcuError::illegal_argument(errors.join("\n")));
        }
        Ok(())
    }

    /// `transliterate(text, pos, false)`.
    pub fn transliterate(
        &self,
        data: &Data,
        text: &mut Vec<u16>,
        pos: &mut Position,
        state: &mut MatchState,
    ) -> Result<(), IcuError> {
        let c = char32_at(text, pos.start);
        let index_byte = (c & 0xff) as usize;
        let (a, b) = (
            self.index.get(index_byte).copied().unwrap_or(0),
            self.index.get(index_byte + 1).copied().unwrap_or(0),
        );
        for &r in &self.rules[a.min(b)..b] {
            if self.rule_vector[r].match_and_replace(data, text, pos, state)? == U_MATCH {
                return Ok(());
            }
        }
        pos.start += char_count(c) as i32;
        Ok(())
    }

    /// `addSourceTargetSet`'s source half.
    pub fn add_source_set(&self, data: &Data, filter: &UnicodeSet, source: &mut UnicodeSet) {
        for r in &self.rule_vector {
            r.add_source_set(data, filter, source);
        }
    }
}

/// `RuleBasedTransliterator.handleTransliterate(text, index, false)`.
// ARITH: positions within the text; the loop limit as Java computes it.
#[allow(clippy::arithmetic_side_effects)]
pub fn handle_rule_based(
    data: &Data,
    text: &mut Vec<u16>,
    index: &mut Position,
) -> Result<(), IcuError> {
    let mut state = MatchState::new(data.variables.len());
    let mut loop_count = 0i32;
    // Java: (limit - start) << 4 in int arithmetic.
    let mut loop_limit = (index.limit - index.start).wrapping_shl(4);
    if loop_limit < 0 {
        loop_limit = i32::MAX;
    }
    while index.start < index.limit && loop_count <= loop_limit {
        data.rule_set.transliterate(data, text, index, &mut state)?;
        loop_count = loop_count.saturating_add(1);
    }
    Ok(())
}
