//! `com.ibm.icu.text.UnicodeSet`: a set of code points (and strings), and
//! its pattern syntax -- `[a-z]`, `[^...]`, nested sets with `&` and `-`,
//! `[:Latin:]`, `\p{gc=Lu}`, `\P{...}`, escapes and `{strings}` -- as
//! `UnicodeSet(String)` parses it (`applyPattern` with `IGNORE_SPACE`, over
//! `RuleCharacterIterator` and `Utility.unescapeAndLengthAt`).
//!
//! Properties resolve through [`uprops`](mod@crate::icu4j::uprops) as `applyPropertyAlias` does: a
//! lone name is a General_Category (mask) value, else a Script value, else
//! a binary property, else `ANY`, `ASCII` or `Assigned`; `name=value` takes
//! any binary, enumerated or mask property, `Script_Extensions`, and a
//! numeric `ccc`. **Not ported** (typed `UnsupportedOperation`): `\N{name}`
//! and `na=` (character names), `Numeric_Value`, `Age`, `Identifier_Type`,
//! and string ranges `{ab}-{ad}`; variables (`$x`) exist only with a
//! symbol table, which no Lucene caller passes, so `$` is a literal or the
//! `[...$]` anchor (U+FFFF), as in Java.
//!
//! The code point set is a sorted list of disjoint inclusive ranges.

use std::collections::BTreeSet;

use crate::icu4j::uprops::{self, uprops, PropKind};
use crate::icu4j::utf16;
use crate::IcuError;

/// `UnicodeSet.SpanCondition`, for code-point-only sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanCondition {
    /// `NOT_CONTAINED`: span while the set does not contain the code point.
    NotContained,
    /// `SIMPLE` (and `CONTAINED`): span while it does.
    Simple,
}

/// A set of code points and strings.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UnicodeSet {
    ranges: Vec<(u32, u32)>,
    strings: BTreeSet<Vec<u16>>,
}

const MAX_VALUE: u32 = 0x10ffff;
const MAX_DEPTH: usize = 100;

impl UnicodeSet {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// `new UnicodeSet(pattern)`.
    pub fn from_pattern(pattern: &str) -> Result<UnicodeSet, IcuError> {
        let text = utf16::units(pattern);
        let mut chars = Chars::new(&text, 0, None);
        let mut set = UnicodeSet::new();
        set.apply_pattern(&mut chars, 0)?;
        // Skip trailing whitespace (IGNORE_SPACE), then require the end.
        chars.skip_ignored();
        if chars.pos != text.len() {
            return Err(IcuError::illegal_argument(format!(
                "Parse of \"{pattern}\" failed at {}",
                chars.pos
            )));
        }
        Ok(set)
    }

    /// `new UnicodeSet(pattern, pos, symbols)` (`IGNORE_SPACE`): a set
    /// parsed from `text` at `*pos`, which moves past it (and any white
    /// space after it).
    pub fn parse_at(
        text: &[u16],
        pos: &mut usize,
        symbols: Option<&dyn SymbolTable>,
    ) -> Result<UnicodeSet, IcuError> {
        let mut chars = Chars::new(text, *pos, symbols);
        let mut set = UnicodeSet::new();
        set.apply_pattern(&mut chars, 0)?;
        if chars.in_variable() {
            return Err(chars.syntax_error("Extra chars in variable value"));
        }
        *pos = chars.pos;
        Ok(set)
    }

    /// `size()`: code points plus strings.
    pub fn size(&self) -> usize {
        self.ranges
            .iter()
            .map(|&(a, b)| (b.saturating_sub(a) as usize).saturating_add(1))
            .sum::<usize>()
            .saturating_add(self.strings.len())
    }

    /// `containsNone(set)`: no code point and no string in common.
    pub fn contains_none(&self, other: &UnicodeSet) -> bool {
        let (mut i, mut j) = (0, 0);
        while i < self.ranges.len() && j < other.ranges.len() {
            let (a, b) = self.ranges[i];
            let (c, d) = other.ranges[j];
            if b < c {
                i = i.saturating_add(1);
            } else if d < a {
                j = j.saturating_add(1);
            } else {
                return false;
            }
        }
        self.strings.is_disjoint(&other.strings)
    }

    /// `containsSome(set)`.
    pub fn contains_some(&self, other: &UnicodeSet) -> bool {
        !self.contains_none(other)
    }

    /// `addAll(CharSequence)`: each code point of `s`.
    pub fn add_all_code_points(&mut self, s: &[u16]) {
        let mut i = 0;
        while i < s.len() {
            let c = utf16::code_point_at(s, i);
            self.add(c as u32);
            i = i.saturating_add(if c > 0xffff { 2 } else { 1 });
        }
    }

    /// `matchesIndexValue(v)`: whether a code point or a string's first
    /// code point of the set has `v` as its low byte (Java's test of a
    /// range spanning a 256 block, kept as is).
    pub fn matches_index_value(&self, v: i32) -> bool {
        for &(low, high) in &self.ranges {
            let (low, high) = (low as i32, high as i32);
            if (low & !0xff) == (high & !0xff) {
                if (low & 0xff) <= v && v <= (high & 0xff) {
                    return true;
                }
            } else if (low & 0xff) <= v || v <= (high & 0xff) {
                return true;
            }
        }
        self.strings
            .iter()
            .any(|s| !s.is_empty() && (utf16::code_point_at(s, 0) & 0xff) == v)
    }

    /// `toPattern(false)`: a pattern that parses back to this set.
    pub fn to_pattern(&self) -> String {
        let mut out = String::from("[");
        let esc = |out: &mut String, c: u32| {
            let ch = char::from_u32(c);
            match ch {
                Some(ch) if ch.is_ascii_alphanumeric() => out.push(ch),
                _ if c <= 0xffff => out.push_str(&format!("\\u{c:04X}")),
                _ => out.push_str(&format!("\\U{c:08X}")),
            }
        };
        for &(a, b) in &self.ranges {
            esc(&mut out, a);
            if b != a {
                out.push('-');
                esc(&mut out, b);
            }
        }
        for s in &self.strings {
            out.push('{');
            let mut i = 0;
            while i < s.len() {
                let c = utf16::code_point_at(s, i);
                esc(&mut out, c as u32);
                i = i.saturating_add(if c > 0xffff { 2 } else { 1 });
            }
            out.push('}');
        }
        out.push(']');
        out
    }

    /// `UnicodeSet.resemblesPattern(pattern, pos)`.
    pub fn resembles_pattern(text: &[u16], pos: usize) -> bool {
        (pos.saturating_add(1) < text.len() && text[pos] == '[' as u16)
            || resembles_property_pattern_at(text, pos)
    }

    /// The set `[start-end]`.
    pub fn from_range(start: u32, end: u32) -> UnicodeSet {
        let mut s = UnicodeSet::new();
        s.add_range(start, end);
        s
    }

    /// The code point ranges, sorted and disjoint.
    pub fn ranges(&self) -> &[(u32, u32)] {
        &self.ranges
    }

    /// The multi-character strings, sorted.
    pub fn strings(&self) -> impl Iterator<Item = &[u16]> {
        self.strings.iter().map(Vec::as_slice)
    }

    /// Whether the set holds multi-character strings.
    pub fn has_strings(&self) -> bool {
        !self.strings.is_empty()
    }

    /// `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty() && self.strings.is_empty()
    }

    /// `clear()`.
    pub fn clear(&mut self) {
        self.ranges.clear();
        self.strings.clear();
    }

    /// `contains(c)`.
    #[inline]
    pub fn contains(&self, c: i32) -> bool {
        let Ok(c) = u32::try_from(c) else {
            return false;
        };
        let i = self.ranges.partition_point(|&(_, e)| e < c);
        self.ranges.get(i).is_some_and(|&(s, _)| s <= c)
    }

    /// `add(start, end)`.
    pub fn add_range(&mut self, start: u32, end: u32) {
        if start > end || start > MAX_VALUE {
            return;
        }
        let end = end.min(MAX_VALUE);
        let mut out = Vec::with_capacity(self.ranges.len().saturating_add(1));
        let (mut s, mut e) = (start, end);
        let mut placed = false;
        for &(a, b) in &self.ranges {
            if b.saturating_add(1) < s {
                out.push((a, b));
            } else if e.saturating_add(1) < a {
                if !placed {
                    out.push((s, e));
                    placed = true;
                }
                out.push((a, b));
            } else {
                s = s.min(a);
                e = e.max(b);
            }
        }
        if !placed {
            out.push((s, e));
        }
        out.sort_unstable();
        self.ranges = out;
    }

    /// `add(c)`.
    pub fn add(&mut self, c: u32) {
        self.add_range(c, c);
    }

    /// `add(CharSequence)`: a single code point joins the ranges, anything
    /// else the strings.
    pub fn add_string(&mut self, s: &[u16]) {
        let c = utf16::code_point_at(s, 0);
        if !s.is_empty() && s.len() == if c > 0xffff { 2 } else { 1 } {
            self.add(c as u32);
        } else {
            self.strings.insert(s.to_vec());
        }
    }

    /// `complement()` of the code points (strings are kept).
    pub fn complement(&mut self) {
        let mut out = Vec::with_capacity(self.ranges.len().saturating_add(1));
        let mut next = 0u32;
        for &(a, b) in &self.ranges {
            if a > next {
                out.push((next, a.saturating_sub(1)));
            }
            next = b.saturating_add(1);
        }
        if next <= MAX_VALUE {
            out.push((next, MAX_VALUE));
        }
        self.ranges = out;
    }

    /// `removeAllStrings()`.
    pub fn remove_all_strings(&mut self) {
        self.strings.clear();
    }

    /// `addAll(set)`.
    pub fn add_all(&mut self, other: &UnicodeSet) {
        // Merge the two sorted range lists, coalescing adjacent ranges.
        let mut out: Vec<(u32, u32)> =
            Vec::with_capacity(self.ranges.len().saturating_add(other.ranges.len()));
        let (mut i, mut j) = (0usize, 0usize);
        loop {
            let next = match (self.ranges.get(i), other.ranges.get(j)) {
                (Some(&a), Some(&b)) => {
                    if a.0 <= b.0 {
                        i = i.saturating_add(1);
                        a
                    } else {
                        j = j.saturating_add(1);
                        b
                    }
                }
                (Some(&a), None) => {
                    i = i.saturating_add(1);
                    a
                }
                (None, Some(&b)) => {
                    j = j.saturating_add(1);
                    b
                }
                (None, None) => break,
            };
            match out.last_mut() {
                Some(last) if next.0 <= last.1.saturating_add(1) => last.1 = last.1.max(next.1),
                _ => out.push(next),
            }
        }
        self.ranges = out;
        self.strings.extend(other.strings.iter().cloned());
    }

    /// `retainAll(set)`.
    pub fn retain_all(&mut self, other: &UnicodeSet) {
        let mut out = Vec::new();
        let (mut i, mut j) = (0, 0);
        while let (Some(&(a, b)), Some(&(c, d))) = (self.ranges.get(i), other.ranges.get(j)) {
            let s = a.max(c);
            let e = b.min(d);
            if s <= e {
                out.push((s, e));
            }
            if b < d {
                i = i.wrapping_add(1);
            } else {
                j = j.wrapping_add(1);
            }
        }
        self.ranges = out;
        self.strings.retain(|s| other.strings.contains(s));
    }

    /// `removeAll(set)`.
    pub fn remove_all(&mut self, other: &UnicodeSet) {
        let mut inv = UnicodeSet {
            ranges: other.ranges.clone(),
            strings: BTreeSet::new(),
        };
        inv.complement();
        let strings = std::mem::take(&mut self.strings);
        self.retain_all(&inv);
        self.strings = strings
            .into_iter()
            .filter(|s| !other.strings.contains(s))
            .collect();
    }

    /// `span(s, start, spanCondition)` for a set without strings: the end of
    /// the run of code points from `start` that are (`Simple`) or are not
    /// (`NotContained`) in the set.
    pub fn span(&self, s: &[u16], start: usize, cond: SpanCondition) -> usize {
        let want = cond == SpanCondition::Simple;
        let mut i = start;
        while i < s.len() {
            let c = utf16::code_point_at(s, i);
            if self.contains(c) != want {
                break;
            }
            i = i.saturating_add(if c > 0xffff { 2 } else { 1 });
        }
        i
    }

    /// `spanBack(s, limit, spanCondition)` for a set without strings.
    pub fn span_back(&self, s: &[u16], limit: usize, cond: SpanCondition) -> usize {
        let want = cond == SpanCondition::Simple;
        let mut i = limit.min(s.len());
        while i > 0 {
            let c = utf16::code_point_before(s, i);
            if self.contains(c) != want {
                break;
            }
            i = i.saturating_sub(if c > 0xffff { 2 } else { 1 });
        }
        i
    }

    /// `applyIntPropertyValue(prop, value)`.
    pub fn apply_int_property_value(&mut self, prop: i32, value: i32) -> Result<(), IcuError> {
        self.clear();
        let p = uprops();
        if prop == uprops::GENERAL_CATEGORY_MASK {
            let gc = p
                .property(uprops::GENERAL_CATEGORY)
                .ok_or_else(|| IcuError::new("no General_Category data"))?;
            for (s, e, v) in gc.runs() {
                if (0..32).contains(&v) && (1i32.wrapping_shl(v as u32) & value) != 0 {
                    self.add_range(s, e);
                }
            }
        } else if prop == uprops::SCRIPT_EXTENSIONS {
            let scx = p
                .property(uprops::SCRIPT_EXTENSIONS)
                .ok_or_else(|| IcuError::new("no Script_Extensions data"))?;
            let lists = p.script_lists();
            for (s, e, v) in scx.runs() {
                let has = usize::try_from(v)
                    .ok()
                    .and_then(|i| lists.get(i))
                    .is_some_and(|l| l.iter().any(|&x| i32::from(x) == value));
                if has {
                    self.add_range(s, e);
                }
            }
        } else {
            let Some(pr) = p.property(prop) else {
                return Err(IcuError::illegal_argument(format!(
                    "unsupported property {prop}"
                )));
            };
            if pr.kind == PropKind::Binary && value != 0 && value != 1 {
                return Ok(());
            }
            for (s, e, v) in pr.runs() {
                if v == value {
                    self.add_range(s, e);
                }
            }
            if pr.kind == PropKind::Binary && value == 1 {
                self.strings.extend(pr.strings.iter().cloned());
            }
        }
        Ok(())
    }

    /// `applyPropertyAlias(propertyAlias, valueAlias, null)`.
    pub fn apply_property_alias(
        &mut self,
        prop_alias: &str,
        value_alias: &str,
    ) -> Result<(), IcuError> {
        let p = uprops();
        let (prop, value, invert);
        if !value_alias.is_empty() {
            let pr = p
                .property_by_alias(prop_alias)
                .ok_or_else(|| invalid_name(prop_alias))?;
            let mut id = pr.id;
            let pr = if id == uprops::GENERAL_CATEGORY {
                id = uprops::GENERAL_CATEGORY_MASK;
                p.property(id).unwrap_or(pr)
            } else {
                pr
            };
            match pr.kind {
                PropKind::Binary | PropKind::Enumerated | PropKind::GeneralCategoryMask => {
                    value = match p.value_by_alias(pr, value_alias) {
                        Some(v) => v,
                        None if id == uprops::CANONICAL_COMBINING_CLASS
                            || id == uprops::LEAD_CANONICAL_COMBINING_CLASS
                            || id == uprops::TRAIL_CANONICAL_COMBINING_CLASS =>
                        {
                            let v: i32 = trim_white_space(value_alias).parse().map_err(|_| {
                                IcuError::illegal_argument(format!(
                                    "NumberFormatException: For input string: \"{value_alias}\""
                                ))
                            })?;
                            if !(0..=255).contains(&v) {
                                return Err(invalid_name(value_alias));
                            }
                            v
                        }
                        None => return Err(invalid_name(value_alias)),
                    };
                }
                PropKind::Age => {
                    let version = parse_version(&munge_char_name(value_alias))?;
                    self.clear();
                    for (s, e, v) in pr.runs() {
                        if v != 0 && (v as u32) <= version {
                            self.add_range(s, e);
                        }
                    }
                    return Ok(());
                }
                PropKind::NumericValue => {
                    let want = parse_java_double(value_alias)?.to_bits();
                    let wanted: Vec<i32> = pr
                        .values
                        .iter()
                        .filter(|(_, n)| {
                            n.first().and_then(|h| u64::from_str_radix(h, 16).ok()) == Some(want)
                        })
                        .map(|&(v, _)| v)
                        .collect();
                    self.clear();
                    for (s, e, v) in pr.runs() {
                        if wanted.contains(&v) {
                            self.add_range(s, e);
                        }
                    }
                    return Ok(());
                }
                PropKind::Other => {
                    return Err(match pr.names.first().map(String::as_str) {
                        Some("na" | "ID_Type") => IcuError::unsupported(format!(
                            "UnicodeSet property {prop_alias} is not ported"
                        )),
                        _ => IcuError::illegal_argument("Unsupported property"),
                    });
                }
                PropKind::ScriptExtensions => {
                    let sc = p
                        .property(uprops::SCRIPT)
                        .ok_or_else(|| IcuError::new("no Script data"))?;
                    value = p
                        .value_by_alias(sc, value_alias)
                        .ok_or_else(|| invalid_name(value_alias))?;
                }
            }
            prop = id;
            invert = false;
        } else {
            let gcm = p
                .property(uprops::GENERAL_CATEGORY_MASK)
                .ok_or_else(|| IcuError::new("no General_Category data"))?;
            let sc = p
                .property(uprops::SCRIPT)
                .ok_or_else(|| IcuError::new("no Script data"))?;
            if let Some(v) = p.value_by_alias(gcm, prop_alias) {
                prop = uprops::GENERAL_CATEGORY_MASK;
                value = v;
                invert = false;
            } else if let Some(v) = p.value_by_alias(sc, prop_alias) {
                prop = uprops::SCRIPT;
                value = v;
                invert = false;
            } else {
                match p.property_by_alias(prop_alias) {
                    Some(pr) if pr.kind == PropKind::Binary => {
                        prop = pr.id;
                        value = 1;
                        invert = false;
                    }
                    Some(_) => return Err(IcuError::illegal_argument("Missing property value")),
                    None => {
                        if uprops::compare_names("ANY", prop_alias) {
                            self.clear();
                            self.add_range(0, MAX_VALUE);
                            return Ok(());
                        } else if uprops::compare_names("ASCII", prop_alias) {
                            self.clear();
                            self.add_range(0, 0x7f);
                            return Ok(());
                        } else if uprops::compare_names("Assigned", prop_alias) {
                            prop = uprops::GENERAL_CATEGORY_MASK;
                            value = 1 << uprops::UNASSIGNED;
                            invert = true;
                        } else {
                            return Err(IcuError::illegal_argument(format!(
                                "Invalid property alias: {prop_alias}={value_alias}"
                            )));
                        }
                    }
                }
            }
        }
        self.apply_int_property_value(prop, value)?;
        if invert {
            self.complement();
            self.remove_all_strings();
        }
        Ok(())
    }

    /// `applyPattern(chars, symbols, rebuiltPat, options, depth)`.
    // ARITH: depth is bounded by MAX_DEPTH.
    #[allow(clippy::arithmetic_side_effects)]
    fn apply_pattern(&mut self, chars: &mut Chars<'_>, depth: usize) -> Result<(), IcuError> {
        if depth > MAX_DEPTH {
            return Err(chars.syntax_error("Pattern nested too deeply"));
        }
        const START: u8 = 0;
        const RANGE: u8 = 1;
        const SET: u8 = 2;
        let mut last_item = START;
        let mut last_char = 0i32;
        let mut in_bracket = false;
        let mut out_bracket = false;
        let mut op = 0u8;
        let mut invert = false;
        let mut last_string: Option<Vec<u16>> = None;
        self.clear();
        while !out_bracket && !chars.at_end() {
            let mut c = 0i32;
            let mut literal = false;
            let mut nested_mode = 0u8; // 1 nested set, 2 property pattern
            let mut preparsed: Option<UnicodeSet> = None;
            if chars.resembles_property_pattern()? {
                nested_mode = 2;
            } else {
                let mut backup = chars.get_pos();
                c = chars.next()?;
                literal = chars.escaped;
                if c == '[' as i32 && !literal {
                    if in_bracket {
                        chars.set_pos(backup);
                        nested_mode = 1;
                    } else {
                        in_bracket = true;
                        backup = chars.get_pos();
                        c = chars.next()?;
                        literal = chars.escaped;
                        if c == '^' as i32 && !literal {
                            invert = true;
                            backup = chars.get_pos();
                            c = chars.next()?;
                        }
                        if c == '-' as i32 {
                            literal = true;
                        } else {
                            chars.set_pos(backup);
                            continue;
                        }
                    }
                } else if let Some(sym) = chars.symbols {
                    match sym.lookup_matcher(c) {
                        Some(SymbolMatcher::Set(m)) => {
                            preparsed = Some(m.clone());
                            nested_mode = 3;
                        }
                        Some(SymbolMatcher::Other) => {
                            return Err(chars.syntax_error("Syntax error"))
                        }
                        None => {}
                    }
                }
            }
            if nested_mode != 0 {
                if last_item == RANGE {
                    if op != 0 {
                        return Err(chars.syntax_error("Char expected after operator"));
                    }
                    self.add(last_char as u32);
                    op = 0;
                }
                let mut nested = UnicodeSet::new();
                if nested_mode == 1 {
                    nested.apply_pattern(chars, depth + 1)?;
                } else if nested_mode == 2 {
                    chars.skip_ignored();
                    nested.apply_property_pattern_at(chars)?;
                } else if let Some(p) = preparsed {
                    nested = p;
                }
                if !in_bracket {
                    *self = nested;
                    out_bracket = true;
                    break;
                }
                match op {
                    b'-' => self.remove_all(&nested),
                    b'&' => self.retain_all(&nested),
                    _ => self.add_all(&nested),
                }
                op = 0;
                last_item = SET;
                continue;
            }
            if !in_bracket {
                return Err(chars.syntax_error("Missing '['"));
            }
            if !literal {
                match u8::try_from(c).unwrap_or(0) {
                    b']' => {
                        if last_item == RANGE {
                            self.add(last_char as u32);
                        }
                        if op == b'-' {
                            self.add(u32::from(op));
                        } else if op == b'&' {
                            return Err(chars.syntax_error("Trailing '&'"));
                        }
                        out_bracket = true;
                        continue;
                    }
                    b'-' => {
                        if op == 0 {
                            if last_item != START || last_string.is_some() {
                                op = b'-';
                                continue;
                            }
                            self.add(u32::from(b'-'));
                            c = chars.next()?;
                            literal = chars.escaped;
                            if c == ']' as i32 && !literal {
                                out_bracket = true;
                                continue;
                            }
                        }
                        return Err(chars.syntax_error("'-' not after char, string, or set"));
                    }
                    b'&' => {
                        if last_item == SET && op == 0 {
                            op = b'&';
                            continue;
                        }
                        return Err(chars.syntax_error("'&' not after set"));
                    }
                    b'^' => return Err(chars.syntax_error("'^' not after '['")),
                    b'{' => {
                        if op != 0 && op != b'-' {
                            return Err(chars.syntax_error("Missing operand after operator"));
                        }
                        if last_item == RANGE {
                            self.add(last_char as u32);
                        }
                        last_item = START;
                        let mut buf = Vec::new();
                        let mut ok = false;
                        while !chars.at_end() {
                            c = chars.next()?;
                            if c == '}' as i32 && !chars.escaped {
                                ok = true;
                                break;
                            }
                            utf16::push_code_point(&mut buf, c);
                        }
                        if !ok {
                            return Err(chars.syntax_error("Invalid multicharacter string"));
                        }
                        if op == b'-' {
                            let Some(last) = last_string.take() else {
                                // Java: StringRange.expand("", ...) fails.
                                return Err(chars.syntax_error("Invalid range"));
                            };
                            match (single_code_point(&last), single_code_point(&buf)) {
                                (Some(a), Some(b)) => self.add_range(a, b),
                                _ => {
                                    return Err(IcuError::unsupported(
                                        "UnicodeSet string ranges ({ab}-{ad}) are not ported",
                                    ))
                                }
                            }
                            op = 0;
                        } else {
                            self.add_string(&buf);
                            last_string = Some(buf);
                        }
                        continue;
                    }
                    b'$' => {
                        let backup = chars.get_pos();
                        c = chars.next()?;
                        let anchor = c == ']' as i32 && !chars.escaped;
                        if chars.symbols.is_none() && !anchor {
                            // No symbol table: a literal '$'.
                            c = '$' as i32;
                            chars.set_pos(backup);
                        } else if anchor && op == 0 {
                            if last_item == RANGE {
                                self.add(last_char as u32);
                            }
                            self.add(0xffff);
                            out_bracket = true;
                            continue;
                        } else {
                            return Err(chars.syntax_error("Unquoted '$'"));
                        }
                    }
                    _ => {}
                }
            }
            match last_item {
                START => {
                    if op == b'-' && last_string.is_some() {
                        return Err(chars.syntax_error("Invalid range"));
                    }
                    last_item = RANGE;
                    last_char = c;
                    last_string = None;
                }
                RANGE => {
                    if op == b'-' {
                        if last_string.is_some() || last_char >= c {
                            return Err(chars.syntax_error("Invalid range"));
                        }
                        self.add_range(last_char as u32, c as u32);
                        last_item = START;
                        op = 0;
                    } else {
                        self.add(last_char as u32);
                        last_char = c;
                    }
                }
                _ => {
                    if op != 0 {
                        return Err(chars.syntax_error("Set expected after operator"));
                    }
                    last_char = c;
                    last_item = RANGE;
                }
            }
        }
        if !out_bracket {
            return Err(chars.syntax_error("Missing ']'"));
        }
        chars.skip_ignored();
        if invert {
            self.complement();
            self.remove_all_strings();
        }
        Ok(())
    }

    /// `applyPropertyPattern(chars, rebuiltPat, symbols)`: `[:...:]`,
    /// `\p{...}`, `\P{...}` or `\N{...}` at the iterator.
    // ARITH: positions index the pattern.
    #[allow(clippy::arithmetic_side_effects)]
    fn apply_property_pattern_at(&mut self, chars: &mut Chars<'_>) -> Result<(), IcuError> {
        let t: Vec<u16> = chars.lookahead().to_vec();
        let t = &t[..];
        let mut pos = 0usize;
        let fail = |chars: &Chars<'_>| chars.syntax_error("Invalid property pattern");
        if pos + 5 > t.len() {
            return Err(fail(chars));
        }
        let posix;
        let mut invert = false;
        if t[pos] == '[' as u16 && t[pos + 1] == ':' as u16 {
            posix = true;
            pos = skip_white_space(t, pos + 2);
            if pos < t.len() && t[pos] == '^' as u16 {
                pos += 1;
                invert = true;
            }
        } else if t[pos] == '\\' as u16
            && (t[pos + 1] == 'p' as u16 || t[pos + 1] == 'P' as u16 || t[pos + 1] == 'N' as u16)
        {
            let c = t[pos + 1];
            if c == 'N' as u16 {
                return Err(IcuError::unsupported(
                    "UnicodeSet \\N{name} (character names) is not ported",
                ));
            }
            posix = false;
            invert = c == 'P' as u16;
            pos = skip_white_space(t, pos + 2);
            if pos == t.len() || t[pos] != '{' as u16 {
                return Err(fail(chars));
            }
            pos += 1;
        } else {
            return Err(fail(chars));
        }
        let close_pat: &[u16] = if posix { &[0x3a, 0x5d] } else { &[0x7d] };
        let Some(close) = find(t, pos, close_pat) else {
            return Err(fail(chars));
        };
        let equals = find(t, pos, &[0x3d]);
        let (prop, value) = match equals {
            Some(e) if e < close => (utf16::string(&t[pos..e]), utf16::string(&t[e + 1..close])),
            _ => (utf16::string(&t[pos..close]), String::new()),
        };
        self.apply_property_alias(&prop, &value)?;
        if invert {
            self.complement();
            self.remove_all_strings();
        }
        chars.jumpahead(close + if posix { 2 } else { 1 })?;
        Ok(())
    }
}

/// `Double.parseDouble` for decimal input (an optional `f`/`d` suffix, Java
/// trims ASCII controls and spaces); a hexadecimal float is refused as
/// `NumberFormatException` (Java would read it).
fn parse_java_double(s: &str) -> Result<f64, IcuError> {
    let t = s.trim_matches(|c: char| c <= ' ');
    let t = t.strip_suffix(['d', 'D', 'f', 'F']).unwrap_or(t);
    let ok = !t.is_empty()
        && t.chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-'));
    match t.parse::<f64>() {
        Ok(v) if ok => Ok(v),
        _ => Err(IcuError::illegal_argument(format!(
            "NumberFormatException: For input string: \"{s}\""
        ))),
    }
}

/// `UnicodeSet.mungeCharName`: trims, and collapses runs of white space to
/// one space.
fn munge_char_name(s: &str) -> String {
    let mut out = String::new();
    for c in trim_white_space(s).chars() {
        if is_pattern_white_space(c as i32) {
            if out.ends_with(' ') {
                continue;
            }
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

/// `VersionInfo.getInstance(String)`, packed as `Age` runs are.
fn parse_version(v: &str) -> Result<u32, IcuError> {
    let invalid = || IcuError::illegal_argument(format!("Invalid version number: {v}"));
    let mut fields = [0u32; 4];
    let mut count = 0usize;
    let mut index = 0usize;
    let units: Vec<u16> = v.encode_utf16().collect();
    while count < 4 && index < units.len() {
        let c = units[index];
        if c == u16::from(b'.') {
            count = count.saturating_add(1);
        } else {
            let d = c.wrapping_sub(u16::from(b'0'));
            if d > 9 {
                return Err(invalid());
            }
            fields[count] = fields[count]
                .saturating_mul(10)
                .saturating_add(u32::from(d));
        }
        index = index.saturating_add(1);
    }
    if index != units.len() || fields.iter().any(|&f| f > 255) {
        return Err(invalid());
    }
    Ok((fields[0] << 24) | (fields[1] << 16) | (fields[2] << 8) | fields[3])
}

/// `UCharacter.getPropertyEnum`/`getPropertyValueEnum`'s
/// `IllegalIcuArgumentException`.
fn invalid_name(name: &str) -> IcuError {
    IcuError::with_kind(
        crate::IcuErrorKind::IllegalIcuArgument,
        format!("Invalid name: {name}"),
    )
}

/// `CharSequences.getSingleCodePoint`.
fn single_code_point(s: &[u16]) -> Option<u32> {
    let c = utf16::code_point_at(s, 0);
    (!s.is_empty() && s.len() == if c > 0xffff { 2 } else { 1 }).then_some(c as u32)
}

fn find(t: &[u16], from: usize, pat: &[u16]) -> Option<usize> {
    let rest = t.get(from..)?;
    rest.windows(pat.len())
        .position(|w| w == pat)
        .map(|p| p.saturating_add(from))
}

/// `PatternProps.isWhiteSpace(c)`.
pub fn is_pattern_white_space(c: i32) -> bool {
    matches!(
        c,
        0x09..=0x0d | 0x20 | 0x85 | 0x200e | 0x200f | 0x2028 | 0x2029
    )
}

/// `PatternProps.skipWhiteSpace(s, i)`.
fn skip_white_space(t: &[u16], mut i: usize) -> usize {
    while i < t.len() && is_pattern_white_space(i32::from(t[i])) {
        i = i.saturating_add(1);
    }
    i
}

/// `PatternProps.trimWhiteSpace(s)`.
fn trim_white_space(s: &str) -> &str {
    s.trim_matches(|c: char| is_pattern_white_space(c as i32))
}

/// What a symbol table's stand-in character names (`SymbolTable.lookupMatcher`).
pub enum SymbolMatcher<'a> {
    /// A set, which a pattern nests as `[...]`.
    Set(&'a UnicodeSet),
    /// Any other matcher (a string or quantifier), which a set cannot hold.
    Other,
}

/// `com.ibm.icu.text.SymbolTable`: the variables a transliterator's rules
/// define, as a set pattern sees them.
pub trait SymbolTable {
    /// `lookup(name)`: a variable's value (characters, often one stand-in).
    fn lookup(&self, name: &str) -> Option<Vec<u16>>;
    /// `lookupMatcher(ch)`: the matcher a stand-in character names.
    fn lookup_matcher(&self, c: i32) -> Option<SymbolMatcher<'_>>;
    /// `parseReference(text, pos, limit)`: the name after a `$` at `pos`,
    /// and the position after it.
    fn parse_reference(&self, text: &[u16], pos: usize, limit: usize) -> Option<(String, usize)>;
}

/// `RuleCharacterIterator` with `PARSE_ESCAPES`, `SKIP_WHITESPACE` and,
/// given a symbol table, `PARSE_VARIABLES` (a `$name` reads as its value).
struct Chars<'a> {
    text: &'a [u16],
    pos: usize,
    escaped: bool,
    symbols: Option<&'a dyn SymbolTable>,
    buf: Option<Vec<u16>>,
    buf_pos: usize,
}

/// `RuleCharacterIterator.Position`.
#[derive(Clone)]
struct CharsPos {
    buf: Option<Vec<u16>>,
    buf_pos: usize,
    pos: usize,
}

// ARITH: (the whole impl) positions index the text or the variable buffer,
// each step bounded by its length.
#[allow(clippy::arithmetic_side_effects)]
impl<'a> Chars<'a> {
    fn new(text: &'a [u16], pos: usize, symbols: Option<&'a dyn SymbolTable>) -> Self {
        Chars {
            text,
            pos,
            escaped: false,
            symbols,
            buf: None,
            buf_pos: 0,
        }
    }

    fn at_end(&self) -> bool {
        self.buf.is_none() && self.pos >= self.text.len()
    }

    fn in_variable(&self) -> bool {
        self.buf.is_some()
    }

    fn get_pos(&self) -> CharsPos {
        CharsPos {
            buf: self.buf.clone(),
            buf_pos: self.buf_pos,
            pos: self.pos,
        }
    }

    fn set_pos(&mut self, p: CharsPos) {
        self.buf = p.buf;
        self.buf_pos = p.buf_pos;
        self.pos = p.pos;
    }

    // SENTINEL: `-1` = `RuleCharacterIterator.DONE`, the end of the text.
    fn current_cp(&self) -> i32 {
        match &self.buf {
            Some(b) => {
                if self.buf_pos < b.len() {
                    utf16::code_point_at(b, self.buf_pos)
                } else {
                    -1
                }
            }
            None => {
                if self.pos < self.text.len() {
                    utf16::code_point_at(self.text, self.pos)
                } else {
                    -1
                }
            }
        }
    }

    fn advance(&mut self, c: i32) {
        let n = if c > 0xffff { 2 } else { 1 };
        match &self.buf {
            Some(b) => {
                self.buf_pos += n;
                if self.buf_pos >= b.len() {
                    self.buf = None;
                }
            }
            None => self.pos = (self.pos + n).min(self.text.len()),
        }
    }

    /// `jumpahead(count)`.
    fn jumpahead(&mut self, count: usize) -> Result<(), IcuError> {
        match &self.buf {
            Some(b) => {
                self.buf_pos += count;
                if self.buf_pos > b.len() {
                    return Err(IcuError::illegal_argument("jumpahead past the variable"));
                }
                if self.buf_pos == b.len() {
                    self.buf = None;
                }
            }
            None => {
                self.pos += count;
                if self.pos > self.text.len() {
                    return Err(IcuError::illegal_argument("jumpahead past the text"));
                }
            }
        }
        Ok(())
    }

    /// `lookahead()`: the rest of the current buffer.
    fn lookahead(&self) -> &[u16] {
        match &self.buf {
            Some(b) => b.get(self.buf_pos..).unwrap_or(&[]),
            None => self.text.get(self.pos..).unwrap_or(&[]),
        }
    }

    /// `next(PARSE_VARIABLES | PARSE_ESCAPES | SKIP_WHITESPACE)`.
    fn next(&mut self) -> Result<i32, IcuError> {
        self.next_opts(true, true)
    }

    /// `next(options)`.
    fn next_opts(&mut self, parse_escapes: bool, skip_white_space: bool) -> Result<i32, IcuError> {
        self.escaped = false;
        loop {
            let c = self.current_cp();
            self.advance(c);
            if c == '$' as i32 && self.buf.is_none() {
                if let Some(sym) = self.symbols {
                    let Some((name, end)) =
                        sym.parse_reference(self.text, self.pos, self.text.len())
                    else {
                        return Ok(c);
                    };
                    self.pos = end;
                    self.buf_pos = 0;
                    let value = sym.lookup(&name).ok_or_else(|| {
                        IcuError::illegal_argument(format!("Undefined variable: {name}"))
                    })?;
                    if value.is_empty() {
                        // Java: an empty value reads past its end.
                        return Err(IcuError::illegal_argument(format!(
                            "Empty variable: {name}"
                        )));
                    }
                    self.buf = Some(value);
                    continue;
                }
            }
            if skip_white_space && is_pattern_white_space(c) {
                continue;
            }
            if c == '\\' as i32 && parse_escapes {
                let (cp, len) = unescape_at(self.lookahead(), 0)
                    .ok_or_else(|| IcuError::illegal_argument("Invalid escape"))?;
                self.jumpahead(len)?;
                self.escaped = true;
                return Ok(cp);
            }
            return Ok(c);
        }
    }

    /// `skipIgnored(SKIP_WHITESPACE)`.
    fn skip_ignored(&mut self) {
        loop {
            let c = self.current_cp();
            if !is_pattern_white_space(c) {
                break;
            }
            self.advance(c);
        }
    }

    /// `resemblesPropertyPattern(chars, opts)`: `[:` or `\p`, `\P`, `\N`
    /// (white space skipped before the first character only).
    fn resembles_property_pattern(&mut self) -> Result<bool, IcuError> {
        let p = self.get_pos();
        let c = self.next_opts(false, true)?;
        let mut result = false;
        if c == '[' as i32 || c == '\\' as i32 {
            let d = self.next_opts(false, false)?;
            result = if c == '[' as i32 {
                d == ':' as i32
            } else {
                d == 'N' as i32 || d == 'p' as i32 || d == 'P' as i32
            };
        }
        self.set_pos(p);
        Ok(result)
    }

    fn syntax_error(&self, msg: &str) -> IcuError {
        let (a, b) = self.text.split_at(self.pos.min(self.text.len()));
        IcuError::illegal_argument(format!(
            "Error: {msg} at \"{}|{}\"",
            utf16::string(a),
            utf16::string(b)
        ))
    }
}

/// `UnicodeSet.resemblesPropertyPattern(pattern, pos)`: `[:`, `\\p`/`\\P`
/// or `\\N` at `pos` with at least five characters left.
fn resembles_property_pattern_at(t: &[u16], pos: usize) -> bool {
    if pos.saturating_add(5) > t.len() {
        return false;
    }
    match (t[pos], t[pos.saturating_add(1)]) {
        (0x5b, 0x3a) => true,
        (0x5c, d) => d == 'p' as u16 || d == 'P' as u16 || d == 'N' as u16,
        _ => false,
    }
}

fn digit(c: i32, radix: u32) -> Option<i32> {
    char::from_u32(c as u32)
        .and_then(|ch| ch.to_digit(radix))
        .map(|d| d as i32)
}

/// `Utility.unescapeAndLengthAt(s, offset)`: the code point an escape
/// after a backslash at `offset - 1` denotes and the units it spans after
/// the backslash; `None` for a malformed escape.
// ARITH: offsets index the text; digit accumulation is capped at 8 hex
// digits (checked against 0x110000 below) or 3 octal digits.
#[allow(clippy::arithmetic_side_effects)]
pub fn unescape_at(s: &[u16], offset: usize) -> Option<(i32, usize)> {
    let length = s.len();
    if offset >= length {
        return None;
    }
    let start = offset;
    let mut offset = offset;
    let mut c = i32::from(s[offset]);
    offset += 1;
    let (mut min_dig, mut max_dig, mut n, mut bits, mut result, mut braces) =
        (0, 0, 0, 4, 0i64, false);
    match c {
        0x75 => {
            min_dig = 4;
            max_dig = 4;
        }
        0x55 => {
            min_dig = 8;
            max_dig = 8;
        }
        0x78 => {
            min_dig = 1;
            if offset < length && s[offset] == '{' as u16 {
                offset += 1;
                braces = true;
                max_dig = 8;
            } else {
                max_dig = 2;
            }
        }
        _ => {
            if let Some(d) = digit(c, 8) {
                min_dig = 1;
                max_dig = 3;
                n = 1;
                bits = 3;
                result = i64::from(d);
            }
        }
    }
    if min_dig != 0 {
        while offset < length && n < max_dig {
            c = i32::from(s[offset]);
            let Some(d) = digit(c, if bits == 3 { 8 } else { 16 }) else {
                break;
            };
            result = (result << bits) | i64::from(d);
            offset += 1;
            n += 1;
        }
        if n < min_dig {
            return None;
        }
        if braces {
            if c != '}' as i32 {
                return None;
            }
            offset += 1;
        }
        if !(0..0x110000).contains(&result) {
            return None;
        }
        let mut result = result as i32;
        if offset < length && utf16::is_lead(result) {
            let mut ahead = offset + 1;
            c = i32::from(s[offset]);
            if c == '\\' as i32 && ahead < length {
                let tail_limit = (ahead + 11).min(length);
                if let Some((cp, len)) = unescape_at(&s[..tail_limit], ahead) {
                    c = cp;
                    ahead += len;
                }
            }
            if utf16::is_trail(c) {
                offset = ahead;
                result = utf16::to_code_point(result, c);
            }
        }
        return Some((result, offset - start));
    }
    const MAP: [(u8, i32); 8] = [
        (b'a', 7),
        (b'b', 8),
        (b'e', 0x1b),
        (b'f', 0xc),
        (b'n', 0xa),
        (b'r', 0xd),
        (b't', 9),
        (b'v', 0xb),
    ];
    for (k, v) in MAP {
        if c == i32::from(k) {
            return Some((v, offset - start));
        } else if c < i32::from(k) {
            break;
        }
    }
    if c == 'c' as i32 && offset < length {
        let x = utf16::code_point_at(s, offset);
        return Some((x & 0x1f, offset + if x > 0xffff { 2 } else { 1 } - start));
    }
    if utf16::is_lead(c) && offset < length && utf16::is_trail(i32::from(s[offset])) {
        c = utf16::to_code_point(c, i32::from(s[offset]));
        offset += 1;
    }
    Some((c, offset - start))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(p: &str) -> UnicodeSet {
        UnicodeSet::from_pattern(p).unwrap_or_else(|e| panic!("{p}: {e}"))
    }

    fn err(p: &str) -> IcuError {
        UnicodeSet::from_pattern(p).expect_err(p)
    }

    /// Variables: `$vowel` = `[aeiou]` (as a stand-in U+F000), `$xy` = "xy",
    /// `$other` names a non-set matcher (U+F001), `$empty` is empty.
    struct Symbols {
        vowels: UnicodeSet,
    }

    impl SymbolTable for Symbols {
        fn lookup(&self, name: &str) -> Option<Vec<u16>> {
            match name {
                "vowel" => Some(vec![0xf000]),
                "xy" => Some(utf16::units("xy")),
                "other" => Some(vec![0xf001]),
                "empty" => Some(Vec::new()),
                _ => None,
            }
        }

        fn lookup_matcher(&self, c: i32) -> Option<SymbolMatcher<'_>> {
            match c {
                0xf000 => Some(SymbolMatcher::Set(&self.vowels)),
                0xf001 => Some(SymbolMatcher::Other),
                _ => None,
            }
        }

        fn parse_reference(
            &self,
            text: &[u16],
            pos: usize,
            limit: usize,
        ) -> Option<(String, usize)> {
            let mut i = pos;
            while i < limit && (text[i] as u8 as char).is_ascii_alphabetic() && text[i] < 0x80 {
                i = i.saturating_add(1);
            }
            (i > pos).then(|| (utf16::string(&text[pos..i]), i))
        }
    }

    #[test]
    fn symbol_tables() {
        let sym = Symbols {
            vowels: set("[aeiou]"),
        };
        let parse = |p: &str| -> Result<(UnicodeSet, usize), IcuError> {
            let t = utf16::units(p);
            let mut pos = 0;
            let s = UnicodeSet::parse_at(&t, &mut pos, Some(&sym))?;
            Ok((s, pos))
        };
        let (s, end) = parse("[$vowel z] rest").unwrap();
        assert_eq!(s, set("[aeiouz]"));
        assert_eq!(end, 11);
        assert_eq!(parse("[$xy]").unwrap().0, set("[xy]"));
        assert_eq!(parse("[a$]").unwrap().0, set("[a\\uffff]"));
        assert!(parse("[$nope]").is_err());
        assert!(parse("[$other]").is_err());
        assert!(parse("[$empty]").is_err());
        assert!(parse("[a-$]").is_err());
        // A variable value that does not close its set inside it.
        assert!(parse("[[:L:]$xy").is_err());
        assert_eq!(parse("[[:Lu:]&[A-C]]").unwrap().0, set("[A-C]"));
        assert_eq!(parse("\\p{Lu}").unwrap().0, set("[:Lu:]"));
        assert!(UnicodeSet::resembles_pattern(&utf16::units("[a]"), 0));
        assert!(UnicodeSet::resembles_pattern(&utf16::units("\\p{L}"), 0));
        assert!(!UnicodeSet::resembles_pattern(&utf16::units("ab"), 0));
        assert!(!UnicodeSet::resembles_pattern(&utf16::units("\\p"), 0));
    }

    #[test]
    fn set_queries() {
        let a = set("[a-c{xy}]");
        assert_eq!(a.size(), 4);
        assert!(a.contains_some(&set("[c-e]")));
        assert!(a.contains_none(&set("[d-e]")));
        assert!(a.contains_some(&set("[{xy}]")));
        assert!(a.matches_index_value(0x61));
        assert!(!a.matches_index_value(0x64));
        assert!(set("[\\u00fe-\\u0101]").matches_index_value(0x01));
        assert!(set("[{\\u0178z}]").matches_index_value(0x78));
        let mut b = UnicodeSet::new();
        b.add_all_code_points(&utf16::units("b\u{1F600}"));
        assert_eq!(b, set("[b\\U0001F600]"));
        let p = set("[a-c\\u0300{xy}\\U0001F600]");
        assert_eq!(set(&p.to_pattern()), p);
        let mut c = set("[a c-e x]");
        c.add_all(&set("[b f-g z]"));
        assert_eq!(c, set("[a-gxz]"));
    }

    #[test]
    fn literals_ranges_and_negation() {
        let s = set("[a-c x]");
        assert_eq!(s.ranges(), &[(0x61, 0x63), (0x78, 0x78)]);
        let s = set("[^a-c]");
        assert!(!s.contains('b' as i32) && s.contains('d' as i32) && s.contains(0x10ffff));
        assert!(set("[-a]").contains('-' as i32));
        assert!(set("[^-a]").contains('b' as i32));
        assert!(set("[a-z-]").contains('-' as i32));
        assert!(set("[\\u0041\\x42\\x{43}\\U00000044\\105]").ranges() == [(0x41, 0x45)]);
        assert!(set("[\\t\\n]").contains(9));
        assert!(set("[\\cA]").contains(1));
        assert!(set("[\\-]").contains('-' as i32));
        assert!(set("[\\uD801\\uDC00]").contains(0x10400));
        assert!(set("[a$]").contains(0xffff));
        assert!(set("[$a]").contains('$' as i32));
        assert!(set("[$x]").contains('x' as i32));
        assert!(set("[{abc}]").has_strings());
        assert!(set("[{a}]").contains('a' as i32));
        assert!(set("[{a}-{c}]").contains('b' as i32));
        assert!(set("[a-b{xy}c]").has_strings());
        assert!(set("  [ a ]  ").contains('a' as i32));
        assert!(set("[]").is_empty());
    }

    #[test]
    fn operators_and_nesting() {
        let s = set("[[a-z]-[aeiou]]");
        assert!(s.contains('b' as i32) && !s.contains('e' as i32));
        let s = set("[[a-z]&[c-e]]");
        assert_eq!(s.ranges(), &[(0x63, 0x65)]);
        let s = set("[[a-c][x-z]]");
        assert_eq!(s.ranges(), &[(0x61, 0x63), (0x78, 0x7a)]);
        let s = set("[a[x]]");
        assert!(s.contains('a' as i32));
        assert!(set("[[a][b]-]").contains('-' as i32));
    }

    #[test]
    fn properties() {
        assert!(set("[:Latin:]").contains('a' as i32));
        assert!(set("[:^Latin:]").contains('1' as i32));
        assert!(set("\\p{Lu}").contains('A' as i32));
        assert!(!set("\\P{Lu}").contains('A' as i32));
        assert!(set("\\p{gc=Lu}").contains('A' as i32));
        assert!(set("\\p{General_Category=Letter}").contains('a' as i32));
        assert!(set("[\\p{sc=Grek}]").contains(0x3b1));
        assert!(set("[\\p{scx=Hani}]").contains(0x3001));
        assert!(set("[:Nonspacing Mark:]").contains(0x301));
        assert!(set("[:White_Space:]").contains(' ' as i32));
        assert!(!set("[:White_Space=No:]").contains(' ' as i32));
        assert!(set("[:lb=SA:]").contains(0xe01));
        assert!(set("[:ccc=230:]").contains(0x301));
        assert!(set("[:ANY:]").contains(0x10ffff));
        assert!(set("[:ascii:]").contains(0x7f));
        assert!(!set("[:Assigned:]").contains(0x10ffff));
        assert!(set("[[:Emoji:][:Extended_Pictographic:]]").contains(0x1f600));
        assert!(set("[[:Thai:]&[:LineBreak=SA:]]").contains(0xe01));
        assert!(set("[:ccc=Above:]").contains(0x301));
    }

    #[test]
    fn errors() {
        for p in [
            "a",
            "[a",
            "[a-]x",
            "[z-a]",
            "[a-[b]]",
            "[[a]&]",
            "[a&[b]]",
            "[^^]",
            "[{ab]",
            "[[a]-b]",
            "[{ab}-c]",
            "[a-{bc}]",
            "[[a]&{b}]",
            "[[:Latin:]&]",
            "\\p{Lu",
            "\\p{NoSuchThing}",
            "\\p{gc=Nope}",
            "\\p{Nope=x}",
            "[:Line_Break:]",
            "[:ccc=300:]",
            "[:ccc=x1:]",
            "[\\",
            "[\\u12]",
            "[\\x{12]",
            "[\\U00110000]",
            "[a] b",
            "[:L:",
            "\\px",
            "[[a]-$]",
        ] {
            let e = err(p);
            assert!(
                matches!(
                    e.kind(),
                    crate::IcuErrorKind::IllegalArgument | crate::IcuErrorKind::IllegalIcuArgument
                ),
                "{p}: {e}"
            );
        }
        for p in [
            "[\\N{LATIN SMALL LETTER A}]",
            "[{ab}-{ad}]",
            "\\p{name=x}",
            "\\p{ID_Type=x}",
        ] {
            assert_eq!(
                err(p).kind(),
                crate::IcuErrorKind::UnsupportedOperation,
                "{p}"
            );
        }
        let mut deep = "[".repeat(102);
        deep.push_str(&"]".repeat(102));
        assert!(err(&deep).message().contains("nested"));
    }

    #[test]
    fn set_algebra_and_spans() {
        let mut s = UnicodeSet::from_range(0x61, 0x7a);
        s.add_range(0x30, 0x39);
        s.add_range(0x3a, 0x40);
        s.add_range(5, 2);
        s.add_range(0x200000, 0x200001);
        assert_eq!(s.ranges(), &[(0x30, 0x40), (0x61, 0x7a)]);
        let mut t = s.clone();
        t.remove_all(&UnicodeSet::from_range(0x35, 0x62));
        assert_eq!(t.ranges(), &[(0x30, 0x34), (0x63, 0x7a)]);
        t.add_string(&utf16::units("xy"));
        let mut u = t.clone();
        u.retain_all(&s);
        assert!(!u.has_strings());
        u.add_all(&t);
        assert!(u.has_strings());
        u.remove_all(&t);
        assert!(!u.has_strings());
        u.add_string(&utf16::units("\u{10400}"));
        assert!(u.contains(0x10400) && !u.contains(-1));
        let text = utf16::units("abc123\u{10400}x");
        let s = UnicodeSet::from_range(0x61, 0x7a);
        assert_eq!(s.span(&text, 0, SpanCondition::Simple), 3);
        assert_eq!(s.span(&text, 3, SpanCondition::NotContained), 8);
        assert_eq!(s.span_back(&text, text.len(), SpanCondition::Simple), 8);
        assert_eq!(s.span_back(&text, 8, SpanCondition::NotContained), 3);
        let mut e = UnicodeSet::new();
        e.add(0);
        e.complement();
        assert_eq!(e.ranges(), &[(1, 0x10ffff)]);
        e.complement();
        assert_eq!(e.ranges(), &[(0, 0)]);
        e.clear();
        assert!(e.is_empty());
        assert!(e.apply_int_property_value(0x7777, 1).is_err());
        e.apply_int_property_value(0, 7).unwrap(); // a binary property, value 7: empty
        assert!(e.is_empty());
    }

    #[test]
    fn unescape() {
        let u = |s: &str| unescape_at(&utf16::units(s), 0);
        assert_eq!(u("u0041"), Some((0x41, 5)));
        assert_eq!(u("x{1F600}"), Some((0x1f600, 8)));
        assert_eq!(u("x4"), Some((4, 2)));
        assert_eq!(u("101"), Some((0o101, 3)));
        assert_eq!(u("q"), Some(('q' as i32, 1)));
        assert_eq!(u("z"), Some(('z' as i32, 1)));
        assert_eq!(u("\u{10400}"), Some((0x10400, 2)));
        assert_eq!(u("c\u{10400}"), Some((0, 3)));
        assert_eq!(u("uD801\\x{DC00}"), Some((0x10400, 13)));
        assert_eq!(u("uD801\\q"), Some((0xd801, 5)));
        assert_eq!(u("u12"), None);
        assert_eq!(unescape_at(&[], 0), None);
        assert_eq!(trim_white_space(" 12\t"), "12");
    }
}
