//! `com.ibm.icu.text.Transliterator` and the transliterators Lucene's
//! `ICUTransformFilter` can name: the rule-based ones ICU ships
//! (`translit/root.res`: script conversions such as `Cyrillic-Latin`,
//! `Han-Latin`, `Traditional-Simplified`, `Katakana-Hiragana`,
//! `Fullwidth-Halfwidth`, ...), compound IDs (`NFD; [:Nonspacing Mark:]
//! Remove; NFC`), global and per-element filters, `Any-` script dispatch,
//! normalization (`NFC` .. `FCC`), case (`Lower`, `Upper`, `Title`,
//! `CaseFold`), `Null`, `Remove`, the internal word-break inserter, and
//! transliterators built from a caller's rules.
//!
//! The text is a UTF-16 buffer (`Replaceable`), positions Java's `int`s.
//! Ported: non-incremental transliteration (`transliterate`,
//! `filteredTransliterate(text, pos, false)`), what Lucene and the
//! transliterators' nested calls use. Not ported (typed
//! `UnsupportedOperation`): incremental (keyboard) transliteration, the
//! `Any-Name`/`Name-Any` and `Hex` escape transliterators (`unames.icu`,
//! `EscapeTransliterator`), `Any-` targets named by a locale rather than a
//! script (`Any-am_FONIPA`, which need CLDR's likely subtags), and display
//! names.

pub mod id;
pub mod parser;
pub mod registry;
pub mod rules;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::icu4j::normalizer2::Normalizer2;
use crate::icu4j::translit::rules::{handle_rule_based, Data};
use crate::icu4j::ucase::{self, ContextIterator};
use crate::icu4j::unicode_set::UnicodeSet;
use crate::icu4j::uprops;
use crate::icu4j::utf16;
use crate::IcuError;

/// `Transliterator.FORWARD` / `REVERSE`.
pub const FORWARD: i32 = 0;
pub const REVERSE: i32 = 1;

/// `Transliterator.Position`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Position {
    pub context_start: i32,
    pub context_limit: i32,
    pub start: i32,
    pub limit: i32,
}

/// `UTF16.getCharCount(c)`.
#[inline]
pub(crate) fn char_count(c: i32) -> usize {
    if c >= 0x10000 {
        2
    } else {
        1
    }
}

/// `Replaceable.char32At(offset)` (`UTF16.charAt`): the code point at a
/// unit, paired with its neighbour when that completes a surrogate pair;
/// U+FFFF out of range (where Java throws; no caller reads out of range).
pub(crate) fn char32_at(text: &[u16], offset: i32) -> i32 {
    let Some(i) = usize::try_from(offset).ok().filter(|&i| i < text.len()) else {
        return 0xffff;
    };
    let single = i32::from(text[i]);
    if utf16::is_lead(single) {
        if let Some(&t) = text.get(i.saturating_add(1)) {
            if utf16::is_trail(i32::from(t)) {
                return utf16::to_code_point(single, i32::from(t));
            }
        }
    } else if utf16::is_trail(single) && i > 0 {
        let l = i32::from(text[i.saturating_sub(1)]);
        if utf16::is_lead(l) {
            return utf16::to_code_point(l, single);
        }
    }
    single
}

/// The `[start, limit)` range of `text` clamped to it.
fn clamp_range(text: &[u16], start: i32, limit: i32) -> (usize, usize) {
    let n = text.len();
    let s = usize::try_from(start).unwrap_or(0).min(n);
    let l = usize::try_from(limit).unwrap_or(0).min(n).max(s);
    (s, l)
}

/// `Replaceable.replace(start, limit, text)`.
pub(crate) fn replace_text(text: &mut Vec<u16>, start: i32, limit: i32, with: &[u16]) {
    let (s, l) = clamp_range(text, start, limit);
    text.splice(s..l, with.iter().copied());
}

/// `Replaceable.copy(start, limit, dest)`.
pub(crate) fn copy_text(text: &mut Vec<u16>, start: i32, limit: i32, dest: i32) {
    let (s, l) = clamp_range(text, start, limit);
    if s == l {
        return;
    }
    let copy: Vec<u16> = text[s..l].to_vec();
    replace_text(text, dest, dest, &copy);
}

/// Which case transliterator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseKind {
    Lower,
    Upper,
    Title,
    Fold,
}

/// `AnyTransliterator`'s state: its target and the per-script
/// transliterators it has built.
#[derive(Debug)]
pub struct AnyState {
    target: String,
    target_script: i32,
    cache: Mutex<HashMap<i32, Option<Arc<Transliterator>>>>,
}

/// The kinds of transliterator (Java's subclasses).
#[derive(Debug, Clone)]
pub enum Kind {
    Null,
    Remove,
    RuleBased(Arc<Data>),
    Compound(Arc<Vec<Transliterator>>),
    Normalization(Normalizer2),
    Case(CaseKind),
    Any(Arc<AnyState>),
    Break,
}

/// `Transliterator`.
#[derive(Debug, Clone)]
pub struct Transliterator {
    id: String,
    filter: Option<Arc<UnicodeSet>>,
    kind: Kind,
}

const SCRIPT_COMMON: i32 = 0;
const SCRIPT_INHERITED: i32 = 1;
const SCRIPT_INVALID: i32 = -1;

/// `UScript.getScript(c)`.
fn script_of(c: i32) -> i32 {
    uprops::script(c)
}

// ARITH: (the whole impl) text positions and deltas, Java's int
// arithmetic over texts shorter than 2^31 units.
#[allow(clippy::arithmetic_side_effects)]
impl Transliterator {
    pub(crate) fn new(id: &str, kind: Kind) -> Transliterator {
        Transliterator {
            id: id.to_string(),
            filter: None,
            kind,
        }
    }

    /// `getID()`.
    pub fn id(&self) -> &str {
        &self.id
    }

    pub(crate) fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }

    /// `getFilter()`.
    pub fn filter(&self) -> Option<&UnicodeSet> {
        self.filter.as_deref()
    }

    /// `setFilter(filter)`.
    pub fn set_filter(&mut self, filter: Option<UnicodeSet>) {
        self.filter = filter.map(Arc::new);
    }

    /// Whether this is a single rule-based transliterator (Java's
    /// `instanceof RuleBasedTransliterator`).
    pub fn is_rule_based(&self) -> bool {
        matches!(self.kind, Kind::RuleBased(_))
    }

    pub(crate) fn kind(&self) -> &Kind {
        &self.kind
    }

    /// `Transliterator.getInstance(id, dir)`.
    pub fn get_instance(id: &str, dir: i32) -> Result<Transliterator, IcuError> {
        registry::get_instance(id, dir)
    }

    /// `Transliterator.createFromRules(id, rules, dir)`.
    pub fn create_from_rules(id: &str, rules: &str, dir: i32) -> Result<Transliterator, IcuError> {
        registry::create_from_rules(id, rules, dir)
    }

    /// `transliterate(String)`.
    pub fn transliterate(&self, s: &str) -> Result<String, IcuError> {
        let mut text: Vec<u16> = s.encode_utf16().collect();
        self.transliterate_units(&mut text)?;
        Ok(String::from_utf16_lossy(&text))
    }

    /// `transliterate(Replaceable)` over UTF-16 units.
    pub fn transliterate_units(&self, text: &mut Vec<u16>) -> Result<(), IcuError> {
        let len = text.len() as i32;
        self.transliterate_range(text, 0, len)?;
        Ok(())
    }

    /// `transliterate(Replaceable, start, limit)`: the new limit, or -1
    /// for an invalid range.
    pub fn transliterate_range(
        &self,
        text: &mut Vec<u16>,
        start: i32,
        limit: i32,
    ) -> Result<i32, IcuError> {
        if start < 0 || limit < start || (text.len() as i32) < limit {
            return Ok(-1);
        }
        let mut pos = Position {
            context_start: start,
            context_limit: limit,
            start,
            limit,
        };
        self.filtered_transliterate_impl(text, &mut pos, true)?;
        Ok(pos.limit)
    }

    /// `filteredTransliterate(text, index, false)`, what Lucene's
    /// `ICUTransformFilter` calls on each term.
    pub fn filtered_transliterate(
        &self,
        text: &mut Vec<u16>,
        index: &mut Position,
    ) -> Result<(), IcuError> {
        self.filtered_transliterate_impl(text, index, false)
    }

    /// `filteredTransliterate(text, index, false, rollback)`.
    fn filtered_transliterate_impl(
        &self,
        text: &mut Vec<u16>,
        index: &mut Position,
        rollback: bool,
    ) -> Result<(), IcuError> {
        if self.filter.is_none() && !rollback {
            return self.handle(text, index);
        }
        let mut global_limit = index.limit;
        loop {
            if let Some(filter) = &self.filter {
                while index.start < global_limit {
                    let c = char32_at(text, index.start);
                    if filter.contains(c) {
                        break;
                    }
                    index.start += char_count(c) as i32;
                }
                index.limit = index.start;
                while index.limit < global_limit {
                    let c = char32_at(text, index.limit);
                    if !filter.contains(c) {
                        break;
                    }
                    index.limit += char_count(c) as i32;
                }
            }
            if index.start == index.limit {
                break;
            }
            let limit = index.limit;
            self.handle(text, index)?;
            let delta = index.limit - limit;
            if index.start != index.limit {
                return Err(IcuError::new(format!(
                    "ERROR: Incomplete non-incremental transliteration by {}",
                    self.id
                )));
            }
            global_limit += delta;
            if self.filter.is_none() {
                break;
            }
        }
        index.limit = global_limit;
        Ok(())
    }

    /// `handleTransliterate(text, pos, false)`.
    fn handle(&self, text: &mut Vec<u16>, pos: &mut Position) -> Result<(), IcuError> {
        match &self.kind {
            Kind::Null => {
                pos.start = pos.limit;
                Ok(())
            }
            Kind::Remove => {
                replace_text(text, pos.start, pos.limit, &[]);
                let len = pos.limit - pos.start;
                pos.context_limit -= len;
                pos.limit -= len;
                Ok(())
            }
            Kind::RuleBased(data) => handle_rule_based(data, text, pos),
            Kind::Compound(trans) => {
                if trans.is_empty() {
                    pos.start = pos.limit;
                    return Ok(());
                }
                let compound_limit = pos.limit;
                let compound_start = pos.start;
                let mut delta = 0;
                for t in trans.iter() {
                    pos.start = compound_start;
                    let limit = pos.limit;
                    if pos.start == pos.limit {
                        break;
                    }
                    t.filtered_transliterate(text, pos)?;
                    if pos.start != pos.limit {
                        return Err(IcuError::new(format!(
                            "ERROR: Incomplete non-incremental transliteration by {}",
                            t.id
                        )));
                    }
                    delta += pos.limit - limit;
                }
                pos.limit = compound_limit + delta;
                Ok(())
            }
            Kind::Normalization(norm2) => {
                handle_normalization(norm2, text, pos);
                Ok(())
            }
            Kind::Case(kind) => {
                handle_case(*kind, text, pos);
                Ok(())
            }
            Kind::Any(state) => self.handle_any(state, text, pos),
            Kind::Break => handle_break(text, pos),
        }
    }

    /// `AnyTransliterator.handleTransliterate(text, pos, false)`.
    fn handle_any(
        &self,
        state: &AnyState,
        text: &mut Vec<u16>,
        pos: &mut Position,
    ) -> Result<(), IcuError> {
        let all_start = pos.start;
        let mut all_limit = pos.limit;
        let mut it = ScriptRunIterator {
            text_start: pos.context_start,
            text_limit: pos.context_limit,
            script_code: SCRIPT_INVALID,
            start: pos.context_start,
            limit: pos.context_start,
        };
        while it.next(text) {
            if it.limit <= all_start {
                continue;
            }
            let Some(t) = get_any_transliterator(state, it.script_code)? else {
                pos.start = it.limit;
                continue;
            };
            pos.start = all_start.max(it.start);
            pos.limit = all_limit.min(it.limit);
            let limit = pos.limit;
            t.filtered_transliterate(text, pos)?;
            let delta = pos.limit - limit;
            all_limit += delta;
            it.limit += delta;
            it.text_limit += delta;
            if it.limit >= all_limit {
                break;
            }
        }
        pos.limit = all_limit;
        Ok(())
    }

    /// `getSourceSet()`: the characters this transliterator can change
    /// (`addSourceTargetSet`'s source half), within its filter.
    pub fn source_set(&self) -> Result<UnicodeSet, IcuError> {
        let mut result = UnicodeSet::new();
        let all = UnicodeSet::from_range(0, 0x10ffff);
        let filter = self.filter_as_unicode_set(&all);
        self.add_source_set(&filter, &mut result)?;
        Ok(result)
    }

    /// `getFilterAsUnicodeSet(externalFilter)`.
    fn filter_as_unicode_set(&self, external: &UnicodeSet) -> UnicodeSet {
        match &self.filter {
            None => external.clone(),
            Some(f) => {
                let mut s = external.clone();
                s.retain_all(f);
                s
            }
        }
    }

    fn add_source_set(
        &self,
        input_filter: &UnicodeSet,
        source: &mut UnicodeSet,
    ) -> Result<(), IcuError> {
        let my_filter = self.filter_as_unicode_set(input_filter);
        match &self.kind {
            Kind::RuleBased(data) => data.rule_set.add_source_set(data, &my_filter, source),
            Kind::Remove | Kind::Any(_) => source.add_all(&my_filter),
            Kind::Null => {}
            _ => {
                return Err(IcuError::unsupported(
                    "getSourceSet of a compound, normalization, case or break transliterator is not ported",
                ))
            }
        }
        Ok(())
    }

    /// `getInverse()`.
    pub fn inverse(&self) -> Result<Transliterator, IcuError> {
        Self::get_instance(&self.id, REVERSE)
    }
}

/// `AnyTransliterator.ScriptRunIterator`.
struct ScriptRunIterator {
    text_start: i32,
    text_limit: i32,
    script_code: i32,
    start: i32,
    limit: i32,
}

// ARITH: (the whole impl) positions within the text.
#[allow(clippy::arithmetic_side_effects)]
impl ScriptRunIterator {
    fn next(&mut self, text: &[u16]) -> bool {
        self.script_code = SCRIPT_INVALID;
        self.start = self.limit;
        if self.start == self.text_limit {
            return false;
        }
        while self.start > self.text_start {
            let s = script_of(char32_at(text, self.start - 1));
            if s == SCRIPT_COMMON || s == SCRIPT_INHERITED {
                self.start -= 1;
            } else {
                break;
            }
        }
        while self.limit < self.text_limit {
            let s = script_of(char32_at(text, self.limit));
            if s != SCRIPT_COMMON && s != SCRIPT_INHERITED {
                if self.script_code == SCRIPT_INVALID {
                    self.script_code = s;
                } else if s != self.script_code {
                    break;
                }
            }
            self.limit += 1;
        }
        true
    }
}

/// `AnyTransliterator.isWide(script)`: Bopomofo, Han, Hangul, Hiragana,
/// Katakana.
fn is_wide(script: i32) -> bool {
    matches!(script, 5 | 17 | 18 | 20 | 22)
}

/// `AnyTransliterator.WidthFix.INSTANCE`.
fn width_fix() -> Result<Arc<Transliterator>, IcuError> {
    static W: Mutex<Option<Arc<Transliterator>>> = Mutex::new(None);
    let mut guard = W.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(t) = guard.as_ref() {
        return Ok(t.clone());
    }
    let t = Arc::new(Transliterator::get_instance(
        "[[:dt=Nar:][:dt=Wide:]] nfkd",
        FORWARD,
    )?);
    *guard = Some(t.clone());
    Ok(t)
}

/// `AnyTransliterator.getTransliterator(source)`.
fn get_any_transliterator(
    state: &AnyState,
    source: i32,
) -> Result<Option<Arc<Transliterator>>, IcuError> {
    if source == state.target_script || source == SCRIPT_INVALID {
        return if is_wide(state.target_script) {
            Ok(None)
        } else {
            Ok(Some(width_fix()?))
        };
    }
    if let Some(t) = state
        .cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&source)
    {
        return Ok(t.clone());
    }
    let source_name = uprops::uprops()
        .script_name(source)
        .map(|(_, long)| long.to_string())
        .unwrap_or_default();
    let mut t =
        Transliterator::get_instance(&format!("{source_name}-{}", state.target), FORWARD).ok();
    if t.is_none() {
        t = Transliterator::get_instance(
            &format!("{source_name}-Latin;Latin-{}", state.target),
            FORWARD,
        )
        .ok();
    }
    let result = match t {
        Some(t) => {
            let t = if is_wide(state.target_script) {
                t
            } else {
                let mut c = Transliterator::new(
                    "",
                    Kind::Compound(Arc::new(vec![(*width_fix()?).clone(), t])),
                );
                c.set_id("");
                c
            };
            Some(Arc::new(t))
        }
        None if !is_wide(state.target_script) => return Ok(Some(width_fix()?)),
        None => None,
    };
    state
        .cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry(source)
        .or_insert(result.clone());
    Ok(result)
}

/// A new `AnyTransliterator`.
pub(crate) fn any_transliterator(
    id: &str,
    target: &str,
    variant: &str,
    target_script: i32,
) -> Transliterator {
    let target = if variant.is_empty() {
        target.to_string()
    } else {
        format!("{target}/{variant}")
    };
    Transliterator::new(
        id,
        Kind::Any(Arc::new(AnyState {
            target,
            target_script,
            cache: Mutex::new(HashMap::new()),
        })),
    )
}

/// `NormalizationTransliterator.handleTransliterate(text, offsets, false)`.
// ARITH: positions within the text.
#[allow(clippy::arithmetic_side_effects)]
fn handle_normalization(norm2: &Normalizer2, text: &mut Vec<u16>, offsets: &mut Position) {
    let mut start = offsets.start;
    let mut limit = offsets.limit;
    if start >= limit {
        return;
    }
    let mut segment: Vec<u16> = Vec::new();
    let mut normalized: Vec<u16> = Vec::new();
    let mut c = char32_at(text, start);
    loop {
        let prev = start;
        segment.clear();
        loop {
            utf16::push_code_point(&mut segment, c);
            start += char_count(c) as i32;
            if start >= limit {
                break;
            }
            c = char32_at(text, start);
            if norm2.has_boundary_before(c) {
                break;
            }
        }
        norm2.normalize_to(&segment, &mut normalized);
        if segment != normalized {
            replace_text(text, prev, start, &normalized);
            let delta = normalized.len() as i32 - (start - prev);
            start += delta;
            limit += delta;
        }
        if start >= limit {
            break;
        }
    }
    offsets.start = start;
    offsets.context_limit += limit - offsets.limit;
    offsets.limit = limit;
}

/// `ReplaceableContextIterator` over a snapshot of the text (case mapping
/// reads the context; the caller replaces afterwards).
struct CaseContext<'a> {
    text: &'a [u16],
    cp_start: i32,
    cp_limit: i32,
    context_start: i32,
    context_limit: i32,
    index: i32,
    dir: i32,
}

// ARITH: (the whole impl) positions within the text.
#[allow(clippy::arithmetic_side_effects)]
impl ContextIterator for CaseContext<'_> {
    fn reset(&mut self, dir: i32) {
        if dir > 0 {
            self.dir = 1;
            self.index = self.cp_limit;
        } else if dir < 0 {
            self.dir = -1;
            self.index = self.cp_start;
        } else {
            self.dir = 0;
            self.index = 0;
        }
    }

    // SENTINEL: `-1` = the end of the context (`UCaseProps.ContextIterator`).
    fn next_context(&mut self) -> i32 {
        if self.dir > 0 {
            if self.index < self.context_limit {
                let c = char32_at(self.text, self.index);
                self.index += char_count(c) as i32;
                return c;
            }
        } else if self.dir < 0 && self.index > self.context_start {
            let c = char32_at(self.text, self.index - 1);
            self.index -= char_count(c) as i32;
            return c;
        }
        -1
    }
}

/// The case transliterators' `handleTransliterate(text, offsets, false)`
/// (`Any-Lower`, `Any-Upper`, `Any-Title` in `ULocale.US`, `Any-CaseFold`).
// ARITH: positions within the text.
#[allow(clippy::arithmetic_side_effects)]
fn handle_case(kind: CaseKind, text: &mut Vec<u16>, offsets: &mut Position) {
    if offsets.start >= offsets.limit {
        return;
    }
    let csp = ucase::instance();
    let loc = ucase::LOC_ROOT;
    let len = text.len() as i32;
    let mut limit = offsets.limit.clamp(0, len);
    let context_start = offsets.context_start.clamp(0, len);
    let mut context_limit = offsets.context_limit.clamp(context_start, len);
    let mut do_title = true;
    if kind == CaseKind::Title {
        let mut start = offsets.start - 1;
        while start >= offsets.context_start {
            let c = char32_at(text, start);
            let t = csp.get_type_or_ignorable(c);
            if t > 0 {
                do_title = false;
                break;
            } else if t == 0 {
                break;
            }
            start -= char_count(c) as i32;
        }
    }
    let mut cp_limit = offsets.start;
    let mut out: Vec<u16> = Vec::new();
    while cp_limit < limit {
        let cp_start = cp_limit;
        let c = char32_at(text, cp_limit);
        cp_limit += char_count(c) as i32;
        out.clear();
        let mut ctx = CaseContext {
            text,
            cp_start,
            cp_limit,
            context_start,
            context_limit,
            index: 0,
            dir: 0,
        };
        let result = match kind {
            CaseKind::Lower => csp.to_full_lower(c, &mut ctx, &mut out, loc),
            CaseKind::Upper => csp.to_full_upper(c, &mut ctx, &mut out, loc),
            CaseKind::Fold => csp.to_full_folding(c, &mut out, false),
            CaseKind::Title => {
                let t = csp.get_type_or_ignorable(c);
                let r = if do_title {
                    csp.to_full_title(c, &mut ctx, &mut out, loc)
                } else {
                    csp.to_full_lower(c, &mut ctx, &mut out, loc)
                };
                do_title = t == 0;
                r
            }
        };
        if result < 0 {
            continue;
        }
        let replacement: Vec<u16> = if result <= ucase::MAX_STRING_LENGTH {
            out.clone()
        } else {
            let mut v = Vec::new();
            utf16::push_code_point(&mut v, result);
            v
        };
        let delta = replacement.len() as i32 - (cp_limit - cp_start);
        replace_text(text, cp_start, cp_limit, &replacement);
        cp_limit += delta;
        limit += delta;
        context_limit += delta;
        if delta != 0 {
            offsets.limit += delta;
            offsets.context_limit += delta;
        }
    }
    offsets.start = offsets.limit;
}

/// `BreakTransliterator.handleTransliterate(text, pos, false)`: a space
/// between letters (or marks) at each Thai word break.
// ARITH: positions within the text.
#[allow(clippy::arithmetic_side_effects)]
fn handle_break(text: &mut Vec<u16>, pos: &mut Position) -> Result<(), IcuError> {
    use crate::icu4j::rbbi::{RuleBasedBreakIterator, DONE};
    const LETTER_OR_MARK_MASK: i32 =
        (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8);
    let (s, l) = clamp_range(text, pos.start, pos.limit);
    let mut bi = RuleBasedBreakIterator::from_data(crate::segmentation::config::cjk_rules());
    bi.set_text(&text[s..l]);
    let mut boundaries: Vec<i32> = Vec::new();
    let mut b = bi.first();
    while b != DONE {
        let boundary = b + s as i32;
        if boundary >= pos.limit {
            break;
        }
        if boundary != 0 {
            let cp = char32_at(text, boundary - 1);
            let ok1 = (1 << uprops::char_type(cp)) & LETTER_OR_MARK_MASK != 0;
            let cp = char32_at(text, boundary);
            let ok2 = (1 << uprops::char_type(cp)) & LETTER_OR_MARK_MASK != 0;
            if ok1 && ok2 {
                boundaries.push(boundary);
            }
        }
        b = bi.next();
    }
    let insertion = [0x20u16];
    let delta = boundaries.len() as i32;
    for &boundary in boundaries.iter().rev() {
        replace_text(text, boundary, boundary, &insertion);
    }
    pos.context_limit += delta;
    pos.limit += delta;
    pos.start = pos.limit;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tr(id: &str) -> Transliterator {
        Transliterator::get_instance(id, FORWARD).unwrap()
    }

    #[test]
    fn ranges_and_positions() {
        let t = tr("Any-Upper");
        assert_eq!(t.transliterate("abc").unwrap(), "ABC");
        let mut text: Vec<u16> = "abcd".encode_utf16().collect();
        assert_eq!(t.transliterate_range(&mut text, 1, 3).unwrap(), 3);
        assert_eq!(String::from_utf16_lossy(&text), "aBCd");
        assert_eq!(t.transliterate_range(&mut text, -1, 2).unwrap(), -1);
        assert_eq!(t.transliterate_range(&mut text, 3, 2).unwrap(), -1);
        assert_eq!(t.transliterate_range(&mut text, 0, 9).unwrap(), -1);
        // Empty ranges leave the text alone.
        for id in [
            "NFD",
            "Any-Lower",
            "Any-Title",
            "Any-CaseFold",
            "Remove",
            "Null",
        ] {
            let mut text: Vec<u16> = "Ab".encode_utf16().collect();
            assert_eq!(tr(id).transliterate_range(&mut text, 1, 1).unwrap(), 1);
            assert_eq!(text, "Ab".encode_utf16().collect::<Vec<_>>());
        }
        assert_eq!(char32_at(&[0x41], 5), 0xffff);
        assert_eq!(char32_at(&[0xd801, 0xdc00], 1), 0x10400);
        assert_eq!(char32_at(&[0xd801, 0x41], 0), 0xd801);
        assert_eq!(char32_at(&[0x41, 0xdc00], 1), 0xdc00);
        let mut v = vec![1, 2, 3];
        copy_text(&mut v, 1, 1, 0);
        assert_eq!(v, [1, 2, 3]);
    }

    #[test]
    fn title_case_reads_the_context() {
        // The context starts at the range (Java's `Position(start, limit,
        // start)`), so the letter before it does not count (outputs from
        // ICU4J 77.1).
        let t = tr("Any-Title");
        let mut text: Vec<u16> = "ab cd".encode_utf16().collect();
        t.transliterate_range(&mut text, 1, 5).unwrap();
        assert_eq!(String::from_utf16_lossy(&text), "aB Cd");
        let mut text: Vec<u16> = "a'b".encode_utf16().collect();
        t.transliterate_range(&mut text, 2, 3).unwrap();
        assert_eq!(String::from_utf16_lossy(&text), "a'B");
        // Within a range, a letter or an ignorable before the cursor does.
        let mut pos = Position {
            context_start: 0,
            context_limit: 3,
            start: 2,
            limit: 3,
        };
        let mut text: Vec<u16> = "a'b".encode_utf16().collect();
        t.filtered_transliterate(&mut text, &mut pos).unwrap();
        assert_eq!(String::from_utf16_lossy(&text), "a'b");
        let mut pos = Position {
            context_start: 0,
            context_limit: 3,
            start: 2,
            limit: 3,
        };
        let mut text: Vec<u16> = ". b".encode_utf16().collect();
        t.filtered_transliterate(&mut text, &mut pos).unwrap();
        assert_eq!(String::from_utf16_lossy(&text), ". B");
        let mut text: Vec<u16> = " b".encode_utf16().collect();
        t.transliterate_range(&mut text, 1, 2).unwrap();
        assert_eq!(String::from_utf16_lossy(&text), " B");
        assert_eq!(tr("Any-Lower").transliterate("ΟΔΟΣ Σ").unwrap(), "οδος σ");
    }

    #[test]
    fn compound_and_any_paths() {
        // A pass that cannot finish inside a compound.
        let t = Transliterator::create_from_rules("T", "::Null; b* > Y;", FORWARD).unwrap();
        assert!(t.transliterate("abc").is_err());
        let empty = Transliterator::new("E", Kind::Compound(Arc::new(Vec::new())));
        assert_eq!(empty.transliterate("ab").unwrap(), "ab");
        // Any- dispatch over runs after the start of the context.
        let t = tr("[^a] Any-Latin");
        assert_eq!(t.transliterate("aaМосква").unwrap(), "aaMoskva");
        let t = tr("Any-Latin");
        let mut text: Vec<u16> = "Мос ква".encode_utf16().collect();
        t.transliterate_range(&mut text, 4, 7).unwrap();
        assert_eq!(String::from_utf16_lossy(&text), "Мос kva");
        assert_eq!(tr("Any-Hangul").transliterate("한 a").unwrap(), "한 아");
        assert!(Transliterator::get_instance("Any-Han", FORWARD).is_err());
    }

    #[test]
    fn source_sets_and_inverses() {
        let t = tr("[a-c] Remove");
        let set = t.source_set().unwrap();
        assert!(set.contains(0x61) && !set.contains(0x64));
        assert!(tr("Null").source_set().unwrap().is_empty());
        assert!(tr("Any-Latin").source_set().is_ok());
        assert!(tr("NFD; Lower").source_set().is_err());
        let t = Transliterator::create_from_rules("T", "a > b;", FORWARD).unwrap();
        assert!(t.source_set().unwrap().contains(0x61));
        assert_eq!(
            tr("Latin-Cyrillic").inverse().unwrap().id(),
            "Cyrillic-Latin"
        );
        assert_eq!(tr("Any-Upper").inverse().unwrap().id(), "Any-Lower");
        let mut t = tr("Any-Upper");
        assert!(t.filter().is_none());
        t.set_filter(Some(UnicodeSet::from_pattern("[a]").unwrap()));
        assert_eq!(t.transliterate("ab").unwrap(), "Ab");
        assert!(!t.is_rule_based());
        assert!(Transliterator::create_from_rules("T", "a > b;", FORWARD)
            .unwrap()
            .is_rule_based());
    }
}
