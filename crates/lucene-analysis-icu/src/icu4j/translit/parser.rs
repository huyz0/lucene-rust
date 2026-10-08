//! `com.ibm.icu.text.TransliteratorParser`: transliteration rules --
//! `::ID;` blocks, `$variable = ...;` definitions, `a {b} c > |x @@;` rules
//! (`>`, `<`, `<>` and their arrows), sets, quoted strings, escapes,
//! `(segments)` and `$1` references, `* + ?` quantifiers, `&Translit(...)`
//! functions, `^`/`$` anchors, `.`, `use variable range` pragmas, `#`
//! comments -- into rule data and ID blocks, with Java's errors.

use std::collections::HashMap;
use std::sync::Arc;

use crate::icu4j::translit::id::{
    self, is_id_part, is_id_start_char, parse_char, skip_white_space,
};
use crate::icu4j::translit::registry;
use crate::icu4j::translit::rules::{Data, Rule, StringReplacer, Var};
use crate::icu4j::translit::{FORWARD, REVERSE};
use crate::icu4j::unicode_set::{
    is_pattern_white_space, unescape_at, SymbolMatcher, SymbolTable, UnicodeSet,
};
use crate::icu4j::utf16;
use crate::{IcuError, IcuErrorKind};

const OPERATORS: &[u16] = &[0x3d, 0x3e, 0x3c, 0x2190, 0x2192, 0x2194];
const HALF_ENDERS: &[u16] = &[0x3d, 0x3e, 0x3c, 0x2190, 0x2192, 0x2194, 0x3b];
const FWDREV: u16 = 0x7e; // '~', the internal <> operator
const DOT_SET: &str = "[^[:Zp:][:Zl:]\\r\\n$]";

/// The result of parsing: the ID blocks and rule data in order, and the
/// compound filter.
#[derive(Debug, Default)]
pub struct Parsed {
    pub id_block_vector: Vec<String>,
    pub data_vector: Vec<Arc<Data>>,
    pub compound_filter: Option<UnicodeSet>,
}

fn icu_illegal(msg: impl Into<String>) -> IcuError {
    IcuError::with_kind(IcuErrorKind::IllegalIcuArgument, msg)
}

/// `TransliteratorParser`.
struct Parser {
    data_vector: Vec<Data>,
    id_block_vector: Vec<String>,
    cur_data: Option<Data>,
    compound_filter: Option<UnicodeSet>,
    direction: i32,
    variables_vector: Vec<Var>,
    variable_names: HashMap<String, Vec<u16>>,
    segment_standins: Vec<u16>,
    segment_objects: Vec<Option<usize>>,
    variable_next: u16,
    variable_limit: u16,
    variables_base: u16,
    undefined_variable_name: Option<String>,
    dot_stand_in: i32,
}

/// `RuleHalf`.
#[derive(Debug, Default)]
struct RuleHalf {
    text: Vec<u16>,
    cursor: i32,
    ante: i32,
    post: i32,
    cursor_offset: i32,
    cursor_offset_pos: i32,
    anchor_start: bool,
    anchor_end: bool,
    next_segment_number: usize,
    /// How deep `parse_section` has recursed into `(...)` and `&F(...)`.
    depth: u32,
}

/// Rust-forced change: how deeply segments and function calls may nest in
/// one rule half. Java recurses until a `StackOverflowError`; a rule nested
/// deeper fails to parse here instead of overflowing the stack.
const MAX_SECTION_DEPTH: u32 = 64;

/// The symbol table a set pattern inside a rule sees (`ParseData`).
struct ParseData<'a> {
    names: &'a HashMap<String, Vec<u16>>,
    vars: &'a [Var],
    base: u16,
}

impl SymbolTable for ParseData<'_> {
    fn lookup(&self, name: &str) -> Option<Vec<u16>> {
        self.names.get(name).cloned()
    }

    fn lookup_matcher(&self, c: i32) -> Option<SymbolMatcher<'_>> {
        let i = usize::try_from(c.wrapping_sub(i32::from(self.base))).ok()?;
        match self.vars.get(i)? {
            Var::Set(s) => Some(SymbolMatcher::Set(s)),
            Var::Pending => None,
            _ => Some(SymbolMatcher::Other),
        }
    }

    fn parse_reference(&self, text: &[u16], pos: usize, limit: usize) -> Option<(String, usize)> {
        parse_reference(text, pos, limit)
    }
}

/// `ParseData.parseReference(text, pos, limit)`: an identifier (start and
/// part tested per UTF-16 unit, as Java does).
fn parse_reference(text: &[u16], pos: usize, limit: usize) -> Option<(String, usize)> {
    let mut i = pos;
    while i < limit.min(text.len()) {
        let c = i32::from(text[i]);
        if (i == pos && !is_id_start_char(c)) || !is_id_part(c) {
            break;
        }
        i = i.saturating_add(1);
    }
    if i == pos {
        return None;
    }
    Some((String::from_utf16_lossy(&text[pos..i]), i))
}

fn syntax_error(msg: &str, rule: &[u16], start: usize) -> IcuError {
    let end = rule_end(rule, start, rule.len());
    icu_illegal(format!(
        "{msg} in \"{}\"",
        String::from_utf16_lossy(rule.get(start..end).unwrap_or(&[]))
    ))
}

/// `ruleEnd(rule, start, limit)`: the next unquoted, unescaped `;`.
fn rule_end(rule: &[u16], start: usize, limit: usize) -> usize {
    let mut i = start;
    while i < limit {
        let c = rule[i];
        if c == '\\' as u16 {
            i = i.saturating_add(1);
        } else if c == '\'' as u16 {
            i = i.saturating_add(1);
            while i < limit && rule[i] != '\'' as u16 {
                i = i.saturating_add(1);
            }
        } else if c == ';' as u16 {
            return i;
        }
        i = i.saturating_add(1);
    }
    limit
}

/// `Utility.parsePattern(rule, pos, limit, pattern, parsedInts)`.
fn parse_pattern(
    rule: &[u16],
    mut pos: usize,
    limit: usize,
    pattern: &str,
    ints: &mut Vec<i32>,
) -> Option<usize> {
    for cpat in pattern.chars() {
        match cpat {
            ' ' => {
                if pos >= limit {
                    return None;
                }
                let c = rule[pos];
                pos = pos.saturating_add(1);
                if !is_pattern_white_space(i32::from(c)) {
                    return None;
                }
                pos = skip_white_space(rule, pos);
            }
            '~' => pos = skip_white_space(rule, pos),
            '#' => {
                let p = pos;
                let (v, np) = parse_integer(rule, p, limit);
                if np == p {
                    return None;
                }
                ints.push(v);
                pos = np;
            }
            _ => {
                if pos >= limit {
                    return None;
                }
                let c = char::from_u32(u32::from(rule[pos])).map_or(rule[pos] as u32, |ch| {
                    ch.to_lowercase().next().map_or(ch as u32, |l| l as u32)
                });
                pos = pos.saturating_add(1);
                if c != cpat as u32 {
                    return None;
                }
            }
        }
    }
    Some(pos)
}

/// `Utility.parseInteger(rule, pos, limit)`: the value and the position
/// after it (unchanged when no digit was read).
// ARITH: overflow is detected as Java detects it (a value not above the
// previous one returns 0).
#[allow(clippy::arithmetic_side_effects)]
fn parse_integer(rule: &[u16], pos: usize, limit: usize) -> (i32, usize) {
    let mut count = 0;
    let mut value: i32 = 0;
    let mut p = pos;
    let mut radix = 10;
    let lower = |i: usize| rule.get(i).map(|&u| (u as u8).to_ascii_lowercase());
    if lower(p) == Some(b'0') && lower(p + 1) == Some(b'x') && rule[p + 1] < 0x80 {
        p += 2;
        radix = 16;
    } else if p < limit && rule[p] == '0' as u16 {
        p += 1;
        count = 1;
        radix = 8;
    }
    while p < limit {
        let d = char::from_u32(u32::from(rule[p])).and_then(|c| c.to_digit(radix));
        p += 1;
        let Some(d) = d else {
            p -= 1;
            break;
        };
        count += 1;
        let v = value.wrapping_mul(radix as i32).wrapping_add(d as i32);
        if v <= value {
            return (0, pos);
        }
        value = v;
    }
    if count > 0 {
        (value, p)
    } else {
        (value, pos)
    }
}

// ARITH: (the whole impl) positions within the rule text and stand-in
// arithmetic within the private-use variable range (checked against its
// limit before each new stand-in).
#[allow(clippy::arithmetic_side_effects)]
impl Parser {
    fn new(direction: i32) -> Self {
        Parser {
            data_vector: Vec::new(),
            id_block_vector: Vec::new(),
            cur_data: None,
            compound_filter: None,
            direction,
            variables_vector: Vec::new(),
            variable_names: HashMap::new(),
            segment_standins: Vec::new(),
            segment_objects: Vec::new(),
            variable_next: 0,
            variable_limit: 0,
            variables_base: 0,
            undefined_variable_name: None,
            dot_stand_in: -1,
        }
    }

    /// `parseRules(ruleArray, dir)` over one rules string.
    fn parse_rules(&mut self, rules: &[u16]) -> Result<(), IcuError> {
        let mut parsing_ids = true;
        let mut rule_count = 0;
        let mut errors: Vec<IcuError> = Vec::new();
        let mut id_block_result = String::new();
        let mut compound_filter_offset = -1;
        let rule = rules;
        let limit = rule.len();
        let mut pos = 0usize;
        while pos < limit {
            let c = rule[pos];
            pos += 1;
            if is_pattern_white_space(i32::from(c)) {
                continue;
            }
            if c == '#' as u16 {
                match rule[pos..].iter().position(|&u| u == '\n' as u16) {
                    Some(i) => {
                        pos += i + 1;
                        continue;
                    }
                    None => break,
                }
            }
            if c == ';' as u16 {
                continue;
            }
            rule_count += 1;
            pos -= 1;
            let result: Result<usize, IcuError> = (|| {
                if pos + 3 <= limit && rule[pos] == ':' as u16 && rule[pos + 1] == ':' as u16 {
                    pos += 2;
                    while pos < limit && is_pattern_white_space(i32::from(rule[pos])) {
                        pos += 1;
                    }
                    let mut p = pos;
                    if !parsing_ids {
                        if let Some(d) = self.cur_data.take() {
                            if self.direction == FORWARD {
                                self.data_vector.push(d);
                            } else {
                                self.data_vector.insert(0, d);
                            }
                        }
                        parsing_ids = true;
                    }
                    let single = id::parse_single_id(rule, &mut p, self.direction)?;
                    if p != pos && parse_char(rule, &mut p, ';') {
                        if let Some(single) = single {
                            if self.direction == FORWARD {
                                id_block_result.push_str(&single.canon_id);
                                id_block_result.push(';');
                            } else {
                                id_block_result.insert_str(0, &format!("{};", single.canon_id));
                            }
                        }
                    } else {
                        let mut with_parens = -1;
                        let f = id::parse_global_filter(
                            rule,
                            &mut p,
                            self.direction,
                            &mut with_parens,
                            None,
                        );
                        if f.is_some() && parse_char(rule, &mut p, ';') {
                            if (self.direction == FORWARD) == (with_parens == 0) {
                                if self.compound_filter.is_some() {
                                    return Err(syntax_error("Multiple global filters", rule, pos));
                                }
                                self.compound_filter = f;
                                compound_filter_offset = rule_count;
                            }
                        } else {
                            return Err(syntax_error("Invalid ::ID", rule, pos));
                        }
                    }
                    Ok(p)
                } else {
                    if parsing_ids {
                        if self.direction == FORWARD {
                            self.id_block_vector.push(id_block_result.clone());
                        } else {
                            self.id_block_vector.insert(0, id_block_result.clone());
                        }
                        id_block_result.clear();
                        parsing_ids = false;
                        self.cur_data = Some(Data::default());
                        self.set_variable_range(0xf000, 0xf8ff)?;
                    }
                    if parse_pattern(rule, pos, limit, "use ", &mut Vec::new()).is_some() {
                        match self.parse_pragma(rule, pos, limit)? {
                            Some(p) => Ok(p),
                            None => Err(syntax_error("Unrecognized pragma", rule, pos)),
                        }
                    } else {
                        self.parse_rule(rule, pos, limit)
                    }
                }
            })();
            match result {
                Ok(p) => pos = p,
                Err(e)
                    if matches!(
                        e.kind(),
                        IcuErrorKind::IllegalArgument | IcuErrorKind::IllegalIcuArgument
                    ) =>
                {
                    if errors.len() == 30 {
                        errors.push(icu_illegal(
                            "\nMore than 30 errors; further messages squelched",
                        ));
                        break;
                    }
                    errors.push(e);
                    pos = rule_end(rule, pos, limit) + 1;
                }
                Err(e) => return Err(e),
            }
        }
        if parsing_ids && !id_block_result.is_empty() {
            if self.direction == FORWARD {
                self.id_block_vector.push(id_block_result);
            } else {
                self.id_block_vector.insert(0, id_block_result);
            }
        } else if !parsing_ids {
            if let Some(d) = self.cur_data.take() {
                if self.direction == FORWARD {
                    self.data_vector.push(d);
                } else {
                    self.data_vector.insert(0, d);
                }
            }
        }
        for d in &mut self.data_vector {
            d.variables = self.variables_vector.clone();
            d.variable_names = self.variable_names.clone();
        }
        let finish: Result<(), IcuError> = (|| {
            if self.compound_filter.is_some()
                && ((self.direction == FORWARD && compound_filter_offset != 1)
                    || (self.direction == REVERSE && compound_filter_offset != rule_count))
            {
                return Err(icu_illegal("Compound filters misplaced"));
            }
            for d in &mut self.data_vector {
                let mut rs = std::mem::take(&mut d.rule_set);
                let r = rs.freeze(d);
                d.rule_set = rs;
                r?;
            }
            if self.id_block_vector.len() == 1 && self.id_block_vector[0].is_empty() {
                self.id_block_vector.remove(0);
            }
            Ok(())
        })();
        if let Err(e) = finish {
            errors.push(e);
        }
        if !errors.is_empty() {
            return Err(errors.swap_remove(0));
        }
        Ok(())
    }

    fn cur_base(&self) -> u16 {
        self.variables_base
    }

    /// `parseRule(rule, pos, limit)`.
    fn parse_rule(&mut self, rule: &[u16], pos: usize, limit: usize) -> Result<usize, IcuError> {
        let start = pos;
        self.segment_standins.clear();
        self.segment_objects.clear();
        let mut left = RuleHalf::new();
        let mut right = RuleHalf::new();
        self.undefined_variable_name = None;
        let mut pos = left.parse(rule, pos, limit, self)?;
        let mut operator;
        if pos == limit || {
            pos -= 1;
            operator = rule[pos];
            !OPERATORS.contains(&operator)
        } {
            return Err(syntax_error(&format!("No operator pos={pos}"), rule, start));
        }
        operator = rule[pos];
        pos += 1;
        if operator == '<' as u16 && pos < limit && rule[pos] == '>' as u16 {
            pos += 1;
            operator = FWDREV;
        }
        operator = match operator {
            0x2192 => '>' as u16,
            0x2190 => '<' as u16,
            0x2194 => FWDREV,
            o => o,
        };
        pos = right.parse(rule, pos, limit, self)?;
        if pos < limit {
            pos -= 1;
            if rule[pos] == ';' as u16 {
                pos += 1;
            } else {
                return Err(syntax_error("Unquoted operator", rule, start));
            }
        }
        if operator == '=' as u16 {
            let Some(name) = self.undefined_variable_name.clone() else {
                return Err(syntax_error(
                    "Missing '$' or duplicate definition",
                    rule,
                    start,
                ));
            };
            if left.text.len() != 1 || left.text[0] != self.variable_limit {
                return Err(syntax_error("Malformed LHS", rule, start));
            }
            if left.anchor_start || left.anchor_end || right.anchor_start || right.anchor_end {
                return Err(syntax_error("Malformed variable def", rule, start));
            }
            self.variable_names.insert(name, right.text.clone());
            self.variable_limit = self.variable_limit.wrapping_add(1);
            return Ok(pos);
        }
        if let Some(name) = &self.undefined_variable_name {
            return Err(syntax_error(
                &format!("Undefined variable ${name}"),
                rule,
                start,
            ));
        }
        if self.segment_standins.len() > self.segment_objects.len() {
            return Err(syntax_error("Undefined segment reference", rule, start));
        }
        if self.segment_standins.contains(&0) || self.segment_objects.iter().any(Option::is_none) {
            return Err(syntax_error("Internal error", rule, start));
        }
        if operator != FWDREV && ((self.direction == FORWARD) != (operator == '>' as u16)) {
            return Ok(pos);
        }
        if self.direction == REVERSE {
            std::mem::swap(&mut left, &mut right);
        }
        if operator == FWDREV {
            right.remove_context();
            left.cursor = -1;
            left.cursor_offset = 0;
        }
        if left.ante < 0 {
            left.ante = 0;
        }
        if left.post < 0 {
            left.post = left.text.len() as i32;
        }
        if right.ante >= 0
            || right.post >= 0
            || left.cursor >= 0
            || (right.cursor_offset != 0 && right.cursor < 0)
            || right.anchor_start
            || right.anchor_end
            || !self.is_valid_input(&left.text)
            || !self.is_valid_output(&right.text)
            || left.ante > left.post
        {
            return Err(syntax_error("Malformed rule", rule, start));
        }
        let segments: Vec<usize> = self.segment_objects.iter().flatten().copied().collect();
        let r = Rule::new(
            left.text,
            left.ante,
            left.post,
            right.text,
            right.cursor,
            right.cursor_offset,
            segments,
            left.anchor_start,
            left.anchor_end,
        )?;
        if let Some(d) = self.cur_data.as_mut() {
            d.rule_set.add_rule(r);
        }
        Ok(pos)
    }

    fn var_index(&self, c: i32) -> Option<usize> {
        usize::try_from(c - i32::from(self.cur_base()))
            .ok()
            .filter(|&i| i < self.variables_vector.len())
    }

    /// `ParseData.isMatcher` over a text.
    fn is_valid_input(&self, text: &[u16]) -> bool {
        let mut i = 0;
        while i < text.len() {
            let c = utf16::code_point_at(text, i);
            i += if c > 0xffff { 2 } else { 1 };
            if let Some(v) = self.var_index(c) {
                if !matches!(
                    self.variables_vector[v],
                    Var::Set(_) | Var::Matcher { .. } | Var::Quantifier { .. }
                ) {
                    return false;
                }
            }
        }
        true
    }

    /// `ParseData.isReplacer` over a text.
    fn is_valid_output(&self, text: &[u16]) -> bool {
        let mut i = 0;
        while i < text.len() {
            let c = utf16::code_point_at(text, i);
            i += if c > 0xffff { 2 } else { 1 };
            if let Some(v) = self.var_index(c) {
                if !matches!(
                    self.variables_vector[v],
                    Var::Matcher { .. } | Var::Function { .. }
                ) {
                    return false;
                }
            }
        }
        true
    }

    /// `setVariableRange(start, end)`.
    fn set_variable_range(&mut self, start: i32, end: i32) -> Result<(), IcuError> {
        if start > end || start < 0 || end > 0xffff {
            return Err(icu_illegal(format!(
                "Invalid variable range {start}, {end}"
            )));
        }
        self.variables_base = start as u16;
        if let Some(d) = self.cur_data.as_mut() {
            d.variables_base = start as u16;
        }
        if self.data_vector.is_empty() {
            self.variable_next = start as u16;
            self.variable_limit = (end + 1) as u16;
        }
        Ok(())
    }

    /// `checkVariableRange(ch, rule, start)`.
    fn check_variable_range(&self, ch: i32, rule: &[u16], start: usize) -> Result<(), IcuError> {
        if ch >= i32::from(self.cur_base()) && ch < i32::from(self.variable_limit) {
            return Err(syntax_error(
                "Variable range character in rule",
                rule,
                start,
            ));
        }
        Ok(())
    }

    /// `parsePragma(rule, pos, limit)`.
    fn parse_pragma(
        &mut self,
        rule: &[u16],
        pos: usize,
        limit: usize,
    ) -> Result<Option<usize>, IcuError> {
        let pos = pos + 4;
        let mut ints = Vec::new();
        if let Some(p) = parse_pattern(rule, pos, limit, "~variable range # #~;", &mut ints) {
            self.set_variable_range(ints[0], ints[1])?;
            return Ok(Some(p));
        }
        ints.clear();
        if parse_pattern(rule, pos, limit, "~maximum backup #~;", &mut ints).is_some() {
            return Err(icu_illegal("use maximum backup pragma not implemented yet"));
        }
        if parse_pattern(rule, pos, limit, "~nfd rules~;", &mut ints).is_some()
            || parse_pattern(rule, pos, limit, "~nfc rules~;", &mut ints).is_some()
        {
            return Err(icu_illegal(
                "use normalize rules pragma not implemented yet",
            ));
        }
        Ok(None)
    }

    /// `parseSet(rule, pos)`.
    fn parse_set(&mut self, rule: &[u16], pos: &mut usize) -> Result<u16, IcuError> {
        let set = {
            let pd = ParseData {
                names: &self.variable_names,
                vars: &self.variables_vector,
                base: self.cur_base(),
            };
            UnicodeSet::parse_at(rule, pos, Some(&pd))?
        };
        if self.variable_next >= self.variable_limit {
            return Err(IcuError::new("Private use variables exhausted"));
        }
        self.generate_stand_in_for(Var::Set(set))
    }

    /// `generateStandInFor(obj)` (a new object every time).
    fn generate_stand_in_for(&mut self, v: Var) -> Result<u16, IcuError> {
        if self.variable_next >= self.variable_limit {
            return Err(IcuError::new("Variable range exhausted"));
        }
        self.variables_vector.push(v);
        let c = self.variable_next;
        self.variable_next += 1;
        Ok(c)
    }

    /// `getSegmentStandin(seg)`.
    fn segment_standin(&mut self, seg: usize) -> Result<u16, IcuError> {
        // Rust-forced change: Java grows its stand-in buffer to `seg` first
        // and runs out of memory for `$999999999`; a segment past what the
        // variable range can number fails here instead.
        if seg > usize::from(self.variable_limit.wrapping_sub(self.cur_base())) {
            return Err(IcuError::new("Variable range exhausted"));
        }
        if self.segment_standins.len() < seg {
            self.segment_standins.resize(seg, 0);
        }
        let mut c = self.segment_standins[seg - 1];
        if c == 0 {
            if self.variable_next >= self.variable_limit {
                return Err(IcuError::new("Variable range exhausted"));
            }
            c = self.variable_next;
            self.variable_next += 1;
            self.variables_vector.push(Var::Pending);
            self.segment_standins[seg - 1] = c;
        }
        Ok(c)
    }

    /// `setSegmentObject(seg, obj)`.
    fn set_segment_object(&mut self, seg: usize, m: Var) -> Result<(), IcuError> {
        while self.segment_objects.len() < seg {
            self.segment_objects.push(None);
        }
        let index = usize::from(self.segment_standin(seg)? - self.cur_base());
        if self.segment_objects[seg - 1].is_some()
            || !matches!(self.variables_vector.get(index), Some(Var::Pending))
        {
            return Err(IcuError::new("segment object set twice"));
        }
        self.segment_objects[seg - 1] = Some(index);
        self.variables_vector[index] = m;
        Ok(())
    }

    /// `getDotStandIn()`.
    fn dot_stand_in(&mut self) -> Result<u16, IcuError> {
        if self.dot_stand_in == -1 {
            let set = UnicodeSet::from_pattern(DOT_SET)?;
            self.dot_stand_in = i32::from(self.generate_stand_in_for(Var::Set(set))?);
        }
        Ok(self.dot_stand_in as u16)
    }

    /// `appendVariableDef(name, buf)`.
    fn append_variable_def(&mut self, name: &str, buf: &mut Vec<u16>) -> Result<(), IcuError> {
        match self.variable_names.get(name) {
            Some(v) => buf.extend_from_slice(v),
            None => {
                if self.undefined_variable_name.is_none() {
                    self.undefined_variable_name = Some(name.to_string());
                    if self.variable_next >= self.variable_limit {
                        return Err(IcuError::new("Private use variables exhausted"));
                    }
                    self.variable_limit -= 1;
                    buf.push(self.variable_limit);
                } else {
                    return Err(icu_illegal(format!("Undefined variable ${name}")));
                }
            }
        }
        Ok(())
    }
}

/// `ILLEGAL_TOP`, `ILLEGAL_SEG`, `ILLEGAL_FUNC`.
fn illegal_top(c: u16) -> bool {
    c == ')' as u16
}
fn illegal_seg(c: u16) -> bool {
    matches!(c, 0x7b | 0x7d | 0x7c | 0x40)
}
fn illegal_func(c: u16) -> bool {
    matches!(
        c,
        0x5e | 0x28 | 0x2e | 0x2a | 0x2b | 0x3f | 0x7b | 0x7d | 0x7c | 0x40
    )
}

// ARITH: (the whole impl) positions within the rule text and its buffer.
#[allow(clippy::arithmetic_side_effects)]
impl RuleHalf {
    fn new() -> Self {
        RuleHalf {
            cursor: -1,
            ante: -1,
            post: -1,
            next_segment_number: 1,
            depth: 0,
            ..Default::default()
        }
    }

    /// `parse(rule, pos, limit, parser)`.
    fn parse(
        &mut self,
        rule: &[u16],
        pos: usize,
        limit: usize,
        parser: &mut Parser,
    ) -> Result<usize, IcuError> {
        let start = pos;
        let mut buf = Vec::new();
        let pos = self.parse_section(rule, pos, limit, parser, &mut buf, illegal_top, false)?;
        self.text = buf;
        if self.cursor_offset > 0 && self.cursor != self.cursor_offset_pos {
            return Err(syntax_error("Misplaced |", rule, start));
        }
        Ok(pos)
    }

    /// `parseSection(rule, pos, limit, parser, buf, illegal, isSegment)`.
    #[allow(clippy::too_many_arguments)]
    fn parse_section(
        &mut self,
        rule: &[u16],
        pos: usize,
        limit: usize,
        parser: &mut Parser,
        buf: &mut Vec<u16>,
        illegal: fn(u16) -> bool,
        is_segment: bool,
    ) -> Result<usize, IcuError> {
        if self.depth >= MAX_SECTION_DEPTH {
            return Err(syntax_error("Segments nested too deeply", rule, pos));
        }
        self.depth += 1;
        let result = self.parse_section_nested(rule, pos, limit, parser, buf, illegal, is_segment);
        self.depth -= 1;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_section_nested(
        &mut self,
        rule: &[u16],
        mut pos: usize,
        limit: usize,
        parser: &mut Parser,
        buf: &mut Vec<u16>,
        illegal: fn(u16) -> bool,
        is_segment: bool,
    ) -> Result<usize, IcuError> {
        let start = pos;
        let mut quote_start: i32 = -1;
        let mut quote_limit: i32 = -1;
        let mut var_start: i32 = -1;
        let mut var_limit: i32 = -1;
        let buf_start = buf.len();
        while pos < limit {
            let mut c = rule[pos];
            pos += 1;
            if is_pattern_white_space(i32::from(c)) {
                continue;
            }
            if HALF_ENDERS.contains(&c) {
                if is_segment {
                    return Err(syntax_error("Unclosed segment", rule, start));
                }
                break;
            }
            if self.anchor_end {
                return Err(syntax_error("Malformed variable reference", rule, start));
            }
            if UnicodeSet::resembles_pattern(rule, pos - 1) {
                let mut pp = pos - 1;
                let s = parser.parse_set(rule, &mut pp)?;
                buf.push(s);
                pos = pp;
                continue;
            }
            if c == '\\' as u16 {
                if pos == limit {
                    return Err(syntax_error("Trailing backslash", rule, start));
                }
                let Some((escaped, len)) = unescape_at(rule, pos) else {
                    return Err(syntax_error("Malformed escape", rule, start));
                };
                pos += len;
                parser.check_variable_range(escaped, rule, start)?;
                utf16::push_code_point(buf, escaped);
                continue;
            }
            if c == '\'' as u16 {
                let find = |from: usize| {
                    rule[from.min(rule.len())..]
                        .iter()
                        .position(|&u| u == '\'' as u16)
                        .map(|i| i + from)
                };
                let mut iq = find(pos);
                if iq == Some(pos) {
                    buf.push(c);
                    pos += 1;
                } else {
                    quote_start = buf.len() as i32;
                    loop {
                        let Some(q) = iq else {
                            return Err(syntax_error("Unterminated quote", rule, start));
                        };
                        buf.extend_from_slice(&rule[pos..q]);
                        pos = q + 1;
                        if pos < limit && rule[pos] == '\'' as u16 {
                            iq = find(pos + 1);
                        } else {
                            break;
                        }
                    }
                    quote_limit = buf.len() as i32;
                    for &unit in &buf[quote_start as usize..quote_limit as usize] {
                        parser.check_variable_range(i32::from(unit), rule, start)?;
                    }
                }
                continue;
            }
            parser.check_variable_range(i32::from(c), rule, start)?;
            if illegal(c) {
                return Err(syntax_error(
                    &format!("Illegal character '{}'", String::from_utf16_lossy(&[c])),
                    rule,
                    start,
                ));
            }
            match c {
                0x5e => {
                    // ^
                    if buf.is_empty() && !self.anchor_start {
                        self.anchor_start = true;
                    } else {
                        return Err(syntax_error("Misplaced anchor start", rule, start));
                    }
                }
                0x28 => {
                    // (
                    let buf_seg_start = buf.len();
                    let segment_number = self.next_segment_number;
                    self.next_segment_number += 1;
                    pos = self.parse_section(rule, pos, limit, parser, buf, illegal_seg, true)?;
                    let m = Var::Matcher {
                        pattern: buf[buf_seg_start..].to_vec(),
                        segment: segment_number,
                    };
                    parser.set_segment_object(segment_number, m)?;
                    buf.truncate(buf_seg_start);
                    let s = parser.segment_standin(segment_number)?;
                    buf.push(s);
                }
                0x26 | 0x2206 => {
                    // & or the increment sign: a function
                    let mut iref = pos;
                    let single = id::parse_filter_id(rule, &mut iref)?;
                    let Some(single) = single.filter(|_| parse_char(rule, &mut iref, '(')) else {
                        return Err(syntax_error("Invalid function", rule, start));
                    };
                    let t = registry::single_instance(&single)
                        .map_err(|_| syntax_error("Invalid function ID", rule, start))?
                        .ok_or_else(|| syntax_error("Invalid function ID", rule, start))?;
                    let buf_seg_start = buf.len();
                    pos = self.parse_section(rule, iref, limit, parser, buf, illegal_func, true)?;
                    let r = Var::Function {
                        translit: Arc::new(t),
                        replacer: StringReplacer {
                            output: buf[buf_seg_start..].to_vec(),
                            cursor_pos: 0,
                            has_cursor: false,
                        },
                    };
                    buf.truncate(buf_seg_start);
                    let s = parser.generate_stand_in_for(r)?;
                    buf.push(s);
                }
                0x24 => {
                    // $
                    if pos == limit {
                        self.anchor_end = true;
                        continue;
                    }
                    c = rule[pos];
                    let r = char::from_u32(u32::from(c)).and_then(|ch| ch.to_digit(10));
                    if let Some(r) = r.filter(|&r| (1..=9).contains(&r)) {
                        let _ = r;
                        let mut p = pos;
                        let mut n: i32 = 0;
                        while p < rule.len() {
                            let Some(d) =
                                char::from_u32(u32::from(rule[p])).and_then(|ch| ch.to_digit(10))
                            else {
                                break;
                            };
                            n = n.wrapping_mul(10).wrapping_add(d as i32);
                            if n < 0 {
                                return Err(syntax_error(
                                    "Undefined segment reference",
                                    rule,
                                    start,
                                ));
                            }
                            p += 1;
                        }
                        pos = p;
                        let s = parser.segment_standin(n as usize)?;
                        buf.push(s);
                    } else {
                        match parse_reference(rule, pos, limit) {
                            None => {
                                self.anchor_end = true;
                                continue;
                            }
                            Some((name, end)) => {
                                pos = end;
                                var_start = buf.len() as i32;
                                parser.append_variable_def(&name, buf)?;
                                var_limit = buf.len() as i32;
                            }
                        }
                    }
                }
                0x2e => {
                    // .
                    let s = parser.dot_stand_in()?;
                    buf.push(s);
                }
                0x2a | 0x2b | 0x3f => {
                    // * + ?
                    if is_segment && buf.len() == buf_start {
                        return Err(syntax_error("Misplaced quantifier", rule, start));
                    }
                    let len = buf.len() as i32;
                    let (qstart, qlimit) = if len == quote_limit {
                        (quote_start, quote_limit)
                    } else if len == var_limit {
                        (var_start, var_limit)
                    } else {
                        (len - 1, len)
                    };
                    // Java's StringMatcher(buf, qstart, qlimit) throws on
                    // an empty range before the quantifier ("* > x").
                    if qstart < 0 || qlimit > len {
                        return Err(icu_illegal(format!(
                            "Failure in rule: {}",
                            utf16::string(&rule[start..pos.min(rule.len())])
                        )));
                    }
                    let (qs, ql) = (qstart as usize, qlimit as usize);
                    let pattern = buf.get(qs..ql).unwrap_or(&[]).to_vec();
                    let (min, max) = match c {
                        0x2b => (1, i32::MAX),
                        0x3f => (0, 1),
                        _ => (0, i32::MAX),
                    };
                    buf.truncate(qs);
                    let s = parser.generate_stand_in_for(Var::Quantifier { pattern, min, max })?;
                    buf.push(s);
                }
                0x29 => break, // )
                0x7b => {
                    // {
                    if self.ante >= 0 {
                        return Err(syntax_error("Multiple ante contexts", rule, start));
                    }
                    self.ante = buf.len() as i32;
                }
                0x7d => {
                    // }
                    if self.post >= 0 {
                        return Err(syntax_error("Multiple post contexts", rule, start));
                    }
                    self.post = buf.len() as i32;
                }
                0x7c => {
                    // |
                    if self.cursor >= 0 {
                        return Err(syntax_error("Multiple cursors", rule, start));
                    }
                    self.cursor = buf.len() as i32;
                }
                0x40 => {
                    // @
                    if self.cursor_offset < 0 {
                        if !buf.is_empty() {
                            return Err(syntax_error("Misplaced @", rule, start));
                        }
                        self.cursor_offset -= 1;
                    } else if self.cursor_offset > 0 {
                        if buf.len() as i32 != self.cursor_offset_pos || self.cursor >= 0 {
                            return Err(syntax_error("Misplaced @", rule, start));
                        }
                        self.cursor_offset += 1;
                    } else if self.cursor == 0 && buf.is_empty() {
                        self.cursor_offset = -1;
                    } else if self.cursor < 0 {
                        self.cursor_offset_pos = buf.len() as i32;
                        self.cursor_offset = 1;
                    } else {
                        return Err(syntax_error("Misplaced @", rule, start));
                    }
                }
                _ => {
                    if (0x21..=0x7e).contains(&c) && !(c as u8).is_ascii_alphanumeric() {
                        return Err(syntax_error(
                            &format!("Unquoted {}", c as u8 as char),
                            rule,
                            start,
                        ));
                    }
                    buf.push(c);
                }
            }
        }
        Ok(pos)
    }

    /// `removeContext()`.
    fn remove_context(&mut self) {
        let a = if self.ante < 0 { 0 } else { self.ante as usize };
        let p = if self.post < 0 {
            self.text.len()
        } else {
            self.post as usize
        };
        self.text = self.text.get(a..p).unwrap_or(&[]).to_vec();
        self.ante = -1;
        self.post = -1;
        self.anchor_start = false;
        self.anchor_end = false;
    }
}

/// `TransliteratorParser.parse(rules, dir)`.
pub fn parse(rules: &str, dir: i32) -> Result<Parsed, IcuError> {
    let units: Vec<u16> = rules.encode_utf16().collect();
    let mut p = Parser::new(dir);
    p.parse_rules(&units)?;
    Ok(Parsed {
        id_block_vector: p.id_block_vector,
        data_vector: p.data_vector.into_iter().map(Arc::new).collect(),
        compound_filter: p.compound_filter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huge_segment_references_fail_without_allocating() {
        for rules in ["$999999999 > x;", "x > $65535;"] {
            assert!(parse(rules, FORWARD).is_err(), "{rules}");
        }
        assert!(parse("(a) > $1;", FORWARD).is_ok());
    }

    #[test]
    fn self_referencing_segments_do_not_overflow_the_stack() {
        // Java recurses until a StackOverflowError; nesting is bounded here.
        for rules in [
            "($1) > x;",
            "($1 a) > x;",
            "(a $1) > x;",
            "($2)($1) > x;",
            "($1)+ > x;",
        ] {
            let t = crate::icu4j::translit::Transliterator::create_from_rules("T", rules, FORWARD)
                .unwrap();
            let _ = t.transliterate("aaaa");
            let _ = t.source_set();
        }
        let deep = format!("{}a{} > x;", "(".repeat(5000), ")".repeat(5000));
        assert!(parse(&deep, FORWARD).is_err());
        let deep = format!("a > {}b{};", "&Any-Upper(".repeat(5000), ")".repeat(5000));
        assert!(parse(&deep, FORWARD).is_err());
    }
}
