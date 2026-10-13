//! `com.ibm.icu.text.RBBIRuleScanner` and `RBBISymbolTable`: the break rules'
//! text, scanned by the state table in `parse_table.rs` into the syntax trees
//! of `node.rs`.
//!
//! Positions are UTF-16 indices into the rules, as Java's. Every Java
//! `error(code)` throws, so here it is an `Err` that unwinds the scan: the
//! message is Java's `IllegalArgumentException("Error <code> at line <n>
//! column <n>")`. Differs: `parseReference`'s identifier test is the
//! general-category definition of `ID_Start`/`ID_Continue` (letters, letter
//! numbers, marks, decimal digits, connector punctuation, and the format and
//! control characters `Character.isIdentifierIgnorable` skips), without the
//! few `Other_ID_*` characters.

use std::cell::Cell;
use std::collections::HashMap;

use super::node::{
    Node, END_MARK, LOOK_AHEAD, OP_CAT, OP_LPAREN, OP_OR, OP_PLUS, OP_QUESTION, OP_STAR, OP_START,
    PREC_LPAREN, PREC_OP_CAT, PREC_START, SET_REF, TAG, USET, VAR_REF,
};
use super::parse_table::*;
use super::RuleBuilder;
use crate::icu4j::unicode_set::{unescape_at, SymbolMatcher, SymbolTable, UnicodeSet};
use crate::icu4j::{uprops, utf16};
use crate::IcuError;

/// `RBBIRuleBuilder.U_ILLEGAL_CHAR_FOUND`.
const U_ILLEGAL_CHAR_FOUND: i32 = 12;
/// `RBBIRuleBuilder.U_BRK_INTERNAL_ERROR`.
const U_BRK_INTERNAL_ERROR: i32 = 66049;
/// `U_BRK_HEX_DIGITS_EXPECTED`.
const U_BRK_HEX_DIGITS_EXPECTED: i32 = 66050;
/// `U_BRK_RULE_SYNTAX`.
const U_BRK_RULE_SYNTAX: i32 = 66052;
/// `U_BRK_ASSIGN_ERROR`.
const U_BRK_ASSIGN_ERROR: i32 = 66054;
/// `U_BRK_VARIABLE_REDEFINITION`.
const U_BRK_VARIABLE_REDEFINITION: i32 = 66055;
/// `U_BRK_MISMATCHED_PAREN`.
const U_BRK_MISMATCHED_PAREN: i32 = 66056;
/// `U_BRK_NEW_LINE_IN_QUOTED_STRING`.
const U_BRK_NEW_LINE_IN_QUOTED_STRING: i32 = 66057;
/// `U_BRK_UNDEFINED_VARIABLE`.
const U_BRK_UNDEFINED_VARIABLE: i32 = 66058;
/// `U_BRK_RULE_EMPTY_SET`.
const U_BRK_RULE_EMPTY_SET: i32 = 66060;
/// `U_BRK_UNRECOGNIZED_OPTION`.
const U_BRK_UNRECOGNIZED_OPTION: i32 = 66061;
/// `U_BRK_MALFORMED_RULE_TAG`.
const U_BRK_MALFORMED_RULE_TAG: i32 = 66062;
/// `U_BRK_MALFORMED_SET`.
const U_BRK_MALFORMED_SET: i32 = 66063;

/// `RBBIRuleScanner.kStackSize`.
const STACK_SIZE: usize = 100;

/// The scanner's built-in character sets, in the order of the parse table's
/// classes 128 to 132 (`fRuleSets`).
const RULE_SET_PATTERNS: [&str; 5] = [
    "[0-9]",
    r"[_\p{L}\p{N}]",
    r"[_\p{L}]",
    // ICU writes the range as a pair of \u escapes, 0020 to 007f; the \x
    // escapes are the same set
    r"[^[\p{Z}\x20-\x7f]-[\p{L}]-[\p{N}]]",
    r"[\p{Pattern_White_Space}]",
];

/// The `rule_char` set's pattern (index 3), which `unquoted_literals`
/// restores.
const RULE_CHAR_INDEX: usize = 3;

/// `RBBIRuleScanner.RBBIRuleChar`.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct RuleChar {
    /// `fChar`; `-1` at the end of the rules.
    pub ch: i32,
    pub escaped: bool,
}

/// The scanner's state (`RBBIRuleScanner`'s fields).
#[derive(Debug)]
pub(super) struct ScannerState {
    pub scan_index: usize,
    pub next_index: usize,
    pub quote_mode: bool,
    pub line_num: i32,
    pub char_num: i32,
    pub last_char: i32,
    pub c: RuleChar,
    pub stack: [u16; STACK_SIZE],
    pub stack_ptr: isize,
    pub node_stack: [Option<usize>; STACK_SIZE],
    pub node_stack_ptr: usize,
    pub reverse_rule: bool,
    pub look_ahead_rule: bool,
    pub no_chain_in_rule: bool,
    /// `RBBISymbolTable.fHashTable`: a variable's name to its defining
    /// `varRef` node.
    pub symbols: HashMap<String, usize>,
    /// `fSetTable`: a set's source text to its `uset` node.
    pub set_table: HashMap<Vec<u16>, usize>,
    pub rule_sets: Vec<UnicodeSet>,
    pub rule_num: i32,
    pub option_start: usize,
}

impl ScannerState {
    pub fn new() -> Result<ScannerState, IcuError> {
        let rule_sets = RULE_SET_PATTERNS
            .iter()
            .map(|p| UnicodeSet::from_pattern(p))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ScannerState {
            scan_index: 0,
            next_index: 0,
            quote_mode: false,
            line_num: 1,
            char_num: 0,
            last_char: 0,
            c: RuleChar::default(),
            stack: [0; STACK_SIZE],
            stack_ptr: 0,
            node_stack: [None; STACK_SIZE],
            node_stack_ptr: 0,
            reverse_rule: false,
            look_ahead_rule: false,
            no_chain_in_rule: false,
            symbols: HashMap::new(),
            set_table: HashMap::new(),
            rule_sets,
            rule_num: 0,
            option_start: 0,
        })
    }
}

/// `RBBISymbolTable` as a [`SymbolTable`]: a variable defined as a set
/// answers with a stand-in character whose matcher is that set.
struct Symbols<'a> {
    entries: &'a HashMap<String, usize>,
    nodes: &'a [Node],
    /// `fCachedSetLookup`: the `uset` node of the last variable looked up.
    cached: Cell<Option<usize>>,
}

/// `Character.isUnicodeIdentifierStart` by general category.
fn is_identifier_start(c: i32) -> bool {
    // Lu, Ll, Lt, Lm, Lo, Nl
    matches!(uprops::char_type(c), 1..=5 | 10)
}

/// `Character.isUnicodeIdentifierPart` by general category, or ignorable.
fn is_identifier_part(c: i32) -> bool {
    // Letters, Mn, Mc, Nd, Nl, Pc, and Character.isIdentifierIgnorable
    matches!(uprops::char_type(c), 1..=10 | 22)
        || (0..=8).contains(&c)
        || (0xe..=0x1b).contains(&c)
        || (0x7f..=0x9f).contains(&c)
        || uprops::char_type(c) == 16
}

impl SymbolTable for Symbols<'_> {
    fn lookup(&self, name: &str) -> Option<Vec<u16>> {
        let mut ret = *self.entries.get(name)?;
        // Walk through any chain of variable-to-variable assignments.
        loop {
            let left = self.nodes[ret].left?;
            if self.nodes[left].kind == VAR_REF {
                ret = left;
            } else {
                break;
            }
        }
        let set_ref = self.nodes[ret].left?;
        if self.nodes[set_ref].kind != SET_REF {
            // Java: error(U_BRK_MALFORMED_SET), which `scanSet` reports
            return None;
        }
        self.cached.set(self.nodes[set_ref].left);
        Some(vec![0xffff])
    }

    fn lookup_matcher(&self, c: i32) -> Option<SymbolMatcher<'_>> {
        if c != 0xffff {
            return None;
        }
        let id = self.cached.take()?;
        self.nodes[id].input_set.as_ref().map(SymbolMatcher::Set)
    }

    fn parse_reference(&self, text: &[u16], pos: usize, limit: usize) -> Option<(String, usize)> {
        let start = pos;
        let mut i = start;
        while i < limit {
            let c = utf16::code_point_at(text, i);
            if (i == start && !is_identifier_start(c)) || !is_identifier_part(c) {
                break;
            }
            i = i.saturating_add(if c > 0xffff { 2 } else { 1 });
        }
        // Java answers "" when no name follows, which then looks up as an
        // undefined variable.
        Some((utf16::string(text.get(start..i).unwrap_or(&[])), i))
    }
}

// ARITH: (the whole impl) positions index the rules or the 100-entry stacks;
// every push is bounds-checked first, every pop is checked against zero, and
// line, column and rule counts are bounded by the length of the rules.
#[allow(clippy::arithmetic_side_effects)]
impl RuleBuilder {
    /// `RBBIRuleScanner.error(code)`.
    pub(super) fn error(&self, code: i32) -> IcuError {
        IcuError::illegal_argument(format!(
            "Error {code} at line {} column {}",
            self.sc.line_num, self.sc.char_num
        ))
    }

    fn internal_error(&self) -> IcuError {
        self.error(U_BRK_INTERNAL_ERROR)
    }

    /// `rules.substring(start, end)`.
    fn substring(&self, start: usize, end: usize) -> String {
        utf16::string(self.rules.get(start..end).unwrap_or(&[]))
    }

    /// `nextCharLL()`: the next code point and its line and column
    /// accounting; `-1` at the end.
    fn next_char_ll(&mut self) -> Result<i32, IcuError> {
        if self.sc.next_index >= self.rules.len() {
            return Ok(-1);
        }
        let c = utf16::code_point_at(&self.rules, self.sc.next_index);
        if (0xd800..0xe000).contains(&c) {
            return Err(self.error(U_ILLEGAL_CHAR_FOUND));
        }
        self.sc.next_index += if c > 0xffff { 2 } else { 1 };
        if c == 13 || c == 0x85 || c == 0x2028 || (c == 10 && self.sc.last_char != 13) {
            self.sc.line_num += 1;
            self.sc.char_num = 0;
            if self.sc.quote_mode {
                return Err(self.error(U_BRK_NEW_LINE_IN_QUOTED_STRING));
            }
        } else if c != 10 {
            self.sc.char_num += 1;
        }
        self.sc.last_char = c;
        Ok(c)
    }

    /// `nextChar(RBBIRuleChar c)`: the next character of the rules into
    /// `self.sc.c`, with quoting, `#` comments and `\` escapes resolved.
    fn next_char(&mut self) -> Result<(), IcuError> {
        self.sc.scan_index = self.sc.next_index;
        self.sc.c.ch = self.next_char_ll()?;
        self.sc.c.escaped = false;
        if self.sc.c.ch == i32::from(b'\'') {
            if self.sc.next_index < self.rules.len()
                && self.rules[self.sc.next_index] == u16::from(b'\'')
            {
                // a doubled quote is a literal quote
                self.sc.c.ch = self.next_char_ll()?;
                self.sc.c.escaped = true;
            } else {
                // a quote starts or ends a quoted string, which reads as a
                // parenthesised group of escaped characters
                self.sc.quote_mode = !self.sc.quote_mode;
                self.sc.c.ch = if self.sc.quote_mode {
                    i32::from(b'(')
                } else {
                    i32::from(b')')
                };
                self.sc.c.escaped = false;
                return Ok(());
            }
        }
        if self.sc.c.ch == -1 {
            return Ok(());
        }
        if self.sc.quote_mode {
            self.sc.c.escaped = true;
            return Ok(());
        }
        if self.sc.c.ch == i32::from(b'#') {
            // a comment runs to the end of the line, which is then the
            // current character
            loop {
                self.sc.c.ch = self.next_char_ll()?;
                if matches!(self.sc.c.ch, -1 | 13 | 10 | 0x85 | 0x2028) {
                    break;
                }
            }
        }
        if self.sc.c.ch == -1 {
            return Ok(());
        }
        if self.sc.c.ch == i32::from(b'\\') {
            self.sc.c.escaped = true;
            let Some((cp, len)) = unescape_at(&self.rules, self.sc.next_index) else {
                return Err(self.error(U_BRK_HEX_DIGITS_EXPECTED));
            };
            self.sc.c.ch = cp;
            self.sc.char_num += i32::try_from(len).unwrap_or(i32::MAX);
            self.sc.next_index += len;
        }
        Ok(())
    }

    /// `parse()`: the whole rules into `self.tree_roots`.
    pub(super) fn parse(&mut self) -> Result<(), IcuError> {
        let mut state: usize = 1;
        self.next_char()?;
        while state != 0 {
            // the first row of the state whose class matches the character
            let mut idx = state;
            let row = loop {
                let Some(row) = TABLE.get(idx) else {
                    return Err(self.internal_error());
                };
                if self.row_matches(row) {
                    break *row;
                }
                idx += 1;
            };
            if !self.do_parse_actions(row.action)? {
                break;
            }
            if row.push_state != 0 {
                self.sc.stack_ptr += 1;
                let ptr = usize::try_from(self.sc.stack_ptr).unwrap_or(STACK_SIZE);
                if ptr >= STACK_SIZE {
                    return Err(self.internal_error());
                }
                self.sc.stack[ptr] = row.push_state;
            }
            if row.next_char {
                self.next_char()?;
            }
            if row.next_state != 255 {
                state = usize::from(row.next_state);
            } else {
                let ptr = usize::try_from(self.sc.stack_ptr).unwrap_or(STACK_SIZE);
                state = usize::from(*self.sc.stack.get(ptr).unwrap_or(&0));
                self.sc.stack_ptr -= 1;
                if self.sc.stack_ptr < 0 {
                    return Err(self.internal_error());
                }
            }
        }
        if self.tree_roots[0].is_none() {
            return Err(self.error(U_BRK_RULE_SYNTAX));
        }
        Ok(())
    }

    /// Whether the current character is in the class of `row`.
    fn row_matches(&self, row: &Row) -> bool {
        let c = &self.sc.c;
        let class = row.char_class;
        if class < 127 && !c.escaped && i32::from(class) == c.ch {
            return true;
        }
        match class {
            255 => true,
            254 => c.escaped,
            253 => c.escaped && (c.ch == i32::from(b'P') || c.ch == i32::from(b'p')),
            252 => c.ch == -1,
            128..=239 => {
                !c.escaped
                    && c.ch != -1
                    && self
                        .sc
                        .rule_sets
                        .get(usize::from(class - 128))
                        .is_some_and(|s| s.contains(c.ch))
            }
            _ => false,
        }
    }

    /// `pushNewNode(type)`.
    fn push_new_node(&mut self, kind: u8) -> Result<usize, IcuError> {
        self.sc.node_stack_ptr += 1;
        if self.sc.node_stack_ptr >= STACK_SIZE {
            return Err(self.internal_error());
        }
        let id = self.arena.new_node(kind);
        self.sc.node_stack[self.sc.node_stack_ptr] = Some(id);
        Ok(id)
    }

    /// `fNodeStack[fNodeStackPtr - back]`.
    fn stack_node(&self, back: usize) -> Result<usize, IcuError> {
        self.sc
            .node_stack_ptr
            .checked_sub(back)
            .and_then(|i| self.sc.node_stack.get(i).copied().flatten())
            .ok_or_else(|| self.internal_error())
    }

    /// `fNodeStack[fNodeStackPtr--]`.
    fn pop_node(&mut self) -> Result<usize, IcuError> {
        let n = self.stack_node(0)?;
        self.sc.node_stack_ptr -= 1;
        Ok(n)
    }

    /// `fixOpStack(p)`: reduces the operators on the node stack whose
    /// precedence is at least `p`; for a parenthesis or the start of an
    /// expression, checks it is the one expected and drops it.
    fn fix_op_stack(&mut self, p: u8) -> Result<(), IcuError> {
        let mut n;
        loop {
            n = self.stack_node(1)?;
            let prec = self.arena.nodes[n].precedence;
            if prec == 0 {
                return Err(self.internal_error());
            }
            if prec < p || prec <= PREC_LPAREN {
                break;
            }
            let top = self.stack_node(0)?;
            self.arena.nodes[n].right = Some(top);
            self.arena.nodes[top].parent = Some(n);
            self.sc.node_stack_ptr -= 1;
        }
        if p <= PREC_LPAREN {
            if self.arena.nodes[n].precedence != p {
                return Err(self.error(U_BRK_MISMATCHED_PAREN));
            }
            let top = self.stack_node(0)?;
            let below = self.sc.node_stack_ptr - 1;
            self.sc.node_stack[below] = Some(top);
            self.sc.node_stack_ptr -= 1;
        }
        Ok(())
    }

    /// `findSetFor(key, node, setToAdopt)`: `node`'s left child is the
    /// `uset` for `key`, made from `adopt` when there is none yet.
    fn find_set_for(
        &mut self,
        key: Vec<u16>,
        node: usize,
        adopt: Option<UnicodeSet>,
    ) -> Result<(), IcuError> {
        if let Some(&u) = self.sc.set_table.get(&key) {
            self.arena.nodes[node].left = Some(u);
            return Ok(());
        }
        let set = match adopt {
            Some(s) => s,
            None if key == utf16::units("any") => UnicodeSet::from_range(0, 0x10ffff),
            None => {
                let c = u32::try_from(utf16::code_point_at(&key, 0)).unwrap_or(0);
                UnicodeSet::from_range(c, c)
            }
        };
        let u = self.arena.new_node(USET);
        self.arena.nodes[u].input_set = Some(set);
        self.arena.nodes[u].parent = Some(node);
        self.arena.nodes[node].left = Some(u);
        self.arena.nodes[u].text = utf16::string(&key);
        self.usets.push(u);
        self.sc.set_table.insert(key, u);
        Ok(())
    }

    /// The node's source text: the rules from its first to its last
    /// position.
    fn set_node_text(&mut self, n: usize) {
        let (a, b) = (self.arena.nodes[n].first_pos, self.arena.nodes[n].last_pos);
        self.arena.nodes[n].text = self.substring(a, b);
    }

    /// `scanSet()`: the `[...]` at the scan index into a set-reference node.
    fn scan_set(&mut self) -> Result<(), IcuError> {
        let start = self.sc.scan_index;
        let mut pos = start;
        let parsed = {
            let symbols = Symbols {
                entries: &self.sc.symbols,
                nodes: &self.arena.nodes,
                cached: Cell::new(None),
            };
            UnicodeSet::parse_at(&self.rules, &mut pos, Some(&symbols))
        };
        let Ok(set) = parsed else {
            return Err(self.error(U_BRK_MALFORMED_SET));
        };
        let mut without_strings = set.clone();
        without_strings.remove_all_strings();
        if without_strings.is_empty() {
            return Err(self.error(U_BRK_RULE_EMPTY_SET));
        }
        while self.sc.next_index < pos {
            self.next_char_ll()?;
        }
        let n = self.push_new_node(SET_REF)?;
        self.arena.nodes[n].first_pos = start;
        self.arena.nodes[n].last_pos = self.sc.next_index;
        self.set_node_text(n);
        let key = self
            .rules
            .get(start..self.sc.next_index)
            .unwrap_or(&[])
            .to_vec();
        self.find_set_for(key, n, Some(set))
    }

    /// `doParseActions(action)`: false stops the scan.
    fn do_parse_actions(&mut self, action: u8) -> Result<bool, IcuError> {
        match action {
            DO_EXPR_START => {
                self.push_new_node(OP_START)?;
                self.sc.rule_num += 1;
            }
            DO_NO_CHAIN => self.sc.no_chain_in_rule = true,
            DO_EXPR_OR_OPERATOR => self.binary_operator(OP_OR)?,
            DO_EXPR_CAT_OPERATOR => self.binary_operator(OP_CAT)?,
            DO_L_PAREN => {
                self.push_new_node(OP_LPAREN)?;
            }
            DO_EXPR_R_PAREN => self.fix_op_stack(PREC_LPAREN)?,
            DO_EXPR_FINISHED | DO_NOP => {}
            DO_START_ASSIGN => {
                let n = self.stack_node(1)?;
                self.arena.nodes[n].first_pos = self.sc.next_index;
                self.push_new_node(OP_START)?;
            }
            DO_END_ASSIGN => {
                self.fix_op_stack(PREC_START)?;
                let start = self.stack_node(2)?;
                let var = self.stack_node(1)?;
                let rhs = self.stack_node(0)?;
                self.arena.nodes[rhs].first_pos = self.arena.nodes[start].first_pos;
                self.arena.nodes[rhs].last_pos = self.sc.scan_index;
                self.set_node_text(rhs);
                self.arena.nodes[var].left = Some(rhs);
                self.arena.nodes[rhs].parent = Some(var);
                let name = self.arena.nodes[var].text.clone();
                if self.sc.symbols.contains_key(&name) {
                    return Err(self.error(U_BRK_VARIABLE_REDEFINITION));
                }
                self.sc.symbols.insert(name, var);
                self.sc.node_stack_ptr -= 3;
            }
            DO_END_OF_RULE => self.end_of_rule()?,
            DO_RULE_ERROR => return Err(self.error(U_BRK_RULE_SYNTAX)),
            DO_VARIABLE_NAME_EXPECTED_ERR => return Err(self.error(U_BRK_RULE_SYNTAX)),
            DO_UNARY_OP_PLUS => self.unary_operator(OP_PLUS)?,
            DO_UNARY_OP_QUESTION => self.unary_operator(OP_QUESTION)?,
            DO_UNARY_OP_STAR => self.unary_operator(OP_STAR)?,
            DO_RULE_CHAR => {
                let n = self.push_new_node(SET_REF)?;
                // Java: `String.valueOf((char) fC.fChar)`, which keeps the
                // low 16 bits of a supplementary character
                let unit = (self.sc.c.ch & 0xffff) as u16;
                self.arena.nodes[n].first_pos = self.sc.scan_index;
                self.arena.nodes[n].last_pos = self.sc.next_index;
                self.set_node_text(n);
                self.find_set_for(vec![unit], n, None)?;
            }
            DO_DOT_ANY => {
                let n = self.push_new_node(SET_REF)?;
                self.arena.nodes[n].first_pos = self.sc.scan_index;
                self.arena.nodes[n].last_pos = self.sc.next_index;
                self.set_node_text(n);
                self.find_set_for(utf16::units("any"), n, None)?;
            }
            DO_SLASH => {
                let n = self.push_new_node(LOOK_AHEAD)?;
                self.arena.nodes[n].val = self.sc.rule_num;
                self.arena.nodes[n].first_pos = self.sc.scan_index;
                self.arena.nodes[n].last_pos = self.sc.next_index;
                self.set_node_text(n);
                self.sc.look_ahead_rule = true;
            }
            DO_START_TAG_VALUE => {
                let n = self.push_new_node(TAG)?;
                self.arena.nodes[n].val = 0;
                self.arena.nodes[n].first_pos = self.sc.scan_index;
                self.arena.nodes[n].last_pos = self.sc.next_index;
            }
            DO_TAG_DIGIT => {
                let n = self.stack_node(0)?;
                let digit = i64::from(self.sc.c.ch - i32::from(b'0'));
                let v = i64::from(self.arena.nodes[n].val) * 10 + digit;
                if v > i64::from(i32::MAX) {
                    return Err(self.error(U_BRK_RULE_SYNTAX));
                }
                self.arena.nodes[n].val = i32::try_from(v).unwrap_or(i32::MAX);
            }
            DO_TAG_VALUE => {
                let n = self.stack_node(0)?;
                self.arena.nodes[n].last_pos = self.sc.next_index;
                self.set_node_text(n);
            }
            DO_TAG_EXPECTED_ERROR => return Err(self.error(U_BRK_MALFORMED_RULE_TAG)),
            DO_OPTION_START => self.sc.option_start = self.sc.scan_index,
            DO_OPTION_END => self.option_end()?,
            DO_REVERSE_DIR => self.sc.reverse_rule = true,
            DO_START_VARIABLE_NAME => {
                let n = self.push_new_node(VAR_REF)?;
                self.arena.nodes[n].first_pos = self.sc.scan_index;
            }
            DO_END_VARIABLE_NAME => {
                let n = self.stack_node(0)?;
                if self.arena.nodes[n].kind != VAR_REF {
                    return Err(self.internal_error());
                }
                self.arena.nodes[n].last_pos = self.sc.scan_index;
                let first = self.arena.nodes[n].first_pos;
                let name = self.substring(first + 1, self.sc.scan_index);
                self.arena.nodes[n].left = self.sc.symbols.get(&name).copied();
                self.arena.nodes[n].text = name;
            }
            DO_CHECK_VAR_DEF => {
                let n = self.stack_node(0)?;
                if self.arena.nodes[n].left.is_none() {
                    return Err(self.error(U_BRK_UNDEFINED_VARIABLE));
                }
            }
            DO_RULE_ERROR_ASSIGN_EXPR => return Err(self.error(U_BRK_ASSIGN_ERROR)),
            DO_EXIT => return Ok(false),
            DO_SCAN_UNICODE_SET => self.scan_set()?,
            _ => return Err(self.internal_error()),
        }
        Ok(true)
    }

    /// `doExprOrOperator` and `doExprCatOperator`: the operand on top of the
    /// stack becomes the left child of a new operator node, pushed.
    fn binary_operator(&mut self, kind: u8) -> Result<(), IcuError> {
        self.fix_op_stack(PREC_OP_CAT)?;
        let operand = self.pop_node()?;
        let op = self.push_new_node(kind)?;
        self.arena.nodes[op].left = Some(operand);
        self.arena.nodes[operand].parent = Some(op);
        Ok(())
    }

    /// `doUnaryOpPlus`, `Question` and `Star`: the top operand under a new
    /// operator node.
    fn unary_operator(&mut self, kind: u8) -> Result<(), IcuError> {
        let operand = self.pop_node()?;
        let op = self.push_new_node(kind)?;
        self.arena.nodes[op].left = Some(operand);
        self.arena.nodes[operand].parent = Some(op);
        Ok(())
    }

    /// `doEndOfRule`: the finished rule joins the rules of its direction.
    fn end_of_rule(&mut self) -> Result<(), IcuError> {
        self.fix_op_stack(PREC_START)?;
        if self.sc.node_stack_ptr != 1 {
            return Err(self.internal_error());
        }
        let mut this_rule = self.stack_node(0)?;
        if self.sc.look_ahead_rule {
            // a look-ahead rule ends in a marker that carries its number
            let end = self.push_new_node(END_MARK)?;
            let cat = self.push_new_node(OP_CAT)?;
            self.sc.node_stack_ptr -= 2;
            self.arena.nodes[cat].left = Some(this_rule);
            self.arena.nodes[cat].right = Some(end);
            let slot = self.sc.node_stack_ptr;
            self.sc.node_stack[slot] = Some(cat);
            self.arena.nodes[end].val = self.sc.rule_num;
            this_rule = cat;
        }
        self.arena.nodes[this_rule].rule_root = true;
        if self.chain_rules && !self.sc.no_chain_in_rule {
            self.arena.nodes[this_rule].chain_in = true;
        }
        // the rule joins the rules so far of its direction (reverse rules
        // are the safe-reverse tree, which this port does not build)
        let dest = if self.sc.reverse_rule {
            3
        } else {
            self.default_tree
        };
        let top = self.stack_node(0)?;
        match self.tree_roots[dest] {
            Some(prev) => {
                let or = self.push_new_node(OP_OR)?;
                self.arena.nodes[or].left = Some(prev);
                self.arena.nodes[prev].parent = Some(or);
                self.arena.nodes[or].right = Some(top);
                self.arena.nodes[top].parent = Some(or);
                self.tree_roots[dest] = Some(or);
            }
            None => self.tree_roots[dest] = Some(top),
        }
        self.sc.reverse_rule = false;
        self.sc.look_ahead_rule = false;
        self.sc.no_chain_in_rule = false;
        self.sc.node_stack_ptr = 0;
        Ok(())
    }

    /// `doOptionEnd`: a `!!option;` setting.
    fn option_end(&mut self) -> Result<(), IcuError> {
        let name = self.substring(self.sc.option_start, self.sc.scan_index);
        match name.as_str() {
            "chain" => self.chain_rules = true,
            "forward" => self.default_tree = 0,
            "reverse" => self.default_tree = 1,
            "safe_forward" => self.default_tree = 2,
            "safe_reverse" => self.default_tree = 3,
            "lookAheadHardBreak" => self.look_ahead_hard_break = true,
            "quoted_literals_only" => self.sc.rule_sets[RULE_CHAR_INDEX].clear(),
            "unquoted_literals" => {
                self.sc.rule_sets[RULE_CHAR_INDEX] =
                    UnicodeSet::from_pattern(RULE_SET_PATTERNS[RULE_CHAR_INDEX])?;
            }
            _ => return Err(self.error(U_BRK_UNRECOGNIZED_OPTION)),
        }
        Ok(())
    }
}
