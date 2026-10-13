//! `org.apache.lucene.queryparser.classic.QueryParserTokenManager` (the
//! JavaCC 7 lexer generated from `QueryParser.jj`) over
//! `org.apache.lucene.queryparser.charstream.FastCharStream`: ported state
//! for state, so the token a query string splits into, and the position and
//! text of a lexical error, are Lucene's. The three lexical states are
//! `Boost` (0, after `^`), `Range` (1, inside `[`/`{`) and `DEFAULT` (2).
//!
//! The generated NFAs also carry arms for states no transition or start
//! ever enters (`DEFAULT`'s 3, 5, 7, 9, 12, 13, 14, 16, 21, 32, 36, 41 and
//! 48, `Range`'s 1): those are left out, which changes nothing.

// The NFA arms keep the generated `switch`'s shape -- one arm per state,
// its tests inside -- so each reads against `QueryParserTokenManager`.
#![allow(clippy::collapsible_match)]

use super::constants::*;

/// A token: its kind, its first and one-past-last UTF-16 column (what
/// `FastCharStream` reports, the line always 1) and its text.
#[derive(Debug, Clone, Default)]
pub(crate) struct Token {
    pub kind: i32,
    pub begin_column: i32,
    pub image: Vec<u16>,
    /// The index of the next token in the parser's token list, once read.
    pub next: Option<usize>,
}

/// `TokenMgrError`: a lexical error, and its message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TokenMgrError(pub String);

/// `FastCharStream` over the whole query in memory: `bufferStart` is always
/// 0, so a column is an offset into the query's UTF-16 units.
#[derive(Debug, Default)]
pub(crate) struct CharStream {
    buf: Vec<u16>,
    pos: usize,
    token_start: usize,
}

impl CharStream {
    pub(crate) fn new(query: &str) -> Self {
        CharStream {
            buf: query.encode_utf16().collect(),
            pos: 0,
            token_start: 0,
        }
    }

    /// `readChar()`; `None` is the `IOException` at the end.
    fn read_char(&mut self) -> Option<i32> {
        let c = *self.buf.get(self.pos)?;
        self.pos += 1;
        Some(i32::from(c))
    }

    /// `BeginToken()`.
    fn begin_token(&mut self) -> Option<i32> {
        self.token_start = self.pos;
        self.read_char()
    }

    fn backup(&mut self, amount: i32) {
        self.pos -= amount as usize;
    }

    fn image(&self) -> Vec<u16> {
        self.buf[self.token_start..self.pos].to_vec()
    }

    fn begin_column(&self) -> i32 {
        self.token_start as i32
    }

    fn end_column(&self) -> i32 {
        self.pos as i32
    }
}

const NO_KIND: i32 = 0x7fff_ffff;

static JJ_BIT_VEC0: [u64; 4] = [0x1, 0x0, 0x0, 0x0];
static JJ_BIT_VEC1: [u64; 4] = [
    0xffff_ffff_ffff_fffe,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
];
static JJ_BIT_VEC3: [u64; 4] = [0x0, 0x0, 0xffff_ffff_ffff_ffff, 0xffff_ffff_ffff_ffff];
static JJ_BIT_VEC4: [u64; 4] = [
    0xfffe_ffff_ffff_fffe,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
];

static JJ_NEXT_STATES: [i32; 32] = [
    37, 39, 40, 17, 18, 20, 42, 43, 45, 46, 31, 22, 23, 25, 26, 24, 25, 26, 45, 46, 31, 44, 47, 35,
    22, 28, 29, 0, 1, 2, 4, 5,
];

/// `jjstrLiteralImages`: the fixed text of a literal token kind.
fn literal_image(kind: i32) -> Option<&'static str> {
    Some(match kind {
        0 => "",
        PLUS => "+",
        MINUS => "-",
        LPAREN => "(",
        RPAREN => ")",
        COLON => ":",
        STAR => "*",
        CARAT => "^",
        RANGEIN_START => "[",
        RANGEEX_START => "{",
        RANGE_TO => "TO",
        RANGEIN_END => "]",
        RANGEEX_END => "}",
        _ => return None,
    })
}

/// `jjnewLexState`.
// SENTINEL: `-1` = the kind leaves the lexical state as it is.
fn new_lex_state(kind: i32) -> i32 {
    match kind {
        CARAT => BOOST,
        RANGEIN_START | RANGEEX_START => RANGE,
        NUMBER | RANGEIN_END | RANGEEX_END => DEFAULT,
        _ => -1,
    }
}

/// `jjtoToken` (`0x1ffffff01`): every kind but the skipped whitespace (7).
fn is_token(kind: i32) -> bool {
    kind == 0 || (8..=32).contains(&kind)
}

fn can_move_0(hi_byte: i32, i2: usize, l2: u64) -> bool {
    match hi_byte {
        48 => JJ_BIT_VEC0[i2] & l2 != 0,
        _ => false,
    }
}

fn can_move_1(hi_byte: i32, i1: usize, i2: usize, l1: u64, l2: u64) -> bool {
    match hi_byte {
        0 => JJ_BIT_VEC3[i2] & l2 != 0,
        _ => JJ_BIT_VEC1[i1] & l1 != 0,
    }
}

fn can_move_2(hi_byte: i32, i1: usize, i2: usize, l1: u64, l2: u64) -> bool {
    match hi_byte {
        0 => JJ_BIT_VEC3[i2] & l2 != 0,
        48 => JJ_BIT_VEC1[i2] & l2 != 0,
        _ => JJ_BIT_VEC4[i1] & l1 != 0,
    }
}

/// `TokenMgrError.addEscapes` (and `ParseException.add_escapes`).
pub(crate) fn add_escapes(s: &[u16]) -> String {
    let mut out = String::new();
    for &c in s {
        match c {
            0x08 => out.push_str("\\b"),
            0x09 => out.push_str("\\t"),
            0x0a => out.push_str("\\n"),
            0x0c => out.push_str("\\f"),
            0x0d => out.push_str("\\r"),
            0x22 => out.push_str("\\\""),
            0x27 => out.push_str("\\'"),
            0x5c => out.push_str("\\\\"),
            c if !(0x20..=0x7e).contains(&c) => out.push_str(&format!("\\u{c:04x}")),
            c => out.push(char::from(c as u8)),
        }
    }
    out
}

/// `QueryParserTokenManager`.
pub(crate) struct TokenManager {
    input: CharStream,
    cur_char: i32,
    cur_lex_state: i32,
    jjnew_state_cnt: usize,
    jjround: i32,
    jjmatched_pos: i32,
    jjmatched_kind: i32,
    jjrounds: [i32; 49],
    jjstate_set: [i32; 98],
}

impl TokenManager {
    pub(crate) fn new(input: CharStream) -> Self {
        let mut tm = TokenManager {
            input,
            cur_char: 0,
            cur_lex_state: DEFAULT,
            jjnew_state_cnt: 0,
            jjround: 0,
            jjmatched_pos: 0,
            jjmatched_kind: 0,
            jjrounds: [0; 49],
            jjstate_set: [0; 98],
        };
        tm.re_init_rounds();
        tm
    }

    fn re_init_rounds(&mut self) {
        self.jjround = 0x8000_0001_u32 as i32;
        self.jjrounds = [i32::MIN; 49];
    }

    fn read(&mut self) -> bool {
        match self.input.read_char() {
            Some(c) => {
                self.cur_char = c;
                true
            }
            None => false,
        }
    }

    fn push(&mut self, state: i32) {
        self.jjstate_set[self.jjnew_state_cnt] = state;
        self.jjnew_state_cnt += 1;
    }

    /// `jjCheckNAdd`.
    fn check_n_add(&mut self, state: i32) {
        if self.jjrounds[state as usize] != self.jjround {
            self.push(state);
            self.jjrounds[state as usize] = self.jjround;
        }
    }

    /// `jjAddStates`.
    fn add_states(&mut self, start: usize, end: usize) {
        for &state in &JJ_NEXT_STATES[start..=end] {
            self.push(state);
        }
    }

    /// `jjCheckNAddTwoStates`.
    fn two(&mut self, s1: i32, s2: i32) {
        self.check_n_add(s1);
        self.check_n_add(s2);
    }

    /// `jjCheckNAddStates`.
    fn states(&mut self, start: usize, end: usize) {
        for &state in &JJ_NEXT_STATES[start..=end] {
            self.check_n_add(state);
        }
    }

    /// `jjStopAtPos`.
    fn stop_at_pos(&mut self, pos: i32, kind: i32) -> i32 {
        self.jjmatched_kind = kind;
        self.jjmatched_pos = pos;
        pos + 1
    }

    /// The round bookkeeping that ends each step of every `jjMoveNfa_*`:
    /// records `kind`, advances, and swaps the two halves of the state set.
    /// `Err(cur_pos)` returns from the NFA.
    fn nfa_step(
        &mut self,
        kind: &mut i32,
        cur_pos: &mut i32,
        i: &mut usize,
        starts_at: &mut usize,
        size: usize,
    ) -> Result<(), i32> {
        if *kind != NO_KIND {
            self.jjmatched_kind = *kind;
            self.jjmatched_pos = *cur_pos;
            *kind = NO_KIND;
        }
        *cur_pos += 1;
        *i = self.jjnew_state_cnt;
        let old = *starts_at;
        self.jjnew_state_cnt = old;
        *starts_at = size - old;
        if *i == *starts_at {
            return Err(*cur_pos);
        }
        if !self.read() {
            return Err(*cur_pos);
        }
        Ok(())
    }

    fn next_round(&mut self) {
        self.jjround = self.jjround.wrapping_add(1);
        if self.jjround == 0x7fff_ffff {
            self.re_init_rounds();
        }
    }

    // ------------------------------------------------------ DEFAULT (2)

    fn move_string_literal_dfa0_2(&mut self) -> i32 {
        match self.cur_char {
            40 => self.stop_at_pos(0, LPAREN),
            41 => self.stop_at_pos(0, RPAREN),
            42 => self.start_nfa_with_states_2(0, STAR, 49),
            43 => self.start_nfa_with_states_2(0, PLUS, 15),
            45 => self.start_nfa_with_states_2(0, MINUS, 15),
            58 => self.stop_at_pos(0, COLON),
            91 => self.stop_at_pos(0, RANGEIN_START),
            94 => self.stop_at_pos(0, CARAT),
            123 => self.stop_at_pos(0, RANGEEX_START),
            _ => self.move_nfa_2(0, 0),
        }
    }

    fn start_nfa_with_states_2(&mut self, pos: i32, kind: i32, state: i32) -> i32 {
        self.jjmatched_kind = kind;
        self.jjmatched_pos = pos;
        if !self.read() {
            return pos + 1;
        }
        self.move_nfa_2(state, pos + 1)
    }

    fn move_nfa_2(&mut self, start_state: i32, mut cur_pos: i32) -> i32 {
        let mut starts_at = 0usize;
        self.jjnew_state_cnt = 49;
        let mut i = 1usize;
        self.jjstate_set[0] = start_state;
        let mut kind = NO_KIND;
        loop {
            self.next_round();
            let c = self.cur_char;
            if c < 64 {
                let l = 1u64 << c;
                loop {
                    i -= 1;
                    match self.jjstate_set[i] {
                        49 | 33 => {
                            if 0xfbff_7cf8_ffff_d9ff & l != 0 {
                                kind = kind.min(23);
                                self.two(33, 34);
                            }
                        }
                        0 => {
                            if 0xfbff_54f8_ffff_d9ff & l != 0 {
                                kind = kind.min(23);
                                self.two(33, 34);
                            } else if 0x1_0000_2600 & l != 0 {
                                kind = kind.min(7);
                            } else if 0x2802_0000_0000 & l != 0 {
                                self.push(15);
                            } else if c == 47 {
                                self.states(0, 2);
                            } else if c == 34 {
                                self.states(3, 5);
                            }
                            if 0x7bff_50f8_ffff_d9ff & l != 0 {
                                kind = kind.min(20);
                                self.states(6, 10);
                            } else if c == 42 {
                                kind = kind.min(22);
                            } else if c == 33 {
                                kind = kind.min(10);
                            }
                            if c == 38 {
                                self.push(4);
                            }
                        }
                        4 => {
                            if c == 38 {
                                kind = kind.min(8);
                            }
                        }
                        15 => {
                            if 0x1_0000_2600 & l != 0 {
                                kind = kind.min(13);
                            }
                        }
                        17 => {
                            if 0xffff_fffb_ffff_ffff & l != 0 {
                                self.states(3, 5);
                            }
                        }
                        19 => self.states(3, 5),
                        20 => {
                            if c == 34 {
                                kind = kind.min(19);
                            }
                        }
                        22 => {
                            if 0x3ff_0000_0000_0000 & l != 0 {
                                kind = kind.min(21);
                                self.states(11, 14);
                            }
                        }
                        23 => {
                            if c == 46 {
                                self.check_n_add(24);
                            }
                        }
                        24 => {
                            if 0x3ff_0000_0000_0000 & l != 0 {
                                kind = kind.min(21);
                                self.states(15, 17);
                            }
                        }
                        25 => {
                            if 0x7bff_78f8_ffff_d9ff & l != 0 {
                                kind = kind.min(21);
                                self.two(25, 26);
                            }
                        }
                        27 => {
                            kind = kind.min(21);
                            self.two(25, 26);
                        }
                        28 => {
                            if 0x7bff_78f8_ffff_d9ff & l != 0 {
                                kind = kind.min(21);
                                self.two(28, 29);
                            }
                        }
                        30 => {
                            kind = kind.min(21);
                            self.two(28, 29);
                        }
                        31 => {
                            if c == 42 {
                                kind = kind.min(22);
                            }
                        }
                        35 => {
                            kind = kind.min(23);
                            self.two(33, 34);
                        }
                        38 => {
                            if c == 47 {
                                self.states(0, 2);
                            }
                        }
                        37 => {
                            if 0xffff_7fff_ffff_ffff & l != 0 {
                                self.states(0, 2);
                            }
                        }
                        40 => {
                            if c == 47 {
                                kind = kind.min(24);
                            }
                        }
                        42 => {
                            if 0x7bff_78f8_ffff_d9ff & l != 0 {
                                kind = kind.min(20);
                                self.two(42, 43);
                            }
                        }
                        44 => {
                            kind = kind.min(20);
                            self.two(42, 43);
                        }
                        45 => {
                            if 0x7bff_78f8_ffff_d9ff & l != 0 {
                                self.states(18, 20);
                            }
                        }
                        47 => self.states(18, 20),
                        _ => {}
                    }
                    if i == starts_at {
                        break;
                    }
                }
            } else if c < 128 {
                let l = 1u64 << (c & 0o77);
                const TERM: u64 = 0x97ff_ffff_87ff_ffff;
                loop {
                    i -= 1;
                    match self.jjstate_set[i] {
                        49 => {
                            if TERM & l != 0 {
                                kind = kind.min(23);
                                self.two(33, 34);
                            } else if c == 92 {
                                self.check_n_add(35);
                            }
                        }
                        0 => {
                            if TERM & l != 0 {
                                kind = kind.min(20);
                                self.states(6, 10);
                            } else if c == 92 {
                                self.states(21, 23);
                            } else if c == 126 {
                                kind = kind.min(21);
                                self.states(24, 26);
                            }
                            if TERM & l != 0 {
                                kind = kind.min(23);
                                self.two(33, 34);
                            }
                            if c == 78 {
                                self.push(11);
                            } else if c == 124 {
                                self.push(8);
                            } else if c == 79 {
                                self.push(6);
                            } else if c == 65 {
                                self.push(2);
                            }
                        }
                        1 => {
                            if c == 68 {
                                kind = kind.min(8);
                            }
                        }
                        2 => {
                            if c == 78 {
                                self.push(1);
                            }
                        }
                        6 => {
                            if c == 82 {
                                kind = kind.min(9);
                            }
                        }
                        8 => {
                            if c == 124 {
                                kind = kind.min(9);
                            }
                        }
                        10 => {
                            if c == 84 {
                                kind = kind.min(10);
                            }
                        }
                        11 => {
                            if c == 79 {
                                self.push(10);
                            }
                        }
                        17 => {
                            if 0xffff_ffff_efff_ffff & l != 0 {
                                self.states(3, 5);
                            }
                        }
                        18 => {
                            if c == 92 {
                                self.push(19);
                            }
                        }
                        19 => self.states(3, 5),
                        25 => {
                            if TERM & l != 0 {
                                kind = kind.min(21);
                                self.two(25, 26);
                            }
                        }
                        26 => {
                            if c == 92 {
                                self.push(27);
                            }
                        }
                        27 => {
                            kind = kind.min(21);
                            self.two(25, 26);
                        }
                        28 => {
                            if TERM & l != 0 {
                                kind = kind.min(21);
                                self.two(28, 29);
                            }
                        }
                        29 => {
                            if c == 92 {
                                self.push(30);
                            }
                        }
                        30 => {
                            kind = kind.min(21);
                            self.two(28, 29);
                        }
                        33 => {
                            if TERM & l != 0 {
                                kind = kind.min(23);
                                self.two(33, 34);
                            }
                        }
                        34 => {
                            if c == 92 {
                                self.check_n_add(35);
                            }
                        }
                        35 => {
                            kind = kind.min(23);
                            self.two(33, 34);
                        }
                        37 => self.add_states(0, 2),
                        39 => {
                            if c == 92 {
                                self.push(38);
                            }
                        }
                        42 => {
                            if TERM & l != 0 {
                                kind = kind.min(20);
                                self.two(42, 43);
                            }
                        }
                        43 => {
                            if c == 92 {
                                self.check_n_add(44);
                            }
                        }
                        44 => {
                            kind = kind.min(20);
                            self.two(42, 43);
                        }
                        45 => {
                            if TERM & l != 0 {
                                self.states(18, 20);
                            }
                        }
                        46 => {
                            if c == 92 {
                                self.check_n_add(47);
                            }
                        }
                        47 => self.states(18, 20),
                        _ => {}
                    }
                    if i == starts_at {
                        break;
                    }
                }
            } else {
                let hi_byte = c >> 8;
                let i1 = (hi_byte >> 6) as usize;
                let l1 = 1u64 << (hi_byte & 0o77);
                let i2 = ((c & 0xff) >> 6) as usize;
                let l2 = 1u64 << (c & 0o77);
                let can0 = can_move_0(hi_byte, i2, l2);
                let can1 = can_move_1(hi_byte, i1, i2, l1, l2);
                let can2 = can_move_2(hi_byte, i1, i2, l1, l2);
                loop {
                    i -= 1;
                    match self.jjstate_set[i] {
                        49 | 33 => {
                            if can2 {
                                kind = kind.min(23);
                                self.two(33, 34);
                            }
                        }
                        0 => {
                            if can0 {
                                kind = kind.min(7);
                            }
                            if can2 {
                                kind = kind.min(23);
                                self.two(33, 34);
                            }
                            if can2 {
                                kind = kind.min(20);
                                self.states(6, 10);
                            }
                        }
                        15 => {
                            if can0 {
                                kind = kind.min(13);
                            }
                        }
                        17 | 19 => {
                            if can1 {
                                self.states(3, 5);
                            }
                        }
                        25 => {
                            if can2 {
                                kind = kind.min(21);
                                self.two(25, 26);
                            }
                        }
                        27 => {
                            if can1 {
                                kind = kind.min(21);
                                self.two(25, 26);
                            }
                        }
                        28 => {
                            if can2 {
                                kind = kind.min(21);
                                self.two(28, 29);
                            }
                        }
                        30 => {
                            if can1 {
                                kind = kind.min(21);
                                self.two(28, 29);
                            }
                        }
                        35 => {
                            if can1 {
                                kind = kind.min(23);
                                self.two(33, 34);
                            }
                        }
                        37 => {
                            if can1 {
                                self.add_states(0, 2);
                            }
                        }
                        42 => {
                            if can2 {
                                kind = kind.min(20);
                                self.two(42, 43);
                            }
                        }
                        44 => {
                            if can1 {
                                kind = kind.min(20);
                                self.two(42, 43);
                            }
                        }
                        45 => {
                            if can2 {
                                self.states(18, 20);
                            }
                        }
                        47 => {
                            if can1 {
                                self.states(18, 20);
                            }
                        }
                        _ => {}
                    }
                    if i == starts_at {
                        break;
                    }
                }
            }
            if let Err(pos) = self.nfa_step(&mut kind, &mut cur_pos, &mut i, &mut starts_at, 49) {
                return pos;
            }
        }
    }

    // -------------------------------------------------------- Boost (0)

    fn move_nfa_0(&mut self, start_state: i32, mut cur_pos: i32) -> i32 {
        let mut starts_at = 0usize;
        self.jjnew_state_cnt = 3;
        let mut i = 1usize;
        self.jjstate_set[0] = start_state;
        let mut kind = NO_KIND;
        loop {
            self.next_round();
            let c = self.cur_char;
            if c < 64 {
                let l = 1u64 << c;
                loop {
                    i -= 1;
                    match self.jjstate_set[i] {
                        0 => {
                            if 0x3ff_0000_0000_0000 & l != 0 {
                                kind = kind.min(27);
                                self.add_states(27, 28);
                            }
                        }
                        1 => {
                            if c == 46 {
                                self.check_n_add(2);
                            }
                        }
                        2 => {
                            if 0x3ff_0000_0000_0000 & l != 0 {
                                kind = kind.min(27);
                                self.check_n_add(2);
                            }
                        }
                        _ => {}
                    }
                    if i == starts_at {
                        break;
                    }
                }
            } else {
                // no state moves on these characters
                i = starts_at;
            }
            if let Err(pos) = self.nfa_step(&mut kind, &mut cur_pos, &mut i, &mut starts_at, 3) {
                return pos;
            }
        }
    }

    // -------------------------------------------------------- Range (1)

    /// `jjStopStringLiteralDfa_1`: the NFA state to go on in.
    // SENTINEL: `-1` = none: `jjMoveNfa_1` starts from no state, which
    // matches nothing.
    fn stop_string_literal_dfa_1(&mut self, pos: i32, active0: u64) -> i32 {
        if pos == 0 && active0 & 0x1000_0000 != 0 {
            self.jjmatched_kind = RANGE_GOOP;
            return 6;
        }
        -1
    }

    fn move_string_literal_dfa0_1(&mut self) -> i32 {
        match self.cur_char {
            84 => self.move_string_literal_dfa1_1(0x1000_0000),
            93 => self.stop_at_pos(0, RANGEIN_END),
            125 => self.stop_at_pos(0, RANGEEX_END),
            _ => self.move_nfa_1(0, 0),
        }
    }

    fn move_string_literal_dfa1_1(&mut self, active0: u64) -> i32 {
        if !self.read() {
            self.stop_string_literal_dfa_1(0, active0);
            return 1;
        }
        if self.cur_char == 79 && active0 & 0x1000_0000 != 0 {
            return self.start_nfa_with_states_1(1, RANGE_TO, 6);
        }
        let state = self.stop_string_literal_dfa_1(0, active0);
        self.move_nfa_1(state, 1)
    }

    fn start_nfa_with_states_1(&mut self, pos: i32, kind: i32, state: i32) -> i32 {
        self.jjmatched_kind = kind;
        self.jjmatched_pos = pos;
        if !self.read() {
            return pos + 1;
        }
        self.move_nfa_1(state, pos + 1)
    }

    fn move_nfa_1(&mut self, start_state: i32, mut cur_pos: i32) -> i32 {
        let mut starts_at = 0usize;
        self.jjnew_state_cnt = 7;
        let mut i = 1usize;
        self.jjstate_set[0] = start_state;
        let mut kind = NO_KIND;
        loop {
            self.next_round();
            let c = self.cur_char;
            if c < 64 {
                let l = 1u64 << c;
                loop {
                    i -= 1;
                    match self.jjstate_set[i] {
                        0 => {
                            if 0xffff_fffe_ffff_ffff & l != 0 {
                                kind = kind.min(32);
                                self.check_n_add(6);
                            }
                            if 0x1_0000_2600 & l != 0 {
                                kind = kind.min(7);
                            } else if c == 34 {
                                self.two(2, 4);
                            }
                        }
                        2 => {
                            if 0xffff_fffb_ffff_ffff & l != 0 {
                                self.states(29, 31);
                            }
                        }
                        3 => {
                            if c == 34 {
                                self.states(29, 31);
                            }
                        }
                        5 => {
                            if c == 34 {
                                kind = kind.min(31);
                            }
                        }
                        6 => {
                            if 0xffff_fffe_ffff_ffff & l != 0 {
                                kind = kind.min(32);
                                self.check_n_add(6);
                            }
                        }
                        _ => {}
                    }
                    if i == starts_at {
                        break;
                    }
                }
            } else if c < 128 {
                let l = 1u64 << (c & 0o77);
                loop {
                    i -= 1;
                    match self.jjstate_set[i] {
                        0 | 6 => {
                            if 0xdfff_ffff_dfff_ffff & l != 0 {
                                kind = kind.min(32);
                                self.check_n_add(6);
                            }
                        }
                        2 => self.add_states(29, 31),
                        4 => {
                            if c == 92 {
                                self.push(3);
                            }
                        }
                        _ => {}
                    }
                    if i == starts_at {
                        break;
                    }
                }
            } else {
                let hi_byte = c >> 8;
                let i1 = (hi_byte >> 6) as usize;
                let l1 = 1u64 << (hi_byte & 0o77);
                let i2 = ((c & 0xff) >> 6) as usize;
                let l2 = 1u64 << (c & 0o77);
                let can0 = can_move_0(hi_byte, i2, l2);
                let can1 = can_move_1(hi_byte, i1, i2, l1, l2);
                loop {
                    i -= 1;
                    match self.jjstate_set[i] {
                        0 => {
                            if can0 {
                                kind = kind.min(7);
                            }
                            if can1 {
                                kind = kind.min(32);
                                self.check_n_add(6);
                            }
                        }
                        2 => {
                            if can1 {
                                self.add_states(29, 31);
                            }
                        }
                        6 => {
                            if can1 {
                                kind = kind.min(32);
                                self.check_n_add(6);
                            }
                        }
                        _ => {}
                    }
                    if i == starts_at {
                        break;
                    }
                }
            }
            if let Err(pos) = self.nfa_step(&mut kind, &mut cur_pos, &mut i, &mut starts_at, 7) {
                return pos;
            }
        }
    }

    // ---------------------------------------------------------- tokens

    /// `jjFillToken()`.
    fn fill_token(&self) -> Token {
        let image = match literal_image(self.jjmatched_kind) {
            Some(s) => s.encode_utf16().collect(),
            None => self.input.image(),
        };
        Token {
            kind: self.jjmatched_kind,
            begin_column: self.input.begin_column(),
            image,
            next: None,
        }
    }

    /// `getNextToken()`.
    pub(crate) fn next_token(&mut self) -> Result<Token, TokenMgrError> {
        loop {
            match self.input.begin_token() {
                Some(c) => self.cur_char = c,
                None => {
                    self.jjmatched_kind = 0;
                    self.jjmatched_pos = -1;
                    return Ok(self.fill_token());
                }
            }
            self.jjmatched_kind = NO_KIND;
            self.jjmatched_pos = 0;
            let cur_pos = match self.cur_lex_state {
                BOOST => self.move_nfa_0(0, 0),
                RANGE => self.move_string_literal_dfa0_1(),
                _ => self.move_string_literal_dfa0_2(),
            };
            if self.jjmatched_kind != NO_KIND {
                if self.jjmatched_pos + 1 < cur_pos {
                    self.input.backup(cur_pos - self.jjmatched_pos - 1);
                }
                let kind = self.jjmatched_kind;
                let new_state = new_lex_state(kind);
                if is_token(kind) {
                    let token = self.fill_token();
                    if new_state != -1 {
                        self.cur_lex_state = new_state;
                    }
                    return Ok(token);
                }
                if new_state != -1 {
                    self.cur_lex_state = new_state;
                }
                continue;
            }
            return Err(self.lexical_error(cur_pos));
        }
    }

    /// The `TokenMgrError` `getNextToken` throws when no token matches.
    fn lexical_error(&mut self, cur_pos: i32) -> TokenMgrError {
        let mut error_line = 1;
        let mut error_column = self.input.end_column();
        let eof_seen;
        let error_after;
        if self.input.read_char().is_some() {
            self.input.backup(1);
            eof_seen = false;
            self.input.backup(1);
            error_after = if cur_pos <= 1 {
                Vec::new()
            } else {
                self.input.image()
            };
        } else {
            eof_seen = true;
            error_after = if cur_pos <= 1 {
                Vec::new()
            } else {
                self.input.image()
            };
            if self.cur_char == 0x0a || self.cur_char == 0x0d {
                error_line += 1;
                error_column = 0;
            } else {
                error_column += 1;
            }
        }
        // `LexicalErr`: `String.valueOf(curChar)` of an `int`, so the
        // character is written as its decimal code.
        let encountered = if eof_seen {
            "<EOF>".to_string()
        } else {
            format!("'{}' ({}),", self.cur_char, self.cur_char)
        };
        let after = if error_after.is_empty() {
            String::new()
        } else {
            format!(" after prefix \"{}\"", add_escapes(&error_after))
        };
        let state = if self.cur_lex_state == 0 {
            String::new()
        } else {
            format!(" (in lexical state {})", self.cur_lex_state)
        };
        TokenMgrError(format!(
            "Lexical error at line {error_line}, column {error_column}.  Encountered: {encountered}{after}{state}"
        ))
    }
}
