//! `org.apache.lucene.queryparser.surround.parser.QueryParserTokenManager`
//! (the JavaCC 7 lexer generated from the surround `QueryParser.jj`), state
//! for state. Two lexical states: `Boost` (0, after `^`) and `DEFAULT` (1).
//! A term character is anything but white space and `, ? * ( ) : ^ "`.
//!
//! The generated NFA also carries arms for states no transition or start
//! enters (alternative start states: 2, 4, 7, 10, 13, 16, 19, 22, 23 and
//! 31): those are left out, which changes nothing.

#![allow(clippy::collapsible_match)]

use std::ops::{Deref, DerefMut};

use super::constants::*;
use crate::queryparser::javacc::{CharStream, Dfa, Lexer, NfaCore, Token, TokenMgrError, NO_KIND};

static JJ_BIT_VEC0: [u64; 4] = [
    0xffff_ffff_ffff_fffe,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
];
static JJ_BIT_VEC2: [u64; 4] = [0x0, 0x0, 0xffff_ffff_ffff_ffff, 0xffff_ffff_ffff_ffff];

/// `jjnextStates`.
static JJ_NEXT_STATES: [i32; 19] = [
    32, 33, 34, 35, 37, 24, 27, 28, 20, 17, 21, 18, 27, 28, 30, 24, 25, 0, 1,
];

/// The states of the term NFA that a character outside the Basic Latin and
/// Latin-1 planes below U+0100 moves (`jjCanMove_0`).
fn can_move_0(hi_byte: i32, i1: usize, i2: usize, l1: u64, l2: u64) -> bool {
    match hi_byte {
        0 => JJ_BIT_VEC2[i2] & l2 != 0,
        _ => JJ_BIT_VEC0[i1] & l1 != 0,
    }
}

pub(crate) struct TokenManager {
    core: NfaCore,
}

impl Deref for TokenManager {
    type Target = NfaCore;
    fn deref(&self) -> &NfaCore {
        &self.core
    }
}

impl DerefMut for TokenManager {
    fn deref_mut(&mut self) -> &mut NfaCore {
        &mut self.core
    }
}

impl TokenManager {
    pub(crate) fn new(input: CharStream) -> Self {
        TokenManager {
            core: NfaCore::new(input, 38, &JJ_NEXT_STATES, DEFAULT),
        }
    }

    // ------------------------------------------------------- DEFAULT (1)

    fn move_string_literal_dfa0_1(&mut self) -> i32 {
        match self.cur_char {
            40 => self.stop_at_pos(0, LPAREN),
            41 => self.stop_at_pos(0, RPAREN),
            44 => self.stop_at_pos(0, COMMA),
            58 => self.stop_at_pos(0, COLON),
            94 => self.stop_at_pos(0, CARAT),
            _ => self.move_nfa_1(0, 0),
        }
    }

    fn move_nfa_1(&mut self, start_state: i32, mut cur_pos: i32) -> i32 {
        let mut starts_at = 0usize;
        self.nfa_start(start_state, 38);
        let mut i = 1usize;
        let mut kind = NO_KIND;
        loop {
            self.next_round();
            let c = self.cur_char;
            if c < 64 {
                let l = 1u64 << c;
                loop {
                    i -= 1;
                    match self.state_at(i) {
                        0 => {
                            if 0x7bff_e8fa_ffff_d9ff & l != 0 {
                                kind = kind.min(22);
                                self.states(0, 4);
                            } else if 0x1_0000_2600 & l != 0 {
                                kind = kind.min(7);
                            } else if c == 34 {
                                self.states(5, 7);
                            }
                            if 0x3fc_0000_0000_0000 & l != 0 {
                                self.states(8, 11);
                            } else if c == 49 {
                                self.two(20, 21);
                            }
                        }
                        20 => {
                            if 0x3ff_0000_0000_0000 & l != 0 {
                                self.check_n_add(17);
                            }
                        }
                        21 => {
                            if 0x3ff_0000_0000_0000 & l != 0 {
                                self.check_n_add(18);
                            }
                        }
                        24 => {
                            if 0xffff_fffb_ffff_ffff & l != 0 {
                                self.two(24, 25);
                            }
                        }
                        25 => {
                            if c == 34 {
                                self.push(26);
                            }
                        }
                        26 => {
                            if c == 42 {
                                kind = kind.min(18);
                            }
                        }
                        27 => {
                            if 0xffff_fffb_ffff_ffff & l != 0 {
                                self.states(12, 14);
                            }
                        }
                        29 => {
                            if c == 34 {
                                self.states(12, 14);
                            }
                        }
                        30 => {
                            if c == 34 {
                                kind = kind.min(19);
                            }
                        }
                        32 => {
                            if 0x7bff_e8fa_ffff_d9ff & l != 0 {
                                self.two(32, 33);
                            }
                        }
                        33 => {
                            if c == 42 {
                                kind = kind.min(20);
                            }
                        }
                        34 => {
                            if 0x7bff_e8fa_ffff_d9ff & l != 0 {
                                self.two(34, 35);
                            }
                        }
                        35 => {
                            if 0x8000_0400_0000_0000 & l != 0 {
                                kind = kind.min(21);
                                self.two(35, 36);
                            }
                        }
                        36 => {
                            if 0xfbff_ecfa_ffff_d9ff & l != 0 {
                                kind = kind.min(21);
                                self.check_n_add(36);
                            }
                        }
                        37 => {
                            if 0x7bff_e8fa_ffff_d9ff & l != 0 {
                                kind = kind.min(22);
                                self.check_n_add(37);
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
                    match self.state_at(i) {
                        0 => {
                            if 0xffff_ffff_bfff_ffff & l != 0 {
                                kind = kind.min(22);
                                self.states(0, 4);
                            }
                            if 0x4000_0000_4000 & l != 0 {
                                kind = kind.min(12);
                            } else if 0x0080_0000_0080_0000 & l != 0 {
                                kind = kind.min(11);
                            } else if c == 97 {
                                self.push(9);
                            } else if c == 65 {
                                self.push(6);
                            } else if c == 111 {
                                self.push(3);
                            } else if c == 79 {
                                self.push(1);
                            }
                            if c == 110 {
                                self.push(15);
                            } else if c == 78 {
                                self.push(12);
                            }
                        }
                        1 => {
                            if c == 82 {
                                kind = kind.min(8);
                            }
                        }
                        3 => {
                            if c == 114 {
                                kind = kind.min(8);
                            }
                        }
                        5 => {
                            if c == 68 {
                                kind = kind.min(9);
                            }
                        }
                        6 => {
                            if c == 78 {
                                self.push(5);
                            }
                        }
                        8 => {
                            if c == 100 {
                                kind = kind.min(9);
                            }
                        }
                        9 => {
                            if c == 110 {
                                self.push(8);
                            }
                        }
                        11 => {
                            if c == 84 {
                                kind = kind.min(10);
                            }
                        }
                        12 => {
                            if c == 79 {
                                self.push(11);
                            }
                        }
                        14 => {
                            if c == 116 {
                                kind = kind.min(10);
                            }
                        }
                        15 => {
                            if c == 111 {
                                self.push(14);
                            }
                        }
                        17 => {
                            if 0x0080_0000_0080_0000 & l != 0 {
                                kind = kind.min(11);
                            }
                        }
                        18 => {
                            if 0x4000_0000_4000 & l != 0 {
                                kind = kind.min(12);
                            }
                        }
                        24 => self.add_states(15, 16),
                        27 => {
                            if 0xffff_ffff_efff_ffff & l != 0 {
                                self.states(12, 14);
                            }
                        }
                        28 => {
                            if c == 92 {
                                self.push(29);
                            }
                        }
                        29 => {
                            if c == 92 {
                                self.states(12, 14);
                            }
                        }
                        32 => {
                            if 0xffff_ffff_bfff_ffff & l != 0 {
                                self.two(32, 33);
                            }
                        }
                        34 => {
                            if 0xffff_ffff_bfff_ffff & l != 0 {
                                self.two(34, 35);
                            }
                        }
                        36 => {
                            if 0xffff_ffff_bfff_ffff & l != 0 {
                                kind = kind.min(21);
                                // no round check, as generated
                                self.push(36);
                            }
                        }
                        37 => {
                            if 0xffff_ffff_bfff_ffff & l != 0 {
                                kind = kind.min(22);
                                self.check_n_add(37);
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
                let can = can_move_0(hi_byte, i1, i2, l1, l2);
                loop {
                    i -= 1;
                    match self.state_at(i) {
                        0 => {
                            if can {
                                kind = kind.min(22);
                                self.states(0, 4);
                            }
                        }
                        24 => {
                            if can {
                                self.add_states(15, 16);
                            }
                        }
                        27 => {
                            if can {
                                self.add_states(12, 14);
                            }
                        }
                        32 => {
                            if can {
                                self.two(32, 33);
                            }
                        }
                        34 => {
                            if can {
                                self.two(34, 35);
                            }
                        }
                        36 => {
                            if can {
                                kind = kind.min(21);
                                self.push(36);
                            }
                        }
                        37 => {
                            if can {
                                kind = kind.min(22);
                                self.check_n_add(37);
                            }
                        }
                        _ => {}
                    }
                    if i == starts_at {
                        break;
                    }
                }
            }
            if let Err(pos) = self.nfa_step(&mut kind, &mut cur_pos, &mut i, &mut starts_at, 38) {
                return pos;
            }
        }
    }

    // --------------------------------------------------------- Boost (0)

    fn move_nfa_0(&mut self, start_state: i32, mut cur_pos: i32) -> i32 {
        let mut starts_at = 0usize;
        self.nfa_start(start_state, 3);
        let mut i = 1usize;
        let mut kind = NO_KIND;
        loop {
            self.next_round();
            let c = self.cur_char;
            if c < 64 {
                let l = 1u64 << c;
                loop {
                    i -= 1;
                    match self.state_at(i) {
                        0 => {
                            if 0x3ff_0000_0000_0000 & l != 0 {
                                kind = kind.min(23);
                                self.add_states(17, 18);
                            }
                        }
                        1 => {
                            if c == 46 {
                                self.check_n_add(2);
                            }
                        }
                        2 => {
                            if 0x3ff_0000_0000_0000 & l != 0 {
                                kind = kind.min(23);
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
}

impl Dfa for TokenManager {
    fn dispatch(&mut self) -> i32 {
        match self.cur_lex_state {
            BOOST => self.move_nfa_0(0, 0),
            _ => self.move_string_literal_dfa0_1(),
        }
    }

    /// `jjnewLexState`.
    // SENTINEL: `-1` = the kind does not change the lexical state.
    fn new_lex_state(kind: i32) -> i32 {
        match kind {
            CARAT => BOOST,
            NUMBER => DEFAULT,
            _ => -1,
        }
    }

    /// `jjtoToken` (`0xffff01`): every kind but the skipped white space (7).
    fn is_token(kind: i32) -> bool {
        kind == 0 || (8..=23).contains(&kind)
    }

    /// `jjstrLiteralImages`.
    fn literal_image(kind: i32) -> Option<&'static str> {
        Some(match kind {
            0 => "",
            LPAREN => "(",
            RPAREN => ")",
            COMMA => ",",
            COLON => ":",
            CARAT => "^",
            _ => return None,
        })
    }
}

impl Lexer for TokenManager {
    fn next_token(&mut self) -> Result<Token, TokenMgrError> {
        self.next_token_loop()
    }
}
