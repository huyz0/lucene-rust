//! `org.apache.lucene.queryparser.flexible.standard.parser.StandardSyntaxParserTokenManager`
//! (the JavaCC 7 lexer generated from `StandardSyntaxParser.jj`), state for
//! state. This file is the output of `crates/lucene-search/tools/gen_flexible.sh`
//! (`javacc_lexer.py` over the generated Java): do not edit it by hand. The
//! three lexical states are `Function` (0, after `fn:`), `Range` (1, inside
//! `[`/`{`) and `DEFAULT` (2). NFA arms of states nothing can enter are left
//! out, which changes nothing. Some kept transitions no query reaches (a
//! keyword DFA level's stop state for a kind that cannot stop there), as in
//! Lucene's generated Java, so the file is not fully covered.

// The generated code's shape is kept: one arm per NFA state, each case's
// tests inside it, the string-literal DFA's returns as written.
#![allow(
    clippy::collapsible_match,
    clippy::collapsible_else_if,
    clippy::collapsible_if,
    clippy::needless_return,
    clippy::match_single_binding,
    clippy::single_match,
    clippy::if_same_then_else
)]

use std::ops::{Deref, DerefMut};

use crate::queryparser::javacc::{CharStream, Dfa, Lexer, NfaCore, Token, TokenMgrError, NO_KIND};

static JJ_BIT_VEC0: [u64; 4] = [0x1, 0x0, 0x0, 0x0];
static JJ_BIT_VEC1: [u64; 4] = [
    0xfffffffffffffffe,
    0xffffffffffffffff,
    0xffffffffffffffff,
    0xffffffffffffffff,
];
static JJ_BIT_VEC3: [u64; 4] = [0x0, 0x0, 0xffffffffffffffff, 0xffffffffffffffff];
static JJ_BIT_VEC4: [u64; 4] = [
    0xfffefffffffffffe,
    0xffffffffffffffff,
    0xffffffffffffffff,
    0xffffffffffffffff,
];

/// `jjnextStates`.
static JJ_NEXT_STATES: [i32; 29] = [
    28, 30, 31, 15, 16, 18, 180, 185, 192, 198, 98, 110, 123, 135, 147, 158, 166, 173, 79, 84, 63,
    72, 45, 52, 19, 36, 2, 4, 5,
];

/// `jjCanMove_0`.
fn can_move_0(hi_byte: i32, _i1: usize, i2: usize, _l1: u64, l2: u64) -> bool {
    match hi_byte {
        48 => {
            return (JJ_BIT_VEC0[i2] & l2) != 0;
        }
        _ => {
            return false;
        }
    }
}

/// `jjCanMove_1`.
fn can_move_1(hi_byte: i32, i1: usize, i2: usize, l1: u64, l2: u64) -> bool {
    match hi_byte {
        0 => {
            return (JJ_BIT_VEC3[i2] & l2) != 0;
        }
        _ => {
            if (JJ_BIT_VEC1[i1] & l1) != 0 {
                return true;
            }
            return false;
        }
    }
}

/// `jjCanMove_2`.
fn can_move_2(hi_byte: i32, i1: usize, i2: usize, l1: u64, l2: u64) -> bool {
    match hi_byte {
        0 => {
            return (JJ_BIT_VEC3[i2] & l2) != 0;
        }
        48 => {
            return (JJ_BIT_VEC1[i2] & l2) != 0;
        }
        _ => {
            if (JJ_BIT_VEC4[i1] & l1) != 0 {
                return true;
            }
            return false;
        }
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
            core: NfaCore::new(input, 199, &JJ_NEXT_STATES, 2),
        }
    }

    // SENTINEL: `-1` = no state: the NFA it starts matches nothing.
    fn stop_string_literal_dfa_0(&mut self, pos: i32, active0: u64) -> i32 {
        match pos {
            0 => {
                if (active0 & 0x800000000000) != 0 {
                    return 19;
                }
                if (active0 & 0x80000000) != 0 {
                    return 79;
                }
                if (active0 & 0x400000000) != 0 {
                    return 63;
                }
                return -1;
            }
            1 => {
                if (active0 & 0x400000000) != 0 {
                    return 62;
                }
                if (active0 & 0x800000000000) != 0 {
                    return 18;
                }
                return -1;
            }
            2 => {
                if (active0 & 0x800000000000) != 0 {
                    return 17;
                }
                if (active0 & 0x400000000) != 0 {
                    return 61;
                }
                return -1;
            }
            3 => {
                if (active0 & 0x400000000) != 0 {
                    return 60;
                }
                if (active0 & 0x800000000000) != 0 {
                    return 16;
                }
                return -1;
            }
            4 => {
                if (active0 & 0x800000000000) != 0 {
                    return 15;
                }
                if (active0 & 0x400000000) != 0 {
                    return 59;
                }
                return -1;
            }
            5 => {
                if (active0 & 0x400000000) != 0 {
                    return 58;
                }
                if (active0 & 0x800000000000) != 0 {
                    return 14;
                }
                return -1;
            }
            6 => {
                if (active0 & 0x800000000000) != 0 {
                    return 13;
                }
                if (active0 & 0x400000000) != 0 {
                    return 57;
                }
                return -1;
            }
            7 => {
                if (active0 & 0x800000000000) != 0 {
                    return 12;
                }
                return -1;
            }
            8 => {
                if (active0 & 0x800000000000) != 0 {
                    return 11;
                }
                return -1;
            }
            _ => {
                return -1;
            }
        }
    }

    fn start_nfa_0(&mut self, pos: i32, active0: u64) -> i32 {
        let state = self.stop_string_literal_dfa_0(pos, active0);
        return self.move_nfa_0(state, pos + 1);
    }

    fn move_string_literal_dfa0_0(&mut self) -> i32 {
        match self.cur_char {
            40 => {
                return self.stop_at_pos(0, 29);
            }
            97 => {
                return self.move_string_literal_dfa1_0(0x80000000);
            }
            98 => {
                return self.move_string_literal_dfa1_0(0x100000000);
            }
            99 => {
                return self.move_string_literal_dfa1_0(0x400000000);
            }
            101 => {
                return self.move_string_literal_dfa1_0(0x800000000);
            }
            111 => {
                return self.move_string_literal_dfa1_0(0x301000000000);
            }
            112 => {
                return self.move_string_literal_dfa1_0(0x400000000000);
            }
            117 => {
                return self.move_string_literal_dfa1_0(0x800000000000);
            }
            119 => {
                return self.move_string_literal_dfa1_0(0x6000000000000);
            }
            _ => {
                return self.move_nfa_0(0, 0);
            }
        }
    }

    fn move_string_literal_dfa1_0(&mut self, active0: u64) -> i32 {
        if !self.read() {
            self.stop_string_literal_dfa_0(0, active0);
            return 1;
        }
        match self.cur_char {
            101 => {
                return self.move_string_literal_dfa2_0(active0, 0x100000000);
            }
            102 => {
                return self.move_string_literal_dfa2_0(active0, 0x80000000);
            }
            104 => {
                return self.move_string_literal_dfa2_0(active0, 0x400000000000);
            }
            105 => {
                return self.move_string_literal_dfa2_0(active0, 0x6000000000000);
            }
            110 => {
                return self.move_string_literal_dfa2_0(active0, 0x800000000000);
            }
            111 => {
                return self.move_string_literal_dfa2_0(active0, 0x400000000);
            }
            114 => {
                if (active0 & 0x1000000000) != 0 {
                    self.jjmatched_kind = 36;
                    self.jjmatched_pos = 1;
                }
                return self.move_string_literal_dfa2_0(active0, 0x100000000000);
            }
            118 => {
                return self.move_string_literal_dfa2_0(active0, 0x200000000000);
            }
            120 => {
                return self.move_string_literal_dfa2_0(active0, 0x800000000);
            }
            _ => {}
        }
        return self.start_nfa_0(0, active0);
    }

    fn move_string_literal_dfa2_0(&mut self, old0: u64, mut active0: u64) -> i32 {
        active0 &= old0;
        if active0 == 0 {
            return self.start_nfa_0(0, old0);
        }
        if !self.read() {
            self.stop_string_literal_dfa_0(1, active0);
            return 2;
        }
        match self.cur_char {
            100 => {
                return self.move_string_literal_dfa3_0(active0, 0x100000000000);
            }
            101 => {
                return self.move_string_literal_dfa3_0(active0, 0x200000000000);
            }
            102 => {
                return self.move_string_literal_dfa3_0(active0, 0x100000000);
            }
            108 => {
                return self.move_string_literal_dfa3_0(active0, 0x2000000000000);
            }
            110 => {
                return self.move_string_literal_dfa3_0(active0, 0x400000000);
            }
            111 => {
                return self.move_string_literal_dfa3_0(active0, 0x800000000000);
            }
            114 => {
                return self.move_string_literal_dfa3_0(active0, 0x400000000000);
            }
            116 => {
                return self.move_string_literal_dfa3_0(active0, 0x4000880000000);
            }
            _ => {}
        }
        return self.start_nfa_0(1, active0);
    }

    fn move_string_literal_dfa3_0(&mut self, old0: u64, mut active0: u64) -> i32 {
        active0 &= old0;
        if active0 == 0 {
            return self.start_nfa_0(1, old0);
        }
        if !self.read() {
            self.stop_string_literal_dfa_0(2, active0);
            return 3;
        }
        match self.cur_char {
            97 => {
                return self.move_string_literal_dfa4_0(active0, 0x400000000000);
            }
            100 => {
                return self.move_string_literal_dfa4_0(active0, 0x2000000000000);
            }
            101 => {
                return self.move_string_literal_dfa4_0(active0, 0x100880000000);
            }
            104 => {
                return self.move_string_literal_dfa4_0(active0, 0x4000000000000);
            }
            111 => {
                return self.move_string_literal_dfa4_0(active0, 0x100000000);
            }
            114 => {
                return self.move_string_literal_dfa4_0(active0, 0xa00000000000);
            }
            116 => {
                return self.move_string_literal_dfa4_0(active0, 0x400000000);
            }
            _ => {}
        }
        return self.start_nfa_0(2, active0);
    }

    fn move_string_literal_dfa4_0(&mut self, old0: u64, mut active0: u64) -> i32 {
        active0 &= old0;
        if active0 == 0 {
            return self.start_nfa_0(2, old0);
        }
        if !self.read() {
            self.stop_string_literal_dfa_0(3, active0);
            return 4;
        }
        match self.cur_char {
            97 => {
                return self.move_string_literal_dfa5_0(active0, 0x400000000);
            }
            99 => {
                return self.move_string_literal_dfa5_0(active0, 0x2000000000000);
            }
            100 => {
                return self.move_string_literal_dfa5_0(active0, 0x800000000000);
            }
            105 => {
                return self.move_string_literal_dfa5_0(active0, 0x4000000000000);
            }
            108 => {
                return self.move_string_literal_dfa5_0(active0, 0x200000000000);
            }
            110 => {
                return self.move_string_literal_dfa5_0(active0, 0x800000000);
            }
            114 => {
                if (active0 & 0x80000000) != 0 {
                    return self.stop_at_pos(4, 31);
                }
                return self.move_string_literal_dfa5_0(active0, 0x100100000000);
            }
            115 => {
                return self.move_string_literal_dfa5_0(active0, 0x400000000000);
            }
            _ => {}
        }
        return self.start_nfa_0(3, active0);
    }

    fn move_string_literal_dfa5_0(&mut self, old0: u64, mut active0: u64) -> i32 {
        active0 &= old0;
        if active0 == 0 {
            return self.start_nfa_0(3, old0);
        }
        if !self.read() {
            self.stop_string_literal_dfa_0(4, active0);
            return 5;
        }
        match self.cur_char {
            97 => {
                return self.move_string_literal_dfa6_0(active0, 0x2200000000000);
            }
            100 => {
                if (active0 & 0x800000000) != 0 {
                    return self.stop_at_pos(5, 35);
                }
            }
            101 => {
                if (active0 & 0x100000000) != 0 {
                    return self.stop_at_pos(5, 32);
                } else if (active0 & 0x400000000000) != 0 {
                    return self.stop_at_pos(5, 46);
                }
                return self.move_string_literal_dfa6_0(active0, 0x900000000000);
            }
            105 => {
                return self.move_string_literal_dfa6_0(active0, 0x400000000);
            }
            110 => {
                if (active0 & 0x4000000000000) != 0 {
                    return self.stop_at_pos(5, 50);
                }
            }
            _ => {}
        }
        return self.start_nfa_0(4, active0);
    }

    fn move_string_literal_dfa6_0(&mut self, old0: u64, mut active0: u64) -> i32 {
        active0 &= old0;
        if active0 == 0 {
            return self.start_nfa_0(4, old0);
        }
        if !self.read() {
            self.stop_string_literal_dfa_0(5, active0);
            return 6;
        }
        match self.cur_char {
            100 => {
                if (active0 & 0x100000000000) != 0 {
                    return self.stop_at_pos(6, 44);
                }
            }
            110 => {
                return self.move_string_literal_dfa7_0(active0, 0x400000000);
            }
            112 => {
                return self.move_string_literal_dfa7_0(active0, 0x200000000000);
            }
            114 => {
                return self.move_string_literal_dfa7_0(active0, 0x2800000000000);
            }
            _ => {}
        }
        return self.start_nfa_0(5, active0);
    }

    fn move_string_literal_dfa7_0(&mut self, old0: u64, mut active0: u64) -> i32 {
        active0 &= old0;
        if active0 == 0 {
            return self.start_nfa_0(5, old0);
        }
        if !self.read() {
            self.stop_string_literal_dfa_0(6, active0);
            return 7;
        }
        match self.cur_char {
            100 => {
                if (active0 & 0x2000000000000) != 0 {
                    return self.stop_at_pos(7, 49);
                }
            }
            101 => {
                return self.move_string_literal_dfa8_0(active0, 0x800000000000);
            }
            105 => {
                return self.move_string_literal_dfa8_0(active0, 0x400000000);
            }
            112 => {
                return self.move_string_literal_dfa8_0(active0, 0x200000000000);
            }
            _ => {}
        }
        return self.start_nfa_0(6, active0);
    }

    fn move_string_literal_dfa8_0(&mut self, old0: u64, mut active0: u64) -> i32 {
        active0 &= old0;
        if active0 == 0 {
            return self.start_nfa_0(6, old0);
        }
        if !self.read() {
            self.stop_string_literal_dfa_0(7, active0);
            return 8;
        }
        match self.cur_char {
            100 => {
                if (active0 & 0x800000000000) != 0 {
                    return self.start_nfa_with_states_0(8, 47, 11);
                }
            }
            105 => {
                return self.move_string_literal_dfa9_0(active0, 0x200000000000);
            }
            110 => {
                return self.move_string_literal_dfa9_0(active0, 0x400000000);
            }
            _ => {}
        }
        return self.start_nfa_0(7, active0);
    }

    fn move_string_literal_dfa9_0(&mut self, old0: u64, mut active0: u64) -> i32 {
        active0 &= old0;
        if active0 == 0 {
            return self.start_nfa_0(7, old0);
        }
        if !self.read() {
            self.stop_string_literal_dfa_0(8, active0);
            return 9;
        }
        match self.cur_char {
            103 => {
                if (active0 & 0x400000000) != 0 {
                    return self.stop_at_pos(9, 34);
                }
            }
            110 => {
                return self.move_string_literal_dfa10_0(active0, 0x200000000000);
            }
            _ => {}
        }
        return self.start_nfa_0(8, active0);
    }

    fn move_string_literal_dfa10_0(&mut self, old0: u64, mut active0: u64) -> i32 {
        active0 &= old0;
        if active0 == 0 {
            return self.start_nfa_0(8, old0);
        }
        if !self.read() {
            self.stop_string_literal_dfa_0(9, active0);
            return 10;
        }
        match self.cur_char {
            103 => {
                if (active0 & 0x200000000000) != 0 {
                    return self.stop_at_pos(10, 45);
                }
            }
            _ => {}
        }
        return self.start_nfa_0(9, active0);
    }

    fn start_nfa_with_states_0(&mut self, pos: i32, kind: i32, state: i32) -> i32 {
        self.jjmatched_kind = kind;
        self.jjmatched_pos = pos;
        if !self.read() {
            return pos + 1;
        }
        return self.move_nfa_0(state, pos + 1);
    }

    fn move_nfa_0(&mut self, start_state: i32, mut cur_pos: i32) -> i32 {
        let mut starts_at = 0usize;
        self.nfa_start(start_state, 199);
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
                            if (0x100002600 & l) != 0 {
                                kind = 7;
                            }
                        }
                        _ => {}
                    }
                    if i == starts_at {
                        break;
                    }
                }
            } else if c < 128 {
                loop {
                    i -= 1;
                    match self.state_at(i) {
                        18 => {
                            if c == 111 {
                                self.push(34);
                            }
                            if c == 111 {
                                self.push(17);
                            }
                        }
                        13 => {
                            if c == 101 {
                                self.push(29);
                            }
                            if c == 101 {
                                self.push(12);
                            }
                        }
                        60 => {
                            if c == 97 {
                                self.push(68);
                            }
                            if c == 97 {
                                self.push(59);
                            }
                        }
                        19 => {
                            if c == 110 {
                                self.push(35);
                            }
                            if c == 110 {
                                self.push(18);
                            }
                        }
                        57 => {
                            if c == 101 {
                                self.push(65);
                            }
                            if c == 101 {
                                self.push(56);
                            }
                        }
                        58 => {
                            if c == 110 {
                                self.push(66);
                            }
                            if c == 110 {
                                self.push(57);
                            }
                        }
                        59 => {
                            if c == 105 {
                                self.push(67);
                            }
                            if c == 105 {
                                self.push(58);
                            }
                        }
                        61 => {
                            if c == 116 {
                                self.push(69);
                            }
                            if c == 116 {
                                self.push(60);
                            }
                        }
                        14 => {
                            if c == 114 {
                                self.push(30);
                            }
                            if c == 114 {
                                self.push(13);
                            }
                        }
                        62 => {
                            if c == 110 {
                                self.push(70);
                            }
                            if c == 110 {
                                self.push(61);
                            }
                        }
                        15 => {
                            if c == 101 {
                                self.push(31);
                            }
                            if c == 101 {
                                self.push(14);
                            }
                        }
                        63 => {
                            if c == 111 {
                                self.push(71);
                            }
                            if c == 111 {
                                self.push(62);
                            }
                        }
                        16 => {
                            if c == 100 {
                                self.push(32);
                            }
                            if c == 100 {
                                self.push(15);
                            }
                        }
                        0 => {
                            if c == 109 {
                                self.add_states(6, 9);
                            } else if c == 110 {
                                self.add_states(10, 17);
                            } else if c == 97 {
                                self.add_states(18, 19);
                            } else if c == 99 {
                                self.add_states(20, 21);
                            } else if c == 102 {
                                self.add_states(22, 23);
                            } else if c == 117 {
                                self.add_states(24, 25);
                            }
                        }
                        11 => {
                            if c == 110 {
                                self.push(27);
                            } else if c == 78 {
                                self.push(10);
                            }
                        }
                        17 => {
                            if c == 114 {
                                self.push(33);
                            }
                            if c == 114 {
                                self.push(16);
                            }
                        }
                        79 => {
                            if c == 116 {
                                self.push(83);
                            }
                            if c == 116 {
                                self.push(78);
                            }
                        }
                        12 => {
                            if c == 100 {
                                self.push(28);
                            }
                            if c == 100 {
                                self.push(11);
                            }
                        }
                        2 => {
                            if c == 115 && kind > 48 {
                                kind = 48;
                            }
                        }
                        3 | 20 => {
                            if c == 112 {
                                self.check_n_add(2);
                            }
                        }
                        4 => {
                            if c == 97 {
                                self.push(3);
                            }
                        }
                        5 => {
                            if c == 108 {
                                self.push(4);
                            }
                        }
                        6 => {
                            if c == 114 {
                                self.push(5);
                            }
                        }
                        7 => {
                            if c == 101 {
                                self.push(6);
                            }
                        }
                        8 => {
                            if c == 118 {
                                self.push(7);
                            }
                        }
                        9 => {
                            if c == 79 {
                                self.push(8);
                            }
                        }
                        10 => {
                            if c == 111 {
                                self.push(9);
                            }
                        }
                        21 => {
                            if c == 97 {
                                self.push(20);
                            }
                        }
                        22 => {
                            if c == 108 {
                                self.push(21);
                            }
                        }
                        23 => {
                            if c == 114 {
                                self.push(22);
                            }
                        }
                        24 => {
                            if c == 101 {
                                self.push(23);
                            }
                        }
                        25 => {
                            if c == 118 {
                                self.push(24);
                            }
                        }
                        26 => {
                            if c == 111 {
                                self.push(25);
                            }
                        }
                        27 => {
                            if c == 111 {
                                self.push(26);
                            }
                        }
                        28 => {
                            if c == 110 {
                                self.push(27);
                            }
                        }
                        29 => {
                            if c == 100 {
                                self.push(28);
                            }
                        }
                        30 => {
                            if c == 101 {
                                self.push(29);
                            }
                        }
                        31 => {
                            if c == 114 {
                                self.push(30);
                            }
                        }
                        32 => {
                            if c == 101 {
                                self.push(31);
                            }
                        }
                        33 => {
                            if c == 100 {
                                self.push(32);
                            }
                        }
                        34 => {
                            if c == 114 {
                                self.push(33);
                            }
                        }
                        35 => {
                            if c == 111 {
                                self.push(34);
                            }
                        }
                        36 => {
                            if c == 110 {
                                self.push(35);
                            }
                        }
                        38 => {
                            if c == 109 && kind > 37 {
                                kind = 37;
                            }
                        }
                        39 | 46 => {
                            if c == 114 {
                                self.check_n_add(38);
                            }
                        }
                        40 => {
                            if c == 101 {
                                self.push(39);
                            }
                        }
                        41 => {
                            if c == 116 {
                                self.push(40);
                            }
                        }
                        42 => {
                            if c == 121 {
                                self.push(41);
                            }
                        }
                        43 => {
                            if c == 122 {
                                self.push(42);
                            }
                        }
                        44 => {
                            if c == 122 {
                                self.push(43);
                            }
                        }
                        45 => {
                            if c == 117 {
                                self.push(44);
                            }
                        }
                        47 => {
                            if c == 101 {
                                self.push(46);
                            }
                        }
                        48 => {
                            if c == 84 {
                                self.push(47);
                            }
                        }
                        49 => {
                            if c == 121 {
                                self.push(48);
                            }
                        }
                        50 => {
                            if c == 122 {
                                self.push(49);
                            }
                        }
                        51 => {
                            if c == 122 {
                                self.push(50);
                            }
                        }
                        52 => {
                            if c == 117 {
                                self.push(51);
                            }
                        }
                        54 => {
                            if c == 121 && kind > 33 {
                                kind = 33;
                            }
                        }
                        55 => {
                            if c == 66 {
                                self.check_n_add(54);
                            }
                        }
                        56 => {
                            if c == 100 {
                                self.push(55);
                            }
                        }
                        64 => {
                            if c == 98 {
                                self.check_n_add(54);
                            }
                        }
                        65 => {
                            if c == 100 {
                                self.push(64);
                            }
                        }
                        66 => {
                            if c == 101 {
                                self.push(65);
                            }
                        }
                        67 => {
                            if c == 110 {
                                self.push(66);
                            }
                        }
                        68 => {
                            if c == 105 {
                                self.push(67);
                            }
                        }
                        69 => {
                            if c == 97 {
                                self.push(68);
                            }
                        }
                        70 => {
                            if c == 116 {
                                self.push(69);
                            }
                        }
                        71 => {
                            if c == 110 {
                                self.push(70);
                            }
                        }
                        72 => {
                            if c == 111 {
                                self.push(71);
                            }
                        }
                        74 => {
                            if c == 116 && kind > 30 {
                                kind = 30;
                            }
                        }
                        75 | 80 => {
                            if c == 115 {
                                self.check_n_add(74);
                            }
                        }
                        76 => {
                            if c == 97 {
                                self.push(75);
                            }
                        }
                        77 => {
                            if c == 101 {
                                self.push(76);
                            }
                        }
                        78 => {
                            if c == 108 {
                                self.push(77);
                            }
                        }
                        81 => {
                            if c == 97 {
                                self.push(80);
                            }
                        }
                        82 => {
                            if c == 101 {
                                self.push(81);
                            }
                        }
                        83 => {
                            if c == 76 {
                                self.push(82);
                            }
                        }
                        84 => {
                            if c == 116 {
                                self.push(83);
                            }
                        }
                        86 => {
                            if c == 103 && kind > 40 {
                                kind = 40;
                            }
                        }
                        87 | 99 => {
                            if c == 110 {
                                self.check_n_add(86);
                            }
                        }
                        88 => {
                            if c == 105 {
                                self.push(87);
                            }
                        }
                        89 => {
                            if c == 112 {
                                self.push(88);
                            }
                        }
                        90 => {
                            if c == 112 {
                                self.push(89);
                            }
                        }
                        91 => {
                            if c == 97 {
                                self.push(90);
                            }
                        }
                        92 => {
                            if c == 108 {
                                self.push(91);
                            }
                        }
                        93 => {
                            if c == 114 {
                                self.push(92);
                            }
                        }
                        94 => {
                            if c == 101 {
                                self.push(93);
                            }
                        }
                        95 => {
                            if c == 118 {
                                self.push(94);
                            }
                        }
                        96 => {
                            if c == 79 {
                                self.push(95);
                            }
                        }
                        97 => {
                            if c == 110 {
                                self.push(96);
                            }
                        }
                        98 => {
                            if c == 111 {
                                self.push(97);
                            }
                        }
                        100 => {
                            if c == 105 {
                                self.push(99);
                            }
                        }
                        101 => {
                            if c == 112 {
                                self.push(100);
                            }
                        }
                        102 => {
                            if c == 112 {
                                self.push(101);
                            }
                        }
                        103 => {
                            if c == 97 {
                                self.push(102);
                            }
                        }
                        104 => {
                            if c == 108 {
                                self.push(103);
                            }
                        }
                        105 => {
                            if c == 114 {
                                self.push(104);
                            }
                        }
                        106 => {
                            if c == 101 {
                                self.push(105);
                            }
                        }
                        107 => {
                            if c == 118 {
                                self.push(106);
                            }
                        }
                        108 => {
                            if c == 111 {
                                self.push(107);
                            }
                        }
                        109 => {
                            if c == 110 {
                                self.push(108);
                            }
                        }
                        110 => {
                            if c == 111 {
                                self.push(109);
                            }
                        }
                        111 => {
                            if c == 121 && kind > 41 {
                                kind = 41;
                            }
                        }
                        112 => {
                            if c == 66 {
                                self.check_n_add(111);
                            }
                        }
                        113 => {
                            if c == 100 {
                                self.push(112);
                            }
                        }
                        114 => {
                            if c == 101 {
                                self.push(113);
                            }
                        }
                        115 => {
                            if c == 110 {
                                self.push(114);
                            }
                        }
                        116 => {
                            if c == 105 {
                                self.push(115);
                            }
                        }
                        117 => {
                            if c == 97 {
                                self.push(116);
                            }
                        }
                        118 => {
                            if c == 116 {
                                self.push(117);
                            }
                        }
                        119 => {
                            if c == 110 {
                                self.push(118);
                            }
                        }
                        120 => {
                            if c == 111 {
                                self.push(119);
                            }
                        }
                        121 => {
                            if c == 67 {
                                self.push(120);
                            }
                        }
                        122 => {
                            if c == 116 {
                                self.push(121);
                            }
                        }
                        123 => {
                            if c == 111 {
                                self.push(122);
                            }
                        }
                        124 => {
                            if c == 98 {
                                self.check_n_add(111);
                            }
                        }
                        125 => {
                            if c == 100 {
                                self.push(124);
                            }
                        }
                        126 => {
                            if c == 101 {
                                self.push(125);
                            }
                        }
                        127 => {
                            if c == 110 {
                                self.push(126);
                            }
                        }
                        128 => {
                            if c == 105 {
                                self.push(127);
                            }
                        }
                        129 => {
                            if c == 97 {
                                self.push(128);
                            }
                        }
                        130 => {
                            if c == 116 {
                                self.push(129);
                            }
                        }
                        131 => {
                            if c == 110 {
                                self.push(130);
                            }
                        }
                        132 => {
                            if c == 111 {
                                self.push(131);
                            }
                        }
                        133 => {
                            if c == 99 {
                                self.push(132);
                            }
                        }
                        134 => {
                            if c == 116 {
                                self.push(133);
                            }
                        }
                        135 => {
                            if c == 111 {
                                self.push(134);
                            }
                        }
                        136 => {
                            if c == 103 && kind > 42 {
                                kind = 42;
                            }
                        }
                        137 | 148 => {
                            if c == 110 {
                                self.check_n_add(136);
                            }
                        }
                        138 => {
                            if c == 105 {
                                self.push(137);
                            }
                        }
                        139 => {
                            if c == 110 {
                                self.push(138);
                            }
                        }
                        140 => {
                            if c == 105 {
                                self.push(139);
                            }
                        }
                        141 => {
                            if c == 97 {
                                self.push(140);
                            }
                        }
                        142 => {
                            if c == 116 {
                                self.push(141);
                            }
                        }
                        143 => {
                            if c == 110 {
                                self.push(142);
                            }
                        }
                        144 => {
                            if c == 111 {
                                self.push(143);
                            }
                        }
                        145 => {
                            if c == 67 {
                                self.push(144);
                            }
                        }
                        146 => {
                            if c == 116 {
                                self.push(145);
                            }
                        }
                        147 => {
                            if c == 111 {
                                self.push(146);
                            }
                        }
                        149 => {
                            if c == 105 {
                                self.push(148);
                            }
                        }
                        150 => {
                            if c == 110 {
                                self.push(149);
                            }
                        }
                        151 => {
                            if c == 105 {
                                self.push(150);
                            }
                        }
                        152 => {
                            if c == 97 {
                                self.push(151);
                            }
                        }
                        153 => {
                            if c == 116 {
                                self.push(152);
                            }
                        }
                        154 => {
                            if c == 110 {
                                self.push(153);
                            }
                        }
                        155 => {
                            if c == 111 {
                                self.push(154);
                            }
                        }
                        156 => {
                            if c == 99 {
                                self.push(155);
                            }
                        }
                        157 => {
                            if c == 116 {
                                self.push(156);
                            }
                        }
                        158 => {
                            if c == 111 {
                                self.push(157);
                            }
                        }
                        159 => {
                            if c == 110 && kind > 43 {
                                kind = 43;
                            }
                        }
                        160 | 167 => {
                            if c == 105 {
                                self.check_n_add(159);
                            }
                        }
                        161 => {
                            if c == 104 {
                                self.push(160);
                            }
                        }
                        162 => {
                            if c == 116 {
                                self.push(161);
                            }
                        }
                        163 => {
                            if c == 105 {
                                self.push(162);
                            }
                        }
                        164 => {
                            if c == 87 {
                                self.push(163);
                            }
                        }
                        165 => {
                            if c == 116 {
                                self.push(164);
                            }
                        }
                        166 => {
                            if c == 111 {
                                self.push(165);
                            }
                        }
                        168 => {
                            if c == 104 {
                                self.push(167);
                            }
                        }
                        169 => {
                            if c == 116 {
                                self.push(168);
                            }
                        }
                        170 => {
                            if c == 105 {
                                self.push(169);
                            }
                        }
                        171 => {
                            if c == 119 {
                                self.push(170);
                            }
                        }
                        172 => {
                            if c == 116 {
                                self.push(171);
                            }
                        }
                        173 => {
                            if c == 111 {
                                self.push(172);
                            }
                        }
                        175 => {
                            if c == 115 && kind > 38 {
                                kind = 38;
                            }
                        }
                        176 | 181 => {
                            if c == 112 {
                                self.check_n_add(175);
                            }
                        }
                        177 => {
                            if c == 97 {
                                self.push(176);
                            }
                        }
                        178 => {
                            if c == 103 {
                                self.push(177);
                            }
                        }
                        179 => {
                            if c == 120 {
                                self.push(178);
                            }
                        }
                        180 => {
                            if c == 97 {
                                self.push(179);
                            }
                        }
                        182 => {
                            if c == 97 {
                                self.push(181);
                            }
                        }
                        183 => {
                            if c == 71 {
                                self.push(182);
                            }
                        }
                        184 => {
                            if c == 120 {
                                self.push(183);
                            }
                        }
                        185 => {
                            if c == 97 {
                                self.push(184);
                            }
                        }
                        186 => {
                            if c == 104 && kind > 39 {
                                kind = 39;
                            }
                        }
                        187 | 193 => {
                            if c == 116 {
                                self.check_n_add(186);
                            }
                        }
                        188 => {
                            if c == 100 {
                                self.push(187);
                            }
                        }
                        189 => {
                            if c == 105 {
                                self.push(188);
                            }
                        }
                        190 => {
                            if c == 119 {
                                self.push(189);
                            }
                        }
                        191 => {
                            if c == 120 {
                                self.push(190);
                            }
                        }
                        192 => {
                            if c == 97 {
                                self.push(191);
                            }
                        }
                        194 => {
                            if c == 100 {
                                self.push(193);
                            }
                        }
                        195 => {
                            if c == 105 {
                                self.push(194);
                            }
                        }
                        196 => {
                            if c == 87 {
                                self.push(195);
                            }
                        }
                        197 => {
                            if c == 120 {
                                self.push(196);
                            }
                        }
                        198 => {
                            if c == 97 {
                                self.push(197);
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
                let l1 = 1u64 << (hi_byte & 63);
                let i2 = ((c & 0xff) >> 6) as usize;
                let l2 = 1u64 << (c & 63);
                loop {
                    i -= 1;
                    match self.state_at(i) {
                        0 => {
                            if can_move_0(hi_byte, i1, i2, l1, l2) && kind > 7 {
                                kind = 7;
                            }
                        }
                        _ => {}
                    }
                    if i == starts_at {
                        break;
                    }
                }
            }
            if let Err(pos) = self.nfa_step(&mut kind, &mut cur_pos, &mut i, &mut starts_at, 199) {
                return pos;
            }
        }
    }

    // SENTINEL: `-1` = no state: the NFA it starts matches nothing.
    fn stop_string_literal_dfa_1(&mut self, pos: i32, active0: u64) -> i32 {
        match pos {
            0 => {
                if (active0 & 0x8000000000000) != 0 {
                    self.jjmatched_kind = 55;
                    return 6;
                }
                return -1;
            }
            _ => {
                return -1;
            }
        }
    }

    fn start_nfa_1(&mut self, pos: i32, active0: u64) -> i32 {
        let state = self.stop_string_literal_dfa_1(pos, active0);
        return self.move_nfa_1(state, pos + 1);
    }

    fn move_string_literal_dfa0_1(&mut self) -> i32 {
        match self.cur_char {
            84 => {
                return self.move_string_literal_dfa1_1(0x8000000000000);
            }
            93 => {
                return self.stop_at_pos(0, 52);
            }
            125 => {
                return self.stop_at_pos(0, 53);
            }
            _ => {
                return self.move_nfa_1(0, 0);
            }
        }
    }

    fn move_string_literal_dfa1_1(&mut self, active0: u64) -> i32 {
        if !self.read() {
            self.stop_string_literal_dfa_1(0, active0);
            return 1;
        }
        match self.cur_char {
            79 => {
                if (active0 & 0x8000000000000) != 0 {
                    return self.start_nfa_with_states_1(1, 51, 6);
                }
            }
            _ => {}
        }
        return self.start_nfa_1(0, active0);
    }

    fn start_nfa_with_states_1(&mut self, pos: i32, kind: i32, state: i32) -> i32 {
        self.jjmatched_kind = kind;
        self.jjmatched_pos = pos;
        if !self.read() {
            return pos + 1;
        }
        return self.move_nfa_1(state, pos + 1);
    }

    fn move_nfa_1(&mut self, start_state: i32, mut cur_pos: i32) -> i32 {
        let mut starts_at = 0usize;
        self.nfa_start(start_state, 7);
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
                            if (0xfffffffeffffffff & l) != 0 {
                                kind = kind.min(55);
                                {
                                    self.check_n_add(6);
                                }
                            }
                            if (0x100002600 & l) != 0 {
                                kind = kind.min(7);
                            } else if c == 34 {
                                self.two(2, 4);
                            }
                        }
                        2 => {
                            if (0xfffffffbffffffff & l) != 0 {
                                self.states(26, 28);
                            }
                        }
                        3 => {
                            if c == 34 {
                                self.states(26, 28);
                            }
                        }
                        5 => {
                            if c == 34 && kind > 54 {
                                kind = 54;
                            }
                        }
                        6 => 'arm: {
                            if (0xfffffffeffffffff & l) == 0 {
                                break 'arm;
                            }
                            kind = kind.min(55);
                            {
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
                let l = 1u64 << (c & 63);
                loop {
                    i -= 1;
                    match self.state_at(i) {
                        0 | 6 => 'arm: {
                            if (0xdfffffffdfffffff & l) == 0 {
                                break 'arm;
                            }
                            kind = kind.min(55);
                            {
                                self.check_n_add(6);
                            }
                        }
                        2 => {
                            self.add_states(26, 28);
                        }
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
                let l1 = 1u64 << (hi_byte & 63);
                let i2 = ((c & 0xff) >> 6) as usize;
                let l2 = 1u64 << (c & 63);
                loop {
                    i -= 1;
                    match self.state_at(i) {
                        0 => {
                            if can_move_0(hi_byte, i1, i2, l1, l2) {
                                kind = kind.min(7);
                            }
                            if can_move_1(hi_byte, i1, i2, l1, l2) {
                                kind = kind.min(55);
                                {
                                    self.check_n_add(6);
                                }
                            }
                        }
                        2 => {
                            if can_move_1(hi_byte, i1, i2, l1, l2) {
                                self.add_states(26, 28);
                            }
                        }
                        6 => 'arm: {
                            if !can_move_1(hi_byte, i1, i2, l1, l2) {
                                break 'arm;
                            }
                            kind = kind.min(55);
                            {
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

    // SENTINEL: `-1` = no state: the NFA it starts matches nothing.
    fn stop_string_literal_dfa_2(&mut self, pos: i32, active0: u64) -> i32 {
        match pos {
            0 => {
                if (active0 & 0x800) != 0 {
                    self.jjmatched_kind = 25;
                    return 32;
                }
                return -1;
            }
            1 => {
                if (active0 & 0x800) != 0 {
                    self.jjmatched_kind = 25;
                    self.jjmatched_pos = 1;
                    return 32;
                }
                return -1;
            }
            _ => {
                return -1;
            }
        }
    }

    fn start_nfa_2(&mut self, pos: i32, active0: u64) -> i32 {
        let state = self.stop_string_literal_dfa_2(pos, active0);
        return self.move_nfa_2(state, pos + 1);
    }

    fn move_string_literal_dfa0_2(&mut self) -> i32 {
        match self.cur_char {
            40 => {
                return self.stop_at_pos(0, 29);
            }
            41 => {
                return self.stop_at_pos(0, 14);
            }
            43 => {
                return self.stop_at_pos(0, 12);
            }
            45 => {
                return self.stop_at_pos(0, 13);
            }
            58 => {
                return self.stop_at_pos(0, 15);
            }
            60 => {
                self.jjmatched_kind = 17;
                return self.move_string_literal_dfa1_2(0x40000);
            }
            61 => {
                return self.stop_at_pos(0, 16);
            }
            62 => {
                self.jjmatched_kind = 19;
                return self.move_string_literal_dfa1_2(0x100000);
            }
            64 => {
                return self.stop_at_pos(0, 56);
            }
            91 => {
                return self.stop_at_pos(0, 27);
            }
            94 => {
                return self.stop_at_pos(0, 21);
            }
            102 => {
                return self.move_string_literal_dfa1_2(0x800);
            }
            123 => {
                return self.stop_at_pos(0, 28);
            }
            126 => {
                return self.stop_at_pos(0, 22);
            }
            _ => {
                return self.move_nfa_2(0, 0);
            }
        }
    }

    fn move_string_literal_dfa1_2(&mut self, active0: u64) -> i32 {
        if !self.read() {
            self.stop_string_literal_dfa_2(0, active0);
            return 1;
        }
        match self.cur_char {
            61 => {
                if (active0 & 0x40000) != 0 {
                    return self.stop_at_pos(1, 18);
                } else if (active0 & 0x100000) != 0 {
                    return self.stop_at_pos(1, 20);
                }
            }
            110 => {
                return self.move_string_literal_dfa2_2(active0, 0x800);
            }
            _ => {}
        }
        return self.start_nfa_2(0, active0);
    }

    fn move_string_literal_dfa2_2(&mut self, old0: u64, mut active0: u64) -> i32 {
        active0 &= old0;
        if active0 == 0 {
            return self.start_nfa_2(0, old0);
        }
        if !self.read() {
            self.stop_string_literal_dfa_2(1, active0);
            return 2;
        }
        match self.cur_char {
            58 => {
                if (active0 & 0x800) != 0 {
                    return self.stop_at_pos(2, 11);
                }
            }
            _ => {}
        }
        return self.start_nfa_2(1, active0);
    }

    fn move_nfa_2(&mut self, start_state: i32, mut cur_pos: i32) -> i32 {
        let mut starts_at = 0usize;
        self.nfa_start(start_state, 32);
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
                        32 | 23 => 'arm: {
                            if (0x8bff7cf8ffffd9ff & l) == 0 {
                                break 'arm;
                            }
                            kind = kind.min(25);
                            {
                                self.two(23, 24);
                            }
                        }
                        0 => {
                            if (0x8bff54f8ffffd9ff & l) != 0 {
                                kind = kind.min(25);
                                {
                                    self.two(23, 24);
                                }
                            } else if (0x100002600 & l) != 0 {
                                kind = kind.min(7);
                            } else if c == 47 {
                                self.states(0, 2);
                            } else if c == 34 {
                                self.states(3, 5);
                            } else if c == 33 {
                                kind = kind.min(10);
                            }
                            if (0x3ff000000000000 & l) != 0 {
                                kind = kind.min(24);
                                {
                                    self.two(19, 20);
                                }
                            } else if c == 38 {
                                self.push(4);
                            }
                        }
                        4 => {
                            if c == 38 && kind > 8 {
                                kind = 8;
                            }
                        }
                        15 => {
                            if (0xfffffffbffffffff & l) != 0 {
                                self.states(3, 5);
                            }
                        }
                        17 => {
                            self.states(3, 5);
                        }
                        18 => {
                            if c == 34 && kind > 23 {
                                kind = 23;
                            }
                        }
                        19 => 'arm: {
                            if (0x3ff000000000000 & l) == 0 {
                                break 'arm;
                            }
                            kind = kind.min(24);
                            {
                                self.two(19, 20);
                            }
                        }
                        20 => {
                            if c == 46 {
                                self.check_n_add(21);
                            }
                        }
                        21 => 'arm: {
                            if (0x3ff000000000000 & l) == 0 {
                                break 'arm;
                            }
                            kind = kind.min(24);
                            {
                                self.check_n_add(21);
                            }
                        }
                        25 => {
                            kind = kind.min(25);
                            {
                                self.two(23, 24);
                            }
                        }
                        29 => {
                            if c == 47 {
                                self.states(0, 2);
                            }
                        }
                        28 => {
                            if (0xffff7fffffffffff & l) != 0 {
                                self.states(0, 2);
                            }
                        }
                        31 => {
                            if c == 47 && kind > 26 {
                                kind = 26;
                            }
                        }
                        _ => {}
                    }
                    if i == starts_at {
                        break;
                    }
                }
            } else if c < 128 {
                let l = 1u64 << (c & 63);
                loop {
                    i -= 1;
                    match self.state_at(i) {
                        32 => {
                            if (0x97ffffff87fffffe & l) != 0 {
                                kind = kind.min(25);
                                {
                                    self.two(23, 24);
                                }
                            } else if c == 92 {
                                self.two(25, 25);
                            }
                        }
                        0 => {
                            if (0x97ffffff87fffffe & l) != 0 {
                                kind = kind.min(25);
                                {
                                    self.two(23, 24);
                                }
                            } else if c == 92 {
                                self.check_n_add(25);
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
                            if c == 68 && kind > 8 {
                                kind = 8;
                            }
                        }
                        2 => {
                            if c == 78 {
                                self.push(1);
                            }
                        }
                        6 => {
                            if c == 82 && kind > 9 {
                                kind = 9;
                            }
                        }
                        8 => {
                            if c == 124 && kind > 9 {
                                kind = 9;
                            }
                        }
                        10 => {
                            if c == 84 && kind > 10 {
                                kind = 10;
                            }
                        }
                        11 => {
                            if c == 79 {
                                self.push(10);
                            }
                        }
                        15 => {
                            if (0xffffffffefffffff & l) != 0 {
                                self.states(3, 5);
                            }
                        }
                        16 => {
                            if c == 92 {
                                self.push(17);
                            }
                        }
                        17 => {
                            self.states(3, 5);
                        }
                        23 => 'arm: {
                            if (0x97ffffff87fffffe & l) == 0 {
                                break 'arm;
                            }
                            kind = kind.min(25);
                            {
                                self.two(23, 24);
                            }
                        }
                        24 => {
                            if c == 92 {
                                self.two(25, 25);
                            }
                        }
                        25 => {
                            kind = kind.min(25);
                            {
                                self.two(23, 24);
                            }
                        }
                        28 => {
                            self.add_states(0, 2);
                        }
                        30 => {
                            if c == 92 {
                                self.push(29);
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
                let l1 = 1u64 << (hi_byte & 63);
                let i2 = ((c & 0xff) >> 6) as usize;
                let l2 = 1u64 << (c & 63);
                loop {
                    i -= 1;
                    match self.state_at(i) {
                        32 | 23 => 'arm: {
                            if !can_move_2(hi_byte, i1, i2, l1, l2) {
                                break 'arm;
                            }
                            kind = kind.min(25);
                            {
                                self.two(23, 24);
                            }
                        }
                        0 => {
                            if can_move_0(hi_byte, i1, i2, l1, l2) {
                                kind = kind.min(7);
                            }
                            if can_move_2(hi_byte, i1, i2, l1, l2) {
                                kind = kind.min(25);
                                {
                                    self.two(23, 24);
                                }
                            }
                        }
                        15 | 17 => {
                            if can_move_1(hi_byte, i1, i2, l1, l2) {
                                self.states(3, 5);
                            }
                        }
                        25 => 'arm: {
                            if !can_move_1(hi_byte, i1, i2, l1, l2) {
                                break 'arm;
                            }
                            kind = kind.min(25);
                            {
                                self.two(23, 24);
                            }
                        }
                        28 => {
                            if can_move_1(hi_byte, i1, i2, l1, l2) {
                                self.add_states(0, 2);
                            }
                        }
                        _ => {}
                    }
                    if i == starts_at {
                        break;
                    }
                }
            }
            if let Err(pos) = self.nfa_step(&mut kind, &mut cur_pos, &mut i, &mut starts_at, 32) {
                return pos;
            }
        }
    }
}

impl Dfa for TokenManager {
    fn dispatch(&mut self) -> i32 {
        match self.cur_lex_state {
            0 => self.move_string_literal_dfa0_0(),
            1 => self.move_string_literal_dfa0_1(),
            _ => self.move_string_literal_dfa0_2(),
        }
    }

    /// `jjnewLexState`.
    // SENTINEL: `-1` = the kind does not change the lexical state.
    fn new_lex_state(kind: i32) -> i32 {
        match kind {
            11 => 0,
            27 | 28 => 1,
            29 | 52 | 53 => 2,
            _ => -1,
        }
    }

    /// `jjtoToken` (`0x1ffffffffffff01`).
    fn is_token(kind: i32) -> bool {
        const JJ_TO_TOKEN: [u64; 1] = [0x1ffffffffffff01];
        let k = kind as usize;
        JJ_TO_TOKEN[k >> 6] & (1u64 << (k & 63)) != 0
    }

    /// `jjstrLiteralImages`.
    fn literal_image(kind: i32) -> Option<&'static str> {
        Some(match kind {
            0 => "",
            11 => "fn:",
            12 => "+",
            13 => "-",
            14 => ")",
            15 => ":",
            16 => "=",
            17 => "<",
            18 => "<=",
            19 => ">",
            20 => ">=",
            21 => "^",
            22 => "~",
            27 => "[",
            28 => "{",
            29 => "(",
            31 => "after",
            32 => "before",
            34 => "containing",
            35 => "extend",
            36 => "or",
            44 => "ordered",
            45 => "overlapping",
            46 => "phrase",
            47 => "unordered",
            49 => "wildcard",
            50 => "within",
            51 => "TO",
            52 => "]",
            53 => "}",
            56 => "@",
            _ => return None,
        })
    }
}

impl Lexer for TokenManager {
    fn next_token(&mut self) -> Result<Token, TokenMgrError> {
        self.next_token_loop()
    }
}
