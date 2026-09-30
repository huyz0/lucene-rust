//! `UTF32ToUTF8`: rewrite a code-point automaton as the equivalent UTF-8
//! byte automaton, one code-point range edge at a time.

use super::automaton::{Automaton, Builder, Transition, TransitionAccessor};

const START_CODES: [i32; 4] = [0, 128, 2048, 65536];
const END_CODES: [i32; 4] = [127, 2047, 65535, 1_114_111];

/// `MASKS[i] = (2 << (i - 1)) - 1` for `i` in `1..8`, `MASKS[0] = 0`.
const MASKS: [i32; 8] = [0, 1, 3, 7, 15, 31, 63, 127];

#[derive(Clone, Copy, Default)]
struct Utf8Sequence {
    value: [i32; 4],
    bits: [i32; 4],
    len: usize,
}

impl Utf8Sequence {
    fn byte_at(&self, i: usize) -> i32 {
        self.value[i] & 0xFF
    }

    fn num_bits(&self, i: usize) -> i32 {
        self.bits[i]
    }

    fn set(&mut self, code: i32) {
        if code < 128 {
            self.value[0] = code;
            self.bits[0] = 7;
            self.len = 1;
        } else if code < 2048 {
            self.value[0] = (6 << 5) | (code >> 6);
            self.bits[0] = 5;
            self.set_rest(code, 1);
            self.len = 2;
        } else if code < 65536 {
            self.value[0] = (14 << 4) | (code >> 12);
            self.bits[0] = 4;
            self.set_rest(code, 2);
            self.len = 3;
        } else {
            self.value[0] = (30 << 3) | (code >> 18);
            self.bits[0] = 3;
            self.set_rest(code, 3);
            self.len = 4;
        }
        self.value[0] &= 0xFF;
    }

    fn set_first_byte(&mut self, code: i32) {
        if code < 128 {
            self.value[0] = code;
            self.len = 1;
        } else if code < 2048 {
            self.value[0] = (6 << 5) | (code >> 6);
            self.len = 2;
        } else if code < 65536 {
            self.value[0] = (14 << 4) | (code >> 12);
            self.len = 3;
        } else {
            self.value[0] = (30 << 3) | (code >> 18);
            self.len = 4;
        }
        self.value[0] &= 0xFF;
    }

    fn set_rest(&mut self, mut code: i32, num_bytes: usize) {
        for i in 0..num_bytes {
            self.value[num_bytes - i] = 128 | (code & MASKS[6]);
            self.bits[num_bytes - i] = 6;
            code >>= 6;
        }
    }
}

/// `UTF32ToUTF8`.
#[derive(Default)]
pub struct Utf32ToUtf8 {
    utf8: Builder,
}

impl Utf32ToUtf8 {
    /// `new UTF32ToUTF8()`.
    pub fn new() -> Self {
        Self::default()
    }

    fn convert_one_edge(&mut self, start: i32, end: i32, start_cp: i32, end_cp: i32) {
        let mut s = Utf8Sequence::default();
        let mut e = Utf8Sequence::default();
        s.set(start_cp);
        e.set(end_cp);
        self.build(start, end, &s, &e, 0);
    }

    fn build(&mut self, start: i32, end: i32, su: &Utf8Sequence, eu: &Utf8Sequence, upto: usize) {
        if su.byte_at(upto) == eu.byte_at(upto) {
            if upto == su.len - 1 && upto == eu.len - 1 {
                self.utf8
                    .add_transition(start, end, su.byte_at(upto), eu.byte_at(upto));
            } else {
                let n = self.utf8.create_state();
                self.utf8.add_transition_label(start, n, su.byte_at(upto));
                self.build(n, end, su, eu, 1 + upto);
            }
        } else if su.len == eu.len {
            if upto == su.len - 1 {
                self.utf8
                    .add_transition(start, end, su.byte_at(upto), eu.byte_at(upto));
            } else {
                self.start(start, end, su, upto, false);
                if eu.byte_at(upto) - su.byte_at(upto) > 1 {
                    self.all(
                        start,
                        end,
                        su.byte_at(upto) + 1,
                        eu.byte_at(upto) - 1,
                        su.len - upto - 1,
                    );
                }
                self.end(start, end, eu, upto, false);
            }
        } else {
            self.start(start, end, su, upto, true);
            let mut byte_count = 1 + su.len - upto;
            let limit = eu.len - upto;
            while byte_count < limit {
                let mut a = Utf8Sequence::default();
                let mut b = Utf8Sequence::default();
                a.set_first_byte(START_CODES[byte_count - 1]);
                b.set_first_byte(END_CODES[byte_count - 1]);
                self.all(start, end, a.byte_at(0), b.byte_at(0), a.len - 1);
                byte_count += 1;
            }
            self.end(start, end, eu, upto, true);
        }
    }

    fn start(&mut self, start: i32, end: i32, su: &Utf8Sequence, upto: usize, do_all: bool) {
        let mask = MASKS[su.num_bits(upto) as usize];
        if upto == su.len - 1 {
            self.utf8
                .add_transition(start, end, su.byte_at(upto), su.byte_at(upto) | mask);
        } else {
            let n = self.utf8.create_state();
            self.utf8.add_transition_label(start, n, su.byte_at(upto));
            self.start(n, end, su, 1 + upto, true);
            let end_code = su.byte_at(upto) | mask;
            if do_all && su.byte_at(upto) != end_code {
                self.all(
                    start,
                    end,
                    su.byte_at(upto) + 1,
                    end_code,
                    su.len - upto - 1,
                );
            }
        }
    }

    fn end(&mut self, start: i32, end: i32, eu: &Utf8Sequence, upto: usize, do_all: bool) {
        let mask = MASKS[eu.num_bits(upto) as usize];
        if upto == eu.len - 1 {
            self.utf8
                .add_transition(start, end, eu.byte_at(upto) & !mask, eu.byte_at(upto));
        } else {
            let start_code = if eu.len == 2 {
                0xC2
            } else if eu.len == 3 && upto == 1 && eu.byte_at(0) == 0xE0 {
                0xA0
            } else if eu.len == 4 && upto == 1 && eu.byte_at(0) == 0xF0 {
                0x90
            } else {
                eu.byte_at(upto) & !mask
            };
            if do_all && eu.byte_at(upto) != start_code {
                self.all(
                    start,
                    end,
                    start_code,
                    eu.byte_at(upto) - 1,
                    eu.len - upto - 1,
                );
            }
            let n = self.utf8.create_state();
            self.utf8.add_transition_label(start, n, eu.byte_at(upto));
            self.end(n, end, eu, 1 + upto, true);
        }
    }

    fn all(&mut self, start: i32, end: i32, start_code: i32, end_code: i32, mut left: usize) {
        if left == 0 {
            self.utf8.add_transition(start, end, start_code, end_code);
        } else {
            let mut last_n = self.utf8.create_state();
            self.utf8
                .add_transition(start, last_n, start_code, end_code);
            while left > 1 {
                let n = self.utf8.create_state();
                self.utf8.add_transition(last_n, n, 128, 191);
                left -= 1;
                last_n = n;
            }
            self.utf8.add_transition(last_n, end, 128, 191);
        }
    }

    /// `convert(utf32)`: the byte automaton accepting the UTF-8 encodings of
    /// `utf32`'s strings (surrogate code points encode as their 3-byte
    /// CESU-style sequences, as in Java).
    pub fn convert(&mut self, utf32: &Automaton) -> Automaton {
        if utf32.get_num_states() == 0 {
            return utf32.clone();
        }
        let mut map = vec![-1i32; utf32.get_num_states() as usize];
        let mut pending: Vec<i32> = vec![0];
        self.utf8 = Builder::new();
        let s0 = self.utf8.create_state();
        self.utf8.set_accept(s0, utf32.is_accept(0));
        map[0] = s0;
        let mut scratch = Transition::new();
        while let Some(utf32_state) = pending.pop() {
            let utf8_state = map[utf32_state as usize];
            let n = utf32.init_transition(utf32_state, &mut scratch);
            for _ in 0..n {
                utf32.get_next_transition(&mut scratch);
                let dest32 = scratch.dest;
                let mut dest8 = map[dest32 as usize];
                if dest8 == -1 {
                    dest8 = self.utf8.create_state();
                    self.utf8.set_accept(dest8, utf32.is_accept(dest32));
                    map[dest32 as usize] = dest8;
                    pending.push(dest32);
                }
                self.convert_one_edge(utf8_state, dest8, scratch.min, scratch.max);
            }
        }
        std::mem::take(&mut self.utf8).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::automata::{make_any_char, make_char_range};
    use crate::automaton::operations::{determinize, run_ints};

    fn accepts(a: &Automaton, s: &str) -> bool {
        let bytes: Vec<i32> = s.bytes().map(i32::from).collect();
        run_ints(a, &bytes)
    }

    #[test]
    fn converts_every_length() {
        let a = Utf32ToUtf8::new().convert(&make_any_char());
        let d = determinize(&a, i32::MAX).unwrap();
        for s in [
            "a",
            "\u{7f}",
            "\u{80}",
            "\u{7ff}",
            "\u{800}",
            "\u{ffff}",
            "\u{10000}",
            "\u{10ffff}",
        ] {
            assert!(accepts(&d, s), "{s:?}");
        }
        assert!(!accepts(&d, "ab"));
        let r = Utf32ToUtf8::new().convert(&make_char_range(0x3b1, 0x1f600));
        let d = determinize(&r, i32::MAX).unwrap();
        assert!(accepts(&d, "\u{3b1}") && accepts(&d, "\u{1f600}") && accepts(&d, "\u{4e2d}"));
        assert!(!accepts(&d, "a") && !accepts(&d, "\u{1f601}"));
        let empty = Automaton::new();
        assert_eq!(Utf32ToUtf8::new().convert(&empty).get_num_states(), 0);
    }
}
