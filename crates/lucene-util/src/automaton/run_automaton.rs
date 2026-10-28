//! `RunAutomaton`, `ByteRunAutomaton`, `CharacterRunAutomaton` and the
//! `ByteRunnable` interface: a deterministic automaton flattened into a
//! `state x character class` table for fast stepping.

use super::automaton::{append_char_string, Automaton, Transition, TransitionAccessor};
use super::error::AutomatonError;
use super::operations;
use super::utf32_to_utf8::Utf32ToUtf8;
use super::MAX_CODE_POINT;

/// `ByteRunnable`: something that steps over bytes.
pub trait ByteRunnable {
    /// `step(state, c)`: the next state on byte `c`.
    // SENTINEL: -1 means `c` leads nowhere (the string is rejected).
    fn step(&self, state: i32, c: i32) -> i32;
    /// `isAccept(state)`.
    fn is_accept(&self, state: i32) -> bool;
    /// `getSize()`: number of states.
    fn get_size(&self) -> i32;
    /// `run(bytes)`: whether the byte string is accepted.
    fn run(&self, s: &[u8]) -> bool {
        let mut p = 0;
        for &b in s {
            p = self.step(p, i32::from(b));
            if p == -1 {
                return false;
            }
        }
        self.is_accept(p)
    }
}

/// `RunAutomaton`: the tabulated form of a deterministic [`Automaton`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunAutomaton {
    pub(crate) automaton: Automaton,
    alphabet_size: i32,
    size: i32,
    accept: Vec<bool>,
    /// `delta(state, c) = transitions[state * stride + class(c)]`; every
    /// entry is `-1` or a state below `size`.
    transitions: Vec<i32>,
    points: Vec<i32>,
    /// `points.len()`, the row length of `transitions` (at least 1).
    stride: usize,
    /// The class of every label below 256. Java sizes its `classmap` by
    /// `min(256, alphabetSize)`; filling all 256 changes nothing, because the
    /// table entry is exactly what [`RunAutomaton::get_char_class`] returns
    /// (the last point at or below the label). A class fits a byte: the
    /// points are strictly increasing from `points[0] == 0`, so
    /// `points[i] >= i`, and class `i` of a label `j < 256` has
    /// `points[i] <= j`, so `i <= 255`.
    classmap: Box<[u8; 256]>,
}

impl RunAutomaton {
    /// `new RunAutomaton(a, alphabetSize)`.
    ///
    /// # Errors
    /// `IllegalArgument("Automaton must be deterministic")`.
    pub fn new(a: Automaton, alphabet_size: i32) -> Result<Self, AutomatonError> {
        if !a.is_deterministic() {
            return Err(AutomatonError::IllegalArgument(
                "Automaton must be deterministic".into(),
            ));
        }
        let points = a.get_start_points();
        let size = a.get_num_states().max(1);
        let np = points.len();
        let mut accept = vec![false; size as usize];
        let mut transitions = vec![-1i32; size as usize * np];
        let mut t = Transition::new();
        for n in 0..size {
            if a.is_accept(n) {
                accept[n as usize] = true;
            }
            if n >= a.get_num_states() {
                continue;
            }
            // Java looks each point up with `next` (a binary search from the
            // last hit). A deterministic state's transitions are sorted by
            // `min` and disjoint, and the points ascend, so one merge-like
            // pass over both gives the same row: a point's destination is
            // that of the transition covering it, or -1 between them.
            let row = &mut transitions[n as usize * np..(n as usize + 1) * np];
            let count = a.init_transition(n, &mut t);
            let mut left = count;
            let (mut lo, mut hi, mut dest) = (i32::MAX, i32::MAX, -1);
            if left > 0 {
                a.get_next_transition(&mut t);
                left -= 1;
                (lo, hi, dest) = (t.min, t.max, t.dest);
            }
            for (slot, &p) in row.iter_mut().zip(&points) {
                while hi < p && left > 0 {
                    a.get_next_transition(&mut t);
                    left -= 1;
                    (lo, hi, dest) = (t.min, t.max, t.dest);
                }
                *slot = if lo <= p && p <= hi { dest } else { -1 };
            }
            debug_assert!(points
                .iter()
                .zip(row.iter())
                .all(|(&p, &d)| a.step(n, p) == d));
        }
        debug_assert!(np >= 1 && points[0] == 0);
        let mut classmap = Box::new([0u8; 256]);
        let mut i = 0usize;
        for (j, slot) in classmap.iter_mut().enumerate() {
            if i + 1 < np && j as i32 == points[i + 1] {
                i += 1;
            }
            *slot = u8::try_from(i).expect("a class below 256 is at most 255");
        }
        Ok(RunAutomaton {
            automaton: a,
            alphabet_size,
            size,
            accept,
            transitions,
            stride: np,
            points,
            classmap,
        })
    }

    /// The deterministic automaton this was built from.
    pub fn automaton(&self) -> &Automaton {
        &self.automaton
    }

    /// `getSize()`.
    #[inline]
    pub fn get_size(&self) -> i32 {
        self.size
    }

    /// `isAccept(state)`.
    #[inline]
    pub fn is_accept(&self, state: i32) -> bool {
        self.accept[state as usize]
    }

    /// `getCharIntervals()`: the character-class start points.
    pub fn get_char_intervals(&self) -> Vec<i32> {
        self.points.clone()
    }

    fn get_char_class(&self, c: i32) -> usize {
        let (mut a, mut b) = (0usize, self.points.len());
        while b - a > 1 {
            let d = (a + b) >> 1;
            if self.points[d] > c {
                b = d;
            } else if self.points[d] < c {
                a = d;
            } else {
                return d;
            }
        }
        a
    }

    /// `step(state, c)`.
    // SENTINEL: -1 means no transition on `c`.
    #[inline]
    pub fn step(&self, state: i32, c: i32) -> i32 {
        let cls = if (c as u32) < 256 {
            self.classmap[c as usize] as usize
        } else {
            self.get_char_class(c)
        };
        self.transitions[state as usize * self.stride + cls]
    }

    /// `step(state, b)` for a byte label: one table load for the class, one
    /// for the transition.
    // SENTINEL: -1 means no transition on `b`.
    #[inline]
    pub fn step_byte(&self, state: i32, b: u8) -> i32 {
        self.transitions[state as usize * self.stride + self.classmap[b as usize] as usize]
    }

    /// Whether the label sequence is accepted: the loop behind
    /// [`CharacterRunAutomaton::run`], with the transition read unchecked
    /// for the reason given in [`Self::run_bytes`].
    #[inline]
    fn run_labels(&self, labels: impl IntoIterator<Item = i32>) -> bool {
        let mut p = 0usize;
        for c in labels {
            let cls = if (c as u32) < 256 {
                self.classmap[c as usize] as usize
            } else {
                self.get_char_class(c)
            };
            debug_assert!(p < self.size as usize && cls < self.stride);
            // SAFETY: as in `run_bytes`: `p < size` and `cls < stride`
            // (`get_char_class` returns an index into `points` too), and
            // `transitions.len() == size * stride`.
            let next = unsafe { *self.transitions.get_unchecked(p * self.stride + cls) };
            if next < 0 {
                return false;
            }
            p = next as usize;
        }
        self.accept[p]
    }

    /// Whether the label sequence `s` (labels below 256) is accepted: the
    /// loop behind [`ByteRunAutomaton`]'s `run`.
    #[inline]
    fn run_bytes(&self, s: &[u8]) -> bool {
        debug_assert!(self.size >= 1 && self.transitions.len() == self.size as usize * self.stride);
        let mut p = 0usize;
        for &b in s {
            let cls = self.classmap[b as usize] as usize;
            debug_assert!(p < self.size as usize && cls < self.stride);
            // SAFETY: `transitions.len() == size * stride` (built so in
            // `new`, never changed). `p < size`: it starts at 0 and `size >=
            // 1`, and every later `p` is a non-negative entry of
            // `transitions`, which is a destination state of the automaton,
            // below its state count and so below `size`. `cls < stride`: a
            // class is an index into `points` (see the `classmap` field). So
            // `p * stride + cls < (p + 1) * stride <= size * stride`.
            let next = unsafe { *self.transitions.get_unchecked(p * self.stride + cls) };
            if next < 0 {
                return false;
            }
            p = next as usize;
        }
        self.accept[p]
    }
}

impl std::fmt::Display for RunAutomaton {
    /// Lucene's `RunAutomaton.toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut b = String::from("initial state: 0\n");
        let np = self.points.len();
        for i in 0..self.size as usize {
            b.push_str(&format!("state {i}"));
            b.push_str(if self.accept[i] {
                " [accept]:\n"
            } else {
                " [reject]:\n"
            });
            for j in 0..np {
                let k = self.transitions[i * np + j];
                if k != -1 {
                    let min = self.points[j];
                    let max = if j + 1 < np {
                        self.points[j + 1] - 1
                    } else {
                        self.alphabet_size
                    };
                    b.push(' ');
                    append_char_string(min, &mut b);
                    if min != max {
                        b.push('-');
                        append_char_string(max, &mut b);
                    }
                    b.push_str(&format!(" -> {k}\n"));
                }
            }
        }
        f.write_str(&b)
    }
}

/// `ByteRunAutomaton`: a [`RunAutomaton`] over UTF-8 bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByteRunAutomaton(pub RunAutomaton);

impl ByteRunAutomaton {
    /// `new ByteRunAutomaton(a, isBinary)`: `a` is converted from code points
    /// to UTF-8 (and re-determinized) unless `is_binary`.
    ///
    /// # Errors
    /// `IllegalArgument` when `a` is not deterministic.
    pub fn new(a: &Automaton, is_binary: bool) -> Result<Self, AutomatonError> {
        let a = if is_binary { a.clone() } else { convert(a)? };
        Ok(ByteRunAutomaton(RunAutomaton::new(a, 256)?))
    }

    /// `new ByteRunAutomaton(a, true)` taking the binary automaton by value,
    /// so it is moved into the table rather than copied.
    ///
    /// # Errors
    /// `IllegalArgument` when `a` is not deterministic.
    pub fn from_binary(a: Automaton) -> Result<Self, AutomatonError> {
        Ok(ByteRunAutomaton(RunAutomaton::new(a, 256)?))
    }

    /// The underlying [`RunAutomaton`].
    pub fn run_automaton(&self) -> &RunAutomaton {
        &self.0
    }
}

fn convert(a: &Automaton) -> Result<Automaton, AutomatonError> {
    if !a.is_deterministic() {
        return Err(AutomatonError::IllegalArgument(
            "Automaton must be deterministic".into(),
        ));
    }
    Ok(operations::determinize_owned(
        Utf32ToUtf8::new().convert(a),
        i32::MAX,
    )?)
}

impl ByteRunnable for ByteRunAutomaton {
    #[inline]
    fn step(&self, state: i32, c: i32) -> i32 {
        self.0.step(state, c)
    }
    #[inline]
    fn is_accept(&self, state: i32) -> bool {
        self.0.is_accept(state)
    }
    #[inline]
    fn get_size(&self) -> i32 {
        self.0.get_size()
    }
    /// The byte loop on the flat table: no virtual call or class search per
    /// byte, even when called through `&dyn ByteRunnable`.
    #[inline]
    fn run(&self, s: &[u8]) -> bool {
        self.0.run_bytes(s)
    }
}

/// `CharacterRunAutomaton`: a [`RunAutomaton`] over code points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CharacterRunAutomaton(pub RunAutomaton);

impl CharacterRunAutomaton {
    /// `new CharacterRunAutomaton(a)`.
    ///
    /// # Errors
    /// `IllegalArgument` when `a` is not deterministic.
    pub fn new(a: &Automaton) -> Result<Self, AutomatonError> {
        Ok(CharacterRunAutomaton(RunAutomaton::new(
            a.clone(),
            MAX_CODE_POINT + 1,
        )?))
    }

    /// `run(String)`.
    pub fn run(&self, s: &str) -> bool {
        self.run_code_points(s.chars().map(|c| c as i32))
    }

    /// `run(char[], ...)` over code points already decoded.
    pub fn run_code_points(&self, cps: impl IntoIterator<Item = i32>) -> bool {
        self.0.run_labels(cps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::automata::{make_char_range, make_string};
    use crate::automaton::operations::union2;

    #[test]
    fn run_automata() {
        let a = make_string("h\u{e9}");
        let c = CharacterRunAutomaton::new(&a).unwrap();
        assert!(c.run("h\u{e9}") && !c.run("he") && !c.run("h"));
        let b = ByteRunAutomaton::new(&a, false).unwrap();
        assert!(b.run("h\u{e9}".as_bytes()) && !b.run(b"he"));
        assert!(b.get_size() >= 3);
        assert_eq!(b.run_automaton().get_char_intervals()[0], 0);
        let text = c.0.to_string();
        assert!(text.starts_with("initial state: 0\nstate 0 [reject]:\n h -> 1\n"));
        let nfa = union2(&make_string("ab"), &make_string("ac"));
        if !nfa.is_deterministic() {
            assert!(CharacterRunAutomaton::new(&nfa).is_err());
            assert!(ByteRunAutomaton::new(&nfa, false).is_err());
        }
        // The flat byte loop agrees with stepping one label at a time.
        let words = crate::automaton::RegExp::new("(ab|c[d-f]+)*\u{e9}?")
            .unwrap()
            .to_automaton()
            .unwrap();
        let words = operations::determinize(&words, 10_000).unwrap();
        let br = ByteRunAutomaton::new(&words, false).unwrap();
        for t in [
            "",
            "ab",
            "abcdd",
            "cfe\u{e9}",
            "ac",
            "\u{e9}",
            "abab\u{e9}x",
            "\u{ff}",
        ] {
            let by_step = {
                let mut p = 0;
                for &x in t.as_bytes() {
                    let q = br.0.step_byte(p, x);
                    assert_eq!(q, ByteRunnable::step(&br, p, i32::from(x)));
                    p = q;
                    if p == -1 {
                        break;
                    }
                }
                p != -1 && br.is_accept(p)
            };
            assert_eq!(br.run(t.as_bytes()), by_step, "{t}");
            let as_dyn: &dyn ByteRunnable = &br;
            assert_eq!(as_dyn.run(t.as_bytes()), by_step, "{t}");
        }
        // Labels past the class table, and negative ones, take the search.
        let wide = CharacterRunAutomaton::new(&make_char_range(300, 400)).unwrap();
        assert_eq!(wide.0.step(0, 350), wide.0.step(0, 300));
        assert_eq!(wide.0.step(0, -5), -1);
        assert_eq!(wide.0.step(0, 255), -1);
        let big = CharacterRunAutomaton::new(&make_char_range(0x1000, 0x10FFFF)).unwrap();
        assert!(big.run("\u{10FFFF}") && !big.run("a"));
        let empty = CharacterRunAutomaton::new(&Automaton::new()).unwrap();
        assert!(!empty.run(""));
        assert_eq!(empty.0.automaton().get_num_states(), 0);
    }
}
