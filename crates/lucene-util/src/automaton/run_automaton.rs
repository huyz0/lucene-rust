//! `RunAutomaton`, `ByteRunAutomaton`, `CharacterRunAutomaton` and the
//! `ByteRunnable` interface: a deterministic automaton flattened into a
//! `state x character class` table for fast stepping.

use super::automaton::{append_char_string, Automaton, Transition};
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
    /// `delta(state, c) = transitions[state * points.len() + class(c)]`.
    transitions: Vec<i32>,
    points: Vec<i32>,
    classmap: Vec<i32>,
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
            t.source = n;
            t.transition_upto = -1;
            for (c, &p) in points.iter().enumerate() {
                let dest = a.next(&mut t, p);
                transitions[n as usize * np + c] = dest;
            }
        }
        let cm_len = alphabet_size.min(256) as usize;
        let mut classmap = vec![0i32; cm_len];
        let mut i = 0usize;
        for (j, slot) in classmap.iter_mut().enumerate() {
            if i + 1 < np && j as i32 == points[i + 1] {
                i += 1;
            }
            *slot = i as i32;
        }
        Ok(RunAutomaton {
            automaton: a,
            alphabet_size,
            size,
            accept,
            transitions,
            points,
            classmap,
        })
    }

    /// The deterministic automaton this was built from.
    pub fn automaton(&self) -> &Automaton {
        &self.automaton
    }

    /// `getSize()`.
    pub fn get_size(&self) -> i32 {
        self.size
    }

    /// `isAccept(state)`.
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
    pub fn step(&self, state: i32, c: i32) -> i32 {
        let cls = match usize::try_from(c).ok().and_then(|u| self.classmap.get(u)) {
            Some(&cls) => cls as usize,
            None => self.get_char_class(c),
        };
        self.transitions[state as usize * self.points.len() + cls]
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
    Ok(operations::determinize(
        &Utf32ToUtf8::new().convert(a),
        i32::MAX,
    )?)
}

impl ByteRunnable for ByteRunAutomaton {
    fn step(&self, state: i32, c: i32) -> i32 {
        self.0.step(state, c)
    }
    fn is_accept(&self, state: i32) -> bool {
        self.0.is_accept(state)
    }
    fn get_size(&self) -> i32 {
        self.0.get_size()
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
        let mut p = 0;
        for cp in cps {
            p = self.0.step(p, cp);
            if p == -1 {
                return false;
            }
        }
        self.0.is_accept(p)
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
        let big = CharacterRunAutomaton::new(&make_char_range(0x1000, 0x10FFFF)).unwrap();
        assert!(big.run("\u{10FFFF}") && !big.run("a"));
        let empty = CharacterRunAutomaton::new(&Automaton::new()).unwrap();
        assert!(!empty.run(""));
        assert_eq!(empty.0.automaton().get_num_states(), 0);
    }
}
