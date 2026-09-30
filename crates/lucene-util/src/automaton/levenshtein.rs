//! `LevenshteinAutomata`: the DFA of every string within edit distance `n`
//! (1 or 2; optionally counting a transposition as one edit) of a word,
//! built from Lucene's precomputed parametric descriptions
//! (`Lev1ParametricDescription` & co., in [`super::lev_tables`]).

use super::automata;
use super::automaton::Automaton;
use super::error::AutomatonError;
use super::lev_tables::{Description, LEV1, LEV1T, LEV2, LEV2T};
use super::operations;
use super::MAX_CODE_POINT;

/// `LevenshteinAutomata.MAXIMUM_SUPPORTED_DISTANCE`.
pub const MAXIMUM_SUPPORTED_DISTANCE: i32 = 2;

/// A `ParametricDescription` bound to a word length `w`.
struct Parametric {
    w: i32,
    d: &'static Description,
}

impl Parametric {
    fn size(&self) -> i32 {
        self.d.min_errors.len() as i32 * (self.w + 1)
    }

    fn is_accept(&self, abs_state: i32) -> bool {
        let state = abs_state / (self.w + 1);
        let offset = abs_state % (self.w + 1);
        self.w - offset + self.d.min_errors[state as usize] <= self.d.n
    }

    fn get_position(&self, abs_state: i32) -> i32 {
        abs_state % (self.w + 1)
    }

    /// `transition(absState, position, vector)`.
    // SENTINEL: -1 means the transition leads to the dead state.
    fn transition(&self, abs_state: i32, position: i32, vector: i32) -> i32 {
        let mut state = abs_state / (self.w + 1);
        let mut offset = abs_state % (self.w + 1);
        let last = self.d.levels.len() as i32 - 1;
        let level = &self.d.levels[(self.w - position).min(last) as usize];
        if state < level.state_limit {
            let loc = vector * level.state_limit + state;
            offset += unpack(level.offset_incrs, loc, level.offset_bits);
            state = unpack(level.to_states, loc, level.to_bits) - 1;
        }
        if state == -1 {
            -1
        } else {
            state * (self.w + 1) + offset
        }
    }
}

/// `ParametricDescription.unpack`.
fn unpack(data: &[u64], index: i32, bits_per_value: u32) -> i32 {
    let bit_loc = u64::from(bits_per_value) * index as u64;
    let data_loc = (bit_loc >> 6) as usize;
    let bit_start = (bit_loc & 63) as u32;
    let mask = |bits: u32| (1u64 << bits) - 1;
    if bit_start + bits_per_value <= 64 {
        ((data[data_loc] >> bit_start) & mask(bits_per_value)) as i32
    } else {
        let part = 64 - bit_start;
        (((data[data_loc] >> bit_start) & mask(part))
            + ((data[1 + data_loc] & mask(bits_per_value - part)) << part)) as i32
    }
}

/// `LevenshteinAutomata`.
pub struct LevenshteinAutomata {
    word: Vec<i32>,
    alphabet: Vec<i32>,
    range_lower: Vec<i32>,
    range_upper: Vec<i32>,
    descriptions: [Option<Parametric>; 3],
}

impl LevenshteinAutomata {
    /// `new LevenshteinAutomata(String, withTranspositions)`: over code
    /// points.
    pub fn new(input: &str, with_transpositions: bool) -> Self {
        let word: Vec<i32> = input.chars().map(|c| c as i32).collect();
        Self::with_alphabet(word, MAX_CODE_POINT, with_transpositions)
            .expect("code points never exceed MAX_CODE_POINT")
    }

    /// `new LevenshteinAutomata(int[] word, alphaMax, withTranspositions)`.
    ///
    /// # Errors
    /// `IllegalArgument` when a symbol of `word` exceeds `alpha_max`.
    pub fn with_alphabet(
        word: Vec<i32>,
        alpha_max: i32,
        with_transpositions: bool,
    ) -> Result<Self, AutomatonError> {
        for &v in &word {
            if v > alpha_max {
                return Err(AutomatonError::IllegalArgument(format!(
                    "alphaMax exceeded by symbol {v} in word"
                )));
            }
        }
        let mut alphabet = word.clone();
        alphabet.sort_unstable();
        alphabet.dedup();
        let mut range_lower = Vec::new();
        let mut range_upper = Vec::new();
        let mut lower = 0;
        for &higher in &alphabet {
            if higher > lower {
                range_lower.push(lower);
                range_upper.push(higher - 1);
            }
            lower = higher + 1;
        }
        if lower <= alpha_max {
            range_lower.push(lower);
            range_upper.push(alpha_max);
        }
        let w = word.len() as i32;
        let descriptions = [
            None,
            Some(Parametric {
                w,
                d: if with_transpositions { &LEV1T } else { &LEV1 },
            }),
            Some(Parametric {
                w,
                d: if with_transpositions { &LEV2T } else { &LEV2 },
            }),
        ];
        Ok(LevenshteinAutomata {
            word,
            alphabet,
            range_lower,
            range_upper,
            descriptions,
        })
    }

    /// `toAutomaton(n)`.
    pub fn to_automaton(&self, n: i32) -> Option<Automaton> {
        self.to_automaton_with_prefix(n, "")
    }

    /// `toAutomaton(n, prefix)`: `prefix` followed by the edit-distance-`n`
    /// language of the word; `None` for `n` above
    /// [`MAXIMUM_SUPPORTED_DISTANCE`].
    pub fn to_automaton_with_prefix(&self, n: i32, prefix: &str) -> Option<Automaton> {
        if n == 0 {
            let mut labels: Vec<i32> = prefix.chars().map(|c| c as i32).collect();
            labels.extend_from_slice(&self.word);
            return Some(automata::make_string_ints(&labels));
        }
        let description = self.descriptions.get(usize::try_from(n).ok()?)?.as_ref()?;
        let range = 2 * n + 1;
        let num_states = description.size();
        let mut a = Automaton::new();
        let mut last = a.create_state();
        for c in prefix.chars() {
            let st = a.create_state();
            a.add_transition(last, st, c as i32, c as i32);
            last = st;
        }
        let state_offset = last;
        a.set_accept(last, description.is_accept(0));
        for i in 1..num_states {
            let st = a.create_state();
            a.set_accept(st, description.is_accept(i));
        }
        let wl = self.word.len() as i32;
        for k in 0..num_states {
            let xpos = description.get_position(k);
            if xpos < 0 {
                continue;
            }
            let end = xpos + (wl - xpos).min(range);
            for &ch in &self.alphabet {
                let cvec = self.get_vector(ch, xpos, end);
                let dest = description.transition(k, xpos, cvec);
                if dest >= 0 {
                    a.add_transition(state_offset + k, state_offset + dest, ch, ch);
                }
            }
            let dest = description.transition(k, xpos, 0);
            if dest >= 0 {
                for (&lo, &hi) in self.range_lower.iter().zip(&self.range_upper) {
                    a.add_transition(state_offset + k, state_offset + dest, lo, hi);
                }
            }
        }
        a.finish_state();
        Some(operations::remove_dead_states(&a))
    }

    fn get_vector(&self, x: i32, pos: i32, end: i32) -> i32 {
        let mut vector = 0;
        for i in pos..end {
            vector <<= 1;
            if self.word[i as usize] == x {
                vector |= 1;
            }
        }
        vector
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::operations::run;

    #[test]
    fn edit_distances() {
        let lev = LevenshteinAutomata::new("kitten", false);
        let a0 = lev.to_automaton(0).unwrap();
        assert!(run(&a0, "kitten") && !run(&a0, "sitten"));
        let a1 = lev.to_automaton(1).unwrap();
        assert!(a1.is_deterministic());
        assert!(run(&a1, "sitten") && run(&a1, "kiten") && run(&a1, "kittens"));
        assert!(!run(&a1, "sittin") && !run(&a1, "iktten"));
        let a2 = lev.to_automaton(2).unwrap();
        assert!(run(&a2, "sittin") && !run(&a2, "sitting"));
        assert!(lev.to_automaton(3).is_none());
        assert!(lev.to_automaton(-1).is_none());
        let t = LevenshteinAutomata::new("kitten", true);
        assert!(run(&t.to_automaton(1).unwrap(), "iktten"));
        let p = lev.to_automaton_with_prefix(1, "pre").unwrap();
        assert!(run(&p, "presitten") && !run(&p, "sitten"));
        let p0 = lev.to_automaton_with_prefix(0, "pre").unwrap();
        assert!(run(&p0, "prekitten"));
        assert!(LevenshteinAutomata::with_alphabet(vec![300], 255, false).is_err());
        let empty = LevenshteinAutomata::new("", false);
        let e1 = empty.to_automaton(1).unwrap();
        assert!(run(&e1, "") && run(&e1, "x") && !run(&e1, "xy"));
    }
}
