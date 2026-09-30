//! `CompiledAutomaton`: an automaton prepared for term-dictionary
//! intersection -- classified into `NONE`/`ALL`/`SINGLE`/`NORMAL`, converted
//! to UTF-8 and tabulated, with its common suffix precomputed.
//!
//! `getTermsEnum` and `visit` need `Terms`/`QueryVisitor`, which live above
//! this crate; callers dispatch on [`CompiledAutomaton::automaton_type`]
//! themselves.

use super::automaton::{Automaton, Transition, TransitionAccessor};
use super::code_points_to_utf8;
use super::error::AutomatonError;
use super::nfa_run_automaton::NfaRunAutomaton;
use super::operations;
use super::run_automaton::{ByteRunAutomaton, ByteRunnable};
use super::utf32_to_utf8::Utf32ToUtf8;

/// `CompiledAutomaton.AUTOMATON_TYPE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub enum AutomatonType {
    /// Accepts nothing.
    NONE,
    /// Accepts everything.
    ALL,
    /// Accepts exactly one term.
    SINGLE,
    /// Anything else: intersect.
    NORMAL,
}

/// `CompiledAutomaton`.
pub struct CompiledAutomaton {
    /// `type`.
    pub automaton_type: AutomatonType,
    /// `term`: the one accepted term (UTF-8, or raw bytes if binary) for
    /// [`AutomatonType::SINGLE`].
    pub term: Option<Vec<u8>>,
    /// `runAutomaton` (for `NORMAL` when determinized).
    pub run_automaton: Option<ByteRunAutomaton>,
    /// `nfaRunAutomaton` (for `NORMAL` when left non-deterministic).
    pub nfa_run_automaton: Option<NfaRunAutomaton>,
    /// `commonSuffixRef`: the suffix every accepted term shares, if any and
    /// if cheap enough to compute.
    pub common_suffix_ref: Option<Vec<u8>>,
    /// `finite`: whether the caller declared the language finite.
    pub finite: bool,
    /// `sinkState`: an accept state looping on every byte, or `-1`.
    pub sink_state: i32,
}

impl CompiledAutomaton {
    /// `new CompiledAutomaton(automaton)`: not finite, simplified, code
    /// points.
    ///
    /// # Errors
    /// As [`CompiledAutomaton::with_options`].
    pub fn new(automaton: &Automaton) -> Result<Self, AutomatonError> {
        Self::with_options(automaton, false, true, false)
    }

    /// `new CompiledAutomaton(automaton, finite, simplify, isBinary)`.
    ///
    /// # Errors
    /// Only from UTF-8 conversion/determinization, which Lucene runs without
    /// a work limit, so in practice none.
    pub fn with_options(
        automaton: &Automaton,
        finite: bool,
        simplify: bool,
        is_binary: bool,
    ) -> Result<Self, AutomatonError> {
        let placeholder;
        let automaton = if automaton.get_num_states() == 0 {
            let mut a = Automaton::new();
            a.create_state();
            placeholder = a;
            &placeholder
        } else {
            automaton
        };
        let simple = |t: AutomatonType, term: Option<Vec<u8>>, finite: bool| CompiledAutomaton {
            automaton_type: t,
            term,
            run_automaton: None,
            nfa_run_automaton: None,
            common_suffix_ref: None,
            finite,
            sink_state: -1,
        };
        if simplify && automaton.is_deterministic() {
            if operations::is_empty(automaton) {
                return Ok(simple(AutomatonType::NONE, None, true));
            }
            let is_total = if is_binary {
                operations::is_total_range(automaton, 0, 0xff)
            } else {
                operations::is_total(automaton)
            };
            if is_total {
                return Ok(simple(AutomatonType::ALL, None, false));
            }
            if let Some(singleton) = operations::get_singleton(automaton)? {
                let term = if is_binary {
                    singleton.iter().map(|&b| b as u8).collect()
                } else {
                    code_points_to_utf8(&singleton)
                };
                return Ok(simple(AutomatonType::SINGLE, Some(term), true));
            }
        }
        let binary = if is_binary {
            automaton.clone()
        } else {
            Utf32ToUtf8::new().convert(automaton)
        };
        let common_suffix_ref = if finite
            || automaton.get_num_states() + automaton.get_total_num_transitions() > 1000
        {
            None
        } else {
            let suffix = operations::get_common_suffix_bytes_ref(&binary)?;
            if suffix.is_empty() {
                None
            } else {
                Some(suffix)
            }
        };
        let mut out = CompiledAutomaton {
            automaton_type: AutomatonType::NORMAL,
            term: None,
            run_automaton: None,
            nfa_run_automaton: None,
            common_suffix_ref,
            finite,
            sink_state: -1,
        };
        if !automaton.is_deterministic() && !binary.is_deterministic() {
            out.nfa_run_automaton = Some(NfaRunAutomaton::with_alphabet(binary, 0xff));
        } else {
            let det = operations::determinize(&binary, i32::MAX)?;
            let run = ByteRunAutomaton::new(&det, true)?;
            // SENTINEL-OK: `sinkState` is public with the same `-1` meaning
            // in Java; it is stored, not used as an index.
            out.sink_state = find_sink_state(run.run_automaton().automaton());
            out.run_automaton = Some(run);
        }
        Ok(out)
    }

    /// `automaton`: the determinized UTF-8 automaton behind `run_automaton`.
    pub fn automaton(&self) -> Option<&Automaton> {
        self.run_automaton
            .as_ref()
            .map(|r| r.run_automaton().automaton())
    }

    /// `getByteRunnable()`.
    pub fn get_byte_runnable(&self) -> Option<&dyn ByteRunnable> {
        match (&self.nfa_run_automaton, &self.run_automaton) {
            (Some(n), _) => Some(n),
            (None, Some(r)) => Some(r),
            (None, None) => None,
        }
    }

    /// `getTransitionAccessor()`.
    pub fn get_transition_accessor(&self) -> Option<&dyn TransitionAccessor> {
        match (&self.nfa_run_automaton, self.automaton()) {
            (Some(n), _) => Some(n),
            (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }

    fn add_tail(
        &self,
        mut state: i32,
        term: &mut Vec<u8>,
        mut idx: usize,
        lead_label: i32,
    ) -> Vec<u8> {
        let (automaton, _) = self.normal_parts();
        let mut t = Transition::new();
        let mut max_index = -1;
        let n = automaton.init_transition(state, &mut t);
        for i in 0..n {
            automaton.get_next_transition(&mut t);
            if t.min < lead_label {
                max_index = i;
            } else {
                break;
            }
        }
        automaton.get_transition(state, max_index, &mut t);
        let floor_label = if t.max > lead_label - 1 {
            lead_label - 1
        } else {
            t.max
        };
        set_byte_at(term, idx, floor_label as u8);
        state = t.dest;
        idx += 1;
        loop {
            let n = automaton.get_num_transitions(state);
            if n == 0 {
                term.truncate(idx);
                return term.clone();
            }
            automaton.get_transition(state, n - 1, &mut t);
            set_byte_at(term, idx, t.max as u8);
            state = t.dest;
            idx += 1;
        }
    }

    fn normal_parts(&self) -> (&Automaton, &ByteRunAutomaton) {
        let run = self
            .run_automaton
            .as_ref()
            .expect("floor() needs a determinized NORMAL automaton");
        (run.run_automaton().automaton(), run)
    }

    /// `floor(input, output)`: the largest accepted term `<= input`, or
    /// `None`. Only valid for a determinized [`AutomatonType::NORMAL`] with a
    /// finite language -- as in Lucene, an infinite one can loop forever.
    ///
    /// # Panics
    /// When there is no `run_automaton` (Java would throw a
    /// `NullPointerException`).
    pub fn floor(&self, input: &[u8]) -> Option<Vec<u8>> {
        let (automaton, run) = self.normal_parts();
        let mut output: Vec<u8> = Vec::new();
        let mut state = 0;
        if input.is_empty() {
            return if run.is_accept(state) {
                Some(Vec::new())
            } else {
                None
            };
        }
        let mut stack: Vec<i32> = Vec::new();
        let mut idx = 0usize;
        let mut t = Transition::new();
        loop {
            let mut label = i32::from(input[idx]);
            let mut next_state = run.step(state, label);
            if idx == input.len() - 1 {
                if next_state != -1 && run.is_accept(next_state) {
                    set_byte_at(&mut output, idx, label as u8);
                    output.truncate(input.len());
                    return Some(output);
                }
                next_state = -1;
            }
            if next_state == -1 {
                loop {
                    let n = automaton.get_num_transitions(state);
                    if n == 0 {
                        output.truncate(idx);
                        return Some(output);
                    }
                    automaton.get_transition(state, 0, &mut t);
                    if label - 1 < t.min {
                        if run.is_accept(state) {
                            output.truncate(idx);
                            return Some(output);
                        }
                        state = stack.pop()?;
                        idx -= 1;
                        label = i32::from(input[idx]);
                    } else {
                        break;
                    }
                }
                return Some(self.add_tail(state, &mut output, idx, label));
            }
            set_byte_at(&mut output, idx, label as u8);
            stack.push(state);
            state = next_state;
            idx += 1;
        }
    }
}

/// `BytesRefBuilder.grow(idx + 1); setByteAt(idx, b)`, keeping the length at
/// least `idx + 1` (`floor` trims it explicitly before returning).
fn set_byte_at(v: &mut Vec<u8>, idx: usize, b: u8) {
    if v.len() <= idx {
        v.resize(idx + 1, 0);
    }
    v[idx] = b;
}

/// Java's private `findSinkState`: the first accept state with a `0..=255`
/// self loop, or `-1`.
// SENTINEL: -1 means no sink state.
fn find_sink_state(automaton: &Automaton) -> i32 {
    let mut t = Transition::new();
    for s in 0..automaton.get_num_states() {
        if automaton.is_accept(s) {
            let n = automaton.init_transition(s, &mut t);
            for _ in 0..n {
                automaton.get_next_transition(&mut t);
                if t.dest == s && t.min == 0 && t.max == 0xff {
                    return s;
                }
            }
        }
    }
    -1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::RegExp;

    fn compile(re: &str) -> CompiledAutomaton {
        let a = RegExp::new(re).unwrap().to_automaton().unwrap();
        let a = operations::determinize(&a, 10_000).unwrap();
        CompiledAutomaton::new(&a).unwrap()
    }

    #[test]
    fn classifies() {
        assert_eq!(compile("#").automaton_type, AutomatonType::NONE);
        assert_eq!(compile(".*").automaton_type, AutomatonType::ALL);
        let s = compile("h\u{e9}");
        assert_eq!(s.automaton_type, AutomatonType::SINGLE);
        assert_eq!(s.term.as_deref(), Some("h\u{e9}".as_bytes()));
        let n = compile("a.*ing");
        assert_eq!(n.automaton_type, AutomatonType::NORMAL);
        assert_eq!(n.common_suffix_ref.as_deref(), Some(&b"ing"[..]));
        assert!(n.get_byte_runnable().unwrap().run(b"abcing"));
        assert!(n.get_transition_accessor().is_some());
        let p = compile("ab.*");
        assert_eq!(p.sink_state, -1, "UTF-8 .* is not a 0..=255 loop");
        assert!(p.common_suffix_ref.is_none());
        let prefix =
            crate::automaton::automata::make_binary_interval(Some(b"ab"), true, None, true)
                .unwrap();
        let p = CompiledAutomaton::with_options(&prefix, false, true, true).unwrap();
        assert!(p.sink_state >= 0);
        let empty = CompiledAutomaton::new(&Automaton::new()).unwrap();
        assert_eq!(empty.automaton_type, AutomatonType::NONE);
        let bin = crate::automaton::automata::make_any_binary();
        let b = CompiledAutomaton::with_options(&bin, false, true, true).unwrap();
        assert_eq!(b.automaton_type, AutomatonType::ALL);
        let one = crate::automaton::automata::make_binary(&[0xFF, 0]);
        let b = CompiledAutomaton::with_options(&one, false, true, true).unwrap();
        assert_eq!(b.term.as_deref(), Some(&[0xFF, 0][..]));
        let nfa = RegExp::new("(a|ab)*c").unwrap().to_automaton().unwrap();
        assert!(!nfa.is_deterministic());
        let c = CompiledAutomaton::new(&nfa).unwrap();
        assert!(c.nfa_run_automaton.is_some());
        assert!(c.get_byte_runnable().unwrap().run(b"ababc"));
        assert!(c.get_transition_accessor().is_some());
        assert!(c.automaton().is_none());
        let none = compile("#");
        assert!(none.get_byte_runnable().is_none() && none.get_transition_accessor().is_none());
    }

    #[test]
    fn floor() {
        let c = compile("(abc|abd|b|x.z)");
        assert_eq!(c.floor(b"abe").as_deref(), Some(&b"abd"[..]));
        assert_eq!(c.floor(b"abd").as_deref(), Some(&b"abd"[..]));
        assert_eq!(c.floor(b"abcc").as_deref(), Some(&b"abc"[..]));
        assert_eq!(c.floor(b"aa"), None);
        assert_eq!(c.floor(b""), None);
        assert_eq!(c.floor(b"c").as_deref(), Some(&b"b"[..]));
        assert_eq!(c.floor(b"zz").as_deref(), Some("x\u{10FFFF}z".as_bytes()));
        let e = compile("a?");
        assert_eq!(e.floor(b"").as_deref(), Some(&b""[..]));
    }
}
