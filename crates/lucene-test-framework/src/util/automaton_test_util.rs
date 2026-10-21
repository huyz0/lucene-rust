//! `AutomatonTestUtil`: random regular expressions and automata, random
//! strings an automaton accepts, and brics' original (slow, obviously
//! correct) algorithms to check the optimised ones against.
//!
//! `isFinite`, `sameLanguage`, `subsetOf`, `determinizeSimple` and
//! `minimizeSimple` already live in `lucene_util::automaton::operations`
//! (Lucene 10 moved them out of core, and the port's own tests needed
//! them); they are re-exported here under their test-framework names.

use std::collections::{HashSet, VecDeque};

use lucene_util::automaton::operations;
use lucene_util::automaton::{Automaton, AutomatonError, RegExp, Transition, TransitionAccessor};
use lucene_util::java_random::JavaRandom;

pub use lucene_util::automaton::operations::{
    determinize_simple, determinize_simple_from, is_finite, reverse_original, same_language,
    subset_of,
};

use super::test_util::next_int;

/// `DEFAULT_MAX_DETERMINIZED_STATES`.
pub const DEFAULT_MAX_DETERMINIZED_STATES: i32 = 1_000_000;

const UNI_SUR_HIGH_START: i32 = 0xD800;
const UNI_SUR_HIGH_END: i32 = 0xDBFF;
const UNI_SUR_LOW_END: i32 = 0xDFFF;

/// `minimizeSimple(a)`: Brzozowski's minimization.
pub fn minimize_simple(a: &Automaton) -> Automaton {
    operations::minimize(a)
}

/// `randomRegexp(r)`: a random string that `RegExp` (no optional syntax)
/// parses, retried until one does.
pub fn random_regexp(r: &mut JavaRandom) -> String {
    loop {
        let units = random_regexp_units(r);
        // Undefined Unicode (lone surrogates) is generated too; skip it.
        let Ok(regexp) = String::from_utf16(&units) else {
            continue;
        };
        if RegExp::with_flags(&regexp, RegExp::NONE, 0).is_ok() {
            return regexp;
        }
    }
}

fn random_regexp_units(r: &mut JavaRandom) -> Vec<u16> {
    let end = r.next_int_bounded(20) as usize;
    let mut buffer = vec![0u16; end];
    let mut i = 0;
    while i < end {
        let t = r.next_int_bounded(15);
        if t == 0 && i + 1 < end {
            buffer[i] = next_int(r, 0xd800, 0xdbff) as u16;
            i += 1;
            buffer[i] = next_int(r, 0xdc00, 0xdfff) as u16;
        } else {
            buffer[i] = match t {
                0 | 1 => r.next_int_bounded(0x80) as u16,
                2 => next_int(r, 0x80, 0x800) as u16,
                3 => next_int(r, 0x800, 0xd7ff) as u16,
                4 => next_int(r, 0xe000, 0xffff) as u16,
                5 => u16::from(b'.'),
                6 => u16::from(b'?'),
                7 => u16::from(b'*'),
                8 => u16::from(b'+'),
                9 => u16::from(b'('),
                10 => u16::from(b')'),
                11 => u16::from(b'-'),
                12 => u16::from(b'['),
                13 => u16::from(b']'),
                _ => u16::from(b'|'),
            };
        }
        i += 1;
    }
    buffer
}

/// `getRandomCodePoint(r, min, max)`: a code point of `min..=max` that is
/// not a surrogate.
///
/// # Errors
///
/// `IllegalArgument` when the range holds only surrogates.
fn random_code_point(r: &mut JavaRandom, min: i32, max: i32) -> Result<i32, AutomatonError> {
    let only = || {
        Err(AutomatonError::IllegalArgument(format!(
            "transition accepts only surrogates: min={min} max={max}"
        )))
    };
    let code = if max < UNI_SUR_HIGH_START || min > UNI_SUR_HIGH_END {
        min + r.next_int_bounded(max - min + 1)
    } else if min >= UNI_SUR_HIGH_START {
        if max > UNI_SUR_LOW_END {
            1 + UNI_SUR_LOW_END + r.next_int_bounded(max - UNI_SUR_LOW_END)
        } else {
            return only();
        }
    } else if max <= UNI_SUR_LOW_END {
        // `min < UNI_SUR_HIGH_START` holds here (the branch above took the rest).
        min + r.next_int_bounded(UNI_SUR_HIGH_START - min)
    } else {
        let gap1 = UNI_SUR_HIGH_START - min;
        let gap2 = max - UNI_SUR_LOW_END;
        let c = r.next_int_bounded(gap1 + gap2);
        if c < gap1 {
            min + c
        } else {
            UNI_SUR_LOW_END + c - gap1 + 1
        }
    };
    debug_assert!(
        code >= min && code <= max && !(UNI_SUR_HIGH_START..=UNI_SUR_LOW_END).contains(&code)
    );
    Ok(code)
}

/// `RandomAcceptedStrings`: random strings (code points) an automaton
/// accepts, half the time steering towards an accept state.
pub struct RandomAcceptedStrings<'a> {
    a: &'a Automaton,
    transitions: Vec<Vec<Transition>>,
    /// `leadsToAccept`, by `(state, index into transitions[state])`: Java
    /// keys it by `Transition` identity.
    leads_to_accept: HashSet<(usize, usize)>,
}

impl<'a> RandomAcceptedStrings<'a> {
    /// `new RandomAcceptedStrings(a)`.
    ///
    /// # Errors
    ///
    /// `IllegalArgument` for an automaton without states.
    pub fn new(a: &'a Automaton) -> Result<Self, AutomatonError> {
        if a.get_num_states() == 0 {
            return Err(AutomatonError::IllegalArgument(
                "this automaton accepts nothing".into(),
            ));
        }
        let transitions = a.get_sorted_transitions();
        let n = transitions.len();
        // Every transition arriving at each state, in (source, order) order.
        let mut arriving: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        let mut q = VecDeque::new();
        let mut seen = vec![false; n];
        for (s, ts) in transitions.iter().enumerate() {
            for (i, t) in ts.iter().enumerate() {
                arriving[t.dest as usize].push((s, i));
            }
            if a.is_accept(s as i32) {
                q.push_back(s);
                seen[s] = true;
            }
        }
        // Breadth-first, backwards from the accept states.
        let mut leads_to_accept = HashSet::new();
        while let Some(s) = q.pop_front() {
            for &(from, i) in &arriving[s] {
                if !seen[from] {
                    q.push_back(from);
                    seen[from] = true;
                    leads_to_accept.insert((from, i));
                }
            }
        }
        Ok(Self {
            a,
            transitions,
            leads_to_accept,
        })
    }

    /// `getRandomAcceptedString(r)`.
    ///
    /// # Errors
    ///
    /// `IllegalState("this automaton has dead states")` on reaching a
    /// non-accepting state without transitions (Java: `RuntimeException`),
    /// `IllegalArgument` on a transition of surrogates only.
    pub fn get_random_accepted_string(
        &self,
        r: &mut JavaRandom,
    ) -> Result<Vec<i32>, AutomatonError> {
        let mut code_points = Vec::new();
        let mut s = 0usize;
        loop {
            let ts = &self.transitions[s];
            if self.a.is_accept(s as i32) && (ts.is_empty() || r.next_boolean()) {
                break;
            }
            if ts.is_empty() {
                return Err(AutomatonError::IllegalState(
                    "this automaton has dead states".into(),
                ));
            }
            let len = ts.len() as i32;
            let t = if r.next_boolean() {
                // The transitions known to be on a shortest path to accept.
                let to_accept: Vec<usize> = (0..ts.len())
                    .filter(|&i| self.leads_to_accept.contains(&(s, i)))
                    .collect();
                if to_accept.is_empty() {
                    // Fine: a jump into a cycle.
                    &ts[r.next_int_bounded(len) as usize]
                } else {
                    &ts[to_accept[r.next_int_bounded(to_accept.len() as i32) as usize]]
                }
            } else {
                &ts[r.next_int_bounded(len) as usize]
            };
            code_points.push(random_code_point(r, t.min, t.max)?);
            s = t.dest as usize;
        }
        Ok(code_points)
    }
}

fn random_single_automaton(r: &mut JavaRandom) -> Automaton {
    loop {
        let regexp = random_regexp(r);
        let a1 = RegExp::with_flags(&regexp, RegExp::NONE, 0)
            .and_then(|re| re.to_automaton())
            .expect("random_regexp returns a regexp that parses");
        if r.next_boolean() {
            match operations::complement(&a1, DEFAULT_MAX_DETERMINIZED_STATES) {
                Ok(c) => return c,
                // Rarely the random regexp is too hard; try again.
                Err(_) => continue,
            }
        }
        return a1;
    }
}

/// `randomAutomaton(random)`: two random regexps' automata (each maybe
/// complemented), concatenated, unioned, intersected or subtracted.
pub fn random_automaton(r: &mut JavaRandom) -> Automaton {
    let a1 = random_single_automaton(r);
    let a2 = random_single_automaton(r);
    match r.next_int_bounded(4) {
        0 => operations::concatenate(&[&a1, &a2]),
        1 => operations::union(&[&a1, &a2]),
        2 => operations::intersection(&a1, &a2),
        _ => operations::minus(&a1, &a2, DEFAULT_MAX_DETERMINIZED_STATES)
            .expect("minus over determinized random automata stays under the limit"),
    }
}

/// `assertMinimalDFA(a)`: a clean DFA as small as Brzozowski's.
///
/// # Panics
///
/// When it is not.
pub fn assert_minimal_dfa(a: &Automaton) {
    assert_clean_dfa(a);
    assert_eq!(
        minimize_simple(a).get_num_states(),
        a.get_num_states(),
        "not minimal"
    );
}

/// `assertCleanDFA(a)`: deterministic, no dead states.
///
/// # Panics
///
/// When it is not.
pub fn assert_clean_dfa(a: &Automaton) {
    assert_clean_nfa(a);
    assert!(a.is_deterministic(), "must be deterministic");
}

/// `assertCleanNFA(a)`: no dead states of any kind.
///
/// # Panics
///
/// On the first kind found.
pub fn assert_clean_nfa(a: &Automaton) {
    assert!(
        !operations::has_dead_states_from_initial(a),
        "has dead states reachable from initial"
    );
    assert!(
        !operations::has_dead_states_to_accept(a),
        "has dead states leading to accept"
    );
    assert!(
        !operations::has_dead_states(a),
        "has unreachable dead states (ghost states)"
    );
}

/// `assertNoDetachedStates(a)`: no state unreachable from the initial one.
///
/// # Panics
///
/// When some are.
pub fn assert_no_detached_states(a: &Automaton) {
    let a2 = operations::remove_dead_states(a);
    assert_eq!(
        a.get_num_states(),
        a2.get_num_states(),
        "automaton has {} detached states",
        a.get_num_states() - a2.get_num_states()
    );
}

/// `isDeterministicSlow(a)`: no state's transitions overlap; agrees with
/// `a.is_deterministic()` (asserted, as Java does).
pub fn is_deterministic_slow(a: &Automaton) -> bool {
    let mut t = Transition::new();
    for s in 0..a.get_num_states() {
        let count = a.init_transition(s, &mut t);
        let mut last_max = -1;
        for _ in 0..count {
            a.get_next_transition(&mut t);
            if t.min <= last_max {
                assert!(!a.is_deterministic());
                return false;
            }
            last_max = t.max;
        }
    }
    assert!(a.is_deterministic());
    true
}

/// `getFiniteStringsRecursive(a, limit)`: the accepted strings, found
/// depth-first; stops (returning what it has) at a cycle or past `limit`
/// strings (`limit < 0`: no limit).
pub fn get_finite_strings_recursive(a: &Automaton, limit: i32) -> HashSet<Vec<i32>> {
    let mut strings = HashSet::new();
    let mut path_states = HashSet::new();
    let mut path = Vec::new();
    finite_strings(a, 0, &mut path_states, &mut strings, &mut path, limit);
    strings
}

fn finite_strings(
    a: &Automaton,
    s: i32,
    path_states: &mut HashSet<i32>,
    strings: &mut HashSet<Vec<i32>>,
    path: &mut Vec<i32>,
    limit: i32,
) -> bool {
    path_states.insert(s);
    let mut t = Transition::new();
    let count = a.init_transition(s, &mut t);
    for _ in 0..count {
        a.get_next_transition(&mut t);
        if path_states.contains(&t.dest) {
            return false;
        }
        let (dest, min, max) = (t.dest, t.min, t.max);
        for n in min..=max {
            path.push(n);
            if a.is_accept(dest) {
                strings.insert(path.clone());
                if limit >= 0 && strings.len() > limit as usize {
                    return false;
                }
            }
            if !finite_strings(a, dest, path_states, strings, path, limit) {
                return false;
            }
            path.pop();
        }
    }
    path_states.remove(&s);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_util::automaton::automata;

    #[test]
    fn random_regexps_parse() {
        let mut r = JavaRandom::new(1);
        for _ in 0..200 {
            let re = random_regexp(&mut r);
            assert!(RegExp::with_flags(&re, RegExp::NONE, 0).is_ok(), "{re:?}");
        }
    }

    #[test]
    fn accepted_strings_are_accepted() {
        let mut r = JavaRandom::new(2);
        for _ in 0..100 {
            let a = random_automaton(&mut r);
            let Ok(ras) = RandomAcceptedStrings::new(&a) else {
                assert_eq!(a.get_num_states(), 0);
                continue;
            };
            // `run` steps a DFA only; the random automaton may be an NFA.
            let dfa = determinize_simple(&a);
            for _ in 0..5 {
                match ras.get_random_accepted_string(&mut r) {
                    Ok(s) => assert!(operations::run_ints(&dfa, &s), "{s:?}"),
                    Err(e) => assert!(
                        e.to_string().contains("dead states")
                            || e.to_string().contains("surrogates")
                    ),
                }
            }
        }
        assert!(RandomAcceptedStrings::new(&automata::make_empty()).is_err());
    }

    #[test]
    fn dead_states_and_surrogates_are_errors() {
        // State 0 -> 1 on 'a'; 1 is neither accepting nor has transitions.
        let mut a = Automaton::new();
        let s0 = a.create_state();
        let s1 = a.create_state();
        let s2 = a.create_state();
        a.add_transition(s0, s1, 'a' as i32, 'a' as i32);
        a.add_transition(s0, s2, 'b' as i32, 'b' as i32);
        a.set_accept(s2, true);
        a.finish_state();
        let ras = RandomAcceptedStrings::new(&a).unwrap();
        let mut r = JavaRandom::new(3);
        let mut dead = false;
        for _ in 0..50 {
            dead |= ras.get_random_accepted_string(&mut r).is_err();
        }
        assert!(dead);
        let mut r = JavaRandom::new(3);
        assert!(random_code_point(&mut r, 0xD800, 0xDFFF).is_err());
        for _ in 0..100 {
            let c = random_code_point(&mut r, 0xD000, 0xDFFF).unwrap();
            assert!(c < 0xD800);
            let c = random_code_point(&mut r, 0xD900, 0xE005).unwrap();
            assert!(c > 0xDFFF);
            let c = random_code_point(&mut r, 0xD7FE, 0xE001).unwrap();
            assert!(!(0xD800..=0xDFFF).contains(&c));
            let c = random_code_point(&mut r, 0x10000, 0x10FFFF).unwrap();
            assert!(c >= 0x10000);
        }
    }

    #[test]
    fn brics_helpers() {
        let ab = automata::make_string("ab");
        let nfa = operations::union(&[&ab, &automata::make_string("ac")]);
        let min = minimize_simple(&nfa);
        assert_minimal_dfa(&min);
        assert_clean_nfa(&nfa);
        assert_no_detached_states(&nfa);
        assert!(is_deterministic_slow(&min));
        assert!(!is_deterministic_slow(&nfa));
        let strings = get_finite_strings_recursive(&nfa, -1);
        assert_eq!(strings.len(), 2);
        assert!(strings.contains(&vec!['a' as i32, 'c' as i32]));
        assert_eq!(get_finite_strings_recursive(&nfa, 0).len(), 1);
        let star = automata::make_any_string();
        assert!(get_finite_strings_recursive(&star, -1).is_empty());
        let range = automata::make_char_range('a' as i32, 'c' as i32);
        assert_eq!(get_finite_strings_recursive(&range, -1).len(), 3);
        assert!(is_finite(&nfa).unwrap());
        assert!(same_language(&min, &determinize_simple(&nfa)).unwrap());
        assert!(subset_of(&min, &min).unwrap());
        let (rev, init) = reverse_original(&ab);
        assert!(operations::run_ints(
            &determinize_simple_from(&rev, &init),
            &['b' as i32, 'a' as i32]
        ));
    }

    #[test]
    #[should_panic(expected = "must be deterministic")]
    fn an_nfa_is_not_a_clean_dfa() {
        let nfa = operations::union(&[&automata::make_string("ab"), &automata::make_string("ac")]);
        assert_clean_dfa(&nfa);
    }

    #[test]
    #[should_panic(expected = "detached states")]
    fn detached_states_are_reported() {
        let mut a = Automaton::new();
        let s0 = a.create_state();
        a.create_state();
        a.set_accept(s0, true);
        a.finish_state();
        assert_no_detached_states(&a);
    }

    #[test]
    #[should_panic(expected = "has unreachable dead states")]
    fn ghost_states_are_reported() {
        let mut a = Automaton::new();
        let s0 = a.create_state();
        a.create_state();
        a.set_accept(s0, true);
        a.finish_state();
        assert_clean_nfa(&a);
    }
}
