//! `NFARunAutomaton`: run a non-deterministic automaton by determinizing it
//! lazily, one DFA state and one character class at a time.
//!
//! Lucene mutates its DFA cache from `step`, `getNumTransitions` and
//! `getTransition`; here that cache sits in a [`RefCell`] so the type can
//! implement the `&self` [`ByteRunnable`] and [`TransitionAccessor`]
//! interfaces like the other automata.

use std::cell::RefCell;
use std::collections::HashMap;

use super::automaton::{Automaton, Transition, TransitionAccessor};
use super::operations::PointTransitionSet;
use super::run_automaton::ByteRunnable;
use super::state_set::StateSet;
use super::MAX_CODE_POINT;

const MISSING: i32 = -1;
const NOT_COMPUTED: i32 = -2;

struct DState {
    nfa_states: Vec<i32>,
    transitions: Option<Vec<i32>>,
    is_accept: bool,
    computed_transitions: usize,
    outgoing_transitions: i32,
}

struct Cache {
    dstates: Vec<DState>,
    ord: HashMap<Vec<i32>, i32>,
}

/// `NFARunAutomaton`.
pub struct NfaRunAutomaton {
    automaton: Automaton,
    points: Vec<i32>,
    alphabet_size: i32,
    classmap: Vec<i32>,
    cache: RefCell<Cache>,
}

impl NfaRunAutomaton {
    /// `new NFARunAutomaton(automaton)`: over code points.
    pub fn new(automaton: Automaton) -> Self {
        Self::with_alphabet(automaton, MAX_CODE_POINT + 1)
    }

    /// `new NFARunAutomaton(automaton, alphabetSize)`.
    pub fn with_alphabet(mut automaton: Automaton, alphabet_size: i32) -> Self {
        if automaton.get_num_states() == 0 {
            // Java reads state 0 of an empty automaton as a transition-less,
            // rejecting state; give it one so the reads are in bounds.
            automaton.create_state();
        }
        let points = automaton.get_start_points();
        let cm_len = alphabet_size.clamp(0, 256) as usize;
        let mut classmap = vec![0i32; cm_len];
        let mut i = 0usize;
        for (j, slot) in classmap.iter_mut().enumerate() {
            if i + 1 < points.len() && j as i32 == points[i + 1] {
                i += 1;
            }
            *slot = i as i32;
        }
        let r = NfaRunAutomaton {
            automaton,
            points,
            alphabet_size,
            classmap,
            cache: RefCell::new(Cache {
                dstates: Vec::new(),
                ord: HashMap::new(),
            }),
        };
        {
            let mut c = r.cache.borrow_mut();
            r.find_dstate(&mut c, Some(vec![0]));
        }
        r
    }

    /// `run(int[])`: whether the label sequence is accepted.
    pub fn run(&self, s: &[i32]) -> bool {
        let mut p = 0;
        for &c in s {
            p = ByteRunnable::step(self, p, c);
            if p == MISSING {
                return false;
            }
        }
        self.cache.borrow().dstates[p as usize].is_accept
    }

    fn char_class(&self, c: i32) -> usize {
        if let Some(&cls) = usize::try_from(c).ok().and_then(|u| self.classmap.get(u)) {
            return cls as usize;
        }
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

    fn new_dstate(&self, nfa_states: Vec<i32>) -> DState {
        let is_accept = nfa_states.iter().any(|&s| self.automaton.is_accept(s));
        DState {
            nfa_states,
            transitions: None,
            is_accept,
            computed_transitions: 0,
            outgoing_transitions: 0,
        }
    }

    // SENTINEL: -1 (MISSING) when `set` is None -- the empty NFA state set.
    fn find_dstate(&self, cache: &mut Cache, set: Option<Vec<i32>>) -> i32 {
        let Some(set) = set else {
            return MISSING;
        };
        if let Some(&o) = cache.ord.get(&set) {
            return o;
        }
        let o = cache.dstates.len() as i32;
        cache.ord.insert(set.clone(), o);
        let d = self.new_dstate(set);
        cache.dstates.push(d);
        o
    }

    fn init_transitions(&self, d: &mut DState) {
        if d.transitions.is_none() {
            d.transitions = Some(vec![NOT_COMPUTED; self.points.len()]);
        }
    }

    fn assign(d: &mut DState, cls: usize, dest: i32) {
        let tr = d.transitions.as_mut().expect("initialized");
        if tr[cls] == NOT_COMPUTED {
            d.computed_transitions += 1;
            tr[cls] = dest;
            if dest != MISSING {
                d.outgoing_transitions += 1;
            }
        }
    }

    /// Java's `DState.step(c)`: the NFA successor set on `c` and the widest
    /// label range around `c` on which it is the same.
    fn nfa_step(&self, nfa_states: &[i32], c: i32) -> Option<(Vec<i32>, i32, i32)> {
        let mut set = StateSet::new(5);
        let (mut left, mut right) = (-1i32, self.alphabet_size);
        let mut t = Transition::new();
        for &s in nfa_states {
            let n = self.automaton.init_transition(s, &mut t);
            for _ in 0..n {
                self.automaton.get_next_transition(&mut t);
                if t.min <= c && t.max >= c {
                    set.incr(t.dest);
                    left = left.max(t.min);
                    right = right.min(t.max);
                }
                if t.max < c {
                    left = left.max(t.max + 1);
                }
                if t.min > c {
                    right = right.min(t.min - 1);
                    break;
                }
            }
        }
        if set.is_empty() {
            return None;
        }
        Some((set.members().to_vec(), left, right))
    }

    fn next_state(&self, state: i32, cls: usize) -> i32 {
        let mut cache = self.cache.borrow_mut();
        let si = state as usize;
        self.init_transitions(&mut cache.dstates[si]);
        let cur = cache.dstates[si].transitions.as_ref().expect("initialized")[cls];
        if cur != NOT_COMPUTED {
            return cur;
        }
        let stepped = self.nfa_step(&cache.dstates[si].nfa_states, self.points[cls]);
        let range = stepped.as_ref().map(|&(_, l, r)| (l, r));
        let dest = self.find_dstate(&mut cache, stepped.map(|(s, _, _)| s));
        let d = &mut cache.dstates[si];
        Self::assign(d, cls, dest);
        if let Some((lo, hi)) = range {
            let mut c = cls;
            while c > 0 && self.points[c - 1] >= lo {
                c -= 1;
                Self::assign(d, c, dest);
            }
            c = cls;
            while c + 1 < self.points.len() && self.points[c + 1] <= hi {
                c += 1;
                Self::assign(d, c, dest);
            }
        }
        dest
    }

    /// Java's `DState.determinize()`: compute every outgoing transition.
    fn determinize_state(&self, state: i32) {
        let si = state as usize;
        {
            let cache = self.cache.borrow();
            let d = &cache.dstates[si];
            if let Some(tr) = &d.transitions {
                if d.computed_transitions == tr.len() {
                    return;
                }
            }
        }
        let mut cache = self.cache.borrow_mut();
        self.init_transitions(&mut cache.dstates[si]);
        let mut set = PointTransitionSet::default();
        let mut t = Transition::new();
        for &s in &cache.dstates[si].nfa_states {
            let n = self.automaton.init_transition(s, &mut t);
            for _ in 0..n {
                self.automaton.get_next_transition(&mut t);
                set.add(&t);
            }
        }
        let np = self.points.len();
        if set.is_empty() {
            let d = &mut cache.dstates[si];
            d.transitions = Some(vec![MISSING; np]);
            d.computed_transitions = np;
            return;
        }
        let pts = set.take_sorted();
        let mut states = StateSet::new(5);
        let mut last_point = -1;
        let mut char_class = 0usize;
        for pt in &pts {
            let point = pt.point;
            if !states.is_empty() {
                let members = states.members().to_vec();
                let ord = self.find_dstate(&mut cache, Some(members));
                let d = &mut cache.dstates[si];
                while self.points[char_class] < last_point {
                    Self::assign(d, char_class, MISSING);
                    char_class += 1;
                }
                while char_class < np && self.points[char_class] < point {
                    Self::assign(d, char_class, ord);
                    char_class += 1;
                }
            }
            for &dest in &pt.ends {
                states.decr(dest);
            }
            for &dest in &pt.starts {
                states.incr(dest);
            }
            last_point = point;
        }
        let d = &mut cache.dstates[si];
        let tr = d.transitions.as_mut().expect("initialized");
        for slot in tr.iter_mut().skip(char_class) {
            // Java's Arrays.fill(transitions, charClass, len, MISSING) also
            // overwrites any NOT_COMPUTED tail without counting it.
            *slot = MISSING;
        }
        d.computed_transitions = np;
    }

    fn set_transition_accordingly(&self, t: &mut Transition) {
        let cache = self.cache.borrow();
        let tr = cache.dstates[t.source as usize]
            .transitions
            .as_ref()
            .expect("determinized");
        let upto = t.transition_upto as usize;
        t.dest = tr[upto];
        t.min = self.points[upto];
        t.max = if upto == self.points.len() - 1 {
            self.alphabet_size - 1
        } else {
            self.points[upto + 1] - 1
        };
    }
}

impl ByteRunnable for NfaRunAutomaton {
    fn step(&self, state: i32, c: i32) -> i32 {
        self.next_state(state, self.char_class(c))
    }

    fn is_accept(&self, state: i32) -> bool {
        self.cache.borrow().dstates[state as usize].is_accept
    }

    /// Java returns its (over-allocated) cache array length; the number of
    /// DFA states discovered so far is returned here.
    fn get_size(&self) -> i32 {
        self.cache.borrow().dstates.len() as i32
    }
}

impl TransitionAccessor for NfaRunAutomaton {
    fn init_transition(&self, state: i32, t: &mut Transition) -> i32 {
        t.source = state;
        t.transition_upto = -1;
        self.get_num_transitions(state)
    }

    fn get_next_transition(&self, t: &mut Transition) {
        {
            let cache = self.cache.borrow();
            let tr = cache.dstates[t.source as usize]
                .transitions
                .as_ref()
                .expect("determinized");
            loop {
                t.transition_upto += 1;
                if tr[t.transition_upto as usize] != MISSING {
                    break;
                }
            }
        }
        self.set_transition_accordingly(t);
    }

    fn get_num_transitions(&self, state: i32) -> i32 {
        self.determinize_state(state);
        self.cache.borrow().dstates[state as usize].outgoing_transitions
    }

    fn get_transition(&self, state: i32, index: i32, t: &mut Transition) {
        self.determinize_state(state);
        {
            let cache = self.cache.borrow();
            let tr = cache.dstates[state as usize]
                .transitions
                .as_ref()
                .expect("determinized");
            let mut outgoing = -1;
            t.transition_upto = -1;
            t.source = state;
            while outgoing < index && t.transition_upto + 1 < tr.len() as i32 {
                t.transition_upto += 1;
                if tr[t.transition_upto as usize] != MISSING {
                    outgoing += 1;
                }
            }
        }
        self.set_transition_accordingly(t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::RegExp;

    #[test]
    fn lazily_determinizes() {
        let a = RegExp::new("(a|ab)*c.").unwrap().to_automaton().unwrap();
        let n = NfaRunAutomaton::new(a.clone());
        let d = crate::automaton::operations::determinize(&a, 10_000).unwrap();
        for s in ["c!", "ababc\u{10000}", "abc", "aab", "", "ac"] {
            let cps: Vec<i32> = s.chars().map(|c| c as i32).collect();
            assert_eq!(n.run(&cps), crate::automaton::operations::run(&d, s), "{s}");
        }
        assert!(n.get_size() >= 1);
        let mut t = Transition::new();
        let count = n.init_transition(0, &mut t);
        assert!(count > 0);
        let mut labels = Vec::new();
        for _ in 0..count {
            n.get_next_transition(&mut t);
            labels.push((t.min, t.max));
        }
        let mut t2 = Transition::new();
        n.get_transition(0, count - 1, &mut t2);
        assert_eq!((t2.min, t2.max), *labels.last().unwrap());
        assert!(!ByteRunnable::is_accept(&n, 0));
        assert!(!n.run(&[0x7FFF_FFFF]));
        let empty = NfaRunAutomaton::with_alphabet(Automaton::new(), 256);
        assert_eq!(empty.get_num_transitions(0), 0);
    }
}
