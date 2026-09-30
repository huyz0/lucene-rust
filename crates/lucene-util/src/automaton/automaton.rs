//! `Automaton`, `Automaton.Builder`, `Transition`, `TransitionAccessor` and
//! `StatePair`.
//!
//! The representation is Lucene's: `states` holds two ints per state (the
//! offset of its first transition in `transitions`, or `-1` before it has
//! any, then its transition count) and `transitions` three per transition
//! (`dest, min, max`). Transitions must be added one source state at a time;
//! [`Automaton::finish_state`] then sorts and merges the current state's
//! transitions, which is what makes two automata built the same way compare
//! state for state and transition for transition.

use super::MAX_CODE_POINT;
use super::MIN_CODE_POINT;

/// `Transition`: one `min..=max` labelled edge `source -> dest`, plus the
/// cursor [`TransitionAccessor::get_next_transition`] advances.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Transition {
    /// Source state.
    pub source: i32,
    /// Destination state.
    pub dest: i32,
    /// Minimum accepted label (inclusive).
    pub min: i32,
    /// Maximum accepted label (inclusive).
    pub max: i32,
    /// Java's package-private `transitionUpto` cursor.
    pub(crate) transition_upto: i32,
}

impl Default for Transition {
    fn default() -> Self {
        Transition {
            source: 0,
            dest: 0,
            min: 0,
            max: 0,
            transition_upto: -1,
        }
    }
}

impl Transition {
    /// `new Transition()`.
    pub fn new() -> Self {
        Self::default()
    }
}

/// `TransitionAccessor`: iterate the transitions of a state.
pub trait TransitionAccessor {
    /// Initialize `t` for iterating `state`'s transitions; returns their count.
    fn init_transition(&self, state: i32, t: &mut Transition) -> i32;
    /// Fill `t` with the next transition of the state `init_transition` set up.
    fn get_next_transition(&self, t: &mut Transition);
    /// How many transitions `state` has.
    fn get_num_transitions(&self, state: i32) -> i32;
    /// Fill `t` with `state`'s `index`-th transition.
    fn get_transition(&self, state: i32, index: i32, t: &mut Transition);
}

/// `StatePair`: a pair of states of two automata in a product construction.
/// Equality and hashing are over `(s1, s2)` only, as in Java.
#[derive(Clone, Copy, Debug)]
pub struct StatePair {
    /// The product state, `-1` until assigned.
    pub s: i32,
    /// State of the first automaton.
    pub s1: i32,
    /// State of the second automaton.
    pub s2: i32,
}

impl StatePair {
    /// `new StatePair(s1, s2)`.
    pub fn new(s1: i32, s2: i32) -> Self {
        StatePair { s: -1, s1, s2 }
    }
}

impl PartialEq for StatePair {
    fn eq(&self, other: &Self) -> bool {
        self.s1 == other.s1 && self.s2 == other.s2
    }
}
impl Eq for StatePair {}
impl std::hash::Hash for StatePair {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.s1.wrapping_mul(31).wrapping_add(self.s2)).hash(state);
    }
}

/// `Automaton`: a (possibly non-deterministic) finite automaton over `i32`
/// labels, state 0 initial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Automaton {
    /// Two ints per state: first-transition offset (`-1` = none yet), count.
    states: Vec<i32>,
    /// Accept flag per state (Java's `BitSet`).
    is_accept: Vec<bool>,
    /// Three ints per transition: dest, min, max.
    transitions: Vec<i32>,
    /// The state transitions are currently being added to, or `-1`.
    cur_state: i32,
    /// False once any state is found to have overlapping transitions.
    deterministic: bool,
}

impl Default for Automaton {
    fn default() -> Self {
        Self::new()
    }
}

fn idx(v: i32) -> usize {
    usize::try_from(v).expect("negative state or index")
}

impl Automaton {
    /// `new Automaton()`.
    pub fn new() -> Self {
        Automaton {
            states: Vec::new(),
            is_accept: Vec::new(),
            transitions: Vec::new(),
            cur_state: -1,
            deterministic: true,
        }
    }

    /// `new Automaton(numStates, numTransitions)`: capacity hints only.
    pub fn with_capacity(num_states: usize, num_transitions: usize) -> Self {
        let mut a = Self::new();
        a.states.reserve(num_states * 2);
        a.is_accept.reserve(num_states);
        a.transitions.reserve(num_transitions * 3);
        a
    }

    /// `createState()`: a new state, returning its number.
    pub fn create_state(&mut self) -> i32 {
        let state = self.get_num_states();
        self.states.push(-1);
        self.states.push(0);
        self.is_accept.push(false);
        state
    }

    /// `setAccept(state, accept)`.
    ///
    /// # Panics
    /// If `state` does not exist (Java's `Objects.checkIndex`).
    pub fn set_accept(&mut self, state: i32, accept: bool) {
        assert!(
            state >= 0 && state < self.get_num_states(),
            "state {state} out of bounds"
        );
        self.is_accept[idx(state)] = accept;
    }

    /// `getSortedTransitions()`: every state's transitions, in order.
    pub fn get_sorted_transitions(&self) -> Vec<Vec<Transition>> {
        let n = self.get_num_states();
        let mut out = Vec::with_capacity(idx(n));
        for s in 0..n {
            let count = self.get_num_transitions(s);
            let mut row = Vec::with_capacity(idx(count));
            for t in 0..count {
                let mut tr = Transition::new();
                self.get_transition(s, t, &mut tr);
                row.push(tr);
            }
            out.push(row);
        }
        out
    }

    /// `getAcceptStates()`: the accept flag of every state.
    pub fn get_accept_states(&self) -> &[bool] {
        &self.is_accept
    }

    /// Number of accept states (`getAcceptStates().cardinality()`).
    pub fn accept_count(&self) -> usize {
        self.is_accept.iter().filter(|&&b| b).count()
    }

    /// `isAccept(state)`; like Java's `BitSet.get`, false past the last state.
    pub fn is_accept(&self, state: i32) -> bool {
        usize::try_from(state)
            .ok()
            .and_then(|s| self.is_accept.get(s))
            .copied()
            .unwrap_or(false)
    }

    /// `addTransition(source, dest, label)`.
    pub fn add_transition_label(&mut self, source: i32, dest: i32, label: i32) {
        self.add_transition(source, dest, label, label);
    }

    /// `addTransition(source, dest, min, max)`.
    ///
    /// # Panics
    /// If either state does not exist, or `source` already had its
    /// transitions finished (Java's `IllegalStateException`: transitions must
    /// be added one source state at a time).
    pub fn add_transition(&mut self, source: i32, dest: i32, min: i32, max: i32) {
        let bounds = self.get_num_states();
        assert!(
            source >= 0 && source < bounds,
            "source {source} out of bounds"
        );
        assert!(dest >= 0 && dest < bounds, "dest {dest} out of bounds");
        if self.cur_state != source {
            if self.cur_state != -1 {
                self.finish_current_state();
            }
            self.cur_state = source;
            let s = idx(source);
            if self.states[2 * s] != -1 {
                panic!("from state ({source}) already had transitions added");
            }
            self.states[2 * s] = self.transitions.len() as i32;
        }
        self.transitions.push(dest);
        self.transitions.push(min);
        self.transitions.push(max);
        self.states[2 * idx(self.cur_state) + 1] += 1;
    }

    /// `addEpsilon(source, dest)`: copy `dest`'s transitions (and accept
    /// flag) onto `source`.
    pub fn add_epsilon(&mut self, source: i32, dest: i32) {
        let mut t = Transition::new();
        let count = self.init_transition(dest, &mut t);
        for _ in 0..count {
            self.get_next_transition(&mut t);
            self.add_transition(source, t.dest, t.min, t.max);
        }
        if self.is_accept(dest) {
            self.set_accept(source, true);
        }
    }

    /// `copy(other)`: append all of `other`'s states and transitions,
    /// renumbered after this automaton's.
    pub fn copy(&mut self, other: &Automaton) {
        let state_offset = self.get_num_states();
        let trans_offset = self.transitions.len() as i32;
        for pair in other.states.chunks_exact(2) {
            let first = if pair[0] != -1 {
                pair[0] + trans_offset
            } else {
                -1
            };
            self.states.push(first);
            self.states.push(pair[1]);
        }
        self.is_accept.extend(
            other
                .is_accept
                .iter()
                .copied()
                .take(idx(other.get_num_states())),
        );
        for tr in other.transitions.chunks_exact(3) {
            self.transitions.push(tr[0] + state_offset);
            self.transitions.push(tr[1]);
            self.transitions.push(tr[2]);
        }
        if !other.deterministic {
            self.deterministic = false;
        }
    }

    /// Java's `finishCurrentState`: sort the current state's transitions by
    /// `(dest, min, max)`, merge touching/overlapping ranges to the same
    /// destination, re-sort by `(min, max, dest)`, and clear `deterministic`
    /// if any two now overlap.
    fn finish_current_state(&mut self) {
        let cur = idx(self.cur_state);
        let num = idx(self.states[2 * cur + 1]);
        let offset = idx(self.states[2 * cur]);
        let mut ts: Vec<(i32, i32, i32)> = self.transitions[offset..offset + 3 * num]
            .chunks_exact(3)
            .map(|c| (c[0], c[1], c[2]))
            .collect();
        ts.sort_unstable();
        let mut merged: Vec<(i32, i32, i32)> = Vec::with_capacity(num);
        let (mut dest, mut min, mut max) = (-1i32, -1i32, -1i32);
        for &(t_dest, t_min, t_max) in &ts {
            if dest == t_dest {
                if t_min <= max.saturating_add(1) {
                    if t_max > max {
                        max = t_max;
                    }
                } else {
                    if dest != -1 {
                        merged.push((dest, min, max));
                    }
                    min = t_min;
                    max = t_max;
                }
            } else {
                if dest != -1 {
                    merged.push((dest, min, max));
                }
                dest = t_dest;
                min = t_min;
                max = t_max;
            }
        }
        if dest != -1 {
            merged.push((dest, min, max));
        }
        merged.sort_unstable_by_key(|&(d, lo, hi)| (lo, hi, d));
        let upto = merged.len();
        self.transitions.truncate(offset);
        for &(d, lo, hi) in &merged {
            self.transitions.push(d);
            self.transitions.push(lo);
            self.transitions.push(hi);
        }
        self.states[2 * cur + 1] = upto as i32;
        if self.deterministic && upto > 1 {
            let mut last_max = merged[0].2;
            for &(_, lo, hi) in &merged[1..] {
                if lo <= last_max {
                    self.deterministic = false;
                    break;
                }
                last_max = hi;
            }
        }
    }

    /// `isDeterministic()`: no state has overlapping transitions. Like
    /// Java, this can be `false` for an automaton that is in fact
    /// deterministic in language terms but was built from overlapping edges.
    pub fn is_deterministic(&self) -> bool {
        self.deterministic
    }

    /// `finishState()`: finish the state transitions were being added to.
    pub fn finish_state(&mut self) {
        if self.cur_state != -1 {
            self.finish_current_state();
            self.cur_state = -1;
        }
    }

    /// `getNumStates()`.
    pub fn get_num_states(&self) -> i32 {
        (self.states.len() / 2) as i32
    }

    /// `getNumTransitions()`: across all states.
    pub fn get_total_num_transitions(&self) -> i32 {
        (self.transitions.len() / 3) as i32
    }

    /// `getStartPoints()`: every label at which some transition starts or
    /// ends+1, plus 0, sorted and deduplicated.
    pub fn get_start_points(&self) -> Vec<i32> {
        let mut points = vec![MIN_CODE_POINT];
        for pair in self.states.chunks_exact(2) {
            if pair[0] < 0 {
                continue;
            }
            let start = idx(pair[0]);
            for tr in self.transitions[start..start + 3 * idx(pair[1])].chunks_exact(3) {
                points.push(tr[1]);
                if tr[2] < MAX_CODE_POINT {
                    points.push(tr[2] + 1);
                }
            }
        }
        points.sort_unstable();
        points.dedup();
        points
    }

    /// `step(state, label)`: the destination on `label`, by binary search.
    /// Only meaningful on a deterministic automaton.
    // SENTINEL: -1 means no transition accepts `label`.
    pub fn step(&self, state: i32, label: i32) -> i32 {
        self.next_impl(state, 0, label, None)
    }

    /// `next(transition, label)`: like [`Automaton::step`] from
    /// `transition.source`, searching from `transition`'s cursor and leaving
    /// the found transition (or the insertion point) in it.
    // SENTINEL: -1 means no transition accepts `label`.
    pub fn next(&self, transition: &mut Transition, label: i32) -> i32 {
        let (source, upto) = (transition.source, transition.transition_upto);
        self.next_impl(source, upto, label, Some(transition))
    }

    fn next_impl(
        &self,
        state: i32,
        from_index: i32,
        label: i32,
        transition: Option<&mut Transition>,
    ) -> i32 {
        let si = 2 * idx(state);
        // A state past the end reads as transition-less, as state 0 of an
        // empty Java automaton does (its arrays are pre-sized, zero-filled).
        let (first, num) = match self.states.get(si..si + 2) {
            Some(pair) => (pair[0], pair[1]),
            None => (-1, 0),
        };
        let mut low = from_index.max(0);
        let mut high = num - 1;
        while low <= high {
            let mid = (low + high) >> 1;
            let ti = idx(first + 3 * mid);
            let min_label = self.transitions[ti + 1];
            if min_label > label {
                high = mid - 1;
            } else {
                let max_label = self.transitions[ti + 2];
                if max_label < label {
                    low = mid + 1;
                } else {
                    let dest = self.transitions[ti];
                    if let Some(t) = transition {
                        t.dest = dest;
                        t.min = min_label;
                        t.max = max_label;
                        t.transition_upto = mid;
                    }
                    return dest;
                }
            }
        }
        if let Some(t) = transition {
            t.dest = -1;
            t.transition_upto = low;
        }
        -1
    }

    /// `toDot()`: a Graphviz rendering, byte for byte as Lucene writes it.
    pub fn to_dot(&self) -> String {
        let mut b = String::new();
        b.push_str("digraph Automaton {\n");
        b.push_str("  rankdir = LR\n");
        b.push_str("  node [width=0.2, height=0.2, fontsize=8]\n");
        let n = self.get_num_states();
        if n > 0 {
            b.push_str("  initial [shape=plaintext,label=\"\"]\n");
            b.push_str("  initial -> 0\n");
        }
        let mut t = Transition::new();
        for state in 0..n {
            if self.is_accept(state) {
                b.push_str(&format!(
                    "  {state} [shape=doublecircle,label=\"{state}\"]\n"
                ));
            } else {
                b.push_str(&format!("  {state} [shape=circle,label=\"{state}\"]\n"));
            }
            let count = self.init_transition(state, &mut t);
            for _ in 0..count {
                self.get_next_transition(&mut t);
                b.push_str(&format!("  {state} -> {} [label=\"", t.dest));
                append_char_string(t.min, &mut b);
                if t.max != t.min {
                    b.push('-');
                    append_char_string(t.max, &mut b);
                }
                b.push_str("\"]\n");
            }
        }
        b.push('}');
        b
    }
}

/// Java's `Automaton.appendCharString`: printable ASCII as itself, anything
/// else as `\\U` plus eight hex digits.
pub(crate) fn append_char_string(c: i32, b: &mut String) {
    if (0x21..=0x7e).contains(&c) && c != '\\' as i32 && c != '"' as i32 {
        b.push(char::from_u32(c as u32).unwrap_or('?'));
    } else {
        b.push_str(&format!("\\\\U{:08x}", c as u32));
    }
}

impl TransitionAccessor for Automaton {
    fn init_transition(&self, state: i32, t: &mut Transition) -> i32 {
        t.source = state;
        t.transition_upto = self.states.get(2 * idx(state)).copied().unwrap_or(-1);
        self.get_num_transitions(state)
    }

    fn get_next_transition(&self, t: &mut Transition) {
        let mut i = idx(t.transition_upto);
        t.dest = self.transitions[i];
        t.min = self.transitions[i + 1];
        t.max = self.transitions[i + 2];
        i += 3;
        t.transition_upto = i as i32;
    }

    fn get_num_transitions(&self, state: i32) -> i32 {
        // Past the last state: none (see `next_impl`).
        let count = self.states.get(2 * idx(state) + 1).copied().unwrap_or(0);
        if count == -1 {
            0
        } else {
            count
        }
    }

    fn get_transition(&self, state: i32, index: i32, t: &mut Transition) {
        let i = idx(self.states[2 * idx(state)] + 3 * index);
        t.source = state;
        t.dest = self.transitions[i];
        t.min = self.transitions[i + 1];
        t.max = self.transitions[i + 2];
    }
}

/// `Automaton.Builder`: collect transitions in any order, then
/// [`Builder::finish`] sorts them by `(source, min, max, dest)` and builds the
/// [`Automaton`].
#[derive(Clone, Debug, Default)]
pub struct Builder {
    is_accept: Vec<bool>,
    transitions: Vec<[i32; 4]>,
}

impl Builder {
    /// `new Automaton.Builder()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// `addTransition(source, dest, label)`.
    pub fn add_transition_label(&mut self, source: i32, dest: i32, label: i32) {
        self.add_transition(source, dest, label, label);
    }

    /// `addTransition(source, dest, min, max)`.
    pub fn add_transition(&mut self, source: i32, dest: i32, min: i32, max: i32) {
        self.transitions.push([source, dest, min, max]);
    }

    /// `addEpsilon(source, dest)`: copy the transitions *already added* from
    /// `dest` onto `source`, and `dest`'s accept flag.
    pub fn add_epsilon(&mut self, source: i32, dest: i32) {
        let n = self.transitions.len();
        for upto in 0..n {
            let [s, d, lo, hi] = self.transitions[upto];
            if s == dest {
                self.add_transition(source, d, lo, hi);
            }
        }
        if self.is_accept(dest) {
            self.set_accept(source, true);
        }
    }

    /// `finish()`: the built automaton.
    pub fn finish(&mut self) -> Automaton {
        let num_states = self.get_num_states();
        let mut a = Automaton::with_capacity(idx(num_states), self.transitions.len());
        for state in 0..num_states {
            a.create_state();
            a.set_accept(state, self.is_accept(state));
        }
        self.transitions
            .sort_unstable_by_key(|&[s, d, lo, hi]| (s, lo, hi, d));
        for &[s, d, lo, hi] in &self.transitions {
            a.add_transition(s, d, lo, hi);
        }
        a.finish_state();
        a
    }

    /// `createState()`.
    pub fn create_state(&mut self) -> i32 {
        self.is_accept.push(false);
        (self.is_accept.len() - 1) as i32
    }

    /// `setAccept(state, accept)`.
    ///
    /// # Panics
    /// If `state` does not exist.
    pub fn set_accept(&mut self, state: i32, accept: bool) {
        assert!(
            state >= 0 && state < self.get_num_states(),
            "state {state} out of bounds"
        );
        self.is_accept[idx(state)] = accept;
    }

    /// `isAccept(state)`.
    pub fn is_accept(&self, state: i32) -> bool {
        usize::try_from(state)
            .ok()
            .and_then(|s| self.is_accept.get(s))
            .copied()
            .unwrap_or(false)
    }

    /// `getNumStates()`.
    pub fn get_num_states(&self) -> i32 {
        self.is_accept.len() as i32
    }

    /// `copy(other)`: append `other`'s states and transitions.
    pub fn copy(&mut self, other: &Automaton) {
        let offset = self.get_num_states();
        let n = other.get_num_states();
        self.copy_states(other);
        let mut t = Transition::new();
        for s in 0..n {
            let count = other.init_transition(s, &mut t);
            for _ in 0..count {
                other.get_next_transition(&mut t);
                self.add_transition(offset + s, offset + t.dest, t.min, t.max);
            }
        }
    }

    /// `copyStates(other)`: append `other`'s states (accept flags only).
    pub fn copy_states(&mut self, other: &Automaton) {
        for s in 0..other.get_num_states() {
            let ns = self.create_state();
            self.set_accept(ns, other.is_accept(s));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_state_merges_and_sorts() {
        let mut a = Automaton::new();
        let s0 = a.create_state();
        let s1 = a.create_state();
        let s2 = a.create_state();
        a.add_transition(s0, s1, 5, 7);
        a.add_transition(s0, s1, 'a' as i32, 'c' as i32);
        a.add_transition(s0, s1, 'd' as i32, 'f' as i32);
        a.add_transition(s0, s2, 1, 2);
        a.finish_state();
        assert_eq!(a.get_num_transitions(s0), 3);
        let ts = a.get_sorted_transitions();
        assert_eq!((ts[0][0].min, ts[0][0].max, ts[0][0].dest), (1, 2, 2));
        assert_eq!((ts[0][2].min, ts[0][2].max), ('a' as i32, 'f' as i32));
        assert!(a.is_deterministic());
        assert_eq!(a.get_start_points(), vec![0, 1, 3, 5, 8, 97, 103]);
        assert_eq!(a.step(0, 'b' as i32), 1);
        assert_eq!(a.step(0, 4), -1);
        let mut t = Transition::new();
        t.source = 0;
        assert_eq!(a.next(&mut t, 100), 1);
        assert_eq!(t.transition_upto, 2);
        assert_eq!(a.next(&mut t, 200), -1);
        assert_eq!(t.transition_upto, 3);
    }

    #[test]
    fn overlapping_is_nondeterministic_and_dot_renders() {
        let mut a = Automaton::new();
        a.create_state();
        a.create_state();
        a.set_accept(1, true);
        a.add_transition(0, 0, 0, 10);
        a.add_transition(0, 1, 5, 5);
        a.finish_state();
        assert!(!a.is_deterministic());
        let dot = a.to_dot();
        assert!(dot.contains("1 [shape=doublecircle"));
        assert!(dot.contains("\\\\U00000000-\\\\U0000000a"));
        assert!(!a.is_accept(-1));
        assert!(!a.is_accept(9));
        assert_eq!(a.accept_count(), 1);
    }

    #[test]
    #[should_panic(expected = "already had transitions added")]
    fn re_adding_to_finished_state_panics() {
        let mut a = Automaton::new();
        a.create_state();
        a.create_state();
        a.add_transition(0, 1, 1, 1);
        a.add_transition(1, 0, 1, 1);
        a.add_transition(0, 1, 2, 2);
    }

    #[test]
    fn builder_epsilon_and_copy() {
        let mut b = Builder::new();
        let s0 = b.create_state();
        let s1 = b.create_state();
        let s2 = b.create_state();
        b.set_accept(s2, true);
        b.add_transition_label(s1, s2, 'x' as i32);
        b.add_epsilon(s0, s1);
        let a = b.finish();
        assert!(a.is_accept(2));
        assert_eq!(a.step(0, 'x' as i32), 2);
        let mut c = Automaton::new();
        c.copy(&a);
        c.copy(&a);
        assert_eq!(c.get_num_states(), 6);
        assert_eq!(c.step(3, 'x' as i32), 5);
        let mut b2 = Builder::new();
        b2.copy(&a);
        assert_eq!(b2.finish(), a);
        let p = StatePair::new(1, 2);
        assert_eq!(p, StatePair { s: 7, s1: 1, s2: 2 });
    }
}
