//! `Operations`: the automaton algebra, plus the test-framework helpers
//! (`AutomatonTestUtil`) Lucene 10 moved out of core.
//!
//! Every function follows its Java counterpart step for step, so the
//! automata it returns have Lucene's state numbering and transition layout,
//! not merely the same language.

use std::collections::{HashMap, HashSet, VecDeque};

use super::automata;
use super::automaton::{Automaton, Builder, StatePair, Transition, TransitionAccessor};
use super::error::{AutomatonError, TooComplexToDeterminize};
use super::state_set::{FrozenIntSet, StateSet};
use super::{MAX_CODE_POINT, MIN_CODE_POINT};

/// `Operations.DEFAULT_DETERMINIZE_WORK_LIMIT`.
pub const DEFAULT_DETERMINIZE_WORK_LIMIT: i32 = 10_000;

/// `AutomatonTestUtil.MAX_RECURSION_LEVEL`, [`is_finite`]'s depth bound.
pub const MAX_RECURSION_LEVEL: i32 = 1000;

fn idx(v: i32) -> usize {
    usize::try_from(v).expect("negative state")
}

/// `concatenate(a1, a2)`.
pub fn concatenate2(a1: &Automaton, a2: &Automaton) -> Automaton {
    concatenate(&[a1, a2])
}

/// `concatenate(list)`: the language of each automaton in turn.
pub fn concatenate(list: &[&Automaton]) -> Automaton {
    let mut result = Automaton::new();
    for a in list {
        if a.get_num_states() == 0 {
            return automata::make_empty();
        }
        for _ in 0..a.get_num_states() {
            result.create_state();
        }
    }
    let mut state_offset = 0;
    let mut t = Transition::new();
    for (i, a) in list.iter().enumerate() {
        let num_states = a.get_num_states();
        let next_a = list.get(i + 1).copied();
        for s in 0..num_states {
            let mut num_transitions = a.init_transition(s, &mut t);
            for _ in 0..num_transitions {
                a.get_next_transition(&mut t);
                result.add_transition(state_offset + s, state_offset + t.dest, t.min, t.max);
            }
            if a.is_accept(s) {
                let mut follow_a = next_a;
                let mut follow_offset = state_offset;
                let mut upto = i + 1;
                loop {
                    match follow_a {
                        Some(f) => {
                            num_transitions = f.init_transition(0, &mut t);
                            for _ in 0..num_transitions {
                                f.get_next_transition(&mut t);
                                result.add_transition(
                                    state_offset + s,
                                    follow_offset + num_states + t.dest,
                                    t.min,
                                    t.max,
                                );
                            }
                            if f.is_accept(0) {
                                follow_offset += f.get_num_states();
                                follow_a = list.get(upto + 1).copied();
                                upto += 1;
                            } else {
                                break;
                            }
                        }
                        None => {
                            result.set_accept(state_offset + s, true);
                            break;
                        }
                    }
                }
            }
        }
        state_offset += num_states;
    }
    if result.get_num_states() == 0 {
        result.create_state();
    }
    result.finish_state();
    remove_dead_states(&result)
}

/// `optional(a)`: the language of `a` plus the empty string.
pub fn optional(a: &Automaton) -> Automaton {
    if a.is_accept(0) {
        return a.clone();
    }
    let mut has_to_initial = false;
    let mut t = Transition::new();
    'outer: for state in 0..a.get_num_states() {
        let count = a.init_transition(state, &mut t);
        for _ in 0..count {
            a.get_next_transition(&mut t);
            if t.dest == 0 {
                has_to_initial = true;
                break 'outer;
            }
        }
    }
    if !has_to_initial {
        let mut result = Automaton::new();
        result.copy(a);
        if result.get_num_states() == 0 {
            result.create_state();
        }
        result.set_accept(0, true);
        return result;
    }
    let mut result = Automaton::new();
    result.create_state();
    result.set_accept(0, true);
    if a.get_num_states() > 0 {
        result.copy(a);
        result.add_epsilon(0, 1);
    }
    result.finish_state();
    result
}

/// `repeat(a)`: Kleene star.
pub fn repeat(a: &Automaton) -> Automaton {
    if a.get_num_states() == 0 {
        return a.clone();
    }
    if a.is_accept(0) && a.accept_count() == 1 {
        return a.clone();
    }
    let mut builder = Builder::new();
    builder.create_state();
    builder.set_accept(0, true);
    let mut t = Transition::new();
    let n = a.get_num_states();
    let mut state_map = vec![0i32; idx(n)];
    for state in 0..n {
        if !a.is_accept(state) {
            state_map[idx(state)] = builder.create_state();
        } else if a.get_num_transitions(state) == 0 {
            state_map[idx(state)] = 0;
        } else {
            let ns = builder.create_state();
            state_map[idx(state)] = ns;
            builder.set_accept(ns, true);
        }
    }
    for state in 0..n {
        let src = state_map[idx(state)];
        let count = a.init_transition(state, &mut t);
        for _ in 0..count {
            a.get_next_transition(&mut t);
            builder.add_transition(src, state_map[idx(t.dest)], t.min, t.max);
        }
    }
    let count = a.init_transition(0, &mut t);
    for _ in 0..count {
        a.get_next_transition(&mut t);
        builder.add_transition(0, state_map[idx(t.dest)], t.min, t.max);
    }
    for s in 0..n {
        if a.is_accept(s) && state_map[idx(s)] != 0 {
            let count = a.init_transition(0, &mut t);
            for _ in 0..count {
                a.get_next_transition(&mut t);
                builder.add_transition(state_map[idx(s)], state_map[idx(t.dest)], t.min, t.max);
            }
        }
    }
    remove_dead_states(&builder.finish())
}

/// `repeat(a, count)`: `count` or more repetitions.
pub fn repeat_min(a: &Automaton, count: i32) -> Automaton {
    if count == 0 {
        return repeat(a);
    }
    let star = repeat(a);
    let mut list: Vec<&Automaton> = (0..count).map(|_| a).collect();
    list.push(&star);
    concatenate(&list)
}

/// `repeat(a, min, max)`: between `min` and `max` repetitions.
pub fn repeat_range(a: &Automaton, min: i32, max: i32) -> Automaton {
    if min > max {
        return automata::make_empty();
    }
    let b = if min == 0 {
        automata::make_empty_string()
    } else if min == 1 {
        let mut b = Automaton::new();
        b.copy(a);
        b
    } else {
        let list: Vec<&Automaton> = (0..min).map(|_| a).collect();
        concatenate(&list)
    };
    let mut prev_accept = accept_set(&b, 0);
    let mut builder = Builder::new();
    builder.copy(&b);
    for _ in min..max {
        let num_states = builder.get_num_states();
        builder.copy(a);
        for &s in &prev_accept {
            builder.add_epsilon(s, num_states);
        }
        prev_accept = accept_set(a, num_states);
    }
    remove_dead_states(&builder.finish())
}

fn accept_set(a: &Automaton, offset: i32) -> Vec<i32> {
    (0..a.get_num_states())
        .filter(|&s| a.is_accept(s))
        .map(|s| offset + s)
        .collect()
}

/// `complement(a, determinizeWorkLimit)`: every string `a` rejects.
pub fn complement(a: &Automaton, work_limit: i32) -> Result<Automaton, TooComplexToDeterminize> {
    let mut a = totalize(&determinize(a, work_limit)?);
    for p in 0..a.get_num_states() {
        let acc = a.is_accept(p);
        a.set_accept(p, !acc);
    }
    Ok(remove_dead_states(&a))
}

/// `minus(a1, a2, determinizeWorkLimit)`: `a1`'s language without `a2`'s.
pub fn minus(
    a1: &Automaton,
    a2: &Automaton,
    work_limit: i32,
) -> Result<Automaton, TooComplexToDeterminize> {
    if is_empty(a1) || std::ptr::eq(a1, a2) {
        return Ok(automata::make_empty());
    }
    if is_empty(a2) {
        return Ok(a1.clone());
    }
    Ok(intersection(a1, &complement(a2, work_limit)?))
}

/// `intersection(a1, a2)`: the product automaton.
pub fn intersection(a1: &Automaton, a2: &Automaton) -> Automaton {
    if std::ptr::eq(a1, a2) || a1.get_num_states() == 0 {
        return a1.clone();
    }
    if a2.get_num_states() == 0 {
        return a2.clone();
    }
    let t1s = a1.get_sorted_transitions();
    let t2s = a2.get_sorted_transitions();
    let mut c = Automaton::new();
    c.create_state();
    let mut worklist: VecDeque<StatePair> = VecDeque::new();
    let mut newstates: HashMap<(i32, i32), i32> = HashMap::new();
    worklist.push_back(StatePair { s: 0, s1: 0, s2: 0 });
    newstates.insert((0, 0), 0);
    while let Some(p) = worklist.pop_front() {
        c.set_accept(p.s, a1.is_accept(p.s1) && a2.is_accept(p.s2));
        let t1 = &t1s[idx(p.s1)];
        let t2 = &t2s[idx(p.s2)];
        let mut b2 = 0;
        for tr1 in t1 {
            while b2 < t2.len() && t2[b2].max < tr1.min {
                b2 += 1;
            }
            let mut n2 = b2;
            while n2 < t2.len() && tr1.max >= t2[n2].min {
                let tr2 = &t2[n2];
                if tr2.max >= tr1.min {
                    let key = (tr1.dest, tr2.dest);
                    let r = match newstates.get(&key) {
                        Some(&r) => r,
                        None => {
                            let s = c.create_state();
                            worklist.push_back(StatePair {
                                s,
                                s1: key.0,
                                s2: key.1,
                            });
                            newstates.insert(key, s);
                            s
                        }
                    };
                    let min = tr1.min.max(tr2.min);
                    let max = tr1.max.min(tr2.max);
                    c.add_transition(p.s, r, min, max);
                }
                n2 += 1;
            }
        }
    }
    c.finish_state();
    remove_dead_states(&c)
}

/// `hasDeadStates(a)`: some state is unreachable or cannot reach an accept
/// state.
pub fn has_dead_states(a: &Automaton) -> bool {
    let live = get_live_states(a);
    live.iter().filter(|&&b| b).count() < idx(a.get_num_states())
}

/// `hasDeadStatesFromInitial(a)`: some reachable state cannot reach an
/// accept state.
pub fn has_dead_states_from_initial(a: &Automaton) -> bool {
    let from = get_live_states_from_initial(a);
    let to = get_live_states_to_accept(a);
    from.iter().zip(&to).any(|(&f, &t)| f && !t)
}

/// `hasDeadStatesToAccept(a)`: some state that reaches an accept state is
/// unreachable from the initial state.
pub fn has_dead_states_to_accept(a: &Automaton) -> bool {
    let from = get_live_states_from_initial(a);
    let to = get_live_states_to_accept(a);
    to.iter().zip(&from).any(|(&t, &f)| t && !f)
}

/// `union(a1, a2)`.
pub fn union2(a1: &Automaton, a2: &Automaton) -> Automaton {
    union(&[a1, a2])
}

/// `union(list)`: the language of any automaton in the list.
pub fn union(list: &[&Automaton]) -> Automaton {
    let mut result = Automaton::new();
    result.create_state();
    for a in list {
        result.copy(a);
    }
    let mut state_offset = 1;
    for a in list {
        if a.get_num_states() == 0 {
            continue;
        }
        result.add_epsilon(0, state_offset);
        state_offset += a.get_num_states();
    }
    result.finish_state();
    merge_accept_states_with_no_transition(&remove_dead_states(&result))
}

/// `determinize(a, workLimit)`: subset construction.
///
/// # Errors
/// [`TooComplexToDeterminize`] once the summed size of the expanded state
/// sets reaches `10 * work_limit`.
pub fn determinize(a: &Automaton, work_limit: i32) -> Result<Automaton, TooComplexToDeterminize> {
    if a.is_deterministic() || a.get_num_states() <= 1 {
        return Ok(a.clone());
    }
    let mut b = Builder::new();
    b.create_state();
    b.set_accept(0, a.is_accept(0));
    let mut worklist: VecDeque<FrozenIntSet> = VecDeque::new();
    let mut newstate: HashMap<Vec<i32>, i32> = HashMap::new();
    worklist.push_back(FrozenIntSet::singleton(0, 0));
    newstate.insert(vec![0], 0);
    let mut points = PointTransitionSet::default();
    let mut states_set = StateSet::new(5);
    let mut t = Transition::new();
    let mut effort_spent: i64 = 0;
    let effort_limit = i64::from(work_limit) * 10;
    while let Some(s) = worklist.pop_front() {
        effort_spent += s.values.len() as i64;
        if effort_spent >= effort_limit {
            return Err(TooComplexToDeterminize {
                num_states: a.get_num_states(),
                num_transitions: a.get_total_num_transitions(),
                determinize_work_limit: work_limit,
                regexp: None,
            });
        }
        for &s0 in &s.values {
            let n = a.init_transition(s0, &mut t);
            for _ in 0..n {
                a.get_next_transition(&mut t);
                points.add(&t);
            }
        }
        if points.is_empty() {
            continue;
        }
        let pts = points.take_sorted();
        let mut last_point = -1;
        let mut acc_count = 0i32;
        let r = s.state;
        for pt in &pts {
            let point = pt.point;
            if !states_set.is_empty() {
                let key = states_set.members();
                let q = match newstate.get(key) {
                    Some(&q) => q,
                    None => {
                        let q = b.create_state();
                        let frozen = states_set.freeze(q);
                        b.set_accept(q, acc_count > 0);
                        newstate.insert(frozen.values.clone(), q);
                        worklist.push_back(frozen);
                        q
                    }
                };
                b.add_transition(r, q, last_point, point - 1);
            }
            for &dest in &pt.ends {
                states_set.decr(dest);
                acc_count -= i32::from(a.is_accept(dest));
            }
            for &dest in &pt.starts {
                states_set.incr(dest);
                acc_count += i32::from(a.is_accept(dest));
            }
            last_point = point;
        }
        debug_assert!(states_set.is_empty());
    }
    Ok(b.finish())
}

/// One label boundary of the determinize sweep: the destinations of the
/// transitions starting at it, and of those ending just before it.
#[derive(Default)]
pub(crate) struct PointTransitions {
    pub(crate) point: i32,
    pub(crate) starts: Vec<i32>,
    pub(crate) ends: Vec<i32>,
}

/// Java's `PointTransitionSet`: transitions bucketed by boundary label.
#[derive(Default)]
pub(crate) struct PointTransitionSet {
    points: Vec<PointTransitions>,
    index: HashMap<i32, usize>,
}

impl PointTransitionSet {
    fn find(&mut self, point: i32) -> &mut PointTransitions {
        let len = self.points.len();
        let i = *self.index.entry(point).or_insert(len);
        if i == len {
            self.points.push(PointTransitions {
                point,
                ..PointTransitions::default()
            });
        }
        &mut self.points[i]
    }

    /// `add(t)`: `t` starts at `t.min` and ends before `t.max + 1`.
    pub(crate) fn add(&mut self, t: &Transition) {
        self.find(t.min).starts.push(t.dest);
        self.find(t.max + 1).ends.push(t.dest);
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The boundaries in ascending order, leaving the set empty (Java's
    /// `sort()` followed by the sweep's per-point resets and `reset()`).
    pub(crate) fn take_sorted(&mut self) -> Vec<PointTransitions> {
        self.index.clear();
        let mut pts = std::mem::take(&mut self.points);
        pts.sort_unstable_by_key(|p| p.point);
        pts
    }
}

/// `isEmpty(a)`: `a` accepts no string.
pub fn is_empty(a: &Automaton) -> bool {
    if a.get_num_states() == 0 {
        return true;
    }
    if !a.is_accept(0) && a.get_num_transitions(0) == 0 {
        return true;
    }
    if a.is_accept(0) {
        return false;
    }
    let mut work: VecDeque<i32> = VecDeque::new();
    let mut seen = vec![false; idx(a.get_num_states())];
    work.push_back(0);
    seen[0] = true;
    let mut t = Transition::new();
    while let Some(state) = work.pop_front() {
        if a.is_accept(state) {
            return false;
        }
        let count = a.init_transition(state, &mut t);
        for _ in 0..count {
            a.get_next_transition(&mut t);
            if !seen[idx(t.dest)] {
                work.push_back(t.dest);
                seen[idx(t.dest)] = true;
            }
        }
    }
    true
}

/// `isTotal(a)`: `a` accepts every code-point string.
pub fn is_total(a: &Automaton) -> bool {
    is_total_range(a, MIN_CODE_POINT, MAX_CODE_POINT)
}

/// `isTotal(a, minAlphabet, maxAlphabet)`: every live state accepts and
/// covers the whole alphabet.
pub fn is_total_range(a: &Automaton, min_alphabet: i32, max_alphabet: i32) -> bool {
    let live = get_live_states(a);
    let mut spare = Transition::new();
    let mut seen = 0;
    for state in 0..a.get_num_states() {
        if !live[idx(state)] {
            continue;
        }
        if !a.is_accept(state) {
            return false;
        }
        let mut previous = min_alphabet - 1;
        for tr in 0..a.get_num_transitions(state) {
            a.get_transition(state, tr, &mut spare);
            if spare.min > previous + 1 {
                return false;
            }
            previous = spare.max;
        }
        if previous < max_alphabet {
            return false;
        }
        seen += 1;
    }
    seen > 0
}

/// `run(a, String)`: whether deterministic `a` accepts `s`.
pub fn run(a: &Automaton, s: &str) -> bool {
    let mut state = 0;
    for c in s.chars() {
        let next = a.step(state, c as i32);
        if next == -1 {
            return false;
        }
        state = next;
    }
    a.is_accept(state)
}

/// `run(a, IntsRef)`: whether deterministic `a` accepts the label sequence.
pub fn run_ints(a: &Automaton, s: &[i32]) -> bool {
    let mut state = 0;
    for &label in s {
        let next = a.step(state, label);
        if next == -1 {
            return false;
        }
        state = next;
    }
    a.is_accept(state)
}

fn get_live_states(a: &Automaton) -> Vec<bool> {
    let mut live = get_live_states_from_initial(a);
    let to = get_live_states_to_accept(a);
    for (l, t) in live.iter_mut().zip(to) {
        *l = *l && t;
    }
    live
}

fn get_live_states_from_initial(a: &Automaton) -> Vec<bool> {
    let n = a.get_num_states();
    let mut live = vec![false; idx(n)];
    if n == 0 {
        return live;
    }
    let mut work: VecDeque<i32> = VecDeque::new();
    live[0] = true;
    work.push_back(0);
    let mut t = Transition::new();
    while let Some(s) = work.pop_front() {
        let count = a.init_transition(s, &mut t);
        for _ in 0..count {
            a.get_next_transition(&mut t);
            if !live[idx(t.dest)] {
                live[idx(t.dest)] = true;
                work.push_back(t.dest);
            }
        }
    }
    live
}

fn get_live_states_to_accept(a: &Automaton) -> Vec<bool> {
    let n = a.get_num_states();
    let mut reverse: Vec<Vec<i32>> = vec![Vec::new(); idx(n)];
    let mut t = Transition::new();
    for s in 0..n {
        let count = a.init_transition(s, &mut t);
        for _ in 0..count {
            a.get_next_transition(&mut t);
            reverse[idx(t.dest)].push(s);
        }
    }
    let mut live = vec![false; idx(n)];
    let mut work: VecDeque<i32> = VecDeque::new();
    for s in 0..n {
        if a.is_accept(s) {
            live[idx(s)] = true;
            work.push_back(s);
        }
    }
    while let Some(s) = work.pop_front() {
        for &src in &reverse[idx(s)] {
            if !live[idx(src)] {
                live[idx(src)] = true;
                work.push_back(src);
            }
        }
    }
    live
}

/// `removeDeadStates(a)`: drop states unreachable from the initial state or
/// unable to reach an accept state, renumbering the rest in order. Returns
/// a clone of `a` when there are none.
pub fn remove_dead_states(a: &Automaton) -> Automaton {
    let n = a.get_num_states();
    let live = get_live_states(a);
    if live.iter().all(|&b| b) {
        return a.clone();
    }
    let mut map = vec![0i32; idx(n)];
    let mut result = Automaton::new();
    for i in 0..n {
        if live[idx(i)] {
            map[idx(i)] = result.create_state();
            result.set_accept(map[idx(i)], a.is_accept(i));
        }
    }
    let mut t = Transition::new();
    for i in 0..n {
        if live[idx(i)] {
            let count = a.init_transition(i, &mut t);
            for _ in 0..count {
                a.get_next_transition(&mut t);
                if live[idx(t.dest)] {
                    result.add_transition(map[idx(i)], map[idx(t.dest)], t.min, t.max);
                }
            }
        }
    }
    result.finish_state();
    result
}

/// Java's package-private `mergeAcceptStatesWithNoTransition`: collapse all
/// accept states without outgoing transitions into the first of them.
pub(crate) fn merge_accept_states_with_no_transition(a: &Automaton) -> Automaton {
    let n = a.get_num_states();
    let combined: Vec<i32> = (0..n)
        .filter(|&i| a.is_accept(i) && a.get_num_transitions(i) == 0)
        .collect();
    if combined.len() <= 1 {
        return a.clone();
    }
    let remap = |s: i32| -> i32 {
        match combined.binary_search(&s) {
            Ok(_) => combined[0],
            Err(i) => {
                if i <= 1 {
                    s
                } else {
                    s - (i as i32 - 1)
                }
            }
        }
    };
    let mut result = Automaton::new();
    for s in 0..n {
        let rs = remap(s);
        while result.get_num_states() <= rs {
            result.create_state();
        }
        if a.is_accept(s) {
            result.set_accept(rs, true);
        }
    }
    let mut t = Transition::new();
    for s in 0..n {
        let rs = remap(s);
        let count = a.init_transition(s, &mut t);
        for _ in 0..count {
            a.get_next_transition(&mut t);
            result.add_transition(rs, remap(t.dest), t.min, t.max);
        }
    }
    result.finish_state();
    result
}

/// `getCommonPrefix(a)`: the longest label sequence every accepted string
/// starts with. Java returns it as a `String`; code points are returned
/// here, since a label need not be a valid `char`.
///
/// # Errors
/// `IllegalArgument` when `a` has states reachable from the initial state
/// that cannot reach an accept state.
pub fn get_common_prefix(a: &Automaton) -> Result<Vec<i32>, AutomatonError> {
    if has_dead_states_from_initial(a) {
        return Err(AutomatonError::IllegalArgument(
            "input automaton has dead states".into(),
        ));
    }
    let mut out = Vec::new();
    if is_empty(a) {
        return Ok(out);
    }
    let n = idx(a.get_num_states());
    let mut scratch = Transition::new();
    let mut current = vec![false; n];
    let mut next = vec![false; n];
    current[0] = true;
    'algorithm: loop {
        let mut label = -1;
        for (state, &live) in current.iter().enumerate() {
            if !live {
                continue;
            }
            let st = state as i32;
            if a.is_accept(st) {
                break 'algorithm;
            }
            for tr in 0..a.get_num_transitions(st) {
                a.get_transition(st, tr, &mut scratch);
                if label == -1 {
                    label = scratch.min;
                }
                if scratch.min != scratch.max || scratch.min != label {
                    break 'algorithm;
                }
                next[idx(scratch.dest)] = true;
            }
        }
        out.push(label);
        std::mem::swap(&mut current, &mut next);
        next.iter_mut().for_each(|b| *b = false);
    }
    Ok(out)
}

/// `getCommonPrefixBytesRef(a)`: [`get_common_prefix`] as bytes.
///
/// # Errors
/// As [`get_common_prefix`], plus `IllegalState("automaton is not binary")`
/// for a prefix label above 255.
pub fn get_common_prefix_bytes_ref(a: &Automaton) -> Result<Vec<u8>, AutomatonError> {
    let prefix = get_common_prefix(a)?;
    let mut out = Vec::with_capacity(prefix.len());
    for cp in prefix {
        match u8::try_from(cp) {
            Ok(b) => out.push(b),
            Err(_) => {
                return Err(AutomatonError::IllegalState(
                    "automaton is not binary".into(),
                ));
            }
        }
    }
    Ok(out)
}

/// `getSingleton(a)`: the one string a deterministic `a` accepts, if it
/// accepts exactly one along a simple chain.
///
/// # Errors
/// `IllegalArgument` when `a` is not deterministic.
pub fn get_singleton(a: &Automaton) -> Result<Option<Vec<i32>>, AutomatonError> {
    if !a.is_deterministic() {
        return Err(AutomatonError::IllegalArgument(
            "input automaton must be deterministic".into(),
        ));
    }
    let mut out = Vec::new();
    let mut visited: HashSet<i32> = HashSet::new();
    let mut s = 0;
    let mut t = Transition::new();
    loop {
        visited.insert(s);
        if !a.is_accept(s) {
            if a.get_num_transitions(s) == 1 {
                a.get_transition(s, 0, &mut t);
                if t.min == t.max && !visited.contains(&t.dest) {
                    out.push(t.min);
                    s = t.dest;
                    continue;
                }
            }
        } else if a.get_num_transitions(s) == 0 {
            return Ok(Some(out));
        }
        return Ok(None);
    }
}

/// `getCommonSuffixBytesRef(a)`: the longest byte suffix every accepted
/// string shares.
///
/// # Errors
/// As [`get_common_prefix_bytes_ref`] on the reversed automaton.
pub fn get_common_suffix_bytes_ref(a: &Automaton) -> Result<Vec<u8>, AutomatonError> {
    let r = reverse(a);
    let mut out = get_common_prefix_bytes_ref(&r)?;
    out.reverse();
    Ok(out)
}

/// `reverse(a)`: the automaton accepting every accepted string reversed.
pub fn reverse(a: &Automaton) -> Automaton {
    reverse_with_initials(a, None)
}

fn reverse_with_initials(a: &Automaton, mut initials: Option<&mut Vec<i32>>) -> Automaton {
    if is_empty(a) {
        return Automaton::new();
    }
    let n = a.get_num_states();
    let mut builder = Builder::new();
    builder.create_state();
    for _ in 0..n {
        builder.create_state();
    }
    builder.set_accept(1, true);
    let mut t = Transition::new();
    for s in 0..n {
        let count = a.init_transition(s, &mut t);
        for _ in 0..count {
            a.get_next_transition(&mut t);
            builder.add_transition(t.dest + 1, s + 1, t.min, t.max);
        }
    }
    let mut result = builder.finish();
    for s in 0..n {
        if a.is_accept(s) {
            result.add_epsilon(0, s + 1);
            if let Some(v) = initials.as_deref_mut() {
                v.push(s + 1);
            }
        }
    }
    result.finish_state();
    if initials.is_some() {
        // AutomatonTestUtil.reverseOriginal: no dead-state removal.
        result
    } else {
        remove_dead_states(&result)
    }
}

/// Java's package-private `totalize(a)`: add a dead state and route every
/// uncovered label to it.
pub(crate) fn totalize(a: &Automaton) -> Automaton {
    let mut result = Automaton::new();
    let n = a.get_num_states();
    for i in 0..n {
        result.create_state();
        result.set_accept(i, a.is_accept(i));
    }
    let dead = result.create_state();
    result.add_transition(dead, dead, MIN_CODE_POINT, MAX_CODE_POINT);
    let mut t = Transition::new();
    for i in 0..n {
        let mut maxi = MIN_CODE_POINT;
        let count = a.init_transition(i, &mut t);
        for _ in 0..count {
            a.get_next_transition(&mut t);
            result.add_transition(i, t.dest, t.min, t.max);
            if t.min > maxi {
                result.add_transition(i, dead, maxi, t.min - 1);
            }
            if t.max + 1 > maxi {
                maxi = t.max + 1;
            }
        }
        if maxi <= MAX_CODE_POINT {
            result.add_transition(i, dead, maxi, MAX_CODE_POINT);
        }
    }
    result.finish_state();
    result
}

/// `topoSortStates(a)`: the reachable states in topological order.
///
/// # Errors
/// `IllegalArgument("Input automaton has a cycle.")`.
pub fn topo_sort_states(a: &Automaton) -> Result<Vec<i32>, AutomatonError> {
    let n = a.get_num_states();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut on_stack = vec![false; idx(n)];
    let mut visited = vec![false; idx(n)];
    let mut stack: Vec<i32> = vec![0];
    let mut states: Vec<i32> = Vec::with_capacity(idx(n));
    let mut t = Transition::new();
    while let Some(&state) = stack.last() {
        let count = a.init_transition(state, &mut t);
        let mut pushed = false;
        for _ in 0..count {
            a.get_next_transition(&mut t);
            if !visited[idx(t.dest)] {
                visited[idx(t.dest)] = true;
                stack.push(t.dest);
                on_stack[idx(state)] = true;
                pushed = true;
                break;
            } else if on_stack[idx(t.dest)] {
                return Err(AutomatonError::IllegalArgument(
                    "Input automaton has a cycle.".into(),
                ));
            }
        }
        if !pushed {
            on_stack[idx(state)] = false;
            stack.pop();
            states.push(state);
        }
    }
    states.reverse();
    Ok(states)
}

// ---------------------------------------------------------------------------
// AutomatonTestUtil (lucene-test-framework) helpers
// ---------------------------------------------------------------------------

/// `AutomatonTestUtil.isFinite(a)`: `a` accepts finitely many strings
/// (no cycle is reachable).
///
/// # Errors
/// `IllegalArgument("input automaton is too large: <level>")` past
/// [`MAX_RECURSION_LEVEL`], as in Java.
pub fn is_finite(a: &Automaton) -> Result<bool, AutomatonError> {
    let n = a.get_num_states();
    if n == 0 {
        return Ok(true);
    }
    let mut path = vec![false; idx(n)];
    let mut visited = vec![false; idx(n)];
    is_finite_rec(a, 0, &mut path, &mut visited, 0)
}

fn is_finite_rec(
    a: &Automaton,
    state: i32,
    path: &mut [bool],
    visited: &mut [bool],
    level: i32,
) -> Result<bool, AutomatonError> {
    if level > MAX_RECURSION_LEVEL {
        return Err(AutomatonError::IllegalArgument(format!(
            "input automaton is too large: {level}"
        )));
    }
    path[idx(state)] = true;
    let mut scratch = Transition::new();
    let n = a.get_num_transitions(state);
    for tr in 0..n {
        a.get_transition(state, tr, &mut scratch);
        let d = idx(scratch.dest);
        if path[d] || (!visited[d] && !is_finite_rec(a, scratch.dest, path, visited, level + 1)?) {
            return Ok(false);
        }
    }
    path[idx(state)] = false;
    visited[idx(state)] = true;
    Ok(true)
}

/// `AutomatonTestUtil.sameLanguage(a1, a2)`: both deterministic automata
/// accept the same strings.
///
/// # Errors
/// As [`subset_of`].
pub fn same_language(a1: &Automaton, a2: &Automaton) -> Result<bool, AutomatonError> {
    if std::ptr::eq(a1, a2) {
        return Ok(true);
    }
    Ok(subset_of(a2, a1)? && subset_of(a1, a2)?)
}

/// `AutomatonTestUtil.subsetOf(a1, a2)`: every string deterministic `a1`
/// accepts, deterministic `a2` accepts.
///
/// # Errors
/// `IllegalArgument` when either is not deterministic.
pub fn subset_of(a1: &Automaton, a2: &Automaton) -> Result<bool, AutomatonError> {
    if !a1.is_deterministic() {
        return Err(AutomatonError::IllegalArgument(
            "a1 must be deterministic".into(),
        ));
    }
    if !a2.is_deterministic() {
        return Err(AutomatonError::IllegalArgument(
            "a2 must be deterministic".into(),
        ));
    }
    if a1.get_num_states() == 0 {
        return Ok(true);
    } else if a2.get_num_states() == 0 {
        return Ok(is_empty(a1));
    }
    let t1s = a1.get_sorted_transitions();
    let t2s = a2.get_sorted_transitions();
    let mut worklist: VecDeque<(i32, i32)> = VecDeque::new();
    let mut visited: HashSet<(i32, i32)> = HashSet::new();
    worklist.push_back((0, 0));
    visited.insert((0, 0));
    while let Some((s1, s2)) = worklist.pop_front() {
        if a1.is_accept(s1) && !a2.is_accept(s2) {
            return Ok(false);
        }
        let t1 = &t1s[idx(s1)];
        let t2 = &t2s[idx(s2)];
        let mut b2 = 0;
        for tr1 in t1 {
            while b2 < t2.len() && t2[b2].max < tr1.min {
                b2 += 1;
            }
            let (mut min1, mut max1) = (tr1.min, tr1.max);
            let mut n2 = b2;
            while n2 < t2.len() && tr1.max >= t2[n2].min {
                if t2[n2].min > min1 {
                    return Ok(false);
                }
                if t2[n2].max < MAX_CODE_POINT {
                    min1 = t2[n2].max + 1;
                } else {
                    min1 = MAX_CODE_POINT;
                    max1 = MIN_CODE_POINT;
                }
                let q = (tr1.dest, t2[n2].dest);
                if visited.insert(q) {
                    worklist.push_back(q);
                }
                n2 += 1;
            }
            if min1 <= max1 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// `AutomatonTestUtil.determinizeSimple(a)`: textbook subset construction
/// over `getStartPoints`, without a work limit.
pub fn determinize_simple(a: &Automaton) -> Automaton {
    determinize_simple_from(a, &[0])
}

fn determinize_simple_from(a: &Automaton, initial: &[i32]) -> Automaton {
    if a.get_num_states() == 0 {
        return a.clone();
    }
    let points = a.get_start_points();
    let mut init: Vec<i32> = initial.to_vec();
    init.sort_unstable();
    init.dedup();
    let mut newstate: HashMap<Vec<i32>, i32> = HashMap::new();
    let mut worklist: VecDeque<Vec<i32>> = VecDeque::new();
    let mut result = Builder::new();
    result.create_state();
    newstate.insert(init.clone(), 0);
    worklist.push_back(init);
    let mut t = Transition::new();
    while let Some(s) = worklist.pop_front() {
        let r = newstate[&s];
        if s.iter().any(|&q| a.is_accept(q)) {
            result.set_accept(r, true);
        }
        for (n, &point) in points.iter().enumerate() {
            let mut p: Vec<i32> = Vec::new();
            for &q in &s {
                let count = a.init_transition(q, &mut t);
                for _ in 0..count {
                    a.get_next_transition(&mut t);
                    if t.min <= point && point <= t.max {
                        p.push(t.dest);
                    }
                }
            }
            p.sort_unstable();
            p.dedup();
            let q = match newstate.get(&p) {
                Some(&q) => q,
                None => {
                    let q = result.create_state();
                    newstate.insert(p.clone(), q);
                    worklist.push_back(p);
                    q
                }
            };
            let max = points.get(n + 1).map_or(MAX_CODE_POINT, |&next| next - 1);
            result.add_transition(r, q, point, max);
        }
    }
    remove_dead_states(&result.finish())
}

/// `AutomatonTestUtil.minimizeSimple(a)`: Brzozowski minimization (reverse,
/// determinize, twice) -- the minimal DFA for `a`'s language. Lucene 10
/// dropped `MinimizationOperations` from core; this is the minimizer its
/// tests use.
pub fn minimize(a: &Automaton) -> Automaton {
    let mut initials = Vec::new();
    let r = reverse_with_initials(a, Some(&mut initials));
    let d = determinize_simple_from(&r, &initials);
    initials.clear();
    let r2 = reverse_with_initials(&d, Some(&mut initials));
    determinize_simple_from(&r2, &initials)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::automata::{make_any_string, make_char, make_char_range, make_string};

    #[test]
    fn algebra_basics() {
        let ab = make_string("ab");
        let a = make_char('a' as i32);
        let u = union2(&ab, &a);
        assert!(run(&determinize(&u, 100).unwrap(), "ab"));
        let star = repeat(&a);
        let d = determinize(&star, 100).unwrap();
        assert!(run(&d, "") && run(&d, "aaaa") && !run(&d, "b"));
        assert!(same_language(&d, &d).unwrap());
        let two_plus = repeat_min(&a, 2);
        assert!(!run(&determinize(&two_plus, 100).unwrap(), "a"));
        let r = repeat_range(&a, 1, 3);
        let rd = determinize(&r, 100).unwrap();
        assert!(run(&rd, "aaa") && !run(&rd, "aaaa") && !run(&rd, ""));
        assert!(is_empty(&repeat_range(&a, 3, 1)));
        assert!(is_empty(&minus(&a, &a, 100).unwrap()));
        assert_eq!(minus(&a, &automata::make_empty(), 100).unwrap(), a);
        assert!(is_empty(&minus(&automata::make_empty(), &a, 100).unwrap()));
        assert_eq!(intersection(&a, &a), a);
        assert!(is_empty(&intersection(&automata::make_empty(), &a)));
        assert!(is_empty(&intersection(&a, &automata::make_empty())));
        assert!(is_empty(&concatenate(&[&a, &automata::make_empty()])));
        assert!(is_total(&make_any_string()));
        assert!(!is_total(&a));
        assert!(!is_total(&automata::make_empty()));
        assert!(run_ints(&ab, &['a' as i32, 'b' as i32]));
        assert!(!run_ints(&ab, &['a' as i32]));
        assert!(!run_ints(&ab, &['b' as i32]));
        assert_eq!(optional(&make_any_string()), make_any_string());
        let nfa = union2(&make_string("ab"), &make_string("ac"));
        let simple = determinize_simple(&nfa);
        assert!(simple.is_deterministic());
        assert!(same_language(&simple, &minimize(&nfa)).unwrap());
        assert!(subset_of(&make_string("ab"), &simple).unwrap());
        assert!(!subset_of(&simple, &make_string("ab")).unwrap());
        assert!(subset_of(&automata::make_empty(), &simple).unwrap());
        assert!(!subset_of(&simple, &automata::make_empty()).unwrap());
        assert_eq!(
            determinize_simple(&automata::make_empty()).get_num_states(),
            0
        );
        assert_eq!(nfa.get_accept_states().len(), nfa.get_num_states() as usize);
    }

    #[test]
    fn prefix_suffix_singleton_topo() {
        let a = make_string("abc");
        assert_eq!(get_common_prefix(&a).unwrap(), vec![97, 98, 99]);
        assert_eq!(get_common_suffix_bytes_ref(&a).unwrap(), b"abc".to_vec());
        assert_eq!(get_singleton(&a).unwrap(), Some(vec![97, 98, 99]));
        assert_eq!(topo_sort_states(&a).unwrap(), vec![0, 1, 2, 3]);
        assert!(topo_sort_states(&automata::make_empty())
            .unwrap()
            .is_empty());
        let star = repeat(&make_char('x' as i32));
        assert!(topo_sort_states(&star).is_err());
        assert!(!is_finite(&star).unwrap());
        assert!(is_finite(&automata::make_empty()).unwrap());
        let big = make_char_range(0x100, 0x200);
        assert!(matches!(
            get_common_prefix_bytes_ref(&big),
            Ok(ref v) if v.is_empty()
        ));
        let snow = make_string("\u{2603}");
        assert_eq!(
            get_common_prefix_bytes_ref(&snow),
            Err(AutomatonError::IllegalState(
                "automaton is not binary".into()
            ))
        );
        // A dead state reachable from the initial state.
        let mut dead = Automaton::new();
        dead.create_state();
        dead.create_state();
        dead.create_state();
        dead.set_accept(1, true);
        dead.add_transition(0, 1, 1, 1);
        dead.add_transition(0, 2, 2, 2);
        dead.finish_state();
        assert!(get_common_prefix(&dead).is_err());
        assert!(has_dead_states(&dead));
        assert!(has_dead_states_from_initial(&dead));
        assert!(!has_dead_states_to_accept(&dead));
        let nfa = union2(&make_string("ab"), &make_string("ac"));
        assert!(get_singleton(&nfa).is_err() || !nfa.is_deterministic());
        assert!(subset_of(&nfa, &a).is_err() || nfa.is_deterministic());
        assert!(subset_of(&a, &nfa).is_err() || nfa.is_deterministic());
    }

    #[test]
    fn is_finite_recursion_bound() {
        let long = "x".repeat(1100);
        let a = make_string(&long);
        assert!(matches!(
            is_finite(&a),
            Err(AutomatonError::IllegalArgument(ref m)) if m.starts_with("input automaton is too large")
        ));
    }

    #[test]
    fn too_complex() {
        let re = crate::automaton::RegExp::new("(a|b)*a(a|b){12}").unwrap();
        let a = re.to_automaton().unwrap();
        let e = determinize(&a, 100).unwrap_err();
        assert_eq!(e.determinize_work_limit, 100);
        assert!(e.to_string().starts_with("Determinizing automaton with"));
    }
}
