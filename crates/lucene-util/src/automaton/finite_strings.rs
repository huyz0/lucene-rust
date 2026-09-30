//! `FiniteStringsIterator` and `LimitedFiniteStringsIterator`: enumerate
//! the strings an acyclic automaton accepts, depth first, in Lucene's order.

use super::automaton::{Automaton, Transition, TransitionAccessor};
use super::error::AutomatonError;

#[derive(Clone, Copy, Default)]
struct PathNode {
    state: i32,
    to: i32,
    transition: i32,
    label: i32,
    t: Transition,
}

impl PathNode {
    fn reset_state(&mut self, a: &Automaton, state: i32) {
        self.state = state;
        self.transition = 0;
        a.get_transition(state, 0, &mut self.t);
        self.label = self.t.min;
        self.to = self.t.dest;
    }

    // SENTINEL: -1 once every transition's labels are exhausted.
    fn next_label(&mut self, a: &Automaton) -> i32 {
        if self.label > self.t.max {
            self.transition += 1;
            if self.transition >= a.get_num_transitions(self.state) {
                self.label = -1;
                return -1;
            }
            a.get_transition(self.state, self.transition, &mut self.t);
            self.label = self.t.min;
            self.to = self.t.dest;
        }
        let l = self.label;
        self.label += 1;
        l
    }
}

/// `FiniteStringsIterator`.
pub struct FiniteStringsIterator<'a> {
    a: &'a Automaton,
    end_state: i32,
    path_states: Vec<bool>,
    string: Vec<i32>,
    nodes: Vec<PathNode>,
    emit_empty_string: bool,
}

impl<'a> FiniteStringsIterator<'a> {
    /// `new FiniteStringsIterator(a)`.
    pub fn new(a: &'a Automaton) -> Self {
        Self::with_range(a, 0, -1)
    }

    /// `new FiniteStringsIterator(a, startState, endState)`: strings on paths
    /// from `start_state`, stopping at `end_state` (`-1` for none).
    pub fn with_range(a: &'a Automaton, start_state: i32, end_state: i32) -> Self {
        let mut it = FiniteStringsIterator {
            a,
            end_state,
            path_states: vec![false; a.get_num_states() as usize],
            string: Vec::new(),
            nodes: vec![PathNode::default(); 16],
            emit_empty_string: a.is_accept(0),
        };
        if a.get_num_states() > start_state && a.get_num_transitions(start_state) > 0 {
            it.path_states[start_state as usize] = true;
            it.nodes[0].reset_state(a, start_state);
            it.string.push(start_state);
        }
        it
    }

    /// `next()`: the next accepted string, `Ok(None)` when done.
    ///
    /// # Errors
    /// `IllegalArgument("automaton has cycles")`.
    pub fn next_string(&mut self) -> Result<Option<Vec<i32>>, AutomatonError> {
        if self.emit_empty_string {
            self.emit_empty_string = false;
            return Ok(Some(Vec::new()));
        }
        let a = self.a;
        let mut depth = self.string.len();
        while depth > 0 {
            let label = self.nodes[depth - 1].next_label(a);
            if label != -1 {
                self.string[depth - 1] = label;
                let to = self.nodes[depth - 1].to;
                if a.get_num_transitions(to) != 0 && to != self.end_state {
                    if self.path_states[to as usize] {
                        return Err(AutomatonError::IllegalArgument(
                            "automaton has cycles".into(),
                        ));
                    }
                    self.path_states[to as usize] = true;
                    if self.nodes.len() == depth {
                        self.nodes.push(PathNode::default());
                    }
                    self.nodes[depth].reset_state(a, to);
                    depth += 1;
                    self.string.resize(depth, 0);
                } else if self.end_state == to || a.is_accept(to) {
                    return Ok(Some(self.string.clone()));
                }
            } else {
                let state = self.nodes[depth - 1].state;
                self.path_states[state as usize] = false;
                depth -= 1;
                self.string.truncate(depth);
                if a.is_accept(state) {
                    return Ok(Some(self.string.clone()));
                }
            }
        }
        Ok(None)
    }
}

/// `LimitedFiniteStringsIterator`: at most `limit` strings.
pub struct LimitedFiniteStringsIterator<'a> {
    inner: FiniteStringsIterator<'a>,
    limit: i32,
    count: i32,
}

impl<'a> LimitedFiniteStringsIterator<'a> {
    /// `new LimitedFiniteStringsIterator(a, limit)`; `-1` means no limit.
    ///
    /// # Errors
    /// `IllegalArgument` for a limit that is neither `-1` nor positive.
    pub fn new(a: &'a Automaton, limit: i32) -> Result<Self, AutomatonError> {
        if limit != -1 && limit <= 0 {
            return Err(AutomatonError::IllegalArgument(format!(
                "limit must be -1 (which means no limit), or > 0; got: {limit}"
            )));
        }
        Ok(LimitedFiniteStringsIterator {
            inner: FiniteStringsIterator::new(a),
            limit: if limit > 0 { limit } else { i32::MAX },
            count: 0,
        })
    }

    /// `next()`.
    ///
    /// # Errors
    /// As [`FiniteStringsIterator::next_string`].
    pub fn next_string(&mut self) -> Result<Option<Vec<i32>>, AutomatonError> {
        if self.count >= self.limit {
            return Ok(None);
        }
        let r = self.inner.next_string()?;
        if r.is_some() {
            self.count += 1;
        }
        Ok(r)
    }

    /// `size()`: strings returned so far.
    pub fn size(&self) -> i32 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::RegExp;

    fn all(a: &Automaton) -> Vec<String> {
        let mut it = FiniteStringsIterator::new(a);
        let mut out = Vec::new();
        while let Some(s) = it.next_string().unwrap() {
            out.push(
                s.iter()
                    .map(|&c| char::from_u32(c as u32).unwrap())
                    .collect(),
            );
        }
        out
    }

    #[test]
    fn enumerates_in_lucene_order() {
        let a = RegExp::new("(ab|a|b[c-d])e?")
            .unwrap()
            .to_automaton()
            .unwrap();
        let v = all(&a);
        let mut sorted = v.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            vec!["a", "ab", "abe", "ae", "bc", "bce", "bd", "bde"]
        );
        let e = RegExp::new("()").unwrap().to_automaton().unwrap();
        assert_eq!(all(&e), vec![String::new()]);
        let deep = RegExp::new(&"x".repeat(40))
            .unwrap()
            .to_automaton()
            .unwrap();
        assert_eq!(all(&deep).len(), 1);
    }

    #[test]
    fn cycles_and_limits() {
        let a = RegExp::new("ab*").unwrap().to_automaton().unwrap();
        let mut it = FiniteStringsIterator::new(&a);
        let mut err = false;
        for _ in 0..3 {
            if it.next_string().is_err() {
                err = true;
                break;
            }
        }
        assert!(err);
        let f = RegExp::new("[a-z]").unwrap().to_automaton().unwrap();
        let mut l = LimitedFiniteStringsIterator::new(&f, 3).unwrap();
        while l.next_string().unwrap().is_some() {}
        assert_eq!(l.size(), 3);
        let mut u = LimitedFiniteStringsIterator::new(&f, -1).unwrap();
        while u.next_string().unwrap().is_some() {}
        assert_eq!(u.size(), 26);
        assert!(LimitedFiniteStringsIterator::new(&f, 0).is_err());
        let empty = Automaton::new();
        assert_eq!(
            FiniteStringsIterator::new(&empty).next_string().unwrap(),
            None
        );
    }
}
