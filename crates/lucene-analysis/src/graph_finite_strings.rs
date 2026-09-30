//! `org.apache.lucene.util.graph.GraphTokenStreamFiniteStrings`: a token
//! graph (tokens spanning several positions, stacked synonyms) as a
//! deterministic automaton over token ids, whose finite strings are the
//! graph's linear paths.
//!
//! `lucene_search::query_builder::QueryBuilder`'s graph queries are built
//! from it: [`GraphTokenStreamFiniteStrings::articulation_points`]
//! cut the graph into segments every path crosses, a segment with
//! [side paths](GraphTokenStreamFiniteStrings::has_side_path) becomes a
//! disjunction of its [finite strings](GraphTokenStreamFiniteStrings::finite_strings),
//! and one without is a synonym set of its [terms](GraphTokenStreamFiniteStrings::terms).
//!
//! Java lives in `util.graph` because it builds an `Automaton`; it reads a
//! `TokenStream`, so here it sits with the token streams, over
//! `lucene_util::automaton`.

use lucene_util::automaton::{
    operations, Automaton, Builder as AutomatonBuilder, FiniteStringsIterator, Transition,
    TransitionAccessor, DEFAULT_DETERMINIZE_WORK_LIMIT,
};

use crate::attributes::AttributeSource;
use crate::token_stream::TokenStream;
use crate::AnalysisError;

/// `MAX_RECURSION_LEVEL`: the deepest `articulationPointsRecurse` goes.
const MAX_RECURSION_LEVEL: i32 = 1000;

/// `GraphTokenStreamFiniteStrings`.
#[derive(Debug, Clone)]
pub struct GraphTokenStreamFiniteStrings {
    /// `tokens[id]`: each token's attributes, as a linear path replays them
    /// (position length 1; a stacked token with its position's increment).
    tokens: Vec<AttributeSource>,
    /// The determinized automaton, dead states removed.
    det: Automaton,
}

impl GraphTokenStreamFiniteStrings {
    /// `new GraphTokenStreamFiniteStrings(in)`: resets, consumes and ends
    /// `input`, then determinizes the graph.
    ///
    /// # Errors
    /// A first token with a position increment below 1 (`Malformed
    /// TokenStream`), a graph too complex to determinize, and `input`'s
    /// own errors.
    pub fn new(input: &mut dyn TokenStream) -> Result<Self, AnalysisError> {
        let mut tokens = Vec::new();
        let aut = build(input, &mut tokens)?;
        let det = operations::determinize(&aut, DEFAULT_DETERMINIZE_WORK_LIMIT)
            .map_err(|e| AnalysisError::IllegalState(e.to_string()))?;
        Ok(Self {
            tokens,
            det: operations::remove_dead_states(&det),
        })
    }

    /// `hasSidePath(state)`: whether `state`'s transitions lead to more than
    /// one state.
    pub fn has_side_path(&self, state: i32) -> bool {
        let mut t = Transition::new();
        let num = self.det.init_transition(state, &mut t);
        if num <= 1 {
            return false;
        }
        self.det.get_next_transition(&mut t);
        let dest = t.dest;
        for _ in 1..num {
            self.det.get_next_transition(&mut t);
            if dest != t.dest {
                return true;
            }
        }
        false
    }

    /// `getTerms(state)`: the tokens leaving `state`, in transition order.
    pub fn terms(&self, state: i32) -> Vec<&AttributeSource> {
        let mut t = Transition::new();
        let num = self.det.init_transition(state, &mut t);
        let mut out = Vec::new();
        for _ in 0..num {
            self.det.get_next_transition(&mut t);
            for id in t.min..=t.max {
                if let Some(tok) = usize::try_from(id).ok().and_then(|i| self.tokens.get(i)) {
                    out.push(tok);
                }
            }
        }
        out
    }

    /// `getFiniteStrings()`: every path of the whole graph.
    ///
    /// # Errors
    /// As [`Self::finite_strings_between`].
    pub fn finite_strings(&self) -> Result<Vec<FiniteStringsTokenStream>, AnalysisError> {
        self.finite_strings_between(0, -1)
    }

    /// `getFiniteStrings(startState, endState)`: every path from
    /// `start_state` to `end_state` (`-1`: to an accept state), each as a
    /// linear token stream.
    ///
    /// # Errors
    /// An automaton that is not finite (`FiniteStringsIterator`'s refusal).
    pub fn finite_strings_between(
        &self,
        start_state: i32,
        end_state: i32,
    ) -> Result<Vec<FiniteStringsTokenStream>, AnalysisError> {
        let mut it = FiniteStringsIterator::with_range(&self.det, start_state, end_state);
        let mut out = Vec::new();
        while let Some(ids) = it
            .next_string()
            .map_err(|e| AnalysisError::IllegalArgument(e.to_string()))?
        {
            out.push(FiniteStringsTokenStream {
                atts: self.tokens.first().cloned().unwrap_or_default(),
                tokens: ids
                    .iter()
                    .filter_map(|&id| usize::try_from(id).ok())
                    .filter_map(|i| self.tokens.get(i).cloned())
                    .collect(),
                offset: 0,
            });
        }
        Ok(out)
    }

    /// `articulationPoints()`: the states every path from the start to the
    /// end goes through, in increasing order (Tarjan's articulation points
    /// over the undirected graph).
    ///
    /// # Errors
    /// A graph deeper than [`MAX_RECURSION_LEVEL`] states.
    pub fn articulation_points(&self) -> Result<Vec<i32>, AnalysisError> {
        let n = self.det.get_num_states();
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut undirect = AutomatonBuilder::new();
        undirect.copy(&self.det);
        let mut t = Transition::new();
        for i in 0..n {
            let num = self.det.init_transition(i, &mut t);
            for _ in 0..num {
                self.det.get_next_transition(&mut t);
                undirect.add_transition_label(t.dest, i, t.min);
            }
        }
        let undirected = undirect.finish();
        let size = usize::try_from(n).unwrap_or(0);
        let mut walk = Walk {
            visited: vec![false; size],
            depth: vec![0; size],
            low: vec![0; size],
            parent: vec![-1; size],
            points: Vec::new(),
        };
        walk.recurse(&undirected, 0, 0)?;
        walk.points.reverse();
        Ok(walk.points)
    }
}

/// `articulationPointsRecurse`'s arrays.
struct Walk {
    visited: Vec<bool>,
    depth: Vec<i32>,
    low: Vec<i32>,
    parent: Vec<i32>,
    points: Vec<i32>,
}

impl Walk {
    fn recurse(&mut self, a: &Automaton, state: i32, d: i32) -> Result<(), AnalysisError> {
        let s = state as usize;
        self.visited[s] = true;
        self.depth[s] = d;
        self.low[s] = d;
        let mut child_count = 0;
        let mut is_articulation = false;
        let mut t = Transition::new();
        let num = a.init_transition(state, &mut t);
        for _ in 0..num {
            a.get_next_transition(&mut t);
            let dest = t.dest as usize;
            if !self.visited[dest] {
                self.parent[dest] = state;
                if d < MAX_RECURSION_LEVEL {
                    // The walk borrows `t`; a fresh one per level, as Java's.
                    let saved = t;
                    self.recurse(a, saved.dest, d + 1)?;
                    t = saved;
                } else {
                    return Err(AnalysisError::IllegalArgument(
                        "Exceeded maximum recursion level during graph analysis".to_string(),
                    ));
                }
                child_count += 1;
                if self.low[dest] >= self.depth[s] {
                    is_articulation = true;
                }
                self.low[s] = self.low[s].min(self.low[dest]);
            } else if t.dest != self.parent[s] {
                self.low[s] = self.low[s].min(self.depth[dest]);
            }
        }
        if (self.parent[s] != -1 && is_articulation) || (self.parent[s] == -1 && child_count > 1) {
            self.points.push(state);
        }
        Ok(())
    }
}

/// `GraphTokenStreamFiniteStrings.build`: the token graph as an automaton
/// whose transition labels are token ids, `tokens` filled with each token's
/// attributes as a linear path replays it.
fn build(
    input: &mut dyn TokenStream,
    tokens: &mut Vec<AttributeSource>,
) -> Result<Automaton, AnalysisError> {
    let mut builder = AutomatonBuilder::new();
    input.reset()?;
    let mut pos: i32 = -1;
    let mut prev_incr = 1;
    let mut state: i32 = -1;
    let mut id: i32 = -1;
    let mut gap = 0;
    while input.increment_token()? {
        let atts = input.attributes();
        let current_incr = atts.position_increment();
        if pos == -1 && current_incr < 1 {
            return Err(AnalysisError::IllegalState(
                "Malformed TokenStream, start token can't have increment less than 1".to_string(),
            ));
        }
        if current_incr == 0 {
            if gap > 0 {
                pos -= gap;
            }
        } else {
            pos += 1;
            gap = current_incr - 1;
        }
        let end_pos = pos + atts.position_length() + gap;
        while state < end_pos {
            state = builder.create_state();
        }
        id += 1;
        let mut token = atts.clone();
        builder.add_transition_label(pos, end_pos, id);
        pos += gap;
        // Linear paths: position length 1, and a stacked token takes its
        // position's increment.
        token.set_position_length(1)?;
        if current_incr == 0 {
            token.set_position_increment(prev_incr)?;
        }
        tokens.push(token);
        if current_incr > 0 {
            prev_incr = current_incr;
        }
    }
    input.end()?;
    if state != -1 {
        builder.set_accept(state, true);
    }
    Ok(builder.finish())
}

/// `FiniteStringsTokenStream`: one path of the graph, token after token.
#[derive(Debug, Clone)]
pub struct FiniteStringsTokenStream {
    atts: AttributeSource,
    tokens: Vec<AttributeSource>,
    offset: usize,
}

impl FiniteStringsTokenStream {
    /// The path's tokens.
    pub fn tokens(&self) -> &[AttributeSource] {
        &self.tokens
    }
}

impl TokenStream for FiniteStringsTokenStream {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        let Some(tok) = self.tokens.get(self.offset) else {
            return Ok(false);
        };
        // `clearAttributes(); tokens[id].copyTo(this)`.
        self.atts = tok.clone();
        self.offset += 1;
        Ok(true)
    }

    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.offset = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tokens `(term, posInc, posLen)`, as `CannedTokenStream` hands them.
    struct Canned {
        atts: AttributeSource,
        tokens: Vec<(&'static str, i32, i32)>,
        upto: usize,
    }

    impl TokenStream for Canned {
        fn attributes(&self) -> &AttributeSource {
            &self.atts
        }
        fn attributes_mut(&mut self) -> &mut AttributeSource {
            &mut self.atts
        }
        fn increment_token(&mut self) -> Result<bool, AnalysisError> {
            self.atts.clear_attributes();
            let Some(&(t, inc, len)) = self.tokens.get(self.upto) else {
                return Ok(false);
            };
            self.upto += 1;
            self.atts.set_term(t);
            self.atts.set_position_increment(inc)?;
            self.atts.set_position_length(len)?;
            Ok(true)
        }
        fn reset(&mut self) -> Result<(), AnalysisError> {
            self.upto = 0;
            Ok(())
        }
    }

    fn graph(tokens: Vec<(&'static str, i32, i32)>) -> GraphTokenStreamFiniteStrings {
        GraphTokenStreamFiniteStrings::new(&mut Canned {
            atts: AttributeSource::new(),
            tokens,
            upto: 0,
        })
        .unwrap()
    }

    fn paths(g: &GraphTokenStreamFiniteStrings, start: i32, end: i32) -> Vec<String> {
        let mut out: Vec<String> = g
            .finite_strings_between(start, end)
            .unwrap()
            .into_iter()
            .map(|mut ts| {
                let mut words = Vec::new();
                ts.reset().unwrap();
                while ts.increment_token().unwrap() {
                    let a = ts.attributes();
                    words.push(format!(
                        "{}/{}/{}",
                        a.term(),
                        a.position_increment(),
                        a.position_length()
                    ));
                }
                words.join(" ")
            })
            .collect();
        out.sort();
        out
    }

    /// `TestGraphTokenStreamFiniteStrings.testMultiTermSynonyms` and its
    /// neighbours: "fast wi fi network" with "wifi" over "wi fi".
    #[test]
    fn a_multi_token_synonym_splits_into_paths() {
        let g = graph(vec![
            ("fast", 1, 1),
            ("wi", 1, 1),
            ("wifi", 0, 2),
            ("fi", 1, 1),
            ("network", 1, 1),
        ]);
        assert_eq!(
            paths(&g, 0, -1),
            vec![
                "fast/1/1 wi/1/1 fi/1/1 network/1/1",
                "fast/1/1 wifi/1/1 network/1/1"
            ]
        );
        assert_eq!(g.articulation_points().unwrap(), vec![1, 3]);
        assert!(!g.has_side_path(0));
        assert!(g.has_side_path(1));
        assert!(!g.has_side_path(3));
        let terms: Vec<&str> = g.terms(0).iter().map(|a| a.term()).collect();
        assert_eq!(terms, vec!["fast"]);
        assert_eq!(paths(&g, 1, 3), vec!["wi/1/1 fi/1/1", "wifi/1/1"]);
    }

    /// Stacked single-position synonyms share a transition range; a hole
    /// (increment 2) keeps its gap in the stacked token's increment.
    #[test]
    fn stacked_synonyms_holes_and_errors() {
        let g = graph(vec![("a", 1, 1), ("b", 0, 1), ("c", 2, 1), ("d", 0, 1)]);
        let terms: Vec<&str> = g.terms(0).iter().map(|a| a.term()).collect();
        assert_eq!(terms, vec!["a", "b"]);
        assert!(!g.has_side_path(0));
        assert_eq!(
            paths(&g, 0, -1),
            vec!["a/1/1 c/2/1", "a/1/1 d/2/1", "b/1/1 c/2/1", "b/1/1 d/2/1"]
        );
        assert!(!g.has_side_path(99));
        let empty = graph(vec![]);
        assert!(empty.articulation_points().unwrap().is_empty());
        assert!(empty.finite_strings().unwrap().is_empty());
        let err = GraphTokenStreamFiniteStrings::new(&mut Canned {
            atts: AttributeSource::new(),
            tokens: vec![("a", 0, 1)],
            upto: 0,
        })
        .err()
        .unwrap();
        assert!(err.to_string().contains("Malformed TokenStream"));
    }

    /// A graph deeper than the recursion limit is refused, as Java's.
    #[test]
    fn a_very_long_graph_exceeds_the_recursion_level() {
        let tokens: Vec<(&'static str, i32, i32)> = (0..1100).map(|_| ("x", 1, 1)).collect();
        let g = graph(tokens);
        let err = g.articulation_points().err().unwrap();
        assert!(err.to_string().contains("maximum recursion level"));
    }
}
