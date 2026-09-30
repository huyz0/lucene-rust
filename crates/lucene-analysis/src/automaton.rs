//! `TokenStreamToAutomaton` and `AutomatonToTokenStream`, over a minimal
//! automaton.
//!
//! Java builds and reads `org.apache.lucene.util.automaton.Automaton`. That
//! API is not in this workspace yet (it is being ported into `lucene-util`),
//! and this crate depends on nothing in the workspace, so the two converters
//! run over [`Automaton`] here: the part of Java's class they use -- states,
//! accept bits, and per-state transitions in Java's canonical form (adjacent
//! ranges to the same destination merged, sorted by min, max, dest, exactly
//! what `Automaton.Builder.finish()` + `finishState()` produce). The
//! differential fixture compares that form state for state with Java's
//! `getSortedTransitions()`.

use crate::attributes::AttributeSource;
use crate::token_stream::TokenStream;
use crate::AnalysisError;

/// `TokenStreamToAutomaton.POS_SEP`: the label between positions.
pub const POS_SEP: i32 = 0x001f;

/// `TokenStreamToAutomaton.HOLE`: the label of a position no token covers.
pub const HOLE: i32 = 0x001e;

/// `org.apache.lucene.util.automaton.Transition`: `min..=max` to `dest`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Transition {
    pub dest: usize,
    pub min: i32,
    pub max: i32,
}

/// The finished automaton (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Automaton {
    accept: Vec<bool>,
    transitions: Vec<Vec<Transition>>,
}

impl Automaton {
    /// `getNumStates()`.
    pub fn num_states(&self) -> usize {
        self.accept.len()
    }

    /// `isAccept(int)`.
    pub fn is_accept(&self, state: usize) -> bool {
        self.accept[state]
    }

    /// `getNumTransitions()`.
    pub fn num_transitions(&self) -> usize {
        self.transitions.iter().map(Vec::len).sum()
    }

    /// The transitions leaving `state`, sorted by min, max, dest
    /// (`getSortedTransitions()[state]`).
    pub fn transitions(&self, state: usize) -> &[Transition] {
        &self.transitions[state]
    }
}

/// `Automaton.Builder`.
#[derive(Debug, Clone, Default)]
pub struct AutomatonBuilder {
    accept: Vec<bool>,
    /// (source, dest, min, max).
    transitions: Vec<(usize, usize, i32, i32)>,
}

impl AutomatonBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// `createState()`.
    pub fn create_state(&mut self) -> usize {
        self.accept.push(false);
        self.accept.len() - 1
    }

    /// `setAccept(int, boolean)`.
    pub fn set_accept(&mut self, state: usize, accept: bool) {
        self.accept[state] = accept;
    }

    /// `isAccept(int)`.
    pub fn is_accept(&self, state: usize) -> bool {
        self.accept[state]
    }

    /// `getNumStates()`.
    pub fn num_states(&self) -> usize {
        self.accept.len()
    }

    /// `addTransition(int source, int dest, int label)`.
    pub fn add_transition(&mut self, source: usize, dest: usize, label: i32) {
        self.add_transition_range(source, dest, label, label);
    }

    /// `addTransition(int source, int dest, int min, int max)`.
    pub fn add_transition_range(&mut self, source: usize, dest: usize, min: i32, max: i32) {
        self.transitions.push((source, dest, min, max));
    }

    /// `finish()`: per state, `finishCurrentState()`'s merge of adjacent or
    /// overlapping ranges to one destination, then the min/max/dest order.
    pub fn finish(self) -> Automaton {
        let mut per_state: Vec<Vec<Transition>> = vec![Vec::new(); self.accept.len()];
        for (src, dest, min, max) in self.transitions {
            per_state[src].push(Transition { dest, min, max });
        }
        for ts in &mut per_state {
            // destMinMaxSorter
            ts.sort_by_key(|t| (t.dest, t.min, t.max));
            let mut merged: Vec<Transition> = Vec::with_capacity(ts.len());
            for t in ts.iter() {
                match merged.last_mut() {
                    Some(last) if last.dest == t.dest && t.min <= last.max.saturating_add(1) => {
                        last.max = last.max.max(t.max);
                    }
                    _ => merged.push(*t),
                }
            }
            // minMaxDestSorter
            merged.sort_by_key(|t| (t.min, t.max, t.dest));
            *ts = merged;
        }
        Automaton {
            accept: self.accept,
            transitions: per_state,
        }
    }
}

/// `TokenStreamToAutomaton.Position`.
#[derive(Clone, Copy)]
struct Position {
    arriving: i64,
    leaving: i64,
}

impl Default for Position {
    fn default() -> Self {
        Position {
            arriving: -1,
            leaving: -1,
        }
    }
}

/// `RollingBuffer<Position>` as the converter uses it: `get` creates every
/// position up to the one asked for; `max_pos` is the highest created.
#[derive(Default)]
struct Positions(Vec<Position>);

impl Positions {
    fn get(&mut self, pos: i64) -> &mut Position {
        let pos = usize::try_from(pos).expect("positions are non-negative");
        if pos >= self.0.len() {
            self.0.resize(pos + 1, Position::default());
        }
        &mut self.0[pos]
    }

    fn max_pos(&self) -> i64 {
        self.0.len() as i64 - 1
    }
}

/// `org.apache.lucene.analysis.TokenStreamToAutomaton`: the token graph as an
/// automaton over term bytes (or code points), positions separated by
/// [`POS_SEP`], holes marked with [`HOLE`].
pub struct TokenStreamToAutomaton {
    preserve_position_increments: bool,
    final_offset_gap_as_hole: bool,
    unicode_arcs: bool,
    change_token: Option<ChangeToken>,
}

/// The `changeToken(BytesRef)` hook.
type ChangeToken = Box<dyn Fn(&[u8]) -> Vec<u8> + Send + Sync>;

impl Default for TokenStreamToAutomaton {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenStreamToAutomaton {
    /// `new TokenStreamToAutomaton()`.
    pub fn new() -> Self {
        TokenStreamToAutomaton {
            preserve_position_increments: true,
            final_offset_gap_as_hole: false,
            unicode_arcs: false,
            change_token: None,
        }
    }

    /// `setPreservePositionIncrements(boolean)`.
    pub fn set_preserve_position_increments(&mut self, enable: bool) {
        self.preserve_position_increments = enable;
    }

    /// `setFinalOffsetGapAsHole(boolean)`.
    pub fn set_final_offset_gap_as_hole(&mut self, enable: bool) {
        self.final_offset_gap_as_hole = enable;
    }

    /// `setUnicodeArcs(boolean)`: label arcs with code points, not UTF-8
    /// bytes.
    pub fn set_unicode_arcs(&mut self, enable: bool) {
        self.unicode_arcs = enable;
    }

    /// The protected `changeToken(BytesRef)` hook a Java subclass overrides.
    pub fn set_change_token(&mut self, f: impl Fn(&[u8]) -> Vec<u8> + Send + Sync + 'static) {
        self.change_token = Some(Box::new(f));
    }

    /// `toAutomaton(TokenStream)`: resets, consumes and ends `input` (the
    /// caller closes it, as in Java).
    pub fn to_automaton(&self, input: &mut dyn TokenStream) -> Result<Automaton, AnalysisError> {
        let mut builder = AutomatonBuilder::new();
        builder.create_state();

        input.reset()?;

        // Only temporarily holds states ahead of our current position:
        let mut positions = Positions::default();

        let mut pos: i64 = -1;
        let mut leaving_state: i64 = -1;
        let mut max_offset = 0;
        while input.increment_token()? {
            let atts: &AttributeSource = input.attributes();
            let mut pos_inc = atts.position_increment() as i64;
            if !self.preserve_position_increments && pos_inc > 1 {
                pos_inc = 1;
            }
            if pos == -1 && pos_inc == 0 {
                return Err(AnalysisError::IllegalState(
                    "first token must have a position increment > 0".to_string(),
                ));
            }

            if pos_inc > 0 {
                // New node:
                pos += pos_inc;

                let arriving = positions.get(pos).arriving;
                if arriving == -1 {
                    // No token ever arrived to this position
                    if pos == 0 {
                        // OK: this is the first token
                        positions.get(pos).leaving = 0;
                    } else {
                        // This means there's a hole (eg, StopFilter does this):
                        positions.get(pos).leaving = builder.create_state() as i64;
                        add_holes(&mut builder, &mut positions, pos);
                    }
                } else {
                    let leaving = builder.create_state();
                    positions.get(pos).leaving = leaving as i64;
                    builder.add_transition(arriving as usize, leaving, POS_SEP);
                    if pos_inc > 1 {
                        // A token spanned over a hole; add holes "under" it:
                        add_holes(&mut builder, &mut positions, pos);
                    }
                }
                leaving_state = positions.get(pos).leaving;
            }

            let end_pos = pos + atts.position_length() as i64;

            let changed;
            let term_utf8: &[u8] = match &self.change_token {
                Some(f) => {
                    changed = f(atts.term_bytes());
                    &changed
                }
                None => atts.term_bytes(),
            };
            let end_arriving = {
                let end_pos_data = positions.get(end_pos);
                if end_pos_data.arriving == -1 {
                    end_pos_data.arriving = builder.create_state() as i64;
                }
                end_pos_data.arriving as usize
            };

            let labels: Vec<i32> = if self.unicode_arcs {
                // BytesRef.utf8ToString() then code points.
                String::from_utf8_lossy(term_utf8)
                    .chars()
                    .map(|c| c as i32)
                    .collect()
            } else {
                term_utf8.iter().map(|&b| b as i32).collect()
            };

            let mut state = leaving_state as usize;
            let term_len = labels.len();
            for (i, &c) in labels.iter().enumerate() {
                let next_state = if i == term_len - 1 {
                    end_arriving
                } else {
                    builder.create_state()
                };
                builder.add_transition(state, next_state, c);
                state = next_state;
            }

            max_offset = max_offset.max(atts.end_offset());
        }

        input.end()?;

        let atts = input.attributes();
        let mut end_pos_inc = atts.position_increment();
        if end_pos_inc == 0 && self.final_offset_gap_as_hole && atts.end_offset() > max_offset {
            end_pos_inc = 1;
        } else if end_pos_inc > 0 && !self.preserve_position_increments {
            end_pos_inc = 0;
        }

        let end_state = if end_pos_inc > 0 {
            // there were hole(s) after the last token
            let end_state = builder.create_state();

            // add trailing holes now:
            let mut last_state = end_state;
            loop {
                let state1 = builder.create_state();
                builder.add_transition(last_state, state1, HOLE);
                end_pos_inc -= 1;
                if end_pos_inc == 0 {
                    builder.set_accept(state1, true);
                    break;
                }
                let state2 = builder.create_state();
                builder.add_transition(state1, state2, POS_SEP);
                last_state = state2;
            }
            Some(end_state)
        } else {
            None
        };

        pos += 1;
        while pos <= positions.max_pos() {
            let arriving = positions.get(pos).arriving;
            if arriving != -1 {
                match end_state {
                    Some(end_state) => {
                        builder.add_transition(arriving as usize, end_state, POS_SEP)
                    }
                    None => builder.set_accept(arriving as usize, true),
                }
            }
            pos += 1;
        }

        Ok(builder.finish())
    }
}

/// `TokenStreamToAutomaton.addHoles`.
fn add_holes(builder: &mut AutomatonBuilder, positions: &mut Positions, mut pos: i64) {
    loop {
        let pos_data = *positions.get(pos);
        let prev = *positions.get(pos - 1);
        if !(pos_data.arriving == -1 || prev.leaving == -1) {
            break;
        }
        if pos_data.arriving == -1 {
            let arriving = builder.create_state();
            positions.get(pos).arriving = arriving as i64;
            builder.add_transition(arriving, pos_data.leaving as usize, POS_SEP);
        }
        if prev.leaving == -1 {
            let leaving = if pos == 1 { 0 } else { builder.create_state() };
            positions.get(pos - 1).leaving = leaving as i64;
            if prev.arriving != -1 {
                builder.add_transition(prev.arriving as usize, leaving, POS_SEP);
            }
        }
        let from = positions.get(pos - 1).leaving as usize;
        let to = positions.get(pos).arriving as usize;
        builder.add_transition(from, to, HOLE);
        pos -= 1;
        if pos <= 0 {
            break;
        }
    }
}

/// `TokenStreamToAutomaton.toAutomaton` with the default settings.
pub fn token_stream_to_automaton(input: &mut dyn TokenStream) -> Result<Automaton, AnalysisError> {
    TokenStreamToAutomaton::new().to_automaton(input)
}

/// `AutomatonToTokenStream.toTokenStream(Automaton)`: an acyclic automaton
/// as a token graph, one single-`char` token per transition label, each
/// state's topological layer its position (offsets are layer numbers).
pub fn automaton_to_token_stream(automaton: &Automaton) -> Result<TopoTokenStream, AnalysisError> {
    let n = automaton.num_states();
    let mut position_nodes: Vec<Vec<usize>> = Vec::new();

    let mut indegree = vec![0i64; n];
    for s in 0..n {
        for t in automaton.transitions(s) {
            indegree[t.dest] += 1;
        }
    }
    if indegree.first().copied().unwrap_or(0) != 0 {
        return Err(AnalysisError::IllegalArgument(
            "Start node has incoming edges, creating cycle".to_string(),
        ));
    }

    let mut no_incoming_edges = std::collections::VecDeque::new();
    // IntIntHashMap.get of a missing key is 0.
    let mut id_to_pos = vec![0usize; n];
    if n > 0 {
        no_incoming_edges.push_back((0usize, 0usize));
    }
    while let Some((id, pos)) = no_incoming_edges.pop_front() {
        for t in automaton.transitions(id) {
            indegree[t.dest] -= 1;
            if indegree[t.dest] == 0 {
                no_incoming_edges.push_back((t.dest, pos + 1));
            }
        }
        if position_nodes.len() == pos {
            position_nodes.push(vec![id]);
        } else {
            position_nodes[pos].push(id);
        }
        id_to_pos[id] = pos;
    }

    if indegree.iter().any(|&d| d != 0) {
        return Err(AnalysisError::IllegalArgument(
            "Cycle found in automaton".to_string(),
        ));
    }

    let last_layer = position_nodes.len().saturating_sub(1);
    let mut edges_by_layer = Vec::with_capacity(position_nodes.len());
    for layer in &position_nodes {
        let mut edges = Vec::new();
        for &state in layer {
            for t in automaton.transitions(state) {
                // each edge in the token stream can only be one value,
                // though a transition takes a range.
                for val in t.min..=t.max {
                    let dest_layer = id_to_pos[t.dest];
                    edges.push((dest_layer, val));
                    // If there's an intermediate accept state, add an edge
                    // to the terminal state.
                    if automaton.is_accept(t.dest) && dest_layer != last_layer {
                        edges.push((last_layer, val));
                    }
                }
            }
        }
        edges_by_layer.push(edges);
    }

    Ok(TopoTokenStream {
        atts: AttributeSource::new(),
        edges_by_pos: edges_by_layer,
        current_pos: 0,
        current_edge_index: 0,
    })
}

/// `AutomatonToTokenStream.TopoTokenStream`.
pub struct TopoTokenStream {
    atts: AttributeSource,
    /// Per layer: (destination layer, label).
    edges_by_pos: Vec<Vec<(usize, i32)>>,
    current_pos: usize,
    current_edge_index: usize,
}

impl TokenStream for TopoTokenStream {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        self.atts.clear_attributes();
        while self.current_pos < self.edges_by_pos.len()
            && self.current_edge_index == self.edges_by_pos[self.current_pos].len()
        {
            self.current_edge_index = 0;
            self.current_pos += 1;
        }
        if self.current_pos == self.edges_by_pos.len() {
            return Ok(false);
        }
        let (destination, value) = self.edges_by_pos[self.current_pos][self.current_edge_index];

        // charAttr.append((char) value): Java truncates to 16 bits.
        self.atts.set_term_utf16(&[value as u16]);
        self.atts
            .set_position_increment(i32::from(self.current_edge_index == 0))?;
        // Java sets a length < 1 unchecked-by-assert only when a
        // transition goes backwards, which a topological order rules out.
        self.atts
            .set_position_length(destination as i32 - self.current_pos as i32)?;
        self.atts
            .set_offset(self.current_pos as i32, destination as i32)?;

        self.current_edge_index += 1;
        Ok(true)
    }

    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.atts.clear_attributes();
        self.current_pos = 0;
        self.current_edge_index = 0;
        Ok(())
    }

    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.clear_attributes();
        self.atts.set_position_increment(0)?;
        // -1 because we don't count the terminal state as a position in the
        // TokenStream
        let last = self.edges_by_pos.len() as i32 - 1;
        self.atts.set_offset(last, last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_stream::consume;

    struct Canned {
        atts: AttributeSource,
        tokens: Vec<(&'static str, i32, i32, i32)>,
        upto: usize,
        end: (i32, i32),
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
            let Some(&(t, inc, len, end)) = self.tokens.get(self.upto) else {
                return Ok(false);
            };
            self.upto += 1;
            self.atts.set_term(t);
            self.atts.set_position_increment(inc)?;
            self.atts.set_position_length(len)?;
            self.atts.set_offset(0, end)?;
            Ok(true)
        }
        fn reset(&mut self) -> Result<(), AnalysisError> {
            self.upto = 0;
            Ok(())
        }
        fn end(&mut self) -> Result<(), AnalysisError> {
            self.atts.end_attributes();
            self.atts.set_position_increment(self.end.0)?;
            self.atts.set_offset(self.end.1, self.end.1)
        }
    }

    fn canned(tokens: Vec<(&'static str, i32, i32, i32)>, end: (i32, i32)) -> Canned {
        Canned {
            atts: AttributeSource::new(),
            tokens,
            upto: 0,
            end,
        }
    }

    /// Every accepted string, labels as chars (small acyclic automata only).
    fn strings(a: &Automaton) -> Vec<String> {
        fn walk(a: &Automaton, s: usize, cur: &mut String, out: &mut Vec<String>) {
            if a.is_accept(s) {
                out.push(cur.clone());
            }
            for t in a.transitions(s) {
                for l in t.min..=t.max {
                    let c = match l {
                        POS_SEP => '|',
                        HOLE => '_',
                        l => char::from_u32(l as u32).unwrap(),
                    };
                    cur.push(c);
                    walk(a, t.dest, cur, out);
                    cur.pop();
                }
            }
        }
        let mut out = Vec::new();
        walk(a, 0, &mut String::new(), &mut out);
        out.sort();
        out
    }

    #[test]
    fn linear_stream() {
        let mut ts = canned(vec![("ab", 1, 1, 2), ("c", 1, 1, 4)], (0, 4));
        let a = token_stream_to_automaton(&mut ts).unwrap();
        assert_eq!(strings(&a), vec!["ab|c"]);
    }

    #[test]
    fn synonyms_holes_and_trailing_holes() {
        // "wifi" spanning "wi fi", then a hole (stopword), then "x", then two
        // trailing removed positions.
        let mut ts = canned(
            vec![
                ("wifi", 1, 2, 5),
                ("wi", 0, 1, 2),
                ("fi", 1, 1, 5),
                ("x", 2, 1, 9),
            ],
            (2, 12),
        );
        let a = token_stream_to_automaton(&mut ts).unwrap();
        assert_eq!(strings(&a), vec!["wifi|_|x|_|_", "wi|fi|_|x|_|_"]);
        let mut conv = TokenStreamToAutomaton::default();
        conv.set_preserve_position_increments(false);
        let a = conv.to_automaton(&mut ts).unwrap();
        assert_eq!(strings(&a), vec!["wifi|x", "wi|fi|x"]);
    }

    #[test]
    fn final_offset_gap_unicode_arcs_and_change_token() {
        let mut ts = canned(vec![("é", 1, 1, 1)], (0, 5));
        let a = token_stream_to_automaton(&mut ts).unwrap();
        // two UTF-8 byte arcs
        assert_eq!(a.num_states(), 3);
        let mut conv = TokenStreamToAutomaton::new();
        conv.set_unicode_arcs(true);
        conv.set_final_offset_gap_as_hole(true);
        let a = conv.to_automaton(&mut ts).unwrap();
        assert_eq!(strings(&a), vec!["é|_"]);
        let mut conv = TokenStreamToAutomaton::new();
        conv.set_change_token(|b| b.iter().map(|&x| x.to_ascii_uppercase()).collect());
        let mut ts = canned(vec![("ab", 1, 1, 1)], (0, 2));
        let a = conv.to_automaton(&mut ts).unwrap();
        assert_eq!(strings(&a), vec!["AB"]);
        let mut ts = canned(vec![("a", 0, 1, 1)], (0, 2));
        assert!(conv.to_automaton(&mut ts).is_err());
    }

    #[test]
    fn builder_merges_adjacent_ranges_and_sorts() {
        let mut b = AutomatonBuilder::new();
        let s0 = b.create_state();
        let s1 = b.create_state();
        let s2 = b.create_state();
        b.add_transition(s0, s1, b'c' as i32);
        b.add_transition(s0, s1, b'a' as i32);
        b.add_transition(s0, s1, b'b' as i32);
        b.add_transition(s0, s2, b'a' as i32);
        b.add_transition_range(s0, s1, b'x' as i32, b'z' as i32);
        b.set_accept(s1, true);
        assert!(b.is_accept(s1));
        assert_eq!(b.num_states(), 3);
        let a = b.finish();
        assert_eq!(
            a.transitions(s0),
            &[
                Transition {
                    dest: 2,
                    min: 97,
                    max: 97
                },
                Transition {
                    dest: 1,
                    min: 97,
                    max: 99
                },
                Transition {
                    dest: 1,
                    min: 120,
                    max: 122
                },
            ]
        );
        assert_eq!(a.num_transitions(), 3);
    }

    #[test]
    fn automaton_round_trips_to_a_token_graph() {
        // 0 -a-> 1 -b-> 2(accept); 0 -c-> 2
        let mut b = AutomatonBuilder::new();
        for _ in 0..3 {
            b.create_state();
        }
        b.add_transition(0, 1, b'a' as i32);
        b.add_transition(1, 2, b'b' as i32);
        b.add_transition(0, 2, b'c' as i32);
        b.set_accept(2, true);
        let a = b.finish();
        let mut ts = automaton_to_token_stream(&a).unwrap();
        let mut toks = Vec::new();
        let end = consume(&mut ts, |x| {
            toks.push((
                x.term().to_string(),
                x.position_increment(),
                x.position_length(),
                x.start_offset(),
                x.end_offset(),
            ))
        })
        .unwrap();
        assert_eq!(
            toks,
            vec![
                ("a".to_string(), 1, 1, 0, 1),
                ("c".to_string(), 0, 2, 0, 2),
                ("b".to_string(), 1, 1, 1, 2),
            ]
        );
        assert_eq!((end.start_offset(), end.position_increment()), (2, 0));
    }

    #[test]
    fn cycles_are_rejected() {
        let mut b = AutomatonBuilder::new();
        b.create_state();
        b.create_state();
        b.add_transition(0, 1, 1);
        b.add_transition(1, 0, 1);
        let err = automaton_to_token_stream(&b.finish()).err().unwrap();
        assert!(err.to_string().contains("Start node has incoming edges"));
        let mut b = AutomatonBuilder::new();
        for _ in 0..3 {
            b.create_state();
        }
        b.add_transition(0, 1, 1);
        b.add_transition(1, 2, 1);
        b.add_transition(2, 1, 1);
        let err = automaton_to_token_stream(&b.finish()).err().unwrap();
        assert!(err.to_string().contains("Cycle found"));
        // intermediate accept adds an edge to the terminal layer
        let mut b = AutomatonBuilder::new();
        for _ in 0..3 {
            b.create_state();
        }
        b.add_transition(0, 1, b'a' as i32);
        b.add_transition(1, 2, b'b' as i32);
        b.set_accept(1, true);
        b.set_accept(2, true);
        let mut ts = automaton_to_token_stream(&b.finish()).unwrap();
        let mut n = 0;
        consume(&mut ts, |_| n += 1).unwrap();
        assert_eq!(n, 3);
        assert!(automaton_to_token_stream(&Automaton::default()).is_ok());
    }
}
