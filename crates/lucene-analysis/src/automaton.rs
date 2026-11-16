//! `TokenStreamToAutomaton` and `AutomatonToTokenStream`.
//!
//! Both convert to and from `org.apache.lucene.util.automaton.Automaton`,
//! which is [`lucene_util::automaton::Automaton`] here (the full port; this
//! crate's only workspace dependency). Java's `Automaton.Builder` is
//! [`AutomatonBuilder`]; its `finish()` sorts and merges each state's
//! transitions exactly as Java's does, and the differential fixture compares
//! the result state for state with Java's `getSortedTransitions()`.

use crate::attributes::AttributeSource;
use crate::token_stream::TokenStream;
use crate::AnalysisError;
pub use lucene_util::automaton::{Automaton, Builder as AutomatonBuilder, Transition};

/// `TokenStreamToAutomaton.POS_SEP`: the label between positions.
pub const POS_SEP: i32 = 0x001f;

/// `TokenStreamToAutomaton.HOLE`: the label of a position no token covers.
pub const HOLE: i32 = 0x001e;

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
            // ALLOC: `pos` is a position the token stream reached, one per
            // token at most -- bounded by the input, not by configuration.
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
                        positions.get(pos).leaving = i64::from(builder.create_state());
                        add_holes(&mut builder, &mut positions, pos);
                    }
                } else {
                    let leaving = builder.create_state();
                    positions.get(pos).leaving = i64::from(leaving);
                    builder.add_transition_label(arriving as i32, leaving, POS_SEP);
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
                    end_pos_data.arriving = i64::from(builder.create_state());
                }
                end_pos_data.arriving as i32
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

            let mut state = leaving_state as i32;
            let term_len = labels.len();
            for (i, &c) in labels.iter().enumerate() {
                let next_state = if i == term_len - 1 {
                    end_arriving
                } else {
                    builder.create_state()
                };
                builder.add_transition_label(state, next_state, c);
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
                builder.add_transition_label(last_state, state1, HOLE);
                end_pos_inc -= 1;
                if end_pos_inc == 0 {
                    builder.set_accept(state1, true);
                    break;
                }
                let state2 = builder.create_state();
                builder.add_transition_label(state1, state2, POS_SEP);
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
                        builder.add_transition_label(arriving as i32, end_state, POS_SEP)
                    }
                    None => builder.set_accept(arriving as i32, true),
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
            positions.get(pos).arriving = i64::from(arriving);
            builder.add_transition_label(arriving, pos_data.leaving as i32, POS_SEP);
        }
        if prev.leaving == -1 {
            let leaving = if pos == 1 { 0 } else { builder.create_state() };
            positions.get(pos - 1).leaving = i64::from(leaving);
            if prev.arriving != -1 {
                builder.add_transition_label(prev.arriving as i32, leaving, POS_SEP);
            }
        }
        let from = positions.get(pos - 1).leaving as i32;
        let to = positions.get(pos).arriving as i32;
        builder.add_transition_label(from, to, HOLE);
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
    let n = usize::try_from(automaton.get_num_states()).unwrap_or(0);
    let transitions: Vec<Vec<Transition>> = automaton.get_sorted_transitions();
    let mut position_nodes: Vec<Vec<usize>> = Vec::new();

    let mut indegree = vec![0i64; n];
    for ts in &transitions {
        for t in ts {
            indegree[t.dest as usize] += 1;
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
        for t in &transitions[id] {
            let dest = t.dest as usize;
            indegree[dest] -= 1;
            if indegree[dest] == 0 {
                no_incoming_edges.push_back((dest, pos + 1));
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
            for t in &transitions[state] {
                // each edge in the token stream can only be one value,
                // though a transition takes a range.
                for val in t.min..=t.max {
                    let dest_layer = id_to_pos[t.dest as usize];
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
    /// A source, not a wrapper: no conditional wrapper below it.
    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
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
mod tests;
