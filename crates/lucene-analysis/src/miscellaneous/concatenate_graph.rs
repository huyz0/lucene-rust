//! `org.apache.lucene.analysis.miscellaneous.ConcatenateGraphFilter`: every
//! path through the input token graph, its terms joined by a separator, as
//! one token each (what completion suggesters index).
//!
//! The filter is its own `TokenStream` (Java's does not share its input's
//! attributes), so the output's attributes are its own cleared set plus the
//! term, offsets and increment it writes. Java's lazy
//! `LimitedFiniteStringsIterator` is drained into a list on the first
//! `incrementToken` (the iterator borrows the automaton); the strings and
//! their order are the same.

use lucene_util::automaton::{
    operations, Automaton, LimitedFiniteStringsIterator, Transition, TransitionAccessor,
    DEFAULT_DETERMINIZE_WORK_LIMIT,
};

use crate::attributes::AttributeSource;
use crate::automaton::{TokenStreamToAutomaton, HOLE, POS_SEP};
use crate::token_stream::{TokenStream, Tokenizer};
use crate::AnalysisError;

/// `ConcatenateGraphFilter.SEP_LABEL` (`TokenStreamToAutomaton.POS_SEP`).
pub const SEP_LABEL: char = '\u{1F}';
/// `ConcatenateGraphFilter.DEFAULT_MAX_GRAPH_EXPANSIONS`.
pub const DEFAULT_MAX_GRAPH_EXPANSIONS: i32 = DEFAULT_DETERMINIZE_WORK_LIMIT;

fn illegal(e: impl std::fmt::Display) -> AnalysisError {
    AnalysisError::IllegalArgument(e.to_string())
}

/// `org.apache.lucene.analysis.miscellaneous.ConcatenateGraphFilter`.
pub struct ConcatenateGraphFilter<I> {
    atts: AttributeSource,
    input: I,
    token_separator: Option<char>,
    preserve_position_increments: bool,
    max_graph_expansions: i32,
    strings: Option<Vec<Vec<i32>>>,
    next: usize,
    was_reset: bool,
    end_offset: i32,
}

impl<I: TokenStream> ConcatenateGraphFilter<I> {
    /// `new ConcatenateGraphFilter(TokenStream)`: separator [`SEP_LABEL`],
    /// position increments preserved, [`DEFAULT_MAX_GRAPH_EXPANSIONS`].
    pub fn new(input: I) -> Self {
        Self::with_options(input, Some(SEP_LABEL), true, DEFAULT_MAX_GRAPH_EXPANSIONS)
    }

    /// `new ConcatenateGraphFilter(TokenStream, Character tokenSeparator,
    /// boolean preservePositionIncrements, int maxGraphExpansions)`.
    pub fn with_options(
        input: I,
        token_separator: Option<char>,
        preserve_position_increments: bool,
        max_graph_expansions: i32,
    ) -> Self {
        ConcatenateGraphFilter {
            atts: AttributeSource::new(),
            input,
            token_separator,
            preserve_position_increments,
            max_graph_expansions,
            strings: None,
            next: 0,
            was_reset: false,
            end_offset: -1,
        }
    }

    /// `toAutomaton()`: the determinized graph, separators replaced.
    pub fn to_automaton(&mut self) -> Result<Automaton, AnalysisError> {
        let mut tsta = TokenStreamToAutomaton::new();
        if let Some(sep) = self.token_separator {
            // Java: EscapingTokenStreamToAutomaton doubles a separator byte.
            let sep_label = u8::try_from(u32::from(sep)).map_err(illegal)?;
            tsta.set_change_token(move |bytes: &[u8]| {
                let mut out = Vec::with_capacity(bytes.len());
                for &b in bytes {
                    if b == sep_label {
                        out.push(sep_label);
                    }
                    out.push(b);
                }
                out
            });
        }
        tsta.set_preserve_position_increments(self.preserve_position_increments);
        tsta.set_unicode_arcs(false);
        let automaton = tsta.to_automaton(&mut self.input)?;
        let automaton = replace_sep(&automaton, self.token_separator)?;
        operations::determinize(&automaton, self.max_graph_expansions).map_err(illegal)
    }
}

/// `ConcatenateGraphFilter.replaceSep`: `POS_SEP` arcs become the separator
/// (or epsilons), `HOLE` arcs epsilons.
fn replace_sep(a: &Automaton, sep: Option<char>) -> Result<Automaton, AnalysisError> {
    let mut result = Automaton::new();
    let n = a.get_num_states();
    for s in 0..n {
        result.create_state();
        result.set_accept(s, a.is_accept(s));
    }
    let topo = operations::topo_sort_states(a).map_err(illegal)?;
    let mut t = Transition::new();
    for &state in topo.iter().rev() {
        let count = a.init_transition(state, &mut t);
        for _ in 0..count {
            a.get_next_transition(&mut t);
            if t.min == POS_SEP {
                match sep {
                    Some(c) => result.add_transition_label(state, t.dest, c as i32),
                    None => result.add_epsilon(state, t.dest),
                }
            } else if t.min == HOLE {
                result.add_epsilon(state, t.dest);
            } else {
                result.add_transition(state, t.dest, t.min, t.max);
            }
        }
    }
    result.finish_state();
    Ok(result)
}

impl<I: TokenStream> TokenStream for ConcatenateGraphFilter<I> {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    // Java: ConcatenateGraphFilter.incrementToken
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        if self.strings.is_none() {
            if !self.was_reset {
                return Err(AnalysisError::IllegalState(
                    "reset() missing before incrementToken".into(),
                ));
            }
            let automaton = self.to_automaton()?;
            let mut it = LimitedFiniteStringsIterator::new(&automaton, self.max_graph_expansions)
                .map_err(illegal)?;
            let mut all = Vec::new();
            while let Some(s) = it.next_string().map_err(illegal)? {
                all.push(s);
            }
            self.strings = Some(all);
            self.next = 0;
            self.end_offset = self.input.attributes().end_offset();
        }
        let strings = self.strings.as_ref().expect("filled above");
        let Some(string) = strings.get(self.next) else {
            return Ok(false);
        };
        // Util.toBytesRef: each label is a byte.
        let bytes: Vec<u8> = string.iter().map(|&l| l as u8).collect();
        self.next += 1;
        self.atts.clear_attributes();
        if self.next > 1 {
            self.atts.set_position_increment(0)?;
        }
        self.atts.set_offset(0, self.end_offset)?;
        self.atts.set_term(&String::from_utf8_lossy(&bytes));
        Ok(true)
    }

    // Java: ConcatenateGraphFilter.reset
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.was_reset = true;
        Ok(())
    }

    // Java: ConcatenateGraphFilter.end
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        if self.strings.is_none() {
            self.input.end()?;
        }
        if self.end_offset != -1 {
            self.atts.set_offset(0, self.end_offset)?;
        }
        Ok(())
    }

    // Java: ConcatenateGraphFilter.close
    fn close(&mut self) -> Result<(), AnalysisError> {
        self.input.close()?;
        self.strings = None;
        self.was_reset = false;
        self.end_offset = -1;
        Ok(())
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        self.input.as_tokenizer()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::{render, Canned};

    #[test]
    fn concatenates_every_path() {
        let mut f = ConcatenateGraphFilter::new(Canned::parse("a:0:1:1:1 b:2:3:1:1|5|0"));
        assert_eq!(render(&mut f), "a\u{1F}b:0:5:1:1|5|0");
        let mut f = ConcatenateGraphFilter::new(Canned::parse("a:0:1:1:1 x:0:1:0:1 b:2:3:1:1|5|0"));
        assert_eq!(render(&mut f), "a\u{1F}b:0:5:1:1 x\u{1F}b:0:5:0:1|5|0");
        let mut f = ConcatenateGraphFilter::with_options(
            Canned::parse("a:0:1:1:1 b:2:3:2:1|5|0"),
            None,
            true,
            100,
        );
        assert_eq!(render(&mut f), "ab:0:5:1:1|5|0");
        let mut c = Canned::parse("x:0:1:1:1");
        c.set_terms(&["a\u{1F}b"]);
        let mut f = ConcatenateGraphFilter::new(c);
        assert_eq!(render(&mut f), "a\u{1F}\u{1F}b:0:0:1:1|0|0");
    }

    #[test]
    fn contract() {
        let mut f = ConcatenateGraphFilter::new(Canned::parse("a:0:1:1:1|1|0"));
        assert!(f.increment_token().is_err(), "reset() missing");
        f.reset().unwrap();
        f.end().unwrap();
        f.close().unwrap();
        let mut f = ConcatenateGraphFilter::with_options(
            Canned::parse("a:0:1:1:1|1|0"),
            Some('😀'),
            true,
            10,
        );
        f.reset().unwrap();
        assert!(f.increment_token().is_err(), "a separator above a byte");
    }
}
