//! `org.apache.lucene.analysis.miscellaneous.ConcatenateGraphFilter`: every
//! path through the input token graph, its terms joined by a separator, as
//! one token each (what completion suggesters index).
//!
//! The filter is its own `TokenStream` (Java's does not share its input's
//! attributes), so the output's attributes are its own cleared set plus the
//! term, offsets and increment it writes. The strings are produced one per
//! `incrementToken`, as Java's `LimitedFiniteStringsIterator` produces them:
//! the filter keeps the automaton and a `LimitedFiniteStringsCursor` over it
//! (up to `maxGraphExpansions` paths, each as long as the input, are never
//! held at once).
//!
//! Each label becomes one byte (`Util.toBytesRef`'s `(byte)` cast), so a
//! separator above U+007F reaches the index as its low byte, as in Java
//! (which also escapes term bytes equal to `(byte) separator`). Where those
//! bytes are not UTF-8 the token carries them as its binary term
//! ([`AttributeSource::set_bytes_term`], Java's
//! `BytesRefBuilderTermAttribute`); Java's `CharTermAttribute` copy of them is
//! a lenient decode no index reads.

use lucene_util::automaton::{
    operations, Automaton, LimitedFiniteStringsCursor, Transition, TransitionAccessor,
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
    /// The determinized graph and the cursor over its finite strings.
    strings: Option<(Automaton, LimitedFiniteStringsCursor)>,
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
            was_reset: false,
            end_offset: -1,
        }
    }

    /// `toAutomaton()`: the determinized graph, separators replaced.
    pub fn to_automaton(&mut self) -> Result<Automaton, AnalysisError> {
        let mut tsta = TokenStreamToAutomaton::new();
        if let Some(sep) = self.token_separator {
            // Java: EscapingTokenStreamToAutomaton doubles a separator byte;
            // `(byte) sepLabel` keeps the low byte.
            let sep_label = (u32::from(sep) & 0xFF) as u8;
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
            let cursor = LimitedFiniteStringsCursor::new(&automaton, self.max_graph_expansions)
                .map_err(illegal)?;
            self.strings = Some((automaton, cursor));
            self.end_offset = self.input.attributes().end_offset();
        }
        let (automaton, cursor) = self.strings.as_mut().expect("filled above");
        let Some(string) = cursor.next_string(automaton).map_err(illegal)? else {
            return Ok(false);
        };
        // Util.toBytesRef: each label is a byte.
        let bytes: Vec<u8> = string.iter().map(|&l| (l & 0xFF) as u8).collect();
        let stacked = cursor.size() > 1;
        self.atts.clear_attributes();
        if stacked {
            self.atts.set_position_increment(0)?;
        }
        self.atts.set_offset(0, self.end_offset)?;
        match String::from_utf8(bytes) {
            Ok(term) => self.atts.set_term(&term),
            Err(e) => {
                self.atts.set_term(&String::from_utf8_lossy(e.as_bytes()));
                self.atts.set_bytes_term(Some(e.into_bytes()));
            }
        }
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

    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        self.input.conditional_root()
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

    /// Java casts the separator to a byte (`(byte) sepLabel` when escaping,
    /// `Util.toBytesRef`'s `(byte)` per label), so a separator above U+007F
    /// reaches the index as that raw byte; Lucene 10.5.0 gives `61 e9 62`
    /// for `é` and `61 00 62` for U+0100 over `a b`.
    #[test]
    fn separators_above_ascii_are_raw_bytes() {
        let run = |sep: char, spec: &str| -> Vec<Vec<u8>> {
            let mut f =
                ConcatenateGraphFilter::with_options(Canned::parse(spec), Some(sep), true, 100);
            let mut out = Vec::new();
            crate::token_stream::consume(&mut f, |a| out.push(a.term_bytes().to_vec())).unwrap();
            out
        };
        let ab = "a:0:1:1:1 b:2:3:1:1|3|0";
        assert_eq!(run('\u{e9}', ab), vec![vec![0x61, 0xE9, 0x62]]);
        assert_eq!(run('\u{ff}', ab), vec![vec![0x61, 0xFF, 0x62]]);
        assert_eq!(run('\u{100}', ab), vec![vec![0x61, 0x00, 0x62]]);
        assert_eq!(run('\u{141}', ab), vec![vec![0x61, 0x41, 0x62]]);
        assert_eq!(run('\u{1F}', ab), vec![vec![0x61, 0x1F, 0x62]]);
        // A term byte equal to the separator byte is doubled.
        let mut c = Canned::parse("x:0:1:1:1");
        c.set_terms(&["\u{e9}"]);
        let mut f = ConcatenateGraphFilter::with_options(c, Some('\u{a9}'), true, 100);
        let mut out = Vec::new();
        crate::token_stream::consume(&mut f, |a| out.push(a.term_bytes().to_vec())).unwrap();
        assert_eq!(out, vec![vec![0xC3, 0xA9, 0xA9]]);
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
        assert!(f.increment_token().unwrap(), "the separator's low byte");
        assert_eq!(f.attributes().term_bytes(), b"a");
        assert!(!f.increment_token().unwrap());
    }
}
