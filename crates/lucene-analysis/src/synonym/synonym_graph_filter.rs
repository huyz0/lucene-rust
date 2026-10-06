//! `org.apache.lucene.analysis.synonym.SynonymGraphFilter`: applies a
//! [`SynonymMap`] and emits a correct token graph -- a multi-word synonym is
//! a side path of its own positions, `positionLength` spanning the input it
//! replaces. Follow it with `FlattenGraphFilter` to index the graph.
//!
//! Ported state for state: a lookahead [`RollingBuffer`] of captured input
//! tokens, the longest match found by walking the map one code point at a
//! time ([`SynonymMap::step`], Java's `fst.findTargetArc`), and an output
//! buffer of `(state | term, startNode, endNode)` tokens.
//!
//! Differs: the term is walked by Rust `char`s, so a term holding an
//! unpaired surrogate (which a Rust term holds as U+FFFD) matches as U+FFFD.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::attributes::State;
use crate::java_character::to_lower_case;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::util::rolling_buffer::{Resettable, RollingBuffer};
use crate::AnalysisError;

use super::synonym_map::{NodeId, SynonymMap, WORD_SEPARATOR};

/// `SynonymGraphFilter.TYPE_SYNONYM`.
pub const TYPE_SYNONYM: &str = "SYNONYM";

/// `SynonymGraphFilter.BufferedInputToken`.
#[derive(Debug, Default)]
struct BufferedInputToken {
    term: String,
    state: Option<State>,
    start_offset: i32,
    end_offset: i32,
}

impl Resettable for BufferedInputToken {
    fn reset(&mut self) {
        self.state = None;
        self.term.clear();
        self.start_offset = -1;
        self.end_offset = -1;
    }
}

/// `SynonymGraphFilter.BufferedOutputToken`: `state` is set for an input
/// token, `term` for a synonym.
#[derive(Debug)]
struct BufferedOutputToken {
    state: Option<State>,
    term: String,
    start_node: i32,
    end_node: i32,
}

/// Walks `term`'s code points from `node` (`Character.toLowerCase`d when
/// `ignore_case`); `None` when the map has no such path.
#[inline]
pub(crate) fn walk_term(
    map: &SynonymMap,
    mut node: NodeId,
    term: &str,
    ignore_case: bool,
) -> Option<NodeId> {
    for c in term.chars() {
        let cp = c as u32;
        let label = if ignore_case { to_lower_case(cp) } else { cp };
        node = map.step(node, label)?;
    }
    Some(node)
}

/// `SynonymGraphFilter`.
pub struct SynonymGraphFilter<I> {
    input: I,
    synonyms: Arc<SynonymMap>,
    ignore_case: bool,
    output_buffer: VecDeque<BufferedOutputToken>,
    next_node_out: i32,
    last_node_out: i32,
    max_lookahead_used: usize,
    capture_count: usize,
    live_token: bool,
    match_start_offset: i32,
    match_end_offset: i32,
    finished: bool,
    lookahead_next_read: i32,
    lookahead_next_write: i32,
    lookahead: RollingBuffer<BufferedInputToken>,
}

impl<I: TokenStream> SynonymGraphFilter<I> {
    /// `new SynonymGraphFilter(input, synonyms, ignoreCase)`: `ignore_case`
    /// lowercases the input (`Character.toLowerCase`) before matching; the
    /// map's keys must then be lowercase.
    pub fn new(input: I, synonyms: Arc<SynonymMap>, ignore_case: bool) -> Self {
        SynonymGraphFilter {
            input,
            synonyms,
            ignore_case,
            output_buffer: VecDeque::new(),
            next_node_out: 0,
            last_node_out: 0,
            max_lookahead_used: 0,
            capture_count: 0,
            live_token: false,
            match_start_offset: 0,
            match_end_offset: 0,
            finished: false,
            lookahead_next_read: 0,
            lookahead_next_write: 0,
            lookahead: RollingBuffer::new(),
        }
    }

    /// `getCaptureCount()`.
    pub fn capture_count(&self) -> usize {
        self.capture_count
    }

    /// `getMaxLookaheadUsed()`.
    pub fn max_lookahead_used(&self) -> usize {
        self.max_lookahead_used
    }

    // Java: releaseBufferedToken
    fn release_buffered_token(&mut self) -> Result<(), AnalysisError> {
        let token = self
            .output_buffer
            .pop_front()
            .expect("output buffer is non-empty");
        let (start, end) = (self.match_start_offset, self.match_end_offset);
        let a = self.input.attributes_mut();
        match &token.state {
            Some(state) => a.restore_state(state),
            None => {
                a.clear_attributes();
                a.set_term(&token.term);
                a.set_offset(start, end)?;
                a.set_token_type(TYPE_SYNONYM);
            }
        }
        a.set_position_increment(token.start_node - self.last_node_out)?;
        self.last_node_out = token.start_node;
        a.set_position_length(token.end_node - token.start_node)
    }

    // Java: parse
    fn parse(&mut self) -> Result<bool, AnalysisError> {
        let map = Arc::clone(&self.synonyms);
        let mut match_output: Option<NodeId> = None;
        let mut match_input_length = 0;
        let mut node = map.root();
        let mut match_length = 0;
        let mut do_final_capture = false;
        let mut lookahead_upto = self.lookahead_next_read;
        self.match_start_offset = -1;

        loop {
            let input_end_offset;
            let walked = if lookahead_upto <= self.lookahead.max_pos() {
                let token = self.lookahead.get(lookahead_upto);
                lookahead_upto += 1;
                input_end_offset = token.end_offset;
                if self.match_start_offset == -1 {
                    self.match_start_offset = token.start_offset;
                }
                walk_term(&map, node, &token.term, self.ignore_case)
            } else {
                debug_assert!(self.finished || !self.live_token);
                if self.finished {
                    break;
                } else if self.input.increment_token()? {
                    self.live_token = true;
                    let a = self.input.attributes();
                    if self.match_start_offset == -1 {
                        self.match_start_offset = a.start_offset();
                    }
                    input_end_offset = a.end_offset();
                    lookahead_upto += 1;
                    walk_term(&map, node, a.term(), self.ignore_case)
                } else {
                    self.finished = true;
                    break;
                }
            };
            match_length += 1;
            let Some(after) = walked else {
                break;
            };
            node = after;
            if map.entry(node).is_some() {
                match_output = Some(node);
                match_input_length = match_length;
                self.match_end_offset = input_end_offset;
            }
            match map.step(node, WORD_SEPARATOR as u32) {
                None => break,
                Some(next) => {
                    node = next;
                    do_final_capture = true;
                    if self.live_token {
                        self.capture();
                    }
                }
            }
        }

        if do_final_capture && self.live_token && !self.finished {
            self.capture();
        }

        match match_output {
            Some(found) => {
                if self.live_token {
                    self.capture();
                }
                self.buffer_output_tokens(&map, found, match_input_length);
                self.lookahead_next_read += match_input_length;
                self.lookahead.free_before(self.lookahead_next_read);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    // Java: bufferOutputTokens
    fn buffer_output_tokens(&mut self, map: &SynonymMap, found: NodeId, match_input_length: i32) {
        let entry = map.entry(found).expect("a final node");
        let keep_orig = entry.keep_orig;
        let mut total_path_nodes = if keep_orig { match_input_length - 1 } else { 0 };

        let mut paths: Vec<Vec<&str>> = Vec::with_capacity(entry.ords.len());
        for &ord in &entry.ords {
            let path: Vec<&str> = map.word(ord).split(WORD_SEPARATOR).collect();
            total_path_nodes += path.len() as i32 - 1;
            paths.push(path);
        }

        let start_node = self.next_node_out;
        let end_node = start_node + total_path_nodes + 1;
        let mut new_node_count = 0;
        for path in &paths {
            let path_end_node = if path.len() == 1 {
                end_node
            } else {
                let n = self.next_node_out + new_node_count + 1;
                new_node_count += path.len() as i32 - 1;
                n
            };
            self.output_buffer.push_back(BufferedOutputToken {
                state: None,
                term: path[0].to_string(),
                start_node,
                end_node: path_end_node,
            });
        }

        if keep_orig {
            let input_end_node = if match_input_length == 1 {
                end_node
            } else {
                self.next_node_out + new_node_count + 1
            };
            let token = self.lookahead.get(self.lookahead_next_read);
            self.output_buffer.push_back(BufferedOutputToken {
                state: token.state.clone(),
                term: token.term.clone(),
                start_node,
                end_node: input_end_node,
            });
        }

        self.next_node_out = end_node;

        for (path_id, path) in paths.iter().enumerate() {
            if path.len() > 1 {
                let mut last_node = self.output_buffer[path_id].end_node;
                for word in &path[1..path.len() - 1] {
                    self.output_buffer.push_back(BufferedOutputToken {
                        state: None,
                        term: word.to_string(),
                        start_node: last_node,
                        end_node: last_node + 1,
                    });
                    last_node += 1;
                }
                self.output_buffer.push_back(BufferedOutputToken {
                    state: None,
                    term: path[path.len() - 1].to_string(),
                    start_node: last_node,
                    end_node,
                });
            }
        }

        if keep_orig && match_input_length > 1 {
            let first = self.output_buffer[paths.len()].end_node;
            for i in 1..match_input_length {
                let last_node = first + i - 1;
                let token = self.lookahead.get(self.lookahead_next_read + i);
                let end = if i == match_input_length - 1 {
                    end_node
                } else {
                    last_node + 1
                };
                self.output_buffer.push_back(BufferedOutputToken {
                    state: token.state.clone(),
                    term: token.term.clone(),
                    start_node: last_node,
                    end_node: end,
                });
            }
        }
    }

    // Java: capture
    fn capture(&mut self) {
        debug_assert!(self.live_token);
        self.live_token = false;
        let a = self.input.attributes();
        let token = self.lookahead.get(self.lookahead_next_write);
        self.lookahead_next_write += 1;
        token.state = Some(a.capture_state());
        token.start_offset = a.start_offset();
        token.end_offset = a.end_offset();
        debug_assert!(token.term.is_empty());
        token.term.push_str(a.term());
        self.capture_count += 1;
        self.max_lookahead_used = self.max_lookahead_used.max(self.lookahead.buffer_size());
    }
}

impl<I: TokenStream> TokenFilter for SynonymGraphFilter<I> {
    crate::filter_input!();

    // Java: SynonymGraphFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.output_buffer.is_empty() {
            self.release_buffered_token()?;
            return Ok(true);
        }
        if self.parse()? {
            self.release_buffered_token()?;
            return Ok(true);
        }
        if self.lookahead_next_read == self.lookahead_next_write {
            if self.finished {
                return Ok(false);
            }
            self.live_token = false;
        } else {
            let state = self.lookahead.get(self.lookahead_next_read).state.take();
            self.lookahead_next_read += 1;
            if let Some(state) = state {
                self.input.attributes_mut().restore_state(&state);
            }
            self.lookahead.free_before(self.lookahead_next_read);
        }
        let a = self.input.attributes();
        self.last_node_out += a.position_increment();
        self.next_node_out = self.last_node_out + a.position_length();
        Ok(true)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.lookahead.reset();
        self.lookahead_next_write = 0;
        self.lookahead_next_read = 0;
        self.capture_count = 0;
        self.last_node_out = -1;
        self.next_node_out = 0;
        self.match_start_offset = -1;
        self.match_end_offset = -1;
        self.finished = false;
        self.live_token = false;
        self.output_buffer.clear();
        self.max_lookahead_used = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synonym::SynonymMapBuilder;
    use crate::util::canned::{render, Canned};

    fn map(rules: &[(&str, &str, bool)]) -> Arc<SynonymMap> {
        let mut b = SynonymMapBuilder::default();
        for (i, o, keep) in rules {
            b.add(i, o, *keep).unwrap();
        }
        Arc::new(b.build().unwrap())
    }

    fn run(spec: &str, terms: &[&str], m: &Arc<SynonymMap>, ignore_case: bool) -> String {
        let mut c = Canned::parse(spec);
        c.set_terms(terms);
        let mut f = SynonymGraphFilter::new(c, Arc::clone(m), ignore_case);
        let out = render(&mut f);
        // Reuse after reset gives the same output.
        assert_eq!(render(&mut f), out);
        out
    }

    // Expected outputs are what Lucene 10.5.0's filter gave for each case.
    #[test]
    fn multi_word_synonym_is_a_side_path() {
        let m = map(&[("wi\0fi", "wifi", true), ("wifi", "wi\0fi", true)]);
        assert_eq!(
            run(
                "a:0:1:1:1 wi:2:4:1:1 fi:5:7:1:1 b:8:9:1:1|9|0",
                &["a", "wi", "fi", "b"],
                &m,
                false
            ),
            "a:0:1:1:1 wifi:2:7:1:2 wi:2:4:0:1 fi:5:7:1:1 b:8:9:1:1|9|0"
        );
        assert_eq!(
            run("wifi:0:4:1:1|4|0", &["wifi"], &m, false),
            "wi:0:4:1:1 wifi:0:4:0:2 fi:0:4:1:1|4|0"
        );
    }

    #[test]
    fn longest_match_wins_and_partial_matches_replay() {
        let m = map(&[("a", "x", false), ("a\0b\0c", "y", false)]);
        assert_eq!(
            run(
                "a:0:1:1:1 b:2:3:1:1 d:4:5:1:1|5|0",
                &["a", "b", "d"],
                &m,
                false
            ),
            "x:0:1:1:1 b:2:3:1:1 d:4:5:1:1|5|0"
        );
        assert_eq!(
            run(
                "a:0:1:1:1 b:2:3:1:1 c:4:5:1:1|5|0",
                &["a", "b", "c"],
                &m,
                false
            ),
            "y:0:5:1:1|5|0"
        );
    }

    #[test]
    fn ignore_case_lowercases_the_input() {
        let m = map(&[("dog", "hound", true)]);
        assert_eq!(
            run("DOG:0:3:1:1|3|0", &["DOG"], &m, true),
            "hound:0:3:1:1 DOG:0:3:0:1|3|0"
        );
        assert_eq!(
            run("DOG:0:3:1:1|3|0", &["DOG"], &m, false),
            "DOG:0:3:1:1|3|0"
        );
    }

    #[test]
    fn keep_orig_over_several_input_words() {
        let m = map(&[("a\0b\0c", "x\0y", true)]);
        let mut c = Canned::parse("a:0:1:1:1 b:2:3:1:1 c:4:5:1:1|5|0");
        c.set_terms(&["a", "b", "c"]);
        let mut f = SynonymGraphFilter::new(c, m, false);
        assert_eq!(
            render(&mut f),
            "x:0:5:1:1 a:0:1:0:2 y:0:5:1:3 b:2:3:1:1 c:4:5:1:1|5|0"
        );
        assert!(f.capture_count() >= 3);
        assert!(f.max_lookahead_used() >= 3);
    }
}
