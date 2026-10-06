//! `org.apache.lucene.analysis.synonym.SynonymFilter` (deprecated in Lucene
//! for [`SynonymGraphFilter`](super::SynonymGraphFilter)): applies a
//! [`SynonymMap`] by stacking each output word on the input position it
//! lines up with, so a multi-word synonym's later words overlap later input
//! words (an incorrect graph, kept for compatibility).
//!
//! Ported state for state: two rolling arrays of `1 + maxHorizontalContext`
//! slots, `futureInputs` (captured input tokens, their keep/matched flags)
//! and `futureOutputs` (the output words pending at each position).
//!
//! Differs: as in [`SynonymGraphFilter`](super::SynonymGraphFilter), terms
//! are walked by Rust `char`s.

use std::sync::Arc;

use crate::attributes::State;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::AnalysisError;

use super::synonym_graph_filter::walk_term;
use super::synonym_map::{NodeId, SynonymMap, WORD_SEPARATOR};

/// `SynonymFilter.TYPE_SYNONYM`.
pub const TYPE_SYNONYM: &str = "SYNONYM";

/// `SynonymFilter.PendingInput`.
#[derive(Debug)]
struct PendingInput {
    term: String,
    state: Option<State>,
    keep_orig: bool,
    matched: bool,
    consumed: bool,
    start_offset: i32,
    end_offset: i32,
}

impl PendingInput {
    fn new() -> Self {
        PendingInput {
            term: String::new(),
            state: None,
            keep_orig: false,
            matched: false,
            consumed: true,
            start_offset: 0,
            end_offset: 0,
        }
    }

    fn reset(&mut self) {
        self.state = None;
        self.consumed = true;
        self.keep_orig = false;
        self.matched = false;
    }
}

/// `SynonymFilter.PendingOutputs`.
#[derive(Debug)]
struct PendingOutputs {
    outputs: Vec<String>,
    end_offsets: Vec<i32>,
    pos_lengths: Vec<i32>,
    upto: usize,
    count: usize,
    pos_incr: i32,
    last_end_offset: i32,
    last_pos_length: i32,
}

impl PendingOutputs {
    fn new() -> Self {
        PendingOutputs {
            outputs: Vec::new(),
            end_offsets: Vec::new(),
            pos_lengths: Vec::new(),
            upto: 0,
            count: 0,
            pos_incr: 1,
            last_end_offset: 0,
            last_pos_length: 0,
        }
    }

    fn reset(&mut self) {
        self.upto = 0;
        self.count = 0;
        self.pos_incr = 1;
    }

    /// `pullNext()`: the next output word (an index into `outputs`, valid
    /// until the next `add`).
    fn pull_next(&mut self) -> usize {
        debug_assert!(self.upto < self.count);
        self.last_end_offset = self.end_offsets[self.upto];
        self.last_pos_length = self.pos_lengths[self.upto];
        let result = self.upto;
        self.upto += 1;
        self.pos_incr = 0;
        if self.upto == self.count {
            self.reset();
        }
        result
    }

    fn add(&mut self, output: &str, end_offset: i32, pos_length: i32) {
        if self.count == self.outputs.len() {
            self.outputs.push(String::new());
            self.end_offsets.push(0);
            self.pos_lengths.push(0);
        }
        self.outputs[self.count].clear();
        self.outputs[self.count].push_str(output);
        self.end_offsets[self.count] = end_offset;
        self.pos_lengths[self.count] = pos_length;
        self.count += 1;
    }
}

/// `SynonymFilter`.
pub struct SynonymFilter<I> {
    input: I,
    synonyms: Arc<SynonymMap>,
    ignore_case: bool,
    roll_buffer_size: usize,
    capture_count: usize,
    input_skip_count: usize,
    future_inputs: Vec<PendingInput>,
    future_outputs: Vec<PendingOutputs>,
    next_write: usize,
    next_read: usize,
    finished: bool,
    last_start_offset: i32,
    last_end_offset: i32,
}

impl<I: TokenStream> SynonymFilter<I> {
    /// `new SynonymFilter(input, synonyms, ignoreCase)`.
    pub fn new(input: I, synonyms: Arc<SynonymMap>, ignore_case: bool) -> Self {
        let roll_buffer_size = 1 + synonyms.max_horizontal_context;
        SynonymFilter {
            input,
            synonyms,
            ignore_case,
            roll_buffer_size,
            capture_count: 0,
            input_skip_count: 0,
            future_inputs: (0..roll_buffer_size).map(|_| PendingInput::new()).collect(),
            future_outputs: (0..roll_buffer_size)
                .map(|_| PendingOutputs::new())
                .collect(),
            next_write: 0,
            next_read: 0,
            finished: false,
            last_start_offset: 0,
            last_end_offset: 0,
        }
    }

    /// `getCaptureCount()`.
    pub fn capture_count(&self) -> usize {
        self.capture_count
    }

    fn roll_incr(&self, count: usize) -> usize {
        let count = count + 1;
        if count == self.roll_buffer_size {
            0
        } else {
            count
        }
    }

    // Java: capture
    fn capture(&mut self) {
        self.capture_count += 1;
        let a = self.input.attributes();
        let input = &mut self.future_inputs[self.next_write];
        input.state = Some(a.capture_state());
        input.consumed = false;
        input.term.clear();
        input.term.push_str(a.term());
        self.next_write = self.roll_incr(self.next_write);
        debug_assert!(self.next_write != self.next_read);
    }

    // Java: parse
    fn parse(&mut self) -> Result<(), AnalysisError> {
        debug_assert_eq!(self.input_skip_count, 0);
        let map = Arc::clone(&self.synonyms);
        let mut cur_next_read = self.next_read;
        let mut match_output: Option<NodeId> = None;
        let mut match_input_length = 0;
        let mut match_end_offset = -1;
        let mut node = map.root();
        let mut token_count = 0;

        loop {
            let input_end_offset;
            let walked = if cur_next_read == self.next_write {
                if self.finished {
                    break;
                }
                debug_assert!(self.future_inputs[self.next_write].consumed);
                if self.input.increment_token()? {
                    let (start, end) = {
                        let a = self.input.attributes();
                        (a.start_offset(), a.end_offset())
                    };
                    let nw = self.next_write;
                    self.future_inputs[nw].start_offset = start;
                    self.future_inputs[nw].end_offset = end;
                    self.last_start_offset = start;
                    self.last_end_offset = end;
                    input_end_offset = end;
                    if self.next_read != self.next_write {
                        self.capture();
                    } else {
                        self.future_inputs[nw].consumed = false;
                    }
                    walk_term(&map, node, self.input.attributes().term(), self.ignore_case)
                } else {
                    self.finished = true;
                    break;
                }
            } else {
                let fi = &self.future_inputs[cur_next_read];
                input_end_offset = fi.end_offset;
                walk_term(&map, node, &fi.term, self.ignore_case)
            };
            token_count += 1;
            let Some(after) = walked else {
                break;
            };
            node = after;
            if map.entry(node).is_some() {
                match_output = Some(node);
                match_input_length = token_count;
                match_end_offset = input_end_offset;
            }
            match map.step(node, WORD_SEPARATOR as u32) {
                None => break,
                Some(next) => {
                    node = next;
                    if self.next_read == self.next_write {
                        self.capture();
                    }
                }
            }
            cur_next_read = self.roll_incr(cur_next_read);
        }

        if self.next_read == self.next_write && !self.finished {
            self.next_write = self.roll_incr(self.next_write);
        }

        if let Some(found) = match_output {
            self.input_skip_count = match_input_length;
            self.add_output(&map, found, match_input_length as i32, match_end_offset);
        } else if self.next_read != self.next_write {
            self.input_skip_count = 1;
        } else {
            debug_assert!(self.finished);
        }
        Ok(())
    }

    // Java: addOutput
    fn add_output(
        &mut self,
        map: &SynonymMap,
        found: NodeId,
        match_input_length: i32,
        match_end_offset: i32,
    ) {
        let entry = map.entry(found).expect("a final node");
        let keep_orig = entry.keep_orig;
        for &ord in &entry.ords {
            let phrase = map.word(ord);
            let mut output_upto = self.next_read;
            let single = !phrase.contains(WORD_SEPARATOR);
            for word in phrase.split(WORD_SEPARATOR) {
                let (end_offset, pos_len) = if single {
                    (
                        match_end_offset,
                        if keep_orig { match_input_length } else { 1 },
                    )
                } else {
                    (-1, 1)
                };
                self.future_outputs[output_upto].add(word, end_offset, pos_len);
                output_upto = self.roll_incr(output_upto);
            }
        }
        let mut upto = self.next_read;
        for _ in 0..match_input_length {
            self.future_inputs[upto].keep_orig |= keep_orig;
            self.future_inputs[upto].matched = true;
            upto = self.roll_incr(upto);
        }
    }
}

impl<I: TokenStream> TokenFilter for SynonymFilter<I> {
    crate::filter_input!();

    // Java: SynonymFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        loop {
            while self.input_skip_count != 0 {
                let nr = self.next_read;
                let (consumed, keep_orig, matched) = {
                    let fi = &self.future_inputs[nr];
                    (fi.consumed, fi.keep_orig, fi.matched)
                };
                let (upto, count) = {
                    let fo = &self.future_outputs[nr];
                    (fo.upto, fo.count)
                };
                if !consumed && (keep_orig || !matched) {
                    if let Some(state) = self.future_inputs[nr].state.take() {
                        self.input.attributes_mut().restore_state(&state);
                    } else {
                        debug_assert_eq!(self.input_skip_count, 1);
                    }
                    self.future_inputs[nr].reset();
                    if count > 0 {
                        self.future_outputs[nr].pos_incr = 0;
                    } else {
                        self.next_read = self.roll_incr(nr);
                        self.input_skip_count -= 1;
                    }
                    return Ok(true);
                } else if upto < count {
                    self.future_inputs[nr].reset();
                    let fo = &mut self.future_outputs[nr];
                    let pos_incr = fo.pos_incr;
                    let idx = fo.pull_next();
                    let mut end_offset = fo.last_end_offset;
                    let pos_len = fo.last_pos_length;
                    let now_empty = fo.count == 0;
                    let fi = &self.future_inputs[nr];
                    if end_offset == -1 {
                        end_offset = fi.end_offset;
                    }
                    let start = fi.start_offset;
                    let a = self.input.attributes_mut();
                    a.clear_attributes();
                    a.set_term(&self.future_outputs[nr].outputs[idx]);
                    a.set_token_type(TYPE_SYNONYM);
                    a.set_offset(start, end_offset)?;
                    a.set_position_increment(pos_incr)?;
                    a.set_position_length(pos_len)?;
                    if now_empty {
                        self.next_read = self.roll_incr(nr);
                        self.input_skip_count -= 1;
                    }
                    return Ok(true);
                } else {
                    self.future_inputs[nr].reset();
                    self.next_read = self.roll_incr(nr);
                    self.input_skip_count -= 1;
                }
            }

            if self.finished && self.next_read == self.next_write {
                let nr = self.next_read;
                let fo = &mut self.future_outputs[nr];
                if fo.upto < fo.count {
                    let pos_incr = fo.pos_incr;
                    let idx = fo.pull_next();
                    let now_empty = fo.count == 0;
                    self.future_inputs[nr].reset();
                    if now_empty {
                        self.next_read = self.roll_incr(nr);
                        self.next_write = self.next_read;
                    }
                    let (start, end) = (self.last_start_offset, self.last_end_offset);
                    let a = self.input.attributes_mut();
                    a.clear_attributes();
                    a.set_offset(start, end)?;
                    a.set_term(&self.future_outputs[nr].outputs[idx]);
                    a.set_token_type(TYPE_SYNONYM);
                    a.set_position_increment(pos_incr)?;
                    return Ok(true);
                }
                return Ok(false);
            }

            self.parse()?;
        }
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.capture_count = 0;
        self.finished = false;
        self.input_skip_count = 0;
        self.next_read = 0;
        self.next_write = 0;
        for input in &mut self.future_inputs {
            input.reset();
        }
        for output in &mut self.future_outputs {
            output.reset();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synonym::SynonymMapBuilder;
    use crate::util::canned::{render, Canned};

    fn run(rules: &[(&str, &str, bool)], spec: &str, terms: &[&str], ignore_case: bool) -> String {
        let mut b = SynonymMapBuilder::default();
        for (i, o, keep) in rules {
            b.add(i, o, *keep).unwrap();
        }
        let mut c = Canned::parse(spec);
        c.set_terms(terms);
        let mut f = SynonymFilter::new(c, Arc::new(b.build().unwrap()), ignore_case);
        let out = render(&mut f);
        assert_eq!(render(&mut f), out);
        assert!(f.capture_count() <= terms.len());
        out
    }

    // Expected outputs are what Lucene 10.5.0's filter gave for each case.
    #[test]
    fn single_and_multi_word_outputs_stack_on_input_positions() {
        assert_eq!(
            run(
                &[("a", "x\0y", true)],
                "a:0:1:1:1 b:2:3:1:1|3|0",
                &["a", "b"],
                false
            ),
            "a:0:1:1:1 x:0:1:0:1 b:2:3:1:1 y:2:3:0:1|3|0"
        );
        assert_eq!(
            run(
                &[("a\0b", "z", false)],
                "a:0:1:1:1 b:2:3:1:1 c:4:5:1:1|5|0",
                &["a", "b", "c"],
                false
            ),
            "z:0:3:1:1 c:4:5:1:1|5|0"
        );
    }

    #[test]
    fn outputs_past_the_end_of_input_are_flushed() {
        assert_eq!(
            run(&[("a", "x\0y\0z", false)], "a:0:1:1:1|1|0", &["A"], true),
            "x:0:1:1:1 y:0:1:1:1 z:0:1:1:1|1|0"
        );
    }
}
