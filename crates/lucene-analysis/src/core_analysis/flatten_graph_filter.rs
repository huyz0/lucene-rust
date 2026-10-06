//! `org.apache.lucene.analysis.core.FlattenGraphFilter`: squashes a token
//! graph (tokens with `positionLength > 1`, side paths) into a "sausage" an
//! index can hold, ported state for state.
//!
//! Java's `IntArrayList` of input node ids is a `Vec<i32>`; its
//! `removeElement` removes the first occurrence, as here.

use crate::attributes::{AttributeSource, State};
use crate::token_stream::{TokenFilter, TokenStream};
use crate::util::rolling_buffer::{Resettable, RollingBuffer};
use crate::AnalysisError;

/// `FlattenGraphFilter.InputNode`.
#[derive(Debug)]
struct InputNode {
    tokens: Vec<State>,
    node: i32,
    max_to_node: i32,
    min_to_node: i32,
    output_node: i32,
    next_out: usize,
}

impl Default for InputNode {
    fn default() -> Self {
        InputNode {
            tokens: Vec::new(),
            node: -1,
            max_to_node: -1,
            min_to_node: i32::MAX,
            output_node: -1,
            next_out: 0,
        }
    }
}

impl Resettable for InputNode {
    fn reset(&mut self) {
        self.tokens.clear();
        self.node = -1;
        self.output_node = -1;
        self.max_to_node = -1;
        self.min_to_node = i32::MAX;
        self.next_out = 0;
    }
}

/// `FlattenGraphFilter.OutputNode`.
#[derive(Debug)]
struct OutputNode {
    input_nodes: Vec<i32>,
    node: i32,
    next_out: usize,
    start_offset: i32,
    end_offset: i32,
}

impl Default for OutputNode {
    fn default() -> Self {
        OutputNode {
            input_nodes: Vec::new(),
            node: -1,
            next_out: 0,
            start_offset: -1,
            end_offset: -1,
        }
    }
}

impl Resettable for OutputNode {
    fn reset(&mut self) {
        self.input_nodes.clear();
        self.node = -1;
        self.next_out = 0;
        self.start_offset = -1;
        self.end_offset = -1;
    }
}

fn remove_element(list: &mut Vec<i32>, value: i32) -> bool {
    match list.iter().position(|&v| v == value) {
        Some(i) => {
            list.remove(i);
            true
        }
        None => false,
    }
}

/// `org.apache.lucene.analysis.core.FlattenGraphFilter`.
pub struct FlattenGraphFilter<I> {
    input: I,
    input_nodes: RollingBuffer<InputNode>,
    output_nodes: RollingBuffer<OutputNode>,
    input_from: i32,
    output_from: i32,
    done: bool,
    last_output_from: i32,
    final_offset: i32,
    final_pos_inc: i32,
    max_lookahead_used: usize,
    last_start_offset: i32,
}

impl<I: TokenStream> FlattenGraphFilter<I> {
    /// `new FlattenGraphFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        FlattenGraphFilter {
            input,
            input_nodes: RollingBuffer::new(),
            output_nodes: RollingBuffer::new(),
            input_from: 0,
            output_from: 0,
            done: false,
            last_output_from: 0,
            final_offset: 0,
            final_pos_inc: 0,
            max_lookahead_used: 0,
            last_start_offset: 0,
        }
    }

    /// `getMaxLookaheadUsed()`.
    pub fn max_lookahead_used(&self) -> usize {
        self.max_lookahead_used
    }

    fn atts(&mut self) -> &mut AttributeSource {
        self.input.attributes_mut()
    }

    // Java: releaseBufferedToken
    fn release_buffered_token(&mut self) -> Result<bool, AnalysisError> {
        while self.output_from < self.output_nodes.max_pos() {
            let output_from = self.output_from;
            let (ids, next_out) = {
                let output = self.output_nodes.get(output_from);
                if output.input_nodes.is_empty() {
                    self.output_from += 1;
                    continue;
                }
                (output.input_nodes.clone(), output.next_out)
            };
            let mut max_to_node = -1;
            for &id in &ids {
                let input_node = self.input_nodes.get(id);
                debug_assert_eq!(input_node.output_node, output_from);
                max_to_node = max_to_node.max(input_node.max_to_node);
            }
            if max_to_node <= self.input_from || self.done {
                let input_node_id = ids[next_out];
                let (empty, node, state) = {
                    let n = self.input_nodes.get(input_node_id);
                    (
                        n.tokens.is_empty(),
                        n.node,
                        n.tokens.get(n.next_out).cloned(),
                    )
                };
                // Java's loop condition keeps outputFrom below getMaxPos, so
                // this guard is never true; it is kept as Java has it.
                if self.done && empty && output_from >= self.output_nodes.max_pos() {
                    return Ok(false);
                }
                let Some(state) = state else {
                    let output = self.output_nodes.get(output_from);
                    if output.input_nodes.len() > 1 {
                        output.next_out += 1;
                        if output.next_out < output.input_nodes.len() {
                            continue;
                        }
                    }
                    self.free_before(output_from);
                    continue;
                };
                self.atts().restore_state(&state);
                let inc = output_from - self.last_output_from;
                self.atts().set_position_increment(inc)?;
                let to_input_node_id = node + self.atts().position_length();
                let to_output_node = self.input_nodes.get(to_input_node_id).output_node;
                self.atts()
                    .set_position_length(to_output_node - output_from)?;
                self.last_output_from = output_from;
                let input_done = {
                    let n = self.input_nodes.get(input_node_id);
                    n.next_out += 1;
                    n.next_out == n.tokens.len()
                };
                let end_node_offset = self.output_nodes.get(to_output_node).end_offset;
                let output_start = self.output_nodes.get(output_from).start_offset;
                let start_offset = self.last_start_offset.max(output_start);
                let end_offset = start_offset.max(end_node_offset);
                self.atts().set_offset(start_offset, end_offset)?;
                self.last_start_offset = start_offset;
                if input_done {
                    let output = self.output_nodes.get(output_from);
                    output.next_out += 1;
                    if output.next_out == output.input_nodes.len() {
                        self.free_before(output_from);
                    }
                }
                return Ok(true);
            } else {
                return Ok(false);
            }
        }
        Ok(false)
    }

    // Java: freeBefore(OutputNode)
    fn free_before(&mut self, output_pos: i32) {
        self.output_from += 1;
        let free_before = self
            .output_nodes
            .get(output_pos)
            .input_nodes
            .iter()
            .copied()
            .min()
            .expect("an output node being freed has input nodes");
        self.input_nodes.free_before(free_before);
        self.output_nodes.free_before(self.output_from);
    }

    // Java: recoverFromHole
    fn recover_from_hole(&mut self, src_pos: i32, start_offset: i32, posinc: i32) -> i32 {
        let input_from = self.input_from;
        self.input_nodes.get(src_pos).node = input_from;
        let previous_input_from = input_from - posinc;
        let out_index = if previous_input_from >= 0 {
            let min_to_node = self.input_nodes.get(previous_input_from).min_to_node;
            if min_to_node < input_from {
                self.input_nodes.get(min_to_node).output_node + 1
            } else {
                self.output_nodes.max_pos()
            }
        } else {
            self.output_nodes.max_pos() + 1
        };
        self.input_nodes.get(src_pos).output_node = out_index;
        let out_src = self.output_nodes.get(out_index);
        if out_src.node == -1 {
            out_src.node = out_index;
            out_src.start_offset = start_offset;
        } else {
            out_src.start_offset = start_offset.max(out_src.start_offset);
        }
        out_src.input_nodes.push(input_from);
        out_index
    }
}

impl<I: TokenStream> TokenFilter for FlattenGraphFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        &self.input
    }

    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    // Java: FlattenGraphFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        loop {
            if self.release_buffered_token()? {
                return Ok(true);
            } else if self.done {
                return Ok(false);
            }
            if self.input.increment_token()? {
                let a = self.input.attributes();
                let position_increment = a.position_increment();
                let (start_offset, end_offset, pos_len) =
                    (a.start_offset(), a.end_offset(), a.position_length());
                self.input_from += position_increment;
                let input_from = self.input_from;
                let input_to = input_from + pos_len;
                let (src_node, src_output_node) = {
                    let src = self.input_nodes.get(input_from);
                    (src.node, src.output_node)
                };
                if src_node == -1 {
                    self.recover_from_hole(input_from, start_offset, position_increment);
                } else {
                    let mut out_src = src_output_node;
                    if position_increment > 1 {
                        let prev = self.input_nodes.get(input_from - position_increment);
                        let (prev_out, prev_min_to) = (prev.output_node, prev.min_to_node);
                        if src_output_node - prev_out <= 1 && prev_min_to != input_from {
                            remove_element(
                                &mut self.output_nodes.get(out_src).input_nodes,
                                input_from,
                            );
                            self.input_nodes.get(input_from).output_node = -1;
                            let prev_end_offset = self.output_nodes.get(out_src).end_offset;
                            out_src = self.recover_from_hole(
                                input_from,
                                start_offset,
                                position_increment,
                            );
                            self.output_nodes.get(out_src).end_offset = prev_end_offset;
                        }
                    }
                    let o = self.output_nodes.get(out_src);
                    if o.start_offset == -1 || start_offset > o.start_offset {
                        o.start_offset = start_offset.max(o.start_offset);
                    }
                }
                let state = self.input.attributes().capture_state();
                let src = self.input_nodes.get(input_from);
                src.tokens.push(state);
                src.max_to_node = src.max_to_node.max(input_to);
                src.min_to_node = src.min_to_node.min(input_to);
                let src_output_node = src.output_node;
                self.max_lookahead_used =
                    self.max_lookahead_used.max(self.input_nodes.buffer_size());
                let dest = self.input_nodes.get(input_to);
                if dest.node == -1 {
                    dest.node = input_to;
                }
                let dest_output_node = dest.output_node;
                let output_end_node = src_output_node + 1;
                let mut dest_out = dest_output_node;
                if output_end_node > dest_output_node {
                    if dest_output_node != -1 {
                        let removed = remove_element(
                            &mut self.output_nodes.get(dest_output_node).input_nodes,
                            input_to,
                        );
                        debug_assert!(removed);
                    }
                    self.output_nodes
                        .get(output_end_node)
                        .input_nodes
                        .push(input_to);
                    self.input_nodes.get(input_to).output_node = output_end_node;
                    dest_out = output_end_node;
                }
                let out_dest = self.output_nodes.get(dest_out);
                if out_dest.end_offset == -1 || end_offset < out_dest.end_offset {
                    out_dest.end_offset = end_offset;
                }
            } else {
                self.input.end()?;
                let a = self.input.attributes();
                self.final_pos_inc = a.position_increment();
                self.final_offset = a.end_offset();
                self.done = true;
            }
        }
    }

    // Java: FlattenGraphFilter.end
    fn end_filter(&mut self) -> Result<(), AnalysisError> {
        if !self.done {
            self.input.end()?;
        }
        self.atts().clear_attributes();
        if self.done {
            let (inc, off) = (self.final_pos_inc, self.final_offset);
            self.atts().set_position_increment(inc)?;
            self.atts().set_offset(off, off)
        } else {
            self.input.end()
        }
    }

    // Java: FlattenGraphFilter.reset
    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.input_from = -1;
        self.input_nodes.reset();
        {
            let n = self.input_nodes.get(0);
            n.node = 0;
            n.output_node = 0;
        }
        self.output_nodes.reset();
        {
            let out = self.output_nodes.get(0);
            out.node = 0;
            out.input_nodes.push(0);
            out.start_offset = 0;
        }
        self.output_from = 0;
        self.last_output_from = -1;
        self.done = false;
        self.final_pos_inc = -1;
        self.final_offset = -1;
        self.last_start_offset = 0;
        self.max_lookahead_used = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::{render, Canned};

    /// `(input, what Lucene 10.5.0's FlattenGraphFilter gave for it)`.
    const CASES: &[(&str, &str)] = &[
        (
            "wtf:0:3:1:5 what:0:1:0:1 wow:0:3:0:3 the:1:2:1:1 fudge:2:3:1:1 that's:1:2:1:1 funny:2:3:1:1 happened:4:12:1:1|12|0",
            "wtf:0:3:1:5 what:0:1:0:1 wow:0:3:0:3 the:1:2:1:1 fudge:2:3:1:1 that's:2:2:1:1 funny:2:3:1:1 happened:4:12:1:1|12|0",
        ),
        (
            "wizard:0:6:1:1 wiz:0:6:0:2 of:7:9:1:1 oz:10:12:1:1|12|0",
            "wizard:0:6:1:1 wiz:0:9:0:2 of:7:9:1:1 oz:10:12:1:1|12|0",
        ),
        (
            "hello:0:5:1:1 hole:10:14:2:1 world:15:20:1:1|20|2",
            "hello:0:5:1:1 hole:10:14:2:1 world:15:20:1:1|20|2",
        ),
        (
            "hello:0:5:1:1 big:6:9:1:2 x:10:11:3:1 y:12:13:1:1|13|1",
            "hello:0:5:1:1 big:6:9:1:1 x:10:11:2:1 y:12:13:1:1|13|1",
        ),
        (
            "a:0:1:1:2 b:0:1:0:1 c:2:3:2:1 d:4:5:1:1|5|0",
            "a:0:1:1:2 b:0:1:0:1 c:2:3:2:1 d:4:5:1:1|5|0",
        ),
        (
            "a:0:1:1:3 b:0:1:0:1 c:2:3:1:1 e:4:5:2:1|8|3",
            "a:0:1:1:3 b:0:1:0:1 c:2:3:1:1 e:4:5:2:1|8|3",
        ),
        (
            "x:0:1:1:1 y:2:3:3:2 z:2:3:0:1 w:4:5:1:1|9|1",
            "x:0:1:1:1 y:2:5:2:2 z:2:3:0:1 w:4:5:1:1|9|1",
        ),
    ];

    #[test]
    fn flattens_as_lucene_does() {
        for (input, expected) in CASES {
            let mut f = FlattenGraphFilter::new(Canned::parse(input));
            assert_eq!(render(&mut f), *expected, "input {input}");
            assert!(f.max_lookahead_used() >= 1);
            // Reuse: a second pass over the same canned tokens is identical.
            assert_eq!(render(&mut f), *expected, "reused, input {input}");
        }
    }

    #[test]
    fn end_before_exhaustion_forwards_the_input_end() {
        let mut f = FlattenGraphFilter::new(Canned::parse("a:0:1:1:1 b:2:3:1:1|7|2"));
        f.reset().unwrap();
        assert!(f.increment_token().unwrap());
        f.end().unwrap();
        assert_eq!(f.attributes().end_offset(), 7);
        assert_eq!(f.attributes().position_increment(), 2);
    }
}
