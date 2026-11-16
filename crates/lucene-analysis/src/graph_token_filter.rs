//! `org.apache.lucene.analysis.GraphTokenFilter`: a filter that can walk
//! every path through the token graph starting at each base token.
//!
//! # Shape
//!
//! Java's class is abstract: a subclass's `incrementToken()` drives the
//! protected `incrementBaseToken`/`incrementGraphToken`/`incrementGraph`,
//! and inherits `reset`/`end`. The port is a struct a filter *contains*: the
//! filter implements [`TokenFilter`] with `Input = GraphTokenFilter<I>`, so
//! its inherited `reset`/`end`/`close` (the trait's forwarding defaults) run
//! this struct's, exactly as Java's subclass inherits them, and its
//! `increment` calls the three `increment_*` methods.
//!
//! Java's linked `Token` objects (each a cloned `AttributeSource` plus a
//! `nextToken` pointer, recycled through a pool) are an arena of captured
//! [`AttributeSource`]s linked by index, recycled through the same FIFO pool,
//! with Java's two limits ([`MAX_GRAPH_STACK_SIZE`], [`MAX_TOKEN_CACHE_SIZE`])
//! and their `IllegalStateException` messages. `currentGraph` keeps Java's
//! list operations (`add(index, ..)` inserts), because the stale entries they
//! leave are part of which input tokens are read ahead.

use std::collections::VecDeque;

use crate::attributes::AttributeSource;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::AnalysisError;

/// `GraphTokenFilter.MAX_GRAPH_STACK_SIZE`.
pub const MAX_GRAPH_STACK_SIZE: usize = 1000;

/// `GraphTokenFilter.MAX_TOKEN_CACHE_SIZE`.
pub const MAX_TOKEN_CACHE_SIZE: usize = 100;

/// `GraphTokenFilter.Token`.
struct GraphToken {
    atts: AttributeSource,
    next: Option<usize>,
}

/// `GraphTokenFilter` (see the module docs for how a filter uses it).
pub struct GraphTokenFilter<I> {
    input: I,
    arena: Vec<GraphToken>,
    token_pool: VecDeque<usize>,
    current_graph: Vec<Option<usize>>,
    base_token: Option<usize>,
    graph_depth: usize,
    graph_pos: usize,
    trailing_positions: i32,
    final_offsets: i32,
    stack_size: usize,
    cache_size: usize,
}

impl<I: TokenStream> GraphTokenFilter<I> {
    /// `GraphTokenFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        GraphTokenFilter {
            input,
            arena: Vec::new(),
            token_pool: VecDeque::new(),
            current_graph: Vec::new(),
            base_token: None,
            graph_depth: 0,
            graph_pos: 0,
            trailing_positions: -1,
            final_offsets: -1,
            stack_size: 0,
            cache_size: 0,
        }
    }

    /// `incrementBaseToken()`: move the root of the graph to the next token
    /// in the stream; `false` at the end.
    pub fn increment_base_token(&mut self) -> Result<bool, AnalysisError> {
        self.stack_size = 0;
        self.graph_depth = 0;
        self.graph_pos = 0;
        let old_base = self.base_token;
        self.base_token = self.next_token_in_stream(self.base_token)?;
        let Some(base) = self.base_token else {
            return Ok(false);
        };
        self.current_graph.clear();
        self.current_graph.push(Some(base));
        self.copy_to_this(base);
        self.recycle_token(old_base);
        Ok(true)
    }

    /// `incrementGraphToken()`: the next token on the current path through
    /// the graph; `false` when the path ends.
    pub fn increment_graph_token(&mut self) -> Result<bool, AnalysisError> {
        if self.graph_pos < self.graph_depth {
            self.graph_pos += 1;
            if let Some(t) = self.graph_at(self.graph_pos) {
                self.copy_to_this(t);
            }
            return Ok(true);
        }
        let from = self.graph_at(self.graph_depth);
        let Some(token) = self.next_token_in_graph(from)? else {
            return Ok(false);
        };
        self.graph_depth += 1;
        self.graph_pos += 1;
        let at = self.graph_depth.min(self.current_graph.len());
        self.current_graph.insert(at, Some(token));
        self.copy_to_this(token);
        Ok(true)
    }

    /// `incrementGraph()`: move to the next path through the graph from the
    /// current base token, resetting the attributes to the base token's;
    /// `false` when every path has been visited.
    pub fn increment_graph(&mut self) -> Result<bool, AnalysisError> {
        if self.base_token.is_none() {
            return Ok(false);
        }
        self.graph_pos = 0;
        for i in (1..=self.graph_depth).rev() {
            let at_i = self.graph_at(i);
            if !self.last_in_stack(at_i)? {
                let next = self.next_token_in_stream(at_i)?;
                self.current_graph[i] = next;
                for j in (i + 1)..self.graph_depth {
                    let at_j = self.graph_at(j);
                    let next = self.next_token_in_graph(at_j)?;
                    self.current_graph[j] = next;
                }
                let over = self.stack_size > MAX_GRAPH_STACK_SIZE;
                self.stack_size += 1;
                if over {
                    return Err(AnalysisError::IllegalState(format!(
                        "Too many graph paths (> {MAX_GRAPH_STACK_SIZE})"
                    )));
                }
                if let Some(t0) = self.graph_at(0) {
                    self.copy_to_this(t0);
                }
                self.graph_depth = i;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// `getTrailingPositions()`: the position increment `end()` reported, or
    /// `-1` before the input is exhausted.
    pub fn trailing_positions(&self) -> i32 {
        self.trailing_positions
    }

    /// `cachedTokenCount()`.
    pub fn cached_token_count(&self) -> usize {
        self.cache_size
    }

    fn graph_at(&self, i: usize) -> Option<usize> {
        self.current_graph.get(i).copied().flatten()
    }

    /// `token.attSource.copyTo(this)`.
    fn copy_to_this(&mut self, token: usize) {
        let GraphTokenFilter { input, arena, .. } = self;
        input.attributes_mut().restore_state(&arena[token].atts);
    }

    fn new_token(&mut self) -> Result<usize, AnalysisError> {
        match self.token_pool.pop_front() {
            None => {
                self.cache_size += 1;
                if self.cache_size > MAX_TOKEN_CACHE_SIZE {
                    return Err(AnalysisError::IllegalState(format!(
                        "Too many cached tokens (> {MAX_TOKEN_CACHE_SIZE})"
                    )));
                }
                self.arena.push(GraphToken {
                    atts: self.input.attributes().capture_state(),
                    next: None,
                });
                Ok(self.arena.len() - 1)
            }
            Some(token) => {
                let GraphTokenFilter { input, arena, .. } = self;
                arena[token].atts.clone_from(input.attributes());
                arena[token].next = None;
                Ok(token)
            }
        }
    }

    fn recycle_token(&mut self, token: Option<usize>) {
        if let Some(t) = token {
            self.arena[t].next = None;
            self.token_pool.push_back(t);
        }
    }

    fn next_token_in_graph(
        &mut self,
        token: Option<usize>,
    ) -> Result<Option<usize>, AnalysisError> {
        let Some(mut token) = token else {
            return Ok(None);
        };
        let mut remaining = self.arena[token].atts.position_length();
        loop {
            match self.next_token_in_stream(Some(token))? {
                None => return Ok(None),
                Some(t) => token = t,
            }
            remaining -= self.arena[token].atts.position_increment();
            if remaining <= 0 {
                return Ok(Some(token));
            }
        }
    }

    /// Whether the token after `token` is *not* at the same position.
    fn last_in_stack(&mut self, token: Option<usize>) -> Result<bool, AnalysisError> {
        let next = self.next_token_in_stream(token)?;
        Ok(match next {
            None => true,
            Some(n) => self.arena[n].atts.position_increment() != 0,
        })
    }

    fn next_token_in_stream(
        &mut self,
        token: Option<usize>,
    ) -> Result<Option<usize>, AnalysisError> {
        if let Some(t) = token {
            if let Some(next) = self.arena[t].next {
                return Ok(Some(next));
            }
        }
        if self.trailing_positions != -1 {
            // already hit the end
            return Ok(None);
        }
        if !self.input.increment_token()? {
            self.input.end()?;
            self.trailing_positions = self.input.attributes().position_increment();
            self.final_offsets = self.input.attributes().end_offset();
            return Ok(None);
        }
        let new = self.new_token()?;
        if let Some(t) = token {
            self.arena[t].next = Some(new);
        }
        Ok(Some(new))
    }
}

impl<I: TokenStream> TokenFilter for GraphTokenFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        &self.input
    }

    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    /// Java's class is abstract; used on its own, the port walks the base
    /// tokens only (every input token once, unchanged).
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        self.increment_base_token()
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        // new attributes can be added between reset() calls, so we can't
        // reuse token objects from a previous run
        self.token_pool.clear();
        self.arena.clear();
        self.current_graph.clear();
        self.cache_size = 0;
        self.graph_depth = 0;
        self.trailing_positions = -1;
        self.final_offsets = -1;
        self.base_token = None;
        Ok(())
    }

    fn end_filter(&mut self) -> Result<(), AnalysisError> {
        if self.trailing_positions == -1 {
            self.input.end()?;
            self.trailing_positions = self.input.attributes().position_increment();
            self.final_offsets = self.input.attributes().end_offset();
            Ok(())
        } else {
            let atts = self.input.attributes_mut();
            atts.end_attributes();
            atts.set_position_increment(self.trailing_positions)?;
            atts.set_offset(self.final_offsets, self.final_offsets)
        }
    }
}

#[cfg(test)]
mod tests;
