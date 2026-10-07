//! `org.apache.lucene.analysis.sinks`: `TeeSinkTokenFilter` -- passes its
//! input through while recording each token's state, which any number of
//! sinks ([`TeeSinkTokenFilter::new_sink_token_stream`]) replay later.
//!
//! Differs: the recorded states are shared through an `Arc<Mutex<..>>`
//! (Java shares a plain list); a sink iterates the list by index, so a token
//! the tee adds after the sink's `reset` is replayed where Java's
//! `ArrayList` iterator would throw `ConcurrentModificationException`.

use std::sync::{Arc, Mutex};

use crate::attributes::{AttributeSource, State};
use crate::token_stream::{TokenFilter, TokenStream};
use crate::AnalysisError;

/// `TeeSinkTokenFilter.States`.
#[derive(Debug, Default)]
struct States {
    states: Vec<State>,
    final_state: Option<State>,
}

fn lock(s: &Mutex<States>) -> std::sync::MutexGuard<'_, States> {
    s.lock().unwrap_or_else(|e| e.into_inner())
}

/// `TeeSinkTokenFilter`.
pub struct TeeSinkTokenFilter<I> {
    input: I,
    cached: Arc<Mutex<States>>,
}

impl<I: TokenStream> TeeSinkTokenFilter<I> {
    /// `new TeeSinkTokenFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        TeeSinkTokenFilter {
            input,
            cached: Arc::new(Mutex::new(States::default())),
        }
    }

    /// `newSinkTokenStream()`: a stream over a copy of this filter's
    /// attributes (`cloneAttributes`) replaying the recorded states.
    pub fn new_sink_token_stream(&self) -> SinkTokenStream {
        SinkTokenStream {
            atts: self.input.attributes().clone(),
            cached: Arc::clone(&self.cached),
            next: None,
        }
    }

    /// `consumeAllTokens()`.
    pub fn consume_all_tokens(&mut self) -> Result<(), AnalysisError> {
        while self.increment_token()? {}
        Ok(())
    }
}

impl<I: TokenStream> TokenFilter for TeeSinkTokenFilter<I> {
    crate::filter_input!();

    // Java: TeeSinkTokenFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if self.input.increment_token()? {
            let state = self.input.attributes().capture_state();
            lock(&self.cached).states.push(state);
            return Ok(true);
        }
        Ok(false)
    }

    fn end_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.end()?;
        let state = self.input.attributes().capture_state();
        lock(&self.cached).final_state = Some(state);
        Ok(())
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        {
            let mut c = lock(&self.cached);
            c.final_state = None;
            c.states.clear();
        }
        self.input.reset()
    }
}

/// `TeeSinkTokenFilter.SinkTokenStream`.
pub struct SinkTokenStream {
    atts: AttributeSource,
    cached: Arc<Mutex<States>>,
    /// The next state to replay; `None` before `reset` (Java's null
    /// iterator).
    next: Option<usize>,
}

impl TokenStream for SinkTokenStream {
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
        let Some(i) = self.next else {
            return Err(AnalysisError::IllegalState(
                "SinkTokenStream consumed before reset()".into(),
            ));
        };
        let c = lock(&self.cached);
        match c.states.get(i) {
            Some(s) => {
                self.atts.restore_state(s);
                self.next = Some(i + 1);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    fn end(&mut self) -> Result<(), AnalysisError> {
        if let Some(f) = &lock(&self.cached).final_state {
            self.atts.restore_state(f);
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.next = Some(0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::StrReader;
    use crate::util::WhitespaceTokenizer;
    use crate::{LowerCaseFilter, Tokenizer};

    fn terms(ts: &mut dyn TokenStream) -> (Vec<String>, i32) {
        let mut out = Vec::new();
        let end = crate::token_stream::consume(ts, |a| out.push(a.term().to_string())).unwrap();
        (out, end.end_offset())
    }

    #[test]
    fn sinks_replay_what_the_tee_saw() {
        let mut t = WhitespaceTokenizer::new();
        t.set_reader(Box::new(StrReader::new("The Quick fox")))
            .unwrap();
        let mut tee = TeeSinkTokenFilter::new(t);
        let mut sink = tee.new_sink_token_stream();
        assert!(sink.increment_token().is_err());
        let (seen, end) = terms(&mut tee);
        assert_eq!(seen, ["The", "Quick", "fox"]);
        assert_eq!(end, 13);
        let mut lower = LowerCaseFilter::new(tee.new_sink_token_stream());
        assert_eq!(terms(&mut lower).0, ["the", "quick", "fox"]);
        let (again, end) = terms(&mut sink);
        assert_eq!(again, seen);
        assert_eq!(end, 13);
        let mut t = WhitespaceTokenizer::new();
        t.set_reader(Box::new(StrReader::new("a b"))).unwrap();
        let mut tee = TeeSinkTokenFilter::new(t);
        tee.reset().unwrap();
        tee.consume_all_tokens().unwrap();
        let mut s = tee.new_sink_token_stream();
        assert_eq!(terms(&mut s).0, ["a", "b"]);
    }
}
