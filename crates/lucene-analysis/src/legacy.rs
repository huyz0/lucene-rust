//! Streaming adapters that put this crate's `Vec<Token>`-era filters
//! (analysis-common classes ported ahead of M11: ASCII folding, the Porter and
//! Snowball English stemmers, the synonym filter) into the streaming
//! [`TokenStream`] model, so [`crate::Analyzer::standard`]'s `with_*`
//! builders keep their behaviour on the new chain.
//!
//! These are not ports of a Java class; M11 replaces them with the real
//! streaming filters.

use std::collections::HashMap;
use std::sync::Arc;

use crate::attributes::AttributeSource;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::{AnalysisError, SynonymFilter, Token};

/// A 1:1 term rewrite (offsets, increments and every other attribute kept),
/// the shape of `ASCIIFoldingFilter`, `PorterStemFilter` and
/// `SnowballFilter`.
pub(crate) struct TermRewriteFilter<I> {
    input: I,
    rewrite: fn(&str) -> Option<String>,
}

impl<I: TokenStream> TermRewriteFilter<I> {
    pub(crate) fn ascii_folding(input: I) -> Self {
        TermRewriteFilter {
            input,
            rewrite: crate::AsciiFoldingFilter::fold_term,
        }
    }

    pub(crate) fn porter(input: I) -> Self {
        TermRewriteFilter {
            input,
            rewrite: |t| Some(crate::porter::stem(t)),
        }
    }

    pub(crate) fn snowball_english(input: I) -> Self {
        TermRewriteFilter {
            input,
            rewrite: |t| Some(crate::snowball_english::stem(t)),
        }
    }
}

impl<I: TokenStream> TokenFilter for TermRewriteFilter<I> {
    type Input = I;
    fn input(&self) -> &I {
        &self.input
    }
    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        if let Some(new) = (self.rewrite)(self.input.attributes().term()) {
            self.input.attributes_mut().set_term(&new);
        }
        Ok(true)
    }
}

/// Runs [`SynonymFilter::apply`]/[`SynonymFilter::apply_bidirectional`] over
/// the whole input: drains it, rewrites the token list, replays it. A token
/// the rewrite kept as it was replays its full attribute state; an injected
/// one gets cleared attributes with `SynonymGraphFilter`'s `"SYNONYM"` type.
pub(crate) struct SynonymAdapter<I> {
    input: I,
    synonyms: Arc<HashMap<String, Vec<String>>>,
    bidirectional: bool,
    out: Option<Vec<AttributeSource>>,
    next: usize,
    end_state: Option<AttributeSource>,
}

impl<I: TokenStream> SynonymAdapter<I> {
    pub(crate) fn new(
        input: I,
        synonyms: Arc<HashMap<String, Vec<String>>>,
        bidirectional: bool,
    ) -> Self {
        SynonymAdapter {
            input,
            synonyms,
            bidirectional,
            out: None,
            next: 0,
            end_state: None,
        }
    }

    fn fill(&mut self) -> Result<Vec<AttributeSource>, AnalysisError> {
        let mut states = Vec::new();
        let mut tokens = Vec::new();
        while self.input.increment_token()? {
            let a = self.input.attributes();
            tokens.push(Token {
                term: a.term().to_string(),
                start_offset: a.start_offset(),
                end_offset: a.end_offset(),
                position_increment: a.position_increment(),
                position_length: a.position_length(),
            });
            states.push(a.capture_state());
        }
        self.input.end()?;
        self.end_state = Some(self.input.attributes().capture_state());
        let rewritten = if self.bidirectional {
            SynonymFilter::apply_bidirectional(tokens.clone(), &self.synonyms)
        } else {
            SynonymFilter::apply(tokens.clone(), &self.synonyms)
        };
        let mut out = Vec::with_capacity(rewritten.len());
        let mut j = 0;
        for t in rewritten {
            if j < tokens.len() && tokens[j] == t {
                out.push(states[j].clone());
                j += 1;
                continue;
            }
            let mut a = AttributeSource::new();
            a.set_term(&t.term);
            a.set_offset(t.start_offset, t.end_offset)?;
            a.set_position_increment(t.position_increment)?;
            a.set_position_length(t.position_length)?;
            a.set_token_type("SYNONYM");
            out.push(a);
        }
        Ok(out)
    }
}

impl<I: TokenStream> TokenFilter for SynonymAdapter<I> {
    type Input = I;
    fn input(&self) -> &I {
        &self.input
    }
    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if self.out.is_none() {
            let out = self.fill()?;
            self.out = Some(out);
            self.next = 0;
        }
        let out = self.out.as_ref().expect("filled above");
        let Some(state) = out.get(self.next) else {
            return Ok(false);
        };
        self.next += 1;
        self.input.attributes_mut().restore_state(state);
        Ok(true)
    }
    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.out = None;
        self.end_state = None;
        self.next = 0;
        self.input.reset()
    }
    fn end_filter(&mut self) -> Result<(), AnalysisError> {
        match &self.end_state {
            Some(state) => {
                self.input.attributes_mut().restore_state(state);
                Ok(())
            }
            None => self.input.end(),
        }
    }
}
