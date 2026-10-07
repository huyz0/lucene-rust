//! `org.apache.lucene.analysis.miscellaneous.ConditionalTokenFilter` and
//! `ProtectedTermFilter`.
//!
//! # Shape
//!
//! Java builds the delegate chain over a `OneTimeWrapper` -- an inner class
//! sharing the outer filter's fields and attribute source -- and the outer
//! filter reads its input directly while the delegate reads it through the
//! wrapper. In the port the attributes live in the chain's source, so the
//! wrapper *owns* the input, and with it every field the two classes share
//! (`state`, `bufferedState`, `exhausted`, `adjustPosition`, `endState`,
//! `endOffset`) and the `shouldFilter()` predicate. The outer filter owns
//! the delegate and reaches the wrapper at the bottom of it through
//! [`ConditionalRoot`], implemented for the wrapper and for every
//! [`TokenFilter`] over it.

use std::sync::Arc;

use crate::attributes::{AttributeSource, State};
use crate::token_stream::{TokenFilter, TokenStream, Tokenizer};
use crate::{AnalysisError, CharArraySet};

/// `ConditionalTokenFilter.TokenState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenState {
    Reading,
    Prebuffering,
    Delegating,
}

/// `ConditionalTokenFilter.shouldFilter()`.
pub trait ShouldFilter: Send {
    /// Whether the current token goes through the delegate.
    fn should_filter(&mut self, attributes: &AttributeSource) -> Result<bool, AnalysisError>;
}

impl<F: FnMut(&AttributeSource) -> bool + Send> ShouldFilter for F {
    fn should_filter(&mut self, attributes: &AttributeSource) -> Result<bool, AnalysisError> {
        Ok(self(attributes))
    }
}

/// `ConditionalTokenFilter.OneTimeWrapper`, owning the input and the state
/// the outer filter shares with it (see the module docs).
pub struct OneTimeWrapper<I, C> {
    input: I,
    cond: C,
    state: TokenState,
    buffered_state: Option<State>,
    exhausted: bool,
    adjust_position: bool,
    end_state: Option<State>,
    end_offset: i32,
}

impl<I: TokenStream + 'static, C: ShouldFilter + 'static> TokenStream for OneTimeWrapper<I, C> {
    fn attributes(&self) -> &AttributeSource {
        self.input.attributes()
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        self.input.attributes_mut()
    }

    // Java: OneTimeWrapper.incrementToken
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        if self.state == TokenState::Prebuffering {
            if self.input.attributes().position_increment() == 0 {
                self.adjust_position = true;
                self.input.attributes_mut().set_position_increment(1)?;
            }
            self.state = TokenState::Delegating;
            return Ok(true);
        }
        debug_assert_eq!(self.state, TokenState::Delegating);
        if self.input.increment_token()? {
            if self.cond.should_filter(self.input.attributes())? {
                return Ok(true);
            }
            self.end_offset = self.input.attributes().end_offset();
            self.buffered_state = Some(self.input.attributes().capture_state());
        } else {
            self.exhausted = true;
        }
        Ok(false)
    }

    /// Java: `OneTimeWrapper.reset()` does nothing.
    fn reset(&mut self) -> Result<(), AnalysisError> {
        Ok(())
    }

    // Java: OneTimeWrapper.end
    fn end(&mut self) -> Result<(), AnalysisError> {
        if self.exhausted {
            if self.end_state.is_none() {
                self.input.end()?;
                self.end_state = Some(self.input.attributes().capture_state());
            }
            self.end_offset = self.input.attributes().end_offset();
        }
        let a = self.input.attributes_mut();
        a.end_attributes();
        a.set_offset(self.end_offset, self.end_offset)
    }

    /// Java's wrapper inherits `TokenStream.close()`, which does nothing:
    /// the outer filter closes the input.
    fn close(&mut self) -> Result<(), AnalysisError> {
        Ok(())
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        self.input.as_tokenizer()
    }

    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

/// Reaches the [`OneTimeWrapper`] at the bottom of a delegate chain.
pub trait ConditionalRoot<I, C> {
    /// The wrapper.
    fn root(&mut self) -> &mut OneTimeWrapper<I, C>;
}

impl<I, C> ConditionalRoot<I, C> for OneTimeWrapper<I, C> {
    fn root(&mut self) -> &mut OneTimeWrapper<I, C> {
        self
    }
}

impl<I, C, F> ConditionalRoot<I, C> for F
where
    F: TokenFilter,
    F::Input: ConditionalRoot<I, C>,
{
    fn root(&mut self) -> &mut OneTimeWrapper<I, C> {
        self.input_mut().root()
    }
}

/// `org.apache.lucene.analysis.miscellaneous.ConditionalTokenFilter`: runs the
/// tokens [`ShouldFilter`] selects through a delegate chain, the others
/// past it.
pub struct ConditionalTokenFilter<I, C, D> {
    delegate: D,
    last_token_filtered: bool,
    _marker: std::marker::PhantomData<fn() -> (I, C)>,
}

impl<I, C, D> ConditionalTokenFilter<I, C, D>
where
    I: TokenStream,
    C: ShouldFilter,
    D: TokenStream + ConditionalRoot<I, C>,
{
    /// `ConditionalTokenFilter(TokenStream, Function<TokenStream,
    /// TokenStream>)` with `shouldFilter()` as `cond`.
    pub fn new(input: I, cond: C, factory: impl FnOnce(OneTimeWrapper<I, C>) -> D) -> Self {
        let wrapper = OneTimeWrapper {
            input,
            cond,
            state: TokenState::Reading,
            buffered_state: None,
            exhausted: false,
            adjust_position: false,
            end_state: None,
            end_offset: -1,
        };
        ConditionalTokenFilter {
            delegate: factory(wrapper),
            last_token_filtered: false,
            _marker: std::marker::PhantomData,
        }
    }

    // Java: ConditionalTokenFilter.endDelegating
    fn end_delegating(&mut self) -> Result<bool, AnalysisError> {
        if self.delegate.root().buffered_state.is_none() {
            debug_assert!(self.delegate.root().exhausted);
            return Ok(false);
        }
        self.delegate.end()?;
        let w = self.delegate.root();
        let pos_inc = w.input.attributes().position_increment();
        let buffered = w.buffered_state.take().expect("checked above");
        let a = w.input.attributes_mut();
        a.restore_state(&buffered);
        a.set_position_increment(a.position_increment() + pos_inc)?;
        if w.adjust_position {
            let a = w.input.attributes_mut();
            a.set_position_increment(a.position_increment() - 1)?;
            w.adjust_position = false;
        }
        Ok(true)
    }
}

impl<I, C, D> TokenFilter for ConditionalTokenFilter<I, C, D>
where
    I: TokenStream,
    C: ShouldFilter,
    D: TokenStream + ConditionalRoot<I, C>,
{
    type Input = D;

    fn input(&self) -> &D {
        &self.delegate
    }

    fn input_mut(&mut self) -> &mut D {
        &mut self.delegate
    }

    // Java: ConditionalTokenFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        self.last_token_filtered = false;
        // Java loops `while (true)`, but every branch returns: the state is
        // READING or DELEGATING here (PREBUFFERING never survives a call).
        let state = self.delegate.root().state;
        if state == TokenState::Reading {
            let w = self.delegate.root();
            if let Some(buffered) = w.buffered_state.take() {
                w.input.attributes_mut().restore_state(&buffered);
                self.last_token_filtered = false;
                return Ok(true);
            }
            if w.exhausted {
                return Ok(false);
            }
            if !w.input.increment_token()? {
                w.exhausted = true;
                return Ok(false);
            }
            if w.cond.should_filter(w.input.attributes())? {
                self.last_token_filtered = true;
                w.state = TokenState::Prebuffering;
                self.delegate.reset()?;
                let more = self.delegate.increment_token()?;
                let w = self.delegate.root();
                if more {
                    w.state = TokenState::Delegating;
                    if w.adjust_position {
                        let a = w.input.attributes_mut();
                        a.set_position_increment(a.position_increment() - 1)?;
                    }
                    w.adjust_position = false;
                } else {
                    w.state = TokenState::Reading;
                    return self.end_delegating();
                }
            }
            return Ok(true);
        }
        self.last_token_filtered = true;
        if self.delegate.increment_token()? {
            return Ok(true);
        }
        self.delegate.root().state = TokenState::Reading;
        self.end_delegating()
    }

    // Java: ConditionalTokenFilter.reset
    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.delegate.root().input.reset()?;
        self.delegate.reset()?;
        let w = self.delegate.root();
        w.state = TokenState::Reading;
        self.last_token_filtered = false;
        w.buffered_state = None;
        w.exhausted = false;
        w.adjust_position = false;
        w.end_offset = -1;
        w.end_state = None;
        Ok(())
    }

    // Java: ConditionalTokenFilter.end
    fn end_filter(&mut self) -> Result<(), AnalysisError> {
        let w = self.delegate.root();
        match &w.end_state {
            None => {
                w.input.end()?;
                w.end_state = Some(w.input.attributes().capture_state());
            }
            Some(s) => {
                let s = s.clone();
                w.input.attributes_mut().restore_state(&s);
            }
        }
        w.end_offset = w.input.attributes().end_offset();
        if self.last_token_filtered {
            self.delegate.end()?;
            let w = self.delegate.root();
            w.end_state = Some(w.input.attributes().capture_state());
        }
        Ok(())
    }

    // Java: ConditionalTokenFilter.close
    fn close_filter(&mut self) -> Result<(), AnalysisError> {
        self.delegate.root().input.close()?;
        self.delegate.close()
    }

    /// Past this filter's own wrapper: an enclosing conditional filter's
    /// wrapper is below it.
    fn conditional_root_filter(&mut self) -> Option<&mut dyn std::any::Any> {
        self.delegate.root().input.conditional_root()
    }
}

/// `ProtectedTermFilter.shouldFilter()`: the term is *not* protected.
pub struct NotProtected(Arc<CharArraySet>);

impl ShouldFilter for NotProtected {
    fn should_filter(&mut self, a: &AttributeSource) -> Result<bool, AnalysisError> {
        Ok(!self.0.contains(a.term()))
    }
}

/// `org.apache.lucene.analysis.miscellaneous.ProtectedTermFilter`.
pub type ProtectedTermFilter<I, D> = ConditionalTokenFilter<I, NotProtected, D>;

/// `new ProtectedTermFilter(CharArraySet, TokenStream, Function)`.
pub fn protected_term_filter<I, D>(
    protected_terms: Arc<CharArraySet>,
    input: I,
    factory: impl FnOnce(OneTimeWrapper<I, NotProtected>) -> D,
) -> ProtectedTermFilter<I, D>
where
    I: TokenStream,
    D: TokenStream + ConditionalRoot<I, NotProtected>,
{
    ConditionalTokenFilter::new(input, NotProtected(protected_terms), factory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::miscellaneous::KeywordRepeatFilter;
    use crate::util::canned::{render, Canned};
    use crate::LowerCaseFilter;

    #[test]
    fn protected_terms_skip_the_delegate() {
        let protected = Arc::new(CharArraySet::from_words(["FOO"], false));
        let mut f = protected_term_filter(
            protected,
            Canned::parse("FOO:0:3:1:1 BAR:4:7:1:1 Baz:8:11:1:1|11|1"),
            LowerCaseFilter::new,
        );
        assert_eq!(render(&mut f), "FOO:0:3:1:1 bar:4:7:1:1 baz:8:11:1:1|11|0");
        // Reused.
        assert_eq!(render(&mut f), "FOO:0:3:1:1 bar:4:7:1:1 baz:8:11:1:1|11|0");
    }

    /// `(input, what Lucene gave)` for a delegate that drops `x`, applied
    /// to every token but `a`.
    const DROPPING: &[(&str, &str)] = &[
        (
            "x:0:1:1:1 y:2:3:0:1 a:4:5:1:1 x:6:7:1:1 z:8:9:1:1|9|0",
            "y:2:3:1:1 a:4:5:1:1 z:8:9:2:1|9|0",
        ),
        (
            "y:0:1:1:1 x:2:3:0:1 a:4:5:1:1|9|2",
            "y:0:1:1:1 a:4:5:1:1|9|2",
        ),
        ("x:0:1:1:1 x:2:3:1:1|5|1", "|5|2"),
        (
            "a:0:1:1:1 y:2:3:0:1 x:4:5:1:1|7|0",
            "a:0:1:1:1 y:2:3:0:1|7|1",
        ),
    ];

    #[test]
    fn a_delegate_that_drops_tokens() {
        for (input, expected) in DROPPING {
            let cond = |a: &AttributeSource| a.term() != "a";
            let stop = Arc::new(CharArraySet::from_words(["x"], false));
            let mut f = ConditionalTokenFilter::new(Canned::parse(input), cond, |w| {
                crate::StopFilter::new(w, stop)
            });
            assert_eq!(render(&mut f), *expected, "input {input}");
            // A second end() restores the captured end state.
            f.end().unwrap();
            assert_eq!(&render(&mut f), expected, "reused, input {input}");
        }
    }

    #[test]
    fn a_delegate_that_adds_tokens_and_stacked_input() {
        // KeywordRepeat doubles each filtered token; "a" is not filtered.
        let cond = |a: &AttributeSource| a.term() != "a";
        let mut f = ConditionalTokenFilter::new(
            Canned::parse("x:0:1:1:1 y:0:1:0:1 a:2:3:1:1 z:4:5:1:1|5|0"),
            cond,
            KeywordRepeatFilter::new,
        );
        assert_eq!(
            render(&mut f),
            "x:0:1:1:1 x:0:1:0:1 y:0:1:0:1 y:0:1:0:1 a:2:3:1:1 z:4:5:1:1 z:4:5:0:1|5|0"
        );
        f.close().unwrap();
        // Ending on an unfiltered token.
        let cond = |a: &AttributeSource| a.term() == "x";
        let mut f = ConditionalTokenFilter::new(
            Canned::parse("x:0:1:1:1 b:2:3:1:1|3|1"),
            cond,
            LowerCaseFilter::new,
        );
        assert_eq!(render(&mut f), "x:0:1:1:1 b:2:3:1:1|3|1");
    }
}
