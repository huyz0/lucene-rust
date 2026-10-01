//! The streaming analysis model: Java's `TokenStream`, `Tokenizer`,
//! `TokenFilter`, `FilteringTokenFilter` and `CachingTokenFilter`.
//!
//! # How a chain shares its attributes
//!
//! In Java every stage of a chain is an `AttributeSource` sharing one
//! attribute map: a `TokenFilter`'s constructor calls `super(input)`, which
//! aliases the input's attributes. The Rust shape is ownership instead of
//! aliasing: the [`AttributeSource`] lives in the chain's source (the
//! tokenizer), and a filter, which owns its input, reaches the same
//! attributes *through* it -- [`TokenFilter`]'s blanket [`TokenStream`] impl
//! forwards [`TokenStream::attributes`] to [`TokenFilter::input`]. So there
//! is still exactly one attribute set per chain, as in Java, with no
//! `Rc<RefCell<_>>`.
//!
//! # The lifecycle
//!
//! Java's contract, unchanged: `reset()`, `increment_token()` until it
//! returns `false`, `end()`, `close()`. Every method returns `Result` where
//! Java throws `IOException` (or `IllegalStateException` for a contract
//! violation).

use crate::attributes::AttributeSource;
use crate::reader::CharReader;
use crate::AnalysisError;

/// `org.apache.lucene.analysis.TokenStream`.
pub trait TokenStream: Send {
    /// The chain's shared attributes (Java: this stream *is* the
    /// `AttributeSource`).
    fn attributes(&self) -> &AttributeSource;

    /// The chain's shared attributes, mutably.
    fn attributes_mut(&mut self) -> &mut AttributeSource;

    /// `incrementToken()`: advance to the next token, `false` at the end.
    fn increment_token(&mut self) -> Result<bool, AnalysisError>;

    /// `reset()`: Java's base does nothing.
    fn reset(&mut self) -> Result<(), AnalysisError> {
        Ok(())
    }

    /// `end()`: Java's base is `endAttributes()`.
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.attributes_mut().end_attributes();
        Ok(())
    }

    /// `close()`: Java's base does nothing.
    fn close(&mut self) -> Result<(), AnalysisError> {
        Ok(())
    }

    /// The [`Tokenizer`] at the source of this chain, if it has one -- how
    /// [`crate::TokenStreamComponents`] hands a new reader to the tokenizer
    /// a filter chain wraps (Java keeps a second reference to it; Rust
    /// reaches it through the chain, like `TokenFilter.unwrap()`).
    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        None
    }
}

impl TokenStream for Box<dyn TokenStream> {
    fn attributes(&self) -> &AttributeSource {
        (**self).attributes()
    }
    fn attributes_mut(&mut self) -> &mut AttributeSource {
        (**self).attributes_mut()
    }
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        (**self).increment_token()
    }
    fn reset(&mut self) -> Result<(), AnalysisError> {
        (**self).reset()
    }
    fn end(&mut self) -> Result<(), AnalysisError> {
        (**self).end()
    }
    fn close(&mut self) -> Result<(), AnalysisError> {
        (**self).close()
    }
    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        (**self).as_tokenizer()
    }
}

/// `org.apache.lucene.analysis.Tokenizer`: a [`TokenStream`] whose input is
/// a [`CharReader`].
///
/// Implementations keep a [`TokenizerInput`] (Java's `input` /
/// `inputPending` fields and their state machine) and delegate
/// [`Self::set_reader`], `reset`, `close` and offset correction to it.
pub trait Tokenizer: TokenStream {
    /// `setReader(Reader)`: the input for the next `reset()`.
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError>;
}

/// Java's `Tokenizer.ILLEGAL_STATE_READER` message.
const ILLEGAL_STATE_READER_MSG: &str = "TokenStream contract violation: reset()/close() call missing, reset() called multiple times, or subclass does not call super.reset(). Please see Javadocs of TokenStream class for more information about the correct consuming workflow.";

/// The reader half of Java's `Tokenizer`: `input`, `inputPending`, and the
/// `ILLEGAL_STATE_READER` sentinel as `None`.
#[derive(Default)]
pub struct TokenizerInput {
    input: Option<Box<dyn CharReader>>,
    pending: Option<Box<dyn CharReader>>,
}

impl TokenizerInput {
    pub fn new() -> Self {
        Self::default()
    }

    /// `Tokenizer.setReader`.
    pub fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        if self.input.is_some() {
            return Err(AnalysisError::IllegalState(
                "TokenStream contract violation: close() call missing".to_string(),
            ));
        }
        self.pending = Some(input);
        Ok(())
    }

    /// `Tokenizer.reset()`: `input = inputPending; inputPending = ILLEGAL`.
    pub fn reset(&mut self) {
        self.input = self.pending.take();
    }

    /// `Tokenizer.close()`: closes the input; both back to the sentinel.
    pub fn close(&mut self) -> Result<(), AnalysisError> {
        self.pending = None;
        match self.input.take() {
            Some(mut input) => input.close(),
            None => Ok(()),
        }
    }

    /// `Tokenizer.correctOffset`.
    pub fn correct_offset(&self, current_off: i32) -> i32 {
        match &self.input {
            Some(input) => input.correct_offset(current_off),
            None => current_off,
        }
    }

    /// [`CharReader::whole_text`] of the live input, `None` without one.
    pub fn whole_text(&self) -> Option<&str> {
        self.input.as_deref().and_then(|r| r.whole_text())
    }

    /// The live input; reading it outside `reset()`..`close()` is Java's
    /// `ILLEGAL_STATE_READER` error.
    pub fn reader(&mut self) -> Result<&mut dyn CharReader, AnalysisError> {
        match self.input.as_deref_mut() {
            Some(r) => Ok(r),
            None => Err(AnalysisError::IllegalState(
                ILLEGAL_STATE_READER_MSG.to_string(),
            )),
        }
    }
}

/// `org.apache.lucene.analysis.TokenFilter`: a stream whose tokens come from
/// another stream.
///
/// Implement [`Self::input`]/[`Self::input_mut`] and [`Self::increment`]
/// (Java's `incrementToken()`); override [`Self::reset_filter`],
/// [`Self::end_filter`] or [`Self::close_filter`] where Java's filter
/// overrides `reset`/`end`/`close` (their defaults forward to the input, as
/// Java's `TokenFilter` does). Every `TokenFilter` is then a [`TokenStream`]
/// sharing its input's attributes, so filters chain by ownership:
/// `LowerCaseFilter::new(StandardTokenizer::new())`.
///
/// The hooks are named apart from [`TokenStream`]'s methods so a concrete
/// filter, which has both traits, never makes a call ambiguous.
pub trait TokenFilter: Send {
    /// The type of the wrapped stream.
    type Input: TokenStream;

    /// `TokenFilter.input`.
    fn input(&self) -> &Self::Input;

    /// `TokenFilter.input`, mutably.
    fn input_mut(&mut self) -> &mut Self::Input;

    /// Java's `incrementToken()` override.
    fn increment(&mut self) -> Result<bool, AnalysisError>;

    /// Java's `reset()`; the default is `TokenFilter.reset()`: `input.reset()`.
    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input_mut().reset()
    }

    /// Java's `end()`; the default is `TokenFilter.end()`: `input.end()`.
    fn end_filter(&mut self) -> Result<(), AnalysisError> {
        self.input_mut().end()
    }

    /// Java's `close()`; the default is `TokenFilter.close()`: `input.close()`.
    fn close_filter(&mut self) -> Result<(), AnalysisError> {
        self.input_mut().close()
    }
}

impl<F: TokenFilter> TokenStream for F {
    fn attributes(&self) -> &AttributeSource {
        self.input().attributes()
    }
    fn attributes_mut(&mut self) -> &mut AttributeSource {
        self.input_mut().attributes_mut()
    }
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        self.increment()
    }
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.reset_filter()
    }
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.end_filter()
    }
    fn close(&mut self) -> Result<(), AnalysisError> {
        self.close_filter()
    }
    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        self.input_mut().as_tokenizer()
    }
}

/// `FilteringTokenFilter.accept()`: whether the current token (read from the
/// chain's attributes) is kept.
///
/// Implemented for any `FnMut(&AttributeSource) -> bool`.
pub trait Accept: Send {
    /// `accept()`.
    fn accept(&mut self, attributes: &AttributeSource) -> Result<bool, AnalysisError>;

    /// Called from the filter's `reset()`, for a predicate with state.
    fn reset(&mut self) {}
}

impl<P: FnMut(&AttributeSource) -> bool + Send> Accept for P {
    fn accept(&mut self, attributes: &AttributeSource) -> Result<bool, AnalysisError> {
        Ok(self(attributes))
    }
}

/// `org.apache.lucene.analysis.FilteringTokenFilter`: drops the tokens its
/// [`Accept`] predicate rejects, carrying their position increments onto the
/// next kept token -- and, in `end()`, onto the final increment, so trailing
/// dropped tokens still advance the position.
pub struct FilteringTokenFilter<I, P> {
    input: I,
    predicate: P,
    skipped_positions: i32,
}

impl<I: TokenStream, P: Accept> FilteringTokenFilter<I, P> {
    pub fn new(input: I, predicate: P) -> Self {
        FilteringTokenFilter {
            input,
            predicate,
            skipped_positions: 0,
        }
    }

    /// The predicate.
    pub fn predicate(&self) -> &P {
        &self.predicate
    }
}

impl<I: TokenStream, P: Accept> TokenFilter for FilteringTokenFilter<I, P> {
    type Input = I;

    fn input(&self) -> &I {
        &self.input
    }

    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        self.skipped_positions = 0;
        while self.input.increment_token()? {
            if self.predicate.accept(self.input.attributes())? {
                if self.skipped_positions != 0 {
                    let atts = self.input.attributes_mut();
                    let inc = atts
                        .position_increment()
                        .saturating_add(self.skipped_positions);
                    atts.set_position_increment(inc)?;
                }
                return Ok(true);
            }
            self.skipped_positions = self
                .skipped_positions
                .saturating_add(self.input.attributes().position_increment());
        }
        Ok(false)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.predicate.reset();
        self.skipped_positions = 0;
        Ok(())
    }

    fn end_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.end()?;
        let atts = self.input.attributes_mut();
        let inc = atts
            .position_increment()
            .saturating_add(self.skipped_positions);
        atts.set_position_increment(inc)
    }
}

/// `org.apache.lucene.analysis.CachingTokenFilter`: consumes its input on
/// the first `increment_token`, caching every token's attribute state (and
/// the state after `end()`), then replays the cache after every `reset()`.
pub struct CachingTokenFilter<I> {
    input: I,
    cache: Option<Vec<AttributeSource>>,
    next: usize,
    final_state: Option<AttributeSource>,
}

impl<I: TokenStream> CachingTokenFilter<I> {
    pub fn new(input: I) -> Self {
        CachingTokenFilter {
            input,
            cache: None,
            next: 0,
            final_state: None,
        }
    }

    /// `isCached()`.
    pub fn is_cached(&self) -> bool {
        self.cache.is_some()
    }

    fn fill_cache(&mut self) -> Result<Vec<AttributeSource>, AnalysisError> {
        let mut cache = Vec::with_capacity(64);
        while self.input.increment_token()? {
            cache.push(self.input.attributes().capture_state());
        }
        // capture final state
        self.input.end()?;
        self.final_state = Some(self.input.attributes().capture_state());
        Ok(cache)
    }
}

impl<I: TokenStream> TokenFilter for CachingTokenFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        &self.input
    }

    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if self.cache.is_none() {
            // fill cache lazily
            let cache = self.fill_cache()?;
            self.cache = Some(cache);
            self.next = 0;
        }
        let cache = self.cache.as_ref().expect("filled above");
        let Some(state) = cache.get(self.next) else {
            return Ok(false);
        };
        self.next += 1;
        self.input.attributes_mut().restore_state(state);
        Ok(true)
    }

    /// Java: propagates `reset` until the first `incrementToken`, then only
    /// rewinds the cache.
    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        if self.cache.is_none() {
            self.input.reset()
        } else {
            self.next = 0;
            Ok(())
        }
    }

    fn end_filter(&mut self) -> Result<(), AnalysisError> {
        if let Some(state) = &self.final_state {
            self.input.attributes_mut().restore_state(state);
        }
        Ok(())
    }
}

/// Drains a stream the way a consumer should (`reset`, every
/// `increment_token`, `end`, `close`), calling `f` with the attributes of
/// each token, and returns the attributes as `end()` left them.
pub fn consume(
    stream: &mut dyn TokenStream,
    mut f: impl FnMut(&AttributeSource),
) -> Result<AttributeSource, AnalysisError> {
    stream.reset()?;
    while stream.increment_token()? {
        f(stream.attributes());
    }
    stream.end()?;
    let end = stream.attributes().clone();
    stream.close()?;
    Ok(end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::StrReader;
    use crate::{LowerCaseFilter, StandardTokenizer};

    fn tokenizer(text: &str) -> StandardTokenizer {
        let mut t = StandardTokenizer::new();
        t.set_reader(Box::new(StrReader::new(text))).unwrap();
        t
    }

    fn terms(ts: &mut dyn TokenStream) -> (Vec<(String, i32)>, AttributeSource) {
        let mut out = Vec::new();
        let end = consume(ts, |a| {
            out.push((a.term().to_string(), a.position_increment()))
        })
        .unwrap();
        (out, end)
    }

    #[test]
    fn filtering_carries_skipped_increments_into_the_next_token_and_end() {
        let mut f = FilteringTokenFilter::new(tokenizer("a bb c dd e"), |a: &AttributeSource| {
            a.term().len() > 1
        });
        let (t, end) = terms(&mut f);
        assert_eq!(t, vec![("bb".to_string(), 2), ("dd".to_string(), 2)]);
        assert_eq!(end.position_increment(), 1, "the trailing \"e\"");
        assert_eq!(end.end_offset(), 11);
        let _ = f.predicate();
    }

    /// A predicate with state: keeps every other token.
    struct EveryOther(bool);

    impl Accept for EveryOther {
        fn accept(&mut self, _a: &AttributeSource) -> Result<bool, AnalysisError> {
            self.0 = !self.0;
            Ok(self.0)
        }
        fn reset(&mut self) {
            self.0 = false;
        }
    }

    #[test]
    fn a_stateful_predicate_is_reset_with_the_filter() {
        let mut f = FilteringTokenFilter::new(tokenizer("a b c"), EveryOther(true));
        let (t, _) = terms(&mut f);
        assert_eq!(t, vec![("a".to_string(), 1), ("c".to_string(), 2)]);
        f.input_mut()
            .set_reader(Box::new(StrReader::new("x y")))
            .unwrap();
        let (t, _) = terms(&mut f);
        assert_eq!(t, vec![("x".to_string(), 1)]);
    }

    #[test]
    fn caching_replays_tokens_and_the_end_state() {
        let mut c = CachingTokenFilter::new(LowerCaseFilter::new(tokenizer("A b ")));
        assert!(!c.is_cached());
        c.reset().unwrap();
        let mut first = Vec::new();
        while c.increment_token().unwrap() {
            first.push(c.attributes().term().to_string());
        }
        assert!(c.is_cached());
        c.end().unwrap();
        assert_eq!(c.attributes().end_offset(), 4);
        // a second pass replays without touching the input
        c.reset().unwrap();
        let mut second = Vec::new();
        while c.increment_token().unwrap() {
            second.push(c.attributes().term().to_string());
        }
        c.end().unwrap();
        assert_eq!(first, second);
        assert_eq!(first, vec!["a", "b"]);
        assert_eq!(c.attributes().end_offset(), 4);
        c.close().unwrap();
        assert!(c.as_tokenizer().is_some());
    }

    #[test]
    fn a_boxed_stream_forwards_everything() {
        let mut b: Box<dyn TokenStream> = Box::new(tokenizer("q r"));
        assert!(b.as_tokenizer().is_some());
        let (t, end) = terms(&mut b);
        assert_eq!(t.len(), 2);
        assert_eq!(end.end_offset(), 3);
        b.attributes_mut().set_term("z");
        assert_eq!(b.attributes().term(), "z");
    }

    /// A bare `TokenStream` with the trait's default `reset`/`end`/`close`.
    struct Bare(AttributeSource);

    impl TokenStream for Bare {
        fn attributes(&self) -> &AttributeSource {
            &self.0
        }
        fn attributes_mut(&mut self) -> &mut AttributeSource {
            &mut self.0
        }
        fn increment_token(&mut self) -> Result<bool, AnalysisError> {
            Ok(false)
        }
    }

    #[test]
    fn default_lifecycle_and_tokenizer_input_contract() {
        let mut bare = Bare(AttributeSource::new());
        let end = consume(&mut bare, |_| {}).unwrap();
        assert_eq!(end.position_increment(), 0);
        assert!(bare.as_tokenizer().is_none());

        let mut input = TokenizerInput::new();
        assert_eq!(input.correct_offset(5), 5);
        assert!(input.reader().is_err());
        input.set_reader(Box::new(StrReader::new("x"))).unwrap();
        input.reset();
        assert!(input.reader().is_ok());
        assert!(input.set_reader(Box::new(StrReader::new("y"))).is_err());
        input.close().unwrap();
        input.close().unwrap();
        assert!(input.reader().is_err());
    }
}
