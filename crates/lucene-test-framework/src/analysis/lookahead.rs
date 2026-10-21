//! `LookaheadTokenFilter` and the two test filters built on it,
//! `MockRandomLookaheadTokenFilter` and `MockGraphTokenFilter`.
//!
//! Java's abstract class is a [`LookaheadTokenFilter`] value that a
//! "subclass" embeds next to its own state; Java's overridable
//! `afterPosition()` is the [`LookaheadHooks`] the subclass passes to
//! [`LookaheadTokenFilter::next_token`], so the hook can reach both the
//! lookahead state and the subclass's fields without aliasing. Java's
//! per-position subclass data (`Position` subclasses) is the `extra` field
//! of [`Position`]. Java's `assert`s stay assertions (they panic).

use lucene_analysis::attributes::{AttributeSource, State};
use lucene_analysis::util::rolling_buffer::{Resettable, RollingBuffer};
use lucene_analysis::{AnalysisError, TokenFilter, TokenStream};
use lucene_util::java_random::JavaRandom;

use crate::util::test_util;

/// `LookaheadTokenFilter.Position`: the state held for one position, plus
/// a subclass's own data in `extra`.
#[derive(Debug)]
pub struct Position<X = ()> {
    /// Buffered input tokens at this position.
    pub input_tokens: Vec<State>,
    /// Next buffered token to be returned to the consumer.
    pub next_read: usize,
    /// Any token leaving from this position should have this start offset.
    pub start_offset: i32,
    /// Any token arriving at this position should have this end offset.
    pub end_offset: i32,
    /// A subclass's per-position data; reset to its default with the
    /// position.
    pub extra: X,
}

impl<X: Default> Default for Position<X> {
    fn default() -> Self {
        Position {
            input_tokens: Vec::new(),
            next_read: 0,
            start_offset: -1,
            end_offset: -1,
            extra: X::default(),
        }
    }
}

impl<X: Default> Resettable for Position<X> {
    fn reset(&mut self) {
        self.input_tokens.clear();
        self.next_read = 0;
        self.start_offset = -1;
        self.end_offset = -1;
        self.extra = X::default();
    }
}

impl<X> Position<X> {
    /// `add(state)`.
    pub fn add(&mut self, state: State) {
        self.input_tokens.push(state);
    }

    /// `nextState()`.
    pub fn next_state(&mut self) -> &State {
        assert!(self.next_read < self.input_tokens.len());
        self.next_read += 1;
        &self.input_tokens[self.next_read - 1]
    }
}

/// Java's overridable `afterPosition()`: called when every input token
/// leaving a position has been returned. Call
/// [`LookaheadTokenFilter::insert_token`] and then set the attributes to
/// inject a token starting at this position.
pub trait LookaheadHooks<I: TokenStream, X: Default> {
    /// `afterPosition()`.
    ///
    /// # Errors
    /// What reading ahead from the input returns.
    fn after_position(&mut self, la: &mut LookaheadTokenFilter<I, X>) -> Result<(), AnalysisError>;
}

/// Java's default `afterPosition()`, which does nothing.
pub struct NoHooks;

impl<I: TokenStream, X: Default> LookaheadHooks<I, X> for NoHooks {
    fn after_position(&mut self, _: &mut LookaheadTokenFilter<I, X>) -> Result<(), AnalysisError> {
        Ok(())
    }
}

/// `org.apache.lucene.tests.analysis.LookaheadTokenFilter`.
pub struct LookaheadTokenFilter<I, X: Default = ()> {
    input: I,
    /// Position of the last read input token.
    pub input_pos: i32,
    /// Position of the next possible output token to return.
    pub output_pos: i32,
    /// True once the input has returned its last token.
    pub end: bool,
    token_pending: bool,
    insert_pending: bool,
    /// The buffered positions.
    pub positions: RollingBuffer<Position<X>>,
}

impl<I: TokenStream, X: Default> LookaheadTokenFilter<I, X> {
    /// `LookaheadTokenFilter(input)`.
    pub fn new(input: I) -> Self {
        LookaheadTokenFilter {
            input,
            input_pos: 0,
            output_pos: 0,
            end: false,
            token_pending: false,
            insert_pending: false,
            positions: RollingBuffer::new(),
        }
    }

    /// The wrapped stream.
    pub fn input(&self) -> &I {
        &self.input
    }

    /// The wrapped stream, mutably.
    pub fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    /// The chain's attributes.
    pub fn attributes_mut(&mut self) -> &mut AttributeSource {
        self.input.attributes_mut()
    }

    fn flush_pending(&mut self) {
        if self.token_pending {
            let state = self.input.attributes().capture_state();
            self.positions.get(self.input_pos).add(state);
            self.token_pending = false;
        }
    }

    /// `insertToken()`: call only from [`LookaheadHooks::after_position`],
    /// then set the token's attributes.
    pub fn insert_token(&mut self) {
        self.flush_pending();
        assert!(!self.insert_pending);
        self.insert_pending = true;
    }

    /// `peekToken()`: reads one more input token into the buffer; whether
    /// there was one.
    ///
    /// # Errors
    /// What the input returns.
    pub fn peek_token(&mut self) -> Result<bool, AnalysisError> {
        assert!(!self.end);
        assert!(self.input_pos == -1 || self.output_pos <= self.input_pos);
        self.flush_pending();
        let got_token = self.input.increment_token()?;
        if got_token {
            let atts = self.input.attributes();
            let (pos_inc, pos_len) = (atts.position_increment(), atts.position_length());
            let (start_offset, end_offset) = (atts.start_offset(), atts.end_offset());
            self.input_pos += pos_inc;
            assert!(self.input_pos >= 0);
            let input_pos = self.input_pos;
            let start_pos_data = self.positions.get(input_pos);
            if start_pos_data.start_offset == -1 {
                start_pos_data.start_offset = start_offset;
            } else {
                // Make sure our input isn't messing up offsets:
                assert!(
                    start_pos_data.start_offset == start_offset,
                    "prev startOffset={} vs new startOffset={start_offset} inputPos={input_pos}",
                    start_pos_data.start_offset
                );
            }
            let end_pos_data = self.positions.get(input_pos + pos_len);
            if end_pos_data.end_offset == -1 {
                end_pos_data.end_offset = end_offset;
            } else {
                assert!(
                    end_pos_data.end_offset == end_offset,
                    "prev endOffset={} vs new endOffset={end_offset} inputPos={input_pos}",
                    end_pos_data.end_offset
                );
            }
            self.token_pending = true;
        } else {
            self.end = true;
        }
        Ok(got_token)
    }

    /// `nextToken()`: call when done looking ahead; sets the next token to
    /// return, and whether there is one.
    ///
    /// # Errors
    /// What the input or the hooks return.
    pub fn next_token<H: LookaheadHooks<I, X>>(
        &mut self,
        hooks: &mut H,
    ) -> Result<bool, AnalysisError> {
        // While loop here in case we have to skip over a hole from the input:
        loop {
            let pos = self.positions.get(self.output_pos);
            let (has_buffered, has_start) = (
                pos.next_read < pos.input_tokens.len(),
                pos.start_offset != -1,
            );
            // See if we have a previously buffered token to return at the
            // current position:
            if has_buffered {
                // This position has buffered tokens to serve up:
                self.flush_pending();
                let state = self.positions.get(self.output_pos).next_state().clone();
                self.input.attributes_mut().restore_state(&state);
                return Ok(true);
            }
            if self.input_pos == -1 || self.output_pos == self.input_pos {
                // No more buffered tokens: we may still get input tokens at
                // this position.
                if self.token_pending {
                    // Fast path: just return the token we had just incr'd,
                    // without having captured/restored its state:
                    self.token_pending = false;
                    return Ok(true);
                } else if self.end || !self.peek_token()? {
                    hooks.after_position(self)?;
                    if self.insert_pending {
                        // Subclass inserted a token at this same position:
                        self.assert_inserted_token_consistent();
                        self.insert_pending = false;
                        return Ok(true);
                    }
                    return Ok(false);
                }
            } else {
                if has_start {
                    // This position had at least one token leaving:
                    hooks.after_position(self)?;
                    if self.insert_pending {
                        self.assert_inserted_token_consistent();
                        self.insert_pending = false;
                        return Ok(true);
                    }
                }
                // Done with this position; move on:
                self.output_pos += 1;
                self.positions.free_before(self.output_pos);
            }
        }
    }

    /// `insertedTokenConsistent()`: an inserted token must have looked ahead
    /// far enough to know its end offset.
    fn assert_inserted_token_consistent(&mut self) {
        let atts = self.input.attributes();
        let (pos_len, end_offset) = (atts.position_length(), atts.end_offset());
        let end_pos_data = self.positions.get(self.output_pos + pos_len);
        assert!(end_pos_data.end_offset != -1);
        assert!(
            end_offset == end_pos_data.end_offset,
            "offsetAtt.endOffset={end_offset} vs expected={}",
            end_pos_data.end_offset
        );
    }

    /// `reset()`.
    ///
    /// # Errors
    /// What the input's reset returns.
    pub fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.positions.reset();
        self.input_pos = -1;
        self.output_pos = 0;
        self.token_pending = false;
        self.end = false;
        Ok(())
    }
}

/// `org.apache.lucene.tests.analysis.MockRandomLookaheadTokenFilter`:
/// randomly peeks at future tokens.
pub struct MockRandomLookaheadTokenFilter<I> {
    la: LookaheadTokenFilter<I>,
    hooks: RandomPeek,
    seed: i64,
}

struct RandomPeek(JavaRandom);

impl<I: TokenStream> LookaheadHooks<I, ()> for RandomPeek {
    fn after_position(&mut self, la: &mut LookaheadTokenFilter<I>) -> Result<(), AnalysisError> {
        if !la.end && self.0.next_int_bounded(4) == 2 {
            la.peek_token()?;
        }
        Ok(())
    }
}

impl<I: TokenStream> MockRandomLookaheadTokenFilter<I> {
    /// `MockRandomLookaheadTokenFilter(random, in)`: draws its seed from
    /// `random`.
    pub fn new(random: &mut JavaRandom, input: I) -> Self {
        let seed = random.next_long();
        MockRandomLookaheadTokenFilter {
            la: LookaheadTokenFilter::new(input),
            hooks: RandomPeek(JavaRandom::new(seed)),
            seed,
        }
    }
}

impl<I: TokenStream> TokenFilter for MockRandomLookaheadTokenFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        self.la.input()
    }

    fn input_mut(&mut self) -> &mut I {
        self.la.input_mut()
    }

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.la.end {
            while self.hooks.0.next_int_bounded(3) == 1 {
                if !self.la.peek_token()? {
                    break;
                }
            }
        }
        self.la.next_token(&mut self.hooks)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.la.reset()?;
        self.hooks.0.set_seed(self.seed);
        Ok(())
    }
}

/// `org.apache.lucene.tests.analysis.MockGraphTokenFilter`: randomly
/// inserts overlapping (`posInc=0`) tokens, sometimes with `posLength > 1`.
pub struct MockGraphTokenFilter<I> {
    la: LookaheadTokenFilter<I>,
    /// `null` until `reset()` and after `close()`.
    hooks: Option<GraphInsert>,
    seed: i64,
}

struct GraphInsert(JavaRandom);

impl<I: TokenStream> LookaheadHooks<I, ()> for GraphInsert {
    fn after_position(&mut self, la: &mut LookaheadTokenFilter<I>) -> Result<(), AnalysisError> {
        let random = &mut self.0;
        if random.next_int_bounded(7) != 5 {
            return Ok(());
        }
        let pos_length = test_util::next_int(random, 1, 5);
        let end_pos = la.output_pos + pos_length;
        // Look ahead as needed until we figure out the right endOffset:
        while !la.end && la.positions.get(end_pos).end_offset == -1 && la.input_pos <= end_pos {
            if !la.peek_token()? {
                break;
            }
        }
        let end_offset = la.positions.get(end_pos).end_offset;
        if end_offset != -1 {
            // Notify the lookahead that we are injecting a token:
            la.insert_token();
            let start_offset = la.positions.get(la.output_pos).start_offset;
            let term = test_util::random_unicode_string(random);
            let atts = la.attributes_mut();
            atts.clear_attributes();
            atts.set_position_length(pos_length)?;
            atts.term_mut().push_str(&term);
            atts.set_position_increment(0)?;
            atts.set_offset(start_offset, end_offset)?;
        }
        // Otherwise either the tokens ended before our posLength, or our
        // posLength ended inside a hole from the input: skip the insert.
        Ok(())
    }
}

impl<I: TokenStream> MockGraphTokenFilter<I> {
    /// `MockGraphTokenFilter(random, input)`: draws its seed from `random`.
    pub fn new(random: &mut JavaRandom, input: I) -> Self {
        MockGraphTokenFilter {
            la: LookaheadTokenFilter::new(input),
            hooks: None,
            seed: random.next_long(),
        }
    }
}

impl<I: TokenStream> TokenFilter for MockGraphTokenFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        self.la.input()
    }

    fn input_mut(&mut self) -> &mut I {
        self.la.input_mut()
    }

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        match self.hooks.as_mut() {
            Some(hooks) => self.la.next_token(hooks),
            None => Err(AnalysisError::IllegalState(
                "incrementToken called in wrong state!".to_string(),
            )),
        }
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.la.reset()?;
        // "Deterministically random": BaseTokenStreamTestCase pulls tokens
        // twice on the same input and asserts they are the same.
        self.hooks = Some(GraphInsert(JavaRandom::new(self.seed)));
        Ok(())
    }

    fn close_filter(&mut self) -> Result<(), AnalysisError> {
        self.la.input_mut().close()?;
        self.hooks = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::canned::CannedTokenStream;
    use crate::analysis::token::Token;
    use lucene_analysis::token_stream::consume;

    fn terms<T: TokenStream>(ts: &mut T) -> Vec<(String, i32, i32, i32, i32)> {
        let mut out = Vec::new();
        consume(ts, |a| {
            out.push((
                a.term().to_string(),
                a.position_increment(),
                a.position_length(),
                a.start_offset(),
                a.end_offset(),
            ))
        })
        .unwrap();
        out
    }

    fn canned() -> CannedTokenStream {
        CannedTokenStream::new(vec![
            Token::new("a", 0, 1),
            Token::new("b", 2, 3),
            Token::with_pos_inc("c", 2, 4, 5),
            Token::new("d", 6, 7),
        ])
    }

    #[test]
    fn random_lookahead_passes_tokens_through() {
        let expected = terms(&mut canned());
        for seed in 0..50 {
            let mut f = MockRandomLookaheadTokenFilter::new(&mut JavaRandom::new(seed), canned());
            assert_eq!(terms(&mut f), expected, "seed {seed}");
            // reset replays the same draws
            assert_eq!(terms(&mut f), expected, "seed {seed}");
        }
    }

    #[test]
    fn graph_filter_inserts_consistent_tokens_and_replays() {
        let mut inserted = 0;
        for seed in 0..50 {
            let mut f = MockGraphTokenFilter::new(&mut JavaRandom::new(seed), canned());
            let first = terms(&mut f);
            assert_eq!(terms(&mut f), first, "seed {seed}");
            inserted += first.len() - 4;
            let originals: Vec<_> = first
                .iter()
                .filter(|t| t.0.len() == 1 && t.0 <= "d".into())
                .collect();
            assert!(originals.len() >= 4);
        }
        assert!(inserted > 0, "no seed inserted a token");
    }

    #[test]
    fn graph_filter_refuses_before_reset_and_after_close() {
        let mut f = MockGraphTokenFilter::new(&mut JavaRandom::new(1), canned());
        assert!(matches!(
            f.increment_token(),
            Err(AnalysisError::IllegalState(_))
        ));
        f.reset().unwrap();
        assert!(f.increment_token().unwrap());
        f.close().unwrap();
        let e = f.increment_token().unwrap_err();
        assert_eq!(
            e,
            AnalysisError::IllegalState("incrementToken called in wrong state!".into())
        );
    }

    #[test]
    fn position_bookkeeping() {
        let mut p: Position<u8> = Position {
            extra: 3,
            ..Position::default()
        };
        p.add(AttributeSource::new());
        let _ = p.next_state();
        p.reset();
        assert_eq!(
            (
                p.input_tokens.len(),
                p.next_read,
                p.start_offset,
                p.end_offset,
                p.extra
            ),
            (0, 0, -1, -1, 0)
        );
    }

    #[test]
    #[should_panic(expected = "prev startOffset=0 vs new startOffset=1")]
    fn inconsistent_input_offsets_trip_the_assertion() {
        let ts = CannedTokenStream::new(vec![
            Token::new("a", 0, 1),
            Token::with_pos_inc("b", 0, 1, 1),
        ]);
        let mut la: LookaheadTokenFilter<_> = LookaheadTokenFilter::new(ts);
        la.reset().unwrap();
        while la.peek_token().unwrap() {}
    }

    #[test]
    #[should_panic(expected = "prev endOffset=1 vs new endOffset=2")]
    fn inconsistent_end_offsets_trip_the_assertion() {
        let ts = CannedTokenStream::new(vec![
            Token::new("a", 0, 1),
            Token::with_pos_inc("b", 0, 0, 2),
        ]);
        let mut la: LookaheadTokenFilter<_> = LookaheadTokenFilter::new(ts);
        la.reset().unwrap();
        while la.peek_token().unwrap() {}
    }

    /// Inserts a token whose end offset does not match the lookahead's.
    struct BadInsert;
    impl<I: TokenStream> LookaheadHooks<I, ()> for BadInsert {
        fn after_position(
            &mut self,
            la: &mut LookaheadTokenFilter<I>,
        ) -> Result<(), AnalysisError> {
            if la.output_pos == 0 && !la.end {
                la.peek_token()?;
                la.insert_token();
                let atts = la.attributes_mut();
                atts.clear_attributes();
                atts.set_term("x");
                atts.set_position_increment(0)?;
                atts.set_offset(0, 99)?;
            }
            Ok(())
        }
    }

    #[test]
    #[should_panic(expected = "offsetAtt.endOffset=99 vs expected=1")]
    fn an_inconsistent_insert_trips_the_assertion() {
        let mut la: LookaheadTokenFilter<_> = LookaheadTokenFilter::new(canned());
        la.reset().unwrap();
        let mut hooks = BadInsert;
        while la.next_token(&mut hooks).unwrap() {}
    }

    #[test]
    fn no_hooks_is_a_pass_through() {
        let expected = terms(&mut canned());
        let mut la: LookaheadTokenFilter<_> = LookaheadTokenFilter::new(canned());
        la.reset().unwrap();
        let mut got = 0;
        while la.next_token(&mut NoHooks).unwrap() {
            got += 1;
        }
        assert_eq!(got, expected.len());
    }
}
