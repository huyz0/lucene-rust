//! `ValidatingTokenFilter`: checks, as tokens pass through it, that the
//! first position increment is positive, offsets never go backwards, and the
//! graph's offsets are consistent (every token leaving a position has the
//! same start offset, every token arriving at one the same end offset).
//!
//! Differs: Java finds the attributes with `getAttribute` and skips a check
//! whose attribute is absent; every attribute is present here, so every
//! check runs. On failure Java prints the chain's validating filters'
//! recent tokens to stderr (`dumpValidatingTokenFilters`); the error here
//! carries this filter's [`ValidatingTokenFilter::dump`], since a filter
//! cannot see the filters above it.

use std::collections::{HashMap, VecDeque};

use lucene_analysis::{AnalysisError, TokenFilter, TokenStream};

use super::token::Token;

/// `MAX_DEBUG_TOKENS`: how many recent tokens the dump keeps.
const MAX_DEBUG_TOKENS: usize = 20;

/// `org.apache.lucene.tests.analysis.ValidatingTokenFilter`.
pub struct ValidatingTokenFilter<I> {
    input: I,
    name: String,
    pos: i32,
    last_start_offset: i32,
    pos_to_start_offset: HashMap<i32, i32>,
    pos_to_end_offset: HashMap<i32, i32>,
    tokens: VecDeque<Token>,
}

impl<I: TokenStream> ValidatingTokenFilter<I> {
    /// `ValidatingTokenFilter(in, name)`.
    pub fn new(input: I, name: &str) -> Self {
        ValidatingTokenFilter {
            input,
            name: name.to_string(),
            pos: 0,
            last_start_offset: 0,
            pos_to_start_offset: HashMap::new(),
            pos_to_end_offset: HashMap::new(),
            tokens: VecDeque::new(),
        }
    }

    /// `dump()`: the name and the last 20 tokens, as
    /// `term<[start-end] +posInc>`.
    pub fn dump(&self) -> String {
        let mut buf = format!("{}: ", self.name);
        for t in &self.tokens {
            buf.push_str(&format!(
                "{}<[{}-{}] +{}> ",
                t,
                t.start_offset(),
                t.end_offset(),
                t.position_increment()
            ));
        }
        buf
    }

    fn fail(&self, message: String) -> AnalysisError {
        AnalysisError::IllegalState(format!("{}: {message}\n{}", self.name, self.dump()))
    }
}

impl<I: TokenStream> TokenFilter for ValidatingTokenFilter<I> {
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
        let atts = self.input.attributes();
        let pos_inc = atts.position_increment();
        let start_offset = atts.start_offset();
        let end_offset = atts.end_offset();
        let pos_len = atts.position_length();
        let term = atts.term().to_string();

        if self.tokens.len() == MAX_DEBUG_TOKENS {
            self.tokens.pop_front();
        }
        self.tokens.push_back(Token::with_pos_inc(
            &term,
            pos_inc,
            start_offset,
            end_offset,
        ));

        self.pos += pos_inc;
        if self.pos == -1 {
            return Err(self.fail("first posInc must be > 0".to_string()));
        }
        if start_offset < self.last_start_offset {
            return Err(self.fail(format!(
                "offsets must not go backwards startOffset={start_offset} is < lastStartOffset={}",
                self.last_start_offset
            )));
        }
        self.last_start_offset = start_offset;

        match self.pos_to_start_offset.get(&self.pos) {
            // First time we've seen a token leaving from this position:
            None => {
                self.pos_to_start_offset.insert(self.pos, start_offset);
            }
            // We've seen a token leaving from this position before; verify the startOffset is
            // the same:
            Some(&old) if old != start_offset => {
                return Err(self.fail(format!(
                    "inconsistent startOffset at pos={}: {old} vs {start_offset}; token={term}",
                    self.pos
                )));
            }
            Some(_) => {}
        }
        let end_pos = self.pos + pos_len;
        match self.pos_to_end_offset.get(&end_pos) {
            None => {
                self.pos_to_end_offset.insert(end_pos, end_offset);
            }
            Some(&old) if old != end_offset => {
                return Err(self.fail(format!(
                    "inconsistent endOffset at pos={end_pos}: {old} vs {end_offset}; token={term}"
                )));
            }
            Some(_) => {}
        }
        Ok(true)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.pos = -1;
        self.pos_to_start_offset.clear();
        self.pos_to_end_offset.clear();
        self.last_start_offset = 0;
        self.tokens.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::canned::CannedTokenStream;
    use crate::analysis::token::Token;
    use lucene_analysis::token_stream::consume;

    fn validate(tokens: Vec<Token>) -> Result<usize, String> {
        let mut f = ValidatingTokenFilter::new(CannedTokenStream::new(tokens), "v");
        let mut n = 0;
        consume(&mut f, |_| n += 1)
            .map(|_| n)
            .map_err(|e| e.to_string())
    }

    #[test]
    fn a_consistent_graph_passes() {
        assert_eq!(
            validate(vec![
                Token::new("a", 0, 1),
                Token::with_pos_len("ab", 0, 0, 3, 2),
                Token::new("b", 2, 3),
            ]),
            Ok(3)
        );
    }

    #[test]
    fn every_check_fires() {
        let e = validate(vec![Token::with_pos_inc("a", 0, 0, 1)]).unwrap_err();
        assert!(e.contains("v: first posInc must be > 0"), "{e}");
        assert!(e.contains("a<[0-1] +0>"), "{e}");
        let e = validate(vec![Token::new("a", 2, 3), Token::new("b", 1, 4)]).unwrap_err();
        assert!(e.contains("offsets must not go backwards startOffset=1 is < lastStartOffset=2"));
        let e = validate(vec![
            Token::new("a", 0, 1),
            Token::with_pos_inc("b", 0, 1, 1),
        ])
        .unwrap_err();
        assert!(
            e.contains("inconsistent startOffset at pos=0: 0 vs 1; token=b"),
            "{e}"
        );
        let e = validate(vec![
            Token::new("a", 0, 1),
            Token::with_pos_inc("b", 0, 0, 2),
        ])
        .unwrap_err();
        assert!(
            e.contains("inconsistent endOffset at pos=1: 1 vs 2; token=b"),
            "{e}"
        );
    }

    #[test]
    fn the_dump_keeps_the_last_twenty_tokens() {
        let tokens: Vec<Token> = (0..25)
            .map(|i| Token::new(&format!("t{i}"), i, i + 1))
            .collect();
        let mut f = ValidatingTokenFilter::new(CannedTokenStream::new(tokens), "v");
        consume(&mut f, |_| {}).unwrap();
        let d = f.dump();
        assert!(d.starts_with("v: t5<[5-6] +1> "), "{d}");
        assert!(!d.contains("t4<"));
    }
}
