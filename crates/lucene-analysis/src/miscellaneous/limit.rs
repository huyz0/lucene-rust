//! `LimitTokenCountFilter`, `LimitTokenOffsetFilter`,
//! `LimitTokenPositionFilter`.

use crate::token_stream::{TokenFilter, TokenStream};
use crate::AnalysisError;

/// `org.apache.lucene.analysis.miscellaneous.LimitTokenCountFilter`.
pub struct LimitTokenCountFilter<I> {
    input: I,
    max_token_count: i32,
    consume_all_tokens: bool,
    token_count: i32,
    exhausted: bool,
}

impl<I: TokenStream> LimitTokenCountFilter<I> {
    /// `new LimitTokenCountFilter(TokenStream, int, boolean consumeAllTokens)`.
    pub fn new(
        input: I,
        max_token_count: i32,
        consume_all_tokens: bool,
    ) -> Result<Self, AnalysisError> {
        if max_token_count < 1 {
            return Err(AnalysisError::IllegalArgument(
                "maxTokenCount must be greater than zero".into(),
            ));
        }
        Ok(LimitTokenCountFilter {
            input,
            max_token_count,
            consume_all_tokens,
            token_count: 0,
            exhausted: false,
        })
    }
}

impl<I: TokenStream> TokenFilter for LimitTokenCountFilter<I> {
    crate::filter_input!();
    // Java: LimitTokenCountFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if self.exhausted {
            Ok(false)
        } else if self.token_count < self.max_token_count {
            if self.input.increment_token()? {
                self.token_count += 1;
                Ok(true)
            } else {
                self.exhausted = true;
                Ok(false)
            }
        } else {
            while self.consume_all_tokens && self.input.increment_token()? {}
            Ok(false)
        }
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.token_count = 0;
        self.exhausted = false;
        Ok(())
    }
}

/// `org.apache.lucene.analysis.miscellaneous.LimitTokenOffsetFilter`.
pub struct LimitTokenOffsetFilter<I> {
    input: I,
    max_start_offset: i32,
    consume_all_tokens: bool,
}

impl<I: TokenStream> LimitTokenOffsetFilter<I> {
    /// `new LimitTokenOffsetFilter(TokenStream, int, boolean consumeAllTokens)`.
    pub fn new(
        input: I,
        max_start_offset: i32,
        consume_all_tokens: bool,
    ) -> Result<Self, AnalysisError> {
        if max_start_offset < 0 {
            return Err(AnalysisError::IllegalArgument(
                "maxStartOffset must be >= zero".into(),
            ));
        }
        Ok(LimitTokenOffsetFilter {
            input,
            max_start_offset,
            consume_all_tokens,
        })
    }
}

impl<I: TokenStream> TokenFilter for LimitTokenOffsetFilter<I> {
    crate::filter_input!();
    // Java: LimitTokenOffsetFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        if self.input.attributes().start_offset() <= self.max_start_offset {
            return Ok(true);
        }
        if self.consume_all_tokens {
            while self.input.increment_token()? {}
        }
        Ok(false)
    }
}

/// `org.apache.lucene.analysis.miscellaneous.LimitTokenPositionFilter`.
pub struct LimitTokenPositionFilter<I> {
    input: I,
    max_token_position: i32,
    consume_all_tokens: bool,
    token_position: i32,
    exhausted: bool,
}

impl<I: TokenStream> LimitTokenPositionFilter<I> {
    /// `new LimitTokenPositionFilter(TokenStream, int, boolean consumeAllTokens)`.
    pub fn new(
        input: I,
        max_token_position: i32,
        consume_all_tokens: bool,
    ) -> Result<Self, AnalysisError> {
        if max_token_position < 1 {
            return Err(AnalysisError::IllegalArgument(
                "maxTokenPosition must be greater than zero".into(),
            ));
        }
        Ok(LimitTokenPositionFilter {
            input,
            max_token_position,
            consume_all_tokens,
            token_position: 0,
            exhausted: false,
        })
    }
}

impl<I: TokenStream> TokenFilter for LimitTokenPositionFilter<I> {
    crate::filter_input!();
    // Java: LimitTokenPositionFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if self.exhausted {
            return Ok(false);
        }
        if self.input.increment_token()? {
            self.token_position = self
                .token_position
                .saturating_add(self.input.attributes().position_increment());
            if self.token_position <= self.max_token_position {
                Ok(true)
            } else {
                while self.consume_all_tokens && self.input.increment_token()? {}
                self.exhausted = true;
                Ok(false)
            }
        } else {
            self.exhausted = true;
            Ok(false)
        }
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.token_position = 0;
        self.exhausted = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::{render, Canned};

    const IN: &str = "a:0:1:1:1 b:2:3:1:1 c:4:5:2:1 d:6:7:1:1|9|1";

    #[test]
    fn limits() {
        for all in [false, true] {
            let mut f = LimitTokenCountFilter::new(Canned::parse(IN), 2, all).unwrap();
            assert_eq!(render(&mut f), "a:0:1:1:1 b:2:3:1:1|9|1");
            let mut f = LimitTokenCountFilter::new(Canned::parse(IN), 9, all).unwrap();
            assert_eq!(render(&mut f), format!("{}", IN));
            let mut f = LimitTokenOffsetFilter::new(Canned::parse(IN), 3, all).unwrap();
            assert_eq!(render(&mut f), "a:0:1:1:1 b:2:3:1:1|9|1");
            let mut f = LimitTokenPositionFilter::new(Canned::parse(IN), 3, all).unwrap();
            assert_eq!(render(&mut f), "a:0:1:1:1 b:2:3:1:1|9|1");
            let mut f = LimitTokenPositionFilter::new(Canned::parse(IN), 9, all).unwrap();
            assert_eq!(render(&mut f), IN);
            let mut f = LimitTokenOffsetFilter::new(Canned::parse(IN), 9, all).unwrap();
            assert_eq!(render(&mut f), IN);
        }
        assert!(LimitTokenCountFilter::new(Canned::parse(""), 0, false).is_err());
        assert!(LimitTokenOffsetFilter::new(Canned::parse(""), -1, false).is_err());
        assert!(LimitTokenPositionFilter::new(Canned::parse(""), 0, false).is_err());
    }

    #[test]
    fn exhausted_stays_exhausted() {
        let mut f = LimitTokenCountFilter::new(Canned::parse("a:0:1:1:1"), 5, false).unwrap();
        f.reset().unwrap();
        assert!(f.increment_token().unwrap());
        assert!(!f.increment_token().unwrap());
        assert!(!f.increment_token().unwrap());
        let mut f = LimitTokenPositionFilter::new(Canned::parse("a:0:1:1:1"), 5, false).unwrap();
        f.reset().unwrap();
        assert!(f.increment_token().unwrap());
        assert!(!f.increment_token().unwrap());
        assert!(!f.increment_token().unwrap());
    }
}
