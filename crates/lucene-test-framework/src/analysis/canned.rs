//! `CannedTokenStream` and `CannedBinaryTokenStream`: streams that replay a
//! fixed list of tokens.

use lucene_analysis::attributes::AttributeSource;
use lucene_analysis::{AnalysisError, TokenStream};

use super::token::Token;

/// `org.apache.lucene.tests.analysis.CannedTokenStream`.
pub struct CannedTokenStream {
    atts: AttributeSource,
    tokens: Vec<Token>,
    upto: usize,
    final_offset: i32,
    final_pos_inc: i32,
}

impl CannedTokenStream {
    /// `CannedTokenStream(tokens...)`: final offset and increment 0.
    pub fn new(tokens: Vec<Token>) -> Self {
        Self::with_end(0, 0, tokens)
    }

    /// `CannedTokenStream(finalPosInc, finalOffset, tokens...)`.
    pub fn with_end(final_pos_inc: i32, final_offset: i32, tokens: Vec<Token>) -> Self {
        CannedTokenStream {
            atts: AttributeSource::new(),
            tokens,
            upto: 0,
            final_offset,
            final_pos_inc,
        }
    }
}

impl TokenStream for CannedTokenStream {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        let Some(token) = self.tokens.get(self.upto) else {
            return Ok(false);
        };
        self.upto += 1;
        self.atts.clear_attributes();
        token.copy_to(&mut self.atts)?;
        Ok(true)
    }

    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.upto = 0;
        Ok(())
    }

    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        self.atts.set_position_increment(self.final_pos_inc)?;
        self.atts.set_offset(self.final_offset, self.final_offset)
    }

    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
}

/// `CannedBinaryTokenStream.BinaryToken`: a binary term with its position
/// increment and length (offsets stay 0, as Java's constructors leave them).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryToken {
    /// The term bytes.
    pub term: Vec<u8>,
    /// The position increment.
    pub pos_inc: i32,
    /// The position length.
    pub pos_len: i32,
    /// The start offset.
    pub start_offset: i32,
    /// The end offset.
    pub end_offset: i32,
}

impl BinaryToken {
    /// `BinaryToken(term)`.
    pub fn new(term: &[u8]) -> Self {
        Self::with_positions(term, 1, 1)
    }

    /// `BinaryToken(term, posInc, posLen)`.
    pub fn with_positions(term: &[u8], pos_inc: i32, pos_len: i32) -> Self {
        BinaryToken {
            term: term.to_vec(),
            pos_inc,
            pos_len,
            start_offset: 0,
            end_offset: 0,
        }
    }
}

/// `org.apache.lucene.tests.analysis.CannedBinaryTokenStream`: binary terms
/// through the `BytesTermAttribute` (the attribute source's bytes term).
pub struct CannedBinaryTokenStream {
    atts: AttributeSource,
    tokens: Vec<BinaryToken>,
    upto: usize,
}

impl CannedBinaryTokenStream {
    /// `CannedBinaryTokenStream(tokens...)`.
    pub fn new(tokens: Vec<BinaryToken>) -> Self {
        CannedBinaryTokenStream {
            atts: AttributeSource::new(),
            tokens,
            upto: 0,
        }
    }
}

impl TokenStream for CannedBinaryTokenStream {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        let Some(token) = self.tokens.get(self.upto) else {
            return Ok(false);
        };
        self.upto += 1;
        self.atts.clear_attributes();
        self.atts.set_bytes_term(Some(token.term.clone()));
        self.atts.set_position_increment(token.pos_inc)?;
        self.atts.set_position_length(token.pos_len)?;
        self.atts.set_offset(token.start_offset, token.end_offset)?;
        Ok(true)
    }

    /// Java's base `reset()`, which does not rewind (as Java's does not).
    fn reset(&mut self) -> Result<(), AnalysisError> {
        Ok(())
    }

    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::token_stream::consume;

    #[test]
    fn canned_replays_tokens_and_the_end_state() {
        let mut ts = CannedTokenStream::with_end(
            2,
            9,
            vec![Token::new("a", 0, 1), Token::with_pos_len("b", 0, 0, 3, 2)],
        );
        for _ in 0..2 {
            let mut got = Vec::new();
            let end = consume(&mut ts, |a| {
                got.push((
                    a.term().to_string(),
                    a.position_increment(),
                    a.position_length(),
                ))
            })
            .unwrap();
            assert_eq!(got, [("a".into(), 1, 1), ("b".into(), 0, 2)]);
            assert_eq!((end.position_increment(), end.end_offset()), (2, 9));
        }
        assert!(ts.conditional_root().is_none());
    }

    #[test]
    fn binary_tokens_land_in_the_bytes_term() {
        let mut ts = CannedBinaryTokenStream::new(vec![
            BinaryToken::new(&[0xff, 0]),
            BinaryToken::with_positions(b"x", 0, 3),
        ]);
        let mut got = Vec::new();
        consume(&mut ts, |a| {
            got.push((
                a.term_bytes().to_vec(),
                a.position_increment(),
                a.position_length(),
            ))
        })
        .unwrap();
        assert_eq!(got, [(vec![0xff, 0], 1, 1), (b"x".to_vec(), 0, 3)]);
        assert!(ts.conditional_root().is_none());
        ts.attributes_mut().set_term("q");
        assert_eq!(ts.attributes().term(), "q");
    }
}
