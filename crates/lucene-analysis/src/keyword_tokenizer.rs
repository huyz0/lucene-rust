//! analysis-common's `org.apache.lucene.analysis.core.KeywordTokenizer`,
//! ported ahead of M11 because [`crate::Analyzer::keyword`] is built on it.

use crate::attributes::AttributeSource;
use crate::reader::CharReader;
use crate::token_stream::{TokenStream, Tokenizer, TokenizerInput};
use crate::AnalysisError;

/// `KeywordTokenizer.DEFAULT_BUFFER_SIZE`.
const DEFAULT_BUFFER_SIZE: usize = 256;

/// `KeywordTokenizer`: the whole input as one token.
pub struct KeywordTokenizer {
    atts: AttributeSource,
    input: TokenizerInput,
    done: bool,
    final_offset: i32,
    buffer: Vec<u16>,
}

impl Default for KeywordTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl KeywordTokenizer {
    pub fn new() -> Self {
        KeywordTokenizer {
            atts: AttributeSource::new(),
            input: TokenizerInput::new(),
            done: false,
            final_offset: 0,
            buffer: Vec::with_capacity(DEFAULT_BUFFER_SIZE),
        }
    }
}

impl TokenStream for KeywordTokenizer {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }
    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        if self.done {
            return Ok(false);
        }
        self.atts.clear_attributes();
        self.done = true;
        self.buffer.clear();
        let reader = self.input.reader()?;
        let mut chunk = [0u16; DEFAULT_BUFFER_SIZE];
        loop {
            let n = reader.read(&mut chunk)?;
            if n == 0 {
                break;
            }
            self.buffer.extend_from_slice(&chunk[..n]);
        }
        self.atts.set_term_utf16(&self.buffer);
        let upto = self.buffer.len() as i32;
        self.final_offset = self.input.correct_offset(upto);
        let start = self.input.correct_offset(0);
        self.atts.set_offset(start, self.final_offset)?;
        Ok(true)
    }
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        // set final offset
        self.atts.set_offset(self.final_offset, self.final_offset)
    }
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset();
        self.done = false;
        Ok(())
    }
    fn close(&mut self) -> Result<(), AnalysisError> {
        self.input.close()
    }
    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        Some(self)
    }
}

impl Tokenizer for KeywordTokenizer {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.input.set_reader(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::StrReader;
    use crate::token_stream::consume;

    #[test]
    fn whole_input_is_one_token_with_java_offsets() {
        let mut t = KeywordTokenizer::default();
        let long = "é".repeat(600) + "😀";
        t.set_reader(Box::new(StrReader::new(long.as_str())))
            .unwrap();
        let mut toks = Vec::new();
        let end = consume(&mut t, |a| {
            toks.push((a.term().to_string(), a.start_offset(), a.end_offset()))
        })
        .unwrap();
        assert_eq!(toks, vec![(long.clone(), 0, 602)]);
        assert_eq!((end.start_offset(), end.end_offset()), (602, 602));
        // empty input: one empty token
        t.set_reader(Box::new(StrReader::new(""))).unwrap();
        let mut n = 0;
        consume(&mut t, |a| {
            assert_eq!(a.term(), "");
            n += 1
        })
        .unwrap();
        assert_eq!(n, 1);
        assert!(t.as_tokenizer().is_some());
        // reading before reset is the illegal-state reader
        let mut t = KeywordTokenizer::new();
        assert!(t.increment_token().is_err());
    }
}
