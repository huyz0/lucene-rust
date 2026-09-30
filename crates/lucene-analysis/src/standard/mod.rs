//! `org.apache.lucene.analysis.standard`: `StandardTokenizer` (the UAX#29
//! word-break grammar, via the JFlex scanner in [`tokenizer_impl`]) and
//! `StandardAnalyzer`.

#[rustfmt::skip]
mod tables;
mod tokenizer_impl;

use std::sync::Arc;

use tokenizer_impl::{StandardTokenizerImpl, YYEOF};

use crate::analyzer::{AnalyzerDefinition, TokenStreamComponents};
use crate::attributes::AttributeSource;
use crate::reader::CharReader;
use crate::token_stream::{TokenStream, Tokenizer, TokenizerInput};
use crate::{AnalysisError, CharArraySet, LowerCaseFilter, StopFilter, StopwordAnalyzerBase};

/// `StandardTokenizer.ALPHANUM`.
pub const ALPHANUM: usize = 0;
/// `StandardTokenizer.NUM`.
pub const NUM: usize = 1;
/// `StandardTokenizer.SOUTHEAST_ASIAN`.
pub const SOUTHEAST_ASIAN: usize = 2;
/// `StandardTokenizer.IDEOGRAPHIC`.
pub const IDEOGRAPHIC: usize = 3;
/// `StandardTokenizer.HIRAGANA`.
pub const HIRAGANA: usize = 4;
/// `StandardTokenizer.KATAKANA`.
pub const KATAKANA: usize = 5;
/// `StandardTokenizer.HANGUL`.
pub const HANGUL: usize = 6;
/// `StandardTokenizer.EMOJI`.
pub const EMOJI: usize = 7;

/// `StandardTokenizer.TOKEN_TYPES`.
pub const TOKEN_TYPES: [&str; 8] = [
    "<ALPHANUM>",
    "<NUM>",
    "<SOUTHEAST_ASIAN>",
    "<IDEOGRAPHIC>",
    "<HIRAGANA>",
    "<KATAKANA>",
    "<HANGUL>",
    "<EMOJI>",
];

/// `StandardTokenizer.MAX_TOKEN_LENGTH_LIMIT`.
pub const MAX_TOKEN_LENGTH_LIMIT: i32 = 1024 * 1024;

/// `StandardAnalyzer.DEFAULT_MAX_TOKEN_LENGTH`.
pub const DEFAULT_MAX_TOKEN_LENGTH: i32 = 255;

/// `org.apache.lucene.analysis.standard.StandardTokenizer`: UAX#29 word
/// segmentation with Lucene's token types.
///
/// `maxTokenLength` is also the scanner's buffer size, exactly as in Java,
/// which is what gives a too-long run its Lucene behaviour: the scanner
/// cannot see past a full buffer, so the run is cut into
/// `maxTokenLength`-sized tokens rather than skipped. (The skip branch of
/// `incrementToken` -- `yylength() > maxTokenLength` -- is ported too; it is
/// only reachable when a supplementary character straddles the buffer end.)
///
/// Offsets are UTF-16 code units, Java's `char` offsets (see
/// [`crate::reader`]).
pub struct StandardTokenizer {
    atts: AttributeSource,
    input: TokenizerInput,
    scanner: StandardTokenizerImpl,
    skipped_positions: i32,
    max_token_length: i32,
}

impl Default for StandardTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl StandardTokenizer {
    /// `new StandardTokenizer()`.
    pub fn new() -> Self {
        StandardTokenizer {
            atts: AttributeSource::new(),
            input: TokenizerInput::new(),
            scanner: StandardTokenizerImpl::new(),
            skipped_positions: 0,
            max_token_length: DEFAULT_MAX_TOKEN_LENGTH,
        }
    }

    /// `setMaxTokenLength(int)`, with Java's bounds.
    pub fn set_max_token_length(&mut self, length: i32) -> Result<(), AnalysisError> {
        if length < 1 {
            return Err(AnalysisError::IllegalArgument(
                "maxTokenLength must be greater than zero".to_string(),
            ));
        } else if length > MAX_TOKEN_LENGTH_LIMIT {
            return Err(AnalysisError::IllegalArgument(format!(
                "maxTokenLength may not exceed {MAX_TOKEN_LENGTH_LIMIT}"
            )));
        }
        if length != self.max_token_length {
            self.max_token_length = length;
            self.scanner.set_buffer_size(length as usize);
        }
        Ok(())
    }

    /// `getMaxTokenLength()`.
    pub fn max_token_length(&self) -> i32 {
        self.max_token_length
    }
}

impl TokenStream for StandardTokenizer {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        self.atts.clear_attributes();
        self.skipped_positions = 0;

        loop {
            let token_type = self.scanner.get_next_token(self.input.reader()?)?;

            if token_type == YYEOF {
                return Ok(false);
            }

            if self.scanner.yylength() <= self.max_token_length as usize {
                self.atts
                    .set_position_increment(self.skipped_positions.saturating_add(1))?;
                self.atts.set_term_utf16(self.scanner.text());
                let start = self.scanner.yychar();
                let end = start.saturating_add(self.scanner.yylength() as i32);
                self.atts.set_offset(
                    self.input.correct_offset(start),
                    self.input.correct_offset(end),
                )?;
                self.atts.set_token_type(TOKEN_TYPES[token_type as usize]);
                return Ok(true);
            }
            // When we skip a too-long term, we still increment the
            // position increment
            self.skipped_positions = self.skipped_positions.saturating_add(1);
        }
    }

    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        // set final offset
        let final_offset = self.input.correct_offset(
            self.scanner
                .yychar()
                .saturating_add(self.scanner.yylength() as i32),
        );
        self.atts.set_offset(final_offset, final_offset)?;
        // adjust any skipped tokens
        let inc = self
            .atts
            .position_increment()
            .saturating_add(self.skipped_positions);
        self.atts.set_position_increment(inc)
    }

    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset();
        self.scanner.yyreset();
        self.skipped_positions = 0;
        Ok(())
    }

    fn close(&mut self) -> Result<(), AnalysisError> {
        let closed = self.input.close();
        self.scanner.yyreset();
        closed
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        Some(self)
    }
}

impl Tokenizer for StandardTokenizer {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.input.set_reader(input)
    }
}

/// `org.apache.lucene.analysis.standard.StandardAnalyzer`:
/// [`StandardTokenizer`] + [`LowerCaseFilter`] + [`StopFilter`], and
/// [`LowerCaseFilter`] alone for `normalize`.
///
/// Wrap it in an [`crate::Analyzer`] to use it (`Analyzer::new`). Java's
/// mutable `setMaxTokenLength` -- which its components re-read on every
/// `setReader` -- is a construction-time setting here, since an
/// `AnalyzerDefinition` is shared immutably.
#[derive(Debug, Clone)]
pub struct StandardAnalyzer {
    base: StopwordAnalyzerBase,
    max_token_length: i32,
}

impl Default for StandardAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StandardAnalyzer {
    /// `new StandardAnalyzer()`: no stopwords (Java's default since 7.0).
    pub fn new() -> Self {
        Self::with_stopwords(CharArraySet::empty())
    }

    /// `new StandardAnalyzer(CharArraySet)`.
    pub fn with_stopwords(stopwords: CharArraySet) -> Self {
        StandardAnalyzer {
            base: StopwordAnalyzerBase::new(Some(stopwords)),
            max_token_length: DEFAULT_MAX_TOKEN_LENGTH,
        }
    }

    /// `new StandardAnalyzer(Reader)`: stopwords in
    /// [`crate::wordlist_loader::get_word_set`]'s format.
    pub fn from_stopword_reader(reader: impl std::io::Read) -> Result<Self, AnalysisError> {
        Ok(Self::with_stopwords(
            StopwordAnalyzerBase::load_stopword_set(reader)?,
        ))
    }

    /// `setMaxTokenLength(int)`: validated when components are created, as
    /// Java's `StandardTokenizer.setMaxTokenLength` validates it there.
    pub fn with_max_token_length(mut self, length: i32) -> Self {
        self.max_token_length = length;
        self
    }

    /// `getMaxTokenLength()`.
    pub fn max_token_length(&self) -> i32 {
        self.max_token_length
    }

    /// `getStopwordSet()`.
    pub fn stopword_set(&self) -> &Arc<CharArraySet> {
        self.base.stopword_set()
    }
}

impl AnalyzerDefinition for StandardAnalyzer {
    fn create_components(&self, _field_name: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let mut src = StandardTokenizer::new();
        src.set_max_token_length(self.max_token_length)?;
        let tok = LowerCaseFilter::new(src);
        let tok = StopFilter::new(tok, Arc::clone(self.base.stopword_set()));
        Ok(TokenStreamComponents::new(tok))
    }

    fn normalize(&self, _field_name: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(LowerCaseFilter::new(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::StrReader;
    use crate::token_stream::consume;

    /// (term, start, end, posInc, type).
    type Tok = (String, i32, i32, i32, String);

    fn run(tok: &mut StandardTokenizer, text: &str) -> (Vec<Tok>, AttributeSource) {
        tok.set_reader(Box::new(StrReader::new(text))).unwrap();
        let mut out = Vec::new();
        let end = consume(tok, |a| {
            out.push((
                a.term().to_string(),
                a.start_offset(),
                a.end_offset(),
                a.position_increment(),
                a.token_type().to_string(),
            ))
        })
        .unwrap();
        (out, end)
    }

    #[test]
    fn token_types_and_offsets() {
        let mut tok = StandardTokenizer::new();
        let (toks, end) = run(&mut tok, "Hi 42 日本 ひら カタカナ 한국 ไทย 😀!");
        let types: Vec<&str> = toks.iter().map(|t| t.4.as_str()).collect();
        assert_eq!(
            types,
            vec![
                "<ALPHANUM>",
                "<NUM>",
                "<IDEOGRAPHIC>",
                "<IDEOGRAPHIC>",
                "<HIRAGANA>",
                "<HIRAGANA>",
                "<KATAKANA>",
                "<HANGUL>",
                "<SOUTHEAST_ASIAN>",
                "<EMOJI>",
            ]
        );
        assert_eq!(toks[9].0, "😀");
        // the emoji is two UTF-16 units
        assert_eq!((toks[9].1, toks[9].2), (24, 26));
        assert_eq!((end.start_offset(), end.end_offset()), (27, 27));
        assert_eq!(end.position_increment(), 0);
    }

    #[test]
    fn long_runs_split_at_the_buffer_size() {
        let mut tok = StandardTokenizer::new();
        tok.set_max_token_length(5).unwrap();
        let (toks, _) = run(&mut tok, "abcdefghijkl xy");
        let terms: Vec<&str> = toks.iter().map(|t| t.0.as_str()).collect();
        assert_eq!(terms, vec!["abcde", "fghij", "kl", "xy"]);
        assert_eq!(tok.max_token_length(), 5);
        // reusable after close
        let (toks, _) = run(&mut tok, "q");
        assert_eq!(toks.len(), 1);
    }

    #[test]
    fn max_token_length_bounds() {
        let mut tok = StandardTokenizer::default();
        assert!(tok.set_max_token_length(0).is_err());
        assert!(tok
            .set_max_token_length(MAX_TOKEN_LENGTH_LIMIT + 1)
            .is_err());
        assert!(tok.set_max_token_length(MAX_TOKEN_LENGTH_LIMIT).is_ok());
    }

    #[test]
    fn contract_violations_are_errors() {
        let mut tok = StandardTokenizer::new();
        // increment before reset: the illegal-state reader.
        assert!(tok.increment_token().is_err());
        tok.set_reader(Box::new(StrReader::new("a"))).unwrap();
        tok.reset().unwrap();
        // setReader again without close.
        let err = tok.set_reader(Box::new(StrReader::new("b"))).unwrap_err();
        assert!(err.to_string().contains("close() call missing"), "{err}");
        tok.close().unwrap();
        // reset twice: the second leaves the illegal-state reader.
        tok.set_reader(Box::new(StrReader::new("c"))).unwrap();
        tok.reset().unwrap();
        tok.reset().unwrap();
        assert!(tok.increment_token().is_err());
    }
}
