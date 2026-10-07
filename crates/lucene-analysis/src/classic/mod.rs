//! `org.apache.lucene.analysis.classic`: `ClassicTokenizer` (the pre-3.1
//! `StandardTokenizer` grammar: acronyms, companies, emails, hosts,
//! apostrophes; the JFlex scanner `ClassicTokenizerImpl` over its extracted
//! tables, see [`crate::util::jflex`]), `ClassicFilter` and
//! `ClassicAnalyzer`.
//!
//! The scanner uses JFlex's standard skeleton (`ZZ_BUFFERSIZE` 4096, the
//! buffer doubles for a longer match); `maxTokenLength` only skips tokens.

use std::sync::{Arc, LazyLock};

use crate::analyzer::{AnalyzerDefinition, TokenStreamComponents};
use crate::attributes::AttributeSource;
use crate::reader::CharReader;
use crate::standard::DEFAULT_MAX_TOKEN_LENGTH;
use crate::token_stream::{TokenFilter, TokenStream, Tokenizer, TokenizerInput};
use crate::util::jflex::{JFlexScanner, JFlexTables, Scan, YYEOF};
use crate::{
    AnalysisError, CharArraySet, LowerCaseFilter, StopFilter, StopwordAnalyzerBase,
    ENGLISH_STOP_WORDS,
};

/// `ClassicTokenizerImpl`'s tables (`tools/ExtractJFlexTables.java`).
static TABLES: LazyLock<JFlexTables> = LazyLock::new(|| {
    JFlexTables::load(include_bytes!("tables.bin.z")).expect("the generated classic scanner tables")
});

/// `ClassicTokenizerImpl.ZZ_BUFFERSIZE`.
const ZZ_BUFFERSIZE: usize = 4096;

/// `ClassicTokenizer.ALPHANUM`.
pub const ALPHANUM: usize = 0;
/// `ClassicTokenizer.APOSTROPHE`.
pub const APOSTROPHE: usize = 1;
/// `ClassicTokenizer.ACRONYM`.
pub const ACRONYM: usize = 2;
/// `ClassicTokenizer.COMPANY`.
pub const COMPANY: usize = 3;
/// `ClassicTokenizer.EMAIL`.
pub const EMAIL: usize = 4;
/// `ClassicTokenizer.HOST`.
pub const HOST: usize = 5;
/// `ClassicTokenizer.NUM`.
pub const NUM: usize = 6;
/// `ClassicTokenizer.CJ`.
pub const CJ: usize = 7;
/// `ClassicTokenizer.ACRONYM_DEP` (reported as `<HOST>`, its trailing dot
/// removed).
pub const ACRONYM_DEP: usize = 8;

/// `ClassicTokenizer.TOKEN_TYPES`.
pub const TOKEN_TYPES: [&str; 9] = [
    "<ALPHANUM>",
    "<APOSTROPHE>",
    "<ACRONYM>",
    "<COMPANY>",
    "<EMAIL>",
    "<HOST>",
    "<NUM>",
    "<CJ>",
    "<ACRONYM_DEP>",
];

/// `ClassicTokenizerImpl.getNextToken`: the generated action `switch` over
/// [`JFlexScanner::scan`].
fn get_next_token(s: &mut JFlexScanner, reader: &mut dyn CharReader) -> Result<i32, AnalysisError> {
    loop {
        let action = match s.scan(reader)? {
            Scan::Eof => return Ok(YYEOF),
            Scan::Action(a) => a,
        };
        let token = match action {
            1 => continue, // ignore
            2 => ALPHANUM,
            3 => CJ,
            4 => NUM,
            5 => HOST,
            6 => COMPANY,
            7 => APOSTROPHE,
            8 => ACRONYM_DEP,
            9 => ACRONYM,
            10 => EMAIL,
            _ => {
                return Err(AnalysisError::IllegalState(
                    "Error: could not match input".into(),
                ))
            }
        };
        return Ok(token as i32);
    }
}

/// `org.apache.lucene.analysis.classic.ClassicTokenizer`.
pub struct ClassicTokenizer {
    atts: AttributeSource,
    input: TokenizerInput,
    scanner: JFlexScanner,
    skipped_positions: i32,
    max_token_length: i32,
}

impl Default for ClassicTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ClassicTokenizer {
    /// `new ClassicTokenizer()`.
    pub fn new() -> Self {
        ClassicTokenizer {
            atts: AttributeSource::new(),
            input: TokenizerInput::new(),
            scanner: JFlexScanner::with_growth(&TABLES, ZZ_BUFFERSIZE, true),
            skipped_positions: 0,
            max_token_length: DEFAULT_MAX_TOKEN_LENGTH,
        }
    }

    /// `setMaxTokenLength(int)`: longer tokens are skipped (their positions
    /// counted).
    pub fn set_max_token_length(&mut self, length: i32) -> Result<(), AnalysisError> {
        if length < 1 {
            return Err(AnalysisError::IllegalArgument(
                "maxTokenLength must be greater than zero".into(),
            ));
        }
        self.max_token_length = length;
        Ok(())
    }

    /// `getMaxTokenLength()`.
    pub fn max_token_length(&self) -> i32 {
        self.max_token_length
    }
}

impl TokenStream for ClassicTokenizer {
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

    // Java: ClassicTokenizer.incrementToken
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        self.atts.clear_attributes();
        self.skipped_positions = 0;
        loop {
            let token_type = get_next_token(&mut self.scanner, self.input.reader()?)?;
            if token_type == YYEOF {
                return Ok(false);
            }
            let len = self.scanner.yylength();
            if len <= self.max_token_length as usize {
                self.atts
                    .set_position_increment(self.skipped_positions + 1)?;
                let start = self.scanner.yychar as i32;
                let (s, e) = (
                    self.input.correct_offset(start),
                    self.input.correct_offset(start + len as i32),
                );
                self.atts.set_offset(s, e)?;
                let text = self.scanner.text();
                if token_type as usize == ACRONYM_DEP {
                    self.atts.set_token_type(TOKEN_TYPES[HOST]);
                    // remove extra '.'
                    self.atts.set_term_utf16(&text[..len - 1]);
                } else {
                    self.atts.set_token_type(TOKEN_TYPES[token_type as usize]);
                    self.atts.set_term_utf16(text);
                }
                return Ok(true);
            }
            self.skipped_positions += 1;
        }
    }

    // Java: ClassicTokenizer.end
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        let f = self
            .input
            .correct_offset(self.scanner.yychar as i32 + self.scanner.yylength() as i32);
        self.atts.set_offset(f, f)?;
        let inc = self.atts.position_increment() + self.skipped_positions;
        self.atts.set_position_increment(inc)
    }

    // Java: ClassicTokenizer.reset
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset();
        self.scanner.yyreset();
        self.skipped_positions = 0;
        Ok(())
    }

    // Java: ClassicTokenizer.close
    fn close(&mut self) -> Result<(), AnalysisError> {
        self.scanner.yyreset();
        self.input.close()
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        Some(self)
    }
}

impl Tokenizer for ClassicTokenizer {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.input.set_reader(input)
    }
}

/// `org.apache.lucene.analysis.classic.ClassicFilter`: strips a trailing
/// `'s`/`'S` from `<APOSTROPHE>` tokens and the dots from `<ACRONYM>`s.
///
/// Differs: the type is compared by value (Java compares the `String`
/// reference, which is the same for every type `ClassicTokenizer` sets).
pub struct ClassicFilter<I> {
    input: I,
    buf: Vec<u16>,
}

impl<I: TokenStream> ClassicFilter<I> {
    /// `new ClassicFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        ClassicFilter {
            input,
            buf: Vec::new(),
        }
    }
}

impl<I: TokenStream> TokenFilter for ClassicFilter<I> {
    crate::filter_input!();

    // Java: ClassicFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        let ty = a.token_type();
        if ty == TOKEN_TYPES[APOSTROPHE] {
            let t = a.term();
            if t.ends_with("'s") || t.ends_with("'S") {
                let n = t.len() - 2;
                a.term_mut().truncate(n);
            }
        } else if ty == TOKEN_TYPES[ACRONYM] {
            crate::util::with_utf16_term(a, &mut self.buf, |b| {
                b.retain(|&c| c != u16::from(b'.'));
                true
            });
        }
        Ok(true)
    }
}

/// `org.apache.lucene.analysis.classic.ClassicAnalyzer`: [`ClassicTokenizer`],
/// [`ClassicFilter`], `LowerCaseFilter`, `StopFilter`.
///
/// Differs: `setMaxTokenLength` is construction-time (as for
/// [`crate::StandardAnalyzer`]).
#[derive(Debug, Clone)]
pub struct ClassicAnalyzer {
    base: StopwordAnalyzerBase,
    max_token_length: i32,
}

impl Default for ClassicAnalyzer {
    /// `new ClassicAnalyzer()`: `STOP_WORDS_SET` (English).
    fn default() -> Self {
        Self::new(CharArraySet::from_words(ENGLISH_STOP_WORDS, false))
    }
}

impl ClassicAnalyzer {
    /// `ClassicAnalyzer.DEFAULT_MAX_TOKEN_LENGTH`.
    pub const DEFAULT_MAX_TOKEN_LENGTH: i32 = 255;

    /// `new ClassicAnalyzer(CharArraySet)`.
    pub fn new(stop_words: CharArraySet) -> Self {
        ClassicAnalyzer {
            base: StopwordAnalyzerBase::new(Some(stop_words)),
            max_token_length: Self::DEFAULT_MAX_TOKEN_LENGTH,
        }
    }

    /// `setMaxTokenLength`, at construction.
    pub fn with_max_token_length(mut self, length: i32) -> Self {
        self.max_token_length = length;
        self
    }
}

impl AnalyzerDefinition for ClassicAnalyzer {
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let mut src = ClassicTokenizer::new();
        src.set_max_token_length(self.max_token_length)?;
        Ok(TokenStreamComponents::new(StopFilter::new(
            LowerCaseFilter::new(ClassicFilter::new(src)),
            Arc::clone(self.base.stopword_set()),
        )))
    }

    fn normalize(&self, _field: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(LowerCaseFilter::new(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::StrReader;
    use crate::Analyzer;

    fn run(t: &mut dyn TokenStream) -> Vec<(String, String, i32)> {
        let mut out = Vec::new();
        crate::token_stream::consume(t, |a| {
            out.push((
                a.term().to_string(),
                a.token_type().to_string(),
                a.position_increment(),
            ))
        })
        .unwrap();
        out
    }

    // Expected outputs are Lucene 10.5.0's.
    #[test]
    fn classic_types_and_filter() {
        let mut t = ClassicTokenizer::new();
        t.set_reader(Box::new(StrReader::new(
            "I.B.M. AT&T O'Reilly's bob@x.com www.x.org 1.2 wi.fi.",
        )))
        .unwrap();
        let mut f = ClassicFilter::new(t);
        let toks = run(&mut f);
        let terms: Vec<_> = toks.iter().map(|t| (t.0.as_str(), t.1.as_str())).collect();
        assert_eq!(
            terms,
            vec![
                ("IBM", "<ACRONYM>"),
                ("AT&T", "<COMPANY>"),
                ("O'Reilly", "<APOSTROPHE>"),
                ("bob@x.com", "<EMAIL>"),
                ("www.x.org", "<HOST>"),
                ("1.2", "<HOST>"),
                ("wi.fi", "<HOST>"),
            ]
        );
    }

    #[test]
    fn max_token_length_skips_and_counts() {
        let mut t = ClassicTokenizer::new();
        assert!(t.set_max_token_length(0).is_err());
        t.set_max_token_length(3).unwrap();
        assert_eq!(t.max_token_length(), 3);
        t.set_reader(Box::new(StrReader::new("ab abcdef cd")))
            .unwrap();
        let toks = run(&mut t);
        assert_eq!(toks[1], ("cd".to_string(), "<ALPHANUM>".to_string(), 2));
        let a = Analyzer::new(ClassicAnalyzer::default().with_max_token_length(10));
        assert_eq!(a.normalize("f", "AbC").unwrap(), b"abc");
    }
}
