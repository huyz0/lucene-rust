//! `org.apache.lucene.analysis.email`: `UAX29URLEmailTokenizer` (the JFlex
//! scanner `UAX29URLEmailTokenizerImpl` over its extracted tables, see
//! [`crate::util::jflex`]) and `UAX29URLEmailAnalyzer`.

use std::sync::{Arc, LazyLock};

use crate::analyzer::{AnalyzerDefinition, TokenStreamComponents};
use crate::attributes::AttributeSource;
use crate::reader::CharReader;
use crate::standard::{self, DEFAULT_MAX_TOKEN_LENGTH};
use crate::token_stream::{TokenStream, Tokenizer, TokenizerInput};
use crate::util::jflex::{JFlexScanner, JFlexTables, Scan, YYEOF};
use crate::{
    AnalysisError, CharArraySet, LowerCaseFilter, StopFilter, StopwordAnalyzerBase,
    ENGLISH_STOP_WORDS,
};

/// `UAX29URLEmailTokenizerImpl`'s tables (`tools/ExtractJFlexTables.java`).
static TABLES: LazyLock<JFlexTables> = LazyLock::new(|| {
    JFlexTables::load(include_bytes!("tables.bin.z")).expect("the generated email scanner tables")
});

/// `UAX29URLEmailTokenizer.ALPHANUM` .. `EMOJI`.
pub const ALPHANUM: usize = 0;
/// `UAX29URLEmailTokenizer.NUM`.
pub const NUM: usize = 1;
/// `UAX29URLEmailTokenizer.SOUTHEAST_ASIAN`.
pub const SOUTHEAST_ASIAN: usize = 2;
/// `UAX29URLEmailTokenizer.IDEOGRAPHIC`.
pub const IDEOGRAPHIC: usize = 3;
/// `UAX29URLEmailTokenizer.HIRAGANA`.
pub const HIRAGANA: usize = 4;
/// `UAX29URLEmailTokenizer.KATAKANA`.
pub const KATAKANA: usize = 5;
/// `UAX29URLEmailTokenizer.HANGUL`.
pub const HANGUL: usize = 6;
/// `UAX29URLEmailTokenizer.URL`.
pub const URL: usize = 7;
/// `UAX29URLEmailTokenizer.EMAIL`.
pub const EMAIL: usize = 8;
/// `UAX29URLEmailTokenizer.EMOJI`.
pub const EMOJI: usize = 9;

/// `UAX29URLEmailTokenizer.TOKEN_TYPES`.
pub const TOKEN_TYPES: [&str; 10] = [
    standard::TOKEN_TYPES[standard::ALPHANUM],
    standard::TOKEN_TYPES[standard::NUM],
    standard::TOKEN_TYPES[standard::SOUTHEAST_ASIAN],
    standard::TOKEN_TYPES[standard::IDEOGRAPHIC],
    standard::TOKEN_TYPES[standard::HIRAGANA],
    standard::TOKEN_TYPES[standard::KATAKANA],
    standard::TOKEN_TYPES[standard::HANGUL],
    "<URL>",
    "<EMAIL>",
    standard::TOKEN_TYPES[standard::EMOJI],
];

/// `UAX29URLEmailTokenizer.MAX_TOKEN_LENGTH_LIMIT`.
pub const MAX_TOKEN_LENGTH_LIMIT: i32 = 1024 * 1024;

/// `UAX29URLEmailTokenizerImpl.YYINITIAL`.
const YYINITIAL: usize = 0;
/// `UAX29URLEmailTokenizerImpl.AVOID_BAD_URL`.
const AVOID_BAD_URL: usize = 2;

/// `UAX29URLEmailTokenizerImpl.getNextToken`: the generated action `switch`
/// over [`JFlexScanner::scan`].
fn get_next_token(s: &mut JFlexScanner, reader: &mut dyn CharReader) -> Result<i32, AnalysisError> {
    loop {
        let action = match s.scan(reader)? {
            // Both lexical states return YYEOF at the end.
            Scan::Eof => return Ok(YYEOF),
            Scan::Action(a) => a,
        };
        let token = match action {
            // Not numeric, word, ideographic, hiragana, emoji or SE Asian -- ignore it.
            1 => continue,
            2 => NUM,
            3 => ALPHANUM,
            4 => EMOJI,
            5 => SOUTHEAST_ASIAN,
            6 => HANGUL,
            7 => IDEOGRAPHIC,
            8 => KATAKANA,
            9 => HIRAGANA,
            10 => EMAIL,
            11 => return Ok(URL as i32),
            12 => {
                // lookahead expression with fixed lookahead length
                s.marked_pos = s.offset_by_code_points(s.marked_pos, -1);
                URL
            }
            13 => URL,
            14 => {
                // lookahead expression with fixed lookahead length
                s.marked_pos = s.offset_by_code_points(s.marked_pos, -1);
                s.lexical_state = AVOID_BAD_URL;
                let n = s.yylength();
                s.yypushback(n)?;
                continue;
            }
            15 => {
                // lookahead expression with fixed base length
                s.marked_pos = s.offset_by_code_points(s.start_read, 6);
                ALPHANUM
            }
            _ => {
                return Err(AnalysisError::IllegalState(
                    "Error: could not match input".into(),
                ))
            }
        };
        s.lexical_state = YYINITIAL;
        return Ok(token as i32);
    }
}

/// `org.apache.lucene.analysis.email.UAX29URLEmailTokenizer`: UAX#29 word
/// segmentation that keeps URLs and email addresses whole.
pub struct UAX29URLEmailTokenizer {
    atts: AttributeSource,
    input: TokenizerInput,
    scanner: JFlexScanner,
    skipped_positions: i32,
    max_token_length: i32,
}

impl Default for UAX29URLEmailTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl UAX29URLEmailTokenizer {
    /// `new UAX29URLEmailTokenizer()`.
    pub fn new() -> Self {
        UAX29URLEmailTokenizer {
            atts: AttributeSource::new(),
            input: TokenizerInput::new(),
            scanner: JFlexScanner::new(&TABLES, DEFAULT_MAX_TOKEN_LENGTH as usize),
            skipped_positions: 0,
            max_token_length: DEFAULT_MAX_TOKEN_LENGTH,
        }
    }

    /// `setMaxTokenLength(int)`.
    pub fn set_max_token_length(&mut self, length: i32) -> Result<(), AnalysisError> {
        if length < 1 {
            return Err(AnalysisError::IllegalArgument(
                "maxTokenLength must be greater than zero".into(),
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

impl TokenStream for UAX29URLEmailTokenizer {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    // Java: UAX29URLEmailTokenizer.incrementToken
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        self.atts.clear_attributes();
        self.skipped_positions = 0;
        loop {
            let token_type = get_next_token(&mut self.scanner, self.input.reader()?)?;
            if token_type == YYEOF {
                return Ok(false);
            }
            if self.scanner.yylength() <= self.max_token_length as usize {
                self.atts
                    .set_position_increment(self.skipped_positions + 1)?;
                self.atts.set_term_utf16(self.scanner.text());
                let start = self.scanner.yychar as i32;
                let len = self.scanner.yylength() as i32;
                let (s, e) = (
                    self.input.correct_offset(start),
                    self.input.correct_offset(start + len),
                );
                self.atts.set_offset(s, e)?;
                self.atts.set_token_type(TOKEN_TYPES[token_type as usize]);
                return Ok(true);
            }
            self.skipped_positions += 1;
        }
    }

    // Java: UAX29URLEmailTokenizer.end
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        let f = self
            .input
            .correct_offset(self.scanner.yychar as i32 + self.scanner.yylength() as i32);
        self.atts.set_offset(f, f)?;
        let inc = self.atts.position_increment() + self.skipped_positions;
        self.atts.set_position_increment(inc)
    }

    // Java: UAX29URLEmailTokenizer.reset
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset();
        self.scanner.yyreset();
        self.skipped_positions = 0;
        Ok(())
    }

    // Java: UAX29URLEmailTokenizer.close
    fn close(&mut self) -> Result<(), AnalysisError> {
        self.scanner.yyreset();
        self.input.close()
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        Some(self)
    }
}

impl Tokenizer for UAX29URLEmailTokenizer {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.input.set_reader(input)
    }
}

/// `org.apache.lucene.analysis.email.UAX29URLEmailAnalyzer`:
/// [`UAX29URLEmailTokenizer`], `LowerCaseFilter`, `StopFilter`.
///
/// Differs: `setMaxTokenLength` is construction-time (as for
/// [`crate::StandardAnalyzer`]).
#[derive(Debug, Clone)]
pub struct UAX29URLEmailAnalyzer {
    base: StopwordAnalyzerBase,
    max_token_length: i32,
}

impl Default for UAX29URLEmailAnalyzer {
    /// `new UAX29URLEmailAnalyzer()`: `STOP_WORDS_SET` (English).
    fn default() -> Self {
        Self::new(CharArraySet::from_words(ENGLISH_STOP_WORDS, false))
    }
}

impl UAX29URLEmailAnalyzer {
    /// `new UAX29URLEmailAnalyzer(CharArraySet)`.
    pub fn new(stop_words: CharArraySet) -> Self {
        UAX29URLEmailAnalyzer {
            base: StopwordAnalyzerBase::new(Some(stop_words)),
            max_token_length: DEFAULT_MAX_TOKEN_LENGTH,
        }
    }

    /// `setMaxTokenLength`, at construction.
    pub fn with_max_token_length(mut self, length: i32) -> Self {
        self.max_token_length = length;
        self
    }
}

impl AnalyzerDefinition for UAX29URLEmailAnalyzer {
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let mut src = UAX29URLEmailTokenizer::new();
        src.set_max_token_length(self.max_token_length)?;
        Ok(TokenStreamComponents::new(StopFilter::new(
            LowerCaseFilter::new(src),
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
    use crate::util::canned::render;
    use crate::Analyzer;

    fn run(t: &mut UAX29URLEmailTokenizer, text: &str) -> Vec<(String, String)> {
        t.set_reader(Box::new(StrReader::new(text))).unwrap();
        let mut out = Vec::new();
        crate::token_stream::consume(t, |a| {
            out.push((a.term().to_string(), a.token_type().to_string()))
        })
        .unwrap();
        out
    }

    #[test]
    fn urls_emails_and_words() {
        let mut t = UAX29URLEmailTokenizer::new();
        let toks = run(
            &mut t,
            "mail a.b@example.com or see https://x.org/p?q=1 now 42",
        );
        assert_eq!(
            toks,
            vec![
                ("mail".into(), "<ALPHANUM>".into()),
                ("a.b@example.com".into(), "<EMAIL>".into()),
                ("or".into(), "<ALPHANUM>".into()),
                ("see".into(), "<ALPHANUM>".into()),
                ("https://x.org/p?q=1".into(), "<URL>".into()),
                ("now".into(), "<ALPHANUM>".into()),
                ("42".into(), "<NUM>".into()),
            ]
        );
    }

    /// `(text, what Lucene's tokenizer gave: term type start end posInc|...|end posInc)`.
    const CASES: &[(&str, &str)] = &[
        ("http://x.org. www.x.com, mailto:a@b.c https:x ftp: a.b.c.d", "http://x.org <URL> 0 12 1|www.x.com <URL> 14 23 1|mailto:a <ALPHANUM> 25 33 1|b.c <ALPHANUM> 34 37 1|https:x <ALPHANUM> 38 45 1|ftp <ALPHANUM> 46 49 1|a.b.c.d <ALPHANUM> 51 58 1|58 0"),
        ("xhttp://y http://[::1]:80/p foo@bar <http://x> .com e.g. v1.2.3", "xhttp <ALPHANUM> 0 5 1|y <ALPHANUM> 8 9 1|http://[::1]:80/p <URL> 10 27 1|foo <ALPHANUM> 28 31 1|bar <ALPHANUM> 32 35 1|http://x <URL> 37 45 1|com <ALPHANUM> 48 51 1|e.g <ALPHANUM> 52 55 1|v1.2.3 <ALPHANUM> 57 63 1|63 0"),
        ("file:///etc HTTPS://WWW.EXAMPLE.COM/ user@[192.168.0.1] www.example.com.", "file:///etc <URL> 0 11 1|HTTPS://WWW.EXAMPLE.COM/ <URL> 12 36 1|user@[192.168.0.1] <EMAIL> 37 55 1|www.example.com <URL> 56 71 1|72 0"),
        ("example.co.uk/path) http:// x.y a_b@c-d.ef http://example.com/a?b=c&d=e#f,", "example.co.uk/path <URL> 0 18 1|http <ALPHANUM> 20 24 1|x.y <ALPHANUM> 28 31 1|a_b <ALPHANUM> 32 35 1|c <ALPHANUM> 36 37 1|d.ef <ALPHANUM> 38 42 1|http://example.com/a?b=c&d=e#f, <URL> 43 74 1|74 0"),
        ("wwwx.com.au/ http://a..b x@y.z.w 中国人 ภาษาไทย 한국어 カタカナ ひらがな 😀 42", "wwwx.com.au/ <URL> 0 12 1|http://a <URL> 13 21 1|b <ALPHANUM> 23 24 1|x <ALPHANUM> 25 26 1|y.z.w <ALPHANUM> 27 32 1|中 <IDEOGRAPHIC> 33 34 1|国 <IDEOGRAPHIC> 34 35 1|人 <IDEOGRAPHIC> 35 36 1|ภาษาไทย <SOUTHEAST_ASIAN> 37 44 1|한국어 <HANGUL> 45 48 1|カタカナ <KATAKANA> 49 53 1|ひ <HIRAGANA> 54 55 1|ら <HIRAGANA> 55 56 1|が <HIRAGANA> 56 57 1|な <HIRAGANA> 57 58 1|😀 <EMOJI> 59 61 1|42 <NUM> 62 64 1|64 0"),
    ];

    fn java_format(t: &mut UAX29URLEmailTokenizer, text: &str) -> String {
        t.set_reader(Box::new(StrReader::new(text))).unwrap();
        let mut out = String::new();
        let end = crate::token_stream::consume(t, |a| {
            out.push_str(&format!(
                "{} {} {} {} {}|",
                a.term(),
                a.token_type(),
                a.start_offset(),
                a.end_offset(),
                a.position_increment()
            ))
        })
        .unwrap();
        out + &format!("{} {}", end.end_offset(), end.position_increment())
    }

    #[test]
    fn matches_lucene_on_urls_emails_and_scripts() {
        let mut t = UAX29URLEmailTokenizer::new();
        for (text, expected) in CASES {
            assert_eq!(java_format(&mut t, text), *expected, "{text}");
        }
        let long = format!("http://{}.com x", "a".repeat(300));
        let got = java_format(&mut t, &long);
        assert!(
            got.ends_with(".com <URL> 255 311 1|x <ALPHANUM> 312 313 1|313 0"),
            "{got}"
        );
        let mut t = UAX29URLEmailTokenizer::new();
        t.set_max_token_length(5).unwrap();
        assert_eq!(
            java_format(&mut t, "abcdefghij http://abc.de x"),
            "abcde <ALPHANUM> 0 5 1|fghij <ALPHANUM> 5 10 1|http <ALPHANUM> 11 15 1|abc.d <ALPHANUM> 18 23 1|e <ALPHANUM> 23 24 1|x <ALPHANUM> 25 26 1|26 0"
        );
    }

    #[test]
    fn max_token_length_and_analyzer() {
        let mut t = UAX29URLEmailTokenizer::new();
        assert!(t.set_max_token_length(0).is_err());
        assert!(t.set_max_token_length(MAX_TOKEN_LENGTH_LIMIT + 1).is_err());
        t.set_max_token_length(3).unwrap();
        assert_eq!(t.max_token_length(), 3);
        t.set_reader(Box::new(StrReader::new("abcdefg hi")))
            .unwrap();
        assert_eq!(
            render(&mut t),
            "abc:0:3:1:1 def:3:6:1:1 g:6:7:1:1 hi:8:10:1:1|10|0"
        );
        let a = Analyzer::new(UAX29URLEmailAnalyzer::default().with_max_token_length(255));
        let terms: Vec<String> = a
            .analyze("The URL HTTP://X.ORG")
            .into_iter()
            .map(|t| t.term)
            .collect();
        assert_eq!(terms, vec!["url", "http://x.org"]);
        assert_eq!(a.normalize("f", "AB").unwrap(), b"ab");
        let bad = Analyzer::new(
            UAX29URLEmailAnalyzer::new(CharArraySet::empty()).with_max_token_length(0),
        );
        assert!(bad.try_analyze_stream("x").is_err());
    }
}
