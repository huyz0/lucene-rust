//! `org.apache.lucene.analysis.ngram`: `NGramTokenizer`, `EdgeNGramTokenizer`,
//! and the streaming `NGramTokenFilter` / `EdgeNGramTokenFilter` (the
//! crate-root `NGramTokenFilter`/`EdgeNGramTokenFilter` are the older
//! `Vec<Token>` API).

use crate::attributes::{AttributeSource, State};
use crate::java_character::{char_count, code_point_at, push_utf16};
use crate::reader::CharReader;
use crate::token_stream::{TokenFilter, TokenStream, Tokenizer, TokenizerInput};
use crate::util::char_tokenizer::{CharacterBuffer, TokenChar};
use crate::util::configured_vec;
use crate::AnalysisError;

/// `NGramTokenizer.DEFAULT_MIN_NGRAM_SIZE`.
pub const DEFAULT_MIN_NGRAM_SIZE: i32 = 1;
/// `NGramTokenizer.DEFAULT_MAX_NGRAM_SIZE`.
pub const DEFAULT_MAX_NGRAM_SIZE: i32 = 2;

fn check_grams(min_gram: i32, max_gram: i32) -> Result<(), AnalysisError> {
    if min_gram < 1 {
        return Err(AnalysisError::IllegalArgument(
            "minGram must be greater than zero".into(),
        ));
    }
    if min_gram > max_gram {
        return Err(AnalysisError::IllegalArgument(
            "minGram must not be greater than maxGram".into(),
        ));
    }
    Ok(())
}

/// `NGramTokenizer.isTokenChar`'s default: every code point.
#[derive(Debug, Clone, Copy, Default)]
pub struct AnyChar;

impl TokenChar for AnyChar {
    fn is_token_char(&self, _c: u32) -> bool {
        true
    }
}

/// `org.apache.lucene.analysis.ngram.NGramTokenizer` (and, with `edgesOnly`,
/// `EdgeNGramTokenizer`), generic over Java's overridable `isTokenChar`.
pub struct NGramTokenizer<P = AnyChar> {
    atts: AttributeSource,
    input: TokenizerInput,
    predicate: P,
    char_buffer: CharacterBuffer,
    buffer: Vec<u32>,
    buffer_start: i32,
    buffer_end: i32,
    offset: i32,
    gram_size: i32,
    min_gram: i32,
    max_gram: i32,
    exhausted: bool,
    last_checked_char: i32,
    last_non_token_char: i32,
    edges_only: bool,
    term: Vec<u16>,
}

/// `org.apache.lucene.analysis.ngram.EdgeNGramTokenizer`.
pub type EdgeNGramTokenizer = NGramTokenizer<AnyChar>;

impl NGramTokenizer<AnyChar> {
    /// `new NGramTokenizer(int minGram, int maxGram)`.
    pub fn new(min_gram: i32, max_gram: i32) -> Result<Self, AnalysisError> {
        Self::with_predicate(min_gram, max_gram, false, AnyChar)
    }

    /// `new EdgeNGramTokenizer(int minGram, int maxGram)`.
    pub fn edge(min_gram: i32, max_gram: i32) -> Result<Self, AnalysisError> {
        Self::with_predicate(min_gram, max_gram, true, AnyChar)
    }
}

impl<P: TokenChar> NGramTokenizer<P> {
    /// `NGramTokenizer(int, int, boolean edgesOnly)` with an `isTokenChar`.
    pub fn with_predicate(
        min_gram: i32,
        max_gram: i32,
        edges_only: bool,
        predicate: P,
    ) -> Result<Self, AnalysisError> {
        check_grams(min_gram, max_gram)?;
        // 2 * maxGram in case all code points take 2 chars, + 1024 for reading.
        // Java sizes an int[] and char[] of `2 * maxGram + 1024`; past
        // Integer.MAX_VALUE that is a NegativeArraySizeException.
        let size = max_gram
            .checked_mul(2)
            .and_then(|m| m.checked_add(1024))
            .ok_or_else(|| {
                AnalysisError::IllegalArgument("NegativeArraySizeException: maxGram".into())
            })? as usize;
        Ok(NGramTokenizer {
            atts: AttributeSource::new(),
            input: TokenizerInput::new(),
            predicate,
            // ALLOC: `configured_vec` caps both at MAX_CONFIGURED_ALLOCATION.
            char_buffer: CharacterBuffer::with_buffer(configured_vec(size, 0, "maxGram")?),
            buffer: configured_vec(size, 0, "maxGram")?,
            buffer_start: 0,
            buffer_end: 0,
            offset: 0,
            gram_size: 0,
            min_gram,
            max_gram,
            exhausted: false,
            last_checked_char: 0,
            last_non_token_char: 0,
            edges_only,
            term: Vec::new(),
        })
    }

    // Java: NGramTokenizer.updateLastNonTokenChar
    fn update_last_non_token_char(&mut self) {
        let term_end = self.buffer_start + self.gram_size - 1;
        if term_end > self.last_checked_char {
            let mut i = term_end;
            while i > self.last_checked_char {
                if !self.predicate.is_token_char(self.buffer[i as usize]) {
                    self.last_non_token_char = i;
                    break;
                }
                i -= 1;
            }
            self.last_checked_char = term_end;
        }
    }

    // Java: NGramTokenizer.consume
    fn consume(&mut self) {
        self.offset += char_count(self.buffer[self.buffer_start as usize]) as i32;
        self.buffer_start += 1;
    }
}

impl<P: TokenChar> TokenStream for NGramTokenizer<P> {
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

    // Java: NGramTokenizer.incrementToken
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        self.atts.clear_attributes();
        loop {
            if self.buffer_start >= self.buffer_end - self.max_gram - 1 && !self.exhausted {
                let (s, e) = (self.buffer_start as usize, self.buffer_end as usize);
                self.buffer.copy_within(s..e, 0);
                self.buffer_end -= self.buffer_start;
                self.last_checked_char -= self.buffer_start;
                self.last_non_token_char -= self.buffer_start;
                self.buffer_start = 0;
                let room = self.buffer.len() - self.buffer_end as usize;
                let reader = self.input.reader()?;
                self.exhausted = !self.char_buffer.fill_n(reader, room)?;
                // CharacterUtils.toCodePoints
                let cb = &self.char_buffer;
                let mut i = 0;
                let mut n = self.buffer_end as usize;
                while i < cb.length {
                    let cp = code_point_at(&cb.buffer, i, cb.length);
                    self.buffer[n] = cp;
                    n += 1;
                    i += char_count(cp);
                }
                self.buffer_end = n as i32;
            }
            if self.gram_size > self.max_gram
                || self.buffer_start + self.gram_size > self.buffer_end
            {
                if self.buffer_start + 1 + self.min_gram > self.buffer_end {
                    debug_assert!(self.exhausted);
                    return Ok(false);
                }
                self.consume();
                self.gram_size = self.min_gram;
            }
            self.update_last_non_token_char();
            let contains_non_token = self.last_non_token_char >= self.buffer_start
                && self.last_non_token_char < self.buffer_start + self.gram_size;
            let edge_after_token_char =
                self.edges_only && self.last_non_token_char != self.buffer_start - 1;
            if contains_non_token || edge_after_token_char {
                self.consume();
                self.gram_size = self.min_gram;
                continue;
            }
            self.term.clear();
            let s = self.buffer_start as usize;
            for &cp in &self.buffer[s..s + self.gram_size as usize] {
                push_utf16(&mut self.term, cp);
            }
            let length = self.term.len() as i32;
            self.atts.set_term_utf16(&self.term);
            self.atts.set_position_increment(1)?;
            self.atts.set_position_length(1)?;
            let start = self.input.correct_offset(self.offset);
            let end = self.input.correct_offset(self.offset + length);
            self.atts.set_offset(start, end)?;
            self.gram_size += 1;
            return Ok(true);
        }
    }

    // Java: NGramTokenizer.end
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        let mut end_offset = self.offset;
        for &cp in &self.buffer[self.buffer_start as usize..self.buffer_end as usize] {
            end_offset += char_count(cp) as i32;
        }
        let end_offset = self.input.correct_offset(end_offset);
        self.atts.set_offset(end_offset, end_offset)
    }

    // Java: NGramTokenizer.reset
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset();
        self.buffer_start = self.buffer.len() as i32;
        self.buffer_end = self.buffer_start;
        self.last_non_token_char = self.buffer_start - 1;
        self.last_checked_char = self.buffer_start - 1;
        self.offset = 0;
        self.gram_size = self.min_gram;
        self.exhausted = false;
        self.char_buffer.reset();
        Ok(())
    }

    fn close(&mut self) -> Result<(), AnalysisError> {
        self.input.close()
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        Some(self)
    }
}

impl<P: TokenChar> Tokenizer for NGramTokenizer<P> {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.input.set_reader(input)
    }
}

/// The state `NGramTokenFilter` and `EdgeNGramTokenFilter` share.
struct GramState {
    min_gram: i32,
    max_gram: i32,
    preserve_original: bool,
    /// `curTermBuffer`, as code points (`None` between input tokens).
    cur_term: Option<Vec<char>>,
    cur_gram_size: i32,
    cur_pos: i32,
    cur_pos_incr: i32,
    state: Option<State>,
}

impl GramState {
    fn new(min_gram: i32, max_gram: i32, preserve_original: bool) -> Result<Self, AnalysisError> {
        check_grams(min_gram, max_gram)?;
        Ok(GramState {
            min_gram,
            max_gram,
            preserve_original,
            cur_term: None,
            cur_gram_size: 0,
            cur_pos: 0,
            cur_pos_incr: 0,
            state: None,
        })
    }

    /// The shared head of both `incrementToken`s: reads the next input token
    /// into `cur_term`. `Ok(Some(true))` when a too-short token is passed
    /// through as is, `Ok(None)` at the end of the input.
    fn next_input<I: TokenStream>(&mut self, input: &mut I) -> Result<Option<bool>, AnalysisError> {
        if !input.increment_token()? {
            return Ok(None);
        }
        let a = input.attributes();
        self.state = Some(a.capture_state());
        let chars: Vec<char> = a.term().chars().collect();
        self.cur_pos_incr += a.position_increment();
        self.cur_pos = 0;
        if self.preserve_original && (chars.len() as i32) < self.min_gram {
            let inc = self.cur_pos_incr;
            input.attributes_mut().set_position_increment(inc)?;
            self.cur_pos_incr = 0;
            return Ok(Some(true));
        }
        self.cur_term = Some(chars);
        self.cur_gram_size = self.min_gram;
        Ok(Some(false))
    }

    /// Restores the input token and writes `chars` with `inc`.
    fn emit(
        &mut self,
        a: &mut AttributeSource,
        chars: &[char],
        inc: i32,
    ) -> Result<(), AnalysisError> {
        a.restore_state(self.state.as_ref().expect("a token was read"));
        let s: String = chars.iter().collect();
        a.set_term(&s);
        a.set_position_increment(inc)
    }
}

/// `org.apache.lucene.analysis.ngram.NGramTokenFilter`, streaming.
pub struct NGramTokenFilter<I> {
    input: I,
    g: GramState,
}

impl<I: TokenStream> NGramTokenFilter<I> {
    /// `new NGramTokenFilter(TokenStream, int minGram, int maxGram, boolean preserveOriginal)`.
    pub fn new(
        input: I,
        min_gram: i32,
        max_gram: i32,
        preserve_original: bool,
    ) -> Result<Self, AnalysisError> {
        Ok(NGramTokenFilter {
            input,
            g: GramState::new(min_gram, max_gram, preserve_original)?,
        })
    }
}

impl<I: TokenStream> TokenFilter for NGramTokenFilter<I> {
    crate::filter_input!();
    // Java: NGramTokenFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        loop {
            if self.g.cur_term.is_none() {
                match self.g.next_input(&mut self.input)? {
                    None => return Ok(false),
                    Some(true) => return Ok(true),
                    Some(false) => {}
                }
            }
            let g = &mut self.g;
            let term = g.cur_term.take().expect("set above");
            let count = term.len() as i32;
            if g.cur_gram_size > g.max_gram || g.cur_pos + g.cur_gram_size > count {
                g.cur_pos += 1;
                g.cur_gram_size = g.min_gram;
            }
            if g.cur_pos + g.cur_gram_size <= count {
                let (s, n) = (g.cur_pos as usize, g.cur_gram_size as usize);
                let inc = g.cur_pos_incr;
                g.emit(self.input.attributes_mut(), &term[s..s + n], inc)?;
                g.cur_pos_incr = 0;
                g.cur_gram_size += 1;
                g.cur_term = Some(term);
                return Ok(true);
            } else if g.preserve_original && count > g.max_gram {
                g.emit(self.input.attributes_mut(), &term, 0)?;
                return Ok(true);
            }
        }
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.g.cur_term = None;
        self.g.cur_pos_incr = 0;
        Ok(())
    }

    fn end_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.end()?;
        let inc = self.g.cur_pos_incr;
        self.input.attributes_mut().set_position_increment(inc)
    }
}

/// `org.apache.lucene.analysis.ngram.EdgeNGramTokenFilter`, streaming.
pub struct EdgeNGramTokenFilter<I> {
    input: I,
    g: GramState,
}

impl<I: TokenStream> EdgeNGramTokenFilter<I> {
    /// `new EdgeNGramTokenFilter(TokenStream, int minGram, int maxGram, boolean preserveOriginal)`.
    pub fn new(
        input: I,
        min_gram: i32,
        max_gram: i32,
        preserve_original: bool,
    ) -> Result<Self, AnalysisError> {
        Ok(EdgeNGramTokenFilter {
            input,
            g: GramState::new(min_gram, max_gram, preserve_original)?,
        })
    }
}

impl<I: TokenStream> TokenFilter for EdgeNGramTokenFilter<I> {
    crate::filter_input!();
    // Java: EdgeNGramTokenFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        loop {
            if self.g.cur_term.is_none() {
                match self.g.next_input(&mut self.input)? {
                    None => return Ok(false),
                    Some(true) => return Ok(true),
                    Some(false) => {}
                }
            }
            let g = &mut self.g;
            let term = g.cur_term.take().expect("set above");
            let count = term.len() as i32;
            if g.cur_gram_size <= count {
                if g.cur_gram_size <= g.max_gram {
                    let inc = g.cur_pos_incr;
                    g.emit(
                        self.input.attributes_mut(),
                        &term[..g.cur_gram_size as usize],
                        inc,
                    )?;
                    g.cur_pos_incr = 0;
                    g.cur_gram_size += 1;
                    g.cur_term = Some(term);
                    return Ok(true);
                } else if g.preserve_original {
                    g.emit(self.input.attributes_mut(), &term, 0)?;
                    return Ok(true);
                }
            }
        }
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.g.cur_term = None;
        self.g.cur_pos_incr = 0;
        Ok(())
    }

    fn end_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.end()?;
        let inc = self.g.cur_pos_incr;
        self.input.attributes_mut().set_position_increment(inc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::StrReader;
    use crate::util::canned::{render, Canned};

    fn tok(mut t: NGramTokenizer, text: &str) -> String {
        t.set_reader(Box::new(StrReader::new(text))).unwrap();
        render(&mut t)
    }

    #[test]
    fn tokenizers() {
        assert_eq!(
            tok(NGramTokenizer::new(1, 2).unwrap(), "abc"),
            "a:0:1:1:1 ab:0:2:1:1 b:1:2:1:1 bc:1:3:1:1 c:2:3:1:1|3|0"
        );
        assert_eq!(
            tok(NGramTokenizer::new(2, 2).unwrap(), "a😀"),
            "a😀:0:3:1:1|3|0"
        );
        assert_eq!(
            tok(NGramTokenizer::edge(1, 3).unwrap(), "abcd"),
            "a:0:1:1:1 ab:0:2:1:1 abc:0:3:1:1|4|0"
        );
        assert_eq!(tok(NGramTokenizer::new(3, 3).unwrap(), "ab"), "|2|0");
        // Past the 1024 + 2 * maxGram buffer.
        let long = "x".repeat(3000);
        let mut t = NGramTokenizer::new(1, 1).unwrap();
        t.set_reader(Box::new(StrReader::new(long))).unwrap();
        let mut n = 0;
        let end = crate::token_stream::consume(&mut t, |_| n += 1).unwrap();
        assert_eq!((n, end.end_offset()), (3000, 3000));
        let ws = |c: u32| c != u32::from(b' ');
        let mut t = NGramTokenizer::with_predicate(1, 2, true, ws).unwrap();
        t.set_reader(Box::new(StrReader::new("ab cd"))).unwrap();
        assert_eq!(
            render(&mut t),
            "a:0:1:1:1 ab:0:2:1:1 c:3:4:1:1 cd:3:5:1:1|5|0"
        );
        assert!(NGramTokenizer::new(0, 1).is_err());
        assert!(NGramTokenizer::new(3, 2).is_err());
        assert!(NGramTokenizer::new(1, i32::MAX).is_err());
    }

    #[test]
    fn filters() {
        let mut f = NGramTokenFilter::new(
            Canned::parse("abc:0:3:1:1 x:4:5:2:1 de:6:8:1:1|8|1"),
            2,
            2,
            false,
        )
        .unwrap();
        assert_eq!(render(&mut f), "ab:0:3:1:1 bc:0:3:0:1 de:6:8:3:1|8|0");
        let mut f = NGramTokenFilter::new(
            Canned::parse("abcd:0:4:1:1 x:5:6:1:1 y:7:8:1:1|8|0"),
            2,
            3,
            true,
        )
        .unwrap();
        assert_eq!(render(&mut f), "ab:0:4:1:1 abc:0:4:0:1 bc:0:4:0:1 bcd:0:4:0:1 cd:0:4:0:1 abcd:0:4:0:1 x:5:6:1:1 y:7:8:1:1|8|0");
        let mut f = EdgeNGramTokenFilter::new(
            Canned::parse("abcd:0:4:1:1 x:5:6:1:1 y:7:8:1:1|8|0"),
            2,
            3,
            true,
        )
        .unwrap();
        assert_eq!(
            render(&mut f),
            "ab:0:4:1:1 abc:0:4:0:1 abcd:0:4:0:1 x:5:6:1:1 y:7:8:1:1|8|0"
        );
        let mut f =
            EdgeNGramTokenFilter::new(Canned::parse("abcd:0:4:1:1 x:5:6:1:1|6|0"), 2, 2, false)
                .unwrap();
        assert_eq!(render(&mut f), "ab:0:4:1:1|6|1");
        assert!(EdgeNGramTokenFilter::new(Canned::parse(""), 2, 1, false).is_err());
    }
}
