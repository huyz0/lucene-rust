//! `org.apache.lucene.analysis.util.CharTokenizer` and the core
//! tokenizers built on it (`WhitespaceTokenizer`, `LetterTokenizer`,
//! `UnicodeWhitespaceTokenizer`), with `CharacterUtils.fill`'s
//! surrogate-aware 4096-char I/O buffer.
//!
//! Java subclasses override `isTokenChar(int)`; the port is generic over a
//! [`TokenChar`] predicate, and each Java subclass is a type alias with its
//! constructors (`WhitespaceTokenizer::new()`).

use crate::attributes::AttributeSource;
use crate::java_character::{self, code_point_at};
use crate::reader::CharReader;
use crate::token_stream::{TokenStream, Tokenizer, TokenizerInput};
use crate::AnalysisError;

/// `CharTokenizer.DEFAULT_MAX_WORD_LEN`.
pub const DEFAULT_MAX_WORD_LEN: usize = 255;
/// `StandardTokenizer.MAX_TOKEN_LENGTH_LIMIT`, `CharTokenizer`'s upper bound.
pub const MAX_TOKEN_LENGTH_LIMIT: usize = 1024 * 1024;
/// `CharTokenizer.IO_BUFFER_SIZE`.
const IO_BUFFER_SIZE: usize = 4096;

/// `CharTokenizer.isTokenChar(int)`.
pub trait TokenChar: Send {
    /// Whether code point `c` belongs to a token.
    fn is_token_char(&self, c: u32) -> bool;
}

impl<F: Fn(u32) -> bool + Send> TokenChar for F {
    fn is_token_char(&self, c: u32) -> bool {
        self(c)
    }
}

/// `WhitespaceTokenizer.isTokenChar`: `!Character.isWhitespace(c)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NotJavaWhitespace;

impl TokenChar for NotJavaWhitespace {
    #[inline(always)]
    fn is_token_char(&self, c: u32) -> bool {
        !java_character::is_whitespace(c)
    }
}

/// `LetterTokenizer.isTokenChar`: `Character.isLetter(c)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct JavaLetter;

impl TokenChar for JavaLetter {
    #[inline(always)]
    fn is_token_char(&self, c: u32) -> bool {
        java_character::is_letter(c)
    }
}

/// `UnicodeWhitespaceTokenizer.isTokenChar`:
/// `!UnicodeProps.WHITESPACE.get(c)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NotUnicodeWhitespace;

impl TokenChar for NotUnicodeWhitespace {
    #[inline]
    fn is_token_char(&self, c: u32) -> bool {
        !java_character::is_unicode_whitespace(c)
    }
}

/// `CharacterUtils.CharacterBuffer`.
pub(crate) struct CharacterBuffer {
    pub(crate) buffer: Vec<u16>,
    pub(crate) length: usize,
    last_trailing_high_surrogate: u16,
}

impl CharacterBuffer {
    /// `CharacterUtils.newCharacterBuffer(int)`.
    pub(crate) fn new(size: usize) -> Self {
        Self::with_buffer(vec![0; size])
    }

    /// `CharacterUtils.newCharacterBuffer(int)` over an allocated buffer.
    pub(crate) fn with_buffer(buffer: Vec<u16>) -> Self {
        assert!(buffer.len() >= 2, "buffersize must be >= 2");
        CharacterBuffer {
            buffer,
            length: 0,
            last_trailing_high_surrogate: 0,
        }
    }

    /// `CharacterBuffer.reset()`.
    pub(crate) fn reset(&mut self) {
        self.length = 0;
        self.last_trailing_high_surrogate = 0;
    }

    /// `CharacterUtils.fill(CharacterBuffer, Reader)`: reads until the buffer
    /// is full or the reader is exhausted, holding back a trailing high
    /// surrogate for the next fill. Returns whether the buffer was filled.
    pub(crate) fn fill(&mut self, reader: &mut dyn CharReader) -> Result<bool, AnalysisError> {
        let n = self.buffer.len();
        self.fill_n(reader, n)
    }

    /// `CharacterUtils.fill(CharacterBuffer, Reader, int numChars)`: as
    /// [`Self::fill`], reading at most `num_chars` units.
    pub(crate) fn fill_n(
        &mut self,
        reader: &mut dyn CharReader,
        num_chars: usize,
    ) -> Result<bool, AnalysisError> {
        assert!(
            (2..=self.buffer.len()).contains(&num_chars),
            "numChars must be >= 2 and <= the buffer size"
        );
        let offset = if self.last_trailing_high_surrogate != 0 {
            self.buffer[0] = self.last_trailing_high_surrogate;
            self.last_trailing_high_surrogate = 0;
            1
        } else {
            0
        };
        let read = read_fully(reader, &mut self.buffer[offset..num_chars])?;
        self.length = offset + read;
        let result = self.length == num_chars;
        if self.length < num_chars {
            return Ok(result);
        }
        if java_character::is_high_surrogate(self.buffer[self.length - 1]) {
            self.length -= 1;
            self.last_trailing_high_surrogate = self.buffer[self.length];
        }
        Ok(result)
    }
}

/// `CharacterUtils.readFully`.
pub(crate) fn read_fully(
    reader: &mut dyn CharReader,
    dest: &mut [u16],
) -> Result<usize, AnalysisError> {
    let mut read = 0;
    while read < dest.len() {
        let r = reader.read(&mut dest[read..])?;
        if r == 0 {
            break;
        }
        read += r;
    }
    Ok(read)
}

/// `org.apache.lucene.analysis.util.CharTokenizer`: maximal runs of
/// [`TokenChar`] code points, cut at `maxTokenLen` UTF-16 units.
pub struct CharTokenizer<P> {
    atts: AttributeSource,
    input: TokenizerInput,
    predicate: P,
    offset: usize,
    buffer_index: usize,
    data_len: usize,
    final_offset: i32,
    max_token_len: usize,
    io_buffer: CharacterBuffer,
    /// Scanning the reader's whole text as UTF-8 from `byte_pos` (see
    /// `increment_utf8`); `offset` is then the UTF-16 offset of `byte_pos`.
    utf8: bool,
    byte_pos: usize,
}

/// `org.apache.lucene.analysis.core.WhitespaceTokenizer`.
pub type WhitespaceTokenizer = CharTokenizer<NotJavaWhitespace>;
/// `org.apache.lucene.analysis.core.LetterTokenizer`.
pub type LetterTokenizer = CharTokenizer<JavaLetter>;
/// `org.apache.lucene.analysis.core.UnicodeWhitespaceTokenizer`.
pub type UnicodeWhitespaceTokenizer = CharTokenizer<NotUnicodeWhitespace>;

impl<P: TokenChar + Default> Default for CharTokenizer<P> {
    fn default() -> Self {
        Self::from_predicate(P::default())
    }
}

impl<P: TokenChar + Default> CharTokenizer<P> {
    /// The no-argument constructor (`maxTokenLen` 255).
    pub fn new() -> Self {
        Self::default()
    }

    /// The `(int maxTokenLen)` constructor.
    pub fn with_max_token_len(max_token_len: usize) -> Result<Self, AnalysisError> {
        Self::from_predicate_with_max(P::default(), max_token_len)
    }
}

impl<P: TokenChar> CharTokenizer<P> {
    /// `CharTokenizer.fromTokenCharPredicate`.
    pub fn from_predicate(predicate: P) -> Self {
        CharTokenizer {
            atts: AttributeSource::new(),
            input: TokenizerInput::new(),
            predicate,
            offset: 0,
            buffer_index: 0,
            data_len: 0,
            final_offset: 0,
            max_token_len: DEFAULT_MAX_WORD_LEN,
            io_buffer: CharacterBuffer::new(IO_BUFFER_SIZE),
            utf8: false,
            byte_pos: 0,
        }
    }

    /// `CharTokenizer(AttributeFactory, int maxTokenLen)` over a predicate.
    pub fn from_predicate_with_max(
        predicate: P,
        max_token_len: usize,
    ) -> Result<Self, AnalysisError> {
        if max_token_len > MAX_TOKEN_LENGTH_LIMIT || max_token_len == 0 {
            return Err(AnalysisError::IllegalArgument(format!(
                "maxTokenLen must be greater than 0 and less than {MAX_TOKEN_LENGTH_LIMIT} passed: {max_token_len}"
            )));
        }
        let mut t = Self::from_predicate(predicate);
        t.max_token_len = max_token_len;
        Ok(t)
    }

    fn correct(&self, off: usize) -> i32 {
        self.input
            .correct_offset(i32::try_from(off).unwrap_or(i32::MAX))
    }
}

/// `CharTokenizer.incrementToken` over a plain string input: the same tokens,
/// offsets and cuts as reading its UTF-16 units a buffer at a time, scanned
/// where the text lies. A string reader's offsets are never corrected, and
/// its text holds no unpaired surrogate. `offset` is the UTF-16 offset of
/// `byte_pos`.
fn scan_utf8<P: TokenChar>(
    text: &str,
    predicate: &P,
    atts: &mut AttributeSource,
    offset: &mut usize,
    byte_pos: &mut usize,
    final_offset: &mut i32,
    max_token_len: usize,
) -> Result<bool, AnalysisError> {
    let bytes = text.as_bytes();
    let mut i = *byte_pos;
    let mut off = *offset;
    // skip the characters that are not token characters
    let start = loop {
        if i >= bytes.len() {
            *byte_pos = i;
            *offset = off;
            *final_offset = i32::try_from(off).unwrap_or(i32::MAX);
            return Ok(false);
        }
        let (c, n) = decode_utf8(bytes, i);
        if predicate.is_token_char(c) {
            break i;
        }
        i += n;
        off += if n == 4 { 2 } else { 1 };
    };
    let start_off = off;
    let mut length = 0usize;
    while i < bytes.len() {
        // A run of ASCII token characters: one unit and one byte each.
        let run_end = bytes.len().min(i + (max_token_len - length));
        let run = bytes[i..run_end]
            .iter()
            .position(|&b| b >= 0x80 || !predicate.is_token_char(u32::from(b)))
            .unwrap_or(run_end - i);
        i += run;
        off += run;
        length += run;
        if length >= max_token_len || i >= bytes.len() || bytes[i] < 0x80 {
            break;
        }
        let (c, n) = decode_utf8(bytes, i);
        if !predicate.is_token_char(c) {
            break;
        }
        let units = if n == 4 { 2 } else { 1 };
        i += n;
        off += units;
        length += units;
        if length >= max_token_len {
            break;
        }
    }
    *byte_pos = i;
    *offset = off;
    atts.set_term(&text[start..i]);
    let (s, e) = (
        i32::try_from(start_off).unwrap_or(i32::MAX),
        i32::try_from(off).unwrap_or(i32::MAX),
    );
    *final_offset = e;
    atts.set_offset(s, e)?;
    Ok(true)
}

/// The code point at byte `i` of valid UTF-8 and its length in bytes.
#[inline(always)]
fn decode_utf8(bytes: &[u8], i: usize) -> (u32, usize) {
    let b0 = u32::from(bytes[i]);
    if b0 < 0x80 {
        return (b0, 1);
    }
    let cont = |k: usize| u32::from(bytes[i + k]) & 0x3F;
    if b0 < 0xE0 {
        (((b0 & 0x1F) << 6) | cont(1), 2)
    } else if b0 < 0xF0 {
        (((b0 & 0x0F) << 12) | (cont(1) << 6) | cont(2), 3)
    } else {
        (
            ((b0 & 0x07) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3),
            4,
        )
    }
}

/// `CharTokenizer.fromSeparatorCharPredicate`: tokens are the runs of code
/// points `separator` rejects.
pub fn from_separator_char_predicate<F: Fn(u32) -> bool + Send>(
    separator: F,
) -> CharTokenizer<impl TokenChar> {
    CharTokenizer::from_predicate(move |c: u32| !separator(c))
}

impl<P: TokenChar> TokenStream for CharTokenizer<P> {
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

    // Java: CharTokenizer.incrementToken
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        self.atts.clear_attributes();
        if self.utf8 {
            let Self {
                input,
                atts,
                predicate,
                offset,
                byte_pos,
                final_offset,
                max_token_len,
                ..
            } = self;
            if let Some(text) = input.whole_text() {
                return scan_utf8(
                    text,
                    predicate,
                    atts,
                    offset,
                    byte_pos,
                    final_offset,
                    *max_token_len,
                );
            }
        }
        let mut length = 0usize;
        let mut start = 0usize;
        let mut end = 0usize;
        loop {
            if self.buffer_index >= self.data_len {
                self.offset += self.data_len;
                let reader = self.input.reader()?;
                self.io_buffer.fill(reader)?;
                if self.io_buffer.length == 0 {
                    self.data_len = 0; // so next offset += dataLen won't decrement offset
                    if length > 0 {
                        break;
                    } else {
                        self.final_offset = self.correct(self.offset);
                        return Ok(false);
                    }
                }
                self.data_len = self.io_buffer.length;
                self.buffer_index = 0;
            }
            let c = code_point_at(
                &self.io_buffer.buffer,
                self.buffer_index,
                self.io_buffer.length,
            );
            let char_count = java_character::char_count(c);
            self.buffer_index += char_count;
            if self.predicate.is_token_char(c) {
                if length == 0 {
                    start = self.offset + self.buffer_index - char_count;
                    end = start;
                }
                end += char_count;
                // Java appends to the term's char[]; the term here is UTF-8,
                // so the code point is pushed whole and `length` counts its
                // UTF-16 units. `code_point_at` pairs every surrogate it can
                // (a trailing high surrogate is held back for the next
                // fill), so a lone one is unpaired in Java's buffer too and
                // becomes U+FFFD, as `set_term_utf16` maps it.
                self.atts
                    .term_mut()
                    .push(char::from_u32(c).unwrap_or(char::REPLACEMENT_CHARACTER));
                length += char_count;
                if length >= self.max_token_len {
                    break;
                }
            } else if length > 0 {
                break;
            }
        }
        let s = self.correct(start);
        self.final_offset = self.correct(end);
        self.atts.set_offset(s, self.final_offset)?;
        Ok(true)
    }

    // Java: CharTokenizer.end
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        self.atts.set_offset(self.final_offset, self.final_offset)
    }

    // Java: CharTokenizer.reset
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset();
        self.buffer_index = 0;
        self.offset = 0;
        self.data_len = 0;
        self.final_offset = 0;
        self.io_buffer.reset();
        // A plain string input is scanned as UTF-8, where it lies.
        self.utf8 = self.input.whole_text().is_some();
        self.byte_pos = 0;
        Ok(())
    }

    fn close(&mut self) -> Result<(), AnalysisError> {
        self.input.close()
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        Some(self)
    }
}

impl<P: TokenChar> Tokenizer for CharTokenizer<P> {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.input.set_reader(input)
    }

    fn spare_reader(&mut self) -> Option<Box<crate::reader::StrReader>> {
        self.input.take_spare()
    }
}

#[cfg(test)]
mod tests;
