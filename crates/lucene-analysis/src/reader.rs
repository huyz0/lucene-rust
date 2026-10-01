//! Character input: Java's `java.io.Reader` as tokenizers see it, and
//! `org.apache.lucene.analysis.CharFilter`.
//!
//! Java tokenizers read UTF-16 `char`s, and every offset analysis reports is
//! an index into that `char` sequence. The port keeps the unit: a
//! [`CharReader`] fills a `u16` buffer, so a tokenizer's offsets are Java's
//! exactly, and a [`CharFilter`]'s offset corrections are expressed in the
//! same code units Java's are.

use crate::AnalysisError;

/// Java's `java.io.Reader`, as analysis uses it: a source of UTF-16 code
/// units, plus the `CharFilter.correctOffset` chain a `Tokenizer` consults.
///
/// Rust-forced changes: `read(char[], off, len)` returning `-1` at end of
/// input returns `Ok(0)` here (for a non-empty `buf`), in the manner of
/// `std::io::Read`; `Tokenizer.correctOffset`'s `input instanceof CharFilter`
/// test is [`Self::correct_offset`], the identity for a plain reader and the
/// chained correction for a [`CharFilter`].
pub trait CharReader: Send {
    /// Reads up to `buf.len()` code units; `Ok(0)` means end of input.
    fn read(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError>;

    /// `Reader.close()`.
    fn close(&mut self) -> Result<(), AnalysisError> {
        Ok(())
    }

    /// `CharFilter.correctOffset(int)` for a char filter; the identity for
    /// any other reader.
    fn correct_offset(&self, current_off: i32) -> i32 {
        current_off
    }

    /// The whole input as a Rust string, for a reader that holds it as one
    /// and has not been read from yet, with no offset correction (a
    /// [`StrReader`]); `None` for every other reader. A tokenizer may then
    /// scan the UTF-8 text directly instead of reading UTF-16 code units --
    /// provided it reports exactly what reading them would have given.
    fn whole_text(&self) -> Option<&str> {
        None
    }
}

impl CharReader for Box<dyn CharReader> {
    fn read(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        (**self).read(buf)
    }

    fn close(&mut self) -> Result<(), AnalysisError> {
        (**self).close()
    }

    fn correct_offset(&self, current_off: i32) -> i32 {
        (**self).correct_offset(current_off)
    }

    fn whole_text(&self) -> Option<&str> {
        (**self).whole_text()
    }
}

/// Java's `StringReader` (and `Analyzer`'s `ReusableStringReader`): a Rust
/// string read as the UTF-16 code units Java's `String` holds.
///
/// Encoding happens as the reader is drained, with an ASCII fast path, so the
/// text is never materialised as a second `Vec<u16>`; a supplementary
/// character that straddles the end of `buf` leaves its low surrogate pending
/// for the next call, as a Java `String`'s `char`s would.
#[derive(Debug, Clone, Default)]
pub struct StrReader {
    text: String,
    /// Byte position of the next unread `char`.
    pos: usize,
    /// A low surrogate owed from a supplementary character split across two
    /// reads.
    pending_low: Option<u16>,
}

impl StrReader {
    pub fn new(text: impl Into<String>) -> Self {
        StrReader {
            text: text.into(),
            pos: 0,
            pending_low: None,
        }
    }

    /// `ReusableStringReader.setValue`: reuse this reader's allocation.
    pub fn set_value(&mut self, text: &str) {
        self.text.clear();
        self.text.push_str(text);
        self.pos = 0;
        self.pending_low = None;
    }
}

impl CharReader for StrReader {
    fn whole_text(&self) -> Option<&str> {
        (self.pos == 0 && self.pending_low.is_none()).then_some(self.text.as_str())
    }

    fn read(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        let mut n = 0;
        if buf.is_empty() {
            return Ok(0);
        }
        if let Some(low) = self.pending_low.take() {
            buf[0] = low;
            n = 1;
        }
        let bytes = self.text.as_bytes();
        // ASCII run: one unit per byte.
        while n < buf.len() && self.pos < bytes.len() && bytes[self.pos] < 0x80 {
            buf[n] = bytes[self.pos] as u16;
            n += 1;
            self.pos += 1;
        }
        while n < buf.len() && self.pos < bytes.len() {
            // `text` is valid UTF-8, so the lead byte gives the length and
            // every continuation byte is present.
            let b0 = u32::from(bytes[self.pos]);
            let cont = |k: usize| u32::from(bytes[self.pos + k]) & 0x3F;
            let (cp, len) = if b0 < 0x80 {
                (b0, 1)
            } else if b0 < 0xE0 {
                (((b0 & 0x1F) << 6) | cont(1), 2)
            } else if b0 < 0xF0 {
                (((b0 & 0x0F) << 12) | (cont(1) << 6) | cont(2), 3)
            } else {
                (
                    ((b0 & 0x07) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3),
                    4,
                )
            };
            self.pos += len;
            if cp < 0x10000 {
                buf[n] = cp as u16;
                n += 1;
            } else {
                let v = cp - 0x10000;
                buf[n] = 0xD800 | (v >> 10) as u16;
                n += 1;
                let low = 0xDC00 | (v & 0x3FF) as u16;
                if n < buf.len() {
                    buf[n] = low;
                    n += 1;
                } else {
                    self.pending_low = Some(low);
                }
            }
        }
        Ok(n)
    }
}

/// `org.apache.lucene.analysis.CharFilter`: a reader that rewrites its input
/// before tokenization and can map an offset in its output back to the
/// corresponding offset in its input.
///
/// Implement [`Self::read_filtered`] (Java's `read(char[], int, int)`) and
/// [`Self::correct`]; every `CharFilter` is then a [`CharReader`] whose
/// [`CharReader::correct_offset`] is Java's final `correctOffset` -- this
/// filter's `correct`, then the input's own correction if the input is a
/// char filter too -- and whose `close` closes the input.
pub trait CharFilter: Send {
    /// `CharFilter.input`.
    fn input(&self) -> &dyn CharReader;

    /// `CharFilter.input`, mutably (for reading and closing).
    fn input_mut(&mut self) -> &mut dyn CharReader;

    /// `Reader.read(char[], int, int)`: `Ok(0)` at end of input.
    fn read_filtered(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError>;

    /// `CharFilter.correct(int)`: this filter's own output-to-input offset
    /// map.
    fn correct(&self, current_off: i32) -> i32;
}

impl<F: CharFilter> CharReader for F {
    fn read(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        self.read_filtered(buf)
    }

    fn close(&mut self) -> Result<(), AnalysisError> {
        self.input_mut().close()
    }

    /// Java: `CharFilter.correctOffset` -- "chains the corrected offset
    /// through the input CharFilter(s)".
    fn correct_offset(&self, current_off: i32) -> i32 {
        let corrected = self.correct(current_off);
        self.input().correct_offset(corrected)
    }
}

/// Reads everything left in `reader` into a `String` (unpaired surrogates
/// become U+FFFD).
pub fn read_to_string(reader: &mut dyn CharReader) -> Result<String, AnalysisError> {
    let mut units = Vec::new();
    let mut buf = [0u16; 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        units.extend_from_slice(&buf[..n]);
    }
    Ok(char::decode_utf16(units)
        .map(|r| r.unwrap_or(char::REPLACEMENT_CHARACTER))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(r: &mut dyn CharReader, chunk: usize) -> Vec<u16> {
        let mut out = Vec::new();
        let mut buf = vec![0u16; chunk];
        loop {
            let n = r.read(&mut buf).unwrap();
            if n == 0 {
                return out;
            }
            out.extend_from_slice(&buf[..n]);
        }
    }

    #[test]
    fn str_reader_yields_javas_utf16_in_any_chunk_size() {
        let text = "ab é 😀x 世界 👨‍👩‍👧";
        let want: Vec<u16> = text.encode_utf16().collect();
        for chunk in 1..8 {
            let mut r = StrReader::new(text);
            assert_eq!(drain(&mut r, chunk), want, "chunk {chunk}");
        }
        let mut r = StrReader::new(text);
        assert_eq!(r.read(&mut []).unwrap(), 0);
        assert_eq!(r.whole_text(), Some(text));
        r.set_value("zz");
        assert_eq!(r.whole_text(), Some("zz"));
        assert_eq!(drain(&mut r, 3), vec![b'z' as u16; 2]);
        // Once read from, the reader is no longer a whole text.
        assert_eq!(r.whole_text(), None);
        r.close().unwrap();
        assert_eq!(r.correct_offset(7), 7);
    }

    /// Doubles every `a`; corrects offsets past each inserted unit.
    struct Doubler {
        input: Box<dyn CharReader>,
        inserted_at: Vec<i32>,
        out: Option<Vec<u16>>,
        served: usize,
    }

    impl Doubler {
        fn new(input: Box<dyn CharReader>) -> Self {
            Doubler {
                input,
                inserted_at: vec![],
                out: None,
                served: 0,
            }
        }
    }

    impl CharFilter for Doubler {
        fn input(&self) -> &dyn CharReader {
            &*self.input
        }
        fn input_mut(&mut self) -> &mut dyn CharReader {
            &mut *self.input
        }
        fn read_filtered(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
            if self.out.is_none() {
                let src: Vec<u16> = read_to_string(&mut *self.input)?.encode_utf16().collect();
                let mut out = Vec::new();
                for u in src {
                    out.push(u);
                    if u == b'a' as u16 {
                        out.push(u);
                        self.inserted_at.push(out.len() as i32 - 1);
                    }
                }
                self.out = Some(out);
            }
            let out = self.out.as_ref().unwrap();
            let n = buf.len().min(out.len() - self.served);
            buf[..n].copy_from_slice(&out[self.served..self.served + n]);
            self.served += n;
            Ok(n)
        }
        fn correct(&self, off: i32) -> i32 {
            off - self.inserted_at.iter().filter(|&&p| p < off).count() as i32
        }
    }

    #[test]
    fn char_filters_chain_their_corrections() {
        let inner = Doubler::new(Box::new(StrReader::new("xa")));
        let mut outer = Doubler::new(Box::new(inner));
        let out = drain(&mut outer, 4);
        assert_eq!(String::from_utf16(&out).unwrap(), "xaaaa");
        // end of the output (5) maps to the end of the original input (2).
        assert_eq!(outer.correct_offset(5), 2);
        assert_eq!(outer.correct_offset(1), 1);
        outer.close().unwrap();
        let mut boxed: Box<dyn CharReader> = Box::new(StrReader::new("q"));
        assert_eq!(read_to_string(&mut boxed).unwrap(), "q");
        assert_eq!(boxed.correct_offset(3), 3);
        boxed.close().unwrap();
    }
}
