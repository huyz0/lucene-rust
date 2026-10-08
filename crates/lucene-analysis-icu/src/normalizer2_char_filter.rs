//! `org.apache.lucene.analysis.icu.ICUNormalizer2CharFilter`: normalizes
//! the character stream before tokenization, reading 128 units at a time
//! (`CharacterUtils.fill`), passing through the quick-check-"yes" prefix
//! unchanged and normalizing up to each normalization boundary, with
//! `BaseCharFilter` offset corrections for every length change.

use lucene_analysis::charfilter::OffsetCorrections;
use lucene_analysis::{AnalysisError, CharFilter, CharReader};

use crate::icu4j::normalizer2::Normalizer2;
use crate::icu4j::utf16;

/// `CharacterUtils.CharacterBuffer` and `fill`.
struct CharacterBuffer {
    buffer: Vec<u16>,
    length: usize,
    last_trailing_high_surrogate: Option<u16>,
}

impl CharacterBuffer {
    /// `CharacterUtils.fill(buffer, reader)`: `true` if the buffer was
    /// filled completely; a trailing high surrogate is held back for the
    /// next fill.
    fn fill(&mut self, reader: &mut dyn CharReader) -> Result<bool, AnalysisError> {
        let num_chars = self.buffer.len();
        let offset = match self.last_trailing_high_surrogate.take() {
            Some(h) => {
                self.buffer[0] = h;
                1
            }
            None => 0,
        };
        let mut filled = offset;
        while filled < num_chars {
            let n = reader.read(&mut self.buffer[filled..])?;
            if n == 0 {
                break;
            }
            filled = filled.saturating_add(n).min(num_chars);
        }
        self.length = filled;
        if filled < num_chars {
            return Ok(false);
        }
        let last = filled.saturating_sub(1);
        if utf16::is_lead(i32::from(self.buffer[last])) {
            self.last_trailing_high_surrogate = Some(self.buffer[last]);
            self.length = last;
        }
        Ok(true)
    }
}

/// `ICUNormalizer2CharFilter`.
pub struct ICUNormalizer2CharFilter<R> {
    input: R,
    normalizer: Normalizer2,
    input_buffer: Vec<u16>,
    result_buffer: Vec<u16>,
    input_finished: bool,
    after_quick_check_yes: bool,
    checked_input_boundary: usize,
    char_count: i32,
    tmp: CharacterBuffer,
    corrections: OffsetCorrections,
}

impl<R: CharReader> ICUNormalizer2CharFilter<R> {
    /// `ICUNormalizer2CharFilter(Reader)`: NFKC_Casefold.
    pub fn new(input: R) -> Self {
        Self::with_normalizer(input, Normalizer2::nfkc_casefold())
    }

    /// `ICUNormalizer2CharFilter(Reader, Normalizer2)`.
    pub fn with_normalizer(input: R, normalizer: Normalizer2) -> Self {
        Self::with_buffer_size(input, normalizer, 128)
    }

    /// `ICUNormalizer2CharFilter(Reader, Normalizer2, int bufferSize)`
    /// (package-private in Java; at least 2).
    pub fn with_buffer_size(input: R, normalizer: Normalizer2, buffer_size: usize) -> Self {
        ICUNormalizer2CharFilter {
            input,
            normalizer,
            input_buffer: Vec::new(),
            result_buffer: Vec::new(),
            input_finished: false,
            after_quick_check_yes: false,
            checked_input_boundary: 0,
            char_count: 0,
            tmp: CharacterBuffer {
                buffer: vec![0; buffer_size.max(2)],
                length: 0,
                last_trailing_high_surrogate: None,
            },
            corrections: OffsetCorrections::default(),
        }
    }

    /// `readInputToBuffer()`.
    fn read_input_to_buffer(&mut self) -> Result<(), AnalysisError> {
        if !self.tmp.fill(&mut self.input)? {
            self.input_finished = true;
        }
        self.input_buffer
            .extend_from_slice(&self.tmp.buffer[..self.tmp.length]);
        self.checked_input_boundary = self.checked_input_boundary.saturating_sub(1);
        Ok(())
    }

    /// `readAndNormalizeFromInput()`.
    fn read_and_normalize_from_input(&mut self) -> usize {
        if self.input_buffer.is_empty() {
            self.after_quick_check_yes = false;
            return 0;
        }
        if !self.after_quick_check_yes {
            let res_len = self.read_from_input_while_span_quick_check_yes();
            if res_len > 0 {
                return res_len;
            }
        }
        let res_len = self.read_from_io_normalize_upto_boundary();
        if res_len > 0 {
            self.after_quick_check_yes = false;
        }
        res_len
    }

    /// `readFromInputWhileSpanQuickCheckYes()`.
    // ARITH: end <= input_buffer.len() and a code point's units are at most
    // end; char_count counts units of a Java-sized (i32) stream.
    #[allow(clippy::arithmetic_side_effects)]
    fn read_from_input_while_span_quick_check_yes(&mut self) -> usize {
        self.after_quick_check_yes = true;
        let mut end = self.normalizer.span_quick_check_yes(&self.input_buffer);
        if end > 0 {
            if end == self.input_buffer.len() {
                let mut cp = utf16::code_point_before(&self.input_buffer, end);
                if !self.normalizer.has_boundary_after(cp) {
                    // The quick check holds through the buffer's end, but
                    // the end is not a normalization boundary: back off.
                    self.after_quick_check_yes = false;
                    end -= if cp > 0xffff { 2 } else { 1 };
                    while end > 0 && !self.normalizer.has_boundary_before(cp) {
                        cp = utf16::code_point_before(&self.input_buffer, end);
                        end -= if cp > 0xffff { 2 } else { 1 };
                    }
                    if end == 0 {
                        return 0;
                    }
                }
            }
            self.result_buffer
                .extend_from_slice(&self.input_buffer[..end]);
            self.input_buffer.drain(..end);
            self.checked_input_boundary = self.checked_input_boundary.saturating_sub(end);
            self.char_count += end as i32;
        }
        end
    }

    /// `readFromIoNormalizeUptoBoundary()`.
    // ARITH: checked_input_boundary < buf_len before each step.
    #[allow(clippy::arithmetic_side_effects)]
    fn read_from_io_normalize_upto_boundary(&mut self) -> usize {
        if self.input_buffer.is_empty() {
            return 0;
        }
        let mut found_boundary = false;
        let buf_len = self.input_buffer.len();
        while self.checked_input_boundary < buf_len {
            let c = utf16::code_point_at(&self.input_buffer, self.checked_input_boundary);
            self.checked_input_boundary += if c > 0xffff { 2 } else { 1 };
            if self.checked_input_boundary < buf_len
                && self.normalizer.has_boundary_before(utf16::code_point_at(
                    &self.input_buffer,
                    self.checked_input_boundary,
                ))
            {
                found_boundary = true;
                break;
            }
        }
        if !found_boundary && self.checked_input_boundary >= buf_len && self.input_finished {
            found_boundary = true;
            self.checked_input_boundary = buf_len;
        }
        if !found_boundary {
            return 0;
        }
        self.normalize_input_upto(self.checked_input_boundary)
    }

    /// `normalizeInputUpto(length)`.
    fn normalize_input_upto(&mut self, length: usize) -> usize {
        let length = length.min(self.input_buffer.len());
        let dest_orig_len = self.result_buffer.len();
        self.normalizer
            .normalize_second_and_append(&mut self.result_buffer, &self.input_buffer[..length]);
        self.input_buffer.drain(..length);
        self.checked_input_boundary = self.checked_input_boundary.saturating_sub(length);
        let result_length = self.result_buffer.len().saturating_sub(dest_orig_len);
        self.record_offset_diff(length, result_length);
        result_length
    }

    /// `recordOffsetDiff(inputLength, outputLength)`.
    // ARITH: lengths of buffers within one Java-sized (i32) stream.
    #[allow(clippy::arithmetic_side_effects)]
    fn record_offset_diff(&mut self, input_length: usize, output_length: usize) {
        let (input_length, output_length) = (input_length as i32, output_length as i32);
        if input_length == output_length {
            self.char_count += output_length;
            return;
        }
        let diff = input_length - output_length;
        let cumu_diff = self.corrections.last_cumulative_diff();
        if diff < 0 {
            for i in 1..=-diff {
                self.corrections.add(self.char_count + i, cumu_diff - i);
            }
        } else {
            self.corrections
                .add(self.char_count + output_length, cumu_diff + diff);
        }
        self.char_count += output_length;
    }

    /// `outputFromResultBuffer(cbuf, begin, len)`.
    fn output_from_result_buffer(&mut self, buf: &mut [u16]) -> usize {
        let len = self.result_buffer.len().min(buf.len());
        buf[..len].copy_from_slice(&self.result_buffer[..len]);
        self.result_buffer.drain(..len);
        len
    }
}

impl<R: CharReader> CharFilter for ICUNormalizer2CharFilter<R> {
    fn input(&self) -> &dyn CharReader {
        &self.input
    }

    fn input_mut(&mut self) -> &mut dyn CharReader {
        &mut self.input
    }

    // Java: ICUNormalizer2CharFilter.read(char[], int, int)
    fn read_filtered(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        if buf.is_empty() {
            return Err(AnalysisError::IllegalArgument("len <= 0".into()));
        }
        while !self.input_finished
            || !self.input_buffer.is_empty()
            || !self.result_buffer.is_empty()
        {
            if !self.result_buffer.is_empty() {
                let n = self.output_from_result_buffer(buf);
                if n > 0 {
                    return Ok(n);
                }
            }
            if self.read_and_normalize_from_input() > 0 {
                let n = self.output_from_result_buffer(buf);
                if n > 0 {
                    return Ok(n);
                }
            }
            self.read_input_to_buffer()?;
        }
        Ok(0)
    }

    fn correct(&self, current_off: i32) -> i32 {
        self.corrections.correct(current_off)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::StrReader;

    fn drain(f: &mut dyn CharReader, chunk: usize) -> String {
        let mut out = Vec::new();
        let mut buf = vec![0u16; chunk];
        loop {
            let n = f.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
        }
        String::from_utf16(&out).unwrap()
    }

    #[test]
    fn normalizes_across_buffers() {
        let text = "ﬁ".repeat(100) + "Ａｂ" + &"e\u{301}".repeat(80);
        for chunk in [1, 3, 128, 1000] {
            let mut f = ICUNormalizer2CharFilter::new(StrReader::new(text.clone()));
            let out = drain(&mut f, chunk);
            assert_eq!(out, "fi".repeat(100) + "ab" + &"\u{e9}".repeat(80));
            assert_eq!(f.correct_offset(0), 0);
            assert!(f.correct_offset(200) < 200);
        }
        let mut f = ICUNormalizer2CharFilter::with_buffer_size(
            StrReader::new("\u{10400}".repeat(5)),
            Normalizer2::nfc(),
            3,
        );
        assert_eq!(drain(&mut f, 2), "\u{10400}".repeat(5));
        assert!(f.read(&mut []).is_err());
        f.close().unwrap();
    }
}
