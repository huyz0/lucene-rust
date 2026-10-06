//! `org.apache.lucene.analysis.reverse`: `ReverseStringFilter` -- each term
//! reversed (surrogate pairs kept in order), optionally after a marker char
//! (for leading-wildcard indexing).

use crate::token_stream::{TokenFilter, TokenStream};
use crate::AnalysisError;

/// `ReverseStringFilter.NOMARKER`.
const NOMARKER: u16 = 0xFFFF;
/// `ReverseStringFilter.START_OF_HEADING_MARKER`.
pub const START_OF_HEADING_MARKER: u16 = 0x0001;
/// `ReverseStringFilter.INFORMATION_SEPARATOR_MARKER`.
pub const INFORMATION_SEPARATOR_MARKER: u16 = 0x001F;
/// `ReverseStringFilter.PUA_EC00_MARKER`.
pub const PUA_EC00_MARKER: u16 = 0xEC00;
/// `ReverseStringFilter.RTL_DIRECTION_MARKER`.
pub const RTL_DIRECTION_MARKER: u16 = 0x200F;

fn is_surrogate_pair(high: u16, low: u16) -> bool {
    (0xD800..=0xDBFF).contains(&high) && (0xDC00..=0xDFFF).contains(&low)
}

/// `ReverseStringFilter.reverse(char[], int start, int len)`: Apache
/// Harmony's `reverse0`, a surrogate pair staying a pair.
pub fn reverse(buffer: &mut [u16], start: usize, len: usize) {
    if len < 2 {
        return;
    }
    let mut end = start + len - 1;
    let mut front_high = buffer[start];
    let mut end_low = buffer[end];
    let (mut allow_front_sur, mut allow_end_sur) = (true, true);
    let mid = start + (len >> 1);
    let mut i = start;
    while i < mid {
        let front_low = buffer[i + 1];
        let end_high = buffer[end - 1];
        let sur_at_front = allow_front_sur && is_surrogate_pair(front_high, front_low);
        if sur_at_front && len < 3 {
            return;
        }
        let sur_at_end = allow_end_sur && is_surrogate_pair(end_high, end_low);
        allow_front_sur = true;
        allow_end_sur = true;
        if sur_at_front == sur_at_end {
            if sur_at_front {
                buffer[end] = front_low;
                end -= 1;
                buffer[end] = front_high;
                buffer[i] = end_high;
                i += 1;
                buffer[i] = end_low;
                front_high = buffer[i + 1];
                end_low = buffer[end - 1];
            } else {
                buffer[end] = front_high;
                buffer[i] = end_low;
                front_high = front_low;
                end_low = end_high;
            }
        } else if sur_at_front {
            buffer[end] = front_low;
            buffer[i] = end_low;
            end_low = end_high;
            allow_front_sur = false;
        } else {
            buffer[end] = front_high;
            buffer[i] = end_high;
            front_high = front_low;
            allow_end_sur = false;
        }
        i += 1;
        end -= 1;
    }
    if len & 1 == 1 && !(allow_front_sur && allow_end_sur) {
        buffer[end] = if allow_front_sur { end_low } else { front_high };
    }
}

/// `ReverseStringFilter.reverse(String)`.
pub fn reverse_string(input: &str) -> String {
    let mut b: Vec<u16> = input.encode_utf16().collect();
    let n = b.len();
    reverse(&mut b, 0, n);
    String::from_utf16_lossy(&b)
}

/// `ReverseStringFilter`.
pub struct ReverseStringFilter<I> {
    input: I,
    marker: u16,
    buf: Vec<u16>,
}

impl<I: TokenStream> ReverseStringFilter<I> {
    /// `new ReverseStringFilter(TokenStream)`: no marker.
    pub fn new(input: I) -> Self {
        Self::with_marker(input, NOMARKER)
    }

    /// `new ReverseStringFilter(TokenStream, char marker)`: `marker` is
    /// appended before reversing (so it leads the reversed term).
    pub fn with_marker(input: I, marker: u16) -> Self {
        ReverseStringFilter {
            input,
            marker,
            buf: Vec::new(),
        }
    }
}

impl<I: TokenStream> TokenFilter for ReverseStringFilter<I> {
    crate::filter_input!();

    // Java: ReverseStringFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let marker = self.marker;
        crate::util::with_utf16_term(self.input.attributes_mut(), &mut self.buf, |b| {
            if marker != NOMARKER {
                b.push(marker);
            }
            let n = b.len();
            reverse(b, 0, n);
            true
        });
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected outputs are Lucene 10.5.0's (TestReverseStringFilter's cases).
    #[test]
    fn reverses_keeping_surrogate_pairs() {
        assert_eq!(reverse_string("A"), "A");
        assert_eq!(reverse_string("BA"), "AB");
        assert_eq!(reverse_string("CBA"), "ABC");
        assert_eq!(reverse_string("易经"), "经易");
        assert_eq!(reverse_string("😀"), "😀");
        assert_eq!(reverse_string("a😀b"), "b😀a");
        assert_eq!(reverse_string("😀ab"), "ba😀");
        assert_eq!(reverse_string("ab😀"), "😀ba");
        assert_eq!(reverse_string("😀a😁"), "😁a😀");
        assert_eq!(reverse_string("😀😁"), "😁😀");
        assert_eq!(reverse_string("😀😁a"), "a😁😀");
        assert_eq!(reverse_string(""), "");
    }
}
