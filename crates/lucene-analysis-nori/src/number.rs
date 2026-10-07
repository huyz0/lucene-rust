//! `org.apache.lucene.analysis.ko.KoreanNumberFilter`: joins a run of
//! number tokens (Arabic and Hangul numerals, `.`/`,` and their full-width
//! forms) into one token holding the number in Arabic digits (`삼백이십`
//! -> `320`, `１２，８００` -> `12800`, `3.2만` -> `32000`).
//!
//! The algorithm is `KoreanNumberFilter`'s with Hangul tables, as in
//! Java; `lucene-analysis-kuromoji/src/number.rs` is its twin.

use lucene_analysis::util::big_decimal::BigDecimal;
use lucene_analysis::{AnalysisError, State, TokenFilter, TokenStream};

/// `numerals[c]`: the value of a Hangul numeral.
fn hangul_numeral_value(c: u16) -> Option<u8> {
    Some(match c {
        0xC601 => 0, // 영
        0xC77C => 1, // 일
        0xC774 => 2, // 이
        0xC0BC => 3, // 삼
        0xC0AC => 4, // 사
        0xC624 => 5, // 오
        0xC721 => 6, // 육
        0xCE60 => 7, // 칠
        0xD314 => 8, // 팔
        0xAD6C => 9, // 구
        _ => return None,
    })
}

/// `exponents[c]`: the power of ten of a Hangul multiplier (`0`: none).
fn exponent(c: u16) -> u32 {
    match c {
        0xC2ED => 1,  // 십
        0xBC31 => 2,  // 백
        0xCC9C => 3,  // 천
        0xB9CC => 4,  // 만
        0xC5B5 => 8,  // 억
        0xC870 => 12, // 조
        0xACBD => 16, // 경
        0xD574 => 20, // 해
        _ => 0,
    }
}

fn is_half_width_arabic_numeral(c: u16) -> bool {
    (u16::from(b'0')..=u16::from(b'9')).contains(&c)
}

fn is_full_width_arabic_numeral(c: u16) -> bool {
    (0xFF10..=0xFF19).contains(&c)
}

/// `isArabicNumeral(c)`.
pub fn is_arabic_numeral(c: u16) -> bool {
    is_half_width_arabic_numeral(c) || is_full_width_arabic_numeral(c)
}

fn arabic_numeral_value(c: u16) -> u16 {
    if is_half_width_arabic_numeral(c) {
        c.wrapping_sub(u16::from(b'0'))
    } else {
        c.wrapping_sub(0xFF10)
    }
}

fn is_decimal_point(c: u16) -> bool {
    c == u16::from(b'.') || c == 0xFF0E
}

fn is_thousand_separator(c: u16) -> bool {
    c == u16::from(b',') || c == 0xFF0C
}

/// `isNumeral(char)`.
pub fn is_numeral_char(c: u16) -> bool {
    is_arabic_numeral(c) || hangul_numeral_value(c).is_some() || exponent(c) > 0
}

/// `isNumeral(String)`.
pub fn is_numeral(s: &[u16]) -> bool {
    s.iter().all(|&c| is_numeral_char(c))
}

/// `isNumeralPunctuation(String)`.
pub fn is_numeral_punctuation(s: &[u16]) -> bool {
    s.iter()
        .all(|&c| is_decimal_point(c) || is_thousand_separator(c))
}

/// `NumberBuffer`.
struct NumberBuffer<'a> {
    units: &'a [u16],
    position: usize,
}

impl NumberBuffer<'_> {
    fn current(&self) -> Option<u16> {
        self.units.get(self.position).copied()
    }
    fn advance(&mut self) {
        self.position = self.position.saturating_add(1);
    }
}

/// `parseBasicNumber`: `None` where Java returns `null`; `Err` where
/// `new BigDecimal` throws `NumberFormatException`.
fn parse_basic_number(buffer: &mut NumberBuffer<'_>) -> Result<Option<BigDecimal>, ()> {
    let mut builder = String::new();
    while let Some(c) = buffer.current() {
        if is_arabic_numeral(c) {
            builder.push(char::from(b'0'.wrapping_add(arabic_numeral_value(c) as u8)));
        } else if let Some(v) = hangul_numeral_value(c) {
            builder.push(char::from(b'0'.wrapping_add(v)));
        } else if is_decimal_point(c) {
            builder.push('.');
        } else if is_thousand_separator(c) {
            // Just skip and move to the next character
        } else {
            break;
        }
        buffer.advance();
    }
    if builder.is_empty() {
        return Ok(None);
    }
    BigDecimal::parse(&builder).map(Some).ok_or(())
}

/// `parseLargeHangulNumeral`.
fn parse_large_hangul_numeral(buffer: &mut NumberBuffer<'_>) -> Option<BigDecimal> {
    let power = exponent(buffer.current()?);
    if power > 3 {
        buffer.advance();
        return Some(BigDecimal::ten_pow(power));
    }
    None
}

/// `parseMediumHangulNumeral`.
fn parse_medium_hangul_numeral(buffer: &mut NumberBuffer<'_>) -> Option<BigDecimal> {
    let power = exponent(buffer.current()?);
    if (1..=3).contains(&power) {
        buffer.advance();
        return Some(BigDecimal::ten_pow(power));
    }
    None
}

/// `parseMediumPair`.
fn parse_medium_pair(buffer: &mut NumberBuffer<'_>) -> Result<Option<BigDecimal>, ()> {
    let first = parse_basic_number(buffer)?;
    let second = parse_medium_hangul_numeral(buffer);
    Ok(match (first, second) {
        (None, None) => None,
        (f, None) => f,
        (None, s) => s,
        (Some(f), Some(s)) => Some(f.multiply(&s)),
    })
}

/// `parseMediumNumber`.
fn parse_medium_number(buffer: &mut NumberBuffer<'_>) -> Result<Option<BigDecimal>, ()> {
    let Some(mut sum) = parse_medium_pair(buffer)? else {
        return Ok(None);
    };
    while let Some(r) = parse_medium_pair(buffer)? {
        sum = sum.add(&r);
    }
    Ok(Some(BigDecimal::zero().add(&sum)))
}

/// `parseLargePair`.
fn parse_large_pair(buffer: &mut NumberBuffer<'_>) -> Result<Option<BigDecimal>, ()> {
    let first = parse_medium_number(buffer)?;
    let second = parse_large_hangul_numeral(buffer);
    Ok(match (first, second) {
        (None, None) => None,
        (f, None) => f,
        (None, s) => s,
        (Some(f), Some(s)) => Some(f.multiply(&s)),
    })
}

/// `parseNumber`.
fn parse_number(buffer: &mut NumberBuffer<'_>) -> Result<Option<BigDecimal>, ()> {
    let Some(mut sum) = parse_large_pair(buffer)? else {
        return Ok(None);
    };
    while let Some(r) = parse_large_pair(buffer)? {
        sum = sum.add(&r);
    }
    Ok(Some(BigDecimal::zero().add(&sum)))
}

/// `normalizeNumber(number)`: the number in Arabic digits, or `number`
/// itself when it does not parse.
pub fn normalize_number(number: &str) -> String {
    let units: Vec<u16> = number.encode_utf16().collect();
    let mut buffer = NumberBuffer {
        units: &units,
        position: 0,
    };
    match parse_number(&mut buffer) {
        Ok(Some(n)) => n.strip_trailing_zeros().to_plain_string(),
        _ => number.to_string(),
    }
}

/// `KoreanNumberFilter`.
pub struct KoreanNumberFilter<I> {
    input: I,
    state: Option<State>,
    numeral: String,
    fall_through_tokens: i32,
    exhausted: bool,
}

impl<I: TokenStream> KoreanNumberFilter<I> {
    /// `new KoreanNumberFilter(input)`.
    pub fn new(input: I) -> Self {
        KoreanNumberFilter {
            input,
            state: None,
            numeral: String::new(),
            fall_through_tokens: 0,
            exhausted: false,
        }
    }
}

impl<I: TokenStream> TokenFilter for KoreanNumberFilter<I> {
    type Input = I;
    fn input(&self) -> &I {
        &self.input
    }
    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        // Emit previously captured token we read past earlier
        if let Some(state) = self.state.take() {
            self.input.attributes_mut().restore_state(&state);
            return Ok(true);
        }
        if self.exhausted {
            return Ok(false);
        }
        if !self.input.increment_token()? {
            self.exhausted = true;
            return Ok(false);
        }
        let atts = self.input.attributes();
        if atts.is_keyword() {
            return Ok(true);
        }
        if self.fall_through_tokens > 0 {
            self.fall_through_tokens = self.fall_through_tokens.wrapping_sub(1);
            return Ok(true);
        }
        if atts.position_increment() == 0 {
            self.fall_through_tokens = atts.position_length().wrapping_sub(1);
            return Ok(true);
        }

        let mut more_tokens = true;
        let mut composed_number_token = false;
        let mut start_offset = 0;
        let mut end_offset = 0;
        let pre_composition_state = atts.capture_state();
        let mut term = atts.term().to_string();
        let mut numeral_term = is_numeral(&term.encode_utf16().collect::<Vec<_>>());

        while more_tokens && numeral_term {
            let atts = self.input.attributes();
            if !composed_number_token {
                start_offset = atts.start_offset();
                composed_number_token = true;
            }
            end_offset = atts.end_offset();
            more_tokens = self.input.increment_token()?;
            if !more_tokens {
                self.exhausted = true;
            }
            let atts = self.input.attributes_mut();
            if atts.position_increment() == 0 {
                // This token is a stacked/synonym token, capture number of
                // tokens "under" this token, except the first token, which
                // we will emit below after restoring state
                self.fall_through_tokens = atts.position_length().wrapping_sub(1);
                self.state = Some(atts.capture_state());
                atts.restore_state(&pre_composition_state);
                return Ok(more_tokens);
            }
            self.numeral.push_str(&term);
            if more_tokens {
                term = atts.term().to_string();
                let units: Vec<u16> = term.encode_utf16().collect();
                numeral_term = is_numeral(&units) || is_numeral_punctuation(&units);
            }
        }

        if composed_number_token {
            let atts = self.input.attributes_mut();
            if more_tokens {
                // We have read past all numerals and there are still tokens
                // left, so capture the state of this token and emit it on
                // our next incrementToken()
                self.state = Some(atts.capture_state());
            }
            let normalized = normalize_number(&self.numeral);
            atts.set_term(&normalized);
            atts.set_offset(start_offset, end_offset)?;
            self.numeral.clear();
            return Ok(true);
        }
        Ok(more_tokens)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.fall_through_tokens = 0;
        self.numeral.clear();
        self.state = None;
        self.exhausted = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_like_java() {
        for (input, want) in [
            ("삼백이십", "320"),
            ("１２，８００", "12800"),
            ("3.2만", "32000"),
            ("일천만", "10000000"),
            ("이영이사", "2024"),
            ("십", "10"),
            ("만", "10000"),
            ("1.2.3", "1.2.3"),
            ("삼억오천만", "350000000"),
            ("x", "x"),
        ] {
            assert_eq!(normalize_number(input), want, "{input}");
        }
        assert!(is_numeral(&"백".encode_utf16().collect::<Vec<_>>()));
        assert!(is_numeral_punctuation(
            &",．".encode_utf16().collect::<Vec<_>>()
        ));
    }
}
