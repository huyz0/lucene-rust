//! `org.apache.lucene.analysis.payloads`: `DelimitedPayloadTokenFilter` and
//! its encoders, `NumericPayloadTokenFilter`, `TypeAsPayloadTokenFilter`,
//! `TokenOffsetPayloadTokenFilter`, `PayloadHelper`; and Java's
//! `Float.parseFloat`, which the float encoder and `DelimitedBoostTokenFilter`
//! share.

use crate::token_stream::{TokenFilter, TokenStream};
use crate::AnalysisError;

fn nfe(s: &str) -> AnalysisError {
    AnalysisError::IllegalArgument(format!("NumberFormatException: For input string: \"{s}\""))
}

/// `Float.parseFloat(String)`: Java's grammar (surrounding whitespace, a
/// sign, `NaN`/`Infinity`, digits with an optional point and exponent, an
/// optional `f`/`F`/`d`/`D` suffix), correctly rounded. Hexadecimal
/// literals are not supported (a `NumberFormatException`, which Java would
/// not throw).
pub fn parse_java_float(s: &str) -> Result<f32, AnalysisError> {
    let t = s.trim_matches(|c: char| c <= ' ');
    let (sign, body) = match t.strip_prefix('-') {
        Some(b) => ("-", b),
        None => ("", t.strip_prefix('+').unwrap_or(t)),
    };
    match body {
        "NaN" => return Ok(f32::NAN),
        "Infinity" => {
            return Ok(if sign == "-" {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            })
        }
        _ => {}
    }
    let body = body.strip_suffix(['f', 'F', 'd', 'D']).unwrap_or(body);
    let bytes = body.as_bytes();
    let mut i = 0;
    let digits = |i: &mut usize| {
        let s = *i;
        while *i < bytes.len() && bytes[*i].is_ascii_digit() {
            *i += 1;
        }
        *i - s
    };
    let int_digits = digits(&mut i);
    let mut frac_digits = 0;
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        frac_digits = digits(&mut i);
    }
    if int_digits + frac_digits == 0 {
        return Err(nfe(s));
    }
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        i += 1;
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        if digits(&mut i) == 0 {
            return Err(nfe(s));
        }
    }
    if i != bytes.len() {
        return Err(nfe(s));
    }
    format!("{sign}{body}").parse::<f32>().map_err(|_| nfe(s))
}

/// `PayloadHelper.encodeFloat`: big-endian IEEE bits.
pub fn encode_float(payload: f32) -> [u8; 4] {
    payload.to_bits().to_be_bytes()
}

/// `PayloadHelper.encodeInt`: big-endian.
pub fn encode_int(payload: i32) -> [u8; 4] {
    payload.to_be_bytes()
}

/// `PayloadHelper.decodeFloat`.
pub fn decode_float(bytes: &[u8]) -> Option<f32> {
    Some(f32::from_bits(u32::from_be_bytes(
        bytes.get(..4)?.try_into().ok()?,
    )))
}

/// `PayloadHelper.decodeInt`.
pub fn decode_int(bytes: &[u8]) -> Option<i32> {
    Some(i32::from_be_bytes(bytes.get(..4)?.try_into().ok()?))
}

/// `org.apache.lucene.analysis.payloads.PayloadEncoder`.
pub trait PayloadEncoder: Send + Sync {
    /// `encode(char[], int, int)`: the payload for the text after the
    /// delimiter.
    fn encode(&self, text: &[u16]) -> Result<Vec<u8>, AnalysisError>;
}

/// `org.apache.lucene.analysis.payloads.FloatEncoder`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FloatEncoder;

impl PayloadEncoder for FloatEncoder {
    fn encode(&self, text: &[u16]) -> Result<Vec<u8>, AnalysisError> {
        Ok(encode_float(parse_java_float(&String::from_utf16_lossy(text))?).to_vec())
    }
}

/// `org.apache.lucene.analysis.payloads.IntegerEncoder` (`ArrayUtil.parseInt`).
#[derive(Debug, Clone, Copy, Default)]
pub struct IntegerEncoder;

impl PayloadEncoder for IntegerEncoder {
    fn encode(&self, text: &[u16]) -> Result<Vec<u8>, AnalysisError> {
        Ok(encode_int(crate::miscellaneous::parse_int(text)?).to_vec())
    }
}

/// `org.apache.lucene.analysis.payloads.IdentityEncoder` with its default
/// UTF-8 charset (an unpaired surrogate encodes as `?`, as Java's encoder
/// replaces it).
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityEncoder;

impl PayloadEncoder for IdentityEncoder {
    fn encode(&self, text: &[u16]) -> Result<Vec<u8>, AnalysisError> {
        let mut out = Vec::with_capacity(text.len());
        for r in char::decode_utf16(text.iter().copied()) {
            let c = r.unwrap_or('?');
            let mut b = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut b).as_bytes());
        }
        Ok(out)
    }
}

/// `DelimitedPayloadTokenFilter.DEFAULT_DELIMITER`.
pub const DEFAULT_DELIMITER: char = '|';

/// `org.apache.lucene.analysis.payloads.DelimitedPayloadTokenFilter`:
/// `term|payload` -> `term` with the encoded payload.
pub struct DelimitedPayloadTokenFilter<I, E> {
    input: I,
    delimiter: u16,
    encoder: E,
    buf: Vec<u16>,
}

impl<I: TokenStream, E: PayloadEncoder> DelimitedPayloadTokenFilter<I, E> {
    /// `new DelimitedPayloadTokenFilter(TokenStream, char delimiter, PayloadEncoder)`.
    pub fn new(input: I, delimiter: u16, encoder: E) -> Self {
        DelimitedPayloadTokenFilter {
            input,
            delimiter,
            encoder,
            buf: Vec::new(),
        }
    }
}

impl<I: TokenStream, E: PayloadEncoder> TokenFilter for DelimitedPayloadTokenFilter<I, E> {
    crate::filter_input!();
    // Java: DelimitedPayloadTokenFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        self.buf.clear();
        self.buf.extend(a.term().encode_utf16());
        match self.buf.iter().position(|&u| u == self.delimiter) {
            Some(i) => {
                let payload = self.encoder.encode(&self.buf[i + 1..])?;
                a.set_payload(Some(payload));
                a.set_term_utf16(&self.buf[..i]);
            }
            None => a.set_payload(None),
        }
        Ok(true)
    }
}

/// `org.apache.lucene.analysis.payloads.NumericPayloadTokenFilter`: a fixed
/// float payload on every token of one type.
pub struct NumericPayloadTokenFilter<I> {
    input: I,
    type_match: String,
    payload: [u8; 4],
}

impl<I: TokenStream> NumericPayloadTokenFilter<I> {
    /// `new NumericPayloadTokenFilter(TokenStream, float payload, String typeMatch)`.
    pub fn new(input: I, payload: f32, type_match: &str) -> Self {
        NumericPayloadTokenFilter {
            input,
            type_match: type_match.to_string(),
            payload: encode_float(payload),
        }
    }
}

impl<I: TokenStream> TokenFilter for NumericPayloadTokenFilter<I> {
    crate::filter_input!();
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        if a.token_type() == self.type_match {
            a.set_payload(Some(self.payload.to_vec()));
        }
        Ok(true)
    }
}

/// `org.apache.lucene.analysis.payloads.TypeAsPayloadTokenFilter`: the type's
/// UTF-8 bytes as the payload.
pub struct TypeAsPayloadTokenFilter<I> {
    input: I,
}

impl<I: TokenStream> TypeAsPayloadTokenFilter<I> {
    /// `new TypeAsPayloadTokenFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        TypeAsPayloadTokenFilter { input }
    }
}

impl<I: TokenStream> TokenFilter for TypeAsPayloadTokenFilter<I> {
    crate::filter_input!();
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        if !a.token_type().is_empty() {
            let p = a.token_type().as_bytes().to_vec();
            a.set_payload(Some(p));
        }
        Ok(true)
    }
}

/// `org.apache.lucene.analysis.payloads.TokenOffsetPayloadTokenFilter`: the
/// start and end offsets, big-endian, as the payload.
pub struct TokenOffsetPayloadTokenFilter<I> {
    input: I,
}

impl<I: TokenStream> TokenOffsetPayloadTokenFilter<I> {
    /// `new TokenOffsetPayloadTokenFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        TokenOffsetPayloadTokenFilter { input }
    }
}

impl<I: TokenStream> TokenFilter for TokenOffsetPayloadTokenFilter<I> {
    crate::filter_input!();
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        let mut data = encode_int(a.start_offset()).to_vec();
        data.extend_from_slice(&encode_int(a.end_offset()));
        a.set_payload(Some(data));
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::Canned;

    fn payloads(ts: &mut dyn TokenStream) -> Vec<(String, Option<Vec<u8>>)> {
        let mut out = Vec::new();
        crate::token_stream::consume(ts, |a| {
            out.push((a.term().to_string(), a.payload().map(<[u8]>::to_vec)))
        })
        .unwrap();
        out
    }

    #[test]
    fn java_float_grammar() {
        for (s, v) in [
            ("1.5", 1.5f32),
            (" -2e3 ", -2000.0),
            ("3f", 3.0),
            (".5", 0.5),
            ("5.", 5.0),
            ("+1D", 1.0),
            ("-Infinity", f32::NEG_INFINITY),
            ("1E-2", 0.01),
        ] {
            assert_eq!(parse_java_float(s).unwrap(), v, "{s}");
        }
        assert!(parse_java_float("NaN").unwrap().is_nan());
        for bad in ["", ".", "inf", "nan", "1e", "1x", "0x1p3", "e5", "1.5ff"] {
            assert!(parse_java_float(bad).is_err(), "{bad}");
        }
        assert_eq!(decode_float(&encode_float(2.5)), Some(2.5));
        assert_eq!(decode_int(&encode_int(-7)), Some(-7));
        assert_eq!(decode_int(&[1]), None);
    }

    #[test]
    fn delimited_and_fixed_payloads() {
        let mut c = Canned::parse("a:0:1:1:1 b:0:1:1:1 c:0:1:1:1");
        c.set_terms(&["x|2.5", "y", "z|ab"]);
        let mut f = DelimitedPayloadTokenFilter::new(c, u16::from(b'|'), IdentityEncoder);
        assert_eq!(payloads(&mut f)[2], ("z".into(), Some(b"ab".to_vec())));
        let mut c = Canned::parse("a:0:1:1:1 b:0:1:1:1");
        c.set_terms(&["x|2.5", "y"]);
        let mut f = DelimitedPayloadTokenFilter::new(c, u16::from(b'|'), FloatEncoder);
        assert_eq!(
            payloads(&mut f),
            vec![
                ("x".into(), Some(vec![0x40, 0x20, 0, 0])),
                ("y".into(), None)
            ]
        );
        let mut c = Canned::parse("a:0:1:1:1");
        c.set_terms(&["x|7"]);
        let mut f = DelimitedPayloadTokenFilter::new(c, u16::from(b'|'), IntegerEncoder);
        assert_eq!(payloads(&mut f), vec![("x".into(), Some(vec![0, 0, 0, 7]))]);
        assert_eq!(IdentityEncoder.encode(&[0xD800, 0x61]).unwrap(), b"?a");
        let mut c = Canned::parse("a:0:1:1:1 2:2:3:1:1");
        c.set_types(&["<ALPHANUM>", "<NUM>"]);
        let mut f = NumericPayloadTokenFilter::new(c, 3.5, "<NUM>");
        assert_eq!(payloads(&mut f)[1].1, Some(encode_float(3.5).to_vec()));
        let mut c = Canned::parse("a:0:1:1:1 b:5:9:1:1");
        c.set_types(&["", "w"]);
        let mut f = TypeAsPayloadTokenFilter::new(c);
        assert_eq!(
            payloads(&mut f)
                .iter()
                .map(|p| p.1.clone())
                .collect::<Vec<_>>(),
            vec![None, Some(b"w".to_vec())]
        );
        let mut f = TokenOffsetPayloadTokenFilter::new(Canned::parse("b:5:9:1:1"));
        assert_eq!(payloads(&mut f)[0].1, Some(vec![0, 0, 0, 5, 0, 0, 0, 9]));
    }
}
