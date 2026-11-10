//! The bytes of one multi-vector, as its `BINARY` doc value holds them.
//!
//! | Encoding | Bytes |
//! | --- | --- |
//! | [`TokenEncoding::Float32`] | `int dim`, then every token's `dim` floats; all little-endian (Lucene's `LateInteractionField`) |
//! | [`TokenEncoding::Float16`] | `int -1`, `int dim`, `int tokens`, then the tokens in blocks of 16, component-major, IEEE half floats |
//! | [`TokenEncoding::Int8`] | `int -2`, `int dim`, `int tokens`, a `float` scale per token, then blocks of 16 as above, signed bytes |
//!
//! A block holds tokens `16 b .. 16 b + 16`: its entry `k * 16 + l` is
//! component `k` of token `16 b + l`, and lanes past the last token are zero.
//! That is the layout [`lucene_util::max_sim`] scores, so a value decodes
//! with one pass and no transpose. Lucene's format starts with a positive
//! dimension, so a negative first `int` marks this crate's encodings.
//!
//! Int8 stores each token as `round(x / scale)` with `scale = max|x| / 127`,
//! which keeps a token's direction to within 1/254 of its largest component.
//! Both compact encodings take finite components only, float16 those up to
//! 65504 in magnitude; `float` stores any float, as Lucene does.

use lucene_codecs::vectors::f32_to_f16;
use lucene_util::max_sim::{blocked_len, f16_to_f32, transpose_into_blocks, Blocks, LANES};

use crate::{illegal, Result};

const FLOAT16_MARKER: i32 = -1;
const INT8_MARKER: i32 = -2;
const HEADER: usize = 12;
/// The largest finite half float.
const HALF_MAX: f32 = 65504.0;

/// How a multi-vector's token components are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenEncoding {
    /// 4 bytes per component, Lucene's `LateInteractionField` layout.
    Float32,
    /// 2 bytes per component; lossless for ColBERT-style unit vectors in
    /// practice (measured: same nDCG@10 as `Float32`).
    Float16,
    /// 1 byte per component plus a 4-byte scale per token.
    Int8,
}

impl TokenEncoding {
    /// The name a mapping uses: `float`, `float16` or `int8`.
    pub fn name(self) -> &'static str {
        match self {
            TokenEncoding::Float32 => "float",
            TokenEncoding::Float16 => "float16",
            TokenEncoding::Int8 => "int8",
        }
    }

    /// The encoding called `name` ([`Self::name`]).
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::Float32, Self::Float16, Self::Int8]
            .into_iter()
            .find(|e| e.name() == name)
    }

    /// The encoded size of `tokens` tokens of `dim` components, or `None`
    /// when it does not fit a `usize`.
    pub fn encoded_len(self, tokens: usize, dim: usize) -> Option<usize> {
        match self {
            TokenEncoding::Float32 => tokens.checked_mul(dim)?.checked_mul(4)?.checked_add(4),
            TokenEncoding::Float16 => blocked_len(tokens, dim)?
                .checked_mul(2)?
                .checked_add(HEADER),
            TokenEncoding::Int8 => blocked_len(tokens, dim)?
                .checked_add(tokens.checked_mul(4)?)?
                .checked_add(HEADER),
        }
    }
}

fn int(v: usize, what: &str) -> Result<[u8; 4]> {
    i32::try_from(v)
        .map(i32::to_le_bytes)
        .map_err(|_| illegal(format!("{what} {v} does not fit an int")))
}

/// Encodes `tokens` (token-major, `dim` components each) as `encoding`.
///
/// # Errors
/// [`crate::Error::IllegalArgument`] for no tokens, a zero `dim`, or data
/// that is not whole tokens.
pub fn encode(tokens: &[f32], dim: usize, encoding: TokenEncoding) -> Result<Vec<u8>> {
    if dim == 0 {
        return Err(illegal(
            "Composing token vectors should not be null or empty",
        ));
    }
    if tokens.is_empty() {
        return Err(illegal("Value should not be null or empty"));
    }
    if !tokens.len().is_multiple_of(dim) {
        return Err(illegal(format!(
            "{} components are not whole tokens of dimension {dim}",
            tokens.len()
        )));
    }
    // ARITH: dim > 0 (checked above).
    #[allow(clippy::arithmetic_side_effects)]
    let n = tokens.len() / dim;
    let mut out = Vec::with_capacity(encoding.encoded_len(n, dim).unwrap_or(0));
    if encoding == TokenEncoding::Float32 {
        out.extend_from_slice(&int(dim, "dimension")?);
        for x in tokens {
            out.extend_from_slice(&x.to_le_bytes());
        }
        return Ok(out);
    }
    if let Some(x) = tokens.iter().find(|x| !x.is_finite()) {
        return Err(illegal(format!(
            "{} stores finite components only, got {x}",
            encoding.name()
        )));
    }
    if encoding == TokenEncoding::Float16 {
        if let Some(x) = tokens.iter().find(|x| x.abs() > HALF_MAX) {
            return Err(illegal(format!(
                "float16 stores components up to {HALF_MAX} in magnitude, got {x}"
            )));
        }
    }
    let marker = if encoding == TokenEncoding::Float16 {
        FLOAT16_MARKER
    } else {
        INT8_MARKER
    };
    out.extend_from_slice(&marker.to_le_bytes());
    out.extend_from_slice(&int(dim, "dimension")?);
    out.extend_from_slice(&int(n, "token count")?);
    let mut blocks = Vec::new();
    if encoding == TokenEncoding::Float16 {
        transpose_into_blocks(tokens, dim, &mut blocks);
        for x in blocks {
            out.extend_from_slice(&f32_to_f16(x).to_le_bytes());
        }
        return Ok(out);
    }
    let mut scaled = Vec::with_capacity(tokens.len());
    for t in tokens.chunks_exact(dim) {
        let max = t.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        // A subnormal `max` would underflow to a zero scale: the smallest
        // positive float keeps every scaled component within the clamp.
        let scale = if max > 0.0 {
            (max / 127.0).max(f32::from_bits(1))
        } else {
            1.0
        };
        out.extend_from_slice(&scale.to_le_bytes());
        scaled.extend(t.iter().map(|&x| (x / scale).round().clamp(-127.0, 127.0)));
    }
    transpose_into_blocks(&scaled, dim, &mut blocks);
    out.extend(blocks.into_iter().map(|x| (x as i8) as u8));
    Ok(out)
}

fn read_int(b: &[u8], at: usize) -> Option<i32> {
    b.get(at..at.checked_add(4)?)
        .map(|s| i32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn not_lucene() -> crate::Error {
    illegal("Provided payload does not appear to have been encoded via LateInteractionField.encode")
}

fn bad() -> crate::Error {
    illegal("Provided payload is not a multi-vector this field encodes")
}

/// Decodes `value` into `out` in the block layout `lucene_util::max_sim`
/// scores, as `f32`s, replacing what `out` held, and returns `(dim, tokens)`.
///
/// # Errors
/// [`crate::Error::IllegalArgument`] when `value` is not a whole multi-vector
/// of any [`TokenEncoding`].
pub fn decode_into_blocks(value: &[u8], out: &mut Vec<f32>) -> Result<(usize, usize)> {
    let mut scratch = Vec::new();
    let (blocks, dim, n) = blocks_of(value, &mut scratch)?;
    match blocks {
        Blocks::F32(b) => {
            out.clear();
            out.extend_from_slice(b);
        }
        Blocks::F16(b) => {
            out.clear();
            out.extend(
                b.chunks_exact(2)
                    .map(|h| f16_to_f32(u16::from_le_bytes([h[0], h[1]]))),
            );
        }
    }
    Ok((dim, n))
}

/// `value`'s tokens in the block layout, ready to score, with `(dim,
/// tokens)`: a `float16` value is scored in place, the others are widened
/// into `scratch` (Lucene's layout is transposed as it is read).
///
/// # Errors
/// As [`decode_into_blocks`].
pub fn blocks_of<'v>(
    value: &'v [u8],
    scratch: &'v mut Vec<f32>,
) -> Result<(Blocks<'v>, usize, usize)> {
    let first = read_int(value, 0).ok_or_else(bad)?;
    if first > 0 {
        let dim = usize::try_from(first).map_err(|_| not_lucene())?;
        let body = &value[4..];
        let width = dim.checked_mul(4).ok_or_else(not_lucene)?;
        if !body.len().is_multiple_of(width) {
            return Err(not_lucene());
        }
        // ARITH: width = 4 dim > 0.
        #[allow(clippy::arithmetic_side_effects)]
        let n = body.len() / width;
        let len = blocked_len(n, dim).ok_or_else(not_lucene)?;
        scratch.clear();
        scratch.resize(len, 0.0);
        // ARITH: token i < n and component k < dim land at
        // (i / 16) dim 16 + k 16 + i % 16 < len.
        #[allow(clippy::arithmetic_side_effects)]
        for (i, token) in body.chunks_exact(width).enumerate() {
            let base = (i / LANES) * dim * LANES + i % LANES;
            for (k, f) in token.chunks_exact(4).enumerate() {
                scratch[base + k * LANES] = f32::from_le_bytes([f[0], f[1], f[2], f[3]]);
            }
        }
        return Ok((Blocks::F32(scratch), dim, n));
    }
    let dim = usize::try_from(read_int(value, 4).ok_or_else(bad)?).map_err(|_| bad())?;
    let n = usize::try_from(read_int(value, 8).ok_or_else(bad)?).map_err(|_| bad())?;
    if dim == 0 {
        return Err(bad());
    }
    let len = blocked_len(n, dim).ok_or_else(bad)?;
    match first {
        FLOAT16_MARKER => {
            let body = &value[HEADER..];
            if Some(body.len()) != len.checked_mul(2) {
                return Err(bad());
            }
            Ok((Blocks::F16(body), dim, n))
        }
        INT8_MARKER => {
            let end = n
                .checked_mul(4)
                .and_then(|s| s.checked_add(HEADER))
                .ok_or_else(bad)?;
            let scales = value.get(HEADER..end).ok_or_else(bad)?;
            let body = &value[end..];
            if body.len() != len {
                return Err(bad());
            }
            scratch.clear();
            scratch.resize(len, 0.0);
            // ARITH: `body.len() == len` = blocks * dim * 16 (checked above), so
            // `dim * LANES` fits, and block `b` holds tokens below
            // `min(16 (b + 1), n)`, each with a scale at 4 * token.
            #[allow(clippy::arithmetic_side_effects)]
            for (b, (src, dst)) in body
                .chunks_exact(dim * LANES)
                .zip(scratch.chunks_exact_mut(dim * LANES))
                .enumerate()
            {
                let mut lane_scale = [0f32; LANES];
                for (l, s) in lane_scale.iter_mut().enumerate().take(n - b * LANES) {
                    let at = 4 * (b * LANES + l);
                    *s = f32::from_le_bytes([
                        scales[at],
                        scales[at + 1],
                        scales[at + 2],
                        scales[at + 3],
                    ]);
                }
                for (row_src, row_dst) in src.chunks_exact(LANES).zip(dst.chunks_exact_mut(LANES)) {
                    for l in 0..LANES {
                        row_dst[l] = f32::from(row_src[l] as i8) * lane_scale[l];
                    }
                }
            }
            Ok((Blocks::F32(scratch), dim, n))
        }
        _ => Err(bad()),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;

    fn tokens(n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 64.0)
            .collect()
    }

    #[test]
    fn float32_is_lucenes_late_interaction_layout() {
        let t = tokens(3, 4);
        let got = encode(&t, 4, TokenEncoding::Float32).unwrap();
        let rows: Vec<Vec<f32>> = t.chunks(4).map(<[f32]>::to_vec).collect();
        let want = lucene_search::values_source::encode_multi_vector(&rows).unwrap();
        assert_eq!(got, want);
        assert_eq!(Some(got.len()), TokenEncoding::Float32.encoded_len(3, 4));
    }

    #[test]
    fn every_encoding_decodes_to_the_blocks_of_its_tokens() {
        for (n, dim) in [(1, 1), (17, 5), (40, 96)] {
            let t = tokens(n, dim);
            let mut want = Vec::new();
            transpose_into_blocks(&t, dim, &mut want);
            for (enc, tol) in [
                (TokenEncoding::Float32, 0.0f32),
                (TokenEncoding::Float16, 1e-3),
                (TokenEncoding::Int8, 1.0 / 127.0),
            ] {
                let bytes = encode(&t, dim, enc).unwrap();
                assert_eq!(Some(bytes.len()), enc.encoded_len(n, dim), "{enc:?}");
                let mut got = Vec::new();
                assert_eq!(decode_into_blocks(&bytes, &mut got).unwrap(), (dim, n));
                assert_eq!(got.len(), want.len());
                for (g, w) in got.iter().zip(&want) {
                    assert!((g - w).abs() <= tol * w.abs().max(1.0), "{enc:?} {g} {w}");
                }
            }
        }
    }

    #[test]
    fn bad_values_are_refused() {
        let mut out = Vec::new();
        for v in [
            &[][..],
            &[1, 0, 0][..],
            &[2, 0, 0, 0, 1, 2, 3, 4][..],
            &[0xfd, 0xff, 0xff, 0xff, 1, 0, 0, 0, 1, 0, 0, 0][..],
        ] {
            assert!(decode_into_blocks(v, &mut out).is_err(), "{v:?}");
        }
        let mut short = encode(&tokens(2, 3), 3, TokenEncoding::Float16).unwrap();
        short.pop();
        assert!(decode_into_blocks(&short, &mut out).is_err());
        assert!(encode(&[], 3, TokenEncoding::Int8).is_err());
        assert!(encode(&[1.0; 4], 3, TokenEncoding::Int8).is_err());
        assert!(encode(&[1.0; 4], 0, TokenEncoding::Int8).is_err());
    }

    #[test]
    fn a_zero_token_keeps_a_unit_scale_and_bad_headers_are_refused() {
        let bytes = encode(&[0.0, 0.0, -1.0, 0.0], 2, TokenEncoding::Int8).unwrap();
        let mut out = Vec::new();
        assert_eq!(decode_into_blocks(&bytes, &mut out).unwrap(), (2, 2));
        assert_eq!((out[0], out[16], out[1], out[17]), (0.0, 0.0, -1.0, 0.0));
        // scales cut short, an unknown marker, a zero dimension
        let mut cut = bytes.clone();
        cut.truncate(15);
        assert!(decode_into_blocks(&cut, &mut out).is_err());
        let mut unknown = bytes.clone();
        unknown[0] = 0xfd;
        assert!(decode_into_blocks(&unknown, &mut out).is_err());
        let mut zero_dim = bytes;
        zero_dim[4..8].copy_from_slice(&0i32.to_le_bytes());
        assert!(decode_into_blocks(&zero_dim, &mut out).is_err());
        // a token count whose blocks overflow
        let mut huge = encode(&[1.0], 1, TokenEncoding::Float16).unwrap();
        huge[4..8].copy_from_slice(&i32::MAX.to_le_bytes());
        huge[8..12].copy_from_slice(&i32::MAX.to_le_bytes());
        assert!(decode_into_blocks(&huge, &mut out).is_err());
    }

    #[test]
    fn compact_encodings_take_finite_components_in_range() {
        for enc in [TokenEncoding::Float16, TokenEncoding::Int8] {
            for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let err = encode(&[1.0, x], 2, enc).unwrap_err();
                assert!(err.to_string().contains("finite"), "{enc:?} {err}");
            }
        }
        assert!(encode(&[65504.0, -65504.0], 2, TokenEncoding::Float16).is_ok());
        let err = encode(&[1.0, 65505.0], 2, TokenEncoding::Float16).unwrap_err();
        assert!(err.to_string().contains("65504"), "{err}");
        // `float` keeps what Lucene keeps
        assert!(encode(&[f32::NAN, f32::INFINITY], 2, TokenEncoding::Float32).is_ok());
        // a subnormal token keeps a non-zero scale and its direction
        let tiny = f32::from_bits(3);
        let bytes = encode(&[tiny, -tiny], 2, TokenEncoding::Int8).unwrap();
        let mut out = Vec::new();
        decode_into_blocks(&bytes, &mut out).unwrap();
        assert!(out[0] > 0.0 && out[16] < 0.0, "{} {}", out[0], out[16]);
    }

    #[test]
    fn a_malformed_lucene_value_reports_lucenes_message() {
        let mut out = Vec::new();
        let err = decode_into_blocks(&[2, 0, 0, 0, 1, 2, 3, 4], &mut out).unwrap_err();
        assert!(
            err.to_string().contains("LateInteractionField.encode"),
            "{err}"
        );
    }

    #[test]
    fn names_round_trip() {
        for e in [
            TokenEncoding::Float32,
            TokenEncoding::Float16,
            TokenEncoding::Int8,
        ] {
            assert_eq!(TokenEncoding::from_name(e.name()), Some(e));
        }
        assert_eq!(TokenEncoding::from_name("bit"), None);
    }
}
