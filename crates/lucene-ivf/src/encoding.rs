//! The element type of the full-precision (rerank) vectors in `.ivv`.
//!
//! Rerank reads one vector per candidate, at random, so the bytes a vector
//! takes are what a query pays for on disk or through a blob cache. Every
//! encoding decodes to `f32`, and scores are computed on the decoded vector,
//! so a reranked score is always `similarity.score(query, vector(doc))`.
//!
//! | encoding | bytes per vector | element |
//! |---|---|---|
//! | [`VectorEncoding::F32`] | `4 * dim` | `f32` LE |
//! | [`VectorEncoding::F16`] | `2 * dim` | IEEE binary16 LE, rounded to nearest even |
//! | [`VectorEncoding::Bf16`] | `2 * dim` | the high half of the `f32`, rounded to nearest even |
//! | [`VectorEncoding::Int8`] | `dim + 8` | `f32` LE offset, `f32` LE scale, then `dim` `u8` codes of the residual |
//!
//! `Int8` quantizes each vector's residual `r = x - c` from its primary
//! cluster's centroid `c` (kept in `.ivm`), with a range of its own: the
//! offset is the residual's smallest component, the scale its range over
//! 255, and component `i` decodes as `c[i] + offset + scale * code[i]`
//! (error at most `scale / 2`). A residual spans a fraction of the vector's
//! own range, and a per-vector range is narrower than per-dimension ranges
//! over a whole field, which its outliers stretch, so both choices shrink
//! the step; neither needs training, and a vector decodes from its own
//! bytes and its centroid (a merge reads any mix of encodings).

use lucene_util::vector_util;

/// How the `.ivv` file stores each component of a full-precision vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorEncoding {
    /// 32-bit floats, exact.
    #[default]
    F32,
    /// IEEE half precision: 11 significant bits, magnitudes up to 65504.
    F16,
    /// bfloat16: 8 significant bits, the range of `f32`.
    Bf16,
    /// Per-vector scalar quantization to 8 bits (see the module docs).
    Int8,
}

/// Bytes of an `Int8` vector's offset and scale.
const INT8_HEADER: usize = 8;

impl VectorEncoding {
    /// Every encoding, by ordinal.
    pub const ALL: [VectorEncoding; 4] = [
        VectorEncoding::F32,
        VectorEncoding::F16,
        VectorEncoding::Bf16,
        VectorEncoding::Int8,
    ];

    /// The ordinal recorded in `.ivm`.
    pub fn ordinal(self) -> u8 {
        match self {
            VectorEncoding::F32 => 0,
            VectorEncoding::F16 => 1,
            VectorEncoding::Bf16 => 2,
            VectorEncoding::Int8 => 3,
        }
    }

    /// The encoding of a recorded ordinal.
    pub fn from_ordinal(o: u8) -> Option<Self> {
        Self::ALL.get(usize::from(o)).copied()
    }

    /// A short name: `f32`, `f16`, `bf16` or `int8`.
    pub fn name(self) -> &'static str {
        match self {
            VectorEncoding::F32 => "f32",
            VectorEncoding::F16 => "f16",
            VectorEncoding::Bf16 => "bf16",
            VectorEncoding::Int8 => "int8",
        }
    }

    /// The encoding named `name` (see [`VectorEncoding::name`]).
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|e| e.name() == name)
    }

    /// Bytes of one vector of `dimension` components, `None` on overflow.
    pub fn vector_bytes(self, dimension: usize) -> Option<usize> {
        match self {
            VectorEncoding::F32 => dimension.checked_mul(4),
            VectorEncoding::F16 | VectorEncoding::Bf16 => dimension.checked_mul(2),
            VectorEncoding::Int8 => dimension.checked_add(INT8_HEADER),
        }
    }

    /// Whether a vector is stored relative to its primary centroid.
    pub fn residual(self) -> bool {
        self == VectorEncoding::Int8
    }

    /// Encodes `v`, whose primary centroid is `centroid` (read only when
    /// [`VectorEncoding::residual`]), into `out`,
    /// [`VectorEncoding::vector_bytes`] long. Returns `false` when a finite
    /// component does not decode to a finite value (beyond `f16`'s range,
    /// or past `f32::MAX`).
    ///
    /// # Panics
    ///
    /// When `out` is not `vector_bytes(v.len())` long, or a residual
    /// encoding's `centroid` is not `v.len()` long.
    pub fn encode(self, v: &[f32], centroid: &[f32], out: &mut [u8]) -> bool {
        assert_eq!(Some(out.len()), self.vector_bytes(v.len()));
        match self {
            VectorEncoding::F32 => {
                for (o, x) in out.chunks_exact_mut(4).zip(v) {
                    o.copy_from_slice(&x.to_le_bytes());
                }
                v.iter().all(|x| x.is_finite())
            }
            VectorEncoding::F16 => {
                let mut finite = true;
                for (o, &x) in out.chunks_exact_mut(2).zip(v) {
                    let h = f32_to_f16(x);
                    finite &= h & 0x7c00 != 0x7c00;
                    o.copy_from_slice(&h.to_le_bytes());
                }
                finite
            }
            VectorEncoding::Bf16 => {
                let mut finite = true;
                for (o, &x) in out.chunks_exact_mut(2).zip(v) {
                    let h = f32_to_bf16(x);
                    finite &= h & 0x7f80 != 0x7f80;
                    o.copy_from_slice(&h.to_le_bytes());
                }
                finite
            }
            VectorEncoding::Int8 => {
                assert_eq!(centroid.len(), v.len());
                // The residual is recomputed where needed rather than
                // collected: every vector of a field is encoded.
                let residual = || v.iter().zip(centroid).map(|(x, c)| x - c);
                let finite = residual().all(f32::is_finite);
                // Adding the centroid back can still overflow.
                finite && encode_int8(residual, out) && {
                    let (head, codes) = out.split_at(INT8_HEADER);
                    let offset = f32::from_le_bytes([head[0], head[1], head[2], head[3]]);
                    let scale = f32::from_le_bytes([head[4], head[5], head[6], head[7]]);
                    codes
                        .iter()
                        .zip(centroid)
                        .all(|(&q, &c)| (c + (offset + scale * f32::from(q))).is_finite())
                }
            }
        }
    }

    /// Decodes one vector of `out.len()` components from `bytes`, adding
    /// `centroid` back for a residual encoding.
    ///
    /// # Panics
    ///
    /// When `bytes` is not `vector_bytes(out.len())` long, or a residual
    /// encoding's `centroid` is not `out.len()` long.
    pub fn decode(self, bytes: &[u8], centroid: &[f32], out: &mut [f32]) {
        assert_eq!(Some(bytes.len()), self.vector_bytes(out.len()));
        match self {
            VectorEncoding::F32 => match vector_util::le_f32s(bytes) {
                Some(v) => out.copy_from_slice(v),
                None => {
                    for (o, b) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                        *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    }
                }
            },
            VectorEncoding::F16 => {
                for (o, b) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                    *o = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                }
            }
            VectorEncoding::Bf16 => {
                for (o, b) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                    *o = bf16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                }
            }
            VectorEncoding::Int8 => {
                let (head, codes) = bytes.split_at(INT8_HEADER);
                let offset = f32::from_le_bytes([head[0], head[1], head[2], head[3]]);
                let scale = f32::from_le_bytes([head[4], head[5], head[6], head[7]]);
                assert_eq!(centroid.len(), out.len());
                for ((o, &q), &c) in out.iter_mut().zip(codes).zip(centroid) {
                    *o = c + (offset + scale * f32::from(q));
                }
            }
        }
    }
}

/// Per-vector min/max quantization of `v` (a residual): offset, scale,
/// then one code per component, rounded to the nearest level. `false` when
/// the top level does not decode to a finite value.
fn encode_int8<I: Iterator<Item = f32>>(v: impl Fn() -> I, out: &mut [u8]) -> bool {
    let (head, codes) = out.split_at_mut(INT8_HEADER);
    let min = v().fold(f32::INFINITY, f32::min);
    let max = v().fold(f32::NEG_INFINITY, f32::max);
    // In f64 so a range wider than `f32::MAX` does not overflow.
    let range = f64::from(max) - f64::from(min);
    #[allow(clippy::cast_possible_truncation)]
    let scale = if range > 0.0 {
        (range / 255.0) as f32
    } else {
        0.0
    };
    head[..4].copy_from_slice(&min.to_le_bytes());
    head[4..].copy_from_slice(&scale.to_le_bytes());
    if scale > 0.0 {
        let scale = f64::from(scale);
        for (c, x) in codes.iter_mut().zip(v()) {
            let level = ((f64::from(x) - f64::from(min)) / scale).round();
            // Clamped to a code: the cast cannot truncate.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let level = level.clamp(0.0, 255.0) as u8;
            *c = level;
        }
    } else {
        codes.fill(0);
    }
    // Decoded values are monotone in the code, so the top level bounds them.
    (min + scale * 255.0).is_finite()
}

/// `f32` to bfloat16, rounded to nearest, ties to even. Infinities stay
/// infinite; a `NaN` stays a `NaN`.
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        // Keep it a NaN: rounding could carry its mantissa into zero.
        return ((b >> 16) | 0x40) as u16;
    }
    // ARITH: `b + 0x8000` at most, carried into the exponent as rounding
    // up should; a non-NaN is at most `0xff80_0000`, so this cannot wrap.
    #[allow(clippy::arithmetic_side_effects)]
    let rounded = b + 0x7fff + ((b >> 16) & 1);
    (rounded >> 16) as u16
}

/// bfloat16 to `f32`, exact.
#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// `f32` to IEEE half precision, rounded to nearest, ties to even:
/// magnitudes from 65520 up become infinite, those below the smallest
/// subnormal half round to subnormals or zero. A `NaN` stays a `NaN`.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = (b >> 23) & 0xff;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        let nan = if man == 0 {
            0
        } else {
            0x200 | (man >> 13) as u16
        };
        return sign | 0x7c00 | nan;
    }
    // ARITH: `exp` is a byte, `man` 23 bits; the shifts are below 32 and
    // every sum stays within the 16 bits of a half.
    #[allow(clippy::arithmetic_side_effects)]
    {
        // The half's biased exponent: `exp - 127 + 15`.
        let e = i32::try_from(exp).unwrap_or(0) - 112;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        if e <= 0 {
            // Subnormal or zero: shift the significand (implicit bit
            // included) right by `13 + 1 - e`. Below `e = -10` even the
            // rounding bit is gone.
            if e < -10 {
                return sign;
            }
            let shift = u32::try_from(14 - e).unwrap_or(31);
            return sign | round_shift(man | 0x80_0000, shift) as u16;
        }
        // Normal: a carry out of the mantissa bumps the exponent, up to
        // infinity, as rounding should.
        let bits = (u32::try_from(e).unwrap_or(0) << 23) | man;
        sign | round_shift(bits, 13) as u16
    }
}

/// `v / 2^shift`, rounded to nearest, ties to even (`1 <= shift < 32`).
// ARITH: `shift` is in 1..32 and the quotient is below `v`, so adding one
// cannot overflow.
#[allow(clippy::arithmetic_side_effects)]
fn round_shift(v: u32, shift: u32) -> u32 {
    let q = v >> shift;
    let rem = v & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    if rem > half || (rem == half && q & 1 == 1) {
        q + 1
    } else {
        q
    }
}

/// IEEE half precision to `f32`, exact, subnormals, infinities and `NaN`s
/// included. Branch-free, so a loop over a vector vectorizes.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let h = u32::from(h);
    let sign = (h & 0x8000) << 16;
    let em = h & 0x7fff;
    // The half's exponent and mantissa as an `f32`'s are worth `2^-112`
    // of the half: scaling by `2^112` is exact for normals and subnormals.
    let finite = f32::from_bits(em << 13) * f32::from_bits(0x7780_0000);
    let special = f32::from_bits((em << 13) | 0x7f80_0000);
    let v = if em >= 0x7c00 { special } else { finite };
    f32::from_bits(v.to_bits() | sign)
}

#[cfg(test)]
mod tests;
