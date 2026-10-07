//! The part of `java.math.BigDecimal` Kuromoji's `JapaneseNumberFilter`
//! and Nori's `KoreanNumberFilter` use: `new BigDecimal(String)` of plain
//! digits with at most one decimal point, `add`, `multiply`,
//! `TEN.pow(n)`, `stripTrailingZeros()` and `toPlainString()`.
//!
//! A value is an unscaled non-negative integer (decimal digits, most
//! significant first) and a scale, as in Java; the filters never make a
//! negative number or an exponent, so neither is supported.

use std::cmp::Ordering;

/// A non-negative `BigDecimal`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BigDecimal {
    /// Unscaled value, most significant digit first, no leading zeros
    /// (zero is `[0]`).
    digits: Vec<u8>,
    scale: i64,
}

fn trim(mut d: Vec<u8>) -> Vec<u8> {
    let nz = d
        .iter()
        .position(|&x| x != 0)
        .unwrap_or(d.len().saturating_sub(1));
    d.drain(..nz);
    if d.is_empty() {
        d.push(0);
    }
    d
}

impl BigDecimal {
    /// `BigDecimal.ZERO`.
    pub fn zero() -> Self {
        BigDecimal {
            digits: vec![0],
            scale: 0,
        }
    }

    /// `BigDecimal.TEN.pow(n)`: unscaled `10^n`, scale 0.
    pub fn ten_pow(n: u32) -> Self {
        let mut digits = vec![1u8];
        // ALLOC: the number filters' exponents go up to 20.
        digits.resize(usize::try_from(n).unwrap_or(0).saturating_add(1), 0);
        BigDecimal { digits, scale: 0 }
    }

    /// `new BigDecimal(String)` for ASCII digits and at most one `.`;
    /// `None` where Java throws `NumberFormatException` (no digit, a second
    /// point, another character).
    pub fn parse(s: &str) -> Option<Self> {
        let mut digits = Vec::with_capacity(s.len());
        let mut scale: i64 = 0;
        let mut seen_point = false;
        for b in s.bytes() {
            match b {
                b'0'..=b'9' => {
                    digits.push(b - b'0');
                    if seen_point {
                        scale = scale.saturating_add(1);
                    }
                }
                b'.' if !seen_point => seen_point = true,
                _ => return None,
            }
        }
        if digits.is_empty() {
            return None;
        }
        Some(BigDecimal {
            digits: trim(digits),
            scale,
        })
    }

    /// The unscaled digits extended to `scale` (which must not be below
    /// this value's).
    fn rescaled(&self, scale: i64) -> Vec<u8> {
        let extra = usize::try_from(scale.saturating_sub(self.scale)).unwrap_or(0);
        let mut d = self.digits.clone();
        // ALLOC: a scale is the count of digits after a parsed point, so
        // the difference is bounded by the parsed text's length.
        d.resize(d.len().saturating_add(extra), 0);
        d
    }

    /// `add(other)`: the scale is the larger one.
    pub fn add(&self, other: &BigDecimal) -> BigDecimal {
        let scale = self.scale.max(other.scale);
        let (a, b) = (self.rescaled(scale), other.rescaled(scale));
        let n = a.len().max(b.len());
        let mut out = Vec::with_capacity(n.saturating_add(1));
        let mut carry = 0u8;
        let digit = |v: &[u8], i: usize| -> u8 {
            if i < v.len() {
                v[v.len() - 1 - i]
            } else {
                0
            }
        };
        for i in 0..n {
            let s = digit(&a, i) + digit(&b, i) + carry;
            out.push(s % 10);
            carry = s / 10;
        }
        if carry > 0 {
            out.push(carry);
        }
        out.reverse();
        BigDecimal {
            digits: trim(out),
            scale,
        }
    }

    /// `multiply(other)`: the scales add.
    pub fn multiply(&self, other: &BigDecimal) -> BigDecimal {
        let (a, b) = (&self.digits, &other.digits);
        let mut acc = vec![0u32; a.len() + b.len()];
        for (i, &x) in a.iter().rev().enumerate() {
            for (j, &y) in b.iter().rev().enumerate() {
                acc[i + j] += u32::from(x) * u32::from(y);
            }
        }
        let mut carry = 0u32;
        let mut out = Vec::with_capacity(acc.len());
        for v in acc {
            let s = v + carry;
            out.push((s % 10) as u8);
            carry = s / 10;
        }
        while carry > 0 {
            out.push((carry % 10) as u8);
            carry /= 10;
        }
        out.reverse();
        BigDecimal {
            digits: trim(out),
            scale: self.scale.saturating_add(other.scale),
        }
    }

    /// `stripTrailingZeros()`: zero is `BigDecimal.ZERO`.
    pub fn strip_trailing_zeros(&self) -> BigDecimal {
        if self.digits == [0] {
            return BigDecimal::zero();
        }
        let zeros = self.digits.iter().rev().take_while(|&&d| d == 0).count();
        BigDecimal {
            digits: self.digits[..self.digits.len() - zeros].to_vec(),
            scale: self.scale.saturating_sub(i64::try_from(zeros).unwrap_or(0)),
        }
    }

    /// `toPlainString()`.
    pub fn to_plain_string(&self) -> String {
        let digits: String = self.digits.iter().map(|d| char::from(b'0' + d)).collect();
        match self.scale.cmp(&0) {
            Ordering::Equal => digits,
            Ordering::Less => {
                let zeros = usize::try_from(self.scale.unsigned_abs()).unwrap_or(0);
                if self.digits == [0] {
                    return digits;
                }
                digits + &"0".repeat(zeros)
            }
            Ordering::Greater => {
                let scale = usize::try_from(self.scale).unwrap_or(usize::MAX);
                if digits.len() > scale {
                    let (int, frac) = digits.split_at(digits.len() - scale);
                    format!("{int}.{frac}")
                } else {
                    format!("0.{}{digits}", "0".repeat(scale - digits.len()))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(s: &str) -> BigDecimal {
        BigDecimal::parse(s).unwrap()
    }

    #[test]
    fn plain_strings_like_java() {
        assert_eq!(n("12800").strip_trailing_zeros().to_plain_string(), "12800");
        assert_eq!(n("1.50").strip_trailing_zeros().to_plain_string(), "1.5");
        assert_eq!(n(".5").to_plain_string(), "0.5");
        assert_eq!(n("1.").to_plain_string(), "1");
        assert_eq!(n("007").to_plain_string(), "7");
        assert_eq!(n("0.000").strip_trailing_zeros().to_plain_string(), "0");
        assert_eq!(n("0.0012").to_plain_string(), "0.0012");
        for bad in ["", ".", "1.2.3", "a", "1,0"] {
            assert!(BigDecimal::parse(bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn arithmetic_like_java() {
        let x = n("3").multiply(&BigDecimal::ten_pow(4)).add(&n("5.25"));
        assert_eq!(x.to_plain_string(), "30005.25");
        let big = n("999").multiply(&BigDecimal::ten_pow(20));
        assert_eq!(big.to_plain_string(), format!("999{}", "0".repeat(20)));
        assert_eq!(n("1.5").multiply(&n("1.5")).to_plain_string(), "2.25");
        assert_eq!(n("99").add(&n("1")).to_plain_string(), "100");
        assert_eq!(BigDecimal::zero().add(&n("0.10")).to_plain_string(), "0.10");
        let s = n("100")
            .multiply(&BigDecimal::ten_pow(3))
            .strip_trailing_zeros();
        assert_eq!(s.to_plain_string(), "100000");
        assert_eq!(
            n("0")
                .multiply(&BigDecimal::ten_pow(3))
                .strip_trailing_zeros()
                .to_plain_string(),
            "0"
        );
    }
}
