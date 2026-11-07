//! Shortest decimal digits of an `f32` (Ryu, Ulf Adams, PLDI 2018), for Java's
//! `Float.toString`: the fewest digits that read back as the value, of those the closest, a tie
//! going to the even digit -- Java's choice too (JDK 19 and later), except that Java prints at
//! least two digits, which [`crate::attributes::push_java_float_string`] settles when this gives
//! one.

/// Bits of precision kept in [`POW5_INV`] and [`POW5`].
const POW5_INV_BITCOUNT: i32 = 59;
const POW5_BITCOUNT: i32 = 61;

/// `ceil(log2(5^e))`, plus one for `e = 0` (`pow5bits` in the reference implementation).
/// ARITH: `e` is in `0..=47`, so the product stays far inside `i32`.
#[allow(clippy::arithmetic_side_effects)]
const fn pow5bits(e: i32) -> i32 {
    ((e * 1_217_359) >> 19) + 1
}

/// `floor(log10(2^e))` for `e` in `0..=1650`.
/// ARITH: callers pass `e` in `0..=104` (an f32's binary exponent), far inside `i32`.
#[allow(clippy::arithmetic_side_effects)]
const fn log10_pow2(e: i32) -> i32 {
    (e * 78_913) >> 18
}

/// `floor(log10(5^e))` for `e` in `0..=2620`.
/// ARITH: callers pass `e` in `1..=151` (an f32's negated binary exponent), far inside `i32`.
#[allow(clippy::arithmetic_side_effects)]
const fn log10_pow5(e: i32) -> i32 {
    (e * 732_923) >> 20
}

/// `floor(2^(pow5bits(i) - 1 + 59) / 5^i) + 1`: `5^-i` scaled to 59 significant bits, rounded up.
/// The largest shift, at `i = 30`, is exactly 128, where `u128::MAX / 5^i` is the same floor
/// (no power of five above one divides `2^128`).
/// ARITH: `5^30 < 2^70` and the quotient is below `2^60`, so `q + 1` fits `u64`.
#[allow(clippy::arithmetic_side_effects, clippy::cast_sign_loss)]
const POW5_INV: [u64; 31] = {
    let mut t = [0u64; 31];
    let mut i = 0;
    while i < 31 {
        let p = 5u128.pow(i as u32);
        let shift = pow5bits(i) - 1 + POW5_INV_BITCOUNT;
        let q = if shift == 128 {
            u128::MAX / p
        } else {
            (1u128 << shift) / p
        };
        t[i as usize] = (q + 1) as u64;
        i += 1;
    }
    t
};

/// `5^i`'s top 61 bits.
/// ARITH: `5^46 < 2^107` fits `u128`; the shifts are by `0..=60` and `0..=46`.
#[allow(clippy::arithmetic_side_effects, clippy::cast_sign_loss)]
const POW5: [u64; 47] = {
    let mut t = [0u64; 47];
    let mut i = 0;
    while i < 47 {
        let p = 5u128.pow(i as u32);
        let bits = pow5bits(i);
        t[i as usize] = if bits <= POW5_BITCOUNT {
            (p << (POW5_BITCOUNT - bits)) as u64
        } else {
            (p >> (bits - POW5_BITCOUNT)) as u64
        };
        i += 1;
    }
    t
};

/// `(m * factor) >> shift` for `shift > 32`, the product kept to 96 bits.
/// ARITH: each partial product is a 32-bit by 32-bit one, and `(lo >> 32) + hi < 2^64`.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn mul_shift(m: u32, factor: u64, shift: i32) -> u32 {
    let lo = u64::from(m) * (factor & 0xffff_ffff);
    let hi = u64::from(m) * (factor >> 32);
    (((lo >> 32) + hi) >> (shift - 32)) as u32
}

#[allow(clippy::cast_sign_loss)]
fn mul_pow5_inv_div_pow2(m: u32, q: i32, j: i32) -> u32 {
    mul_shift(m, POW5_INV[q as usize], j)
}

#[allow(clippy::cast_sign_loss)]
fn mul_pow5_div_pow2(m: u32, i: i32, j: i32) -> u32 {
    mul_shift(m, POW5[i as usize], j)
}

/// Whether `5^p` divides `v`.
/// ARITH: `count` is at most 13 (`5^14 > u32::MAX`).
#[allow(clippy::arithmetic_side_effects)]
fn multiple_of_pow5(mut v: u32, p: i32) -> bool {
    let mut count = 0;
    while v != 0 && v.is_multiple_of(5) {
        v /= 5;
        count += 1;
    }
    count >= p
}

/// Whether `2^p` divides `v`.
/// ARITH: callers pass `p` in `1..=29`.
#[allow(clippy::arithmetic_side_effects, clippy::cast_sign_loss)]
fn multiple_of_pow2(v: u32, p: i32) -> bool {
    v & ((1u32 << p) - 1) == 0
}

/// The shortest digits of positive finite nonzero `v`, closest, ties to even, as
/// `(digits, exponent)` with `v ≈ digits * 10^exponent` (`digits` has no trailing zero beyond
/// what the shortest form needs).
///
/// The arithmetic is the reference `f2d`: every intermediate stays inside its type for the
/// `f32` exponent range, which the reference proves and the module's tests check.
/// ARITH: `mv`, `mp`, `mm` are below `2^27`; `vp - 1` follows `vp > vm >= 0`.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
pub(crate) fn shortest(v: f32) -> (u32, i32) {
    const MANTISSA_BITS: i32 = 23;
    const BIAS: i32 = 127;
    let bits = v.to_bits();
    let ieee_mantissa = bits & ((1 << MANTISSA_BITS) - 1);
    let ieee_exponent = ((bits >> MANTISSA_BITS) & 0xff) as i32;
    let (e2, m2) = if ieee_exponent == 0 {
        (1 - BIAS - MANTISSA_BITS - 2, ieee_mantissa)
    } else {
        (
            ieee_exponent - BIAS - MANTISSA_BITS - 2,
            (1 << MANTISSA_BITS) | ieee_mantissa,
        )
    };
    let accept_bounds = m2 & 1 == 0;
    let mv = 4 * m2;
    let mp = 4 * m2 + 2;
    let mm_shift = u32::from(ieee_mantissa != 0 || ieee_exponent <= 1);
    let mm = 4 * m2 - 1 - mm_shift;

    let (mut vr, mut vp, mut vm, e10);
    let mut vm_trailing_zeros = false;
    let mut vr_trailing_zeros = false;
    let mut last_removed = 0u32;
    if e2 >= 0 {
        let q = log10_pow2(e2);
        e10 = q;
        let k = POW5_INV_BITCOUNT + pow5bits(q) - 1;
        let i = -e2 + q + k;
        vr = mul_pow5_inv_div_pow2(mv, q, i);
        vp = mul_pow5_inv_div_pow2(mp, q, i);
        vm = mul_pow5_inv_div_pow2(mm, q, i);
        if q != 0 && (vp - 1) / 10 <= vm / 10 {
            let l = POW5_INV_BITCOUNT + pow5bits(q - 1) - 1;
            last_removed = mul_pow5_inv_div_pow2(mv, q - 1, -e2 + q - 1 + l) % 10;
        }
        if q <= 9 {
            if mv % 5 == 0 {
                vr_trailing_zeros = multiple_of_pow5(mv, q);
            } else if accept_bounds {
                vm_trailing_zeros = multiple_of_pow5(mm, q);
            } else {
                vp -= u32::from(multiple_of_pow5(mp, q));
            }
        }
    } else {
        let q = log10_pow5(-e2);
        e10 = q + e2;
        let i = -e2 - q;
        let k = pow5bits(i) - POW5_BITCOUNT;
        let j = q - k;
        vr = mul_pow5_div_pow2(mv, i, j);
        vp = mul_pow5_div_pow2(mp, i, j);
        vm = mul_pow5_div_pow2(mm, i, j);
        if q != 0 && (vp - 1) / 10 <= vm / 10 {
            let j = q - 1 - (pow5bits(i + 1) - POW5_BITCOUNT);
            last_removed = mul_pow5_div_pow2(mv, i + 1, j) % 10;
        }
        if q <= 1 {
            vr_trailing_zeros = true;
            if accept_bounds {
                vm_trailing_zeros = mm_shift == 1;
            } else {
                vp -= 1;
            }
        } else if q < 31 {
            vr_trailing_zeros = multiple_of_pow2(mv, q - 1);
        }
    }

    let mut removed = 0;
    let output = if vm_trailing_zeros || vr_trailing_zeros {
        while vp / 10 > vm / 10 {
            vm_trailing_zeros &= vm % 10 == 0;
            vr_trailing_zeros &= last_removed == 0;
            last_removed = vr % 10;
            vr /= 10;
            vp /= 10;
            vm /= 10;
            removed += 1;
        }
        if vm_trailing_zeros {
            while vm % 10 == 0 {
                vr_trailing_zeros &= last_removed == 0;
                last_removed = vr % 10;
                vr /= 10;
                vm /= 10;
                removed += 1;
            }
        }
        if vr_trailing_zeros && last_removed == 5 && vr % 2 == 0 {
            // An exact tie: to the even digit.
            last_removed = 4;
        }
        vr + u32::from((vr == vm && (!accept_bounds || !vm_trailing_zeros)) || last_removed >= 5)
    } else {
        while vp / 10 > vm / 10 {
            last_removed = vr % 10;
            vr /= 10;
            vp /= 10;
            vm /= 10;
            removed += 1;
        }
        vr + u32::from(vr == vm || last_removed >= 5)
    };
    (output, e10 + removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The digits and exponent from Rust's own shortest rendering, `{:e}`.
    fn rust_shortest(v: f32) -> (u32, i32) {
        let s = format!("{v:e}");
        let (m, e) = s.split_once('e').unwrap();
        let digits: String = m.chars().filter(char::is_ascii_digit).collect();
        let n = digits.len() as i32;
        (digits.parse().unwrap(), e.parse::<i32>().unwrap() - (n - 1))
    }

    fn normalize((mut d, mut e): (u32, i32)) -> (u32, i32) {
        while d != 0 && d % 10 == 0 {
            d /= 10;
            e += 1;
        }
        (d, e)
    }

    /// Same value as Rust's shortest everywhere, the same digits wherever Rust's is not a
    /// rounded-up tie (Rust rounds a tie up, Ryu and Java to even).
    fn agrees(bits: u32) {
        let v = f32::from_bits(bits);
        if !v.is_finite() || v <= 0.0 {
            return;
        }
        let (d, e) = normalize(shortest(v));
        assert_eq!(
            format!("{d}e{e}").parse::<f32>().unwrap(),
            v,
            "{bits:#x}: {d}e{e} does not read back"
        );
        let (rd, re) = normalize(rust_shortest(v));
        if (d, e) != (rd, re) {
            // Only a tie may differ, and then by one in the last digit, to even.
            let (dn, rn) = (d.to_string().len(), rd.to_string().len());
            assert_eq!((dn, e), (rn, re), "{bits:#x}: {d}e{e} vs {rd}e{re}");
            assert_eq!(rd, d + 1, "{bits:#x}: {d}e{e} vs {rd}e{re}");
            assert_eq!(d % 2, 0, "{bits:#x}");
        }
    }

    #[test]
    fn tables_match_the_reference_endpoints() {
        assert_eq!(POW5_INV[0], 576_460_752_303_423_489);
        assert_eq!(POW5[0], 1_152_921_504_606_846_976);
        assert_eq!(POW5_INV[1], 461_168_601_842_738_791);
        assert_eq!(POW5[1], 1_441_151_880_758_558_720);
    }

    #[test]
    fn shortest_digits_read_back_and_match_rust_but_for_ties() {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..2_000_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            agrees((x >> 32) as u32);
        }
        for bits in (1..100_000).chain(0x7f7f_0000..0x7f80_0000) {
            agrees(bits);
        }
        for i in 0..200_000u32 {
            agrees(0x3f80_0000 - 100_000 + i);
        }
        assert_eq!(shortest(f32::from_bits(0x4884_69f4)), (27_118_362, -2));
        // `Float.MIN_VALUE`: one digit here, which the caller widens to Java's two.
        assert_eq!(shortest(f32::from_bits(1)), (1, -45));
    }

    /// Every positive finite `f32` (`cargo test --release -- --ignored`, about a minute on four
    /// cores).
    #[test]
    #[ignore = "exhaustive: run by hand after changing this module"]
    fn every_f32_agrees() {
        let threads = 4u32;
        let chunk = 0x7f80_0000 / threads;
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                std::thread::spawn(move || {
                    for bits in t * chunk + 1..(t + 1) * chunk + 1 {
                        agrees(bits);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}
