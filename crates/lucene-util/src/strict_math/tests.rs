#![cfg(test)]
use super::*;

/// `StrictMath.exp`/`log` of hand-picked arguments (every branch of
/// both), as JDK 21 and 25 answer them, then digests of 200,000
/// pseudo-random arguments each (`(x bits, exp bits, log bits)`).
#[test]
fn exp_and_log_are_the_jdks_bit_for_bit() {
    let fixed: [(u64, u64, u64); 35] = [
        (0x0000000000000000, 0x3ff0000000000000, 0xfff0000000000000),
        (0x8000000000000000, 0x3ff0000000000000, 0xfff0000000000000),
        (0x3ff0000000000000, 0x4005bf0a8b14576a, 0x0000000000000000),
        (0xbff0000000000000, 0x3fd78b56362cef38, 0xfff8000000000000),
        (0x3fe0000000000000, 0x3ffa61298e1e069c, 0xbfe62e42fefa39ef),
        (0x3fd0000000000000, 0x3ff48b5e3c3e8186, 0xbff62e42fefa39ef),
        (0x4000000000000000, 0x401d8e64b8d4ddae, 0x3fe62e42fefa39ef),
        (0x4024000000000000, 0x40d5829dcf950560, 0x40026bb1bbb55516),
        (0xc024000000000000, 0x3f07cd79b5647c9a, 0xfff8000000000000),
        (0x4059000000000000, 0x48f3494a9b171bf5, 0x40126bb1bbb55516),
        (0xc059000000000000, 0x36ea8c1f14e2af5d, 0xfff8000000000000),
        (0x4086280000000000, 0x7fdd422d2be5dc9b, 0x401a416357d87f55),
        (0x40862e6666666666, 0x7ff0000000000000, 0x401a428af76e005d),
        (0xc087480000000000, 0x0000000000000001, 0xfff8000000000000),
        (0xc08749999999999a, 0x0000000000000000, 0xfff8000000000000),
        (0xc087500000000000, 0x0000000000000000, 0xfff8000000000000),
        (0x3ddb7cdfd9d7bdbb, 0x3ff000000006df38, 0xc037069e2aa2aa5b),
        (0xbddb7cdfd9d7bdbb, 0x3feffffffff24190, 0xfff8000000000000),
        (0x01a56e1fc2f8f359, 0x3ff0000000000000, 0xc085963447f87fb5),
        (0x0000000000000001, 0x3ff0000000000000, 0xc0874385446d71c3),
        (0x000fd1d7d505cd02, 0x3ff0000000000000, 0xc0862343131aa1ed),
        (0x7e37e43c8800759c, 0x7ff0000000000000, 0x4085963447f87fb5),
        (0x3fd62e42fead449c, 0x3ff6a09e66640654, 0xbff0f45e26abfb60),
        (0x3fd62eb1c432ca58, 0x3ff6a0c5906bc779, 0xbff0f40e3fd2c1b4),
        (0x3ff0a29c779a6b51, 0x4006a07f99728ef9, 0x3fa3eeec316ff708),
        (0x3ff0a3d70a3d70a4, 0x4006a23c87af69e8, 0x3fa414bcc0a3665d),
        (0x7ff0000000000000, 0x7ff0000000000000, 0x7ff0000000000000),
        (0xfff0000000000000, 0x0000000000000000, 0xfff8000000000000),
        (0x7ff8000000000000, 0x7ff8000000000000, 0x7ff8000000000000),
        (0x3ff000001ad7f29b, 0x4005bf0aaf904984, 0x3e7ad7f2847b6492),
        (0x3fefffffca501acb, 0x4005bf0a6698658d, 0xbe7ad7f2b1049b9f),
        (0x3fd5555555555555, 0x3ff6546db1ba2d13, 0xbff193ea7aad030b),
        (0x4002aaaaaaaaaaab, 0x40249fe058a62f2d, 0x3feb1d10670aae9a),
        (0xc0862066151add8b, 0x0016a09e667f3bcc, 0xfff8000000000000),
        (0xc0864f886378b146, 0x0000100000000000, 0xfff8000000000000),
    ];
    // A NaN's sign bit is the hardware's choice for `(x - x) / 0` (x86
    // sets it, aarch64 does not) and no Java code can observe it, so a
    // NaN only has to be a NaN.
    let same = |got: f64, want: u64| {
        let want = f64::from_bits(want);
        if want.is_nan() {
            got.is_nan()
        } else {
            got.to_bits() == want.to_bits()
        }
    };
    for (x, e, l) in fixed {
        let x = f64::from_bits(x);
        assert!(same(exp(x), e), "exp({x:e}) = {:#x}", exp(x).to_bits());
        assert!(same(log(x), l), "log({x:e}) = {:#x}", log(x).to_bits());
    }
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let (mut h1, mut h2) = (0xcbf2_9ce4_8422_2325u64, 0xcbf2_9ce4_8422_2325u64);
    for i in 0..200_000 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
        let xe = (unit - 0.5) * 1500.0;
        let xl = if i % 3 == 0 {
            unit * 4.0
        } else {
            f64::from_bits((state >> 1) & 0x7fef_ffff_ffff_ffff)
        };
        h1 = (h1 ^ exp(xe).to_bits()).wrapping_mul(0x0100_0000_01b3);
        h2 = (h2 ^ log(xl).to_bits()).wrapping_mul(0x0100_0000_01b3);
    }
    assert_eq!(h1, 0x4646_7ffa_1f1a_6286);
    assert_eq!(h2, 0xbe29_7cda_956b_41c3);
}

/// `StrictMath.pow` of hand-picked arguments (each special case and
/// branch: NaN, infinities, signed zeros, odd/even/non-integer `y` for a
/// negative `x`, `|y| > 2^31`, overflow, underflow, subnormal results and
/// arguments) as JDK 25 answers them, then the digest of 300,000
/// pseudo-random pairs (`crates/lucene-util/tools/StrictMathPowRef.java`).
#[test]
fn pow_is_the_jdks_bit_for_bit() {
    let fixed: [(u64, u64, u64); 42] = [
        (0x4000000000000000, 0x4024000000000000, 0x4090000000000000),
        (0x4000000000000000, 0x3fe0000000000000, 0x3ff6a09e667f3bcd),
        (0xc020000000000000, 0x3fd5555555555555, 0xfff8000000000000),
        (0xc000000000000000, 0x4008000000000000, 0xc020000000000000),
        (0xc000000000000000, 0x4010000000000000, 0x4030000000000000),
        (0xc000000000000000, 0x3fe0000000000000, 0xfff8000000000000),
        (0x0000000000000000, 0xbff0000000000000, 0x7ff0000000000000),
        (0x8000000000000000, 0xc008000000000000, 0xfff0000000000000),
        (0x8000000000000000, 0xc000000000000000, 0x7ff0000000000000),
        (0xfff0000000000000, 0x3fe0000000000000, 0x7ff0000000000000),
        (0xfff0000000000000, 0x4008000000000000, 0xfff0000000000000),
        (0xfff0000000000000, 0xc008000000000000, 0x8000000000000000),
        (0xbff0000000000000, 0x7ff0000000000000, 0xfff8000000000000),
        (0x3fe0000000000000, 0xfff0000000000000, 0x7ff0000000000000),
        (0x4008000000000000, 0xfff0000000000000, 0x0000000000000000),
        (0x7ff8000000000000, 0x0000000000000000, 0x3ff0000000000000),
        (0x3ff0000000000000, 0x7ff8000000000000, 0x7ff8000000000000),
        (0xbff0000000000000, 0x7e37e43c8800759c, 0x3ff0000000000000),
        (0xbff0000000000000, 0x4004000000000000, 0xfff8000000000000),
        (0x3ff000001ad7f29b, 0x4202a05f20000000, 0x7ff0000000000000),
        (0x3fefffffca501acb, 0x4202a05f20000000, 0x0000000000000000),
        (0x3ff000001ad7f29b, 0xc202a05f20000000, 0x0000000000000000),
        (0x3fe0000000000000, 0x41e65a0bc0000000, 0x0000000000000000),
        (0x3ff8000000000000, 0x41e65a0bc0000000, 0x7ff0000000000000),
        (0x3ff000000006df38, 0x41e65a0bc0000000, 0x3ff5990595894040),
        (0x3feffffffff24190, 0xc1e65a0bc0000000, 0x3ff59905958c08a9),
        (0x4024000000000000, 0x4073400000000000, 0x7fe1ccf385ebc8a0),
        (0x4024000000000000, 0x4073500000000000, 0x7ff0000000000000),
        (0x4024000000000000, 0xc074300000000000, 0x0000000000000002),
        (0x4024000000000000, 0xc074400000000000, 0x0000000000000000),
        (0x4000000000000000, 0x408fffffcb923a2a, 0x7fefff6ea43bd988),
        (0x4000000000000000, 0xc090c80000000000, 0x0000000000000001),
        (0x4000000000000000, 0xc090cc0000000000, 0x0000000000000000),
        (0x4000000000000000, 0xc090ca0000000000, 0x0000000000000001),
        (0x0000000000000001, 0x3fd3333333333333, 0x2bcbdb8cdadbe161),
        (0x000012688b70e62b, 0x3ff8000000000000, 0x0000000000000000),
        (0x401c000000000000, 0xbfe0000000000000, 0x3fd83091e6a7f7e6),
        (0x3ffb333333333333, 0x3ff4cccccccccccd, 0x3fffe4c32630d4d8),
        (0x3ffbb67ae8584caa, 0x401c666666666666, 0x4048b3faa452ed20),
        (0x4005bf0a8b145769, 0x408628b76e3a7b61, 0x7fdfffffffffffc6),
        (0xc008000000000000, 0xc340000000000000, 0x0000000000000000),
        (0xc008000000000000, 0x4341c37937e08000, 0x7ff0000000000000),
    ];
    for (x, y, want) in fixed {
        let (x, y) = (f64::from_bits(x), f64::from_bits(y));
        let got = pow(x, y);
        let want = f64::from_bits(want);
        assert!(
            if want.is_nan() {
                got.is_nan()
            } else {
                got.to_bits() == want.to_bits()
            },
            "pow({x:e}, {y:e}) = {got:e}, want {want:e}"
        );
    }
    fn next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        *state
    }
    fn unit(state: &mut u64) -> f64 {
        (next(state) >> 11) as f64 / (1u64 << 53) as f64
    }
    let s = &mut 0x9E37_79B9_7F4A_7C15u64;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for i in 0..300_000 {
        let (x, y) = match i % 3 {
            0 => (unit(s) * 20.0 - 5.0, unit(s) * 60.0 - 30.0),
            1 => {
                let x = f64::from_bits((next(s) >> 1) & 0x7fef_ffff_ffff_ffff);
                (x, unit(s) * 4.0 - 2.0)
            }
            _ => (1.0 + (unit(s) - 0.5) * 1e-6, (unit(s) - 0.5) * 1e10),
        };
        // `Double.doubleToLongBits`: one NaN, whatever sign the platform
        // gives `(x - x) / (x - x)` (negative on x86-64, positive on
        // aarch64).
        let got = pow(x, y);
        let bits = if got.is_nan() {
            f64::NAN.to_bits()
        } else {
            got.to_bits()
        };
        h = (h ^ bits).wrapping_mul(0x0100_0000_01b3);
    }
    assert_eq!(h, 0xc929_8d43_7950_60a9);
}

/// Every branch of `expm1`, `cosh`, `sinh`, `tanh` and `log10` against
/// values that hold to a few ulps (the bit-for-bit comparison with the JDK
/// is `lucene-search`'s `expressions_fixtures.rs`).
#[test]
fn hyperbolic_and_log10_branches() {
    let near = |got: f64, want: f64| {
        assert!(
            (got - want).abs() <= want.abs() * 4e-16,
            "{got:e} vs {want:e}"
        );
    };
    // expm1: tiny, |x| < 0.5 ln2, k = +-1, k in 2..20, k >= 20, k <= -2,
    // k > 56, k == 1024, overflow, underflow, inf, NaN
    assert_eq!(expm1(1e-20), 1e-20);
    near(expm1(0.3), 0.3498588075760031);
    near(expm1(0.6), 0.8221188003905089);
    near(expm1(-0.6), -0.45118836390597356);
    near(expm1(-0.9), -0.5934303402594009);
    near(expm1(3.0), 19.085536923187668);
    near(expm1(20.0), 4.851651944097903e8);
    near(expm1(-5.0), -0.9932620530009145);
    near(expm1(50.0), 5.184705528587072e21);
    assert!(expm1(710.0).is_infinite());
    assert_eq!(expm1(-1000.0), -1.0);
    assert_eq!(expm1(f64::INFINITY), f64::INFINITY);
    assert_eq!(expm1(f64::NEG_INFINITY), -1.0);
    assert!(expm1(f64::NAN).is_nan());
    near(expm1(709.7), 1.6549840276802644e308);
    // cosh
    assert_eq!(cosh(1e-10), 1.0);
    near(cosh(0.3), 1.0453385141288605);
    near(cosh(-5.0), 74.20994852478785);
    near(cosh(30.0), 5.343237290762231e12);
    near(cosh(710.0), 1.1169973830808557e308);
    assert!(cosh(711.0).is_infinite());
    assert_eq!(cosh(f64::NEG_INFINITY), f64::INFINITY);
    assert!(cosh(f64::NAN).is_nan());
    // sinh
    assert_eq!(sinh(1e-20), 1e-20);
    near(sinh(0.5), 0.5210953054937474);
    near(sinh(-2.0), -3.626860407847019);
    near(sinh(30.0), 5.343237290762231e12);
    near(sinh(-710.0), -1.1169973830808557e308);
    assert!(sinh(711.0).is_infinite());
    assert_eq!(sinh(f64::INFINITY), f64::INFINITY);
    assert!(sinh(f64::NAN).is_nan());
    // tanh
    assert_eq!(tanh(1e-20), 1e-20);
    near(tanh(0.5), 0.46211715726000974);
    near(tanh(-2.0), -0.9640275800758169);
    assert_eq!(tanh(30.0), 1.0);
    assert_eq!(tanh(-30.0), -1.0);
    assert_eq!(tanh(f64::INFINITY), 1.0);
    assert_eq!(tanh(f64::NEG_INFINITY), -1.0);
    assert!(tanh(f64::NAN).is_nan());
    // log10: powers of ten, a subnormal, zero, negative, inf, NaN
    assert_eq!(log10(1000.0), 3.0);
    assert_eq!(log10(1.0), 0.0);
    near(log10(2.0), std::f64::consts::LOG10_2);
    near(log10(5e-324), -323.3062153431158);
    assert_eq!(log10(0.0), f64::NEG_INFINITY);
    assert!(log10(-1.0).is_nan());
    assert_eq!(log10(f64::INFINITY), f64::INFINITY);
    assert!(log10(f64::NAN).is_nan());
}

/// Payne-Hanek reduction (`|x| > 2^19 * pi/2`): the classic hard
/// arguments, then the digest of `sin`, `cos` and `tan` over 100,000
/// pseudo-random ones -- exponents 20 to 1023 of both signs, and every
/// fourth a small odd integer times a power of two, whose long runs of
/// zero chunks take `__kernel_rem_pio2`'s recomputation path -- as JDK 25
/// answers them (`crates/lucene-util/tools/StrictMathTrigRef.java`).
#[test]
fn large_arguments_are_the_jdks_bit_for_bit() {
    let fixed: [(f64, u64, u64, u64); 7] = [
        (
            1e22,
            0xbfeb453ab76bf397,
            0x3fe0be2cef01c8f4,
            0xbffa0f79c1b6b258,
        ),
        (
            1e9,
            0x3fe1778cae83c69b,
            0x3feacff8c7364234,
            0x3fe4d8b249e3dba7,
        ),
        (
            -1e9,
            0xbfe1778cae83c69b,
            0x3feacff8c7364234,
            0xbfe4d8b249e3dba7,
        ),
        (
            1e300,
            0xbfea2c16b010e385,
            0xbfe2699022adc4c1,
            0x3ff6be411f37ac77,
        ),
        (
            f64::MAX,
            0x3f7452fc98b34e97,
            0xbfefffe62ecfab75,
            0xbf74530cfe729484,
        ),
        (
            1_152_921_504_606_846_976.0,
            0xbfea94adab06665c,
            0xbfe1d146047d6947,
            0x3ff7de90e3bee983,
        ),
        (
            1.2676506002282294e30,
            0xbfebe8ed97ac1f59,
            0x3fdf4eb3ff66e36d,
            0xbffc86fbfa8cecc3,
        ),
    ];
    for (x, s, c, t) in fixed {
        assert_eq!(sin(x).to_bits(), s, "sin({x:e})");
        assert_eq!(cos(x).to_bits(), c, "cos({x:e})");
        assert_eq!(tan(x).to_bits(), t, "tan({x:e})");
        assert_eq!(sin_cos(x), (sin(x), cos(x)));
    }
    assert!(sin(f64::INFINITY).is_nan() && cos(f64::NAN).is_nan());
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for i in 0..100_000 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let x = if i % 4 == 3 {
            let odd = ((state >> 40) | 1) as f64;
            odd * f64::from_bits((1023 + (state >> 8) % 1000) << 52)
        } else {
            let exp = 1043 + ((state >> 1) % 1004);
            f64::from_bits((state & 0x800f_ffff_ffff_ffff) | (exp << 52))
        };
        for v in [sin(x), cos(x), tan(x)] {
            h = (h ^ v.to_bits()).wrapping_mul(0x0100_0000_01b3);
        }
    }
    assert_eq!(h, 0x22d7_6e41_d738_3d7b);
}

#[test]
fn sin_cos_is_sin_and_cos_bit_for_bit() {
    let mut args = vec![
        0.0,
        -0.0,
        1e-300,
        -1e-9,
        0.5,
        std::f64::consts::FRAC_PI_4,
        std::f64::consts::FRAC_PI_2,
        -std::f64::consts::FRAC_PI_2,
        std::f64::consts::PI,
        -3.0,
        100.0,
        1e6,
        1e9,
        1e300,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..20_000 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
        args.push((unit - 0.5) * 40.0);
    }
    for n in 1..40 {
        args.push(f64::from(n) * std::f64::consts::FRAC_PI_2);
    }
    for x in args {
        let (s, c) = sin_cos(x);
        assert_eq!(s.to_bits(), sin(x).to_bits(), "sin({x:e})");
        assert_eq!(c.to_bits(), cos(x).to_bits(), "cos({x:e})");
    }
}

#[test]
fn hypot_matches_the_jdk_on_every_branch() {
    // (x, y, Math.hypot(x, y)) as raw bits, from a JDK 21 / 25 run:
    // the plain and the 2^60-ratio paths, both scalings, a subnormal
    // `b`, both medium-size formulas, and two where libm's correctly
    // rounded result is an ulp away from fdlibm's.
    let cases: [(u64, u64, u64); 12] = [
        (0x4008000000000000, 0x4010000000000000, 0x4014000000000000),
        (0x7e37e43c8800759c, 0x7e37e43c8800759c, 0x7e40e4d50f99b211),
        (0x01a56e1fc2f8f359, 0x01a56e1fc2f8f359, 0x01ae4e8d12762225),
        (0x000012688b70e62b, 0x00003739a252b281, 0x00003a365ff2ea11),
        (0x0000000000002788, 0x3ff0000000000000, 0x3ff0000000000000),
        (0x7e37e43c8800759c, 0x01a56e1fc2f8f359, 0x7e37e43c8800759c),
        (0x20a0000000000000, 0x20a8000000000000, 0x20acd82b446159f3),
        (0x5f41000000000000, 0x5f30000000000000, 0x5f42c9cda6892035),
        (0x405bc8e568280000, 0xc024da809e880000, 0x405be8215aac73bd),
        (0x3fd42e57611fbfb0, 0x3fd42e57f34a0ac7, 0x3fdc8a4fc19867ce),
        (0x3ff0000000000000, 0x3c670ef54646d497, 0x3ff0000000000000),
        (0x000024d116e1cc56, 0x01a56e1fc2f8f359, 0x01a56e1fc2f8f359),
    ];
    for (x, y, want) in cases {
        let (x, y) = (f64::from_bits(x), f64::from_bits(y));
        assert_eq!(hypot(x, y).to_bits(), want, "hypot({x:e}, {y:e})");
        assert_eq!(hypot(y, -x).to_bits(), want, "symmetric");
    }
    assert_eq!(hypot(0.0, 0.0), 0.0);
    assert_eq!(hypot(f64::NAN, f64::INFINITY), f64::INFINITY);
    assert_eq!(hypot(f64::NEG_INFINITY, 1.0), f64::INFINITY);
    assert!(hypot(f64::NAN, 1.0).is_nan());
}

#[test]
fn special_values() {
    assert!(sin(f64::NAN).is_nan());
    assert!(cos(f64::INFINITY).is_nan());
    assert!(sin(f64::NEG_INFINITY).is_nan());
    assert_eq!(sin(0.0), 0.0);
    assert_eq!(sin(-0.0).to_bits(), (-0.0f64).to_bits());
    assert_eq!(cos(0.0), 1.0);
    assert_eq!(cos(1e-10), 1.0);
    assert_eq!(asin(1.0), std::f64::consts::FRAC_PI_2);
    assert_eq!(asin(-1.0), -std::f64::consts::FRAC_PI_2);
    assert!(asin(1.5).is_nan());
    assert!(asin(f64::NAN).is_nan());
    assert_eq!(asin(1e-10), 1e-10);
    assert_eq!(acos(1.0), 0.0);
    assert_eq!(acos(-1.0), std::f64::consts::PI);
    assert!(acos(-1.5).is_nan());
    assert_eq!(acos(1e-20), std::f64::consts::FRAC_PI_2);
}

#[test]
fn tan_atan_atan2_special_values() {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    // tan: NaN for infinities and NaN, odd, exact near zero, and the
    // platform fallback past fdlibm's reduction range.
    assert!(tan(f64::INFINITY).is_nan());
    assert!(tan(f64::NAN).is_nan());
    assert_eq!(tan(0.0), 0.0);
    assert_eq!(tan(1e-300), 1e-300);
    assert_eq!(tan(-0.5).to_bits(), (-tan(0.5)).to_bits());
    assert!((tan(1e10) - 1e10f64.tan()).abs() < 1e-6);
    assert!((tan(FRAC_PI_4) - 1.0).abs() < 1e-15);
    // atan: NaN, saturation at +-pi/2, tiny arguments, every interval.
    assert!(atan(f64::NAN).is_nan());
    assert_eq!(atan(1e300), FRAC_PI_2);
    assert_eq!(atan(-1e300), -FRAC_PI_2);
    assert_eq!(atan(f64::INFINITY), FRAC_PI_2);
    assert_eq!(atan(1e-30), 1e-30);
    for x in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 40.0, -0.6, -2.0, -40.0] {
        assert!((atan(x) - x.atan()).abs() <= 2e-16 * x.atan().abs(), "{x}");
    }
    // atan2: NaN, zeros, infinities, the axes, huge and tiny ratios.
    assert!(atan2(f64::NAN, 1.0).is_nan());
    assert!(atan2(1.0, f64::NAN).is_nan());
    assert_eq!(atan2(0.5, 1.0).to_bits(), atan(0.5).to_bits());
    assert_eq!(atan2(0.0, 1.0), 0.0);
    assert_eq!(atan2(-0.0, 1.0).to_bits(), (-0.0f64).to_bits());
    assert_eq!(atan2(0.0, -1.0), PI);
    assert_eq!(atan2(-0.0, -1.0), -PI);
    assert_eq!(atan2(1.0, 0.0), FRAC_PI_2);
    assert_eq!(atan2(-1.0, 0.0), -FRAC_PI_2);
    assert_eq!(atan2(f64::INFINITY, f64::INFINITY), FRAC_PI_4);
    assert_eq!(atan2(-f64::INFINITY, f64::INFINITY), -FRAC_PI_4);
    assert_eq!(atan2(f64::INFINITY, -f64::INFINITY), 3.0 * FRAC_PI_4);
    assert_eq!(atan2(-f64::INFINITY, -f64::INFINITY), -3.0 * FRAC_PI_4);
    assert_eq!(atan2(1.0, f64::INFINITY), 0.0);
    assert_eq!(atan2(-1.0, f64::INFINITY).to_bits(), (-0.0f64).to_bits());
    assert_eq!(atan2(1.0, -f64::INFINITY), PI);
    assert_eq!(atan2(-1.0, -f64::INFINITY), -PI);
    assert_eq!(atan2(f64::INFINITY, 1.0), FRAC_PI_2);
    assert_eq!(atan2(-f64::INFINITY, 1.0), -FRAC_PI_2);
    assert_eq!(atan2(1e300, 1e-300), FRAC_PI_2);
    assert_eq!(atan2(1e-300, -1e300), PI);
    assert_eq!(atan2(-1e-300, -1e300), -PI);
    assert_eq!(atan2(1e-300, 1e300), 0.0);
    for (y, x) in [(1.0, -2.0), (-1.0, -2.0), (3.0, -0.5), (-0.1, 5.0)] {
        let want = f64::atan2(y, x);
        assert!((atan2(y, x) - want).abs() <= 2e-16 * want.abs(), "{y} {x}");
    }
}

#[test]
fn close_to_libm_everywhere() {
    // Bit-identity with the JDK is the fixture test's job; here every
    // branch of the reduction must at least land within an ulp or two.
    let mut x = -40.0f64;
    while x < 40.0 {
        assert!(
            (sin(x) - x.sin()).abs() <= 2e-16 * x.sin().abs().max(1.0),
            "{x}"
        );
        assert!(
            (cos(x) - x.cos()).abs() <= 2e-16 * x.cos().abs().max(1.0),
            "{x}"
        );
        x += 0.0137;
    }
    // Exact multiples of pi/2 exercise the refinement branches.
    for n in 1..40 {
        let x = f64::from(n) * std::f64::consts::FRAC_PI_2;
        assert!((sin(x) - x.sin()).abs() < 1e-15, "{n}");
        assert!((cos(-x) - (-x).cos()).abs() < 1e-15, "{n}");
    }
    // Past the medium range: libm.
    assert_eq!(sin(1e300), 1e300f64.sin());
    assert_eq!(cos(1e300), 1e300f64.cos());
    let mut a = -1.0f64;
    while a <= 1.0 {
        assert!((asin(a) - a.asin()).abs() <= 4e-16, "{a}");
        assert!((acos(a) - a.acos()).abs() <= 8e-16, "{a}");
        a += 0.00093;
    }
}
