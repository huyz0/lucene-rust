//! `java.lang.StrictMath`'s `sin`, `cos`, `asin` and `acos`: the fdlibm
//! algorithms Java specifies bit for bit (`java.lang.FdLibm`).
//!
//! Lucene's geo code and `SloppyMath` depend on these exact bits: SloppyMath
//! builds its lookup tables from `StrictMath.sin/cos/asin` at class-load time,
//! and `Tessellator.angle` breaks ties between hole bridges on `Math.acos`
//! (which the JDK delegates to `StrictMath.acos`). Rust's `f64::sin` & co. go
//! to the platform libm, which is usually -- not always -- correctly rounded,
//! where fdlibm is not, so the two can differ in the last bit. The ports here
//! are checked against the JDK's own output by
//! `crates/lucene-util/tests/geo_fixtures.rs` (`fixtures/data/geo/strict_math.tsv`).
//!
//! Only the argument ranges Lucene reaches are ported in full: `sin`/`cos`
//! reduce arguments up to `2^19 * pi/2` with fdlibm's medium-size Cody-Waite
//! path; beyond that (never reached by Lucene: `SloppyMath.cos` falls back to
//! `Math.cos` only past ~4e6 radians, which it then hands to this module)
//! the platform libm is used, and the module says so rather than pretending.
//!
//! `x - x` and `(x - x) / (x - x)` are fdlibm's spelling of "the NaN the
//! FPU makes": its sign and payload are the JDK's raw bits, which the
//! fixture compares, so they stay (hence `clippy::eq_op` is allowed).
#![allow(clippy::eq_op)]

/// High word of a double (`__HI(x)`).
#[inline]
fn hi(x: f64) -> i32 {
    (x.to_bits() >> 32) as u32 as i32
}

/// Low word of a double (`__LO(x)`).
#[inline]
fn lo(x: f64) -> u32 {
    x.to_bits() as u32
}

/// `x` with its low word replaced by zero.
#[inline]
fn with_lo_zero(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0xFFFF_FFFF_0000_0000)
}

const S1: f64 = f64::from_bits(0xBFC5_5555_5555_5549);
const S2: f64 = f64::from_bits(0x3F81_1111_1110_F8A6);
const S3: f64 = f64::from_bits(0xBF2A_01A0_19C1_61D5);
const S4: f64 = f64::from_bits(0x3EC7_1DE3_57B1_FE7D);
const S5: f64 = f64::from_bits(0xBE5A_E5E6_8A2B_9CEB);
const S6: f64 = f64::from_bits(0x3DE5_D93A_5ACF_D57C);

/// fdlibm `__kernel_sin(x, y, iy)` on `[-pi/4, pi/4]`.
fn kernel_sin(x: f64, y: f64, iy: i32) -> f64 {
    let ix = hi(x) & 0x7fff_ffff;
    if ix < 0x3e40_0000 && (x as i32) == 0 {
        return x;
    }
    let z = x * x;
    let v = z * x;
    let r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    if iy == 0 {
        x + v * (S1 + z * r)
    } else {
        x - ((z * (0.5 * y - v * r) - y) - v * S1)
    }
}

const C1: f64 = f64::from_bits(0x3FA5_5555_5555_554C);
const C2: f64 = f64::from_bits(0xBF56_C16C_16C1_5177);
const C3: f64 = f64::from_bits(0x3EFA_01A0_19CB_1590);
const C4: f64 = f64::from_bits(0xBE92_7E4F_809C_52AD);
const C5: f64 = f64::from_bits(0x3E21_EE9E_BDB4_B1C4);
const C6: f64 = f64::from_bits(0xBDA8_FAE9_BE88_38D4);

/// fdlibm `__kernel_cos(x, y)` on `[-pi/4, pi/4]`.
fn kernel_cos(x: f64, y: f64) -> f64 {
    let ix = hi(x) & 0x7fff_ffff;
    if ix < 0x3e40_0000 && (x as i32) == 0 {
        return 1.0;
    }
    let z = x * x;
    let r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    if ix < 0x3FD3_3333 {
        1.0 - (0.5 * z - (z * r - x * y))
    } else {
        let qx = if ix > 0x3fe9_0000 {
            0.28125
        } else {
            // __HI(qx) = ix - 0x00200000, __LO(qx) = 0.
            f64::from_bits(u64::from((ix - 0x0020_0000) as u32) << 32)
        };
        let hz = 0.5 * z - qx;
        let a = 1.0 - qx;
        a - (hz - (z * r - x * y))
    }
}

const INVPIO2: f64 = f64::from_bits(0x3FE4_5F30_6DC9_C883);
const PIO2_1: f64 = f64::from_bits(0x3FF9_21FB_5440_0000);
const PIO2_1T: f64 = f64::from_bits(0x3DD0_B461_1A62_6331);
const PIO2_2: f64 = f64::from_bits(0x3DD0_B461_1A60_0000);
const PIO2_2T: f64 = f64::from_bits(0x3BA3_198A_2E03_7073);
const PIO2_3: f64 = f64::from_bits(0x3BA3_198A_2E00_0000);
const PIO2_3T: f64 = f64::from_bits(0x397B_839A_2520_49C1);

/// High words of `n * pi/2` for `n = 1..=32` (fdlibm's `npio2_hw`).
const NPIO2_HW: [i32; 32] = [
    0x3FF921FB, 0x400921FB, 0x4012D97C, 0x401921FB, 0x401F6A7A, 0x4022D97C, 0x4025FDBB, 0x402921FB,
    0x402C463A, 0x402F6A7A, 0x4031475C, 0x4032D97C, 0x40346B9C, 0x4035FDBB, 0x40378FDB, 0x403921FB,
    0x403AB41B, 0x403C463A, 0x403DD85A, 0x403F6A7A, 0x40407E4C, 0x4041475C, 0x4042106C, 0x4042D97C,
    0x4043A28C, 0x40446B9C, 0x404534AC, 0x4045FDBB, 0x4046C6CB, 0x40478FDB, 0x404858EB, 0x404921FB,
];

/// fdlibm `__ieee754_rem_pio2` for `|x| <= 2^19 * pi/2`: returns `n` and
/// `x - n*pi/2` as the pair `(y0, y1)`. `None` past that range.
fn rem_pio2(x: f64) -> Option<(i32, f64, f64)> {
    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        return Some((0, x, 0.0));
    }
    if ix < 0x4002_d97c {
        // |x| < 3pi/4, special case with n = +-1.
        return Some(if hx > 0 {
            let mut z = x - PIO2_1;
            if ix != 0x3ff9_21fb {
                let y0 = z - PIO2_1T;
                (1, y0, (z - y0) - PIO2_1T)
            } else {
                z -= PIO2_2;
                let y0 = z - PIO2_2T;
                (1, y0, (z - y0) - PIO2_2T)
            }
        } else {
            let mut z = x + PIO2_1;
            if ix != 0x3ff9_21fb {
                let y0 = z + PIO2_1T;
                (-1, y0, (z - y0) + PIO2_1T)
            } else {
                z += PIO2_2;
                let y0 = z + PIO2_2T;
                (-1, y0, (z - y0) + PIO2_2T)
            }
        });
    }
    if ix > 0x4139_21fb {
        return None;
    }
    // Medium size.
    let t = x.abs();
    let n = (t * INVPIO2 + 0.5) as i32;
    let f_n = f64::from(n);
    let mut r = t - f_n * PIO2_1;
    let mut w = f_n * PIO2_1T;
    let y0;
    if n < 32 && ix != NPIO2_HW[(n - 1) as usize] {
        y0 = r - w;
    } else {
        let j = ix >> 20;
        let mut y = r - w;
        let i = j - ((hi(y) >> 20) & 0x7ff);
        if i > 16 {
            let t2 = r;
            w = f_n * PIO2_2;
            r = t2 - w;
            w = f_n * PIO2_2T - ((t2 - r) - w);
            y = r - w;
            let i = j - ((hi(y) >> 20) & 0x7ff);
            if i > 49 {
                let t3 = r;
                w = f_n * PIO2_3;
                r = t3 - w;
                w = f_n * PIO2_3T - ((t3 - r) - w);
                y = r - w;
            }
        }
        y0 = y;
    }
    let y1 = (r - y0) - w;
    if hx < 0 {
        Some((-n, -y0, -y1))
    } else {
        Some((n, y0, y1))
    }
}

/// `StrictMath.sin`.
pub fn sin(x: f64) -> f64 {
    let ix = hi(x) & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        return kernel_sin(x, 0.0, 0);
    }
    if ix >= 0x7ff0_0000 {
        return x - x;
    }
    match rem_pio2(x) {
        Some((n, y0, y1)) => match n & 3 {
            0 => kernel_sin(y0, y1, 1),
            1 => kernel_cos(y0, y1),
            2 => -kernel_sin(y0, y1, 1),
            _ => -kernel_cos(y0, y1),
        },
        None => x.sin(),
    }
}

/// `StrictMath.cos`.
pub fn cos(x: f64) -> f64 {
    let ix = hi(x) & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        return kernel_cos(x, 0.0);
    }
    if ix >= 0x7ff0_0000 {
        return x - x;
    }
    match rem_pio2(x) {
        Some((n, y0, y1)) => match n & 3 {
            0 => kernel_cos(y0, y1),
            1 => -kernel_sin(y0, y1, 1),
            2 => -kernel_cos(y0, y1),
            _ => kernel_sin(y0, y1, 1),
        },
        None => x.cos(),
    }
}

const PI: f64 = f64::from_bits(0x4009_21FB_5444_2D18);
const PIO2_HI: f64 = f64::from_bits(0x3FF9_21FB_5444_2D18);
const PIO2_LO: f64 = f64::from_bits(0x3C91_A626_3314_5C07);
const PIO4_HI: f64 = f64::from_bits(0x3FE9_21FB_5444_2D18);
const PS0: f64 = f64::from_bits(0x3fc5_5555_5555_5555);
const PS1: f64 = f64::from_bits(0xbfd4_d612_03eb_6f7d);
const PS2: f64 = f64::from_bits(0x3fc9_c155_0e88_4455);
const PS3: f64 = f64::from_bits(0xbfa4_8228_b568_8f3b);
const PS4: f64 = f64::from_bits(0x3f49_efe0_7501_b288);
const PS5: f64 = f64::from_bits(0x3f02_3de1_0dfd_f709);
const QS1: f64 = f64::from_bits(0xc003_3a27_1c8a_2d4b);
const QS2: f64 = f64::from_bits(0x4000_2ae5_9c59_8ac8);
const QS3: f64 = f64::from_bits(0xbfe6_066c_1b8d_0159);
const QS4: f64 = f64::from_bits(0x3fb3_b8c5_b12e_9282);

#[inline]
fn p_of(t: f64) -> f64 {
    t * (PS0 + t * (PS1 + t * (PS2 + t * (PS3 + t * (PS4 + t * PS5)))))
}

#[inline]
fn q_of(t: f64) -> f64 {
    1.0 + t * (QS1 + t * (QS2 + t * (QS3 + t * QS4)))
}

/// `StrictMath.asin` (fdlibm `e_asin.c`).
pub fn asin(x: f64) -> f64 {
    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x3ff0_0000 {
        // |x| >= 1
        if ((ix - 0x3ff0_0000) as u32 | lo(x)) == 0 {
            // asin(1) = +-pi/2 with inexact
            return x * PIO2_HI + x * PIO2_LO;
        }
        return (x - x) / (x - x);
    } else if ix < 0x3fe0_0000 {
        // |x| < 0.5
        if ix < 0x3e40_0000 {
            // fdlibm: `if (huge + x > one) return x;` -- always true here.
            return x;
        }
        let t = x * x;
        let w = p_of(t) / q_of(t);
        return x + x * w;
    }
    // 1 > |x| >= 0.5
    let w = 1.0 - x.abs();
    let t = w * 0.5;
    let p = p_of(t);
    let q = q_of(t);
    let s = t.sqrt();
    let t = if ix >= 0x3FEF_3333 {
        // |x| > 0.975
        let w = p / q;
        PIO2_HI - (2.0 * (s + s * w) - PIO2_LO)
    } else {
        let w = with_lo_zero(s);
        let c = (t - w * w) / (s + w);
        let r = p / q;
        let p = 2.0 * s * r - (PIO2_LO - 2.0 * c);
        let q = PIO4_HI - 2.0 * w;
        PIO4_HI - (p - q)
    };
    if hx > 0 {
        t
    } else {
        -t
    }
}

/// `StrictMath.acos` (fdlibm `e_acos.c`); `Math.acos` delegates to it.
pub fn acos(x: f64) -> f64 {
    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x3ff0_0000 {
        // |x| >= 1
        if ((ix - 0x3ff0_0000) as u32 | lo(x)) == 0 {
            return if hx > 0 { 0.0 } else { PI + 2.0 * PIO2_LO };
        }
        return (x - x) / (x - x);
    }
    if ix < 0x3fe0_0000 {
        // |x| < 0.5
        if ix <= 0x3c60_0000 {
            return PIO2_HI + PIO2_LO;
        }
        let z = x * x;
        let r = p_of(z) / q_of(z);
        PIO2_HI - (x - (PIO2_LO - x * r))
    } else if hx < 0 {
        // x < -0.5
        let z = (1.0 + x) * 0.5;
        let p = p_of(z);
        let q = q_of(z);
        let s = z.sqrt();
        let r = p / q;
        let w = r * s - PIO2_LO;
        PI - 2.0 * (s + w)
    } else {
        // x > 0.5
        let z = (1.0 - x) * 0.5;
        let s = z.sqrt();
        let df = with_lo_zero(s);
        let c = (z - df * df) / (s + df);
        let p = p_of(z);
        let q = q_of(z);
        let r = p / q;
        let w = r * s + c;
        2.0 * (df + w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
