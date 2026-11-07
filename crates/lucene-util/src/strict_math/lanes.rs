//! Slice forms of [`log`](super::log), [`exp`](super::exp), [`sin`](super::sin)
//! and [`cos`](super::cos) that return the same bits as the scalar ports.
//!
//! Each lane runs the scalar algorithm's main path with every branch turned
//! into a select, so the compiler vectorises the loop (AVX2 on x86-64-v3,
//! NEON on aarch64). Operations and their order are the scalar ones, and Rust
//! never fuses a multiply and an add, so the rounding is the same. Lanes the
//! main path does not cover (zero, negative, subnormal and non-finite inputs,
//! huge trig arguments, arguments next to a multiple of pi/2, results near
//! underflow) are flagged and recomputed with the scalar function.

use super::{
    hi, C1, C2, C3, C4, C5, C6, INVPIO2, PIO2_1, PIO2_1T, PIO2_2, PIO2_2T, PIO2_3, PIO2_3T, S1, S2,
    S3, S4, S5, S6,
};

const LN2_HI: f64 = f64::from_bits(0x3FE6_2E42_FEE0_0000);
const LN2_LO: f64 = f64::from_bits(0x3DEA_39EF_3579_3C76);
const LG1: f64 = f64::from_bits(0x3FE5_5555_5555_5593);
const LG2: f64 = f64::from_bits(0x3FD9_9999_9997_FA04);
const LG3: f64 = f64::from_bits(0x3FD2_4924_9422_9359);
const LG4: f64 = f64::from_bits(0x3FCC_71C5_1D8E_78AF);
const LG5: f64 = f64::from_bits(0x3FC7_4664_96CB_03DE);
const LG6: f64 = f64::from_bits(0x3FC3_9A09_D078_C69F);
const LG7: f64 = f64::from_bits(0x3FC2_F112_DF3E_5244);

/// The main path of [`log`](super::log), and whether the lane needs the scalar.
#[inline(always)]
fn log_lane(x: f64) -> (f64, bool) {
    let bits = x.to_bits();
    let hx0 = (bits >> 32) as i32;
    let off = !(0x0010_0000..0x7ff0_0000).contains(&hx0);
    let hx = hx0 & 0x000f_ffff;
    let i = (hx + 0x95f64) & 0x10_0000;
    let xn =
        f64::from_bits((u64::from((hx | (i ^ 0x3ff0_0000)) as u32) << 32) | (bits & 0xffff_ffff));
    let k = (hx0 >> 20) - 1023 + (i >> 20);
    let f = xn - 1.0;
    let tiny = (0x000f_ffff & (2 + hx)) < 3;
    let s = f / (2.0 + f);
    let dk = f64::from(k);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let ii = (hx - 0x6147a) | (0x6b851 - hx);
    let r = t2 + t1;
    let hfsq = 0.5 * f * f;
    let v = if ii > 0 {
        if k == 0 {
            f - (hfsq - s * (hfsq + r))
        } else {
            dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
        }
    } else if k == 0 {
        f - s * (f - r)
    } else {
        dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
    };
    (v, off | tiny)
}

/// Lanes per pass: the flags of one pass stay on the stack.
const PASS: usize = 256;

/// `out[i] = scalar(xs[i])` for each `i` below the shorter length: `lane`
/// over every element first, then `scalar` again where `lane` flagged it.
#[inline(always)]
fn lanes(xs: &[f64], out: &mut [f64], lane: impl Fn(f64) -> (f64, bool), scalar: fn(f64) -> f64) {
    let n = xs.len().min(out.len());
    let (xs, out) = (&xs[..n], &mut out[..n]);
    for (xc, oc) in xs.chunks(PASS).zip(out.chunks_mut(PASS)) {
        let mut sp = [0u8; PASS];
        let mut any = 0u8;
        for ((o, &x), f) in oc.iter_mut().zip(xc).zip(sp.iter_mut()) {
            let (v, s) = lane(x);
            *o = v;
            *f = s as u8;
            any |= s as u8;
        }
        if any != 0 {
            for ((o, &x), &f) in oc.iter_mut().zip(xc).zip(&sp) {
                if f != 0 {
                    *o = scalar(x);
                }
            }
        }
    }
}

/// `out[i] = log(xs[i])`, bit for bit, for each `i` below the shorter length.
pub fn log_slice(xs: &[f64], out: &mut [f64]) {
    lanes(xs, out, log_lane, super::log);
}

/// fdlibm `__kernel_sin(x, y, 1)`.
#[inline(always)]
fn ksin(x: f64, y: f64) -> f64 {
    let ix = hi(x) & 0x7fff_ffff;
    let z = x * x;
    let v = z * x;
    let r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    let s = x - ((z * (0.5 * y - v * r) - y) - v * S1);
    if ix < 0x3e40_0000 {
        x
    } else {
        s
    }
}

/// fdlibm `__kernel_sin(x, 0, 0)`, the form `sin` uses on `[-pi/4, pi/4]`.
#[inline(always)]
fn ksin0(x: f64) -> f64 {
    let ix = hi(x) & 0x7fff_ffff;
    let z = x * x;
    let v = z * x;
    let r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    let s = x + v * (S1 + z * r);
    if ix < 0x3e40_0000 {
        x
    } else {
        s
    }
}

/// fdlibm `__kernel_cos(x, y)`.
#[inline(always)]
fn kcos(x: f64, y: f64) -> f64 {
    let ix = hi(x) & 0x7fff_ffff;
    let z = x * x;
    let r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    let qx = if ix > 0x3fe9_0000 {
        0.28125
    } else {
        f64::from_bits(u64::from((ix - 0x0020_0000) as u32) << 32)
    };
    let small = 1.0 - (0.5 * z - (z * r - x * y));
    let big = (1.0 - qx) - ((0.5 * z - qx) - (z * r - x * y));
    if ix < 0x3e40_0000 {
        1.0
    } else if ix < 0x3FD3_3333 {
        small
    } else {
        big
    }
}

/// fdlibm `rem_pio2` as `(n, y0, y1, needs scalar)`. The medium-size
/// reduction gives the same bits as the `|x| < 3pi/4` special case except at
/// `|x|`'s high word `0x3ff921fb`; next to a multiple of pi/2 below `32 pi/2`
/// the scalar's refinement depends on a table, so those lanes go scalar too.
#[inline(always)]
fn reduce(x: f64) -> (i32, f64, f64, bool) {
    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    let t = x.abs();
    // Clamped so the conversion is defined for every lane; the lanes the
    // clamp changes (huge, infinite, NaN) take the scalar path.
    let q = t * INVPIO2 + 0.5;
    let q = if q < 1.0e9 { q } else { 1.0e9 };
    // SAFETY: `q` is finite and in [0, 1e9]
    let n: i32 = unsafe { q.to_int_unchecked() };
    let f_n = f64::from(n);
    let r = t - f_n * PIO2_1;
    let w = f_n * PIO2_1T;
    let y = r - w;
    let j = ix >> 20;
    let i = j - ((hi(y) >> 20) & 0x7ff);
    let w2 = f_n * PIO2_2;
    let r2 = r - w2;
    let w2 = f_n * PIO2_2T - ((r - r2) - w2);
    let y2 = r2 - w2;
    let i2 = j - ((hi(y2) >> 20) & 0x7ff);
    let w3 = f_n * PIO2_3;
    let r3 = r2 - w3;
    let w3 = f_n * PIO2_3T - ((r2 - r3) - w3);
    let y3 = r3 - w3;
    let (rr, ww, yy) = if n >= 32 && i > 16 {
        if i2 > 49 {
            (r3, w3, y3)
        } else {
            (r2, w2, y2)
        }
    } else {
        (r, w, y)
    };
    let y1 = (rr - yy) - ww;
    let small = ix <= 0x3fe9_21fb;
    let special = ix > 0x4139_21fb || ix == 0x3ff9_21fb || (n < 32 && i > 16 && !small);
    let (n, y0, y1) = if small {
        (0, x, 0.0)
    } else if hx < 0 {
        (-n, -yy, -y1)
    } else {
        (n, yy, y1)
    };
    (n, y0, y1, special)
}

#[inline(always)]
fn cos_lane(x: f64) -> (f64, bool) {
    let (n, y0, y1, sp) = reduce(x);
    let c = kcos(y0, y1);
    let s = ksin(y0, y1);
    let q = n & 3;
    let v = if q & 1 == 0 { c } else { s };
    let v = if (q + 1) & 2 != 0 { -v } else { v };
    (v, sp)
}
#[inline(always)]
fn sin_lane(x: f64) -> (f64, bool) {
    let (n, y0, y1, sp) = reduce(x);
    let c = kcos(y0, y1);
    let s = if hi(x) & 0x7fff_ffff <= 0x3fe9_21fb {
        ksin0(y0)
    } else {
        ksin(y0, y1)
    };
    let q = n & 3;
    let v = if q & 1 == 0 { s } else { c };
    let v = if q & 2 != 0 { -v } else { v };
    (v, sp)
}

/// `out[i] = cos(xs[i])`, bit for bit, for each `i` below the shorter length.
pub fn cos_slice(xs: &[f64], out: &mut [f64]) {
    lanes(xs, out, cos_lane, super::cos);
}
/// `out[i] = sin(xs[i])`, bit for bit, for each `i` below the shorter length.
pub fn sin_slice(xs: &[f64], out: &mut [f64]) {
    lanes(xs, out, sin_lane, super::sin);
}

const INVLN2: f64 = f64::from_bits(0x3FF7_1547_652B_82FE);
const P1: f64 = f64::from_bits(0x3FC5_5555_5555_553E);
const P2: f64 = f64::from_bits(0xBF66_C16C_16BE_BD93);
const P3: f64 = f64::from_bits(0x3F11_566A_AF25_DE2C);
const P4: f64 = f64::from_bits(0xBEBB_BD41_C5D2_6BF1);
const P5: f64 = f64::from_bits(0x3E66_3769_72BE_A4D0);

/// The main path of [`exp`](super::exp), and whether the lane needs the scalar.
#[inline(always)]
fn exp_lane(x: f64) -> (f64, bool) {
    let hx = hi(x);
    let neg = hx < 0;
    let hx = hx & 0x7fff_ffff;
    let q = INVLN2 * x + if neg { -0.5 } else { 0.5 };
    let q = if q > -2000.0 { q } else { -2000.0 };
    let q = if q < 2000.0 { q } else { 2000.0 };
    // SAFETY: `q` is finite and in [-2000, 2000]
    let kq: i32 = unsafe { q.to_int_unchecked() };
    let near = hx < 0x3FF0_A2B2;
    let k1 = if neg { -1 } else { 1 };
    let k = if near { k1 } else { kq };
    let tk = f64::from(k);
    let (lnhi, lnlo) = if neg && near {
        (-LN2_HI, -LN2_LO)
    } else {
        (LN2_HI, LN2_LO)
    };
    let hi_r = if near { x - lnhi } else { x - tk * LN2_HI };
    let lo_r = if near { lnlo } else { tk * LN2_LO };
    let reduced = hx > 0x3fd6_2e42;
    let k = if reduced { k } else { 0 };
    let (hi_r, lo_r) = if reduced { (hi_r, lo_r) } else { (0.0, 0.0) };
    let xr = if reduced { hi_r - lo_r } else { x };
    let t = xr * xr;
    let c = xr - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    let v0 = 1.0 - ((xr * c) / (c - 2.0) - xr);
    let y = 1.0 - ((lo_r - (xr * c) / (2.0 - c)) - hi_r);
    let yk = f64::from_bits(y.to_bits().wrapping_add((i64::from(k) as u64) << 52));
    let v = if k == 0 { v0 } else { yk };
    let v = if hx < 0x3e30_0000 { 1.0 + x } else { v };
    (v, hx >= 0x4086_2E42 || k < -1021)
}

/// `out[i] = exp(xs[i])`, bit for bit, for each `i` below the shorter length.
pub fn exp_slice(xs: &[f64], out: &mut [f64]) {
    lanes(xs, out, exp_lane, super::exp);
}

#[cfg(test)]
mod tests {
    type Pair = (fn(f64) -> f64, fn(&[f64], &mut [f64]));

    const FUNCS: [(&str, Pair); 4] = [
        ("log", (super::super::log, super::log_slice)),
        ("exp", (super::super::exp, super::exp_slice)),
        ("sin", (super::super::sin, super::sin_slice)),
        ("cos", (super::super::cos, super::cos_slice)),
    ];

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn inputs() -> Vec<f64> {
        let mut r = Rng(0x9E37_79B9_7F4A_7C15);
        let mut xs = Vec::new();
        for _ in 0..20_000 {
            xs.push(f64::from_bits(r.next()));
            let e = (r.next() % 90) as i32 - 45;
            let sign = if r.next() & 1 == 0 { 1.0 } else { -1.0 };
            xs.push(sign * r.unit() * 2f64.powi(e));
            xs.push((r.unit() - 0.5) * 2000.0);
            let k = (r.next() % 70) as f64;
            let near = k * std::f64::consts::FRAC_PI_2 * (1.0 + (r.unit() - 0.5) * 1e-9);
            xs.push(f64::from_bits(near.to_bits() ^ (r.next() & 0xff)));
        }
        let specials = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::MIN_POSITIVE,
            f64::MIN_POSITIVE / 3.0,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            709.78,
            -745.1,
            -708.5,
            1e-300,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_4,
            3.0 * std::f64::consts::FRAC_PI_4,
            843_314_856.0,
            843_314_857.0,
        ];
        for x in specials {
            xs.extend([x, -x, f64::from_bits(x.to_bits() + 1)]);
        }
        xs
    }

    #[test]
    fn slices_match_the_scalar_bits() {
        let xs = inputs();
        let mut out = vec![0.0; xs.len()];
        for (name, (scalar, slice)) in FUNCS {
            slice(&xs, &mut out);
            for (x, got) in xs.iter().zip(&out) {
                let want = scalar(*x);
                assert_eq!(
                    want.to_bits(),
                    got.to_bits(),
                    "{name}({x:e}) = {want:e}, slice {got:e}"
                );
            }
        }
    }

    #[test]
    fn slices_stop_at_the_shorter_length() {
        let xs = [0.5, 1.5, 2.5];
        let mut out = [7.0; 5];
        super::log_slice(&xs, &mut out);
        assert_eq!(out[3..], [7.0, 7.0]);
        let mut short = [0.0; 2];
        super::cos_slice(&xs, &mut short);
        assert_eq!(short, [super::super::cos(0.5), super::super::cos(1.5)]);
    }
}
