//! `java.lang.StrictMath`'s `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`,
//! `hypot`, `exp`, `log`, `pow`, `expm1`, `cosh`, `sinh`, `tanh` and `log10`: the fdlibm
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
//! `sin`/`cos`/`tan` reduce arguments up to `2^19 * pi/2` with fdlibm's
//! medium-size Cody-Waite path and larger ones with its Payne-Hanek
//! `__kernel_rem_pio2`, as `java.lang.FdLibm` does.
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

/// `2/pi` in 24-bit chunks (fdlibm's `two_over_pi`, `ipio2`).
const TWO_OVER_PI: [i32; 66] = [
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163,
    0xABDEBB, 0xC561B7, 0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
    0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C,
    0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
    0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7, 0x8A5292,
    0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
    0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA,
    0x73A8C9, 0x60E27B, 0xC08C6B,
];

/// `pi/2` in 24-bit chunks (fdlibm's `PIo2`).
const PIO2_CHUNKS: [f64; 8] = [
    f64::from_bits(0x3FF9_21FB_4000_0000),
    f64::from_bits(0x3E74_442D_0000_0000),
    f64::from_bits(0x3CF8_4698_8000_0000),
    f64::from_bits(0x3B78_CC51_6000_0000),
    f64::from_bits(0x39F0_1B83_8000_0000),
    f64::from_bits(0x387A_2520_4000_0000),
    f64::from_bits(0x36E3_8222_8000_0000),
    f64::from_bits(0x3569_F31D_0000_0000),
];

/// `2^n` for the small exponents `__kernel_rem_pio2` scales by (`scalbn(1, n)`).
fn pow2(n: i32) -> f64 {
    f64::from_bits(((n + 1023) as u64) << 52)
}

/// fdlibm `__kernel_rem_pio2` for double precision (`prec == 2`): the
/// Payne-Hanek reduction of `x[0..nx]` (24-bit chunks scaled by `2^e0`)
/// by `pi/2`. Returns `n mod 8` and `(y0, y1)`.
fn kernel_rem_pio2(x: &[f64; 3], e0: i32, nx: usize) -> (i32, f64, f64) {
    const TWO24: f64 = f64::from_bits(0x4170_0000_0000_0000);
    const TWON24: f64 = f64::from_bits(0x3E70_0000_0000_0000);
    const JK: usize = 4; // init_jk[2]
    let jx = nx - 1;
    let mut jv = (e0 - 3) / 24;
    if jv < 0 {
        jv = 0;
    }
    let mut q0 = e0 - 24 * (jv + 1);
    let jv = jv as usize;

    // set up f[0] to f[jx+jk] where f[jx+jk] = ipio2[jv+jk]
    let mut f = [0.0f64; 32];
    for (i, fi) in f.iter_mut().enumerate().take(jx + JK + 1) {
        // `j = jv - jx + i`; chunks before the table's start are zero
        *fi = (jv + i)
            .checked_sub(jx)
            .map_or(0.0, |j| f64::from(TWO_OVER_PI[j]));
    }
    // compute q[0], q[1], ..., q[jk]
    let mut q = [0.0f64; 32];
    for i in 0..=JK {
        let mut fw = 0.0;
        for j in 0..=jx {
            fw += x[j] * f[jx + i - j];
        }
        q[i] = fw;
    }

    let mut jz = JK;
    let mut iq = [0i32; 32];
    let n;
    let ih;
    let mut z;
    loop {
        // distill q[] into iq[] reversingly
        let mut i = 0;
        let mut j = jz;
        z = q[jz];
        while j > 0 {
            let fw = f64::from((TWON24 * z) as i32);
            iq[i] = (z - TWO24 * fw) as i32;
            z = q[j - 1] + fw;
            i += 1;
            j -= 1;
        }
        // compute n
        z *= pow2(q0); // actual value of z
        z -= 8.0 * (z * 0.125).floor(); // trim off integer >= 8
        let mut nn = z as i32;
        z -= f64::from(nn);
        let mut ihh = 0;
        if q0 > 0 {
            // need iq[jz-1] to determine n
            let i = iq[jz - 1] >> (24 - q0);
            nn += i;
            iq[jz - 1] -= i << (24 - q0);
            ihh = iq[jz - 1] >> (23 - q0);
        } else if q0 == 0 {
            ihh = iq[jz - 1] >> 23;
        } else if z >= 0.5 {
            ihh = 2;
        }
        if ihh > 0 {
            // q > 0.5
            nn += 1;
            let mut carry = 0;
            for v in iq.iter_mut().take(jz) {
                // compute 1-q
                let j = *v;
                if carry == 0 {
                    if j != 0 {
                        carry = 1;
                        *v = 0x100_0000 - j;
                    }
                } else {
                    *v = 0xff_ffff - j;
                }
            }
            if q0 > 0 {
                // rare case: chance is 1 in 12
                match q0 {
                    1 => iq[jz - 1] &= 0x7f_ffff,
                    2 => iq[jz - 1] &= 0x3f_ffff,
                    _ => {}
                }
            }
            if ihh == 2 {
                z = 1.0 - z;
                if carry != 0 {
                    z -= pow2(q0);
                }
            }
        }
        // check if recomputation is needed
        if z == 0.0 {
            let mut j = 0;
            for i in (JK..jz).rev() {
                j |= iq[i];
            }
            if j == 0 {
                // need recomputation
                let mut k = 1;
                while iq[JK - k] == 0 {
                    k += 1; // k = no. of terms needed
                }
                for i in jz + 1..=jz + k {
                    // add q[jz+1] to q[jz+k]
                    f[jx + i] = f64::from(TWO_OVER_PI[jv + i]);
                    let mut fw = 0.0;
                    for j in 0..=jx {
                        fw += x[j] * f[jx + i - j];
                    }
                    q[i] = fw;
                }
                jz += k;
                continue;
            }
        }
        n = nn;
        ih = ihh;
        break;
    }

    // chop off zero terms
    if z == 0.0 {
        jz -= 1;
        q0 -= 24;
        while iq[jz] == 0 {
            jz -= 1;
            q0 -= 24;
        }
    } else {
        // break z into 24-bit if necessary
        z *= pow2(-q0);
        if z >= TWO24 {
            let fw = f64::from((TWON24 * z) as i32);
            iq[jz] = (z - TWO24 * fw) as i32;
            jz += 1;
            q0 += 24;
            iq[jz] = fw as i32;
        } else {
            iq[jz] = z as i32;
        }
    }

    // convert integer "bit" chunk to floating-point value
    let mut fw = pow2(q0);
    for i in (0..=jz).rev() {
        q[i] = fw * f64::from(iq[i]);
        fw *= TWON24;
    }
    // compute PIo2[0..jp] * q[jz..0]
    let mut fq = [0.0f64; 32];
    for i in (0..=jz).rev() {
        let mut fw = 0.0;
        let mut k = 0;
        while k <= JK && k <= jz - i {
            fw += PIO2_CHUNKS[k] * q[i + k];
            k += 1;
        }
        fq[jz - i] = fw;
    }
    // compress fq[] into y[]
    let mut fw = 0.0;
    for i in (0..=jz).rev() {
        fw += fq[i];
    }
    let y0 = if ih == 0 { fw } else { -fw };
    let mut fw = fq[0] - fw;
    for v in fq.iter().take(jz + 1).skip(1) {
        fw += v;
    }
    let y1 = if ih == 0 { fw } else { -fw };
    (n & 7, y0, y1)
}

/// The large-argument branch of `__ieee754_rem_pio2` (`|x| > 2^19 * pi/2`,
/// finite): reduce with Payne-Hanek.
fn rem_pio2_large(x: f64) -> (i32, f64, f64) {
    const TWO24: f64 = f64::from_bits(0x4170_0000_0000_0000);
    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    // set z = scalbn(|x|, ilogb(x) - 23)
    let e0 = (ix >> 20) - 1046; // e0 = ilogb(z) - 23
    let mut z = f64::from_bits((u64::from((ix - (e0 << 20)) as u32) << 32) | u64::from(lo(x)));
    let mut tx = [0.0f64; 3];
    for t in tx.iter_mut().take(2) {
        *t = f64::from(z as i32);
        z = (z - *t) * TWO24;
    }
    tx[2] = z;
    let mut nx = 3;
    while tx[nx - 1] == 0.0 {
        nx -= 1; // skip zero term
    }
    let (n, y0, y1) = kernel_rem_pio2(&tx, e0, nx);
    if hx < 0 {
        (-n, -y0, -y1)
    } else {
        (n, y0, y1)
    }
}

/// fdlibm `__ieee754_rem_pio2` for finite `x`: returns `n` and
/// `x - n*pi/2` as the pair `(y0, y1)`.
fn rem_pio2(x: f64) -> (i32, f64, f64) {
    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        return (0, x, 0.0);
    }
    if ix < 0x4002_d97c {
        // |x| < 3pi/4, special case with n = +-1.
        return if hx > 0 {
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
        };
    }
    if ix > 0x4139_21fb {
        return rem_pio2_large(x);
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
        (-n, -y0, -y1)
    } else {
        (n, y0, y1)
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
    let (n, y0, y1) = rem_pio2(x);
    match n & 3 {
        0 => kernel_sin(y0, y1, 1),
        1 => kernel_cos(y0, y1),
        2 => -kernel_sin(y0, y1, 1),
        _ => -kernel_cos(y0, y1),
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
    let (n, y0, y1) = rem_pio2(x);
    match n & 3 {
        0 => kernel_cos(y0, y1),
        1 => -kernel_sin(y0, y1, 1),
        2 => -kernel_cos(y0, y1),
        _ => kernel_sin(y0, y1, 1),
    }
}

/// `(StrictMath.sin(x), StrictMath.cos(x))`, bit for bit, with the argument
/// reduced once: [`sin`] and [`cos`] run the same `rem_pio2` on the same
/// `x`, so sharing it changes no result (geo3d builds every point from a
/// latitude's and a longitude's sine and cosine).
pub fn sin_cos(x: f64) -> (f64, f64) {
    let ix = hi(x) & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        return (kernel_sin(x, 0.0, 0), kernel_cos(x, 0.0));
    }
    if ix >= 0x7ff0_0000 {
        return (x - x, x - x);
    }
    let (n, y0, y1) = rem_pio2(x);
    let (s, c) = (kernel_sin(y0, y1, 1), kernel_cos(y0, y1));
    match n & 3 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
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

/// `x` with its high word replaced by `hi` and its low word zero
/// (`__HI(0.0, hi)`).
#[inline]
fn from_hi(hi: i32) -> f64 {
    f64::from_bits(u64::from(hi as u32) << 32)
}

/// `StrictMath.hypot` (`FdLibm.Hypot`, fdlibm `e_hypot.c`); `Math.hypot`
/// delegates to it. `sqrt(x^2 + y^2)` without undue overflow or underflow,
/// within an ulp -- and, being fdlibm, not always the correctly rounded
/// result a platform `hypot` returns, which is why `ShapeDocValues`' line
/// lengths need this one.
pub fn hypot(x: f64, y: f64) -> f64 {
    const TWO_MINUS_600: f64 = f64::from_bits(0x1A70_0000_0000_0000);
    const TWO_PLUS_600: f64 = f64::from_bits(0x6570_0000_0000_0000);
    // `0x1.00000_ffff_ffffp500`.
    const NEAR_TWO_500: f64 = f64::from_bits(0x5F30_0000_FFFF_FFFF);
    const TWO_MINUS_500: f64 = f64::from_bits(0x20B0_0000_0000_0000);
    const TWO_1022: f64 = f64::from_bits(0x7FD0_0000_0000_0000);
    let mut a = x.abs();
    let mut b = y.abs();
    if !a.is_finite() || !b.is_finite() {
        if a == f64::INFINITY || b == f64::INFINITY {
            return f64::INFINITY;
        }
        // Propagate NaN significand bits.
        return a + b;
    }
    if b > a {
        std::mem::swap(&mut a, &mut b);
    }
    let mut ha = hi(a);
    let mut hb = hi(b);
    if ha.wrapping_sub(hb) > 0x03c0_0000 {
        // x / y > 2**60
        return a + b;
    }
    let mut k: i32 = 0;
    if a > NEAR_TWO_500 {
        // scale a and b by 2**-600
        ha = ha.wrapping_sub(0x2580_0000);
        hb = hb.wrapping_sub(0x2580_0000);
        a *= TWO_MINUS_600;
        b *= TWO_MINUS_600;
        k += 600;
    }
    if b < TWO_MINUS_500 {
        if b < f64::MIN_POSITIVE {
            // subnormal b or 0
            if b == 0.0 {
                return a;
            }
            b *= TWO_1022;
            a *= TWO_1022;
            k -= 1022;
        } else {
            // scale a and b by 2^600
            ha = ha.wrapping_add(0x2580_0000);
            hb = hb.wrapping_add(0x2580_0000);
            a *= TWO_PLUS_600;
            b *= TWO_PLUS_600;
            k -= 600;
        }
    }
    // medium size a and b
    let mut w = a - b;
    if w > b {
        let t1 = from_hi(ha);
        let t2 = a - t1;
        w = (t1 * t1 - (b * (-b) - t2 * (a + t1))).sqrt();
    } else {
        a = a + a;
        let y1 = from_hi(hb);
        let y2 = b - y1;
        let t1 = from_hi(ha.wrapping_add(0x0010_0000));
        let t2 = a - t1;
        w = (t1 * y1 - (w * (-w) - (t1 * y2 + t2 * b))).sqrt();
    }
    if k != 0 {
        // `Math.powerOfTwoD(k)`: k is one of 600, -600, -1022 -- a normal
        // power of two. Both scalings (600 then -600 or -1022) would need
        // `a > 2^500` with `b` below `2^100`, a ratio past the `2^60` that
        // already returned `a + b`.
        f64::from_bits(((i64::from(k) + 1023) as u64) << 52) * w
    } else {
        w
    }
}

/// fdlibm `__kernel_tan`'s coefficients (`T[0..12]`), then `pio4`/`pio4lo`.
const TAN_T: [f64; 13] = [
    f64::from_bits(0x3FD5_5555_5555_5563),
    f64::from_bits(0x3FC1_1111_1110_FE7A),
    f64::from_bits(0x3FAB_A1BA_1BB3_41FE),
    f64::from_bits(0x3F96_64F4_8406_D637),
    f64::from_bits(0x3F82_26E3_E96E_8493),
    f64::from_bits(0x3F6D_6D22_C956_0328),
    f64::from_bits(0x3F57_DBC8_FEE0_8315),
    f64::from_bits(0x3F43_44D8_F2F2_6501),
    f64::from_bits(0x3F30_26F7_1A8D_1068),
    f64::from_bits(0x3F14_7E88_A037_92A6),
    f64::from_bits(0x3F12_B80F_32F0_A7E9),
    f64::from_bits(0xBEF3_75CB_DB60_5373),
    f64::from_bits(0x3EFB_2A70_74BF_7AD4),
];
const PIO4_LO: f64 = f64::from_bits(0x3C81_A626_3314_5C07);

/// `x` with its low word replaced by zero, `__LO(x) = 0`.
#[inline]
fn lo_zero(x: f64) -> f64 {
    with_lo_zero(x)
}

/// fdlibm `__kernel_tan(x, y, iy)` on `[-pi/4, pi/4]` (`FdLibm.Tan.kernelTan`):
/// `tan(x + y)` when `iy == 1`, `-1 / tan(x + y)` when `iy == -1`.
fn kernel_tan(mut x: f64, mut y: f64, iy: i32) -> f64 {
    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    if ix < 0x3e30_0000 && (x as i32) == 0 {
        // |x| < 2^-28
        if ((ix as u32 | lo(x)) | (iy + 1) as u32) == 0 {
            return 1.0 / x.abs();
        } else if iy == 1 {
            return x;
        } else {
            // compute -1 / (x + y) carefully
            let w = x + y;
            let z = lo_zero(w);
            let v = y - (z - x);
            let a = -1.0 / w;
            let t = lo_zero(a);
            let s = 1.0 + t * z;
            return t + a * (s + t * v);
        }
    }
    if ix >= 0x3FE5_9428 {
        // |x| >= 0.6744
        if hx < 0 {
            x = -x;
            y = -y;
        }
        let z = PIO4_HI - x;
        let w = PIO4_LO - y;
        x = z + w;
        y = 0.0;
    }
    let t = &TAN_T;
    let z = x * x;
    let w = z * z;
    let r = t[1] + w * (t[3] + w * (t[5] + w * (t[7] + w * (t[9] + w * t[11]))));
    let v = z * (t[2] + w * (t[4] + w * (t[6] + w * (t[8] + w * (t[10] + w * t[12])))));
    let s = z * x;
    let mut r = y + z * (s * (r + v) + y);
    r += t[0] * s;
    let w = x + r;
    if ix >= 0x3FE5_9428 {
        let v = f64::from(iy);
        return f64::from(1 - ((hx >> 30) & 2)) * (v - 2.0 * (x - (w * w / (w + v) - r)));
    }
    if iy == 1 {
        w
    } else {
        // compute -1.0 / (x + r) accurately
        let z = lo_zero(w);
        let v = r - (z - x);
        let a = -1.0 / w;
        let t = lo_zero(a);
        let s = 1.0 + t * z;
        t + a * (s + t * v)
    }
}

/// `StrictMath.tan` (fdlibm `s_tan.c`). `Math.tan` is a HotSpot intrinsic
/// on x86-64 whose last bit can differ; see `spatial3d`'s module doc.
pub fn tan(x: f64) -> f64 {
    let ix = hi(x) & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        return kernel_tan(x, 0.0, 1);
    }
    if ix >= 0x7ff0_0000 {
        return x - x;
    }
    let (n, y0, y1) = rem_pio2(x);
    kernel_tan(y0, y1, 1 - ((n & 1) << 1))
}

const ATAN_HI: [f64; 4] = [
    f64::from_bits(0x3FDD_AC67_0561_BB4F),
    f64::from_bits(0x3FE9_21FB_5444_2D18),
    f64::from_bits(0x3FEF_730B_D281_F69B),
    f64::from_bits(0x3FF9_21FB_5444_2D18),
];
const ATAN_LO: [f64; 4] = [
    f64::from_bits(0x3C7A_2B7F_222F_65E2),
    f64::from_bits(0x3C81_A626_3314_5C07),
    f64::from_bits(0x3C70_0788_7AF0_CBBD),
    f64::from_bits(0x3C91_A626_3314_5C07),
];
const AT: [f64; 11] = [
    f64::from_bits(0x3FD5_5555_5555_550D),
    f64::from_bits(0xBFC9_9999_9998_EBC4),
    f64::from_bits(0x3FC2_4924_9200_83FF),
    f64::from_bits(0xBFBC_71C6_FE23_1671),
    f64::from_bits(0x3FB7_45CD_C54C_206E),
    f64::from_bits(0xBFB3_B0F2_AF74_9A6D),
    f64::from_bits(0x3FB1_0D66_A0D0_3D51),
    f64::from_bits(0xBFAD_DE2D_52DE_FD9A),
    f64::from_bits(0x3FA9_7B4B_2476_0DEB),
    f64::from_bits(0xBFA2_B444_2C6A_6C2F),
    f64::from_bits(0x3F90_AD3A_E322_DA11),
];

/// `StrictMath.atan` (fdlibm `s_atan.c`); `Math.atan` delegates to it.
pub fn atan(x: f64) -> f64 {
    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    let mut x = x;
    let id: i32;
    if ix >= 0x4410_0000 {
        // |x| >= 2^66
        if ix > 0x7ff0_0000 || (ix == 0x7ff0_0000 && lo(x) != 0) {
            return x + x; // NaN
        }
        return if hx > 0 {
            ATAN_HI[3] + ATAN_LO[3]
        } else {
            -ATAN_HI[3] - ATAN_LO[3]
        };
    }
    if ix < 0x3fdc_0000 {
        // |x| < 0.4375
        if ix < 0x3e20_0000 {
            // |x| < 2^-29; fdlibm's `huge + x > one` is always true
            return x;
        }
        id = -1;
    } else {
        x = x.abs();
        if ix < 0x3ff3_0000 {
            // |x| < 1.1875
            if ix < 0x3fe6_0000 {
                // 7/16 <= |x| < 11/16
                id = 0;
                x = (2.0 * x - 1.0) / (2.0 + x);
            } else {
                // 11/16 <= |x| < 19/16
                id = 1;
                x = (x - 1.0) / (x + 1.0);
            }
        } else if ix < 0x4003_8000 {
            // |x| < 2.4375
            id = 2;
            x = (x - 1.5) / (1.0 + 1.5 * x);
        } else {
            // 2.4375 <= |x| < 2^66
            id = 3;
            x = -1.0 / x;
        }
    }
    let z = x * x;
    let w = z * z;
    let s1 = z * (AT[0] + w * (AT[2] + w * (AT[4] + w * (AT[6] + w * (AT[8] + w * AT[10])))));
    let s2 = w * (AT[1] + w * (AT[3] + w * (AT[5] + w * (AT[7] + w * AT[9]))));
    if id < 0 {
        x - x * (s1 + s2)
    } else {
        let i = id as usize;
        let z = ATAN_HI[i] - ((x * (s1 + s2) - ATAN_LO[i]) - x);
        if hx < 0 {
            -z
        } else {
            z
        }
    }
}

/// `StrictMath.atan2` (fdlibm `e_atan2.c`); `Math.atan2` delegates to it.
pub fn atan2(y: f64, x: f64) -> f64 {
    const TINY: f64 = 1.0e-300;
    const PI_O_4: f64 = f64::from_bits(0x3FE9_21FB_5444_2D18);
    const PI_O_2: f64 = f64::from_bits(0x3FF9_21FB_5444_2D18);
    const PI_LO: f64 = f64::from_bits(0x3CA1_A626_3314_5C07);
    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    let lx = lo(x);
    let hy = hi(y);
    let iy = hy & 0x7fff_ffff;
    let ly = lo(y);
    if (ix as u32 | ((lx | lx.wrapping_neg()) >> 31)) > 0x7ff0_0000
        || (iy as u32 | ((ly | ly.wrapping_neg()) >> 31)) > 0x7ff0_0000
    {
        return x + y; // x or y is NaN
    }
    if (hx.wrapping_sub(0x3ff0_0000) as u32 | lx) == 0 {
        return atan(y); // x = 1.0
    }
    let m = ((hy >> 31) & 1) | ((hx >> 30) & 2); // 2 * sign(x) + sign(y)
    if (iy as u32 | ly) == 0 {
        // y = 0
        return match m {
            0 | 1 => y,
            2 => PI + TINY,
            _ => -PI - TINY,
        };
    }
    if (ix as u32 | lx) == 0 {
        // x = 0
        return if hy < 0 {
            -PI_O_2 - TINY
        } else {
            PI_O_2 + TINY
        };
    }
    if ix == 0x7ff0_0000 {
        // x is INF
        if iy == 0x7ff0_0000 {
            return match m {
                0 => PI_O_4 + TINY,
                1 => -PI_O_4 - TINY,
                2 => 3.0 * PI_O_4 + TINY,
                _ => -3.0 * PI_O_4 - TINY,
            };
        }
        return match m {
            0 => 0.0,
            1 => -0.0,
            2 => PI + TINY,
            _ => -PI - TINY,
        };
    }
    if iy == 0x7ff0_0000 {
        // y is INF
        return if hy < 0 {
            -PI_O_2 - TINY
        } else {
            PI_O_2 + TINY
        };
    }
    // compute y / x
    let k = (iy - ix) >> 20;
    let z = if k > 60 {
        // |y / x| > 2^60. JDK's FdLibm.Atan2 leaves `m` alone here (later
        // fdlibm and musl add `m &= 1`), so x < 0 still takes the
        // `pi - (z - pi_lo)` arm below: one ulp above pi/2, as the
        // `GenGeo3dMath` atan2 specials pin.
        PI_O_2 + 0.5 * PI_LO
    } else if hx < 0 && k < -60 {
        0.0 // |y| / x < -2^60
    } else {
        atan((y / x).abs())
    };
    match m {
        0 => z,
        1 => -z, // `__HI(z) ^= 0x80000000`
        2 => PI - (z - PI_LO),
        _ => (z - PI_LO) - PI,
    }
}

/// `StrictMath.exp` (`FdLibm.Exp`, fdlibm `e_exp.c`): OpenNLP's maxent and
/// perceptron models normalise their outcome scores with it, and the beam
/// search ranks sequences on the results, so a last-bit difference from the
/// platform `exp` can change a tag.
pub fn exp(x: f64) -> f64 {
    const HALF: [f64; 2] = [0.5, -0.5];
    const HUGE: f64 = f64::from_bits(0x7E37_E43C_8800_759C);
    const TWOM1000: f64 = f64::from_bits(0x0170_0000_0000_0000);
    const O_THRESHOLD: f64 = f64::from_bits(0x4086_2E42_FEFA_39EF);
    const U_THRESHOLD: f64 = f64::from_bits(0xC087_4910_D52D_3051);
    const LN2_HI: [f64; 2] = [
        f64::from_bits(0x3FE6_2E42_FEE0_0000),
        f64::from_bits(0xBFE6_2E42_FEE0_0000),
    ];
    const LN2_LO: [f64; 2] = [
        f64::from_bits(0x3DEA_39EF_3579_3C76),
        f64::from_bits(0xBDEA_39EF_3579_3C76),
    ];
    const INVLN2: f64 = f64::from_bits(0x3FF7_1547_652B_82FE);
    const P1: f64 = f64::from_bits(0x3FC5_5555_5555_553E);
    const P2: f64 = f64::from_bits(0xBF66_C16C_16BE_BD93);
    const P3: f64 = f64::from_bits(0x3F11_566A_AF25_DE2C);
    const P4: f64 = f64::from_bits(0xBEBB_BD41_C5D2_6BF1);
    const P5: f64 = f64::from_bits(0x3E66_3769_72BE_A4D0);

    let mut x = x;
    let mut hi_part = 0.0;
    let mut lo_part = 0.0;
    let mut k: i32 = 0;
    let hx = hi(x);
    let xsb = ((hx >> 31) & 1) as usize;
    let hx = hx & 0x7fff_ffff;
    // filter out non-finite argument
    if hx >= 0x4086_2E42 {
        // |x| >= 709.78...
        if hx >= 0x7ff0_0000 {
            if ((hx & 0xf_ffff) as u32 | lo(x)) != 0 {
                return x + x; // NaN
            }
            return if xsb == 0 { x } else { 0.0 }; // exp(+-inf) = {inf, 0}
        }
        if x > O_THRESHOLD {
            return HUGE * HUGE; // overflow
        }
        if x < U_THRESHOLD {
            return TWOM1000 * TWOM1000; // underflow
        }
    }
    // argument reduction
    if hx > 0x3fd6_2e42 {
        // |x| > 0.5 ln2
        if hx < 0x3FF0_A2B2 {
            // and |x| < 1.5 ln2
            hi_part = x - LN2_HI[xsb];
            lo_part = LN2_LO[xsb];
            k = 1 - 2 * xsb as i32;
        } else {
            k = (INVLN2 * x + HALF[xsb]) as i32;
            let t = f64::from(k);
            hi_part = x - t * LN2_HI[0]; // t*ln2HI is exact here
            lo_part = t * LN2_LO[0];
        }
        x = hi_part - lo_part;
    } else if hx < 0x3e30_0000 {
        // |x| < 2**-28
        if HUGE + x > 1.0 {
            return 1.0 + x; // trigger inexact
        }
    }
    // x is now in primary range
    let t = x * x;
    let c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if k == 0 {
        return 1.0 - ((x * c) / (c - 2.0) - x);
    }
    let y = 1.0 - ((lo_part - (x * c) / (2.0 - c)) - hi_part);
    let with_hi = |y: f64, h: i32| f64::from_bits((u64::from(h as u32) << 32) | u64::from(lo(y)));
    if k >= -1021 {
        with_hi(y, hi(y).wrapping_add(k << 20)) // add k to y's exponent
    } else {
        with_hi(y, hi(y).wrapping_add((k + 1000) << 20)) * TWOM1000
    }
}

/// `StrictMath.log` (`FdLibm.Log`, fdlibm `e_log.c`): OpenNLP's uniform
/// prior and its sequence scores (`Sequence`: `score += StrictMath.log(p)`).
pub fn log(x: f64) -> f64 {
    const LN2_HI: f64 = f64::from_bits(0x3FE6_2E42_FEE0_0000);
    const LN2_LO: f64 = f64::from_bits(0x3DEA_39EF_3579_3C76);
    const TWO54: f64 = f64::from_bits(0x4350_0000_0000_0000);
    const LG1: f64 = f64::from_bits(0x3FE5_5555_5555_5593);
    const LG2: f64 = f64::from_bits(0x3FD9_9999_9997_FA04);
    const LG3: f64 = f64::from_bits(0x3FD2_4924_9422_9359);
    const LG4: f64 = f64::from_bits(0x3FCC_71C5_1D8E_78AF);
    const LG5: f64 = f64::from_bits(0x3FC7_4664_96CB_03DE);
    const LG6: f64 = f64::from_bits(0x3FC3_9A09_D078_C69F);
    const LG7: f64 = f64::from_bits(0x3FC2_F112_DF3E_5244);
    const ZERO: f64 = 0.0;

    let mut x = x;
    let mut hx = hi(x);
    let lx = lo(x);
    let mut k: i32 = 0;
    if hx < 0x0010_0000 {
        // x < 2**-1022
        if ((hx & 0x7fff_ffff) as u32 | lx) == 0 {
            return -TWO54 / ZERO; // log(+-0)=-inf
        }
        if hx < 0 {
            return (x - x) / ZERO; // log(-#) = NaN
        }
        k -= 54;
        x *= TWO54; // subnormal number, scale up x
        hx = hi(x);
    }
    if hx >= 0x7ff0_0000 {
        return x + x;
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000f_ffff;
    let i = (hx + 0x95f64) & 0x10_0000;
    // normalize x or x/2
    x = f64::from_bits((u64::from((hx | (i ^ 0x3ff0_0000)) as u32) << 32) | u64::from(lo(x)));
    k += i >> 20;
    let f = x - 1.0;
    if (0x000f_ffff & (2 + hx)) < 3 {
        // |f| < 2**-20
        if f == ZERO {
            if k == 0 {
                return ZERO;
            }
            let dk = f64::from(k);
            return dk * LN2_HI + dk * LN2_LO;
        }
        let r = f * f * (0.5 - f64::from_bits(0x3FD5_5555_5555_5555) * f);
        if k == 0 {
            return f - r;
        }
        let dk = f64::from(k);
        return dk * LN2_HI - ((r - dk * LN2_LO) - f);
    }
    let s = f / (2.0 + f);
    let dk = f64::from(k);
    let z = s * s;
    let mut i = hx - 0x6147a;
    let w = z * z;
    let j = 0x6b851 - hx;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    i |= j;
    let r = t2 + t1;
    if i > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 {
            f - (hfsq - s * (hfsq + r))
        } else {
            dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
        }
    } else if k == 0 {
        f - s * (f - r)
    } else {
        dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
    }
}

/// `StrictMath.expm1` (`FdLibm.Expm1`, fdlibm `s_expm1.c`).
pub fn expm1(x: f64) -> f64 {
    const HUGE: f64 = 1.0e300;
    const TINY: f64 = 1.0e-300;
    const O_THRESHOLD: f64 = f64::from_bits(0x4086_2E42_FEFA_39EF);
    const LN2_HI: f64 = f64::from_bits(0x3FE6_2E42_FEE0_0000);
    const LN2_LO: f64 = f64::from_bits(0x3DEA_39EF_3579_3C76);
    const INVLN2: f64 = f64::from_bits(0x3FF7_1547_652B_82FE);
    const Q1: f64 = f64::from_bits(0xBFA1_1111_1111_10F4);
    const Q2: f64 = f64::from_bits(0x3F5A_01A0_19FE_5585);
    const Q3: f64 = f64::from_bits(0xBF14_CE19_9EAA_DBB7);
    const Q4: f64 = f64::from_bits(0x3ED0_CFCA_86E6_5239);
    const Q5: f64 = f64::from_bits(0xBE8A_FDB7_6E09_C32D);

    let mut x = x;
    let hx = hi(x);
    let xsb = hx < 0;
    let hx = hx & 0x7fff_ffff;
    let mut c = 0.0;
    let k: i32;
    // filter out huge and non-finite argument
    if hx >= 0x4043_687A {
        // |x| >= 56 * ln2
        if hx >= 0x4086_2E42 {
            if hx >= 0x7ff0_0000 {
                if ((hx & 0xf_ffff) as u32 | lo(x)) != 0 {
                    return x + x; // NaN
                }
                return if xsb { -1.0 } else { x }; // exp(+-inf) = {inf, -1}
            }
            if x > O_THRESHOLD {
                return HUGE * HUGE; // overflow
            }
        }
        if xsb && x + TINY < 0.0 {
            return TINY - 1.0; // raise inexact
        }
    }
    // argument reduction
    if hx > 0x3fd6_2e42 {
        // |x| > 0.5 ln2
        let (h, l);
        if hx < 0x3FF0_A2B2 {
            // and |x| < 1.5 ln2
            if !xsb {
                h = x - LN2_HI;
                l = LN2_LO;
                k = 1;
            } else {
                h = x + LN2_HI;
                l = -LN2_LO;
                k = -1;
            }
        } else {
            k = (INVLN2 * x + if xsb { -0.5 } else { 0.5 }) as i32;
            let t = f64::from(k);
            h = x - t * LN2_HI; // t*ln2_hi is exact here
            l = t * LN2_LO;
        }
        x = h - l;
        c = (h - x) - l;
    } else if hx < 0x3c90_0000 {
        // |x| < 2**-54: return x, with inexact raised
        let t = HUGE + x;
        return x - (t - (HUGE + x));
    } else {
        k = 0;
    }
    // x is now in primary range
    let hfx = 0.5 * x;
    let hxs = x * hfx;
    let r1 = 1.0 + hxs * (Q1 + hxs * (Q2 + hxs * (Q3 + hxs * (Q4 + hxs * Q5))));
    let t = 3.0 - r1 * hfx;
    let mut e = hxs * ((r1 - t) / (6.0 - x * t));
    if k == 0 {
        return x - (x * e - hxs); // c is 0
    }
    e = x * (e - c) - c;
    e -= hxs;
    let add_exp = |y: f64, k: i32| {
        f64::from_bits((u64::from(hi(y).wrapping_add(k << 20) as u32) << 32) | u64::from(lo(y)))
    };
    if k == -1 {
        return 0.5 * (x - e) - 0.5;
    }
    if k == 1 {
        if x < -0.25 {
            return -2.0 * (e - (x + 0.5));
        }
        return 1.0 + 2.0 * (x - e);
    }
    if k <= -2 || k > 56 {
        // suffice to return exp(x)-1
        let mut y = 1.0 - (e - x);
        if k == 1024 {
            y = y * 2.0 * f64::from_bits(0x7FE0_0000_0000_0000);
        } else {
            y = add_exp(y, k);
        }
        return y - 1.0;
    }
    if k < 20 {
        let t = f64::from_bits(u64::from((0x3ff0_0000 - (0x20_0000 >> k)) as u32) << 32); // 1 - 2^-k
        let y = t - (e - x);
        add_exp(y, k)
    } else {
        let t = f64::from_bits(u64::from(((0x3ff - k) << 20) as u32) << 32); // 2^-k
        let y = x - (e + t) + 1.0;
        add_exp(y, k)
    }
}

/// `StrictMath.cosh` (`FdLibm.Cosh`, fdlibm `e_cosh.c`).
pub fn cosh(x: f64) -> f64 {
    const HUGE: f64 = 1.0e300;
    let ix = hi(x) & 0x7fff_ffff;
    // x is INF or NaN
    if ix >= 0x7ff0_0000 {
        return x * x;
    }
    let ax = x.abs();
    // |x| in [0, 0.5*ln2]: 1 + expm1(|x|)^2 / (2 * exp(|x|))
    if ix < 0x3fd6_2e43 {
        let t = expm1(ax);
        let w = 1.0 + t;
        if ix < 0x3c80_0000 {
            return w; // cosh(tiny) = 1
        }
        return 1.0 + (t * t) / (w + w);
    }
    // |x| in [0.5*ln2, 22]: (exp(|x|) + 1/exp(|x|)) / 2
    if ix < 0x4036_0000 {
        let t = exp(ax);
        return 0.5 * t + 0.5 / t;
    }
    // |x| in [22, log(maxdouble)]: 0.5 * exp(|x|)
    if ix < 0x4086_2E42 {
        return 0.5 * exp(ax);
    }
    // |x| in [log(maxdouble), overflowthreshold]
    if ix < 0x4086_33CE || (ix == 0x4086_33ce && lo(x) <= 0x8fb9_f87d) {
        let w = exp(0.5 * ax);
        let t = 0.5 * w;
        return t * w;
    }
    // |x| > overflowthreshold
    HUGE * HUGE
}

/// `StrictMath.sinh` (`FdLibm.Sinh`, fdlibm `e_sinh.c`).
pub fn sinh(x: f64) -> f64 {
    const SHUGE: f64 = 1.0e307;
    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    // x is INF or NaN
    if ix >= 0x7ff0_0000 {
        return x + x;
    }
    let h = if hx < 0 { -0.5 } else { 0.5 };
    let ax = x.abs();
    // |x| in [0, 22]: expm1-based
    if ix < 0x4036_0000 {
        if ix < 0x3e30_0000 && SHUGE + x > 1.0 {
            return x; // sinh(tiny) = tiny with inexact
        }
        let t = expm1(ax);
        if ix < 0x3ff0_0000 {
            return h * (2.0 * t - t * t / (t + 1.0));
        }
        return h * (t + t / (t + 1.0));
    }
    // |x| in [22, log(maxdouble)]: h * exp(|x|)
    if ix < 0x4086_2E42 {
        return h * exp(ax);
    }
    // |x| in [log(maxdouble), overflowthreshold]
    if ax <= f64::from_bits(0x4086_33CE_8FB9_F87D) {
        let w = exp(0.5 * ax);
        let t = h * w;
        return t * w;
    }
    // |x| > overflowthreshold
    x * SHUGE
}

/// `StrictMath.tanh` (`FdLibm.Tanh`, fdlibm `s_tanh.c`).
pub fn tanh(x: f64) -> f64 {
    const TINY: f64 = 1.0e-300;
    let jx = hi(x);
    let ix = jx & 0x7fff_ffff;
    // x is INF or NaN
    if ix >= 0x7ff0_0000 {
        return if jx >= 0 {
            1.0 / x + 1.0
        } else {
            1.0 / x - 1.0
        };
    }
    let ax = x.abs();
    let z;
    if ix < 0x4036_0000 {
        // |x| < 22
        if ix < 0x3c80_0000 {
            return x * (1.0 + x); // tanh(small) = small
        }
        if ix >= 0x3ff0_0000 {
            // |x| >= 1
            let t = expm1(2.0 * ax);
            z = 1.0 - 2.0 / (t + 2.0);
        } else {
            let t = expm1(-2.0 * ax);
            z = -t / (t + 2.0);
        }
    } else {
        z = 1.0 - TINY; // |x| > 22: raise inexact
    }
    if jx >= 0 {
        z
    } else {
        -z
    }
}

/// `StrictMath.log10` (`FdLibm.Log10`, fdlibm `e_log10.c`).
pub fn log10(x: f64) -> f64 {
    const TWO54: f64 = f64::from_bits(0x4350_0000_0000_0000);
    const IVLN10: f64 = f64::from_bits(0x3FDB_CB7B_1526_E50E);
    const LOG10_2HI: f64 = f64::from_bits(0x3FD3_4413_509F_6000);
    const LOG10_2LO: f64 = f64::from_bits(0x3D59_FEF3_11F1_2B36);
    const ZERO: f64 = 0.0;

    let mut x = x;
    let mut hx = hi(x);
    let lx = lo(x);
    let mut k: i32 = 0;
    if hx < 0x0010_0000 {
        // x < 2**-1022
        if ((hx & 0x7fff_ffff) as u32 | lx) == 0 {
            return -TWO54 / ZERO; // log(+-0) = -inf
        }
        if hx < 0 {
            return (x - x) / ZERO; // log(-#) = NaN
        }
        k -= 54;
        x *= TWO54; // subnormal number, scale up x
        hx = hi(x);
    }
    if hx >= 0x7ff0_0000 {
        return x + x;
    }
    k += (hx >> 20) - 1023;
    let i = ((k as u32 & 0x8000_0000) >> 31) as i32;
    hx = (hx & 0x000f_ffff) | ((0x3ff - i) << 20);
    let y = f64::from(k + i);
    x = f64::from_bits((u64::from(hx as u32) << 32) | u64::from(lo(x)));
    let z = y * LOG10_2LO + IVLN10 * log(x);
    z + y * LOG10_2HI
}

/// `x` with its high word replaced by `h`.
#[inline]
fn with_hi(x: f64, h: i32) -> f64 {
    f64::from_bits((u64::from(h as u32) << 32) | u64::from(lo(x)))
}

/// `Math.scalb(z, n)` for the `n` `pow` passes (`> -1100`): one rounding.
fn scalb(z: f64, n: i32) -> f64 {
    let pow2 = |e: i32| f64::from_bits(((e + 1023) as u64) << 52);
    if n < -1000 {
        z * pow2(-1000) * pow2(n + 1000)
    } else {
        z * pow2(n)
    }
}

/// `StrictMath.pow` (`FdLibm.Pow`, the JDK's translation of fdlibm
/// `e_pow.c`, with its fast paths for `y` of `2`, `0.5` and `+-1`).
pub fn pow(x: f64, y: f64) -> f64 {
    const INFINITY: f64 = f64::INFINITY;
    // y == zero: x**0 = 1
    if y == 0.0 {
        return 1.0;
    }
    // +/-NaN return x + y to propagate NaN significands
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    let y_abs = y.abs();
    let mut x_abs = x.abs();
    // Special values of y
    if y == 2.0 {
        return x * x;
    } else if y == 0.5 {
        if x >= -f64::MAX {
            // handle x == -infinity later
            return (x + 0.0).sqrt(); // add 0.0 to handle x == -0.0
        }
    } else if y_abs == 1.0 {
        return if y == 1.0 { x } else { 1.0 / x };
    } else if y_abs == INFINITY {
        return if x_abs == 1.0 {
            y - y // inf**+/-1 is NaN
        } else if x_abs > 1.0 {
            if y >= 0.0 {
                y
            } else {
                0.0
            }
        } else if y < 0.0 {
            -y
        } else {
            0.0
        };
    }
    let hx = hi(x);
    let mut ix = hx & 0x7fff_ffff;
    // When x < 0, whether y is an odd integer: 0 not an integer, 1 odd, 2 even
    let mut y_is_int = 0;
    if hx < 0 {
        if y_abs >= f64::from_bits(0x4340_0000_0000_0000) {
            y_is_int = 2; // |y| >= 2^53: an even integer
        } else if y_abs >= 1.0 {
            let y_abs_as_long = y_abs as i64;
            if y_abs_as_long as f64 == y_abs {
                y_is_int = 2 - (y_abs_as_long & 1) as i32;
            }
        }
    }
    // Special value of x
    if x_abs == 0.0 || x_abs == INFINITY || x_abs == 1.0 {
        let mut z = x_abs; // x is +/-0, +/-inf, +/-1
        if y < 0.0 {
            z = 1.0 / z;
        }
        if hx < 0 {
            if ((ix - 0x3ff0_0000) | y_is_int) == 0 {
                z = (z - z) / (z - z); // (-1)**non-int is NaN
            } else if y_is_int == 1 {
                z = -z; // (x < 0)**odd = -(|x|**odd); fdlibm's `-1.0 * z`
            }
        }
        return z;
    }
    let mut n = (hx >> 31) + 1;
    // (x < 0)**(non-int) is NaN
    if (n | y_is_int) == 0 {
        return (x - x) / (x - x);
    }
    let mut s = 1.0; // the sign of the result
    if (n | (y_is_int - 1)) == 0 {
        s = -1.0; // (-ve)**(odd int)
    }
    let (t1, t2);
    if y_abs > f64::from_bits(0x41E0_0000_FFFF_FFFF) {
        // |y| > ~2**31
        const INV_LN2: f64 = f64::from_bits(0x3FF7_1547_652B_82FE);
        const INV_LN2_H: f64 = f64::from_bits(0x3FF7_1547_6000_0000);
        const INV_LN2_L: f64 = f64::from_bits(0x3E54_AE0B_F85D_DF44);
        // Over/underflow if x is not close to one
        if x_abs < f64::from_bits(0x3FEF_FFFF_0000_0000) {
            return if y < 0.0 { s * INFINITY } else { s * 0.0 };
        }
        if x_abs > f64::from_bits(0x3FF0_0000_FFFF_FFFF) {
            return if y > 0.0 { s * INFINITY } else { s * 0.0 };
        }
        // |1-x| is tiny <= 2**-20: log(x) by x - x^2/2 + x^3/3 - x^4/4
        let t = x_abs - 1.0; // t has 20 trailing zeros
        let w = (t * t) * (0.5 - t * (f64::from_bits(0x3FD5_5555_5555_5555) - t * 0.25));
        let u = INV_LN2_H * t; // INV_LN2_H has 21 sig. bits
        let v = t * INV_LN2_L - w * INV_LN2;
        let t1_ = with_lo_zero(u + v);
        t1 = t1_;
        t2 = v - (t1_ - u);
    } else {
        const CP: f64 = f64::from_bits(0x3FEE_C709_DC3A_03FD);
        const CP_H: f64 = f64::from_bits(0x3FEE_C709_E000_0000);
        const CP_L: f64 = f64::from_bits(0xBE3E_2FE0_145B_01F5);
        const BP: [f64; 2] = [1.0, 1.5];
        const DP_H: [f64; 2] = [0.0, f64::from_bits(0x3FE2_B803_4000_0000)];
        const DP_L: [f64; 2] = [0.0, f64::from_bits(0x3E4C_FDEB_43CF_D006)];
        const L1: f64 = f64::from_bits(0x3FE3_3333_3333_3303);
        const L2: f64 = f64::from_bits(0x3FDB_6DB6_DB6F_ABFF);
        const L3: f64 = f64::from_bits(0x3FD5_5555_518F_264D);
        const L4: f64 = f64::from_bits(0x3FD1_7460_A91D_4101);
        const L5: f64 = f64::from_bits(0x3FCD_864A_93C9_DB65);
        const L6: f64 = f64::from_bits(0x3FCA_7E28_4A45_4EEF);
        n = 0;
        // Take care of subnormal numbers
        if ix < 0x0010_0000 {
            x_abs *= f64::from_bits(0x4340_0000_0000_0000); // 2^53
            n -= 53;
            ix = hi(x_abs);
        }
        n += (ix >> 20) - 0x3ff;
        let j = ix & 0x000f_ffff;
        // Determine interval
        ix = j | 0x3ff0_0000; // normalize ix
        let k = if j <= 0x3988E {
            0 // |x| < sqrt(3/2)
        } else if j < 0xBB67A {
            1 // |x| < sqrt(3)
        } else {
            n += 1;
            ix -= 0x0010_0000;
            0
        };
        x_abs = with_hi(x_abs, ix);
        // ss = s_h + s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5)
        let u = x_abs - BP[k];
        let v = 1.0 / (x_abs + BP[k]);
        let ss = u * v;
        let s_h = with_lo_zero(ss);
        // t_h = x_abs + BP[k] High
        let t_h = with_hi(
            0.0,
            ((ix >> 1) | 0x2000_0000) + 0x0008_0000 + ((k as i32) << 18),
        );
        let t_l = x_abs - (t_h - BP[k]);
        let s_l = v * ((u - s_h * t_h) - s_h * t_l);
        // log(x_abs)
        let mut s2 = ss * ss;
        let mut r = s2 * s2 * (L1 + s2 * (L2 + s2 * (L3 + s2 * (L4 + s2 * (L5 + s2 * L6)))));
        r += s_l * (s_h + ss);
        s2 = s_h * s_h;
        let t_h = with_lo_zero(3.0 + s2 + r);
        let t_l = r - ((t_h - 3.0) - s2);
        // u + v = ss * (1 + ...)
        let u = s_h * t_h;
        let v = s_l * t_h + t_l * ss;
        // 2/(3log2) * (ss + ...)
        let p_h = with_lo_zero(u + v);
        let p_l = v - (p_h - u);
        let z_h = CP_H * p_h; // CP_H + CP_L = 2/(3*log2)
        let z_l = CP_L * p_h + p_l * CP + DP_L[k];
        // log2(x_abs) = (ss + ..)*2/(3*log2) = n + DP_H + z_h + z_l
        let t = f64::from(n);
        let t1_ = with_lo_zero(((z_h + z_l) + DP_H[k]) + t);
        t1 = t1_;
        t2 = z_l - (((t1_ - t) - DP_H[k]) - z_h);
    }
    // Split up y into (y1 + y2) and compute (y1 + y2) * (t1 + t2)
    let y1 = with_lo_zero(y);
    let p_l = (y - y1) * t1 + y * t2;
    let mut p_h = y1 * t1;
    let mut z = p_l + p_h;
    let mut j = hi(z);
    let i = lo(z) as i32;
    if j >= 0x4090_0000 {
        // z >= 1024
        if ((j - 0x4090_0000) | i) != 0 {
            return s * INFINITY; // overflow
        }
        const OVT: f64 = 8.008_566_259_537_294e-17; // -(1024-log2(ovfl+.5ulp))
        if p_l + OVT > z - p_h {
            return s * INFINITY; // overflow
        }
    } else if (j & 0x7fff_ffff) >= 0x4090_cc00 {
        // z <= -1075
        if (j.wrapping_sub(0xc090_cc00_u32 as i32) | i) != 0 {
            return s * 0.0; // underflow
        }
        if p_l <= z - p_h {
            return s * 0.0; // underflow
        }
    }
    // 2**(p_h + p_l)
    const P1: f64 = f64::from_bits(0x3FC5_5555_5555_553E);
    const P2: f64 = f64::from_bits(0xBF66_C16C_16BE_BD93);
    const P3: f64 = f64::from_bits(0x3F11_566A_AF25_DE2C);
    const P4: f64 = f64::from_bits(0xBEBB_BD41_C5D2_6BF1);
    const P5: f64 = f64::from_bits(0x3E66_3769_72BE_A4D0);
    const LG2: f64 = f64::from_bits(0x3FE6_2E42_FEFA_39EF);
    const LG2_H: f64 = f64::from_bits(0x3FE6_2E43_0000_0000);
    const LG2_L: f64 = f64::from_bits(0xBE20_5C61_0CA8_6C39);
    let i = j & 0x7fff_ffff;
    let mut k = (i >> 20) - 0x3ff;
    let mut n = 0;
    if i > 0x3fe0_0000 {
        // |z| > 0.5: n = [z + 0.5]
        n = j + (0x0010_0000 >> (k + 1));
        k = ((n & 0x7fff_ffff) >> 20) - 0x3ff; // new k for n
        let t = with_hi(0.0, n & !(0x000f_ffff >> k));
        n = ((n & 0x000f_ffff) | 0x0010_0000) >> (20 - k);
        if j < 0 {
            n = -n;
        }
        p_h -= t;
    }
    let t = with_lo_zero(p_l + p_h);
    let u = t * LG2_H;
    let v = (p_l - (t - p_h)) * LG2 + t * LG2_L;
    z = u + v;
    let w = v - (z - u);
    let t = z * z;
    let t1 = z - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    let r = (z * t1) / (t1 - 2.0) - (w + z * w);
    z = 1.0 - (r - z);
    j = hi(z).wrapping_add(n << 20);
    if (j >> 20) <= 0 {
        z = scalb(z, n); // subnormal output
    } else {
        z = with_hi(z, hi(z).wrapping_add(n << 20));
    }
    s * z
}

#[cfg(test)]
mod tests {
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
            h = (h ^ pow(x, y).to_bits()).wrapping_mul(0x0100_0000_01b3);
        }
        assert_eq!(h, 0x4929_8d43_7950_60a9);
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
}
