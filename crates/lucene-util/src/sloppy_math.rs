//! Port of `org.apache.lucene.util.SloppyMath`: table-driven `cos`, `sin`,
//! `asin` and the haversine distance Lucene's geo queries and distance sort
//! are defined by.
//!
//! The lookup tables are built exactly as Java's static initializer builds
//! them -- from `StrictMath.sin/cos/asin`, which [`crate::strict_math`] ports
//! bit for bit -- so every value here is Java's value, not merely close to
//! it. That matters: a distance query's sort-key cut-off
//! (`GeoUtils.distanceQuerySortKey`) is an exact double, and a point one ulp
//! either side of it is in or out of the result.
//!
//! Rust-only differences: the tables are built once on first use
//! (`OnceLock`) instead of in a class initializer; `Math.cos` (the fallback
//! for arguments past ~4e6 radians, which no geo caller reaches) is the
//! platform libm's `cos` -- HotSpot's `Math.cos` is a per-platform intrinsic
//! no portable code reproduces bit for bit, so there the port is within an
//! ulp of Java on x86-64 rather than identical.

use std::sync::OnceLock;

use crate::strict_math;

/// Earth's mean radius in meters (`TO_METERS`).
const TO_METERS: f64 = 6_371_008.771_4;

const ONE_DIV_F2: f64 = 1.0 / 2.0;
const ONE_DIV_F3: f64 = 1.0 / 6.0;
const ONE_DIV_F4: f64 = 1.0 / 24.0;

const PIO2_HI: f64 = f64::from_bits(0x3FF9_21FB_5440_0000);
const PIO2_LO: f64 = f64::from_bits(0x3DD0_B461_1A62_6331);
const TWOPI_HI: f64 = 4.0 * PIO2_HI;
const TWOPI_LO: f64 = 4.0 * PIO2_LO;
const SIN_COS_TABS_SIZE: usize = (1 << 11) + 1;
const SIN_COS_DELTA_HI: f64 = TWOPI_HI / (SIN_COS_TABS_SIZE - 1) as f64;
const SIN_COS_DELTA_LO: f64 = TWOPI_LO / (SIN_COS_TABS_SIZE - 1) as f64;
const SIN_COS_INDEXER: f64 = 1.0 / (SIN_COS_DELTA_HI + SIN_COS_DELTA_LO);
/// Above this, `cos` reduces with the full-precision function instead.
const SIN_COS_MAX_VALUE_FOR_INT_MODULO: f64 = ((i32::MAX >> 9) as f64 / SIN_COS_INDEXER) * 0.99;

const ASIN_TABS_SIZE: usize = (1 << 13) + 1;
const ASIN_PIO2_HI: f64 = f64::from_bits(0x3FF9_21FB_5444_2D18);
const ASIN_PIO2_LO: f64 = f64::from_bits(0x3C91_A626_3314_5C07);
const ASIN_PS0: f64 = f64::from_bits(0x3fc5_5555_5555_5555);
const ASIN_PS1: f64 = f64::from_bits(0xbfd4_d612_03eb_6f7d);
const ASIN_PS2: f64 = f64::from_bits(0x3fc9_c155_0e88_4455);
const ASIN_PS3: f64 = f64::from_bits(0xbfa4_8228_b568_8f3b);
const ASIN_PS4: f64 = f64::from_bits(0x3f49_efe0_7501_b288);
const ASIN_PS5: f64 = f64::from_bits(0x3f02_3de1_0dfd_f709);
const ASIN_QS1: f64 = f64::from_bits(0xc003_3a27_1c8a_2d4b);
const ASIN_QS2: f64 = f64::from_bits(0x4000_2ae5_9c59_8ac8);
const ASIN_QS3: f64 = f64::from_bits(0xbfe6_066c_1b8d_0159);
const ASIN_QS4: f64 = f64::from_bits(0x3fb3_b8c5_b12e_9282);

/// `Math.PI / 2D`.
const PIO2: f64 = std::f64::consts::PI / 2.0;

/// Java's seven parallel arrays, interleaved (stage 3, see `cos`): one
/// `(cos, sin)` pair and one `(asin, der1..der4)` row per index, so a lookup
/// touches one cache line, and fixed-size so the masked `cos` index needs no
/// bounds check.
struct Tables {
    sin_cos: Box<[[f64; 2]; SIN_COS_TABS_SIZE]>,
    asin_max_value_for_tabs: f64,
    asin_delta: f64,
    asin_indexer: f64,
    asin: Box<[[f64; 5]; ASIN_TABS_SIZE]>,
}

/// Java's static initializer.
fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let pi_index = (SIN_COS_TABS_SIZE - 1) / 2;
        let pi_mul_2_index = 2 * pi_index;
        let pi_mul_0_5_index = pi_index / 2;
        let pi_mul_1_5_index = 3 * pi_index / 2;
        let mut sin = vec![0.0; SIN_COS_TABS_SIZE];
        let mut cos = vec![0.0; SIN_COS_TABS_SIZE];
        for i in 0..SIN_COS_TABS_SIZE {
            let fi = i as f64;
            let angle = fi * SIN_COS_DELTA_HI + fi * SIN_COS_DELTA_LO;
            let mut sin_angle = strict_math::sin(angle);
            let mut cos_angle = strict_math::cos(angle);
            if i == pi_index || i == pi_mul_2_index {
                sin_angle = 0.0;
            } else if i == pi_mul_0_5_index || i == pi_mul_1_5_index {
                cos_angle = 0.0;
            }
            sin[i] = sin_angle;
            cos[i] = cos_angle;
        }
        let asin_max_value_for_tabs = strict_math::sin(73.0f64.to_radians());
        let asin_delta = asin_max_value_for_tabs / (ASIN_TABS_SIZE - 1) as f64;
        let asin_indexer = 1.0 / asin_delta;
        let mut asin = vec![0.0; ASIN_TABS_SIZE];
        let mut der1 = vec![0.0; ASIN_TABS_SIZE];
        let mut der2 = vec![0.0; ASIN_TABS_SIZE];
        let mut der3 = vec![0.0; ASIN_TABS_SIZE];
        let mut der4 = vec![0.0; ASIN_TABS_SIZE];
        for i in 0..ASIN_TABS_SIZE {
            let x = i as f64 * asin_delta;
            asin[i] = strict_math::asin(x);
            let one_minus_x_sq_inv = 1.0 / (1.0 - x * x);
            let inv0_5 = one_minus_x_sq_inv.sqrt();
            let inv1_5 = inv0_5 * one_minus_x_sq_inv;
            let inv2_5 = inv1_5 * one_minus_x_sq_inv;
            let inv3_5 = inv2_5 * one_minus_x_sq_inv;
            der1[i] = inv0_5;
            der2[i] = (x * inv1_5) * ONE_DIV_F2;
            der3[i] = ((1.0 + 2.0 * x * x) * inv2_5) * ONE_DIV_F3;
            der4[i] = ((5.0 + 2.0 * x * (2.0 + x * (5.0 - 2.0 * x))) * inv3_5) * ONE_DIV_F4;
        }
        let sin_cos: Vec<[f64; 2]> = cos.iter().zip(&sin).map(|(&c, &s)| [c, s]).collect();
        let asin_rows: Vec<[f64; 5]> = (0..ASIN_TABS_SIZE)
            .map(|i| [asin[i], der1[i], der2[i], der3[i], der4[i]])
            .collect();
        Tables {
            sin_cos: sin_cos.into_boxed_slice().try_into().expect("table size"),
            asin_max_value_for_tabs,
            asin_delta,
            asin_indexer,
            asin: asin_rows.into_boxed_slice().try_into().expect("table size"),
        }
    })
}

/// `SloppyMath.haversinMeters(lat1, lon1, lat2, lon2)`: the haversine
/// distance in meters between two points in decimal degrees.
///
/// `inline(always)`, as the three haversine entry points are: the JVM
/// inlines them into the caller's loop, where independent distances overlap;
/// as a call each one is a serial chain (measured 0.84x -> 1.1x of Lucene).
#[inline(always)]
pub fn haversin_meters(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let t = tables();
    meters_with(t, sort_key_with(t, lat1, lon1, lat2, lon2))
}

/// `SloppyMath.haversinMeters(sortKey)`: the distance for a value of
/// [`haversin_sort_key`].
#[inline(always)]
pub fn haversin_meters_from_sort_key(sort_key: f64) -> f64 {
    meters_with(tables(), sort_key)
}

#[inline(always)]
fn meters_with(t: &Tables, sort_key: f64) -> f64 {
    // Java's Math.min: NaN-propagating.
    let h = (sort_key * 0.5).sqrt();
    let m = if h.is_nan() { h } else { h.min(1.0) };
    TO_METERS * 2.0 * asin_with(t, m)
}

/// `SloppyMath.haversinSortKey`: compares like the distance, cheaper.
#[inline(always)]
pub fn haversin_sort_key(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    sort_key_with(tables(), lat1, lon1, lat2, lon2)
}

#[inline(always)]
fn sort_key_with(t: &Tables, lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let x1 = lat1.to_radians();
    let x2 = lat2.to_radians();
    let h1 = 1.0 - cos_with(t, x1 - x2);
    let h2 = 1.0 - cos_with(t, (lon1 - lon2).to_radians());
    let h = h1 + cos_with(t, x1) * cos_with(t, x2) * h2;
    // clobber crazy precision so subsequent rounding does not create ties.
    f64::from_bits(h.to_bits() & 0xFFFF_FFFF_FFFF_FFF8)
}

/// `SloppyMath.cos`: error around 1e-15.
pub fn cos(a: f64) -> f64 {
    cos_with(tables(), a)
}

/// `cos` with the tables already in hand (`haversin_sort_key` makes four
/// calls; one `OnceLock` read serves them all).
#[inline(always)]
fn cos_with(t: &Tables, a: f64) -> f64 {
    let a = if a < 0.0 { -a } else { a };
    if a.is_nan() || a > SIN_COS_MAX_VALUE_FOR_INT_MODULO {
        return cos_far(t, a);
    }
    // Java's `(int)` cast, which saturates; the argument is in range here, so
    // the cast need not clamp -- a clamping `as` puts a `maxsd`/`minsd` pair
    // on the critical path of all four `cos` calls of a haversine.
    // SAFETY: `0 <= a <= SIN_COS_MAX_VALUE_FOR_INT_MODULO` (NaN fails the
    // test above), so the operand is finite and below `(i32::MAX >> 9) + 1`.
    let mut index = unsafe { (a * SIN_COS_INDEXER + 0.5).to_int_unchecked::<i32>() };
    let delta = (a - f64::from(index) * SIN_COS_DELTA_HI) - f64::from(index) * SIN_COS_DELTA_LO;
    index &= (SIN_COS_TABS_SIZE - 2) as i32;
    let [index_cos, index_sin] = t.sin_cos[index as usize & (SIN_COS_TABS_SIZE - 2)];
    index_cos
        + delta
            * (-index_sin
                + delta
                    * (-index_cos * ONE_DIV_F2
                        + delta * (index_sin * ONE_DIV_F3 + delta * index_cos * ONE_DIV_F4)))
}

/// `cos` of NaN or past the table's reach: `Math.cos`, or for NaN the table
/// arithmetic at Java's `(int) NaN == 0`, which yields NaN through `delta`.
#[cold]
#[inline(never)]
fn cos_far(t: &Tables, a: f64) -> f64 {
    if a > SIN_COS_MAX_VALUE_FOR_INT_MODULO {
        return libm_cos(a);
    }
    let delta = (a - 0.0 * SIN_COS_DELTA_HI) - 0.0 * SIN_COS_DELTA_LO;
    let [index_cos, index_sin] = t.sin_cos[0];
    index_cos
        + delta
            * (-index_sin
                + delta
                    * (-index_cos * ONE_DIV_F2
                        + delta * (index_sin * ONE_DIV_F3 + delta * index_cos * ONE_DIV_F4)))
}

/// `Math.cos`, off the hot path.
#[cold]
#[inline(never)]
fn libm_cos(a: f64) -> f64 {
    a.cos()
}

/// `SloppyMath.sin`, as `cos(a - pi/2)` (so `sin(0) != 0`).
pub fn sin(a: f64) -> f64 {
    cos(a - PIO2)
}

/// `SloppyMath.asin`: error around 1e-7.
pub fn asin(a: f64) -> f64 {
    asin_with(tables(), a)
}

#[inline(always)]
fn asin_with(t: &Tables, a: f64) -> f64 {
    let (a, negate) = if a < 0.0 { (-a, true) } else { (a, false) };
    let result = if a <= t.asin_max_value_for_tabs {
        // SAFETY: `0 <= a <= asin_max_value_for_tabs` (NaN fails the test),
        // so the operand is finite and at most `ASIN_TABS_SIZE - 0.5`.
        let index = unsafe { (a * t.asin_indexer + 0.5).to_int_unchecked::<i32>() };
        // `index * ASIN_DELTA` converts the `int` (`cvtsi2sd`); through a
        // `usize` it would take the unsigned 64-bit conversion's longer path.
        let delta = a - f64::from(index) * t.asin_delta;
        let [v, d1, d2, d3, d4] = t.asin[index as usize];
        v + delta * (d1 + delta * (d2 + delta * (d3 + delta * d4)))
    } else if a < 1.0 {
        // derived from fdlibm
        let t = (1.0 - a) * 0.5;
        let p = t
            * (ASIN_PS0
                + t * (ASIN_PS1 + t * (ASIN_PS2 + t * (ASIN_PS3 + t * (ASIN_PS4 + t * ASIN_PS5)))));
        let q = 1.0 + t * (ASIN_QS1 + t * (ASIN_QS2 + t * (ASIN_QS3 + t * ASIN_QS4)));
        let s = t.sqrt();
        let z = s + s * (p / q);
        ASIN_PIO2_HI - ((z + z) - ASIN_PIO2_LO)
    } else if a == 1.0 {
        std::f64::consts::PI / 2.0
    } else {
        return f64::NAN;
    };
    if negate {
        -result
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haversin_known_distances() {
        assert_eq!(haversin_meters(0.0, 0.0, 0.0, 0.0), 0.0);
        let d = haversin_meters(40.7143528, -74.0059731, 40.7143528, -74.0059731);
        assert_eq!(d, 0.0);
        // Lucene's TestSloppyMath: one degree of longitude at the equator.
        let d = haversin_meters(0.0, 0.0, 0.0, 1.0);
        assert!((d - 111_195.08).abs() < 1.0, "{d}");
        // Antipodes saturate at half the circumference.
        let d = haversin_meters(0.0, 0.0, 0.0, 180.0);
        assert!((d - TO_METERS * std::f64::consts::PI).abs() < 1.0, "{d}");
        assert!(haversin_meters_from_sort_key(f64::NAN).is_nan());
        assert_eq!(
            haversin_meters_from_sort_key(4.0),
            haversin_meters_from_sort_key(2.0)
        );
    }

    #[test]
    fn nan_and_far_arguments_take_the_cold_path() {
        assert!(cos(f64::NAN).is_nan());
        assert!(cos(-f64::NAN).is_nan());
        assert!(sin(f64::NAN).is_nan());
        assert!(haversin_sort_key(f64::NAN, 0.0, 0.0, 0.0).is_nan());
        // Past the table's reach: `Math.cos`.
        let far = 1e10;
        assert_eq!(cos(far), far.cos());
        assert_eq!(cos(-far), far.cos());
        assert!(asin(f64::NAN).is_nan());
        assert_eq!(asin(1.0), std::f64::consts::PI / 2.0);
        assert!(asin(1.5).is_nan());
        assert_eq!(asin(-0.5), -asin(0.5));
    }

    #[test]
    fn sloppy_trig_is_close() {
        let mut x = -10.0f64;
        while x < 10.0 {
            assert!((cos(x) - x.cos()).abs() < 1e-14, "{x}");
            assert!((sin(x) - x.sin()).abs() < 1e-12, "{x}");
            x += 0.01;
        }
        assert!(cos(f64::NAN).is_nan());
        assert!(cos(f64::INFINITY).is_nan());
        assert_eq!(cos(1e10), 1e10f64.cos());
        let mut a = -1.0f64;
        while a <= 1.0 {
            assert!((asin(a) - a.asin()).abs() < 1e-7, "{a}");
            a += 0.001;
        }
        assert_eq!(asin(1.0), std::f64::consts::FRAC_PI_2);
        assert_eq!(asin(-1.0), -std::f64::consts::FRAC_PI_2);
        assert!(asin(1.0001).is_nan());
        assert!(asin(f64::NAN).is_nan());
        assert!((asin(0.99) - 0.99f64.asin()).abs() < 1e-12);
    }
}
