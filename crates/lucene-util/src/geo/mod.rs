//! Port of Lucene 10.5.0's `org.apache.lucene.geo` package: lat/lon and
//! cartesian geometries, their integer encodings, the `Component2D`
//! relation trees the geo queries are built on, the `Tessellator` that
//! turns a polygon into the triangles `LatLonShape`/`XYShape` index, and the
//! WKT/GeoJSON parsers.
//!
//! Everything here is pure geometry with no index dependency, so it lives in
//! `lucene-util`, the lowest crate the `document` fields and queries (M9
//! T9.2/T9.3) can reach. It carries `forbid(unsafe_code)` even though its
//! crate may hold `unsafe` elsewhere.
//!
//! # Fidelity
//!
//! Encoding and relations are quantization-sensitive: a point an ulp either
//! side of an edge flips in or out of a result. So every function keeps
//! Java's floating-point expression *as written* -- operand order,
//! association, `float` versus `double` intermediates, `Math.min`'s NaN and
//! signed-zero rules ([`java_min`]) -- and the differential tests
//! (`crates/lucene-util/tests/geo_fixtures.rs`) compare results bit for bit
//! against real Lucene over a seeded random corpus.
//!
//! # What Rust changes
//!
//! - Exceptions become [`GeoError`]: Java's `IllegalArgumentException`,
//!   `ParseException` (with its offset), and the two runtime exceptions the
//!   GeoJSON parser can leak (`NumberFormatException`, `NullPointerException`).
//!   Messages are Java's, including `Double.toString` formatting
//!   ([`java_double_string`]).
//! - Java's abstract `Geometry`/`LatLonGeometry`/`XYGeometry` hierarchy is
//!   the enums [`LatLonGeometry`] and [`XYGeometry`]; `Component2D` is the
//!   trait [`Component2D`], built behind a `Box<dyn Component2D>` as Java
//!   builds it behind an interface. The `within*` methods return a
//!   `Result` because `ComponentTree` throws from them.
//! - Null checks Java needs for arrays and varargs have no Rust counterpart.

#![forbid(unsafe_code)]

mod circle;
mod circle2d;
mod component2d;
mod component_tree;
mod edge_tree;
mod geo_encoding_utils;
mod geo_utils;
mod lat_lon_geometry;
mod line;
mod line2d;
mod point;
mod point2d;
mod polygon;
mod polygon2d;
mod rectangle;
mod rectangle2d;
pub mod simple_geojson_polygon_parser;
pub mod simple_wkt_shape_parser;
pub mod tessellator;
mod xy_circle;
mod xy_encoding_utils;
mod xy_geometry;
mod xy_line;
mod xy_point;
mod xy_polygon;
mod xy_rectangle;

pub use circle::Circle;
pub use component2d::{point_in_triangle, Component2D, WithinRelation};
pub use geo_encoding_utils::{Component2DPredicate, DistancePredicate, GeoEncodingUtils};
pub use geo_utils::{GeoUtils, WindingOrder};
pub use lat_lon_geometry::LatLonGeometry;
pub use line::Line;
pub use point::Point;
pub use polygon::Polygon;
pub use rectangle::Rectangle;
pub use xy_circle::XYCircle;
pub use xy_encoding_utils::XYEncodingUtils;
pub use xy_geometry::XYGeometry;
pub use xy_line::XYLine;
pub use xy_point::XYPoint;
pub use xy_polygon::XYPolygon;
pub use xy_rectangle::XYRectangle;

pub use crate::point_values_relation::Relation;

/// The exceptions Lucene's geo package throws.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GeoError {
    /// `IllegalArgumentException`: an invalid coordinate, a malformed
    /// geometry, a polygon the tessellator cannot triangulate.
    #[error("{0}")]
    IllegalArgument(String),
    /// `java.text.ParseException`: malformed WKT or GeoJSON. `offset` is
    /// Java's `getErrorOffset()` (a line number for WKT, a UTF-16 offset for
    /// GeoJSON).
    #[error("{message}")]
    Parse {
        /// The exception message.
        message: String,
        /// `getErrorOffset()`.
        offset: i32,
    },
    /// `NumberFormatException` escaping the GeoJSON parser's `\u` escape.
    #[error("{0}")]
    NumberFormat(String),
    /// `NullPointerException` the GeoJSON parser throws when a `null` or
    /// object value sits where it reads `o.getClass()`.
    #[error("{0}")]
    NullPointer(String),
    /// `IndexOutOfBoundsException` the GeoJSON parser throws on an empty
    /// polygon coordinates array.
    #[error("{0}")]
    IndexOutOfBounds(String),
}

impl GeoError {
    pub(crate) fn illegal(msg: impl Into<String>) -> GeoError {
        GeoError::IllegalArgument(msg.into())
    }

    /// The Java exception class this stands for.
    pub fn java_class(&self) -> &'static str {
        match self {
            GeoError::IllegalArgument(_) => "java.lang.IllegalArgumentException",
            GeoError::Parse { .. } => "java.text.ParseException",
            GeoError::NumberFormat(_) => "java.lang.NumberFormatException",
            GeoError::NullPointer(_) => "java.lang.NullPointerException",
            GeoError::IndexOutOfBounds(_) => "java.lang.IndexOutOfBoundsException",
        }
    }
}

/// `Math.min(double, double)`: NaN wins, and `-0.0` is less than `0.0`.
#[inline]
pub(crate) fn java_min(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if a == 0.0 && b == 0.0 && b.to_bits() == (-0.0f64).to_bits() {
        return b;
    }
    if a <= b {
        a
    } else {
        b
    }
}

/// `Math.max(double, double)`: NaN wins, and `0.0` is greater than `-0.0`.
#[inline]
pub(crate) fn java_max(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if a == 0.0 && b == 0.0 && a.to_bits() == (-0.0f64).to_bits() {
        return b;
    }
    if a >= b {
        a
    } else {
        b
    }
}

/// `Math.min(float, float)`.
#[inline]
pub(crate) fn java_min_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        return a;
    }
    if a == 0.0 && b == 0.0 && b.to_bits() == (-0.0f32).to_bits() {
        return b;
    }
    if a <= b {
        a
    } else {
        b
    }
}

/// `Math.max(float, float)`.
#[inline]
pub(crate) fn java_max_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        return a;
    }
    if a == 0.0 && b == 0.0 && a.to_bits() == (-0.0f32).to_bits() {
        return b;
    }
    if a >= b {
        a
    } else {
        b
    }
}

/// `Double.toString`: the shortest round-tripping decimal (JDK 19+), in
/// Java's layout -- `1.0`, `1.0E-5`, `NaN`, `Infinity`. Where two decimals of
/// that length are equally close, Java takes the one with the even last
/// digit; Rust's formatter need not, so ties are re-resolved here.
pub fn java_double_string(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let sci = java_tie_even(
        format!("{:e}", v.abs()),
        || format!("{:.800e}", v.abs()),
        |s| s.parse::<f64>().ok() == Some(v.abs()),
    );
    java_layout(
        v.is_sign_negative(),
        v.abs() == 0.0,
        v.abs() >= 1e-3 && v.abs() < 1e7,
        &sci,
    )
}

/// `Float.toString`, as [`java_double_string`] for a `float`.
pub fn java_float_string(v: f32) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let sci = java_tie_even(
        format!("{:e}", v.abs()),
        || format!("{:.200e}", v.abs()),
        |s| s.parse::<f32>().ok() == Some(v.abs()),
    );
    java_layout(
        v.is_sign_negative(),
        v.abs() == 0.0,
        v.abs() >= 1e-3 && v.abs() < 1e7,
        &sci,
    )
}

/// Re-resolves a tie between two equally short, equally close decimals to
/// the even last digit. `shortest` and `exact` are `{:e}` renderings.
fn java_tie_even(
    shortest: String,
    exact: impl FnOnce() -> String,
    round_trips: impl Fn(&str) -> bool,
) -> String {
    let (mant, exp) = match shortest.split_once('e') {
        Some(p) => p,
        None => return shortest,
    };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let n = digits.len();
    let last = digits.as_bytes()[n - 1] - b'0';
    if n < 2 || last.is_multiple_of(2) {
        return shortest;
    }
    let exact = exact();
    let (emant, eexp) = match exact.split_once('e') {
        Some(p) => p,
        None => return shortest,
    };
    if eexp != exp {
        return shortest;
    }
    let edigits: String = emant.chars().filter(|c| *c != '.').collect();
    if edigits.len() <= n {
        return shortest;
    }
    let (lo, tail) = edigits.split_at(n);
    let tie = tail.starts_with('5') && tail[1..].bytes().all(|b| b == b'0');
    if !tie {
        return shortest;
    }
    // The other candidate: `lo` if Rust printed `lo + 1`, else `lo + 1`
    // (whose last digit cannot carry: `lo` ends in the odd digit here).
    let other = if digits == lo {
        let mut b = lo.as_bytes().to_vec();
        b[n - 1] += 1;
        String::from_utf8(b).unwrap_or_default()
    } else {
        lo.to_string()
    };
    let candidate = format!("{}.{}e{}", &other[..1], &other[1..], exp);
    if round_trips(&candidate) {
        candidate
    } else {
        shortest
    }
}

/// Java's layout of a `{:e}` rendering: plain inside `[1e-3, 1e7)`,
/// `d.dddE[-]n` outside.
fn java_layout(negative: bool, zero: bool, plain: bool, sci: &str) -> String {
    let sign = if negative { "-" } else { "" };
    if zero {
        return format!("{sign}0.0");
    }
    let (mant, exp) = sci.split_once('e').unwrap_or((sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    if plain {
        // digits d1 d2 ... with the point after position exp + 1
        let point = exp + 1;
        let s = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point as usize >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        return format!("{sign}{s}");
    }
    let frac = if digits.len() > 1 { &digits[1..] } else { "0" };
    format!("{sign}{}.{}E{}", &digits[..1], frac, exp)
}

/// `Double.parseDouble`: Java's grammar, which is not Rust's -- an optional
/// sign, then `NaN`, `Infinity`, a hex float (`0x1.8p3`), or a decimal with
/// an optional exponent, each optionally ending in a `f`/`F`/`d`/`D` type
/// suffix; surrounding whitespace (`<= ' '`) is trimmed. `None` where Java
/// throws `NumberFormatException`.
pub(crate) fn java_parse_double(s: &str) -> Option<f64> {
    let t = s.trim_matches(|c: char| c <= ' ');
    let (negative, body) = match t.as_bytes().first() {
        Some(b'+') => (false, &t[1..]),
        Some(b'-') => (true, &t[1..]),
        _ => (false, t),
    };
    let v = if body == "NaN" {
        return Some(f64::NAN);
    } else if body == "Infinity" {
        f64::INFINITY
    } else if let Some(hex) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        parse_hex_float(hex)?
    } else {
        parse_decimal(body)?
    };
    Some(if negative { -v } else { v })
}

fn parse_decimal(body: &str) -> Option<f64> {
    let b = body.as_bytes();
    let mut i = 0;
    let mut digits = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
        digits += 1;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return None;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == start {
            return None;
        }
    }
    let end = i;
    if i < b.len() && matches!(b[i], b'f' | b'F' | b'd' | b'D') {
        i += 1;
    }
    if i != b.len() {
        return None;
    }
    body[..end].parse::<f64>().ok()
}

/// The part of a hex float after `0x`: hex digits with an optional point,
/// a mandatory binary exponent, an optional type suffix. Correctly rounded
/// (round half to even), subnormals included.
fn parse_hex_float(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut mant: u128 = 0;
    let mut sticky = false;
    let mut e2: i64 = 0;
    let mut digits = 0;
    let mut seen_point = false;
    let mut significant = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'.' && !seen_point {
            seen_point = true;
            i += 1;
            continue;
        }
        let d = match (c as char).to_digit(16) {
            Some(d) => d as u128,
            None => break,
        };
        digits += 1;
        if mant == 0 && d == 0 {
            if seen_point {
                e2 -= 4;
            }
        } else if significant < 30 {
            mant = (mant << 4) | d;
            significant += 1;
            if seen_point {
                e2 -= 4;
            }
        } else {
            sticky |= d != 0;
            if !seen_point {
                e2 += 4;
            }
        }
        i += 1;
    }
    if digits == 0 || i >= b.len() || !(b[i] == b'p' || b[i] == b'P') {
        return None;
    }
    i += 1;
    let mut exp_negative = false;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        exp_negative = b[i] == b'-';
        i += 1;
    }
    let start = i;
    let mut exp: i64 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        exp = (exp * 10 + i64::from(b[i] - b'0')).min(1 << 40);
        i += 1;
    }
    if i == start {
        return None;
    }
    if i < b.len() && matches!(b[i], b'f' | b'F' | b'd' | b'D') {
        i += 1;
    }
    if i != b.len() {
        return None;
    }
    if mant == 0 {
        return Some(0.0);
    }
    let e2 = e2 + if exp_negative { -exp } else { exp };
    let nbits = 128 - i64::from(mant.leading_zeros());
    let lead = nbits - 1 + e2; // unbiased exponent of the leading bit
    if lead > 1023 {
        return Some(f64::INFINITY);
    }
    let keep = if lead >= -1022 {
        53
    } else {
        53 - (-1022 - lead)
    };
    if keep < 0 {
        return Some(0.0);
    }
    let shift = nbits - keep;
    let mut m: u128 = if shift > 0 {
        let s = shift as u32;
        let round = (mant >> (s - 1)) & 1 == 1;
        let rest = sticky || (mant & ((1u128 << (s - 1)) - 1)) != 0;
        let mut m = mant >> s;
        if round && (rest || m & 1 == 1) {
            m += 1;
        }
        m
    } else {
        mant << (-shift) as u32
    };
    let mut lead = lead;
    if keep == 53 && m == 1u128 << 53 {
        m >>= 1;
        lead += 1;
        if lead > 1023 {
            return Some(f64::INFINITY);
        }
    }
    let bits = if lead >= -1022 && keep == 53 {
        (((lead + 1023) as u64) << 52) | (m as u64 & ((1u64 << 52) - 1))
    } else {
        // subnormal (a round-up into the normal range sets the exponent bit)
        m as u64
    };
    Some(f64::from_bits(bits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_min_max_rules() {
        assert!(java_min(f64::NAN, 1.0).is_nan());
        assert!(java_min(1.0, f64::NAN).is_nan());
        assert!(java_max(1.0, f64::NAN).is_nan());
        assert!(java_max(f64::NAN, 1.0).is_nan());
        assert_eq!(java_min(0.0, -0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(java_min(-0.0, 0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(java_max(-0.0, 0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(java_max(0.0, -0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(java_min(1.0, 2.0), 1.0);
        assert_eq!(java_max(1.0, 2.0), 2.0);
        assert!(java_min_f32(f32::NAN, 1.0).is_nan());
        assert!(java_max_f32(1.0, f32::NAN).is_nan());
        assert!(java_max_f32(f32::NAN, 1.0).is_nan());
        assert_eq!(java_min_f32(0.0, -0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(java_max_f32(-0.0, 0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(java_min_f32(3.0, 2.0), 2.0);
        assert_eq!(java_max_f32(3.0, 2.0), 3.0);
    }

    #[test]
    fn java_strings() {
        assert_eq!(java_double_string(1.0), "1.0");
        assert_eq!(java_double_string(-90.0), "-90.0");
        assert_eq!(java_double_string(1e-5), "1.0E-5");
        assert_eq!(java_double_string(-1.5e10), "-1.5E10");
        assert_eq!(java_double_string(1.2345e300), "1.2345E300");
        assert_eq!(java_double_string(f64::NAN), "NaN");
        assert_eq!(java_double_string(f64::NEG_INFINITY), "-Infinity");
        assert_eq!(java_double_string(-0.0), "-0.0");
        assert_eq!(java_double_string(f64::MAX), "1.7976931348623157E308");
        assert_eq!(java_float_string(f32::MAX), "3.4028235E38");
        assert_eq!(java_float_string(f32::INFINITY), "Infinity");
        assert_eq!(java_float_string(0.1), "0.1");
        assert_eq!(java_float_string(-1e-4), "-1.0E-4");
        // A tie between two shortest decimals resolves to the even digit.
        assert_eq!(
            java_double_string(f64::from_bits(0xc08c327620000000)),
            "-902.3076782226562"
        );
        assert_eq!(java_double_string(123.25), "123.25");
        assert_eq!(java_double_string(0.001), "0.001");
        assert_eq!(java_double_string(9999999.0), "9999999.0");
        assert_eq!(java_double_string(1e7), "1.0E7");
        assert_eq!(java_float_string(0.5), "0.5");
    }

    #[test]
    fn java_double_grammar() {
        let p = java_parse_double;
        assert_eq!(p("1"), Some(1.0));
        assert_eq!(p(" -1.5e3 "), Some(-1500.0));
        assert_eq!(p("+.5"), Some(0.5));
        assert_eq!(p("5."), Some(5.0));
        assert_eq!(p("1.1f"), Some(1.1));
        assert_eq!(p("2D"), Some(2.0));
        assert_eq!(p("1e+2"), Some(100.0));
        assert!(p("NaN").unwrap().is_nan());
        assert!(p("-NaN").unwrap().is_nan());
        assert_eq!(p("-Infinity"), Some(f64::NEG_INFINITY));
        for bad in [
            "", ".", "e5", "1e", "1e+", "nan", "inf", "infinity", "1x", "1.2.3", "--1", "1fd",
            "0x", "0x1", "0xp1", "0x1p", "0x1q1",
        ] {
            assert_eq!(p(bad), None, "{bad:?}");
        }
        assert_eq!(p("0x1p0"), Some(1.0));
        assert_eq!(p("0x1.8p1"), Some(3.0));
        assert_eq!(p("-0X.8P-1d"), Some(-0.25));
        assert_eq!(p("0x0p0"), Some(0.0));
        assert_eq!(p("0x00.000p5"), Some(0.0));
        assert_eq!(p("0x1p-1074"), Some(f64::from_bits(1)));
        assert_eq!(p("0x1p-1075"), Some(0.0));
        assert_eq!(p("0x1.1p-1075"), Some(f64::from_bits(1)));
        assert_eq!(p("0x1p-1080"), Some(0.0));
        assert_eq!(p("0x1p1024"), Some(f64::INFINITY));
        assert_eq!(p("0x1.fffffffffffffp1023"), Some(f64::MAX));
        assert_eq!(p("0x1.fffffffffffff8p1023"), Some(f64::INFINITY));
        assert_eq!(p("0x1.00000000000008p0"), Some(1.0));
        assert_eq!(p("0x1.00000000000018p0"), Some(1.0 + 2.0 * f64::EPSILON));
        assert_eq!(
            p("0x1.000000000000080000000000000000001p0"),
            Some(1.0 + f64::EPSILON)
        );
        assert_eq!(p("0x0.0000000000001p-1022"), Some(f64::from_bits(1)));
        assert_eq!(
            p("0x123456789abcdef0123456789abcdefp0"),
            Some(1.512366075204171e36)
        );
    }

    #[test]
    fn error_classes() {
        assert_eq!(
            GeoError::illegal("x").java_class(),
            "java.lang.IllegalArgumentException"
        );
        let p = GeoError::Parse {
            message: "m".into(),
            offset: 3,
        };
        assert_eq!(p.java_class(), "java.text.ParseException");
        assert_eq!(p.to_string(), "m");
        assert_eq!(
            GeoError::NumberFormat("n".into()).java_class(),
            "java.lang.NumberFormatException"
        );
        assert_eq!(
            GeoError::NullPointer("n".into()).java_class(),
            "java.lang.NullPointerException"
        );
        assert_eq!(
            GeoError::IndexOutOfBounds("n".into()).java_class(),
            "java.lang.IndexOutOfBoundsException"
        );
    }
}
