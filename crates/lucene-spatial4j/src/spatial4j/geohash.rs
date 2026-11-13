//! `GeohashUtils` (`org.locationtech.spatial4j.io`): geohash strings --
//! base-32 characters, each five bits interleaving longitude (first) and
//! latitude halvings.

use std::sync::{Arc, OnceLock};

use super::context::SpatialContext;
use super::shape::{Point, Rectangle};
use super::{Error, Result};

/// `BASE_32` (sorted).
pub const BASE_32: [u8; 32] = *b"0123456789bcdefghjkmnpqrstuvwxyz";

/// `MAX_PRECISION`.
pub const MAX_PRECISION: usize = 24;

const BITS: [u32; 5] = [16, 8, 4, 2, 1];

/// `BASE_32_IDX.length`: Java's sparse table spans `'0'..='z'`, `-500`
/// (`Arrays.fill`) where a character is not in the alphabet.
const BASE_32_IDX_LEN: usize = (b'z' - b'0' + 1) as usize;

/// `BASE_32_IDX[c - BASE_32[0]]` for an ASCII character `c` that is not a
/// (lowercased) geohash character, as Java indexes the table: unchecked, so
/// a character between `'0'` and `'z'` reads `-500` (and decodes as its
/// bits), and anything else -- including any non-ASCII unit, all past
/// `'z'` -- is `ArrayIndexOutOfBoundsException`. `unit` is the UTF-16 unit
/// Java reads there, for the message.
#[cold]
#[inline(never)]
fn not_base32(c: u8, unit: u16) -> Result<u32> {
    if (b'0'..=b'z').contains(&c) {
        return Ok(-500i32 as u32);
    }
    let i = i32::from(unit) - i32::from(BASE_32[0]);
    Err(Error::ArrayIndexOutOfBounds(format!(
        "Index {i} out of bounds for length {BASE_32_IDX_LEN}"
    )))
}

/// `encodeLatLon(latitude, longitude, precision)`.
pub fn encode_lat_lon(latitude: f64, longitude: f64, precision: usize) -> String {
    let mut lat_interval = [-90.0f64, 90.0];
    let mut lng_interval = [-180.0f64, 180.0];
    let mut geohash = String::with_capacity(precision);
    let mut is_even = true;
    let mut bit = 0usize;
    let mut ch = 0u32;
    while geohash.len() < precision {
        if is_even {
            let mid = (lng_interval[0] + lng_interval[1]) / 2.0;
            if longitude > mid {
                ch |= BITS[bit];
                lng_interval[0] = mid;
            } else {
                lng_interval[1] = mid;
            }
        } else {
            let mid = (lat_interval[0] + lat_interval[1]) / 2.0;
            if latitude > mid {
                ch |= BITS[bit];
                lat_interval[0] = mid;
            } else {
                lat_interval[1] = mid;
            }
        }
        is_even = !is_even;
        if bit < 4 {
            bit += 1;
        } else {
            geohash.push(BASE_32[ch as usize] as char);
            bit = 0;
            ch = 0;
        }
    }
    geohash
}

/// `decodeBoundary(geohash, ctx)` before the rectangle is made:
/// `[minX, maxX, minY, maxY]`. As in Java, the lookup is unchecked: a
/// character between `'0'` and `'z'` outside the alphabet decodes as
/// `-500`'s bits, anything else is `ArrayIndexOutOfBoundsException` (see
/// [`not_base32`]). Java walks UTF-16 units: an ASCII byte is its unit, and
/// any non-ASCII unit is past `'z'`.
pub fn decode_boundary_values(geohash: &str) -> Result<[f64; 4]> {
    let (mut min_y, mut max_y, mut min_x, mut max_x) = (-90.0f64, 90.0f64, -180.0f64, 180.0f64);
    let mut is_even = true;
    for (i, c) in geohash.bytes().enumerate() {
        // Java lowercases `'A'..='Z'` only, as `to_ascii_lowercase` does.
        // (A scan of the 32 sorted characters measured faster here than a
        // lookup table.)
        let lower = c.to_ascii_lowercase();
        let cd = match BASE_32.iter().position(|&b| b == lower) {
            Some(idx) => idx as u32,
            None => {
                // Every earlier byte was ASCII, so `i` starts a character:
                // the unit Java reads there is its first UTF-16 unit.
                let unit = geohash[i..].encode_utf16().next().unwrap_or(u16::from(c));
                not_base32(lower, unit)?
            }
        };
        for mask in BITS {
            if is_even {
                if cd & mask != 0 {
                    min_x = (min_x + max_x) / 2.0;
                } else {
                    max_x = (min_x + max_x) / 2.0;
                }
            } else if cd & mask != 0 {
                min_y = (min_y + max_y) / 2.0;
            } else {
                max_y = (min_y + max_y) / 2.0;
            }
            is_even = !is_even;
        }
    }
    Ok([min_x, max_x, min_y, max_y])
}

/// `decodeBoundary(geohash, ctx)`.
pub fn decode_boundary(geohash: &str, ctx: &Arc<SpatialContext>) -> Result<Arc<dyn Rectangle>> {
    let [a, b, c, d] = decode_boundary_values(geohash)?;
    ctx.rect(a, b, c, d)
}

/// `decode(geohash, ctx)`: the box's center.
pub fn decode(geohash: &str, ctx: &Arc<SpatialContext>) -> Result<Arc<dyn Point>> {
    let r = decode_boundary(geohash, ctx)?;
    let latitude = (r.min_y() + r.max_y()) / 2.0;
    let longitude = (r.min_x() + r.max_x()) / 2.0;
    ctx.point_xy(longitude, latitude)
}

/// `getSubGeohashes(baseGeohash)`: the 32 children, sorted.
pub fn sub_geohashes(base: &str) -> Vec<String> {
    BASE_32
        .iter()
        .map(|&c| {
            let mut s = String::with_capacity(base.len() + 1);
            s.push_str(base);
            s.push(c as char);
            s
        })
        .collect()
}

/// `hashLenToLatHeight` / `hashLenToLonWidth`.
fn tables() -> &'static ([f64; MAX_PRECISION + 1], [f64; MAX_PRECISION + 1]) {
    static T: OnceLock<([f64; MAX_PRECISION + 1], [f64; MAX_PRECISION + 1])> = OnceLock::new();
    T.get_or_init(|| {
        let mut lat = [0.0; MAX_PRECISION + 1];
        let mut lon = [0.0; MAX_PRECISION + 1];
        lat[0] = 90.0 * 2.0;
        lon[0] = 180.0 * 2.0;
        let mut even = false;
        for i in 1..=MAX_PRECISION {
            lat[i] = lat[i - 1] / if even { 8.0 } else { 4.0 };
            lon[i] = lon[i - 1] / if even { 4.0 } else { 8.0 };
            even = !even;
        }
        (lat, lon)
    })
}

/// `lookupDegreesSizeForHashLen(hashLen)`: `[latHeight, lonWidth]`.
pub fn lookup_degrees_size_for_hash_len(hash_len: usize) -> [f64; 2] {
    let (lat, lon) = tables();
    [lat[hash_len], lon[hash_len]]
}

/// `lookupHashLenForWidthHeight(lonErr, latErr)`: the shortest hash whose
/// cells are smaller than both errors.
pub fn lookup_hash_len_for_width_height(lon_err: f64, lat_err: f64) -> usize {
    let (lat, lon) = tables();
    for len in 1..MAX_PRECISION {
        if lat[len] < lat_err && lon[len] < lon_err {
            return len;
        }
    }
    MAX_PRECISION
}
