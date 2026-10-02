//! Port of the subset of Google's S2 geometry library
//! (`com.google.common.geometry`, the `io.sgr:s2-geometry-library-java`
//! 1.0.0 jar Lucene 10.5.0 depends on; Apache License 2.0) that
//! `S2PrefixTree` uses: `S2CellId` (the Hilbert-curve cell ids and their
//! hierarchy), `S2Cell`'s vertices, `S2LatLng.toPoint`, `S2Point`,
//! `S2Projections`' quadratic projection and its `Metric`s.
//!
//! The rest of the library (regions, coverers, loops, polygons, edge
//! indexes) is not used by spatial-extras -- Lucene relates an S2 cell to a
//! query shape through Geo3D (`GeoS2Shape`), not through S2's own regions.
//! `crates/lucene-util/tests/spatial4j_fixtures.rs` compares cell ids,
//! levels, tokens and vertices against the real jar.
//!
//! The Java classes are `strictfp`; `Math.sin`/`cos` are `StrictMath`'s here
//! as everywhere in this port (the fixtures are generated with HotSpot's
//! trig intrinsics off).

#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::fmt;
use std::sync::OnceLock;

use crate::strict_math::{cos, sin};

/// `S2Point`: a point in 3-space (not necessarily unit length).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct S2Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl S2Point {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        S2Point { x, y, z }
    }

    /// `get(axis)`.
    pub fn get(&self, axis: usize) -> f64 {
        match axis {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }

    /// `largestAbsComponent()`.
    pub fn largest_abs_component(&self) -> usize {
        let (x, y, z) = (self.x.abs(), self.y.abs(), self.z.abs());
        if x > y {
            if x > z {
                0
            } else {
                2
            }
        } else if y > z {
            1
        } else {
            2
        }
    }

    /// `norm2()`.
    pub fn norm2(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// `S2Point.normalize(p)`.
    pub fn normalize(&self) -> S2Point {
        let mut norm = self.norm2().sqrt();
        if norm != 0.0 {
            norm = 1.0 / norm;
        }
        S2Point::new(self.x * norm, self.y * norm, self.z * norm)
    }
}

/// `S2LatLng`: a latitude/longitude in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct S2LatLng {
    lat_radians: f64,
    lng_radians: f64,
}

impl S2LatLng {
    /// `fromRadians(lat, lng)`.
    pub fn from_radians(lat_radians: f64, lng_radians: f64) -> Self {
        S2LatLng {
            lat_radians,
            lng_radians,
        }
    }

    /// `fromDegrees(lat, lng)`: `S1Angle.degrees` multiplies by
    /// `Math.PI / 180`.
    pub fn from_degrees(lat_degrees: f64, lng_degrees: f64) -> Self {
        let k = std::f64::consts::PI / 180.0;
        Self::from_radians(lat_degrees * k, lng_degrees * k)
    }

    /// `toPoint()`.
    pub fn to_point(&self) -> S2Point {
        let phi = self.lat_radians;
        let theta = self.lng_radians;
        let cosphi = cos(phi);
        S2Point::new(cos(theta) * cosphi, sin(theta) * cosphi, sin(phi))
    }
}

/// `S2Projections` (the quadratic projection Java compiles in).
pub mod projections {
    use super::S2Point;

    /// `stToUV(s)`.
    pub fn st_to_uv(s: f64) -> f64 {
        if s >= 0.0 {
            (1.0 / 3.0) * ((1.0 + s) * (1.0 + s) - 1.0)
        } else {
            (1.0 / 3.0) * (1.0 - (1.0 - s) * (1.0 - s))
        }
    }

    /// `uvToST(u)`.
    pub fn uv_to_st(u: f64) -> f64 {
        if u >= 0.0 {
            (1.0 + 3.0 * u).sqrt() - 1.0
        } else {
            1.0 - (1.0 - 3.0 * u).sqrt()
        }
    }

    /// `faceUvToXyz(face, u, v)`.
    pub fn face_uv_to_xyz(face: i32, u: f64, v: f64) -> S2Point {
        match face {
            0 => S2Point::new(1.0, u, v),
            1 => S2Point::new(-u, 1.0, v),
            2 => S2Point::new(-u, -v, 1.0),
            3 => S2Point::new(-1.0, -v, -u),
            4 => S2Point::new(v, -1.0, -u),
            _ => S2Point::new(v, u, -1.0),
        }
    }

    /// `validFaceXyzToUv(face, p)`.
    pub fn valid_face_xyz_to_uv(face: i32, p: &S2Point) -> (f64, f64) {
        match face {
            0 => (p.y / p.x, p.z / p.x),
            1 => (-p.x / p.y, p.z / p.y),
            2 => (-p.x / p.z, -p.y / p.z),
            3 => (p.z / p.x, p.y / p.x),
            4 => (p.z / p.y, -p.x / p.y),
            _ => (-p.y / p.z, -p.x / p.z),
        }
    }

    /// `xyzToFace(p)`.
    pub fn xyz_to_face(p: &S2Point) -> i32 {
        let mut face = p.largest_abs_component() as i32;
        if p.get(face as usize) < 0.0 {
            face += 3;
        }
        face
    }

    /// `S2.Metric`: a cell measure that scales as `deriv * 2^(-dim*level)`.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Metric {
        deriv: f64,
        dim: i32,
    }

    impl Metric {
        pub const fn new(dim: i32, deriv: f64) -> Self {
            Metric { deriv, dim }
        }

        /// `deriv()`.
        pub fn deriv(&self) -> f64 {
            self.deriv
        }

        /// `getValue(level)`: `StrictMath.scalb(deriv, dim * (1 - level))`.
        pub fn get_value(&self, level: i32) -> f64 {
            scalb(self.deriv, self.dim * (1 - level))
        }

        /// `getMinLevel(value)`: the minimum level whose measure is at
        /// most `value`.
        pub fn get_min_level(&self, value: f64) -> i32 {
            if value <= 0.0 {
                return super::MAX_LEVEL;
            }
            let exponent = exp(value / ((1 << self.dim) as f64 * self.deriv));
            0.max(super::MAX_LEVEL.min(-((exponent - 1) >> (self.dim - 1))))
        }

        /// `getMaxLevel(value)`.
        pub fn get_max_level(&self, value: f64) -> i32 {
            if value <= 0.0 {
                return super::MAX_LEVEL;
            }
            let exponent = exp((1 << self.dim) as f64 * self.deriv / value);
            0.max(super::MAX_LEVEL.min((exponent - 1) >> (self.dim - 1)))
        }
    }

    /// `S2.exp(v)`: the binary exponent of `v` plus one (frexp's).
    pub fn exp(v: f64) -> i32 {
        if v == 0.0 {
            return 0;
        }
        let bits = v.to_bits() as i64;
        (((0x7ff0_0000_0000_0000i64 & bits) >> 52) as i32) - 1022
    }

    /// `StrictMath.scalb(d, scaleFactor)`: `d * 2^scaleFactor`, correctly
    /// rounded (gradual underflow included).
    pub fn scalb(d: f64, scale_factor: i32) -> f64 {
        // Split the scaling so no intermediate overflows or underflows
        // early; each step is exact except possibly the last.
        let mut d = d;
        let mut n = scale_factor.clamp(-2200, 2200);
        while n > 1000 {
            d *= 2f64.powi(1000);
            n -= 1000;
        }
        while n < -1000 {
            d *= 2f64.powi(-1000);
            n += 1000;
            if d == 0.0 {
                return d;
            }
        }
        if n < -1022 {
            d *= 2f64.powi(-1022);
            n += 1022;
        }
        d * 2f64.powi(n)
    }

    /// `S2Projections.MAX_ANGLE_SPAN`'s derivative (quadratic projection).
    // Java's literal, digit for digit.
    #[allow(clippy::excessive_precision)]
    pub const MAX_ANGLE_SPAN_DERIV: f64 = 0.85244858959960922;

    /// `S2Projections.MAX_WIDTH`.
    pub const MAX_WIDTH: Metric = Metric::new(1, MAX_ANGLE_SPAN_DERIV);
}

/// `S2CellId.FACE_BITS`.
pub const FACE_BITS: i32 = 3;
/// `S2CellId.NUM_FACES`.
pub const NUM_FACES: i32 = 6;
/// `S2CellId.MAX_LEVEL`.
pub const MAX_LEVEL: i32 = 30;
/// `S2CellId.POS_BITS`.
pub const POS_BITS: i32 = 2 * MAX_LEVEL + 1;
/// `S2CellId.MAX_SIZE`.
pub const MAX_SIZE: i32 = 1 << MAX_LEVEL;

const LOOKUP_BITS: i32 = 4;
const SWAP_MASK: i32 = 0x01;
const INVERT_MASK: i32 = 0x02;

/// `S2.POS_TO_ORIENTATION`.
const POS_TO_ORIENTATION: [i32; 4] = [SWAP_MASK, 0, 0, INVERT_MASK + SWAP_MASK];
/// `S2.POS_TO_IJ`.
const POS_TO_IJ: [[i32; 4]; 4] = [[0, 1, 3, 2], [0, 2, 3, 1], [3, 2, 0, 1], [3, 1, 0, 2]];

struct Lookup {
    pos: Vec<i32>,
    ij: Vec<i32>,
}

fn lookup() -> &'static Lookup {
    static L: OnceLock<Lookup> = OnceLock::new();
    L.get_or_init(|| {
        let n = 1usize << (2 * LOOKUP_BITS + 2);
        let mut l = Lookup {
            pos: vec![0; n],
            ij: vec![0; n],
        };
        init_lookup_cell(&mut l, 0, 0, 0, 0, 0, 0);
        init_lookup_cell(&mut l, 0, 0, 0, SWAP_MASK, 0, SWAP_MASK);
        init_lookup_cell(&mut l, 0, 0, 0, INVERT_MASK, 0, INVERT_MASK);
        init_lookup_cell(
            &mut l,
            0,
            0,
            0,
            SWAP_MASK | INVERT_MASK,
            0,
            SWAP_MASK | INVERT_MASK,
        );
        l
    })
}

/// `S2CellId.initLookupCell(..)`.
fn init_lookup_cell(
    l: &mut Lookup,
    level: i32,
    i: i32,
    j: i32,
    orig_orientation: i32,
    pos: i32,
    orientation: i32,
) {
    if level == LOOKUP_BITS {
        let ij = (i << LOOKUP_BITS) + j;
        l.pos[((ij << 2) + orig_orientation) as usize] = (pos << 2) + orientation;
        l.ij[((pos << 2) + orig_orientation) as usize] = (ij << 2) + orientation;
    } else {
        let level = level + 1;
        let i = i << 1;
        let j = j << 1;
        let pos = pos << 2;
        for sub_pos in 0..4 {
            let ij = POS_TO_IJ[orientation as usize][sub_pos as usize];
            let orientation_mask = POS_TO_ORIENTATION[sub_pos as usize];
            init_lookup_cell(
                l,
                level,
                i + ((ij as u32) >> 1) as i32,
                j + (ij & 1),
                orig_orientation,
                pos + sub_pos,
                orientation ^ orientation_mask,
            );
        }
    }
}

/// `Math.round(double)`: the closest `long`, ties toward positive
/// infinity, NaN to 0, saturating.
fn java_round(x: f64) -> i64 {
    if x.is_nan() {
        return 0;
    }
    let f = x.floor();
    let r = if x - f >= 0.5 { f + 1.0 } else { f };
    r as i64
}

/// `S2CellId`: a cell of the S2 hierarchy -- three face bits, then the
/// Hilbert-curve position, then a marker bit whose position gives the
/// level. Ordered as an unsigned 64-bit integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct S2CellId {
    id: i64,
}

impl S2CellId {
    /// `new S2CellId(id)`.
    pub const fn new(id: i64) -> Self {
        S2CellId { id }
    }

    /// `none()`.
    pub fn none() -> Self {
        S2CellId { id: 0 }
    }

    /// `fromFacePosLevel(face, pos, level)`.
    pub fn from_face_pos_level(face: i32, pos: i64, level: i32) -> Self {
        S2CellId::new(((face as i64) << POS_BITS).wrapping_add(pos | 1)).parent(level)
    }

    /// `fromPoint(p)`.
    pub fn from_point(p: &S2Point) -> Self {
        let face = projections::xyz_to_face(p);
        let (u, v) = projections::valid_face_xyz_to_uv(face, p);
        let i = st_to_ij(projections::uv_to_st(u));
        let j = st_to_ij(projections::uv_to_st(v));
        Self::from_face_ij(face, i, j)
    }

    /// `fromLatLng(ll)`.
    pub fn from_lat_lng(ll: &S2LatLng) -> Self {
        Self::from_point(&ll.to_point())
    }

    /// `id()`.
    pub fn id(&self) -> i64 {
        self.id
    }

    /// `isValid()`.
    pub fn is_valid(&self) -> bool {
        self.face() < NUM_FACES && (self.lowest_on_bit() & 0x1555_5555_5555_5555) != 0
    }

    /// `face()`.
    pub fn face(&self) -> i32 {
        ((self.id as u64) >> POS_BITS) as i32
    }

    /// `pos()`.
    pub fn pos(&self) -> i64 {
        self.id & ((-1i64 as u64) >> FACE_BITS) as i64
    }

    /// `level()`.
    pub fn level(&self) -> i32 {
        if self.is_leaf() {
            return MAX_LEVEL;
        }
        let mut x = self.id as i32;
        let mut level = -1;
        if x != 0 {
            level += 16;
        } else {
            x = ((self.id as u64) >> 32) as i32;
        }
        x &= x.wrapping_neg();
        if x & 0x0000_5555 != 0 {
            level += 8;
        }
        if x & 0x0055_0055 != 0 {
            level += 4;
        }
        if x & 0x0505_0505 != 0 {
            level += 2;
        }
        if x & 0x1111_1111 != 0 {
            level += 1;
        }
        level
    }

    /// `isLeaf()`.
    pub fn is_leaf(&self) -> bool {
        (self.id as i32) & 1 != 0
    }

    /// `isFace()`.
    pub fn is_face(&self) -> bool {
        (self.id & (Self::lowest_on_bit_for_level(0) - 1)) == 0
    }

    /// `childPosition(level)`.
    pub fn child_position(&self, level: i32) -> i32 {
        (((self.id as u64) >> (2 * (MAX_LEVEL - level) + 1)) as i32) & 3
    }

    /// `rangeMin()`.
    pub fn range_min(&self) -> Self {
        S2CellId::new(self.id.wrapping_sub(self.lowest_on_bit().wrapping_sub(1)))
    }

    /// `rangeMax()`.
    pub fn range_max(&self) -> Self {
        S2CellId::new(self.id.wrapping_add(self.lowest_on_bit().wrapping_sub(1)))
    }

    /// `contains(other)`.
    pub fn contains(&self, other: &S2CellId) -> bool {
        *other >= self.range_min() && *other <= self.range_max()
    }

    /// `intersects(other)`.
    pub fn intersects(&self, other: &S2CellId) -> bool {
        other.range_min() <= self.range_max() && other.range_max() >= self.range_min()
    }

    /// `parent()`.
    pub fn parent_one(&self) -> Self {
        let new_lsb = self.lowest_on_bit().wrapping_shl(2);
        S2CellId::new((self.id & new_lsb.wrapping_neg()) | new_lsb)
    }

    /// `parent(level)`.
    pub fn parent(&self, level: i32) -> Self {
        let new_lsb = Self::lowest_on_bit_for_level(level);
        S2CellId::new((self.id & new_lsb.wrapping_neg()) | new_lsb)
    }

    /// `childBegin()`.
    pub fn child_begin_one(&self) -> Self {
        let old_lsb = self.lowest_on_bit();
        S2CellId::new(
            self.id
                .wrapping_sub(old_lsb)
                .wrapping_add(((old_lsb as u64) >> 2) as i64),
        )
    }

    /// `childBegin(level)`.
    pub fn child_begin(&self, level: i32) -> Self {
        S2CellId::new(
            self.id
                .wrapping_sub(self.lowest_on_bit())
                .wrapping_add(Self::lowest_on_bit_for_level(level)),
        )
    }

    /// `childEnd(level)`.
    pub fn child_end(&self, level: i32) -> Self {
        S2CellId::new(
            self.id
                .wrapping_add(self.lowest_on_bit())
                .wrapping_add(Self::lowest_on_bit_for_level(level)),
        )
    }

    /// `next()`.
    pub fn next(&self) -> Self {
        S2CellId::new(self.id.wrapping_add(self.lowest_on_bit().wrapping_shl(1)))
    }

    /// `prev()`.
    pub fn prev(&self) -> Self {
        S2CellId::new(self.id.wrapping_sub(self.lowest_on_bit().wrapping_shl(1)))
    }

    /// `toToken()`.
    pub fn to_token(&self) -> String {
        if self.id == 0 {
            return "X".into();
        }
        let hex = format!("{:016x}", self.id as u64);
        hex.trim_end_matches('0').to_string()
    }

    /// `fromFaceIJ(face, i, j)`.
    pub fn from_face_ij(face: i32, i: i32, j: i32) -> Self {
        let mut n: [i64; 2] = [0, (face << (POS_BITS - 33)) as i64];
        let mut bits = face & SWAP_MASK;
        let l = lookup();
        for k in (0..=7).rev() {
            let mask = (1 << LOOKUP_BITS) - 1;
            bits += ((i >> (k * LOOKUP_BITS)) & mask) << (LOOKUP_BITS + 2);
            bits += ((j >> (k * LOOKUP_BITS)) & mask) << 2;
            bits = l.pos[bits as usize];
            n[(k >> 2) as usize] |= ((bits as i64) >> 2) << ((k & 3) * 2 * LOOKUP_BITS);
            bits &= SWAP_MASK | INVERT_MASK;
        }
        S2CellId::new(
            ((n[1] << 32).wrapping_add(n[0]))
                .wrapping_shl(1)
                .wrapping_add(1),
        )
    }

    /// `toFaceIJOrientation(pi, pj, orientation)`: `(face, i, j,
    /// orientation)`.
    pub fn to_face_ij_orientation(&self) -> (i32, i32, i32, i32) {
        let face = self.face();
        let mut bits = face & SWAP_MASK;
        let (mut i, mut j) = (0i32, 0i32);
        let l = lookup();
        for k in (0..=7).rev() {
            let nbits = if k == 7 {
                MAX_LEVEL - 7 * LOOKUP_BITS
            } else {
                LOOKUP_BITS
            };
            bits += ((((self.id as u64) >> (k * 2 * LOOKUP_BITS + 1)) as i32)
                & ((1 << (2 * nbits)) - 1))
                << 2;
            bits = l.ij[bits as usize];
            i = i.wrapping_add((bits >> (LOOKUP_BITS + 2)) << (k * LOOKUP_BITS));
            j = j.wrapping_add(((bits >> 2) & ((1 << LOOKUP_BITS) - 1)) << (k * LOOKUP_BITS));
            bits &= SWAP_MASK | INVERT_MASK;
        }
        if (self.lowest_on_bit() & 0x1111_1111_1111_1110) != 0 {
            bits ^= SWAP_MASK;
        }
        (face, i, j, bits)
    }

    /// `lowestOnBit()`.
    pub fn lowest_on_bit(&self) -> i64 {
        self.id & self.id.wrapping_neg()
    }

    /// `lowestOnBitForLevel(level)`.
    pub fn lowest_on_bit_for_level(level: i32) -> i64 {
        1i64 << (2 * (MAX_LEVEL - level))
    }
}

/// `stToIJ(s)`.
fn st_to_ij(s: f64) -> i32 {
    let m = (MAX_SIZE / 2) as f64;
    let r = java_round(m * s + (m - 0.5));
    0i64.max(((2 * (MAX_SIZE / 2) - 1) as i64).min(r)) as i32
}

impl PartialOrd for S2CellId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for S2CellId {
    /// `compareTo`: unsigned.
    fn cmp(&self, other: &Self) -> Ordering {
        (self.id as u64).cmp(&(other.id as u64))
    }
}

impl fmt::Display for S2CellId {
    /// `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(face={}, pos={:x}, level={})",
            self.face(),
            self.pos(),
            self.level()
        )
    }
}

/// `S2Cell`'s geometry: the face, level, orientation and the (u, v) bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct S2Cell {
    pub face: i32,
    pub level: i32,
    pub orientation: i32,
    pub cell_id: S2CellId,
    /// `uv[d][0..2]`: the cell's u (d = 0) and v (d = 1) range.
    pub uv: [[f64; 2]; 2],
}

impl S2Cell {
    /// `new S2Cell(id)`.
    pub fn new(id: S2CellId) -> Self {
        let max_cell_size = 1i32 << MAX_LEVEL;
        let (face, i, j, orientation) = id.to_face_ij_orientation();
        let level = id.level();
        let cell_size = 1i32 << (MAX_LEVEL - level);
        let mut uv = [[0.0; 2]; 2];
        for (d, ij) in [i, j].into_iter().enumerate() {
            let sij_lo = (ij & cell_size.wrapping_neg())
                .wrapping_mul(2)
                .wrapping_sub(max_cell_size);
            let sij_hi = sij_lo.wrapping_add(cell_size.wrapping_mul(2));
            uv[d][0] = projections::st_to_uv((1.0 / max_cell_size as f64) * sij_lo as f64);
            uv[d][1] = projections::st_to_uv((1.0 / max_cell_size as f64) * sij_hi as f64);
        }
        S2Cell {
            face: face as i8 as i32,
            level: level as i8 as i32,
            orientation: orientation as i8 as i32,
            cell_id: id,
            uv,
        }
    }

    /// `getVertexRaw(k)`: the k-th corner, counter-clockwise, not unit
    /// length.
    pub fn vertex_raw(&self, k: usize) -> S2Point {
        projections::face_uv_to_xyz(
            self.face,
            self.uv[0][(k >> 1) ^ (k & 1)],
            self.uv[1][k >> 1],
        )
    }

    /// `getVertex(k)`.
    pub fn vertex(&self, k: usize) -> S2Point {
        self.vertex_raw(k).normalize()
    }
}

#[cfg(test)]
mod tests;
