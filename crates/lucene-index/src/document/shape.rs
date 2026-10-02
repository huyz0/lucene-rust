//! The shape fields: `ShapeField` (the triangle encoding), `LatLonShape` and
//! `XYShape` (the field factories), and the shape doc-values fields
//! `LatLonShapeDocValuesField` / `XYShapeDocValuesField` (whose bytes
//! [`super::shape_doc_values`] builds and reads).
//!
//! What reaches disk:
//!
//! - Every indexed shape is a set of triangles -- a polygon's
//!   `Tessellator` mesh, a line's segments as "flat" triangles `(a, b, a)`,
//!   a point as `(p, p, p)` -- and each triangle is one value of a seven
//!   dimension, four-byte point whose first four dimensions are indexed
//!   (`ShapeField.TYPE`, `setDimensions(7, 4, 4)`). The value is
//!   `encodeTriangle`'s: the triangle rotated so its westmost vertex comes
//!   first and turned counter-clockwise, then `minY, minX, maxY, maxX` (the
//!   four indexed dimensions: its bounding box), the one remaining vertex
//!   coordinate pair `y, x` that the box does not already give, and a word
//!   whose low three bits say which of eight vertex layouts to rebuild and
//!   whose bits 3..5 are the three edges' "belongs to the original shape"
//!   flags. Each `int` is `NumericUtils.intToSortableBytes` (big-endian,
//!   sign bit flipped). Lat/lon shapes encode `x` = longitude, `y` =
//!   latitude with `GeoEncodingUtils`; cartesian ones `XYEncodingUtils`.
//! - A shape doc value is one `BINARY` value per document: the
//!   [`ShapeDocValues`] tree of the shape's triangles.
//!
//! The query side (`LatLonShapeQuery`, `XYShapeQuery`, the bounding-box and
//! doc-values queries, `SpatialQuery`'s shape visitors) lives in
//! `lucene_search::document::geo`, which decodes what this writes.
//!
//! # Rust shapes
//!
//! - `ShapeField.Triangle` is [`ShapeTriangle`]; the `Field[]` the
//!   factories return are `Vec<ShapeTriangle>`.
//! - `decodeTriangle(byte[], DecodedTriangle)` fills a scratch object in
//!   Java; [`ShapeField::decode_triangle`] returns the (`Copy`) value.
//! - Java's overloads are separate functions named for their argument
//!   (`create_line_fields`, `create_point_doc_value_field`, ...).

use std::borrow::Cow;

use lucene_analysis::Analyzer;
use lucene_util::geo::tessellator::{self, Triangle as TessellatedTriangle};
use lucene_util::geo::{
    GeoEncodingUtils, GeoError, GeoUtils, Line, Point, Polygon, Rectangle, XYEncodingUtils, XYLine,
    XYPoint, XYPolygon, XYRectangle,
};

use super::numeric::{int_to_sortable_bytes, sortable_bytes_to_int};
use super::shape_doc_values::{ShapeDocValues, ShapeEncoding};
use super::{illegal, DocValuesType, FieldTokens, FieldType, IndexableField, Result};

fn geo(e: GeoError) -> super::Error {
    illegal(e.to_string())
}

/// `ShapeField`: the triangle encoding every shape field shares.
#[derive(Debug, Clone, Copy)]
pub struct ShapeField;

/// The width of an encoded triangle: seven four-byte dimensions.
pub const TRIANGLE_BYTES: usize = 7 * ShapeField::BYTES;

// `ShapeField`'s eight vertex layouts (the low three bits of the seventh
// dimension): which of the four bounding-box values and the two free
// values are vertex a, b and c.
const MINY_MINX_MAXY_MAXX_Y_X: i32 = 0;
const MINY_MINX_Y_X_MAXY_MAXX: i32 = 1;
const MAXY_MINX_Y_X_MINY_MAXX: i32 = 2;
const MAXY_MINX_MINY_MAXX_Y_X: i32 = 3;
const Y_MINX_MINY_X_MAXY_MAXX: i32 = 4;
const Y_MINX_MINY_MAXX_MAXY_X: i32 = 5;
const MAXY_MINX_MINY_X_Y_MAXX: i32 = 6;
const MINY_MINX_Y_MAXX_MAXY_X: i32 = 7;

/// `GeoUtils.orient` over encoded `int`s, which Java widens to `double`.
#[inline]
fn orient(a_x: i32, a_y: i32, b_x: i32, b_y: i32, c_x: i32, c_y: i32) -> i32 {
    GeoUtils::orient(
        f64::from(a_x),
        f64::from(a_y),
        f64::from(b_x),
        f64::from(b_y),
        f64::from(c_x),
        f64::from(c_y),
    )
}

/// The `int` at dimension `dim` of an encoded triangle.
// ARITH: every caller passes a constant `dim < 7`, so `at + 4 <= 28`.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
fn dim(t: &[u8; TRIANGLE_BYTES], dim: usize) -> i32 {
    let at = dim * ShapeField::BYTES;
    sortable_bytes_to_int(&t[at..at + ShapeField::BYTES])
}

impl ShapeField {
    /// `BYTES`: vertex coordinates are encoded as four-byte integers.
    pub const BYTES: usize = 4;

    /// `ShapeField.TYPE`: seven four-byte dimensions, the first four (the
    /// triangle's bounding box) indexed.
    pub fn field_type_of() -> FieldType {
        let mut ft = FieldType::new();
        ft.set_dimensions_with_index(7, 4, 4)
            .expect("seven four-byte dimensions, four indexed, are valid");
        ft.frozen()
    }

    /// `encodeTriangle(bytes, aY, aX, ab, bY, bX, bc, cY, cX, ca)`: the
    /// triangle rotated so its minimum `x` comes first (for three vertices
    /// on one meridian, so the middle one does not), made counter-clockwise,
    /// and packed as its bounding box, the remaining vertex coordinate pair
    /// and the layout/edge bits.
    ///
    /// # Errors
    /// A triangle none of the eight layouts can describe, with Java's
    /// message (`Could not encode the provided triangle`).
    #[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
    pub fn encode_triangle(
        a_y: i32,
        a_x: i32,
        ab: bool,
        b_y: i32,
        b_x: i32,
        bc: bool,
        c_y: i32,
        c_x: i32,
        ca: bool,
    ) -> Result<[u8; TRIANGLE_BYTES]> {
        let (mut a_x, mut a_y, mut ab) = (a_x, a_y, ab);
        let (mut b_x, mut b_y, mut bc) = (b_x, b_y, bc);
        let (mut c_x, mut c_y, mut ca) = (c_x, c_y, ca);
        // rotate edges and place minX at the beginning
        if b_x < a_x || c_x < a_x {
            let (temp_x, temp_y, temp_bool) = (a_x, a_y, ab);
            if b_x < c_x {
                a_x = b_x;
                a_y = b_y;
                ab = bc;
                b_x = c_x;
                b_y = c_y;
                bc = ca;
                c_x = temp_x;
                c_y = temp_y;
                ca = temp_bool;
            } else {
                a_x = c_x;
                a_y = c_y;
                ab = ca;
                c_x = b_x;
                c_y = b_y;
                ca = bc;
                b_x = temp_x;
                b_y = temp_y;
                bc = temp_bool;
            }
        } else if a_x == b_x && a_x == c_x {
            // degenerated case, all points with same longitude
            // we need to prevent that aX is in the middle (not part of the MBS)
            if b_y < a_y || c_y < a_y {
                let (temp_x, temp_y, temp_bool) = (a_x, a_y, ab);
                if b_y < c_y {
                    a_x = b_x;
                    a_y = b_y;
                    ab = bc;
                    b_x = c_x;
                    b_y = c_y;
                    bc = ca;
                    c_x = temp_x;
                    c_y = temp_y;
                    ca = temp_bool;
                } else {
                    a_x = c_x;
                    a_y = c_y;
                    ab = ca;
                    c_x = b_x;
                    c_y = b_y;
                    ca = bc;
                    b_x = temp_x;
                    b_y = temp_y;
                    bc = temp_bool;
                }
            }
        }

        // change orientation if CW
        if orient(a_x, a_y, b_x, b_y, c_x, c_y) == -1 {
            // swap b with c
            let (temp_x, temp_y, temp_bool) = (b_x, b_y, ab);
            // aX and aY do not change, ab becomes bc
            ab = bc;
            b_x = c_x;
            b_y = c_y;
            // bc does not change, ca becomes ab
            c_x = temp_x;
            c_y = temp_y;
            ca = temp_bool;
        }

        let min_x = a_x;
        let min_y = a_y.min(b_y.min(c_y));
        let max_x = a_x.max(b_x.max(c_x));
        let max_y = a_y.max(b_y.max(c_y));

        let (mut bits, x, y);
        if min_y == a_y {
            if max_y == b_y && max_x == b_x {
                y = c_y;
                x = c_x;
                bits = MINY_MINX_MAXY_MAXX_Y_X;
            } else if max_y == c_y && max_x == c_x {
                y = b_y;
                x = b_x;
                bits = MINY_MINX_Y_X_MAXY_MAXX;
            } else {
                y = b_y;
                x = c_x;
                bits = MINY_MINX_Y_MAXX_MAXY_X;
            }
        } else if max_y == a_y {
            if min_y == b_y && max_x == b_x {
                y = c_y;
                x = c_x;
                bits = MAXY_MINX_MINY_MAXX_Y_X;
            } else if min_y == c_y && max_x == c_x {
                y = b_y;
                x = b_x;
                bits = MAXY_MINX_Y_X_MINY_MAXX;
            } else {
                y = c_y;
                x = b_x;
                bits = MAXY_MINX_MINY_X_Y_MAXX;
            }
        } else if max_x == b_x && min_y == b_y {
            y = a_y;
            x = c_x;
            bits = Y_MINX_MINY_MAXX_MAXY_X;
        } else if max_x == c_x && max_y == c_y {
            y = a_y;
            x = b_x;
            bits = Y_MINX_MINY_X_MAXY_MAXX;
        } else {
            return Err(illegal("Could not encode the provided triangle"));
        }
        if ab {
            bits |= 1 << 3;
        }
        if bc {
            bits |= 1 << 4;
        }
        if ca {
            bits |= 1 << 5;
        }
        let mut bytes = [0u8; TRIANGLE_BYTES];
        for (out, v) in bytes
            .chunks_exact_mut(Self::BYTES)
            .zip([min_y, min_x, max_y, max_x, y, x, bits])
        {
            out.copy_from_slice(&int_to_sortable_bytes(v));
        }
        Ok(bytes)
    }

    /// `decodeTriangle(byte[], DecodedTriangle)`: the triangle
    /// [`Self::encode_triangle`] packed, its type resolved
    /// (`resolveTriangleType`). Every one of the eight layout codes decodes,
    /// so this cannot fail; an encoding no writer produced decodes to some
    /// triangle (Java asserts it is counter-clockwise, assertions off).
    #[inline]
    pub fn decode_triangle(t: &[u8; TRIANGLE_BYTES]) -> DecodedTriangle {
        let bits = dim(t, 6);
        // extract the first three bits
        let t_code = bits & 7;
        let (a_y, a_x, b_y, b_x, c_y, c_x) = match t_code {
            MINY_MINX_MAXY_MAXX_Y_X => (
                dim(t, 0),
                dim(t, 1),
                dim(t, 2),
                dim(t, 3),
                dim(t, 4),
                dim(t, 5),
            ),
            MINY_MINX_Y_X_MAXY_MAXX => (
                dim(t, 0),
                dim(t, 1),
                dim(t, 4),
                dim(t, 5),
                dim(t, 2),
                dim(t, 3),
            ),
            MAXY_MINX_Y_X_MINY_MAXX => (
                dim(t, 2),
                dim(t, 1),
                dim(t, 4),
                dim(t, 5),
                dim(t, 0),
                dim(t, 3),
            ),
            MAXY_MINX_MINY_MAXX_Y_X => (
                dim(t, 2),
                dim(t, 1),
                dim(t, 0),
                dim(t, 3),
                dim(t, 4),
                dim(t, 5),
            ),
            Y_MINX_MINY_X_MAXY_MAXX => (
                dim(t, 4),
                dim(t, 1),
                dim(t, 0),
                dim(t, 5),
                dim(t, 2),
                dim(t, 3),
            ),
            Y_MINX_MINY_MAXX_MAXY_X => (
                dim(t, 4),
                dim(t, 1),
                dim(t, 0),
                dim(t, 3),
                dim(t, 2),
                dim(t, 5),
            ),
            MAXY_MINX_MINY_X_Y_MAXX => (
                dim(t, 2),
                dim(t, 1),
                dim(t, 0),
                dim(t, 5),
                dim(t, 4),
                dim(t, 3),
            ),
            // MINY_MINX_Y_MAXX_MAXY_X: `bits & 7` has no other value.
            _ => (
                dim(t, 0),
                dim(t, 1),
                dim(t, 4),
                dim(t, 3),
                dim(t, 2),
                dim(t, 5),
            ),
        };
        let mut triangle = DecodedTriangle {
            a_x,
            a_y,
            b_x,
            b_y,
            c_x,
            c_y,
            ab: bits & (1 << 3) == 1 << 3,
            bc: bits & (1 << 4) == 1 << 4,
            ca: bits & (1 << 5) == 1 << 5,
            kind: TriangleType::Triangle,
        };
        Self::resolve_triangle_type(&mut triangle);
        triangle
    }

    /// `resolveTriangleType(triangle)`: a point when all three vertices
    /// coincide; a line when two do, the duplicate removed (`a`-`b` is the
    /// segment, `c` repeats `a`) and the removed edges' flags merged into
    /// `ab`; otherwise a triangle.
    pub fn resolve_triangle_type(triangle: &mut DecodedTriangle) {
        let t = triangle;
        if t.a_x == t.b_x && t.a_y == t.b_y {
            if t.a_x == t.c_x && t.a_y == t.c_y {
                t.kind = TriangleType::Point;
            } else {
                // a and b are identical, remove ab, and merge bc and ca
                t.ab = t.bc | t.ca;
                t.b_x = t.c_x;
                t.b_y = t.c_y;
                t.c_x = t.a_x;
                t.c_y = t.a_y;
                t.kind = TriangleType::Line;
            }
        } else if t.a_x == t.c_x && t.a_y == t.c_y {
            // a and c are identical, remove ac, and merge ab and bc
            t.ab |= t.bc;
            t.kind = TriangleType::Line;
        } else if t.b_x == t.c_x && t.b_y == t.c_y {
            // b and c are identical, remove bc, and merge ab and ca
            t.ab |= t.ca;
            t.c_x = t.a_x;
            t.c_y = t.a_y;
            t.kind = TriangleType::Line;
        } else {
            t.kind = TriangleType::Triangle;
        }
    }
}

/// `ShapeField.DecodedTriangle.TYPE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TriangleType {
    /// All coordinates are equal.
    Point,
    /// First and third coordinates are equal.
    Line,
    /// All coordinates are different.
    #[default]
    Triangle,
}

impl TriangleType {
    /// `ordinal()`.
    pub fn ordinal(self) -> i32 {
        match self {
            TriangleType::Point => 0,
            TriangleType::Line => 1,
            TriangleType::Triangle => 2,
        }
    }

    /// `TYPE.values()[ordinal]`: `None` outside the three.
    pub fn from_ordinal(ordinal: i32) -> Option<TriangleType> {
        match ordinal {
            0 => Some(TriangleType::Point),
            1 => Some(TriangleType::Line),
            2 => Some(TriangleType::Triangle),
            _ => None,
        }
    }
}

impl std::fmt::Display for TriangleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            TriangleType::Point => "POINT",
            TriangleType::Line => "LINE",
            TriangleType::Triangle => "TRIANGLE",
        })
    }
}

/// `ShapeField.DecodedTriangle`: a triangle's encoded vertices, which of
/// its edges belong to the original shape, and its type.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecodedTriangle {
    pub a_x: i32,
    pub a_y: i32,
    pub b_x: i32,
    pub b_y: i32,
    pub c_x: i32,
    pub c_y: i32,
    /// Edge `a`-`b` belongs to the original shape.
    pub ab: bool,
    /// Edge `b`-`c` belongs to the original shape.
    pub bc: bool,
    /// Edge `c`-`a` belongs to the original shape.
    pub ca: bool,
    /// `type`.
    pub kind: TriangleType,
}

impl DecodedTriangle {
    /// `setValues(aX, aY, ab, bX, bY, bc, cX, cY, ca)` on a new triangle of
    /// type `kind`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: TriangleType,
        a_x: i32,
        a_y: i32,
        ab: bool,
        b_x: i32,
        b_y: i32,
        bc: bool,
        c_x: i32,
        c_y: i32,
        ca: bool,
    ) -> Self {
        DecodedTriangle {
            a_x,
            a_y,
            b_x,
            b_y,
            c_x,
            c_y,
            ab,
            bc,
            ca,
            kind,
        }
    }
}

/// `equals`: vertices and edge flags; the type is not compared.
impl PartialEq for DecodedTriangle {
    fn eq(&self, o: &Self) -> bool {
        (self.a_x == o.a_x && self.b_x == o.b_x && self.c_x == o.c_x)
            && (self.a_y == o.a_y && self.b_y == o.b_y && self.c_y == o.c_y)
            && (self.ab == o.ab && self.bc == o.bc && self.ca == o.ca)
    }
}

impl Eq for DecodedTriangle {}

/// `toString`: `aX, aY bX, bY cX, cY [ab,bc,ca]`.
impl std::fmt::Display for DecodedTriangle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}, {} {}, {} {}, {} [{},{},{}]",
            self.a_x, self.a_y, self.b_x, self.b_y, self.c_x, self.c_y, self.ab, self.bc, self.ca
        )
    }
}

/// `ShapeField.Triangle`: one encoded triangle of a shape, a seven
/// dimension point value.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeTriangle {
    name: String,
    field_type: FieldType,
    packed: [u8; TRIANGLE_BYTES],
}

impl ShapeTriangle {
    /// `Triangle(name, aX, aY, bX, bY, cX, cY)`, the constructor for points
    /// and lines: every edge belongs to the shape.
    ///
    /// # Errors
    /// As [`ShapeField::encode_triangle`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        a_x: i32,
        a_y: i32,
        b_x: i32,
        b_y: i32,
        c_x: i32,
        c_y: i32,
    ) -> Result<Self> {
        Ok(ShapeTriangle {
            name: name.into(),
            field_type: ShapeField::field_type_of(),
            packed: ShapeField::encode_triangle(a_y, a_x, true, b_y, b_x, true, c_y, c_x, true)?,
        })
    }

    /// `Triangle(name, Tessellator.Triangle)`: a tessellated triangle with
    /// its edge flags.
    ///
    /// # Errors
    /// As [`ShapeField::encode_triangle`].
    pub fn from_tessellated(name: impl Into<String>, t: &TessellatedTriangle) -> Result<Self> {
        Ok(ShapeTriangle {
            name: name.into(),
            field_type: ShapeField::field_type_of(),
            packed: ShapeField::encode_triangle(
                t.encoded_y(0),
                t.encoded_x(0),
                t.is_edge_from_polygon(0),
                t.encoded_y(1),
                t.encoded_x(1),
                t.is_edge_from_polygon(1),
                t.encoded_y(2),
                t.encoded_x(2),
                t.is_edge_from_polygon(2),
            )?,
        })
    }

    /// `setTriangleValue(aX, aY, abFromShape, bX, bY, bcFromShape, cX, cY,
    /// caFromShape)`.
    ///
    /// # Errors
    /// As [`ShapeField::encode_triangle`]; the value is unchanged then.
    #[allow(clippy::too_many_arguments)]
    pub fn set_triangle_value(
        &mut self,
        a_x: i32,
        a_y: i32,
        ab_from_shape: bool,
        b_x: i32,
        b_y: i32,
        bc_from_shape: bool,
        c_x: i32,
        c_y: i32,
        ca_from_shape: bool,
    ) -> Result<()> {
        self.packed = ShapeField::encode_triangle(
            a_y,
            a_x,
            ab_from_shape,
            b_y,
            b_x,
            bc_from_shape,
            c_y,
            c_x,
            ca_from_shape,
        )?;
        Ok(())
    }

    /// The encoded triangle (`binaryValue()`).
    pub fn packed(&self) -> &[u8; TRIANGLE_BYTES] {
        &self.packed
    }

    /// The field name.
    pub fn field_name(&self) -> &str {
        &self.name
    }
}

impl IndexableField for ShapeTriangle {
    fn name(&self) -> &str {
        &self.name
    }
    fn field_type(&self) -> &FieldType {
        &self.field_type
    }
    fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
        Some(Cow::Borrowed(&self.packed))
    }
    fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
        Ok(None)
    }
}

/// The tessellation as the doc-values triangles `createDocValueField(name,
/// polygon)` builds: type `TRIANGLE` as tessellated, unrotated. Java passes
/// the first edge's flag for the second edge too (`isEdgefromPolygon(0)`
/// twice), and so does this, since the bytes on disk depend on it.
fn doc_value_triangles(tessellation: &[TessellatedTriangle]) -> Vec<DecodedTriangle> {
    tessellation
        .iter()
        .map(|t| {
            DecodedTriangle::new(
                TriangleType::Triangle,
                t.encoded_x(0),
                t.encoded_y(0),
                t.is_edge_from_polygon(0),
                t.encoded_x(1),
                t.encoded_y(1),
                t.is_edge_from_polygon(0),
                t.encoded_x(2),
                t.encoded_y(2),
                t.is_edge_from_polygon(2),
            )
        })
        .collect()
}

/// A line's "flat" triangles `(i, i + 1, i)` as doc-values triangles.
fn line_doc_value_triangles(points: &[(i32, i32)]) -> Vec<DecodedTriangle> {
    points
        .windows(2)
        .map(|w| {
            let ((ax, ay), (bx, by)) = (w[0], w[1]);
            DecodedTriangle::new(TriangleType::Line, ax, ay, true, bx, by, true, ax, ay, true)
        })
        .collect()
}

/// A line's "flat" triangle fields `(i, i + 1, i)`.
fn line_fields(name: &str, points: &[(i32, i32)]) -> Result<Vec<ShapeTriangle>> {
    points
        .windows(2)
        .map(|w| {
            let ((ax, ay), (bx, by)) = (w[0], w[1]);
            ShapeTriangle::new(name, ax, ay, bx, by, ax, ay)
        })
        .collect()
}

/// `createDocValueField(name, Field[] indexableFields)`: every field's
/// triangle decoded.
fn decoded(fields: &[ShapeTriangle]) -> Vec<DecodedTriangle> {
    fields
        .iter()
        .map(|f| ShapeField::decode_triangle(f.packed()))
        .collect()
}

/// `LatLonShape`: the factories of an indexed lat/lon shape (its triangle
/// fields) and of its doc value.
#[derive(Debug, Clone, Copy)]
pub struct LatLonShape;

impl LatLonShape {
    /// `createIndexableFields(fieldName, polygon)`.
    ///
    /// # Errors
    /// A polygon the `Tessellator` rejects, with its message.
    pub fn create_indexable_fields(
        field_name: &str,
        polygon: &Polygon,
    ) -> Result<Vec<ShapeTriangle>> {
        Self::create_indexable_fields_checked(field_name, polygon, false)
    }

    /// `createIndexableFields(fieldName, polygon, checkSelfIntersections)`.
    ///
    /// # Errors
    /// As [`Self::create_indexable_fields`].
    pub fn create_indexable_fields_checked(
        field_name: &str,
        polygon: &Polygon,
        check_self_intersections: bool,
    ) -> Result<Vec<ShapeTriangle>> {
        // the lionshare of the indexing is done by the tessellator
        let tessellation =
            tessellator::tessellate(polygon, check_self_intersections).map_err(geo)?;
        tessellation
            .iter()
            .map(|t| ShapeTriangle::from_tessellated(field_name, t))
            .collect()
    }

    /// `createIndexableFields(fieldName, line)`: one flat triangle per
    /// segment.
    ///
    /// # Errors
    /// Never for a valid [`Line`]; the encoding's errors otherwise.
    pub fn create_line_fields(field_name: &str, line: &Line) -> Result<Vec<ShapeTriangle>> {
        line_fields(field_name, &Self::encode_line(line)?)
    }

    /// `createIndexableFields(fieldName, lat, lon)`.
    ///
    /// # Errors
    /// An invalid latitude or longitude, with `GeoUtils`' message.
    pub fn create_point_fields(field_name: &str, lat: f64, lon: f64) -> Result<Vec<ShapeTriangle>> {
        let (x, y) = Self::encode(lat, lon)?;
        Ok(vec![ShapeTriangle::new(field_name, x, y, x, y, x, y)?])
    }

    /// `createDocValueField(fieldName, polygon)`.
    ///
    /// # Errors
    /// As [`Self::create_indexable_fields`].
    pub fn create_doc_value_field(
        field_name: &str,
        polygon: &Polygon,
    ) -> Result<LatLonShapeDocValuesField> {
        Self::create_doc_value_field_checked(field_name, polygon, false)
    }

    /// `createDocValueField(fieldName, polygon, checkSelfIntersections)`.
    ///
    /// # Errors
    /// As [`Self::create_indexable_fields`].
    pub fn create_doc_value_field_checked(
        field_name: &str,
        polygon: &Polygon,
        check_self_intersections: bool,
    ) -> Result<LatLonShapeDocValuesField> {
        let tessellation =
            tessellator::tessellate(polygon, check_self_intersections).map_err(geo)?;
        LatLonShapeDocValuesField::from_triangles(field_name, &doc_value_triangles(&tessellation))
    }

    /// `createDocValueField(fieldName, line)`.
    ///
    /// # Errors
    /// Never for a valid [`Line`].
    pub fn create_line_doc_value_field(
        field_name: &str,
        line: &Line,
    ) -> Result<LatLonShapeDocValuesField> {
        LatLonShapeDocValuesField::from_triangles(
            field_name,
            &line_doc_value_triangles(&Self::encode_line(line)?),
        )
    }

    /// `createDocValueField(fieldName, lat, lon)`.
    ///
    /// # Errors
    /// An invalid latitude or longitude.
    pub fn create_point_doc_value_field(
        field_name: &str,
        lat: f64,
        lon: f64,
    ) -> Result<LatLonShapeDocValuesField> {
        let (x, y) = Self::encode(lat, lon)?;
        let t = DecodedTriangle::new(TriangleType::Point, x, y, true, x, y, true, x, y, true);
        LatLonShapeDocValuesField::from_triangles(field_name, &[t])
    }

    /// `createDocValueField(fieldName, BytesRef binaryValue)`.
    ///
    /// # Errors
    /// Bytes that are not a shape doc value.
    pub fn create_doc_value_field_from_bytes(
        field_name: &str,
        binary_value: Vec<u8>,
    ) -> Result<LatLonShapeDocValuesField> {
        LatLonShapeDocValuesField::from_bytes(field_name, binary_value)
    }

    /// `createDocValueField(fieldName, List<DecodedTriangle> tessellation)`.
    ///
    /// # Errors
    /// An empty tessellation.
    pub fn create_doc_value_field_from_triangles(
        field_name: &str,
        tessellation: &[DecodedTriangle],
    ) -> Result<LatLonShapeDocValuesField> {
        LatLonShapeDocValuesField::from_triangles(field_name, tessellation)
    }

    /// `createDocValueField(fieldName, Field[] indexableFields)`: the doc
    /// value of a shape already turned into triangle fields -- how several
    /// geometries become one document's single doc value.
    ///
    /// # Errors
    /// No fields.
    pub fn create_doc_value_field_from_fields(
        field_name: &str,
        indexable_fields: &[ShapeTriangle],
    ) -> Result<LatLonShapeDocValuesField> {
        LatLonShapeDocValuesField::from_triangles(field_name, &decoded(indexable_fields))
    }

    /// `createLatLonShapeDocValues(BytesRef)`.
    ///
    /// # Errors
    /// Bytes that are not a shape doc value.
    pub fn create_lat_lon_shape_doc_values(bytes: &[u8]) -> Result<LatLonShapeDocValues<'_>> {
        LatLonShapeDocValues::new(Cow::Borrowed(bytes))
    }

    fn encode(lat: f64, lon: f64) -> Result<(i32, i32)> {
        Ok((
            GeoEncodingUtils::encode_longitude(lon).map_err(geo)?,
            GeoEncodingUtils::encode_latitude(lat).map_err(geo)?,
        ))
    }

    fn encode_line(line: &Line) -> Result<Vec<(i32, i32)>> {
        (0..line.num_points())
            .map(|i| Self::encode(line.lat(i), line.lon(i)))
            .collect()
    }
}

/// `XYShape`: the factories of an indexed cartesian shape and of its doc
/// value.
#[derive(Debug, Clone, Copy)]
pub struct XYShape;

impl XYShape {
    /// `createIndexableFields(fieldName, polygon)`.
    ///
    /// # Errors
    /// A polygon the `Tessellator` rejects, with its message.
    pub fn create_indexable_fields(
        field_name: &str,
        polygon: &XYPolygon,
    ) -> Result<Vec<ShapeTriangle>> {
        Self::create_indexable_fields_checked(field_name, polygon, false)
    }

    /// `createIndexableFields(fieldName, polygon, checkSelfIntersections)`.
    ///
    /// # Errors
    /// As [`Self::create_indexable_fields`].
    pub fn create_indexable_fields_checked(
        field_name: &str,
        polygon: &XYPolygon,
        check_self_intersections: bool,
    ) -> Result<Vec<ShapeTriangle>> {
        let tessellation =
            tessellator::tessellate_xy(polygon, check_self_intersections).map_err(geo)?;
        tessellation
            .iter()
            .map(|t| ShapeTriangle::from_tessellated(field_name, t))
            .collect()
    }

    /// `createIndexableFields(fieldName, line)`.
    ///
    /// # Errors
    /// Never for a valid [`XYLine`].
    pub fn create_line_fields(field_name: &str, line: &XYLine) -> Result<Vec<ShapeTriangle>> {
        line_fields(field_name, &Self::encode_line(line)?)
    }

    /// `createIndexableFields(fieldName, x, y)`.
    ///
    /// # Errors
    /// A coordinate that is not finite.
    pub fn create_point_fields(field_name: &str, x: f32, y: f32) -> Result<Vec<ShapeTriangle>> {
        let (x, y) = Self::encode(x, y)?;
        Ok(vec![ShapeTriangle::new(field_name, x, y, x, y, x, y)?])
    }

    /// `createDocValueField(fieldName, polygon)`.
    ///
    /// # Errors
    /// As [`Self::create_indexable_fields`].
    pub fn create_doc_value_field(
        field_name: &str,
        polygon: &XYPolygon,
    ) -> Result<XYShapeDocValuesField> {
        Self::create_doc_value_field_checked(field_name, polygon, false)
    }

    /// `createDocValueField(fieldName, polygon, checkSelfIntersections)`.
    ///
    /// # Errors
    /// As [`Self::create_indexable_fields`].
    pub fn create_doc_value_field_checked(
        field_name: &str,
        polygon: &XYPolygon,
        check_self_intersections: bool,
    ) -> Result<XYShapeDocValuesField> {
        let tessellation =
            tessellator::tessellate_xy(polygon, check_self_intersections).map_err(geo)?;
        XYShapeDocValuesField::from_triangles(field_name, &doc_value_triangles(&tessellation))
    }

    /// `createDocValueField(fieldName, line)`.
    ///
    /// # Errors
    /// Never for a valid [`XYLine`].
    pub fn create_line_doc_value_field(
        field_name: &str,
        line: &XYLine,
    ) -> Result<XYShapeDocValuesField> {
        XYShapeDocValuesField::from_triangles(
            field_name,
            &line_doc_value_triangles(&Self::encode_line(line)?),
        )
    }

    /// `createDocValueField(fieldName, x, y)`.
    ///
    /// # Errors
    /// A coordinate that is not finite.
    pub fn create_point_doc_value_field(
        field_name: &str,
        x: f32,
        y: f32,
    ) -> Result<XYShapeDocValuesField> {
        let (x, y) = Self::encode(x, y)?;
        let t = DecodedTriangle::new(TriangleType::Point, x, y, true, x, y, true, x, y, true);
        XYShapeDocValuesField::from_triangles(field_name, &[t])
    }

    /// `createDocValueField(fieldName, BytesRef binaryValue)`.
    ///
    /// # Errors
    /// Bytes that are not a shape doc value.
    pub fn create_doc_value_field_from_bytes(
        field_name: &str,
        binary_value: Vec<u8>,
    ) -> Result<XYShapeDocValuesField> {
        XYShapeDocValuesField::from_bytes(field_name, binary_value)
    }

    /// `createDocValueField(fieldName, List<DecodedTriangle> tessellation)`.
    ///
    /// # Errors
    /// An empty tessellation.
    pub fn create_doc_value_field_from_triangles(
        field_name: &str,
        tessellation: &[DecodedTriangle],
    ) -> Result<XYShapeDocValuesField> {
        XYShapeDocValuesField::from_triangles(field_name, tessellation)
    }

    /// The `Field[]` form of [`LatLonShape::create_doc_value_field_from_fields`];
    /// Java's `XYShape` has no such overload, but the triangles decode the
    /// same way.
    ///
    /// # Errors
    /// No fields.
    pub fn create_doc_value_field_from_fields(
        field_name: &str,
        indexable_fields: &[ShapeTriangle],
    ) -> Result<XYShapeDocValuesField> {
        XYShapeDocValuesField::from_triangles(field_name, &decoded(indexable_fields))
    }

    /// `createXYShapeDocValues(BytesRef)`.
    ///
    /// # Errors
    /// Bytes that are not a shape doc value.
    pub fn create_xy_shape_doc_values(bytes: &[u8]) -> Result<XYShapeDocValues<'_>> {
        XYShapeDocValues::new(Cow::Borrowed(bytes))
    }

    fn encode(x: f32, y: f32) -> Result<(i32, i32)> {
        Ok((
            XYEncodingUtils::encode(x).map_err(geo)?,
            XYEncodingUtils::encode(y).map_err(geo)?,
        ))
    }

    fn encode_line(line: &XYLine) -> Result<Vec<(i32, i32)>> {
        (0..line.num_points())
            .map(|i| Self::encode(line.x_at(i), line.y_at(i)))
            .collect()
    }
}

// ---------------------------------------------------------------- doc values

/// `LatLonShapeDocValues`: a lat/lon shape doc value, with its centroid
/// and bounding box decoded.
#[derive(Debug, Clone)]
pub struct LatLonShapeDocValues<'a> {
    values: ShapeDocValues<'a>,
    centroid: Point,
    bounding_box: Rectangle,
}

impl<'a> LatLonShapeDocValues<'a> {
    /// `LatLonShapeDocValues(BytesRef)`.
    ///
    /// # Errors
    /// Bytes that are not a shape doc value, or whose centroid or bounding
    /// box is not a valid point / rectangle.
    pub fn new(binary_value: Cow<'a, [u8]>) -> Result<Self> {
        Self::from_values(ShapeDocValues::from_bytes(
            ShapeEncoding::LatLon,
            binary_value,
        )?)
    }

    /// `LatLonShapeDocValues(List<DecodedTriangle>)`.
    ///
    /// # Errors
    /// An empty tessellation.
    pub fn from_triangles(
        tessellation: &[DecodedTriangle],
    ) -> Result<LatLonShapeDocValues<'static>> {
        LatLonShapeDocValues::from_values(ShapeDocValues::from_triangles(
            ShapeEncoding::LatLon,
            tessellation,
        )?)
    }

    fn from_values(values: ShapeDocValues<'a>) -> Result<Self> {
        let e = ShapeEncoding::LatLon;
        // computeCentroid / computeBoundingBox
        let centroid = Point::new(
            e.decode_y(values.encoded_centroid_y()),
            e.decode_x(values.encoded_centroid_x()),
        )
        .map_err(geo)?;
        let bounding_box = Rectangle::new(
            e.decode_y(values.encoded_min_y()),
            e.decode_y(values.encoded_max_y()),
            e.decode_x(values.encoded_min_x()),
            e.decode_x(values.encoded_max_x()),
        )
        .map_err(geo)?;
        Ok(LatLonShapeDocValues {
            values,
            centroid,
            bounding_box,
        })
    }

    /// `getCentroid()`.
    pub fn centroid(&self) -> &Point {
        &self.centroid
    }

    /// `getBoundingBox()`.
    pub fn bounding_box(&self) -> &Rectangle {
        &self.bounding_box
    }

    /// The shape doc value itself (`relate`, the encoded bounds, ...).
    pub fn values(&self) -> &ShapeDocValues<'a> {
        &self.values
    }
}

/// `XYShapeDocValues`: a cartesian shape doc value, with its centroid and
/// bounding box decoded.
#[derive(Debug, Clone)]
pub struct XYShapeDocValues<'a> {
    values: ShapeDocValues<'a>,
    centroid: XYPoint,
    bounding_box: XYRectangle,
}

impl<'a> XYShapeDocValues<'a> {
    /// `XYShapeDocValues(BytesRef)`.
    ///
    /// # Errors
    /// Bytes that are not a shape doc value, or whose bounding box is not a
    /// valid rectangle.
    pub fn new(binary_value: Cow<'a, [u8]>) -> Result<Self> {
        Self::from_values(ShapeDocValues::from_bytes(ShapeEncoding::XY, binary_value)?)
    }

    /// `XYShapeDocValues(List<DecodedTriangle>)`.
    ///
    /// # Errors
    /// An empty tessellation.
    pub fn from_triangles(tessellation: &[DecodedTriangle]) -> Result<XYShapeDocValues<'static>> {
        XYShapeDocValues::from_values(ShapeDocValues::from_triangles(
            ShapeEncoding::XY,
            tessellation,
        )?)
    }

    fn from_values(values: ShapeDocValues<'a>) -> Result<Self> {
        let e = ShapeEncoding::XY;
        // computeCentroid / computeBoundingBox: `(float)` of the decoded
        // doubles, which are floats already.
        let centroid = XYPoint::new(
            e.decode_x(values.encoded_centroid_x()) as f32,
            e.decode_y(values.encoded_centroid_y()) as f32,
        )
        .map_err(geo)?;
        let bounding_box = XYRectangle::new(
            e.decode_x(values.encoded_min_x()) as f32,
            e.decode_x(values.encoded_max_x()) as f32,
            e.decode_y(values.encoded_min_y()) as f32,
            e.decode_y(values.encoded_max_y()) as f32,
        )
        .map_err(geo)?;
        Ok(XYShapeDocValues {
            values,
            centroid,
            bounding_box,
        })
    }

    /// `getCentroid()`.
    pub fn centroid(&self) -> &XYPoint {
        &self.centroid
    }

    /// `getBoundingBox()`.
    pub fn bounding_box(&self) -> &XYRectangle {
        &self.bounding_box
    }

    /// The shape doc value itself.
    pub fn values(&self) -> &ShapeDocValues<'a> {
        &self.values
    }
}

/// `ShapeDocValuesField.FIELD_TYPE`: `BINARY` doc values, norms omitted.
fn shape_doc_values_type() -> FieldType {
    let mut ft = FieldType::new();
    ft.set_doc_values_type(DocValuesType::Binary)
        .expect("unfrozen");
    ft.set_omit_norms(true).expect("unfrozen");
    ft.frozen()
}

macro_rules! shape_dv_field {
    ($(#[$m:meta])* $ty:ident, $values:ident, $centroid:ty, $bbox:ty) => {
        $(#[$m])*
        #[derive(Debug, Clone)]
        pub struct $ty {
            name: String,
            field_type: FieldType,
            values: $values<'static>,
        }

        impl $ty {
            /// The constructor from a tessellation.
            ///
            /// # Errors
            /// An empty tessellation.
            pub fn from_triangles(name: &str, tessellation: &[DecodedTriangle]) -> Result<Self> {
                Ok($ty {
                    name: name.to_owned(),
                    field_type: shape_doc_values_type(),
                    values: $values::from_triangles(tessellation)?,
                })
            }

            /// The constructor from a serialized value.
            ///
            /// # Errors
            /// Bytes that are not a shape doc value.
            pub fn from_bytes(name: &str, binary_value: Vec<u8>) -> Result<Self> {
                Ok($ty {
                    name: name.to_owned(),
                    field_type: shape_doc_values_type(),
                    values: $values::new(Cow::Owned(binary_value))?,
                })
            }

            /// `ShapeDocValuesField.FIELD_TYPE`.
            pub fn field_type_of() -> FieldType {
                shape_doc_values_type()
            }

            /// `numberOfTerms()`: the number of triangles.
            pub fn number_of_terms(&self) -> i32 {
                self.values.values().number_of_terms()
            }

            /// `getCentroid()`.
            pub fn centroid(&self) -> &$centroid {
                self.values.centroid()
            }

            /// `getBoundingBox()`.
            pub fn bounding_box(&self) -> &$bbox {
                self.values.bounding_box()
            }

            /// `getHighestDimensionType()`.
            pub fn highest_dimension_type(&self) -> TriangleType {
                self.values.values().highest_dimension()
            }

            /// The doc value.
            pub fn doc_values(&self) -> &$values<'static> {
                &self.values
            }
        }

        impl IndexableField for $ty {
            fn name(&self) -> &str {
                &self.name
            }
            fn field_type(&self) -> &FieldType {
                &self.field_type
            }
            fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
                Some(Cow::Borrowed(self.values.values().binary_value()))
            }
            fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
                Ok(None)
            }
        }
    };
}

shape_dv_field!(
    /// `LatLonShapeDocValuesField`: a lat/lon shape as one `BINARY` doc
    /// value, for the slow doc-values shape queries.
    LatLonShapeDocValuesField,
    LatLonShapeDocValues,
    Point,
    Rectangle
);
shape_dv_field!(
    /// `XYShapeDocValuesField`: a cartesian shape as one `BINARY` doc value.
    XYShapeDocValuesField,
    XYShapeDocValues,
    XYPoint,
    XYRectangle
);

#[cfg(test)]
mod tests;
