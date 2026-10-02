//! `ShapeDocValues`: the `BINARY` doc value of a `LatLonShape` / `XYShape`
//! -- the shape's triangles as a serialized, balanced 2-d tree that a query
//! can relate a `Component2D` to without decoding every triangle.
//!
//! # The format (version 0)
//!
//! ```text
//! byte    version (0)
//! vint    numberOfTerms (triangles)
//! vlong x4  root bounding box minX, maxX, minY, maxY, each as (long) v - Integer.MIN_VALUE
//! vlong x2  centroid x, y (encoded), the same way
//! vint    highest dimension type (POINT 0, LINE 1, TRIANGLE 2)
//! vint    root header
//! component  root triangle, relative to the root's maxX/maxY
//! node*   every other node, depth first (pre-order):
//!   vint    the node's byteSize (its bounds, header, component and subtrees)
//!   vlong x4  parent.maxX - minX, parent.maxY - minY, parent.maxX - maxX, parent.maxY - maxY
//!   vint    header
//!   component  relative to the parent's maxX/maxY
//! ```
//!
//! A header's bits: `0x01` a right subtree, `0x02` a left one, `0x04` the
//! component is a point, `0x08` a line (neither: a triangle), `0x10`/`0x20`/
//! `0x40` the `ab`/`bc`/`ca` edges belong to the shape. A component is
//! `pMaxX - aX, pMaxY - aY` (a point), then `b` (a line or triangle), then
//! `c` (a triangle), each a vlong.
//!
//! The tree is `ComponentTree`'s: each level selects the median triangle by
//! `minX, maxX` or `minY, maxY` (`ArrayUtil.select`, an `IntroSelector`,
//! ported swap for swap since the arrangement is on disk) and pulls its
//! children's bounds up. The centroid is the triangles' area-weighted (lines:
//! length-weighted; points: plain) average in the encoding's decoded space.
//!
//! # Rust shapes
//!
//! - Java's abstract `ShapeDocValues` with `LatLonShapeDocValues` /
//!   `XYShapeDocValues` subclasses supplying an `Encoder` is
//!   [`ShapeDocValues`] with a [`ShapeEncoding`]; the subclasses (which add
//!   the decoded centroid and bounding box) live in [`super::shape`].
//! - Java's `IOException`s, `Math.toIntExact`'s `ArithmeticException`, and
//!   the `ArrayIndexOutOfBoundsException` of an unknown type ordinal are
//!   [`Error`](super::Error)s here; so is a skip outside the value (Java's
//!   `ByteArrayDataInput.skipBytes` moves without checking), and so is a
//!   tree nested more than [`MAX_DEPTH`] deep -- a valid tree is balanced,
//!   so this only refuses a corrupt one, which Java would recurse into until
//!   its stack overflowed.
//! - `relate` does not re-read the header it already read when the value
//!   was opened (Java's `ShapeComparator` rewinds and reads it again); the
//!   bytes, and so the answer, are the same.
//! - `IntroSelector`'s pathological-input shuffle uses a fixed seed (as
//!   `ComponentTree`'s port does); Java's is unseeded, so Java itself does
//!   not reproduce its own bytes in that case.

use std::borrow::Cow;

use lucene_util::geo::{Component2D, GeoEncodingUtils, GeoError, Relation, XYEncodingUtils};
use lucene_util::sorter::{intro_select, IntroTarget};
use lucene_util::splittable_random::SplittableRandom;
use lucene_util::strict_math;

use super::shape::{DecodedTriangle, TriangleType};
use super::{illegal, Result};

/// `VERSION`: the doc-value format version.
pub const VERSION: u8 = 0;

/// The deepest tree [`ShapeDocValues::relate`] descends. A tree of `n`
/// triangles built by median selection is about `log2(n)` deep; 128 is far
/// past any `int` count.
pub const MAX_DEPTH: u32 = 128;

fn geo(e: GeoError) -> super::Error {
    illegal(e.to_string())
}

fn corrupt(what: &str) -> super::Error {
    illegal(format!(
        "unable to read binary shape doc value field. {what}"
    ))
}

/// `ShapeDocValues.Encoder`: the coordinate encoding of the shape --
/// `GeoEncodingUtils` (`x` longitude, `y` latitude) or `XYEncodingUtils`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShapeEncoding {
    /// `LatLonShapeDocValues`' encoder.
    LatLon,
    /// `XYShapeDocValues`' encoder.
    XY,
}

impl ShapeEncoding {
    /// `encodeX(double)`.
    ///
    /// # Errors
    /// A value outside the encoding's range.
    pub fn encode_x(self, x: f64) -> Result<i32> {
        match self {
            ShapeEncoding::LatLon => GeoEncodingUtils::encode_longitude(x).map_err(geo),
            ShapeEncoding::XY => XYEncodingUtils::encode(x as f32).map_err(geo),
        }
    }

    /// `encodeY(double)`.
    ///
    /// # Errors
    /// A value outside the encoding's range.
    pub fn encode_y(self, y: f64) -> Result<i32> {
        match self {
            ShapeEncoding::LatLon => GeoEncodingUtils::encode_latitude(y).map_err(geo),
            ShapeEncoding::XY => XYEncodingUtils::encode(y as f32).map_err(geo),
        }
    }

    /// `decodeX(int)`.
    #[inline]
    pub fn decode_x(self, encoded: i32) -> f64 {
        match self {
            ShapeEncoding::LatLon => GeoEncodingUtils::decode_longitude(encoded),
            ShapeEncoding::XY => f64::from(XYEncodingUtils::decode(encoded)),
        }
    }

    /// `decodeY(int)`.
    #[inline]
    pub fn decode_y(self, encoded: i32) -> f64 {
        match self {
            ShapeEncoding::LatLon => GeoEncodingUtils::decode_latitude(encoded),
            ShapeEncoding::XY => f64::from(XYEncodingUtils::decode(encoded)),
        }
    }
}

/// `vLongSize(long)`: the bytes of `i` as a vlong.
// ARITH: `size` counts the 7-bit groups of a 64-bit value, at most 10.
#[allow(clippy::arithmetic_side_effects)]
pub fn v_long_size(i: i64) -> i32 {
    let mut i = i as u64;
    let mut size = 1;
    while i & !0x7F != 0 {
        i >>= 7;
        size += 1;
    }
    size
}

/// `vIntSize(int)`: the bytes of `i` as a vint.
// ARITH: `size` counts the 7-bit groups of a 32-bit value, at most 5.
#[allow(clippy::arithmetic_side_effects)]
pub fn v_int_size(i: i32) -> i32 {
    let mut i = i as u32;
    let mut size = 1;
    while i & !0x7F != 0 {
        i >>= 7;
        size += 1;
    }
    size
}

/// `(long) a - b` for two `int`s, which cannot overflow a `long`.
// ARITH: two `i32`s widened to `i64` differ by less than 2^32.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
fn diff(a: i32, b: i32) -> i64 {
    i64::from(a) - i64::from(b)
}

/// `(long) v - Integer.MIN_VALUE`: an `int` as a non-negative `long`.
#[inline]
fn translate(v: i32) -> i64 {
    diff(v, i32::MIN)
}

// ---------------------------------------------------------------- writing

/// `ShapeDocValues.TreeNode`.
struct TreeNode {
    triangle: DecodedTriangle,
    /// centroid running stats (in encoded space) for this tree node
    mid_x: f64,
    mid_y: f64,
    signed_area: f64,
    length: f64,
    highest_type: TriangleType,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    left: Option<usize>,
    right: Option<usize>,
    parent: Option<usize>,
    /// header size is one byte; remaining is accumulated on construction
    byte_size: i32,
}

impl TreeNode {
    fn new(t: DecodedTriangle, encoder: ShapeEncoding) -> TreeNode {
        let ax = encoder.decode_x(t.a_x);
        let ay = encoder.decode_y(t.a_y);
        let (mid_x, mid_y, signed_area, length) = match t.kind {
            TriangleType::Point => (ax, ay, 0.0, 0.0),
            TriangleType::Line => {
                let bx = encoder.decode_x(t.b_x);
                let by = encoder.decode_y(t.b_y);
                let length = strict_math::hypot(ax - bx, ay - by);
                // weight by length
                (
                    (0.5 * (ax + bx)) * length,
                    (0.5 * (ay + by)) * length,
                    0.0,
                    length,
                )
            }
            TriangleType::Triangle => {
                let bx = encoder.decode_x(t.b_x);
                let by = encoder.decode_y(t.b_y);
                let cx = encoder.decode_x(t.c_x);
                let cy = encoder.decode_y(t.c_y);
                let signed_area = (0.5 * ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))).abs();
                // weight by signed area
                (
                    ((ax + bx + cx) / 3.0) * signed_area,
                    ((ay + by + cy) / 3.0) * signed_area,
                    signed_area,
                    0.0,
                )
            }
        };
        TreeNode {
            min_x: t.a_x.min(t.b_x).min(t.c_x),
            min_y: t.a_y.min(t.b_y).min(t.c_y),
            max_x: t.a_x.max(t.b_x).max(t.c_x),
            max_y: t.a_y.max(t.b_y).max(t.c_y),
            triangle: t,
            mid_x,
            mid_y,
            signed_area,
            length,
            highest_type: t.kind,
            left: None,
            right: None,
            parent: None,
            byte_size: 1,
        }
    }
}

/// `ArrayUtil.select` over the nodes (by index), ordered by
/// `comparingInt(minX).thenComparingInt(maxX)` or the `y` equivalent.
struct Select<'a> {
    nodes: &'a [TreeNode],
    order: &'a mut [usize],
    split_x: bool,
    pivot: (i32, i32),
}

impl Select<'_> {
    fn key(&self, i: usize) -> (i32, i32) {
        let n = &self.nodes[self.order[i]];
        if self.split_x {
            (n.min_x, n.max_x)
        } else {
            (n.min_y, n.max_y)
        }
    }
}

impl IntroTarget for Select<'_> {
    fn swap(&mut self, i: usize, j: usize) {
        self.order.swap(i, j);
    }
    fn set_pivot(&mut self, i: usize) {
        self.pivot = self.key(i);
    }
    // SENTINEL: none -- `-1` is "less than", a comparator result.
    fn compare_pivot(&mut self, j: usize) -> i32 {
        match self.pivot.cmp(&self.key(j)) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }
    }
}

/// The tree under construction: the nodes and their depth-first order.
struct Builder {
    encoder: ShapeEncoding,
    nodes: Vec<TreeNode>,
    dfs: Vec<usize>,
}

impl Builder {
    /// `buildTree(tessellation, dfsSerialized)`: the root's index.
    fn build_tree(&mut self, tessellation: &[DecodedTriangle]) -> Result<usize> {
        if let [t] = tessellation {
            let mut node = TreeNode::new(*t, self.encoder);
            if t.kind == TriangleType::Line {
                if node.length != 0.0 {
                    node.mid_x /= node.length;
                    node.mid_y /= node.length;
                }
            } else if t.kind == TriangleType::Triangle && node.signed_area != 0.0 {
                node.mid_x /= node.signed_area;
                node.mid_y /= node.signed_area;
            }
            node.highest_type = t.kind;
            self.nodes.push(node);
            self.dfs.push(0);
            return Ok(0);
        }
        let mut min_y = i32::MAX;
        let mut min_x = i32::MAX;
        // running stats for computing centroid
        let mut total_signed_area = 0.0;
        let mut total_length = 0.0;
        let (mut num_x_pnt, mut num_y_pnt) = (0.0, 0.0);
        let (mut num_x_lin, mut num_y_lin) = (0.0, 0.0);
        let (mut num_x_ply, mut num_y_ply) = (0.0, 0.0);
        let mut highest_type = TriangleType::Point;
        for t in tessellation {
            let node = TreeNode::new(*t, self.encoder);
            // compute the bbox values up front
            min_y = min_y.min(node.min_y);
            min_x = min_x.min(node.min_x);
            // compute the running centroid stats
            total_signed_area += node.signed_area;
            total_length += node.length;
            match t.kind {
                TriangleType::Point => {
                    num_x_pnt += node.mid_x;
                    num_y_pnt += node.mid_y;
                }
                TriangleType::Line => {
                    if highest_type == TriangleType::Point {
                        highest_type = TriangleType::Line;
                    }
                    num_x_lin += node.mid_x;
                    num_y_lin += node.mid_y;
                }
                TriangleType::Triangle => {
                    highest_type = TriangleType::Triangle;
                    num_x_ply += node.mid_x;
                    num_y_ply += node.mid_y;
                }
            }
            self.nodes.push(node);
        }
        let count = self.nodes.len();
        let mut order: Vec<usize> = (0..count).collect();
        let root = self
            .create_tree(&mut order, 0, count, false, None)
            .ok_or_else(|| illegal("an empty tessellation has no shape doc value"))?;
        let r = &mut self.nodes[root];
        // pull up min values for the root node so the bbox is consistent
        r.min_y = min_y;
        r.min_x = min_x;
        // set the highest dimensional type
        r.highest_type = highest_type;
        // compute centroid values for the root node so the centroid is consistent
        match highest_type {
            TriangleType::Point => {
                // `numXPnt / i`: the count as a double.
                r.mid_x = num_x_pnt / count as f64;
                r.mid_y = num_y_pnt / count as f64;
            }
            TriangleType::Line => {
                // numerator is sum of segment midPoints times segment length
                r.mid_x = num_x_lin;
                r.mid_y = num_y_lin;
                if total_length != 0.0 {
                    r.mid_x /= total_length;
                    r.mid_y /= total_length;
                }
            }
            TriangleType::Triangle => {
                // numerator is sum of triangle centroids times triangle signed area
                r.mid_x = num_x_ply;
                r.mid_y = num_y_ply;
                if total_signed_area != 0.0 {
                    r.mid_x /= total_signed_area;
                    r.mid_y /= total_signed_area;
                }
            }
        }
        Ok(root)
    }

    /// `createTree(triangles, low, high, splitX, parent, dfsSerialized)`
    /// over `order[low..high_exclusive]` (Java's `high` is inclusive).
    // ARITH: `low < high_exclusive <= order.len()`, so `high_exclusive - 1`
    // does not underflow, `mid` lies in `low..=high`, and `mid + 1`, `high +
    // 1` are at most `order.len()`.
    #[allow(clippy::arithmetic_side_effects)]
    fn create_tree(
        &mut self,
        order: &mut [usize],
        low: usize,
        high_exclusive: usize,
        split_x: bool,
        parent: Option<usize>,
    ) -> Option<usize> {
        if low >= high_exclusive {
            return None;
        }
        let high = high_exclusive - 1;
        // add midpoint
        let mid = low + (high - low) / 2;
        if low < high {
            let mut select = Select {
                nodes: &self.nodes,
                order,
                split_x,
                pivot: (0, 0),
            };
            let mut random = SplittableRandom::new(0);
            intro_select(&mut select, low, high + 1, mid, &mut random);
        }
        let new_node = order[mid];
        self.dfs.push(new_node);
        self.nodes[new_node].parent = parent;
        // add children
        let left = self.create_tree(order, low, mid, !split_x, Some(new_node));
        let right = self.create_tree(order, mid + 1, high_exclusive, !split_x, Some(new_node));
        self.nodes[new_node].left = left;
        self.nodes[new_node].right = right;
        // pull up values to this node
        for child in [left, right].into_iter().flatten() {
            let (cx0, cy0, cx1, cy1) = {
                let c = &self.nodes[child];
                (c.min_x, c.min_y, c.max_x, c.max_y)
            };
            let n = &mut self.nodes[new_node];
            n.min_x = n.min_x.min(cx0);
            n.min_y = n.min_y.min(cy0);
            n.max_x = n.max_x.max(cx1);
            n.max_y = n.max_y.max(cy1);
        }
        // adjust byteSize based on new parent bbox values
        let (p_max_x, p_max_y) = (self.nodes[new_node].max_x, self.nodes[new_node].max_y);
        for child in [left, right].into_iter().flatten() {
            let c = &self.nodes[child];
            // bounding box size
            let mut size = c.byte_size;
            for d in [
                diff(p_max_x, c.min_x),
                diff(p_max_y, c.min_y),
                diff(p_max_x, c.max_x),
                diff(p_max_y, c.max_y),
            ] {
                size = size.wrapping_add(v_long_size(d));
            }
            // component size
            size = size.wrapping_add(component_size(&c.triangle, p_max_x, p_max_y));
            self.nodes[child].byte_size = size;
            // include byte size (if needed to be skipped)
            let n = &mut self.nodes[new_node];
            n.byte_size = n
                .byte_size
                .wrapping_add(v_int_size(size))
                .wrapping_add(size);
        }
        Some(new_node)
    }
}

/// `computeComponentSize(node, maxX, maxY)`.
fn component_size(t: &DecodedTriangle, max_x: i32, max_y: i32) -> i32 {
    let mut size = v_long_size(diff(max_x, t.a_x)).wrapping_add(v_long_size(diff(max_y, t.a_y)));
    if t.kind != TriangleType::Point {
        size = size
            .wrapping_add(v_long_size(diff(max_x, t.b_x)))
            .wrapping_add(v_long_size(diff(max_y, t.b_y)));
    }
    if t.kind == TriangleType::Triangle {
        size = size
            .wrapping_add(v_long_size(diff(max_x, t.c_x)))
            .wrapping_add(v_long_size(diff(max_y, t.c_y)));
    }
    size
}

/// `writeVInt`.
fn write_vint(out: &mut Vec<u8>, v: i32) {
    let mut v = v as u32;
    while v & !0x7F != 0 {
        out.push((v & 0x7F) as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// `writeVLong` of a non-negative value.
fn write_vlong(out: &mut Vec<u8>, v: i64) {
    debug_assert!(v >= 0, "negative vlong {v}");
    let mut v = v as u64;
    while v & !0x7F != 0 {
        out.push((v & 0x7F) as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// `Writer.writeComponent(node, pMaxX, pMaxY)`.
fn write_component(out: &mut Vec<u8>, t: &DecodedTriangle, p_max_x: i32, p_max_y: i32) {
    write_vlong(out, diff(p_max_x, t.a_x));
    write_vlong(out, diff(p_max_y, t.a_y));
    if t.kind != TriangleType::Point {
        write_vlong(out, diff(p_max_x, t.b_x));
        write_vlong(out, diff(p_max_y, t.b_y));
    }
    if t.kind == TriangleType::Triangle {
        write_vlong(out, diff(p_max_x, t.c_x));
        write_vlong(out, diff(p_max_y, t.c_y));
    }
}

/// `Writer.writeHeader(node)`.
fn write_header(out: &mut Vec<u8>, node: &TreeNode) {
    let mut header = 0x00;
    // write left / right subtree
    if node.right.is_some() {
        header |= 0x01;
    }
    if node.left.is_some() {
        header |= 0x02;
    }
    // write type
    match node.triangle.kind {
        TriangleType::Point => header |= 0x04,
        TriangleType::Line => header |= 0x08,
        TriangleType::Triangle => {}
    }
    // write edge member of original shape
    if node.triangle.ab {
        header |= 0x10;
    }
    if node.triangle.bc {
        header |= 0x20;
    }
    if node.triangle.ca {
        header |= 0x40;
    }
    write_vint(out, header);
}

/// `computeBinaryValue(tessellation)`: `buildTree` then `Writer`.
fn compute_binary_value(
    encoder: ShapeEncoding,
    tessellation: &[DecodedTriangle],
) -> Result<Vec<u8>> {
    let mut b = Builder {
        encoder,
        nodes: Vec::with_capacity(tessellation.len()),
        dfs: Vec::with_capacity(tessellation.len()),
    };
    let root = b.build_tree(tessellation)?;
    let mut out = Vec::new();
    // write encoding version
    out.push(VERSION);
    // write number of terms (triangles); Java's `List.size()`
    write_vint(
        &mut out,
        i32::try_from(b.dfs.len()).map_err(|_| illegal("too many triangles"))?,
    );
    // write root
    let r = &b.nodes[root];
    // write bounding box; convert to variable long by translating
    write_vlong(&mut out, translate(r.min_x));
    write_vlong(&mut out, translate(r.max_x));
    write_vlong(&mut out, translate(r.min_y));
    write_vlong(&mut out, translate(r.max_y));
    // write centroid
    write_vlong(&mut out, translate(encoder.encode_x(r.mid_x)?));
    write_vlong(&mut out, translate(encoder.encode_y(r.mid_y)?));
    // write highest dimensional type
    write_vint(&mut out, r.highest_type.ordinal());
    // write header
    write_header(&mut out, r);
    // write component
    write_component(&mut out, &r.triangle, r.max_x, r.max_y);
    for &i in &b.dfs[1..] {
        // writeNode: subtree total size, max bounds, header, component
        let node = &b.nodes[i];
        let parent = &b.nodes[node.parent.expect("every node but the root has a parent")];
        write_vint(&mut out, node.byte_size);
        write_vlong(&mut out, diff(parent.max_x, node.min_x));
        write_vlong(&mut out, diff(parent.max_y, node.min_y));
        write_vlong(&mut out, diff(parent.max_x, node.max_x));
        write_vlong(&mut out, diff(parent.max_y, node.max_y));
        write_header(&mut out, node);
        write_component(&mut out, &node.triangle, parent.max_x, parent.max_y);
    }
    Ok(out)
}

// ---------------------------------------------------------------- reading

/// `ShapeDocValues.Reader`: a cursor over the value, with Java's
/// `DataInput` varint rules.
#[derive(Clone)]
struct Reader<'b> {
    data: &'b [u8],
    pos: usize,
}

impl Reader<'_> {
    fn read_byte(&mut self) -> Result<u8> {
        let b = *self
            .data
            .get(self.pos)
            .ok_or_else(|| corrupt("read past EOF"))?;
        // `pos < data.len()`, so this cannot overflow.
        self.pos = self.pos.saturating_add(1);
        Ok(b)
    }

    /// `readVInt`: at most five bytes, the fifth's high nibble clear.
    // ARITH: `i < 5` indexes the bytes after `pos`, so `7 * i <= 28` and `pos
    // + i + 1 <= data.len()`.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn read_vint(&mut self) -> Result<i32> {
        let rest = self.data.get(self.pos..).unwrap_or_default();
        if let Some(w) = rest.first_chunk::<5>() {
            // Five bytes in view: a fixed-width loop the compiler unrolls.
            let mut v: u32 = 0;
            for (i, &b) in w.iter().enumerate() {
                if i == 4 && b & 0xF0 != 0 {
                    return Err(corrupt("Invalid vInt detected (too many bits)"));
                }
                v |= u32::from(b & 0x7F) << (7 * i);
                if b & 0x80 == 0 {
                    self.pos += i + 1;
                    return Ok(v as i32);
                }
            }
        }
        let mut v: u32 = 0;
        for (i, &b) in rest.iter().take(5).enumerate() {
            if i == 4 && b & 0xF0 != 0 {
                return Err(corrupt("Invalid vInt detected (too many bits)"));
            }
            v |= u32::from(b & 0x7F) << (7 * i);
            if b & 0x80 == 0 {
                self.pos += i + 1;
                return Ok(v as i32);
            }
        }
        Err(corrupt("read past EOF"))
    }

    /// `readVLong`: at most nine bytes, never negative.
    // ARITH: `i < 9` indexes the bytes after `pos`, so `7 * i <= 56` and `pos
    // + i + 1 <= data.len()`.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn read_vlong(&mut self) -> Result<i64> {
        let rest = self.data.get(self.pos..).unwrap_or_default();
        if let Some(w) = rest.first_chunk::<9>() {
            // Nine bytes in view: a fixed-width loop the compiler unrolls.
            let mut v: u64 = 0;
            for (i, &b) in w.iter().enumerate() {
                v |= u64::from(b & 0x7F) << (7 * i);
                if b & 0x80 == 0 {
                    self.pos += i + 1;
                    return Ok(v as i64);
                }
            }
            return Err(corrupt(
                "Invalid vLong detected (negative values disallowed)",
            ));
        }
        let mut v: u64 = 0;
        for (i, &b) in rest.iter().take(9).enumerate() {
            v |= u64::from(b & 0x7F) << (7 * i);
            if b & 0x80 == 0 {
                self.pos += i + 1;
                return Ok(v as i64);
            }
        }
        if rest.len() >= 9 {
            return Err(corrupt(
                "Invalid vLong detected (negative values disallowed)",
            ));
        }
        Err(corrupt("read past EOF"))
    }

    /// `Math.toIntExact(base - readVLong())`.
    #[inline]
    fn read_relative(&mut self, base: i32) -> Result<i32> {
        let v = self.read_vlong()?;
        // `v >= 0` and `base` is an `int`, so the `long` difference cannot
        // overflow; whether it fits an `int` is toIntExact's question.
        i64::from(base)
            .checked_sub(v)
            .and_then(|d| i32::try_from(d).ok())
            .ok_or_else(|| corrupt("integer overflow"))
    }

    /// `Math.toIntExact(readVLong() + Integer.MIN_VALUE)`.
    fn read_translated(&mut self) -> Result<i32> {
        let v = self.read_vlong()?;
        v.checked_add(i64::from(i32::MIN))
            .and_then(|d| i32::try_from(d).ok())
            .ok_or_else(|| corrupt("integer overflow"))
    }

    /// `skipBytes(count)`, refused outside the value.
    fn skip_bytes(&mut self, count: i32) -> Result<()> {
        let end = usize::try_from(count)
            .ok()
            .and_then(|c| self.pos.checked_add(c))
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| corrupt("skip outside the value"))?;
        self.pos = end;
        Ok(())
    }
}

/// `Reader.Header.readType(bits)`.
fn read_type(bits: i32) -> TriangleType {
    if bits & 0x04 == 0x04 {
        TriangleType::Point
    } else if bits & 0x08 == 0x08 {
        TriangleType::Line
    } else {
        TriangleType::Triangle
    }
}

/// `ShapeDocValues` (with its `ShapeComparator`): a shape doc value, its
/// header decoded, and the tree walk that relates it to a query.
#[derive(Debug, Clone)]
pub struct ShapeDocValues<'a> {
    data: Cow<'a, [u8]>,
    encoding: ShapeEncoding,
    number_of_terms: i32,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    centroid_x: i32,
    centroid_y: i32,
    highest_dimension: TriangleType,
    /// Where the header ends and the tree begins: `relate` starts there
    /// with the bounding box already read, where Java re-reads the header.
    header_end: usize,
}

impl<'a> ShapeDocValues<'a> {
    /// `ShapeDocValues(List<DecodedTriangle> tessellation)`: the value of
    /// these triangles.
    ///
    /// # Errors
    /// An empty tessellation (Java: a `NullPointerException`), or a
    /// centroid the encoding cannot encode.
    pub fn from_triangles(
        encoding: ShapeEncoding,
        tessellation: &[DecodedTriangle],
    ) -> Result<ShapeDocValues<'static>> {
        let data = compute_binary_value(encoding, tessellation)?;
        ShapeDocValues::from_bytes(encoding, Cow::Owned(data))
    }

    /// `ShapeDocValues(BytesRef binaryValue)`: `ShapeComparator`'s header
    /// read.
    ///
    /// # Errors
    /// A truncated or malformed header (Java: `unable to read binary shape
    /// doc value field.`), or an unknown dimension type.
    pub fn from_bytes(encoding: ShapeEncoding, data: Cow<'a, [u8]>) -> Result<Self> {
        let mut r = Reader {
            data: &data,
            pos: 0,
        };
        // the version is only asserted
        r.read_byte()?;
        let number_of_terms = r.read_vint()?;
        let min_x = r.read_translated()?;
        let max_x = r.read_translated()?;
        let min_y = r.read_translated()?;
        let max_y = r.read_translated()?;
        let centroid_x = r.read_translated()?;
        let centroid_y = r.read_translated()?;
        let ordinal = r.read_vint()?;
        let highest_dimension = TriangleType::from_ordinal(ordinal)
            .ok_or_else(|| corrupt(&format!("Index {ordinal} out of bounds for length 3")))?;
        let header_end = r.pos;
        Ok(ShapeDocValues {
            data,
            encoding,
            header_end,
            number_of_terms,
            min_x,
            max_x,
            min_y,
            max_y,
            centroid_x,
            centroid_y,
            highest_dimension,
        })
    }

    /// `binaryValue()`: the serialized value.
    pub fn binary_value(&self) -> &[u8] {
        &self.data
    }

    /// The coordinate encoding.
    pub fn encoding(&self) -> ShapeEncoding {
        self.encoding
    }

    /// `numberOfTerms()`: the number of triangles.
    pub fn number_of_terms(&self) -> i32 {
        self.number_of_terms
    }

    /// `getEncodedMinX()`.
    pub fn encoded_min_x(&self) -> i32 {
        self.min_x
    }

    /// `getEncodedMinY()`.
    pub fn encoded_min_y(&self) -> i32 {
        self.min_y
    }

    /// `getEncodedMaxX()`.
    pub fn encoded_max_x(&self) -> i32 {
        self.max_x
    }

    /// `getEncodedMaxY()`.
    pub fn encoded_max_y(&self) -> i32 {
        self.max_y
    }

    /// `getEncodedCentroidX()`.
    pub fn encoded_centroid_x(&self) -> i32 {
        self.centroid_x
    }

    /// `getEncodedCentroidY()`.
    pub fn encoded_centroid_y(&self) -> i32 {
        self.centroid_y
    }

    /// `getHighestDimension()`.
    pub fn highest_dimension(&self) -> TriangleType {
        self.highest_dimension
    }

    /// `relate(Component2D)`: how the shape relates to the query --
    /// `ShapeComparator.relate`. `CELL_INSIDE_QUERY` or
    /// `CELL_OUTSIDE_QUERY` straight from the bounding box when that
    /// decides; otherwise `CELL_CROSSES_QUERY` as soon as one triangle
    /// intersects the query, `CELL_OUTSIDE_QUERY` when none does.
    ///
    /// # Errors
    /// A truncated or corrupt tree.
    pub fn relate(&self, query: &dyn Component2D) -> Result<Relation> {
        Comparator {
            r: Reader {
                data: &self.data,
                pos: self.header_end,
            },
            encoder: self.encoding,
            query,
        }
        .relate_root(self.min_x, self.max_x, self.min_y, self.max_y)
    }
}

/// `ShapeComparator`'s walk over one value.
struct Comparator<'b, 'q> {
    r: Reader<'b>,
    encoder: ShapeEncoding,
    query: &'q dyn Component2D,
}

impl Comparator<'_, '_> {
    /// `relate(Component2D)`: the root. The header (version, number of
    /// terms, bounding box, centroid, highest dimension) was read when the
    /// value was opened; the walk starts after it.
    fn relate_root(
        &mut self,
        t_min_x: i32,
        t_max_x: i32,
        t_min_y: i32,
        t_max_y: i32,
    ) -> Result<Relation> {
        let (query, enc) = (self.query, self.encoder);
        // relate the query to the shape bounding box
        let r = query.relate(
            enc.decode_x(t_min_x),
            enc.decode_x(t_max_x),
            enc.decode_y(t_min_y),
            enc.decode_y(t_max_y),
        );
        if r != Relation::CellCrossesQuery {
            return Ok(r);
        }
        // traverse the tessellation tree
        // get the header
        let header_bits = self.r.read_vint()?;
        let x = self.r.read_relative(t_max_x)?;
        // relate the component
        if self.relate_component(read_type(header_bits), t_max_x, t_max_y, enc.decode_x(x))?
            == Relation::CellCrossesQuery
        {
            return Ok(Relation::CellCrossesQuery);
        }
        let mut r = Relation::CellOutsideQuery;
        // recurse the left subtree
        if header_bits & 0x02 == 0x02 {
            let size = self.r.read_vint()?;
            r = self.relate(false, t_max_x, t_max_y, size, 1)?;
            if r == Relation::CellCrossesQuery {
                return Ok(Relation::CellCrossesQuery);
            }
        }
        // recurse the right subtree
        if header_bits & 0x01 == 0x01 && query.max_x() >= enc.decode_x(t_min_x) {
            let size = self.r.read_vint()?;
            r = self.relate(false, t_max_x, t_max_y, size, 1)?;
            if r == Relation::CellCrossesQuery {
                return Ok(Relation::CellCrossesQuery);
            }
        }
        Ok(r)
    }

    /// `relate(queryComponent2D, splitX, pMaxX, pMaxY, nodeSize)`: one
    /// subtree.
    fn relate(
        &mut self,
        split_x: bool,
        p_max_x: i32,
        p_max_y: i32,
        node_size: i32,
        depth: u32,
    ) -> Result<Relation> {
        if depth > MAX_DEPTH {
            return Err(corrupt("tree too deep"));
        }
        let (query, enc) = (self.query, self.encoder);
        // mark the data position because we need to subtract the maxX, maxY,
        // and header from node bytesize
        let pre_pos = self.r.pos;
        // read the minX and minY
        let t_min_x = self.r.read_relative(p_max_x)?;
        let t_min_y = self.r.read_relative(p_max_y)?;
        // read the maxX and maxY
        let t_max_x = self.r.read_relative(p_max_x)?;
        let t_max_y = self.r.read_relative(p_max_y)?;
        // read the header
        let header_bits = self.r.read_vint()?;
        // subtract the bbox and header byteSize to get remaining node
        // byteSize; at most 41 bytes were read, so the count fits an int
        let read = i32::try_from(self.r.pos.saturating_sub(pre_pos)).unwrap_or(i32::MAX);
        let node_size = node_size.wrapping_sub(read);

        // base case query is disjoint
        if query.min_x() > enc.decode_x(t_max_x) || query.min_y() > enc.decode_y(t_max_y) {
            // now skip the entire subtree
            self.r.skip_bytes(node_size)?;
            return Ok(Relation::CellOutsideQuery);
        }

        // traverse the tessellation tree
        let x = self.r.read_relative(p_max_x)?;
        // relate the component
        if self.relate_component(read_type(header_bits), p_max_x, p_max_y, enc.decode_x(x))?
            == Relation::CellCrossesQuery
        {
            return Ok(Relation::CellCrossesQuery);
        }

        // traverse left subtree
        if header_bits & 0x02 == 0x02 {
            let size = self.r.read_vint()?;
            if self.relate(!split_x, t_max_x, t_max_y, size, depth.saturating_add(1))?
                == Relation::CellCrossesQuery
            {
                return Ok(Relation::CellCrossesQuery);
            }
        }

        // traverse right subtree
        if header_bits & 0x01 == 0x01 {
            let size = self.r.read_vint()?;
            if (!split_x && query.max_y() >= enc.decode_y(t_min_y))
                || (split_x && query.max_x() >= enc.decode_x(t_min_x))
            {
                if self.relate(!split_x, t_max_x, t_max_y, size, depth.saturating_add(1))?
                    == Relation::CellCrossesQuery
                {
                    return Ok(Relation::CellCrossesQuery);
                }
            } else {
                // skip the subtree if the bbox doesn't match
                self.r.skip_bytes(size)?;
            }
        }
        Ok(Relation::CellOutsideQuery)
    }

    /// `relateComponent(type, bbox, pMaxX, pMaxY, x, queryComponent2D)`:
    /// whether this node's own point, line or triangle intersects the query
    /// (`CELL_CROSSES_QUERY`) or not (`CELL_OUTSIDE_QUERY`).
    fn relate_component(
        &mut self,
        kind: TriangleType,
        p_max_x: i32,
        p_max_y: i32,
        ax: f64,
    ) -> Result<Relation> {
        let (query, enc) = (self.query, self.encoder);
        let hit = match kind {
            TriangleType::Point => {
                let y = self.r.read_relative(p_max_y)?;
                query.contains(ax, enc.decode_y(y))
            }
            TriangleType::Line => {
                let ay = self.r.read_relative(p_max_y)?;
                let bx = enc.decode_x(self.r.read_relative(p_max_x)?);
                let by = self.r.read_relative(p_max_y)?;
                query.intersects_line(ax, enc.decode_y(ay), bx, enc.decode_y(by))
            }
            TriangleType::Triangle => {
                let ay = self.r.read_relative(p_max_y)?;
                let bx = enc.decode_x(self.r.read_relative(p_max_x)?);
                let by = self.r.read_relative(p_max_y)?;
                let cx = enc.decode_x(self.r.read_relative(p_max_x)?);
                let cy = self.r.read_relative(p_max_y)?;
                query.intersects_triangle(
                    ax,
                    enc.decode_y(ay),
                    bx,
                    enc.decode_y(by),
                    cx,
                    enc.decode_y(cy),
                )
            }
        };
        Ok(if hit {
            Relation::CellCrossesQuery
        } else {
            Relation::CellOutsideQuery
        })
    }
}

#[cfg(test)]
mod tests;
