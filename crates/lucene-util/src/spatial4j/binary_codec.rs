//! `BinaryCodec` (`org.locationtech.spatial4j.io`): a compact binary form
//! of points, rectangles, circles and shape collections -- a type byte,
//! then big-endian doubles (`DataOutputStream.writeDouble`, NaN
//! canonicalised).

use std::fmt;
use std::sync::Arc;

use super::context::SpatialContext;
use super::shape::Shape;
use super::{Error, Result};

const TYPE_POINT: u8 = 1;
const TYPE_RECT: u8 = 2;
const TYPE_CIRCLE: u8 = 3;
const TYPE_COLL: u8 = 4;

/// How deep shape collections may nest in a binary shape. Java recurses
/// until `StackOverflowError`; a Rust stack overflow aborts the process
/// (no `catch_unwind` sees it), so bytes nesting deeper -- six bytes a
/// level, a corrupt doc value's worth -- are that error instead.
pub const MAX_NESTING: u32 = 64;

/// A `DataInput` over bytes (big-endian, as `DataInputStream`).
#[derive(Debug, Clone)]
pub struct DataInput<'a> {
    bytes: &'a [u8],
    pos: usize,
    /// Collections being read, outermost first.
    depth: u32,
}

impl<'a> DataInput<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        DataInput {
            bytes,
            pos: 0,
            depth: 0,
        }
    }

    /// Bytes read so far.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// The unread bytes.
    pub fn remaining(&self) -> &'a [u8] {
        &self.bytes[self.pos..]
    }

    /// Skips `n` bytes (after another reader consumed them).
    pub fn advance(&mut self, n: usize) {
        self.pos = self.pos.saturating_add(n).min(self.bytes.len());
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self.pos.checked_add(N).filter(|&e| e <= self.bytes.len());
        let Some(end) = end else {
            return Err(Error::Io("java.io.EOFException".into()));
        };
        let mut b = [0u8; N];
        b.copy_from_slice(&self.bytes[self.pos..end]);
        self.pos = end;
        Ok(b)
    }

    /// `readByte()`.
    pub fn read_byte(&mut self) -> Result<i8> {
        Ok(self.take::<1>()?[0] as i8)
    }

    /// `readInt()`.
    pub fn read_int(&mut self) -> Result<i32> {
        Ok(i32::from_be_bytes(self.take::<4>()?))
    }

    /// `readDouble()`.
    pub fn read_double(&mut self) -> Result<f64> {
        Ok(f64::from_bits(u64::from_be_bytes(self.take::<8>()?)))
    }
}

/// `DataOutput.writeDouble(v)`: `doubleToLongBits` (canonical NaN),
/// big-endian.
pub fn write_double(out: &mut Vec<u8>, v: f64) {
    let bits = if v.is_nan() {
        0x7ff8_0000_0000_0000u64
    } else {
        v.to_bits()
    };
    out.extend_from_slice(&bits.to_be_bytes());
}

/// `BinaryCodec`: what `SpatialContext.getBinaryCodec()` returns.
pub trait BinaryCodec: Send + Sync + fmt::Debug {
    /// `readShape(dataInput)`.
    fn read_shape(
        &self,
        ctx: &Arc<SpatialContext>,
        input: &mut DataInput<'_>,
    ) -> Result<Arc<dyn Shape>>;
    /// `writeShape(dataOutput, s)`.
    fn write_shape(&self, out: &mut Vec<u8>, s: &dyn Shape) -> Result<()>;
}

/// Spatial4j's own `BinaryCodec`.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultBinaryCodec;

impl DefaultBinaryCodec {
    /// `readShapeByTypeIfSupported(dataInput, type)`.
    fn read_shape_by_type(
        &self,
        ctx: &Arc<SpatialContext>,
        input: &mut DataInput<'_>,
        ty: i8,
    ) -> Result<Option<Arc<dyn Shape>>> {
        Ok(Some(match ty as u8 {
            TYPE_POINT => {
                let x = input.read_double()?;
                let y = input.read_double()?;
                ctx.point_xy(x, y)?
            }
            TYPE_RECT => {
                let a = input.read_double()?;
                let b = input.read_double()?;
                let c = input.read_double()?;
                let d = input.read_double()?;
                ctx.rect(a, b, c, d)?
            }
            TYPE_CIRCLE => {
                let x = input.read_double()?;
                let y = input.read_double()?;
                let p = ctx.point_xy(x, y)?;
                let r = input.read_double()?;
                ctx.circle_at(&p, r)?
            }
            TYPE_COLL => {
                if input.depth >= MAX_NESTING {
                    return Err(Error::Runtime(format!(
                        "java.lang.StackOverflowError: shape collections nested deeper than {MAX_NESTING}"
                    )));
                }
                input.depth += 1;
                let shapes = self.read_collection(ctx, input);
                input.depth -= 1;
                Arc::new(ctx.collection(shapes?)?)
            }
            _ => return Ok(None),
        }))
    }

    /// A collection's members (its type byte, size, then the shapes);
    /// grown as read, so a size off the stream sizes no allocation.
    fn read_collection(
        &self,
        ctx: &Arc<SpatialContext>,
        input: &mut DataInput<'_>,
    ) -> Result<Vec<Arc<dyn Shape>>> {
        let ty = input.read_byte()?;
        let size = input.read_int()?;
        let mut shapes = Vec::new();
        for _ in 0..size.max(0) {
            if ty == 0 {
                shapes.push(self.read_shape(ctx, input)?);
            } else {
                match self.read_shape_by_type(ctx, input, ty)? {
                    Some(s) => shapes.push(s),
                    None => {
                        return Err(Error::InvalidShape(format!("Unsupported shape byte {ty}")))
                    }
                }
            }
        }
        Ok(shapes)
    }

    /// `typeForShape(s)`.
    fn type_for_shape(s: &dyn Shape) -> u8 {
        if s.as_point().is_some() {
            TYPE_POINT
        } else if s.as_rectangle().is_some() {
            TYPE_RECT
        } else if s.as_circle().is_some() {
            TYPE_CIRCLE
        } else if s
            .as_any()
            .downcast_ref::<super::collection::ShapeCollection>()
            .is_some()
        {
            TYPE_COLL
        } else {
            0
        }
    }
}

impl BinaryCodec for DefaultBinaryCodec {
    fn read_shape(
        &self,
        ctx: &Arc<SpatialContext>,
        input: &mut DataInput<'_>,
    ) -> Result<Arc<dyn Shape>> {
        let ty = input.read_byte()?;
        match self.read_shape_by_type(ctx, input, ty)? {
            Some(s) => Ok(s),
            None => Err(Error::IllegalArgument(format!(
                "Unsupported shape byte {ty}"
            ))),
        }
    }

    fn write_shape(&self, out: &mut Vec<u8>, s: &dyn Shape) -> Result<()> {
        let ty = Self::type_for_shape(s);
        out.push(ty);
        match ty {
            TYPE_POINT => {
                let p = s.as_point().expect("typed as a point");
                write_double(out, p.x());
                write_double(out, p.y());
            }
            TYPE_RECT => {
                let r = s.as_rectangle().expect("typed as a rectangle");
                write_double(out, r.min_x());
                write_double(out, r.max_x());
                write_double(out, r.min_y());
                write_double(out, r.max_y());
            }
            TYPE_CIRCLE => {
                let c = s.as_circle().expect("typed as a circle");
                let p = c.center()?;
                write_double(out, p.x());
                write_double(out, p.y());
                write_double(out, c.radius());
            }
            TYPE_COLL => {
                let col = s
                    .as_any()
                    .downcast_ref::<super::collection::ShapeCollection>()
                    .expect("typed as a collection");
                out.push(0);
                out.extend_from_slice(&(col.size() as i32).to_be_bytes());
                for shape in col.shapes() {
                    self.write_shape(out, &**shape)?;
                }
            }
            _ => {
                return Err(Error::IllegalArgument(format!(
                    "Unsupported shape class {}",
                    java_class_name(s)
                )))
            }
        }
        Ok(())
    }
}

/// The Java class name of a shape, for messages.
pub fn java_class_name(s: &dyn Shape) -> &'static str {
    use super::buffered_line::{BufferedLine, BufferedLineString};
    let any = s.as_any();
    if any.is::<BufferedLineString>() {
        "org.locationtech.spatial4j.shape.impl.BufferedLineString"
    } else if any.is::<BufferedLine>() {
        "org.locationtech.spatial4j.shape.impl.BufferedLine"
    } else if let Some(name) = crate::spatial_extras::spatial4j::geo3d_class_name(s) {
        name
    } else {
        "org.locationtech.spatial4j.shape.Shape"
    }
}
