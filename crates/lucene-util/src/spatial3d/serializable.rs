//! `SerializableObject`'s stream primitives
//! (`org.apache.lucene.spatial3d.geom.SerializableObject`): geo3d's own
//! little-endian encoding of ints, longs, doubles, booleans, strings, byte
//! arrays and bit sets, which every shape's `write` and stream constructor is
//! built from.
//!
//! The format, as Java writes it:
//!
//! ```text
//! int      4 bytes, little-endian
//! long     int(low 32 bits), int(high 32 bits)
//! double   long(Double.doubleToLongBits)      -- NaN canonicalized
//! boolean  1 byte, 0 or 1 (any non-zero reads as true)
//! bytes    int(length), the bytes
//! string   bytes(UTF-8)
//! bitset   bytes(BitSet.toByteArray())          -- little-endian, trailing zero bytes trimmed
//! class    boolean(true), byte(StandardObjects code)   or   boolean(false), string(class name)
//! ```
//!
//! Reading follows `InputStream.read()`: past the end it yields `-1`, which
//! `readInt` masks into `0xff` bytes as Java does; `readBoolean` and
//! `readByteArray` fail there with Java's message.

use super::jmath::double_to_long_bits;
use super::{Error, Result};

/// The read side of a stream: a cursor over bytes, `InputStream`-shaped.
pub struct Input<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Input<'a> {
    /// A stream over `bytes`.
    pub fn new(bytes: &'a [u8]) -> Input<'a> {
        Input { bytes, pos: 0 }
    }

    /// `InputStream.read()`: the next byte, or `-1` at the end.
    //
    // SENTINEL: `-1` = end of the stream. Every caller is in this module or
    // `standard_objects.rs`: `read_int` masks it into `0xff` bytes, as
    // Java's `readInt` does (a deliberate, Java-equal non-check);
    // `read_boolean` matches it as end of stream; `read_class` rejects it
    // through `u8::try_from`.
    pub fn read_byte(&mut self) -> i32 {
        match self.bytes.get(self.pos) {
            Some(&b) => {
                self.pos += 1;
                i32::from(b)
            }
            None => -1,
        }
    }

    /// Bytes not yet read.
    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }
}

fn eof() -> Error {
    Error::Io("Unexpected end of input stream".into())
}

/// `writeInt`.
pub fn write_int(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// `readInt`: four `read()`s, each masked, so the end of the stream reads as
/// `0xff` bytes rather than failing -- Java's behaviour.
pub fn read_int(input: &mut Input<'_>) -> Result<i32> {
    let l1 = input.read_byte() & 0x0000_00ff;
    let l2 = (input.read_byte() << 8) & 0x0000_ff00;
    let l3 = (input.read_byte() << 16) & 0x00ff_0000;
    let l4 = ((input.read_byte() as u32) << 24) as i32 & (0xff00_0000u32 as i32);
    Ok(l1.wrapping_add(l2).wrapping_add(l3).wrapping_add(l4))
}

/// `writeLong`: low int then high int.
pub fn write_long(out: &mut Vec<u8>, value: i64) {
    write_int(out, value as i32);
    write_int(out, (value >> 32) as i32);
}

/// `readLong`.
pub fn read_long(input: &mut Input<'_>) -> Result<i64> {
    let lower = i64::from(read_int(input)?) & 0x0000_0000_ffff_ffff;
    let upper = (i64::from(read_int(input)?) << 32) & (0xffff_ffff_0000_0000u64 as i64);
    Ok(lower.wrapping_add(upper))
}

/// `writeDouble`: `Double.doubleToLongBits`, so every NaN is written as the
/// canonical one.
pub fn write_double(out: &mut Vec<u8>, value: f64) {
    write_long(out, double_to_long_bits(value));
}

/// `readDouble`.
pub fn read_double(input: &mut Input<'_>) -> Result<f64> {
    Ok(f64::from_bits(read_long(input)? as u64))
}

/// `writeBoolean`.
pub fn write_boolean(out: &mut Vec<u8>, value: bool) {
    out.push(u8::from(value));
}

/// `readBoolean`.
pub fn read_boolean(input: &mut Input<'_>) -> Result<bool> {
    match input.read_byte() {
        -1 => Err(eof()),
        v => Ok(v != 0),
    }
}

/// `writeByteArray`.
pub fn write_byte_array(out: &mut Vec<u8>, bytes: &[u8]) {
    write_int(out, bytes.len() as i32);
    out.extend_from_slice(bytes);
}

/// `readByteArray`. A negative length is Java's `NegativeArraySizeException`.
pub fn read_byte_array(input: &mut Input<'_>) -> Result<Vec<u8>> {
    let len = read_int(input)?;
    let len = usize::try_from(len).map_err(|_| Error::Io(format!("Negative array size: {len}")))?;
    if input.remaining() < len {
        return Err(eof());
    }
    let start = input.pos;
    input.pos += len;
    Ok(input.bytes[start..start + len].to_vec())
}

/// `writeString`: UTF-8 bytes.
pub fn write_string(out: &mut Vec<u8>, value: &str) {
    write_byte_array(out, value.as_bytes());
}

/// `readString`: `new String(bytes, UTF_8)`, malformed sequences replaced.
pub fn read_string(input: &mut Input<'_>) -> Result<String> {
    Ok(String::from_utf8_lossy(&read_byte_array(input)?).into_owned())
}

/// `writeBitSet`: `BitSet.toByteArray()` -- little-endian bytes of the set
/// bits, trailing zero bytes trimmed.
pub fn write_bit_set(out: &mut Vec<u8>, bits: &[bool]) {
    let len = bits.iter().rposition(|&b| b).map_or(0, |i| i / 8 + 1);
    let mut bytes = vec![0u8; len];
    for (i, _) in bits.iter().enumerate().filter(|(_, &b)| b) {
        bytes[i / 8] |= 1 << (i % 8);
    }
    write_byte_array(out, &bytes);
}

/// `readBitSet`: `BitSet.valueOf(bytes)`, as a predicate over bit indexes.
pub fn read_bit_set(input: &mut Input<'_>) -> Result<BitSet> {
    Ok(BitSet(read_byte_array(input)?))
}

/// A `java.util.BitSet` read back from its byte array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitSet(Vec<u8>);

impl BitSet {
    /// `BitSet.get(i)`: false past the stored bytes.
    pub fn get(&self, i: usize) -> bool {
        self.0.get(i / 8).is_some_and(|b| b & (1 << (i % 8)) != 0)
    }
}

/// `writeInt`/`writeLong`'s inverse for an array's count: a negative one is
/// Java's `NegativeArraySizeException`.
///
/// Every element of every array the package writes takes at least one byte,
/// so a count above the bytes left is a truncated or corrupt stream. Java
/// allocates the array first (an `OutOfMemoryError` for a large count) and
/// reads its elements past the end as `-1` bytes; here it is the
/// end-of-stream error, so a count off the stream can neither size an
/// allocation nor drive a two-billion-step loop.
pub(crate) fn read_count(input: &mut Input<'_>) -> Result<usize> {
    let count = read_int(input)?;
    let count =
        usize::try_from(count).map_err(|_| Error::Io(format!("Negative array size: {count}")))?;
    if count > input.remaining() {
        return Err(eof());
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_java_end_of_stream() {
        let mut out = Vec::new();
        write_int(&mut out, -2);
        write_long(&mut out, 0x0123_4567_89ab_cdef);
        write_double(&mut out, f64::from_bits(0xfff8_0000_0000_0001));
        write_double(&mut out, -1.5);
        write_boolean(&mut out, true);
        write_string(&mut out, "héllo");
        write_bit_set(
            &mut out,
            &[
                true, false, false, false, false, false, false, false, false, true, false,
            ],
        );
        write_bit_set(&mut out, &[false, false]);
        let mut input = Input::new(&out);
        assert_eq!(read_int(&mut input).unwrap(), -2);
        assert_eq!(read_long(&mut input).unwrap(), 0x0123_4567_89ab_cdef);
        assert_eq!(
            read_double(&mut input).unwrap().to_bits(),
            0x7ff8_0000_0000_0000
        );
        assert_eq!(read_double(&mut input).unwrap(), -1.5);
        assert!(read_boolean(&mut input).unwrap());
        assert_eq!(read_string(&mut input).unwrap(), "héllo");
        let bits = read_bit_set(&mut input).unwrap();
        assert!(bits.get(0) && bits.get(9) && !bits.get(1) && !bits.get(100));
        assert_eq!(read_bit_set(&mut input).unwrap(), BitSet(vec![]));
        assert_eq!(input.remaining(), 0);
        // Past the end: `read()` is -1, which `readInt` masks to 0xff bytes.
        assert_eq!(read_int(&mut input).unwrap(), -1);
        assert!(read_boolean(&mut input).is_err());
        let mut short = Input::new(&[5, 0, 0, 0, 1]);
        assert!(read_byte_array(&mut short).is_err());
        let mut neg = Input::new(&[0xff, 0xff, 0xff, 0xff]);
        assert!(read_byte_array(&mut neg).is_err());
        let mut neg = Input::new(&[0xfe, 0xff, 0xff, 0xff]);
        assert!(read_count(&mut neg).is_err());
        // A count beyond the bytes left: end of stream, not a huge loop.
        let mut huge = Input::new(&[0xff, 0xff, 0xff, 0x7f, 0, 0]);
        assert_eq!(
            read_count(&mut huge).unwrap_err().to_string(),
            eof().to_string()
        );
        let mut fits = Input::new(&[2, 0, 0, 0, 7, 7]);
        assert_eq!(read_count(&mut fits).unwrap(), 2);
    }
}
