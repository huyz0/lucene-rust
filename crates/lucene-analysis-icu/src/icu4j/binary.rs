//! `com.ibm.icu.impl.ICUBinary`: the common header of ICU's binary data
//! files and a bounds-checked reader over the bytes after it.
//!
//! Layout (`ICUBinary.readHeader`): `uint16 headerSize`, the magic bytes
//! `0xda 0x27`, then `UDataInfo` -- `uint16 size`, `uint16 reserved`,
//! `uint8 isBigEndian`, `uint8 charsetFamily` (0, ASCII), `uint8
//! sizeofUChar` (2), `uint8 reserved`, `uint8 dataFormat[4]`, `uint8
//! formatVersion[4]`, `uint8 dataVersion[4]` -- padded to `headerSize`.
//! Every multi-byte value after the header is in the byte order
//! `isBigEndian` names (ICU4J ships big-endian data; both are read).

use crate::IcuError;

/// A cursor over an ICU data file, Java's `ByteBuffer` as ICU4J reads it:
/// every read is bounds-checked and fails with a typed error rather than
/// Java's `BufferUnderflowException`.
#[derive(Debug, Clone)]
pub struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    big_endian: bool,
}

fn underflow() -> IcuError {
    IcuError::new("ICU data: buffer underflow")
}

impl<'a> ByteReader<'a> {
    /// A big-endian reader at position 0.
    pub fn new(bytes: &'a [u8]) -> Self {
        ByteReader {
            bytes,
            pos: 0,
            big_endian: true,
        }
    }

    /// The current position.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Bytes left after the position.
    pub fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    /// Whether multi-byte values are read big-endian.
    pub fn is_big_endian(&self) -> bool {
        self.big_endian
    }

    /// Sets the byte order (`ByteBuffer.order`).
    pub fn set_big_endian(&mut self, big_endian: bool) {
        self.big_endian = big_endian;
    }

    /// Moves to `pos` (`ByteBuffer.position(int)`).
    pub fn seek(&mut self, pos: usize) -> Result<(), IcuError> {
        if pos > self.bytes.len() {
            return Err(underflow());
        }
        self.pos = pos;
        Ok(())
    }

    /// `ICUBinary.skipBytes`.
    pub fn skip(&mut self, n: usize) -> Result<(), IcuError> {
        let end = self.pos.checked_add(n).ok_or_else(underflow)?;
        self.seek(end)
    }

    /// The next `n` bytes.
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], IcuError> {
        let end = self.pos.checked_add(n).ok_or_else(underflow)?;
        let s = self.bytes.get(self.pos..end).ok_or_else(underflow)?;
        self.pos = end;
        Ok(s)
    }

    /// The bytes after the position (not consumed).
    pub fn rest(&self) -> &'a [u8] {
        self.bytes.get(self.pos..).unwrap_or(&[])
    }

    /// `get()`.
    pub fn u8(&mut self) -> Result<u8, IcuError> {
        Ok(self.take(1)?[0])
    }

    /// `getChar()`.
    pub fn u16(&mut self) -> Result<u16, IcuError> {
        let b = self.take(2)?;
        let a = [b[0], b[1]];
        Ok(if self.big_endian {
            u16::from_be_bytes(a)
        } else {
            u16::from_le_bytes(a)
        })
    }

    /// `getInt()`.
    pub fn i32(&mut self) -> Result<i32, IcuError> {
        let b = self.take(4)?;
        let a = [b[0], b[1], b[2], b[3]];
        Ok(if self.big_endian {
            i32::from_be_bytes(a)
        } else {
            i32::from_le_bytes(a)
        })
    }

    /// `ICUBinary.getChars`: `n` UTF-16 code units.
    pub fn u16s(&mut self, n: usize) -> Result<Vec<u16>, IcuError> {
        let bytes = self.take(n.checked_mul(2).ok_or_else(underflow)?)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|b| {
                if self.big_endian {
                    u16::from_be_bytes([b[0], b[1]])
                } else {
                    u16::from_le_bytes([b[0], b[1]])
                }
            })
            .collect())
    }

    /// `ICUBinary.getInts`: `n` 32-bit values.
    pub fn i32s(&mut self, n: usize) -> Result<Vec<i32>, IcuError> {
        let bytes = self.take(n.checked_mul(4).ok_or_else(underflow)?)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| {
                let a = [b[0], b[1], b[2], b[3]];
                if self.big_endian {
                    i32::from_be_bytes(a)
                } else {
                    i32::from_le_bytes(a)
                }
            })
            .collect())
    }
}

impl ByteReader<'_> {
    /// `ICUBinary.getLongs`: `n` 64-bit values.
    pub fn i64s(&mut self, n: usize) -> Result<Vec<i64>, IcuError> {
        let bytes = self.take(n.checked_mul(8).ok_or_else(underflow)?)?;
        Ok(bytes
            .chunks_exact(8)
            .map(|b| {
                let a = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
                if self.big_endian {
                    i64::from_be_bytes(a)
                } else {
                    i64::from_le_bytes(a)
                }
            })
            .collect())
    }
}

/// `ICUBinary.readHeader`: checks the magic bytes, the charset family and
/// `UChar` size, the four-byte `data_format`, and the format version
/// (`accept`); sets the byte order and moves past the header. Returns the
/// format version.
pub fn read_header<'a>(
    bytes: &'a [u8],
    data_format: u32,
    accept: impl Fn(&[u8; 4]) -> bool,
) -> Result<(ByteReader<'a>, [u8; 4]), IcuError> {
    if bytes.len() < 24 {
        return Err(IcuError::new("ICU data: header too short"));
    }
    if bytes[2] != 0xda || bytes[3] != 0x27 {
        return Err(IcuError::new(
            "Data file authentication failed: magic number",
        ));
    }
    let big_endian = bytes[8];
    if big_endian > 1 || bytes[9] != 0 || bytes[10] != 2 {
        return Err(IcuError::new("Data file authentication failed: header"));
    }
    let mut r = ByteReader::new(bytes);
    r.set_big_endian(big_endian != 0);
    let header_size = usize::from(r.u16()?);
    r.seek(4)?;
    let sizeof_info = usize::from(r.u16()?);
    // ARITH: sizeof_info is a u16 widened to usize.
    if sizeof_info < 20 || header_size < sizeof_info.saturating_add(4) {
        return Err(IcuError::new("Internal Error: Header size error"));
    }
    let format_version = [bytes[16], bytes[17], bytes[18], bytes[19]];
    if bytes[12..16] != data_format.to_be_bytes() || !accept(&format_version) {
        return Err(IcuError::new(format!(
            "Data file authentication failed: data format {:02x}{:02x}{:02x}{:02x}, format version {}.{}.{}.{}",
            bytes[12], bytes[13], bytes[14], bytes[15],
            format_version[0], format_version[1], format_version[2], format_version[3]
        )));
    }
    r.seek(header_size)?;
    Ok((r, format_version))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(fmt: &[u8; 4], big: u8) -> Vec<u8> {
        let mut h = vec![0u8; 32];
        h[1] = 32;
        h[2] = 0xda;
        h[3] = 0x27;
        h[5] = 20;
        if big == 0 {
            h[0] = 32;
            h[1] = 0;
            h[4] = 20;
            h[5] = 0;
        }
        h[8] = big;
        h[10] = 2;
        h[12..16].copy_from_slice(fmt);
        h[16] = 5;
        h
    }

    #[test]
    fn reads_both_byte_orders() {
        for big in [0u8, 1] {
            let mut b = header(b"Nrm2", big);
            b.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
            let (mut r, v) = read_header(&b, 0x4e72_6d32, |v| v[0] == 5).unwrap();
            assert_eq!(v[0], 5);
            assert_eq!(r.position(), 32);
            assert_eq!(r.is_big_endian(), big == 1);
            let x = r.u16().unwrap();
            assert_eq!(x, if big == 1 { 0x0102 } else { 0x0201 });
            assert_eq!(r.remaining(), 4);
            assert_eq!(r.rest(), &[3, 4, 5, 6]);
            let i = r.i32().unwrap();
            assert_eq!(i, if big == 1 { 0x0304_0506 } else { 0x0605_0403 });
            assert!(r.u8().is_err());
        }
    }

    #[test]
    fn rejects_bad_headers() {
        assert!(read_header(&[0; 8], 0, |_| true).is_err());
        let mut b = header(b"Nrm2", 1);
        b[2] = 0;
        assert!(read_header(&b, 0x4e72_6d32, |_| true)
            .unwrap_err()
            .message()
            .contains("magic"));
        let mut b = header(b"Nrm2", 1);
        b[8] = 2;
        assert!(read_header(&b, 0x4e72_6d32, |_| true).is_err());
        let mut b = header(b"Nrm2", 1);
        b[10] = 1;
        assert!(read_header(&b, 0x4e72_6d32, |_| true).is_err());
        let mut b = header(b"Nrm2", 1);
        b[5] = 10;
        assert!(read_header(&b, 0x4e72_6d32, |_| true)
            .unwrap_err()
            .message()
            .contains("size"));
        let b = header(b"Brk ", 1);
        let e = read_header(&b, 0x4e72_6d32, |_| true).unwrap_err();
        assert!(e.message().contains("data format 42726b20"), "{e:?}");
        let b = header(b"Nrm2", 1);
        assert!(read_header(&b, 0x4e72_6d32, |v| v[0] == 4).is_err());
        let mut b = header(b"Nrm2", 1);
        b[1] = 200;
        assert!(read_header(&b, 0x4e72_6d32, |_| true).is_err());
    }

    #[test]
    fn bounded_reads() {
        let b = [0u8, 1, 0, 2, 0, 0, 0, 3, 9];
        let mut r = ByteReader::new(&b);
        assert_eq!(r.u16s(2).unwrap(), vec![1, 2]);
        assert_eq!(r.i32s(1).unwrap(), vec![3]);
        assert!(r.i32s(1).is_err());
        assert!(r.u16s(usize::MAX).is_err());
        assert!(r.i32s(usize::MAX).is_err());
        assert!(r.skip(usize::MAX).is_err());
        assert!(r.skip(2).is_err());
        r.skip(1).unwrap();
        assert_eq!(r.remaining(), 0);
        assert!(r.take(usize::MAX).is_err());
        r.set_big_endian(false);
        r.seek(0).unwrap();
        assert_eq!(r.u16s(1).unwrap(), vec![0x100]);
        assert_eq!(r.i32s(1).unwrap(), vec![0x200]);
        assert!(r.seek(100).is_err());
        assert_eq!(ByteReader::new(&[]).rest(), &[] as &[u8]);
    }
}
