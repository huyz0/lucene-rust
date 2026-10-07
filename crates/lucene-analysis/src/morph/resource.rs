//! The reads the morph dictionaries make: `InputStreamDataInput` over a
//! file's bytes (`readByte`, `readVInt`, `readVLong`, `readZInt`,
//! `readString`, `readBytes`) and `CodecUtil.checkHeader`.
//!
//! Every failure is an [`AnalysisError::Io`] whose message starts with the
//! simple name of the exception Java throws (`EOFException`,
//! `CorruptIndexException`, `IndexFormatTooOldException`, ...).

use crate::AnalysisError;

/// `CodecUtil.CODEC_MAGIC`.
pub const CODEC_MAGIC: u32 = 0x3fd7_6c17;

/// An [`AnalysisError::Io`] reading `"<exception>: <message>"`.
pub fn io_error(exception: &str, message: impl std::fmt::Display) -> AnalysisError {
    AnalysisError::Io(format!("{exception}: {message}"))
}

/// The bytes of one dictionary file, read front to back.
#[derive(Debug, Clone)]
pub struct ResourceInput<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ResourceInput<'a> {
    /// A reader at the start of `bytes`.
    pub fn new(bytes: &'a [u8]) -> Self {
        ResourceInput { bytes, pos: 0 }
    }

    /// Bytes not yet read.
    pub fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn eof() -> AnalysisError {
        io_error("EOFException", "read past EOF")
    }

    /// `readByte()`.
    pub fn read_byte(&mut self) -> Result<u8, AnalysisError> {
        let b = *self.bytes.get(self.pos).ok_or_else(Self::eof)?;
        self.pos = self.pos.checked_add(1).ok_or_else(Self::eof)?;
        Ok(b)
    }

    /// `readBytes(len)`: the next `len` bytes, borrowed.
    pub fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], AnalysisError> {
        let end = self.pos.checked_add(len).ok_or_else(Self::eof)?;
        let out = self.bytes.get(self.pos..end).ok_or_else(Self::eof)?;
        self.pos = end;
        Ok(out)
    }

    /// `CodecUtil.readBEInt`.
    pub fn read_be_int(&mut self) -> Result<i32, AnalysisError> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// `DataInput.readVInt()`, with Java's "too many bits" check.
    pub fn read_vint(&mut self) -> Result<i32, AnalysisError> {
        let mut result: u32 = 0;
        for shift in [0u32, 7, 14, 21] {
            let b = self.read_byte()?;
            result |= u32::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                return Ok(result as i32);
            }
        }
        let b = self.read_byte()?;
        result |= u32::from(b & 0x0F) << 28;
        if b & 0xF0 == 0 {
            return Ok(result as i32);
        }
        Err(io_error(
            "IOException",
            "Invalid vInt detected (too many bits)",
        ))
    }

    /// `DataInput.readVLong()`: at most nine bytes, never negative.
    pub fn read_vlong(&mut self) -> Result<i64, AnalysisError> {
        let mut result: u64 = 0;
        for shift in [0u32, 7, 14, 21, 28, 35, 42, 49, 56] {
            let b = self.read_byte()?;
            result |= u64::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                return Ok(result as i64);
            }
        }
        Err(io_error(
            "IOException",
            "Invalid vLong detected (negative values disallowed)",
        ))
    }

    /// `DataInput.readZInt()`.
    pub fn read_zint(&mut self) -> Result<i32, AnalysisError> {
        let v = self.read_vint()? as u32;
        Ok(((v >> 1) as i32) ^ ((v & 1) as i32).wrapping_neg())
    }

    /// `DataInput.readString()`: a vint length, then that many bytes of
    /// UTF-8 (malformed sequences become U+FFFD, as `new String(bytes,
    /// UTF_8)` makes them).
    pub fn read_string(&mut self) -> Result<String, AnalysisError> {
        let len = self.read_vint()?;
        let len = usize::try_from(len).map_err(|_| io_error("NegativeArraySizeException", len))?;
        let bytes = self.read_bytes(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    /// `CodecUtil.checkHeader(in, codec, minVersion, maxVersion)`: the
    /// version read.
    pub fn check_header(
        &mut self,
        codec: &str,
        min_version: i32,
        max_version: i32,
    ) -> Result<i32, AnalysisError> {
        let actual = self.read_be_int()? as u32;
        if actual != CODEC_MAGIC {
            return Err(io_error(
                "CorruptIndexException",
                format!(
                    "codec header mismatch: actual header={actual} vs expected header={CODEC_MAGIC}"
                ),
            ));
        }
        let actual_codec = self.read_string()?;
        if actual_codec != codec {
            return Err(io_error(
                "CorruptIndexException",
                format!("codec mismatch: actual codec={actual_codec} vs expected codec={codec}"),
            ));
        }
        let version = self.read_be_int()?;
        if version < min_version {
            return Err(io_error(
                "IndexFormatTooOldException",
                format!("version {version} (needs to be between {min_version} and {max_version})"),
            ));
        }
        if version > max_version {
            return Err(io_error(
                "IndexFormatTooNewException",
                format!("version {version} (needs to be between {min_version} and {max_version})"),
            ));
        }
        Ok(version)
    }
}

/// Builders for hand-made dictionary files (tests only).
#[cfg(test)]
pub(crate) mod test_util {
    #![allow(clippy::arithmetic_side_effects)]
    use super::CODEC_MAGIC;

    /// A codec header as `CodecUtil.writeHeader` writes it.
    pub(crate) fn header(codec: &str, version: i32) -> Vec<u8> {
        let mut out = CODEC_MAGIC.to_be_bytes().to_vec();
        out.push(codec.len() as u8);
        out.extend_from_slice(codec.as_bytes());
        out.extend_from_slice(&version.to_be_bytes());
        out
    }

    /// `DataOutput.writeVInt`/`writeVLong`.
    pub(crate) fn vlong(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push((v as u8 & 0x7F) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }
}

#[cfg(test)]
mod tests {
    use super::test_util::{header, vlong};
    use super::*;

    fn msg(e: AnalysisError) -> String {
        match e {
            AnalysisError::Io(m) => m,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn scalar_reads_match_data_input() {
        let mut b = Vec::new();
        vlong(&mut b, 300);
        vlong(&mut b, u64::from(u32::MAX >> 1));
        vlong(&mut b, 1 << 40);
        vlong(&mut b, 3); // zint -2
        b.extend_from_slice(&[3, b'a', 0xFF, b'c']);
        b.extend_from_slice(&7i32.to_be_bytes());
        let mut r = ResourceInput::new(&b);
        assert_eq!(r.read_vint().unwrap(), 300);
        assert_eq!(r.read_vint().unwrap(), i32::MAX);
        assert_eq!(r.read_vlong().unwrap(), 1 << 40);
        assert_eq!(r.read_zint().unwrap(), -2);
        assert_eq!(r.read_string().unwrap(), "a\u{FFFD}c");
        assert_eq!(r.read_be_int().unwrap(), 7);
        assert_eq!(r.remaining(), 0);
        assert!(msg(r.read_byte().unwrap_err()).starts_with("EOFException"));
        assert!(r.read_bytes(usize::MAX).is_err());
    }

    #[test]
    fn malformed_varints_and_strings_fail() {
        let five = [0xFF, 0xFF, 0xFF, 0xFF, 0x1F];
        assert!(msg(ResourceInput::new(&five).read_vint().unwrap_err()).contains("too many bits"));
        let mut ok = five;
        ok[4] = 0x0F;
        assert_eq!(ResourceInput::new(&ok).read_vint().unwrap(), -1);
        let ten = [0xFF; 10];
        assert!(msg(ResourceInput::new(&ten).read_vlong().unwrap_err()).contains("negative"));
        let neg_len = [0xFF, 0xFF, 0xFF, 0xFF, 0x0F];
        assert!(msg(ResourceInput::new(&neg_len).read_string().unwrap_err())
            .starts_with("NegativeArraySizeException"));
        assert!(ResourceInput::new(&[5, b'a']).read_string().is_err());
    }

    #[test]
    fn check_header_like_codec_util() {
        let h = header("ko_cc", 1);
        assert_eq!(
            ResourceInput::new(&h).check_header("ko_cc", 1, 1).unwrap(),
            1
        );
        let e = msg(ResourceInput::new(&h)
            .check_header("ja_cc", 1, 1)
            .unwrap_err());
        assert_eq!(
            e,
            "CorruptIndexException: codec mismatch: actual codec=ko_cc vs expected codec=ja_cc"
        );
        let e = msg(ResourceInput::new(&h)
            .check_header("ko_cc", 2, 3)
            .unwrap_err());
        assert!(e.starts_with("IndexFormatTooOldException"), "{e}");
        let e = msg(ResourceInput::new(&header("ko_cc", 4))
            .check_header("ko_cc", 2, 3)
            .unwrap_err());
        assert!(e.starts_with("IndexFormatTooNewException"), "{e}");
        let mut bad = h.clone();
        bad[0] ^= 1;
        let e = msg(ResourceInput::new(&bad)
            .check_header("ko_cc", 1, 1)
            .unwrap_err());
        assert!(e.contains("codec header mismatch"), "{e}");
        for cut in 0..h.len() {
            assert!(ResourceInput::new(&h[..cut])
                .check_header("ko_cc", 1, 1)
                .is_err());
        }
    }
}
