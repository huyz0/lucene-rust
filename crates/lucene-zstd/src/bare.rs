//! Bare frames: zstd frames without the magic number, the plain length, the
//! dictionary id or a checksum, for formats that compress many small blocks
//! and already record each block's length and dictionary themselves. A bare
//! frame is 8 or more bytes shorter than a standard one; only a
//! [`BareDecompressor`] reads it.

use std::cell::Cell;
use std::io::Cursor;

use zstd_safe::{CCtx, CParameter, DCtx, DParameter, DictAttachPref, FrameFormat};

use crate::{
    append_with, check_level, compress_bound, finish, zstd_error, DecoderDictionary,
    EncoderDictionary, Result,
};

/// Compresses blocks into bare frames at one level, optionally against one
/// prepared dictionary for the compressor's whole life.
pub struct BareCompressor<'d> {
    cctx: CCtx<'d>,
}

impl std::fmt::Debug for BareCompressor<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BareCompressor").finish_non_exhaustive()
    }
}

impl<'d> BareCompressor<'d> {
    /// A compressor at `level`, against `dict` when given. With a dictionary
    /// the dictionary's tables are searched in place rather than copied per
    /// block, which suits blocks of a few KiB or less.
    ///
    /// # Errors
    /// `level` is outside [`crate::levels`].
    pub fn new(level: i32, dict: Option<&'d EncoderDictionary>) -> Result<Self> {
        check_level(level)?;
        let mut cctx = CCtx::create();
        for p in [
            CParameter::Format(FrameFormat::Magicless),
            CParameter::ContentSizeFlag(false),
            CParameter::DictIdFlag(false),
            CParameter::ChecksumFlag(false),
            CParameter::CompressionLevel(level),
        ] {
            cctx.set_parameter(p).map_err(zstd_error)?;
        }
        if let Some(d) = dict {
            cctx.set_parameter(CParameter::ForceAttachDict(DictAttachPref::ForceAttach))
                .map_err(zstd_error)?;
            cctx.ref_cdict(&d.cdict).map_err(zstd_error)?;
        }
        Ok(Self { cctx })
    }

    /// Appends `src` as one bare frame to `out` and returns its length.
    ///
    /// # Errors
    /// libzstd failed (it does not for any input).
    pub fn compress(&mut self, src: &[u8], out: &mut Vec<u8>) -> Result<usize> {
        let cctx = &mut self.cctx;
        append_with(out, compress_bound(src.len()), |dst| {
            cctx.compress2(dst, src)
        })
    }
}

/// Decompresses bare frames.
pub struct BareDecompressor {
    dctx: DCtx<'static>,
}

impl std::fmt::Debug for BareDecompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BareDecompressor").finish_non_exhaustive()
    }
}

impl Default for BareDecompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl BareDecompressor {
    /// A fresh context.
    pub fn new() -> Self {
        let mut dctx = DCtx::create();
        // Setting the format on a fresh context cannot fail.
        let _ = dctx.set_parameter(DParameter::Format(FrameFormat::Magicless));
        Self { dctx }
    }

    /// Decompresses the bare frame `src` (compressed against `dict`), which
    /// must hold exactly `expected` bytes, and appends them to `out`.
    ///
    /// # Errors
    /// The frame is corrupt, needs another dictionary, or holds a different
    /// number of bytes; `out` is left as it was.
    pub fn decompress(
        &mut self,
        src: &[u8],
        dict: Option<&DecoderDictionary>,
        expected: usize,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        let start = out.len();
        // libzstd writes into the spare capacity and never past it.
        out.reserve(expected);
        let mut cursor = Cursor::new(&mut *out);
        cursor.set_position(start as u64);
        let n = self.run(src, dict, &mut cursor);
        finish(out, start, expected, n)
    }

    /// Decompresses the bare frame `src` into exactly all of `dst`.
    ///
    /// # Errors
    /// As [`Self::decompress`].
    pub fn decompress_exact(
        &mut self,
        src: &[u8],
        dict: Option<&DecoderDictionary>,
        dst: &mut [u8],
    ) -> Result<()> {
        let expected = dst.len();
        match self.run(src, dict, dst) {
            Ok(n) if n == expected => Ok(()),
            Ok(n) => Err(crate::Error::Length {
                expected: expected as u64,
                actual: n as u64,
            }),
            Err(code) => Err(zstd_error(code)),
        }
    }

    fn run<C: zstd_safe::WriteBuf + ?Sized>(
        &mut self,
        src: &[u8],
        dict: Option<&DecoderDictionary>,
        dst: &mut C,
    ) -> std::result::Result<usize, usize> {
        match dict {
            Some(d) => self.dctx.decompress_using_ddict(dst, src, &d.ddict),
            None => self.dctx.decompress(dst, src),
        }
    }
}

thread_local! {
    /// [`with_bare_decompressor`]'s context.
    static BARE: Cell<Option<Box<BareDecompressor>>> = const { Cell::new(None) };
}

/// Runs `f` with this thread's reusable [`BareDecompressor`]. A nested call
/// gets a fresh one.
pub fn with_bare_decompressor<R>(f: impl FnOnce(&mut BareDecompressor) -> R) -> R {
    let mut d = BARE
        .with(Cell::take)
        .unwrap_or_else(|| Box::new(BareDecompressor::new()));
    let r = f(&mut d);
    BARE.with(|cell| cell.set(Some(d)));
    r
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use crate::{Compressor, Error};

    fn text(len: usize, seed: usize) -> Vec<u8> {
        let words: [&[u8]; 5] = [b"term ", b"suffix ", b"block ", b"tree ", b"leaf "];
        let mut out = Vec::with_capacity(len);
        let mut i = seed;
        while out.len() < len {
            out.extend_from_slice(words[(i * 5 + i / 2) % words.len()]);
            i += 1;
        }
        out.truncate(len);
        out
    }

    #[test]
    fn bare_frames_round_trip_and_are_shorter() {
        let dict = text(8192, 2);
        let enc = EncoderDictionary::new(&dict, 3).unwrap();
        let dec = DecoderDictionary::new(&dict);
        let mut plain_c = BareCompressor::new(3, None).unwrap();
        let mut dict_c = BareCompressor::new(3, Some(&enc)).unwrap();
        let mut std_c = Compressor::new(3).unwrap();
        let mut d = BareDecompressor::new();
        for len in [0, 1, 37, 1000, 70_000] {
            let src = text(len, 7);
            for (c, dd) in [(&mut plain_c, None), (&mut dict_c, Some(&dec))] {
                let mut frame = vec![9u8];
                let n = c.compress(&src, &mut frame).unwrap();
                assert_eq!(frame.len(), n + 1);
                let mut out = b"ab".to_vec();
                d.decompress(&frame[1..], dd, len, &mut out).unwrap();
                assert_eq!(&out[2..], &src[..]);
                let mut exact = vec![0u8; len];
                d.decompress_exact(&frame[1..], dd, &mut exact).unwrap();
                assert_eq!(exact, src);
                if dd.is_none() {
                    let mut standard = Vec::new();
                    std_c.compress(&src, &mut standard).unwrap();
                    assert!(n + 4 <= standard.len(), "{len}: {n} vs {}", standard.len());
                }
            }
        }
    }

    #[test]
    fn wrong_lengths_and_dictionaries_fail_without_writing() {
        let dict = text(4096, 1);
        let enc = EncoderDictionary::new(&dict, 3).unwrap();
        let mut c = BareCompressor::new(3, Some(&enc)).unwrap();
        let src = text(2000, 3);
        let mut frame = Vec::new();
        c.compress(&src, &mut frame).unwrap();
        let dec = DecoderDictionary::new(&dict);
        let mut d = BareDecompressor::new();
        let mut out = b"keep".to_vec();
        // Too little room fails in libzstd, too much fails the length check.
        assert!(d.decompress(&frame, Some(&dec), 1999, &mut out).is_err());
        assert_eq!(out, b"keep");
        let e = d
            .decompress(&frame, Some(&dec), 2001, &mut out)
            .unwrap_err();
        assert!(matches!(e, Error::Length { .. }), "{e:?}");
        assert_eq!(out, b"keep");
        // No dictionary or another one: an error or other bytes, never a panic.
        let other = DecoderDictionary::new(&text(4096, 99));
        for dd in [None, Some(&other)] {
            let mut o = Vec::new();
            if d.decompress(&frame, dd, 2000, &mut o).is_ok() {
                assert_ne!(o, src);
            }
        }
        assert!(d.decompress(&[1, 2, 3], None, 10, &mut out).is_err());
        assert_eq!(out, b"keep");
        with_bare_decompressor(|d| {
            let mut o = Vec::new();
            d.decompress(&frame, Some(&dec), 2000, &mut o).unwrap();
            assert_eq!(o, src);
        });
    }
}
