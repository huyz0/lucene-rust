//! Zstandard block compression for index formats.
//!
//! A thin layer over libzstd (the C library zstd-jni wraps on the JVM) for
//! formats that compress independent blocks and read them back at random:
//! stored-fields chunks today, and posting lists or vector partitions
//! tomorrow. Nothing here knows about any one file format.
//!
//! - [`Compressor`] and [`Decompressor`] own reusable libzstd contexts, so a
//!   writer or a reading thread pays for context setup once rather than per
//!   block. [`with_decompressor`] lends a per-thread one to readers that have
//!   no state of their own to keep it in.
//! - Every block is one standard zstd frame. A block can be compressed
//!   against a dictionary: either bytes passed per call (a *raw-content*
//!   dictionary, such as a chunk's own first bytes), or a prepared
//!   [`EncoderDictionary`]/[`DecoderDictionary`] built once and reused, for
//!   example one trained (`train_dictionary`, feature `dict-builder`) on sample blocks.
//! - Decompression always states the plain length it expects and fails on
//!   any other, so a length read off disk can never make it write past what
//!   the caller planned for.
//!
//! Dictionary bytes are interpreted the way libzstd does by default, which
//! is also zstd-jni's: bytes that begin with the zstd dictionary magic
//! number are parsed as a formatted dictionary, anything else is raw content.

#![forbid(unsafe_code)]

use std::cell::Cell;
use std::io::Cursor;

use zstd_safe::{CCtx, CDict, DCtx, DDict};

/// libzstd's default level, and OpenSearch's `index.codec.compression_level`
/// default.
pub const DEFAULT_LEVEL: i32 = 3;

/// The levels [`Compressor::new`] accepts: libzstd's negative "fast" levels
/// through its highest.
pub fn levels() -> std::ops::RangeInclusive<i32> {
    zstd_safe::min_c_level()..=zstd_safe::max_c_level()
}

/// A compression or decompression failure.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    /// libzstd rejected the input (a corrupt frame, a missing dictionary, a
    /// destination too small).
    #[error("zstd: {0}")]
    Zstd(&'static str),
    /// A frame decoded to a different length than the caller expected.
    #[error("zstd frame holds {actual} bytes, expected {expected}")]
    Length { expected: u64, actual: u64 },
    /// A level outside [`levels`].
    #[error("zstd level {0} is outside {min}..={max}", min = zstd_safe::min_c_level(), max = zstd_safe::max_c_level())]
    Level(i32),
}

/// This crate's result.
pub type Result<T> = std::result::Result<T, Error>;

fn zstd_error(code: usize) -> Error {
    Error::Zstd(zstd_safe::get_error_name(code))
}

/// The most bytes one frame of `len` plain bytes can take.
pub fn compress_bound(len: usize) -> usize {
    zstd_safe::compress_bound(len)
}

/// A dictionary prepared for compression at one level (libzstd's `CDict`):
/// worth building when many blocks share one dictionary.
pub struct EncoderDictionary {
    cdict: CDict<'static>,
}

impl EncoderDictionary {
    /// Prepares `dict` (copied) for compressing at `level`.
    ///
    /// # Errors
    /// `level` is outside [`levels`].
    pub fn new(dict: &[u8], level: i32) -> Result<Self> {
        check_level(level)?;
        Ok(Self {
            cdict: CDict::create(dict, level),
        })
    }
}

impl std::fmt::Debug for EncoderDictionary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncoderDictionary").finish_non_exhaustive()
    }
}

/// A dictionary prepared for decompression (libzstd's `DDict`).
pub struct DecoderDictionary {
    ddict: DDict<'static>,
}

impl DecoderDictionary {
    /// Prepares `dict` (copied) for decompression.
    pub fn new(dict: &[u8]) -> Self {
        Self {
            ddict: DDict::create(dict),
        }
    }
}

impl std::fmt::Debug for DecoderDictionary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecoderDictionary").finish_non_exhaustive()
    }
}

fn check_level(level: i32) -> Result<()> {
    if levels().contains(&level) {
        Ok(())
    } else {
        Err(Error::Level(level))
    }
}

/// Appends to `out` what `write` puts into the spare capacity after `bound`
/// more bytes are reserved, and returns how many it wrote. Nothing is zeroed
/// first: libzstd writes every byte it reports.
fn append_with(
    out: &mut Vec<u8>,
    bound: usize,
    write: impl FnOnce(&mut Cursor<&mut Vec<u8>>) -> std::result::Result<usize, usize>,
) -> Result<usize> {
    let start = out.len();
    out.reserve(bound);
    let mut cursor = Cursor::new(&mut *out);
    cursor.set_position(start as u64);
    match write(&mut cursor) {
        Ok(n) => Ok(n),
        Err(code) => {
            out.truncate(start);
            Err(zstd_error(code))
        }
    }
}

/// A reusable compression context at one level.
pub struct Compressor {
    cctx: CCtx<'static>,
    level: i32,
}

impl std::fmt::Debug for Compressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compressor")
            .field("level", &self.level)
            .finish_non_exhaustive()
    }
}

impl Compressor {
    /// A compressor at `level`.
    ///
    /// # Errors
    /// `level` is outside [`levels`].
    pub fn new(level: i32) -> Result<Self> {
        check_level(level)?;
        Ok(Self {
            cctx: CCtx::create(),
            level,
        })
    }

    /// The level this compressor compresses at.
    pub fn level(&self) -> i32 {
        self.level
    }

    /// Appends `src` as one frame to `out` and returns the frame's length.
    ///
    /// # Errors
    /// libzstd failed (it does not for any input at a valid level).
    pub fn compress(&mut self, src: &[u8], out: &mut Vec<u8>) -> Result<usize> {
        let level = self.level;
        let cctx = &mut self.cctx;
        append_with(out, compress_bound(src.len()), |dst| {
            cctx.compress(dst, src, level)
        })
    }

    /// [`Self::compress`] against `dict`, which the decompressor must be
    /// given too. An empty `dict` is no dictionary.
    ///
    /// # Errors
    /// As [`Self::compress`].
    pub fn compress_with_dict(
        &mut self,
        dict: &[u8],
        src: &[u8],
        out: &mut Vec<u8>,
    ) -> Result<usize> {
        let level = self.level;
        let cctx = &mut self.cctx;
        append_with(out, compress_bound(src.len()), |dst| {
            cctx.compress_using_dict(dst, src, dict, level)
        })
    }

    /// [`Self::compress`] against a prepared dictionary, at the level it was
    /// prepared for.
    ///
    /// # Errors
    /// As [`Self::compress`].
    pub fn compress_with_prepared(
        &mut self,
        dict: &EncoderDictionary,
        src: &[u8],
        out: &mut Vec<u8>,
    ) -> Result<usize> {
        let cctx = &mut self.cctx;
        append_with(out, compress_bound(src.len()), |dst| {
            cctx.compress_using_cdict(dst, src, &dict.cdict)
        })
    }
}

/// The dictionary a frame was compressed against, as a decompressor needs it.
#[derive(Debug, Clone, Copy)]
pub enum Dictionary<'a> {
    /// No dictionary.
    None,
    /// Dictionary bytes, given per call.
    Bytes(&'a [u8]),
    /// A prepared dictionary.
    Prepared(&'a DecoderDictionary),
}

/// A reusable decompression context.
pub struct Decompressor {
    dctx: DCtx<'static>,
}

impl std::fmt::Debug for Decompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decompressor").finish_non_exhaustive()
    }
}

impl Default for Decompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Decompressor {
    /// A fresh context.
    pub fn new() -> Self {
        Self {
            dctx: DCtx::create(),
        }
    }

    /// Decompresses the frame `src` (compressed against `dict`), which must
    /// hold exactly `expected` bytes, and appends them to `out`.
    ///
    /// A frame that records its plain length is checked against `expected`
    /// before anything is reserved, so a corrupt length cannot make this
    /// allocate more than the caller expects.
    ///
    /// # Errors
    /// The frame is corrupt, needs another dictionary, or holds a different
    /// number of bytes; `out` is left as it was.
    pub fn decompress(
        &mut self,
        src: &[u8],
        dict: Dictionary<'_>,
        expected: usize,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        if let Ok(Some(declared)) = zstd_safe::get_frame_content_size(src) {
            if declared != expected as u64 {
                return Err(Error::Length {
                    expected: expected as u64,
                    actual: declared,
                });
            }
        }
        let start = out.len();
        // Room for `expected` bytes. libzstd writes into the spare capacity
        // and never past it, so a frame holding more fails or is caught by
        // the length check below; `out` never grows on its own.
        out.reserve(expected);
        let dctx = &mut self.dctx;
        let mut cursor = Cursor::new(&mut *out);
        cursor.set_position(start as u64);
        let n = match dict {
            Dictionary::None => dctx.decompress(&mut cursor, src),
            Dictionary::Bytes(d) => dctx.decompress_using_dict(&mut cursor, src, d),
            Dictionary::Prepared(d) => dctx.decompress_using_ddict(&mut cursor, src, &d.ddict),
        };
        finish(out, start, expected, n)
    }

    /// Decompresses the frame `src` into the front of `dst` and returns the
    /// number of bytes written; a frame larger than `dst` is an error.
    ///
    /// # Errors
    /// The frame is corrupt, needs another dictionary, or does not fit.
    pub fn decompress_to_slice(
        &mut self,
        src: &[u8],
        dict: Dictionary<'_>,
        dst: &mut [u8],
    ) -> Result<usize> {
        let dctx = &mut self.dctx;
        match dict {
            Dictionary::None => dctx.decompress(dst, src),
            Dictionary::Bytes(d) => dctx.decompress_using_dict(dst, src, d),
            Dictionary::Prepared(d) => dctx.decompress_using_ddict(dst, src, &d.ddict),
        }
        .map_err(zstd_error)
    }
}

fn finish(
    out: &mut Vec<u8>,
    start: usize,
    expected: usize,
    n: std::result::Result<usize, usize>,
) -> Result<()> {
    match n {
        Ok(n) if n == expected => Ok(()),
        Ok(n) => {
            out.truncate(start);
            Err(Error::Length {
                expected: expected as u64,
                actual: n as u64,
            })
        }
        Err(code) => {
            out.truncate(start);
            Err(zstd_error(code))
        }
    }
}

thread_local! {
    /// [`with_decompressor`]'s context.
    static DECOMPRESSOR: Cell<Option<Box<Decompressor>>> = const { Cell::new(None) };
}

/// Runs `f` with this thread's reusable [`Decompressor`]. A nested call gets
/// a fresh one rather than the borrowed one.
pub fn with_decompressor<R>(f: impl FnOnce(&mut Decompressor) -> R) -> R {
    let mut d = DECOMPRESSOR
        .with(Cell::take)
        .unwrap_or_else(|| Box::new(Decompressor::new()));
    let r = f(&mut d);
    DECOMPRESSOR.with(|cell| cell.set(Some(d)));
    r
}

/// Trains a dictionary of at most `max_size` bytes on `samples`, for
/// [`EncoderDictionary`] and [`DecoderDictionary`].
///
/// # Errors
/// libzstd could not train one (too few or too small samples).
#[cfg(feature = "dict-builder")]
pub fn train_dictionary(samples: &[&[u8]], max_size: usize) -> Result<Vec<u8>> {
    let mut joined = Vec::with_capacity(samples.iter().map(|s| s.len()).sum());
    let mut sizes = Vec::with_capacity(samples.len());
    for s in samples {
        joined.extend_from_slice(s);
        sizes.push(s.len());
    }
    let mut dict = Vec::with_capacity(max_size);
    zstd_safe::train_from_buffer(&mut dict, &joined, &sizes).map_err(zstd_error)?;
    Ok(dict)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    fn text(len: usize, seed: usize) -> Vec<u8> {
        let words: [&[u8]; 6] = [
            b"lucene ", b"zstd ", b"block ", b"index ", b"stored ", b"field ",
        ];
        let mut out = Vec::with_capacity(len);
        let mut i = seed;
        while out.len() < len {
            out.extend_from_slice(words[(i * 7 + i / 3) % words.len()]);
            i += 1;
        }
        out.truncate(len);
        out
    }

    #[test]
    fn frames_round_trip_with_and_without_dictionaries() {
        let dict = text(4096, 1);
        let prepared_c = EncoderDictionary::new(&dict, 5).unwrap();
        let prepared_d = DecoderDictionary::new(&dict);
        let mut c = Compressor::new(5).unwrap();
        assert_eq!(c.level(), 5);
        for len in [1, 100, 10_000, 200_000] {
            let src = text(len, 3);
            let mut frames = vec![b'x'];
            let a = c.compress(&src, &mut frames).unwrap();
            let b = c.compress_with_dict(&dict, &src, &mut frames).unwrap();
            let p = c
                .compress_with_prepared(&prepared_c, &src, &mut frames)
                .unwrap();
            assert_eq!(frames.len(), 1 + a + b + p);
            let (fa, rest) = frames[1..].split_at(a);
            let (fb, fp) = rest.split_at(b);
            let mut d = Decompressor::new();
            for (frame, dict) in [
                (fa, Dictionary::None),
                (fb, Dictionary::Bytes(&dict)),
                (fp, Dictionary::Prepared(&prepared_d)),
                (fp, Dictionary::Bytes(&dict)),
            ] {
                let mut out = b"prefix".to_vec();
                d.decompress(frame, dict, len, &mut out).unwrap();
                assert_eq!(&out[..6], b"prefix");
                assert_eq!(&out[6..], &src[..]);
                let mut slice = vec![0u8; len + 9];
                assert_eq!(d.decompress_to_slice(frame, dict, &mut slice).unwrap(), len);
                assert_eq!(&slice[..len], &src[..]);
            }
        }
    }

    #[test]
    fn the_expected_length_is_enforced() {
        let src = text(5000, 0);
        let mut frame = Vec::new();
        Compressor::new(DEFAULT_LEVEL)
            .unwrap()
            .compress(&src, &mut frame)
            .unwrap();
        let mut d = Decompressor::new();
        let mut out = vec![7u8];
        for wrong in [0, 4999, 5001] {
            let e = d
                .decompress(&frame, Dictionary::None, wrong, &mut out)
                .unwrap_err();
            assert_eq!(
                e,
                Error::Length {
                    expected: wrong as u64,
                    actual: 5000
                }
            );
            assert_eq!(out, [7]);
        }
        // Spare capacity beyond the expected length does not hide a longer frame.
        let mut roomy = Vec::with_capacity(1 << 16);
        d.decompress(&frame, Dictionary::None, 5000, &mut roomy)
            .unwrap();
        assert_eq!(roomy, src);
        let mut headerless = Vec::with_capacity(compress_bound(src.len()));
        let mut cctx = CCtx::create();
        cctx.set_parameter(zstd_safe::CParameter::ContentSizeFlag(false))
            .unwrap();
        cctx.compress2(&mut headerless, &src).unwrap();
        let mut roomy = Vec::with_capacity(1 << 16);
        assert_eq!(
            d.decompress(&headerless, Dictionary::None, 4000, &mut roomy)
                .unwrap_err(),
            Error::Length {
                expected: 4000,
                actual: 5000
            }
        );
        assert!(roomy.is_empty());
    }

    #[test]
    fn a_frame_without_its_dictionary_or_corrupted_fails_cleanly() {
        let dict = text(2048, 9);
        let src = text(3000, 9);
        let mut frame = Vec::new();
        Compressor::new(1)
            .unwrap()
            .compress_with_dict(&dict, &src, &mut frame)
            .unwrap();
        let mut d = Decompressor::new();
        let mut out = Vec::new();
        assert!(
            d.decompress(&frame, Dictionary::None, 3000, &mut out)
                .is_err()
                || out != src
        );
        let mut out = Vec::new();
        assert!(matches!(
            d.decompress(
                &frame[..frame.len() / 2],
                Dictionary::Bytes(&dict),
                3000,
                &mut out
            ),
            Err(Error::Zstd(_))
        ));
        assert!(out.is_empty());
        let mut small = [0u8; 10];
        assert!(d
            .decompress_to_slice(&frame, Dictionary::Bytes(&dict), &mut small)
            .is_err());
    }

    #[test]
    fn levels_are_checked() {
        assert!(levels().contains(&DEFAULT_LEVEL));
        let too_high = *levels().end() + 1;
        assert_eq!(
            Compressor::new(too_high).unwrap_err(),
            Error::Level(too_high)
        );
        assert!(EncoderDictionary::new(b"d", too_high).is_err());
        assert!(Error::Level(99).to_string().contains("99"));
        let c = Compressor::new(-5).unwrap();
        assert!(format!("{c:?}").contains("-5"));
        assert!(format!("{:?}", Decompressor::default()).contains("Decompressor"));
        assert!(format!("{:?}", DecoderDictionary::new(b"d")).contains("Decoder"));
        assert!(format!("{:?}", EncoderDictionary::new(b"d", 1).unwrap()).contains("Encoder"));
    }

    #[test]
    fn the_thread_decompressor_is_reused_and_nesting_is_safe() {
        let src = text(1000, 2);
        let mut frame = Vec::new();
        Compressor::new(DEFAULT_LEVEL)
            .unwrap()
            .compress(&src, &mut frame)
            .unwrap();
        let out = with_decompressor(|d| {
            let inner = with_decompressor(|d2| {
                let mut o = Vec::new();
                d2.decompress(&frame, Dictionary::None, 1000, &mut o)
                    .unwrap();
                o
            });
            let mut o = Vec::new();
            d.decompress(&frame, Dictionary::None, 1000, &mut o)
                .unwrap();
            assert_eq!(o, inner);
            o
        });
        assert_eq!(out, src);
    }

    #[cfg(feature = "dict-builder")]
    #[test]
    fn a_trained_dictionary_compresses_small_blocks_better() {
        let samples: Vec<Vec<u8>> = (0..2000)
            .map(|i| {
                format!(
                    "{{\"id\":{i},\"user\":\"user{}\",\"status\":\"ok\",\"tags\":[\"a\",\"b\"]}}",
                    i % 37
                )
                .into_bytes()
            })
            .collect();
        let refs: Vec<&[u8]> = samples.iter().map(Vec::as_slice).collect();
        let dict = train_dictionary(&refs, 4096).unwrap();
        let e = EncoderDictionary::new(&dict, 3).unwrap();
        let dd = DecoderDictionary::new(&dict);
        let mut c = Compressor::new(3).unwrap();
        let (mut plain, mut with) = (Vec::new(), Vec::new());
        c.compress(&samples[5], &mut plain).unwrap();
        c.compress_with_prepared(&e, &samples[5], &mut with)
            .unwrap();
        assert!(with.len() < plain.len());
        let mut out = Vec::new();
        Decompressor::new()
            .decompress(&with, Dictionary::Prepared(&dd), samples[5].len(), &mut out)
            .unwrap();
        assert_eq!(out, samples[5]);
        assert!(train_dictionary(&[], 4096).is_err());
    }
}
