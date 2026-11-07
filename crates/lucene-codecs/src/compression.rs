//! Java's `CompressionMode`s, as the `Lucene90Compressing*` stored-fields and
//! term-vectors formats use them: one self-contained compressed unit per
//! chunk (or per `chunkSize` slice of an oversized stored-fields chunk).
//!
//! - `FAST` and `FAST_DECOMPRESSION`: one LZ4 block (`LZ4.compress` with the
//!   fast or the high-compression hash table).
//! - `HIGH_COMPRESSION`: a vint compressed length, then raw DEFLATE at level 6.
//! - `DUMMY` (the test framework's `DummyCompressingCodec`): the bytes as they
//!   are.
//! - `BEST_SPEED` and `BEST_COMPRESSION` (`LZ4WithPresetDictCompressionMode`,
//!   `DeflateWithPresetDictCompressionMode`): a dictionary plus ten sub-blocks
//!   that may refer back into it, each skippable by its recorded length.
//! - OpenSearch custom-codecs' `ZstdCompressionMode` (`index.codec: zstd`):
//!   the same dictionary-plus-ten-sub-blocks shape in zstd frames, and
//!   `ZstdNoDictCompressionMode` (`zstd_no_dict`): ten independent zstd
//!   frames.
//!
//! [`decompress`] always leaves its input at the end of the unit, so a sliced
//! chunk's units can be read back to back. Java's plain-LZ4 decompressor stops
//! once it has the requested range; this port reads the whole block.

use std::fmt;

use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;
use lucene_store::Result;

use crate::deflate;
use crate::deflater::Deflater;
use crate::lz4;
use crate::lz4_hc;

/// One of Java's compression modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Compression {
    /// `CompressionMode.FAST`.
    Fast,
    /// `CompressionMode.FAST_DECOMPRESSION`.
    FastDecompression,
    /// `CompressionMode.HIGH_COMPRESSION`.
    HighCompression,
    /// `DummyCompressingCodec.DUMMY`.
    Dummy,
    /// `DeflateWithPresetDictCompressionMode` (`Mode.BEST_COMPRESSION`).
    DeflateWithPreset,
    /// `LZ4WithPresetDictCompressionMode` (`Mode.BEST_SPEED`).
    Lz4WithPreset,
    /// OpenSearch custom-codecs' `ZstdCompressionMode`.
    Zstd,
    /// OpenSearch custom-codecs' `ZstdNoDictCompressionMode`.
    ZstdNoDict,
}

impl Compression {
    /// The most one compressed byte can inflate to, a sanity bound on a
    /// chunk's claimed decompressed size.
    pub fn max_expansion(self) -> usize {
        match self {
            Self::Fast | Self::FastDecompression | Self::Lz4WithPreset => LZ4_MAX_EXPANSION,
            Self::HighCompression | Self::DeflateWithPreset => DEFLATE_MAX_EXPANSION,
            Self::Dummy => 1,
            Self::Zstd | Self::ZstdNoDict => ZSTD_MAX_EXPANSION,
        }
    }
}

impl fmt::Display for Compression {
    /// The mode's Java `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Fast => "FAST",
            Self::FastDecompression => "FAST_DECOMPRESSION",
            Self::HighCompression => "HIGH_COMPRESSION",
            Self::Dummy => "DUMMY",
            Self::DeflateWithPreset => "BEST_COMPRESSION",
            Self::Lz4WithPreset => "BEST_SPEED",
            Self::Zstd => "ZSTD",
            Self::ZstdNoDict => "ZSTD_NO_DICT",
        })
    }
}

/// The constructor arguments of `Lucene90CompressingStoredFieldsFormat` and
/// `Lucene90CompressingTermVectorsFormat`, which `CompressingCodec` passes to
/// both: the data file's codec name, the files' segment suffix, the mode and
/// the chunk geometry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompressingFormat {
    pub format_name: &'static str,
    pub segment_suffix: &'static str,
    pub compression: Compression,
    pub chunk_size: i32,
    pub max_docs_per_chunk: i32,
    /// `blockShift` (term vectors call it `blockSize`) of the chunk index.
    pub block_shift: i32,
}

/// The data codec names `CompressingCodec`'s six test codecs write, and their
/// modes: what a reader that only has the file needs to decode it.
pub const COMPRESSING_DATA_CODECS: [(&str, Compression); 6] = [
    ("FastCompressingStoredFieldsData", Compression::Fast),
    (
        "FastDecompressionCompressingStoredFieldsData",
        Compression::FastDecompression,
    ),
    (
        "HighCompressionCompressingStoredFieldsData",
        Compression::HighCompression,
    ),
    ("DummyCompressingStoredFieldsData", Compression::Dummy),
    (
        "DeflateWithPresetCompressingStoredFieldsData",
        Compression::DeflateWithPreset,
    ),
    (
        "LZ4WithPresetCompressingStoredFieldsData",
        Compression::Lz4WithPreset,
    ),
];

/// Readers cannot learn a compressing codec's `maxDocsPerChunk` from its
/// files, so they bound a chunk's document count by this instead, and
/// `CompressingCodec` installations refuse a larger setting.
pub const COMPRESSING_MAX_DOCS_PER_CHUNK: i32 = 1 << 16;

/// The two preset-dictionary framings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Preset {
    Lz4,
    Deflate,
}

impl Compressor {
    /// Java's `Compressor.compress(in, out)`: appends one unit for `payload`.
    // ARITH: none beyond the preset writer's own, which documents them.
    pub fn compress(&mut self, compression: Compression, payload: &[u8], out: &mut Vec<u8>) {
        match compression {
            Compression::Fast => match self.lz4_hc.as_mut() {
                Some(hc) => hc.compress_into(payload, out),
                None => lz4::compress_into(payload, out, &mut self.ht),
            },
            Compression::FastDecompression => {
                let ht = self
                    .high
                    .get_or_insert_with(|| Box::new(lz4::HighCompressionHashTable::new()));
                lz4::compress_into(payload, out, ht.as_mut());
            }
            Compression::HighCompression => {
                let deflater = self
                    .deflater
                    .get_or_insert_with(|| Box::new(Deflater::new()));
                deflater.reset();
                write_deflate_unit(deflater, &mut self.compressed, out, payload);
            }
            Compression::Dummy => out.write_bytes(payload),
            Compression::Lz4WithPreset => write_preset(Preset::Lz4, self, out, payload),
            Compression::DeflateWithPreset => write_preset(Preset::Deflate, self, out, payload),
            Compression::Zstd | Compression::ZstdNoDict => {
                let z = self.zstd.get_or_insert_with(|| {
                    Box::new(
                        lucene_zstd::Compressor::new(lucene_zstd::DEFAULT_LEVEL)
                            .expect("the default level is valid"),
                    )
                });
                write_zstd(
                    z,
                    &mut self.compressed,
                    out,
                    payload,
                    compression == Compression::Zstd,
                );
            }
        }
    }

    /// The zstd level [`Compression::Zstd`] and [`Compression::ZstdNoDict`]
    /// units are written at (`index.codec.compression_level`).
    ///
    /// # Errors
    /// `level` is not a zstd level.
    pub fn set_zstd_level(&mut self, level: i32) -> std::result::Result<(), lucene_zstd::Error> {
        if self.zstd.as_ref().is_some_and(|z| z.level() == level) {
            return Ok(());
        }
        self.zstd = Some(Box::new(lucene_zstd::Compressor::new(level)?));
        Ok(())
    }
}

thread_local! {
    /// [`take_scratch`]'s buffer.
    static SCRATCH: std::cell::Cell<Vec<u8>> = const { std::cell::Cell::new(Vec::new()) };
}

/// A per-thread decompression buffer of at least `len` bytes, handed back by
/// [`Scratch`]'s `Drop`: a preset-dictionary read (every random-access
/// stored-fields document) decompresses its dictionary and one sub-block into
/// it instead of allocating and zeroing `dict + block` bytes per document.
///
/// Its bytes are stale from the last use and never cleared: both
/// decompressors write every byte of the range they are asked for before any
/// back-reference can read it, and the callers check the written length.
fn take_scratch(len: usize) -> Scratch {
    let mut buf = SCRATCH.with(std::cell::Cell::take);
    if buf.len() < len {
        buf.resize(len, 0);
    }
    Scratch(buf)
}

/// See [`take_scratch`].
struct Scratch(Vec<u8>);

impl std::ops::Deref for Scratch {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl std::ops::DerefMut for Scratch {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.0);
        // Keep the larger of this and whatever a nested use put back.
        SCRATCH.with(|cell| {
            let other = cell.take();
            cell.set(if other.capacity() > buf.capacity() {
                other
            } else {
                buf
            });
        });
    }
}

/// Of the unit just decompressed into `out[start..]`, keeps only its bytes
/// `offset..want_end`: the decompressor writes straight into `out` (no
/// scratch buffer, no second copy), and a whole-unit read -- every term
/// vectors chunk, every [`crate::stored_fields::ChunkCursor`] load -- keeps
/// all of it with no move at all.
// ARITH: `start + want_end <= out.len()` since the unit occupies
// `out[start..start + original_length]` and `want_end <= original_length`.
#[allow(clippy::arithmetic_side_effects)]
fn keep_range(out: &mut Vec<u8>, start: usize, offset: usize, want_end: usize) {
    if offset > 0 {
        out.copy_within(start + offset..start + want_end, start);
    }
    out.truncate(start + want_end - offset);
}

/// Java's `Decompressor.decompress(in, originalLength, offset, length,
/// bytes)`: appends bytes `offset..offset + length` of the unit at `input`,
/// whose plain text is `original_length` bytes, to `out`. A unit of no bytes
/// is not read at all, since the writers here never write one.
pub fn decompress(
    compression: Compression,
    input: &mut SliceInput,
    original_length: usize,
    offset: usize,
    length: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    if original_length == 0 {
        return Ok(());
    }
    // ARITH: callers pass a range inside the unit.
    #[allow(clippy::arithmetic_side_effects)]
    let want_end = offset + length;
    if want_end > original_length {
        return Err(lucene_store::Error::Corrupted(format!(
            "range {offset}+{length} outside a {original_length} byte unit"
        )));
    }
    // The unit is decompressed straight into `out[start..end]`.
    let start = out.len();
    let end = start.checked_add(original_length).ok_or_else(|| {
        lucene_store::Error::Corrupted(format!("{original_length} byte unit overflows"))
    })?;
    match compression {
        Compression::Fast | Compression::FastDecompression => {
            out.resize(end, 0);
            if lz4::decompress_slice(input, original_length, out, start)? != end {
                out.truncate(start);
                return Err(lucene_store::Error::Corrupted(
                    "LZ4 block length mismatch".into(),
                ));
            }
            keep_range(out, start, offset, want_end);
        }
        Compression::HighCompression => {
            let compressed_length = read_compressed_length(input)?;
            out.resize(end, 0);
            if let Err(e) =
                deflate::decompress(input, compressed_length, original_length, out, start)
            {
                out.truncate(start);
                return Err(e);
            }
            keep_range(out, start, offset, want_end);
        }
        Compression::Dummy => {
            let at = input.position();
            // Borrowed, not read into a scratch buffer: `slice` bounds-checks
            // the end, and a sum that would overflow is no valid end either.
            let unit = input.slice(at, at.saturating_add(original_length))?;
            out.extend_from_slice(&unit[offset..want_end]);
            input.skip(original_length)?;
        }
        Compression::Lz4WithPreset => {
            decompress_preset(Preset::Lz4, input, original_length, offset, length, out)?
        }
        Compression::DeflateWithPreset => {
            decompress_preset(Preset::Deflate, input, original_length, offset, length, out)?
        }
        Compression::Zstd => decompress_zstd(true, input, original_length, offset, length, out)?,
        Compression::ZstdNoDict => {
            decompress_zstd(false, input, original_length, offset, length, out)?
        }
    }
    Ok(())
}

/// The LZ4 block format's worst-case expansion ratio: the cheapest way to
/// emit output bytes is a match whose length is extended by `0xFF`
/// continuation bytes, each of which contributes 255 bytes, so no LZ4 block
/// inflates to more than 255x its own compressed size. Same constant, same
/// reasoning, as `term_vectors.rs`'s `LZ4_MAX_EXPANSION`.
const LZ4_MAX_EXPANSION: usize = 255;
/// DEFLATE's worst-case expansion ratio, the figure zlib documents: a
/// length-258 match can cost as little as two bits once the Huffman tables
/// are amortised, so 258 * 8 / 2 = 1032 output bytes per input byte.
/// Deliberately the loose end of the range -- this is a ceiling that must
/// never reject a file a DEFLATE mode actually wrote.
const DEFLATE_MAX_EXPANSION: usize = 1032;

/// zstd's worst-case expansion ratio: a frame is at least a 4-byte magic,
/// a 2-byte header and a 3-byte block header, and one RLE block of one byte
/// expands to at most 128 KiB (`ZSTD_BLOCKSIZE_MAX`), with each further block
/// costing 4 bytes. 128 KiB / 4 bounds every frame.
const ZSTD_MAX_EXPANSION: usize = (128 << 10) / 4;

/// `NUM_SUB_BLOCKS` in both `LZ4WithPresetDictCompressionMode` and
/// `DeflateWithPresetDictCompressionMode`: every chunk is framed as a
/// dictionary prefix plus this many equal-sized sub-blocks.
const NUM_SUB_BLOCKS: usize = 10;
/// `LZ4WithPresetDictCompressionMode.DICT_SIZE_FACTOR` -- the dictionary is
/// about 2x smaller than a sub-block.
const LZ4_DICT_SIZE_FACTOR: usize = 2;
/// `DeflateWithPresetDictCompressionMode.DICT_SIZE_FACTOR` -- about 6x
/// smaller than a sub-block.
const DEFLATE_DICT_SIZE_FACTOR: usize = 6;

/// Per-chunk compressor scratch, hoisted out of the flush path exactly as
/// Java's `Compressor` instances are: the LZ4 hash table is 16 kB and is
/// deliberately never cleared, and the two byte buffers are reused across
/// every chunk of a segment.
#[derive(Default)]
pub struct Compressor {
    ht: lz4::FastCompressionHashTable,
    /// `LZ4HighCompressor.ht`, created with the first `FAST_DECOMPRESSION`
    /// unit.
    high: Option<Box<lz4::HighCompressionHashTable>>,
    buffer: Vec<u8>,
    compressed: Vec<u8>,
    lengths: Vec<usize>,
    /// `DeflateWithPresetDictCompressor.compressor` (or `DeflateCompressor`'s
    /// for `HIGH_COMPRESSION`), created with the first DEFLATE unit and kept
    /// for the writer's life: zlib's window
    /// and chain state survive `reset()`, and the bytes depend on them.
    deflater: Option<Box<Deflater>>,
    /// When set, the LZ4 modes (`FAST` and `BEST_SPEED`) find their matches
    /// with [`lz4_hc::Lz4HcCompressor`]: the same LZ4 block format, smaller
    /// and several times slower to write. See [`Self::set_lz4_high_compression`].
    lz4_hc: Option<Box<lz4_hc::Lz4HcCompressor>>,
    /// The zstd context, created with the first zstd unit or by
    /// [`Compressor::set_zstd_level`].
    zstd: Option<Box<lucene_zstd::Compressor>>,
}

impl Compressor {
    /// Writes the LZ4 modes' blocks with the high-compression match finder
    /// (`true`) or with Java's fast hash table (`false`, the default). Any
    /// LZ4 reader, Java's included, reads either.
    pub fn set_lz4_high_compression(&mut self, on: bool) {
        if !on {
            self.lz4_hc = None;
        } else if self.lz4_hc.is_none() {
            self.lz4_hc = Some(Box::default());
        }
    }

    /// Whether [`Self::set_lz4_high_compression`] is on.
    pub fn lz4_high_compression(&self) -> bool {
        self.lz4_hc.is_some()
    }
}

/// Appends one preset-dictionary compression unit for `payload`
/// (`LZ4WithPresetDictCompressionMode` or
/// `DeflateWithPresetDictCompressionMode`). Both emit a dictionary prefix plus
/// [`NUM_SUB_BLOCKS`] sub-blocks, with different geometry and a different
/// compressor; note the vint framing differs too (LZ4 batches every
/// sub-block's compressed length up front, DEFLATE interleaves each one
/// immediately before its bytes -- see [`decompress`]).
// ARITH: for the whole function -- `len` is `payload.len()`, a slice of this
// process's own chunk buffer, so `len <= isize::MAX`. In both arms
// `dict_length <= len / 60 <= len` (`NUM_SUB_BLOCKS * DICT_SIZE_FACTOR` is 20
// or 60, and the LZ4 arm only shrinks it further with `.min(MAX_DISTANCE)`),
// which makes `len - dict_length` non-negative; `block_length` is then at
// most `ceil(len / 10)`, so `dict_length + block_length < len` for any `len`
// past the handful of bytes where both round to zero, and can never leave
// `usize`. The sub-block loop keeps `start < len`, so `len - start` is
// positive, `l <= len - start` bounds `start + l` by `len`, and
// `dict_length + l <= dict_length + block_length` is exactly the buffer
// length resized to above. `compressed.len() - mark` subtracts a length
// recorded before a call that only appends.
#[allow(clippy::arithmetic_side_effects)]
fn write_preset(preset: Preset, scratch: &mut Compressor, out: &mut Vec<u8>, payload: &[u8]) {
    let len = payload.len();
    match preset {
        Preset::Lz4 => {
            let dict_length =
                (len / (NUM_SUB_BLOCKS * LZ4_DICT_SIZE_FACTOR)).min(lz4::MAX_DISTANCE);
            let block_length = (len - dict_length).div_ceil(NUM_SUB_BLOCKS);
            out.write_vint(dict_length as i32);
            out.write_vint(block_length as i32);

            // `resize`, not `clear` + `resize`: this is Java's
            // `ArrayUtil.growNoCopy`, and re-zeroing the whole buffer every
            // chunk would be pure waste. Every byte the compressor reads
            // (`buffer[..dictLength + l]`) is written below before it does;
            // a shorter final sub-block leaves stale bytes past `l`, which
            // are outside the range `compress_with_dictionary` looks at.
            scratch.buffer.resize(dict_length + block_length, 0);
            scratch.compressed.clear();
            scratch.lengths.clear();

            // The dictionary is compressed with no dictionary of its own.
            scratch.buffer[..dict_length].copy_from_slice(&payload[..dict_length]);
            let mark = scratch.compressed.len();
            lz4_unit(scratch, 0, dict_length);
            scratch.lengths.push(scratch.compressed.len() - mark);

            // ...then each sub-block, back-referencing into it.
            let mut start = dict_length;
            while start < len {
                let l = block_length.min(len - start);
                scratch.buffer[dict_length..dict_length + l]
                    .copy_from_slice(&payload[start..start + l]);
                let mark = scratch.compressed.len();
                lz4_unit(scratch, dict_length, l);
                scratch.lengths.push(scratch.compressed.len() - mark);
                start += l;
            }

            // Java writes every unit's compressed length up front (that is
            // what lets a reader skip whole sub-blocks), then the bytes.
            for &l in scratch.lengths.iter() {
                out.write_vint(l as i32);
            }
            out.write_bytes(&scratch.compressed);
        }
        Preset::Deflate => {
            let dict_length = len / (NUM_SUB_BLOCKS * DEFLATE_DICT_SIZE_FACTOR);
            let block_length = (len - dict_length).div_ceil(NUM_SUB_BLOCKS);

            out.write_vint(dict_length as i32);
            out.write_vint(block_length as i32);

            // `DeflateWithPresetDictCompressor.compress`: the dictionary on
            // its own, then every sub-block with the dictionary preset.
            let deflater = scratch
                .deflater
                .get_or_insert_with(|| Box::new(Deflater::new()));
            let dict = &payload[..dict_length];
            deflater.reset();
            write_deflate_unit(deflater, &mut scratch.compressed, out, dict);

            let mut start = dict_length;
            while start < len {
                let this_block = block_length.min(len - start);
                deflater.reset();
                deflater.set_dictionary(dict);
                write_deflate_unit(
                    deflater,
                    &mut scratch.compressed,
                    out,
                    &payload[start..start + this_block],
                );
                start += this_block;
            }
        }
    }
}

/// Appends one LZ4 block for `scratch.buffer[dict_len..][..len]`, with the
/// `dict_len` bytes before it as its dictionary, to `scratch.compressed`.
fn lz4_unit(scratch: &mut Compressor, dict_len: usize, len: usize) {
    match scratch.lz4_hc.as_mut() {
        Some(hc) => {
            hc.compress_with_dictionary(&scratch.buffer, 0, dict_len, len, &mut scratch.compressed)
        }
        None => lz4::compress_with_dictionary(
            &scratch.buffer,
            0,
            dict_len,
            len,
            &mut scratch.compressed,
            &mut scratch.ht,
        ),
    }
}

/// One `DeflateWithPresetDictCompressor.doCompress` unit: a vint compressed
/// length followed by that many raw-DEFLATE bytes.
///
/// The zero-length shortcut is Java's, verbatim (`if (len == 0) { writeVInt(0);
/// return; }`), and it is the common case for small chunks: every chunk under
/// `NUM_SUB_BLOCKS * DEFLATE_DICT_SIZE_FACTOR` (60) bytes has a zero-length
/// dictionary. Compressing the empty input instead would emit a real 2-byte
/// empty DEFLATE stream, which Java's reader does in fact still accept
/// (verified against Lucene 10.5.0 via `VerifyStoredFields`, segment `_3`) --
/// so this is a framing-fidelity and size fix, not a corruption fix.
fn write_deflate_unit(
    deflater: &mut Deflater,
    compressed: &mut Vec<u8>,
    out: &mut Vec<u8>,
    plain: &[u8],
) {
    if plain.is_empty() {
        out.write_vint(0);
        return;
    }
    compressed.clear();
    deflater.compress(plain, compressed);
    out.write_vint(compressed.len() as i32);
    out.write_bytes(compressed);
}

/// One sub-block's *compressed* byte count, as a length rather than as a raw
/// vint: these are only ever used to skip or to bound a decompressor, and a
/// negative one sign-extended through `as usize` into ~2^64 -- which then
/// overflowed the running total the `length == 0` skip path sums up.
fn read_compressed_length(input: &mut SliceInput) -> Result<usize> {
    let len = input.read_vint()?;
    if len < 0 {
        return Err(lucene_store::Error::Corrupted(format!(
            "negative compressed sub-block length: {len}"
        )));
    }
    Ok(len as usize)
}

/// Decompresses the `[offset, offset+length)` slice of one preset-dictionary
/// compression unit (`LZ4WithPresetDictCompressionMode` for
/// `Mode.BEST_SPEED`, `DeflateWithPresetDictCompressionMode` for
/// `Mode.BEST_COMPRESSION`) and appends it to `out`: a dictionary prefix
/// followed by fixed-size sub-blocks, each able to back-reference into the
/// dictionary but not into each other. `input` is always left positioned at
/// the end of the unit, whether or not anything was wanted from it, because
/// a `sliced` chunk's units are read back to back off the same reader.
///
/// The sub-blocks a request does not intersect are **skipped by their
/// recorded compressed length**, not decompressed and thrown away -- this is
/// Java's `Decompressor.decompress(in, originalLength, offset, length, bytes)`
/// contract, and it is why the format bothers to record those lengths at all.
/// Fetching one document out of a full 80kB BEST_SPEED chunk therefore costs
/// one sub-block (~8kB) plus the dictionary, not the whole chunk.
///
/// Both formats share `dictLength`/`blockLength` framing up front, but
/// differ in where each unit's *compressed*-length vint sits relative to
/// its own compressed bytes -- easy to get backwards, so this is worth
/// spelling out precisely:
/// - LZ4 (`LZ4WithPresetDictCompressionMode.readCompressedLengths`) batches
///   **every** unit's compressed length (the dictionary's, then each
///   block's) together up front, before any of the actual compressed bytes.
/// - DEFLATE (`DeflateWithPresetDictCompressionMode.decompress`/
///   `doDecompress`) interleaves each unit's compressed-length vint
///   immediately before that same unit's compressed bytes -- not batched at
///   all. DEFLATE isn't self-terminating, so [`deflate::decompress`] needs
///   that length passed in explicitly, read at the point of use.
fn decompress_preset(
    preset: Preset,
    input: &mut SliceInput,
    original_length: usize,
    offset: usize,
    length: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    if original_length == 0 {
        // The writer emits nothing at all for an empty unit.
        return Ok(());
    }
    // A negative vint sign-extends through `as usize` into ~2^64, which the
    // `> original_length` test below rejects along with every other
    // impossible value -- `original_length` is at most 2^48 (see
    // `read_chunk_header`), so no negative can slip under it.
    let dict_length = input.read_vint()? as usize;
    let block_length = input.read_vint()? as usize;
    // A corrupt header must not spin forever (or allocate wildly): Java
    // divides by `blockLength` when sizing its length array and would throw
    // ArithmeticException on zero, and `new byte[dictLength + blockLength]`
    // would throw rather than abort the process the way `vec![0u8; n]` does.
    // A well-formed unit always has `dictLength <= originalLength`,
    // `blockLength <= originalLength` and `blockLength > 0` whenever the
    // sub-blocks have anything left to cover.
    if dict_length > original_length
        || block_length > original_length
        || (block_length == 0 && dict_length < original_length)
    {
        return Err(lucene_store::Error::Corrupted(format!(
            "invalid compression unit framing: dictLength={dict_length}, blockLength={block_length}, originalLength={original_length}"
        )));
    }
    // ARITH: `dict_length <= original_length` was just established, and
    // `.max(1)` keeps the divisor non-zero. `offset + length` is bounded by
    // `original_length` at both call sites in `decompress_range` -- the
    // non-sliced one passes a sub-range of `0..total_length` with
    // `original_length == total_length`, the sliced one clamps both ends to
    // the unit's own `unit_len` -- and `original_length` is itself at most
    // 2^48, so neither sum can leave `usize`.
    #[allow(clippy::arithmetic_side_effects)]
    let (num_blocks, want_end) = (
        (original_length - dict_length).div_ceil(block_length.max(1)),
        offset + length,
    );
    debug_assert!(want_end <= original_length);

    match preset {
        Preset::Lz4 => {
            let dict_compressed_length = read_compressed_length(input)?;
            // On the stack for every unit a Lucene writer emits (it always
            // cuts `NUM_SUB_BLOCKS` = 10 blocks). Otherwise not
            // `with_capacity(num_blocks)`: `num_blocks` is derived from
            // on-disk lengths, and a corrupt one would reserve gigabytes
            // before the first `read_vint` had a chance to hit EOF.
            let mut inline = [0usize; 16];
            let mut spilled = Vec::new();
            let block_compressed_lengths: &[usize] = if num_blocks <= inline.len() {
                for slot in &mut inline[..num_blocks] {
                    *slot = read_compressed_length(input)?;
                }
                &inline[..num_blocks]
            } else {
                for _ in 0..num_blocks {
                    spilled.push(read_compressed_length(input)?);
                }
                &spilled
            };
            if length == 0 {
                // `checked_add`, folded over the run: these are byte counts
                // off disk, and even with each one bounded to `i32::MAX` a
                // long enough run wraps the total to a small skip that then
                // leaves the reader mid-unit -- the next `sliced` unit would
                // parse compressed bytes as a framing header and decode
                // plausible garbage.
                let mut total = dict_compressed_length;
                for &l in block_compressed_lengths {
                    total = total.checked_add(l).ok_or_else(|| {
                        lucene_store::Error::Corrupted(
                            "compression unit's sub-block lengths overflow".into(),
                        )
                    })?;
                }
                input.skip(total)?;
                return Ok(());
            }

            // ARITH: both lengths were bounded by `original_length` above, so
            // the buffer is at most twice a value already bounded by what the
            // remaining bytes could inflate to (`read_chunk_header`).
            #[allow(clippy::arithmetic_side_effects)]
            let buffer_len = dict_length + block_length;
            let mut scratch = take_scratch(buffer_len);
            let buffer = &mut scratch[..buffer_len];
            // The dictionary is always decompressed: even when none of its
            // bytes are wanted, the sub-blocks back-reference into it.
            if lz4::decompress_slice(input, dict_length, buffer, 0)? != dict_length {
                return Err(lucene_store::Error::Corrupted(
                    "illegal dict length in LZ4 compression unit".into(),
                ));
            }
            if offset < dict_length {
                out.extend_from_slice(&buffer[offset..dict_length.min(want_end)]);
            }

            let mut plain = dict_length;
            // ARITH: `plain` starts at `dict_length <= original_length` and
            // advances by `this_len = min(block_length, original_length -
            // plain)`, so it never passes `original_length`; the loop runs
            // exactly `num_blocks = ceil((original_length - dict_length) /
            // block_length)` times, which is the number of steps that takes.
            // `lo >= plain` by the `.max(plain)` and `hi <= plain + this_len`
            // by the `.min`, so `lo - plain` and `hi - plain` are both in
            // `0..=this_len <= block_length`, keeping every `dict_length + ..`
            // index inside `buffer`.
            #[allow(clippy::arithmetic_side_effects)]
            for &compressed_length in block_compressed_lengths {
                let this_len = block_length.min(original_length - plain);
                let lo = offset.max(plain);
                let hi = want_end.min(plain + this_len);
                if lo < hi {
                    lz4::decompress_slice(input, this_len, buffer, dict_length)?;
                    out.extend_from_slice(
                        &buffer[dict_length + lo - plain..dict_length + hi - plain],
                    );
                } else {
                    input.skip(compressed_length)?;
                }
                plain += this_len;
            }
        }
        Preset::Deflate => {
            let dict_compressed_length = read_compressed_length(input)?;
            if length == 0 {
                input.skip(dict_compressed_length)?;
                for _ in 0..num_blocks {
                    let compressed_length = read_compressed_length(input)?;
                    input.skip(compressed_length)?;
                }
                return Ok(());
            }

            // ARITH: as in the `Lz4` arm -- both lengths are bounded by
            // `original_length`.
            #[allow(clippy::arithmetic_side_effects)]
            let buffer_len = dict_length + block_length;
            let mut scratch = take_scratch(buffer_len);
            let buffer = &mut scratch[..buffer_len];
            deflate::decompress(input, dict_compressed_length, dict_length, buffer, 0)?;
            if offset < dict_length {
                out.extend_from_slice(&buffer[offset..dict_length.min(want_end)]);
            }

            let mut plain = dict_length;
            // ARITH: identical to the `Lz4` arm's loop above.
            #[allow(clippy::arithmetic_side_effects)]
            for _ in 0..num_blocks {
                let this_len = block_length.min(original_length - plain);
                let compressed_length = read_compressed_length(input)?;
                let lo = offset.max(plain);
                let hi = want_end.min(plain + this_len);
                if lo < hi {
                    deflate::decompress(input, compressed_length, this_len, buffer, dict_length)?;
                    out.extend_from_slice(
                        &buffer[dict_length + lo - plain..dict_length + hi - plain],
                    );
                } else {
                    input.skip(compressed_length)?;
                }
                plain += this_len;
            }
        }
    }
    Ok(())
}

/// `ZstdCompressionMode.DICT_SIZE_FACTOR`: the dictionary is about 6x smaller
/// than a sub-block, as for DEFLATE.
const ZSTD_DICT_SIZE_FACTOR: usize = 6;

/// Appends one `ZstdCompressionMode` (`dict`) or `ZstdNoDictCompressionMode`
/// unit for `payload`.
///
/// With a dictionary: `vint dictLength, vint blockLength`, then the
/// dictionary as a plain frame and each of up to [`NUM_SUB_BLOCKS`]
/// sub-blocks as a frame compressed against it, each frame preceded by its
/// vint length (a zero-length dictionary is just the vint `0`). Without:
/// `vint blockLength`, then the sub-blocks the same way. The geometry is
/// Java's to the byte, so Java reads these units and this port reads Java's.
// ARITH: `len` is a slice length; `dict_length <= len / 60` keeps
// `len - dict_length` non-negative, the loop keeps `start < len` and
// `start + l <= len`.
#[allow(clippy::arithmetic_side_effects)]
fn write_zstd(
    z: &mut lucene_zstd::Compressor,
    frame: &mut Vec<u8>,
    out: &mut Vec<u8>,
    payload: &[u8],
    dict: bool,
) {
    let len = payload.len();
    let dict_length = if dict {
        len / (NUM_SUB_BLOCKS * ZSTD_DICT_SIZE_FACTOR)
    } else {
        0
    };
    let block_length = (len - dict_length).div_ceil(NUM_SUB_BLOCKS);
    if dict {
        out.write_vint(dict_length as i32);
    }
    out.write_vint(block_length as i32);
    let dictionary = &payload[..dict_length];
    if dict {
        write_zstd_frame(out, frame, |f| z.compress(dictionary, f), dict_length);
    }
    // Java prepares the dictionary once per unit (`ZstdDictCompress`); so
    // does this, when there is one.
    let prepared = (dict_length > 0).then(|| {
        lucene_zstd::EncoderDictionary::new(dictionary, z.level())
            .expect("the compressor's own level is valid")
    });
    let mut start = dict_length;
    while start < len {
        let l = block_length.min(len - start);
        let block = &payload[start..start + l];
        match &prepared {
            Some(d) => write_zstd_frame(out, frame, |f| z.compress_with_prepared(d, block, f), l),
            None => write_zstd_frame(out, frame, |f| z.compress(block, f), l),
        }
        start += l;
    }
}

/// One `vint compressedLength, frame` pair, or the vint `0` for no bytes.
/// The frame's length precedes it as a vint, so it is compressed into the
/// writer's reusable `frame` buffer first and then copied behind the length.
fn write_zstd_frame(
    out: &mut Vec<u8>,
    frame: &mut Vec<u8>,
    compress: impl FnOnce(&mut Vec<u8>) -> lucene_zstd::Result<usize>,
    plain_len: usize,
) {
    if plain_len == 0 {
        out.write_vint(0);
        return;
    }
    frame.clear();
    // A valid level never fails; the levels are checked where they are set.
    let n = compress(frame).expect("zstd compression at a valid level");
    out.write_vint(n as i32);
    out.extend_from_slice(frame);
}

/// Appends bytes `offset..offset + length` of one zstd unit (see
/// [`write_zstd`]) to `out`, skipping the sub-blocks the range does not
/// touch by their recorded lengths, and leaves `input` at the end of the
/// unit. The dictionary frame is decompressed only when a wanted sub-block
/// needs it.
// ARITH: framing values are checked against `original_length` before use,
// exactly as in `decompress_preset`; `plain` advances by
// `min(block_length, original_length - plain)` so never passes
// `original_length`, and `lo - plain`, `hi - plain` lie in `0..=this_len`.
#[allow(clippy::arithmetic_side_effects)]
fn decompress_zstd(
    dict: bool,
    input: &mut SliceInput,
    original_length: usize,
    offset: usize,
    length: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    let dict_length = if dict { input.read_vint()? as usize } else { 0 };
    let block_length = input.read_vint()? as usize;
    if dict_length > original_length
        || block_length > original_length
        || (block_length == 0 && dict_length < original_length)
    {
        return Err(lucene_store::Error::Corrupted(format!(
            "invalid zstd unit framing: dictLength={dict_length}, blockLength={block_length}, originalLength={original_length}"
        )));
    }
    let num_blocks = (original_length - dict_length).div_ceil(block_length.max(1));
    let want_end = offset + length;

    let dict_frame = if dict {
        let n = read_compressed_length(input)?;
        if (n == 0) != (dict_length == 0) {
            return Err(lucene_store::Error::Corrupted(format!(
                "zstd dictionary of {dict_length} bytes in a {n} byte frame"
            )));
        }
        let at = input.position();
        let frame = input.slice(at, at.saturating_add(n))?;
        input.skip(n)?;
        frame
    } else {
        &[]
    };
    // The dictionary goes into this thread's reusable scratch buffer, as
    // the preset-dictionary modes' does.
    let mut scratch = take_scratch(dict_length);
    let dictionary = &mut scratch[..dict_length];
    let mut dict_ready = dict_length == 0;
    let mut ensure_dict = |dictionary: &mut [u8]| -> Result<()> {
        if !dict_ready {
            let n = lucene_zstd::with_decompressor(|d| {
                d.decompress_to_slice(dict_frame, lucene_zstd::Dictionary::None, dictionary)
            })
            .map_err(|e| lucene_store::Error::Corrupted(format!("zstd stored fields unit: {e}")))?;
            if n != dict_length {
                return Err(lucene_store::Error::Corrupted(format!(
                    "zstd dictionary frame holds {n} bytes, expected {dict_length}"
                )));
            }
            dict_ready = true;
        }
        Ok(())
    };
    if length > 0 && offset < dict_length {
        ensure_dict(dictionary)?;
        out.extend_from_slice(&dictionary[offset..dict_length.min(want_end)]);
    }

    let mut plain = dict_length;
    for _ in 0..num_blocks {
        let this_len = block_length.min(original_length - plain);
        let n = read_compressed_length(input)?;
        let lo = offset.max(plain);
        let hi = want_end.min(plain + this_len);
        if length > 0 && lo < hi {
            let at = input.position();
            let frame = input.slice(at, at.saturating_add(n))?;
            ensure_dict(dictionary)?;
            let d = if dict_length > 0 {
                lucene_zstd::Dictionary::Bytes(dictionary)
            } else {
                lucene_zstd::Dictionary::None
            };
            let start = out.len();
            zstd_frame(frame, d, this_len, out)?;
            keep_range(out, start, lo - plain, hi - plain);
        }
        input.skip(n)?;
        plain += this_len;
    }
    Ok(())
}

/// One zstd frame of exactly `plain_len` bytes appended to `out`, on this
/// thread's reusable decompression context.
fn zstd_frame(
    frame: &[u8],
    dict: lucene_zstd::Dictionary<'_>,
    plain_len: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    lucene_zstd::with_decompressor(|d| d.decompress(frame, dict, plain_len, out))
        .map_err(|e| lucene_store::Error::Corrupted(format!("zstd stored fields unit: {e}")))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    const ALL: [Compression; 8] = [
        Compression::Fast,
        Compression::FastDecompression,
        Compression::HighCompression,
        Compression::Dummy,
        Compression::DeflateWithPreset,
        Compression::Lz4WithPreset,
        Compression::Zstd,
        Compression::ZstdNoDict,
    ];

    fn payload(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| b"abcabcxyz0123"[(i * 7 + i / 13) % 13])
            .collect()
    }

    #[test]
    fn every_mode_round_trips_any_range_and_leaves_the_input_after_its_unit() {
        for compression in ALL {
            let mut compressor = Compressor::default();
            for len in [1, 7, 60, 300, 5000] {
                let first = payload(len);
                let second = payload(len + 3);
                let mut out = Vec::new();
                compressor.compress(compression, &first, &mut out);
                compressor.compress(compression, &second, &mut out);
                for (offset, length) in [(0, len), (len / 3, len / 2), (len - 1, 1), (0, 0)] {
                    let mut input = SliceInput::new(&out);
                    let mut plain = Vec::new();
                    decompress(compression, &mut input, len, offset, length, &mut plain).unwrap();
                    assert_eq!(plain, first[offset..offset + length], "{compression} {len}");
                    let mut rest = Vec::new();
                    decompress(compression, &mut input, len + 3, 0, len + 3, &mut rest).unwrap();
                    assert_eq!(rest, second, "{compression} {len}");
                    assert_eq!(input.position(), out.len());
                }
            }
        }
    }

    #[test]
    fn a_range_past_the_unit_is_corruption() {
        let mut out = Vec::new();
        Compressor::default().compress(Compression::Dummy, b"abc", &mut out);
        let mut plain = Vec::new();
        let error = decompress(
            Compression::Dummy,
            &mut SliceInput::new(&out),
            3,
            2,
            2,
            &mut plain,
        );
        assert!(matches!(error, Err(lucene_store::Error::Corrupted(_))));
    }

    #[test]
    fn displays_are_javas() {
        let names: Vec<String> = ALL.iter().map(ToString::to_string).collect();
        assert_eq!(
            names,
            [
                "FAST",
                "FAST_DECOMPRESSION",
                "HIGH_COMPRESSION",
                "DUMMY",
                "BEST_COMPRESSION",
                "BEST_SPEED",
                "ZSTD",
                "ZSTD_NO_DICT"
            ]
        );
    }
}
