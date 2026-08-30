//! Port of `org.apache.lucene.codecs.lucene90.Lucene90StoredFieldsFormat`
//! (`.fdt` data + `.fdx` index + `.fdm` meta) — read-only. Both compression
//! modes are supported: `Mode.BEST_SPEED` (the default; LZ4, ~80KB chunks)
//! and `Mode.BEST_COMPRESSION` (DEFLATE, ~480KB chunks). The mode is baked
//! into the `.fdt` data codec name itself (`...FastData` vs `...HighData`),
//! so `open` detects it there rather than needing the caller to specify it;
//! see [`Mode`] and [`decompress_unit`].
//!
//! Stored fields (the original field values, as opposed to their indexed or
//! doc-values forms) are grouped into **chunks** of up to ~1024 (BEST_SPEED)
//! or ~4096 (BEST_COMPRESSION) documents each, concatenated and compressed
//! together (better ratio than per-document compression). Three files:
//! - `.fdt`: `IndexHeader, <chunk>*, Footer`. Each chunk: `docBase` (vint),
//!   a `token` (vint: `chunkDocs = token >>> 2`, `sliced = token & 1`,
//!   `dirty = token & 2` -- the last only matters to a writer's merge
//!   heuristics, ignored here), each doc's field count and length (via
//!   [`read_bulk_ints`]), then the compressed payload -- one
//!   [`decompress_unit`] if `!sliced`, or several `chunk_size`-decompressed
//!   units back to back if `sliced` (only large chunks get split this way;
//!   `chunk_size` is read from `.fdm`, not hardcoded, since it differs
//!   between the two modes).
//! - `.fdx`: `IndexHeader, <two DirectMonotonicReader-encoded arrays>, Footer`
//!   -- chunk doc-bases and chunk file-offsets, giving O(log chunks) lookup
//!   from a doc id to its chunk's `.fdt` offset.
//! - `.fdm`: metadata about those two arrays (this port merges `.fdm`+`.fdx`
//!   parsing into one [`open`] call, since one is meaningless without the
//!   other and Java itself always opens both together).
//!
//! Per-document payload (once decompressed): `numStoredFields` entries of
//! `infoAndBits` (vlong: field number `<< 3 | type tag`) followed by the
//! field's value in one of six encodings -- see [`visit_field`].

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

use crate::deflate;
use crate::direct_monotonic;
use crate::lz4;

const DATA_CODEC_BEST_SPEED: &str = "Lucene90StoredFieldsFastData";
const DATA_CODEC_BEST_COMPRESSION: &str = "Lucene90StoredFieldsHighData";
const META_CODEC: &str = "Lucene90FieldsIndexMeta";
const INDEX_CODEC: &str = "Lucene90FieldsIndexIdx";
const VERSION_START: i32 = 1;
const VERSION_CURRENT: i32 = 1;
const META_VERSION_START: i32 = 0;
const INDEX_VERSION_START: i32 = 0;
const INDEX_VERSION_CURRENT: i32 = 0;

const TYPE_STRING: i64 = 0x00;
const TYPE_BYTE_ARR: i64 = 0x01;
const TYPE_NUMERIC_INT: i64 = 0x02;
const TYPE_NUMERIC_FLOAT: i64 = 0x03;
const TYPE_NUMERIC_LONG: i64 = 0x04;
const TYPE_NUMERIC_DOUBLE: i64 = 0x05;
const TYPE_BITS: i64 = 3;
const TYPE_MASK: i64 = (1 << TYPE_BITS) - 1;

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
/// never reject a file `Mode.BEST_COMPRESSION` actually wrote.
const DEFLATE_MAX_EXPANSION: usize = 1032;

const SECOND: i64 = 1000;
const HOUR: i64 = 60 * 60 * SECOND;
const DAY: i64 = 24 * HOUR;
const SECOND_ENCODING: u8 = 0x40;
const HOUR_ENCODING: u8 = 0x80;
const DAY_ENCODING: u8 = 0xC0;
const DAY_ENCODING_MASK: u8 = 0xC0;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("doc {0} is out of range (maxDoc={1})")]
    DocOutOfRange(i32, i32),
    #[error("corrupted chunk: docID={doc_id}, docBase={doc_base}, chunkDocs={chunk_docs}, maxDoc={max_doc}")]
    CorruptChunkBounds {
        doc_id: i32,
        doc_base: i32,
        chunk_docs: i32,
        max_doc: i32,
    },
    #[error("corrupted stored fields: length={length}, numStoredFields={num_stored_fields}")]
    LengthFieldCountMismatch { length: i64, num_stored_fields: i64 },
    #[error("unsupported bits-per-value in bulk int array: {0}")]
    UnsupportedBulkIntWidth(u8),
    #[error("unknown stored field type tag: {0:#x}")]
    UnknownTypeTag(i64),
    #[error("index meta's numChunks ({index_num_chunks}) should be exactly one more than the outer meta's ({outer_num_chunks}) -- the index arrays carry one extra sentinel entry")]
    NumChunksMismatch {
        index_num_chunks: i64,
        outer_num_chunks: i64,
    },
    #[error("more dirty chunks ({0}) than chunks ({1})")]
    TooManyDirtyChunks(i64, i64),
    #[error("dirty chunks ({0}) and dirty docs ({1}) must both be zero or both nonzero")]
    DirtyChunksDocsMismatch(i64, i64),
    #[error("more dirty chunks ({0}) than dirty docs ({1})")]
    TooManyDirtyDocsChunks(i64, i64),
    #[error("cannot bulk-copy chunks from this reader: sameCompressionMode={same_mode}, readerChunkSize={reader_chunk_size}, writerChunkSize={writer_chunk_size}, tooDirty={too_dirty}")]
    BulkCopyNotPermitted {
        reader_chunk_size: i32,
        writer_chunk_size: i32,
        same_mode: bool,
        too_dirty: bool,
    },
    #[error("inverted document range: fromDoc={from_doc} > toDoc={to_doc}")]
    InvertedDocRange { from_doc: i32, to_doc: i32 },
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    String(String),
    Binary(Vec<u8>),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
}

#[derive(Debug, Clone)]
pub struct StoredField {
    pub field_number: i32,
    pub value: FieldValue,
}

#[derive(Debug, Clone, Default)]
pub struct Document {
    pub fields: Vec<StoredField>,
}

/// Which per-unit compressor was used to write this segment's `.fdt` --
/// baked into the data codec name itself, so `open` detects it from the
/// header rather than needing the caller to specify it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    BestSpeed,
    BestCompression,
}

/// Parsed `.fdm` metadata plus the `.fdx`-relative pointers/arrays it
/// describes; `document()` also needs the whole `.fdt` file's bytes.
pub struct StoredFieldsReader<'d> {
    fdt: &'d [u8],
    fdx: &'d [u8],
    mode: Mode,
    chunk_size: i32,
    max_doc: i32,
    num_chunks: i64,
    num_dirty_chunks: i64,
    num_dirty_docs: i64,
    max_pointer: i64,
    docs_start_pointer: i64,
    docs_end_pointer: i64,
    docs_meta: direct_monotonic::Meta,
    start_pointers_start_pointer: i64,
    start_pointers_end_pointer: i64,
    start_pointers_meta: direct_monotonic::Meta,
}

/// Parses `.fdt`+`.fdm`+`.fdx` (already read into memory) and returns a
/// reader over `fdt`/`fdx`'s bytes. Both `Mode.BEST_SPEED` (LZ4, the
/// default) and `Mode.BEST_COMPRESSION` (DEFLATE) are supported -- the mode
/// is detected from the `.fdt` data codec name itself, which differs per
/// mode (`Lucene90StoredFieldsFastData` vs `...HighData`).
pub fn open<'d>(
    fdt: &'d [u8],
    fdx: &'d [u8],
    fdm: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<StoredFieldsReader<'d>> {
    let mut fdt_input = SliceInput::new(fdt);
    // The data codec name is mode-specific (`...FastData` for BEST_SPEED,
    // `...HighData` for BEST_COMPRESSION); peek it before the real header
    // check so we know which one to expect (and which compressor to use
    // later), then rewind -- `check_index_header` re-reads it from scratch.
    let header_start = fdt_input.position();
    let peek_magic = fdt_input.read_be_u32()?;
    if peek_magic != codec_util::CODEC_MAGIC {
        return Err(lucene_store::Error::Corrupted(format!(
            "codec header mismatch: actual header={peek_magic:#x} vs expected header={:#x}",
            codec_util::CODEC_MAGIC
        ))
        .into());
    }
    let data_codec = fdt_input.read_string()?;
    let mode = match data_codec.as_str() {
        DATA_CODEC_BEST_SPEED => Mode::BestSpeed,
        DATA_CODEC_BEST_COMPRESSION => Mode::BestCompression,
        other => {
            return Err(lucene_store::Error::Corrupted(format!(
                "unknown stored fields data codec: {other}"
            ))
            .into())
        }
    };
    fdt_input.seek(header_start)?;

    let fdt_header = codec_util::check_index_header(
        &mut fdt_input,
        &data_codec,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    codec_util::retrieve_checksum(fdt)?;

    let mut meta_input = SliceInput::new(fdm);
    codec_util::check_index_header(
        &mut meta_input,
        META_CODEC,
        META_VERSION_START,
        fdt_header.version,
        segment_id,
        segment_suffix,
    )?;
    let chunk_size = meta_input.read_vint()?;
    // Java never validates this (its writer's `chunkSize` is a positive
    // constant), but the reader divides a `sliced` chunk by it, so a corrupt
    // zero or negative would either loop forever or wrap.
    if chunk_size <= 0 {
        return Err(lucene_store::Error::Corrupted(format!(
            "stored fields chunkSize must be positive, got {chunk_size}"
        ))
        .into());
    }

    let max_doc = meta_input.read_i32()?;
    // Java takes `numDocs` from `SegmentInfo.maxDoc()`, an already-validated
    // value, and only ever compares the `.fdm`'s own `maxDoc` against a doc
    // id (`Objects.checkIndex`). This port has no `SegmentInfo` to hand, so
    // the `.fdm` copy *is* the document count -- and every chunk-bounds check
    // below is stated relative to it. A negative one would make
    // `doc_base + chunk_docs > max_doc` accept anything.
    if max_doc < 0 {
        return Err(lucene_store::Error::Corrupted(format!(
            "stored fields maxDoc must not be negative, got {max_doc}"
        ))
        .into());
    }
    let block_shift = meta_input.read_i32()? as u32;
    // The index arrays (`docs`/`startPointers`) carry `totalChunks + 1`
    // entries, not `totalChunks`: a sentinel final entry (doc base ==
    // `maxDoc`, start pointer == `maxPointer`) that lets index-array code
    // treat the last real chunk uniformly with the rest, at the cost of
    // this index count differing by exactly one from the "real" chunk
    // count read below (see `FieldsIndexWriter.finish`).
    let index_num_chunks = meta_input.read_i32()? as i64;
    let docs_start_pointer = meta_input.read_i64()?;
    let docs_meta = direct_monotonic::load_meta(&mut meta_input, index_num_chunks, block_shift)?;
    let docs_end_pointer = meta_input.read_i64()?;
    let start_pointers_start_pointer = docs_end_pointer;
    let start_pointers_meta =
        direct_monotonic::load_meta(&mut meta_input, index_num_chunks, block_shift)?;
    let start_pointers_end_pointer = meta_input.read_i64()?;
    let max_pointer = meta_input.read_i64()?;

    let num_chunks = meta_input.read_vlong()?;
    // A vlong is nine bytes wide, so `num_chunks` can arrive negative or at
    // `i64::MAX`; `num_chunks + 1` overflows at the latter. `num_chunks` is
    // handed to `direct_monotonic::floor_index` as a search bound on every
    // document lookup, so pin it non-negative here rather than relying on
    // that function's own guard.
    if num_chunks < 0 || num_chunks.checked_add(1) != Some(index_num_chunks) {
        return Err(Error::NumChunksMismatch {
            index_num_chunks,
            outer_num_chunks: num_chunks,
        });
    }
    let num_dirty_chunks = meta_input.read_vlong()?;
    let num_dirty_docs = meta_input.read_vlong()?;
    if num_chunks < num_dirty_chunks {
        return Err(Error::TooManyDirtyChunks(num_dirty_chunks, num_chunks));
    }
    if (num_dirty_chunks == 0) != (num_dirty_docs == 0) {
        return Err(Error::DirtyChunksDocsMismatch(
            num_dirty_chunks,
            num_dirty_docs,
        ));
    }
    if num_dirty_docs < num_dirty_chunks {
        return Err(Error::TooManyDirtyDocsChunks(
            num_dirty_chunks,
            num_dirty_docs,
        ));
    }
    codec_util::check_footer(&mut meta_input, fdm.len())?;

    let mut fdx_input = SliceInput::new(fdx);
    codec_util::check_index_header(
        &mut fdx_input,
        INDEX_CODEC,
        INDEX_VERSION_START,
        INDEX_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    codec_util::retrieve_checksum(fdx)?;

    // `max_pointer` marks where the last chunk's compressed bytes end, i.e.
    // exactly where the footer must start -- a cheap real corruption check
    // (truncation, or a `.fdt` from a different segment), same spirit as
    // `compound_format`'s total-length cross-check.
    //
    // `max_pointer` is a raw `i64` off the `.fdm`, so both a negative one and
    // one near `i64::MAX` have to be rejected *before* the comparison rather
    // than through it: `(-1i64) as usize + FOOTER_LENGTH` wraps to 15, which
    // a 15-byte `.fdt` would then match. Folding the conversion and the
    // addition into one fallible step is what makes the equality below mean
    // what it says -- and it is what establishes the invariant every later
    // `max_pointer as usize` in this file relies on:
    // `0 <= max_pointer <= fdt.len() - FOOTER_LENGTH`.
    let expected_fdt_len = usize::try_from(max_pointer)
        .ok()
        .and_then(|p| p.checked_add(codec_util::FOOTER_LENGTH));
    if expected_fdt_len != Some(fdt.len()) {
        return Err(lucene_store::Error::Corrupted(format!(
            ".fdt length should be maxPointer={max_pointer} + footer bytes, but is {}",
            fdt.len()
        ))
        .into());
    }

    Ok(StoredFieldsReader {
        fdt,
        fdx,
        mode,
        chunk_size,
        max_doc,
        num_chunks,
        num_dirty_chunks,
        num_dirty_docs,
        max_pointer,
        docs_start_pointer,
        docs_end_pointer,
        docs_meta,
        start_pointers_start_pointer,
        start_pointers_end_pointer,
        start_pointers_meta,
    })
}

impl<'d> StoredFieldsReader<'d> {
    pub fn max_doc(&self) -> i32 {
        self.max_doc
    }

    /// The `maxDocsPerChunk` the writer of this `.fdt` must have used --
    /// determined by the data codec name `open` matched, since
    /// `Lucene90StoredFieldsFormat.impl` is the only thing that writes either
    /// name and it pins the value per mode. Used as a hard upper bound on a
    /// chunk header's document count; see [`Self::read_chunk_header`].
    fn max_docs_per_chunk(&self) -> i32 {
        match self.mode {
            Mode::BestSpeed => BEST_SPEED_MAX_DOCS_PER_CHUNK as i32,
            Mode::BestCompression => BEST_COMPRESSION_MAX_DOCS_PER_CHUNK as i32,
        }
    }

    /// The most this segment's compressor can inflate a byte by, per
    /// [`LZ4_MAX_EXPANSION`]/[`DEFLATE_MAX_EXPANSION`].
    fn max_expansion(&self) -> usize {
        match self.mode {
            Mode::BestSpeed => LZ4_MAX_EXPANSION,
            Mode::BestCompression => DEFLATE_MAX_EXPANSION,
        }
    }

    fn docs_region(&self) -> Result<&'d [u8]> {
        self.fdx
            .get(self.docs_start_pointer as usize..self.docs_end_pointer as usize)
            .ok_or(lucene_store::Error::Eof { offset: 0 }.into())
    }

    fn start_pointers_region(&self) -> Result<&'d [u8]> {
        self.fdx
            .get(
                self.start_pointers_start_pointer as usize
                    ..self.start_pointers_end_pointer as usize,
            )
            .ok_or(lucene_store::Error::Eof { offset: 0 }.into())
    }

    fn block_start_pointer(&self, block_index: i64) -> Result<i64> {
        Ok(direct_monotonic::get(
            self.start_pointers_region()?,
            &self.start_pointers_meta,
            block_index,
        )?)
    }

    /// Reads the given document's stored fields.
    ///
    /// A **random-access** read (see [`Self::serialized_document`]); a
    /// sequential scan should go through [`ChunkCursor`] and
    /// [`parse_document`] instead.
    pub fn document(&self, doc_id: i32) -> Result<Document> {
        let doc = self.serialized_document(doc_id)?;
        parse_document(doc.num_stored_fields, &doc.bytes)
    }

    /// `StoredFields.document(docID, StoredFieldVisitor)`: reads the given
    /// document's stored fields, asking `visitor` about each one *before*
    /// decoding it.
    ///
    /// This is the shape Java retrieves a hit's fields in, and the reason it
    /// exists here is [`Self::document`]'s cost: that method allocates a
    /// `String` or `Vec` for every field of the document, so pulling one
    /// field out of a wide document pays for all the others. With a visitor,
    /// an unwanted `STRING`/`BYTE_ARR` field costs its length vint and a
    /// cursor bump ([`skip_field`]), and [`VisitStatus::Stop`] ends the
    /// document early.
    ///
    /// The chunk containing `doc_id` is still decompressed (only over this
    /// document's own byte range -- see [`Self::serialized_document`]); what
    /// the visitor saves is the per-field decode and its allocations, not the
    /// I/O.
    pub fn visit_document(&self, doc_id: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        let doc = self.serialized_document(doc_id)?;
        visit_document(doc.num_stored_fields, &doc.bytes, visitor)
    }

    /// Reads the chunk containing `doc_id`: its header (doc base, doc count,
    /// `sliced` flag, per-document field counts and payload offsets) plus an
    /// input positioned on the first compression unit.
    fn read_chunk_header(&self, doc_id: i32) -> Result<(ChunkHeader, SliceInput<'d>)> {
        if doc_id < 0 || doc_id >= self.max_doc {
            return Err(Error::DocOutOfRange(doc_id, self.max_doc));
        }

        let block_index = direct_monotonic::floor_index(
            self.docs_region()?,
            &self.docs_meta,
            0,
            self.num_chunks,
            doc_id as i64,
        )?;
        let block_start = self.block_start_pointer(block_index)?;

        let mut input = SliceInput::new(self.fdt);
        input.seek(block_start as usize)?;
        let doc_base = input.read_vint()?;
        let token = input.read_vint()?;
        // Java's `token >>> 2`: an *unsigned* shift, so a corrupt token with
        // its sign bit set yields a large positive count there and must here
        // too. A signed `>>` gave a negative `chunk_docs`, and `chunk_docs as
        // usize` on a negative value is ~2^64 -- the count that sizes both
        // `read_bulk_ints`' `vec![0i64; count]` and `offsets` below.
        let chunk_docs = ((token as u32) >> 2) as i32;
        // Every bound `chunk_docs` needs, established once, before it is used
        // as a length. Three of the five conditions below are Java's
        // `contains(docID)` and `docBase + chunkDocs > numDocs`, restated so
        // that no intermediate can overflow: `doc_id - doc_base` is evaluated
        // only once
        // `doc_base <= doc_id` holds (so it lands in `0..=i32::MAX`), and
        // `max_doc - chunk_docs` only once `0 <= chunk_docs <= 4096` and
        // `max_doc >= 0` do (so it lands in `-4096..=i32::MAX`).
        //
        // `chunk_docs <= max_docs_per_chunk` is the one bound Java does not
        // state, and it is what turns the count into something safe to
        // allocate from. It is exact rather than defensive: the data codec
        // name this reader matched in `open` is written by exactly one Java
        // class -- `Lucene90StoredFieldsFormat.impl`, whose two `new
        // Lucene90CompressingStoredFieldsFormat(...)` calls pin
        // `maxDocsPerChunk` at 1024 (`...FastData`) and 4096
        // (`...HighData`) -- and `Lucene90CompressingStoredFieldsWriter`
        // re-checks `numBufferedDocs >= maxDocsPerChunk` after every single
        // document, so no chunk it (or this port's writer, or its bulk-copy
        // path) emits can exceed it. Without it, `maxDoc` and a token are the
        // only things standing between a 40-byte file and a
        // `vec![0i64; 2^31]` -- a 17 GB reservation, which *aborts* rather
        // than unwinding (`docs/arithmetic-gate.md`).
        //
        // ARITH: `&&` short-circuits left to right, so `doc_id - doc_base` is
        // only evaluated under `doc_id >= doc_base >= 0` (difference in
        // `0..=i32::MAX`) and `self.max_doc - chunk_docs` only under
        // `0 <= chunk_docs <= 4096` with `max_doc >= 0` (`open` rejects a
        // negative one), so the difference is in `-4096..=i32::MAX`.
        #[allow(clippy::arithmetic_side_effects)]
        let bounds_ok = doc_base >= 0
            && chunk_docs <= self.max_docs_per_chunk()
            && doc_id >= doc_base
            && doc_id - doc_base < chunk_docs
            && doc_base <= self.max_doc - chunk_docs;
        if !bounds_ok {
            return Err(Error::CorruptChunkBounds {
                doc_id,
                doc_base,
                chunk_docs,
                max_doc: self.max_doc,
            });
        }
        let sliced = token & 1 != 0;

        let (num_stored_fields, offsets) = if chunk_docs == 1 {
            let n = input.read_vint()? as i64;
            let len = input.read_vint()? as i64;
            if n < 0 || len < 0 {
                return Err(Error::LengthFieldCountMismatch {
                    length: len,
                    num_stored_fields: n,
                });
            }
            (vec![n], vec![0i64, len])
        } else {
            let num_stored_fields = read_bulk_ints(&mut input, chunk_docs as usize)?;
            let raw_lengths = read_bulk_ints(&mut input, chunk_docs as usize)?;
            // ARITH: `chunk_docs` is in `1..=4096` (checked above), so
            // `chunk_docs + 1` is far inside `i32`, and both `as usize` casts
            // are of a non-negative value.
            #[allow(clippy::arithmetic_side_effects)]
            let mut offsets = Vec::with_capacity(chunk_docs as usize + 1);
            offsets.push(0i64);
            // ARITH: `read_bulk_ints` guarantees every value it returns is
            // non-negative and at most `u32::MAX` (the widest shape is
            // 32-bit, read unsigned; the all-equal shape rejects a negative
            // constant), so over at most 4096 documents the running sum stays
            // under 2^44. Non-negativity is also what keeps `offsets`
            // monotonic, which `serialized_document`'s
            // `offsets[i + 1] as usize - offsets[i]` depends on -- a
            // decreasing pair underflowed that to a ~2^64 slice length.
            #[allow(clippy::arithmetic_side_effects)]
            for (i, &len) in raw_lengths.iter().enumerate() {
                offsets.push(offsets[i] + len);
                // Java's "only the empty document has a serialized length of
                // 0" check, folded into the same pass rather than a second
                // one over the same two arrays.
                if (len == 0) != (num_stored_fields[i] == 0) {
                    return Err(Error::LengthFieldCountMismatch {
                        length: len,
                        num_stored_fields: num_stored_fields[i],
                    });
                }
            }
            (num_stored_fields, offsets)
        };
        // The chunk's decompressed size is claimed by its own header and
        // related to nothing on the wire, yet it sizes `decompress_range`'s
        // output `Vec` and `decompress_unit`'s `vec![0u8; dictLength +
        // blockLength]`. Bound it by what the bytes that are left could
        // possibly inflate to -- checked **once per chunk**, never per
        // document or per sub-block. See [`LZ4_MAX_EXPANSION`] and
        // [`DEFLATE_MAX_EXPANSION`] for where the two ratios come from.
        //
        // This is a ceiling, not a tight bound: on a large `.fdt` it rejects
        // nothing a well-formed chunk could claim. What it does remove is the
        // shape that matters -- a small file whose header names a size no
        // decompressor could ever produce from it.
        let total_length = *offsets.last().unwrap_or(&0);
        // Every path into `offsets` above is non-negative; this pins the
        // claim that makes the `as u64` below, and every `as usize` in
        // `decompress_range`, faithful rather than sign-extending.
        debug_assert!(total_length >= 0);
        let compressed_left = self.fdt.len().saturating_sub(input.position());
        let ceiling = compressed_left.saturating_mul(self.max_expansion()) as u64;
        if total_length as u64 > ceiling {
            return Err(lucene_store::Error::Corrupted(format!(
                "stored fields chunk at docBase={doc_base} claims {total_length} decompressed \
                 bytes, which {compressed_left} remaining bytes cannot produce"
            ))
            .into());
        }

        Ok((
            ChunkHeader {
                doc_base,
                chunk_docs,
                sliced,
                num_stored_fields,
                offsets,
            },
            input,
        ))
    }

    /// Decompresses `header`'s payload bytes `want_start..want_end` from
    /// `input` (positioned on the chunk's first compression unit), the way
    /// Java's `Decompressor.decompress(in, originalLength, offset, length,
    /// bytes)` does: whole sub-blocks the range does not intersect are
    /// skipped by their recorded compressed length rather than inflated and
    /// discarded (see [`decompress_unit`]).
    fn decompress_range(
        &self,
        header: &ChunkHeader,
        input: &mut SliceInput<'d>,
        want_start: usize,
        want_end: usize,
    ) -> Result<Vec<u8>> {
        // ARITH: the invariant the whole function rests on is that `header` can
        // only have come from `read_chunk_header`, which established that
        // `offsets` is non-decreasing from 0 and that its last entry (this
        // `total_length`) is non-negative, under 2^44, and no larger than the
        // remaining `.fdt` bytes could inflate to. Both call sites derive
        // `want_start <= want_end <= total_length` from that same array:
        // `serialized_document` passes `offsets[i] .. offsets[i+1]` and
        // `read_chunk` passes `0 .. total_length`. So every subtraction below
        // is of a smaller value from a larger one, and `unit_start +
        // unit_len` is at most `total_length`.
        let total_length = header.total_length() as usize;
        debug_assert!(want_start <= want_end && want_end <= total_length);
        // ARITH: `want_start <= want_end`, per the invariant above.
        #[allow(clippy::arithmetic_side_effects)]
        let wanted = want_end - want_start;
        let sliced = header.sliced;
        let mut doc_bytes = Vec::with_capacity(wanted);
        if sliced {
            // A `sliced` chunk is several independent `chunk_size`-plaintext
            // units back to back (`Lucene90CompressingStoredFieldsWriter.flush`
            // slices the buffer at `chunkSize`), so the wanted range has to be
            // cut against each unit's own extent.
            let chunk_size = self.chunk_size as usize; // validated positive in `open`
            let mut unit_start = 0usize;
            // ARITH: the loop guard gives `unit_start < total_length`, so
            // `total_length - unit_start` is positive and `unit_len` is at
            // most that -- hence `unit_start + unit_len <= total_length`,
            // which both bounds the sum and re-establishes the guard's
            // precondition for the next iteration. `lo >= unit_start` by the
            // `.max(unit_start)`.
            #[allow(clippy::arithmetic_side_effects)]
            while unit_start < total_length && unit_start < want_end {
                let unit_len = chunk_size.min(total_length - unit_start);
                let lo = want_start.max(unit_start).min(unit_start + unit_len);
                let hi = want_end.min(unit_start + unit_len);
                decompress_unit(
                    self.mode,
                    input,
                    unit_len,
                    lo - unit_start,
                    hi.saturating_sub(lo),
                    &mut doc_bytes,
                )?;
                unit_start += unit_len;
            }
        } else {
            decompress_unit(
                self.mode,
                input,
                total_length,
                want_start,
                wanted,
                &mut doc_bytes,
            )?;
        }
        if doc_bytes.len() != wanted {
            return Err(lucene_store::Error::Corrupted(format!(
                "expected {wanted} decompressed bytes, got {}",
                doc_bytes.len()
            ))
            .into());
        }
        Ok(doc_bytes)
    }

    /// Port of `Lucene90CompressingStoredFieldsReader.serializedDocument`:
    /// one document's *serialized* stored-field bytes -- the exact bytes the
    /// writer buffered for it, decompressed but not parsed into
    /// [`StoredField`]s -- plus its stored-field count. This is what
    /// [`StoredFieldsReader::document`] parses.
    ///
    /// A **random-access** read: only the sub-blocks this one document
    /// intersects are inflated. Reading a run of documents costs less through
    /// [`ChunkCursor`], which keeps the whole decompressed chunk.
    pub fn serialized_document(&self, doc_id: i32) -> Result<SerializedDocument> {
        let (header, mut input) = self.read_chunk_header(doc_id)?;
        let index = header.index_of(doc_id);
        // ARITH: `read_chunk_header` built `offsets` with `chunk_docs + 1`
        // non-decreasing non-negative entries and `index_of` returns
        // `doc_id - doc_base < chunk_docs`, so `index + 1` is in range and
        // `offsets[index + 1] >= offsets[index]`.
        #[allow(clippy::arithmetic_side_effects)]
        let (doc_offset, doc_length) = {
            let start = header.offsets[index] as usize;
            (start, header.offsets[index + 1] as usize - start)
        };
        let num_stored_fields = header.num_stored_fields[index];
        if doc_length == 0 {
            return Ok(SerializedDocument {
                num_stored_fields,
                bytes: Vec::new(),
            });
        }
        // ARITH: `doc_offset + doc_length` is `offsets[index + 1]`, already
        // read above and bounded by the chunk's total length.
        #[allow(clippy::arithmetic_side_effects)]
        let want_end = doc_offset + doc_length;
        let bytes = self.decompress_range(&header, &mut input, doc_offset, want_end)?;
        Ok(SerializedDocument {
            num_stored_fields,
            bytes,
        })
    }

    /// Decompresses the **whole** chunk containing `doc_id` -- the port of
    /// Java's `BlockState`, which its reader keeps cached precisely because a
    /// merge (or any other sequential scan) reads every document of a chunk
    /// in turn. Reading them one at a time through
    /// [`Self::serialized_document`] re-inflates the sub-blocks each of them
    /// intersects, so a 1024-document chunk pays for its ten sub-blocks about
    /// a hundred times each.
    ///
    /// Use [`ChunkCursor`] rather than calling this directly: it holds on to
    /// the chunk and only reloads when a document falls outside it.
    pub fn read_chunk(&self, doc_id: i32) -> Result<DecompressedChunk> {
        let (header, mut input) = self.read_chunk_header(doc_id)?;
        let total = header.total_length() as usize;
        let bytes = if total == 0 {
            Vec::new()
        } else {
            self.decompress_range(&header, &mut input, 0, total)?
        };
        Ok(DecompressedChunk { header, bytes })
    }

    /// Which compressor this segment's `.fdt` was written with -- a
    /// bulk chunk copy is only legal into a writer using the same one
    /// (`Lucene90CompressingStoredFieldsWriter.getMergeStrategy`'s
    /// `reader.getCompressionMode() == compressionMode`).
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// `getChunkSize()` -- the writer's plaintext-bytes-per-chunk trigger,
    /// recorded in `.fdm`. A `sliced` chunk is split back into compression
    /// units by this value, so a bulk chunk copy is only legal between two
    /// segments that agree on it.
    pub fn chunk_size(&self) -> i32 {
        self.chunk_size
    }

    /// `getNumChunks()`.
    pub fn num_chunks(&self) -> i64 {
        self.num_chunks
    }

    /// `getNumDirtyChunks()` -- chunks the writer had to force-flush before
    /// they reached `chunkSize`/`maxDocsPerChunk`.
    pub fn num_dirty_chunks(&self) -> i64 {
        self.num_dirty_chunks
    }

    /// `getNumDirtyDocs()` -- the documents living in those dirty chunks.
    pub fn num_dirty_docs(&self) -> i64 {
        self.num_dirty_docs
    }

    /// `getMaxPointer()` -- the `.fdt` offset one past the last chunk's
    /// bytes, i.e. exactly where the footer starts.
    pub fn max_pointer(&self) -> i64 {
        self.max_pointer
    }

    /// Port of `StoredFieldsReader.checkIntegrity()` --
    /// `CodecUtil.checksumEntireFile(fieldsStream)`: recomputes the `.fdt`'s
    /// CRC over every byte before the footer and compares it with the stored
    /// one.
    ///
    /// `open` deliberately only calls `retrieve_checksum`, which validates the
    /// footer's *shape*, because a random-access reader should not CRC a whole
    /// file it will read a few kilobytes of. A **merge** is different, and
    /// Java runs this on every source reader before it picks a merge strategy:
    /// the bulk path copies a source's compressed bytes verbatim and then
    /// writes a freshly computed, valid footer over them, which would launder
    /// a bit flip in the source into a merged segment that passes every
    /// checksum from then on. ("bulk merge is scary: its caused corruption
    /// bugs in the past.")
    pub fn check_integrity(&self) -> Result<()> {
        codec_util::check_whole_file_footer(self.fdt, self.max_pointer as usize)?;
        Ok(())
    }

    /// The raw `.fdt` bytes this reader was opened over, for the bulk-copy
    /// merge path (`getFieldsStream()`).
    pub fn fdt(&self) -> &'d [u8] {
        self.fdt
    }

    /// The `.fdx` chunk index entry covering `doc_id`: the first document
    /// id of that chunk and the chunk's `.fdt` start offset. `Lucene90`'s
    /// `FieldsIndexReader.getStartPointer(docID)` is the second half; the
    /// first half is what the bulk-copy path needs in order to tell whether
    /// a document sits exactly on a chunk boundary.
    pub fn chunk_for_doc(&self, doc_id: i32) -> Result<ChunkIndexEntry> {
        if doc_id < 0 || doc_id >= self.max_doc {
            return Err(Error::DocOutOfRange(doc_id, self.max_doc));
        }
        let block_index = direct_monotonic::floor_index(
            self.docs_region()?,
            &self.docs_meta,
            0,
            self.num_chunks,
            doc_id as i64,
        )?;
        let doc_base = direct_monotonic::get(self.docs_region()?, &self.docs_meta, block_index)?;
        let start_pointer = self.block_start_pointer(block_index)?;
        Ok(ChunkIndexEntry {
            doc_base: doc_base as i32,
            start_pointer,
        })
    }
}

/// What a [`StoredFieldVisitor`] wants done with the field it was just asked
/// about -- port of `StoredFieldVisitor.Status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisitStatus {
    /// `Status.YES`: decode the value and call the matching `*_field` method.
    Yes,
    /// `Status.NO`: skip the value's bytes without decoding it.
    No,
    /// `Status.STOP`: stop reading this document entirely; every remaining
    /// field is left undecoded.
    Stop,
}

/// Port of `org.apache.lucene.index.StoredFieldVisitor`: the pull-free half
/// of stored-field retrieval, where the reader hands the caller one field at
/// a time and the caller decides -- *before the value is decoded* -- whether
/// it wants it, does not want it, or is done.
///
/// [`StoredFieldsReader::document`] materialises every field of a document
/// into an owned [`Document`], so retrieving one field of a wide document
/// costs a `String`/`Vec` for every other field too. Java never pays that:
/// `Lucene90CompressingStoredFieldsReader.document(docID, visitor)` asks
/// `needsField` first and calls `skipField` (which only advances the cursor)
/// for a `NO`, so an unwanted field costs its length vint and nothing else.
///
/// Every value method has a no-op default, exactly as Java's abstract class
/// does -- a visitor that only wants strings overrides `string_field` and
/// leaves the rest. `needs_field` has no default, because a visitor that does
/// not answer it has not said what it is for.
///
/// **Fields are identified by number, not by `FieldInfo`.** This reader
/// decodes `.fdt` alone and has never been handed a `FieldInfos` (see
/// [`open`]'s signature); the field *number* is what the wire format
/// actually carries, and a caller that wants names already has the `.fnm`
/// mapping that [`crate::field_infos`] produced.
pub trait StoredFieldVisitor {
    /// `StoredFieldVisitor.needsField(FieldInfo)`.
    fn needs_field(&mut self, field_number: i32) -> Result<VisitStatus>;

    /// `StoredFieldVisitor.stringField(FieldInfo, String)`. Borrowed, not
    /// owned: a visitor that only measures or matches the value never
    /// allocates it.
    fn string_field(&mut self, _field_number: i32, _value: &str) -> Result<()> {
        Ok(())
    }

    /// `StoredFieldVisitor.binaryField(FieldInfo, byte[])`.
    fn binary_field(&mut self, _field_number: i32, _value: &[u8]) -> Result<()> {
        Ok(())
    }

    /// `StoredFieldVisitor.intField(FieldInfo, int)`.
    fn int_field(&mut self, _field_number: i32, _value: i32) -> Result<()> {
        Ok(())
    }

    /// `StoredFieldVisitor.longField(FieldInfo, long)`.
    fn long_field(&mut self, _field_number: i32, _value: i64) -> Result<()> {
        Ok(())
    }

    /// `StoredFieldVisitor.floatField(FieldInfo, float)`.
    fn float_field(&mut self, _field_number: i32, _value: f32) -> Result<()> {
        Ok(())
    }

    /// `StoredFieldVisitor.doubleField(FieldInfo, double)`.
    fn double_field(&mut self, _field_number: i32, _value: f64) -> Result<()> {
        Ok(())
    }
}

/// Port of `DocumentStoredFieldVisitor`: collects the fields it is asked for
/// into a [`Document`], skipping every other field's bytes.
///
/// [`DocumentVisitor::all`] accepts everything and is exactly what
/// [`parse_document`] runs; [`DocumentVisitor::for_fields`] accepts only the
/// listed field numbers, which is the one-field-of-a-wide-document case.
#[derive(Debug, Clone, Default)]
pub struct DocumentVisitor {
    /// `None` == accept every field (Java's no-argument constructor).
    wanted: Option<Vec<i32>>,
    doc: Document,
}

impl DocumentVisitor {
    /// `new DocumentStoredFieldVisitor()`: takes every field.
    pub fn all() -> Self {
        Self {
            wanted: None,
            doc: Document::default(),
        }
    }

    /// `new DocumentStoredFieldVisitor(Set<String> fieldsToLoad)`, by field
    /// number: takes only these fields and skips the rest.
    pub fn for_fields(field_numbers: &[i32]) -> Self {
        Self {
            wanted: Some(field_numbers.to_vec()),
            doc: Document::default(),
        }
    }

    /// `DocumentStoredFieldVisitor.getDocument()`.
    pub fn into_document(self) -> Document {
        self.doc
    }

    /// The fields collected so far, without consuming the visitor.
    pub fn document(&self) -> &Document {
        &self.doc
    }

    fn push(&mut self, field_number: i32, value: FieldValue) {
        self.doc.fields.push(StoredField {
            field_number,
            value,
        });
    }
}

impl StoredFieldVisitor for DocumentVisitor {
    fn needs_field(&mut self, field_number: i32) -> Result<VisitStatus> {
        // Java's `DocumentStoredFieldVisitor` answers NO, never STOP, even
        // once it has everything it asked for: the fields of a document are
        // in no particular order, so a wanted field can still follow an
        // unwanted one.
        Ok(match &self.wanted {
            None => VisitStatus::Yes,
            Some(w) if w.contains(&field_number) => VisitStatus::Yes,
            Some(_) => VisitStatus::No,
        })
    }

    fn string_field(&mut self, field_number: i32, value: &str) -> Result<()> {
        self.push(field_number, FieldValue::String(value.to_string()));
        Ok(())
    }

    fn binary_field(&mut self, field_number: i32, value: &[u8]) -> Result<()> {
        self.push(field_number, FieldValue::Binary(value.to_vec()));
        Ok(())
    }

    fn int_field(&mut self, field_number: i32, value: i32) -> Result<()> {
        self.push(field_number, FieldValue::Int(value));
        Ok(())
    }

    fn long_field(&mut self, field_number: i32, value: i64) -> Result<()> {
        self.push(field_number, FieldValue::Long(value));
        Ok(())
    }

    fn float_field(&mut self, field_number: i32, value: f32) -> Result<()> {
        self.push(field_number, FieldValue::Float(value));
        Ok(())
    }

    fn double_field(&mut self, field_number: i32, value: f64) -> Result<()> {
        self.push(field_number, FieldValue::Double(value));
        Ok(())
    }
}

/// Walks one document's serialized stored-field bytes, asking `visitor` about
/// each field before decoding it -- the exact loop of
/// `Lucene90CompressingStoredFieldsReader.document(int, StoredFieldVisitor)`,
/// including its "don't `skipField` on the last field value; treat like STOP"
/// shortcut.
///
/// This is the one decode path: [`parse_document`] is this function run with
/// a [`DocumentVisitor::all`].
pub fn visit_document(
    num_stored_fields: i64,
    bytes: &[u8],
    visitor: &mut dyn StoredFieldVisitor,
) -> Result<()> {
    let mut input = SliceInput::new(bytes);
    for field_idx in 0..num_stored_fields {
        let info_and_bits = input.read_vlong()?;
        // Java is `(int) (infoAndBits >>> TYPE_BITS)`. The unsigned shift is
        // written out here rather than left as a signed `>>`: the two agree
        // for every input (they differ only in the top three bits, which the
        // `as i32` discards), but this is the one place in the file where
        // that is true, and the reader should not have to re-derive it.
        let field_number = ((info_and_bits as u64) >> TYPE_BITS) as i32;
        let bits = info_and_bits & TYPE_MASK;
        match visitor.needs_field(field_number)? {
            VisitStatus::Yes => visit_field(&mut input, field_number, bits, visitor)?,
            VisitStatus::No => {
                // ARITH: `field_idx` runs `0..num_stored_fields`, so
                // `num_stored_fields >= 1` here and the decrement cannot
                // underflow.
                #[allow(clippy::arithmetic_side_effects)]
                let is_last = field_idx == num_stored_fields - 1;
                if is_last {
                    return Ok(());
                }
                skip_field(&mut input, bits)?;
            }
            VisitStatus::Stop => return Ok(()),
        }
    }
    Ok(())
}

/// Parses one document's serialized stored-field bytes (a
/// `(fieldNumber << 3) | type` vlong plus the value, repeated
/// `num_stored_fields` times) into [`StoredField`]s -- the half of
/// [`StoredFieldsReader::document`] that is not I/O, split out so a
/// sequential scan can parse straight out of a [`ChunkCursor`]'s
/// already-decompressed chunk.
///
/// Equivalent to [`visit_document`] with a [`DocumentVisitor::all`], and
/// implemented as exactly that so there is one field-decode loop and not two.
/// A caller that wants a subset should say so -- [`DocumentVisitor::for_fields`]
/// -- rather than take the whole document and drop most of it.
pub fn parse_document(num_stored_fields: i64, bytes: &[u8]) -> Result<Document> {
    let mut visitor = DocumentVisitor::all();
    // `num_stored_fields` comes off the chunk header's bulk int array, whose
    // widest shape is 32-bit read unsigned -- so it reaches ~4.3e9, and a
    // `StoredField` is 40-odd bytes. Reserving for the *claim* is the
    // abort-not-unwind shape `docs/arithmetic-gate.md` names; reserving for
    // what the bytes could hold is free, because every field costs at least
    // its `infoAndBits` vlong byte and the loop below hits EOF at exactly the
    // same point either way.
    visitor
        .doc
        .fields
        .reserve((num_stored_fields.max(0) as u64).min(bytes.len() as u64) as usize);
    visit_document(num_stored_fields, bytes, &mut visitor)?;
    Ok(visitor.into_document())
}

/// One document's serialized stored-field bytes -- see
/// [`StoredFieldsReader::serialized_document`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SerializedDocument {
    /// How many stored fields `bytes` encodes.
    pub num_stored_fields: i64,
    /// The document's own byte range of its chunk's decompressed payload.
    pub bytes: Vec<u8>,
}

/// The `.fdx` entry for the chunk containing some document -- see
/// [`StoredFieldsReader::chunk_for_doc`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkIndexEntry {
    /// The chunk's first document id (its `docBase`).
    pub doc_base: i32,
    /// The chunk's start offset in `.fdt`.
    pub start_pointer: i64,
}

/// One chunk's decoded header: everything before its compressed payload.
#[derive(Debug, Clone)]
struct ChunkHeader {
    doc_base: i32,
    chunk_docs: i32,
    sliced: bool,
    num_stored_fields: Vec<i64>,
    /// `chunk_docs + 1` cumulative payload offsets; the last is the chunk's
    /// total decompressed length.
    offsets: Vec<i64>,
}

impl ChunkHeader {
    fn total_length(&self) -> i64 {
        *self.offsets.last().unwrap_or(&0)
    }

    // ARITH: a `ChunkHeader` is only ever constructed by
    // `read_chunk_header`, which rejects the chunk unless
    // `0 <= doc_base <= max_doc - chunk_docs` and `0 < chunk_docs <= 4096`.
    // So `doc_base + chunk_docs <= max_doc <= i32::MAX`, and `index_of`'s
    // subtraction is guarded by the `contains` its two callers check first
    // (`DecompressedChunk::document`) or by `read_chunk_header`'s own
    // `doc_id >= doc_base` (`serialized_document`).
    #[allow(clippy::arithmetic_side_effects)]
    fn contains(&self, doc_id: i32) -> bool {
        doc_id >= self.doc_base && doc_id < self.doc_base + self.chunk_docs
    }

    // ARITH: same invariant as `contains` immediately above -- a
    // `ChunkHeader` only exists once `read_chunk_header` has established
    // `0 <= doc_base <= doc_id`, so the difference is in `0..=i32::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    fn index_of(&self, doc_id: i32) -> usize {
        debug_assert!(self.contains(doc_id));
        (doc_id - self.doc_base) as usize
    }
}

/// A whole chunk, decompressed once -- see
/// [`StoredFieldsReader::read_chunk`].
#[derive(Debug, Clone)]
pub struct DecompressedChunk {
    header: ChunkHeader,
    bytes: Vec<u8>,
}

impl DecompressedChunk {
    /// The chunk's first document id.
    pub fn doc_base(&self) -> i32 {
        self.header.doc_base
    }

    /// How many documents this chunk holds.
    pub fn num_docs(&self) -> i32 {
        self.header.chunk_docs
    }

    /// Whether `doc_id` is one of them.
    pub fn contains(&self, doc_id: i32) -> bool {
        self.header.contains(doc_id)
    }

    /// One of this chunk's documents: its stored-field count and its
    /// serialized bytes, borrowed straight out of the decompressed chunk --
    /// no per-document allocation, no per-document decompression.
    pub fn document(&self, doc_id: i32) -> Option<(i64, &[u8])> {
        if !self.contains(doc_id) {
            return None;
        }
        let i = self.header.index_of(doc_id);
        // ARITH: `contains` above gives `i < chunk_docs`, and `offsets` holds
        // `chunk_docs + 1` entries (`read_chunk_header`).
        #[allow(clippy::arithmetic_side_effects)]
        let end = self.header.offsets[i + 1] as usize;
        let start = self.header.offsets[i] as usize;
        Some((self.header.num_stored_fields[i], &self.bytes[start..end]))
    }
}

/// Java's cached `BlockState`: the decompressed chunk the last read came
/// from, reused for every further document that falls inside it.
///
/// This is what makes a sequential scan -- a merge's DOC path, a
/// `CheckIndex`-style pass -- cost one chunk decompression per chunk instead
/// of one per document.
#[derive(Debug, Default)]
pub struct ChunkCursor {
    chunk: Option<DecompressedChunk>,
}

impl ChunkCursor {
    /// A cursor holding no chunk yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// `doc_id`'s serialized stored-field bytes and field count, loading (and
    /// keeping) its chunk only if the currently held one does not contain it.
    pub fn document(
        &mut self,
        reader: &StoredFieldsReader<'_>,
        doc_id: i32,
    ) -> Result<(i64, &[u8])> {
        if !self.chunk.as_ref().is_some_and(|c| c.contains(doc_id)) {
            self.chunk = Some(reader.read_chunk(doc_id)?);
        }
        Ok(self
            .chunk
            .as_ref()
            .expect("just loaded")
            .document(doc_id)
            .expect("read_chunk(doc_id) returns the chunk containing doc_id"))
    }

    /// Drops the held chunk -- for a caller that is done with one source and
    /// does not want to keep its bytes resident.
    pub fn reset(&mut self) {
        self.chunk = None;
    }
}

/// Port of `Lucene90CompressingStoredFieldsWriter` for `Mode.BEST_SPEED` --
/// the write-side counterpart of [`open`]/[`StoredFieldsReader::document`].
/// Documents are buffered into chunks exactly as Java does (close on the
/// first doc that takes the chunk to [`BEST_SPEED_CHUNK_SIZE`] bytes, or on
/// the [`BEST_SPEED_MAX_DOCS_PER_CHUNK`]th doc, whichever comes first; a
/// chunk that runs out of documents before either trigger is a *dirty*
/// chunk, flagged in its token and tallied in `.fdm`).
///
/// Each chunk's payload is framed as one `LZ4WithPresetDictCompressionMode`
/// unit, with the same dictionary/sub-block geometry Java's compressor
/// picks -- `dictLength = min(64kB, len / (NUM_SUB_BLOCKS * DICT_SIZE_FACTOR))`
/// and `blockLength = ceil((len - dictLength) / NUM_SUB_BLOCKS)` -- so the
/// dictionary really is a preset dictionary the sub-blocks back-reference
/// into ([`crate::lz4::compress_with_dictionary`]), and a reader that only
/// wants one document can skip the sub-blocks it does not intersect.
/// A single [`crate::lz4::FastCompressionHashTable`] is allocated per call
/// and reused for every unit, matching `LZ4WithPresetDictCompressor`'s own
/// per-compressor table (the table is 16kB and never cleared -- allocating
/// one per chunk would dominate the cost of compressing a segment).
pub fn write_best_speed(
    docs: &[Document],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut writer = StoredFieldsWriter::new(Mode::BestSpeed, segment_id, segment_suffix);
    for doc in docs {
        writer.add_document(doc);
    }
    writer.finish()
}

/// Lucene's per-mode chunking parameters, from the two
/// `Lucene90CompressingStoredFieldsFormat` constructions in
/// `Lucene90StoredFieldsFormat.fieldsFormat`: a chunk closes as soon as its
/// buffered bytes reach the chunk size *or* its doc count reaches the
/// per-chunk doc cap, whichever comes first.
const BEST_SPEED_CHUNK_SIZE: usize = 10 * 8 * 1024;
const BEST_SPEED_MAX_DOCS_PER_CHUNK: usize = 1024;
const BEST_COMPRESSION_CHUNK_SIZE: usize = 10 * 48 * 1024;
const BEST_COMPRESSION_MAX_DOCS_PER_CHUNK: usize = 4096;
/// `blockShift` for the `.fdx` monotonic arrays -- the `10` both modes pass.
const INDEX_BLOCK_SHIFT: u32 = 10;
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
struct UnitScratch {
    ht: lz4::FastCompressionHashTable,
    buffer: Vec<u8>,
    compressed: Vec<u8>,
    lengths: Vec<usize>,
}

/// Appends one self-contained compression unit for `payload` -- the only
/// part of the chunk format that differs between [`Mode::BestSpeed`] and
/// [`Mode::BestCompression`]. Both emit a dictionary prefix plus
/// [`NUM_SUB_BLOCKS`] sub-blocks, with different geometry and a different
/// compressor; note the vint framing differs too (LZ4 batches every
/// sub-block's compressed length up front, DEFLATE interleaves each one
/// immediately before its bytes -- see [`decompress_unit`]).
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
fn write_unit(mode: Mode, scratch: &mut UnitScratch, out: &mut Vec<u8>, payload: &[u8]) {
    let len = payload.len();
    match mode {
        Mode::BestSpeed => {
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
            lz4::compress_with_dictionary(
                &scratch.buffer,
                0,
                0,
                dict_length,
                &mut scratch.compressed,
                &mut scratch.ht,
            );
            scratch.lengths.push(scratch.compressed.len() - mark);

            // ...then each sub-block, back-referencing into it.
            let mut start = dict_length;
            while start < len {
                let l = block_length.min(len - start);
                scratch.buffer[dict_length..dict_length + l]
                    .copy_from_slice(&payload[start..start + l]);
                let mark = scratch.compressed.len();
                lz4::compress_with_dictionary(
                    &scratch.buffer,
                    0,
                    dict_length,
                    l,
                    &mut scratch.compressed,
                    &mut scratch.ht,
                );
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
        Mode::BestCompression => {
            let dict_length = len / (NUM_SUB_BLOCKS * DEFLATE_DICT_SIZE_FACTOR);
            let block_length = (len - dict_length).div_ceil(NUM_SUB_BLOCKS);

            out.write_vint(dict_length as i32);
            out.write_vint(block_length as i32);

            write_deflate_unit(out, &payload[..dict_length]);

            let mut start = dict_length;
            while start < len {
                let this_block = block_length.min(len - start);
                write_deflate_unit(out, &payload[start..start + this_block]);
                start += this_block;
            }
        }
    }
}

/// A streaming port of `Lucene90CompressingStoredFieldsWriter`'s
/// buffer/`triggerFlush`/`flush` cycle, plus its three merge paths.
///
/// Real Lucene's writer is inherently streaming -- documents arrive one at a
/// time and are flushed a chunk at a time -- and its merge entry point
/// exploits that to avoid touching document *content* at all wherever it
/// can. Two of the three paths it picks per source segment
/// (`getMergeStrategy`) only exist if the writer can be fed something other
/// than a parsed [`Document`]:
///
/// - **BULK** ([`Self::copy_chunks`]) -- the source's already-compressed
///   chunk bytes are copied verbatim; only each chunk's `docBase` vint and
///   the `.fdx` index entry are rewritten. No decompression, no
///   recompression, no per-field work at all.
/// - **DOC** ([`Self::add_serialized_document`]) -- the source document's
///   *serialized* bytes (from
///   [`StoredFieldsReader::serialized_document`]) are appended to the
///   current chunk buffer without ever being parsed into fields.
/// - **VISITOR** ([`Self::add_document`]) -- the fallback, and the only path
///   this port had before: parse every field, then recompress.
///
/// See `merge.rs`'s `stored_fields_merge_strategy` for the conditions that
/// pick between them; the two fast paths copy bytes that embed source field
/// *numbers*, so they are only legal when the source's field numbering
/// survives the merge unchanged (`MatchingReaders`).
pub struct StoredFieldsWriter {
    mode: Mode,
    chunk_size: usize,
    max_docs_per_chunk: usize,
    segment_id: [u8; ID_LENGTH],
    segment_suffix: String,
    fdt: Vec<u8>,
    /// Cumulative doc counts, one per chunk written so far, seeded with `0`
    /// -- `write_index_and_meta` appends nothing, so the last entry doubles
    /// as `.fdx`'s trailing `maxDoc` sentinel.
    docs_values: Vec<i64>,
    /// Chunk start offsets, seeded with the first chunk's (= the index
    /// header length); the last entry doubles as the `maxPointer` sentinel.
    start_pointers_values: Vec<i64>,
    num_dirty_chunks: i64,
    num_dirty_docs: i64,
    /// Java's `bufferedDocs`/`endOffsets`/`numStoredFields`: the *current
    /// chunk only*, reset per flush.
    chunk_buf: Vec<u8>,
    lengths: Vec<i64>,
    num_stored_fields: Vec<i64>,
    /// Java's `docBase`: the doc id at the start of the current chunk, which
    /// (since buffered docs are always the tail) is also the number of
    /// documents already flushed.
    doc_base: i32,
    scratch: UnitScratch,
}

impl StoredFieldsWriter {
    /// A fresh writer for `mode`'s chunk geometry, with the `.fdt` index
    /// header already written.
    pub fn new(mode: Mode, segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Self {
        let (data_codec, chunk_size, max_docs_per_chunk) = match mode {
            Mode::BestSpeed => (
                DATA_CODEC_BEST_SPEED,
                BEST_SPEED_CHUNK_SIZE,
                BEST_SPEED_MAX_DOCS_PER_CHUNK,
            ),
            Mode::BestCompression => (
                DATA_CODEC_BEST_COMPRESSION,
                BEST_COMPRESSION_CHUNK_SIZE,
                BEST_COMPRESSION_MAX_DOCS_PER_CHUNK,
            ),
        };
        let mut fdt = Vec::new();
        codec_util::write_index_header(
            &mut fdt,
            data_codec,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        );
        let start = fdt.len() as i64;
        Self {
            mode,
            chunk_size,
            max_docs_per_chunk,
            segment_id: *segment_id,
            segment_suffix: segment_suffix.to_string(),
            fdt,
            docs_values: vec![0],
            start_pointers_values: vec![start],
            num_dirty_chunks: 0,
            num_dirty_docs: 0,
            chunk_buf: Vec::new(),
            lengths: Vec::new(),
            num_stored_fields: Vec::new(),
            doc_base: 0,
            scratch: UnitScratch::default(),
        }
    }

    /// `getChunkSize()` -- a bulk chunk copy is only legal from a reader
    /// that agrees with this.
    pub fn chunk_size(&self) -> i32 {
        self.chunk_size as i32
    }

    /// Java's `tooDirty(candidate)`: a source segment is too dirty to
    /// bulk-copy when it has enough dirty documents to make a full chunk
    /// *and* more than 1% of its chunks are dirty. Copying such a segment
    /// verbatim would carry its degraded compression ratio forward forever,
    /// so Java recompresses it instead.
    pub fn too_dirty(&self, reader: &StoredFieldsReader<'_>) -> bool {
        // `saturating_mul` is defensive here rather than load-bearing, and
        // it is worth being exact about why: `numDirtyChunks` is a vlong off
        // the source's `.fdm`, but `open` has already rejected the segment
        // unless `numDirtyChunks <= numChunks`, and `numChunks` is pinned to
        // `indexNumChunks - 1` where `indexNumChunks` is a plain `readInt`.
        // So no reader that exists can carry a `numDirtyChunks` above 2^31,
        // and `* 100` cannot overflow. What saturating buys is that the
        // property does not have to hold for a *future* caller: if it ever
        // did overflow, `i64::MAX > numChunks` makes the segment "too dirty",
        // which refuses the bulk copy and falls back to recompressing --
        // correct for any input, where Java's silent `long` wrap to a
        // negative product would read as "clean" and copy it verbatim.
        reader.num_dirty_docs() > self.max_docs_per_chunk as i64
            && reader.num_dirty_chunks().saturating_mul(100) > reader.num_chunks()
    }

    /// Whether this writer could copy `reader`'s compressed chunks verbatim,
    /// ignoring the caller's own field-numbering and deletions checks:
    /// `getMergeStrategy`'s `compressionMode`/`chunkSize`/`tooDirty` trio.
    /// (The version check Java also makes is `open`'s job here -- this port
    /// has exactly one `VERSION_CURRENT` and refuses to open anything else.)
    pub fn can_bulk_copy(&self, reader: &StoredFieldsReader<'_>) -> bool {
        reader.mode() == self.mode
            && reader.chunk_size() == self.chunk_size as i32
            && !self.too_dirty(reader)
    }

    fn num_buffered_docs(&self) -> usize {
        self.lengths.len()
    }

    fn trigger_flush(&self) -> bool {
        self.chunk_buf.len() >= self.chunk_size
            || self.num_buffered_docs() >= self.max_docs_per_chunk
    }

    /// Java's `finishDocument()`: record the document just appended to
    /// `chunk_buf`, then close the chunk if either trigger fired.
    fn finish_document(&mut self, num_stored_fields: i64, length: i64) {
        self.num_stored_fields.push(num_stored_fields);
        self.lengths.push(length);
        if self.trigger_flush() {
            self.flush(false);
        }
    }

    /// Java's `flush(force)`. `force` is true only for the final flush in
    /// `finish()` and for the pre-bulk-copy flush in `copyChunks`, and that
    /// is exactly what makes a chunk **dirty**: it closed before either
    /// trigger fired, so it is flagged in its token and tallied in `.fdm`.
    fn flush(&mut self, force: bool) {
        debug_assert_ne!(self.trigger_flush(), force);
        let chunk_docs = self.num_buffered_docs();
        // ARITH: both counters are this writer's own, and each `flush` costs
        // at least the four `.fdt` bytes of a chunk header, so they are
        // bounded by `fdt.len() / 4` -- a `Vec` length, hence `isize::MAX`.
        // `chunk_docs` is `lengths.len()`, capped at `max_docs_per_chunk`
        // (4096) by `trigger_flush`. `2 * self.chunk_size` is twice a
        // per-mode constant (81 920 or 491 520).
        #[allow(clippy::arithmetic_side_effects)]
        if force {
            self.num_dirty_chunks += 1;
            self.num_dirty_docs += chunk_docs as i64;
        }
        // Only a chunk holding a single outsized document can reach this,
        // since every other chunk closes on the first doc that crosses
        // `chunk_size`.
        //
        // ARITH: `chunk_size` is a per-mode constant (81 920 or 491 520)
        // chosen in `StoredFieldsWriter::new` and never assigned again, so
        // doubling it is nowhere near `usize`.
        #[allow(clippy::arithmetic_side_effects)]
        let sliced = self.chunk_buf.len() >= 2 * self.chunk_size;

        self.fdt.write_vint(self.doc_base);
        self.fdt
            .write_vint(((chunk_docs as i32) << 2) | if force { 2 } else { 0 } | i32::from(sliced));

        if chunk_docs == 1 {
            self.fdt.write_vint(self.num_stored_fields[0] as i32);
            self.fdt.write_vint(self.lengths[0] as i32);
        } else {
            write_bulk_ints(&mut self.fdt, &self.num_stored_fields);
            write_bulk_ints(&mut self.fdt, &self.lengths);
        }

        if !self.chunk_buf.is_empty() {
            // `chunk_buf` and `fdt` are distinct fields, but the borrow
            // checker cannot see that through `&mut self`; swap the buffer
            // out for the duration of the compression and put it back.
            let payload = std::mem::take(&mut self.chunk_buf);
            if sliced {
                for slice in payload.chunks(self.chunk_size) {
                    write_unit(self.mode, &mut self.scratch, &mut self.fdt, slice);
                }
            } else {
                write_unit(self.mode, &mut self.scratch, &mut self.fdt, &payload);
            }
            self.chunk_buf = payload;
        }

        // Java's `docBase += numBufferedDocs` on an `int`, which wraps
        // silently. A wrapped `doc_base` would put a negative `docBase` vint
        // in the `.fdt` and a non-monotonic value in the `.fdx` doc array --
        // an unopenable segment written without a word of complaint. It takes
        // more than `IndexWriter.MAX_DOCS` (2^31 - 128) documents in one
        // segment to get there, which is a caller bug rather than bad input,
        // so it panics with a name rather than returning a `Result` these
        // infallible `add_*` entry points do not have.
        self.doc_base = self
            .doc_base
            .checked_add(chunk_docs as i32)
            .expect("stored fields writer overflowed i32 document ids");
        self.chunk_buf.clear();
        self.lengths.clear();
        self.num_stored_fields.clear();
        self.record_chunk();
    }

    /// One `.fdx` entry: the cumulative doc count and the offset one past
    /// the chunk just written (which is the *next* chunk's start pointer,
    /// and after the last chunk is `maxPointer`).
    fn record_chunk(&mut self) {
        self.docs_values.push(self.doc_base as i64);
        self.start_pointers_values.push(self.fdt.len() as i64);
    }

    /// VISITOR path: serialize and buffer one parsed document.
    pub fn add_document(&mut self, doc: &Document) {
        let doc_start = self.chunk_buf.len();
        serialize_doc_into(doc, &mut self.chunk_buf);
        // ARITH: `serialize_doc_into` only appends, so the buffer is at least
        // `doc_start` bytes long afterwards.
        #[allow(clippy::arithmetic_side_effects)]
        let length = (self.chunk_buf.len() - doc_start) as i64;
        self.finish_document(doc.fields.len() as i64, length);
    }

    /// DOC path (`copyOneDoc`): buffer one document's already-serialized
    /// bytes. `bytes` must be exactly what
    /// [`StoredFieldsReader::serialized_document`]/[`ChunkCursor::document`]
    /// returned for a segment whose field numbers survive into this one
    /// unchanged -- the encoding embeds field numbers, and nothing here
    /// re-checks them.
    ///
    /// Takes a borrowed slice rather than an owned document so a caller
    /// reading a run of documents through a [`ChunkCursor`] never allocates
    /// per document.
    pub fn add_serialized_document(&mut self, num_stored_fields: i64, bytes: &[u8]) {
        self.chunk_buf.extend_from_slice(bytes);
        self.finish_document(num_stored_fields, bytes.len() as i64);
    }

    /// BULK path (`copyChunks`): copy `reader`'s documents `from_doc`
    /// (inclusive) to `to_doc` (exclusive) into this writer, copying whole
    /// *compressed* chunks verbatim wherever a chunk lies entirely inside
    /// that range.
    ///
    /// # Preconditions
    ///
    /// The caller owns every condition that makes this safe, exactly as
    /// `getMergeStrategy` does in Java:
    /// - [`Self::can_bulk_copy`] holds for `reader` (same compressor, same
    ///   `chunkSize`, not too dirty);
    /// - `reader`'s segment has **no deletions**, and `from_doc..to_doc` is
    ///   a run of consecutive source doc ids that map to consecutive merged
    ///   ids starting at this writer's current document count;
    /// - `reader`'s field numbers are unchanged by the merge
    ///   (`MatchingReaders`), since the copied bytes encode them.
    ///
    /// Only whole chunks are copied. A leading partial chunk (when
    /// `from_doc` is not a chunk boundary) and a trailing partial chunk
    /// (when `to_doc` is not) are copied document at a time through the DOC
    /// path instead -- this is what Java's `isLoaded(docID)` loops do, but
    /// stated as the chunk-boundary condition they are really testing rather
    /// than as a property of the reader's cached block.
    ///
    /// This allocates a fresh [`ChunkCursor`] for those partial chunks, so
    /// every call decompresses its own. That is right for a caller making one
    /// large call per source; a caller that makes **many short calls** on the
    /// same reader -- which is what an index-sorted merge does, because the
    /// sources interleave and every run is a document or two -- must use
    /// [`Self::copy_chunks_with_cursor`] and keep one cursor per reader.
    /// Java gets that for free because `copyOneDoc` reads through the
    /// reader's own cached `BlockState`; here the cache is the caller's.
    pub fn copy_chunks(
        &mut self,
        reader: &StoredFieldsReader<'_>,
        from_doc: i32,
        to_doc: i32,
    ) -> Result<()> {
        let mut cursor = ChunkCursor::new();
        self.copy_chunks_with_cursor(reader, &mut cursor, from_doc, to_doc)
    }

    /// [`Self::copy_chunks`] with a caller-owned [`ChunkCursor`] for the
    /// partial chunks at each end of the run -- the equivalent of Java's
    /// per-reader cached `BlockState`.
    ///
    /// With one cursor per source segment, a run of documents that all fall
    /// in one chunk costs one decompression however many separate calls it
    /// is spread across. Without it, an index-sorted merge decompresses a
    /// whole chunk per document (measured: 2 004 ms versus 13.2 ms for
    /// 80 000 documents across 4 segments).
    pub fn copy_chunks_with_cursor(
        &mut self,
        reader: &StoredFieldsReader<'_>,
        cursor: &mut ChunkCursor,
        from_doc: i32,
        to_doc: i32,
    ) -> Result<()> {
        // A real check, not a `debug_assert`: this is a `pub` method on a
        // library type, and a release-mode caller copying chunks from a
        // reader with a different `chunkSize` would get a segment whose
        // `sliced` chunks are re-split at the wrong boundary -- silently wrong
        // documents, the exact failure class this whole path guards against.
        // (Java can afford an `assert` here because `copyChunks` is private
        // and `merge` is its only caller.)
        if !self.can_bulk_copy(reader) {
            return Err(Error::BulkCopyNotPermitted {
                reader_chunk_size: reader.chunk_size(),
                writer_chunk_size: self.chunk_size as i32,
                same_mode: reader.mode() == self.mode,
                too_dirty: self.too_dirty(reader),
            });
        }
        let max_doc = reader.max_doc();
        if from_doc < 0 || to_doc > max_doc {
            return Err(Error::DocOutOfRange(from_doc.max(to_doc), max_doc));
        }
        if from_doc > to_doc {
            return Err(Error::InvertedDocRange { from_doc, to_doc });
        }
        let mut doc = from_doc;

        // Documents belonging to a chunk that started before `from_doc`.
        while doc < to_doc && reader.chunk_for_doc(doc)?.doc_base != doc {
            let (num_stored_fields, bytes) = cursor.document(reader, doc)?;
            self.chunk_buf.extend_from_slice(bytes);
            self.finish_document(num_stored_fields, bytes.len() as i64);
            // ARITH: the loop guard is `doc < to_doc` and `to_doc <= max_doc
            // <= i32::MAX`, so `doc <= i32::MAX - 1` here.
            #[allow(clippy::arithmetic_side_effects)]
            {
                doc += 1;
            }
        }
        if doc >= to_doc {
            return Ok(());
        }

        let from_pointer = reader.chunk_for_doc(doc)?.start_pointer;
        let to_pointer = if to_doc == max_doc {
            reader.max_pointer()
        } else {
            reader.chunk_for_doc(to_doc)?.start_pointer
        };
        if from_pointer < to_pointer {
            if self.num_buffered_docs() > 0 {
                self.flush(true);
            }
            let mut pointer = from_pointer;
            let fdt = reader.fdt();
            loop {
                let mut input = SliceInput::new(fdt);
                input.seek(pointer as usize)?;
                let base = input.read_vint()?;
                let code = input.read_vint()?;
                // `>>>`, as Java's `code >>> 2` is: a signed shift turned a
                // corrupt token's sign bit into a *negative* chunk count,
                // which the `<= 0` check below does catch -- but only after
                // the two additions that follow it had already been written
                // as if it were a count.
                let chunk_docs = ((code as u32) >> 2) as i32;
                // Java's two `CorruptIndexException`s: the `.fdx` index and
                // the `.fdt` chunk headers are redundant, and disagreeing is
                // how a bad bulk copy announces itself before it writes a
                // segment that reads back plausible-but-wrong documents.
                if base != doc || chunk_docs <= 0 {
                    return Err(Error::CorruptChunkBounds {
                        doc_id: doc,
                        doc_base: base,
                        chunk_docs,
                        max_doc,
                    });
                }
                let body_start = input.position();
                self.fdt.write_vint(self.doc_base); // rebase
                self.fdt.write_vint(code);
                // Both counts come straight off the source `.fdt`'s token, so
                // both sums are `checked_`: `doc + chunk_docs` overflowing is
                // what the `doc > to_doc` check below is *for*, and reaching
                // it through a wrap would let a chunk claiming ~2^29
                // documents land back inside the requested range.
                let advanced = doc
                    .checked_add(chunk_docs)
                    .zip(self.doc_base.checked_add(chunk_docs));
                let Some((next_doc, next_doc_base)) = advanced else {
                    return Err(Error::CorruptChunkBounds {
                        doc_id: to_doc,
                        doc_base: base,
                        chunk_docs,
                        max_doc,
                    });
                };
                doc = next_doc;
                self.doc_base = next_doc_base;
                if doc > to_doc {
                    return Err(Error::CorruptChunkBounds {
                        doc_id: to_doc,
                        doc_base: base,
                        chunk_docs,
                        max_doc,
                    });
                }
                let end_pointer = if doc == max_doc {
                    reader.max_pointer()
                } else {
                    reader.chunk_for_doc(doc)?.start_pointer
                };
                let body = fdt
                    .get(body_start..end_pointer as usize)
                    .ok_or(lucene_store::Error::Eof { offset: body_start })?;
                self.fdt.write_bytes(body);
                // ARITH: `num_dirty_chunks` counts chunks written to this
                // writer's own `.fdt`, each at least four bytes, and
                // `chunk_docs` is now bounded by `to_doc <= max_doc`, so the
                // dirty-doc tally cannot pass `i32::MAX`.
                #[allow(clippy::arithmetic_side_effects)]
                if code & 2 != 0 {
                    self.num_dirty_chunks += 1;
                    self.num_dirty_docs += chunk_docs as i64;
                }
                self.record_chunk();
                pointer = end_pointer;
                if pointer >= to_pointer {
                    break;
                }
            }
        }

        // Trailing documents that do not form a complete chunk.
        //
        // ARITH: as above -- `doc < to_doc <= max_doc <= i32::MAX`.
        #[allow(clippy::arithmetic_side_effects)]
        while doc < to_doc {
            let (num_stored_fields, bytes) = cursor.document(reader, doc)?;
            self.chunk_buf.extend_from_slice(bytes);
            self.finish_document(num_stored_fields, bytes.len() as i64);
            doc += 1;
        }
        Ok(())
    }

    /// Java's `finish(numDocs)`: flush whatever is buffered as a final dirty
    /// chunk, then assemble `.fdt`/`.fdx`/`.fdm`.
    pub fn finish(mut self) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        if self.num_buffered_docs() > 0 {
            self.flush(true);
        }
        let max_doc = self.doc_base;
        let max_pointer = self.fdt.len() as i64;
        codec_util::write_footer(&mut self.fdt);
        let (fdx, fdm) = write_index_and_meta(
            &self.segment_id,
            &self.segment_suffix,
            max_doc,
            &self.docs_values,
            &self.start_pointers_values,
            max_pointer,
            self.chunk_size as i32,
            self.num_dirty_chunks,
            self.num_dirty_docs,
        );
        (self.fdt, fdx, fdm)
    }
}

/// Shared `.fdx`/`.fdm` assembly for both modes: `docs_values` and
/// `start_pointers_values` each hold one entry per chunk plus a trailing
/// sentinel (`maxDoc` and `maxPointer` respectively), so the chunk count both
/// files record is one less than their length. The two modes differ only in
/// the `chunkSize` written to `.fdm`, which real Lucene needs in order to
/// split a `sliced` chunk back into its compression units.
#[allow(clippy::too_many_arguments)]
fn write_index_and_meta(
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    max_doc: i32,
    docs_values: &[i64],
    start_pointers_values: &[i64],
    max_pointer: i64,
    chunk_size: i32,
    num_dirty_chunks: i64,
    num_dirty_docs: i64,
) -> (Vec<u8>, Vec<u8>) {
    debug_assert_eq!(docs_values.len(), start_pointers_values.len());
    let block_shift = INDEX_BLOCK_SHIFT;

    let mut fdx = Vec::new();
    codec_util::write_index_header(
        &mut fdx,
        INDEX_CODEC,
        INDEX_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    let docs_start_pointer = fdx.len() as i64;
    let (docs_meta_bytes, docs_data_bytes) = direct_monotonic::write(docs_values, block_shift);
    fdx.write_bytes(&docs_data_bytes);
    let docs_end_pointer = fdx.len() as i64;
    let (start_pointers_meta_bytes, start_pointers_data_bytes) =
        direct_monotonic::write(start_pointers_values, block_shift);
    fdx.write_bytes(&start_pointers_data_bytes);
    let start_pointers_end_pointer = fdx.len() as i64;
    codec_util::write_footer(&mut fdx);

    let mut fdm = Vec::new();
    codec_util::write_index_header(
        &mut fdm,
        META_CODEC,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    fdm.write_vint(chunk_size);
    fdm.write_i32(max_doc);
    fdm.write_i32(block_shift as i32);
    fdm.write_i32(docs_values.len() as i32); // real chunks + 1 sentinel
    fdm.write_i64(docs_start_pointer);
    fdm.write_bytes(&docs_meta_bytes);
    fdm.write_i64(docs_end_pointer);
    fdm.write_bytes(&start_pointers_meta_bytes);
    fdm.write_i64(start_pointers_end_pointer);
    fdm.write_i64(max_pointer);
    // ARITH: `docs_values` is seeded with one entry in `StoredFieldsWriter::new`
    // and only ever pushed to, so its length is at least 1.
    #[allow(clippy::arithmetic_side_effects)]
    fdm.write_vlong(docs_values.len() as i64 - 1); // numChunks (outer)
    fdm.write_vlong(num_dirty_chunks);
    fdm.write_vlong(num_dirty_docs);
    codec_util::write_footer(&mut fdm);

    (fdx, fdm)
}

/// Port of `Lucene90CompressingStoredFieldsWriter` for `Mode.BEST_COMPRESSION`
/// -- the DEFLATE counterpart of [`write_best_speed`]. Shares that function's
/// chunking and `.fdx`/`.fdm` assembly (both are [`StoredFieldsWriter`]'s,
/// via [`write_index_and_meta`]); only the payload framing differs, per
/// `DeflateWithPresetDictCompressionMode`'s own
/// `dictLength = len / (NUM_SUB_BLOCKS * DICT_SIZE_FACTOR)`,
/// `blockLength = ceil((len - dictLength) / NUM_SUB_BLOCKS)` formulas
/// ([`NUM_SUB_BLOCKS`], [`DEFLATE_DICT_SIZE_FACTOR`]) -- real Lucene reuses
/// this sizing regardless of a chunk's actual size, so this port does too
/// rather than inventing a different threshold. Note the vint framing is
/// *not* the same shape as LZ4's: DEFLATE interleaves each unit's compressed
/// length immediately before that unit's bytes, where LZ4 batches all of
/// them up front (see [`decompress_unit`]).
///
/// Each sub-block (and the dictionary prefix) is compressed independently
/// via [`deflate::compress`] with **no** preset-dictionary back-referencing
/// into the dictionary's plaintext: `miniz_oxide`'s `compress_to_vec`
/// has no preset-dictionary API (see `deflate.rs`'s module doc comment), so
/// each unit is a fully self-contained DEFLATE stream. This is still valid
/// per the wire format [`decompress_unit`] reads -- a sub-block's compressed
/// bytes decompress into `buffer[dict_length..]`, and the dictionary bytes
/// sitting in `buffer[..dict_length]` are available for a *decoder's*
/// back-references to reach into, but nothing requires the *encoder* to
/// have actually produced any cross-unit back-references. The cost is a
/// smaller compression ratio than real Lucene's writer (which does use the
/// dictionary to compress each block), not a correctness gap.
pub fn write_best_compression(
    docs: &[Document],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut writer = StoredFieldsWriter::new(Mode::BestCompression, segment_id, segment_suffix);
    for doc in docs {
        writer.add_document(doc);
    }
    writer.finish()
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
fn write_deflate_unit(out: &mut Vec<u8>, plain: &[u8]) {
    if plain.is_empty() {
        out.write_vint(0);
        return;
    }
    let compressed = deflate::compress(plain);
    out.write_vint(compressed.len() as i32);
    out.write_bytes(&compressed);
}

/// Port of `StoredFieldsInts`'s bulk per-doc array encode, the exact inverse
/// of [`read_bulk_ints`]: an all-equal constant shape, else a fixed 8/16/32-bit
/// width in which every whole 128-value block is bit-transposed across i64
/// words and only the remainder is written value-by-value.
fn write_bulk_ints(out: &mut Vec<u8>, values: &[i64]) {
    if values.iter().all(|&v| v == values[0]) {
        out.push(0);
        out.write_vint(values[0] as i32);
        return;
    }
    let max = *values.iter().max().unwrap();
    let bpv: u8 = if max <= 0xFF {
        8
    } else if max <= 0xFFFF {
        16
    } else {
        32
    };
    out.push(bpv);

    // Mirror of [`read_bulk_ints`]: every whole 128-value block is written in
    // Java's transposed layout (`StoredFieldsInts.writeInts8/16/32`), where
    // word `i`'s `lane`-th slot, MSB-first, holds value `i + lane*num_words`;
    // only the remainder past the last whole block is written value-by-value.
    const BLOCK_SIZE: usize = 128;
    let bpv_usize = bpv as usize;
    // ARITH: `bpv` is assigned 8, 16 or 32 a dozen lines up and nowhere else,
    // so both divisions are by a non-zero constant and `shift` peaks at
    // `(values_per_word - 1) * bpv == 64 - bpv == 56` -- 64 would be the
    // panicking shift, 56 is the real bound. `k` advances by 128 only while
    // `k + 128 <= values.len()`, and `values` is a chunk's per-document array
    // held in memory, so `k + 128` cannot leave `usize`.
    #[allow(clippy::arithmetic_side_effects)]
    let (values_per_word, num_words) = (64 / bpv_usize, BLOCK_SIZE / (64 / bpv_usize));

    let mut k = 0usize;
    // ARITH: same bounds as the `let` above -- `k` advances by 128 only while
    // `k + 128 <= values.len()`, `shift` peaks at 56, and every read index
    // `k + i + lane * num_words` is under `k + 128`.
    #[allow(clippy::arithmetic_side_effects)]
    while k + BLOCK_SIZE <= values.len() {
        for i in 0..num_words {
            let mut w = 0u64;
            for lane in 0..values_per_word {
                let shift = (values_per_word - 1 - lane) * bpv_usize;
                w |= (values[k + i + lane * num_words] as u64) << shift;
            }
            out.write_i64(w as i64);
        }
        k += BLOCK_SIZE;
    }
    for &v in &values[k..] {
        match bpv {
            8 => out.push(v as u8),
            16 => out.extend_from_slice(&(v as u16).to_le_bytes()),
            32 => out.extend_from_slice(&(v as u32).to_le_bytes()),
            _ => unreachable!("bpv is always 8, 16, or 32"),
        }
    }
}

/// Appends one document's `infoAndBits`/value pairs to the chunk buffer --
/// Java's `writeField` overloads writing into `bufferedDocs`.
fn serialize_doc_into(doc: &Document, out: &mut Vec<u8>) {
    for field in &doc.fields {
        let bits = match &field.value {
            FieldValue::String(_) => TYPE_STRING,
            FieldValue::Binary(_) => TYPE_BYTE_ARR,
            FieldValue::Int(_) => TYPE_NUMERIC_INT,
            FieldValue::Float(_) => TYPE_NUMERIC_FLOAT,
            FieldValue::Long(_) => TYPE_NUMERIC_LONG,
            FieldValue::Double(_) => TYPE_NUMERIC_DOUBLE,
        };
        let info_and_bits = ((field.field_number as i64) << TYPE_BITS) | bits;
        out.write_vlong(info_and_bits);
        write_field(out, &field.value);
    }
}

/// Port of `Lucene90CompressingStoredFieldsWriter.writeField` (encode side
/// of [`visit_field`]).
fn write_field(out: &mut Vec<u8>, value: &FieldValue) {
    match value {
        FieldValue::Binary(b) => {
            out.write_vint(b.len() as i32);
            out.write_bytes(b);
        }
        FieldValue::String(s) => out.write_string(s),
        FieldValue::Int(v) => write_zint(out, *v),
        FieldValue::Float(v) => write_zfloat(out, *v),
        FieldValue::Long(v) => write_tlong(out, *v),
        FieldValue::Double(v) => write_zdouble(out, *v),
    }
}

/// Port of `DataOutput.writeZInt` (32-bit zigzag, distinct from the 64-bit
/// `writeZLong` [`DataOutput::write_zlong`] provided elsewhere in this port).
fn write_zint(out: &mut Vec<u8>, v: i32) {
    let zigzag = ((v << 1) ^ (v >> 31)) as u32;
    out.write_vint(zigzag as i32);
}

/// Port of `Lucene90CompressingStoredFieldsWriter.writeZFloat`, the exact
/// inverse of [`read_zfloat`]: one byte for a small integral value in
/// `-1..=125`, four for any other positive float, five (a `0xFF` marker plus
/// the raw bits) for a negative one. `-0f` is excluded from the small-integer
/// case because `(int) -0f == 0` would round-trip it to `+0f`.
fn write_zfloat(out: &mut Vec<u8>, f: f32) {
    let int_val = f as i32;
    // `Float.floatToIntBits` collapses every NaN to the canonical quiet NaN;
    // Rust's `to_bits` preserves the payload, so canonicalize explicitly to
    // emit the same bytes Java would.
    let float_bits = if f.is_nan() { f32::NAN } else { f }.to_bits() as i32;
    if f == int_val as f32
        && (-1..=0x7D).contains(&int_val)
        && float_bits != (-0f32).to_bits() as i32
    {
        // ARITH: this arm is guarded by `(-1..=0x7D).contains(&int_val)`, so
        // the increment lands in `0..=126`.
        #[allow(clippy::arithmetic_side_effects)]
        out.push(0x80 | (1 + int_val) as u8);
    } else if (float_bits as u32) >> 31 == 0 {
        out.push((float_bits >> 24) as u8);
        out.write_i16((((float_bits as u32) >> 8) & 0xFFFF) as i16);
        out.push(float_bits as u8);
    } else {
        out.push(0xFF);
        out.write_i32(float_bits);
    }
}

/// Port of `Lucene90CompressingStoredFieldsWriter.writeZDouble`, the exact
/// inverse of [`read_zdouble`]: one byte for a small integral value in
/// `-1..=124`, five (`0xFE` + float bits) when the value is exactly
/// representable as an `f32`, eight for any other positive double, nine
/// (`0xFF` + raw bits) for a negative one.
fn write_zdouble(out: &mut Vec<u8>, d: f64) {
    let int_val = d as i32;
    // See [`write_zfloat`]: `Double.doubleToLongBits` canonicalizes NaN.
    let double_bits = if d.is_nan() { f64::NAN } else { d }.to_bits() as i64;
    if d == int_val as f64
        && (-1..=0x7C).contains(&int_val)
        && double_bits != (-0f64).to_bits() as i64
    {
        // ARITH: guarded by `(-1..=0x7C).contains(&int_val)`, so the
        // increment lands in `0..=125`.
        #[allow(clippy::arithmetic_side_effects)]
        out.push(0x80 | (int_val + 1) as u8);
    } else if d == d as f32 as f64 {
        out.push(0xFE);
        out.write_i32((d as f32).to_bits() as i32);
    } else if (double_bits as u64) >> 63 == 0 {
        out.push((double_bits >> 56) as u8);
        out.write_i32((((double_bits as u64) >> 24) & 0xFFFF_FFFF) as i32);
        out.write_i16((((double_bits as u64) >> 8) & 0xFFFF) as i16);
        out.push(double_bits as u8);
    } else {
        out.push(0xFF);
        out.write_i64(double_bits);
    }
}

/// Port of `Lucene90CompressingStoredFieldsWriter.writeTLong`, the exact
/// inverse of [`read_tlong`]: the header's top two bits pick a
/// second/hour/day divisor when the value is an exact multiple of one (which
/// is what makes millisecond timestamps cheap), the next bit says whether a
/// vlong follows, and the low five bits carry the bottom of the zigzagged
/// value.
///
/// The divisor tests use Java's `%` semantics on negative values, which
/// Rust's `%` shares (both truncate towards zero), so a negative timestamp
/// picks the same encoding in both.
fn write_tlong(out: &mut Vec<u8>, value: i64) {
    let (header_bits, l) = if value % SECOND != 0 {
        (0u8, value)
    } else if value % DAY == 0 {
        (DAY_ENCODING, value / DAY)
    } else if value % HOUR == 0 {
        (HOUR_ENCODING, value / HOUR)
    } else {
        (SECOND_ENCODING, value / SECOND)
    };

    let zigzag = lucene_util::zigzag::encode(l);
    let upper_bits = zigzag >> 5;
    let mut header = header_bits | (zigzag & 0x1F) as u8;
    if upper_bits != 0 {
        header |= 0x20;
    }
    out.push(header);
    if upper_bits != 0 {
        out.write_vlong(upper_bits as i64);
    }
}

/// A single, self-contained LZ4 "literal run" block wrapping `bytes`
/// verbatim -- no back-reference matches, valid per the LZ4 block spec
/// (a token's high nibble is the literal length, extended past 15 via
/// 0xFF continuation bytes; omitting the final match entirely is legal
/// when the literal run consumes the whole block).
///
/// Test-only: the real writer routes every unit through
/// [`crate::lz4::compress_with_dictionary`]. This is kept as the hand-built
/// shape the reader tests need in order to assemble a `.fdt` byte-for-byte
/// without depending on the compressor.
// ARITH: test-only helper (see `docs/arithmetic-gate.md`'s note on test
// code). `len` is a slice length, `rem` starts at `len - 0x0F` under the
// `len >= 0x0F` guard and only decreases while it is at least 0xFF.
#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
fn encode_literal_lz4(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let len = bytes.len();
    let nibble = len.min(0x0F);
    out.push((nibble as u8) << 4);
    if len >= 0x0F {
        let mut rem = len - 0x0F;
        while rem >= 0xFF {
            out.push(0xFF);
            rem -= 0xFF;
        }
        out.push(rem as u8);
    }
    out.extend_from_slice(bytes);
    out
}

/// Port of `Lucene90CompressingStoredFieldsReader.readField(DataInput,
/// StoredFieldVisitor, FieldInfo, int)`: decodes one value and hands it to
/// the visitor **borrowed**, so a visitor that does not keep it never
/// allocates it.
///
/// The one exception is `BYTE_ARR`, where the length is read off the payload
/// and the bytes are then borrowed straight out of the input rather than
/// copied into a `Vec` first -- Java passes a `StoredFieldDataInput` view for
/// the same reason.
fn visit_field<'a>(
    input: &mut SliceInput<'a>,
    field_number: i32,
    bits: i64,
    visitor: &mut dyn StoredFieldVisitor,
) -> Result<()> {
    match bits {
        TYPE_BYTE_ARR => {
            // The length is a vint out of the
            // decompressed payload, so an unbounded one is a `2^64` slice.
            let length = input.read_length("stored binary field")?;
            let start = input.position();
            input.skip(length)?;
            let bytes: &'a [u8] = input.slice(start, input.position())?;
            visitor.binary_field(field_number, bytes)
        }
        TYPE_STRING => {
            let value = input.read_string()?;
            visitor.string_field(field_number, &value)
        }
        TYPE_NUMERIC_INT => visitor.int_field(field_number, read_zint(input)?),
        TYPE_NUMERIC_FLOAT => visitor.float_field(field_number, read_zfloat(input)?),
        TYPE_NUMERIC_LONG => visitor.long_field(field_number, read_tlong(input)?),
        TYPE_NUMERIC_DOUBLE => visitor.double_field(field_number, read_zdouble(input)?),
        other => Err(Error::UnknownTypeTag(other)),
    }
}

/// Port of `Lucene90CompressingStoredFieldsReader.skipField`: advances the
/// cursor past one value without decoding it.
///
/// The numeric cases still *read* their bytes, because their encodings are
/// variable-length and self-delimiting -- Java calls the very same
/// `readZFloat`/`readTLong`/`readZDouble` here. What is saved is the
/// `String`/`Vec` a `STRING`/`BYTE_ARR` value would have allocated, which is
/// the whole of the cost for the wide documents this exists for.
fn skip_field(input: &mut SliceInput, bits: i64) -> Result<()> {
    match bits {
        TYPE_BYTE_ARR | TYPE_STRING => {
            let length = input.read_length("stored field")?;
            input.skip(length)?;
        }
        TYPE_NUMERIC_INT => {
            read_zint(input)?;
        }
        TYPE_NUMERIC_FLOAT => {
            read_zfloat(input)?;
        }
        TYPE_NUMERIC_LONG => {
            read_tlong(input)?;
        }
        TYPE_NUMERIC_DOUBLE => {
            read_zdouble(input)?;
        }
        other => return Err(Error::UnknownTypeTag(other)),
    }
    Ok(())
}

/// Port of `DataInput.readZInt`: `BitUtil.zigZagDecode` applied to a 32-bit
/// vint (distinct from [`lucene_util::zigzag::decode`], which is the 64-bit
/// vlong variant used elsewhere in this port).
fn read_zint(input: &mut SliceInput) -> Result<i32> {
    let v = input.read_vint()? as u32;
    // ARITH: `v & 1` is 0 or 1, so the unary negation is of 0 or 1 and can
    // never reach `i32::MIN`.
    #[allow(clippy::arithmetic_side_effects)]
    let sign = -((v & 1) as i32);
    Ok(((v >> 1) as i32) ^ sign)
}

/// Port of `Lucene90CompressingStoredFieldsReader.readZFloat`: 1-5 bytes,
/// small integral values (`-1..=125`) collapse to a single byte.
fn read_zfloat(input: &mut SliceInput) -> Result<f32> {
    let b = input.read_byte()? as i32;
    if b == 0xFF {
        Ok(f32::from_bits(input.read_i32()? as u32))
    } else if b & 0x80 != 0 {
        // ARITH: `b` is a byte widened to `i32`, so `b & 0x7f` is in
        // `0..=127` and the decrement lands in `-1..=126`.
        #[allow(clippy::arithmetic_side_effects)]
        Ok(((b & 0x7f) - 1) as f32)
    } else {
        let bits =
            (b << 24) | ((input.read_i16()? as u16 as i32) << 8) | (input.read_byte()? as i32);
        Ok(f32::from_bits(bits as u32))
    }
}

/// Port of `Lucene90CompressingStoredFieldsReader.readZDouble`: 1-9 bytes,
/// small integral values (`-1..=124`) collapse to a single byte, and a
/// double that's exactly representable as a float collapses to 5 bytes.
fn read_zdouble(input: &mut SliceInput) -> Result<f64> {
    let b = input.read_byte()? as i32;
    if b == 0xFF {
        Ok(f64::from_bits(input.read_i64()? as u64))
    } else if b == 0xFE {
        Ok(f32::from_bits(input.read_i32()? as u32) as f64)
    } else if b & 0x80 != 0 {
        // ARITH: as in `read_zfloat` -- `b & 0x7f` is in `0..=127`.
        #[allow(clippy::arithmetic_side_effects)]
        Ok(((b & 0x7f) - 1) as f64)
    } else {
        let bits = ((b as i64) << 56)
            | ((input.read_i32()? as u32 as i64) << 24)
            | ((input.read_i16()? as u16 as i64) << 8)
            | (input.read_byte()? as i64);
        Ok(f64::from_bits(bits as u64))
    }
}

/// Port of `Lucene90CompressingStoredFieldsReader.readTLong`: zigzag body
/// plus a scale factor (seconds/hours/days) for date-shaped longs.
fn read_tlong(input: &mut SliceInput) -> Result<i64> {
    let header = input.read_byte()?;
    let mut bits = (header & 0x1F) as u64;
    if header & 0x20 != 0 {
        bits |= (input.read_vlong()? as u64) << 5;
    }
    let l = lucene_util::zigzag::decode(bits);
    // `wrapping_mul`, not `checked_mul`: Java's `l * DAY` is a `long`
    // multiply, which wraps, and `writeTLong` only ever emits an `l` small
    // enough for the product to be exact. A corrupt vlong therefore decodes
    // to the same garbage `long` real Lucene would hand its
    // `StoredFieldVisitor`, rather than becoming a `Result` this port's
    // callers would then have to differ from Java about. Nothing downstream
    // uses the value as a length or an index.
    Ok(match header & DAY_ENCODING_MASK {
        SECOND_ENCODING => l.wrapping_mul(SECOND),
        HOUR_ENCODING => l.wrapping_mul(HOUR),
        DAY_ENCODING => l.wrapping_mul(DAY),
        0 => l,
        _ => unreachable!("only 2 bits, all 4 cases covered"),
    })
}

/// Port of `StoredFieldsInts.readInts`: a length-prefixed bulk int array in
/// one of three shapes -- an all-equal constant (`bpv == 0`), or a fixed
/// 8/16/32-bit width. At the widths, every whole 128-value block is stored
/// **bit-transposed** across `64 / bpv` i64 words (Java reads them with one
/// bulk `readLongs` and then unpacks, which is what makes the layout worth
/// having), and only the remainder past the last whole block is stored
/// value-by-value.
fn read_bulk_ints(input: &mut SliceInput, count: usize) -> Result<Vec<i64>> {
    let bpv = input.read_byte()?;
    match bpv {
        0 => {
            // The all-equal shape is the only one that can carry a negative
            // value -- the 8/16/32-bit shapes are masked. Both arrays this
            // function decodes are counts (`numStoredFields`) or byte lengths
            // (`lengths`), so a negative one is corruption; rejecting it here
            // is one comparison per *array* rather than one per document, and
            // it is what lets `read_chunk_header` build its cumulative
            // `offsets` without a per-document sign check.
            let v = input.read_vint()?;
            if v < 0 {
                return Err(lucene_store::Error::Corrupted(format!(
                    "negative constant in a stored-fields bulk int array: {v}"
                ))
                .into());
            }
            Ok(vec![i64::from(v); count])
        }
        // ARITH: for the whole arm -- `bpv` is one of exactly 8, 16 and 32, so
        // `values_per_word` is 8, 4 or 2 and `num_words` is 16, 32 or 64 --
        // all three divisions are by a non-zero constant and the shift is by
        // at most 32, well inside `u64`. `shift` peaks at
        // `(values_per_word - 1) * bpv == 64 - bpv == 56`, *not* 64, which
        // would be the panicking shift. `k` only ever advances while
        // `k + 128 <= count` or `k < count`, so it stays within `0..=count`
        // and `count` is a chunk's document count, at most 4096
        // (`read_chunk_header`); every write index `k + i + lane * num_words`
        // is under `k + num_words * values_per_word == k + 128 <= count`.
        #[allow(clippy::arithmetic_side_effects)]
        8 | 16 | 32 => {
            // Java transposes each 128-value block across `values_per_word`
            // i64 words (see `StoredFieldsInts.readInts8/16/32`): word `i`'s
            // `lane`-th slot (MSB-first) lands at output position
            // `i + lane*num_words`, not `i*values_per_word + lane`.
            const BLOCK_SIZE: usize = 128;
            let bpv_usize = bpv as usize;
            let values_per_word = 64 / bpv_usize;
            let num_words = BLOCK_SIZE / values_per_word;
            let mask: u64 = (1u64 << bpv_usize) - 1;
            debug_assert!((values_per_word - 1) * bpv_usize <= 56);

            let mut out = vec![0i64; count];
            // Java reads each block through one reusable `long[]`; hoisting
            // the scratch out of the loop keeps a 4096-document chunk to one
            // allocation per array instead of one per 128-value block.
            let mut words = vec![0i64; num_words];
            let mut k = 0usize;
            while k + BLOCK_SIZE <= count {
                input.read_i64s(&mut words)?;
                for (i, &w) in words.iter().enumerate() {
                    let uw = w as u64;
                    for lane in 0..values_per_word {
                        let shift = (values_per_word - 1 - lane) * bpv_usize;
                        out[k + i + lane * num_words] = ((uw >> shift) & mask) as i64;
                    }
                }
                k += BLOCK_SIZE;
            }
            while k < count {
                out[k] = read_scalar(input, bpv)?;
                k += 1;
            }
            Ok(out)
        }
        other => Err(Error::UnsupportedBulkIntWidth(other)),
    }
}

fn read_scalar(input: &mut SliceInput, bpv: u8) -> Result<i64> {
    Ok(match bpv {
        8 => input.read_byte()? as i64,
        16 => input.read_u16()? as i64,
        32 => input.read_i32()? as u32 as i64,
        _ => unreachable!("caller only passes 8, 16, or 32"),
    })
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
        ))
        .into());
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
fn decompress_unit(
    mode: Mode,
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
    // impossible value -- `original_length` is at most 2^44 (see
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
        ))
        .into());
    }
    // ARITH: `dict_length <= original_length` was just established, and
    // `.max(1)` keeps the divisor non-zero. `offset + length` is bounded by
    // `original_length` at both call sites in `decompress_range` -- the
    // non-sliced one passes a sub-range of `0..total_length` with
    // `original_length == total_length`, the sliced one clamps both ends to
    // the unit's own `unit_len` -- and `original_length` is itself at most
    // 2^44, so neither sum can leave `usize`.
    #[allow(clippy::arithmetic_side_effects)]
    let (num_blocks, want_end) = (
        (original_length - dict_length).div_ceil(block_length.max(1)),
        offset + length,
    );
    debug_assert!(want_end <= original_length);

    match mode {
        Mode::BestSpeed => {
            let dict_compressed_length = read_compressed_length(input)?;
            // Not `with_capacity(num_blocks)`: `num_blocks` is derived from
            // on-disk lengths, and a corrupt one would reserve gigabytes
            // before the first `read_vint` had a chance to hit EOF.
            let mut block_compressed_lengths = Vec::new();
            for _ in 0..num_blocks {
                block_compressed_lengths.push(read_compressed_length(input)?);
            }
            if length == 0 {
                // `checked_add`, folded over the run: these are byte counts
                // off disk, and even with each one bounded to `i32::MAX` a
                // long enough run wraps the total to a small skip that then
                // leaves the reader mid-unit -- the next `sliced` unit would
                // parse compressed bytes as a framing header and decode
                // plausible garbage.
                let mut total = dict_compressed_length;
                for &l in &block_compressed_lengths {
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
            let mut buffer = vec![0u8; buffer_len];
            // The dictionary is always decompressed: even when none of its
            // bytes are wanted, the sub-blocks back-reference into it.
            if lz4::decompress(input, dict_length, &mut buffer, 0)? != dict_length {
                return Err(lucene_store::Error::Corrupted(
                    "illegal dict length in LZ4 compression unit".into(),
                )
                .into());
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
            for &compressed_length in &block_compressed_lengths {
                let this_len = block_length.min(original_length - plain);
                let lo = offset.max(plain);
                let hi = want_end.min(plain + this_len);
                if lo < hi {
                    lz4::decompress(input, this_len, &mut buffer, dict_length)?;
                    out.extend_from_slice(
                        &buffer[dict_length + lo - plain..dict_length + hi - plain],
                    );
                } else {
                    input.skip(compressed_length)?;
                }
                plain += this_len;
            }
        }
        Mode::BestCompression => {
            let dict_compressed_length = read_compressed_length(input)?;
            if length == 0 {
                input.skip(dict_compressed_length)?;
                for _ in 0..num_blocks {
                    let compressed_length = read_compressed_length(input)?;
                    input.skip(compressed_length)?;
                }
                return Ok(());
            }

            // ARITH: as in the `BestSpeed` arm -- both lengths are bounded by
            // `original_length`.
            #[allow(clippy::arithmetic_side_effects)]
            let buffer_len = dict_length + block_length;
            let mut buffer = vec![0u8; buffer_len];
            deflate::decompress(input, dict_compressed_length, dict_length, &mut buffer, 0)?;
            if offset < dict_length {
                out.extend_from_slice(&buffer[offset..dict_length.min(want_end)]);
            }

            let mut plain = dict_length;
            // ARITH: identical to the `BestSpeed` arm's loop above.
            #[allow(clippy::arithmetic_side_effects)]
            for _ in 0..num_blocks {
                let this_len = block_length.min(original_length - plain);
                let compressed_length = read_compressed_length(input)?;
                let lo = offset.max(plain);
                let hi = want_end.min(plain + this_len);
                if lo < hi {
                    deflate::decompress(
                        input,
                        compressed_length,
                        this_len,
                        &mut buffer,
                        dict_length,
                    )?;
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

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    fn write_vint(out: &mut Vec<u8>, mut v: i32) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v = ((v as u32) >> 7) as i32;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    fn write_string(out: &mut Vec<u8>, s: &str) {
        write_vint(out, s.len() as i32);
        out.extend_from_slice(s.as_bytes());
    }

    /// A single-block, zero-dictionary `LZ4WithPresetDictCompressionMode`
    /// unit wrapping `bytes` verbatim (see `decompress_unit`'s doc comment
    /// for why the interleaved length vints can be anything -- they're
    /// only used by Java's partial-read optimization).
    fn encode_store_unit(bytes: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        write_vint(&mut out, 0); // dictLength = 0
        write_vint(&mut out, bytes.len().max(1) as i32); // blockLength
        write_vint(&mut out, 0); // dict's compressed length, unused
        write_vint(&mut out, 0); // block0's compressed length, unused
        out.push(0x00); // dict decompress unit: dictLength=0 -> single empty token
        out.extend(encode_literal_lz4(bytes));
        out
    }

    fn id() -> [u8; ID_LENGTH] {
        [7u8; ID_LENGTH]
    }

    /// Builds a valid `.fdt`+`.fdx`+`.fdm` trio for a single chunk containing
    /// exactly one document (`doc_bytes`, its already-encoded field entries).
    /// A single-doc chunk is the simplest valid framing (`numStoredFields`
    /// and length are each a plain vint, no bulk int array); the bulk-array
    /// path is exercised directly via `read_bulk_ints` tests instead.
    fn build_single_chunk_index(doc_bytes: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        build_single_chunk_index_with_meta_overrides(doc_bytes, 1, 0, 0)
    }

    /// The three tail counters are vlongs on the wire, so they are taken as
    /// `i64` here -- a test needs to be able to put a value past `i32` in one.
    fn build_single_chunk_index_with_meta_overrides(
        doc_bytes: &[u8],
        num_chunks_outer: i64,
        num_dirty_chunks: i64,
        num_dirty_docs: i64,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        // .fdt
        let mut fdt = Vec::new();
        fdt.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut fdt, DATA_CODEC_BEST_SPEED);
        fdt.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        fdt.extend_from_slice(&id());
        fdt.push(0); // empty suffix
        let chunk_start = fdt.len() as i64;

        write_vint(&mut fdt, 0); // docBase
        write_vint(&mut fdt, 1 << 2); // token: chunkDocs=1, sliced=0, dirty=0
        write_vint(&mut fdt, 1); // numStoredFields
        write_vint(&mut fdt, doc_bytes.len() as i32); // length
        fdt.extend(encode_store_unit(doc_bytes));
        fdt.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        fdt.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&fdt) as u64;
        fdt.extend_from_slice(&checksum.to_be_bytes());

        // .fdx: docs array [0] and startPointers array [chunk_start], 1 block each (blockShift=0)
        let mut fdx = Vec::new();
        fdx.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut fdx, INDEX_CODEC);
        fdx.extend_from_slice(&(INDEX_VERSION_CURRENT as u32).to_be_bytes());
        fdx.extend_from_slice(&id());
        fdx.push(0);
        let docs_start = fdx.len() as i64;
        // Both arrays are constant (bpv=0), so they need zero bytes here --
        // the constant value itself lives entirely in the .fdm meta below.
        let docs_end = fdx.len() as i64;
        let start_pointers_end = fdx.len() as i64;
        fdx.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        fdx.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&fdx) as u64;
        fdx.extend_from_slice(&checksum.to_be_bytes());

        // .fdm. The index arrays carry one extra sentinel entry beyond the
        // real chunk count (see `open`'s doc comment on `index_num_chunks`):
        // docs = [0 (chunk 0's docBase), maxDoc (sentinel)], startPointers =
        // [chunk_start, maxPointer (sentinel)]. blockShift=0 -> 1 value/block,
        // so that's 2 blocks each.
        let max_doc = 1i32;
        let max_pointer = (fdt.len() - codec_util::FOOTER_LENGTH) as i64;
        let mut fdm = Vec::new();
        fdm.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut fdm, META_CODEC);
        fdm.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        fdm.extend_from_slice(&id());
        fdm.push(0);
        write_vint(&mut fdm, 80 * 1024); // chunkSize (unused by the reader beyond framing)
        fdm.extend_from_slice(&max_doc.to_le_bytes());
        fdm.extend_from_slice(&0i32.to_le_bytes()); // blockShift = 0 -> 1 value per block
        fdm.extend_from_slice(&2i32.to_le_bytes()); // index_num_chunks = totalChunks(1) + 1
        fdm.extend_from_slice(&docs_start.to_le_bytes());
        for min in [0i64, max_doc as i64] {
            fdm.extend_from_slice(&min.to_le_bytes());
            fdm.extend_from_slice(&0i32.to_le_bytes()); // avg bits
            fdm.extend_from_slice(&0i64.to_le_bytes()); // offset
            fdm.push(0); // bpv
        }
        fdm.extend_from_slice(&docs_end.to_le_bytes());
        for min in [chunk_start, max_pointer] {
            fdm.extend_from_slice(&min.to_le_bytes());
            fdm.extend_from_slice(&0i32.to_le_bytes());
            fdm.extend_from_slice(&0i64.to_le_bytes());
            fdm.push(0);
        }
        fdm.extend_from_slice(&start_pointers_end.to_le_bytes());
        fdm.extend_from_slice(&max_pointer.to_le_bytes());
        write_vlong(&mut fdm, num_chunks_outer);
        write_vlong(&mut fdm, num_dirty_chunks);
        write_vlong(&mut fdm, num_dirty_docs);
        fdm.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        fdm.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&fdm) as u64;
        fdm.extend_from_slice(&checksum.to_be_bytes());

        (fdt, fdx, fdm)
    }

    fn field_bytes(field_number: i32, bits: i64, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let info_and_bits = ((field_number as i64) << TYPE_BITS) | bits;
        write_vlong(&mut out, info_and_bits);
        out.extend_from_slice(payload);
        out
    }

    fn write_vlong(out: &mut Vec<u8>, mut v: i64) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v = ((v as u64) >> 7) as i64;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    fn string_field_payload(s: &str) -> Vec<u8> {
        let mut out = Vec::new();
        write_string(&mut out, s);
        out
    }

    #[test]
    fn single_doc_single_string_field_round_trips() {
        let doc_bytes = field_bytes(0, TYPE_STRING, &string_field_payload("hello"));
        let (fdt, fdx, fdm) = build_single_chunk_index(&doc_bytes);
        let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
        assert_eq!(reader.max_doc(), 1);
        let doc = reader.document(0).unwrap();
        assert_eq!(doc.fields.len(), 1);
        assert_eq!(doc.fields[0].field_number, 0);
        assert_eq!(doc.fields[0].value, FieldValue::String("hello".to_string()));
    }

    #[test]
    fn doc_out_of_range_rejected() {
        let field = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
        let (fdt, fdx, fdm) = build_single_chunk_index(&field);
        let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
        assert!(matches!(
            reader.document(1),
            Err(Error::DocOutOfRange(1, 1))
        ));
        assert!(matches!(
            reader.document(-1),
            Err(Error::DocOutOfRange(-1, 1))
        ));
    }

    #[test]
    fn read_zint_round_trips_small_and_large_values() {
        for v in [0i32, 1, -1, 63, -64, 1_000_000, i32::MIN, i32::MAX] {
            let mut out = Vec::new();
            write_vint(&mut out, lucene_util::zigzag::encode(v as i64) as i32);
            let mut input = SliceInput::new(&out);
            assert_eq!(read_zint(&mut input).unwrap(), v, "value {v}");
        }
    }

    #[test]
    fn read_zfloat_small_integer_and_full_encoding() {
        // small integer: b = (value+1)|0x80
        let mut out = vec![((5i32 + 1) as u8) | 0x80];
        let mut input = SliceInput::new(&out);
        assert_eq!(read_zfloat(&mut input).unwrap(), 5.0);

        // full encoding: positive float, first byte < 0x80 and != 0xFF.
        // Layout is `b<<24 | (readShort()&0xFFFF)<<8 | readByte()`, and
        // `readShort`/`readInt` are little-endian, so the middle two bytes
        // on disk are (bits>>8)&0xFF then (bits>>16)&0xFF -- not the
        // natural big-endian byte order of `bits` itself.
        out.clear();
        let bits = 1.5f32.to_bits();
        out.push((bits >> 24) as u8);
        out.push((bits >> 8) as u8);
        out.push((bits >> 16) as u8);
        out.push(bits as u8);
        let mut input = SliceInput::new(&out);
        assert_eq!(read_zfloat(&mut input).unwrap(), 1.5);

        // negative value: leading 0xFF then a plain little-endian `readInt`.
        out.clear();
        out.push(0xFF);
        out.extend_from_slice(&(-2.5f32).to_bits().to_le_bytes());
        let mut input = SliceInput::new(&out);
        assert_eq!(read_zfloat(&mut input).unwrap(), -2.5);
    }

    #[test]
    fn read_zdouble_small_integer_float_and_full_encoding() {
        let mut out = vec![((3i32 + 1) as u8) | 0x80];
        let mut input = SliceInput::new(&out);
        assert_eq!(read_zdouble(&mut input).unwrap(), 3.0);

        // 0xFE marker + a plain little-endian `readInt` holding float bits.
        out = vec![0xFE];
        out.extend_from_slice(&2.5f32.to_bits().to_le_bytes());
        let mut input = SliceInput::new(&out);
        assert_eq!(read_zdouble(&mut input).unwrap(), 2.5);

        // 0xFF marker + a plain little-endian `readLong` holding double bits.
        out = vec![0xFF];
        out.extend_from_slice(&1.25f64.to_bits().to_le_bytes());
        let mut input = SliceInput::new(&out);
        assert_eq!(read_zdouble(&mut input).unwrap(), 1.25);

        // Full positive-double encoding: `b<<56 | (readInt()&0xFFFFFFFF)<<24
        // | (readShort()&0xFFFF)<<8 | readByte()`, with `readInt`/`readShort`
        // little-endian -- same byte-order subtlety as `read_zfloat`.
        out.clear();
        let bits: u64 = 4607182418800017408; // 1.0f64's bit pattern
        out.push((bits >> 56) as u8);
        out.extend_from_slice(&((bits >> 24) as u32).to_le_bytes());
        out.extend_from_slice(&((bits >> 8) as u16).to_le_bytes());
        out.push(bits as u8);
        let mut input = SliceInput::new(&out);
        assert_eq!(read_zdouble(&mut input).unwrap(), 1.0);
    }

    #[test]
    fn read_tlong_uncompressed_and_scaled_encodings() {
        // uncompressed: header bits 0x00, low 5 bits hold zigzag(value) directly if it fits
        let mut out = vec![lucene_util::zigzag::encode(7) as u8];
        let mut input = SliceInput::new(&out);
        assert_eq!(read_tlong(&mut input).unwrap(), 7);

        // second-scaled: value 5000ms = 5s -> zigzag(5)=10, header = 10 | SECOND_ENCODING
        out = vec![10u8 | SECOND_ENCODING];
        let mut input = SliceInput::new(&out);
        assert_eq!(read_tlong(&mut input).unwrap(), 5000);

        // day-scaled: value = 2 days = 2*86_400_000 ms -> zigzag(2)=4, header=4|DAY_ENCODING
        out = vec![4u8 | DAY_ENCODING];
        let mut input = SliceInput::new(&out);
        assert_eq!(read_tlong(&mut input).unwrap(), 2 * DAY);
    }

    #[test]
    fn read_bulk_ints_all_equal_shape() {
        let mut out = Vec::new();
        out.push(0); // bpv=0 marker
        write_vint(&mut out, 42);
        let mut input = SliceInput::new(&out);
        assert_eq!(read_bulk_ints(&mut input, 5).unwrap(), vec![42i64; 5]);
    }

    #[test]
    fn read_bulk_ints_scalar_tail_for_every_nonzero_width() {
        // count < 128 always takes the scalar tail loop (`read_scalar`),
        // regardless of bpv -- exercise all three non-constant widths.
        let mut out = vec![8u8, 10, 250];
        let mut input = SliceInput::new(&out);
        assert_eq!(read_bulk_ints(&mut input, 2).unwrap(), vec![10, 250]);

        out = vec![16u8];
        out.extend_from_slice(&300u16.to_le_bytes());
        out.extend_from_slice(&40000u16.to_le_bytes());
        input = SliceInput::new(&out);
        assert_eq!(read_bulk_ints(&mut input, 2).unwrap(), vec![300, 40000]);

        out = vec![32u8];
        out.extend_from_slice(&70000i32.to_le_bytes());
        out.extend_from_slice(&(-1i32).to_le_bytes());
        input = SliceInput::new(&out);
        assert_eq!(
            read_bulk_ints(&mut input, 2).unwrap(),
            vec![70000, 0xFFFFFFFF]
        );
    }

    #[test]
    fn read_bulk_ints_transposed_block_matches_java_layout() {
        // 128 sequential values 0..128, bpv=8: verifies the word/lane
        // transposition (see `read_bulk_ints`'s doc comment) against a
        // hand-encoded block using Java's exact readInts8 layout formula.
        let values: Vec<i64> = (0..128).collect();
        let mut out = vec![8u8]; // bpv=8
        let values_per_word = 64 / 8;
        let num_words = 128 / values_per_word;
        for w in 0..num_words {
            let mut word: u64 = 0;
            for lane in 0..values_per_word {
                let pos = w + lane * num_words;
                let shift = (values_per_word - 1 - lane) * 8;
                word |= (values[pos] as u64) << shift;
            }
            out.extend_from_slice(&(word as i64).to_le_bytes());
        }
        let mut input = SliceInput::new(&out);
        assert_eq!(read_bulk_ints(&mut input, 128).unwrap(), values);
    }

    #[test]
    fn read_bulk_ints_unsupported_width_rejected() {
        // Real Lucene's writer (`StoredFieldsInts.writeInts`) only ever emits
        // bpv 0 (constant-fill) or the smallest of 8/16/32 that fits the max
        // value, and Java's own `readInts` throws `IOException("Unsupported
        // number of bits per value: " + bpv)` for every other byte value --
        // confirmed against
        // lucene/core/.../codecs/lucene90/compressing/StoredFieldsInts.java.
        // Sweep the whole invalid space (not just one sentinel) to prove the
        // rejection boundary is exactly "not 0, 8, 16, or 32", matching
        // Java's switch/default exactly.
        for width in 0u16..=255 {
            let bpv = width as u8;
            if matches!(bpv, 0 | 8 | 16 | 32) {
                continue;
            }
            let out = vec![bpv];
            let mut input = SliceInput::new(&out);
            assert!(
                matches!(
                    read_bulk_ints(&mut input, 4),
                    Err(Error::UnsupportedBulkIntWidth(w)) if w == bpv
                ),
                "expected UnsupportedBulkIntWidth({bpv}) to be rejected"
            );
        }
    }

    #[test]
    fn wrong_segment_id_rejected() {
        let field = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
        let (fdt, fdx, fdm) = build_single_chunk_index(&field);
        let wrong_id = [9u8; ID_LENGTH];
        assert!(open(&fdt, &fdx, &fdm, &wrong_id, "").is_err());
    }

    #[test]
    fn num_chunks_mismatch_rejected() {
        // The test builder always writes index_num_chunks=2 (1 real chunk +
        // 1 sentinel); passing outer=2 breaks the required index=outer+1
        // relationship (2 != 2+1).
        let field = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
        let (fdt, fdx, fdm) = build_single_chunk_index_with_meta_overrides(&field, 2, 0, 0);
        assert!(matches!(
            open(&fdt, &fdx, &fdm, &id(), ""),
            Err(Error::NumChunksMismatch {
                index_num_chunks: 2,
                outer_num_chunks: 2
            })
        ));
    }

    #[test]
    fn too_many_dirty_chunks_rejected() {
        let field = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
        let (fdt, fdx, fdm) = build_single_chunk_index_with_meta_overrides(&field, 1, 2, 2);
        assert!(matches!(
            open(&fdt, &fdx, &fdm, &id(), ""),
            Err(Error::TooManyDirtyChunks(2, 1))
        ));
    }

    #[test]
    fn dirty_chunks_docs_mismatch_rejected() {
        let field = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
        let (fdt, fdx, fdm) = build_single_chunk_index_with_meta_overrides(&field, 1, 1, 0);
        assert!(matches!(
            open(&fdt, &fdx, &fdm, &id(), ""),
            Err(Error::DirtyChunksDocsMismatch(1, 0))
        ));
    }

    #[test]
    fn wrong_fdt_length_rejected() {
        let field = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
        let (mut fdt, fdx, fdm) = build_single_chunk_index(&field);
        fdt.push(0); // append a stray byte after the footer
        assert!(open(&fdt, &fdx, &fdm, &id(), "").is_err());
    }

    #[test]
    fn multi_doc_chunk_round_trips_through_bulk_int_arrays() {
        let doc0 = field_bytes(0, TYPE_STRING, &string_field_payload("aa"));
        let doc1 = field_bytes(1, TYPE_NUMERIC_INT, &{
            let mut p = Vec::new();
            write_vint(&mut p, lucene_util::zigzag::encode(5) as i32);
            p
        });

        // .fdt: a 2-doc chunk. numStoredFields uses the bpv=0 (all-equal)
        // shape (both docs have exactly 1 field); lengths uses bpv=8 scalar
        // bytes (the docs are different lengths), covering both bulk-array
        // shapes in one chunk.
        let mut fdt = Vec::new();
        fdt.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut fdt, DATA_CODEC_BEST_SPEED);
        fdt.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        fdt.extend_from_slice(&id());
        fdt.push(0);
        let chunk_start = fdt.len() as i64;

        write_vint(&mut fdt, 0); // docBase
        write_vint(&mut fdt, 2 << 2); // chunkDocs=2, sliced=0, dirty=0
        fdt.push(0); // numStoredFields: bpv=0 constant
        write_vint(&mut fdt, 1);
        fdt.push(8); // lengths: bpv=8 (count=2 < 128, so `read_bulk_ints`'s scalar tail loop)
        fdt.push(doc0.len() as u8);
        fdt.push(doc1.len() as u8);
        let payload = [doc0.clone(), doc1.clone()].concat();
        fdt.extend(encode_store_unit(&payload));
        fdt.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        fdt.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&fdt) as u64;
        fdt.extend_from_slice(&checksum.to_be_bytes());

        let (fdx, fdm) = build_fdx_fdm_for_single_chunk(&fdt, 2, chunk_start);
        let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
        assert_eq!(reader.max_doc(), 2);

        let d0 = reader.document(0).unwrap();
        assert_eq!(d0.fields[0].value, FieldValue::String("aa".to_string()));
        let d1 = reader.document(1).unwrap();
        assert_eq!(d1.fields[0].value, FieldValue::Int(5));
    }

    #[test]
    fn sliced_chunk_end_to_end_through_document() {
        // `sliced` only controls how many independent LZ4WithPresetDict units
        // back the chunk, not their size -- so a small payload with the
        // sliced bit set already exercises `document()`'s sliced branch
        // (one loop iteration, since `remaining` < the 80KB unit size).
        let doc_bytes = field_bytes(0, TYPE_STRING, &string_field_payload("sliced"));

        let mut fdt = Vec::new();
        fdt.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut fdt, DATA_CODEC_BEST_SPEED);
        fdt.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        fdt.extend_from_slice(&id());
        fdt.push(0);
        let chunk_start = fdt.len() as i64;

        write_vint(&mut fdt, 0); // docBase
        write_vint(&mut fdt, (1 << 2) | 1); // chunkDocs=1, sliced=1
        write_vint(&mut fdt, 1); // numStoredFields
        write_vint(&mut fdt, doc_bytes.len() as i32);
        fdt.extend(encode_store_unit(&doc_bytes));
        fdt.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        fdt.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&fdt) as u64;
        fdt.extend_from_slice(&checksum.to_be_bytes());

        let (fdx, fdm) = build_fdx_fdm_for_single_chunk(&fdt, 1, chunk_start);
        let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
        let doc = reader.document(0).unwrap();
        assert_eq!(
            doc.fields[0].value,
            FieldValue::String("sliced".to_string())
        );
    }

    #[test]
    fn empty_document_has_no_fields() {
        // A chunk with 2 docs, the first empty (numStoredFields=0, length=0)
        // -- exercises `document()`'s `doc_length == 0` shortcut.
        let doc1 = field_bytes(0, TYPE_STRING, &string_field_payload("x"));

        let mut fdt = Vec::new();
        fdt.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut fdt, DATA_CODEC_BEST_SPEED);
        fdt.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        fdt.extend_from_slice(&id());
        fdt.push(0);
        let chunk_start = fdt.len() as i64;

        write_vint(&mut fdt, 0); // docBase
        write_vint(&mut fdt, 2 << 2); // chunkDocs=2
        fdt.push(8); // numStoredFields: bpv=8, [0, 1]
        fdt.push(0);
        fdt.push(1);
        fdt.push(8); // lengths: bpv=8, [0, doc1.len()]
        fdt.push(0);
        fdt.push(doc1.len() as u8);
        fdt.extend(encode_store_unit(&doc1));
        fdt.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        fdt.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&fdt) as u64;
        fdt.extend_from_slice(&checksum.to_be_bytes());

        let (fdx, fdm) = build_fdx_fdm_for_single_chunk(&fdt, 2, chunk_start);
        let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
        let doc0 = reader.document(0).unwrap();
        assert!(doc0.fields.is_empty());
        let doc1_read = reader.document(1).unwrap();
        assert_eq!(
            doc1_read.fields[0].value,
            FieldValue::String("x".to_string())
        );
    }

    #[test]
    fn corrupt_chunk_bounds_rejected() {
        // A chunk header claiming chunkDocs=1 starting at docBase=0, but the
        // .fdx points a doc id (1) at this same chunk -- out of its range.
        let doc_bytes = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
        let (fdt, fdx, fdm) = build_single_chunk_index(&doc_bytes);
        // Patch maxDoc in .fdm (see build_single_chunk_index: a fixed i32 at
        // a known offset) up to 2 so `document(1)` is in-range per `open`'s
        // own doc-count check, but still out of the single real chunk.
        let mut fdm = fdm;
        let max_doc_offset =
            4 + 1 + META_CODEC.len() + 4 + ID_LENGTH + 1 + vint_len_test(80 * 1024);
        fdm[max_doc_offset..max_doc_offset + 4].copy_from_slice(&2i32.to_le_bytes());
        // Recompute the meta footer checksum after patching maxDoc: it
        // covers everything up to (not including) the trailing 8-byte
        // checksum field itself (footer magic + algorithm id are covered).
        let checksum_at = fdm.len() - 8;
        let checksum = crc32fast::hash(&fdm[..checksum_at]) as u64;
        fdm[checksum_at..].copy_from_slice(&checksum.to_be_bytes());

        let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
        assert!(matches!(
            reader.document(1),
            Err(Error::CorruptChunkBounds { .. })
        ));
    }

    fn vint_len_test(mut v: i32) -> usize {
        let mut n = 1;
        while (v as u32) >= 0x80 {
            v = ((v as u32) >> 7) as i32;
            n += 1;
        }
        n
    }

    #[test]
    fn decompress_unit_zero_length_produces_empty_vec() {
        let mut input = SliceInput::new(&[]);
        let mut out = Vec::new();
        decompress_unit(Mode::BestSpeed, &mut input, 0, 0, 0, &mut out).unwrap();
        assert_eq!(out, Vec::<u8>::new());
    }

    #[test]
    fn long_binary_field_exercises_extended_literal_length_encoding() {
        // >270 bytes forces `encode_literal_lz4`'s extended-length loop
        // (literalLen encoded as 0x0F + continuation bytes).
        let long_value = vec![b'q'; 300];
        let mut payload = Vec::new();
        write_vint(&mut payload, long_value.len() as i32);
        payload.extend_from_slice(&long_value);
        let doc_bytes = field_bytes(0, TYPE_BYTE_ARR, &payload);

        let (fdt, fdx, fdm) = build_single_chunk_index(&doc_bytes);
        let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
        let doc = reader.document(0).unwrap();
        assert_eq!(doc.fields[0].value, FieldValue::Binary(long_value));
    }

    #[test]
    fn large_field_number_exercises_vlong_continuation_byte() {
        // fieldNumber=20 -> infoAndBits = 20<<3 = 160, which needs a vlong
        // continuation byte (>127).
        let doc_bytes = field_bytes(20, TYPE_STRING, &string_field_payload("y"));
        let (fdt, fdx, fdm) = build_single_chunk_index(&doc_bytes);
        let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
        let doc = reader.document(0).unwrap();
        assert_eq!(doc.fields[0].field_number, 20);
        assert_eq!(doc.fields[0].value, FieldValue::String("y".to_string()));
    }

    #[test]
    fn sliced_chunk_splits_decompression_across_units() {
        // Force `sliced=true` with a payload spanning two 80KB-equivalent
        // units -- use a tiny "unit size" stand-in isn't possible (it's a
        // hardcoded constant in `document()`), so this test instead directly
        // exercises `decompress_unit` twice back to back the same way
        // `document()`'s sliced branch does, confirming concatenation is
        // correct. (An end-to-end `document()` sliced test would need a
        // payload > 80KB, impractical for a unit test.)
        let part_a = vec![b'a'; 100];
        let part_b = vec![b'b'; 50];
        let mut compressed = Vec::new();
        compressed.extend(encode_store_unit(&part_a));
        compressed.extend(encode_store_unit(&part_b));
        let mut input = SliceInput::new(&compressed);

        let mut out = Vec::new();
        decompress_unit(
            Mode::BestSpeed,
            &mut input,
            part_a.len(),
            0,
            part_a.len(),
            &mut out,
        )
        .unwrap();
        decompress_unit(
            Mode::BestSpeed,
            &mut input,
            part_b.len(),
            0,
            part_b.len(),
            &mut out,
        )
        .unwrap();

        let mut expected = part_a;
        expected.extend(part_b);
        assert_eq!(out, expected);
    }

    /// Builds a valid `.fdx`+`.fdm` pair for a single chunk of `chunk_docs`
    /// documents starting at `chunk_start` in `fdt`, sharing
    /// `multi_doc_chunk_round_trips_through_bulk_int_arrays`'s and
    /// `build_single_chunk_index`'s index/meta layout.
    fn build_fdx_fdm_for_single_chunk(
        fdt: &[u8],
        chunk_docs: i32,
        chunk_start: i64,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut fdx = Vec::new();
        fdx.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut fdx, INDEX_CODEC);
        fdx.extend_from_slice(&(INDEX_VERSION_CURRENT as u32).to_be_bytes());
        fdx.extend_from_slice(&id());
        fdx.push(0);
        let docs_start = fdx.len() as i64;
        let docs_end = fdx.len() as i64;
        let start_pointers_end = fdx.len() as i64;
        fdx.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        fdx.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&fdx) as u64;
        fdx.extend_from_slice(&checksum.to_be_bytes());

        // Index arrays carry a sentinel entry beyond the 1 real chunk (see
        // `open`'s doc comment on `index_num_chunks`): 2 blocks each.
        let max_pointer = (fdt.len() - codec_util::FOOTER_LENGTH) as i64;
        let mut fdm = Vec::new();
        fdm.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut fdm, META_CODEC);
        fdm.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        fdm.extend_from_slice(&id());
        fdm.push(0);
        write_vint(&mut fdm, 80 * 1024);
        fdm.extend_from_slice(&chunk_docs.to_le_bytes()); // maxDoc
        fdm.extend_from_slice(&0i32.to_le_bytes()); // blockShift
        fdm.extend_from_slice(&2i32.to_le_bytes()); // index_num_chunks = totalChunks(1) + 1
        fdm.extend_from_slice(&docs_start.to_le_bytes());
        for min in [0i64, chunk_docs as i64] {
            fdm.extend_from_slice(&min.to_le_bytes());
            fdm.extend_from_slice(&0i32.to_le_bytes());
            fdm.extend_from_slice(&0i64.to_le_bytes());
            fdm.push(0);
        }
        fdm.extend_from_slice(&docs_end.to_le_bytes());
        for min in [chunk_start, max_pointer] {
            fdm.extend_from_slice(&min.to_le_bytes());
            fdm.extend_from_slice(&0i32.to_le_bytes());
            fdm.extend_from_slice(&0i64.to_le_bytes());
            fdm.push(0);
        }
        fdm.extend_from_slice(&start_pointers_end.to_le_bytes());
        fdm.extend_from_slice(&max_pointer.to_le_bytes());
        write_vint(&mut fdm, 1);
        write_vint(&mut fdm, 0);
        write_vint(&mut fdm, 0);
        fdm.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        fdm.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&fdm) as u64;
        fdm.extend_from_slice(&checksum.to_be_bytes());

        (fdx, fdm)
    }

    /// Decodes one value through [`visit_field`] and hands back what the
    /// visitor was given -- the shape `read_field` used to have, now that
    /// there is one decode path and it is the visitor's.
    fn visit_one(payload: &[u8], bits: i64) -> Result<FieldValue> {
        let mut input = SliceInput::new(payload);
        let mut visitor = DocumentVisitor::all();
        visit_field(&mut input, 0, bits, &mut visitor)?;
        let mut doc = visitor.into_document();
        Ok(doc.fields.remove(0).value)
    }

    #[test]
    fn every_field_value_type_round_trips_through_visit_field() {
        let mut int_payload = Vec::new();
        write_vint(&mut int_payload, lucene_util::zigzag::encode(-7) as i32);
        assert_eq!(
            visit_one(&int_payload, TYPE_NUMERIC_INT).unwrap(),
            FieldValue::Int(-7)
        );

        // zigzag(7) = 14, fits directly in the header's low 5 bits with no
        // continuation byte and no second/hour/day scale applied.
        let mut long_payload = vec![lucene_util::zigzag::encode(7) as u8];
        assert_eq!(
            visit_one(&long_payload, TYPE_NUMERIC_LONG).unwrap(),
            FieldValue::Long(7)
        );

        let mut bin_payload = Vec::new();
        write_vint(&mut bin_payload, 3);
        bin_payload.extend_from_slice(b"xyz");
        assert_eq!(
            visit_one(&bin_payload, TYPE_BYTE_ARR).unwrap(),
            FieldValue::Binary(b"xyz".to_vec())
        );

        let mut str_payload = Vec::new();
        write_string(&mut str_payload, "hi");
        assert_eq!(
            visit_one(&str_payload, TYPE_STRING).unwrap(),
            FieldValue::String("hi".to_string())
        );

        long_payload = vec![((9i32 + 1) as u8) | 0x80];
        assert_eq!(
            visit_one(&long_payload, TYPE_NUMERIC_FLOAT).unwrap(),
            FieldValue::Float(9.0)
        );

        long_payload = vec![((2i32 + 1) as u8) | 0x80];
        assert_eq!(
            visit_one(&long_payload, TYPE_NUMERIC_DOUBLE).unwrap(),
            FieldValue::Double(2.0)
        );

        assert!(matches!(visit_one(&[], 6), Err(Error::UnknownTypeTag(6))));
    }

    /// [`skip_field`] must land the cursor exactly where [`visit_field`]
    /// does, for every encoding -- a skip that stops one byte short or long
    /// misreads every following field of the document rather than failing.
    #[test]
    fn skip_field_advances_exactly_as_far_as_visit_field() {
        let mut int_payload = Vec::new();
        write_vint(&mut int_payload, lucene_util::zigzag::encode(-7) as i32);
        let mut bin_payload = Vec::new();
        write_vint(&mut bin_payload, 3);
        bin_payload.extend_from_slice(b"xyz");
        let mut str_payload = Vec::new();
        write_string(&mut str_payload, "hi");

        let cases: [(Vec<u8>, i64); 6] = [
            (int_payload, TYPE_NUMERIC_INT),
            (
                vec![lucene_util::zigzag::encode(7) as u8],
                TYPE_NUMERIC_LONG,
            ),
            (bin_payload, TYPE_BYTE_ARR),
            (str_payload, TYPE_STRING),
            (vec![((9i32 + 1) as u8) | 0x80], TYPE_NUMERIC_FLOAT),
            (vec![((2i32 + 1) as u8) | 0x80], TYPE_NUMERIC_DOUBLE),
        ];
        for (payload, bits) in cases {
            let mut read = SliceInput::new(&payload);
            let mut visitor = DocumentVisitor::all();
            visit_field(&mut read, 0, bits, &mut visitor).unwrap();
            let mut skipped = SliceInput::new(&payload);
            skip_field(&mut skipped, bits).unwrap();
            assert_eq!(
                read.position(),
                skipped.position(),
                "skip_field disagreed with visit_field for bits={bits}"
            );
        }

        let mut unknown = SliceInput::new(&[]);
        assert!(matches!(
            skip_field(&mut unknown, 6),
            Err(Error::UnknownTypeTag(6))
        ));
    }

    fn id_write() -> [u8; ID_LENGTH] {
        [4u8; ID_LENGTH]
    }

    #[test]
    fn write_best_speed_single_doc_round_trips_through_own_reader() {
        let docs = vec![Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("hello world".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Int(-42),
                },
                StoredField {
                    field_number: 2,
                    value: FieldValue::Long(1_234_567_890_123),
                },
                StoredField {
                    field_number: 3,
                    value: FieldValue::Float(1.5),
                },
                StoredField {
                    field_number: 4,
                    value: FieldValue::Double(2.25),
                },
                StoredField {
                    field_number: 5,
                    value: FieldValue::Binary(vec![1, 2, 3, 4, 5]),
                },
            ],
        }];

        let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        assert_eq!(reader.max_doc(), 1);
        let got = reader.document(0).unwrap();
        assert_eq!(got.fields.len(), docs[0].fields.len());
        for (got_field, want_field) in got.fields.iter().zip(&docs[0].fields) {
            assert_eq!(got_field.field_number, want_field.field_number);
            assert_eq!(got_field.value, want_field.value);
        }
    }

    /// A wide document, read one field at a time through the visitor: the
    /// subset visitor must see exactly the field it asked for, and the
    /// values must equal what the whole-document path produces.
    #[test]
    fn visit_document_takes_only_the_fields_it_asks_for() {
        let docs = vec![wide_document()];
        let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        let whole = reader.document(0).unwrap();

        for want in &whole.fields {
            let mut visitor = DocumentVisitor::for_fields(&[want.field_number]);
            reader.visit_document(0, &mut visitor).unwrap();
            let got = visitor.into_document();
            assert_eq!(got.fields.len(), 1, "field {}", want.field_number);
            assert_eq!(got.fields[0].field_number, want.field_number);
            assert_eq!(got.fields[0].value, want.value);
        }

        // And a subset spanning both a skipped-then-taken and a
        // taken-then-skipped boundary.
        let mut visitor = DocumentVisitor::for_fields(&[1, 4]);
        reader.visit_document(0, &mut visitor).unwrap();
        let got = visitor.into_document();
        assert_eq!(
            got.fields
                .iter()
                .map(|f| f.field_number)
                .collect::<Vec<_>>(),
            vec![1, 4]
        );
    }

    /// `Status.STOP` must end the document where it is asked to, leaving
    /// every later field undecoded -- and `Status.NO` on the *last* field
    /// must take Java's "treat like STOP" shortcut rather than skipping into
    /// the end of the buffer.
    #[test]
    fn visit_document_stops_early_and_skips_the_last_field_without_reading_it() {
        struct StopAfter {
            stop_at: i32,
            seen: Vec<i32>,
            taken: Vec<i32>,
        }
        impl StoredFieldVisitor for StopAfter {
            fn needs_field(&mut self, field_number: i32) -> Result<VisitStatus> {
                self.seen.push(field_number);
                Ok(if field_number == self.stop_at {
                    VisitStatus::Stop
                } else {
                    VisitStatus::Yes
                })
            }
            fn string_field(&mut self, field_number: i32, _value: &str) -> Result<()> {
                self.taken.push(field_number);
                Ok(())
            }
            fn int_field(&mut self, field_number: i32, _value: i32) -> Result<()> {
                self.taken.push(field_number);
                Ok(())
            }
            fn long_field(&mut self, field_number: i32, _value: i64) -> Result<()> {
                self.taken.push(field_number);
                Ok(())
            }
            fn float_field(&mut self, field_number: i32, _value: f32) -> Result<()> {
                self.taken.push(field_number);
                Ok(())
            }
            fn double_field(&mut self, field_number: i32, _value: f64) -> Result<()> {
                self.taken.push(field_number);
                Ok(())
            }
            fn binary_field(&mut self, field_number: i32, _value: &[u8]) -> Result<()> {
                self.taken.push(field_number);
                Ok(())
            }
        }

        let docs = vec![wide_document()];
        let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();

        let mut visitor = StopAfter {
            stop_at: 2,
            seen: Vec::new(),
            taken: Vec::new(),
        };
        reader.visit_document(0, &mut visitor).unwrap();
        assert_eq!(visitor.seen, vec![0, 1, 2], "STOP did not end the walk");
        assert_eq!(visitor.taken, vec![0, 1]);

        // `NO` on the final field: the loop returns instead of calling
        // `skip_field`, so a truncated final value would not be noticed --
        // which is exactly Java's documented behaviour, and what this pins.
        let mut only_first = DocumentVisitor::for_fields(&[0]);
        reader.visit_document(0, &mut only_first).unwrap();
        assert_eq!(only_first.into_document().fields.len(), 1);
    }

    /// Six fields, one per encoding -- the shape a "wide document" test
    /// needs so that every `skip_field` arm is exercised by a *reader*, not
    /// only by the unit test that compares the two cursors.
    fn wide_document() -> Document {
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a fairly long stored value".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Int(-42),
                },
                StoredField {
                    field_number: 2,
                    value: FieldValue::Long(1_234_567_890_123),
                },
                StoredField {
                    field_number: 3,
                    value: FieldValue::Float(1.5),
                },
                StoredField {
                    field_number: 4,
                    value: FieldValue::Double(2.25),
                },
                StoredField {
                    field_number: 5,
                    value: FieldValue::Binary(vec![7u8; 64]),
                },
            ],
        }
    }

    #[test]
    fn write_best_speed_spans_many_chunks_and_every_doc_still_reads_back() {
        // Past `BEST_SPEED_MAX_DOCS_PER_CHUNK`, so the doc-count trigger fires
        // several times and the per-doc arrays cross whole transposed blocks.
        // Docs are small enough that the byte trigger never fires, which pins
        // the chunk boundaries at exact multiples of the doc cap.
        const N: usize = BEST_SPEED_MAX_DOCS_PER_CHUNK * 2 + 37;
        let docs: Vec<Document> = (0..N)
            .map(|i| Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(format!("doc-{i}")),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Long(i as i64 * -3),
                    },
                ],
            })
            .collect();

        let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "seg");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "seg").unwrap();
        assert_eq!(reader.max_doc(), N as i32);

        for i in 0..N {
            let doc = reader.document(i as i32).unwrap();
            assert_eq!(doc.fields.len(), 2, "doc {i}");
            assert_eq!(
                doc.fields[0].value,
                FieldValue::String(format!("doc-{i}")),
                "doc {i}"
            );
            assert_eq!(
                doc.fields[1].value,
                FieldValue::Long(i as i64 * -3),
                "doc {i}"
            );
        }
    }

    #[test]
    fn write_best_speed_closes_a_chunk_on_the_byte_trigger_too() {
        // Each doc carries more than a tenth of the chunk size, so the byte
        // trigger closes chunks long before the doc cap is anywhere near.
        let big = "x".repeat(BEST_SPEED_CHUNK_SIZE / 8);
        let docs: Vec<Document> = (0..40)
            .map(|i| Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String(format!("{i}-{big}")),
                }],
            })
            .collect();

        let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "seg");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "seg").unwrap();
        assert_eq!(reader.max_doc(), 40);
        for i in 0..40 {
            assert_eq!(
                reader.document(i).unwrap().fields[0].value,
                FieldValue::String(format!("{i}-{big}"))
            );
        }
    }

    #[test]
    fn write_best_speed_multi_doc_round_trips_with_varying_field_counts() {
        let docs = vec![
            Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String("doc0".to_string()),
                }],
            },
            Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("doc1-a".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Long(-7),
                    },
                ],
            },
            Document { fields: vec![] },
        ];

        let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "seg");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "seg").unwrap();
        assert_eq!(reader.max_doc(), 3);

        let doc0 = reader.document(0).unwrap();
        assert_eq!(doc0.fields.len(), 1);
        assert_eq!(doc0.fields[0].value, FieldValue::String("doc0".to_string()));

        let doc1 = reader.document(1).unwrap();
        assert_eq!(doc1.fields.len(), 2);
        assert_eq!(
            doc1.fields[0].value,
            FieldValue::String("doc1-a".to_string())
        );
        assert_eq!(doc1.fields[1].value, FieldValue::Long(-7));

        let doc2 = reader.document(2).unwrap();
        assert_eq!(doc2.fields.len(), 0);
    }

    #[test]
    fn write_best_speed_empty_doc_set_produces_zero_max_doc() {
        let (fdt, fdx, fdm) = write_best_speed(&[], &id_write(), "");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        assert_eq!(reader.max_doc(), 0);
    }

    #[test]
    fn write_best_compression_single_doc_round_trips_through_own_reader() {
        let docs = vec![Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("hello world".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Int(-42),
                },
                StoredField {
                    field_number: 2,
                    value: FieldValue::Long(1_234_567_890_123),
                },
                StoredField {
                    field_number: 3,
                    value: FieldValue::Float(1.5),
                },
                StoredField {
                    field_number: 4,
                    value: FieldValue::Double(2.25),
                },
                StoredField {
                    field_number: 5,
                    value: FieldValue::Binary(vec![1, 2, 3, 4, 5]),
                },
            ],
        }];

        let (fdt, fdx, fdm) = write_best_compression(&docs, &id_write(), "");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        assert_eq!(reader.max_doc(), 1);
        let got = reader.document(0).unwrap();
        assert_eq!(got.fields.len(), docs[0].fields.len());
        for (got_field, want_field) in got.fields.iter().zip(&docs[0].fields) {
            assert_eq!(got_field.field_number, want_field.field_number);
            assert_eq!(got_field.value, want_field.value);
        }
    }

    #[test]
    fn write_best_compression_multi_doc_round_trips_with_varying_field_counts() {
        let docs = vec![
            Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String("doc0".to_string()),
                }],
            },
            Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("doc1-a".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Long(-7),
                    },
                ],
            },
            Document { fields: vec![] },
        ];

        let (fdt, fdx, fdm) = write_best_compression(&docs, &id_write(), "seg");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "seg").unwrap();
        assert_eq!(reader.max_doc(), 3);

        let doc0 = reader.document(0).unwrap();
        assert_eq!(doc0.fields.len(), 1);
        assert_eq!(doc0.fields[0].value, FieldValue::String("doc0".to_string()));

        let doc1 = reader.document(1).unwrap();
        assert_eq!(doc1.fields.len(), 2);
        assert_eq!(
            doc1.fields[0].value,
            FieldValue::String("doc1-a".to_string())
        );
        assert_eq!(doc1.fields[1].value, FieldValue::Long(-7));

        let doc2 = reader.document(2).unwrap();
        assert_eq!(doc2.fields.len(), 0);
    }

    #[test]
    fn write_best_compression_empty_doc_set_produces_zero_max_doc() {
        let (fdt, fdx, fdm) = write_best_compression(&[], &id_write(), "");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        assert_eq!(reader.max_doc(), 0);
    }

    #[test]
    fn write_best_compression_large_payload_forces_multiple_sub_blocks() {
        // `dictLength = len / 60`, `blockLength = ceil((len - dictLength) / 10)`
        // (see write_best_compression's doc comment) -- a payload well past a
        // few KB guarantees `blockLength < len`, so `decompress_unit` must
        // walk more than one sub-block to reconstruct the chunk.
        let big_field = "the quick brown fox jumps over the lazy dog ".repeat(2000);
        let docs = vec![
            Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(big_field.clone()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Int(7),
                    },
                ],
            },
            Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String("second doc".to_string()),
                }],
            },
        ];

        let (fdt, fdx, fdm) = write_best_compression(&docs, &id_write(), "");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        assert_eq!(reader.max_doc(), 2);

        let doc0 = reader.document(0).unwrap();
        assert_eq!(doc0.fields[0].value, FieldValue::String(big_field));
        assert_eq!(doc0.fields[1].value, FieldValue::Int(7));

        let doc1 = reader.document(1).unwrap();
        assert_eq!(
            doc1.fields[0].value,
            FieldValue::String("second doc".to_string())
        );
    }

    #[test]
    fn write_zint_round_trips_through_read_zint() {
        for v in [0i32, 1, -1, i32::MIN, i32::MAX] {
            let mut out = Vec::new();
            write_zint(&mut out, v);
            let mut input = SliceInput::new(&out);
            assert_eq!(read_zint(&mut input).unwrap(), v, "value {v}");
        }
    }

    #[test]
    fn write_tlong_round_trips_through_read_tlong_across_every_time_scale() {
        // One value per header branch: sub-5-bit, needing a continuation
        // vlong, exact-day, exact-hour (but not day), exact-second (but not
        // hour), and not a whole second at all -- plus the extremes.
        for v in [
            0i64,
            1,
            -1,
            15,
            -16,
            1_000_000_000_000,
            2 * DAY,
            -3 * DAY,
            5 * HOUR,
            7 * SECOND,
            1_234,
            i64::MIN,
            i64::MAX,
        ] {
            let mut out = Vec::new();
            write_tlong(&mut out, v);
            let mut input = SliceInput::new(&out);
            assert_eq!(read_tlong(&mut input).unwrap(), v, "value {v}");
        }
    }

    /// The whole point of `writeTLong`'s scale headers is that a timestamp
    /// with day/hour/second precision costs far fewer bytes than its
    /// magnitude would otherwise need. Assert the encoding actually shrinks,
    /// not merely that it round-trips (which the always-10-byte encoding this
    /// replaced also did).
    #[test]
    fn write_tlong_uses_the_scale_headers_to_shrink_timestamps() {
        let one_day_ms = DAY;
        let mut scaled = Vec::new();
        write_tlong(&mut scaled, one_day_ms);
        assert_eq!(scaled.len(), 1, "1 day should fit the header's low 5 bits");
        assert_eq!(scaled[0] & DAY_ENCODING_MASK, DAY_ENCODING);

        let mut unscaled = Vec::new();
        write_tlong(&mut unscaled, one_day_ms + 1); // not a whole second
        assert!(
            unscaled.len() > scaled.len(),
            "a non-second-aligned value must not get a scale header"
        );
        assert_eq!(unscaled[0] & DAY_ENCODING_MASK, 0);
    }

    #[test]
    fn write_zfloat_round_trips_through_read_zfloat() {
        for v in [
            0.0f32,
            1.0,
            -1.0,
            125.0,
            126.0,
            1.5,
            -1.5,
            -0.0,
            f32::MIN,
            f32::MAX,
        ] {
            let mut out = Vec::new();
            write_zfloat(&mut out, v);
            let mut input = SliceInput::new(&out);
            let got = read_zfloat(&mut input).unwrap();
            assert_eq!(got, v, "value {v}");
            assert_eq!(
                got.is_sign_negative(),
                v.is_sign_negative(),
                "-0.0 must not collapse to +0.0 ({v})"
            );
        }
        // NaN survives as a NaN (Java canonicalizes it via floatToIntBits).
        let mut out = Vec::new();
        write_zfloat(&mut out, f32::NAN);
        assert!(read_zfloat(&mut SliceInput::new(&out)).unwrap().is_nan());
    }

    /// Small integral floats must take the one-byte encoding, and larger
    /// ones the 4/5-byte ones -- the sizes are the reason this encoding
    /// exists, so assert them rather than only the round trip.
    #[test]
    fn write_zfloat_uses_the_shortest_encoding_per_branch() {
        for (v, expected_len) in [
            (-1.0f32, 1usize),
            (0.0, 1),
            (125.0, 1),
            (126.0, 4), // past the small-integer range, still positive
            (1.5, 4),   // positive, not integral
            (-0.0, 5),  // negative zero is excluded from the small-int case
            (-1.5, 5),  // negative
        ] {
            let mut out = Vec::new();
            write_zfloat(&mut out, v);
            assert_eq!(out.len(), expected_len, "value {v}");
        }
    }

    #[test]
    fn write_zdouble_round_trips_through_read_zdouble() {
        for v in [
            0.0f64,
            1.0,
            -1.0,
            124.0,
            125.0,
            1.5,
            -1.5,
            -0.0,
            f64::MIN,
            f64::MAX,
            std::f64::consts::PI,
        ] {
            let mut out = Vec::new();
            write_zdouble(&mut out, v);
            let mut input = SliceInput::new(&out);
            let got = read_zdouble(&mut input).unwrap();
            assert_eq!(got, v, "value {v}");
            assert_eq!(
                got.is_sign_negative(),
                v.is_sign_negative(),
                "-0.0 must not collapse to +0.0 ({v})"
            );
        }
        let mut out = Vec::new();
        write_zdouble(&mut out, f64::NAN);
        assert!(read_zdouble(&mut SliceInput::new(&out)).unwrap().is_nan());
    }

    #[test]
    fn write_zdouble_uses_the_shortest_encoding_per_branch() {
        for (v, expected_len) in [
            (-1.0f64, 1usize),
            (0.0, 1),
            (124.0, 1),
            (125.0, 5),                 // past small-int, exact as f32
            (1.5, 5),                   // exact as f32
            (-1.5, 5),                  // negative but exact as f32
            (std::f64::consts::PI, 8),  // positive, not f32-exact
            (-std::f64::consts::PI, 9), // negative, not f32-exact
        ] {
            let mut out = Vec::new();
            write_zdouble(&mut out, v);
            assert_eq!(out.len(), expected_len, "value {v}");
        }
    }

    /// A single document larger than `2 * chunkSize` makes the writer mark
    /// the chunk `sliced` and split it into several independent
    /// `chunk_size`-plaintext compression units. The reader then has to cut
    /// the wanted byte range against each unit's own extent -- the one place
    /// two levels of splitting (chunk into units, unit into sub-blocks)
    /// compose. No hand-built fixture covers this; only the real writer
    /// produces it.
    #[test]
    fn writer_produced_sliced_chunk_round_trips_through_the_reader() {
        // ~230KB in one document: past 2 * BEST_SPEED_CHUNK_SIZE (160KB), so
        // three units. Compressible-but-not-trivially so, to keep the LZ4
        // sub-blocks from degenerating into single long matches.
        let big: Vec<u8> = (0..230_000u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
            .collect();
        let docs = vec![
            Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::Binary(big.clone()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String("trailing marker".to_string()),
                    },
                ],
            },
            Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String("next chunk".to_string()),
                }],
            },
        ];

        let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");

        // Prove the chunk really is sliced, rather than assuming the size
        // arithmetic worked out: re-read the first chunk's token.
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        let mut probe = SliceInput::new(&fdt);
        let block_start = reader.block_start_pointer(0).unwrap();
        probe.seek(block_start as usize).unwrap();
        probe.read_vint().unwrap(); // docBase
        let token = probe.read_vint().unwrap();
        assert_eq!(token & 1, 1, "expected the first chunk to be marked sliced");

        assert_eq!(reader.max_doc(), 2);
        let doc0 = reader.document(0).unwrap();
        assert_eq!(doc0.fields.len(), 2);
        assert_eq!(doc0.fields[0].value, FieldValue::Binary(big));
        assert_eq!(
            doc0.fields[1].value,
            FieldValue::String("trailing marker".to_string())
        );
        assert_eq!(
            reader.document(1).unwrap().fields[0].value,
            FieldValue::String("next chunk".to_string())
        );
    }

    /// The reader now decompresses only the sub-blocks a document actually
    /// intersects (skipping the rest by their recorded compressed length),
    /// so the *position* of a document inside the chunk is a real code path:
    /// wholly inside the dictionary prefix, spanning the dictionary/first
    /// sub-block seam, in the middle, and in the last sub-block.
    #[test]
    fn every_document_position_inside_a_multi_sub_block_chunk_reads_back() {
        // ~40KB of payload in one chunk: dictLength = 40000/20 = 2000 and
        // ten ~3.8KB sub-blocks, so documents land in all of the above.
        let docs: Vec<Document> = (0..400i32)
            .map(|i| Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(format!("document number {i} ").repeat(5)),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Int(i),
                    },
                ],
            })
            .collect();

        let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        assert_eq!(reader.max_doc(), 400);
        for i in 0..400i32 {
            let got = reader.document(i).unwrap();
            assert_eq!(got.fields.len(), 2, "doc {i}");
            assert_eq!(
                got.fields[0].value,
                FieldValue::String(format!("document number {i} ").repeat(5)),
                "doc {i}"
            );
            assert_eq!(got.fields[1].value, FieldValue::Int(i), "doc {i}");
        }
    }

    /// Same, for BEST_COMPRESSION: DEFLATE's sub-block lengths are
    /// interleaved rather than batched up front, so its skip path is a
    /// separate branch in `decompress_unit`.
    #[test]
    fn every_document_position_inside_a_deflate_multi_sub_block_chunk_reads_back() {
        let docs: Vec<Document> = (0..300)
            .map(|i| Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String(format!("deflate doc {i} ").repeat(9)),
                }],
            })
            .collect();

        let (fdt, fdx, fdm) = write_best_compression(&docs, &id_write(), "");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        for i in 0..300 {
            assert_eq!(
                reader.document(i).unwrap().fields[0].value,
                FieldValue::String(format!("deflate doc {i} ").repeat(9)),
                "doc {i}"
            );
        }
    }

    /// A `.fdt` whose unit header claims `blockLength == 0` while the unit
    /// still has bytes left to cover used to spin forever counting blocks
    /// (Java divides by it and throws). It must be reported as corruption.
    /// `.fdm`'s `chunkSize` is what a `sliced` chunk is divided by; zero
    /// would make the reader's unit loop make no progress.
    #[test]
    fn zero_chunk_size_in_meta_is_rejected() {
        let doc_bytes = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
        let (fdt, fdx, mut fdm) = build_single_chunk_index(&doc_bytes);
        // Rewrite the chunkSize vint (first byte after the index header) to 0
        // and re-checksum. `build_single_chunk_index` writes 80*1024, which is
        // a 3-byte vint, so overwrite all three with a single 0 plus padding
        // the reader will never reach... simpler: rebuild via a targeted patch.
        let header_len = 4 + 1 + META_CODEC.len() + 4 + ID_LENGTH + 1;
        // 80*1024 = 0x14000 -> vint bytes [0x80, 0x80, 0x05]; make it [0x80, 0x80, 0x00] = 0.
        fdm[header_len + 2] = 0x00;
        let body_len = fdm.len() - codec_util::FOOTER_LENGTH;
        let checksum = crc32fast::hash(&fdm[..body_len + 8]) as u64;
        fdm[body_len + 8..].copy_from_slice(&checksum.to_be_bytes());
        assert!(open(&fdt, &fdx, &fdm, &id(), "").is_err());
    }

    #[test]
    fn zero_block_length_in_a_unit_header_is_rejected_not_hung_on() {
        let mut unit = Vec::new();
        write_vint(&mut unit, 0); // dictLength
        write_vint(&mut unit, 0); // blockLength = 0, but originalLength > 0
        let mut input = SliceInput::new(&unit);
        let mut out = Vec::new();
        let err = decompress_unit(Mode::BestSpeed, &mut input, 8, 0, 8, &mut out).unwrap_err();
        assert!(
            matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
            "unexpected error: {err:?}"
        );
    }

    /// A dictionary longer than the whole unit is likewise impossible and
    /// would otherwise make the block loop's `original_length - plain`
    /// underflow.
    #[test]
    fn dictionary_longer_than_the_unit_is_rejected() {
        let mut unit = Vec::new();
        write_vint(&mut unit, 100); // dictLength > originalLength
        write_vint(&mut unit, 4);
        let mut input = SliceInput::new(&unit);
        let mut out = Vec::new();
        assert!(decompress_unit(Mode::BestSpeed, &mut input, 8, 0, 8, &mut out).is_err());
    }

    #[test]
    fn write_bulk_ints_round_trips_across_whole_transposed_blocks() {
        // Sizes that straddle the 128-value block boundary in every direction:
        // just under one block, exactly one, one plus a tail, and several.
        for count in [127usize, 128, 129, 200, 256, 300] {
            for width_cap in [0xFFi64, 0xFFFF, 0xFFFF_FFFFu32 as i64] {
                // Vary the values so the all-equal shortcut never fires, and
                // make each position distinguishable so a transposition slip
                // shows up as a mismatch rather than a coincidence.
                let values: Vec<i64> = (0..count)
                    .map(|k| {
                        if k == 0 {
                            width_cap
                        } else {
                            (k as i64) % width_cap
                        }
                    })
                    .collect();
                let mut out = Vec::new();
                write_bulk_ints(&mut out, &values);
                let mut input = SliceInput::new(&out);
                assert_eq!(
                    read_bulk_ints(&mut input, count).unwrap(),
                    values,
                    "count={count} width_cap={width_cap}"
                );
            }
        }
    }

    #[test]
    fn write_bulk_ints_all_equal_and_varying_widths_round_trip() {
        for values in [
            vec![5i64, 5, 5, 5],
            vec![1i64, 200, 3, 4],
            vec![1i64, 70000, 3, 4],
            vec![1i64, 4_000_000_000, 3, 4],
        ] {
            let mut out = Vec::new();
            write_bulk_ints(&mut out, &values);
            let mut input = SliceInput::new(&out);
            assert_eq!(read_bulk_ints(&mut input, values.len()).unwrap(), values);
        }
    }

    #[test]
    fn encode_literal_lz4_round_trips_through_lz4_decompress() {
        for payload in [
            Vec::new(),
            b"short".to_vec(),
            vec![0x42; 5000], // forces the 0xFF-continuation length encoding
        ] {
            let encoded = encode_literal_lz4(&payload);
            let mut input = SliceInput::new(&encoded);
            let mut dest = vec![0u8; payload.len()];
            lz4::decompress(&mut input, payload.len(), &mut dest, 0).unwrap();
            assert_eq!(dest, payload);
        }
    }

    // --- streaming writer: the three merge paths ---

    /// `N` small documents, deterministic and self-identifying so a
    /// bulk-copy boundary error shows up as a document holding another
    /// document's values rather than as a decode error.
    fn merge_docs(tag: &str, n: usize) -> Vec<Document> {
        (0..n)
            .map(|i| Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(format!("{tag}-{i}")),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Long(i as i64 * 7 - 3),
                    },
                ],
            })
            .collect()
    }

    fn assert_doc_is(reader: &StoredFieldsReader<'_>, doc_id: i32, tag: &str, i: usize) {
        let doc = reader.document(doc_id).unwrap();
        assert_eq!(doc.fields.len(), 2, "doc {doc_id}");
        assert_eq!(
            doc.fields[0].value,
            FieldValue::String(format!("{tag}-{i}")),
            "doc {doc_id} field 0"
        );
        assert_eq!(
            doc.fields[1].value,
            FieldValue::Long(i as i64 * 7 - 3),
            "doc {doc_id} field 1"
        );
    }

    #[test]
    fn copy_chunks_of_two_whole_segments_reproduces_every_document() {
        // Both sources span several full chunks plus a trailing dirty one, so
        // the copy loop runs over real chunk boundaries. Each source is copied
        // whole, which is exactly what an unsorted, deletion-free merge does.
        const N: usize = BEST_SPEED_MAX_DOCS_PER_CHUNK * 2 + 51;
        let a = merge_docs("a", N);
        let b = merge_docs("b", N);
        let (fdt_a, fdx_a, fdm_a) = write_best_speed(&a, &id_write(), "");
        let (fdt_b, fdx_b, fdm_b) = write_best_speed(&b, &id_write(), "");
        let ra = open(&fdt_a, &fdx_a, &fdm_a, &id_write(), "").unwrap();
        let rb = open(&fdt_b, &fdx_b, &fdm_b, &id_write(), "").unwrap();
        assert!(ra.num_chunks() >= 3, "source should span several chunks");

        let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
        assert!(w.can_bulk_copy(&ra) && w.can_bulk_copy(&rb));
        w.copy_chunks(&ra, 0, N as i32).unwrap();
        w.copy_chunks(&rb, 0, N as i32).unwrap();
        let (fdt, fdx, fdm) = w.finish();

        let merged = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        assert_eq!(merged.max_doc(), 2 * N as i32);
        for i in 0..N {
            assert_doc_is(&merged, i as i32, "a", i);
            assert_doc_is(&merged, (N + i) as i32, "b", i);
        }
        // Every chunk was copied, so nothing was recompressed: the merged
        // `.fdt` is the two sources' chunk bodies plus rebased headers.
        assert_eq!(merged.num_chunks(), ra.num_chunks() + rb.num_chunks());
    }

    #[test]
    fn copy_chunks_of_a_partial_range_copies_the_ragged_ends_document_at_a_time() {
        // A run that starts and ends mid-chunk -- what a sorted merge produces
        // -- must still reproduce exactly those documents. Java reaches the
        // same two document-at-a-time loops through its `isLoaded(docID)`
        // check; this port tests the chunk-boundary condition directly.
        const N: usize = BEST_SPEED_MAX_DOCS_PER_CHUNK * 3;
        let docs = merge_docs("p", N);
        let (fdt_s, fdx_s, fdm_s) = write_best_speed(&docs, &id_write(), "");
        let src = open(&fdt_s, &fdx_s, &fdm_s, &id_write(), "").unwrap();
        assert_eq!(src.num_chunks(), 3);

        let from = 5i32;
        let to = (BEST_SPEED_MAX_DOCS_PER_CHUNK * 2 + 9) as i32;
        let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
        w.copy_chunks(&src, from, to).unwrap();
        let (fdt, fdx, fdm) = w.finish();

        let merged = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        assert_eq!(merged.max_doc(), to - from);
        for (merged_id, source_id) in (from..to).enumerate() {
            assert_doc_is(&merged, merged_id as i32, "p", source_id as usize);
        }
    }

    #[test]
    fn copy_chunks_of_a_range_inside_one_chunk_copies_no_chunk_at_all() {
        // `from_pointer == to_pointer`: no whole chunk fits, so every document
        // goes through the DOC path and the bulk loop must not run.
        let docs = merge_docs("q", 40);
        let (fdt_s, fdx_s, fdm_s) = write_best_speed(&docs, &id_write(), "");
        let src = open(&fdt_s, &fdx_s, &fdm_s, &id_write(), "").unwrap();
        assert_eq!(src.num_chunks(), 1);

        let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
        w.copy_chunks(&src, 10, 20).unwrap();
        let (fdt, fdx, fdm) = w.finish();
        let merged = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        assert_eq!(merged.max_doc(), 10);
        for (merged_id, source_id) in (10..20).enumerate() {
            assert_doc_is(&merged, merged_id as i32, "q", source_id);
        }
    }

    #[test]
    fn copy_chunks_after_buffered_documents_forces_a_dirty_flush_first() {
        // Java's `if (numBufferedDocs > 0) flush(true);`: a bulk copy appends
        // whole chunks, so anything already buffered has to close as its own
        // (dirty) chunk first, or the copied chunks' `docBase`s would be wrong.
        let src_docs = merge_docs("s", BEST_SPEED_MAX_DOCS_PER_CHUNK + 5);
        let (fdt_s, fdx_s, fdm_s) = write_best_speed(&src_docs, &id_write(), "");
        let src = open(&fdt_s, &fdx_s, &fdm_s, &id_write(), "").unwrap();

        let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
        for doc in merge_docs("v", 3).iter() {
            w.add_document(doc);
        }
        w.copy_chunks(&src, 0, src_docs.len() as i32).unwrap();
        let (fdt, fdx, fdm) = w.finish();

        let merged = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        assert_eq!(merged.max_doc(), (3 + src_docs.len()) as i32);
        for i in 0..3 {
            assert_doc_is(&merged, i as i32, "v", i);
        }
        for i in 0..src_docs.len() {
            assert_doc_is(&merged, (3 + i) as i32, "s", i);
        }
        // The three buffered docs became a dirty chunk of their own, and the
        // source's own trailing dirty chunk came across with its bit set.
        assert!(merged.num_dirty_chunks() >= 2);
        assert!(merged.num_dirty_docs() >= 3 + 5);
    }

    #[test]
    fn serialized_documents_round_trip_through_the_doc_path() {
        // The DOC path: bytes copied without ever being parsed into fields.
        let docs = merge_docs("d", 300);
        let (fdt_s, fdx_s, fdm_s) = write_best_speed(&docs, &id_write(), "");
        let src = open(&fdt_s, &fdx_s, &fdm_s, &id_write(), "").unwrap();

        let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
        // Only the even documents, the way a source with deletions merges.
        for i in (0..docs.len()).step_by(2) {
            let d = src.serialized_document(i as i32).unwrap();
            w.add_serialized_document(d.num_stored_fields, &d.bytes);
        }
        let (fdt, fdx, fdm) = w.finish();
        let merged = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        assert_eq!(merged.max_doc(), 150);
        for (merged_id, source_id) in (0..docs.len()).step_by(2).enumerate() {
            assert_doc_is(&merged, merged_id as i32, "d", source_id);
        }
    }

    #[test]
    fn a_documents_serialized_bytes_are_exactly_what_the_writer_buffered() {
        // `serialized_document` is only safe to copy if it is byte-identical
        // to what `serialize_doc_into` produced -- otherwise the DOC path
        // silently rewrites documents.
        let docs = vec![Document {
            fields: vec![
                StoredField {
                    field_number: 2,
                    value: FieldValue::String("hello".to_string()),
                },
                StoredField {
                    field_number: 9,
                    value: FieldValue::Binary(vec![0, 1, 255, 7]),
                },
                StoredField {
                    field_number: 0,
                    value: FieldValue::Double(-0.5),
                },
            ],
        }];
        let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        let got = reader.serialized_document(0).unwrap();
        let mut want = Vec::new();
        serialize_doc_into(&docs[0], &mut want);
        assert_eq!(got.bytes, want);
        assert_eq!(got.num_stored_fields, 3);
    }

    #[test]
    fn an_empty_document_serializes_to_no_bytes_and_no_fields() {
        let docs = vec![Document::default(), merge_docs("e", 1).remove(0)];
        let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        let empty = reader.serialized_document(0).unwrap();
        assert_eq!(empty.num_stored_fields, 0);
        assert!(empty.bytes.is_empty());
        assert!(reader.document(0).unwrap().fields.is_empty());
    }

    #[test]
    fn dirtiness_accumulates_across_bulk_copies_until_the_segment_is_too_dirty() {
        // `tooDirty` is Java's safety switch against a segment's compression
        // ratio degrading forever: every generation of bulk copies carries the
        // previous generation's incomplete chunks across verbatim. A single
        // segment this port flushes can never be too dirty (its one forced
        // flush leaves fewer than `maxDocsPerChunk` dirty docs) -- but a
        // *merged* one accumulates one dirty chunk per bulk-copied source.
        let docs = merge_docs("t", 1);
        let (fdt_s, fdx_s, fdm_s) = write_best_speed(&docs, &id_write(), "");
        let src = open(&fdt_s, &fdx_s, &fdm_s, &id_write(), "").unwrap();
        assert_eq!(src.num_dirty_chunks(), 1);

        let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
        assert!(
            !w.too_dirty(&src),
            "a single flushed segment is never dirty"
        );
        let copies = BEST_SPEED_MAX_DOCS_PER_CHUNK + 1;
        for _ in 0..copies {
            w.copy_chunks(&src, 0, 1).unwrap();
        }
        let (fdt, fdx, fdm) = w.finish();
        let merged = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        assert_eq!(merged.max_doc(), copies as i32);
        assert_eq!(merged.num_dirty_docs(), copies as i64);
        assert_eq!(merged.num_dirty_chunks(), copies as i64);
        for i in 0..copies {
            assert_doc_is(&merged, i as i32, "t", 0);
        }

        let w2 = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
        assert!(w2.too_dirty(&merged));
        assert!(!w2.can_bulk_copy(&merged));
    }

    #[test]
    fn a_best_compression_source_cannot_be_bulk_copied_into_a_best_speed_writer() {
        // Different compressor, so the chunk payload framing differs: copying
        // it verbatim would produce bytes the merged segment's own reader
        // decodes with the wrong decompressor.
        let docs = merge_docs("c", 20);
        let (fdt, fdx, fdm) = write_best_compression(&docs, &id_write(), "");
        let src = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        assert_eq!(src.mode(), Mode::BestCompression);
        assert_eq!(src.chunk_size(), BEST_COMPRESSION_CHUNK_SIZE as i32);

        let w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
        assert!(!w.can_bulk_copy(&src));
        // ...but the DOC path is compressor-independent: serialized bytes are
        // the same either way.
        let mut w = w;
        for i in 0..docs.len() {
            let d = src.serialized_document(i as i32).unwrap();
            w.add_serialized_document(d.num_stored_fields, &d.bytes);
        }
        let (fdt, fdx, fdm) = w.finish();
        let merged = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        assert_eq!(merged.mode(), Mode::BestSpeed);
        for i in 0..docs.len() {
            assert_doc_is(&merged, i as i32, "c", i);
        }
    }

    #[test]
    fn copy_chunks_rejects_an_out_of_range_document_range() {
        let docs = merge_docs("r", 10);
        let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
        let src = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
        assert!(matches!(
            w.copy_chunks(&src, 0, 11),
            Err(Error::DocOutOfRange(_, 10))
        ));
        assert!(matches!(
            w.copy_chunks(&src, 5, 4),
            Err(Error::InvertedDocRange {
                from_doc: 5,
                to_doc: 4
            })
        ));
        // And a writer that cannot legally bulk-copy from this reader at all
        // is refused in release builds too, not only under `debug_assert`.
        let mut deflate_writer = StoredFieldsWriter::new(Mode::BestCompression, &id_write(), "");
        assert!(matches!(
            deflate_writer.copy_chunks(&src, 0, 10),
            Err(Error::BulkCopyNotPermitted {
                same_mode: false,
                ..
            })
        ));
    }

    #[test]
    fn copy_chunks_of_an_empty_range_writes_nothing() {
        let docs = merge_docs("z", 10);
        let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
        let src = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
        w.copy_chunks(&src, 3, 3).unwrap();
        let (fdt, fdx, fdm) = w.finish();
        let merged = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        assert_eq!(merged.max_doc(), 0);
        assert_eq!(merged.num_chunks(), 0);
    }

    #[test]
    fn chunk_for_doc_reports_the_containing_chunks_base_and_offset() {
        const N: usize = BEST_SPEED_MAX_DOCS_PER_CHUNK * 2 + 3;
        let docs = merge_docs("k", N);
        let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        let cap = BEST_SPEED_MAX_DOCS_PER_CHUNK as i32;
        assert_eq!(reader.chunk_for_doc(0).unwrap().doc_base, 0);
        assert_eq!(reader.chunk_for_doc(cap - 1).unwrap().doc_base, 0);
        assert_eq!(reader.chunk_for_doc(cap).unwrap().doc_base, cap);
        assert_eq!(reader.chunk_for_doc(2 * cap).unwrap().doc_base, 2 * cap);
        // Start pointers are strictly increasing and the last chunk's data
        // ends at `maxPointer`.
        assert!(
            reader.chunk_for_doc(0).unwrap().start_pointer
                < reader.chunk_for_doc(cap).unwrap().start_pointer
        );
        assert!(reader.chunk_for_doc(2 * cap).unwrap().start_pointer < reader.max_pointer());
        assert!(reader.chunk_for_doc(N as i32).is_err());
    }

    #[test]
    fn a_chunk_cursor_serves_every_document_of_a_chunk_from_one_decompression() {
        const N: usize = BEST_SPEED_MAX_DOCS_PER_CHUNK * 2 + 17;
        let docs = merge_docs("cc", N);
        let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();

        let mut cursor = ChunkCursor::new();
        for i in 0..N {
            let (num_stored_fields, bytes) = cursor.document(&reader, i as i32).unwrap();
            assert_eq!(num_stored_fields, 2, "doc {i}");
            // Identical to the random-access read, byte for byte.
            let random = reader.serialized_document(i as i32).unwrap();
            assert_eq!(bytes, random.bytes.as_slice(), "doc {i}");
            let parsed = parse_document(num_stored_fields, bytes).unwrap();
            let direct = reader.document(i as i32).unwrap();
            assert_eq!(parsed.fields.len(), direct.fields.len(), "doc {i}");
            for (a, b) in parsed.fields.iter().zip(&direct.fields) {
                assert_eq!(a.field_number, b.field_number, "doc {i}");
                assert_eq!(a.value, b.value, "doc {i}");
            }
        }

        // Backwards, and jumping across chunks, must reload rather than
        // return a stale chunk's document.
        for i in (0..N).rev().step_by(7) {
            let (_, bytes) = cursor.document(&reader, i as i32).unwrap();
            assert_eq!(
                bytes,
                reader
                    .serialized_document(i as i32)
                    .unwrap()
                    .bytes
                    .as_slice(),
                "doc {i}"
            );
        }
        cursor.reset();
        let (_, bytes) = cursor.document(&reader, 0).unwrap();
        assert_eq!(
            bytes,
            reader.serialized_document(0).unwrap().bytes.as_slice()
        );
    }

    #[test]
    fn read_chunk_reports_its_own_extent_and_rejects_documents_outside_it() {
        const N: usize = BEST_SPEED_MAX_DOCS_PER_CHUNK + 40;
        let docs = merge_docs("rc", N);
        let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();

        let first = reader.read_chunk(0).unwrap();
        assert_eq!(first.doc_base(), 0);
        assert_eq!(first.num_docs(), BEST_SPEED_MAX_DOCS_PER_CHUNK as i32);
        assert!(first.contains(0) && first.contains(first.num_docs() - 1));
        assert!(!first.contains(first.num_docs()));
        assert!(first.document(first.num_docs()).is_none());

        let second = reader
            .read_chunk(BEST_SPEED_MAX_DOCS_PER_CHUNK as i32)
            .unwrap();
        assert_eq!(second.doc_base(), BEST_SPEED_MAX_DOCS_PER_CHUNK as i32);
        assert_eq!(second.num_docs(), 40);
        assert!(reader.read_chunk(N as i32).is_err());
    }

    #[test]
    fn a_chunk_holding_only_empty_documents_decompresses_to_nothing() {
        // Every document has zero stored fields, so the chunk's payload is
        // zero bytes -- `read_chunk` must not try to decompress an empty unit.
        let docs = vec![Document::default(); 5];
        let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        let chunk = reader.read_chunk(0).unwrap();
        assert_eq!(chunk.num_docs(), 5);
        for i in 0..5 {
            let (n, bytes) = chunk.document(i).unwrap();
            assert_eq!(n, 0);
            assert!(bytes.is_empty());
            assert!(parse_document(n, bytes).unwrap().fields.is_empty());
        }
    }

    /// A two-chunk source plus a mutation applied to its `.fdt`. Every
    /// mutation here replaces a vint with another of the **same length**, so
    /// `open`'s `maxPointer`-vs-file-length cross-check still passes and the
    /// corruption really does have to be caught by `copy_chunks` itself.
    fn corrupted_source(mutate: impl Fn(&mut Vec<u8>, usize)) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        const N: usize = BEST_SPEED_MAX_DOCS_PER_CHUNK * 2;
        let docs = merge_docs("x", N);
        let (mut fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
        let second_chunk = {
            let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
            reader
                .chunk_for_doc(BEST_SPEED_MAX_DOCS_PER_CHUNK as i32)
                .unwrap()
                .start_pointer as usize
        };
        mutate(&mut fdt, second_chunk);
        (fdt, fdx, fdm)
    }

    #[test]
    fn a_chunk_header_whose_doc_base_disagrees_with_the_index_is_rejected() {
        // Java's `if (base != docID) throw new CorruptIndexException(...)`.
        // The `.fdx` index and the `.fdt` chunk headers are redundant on
        // purpose: disagreeing is how a bad bulk copy announces itself before
        // it writes a segment that reads back plausible but wrong documents.
        let (fdt, fdx, fdm) = corrupted_source(|fdt, at| {
            // docBase 1024 (vint 0x80 0x08) -> 1025 (vint 0x81 0x08).
            assert_eq!(&fdt[at..at + 2], &[0x80, 0x08]);
            fdt[at] = 0x81;
        });
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
        assert!(matches!(
            w.copy_chunks(&reader, 0, reader.max_doc()),
            Err(Error::CorruptChunkBounds { .. })
        ));
    }

    #[test]
    fn a_chunk_header_claiming_no_documents_is_rejected() {
        // Not a case Java has to guard (its own writer never emits it), but a
        // zero-document chunk would make the copy loop spin without ever
        // advancing `docID`.
        let (fdt, fdx, fdm) = corrupted_source(|fdt, at| {
            // Skip the docBase vint, then blank the token to a non-canonical
            // two-byte encoding of 0 so the chunk's length is unchanged.
            let token_at = at + 2;
            assert_eq!(&fdt[token_at..token_at + 2], &[0x80, 0x20]);
            fdt[token_at] = 0x80;
            fdt[token_at + 1] = 0x00;
        });
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
        assert!(matches!(
            w.copy_chunks(&reader, 0, reader.max_doc()),
            Err(Error::CorruptChunkBounds { chunk_docs: 0, .. })
        ));
    }

    #[test]
    fn a_chunk_claiming_more_documents_than_the_requested_range_is_rejected() {
        // Java's second `CorruptIndexException`: `docID > toDocID` after a
        // chunk. The first chunk is made to claim 2048 documents instead of
        // 1024, so copying it would run past the end of the requested run.
        const N: usize = BEST_SPEED_MAX_DOCS_PER_CHUNK * 2;
        let docs = merge_docs("y", N);
        let (mut fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
        let first_chunk = {
            let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
            reader.chunk_for_doc(0).unwrap().start_pointer as usize
        };
        // docBase 0 is one byte; the token follows. 1024 docs -> token 4096
        // (vint 0x80 0x20); 2048 docs -> token 8192 (vint 0x80 0x40).
        let token_at = first_chunk + 1;
        assert_eq!(&fdt[token_at..token_at + 2], &[0x80, 0x20]);
        fdt[token_at + 1] = 0x40;

        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
        assert!(matches!(
            w.copy_chunks(&reader, 0, BEST_SPEED_MAX_DOCS_PER_CHUNK as i32),
            Err(Error::CorruptChunkBounds { .. })
        ));
    }

    // ---------------------------------------------------------------------
    // The arithmetic gate (docs/arithmetic-gate.md): every length, count,
    // offset and width this format reads off disk, driven past the range a
    // Lucene writer could ever have produced. Each of these fails against the
    // code as it stood before batch c27 -- by panicking, by reserving memory
    // proportional to the *claim*, or by accepting the file outright.
    // ---------------------------------------------------------------------

    /// Recomputes a codec file's trailing CRC so that only the *semantic*
    /// invariant under test can fire, never the checksum.
    fn resign(file: &mut [u8]) {
        let at = file.len() - 8;
        let checksum = crc32fast::hash(&file[..at]) as u64;
        file[at..].copy_from_slice(&checksum.to_be_bytes());
    }

    /// Byte offset of `maxDoc` in a `.fdm` built by
    /// `build_single_chunk_index`; everything after it is fixed-width until
    /// the three tail vlongs.
    fn fdm_max_doc_offset() -> usize {
        4 + 1 + META_CODEC.len() + 4 + ID_LENGTH + 1 + vint_len_test(80 * 1024)
    }

    /// `maxPointer` sits 120 bytes past `maxDoc`: blockShift (4) +
    /// indexNumChunks (4) + docsStart (8) + two 21-byte monotonic block
    /// headers (42) + docsEnd (8) + another 42 + startPointersEnd (8).
    fn fdm_max_pointer_offset() -> usize {
        fdm_max_doc_offset() + 4 + 4 + 4 + 8 + 42 + 8 + 42 + 8
    }

    #[test]
    fn negative_max_pointer_is_a_decode_error_not_an_overflow() {
        // `maxPointer` is a raw i64 and `open` cross-checks it against the
        // `.fdt` length. `(-1i64) as usize + FOOTER_LENGTH` wraps to 15, so
        // before the fix the check was not merely bypassable, it *panicked*
        // in a debug build before it could reject anything.
        let doc_bytes = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
        let (fdt, fdx, mut fdm) = build_single_chunk_index(&doc_bytes);
        let at = fdm_max_pointer_offset();
        fdm[at..at + 8].copy_from_slice(&(-1i64).to_le_bytes());
        resign(&mut fdm);
        assert!(matches!(
            open(&fdt, &fdx, &fdm, &id(), ""),
            Err(Error::Store(lucene_store::Error::Corrupted(_)))
        ));
    }

    #[test]
    fn num_chunks_at_the_top_of_the_vlong_range_is_a_decode_error_not_an_overflow() {
        // `index_num_chunks != num_chunks + 1` overflowed for a `numChunks`
        // vlong of `i64::MAX`. `numChunks` also feeds
        // `direct_monotonic::floor_index` as a search bound, so it is pinned
        // non-negative here too.
        let doc_bytes = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
        let (fdt, fdx, fdm) =
            build_single_chunk_index_with_meta_overrides(&doc_bytes, i64::MAX, 0, 0);
        assert!(matches!(
            open(&fdt, &fdx, &fdm, &id(), ""),
            Err(Error::NumChunksMismatch { .. })
        ));
    }

    #[test]
    fn negative_max_doc_is_rejected_by_open() {
        // Java takes `numDocs` from `SegmentInfo`; this port takes it from
        // the `.fdm`, and every chunk-bounds check is stated relative to it.
        // A negative one used to be accepted and then made
        // `doc_base + chunk_docs > max_doc` vacuous.
        let doc_bytes = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
        let (fdt, fdx, mut fdm) = build_single_chunk_index(&doc_bytes);
        let at = fdm_max_doc_offset();
        fdm[at..at + 4].copy_from_slice(&(-1i32).to_le_bytes());
        resign(&mut fdm);
        assert!(matches!(
            open(&fdt, &fdx, &fdm, &id(), ""),
            Err(Error::Store(lucene_store::Error::Corrupted(_)))
        ));
    }

    /// A `.fdt` holding one chunk whose header claims `chunk_docs` documents
    /// with all-equal (`bpv == 0`) field-count and length arrays, plus a
    /// matching `.fdx`/`.fdm` with `max_doc` set as high as the caller likes.
    fn single_chunk_with_claimed_counts(
        chunk_docs: i32,
        num_stored_fields: i32,
        length: i32,
        max_doc: i32,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut fdt = Vec::new();
        fdt.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut fdt, DATA_CODEC_BEST_SPEED);
        fdt.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        fdt.extend_from_slice(&id());
        fdt.push(0);
        let chunk_start = fdt.len() as i64;
        write_vint(&mut fdt, 0); // docBase
        write_vint(&mut fdt, chunk_docs << 2); // token
        if chunk_docs == 1 {
            // The single-document header shape: two bare vints, no bulk
            // arrays at all (`Lucene90CompressingStoredFieldsWriter.flush`).
            write_vint(&mut fdt, num_stored_fields);
            write_vint(&mut fdt, length);
        } else {
            fdt.push(0); // numStoredFields: bpv=0 (all equal)
            write_vint(&mut fdt, num_stored_fields);
            fdt.push(0); // lengths: bpv=0 (all equal)
            write_vint(&mut fdt, length);
        }
        fdt.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        fdt.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&fdt) as u64;
        fdt.extend_from_slice(&checksum.to_be_bytes());
        let (fdx, fdm) = build_fdx_fdm_for_single_chunk(&fdt, max_doc, chunk_start);
        (fdt, fdx, fdm)
    }

    #[test]
    fn a_chunk_claiming_more_documents_than_the_format_allows_is_rejected() {
        // `chunkDocs` off a chunk token sizes `read_bulk_ints`' `vec![0i64;
        // count]` and the `offsets` reservation, and the only thing that used
        // to bound it was `maxDoc` -- itself a raw i32 off the `.fdm`. With
        // `maxDoc = i32::MAX` a 60-byte file could ask for a 17 GB `Vec`,
        // which *aborts* rather than unwinding. 2 000 000 is used here so the
        // pre-fix behaviour (a 16 MB reservation, then EOF) is survivable
        // enough to observe; the shape is identical at i32::MAX.
        let (fdt, fdx, fdm) = single_chunk_with_claimed_counts(2_000_000, 1, 1, i32::MAX);
        let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
        assert!(matches!(
            reader.document(0),
            Err(Error::CorruptChunkBounds { .. })
        ));
        // ...and the legal ceiling really is the per-mode `maxDocsPerChunk`:
        // one document under it still parses its header.
        let (fdt, fdx, fdm) = single_chunk_with_claimed_counts(
            BEST_SPEED_MAX_DOCS_PER_CHUNK as i32,
            0,
            0,
            BEST_SPEED_MAX_DOCS_PER_CHUNK as i32,
        );
        let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
        assert!(reader.document(0).unwrap().fields.is_empty());
        let (fdt, fdx, fdm) = single_chunk_with_claimed_counts(
            BEST_SPEED_MAX_DOCS_PER_CHUNK as i32 + 1,
            0,
            0,
            i32::MAX,
        );
        let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
        assert!(matches!(
            reader.document(0),
            Err(Error::CorruptChunkBounds { .. })
        ));
    }

    #[test]
    fn a_negative_document_length_is_a_decode_error_not_a_backwards_offset() {
        // Only the `bpv == 0` bulk-int shape can carry a negative value (the
        // 8/16/32-bit shapes are masked), and it used to run straight into
        // the cumulative `offsets` array. The result was a *decreasing*
        // offsets array, and `serialized_document`'s
        // `offsets[i + 1] as usize - doc_offset` then underflowed to ~2^64 --
        // which `Vec::with_capacity` turns into a capacity-overflow panic.
        let (fdt, fdx, fdm) = single_chunk_with_claimed_counts(2, 1, -1, 2);
        let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
        assert!(matches!(
            reader.document(0),
            Err(Error::Store(lucene_store::Error::Corrupted(_)))
        ));
        // The single-document header shape reads its length as a bare vint
        // rather than through a bulk array, and needed the same guard.
        let (fdt, fdx, fdm) = single_chunk_with_claimed_counts(1, 1, -1, 1);
        let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
        assert!(matches!(
            reader.document(0),
            Err(Error::LengthFieldCountMismatch { length: -1, .. })
        ));
    }

    #[test]
    fn a_chunk_claiming_more_decompressed_bytes_than_could_be_produced_is_rejected() {
        // Nothing on the wire relates a chunk's decompressed size to the
        // compressed bytes that follow it, so the header can name any size it
        // likes -- and that size flows into `decompress_range`'s output `Vec`
        // and `decompress_unit`'s `vec![0u8; dictLength + blockLength]`. The
        // LZ4 block format expands by at most 255x, so a claim past 255 times
        // the bytes left in the file cannot be produced by *any* input.
        let (fdt, fdx, fdm) = single_chunk_with_claimed_counts(1, 1, 10_000_000, 1);
        let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
        let err = reader.document(0).unwrap_err();
        let Error::Store(lucene_store::Error::Corrupted(msg)) = &err else {
            panic!("unexpected error: {err:?}");
        };
        assert!(msg.contains("decompressed bytes"), "{msg}");

        // The whole-chunk path is where the claim is worst: `read_chunk`
        // reserves the chunk's *entire* declared length in one go, and with
        // up to 1024 documents each declaring an `i32::MAX` length that is a
        // ~2 TB request off a hundred-byte file. Checked in
        // `read_chunk_header`, so both paths are covered by the one check.
        let (fdt, fdx, fdm) = single_chunk_with_claimed_counts(1024, 1, i32::MAX, 1024);
        let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
        let err = reader.read_chunk(0).unwrap_err();
        let Error::Store(lucene_store::Error::Corrupted(msg)) = &err else {
            panic!("unexpected error: {err:?}");
        };
        assert!(msg.contains("decompressed bytes"), "{msg}");
    }

    #[test]
    fn an_oversized_binary_field_length_is_a_decode_error_not_an_allocation() {
        // Java hands its visitor a lazy `StoredFieldDataInput`; this port owns
        // a `Vec<u8>`, so the vint length has to be bounded by the bytes
        // actually left. A negative one used to sign-extend to ~2^64.
        let mut payload = Vec::new();
        write_vint(&mut payload, 1_000_000);
        let field = field_bytes(0, TYPE_BYTE_ARR, &payload);
        let (fdt, fdx, fdm) = build_single_chunk_index(&field);
        let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
        assert!(matches!(
            reader.document(0),
            Err(Error::Store(lucene_store::Error::Corrupted(_)))
        ));

        let mut payload = Vec::new();
        write_vint(&mut payload, -1);
        let field = field_bytes(0, TYPE_BYTE_ARR, &payload);
        let (fdt, fdx, fdm) = build_single_chunk_index(&field);
        let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
        assert!(matches!(
            reader.document(0),
            Err(Error::Store(lucene_store::Error::Corrupted(_)))
        ));
    }

    #[test]
    fn an_absurd_stored_field_count_does_not_reserve_a_slot_per_claimed_field() {
        // `numStoredFields` comes off the chunk header's 32-bit bulk int
        // array, read unsigned, so it reaches ~4.3e9 -- and a `StoredField`
        // is 40-odd bytes. Reserving for the claim is a ~170 GB request: the
        // allocator fails, `handle_alloc_error` **aborts**, and no
        // `catch_unwind` at the FFI boundary can intercept that. Reserving
        // for what the bytes could hold costs nothing and errors identically.
        assert!(parse_document(i64::from(u32::MAX), b"").is_err());
        assert!(parse_document(i64::MAX, b"\x00").is_err());
    }

    #[test]
    fn a_negative_sub_block_compressed_length_is_a_decode_error() {
        // The `length == 0` skip path sums every sub-block's compressed
        // length. A negative vint sign-extended to ~2^64 and overflowed the
        // running total, which in release wraps to a *small* skip -- leaving
        // the reader mid-unit, so the next `sliced` unit's framing header is
        // read out of compressed bytes and decodes plausible garbage.
        let mut unit = Vec::new();
        write_vint(&mut unit, 0); // dictLength
        write_vint(&mut unit, 4); // blockLength
        write_vint(&mut unit, -1); // dictionary's compressed length
        write_vint(&mut unit, 1);
        write_vint(&mut unit, 1);
        let mut input = SliceInput::new(&unit);
        let mut out = Vec::new();
        let err = decompress_unit(Mode::BestSpeed, &mut input, 8, 0, 0, &mut out).unwrap_err();
        assert!(
            matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
            "unexpected error: {err:?}"
        );
    }

    /// Flips one bit of one of the three files, re-signs all three, and
    /// tries to open the segment and read every document out of it. Returns
    /// `true` if the mutation was *rejected* (a typed error from `open`, from
    /// a read, or from `check_integrity`) rather than silently accepted. A
    /// panic, an abort or a reservation proportional to a number just read
    /// off disk is a test failure, which is the point.
    fn byte_flip_is_rejected(
        files: &(Vec<u8>, Vec<u8>, Vec<u8>),
        which: usize,
        at: usize,
        bit: u8,
    ) -> bool {
        let (mut fdt, mut fdx, mut fdm) = files.clone();
        {
            let target = match which {
                0 => &mut fdt,
                1 => &mut fdx,
                _ => &mut fdm,
            };
            target[at] ^= bit;
        }
        resign(&mut fdt);
        resign(&mut fdx);
        resign(&mut fdm);
        let Ok(reader) = open(&fdt, &fdx, &fdm, &id_write(), "") else {
            return true;
        };
        let mut rejected = reader.check_integrity().is_err();
        let mut cursor = ChunkCursor::new();
        for doc in 0..reader.max_doc() {
            // The cursor path decompresses each chunk once; the
            // random-access path re-frames the unit per document, so it is
            // sampled at the chunk edges rather than run for every document.
            match cursor.document(&reader, doc) {
                Ok((n, bytes)) => rejected |= parse_document(n, bytes).is_err(),
                Err(_) => rejected = true,
            }
            if doc % 1021 == 0 {
                rejected |= reader.document(doc).is_err();
            }
        }
        rejected
    }

    fn flip_sweep(files: &(Vec<u8>, Vec<u8>, Vec<u8>), which: usize) -> (usize, usize) {
        let len = match which {
            0 => files.0.len(),
            1 => files.1.len(),
            _ => files.2.len(),
        };
        let (mut rejected, mut total) = (0usize, 0usize);
        // The trailing 16-byte footer is skipped: it is recomputed anyway.
        for at in 0..len - codec_util::FOOTER_LENGTH {
            for bit in [0x01u8, 0x80] {
                total += 1;
                if byte_flip_is_rejected(files, which, at, bit) {
                    rejected += 1;
                }
            }
        }
        (rejected, total)
    }

    #[test]
    fn every_single_bit_flip_in_a_re_signed_index_is_handled_without_panicking() {
        // c25's `.tvd` sweep, applied to the stored-fields `.fdx`/`.fdm`
        // pair. The footers are recomputed after every flip so only the
        // *semantic* invariants can fire -- a checksum catching everything
        // would prove nothing about the decoders. Three chunks, so that the
        // monotonic doc-base and start-pointer arrays actually discriminate
        // (with one chunk every lookup lands on chunk 0 whatever they say).
        let docs = merge_docs("flip", BEST_SPEED_MAX_DOCS_PER_CHUNK * 2 + 52);
        let files = write_best_speed(&docs, &id_write(), "");
        let (fdm_rejected, fdm_total) = flip_sweep(&files, 2);
        let (fdx_rejected, fdx_total) = flip_sweep(&files, 1);
        // Recorded as floors rather than exact counts: a handful of `.fdm`
        // bits genuinely do not matter (the codec version, which `open`
        // accepts across its supported range; `chunkSize`, which only a
        // `sliced` chunk consults; the unused `offset` of a `bpv == 0`
        // monotonic block; `numDirtyDocs`, which Java only sanity-checks).
        assert!(
            fdm_rejected * 100 >= fdm_total * 90,
            ".fdm bit flips rejected: {fdm_rejected}/{fdm_total}"
        );
        assert!(
            fdx_rejected * 100 >= fdx_total * 80,
            ".fdx bit flips rejected: {fdx_rejected}/{fdx_total}"
        );
    }

    #[test]
    fn every_single_bit_flip_in_a_re_signed_data_file_is_handled_without_panicking() {
        // The same sweep over the `.fdt` itself -- chunk headers, the
        // preset-dictionary unit framing, and the compressed payload. A small
        // segment, because every byte is swept: the interesting bytes are the
        // header and framing ones, and the payload is deliberately included
        // so the LZ4 decoder is driven with garbage too.
        //
        // The bar here is *not* a rejection rate. A flip inside a document's
        // compressed payload legitimately decodes to a different but
        // perfectly well-formed document, and Lucene has no way to notice
        // (that is what the checksum this test deliberately repairs is for).
        // The bar is that nothing panics, aborts, or reserves memory
        // proportional to a length it just read.
        let docs = merge_docs("flip", 40);
        let files = write_best_speed(&docs, &id_write(), "");
        let (rejected, total) = flip_sweep(&files, 0);
        // 624/904 at the time of writing; the floor is deliberately far
        // below that, since the exact number tracks the compressor.
        assert!(
            rejected * 100 >= total * 40,
            ".fdt bit flips rejected: {rejected}/{total}"
        );
    }
}
