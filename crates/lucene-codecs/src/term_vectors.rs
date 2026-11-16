//! Port of `org.apache.lucene.codecs.lucene90.Lucene90TermVectorsFormat`
//! (`.tvd` data + `.tvx` index + `.tvm` meta), read and write.
//!
//! Structurally almost identical to [`crate::stored_fields`]: documents are
//! grouped into chunks (never spanning a chunk boundary), indexed the same
//! way via a [`crate::direct_monotonic`]-addressed chunk lookup. The
//! differences are what's inside a chunk and how it's compressed:
//! - Per-doc/per-field bookkeeping (field counts, distinct field numbers,
//!   per-field flags, term counts, term prefix/suffix lengths, term
//!   frequencies, positions, offsets, payload lengths) is packed as
//!   several independent [`crate::block_packed`] streams (each a series
//!   of 64-value blocks) rather than stored fields' single length-prefixed
//!   arrays.
//! - Term and payload *bytes* are LZ4-compressed with `CompressionMode.FAST`,
//!   which -- unlike stored fields' `LZ4WithPresetDictCompressionMode` -- is
//!   a single plain LZ4 unit with no dictionary or sub-blocking at all, so
//!   it's just one [`crate::lz4::decompress`] call per chunk.
//!
//! This port decodes an entire chunk in one pass ([`DecodedChunk`]) rather
//! than replicating Java's skip-arithmetic, which materializes only the
//! requested document's slice of each array and decompresses only that
//! document's stretch of the LZ4 unit. The trade is bounded by the chunk --
//! 4 096 bytes / 128 documents, [`CHUNK_SIZE`]/[`MAX_DOCS_PER_CHUNK`] -- and
//! it is repaid on any sequential walk, because [`ChunkCursor`] keeps the
//! decoded chunk and Java's reader keeps nothing: a merge or a `CheckIndex`
//! pass costs one decode per chunk here and one per *document* there.
//!
//! [`TermVectorsWriter`] is `Lucene90CompressingTermVectorsWriter`: it
//! buffers documents and flushes a chunk on either of Java's two triggers,
//! writes the per-chunk `numFields`/`fieldNums`/`fieldFlags`/`numTerms`
//! bit-packed headers, and carries `numDirtyChunks`/`numDirtyDocs` so a
//! later merge can decide between copying a source's compressed chunks
//! verbatim ([`TermVectorsWriter::copy_chunks`]) and re-encoding them.
//!
//! Two more bit-packing conventions are involved beyond
//! [`crate::direct_reader`] (used here for the per-chunk distinct-field-number
//! offsets and per-field flags arrays): [`crate::packed_ints`] (the generic
//! MSB-first bitstream backing the distinct-field-numbers array itself) and
//! [`crate::block_packed`] (built on top of `packed_ints`).
//!
//! **Positions/offsets delta semantics** (reverse-engineered from
//! `Lucene90CompressingTermVectorsReader`'s exact loop bounds, not just its
//! doc comment -- the comment reads as "one continuous delta chain across
//! the whole field" but the code's cumulative-sum loops deliberately skip
//! index `positionIndex[j]`, each term's first occurrence, for every term
//! after the first): each **term**'s occurrences form their own delta
//! chain, resetting at that term's first occurrence, not one chain spanning
//! the whole field. A term's first occurrence stores an absolute position
//! and an offset delta needing no further addition; later occurrences of
//! that same term delta-decode against the previous occurrence *of that
//! term*. Confirmed against a real fixture with multi-term, multi-occurrence
//! fields (see `tests/term_vectors_fixtures.rs`).

use std::sync::Arc;

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

use crate::block_packed;
use crate::compression::{self, CompressingFormat, Compression, Compressor};
use crate::direct_monotonic;
use crate::direct_reader;
use crate::packed_ints;

const DATA_CODEC: &str = "Lucene90TermVectorsData";
const META_CODEC: &str = "Lucene90TermVectorsIndexMeta";
const INDEX_CODEC: &str = "Lucene90TermVectorsIndexIdx";
const VERSION_START: i32 = 0;
const VERSION_CURRENT: i32 = 0;
const META_VERSION_START: i32 = 0;
const INDEX_VERSION_START: i32 = 0;
const INDEX_VERSION_CURRENT: i32 = 0;

const FLAG_POSITIONS: u8 = 0x01;
const FLAG_OFFSETS: u8 = 0x02;
const FLAG_PAYLOADS: u8 = 0x04;
const FLAGS_BITS: u8 = 4; // direct_writer_bits_required(POSITIONS|OFFSETS|PAYLOADS = 7)

/// `DirectWriter`'s supported bit widths -- `bitsRequired` always rounds up
/// to one of these (see `DirectWriter.roundBits`); term vectors relies on
/// this rounding for the distinct-field-number-offsets array width.
const DIRECT_WRITER_SUPPORTED_BITS: [u32; 14] =
    [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64];

fn direct_writer_bits_required(max_value: i64) -> u8 {
    // ARITH: the `else` arm runs only for `max_value >= 1`, whose `u64`
    // widening has at most 63 leading zeros, so `64 - leading_zeros()` is in
    // `1..=64` and never underflows.
    #[allow(clippy::arithmetic_side_effects)]
    let bits = if max_value <= 0 {
        1
    } else {
        64 - (max_value as u64).leading_zeros()
    };
    DIRECT_WRITER_SUPPORTED_BITS
        .into_iter()
        .find(|&w| w >= bits)
        .unwrap_or(64) as u8
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("doc {0} is out of range (maxDoc={1})")]
    DocOutOfRange(i32, i32),
    #[error(
        "corrupted chunk: docBase={doc_base}, chunkDocs={chunk_docs}, doc={doc}, maxDoc={max_doc}"
    )]
    CorruptChunkBounds {
        doc_base: i32,
        chunk_docs: i32,
        doc: i32,
        max_doc: i32,
    },
    #[error("index meta's numChunks ({index_num_chunks}) should be exactly one more than the outer meta's ({outer_num_chunks})")]
    NumChunksMismatch {
        index_num_chunks: i64,
        outer_num_chunks: i64,
    },
    #[error("more dirty chunks ({0}) than chunks ({1})")]
    TooManyDirtyChunks(i64, i64),
    #[error("dirty chunks ({0}) and dirty docs ({1}) must both be zero or both nonzero")]
    DirtyChunksDocsMismatch(i64, i64),
    #[error("more dirty chunks ({0}) than documents within dirty chunks ({1})")]
    MoreDirtyChunksThanDirtyDocs(i64, i64),
    #[error("invalid flags-array selector: {0}")]
    InvalidFlagsSelector(i32),
    #[error("bulk chunk copy not permitted: reader chunkSize={reader_chunk_size}, writer chunkSize={writer_chunk_size}, tooDirty={too_dirty}")]
    BulkCopyNotPermitted {
        reader_chunk_size: i32,
        writer_chunk_size: i32,
        too_dirty: bool,
    },
    #[error("inverted doc range: from={from_doc}, to={to_doc}")]
    InvertedDocRange { from_doc: i32, to_doc: i32 },
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq)]
pub struct TermVectorTerm {
    pub term: Vec<u8>,
    pub freq: i32,
    /// One entry per occurrence; present only when the field has POSITIONS.
    pub positions: Option<Vec<i32>>,
    /// One entry per occurrence; present only when the field has OFFSETS.
    pub start_offsets: Option<Vec<i32>>,
    pub end_offsets: Option<Vec<i32>>,
    /// One entry per occurrence (possibly empty); present only when the
    /// field has PAYLOADS.
    pub payloads: Option<Vec<Vec<u8>>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TermVectorField {
    pub field_number: i32,
    pub has_positions: bool,
    pub has_offsets: bool,
    pub has_payloads: bool,
    pub terms: Vec<TermVectorTerm>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TermVectorsDocument {
    pub fields: Vec<TermVectorField>,
}

pub struct TermVectorsReader<'d> {
    tvd: &'d [u8],
    tvx: &'d [u8],
    /// The `.tvd` codec name `open` matched, and the compression it names.
    data_codec: &'static str,
    compression: Compression,
    max_doc: i32,
    chunk_size: i32,
    max_pointer: i64,
    num_chunks: i64,
    num_dirty_chunks: i64,
    num_dirty_docs: i64,
    docs_start_pointer: i64,
    docs_end_pointer: i64,
    docs_meta: Arc<direct_monotonic::Meta>,
    start_pointers_start_pointer: i64,
    start_pointers_end_pointer: i64,
    start_pointers_meta: Arc<direct_monotonic::Meta>,
}

/// Parses `.tvd`+`.tvm`+`.tvx` (already read into memory) and returns a
/// reader over `tvd`/`tvx`'s bytes.
pub fn open<'d>(
    tvd: &'d [u8],
    tvx: &'d [u8],
    tvm: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<TermVectorsReader<'d>> {
    let mut tvd_input = SliceInput::new(tvd);
    // `Lucene90TermVectorsFormat` writes `FAST`; the test framework's
    // `CompressingCodec`s write their own names and modes. Peek the name
    // before the real header check, the way stored fields do.
    let header_start = tvd_input.position();
    let peek_magic = tvd_input.read_be_u32()?;
    if peek_magic != codec_util::CODEC_MAGIC {
        return Err(lucene_store::Error::Corrupted(format!(
            "codec header mismatch: actual header={peek_magic:#x} vs expected header={:#x}",
            codec_util::CODEC_MAGIC
        ))
        .into());
    }
    let name = tvd_input.read_string()?;
    let (data_codec, compression) = if name == DATA_CODEC {
        (DATA_CODEC, Compression::Fast)
    } else {
        *compression::COMPRESSING_DATA_CODECS
            .iter()
            .find(|(codec, _)| *codec == name)
            .ok_or_else(|| {
                lucene_store::Error::Corrupted(format!("unknown term vectors data codec: {name}"))
            })?
    };
    tvd_input.seek(header_start)?;
    let tvd_header = codec_util::check_index_header(
        &mut tvd_input,
        data_codec,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    codec_util::retrieve_checksum(tvd)?;

    let mut meta_input = SliceInput::new(tvm);
    codec_util::check_index_header(
        &mut meta_input,
        META_CODEC,
        META_VERSION_START,
        tvd_header.version,
        segment_id,
        segment_suffix,
    )?;
    let _packed_ints_version = meta_input.read_vint()?;
    let chunk_size = meta_input.read_vint()?;

    let max_doc = meta_input.read_i32()?;
    let block_shift = meta_input.read_i32()? as u32;
    let index_num_chunks = meta_input.read_i32()? as i64;
    let docs_start_pointer = meta_input.read_i64()?;
    let docs_meta = direct_monotonic::load_meta(&mut meta_input, index_num_chunks, block_shift)?;
    let docs_end_pointer = meta_input.read_i64()?;
    let start_pointers_start_pointer = docs_end_pointer;
    let start_pointers_meta =
        direct_monotonic::load_meta(&mut meta_input, index_num_chunks, block_shift)?;
    let start_pointers_end_pointer = meta_input.read_i64()?;
    let max_pointer = meta_input.read_i64()?;

    // `numChunks + 1` is a `long` add in Java, which wraps; here it would
    // panic in a debug build for a `.tvm` whose vlong is near `i64::MAX`, and
    // in a release build wrap to `i64::MIN` -- which `index_num_chunks`, an
    // `i32` widened to `i64`, can never equal, so the *conclusion* survives
    // the wrap but the debug build does not. `checked_add` makes both builds
    // report the same corruption.
    let num_chunks = meta_input.read_vlong()?;
    if num_chunks.checked_add(1) != Some(index_num_chunks) {
        return Err(Error::NumChunksMismatch {
            index_num_chunks,
            outer_num_chunks: num_chunks,
        });
    }
    let num_dirty_chunks = meta_input.read_vlong()?;
    let num_dirty_docs = meta_input.read_vlong()?;
    // Java asserts `numChunks >= 0`, `numDirtyChunks >= 0` and
    // `numDirtyDocs >= 0` in its three getters. Assertions are not enough
    // here: `too_dirty` multiplies `numDirtyChunks` by 100, which underflows
    // for a sufficiently negative vlong, and a negative count slips through
    // all four cross-checks below (a negative `numDirtyChunks` is "nonzero",
    // is not greater than `numChunks`, and is not greater than a positive
    // `numDirtyDocs`). Made real checks, which is what turns `too_dirty`'s
    // `* 100` into provably-bounded arithmetic.
    if num_chunks < 0 || num_dirty_chunks < 0 || num_dirty_docs < 0 {
        return Err(lucene_store::Error::Corrupted(format!(
            "negative chunk counts in .tvm: numChunks={num_chunks}, \
             numDirtyChunks={num_dirty_chunks}, numDirtyDocs={num_dirty_docs}"
        ))
        .into());
    }
    if num_chunks < num_dirty_chunks {
        return Err(Error::TooManyDirtyChunks(num_dirty_chunks, num_chunks));
    }
    if (num_dirty_chunks == 0) != (num_dirty_docs == 0) {
        return Err(Error::DirtyChunksDocsMismatch(
            num_dirty_chunks,
            num_dirty_docs,
        ));
    }
    // `Lucene90CompressingTermVectorsReader`'s third dirty-chunk check:
    // "Cannot have more dirty chunks than documents within dirty chunks".
    // Every dirty chunk contributes at least one doc to `numDirtyDocs`
    // (`flush(force=true)` does `numDirtyDocs += pendingDocs.size()` with a
    // non-empty `pendingDocs`), so `numDirtyDocs < numDirtyChunks` can only
    // mean corruption.
    if num_dirty_docs < num_dirty_chunks {
        return Err(Error::MoreDirtyChunksThanDirtyDocs(
            num_dirty_chunks,
            num_dirty_docs,
        ));
    }
    codec_util::check_footer(&mut meta_input, tvm.len())?;

    let mut tvx_input = SliceInput::new(tvx);
    codec_util::check_index_header(
        &mut tvx_input,
        INDEX_CODEC,
        INDEX_VERSION_START,
        INDEX_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    codec_util::retrieve_checksum(tvx)?;

    // `maxPointer` is a raw `i64` off `.tvm`. `as usize` sign-extends a
    // negative one into a huge length and `+ FOOTER_LENGTH` then overflows
    // (a debug-build panic; in release it wraps back down and the equality
    // below decides on a number nobody computed). Fold the widening and the
    // add into one fallible expression, and let the length equality that
    // follows carry the *upper* bound: from here on `max_pointer` is known to
    // be in `0..=tvd.len()`, which is what makes every later
    // `max_pointer as usize` -- `check_integrity`'s and `copy_chunks`' --
    // an in-bounds offset by construction rather than by hope.
    let expected_tvd_len = usize::try_from(max_pointer)
        .ok()
        .and_then(|p| p.checked_add(codec_util::FOOTER_LENGTH));
    if expected_tvd_len != Some(tvd.len()) {
        return Err(lucene_store::Error::Corrupted(format!(
            ".tvd length should be maxPointer={max_pointer} + {} footer bytes, but is {}",
            codec_util::FOOTER_LENGTH,
            tvd.len()
        ))
        .into());
    }

    Ok(TermVectorsReader {
        tvd,
        tvx,
        data_codec,
        compression,
        max_doc,
        chunk_size,
        max_pointer,
        num_chunks,
        num_dirty_chunks,
        num_dirty_docs,
        docs_start_pointer,
        docs_end_pointer,
        docs_meta: Arc::new(docs_meta),
        start_pointers_start_pointer,
        start_pointers_end_pointer,
        start_pointers_meta: Arc::new(start_pointers_meta),
    })
}

impl<'d> TermVectorsReader<'d> {
    /// This reader's parsed metadata over `tvd`/`tvx` instead of the bytes it
    /// was opened on, as [`crate::stored_fields::StoredFieldsReader::rebind`]
    /// does for stored fields: a caller holding the files parses `.tvm` once
    /// rather than per document. `open` validated the metadata against the
    /// files it was given, so `tvd` and `tvx` must be those same files' bytes
    /// (reading through other bytes fails or answers garbage, never undefined
    /// behaviour: every read is bounds-checked).
    pub fn rebind<'e>(&self, tvd: &'e [u8], tvx: &'e [u8]) -> TermVectorsReader<'e> {
        TermVectorsReader {
            tvd,
            tvx,
            data_codec: self.data_codec,
            compression: self.compression,
            max_doc: self.max_doc,
            chunk_size: self.chunk_size,
            max_pointer: self.max_pointer,
            num_chunks: self.num_chunks,
            num_dirty_chunks: self.num_dirty_chunks,
            num_dirty_docs: self.num_dirty_docs,
            docs_start_pointer: self.docs_start_pointer,
            docs_end_pointer: self.docs_end_pointer,
            docs_meta: Arc::clone(&self.docs_meta),
            start_pointers_start_pointer: self.start_pointers_start_pointer,
            start_pointers_end_pointer: self.start_pointers_end_pointer,
            start_pointers_meta: Arc::clone(&self.start_pointers_meta),
        }
    }

    pub fn max_doc(&self) -> i32 {
        self.max_doc
    }

    /// The `.tvd` codec name: a bulk chunk copy is only legal into a writer
    /// of the same format (Java compares the compression modes).
    pub fn data_codec(&self) -> &'static str {
        self.data_codec
    }

    /// The compression the `.tvd` chunks were written with.
    pub fn compression(&self) -> Compression {
        self.compression
    }

    fn docs_region(&self) -> Result<&'d [u8]> {
        self.tvx
            .get(self.docs_start_pointer as usize..self.docs_end_pointer as usize)
            .ok_or(lucene_store::Error::Eof { offset: 0 }.into())
    }

    fn start_pointers_region(&self) -> Result<&'d [u8]> {
        self.tvx
            .get(
                self.start_pointers_start_pointer as usize
                    ..self.start_pointers_end_pointer as usize,
            )
            .ok_or(lucene_store::Error::Eof { offset: 0 }.into())
    }

    /// `.tvm`'s `chunkSize`: how many buffered term/payload bytes make the
    /// writer close a chunk. A bulk chunk copy is only legal between a
    /// reader and a writer that agree on it.
    pub fn chunk_size(&self) -> i32 {
        self.chunk_size
    }

    /// `getNumChunks()`.
    pub fn num_chunks(&self) -> i64 {
        self.num_chunks
    }

    /// `getNumDirtyChunks()` -- chunks that were force-flushed before either
    /// trigger fired.
    pub fn num_dirty_chunks(&self) -> i64 {
        self.num_dirty_chunks
    }

    /// `getNumDirtyDocs()` -- documents living inside those chunks.
    pub fn num_dirty_docs(&self) -> i64 {
        self.num_dirty_docs
    }

    /// `getMaxPointer()`: one past the last chunk's last byte, i.e. where
    /// `.tvd`'s footer starts.
    pub fn max_pointer(&self) -> i64 {
        self.max_pointer
    }

    /// The raw `.tvd` bytes, for [`TermVectorsWriter::copy_chunks`].
    pub fn tvd(&self) -> &'d [u8] {
        self.tvd
    }

    /// Java's `checkIntegrity()` (`CodecUtil.checksumEntireFile`) over
    /// `.tvd`.
    ///
    /// [`open`] deliberately only calls `retrieve_checksum`, which validates
    /// the footer's *shape* and not the CRC -- the right trade for a
    /// random-access reader. It is not enough before a byte-copy merge:
    /// [`TermVectorsWriter::copy_chunks`] copies a source's compressed bytes
    /// verbatim and then writes a freshly computed, valid footer over them,
    /// so a bit flip in the source would be laundered into a merged segment
    /// that passes every checksum from then on. Java runs this on every
    /// source before it picks a merge strategy, and so must every caller
    /// here.
    pub fn check_integrity(&self) -> Result<()> {
        codec_util::check_whole_file_footer(self.tvd, self.max_pointer as usize)?;
        Ok(())
    }

    /// Which chunk holds `doc_id`: its first document id and its `.tvd`
    /// offset. `Lucene90CompressingTermVectorsReader.getIndexReader()`'s
    /// `getStartPointer(docID)` plus the chunk's `docBase`.
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
        let start_pointer = direct_monotonic::get(
            self.start_pointers_region()?,
            &self.start_pointers_meta,
            block_index,
        )?;
        Ok(ChunkIndexEntry {
            doc_base: doc_base as i32,
            start_pointer,
        })
    }

    /// Reads the given document's term vectors, or `None` if it has none.
    ///
    /// Decodes the whole chunk `doc_id` lives in and materialises only that
    /// document's fields. A caller walking a run of documents should go
    /// through a [`ChunkCursor`] instead, which keeps the decoded chunk and
    /// so pays one decode per *chunk* rather than one per document.
    pub fn document(&self, doc_id: i32) -> Result<Option<TermVectorsDocument>> {
        self.read_chunk(doc_id)?.document(doc_id)
    }

    /// Decodes the entire chunk holding `doc_id`: every packed metadata
    /// array and the whole LZ4 unit, shared by all of the chunk's documents.
    pub fn read_chunk(&self, doc_id: i32) -> Result<DecodedChunk> {
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
        let block_start = direct_monotonic::get(
            self.start_pointers_region()?,
            &self.start_pointers_meta,
            block_index,
        )?;

        let mut input = SliceInput::new(self.tvd);
        input.seek(block_start as usize)?;
        let doc_base = input.read_vint()?;
        let token = input.read_vint()?;
        // `Lucene90CompressingTermVectorsReader:380` is
        // `vectorsStream.readVInt() >>> 1` -- an **unsigned** shift, ported
        // here as `>>` until c27. The difference only shows for a negative
        // vint, i.e. only on a corrupt `.tvd`, but it is the difference
        // between `chunk_docs` being an unconstrained negative and being a
        // number as large as `i32::MAX >> 0`, which is what makes
        // `docBase + chunkDocs` genuinely overflow. Java's is an `int` add
        // that wraps; here it panicked in a debug build, and in a release
        // build the wrap carried a corrupt pair *past* all three guards --
        // `doc_base = 1` with `chunk_docs = 0x7FFF_FFFF` wraps to a small
        // negative, which is `<= max_doc`. `checked_add` makes the guard say
        // what it means, and establishes `chunk_docs` in `0..=max_doc` for
        // everything below.
        let chunk_docs = (token as u32 >> 1) as i32;
        // (No `chunk_docs < 0` test: the unsigned shift above makes it
        // non-negative by construction.)
        let chunk_end = doc_base.checked_add(chunk_docs);
        if doc_id < doc_base || chunk_end.is_none_or(|end| doc_id >= end || end > self.max_doc) {
            return Err(Error::CorruptChunkBounds {
                doc_base,
                chunk_docs,
                doc: doc_id,
                max_doc: self.max_doc,
            });
        }

        let num_fields_per_doc: Vec<i64> = if chunk_docs == 1 {
            vec![input.read_vint()? as i64]
        } else {
            block_packed::decode_all(&mut input, chunk_docs as i64)?
        };

        // `num_fields_per_doc` holds `block_packed`-decoded `i64`s, which are
        // whatever the chunk body says. A negative one turns `total_fields`
        // into a huge `usize` through the `as` cast below, and that number
        // then sizes six `Vec::with_capacity` calls -- an **abort**, which no
        // `catch_unwind` at the FFI boundary can intercept. The sum can also
        // overflow `i64` outright. Both are rejected here, once, outside
        // every loop that consumes the result.
        //
        // The reservation is sized from the vector that was actually decoded
        // rather than from `chunk_docs`, so a corrupt count cannot reserve
        // for documents the input never carried.
        let mut field_offsets = Vec::with_capacity(num_fields_per_doc.len().saturating_add(1));
        field_offsets.push(0i64);
        for &n in &num_fields_per_doc {
            let next = if n < 0 {
                None
            } else {
                field_offsets.last().unwrap().checked_add(n)
            };
            field_offsets.push(next.ok_or_else(|| {
                lucene_store::Error::Corrupted(format!(
                    "term vector chunk has a bad per-document field count: {n}"
                ))
            })?);
        }
        let total_fields_i64 = *field_offsets.last().unwrap();
        // Every field in the chunk costs at least one bit in the
        // `allFieldNumOffs` array that follows (`direct_writer_bits_required`
        // never returns 0 -- see its `max_value <= 0 => 1` floor), and that
        // array, plus every other array this chunk has left, lies inside the
        // bytes still ahead of the cursor. So the input itself is a hard
        // ceiling on `total_fields`, and a well-formed chunk loses nothing:
        // a 4 KB chunk admits 32 768 fields against Java's per-chunk maximum
        // of 128 documents x their field counts.
        let field_ceiling = (input.remaining() as u64).saturating_mul(8);
        if total_fields_i64 as u64 > field_ceiling {
            return Err(lucene_store::Error::Corrupted(format!(
                "term vector chunk claims {total_fields_i64} fields, more than the \
                 {} bytes left in it can address",
                input.remaining()
            ))
            .into());
        }
        let total_fields = total_fields_i64 as usize;

        if total_fields == 0 {
            return Ok(DecodedChunk {
                doc_base,
                chunk_docs,
                field_offsets,
                ..DecodedChunk::default()
            });
        }

        // Distinct field numbers in this chunk: a headerless MSB-packed
        // array (see `packed_ints`), not `direct_reader`/`block_packed`.
        let token = input.read_byte()? as u32;
        let bits_per_field_num = token & 0x1F;
        // The count arrives as a 3-bit inline field with a vint escape.
        // Accumulating into a `u32` would turn a negative escape (`read_vint`
        // returns `i32`) into ~4 billion through `as u32`, overflow or wrap
        // the `+=`, and let `packed_ints::byte_count` size `vec![0u8; n]` at up
        // to ~16 GB -- the **abort** shape, which `catch_unwind` at the FFI
        // boundary cannot intercept.
        //
        // `i64` throughout so nothing can wrap, then bounded by two
        // independent facts: the writer's own invariant (`flush_field_nums`
        // emits the *deduplicated* field numbers of the chunk's fields, so
        // there are never more distinct numbers than field instances), and
        // the bytes the packed array must occupy in the stream.
        let mut total_distinct_fields = i64::from(token >> 5);
        if total_distinct_fields == 0x07 {
            let extra = i64::from(input.read_vint()?);
            if extra < 0 {
                return Err(lucene_store::Error::Corrupted(format!(
                    "negative distinct-field-number count extension: {extra}"
                ))
                .into());
            }
            // ARITH: `total_distinct_fields` is 7 here and `extra` is a
            // non-negative `i32` widened to `i64`, so the sum is at most
            // `7 + i32::MAX`; the `+ 1` that follows keeps it under 2^31 + 8.
            #[allow(clippy::arithmetic_side_effects)]
            {
                total_distinct_fields += extra;
            }
        }
        // ARITH: bounded by `7 + i32::MAX + 1` per the proof above.
        #[allow(clippy::arithmetic_side_effects)]
        {
            total_distinct_fields += 1;
        }
        if total_distinct_fields > total_fields_i64 {
            return Err(lucene_store::Error::Corrupted(format!(
                "term vector chunk claims {total_distinct_fields} distinct field numbers \
                 across only {total_fields_i64} fields"
            ))
            .into());
        }
        let field_nums_byte_len =
            packed_ints::byte_count(total_distinct_fields as u64, bits_per_field_num);
        if field_nums_byte_len > input.remaining() {
            return Err(lucene_store::Error::Eof {
                offset: input.position(),
            }
            .into());
        }
        let mut field_nums_bytes = vec![0u8; field_nums_byte_len];
        input.read_bytes(&mut field_nums_bytes)?;
        let mut field_nums = Vec::with_capacity(total_distinct_fields as usize);
        for i in 0..total_distinct_fields {
            field_nums.push(packed_ints::get(&field_nums_bytes, bits_per_field_num, i)?);
        }

        // Field-number offsets (index into `field_nums`) for every field in
        // the chunk, plus per-field flags -- both `direct_reader`-encoded.
        // ARITH: `total_distinct_fields >= 1` -- it is `(token >> 5) + 1` plus
        // a non-negative escape, and the escape's sign is rejected above.
        #[allow(clippy::arithmetic_side_effects)]
        let bits_per_off = direct_writer_bits_required(total_distinct_fields - 1);
        let all_field_num_offs_bytes = read_length_prefixed_slice(&mut input)?.to_vec();
        let flags_selector = input.read_vint()?;
        let all_flags: Vec<u8> = match flags_selector {
            0 => {
                let field_flags_bytes = read_length_prefixed_slice(&mut input)?.to_vec();
                let mut per_field_num_flags = Vec::with_capacity(total_distinct_fields as usize);
                for i in 0..total_distinct_fields {
                    per_field_num_flags
                        .push(direct_reader::get(&field_flags_bytes, FLAGS_BITS, i)? as u8);
                }
                let mut out = Vec::with_capacity(total_fields);
                for i in 0..total_fields as i64 {
                    // Java only *asserts* `fieldNumOff < fieldNums.length`
                    // here, so a corrupt `allFieldNumOffs` entry is an
                    // `ArrayIndexOutOfBoundsException` in a production JVM and
                    // was an index panic here. A typed error instead: this is
                    // reachable from any `.tvd` byte flip.
                    let off = direct_reader::get(&all_field_num_offs_bytes, bits_per_off, i)?;
                    let flags = usize::try_from(off)
                        .ok()
                        .and_then(|off| per_field_num_flags.get(off))
                        .ok_or_else(|| {
                            lucene_store::Error::Corrupted(format!(
                                "term vector fieldNumOff {off} is outside the chunk's \
                                 {total_distinct_fields} distinct field numbers"
                            ))
                        })?;
                    out.push(*flags);
                }
                out
            }
            1 => {
                let flags_bytes = read_length_prefixed_slice(&mut input)?.to_vec();
                let mut out = Vec::with_capacity(total_fields);
                for i in 0..total_fields as i64 {
                    out.push(direct_reader::get(&flags_bytes, FLAGS_BITS, i)? as u8);
                }
                out
            }
            other => return Err(Error::InvalidFlagsSelector(other)),
        };
        // Validated here rather than at every use: `document()` indexes both
        // `field_nums` and `chars_per_term` (one entry per distinct field
        // number) by these, and doing it once per chunk keeps the per-document
        // path free of the check.
        let mut field_num_offs = Vec::with_capacity(total_fields);
        for i in 0..total_fields as i64 {
            let off = direct_reader::get(&all_field_num_offs_bytes, bits_per_off, i)?;
            if off < 0 || off >= total_distinct_fields {
                return Err(lucene_store::Error::Corrupted(format!(
                    "term vector fieldNumOff {off} is outside the chunk's \
                     {total_distinct_fields} distinct field numbers"
                ))
                .into());
            }
            field_num_offs.push(off);
        }

        // Term counts per field, `direct_reader`-encoded. `num_terms_bits` can
        // name up to 64 bits, so an entry is an arbitrary `i64`: a negative one
        // made `term_offsets` non-monotonic (and its `as usize` casts
        // astronomic), and the running sum could overflow outright. Both are
        // rejected in the accumulation loop, which is also where the running
        // sum is built -- so `total_terms` costs one pass, not two.
        let num_terms_bits = input.read_vint()? as u8;
        let num_terms_bytes = read_length_prefixed_slice(&mut input)?.to_vec();
        let mut num_terms = Vec::with_capacity(total_fields);
        let mut term_offsets = Vec::with_capacity(total_fields.saturating_add(1));
        term_offsets.push(0i64);
        for i in 0..total_fields as i64 {
            let n = direct_reader::get(&num_terms_bytes, num_terms_bits, i)?;
            let next = if n < 0 {
                None
            } else {
                term_offsets.last().unwrap().checked_add(n)
            };
            term_offsets.push(next.ok_or_else(|| {
                lucene_store::Error::Corrupted(format!(
                    "term vector chunk has a bad per-field term count: {n}"
                ))
            })?);
            num_terms.push(n);
        }
        let total_terms: i64 = *term_offsets.last().unwrap();

        let prefix_lengths = block_packed::decode_all(&mut input, total_terms)?;
        let suffix_lengths = block_packed::decode_all(&mut input, total_terms)?;
        let term_freqs_minus1 = block_packed::decode_all(&mut input, total_terms)?;

        // Every one of the three `decode_all`s above returned exactly
        // `total_terms` values, which is what makes the `[start..end]` slices
        // below in range. A `freq` is `term_freqs_minus1 + 1` and is a count of
        // occurrences: a negative one would make the per-field sums
        // non-monotonic and `total_positions` unusable as a stream length, and
        // the `+ 1` itself overflows at `i64::MAX`. Checked once per term here,
        // outside every consumer.
        let mut total_positions = 0i64;
        let mut total_offsets = 0i64;
        let mut total_payloads = 0i64;
        let mut field_freq_sums: Vec<i64> = Vec::with_capacity(total_fields);
        for (&f, w) in all_flags.iter().zip(term_offsets.windows(2)) {
            let field_freq_sum = sum_freqs(&term_freqs_minus1[w[0] as usize..w[1] as usize])?;
            field_freq_sums.push(field_freq_sum);
            if f & FLAG_POSITIONS != 0 {
                total_positions = checked_stream_len(total_positions, field_freq_sum)?;
            }
            if f & FLAG_OFFSETS != 0 {
                total_offsets = checked_stream_len(total_offsets, field_freq_sum)?;
            }
            if f & FLAG_PAYLOADS != 0 {
                total_payloads = checked_stream_len(total_payloads, field_freq_sum)?;
            }
        }

        let positions_flat = if total_positions > 0 {
            block_packed::decode_all(&mut input, total_positions)?
        } else {
            Vec::new()
        };
        let (start_offsets_flat, lengths_flat, chars_per_term) = if total_offsets > 0 {
            let mut chars_per_term = Vec::with_capacity(field_nums.len());
            for _ in 0..field_nums.len() {
                chars_per_term.push(f32::from_bits(input.read_i32()? as u32));
            }
            let start_offsets_flat = block_packed::decode_all(&mut input, total_offsets)?;
            let lengths_flat = block_packed::decode_all(&mut input, total_offsets)?;
            (start_offsets_flat, lengths_flat, chars_per_term)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let payload_lengths_flat = if total_payloads > 0 {
            block_packed::decode_all(&mut input, total_payloads)?
        } else {
            Vec::new()
        };

        // Per-field running start indices into the flat positions/offsets/
        // payload-lengths arrays, and each field's total occurrence count --
        // these streams are simple global per-field sequences (unaffected by
        // the per-document byte-buffer grouping below).
        //
        // `field_freq_sums` was computed above alongside the per-stream
        // totals; reusing it here drops a second pass over
        // `term_freqs_minus1`. Each running offset is a partial sum of the
        // same per-field counts the corresponding `total_*` accumulated, and
        // those were `checked_add`ed into an `i64`, so every partial sum is
        // non-negative and no larger than its total -- which is what makes the
        // `+=` below provably in range.
        let mut position_starts = Vec::with_capacity(total_fields);
        let mut offset_starts = Vec::with_capacity(total_fields);
        let mut payload_starts = Vec::with_capacity(total_fields);
        {
            let mut position_off = 0usize;
            let mut offset_off = 0usize;
            let mut payload_off = 0usize;
            for (&flags, &field_freq_sum) in all_flags.iter().zip(&field_freq_sums) {
                position_starts.push(position_off);
                offset_starts.push(offset_off);
                payload_starts.push(payload_off);
                // ARITH: `field_freq_sum >= 0` (`sum_freqs` rejects a negative
                // frequency), and each running offset accumulates exactly the
                // terms its `total_*` did -- the same slice, under the same
                // flag test. `block_packed::decode_all` then materialised
                // `total_*` values into the matching flat vector, so each
                // running offset is bounded by a `Vec` length that already
                // exists, i.e. by `isize::MAX`.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    let step = field_freq_sum as usize;
                    if flags & FLAG_POSITIONS != 0 {
                        position_off += step;
                    }
                    if flags & FLAG_OFFSETS != 0 {
                        offset_off += step;
                    }
                    if flags & FLAG_PAYLOADS != 0 {
                        payload_off += step;
                    }
                }
            }
        }

        // Both sums are of per-term lengths read straight off `.tvd`, with
        // nothing on the wire relating them to the compressed bytes that
        // follow -- so a corrupt chunk names any decompressed size it likes,
        // and `vec![0u8; n]` **aborts** for an absurd `n` rather than failing
        // (`docs/arithmetic-gate.md`: an allocation failure is a dead JVM, not
        // a catchable error). The LZ4 block format expands by at most 255x
        // (one 0xFF length-extension byte yields 255 output bytes), so a
        // decompressed length past that multiple of the bytes left in the
        // chunk cannot be produced by *any* input and is rejected before a
        // byte is allocated.
        //
        // Folding negatives away with `.max(0)` would silently accept a
        // *negative* length and let the per-document cursor below add it as a
        // huge `usize`. A byte length off disk is never negative in anything
        // Lucene wrote, so a negative one is rejected outright: that is what
        // makes every partial sum below a prefix of a total that has already
        // been bounded.
        // `prefix_lengths` does not contribute to the decompressed length --
        // a prefix is shared with the *previous* term, not stored again -- but
        // it is the same kind of value off the same stream, and folding a
        // negative one away with `.max(0)` would decode the term from its
        // suffix alone: accepted, self-consistent and wrong, where Java throws
        // (`System.arraycopy` with a negative `destPos`). Checked
        // here so `build_field` can cast without a reflex.
        sum_byte_lengths(&prefix_lengths, "term prefix")?;
        let total_suffix_len = sum_byte_lengths(&suffix_lengths, "term suffix")?;
        let total_payload_len = sum_byte_lengths(&payload_lengths_flat, "payload")?;
        let decompressed_len = usize::try_from(
            total_suffix_len
                .checked_add(total_payload_len)
                .ok_or_else(|| {
                    lucene_store::Error::Corrupted(
                        "term vector chunk suffix + payload lengths overflow i64".into(),
                    )
                })?,
        )
        .map_err(|_| {
            lucene_store::Error::Corrupted("term vector chunk is larger than usize".into())
        })?;
        let compressed_left = input.len().saturating_sub(input.position());
        let ceiling = compressed_left.saturating_mul(self.compression.max_expansion());
        if decompressed_len > ceiling {
            return Err(lucene_store::Error::Corrupted(format!(
                "term vector chunk claims {decompressed_len} decompressed bytes, which no \
                 {compressed_left} byte {} unit can produce",
                self.compression
            ))
            .into());
        }
        let mut decompressed = Vec::with_capacity(decompressed_len);
        compression::decompress(
            self.compression,
            &mut input,
            decompressed_len,
            0,
            decompressed_len,
            &mut decompressed,
        )?;
        // The LZ4 unit is NOT laid out as [all suffixes][all payloads]; it's
        // grouped **per document**: each document's fields' suffix bytes
        // (in field order), followed immediately by that same document's
        // payload bytes (only for its fields with the PAYLOADS flag) -- then
        // the next document's suffix bytes, and so on. Confirmed by tracing
        // `Lucene90CompressingTermVectorsReader`'s decompress call, which
        // decompresses a contiguous `[docOff+payloadOff, +docLen+payloadLen)`
        // slice per requested document.
        let mut suffix_byte_starts = vec![0usize; total_fields];
        let mut payload_byte_starts = vec![0usize; total_fields];
        let mut cursor = 0usize;
        // ARITH: (whole loop) `suffix_lengths` and `payload_lengths_flat` are
        // now known non-negative, and `field_offsets`/`term_offsets` partition
        // `0..total_fields` and `0..total_terms` respectively -- every field is
        // visited exactly once and every term belongs to exactly one field. So
        // `cursor`'s final value is exactly `total_suffix_len +
        // total_payload_len == decompressed_len`, which `usize::try_from`
        // already accepted; every intermediate value is smaller. The
        // `payload_start..` range is a prefix sum of the same counts
        // `total_payloads` accumulated, and `payload_lengths_flat` holds
        // `total_payloads` values, so its end is in range.
        #[allow(clippy::arithmetic_side_effects)]
        {
            for (doc_fields, _) in field_offsets.windows(2).zip(0..chunk_docs as usize) {
                let fstart = doc_fields[0] as usize;
                let fend = doc_fields[1] as usize;
                for field_idx in fstart..fend {
                    suffix_byte_starts[field_idx] = cursor;
                    let start = term_offsets[field_idx] as usize;
                    let end = term_offsets[field_idx + 1] as usize;
                    cursor += suffix_lengths[start..end].iter().sum::<i64>() as usize;
                }
                for field_idx in fstart..fend {
                    if all_flags[field_idx] & FLAG_PAYLOADS != 0 {
                        payload_byte_starts[field_idx] = cursor;
                        let payload_start = payload_starts[field_idx];
                        let field_payload_len: i64 = payload_lengths_flat
                            [payload_start..payload_start + field_freq_sums[field_idx] as usize]
                            .iter()
                            .sum();
                        cursor += field_payload_len as usize;
                    }
                }
            }
            debug_assert_eq!(cursor, decompressed_len);
        }

        Ok(DecodedChunk {
            doc_base,
            chunk_docs,
            field_offsets,
            field_nums,
            field_num_offs,
            all_flags,
            term_offsets,
            prefix_lengths,
            suffix_lengths,
            term_freqs_minus1,
            positions_flat,
            start_offsets_flat,
            lengths_flat,
            payload_lengths_flat,
            chars_per_term,
            decompressed,
            suffix_byte_starts,
            payload_byte_starts,
            position_starts,
            offset_starts,
            payload_starts,
        })
    }
}

/// Sum of `freq = freqMinus1 + 1` over one field's terms.
///
/// A frequency is an occurrence count: `freqMinus1 < -1` is corruption, and
/// without that check the sum goes negative and every stream length derived
/// from it stops bounding anything. The `+ 1` itself overflows at `i64::MAX`,
/// which is why this is `checked_add` and not a `map(|v| v + 1).sum()`.
fn sum_freqs(freqs_minus1: &[i64]) -> Result<i64> {
    let mut sum = 0i64;
    for &v in freqs_minus1 {
        let freq = v.checked_add(1).filter(|&f| f >= 0).ok_or_else(|| {
            lucene_store::Error::Corrupted(format!("term vector frequency {v} + 1 is not a count"))
        })?;
        sum = sum.checked_add(freq).ok_or_else(|| {
            lucene_store::Error::Corrupted("term vector frequencies overflow i64".into())
        })?;
    }
    Ok(sum)
}

/// Sum of a run of byte lengths read off `.tvd`, rejecting a negative one.
///
/// Deliberately not `saturating_add`: a saturated total would still be
/// compared against the LZ4 expansion ceiling and would still be rejected,
/// but it would report a length nobody wrote. The corruption is the answer.
fn sum_byte_lengths(lengths: &[i64], what: &str) -> Result<i64> {
    let mut sum = 0i64;
    for &v in lengths {
        if v < 0 {
            return Err(
                lucene_store::Error::Corrupted(format!("negative {what} length {v}")).into(),
            );
        }
        sum = sum.checked_add(v).ok_or_else(|| {
            lucene_store::Error::Corrupted(format!("{what} lengths overflow i64"))
        })?;
    }
    Ok(sum)
}

/// One field's occurrence count folded into a whole-chunk stream length.
fn checked_stream_len(total: i64, field_freq_sum: i64) -> Result<i64> {
    total.checked_add(field_freq_sum).ok_or_else(|| {
        lucene_store::Error::Corrupted("term vector chunk stream length overflows i64".into())
            .into()
    })
}

/// Where a chunk begins, from `.tvx`: its first document id and its `.tvd`
/// offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkIndexEntry {
    /// The chunk's first document id (its `docBase`).
    pub doc_base: i32,
    /// The chunk's start offset in `.tvd`.
    pub start_pointer: i64,
}

/// One chunk, fully decoded: every packed metadata array plus the whole
/// decompressed LZ4 unit, shared by all of the chunk's documents.
///
/// Java has no equivalent object -- `Lucene90CompressingTermVectorsReader.get`
/// re-reads the chunk's metadata arrays from the `IndexInput` on every call
/// and decompresses only the requested document's slice. This port decodes
/// the chunk once and lets a [`ChunkCursor`] reuse it, which is what makes a
/// sequential walk (a merge's per-document path, `CheckIndex`'s term-vector
/// pass) cost one decode per chunk instead of one per document.
#[derive(Debug, Clone, Default)]
pub struct DecodedChunk {
    doc_base: i32,
    chunk_docs: i32,
    field_offsets: Vec<i64>,
    field_nums: Vec<i64>,
    field_num_offs: Vec<i64>,
    all_flags: Vec<u8>,
    term_offsets: Vec<i64>,
    prefix_lengths: Vec<i64>,
    suffix_lengths: Vec<i64>,
    term_freqs_minus1: Vec<i64>,
    positions_flat: Vec<i64>,
    start_offsets_flat: Vec<i64>,
    lengths_flat: Vec<i64>,
    payload_lengths_flat: Vec<i64>,
    chars_per_term: Vec<f32>,
    decompressed: Vec<u8>,
    suffix_byte_starts: Vec<usize>,
    payload_byte_starts: Vec<usize>,
    position_starts: Vec<usize>,
    offset_starts: Vec<usize>,
    payload_starts: Vec<usize>,
}

impl DecodedChunk {
    /// The chunk's first document id.
    pub fn doc_base(&self) -> i32 {
        self.doc_base
    }

    /// How many documents this chunk holds.
    pub fn num_docs(&self) -> i32 {
        self.chunk_docs
    }

    /// Whether `doc_id` is one of them.
    ///
    /// ARITH: [`TermVectorsReader::read_chunk`] is the only thing that ever
    /// builds a non-`Default` `DecodedChunk`, and it rejects the chunk unless
    /// `doc_base.checked_add(chunk_docs)` succeeds and lands in
    /// `doc_id+1..=max_doc`. `Default` gives `0 + 0`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn contains(&self, doc_id: i32) -> bool {
        doc_id >= self.doc_base && doc_id < self.doc_base + self.chunk_docs
    }

    /// Materialises one of this chunk's documents, or `None` if it has no
    /// term-vector fields at all.
    ///
    /// ARITH: (whole body) every operand here is an index into this chunk's
    /// own decoded arrays, all of which `read_chunk` established:
    /// `doc_base + chunk_docs` is the `checked_add` [`Self::contains`] names;
    /// `doc_id - doc_base` is in `0..chunk_docs` because `contains` just
    /// returned true; `field_offsets` and `term_offsets` are non-decreasing
    /// prefix sums (`read_chunk` rejects a negative per-document field count
    /// and a negative per-field term count) with one more entry than the
    /// array they address, so every `[i + 1]` is in range and every
    /// difference is non-negative; and `doc_field_start + doc_num_fields` is
    /// `field_offsets[doc_index + 1]`, which is at most `total_fields`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn document(&self, doc_id: i32) -> Result<Option<TermVectorsDocument>> {
        if !self.contains(doc_id) {
            return Err(Error::CorruptChunkBounds {
                doc_base: self.doc_base,
                chunk_docs: self.chunk_docs,
                doc: doc_id,
                max_doc: self.doc_base + self.chunk_docs,
            });
        }
        let doc_index = (doc_id - self.doc_base) as usize;
        let doc_field_start = self.field_offsets[doc_index] as usize;
        let doc_num_fields =
            (self.field_offsets[doc_index + 1] - self.field_offsets[doc_index]) as usize;
        if doc_num_fields == 0 {
            return Ok(None);
        }

        let mut fields = Vec::with_capacity(doc_num_fields);
        for field_idx in doc_field_start..doc_field_start + doc_num_fields {
            fields.push(self.build_field(field_idx)?);
        }
        Ok(Some(TermVectorsDocument { fields }))
    }

    /// One field of one of this chunk's documents -- what
    /// [`Self::document`] would hold for `field_number` -- or `None` if the
    /// document has no vector for it; the document's other fields are not
    /// materialised.
    ///
    /// ARITH: as [`Self::document`].
    #[allow(clippy::arithmetic_side_effects)]
    pub fn field(&self, doc_id: i32, field_number: i32) -> Result<Option<TermVectorField>> {
        if !self.contains(doc_id) {
            return Err(Error::CorruptChunkBounds {
                doc_base: self.doc_base,
                chunk_docs: self.chunk_docs,
                doc: doc_id,
                max_doc: self.doc_base + self.chunk_docs,
            });
        }
        let doc_index = (doc_id - self.doc_base) as usize;
        let start = self.field_offsets[doc_index] as usize;
        let end = self.field_offsets[doc_index + 1] as usize;
        for field_idx in start..end {
            if self.field_nums[self.field_num_offs[field_idx] as usize] as i32 == field_number {
                return self.build_field(field_idx).map(Some);
            }
        }
        Ok(None)
    }

    /// The chunk's `field_idx`-th field, materialised.
    ///
    /// ARITH: as [`Self::document`]; `field_idx` is below `total_fields`.
    #[allow(clippy::arithmetic_side_effects)]
    fn build_field(&self, field_idx: usize) -> Result<TermVectorField> {
        let term_start = self.term_offsets[field_idx] as usize;
        let term_end = self.term_offsets[field_idx + 1] as usize;
        // `field_num_offs` was range-checked against `field_nums` once
        // per chunk in `read_chunk`, so this per-document path does not
        // re-check it. `chars_per_term` carries one entry per distinct
        // field number too, or none at all.
        let field_number = self.field_nums[self.field_num_offs[field_idx] as usize] as i32;
        let field_chars_per_term = if self.chars_per_term.is_empty() {
            0.0
        } else {
            self.chars_per_term[self.field_num_offs[field_idx] as usize]
        };
        build_field(FieldDecodeInput {
            field_number,
            flags: self.all_flags[field_idx],
            term_start,
            term_count: term_end - term_start,
            prefix_lengths: &self.prefix_lengths,
            suffix_lengths: &self.suffix_lengths,
            term_freqs_minus1: &self.term_freqs_minus1,
            suffix_bytes: &self.decompressed,
            suffix_byte_start: self.suffix_byte_starts[field_idx],
            positions_flat: &self.positions_flat,
            position_start: self.position_starts[field_idx],
            start_offsets_flat: &self.start_offsets_flat,
            lengths_flat: &self.lengths_flat,
            offset_start: self.offset_starts[field_idx],
            payload_bytes: &self.decompressed,
            payload_lengths_flat: &self.payload_lengths_flat,
            payload_start: self.payload_starts[field_idx],
            payload_byte_start: self.payload_byte_starts[field_idx],
            chars_per_term: field_chars_per_term,
        })
    }
}

/// The decoded chunk the last read came from, reused for every further
/// document that falls inside it -- the term-vectors twin of
/// [`crate::stored_fields::ChunkCursor`].
#[derive(Debug, Default)]
pub struct ChunkCursor {
    chunk: Option<DecodedChunk>,
}

impl ChunkCursor {
    /// A cursor holding no chunk yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// `doc_id`'s term vectors, decoding its chunk only if the currently
    /// held one does not contain it.
    pub fn document(
        &mut self,
        reader: &TermVectorsReader<'_>,
        doc_id: i32,
    ) -> Result<Option<TermVectorsDocument>> {
        if !self.chunk.as_ref().is_some_and(|c| c.contains(doc_id)) {
            self.chunk = Some(reader.read_chunk(doc_id)?);
        }
        self.chunk.as_ref().expect("just loaded").document(doc_id)
    }

    /// `doc_id`'s term vector of field `field_number` alone, as
    /// [`Self::document`] decodes its chunk.
    pub fn field(
        &mut self,
        reader: &TermVectorsReader<'_>,
        doc_id: i32,
        field_number: i32,
    ) -> Result<Option<TermVectorField>> {
        if !self.chunk.as_ref().is_some_and(|c| c.contains(doc_id)) {
            self.chunk = Some(reader.read_chunk(doc_id)?);
        }
        self.chunk
            .as_ref()
            .expect("just loaded")
            .field(doc_id, field_number)
    }

    /// Drops the held chunk.
    pub fn reset(&mut self) {
        self.chunk = None;
    }
}

/// `Lucene90TermVectorsFormat`'s chunk-size trigger: the writer closes a
/// chunk once this many term-suffix + payload bytes are buffered
/// (`super("Lucene90TermVectorsData", "", CompressionMode.FAST, 1 << 12,
/// 128, 10)`).
pub const CHUNK_SIZE: usize = 1 << 12;
/// `Lucene90TermVectorsFormat`'s document-count trigger.
pub const MAX_DOCS_PER_CHUNK: usize = 128;
/// `Lucene90TermVectorsFormat`'s `blockShift` for the `.tvx`
/// `DirectMonotonicWriter`s.
const INDEX_BLOCK_SHIFT: u32 = 10;

/// One buffered field of a pending document -- Java's `FieldData`.
///
/// Java slices its per-occurrence arrays out of four writer-wide
/// `positionsBuf`/`startOffsetsBuf`/`lengthsBuf`/`payloadLengthsBuf` scratch
/// buffers addressed by `posStart`/`offStart`/`payStart`, purely to avoid
/// allocating per field. Rust owns them per field instead: the buffers exist
/// only for the ≤128 documents of one pending chunk, so the allocation
/// avoidance buys nothing and the offset arithmetic (which is where Java's
/// `addField`/`addDocData` complexity lives) disappears.
#[derive(Debug)]
struct PendingField {
    field_num: i32,
    flags: u8,
    /// One entry per term.
    freqs: Vec<i32>,
    prefix_lengths: Vec<i32>,
    suffix_lengths: Vec<i32>,
    /// One entry per occurrence, absolute (deltas are computed at flush).
    positions: Vec<i32>,
    start_offsets: Vec<i32>,
    /// `endOffset - startOffset` per occurrence.
    lengths: Vec<i32>,
    payload_lengths: Vec<i32>,
}

#[derive(Debug)]
struct PendingDoc {
    fields: Vec<PendingField>,
}

/// Length of the longest common prefix of `a` and `b` --
/// `StringHelper.bytesDifference`, minus its "terms out of order" throw:
/// Java can afford that because `TermVectorsWriter.addAllDocVectors` feeds
/// it a `TermsEnum`, while this port's [`TermVectorsDocument`] is a plain
/// value a caller can build however it likes. Equal terms simply share their
/// whole length, which round-trips (the reader rebuilds
/// `lastTerm[..prefix] + suffix` either way).
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Port of `Lucene90CompressingTermVectorsWriter`: the streaming, chunking
/// term-vectors writer.
///
/// Documents are buffered until either of Java's two triggers fires --
/// [`CHUNK_SIZE`] buffered term-suffix + payload bytes, or
/// [`MAX_DOCS_PER_CHUNK`] documents -- at which point the chunk is flushed
/// as one LZ4 unit plus its bit-packed metadata headers. A chunk closed by
/// anything *other* than a trigger (the final [`Self::finish`], or the
/// pre-bulk-copy flush in [`Self::copy_chunks`]) is **dirty**: flagged in its
/// own token and tallied in `.tvm`, because it is smaller than the format
/// intends and a later merge may want to recompress rather than copy it.
///
/// Deliberate divergences from Java, all compression-ratio-only:
///
/// - Every `direct_reader`/`packed_ints` array uses the exact bit width its
///   own values need rather than a cross-chunk minimisation.
///
/// What is *not* scoped down any more (it was, before batch `c8`): chunking
/// itself, term prefix sharing, `charsPerTerm`, and the
/// `nonChangingFlags` per-field-number flags encoding are all Java's.
pub struct TermVectorsWriter {
    segment_id: [u8; ID_LENGTH],
    segment_suffix: String,
    /// `None` for `Lucene90TermVectorsFormat` (`FAST`); a `CompressingCodec`'s
    /// format otherwise, whose units `compressor` writes as Java's compressor
    /// does.
    format: Option<&'static CompressingFormat>,
    compressor: Compressor,
    /// `Lucene90CompressingTermVectorsWriter`'s `chunkSize` constructor
    /// parameter -- [`CHUNK_SIZE`] for `Lucene90TermVectorsFormat`.
    chunk_size: usize,
    /// Its `maxDocsPerChunk` -- [`MAX_DOCS_PER_CHUNK`] for
    /// `Lucene90TermVectorsFormat`.
    max_docs_per_chunk: usize,
    tvd: Vec<u8>,
    /// Cumulative doc counts, one per chunk written so far, seeded with `0`;
    /// the last entry doubles as `.tvx`'s trailing `maxDoc` sentinel.
    docs_values: Vec<i64>,
    /// Chunk start offsets, seeded with the first chunk's (= the index
    /// header length); the last entry doubles as the `maxPointer` sentinel.
    start_pointers_values: Vec<i64>,
    num_dirty_chunks: i64,
    num_dirty_docs: i64,
    /// Java's `pendingDocs`: the current chunk only, reset per flush.
    pending_docs: Vec<PendingDoc>,
    /// Java's `termSuffixes` (with `payloadBytes` already appended per
    /// document, exactly as `finishDocument` does): the chunk's LZ4 payload,
    /// and the quantity `triggerFlush` measures.
    term_suffixes: Vec<u8>,
    /// Java's `numDocs`: every document this writer has seen, buffered or
    /// flushed or bulk-copied.
    num_docs: i32,
}

impl TermVectorsWriter {
    /// A fresh writer with `Lucene90TermVectorsFormat`'s chunk geometry
    /// ([`CHUNK_SIZE`] bytes / [`MAX_DOCS_PER_CHUNK`] documents) and `.tvd`'s
    /// index header already written.
    pub fn new(segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Self {
        Self::with_geometry(segment_id, segment_suffix, CHUNK_SIZE, MAX_DOCS_PER_CHUNK)
    }

    /// [`Self::new`] with an explicit chunk geometry --
    /// `Lucene90CompressingTermVectorsWriter`'s own `chunkSize`/
    /// `maxDocsPerChunk` constructor parameters, which
    /// `Lucene90TermVectorsFormat` fixes at `1 << 12` / `128`.
    ///
    /// A segment written with a geometry other than [`Self::new`]'s is still
    /// a valid, real-Lucene-readable file (`chunkSize` is recorded in
    /// `.tvm`), but it can never be bulk-copied into a default-geometry
    /// merge: [`Self::can_bulk_copy`] compares the two, exactly as
    /// `canPerformBulkMerge` does.
    pub fn with_geometry(
        segment_id: &[u8; ID_LENGTH],
        segment_suffix: &str,
        chunk_size: usize,
        max_docs_per_chunk: usize,
    ) -> Self {
        Self::build(
            segment_id,
            segment_suffix,
            chunk_size,
            max_docs_per_chunk,
            None,
        )
    }

    /// `Lucene90CompressingTermVectorsWriter` for a `CompressingCodec`'s
    /// format: its data codec name, segment suffix, compression, chunk
    /// geometry and block shift.
    ///
    /// # Panics
    /// A chunk size or document cap that is not positive, or a block shift
    /// outside `DirectMonotonicWriter`'s 2..=22.
    pub fn with_format(segment_id: &[u8; ID_LENGTH], format: &'static CompressingFormat) -> Self {
        assert!(format.chunk_size > 0 && format.max_docs_per_chunk > 0);
        assert!((2..=22).contains(&format.block_shift));
        Self::build(
            segment_id,
            format.segment_suffix,
            format.chunk_size as usize,
            format.max_docs_per_chunk as usize,
            Some(format),
        )
    }

    fn build(
        segment_id: &[u8; ID_LENGTH],
        segment_suffix: &str,
        chunk_size: usize,
        max_docs_per_chunk: usize,
        format: Option<&'static CompressingFormat>,
    ) -> Self {
        // `.tvm` records `chunkSize` as a vint, and `too_dirty` compares
        // `maxDocsPerChunk` against an `i64` doc count, so both must survive
        // the narrowing.
        assert!(chunk_size > 0 && chunk_size <= i32::MAX as usize);
        assert!(max_docs_per_chunk > 0 && max_docs_per_chunk <= i32::MAX as usize);
        let mut tvd = Vec::new();
        codec_util::write_index_header(
            &mut tvd,
            format.map_or(DATA_CODEC, |format| format.format_name),
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        );
        let start = tvd.len() as i64;
        Self {
            segment_id: *segment_id,
            segment_suffix: segment_suffix.to_string(),
            format,
            compressor: Compressor::default(),
            chunk_size,
            max_docs_per_chunk,
            tvd,
            docs_values: vec![0],
            start_pointers_values: vec![start],
            num_dirty_chunks: 0,
            num_dirty_docs: 0,
            pending_docs: Vec::new(),
            term_suffixes: Vec::new(),
            num_docs: 0,
        }
    }

    /// `getChunkSize()` -- a bulk chunk copy is only legal from a reader
    /// that agrees with this.
    pub fn chunk_size(&self) -> i32 {
        self.chunk_size as i32
    }

    /// How many documents this writer has been given so far (Java's
    /// `numDocs`), buffered ones included.
    pub fn num_docs(&self) -> i32 {
        self.num_docs
    }

    /// Java's `tooDirty(candidate)`: a source is too dirty to bulk-copy when
    /// it has enough dirty documents to make a full chunk *and* more than 1%
    /// of its chunks are dirty. Copying such a segment verbatim would carry
    /// its degraded compression ratio forward forever.
    ///
    /// ARITH: [`open`] rejects a `.tvm` whose `numDirtyChunks` is negative or
    /// greater than `numChunks`, and `numChunks + 1` must equal the index's
    /// `numChunks`, which is an `i32`. So `num_dirty_chunks` is in
    /// `0..i32::MAX` and `* 100` is three orders of magnitude inside `i64`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn too_dirty(&self, reader: &TermVectorsReader<'_>) -> bool {
        reader.num_dirty_docs() > self.max_docs_per_chunk as i64
            && reader.num_dirty_chunks() * 100 > reader.num_chunks()
    }

    /// Whether this writer could copy `reader`'s compressed chunks verbatim,
    /// ignoring the caller's own field-numbering and deletions checks --
    /// `canPerformBulkMerge`'s `compressionMode`/`chunkSize`/`version`/
    /// `packedIntsVersion`/`tooDirty` half. (The version checks are [`open`]'s
    /// job here: this port has exactly one `VERSION_CURRENT` and refuses to
    /// open anything else. The data codec name stands for the compression.)
    pub fn can_bulk_copy(&self, reader: &TermVectorsReader<'_>) -> bool {
        reader.data_codec() == self.data_codec()
            && reader.chunk_size() == self.chunk_size as i32
            && !self.too_dirty(reader)
    }

    /// The `.tvd` codec name this writer writes.
    pub fn data_codec(&self) -> &'static str {
        self.format.map_or(DATA_CODEC, |format| format.format_name)
    }

    fn block_shift(&self) -> u32 {
        self.format
            .map_or(INDEX_BLOCK_SHIFT, |format| format.block_shift as u32)
    }

    fn trigger_flush(&self) -> bool {
        self.term_suffixes.len() >= self.chunk_size
            || self.pending_docs.len() >= self.max_docs_per_chunk
    }

    /// Java's `startDocument`/`startField`/`startTerm`/`addPosition`/
    /// `finishDocument` for one whole document.
    ///
    /// # Panics
    ///
    /// If a field's per-occurrence arrays disagree with its flags or with a
    /// term's `freq` -- a caller-side wiring bug that would otherwise emit a
    /// chunk whose streams no longer line up with its own headers -- or if
    /// the writer is handed more than `i32::MAX` documents, which is past
    /// what `.tvm`'s `maxDoc` field can record.
    pub fn add_document(&mut self, doc: &TermVectorsDocument) {
        let mut fields = Vec::with_capacity(doc.fields.len());
        for field in &doc.fields {
            validate_field(field);
            let flags = (if field.has_positions {
                FLAG_POSITIONS
            } else {
                0
            }) | (if field.has_offsets { FLAG_OFFSETS } else { 0 })
                | (if field.has_payloads { FLAG_PAYLOADS } else { 0 });
            let mut pending = PendingField {
                field_num: field.field_number,
                flags,
                freqs: Vec::with_capacity(field.terms.len()),
                prefix_lengths: Vec::with_capacity(field.terms.len()),
                suffix_lengths: Vec::with_capacity(field.terms.len()),
                positions: Vec::new(),
                start_offsets: Vec::new(),
                lengths: Vec::new(),
                payload_lengths: Vec::new(),
            };
            // Java's `lastTerm`, reset at `startField`.
            let mut last_term: &[u8] = &[];
            for term in &field.terms {
                let prefix = if last_term.is_empty() {
                    0
                } else {
                    common_prefix_len(last_term, &term.term)
                };
                pending.freqs.push(term.freq);
                pending.prefix_lengths.push(prefix as i32);
                // ARITH: `common_prefix_len` counts matching leading bytes of
                // `last_term` and `term.term`, so it never exceeds either
                // length; the `if` above makes it 0 for the field's first
                // term.
                #[allow(clippy::arithmetic_side_effects)]
                pending
                    .suffix_lengths
                    .push((term.term.len() - prefix) as i32);
                self.term_suffixes.extend_from_slice(&term.term[prefix..]);
                last_term = &term.term;

                if field.has_positions {
                    pending
                        .positions
                        .extend_from_slice(term.positions.as_ref().unwrap());
                }
                if field.has_offsets {
                    let starts = term.start_offsets.as_ref().unwrap();
                    let ends = term.end_offsets.as_ref().unwrap();
                    pending.start_offsets.extend_from_slice(starts);
                    for (s, e) in starts.iter().zip(ends) {
                        // Java's `lengthsBuf[...] = endOffset - startOffset`
                        // is an `int` subtraction that wraps, and the reader
                        // undoes it with a `wrapping_add` against the term
                        // length. Wrapping is what round-trips a caller's
                        // absurd offset pair, so this is `wrapping_sub`
                        // rather than a check that would reject one.
                        pending.lengths.push(e.wrapping_sub(*s));
                    }
                }
                if field.has_payloads {
                    for payload in term.payloads.as_ref().unwrap() {
                        pending.payload_lengths.push(payload.len() as i32);
                    }
                }
            }
            fields.push(pending);
        }
        // `finishDocument`: the document's payload bytes land immediately
        // after its term suffixes, before the next document's.
        for field in &doc.fields {
            if field.has_payloads {
                for term in &field.terms {
                    for payload in term.payloads.as_ref().unwrap() {
                        self.term_suffixes.extend_from_slice(payload);
                    }
                }
            }
        }
        self.pending_docs.push(PendingDoc { fields });
        // `numDocs` is `.tvm`'s `maxDoc`, an `i32`. Java writes `++numDocs`
        // and leaves the ceiling to `IndexWriter`'s document limit; here a
        // wrap would emit a segment whose `.tvx` disagrees with its own
        // `maxDoc`. This method already panics on a caller wiring bug
        // (`validate_field`), so it panics on this one too rather than
        // silently writing a broken segment.
        self.num_docs = self
            .num_docs
            .checked_add(1)
            .expect("term vectors writer exceeded i32::MAX documents");
        if self.trigger_flush() {
            self.flush(false);
        }
    }

    /// One `.tvx` entry: the cumulative doc count and the offset one past the
    /// chunk just written (which is the *next* chunk's start pointer, and
    /// after the last chunk is `maxPointer`).
    fn record_chunk(&mut self) {
        self.docs_values.push(self.num_docs as i64);
        self.start_pointers_values.push(self.tvd.len() as i64);
    }

    /// Java's `flush(force)`. `force` is what makes a chunk **dirty**: it
    /// closed before either trigger fired.
    fn flush(&mut self, force: bool) {
        debug_assert_ne!(self.trigger_flush(), force);
        let chunk_docs = self.pending_docs.len();
        debug_assert!(chunk_docs > 0);
        // ARITH: `num_dirty_chunks` counts a subset of the chunks this
        // writer has flushed and `num_dirty_docs` a subset of its documents,
        // and `add_document` caps the document count at `i32::MAX` -- so both
        // `i64` counters stay under 2^31.
        #[allow(clippy::arithmetic_side_effects)]
        if force {
            self.num_dirty_chunks += 1;
            self.num_dirty_docs += chunk_docs as i64;
        }
        // `tvd`, `pending_docs` and `term_suffixes` are distinct fields, but
        // the borrow checker cannot see that through `&mut self`; swap the
        // output out for the duration of the flush and put it back.
        let mut out = std::mem::take(&mut self.tvd);
        // ARITH: `pending_docs` holds documents `add_document` has already
        // counted into `num_docs`, so `chunk_docs <= num_docs`.
        #[allow(clippy::arithmetic_side_effects)]
        let doc_base = self.num_docs - chunk_docs as i32;
        out.write_vint(doc_base);
        out.write_vint(((chunk_docs as i32) << 1) | i32::from(force));

        let total_fields = flush_num_fields(&mut out, &self.pending_docs);
        if total_fields > 0 {
            let field_nums = flush_field_nums(&mut out, &self.pending_docs);
            flush_fields(&mut out, &self.pending_docs, &field_nums);
            flush_flags(&mut out, &self.pending_docs, &field_nums);
            flush_num_terms(&mut out, &self.pending_docs);
            flush_term_lengths(&mut out, &self.pending_docs);
            flush_term_freqs(&mut out, &self.pending_docs);
            flush_positions(&mut out, &self.pending_docs);
            flush_offsets(&mut out, &self.pending_docs, &field_nums);
            flush_payload_lengths(&mut out, &self.pending_docs);
            // `Lucene90TermVectorsFormat` is `CompressionMode.FAST`: one LZ4
            // unit through the writer's reused hash table, as Java's.
            let compression = self
                .format
                .map_or(Compression::Fast, |format| format.compression);
            self.compressor
                .compress(compression, &self.term_suffixes, &mut out);
        }
        self.tvd = out;

        self.pending_docs.clear();
        self.term_suffixes.clear();
        self.record_chunk();
    }

    /// BULK path (`copyChunks`): copy `reader`'s documents `from_doc`
    /// (inclusive) to `to_doc` (exclusive) into this writer, copying whole
    /// *compressed* chunks verbatim wherever a chunk lies entirely inside
    /// that range.
    ///
    /// # Preconditions
    ///
    /// The caller owns every condition that makes this safe, exactly as
    /// `canPerformBulkMerge` does in Java:
    /// - [`Self::can_bulk_copy`] holds for `reader`;
    /// - `reader`'s segment has **no deletions**, and `from_doc..to_doc` is a
    ///   run of consecutive source doc ids mapping to consecutive merged ids
    ///   starting at this writer's current document count;
    /// - `reader`'s field numbers are unchanged by the merge
    ///   (`MatchingReaders`), since the copied bytes encode them;
    /// - `reader.check_integrity()` passed -- a byte copy writes a fresh,
    ///   valid footer over copied bytes, so a corrupt source would otherwise
    ///   be laundered into a permanently-valid segment.
    ///
    /// A leading partial chunk (when `from_doc` is not a chunk boundary) and
    /// a trailing partial chunk (when `to_doc` is not) are re-encoded
    /// document at a time instead -- this is what Java's `isLoaded(docID)`
    /// loops do, stated as the chunk-boundary condition they are really
    /// testing rather than as a property of the reader's cached block.
    ///
    /// This allocates a fresh [`ChunkCursor`] for those partial chunks, so
    /// every call decompresses its own. That is right for a caller making one
    /// large call per source; a caller making **many short calls** on the same
    /// reader -- which is what an index-sorted merge does, since the sources
    /// interleave and every run is a document or two -- must use
    /// [`Self::copy_chunks_with_cursor`] and keep one cursor per reader.
    /// Java gets that for free from the reader's own cached block.
    pub fn copy_chunks(
        &mut self,
        reader: &TermVectorsReader<'_>,
        from_doc: i32,
        to_doc: i32,
    ) -> Result<()> {
        let mut cursor = ChunkCursor::new();
        self.copy_chunks_with_cursor(reader, &mut cursor, from_doc, to_doc)
    }

    /// [`Self::copy_chunks`] with a caller-owned [`ChunkCursor`] for the
    /// partial chunks at each end of the run -- the equivalent of Java's
    /// per-reader cached block, and the difference between one decompression
    /// per chunk and one per document when the runs are short.
    pub fn copy_chunks_with_cursor(
        &mut self,
        reader: &TermVectorsReader<'_>,
        cursor: &mut ChunkCursor,
        from_doc: i32,
        to_doc: i32,
    ) -> Result<()> {
        // A real check, not a `debug_assert`: this is a `pub` method on a
        // library type, and a release-mode caller copying from a reader with
        // a different `chunkSize` would produce silently wrong documents.
        if !self.can_bulk_copy(reader) {
            return Err(Error::BulkCopyNotPermitted {
                reader_chunk_size: reader.chunk_size(),
                writer_chunk_size: self.chunk_size as i32,
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
        // ARITH: (both partial-chunk walks) the loop condition establishes
        // `doc < to_doc <= max_doc <= i32::MAX`, so `doc + 1` cannot overflow.
        #[allow(clippy::arithmetic_side_effects)]
        while doc < to_doc && reader.chunk_for_doc(doc)?.doc_base != doc {
            let d = cursor.document(reader, doc)?.unwrap_or_default();
            self.add_document(&d);
            doc += 1;
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
            if !self.pending_docs.is_empty() {
                self.flush(true);
            }
            let mut pointer = from_pointer;
            let tvd = reader.tvd();
            loop {
                let mut input = SliceInput::new(tvd);
                input.seek(pointer as usize)?;
                let base = input.read_vint()?;
                let code = input.read_vint()?;
                // `Lucene90CompressingTermVectorsWriter.copyChunks:858` is
                // `code >>> 1`, unsigned, for the same reason as the reader.
                let chunk_docs = (code as u32 >> 1) as i32;
                // Java's two `CorruptIndexException`s: `.tvx` and the `.tvd`
                // chunk headers are redundant, and disagreeing is how a bad
                // bulk copy announces itself before it writes a segment that
                // reads back plausible-but-wrong vectors.
                if base != doc || chunk_docs <= 0 {
                    return Err(Error::CorruptChunkBounds {
                        doc_base: base,
                        chunk_docs,
                        doc,
                        max_doc,
                    });
                }
                let body_start = input.position();
                self.tvd.write_vint(self.num_docs); // rebase
                self.tvd.write_vint(code);
                // `chunk_docs` is `code >> 1` off a `.tvd` chunk header, so
                // it is an unconstrained positive `i32`: both `doc` and
                // `num_docs` could wrap past `to_doc` and let the guard below
                // pass on a chunk that claims two billion documents.
                let advanced = doc
                    .checked_add(chunk_docs)
                    .zip(self.num_docs.checked_add(chunk_docs));
                let Some((next_doc, next_num_docs)) = advanced else {
                    return Err(Error::CorruptChunkBounds {
                        doc_base: base,
                        chunk_docs,
                        doc,
                        max_doc,
                    });
                };
                doc = next_doc;
                self.num_docs = next_num_docs;
                if doc > to_doc {
                    return Err(Error::CorruptChunkBounds {
                        doc_base: base,
                        chunk_docs,
                        doc: to_doc,
                        max_doc,
                    });
                }
                let end_pointer = if doc == max_doc {
                    reader.max_pointer()
                } else {
                    reader.chunk_for_doc(doc)?.start_pointer
                };
                let body = tvd
                    .get(body_start..end_pointer as usize)
                    .ok_or(lucene_store::Error::Eof { offset: body_start })?;
                self.tvd.write_bytes(body);
                // ARITH: `chunk_docs` is now known to keep `num_docs` inside
                // `i32`, and both counters tally subsets of chunks and
                // documents this writer holds, so neither `i64` can overflow.
                #[allow(clippy::arithmetic_side_effects)]
                if code & 1 != 0 {
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
        // ARITH: as above, `doc < to_doc <= i32::MAX`.
        #[allow(clippy::arithmetic_side_effects)]
        while doc < to_doc {
            let d = cursor.document(reader, doc)?.unwrap_or_default();
            self.add_document(&d);
            doc += 1;
        }
        Ok(())
    }

    /// Java's `finish(numDocs)`: flush whatever is buffered as a final dirty
    /// chunk, then assemble `.tvd`/`.tvx`/`.tvm`.
    pub fn finish(mut self) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        if !self.pending_docs.is_empty() {
            self.flush(true);
        }
        let max_doc = self.num_docs;
        let max_pointer = self.tvd.len() as i64;
        codec_util::write_footer(&mut self.tvd);

        let mut tvx = Vec::new();
        codec_util::write_index_header(
            &mut tvx,
            INDEX_CODEC,
            INDEX_VERSION_CURRENT,
            &self.segment_id,
            &self.segment_suffix,
        );
        let docs_start_pointer = tvx.len() as i64;
        let (docs_meta_bytes, docs_data_bytes) =
            direct_monotonic::write(&self.docs_values, self.block_shift());
        tvx.write_bytes(&docs_data_bytes);
        let docs_end_pointer = tvx.len() as i64;
        let (start_pointers_meta_bytes, start_pointers_data_bytes) =
            direct_monotonic::write(&self.start_pointers_values, self.block_shift());
        tvx.write_bytes(&start_pointers_data_bytes);
        let start_pointers_end_pointer = tvx.len() as i64;
        codec_util::write_footer(&mut tvx);

        let mut tvm = Vec::new();
        codec_util::write_index_header(
            &mut tvm,
            META_CODEC,
            VERSION_CURRENT,
            &self.segment_id,
            &self.segment_suffix,
        );
        // PackedInts.VERSION_CURRENT; unused by this port's own reader, but
        // real Lucene's `BlockPackedReaderIterator` validates it.
        tvm.write_vint(2);
        tvm.write_vint(self.chunk_size as i32);
        tvm.write_i32(max_doc);
        tvm.write_i32(self.block_shift() as i32);
        tvm.write_i32(self.docs_values.len() as i32); // real chunks + 1 sentinel
        tvm.write_i64(docs_start_pointer);
        tvm.write_bytes(&docs_meta_bytes);
        tvm.write_i64(docs_end_pointer);
        tvm.write_bytes(&start_pointers_meta_bytes);
        tvm.write_i64(start_pointers_end_pointer);
        tvm.write_i64(max_pointer);
        // ARITH: `docs_values` is seeded with one entry in `with_geometry`
        // and only ever pushed to, so its length is at least 1.
        #[allow(clippy::arithmetic_side_effects)]
        tvm.write_vlong(self.docs_values.len() as i64 - 1); // numChunks (outer)
        tvm.write_vlong(self.num_dirty_chunks);
        tvm.write_vlong(self.num_dirty_docs);
        codec_util::write_footer(&mut tvm);

        (self.tvd, tvx, tvm)
    }
}

fn validate_field(field: &TermVectorField) {
    for term in &field.terms {
        if field.has_positions {
            assert_eq!(
                term.positions.as_ref().map(|p| p.len()),
                Some(term.freq as usize),
                "positions length must equal freq"
            );
        } else {
            assert!(term.positions.is_none());
        }
        if field.has_offsets {
            assert_eq!(
                term.start_offsets.as_ref().map(|p| p.len()),
                Some(term.freq as usize),
                "start_offsets length must equal freq"
            );
            assert_eq!(
                term.end_offsets.as_ref().map(|p| p.len()),
                Some(term.freq as usize),
                "end_offsets length must equal freq"
            );
        } else {
            assert!(term.start_offsets.is_none() && term.end_offsets.is_none());
        }
        if field.has_payloads {
            assert_eq!(
                term.payloads.as_ref().map(|p| p.len()),
                Some(term.freq as usize),
                "payloads length must equal freq"
            );
        } else {
            assert!(term.payloads.is_none());
        }
    }
}

/// `flushNumFields`.
fn flush_num_fields(out: &mut Vec<u8>, docs: &[PendingDoc]) -> usize {
    if docs.len() == 1 {
        let n = docs[0].fields.len();
        out.write_vint(n as i32);
        return n;
    }
    let counts: Vec<i64> = docs.iter().map(|d| d.fields.len() as i64).collect();
    out.write_bytes(&block_packed::encode_all(&counts));
    counts.iter().sum::<i64>() as usize
}

/// `flushFieldNums` -- the chunk's distinct field numbers, **sorted**, as a
/// headerless `PackedInts.Format.PACKED` array behind a
/// `(min(n-1,7) << 5) | bitsRequired` token.
fn flush_field_nums(out: &mut Vec<u8>, docs: &[PendingDoc]) -> Vec<i64> {
    let mut field_nums: Vec<i64> = docs
        .iter()
        .flat_map(|d| d.fields.iter().map(|f| f.field_num as i64))
        .collect();
    field_nums.sort_unstable();
    field_nums.dedup();

    let max_field_num = *field_nums.last().unwrap();
    // `PackedInts.bitsRequired` is `max(1, 64 - numberOfLeadingZeros(v))` --
    // never 0. That floor is load-bearing rather than cosmetic: real
    // Lucene's reader unconditionally indexes `packedBulkOps[bitsPerValue -
    // 1]`, so a 0-bit width (every field number in the chunk is 0 -- an
    // ordinary single-field index) throws `ArrayIndexOutOfBoundsException`
    // there while decoding fine through this port's own more permissive
    // `packed_ints::get`.
    // ARITH: the `else` arm runs only for `max_field_num >= 1`, whose `u64`
    // widening has at most 63 leading zeros. `field_nums` is non-empty --
    // `flush_field_nums` is called only when `flush_num_fields` returned a
    // positive total, i.e. at least one pending field contributed a number.
    // `td1 - 7` runs only under `inline_td1 == 0x07`, which is `td1.min(7)`
    // and therefore implies `td1 >= 7`.
    #[allow(clippy::arithmetic_side_effects)]
    let bits_per_field_num: u32 = if max_field_num <= 0 {
        1
    } else {
        (64 - (max_field_num as u64).leading_zeros()).min(31)
    };
    // ARITH: `field_nums` is non-empty (see above), so `len() - 1` is fine.
    #[allow(clippy::arithmetic_side_effects)]
    let td1 = (field_nums.len() - 1) as u32;
    let inline_td1 = td1.min(0x07);
    out.write_byte(((inline_td1 << 5) | bits_per_field_num) as u8);
    // ARITH: `inline_td1` is `td1.min(7)`, so `== 7` implies `td1 >= 7`.
    #[allow(clippy::arithmetic_side_effects)]
    if inline_td1 == 0x07 {
        out.write_vint((td1 - 7) as i32);
    }
    out.write_bytes(&packed_ints::encode(&field_nums, bits_per_field_num));
    field_nums
}

/// Index of `field_num` in the sorted distinct list -- Java's
/// `Arrays.binarySearch(fieldNums, fd.fieldNum)`.
fn field_num_off(field_nums: &[i64], field_num: i32) -> usize {
    field_nums
        .binary_search(&(field_num as i64))
        .expect("field_nums holds every pending field's number")
}

/// `flushFields`: one `direct_reader` index into `field_nums` per field.
fn flush_fields(out: &mut Vec<u8>, docs: &[PendingDoc], field_nums: &[i64]) {
    let offs: Vec<i64> = docs
        .iter()
        .flat_map(|d| d.fields.iter())
        .map(|f| field_num_off(field_nums, f.field_num) as i64)
        .collect();
    // ARITH: `field_nums` came from `flush_field_nums`, which is only called
    // for a chunk with at least one field, so its length is at least 1.
    #[allow(clippy::arithmetic_side_effects)]
    let bits = direct_writer_bits_required(field_nums.len() as i64 - 1);
    write_direct(out, &offs, bits);
}

/// A `DirectWriter` block into a scratch buffer, then its length and bytes:
/// `DirectWriter.finish()` pads to a whole read word for the widths a reader
/// fetches as 2/4/8 bytes (12, 20, ...), and the length counts the padding.
fn write_direct(out: &mut Vec<u8>, values: &[i64], bits: u8) {
    let mut bytes = direct_reader::encode(values, bits);
    // ARITH: the padding is at most 7 bytes onto an in-memory buffer's length.
    #[allow(clippy::arithmetic_side_effects)]
    bytes.resize(bytes.len() + direct_reader::padding_bytes_needed(bits), 0);
    out.write_vint(bytes.len() as i32);
    out.write_bytes(&bytes);
}

/// `flushFlags`: selector `0` writes one flag per *distinct field number*
/// when every instance of a field number agrees (the overwhelmingly common
/// case -- a 128-document chunk of two fields is 2 entries instead of 256),
/// selector `1` one flag per field instance otherwise.
fn flush_flags(out: &mut Vec<u8>, docs: &[PendingDoc], field_nums: &[i64]) {
    let mut field_flags: Vec<i32> = vec![-1; field_nums.len()];
    let mut non_changing = true;
    'outer: for d in docs {
        for f in &d.fields {
            let off = field_num_off(field_nums, f.field_num);
            if field_flags[off] == -1 {
                field_flags[off] = f.flags as i32;
            } else if field_flags[off] != f.flags as i32 {
                non_changing = false;
                break 'outer;
            }
        }
    }
    let (selector, values): (i32, Vec<i64>) = if non_changing {
        (0, field_flags.iter().map(|&v| v as i64).collect())
    } else {
        (
            1,
            docs.iter()
                .flat_map(|d| d.fields.iter())
                .map(|f| f.flags as i64)
                .collect(),
        )
    };
    out.write_vint(selector);
    write_direct(out, &values, FLAGS_BITS);
}

/// `flushNumTerms`.
fn flush_num_terms(out: &mut Vec<u8>, docs: &[PendingDoc]) {
    let counts: Vec<i64> = docs
        .iter()
        .flat_map(|d| d.fields.iter())
        .map(|f| f.freqs.len() as i64)
        .collect();
    // Java accumulates `maxNumTerms |= fd.numTerms`, which is the maximum
    // only in the bits-required sense -- the same width either way.
    let max_num_terms = counts.iter().copied().max().unwrap_or(0);
    let bits = direct_writer_bits_required(max_num_terms);
    out.write_vint(bits as i32);
    write_direct(out, &counts, bits);
}

/// `flushTermLengths`: the prefix stream then the suffix stream.
fn flush_term_lengths(out: &mut Vec<u8>, docs: &[PendingDoc]) {
    let prefixes: Vec<i64> = docs
        .iter()
        .flat_map(|d| d.fields.iter())
        .flat_map(|f| f.prefix_lengths.iter().map(|&v| v as i64))
        .collect();
    out.write_bytes(&block_packed::encode_all(&prefixes));
    let suffixes: Vec<i64> = docs
        .iter()
        .flat_map(|d| d.fields.iter())
        .flat_map(|f| f.suffix_lengths.iter().map(|&v| v as i64))
        .collect();
    out.write_bytes(&block_packed::encode_all(&suffixes));
}

/// `flushTermFreqs`.
fn flush_term_freqs(out: &mut Vec<u8>, docs: &[PendingDoc]) {
    let freqs: Vec<i64> = docs
        .iter()
        .flat_map(|d| d.fields.iter())
        // ARITH: each `freq` is an `i32` widened to `i64` before the
        // subtraction, so `freq - 1` is nowhere near an `i64` boundary.
        .flat_map(|f| {
            // ARITH: as above.
            #[allow(clippy::arithmetic_side_effects)]
            f.freqs.iter().map(|&v| v as i64 - 1)
        })
        .collect();
    out.write_bytes(&block_packed::encode_all(&freqs));
}

/// `flushPositions`: per term, the first occurrence absolute and every later
/// one delta-coded against the previous occurrence *of that term*.
fn flush_positions(out: &mut Vec<u8>, docs: &[PendingDoc]) {
    let mut deltas: Vec<i64> = Vec::new();
    for d in docs {
        for f in &d.fields {
            if f.flags & FLAG_POSITIONS == 0 {
                continue;
            }
            // ARITH: `validate_field` asserts `positions.len() == freq` per
            // term, and `f.positions` is those runs concatenated, so `pos`
            // advances exactly `sum(freqs)` times and stays inside
            // `f.positions`. The delta is `wrapping_sub` for the same reason
            // Java's `positionsBuf[...] - previous` is an `int` subtraction:
            // the reader replays it with `wrapping_add`, so wrapping is what
            // round-trips a caller's absurd position pair.
            #[allow(clippy::arithmetic_side_effects)]
            {
                let mut pos = 0usize;
                for &freq in &f.freqs {
                    let mut previous = 0i32;
                    for _ in 0..freq {
                        let position = f.positions[pos];
                        deltas.push(i64::from(position.wrapping_sub(previous)));
                        previous = position;
                        pos += 1;
                    }
                }
            }
        }
    }
    if !deltas.is_empty() {
        out.write_bytes(&block_packed::encode_all(&deltas));
    }
}

/// `flushOffsets`: the per-field-number `charsPerTerm` ratio, then start
/// offsets corrected by it, then term lengths relative to the term's own
/// text length.
fn flush_offsets(out: &mut Vec<u8>, docs: &[PendingDoc], field_nums: &[i64]) {
    let mut sum_pos = vec![0i64; field_nums.len()];
    let mut sum_offsets = vec![0i64; field_nums.len()];
    let mut total_offsets = 0i64;
    for d in docs {
        for f in &d.fields {
            if f.flags & FLAG_OFFSETS == 0 {
                continue;
            }
            // ARITH: `total_offsets` accumulates `Vec` lengths, and
            // `sum_pos`/`sum_offsets` accumulate `i32`s widened to `i64`, one
            // per term of one chunk (at most 128 documents) -- all far inside
            // `i64`. `pos` advances by exactly `sum(freqs)`, which
            // `validate_field` pinned to `f.positions.len()`.
            //
            // The `freq - 1` is *not* provable and is guarded instead: a
            // caller may legally hand this writer a term with `freq == 0`
            // (`validate_field` accepts one, with empty occurrence arrays),
            // and `pos + 0 - 1` underflows when `pos` is 0. Java has the same
            // shape and throws `ArrayIndexOutOfBoundsException`; skipping the
            // term is what keeps `charsPerTerm` a ratio of the occurrences
            // that exist.
            #[allow(clippy::arithmetic_side_effects)]
            {
                total_offsets += f.start_offsets.len() as i64;
                if f.flags & FLAG_POSITIONS == 0 {
                    continue;
                }
                let off = field_num_off(field_nums, f.field_num);
                let mut pos = 0usize;
                for &freq in &f.freqs {
                    if freq > 0 {
                        let last = pos + freq as usize - 1;
                        sum_pos[off] += f.positions[last] as i64;
                        sum_offsets[off] += f.start_offsets[last] as i64;
                        pos += freq as usize;
                    }
                }
            }
        }
    }

    // Java returns early on `!hasOffsets`, i.e. "no field carries the OFFSETS
    // flag". Both readers -- Java's `get` and this port's `read_chunk` --
    // instead decide whether these streams are present from `totalOffsets >
    // 0`, a sum of term frequencies. The two conditions differ for exactly
    // one input Java's own writer can never produce and this port's public
    // `TermVectorsDocument` can: a field carrying OFFSETS with **no terms**.
    // Keying off the flag there would emit a `charsPerTerm` block the reader
    // never consumes, and every following byte of the chunk would decode as
    // garbage.
    if total_offsets == 0 {
        return;
    }

    let chars_per_term: Vec<f32> = (0..field_nums.len())
        .map(|i| {
            if sum_pos[i] <= 0 || sum_offsets[i] <= 0 {
                0.0
            } else {
                (sum_offsets[i] as f64 / sum_pos[i] as f64) as f32
            }
        })
        .collect();
    for &cpt in &chars_per_term {
        out.write_i32(cpt.to_bits() as i32);
    }

    let mut start_deltas: Vec<i64> = Vec::new();
    for d in docs {
        for f in &d.fields {
            if f.flags & FLAG_OFFSETS == 0 {
                continue;
            }
            let cpt = chars_per_term[field_num_off(field_nums, f.field_num)];
            let has_positions = f.flags & FLAG_POSITIONS != 0;
            // ARITH: `pos` walks exactly `sum(freqs)` occurrences, which
            // `validate_field` pinned to `f.start_offsets.len()` (and to
            // `f.positions.len()` when the field has positions). The three
            // `i32` differences are `wrapping_*` for Java parity: Java writes
            // `startOffset - previousOff - correction` as `int` arithmetic and
            // the reader replays it with `wrapping_add`, so wrapping is what
            // round-trips.
            #[allow(clippy::arithmetic_side_effects)]
            {
                let mut pos = 0usize;
                for &freq in &f.freqs {
                    let mut previous_pos = 0i32;
                    let mut previous_off = 0i32;
                    for _ in 0..freq {
                        let position = if has_positions { f.positions[pos] } else { 0 };
                        let start_offset = f.start_offsets[pos];
                        let correction = (cpt * position.wrapping_sub(previous_pos) as f32) as i32;
                        start_deltas.push(i64::from(
                            start_offset
                                .wrapping_sub(previous_off)
                                .wrapping_sub(correction),
                        ));
                        previous_pos = position;
                        previous_off = start_offset;
                        pos += 1;
                    }
                }
            }
        }
    }
    out.write_bytes(&block_packed::encode_all(&start_deltas));

    let mut lengths: Vec<i64> = Vec::new();
    for d in docs {
        for f in &d.fields {
            if f.flags & FLAG_OFFSETS == 0 {
                continue;
            }
            // ARITH: `pos` walks exactly `sum(freqs)` occurrences, pinned to
            // `f.lengths.len()` by `validate_field`, and `i` indexes the
            // per-term arrays `f.freqs` is parallel to. The subtraction is
            // `wrapping_*` for the same Java-parity reason as `start_deltas`:
            // the reader adds the term length back with `wrapping_add`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                let mut pos = 0usize;
                for (i, &freq) in f.freqs.iter().enumerate() {
                    for _ in 0..freq {
                        lengths.push(i64::from(
                            f.lengths[pos]
                                .wrapping_sub(f.prefix_lengths[i])
                                .wrapping_sub(f.suffix_lengths[i]),
                        ));
                        pos += 1;
                    }
                }
            }
        }
    }
    out.write_bytes(&block_packed::encode_all(&lengths));
}

/// `flushPayloadLengths`.
fn flush_payload_lengths(out: &mut Vec<u8>, docs: &[PendingDoc]) {
    let lengths: Vec<i64> = docs
        .iter()
        .flat_map(|d| d.fields.iter())
        .filter(|f| f.flags & FLAG_PAYLOADS != 0)
        .flat_map(|f| f.payload_lengths.iter().map(|&v| v as i64))
        .collect();
    if !lengths.is_empty() {
        out.write_bytes(&block_packed::encode_all(&lengths));
    }
}

/// Convenience wrapper over [`TermVectorsWriter`] for a whole document list
/// -- the shape this module exposed before it could chunk. Emits exactly
/// what feeding the same documents through
/// [`TermVectorsWriter::add_document`] one at a time does, so the segment is
/// chunked at Java's [`CHUNK_SIZE`]/[`MAX_DOCS_PER_CHUNK`] triggers, not
/// written as one chunk.
///
/// `merge.rs` uses the streaming writer directly (it needs
/// [`TermVectorsWriter::copy_chunks`]); this is for a flush, where every
/// document is materialised anyway.
pub fn write_best_speed(
    docs: &[TermVectorsDocument],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut writer = TermVectorsWriter::new(segment_id, segment_suffix);
    for doc in docs {
        writer.add_document(doc);
    }
    writer.finish()
}

/// A single, self-contained LZ4 "literal run" block wrapping `bytes`
/// verbatim -- same style as `stored_fields::encode_literal_lz4`, kept as an
/// independent copy since term vectors' LZ4 unit has no dict/block-length
/// wrapper (see [`open`]'s doc comment: it's a single plain LZ4 unit).
/// Test-only: the writer compresses through [`Compressor`].
#[cfg(test)]
fn encode_literal_lz4(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let len = bytes.len();
    let nibble = len.min(0x0F);
    out.push((nibble as u8) << 4);
    // ARITH: `len - 0x0F` runs only under `len >= 0x0F`, and `rem -= 0xFF`
    // only under `rem >= 0xFF`.
    #[allow(clippy::arithmetic_side_effects)]
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

struct FieldDecodeInput<'a> {
    field_number: i32,
    flags: u8,
    term_start: usize,
    term_count: usize,
    prefix_lengths: &'a [i64],
    suffix_lengths: &'a [i64],
    term_freqs_minus1: &'a [i64],
    suffix_bytes: &'a [u8],
    suffix_byte_start: usize,
    positions_flat: &'a [i64],
    position_start: usize,
    start_offsets_flat: &'a [i64],
    lengths_flat: &'a [i64],
    offset_start: usize,
    payload_bytes: &'a [u8],
    payload_lengths_flat: &'a [i64],
    payload_start: usize,
    payload_byte_start: usize,
    chars_per_term: f32,
}

/// Builds one field's fully-decoded terms. See the module doc for the
/// per-term (not per-field) delta-reset semantics of positions/offsets.
fn build_field(inp: FieldDecodeInput) -> Result<TermVectorField> {
    let has_positions = inp.flags & FLAG_POSITIONS != 0;
    let has_offsets = inp.flags & FLAG_OFFSETS != 0;
    let has_payloads = inp.flags & FLAG_PAYLOADS != 0;

    // Every length below comes off `.tvd`, and none may reach an index or a
    // `Vec::with_capacity` unbounded: a `prefixLength` longer than the term it
    // shares with would panic, and a `freq` or a `numTerms` a corrupt file
    // chose would ask for a multi-petabyte allocation, which **aborts** -- a
    // failure no `catch_unwind` at the FFI boundary can intercept. Re-signed
    // corrupt `.tvd` bodies reach this decoder as well-formed-but-wrong files.
    // The rule (`docs/arithmetic-gate.md`): a length off disk is
    // bounded by the bytes that must exist for it before it sizes anything.
    let bounded = |off: usize, len: usize, have: usize| -> Result<std::ops::Range<usize>> {
        let end = off
            .checked_add(len)
            .filter(|&end| end <= have)
            .ok_or(lucene_store::Error::Eof { offset: off })?;
        Ok(off..end)
    };

    let mut terms = Vec::with_capacity(inp.term_count.min(inp.prefix_lengths.len()));
    let mut previous_term: Vec<u8> = Vec::new();
    let mut suffix_byte_off = inp.suffix_byte_start;
    let mut position_off = inp.position_start;
    let mut offset_off = inp.offset_start;
    let mut payload_off = inp.payload_start;
    let mut payload_byte_off = inp.payload_byte_start;

    for j in 0..inp.term_count {
        let idx = inp.term_start.saturating_add(j);
        let at = |flat: &[i64]| -> Result<i64> {
            flat.get(idx)
                .copied()
                .ok_or(lucene_store::Error::Eof { offset: idx }.into())
        };
        // No `.max(0)` and no `saturating_*` here: `read_chunk` rejected a
        // negative prefix, suffix or payload length and any `freqMinus1` for
        // which `+ 1` would overflow or go negative, so these three casts are
        // exact rather than clamped. Folding a negative away instead would
        // turn a rejection into a plausible wrong term, which is the failure
        // `docs/arithmetic-gate.md` singles out saturation for.
        let raw_prefix = at(inp.prefix_lengths)?;
        let raw_suffix = at(inp.suffix_lengths)?;
        let raw_freq_minus1 = at(inp.term_freqs_minus1)?;
        debug_assert!(
            raw_prefix >= 0 && raw_suffix >= 0 && raw_freq_minus1 >= -1,
            "read_chunk validates prefix/suffix lengths and frequencies"
        );
        let prefix_len = raw_prefix as usize;
        let suffix_len = raw_suffix as usize;
        let freq = raw_freq_minus1.saturating_add(1) as usize;

        // `prefix_len` indexes the *previously decoded* term, which is a
        // completely different part of the file, so a corrupt `.tvd` can name
        // a prefix longer than the term it shares with.
        let prefix = previous_term.get(..prefix_len).ok_or_else(|| {
            lucene_store::Error::Corrupted(format!(
                "term vector prefixLength={prefix_len} exceeds the {} byte previous term",
                previous_term.len()
            ))
        })?;
        let suffix = inp
            .suffix_bytes
            .get(bounded(
                suffix_byte_off,
                suffix_len,
                inp.suffix_bytes.len(),
            )?)
            .ok_or(lucene_store::Error::Eof {
                offset: suffix_byte_off,
            })?;
        // Sized from the two slices that already exist, never from the two
        // lengths off disk.
        let mut term = Vec::with_capacity(prefix.len().saturating_add(suffix.len()));
        term.extend_from_slice(prefix);
        term.extend_from_slice(suffix);
        // ARITH: `bounded` above already computed `suffix_byte_off +
        // suffix_len` with a `checked_add` and rejected it unless it landed
        // inside `suffix_bytes`, so repeating it here cannot overflow.
        #[allow(clippy::arithmetic_side_effects)]
        {
            suffix_byte_off += suffix_len;
        }
        let term_len = term.len() as i32;

        // `freq` sizes three buffers and drives three walks. Slicing the flat
        // streams up front is what bounds it: a `freq` past the end of the
        // stream it indexes is an `Eof` here rather than an abort below.
        let positions_flat = if has_positions {
            &inp.positions_flat[bounded(position_off, freq, inp.positions_flat.len())?]
        } else {
            &[][..]
        };
        let start_offsets_flat = if has_offsets {
            &inp.start_offsets_flat[bounded(offset_off, freq, inp.start_offsets_flat.len())?]
        } else {
            &[][..]
        };
        let lengths_flat = if has_offsets {
            &inp.lengths_flat[bounded(offset_off, freq, inp.lengths_flat.len())?]
        } else {
            &[][..]
        };
        let payload_lengths_flat = if has_payloads {
            &inp.payload_lengths_flat[bounded(payload_off, freq, inp.payload_lengths_flat.len())?]
        } else {
            &[][..]
        };

        // Positions: absolute at this term's first occurrence, delta from
        // the *same term*'s previous occurrence thereafter.
        // `Vec::new()` (not `with_capacity`) when the stream is absent: an
        // empty `Vec` never allocates, so a positions-only field pays for one
        // buffer per term rather than three.
        let mut term_positions = if has_positions {
            Vec::with_capacity(freq)
        } else {
            Vec::new()
        };
        if has_positions {
            let mut absolute = 0i32;
            for (k, &raw) in positions_flat.iter().enumerate() {
                let raw = raw as i32;
                absolute = if k == 0 {
                    raw
                } else {
                    absolute.wrapping_add(raw)
                };
                term_positions.push(absolute);
            }
        }

        let (mut term_start_offsets, mut term_end_offsets) = if has_offsets {
            (Vec::with_capacity(freq), Vec::with_capacity(freq))
        } else {
            (Vec::new(), Vec::new())
        };
        if has_offsets {
            let mut absolute = 0i32;
            for (k, &raw_delta) in start_offsets_flat.iter().enumerate() {
                let raw_delta = raw_delta as i32;
                let position_correction = if has_positions {
                    (inp.chars_per_term * positions_flat[k] as f32) as i32
                } else {
                    0
                };
                let patched = raw_delta.wrapping_add(position_correction);
                absolute = if k == 0 {
                    patched
                } else {
                    absolute.wrapping_add(patched)
                };
                let length = (lengths_flat[k] as i32).wrapping_add(term_len);
                term_start_offsets.push(absolute);
                term_end_offsets.push(absolute.wrapping_add(length));
            }
            // ARITH: `bounded(offset_off, freq, ...)` succeeded above for
            // both `start_offsets_flat` and `lengths_flat`, which means
            // `offset_off + freq` was computed with `checked_add` and fits.
            #[allow(clippy::arithmetic_side_effects)]
            {
                offset_off += freq;
            }
        }
        // ARITH: likewise `bounded(position_off, freq, positions_flat.len())`.
        #[allow(clippy::arithmetic_side_effects)]
        if has_positions {
            position_off += freq;
        }

        let mut term_payloads = if has_payloads {
            Vec::with_capacity(freq)
        } else {
            Vec::new()
        };
        if has_payloads {
            for &raw_len in payload_lengths_flat {
                debug_assert!(raw_len >= 0, "read_chunk validates payload lengths");
                let len = raw_len as usize;
                let bytes =
                    &inp.payload_bytes[bounded(payload_byte_off, len, inp.payload_bytes.len())?];
                term_payloads.push(bytes.to_vec());
                // ARITH: `bounded` just checked `payload_byte_off + len`
                // against `payload_bytes.len()` with a `checked_add`.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    payload_byte_off += len;
                }
            }
            // ARITH: `bounded(payload_off, freq, payload_lengths_flat.len())`
            // succeeded above.
            #[allow(clippy::arithmetic_side_effects)]
            {
                payload_off += freq;
            }
        }

        previous_term = term.clone();
        terms.push(TermVectorTerm {
            term,
            freq: freq as i32,
            positions: has_positions.then_some(term_positions),
            start_offsets: has_offsets.then_some(term_start_offsets),
            end_offsets: has_offsets.then_some(term_end_offsets),
            payloads: has_payloads.then_some(term_payloads),
        });
    }

    Ok(TermVectorField {
        field_number: inp.field_number,
        has_positions,
        has_offsets,
        has_payloads,
        terms,
    })
}

/// ARITH: `skip(len)` returns `Eof` unless `len <= input.remaining()`, so on
/// the success path `start + len` is exactly the new cursor position and is
/// therefore at most `input.len()`. (This is also what bounds a *negative*
/// vint: `as usize` sign-extends it to ~2^64, which no `remaining()` can
/// cover, so `skip` rejects it before the addition happens.)
#[allow(clippy::arithmetic_side_effects)]
fn read_length_prefixed_slice<'a>(input: &mut SliceInput<'a>) -> Result<&'a [u8]> {
    let len = input.read_vint()? as usize;
    let start = input.position();
    input.skip(len)?;
    Ok(input.slice(start, start + len)?)
}

#[cfg(test)]
mod tests;
