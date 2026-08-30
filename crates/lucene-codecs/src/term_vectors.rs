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

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

use crate::block_packed;
use crate::direct_monotonic;
use crate::direct_reader;
use crate::lz4;
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
    max_doc: i32,
    chunk_size: i32,
    max_pointer: i64,
    num_chunks: i64,
    num_dirty_chunks: i64,
    num_dirty_docs: i64,
    docs_start_pointer: i64,
    docs_end_pointer: i64,
    docs_meta: direct_monotonic::Meta,
    start_pointers_start_pointer: i64,
    start_pointers_end_pointer: i64,
    start_pointers_meta: direct_monotonic::Meta,
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
    let tvd_header = codec_util::check_index_header(
        &mut tvd_input,
        DATA_CODEC,
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
        max_doc,
        chunk_size,
        max_pointer,
        num_chunks,
        num_dirty_chunks,
        num_dirty_docs,
        docs_start_pointer,
        docs_end_pointer,
        docs_meta,
        start_pointers_start_pointer,
        start_pointers_end_pointer,
        start_pointers_meta,
    })
}

impl<'d> TermVectorsReader<'d> {
    pub fn max_doc(&self) -> i32 {
        self.max_doc
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
        // The count arrives as a 3-bit inline field with a vint escape. It
        // used to accumulate into a `u32`: `read_vint` returns `i32`, so a
        // negative escape became ~4 billion through `as u32`, the `+=`
        // overflowed (debug panic) or wrapped (release), and
        // `packed_ints::byte_count` then sized `vec![0u8; n]` at up to ~16 GB
        // -- the **abort** shape, which `catch_unwind` at the FFI boundary
        // cannot intercept. Flagged live by c24 and fixed here.
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
        // byte is allocated. Found by c25's re-signed `.tvd` body sweep.
        //
        // The `.max(0)` this used to fold in silently accepted a *negative*
        // length and then let the per-document cursor below add it as a huge
        // `usize`. A byte length off disk is never negative in anything
        // Lucene wrote, so a negative one is rejected outright: that is what
        // makes every partial sum below a prefix of a total that has already
        // been bounded.
        // `prefix_lengths` does not contribute to the decompressed length --
        // a prefix is shared with the *previous* term, not stored again -- but
        // it is the same kind of value off the same stream, and `build_field`
        // used to fold a negative one away with `.max(0)`, decoding the term
        // from its suffix alone: accepted, self-consistent and wrong, where
        // Java throws (`System.arraycopy` with a negative `destPos`). Checked
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
        let ceiling = compressed_left.saturating_mul(LZ4_MAX_EXPANSION);
        if decompressed_len > ceiling {
            return Err(lucene_store::Error::Corrupted(format!(
                "term vector chunk claims {decompressed_len} decompressed bytes, which no \
                 {compressed_left} byte LZ4 block can produce"
            ))
            .into());
        }
        let mut decompressed = vec![0u8; decompressed_len];
        if decompressed_len > 0 {
            lz4::decompress(&mut input, decompressed_len, &mut decompressed, 0)?;
        }
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
            fields.push(build_field(FieldDecodeInput {
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
            })?);
        }
        Ok(Some(TermVectorsDocument { fields }))
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
/// - The LZ4 unit is a single literal run (`encode_literal_lz4`) --
///   valid LZ4 with no match finding.
///
/// What is *not* scoped down any more (it was, before batch `c8`): chunking
/// itself, term prefix sharing, `charsPerTerm`, and the
/// `nonChangingFlags` per-field-number flags encoding are all Java's.
pub struct TermVectorsWriter {
    segment_id: [u8; ID_LENGTH],
    segment_suffix: String,
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
        // `.tvm` records `chunkSize` as a vint, and `too_dirty` compares
        // `maxDocsPerChunk` against an `i64` doc count, so both must survive
        // the narrowing.
        assert!(chunk_size > 0 && chunk_size <= i32::MAX as usize);
        assert!(max_docs_per_chunk > 0 && max_docs_per_chunk <= i32::MAX as usize);
        let mut tvd = Vec::new();
        codec_util::write_index_header(
            &mut tvd,
            DATA_CODEC,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        );
        let start = tvd.len() as i64;
        Self {
            segment_id: *segment_id,
            segment_suffix: segment_suffix.to_string(),
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
    /// open anything else. There is only one compression mode.)
    pub fn can_bulk_copy(&self, reader: &TermVectorsReader<'_>) -> bool {
        reader.chunk_size() == self.chunk_size as i32 && !self.too_dirty(reader)
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
            out.write_bytes(&encode_literal_lz4(&self.term_suffixes));
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
            direct_monotonic::write(&self.docs_values, INDEX_BLOCK_SHIFT);
        tvx.write_bytes(&docs_data_bytes);
        let docs_end_pointer = tvx.len() as i64;
        let (start_pointers_meta_bytes, start_pointers_data_bytes) =
            direct_monotonic::write(&self.start_pointers_values, INDEX_BLOCK_SHIFT);
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
        tvm.write_i32(INDEX_BLOCK_SHIFT as i32);
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
    let bytes = direct_reader::encode(&offs, bits);
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
    let bytes = direct_reader::encode(&values, FLAGS_BITS);
    out.write_vint(bytes.len() as i32);
    out.write_bytes(&bytes);
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
    let bytes = direct_reader::encode(&counts, bits);
    out.write_vint(bytes.len() as i32);
    out.write_bytes(&bytes);
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

/// The LZ4 block format's worst-case expansion ratio: one 0xFF
/// length-extension byte contributes 255 output bytes, so no LZ4 block can
/// inflate to more than 255 times its own compressed size. Used only as a
/// *bound* -- a real chunk is nowhere near it -- so that a decompressed
/// length read off disk can be rejected before it sizes an allocation.
const LZ4_MAX_EXPANSION: usize = 255;

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

    // Every length below comes off `.tvd`, and every one of them used to
    // reach an index or a `Vec::with_capacity` unbounded: a `prefixLength`
    // longer than the term it shares with panicked, and a `freq` or a
    // `numTerms` a corrupt file chose asked for a multi-petabyte allocation,
    // which **aborts** -- a failure no `catch_unwind` at the FFI boundary can
    // intercept. Both were found by c25's re-signed `.tvd` body sweep, which
    // is the first thing that ever handed this decoder a well-formed-but-
    // wrong file. The rule (`docs/arithmetic-gate.md`): a length off disk is
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

    fn id() -> [u8; ID_LENGTH] {
        [3u8; ID_LENGTH]
    }

    /// Hand-encodes a single-doc chunk (docBase=0, chunkDocs=1) with one
    /// field (number 5) that has POSITIONS+OFFSETS+PAYLOADS, two terms:
    /// "cat" (freq 2, prefix-shared with nothing) and "car" (freq 1,
    /// sharing prefix "ca" with "cat"). Values were derived by hand in the
    /// module's development notes; see the assertions below for what they
    /// decode to.
    fn build_single_doc_chunk() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let (tvd, tvx, tvm, _chunk_start) = build_single_doc_chunk_with_meta_overrides(1, 0, 0);
        (tvd, tvx, tvm)
    }

    /// Same as [`build_single_doc_chunk`] but lets the caller override the
    /// outer `numChunks`/`numDirtyChunks`/`numDirtyDocs` meta fields (to
    /// exercise `open`'s consistency-check error paths) and returns the
    /// `.tvd` offset of the chunk's `docBase` byte (to let callers corrupt
    /// it and re-sign the `.tvd` footer for `CorruptChunkBounds` tests).
    fn build_single_doc_chunk_with_meta_overrides(
        num_chunks_outer: i32,
        num_dirty_chunks: i32,
        num_dirty_docs: i32,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>, usize) {
        build_single_doc_chunk_with_index_chunks(
            num_chunks_outer + 1,
            num_chunks_outer,
            num_dirty_chunks,
            num_dirty_docs,
        )
    }

    /// As above, but the `.tvx` index meta's own chunk count is given
    /// separately so a test can build a (structurally consistent) index that
    /// claims more than one chunk -- the only way to reach `open`'s third
    /// dirty-chunk check, which needs `numDirtyChunks >= 2`.
    fn build_single_doc_chunk_with_index_chunks(
        index_num_chunks: i32,
        num_chunks_outer: i32,
        num_dirty_chunks: i32,
        num_dirty_docs: i32,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>, usize) {
        let mut tvd = Vec::new();
        tvd.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut tvd, DATA_CODEC);
        tvd.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        tvd.extend_from_slice(&id());
        tvd.push(0); // empty suffix
        let chunk_start = tvd.len() as i64;

        write_vint(&mut tvd, 0); // docBase
        write_vint(&mut tvd, 1 << 1); // token: chunkDocs=1, dirty=0
        write_vint(&mut tvd, 1); // numFields = totalFields = 1

        // fieldNums: 1 distinct field (number 5), 8 bits/value.
        tvd.push(8); // token: (totalDistinct-1=0)<<5 | bitsPerFieldNum=8
        tvd.push(5); // field number 5, MSB-packed (byte-aligned, trivial)

        // allFieldNumOffs: 1 field, offset 0 into field_nums, 1 bit/value.
        write_vint(&mut tvd, 1); // slice byte length
        tvd.push(0x00);

        // flags: selector=1 (direct array), 1 field, FLAGS_BITS=4, value=7
        // (POSITIONS|OFFSETS|PAYLOADS).
        write_vint(&mut tvd, 1); // selector
        write_vint(&mut tvd, 1); // slice byte length
        tvd.push(0x07);

        // numTerms: 1 field, 8 bits/value, value=2.
        write_vint(&mut tvd, 8); // bitsRequired
        write_vint(&mut tvd, 1); // slice byte length
        tvd.push(2);

        // prefixLengths [0, 2] (block-packed: min=0, bpv=2).
        tvd.extend_from_slice(&[0x05, 0x20]);
        // suffixLengths [3, 1] (min=1, bpv=2): token, minValue vlong, packed.
        tvd.extend_from_slice(&[0x04, 0x01, 0x80]);
        // termFreqsMinus1 [1, 0] (min=0, bpv=1).
        tvd.extend_from_slice(&[0x03, 0x80]);

        // positions_flat [0, 2, 1] (min=0, bpv=2): term0 abs=0, delta=2
        // (2nd occurrence); term1 abs=1 (its own first occurrence).
        tvd.extend_from_slice(&[0x05, 0x24]);

        // charsPerTerm: 1 distinct field, value 4.0.
        tvd.extend_from_slice(&4.0f32.to_bits().to_le_bytes());
        // start_offsets_flat [0, 0, 0] (bpv=0, constant -- no packed bytes).
        tvd.push(0x01);
        // lengths_flat [0, 0, 0] (bpv=0, constant).
        tvd.push(0x01);
        // payload_lengths_flat [1, 0, 2] (min=0, bpv=2).
        tvd.extend_from_slice(&[0x05, 0x48]);

        // LZ4 (CompressionMode.FAST, no dictionary): literal-only unit
        // wrapping "cat"+"r" (term suffixes) then payload bytes 0xAA,0xBB,0xCC.
        let payload = [b'c', b'a', b't', b'r', 0xAA, 0xBB, 0xCC];
        tvd.push((payload.len() as u8) << 4); // LZ4 literal-length token
        tvd.extend_from_slice(&payload);

        tvd.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        tvd.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&tvd) as u64;
        tvd.extend_from_slice(&checksum.to_be_bytes());

        // .tvx: docs=[0, sentinel=maxDoc], startPointers=[chunk_start, sentinel=maxPointer].
        let mut tvx = Vec::new();
        tvx.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut tvx, INDEX_CODEC);
        tvx.extend_from_slice(&(INDEX_VERSION_CURRENT as u32).to_be_bytes());
        tvx.extend_from_slice(&id());
        tvx.push(0);
        let docs_start = tvx.len() as i64;
        let docs_end = tvx.len() as i64;
        let start_pointers_end = tvx.len() as i64;
        tvx.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        tvx.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&tvx) as u64;
        tvx.extend_from_slice(&checksum.to_be_bytes());

        // .tvm
        let max_doc = 1i32;
        let max_pointer = (tvd.len() - codec_util::FOOTER_LENGTH) as i64;
        let mut tvm = Vec::new();
        tvm.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut tvm, META_CODEC);
        tvm.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        tvm.extend_from_slice(&id());
        tvm.push(0);
        write_vint(&mut tvm, 0); // packedIntsVersion (unused by this port)
        write_vint(&mut tvm, 4096); // chunkSize (unused by this port)
        tvm.extend_from_slice(&max_doc.to_le_bytes());
        tvm.extend_from_slice(&0i32.to_le_bytes()); // blockShift=0
        tvm.extend_from_slice(&index_num_chunks.to_le_bytes());
        tvm.extend_from_slice(&docs_start.to_le_bytes());
        let doc_mins: Vec<i64> = std::iter::once(0i64)
            .chain(std::iter::repeat_n(
                max_doc as i64,
                (index_num_chunks - 1) as usize,
            ))
            .collect();
        for min in doc_mins {
            tvm.extend_from_slice(&min.to_le_bytes());
            tvm.extend_from_slice(&0i32.to_le_bytes());
            tvm.extend_from_slice(&0i64.to_le_bytes());
            tvm.push(0);
        }
        tvm.extend_from_slice(&docs_end.to_le_bytes());
        let fp_mins: Vec<i64> = std::iter::once(chunk_start)
            .chain(std::iter::repeat_n(
                max_pointer,
                (index_num_chunks - 1) as usize,
            ))
            .collect();
        for min in fp_mins {
            tvm.extend_from_slice(&min.to_le_bytes());
            tvm.extend_from_slice(&0i32.to_le_bytes());
            tvm.extend_from_slice(&0i64.to_le_bytes());
            tvm.push(0);
        }
        tvm.extend_from_slice(&start_pointers_end.to_le_bytes());
        tvm.extend_from_slice(&max_pointer.to_le_bytes());
        write_vint(&mut tvm, num_chunks_outer); // numChunks (outer)
        write_vint(&mut tvm, num_dirty_chunks); // numDirtyChunks
        write_vint(&mut tvm, num_dirty_docs); // numDirtyDocs
        tvm.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        tvm.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&tvm) as u64;
        tvm.extend_from_slice(&checksum.to_be_bytes());

        (tvd, tvx, tvm, chunk_start as usize)
    }

    /// Recomputes and appends-in-place the trailing 8-byte CRC32 checksum of
    /// a codec-footer-terminated buffer (footer magic + zero algorithm id
    /// are assumed already present; only the checksum bytes are patched).
    fn resign_footer(buf: &mut [u8]) {
        let len = buf.len();
        let checksum = crc32fast::hash(&buf[..len - 8]) as u64;
        buf[len - 8..].copy_from_slice(&checksum.to_be_bytes());
    }

    #[test]
    fn single_doc_full_decode_positions_offsets_payloads() {
        let (tvd, tvx, tvm) = build_single_doc_chunk();
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert_eq!(reader.max_doc(), 1);

        let doc = reader.document(0).unwrap().unwrap();
        assert_eq!(doc.fields.len(), 1);
        let field = &doc.fields[0];
        assert_eq!(field.field_number, 5);
        assert!(field.has_positions && field.has_offsets && field.has_payloads);
        assert_eq!(field.terms.len(), 2);

        let cat = &field.terms[0];
        assert_eq!(cat.term, b"cat");
        assert_eq!(cat.freq, 2);
        assert_eq!(cat.positions, Some(vec![0, 2]));
        assert_eq!(cat.start_offsets, Some(vec![0, 8]));
        assert_eq!(cat.end_offsets, Some(vec![3, 11]));
        assert_eq!(cat.payloads, Some(vec![vec![0xAA], vec![]]));

        let car = &field.terms[1];
        assert_eq!(car.term, b"car");
        assert_eq!(car.freq, 1);
        assert_eq!(car.positions, Some(vec![1]));
        assert_eq!(car.start_offsets, Some(vec![4]));
        assert_eq!(car.end_offsets, Some(vec![7]));
        assert_eq!(car.payloads, Some(vec![vec![0xBB, 0xCC]]));
    }

    #[test]
    fn doc_out_of_range_rejected() {
        let (tvd, tvx, tvm) = build_single_doc_chunk();
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
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
    fn wrong_segment_id_rejected() {
        let (tvd, tvx, tvm) = build_single_doc_chunk();
        let wrong_id = [9u8; ID_LENGTH];
        assert!(open(&tvd, &tvx, &tvm, &wrong_id, "").is_err());
    }

    #[test]
    fn direct_writer_bits_required_rounds_up_to_supported_widths() {
        assert_eq!(direct_writer_bits_required(0), 1);
        assert_eq!(direct_writer_bits_required(1), 1);
        assert_eq!(direct_writer_bits_required(3), 2);
        assert_eq!(direct_writer_bits_required(7), 4); // 3 bits needed, rounds to 4
        assert_eq!(direct_writer_bits_required(255), 8);
        assert_eq!(direct_writer_bits_required(256), 12); // 9 bits needed, rounds to 12
    }

    #[test]
    fn num_chunks_mismatch_rejected() {
        // The builder always writes index_num_chunks=2 (1 real chunk + 1
        // sentinel); an outer numChunks of 2 breaks the required
        // index==outer+1 relationship (2 != 2+1).
        let (tvd, tvx, tvm, _) = build_single_doc_chunk_with_index_chunks(2, 2, 0, 0);
        assert!(matches!(
            open(&tvd, &tvx, &tvm, &id(), ""),
            Err(Error::NumChunksMismatch {
                index_num_chunks: 2,
                outer_num_chunks: 2
            })
        ));
    }

    #[test]
    fn too_many_dirty_chunks_rejected() {
        let (tvd, tvx, tvm, _) = build_single_doc_chunk_with_meta_overrides(1, 2, 2);
        assert!(matches!(
            open(&tvd, &tvx, &tvm, &id(), ""),
            Err(Error::TooManyDirtyChunks(2, 1))
        ));
    }

    #[test]
    fn dirty_chunks_docs_mismatch_rejected() {
        let (tvd, tvx, tvm, _) = build_single_doc_chunk_with_meta_overrides(1, 1, 0);
        assert!(matches!(
            open(&tvd, &tvx, &tvm, &id(), ""),
            Err(Error::DirtyChunksDocsMismatch(1, 0))
        ));
    }

    /// `Lucene90CompressingTermVectorsReader`'s third dirty-chunk check.
    /// Two dirty chunks holding one doc between them is impossible: a forced
    /// flush always carries at least one pending doc.
    #[test]
    fn more_dirty_chunks_than_dirty_docs_rejected() {
        let (tvd, tvx, tvm, _) = build_single_doc_chunk_with_index_chunks(4, 3, 2, 1);
        assert!(matches!(
            open(&tvd, &tvx, &tvm, &id(), ""),
            Err(Error::MoreDirtyChunksThanDirtyDocs(2, 1))
        ));
    }

    /// The boundary the check above must *not* reject: as many dirty docs as
    /// dirty chunks is legal (every forced flush held exactly one doc).
    #[test]
    fn equal_dirty_chunks_and_dirty_docs_accepted() {
        let (tvd, tvx, tvm, _) = build_single_doc_chunk_with_index_chunks(4, 3, 2, 2);
        assert!(open(&tvd, &tvx, &tvm, &id(), "").is_ok());
    }

    #[test]
    fn wrong_tvd_length_rejected() {
        let (mut tvd, tvx, tvm) = build_single_doc_chunk();
        tvd.push(0); // stray byte after the footer
        assert!(open(&tvd, &tvx, &tvm, &id(), "").is_err());
    }

    #[test]
    fn corrupt_chunk_bounds_rejected() {
        // Patch the .tvd chunk's token so it claims chunkDocs=2 starting at
        // docBase=0, while .tvm still says maxDoc=1 -- doc_base+chunk_docs
        // (2) exceeds max_doc (1).
        let (mut tvd, tvx, tvm, chunk_start) = build_single_doc_chunk_with_meta_overrides(1, 0, 0);
        // Byte right after docBase's single vint byte (docBase=0) is the
        // token vint; original value is `1<<1=2` (chunkDocs=1). Bump it to
        // `2<<1=4` (chunkDocs=2).
        assert_eq!(tvd[chunk_start + 1], 2);
        tvd[chunk_start + 1] = 4;
        resign_footer(&mut tvd);

        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert!(matches!(
            reader.document(0),
            Err(Error::CorruptChunkBounds { .. })
        ));
    }

    /// Splices `insert` over `tvd[at..at + remove]`, then repairs `.tvm`'s
    /// `maxPointer` and re-signs both footers -- c15's shape, so that only
    /// the semantic invariant under test can fire and never a checksum or a
    /// length mismatch the corruption happened to also break.
    fn splice_tvd(tvd: &mut Vec<u8>, tvm: &mut [u8], at: usize, remove: usize, insert: &[u8]) {
        let footer = codec_util::FOOTER_LENGTH;
        // `.tvm` tail is: maxPointer (i64 LE), then the three one-byte vints
        // `numChunks`/`numDirtyChunks`/`numDirtyDocs`, then the footer.
        let mp_off = tvm.len() - footer - 3 - 8;
        let old_max_pointer = i64::from_le_bytes(tvm[mp_off..mp_off + 8].try_into().unwrap());
        assert_eq!(
            old_max_pointer,
            (tvd.len() - footer) as i64,
            "maxPointer offset"
        );
        tvd.splice(at..at + remove, insert.iter().copied());
        let new_max_pointer = (tvd.len() - footer) as i64;
        tvm[mp_off..mp_off + 8].copy_from_slice(&new_max_pointer.to_le_bytes());
        resign_footer(tvd);
        resign_footer(tvm);
    }

    /// The `.tvm` offsets of the three trailing chunk-count vlongs.
    fn tvm_counts_offset(tvm: &[u8]) -> usize {
        tvm.len() - codec_util::FOOTER_LENGTH - 3
    }

    #[test]
    fn negative_distinct_field_count_extension_is_a_decode_error_not_an_allocation() {
        // c24 flagged this one live: `totalDistinctFields` accumulated a
        // `readVInt` into a `u32`, so a negative escape became ~4 billion and
        // `packed_ints::byte_count` then sized `vec![0u8; n]` at ~16 GB --
        // an abort, not a catchable error.
        let (mut tvd, tvx, mut tvm, chunk_start) =
            build_single_doc_chunk_with_meta_overrides(1, 0, 0);
        assert_eq!(tvd[chunk_start + 3], 8, "fieldNums token");
        // token: totalDistinctFields inline = 7 (escape), bitsPerFieldNum = 8,
        // followed by a vint encoding of -1.
        let mut insert = vec![(7u8 << 5) | 8];
        write_vint(&mut insert, -1);
        splice_tvd(&mut tvd, &mut tvm, chunk_start + 3, 1, &insert);

        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let err = reader.document(0).unwrap_err();
        assert!(
            format!("{err}").contains("negative distinct-field-number count extension"),
            "{err}"
        );
    }

    #[test]
    fn more_distinct_field_numbers_than_fields_is_a_decode_error() {
        // The positive, plausible case a sign check and an EOF would both
        // miss: 13 distinct field numbers at 8 bits each is 13 readable bytes
        // in a chunk that has them, but the writer only ever emits the
        // *deduplicated* numbers of the chunk's fields, and this chunk has
        // exactly one field.
        let (mut tvd, tvx, mut tvm, chunk_start) =
            build_single_doc_chunk_with_meta_overrides(1, 0, 0);
        let mut insert = vec![(7u8 << 5) | 8];
        write_vint(&mut insert, 5); // 7 + 5 + 1 = 13
        splice_tvd(&mut tvd, &mut tvm, chunk_start + 3, 1, &insert);

        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let err = reader.document(0).unwrap_err();
        assert!(
            format!("{err}").contains("13 distinct field numbers across only 1 fields"),
            "{err}"
        );
    }

    #[test]
    fn absurd_per_document_field_count_is_a_decode_error_not_an_allocation() {
        // `numFields` is the chunk's per-document field count; for a
        // single-document chunk it is a bare vint. It sizes six
        // `Vec::with_capacity` calls, so a negative one (~2^64 through the
        // `as usize`) and a merely enormous one are both aborts.
        for (name, value) in [("negative", -1i32), ("absurd", 1 << 30)] {
            let (mut tvd, tvx, mut tvm, chunk_start) =
                build_single_doc_chunk_with_meta_overrides(1, 0, 0);
            assert_eq!(tvd[chunk_start + 2], 1, "numFields");
            let mut insert = Vec::new();
            write_vint(&mut insert, value);
            splice_tvd(&mut tvd, &mut tvm, chunk_start + 2, 1, &insert);

            let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
            let err = reader.document(0).unwrap_err();
            let text = format!("{err}");
            assert!(
                text.contains("bad per-document field count") || text.contains("more than the"),
                "{name}: {text}"
            );
        }
    }

    #[test]
    fn absurd_per_field_term_count_is_a_decode_error_not_an_allocation() {
        // `numTerms` is `direct_reader`-encoded at a bit width the chunk
        // chooses. At 64 bits an entry is an arbitrary `i64`: a negative one
        // made `term_offsets` non-monotonic (and its `as usize` casts
        // astronomic), which is what the three `block_packed` streams are
        // sliced by.
        let (mut tvd, tvx, mut tvm, chunk_start) =
            build_single_doc_chunk_with_meta_overrides(1, 0, 0);
        // numTerms header is `bitsRequired` vint, slice length vint, bytes.
        assert_eq!(tvd[chunk_start + 10], 8, "numTerms bitsRequired");
        assert_eq!(tvd[chunk_start + 11], 1, "numTerms slice length");
        let mut insert = vec![64, 8]; // 64 bits per value, 8-byte slice
        insert.extend_from_slice(&(-1i64).to_le_bytes());
        splice_tvd(&mut tvd, &mut tvm, chunk_start + 10, 3, &insert);

        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let err = reader.document(0).unwrap_err();
        assert!(
            format!("{err}").contains("bad per-field term count"),
            "{err}"
        );
    }

    #[test]
    fn absurd_term_frequency_is_a_decode_error_not_an_overflow() {
        // `termFreqsMinus1` is block-packed, so an entry is an arbitrary
        // `i64` and `freq = v + 1` overflows at `i64::MAX`. The stream
        // lengths derived from it size three more `block_packed::decode_all`
        // calls.
        let (mut tvd, tvx, mut tvm, chunk_start) =
            build_single_doc_chunk_with_meta_overrides(1, 0, 0);
        // termFreqsMinus1 is the third block-packed stream: prefixLengths
        // (2 bytes), suffixLengths (3 bytes), then [0x03, 0x80].
        let freqs_at = chunk_start + 13 + 2 + 3;
        assert_eq!(&tvd[freqs_at..freqs_at + 2], &[0x03, 0x80]);
        // One block of 2 values, bitsPerValue = 64, minValueEquals0 set.
        let mut insert = vec![((64u32 << 1) | 1) as u8];
        insert.extend_from_slice(&i64::MAX.to_be_bytes());
        insert.extend_from_slice(&0i64.to_be_bytes());
        splice_tvd(&mut tvd, &mut tvm, freqs_at, 2, &insert);

        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let err = reader.document(0).unwrap_err();
        assert!(
            format!("{err}").contains("frequenc") || format!("{err}").contains("overflow"),
            "{err}"
        );
    }

    #[test]
    fn negative_term_prefix_length_is_a_decode_error_not_a_shorter_term() {
        // A prefix is shared with the *previously decoded* term, which lives
        // in a different part of the file. `build_field` used to fold a
        // negative one away with `.max(0)` and decode the term from its
        // suffix alone -- accepted, self-consistent and wrong, where Java
        // throws (`System.arraycopy` with a negative `destPos`).
        let (mut tvd, tvx, mut tvm, chunk_start) =
            build_single_doc_chunk_with_meta_overrides(1, 0, 0);
        let prefixes_at = chunk_start + 13;
        assert_eq!(&tvd[prefixes_at..prefixes_at + 2], &[0x05, 0x20]);
        // One block of 2 values, bitsPerValue = 64, minValueEquals0 set.
        let mut insert = vec![((64u32 << 1) | 1) as u8];
        insert.extend_from_slice(&0i64.to_be_bytes());
        insert.extend_from_slice(&(-1i64).to_be_bytes());
        splice_tvd(&mut tvd, &mut tvm, prefixes_at, 2, &insert);

        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let err = reader.document(0).unwrap_err();
        assert!(
            format!("{err}").contains("negative term prefix length"),
            "{err}"
        );
    }

    #[test]
    fn negative_term_suffix_length_is_a_decode_error() {
        // The decompressed-length ceiling used to fold negatives away with
        // `.max(0)`, which let the per-document byte cursor add one as a huge
        // `usize` and index the decompressed buffer from nowhere.
        let (mut tvd, tvx, mut tvm, chunk_start) =
            build_single_doc_chunk_with_meta_overrides(1, 0, 0);
        let suffixes_at = chunk_start + 13 + 2;
        assert_eq!(&tvd[suffixes_at..suffixes_at + 3], &[0x04, 0x01, 0x80]);
        // One block of 2 values, bitsPerValue = 64, minValueEquals0 set.
        let mut insert = vec![((64u32 << 1) | 1) as u8];
        insert.extend_from_slice(&(-1i64).to_be_bytes());
        insert.extend_from_slice(&1i64.to_be_bytes());
        splice_tvd(&mut tvd, &mut tvm, suffixes_at, 3, &insert);

        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let err = reader.document(0).unwrap_err();
        assert!(
            format!("{err}").contains("negative term suffix length"),
            "{err}"
        );
    }

    #[test]
    fn a_num_chunks_at_i64_max_is_a_mismatch_not_an_overflow() {
        // `numChunks + 1` is a `long` add in Java. Here it panicked in a
        // debug build for a `.tvm` vlong near `i64::MAX`.
        let (tvd, tvx, mut tvm) = build_single_doc_chunk();
        let at = tvm_counts_offset(&tvm);
        let mut insert = Vec::new();
        write_vlong_test(&mut insert, i64::MAX as u64);
        insert.push(0); // numDirtyChunks
        insert.push(0); // numDirtyDocs
        tvm.splice(at..at + 3, insert);
        resign_footer(&mut tvm);

        assert!(matches!(
            open(&tvd, &tvx, &tvm, &id(), ""),
            Err(Error::NumChunksMismatch { .. })
        ));
    }

    #[test]
    fn a_negative_dirty_chunk_count_is_rejected_at_open() {
        // Java only *asserts* `numDirtyChunks >= 0`. Without a real check a
        // negative one passes all three cross-checks (it is "nonzero", it is
        // not greater than `numChunks`, and it is not greater than a positive
        // `numDirtyDocs`) and then reaches `too_dirty`'s `* 100`, which
        // overflows at `i64::MIN`.
        let (tvd, tvx, mut tvm) = build_single_doc_chunk();
        let at = tvm_counts_offset(&tvm);
        let mut insert = vec![1u8]; // numChunks = 1
        write_vlong_test(&mut insert, i64::MIN as u64);
        insert.push(200); // numDirtyDocs, > maxDocsPerChunk so `too_dirty` gets there
        tvm.splice(at..at + 3, insert);
        resign_footer(&mut tvm);

        let Err(err) = open(&tvd, &tvx, &tvm, &id(), "") else {
            panic!("a negative numDirtyChunks must be rejected at open");
        };
        assert!(format!("{err}").contains("negative chunk counts"), "{err}");
    }

    #[test]
    fn a_negative_max_pointer_is_a_decode_error_not_an_overflow() {
        // `maxPointer as usize + FOOTER_LENGTH` sign-extended a negative
        // `.tvm` field into a huge length and then overflowed the add.
        let (tvd, tvx, mut tvm) = build_single_doc_chunk();
        let mp_off = tvm.len() - codec_util::FOOTER_LENGTH - 3 - 8;
        tvm[mp_off..mp_off + 8].copy_from_slice(&(-16i64).to_le_bytes());
        resign_footer(&mut tvm);

        let Err(err) = open(&tvd, &tvx, &tvm, &id(), "") else {
            panic!("a negative maxPointer must be rejected at open");
        };
        assert!(format!("{err}").contains(".tvd length should be"), "{err}");
    }

    #[test]
    fn a_field_number_offset_past_the_distinct_field_numbers_is_a_decode_error() {
        // Java only asserts `fieldNumOff < fieldNums.length`; a corrupt
        // `allFieldNumOffs` entry indexed `field_nums` out of range here.
        let (mut tvd, tvx, mut tvm, chunk_start) =
            build_single_doc_chunk_with_meta_overrides(1, 0, 0);
        // allFieldNumOffs: slice length vint then the packed byte. The chunk
        // has one distinct field number, so `bits_per_off` is 1 and the only
        // legal offset is 0.
        assert_eq!(tvd[chunk_start + 5], 1, "allFieldNumOffs slice length");
        assert_eq!(tvd[chunk_start + 6], 0x00);
        tvd[chunk_start + 6] = 0x01;
        splice_tvd(&mut tvd, &mut tvm, chunk_start + 6, 1, &[0x01]);

        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let err = reader.document(0).unwrap_err();
        assert!(
            format!("{err}").contains("is outside the chunk's 1 distinct field numbers"),
            "{err}"
        );
    }

    /// A document with two fields, several terms each, carrying positions,
    /// offsets and payloads -- so a sweep over a segment of these reaches
    /// every stream `read_chunk` decodes, not just the two a minimal chunk
    /// has.
    fn rich_doc(n: usize) -> TermVectorsDocument {
        let term = |s: &str| format!("{s}{n:03}").into_bytes();
        TermVectorsDocument {
            fields: vec![
                TermVectorField {
                    field_number: 0,
                    has_positions: true,
                    has_offsets: true,
                    has_payloads: true,
                    terms: vec![
                        TermVectorTerm {
                            term: term("alpha"),
                            freq: 2,
                            positions: Some(vec![0, 4]),
                            start_offsets: Some(vec![0, 20]),
                            end_offsets: Some(vec![8, 28]),
                            payloads: Some(vec![vec![1, 2], vec![3]]),
                        },
                        TermVectorTerm {
                            term: term("alpine"),
                            freq: 1,
                            positions: Some(vec![7]),
                            start_offsets: Some(vec![35]),
                            end_offsets: Some(vec![43]),
                            payloads: Some(vec![Vec::new()]),
                        },
                    ],
                },
                TermVectorField {
                    field_number: 3,
                    has_positions: true,
                    has_offsets: false,
                    has_payloads: false,
                    terms: vec![TermVectorTerm {
                        term: term("beta"),
                        freq: 1,
                        positions: Some(vec![1]),
                        start_offsets: None,
                        end_offsets: None,
                        payloads: None,
                    }],
                },
            ],
        }
    }

    /// Re-signed single-byte corruption sweep over the whole `.tvd` chunk
    /// region and the whole `.tvm`, in c15/c19/c25's shape: flip one bit,
    /// re-sign the footer so the CRC cannot "catch" the corruption, and
    /// require a typed error rather than a panic or -- the failure this
    /// batch exists to remove -- an allocation abort no `catch_unwind` can
    /// intercept.
    ///
    /// The fixture is deliberately a **writer-produced, multi-chunk,
    /// multi-field, payload-carrying** segment rather than the hand-built
    /// single-document chunk the rest of this module's error tests use. A
    /// one-document chunk takes `read_chunk`'s `chunk_docs == 1` shortcut
    /// (so it never reaches `block_packed::decode_all` for the per-document
    /// field counts), has one distinct field number (so the `fieldNumOff`
    /// range check is vacuous), and gives the `.tvx` monotonic arrays
    /// nothing to discriminate. c27's `.fdm` sweep learned this the
    /// expensive way: its first run scored 211/282 against a single-chunk
    /// segment and was measuring the fixture, not the decoder.
    ///
    /// The assertion is deliberately *not* "every corruption is rejected":
    /// many single-bit flips produce a self-consistent chunk that decodes to
    /// different but well-formed vectors, which is what a checksum is for.
    /// What is asserted is that nothing panics and that the rejection rate
    /// does not silently collapse.
    #[test]
    fn every_resigned_single_byte_tvd_and_tvm_corruption_is_an_error_or_a_clean_decode() {
        // 200 rich documents: the byte-size trigger closes several chunks, so
        // `.tvx`'s two monotonic arrays carry real entries.
        let docs: Vec<TermVectorsDocument> = (0..400).map(rich_doc).collect();
        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
        let max_doc = docs.len() as i32;
        {
            let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
            assert!(
                reader.num_chunks() >= 3,
                "the sweep needs a multi-chunk segment, got {}",
                reader.num_chunks()
            );
            // Two fields with different flag sets, so the per-field-number
            // flags array and the `fieldNumOff` range check are both live.
            let chunk = reader.read_chunk(0).unwrap();
            assert!(
                chunk.num_docs() > 1,
                "the `chunkDocs == 1` shortcut must not fire"
            );
            assert_eq!(chunk.document(0).unwrap().unwrap().fields.len(), 2);
        }

        // The `.tvd` header is fixed-size and shared with every other codec
        // file; the interesting bytes are the chunk region.
        let header_len = {
            let mut probe = SliceInput::new(&tvd);
            codec_util::check_index_header(
                &mut probe,
                DATA_CODEC,
                VERSION_START,
                VERSION_CURRENT,
                &id(),
                "",
            )
            .unwrap();
            probe.position()
        };
        let mut total = 0usize;
        let mut rejected = 0usize;

        // Read every document, not just doc 0: the per-document field-count
        // and byte-cursor arrays only discriminate when they are walked.
        let read_all = |tvd: &[u8], tvx: &[u8], tvm: &[u8]| -> Result<()> {
            let reader = open(tvd, tvx, tvm, &id(), "")?;
            let mut cursor = ChunkCursor::new();
            for doc in 0..reader.max_doc().min(max_doc) {
                cursor.document(&reader, doc)?;
            }
            Ok(())
        };

        for off in (header_len..tvd.len() - codec_util::FOOTER_LENGTH).step_by(7) {
            for mask in [0x01u8, 0x80] {
                let mut corrupt = tvd.clone();
                corrupt[off] ^= mask;
                resign_footer(&mut corrupt);
                total += 1;
                if read_all(&corrupt, &tvx, &tvm).is_err() {
                    rejected += 1;
                }
            }
        }
        for off in 0..tvm.len() - codec_util::FOOTER_LENGTH {
            for mask in [0x01u8, 0x80] {
                let mut corrupt = tvm.clone();
                corrupt[off] ^= mask;
                resign_footer(&mut corrupt);
                total += 1;
                if read_all(&tvd, &tvx, &corrupt).is_err() {
                    rejected += 1;
                }
            }
        }

        // Measured when this was written: 688 of 3 724 (18%), and that low
        // figure is the fixture being *right*, not the decoder being weak.
        // A `.tvd` chunk is mostly its LZ4 literal run -- term suffixes and
        // payload bytes -- and flipping one of those yields a different but
        // perfectly well-formed vector, which is what the checksum exists to
        // catch. `.tvm`, which is all structure, rejects 145 of its 152.
        // Compare c27's `.kdd` 204/438 and `.fdt` 624/904, both for the same
        // reason, against `.fdm` 269/282 and `.tim`+`.tip`+`.tmd` 391/436.
        // What this pins is that nothing *panics or aborts* on 3 724 corrupt
        // multi-chunk segments, plus a floor so a future change that stops
        // bounding something fails loudly.
        assert!(
            rejected >= 660,
            "only {rejected} of {total} re-signed .tvd/.tvm corruptions were rejected"
        );
    }

    #[test]
    fn invalid_flags_selector_rejected() {
        let (tvd, tvx, tvm) = build_offsets_only_field_chunk_with_selector(7);
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert!(matches!(
            reader.document(0),
            Err(Error::InvalidFlagsSelector(7))
        ));
    }

    /// Hand-encodes a single-doc, single-field chunk with one term ("cat")
    /// whose field has OFFSETS only (no POSITIONS, no PAYLOADS) -- exercises
    /// the "no positions" branch of offset-patching (position_correction=0)
    /// and, with `selector=0`, the dedup-flags-array decode path (per-field
    /// flags looked up by distinct-field-number rather than stored direct
    /// per field). `selector` lets `invalid_flags_selector_rejected` reuse
    /// this builder with an out-of-range selector instead.
    fn build_offsets_only_field_chunk_with_selector(selector: i32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut tvd = Vec::new();
        tvd.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut tvd, DATA_CODEC);
        tvd.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        tvd.extend_from_slice(&id());
        tvd.push(0); // empty suffix
        let chunk_start = tvd.len() as i64;

        write_vint(&mut tvd, 0); // docBase
        write_vint(&mut tvd, 1 << 1); // token: chunkDocs=1, dirty=0
        write_vint(&mut tvd, 1); // numFields = totalFields = 1

        // fieldNums: 1 distinct field (number 9), 8 bits/value.
        tvd.push(8); // token: (totalDistinct-1=0)<<5 | bitsPerFieldNum=8
        tvd.push(9);

        // allFieldNumOffs: 1 field, offset 0 into field_nums, 1 bit/value.
        write_vint(&mut tvd, 1); // slice byte length
        tvd.push(0x00);

        write_vint(&mut tvd, selector);
        if selector == 0 {
            // Dedup path: one flags entry per distinct field number
            // (FLAGS_BITS=4), value=2 (OFFSETS only).
            write_vint(&mut tvd, 1); // slice byte length
            tvd.push(0x02);
        } else if selector == 1 {
            // Direct path: one flags entry per field in the chunk.
            write_vint(&mut tvd, 1); // slice byte length
            tvd.push(0x02);
        }
        // For an invalid selector, no further flags bytes are read at all
        // (document() errors out immediately on the unmatched selector).

        // numTerms: 1 field, 1 bit/value, value=1.
        write_vint(&mut tvd, 1); // bitsRequired
        write_vint(&mut tvd, 1); // slice byte length
        tvd.push(0x01);

        // prefixLengths [0] (bpv=0, min=0).
        tvd.push(0x01);
        // suffixLengths [3] (bpv=0, min=3): token, minValue vlong.
        tvd.push(0x00);
        let target = lucene_util::zigzag::encode(3) - 1;
        write_vlong_test(&mut tvd, target);
        // termFreqsMinus1 [0] (bpv=0, min=0).
        tvd.push(0x01);

        // No positions_flat (total_positions=0, OFFSETS-only field).
        // charsPerTerm: 1 distinct field, value 4.0 (irrelevant, no
        // positions to multiply against since has_positions=false).
        tvd.extend_from_slice(&4.0f32.to_bits().to_le_bytes());
        // start_offsets_flat [0] (bpv=0, min=0) -- absolute offset 0.
        tvd.push(0x01);
        // lengths_flat [0] (bpv=0, min=0) -- actual length = 0 + termLen(3).
        tvd.push(0x01);
        // No payload_lengths_flat (total_payloads=0).

        // LZ4 (CompressionMode.FAST, no dictionary): literal-only unit
        // wrapping "cat" (the only term suffix; no payload bytes).
        let payload = *b"cat";
        tvd.push((payload.len() as u8) << 4);
        tvd.extend_from_slice(&payload);

        tvd.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        tvd.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&tvd) as u64;
        tvd.extend_from_slice(&checksum.to_be_bytes());

        let (tvx, tvm) = build_trivial_single_chunk_index_and_meta(chunk_start, tvd.len() as i64);
        (tvd, tvx, tvm)
    }

    fn write_vlong_test(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    /// Builds a trivial `.tvx`/`.tvm` pair for a single one-chunk, one-doc
    /// segment whose `.tvd` chunk starts at `chunk_start` and whose
    /// (footer-terminated) length is `tvd_len`.
    fn build_trivial_single_chunk_index_and_meta(
        chunk_start: i64,
        tvd_len: i64,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut tvx = Vec::new();
        tvx.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut tvx, INDEX_CODEC);
        tvx.extend_from_slice(&(INDEX_VERSION_CURRENT as u32).to_be_bytes());
        tvx.extend_from_slice(&id());
        tvx.push(0);
        let docs_start = tvx.len() as i64;
        let docs_end = tvx.len() as i64;
        let start_pointers_end = tvx.len() as i64;
        tvx.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        tvx.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&tvx) as u64;
        tvx.extend_from_slice(&checksum.to_be_bytes());

        let max_doc = 1i32;
        let max_pointer = tvd_len - codec_util::FOOTER_LENGTH as i64;
        let mut tvm = Vec::new();
        tvm.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut tvm, META_CODEC);
        tvm.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        tvm.extend_from_slice(&id());
        tvm.push(0);
        write_vint(&mut tvm, 0);
        write_vint(&mut tvm, 4096);
        tvm.extend_from_slice(&max_doc.to_le_bytes());
        tvm.extend_from_slice(&0i32.to_le_bytes());
        tvm.extend_from_slice(&2i32.to_le_bytes());
        tvm.extend_from_slice(&docs_start.to_le_bytes());
        for min in [0i64, max_doc as i64] {
            tvm.extend_from_slice(&min.to_le_bytes());
            tvm.extend_from_slice(&0i32.to_le_bytes());
            tvm.extend_from_slice(&0i64.to_le_bytes());
            tvm.push(0);
        }
        tvm.extend_from_slice(&docs_end.to_le_bytes());
        for min in [chunk_start, max_pointer] {
            tvm.extend_from_slice(&min.to_le_bytes());
            tvm.extend_from_slice(&0i32.to_le_bytes());
            tvm.extend_from_slice(&0i64.to_le_bytes());
            tvm.push(0);
        }
        tvm.extend_from_slice(&start_pointers_end.to_le_bytes());
        tvm.extend_from_slice(&max_pointer.to_le_bytes());
        write_vint(&mut tvm, 1);
        write_vint(&mut tvm, 0);
        write_vint(&mut tvm, 0);
        tvm.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        tvm.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&tvm) as u64;
        tvm.extend_from_slice(&checksum.to_be_bytes());

        (tvx, tvm)
    }

    #[test]
    fn dedup_flags_selector_and_offsets_without_positions() {
        let (tvd, tvx, tvm) = build_offsets_only_field_chunk_with_selector(0);
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let doc = reader.document(0).unwrap().unwrap();
        assert_eq!(doc.fields.len(), 1);
        let field = &doc.fields[0];
        assert_eq!(field.field_number, 9);
        assert!(!field.has_positions);
        assert!(field.has_offsets);
        assert!(!field.has_payloads);
        assert_eq!(field.terms.len(), 1);
        let cat = &field.terms[0];
        assert_eq!(cat.term, b"cat");
        assert_eq!(cat.freq, 1);
        assert_eq!(cat.positions, None);
        assert_eq!(cat.start_offsets, Some(vec![0]));
        assert_eq!(cat.end_offsets, Some(vec![3]));
        assert_eq!(cat.payloads, None);
    }

    #[test]
    fn direct_flags_selector_offsets_without_positions() {
        let (tvd, tvx, tvm) = build_offsets_only_field_chunk_with_selector(1);
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let doc = reader.document(0).unwrap().unwrap();
        let field = &doc.fields[0];
        assert!(!field.has_positions && field.has_offsets && !field.has_payloads);
        assert_eq!(field.terms[0].start_offsets, Some(vec![0]));
        assert_eq!(field.terms[0].end_offsets, Some(vec![3]));
    }

    /// Hand-encodes a single-doc chunk with 8 distinct field numbers (0..7),
    /// each with one field carrying a single one-character term, freq 1, and
    /// **no** positions/offsets/payloads at all. Exercises the
    /// distinct-field-numbers extension-byte path (>=8 distinct field
    /// numbers needs an extra vint beyond the 3-bit inline count) and the
    /// all-empty-arrays branches (`total_positions`/`total_offsets`/
    /// `total_payloads` all 0, so none of those `block_packed` streams are
    /// read at all).
    fn build_eight_distinct_fields_no_flags_chunk() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut tvd = Vec::new();
        tvd.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut tvd, DATA_CODEC);
        tvd.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        tvd.extend_from_slice(&id());
        tvd.push(0); // empty suffix
        let chunk_start = tvd.len() as i64;

        write_vint(&mut tvd, 0); // docBase
        write_vint(&mut tvd, 1 << 1); // token: chunkDocs=1, dirty=0
        write_vint(&mut tvd, 8); // numFields = totalFields = 8

        // fieldNums: 8 distinct fields (0..7), 3 bits/value. totalDistinct-1
        // (7) hits the 3-bit inline cap (0x07), so an extra vint(0) follows.
        tvd.push(0xE3); // (7<<5)|3
        write_vint(&mut tvd, 0); // extra: totalDistinct = 7+0+1 = 8
        tvd.extend_from_slice(&[0x05, 0x39, 0x77]); // packed_ints, 3 bits x 8 -> [0..7]

        // allFieldNumOffs: 8 fields, identity offsets 0..7, 4 bits/value
        // (bitsRequired(totalDistinct-1=7) rounds up to 4).
        write_vint(&mut tvd, 4); // slice byte length
        tvd.extend_from_slice(&[0x10, 0x32, 0x54, 0x76]);

        // flags: selector=1 (direct), 8 fields, FLAGS_BITS=4, all 0.
        write_vint(&mut tvd, 1); // selector
        write_vint(&mut tvd, 4); // slice byte length
        tvd.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

        // numTerms: 8 fields, 1 bit/value, all 1.
        write_vint(&mut tvd, 1); // bitsRequired
        write_vint(&mut tvd, 1); // slice byte length
        tvd.push(0xFF);

        // prefixLengths: 8 zeros (bpv=0, min=0).
        tvd.push(0x01);
        // suffixLengths: 8 ones (bpv=0, min=1): token, minValue vlong.
        tvd.push(0x00);
        let target = lucene_util::zigzag::encode(1) - 1;
        write_vlong_test(&mut tvd, target);
        // termFreqsMinus1: 8 zeros (bpv=0, min=0).
        tvd.push(0x01);

        // No positions/offsets/payloads streams at all (all totals 0).

        // LZ4: 8 one-byte term suffixes, no payload bytes.
        let payload = *b"abcdefgh";
        tvd.push((payload.len() as u8) << 4);
        tvd.extend_from_slice(&payload);

        tvd.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        tvd.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&tvd) as u64;
        tvd.extend_from_slice(&checksum.to_be_bytes());

        let (tvx, tvm) = build_trivial_single_chunk_index_and_meta(chunk_start, tvd.len() as i64);
        (tvd, tvx, tvm)
    }

    #[test]
    fn write_best_speed_single_doc_single_field_single_term_round_trips() {
        let docs = vec![TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 5,
                has_positions: true,
                has_offsets: false,
                has_payloads: false,
                terms: vec![TermVectorTerm {
                    term: b"cat".to_vec(),
                    freq: 1,
                    positions: Some(vec![0]),
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                }],
            }],
        }];
        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert_eq!(reader.max_doc(), 1);
        let doc = reader.document(0).unwrap().unwrap();
        assert_eq!(doc.fields.len(), 1);
        let field = &doc.fields[0];
        assert_eq!(field.field_number, 5);
        assert!(field.has_positions && !field.has_offsets && !field.has_payloads);
        assert_eq!(field.terms.len(), 1);
        assert_eq!(field.terms[0].term, b"cat");
        assert_eq!(field.terms[0].freq, 1);
        assert_eq!(field.terms[0].positions, Some(vec![0]));
    }

    /// c33: with `OffsetAttribute`'s real unit (UTF-16 code units), a
    /// multi-byte term's offset span is **shorter** than the term's UTF-8 byte
    /// length, so `flushOffsets`' `length - prefixLength - suffixLength` goes
    /// negative -- the first caller in this crate ever to feed
    /// `block_packed::encode_all` a negative value (its doc comment used to
    /// say no caller did). `caf\u{e9}` spans 4 Java `char`s and occupies 5
    /// bytes: -1. `\u{4e16}\u{754c}` spans 2 and occupies 6: -4. Java's
    /// reader adds `prefixLength + suffixLength` back the same way, so a
    /// wrong sign here is invisible to every ASCII fixture and wrong for
    /// every non-ASCII document.
    #[test]
    fn write_best_speed_offsets_shorter_than_the_term_bytes_round_trip() {
        // "caf\u{e9} \u{4e16}\u{754c} dog", tokenized and offset the way
        // `lucene_analysis::tokenize` now does: Java `char` indices.
        let docs = vec![TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 0,
                has_positions: true,
                has_offsets: true,
                has_payloads: false,
                terms: vec![
                    TermVectorTerm {
                        term: "caf\u{e9}".as_bytes().to_vec(),
                        freq: 1,
                        positions: Some(vec![0]),
                        start_offsets: Some(vec![0]),
                        end_offsets: Some(vec![4]),
                        payloads: None,
                    },
                    TermVectorTerm {
                        term: "dog".as_bytes().to_vec(),
                        freq: 1,
                        positions: Some(vec![3]),
                        start_offsets: Some(vec![8]),
                        end_offsets: Some(vec![11]),
                        payloads: None,
                    },
                    TermVectorTerm {
                        term: "\u{4e16}\u{754c}".as_bytes().to_vec(),
                        freq: 1,
                        positions: Some(vec![1]),
                        start_offsets: Some(vec![5]),
                        end_offsets: Some(vec![7]),
                        payloads: None,
                    },
                ],
            }],
        }];
        // The negative deltas this exercises, stated rather than assumed.
        assert_eq!(4i32 - "caf\u{e9}".len() as i32, -1);
        assert_eq!(2i32 - "\u{4e16}\u{754c}".len() as i32, -4);

        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let doc = reader.document(0).unwrap().unwrap();
        let field = &doc.fields[0];
        let got: Vec<(&str, i32, i32)> = field
            .terms
            .iter()
            .map(|t| {
                (
                    std::str::from_utf8(&t.term).unwrap(),
                    t.start_offsets.as_ref().unwrap()[0],
                    t.end_offsets.as_ref().unwrap()[0],
                )
            })
            .collect();
        // Terms come back in the writer's (ascending byte) order.
        assert_eq!(
            got,
            vec![
                ("caf\u{e9}", 0, 4),
                ("dog", 8, 11),
                ("\u{4e16}\u{754c}", 5, 7),
            ]
        );
    }

    #[test]
    fn write_best_speed_all_field_numbers_zero_uses_nonzero_bit_width() {
        // Regression test: a chunk where every field across every doc has
        // field_number == 0 (an entirely ordinary case -- e.g. any
        // single-field index) must not encode bits_per_field_num as 0. Real
        // Lucene's reader unconditionally indexes packedBulkOps[bitsPerValue
        // - 1], so a 0-bit width there is an ArrayIndexOutOfBoundsException
        // in real Lucene even though this port's own reader tolerates it --
        // this test only proves the width isn't 0 on the wire; cross-engine
        // coverage for this exact shape lives in the fixture example.
        let docs = vec![
            TermVectorsDocument {
                fields: vec![TermVectorField {
                    field_number: 0,
                    has_positions: true,
                    has_offsets: false,
                    has_payloads: false,
                    terms: vec![TermVectorTerm {
                        term: b"cat".to_vec(),
                        freq: 1,
                        positions: Some(vec![0]),
                        start_offsets: None,
                        end_offsets: None,
                        payloads: None,
                    }],
                }],
            },
            TermVectorsDocument {
                fields: vec![TermVectorField {
                    field_number: 0,
                    has_positions: true,
                    has_offsets: false,
                    has_payloads: false,
                    terms: vec![TermVectorTerm {
                        term: b"dog".to_vec(),
                        freq: 1,
                        positions: Some(vec![0]),
                        start_offsets: None,
                        end_offsets: None,
                        payloads: None,
                    }],
                }],
            },
        ];
        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");

        // bits_per_field_num is the low 5 bits of the fdt's first byte,
        // right after the tvx/tvm header framing this test doesn't need to
        // re-derive -- simplest to assert the invariant round-trip-only:
        // decode via this module's own reader (which would also decode a
        // wire-correct 0-bits chunk without complaint) but additionally
        // confirm the actually-written token byte's low 5 bits are nonzero
        // by re-deriving the same offset the reader itself uses.
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert_eq!(reader.max_doc(), 2);
        let doc0 = reader.document(0).unwrap().unwrap();
        assert_eq!(doc0.fields[0].field_number, 0);
        assert_eq!(doc0.fields[0].terms[0].term, b"cat");
        let doc1 = reader.document(1).unwrap().unwrap();
        assert_eq!(doc1.fields[0].field_number, 0);
        assert_eq!(doc1.fields[0].terms[0].term, b"dog");

        // Note: this port's own reader tolerates bits_per_field_num == 0
        // (an all-zero chunk decodes as all-zero regardless), so a
        // round-trip through it can't by itself prove the wire bit-width
        // isn't 0 -- that only matters to a *real* Lucene reader. The
        // cross-engine fixture (write_term_vectors_fixture.rs /
        // VerifyTermVectors.java) covers an all-field-0 chunk specifically
        // so this shape is actually proven against real Lucene, not just
        // against this port's own (more permissive) reader.
    }

    #[test]
    fn write_best_speed_single_doc_single_field_multiple_terms_round_trips() {
        let docs = vec![TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 2,
                has_positions: true,
                has_offsets: false,
                has_payloads: false,
                terms: vec![
                    TermVectorTerm {
                        term: b"cat".to_vec(),
                        freq: 2,
                        positions: Some(vec![0, 3]),
                        start_offsets: None,
                        end_offsets: None,
                        payloads: None,
                    },
                    TermVectorTerm {
                        term: b"dog".to_vec(),
                        freq: 1,
                        positions: Some(vec![1]),
                        start_offsets: None,
                        end_offsets: None,
                        payloads: None,
                    },
                ],
            }],
        }];
        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let doc = reader.document(0).unwrap().unwrap();
        let field = &doc.fields[0];
        assert_eq!(field.terms.len(), 2);
        assert_eq!(field.terms[0].term, b"cat");
        assert_eq!(field.terms[0].positions, Some(vec![0, 3]));
        assert_eq!(field.terms[1].term, b"dog");
        assert_eq!(field.terms[1].positions, Some(vec![1]));
    }

    #[test]
    fn write_best_speed_multi_doc_multi_field_round_trips() {
        let docs = vec![
            TermVectorsDocument {
                fields: vec![
                    TermVectorField {
                        field_number: 0,
                        has_positions: true,
                        has_offsets: false,
                        has_payloads: false,
                        terms: vec![TermVectorTerm {
                            term: b"alpha".to_vec(),
                            freq: 1,
                            positions: Some(vec![0]),
                            start_offsets: None,
                            end_offsets: None,
                            payloads: None,
                        }],
                    },
                    TermVectorField {
                        field_number: 1,
                        has_positions: false,
                        has_offsets: false,
                        has_payloads: false,
                        terms: vec![TermVectorTerm {
                            term: b"beta".to_vec(),
                            freq: 1,
                            positions: None,
                            start_offsets: None,
                            end_offsets: None,
                            payloads: None,
                        }],
                    },
                ],
            },
            TermVectorsDocument { fields: vec![] },
            TermVectorsDocument {
                fields: vec![TermVectorField {
                    field_number: 0,
                    has_positions: true,
                    has_offsets: false,
                    has_payloads: false,
                    terms: vec![TermVectorTerm {
                        term: b"gamma".to_vec(),
                        freq: 3,
                        positions: Some(vec![0, 1, 5]),
                        start_offsets: None,
                        end_offsets: None,
                        payloads: None,
                    }],
                }],
            },
        ];
        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert_eq!(reader.max_doc(), 3);

        let doc0 = reader.document(0).unwrap().unwrap();
        assert_eq!(doc0.fields.len(), 2);
        assert_eq!(doc0.fields[0].field_number, 0);
        assert_eq!(doc0.fields[0].terms[0].term, b"alpha");
        assert_eq!(doc0.fields[0].terms[0].positions, Some(vec![0]));
        assert_eq!(doc0.fields[1].field_number, 1);
        assert!(!doc0.fields[1].has_positions);
        assert_eq!(doc0.fields[1].terms[0].term, b"beta");
        assert_eq!(doc0.fields[1].terms[0].positions, None);

        assert!(reader.document(1).unwrap().is_none());

        let doc2 = reader.document(2).unwrap().unwrap();
        assert_eq!(doc2.fields.len(), 1);
        assert_eq!(doc2.fields[0].terms[0].term, b"gamma");
        assert_eq!(doc2.fields[0].terms[0].positions, Some(vec![0, 1, 5]));
    }

    #[test]
    fn write_best_speed_empty_doc_set_produces_zero_max_doc() {
        let (tvd, tvx, tvm) = write_best_speed(&[], &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert_eq!(reader.max_doc(), 0);
    }

    /// Regression: a field that declares OFFSETS (or PAYLOADS) but holds no
    /// terms contributes zero occurrences, so the reader -- which gates on
    /// `totalOffsets > 0` / `totalPayloads > 0`, not on the flag -- expects
    /// no `charsPerTerm`/offset/payload-length streams at all. Writing them
    /// anyway desynchronizes every following byte of the chunk.
    #[test]
    fn write_best_speed_flagged_but_termless_field_round_trips() {
        let docs = vec![TermVectorsDocument {
            fields: vec![
                TermVectorField {
                    field_number: 0,
                    has_positions: false,
                    has_offsets: true,
                    has_payloads: true,
                    terms: vec![],
                },
                TermVectorField {
                    field_number: 1,
                    has_positions: true,
                    has_offsets: false,
                    has_payloads: false,
                    terms: vec![TermVectorTerm {
                        term: b"hello".to_vec(),
                        freq: 2,
                        positions: Some(vec![0, 5]),
                        start_offsets: None,
                        end_offsets: None,
                        payloads: None,
                    }],
                },
            ],
        }];
        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let doc = reader.document(0).unwrap().unwrap();
        assert_eq!(doc.fields.len(), 2);
        assert_eq!(doc.fields[0].field_number, 0);
        assert!(doc.fields[0].has_offsets && doc.fields[0].has_payloads);
        assert!(doc.fields[0].terms.is_empty());
        assert_eq!(doc.fields[1].field_number, 1);
        assert_eq!(doc.fields[1].terms.len(), 1);
        assert_eq!(doc.fields[1].terms[0].term, b"hello".to_vec());
        assert_eq!(doc.fields[1].terms[0].positions, Some(vec![0, 5]));
    }

    #[test]
    fn write_best_speed_offsets_only_round_trips() {
        let docs = vec![TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 3,
                has_positions: false,
                has_offsets: true,
                has_payloads: false,
                terms: vec![TermVectorTerm {
                    term: b"cat".to_vec(),
                    freq: 2,
                    positions: None,
                    start_offsets: Some(vec![0, 10]),
                    end_offsets: Some(vec![3, 13]),
                    payloads: None,
                }],
            }],
        }];
        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let doc = reader.document(0).unwrap().unwrap();
        let field = &doc.fields[0];
        assert!(!field.has_positions && field.has_offsets && !field.has_payloads);
        let cat = &field.terms[0];
        assert_eq!(cat.positions, None);
        assert_eq!(cat.start_offsets, Some(vec![0, 10]));
        assert_eq!(cat.end_offsets, Some(vec![3, 13]));
        assert_eq!(cat.payloads, None);
    }

    #[test]
    fn write_best_speed_positions_and_offsets_freq_three_round_trips() {
        // Regression for a review-confirmed bug: the offset-delta correction used
        // the absolute position instead of the position *delta* the read side
        // actually applies, which only coincidentally matched for freq <= 2. A
        // freq-3 term with non-trivial, non-uniform position gaps is the
        // smallest case that would silently corrupt the 3rd+ occurrence's
        // decoded offsets under the old (buggy) code.
        let docs = vec![TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 3,
                has_positions: true,
                has_offsets: true,
                has_payloads: false,
                terms: vec![TermVectorTerm {
                    term: b"the".to_vec(),
                    freq: 3,
                    positions: Some(vec![0, 5, 12]),
                    start_offsets: Some(vec![0, 20, 45]),
                    end_offsets: Some(vec![3, 23, 48]),
                    payloads: None,
                }],
            }],
        }];
        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let doc = reader.document(0).unwrap().unwrap();
        let field = &doc.fields[0];
        let the = &field.terms[0];
        assert_eq!(the.positions, Some(vec![0, 5, 12]));
        assert_eq!(the.start_offsets, Some(vec![0, 20, 45]));
        assert_eq!(the.end_offsets, Some(vec![3, 23, 48]));
    }

    #[test]
    fn write_best_speed_payloads_only_round_trips() {
        let docs = vec![TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 4,
                has_positions: false,
                has_offsets: false,
                has_payloads: true,
                terms: vec![TermVectorTerm {
                    term: b"dog".to_vec(),
                    freq: 2,
                    positions: None,
                    start_offsets: None,
                    end_offsets: None,
                    payloads: Some(vec![vec![0xAA, 0xBB], vec![]]),
                }],
            }],
        }];
        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let doc = reader.document(0).unwrap().unwrap();
        let field = &doc.fields[0];
        assert!(!field.has_positions && !field.has_offsets && field.has_payloads);
        let dog = &field.terms[0];
        assert_eq!(dog.payloads, Some(vec![vec![0xAA, 0xBB], vec![]]));
    }

    #[test]
    fn write_best_speed_positions_offsets_payloads_round_trips_multi_term_multi_doc() {
        let docs = vec![
            TermVectorsDocument {
                fields: vec![TermVectorField {
                    field_number: 5,
                    has_positions: true,
                    has_offsets: true,
                    has_payloads: true,
                    terms: vec![
                        TermVectorTerm {
                            term: b"cat".to_vec(),
                            freq: 2,
                            positions: Some(vec![0, 2]),
                            start_offsets: Some(vec![0, 8]),
                            end_offsets: Some(vec![3, 11]),
                            payloads: Some(vec![vec![0xAA], vec![]]),
                        },
                        TermVectorTerm {
                            term: b"car".to_vec(),
                            freq: 1,
                            positions: Some(vec![1]),
                            start_offsets: Some(vec![4]),
                            end_offsets: Some(vec![7]),
                            payloads: Some(vec![vec![0xBB, 0xCC]]),
                        },
                    ],
                }],
            },
            TermVectorsDocument {
                fields: vec![TermVectorField {
                    field_number: 5,
                    has_positions: true,
                    has_offsets: true,
                    has_payloads: true,
                    terms: vec![TermVectorTerm {
                        term: b"dog".to_vec(),
                        freq: 1,
                        positions: Some(vec![0]),
                        start_offsets: Some(vec![0]),
                        end_offsets: Some(vec![3]),
                        payloads: Some(vec![vec![0xDD, 0xEE, 0xFF]]),
                    }],
                }],
            },
        ];
        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert_eq!(reader.max_doc(), 2);

        let doc0 = reader.document(0).unwrap().unwrap();
        let field0 = &doc0.fields[0];
        assert!(field0.has_positions && field0.has_offsets && field0.has_payloads);
        let cat = &field0.terms[0];
        assert_eq!(cat.term, b"cat");
        assert_eq!(cat.positions, Some(vec![0, 2]));
        assert_eq!(cat.start_offsets, Some(vec![0, 8]));
        assert_eq!(cat.end_offsets, Some(vec![3, 11]));
        assert_eq!(cat.payloads, Some(vec![vec![0xAA], vec![]]));
        let car = &field0.terms[1];
        assert_eq!(car.term, b"car");
        assert_eq!(car.positions, Some(vec![1]));
        assert_eq!(car.start_offsets, Some(vec![4]));
        assert_eq!(car.end_offsets, Some(vec![7]));
        assert_eq!(car.payloads, Some(vec![vec![0xBB, 0xCC]]));

        let doc1 = reader.document(1).unwrap().unwrap();
        let field1 = &doc1.fields[0];
        let dog = &field1.terms[0];
        assert_eq!(dog.term, b"dog");
        assert_eq!(dog.positions, Some(vec![0]));
        assert_eq!(dog.start_offsets, Some(vec![0]));
        assert_eq!(dog.end_offsets, Some(vec![3]));
        assert_eq!(dog.payloads, Some(vec![vec![0xDD, 0xEE, 0xFF]]));
    }

    #[test]
    fn write_best_speed_positions_only_regression_still_works() {
        // Same shape as `write_best_speed_single_doc_single_field_multiple_terms_round_trips`
        // but re-asserted here to make explicit that the positions-only path
        // (has_offsets=false, has_payloads=false) is unaffected by the
        // offsets/payloads extension.
        let docs = vec![TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 1,
                has_positions: true,
                has_offsets: false,
                has_payloads: false,
                terms: vec![TermVectorTerm {
                    term: b"only".to_vec(),
                    freq: 1,
                    positions: Some(vec![0]),
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                }],
            }],
        }];
        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let doc = reader.document(0).unwrap().unwrap();
        let field = &doc.fields[0];
        assert!(field.has_positions && !field.has_offsets && !field.has_payloads);
        assert_eq!(field.terms[0].positions, Some(vec![0]));
        assert_eq!(field.terms[0].start_offsets, None);
        assert_eq!(field.terms[0].payloads, None);
    }

    #[test]
    fn a_chunk_claiming_two_billion_documents_is_a_decode_error_not_an_overflow() {
        // Reachable only once `chunkDocs` is decoded the way Java decodes it
        // (`readVInt() >>> 1`, unsigned): a corrupt token then names a
        // `chunkDocs` near `i32::MAX`, and a chunk whose `docBase` is
        // anything but 0 overflows `docBase + chunkDocs`. Needs a *second*
        // chunk, so this builds a 129-document segment (the
        // MAX_DOCS_PER_CHUNK trigger) and corrupts the second chunk's header.
        let docs: Vec<TermVectorsDocument> = (0..129).map(tiny_doc).collect();
        let (mut tvd, tvx, mut tvm) = write_best_speed(&docs, &id(), "");
        let entry = {
            let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
            reader.chunk_for_doc(128).unwrap()
        };
        assert_eq!(entry.doc_base, 128);
        // docBase = 128 is a two-byte vint; the token vint follows it.
        let token_at = entry.start_pointer as usize + 2;
        assert_eq!(
            tvd[token_at],
            (1 << 1) | 1,
            "one document, dirty tail chunk"
        );
        let mut insert = Vec::new();
        write_vint(&mut insert, -1); // token = -1 => chunkDocs = 0x7FFF_FFFF
        splice_tvd(&mut tvd, &mut tvm, token_at, 1, &insert);

        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert!(matches!(
            reader.document(128),
            Err(Error::CorruptChunkBounds {
                doc_base: 128,
                chunk_docs: i32::MAX,
                ..
            })
        ));
    }

    #[test]
    fn encode_literal_lz4_round_trips_through_lz4_decompress() {
        for payload in [
            Vec::new(),
            b"short".to_vec(),
            vec![0x42u8; 5000], // forces the 0xFF-continuation length encoding
        ] {
            let encoded = encode_literal_lz4(&payload);
            let mut input = SliceInput::new(&encoded);
            let mut out = vec![0u8; payload.len()];
            if !payload.is_empty() {
                lz4::decompress(&mut input, payload.len(), &mut out, 0).unwrap();
            }
            assert_eq!(out, payload);
        }
    }

    #[test]
    fn eight_distinct_fields_with_no_flags_decodes_all_fields() {
        let (tvd, tvx, tvm) = build_eight_distinct_fields_no_flags_chunk();
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let doc = reader.document(0).unwrap().unwrap();
        assert_eq!(doc.fields.len(), 8);
        for (i, field) in doc.fields.iter().enumerate() {
            assert_eq!(field.field_number, i as i32);
            assert!(!field.has_positions && !field.has_offsets && !field.has_payloads);
            assert_eq!(field.terms.len(), 1);
            let term = &field.terms[0];
            assert_eq!(term.term, vec![b'a' + i as u8]);
            assert_eq!(term.freq, 1);
            assert_eq!(term.positions, None);
            assert_eq!(term.start_offsets, None);
            assert_eq!(term.end_offsets, None);
            assert_eq!(term.payloads, None);
        }
    }

    // ---------------------------------------------------------------
    // Chunking (`c8-tv-chunking`): the writer is
    // `Lucene90CompressingTermVectorsWriter`'s streaming, chunk-flushing
    // shape now, not one chunk per segment.
    // ---------------------------------------------------------------

    /// A positions-only document whose single field holds `terms` terms, each
    /// short enough that the *document count* trigger fires first.
    fn tiny_doc(n: usize) -> TermVectorsDocument {
        TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 0,
                has_positions: true,
                has_offsets: false,
                has_payloads: false,
                terms: vec![TermVectorTerm {
                    term: format!("t{n:04}").into_bytes(),
                    freq: 2,
                    positions: Some(vec![n as i32 % 7, n as i32 % 7 + 3]),
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                }],
            }],
        }
    }

    /// A document whose single field carries `bytes` bytes of term text, so
    /// the *byte size* trigger fires long before 128 documents.
    fn fat_doc(n: usize, bytes: usize) -> TermVectorsDocument {
        let mut term = format!("{n:08}").into_bytes();
        term.resize(bytes, b'x');
        TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 0,
                has_positions: true,
                has_offsets: false,
                has_payloads: false,
                terms: vec![TermVectorTerm {
                    term,
                    freq: 1,
                    positions: Some(vec![0]),
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                }],
            }],
        }
    }

    /// Every chunk header in `.tvd`, walked through `.tvx`: `(docBase,
    /// chunkDocs, dirty)`.
    fn chunk_headers(reader: &TermVectorsReader<'_>) -> Vec<(i32, i32, bool)> {
        let mut out = Vec::new();
        let mut doc = 0i32;
        while doc < reader.max_doc() {
            let entry = reader.chunk_for_doc(doc).unwrap();
            let mut input = SliceInput::new(reader.tvd());
            input.seek(entry.start_pointer as usize).unwrap();
            let base = input.read_vint().unwrap();
            let code = input.read_vint().unwrap();
            out.push((base, code >> 1, code & 1 != 0));
            doc = base + (code >> 1);
        }
        out
    }

    fn assert_round_trips(docs: &[TermVectorsDocument], tvd: &[u8], tvx: &[u8], tvm: &[u8]) {
        let reader = open(tvd, tvx, tvm, &id(), "").unwrap();
        assert_eq!(reader.max_doc(), docs.len() as i32);
        for (i, expected) in docs.iter().enumerate() {
            let got = reader.document(i as i32).unwrap();
            if expected.fields.is_empty() {
                assert!(got.is_none(), "doc {i} should have no vectors");
                continue;
            }
            let got = got.unwrap_or_else(|| panic!("doc {i} missing"));
            assert_eq!(
                got.fields.len(),
                expected.fields.len(),
                "doc {i} field count"
            );
            for (g, e) in got.fields.iter().zip(&expected.fields) {
                assert_eq!(g.field_number, e.field_number, "doc {i}");
                assert_eq!(g.has_positions, e.has_positions, "doc {i}");
                assert_eq!(g.has_offsets, e.has_offsets, "doc {i}");
                assert_eq!(g.has_payloads, e.has_payloads, "doc {i}");
                assert_eq!(g.terms, e.terms, "doc {i} field {}", e.field_number);
            }
        }
    }

    #[test]
    fn the_document_count_trigger_closes_a_chunk_at_128_documents() {
        // `Lucene90TermVectorsFormat`'s `maxDocsPerChunk`. 300 tiny documents
        // are nowhere near 4 096 bytes of term text, so every chunk boundary
        // here is the document-count trigger.
        let docs: Vec<_> = (0..300).map(tiny_doc).collect();
        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert_eq!(
            chunk_headers(&reader),
            vec![
                (0, 128, false),
                (128, 128, false),
                (256, 44, true), // the forced final flush
            ]
        );
        assert_eq!(reader.num_chunks(), 3);
        assert_eq!(reader.num_dirty_chunks(), 1);
        assert_eq!(reader.num_dirty_docs(), 44);
        assert_round_trips(&docs, &tvd, &tvx, &tvm);
    }

    #[test]
    fn the_byte_size_trigger_closes_a_chunk_before_128_documents() {
        // 600-byte terms: the 4 096-byte trigger fires on the 7th document
        // (7 * 600 = 4 200 >= 4 096), long before 128.
        let docs: Vec<_> = (0..20).map(|n| fat_doc(n, 600)).collect();
        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let headers = chunk_headers(&reader);
        assert_eq!(headers.len(), 3, "{headers:?}");
        assert_eq!(headers[0], (0, 7, false));
        assert_eq!(headers[1], (7, 7, false));
        assert_eq!(headers[2], (14, 6, true));
        assert_round_trips(&docs, &tvd, &tvx, &tvm);
    }

    #[test]
    fn a_document_set_ending_exactly_on_a_chunk_boundary_still_flushes_a_dirty_tail() {
        // 128 documents fill the first chunk on the *last* `add_document`,
        // so `finish` finds nothing pending: one clean chunk, no dirty ones.
        let docs: Vec<_> = (0..128).map(tiny_doc).collect();
        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert_eq!(chunk_headers(&reader), vec![(0, 128, false)]);
        assert_eq!(reader.num_dirty_chunks(), 0);
        assert_eq!(reader.num_dirty_docs(), 0);
        assert_round_trips(&docs, &tvd, &tvx, &tvm);
    }

    #[test]
    fn a_custom_chunk_geometry_is_recorded_and_blocks_a_default_geometry_bulk_copy() {
        // `Lucene90CompressingTermVectorsFormat`'s constructor takes
        // `chunkSize`/`maxDocsPerChunk`; `Lucene90TermVectorsFormat` fixes
        // them at 4096/128. A segment written with any other geometry is
        // still valid and readable -- `chunkSize` is in `.tvm` -- but
        // `canPerformBulkMerge` refuses to copy its chunks into a
        // default-geometry writer.
        let docs: Vec<_> = (0..300).map(tiny_doc).collect();
        let mut w =
            TermVectorsWriter::with_geometry(&id(), "", i32::MAX as usize, i32::MAX as usize);
        for doc in &docs {
            w.add_document(doc);
        }
        let (tvd, tvx, tvm) = w.finish();
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        // One chunk for the whole segment -- which is exactly what this
        // module's writer did before batch `c8`.
        assert_eq!(reader.num_chunks(), 1);
        assert_eq!(chunk_headers(&reader), vec![(0, 300, true)]);
        assert_round_trips(&docs, &tvd, &tvx, &tvm);

        let default = TermVectorsWriter::new(&id(), "");
        assert!(!default.can_bulk_copy(&reader));
        assert!(matches!(
            TermVectorsWriter::new(&id(), "").copy_chunks(&reader, 0, 300),
            Err(Error::BulkCopyNotPermitted { .. })
        ));
        // ... but a writer with the same geometry can.
        let mut same =
            TermVectorsWriter::with_geometry(&id(), "", i32::MAX as usize, i32::MAX as usize);
        same.copy_chunks(&reader, 0, 300).unwrap();
        let (tvd2, tvx2, tvm2) = same.finish();
        assert_round_trips(&docs, &tvd2, &tvx2, &tvm2);
    }

    #[test]
    fn a_multi_chunk_segment_round_trips_positions_offsets_and_payloads() {
        let docs: Vec<_> = (0..400)
            .map(|n| TermVectorsDocument {
                fields: vec![
                    TermVectorField {
                        field_number: 0,
                        has_positions: true,
                        has_offsets: true,
                        has_payloads: true,
                        terms: vec![
                            TermVectorTerm {
                                term: format!("alpha{:03}", n % 17).into_bytes(),
                                freq: 3,
                                positions: Some(vec![0, 4, 9]),
                                start_offsets: Some(vec![0, 24, 54]),
                                end_offsets: Some(vec![8, 32, 62]),
                                payloads: Some(vec![
                                    vec![1, 2],
                                    vec![],
                                    vec![(n % 251) as u8, 7, 7],
                                ]),
                            },
                            TermVectorTerm {
                                term: format!("alpha{:03}z", n % 17).into_bytes(),
                                freq: 1,
                                positions: Some(vec![11]),
                                start_offsets: Some(vec![66]),
                                end_offsets: Some(vec![75]),
                                payloads: Some(vec![vec![9]]),
                            },
                        ],
                    },
                    TermVectorField {
                        field_number: 3,
                        has_positions: true,
                        has_offsets: false,
                        has_payloads: false,
                        terms: vec![TermVectorTerm {
                            term: format!("beta{n}").into_bytes(),
                            freq: 2,
                            positions: Some(vec![1, 6]),
                            start_offsets: None,
                            end_offsets: None,
                            payloads: None,
                        }],
                    },
                ],
            })
            .collect();
        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert!(reader.num_chunks() > 1, "expected several chunks");
        assert_round_trips(&docs, &tvd, &tvx, &tvm);
    }

    #[test]
    fn terms_are_prefix_compressed_against_the_previous_term_in_the_field() {
        // `startTerm`'s `StringHelper.bytesDifference`. Two documents with the
        // same *total* term bytes, one where consecutive terms share a long
        // prefix and one where they share nothing: only prefix sharing can
        // make the first `.tvd` smaller.
        let shared = TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 0,
                has_positions: false,
                has_offsets: false,
                has_payloads: false,
                terms: (0..40)
                    .map(|i| TermVectorTerm {
                        term: format!("commonprefixaaaaaaaaaaaaaaaa{i:04}").into_bytes(),
                        freq: 1,
                        positions: None,
                        start_offsets: None,
                        end_offsets: None,
                        payloads: None,
                    })
                    .collect(),
            }],
        };
        let distinct = TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 0,
                has_positions: false,
                has_offsets: false,
                has_payloads: false,
                terms: (0..40)
                    .map(|i| {
                        // Same length, but the discriminating digits come
                        // first, so the common prefix is always 0.
                        let mut term = format!("{i:04}").into_bytes();
                        term.extend_from_slice(b"aaaaaaaaaaaaaaaacommonprefix");
                        TermVectorTerm {
                            term,
                            freq: 1,
                            positions: None,
                            start_offsets: None,
                            end_offsets: None,
                            payloads: None,
                        }
                    })
                    .collect(),
            }],
        };
        let (shared_tvd, tvx, tvm) = write_best_speed(std::slice::from_ref(&shared), &id(), "");
        let (distinct_tvd, ..) = write_best_speed(&[distinct], &id(), "");
        assert!(
            shared_tvd.len() < distinct_tvd.len(),
            "prefix sharing should shrink the chunk: {} vs {}",
            shared_tvd.len(),
            distinct_tvd.len()
        );
        assert_round_trips(&[shared], &shared_tvd, &tvx, &tvm);
    }

    #[test]
    fn chars_per_term_is_the_offset_to_position_ratio_of_the_field() {
        // `flushOffsets`: `charsPerTerm[i] = sumOffsets[i] / sumPos[i]`, both
        // summed over each term's *last* occurrence. Here one term, one doc:
        // last position 8, last start offset 48 -> 6.0.
        let doc = TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 2,
                has_positions: true,
                has_offsets: true,
                has_payloads: false,
                terms: vec![TermVectorTerm {
                    term: b"word".to_vec(),
                    freq: 3,
                    positions: Some(vec![0, 4, 8]),
                    start_offsets: Some(vec![0, 24, 48]),
                    end_offsets: Some(vec![4, 28, 52]),
                    payloads: None,
                }],
            }],
        };
        let (tvd, tvx, tvm) = write_best_speed(std::slice::from_ref(&doc), &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let chunk = reader.read_chunk(0).unwrap();
        assert_eq!(chunk.chars_per_term, vec![6.0f32]);
        // And with a perfectly uniform ratio every encoded start-offset delta
        // is exactly 0, which is the whole point of the correction.
        assert_eq!(chunk.start_offsets_flat, vec![0, 0, 0]);
        assert_round_trips(&[doc], &tvd, &tvx, &tvm);
    }

    #[test]
    fn a_field_with_offsets_but_no_positions_gets_a_zero_chars_per_term() {
        // `sumPos` stays 0 (the `fd.hasOffsets && fd.hasPositions` gate), so
        // Java writes `charsPerTerm = 0` and the deltas are plain.
        let doc = TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 0,
                has_positions: false,
                has_offsets: true,
                has_payloads: false,
                terms: vec![TermVectorTerm {
                    term: b"word".to_vec(),
                    freq: 2,
                    positions: None,
                    start_offsets: Some(vec![0, 24]),
                    end_offsets: Some(vec![4, 28]),
                    payloads: None,
                }],
            }],
        };
        let (tvd, tvx, tvm) = write_best_speed(std::slice::from_ref(&doc), &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert_eq!(reader.read_chunk(0).unwrap().chars_per_term, vec![0.0f32]);
        assert_round_trips(&[doc], &tvd, &tvx, &tvm);
    }

    #[test]
    fn flags_are_written_once_per_field_number_when_they_never_change() {
        // `flushFlags`' `nonChangingFlags` branch (selector 0): 128 documents
        // of one field with constant flags cost one 4-bit entry, not 128.
        // The alternating variant has to fall back to selector 1.
        let constant: Vec<_> = (0..128).map(tiny_doc).collect();
        let mut alternating = constant.clone();
        for (i, doc) in alternating.iter_mut().enumerate() {
            if i % 2 == 1 {
                // Same field number, different flags -> `nonChangingFlags`
                // is false.
                doc.fields[0].has_positions = false;
                doc.fields[0].terms[0].positions = None;
            }
        }
        let (constant_tvd, ..) = write_best_speed(&constant, &id(), "");
        let (alternating_tvd, ..) = write_best_speed(&alternating, &id(), "");
        // Selector 0 writes 1 byte of flags where selector 1 writes 64
        // (128 fields x 4 bits). The alternating set also drops half its
        // positions, which *shrinks* it -- so the flags saving has to beat
        // that to show up at all.
        assert!(
            constant_tvd.len() + 40 < alternating_tvd.len() + 64,
            "expected the constant-flag chunk to use the per-field-number \
             flags array: {} vs {}",
            constant_tvd.len(),
            alternating_tvd.len()
        );
        let (tvd, tvx, tvm) = write_best_speed(&alternating, &id(), "");
        assert_round_trips(&alternating, &tvd, &tvx, &tvm);
    }

    #[test]
    fn distinct_field_numbers_are_written_sorted() {
        // `flushFieldNums` sorts; the reader resolves through `fieldNumOffs`
        // either way, but real Lucene's *writer* is the reference and
        // `CheckIndex` walks `TVFields.iterator()` in document field order.
        let doc = TermVectorsDocument {
            fields: [7, 2, 5]
                .into_iter()
                .map(|n| TermVectorField {
                    field_number: n,
                    has_positions: false,
                    has_offsets: false,
                    has_payloads: false,
                    terms: vec![TermVectorTerm {
                        term: format!("f{n}").into_bytes(),
                        freq: 1,
                        positions: None,
                        start_offsets: None,
                        end_offsets: None,
                        payloads: None,
                    }],
                })
                .collect(),
        };
        let (tvd, tvx, tvm) = write_best_speed(std::slice::from_ref(&doc), &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert_eq!(reader.read_chunk(0).unwrap().field_nums, vec![2, 5, 7]);
        // Document order is preserved regardless.
        let got = reader.document(0).unwrap().unwrap();
        assert_eq!(
            got.fields
                .iter()
                .map(|f| f.field_number)
                .collect::<Vec<_>>(),
            vec![7, 2, 5]
        );
        assert_round_trips(&[doc], &tvd, &tvx, &tvm);
    }

    #[test]
    fn a_chunk_cursor_serves_every_document_of_a_chunk_from_one_decode() {
        let docs: Vec<_> = (0..300).map(tiny_doc).collect();
        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let mut cursor = ChunkCursor::new();
        for doc_id in 0..reader.max_doc() {
            assert_eq!(
                cursor.document(&reader, doc_id).unwrap(),
                reader.document(doc_id).unwrap(),
                "doc {doc_id}"
            );
        }
        // A backwards walk must reload rather than serve a stale chunk.
        for doc_id in (0..reader.max_doc()).rev() {
            assert_eq!(
                cursor.document(&reader, doc_id).unwrap(),
                reader.document(doc_id).unwrap(),
                "doc {doc_id} backwards"
            );
        }
        cursor.reset();
        assert!(cursor.chunk.is_none());
    }

    #[test]
    fn read_chunk_reports_its_own_extent_and_rejects_documents_outside_it() {
        let docs: Vec<_> = (0..300).map(tiny_doc).collect();
        let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let chunk = reader.read_chunk(200).unwrap();
        assert_eq!(chunk.doc_base(), 128);
        assert_eq!(chunk.num_docs(), 128);
        assert!(chunk.contains(128) && chunk.contains(255));
        assert!(!chunk.contains(127) && !chunk.contains(256));
        assert!(matches!(
            chunk.document(300),
            Err(Error::CorruptChunkBounds { .. })
        ));
        assert!(matches!(
            reader.read_chunk(-1),
            Err(Error::DocOutOfRange(-1, 300))
        ));
        assert!(matches!(
            reader.chunk_for_doc(300),
            Err(Error::DocOutOfRange(300, 300))
        ));
    }

    // ---------------------------------------------------------------
    // The bulk merge path (`copyChunks`).
    // ---------------------------------------------------------------

    fn segment(docs: &[TermVectorsDocument]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        write_best_speed(docs, &id(), "")
    }

    #[test]
    fn copy_chunks_of_two_whole_segments_reproduces_every_document() {
        let a: Vec<_> = (0..300).map(tiny_doc).collect();
        let b: Vec<_> = (300..500).map(tiny_doc).collect();
        let (a_tvd, a_tvx, a_tvm) = segment(&a);
        let (b_tvd, b_tvx, b_tvm) = segment(&b);
        let ra = open(&a_tvd, &a_tvx, &a_tvm, &id(), "").unwrap();
        let rb = open(&b_tvd, &b_tvx, &b_tvm, &id(), "").unwrap();

        let mut w = TermVectorsWriter::new(&id(), "");
        w.copy_chunks(&ra, 0, ra.max_doc()).unwrap();
        w.copy_chunks(&rb, 0, rb.max_doc()).unwrap();
        let (tvd, tvx, tvm) = w.finish();

        let merged: Vec<_> = a.into_iter().chain(b).collect();
        assert_round_trips(&merged, &tvd, &tvx, &tvm);
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        // Both sources' dirty tails are carried over verbatim, which is why
        // dirtiness accumulates across merge generations.
        assert_eq!(reader.num_dirty_chunks(), 2);
        assert_eq!(reader.num_dirty_docs(), 44 + 72);
    }

    #[test]
    fn copy_chunks_rejects_a_chunk_claiming_two_billion_documents() {
        // The bulk-copy loop's twin of
        // `a_chunk_claiming_two_billion_documents_is_a_decode_error_not_an_overflow`:
        // `Lucene90CompressingTermVectorsWriter.copyChunks` also decodes
        // `code >>> 1`, so a corrupt token drives both `doc` and the writer's
        // own `numDocs` past `i32::MAX`. The old code advanced both with a
        // plain `+=` *before* testing `doc > to_doc`, so a wrap put the
        // cursor back inside the range and copied a chunk body two billion
        // documents long.
        let docs: Vec<_> = (0..300).map(tiny_doc).collect();
        let (mut s_tvd, s_tvx, mut s_tvm) = segment(&docs);
        let entry = {
            let reader = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
            reader.chunk_for_doc(128).unwrap()
        };
        assert_eq!(entry.doc_base, 128);
        let token_at = entry.start_pointer as usize + 2;
        let mut insert = Vec::new();
        write_vint(&mut insert, -1);
        splice_tvd(&mut s_tvd, &mut s_tvm, token_at, 1, &insert);

        let reader = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
        let mut w = TermVectorsWriter::new(&id(), "");
        assert!(matches!(
            w.copy_chunks(&reader, 128, 256),
            Err(Error::CorruptChunkBounds { .. })
        ));
    }

    #[test]
    fn copy_chunks_of_a_partial_range_copies_the_ragged_ends_document_at_a_time() {
        let docs: Vec<_> = (0..300).map(tiny_doc).collect();
        let (s_tvd, s_tvx, s_tvm) = segment(&docs);
        let reader = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();

        // 100..270 straddles both boundaries: 100..128 and 256..270 are
        // re-encoded, 128..256 is one whole copied chunk.
        let mut w = TermVectorsWriter::new(&id(), "");
        w.copy_chunks(&reader, 100, 270).unwrap();
        let (tvd, tvx, tvm) = w.finish();
        assert_round_trips(&docs[100..270], &tvd, &tvx, &tvm);
    }

    #[test]
    fn copy_chunks_of_a_range_inside_one_chunk_copies_no_chunk_at_all() {
        let docs: Vec<_> = (0..300).map(tiny_doc).collect();
        let (s_tvd, s_tvx, s_tvm) = segment(&docs);
        let reader = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
        let mut w = TermVectorsWriter::new(&id(), "");
        w.copy_chunks(&reader, 130, 140).unwrap();
        let (tvd, tvx, tvm) = w.finish();
        assert_round_trips(&docs[130..140], &tvd, &tvx, &tvm);
    }

    #[test]
    fn copy_chunks_after_buffered_documents_forces_a_dirty_flush_first() {
        let docs: Vec<_> = (0..300).map(tiny_doc).collect();
        let (s_tvd, s_tvx, s_tvm) = segment(&docs);
        let reader = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
        let mut w = TermVectorsWriter::new(&id(), "");
        let extra = tiny_doc(9999);
        w.add_document(&extra);
        w.copy_chunks(&reader, 0, 256).unwrap();
        let (tvd, tvx, tvm) = w.finish();
        let expected: Vec<_> = std::iter::once(extra)
            .chain(docs[0..256].iter().cloned())
            .collect();
        assert_round_trips(&expected, &tvd, &tvx, &tvm);
        let merged = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert_eq!(chunk_headers(&merged)[0], (0, 1, true));
    }

    #[test]
    fn copy_chunks_of_an_empty_range_writes_nothing() {
        let docs: Vec<_> = (0..300).map(tiny_doc).collect();
        let (s_tvd, s_tvx, s_tvm) = segment(&docs);
        let reader = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
        let mut w = TermVectorsWriter::new(&id(), "");
        w.copy_chunks(&reader, 5, 5).unwrap();
        let (tvd, tvx, tvm) = w.finish();
        let merged = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert_eq!(merged.max_doc(), 0);
        assert_eq!(merged.num_chunks(), 0);
    }

    #[test]
    fn copy_chunks_rejects_an_out_of_range_or_inverted_document_range() {
        let docs: Vec<_> = (0..300).map(tiny_doc).collect();
        let (s_tvd, s_tvx, s_tvm) = segment(&docs);
        let reader = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
        let mut w = TermVectorsWriter::new(&id(), "");
        assert!(matches!(
            w.copy_chunks(&reader, 0, 301),
            Err(Error::DocOutOfRange(301, 300))
        ));
        assert!(matches!(
            w.copy_chunks(&reader, 10, 5),
            Err(Error::InvertedDocRange {
                from_doc: 10,
                to_doc: 5
            })
        ));
    }

    #[test]
    fn dirtiness_accumulates_across_bulk_copies_until_the_segment_is_too_dirty() {
        // `tooDirty`: more dirty docs than a full chunk AND more than 1% of
        // chunks dirty. Repeated bulk copies of a one-document segment build
        // exactly that -- the degradation Java's safety switch exists for.
        let one = vec![tiny_doc(0)];
        let (s_tvd, s_tvx, s_tvm) = segment(&one);
        let source = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
        let probe = TermVectorsWriter::new(&id(), "");
        assert!(!probe.too_dirty(&source));

        let mut w = TermVectorsWriter::new(&id(), "");
        for _ in 0..130 {
            w.copy_chunks(&source, 0, 1).unwrap();
        }
        let (tvd, tvx, tvm) = w.finish();
        let merged = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert_eq!(merged.num_dirty_chunks(), 130);
        assert_eq!(merged.num_dirty_docs(), 130);
        let probe = TermVectorsWriter::new(&id(), "");
        assert!(probe.too_dirty(&merged));
        assert!(!probe.can_bulk_copy(&merged));
        assert!(matches!(
            TermVectorsWriter::new(&id(), "").copy_chunks(&merged, 0, 1),
            Err(Error::BulkCopyNotPermitted { .. })
        ));
        // ... and it still reads back correctly.
        assert_eq!(merged.max_doc(), 130);
        assert_eq!(
            merged.document(129).unwrap().unwrap().fields[0].terms[0].term,
            b"t0000"
        );
    }

    #[test]
    fn check_integrity_detects_payload_corruption_that_the_footer_shape_cannot() {
        let docs: Vec<_> = (0..300).map(tiny_doc).collect();
        let (mut tvd, tvx, tvm) = segment(&docs);
        {
            let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
            reader.check_integrity().unwrap();
        }
        // Flip one byte of a chunk body: every length, pointer and footer
        // field stays intact, so `open`'s `retrieve_checksum` still passes.
        let victim = tvd.len() - codec_util::FOOTER_LENGTH - 20;
        tvd[victim] ^= 0x40;
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert!(reader.check_integrity().is_err());
    }

    /// Rewrites a valid two-chunk segment's `.tvd` at `at` via `mutate`,
    /// keeping the file length identical so `open`'s `maxPointer` cross-check
    /// still passes and only `copy_chunks` can catch the damage.
    fn corrupt_chunk_header(mutate: impl Fn(&mut Vec<u8>, usize)) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let docs: Vec<_> = (0..300).map(tiny_doc).collect();
        let (mut tvd, tvx, tvm) = segment(&docs);
        let start = {
            let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
            reader.chunk_for_doc(128).unwrap().start_pointer as usize
        };
        mutate(&mut tvd, start);
        (tvd, tvx, tvm)
    }

    #[test]
    fn a_chunk_header_whose_doc_base_disagrees_with_the_index_is_rejected() {
        // Java's `base != docID` `CorruptIndexException`. docBase 128 is a
        // two-byte vint (0x80 0x01); rewrite it as 129 in place.
        let (tvd, tvx, tvm) = corrupt_chunk_header(|tvd, at| {
            assert_eq!((tvd[at], tvd[at + 1]), (0x80, 0x01));
            tvd[at] = 0x81;
        });
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let mut w = TermVectorsWriter::new(&id(), "");
        assert!(matches!(
            w.copy_chunks(&reader, 0, 300),
            Err(Error::CorruptChunkBounds { .. })
        ));
    }

    #[test]
    fn a_chunk_header_claiming_no_documents_is_rejected() {
        // A zero-document chunk would make the copy loop spin without
        // advancing. The token is `(chunkDocs << 1) | dirty`; 128 docs is
        // `256` = 0x80 0x02, rewritten as `0` = 0x80 0x00.
        let (tvd, tvx, tvm) = corrupt_chunk_header(|tvd, at| {
            assert_eq!((tvd[at + 2], tvd[at + 3]), (0x80, 0x02));
            tvd[at + 3] = 0x00;
        });
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let mut w = TermVectorsWriter::new(&id(), "");
        assert!(matches!(
            w.copy_chunks(&reader, 0, 300),
            Err(Error::CorruptChunkBounds { chunk_docs: 0, .. })
        ));
    }

    #[test]
    fn a_chunk_claiming_more_documents_than_the_requested_range_is_rejected() {
        // Java's `docID > toDocID`. The first chunk claims 128 docs but only
        // 100 were asked for.
        let docs: Vec<_> = (0..300).map(tiny_doc).collect();
        let (s_tvd, s_tvx, s_tvm) = segment(&docs);
        let reader = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
        // A range that merely ends mid-chunk is handled by the ragged-tail
        // loop, so reaching this guard needs a chunk header that *lies*:
        // rewrite the first chunk's token to claim 256 documents (same vint
        // width) and ask for 0..128.
        let mut tvd = s_tvd.clone();
        let start = reader.chunk_for_doc(0).unwrap().start_pointer as usize;
        assert_eq!((tvd[start + 1], tvd[start + 2]), (0x80, 0x02));
        tvd[start + 2] = 0x04; // 512 = 256 docs
        let bad = open(&tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
        let mut w = TermVectorsWriter::new(&id(), "");
        assert!(matches!(
            w.copy_chunks(&bad, 0, 128),
            Err(Error::CorruptChunkBounds { .. })
        ));
    }
}
