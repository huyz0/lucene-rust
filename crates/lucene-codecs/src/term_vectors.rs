//! Port of `org.apache.lucene.codecs.lucene90.Lucene90TermVectorsFormat`
//! (`.tvd` data + `.tvx` index + `.tvm` meta) — read-only.
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
//! This port decodes an entire chunk's fields in one pass (all docs, not
//! just the requested one) rather than replicating Java's skip-arithmetic
//! that materializes only the requested document's slice of each array --
//! the same decode-fully trade-off made throughout this port (`IndexedDISI`,
//! stored fields, the terms dictionary): correctness and simplicity over a
//! micro-optimization this phase doesn't need.
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
    #[error("invalid flags-array selector: {0}")]
    InvalidFlagsSelector(i32),
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

#[derive(Debug, Clone)]
pub struct TermVectorField {
    pub field_number: i32,
    pub has_positions: bool,
    pub has_offsets: bool,
    pub has_payloads: bool,
    pub terms: Vec<TermVectorTerm>,
}

#[derive(Debug, Clone, Default)]
pub struct TermVectorsDocument {
    pub fields: Vec<TermVectorField>,
}

pub struct TermVectorsReader<'d> {
    tvd: &'d [u8],
    tvx: &'d [u8],
    max_doc: i32,
    num_chunks: i64,
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
    let _chunk_size = meta_input.read_vint()?;

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

    let num_chunks = meta_input.read_vlong()?;
    if index_num_chunks != num_chunks + 1 {
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

    let expected_tvd_len = max_pointer as usize + codec_util::FOOTER_LENGTH;
    if tvd.len() != expected_tvd_len {
        return Err(lucene_store::Error::Corrupted(format!(
            ".tvd length should be {expected_tvd_len} bytes (maxPointer={max_pointer} + footer), but is {}",
            tvd.len()
        ))
        .into());
    }

    Ok(TermVectorsReader {
        tvd,
        tvx,
        max_doc,
        num_chunks,
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

    /// Reads the given document's term vectors, or `None` if it has none.
    pub fn document(&self, doc_id: i32) -> Result<Option<TermVectorsDocument>> {
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
        let chunk_docs = token >> 1;
        if doc_id < doc_base
            || doc_id >= doc_base + chunk_docs
            || doc_base + chunk_docs > self.max_doc
        {
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

        let mut field_offsets = Vec::with_capacity(chunk_docs as usize + 1);
        field_offsets.push(0i64);
        for &n in &num_fields_per_doc {
            field_offsets.push(field_offsets.last().unwrap() + n);
        }
        let doc_index = (doc_id - doc_base) as usize;
        let doc_field_start = field_offsets[doc_index] as usize;
        let doc_num_fields = num_fields_per_doc[doc_index] as usize;
        let total_fields = *field_offsets.last().unwrap() as usize;

        if doc_num_fields == 0 {
            return Ok(None);
        }

        // Distinct field numbers in this chunk: a headerless MSB-packed
        // array (see `packed_ints`), not `direct_reader`/`block_packed`.
        let token = input.read_byte()? as u32;
        let bits_per_field_num = token & 0x1F;
        let mut total_distinct_fields = token >> 5;
        if total_distinct_fields == 0x07 {
            total_distinct_fields += input.read_vint()? as u32;
        }
        total_distinct_fields += 1;
        let field_nums_byte_len =
            packed_ints::byte_count(total_distinct_fields as i64, bits_per_field_num);
        let mut field_nums_bytes = vec![0u8; field_nums_byte_len];
        input.read_bytes(&mut field_nums_bytes)?;
        let mut field_nums = Vec::with_capacity(total_distinct_fields as usize);
        for i in 0..total_distinct_fields as i64 {
            field_nums.push(packed_ints::get(&field_nums_bytes, bits_per_field_num, i)?);
        }

        // Field-number offsets (index into `field_nums`) for every field in
        // the chunk, plus per-field flags -- both `direct_reader`-encoded.
        let bits_per_off = direct_writer_bits_required(total_distinct_fields as i64 - 1);
        let all_field_num_offs_bytes = read_length_prefixed_slice(&mut input)?.to_vec();
        let flags_selector = input.read_vint()?;
        let all_flags: Vec<u8> = match flags_selector {
            0 => {
                let field_flags_bytes = read_length_prefixed_slice(&mut input)?.to_vec();
                let mut per_field_num_flags = Vec::with_capacity(total_distinct_fields as usize);
                for i in 0..total_distinct_fields as i64 {
                    per_field_num_flags
                        .push(direct_reader::get(&field_flags_bytes, FLAGS_BITS, i)? as u8);
                }
                let mut out = Vec::with_capacity(total_fields);
                for i in 0..total_fields as i64 {
                    let off = direct_reader::get(&all_field_num_offs_bytes, bits_per_off, i)?;
                    out.push(per_field_num_flags[off as usize]);
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
        let mut field_num_offs = Vec::with_capacity(total_fields);
        for i in 0..total_fields as i64 {
            field_num_offs.push(direct_reader::get(
                &all_field_num_offs_bytes,
                bits_per_off,
                i,
            )?);
        }

        // Term counts per field, `direct_reader`-encoded.
        let num_terms_bits = input.read_vint()? as u8;
        let num_terms_bytes = read_length_prefixed_slice(&mut input)?.to_vec();
        let mut num_terms = Vec::with_capacity(total_fields);
        for i in 0..total_fields as i64 {
            num_terms.push(direct_reader::get(&num_terms_bytes, num_terms_bits, i)?);
        }
        let total_terms: i64 = num_terms.iter().sum();

        let prefix_lengths = block_packed::decode_all(&mut input, total_terms)?;
        let suffix_lengths = block_packed::decode_all(&mut input, total_terms)?;
        let term_freqs_minus1 = block_packed::decode_all(&mut input, total_terms)?;

        let mut term_offsets = Vec::with_capacity(total_fields + 1);
        term_offsets.push(0i64);
        for &n in &num_terms {
            term_offsets.push(term_offsets.last().unwrap() + n);
        }

        let mut total_positions = 0i64;
        let mut total_offsets = 0i64;
        let mut total_payloads = 0i64;
        for field_idx in 0..total_fields {
            let f = all_flags[field_idx];
            let start = term_offsets[field_idx] as usize;
            let end = term_offsets[field_idx + 1] as usize;
            let field_freq_sum: i64 = term_freqs_minus1[start..end].iter().map(|&v| v + 1).sum();
            if f & FLAG_POSITIONS != 0 {
                total_positions += field_freq_sum;
            }
            if f & FLAG_OFFSETS != 0 {
                total_offsets += field_freq_sum;
            }
            if f & FLAG_PAYLOADS != 0 {
                total_payloads += field_freq_sum;
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
        let mut field_freq_sums = Vec::with_capacity(total_fields);
        let mut position_starts = Vec::with_capacity(total_fields);
        let mut offset_starts = Vec::with_capacity(total_fields);
        let mut payload_starts = Vec::with_capacity(total_fields);
        {
            let mut position_off = 0usize;
            let mut offset_off = 0usize;
            let mut payload_off = 0usize;
            for field_idx in 0..total_fields {
                let start = term_offsets[field_idx] as usize;
                let end = term_offsets[field_idx + 1] as usize;
                let field_freq_sum: i64 =
                    term_freqs_minus1[start..end].iter().map(|&v| v + 1).sum();
                field_freq_sums.push(field_freq_sum);
                let flags = all_flags[field_idx];
                position_starts.push(position_off);
                offset_starts.push(offset_off);
                payload_starts.push(payload_off);
                if flags & FLAG_POSITIONS != 0 {
                    position_off += field_freq_sum as usize;
                }
                if flags & FLAG_OFFSETS != 0 {
                    offset_off += field_freq_sum as usize;
                }
                if flags & FLAG_PAYLOADS != 0 {
                    payload_off += field_freq_sum as usize;
                }
            }
        }

        let total_suffix_len: i64 = suffix_lengths.iter().sum();
        let total_payload_len: i64 = payload_lengths_flat.iter().sum();
        let decompressed_len = (total_suffix_len + total_payload_len) as usize;
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
        for doc_idx in 0..chunk_docs as usize {
            let fstart = field_offsets[doc_idx] as usize;
            let fend = field_offsets[doc_idx + 1] as usize;
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
        let suffix_bytes: &[u8] = &decompressed;
        let payload_bytes: &[u8] = &decompressed;

        // Only fields inside the requested doc's range are materialized.
        let mut fields = Vec::with_capacity(doc_num_fields);
        for field_idx in doc_field_start..doc_field_start + doc_num_fields {
            let term_start = term_offsets[field_idx] as usize;
            let term_end = term_offsets[field_idx + 1] as usize;
            let term_count = term_end - term_start;
            let flags = all_flags[field_idx];
            let field_number = field_nums[field_num_offs[field_idx] as usize] as i32;
            let field_chars_per_term = if !chars_per_term.is_empty() {
                chars_per_term[field_num_offs[field_idx] as usize]
            } else {
                0.0
            };
            fields.push(build_field(FieldDecodeInput {
                field_number,
                flags,
                term_start,
                term_count,
                prefix_lengths: &prefix_lengths,
                suffix_lengths: &suffix_lengths,
                term_freqs_minus1: &term_freqs_minus1,
                suffix_bytes,
                suffix_byte_start: suffix_byte_starts[field_idx],
                positions_flat: &positions_flat,
                position_start: position_starts[field_idx],
                start_offsets_flat: &start_offsets_flat,
                lengths_flat: &lengths_flat,
                offset_start: offset_starts[field_idx],
                payload_bytes,
                payload_lengths_flat: &payload_lengths_flat,
                payload_start: payload_starts[field_idx],
                payload_byte_start: payload_byte_starts[field_idx],
                chars_per_term: field_chars_per_term,
            })?);
        }

        Ok(Some(TermVectorsDocument { fields }))
    }
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

    let mut terms = Vec::with_capacity(inp.term_count);
    let mut previous_term: Vec<u8> = Vec::new();
    let mut suffix_byte_off = inp.suffix_byte_start;
    let mut position_off = inp.position_start;
    let mut offset_off = inp.offset_start;
    let mut payload_off = inp.payload_start;
    let mut payload_byte_off = inp.payload_byte_start;

    for j in 0..inp.term_count {
        let idx = inp.term_start + j;
        let prefix_len = inp.prefix_lengths[idx] as usize;
        let suffix_len = inp.suffix_lengths[idx] as usize;
        let freq = (inp.term_freqs_minus1[idx] + 1) as usize;
        let term_len = (prefix_len + suffix_len) as i32;

        let mut term = Vec::with_capacity(prefix_len + suffix_len);
        term.extend_from_slice(&previous_term[..prefix_len]);
        let suffix = inp
            .suffix_bytes
            .get(suffix_byte_off..suffix_byte_off + suffix_len)
            .ok_or(lucene_store::Error::Eof {
                offset: suffix_byte_off,
            })?;
        term.extend_from_slice(suffix);
        suffix_byte_off += suffix_len;

        // Positions: absolute at this term's first occurrence, delta from
        // the *same term*'s previous occurrence thereafter.
        let mut term_positions = Vec::with_capacity(freq);
        if has_positions {
            let mut absolute = 0i32;
            for k in 0..freq {
                let raw = inp.positions_flat[position_off + k] as i32;
                absolute = if k == 0 { raw } else { absolute + raw };
                term_positions.push(absolute);
            }
        }

        let mut term_start_offsets = Vec::with_capacity(freq);
        let mut term_end_offsets = Vec::with_capacity(freq);
        if has_offsets {
            let mut absolute = 0i32;
            for k in 0..freq {
                let raw_delta = inp.start_offsets_flat[offset_off + k] as i32;
                let position_correction = if has_positions {
                    (inp.chars_per_term * inp.positions_flat[position_off + k] as f32) as i32
                } else {
                    0
                };
                let patched = raw_delta + position_correction;
                absolute = if k == 0 { patched } else { absolute + patched };
                let length = inp.lengths_flat[offset_off + k] as i32 + term_len;
                term_start_offsets.push(absolute);
                term_end_offsets.push(absolute + length);
            }
            offset_off += freq;
        }
        if has_positions {
            position_off += freq;
        }

        let mut term_payloads = Vec::with_capacity(freq);
        if has_payloads {
            for k in 0..freq {
                let len = inp.payload_lengths_flat[payload_off + k] as usize;
                let bytes = inp
                    .payload_bytes
                    .get(payload_byte_off..payload_byte_off + len)
                    .ok_or(lucene_store::Error::Eof {
                        offset: payload_byte_off,
                    })?;
                term_payloads.push(bytes.to_vec());
                payload_byte_off += len;
            }
            payload_off += freq;
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

fn read_length_prefixed_slice<'a>(input: &mut SliceInput<'a>) -> Result<&'a [u8]> {
    let len = input.read_vint()? as usize;
    let start = input.position();
    input.skip(len)?;
    Ok(input.slice(start, start + len)?)
}

#[cfg(test)]
mod tests {
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
        tvm.extend_from_slice(&2i32.to_le_bytes()); // index_num_chunks = totalChunks(1)+1
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
        let (tvd, tvx, tvm, _) = build_single_doc_chunk_with_meta_overrides(2, 0, 0);
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
        let payload = [b'c', b'a', b't'];
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
}
