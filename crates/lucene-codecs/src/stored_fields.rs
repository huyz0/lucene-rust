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
//!   a `token` (vint: `chunkDocs = token >> 2`, `sliced = token & 1`,
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
//! field's value in one of six encodings -- see [`read_field`].

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
enum Mode {
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

    let max_doc = meta_input.read_i32()?;
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
    let expected_fdt_len = max_pointer as usize + codec_util::FOOTER_LENGTH;
    if fdt.len() != expected_fdt_len {
        return Err(lucene_store::Error::Corrupted(format!(
            ".fdt length should be {expected_fdt_len} bytes (maxPointer={max_pointer} + footer), but is {}",
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
    pub fn document(&self, doc_id: i32) -> Result<Document> {
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
        let chunk_docs = token >> 2;
        if doc_id < doc_base
            || doc_id >= doc_base + chunk_docs
            || doc_base + chunk_docs > self.max_doc
        {
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
            (vec![n], vec![0i64, len])
        } else {
            let num_stored_fields = read_bulk_ints(&mut input, chunk_docs as usize)?;
            let raw_lengths = read_bulk_ints(&mut input, chunk_docs as usize)?;
            let mut offsets = Vec::with_capacity(chunk_docs as usize + 1);
            offsets.push(0i64);
            for &len in &raw_lengths {
                offsets.push(offsets.last().unwrap() + len);
            }
            for i in 0..chunk_docs as usize {
                let len = offsets[i + 1] - offsets[i];
                if (len == 0) != (num_stored_fields[i] == 0) {
                    return Err(Error::LengthFieldCountMismatch {
                        length: len,
                        num_stored_fields: num_stored_fields[i],
                    });
                }
            }
            (num_stored_fields, offsets)
        };

        let index = (doc_id - doc_base) as usize;
        let doc_offset = offsets[index];
        let doc_length = offsets[index + 1] - doc_offset;
        let total_length = *offsets.last().unwrap();
        let doc_num_stored_fields = num_stored_fields[index];

        if doc_length == 0 {
            return Ok(Document::default());
        }

        let chunk_size = self.chunk_size as i64;
        let decompressed = if sliced {
            let mut out = Vec::with_capacity(total_length as usize);
            let mut remaining = total_length;
            while remaining > 0 {
                let to_decompress = remaining.min(chunk_size);
                out.extend(decompress_unit(
                    self.mode,
                    &mut input,
                    to_decompress as usize,
                )?);
                remaining -= to_decompress;
            }
            out
        } else {
            decompress_unit(self.mode, &mut input, total_length as usize)?
        };

        let doc_bytes = decompressed
            .get(doc_offset as usize..(doc_offset + doc_length) as usize)
            .ok_or(lucene_store::Error::Eof { offset: 0 })?;
        let mut doc_input = SliceInput::new(doc_bytes);

        let mut fields = Vec::with_capacity(doc_num_stored_fields as usize);
        for _ in 0..doc_num_stored_fields {
            let info_and_bits = doc_input.read_vlong()?;
            let field_number = (info_and_bits >> TYPE_BITS) as i32;
            let bits = info_and_bits & TYPE_MASK;
            let value = read_field(&mut doc_input, bits)?;
            fields.push(StoredField {
                field_number,
                value,
            });
        }

        Ok(Document { fields })
    }
}

/// Port of `Lucene90CompressingStoredFieldsWriter` -- the write-side
/// counterpart of [`open`]/[`StoredFieldsReader::document`], `Mode.BEST_SPEED`
/// only. First slice of this port's write path (PLAN.md Phase 5): every
/// document goes into a **single chunk** and a **single, un-sliced,
/// zero-dictionary LZ4 unit** (no dictionary/sub-block splitting, no
/// multi-chunk flushing by doc count or byte size) -- correctness and wire
/// compatibility first, matching this port's decode-fully stance on the
/// read side: a real writer's chunking heuristics are a later concern, not
/// a correctness one. `.fdt`/`.fdx`/`.fdm` produced this way are valid,
/// checksummed, Java-Lucene-openable files; only the ceiling on
/// `chunk_docs` (kept under the bulk `read_bulk_ints`' 128-value
/// transposed-block threshold, see [`write_bulk_ints`]) differs from what a
/// real flush would produce.
///
/// The one payload-covering block is now compressed with a **real**
/// back-reference LZ4 compressor ([`crate::lz4::compress`], a scoped-down
/// port of `LZ4.compressWithDictionary` -- see its doc comment for exactly
/// what's scoped out: no preset dictionary, and the simpler
/// `FastCompressionHashTable` match-finding strategy rather than
/// `HighCompressionHashTable`'s hash-chain search). The zero-length
/// "dictionary" unit that precedes it is still emitted via
/// [`encode_literal_lz4`] (a real compressor on an empty input degenerates
/// to the same single zero token anyway, so there's no reason to route it
/// through the real match-finder).
pub fn write_best_speed(
    docs: &[Document],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    assert!(
        docs.len() < 128,
        "write_best_speed's bulk per-doc arrays only implement the scalar-tail \
         encoding (see write_bulk_ints); the 128-value transposed-block path \
         isn't written yet, so chunks must stay under 128 docs"
    );

    let payloads: Vec<Vec<u8>> = docs.iter().map(serialize_doc).collect();
    let num_stored_fields: Vec<i64> = docs.iter().map(|d| d.fields.len() as i64).collect();
    let lengths: Vec<i64> = payloads.iter().map(|p| p.len() as i64).collect();
    let total_length: i64 = lengths.iter().sum();
    let chunk_docs = docs.len() as i32;
    let max_doc = chunk_docs;

    let mut fdt = Vec::new();
    codec_util::write_index_header(
        &mut fdt,
        DATA_CODEC_BEST_SPEED,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    let chunk_start = fdt.len() as i64;

    fdt.write_vint(0); // docBase
    fdt.write_vint(chunk_docs << 2); // token: sliced=0, dirty=0

    if chunk_docs == 1 {
        fdt.write_vint(num_stored_fields[0] as i32);
        fdt.write_vint(lengths[0] as i32);
    } else if chunk_docs > 0 {
        write_bulk_ints(&mut fdt, &num_stored_fields);
        write_bulk_ints(&mut fdt, &lengths);
    }

    if total_length > 0 {
        let payload: Vec<u8> = payloads.concat();
        // One zero-length "dictionary" unit (a single LZ4 token byte, see
        // `lz4::decompress`'s zero-length handling) followed by one
        // literal-only block holding every doc's bytes verbatim.
        let dict_unit = encode_literal_lz4(&[]);
        let block_unit = lz4::compress(&payload);
        fdt.write_vint(0); // dictLength
        fdt.write_vint(payload.len() as i32); // blockLength (one block covers everything)
        fdt.write_vint(dict_unit.len() as i32); // dict's compressed length
        fdt.write_vint(block_unit.len() as i32); // this one block's compressed length
        fdt.write_bytes(&dict_unit);
        fdt.write_bytes(&block_unit);
    }

    let max_pointer = fdt.len() as i64;
    codec_util::write_footer(&mut fdt);

    let block_shift = 0u32;
    let docs_values = [0i64, max_doc as i64];
    let start_pointers_values = [chunk_start, max_pointer];

    let mut fdx = Vec::new();
    codec_util::write_index_header(
        &mut fdx,
        INDEX_CODEC,
        INDEX_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    let docs_start_pointer = fdx.len() as i64;
    let (docs_meta_bytes, docs_data_bytes) = direct_monotonic::write(&docs_values, block_shift);
    fdx.write_bytes(&docs_data_bytes);
    let docs_end_pointer = fdx.len() as i64;
    let (start_pointers_meta_bytes, start_pointers_data_bytes) =
        direct_monotonic::write(&start_pointers_values, block_shift);
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
    fdm.write_vint(80 * 1024); // chunkSize (unused when nothing is sliced)
    fdm.write_i32(max_doc);
    fdm.write_i32(block_shift as i32);
    fdm.write_i32(2); // index_num_chunks = 1 real chunk + 1 sentinel
    fdm.write_i64(docs_start_pointer);
    fdm.write_bytes(&docs_meta_bytes);
    fdm.write_i64(docs_end_pointer);
    fdm.write_bytes(&start_pointers_meta_bytes);
    fdm.write_i64(start_pointers_end_pointer);
    fdm.write_i64(max_pointer);
    fdm.write_vlong(1); // numChunks (outer)
    fdm.write_vlong(0); // numDirtyChunks
    fdm.write_vlong(0); // numDirtyDocs
    codec_util::write_footer(&mut fdm);

    (fdt, fdx, fdm)
}

/// Port of `StoredFieldsInts`'s bulk per-doc array encode -- **scalar-tail
/// path only** (see [`write_best_speed`]'s doc comment): correct only for
/// `values.len() < 128`, since `read_bulk_ints`'s 128-value transposed-block
/// decode path is unimplemented on the write side so far.
fn write_bulk_ints(out: &mut Vec<u8>, values: &[i64]) {
    debug_assert!(values.len() < 128);
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
    for &v in values {
        match bpv {
            8 => out.push(v as u8),
            16 => out.extend_from_slice(&(v as u16).to_le_bytes()),
            32 => out.extend_from_slice(&(v as u32).to_le_bytes()),
            _ => unreachable!("bpv is always 8, 16, or 32"),
        }
    }
}

fn serialize_doc(doc: &Document) -> Vec<u8> {
    let mut out = Vec::new();
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
        write_field(&mut out, &field.value);
    }
    out
}

/// Port of `Lucene90CompressingStoredFieldsWriter.writeField` (encode side
/// of [`read_field`]). Unlike Java's `writeTLong`/`writeZFloat`/
/// `writeZDouble`, which pick the shortest of several encodings for a given
/// value, this always emits the full/worst-case encoding: correct for every
/// value, just not minimal -- a later optimization, not a correctness
/// concern (matches this module's stance on the write path generally).
fn write_field(out: &mut Vec<u8>, value: &FieldValue) {
    match value {
        FieldValue::Binary(b) => {
            out.write_vint(b.len() as i32);
            out.write_bytes(b);
        }
        FieldValue::String(s) => out.write_string(s),
        FieldValue::Int(v) => write_zint(out, *v),
        FieldValue::Float(v) => write_zfloat_full(out, *v),
        FieldValue::Long(v) => write_tlong_full(out, *v),
        FieldValue::Double(v) => write_zdouble_full(out, *v),
    }
}

/// Port of `DataOutput.writeZInt` (32-bit zigzag, distinct from the 64-bit
/// `writeZLong` [`DataOutput::write_zlong`] provided elsewhere in this port).
fn write_zint(out: &mut Vec<u8>, v: i32) {
    let zigzag = ((v << 1) ^ (v >> 31)) as u32;
    out.write_vint(zigzag as i32);
}

/// Always the full 5-byte encoding: marker `0xFF` + 4 raw bytes of `v`'s
/// IEEE-754 bit pattern.
fn write_zfloat_full(out: &mut Vec<u8>, v: f32) {
    out.push(0xFF);
    out.write_i32(v.to_bits() as i32);
}

/// Always the full 9-byte encoding: marker `0xFF` + 8 raw bytes of `v`'s
/// IEEE-754 bit pattern.
fn write_zdouble_full(out: &mut Vec<u8>, v: f64) {
    out.push(0xFF);
    out.write_i64(v.to_bits() as i64);
}

/// Always sets the `0x20` "more bits follow" flag and no time-unit
/// multiplier (`header & 0xC0 == 0`), so the low 5 bits plus a trailing
/// vlong reconstruct the full zigzag-encoded value regardless of magnitude.
fn write_tlong_full(out: &mut Vec<u8>, v: i64) {
    let zigzag = lucene_util::zigzag::encode(v);
    let header = 0x20u8 | (zigzag & 0x1F) as u8;
    out.push(header);
    out.write_vlong((zigzag >> 5) as i64);
}

/// A single, self-contained LZ4 "literal run" block wrapping `bytes`
/// verbatim -- no back-reference matches, valid per the LZ4 block spec
/// (a token's high nibble is the literal length, extended past 15 via
/// 0xFF continuation bytes; omitting the final match entirely is legal
/// when the literal run consumes the whole block).
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

/// Port of `Lucene90CompressingStoredFieldsReader.readField` (the decode
/// side of `StoredFieldsInts`'s sibling per-field value encoding).
fn read_field(input: &mut SliceInput, bits: i64) -> Result<FieldValue> {
    match bits {
        TYPE_BYTE_ARR => {
            let length = input.read_vint()? as usize;
            let mut buf = vec![0u8; length];
            input.read_bytes(&mut buf)?;
            Ok(FieldValue::Binary(buf))
        }
        TYPE_STRING => Ok(FieldValue::String(input.read_string()?)),
        TYPE_NUMERIC_INT => Ok(FieldValue::Int(read_zint(input)?)),
        TYPE_NUMERIC_FLOAT => Ok(FieldValue::Float(read_zfloat(input)?)),
        TYPE_NUMERIC_LONG => Ok(FieldValue::Long(read_tlong(input)?)),
        TYPE_NUMERIC_DOUBLE => Ok(FieldValue::Double(read_zdouble(input)?)),
        other => Err(Error::UnknownTypeTag(other)),
    }
}

/// Port of `DataInput.readZInt`: `BitUtil.zigZagDecode` applied to a 32-bit
/// vint (distinct from [`lucene_util::zigzag::decode`], which is the 64-bit
/// vlong variant used elsewhere in this port).
fn read_zint(input: &mut SliceInput) -> Result<i32> {
    let v = input.read_vint()? as u32;
    Ok(((v >> 1) as i32) ^ -((v & 1) as i32))
}

/// Port of `Lucene90CompressingStoredFieldsReader.readZFloat`: 1-5 bytes,
/// small integral values (`-1..=125`) collapse to a single byte.
fn read_zfloat(input: &mut SliceInput) -> Result<f32> {
    let b = input.read_byte()? as i32;
    if b == 0xFF {
        Ok(f32::from_bits(input.read_i32()? as u32))
    } else if b & 0x80 != 0 {
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
    Ok(match header & DAY_ENCODING_MASK {
        SECOND_ENCODING => l * SECOND,
        HOUR_ENCODING => l * HOUR,
        DAY_ENCODING => l * DAY,
        0 => l,
        _ => unreachable!("only 2 bits, all 4 cases covered"),
    })
}

/// Port of `StoredFieldsInts.readInts`: a length-prefixed bulk int array
/// with three shapes (all-equal constant, or 8/16/32-bit fixed-width),
/// **without** Java's bit-transposed 128-value SIMD-friendly block layout --
/// ported as a plain per-value loop instead, since this port has no bulk
/// per-block hot path yet to justify the transposition's complexity (see
/// `rust-performance` skill). The on-disk bytes are still read exactly:
/// Java's transposed blocks are a *storage* layout, not a different value
/// set, so a value-by-value reader produces identical results, just via
/// [`crate::direct_reader`]-style sequential decode instead of a bulk
/// SIMD-shaped one.
fn read_bulk_ints(input: &mut SliceInput, count: usize) -> Result<Vec<i64>> {
    let bpv = input.read_byte()?;
    match bpv {
        0 => {
            let v = input.read_vint()? as i64;
            Ok(vec![v; count])
        }
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

            let mut out = vec![0i64; count];
            let mut k = 0usize;
            while k + BLOCK_SIZE <= count {
                let mut words = vec![0i64; num_words];
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

/// Decompresses one preset-dictionary compression unit (`LZ4WithPresetDict
/// CompressionMode` for `Mode.BEST_SPEED`, `DeflateWithPresetDictCompression
/// Mode` for `Mode.BEST_COMPRESSION`): a dictionary prefix (whose
/// *compressed* bytes come first) followed by fixed-size sub-blocks, each
/// able to reference back into the dictionary. Unlike Java's reader, this
/// always decompresses the whole unit -- there's no lazy/partial-read path
/// to preserve since this port hands back a fully materialized `Document`
/// rather than a streaming `DataInput`.
///
/// Both formats share `dictLength`/`blockLength` framing up front, but
/// differ in where each unit's *compressed*-length vint sits relative to
/// its own compressed bytes -- easy to get backwards, so this is worth
/// spelling out precisely:
/// - LZ4 (`LZ4WithPresetDictCompressionMode.readCompressedLengths`) batches
///   **every** unit's compressed length (the dictionary's, then each
///   block's) together up front, before any of the actual compressed
///   bytes. LZ4 is self-terminating from the output length alone, so this
///   port reads those vints here and discards them (Java's reader only
///   needs them to support seeking without decompressing everything before
///   a wanted offset, which a full sequential decode doesn't need).
/// - DEFLATE (`DeflateWithPresetDictCompressionMode.decompress`/
///   `doDecompress`) interleaves each unit's compressed-length vint
///   immediately before that same unit's compressed bytes -- not batched at
///   all. DEFLATE isn't self-terminating, so [`deflate::decompress`] needs
///   that length passed in explicitly, read at the point of use.
fn decompress_unit(mode: Mode, input: &mut SliceInput, original_length: usize) -> Result<Vec<u8>> {
    if original_length == 0 {
        return Ok(Vec::new());
    }
    let dict_length = input.read_vint()? as usize;
    let block_length = input.read_vint()? as usize;
    let num_blocks = {
        let mut total = dict_length;
        let mut num_blocks = 0usize;
        while total < original_length {
            total += block_length;
            num_blocks += 1;
        }
        num_blocks
    };

    let mut buffer = vec![0u8; dict_length + block_length];
    let mut out = vec![0u8; original_length];

    match mode {
        Mode::BestSpeed => {
            input.read_vint()?; // dictionary's compressed length, unused
            for _ in 0..num_blocks {
                input.read_vint()?; // block's compressed length, unused
            }
            lz4::decompress(input, dict_length, &mut buffer, 0)?;
            out[..dict_length].copy_from_slice(&buffer[..dict_length]);

            let mut produced = 0usize;
            for _ in 0..num_blocks {
                let to_decompress = block_length.min(original_length - dict_length - produced);
                lz4::decompress(input, to_decompress, &mut buffer, dict_length)?;
                out[dict_length + produced..dict_length + produced + to_decompress]
                    .copy_from_slice(&buffer[dict_length..dict_length + to_decompress]);
                produced += to_decompress;
            }
        }
        Mode::BestCompression => {
            let dict_compressed_length = input.read_vint()? as usize;
            deflate::decompress(input, dict_compressed_length, dict_length, &mut buffer, 0)?;
            out[..dict_length].copy_from_slice(&buffer[..dict_length]);

            let mut produced = 0usize;
            for _ in 0..num_blocks {
                let to_decompress = block_length.min(original_length - dict_length - produced);
                let block_compressed_length = input.read_vint()? as usize;
                deflate::decompress(
                    input,
                    block_compressed_length,
                    to_decompress,
                    &mut buffer,
                    dict_length,
                )?;
                out[dict_length + produced..dict_length + produced + to_decompress]
                    .copy_from_slice(&buffer[dict_length..dict_length + to_decompress]);
                produced += to_decompress;
            }
        }
    }
    Ok(out)
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

    fn build_single_chunk_index_with_meta_overrides(
        doc_bytes: &[u8],
        num_chunks_outer: i32,
        num_dirty_chunks: i32,
        num_dirty_docs: i32,
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
        write_vint(&mut fdm, num_chunks_outer);
        write_vint(&mut fdm, num_dirty_chunks);
        write_vint(&mut fdm, num_dirty_docs);
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
        let out = vec![3u8]; // not one of 0/8/16/32
        let mut input = SliceInput::new(&out);
        assert!(matches!(
            read_bulk_ints(&mut input, 4),
            Err(Error::UnsupportedBulkIntWidth(3))
        ));
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
        assert_eq!(
            decompress_unit(Mode::BestSpeed, &mut input, 0).unwrap(),
            Vec::<u8>::new()
        );
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
        out.extend(decompress_unit(Mode::BestSpeed, &mut input, part_a.len()).unwrap());
        out.extend(decompress_unit(Mode::BestSpeed, &mut input, part_b.len()).unwrap());

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

    #[test]
    fn every_field_value_type_round_trips_through_read_field() {
        let mut int_payload = Vec::new();
        write_vint(&mut int_payload, lucene_util::zigzag::encode(-7) as i32);
        let mut input = SliceInput::new(&int_payload);
        assert_eq!(
            read_field(&mut input, TYPE_NUMERIC_INT).unwrap(),
            FieldValue::Int(-7)
        );

        // zigzag(7) = 14, fits directly in the header's low 5 bits with no
        // continuation byte and no second/hour/day scale applied.
        let mut long_payload = vec![lucene_util::zigzag::encode(7) as u8];
        let mut input = SliceInput::new(&long_payload);
        assert_eq!(
            read_field(&mut input, TYPE_NUMERIC_LONG).unwrap(),
            FieldValue::Long(7)
        );

        let mut bin_payload = Vec::new();
        write_vint(&mut bin_payload, 3);
        bin_payload.extend_from_slice(b"xyz");
        let mut input = SliceInput::new(&bin_payload);
        assert_eq!(
            read_field(&mut input, TYPE_BYTE_ARR).unwrap(),
            FieldValue::Binary(b"xyz".to_vec())
        );

        let mut str_payload = Vec::new();
        write_string(&mut str_payload, "hi");
        let mut input = SliceInput::new(&str_payload);
        assert_eq!(
            read_field(&mut input, TYPE_STRING).unwrap(),
            FieldValue::String("hi".to_string())
        );

        long_payload = vec![((9i32 + 1) as u8) | 0x80];
        let mut input = SliceInput::new(&long_payload);
        assert_eq!(
            read_field(&mut input, TYPE_NUMERIC_FLOAT).unwrap(),
            FieldValue::Float(9.0)
        );

        long_payload = vec![((2i32 + 1) as u8) | 0x80];
        let mut input = SliceInput::new(&long_payload);
        assert_eq!(
            read_field(&mut input, TYPE_NUMERIC_DOUBLE).unwrap(),
            FieldValue::Double(2.0)
        );

        let mut input = SliceInput::new(&[]);
        assert!(matches!(
            read_field(&mut input, 6),
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
    fn write_zint_round_trips_through_read_zint() {
        for v in [0i32, 1, -1, i32::MIN, i32::MAX] {
            let mut out = Vec::new();
            write_zint(&mut out, v);
            let mut input = SliceInput::new(&out);
            assert_eq!(read_zint(&mut input).unwrap(), v, "value {v}");
        }
    }

    #[test]
    fn write_tlong_full_round_trips_through_read_tlong() {
        for v in [0i64, 1, -1, i64::MIN, i64::MAX, 1_000_000_000_000] {
            let mut out = Vec::new();
            write_tlong_full(&mut out, v);
            let mut input = SliceInput::new(&out);
            assert_eq!(read_tlong(&mut input).unwrap(), v, "value {v}");
        }
    }

    #[test]
    fn write_zfloat_full_round_trips_through_read_zfloat() {
        for v in [0.0f32, 1.5, -1.5, f32::MIN, f32::MAX] {
            let mut out = Vec::new();
            write_zfloat_full(&mut out, v);
            let mut input = SliceInput::new(&out);
            assert_eq!(read_zfloat(&mut input).unwrap(), v, "value {v}");
        }
    }

    #[test]
    fn write_zdouble_full_round_trips_through_read_zdouble() {
        for v in [0.0f64, 1.5, -1.5, f64::MIN, f64::MAX] {
            let mut out = Vec::new();
            write_zdouble_full(&mut out, v);
            let mut input = SliceInput::new(&out);
            assert_eq!(read_zdouble(&mut input).unwrap(), v, "value {v}");
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
}
