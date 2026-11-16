//! Port of `org.apache.lucene.codecs.lucene90.Lucene90StoredFieldsFormat`
//! (`.fdt` data + `.fdx` index + `.fdm` meta) — read-only. Both compression
//! modes are supported: `Mode.BEST_SPEED` (the default; LZ4, ~80KB chunks)
//! and `Mode.BEST_COMPRESSION` (DEFLATE, ~480KB chunks). The mode is baked
//! into the `.fdt` data codec name itself (`...FastData` vs `...HighData`),
//! so `open` detects it there rather than needing the caller to specify it;
//! see [`Mode`] and [`compression::decompress`]. The test framework's
//! `CompressingCodec` formats (`Lucene90CompressingStoredFieldsFormat` with
//! its own name, suffix, [`Compression`] and geometry) are read and written
//! the same way, through [`Mode::Compressing`].
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
//!   [`compression::decompress`] if `!sliced`, or several `chunk_size`-decompressed
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

use std::sync::Arc;

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;
use lucene_store::index_output::FsIndexOutput;

use crate::compression::{self, CompressingFormat, Compression, Compressor};
use crate::direct_monotonic;

const DATA_CODEC_BEST_SPEED: &str = "Lucene90StoredFieldsFastData";
const DATA_CODEC_BEST_COMPRESSION: &str = "Lucene90StoredFieldsHighData";
/// OpenSearch custom-codecs' `CustomStoredFieldsZstd` and
/// `CustomStoredFieldsZstdNoDict` formats (`index.codec: zstd` and
/// `zstd_no_dict`).
pub const DATA_CODEC_ZSTD: &str = "CustomStoredFieldsZstd";
pub const DATA_CODEC_ZSTD_NO_DICT: &str = "CustomStoredFieldsZstdNoDict";
/// The Rust engine's paged zstd stored fields ([`Mode::ZstdPaged`]).
pub const DATA_CODEC_ZSTD_PAGED: &str = "PagedZstdStoredFieldsData";
/// The page [`Mode::ZstdPaged`] packs chunks into: a chunk that starts
/// inside a page also ends inside it, so reading one page is enough to
/// decode every chunk that starts there. A storage layer that caches whole
/// pages should use this size or a multiple of it.
///
/// Pages are counted from the start of the `.fdt`; a compound file places
/// such an entry on a page boundary ([`crate::compound_format::write`]).
/// Readers open the file with this size as its alignment hint
/// ([`page_alignment`], `Directory::open_aligned`).
pub const ZSTD_PAGED_PAGE_SIZE: usize = 16 << 10;
const _: () = assert!(lucene_store::alignment::is_valid_hint(ZSTD_PAGED_PAGE_SIZE));
/// The `.si` attribute of a segment with [`Mode::ZstdPaged`] stored fields
/// (value [`ZSTD_PAGED_MODE_VALUE`]); the plugin's Java format reads it to
/// pick its reader.
pub const ZSTD_PAGED_MODE_ATTRIBUTE: &str = "PagedZstdStoredFieldsFormat.mode";
/// [`ZSTD_PAGED_MODE_ATTRIBUTE`]'s value.
pub const ZSTD_PAGED_MODE_VALUE: &str = "ZSTD_PAGED";

/// The alignment hint a segment's `.fdt` (or the `.cfs` holding it) is
/// opened with, from the segment's `.si` attributes: [`ZSTD_PAGED_PAGE_SIZE`]
/// for [`Mode::ZstdPaged`], `None` for every other mode.
pub fn page_alignment(attributes: &[(String, String)]) -> Option<usize> {
    attributes
        .iter()
        .any(|(k, v)| k == ZSTD_PAGED_MODE_ATTRIBUTE && v == ZSTD_PAGED_MODE_VALUE)
        .then_some(ZSTD_PAGED_PAGE_SIZE)
}
/// [`Mode::ZstdPaged`]'s `chunkSize`: plain bytes per chunk, so one frame.
const ZSTD_PAGED_CHUNK_SIZE: usize = 16 << 10;
/// [`Mode::ZstdPaged`]'s `maxDocsPerChunk`.
const ZSTD_PAGED_MAX_DOCS_PER_CHUNK: usize = 1024;
/// The least free room at the end of a page worth filling with a chunk of
/// fewer documents; less than this is padded.
const ZSTD_PAGED_MIN_FILL: usize = 1 << 10;
/// How many smaller chunks a writer tries before padding the page.
const ZSTD_PAGED_FILL_ATTEMPTS: usize = 3;
/// The largest segment dictionary a reader accepts.
pub const ZSTD_PAGED_MAX_DICTIONARY: usize = 1 << 20;
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
    #[error("cannot bulk-copy chunks from this reader: sameCompressionMode={same_mode}, readerChunkSize={reader_chunk_size}, writerChunkSize={writer_chunk_size}, tooDirty={too_dirty}")]
    BulkCopyNotPermitted {
        reader_chunk_size: i32,
        writer_chunk_size: i32,
        same_mode: bool,
        too_dirty: bool,
    },
    #[error("inverted document range: fromDoc={from_doc} > toDoc={to_doc}")]
    InvertedDocRange { from_doc: i32, to_doc: i32 },
    /// A derived-source `_source` this reader does not rebuild (`lucene_search::derived_source`).
    #[error("derived source: {0}")]
    DerivedSource(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    String(String),
    /// A string value shared with its caller rather than copied: written
    /// exactly as [`Self::String`] of the same text, and never produced by a
    /// reader. Rust-only; it keeps a large stored text from being copied
    /// once more on the way to the writer. It is not `==` to a
    /// [`Self::String`] of the same text (compare [`Self::as_text`]).
    Shared(lucene_util::shared_text::SharedText),
    Binary(Vec<u8>),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
}

impl FieldValue {
    /// The text of a [`Self::String`] or [`Self::Shared`] value.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            FieldValue::String(s) => Some(s),
            FieldValue::Shared(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StoredField {
    pub field_number: i32,
    pub value: FieldValue,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Document {
    pub fields: Vec<StoredField>,
}

/// Which stored-fields format a writer writes: `Lucene90StoredFieldsFormat`'s
/// two modes, or a `Lucene90CompressingStoredFieldsFormat` with its own name,
/// suffix, compression and geometry (the test framework's
/// `CompressingCodec`). The compression is baked into the data codec name,
/// so `open` detects it from the header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    BestSpeed,
    BestCompression,
    Compressing(&'static CompressingFormat),
    /// OpenSearch custom-codecs' `Lucene*CustomStoredFieldsFormat`: a
    /// `Lucene90CompressingStoredFieldsFormat` named [`DATA_CODEC_ZSTD`] or
    /// [`DATA_CODEC_ZSTD_NO_DICT`], with `BEST_COMPRESSION`'s geometry.
    Zstd(ZstdMode),
    /// The Rust engine's paged zstd stored fields at a zstd level:
    /// [`DATA_CODEC_ZSTD_PAGED`], 16 KiB chunks of one zstd frame each,
    /// compressed against an optional segment dictionary that the `.fdt`
    /// holds right after its header (`vint length, bytes`), and packed so no
    /// chunk crosses a [`ZSTD_PAGED_PAGE_SIZE`] boundary unless it starts on
    /// one. Every other byte of the three files is `Lucene90`'s.
    ZstdPaged(i32),
}

/// The settings of a [`Mode::Zstd`] format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZstdMode {
    /// `zstd` (each chunk's first bytes as the sub-blocks' dictionary) rather
    /// than `zstd_no_dict`.
    pub dictionary: bool,
    /// The zstd level (`index.codec.compression_level`). Readers ignore it.
    pub level: i32,
}

impl ZstdMode {
    /// `zstd` at `level`.
    pub const fn zstd(level: i32) -> Self {
        Self {
            dictionary: true,
            level,
        }
    }

    /// `zstd_no_dict` at `level`.
    pub const fn zstd_no_dict(level: i32) -> Self {
        Self {
            dictionary: false,
            level,
        }
    }

    /// The `Lucene104CustomCodec.Mode` name the segment records
    /// (`ZSTD`, `ZSTD_NO_DICT`).
    pub const fn mode_name(self) -> &'static str {
        if self.dictionary {
            "ZSTD"
        } else {
            "ZSTD_NO_DICT"
        }
    }
}

impl Mode {
    /// The `.fdt` codec name.
    pub fn data_codec(self) -> &'static str {
        match self {
            Self::BestSpeed => DATA_CODEC_BEST_SPEED,
            Self::BestCompression => DATA_CODEC_BEST_COMPRESSION,
            Self::Compressing(format) => format.format_name,
            Self::Zstd(z) if z.dictionary => DATA_CODEC_ZSTD,
            Self::Zstd(_) => DATA_CODEC_ZSTD_NO_DICT,
            Self::ZstdPaged(_) => DATA_CODEC_ZSTD_PAGED,
        }
    }
    /// The files' segment suffix (empty for both `Lucene90` modes).
    pub fn segment_suffix(self) -> &'static str {
        match self {
            Self::BestSpeed | Self::BestCompression | Self::Zstd(_) | Self::ZstdPaged(_) => "",
            Self::Compressing(format) => format.segment_suffix,
        }
    }
    pub fn compression(self) -> Compression {
        match self {
            Self::BestSpeed => Compression::Lz4WithPreset,
            Self::BestCompression => Compression::DeflateWithPreset,
            Self::Compressing(format) => format.compression,
            Self::Zstd(z) if z.dictionary => Compression::Zstd,
            Self::Zstd(_) => Compression::ZstdNoDict,
            Self::ZstdPaged(_) => Compression::ZstdPaged,
        }
    }
    /// `chunkSize`: the buffered bytes that close a chunk.
    pub fn chunk_size(self) -> i32 {
        match self {
            Self::BestSpeed => BEST_SPEED_CHUNK_SIZE as i32,
            Self::BestCompression | Self::Zstd(_) => BEST_COMPRESSION_CHUNK_SIZE as i32,
            Self::ZstdPaged(_) => ZSTD_PAGED_CHUNK_SIZE as i32,
            Self::Compressing(format) => format.chunk_size,
        }
    }
    /// `maxDocsPerChunk`: the documents that close a chunk.
    pub fn max_docs_per_chunk(self) -> i32 {
        match self {
            Self::BestSpeed => BEST_SPEED_MAX_DOCS_PER_CHUNK as i32,
            Self::BestCompression | Self::Zstd(_) => BEST_COMPRESSION_MAX_DOCS_PER_CHUNK as i32,
            Self::ZstdPaged(_) => ZSTD_PAGED_MAX_DOCS_PER_CHUNK as i32,
            Self::Compressing(format) => format.max_docs_per_chunk,
        }
    }
    /// `blockShift` of the `.fdx` monotonic arrays.
    pub fn block_shift(self) -> i32 {
        match self {
            Self::BestSpeed | Self::BestCompression | Self::Zstd(_) | Self::ZstdPaged(_) => {
                INDEX_BLOCK_SHIFT as i32
            }
            Self::Compressing(format) => format.block_shift,
        }
    }
}

/// Parsed `.fdm` metadata plus the `.fdx`-relative pointers/arrays it
/// describes; `document()` also needs the whole `.fdt` file's bytes.
pub struct StoredFieldsReader<'d> {
    fdt: &'d [u8],
    fdx: &'d [u8],
    /// The `.fdt` codec name `open` matched, and what it says about the
    /// writer: the compression and the most documents one chunk can hold.
    data_codec: &'static str,
    compression: Compression,
    max_docs_per_chunk: i32,
    chunk_size: i32,
    max_doc: i32,
    num_chunks: i64,
    num_dirty_chunks: i64,
    num_dirty_docs: i64,
    max_pointer: i64,
    docs_start_pointer: i64,
    docs_end_pointer: i64,
    /// Shared, not owned: [`StoredFieldsReader::rebind`] hands the parsed
    /// metadata to a reader over the same files without copying its blocks.
    docs_meta: Arc<direct_monotonic::Meta>,
    start_pointers_start_pointer: i64,
    start_pointers_end_pointer: i64,
    start_pointers_meta: Arc<direct_monotonic::Meta>,
    /// A [`Mode::ZstdPaged`] segment's dictionary, if it has one.
    zstd_dictionary: Option<Arc<lucene_zstd::DecoderDictionary>>,
    /// The `.fdt` input to advise ahead of a large document's reads
    /// ([`Self::with_read_ahead`]).
    read_ahead: Option<&'d lucene_store::directory::Input>,
}

/// How much of a chunk [`StoredFieldsReader::prefetch`] (and a caller
/// advising chunks itself, [`StoredFieldsReader::advised_range`]) asks for up
/// front. Only a `sliced` chunk -- one document of at least twice the chunk
/// size -- is longer: its first bytes hold the fields before a large value,
/// and a reader that wants the value reads ahead of itself
/// ([`StoredFieldsReader::with_read_ahead`]).
pub const CHUNK_HEAD_BYTES: usize = 1 << 20;

/// The first read-ahead window of a large document, doubling per window up
/// to [`READ_AHEAD_MAX`].
const READ_AHEAD_MIN: usize = 1 << 20;
const READ_AHEAD_MAX: usize = 32 << 20;

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
    let (data_codec_name, compression, max_docs_per_chunk) = match data_codec.as_str() {
        DATA_CODEC_BEST_SPEED => (
            DATA_CODEC_BEST_SPEED,
            Compression::Lz4WithPreset,
            BEST_SPEED_MAX_DOCS_PER_CHUNK as i32,
        ),
        DATA_CODEC_BEST_COMPRESSION => (
            DATA_CODEC_BEST_COMPRESSION,
            Compression::DeflateWithPreset,
            BEST_COMPRESSION_MAX_DOCS_PER_CHUNK as i32,
        ),
        DATA_CODEC_ZSTD => (
            DATA_CODEC_ZSTD,
            Compression::Zstd,
            BEST_COMPRESSION_MAX_DOCS_PER_CHUNK as i32,
        ),
        DATA_CODEC_ZSTD_NO_DICT => (
            DATA_CODEC_ZSTD_NO_DICT,
            Compression::ZstdNoDict,
            BEST_COMPRESSION_MAX_DOCS_PER_CHUNK as i32,
        ),
        DATA_CODEC_ZSTD_PAGED => (
            DATA_CODEC_ZSTD_PAGED,
            Compression::ZstdPaged,
            ZSTD_PAGED_MAX_DOCS_PER_CHUNK as i32,
        ),
        other => match compression::COMPRESSING_DATA_CODECS
            .iter()
            .find(|(name, _)| *name == other)
        {
            Some(&(name, compression)) => (
                name,
                compression,
                compression::COMPRESSING_MAX_DOCS_PER_CHUNK,
            ),
            None => {
                return Err(lucene_store::Error::Corrupted(format!(
                    "unknown stored fields data codec: {other}"
                ))
                .into())
            }
        },
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
    let zstd_dictionary = if compression == Compression::ZstdPaged {
        read_zstd_dictionary(&mut fdt_input)?
            .map(|d| Arc::new(lucene_zstd::DecoderDictionary::new(d)))
    } else {
        None
    };

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
        data_codec: data_codec_name,
        compression,
        max_docs_per_chunk,
        chunk_size,
        max_doc,
        num_chunks,
        num_dirty_chunks,
        num_dirty_docs,
        max_pointer,
        docs_start_pointer,
        docs_end_pointer,
        docs_meta: Arc::new(docs_meta),
        start_pointers_start_pointer,
        start_pointers_end_pointer,
        start_pointers_meta: Arc::new(start_pointers_meta),
        zstd_dictionary,
        read_ahead: None,
    })
}

/// A [`Mode::ZstdPaged`] `.fdt`'s dictionary, read from `input` positioned
/// right after the index header: `vint length, bytes`, `None` for length 0.
fn read_zstd_dictionary<'d>(input: &mut SliceInput<'d>) -> Result<Option<&'d [u8]>> {
    let len = input.read_vint()?;
    let len = usize::try_from(len)
        .ok()
        .filter(|&l| l <= ZSTD_PAGED_MAX_DICTIONARY)
        .ok_or_else(|| {
            lucene_store::Error::Corrupted(format!("invalid zstd dictionary length {len}"))
        })?;
    if len == 0 {
        return Ok(None);
    }
    let at = input.position();
    let dict = input.slice(at, at.saturating_add(len))?;
    input.skip(len)?;
    Ok(Some(dict))
}

impl<'d> StoredFieldsReader<'d> {
    /// This reader's parsed metadata over `fdt`/`fdx` instead of the bytes it
    /// was opened on -- what Java's reader keeps for the life of a segment
    /// core, so a caller holding the files can parse `.fdm` once rather than
    /// per document. `open` validated the metadata against the files it was
    /// given, so `fdt` and `fdx` must be those same files' bytes (reading
    /// through other bytes fails or answers garbage, never undefined
    /// behaviour: every read is bounds-checked).
    pub fn rebind<'e>(&self, fdt: &'e [u8], fdx: &'e [u8]) -> StoredFieldsReader<'e> {
        StoredFieldsReader {
            fdt,
            fdx,
            data_codec: self.data_codec,
            compression: self.compression,
            max_docs_per_chunk: self.max_docs_per_chunk,
            chunk_size: self.chunk_size,
            max_doc: self.max_doc,
            num_chunks: self.num_chunks,
            num_dirty_chunks: self.num_dirty_chunks,
            num_dirty_docs: self.num_dirty_docs,
            max_pointer: self.max_pointer,
            docs_start_pointer: self.docs_start_pointer,
            docs_end_pointer: self.docs_end_pointer,
            docs_meta: Arc::clone(&self.docs_meta),
            start_pointers_start_pointer: self.start_pointers_start_pointer,
            start_pointers_end_pointer: self.start_pointers_end_pointer,
            start_pointers_meta: Arc::clone(&self.start_pointers_meta),
            zstd_dictionary: self.zstd_dictionary.clone(),
            read_ahead: None,
        }
    }

    /// This reader, advising `fdt` (the input its `.fdt` bytes are) ahead of
    /// the reads of a large document's wanted bytes: windows of 1 MiB
    /// doubling to 32 MiB, each asked for before it is decompressed. Meant
    /// for an `.fdt` that is advised random or filled on demand, where a
    /// chunk is advised or fetched only up to [`CHUNK_HEAD_BYTES`] up front.
    #[must_use]
    pub fn with_read_ahead(mut self, fdt: &'d lucene_store::directory::Input) -> Self {
        self.read_ahead = Some(fdt);
        self
    }

    pub fn max_doc(&self) -> i32 {
        self.max_doc
    }

    /// The `maxDocsPerChunk` the writer of this `.fdt` must have used --
    /// determined by the data codec name `open` matched, since
    /// `Lucene90StoredFieldsFormat.impl` is the only thing that writes either
    /// name and it pins the value per mode. Used as a hard upper bound on a
    /// chunk header's document count; see [`Self::read_chunk_header`].
    fn max_docs_per_chunk(&self) -> i32 {
        self.max_docs_per_chunk
    }

    /// The most this segment's compressor can inflate a byte by, per
    /// [`Compression::max_expansion`].
    fn max_expansion(&self) -> usize {
        self.compression.max_expansion()
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

    /// `StoredFields.prefetch(docID)` for several documents at once: the
    /// `.fdt` chunk holding each is announced in one
    /// [`lucene_store::prefetch`], so a fetch phase over a file read by
    /// range waits for one round trip rather than one per chunk. A no-op
    /// over memory that is not read by range.
    ///
    /// # Errors
    /// A chunk that could not be fetched, or a corrupt `.fdx` index.
    pub fn prefetch(&self, docs: &[i32]) -> Result<()> {
        if !lucene_store::is_prefetchable(self.fdt) {
            return Ok(());
        }
        let ranges: Vec<&[u8]> = self
            .chunk_ranges(docs)?
            .into_iter()
            .filter_map(|r| self.fdt.get(Self::advised_range(r)))
            .collect();
        Ok(lucene_store::prefetch(&ranges)?)
    }

    /// The part of a chunk's `.fdt` range to advise or fetch before reading
    /// it: all of it, or its first [`CHUNK_HEAD_BYTES`] for a larger (sliced)
    /// chunk, whose reader reads ahead of itself as far as it reads at all.
    pub fn advised_range(range: std::ops::Range<usize>) -> std::ops::Range<usize> {
        range.start..range.end.min(range.start.saturating_add(CHUNK_HEAD_BYTES))
    }

    /// The `.fdt` byte range of the chunk holding each of `docs` (documents
    /// outside the segment skipped), in the order given; a document in the
    /// same chunk as the one before it adds no second range.
    ///
    /// # Errors
    /// A corrupt `.fdx` index.
    pub fn chunk_ranges(&self, docs: &[i32]) -> Result<Vec<std::ops::Range<usize>>> {
        let mut ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(docs.len());
        for &doc in docs {
            if let Some(range) = self.chunk_range(doc)? {
                if ranges.last() != Some(&range) {
                    ranges.push(range);
                }
            }
        }
        Ok(ranges)
    }

    /// The `.fdt` byte range of the chunk holding `doc`; `None` for a
    /// document outside the segment.
    ///
    /// # Errors
    /// A corrupt `.fdx` index.
    pub fn chunk_range(&self, doc: i32) -> Result<Option<std::ops::Range<usize>>> {
        if doc < 0 || doc >= self.max_doc {
            return Ok(None);
        }
        let block = direct_monotonic::floor_index(
            self.docs_region()?,
            &self.docs_meta,
            0,
            self.num_chunks,
            i64::from(doc),
        )?;
        let start = self.block_start_pointer(block)?;
        let end = match block.checked_add(1) {
            Some(next) if next < self.num_chunks => self.block_start_pointer(next)?,
            _ => self.max_pointer,
        };
        let start = usize::try_from(start)
            .unwrap_or(usize::MAX)
            .min(self.fdt.len());
        let end = usize::try_from(end)
            .unwrap_or(0)
            .max(start.saturating_add(1))
            .min(self.fdt.len());
        Ok((start < end).then_some(start..end))
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
    /// A document in a `sliced` chunk (one at least twice the chunk size) is
    /// decompressed as it is read, a compression unit at a time, as Java's
    /// sliced `DataInput` does: [`VisitStatus::Stop`], or `No` on the last
    /// field, leaves the rest undecompressed, and a skipped value passes
    /// over every unit it covers whole by its recorded compressed length
    /// rather than inflating it. A smaller document is decompressed over its
    /// own byte range first (see [`Self::serialized_document`]).
    pub fn visit_document(&self, doc_id: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        let (header, input) = self.read_chunk_header(doc_id)?;
        let index = header.index_of(doc_id);
        // ARITH: as in `serialized_document`: `offsets` has `chunk_docs + 1`
        // non-decreasing entries and `index < chunk_docs`.
        #[allow(clippy::arithmetic_side_effects)]
        let (start, end) = (
            header.offsets[index] as usize,
            header.offsets[index + 1] as usize,
        );
        let num_stored_fields = header.num_stored_fields[index];
        if !header.sliced || start == end {
            let mut input = input;
            let bytes = if start == end {
                Vec::new()
            } else {
                self.decompress_range(&header, &mut input, start, end)?
            };
            return visit_document(num_stored_fields, &bytes, visitor);
        }
        let chunk_start = input.position();
        let mut doc = SlicedDocument {
            reader: self,
            input,
            total_length: header.total_length() as usize,
            next_unit: 0,
            end,
            buf: Vec::new(),
            buf_at: start,
            pos: 0,
            // The chunk's head was advised with the chunk (`advised_range`).
            advised_to: chunk_start.saturating_add(CHUNK_HEAD_BYTES),
            window: READ_AHEAD_MIN,
        };
        // ARITH: `start <= end`.
        #[allow(clippy::arithmetic_side_effects)]
        let len = end - start;
        visit_fields(&mut doc, len, num_stored_fields, visitor)
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
        self.chunk_header_at(block_start, doc_id)
    }

    /// [`Self::read_chunk_header`] for the chunk at `.fdt` offset
    /// `block_start`.
    fn chunk_header_at(
        &self,
        block_start: i64,
        doc_id: i32,
    ) -> Result<(ChunkHeader, SliceInput<'d>)> {
        let mut input = SliceInput::new(self.fdt);
        input.seek(
            usize::try_from(block_start).map_err(|_| Error::DocOutOfRange(doc_id, self.max_doc))?,
        )?;
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
        // `max_doc - chunk_docs` only once `0 <= chunk_docs <= 2^16` and
        // `max_doc >= 0` do (so it lands in `-2^16..=i32::MAX`).
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
        // than unwinding (`docs/arithmetic-gate.md`). A `CompressingCodec`
        // name pins no cap, so those take
        // `compression::COMPRESSING_MAX_DOCS_PER_CHUNK`, which their writers
        // refuse to exceed.
        //
        // ARITH: `&&` short-circuits left to right, so `doc_id - doc_base` is
        // only evaluated under `doc_id >= doc_base >= 0` (difference in
        // `0..=i32::MAX`) and `self.max_doc - chunk_docs` only under
        // `0 <= chunk_docs <= 2^16` with `max_doc >= 0` (`open` rejects a
        // negative one), so the difference is in `-2^16..=i32::MAX`.
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
            // ARITH: `chunk_docs` is in `1..=2^16` (checked above), so
            // `chunk_docs + 1` is far inside `i32`, and both `as usize` casts
            // are of a non-negative value.
            #[allow(clippy::arithmetic_side_effects)]
            let mut offsets = Vec::with_capacity(chunk_docs as usize + 1);
            offsets.push(0i64);
            // The lengths are decoded into `offsets[1..]` and summed there in
            // place: `offsets[i + 1] = offsets[i] + lengths[i]`.
            read_bulk_ints_into(&mut input, chunk_docs as usize, &mut offsets)?;
            let mut sum = 0i64;
            let mut mismatch = false;
            // ARITH: `read_bulk_ints` guarantees every value it returns is
            // non-negative and at most `u32::MAX` (the widest shape is
            // 32-bit, read unsigned; the all-equal shape rejects a negative
            // constant), so over at most 2^16 documents the running sum stays
            // under 2^48. Non-negativity is also what keeps `offsets`
            // monotonic, which `serialized_document`'s
            // `offsets[i + 1] as usize - offsets[i]` depends on -- a
            // decreasing pair underflowed that to a ~2^64 slice length.
            #[allow(clippy::arithmetic_side_effects)]
            for (slot, &n) in offsets[1..].iter_mut().zip(&num_stored_fields) {
                let len = *slot;
                // Java's "only the empty document has a serialized length of
                // 0" check, folded into the same pass rather than a second
                // one over the same two arrays.
                mismatch |= (len == 0) != (n == 0);
                sum += len;
                *slot = sum;
            }
            if mismatch {
                return Err(first_length_count_mismatch(&offsets, &num_stored_fields));
            }
            (num_stored_fields, offsets)
        };
        // The chunk's decompressed size is claimed by its own header and
        // related to nothing on the wire, yet it sizes `decompress_range`'s
        // output `Vec` and `decompress_unit`'s `vec![0u8; dictLength +
        // blockLength]`. Bound it by what the bytes that are left could
        // possibly inflate to -- checked **once per chunk**, never per
        // document or per sub-block. See [`Compression::max_expansion`]
        // for where the ratios come from.
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
    /// discarded (see [`compression::decompress`]).
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
        // `total_length`) is non-negative, under 2^48, and no larger than the
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
                self.decompress_unit(
                    input,
                    unit_len,
                    lo - unit_start,
                    hi.saturating_sub(lo),
                    &mut doc_bytes,
                )?;
                unit_start += unit_len;
            }
        } else {
            self.decompress_unit(input, total_length, want_start, wanted, &mut doc_bytes)?;
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

    /// [`compression::decompress`] with this segment's dictionary, if any.
    fn decompress_unit(
        &self,
        input: &mut SliceInput<'d>,
        original_length: usize,
        offset: usize,
        length: usize,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        if self.compression == Compression::ZstdPaged {
            compression::decompress_paged(
                input,
                self.zstd_dictionary.as_deref(),
                original_length,
                offset,
                length,
                out,
            )?;
        } else {
            compression::decompress(
                self.compression,
                input,
                original_length,
                offset,
                length,
                out,
            )?;
        }
        Ok(())
    }

    /// Whether this is a [`Mode::ZstdPaged`] segment with a dictionary.
    pub fn has_zstd_dictionary(&self) -> bool {
        self.zstd_dictionary.is_some()
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

    /// [`Self::read_chunk`] for a chunk that is not sliced (no document of at least twice the chunk
    /// size) and holds at most `max_bytes` plain bytes; `None` for any other, which a caller reads
    /// document by document.
    ///
    /// # Errors
    /// Those of [`Self::read_chunk`].
    pub fn read_small_chunk(
        &self,
        doc_id: i32,
        max_bytes: usize,
    ) -> Result<Option<DecompressedChunk>> {
        let (header, mut input) = self.read_chunk_header(doc_id)?;
        let total = header.total_length() as usize;
        if header.sliced || total > max_bytes {
            return Ok(None);
        }
        let bytes = if total == 0 {
            Vec::new()
        } else {
            self.decompress_range(&header, &mut input, 0, total)?
        };
        Ok(Some(DecompressedChunk { header, bytes }))
    }

    /// The `.fdt` codec name, which names the format that wrote it -- a
    /// bulk chunk copy is only legal into a writer of the same one
    /// (`Lucene90CompressingStoredFieldsWriter.getMergeStrategy`'s
    /// `reader.getCompressionMode() == compressionMode`, a mode being one
    /// format's in practice).
    pub fn data_codec(&self) -> &'static str {
        self.data_codec
    }

    /// The compression the `.fdt` was written with.
    pub fn compression(&self) -> Compression {
        self.compression
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

    /// Called once before the first field with the document's serialized
    /// length and stored-field count, so a visitor that copies the values
    /// out can size its buffer once. Rust-only: Java's visitor has no such
    /// hook (it grows its `byte[]`s instead).
    fn document_size(&mut self, _serialized_len: usize, _num_stored_fields: i64) {}

    /// A `STRING`/`BYTE_ARR` value at least this long that the visitor said
    /// [`VisitStatus::Yes`] to is not decoded: [`Self::deferred_field`] is
    /// told its length instead, and the visitor reads it later if it turns
    /// out to be wanted. Rust-only; the default defers nothing.
    fn defer_values_from(&self) -> usize {
        usize::MAX
    }

    /// A value [`Self::defer_values_from`] left undecoded: its field, whether
    /// it is a string, and its length in bytes.
    fn deferred_field(&mut self, _field_number: i32, _string: bool, _length: usize) -> Result<()> {
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
    visit_fields(
        &mut SliceInput::new(bytes),
        bytes.len(),
        num_stored_fields,
        visitor,
    )
}

/// [`visit_document`]'s loop over either a document already in memory or a
/// [`SlicedDocument`] decompressed as it is read.
fn visit_fields<I: DocInput>(
    input: &mut I,
    len: usize,
    num_stored_fields: i64,
    visitor: &mut dyn StoredFieldVisitor,
) -> Result<()> {
    visitor.document_size(len, num_stored_fields);
    let defer_from = visitor.defer_values_from();
    for field_idx in 0..num_stored_fields {
        let info_and_bits = input.with_window(|i| i.read_vlong().map_err(Into::into))?;
        // Java is `(int) (infoAndBits >>> TYPE_BITS)`. The unsigned shift is
        // written out here rather than left as a signed `>>`: the two agree
        // for every input (they differ only in the top three bits, which the
        // `as i32` discards), but this is the one place in the file where
        // that is true, and the reader should not have to re-derive it.
        let field_number = ((info_and_bits as u64) >> TYPE_BITS) as i32;
        let bits = info_and_bits & TYPE_MASK;
        // ARITH: `field_idx` runs `0..num_stored_fields`, so
        // `num_stored_fields >= 1` here and the decrement cannot underflow.
        #[allow(clippy::arithmetic_side_effects)]
        let is_last = field_idx == num_stored_fields - 1;
        match visitor.needs_field(field_number)? {
            VisitStatus::Yes => visit_field(input, field_number, bits, defer_from, visitor)?,
            // Java's "don't skipField on the last field value; treat like STOP".
            VisitStatus::No if is_last => return Ok(()),
            VisitStatus::No => skip_field(input, bits)?,
            VisitStatus::Stop => return Ok(()),
        }
    }
    Ok(())
}

/// The bytes of one serialized document, as [`visit_fields`] reads them.
trait DocInput {
    /// Runs `f` over the next few bytes (at least [`FIELD_WINDOW`], or all
    /// that are left), advancing past what it read: a field header, a
    /// length or a numeric value.
    fn with_window<T>(&mut self, f: impl FnOnce(&mut SliceInput<'_>) -> Result<T>) -> Result<T>;
    /// The next `len` bytes, consumed.
    fn value(&mut self, len: usize) -> Result<&[u8]>;
    /// Passes over the next `len` bytes.
    fn skip_bytes(&mut self, len: usize) -> Result<()>;
    /// How many bytes are left.
    fn left(&self) -> usize;
}

/// The most bytes a field header, a length or a numeric value takes: a
/// 10-byte vlong, or a `TLong`'s header byte and its 10-byte vlong.
const FIELD_WINDOW: usize = 16;

impl DocInput for SliceInput<'_> {
    fn with_window<T>(&mut self, f: impl FnOnce(&mut SliceInput<'_>) -> Result<T>) -> Result<T> {
        f(self)
    }

    fn value(&mut self, len: usize) -> Result<&[u8]> {
        let start = self.position();
        DataInput::skip(self, len)?;
        Ok(self.slice(start, self.position())?)
    }

    fn skip_bytes(&mut self, len: usize) -> Result<()> {
        Ok(DataInput::skip(self, len)?)
    }

    fn left(&self) -> usize {
        self.len().saturating_sub(self.position())
    }
}

/// One document of a `sliced` chunk, decompressed a compression unit at a
/// time as it is read -- Java's sliced `SerializedDocument` input, with one
/// difference: a skip passes over the units it covers whole by their
/// compressed length instead of inflating them.
///
/// Offsets are into the chunk's decompressed payload. `buf` holds the bytes
/// `buf_at..buf_at + buf.len()`, the read position is `buf_at + pos`, and
/// `input` sits on the first compression unit not yet read, which starts
/// at `next_unit`.
struct SlicedDocument<'r, 'd> {
    reader: &'r StoredFieldsReader<'d>,
    input: SliceInput<'d>,
    total_length: usize,
    next_unit: usize,
    /// Where this document's bytes end.
    end: usize,
    buf: Vec<u8>,
    buf_at: usize,
    pos: usize,
    /// How far into the `.fdt` the reads have been advised
    /// ([`StoredFieldsReader::with_read_ahead`]), and the next window's size.
    advised_to: usize,
    window: usize,
}

impl SlicedDocument<'_, '_> {
    /// Advises the next window of the `.fdt` once the reads reach the end
    /// of the last one, when the reader reads ahead.
    // ARITH: `saturating_add` and `min` keep every sum in `usize`.
    #[allow(clippy::arithmetic_side_effects)]
    fn read_ahead(&mut self) {
        let Some(fdt) = self.reader.read_ahead else {
            return;
        };
        let at = self.input.position();
        if at < self.advised_to {
            return;
        }
        let from = at.max(self.advised_to);
        // Advice only: a failure costs the reads their speed, not their
        // bytes, and the read itself reports any fetch that fails.
        let _ = fdt.will_need(from, self.window);
        self.advised_to = from.saturating_add(self.window);
        self.window = (self.window * 2).min(READ_AHEAD_MAX);
    }

    // ARITH: `buf_at + buf.len() <= end` and `pos <= buf.len()` hold after
    // every method (each appends at most up to `end` and moves `pos` within
    // the buffer), so these sums and differences stay in `0..=end`.
    #[allow(clippy::arithmetic_side_effects)]
    fn at(&self) -> usize {
        self.buf_at + self.pos
    }

    // ARITH: `pos <= buf.len()` (see `at`), so the difference is not negative.
    #[allow(clippy::arithmetic_side_effects)]
    fn buffered(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Makes at least `n` bytes past the read position available (the
    /// caller checks that the document has them), dropping the bytes
    /// already read and appending the following units' share of the
    /// document.
    // ARITH: the loop guard keeps `buf_end < end <= total_length`; the unit
    // that contains `buf_end` starts at or before it (`next_unit` only moves
    // past whole units that end at or before `buf_end`), so `unit_end >
    // buf_end` and `hi - lo` is positive; `unit_len <= total_length -
    // next_unit` as in `decompress_range`.
    #[allow(clippy::arithmetic_side_effects)]
    fn fill(&mut self, n: usize) -> Result<()> {
        if self.buffered() >= n {
            return Ok(());
        }
        self.buf.drain(..self.pos);
        self.buf_at += self.pos;
        self.pos = 0;
        self.buf.reserve(n.saturating_sub(self.buf.len()));
        let chunk_size = self.reader.chunk_size as usize; // validated positive in `open`
        while self.buf.len() < n {
            let buf_end = self.buf_at + self.buf.len();
            if buf_end >= self.end || self.next_unit >= self.total_length {
                return Err(lucene_store::Error::Eof { offset: buf_end }.into());
            }
            let unit_len = chunk_size.min(self.total_length - self.next_unit);
            let unit_end = self.next_unit + unit_len;
            let (lo, hi) = if unit_end <= buf_end {
                (0, 0)
            } else {
                (buf_end.max(self.next_unit), unit_end.min(self.end))
            };
            if hi > lo {
                self.read_ahead();
            }
            let before = self.buf.len();
            self.reader.decompress_unit(
                &mut self.input,
                unit_len,
                lo.saturating_sub(self.next_unit),
                hi - lo,
                &mut self.buf,
            )?;
            if self.buf.len() - before != hi - lo {
                return Err(lucene_store::Error::Corrupted(format!(
                    "expected {} decompressed bytes, got {}",
                    hi - lo,
                    self.buf.len() - before
                ))
                .into());
            }
            self.next_unit = unit_end;
        }
        Ok(())
    }
}

impl DocInput for SlicedDocument<'_, '_> {
    fn with_window<T>(&mut self, f: impl FnOnce(&mut SliceInput<'_>) -> Result<T>) -> Result<T> {
        let n = FIELD_WINDOW.min(self.left());
        self.fill(n)?;
        let mut window = SliceInput::new(&self.buf[self.pos..]);
        let out = f(&mut window)?;
        // ARITH: `window` reads within `buf[pos..]`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.pos += window.position();
        }
        Ok(out)
    }

    fn value(&mut self, len: usize) -> Result<&[u8]> {
        if len > self.left() {
            return Err(lucene_store::Error::Eof { offset: self.at() }.into());
        }
        self.fill(len)?;
        let start = self.pos;
        // ARITH: `fill` made `len` bytes available past `pos`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.pos += len;
        }
        Ok(&self.buf[start..self.pos])
    }

    fn skip_bytes(&mut self, len: usize) -> Result<()> {
        if len > self.left() {
            return Err(lucene_store::Error::Eof { offset: self.at() }.into());
        }
        // ARITH: `len <= remaining = end - at`, so `at + len <= end`; past
        // the buffer, it is restarted empty at the new position, and the next
        // `fill` passes over the units before it.
        #[allow(clippy::arithmetic_side_effects)]
        if len <= self.buffered() {
            self.pos += len;
        } else {
            self.buf_at = self.at() + len;
            self.buf.clear();
            self.pos = 0;
        }
        Ok(())
    }

    // ARITH: `at() <= end`, see `at`.
    #[allow(clippy::arithmetic_side_effects)]
    fn left(&self) -> usize {
        self.end - self.at()
    }
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
    // `0 <= doc_base <= max_doc - chunk_docs` and `0 < chunk_docs <= 2^16`.
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

/// One chunk as a [`StoredFieldsWriter`] wrote it, compressed, kept apart
/// from the `.fdt` ([`StoredFieldsWriter::capture_chunks`]) so its
/// documents can be read while the file is still being written.
#[derive(Debug, Clone)]
pub struct WrittenChunk {
    mode: Mode,
    doc_base: i32,
    docs: i32,
    bytes: Vec<u8>,
}

impl WrittenChunk {
    /// Whether the chunk holds the writer's document `doc_id`.
    pub fn contains(&self, doc_id: i32) -> bool {
        doc_id >= self.doc_base && doc_id.saturating_sub(self.doc_base) < self.docs
    }

    /// The compressed bytes it holds.
    pub fn ram_bytes(&self) -> usize {
        self.bytes.capacity()
    }

    /// The writer's document `doc_id`, decompressed from the chunk.
    ///
    /// # Errors
    /// The chunk does not hold `doc_id`, or does not decode.
    pub fn document(&self, doc_id: i32) -> Result<Document> {
        if !self.contains(doc_id) {
            return Err(Error::DocOutOfRange(
                doc_id,
                self.doc_base.saturating_add(self.docs),
            ));
        }
        let reader = StoredFieldsReader {
            fdt: &self.bytes,
            fdx: &[],
            data_codec: self.mode.data_codec(),
            compression: self.mode.compression(),
            max_docs_per_chunk: self.mode.max_docs_per_chunk(),
            chunk_size: self.mode.chunk_size(),
            max_doc: self.doc_base.saturating_add(self.docs),
            num_chunks: 1,
            num_dirty_chunks: 0,
            num_dirty_docs: 0,
            max_pointer: self.bytes.len() as i64,
            docs_start_pointer: 0,
            docs_end_pointer: 0,
            docs_meta: Arc::new(direct_monotonic::Meta::empty()),
            start_pointers_start_pointer: 0,
            start_pointers_end_pointer: 0,
            start_pointers_meta: Arc::new(direct_monotonic::Meta::empty()),
            zstd_dictionary: None,
            read_ahead: None,
        };
        let (header, mut input) = reader.chunk_header_at(0, doc_id)?;
        let index = header.index_of(doc_id);
        // ARITH: `chunk_header_at` built `offsets` with `chunk_docs + 1`
        // non-decreasing entries and checked `doc_id` is in the chunk.
        #[allow(clippy::arithmetic_side_effects)]
        let (start, end) = (
            header.offsets[index] as usize,
            header.offsets[index + 1] as usize,
        );
        let bytes = if start == end {
            Vec::new()
        } else {
            reader.decompress_range(&header, &mut input, start, end)?
        };
        parse_document(header.num_stored_fields[index], &bytes)
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

/// How many `.fdt` bytes an indexing buffer holds in memory before it
/// gives its stored fields a file ([`StoredFieldsWriter::stream_to`]).
pub const SPILL_BYTES: usize = 1 << 20;

/// How many `.fdt` bytes a streaming writer gathers before handing them to
/// its output: a few chunks, so the writes stay large while the buffer
/// stays small.
const STREAM_BYTES: usize = 64 << 10;

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
/// - **VISITOR** ([`Self::add_document`]) -- the fallback: parse every field,
///   then recompress.
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
    /// `.fdt` bytes already handed to an output by [`Self::spill_fdt`]; `fdt`
    /// holds what follows them.
    fdt_spilled: u64,
    /// Where [`Self::stream_to`] sends the `.fdt`, if anywhere.
    fdt_out: Option<FsIndexOutput>,
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
    scratch: Compressor,
    /// [`Mode::ZstdPaged`]'s chunk under construction, encoded here first so
    /// the writer can see whether it fits the current page.
    page_chunk: Vec<u8>,
    /// The slice [`Self::add_document`] fills while it streams an outsized
    /// document, kept for its capacity.
    slice_buf: Vec<u8>,
    /// The most `.fdt` bytes held at once while streaming a document.
    #[cfg(test)]
    fdt_high_water: usize,
    /// The chunks written since [`Self::capture_chunks`], while it is on.
    captured: Option<Vec<WrittenChunk>>,
    /// [`Self::set_zstd_dictionary`] gave the chunks a dictionary, which a
    /// [`WrittenChunk`] does not carry.
    has_dictionary: bool,
}

/// The least `.fdt` bytes, summed over a merge's sources, for which the
/// merged [`Mode::ZstdPaged`] segment gets a trained dictionary. Training
/// on [`ZSTD_PAGED_SAMPLE_BYTES`] takes about 0.1-0.15 s whatever the merge
/// size, and the dictionary saves 4-11% of the `.fdt`: above this size the
/// training is a small part of the merge and the saving is over a MiB.
pub const ZSTD_PAGED_DICTIONARY_MIN_BYTES: u64 = 16 << 20;
/// The size of a trained [`Mode::ZstdPaged`] dictionary.
pub const ZSTD_PAGED_DICTIONARY_SIZE: usize = 64 << 10;
/// The plain bytes [`train_zstd_paged_dictionary`] samples.
const ZSTD_PAGED_SAMPLE_BYTES: usize = 2 << 20;

/// Trains a [`Mode::ZstdPaged`] dictionary on documents sampled from
/// `readers`' chunks, evenly spread over each reader and in proportion to
/// its size. `None` when the readers hold too little to train on.
///
/// # Errors
/// A sampled chunk cannot be read.
// ARITH: sample counts and indices are bounded by `max_doc` and by
// `ZSTD_PAGED_SAMPLE_BYTES`; `total` sums file lengths.
#[allow(clippy::arithmetic_side_effects)]
pub fn train_zstd_paged_dictionary(readers: &[&StoredFieldsReader<'_>]) -> Result<Option<Vec<u8>>> {
    let total: u64 = readers.iter().map(|r| r.max_pointer().max(0) as u64).sum();
    if total == 0 {
        return Ok(None);
    }
    // Every sample back to back in one buffer, with each one's end.
    let mut samples: Vec<u8> = Vec::new();
    let mut ends: Vec<usize> = Vec::new();
    for reader in readers {
        let max_doc = reader.max_doc();
        if max_doc == 0 {
            continue;
        }
        let share =
            (ZSTD_PAGED_SAMPLE_BYTES as u64 * reader.max_pointer().max(0) as u64 / total) as usize;
        let chunks = share
            .div_ceil(reader.chunk_size().max(1) as usize)
            .clamp(1, reader.num_chunks().max(1) as usize);
        let mut last_base = -1;
        for j in 0..chunks {
            let doc = (j as u64 * max_doc as u64 / chunks as u64) as i32;
            let chunk = reader.read_chunk(doc)?;
            if chunk.doc_base() == last_base {
                continue;
            }
            last_base = chunk.doc_base();
            for d in chunk.doc_base()..chunk.doc_base() + chunk.num_docs() {
                if let Some((_, bytes)) = chunk.document(d) {
                    if !bytes.is_empty() {
                        samples.extend_from_slice(&bytes[..bytes.len().min(ZSTD_PAGED_CHUNK_SIZE)]);
                        ends.push(samples.len());
                    }
                }
            }
        }
    }
    let mut start = 0;
    let refs: Vec<&[u8]> = ends
        .iter()
        .map(|&end| {
            let sample = &samples[start..end];
            start = end;
            sample
        })
        .collect();
    Ok(lucene_zstd::train_dictionary(&refs, ZSTD_PAGED_DICTIONARY_SIZE).ok())
}

/// One chunk's documents, as [`StoredFieldsWriter`] writes them.
struct ChunkToWrite<'a> {
    doc_base: i32,
    num_stored_fields: &'a [i64],
    lengths: &'a [i64],
    payload: &'a [u8],
    /// Closed before either trigger fired: flagged dirty in the token.
    force: bool,
}

impl ChunkToWrite<'_> {
    /// Appends the chunk to `out`: `docBase`, the token, the field counts
    /// and lengths, then the payload as one compressed unit, or as one unit
    /// per `chunk_size` bytes when it is at least twice that (`sliced`).
    fn encode(
        &self,
        scratch: &mut Compressor,
        compression: Compression,
        chunk_size: usize,
        out: &mut Vec<u8>,
    ) {
        let chunk_docs = self.lengths.len();
        // Only a chunk holding a single outsized document can reach this,
        // since every other chunk closes on the first doc that crosses
        // `chunk_size`.
        //
        // ARITH: `chunk_size` is a positive `i32` (`StoredFieldsWriter::new`)
        // never assigned again, so doubling it is nowhere near `usize`.
        #[allow(clippy::arithmetic_side_effects)]
        let sliced = self.payload.len() >= 2 * chunk_size;

        out.write_vint(self.doc_base);
        out.write_vint(
            ((chunk_docs as i32) << 2) | if self.force { 2 } else { 0 } | i32::from(sliced),
        );
        if chunk_docs == 1 {
            out.write_vint(self.num_stored_fields[0] as i32);
            out.write_vint(self.lengths[0] as i32);
        } else {
            write_bulk_ints(out, self.num_stored_fields);
            write_bulk_ints(out, self.lengths);
        }
        if self.payload.is_empty() {
            return;
        }
        if sliced {
            for slice in self.payload.chunks(chunk_size) {
                scratch.compress(compression, slice, out);
            }
        } else {
            scratch.compress(compression, self.payload, out);
        }
    }
}

impl StoredFieldsWriter {
    /// A fresh writer for `mode`'s chunk geometry, with the `.fdt` index
    /// header already written.
    ///
    /// # Panics
    /// A [`Mode::Compressing`] format whose chunk size or document cap is not
    /// positive, whose cap is above
    /// [`compression::COMPRESSING_MAX_DOCS_PER_CHUNK`], or whose block shift
    /// is outside `DirectMonotonicWriter`'s 2..=22 (Java's constructors
    /// reject the first and last).
    pub fn new(mode: Mode, segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Self {
        let (chunk_size, max_docs_per_chunk) = (mode.chunk_size(), mode.max_docs_per_chunk());
        assert!(
            chunk_size > 0,
            "chunkSize must be positive, got {chunk_size}"
        );
        assert!(
            (1..=compression::COMPRESSING_MAX_DOCS_PER_CHUNK).contains(&max_docs_per_chunk),
            "maxDocsPerChunk out of range: {max_docs_per_chunk}"
        );
        assert!(
            (2..=22).contains(&mode.block_shift()),
            "blockShift out of range: {}",
            mode.block_shift()
        );
        let (chunk_size, max_docs_per_chunk) = (chunk_size as usize, max_docs_per_chunk as usize);
        let mut fdt = Vec::new();
        codec_util::write_index_header(
            &mut fdt,
            mode.data_codec(),
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        );
        if matches!(mode, Mode::ZstdPaged(_)) {
            // No dictionary until `set_zstd_dictionary` says otherwise.
            fdt.write_vint(0);
        }
        let start = fdt.len() as i64;
        Self {
            mode,
            chunk_size,
            max_docs_per_chunk,
            segment_id: *segment_id,
            segment_suffix: segment_suffix.to_string(),
            fdt,
            fdt_spilled: 0,
            fdt_out: None,
            docs_values: vec![0],
            start_pointers_values: vec![start],
            num_dirty_chunks: 0,
            num_dirty_docs: 0,
            chunk_buf: Vec::new(),
            lengths: Vec::new(),
            num_stored_fields: Vec::new(),
            doc_base: 0,
            scratch: {
                let mut scratch = Compressor::default();
                let level = match mode {
                    Mode::Zstd(z) => Some(z.level),
                    Mode::ZstdPaged(level) => Some(level),
                    _ => None,
                };
                if let Some(level) = level {
                    // Out-of-range levels clamp to zstd's own range, as
                    // libzstd itself does; `ZstdLucene104Codec::new` rejects
                    // them up front.
                    let levels = lucene_zstd::levels();
                    let level = level.clamp(*levels.start(), *levels.end());
                    scratch
                        .set_zstd_level(level)
                        .expect("a level inside zstd's range");
                }
                scratch
            },
            page_chunk: Vec::new(),
            slice_buf: Vec::new(),
            #[cfg(test)]
            fdt_high_water: 0,
            captured: None,
            has_dictionary: false,
        }
    }

    /// Keeps a copy of each chunk written from now on until
    /// [`Self::take_captured`], so their documents stay readable once the
    /// `.fdt` bytes go to the output. A writer with a zstd dictionary
    /// ([`Self::set_zstd_dictionary`]) captures none.
    pub fn capture_chunks(&mut self) {
        if !self.has_dictionary {
            self.captured = Some(Vec::new());
        }
    }

    /// The chunks written since [`Self::capture_chunks`], which stops
    /// keeping them.
    pub fn take_captured(&mut self) -> Vec<WrittenChunk> {
        self.captured.take().unwrap_or_default()
    }

    /// How many documents it holds: the next document's id.
    pub fn num_docs(&self) -> i32 {
        // ARITH: buffered documents are fewer than a chunk's cap, and
        // `advance_doc_base` keeps `doc_base` inside `i32`.
        self.doc_base
            .saturating_add(self.num_buffered_docs() as i32)
    }

    /// Gives a [`Mode::ZstdPaged`] writer the dictionary every chunk is
    /// compressed against, stored right after the `.fdt` header. The first
    /// chunk then starts on a page boundary. Must come before any document.
    ///
    /// # Panics
    /// The writer is not [`Mode::ZstdPaged`], already holds documents, or
    /// `dict` is larger than [`ZSTD_PAGED_MAX_DICTIONARY`].
    pub fn set_zstd_dictionary(&mut self, dict: &[u8]) {
        assert!(
            matches!(self.mode, Mode::ZstdPaged(_)),
            "a zstd dictionary needs Mode::ZstdPaged"
        );
        assert!(
            self.doc_base == 0
                && self.num_buffered_docs() == 0
                && self.fdt_spilled == 0
                && self.docs_values.len() == 1,
            "set_zstd_dictionary after documents"
        );
        assert!(
            dict.len() <= ZSTD_PAGED_MAX_DICTIONARY,
            "zstd dictionary too large"
        );
        let header = codec_util::index_header_length(self.mode.data_codec(), &self.segment_suffix);
        self.fdt.truncate(header);
        self.fdt.write_vint(dict.len() as i32);
        self.fdt.extend_from_slice(dict);
        self.has_dictionary = true;
        let dict = (!dict.is_empty()).then_some(dict);
        if dict.is_some() {
            let pad = self.fdt.len().next_multiple_of(ZSTD_PAGED_PAGE_SIZE);
            self.fdt.resize(pad, 0);
        }
        self.scratch
            .set_zstd_dictionary(dict)
            .expect("the writer's level is valid");
        self.start_pointers_values[0] = self.fdt.len() as i64;
    }

    /// The heap this writer holds: the compressed `.fdt` so far and the
    /// chunk being buffered -- what `Lucene90CompressingStoredFieldsWriter`
    /// accounts in `ramBytesUsed`, with the `.fdt` in memory instead of on
    /// disk.
    pub fn ram_bytes(&self) -> usize {
        let longs = self
            .docs_values
            .capacity()
            .saturating_add(self.start_pointers_values.capacity())
            .saturating_add(self.lengths.capacity())
            .saturating_add(self.num_stored_fields.capacity())
            .saturating_mul(8);
        self.fdt
            .capacity()
            .saturating_add(self.chunk_buf.capacity())
            .saturating_add(self.page_chunk.capacity())
            .saturating_add(longs)
    }

    /// Gives the segment its id after documents were added: rewrites the id
    /// in the `.fdt` header already written, and the one `.fdx` and `.fdm`
    /// get at [`Self::finish`]. A buffer streams its documents in before
    /// the flush names its segment.
    pub fn set_segment_id(&mut self, segment_id: &[u8; ID_LENGTH]) {
        // Once spilled, the header is on disk and the buffer's start is chunk
        // data, not the id.
        debug_assert_eq!(self.fdt_spilled, 0, "set_segment_id after a spill");
        if self.fdt_spilled > 0 {
            return;
        }
        // `CodecUtil.writeIndexHeader`: the codec header, then the id.
        let start = codec_util::header_length(self.mode.data_codec());
        if let Some(id) = self.fdt.get_mut(start..start.saturating_add(ID_LENGTH)) {
            id.copy_from_slice(segment_id);
        }
        self.segment_id = *segment_id;
    }

    /// Writes `Mode::BestSpeed` chunks with the high-compression LZ4 match
    /// finder: the same format, smaller, slower to write. See
    /// [`Compressor::set_lz4_high_compression`].
    pub fn set_lz4_high_compression(&mut self, on: bool) {
        self.scratch.set_lz4_high_compression(on);
    }

    /// Whether [`Self::set_lz4_high_compression`] is on.
    pub fn lz4_high_compression(&self) -> bool {
        self.scratch.lz4_high_compression()
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
    ///
    /// Never for [`Mode::ZstdPaged`]: each segment's chunks are compressed
    /// against its own dictionary and packed to its own pages.
    pub fn can_bulk_copy(&self, reader: &StoredFieldsReader<'_>) -> bool {
        !matches!(self.mode, Mode::ZstdPaged(_))
            && reader.data_codec() == self.mode.data_codec()
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
        if matches!(self.mode, Mode::ZstdPaged(_)) {
            self.flush_paged(force);
            return;
        }
        debug_assert_ne!(self.trigger_flush(), force);
        let chunk_docs = self.num_buffered_docs();
        // ARITH: both counters are this writer's own, and each `flush` costs
        // at least the four `.fdt` bytes of a chunk header, so they are
        // bounded by `fdt.len() / 4` -- a `Vec` length, hence `isize::MAX`.
        // `chunk_docs` is `lengths.len()`, capped at `max_docs_per_chunk`
        // (at most 2^16, `new`) by `trigger_flush`.
        #[allow(clippy::arithmetic_side_effects)]
        if force {
            self.num_dirty_chunks += 1;
            self.num_dirty_docs += chunk_docs as i64;
        }
        let chunk = ChunkToWrite {
            doc_base: self.doc_base,
            num_stored_fields: &self.num_stored_fields,
            lengths: &self.lengths,
            payload: &self.chunk_buf,
            force,
        };
        chunk.encode(
            &mut self.scratch,
            self.mode.compression(),
            self.chunk_size,
            &mut self.fdt,
        );
        self.advance_doc_base(chunk_docs);
        self.chunk_buf.clear();
        self.lengths.clear();
        self.num_stored_fields.clear();
        self.record_chunk();
    }

    /// Java's `docBase += numBufferedDocs` on an `int`, which wraps
    /// silently. A wrapped `doc_base` would put a negative `docBase` vint in
    /// the `.fdt` and a non-monotonic value in the `.fdx` doc array -- an
    /// unopenable segment written without a word of complaint. It takes more
    /// than `IndexWriter.MAX_DOCS` (2^31 - 128) documents in one segment to
    /// get there, which is a caller bug rather than bad input, so it panics
    /// with a name rather than returning a `Result` these infallible `add_*`
    /// entry points do not have.
    fn advance_doc_base(&mut self, docs: usize) {
        self.doc_base = self
            .doc_base
            .checked_add(docs as i32)
            .expect("stored fields writer overflowed i32 document ids");
    }

    /// [`Self::flush`] for [`Mode::ZstdPaged`]: writes the buffered
    /// documents as chunks that never cross a [`ZSTD_PAGED_PAGE_SIZE`]
    /// boundary unless they start on one. A chunk that does not fit the
    /// room left in the page is first retried with fewer documents (the
    /// rest stay buffered for the next chunk); when that fails too, or the
    /// room is under [`ZSTD_PAGED_MIN_FILL`], the page is padded with zeros
    /// and the chunk starts the next page.
    // ARITH: `pos < PAGE`, so `PAGE - pos` is positive; `docs * room` is at
    // most 2^16 * 2^14; the drained prefix lengths sum the buffered ones.
    #[allow(clippy::arithmetic_side_effects)]
    fn flush_paged(&mut self, force: bool) {
        while self.num_buffered_docs() > 0 && (force || self.trigger_flush()) {
            let all = self.num_buffered_docs();
            let mut docs = all;
            let mut chunk = std::mem::take(&mut self.page_chunk);
            self.encode_buffered(docs, force, &mut chunk);
            let pos = (self.fdt_pointer() as usize) % ZSTD_PAGED_PAGE_SIZE;
            let room = ZSTD_PAGED_PAGE_SIZE - pos;
            if pos != 0 && chunk.len() > room {
                let mut fitted = false;
                // Whether `chunk` no longer holds the whole-buffer encoding.
                let mut overwritten = false;
                if room >= ZSTD_PAGED_MIN_FILL {
                    let mut len = chunk.len();
                    for _ in 0..ZSTD_PAGED_FILL_ATTEMPTS {
                        // Scale the document count to the room, a little
                        // under so the next try does not just miss again.
                        docs = docs * room * 31 / 32 / len;
                        if docs == 0 {
                            break;
                        }
                        overwritten = true;
                        self.encode_buffered(docs, false, &mut chunk);
                        if chunk.len() <= room {
                            fitted = true;
                            break;
                        }
                        len = chunk.len();
                    }
                }
                if !fitted {
                    self.fdt.resize(self.fdt.len() + room, 0);
                    let next = self.fdt_pointer();
                    if let Some(start) = self.start_pointers_values.last_mut() {
                        *start = next;
                    }
                    docs = all;
                    // A chunk's bytes do not depend on where it lands, so
                    // the first encoding is reused unless a try replaced it.
                    if overwritten {
                        self.encode_buffered(docs, force, &mut chunk);
                    }
                }
            }
            self.fdt.extend_from_slice(&chunk);
            self.page_chunk = chunk;
            if docs == all && force {
                self.num_dirty_chunks += 1;
                self.num_dirty_docs += docs as i64;
            }
            self.advance_doc_base(docs);
            let bytes: i64 = self.lengths[..docs].iter().sum();
            self.chunk_buf.drain(..bytes as usize);
            self.lengths.drain(..docs);
            self.num_stored_fields.drain(..docs);
            self.record_chunk();
        }
    }

    /// Encodes the first `docs` buffered documents as one chunk into `out`
    /// (cleared first).
    fn encode_buffered(&mut self, docs: usize, force: bool, out: &mut Vec<u8>) {
        out.clear();
        let bytes: i64 = self.lengths[..docs].iter().sum();
        let chunk = ChunkToWrite {
            doc_base: self.doc_base,
            num_stored_fields: &self.num_stored_fields[..docs],
            lengths: &self.lengths[..docs],
            payload: &self.chunk_buf[..bytes as usize],
            force,
        };
        chunk.encode(
            &mut self.scratch,
            self.mode.compression(),
            self.chunk_size,
            out,
        );
    }

    /// One `.fdx` entry: the cumulative doc count and the offset one past
    /// the chunk just written (which is the *next* chunk's start pointer,
    /// and after the last chunk is `maxPointer`).
    fn record_chunk(&mut self) {
        let chunk_base = self.docs_values.last().copied().unwrap_or(0);
        let chunk_start = self.start_pointers_values.last().copied().unwrap_or(0);
        self.docs_values.push(self.doc_base as i64);
        self.start_pointers_values.push(self.fdt_pointer());
        if let Some(captured) = self.captured.as_mut() {
            // The chunk starts after what was spilled. Spills land here,
            // after a whole chunk, except for a streamed outsized document
            // (`add_streamed`), which spills inside its chunk: it is never
            // taken while chunks are captured (`can_stream`).
            let from = u64::try_from(chunk_start)
                .unwrap_or(0)
                .saturating_sub(self.fdt_spilled);
            let bytes = self
                .fdt
                .get(usize::try_from(from).unwrap_or(usize::MAX)..)
                .unwrap_or_default()
                .to_vec();
            let doc_base = i32::try_from(chunk_base).unwrap_or(i32::MAX);
            captured.push(WrittenChunk {
                mode: self.mode,
                doc_base,
                docs: self.doc_base.saturating_sub(doc_base),
                bytes,
            });
        }
        if self.fdt.len() >= STREAM_BYTES {
            self.spill_fdt();
        }
    }

    /// The `.fdt` bytes held in memory.
    pub fn buffered_fdt_bytes(&self) -> usize {
        self.fdt.len()
    }

    /// The `.fdt` file pointer: bytes spilled plus bytes buffered.
    fn fdt_pointer(&self) -> i64 {
        // ARITH: a file length, far below `i64::MAX`.
        #[allow(clippy::arithmetic_side_effects)]
        let pointer = self.fdt_spilled + self.fdt.len() as u64;
        pointer as i64
    }

    /// Streams the `.fdt` to `out` from here on: chunks go to `out` once
    /// [`STREAM_BYTES`] of them are buffered, so a long write (a merge, a big
    /// indexing buffer) holds about that much instead of the whole file.
    /// Finish with [`Self::finish_streamed`]. [`Self::set_segment_id`] must
    /// come before: the header it rewrites leaves with the first spill.
    pub fn stream_to(&mut self, out: FsIndexOutput) {
        self.fdt_out = Some(out);
    }

    fn spill_fdt(&mut self) {
        if let Some(out) = self.fdt_out.as_mut() {
            out.write_bytes(&self.fdt);
            let first = self.fdt_spilled == 0;
            // ARITH: a file length.
            #[allow(clippy::arithmetic_side_effects)]
            {
                self.fdt_spilled += self.fdt.len() as u64;
            }
            self.fdt.clear();
            // What was buffered before streaming began is not needed again.
            if first && self.fdt.capacity() > 2 * STREAM_BYTES {
                self.fdt = Vec::new();
            }
        }
    }

    /// [`Self::finish`] for a writer set to [`Self::stream_to`]: writes the
    /// rest of the `.fdt` and its footer to the output and hands it back for
    /// the caller to close, with `.fdx` and `.fdm`. A writer that was never
    /// streamed returns `None` for the output and its whole `.fdt` is lost, so
    /// callers only use this after [`Self::stream_to`].
    pub fn finish_streamed(mut self) -> (Option<FsIndexOutput>, Vec<u8>, Vec<u8>) {
        if self.num_buffered_docs() > 0 {
            self.flush(true);
        }
        let max_doc = self.doc_base;
        let max_pointer = self.fdt_pointer();
        self.spill_fdt();
        let Some(mut out) = self.fdt_out.take() else {
            return (None, Vec::new(), Vec::new());
        };
        codec_util::write_footer_to(&mut out);
        let (fdx, fdm) = write_index_and_meta(
            &self.segment_id,
            &self.segment_suffix,
            max_doc,
            &self.docs_values,
            &self.start_pointers_values,
            max_pointer,
            self.chunk_size as i32,
            self.mode.block_shift() as u32,
            self.num_dirty_chunks,
            self.num_dirty_docs,
        );
        (Some(out), fdx, fdm)
    }

    /// VISITOR path: serialize and buffer one parsed document.
    ///
    /// A document with a value of `chunk_size` bytes or more that serializes
    /// to at least twice that is written straight to the `.fdt`, one
    /// `chunk_size` slice at a time, instead of being copied whole into the
    /// chunk buffer and compressed from there: the bytes are the ones the
    /// buffered path writes (the chunk is `sliced` either way), and the
    /// writer holds one slice rather than a copy of the document.
    pub fn add_document(&mut self, doc: &Document) {
        if self.can_stream(doc) {
            if let Some(length) = self.streamed_len(doc) {
                self.add_streamed(doc, length);
                return;
            }
        }
        let doc_start = self.chunk_buf.len();
        serialize_doc_into(doc, &mut self.chunk_buf);
        // ARITH: `serialize_doc_into` only appends, so the buffer is at least
        // `doc_start` bytes long afterwards.
        #[allow(clippy::arithmetic_side_effects)]
        let length = (self.chunk_buf.len() - doc_start) as i64;
        self.finish_document(doc.fields.len() as i64, length);
    }

    /// Whether `doc` may take [`Self::add_streamed`]: some value is at least
    /// `chunk_size` long (a cheap test that leaves ordinary documents on the
    /// buffered path), and nothing needs the chunk's bytes whole.
    fn can_stream(&self, doc: &Document) -> bool {
        !matches!(self.mode, Mode::ZstdPaged(_))
            && self.captured.is_none()
            && doc.fields.iter().any(|f| match &f.value {
                FieldValue::String(s) => s.len() >= self.chunk_size,
                FieldValue::Shared(s) => s.len() >= self.chunk_size,
                FieldValue::Binary(b) => b.len() >= self.chunk_size,
                _ => false,
            })
    }

    /// `doc`'s serialized length when it is at least twice `chunk_size` and
    /// fits the chunk's `int` length, else `None`.
    fn streamed_len(&self, doc: &Document) -> Option<usize> {
        let length = serialized_doc_len(doc);
        // ARITH: `chunk_size` is a positive `i32` (`new`).
        #[allow(clippy::arithmetic_side_effects)]
        let twice = 2 * self.chunk_size;
        (length >= twice && i32::try_from(length).is_ok()).then_some(length)
    }

    /// [`Self::add_document`] for a document of `length` serialized bytes:
    /// closes the chunk it would have closed (the buffered documents, then
    /// this one), writing its header and then its payload slice by slice.
    fn add_streamed(&mut self, doc: &Document, length: usize) {
        self.num_stored_fields.push(doc.fields.len() as i64);
        self.lengths.push(length as i64);
        let chunk_docs = self.lengths.len();
        let compression = self.mode.compression();
        self.fdt.write_vint(self.doc_base);
        // A chunk of twice `chunk_size` or more is `sliced` (bit 0), and it
        // closed on its size, so it is not dirty (bit 1).
        self.fdt.write_vint(((chunk_docs as i32) << 2) | 1);
        if chunk_docs == 1 {
            self.fdt.write_vint(self.num_stored_fields[0] as i32);
            self.fdt.write_vint(self.lengths[0] as i32);
        } else {
            write_bulk_ints(&mut self.fdt, &self.num_stored_fields);
            write_bulk_ints(&mut self.fdt, &self.lengths);
        }
        let mut slice = std::mem::take(&mut self.slice_buf);
        slice.clear();
        let buffered = std::mem::take(&mut self.chunk_buf);
        let mut streamed = 0usize;
        self.put_slices(&mut slice, compression, &buffered, &mut 0);
        self.chunk_buf = buffered;
        let mut field = Vec::new();
        for f in &doc.fields {
            field.clear();
            let bits = match &f.value {
                FieldValue::String(_) | FieldValue::Shared(_) => TYPE_STRING,
                FieldValue::Binary(_) => TYPE_BYTE_ARR,
                FieldValue::Int(_) => TYPE_NUMERIC_INT,
                FieldValue::Float(_) => TYPE_NUMERIC_FLOAT,
                FieldValue::Long(_) => TYPE_NUMERIC_LONG,
                FieldValue::Double(_) => TYPE_NUMERIC_DOUBLE,
            };
            field.write_vlong(((f.field_number as i64) << TYPE_BITS) | bits);
            match &f.value {
                FieldValue::String(text) => {
                    field.write_vint(text.len() as i32);
                    self.put_slices(&mut slice, compression, &field, &mut streamed);
                    self.put_slices(&mut slice, compression, text.as_bytes(), &mut streamed);
                }
                FieldValue::Shared(text) => {
                    field.write_vint(text.len() as i32);
                    self.put_slices(&mut slice, compression, &field, &mut streamed);
                    self.put_slices(&mut slice, compression, text.as_bytes(), &mut streamed);
                }
                FieldValue::Binary(bytes) => {
                    field.write_vint(bytes.len() as i32);
                    self.put_slices(&mut slice, compression, &field, &mut streamed);
                    self.put_slices(&mut slice, compression, bytes, &mut streamed);
                }
                other => {
                    write_field(&mut field, other);
                    self.put_slices(&mut slice, compression, &field, &mut streamed);
                }
            }
        }
        if !slice.is_empty() {
            self.scratch.compress(compression, &slice, &mut self.fdt);
        }
        debug_assert_eq!(
            streamed, length,
            "the streamed bytes are the counted length"
        );
        slice.clear();
        self.slice_buf = slice;
        self.advance_doc_base(chunk_docs);
        self.chunk_buf.clear();
        self.lengths.clear();
        self.num_stored_fields.clear();
        self.record_chunk();
    }

    /// Appends `bytes` to the payload being sliced: each time `slice` holds
    /// `chunk_size` bytes and more follow, it is compressed to the `.fdt` as
    /// one unit (what `payload.chunks(chunk_size)` yields), and the `.fdt`
    /// spills to its output once it holds [`STREAM_BYTES`].
    fn put_slices(
        &mut self,
        slice: &mut Vec<u8>,
        compression: Compression,
        mut bytes: &[u8],
        counted: &mut usize,
    ) {
        // ARITH: an in-memory byte count.
        #[allow(clippy::arithmetic_side_effects)]
        {
            *counted += bytes.len();
        }
        while !bytes.is_empty() {
            if slice.len() == self.chunk_size {
                self.scratch.compress(compression, slice, &mut self.fdt);
                slice.clear();
                #[cfg(test)]
                {
                    self.fdt_high_water = self.fdt_high_water.max(self.fdt.len());
                }
                if self.fdt.len() >= STREAM_BYTES {
                    self.spill_fdt();
                }
            }
            // ARITH: `slice.len() < chunk_size` here, so the room is positive.
            #[allow(clippy::arithmetic_side_effects)]
            let room = self.chunk_size - slice.len();
            let (now, rest) = bytes.split_at(room.min(bytes.len()));
            slice.extend_from_slice(now);
            bytes = rest;
        }
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
                same_mode: reader.data_codec() == self.mode.data_codec(),
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

    /// Already-flushed `.fdt` bytes, excluding a footer. Test-framework codec
    /// adapters copy these to their directory outputs at Java callback boundaries.
    #[cfg(feature = "test-support")]
    pub fn flushed_data(&self) -> &[u8] {
        &self.fdt
    }

    /// Java's initial `finish(numDocs)` flush, before document-count validation
    /// and before opening `.fdx`. Kept separate for fault-injection adapters.
    #[cfg(feature = "test-support")]
    pub fn flush_for_finish(&mut self) -> &[u8] {
        if self.num_buffered_docs() > 0 {
            self.flush(true);
        }
        &self.fdt
    }

    /// Finish bytes plus the `.fdm` boundary written by `FieldsIndexWriter.finish`
    /// before its `.fdx` output closes. Chunk statistics and both outer footers
    /// belong after that close succeeds. Only test-framework adapters need it.
    #[cfg(feature = "test-support")]
    pub fn finish_staged(mut self) -> (Vec<u8>, Vec<u8>, Vec<u8>, usize) {
        self.flush_for_finish();
        let mut statistics = Vec::new();
        let chunks = i64::try_from(self.docs_values.len())
            .expect("chunk count fits a Vec length")
            .checked_sub(1)
            .expect("initial chunk index entry");
        statistics.write_vlong(chunks);
        statistics.write_vlong(self.num_dirty_chunks);
        statistics.write_vlong(self.num_dirty_docs);
        let (fdt, fdx, fdm) = self.finish();
        let index_end = fdm
            .len()
            .checked_sub(codec_util::FOOTER_LENGTH)
            .and_then(|length| length.checked_sub(statistics.len()))
            .expect("metadata includes statistics and footer");
        (fdt, fdx, fdm, index_end)
    }
    /// Java's `finish(numDocs)`: flush whatever is buffered as a final dirty
    /// chunk, then assemble `.fdt`/`.fdx`/`.fdm`.
    pub fn finish(mut self) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        if self.num_buffered_docs() > 0 {
            self.flush(true);
        }
        let max_doc = self.doc_base;
        debug_assert!(
            self.fdt_out.is_none(),
            "a streamed writer finishes streamed"
        );
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
            self.mode.block_shift() as u32,
            self.num_dirty_chunks,
            self.num_dirty_docs,
        );
        (self.fdt, fdx, fdm)
    }
}

/// Shared `.fdx`/`.fdm` assembly for both modes: `docs_values` and
/// `start_pointers_values` each hold one entry per chunk plus a trailing
/// sentinel (`maxDoc` and `maxPointer` respectively), so the chunk count both
/// files record is one less than their length. Formats differ only in the
/// `chunkSize` written to `.fdm`, which real Lucene needs in order to split a
/// `sliced` chunk back into its compression units, and in the block shift.
#[allow(clippy::too_many_arguments)]
fn write_index_and_meta(
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    max_doc: i32,
    docs_values: &[i64],
    start_pointers_values: &[i64],
    max_pointer: i64,
    chunk_size: i32,
    block_shift: u32,
    num_dirty_chunks: i64,
    num_dirty_docs: i64,
) -> (Vec<u8>, Vec<u8>) {
    debug_assert_eq!(docs_values.len(), start_pointers_values.len());

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
/// (`NUM_SUB_BLOCKS`, `DEFLATE_DICT_SIZE_FACTOR` in [`compression`]) -- real Lucene reuses
/// this sizing regardless of a chunk's actual size, so this port does too
/// rather than inventing a different threshold. Note the vint framing is
/// *not* the same shape as LZ4's: DEFLATE interleaves each unit's compressed
/// length immediately before that unit's bytes, where LZ4 batches all of
/// them up front (see [`compression::decompress`]).
///
/// Each sub-block is compressed with the chunk's dictionary preset
/// (`Deflater.setDictionary`), through `deflater::Deflater` -- a port of the zlib
/// `deflate` Java's `Deflater` runs -- so the `.fdt` bytes are Lucene's.
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

/// How many bytes [`serialize_doc_into`] appends for `doc`, counted without
/// appending a string's or a byte array's contents.
fn serialized_doc_len(doc: &Document) -> usize {
    let mut small = Vec::new();
    let mut total = 0usize;
    for field in &doc.fields {
        let bits = match &field.value {
            FieldValue::String(_) | FieldValue::Shared(_) => TYPE_STRING,
            FieldValue::Binary(_) => TYPE_BYTE_ARR,
            FieldValue::Int(_) => TYPE_NUMERIC_INT,
            FieldValue::Float(_) => TYPE_NUMERIC_FLOAT,
            FieldValue::Long(_) => TYPE_NUMERIC_LONG,
            FieldValue::Double(_) => TYPE_NUMERIC_DOUBLE,
        };
        small.clear();
        small.write_vlong(((field.field_number as i64) << TYPE_BITS) | bits);
        match &field.value {
            FieldValue::String(s) => small.write_vint(s.len() as i32),
            FieldValue::Shared(s) => small.write_vint(s.len() as i32),
            FieldValue::Binary(b) => small.write_vint(b.len() as i32),
            other => write_field(&mut small, other),
        }
        let contents = match &field.value {
            FieldValue::String(s) => s.len(),
            FieldValue::Shared(s) => s.len(),
            FieldValue::Binary(b) => b.len(),
            _ => 0,
        };
        // ARITH: a document's serialized length is a `usize` the caller
        // bounds to `i32` before using it (`streamed_len`); the sums here
        // are of in-memory buffers, so they stay under `isize::MAX`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            total += small.len() + contents;
        }
    }
    total
}

/// Appends one document's `infoAndBits`/value pairs to the chunk buffer --
/// Java's `writeField` overloads writing into `bufferedDocs`.
fn serialize_doc_into(doc: &Document, out: &mut Vec<u8>) {
    for field in &doc.fields {
        let bits = match &field.value {
            FieldValue::String(_) | FieldValue::Shared(_) => TYPE_STRING,
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
        FieldValue::Shared(s) => out.write_string(s),
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
fn visit_field<I: DocInput>(
    input: &mut I,
    field_number: i32,
    bits: i64,
    defer_from: usize,
    visitor: &mut dyn StoredFieldVisitor,
) -> Result<()> {
    match bits {
        TYPE_BYTE_ARR | TYPE_STRING => {
            // The length is a vint out of the decompressed payload, so an
            // unbounded one is a `2^64` slice.
            let length = read_value_length(input)?;
            let string = bits == TYPE_STRING;
            if length >= defer_from {
                input.skip_bytes(length)?;
                return visitor.deferred_field(field_number, string, length);
            }
            // Borrowed out of the decompressed bytes: the visitor takes
            // `&str`, so validating in place spares a `String` per field.
            // Same UTF-8 error as `read_string`.
            let bytes = input.value(length)?;
            if string {
                let value = std::str::from_utf8(bytes)
                    .map_err(|_| lucene_store::Error::Corrupted("invalid UTF-8 string".into()))?;
                visitor.string_field(field_number, value)
            } else {
                visitor.binary_field(field_number, bytes)
            }
        }
        TYPE_NUMERIC_INT => visitor.int_field(field_number, input.with_window(read_zint)?),
        TYPE_NUMERIC_FLOAT => visitor.float_field(field_number, input.with_window(read_zfloat)?),
        TYPE_NUMERIC_LONG => visitor.long_field(field_number, input.with_window(read_tlong)?),
        TYPE_NUMERIC_DOUBLE => visitor.double_field(field_number, input.with_window(read_zdouble)?),
        other => Err(Error::UnknownTypeTag(other)),
    }
}

/// A `STRING`/`BYTE_ARR` value's length vint, checked against the bytes the
/// document has left.
fn read_value_length<I: DocInput>(input: &mut I) -> Result<usize> {
    let len = input.with_window(|i| i.read_vint().map_err(Into::into))?;
    let len = usize::try_from(len).map_err(|_| {
        lucene_store::Error::Corrupted(format!("negative stored field length: {len}"))
    })?;
    if len > input.left() {
        return Err(lucene_store::Error::Corrupted(format!(
            "stored field length {len} exceeds {} remaining bytes",
            input.left()
        ))
        .into());
    }
    Ok(len)
}

/// Port of `Lucene90CompressingStoredFieldsReader.skipField`: advances the
/// cursor past one value without decoding it.
///
/// The numeric cases still *read* their bytes, because their encodings are
/// variable-length and self-delimiting -- Java calls the very same
/// `readZFloat`/`readTLong`/`readZDouble` here. What is saved is the
/// `String`/`Vec` a `STRING`/`BYTE_ARR` value would have allocated, which is
/// the whole of the cost for the wide documents this exists for.
fn skip_field<I: DocInput>(input: &mut I, bits: i64) -> Result<()> {
    match bits {
        TYPE_BYTE_ARR | TYPE_STRING => {
            let length = read_value_length(input)?;
            input.skip_bytes(length)?;
        }
        TYPE_NUMERIC_INT => {
            input.with_window(read_zint)?;
        }
        TYPE_NUMERIC_FLOAT => {
            input.with_window(read_zfloat)?;
        }
        TYPE_NUMERIC_LONG => {
            input.with_window(read_tlong)?;
        }
        TYPE_NUMERIC_DOUBLE => {
            input.with_window(read_zdouble)?;
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
    let mut out = Vec::new();
    read_bulk_ints_into(input, count, &mut out)?;
    Ok(out)
}

/// [`read_bulk_ints`], appending the `count` values to `out` -- which lets
/// `read_chunk_header` decode the lengths straight into the `offsets` array
/// it turns them into, rather than through a second array.
fn read_bulk_ints_into(input: &mut SliceInput, count: usize, out: &mut Vec<i64>) -> Result<()> {
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
            out.resize(out.len().saturating_add(count), i64::from(v));
            Ok(())
        }
        8 => read_packed_ints::<8>(input, count, out),
        16 => read_packed_ints::<16>(input, count, out),
        32 => read_packed_ints::<32>(input, count, out),
        other => Err(Error::UnsupportedBulkIntWidth(other)),
    }
}

/// The 8/16/32-bit shapes of [`read_bulk_ints_into`]: full 128-value blocks,
/// then a scalar tail.
///
/// Java packs each block across `64 / BPV` lanes of `128 / (64 / BPV)` words
/// (`StoredFieldsInts.readInts8/16/32`): value `i + lane * num_words` sits in
/// word `i`'s `lane`-th slot, most significant first. Walking lane by lane
/// therefore emits the block in output order, so each lane is one sequential
/// `extend` over the block's words.
// ARITH: `BPV` is 8, 16 or 32, so `values_per_word` is 8, 4 or 2,
// `num_words` 16, 32 or 64 and every shift at most `64 - BPV`. `full <=
// count / 128`, so `full * block_bytes` is at most `count * BPV / 8`, and
// `count` is a chunk's document count, at most 2^16 (`read_chunk_header`).
#[allow(clippy::arithmetic_side_effects)]
fn read_packed_ints<const BPV: usize>(
    input: &mut SliceInput,
    count: usize,
    out: &mut Vec<i64>,
) -> Result<()> {
    const BLOCK_SIZE: usize = 128;
    let values_per_word = 64 / BPV;
    let num_words = BLOCK_SIZE / values_per_word;
    let block_bytes = num_words * 8;
    let mask: u64 = (1u64 << BPV) - 1;
    let full = count / BLOCK_SIZE;
    out.reserve(count);
    let packed = input.slice(input.position(), input.position() + full * block_bytes)?;
    input.skip(packed.len())?;
    let mut words = [0u64; 64];
    for block in packed.chunks_exact(block_bytes) {
        for (w, b) in words.iter_mut().zip(block.chunks_exact(8)) {
            *w = u64::from_le_bytes(b.try_into().expect("chunks_exact(8)"));
        }
        for lane in 0..values_per_word {
            let shift = (values_per_word - 1 - lane) * BPV;
            out.extend(
                words[..num_words]
                    .iter()
                    .map(|&w| ((w >> shift) & mask) as i64),
            );
        }
    }
    for _ in full * BLOCK_SIZE..count {
        out.push(read_scalar(input, BPV as u8)?);
    }
    Ok(())
}

/// The first document of a chunk whose serialized length is zero while its
/// stored-field count is not, or the reverse, from the chunk's cumulative
/// `offsets`.
#[cold]
fn first_length_count_mismatch(offsets: &[i64], num_stored_fields: &[i64]) -> Error {
    // ARITH: `offsets` is a running sum of non-negative lengths, so each
    // adjacent difference is a non-negative length.
    #[allow(clippy::arithmetic_side_effects)]
    let found = offsets
        .windows(2)
        .zip(num_stored_fields)
        .map(|(w, &n)| (w[1] - w[0], n))
        .find(|&(len, n)| (len == 0) != (n == 0));
    let (length, num_stored_fields) = found.unwrap_or((0, 0));
    Error::LengthFieldCountMismatch {
        length,
        num_stored_fields,
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

#[cfg(test)]
mod tests;
