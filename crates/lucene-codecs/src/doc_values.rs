//! Port of `org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat`
//! (`.dvm` metadata + `.dvd` data + `.dvs` skip index) — read-only. All five
//! doc-values types are supported (NUMERIC, BINARY, SORTED, SORTED_SET,
//! SORTED_NUMERIC).
//!
//! A field with a non-`None` `doc_values_skip_index_type` carries an extra
//! [`DocValuesSkipperMeta`] record in `.dvm` (`Lucene90DocValuesProducer
//! .readDocValueSkipperMeta`) pointing at a run of per-interval min/max/doc-
//! count summaries in `.dvs`, decoded eagerly here by [`parse_skip_index`]
//! into a [`DocValuesSkipIndex`] -- see that function's doc comment for the
//! on-disk level structure. This is decode-only: nothing in this crate or
//! `lucene-search`'s `doc_value_query.rs` yet consults a decoded skip index
//! to prune a range scan (see `docs/parity.md` and the follow-up tracked as
//! "Doc-values skip index in range query scan").
//!
//! `SORTED` fields store a per-doc ordinal (reusing the exact NUMERIC entry
//! layout — see [`NumericEntry`]) into a terms dictionary
//! ([`crate::terms_dict`]) mapping each ordinal to its term bytes.
//! `SORTED_NUMERIC`/`SORTED_SET` are the multi-valued forms: zero or more
//! numbers/ordinals per doc, addressed through a [`crate::direct_monotonic`]
//! array of `(start, end)` ranges into a flat values array — unless every
//! doc-with-a-value happens to have exactly one, in which case Java (and
//! this port) collapses back to the single-valued shape with no addresses
//! array at all. A `SORTED_SET` field storing zero-or-one ordinal per doc
//! is written as a plain `SORTED` field with a multi-valued flag byte
//! (see [`SortedSetFieldEntry`]), not as a degenerate `SORTED_NUMERIC`.
//!
//! A NUMERIC field's values can also be split into "varying bits-per-value"
//! blocks (`Lucene90DocValuesConsumer.writeValues`'s `doBlocks` path): rather
//! than one uniform width for the whole field, each `2^blockShift`-value
//! block (16384 in real Lucene) is independently bit-packed, whenever doing
//! so saves at least 10% versus the whole-field width. On disk this is
//! `tableSize < -1` in the meta (`blockShift = -2 - tableSize`); see
//! [`NumericEntry::block_shift`] and [`decode_value_varying_bpv`] for the
//! block layout this decodes.
//!
//! Three encodings for a NUMERIC field's per-doc values (`bitsPerValue` in
//! the meta):
//! - **constant** (`bitsPerValue == 0`): every doc with a value has the same
//!   one, stored directly as `minValue`.
//! - **table-compressed**: a small (`<= 256` entry) lookup table of distinct
//!   values; each doc stores a bit-packed ordinal into it.
//! - **delta/GCD-compressed**: each doc stores a bit-packed `(value - min) /
//!   gcd`; `gcd == 1 && min == 0` is the common case (e.g. plain ordinals).
//!
//! BINARY fields are simpler: a flat concatenated byte blob, addressed either
//! directly (`doc * length`, fixed-width) or through a [`crate::direct_monotonic`]
//! array of end offsets (variable-width).
//!
//! As with [`crate::norms`], docs-with-a-value is one of empty/dense/sparse
//! (dense: implicit by doc id; sparse: via [`crate::indexed_disi`]).

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::DataOutput;

use crate::direct_monotonic;
use crate::direct_reader;
use crate::field_infos::FieldInfos;
use crate::indexed_disi;
use crate::lz4;
use crate::terms_dict::{self, TermsDictEntry};

const SKIP_INDEX_META_CODEC: &str = "Lucene90DocValuesSkipIndex";
const DATA_META_CODEC: &str = "Lucene90DocValuesData";

const META_CODEC: &str = "Lucene90DocValuesMetadata";
const VERSION_START: i32 = 0;
const VERSION_SKIPPER_MAX_VALUE_COUNT: i32 = 2;
const VERSION_CURRENT: i32 = VERSION_SKIPPER_MAX_VALUE_COUNT;

const DOC_VALUES_TYPE_NUMERIC: u8 = 0;
const DOC_VALUES_TYPE_BINARY: u8 = 1;
const DOC_VALUES_TYPE_SORTED: u8 = 2;
const DOC_VALUES_TYPE_SORTED_SET: u8 = 3;
const DOC_VALUES_TYPE_SORTED_NUMERIC: u8 = 4;

const DOCS_WITH_FIELD_EMPTY: i64 = -2;
const DOCS_WITH_FIELD_DENSE: i64 = -1;

/// `Lucene90DocValuesFormat.TERMS_DICT_BLOCK_LZ4_SHIFT`/`_MASK`: 64 terms per
/// LZ4-compressed block (matches [`crate::terms_dict`]'s `BLOCK_SIZE`).
const TERMS_DICT_BLOCK_LZ4_SHIFT: i64 = 6;
const TERMS_DICT_BLOCK_LZ4_MASK: i64 = (1 << TERMS_DICT_BLOCK_LZ4_SHIFT) - 1;
/// `Lucene90DocValuesFormat.TERMS_DICT_REVERSE_INDEX_SHIFT`/`_MASK`: one
/// sampled term address every 1024 ordinals, for the coarse reverse index.
const TERMS_DICT_REVERSE_INDEX_SHIFT: u32 = 10;
const TERMS_DICT_REVERSE_INDEX_MASK: i64 = (1i64 << TERMS_DICT_REVERSE_INDEX_SHIFT) - 1;
/// `DirectMonotonicWriter` block shift for both the terms-address and
/// reverse-index address arrays -- real Lucene always uses 16
/// (`Lucene90DocValuesFormat.DIRECT_MONOTONIC_BLOCK_SHIFT`); this port picks
/// 0 instead, same simplicity-over-compression-ratio choice
/// [`write_single_dense_binary_field`]'s variable-length address array
/// already made for its own [`direct_monotonic::write`] call -- the format
/// stores `block_shift` per-field, so any value round-trips correctly.
const TERMS_DICT_DIRECT_MONOTONIC_BLOCK_SHIFT: u32 = 0;

/// `Lucene90DocValuesFormat.SKIP_INDEX_MAX_LEVEL`: at most 4 levels in the
/// multi-level skip structure written per doc-values field.
const SKIP_INDEX_MAX_LEVEL: u8 = 4;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("unknown field number: {0}")]
    UnknownFieldNumber(i32),
    #[error("field {0} has unknown doc values type byte {1}")]
    UnsupportedFieldType(i32, u8),
    #[error("invalid table size: {0}")]
    InvalidTableSize(i32),
    #[error("invalid bitsPerValue: {0}, field number {1}")]
    InvalidBitsPerValue(u8, i32),
    #[error("doc {0} is out of range (numValues={1})")]
    DocOutOfRange(i32, i64),
    #[error("field {0} has invalid multiValued flag: {1}")]
    InvalidMultiValuedFlag(i32, u8),
    #[error("doc-values skip index level count {0} is out of range (must be 1..={SKIP_INDEX_MAX_LEVEL})")]
    InvalidSkipIndexLevelCount(u8),
    #[error(
        "doc-values skip index offset/length ({0}, {1}) is out of range for a {2}-byte .dvs slice"
    )]
    SkipIndexOutOfRange(i64, i64, usize),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct NumericEntry {
    pub field_number: i32,
    pub docs_with_field_offset: i64,
    pub docs_with_field_length: i64,
    pub jump_table_entry_count: i16,
    pub dense_rank_power: u8,
    pub num_values: i64,
    pub table: Option<Vec<i64>>,
    pub bits_per_value: u8,
    pub min_value: i64,
    pub gcd: i64,
    pub values_offset: i64,
    pub values_length: i64,
    /// `Some(shift)` when this field's values are split into independently
    /// bit-packed blocks of `2^shift` values each (`Lucene90DocValuesConsumer
    /// .writeValues`'s `doBlocks` path, on disk as `tableSize < -1` ->
    /// `blockShift = -2 - tableSize`) rather than one uniform width for the
    /// whole field. `bits_per_value` is then a meaningless `0xFF` sentinel;
    /// each block carries its own width and its own `min`-equivalent delta,
    /// read via [`value_jump_table_offset`](Self::value_jump_table_offset)
    /// -- see [`decode_value_varying_bpv`].
    pub block_shift: Option<u32>,
    /// Absolute byte offset into the whole `.dvd` file of this field's
    /// per-block offset table (one `i64` absolute block-start offset per
    /// block, plus one trailing self-referential entry) -- only meaningful
    /// when [`block_shift`](Self::block_shift) is `Some`.
    pub value_jump_table_offset: i64,
}

impl NumericEntry {
    pub fn is_empty_field(&self) -> bool {
        self.docs_with_field_offset == DOCS_WITH_FIELD_EMPTY
    }

    pub fn is_dense(&self) -> bool {
        self.docs_with_field_offset == DOCS_WITH_FIELD_DENSE
    }
}

#[derive(Debug, Clone)]
pub struct BinaryEntry {
    pub field_number: i32,
    pub docs_with_field_offset: i64,
    pub docs_with_field_length: i64,
    pub jump_table_entry_count: i16,
    pub dense_rank_power: u8,
    pub num_docs_with_field: i32,
    pub min_length: i32,
    pub max_length: i32,
    pub data_offset: i64,
    pub data_length: i64,
    /// `None` for fixed-length fields (`min_length == max_length`), where an
    /// entry's offset is computed directly as `ordinal * max_length`.
    pub addresses: Option<BinaryAddresses>,
}

#[derive(Debug, Clone)]
pub struct BinaryAddresses {
    pub offset: i64,
    pub length: i64,
    pub meta: direct_monotonic::Meta,
}

impl BinaryEntry {
    pub fn is_empty_field(&self) -> bool {
        self.docs_with_field_offset == DOCS_WITH_FIELD_EMPTY
    }

    pub fn is_dense(&self) -> bool {
        self.docs_with_field_offset == DOCS_WITH_FIELD_DENSE
    }

    pub fn is_fixed_length(&self) -> bool {
        self.min_length == self.max_length
    }
}

/// A single-valued SORTED field: a per-doc ordinal ([`NumericEntry`],
/// exactly the same layout NUMERIC fields use) into a terms dictionary.
#[derive(Debug, Clone)]
pub struct SortedEntry {
    pub field_number: i32,
    pub ords: NumericEntry,
    pub terms: TermsDictEntry,
}

/// A `(start, end)` address range array into a flat multi-valued array --
/// present only when a doc-with-a-value can have more than one value; if
/// every doc-with-a-value has exactly one, Java (and this port) collapses
/// to the single-valued shape (`addresses: None`) with no address array at
/// all, since each value's ordinal is then just its doc's rank.
#[derive(Debug, Clone)]
pub struct MultiValueAddresses {
    pub offset: i64,
    pub length: i64,
    pub meta: direct_monotonic::Meta,
}

/// A multi-valued (zero or more per doc) numeric field, or the ordinal
/// half of a multi-valued SORTED_SET field -- both share this exact entry
/// shape (`Lucene90DocValuesProducer.readSortedNumeric`).
#[derive(Debug, Clone)]
pub struct SortedNumericEntry {
    pub field_number: i32,
    pub numeric: NumericEntry,
    pub num_docs_with_field: i32,
    pub addresses: Option<MultiValueAddresses>,
}

/// A SORTED_SET field: either written as a plain single-valued [`SortedEntry`]
/// (every doc has zero or one ordinal) or a true multi-valued form (ordinals
/// via [`SortedNumericEntry`], resolved against the same terms dictionary).
#[derive(Debug, Clone)]
pub enum SortedSetKind {
    Single(SortedEntry),
    Multi {
        ords: SortedNumericEntry,
        terms: TermsDictEntry,
    },
}

#[derive(Debug, Clone)]
pub struct SortedSetEntry {
    pub field_number: i32,
    pub kind: SortedSetKind,
}

/// `Lucene90DocValuesProducer.DocValuesSkipperEntry`: the fixed-size summary
/// recorded in `.dvm` for a field with a doc-values skip index, pointing at
/// its per-interval detail in `.dvs` (see [`DocValuesSkipIndex`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DocValuesSkipperMeta {
    /// Byte offset of this field's skip data within `.dvs`.
    pub offset: i64,
    /// Byte length of this field's skip data within `.dvs`.
    pub length: i64,
    pub min_value: i64,
    pub max_value: i64,
    pub doc_count: i32,
    pub max_doc_id: i32,
    /// Most values per doc seen across the whole field, or `-1` if unknown
    /// (pre-`VERSION_SKIPPER_MAX_VALUE_COUNT` files never recorded it and
    /// `doc_count == 0` is the only case this port can still infer as `0`).
    pub max_value_count: i32,
}

/// One level's summary within a single skip interval:
/// `Lucene90DocValuesConsumer.SkipAccumulator`, as written by `writeLevels`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkipIndexLevelInterval {
    pub min_doc_id: i32,
    pub max_doc_id: i32,
    pub min_value: i64,
    pub max_value: i64,
    pub doc_count: i32,
}

/// One skip interval, one entry per level it was written at.
/// `levels[0]` is the finest (base, every-`skipIndexIntervalSize`-docs)
/// interval; `levels[levels.len() - 1]` is the coarsest. On disk the levels
/// are written coarsest-first (so a lazy/streaming reader can bail out
/// after the first uncompetitive level without reading the rest) but this
/// port reorders them to level-ascending on the way in, since it decodes
/// every interval eagerly and ascending is the more natural iteration order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkipIndexInterval {
    pub levels: Vec<SkipIndexLevelInterval>,
}

/// A field's whole decoded doc-values skip index: the `.dvm` summary plus
/// every interval decoded out of its `.dvs` slice. See [`parse_skip_index`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocValuesSkipIndex {
    pub min_value: i64,
    pub max_value: i64,
    pub doc_count: i32,
    pub max_doc_id: i32,
    pub max_value_count: i32,
    pub intervals: Vec<SkipIndexInterval>,
}

#[derive(Debug, Clone, Default)]
pub struct DocValuesMeta {
    pub numeric: Vec<NumericEntry>,
    pub binary: Vec<BinaryEntry>,
    pub sorted: Vec<SortedEntry>,
    pub sorted_numeric: Vec<SortedNumericEntry>,
    pub sorted_set: Vec<SortedSetEntry>,
    /// Keyed by field number, present only for fields whose `FieldInfo` has
    /// a non-`None` `doc_values_skip_index_type` (mirrors the Java reader's
    /// own `skippers` map -- a skip index is orthogonal to the field's
    /// doc-values *type*, so it isn't folded into any of the `Vec`s above).
    pub skippers: std::collections::HashMap<i32, DocValuesSkipperMeta>,
}

impl DocValuesMeta {
    pub fn numeric_entry(&self, field_number: i32) -> Option<&NumericEntry> {
        self.numeric.iter().find(|e| e.field_number == field_number)
    }

    pub fn binary_entry(&self, field_number: i32) -> Option<&BinaryEntry> {
        self.binary.iter().find(|e| e.field_number == field_number)
    }

    pub fn sorted_entry(&self, field_number: i32) -> Option<&SortedEntry> {
        self.sorted.iter().find(|e| e.field_number == field_number)
    }

    pub fn sorted_numeric_entry(&self, field_number: i32) -> Option<&SortedNumericEntry> {
        self.sorted_numeric
            .iter()
            .find(|e| e.field_number == field_number)
    }

    pub fn sorted_set_entry(&self, field_number: i32) -> Option<&SortedSetEntry> {
        self.sorted_set
            .iter()
            .find(|e| e.field_number == field_number)
    }

    /// The `.dvm` skip-index summary for `field_number`, if that field has
    /// one. Pass it to [`parse_skip_index`] along with the segment's `.dvs`
    /// bytes to decode the full per-interval structure.
    pub fn skipper_meta(&self, field_number: i32) -> Option<&DocValuesSkipperMeta> {
        self.skippers.get(&field_number)
    }
}

/// Parses a whole `.dvm` metadata file already read into memory. `field_infos`
/// must be the segment's already-parsed `.fnm` (used to reject unknown field
/// numbers and to detect unsupported per-field doc-values skip indexes).
pub fn parse_meta(
    buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    field_infos: &FieldInfos,
) -> Result<(i32, DocValuesMeta)> {
    let mut input = SliceInput::new(buf);
    let header = codec_util::check_index_header(
        &mut input,
        META_CODEC,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;

    let mut meta = DocValuesMeta::default();
    loop {
        let field_number = input.read_i32()?;
        if field_number == -1 {
            break;
        }
        let field = field_infos
            .field_by_number(field_number)
            .ok_or(Error::UnknownFieldNumber(field_number))?;

        let ty = input.read_byte()?;
        if field.doc_values_skip_index_type != crate::field_infos::DocValuesSkipIndexType::None {
            let skipper = read_skipper_meta(&mut input, header.version)?;
            meta.skippers.insert(field_number, skipper);
        }
        match ty {
            DOC_VALUES_TYPE_NUMERIC => {
                meta.numeric
                    .push(read_numeric_entry(&mut input, field_number)?);
            }
            DOC_VALUES_TYPE_BINARY => {
                meta.binary
                    .push(read_binary_entry(&mut input, field_number)?);
            }
            DOC_VALUES_TYPE_SORTED => {
                meta.sorted
                    .push(read_sorted_entry(&mut input, field_number)?);
            }
            DOC_VALUES_TYPE_SORTED_NUMERIC => {
                meta.sorted_numeric
                    .push(read_sorted_numeric_entry(&mut input, field_number)?);
            }
            DOC_VALUES_TYPE_SORTED_SET => {
                meta.sorted_set
                    .push(read_sorted_set_entry(&mut input, field_number)?);
            }
            other => return Err(Error::UnsupportedFieldType(field_number, other)),
        }
    }

    codec_util::check_footer(&mut input, buf.len())?;

    Ok((header.version, meta))
}

fn read_binary_entry(input: &mut SliceInput, field_number: i32) -> Result<BinaryEntry> {
    let data_offset = input.read_i64()?;
    let data_length = input.read_i64()?;
    let docs_with_field_offset = input.read_i64()?;
    let docs_with_field_length = input.read_i64()?;
    let jump_table_entry_count = input.read_i16()?;
    let dense_rank_power = input.read_byte()?;
    let num_docs_with_field = input.read_i32()?;
    let min_length = input.read_i32()?;
    let max_length = input.read_i32()?;

    let addresses = if min_length < max_length {
        let addresses_offset = input.read_i64()?;
        let num_addresses = num_docs_with_field as i64 + 1;
        let block_shift = input.read_vint()?;
        let addr_meta = direct_monotonic::load_meta(input, num_addresses, block_shift as u32)?;
        let addresses_length = input.read_i64()?;
        Some(BinaryAddresses {
            offset: addresses_offset,
            length: addresses_length,
            meta: addr_meta,
        })
    } else {
        None
    };

    let _ = field_number;
    Ok(BinaryEntry {
        field_number,
        docs_with_field_offset,
        docs_with_field_length,
        jump_table_entry_count,
        dense_rank_power,
        num_docs_with_field,
        min_length,
        max_length,
        data_offset,
        data_length,
        addresses,
    })
}

fn read_sorted_entry(input: &mut SliceInput, field_number: i32) -> Result<SortedEntry> {
    let ords = read_numeric_entry(input, field_number)?;
    let terms = terms_dict::read_term_dict_entry(input)?;
    Ok(SortedEntry {
        field_number,
        ords,
        terms,
    })
}

/// Port of `Lucene90DocValuesProducer.readSortedNumeric`: a NUMERIC entry
/// (`numValues` = total value count across all docs) plus, only when a
/// doc-with-a-value can hold more than one value, an address array mapping
/// each doc's rank to a `[start, end)` range in that flat value array.
fn read_sorted_numeric_entry(
    input: &mut SliceInput,
    field_number: i32,
) -> Result<SortedNumericEntry> {
    let numeric = read_numeric_entry(input, field_number)?;
    let num_docs_with_field = input.read_i32()?;
    let addresses = if num_docs_with_field as i64 != numeric.num_values {
        let addresses_offset = input.read_i64()?;
        let block_shift = input.read_vint()?;
        let meta =
            direct_monotonic::load_meta(input, num_docs_with_field as i64 + 1, block_shift as u32)?;
        let addresses_length = input.read_i64()?;
        Some(MultiValueAddresses {
            offset: addresses_offset,
            length: addresses_length,
            meta,
        })
    } else {
        None
    };
    Ok(SortedNumericEntry {
        field_number,
        numeric,
        num_docs_with_field,
        addresses,
    })
}

/// Port of `Lucene90DocValuesProducer.readSortedSet`.
fn read_sorted_set_entry(input: &mut SliceInput, field_number: i32) -> Result<SortedSetEntry> {
    let multi_valued = input.read_byte()?;
    let kind = match multi_valued {
        0 => SortedSetKind::Single(read_sorted_entry(input, field_number)?),
        1 => {
            let ords = read_sorted_numeric_entry(input, field_number)?;
            let terms = terms_dict::read_term_dict_entry(input)?;
            SortedSetKind::Multi { ords, terms }
        }
        other => return Err(Error::InvalidMultiValuedFlag(field_number, other)),
    };
    Ok(SortedSetEntry { field_number, kind })
}

fn read_numeric_entry(input: &mut SliceInput, field_number: i32) -> Result<NumericEntry> {
    let docs_with_field_offset = input.read_i64()?;
    let docs_with_field_length = input.read_i64()?;
    let jump_table_entry_count = input.read_i16()?;
    let dense_rank_power = input.read_byte()?;
    let num_values = input.read_i64()?;

    let table_size = input.read_i32()?;
    if table_size > 256 {
        return Err(Error::InvalidTableSize(table_size));
    }
    // `tableSize < -1` means this field's values are split into varying-
    // bits-per-value blocks (`blockShift = -2 - tableSize`) rather than a
    // lookup table; `bits_per_value` below is then a meaningless `0xFF`
    // sentinel (real Lucene writes `numBitsPerValue = 0xFF` in this case),
    // so skip the normal table/enum reads for it.
    let (table, block_shift) = if table_size < -1 {
        let shift = -2i64 - table_size as i64;
        if !(0..=63).contains(&shift) {
            return Err(Error::InvalidTableSize(table_size));
        }
        (None, Some(shift as u32))
    } else if table_size >= 0 {
        let mut t = Vec::with_capacity(table_size as usize);
        for _ in 0..table_size {
            t.push(input.read_i64()?);
        }
        (Some(t), None)
    } else {
        (None, None)
    };

    let bits_per_value = input.read_byte()?;
    if block_shift.is_none()
        && !matches!(
            bits_per_value,
            0 | 1 | 2 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 40 | 48 | 56 | 64
        )
    {
        return Err(Error::InvalidBitsPerValue(bits_per_value, field_number));
    }
    let min_value = input.read_i64()?;
    let gcd = input.read_i64()?;
    let values_offset = input.read_i64()?;
    let values_length = input.read_i64()?;
    let value_jump_table_offset = input.read_i64()?; // only meaningful for varying-bpv blocks

    Ok(NumericEntry {
        field_number,
        docs_with_field_offset,
        docs_with_field_length,
        jump_table_entry_count,
        dense_rank_power,
        num_values,
        table,
        bits_per_value,
        min_value,
        gcd,
        values_offset,
        values_length,
        block_shift,
        value_jump_table_offset,
    })
}

/// Validates a whole `.dvd` data file's header/footer, mirroring
/// [`crate::norms::check_data_header_footer`] (structural-only footer check,
/// no full-file CRC — same rationale: forward-only doc-values access makes a
/// full recompute too costly to do on every open).
pub fn check_data_header_footer(
    buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<i32> {
    let mut input = SliceInput::new(buf);
    let header = codec_util::check_index_header(
        &mut input,
        "Lucene90DocValuesData",
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    codec_util::retrieve_checksum(buf)?;
    Ok(header.version)
}

/// Reads one `Lucene90DocValuesProducer.DocValuesSkipperEntry` out of
/// `.dvm` -- called from [`parse_meta`] right after the field's type byte,
/// for any field whose `FieldInfo` carries a non-`None`
/// `doc_values_skip_index_type`. `meta_version` is the `.dvm` header's own
/// version (not `.dvs`'s), matching Java's `readDocValueSkipperMeta`, which
/// gates the trailing `maxValueCount` field on it.
fn read_skipper_meta(input: &mut SliceInput, meta_version: i32) -> Result<DocValuesSkipperMeta> {
    let offset = input.read_i64()?;
    let length = input.read_i64()?;
    let max_value = input.read_i64()?;
    let min_value = input.read_i64()?;
    let doc_count = input.read_i32()?;
    let max_doc_id = input.read_i32()?;
    let max_value_count = if meta_version >= VERSION_SKIPPER_MAX_VALUE_COUNT {
        input.read_i32()?
    } else if doc_count == 0 {
        0
    } else {
        -1
    };
    Ok(DocValuesSkipperMeta {
        offset,
        length,
        min_value,
        max_value,
        doc_count,
        max_doc_id,
        max_value_count,
    })
}

/// Validates a `.dvs` skip-index file's header/footer -- same structural,
/// no-full-checksum shape as [`check_data_header_footer`], but against
/// `.dvs`'s own codec name.
pub fn check_skip_index_header_footer(
    buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<i32> {
    let mut input = SliceInput::new(buf);
    let header = codec_util::check_index_header(
        &mut input,
        SKIP_INDEX_META_CODEC,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    codec_util::retrieve_checksum(buf)?;
    Ok(header.version)
}

/// Decodes the full per-interval doc-values skip index
/// (`Lucene90DocValuesConsumer.writeLevels`'s inverse) for one field, given
/// its `.dvm` summary (`skipper`) and the segment's whole `.dvs` buffer.
///
/// Real Lucene reads this lazily, interval-by-interval, while `advance`-ing
/// a query scan (`Lucene90DocValuesProducer.getSkipper`); this port instead
/// decodes every interval in `[skipper.offset, skipper.offset +
/// skipper.length)` up front into a `Vec`, which is simpler to reason about
/// and to test, at the cost of not being useful yet for actually skipping
/// (see the module doc comment's TODO on wiring this into a range-query
/// scan).
///
/// Each interval is a run of 1..=[`SKIP_INDEX_MAX_LEVEL`] levels, written
/// coarsest-first:
/// - one `u8` level count,
/// - then, per level from coarsest down to level 0: `maxDocID: i32`,
///   `minDocID: i32`, `maxValue: i64`, `minValue: i64`, `docCount: i32`
///   (28 bytes/level).
///
/// This function reads sequential intervals until the byte window is fully
/// consumed (real Lucene never needs to do this -- it stops once it finds a
/// competitive interval -- so there's no equivalent Java loop to mirror
/// beyond `writeLevels`'s write order).
pub fn parse_skip_index(
    dvs_buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    skipper: &DocValuesSkipperMeta,
) -> Result<DocValuesSkipIndex> {
    check_skip_index_header_footer(dvs_buf, segment_id, segment_suffix)?;

    let offset = usize::try_from(skipper.offset)
        .map_err(|_| Error::SkipIndexOutOfRange(skipper.offset, skipper.length, dvs_buf.len()))?;
    let length = usize::try_from(skipper.length)
        .map_err(|_| Error::SkipIndexOutOfRange(skipper.offset, skipper.length, dvs_buf.len()))?;
    let end = offset
        .checked_add(length)
        .filter(|&e| e <= dvs_buf.len())
        .ok_or(Error::SkipIndexOutOfRange(
            skipper.offset,
            skipper.length,
            dvs_buf.len(),
        ))?;

    let mut input = SliceInput::new(&dvs_buf[offset..end]);
    let mut intervals = Vec::new();
    while input.position() < length {
        let levels = input.read_byte()?;
        if levels == 0 || levels > SKIP_INDEX_MAX_LEVEL {
            return Err(Error::InvalidSkipIndexLevelCount(levels));
        }
        let mut by_level: Vec<Option<SkipIndexLevelInterval>> = vec![None; levels as usize];
        for level in (0..levels as usize).rev() {
            let max_doc_id = input.read_i32()?;
            let min_doc_id = input.read_i32()?;
            let max_value = input.read_i64()?;
            let min_value = input.read_i64()?;
            let doc_count = input.read_i32()?;
            by_level[level] = Some(SkipIndexLevelInterval {
                min_doc_id,
                max_doc_id,
                min_value,
                max_value,
                doc_count,
            });
        }
        intervals.push(SkipIndexInterval {
            levels: by_level
                .into_iter()
                .map(|l| l.expect("every level index was written above"))
                .collect(),
        });
    }

    Ok(DocValuesSkipIndex {
        min_value: skipper.min_value,
        max_value: skipper.max_value,
        doc_count: skipper.doc_count,
        max_doc_id: skipper.max_doc_id,
        max_value_count: skipper.max_value_count,
        intervals,
    })
}

/// Reads the numeric doc-values value for `doc`, handling all three
/// docs-with-a-value shapes (empty/dense/sparse) and all three per-value
/// encodings (constant/table/GCD-delta). `data` is the whole `.dvd` file's
/// bytes. `Ok(None)` means `doc` legitimately has no value.
pub fn numeric_value(data: &[u8], entry: &NumericEntry, doc: i32) -> Result<Option<i64>> {
    if doc < 0 {
        return Err(Error::DocOutOfRange(doc, entry.num_values));
    }
    if entry.is_empty_field() {
        return Ok(None);
    }
    if entry.is_dense() {
        if doc as i64 >= entry.num_values {
            return Err(Error::DocOutOfRange(doc, entry.num_values));
        }
        return Ok(Some(decode_value(data, entry, doc as i64)?));
    }

    let region = data
        .get(
            entry.docs_with_field_offset as usize
                ..(entry.docs_with_field_offset + entry.docs_with_field_length) as usize,
        )
        .ok_or(lucene_store::Error::Eof { offset: 0 })?;
    let doc_ids = indexed_disi::decode_doc_ids(region, entry.dense_rank_power)?;
    match indexed_disi::rank_of(&doc_ids, doc) {
        Some(ordinal) => Ok(Some(decode_value(data, entry, ordinal as i64)?)),
        None => Ok(None),
    }
}

/// Decodes the raw bit-packed value at `ordinal` (doc id for dense fields,
/// rank-among-present-docs for sparse ones) and applies the table or
/// GCD-delta transform to get the final value.
fn decode_value(data: &[u8], entry: &NumericEntry, ordinal: i64) -> Result<i64> {
    if let Some(shift) = entry.block_shift {
        return decode_value_varying_bpv(data, entry, shift, ordinal);
    }
    if entry.bits_per_value == 0 {
        return Ok(entry.min_value);
    }
    let values = data
        .get(entry.values_offset as usize..(entry.values_offset + entry.values_length) as usize)
        .ok_or(lucene_store::Error::Eof { offset: 0 })?;
    let raw = direct_reader::get(values, entry.bits_per_value, ordinal)?;

    if let Some(table) = &entry.table {
        let idx = usize::try_from(raw).map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
        table
            .get(idx)
            .copied()
            .ok_or(lucene_store::Error::Eof { offset: 0 }.into())
    } else {
        Ok(entry.gcd.wrapping_mul(raw).wrapping_add(entry.min_value))
    }
}

/// Decodes `ordinal`'s value out of a varying-bits-per-value field
/// (`entry.block_shift.is_some()`), mirroring
/// `Lucene90DocValuesProducer.VaryingBPVReader.getLongValue`.
///
/// Each block's own start offset (absolute in the whole `.dvd` file) is
/// looked up directly in the per-field jump table at
/// `entry.value_jump_table_offset + block * 8` -- no need to walk prior
/// blocks first, unlike a sequential scan. A block's own header is then:
/// one byte bits-per-value, an `i64` delta (the block's own min value, or
/// its single constant value when bits-per-value is 0), and -- only when
/// bits-per-value is non-zero -- an `i32` byte length followed by that many
/// bytes of bit-packed `(value - delta) / gcd` values (`entry.gcd` is the
/// whole field's GCD, reused by every block).
fn decode_value_varying_bpv(
    data: &[u8],
    entry: &NumericEntry,
    shift: u32,
    ordinal: i64,
) -> Result<i64> {
    let block = ordinal >> shift;
    let jump_table_pos = entry
        .value_jump_table_offset
        .checked_add(
            block
                .checked_mul(8)
                .ok_or(lucene_store::Error::Eof { offset: 0 })?,
        )
        .ok_or(lucene_store::Error::Eof { offset: 0 })?;
    let jump_table_pos =
        usize::try_from(jump_table_pos).map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
    let mut input = SliceInput::new(data);
    input.seek(jump_table_pos)?;
    let block_start =
        usize::try_from(input.read_i64()?).map_err(|_| lucene_store::Error::Eof { offset: 0 })?;

    input.seek(block_start)?;
    let bits_per_value = input.read_byte()?;
    let delta = input.read_i64()?;
    if bits_per_value == 0 {
        return Ok(delta);
    }
    let length = input.read_i32()?;
    let length = usize::try_from(length).map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
    let values_start = input.position();
    let values = data
        .get(values_start..values_start + length)
        .ok_or(lucene_store::Error::Eof { offset: 0 })?;
    let mask = (1i64 << shift) - 1;
    let raw = direct_reader::get(values, bits_per_value, ordinal & mask)?;
    Ok(entry.gcd.wrapping_mul(raw).wrapping_add(delta))
}

/// Reads the binary doc-values value for `doc`, handling all three
/// docs-with-a-value shapes (empty/dense/sparse) and both length shapes
/// (fixed/variable). `data` is the whole `.dvd` file's bytes. `Ok(None)`
/// means `doc` legitimately has no value.
pub fn binary_value<'d>(data: &'d [u8], entry: &BinaryEntry, doc: i32) -> Result<Option<&'d [u8]>> {
    if doc < 0 {
        return Err(Error::DocOutOfRange(doc, entry.num_docs_with_field as i64));
    }
    if entry.is_empty_field() {
        return Ok(None);
    }

    let ordinal = if entry.is_dense() {
        if doc >= entry.num_docs_with_field {
            return Err(Error::DocOutOfRange(doc, entry.num_docs_with_field as i64));
        }
        doc as i64
    } else {
        let region = data
            .get(
                entry.docs_with_field_offset as usize
                    ..(entry.docs_with_field_offset + entry.docs_with_field_length) as usize,
            )
            .ok_or(lucene_store::Error::Eof { offset: 0 })?;
        let doc_ids = indexed_disi::decode_doc_ids(region, entry.dense_rank_power)?;
        match indexed_disi::rank_of(&doc_ids, doc) {
            Some(ordinal) => ordinal as i64,
            None => return Ok(None),
        }
    };

    let bytes_region = data
        .get(entry.data_offset as usize..(entry.data_offset + entry.data_length) as usize)
        .ok_or(lucene_store::Error::Eof { offset: 0 })?;

    let (start, len) = if let Some(addrs) = &entry.addresses {
        let addr_data = data
            .get(addrs.offset as usize..(addrs.offset + addrs.length) as usize)
            .ok_or(lucene_store::Error::Eof { offset: 0 })?;
        let start = direct_monotonic::get(addr_data, &addrs.meta, ordinal)?;
        let end = direct_monotonic::get(addr_data, &addrs.meta, ordinal + 1)?;
        (start, end - start)
    } else {
        let length = entry.max_length as i64;
        (ordinal * length, length)
    };

    let value = bytes_region
        .get(start as usize..(start + len) as usize)
        .ok_or(lucene_store::Error::Eof { offset: 0 })?;
    Ok(Some(value))
}

/// Reads a SORTED field's ordinal for `doc` -- exactly [`numeric_value`]
/// applied to the entry's `ords`, since Java's `readSorted` reuses the
/// NUMERIC entry layout verbatim for ordinals.
pub fn sorted_ord(data: &[u8], entry: &SortedEntry, doc: i32) -> Result<Option<i64>> {
    numeric_value(data, &entry.ords, doc)
}

/// Reads a SORTED_NUMERIC field's values for `doc` (zero or more numbers,
/// in the order Lucene wrote them), or a SORTED_SET field's ordinals when
/// applied to its `ords` entry -- both share this exact decode. Unlike
/// [`numeric_value`], a doc's *ordinal into the flat values array* isn't
/// simply its doc id/rank: that only holds when every doc-with-a-value has
/// exactly one value (`entry.addresses.is_none()`); otherwise each doc's
/// rank indexes an address range `[start, end)` covering its values.
pub fn sorted_numeric_values(
    data: &[u8],
    entry: &SortedNumericEntry,
    doc: i32,
) -> Result<Vec<i64>> {
    if doc < 0 {
        return Err(Error::DocOutOfRange(doc, entry.numeric.num_values));
    }
    if entry.numeric.is_empty_field() {
        return Ok(Vec::new());
    }

    let rank: i64 = if entry.numeric.is_dense() {
        doc as i64
    } else {
        let region = data
            .get(
                entry.numeric.docs_with_field_offset as usize
                    ..(entry.numeric.docs_with_field_offset + entry.numeric.docs_with_field_length)
                        as usize,
            )
            .ok_or(lucene_store::Error::Eof { offset: 0 })?;
        let doc_ids = indexed_disi::decode_doc_ids(region, entry.numeric.dense_rank_power)?;
        match indexed_disi::rank_of(&doc_ids, doc) {
            Some(r) => r as i64,
            None => return Ok(Vec::new()),
        }
    };

    match &entry.addresses {
        None => Ok(vec![decode_value(data, &entry.numeric, rank)?]),
        Some(addrs) => {
            let addr_region = data
                .get(addrs.offset as usize..(addrs.offset + addrs.length) as usize)
                .ok_or(lucene_store::Error::Eof { offset: 0 })?;
            let start = direct_monotonic::get(addr_region, &addrs.meta, rank)?;
            let end = direct_monotonic::get(addr_region, &addrs.meta, rank + 1)?;
            (start..end)
                .map(|i| decode_value(data, &entry.numeric, i))
                .collect()
        }
    }
}

/// Write-side error, kept separate from the read-side [`Error`] since none
/// of its variants are decode failures.
#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    #[error("dense doc-values write requires values.len() == max_doc (every doc must have a value); got {values} values for max_doc={max_doc}")]
    NotDense { values: usize, max_doc: i32 },
    #[error("dense sorted-numeric write requires every doc to have at least one value; doc {0} has none")]
    EmptyMultiValuedDoc(i32),
    #[error("sparse doc-values write requires strictly ascending doc ids; doc {0} is out of order or duplicated")]
    DocIdsNotAscending(i32),
    #[error("sparse doc-values write requires doc {0} < max_doc={1}")]
    DocIdOutOfRange(i32, i32),
    #[error("write_dense_fields requires at least one field")]
    EmptyFieldList,
    #[error(
        "write_dense_fields requires distinct field numbers; field {0} appears more than once"
    )]
    DuplicateFieldNumber(i32),
}

pub type WriteResult<T> = std::result::Result<T, WriteError>;

/// One field's dense (every doc `0..max_doc` has a value) doc-values input,
/// as accepted by [`write_dense_fields`] -- one variant per
/// [`crate::field_infos::DocValuesType`]. Each variant's `values` slice must
/// have exactly `max_doc` entries (the same `max_doc` [`write_dense_fields`]
/// is called with, shared across every field in the call since they all
/// describe the same segment), mirroring each single-field function's own
/// `values.len() == max_doc` requirement ([`WriteError::NotDense`]).
///
/// `SortedNumeric`'s per-doc entry must be non-empty (a doc with zero values
/// is a sparse concept, out of scope for this dense-only entry point --
/// [`WriteError::EmptyMultiValuedDoc`]), same restriction
/// [`write_single_dense_sorted_numeric_field`] already enforces.
/// `SortedSet`'s per-doc entry has the same restriction.
#[derive(Debug, Clone, Copy)]
pub enum DenseField<'a> {
    Numeric(i32, &'a [i64]),
    Binary(i32, &'a [Vec<u8>]),
    Sorted(i32, &'a [Vec<u8>]),
    SortedNumeric(i32, &'a [Vec<i64>]),
    SortedSet(i32, &'a [Vec<Vec<u8>>]),
}

impl DenseField<'_> {
    fn field_number(&self) -> i32 {
        match self {
            DenseField::Numeric(n, _)
            | DenseField::Binary(n, _)
            | DenseField::Sorted(n, _)
            | DenseField::SortedNumeric(n, _)
            | DenseField::SortedSet(n, _) => *n,
        }
    }

    fn len(&self) -> usize {
        match self {
            DenseField::Numeric(_, v) => v.len(),
            DenseField::Binary(_, v) => v.len(),
            DenseField::Sorted(_, v) => v.len(),
            DenseField::SortedNumeric(_, v) => v.len(),
            DenseField::SortedSet(_, v) => v.len(),
        }
    }
}

/// Writes **one or more** distinct dense doc-values fields into a single
/// `.dvm`/`.dvd`/`.dvs` triple -- the multi-field analogue of
/// [`write_single_dense_numeric_field`]/[`write_single_dense_binary_field`]/
/// [`write_single_dense_sorted_field`]/
/// [`write_single_dense_sorted_numeric_field`]/
/// [`write_single_dense_sorted_set_field`], all five of which are now thin
/// one-element-slice wrappers over this function (kept so existing callers
/// are unaffected, same precedent as
/// [`crate::postings_writer::write_single_field`] over
/// [`crate::postings_writer::write_fields`]). `numFields` worth of per-field
/// meta entries are interleaved into the *same* physical meta/data buffers,
/// exactly like a real multi-field segment's `.dvm`/`.dvd`.
///
/// Every field in `fields` must be dense over the same `max_doc` (`values.len()
/// == max_doc` for every field -- [`WriteError::NotDense`]) and have a
/// distinct field number ([`WriteError::DuplicateFieldNumber`]); `fields` must
/// be non-empty ([`WriteError::EmptyFieldList`]).
pub fn write_dense_fields(
    fields: &[DenseField<'_>],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    if fields.is_empty() {
        return Err(WriteError::EmptyFieldList);
    }
    for field in fields {
        if field.len() != max_doc as usize {
            return Err(WriteError::NotDense {
                values: field.len(),
                max_doc,
            });
        }
    }
    for i in 0..fields.len() {
        for j in (i + 1)..fields.len() {
            if fields[i].field_number() == fields[j].field_number() {
                return Err(WriteError::DuplicateFieldNumber(fields[i].field_number()));
            }
        }
    }
    // Validate multi-valued emptiness up front, before any bytes are written,
    // same "fail before touching the buffers" order the single-field
    // functions use.
    for field in fields {
        if let DenseField::SortedNumeric(_, values) = field {
            for (doc, per_doc) in values.iter().enumerate() {
                if per_doc.is_empty() {
                    return Err(WriteError::EmptyMultiValuedDoc(doc as i32));
                }
            }
        }
        if let DenseField::SortedSet(_, values) = field {
            for (doc, per_doc) in values.iter().enumerate() {
                if per_doc.is_empty() {
                    return Err(WriteError::EmptyMultiValuedDoc(doc as i32));
                }
            }
        }
    }

    let mut meta = new_meta_output(segment_id, segment_suffix);
    let mut data = new_data_output(segment_id, segment_suffix);

    for field in fields {
        let field_number = field.field_number();
        meta.write_i32(field_number);
        match field {
            DenseField::Numeric(_, values) => {
                meta.push(DOC_VALUES_TYPE_NUMERIC);
                write_dense_numeric_entry_body(&mut meta, &mut data, values);
            }
            DenseField::Binary(_, values) => {
                meta.push(DOC_VALUES_TYPE_BINARY);
                write_dense_binary_entry_body(&mut meta, &mut data, values, max_doc);
            }
            DenseField::Sorted(_, values) => {
                meta.push(DOC_VALUES_TYPE_SORTED);
                let (dict, ords) = build_sorted_dict_and_ords(values);
                write_dense_numeric_entry_body(&mut meta, &mut data, &ords);
                write_terms_dict(&mut meta, &mut data, &dict);
            }
            DenseField::SortedNumeric(_, values) => {
                meta.push(DOC_VALUES_TYPE_SORTED_NUMERIC);
                write_dense_sorted_numeric_entry_body(&mut meta, &mut data, values);
            }
            DenseField::SortedSet(_, values) => {
                meta.push(DOC_VALUES_TYPE_SORTED_SET);
                write_dense_sorted_set_entry_body(&mut meta, &mut data, values);
            }
        }
    }

    let skip_index =
        finish_field_list_and_footers(&mut meta, &mut data, segment_id, segment_suffix);
    Ok((meta, data, skip_index))
}

/// The BINARY entry body [`write_single_dense_binary_field`] writes after its
/// leading field-number/type byte -- extracted so [`write_dense_fields`] can
/// share it verbatim.
fn write_dense_binary_entry_body(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    values: &[Vec<u8>],
    max_doc: i32,
) {
    let data_offset = data.len() as i64;
    for v in values {
        data.extend_from_slice(v);
    }
    let data_length = data.len() as i64 - data_offset;
    meta.write_i64(data_offset);
    meta.write_i64(data_length);

    // numDocsWithValue == maxDoc: [-1, 0], no IndexedDISI structure.
    meta.write_i64(DOCS_WITH_FIELD_DENSE);
    meta.write_i64(0);
    meta.write_i16(-1); // jumpTableEntryCount
    meta.push(0xFF); // denseRankPower (-1 as u8)

    meta.write_i32(max_doc);
    let min_length = values.iter().map(|v| v.len() as i32).min().unwrap_or(0);
    let max_length = values.iter().map(|v| v.len() as i32).max().unwrap_or(0);
    meta.write_i32(min_length);
    meta.write_i32(max_length);

    if min_length < max_length {
        let mut end = 0i64;
        let mut ends: Vec<i64> = Vec::with_capacity(values.len() + 1);
        ends.push(0);
        for v in values {
            end += v.len() as i64;
            ends.push(end);
        }
        let block_shift = 0u32;
        let (addr_meta, addr_data) = direct_monotonic::write(&ends, block_shift);
        let addresses_offset = data.len() as i64;
        data.extend_from_slice(&addr_data);
        let addresses_length = data.len() as i64 - addresses_offset;

        meta.write_i64(addresses_offset);
        meta.write_vint(block_shift as i32);
        meta.extend_from_slice(&addr_meta);
        meta.write_i64(addresses_length);
    }
}

/// The SORTED_NUMERIC entry body [`write_single_dense_sorted_numeric_field`]
/// writes after its leading field-number/type byte -- extracted so
/// [`write_dense_fields`] can share it verbatim. Caller must already have
/// checked every `values[doc]` is non-empty.
fn write_dense_sorted_numeric_entry_body(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    values: &[Vec<i64>],
) {
    let num_docs_with_field = values.len() as i32;
    let flat: Vec<i64> = values.iter().flatten().copied().collect();

    write_dense_numeric_entry_body(meta, data, &flat);

    meta.write_i32(num_docs_with_field);
    if num_docs_with_field as i64 != flat.len() as i64 {
        let mut end = 0i64;
        let mut ends: Vec<i64> = Vec::with_capacity(values.len() + 1);
        ends.push(0);
        for per_doc in values {
            end += per_doc.len() as i64;
            ends.push(end);
        }
        let block_shift = 0u32;
        let (addr_meta, addr_data) = direct_monotonic::write(&ends, block_shift);
        let addresses_offset = data.len() as i64;
        data.extend_from_slice(&addr_data);
        let addresses_length = data.len() as i64 - addresses_offset;

        meta.write_i64(addresses_offset);
        meta.write_vint(block_shift as i32);
        meta.extend_from_slice(&addr_meta);
        meta.write_i64(addresses_length);
    }
}

/// The SORTED_SET entry body [`write_single_dense_sorted_set_field`] writes
/// after its leading field-number/type byte -- extracted so
/// [`write_dense_fields`] can share it verbatim. Caller must already have
/// checked every `values[doc]` is non-empty.
fn write_dense_sorted_set_entry_body(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    values: &[Vec<Vec<u8>>],
) {
    let mut dict: Vec<Vec<u8>> = values.iter().flatten().cloned().collect();
    dict.sort_unstable();
    dict.dedup();

    let per_doc_ords: Vec<Vec<i64>> = values
        .iter()
        .map(|per_doc| {
            let mut ords: Vec<i64> = per_doc
                .iter()
                .map(|v| dict.binary_search(v).unwrap() as i64)
                .collect();
            ords.sort_unstable();
            ords.dedup();
            ords
        })
        .collect();

    let num_docs_with_field = per_doc_ords.len() as i32;
    let all_single = per_doc_ords.iter().all(|ords| ords.len() == 1);

    if all_single {
        meta.push(0); // multiValued = false: plain SORTED shape.
        let single_ords: Vec<i64> = per_doc_ords.iter().map(|ords| ords[0]).collect();
        write_dense_numeric_entry_body(meta, data, &single_ords);
    } else {
        meta.push(1); // multiValued = true.
        let flat: Vec<i64> = per_doc_ords.iter().flatten().copied().collect();
        write_dense_numeric_entry_body(meta, data, &flat);

        meta.write_i32(num_docs_with_field);
        if num_docs_with_field as i64 != flat.len() as i64 {
            let mut end = 0i64;
            let mut ends: Vec<i64> = Vec::with_capacity(per_doc_ords.len() + 1);
            ends.push(0);
            for ords in &per_doc_ords {
                end += ords.len() as i64;
                ends.push(end);
            }
            let block_shift = 0u32;
            let (addr_meta, addr_data) = direct_monotonic::write(&ends, block_shift);
            let addresses_offset = data.len() as i64;
            data.extend_from_slice(&addr_data);
            let addresses_length = data.len() as i64 - addresses_offset;

            meta.write_i64(addresses_offset);
            meta.write_vint(block_shift as i32);
            meta.extend_from_slice(&addr_meta);
            meta.write_i64(addresses_length);
        }
    }

    write_terms_dict(meta, data, &dict);
}

/// Writes just the NUMERIC entry body (`addNumericField` -> `writeValues`,
/// the `numDocsWithValue == maxDoc` / no-blocks branches, feeding
/// `writeValuesSingleBlock`) into an already-open meta/data pair -- shared by
/// [`write_single_dense_numeric_field`] (a standalone NUMERIC field) and
/// [`write_single_dense_sorted_numeric_field`] (whose per-doc value counts
/// collapse to this exact same flat layout). Does **not** write the leading
/// `field_number`/type byte -- callers that need those (a bare NUMERIC
/// field) write them first, exactly as [`read_numeric_entry`] expects them
/// already consumed by its caller.
///
/// Always dense (`docsWithFieldOffset = -1`, i.e. `values[i]` is doc `i`'s
/// value / doc `i`'s rank into a shared value array); the per-value encoding
/// itself (plain delta, GCD-delta, or table compression) is chosen by
/// [`write_numeric_values_body`] exactly like real Lucene -- see that
/// function's doc comment. Varying-bits-per-value blocks are still deferred
/// -- see [`write_single_dense_numeric_field`]'s doc comment for the full
/// scope statement, which applies here identically.
fn write_dense_numeric_entry_body(meta: &mut Vec<u8>, data: &mut Vec<u8>, values: &[i64]) {
    // numDocsWithValue == maxDoc: meta[-1, 0], no IndexedDISI structure.
    meta.write_i64(DOCS_WITH_FIELD_DENSE);
    meta.write_i64(0);
    meta.write_i16(-1); // jumpTableEntryCount
    meta.push(0xFF); // denseRankPower (-1 as u8)

    write_numeric_values_body(meta, data, values);
}

/// Writes a **sparse** NUMERIC entry body: an [`indexed_disi`]-backed
/// docs-with-field structure (only `doc_ids` have a value, out of `max_doc`
/// total docs) followed by the same per-rank value encoding
/// [`write_dense_numeric_entry_body`] uses for per-doc values -- a sparse
/// field's value array is indexed by *rank among present docs*, not by doc
/// id, exactly what [`numeric_value`]'s sparse branch already expects
/// (`indexed_disi::rank_of` then [`decode_value`] at that rank).
///
/// `doc_ids` must be strictly ascending and every id `< max_doc` -- the
/// caller's job, same contract [`indexed_disi::write`] documents. For a
/// single-valued caller (plain NUMERIC), `values.len() == doc_ids.len()`
/// (each `values[i]` is `doc_ids[i]`'s value). Multi-valued callers (SORTED_
/// NUMERIC) instead pass every doc's values flattened in doc-id order, so
/// `values.len() >= doc_ids.len()` -- `values` only feeds
/// [`write_numeric_values_body`]'s own record count below, which never
/// assumes it lines up 1:1 with `doc_ids`.
///
/// Always writes `dense_rank_power = 0xFF` (no rank table), matching
/// [`indexed_disi::write`]'s own choice (this port never builds one on the
/// write side; the reader tolerates its absence).
fn write_sparse_numeric_entry_body(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    doc_ids: &[i32],
    values: &[i64],
) {
    let disi_bytes = indexed_disi::write(doc_ids);
    let docs_with_field_offset = data.len() as i64;
    data.extend_from_slice(&disi_bytes);
    let docs_with_field_length = data.len() as i64 - docs_with_field_offset;

    meta.write_i64(docs_with_field_offset);
    meta.write_i64(docs_with_field_length);
    meta.write_i16(-1); // jumpTableEntryCount: no jump table written
    meta.push(0xFF); // denseRankPower: no rank table written

    write_numeric_values_body(meta, data, values);
}

/// Euclidean GCD of two `i64`s (matches `MathUtil.gcd`'s contract: always
/// non-negative, `gcd(0, x) == |x|`). Computed in `i128` so the intermediate
/// `unsigned_abs()` can never overflow even for `i64::MIN`; the result
/// itself always fits back in `i64` since a GCD never exceeds the smaller
/// of its two (absolute) inputs.
fn gcd_i64(a: i64, b: i64) -> i64 {
    let mut a = (a as i128).unsigned_abs();
    let mut b = (b as i128).unsigned_abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a as i64
}

/// Shared tail of a NUMERIC entry body -- everything after the
/// docs-with-field header, common to both the dense and sparse shapes:
/// `numValues`, the constant/table/GCD-delta encoding choice, and the
/// bit-packed (or absent, for the constant case) value array itself.
///
/// Port of `Lucene90DocValuesConsumer.writeValues`'s encoding-choice logic
/// (minus its `doBlocks` varying-bits-per-value split, still deferred -- see
/// `docs/parity.md`): computes a running GCD of every value's difference
/// from the first value (mirrors `MathUtil.gcd` accumulation, including
/// Java's overflow guard: values outside `[i64::MIN/2, i64::MAX/2]` abandon
/// GCD tracking for the rest of the scan rather than risk `v - firstValue`
/// overflowing), and a `<= 256`-entry distinct-value set (abandoned, same as
/// Java's `uniqueValues`, the moment a 257th distinct value appears). Then
/// picks whichever of table-compression (`bitsPerValue =
/// unsignedBitsRequired(uniqueValues.len() - 1)`) or GCD/plain-delta
/// (`bitsPerValue = unsignedBitsRequired((max - min) / gcd)`) needs fewer
/// bits, strictly preferring delta on a tie (`<`, not `<=`, exactly like
/// Java) -- this also means a field whose values are already a dense
/// contiguous ordinal range (e.g. a SORTED field's per-doc dictionary
/// ordinals, which reuse this exact function) never accidentally picks table
/// compression: `uniqueValues.len() - 1 == max - min` there, so the `<`
/// comparison always favors delta, matching `Lucene90DocValuesConsumer`'s own
/// explicit `ords ? null : new LongHashSet()` (ordinals never build a
/// `uniqueValues` set at all; this port gets the same *outcome* without
/// needing that special case, since the two paths tie).
fn write_numeric_values_body(meta: &mut Vec<u8>, data: &mut Vec<u8>, values: &[i64]) {
    let num_values = values.len() as i64;
    meta.write_i64(num_values);

    let min = values.iter().copied().min().unwrap_or(0);
    let max = values.iter().copied().max().unwrap_or(0);

    if min >= max {
        // All values equal (including the empty-values case, which can't
        // actually happen for a standalone NUMERIC field since
        // `values.len() == max_doc` and `max_doc` is a real field count, but
        // can for a SORTED_NUMERIC field's flat value array when every doc
        // has zero values -- Java's `min >= max` check covers both "all
        // equal" and "no values" the same way).
        meta.write_i32(-1); // tableSize
        meta.push(0); // bitsPerValue
        meta.write_i64(min);
        meta.write_i64(1); // gcd
        let start_offset = data.len() as i64;
        meta.write_i64(start_offset);
        meta.write_i64(0); // valuesLength
        meta.write_i64(-1); // valueJumpTableOffset
        return;
    }

    // GCD of every value's offset from the first value seen.
    let first_value = values[0];
    let mut gcd: i64 = 0;
    // Real Lucene's own overflow guard only checks the CURRENT value against
    // `[MIN/2, MAX/2]`, same as below -- but Java's `long` subtraction wraps
    // silently on overflow, while Rust's default arithmetic panics in debug
    // builds. If `first_value` itself falls outside that safe range (e.g.
    // `i64::MIN`) while a later value is in-range, `v - first_value` can
    // still overflow even though the per-`v` guard passed. Using
    // `wrapping_sub` here reproduces Java's exact (silently-wrapping, "gcd
    // may come out numerically odd but never crashes") behavior instead of
    // panicking -- this is a correctness-irrelevant edge case Java itself
    // doesn't handle "correctly" either, just without a hard crash.
    for &v in values {
        if gcd == 1 {
            break;
        }
        if !(i64::MIN / 2..=i64::MAX / 2).contains(&v) {
            gcd = 1;
        } else {
            gcd = gcd_i64(gcd, v.wrapping_sub(first_value));
        }
    }
    if gcd == 0 {
        gcd = 1;
    }

    // Distinct-value set, capped at 256 entries (table compression's max
    // table size), same as Java's `uniqueValues`.
    let mut unique: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
    let mut unique_tracked = true;
    for &v in values {
        if !unique_tracked {
            break;
        }
        unique.insert(v);
        if unique.len() > 256 {
            unique_tracked = false;
        }
    }

    // Same wraparound-not-panic reasoning as the GCD loop above: `max - min`
    // can overflow i64 for a pathological value set (e.g. min == i64::MIN),
    // just like Java's `long` subtraction would -- wrap instead of crash.
    let range_bits = direct_reader::unsigned_bits_required(max.wrapping_sub(min) / gcd);
    let use_table = unique_tracked
        && unique.len() > 1
        && direct_reader::unsigned_bits_required(unique.len() as i64 - 1) < range_bits;

    let (bits_per_value, min, gcd, table): (u8, i64, i64, Option<Vec<i64>>) = if use_table {
        let sorted_unique: Vec<i64> = unique.into_iter().collect();
        let bpv = direct_reader::unsigned_bits_required(sorted_unique.len() as i64 - 1);
        (bpv, 0, 1, Some(sorted_unique))
    } else {
        let mut bpv = range_bits;
        let mut min = min;
        // Java: if gcd==1 && min>0 && bits(max) == bits(max-min), drop the
        // min-shift (store raw values instead) since it doesn't save space.
        if gcd == 1 && min > 0 && direct_reader::unsigned_bits_required(max) == bpv {
            min = 0;
            bpv = direct_reader::unsigned_bits_required(max);
        }
        (bpv, min, gcd, None)
    };

    match &table {
        Some(t) => {
            meta.write_i32(t.len() as i32);
            for &v in t {
                meta.write_i64(v);
            }
        }
        None => meta.write_i32(-1),
    }
    meta.push(bits_per_value);
    meta.write_i64(min);
    meta.write_i64(gcd);

    let start_offset = data.len() as i64;
    meta.write_i64(start_offset);

    if bits_per_value != 0 {
        let raw: Vec<i64> = match &table {
            Some(t) => values
                .iter()
                .map(|v| t.binary_search(v).expect("value must be in its own table") as i64)
                .collect(),
            // `v - min` for `v` in `values` is normally in `[0, max - min]`
            // (min/max are this array's own extrema), but `wrapping_sub`
            // keeps the same non-panicking-on-pathological-input guarantee
            // as `range_bits`'s `max.wrapping_sub(min)` above for the same
            // reason -- both derive from the same possibly-huge min/max.
            None => values.iter().map(|&v| v.wrapping_sub(min) / gcd).collect(),
        };
        let mut packed = direct_reader::encode(&raw, bits_per_value);
        packed.extend(std::iter::repeat_n(
            0u8,
            direct_reader::padding_bytes_needed(bits_per_value),
        ));
        data.extend_from_slice(&packed);
    }

    meta.write_i64(data.len() as i64 - start_offset); // valuesLength
    meta.write_i64(-1); // valueJumpTableOffset: no varying-bpv blocks
}

fn new_meta_output(segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Vec<u8> {
    let mut meta: Vec<u8> = Vec::new();
    codec_util::write_index_header(
        &mut meta,
        META_CODEC,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    meta
}

fn new_data_output(segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Vec<u8> {
    let mut data: Vec<u8> = Vec::new();
    codec_util::write_index_header(
        &mut data,
        DATA_META_CODEC,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    data
}

fn finish_field_list_and_footers(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Vec<u8> {
    meta.write_i32(-1); // field list terminator
    codec_util::write_footer(meta);
    codec_util::write_footer(data);

    let mut skip_index: Vec<u8> = Vec::new();
    codec_util::write_index_header(
        &mut skip_index,
        SKIP_INDEX_META_CODEC,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    codec_util::write_footer(&mut skip_index);
    skip_index
}

/// Port of `Lucene90DocValuesConsumer`, scoped to exactly one shape: **a
/// single NUMERIC field, DENSE** (every doc from `0` to `max_doc - 1` has a
/// value) -- the `numDocsWithValue == maxDoc` branch of `writeValues`,
/// followed by its `doBlocks == false` (no varying-bits-per-value blocks)
/// branch, feeding `writeValuesSingleBlock`. The per-value encoding itself
/// (plain delta, GCD-delta, or table compression) is chosen by
/// [`write_numeric_values_body`] exactly like real Lucene's own
/// `uniqueValues`/`gcd` logic -- see that function's doc comment.
///
/// Sparse NUMERIC fields are handled by the sibling
/// [`write_single_sparse_numeric_field`], not here.
///
/// Deliberately not attempted here, all deferred to future slices (see
/// `docs/parity.md`): the varying-bits-per-value block split, per-field
/// doc-values skip indexes, and multiple fields in one `.dvm`/`.dvd`/`.dvs`
/// triple. BINARY ([`write_single_dense_binary_field`]), SORTED_NUMERIC
/// ([`write_single_dense_sorted_numeric_field`]), SORTED
/// ([`write_single_dense_sorted_field`]), and SORTED_SET
/// ([`write_single_dense_sorted_set_field`]) write sides now exist as
/// siblings of this function, all built on this same dense scope. Sparse
/// BINARY ([`write_single_sparse_binary_field`]), sparse SORTED
/// ([`write_single_sparse_sorted_field`]), sparse SORTED_NUMERIC
/// ([`write_single_sparse_sorted_numeric_field`]), and sparse SORTED_SET
/// ([`write_single_sparse_sorted_set_field`]) writing are also now ported --
/// all 5 doc-values types now support sparse writing.
///
/// Returns `(meta_bytes, data_bytes, skip_index_bytes)` -- three separate
/// buffers matching the real writer's three `IndexOutput`s (`.dvm`, `.dvd`,
/// `.dvs`); `.dvs` is always just a header+footer here since no field in
/// this slice's scope ever has a skip index (`Lucene90DocValuesProducer`
/// unconditionally opens `.dvs` once `VERSION_CURRENT >=
/// VERSION_SKIPPER_SEPARATE_FILE`, which this port's `VERSION_CURRENT`
/// always is, so it must exist and pass header/footer checks even when
/// empty).
pub fn write_single_dense_numeric_field(
    field_number: i32,
    values: &[i64],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    write_dense_fields(
        &[DenseField::Numeric(field_number, values)],
        max_doc,
        segment_id,
        segment_suffix,
    )
}

/// Port of `Lucene90DocValuesConsumer.addNumericField`'s **sparse** branch
/// (`numDocsWithValue != maxDoc`, feeding an [`indexed_disi`]-backed
/// docs-with-field structure instead of the `-1`/DENSE marker
/// [`write_single_dense_numeric_field`] always writes) -- the one doc-values
/// type/shape this port's write side extends beyond dense in this slice; see
/// that function's doc comment for the rest of this module's scope
/// statement (BINARY sparse writing is also now ported, see
/// [`write_single_sparse_binary_field`]; SORTED sparse writing is now ported
/// too, see [`write_single_sparse_sorted_field`]; SORTED_NUMERIC sparse
/// writing is now ported as well, see
/// [`write_single_sparse_sorted_numeric_field`]; SORTED_SET sparse writing is
/// now ported too, see [`write_single_sparse_sorted_set_field`] -- this was
/// the last of the 4 non-NUMERIC/BINARY doc-values types to get sparse
/// support, so all 5 doc-values types now support sparse writing. Still
/// deferred is real Lucene's SPARSE-as-shorts-vs-DENSE-bitset
/// choice for any *single* 65536-doc block -- [`indexed_disi::write`]
/// already makes that choice per block exactly like real Lucene, so a
/// `doc_values` big enough to span more than one block, at varying
/// densities, already exercises all three on-disk block shapes; only the
/// jump table and DENSE rank table are never written, both pure
/// random-access speedups this port's whole-structure decode doesn't need).
///
/// `doc_values` need not be sorted by the caller; this function sorts a
/// clone by doc id itself. Each doc id must be unique and `< max_doc`.
pub fn write_single_sparse_numeric_field(
    field_number: i32,
    doc_values: &[(i32, i64)],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut sorted: Vec<(i32, i64)> = doc_values.to_vec();
    sorted.sort_unstable_by_key(|&(doc, _)| doc);
    for i in 1..sorted.len() {
        if sorted[i - 1].0 == sorted[i].0 {
            return Err(WriteError::DocIdsNotAscending(sorted[i].0));
        }
    }
    for &(doc, _) in &sorted {
        if doc < 0 || doc >= max_doc {
            return Err(WriteError::DocIdOutOfRange(doc, max_doc));
        }
    }

    let doc_ids: Vec<i32> = sorted.iter().map(|&(doc, _)| doc).collect();
    let values: Vec<i64> = sorted.iter().map(|&(_, v)| v).collect();

    let mut meta = new_meta_output(segment_id, segment_suffix);
    let mut data = new_data_output(segment_id, segment_suffix);

    meta.write_i32(field_number);
    meta.push(DOC_VALUES_TYPE_NUMERIC);
    write_sparse_numeric_entry_body(&mut meta, &mut data, &doc_ids, &values);

    let skip_index =
        finish_field_list_and_footers(&mut meta, &mut data, segment_id, segment_suffix);
    Ok((meta, data, skip_index))
}

/// Port of `Lucene90DocValuesConsumer.addBinaryField`, scoped to exactly one
/// shape: **a single BINARY field, DENSE** (every doc from `0` to `max_doc -
/// 1` has a value, one `Vec<u8>` per doc in `values`, empty slices allowed --
/// an empty value is still a present value, distinct from "no value").
/// Handles both length shapes real Lucene distinguishes: **fixed-length**
/// (every value the same length -- no address array, `ordinal * length`
/// indexing, matching [`BinaryEntry::is_fixed_length`]) and
/// **variable-length** (a [`crate::direct_monotonic`] end-offset array,
/// [`write`](direct_monotonic::write) with `block_shift = 0` -- the same
/// choice `term_vectors.rs`/`stored_fields.rs`'s own monotonic writers
/// already made, simplicity over compression ratio for an in-memory buffer).
///
/// Deliberately not attempted here: per-field doc-values skip indexes, and
/// multiple fields in one `.dvm`/`.dvd`/`.dvs` triple. Sparse BINARY fields
/// (`IndexedDISI`) are no longer out of scope -- see
/// [`write_single_sparse_binary_field`] below.
///
/// Returns `(meta_bytes, data_bytes, skip_index_bytes)`, same shape as
/// [`write_single_dense_numeric_field`].
pub fn write_single_dense_binary_field(
    field_number: i32,
    values: &[Vec<u8>],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    write_dense_fields(
        &[DenseField::Binary(field_number, values)],
        max_doc,
        segment_id,
        segment_suffix,
    )
}

/// Port of `Lucene90DocValuesConsumer.addBinaryField`'s **sparse** branch
/// (`numDocsWithValue != maxDoc`), the BINARY analogue of
/// [`write_single_sparse_numeric_field`]: an [`indexed_disi`]-backed
/// docs-with-field structure instead of the `-1`/DENSE marker
/// [`write_single_dense_binary_field`] always writes, followed by the value
/// bytes for *only* the docs that have one -- no placeholder bytes for
/// missing docs, matching real Lucene exactly (a sparse field's value array
/// is indexed by rank among present docs, not by doc id, same as sparse
/// NUMERIC's value array).
///
/// `doc_values` need not be sorted by the caller; this function sorts a
/// clone by doc id itself, same contract as
/// [`write_single_sparse_numeric_field`]. Each doc id must be unique and
/// `< max_doc`. Empty `Vec<u8>` values are allowed (an empty value is still
/// present, distinct from absent).
///
/// Handles both length shapes real Lucene distinguishes, same as
/// [`write_single_dense_binary_field`]: fixed-length (no address array,
/// `rank * length` indexing) and variable-length ([`direct_monotonic`]
/// end-offset array over the present docs' values, in rank order).
pub fn write_single_sparse_binary_field(
    field_number: i32,
    doc_values: &[(i32, Vec<u8>)],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut sorted: Vec<(i32, Vec<u8>)> = doc_values.to_vec();
    sorted.sort_unstable_by_key(|&(doc, _)| doc);
    for i in 1..sorted.len() {
        if sorted[i - 1].0 == sorted[i].0 {
            return Err(WriteError::DocIdsNotAscending(sorted[i].0));
        }
    }
    for &(doc, _) in &sorted {
        if doc < 0 || doc >= max_doc {
            return Err(WriteError::DocIdOutOfRange(doc, max_doc));
        }
    }

    let doc_ids: Vec<i32> = sorted.iter().map(|&(doc, _)| doc).collect();
    let values: Vec<Vec<u8>> = sorted.into_iter().map(|(_, v)| v).collect();

    let mut meta = new_meta_output(segment_id, segment_suffix);
    let mut data = new_data_output(segment_id, segment_suffix);

    meta.write_i32(field_number);
    meta.push(DOC_VALUES_TYPE_BINARY);

    let data_offset = data.len() as i64;
    for v in &values {
        data.extend_from_slice(v);
    }
    let data_length = data.len() as i64 - data_offset;
    meta.write_i64(data_offset);
    meta.write_i64(data_length);

    let disi_bytes = indexed_disi::write(&doc_ids);
    let docs_with_field_offset = data.len() as i64;
    data.extend_from_slice(&disi_bytes);
    let docs_with_field_length = data.len() as i64 - docs_with_field_offset;

    meta.write_i64(docs_with_field_offset);
    meta.write_i64(docs_with_field_length);
    meta.write_i16(-1); // jumpTableEntryCount: no jump table written
    meta.push(0xFF); // denseRankPower: no rank table written

    meta.write_i32(values.len() as i32); // numDocsWithField
    let min_length = values.iter().map(|v| v.len() as i32).min().unwrap_or(0);
    let max_length = values.iter().map(|v| v.len() as i32).max().unwrap_or(0);
    meta.write_i32(min_length);
    meta.write_i32(max_length);

    if min_length < max_length {
        let mut end = 0i64;
        let mut ends: Vec<i64> = Vec::with_capacity(values.len() + 1);
        ends.push(0);
        for v in &values {
            end += v.len() as i64;
            ends.push(end);
        }
        let block_shift = 0u32;
        let (addr_meta, addr_data) = direct_monotonic::write(&ends, block_shift);
        let addresses_offset = data.len() as i64;
        data.extend_from_slice(&addr_data);
        let addresses_length = data.len() as i64 - addresses_offset;

        meta.write_i64(addresses_offset);
        meta.write_vint(block_shift as i32);
        meta.extend_from_slice(&addr_meta);
        meta.write_i64(addresses_length);
    }

    let skip_index =
        finish_field_list_and_footers(&mut meta, &mut data, segment_id, segment_suffix);
    Ok((meta, data, skip_index))
}

/// Port of `Lucene90DocValuesConsumer.addSortedNumericField`, scoped to
/// exactly one shape: **a single SORTED_NUMERIC field, DENSE** (every doc
/// from `0` to `max_doc - 1` has **at least one** value -- `values[doc]`
/// must be non-empty, [`WriteError::EmptyMultiValuedDoc`] otherwise; zero
/// values for a doc would need `IndexedDISI`, out of scope here same as
/// every other sparse case in this module). Flattens `values` into one flat
/// array (Java's shared value array across all docs) and writes the address
/// range per doc via [`direct_monotonic::write`] -- **except** when every doc
/// has exactly one value, in which case real Lucene's `readSortedNumeric`
/// collapses the address array away entirely (a doc's rank already *is* its
/// value's index), so this function detects that case and omits the address
/// array too, to stay byte-compatible with what the read side actually
/// expects (`num_docs_with_field == numeric.num_values` is not a stored
/// flag -- the read side infers "no addresses" from that equality, so
/// writing one when it holds would desync the two sides).
///
/// Deliberately not attempted here, same as [`write_single_dense_numeric_field`]:
/// sparse per-doc value *presence* (every doc must have >= 1 value),
/// per-field doc-values skip indexes, and multiple fields in one
/// `.dvm`/`.dvd`/`.dvs` triple.
///
/// Returns `(meta_bytes, data_bytes, skip_index_bytes)`, same shape as
/// [`write_single_dense_numeric_field`].
pub fn write_single_dense_sorted_numeric_field(
    field_number: i32,
    values: &[Vec<i64>],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let max_doc = values.len() as i32;
    write_dense_fields(
        &[DenseField::SortedNumeric(field_number, values)],
        max_doc,
        segment_id,
        segment_suffix,
    )
}

/// Port of `Lucene90DocValuesConsumer.addSortedNumericField`'s **sparse**
/// branch (`numDocsWithField != maxDoc`), the SORTED_NUMERIC analogue of
/// [`write_single_sparse_numeric_field`]. Real Lucene's sparse test here is
/// specifically "does this doc have *zero* values", not "does it have
/// exactly one" -- a doc with 3 values is just as absent from the
/// docs-with-field structure's complement as a doc with 1, and a doc with 0
/// values gets no entry anywhere (no per-doc value-count slot, no address
/// range) and reads back as an empty `Vec` via the same
/// [`indexed_disi::rank_of`]-returns-`None` path
/// [`write_single_sparse_numeric_field`]'s dense-vs-sparse NUMERIC entry
/// already handles.
///
/// This reuses [`write_sparse_numeric_entry_body`] directly rather than
/// reinventing an IndexedDISI dance: that function's `doc_ids`/`values`
/// contract ("`docs_with_field`'s presence bitset over `doc_ids`, then
/// whatever flat value array `values` is") doesn't actually require
/// `values.len() == doc_ids.len()` -- that equality only happens to hold for
/// a bare sparse NUMERIC field (one value per present doc). Here `doc_ids` is
/// every doc with **at least one** value, and `values` is those docs' values
/// flattened in doc order, so `values.len() >= doc_ids.len()`, exactly
/// mirroring how [`write_single_dense_sorted_numeric_field`] feeds its own
/// (dense) `numeric` sub-entry. The `num_docs_with_field`/address-array tail
/// that follows is the same collapse rule that function uses: an address
/// array (via [`direct_monotonic`]) unless every present doc has exactly one
/// value, in which case the read side infers "no addresses" from
/// `num_docs_with_field == numeric.num_values` and this function omits the
/// array to match.
///
/// `doc_values` need not be sorted by the caller; this function sorts a
/// clone by doc id itself, same contract as
/// [`write_single_sparse_numeric_field`]. Each doc id must be unique and
/// `< max_doc`. Every doc in `doc_values` must have a non-empty value list
/// ([`WriteError::EmptyMultiValuedDoc`] otherwise) -- a doc with zero values
/// is represented by *omitting* it from `doc_values` entirely, not by
/// passing an empty `Vec` for it, since "present with zero values" isn't a
/// distinct on-disk state from "absent" in this format.
pub fn write_single_sparse_sorted_numeric_field(
    field_number: i32,
    doc_values: &[(i32, Vec<i64>)],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut sorted: Vec<(i32, Vec<i64>)> = doc_values.to_vec();
    sorted.sort_unstable_by_key(|(doc, _)| *doc);
    for i in 1..sorted.len() {
        if sorted[i - 1].0 == sorted[i].0 {
            return Err(WriteError::DocIdsNotAscending(sorted[i].0));
        }
    }
    for (doc, _) in &sorted {
        if *doc < 0 || *doc >= max_doc {
            return Err(WriteError::DocIdOutOfRange(*doc, max_doc));
        }
    }
    for (doc, per_doc) in &sorted {
        if per_doc.is_empty() {
            return Err(WriteError::EmptyMultiValuedDoc(*doc));
        }
    }

    let doc_ids: Vec<i32> = sorted.iter().map(|(doc, _)| *doc).collect();
    let num_docs_with_field = sorted.len() as i32;
    let flat: Vec<i64> = sorted.iter().flat_map(|(_, v)| v.iter().copied()).collect();

    let mut meta = new_meta_output(segment_id, segment_suffix);
    let mut data = new_data_output(segment_id, segment_suffix);

    meta.write_i32(field_number);
    meta.push(DOC_VALUES_TYPE_SORTED_NUMERIC);
    write_sparse_numeric_entry_body(&mut meta, &mut data, &doc_ids, &flat);

    meta.write_i32(num_docs_with_field);
    if num_docs_with_field as i64 != flat.len() as i64 {
        let mut end = 0i64;
        let mut ends: Vec<i64> = Vec::with_capacity(sorted.len() + 1);
        ends.push(0);
        for (_, per_doc) in &sorted {
            end += per_doc.len() as i64;
            ends.push(end);
        }
        let block_shift = 0u32;
        let (addr_meta, addr_data) = direct_monotonic::write(&ends, block_shift);
        let addresses_offset = data.len() as i64;
        data.extend_from_slice(&addr_data);
        let addresses_length = data.len() as i64 - addresses_offset;

        meta.write_i64(addresses_offset);
        meta.write_vint(block_shift as i32);
        meta.extend_from_slice(&addr_meta);
        meta.write_i64(addresses_length);
    }

    let skip_index =
        finish_field_list_and_footers(&mut meta, &mut data, segment_id, segment_suffix);
    Ok((meta, data, skip_index))
}

/// Longest common byte prefix of `a` and `b` (`StringHelper.bytesDifference`,
/// scoped to the "always in order, never equal" case terms-dict callers
/// guarantee -- real Lucene's version also doubles as an out-of-order
/// assertion, not needed here since [`write_terms_dict`]'s caller already
/// sorts+dedups).
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Port of `Lucene90DocValuesConsumer.addTermsDict` (block-LZ4-compressed
/// terms + block-address array) followed by `writeTermsIndex` (a coarser
/// sampled reverse index, sampled every
/// [`TERMS_DICT_REVERSE_INDEX_MASK`]`+1`th ordinal) -- the machinery
/// `SORTED`'s single-valued dictionary and `SORTED_SET`'s multi-valued one
/// share verbatim in real Lucene (`addSortedField`/`addSortedSetField` both
/// end by calling this same method). `terms` must already be sorted
/// ascending with no duplicates (the caller's job -- this port always builds
/// it from a `BTreeSet`/`sort_unstable+dedup` over the field's raw values).
///
/// Writes directly into `meta`/`data`, mirroring
/// [`write_dense_numeric_entry_body`]'s "body only, no leading
/// field-number/type byte" shape -- appends exactly what
/// [`terms_dict::read_term_dict_entry`] expects to read right after a
/// SORTED/SORTED_SET field's ords entry.
fn write_terms_dict(meta: &mut Vec<u8>, data: &mut Vec<u8>, terms: &[Vec<u8>]) {
    let size = terms.len() as i64;
    meta.write_vlong(size);
    meta.write_i32(TERMS_DICT_DIRECT_MONOTONIC_BLOCK_SHIFT as i32);

    let start = data.len() as i64;
    let mut max_length = 0i32;
    let mut max_block_length = 0i32;
    let mut block_addresses: Vec<i64> = Vec::new();
    let mut block_body: Vec<u8> = Vec::new();

    let mut ord = 0usize;
    while ord < terms.len() {
        let term = &terms[ord];
        block_addresses.push(data.len() as i64 - start);
        data.write_vint(term.len() as i32);
        data.extend_from_slice(term);
        max_length = max_length.max(term.len() as i32);

        // The rest of this 64-term block, prefix-compressed against its
        // immediate predecessor (first term included as the implicit
        // dictionary).
        block_body.clear();
        let block_end = terms
            .len()
            .min(ord + (TERMS_DICT_BLOCK_LZ4_MASK as usize) + 1);
        let mut prev: &[u8] = term;
        for later in &terms[ord + 1..block_end] {
            let prefix_len = common_prefix_len(prev, later);
            let suffix_len = later.len() - prefix_len;
            block_body.push((prefix_len.min(15) as u8) | (((suffix_len - 1).min(15) as u8) << 4));
            if prefix_len >= 15 {
                block_body.write_vint((prefix_len - 15) as i32);
            }
            if suffix_len >= 16 {
                block_body.write_vint((suffix_len - 16) as i32);
            }
            block_body.extend_from_slice(&later[prefix_len..]);
            max_length = max_length.max(later.len() as i32);
            prev = later;
        }

        if !block_body.is_empty() {
            // See [`write_single_dense_sorted_field`]'s doc comment: this
            // port's `lz4::compress` has no preset-dictionary variant, but
            // the decompressor tolerates that (it only ever *allows* matches
            // into the dictionary region, never requires them), so
            // compressing `block_body` on its own still round-trips.
            let compressed = lz4::compress(&block_body);
            data.write_vint(block_body.len() as i32);
            data.extend_from_slice(&compressed);
        }
        max_block_length = max_block_length.max(block_body.len() as i32);
        ord = block_end;
    }

    // `DirectMonotonicWriter` flushes each completed block's meta entry as
    // soon as it fills (`DirectMonotonicWriter.flush`), i.e. interleaved
    // with the term loop above and always finished before `maxLength` is
    // written -- see `read_term_dict_entry`'s read order, which expects this
    // meta array right after `block_shift`, before `max_term_length`.
    let (addr_meta, addr_data) =
        direct_monotonic::write(&block_addresses, TERMS_DICT_DIRECT_MONOTONIC_BLOCK_SHIFT);
    meta.extend_from_slice(&addr_meta);

    meta.write_i32(max_length);
    meta.write_i32(max_block_length);
    meta.write_i64(start);
    meta.write_i64(data.len() as i64 - start);

    let addresses_start = data.len() as i64;
    data.extend_from_slice(&addr_data);
    meta.write_i64(addresses_start);
    meta.write_i64(data.len() as i64 - addresses_start);

    // `writeTermsIndex`: a coarser reverse index, one sampled address + a
    // short disambiguating prefix every `TERMS_DICT_REVERSE_INDEX_MASK + 1`
    // ordinals.
    meta.write_i32(TERMS_DICT_REVERSE_INDEX_SHIFT as i32);
    let index_start = data.len() as i64;
    let mut offset = 0i64;
    let mut index_addresses: Vec<i64> = Vec::new();
    let mut sampled_previous: Vec<u8> = Vec::new();
    for (ord, term) in terms.iter().enumerate() {
        let ord = ord as i64;
        if ord & TERMS_DICT_REVERSE_INDEX_MASK == 0 {
            index_addresses.push(offset);
            let sort_key_len = if ord == 0 {
                0
            } else {
                common_prefix_len(&sampled_previous, term) + 1
            };
            offset += sort_key_len as i64;
            data.extend_from_slice(&term[..sort_key_len]);
        } else if ord & TERMS_DICT_REVERSE_INDEX_MASK == TERMS_DICT_REVERSE_INDEX_MASK {
            sampled_previous.clear();
            sampled_previous.extend_from_slice(term);
        }
    }
    index_addresses.push(offset);

    // Same interleaving note as the block-address writer above: its meta
    // entries are already fully flushed by this point, so they can be
    // written now, before `indexOffset`/`indexLength`.
    let (index_addr_meta, index_addr_data) =
        direct_monotonic::write(&index_addresses, TERMS_DICT_DIRECT_MONOTONIC_BLOCK_SHIFT);
    meta.extend_from_slice(&index_addr_meta);

    meta.write_i64(index_start);
    meta.write_i64(data.len() as i64 - index_start);

    let index_addresses_start = data.len() as i64;
    data.extend_from_slice(&index_addr_data);
    meta.write_i64(index_addresses_start);
    meta.write_i64(data.len() as i64 - index_addresses_start);
}

/// Port of `Lucene90DocValuesConsumer.addSortedField`, scoped to exactly one
/// shape: **a single SORTED field, DENSE** (every doc from `0` to `max_doc -
/// 1` has a value -- `values[doc]` is that doc's raw term bytes; an empty
/// slice is a legitimate present value, same convention
/// [`write_single_dense_binary_field`] uses). Builds the sorted, deduplicated
/// distinct-value dictionary (`BTreeSet` over `values`), maps each doc to its
/// term's ordinal into that dictionary (written as a plain dense NUMERIC
/// entry via [`write_dense_numeric_entry_body`] -- exactly how real Lucene's
/// `doAddSortedField` always wraps a `SortedDocValues` as a singleton
/// `SortedNumericDocValues`, which collapses back to a bare per-doc ordinal
/// with no address array since it's single-valued), then the dictionary
/// itself via [`write_terms_dict`] (shared, byte-for-byte, with what a future
/// SORTED_SET writer would call for the same values).
///
/// Sparse SORTED fields (`IndexedDISI`, i.e. some docs with no value at all)
/// are handled by the sibling [`write_single_sparse_sorted_field`], not here.
///
/// Deliberately not attempted here, same as
/// [`write_single_dense_numeric_field`]: per-field doc-values skip indexes,
/// and multiple fields in one `.dvm`/`.dvd`/`.dvs` triple.
///
/// Returns `(meta_bytes, data_bytes, skip_index_bytes)`, same shape as
/// [`write_single_dense_numeric_field`].
pub fn write_single_dense_sorted_field(
    field_number: i32,
    values: &[Vec<u8>],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    write_dense_fields(
        &[DenseField::Sorted(field_number, values)],
        max_doc,
        segment_id,
        segment_suffix,
    )
}

/// Builds the sorted, deduplicated distinct-value dictionary for a SORTED
/// field's raw per-doc term bytes (a `BTreeSet`-equivalent `sort_unstable` +
/// `dedup` over `values`) and maps each entry of `values` to its ordinal into
/// that dictionary. Shared by [`write_single_dense_sorted_field`] and
/// [`write_single_sparse_sorted_field`] -- in the sparse case, `values` holds
/// only the present docs' terms, in doc-id order, so the returned `ords`
/// align with those docs' *ranks*, exactly what
/// [`write_sparse_numeric_entry_body`] expects.
fn build_sorted_dict_and_ords(values: &[Vec<u8>]) -> (Vec<Vec<u8>>, Vec<i64>) {
    let mut dict: Vec<Vec<u8>> = values.to_vec();
    dict.sort_unstable();
    dict.dedup();

    let ords: Vec<i64> = values
        .iter()
        .map(|v| dict.binary_search(v).unwrap() as i64)
        .collect();
    (dict, ords)
}

/// Port of `Lucene90DocValuesConsumer.addSortedField`'s **sparse** branch
/// (`numDocsWithValue != maxDoc`), the SORTED analogue of
/// [`write_single_sparse_numeric_field`]/[`write_single_sparse_binary_field`]:
/// an [`indexed_disi`]-backed docs-with-field structure instead of the
/// `-1`/DENSE marker [`write_single_dense_sorted_field`] always writes,
/// followed by per-doc ordinals for *only* the docs that have a value --
/// missing docs get no ordinal at all, not even a placeholder, matching
/// [`write_sparse_numeric_entry_body`]'s rank-indexed (not doc-id-indexed)
/// value array. The terms dictionary itself
/// ([`build_sorted_dict_and_ords`]/[`write_terms_dict`]) is built only from
/// the present docs' values, same as real Lucene's `addSortedField`, which
/// only ever sees values for docs `DocIdSetIterator` actually advances to.
///
/// `doc_values` need not be sorted by the caller; this function sorts a
/// clone by doc id itself, same contract as
/// [`write_single_sparse_numeric_field`]. Each doc id must be unique and
/// `< max_doc`. Empty `Vec<u8>` values are allowed (an empty value is still
/// present, distinct from absent).
pub fn write_single_sparse_sorted_field(
    field_number: i32,
    doc_values: &[(i32, Vec<u8>)],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut sorted: Vec<(i32, Vec<u8>)> = doc_values.to_vec();
    sorted.sort_unstable_by_key(|&(doc, _)| doc);
    for i in 1..sorted.len() {
        if sorted[i - 1].0 == sorted[i].0 {
            return Err(WriteError::DocIdsNotAscending(sorted[i].0));
        }
    }
    for &(doc, _) in &sorted {
        if doc < 0 || doc >= max_doc {
            return Err(WriteError::DocIdOutOfRange(doc, max_doc));
        }
    }

    let doc_ids: Vec<i32> = sorted.iter().map(|&(doc, _)| doc).collect();
    let values: Vec<Vec<u8>> = sorted.into_iter().map(|(_, v)| v).collect();

    let (dict, ords) = build_sorted_dict_and_ords(&values);

    let mut meta = new_meta_output(segment_id, segment_suffix);
    let mut data = new_data_output(segment_id, segment_suffix);

    meta.write_i32(field_number);
    meta.push(DOC_VALUES_TYPE_SORTED);
    write_sparse_numeric_entry_body(&mut meta, &mut data, &doc_ids, &ords);
    write_terms_dict(&mut meta, &mut data, &dict);

    let skip_index =
        finish_field_list_and_footers(&mut meta, &mut data, segment_id, segment_suffix);
    Ok((meta, data, skip_index))
}

/// Port of `Lucene90DocValuesConsumer.addSortedSetField`, scoped to exactly
/// one shape: **a single SORTED_SET field, DENSE** (every doc from `0` to
/// `max_doc - 1` has **at least one** value -- `values[doc]` must be
/// non-empty, [`WriteError::EmptyMultiValuedDoc`] otherwise, same
/// "every doc must have a value" contract as
/// [`write_single_dense_sorted_numeric_field`]). Builds the sorted,
/// deduplicated distinct-value dictionary across *every* value of *every*
/// doc (not one value per doc -- a doc's own value set can itself repeat a
/// value, which is deduped away per-doc too, since a sorted set never
/// stores the same ordinal twice for one doc), then writes the per-doc
/// ordinals following the exact same collapse rule
/// [`write_single_dense_sorted_numeric_field`] uses: when every doc has
/// exactly one distinct value, this is written as a plain [`SortedEntry`]
/// (`multiValued = 0`, per-doc ordinal via [`write_dense_numeric_entry_body`],
/// no address array) exactly like [`write_single_dense_sorted_field`] would;
/// otherwise as a true multi-valued form (`multiValued = 1`, flattened
/// ordinals via [`write_dense_numeric_entry_body`] plus a
/// [`direct_monotonic`] address-range array). This matches
/// [`read_sorted_set_entry`]'s own inference exactly -- it decides
/// single-vs-multi purely from the stored `multiValued` flag byte (unlike
/// [`write_single_dense_sorted_numeric_field`]'s address array, whose
/// presence the read side infers from a count equality rather than a flag),
/// and its `Multi` branch in turn infers the address array's own presence
/// from `num_docs_with_field == numeric.num_values` -- which always holds
/// exactly when every doc has one value, i.e. never in the `multiValued = 1`
/// branch this function takes (docs must have >= 1 value each, so any doc
/// with more than one forces the total above `num_docs_with_field`).
/// The dictionary itself is written via [`write_terms_dict`] (shared,
/// byte-for-byte, with [`write_single_dense_sorted_field`]).
///
/// Sparse SORTED_SET fields (some docs with zero values at all) are handled
/// by the sibling [`write_single_sparse_sorted_set_field`], not here.
///
/// Deliberately not attempted here, same as [`write_single_dense_numeric_field`]:
/// per-field doc-values skip indexes, and multiple fields in one
/// `.dvm`/`.dvd`/`.dvs` triple.
///
/// Returns `(meta_bytes, data_bytes, skip_index_bytes)`, same shape as
/// [`write_single_dense_numeric_field`].
pub fn write_single_dense_sorted_set_field(
    field_number: i32,
    values: &[Vec<Vec<u8>>],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    write_dense_fields(
        &[DenseField::SortedSet(field_number, values)],
        max_doc,
        segment_id,
        segment_suffix,
    )
}

/// Port of `Lucene90DocValuesConsumer.addSortedSetField`'s **sparse** branch
/// (some docs have zero values at all), the SORTED_SET analogue of
/// [`write_single_sparse_sorted_numeric_field`]: an [`indexed_disi`]-backed
/// docs-with->=1-ordinal structure instead of the "every doc has a value"
/// contract [`write_single_dense_sorted_set_field`] requires, always taking
/// the `multiValued = 1` shape (this function never collapses to the
/// single-valued plain-`SortedEntry` shape
/// [`write_single_dense_sorted_set_field`] takes when every doc has exactly
/// one distinct value -- [`SortedSetKind::Multi`] decodes correctly
/// regardless of whether every present doc happens to have exactly one
/// ordinal, so that extra collapse is a size optimization real Lucene applies
/// but this port's read side doesn't require).
///
/// A doc's zero-or-more raw values are deduplicated *and* sorted into
/// ordinals, same as [`write_single_dense_sorted_set_field`] (a sorted set
/// never stores the same ordinal twice for one doc, and iterates its ordinals
/// in order) -- unlike SORTED_NUMERIC, whose values keep whatever
/// order/duplicates the caller passed. The terms dictionary is built only
/// from the present docs' values, same rationale as
/// [`write_single_sparse_sorted_field`]: real Lucene's `addSortedSetField`
/// only ever sees values for docs its iterator actually advances to. Per-doc
/// ordinal counts/addresses are written only for present docs, in rank order,
/// same [`write_sparse_numeric_entry_body`]-then-address-array structure
/// [`write_single_sparse_sorted_numeric_field`] uses for its own flattened
/// values.
///
/// `doc_values` need not be sorted by the caller; this function sorts a
/// clone by doc id itself, same contract as
/// [`write_single_sparse_sorted_numeric_field`]. Each doc id must be unique
/// and `< max_doc`. Every doc in `doc_values` must have a non-empty value set
/// ([`WriteError::EmptyMultiValuedDoc`] otherwise) -- a doc with zero values
/// is represented by *omitting* it from `doc_values` entirely, not by passing
/// an empty `Vec` for it, same "absent, not present-with-zero" contract as
/// [`write_single_sparse_sorted_numeric_field`].
pub fn write_single_sparse_sorted_set_field(
    field_number: i32,
    doc_values: &[(i32, Vec<Vec<u8>>)],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut sorted: Vec<(i32, Vec<Vec<u8>>)> = doc_values.to_vec();
    sorted.sort_unstable_by_key(|(doc, _)| *doc);
    for i in 1..sorted.len() {
        if sorted[i - 1].0 == sorted[i].0 {
            return Err(WriteError::DocIdsNotAscending(sorted[i].0));
        }
    }
    for (doc, _) in &sorted {
        if *doc < 0 || *doc >= max_doc {
            return Err(WriteError::DocIdOutOfRange(*doc, max_doc));
        }
    }
    for (doc, per_doc) in &sorted {
        if per_doc.is_empty() {
            return Err(WriteError::EmptyMultiValuedDoc(*doc));
        }
    }

    let doc_ids: Vec<i32> = sorted.iter().map(|(doc, _)| *doc).collect();

    let mut dict: Vec<Vec<u8>> = sorted.iter().flat_map(|(_, v)| v.iter().cloned()).collect();
    dict.sort_unstable();
    dict.dedup();

    let per_doc_ords: Vec<Vec<i64>> = sorted
        .iter()
        .map(|(_, per_doc)| {
            let mut ords: Vec<i64> = per_doc
                .iter()
                .map(|v| dict.binary_search(v).unwrap() as i64)
                .collect();
            ords.sort_unstable();
            ords.dedup();
            ords
        })
        .collect();

    let num_docs_with_field = per_doc_ords.len() as i32;
    let flat: Vec<i64> = per_doc_ords.iter().flatten().copied().collect();

    let mut meta = new_meta_output(segment_id, segment_suffix);
    let mut data = new_data_output(segment_id, segment_suffix);

    meta.write_i32(field_number);
    meta.push(DOC_VALUES_TYPE_SORTED_SET);
    meta.push(1); // multiValued = true.

    write_sparse_numeric_entry_body(&mut meta, &mut data, &doc_ids, &flat);

    meta.write_i32(num_docs_with_field);
    if num_docs_with_field as i64 != flat.len() as i64 {
        let mut end = 0i64;
        let mut ends: Vec<i64> = Vec::with_capacity(per_doc_ords.len() + 1);
        ends.push(0);
        for ords in &per_doc_ords {
            end += ords.len() as i64;
            ends.push(end);
        }
        let block_shift = 0u32;
        let (addr_meta, addr_data) = direct_monotonic::write(&ends, block_shift);
        let addresses_offset = data.len() as i64;
        data.extend_from_slice(&addr_data);
        let addresses_length = data.len() as i64 - addresses_offset;

        meta.write_i64(addresses_offset);
        meta.write_vint(block_shift as i32);
        meta.extend_from_slice(&addr_meta);
        meta.write_i64(addresses_length);
    }

    write_terms_dict(&mut meta, &mut data, &dict);

    let skip_index =
        finish_field_list_and_footers(&mut meta, &mut data, segment_id, segment_suffix);
    Ok((meta, data, skip_index))
}

#[cfg(test)]
mod tests {}
