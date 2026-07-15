//! Port of `org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat`
//! (`.dvm` metadata + `.dvd` data) — read-only. All five doc-values types
//! are supported (NUMERIC, BINARY, SORTED, SORTED_SET, SORTED_NUMERIC);
//! per-field doc-values skip indexes are not (see
//! [`Error::UnsupportedSkipIndex`]).
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
//! Also out of scope for now: "varying bits-per-value" blocks (Java splits
//! a field's values into 16384-value blocks with independently chosen
//! bits-per-value when that's a better fit — `blockShift >= 0` in the meta).
//! Small-to-medium fields Lucene doesn't bother splitting; reading one that
//! was split returns [`Error::UnsupportedVaryingBpvBlocks`].
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

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("unknown field number: {0}")]
    UnknownFieldNumber(i32),
    #[error("field {0} has unknown doc values type byte {1}")]
    UnsupportedFieldType(i32, u8),
    #[error("field {0} has a doc-values skip index, which this port doesn't parse")]
    UnsupportedSkipIndex(i32),
    #[error("field {0} splits values into varying-bits-per-value blocks, which this port doesn't decode")]
    UnsupportedVaryingBpvBlocks(i32),
    #[error("invalid table size: {0}")]
    InvalidTableSize(i32),
    #[error("invalid bitsPerValue: {0}, field number {1}")]
    InvalidBitsPerValue(u8, i32),
    #[error("doc {0} is out of range (numValues={1})")]
    DocOutOfRange(i32, i64),
    #[error("field {0} has invalid multiValued flag: {1}")]
    InvalidMultiValuedFlag(i32, u8),
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

#[derive(Debug, Clone, Default)]
pub struct DocValuesMeta {
    pub numeric: Vec<NumericEntry>,
    pub binary: Vec<BinaryEntry>,
    pub sorted: Vec<SortedEntry>,
    pub sorted_numeric: Vec<SortedNumericEntry>,
    pub sorted_set: Vec<SortedSetEntry>,
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
            return Err(Error::UnsupportedSkipIndex(field_number));
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
    let table = if table_size >= 0 {
        let mut t = Vec::with_capacity(table_size as usize);
        for _ in 0..table_size {
            t.push(input.read_i64()?);
        }
        Some(t)
    } else {
        None
    };
    if table_size < -1 {
        // Varying-bits-per-value blocks: `blockShift = -2 - tableSize`.
        return Err(Error::UnsupportedVaryingBpvBlocks(field_number));
    }

    let bits_per_value = input.read_byte()?;
    if !matches!(
        bits_per_value,
        0 | 1 | 2 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 40 | 48 | 56 | 64
    ) {
        return Err(Error::InvalidBitsPerValue(bits_per_value, field_number));
    }
    let min_value = input.read_i64()?;
    let gcd = input.read_i64()?;
    let values_offset = input.read_i64()?;
    let values_length = input.read_i64()?;
    let _value_jump_table_offset = input.read_i64()?; // only meaningful for varying-bpv blocks

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
}

pub type WriteResult<T> = std::result::Result<T, WriteError>;

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
/// `doc_ids` must be strictly ascending, every id `< max_doc`, and
/// `values.len() == doc_ids.len()` (each `values[i]` is `doc_ids[i]`'s
/// value) -- the caller's job, same contract [`indexed_disi::write`]
/// documents.
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
/// `docs/parity.md`): sparse BINARY/SORTED_NUMERIC/SORTED_SET fields, the
/// varying-bits-per-value block split, per-field doc-values skip indexes,
/// and multiple fields in one `.dvm`/`.dvd`/`.dvs` triple. BINARY
/// ([`write_single_dense_binary_field`]), SORTED_NUMERIC
/// ([`write_single_dense_sorted_numeric_field`]), SORTED
/// ([`write_single_dense_sorted_field`]), and SORTED_SET
/// ([`write_single_dense_sorted_set_field`]) write sides now exist as
/// siblings of this function, all built on this same dense scope.
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
    if values.len() != max_doc as usize {
        return Err(WriteError::NotDense {
            values: values.len(),
            max_doc,
        });
    }

    let mut meta = new_meta_output(segment_id, segment_suffix);
    let mut data = new_data_output(segment_id, segment_suffix);

    // Per-field meta entry (`addNumericField` -> `writeValues`).
    meta.write_i32(field_number);
    meta.push(DOC_VALUES_TYPE_NUMERIC);
    write_dense_numeric_entry_body(&mut meta, &mut data, values);

    let skip_index =
        finish_field_list_and_footers(&mut meta, &mut data, segment_id, segment_suffix);
    Ok((meta, data, skip_index))
}

/// Port of `Lucene90DocValuesConsumer.addNumericField`'s **sparse** branch
/// (`numDocsWithValue != maxDoc`, feeding an [`indexed_disi`]-backed
/// docs-with-field structure instead of the `-1`/DENSE marker
/// [`write_single_dense_numeric_field`] always writes) -- the one doc-values
/// type/shape this port's write side extends beyond dense in this slice; see
/// that function's doc comment for the rest of this module's scope
/// statement (BINARY/SORTED/SORTED_NUMERIC/SORTED_SET sparse writing is
/// still deferred, as is real Lucene's SPARSE-as-shorts-vs-DENSE-bitset
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
/// Deliberately not attempted here, same as [`write_single_dense_numeric_field`]:
/// sparse fields (`IndexedDISI`), per-field doc-values skip indexes, and
/// multiple fields in one `.dvm`/`.dvd`/`.dvs` triple.
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
    if values.len() != max_doc as usize {
        return Err(WriteError::NotDense {
            values: values.len(),
            max_doc,
        });
    }

    let mut meta = new_meta_output(segment_id, segment_suffix);
    let mut data = new_data_output(segment_id, segment_suffix);

    meta.write_i32(field_number);
    meta.push(DOC_VALUES_TYPE_BINARY);

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
    for (doc, per_doc) in values.iter().enumerate() {
        if per_doc.is_empty() {
            return Err(WriteError::EmptyMultiValuedDoc(doc as i32));
        }
    }

    let num_docs_with_field = values.len() as i32;
    let flat: Vec<i64> = values.iter().flatten().copied().collect();

    let mut meta = new_meta_output(segment_id, segment_suffix);
    let mut data = new_data_output(segment_id, segment_suffix);

    meta.write_i32(field_number);
    meta.push(DOC_VALUES_TYPE_SORTED_NUMERIC);
    write_dense_numeric_entry_body(&mut meta, &mut data, &flat);

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
/// Deliberately not attempted here, same as
/// [`write_single_dense_numeric_field`]: sparse fields (`IndexedDISI`,
/// i.e. `Ok(None)`/missing values), per-field doc-values skip indexes, and
/// multiple fields in one `.dvm`/`.dvd`/`.dvs` triple.
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
    if values.len() != max_doc as usize {
        return Err(WriteError::NotDense {
            values: values.len(),
            max_doc,
        });
    }

    let mut dict: Vec<Vec<u8>> = values.to_vec();
    dict.sort_unstable();
    dict.dedup();

    let ords: Vec<i64> = values
        .iter()
        .map(|v| dict.binary_search(v).unwrap() as i64)
        .collect();

    let mut meta = new_meta_output(segment_id, segment_suffix);
    let mut data = new_data_output(segment_id, segment_suffix);

    meta.write_i32(field_number);
    meta.push(DOC_VALUES_TYPE_SORTED);
    write_dense_numeric_entry_body(&mut meta, &mut data, &ords);
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
/// Deliberately not attempted here, same as [`write_single_dense_numeric_field`]:
/// sparse per-doc value *presence* (every doc must have >= 1 value),
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
    if values.len() != max_doc as usize {
        return Err(WriteError::NotDense {
            values: values.len(),
            max_doc,
        });
    }
    for (doc, per_doc) in values.iter().enumerate() {
        if per_doc.is_empty() {
            return Err(WriteError::EmptyMultiValuedDoc(doc as i32));
        }
    }

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

    let mut meta = new_meta_output(segment_id, segment_suffix);
    let mut data = new_data_output(segment_id, segment_suffix);

    meta.write_i32(field_number);
    meta.push(DOC_VALUES_TYPE_SORTED_SET);

    if all_single {
        meta.push(0); // multiValued = false: plain SORTED shape.
        let single_ords: Vec<i64> = per_doc_ords.iter().map(|ords| ords[0]).collect();
        write_dense_numeric_entry_body(&mut meta, &mut data, &single_ords);
    } else {
        meta.push(1); // multiValued = true.
        let flat: Vec<i64> = per_doc_ords.iter().flatten().copied().collect();
        write_dense_numeric_entry_body(&mut meta, &mut data, &flat);

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

    write_terms_dict(&mut meta, &mut data, &dict);

    let skip_index =
        finish_field_list_and_footers(&mut meta, &mut data, segment_id, segment_suffix);
    Ok((meta, data, skip_index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};

    fn numeric_field(number: i32) -> FieldInfo {
        FieldInfo {
            name: format!("f{number}"),
            number,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::Numeric,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: Vec::new(),
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: crate::field_infos::VectorEncoding::Float32,
            vector_similarity_function: crate::field_infos::VectorSimilarityFunction::Euclidean,
        }
    }

    struct EntryBuilder {
        field_number: i32,
        docs_with_field_offset: i64,
        docs_with_field_length: i64,
        num_values: i64,
        table: Option<Vec<i64>>,
        bits_per_value: u8,
        min_value: i64,
        gcd: i64,
        values_offset: i64,
        values_length: i64,
    }

    impl EntryBuilder {
        fn dense(field_number: i32, bits_per_value: u8, num_values: i64) -> Self {
            Self {
                field_number,
                docs_with_field_offset: DOCS_WITH_FIELD_DENSE,
                docs_with_field_length: 0,
                num_values,
                table: None,
                bits_per_value,
                min_value: 0,
                gcd: 1,
                values_offset: 0,
                values_length: 0,
            }
        }

        fn build(&self, out: &mut Vec<u8>) {
            out.extend_from_slice(&self.field_number.to_le_bytes());
            out.push(DOC_VALUES_TYPE_NUMERIC);
            self.build_body(out);
        }

        /// Just the numeric-entry body (no field number / type byte) --
        /// SORTED fields reuse this exact layout for their ordinals.
        fn build_body(&self, out: &mut Vec<u8>) {
            out.extend_from_slice(&self.docs_with_field_offset.to_le_bytes());
            out.extend_from_slice(&self.docs_with_field_length.to_le_bytes());
            out.extend_from_slice(&0i16.to_le_bytes()); // jumpTableEntryCount
            out.push(0); // denseRankPower
            out.extend_from_slice(&self.num_values.to_le_bytes());
            match &self.table {
                Some(t) => {
                    out.extend_from_slice(&(t.len() as i32).to_le_bytes());
                    for v in t {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
                None => out.extend_from_slice(&(-1i32).to_le_bytes()),
            }
            out.push(self.bits_per_value);
            out.extend_from_slice(&self.min_value.to_le_bytes());
            out.extend_from_slice(&self.gcd.to_le_bytes());
            out.extend_from_slice(&self.values_offset.to_le_bytes());
            out.extend_from_slice(&self.values_length.to_le_bytes());
            out.extend_from_slice(&0i64.to_le_bytes()); // valueJumpTableOffset
        }

        /// Round-trips through `build_body`/`read_numeric_entry` rather than
        /// constructing a `NumericEntry` field-by-field, so this stays in
        /// sync with the real parser automatically.
        fn to_entry(&self) -> NumericEntry {
            let mut bytes = Vec::new();
            self.build_body(&mut bytes);
            let mut input = SliceInput::new(&bytes);
            read_numeric_entry(&mut input, self.field_number).unwrap()
        }
    }

    fn build_dvm(id: &[u8; ID_LENGTH], entries: &[EntryBuilder]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut out, META_CODEC);
        out.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        out.extend_from_slice(id);
        out.push(0); // empty suffix
        for e in entries {
            e.build(&mut out);
        }
        out.extend_from_slice(&(-1i32).to_le_bytes());
        out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&out) as u64;
        out.extend_from_slice(&checksum.to_be_bytes());
        out
    }

    fn build_dvd(id: &[u8; ID_LENGTH], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut out, "Lucene90DocValuesData");
        out.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        out.extend_from_slice(id);
        out.push(0);
        out.extend_from_slice(payload);
        out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&out) as u64;
        out.extend_from_slice(&checksum.to_be_bytes());
        out
    }

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

    fn field_infos_with(numbers: &[i32]) -> FieldInfos {
        FieldInfos {
            fields: numbers.iter().map(|&n| numeric_field(n)).collect(),
        }
    }

    #[test]
    fn empty_meta_parses_no_fields() {
        let id = [1u8; ID_LENGTH];
        let buf = build_dvm(&id, &[]);
        let fis = field_infos_with(&[]);
        let (version, meta) = parse_meta(&buf, &id, "", &fis).unwrap();
        assert_eq!(version, VERSION_CURRENT);
        assert_eq!(meta.numeric.len(), 0);
        assert_eq!(meta.binary.len(), 0);
    }

    #[test]
    fn unknown_field_number_rejected() {
        let id = [1u8; ID_LENGTH];
        let e = EntryBuilder::dense(5, 8, 3);
        let buf = build_dvm(&id, &[e]);
        let fis = field_infos_with(&[]);
        assert!(matches!(
            parse_meta(&buf, &id, "", &fis),
            Err(Error::UnknownFieldNumber(5))
        ));
    }

    #[test]
    fn invalid_bits_per_value_rejected() {
        let id = [1u8; ID_LENGTH];
        let mut e = EntryBuilder::dense(0, 8, 3);
        e.bits_per_value = 3; // not a supported width
        let buf = build_dvm(&id, &[e]);
        let fis = field_infos_with(&[0]);
        assert!(matches!(
            parse_meta(&buf, &id, "", &fis),
            Err(Error::InvalidBitsPerValue(3, 0))
        ));
    }

    #[test]
    fn invalid_table_size_rejected() {
        let id = [1u8; ID_LENGTH];
        let mut out = Vec::new();
        out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut out, META_CODEC);
        out.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        out.extend_from_slice(&id);
        out.push(0);
        out.extend_from_slice(&0i32.to_le_bytes()); // field number
        out.push(DOC_VALUES_TYPE_NUMERIC);
        out.extend_from_slice(&DOCS_WITH_FIELD_DENSE.to_le_bytes());
        out.extend_from_slice(&0i64.to_le_bytes());
        out.extend_from_slice(&0i16.to_le_bytes());
        out.push(0);
        out.extend_from_slice(&3i64.to_le_bytes());
        out.extend_from_slice(&300i32.to_le_bytes()); // tableSize > 256
        let fis = field_infos_with(&[0]);
        assert!(matches!(
            parse_meta(&out, &id, "", &fis),
            Err(Error::InvalidTableSize(300))
        ));
    }

    #[test]
    fn varying_bpv_blocks_rejected() {
        let id = [1u8; ID_LENGTH];
        let mut e = EntryBuilder::dense(0, 8, 3);
        e.table = None;
        let mut buf = Vec::new();
        buf.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut buf, META_CODEC);
        buf.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        buf.extend_from_slice(&id);
        buf.push(0);
        buf.extend_from_slice(&e.field_number.to_le_bytes());
        buf.push(DOC_VALUES_TYPE_NUMERIC);
        buf.extend_from_slice(&e.docs_with_field_offset.to_le_bytes());
        buf.extend_from_slice(&e.docs_with_field_length.to_le_bytes());
        buf.extend_from_slice(&0i16.to_le_bytes());
        buf.push(0);
        buf.extend_from_slice(&e.num_values.to_le_bytes());
        buf.extend_from_slice(&(-3i32).to_le_bytes()); // tableSize < -1 -> varying bpv
        let fis = field_infos_with(&[0]);
        assert!(matches!(
            parse_meta(&buf, &id, "", &fis),
            Err(Error::UnsupportedVaryingBpvBlocks(0))
        ));
    }

    #[test]
    fn unsupported_field_type_rejected() {
        let id = [1u8; ID_LENGTH];
        let mut buf = Vec::new();
        buf.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut buf, META_CODEC);
        buf.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        buf.extend_from_slice(&id);
        buf.push(0);
        buf.extend_from_slice(&0i32.to_le_bytes());
        buf.push(9); // not a valid doc values type byte
        let fis = field_infos_with(&[0]);
        assert!(matches!(
            parse_meta(&buf, &id, "", &fis),
            Err(Error::UnsupportedFieldType(0, 9))
        ));
    }

    #[test]
    fn unsupported_skip_index_rejected() {
        let id = [1u8; ID_LENGTH];
        let mut fis = field_infos_with(&[0]);
        fis.fields[0].doc_values_skip_index_type = DocValuesSkipIndexType::Range;
        let e = EntryBuilder::dense(0, 8, 3);
        let mut buf = Vec::new();
        buf.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut buf, META_CODEC);
        buf.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        buf.extend_from_slice(&id);
        buf.push(0);
        buf.extend_from_slice(&e.field_number.to_le_bytes());
        buf.push(DOC_VALUES_TYPE_NUMERIC);
        assert!(matches!(
            parse_meta(&buf, &id, "", &fis),
            Err(Error::UnsupportedSkipIndex(0))
        ));
    }

    #[test]
    fn empty_field_has_no_value_anywhere() {
        let id = [1u8; ID_LENGTH];
        let mut e = EntryBuilder::dense(0, 8, 0);
        e.docs_with_field_offset = DOCS_WITH_FIELD_EMPTY;
        let buf = build_dvm(&id, &[e]);
        let fis = field_infos_with(&[0]);
        let (_, meta) = parse_meta(&buf, &id, "", &fis).unwrap();
        let entry = meta.numeric_entry(0).unwrap();
        assert!(entry.is_empty_field());
        assert_eq!(numeric_value(&[], entry, 0).unwrap(), None);
    }

    #[test]
    fn doc_out_of_range_rejected() {
        let id = [1u8; ID_LENGTH];
        let e = EntryBuilder::dense(0, 8, 3);
        let buf = build_dvm(&id, &[e]);
        let fis = field_infos_with(&[0]);
        let (_, meta) = parse_meta(&buf, &id, "", &fis).unwrap();
        let entry = meta.numeric_entry(0).unwrap();
        assert!(matches!(
            numeric_value(&[0, 0, 0], entry, 3),
            Err(Error::DocOutOfRange(3, 3))
        ));
        assert!(matches!(
            numeric_value(&[0, 0, 0], entry, -1),
            Err(Error::DocOutOfRange(-1, 3))
        ));
    }

    #[test]
    fn constant_value_when_bits_per_value_zero() {
        let id = [1u8; ID_LENGTH];
        let mut e = EntryBuilder::dense(0, 0, 5);
        e.min_value = 42;
        let buf = build_dvm(&id, &[e]);
        let fis = field_infos_with(&[0]);
        let (_, meta) = parse_meta(&buf, &id, "", &fis).unwrap();
        let entry = meta.numeric_entry(0).unwrap();
        for doc in 0..5 {
            assert_eq!(numeric_value(&[], entry, doc).unwrap(), Some(42));
        }
    }

    #[test]
    fn table_compressed_dense_field() {
        let id = [1u8; ID_LENGTH];
        // table [10, 20, 30]; bitsPerValue=2 needed for ordinals 0..2
        let table = vec![10i64, 20, 30];
        let header_len_data = nvd_header_len();
        let mut e = EntryBuilder::dense(0, 2, 4);
        e.table = Some(table.clone());
        e.values_offset = header_len_data as i64;
        e.values_length = 1;
        let buf = build_dvm(&id, &[e]);
        let fis = field_infos_with(&[0]);
        let (_, meta) = parse_meta(&buf, &id, "", &fis).unwrap();
        let entry = meta.numeric_entry(0).unwrap();

        // ordinals for docs 0,1,2,3 = 0,1,2,0 packed 2 bits each into one byte,
        // LSB first: byte = 0 | (1<<2) | (2<<4) | (0<<6) = 0b00_10_01_00 = 0x24
        let payload = [0b0010_0100u8];
        let data = build_dvd(&id, &payload);
        assert_eq!(numeric_value(&data, entry, 0).unwrap(), Some(10));
        assert_eq!(numeric_value(&data, entry, 1).unwrap(), Some(20));
        assert_eq!(numeric_value(&data, entry, 2).unwrap(), Some(30));
        assert_eq!(numeric_value(&data, entry, 3).unwrap(), Some(10));
    }

    #[test]
    fn gcd_delta_compressed_dense_field() {
        let id = [1u8; ID_LENGTH];
        // values 100, 106, 112 -> min=100, gcd=6 -> raw ordinals 0,1,2 (bpv=2)
        let mut e = EntryBuilder::dense(0, 2, 3);
        e.min_value = 100;
        e.gcd = 6;
        e.values_offset = nvd_header_len() as i64;
        e.values_length = 1;
        let buf = build_dvm(&id, &[e]);
        let fis = field_infos_with(&[0]);
        let (_, meta) = parse_meta(&buf, &id, "", &fis).unwrap();
        let entry = meta.numeric_entry(0).unwrap();

        // ordinals 0,1,2 packed 2 bits each: 0 | (1<<2) | (2<<4) = 0b00_10_01_00
        let payload = [0b0010_0100u8];
        let data = build_dvd(&id, &payload);
        assert_eq!(numeric_value(&data, entry, 0).unwrap(), Some(100));
        assert_eq!(numeric_value(&data, entry, 1).unwrap(), Some(106));
        assert_eq!(numeric_value(&data, entry, 2).unwrap(), Some(112));
    }

    #[test]
    fn plain_ordinals_gcd_one_min_zero() {
        let id = [1u8; ID_LENGTH];
        let mut e = EntryBuilder::dense(0, 8, 3);
        e.values_offset = nvd_header_len() as i64;
        e.values_length = 3;
        let buf = build_dvm(&id, &[e]);
        let fis = field_infos_with(&[0]);
        let (_, meta) = parse_meta(&buf, &id, "", &fis).unwrap();
        let entry = meta.numeric_entry(0).unwrap();

        let payload = [5u8, 250, 0];
        let data = build_dvd(&id, &payload);
        assert_eq!(numeric_value(&data, entry, 0).unwrap(), Some(5));
        assert_eq!(numeric_value(&data, entry, 1).unwrap(), Some(250));
        assert_eq!(numeric_value(&data, entry, 2).unwrap(), Some(0));
    }

    #[test]
    fn sparse_field_returns_value_for_present_doc_and_none_for_absent() {
        // Same IndexedDISI SPARSE-block shape as norms.rs's sparse test: docs
        // 1 and 3 present out of a block covering [0, 65536).
        let mut disi_bytes = Vec::new();
        disi_bytes.extend_from_slice(&0u16.to_le_bytes());
        disi_bytes.extend_from_slice(&1u16.to_le_bytes());
        disi_bytes.extend_from_slice(&1u16.to_le_bytes());
        disi_bytes.extend_from_slice(&3u16.to_le_bytes());
        disi_bytes.extend_from_slice(&((i32::MAX >> 16) as u16).to_le_bytes());
        disi_bytes.extend_from_slice(&0u16.to_le_bytes());
        disi_bytes.extend_from_slice(&((i32::MAX & 0xFFFF) as u16).to_le_bytes());

        let disi_length = disi_bytes.len() as i64;
        let mut data = disi_bytes.clone();
        data.push(11);
        data.push(33);

        let mut e = EntryBuilder::dense(0, 8, 2);
        e.docs_with_field_offset = 0;
        e.docs_with_field_length = disi_length;
        e.values_offset = disi_length;
        e.values_length = 2;
        let entry_bytes = {
            let mut out = Vec::new();
            e.build(&mut out);
            out
        };
        let mut input = SliceInput::new(&entry_bytes);
        let _field_number = input.read_i32().unwrap();
        let _ty = input.read_byte().unwrap();
        let entry = read_numeric_entry(&mut input, 0).unwrap();

        assert!(!entry.is_dense());
        assert!(!entry.is_empty_field());
        assert_eq!(numeric_value(&data, &entry, 1).unwrap(), Some(11));
        assert_eq!(numeric_value(&data, &entry, 3).unwrap(), Some(33));
        assert_eq!(numeric_value(&data, &entry, 2).unwrap(), None);
    }

    #[test]
    fn check_data_header_footer_valid() {
        let id = [2u8; ID_LENGTH];
        let data = build_dvd(&id, b"payload-bytes");
        let version = check_data_header_footer(&data, &id, "").unwrap();
        assert_eq!(version, VERSION_CURRENT);
    }

    #[test]
    fn check_data_header_footer_wrong_id_rejected() {
        let id = [2u8; ID_LENGTH];
        let data = build_dvd(&id, b"payload-bytes");
        let wrong_id = [3u8; ID_LENGTH];
        assert!(check_data_header_footer(&data, &wrong_id, "").is_err());
    }

    #[test]
    fn wrong_id_rejected_on_meta() {
        let id = [1u8; ID_LENGTH];
        let buf = build_dvm(&id, &[]);
        let wrong_id = [9u8; ID_LENGTH];
        assert!(matches!(
            parse_meta(&buf, &wrong_id, "", &field_infos_with(&[])),
            Err(Error::Store(_))
        ));
    }

    fn nvd_header_len() -> usize {
        4 + 1 + "Lucene90DocValuesData".len() + 4 + ID_LENGTH + 1
    }

    fn binary_entry_fixed(
        field_number: i32,
        docs_with_field_offset: i64,
        docs_with_field_length: i64,
        num_docs_with_field: i32,
        length: i32,
        data_offset: i64,
        data_length: i64,
    ) -> BinaryEntry {
        BinaryEntry {
            field_number,
            docs_with_field_offset,
            docs_with_field_length,
            jump_table_entry_count: 0,
            dense_rank_power: 0,
            num_docs_with_field,
            min_length: length,
            max_length: length,
            data_offset,
            data_length,
            addresses: None,
        }
    }

    #[test]
    fn binary_empty_field_has_no_value_anywhere() {
        let entry = binary_entry_fixed(0, DOCS_WITH_FIELD_EMPTY, 0, 0, 4, 0, 0);
        assert_eq!(binary_value(&[], &entry, 0).unwrap(), None);
    }

    #[test]
    fn binary_dense_fixed_length() {
        let entry = binary_entry_fixed(0, DOCS_WITH_FIELD_DENSE, 0, 3, 4, 0, 12);
        let data = b"aaaabbbbcccc";
        assert_eq!(binary_value(data, &entry, 0).unwrap(), Some(&b"aaaa"[..]));
        assert_eq!(binary_value(data, &entry, 1).unwrap(), Some(&b"bbbb"[..]));
        assert_eq!(binary_value(data, &entry, 2).unwrap(), Some(&b"cccc"[..]));
    }

    #[test]
    fn binary_dense_out_of_range_rejected() {
        let entry = binary_entry_fixed(0, DOCS_WITH_FIELD_DENSE, 0, 3, 4, 0, 12);
        assert!(matches!(
            binary_value(b"aaaabbbbcccc", &entry, 3),
            Err(Error::DocOutOfRange(3, 3))
        ));
        assert!(matches!(
            binary_value(b"aaaabbbbcccc", &entry, -1),
            Err(Error::DocOutOfRange(-1, 3))
        ));
    }

    #[test]
    fn binary_dense_variable_length() {
        // 3 docs: "abc" (len 3), "defg" (len 4), "hi" (len 2) -> end offsets
        // [0, 3, 7, 9], packed 4 bits each (min=0, avg=0.0 so delta==offset):
        // byte0 = 0 | (3<<4) = 0x30, byte1 = 7 | (9<<4) = 0x97.
        let addr_bytes = [0x30u8, 0x97];
        let mut meta_bytes = Vec::new();
        meta_bytes.extend_from_slice(&0i64.to_le_bytes()); // min
        meta_bytes.extend_from_slice(&(0.0f32.to_bits() as i32).to_le_bytes()); // avg
        meta_bytes.extend_from_slice(&0i64.to_le_bytes()); // offset (within addr_bytes)
        meta_bytes.push(4); // bpv
        let mut input = SliceInput::new(&meta_bytes);
        let addr_meta = direct_monotonic::load_meta(&mut input, 4, 3).unwrap(); // blockShift=3 -> 1 block

        let blob = b"abcdefghi";
        let entry = BinaryEntry {
            field_number: 0,
            docs_with_field_offset: DOCS_WITH_FIELD_DENSE,
            docs_with_field_length: 0,
            jump_table_entry_count: 0,
            dense_rank_power: 0,
            num_docs_with_field: 3,
            min_length: 2,
            max_length: 4,
            data_offset: addr_bytes.len() as i64,
            data_length: blob.len() as i64,
            addresses: Some(BinaryAddresses {
                offset: 0,
                length: addr_bytes.len() as i64,
                meta: addr_meta,
            }),
        };
        let mut data = addr_bytes.to_vec();
        data.extend_from_slice(blob);

        assert_eq!(binary_value(&data, &entry, 0).unwrap(), Some(&b"abc"[..]));
        assert_eq!(binary_value(&data, &entry, 1).unwrap(), Some(&b"defg"[..]));
        assert_eq!(binary_value(&data, &entry, 2).unwrap(), Some(&b"hi"[..]));
    }

    #[test]
    fn binary_sparse_fixed_length() {
        // Same IndexedDISI SPARSE-block shape as the numeric sparse test:
        // docs 1 and 3 present out of a block covering [0, 65536).
        let mut disi_bytes = Vec::new();
        disi_bytes.extend_from_slice(&0u16.to_le_bytes());
        disi_bytes.extend_from_slice(&1u16.to_le_bytes());
        disi_bytes.extend_from_slice(&1u16.to_le_bytes());
        disi_bytes.extend_from_slice(&3u16.to_le_bytes());
        disi_bytes.extend_from_slice(&((i32::MAX >> 16) as u16).to_le_bytes());
        disi_bytes.extend_from_slice(&0u16.to_le_bytes());
        disi_bytes.extend_from_slice(&((i32::MAX & 0xFFFF) as u16).to_le_bytes());
        let disi_length = disi_bytes.len() as i64;

        let mut data = disi_bytes.clone();
        data.extend_from_slice(b"AABB");

        let entry = binary_entry_fixed(0, 0, disi_length, 2, 2, disi_length, 4);
        assert_eq!(binary_value(&data, &entry, 1).unwrap(), Some(&b"AA"[..]));
        assert_eq!(binary_value(&data, &entry, 3).unwrap(), Some(&b"BB"[..]));
        assert_eq!(binary_value(&data, &entry, 2).unwrap(), None);
    }

    #[test]
    fn read_binary_entry_fixed_length_via_full_dvm_parse() {
        let id = [1u8; ID_LENGTH];
        let mut out = Vec::new();
        out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut out, META_CODEC);
        out.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        out.extend_from_slice(&id);
        out.push(0);
        out.extend_from_slice(&0i32.to_le_bytes()); // field number
        out.push(DOC_VALUES_TYPE_BINARY);
        out.extend_from_slice(&100i64.to_le_bytes()); // dataOffset
        out.extend_from_slice(&12i64.to_le_bytes()); // dataLength
        out.extend_from_slice(&DOCS_WITH_FIELD_DENSE.to_le_bytes());
        out.extend_from_slice(&0i64.to_le_bytes());
        out.extend_from_slice(&0i16.to_le_bytes());
        out.push(0);
        out.extend_from_slice(&3i32.to_le_bytes()); // numDocsWithField
        out.extend_from_slice(&4i32.to_le_bytes()); // minLength
        out.extend_from_slice(&4i32.to_le_bytes()); // maxLength (== minLength -> fixed)
        out.extend_from_slice(&(-1i32).to_le_bytes()); // field terminator
        out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&out) as u64;
        out.extend_from_slice(&checksum.to_be_bytes());

        let fis = field_infos_with(&[0]);
        let (_, meta) = parse_meta(&out, &id, "", &fis).unwrap();
        let entry = meta.binary_entry(0).unwrap();
        assert!(entry.is_fixed_length());
        assert!(entry.is_dense());
        assert_eq!(entry.data_offset, 100);
        assert_eq!(entry.data_length, 12);
        assert!(entry.addresses.is_none());
    }

    /// Writes a `DirectMonotonicReader.Meta` block array of all-zero,
    /// bpv=0 blocks (a constant-zero sequence -- correct-shaped but unused
    /// by the SORTED test below, which never touches the address arrays
    /// since `decode_all_terms`/`sorted_ord` don't need them).
    fn write_zero_direct_monotonic_blocks(out: &mut Vec<u8>, num_values: i64, block_shift: u32) {
        let mut num_blocks = num_values >> block_shift;
        if (num_blocks << block_shift) < num_values {
            num_blocks += 1;
        }
        for _ in 0..num_blocks {
            out.extend_from_slice(&0i64.to_le_bytes()); // min
            out.extend_from_slice(&0i32.to_le_bytes()); // avg
            out.extend_from_slice(&0i64.to_le_bytes()); // offset
            out.push(0); // bpv
        }
    }

    #[test]
    fn sorted_field_parses_and_resolves_ordinal_to_term() {
        use crate::terms_dict;

        let id = [1u8; ID_LENGTH];
        let dvd_header_len = nvd_header_len();

        // .dvd layout: [ords packed bits][terms data: "apple" + prefix-compressed "berry"]
        let ords_values_offset = dvd_header_len as i64;
        let ords_values_length = 1i64; // 3 docs * 1 bit, fits in 1 byte
        let terms_data_offset = ords_values_offset + ords_values_length;

        // 2nd term: prefix 0 shared with "apple", suffix "berry" (5 bytes).
        let mut block_body = Vec::new();
        block_body.push(4u8 << 4); // suffixLen field=4 (len 5), prefixLen=0
        block_body.extend_from_slice(b"berry");
        let mut terms_data = Vec::new();
        write_vint(&mut terms_data, 5);
        terms_data.extend_from_slice(b"apple");
        write_vint(&mut terms_data, block_body.len() as i32); // decompressed block length
        terms_data.push((block_body.len() as u8) << 4); // literal-only LZ4 token
        terms_data.extend_from_slice(&block_body);
        let terms_data_length = terms_data.len() as i64;

        // docs [0,1,2] -> ordinals [0,1,0], bpv=1: bit 1 set for doc 1 only.
        let mut dvd_payload = vec![0b0000_0010u8];
        dvd_payload.extend_from_slice(&terms_data);
        let dvd = build_dvd(&id, &dvd_payload);

        let mut ords = EntryBuilder::dense(0, 1, 3);
        ords.values_offset = ords_values_offset;
        ords.values_length = ords_values_length;

        let mut dvm = Vec::new();
        dvm.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut dvm, META_CODEC);
        dvm.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        dvm.extend_from_slice(&id);
        dvm.push(0);
        dvm.extend_from_slice(&0i32.to_le_bytes()); // field number
        dvm.push(DOC_VALUES_TYPE_SORTED);
        ords.build_body(&mut dvm);

        // TermsDictEntry (Lucene90DocValuesProducer.readTermDict).
        write_vint(&mut dvm, 2); // termsDictSize
        dvm.extend_from_slice(&2i32.to_le_bytes()); // blockShift (for the address arrays)
        write_zero_direct_monotonic_blocks(&mut dvm, 1, 2); // addresses: ceil(2/64)=1 block
        dvm.extend_from_slice(&5i32.to_le_bytes()); // maxTermLength
        dvm.extend_from_slice(&8192i32.to_le_bytes()); // maxBlockLength (unused)
        dvm.extend_from_slice(&terms_data_offset.to_le_bytes());
        dvm.extend_from_slice(&terms_data_length.to_le_bytes());
        dvm.extend_from_slice(&0i64.to_le_bytes()); // termsAddressesOffset (unused)
        dvm.extend_from_slice(&0i64.to_le_bytes()); // termsAddressesLength (unused)
        dvm.extend_from_slice(&4i32.to_le_bytes()); // termsDictIndexShift
        write_zero_direct_monotonic_blocks(&mut dvm, 1 + 1, 2); // index: 1+ceil(2/16)=2 blocks
        dvm.extend_from_slice(&0i64.to_le_bytes()); // termsIndexOffset (unused)
        dvm.extend_from_slice(&0i64.to_le_bytes()); // termsIndexLength (unused)
        dvm.extend_from_slice(&0i64.to_le_bytes()); // termsIndexAddressesOffset (unused)
        dvm.extend_from_slice(&0i64.to_le_bytes()); // termsIndexAddressesLength (unused)

        dvm.extend_from_slice(&(-1i32).to_le_bytes()); // field terminator
        dvm.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        dvm.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&dvm) as u64;
        dvm.extend_from_slice(&checksum.to_be_bytes());

        let fis = field_infos_with(&[0]);
        let (_, meta) = parse_meta(&dvm, &id, "", &fis).unwrap();
        let entry = meta.sorted_entry(0).unwrap();

        assert_eq!(sorted_ord(&dvd, entry, 0).unwrap(), Some(0));
        assert_eq!(sorted_ord(&dvd, entry, 1).unwrap(), Some(1));
        assert_eq!(sorted_ord(&dvd, entry, 2).unwrap(), Some(0));

        let terms = terms_dict::decode_all_terms(&dvd, &entry.terms).unwrap();
        assert_eq!(terms, vec![b"apple".to_vec(), b"berry".to_vec()]);
    }

    fn sorted_numeric_entry_no_addresses(numeric: NumericEntry) -> SortedNumericEntry {
        let num_docs_with_field = numeric.num_values as i32;
        SortedNumericEntry {
            field_number: numeric.field_number,
            numeric,
            num_docs_with_field,
            addresses: None,
        }
    }

    #[test]
    fn sorted_numeric_collapses_to_single_value_when_one_per_doc() {
        // 3 docs, dense, 1 value each (numValues == numDocsWithField) ->
        // addresses is None, ordinal == doc id directly.
        let mut e = EntryBuilder::dense(0, 8, 3);
        e.values_offset = nvd_header_len() as i64;
        e.values_length = 3;
        let data = build_dvd(&[1u8; ID_LENGTH], &[10, 20, 30]);
        let entry = sorted_numeric_entry_no_addresses(e.to_entry());

        assert_eq!(sorted_numeric_values(&data, &entry, 0).unwrap(), vec![10]);
        assert_eq!(sorted_numeric_values(&data, &entry, 1).unwrap(), vec![20]);
        assert_eq!(sorted_numeric_values(&data, &entry, 2).unwrap(), vec![30]);
    }

    #[test]
    fn sorted_numeric_dense_multi_value_uses_address_ranges() {
        // 3 docs with value counts [2, 0, 1]: doc0->[10,11], doc1->[], doc2->[12].
        // numValues (total)=3, numDocsWithField=3 (every doc has a value entry,
        // even if empty) -- addresses = [0,2,2,3].
        let header_len = nvd_header_len();
        let mut e = EntryBuilder::dense(0, 8, 3); // numValues=3 (total value count)
        e.values_offset = header_len as i64;
        e.values_length = 3;

        let addr_meta_bytes = {
            let mut out = Vec::new();
            out.extend_from_slice(&0i64.to_le_bytes()); // min
            out.extend_from_slice(&0i32.to_le_bytes()); // avg
            out.extend_from_slice(&0i64.to_le_bytes()); // offset
            out.push(8); // bpv=8: raw byte values, since min=0,avg=0 -> delta==value
            out
        };
        let mut addr_input = SliceInput::new(&addr_meta_bytes);
        let addr_meta = direct_monotonic::load_meta(&mut addr_input, 4, 3).unwrap(); // blockShift=3 -> 1 block for 4 values

        let values = [10u8, 11, 12];
        let addresses_bytes = [0u8, 2, 2, 3]; // the actual address array data (bpv=8 raw bytes)
        let mut data = values.to_vec();
        data.extend_from_slice(&addresses_bytes);
        let addresses_offset = header_len as i64 + values.len() as i64;

        let entry = SortedNumericEntry {
            field_number: 0,
            numeric: e.to_entry(),
            num_docs_with_field: 3,
            addresses: Some(MultiValueAddresses {
                offset: addresses_offset,
                length: addresses_bytes.len() as i64,
                meta: addr_meta,
            }),
        };
        let dvd = build_dvd(&[1u8; ID_LENGTH], &data);

        assert_eq!(
            sorted_numeric_values(&dvd, &entry, 0).unwrap(),
            vec![10, 11]
        );
        assert_eq!(
            sorted_numeric_values(&dvd, &entry, 1).unwrap(),
            Vec::<i64>::new()
        );
        assert_eq!(sorted_numeric_values(&dvd, &entry, 2).unwrap(), vec![12]);
    }

    #[test]
    fn sorted_numeric_empty_field_has_no_values_anywhere() {
        let mut e = EntryBuilder::dense(0, 8, 0);
        e.docs_with_field_offset = DOCS_WITH_FIELD_EMPTY;
        let entry = sorted_numeric_entry_no_addresses(e.to_entry());
        assert_eq!(
            sorted_numeric_values(&[], &entry, 0).unwrap(),
            Vec::<i64>::new()
        );
    }

    #[test]
    fn sorted_numeric_negative_doc_rejected() {
        let e = EntryBuilder::dense(0, 8, 3);
        let entry = sorted_numeric_entry_no_addresses(e.to_entry());
        assert!(matches!(
            sorted_numeric_values(&[], &entry, -1),
            Err(Error::DocOutOfRange(-1, 3))
        ));
    }

    #[test]
    fn parse_meta_sorted_numeric_and_sorted_set_multi_round_trip() {
        let id = [1u8; ID_LENGTH];
        let dvd_header_len = nvd_header_len();

        // SORTED_NUMERIC field (number 0): 2 docs, dense, 1 value/doc ->
        // collapses to no-addresses shape.
        let mut sn_numeric = EntryBuilder::dense(0, 8, 2);
        sn_numeric.values_offset = dvd_header_len as i64;
        sn_numeric.values_length = 2;

        // SORTED_SET field (number 1), multi-valued: reuse the same
        // no-addresses shape for ords, plus the "apple"/"berry" terms dict
        // from `sorted_field_parses_and_resolves_ordinal_to_term`.
        let ss_ords_values_offset = dvd_header_len as i64 + 2;
        let mut ss_ords = EntryBuilder::dense(1, 1, 3);
        ss_ords.values_offset = ss_ords_values_offset;
        ss_ords.values_length = 1;

        let mut block_body = Vec::new();
        block_body.push(4u8 << 4);
        block_body.extend_from_slice(b"berry");
        let mut terms_data = Vec::new();
        write_vint(&mut terms_data, 5);
        terms_data.extend_from_slice(b"apple");
        write_vint(&mut terms_data, block_body.len() as i32);
        terms_data.push((block_body.len() as u8) << 4);
        terms_data.extend_from_slice(&block_body);
        let terms_data_offset = ss_ords_values_offset + 1;

        let mut dvd_payload = vec![10u8, 20]; // SORTED_NUMERIC values
        dvd_payload.push(0b0000_0010); // SORTED_SET ords packed bits (doc1's bit set)
        dvd_payload.extend_from_slice(&terms_data);
        let dvd = build_dvd(&id, &dvd_payload);

        let mut dvm = Vec::new();
        dvm.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut dvm, META_CODEC);
        dvm.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        dvm.extend_from_slice(&id);
        dvm.push(0);

        dvm.extend_from_slice(&0i32.to_le_bytes());
        dvm.push(DOC_VALUES_TYPE_SORTED_NUMERIC);
        sn_numeric.build_body(&mut dvm);
        dvm.extend_from_slice(&2i32.to_le_bytes()); // numDocsWithField == numValues -> no addresses

        dvm.extend_from_slice(&1i32.to_le_bytes());
        dvm.push(DOC_VALUES_TYPE_SORTED_SET);
        dvm.push(1); // multiValued=1
        ss_ords.build_body(&mut dvm);
        dvm.extend_from_slice(&3i32.to_le_bytes()); // numDocsWithField == numValues -> no addresses
        write_vint(&mut dvm, 2); // termsDictSize
        dvm.extend_from_slice(&2i32.to_le_bytes()); // blockShift
        write_zero_direct_monotonic_blocks(&mut dvm, 1, 2);
        dvm.extend_from_slice(&5i32.to_le_bytes()); // maxTermLength
        dvm.extend_from_slice(&8192i32.to_le_bytes());
        dvm.extend_from_slice(&terms_data_offset.to_le_bytes());
        dvm.extend_from_slice(&(terms_data.len() as i64).to_le_bytes());
        dvm.extend_from_slice(&0i64.to_le_bytes());
        dvm.extend_from_slice(&0i64.to_le_bytes());
        dvm.extend_from_slice(&4i32.to_le_bytes());
        write_zero_direct_monotonic_blocks(&mut dvm, 2, 2);
        dvm.extend_from_slice(&0i64.to_le_bytes());
        dvm.extend_from_slice(&0i64.to_le_bytes());
        dvm.extend_from_slice(&0i64.to_le_bytes());
        dvm.extend_from_slice(&0i64.to_le_bytes());

        dvm.extend_from_slice(&(-1i32).to_le_bytes());
        dvm.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        dvm.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&dvm) as u64;
        dvm.extend_from_slice(&checksum.to_be_bytes());

        let fis = field_infos_with(&[0, 1]);
        let (_, meta) = parse_meta(&dvm, &id, "", &fis).unwrap();

        let sn_entry = meta.sorted_numeric_entry(0).unwrap();
        assert_eq!(sorted_numeric_values(&dvd, sn_entry, 0).unwrap(), vec![10]);
        assert_eq!(sorted_numeric_values(&dvd, sn_entry, 1).unwrap(), vec![20]);

        let ss_entry = meta.sorted_set_entry(1).unwrap();
        match &ss_entry.kind {
            SortedSetKind::Multi { ords, terms } => {
                // packed bits 0b0000_0010 -> doc0=ord 0 ("apple"), doc1=ord 1
                // ("berry"), doc2=ord 0 ("apple") -- every doc is dense/1-valued
                // here, so there's no "no value" case in this scenario.
                assert_eq!(sorted_numeric_values(&dvd, ords, 0).unwrap(), vec![0]);
                assert_eq!(sorted_numeric_values(&dvd, ords, 1).unwrap(), vec![1]);
                assert_eq!(sorted_numeric_values(&dvd, ords, 2).unwrap(), vec![0]);
                let terms = terms_dict::decode_all_terms(&dvd, terms).unwrap();
                assert_eq!(terms, vec![b"apple".to_vec(), b"berry".to_vec()]);
            }
            SortedSetKind::Single(_) => panic!("expected Multi"),
        }
    }

    #[test]
    fn write_single_dense_numeric_field_round_trips_through_own_reader() {
        let id = [7u8; ID_LENGTH];
        let values = vec![5i64, 250, 0, 100];
        let (meta_bytes, data_bytes, skip_bytes) =
            write_single_dense_numeric_field(0, &values, values.len() as i32, &id, "").unwrap();

        // .dvs is header+footer only in this slice's scope.
        assert_eq!(
            check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id)
                .unwrap(),
            VERSION_CURRENT
        );

        let version = check_data_header_footer(&data_bytes, &id, "").unwrap();
        assert_eq!(version, VERSION_CURRENT);

        let fis = field_infos_with(&[0]);
        let (meta_version, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
        assert_eq!(meta_version, VERSION_CURRENT);
        let entry = meta.numeric_entry(0).unwrap();
        assert!(entry.is_dense());
        for (doc, &want) in values.iter().enumerate() {
            assert_eq!(
                numeric_value(&data_bytes, entry, doc as i32).unwrap(),
                Some(want)
            );
        }
    }

    #[test]
    fn write_single_dense_numeric_field_all_equal_values_uses_constant_encoding() {
        let id = [8u8; ID_LENGTH];
        let values = vec![42i64; 6];
        let (meta_bytes, data_bytes, _skip_bytes) =
            write_single_dense_numeric_field(0, &values, values.len() as i32, &id, "").unwrap();

        let fis = field_infos_with(&[0]);
        let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
        let entry = meta.numeric_entry(0).unwrap();
        assert_eq!(entry.bits_per_value, 0);
        for doc in 0..values.len() as i32 {
            assert_eq!(numeric_value(&data_bytes, entry, doc).unwrap(), Some(42));
        }
    }

    #[test]
    fn write_single_dense_numeric_field_negative_and_large_values() {
        let id = [9u8; ID_LENGTH];
        let values = vec![-1_000_000i64, 0, 1_000_000, 500_000];
        let (meta_bytes, data_bytes, _) =
            write_single_dense_numeric_field(0, &values, values.len() as i32, &id, "").unwrap();

        let fis = field_infos_with(&[0]);
        let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
        let entry = meta.numeric_entry(0).unwrap();
        for (doc, &want) in values.iter().enumerate() {
            assert_eq!(
                numeric_value(&data_bytes, entry, doc as i32).unwrap(),
                Some(want)
            );
        }
    }

    #[test]
    fn write_single_dense_numeric_field_uses_gcd_compression_when_a_common_divisor_exists() {
        // 300 distinct multiples of 1000 (0, 1000, ..., 299000): more than
        // the 256-entry table cap, so table compression is never even a
        // candidate here (`unique_tracked` gets abandoned partway through) --
        // this isolates the GCD path on its own. Plain delta would need
        // unsignedBitsRequired(299000) = 19 bits; dividing out gcd=1000
        // shrinks that to unsignedBitsRequired(299) = 9 bits, a real,
        // measurable win.
        let id = [20u8; ID_LENGTH];
        let values: Vec<i64> = (0..300).map(|i| i * 1000).collect();
        let (meta_bytes, data_bytes, _) =
            write_single_dense_numeric_field(0, &values, values.len() as i32, &id, "").unwrap();

        let fis = field_infos_with(&[0]);
        let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
        let entry = meta.numeric_entry(0).unwrap();

        assert_eq!(entry.gcd, 1000);
        assert!(entry.table.is_none());
        let plain_delta_bits = direct_reader::unsigned_bits_required(299_000);
        assert!(
            entry.bits_per_value < plain_delta_bits,
            "GCD compression should need fewer bits ({}) than plain delta ({plain_delta_bits})",
            entry.bits_per_value
        );
        assert_eq!(
            entry.bits_per_value,
            direct_reader::unsigned_bits_required(299)
        );

        for (doc, &want) in values.iter().enumerate() {
            assert_eq!(
                numeric_value(&data_bytes, entry, doc as i32).unwrap(),
                Some(want)
            );
        }
    }

    #[test]
    fn write_single_dense_numeric_field_falls_back_to_plain_delta_with_no_common_gcd() {
        // No common divisor > 1 across these five values (consecutive-ish,
        // coprime spread) -- must fall back to gcd = 1 rather than pick some
        // spurious divisor.
        let id = [21u8; ID_LENGTH];
        let values: Vec<i64> = vec![0, 1, 3, 7, 100];
        let (meta_bytes, data_bytes, _) =
            write_single_dense_numeric_field(0, &values, values.len() as i32, &id, "").unwrap();

        let fis = field_infos_with(&[0]);
        let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
        let entry = meta.numeric_entry(0).unwrap();

        assert_eq!(entry.gcd, 1);
        for (doc, &want) in values.iter().enumerate() {
            assert_eq!(
                numeric_value(&data_bytes, entry, doc as i32).unwrap(),
                Some(want)
            );
        }
    }

    #[test]
    fn write_single_dense_numeric_field_gcd_computation_does_not_panic_on_extreme_first_value() {
        // Found in review: the GCD loop's overflow guard only checked the
        // CURRENT value against [i64::MIN/2, i64::MAX/2], never the first
        // value itself. With first_value == i64::MIN and a later in-range
        // value, `v - first_value` overflowed i64 and panicked in debug
        // builds (including `cargo test`) even though the per-value guard
        // passed. Must not panic, regardless of what gcd/encoding it picks.
        let id = [22u8; ID_LENGTH];
        let values: Vec<i64> = vec![i64::MIN, 0, 5, 5, 5];
        let (meta_bytes, data_bytes, _) =
            write_single_dense_numeric_field(0, &values, values.len() as i32, &id, "").unwrap();

        let fis = field_infos_with(&[0]);
        let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
        let entry = meta.numeric_entry(0).unwrap();
        for (doc, &want) in values.iter().enumerate() {
            assert_eq!(
                numeric_value(&data_bytes, entry, doc as i32).unwrap(),
                Some(want)
            );
        }
    }

    #[test]
    fn write_single_dense_numeric_field_extreme_min_does_not_panic_in_delta_encoding_path() {
        // Same class of bug as the test above, but with >256 distinct
        // values so table compression is ruled out (`unique.len() > 256`)
        // and the plain-delta path's own `v - min` computation
        // (`raw.push((v - min) / gcd)`) is what actually gets exercised --
        // the previous test's 3-distinct-value input took the table path
        // instead, which never reaches this line at all.
        let id = [23u8; ID_LENGTH];
        let mut values: Vec<i64> = vec![i64::MIN];
        values.extend(0..300);
        let (meta_bytes, data_bytes, _) =
            write_single_dense_numeric_field(0, &values, values.len() as i32, &id, "").unwrap();

        let fis = field_infos_with(&[0]);
        let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
        let entry = meta.numeric_entry(0).unwrap();
        for (doc, &want) in values.iter().enumerate() {
            assert_eq!(
                numeric_value(&data_bytes, entry, doc as i32).unwrap(),
                Some(want)
            );
        }
    }

    #[test]
    fn write_single_dense_numeric_field_uses_table_compression_for_few_distinct_values() {
        // Only 3 distinct values (0, 1, 1_000_000) repeated across 64 docs,
        // with no shared GCD (0 and 1 are both present, so gcd collapses to
        // 1): table compression needs unsignedBitsRequired(3 - 1) = 2
        // bits/doc, while plain/GCD delta over range [0, 1_000_000] needs
        // unsignedBitsRequired(1_000_000) = 20 bits/doc -- table wins by a
        // wide, checkable margin.
        let id = [22u8; ID_LENGTH];
        let values: Vec<i64> = (0..64)
            .map(|i| match i % 3 {
                0 => 0,
                1 => 1,
                _ => 1_000_000,
            })
            .collect();
        let (meta_bytes, data_bytes, _) =
            write_single_dense_numeric_field(0, &values, values.len() as i32, &id, "").unwrap();

        let fis = field_infos_with(&[0]);
        let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
        let entry = meta.numeric_entry(0).unwrap();

        let table = entry.table.as_ref().expect("table compression expected");
        assert_eq!(table, &vec![0, 1, 1_000_000]);
        assert_eq!(entry.min_value, 0);
        assert_eq!(entry.gcd, 1);
        let plain_delta_bits = direct_reader::unsigned_bits_required(1_000_000);
        assert!(
            entry.bits_per_value < plain_delta_bits,
            "table compression should need fewer bits ({}) than plain delta ({plain_delta_bits})",
            entry.bits_per_value
        );
        assert_eq!(
            entry.bits_per_value,
            direct_reader::unsigned_bits_required(2)
        );

        for (doc, &want) in values.iter().enumerate() {
            assert_eq!(
                numeric_value(&data_bytes, entry, doc as i32).unwrap(),
                Some(want)
            );
        }
    }

    #[test]
    fn write_single_dense_numeric_field_rejects_non_dense_value_count() {
        let id = [1u8; ID_LENGTH];
        let err = write_single_dense_numeric_field(0, &[1, 2, 3], 5, &id, "").unwrap_err();
        assert!(matches!(
            err,
            WriteError::NotDense {
                values: 3,
                max_doc: 5
            }
        ));
    }

    #[test]
    fn write_single_sparse_numeric_field_round_trips_through_own_reader() {
        // Every 3rd doc out of 200000 has a value -- enough docs to span
        // three 65536-doc blocks, so `indexed_disi::write` picks its shape
        // per block from actual density: with 1/3 of docs present, every
        // block here has ~21845 docs present, well above the 4095 SPARSE
        // threshold, so all three blocks land in DENSE-bitset shape. See
        // `write_single_sparse_numeric_field_uses_all_three_block_shapes`
        // below for a case that actually forces SPARSE and ALL too.
        let id = [11u8; ID_LENGTH];
        let max_doc = 200_000i32;
        let doc_values: Vec<(i32, i64)> = (0..max_doc)
            .step_by(3)
            .map(|doc| (doc, (doc as i64) * 7 - 3))
            .collect();

        let (meta_bytes, data_bytes, _skip_bytes) =
            write_single_sparse_numeric_field(0, &doc_values, max_doc, &id, "").unwrap();

        let version = check_data_header_footer(&data_bytes, &id, "").unwrap();
        assert_eq!(version, VERSION_CURRENT);

        let fis = field_infos_with(&[0]);
        let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
        let entry = meta.numeric_entry(0).unwrap();
        assert!(!entry.is_dense());
        assert!(!entry.is_empty_field());

        // `numeric_value` re-decodes the whole IndexedDISI structure on every
        // call (see that function's doc comment / indexed_disi.rs's module
        // doc for why -- a one-shot decode-then-binary-search design, not
        // built for per-call random access), so checking every one of
        // `max_doc` docs here would be O(max_doc^2). Sample instead: every
        // present doc plus a stride of absent ones is still an exhaustive
        // check of both branches without the quadratic blowup.
        let present: std::collections::HashMap<i32, i64> = doc_values.iter().copied().collect();
        for &(doc, want) in doc_values.iter().step_by(97) {
            assert_eq!(numeric_value(&data_bytes, entry, doc).unwrap(), Some(want));
        }
        for doc in (0..max_doc).step_by(97) {
            let got = numeric_value(&data_bytes, entry, doc).unwrap();
            assert_eq!(got, present.get(&doc).copied(), "doc {doc}");
        }
    }

    #[test]
    fn write_single_sparse_numeric_field_uses_all_three_block_shapes() {
        // Block 0 (docs 0..65536): only 10 docs present -> SPARSE.
        // Block 1 (docs 65536..131072): every doc present -> ALL.
        // Block 2 (docs 131072..196608): half the docs present -> DENSE.
        let id = [12u8; ID_LENGTH];
        let max_doc = 196_608i32; // 3 * 65536
        let mut doc_values: Vec<(i32, i64)> = Vec::new();
        for i in 0..10 {
            doc_values.push((i * 1000, i as i64));
        }
        for doc in 65536..131072 {
            doc_values.push((doc, doc as i64));
        }
        for doc in (131072..196608).step_by(2) {
            doc_values.push((doc, doc as i64));
        }

        let (meta_bytes, data_bytes, _skip_bytes) =
            write_single_sparse_numeric_field(0, &doc_values, max_doc, &id, "").unwrap();

        let fis = field_infos_with(&[0]);
        let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
        let entry = meta.numeric_entry(0).unwrap();

        // Same O(n^2)-avoidance rationale as the round-trip test above:
        // check every present doc (all three block shapes exercised) plus a
        // stride of absent ones, not all `max_doc` docs individually.
        let present: std::collections::HashMap<i32, i64> = doc_values.iter().copied().collect();
        // All 10 SPARSE-block docs (cheap: few calls), plus a stride through
        // the ALL and DENSE blocks so every block shape gets both a present-
        // and absent-doc check without O(n^2) blowup.
        for &(doc, want) in doc_values.iter().take(10) {
            assert_eq!(
                numeric_value(&data_bytes, entry, doc).unwrap(),
                Some(want),
                "doc {doc}"
            );
        }
        for doc in (0..max_doc).step_by(4001) {
            let got = numeric_value(&data_bytes, entry, doc).unwrap();
            assert_eq!(got, present.get(&doc).copied(), "doc {doc}");
        }
    }

    #[test]
    fn write_single_sparse_numeric_field_rejects_duplicate_doc_id() {
        let id = [13u8; ID_LENGTH];
        let err =
            write_single_sparse_numeric_field(0, &[(1, 10), (1, 20)], 5, &id, "").unwrap_err();
        assert!(matches!(err, WriteError::DocIdsNotAscending(1)));
    }

    #[test]
    fn write_single_sparse_numeric_field_rejects_out_of_range_doc_id() {
        let id = [14u8; ID_LENGTH];
        let err =
            write_single_sparse_numeric_field(0, &[(0, 10), (5, 20)], 5, &id, "").unwrap_err();
        assert!(matches!(err, WriteError::DocIdOutOfRange(5, 5)));
    }

    #[test]
    fn write_single_dense_numeric_field_still_dense_after_sparse_addition() {
        // Regression: adding the sparse write path must not change the
        // dense path's output at all. Same values/assertions as
        // `write_single_dense_numeric_field_round_trips_through_own_reader`.
        let id = [7u8; ID_LENGTH];
        let values = vec![5i64, 250, 0, 100];
        let (meta_bytes, data_bytes, skip_bytes) =
            write_single_dense_numeric_field(0, &values, values.len() as i32, &id, "").unwrap();

        assert_eq!(
            check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id)
                .unwrap(),
            VERSION_CURRENT
        );
        let version = check_data_header_footer(&data_bytes, &id, "").unwrap();
        assert_eq!(version, VERSION_CURRENT);

        let fis = field_infos_with(&[0]);
        let (meta_version, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
        assert_eq!(meta_version, VERSION_CURRENT);
        let entry = meta.numeric_entry(0).unwrap();
        assert!(entry.is_dense());
        for (doc, &want) in values.iter().enumerate() {
            assert_eq!(
                numeric_value(&data_bytes, entry, doc as i32).unwrap(),
                Some(want)
            );
        }
    }

    /// Same shape as [`check_data_header_footer`] but parameterized over the
    /// expected codec name, for verifying the `.dvs` skip-index file's
    /// header/footer (which uses a different codec string than `.dvd`).
    fn check_data_header_footer_generic(
        buf: &[u8],
        codec: &str,
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<i32> {
        let mut input = SliceInput::new(buf);
        let header = codec_util::check_index_header(
            &mut input,
            codec,
            VERSION_START,
            VERSION_CURRENT,
            segment_id,
            "",
        )?;
        codec_util::retrieve_checksum(buf)?;
        Ok(header.version)
    }

    #[test]
    fn invalid_multi_valued_flag_rejected() {
        let id = [1u8; ID_LENGTH];
        let mut dvm = Vec::new();
        dvm.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut dvm, META_CODEC);
        dvm.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        dvm.extend_from_slice(&id);
        dvm.push(0);
        dvm.extend_from_slice(&0i32.to_le_bytes());
        dvm.push(DOC_VALUES_TYPE_SORTED_SET);
        dvm.push(2); // invalid multiValued flag

        let fis = field_infos_with(&[0]);
        assert!(matches!(
            parse_meta(&dvm, &id, "", &fis),
            Err(Error::InvalidMultiValuedFlag(0, 2))
        ));
    }

    fn binary_field_infos() -> FieldInfos {
        let mut fis = field_infos_with(&[0]);
        fis.fields[0].doc_values_type = DocValuesType::Binary;
        fis
    }

    fn sorted_numeric_field_infos() -> FieldInfos {
        let mut fis = field_infos_with(&[0]);
        fis.fields[0].doc_values_type = DocValuesType::SortedNumeric;
        fis
    }

    #[test]
    fn write_single_dense_binary_field_fixed_length_round_trips() {
        let id = [10u8; ID_LENGTH];
        let values: Vec<Vec<u8>> = vec![b"aa".to_vec(), b"bb".to_vec(), b"cc".to_vec()];
        let (meta_bytes, data_bytes, skip_bytes) =
            write_single_dense_binary_field(0, &values, values.len() as i32, &id, "").unwrap();

        assert_eq!(
            check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id)
                .unwrap(),
            VERSION_CURRENT
        );
        assert_eq!(
            check_data_header_footer(&data_bytes, &id, "").unwrap(),
            VERSION_CURRENT
        );

        let fis = binary_field_infos();
        let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
        let entry = meta.binary_entry(0).unwrap();
        assert!(entry.is_fixed_length());
        for (doc, want) in values.iter().enumerate() {
            assert_eq!(
                binary_value(&data_bytes, entry, doc as i32).unwrap(),
                Some(want.as_slice())
            );
        }
    }

    #[test]
    fn write_single_dense_binary_field_variable_length_round_trips() {
        let id = [11u8; ID_LENGTH];
        let values: Vec<Vec<u8>> = vec![
            b"a".to_vec(),
            b"".to_vec(),
            b"bbbbb".to_vec(),
            b"cc".to_vec(),
        ];
        let (meta_bytes, data_bytes, _skip_bytes) =
            write_single_dense_binary_field(0, &values, values.len() as i32, &id, "").unwrap();

        let fis = binary_field_infos();
        let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
        let entry = meta.binary_entry(0).unwrap();
        assert!(!entry.is_fixed_length());
        for (doc, want) in values.iter().enumerate() {
            assert_eq!(
                binary_value(&data_bytes, entry, doc as i32).unwrap(),
                Some(want.as_slice())
            );
        }
    }

    #[test]
    fn write_single_dense_binary_field_rejects_non_dense_value_count() {
        let id = [1u8; ID_LENGTH];
        let values: Vec<Vec<u8>> = vec![b"a".to_vec(), b"b".to_vec()];
        let err = write_single_dense_binary_field(0, &values, 5, &id, "").unwrap_err();
        assert!(matches!(
            err,
            WriteError::NotDense {
                values: 2,
                max_doc: 5
            }
        ));
    }

    #[test]
    fn write_single_dense_sorted_numeric_field_multi_valued_round_trips() {
        let id = [12u8; ID_LENGTH];
        let values: Vec<Vec<i64>> = vec![vec![1], vec![2, 3], vec![4, 5, 6], vec![0]];
        let (meta_bytes, data_bytes, skip_bytes) =
            write_single_dense_sorted_numeric_field(0, &values, &id, "").unwrap();

        assert_eq!(
            check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id)
                .unwrap(),
            VERSION_CURRENT
        );
        assert_eq!(
            check_data_header_footer(&data_bytes, &id, "").unwrap(),
            VERSION_CURRENT
        );

        let fis = sorted_numeric_field_infos();
        let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
        let entry = meta.sorted_numeric_entry(0).unwrap();
        assert!(entry.addresses.is_some());
        for (doc, want) in values.iter().enumerate() {
            assert_eq!(
                sorted_numeric_values(&data_bytes, entry, doc as i32).unwrap(),
                *want
            );
        }
    }

    #[test]
    fn write_single_dense_sorted_numeric_field_collapses_to_no_addresses_when_all_single_valued() {
        let id = [13u8; ID_LENGTH];
        let values: Vec<Vec<i64>> = vec![vec![10], vec![20], vec![30]];
        let (meta_bytes, data_bytes, _skip_bytes) =
            write_single_dense_sorted_numeric_field(0, &values, &id, "").unwrap();

        let fis = sorted_numeric_field_infos();
        let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
        let entry = meta.sorted_numeric_entry(0).unwrap();
        // Every doc has exactly 1 value -> the read side infers no address
        // array exists at all (num_docs_with_field == numeric.num_values).
        assert!(entry.addresses.is_none());
        for (doc, want) in values.iter().enumerate() {
            assert_eq!(
                sorted_numeric_values(&data_bytes, entry, doc as i32).unwrap(),
                *want
            );
        }
    }

    #[test]
    fn write_single_dense_sorted_numeric_field_all_same_value() {
        let id = [14u8; ID_LENGTH];
        let values: Vec<Vec<i64>> = vec![vec![7, 7], vec![7, 7], vec![7, 7]];
        let (meta_bytes, data_bytes, _skip_bytes) =
            write_single_dense_sorted_numeric_field(0, &values, &id, "").unwrap();

        let fis = sorted_numeric_field_infos();
        let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
        let entry = meta.sorted_numeric_entry(0).unwrap();
        assert_eq!(entry.numeric.bits_per_value, 0); // constant-value encoding
        for doc in 0..values.len() as i32 {
            assert_eq!(
                sorted_numeric_values(&data_bytes, entry, doc).unwrap(),
                vec![7, 7]
            );
        }
    }

    #[test]
    fn write_single_dense_sorted_numeric_field_rejects_empty_doc() {
        let id = [1u8; ID_LENGTH];
        let values: Vec<Vec<i64>> = vec![vec![1], Vec::new(), vec![2]];
        let err = write_single_dense_sorted_numeric_field(0, &values, &id, "").unwrap_err();
        assert!(matches!(err, WriteError::EmptyMultiValuedDoc(1)));
    }

    // --- write_single_dense_sorted_field / write_terms_dict ---

    fn sorted_field_infos() -> FieldInfos {
        let mut fis = field_infos_with(&[0]);
        fis.fields[0].doc_values_type = DocValuesType::Sorted;
        fis
    }

    fn read_sorted_field(meta_bytes: &[u8], id: &[u8; ID_LENGTH], fis: &FieldInfos) -> SortedEntry {
        let (_, meta) = parse_meta(meta_bytes, id, "", fis).unwrap();
        meta.sorted_entry(0).unwrap().clone()
    }

    fn resolved_sorted_values(
        data_bytes: &[u8],
        entry: &SortedEntry,
        max_doc: i32,
    ) -> Vec<Option<Vec<u8>>> {
        let dict = terms_dict::decode_all_terms(data_bytes, &entry.terms).unwrap();
        (0..max_doc)
            .map(|doc| {
                sorted_ord(data_bytes, entry, doc)
                    .unwrap()
                    .map(|ord| dict[ord as usize].clone())
            })
            .collect()
    }

    #[test]
    fn write_single_dense_sorted_field_round_trips_small_dictionary() {
        let id = [20u8; ID_LENGTH];
        let values: Vec<Vec<u8>> = vec![
            b"banana".to_vec(),
            b"apple".to_vec(),
            b"cherry".to_vec(),
            b"apple".to_vec(),
        ];
        let (meta_bytes, data_bytes, skip_bytes) =
            write_single_dense_sorted_field(0, &values, values.len() as i32, &id, "").unwrap();

        assert_eq!(
            check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id)
                .unwrap(),
            VERSION_CURRENT
        );
        assert_eq!(
            check_data_header_footer(&data_bytes, &id, "").unwrap(),
            VERSION_CURRENT
        );

        let fis = sorted_field_infos();
        let entry = read_sorted_field(&meta_bytes, &id, &fis);
        let dict = terms_dict::decode_all_terms(&data_bytes, &entry.terms).unwrap();
        assert_eq!(
            dict,
            vec![b"apple".to_vec(), b"banana".to_vec(), b"cherry".to_vec()]
        );

        let resolved = resolved_sorted_values(&data_bytes, &entry, values.len() as i32);
        assert_eq!(resolved, values.into_iter().map(Some).collect::<Vec<_>>());
    }

    #[test]
    fn write_single_dense_sorted_field_all_docs_share_one_value() {
        let id = [21u8; ID_LENGTH];
        let values: Vec<Vec<u8>> = vec![b"same".to_vec(); 5];
        let (meta_bytes, data_bytes, _skip_bytes) =
            write_single_dense_sorted_field(0, &values, values.len() as i32, &id, "").unwrap();

        let fis = sorted_field_infos();
        let entry = read_sorted_field(&meta_bytes, &id, &fis);
        assert_eq!(entry.ords.bits_per_value, 0); // constant-ordinal encoding
        let resolved = resolved_sorted_values(&data_bytes, &entry, values.len() as i32);
        assert_eq!(resolved, values.into_iter().map(Some).collect::<Vec<_>>());
    }

    #[test]
    fn write_single_dense_sorted_field_empty_bytes_value_is_a_present_value() {
        let id = [22u8; ID_LENGTH];
        let values: Vec<Vec<u8>> = vec![Vec::new(), b"x".to_vec(), Vec::new()];
        let (meta_bytes, data_bytes, _skip_bytes) =
            write_single_dense_sorted_field(0, &values, values.len() as i32, &id, "").unwrap();

        let fis = sorted_field_infos();
        let entry = read_sorted_field(&meta_bytes, &id, &fis);
        let resolved = resolved_sorted_values(&data_bytes, &entry, values.len() as i32);
        assert_eq!(resolved, values.into_iter().map(Some).collect::<Vec<_>>());
    }

    #[test]
    fn write_single_dense_sorted_field_large_dictionary_spans_many_lz4_blocks_and_index_samples() {
        // 300 distinct terms: exercises multiple 64-term LZ4 blocks (4 full +
        // 1 partial). Only one reverse-index sample (ord == 0) fires at this
        // size -- the multi-sample case is covered separately below.
        let id = [23u8; ID_LENGTH];
        let mut values: Vec<Vec<u8>> = (0..300)
            .map(|i: i32| format!("term{i:04}").into_bytes())
            .collect();
        // Make the dictionary genuinely sorted-ascending-as-strings by
        // reusing the zero-padded formatting above, then also shuffle the
        // per-doc assignment so ordinals aren't just doc id.
        values.reverse();
        let (meta_bytes, data_bytes, _skip_bytes) =
            write_single_dense_sorted_field(0, &values, values.len() as i32, &id, "").unwrap();

        let fis = sorted_field_infos();
        let entry = read_sorted_field(&meta_bytes, &id, &fis);
        let dict = terms_dict::decode_all_terms(&data_bytes, &entry.terms).unwrap();
        assert_eq!(dict.len(), 300);
        assert!(dict.windows(2).all(|w| w[0] < w[1]));

        let resolved = resolved_sorted_values(&data_bytes, &entry, values.len() as i32);
        assert_eq!(resolved, values.into_iter().map(Some).collect::<Vec<_>>());
    }

    #[test]
    fn write_single_dense_sorted_field_reverse_index_multi_sample_boundary() {
        // TERMS_DICT_REVERSE_INDEX_MASK is 1023 (sample every 1024th
        // ordinal), so 300 terms only ever hits the `ord == 0` sample.
        // 2200 distinct terms crosses two sample boundaries (ord 1024,
        // 2048), exercising the `ord & MASK == MASK` capture and the
        // `common_prefix_len(&sampled_previous, term) + 1` branch for a
        // sample past the first -- both previously untested.
        let id = [24u8; ID_LENGTH];
        let mut values: Vec<Vec<u8>> = (0..2200)
            .map(|i: i32| format!("term{i:05}").into_bytes())
            .collect();
        values.reverse();
        let (meta_bytes, data_bytes, _skip_bytes) =
            write_single_dense_sorted_field(0, &values, values.len() as i32, &id, "").unwrap();

        let fis = sorted_field_infos();
        let entry = read_sorted_field(&meta_bytes, &id, &fis);
        let dict = terms_dict::decode_all_terms(&data_bytes, &entry.terms).unwrap();
        assert_eq!(dict.len(), 2200);
        assert!(dict.windows(2).all(|w| w[0] < w[1]));

        let resolved = resolved_sorted_values(&data_bytes, &entry, values.len() as i32);
        assert_eq!(resolved, values.into_iter().map(Some).collect::<Vec<_>>());
    }

    #[test]
    fn write_single_dense_sorted_field_rejects_non_dense_value_count() {
        let id = [1u8; ID_LENGTH];
        let values: Vec<Vec<u8>> = vec![b"a".to_vec()];
        let err = write_single_dense_sorted_field(0, &values, 2, &id, "").unwrap_err();
        assert!(matches!(
            err,
            WriteError::NotDense {
                values: 1,
                max_doc: 2
            }
        ));
    }

    // --- write_single_dense_sorted_set_field ---

    fn sorted_set_field_infos() -> FieldInfos {
        let mut fis = field_infos_with(&[0]);
        fis.fields[0].doc_values_type = DocValuesType::SortedSet;
        fis
    }

    fn read_sorted_set_field(
        meta_bytes: &[u8],
        id: &[u8; ID_LENGTH],
        fis: &FieldInfos,
    ) -> SortedSetEntry {
        let (_, meta) = parse_meta(meta_bytes, id, "", fis).unwrap();
        meta.sorted_set_entry(0).unwrap().clone()
    }

    /// Resolves every doc's full (sorted, deduped) value set, regardless of
    /// whether the entry collapsed to [`SortedSetKind::Single`] or stayed
    /// [`SortedSetKind::Multi`].
    fn resolved_sorted_set_values(
        data_bytes: &[u8],
        entry: &SortedSetEntry,
        max_doc: i32,
    ) -> Vec<Vec<Vec<u8>>> {
        match &entry.kind {
            SortedSetKind::Single(sorted) => {
                let dict = terms_dict::decode_all_terms(data_bytes, &sorted.terms).unwrap();
                (0..max_doc)
                    .map(|doc| {
                        sorted_ord(data_bytes, sorted, doc)
                            .unwrap()
                            .map(|ord| vec![dict[ord as usize].clone()])
                            .unwrap_or_default()
                    })
                    .collect()
            }
            SortedSetKind::Multi { ords, terms } => {
                let dict = terms_dict::decode_all_terms(data_bytes, terms).unwrap();
                (0..max_doc)
                    .map(|doc| {
                        sorted_numeric_values(data_bytes, ords, doc)
                            .unwrap()
                            .into_iter()
                            .map(|ord| dict[ord as usize].clone())
                            .collect()
                    })
                    .collect()
            }
        }
    }

    #[test]
    fn write_single_dense_sorted_set_field_small_dictionary_shared_across_docs() {
        let id = [30u8; ID_LENGTH];
        let values: Vec<Vec<Vec<u8>>> = vec![
            vec![b"apple".to_vec(), b"cherry".to_vec()],
            vec![b"banana".to_vec()],
            vec![b"apple".to_vec(), b"banana".to_vec(), b"cherry".to_vec()],
            vec![b"banana".to_vec()],
        ];
        let (meta_bytes, data_bytes, skip_bytes) =
            write_single_dense_sorted_set_field(0, &values, values.len() as i32, &id, "").unwrap();

        assert_eq!(
            check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id)
                .unwrap(),
            VERSION_CURRENT
        );
        assert_eq!(
            check_data_header_footer(&data_bytes, &id, "").unwrap(),
            VERSION_CURRENT
        );

        let fis = sorted_set_field_infos();
        let entry = read_sorted_set_field(&meta_bytes, &id, &fis);
        assert!(matches!(entry.kind, SortedSetKind::Multi { .. }));

        let resolved = resolved_sorted_set_values(&data_bytes, &entry, values.len() as i32);
        let want: Vec<Vec<Vec<u8>>> = values
            .into_iter()
            .map(|mut v| {
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();
        assert_eq!(resolved, want);
    }

    #[test]
    fn write_single_dense_sorted_set_field_all_docs_single_value_collapses_to_no_addresses() {
        let id = [31u8; ID_LENGTH];
        let values: Vec<Vec<Vec<u8>>> = vec![vec![b"same".to_vec()]; 5];
        let (meta_bytes, data_bytes, _skip_bytes) =
            write_single_dense_sorted_set_field(0, &values, values.len() as i32, &id, "").unwrap();

        let fis = sorted_set_field_infos();
        let entry = read_sorted_set_field(&meta_bytes, &id, &fis);
        match &entry.kind {
            SortedSetKind::Single(sorted) => {
                assert_eq!(sorted.ords.bits_per_value, 0); // constant-ordinal encoding
            }
            SortedSetKind::Multi { .. } => panic!("expected Single collapse"),
        }

        let resolved = resolved_sorted_set_values(&data_bytes, &entry, values.len() as i32);
        assert_eq!(resolved, values);
    }

    #[test]
    fn write_single_dense_sorted_set_field_intra_doc_duplicate_raw_values_collapse_per_doc() {
        // A doc's raw input value-set can itself contain duplicates (e.g. a
        // caller indexed the same value twice) -- these must collapse to one
        // ordinal reference per distinct value, same as the dictionary
        // itself is deduplicated at the field level. Doc 0 references
        // "same" three times raw (must resolve to exactly one ordinal); doc
        // 1 has two genuinely distinct values, which is what forces this
        // field to stay a true Multi (a doc referencing only one distinct
        // value, however many times raw, would otherwise collapse the
        // whole field to Single -- proving that distinction independently
        // of the dedicated Single-collapse test above, which never feeds
        // any raw duplicates at all).
        let id = [34u8; ID_LENGTH];
        let values: Vec<Vec<Vec<u8>>> = vec![
            vec![b"same".to_vec(), b"same".to_vec(), b"same".to_vec()],
            vec![b"other".to_vec(), b"third".to_vec()],
        ];
        let (meta_bytes, data_bytes, _skip_bytes) =
            write_single_dense_sorted_set_field(0, &values, values.len() as i32, &id, "").unwrap();

        let fis = sorted_set_field_infos();
        let entry = read_sorted_set_field(&meta_bytes, &id, &fis);
        assert!(matches!(entry.kind, SortedSetKind::Multi { .. }));

        let resolved = resolved_sorted_set_values(&data_bytes, &entry, values.len() as i32);
        assert_eq!(
            resolved,
            vec![
                vec![b"same".to_vec()],
                vec![b"other".to_vec(), b"third".to_vec()],
            ]
        );
    }

    #[test]
    fn write_single_dense_sorted_set_field_varying_value_counts_per_doc() {
        let id = [32u8; ID_LENGTH];
        let values: Vec<Vec<Vec<u8>>> = vec![
            vec![b"a".to_vec()],
            vec![b"a".to_vec(), b"b".to_vec()],
            vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()],
            vec![b"c".to_vec()],
        ];
        let (meta_bytes, data_bytes, _skip_bytes) =
            write_single_dense_sorted_set_field(0, &values, values.len() as i32, &id, "").unwrap();

        let fis = sorted_set_field_infos();
        let entry = read_sorted_set_field(&meta_bytes, &id, &fis);
        assert!(matches!(entry.kind, SortedSetKind::Multi { .. }));

        let resolved = resolved_sorted_set_values(&data_bytes, &entry, values.len() as i32);
        assert_eq!(resolved, values);
    }

    #[test]
    fn write_single_dense_sorted_set_field_large_dictionary_spans_many_lz4_blocks_and_multiple_reverse_index_samples(
    ) {
        // 2200 distinct terms: crosses many 64-term LZ4 blocks and TWO
        // non-trivial 1024-ordinal reverse-index sample boundaries (ord
        // 1024 and ord 2048) -- matching the SORTED large-dictionary test's
        // own term count, chosen there (and reused here) specifically
        // because a smaller count (e.g. 2000) only reaches the first
        // non-trivial sample (ord 1024), never the second. Each doc gets
        // exactly 2 values so the entry stays a true Multi (never collapses
        // to Single).
        let id = [33u8; ID_LENGTH];
        let mut terms: Vec<Vec<u8>> = (0..2200)
            .map(|i: i32| format!("term{i:05}").into_bytes())
            .collect();
        terms.reverse();

        let values: Vec<Vec<Vec<u8>>> = terms.chunks(2).map(|pair| pair.to_vec()).collect();

        let (meta_bytes, data_bytes, _skip_bytes) =
            write_single_dense_sorted_set_field(0, &values, values.len() as i32, &id, "").unwrap();

        let fis = sorted_set_field_infos();
        let entry = read_sorted_set_field(&meta_bytes, &id, &fis);
        let terms_entry = match &entry.kind {
            SortedSetKind::Multi { terms, .. } => terms,
            SortedSetKind::Single(_) => panic!("expected Multi"),
        };
        let dict = terms_dict::decode_all_terms(&data_bytes, terms_entry).unwrap();
        assert_eq!(dict.len(), 2200);
        assert!(dict.windows(2).all(|w| w[0] < w[1]));

        let resolved = resolved_sorted_set_values(&data_bytes, &entry, values.len() as i32);
        let want: Vec<Vec<Vec<u8>>> = values
            .into_iter()
            .map(|mut v| {
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();
        assert_eq!(resolved, want);
    }

    #[test]
    fn write_single_dense_sorted_set_field_rejects_empty_doc_value_set() {
        let id = [1u8; ID_LENGTH];
        let values: Vec<Vec<Vec<u8>>> = vec![vec![b"a".to_vec()], Vec::new(), vec![b"b".to_vec()]];
        let err = write_single_dense_sorted_set_field(0, &values, values.len() as i32, &id, "")
            .unwrap_err();
        assert!(matches!(err, WriteError::EmptyMultiValuedDoc(1)));
    }

    #[test]
    fn write_single_dense_sorted_set_field_rejects_non_dense_value_count() {
        let id = [1u8; ID_LENGTH];
        let values: Vec<Vec<Vec<u8>>> = vec![vec![b"a".to_vec()]];
        let err = write_single_dense_sorted_set_field(0, &values, 2, &id, "").unwrap_err();
        assert!(matches!(
            err,
            WriteError::NotDense {
                values: 1,
                max_doc: 2
            }
        ));
    }
}
