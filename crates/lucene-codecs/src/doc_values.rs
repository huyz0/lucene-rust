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
}

pub type WriteResult<T> = std::result::Result<T, WriteError>;

/// Writes just the NUMERIC entry body (`addNumericField` -> `writeValues`,
/// the `numDocsWithValue == maxDoc` / no-table / no-blocks branches, feeding
/// `writeValuesSingleBlock`) into an already-open meta/data pair -- shared by
/// [`write_single_dense_numeric_field`] (a standalone NUMERIC field) and
/// [`write_single_dense_sorted_numeric_field`] (whose per-doc value counts
/// collapse to this exact same flat layout). Does **not** write the leading
/// `field_number`/type byte -- callers that need those (a bare NUMERIC
/// field) write them first, exactly as [`read_numeric_entry`] expects them
/// already consumed by its caller.
///
/// Always dense (`docsWithFieldOffset = -1`, i.e. `values[i]` is doc `i`'s
/// value / doc `i`'s rank into a shared value array) and plain
/// delta-compressed (`gcd = 1`, no table compression, no varying-bits-per-
/// value blocks) -- see [`write_single_dense_numeric_field`]'s doc comment
/// for the full scope statement, which applies here identically.
fn write_dense_numeric_entry_body(meta: &mut Vec<u8>, data: &mut Vec<u8>, values: &[i64]) {
    // numDocsWithValue == maxDoc: meta[-1, 0], no IndexedDISI structure.
    meta.write_i64(DOCS_WITH_FIELD_DENSE);
    meta.write_i64(0);
    meta.write_i16(-1); // jumpTableEntryCount
    meta.push(0xFF); // denseRankPower (-1 as u8)

    let num_values = values.len() as i64;
    meta.write_i64(num_values);

    let min = values.iter().copied().min().unwrap_or(0);
    let max = values.iter().copied().max().unwrap_or(0);

    let (bits_per_value, min) = if min >= max {
        // All values equal (including the empty-values case, which can't
        // actually happen for a standalone NUMERIC field since
        // `values.len() == max_doc` and `max_doc` is a real field count, but
        // can for a SORTED_NUMERIC field's flat value array when every doc
        // has zero values -- Java's `min >= max` check covers both "all
        // equal" and "no values" the same way).
        (0u8, min)
    } else {
        let mut bpv = direct_reader::unsigned_bits_required(max - min);
        let mut min = min;
        // Java: if gcd==1 && min>0 && bits(max) == bits(max-min), drop the
        // min-shift (store raw values instead) since it doesn't save space.
        if min > 0 && direct_reader::unsigned_bits_required(max) == bpv {
            min = 0;
            bpv = direct_reader::unsigned_bits_required(max);
        }
        (bpv, min)
    };

    meta.write_i32(-1); // tableSize: no table compression in this slice
    meta.push(bits_per_value);
    meta.write_i64(min);
    meta.write_i64(1); // gcd: always 1, no GCD compression in this slice

    let start_offset = data.len() as i64;
    meta.write_i64(start_offset);

    if bits_per_value != 0 {
        let gcd = 1i64;
        let raw: Vec<i64> = values.iter().map(|&v| (v - min) / gcd).collect();
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
/// single NUMERIC field, DENSE (every doc from `0` to `max_doc - 1` has a
/// value), plain delta-compressed** (`bitsPerValue = unsignedBitsRequired(max
/// - min)`, `gcd = 1`) -- the `numDocsWithValue == maxDoc` branch of
/// `writeValues` followed by its `uniqueValues == null` (no table
/// compression attempted) and `doBlocks == false` (no varying-bits-per-value
/// blocks) branches, feeding `writeValuesSingleBlock`.
///
/// Deliberately not attempted here, all deferred to future slices (see
/// `docs/parity.md`): sparse fields (`IndexedDISI`), SORTED/SORTED_SET field
/// types (both need a terms-dictionary write side this port doesn't have
/// yet -- see [`crate::terms_dict`]'s parity row), GCD compression, table
/// compression, the varying-bits-per-value block split, per-field
/// doc-values skip indexes, and multiple fields in one `.dvm`/`.dvd`/`.dvs`
/// triple. BINARY ([`write_single_dense_binary_field`]) and SORTED_NUMERIC
/// ([`write_single_dense_sorted_numeric_field`]) write sides now exist as
/// siblings of this function, both built on this same dense/no-terms-dict
/// scope.
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
}
