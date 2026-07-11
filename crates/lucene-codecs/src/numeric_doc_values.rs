//! Port of `org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat`
//! (`.dvm` metadata + `.dvd` data) — read-only, **NUMERIC fields only**.
//!
//! Scope: the `.dvm` metadata stream interleaves entries for every doc-values
//! field in a segment regardless of type (numeric/binary/sorted/sorted-set/
//! sorted-numeric), and correctly walking past a field this port doesn't
//! decode requires knowing that type's own entry layout (binary needs monotonic
//! address blocks, sorted needs the whole LZ4 terms dictionary, etc). Rather
//! than half-implement those just to skip over their bytes, this module
//! requires every doc-values field in the segment to be `NUMERIC` with no
//! doc-values skip index configured — [`parse_meta`] returns
//! [`Error::UnsupportedFieldType`] / [`Error::UnsupportedSkipIndex`]
//! otherwise. Binary/sorted doc values are a separate future port.
//!
//! Also out of scope for now: "varying bits-per-value" blocks (Java splits
//! a field's values into 16384-value blocks with independently chosen
//! bits-per-value when that's a better fit — `blockShift >= 0` in the meta).
//! Small-to-medium fields Lucene doesn't bother splitting; reading one that
//! was split returns [`Error::UnsupportedVaryingBpvBlocks`].
//!
//! Three encodings for a field's per-doc values (`bitsPerValue` in the meta):
//! - **constant** (`bitsPerValue == 0`): every doc with a value has the same
//!   one, stored directly as `minValue`.
//! - **table-compressed**: a small (`<= 256` entry) lookup table of distinct
//!   values; each doc stores a bit-packed ordinal into it.
//! - **delta/GCD-compressed**: each doc stores a bit-packed `(value - min) /
//!   gcd`; `gcd == 1 && min == 0` is the common case (e.g. plain ordinals).
//!
//! As with [`crate::norms`], docs-with-a-value is one of empty/dense/sparse
//! (dense: implicit by doc id; sparse: via [`crate::indexed_disi`]), and
//! bit-packed values are read with [`direct_reader_get`], a from-scratch
//! generalization of Java's thirteen `DirectReader.DirectPackedReaderN`
//! classes into one bit-position formula (justified: those exist in Java to
//! give the JIT a monomorphic call site per width, a concern that doesn't
//! apply here since this port doesn't yet have a hot per-doc value loop).

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};

use crate::field_infos::FieldInfos;
use crate::indexed_disi;

const META_CODEC: &str = "Lucene90DocValuesMetadata";
const VERSION_START: i32 = 0;
const VERSION_SKIPPER_MAX_VALUE_COUNT: i32 = 2;
const VERSION_CURRENT: i32 = VERSION_SKIPPER_MAX_VALUE_COUNT;

const DOC_VALUES_TYPE_NUMERIC: u8 = 0;

const DOCS_WITH_FIELD_EMPTY: i64 = -2;
const DOCS_WITH_FIELD_DENSE: i64 = -1;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("unknown field number: {0}")]
    UnknownFieldNumber(i32),
    #[error("field {0} has doc values type byte {1}, expected NUMERIC (0)")]
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
pub struct NumericDocValuesMeta {
    pub entries: Vec<NumericEntry>,
}

impl NumericDocValuesMeta {
    pub fn entry(&self, field_number: i32) -> Option<&NumericEntry> {
        self.entries.iter().find(|e| e.field_number == field_number)
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
) -> Result<(i32, NumericDocValuesMeta)> {
    let mut input = SliceInput::new(buf);
    let header = codec_util::check_index_header(
        &mut input,
        META_CODEC,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;

    let mut entries = Vec::new();
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
        if ty != DOC_VALUES_TYPE_NUMERIC {
            return Err(Error::UnsupportedFieldType(field_number, ty));
        }

        entries.push(read_numeric_entry(&mut input, field_number)?);
    }

    codec_util::check_footer(&mut input, buf.len())?;

    Ok((header.version, NumericDocValuesMeta { entries }))
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
    let raw = direct_reader_get(values, entry.bits_per_value, ordinal)?;

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

/// Port of `org.apache.lucene.util.packed.DirectReader.getInstance(...).get(index)`,
/// generalized into a single bit-position formula instead of Java's thirteen
/// width-specialized classes (see module doc for why that's fine here).
/// `bits_per_value` must be one of the widths `DirectWriter` supports (the
/// caller validates this at parse time); `index` addresses the `index`-th
/// `bits_per_value`-wide value packed little-endian (LSB-first within each
/// byte) starting at byte 0 of `slice`.
fn direct_reader_get(slice: &[u8], bits_per_value: u8, index: i64) -> Result<i64> {
    let bit_pos = (index as u128) * (bits_per_value as u128);
    let byte_pos =
        usize::try_from(bit_pos >> 3).map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
    let shift = (bit_pos & 7) as u32;
    let bytes_needed = (shift as usize + bits_per_value as usize).div_ceil(8);

    let bytes = slice
        .get(byte_pos..byte_pos + bytes_needed)
        .ok_or(lucene_store::Error::Eof { offset: byte_pos })?;
    let mut acc: u64 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        acc |= (b as u64) << (8 * i);
    }
    acc >>= shift;
    let mask: u64 = if bits_per_value == 64 {
        u64::MAX
    } else {
        (1u64 << bits_per_value) - 1
    };
    Ok((acc & mask) as i64)
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
        assert_eq!(meta.entries.len(), 0);
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
        buf.push(1); // BINARY
        let fis = field_infos_with(&[0]);
        assert!(matches!(
            parse_meta(&buf, &id, "", &fis),
            Err(Error::UnsupportedFieldType(0, 1))
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
        let entry = meta.entry(0).unwrap();
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
        let entry = meta.entry(0).unwrap();
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
        let entry = meta.entry(0).unwrap();
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
        let entry = meta.entry(0).unwrap();

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
        let entry = meta.entry(0).unwrap();

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
        let entry = meta.entry(0).unwrap();

        let payload = [5u8, 250, 0];
        let data = build_dvd(&id, &payload);
        assert_eq!(numeric_value(&data, entry, 0).unwrap(), Some(5));
        assert_eq!(numeric_value(&data, entry, 1).unwrap(), Some(250));
        assert_eq!(numeric_value(&data, entry, 2).unwrap(), Some(0));
    }

    #[test]
    fn every_byte_aligned_width_round_trips() {
        // bpv=16: value 0x1234 at index 0, then 0xABCD at index 1
        let payload = [0x34, 0x12, 0xCD, 0xAB];
        assert_eq!(direct_reader_get(&payload, 16, 0).unwrap(), 0x1234);
        assert_eq!(direct_reader_get(&payload, 16, 1).unwrap(), 0xABCD);

        // bpv=32
        let payload = [0x01, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(direct_reader_get(&payload, 32, 0).unwrap(), 1);
        assert_eq!(direct_reader_get(&payload, 32, 1).unwrap(), 0xFFFFFFFF);

        // bpv=64: two's complement -1
        let payload = (-1i64).to_le_bytes();
        assert_eq!(direct_reader_get(&payload, 64, 0).unwrap(), -1);
    }

    #[test]
    fn sub_byte_widths_pack_multiple_values_per_byte() {
        // bpv=4: nibbles 0xA, 0xB packed into one byte 0xBA (low nibble first)
        let payload = [0xBA];
        assert_eq!(direct_reader_get(&payload, 4, 0).unwrap(), 0xA);
        assert_eq!(direct_reader_get(&payload, 4, 1).unwrap(), 0xB);

        // bpv=1: bits 1,0,1,1 packed LSB-first -> byte 0b0000_1101
        let payload = [0b0000_1101u8];
        assert_eq!(direct_reader_get(&payload, 1, 0).unwrap(), 1);
        assert_eq!(direct_reader_get(&payload, 1, 1).unwrap(), 0);
        assert_eq!(direct_reader_get(&payload, 1, 2).unwrap(), 1);
        assert_eq!(direct_reader_get(&payload, 1, 3).unwrap(), 1);
    }

    #[test]
    fn direct_reader_get_out_of_range_is_error() {
        let payload = [0u8; 1];
        assert!(direct_reader_get(&payload, 16, 5).is_err());
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
}
