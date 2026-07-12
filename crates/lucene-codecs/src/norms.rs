//! Port of `org.apache.lucene.codecs.lucene90.Lucene90NormsFormat` (`.nvm`
//! metadata + `.nvd` data) — read-only.
//!
//! Norms are a per-field, per-doc score-normalization value (one integer of
//! 0/1/2/4/8 bytes, depending on the range needed). Three shapes exist,
//! selected by `docs_with_field_offset`:
//! - **empty** (`-2`): no document has this field indexed at all.
//! - **dense** (`-1`): every doc up to `maxDoc` has a value — a flat array.
//! - **sparse** (`>= 0`): only some docs have a value, addressed through an
//!   `IndexedDISI` bitset (see [`crate::indexed_disi`]) giving each present
//!   doc's ordinal, which indexes the same flat value array dense fields use.
//!
//! Wire format, `.nvm` (little-endian throughout — no vints, unlike most
//! other formats; header/footer per `codec_util`):
//! ```text
//! IndexHeader(codec="Lucene90NormsMetadata", version=0, id, suffix)
//! per field (terminated by FieldNumber == -1):
//!   FieldNumber          --> i32
//!   DocsWithFieldOffset  --> i64  (-2 empty, -1 dense, >=0 sparse offset into .nvd)
//!   DocsWithFieldLength  --> i64  (sparse bitset length in .nvd, meaningless if not sparse)
//!   JumpTableEntryCount  --> i16
//!   DenseRankPower       --> u8
//!   NumDocsWithField     --> i32
//!   BytesPerNorm         --> u8  (must be one of 0, 1, 2, 4, 8)
//!   NormsOffset          --> i64  (offset into .nvd, or the single constant
//!                            value itself when BytesPerNorm == 0)
//! Footer
//! ```
//!
//! `.nvd` is just `IndexHeader, <raw bytes>, Footer`; dense values for a
//! field live at `NormsOffset + doc * BytesPerNorm`, little-endian,
//! sign-extended to i64 (matching `RandomAccessInput.readByte/Short/Int/Long`).

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

use crate::indexed_disi;

const DATA_CODEC: &str = "Lucene90NormsData";
const METADATA_CODEC: &str = "Lucene90NormsMetadata";
const VERSION_START: i32 = 0;
const VERSION_CURRENT: i32 = 0;

const DOCS_WITH_FIELD_EMPTY: i64 = -2;
const DOCS_WITH_FIELD_DENSE: i64 = -1;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("invalid bytesPerValue: {0}, field number {1}")]
    InvalidBytesPerNorm(u8, i32),
    #[error("doc {0} is out of range (numDocsWithField={1})")]
    DocOutOfRange(i32, i32),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    #[error(
        "write_single_dense_field requires values.len() == max_doc (every doc must have a value); got {values} values for max_doc={max_doc}"
    )]
    NotDense { values: usize, max_doc: i32 },
    #[error(
        "value range [{min}, {max}] for field {field_number} needs more than 1 byte per norm; only the constant (0-byte) and 1-byte-per-doc cases are supported by this writer"
    )]
    RangeTooWide {
        field_number: i32,
        min: i64,
        max: i64,
    },
}

pub type WriteResult<T> = std::result::Result<T, WriteError>;

#[derive(Debug, Clone, Copy)]
pub struct NormsEntry {
    pub field_number: i32,
    pub docs_with_field_offset: i64,
    pub docs_with_field_length: i64,
    pub jump_table_entry_count: i16,
    pub dense_rank_power: u8,
    pub num_docs_with_field: i32,
    pub bytes_per_norm: u8,
    pub norms_offset: i64,
}

impl NormsEntry {
    pub fn is_empty_field(&self) -> bool {
        self.docs_with_field_offset == DOCS_WITH_FIELD_EMPTY
    }

    pub fn is_dense(&self) -> bool {
        self.docs_with_field_offset == DOCS_WITH_FIELD_DENSE
    }
}

#[derive(Debug, Clone)]
pub struct Norms {
    pub entries: Vec<NormsEntry>,
}

impl Norms {
    pub fn entry(&self, field_number: i32) -> Option<&NormsEntry> {
        self.entries.iter().find(|e| e.field_number == field_number)
    }
}

/// Parses a whole `.nvm` metadata file already read into memory.
pub fn parse_meta(
    buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<(i32, Norms)> {
    let mut input = SliceInput::new(buf);
    let header = codec_util::check_index_header(
        &mut input,
        METADATA_CODEC,
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
        let docs_with_field_offset = input.read_i64()?;
        let docs_with_field_length = input.read_i64()?;
        let jump_table_entry_count = input.read_i16()?;
        let dense_rank_power = input.read_byte()?;
        let num_docs_with_field = input.read_i32()?;
        let bytes_per_norm = input.read_byte()?;
        if !matches!(bytes_per_norm, 0 | 1 | 2 | 4 | 8) {
            return Err(Error::InvalidBytesPerNorm(bytes_per_norm, field_number));
        }
        let norms_offset = input.read_i64()?;

        entries.push(NormsEntry {
            field_number,
            docs_with_field_offset,
            docs_with_field_length,
            jump_table_entry_count,
            dense_rank_power,
            num_docs_with_field,
            bytes_per_norm,
            norms_offset,
        });
    }

    codec_util::check_footer(&mut input, buf.len())?;

    Ok((header.version, entries_to_norms(entries)))
}

fn entries_to_norms(entries: Vec<NormsEntry>) -> Norms {
    Norms { entries }
}

/// Validates a whole `.nvd` data file's header/footer (does not decode the
/// per-field regions, which are addressed by absolute offset from `.nvm`
/// entries and have no self-describing structure of their own beyond that).
/// Returns the format version for cross-checking against the meta file's.
pub fn check_data_header_footer(
    buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<i32> {
    let mut input = SliceInput::new(buf);
    let header = codec_util::check_index_header(
        &mut input,
        DATA_CODEC,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    // Norms data files are only checksum-validated structurally on open in
    // Lucene (full-file CRC is too costly for a forward-only read pattern);
    // mirror that by only requiring the footer to *exist* and be
    // well-formed, not that we've read every byte up to it.
    codec_util::retrieve_checksum(buf)?;
    Ok(header.version)
}

/// Reads the norm value for `doc`, handling all three shapes (empty, dense,
/// sparse). `data` is the whole `.nvd` file's bytes. Returns `Ok(None)` when
/// `doc` legitimately has no norm (an empty field, or a doc a sparse field
/// skips) — that is normal, not an error; only a truly out-of-range `doc` or
/// a decode failure is `Err`.
pub fn norm_value(data: &[u8], entry: &NormsEntry, doc: i32) -> Result<Option<i64>> {
    if doc < 0 {
        return Err(Error::DocOutOfRange(doc, entry.num_docs_with_field));
    }
    if entry.is_empty_field() {
        return Ok(None);
    }
    if entry.is_dense() {
        if doc >= entry.num_docs_with_field {
            return Err(Error::DocOutOfRange(doc, entry.num_docs_with_field));
        }
        return Ok(Some(read_value_at_ordinal(data, entry, doc as i64)?));
    }

    // Sparse: docs_with_field_offset/length address an IndexedDISI region.
    let region = data
        .get(
            entry.docs_with_field_offset as usize
                ..(entry.docs_with_field_offset + entry.docs_with_field_length) as usize,
        )
        .ok_or(lucene_store::Error::Eof { offset: 0 })?;
    let doc_ids = indexed_disi::decode_doc_ids(region, entry.dense_rank_power)?;
    match indexed_disi::rank_of(&doc_ids, doc) {
        Some(ordinal) => Ok(Some(read_value_at_ordinal(data, entry, ordinal as i64)?)),
        None => Ok(None),
    }
}

/// Reads the norm value at `ordinal` (either the doc id itself for a dense
/// field, or the doc's rank among docs-with-a-value for a sparse one) — both
/// index the same flat `NormsOffset + ordinal * BytesPerNorm` array shape.
fn read_value_at_ordinal(data: &[u8], entry: &NormsEntry, ordinal: i64) -> Result<i64> {
    if entry.bytes_per_norm == 0 {
        // A single constant value for every doc, encoded directly in the
        // offset field rather than a separate array.
        return Ok(entry.norms_offset);
    }

    let offset = entry.norms_offset + ordinal * (entry.bytes_per_norm as i64);
    let mut input = SliceInput::new(data);
    input.seek(offset as usize)?;
    let value = match entry.bytes_per_norm {
        1 => input.read_byte()? as i8 as i64,
        2 => input.read_i16()? as i64,
        4 => input.read_i32()? as i64,
        8 => input.read_i64()?,
        // Already validated in `parse_meta`.
        _ => unreachable!("bytesPerNorm validated to be one of 0,1,2,4,8"),
    };
    Ok(value)
}

/// Port of `Lucene90NormsConsumer.addNormsField`, scoped to exactly one
/// shape: **a single norms field, DENSE (every doc from `0` to `max_doc - 1`
/// has a value), encoded in at most 1 byte per doc** -- the
/// `numDocsWithValue == maxDoc` branch of `addNormsField` (so no
/// `IndexedDISI` bitset is ever written) feeding `numBytesPerValue`'s `0`
/// (all values equal, `min >= max`) or `1` (`Byte.MIN_VALUE..=Byte.MAX_VALUE`)
/// cases only.
///
/// Deliberately not attempted here, all deferred to future slices (see
/// `docs/parity.md`): sparse fields (`IndexedDISI`, the `numDocsWithValue !=
/// maxDoc` branch), 2/4/8-byte-per-doc widths (returned as
/// [`WriteError::RangeTooWide`] instead of silently truncating), and multiple
/// fields in one `.nvm`/`.nvd` pair.
///
/// Returns `(meta_bytes, data_bytes)` matching the real writer's two
/// `IndexOutput`s (`.nvm`, `.nvd`); unlike doc values, norms have no third
/// (`.dvs`-style) file.
pub fn write_single_dense_field(
    field_number: i32,
    values: &[i64],
    max_doc: i32,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> WriteResult<(Vec<u8>, Vec<u8>)> {
    if values.len() != max_doc as usize {
        return Err(WriteError::NotDense {
            values: values.len(),
            max_doc,
        });
    }

    let mut meta: Vec<u8> = Vec::new();
    codec_util::write_index_header(
        &mut meta,
        METADATA_CODEC,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );

    let mut data: Vec<u8> = Vec::new();
    codec_util::write_index_header(
        &mut data,
        DATA_CODEC,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );

    meta.write_i32(field_number);

    // `numDocsWithValue == maxDoc`: the dense case, meta[-1, 0], no
    // IndexedDISI structure (see `addNormsField`'s middle branch).
    meta.write_i64(DOCS_WITH_FIELD_DENSE);
    meta.write_i64(0);
    meta.write_i16(-1); // jumpTableEntryCount
    meta.push(0xFF); // denseRankPower (-1 as u8)

    let num_docs_with_value = values.len() as i32;
    meta.write_i32(num_docs_with_value);

    let min = values.iter().copied().min().unwrap_or(i64::MAX);
    let max = values.iter().copied().max().unwrap_or(i64::MIN);

    let num_bytes_per_value: u8 = if min >= max {
        0
    } else if min >= i8::MIN as i64 && max <= i8::MAX as i64 {
        1
    } else {
        return Err(WriteError::RangeTooWide {
            field_number,
            min,
            max,
        });
    };

    meta.push(num_bytes_per_value);
    if num_bytes_per_value == 0 {
        meta.write_i64(min);
    } else {
        meta.write_i64(data.len() as i64); // normsOffset
        for &v in values {
            data.push(v as i8 as u8);
        }
    }

    meta.write_i32(-1); // field list terminator
    codec_util::write_footer(&mut meta);
    codec_util::write_footer(&mut data);

    Ok((meta, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only `.nvm`/`.nvd` byte builder, independent of the Java fixture
    /// under `tests/norms_fixtures.rs` (which exercises a real IndexWriter's
    /// output): this covers error/edge paths — invalid bytesPerNorm, empty/
    /// sparse fields, out-of-range docs, and each of the four nonzero byte
    /// widths — that a single realistic fixture doesn't naturally hit all of.
    struct EntryBuilder {
        field_number: i32,
        docs_with_field_offset: i64,
        docs_with_field_length: i64,
        jump_table_entry_count: i16,
        dense_rank_power: u8,
        num_docs_with_field: i32,
        bytes_per_norm: u8,
        norms_offset: i64,
    }

    impl EntryBuilder {
        fn dense(field_number: i32, bytes_per_norm: u8, num_docs: i32, norms_offset: i64) -> Self {
            Self {
                field_number,
                docs_with_field_offset: DOCS_WITH_FIELD_DENSE,
                docs_with_field_length: 0,
                jump_table_entry_count: 0,
                dense_rank_power: 0,
                num_docs_with_field: num_docs,
                bytes_per_norm,
                norms_offset,
            }
        }

        fn build(&self, out: &mut Vec<u8>) {
            out.extend_from_slice(&self.field_number.to_le_bytes());
            out.extend_from_slice(&self.docs_with_field_offset.to_le_bytes());
            out.extend_from_slice(&self.docs_with_field_length.to_le_bytes());
            out.extend_from_slice(&self.jump_table_entry_count.to_le_bytes());
            out.push(self.dense_rank_power);
            out.extend_from_slice(&self.num_docs_with_field.to_le_bytes());
            out.push(self.bytes_per_norm);
            out.extend_from_slice(&self.norms_offset.to_le_bytes());
        }
    }

    fn build_nvm(id: &[u8; ID_LENGTH], entries: &[EntryBuilder]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut out, METADATA_CODEC);
        out.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        out.extend_from_slice(id);
        out.push(0); // empty suffix
        for e in entries {
            e.build(&mut out);
        }
        out.extend_from_slice(&(-1i32).to_le_bytes()); // terminator
        out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&out) as u64;
        out.extend_from_slice(&checksum.to_be_bytes());
        out
    }

    fn build_nvd(id: &[u8; ID_LENGTH], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut out, DATA_CODEC);
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

    /// Length of the index header a `.nvm`/`.nvd` file starts with, for a
    /// given codec name and this module's fixed version/id/empty-suffix
    /// shape — used to compute absolute offsets for hand-built `.nvd` bytes.
    fn nvm_header_len(codec: &str) -> usize {
        4 + 1 + codec.len() + 4 + ID_LENGTH + 1 // magic + vint-len + name + version + id + suffix-len
    }

    #[test]
    fn empty_meta_parses_no_fields() {
        let id = [1u8; ID_LENGTH];
        let buf = build_nvm(&id, &[]);
        let (version, norms) = parse_meta(&buf, &id, "").unwrap();
        assert_eq!(version, 0);
        assert_eq!(norms.entries.len(), 0);
    }

    #[test]
    fn invalid_bytes_per_norm_rejected() {
        let id = [1u8; ID_LENGTH];
        let mut e = EntryBuilder::dense(0, 3, 5, 0); // 3 is not a valid width
        e.bytes_per_norm = 3;
        let buf = build_nvm(&id, &[e]);
        assert!(matches!(
            parse_meta(&buf, &id, ""),
            Err(Error::InvalidBytesPerNorm(3, 0))
        ));
    }

    #[test]
    fn empty_field_has_no_value_anywhere() {
        let id = [1u8; ID_LENGTH];
        let mut e = EntryBuilder::dense(0, 1, 0, 0);
        e.docs_with_field_offset = DOCS_WITH_FIELD_EMPTY;
        let buf = build_nvm(&id, &[e]);
        let (_, norms) = parse_meta(&buf, &id, "").unwrap();
        let entry = norms.entry(0).unwrap();
        assert!(entry.is_empty_field());
        assert_eq!(norm_value(&[], entry, 0).unwrap(), None);
    }

    #[test]
    fn doc_out_of_range_rejected() {
        let id = [1u8; ID_LENGTH];
        let e = EntryBuilder::dense(0, 1, 3, 0);
        let buf = build_nvm(&id, &[e]);
        let (_, norms) = parse_meta(&buf, &id, "").unwrap();
        let entry = norms.entry(0).unwrap();
        assert!(matches!(
            norm_value(&[0, 0, 0], entry, 3),
            Err(Error::DocOutOfRange(3, 3))
        ));
        assert!(matches!(
            norm_value(&[0, 0, 0], entry, -1),
            Err(Error::DocOutOfRange(-1, 3))
        ));
    }

    #[test]
    fn constant_value_when_bytes_per_norm_zero() {
        let id = [1u8; ID_LENGTH];
        let e = EntryBuilder::dense(0, 0, 5, 7); // constant value 7 for all docs
        let buf = build_nvm(&id, &[e]);
        let (_, norms) = parse_meta(&buf, &id, "").unwrap();
        let entry = norms.entry(0).unwrap();
        for doc in 0..5 {
            assert_eq!(norm_value(&[], entry, doc).unwrap(), Some(7));
        }
    }

    #[test]
    fn every_nonzero_byte_width_decodes_correctly() {
        let id = [1u8; ID_LENGTH];

        // width 1: value -5 at doc 0
        let payload1 = vec![(-5i8) as u8];
        let data = build_nvd(&id, &payload1);
        let header_len = nvm_header_len(DATA_CODEC);
        let e = EntryBuilder::dense(0, 1, 1, header_len as i64);
        assert_eq!(
            norm_value(&data, &to_entry(&e), 0).unwrap(),
            Some(-5),
            "width 1"
        );

        // width 2: value -300
        let mut payload2 = Vec::new();
        payload2.extend_from_slice(&(-300i16).to_le_bytes());
        let data = build_nvd(&id, &payload2);
        let e = EntryBuilder::dense(0, 2, 1, header_len as i64);
        assert_eq!(
            norm_value(&data, &to_entry(&e), 0).unwrap(),
            Some(-300),
            "width 2"
        );

        // width 4: value -70000
        let mut payload4 = Vec::new();
        payload4.extend_from_slice(&(-70000i32).to_le_bytes());
        let data = build_nvd(&id, &payload4);
        let e = EntryBuilder::dense(0, 4, 1, header_len as i64);
        assert_eq!(
            norm_value(&data, &to_entry(&e), 0).unwrap(),
            Some(-70000),
            "width 4"
        );

        // width 8: value i64::MIN
        let mut payload8 = Vec::new();
        payload8.extend_from_slice(&i64::MIN.to_le_bytes());
        let data = build_nvd(&id, &payload8);
        let e = EntryBuilder::dense(0, 8, 1, header_len as i64);
        assert_eq!(
            norm_value(&data, &to_entry(&e), 0).unwrap(),
            Some(i64::MIN),
            "width 8"
        );
    }

    #[test]
    fn sparse_field_returns_value_for_present_doc_and_none_for_absent() {
        // IndexedDISI region: a single SPARSE block covering docs [0,65536)
        // with docs 1 and 3 present, then the mandatory sentinel block.
        let mut disi_bytes = Vec::new();
        disi_bytes.extend_from_slice(&0u16.to_le_bytes()); // block 0
        disi_bytes.extend_from_slice(&1u16.to_le_bytes()); // numValues-1 = 1 (2 values)
        disi_bytes.extend_from_slice(&1u16.to_le_bytes()); // doc 1
        disi_bytes.extend_from_slice(&3u16.to_le_bytes()); // doc 3
        disi_bytes.extend_from_slice(&((i32::MAX >> 16) as u16).to_le_bytes());
        disi_bytes.extend_from_slice(&0u16.to_le_bytes()); // numValues-1 = 0 (1 value)
        disi_bytes.extend_from_slice(&((i32::MAX & 0xFFFF) as u16).to_le_bytes());

        // .nvd layout: [ disi_bytes ][ values: byte per present doc, in doc order ]
        let disi_offset = 0i64;
        let disi_length = disi_bytes.len() as i64;
        let mut data = disi_bytes.clone();
        data.push(11); // value for doc 1 (ordinal 0)
        data.push(33); // value for doc 3 (ordinal 1)

        let mut e = EntryBuilder::dense(0, 1, 2, disi_length); // norms right after the DISI region
        e.docs_with_field_offset = disi_offset;
        e.docs_with_field_length = disi_length;
        let entry = to_entry(&e);

        assert_eq!(norm_value(&data, &entry, 1).unwrap(), Some(11));
        assert_eq!(norm_value(&data, &entry, 3).unwrap(), Some(33));
        assert_eq!(norm_value(&data, &entry, 2).unwrap(), None);
    }

    fn to_entry(e: &EntryBuilder) -> NormsEntry {
        NormsEntry {
            field_number: e.field_number,
            docs_with_field_offset: e.docs_with_field_offset,
            docs_with_field_length: e.docs_with_field_length,
            jump_table_entry_count: e.jump_table_entry_count,
            dense_rank_power: e.dense_rank_power,
            num_docs_with_field: e.num_docs_with_field,
            bytes_per_norm: e.bytes_per_norm,
            norms_offset: e.norms_offset,
        }
    }

    #[test]
    fn check_data_header_footer_valid() {
        let id = [2u8; ID_LENGTH];
        let data = build_nvd(&id, b"payload-bytes");
        let version = check_data_header_footer(&data, &id, "").unwrap();
        assert_eq!(version, 0);
    }

    #[test]
    fn check_data_header_footer_wrong_id_rejected() {
        let id = [2u8; ID_LENGTH];
        let data = build_nvd(&id, b"payload-bytes");
        let wrong_id = [3u8; ID_LENGTH];
        assert!(check_data_header_footer(&data, &wrong_id, "").is_err());
    }

    #[test]
    fn wrong_id_rejected_on_meta() {
        let id = [1u8; ID_LENGTH];
        let buf = build_nvm(&id, &[]);
        let wrong_id = [9u8; ID_LENGTH];
        assert!(matches!(
            parse_meta(&buf, &wrong_id, ""),
            Err(Error::Store(_))
        ));
    }

    #[test]
    fn write_single_dense_field_round_trips_through_own_reader() {
        let id = [7u8; ID_LENGTH];
        let values = vec![5i64, -100, 0, 127, -128];
        let (meta_bytes, data_bytes) =
            write_single_dense_field(0, &values, values.len() as i32, &id, "").unwrap();

        let version = check_data_header_footer(&data_bytes, &id, "").unwrap();
        assert_eq!(version, VERSION_CURRENT);

        let (meta_version, norms) = parse_meta(&meta_bytes, &id, "").unwrap();
        assert_eq!(meta_version, VERSION_CURRENT);
        let entry = norms.entry(0).unwrap();
        assert!(entry.is_dense());
        assert_eq!(entry.bytes_per_norm, 1);
        for (doc, &want) in values.iter().enumerate() {
            assert_eq!(
                norm_value(&data_bytes, entry, doc as i32).unwrap(),
                Some(want)
            );
        }
    }

    #[test]
    fn write_single_dense_field_constant_values_uses_zero_byte_encoding() {
        let id = [8u8; ID_LENGTH];
        let values = vec![3i64; 4];
        let (meta_bytes, data_bytes) =
            write_single_dense_field(1, &values, values.len() as i32, &id, "").unwrap();

        let (_, norms) = parse_meta(&meta_bytes, &id, "").unwrap();
        let entry = norms.entry(1).unwrap();
        assert_eq!(entry.bytes_per_norm, 0);
        for doc in 0..values.len() as i32 {
            assert_eq!(norm_value(&data_bytes, entry, doc).unwrap(), Some(3));
        }
        // No per-doc array is written for the constant case.
        assert!(data_bytes.len() < nvm_header_len(DATA_CODEC) + values.len() + 16);
    }

    #[test]
    fn write_single_dense_field_rejects_non_dense_input() {
        let id = [9u8; ID_LENGTH];
        let values = vec![1i64, 2];
        assert!(matches!(
            write_single_dense_field(0, &values, 3, &id, ""),
            Err(WriteError::NotDense {
                values: 2,
                max_doc: 3
            })
        ));
    }

    #[test]
    fn write_single_dense_field_rejects_range_too_wide() {
        let id = [10u8; ID_LENGTH];
        let values = vec![0i64, 300];
        assert!(matches!(
            write_single_dense_field(0, &values, 2, &id, ""),
            Err(WriteError::RangeTooWide {
                field_number: 0,
                min: 0,
                max: 300,
            })
        ));
    }
}
