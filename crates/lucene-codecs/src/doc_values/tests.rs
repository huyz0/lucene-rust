#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

use super::*;

/// Writes `fields` with a skip index on `skipped`, reads it back through
/// this module's reader, and returns it.
fn skip_round_trip(
    fields: &[DenseField<'_>],
    infos: Vec<crate::field_infos::FieldInfo>,
    skipped: i32,
    max_doc: i32,
) -> DocValuesSkipIndex {
    let id = [7u8; ID_LENGTH];
    let (dvm, _, dvs) =
        write_fields_with_skip_indexes(fields, &[skipped], max_doc, &id, "").unwrap();
    let (_, meta) = parse_meta(&dvm, &id, "", &FieldInfos { fields: infos }).unwrap();
    let skipper = *meta.skipper_meta(skipped).unwrap();
    parse_skip_index(&dvs, &id, "", &skipper).unwrap()
}

fn skip_field(
    name: &str,
    number: i32,
    t: crate::field_infos::DocValuesType,
) -> crate::field_infos::FieldInfo {
    crate::field_infos::FieldInfo::new(name, number).with_doc_values(
        t,
        crate::field_infos::DocValuesSkipIndexType::Range,
        -1,
    )
}

/// `isDone`: a full interval keeps growing while the documents stay a
/// dense run of the same single value, and closes at the first change.
#[test]
fn a_skip_interval_extends_over_a_dense_run_of_one_value() {
    let mut values = vec![5i64; 5000];
    values.extend(std::iter::repeat_n(9, 100));
    let index = skip_round_trip(
        &[DenseField::Numeric(0, &values)],
        vec![skip_field(
            "n",
            0,
            crate::field_infos::DocValuesType::Numeric,
        )],
        0,
        5100,
    );
    assert_eq!(
        (index.min_value, index.max_value, index.doc_count),
        (5, 9, 5100)
    );
    let base: Vec<_> = index.intervals.iter().map(|i| i.levels[0]).collect();
    assert_eq!(base.len(), 2);
    assert_eq!(
        (base[0].min_doc_id, base[0].max_doc_id, base[0].doc_count),
        (0, 4999, 5000)
    );
    assert_eq!((base[1].min_value, base[1].max_value), (9, 9));
}

/// A sparse SORTED_SET field's skip index covers its ordinals, and only
/// the documents that have a value; a field without the skip index next
/// to it is written as before.
#[test]
fn a_sparse_sorted_set_skip_index_covers_ordinals() {
    let pairs = vec![
        (1, vec![b"b".to_vec(), b"a".to_vec()]),
        (3, vec![b"c".to_vec()]),
    ];
    let plain = vec![1i64, 2, 3, 4];
    let index = skip_round_trip(
        &[
            DenseField::SparseSortedSet(0, &pairs),
            DenseField::Numeric(1, &plain),
        ],
        vec![
            skip_field("s", 0, crate::field_infos::DocValuesType::SortedSet),
            crate::field_infos::FieldInfo::new("p", 1).with_doc_values(
                crate::field_infos::DocValuesType::Numeric,
                crate::field_infos::DocValuesSkipIndexType::None,
                -1,
            ),
        ],
        0,
        4,
    );
    assert_eq!((index.min_value, index.max_value), (0, 2));
    assert_eq!(
        (index.doc_count, index.max_doc_id, index.max_value_count),
        (2, 3, 2)
    );
}

#[test]
fn a_skip_index_on_a_binary_field_is_refused() {
    let values = vec![b"x".to_vec()];
    let err = write_fields_with_skip_indexes(
        &[DenseField::Binary(3, &values)],
        &[3],
        1,
        &[0u8; ID_LENGTH],
        "",
    )
    .unwrap_err();
    assert!(matches!(err, WriteError::SkipIndexOnBinary(3)));
}
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
    /// `Some(shift)` builds a varying-bits-per-value entry (`tableSize
    /// = -2 - shift`, `bits_per_value` written as the `0xFF` sentinel)
    /// instead of the normal table/no-table shape.
    block_shift: Option<u32>,
    value_jump_table_offset: i64,
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
            block_shift: None,
            value_jump_table_offset: 0,
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
        if let Some(shift) = self.block_shift {
            let table_size: i32 = -2 - shift as i32;
            out.extend_from_slice(&table_size.to_le_bytes());
        } else {
            match &self.table {
                Some(t) => {
                    out.extend_from_slice(&(t.len() as i32).to_le_bytes());
                    for v in t {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
                None => out.extend_from_slice(&(-1i32).to_le_bytes()),
            }
        }
        out.push(if self.block_shift.is_some() {
            0xFF
        } else {
            self.bits_per_value
        });
        out.extend_from_slice(&self.min_value.to_le_bytes());
        out.extend_from_slice(&self.gcd.to_le_bytes());
        out.extend_from_slice(&self.values_offset.to_le_bytes());
        out.extend_from_slice(&self.values_length.to_le_bytes());
        out.extend_from_slice(&self.value_jump_table_offset.to_le_bytes());
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
    out.extend_from_slice(&(VERSION_WRITE_DEFAULT as u32).to_be_bytes());
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
    out.extend_from_slice(&(VERSION_WRITE_DEFAULT as u32).to_be_bytes());
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
    assert_eq!(version, VERSION_WRITE_DEFAULT);
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
    out.extend_from_slice(&(VERSION_WRITE_DEFAULT as u32).to_be_bytes());
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
fn invalid_varying_bpv_shift_rejected() {
    let id = [1u8; ID_LENGTH];
    let mut out = Vec::new();
    out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut out, META_CODEC);
    out.extend_from_slice(&(VERSION_WRITE_DEFAULT as u32).to_be_bytes());
    out.extend_from_slice(&id);
    out.push(0);
    out.extend_from_slice(&0i32.to_le_bytes()); // field number
    out.push(DOC_VALUES_TYPE_NUMERIC);
    out.extend_from_slice(&DOCS_WITH_FIELD_DENSE.to_le_bytes());
    out.extend_from_slice(&0i64.to_le_bytes());
    out.extend_from_slice(&0i16.to_le_bytes());
    out.push(0);
    out.extend_from_slice(&3i64.to_le_bytes());
    // tableSize < -1 encodes blockShift = -2 - tableSize; i32::MIN would
    // overflow that subtraction, and any shift > 63 is unusable as a
    // bit-shift amount -- both must be rejected rather than accepted and
    // later panicking in decode_value_varying_bpv.
    out.extend_from_slice(&i32::MIN.to_le_bytes());
    let fis = field_infos_with(&[0]);
    assert!(matches!(
        parse_meta(&out, &id, "", &fis),
        Err(Error::InvalidTableSize(i32::MIN))
    ));
}

/// Hand-builds a two-block varying-bits-per-value field (block size
/// `2^shift = 4`) with 6 values split `[10, 12, 11, 9]` / `[100, 5]` --
/// block 0 needs 2 bits per value (range 9..12), block 1 needs 7 bits
/// (range 5..100) and a different `bitsPerValue` byte -- and checks
/// every value decodes correctly across the block boundary, plus the
/// all-same-value (`bitsPerValue == 0`) single-value shape within a
/// block.
#[test]
fn varying_bpv_blocks_decoded_across_two_blocks() {
    let shift: u32 = 2; // block size 4
    let block0 = [10i64, 12, 11, 9];
    let block1 = [100i64, 5];

    let mut data = Vec::new();

    // Block 0: min=9, values relative to min packed at bitsPerValue=2.
    let block0_start = data.len() as i64;
    let min0 = *block0.iter().min().unwrap();
    let max0 = *block0.iter().max().unwrap();
    let bpv0 = direct_reader::unsigned_bits_required(max0 - min0);
    let raw0: Vec<i64> = block0.iter().map(|v| v - min0).collect();
    let packed0 = direct_reader::encode(&raw0, bpv0);
    data.push(bpv0);
    data.extend_from_slice(&min0.to_le_bytes());
    data.extend_from_slice(&(packed0.len() as i32).to_le_bytes());
    data.extend_from_slice(&packed0);

    // Block 1: min=5, values relative to min packed at bitsPerValue=7.
    let block1_start = data.len() as i64;
    let min1 = *block1.iter().min().unwrap();
    let max1 = *block1.iter().max().unwrap();
    let bpv1 = direct_reader::unsigned_bits_required(max1 - min1);
    let raw1: Vec<i64> = block1.iter().map(|v| v - min1).collect();
    let packed1 = direct_reader::encode(&raw1, bpv1);
    data.push(bpv1);
    data.extend_from_slice(&min1.to_le_bytes());
    data.extend_from_slice(&(packed1.len() as i32).to_le_bytes());
    data.extend_from_slice(&packed1);

    // Jump table: one absolute block-start offset per block, plus a
    // trailing self-referential entry (unused by the decoder, present
    // for on-disk fidelity with the real writer).
    let jump_table_offset = data.len() as i64;
    data.extend_from_slice(&block0_start.to_le_bytes());
    data.extend_from_slice(&block1_start.to_le_bytes());
    data.extend_from_slice(&jump_table_offset.to_le_bytes());

    let mut e = EntryBuilder::dense(0, 0, (block0.len() + block1.len()) as i64);
    e.block_shift = Some(shift);
    e.value_jump_table_offset = jump_table_offset;
    let entry = e.to_entry();
    assert_eq!(entry.block_shift, Some(shift));

    let all_values: Vec<i64> = block0.iter().chain(block1.iter()).copied().collect();
    for (doc, &want) in all_values.iter().enumerate() {
        let got = numeric_value(&data, &entry, doc as i32).unwrap();
        assert_eq!(got, Some(want), "doc {doc}");
    }
}

#[test]
fn a_block_shift_of_63_masks_without_overflowing() {
    // `blockShift` is `-2 - tableSize` off `.dvm`, and `read_numeric_entry`
    // accepts the whole of `0..=63`. At exactly 63 an `i64` mask
    // `(1i64 << 63) - 1` is `i64::MIN - 1` -- a debug-build panic at the
    // one value the range check lets through.
    // Java's `(1L << 63) - 1` wraps to `Long.MAX_VALUE`, which is what the
    // `u64` computation reproduces exactly.
    assert_eq!(low_bits_mask(63), i64::MAX);
    assert_eq!(low_bits_mask(0), 0);
    assert_eq!(low_bits_mask(1), 1);

    // And it is genuinely file-reachable: `tableSize = -65` is what puts
    // `blockShift` at 63, and `parse_meta` accepts it.
    let id = [61u8; ID_LENGTH];
    let mut meta = Vec::new();
    meta.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut meta, META_CODEC);
    meta.extend_from_slice(&(VERSION_WRITE_DEFAULT as u32).to_be_bytes());
    meta.extend_from_slice(&id);
    meta.push(0);
    let mut e = EntryBuilder::dense(0, 0, 2);
    e.block_shift = Some(63);
    e.build(&mut meta);
    meta.extend_from_slice(&(-1i32).to_le_bytes()); // end of field list
    meta.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    meta.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&meta) as u64;
    meta.extend_from_slice(&checksum.to_be_bytes());
    let (_, parsed) = parse_meta(&meta, &id, "", &field_infos_with(&[0])).unwrap();
    assert_eq!(parsed.numeric_entry(0).unwrap().block_shift, Some(63));

    // Decoding through it must not panic. One block at bitsPerValue = 2.
    let mut data = Vec::new();
    let block_start = data.len() as i64;
    let packed = direct_reader::encode(&[1i64, 2], 2);
    data.push(2);
    data.extend_from_slice(&0i64.to_le_bytes());
    data.extend_from_slice(&(packed.len() as i32).to_le_bytes());
    data.extend_from_slice(&packed);
    let jump_table_offset = data.len() as i64;
    data.extend_from_slice(&block_start.to_le_bytes());
    data.extend_from_slice(&jump_table_offset.to_le_bytes());

    let mut e = EntryBuilder::dense(0, 0, 2);
    e.block_shift = Some(63);
    e.value_jump_table_offset = jump_table_offset;
    let entry = e.to_entry();
    assert_eq!(numeric_value(&data, &entry, 0).unwrap(), Some(1));
    assert_eq!(numeric_value(&data, &entry, 1).unwrap(), Some(2));
}

/// A varying-bpv block where every value is identical takes the
/// `bitsPerValue == 0` single-value shape (no packed data at all,
/// just the constant stored as the block's delta).
#[test]
fn varying_bpv_block_all_same_value() {
    let shift: u32 = 1; // block size 2
    let mut data = Vec::new();
    let block_start = data.len() as i64;
    data.push(0u8); // bitsPerValue == 0
    data.extend_from_slice(&7i64.to_le_bytes()); // constant value

    let jump_table_offset = data.len() as i64;
    data.extend_from_slice(&block_start.to_le_bytes());
    data.extend_from_slice(&jump_table_offset.to_le_bytes());

    let mut e = EntryBuilder::dense(0, 0, 2);
    e.block_shift = Some(shift);
    e.value_jump_table_offset = jump_table_offset;
    let entry = e.to_entry();

    assert_eq!(numeric_value(&data, &entry, 0).unwrap(), Some(7));
    assert_eq!(numeric_value(&data, &entry, 1).unwrap(), Some(7));
}

#[test]
fn unsupported_field_type_rejected() {
    let id = [1u8; ID_LENGTH];
    let mut buf = Vec::new();
    buf.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut buf, META_CODEC);
    buf.extend_from_slice(&(VERSION_WRITE_DEFAULT as u32).to_be_bytes());
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

fn skipper_meta_bytes(m: &DocValuesSkipperMeta) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&m.offset.to_le_bytes());
    out.extend_from_slice(&m.length.to_le_bytes());
    out.extend_from_slice(&m.max_value.to_le_bytes());
    out.extend_from_slice(&m.min_value.to_le_bytes());
    out.extend_from_slice(&m.doc_count.to_le_bytes());
    out.extend_from_slice(&m.max_doc_id.to_le_bytes());
    out.extend_from_slice(&m.max_value_count.to_le_bytes());
    out
}

/// One level's 28-byte on-disk record within a skip interval.
fn level_bytes(l: &SkipIndexLevelInterval) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&l.max_doc_id.to_le_bytes());
    out.extend_from_slice(&l.min_doc_id.to_le_bytes());
    out.extend_from_slice(&l.max_value.to_le_bytes());
    out.extend_from_slice(&l.min_value.to_le_bytes());
    out.extend_from_slice(&l.doc_count.to_le_bytes());
    out
}

/// Builds one on-disk skip interval: a level-count byte, then each
/// level's bytes written coarsest (last in `levels`) first.
fn interval_bytes(levels: &[SkipIndexLevelInterval]) -> Vec<u8> {
    let mut out = vec![levels.len() as u8];
    for l in levels.iter().rev() {
        out.extend_from_slice(&level_bytes(l));
    }
    out
}

fn dvs_bytes(id: &[u8; ID_LENGTH], intervals_body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    codec_util::write_index_header(&mut out, SKIP_INDEX_META_CODEC, VERSION_WRITE_DEFAULT, id, "");
    out.extend_from_slice(intervals_body);
    codec_util::write_footer(&mut out);
    out
}

#[test]
fn skip_index_meta_parsed_instead_of_rejected() {
    let id = [1u8; ID_LENGTH];
    let mut fis = field_infos_with(&[0]);
    fis.fields[0].doc_values_skip_index_type = DocValuesSkipIndexType::Range;
    let e = EntryBuilder::dense(0, 8, 3);
    let skipper = DocValuesSkipperMeta {
        offset: 128,
        length: 29,
        min_value: -5,
        max_value: 500,
        doc_count: 3,
        max_doc_id: 2,
        max_value_count: 1,
    };

    let mut buf = Vec::new();
    buf.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut buf, META_CODEC);
    buf.extend_from_slice(&(VERSION_WRITE_DEFAULT as u32).to_be_bytes());
    buf.extend_from_slice(&id);
    buf.push(0);
    buf.extend_from_slice(&e.field_number.to_le_bytes());
    buf.push(DOC_VALUES_TYPE_NUMERIC);
    buf.extend_from_slice(&skipper_meta_bytes(&skipper));
    e.build_body(&mut buf);
    buf.extend_from_slice(&(-1i32).to_le_bytes());
    buf.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    buf.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&buf) as u64;
    buf.extend_from_slice(&checksum.to_be_bytes());

    let (_, meta) = parse_meta(&buf, &id, "", &fis).unwrap();
    assert_eq!(meta.skipper_meta(0), Some(&skipper));
    // A field with no skip index type set has no entry in the map.
    assert_eq!(meta.skipper_meta(99), None);
}

#[test]
fn skip_index_meta_pre_max_value_count_version_infers_it() {
    // VERSION_START (0) predates VERSION_SKIPPER_MAX_VALUE_COUNT (2), so
    // `maxValueCount` isn't stored on disk and must be inferred: 0 when
    // docCount == 0, -1 (unknown) otherwise.
    let zero_docs_bytes = {
        let mut b = Vec::new();
        b.extend_from_slice(&10i64.to_le_bytes()); // offset
        b.extend_from_slice(&0i64.to_le_bytes()); // length
        b.extend_from_slice(&0i64.to_le_bytes()); // maxValue
        b.extend_from_slice(&0i64.to_le_bytes()); // minValue
        b.extend_from_slice(&0i32.to_le_bytes()); // docCount
        b.extend_from_slice(&(-1i32).to_le_bytes()); // maxDocId
        b
    };
    let mut input_zero_docs = SliceInput::new(&zero_docs_bytes);
    let m = read_skipper_meta(&mut input_zero_docs, VERSION_START).unwrap();
    assert_eq!(m.max_value_count, 0);

    let some_docs_bytes = {
        let mut b = Vec::new();
        b.extend_from_slice(&10i64.to_le_bytes());
        b.extend_from_slice(&0i64.to_le_bytes());
        b.extend_from_slice(&5i64.to_le_bytes());
        b.extend_from_slice(&1i64.to_le_bytes());
        b.extend_from_slice(&4i32.to_le_bytes()); // docCount
        b.extend_from_slice(&3i32.to_le_bytes());
        b
    };
    let mut input_some_docs = SliceInput::new(&some_docs_bytes);
    let m = read_skipper_meta(&mut input_some_docs, VERSION_START).unwrap();
    assert_eq!(m.max_value_count, -1);
}

// --- DocValuesSkipper (Lucene90DocValuesProducer.getSkipper) ---

/// Nine 10-doc level-0 intervals (docs 0..=89). `writeLevels`/`getLevels`
/// would give interval 0 a second level covering intervals 0..7 (docs
/// 0..=79), since `left >= 8 && index % 8 == 0` holds only there.
fn nine_interval_index(sabotage_covered_intervals: bool) -> DocValuesSkipIndex {
    let level0 = |i: i32| SkipIndexLevelInterval {
        min_doc_id: i * 10,
        // Intervals 1..=7 are the ones the level-1 record covers, so a
        // correct skipper never reads them when level 1 already ruled the
        // whole group out. Giving them an impossible `max_doc_id` makes a
        // linear scan stop on one of them and fail the test.
        max_doc_id: if sabotage_covered_intervals && (1..=7).contains(&i) {
            1000
        } else {
            i * 10 + 9
        },
        min_value: i as i64 * 100,
        max_value: i as i64 * 100 + 99,
        doc_count: 10,
    };
    let level1 = SkipIndexLevelInterval {
        min_doc_id: 0,
        max_doc_id: 79,
        min_value: 0,
        max_value: 799,
        doc_count: 80,
    };
    let mut intervals = vec![SkipIndexInterval {
        levels: vec![level0(0), level1],
    }];
    for i in 1..9 {
        intervals.push(SkipIndexInterval {
            levels: vec![level0(i)],
        });
    }
    DocValuesSkipIndex {
        min_value: 0,
        max_value: 899,
        doc_count: 90,
        max_doc_id: 89,
        max_value_count: 1,
        intervals,
    }
}

#[test]
fn skipper_before_first_advance_reports_minus_one_and_one_level() {
    let index = nine_interval_index(false);
    let skipper = DocValuesSkipper::new(&index);
    assert_eq!(skipper.num_levels(), 1);
    assert_eq!(skipper.min_doc_id(0), -1);
    assert_eq!(skipper.max_doc_id(0), -1);
    assert_eq!(skipper.global_min_value(), 0);
    assert_eq!(skipper.global_max_value(), 899);
    assert_eq!(skipper.global_doc_count(), 90);
    assert_eq!(skipper.max_value_count(), 1);
}

#[test]
fn skipper_advance_reads_every_level_of_the_landing_interval() {
    let index = nine_interval_index(false);
    let mut skipper = DocValuesSkipper::new(&index);
    skipper.advance(0);
    assert_eq!(skipper.num_levels(), 2);
    assert_eq!((skipper.min_doc_id(0), skipper.max_doc_id(0)), (0, 9));
    assert_eq!((skipper.min_value(0), skipper.max_value(0)), (0, 99));
    assert_eq!(skipper.doc_count(0), 10);
    assert_eq!((skipper.min_doc_id(1), skipper.max_doc_id(1)), (0, 79));
    assert_eq!((skipper.min_value(1), skipper.max_value(1)), (0, 799));
    assert_eq!(skipper.doc_count(1), 80);
}

/// Java keeps the per-level arrays outside its read loop, so a plain
/// level-0 interval leaves the coarser levels holding the summary read
/// from the last interval that carried one -- and the `levels` widening
/// loop then re-exposes it. Reproduced here: after landing on interval 3
/// (a 1-level record), level 1 still describes docs 0..=79.
fn assert_stale_level_one_survives(skipper: &DocValuesSkipper<'_>) {
    assert_eq!(skipper.num_levels(), 2);
    assert_eq!((skipper.min_doc_id(1), skipper.max_doc_id(1)), (0, 79));
    assert_eq!(skipper.doc_count(1), 80);
}

#[test]
fn skipper_widens_levels_using_a_previous_intervals_coarse_summary() {
    let index = nine_interval_index(false);
    let mut skipper = DocValuesSkipper::new(&index);
    skipper.advance(0);
    skipper.advance(35);
    assert_eq!((skipper.min_doc_id(0), skipper.max_doc_id(0)), (30, 39));
    assert_eq!((skipper.min_value(0), skipper.max_value(0)), (300, 399));
    assert_stale_level_one_survives(&skipper);
}

#[test]
fn skipper_narrows_levels_again_once_the_coarse_interval_is_behind() {
    let index = nine_interval_index(false);
    let mut skipper = DocValuesSkipper::new(&index);
    skipper.advance(0);
    skipper.advance(85);
    assert_eq!((skipper.min_doc_id(0), skipper.max_doc_id(0)), (80, 89));
    // Level 1's stale interval ends at 79 < 85, so it is no longer
    // reported: `levels` stays 1.
    assert_eq!(skipper.num_levels(), 1);
}

/// Bailing out at level 1 must jump the whole `8^1`-interval subtree that
/// level covers, not walk it. The covered intervals carry an impossible
/// `max_doc_id` here, so a linear scan would stop on interval 1.
#[test]
fn skipper_jumps_a_whole_subtree_when_a_coarse_level_rules_it_out() {
    let index = nine_interval_index(true);
    let mut skipper = DocValuesSkipper::new(&index);
    skipper.advance(80);
    assert_eq!((skipper.min_doc_id(0), skipper.max_doc_id(0)), (80, 89));
    assert_eq!(skipper.num_levels(), 1);
}

#[test]
fn skipper_past_the_last_doc_is_exhausted_on_every_level() {
    let index = nine_interval_index(false);
    let mut skipper = DocValuesSkipper::new(&index);
    skipper.advance(0);
    skipper.advance(90);
    for level in 0..SKIP_INDEX_MAX_LEVEL as usize {
        assert_eq!(skipper.min_doc_id(level), NO_MORE_DOCS);
        assert_eq!(skipper.max_doc_id(level), NO_MORE_DOCS);
    }
}

/// A truncated `.dvs` whose `.dvm` summary promises documents the
/// interval list does not cover must report exhaustion, not spin or
/// panic.
#[test]
fn skipper_on_a_truncated_interval_list_reports_exhaustion() {
    let mut index = nine_interval_index(false);
    index.intervals.truncate(1);
    let mut skipper = DocValuesSkipper::new(&index);
    skipper.advance(50);
    assert_eq!(skipper.max_doc_id(0), NO_MORE_DOCS);
}

#[test]
fn skipper_advance_range_skips_to_the_first_intersecting_interval() {
    let index = nine_interval_index(false);
    let mut skipper = DocValuesSkipper::new(&index);
    // Values 500..=520 live only in interval 5 (values 500..=599).
    skipper.advance_range(500, 520);
    assert_eq!((skipper.min_doc_id(0), skipper.max_doc_id(0)), (50, 59));
    assert_eq!((skipper.min_value(0), skipper.max_value(0)), (500, 599));
}

#[test]
fn skipper_advance_range_with_no_matching_interval_exhausts() {
    let index = nine_interval_index(false);
    let mut skipper = DocValuesSkipper::new(&index);
    skipper.advance_range(10_000, 20_000);
    assert_eq!(skipper.min_doc_id(0), NO_MORE_DOCS);
}

/// End-to-end over real `.dvs` bytes: build a two-level interval list with
/// the same writer layout [`parse_skip_index`] decodes, then drive the
/// skipper across it.
#[test]
fn skipper_over_parse_skip_index_output() {
    let id = [3u8; ID_LENGTH];
    let index = nine_interval_index(false);
    let mut body = Vec::new();
    for interval in &index.intervals {
        body.extend_from_slice(&interval_bytes(&interval.levels));
    }
    let length = body.len() as i64;
    let buf = dvs_bytes(&id, &body);
    let offset = (buf.len() as i64) - length - codec_util::FOOTER_LENGTH as i64;
    let skipper_meta = DocValuesSkipperMeta {
        offset,
        length,
        min_value: 0,
        max_value: 899,
        doc_count: 90,
        max_doc_id: 89,
        max_value_count: 1,
    };
    let decoded = parse_skip_index(&buf, &id, "", &skipper_meta).unwrap();
    assert_eq!(decoded.intervals, index.intervals);

    let mut skipper = DocValuesSkipper::new(&decoded);
    skipper.advance(0);
    assert_eq!(skipper.num_levels(), 2);
    skipper.advance(35);
    assert_eq!((skipper.min_doc_id(0), skipper.max_doc_id(0)), (30, 39));
    assert_stale_level_one_survives(&skipper);
}

/// Builds a whole `.dvm` at `version` holding one NUMERIC field with a
/// skip index, whose `maxValueCount` is only written from
/// [`VERSION_SKIPPER_MAX_VALUE_COUNT`] onwards.
fn dvm_with_skipper(
    id: &[u8; ID_LENGTH],
    version: i32,
    type_byte: u8,
    skipper: &DocValuesSkipperMeta,
    entry_body: &[u8],
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut buf, META_CODEC);
    buf.extend_from_slice(&(version as u32).to_be_bytes());
    buf.extend_from_slice(id);
    buf.push(0);
    buf.extend_from_slice(&0i32.to_le_bytes()); // field number
    buf.push(type_byte);
    let skipper_bytes = skipper_meta_bytes(skipper);
    let keep = if version >= VERSION_SKIPPER_MAX_VALUE_COUNT {
        skipper_bytes.len()
    } else {
        skipper_bytes.len() - 4 // no trailing maxValueCount on disk
    };
    buf.extend_from_slice(&skipper_bytes[..keep]);
    buf.extend_from_slice(entry_body);
    buf.extend_from_slice(&(-1i32).to_le_bytes());
    buf.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    buf.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&buf) as u64;
    buf.extend_from_slice(&checksum.to_be_bytes());
    buf
}

/// `Lucene90DocValuesProducer.inferMaxValueCounts`: a pre-version-2 `.dvm`
/// stores no `maxValueCount`, but a NUMERIC field is single-valued by
/// construction, so the producer fills in `1` rather than leaving the `-1`
/// "unknown" sentinel that would make a caller give up on the
/// single-valued fast path.
#[test]
fn pre_version_2_numeric_skipper_max_value_count_is_inferred_as_one() {
    let id = [4u8; ID_LENGTH];
    let mut fis = field_infos_with(&[0]);
    fis.fields[0].doc_values_skip_index_type = DocValuesSkipIndexType::Range;
    let skipper = DocValuesSkipperMeta {
        offset: 0,
        length: 29,
        min_value: 1,
        max_value: 5,
        doc_count: 3,
        max_doc_id: 2,
        max_value_count: -1,
    };
    let mut body = Vec::new();
    EntryBuilder::dense(0, 8, 3).build_body(&mut body);
    let buf = dvm_with_skipper(&id, VERSION_START, DOC_VALUES_TYPE_NUMERIC, &skipper, &body);

    let (version, meta) = parse_meta(&buf, &id, "", &fis).unwrap();
    assert_eq!(version, VERSION_START);
    assert_eq!(meta.skipper_meta(0).unwrap().max_value_count, 1);
}

/// The same file at [`VERSION_WRITE_DEFAULT`] stores `maxValueCount` explicitly,
/// so the inference pass must not run and must not overwrite it.
#[test]
fn current_version_skipper_max_value_count_is_read_not_inferred() {
    let id = [4u8; ID_LENGTH];
    let mut fis = field_infos_with(&[0]);
    fis.fields[0].doc_values_skip_index_type = DocValuesSkipIndexType::Range;
    let skipper = DocValuesSkipperMeta {
        offset: 0,
        length: 29,
        min_value: 1,
        max_value: 5,
        doc_count: 3,
        max_doc_id: 2,
        max_value_count: 7,
    };
    let mut body = Vec::new();
    EntryBuilder::dense(0, 8, 3).build_body(&mut body);
    let buf = dvm_with_skipper(
        &id,
        VERSION_WRITE_DEFAULT,
        DOC_VALUES_TYPE_NUMERIC,
        &skipper,
        &body,
    );

    let (_, meta) = parse_meta(&buf, &id, "", &fis).unwrap();
    assert_eq!(meta.skipper_meta(0).unwrap().max_value_count, 7);
}

/// A genuinely multi-valued SORTED_NUMERIC field (`numValues >
/// numDocsWithField`) cannot be inferred and keeps the `-1` sentinel,
/// exactly like Java's `switch` leaving it alone.
#[test]
fn pre_version_2_multi_valued_sorted_numeric_skipper_stays_unknown() {
    let id = [4u8; ID_LENGTH];
    let mut fis = field_infos_with(&[0]);
    fis.fields[0].doc_values_skip_index_type = DocValuesSkipIndexType::Range;
    fis.fields[0].doc_values_type = DocValuesType::SortedNumeric;
    let skipper = DocValuesSkipperMeta {
        offset: 0,
        length: 29,
        min_value: 1,
        max_value: 5,
        doc_count: 2,
        max_doc_id: 1,
        max_value_count: -1,
    };
    // numValues = 4 across numDocsWithField = 2 -> multi-valued, plus the
    // address array that implies.
    let mut body = Vec::new();
    EntryBuilder::dense(0, 8, 4).build_body(&mut body);
    body.extend_from_slice(&2i32.to_le_bytes()); // numDocsWithField
    body.extend_from_slice(&0i64.to_le_bytes()); // addressesOffset
    write_vint(&mut body, 0); // blockShift
    write_zero_direct_monotonic_blocks(&mut body, 3, 0);
    body.extend_from_slice(&0i64.to_le_bytes()); // addressesLength
    let buf = dvm_with_skipper(
        &id,
        VERSION_START,
        DOC_VALUES_TYPE_SORTED_NUMERIC,
        &skipper,
        &body,
    );

    let (_, meta) = parse_meta(&buf, &id, "", &fis).unwrap();
    assert_eq!(meta.skipper_meta(0).unwrap().max_value_count, -1);
}

/// A `.dvs` file only exists from `VERSION_SKIPPER_SEPARATE_FILE` on, so
/// one claiming `VERSION_START` is corrupt.
#[test]
fn parse_skip_index_rejects_a_version_zero_dvs() {
    let id = [5u8; ID_LENGTH];
    let mut buf = Vec::new();
    buf.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut buf, SKIP_INDEX_META_CODEC);
    buf.extend_from_slice(&(VERSION_START as u32).to_be_bytes());
    buf.extend_from_slice(&id);
    buf.push(0);
    codec_util::write_footer(&mut buf);
    let skipper = DocValuesSkipperMeta {
        offset: 0,
        length: 0,
        min_value: 0,
        max_value: 0,
        doc_count: 0,
        max_doc_id: -1,
        max_value_count: 0,
    };
    assert!(parse_skip_index(&buf, &id, "", &skipper).is_err());
    assert!(check_skip_index_header_footer(&buf, &id, "").is_err());
}

#[test]
fn parse_skip_index_decodes_multi_level_intervals() {
    let id = [9u8; ID_LENGTH];
    let base0 = SkipIndexLevelInterval {
        min_doc_id: 0,
        max_doc_id: 99,
        min_value: -3,
        max_value: 40,
        doc_count: 100,
    };
    let level1 = SkipIndexLevelInterval {
        min_doc_id: 0,
        max_doc_id: 199,
        min_value: -3,
        max_value: 70,
        doc_count: 200,
    };
    let base1 = SkipIndexLevelInterval {
        min_doc_id: 100,
        max_doc_id: 199,
        min_value: 2,
        max_value: 70,
        doc_count: 100,
    };

    let mut body = Vec::new();
    body.extend_from_slice(&interval_bytes(&[base0, level1]));
    body.extend_from_slice(&interval_bytes(&[base1]));

    let header_len = {
        let mut probe = Vec::new();
        codec_util::write_index_header(&mut probe, SKIP_INDEX_META_CODEC, VERSION_WRITE_DEFAULT, &id, "");
        probe.len()
    };
    let dvs = dvs_bytes(&id, &body);

    let skipper = DocValuesSkipperMeta {
        offset: header_len as i64,
        length: body.len() as i64,
        min_value: -3,
        max_value: 70,
        doc_count: 200,
        max_doc_id: 199,
        max_value_count: 1,
    };

    let decoded = parse_skip_index(&dvs, &id, "", &skipper).unwrap();
    assert_eq!(decoded.min_value, -3);
    assert_eq!(decoded.max_value, 70);
    assert_eq!(decoded.doc_count, 200);
    assert_eq!(decoded.max_doc_id, 199);
    assert_eq!(decoded.intervals.len(), 2);
    assert_eq!(decoded.intervals[0].levels, vec![base0, level1]);
    assert_eq!(decoded.intervals[1].levels, vec![base1]);
}

#[test]
fn parse_skip_index_rejects_zero_level_count() {
    let id = [9u8; ID_LENGTH];
    let body = vec![0u8]; // level count 0 is invalid (must be 1..=4)
    let header_len = {
        let mut probe = Vec::new();
        codec_util::write_index_header(&mut probe, SKIP_INDEX_META_CODEC, VERSION_WRITE_DEFAULT, &id, "");
        probe.len()
    };
    let dvs = dvs_bytes(&id, &body);
    let skipper = DocValuesSkipperMeta {
        offset: header_len as i64,
        length: body.len() as i64,
        min_value: 0,
        max_value: 0,
        doc_count: 0,
        max_doc_id: -1,
        max_value_count: 0,
    };
    assert!(matches!(
        parse_skip_index(&dvs, &id, "", &skipper),
        Err(Error::InvalidSkipIndexLevelCount(0))
    ));
}

#[test]
fn parse_skip_index_rejects_level_count_above_max() {
    let id = [9u8; ID_LENGTH];
    let body = vec![SKIP_INDEX_MAX_LEVEL + 1];
    let header_len = {
        let mut probe = Vec::new();
        codec_util::write_index_header(&mut probe, SKIP_INDEX_META_CODEC, VERSION_WRITE_DEFAULT, &id, "");
        probe.len()
    };
    let dvs = dvs_bytes(&id, &body);
    let skipper = DocValuesSkipperMeta {
        offset: header_len as i64,
        length: body.len() as i64,
        min_value: 0,
        max_value: 0,
        doc_count: 0,
        max_doc_id: -1,
        max_value_count: 0,
    };
    assert!(matches!(
        parse_skip_index(&dvs, &id, "", &skipper),
        Err(Error::InvalidSkipIndexLevelCount(n)) if n == SKIP_INDEX_MAX_LEVEL + 1
    ));
}

#[test]
fn parse_skip_index_rejects_truncated_level_body() {
    let id = [9u8; ID_LENGTH];
    // Level count says 1 level (28 bytes) but only 10 bytes follow.
    let mut body = vec![1u8];
    body.extend_from_slice(&[0u8; 10]);
    let header_len = {
        let mut probe = Vec::new();
        codec_util::write_index_header(&mut probe, SKIP_INDEX_META_CODEC, VERSION_WRITE_DEFAULT, &id, "");
        probe.len()
    };
    let dvs = dvs_bytes(&id, &body);
    let skipper = DocValuesSkipperMeta {
        offset: header_len as i64,
        length: body.len() as i64,
        min_value: 0,
        max_value: 0,
        doc_count: 0,
        max_doc_id: -1,
        max_value_count: 0,
    };
    assert!(parse_skip_index(&dvs, &id, "", &skipper).is_err());
}

#[test]
fn parse_skip_index_rejects_offset_length_out_of_range() {
    let id = [9u8; ID_LENGTH];
    let dvs = dvs_bytes(&id, &[]);
    let skipper = DocValuesSkipperMeta {
        offset: dvs.len() as i64, // starts past the end of the buffer
        length: 100,
        min_value: 0,
        max_value: 0,
        doc_count: 0,
        max_doc_id: -1,
        max_value_count: 0,
    };
    assert!(matches!(
        parse_skip_index(&dvs, &id, "", &skipper),
        Err(Error::SkipIndexOutOfRange(_, _, _))
    ));
}

#[test]
fn parse_skip_index_rejects_bad_dvs_header() {
    let id = [9u8; ID_LENGTH];
    let other_id = [8u8; ID_LENGTH];
    let dvs = dvs_bytes(&other_id, &[]);
    let skipper = DocValuesSkipperMeta {
        offset: 0,
        length: 0,
        min_value: 0,
        max_value: 0,
        doc_count: 0,
        max_doc_id: -1,
        max_value_count: 0,
    };
    // Segment id mismatch -> header check fails before offset/length are
    // even consulted.
    assert!(parse_skip_index(&dvs, &id, "", &skipper).is_err());
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
    assert_eq!(version, VERSION_WRITE_DEFAULT);
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
        disi: Default::default(),
        mask: None,
        field_number,
        docs_with_field_offset,
        docs_with_field_length,
        jump_table_entry_count: 0,
        dense_rank_power: indexed_disi::NO_RANK,
        num_docs_with_field,
        min_length: length,
        max_length: length,
        data_offset,
        data_length,
        addresses: None,
        substring: None,
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
        disi: Default::default(),
        mask: None,
        field_number: 0,
        docs_with_field_offset: DOCS_WITH_FIELD_DENSE,
        docs_with_field_length: 0,
        jump_table_entry_count: 0,
        dense_rank_power: indexed_disi::NO_RANK,
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
        substring: None,
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
    out.extend_from_slice(&(VERSION_WRITE_DEFAULT as u32).to_be_bytes());
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
    dvm.extend_from_slice(&(VERSION_WRITE_DEFAULT as u32).to_be_bytes());
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
    dvm.extend_from_slice(&8192i32.to_le_bytes()); // maxBlockLength (bounds the per-block length vint)
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

/// A dense SORTED_NUMERIC entry covers exactly `maxDoc` docs
/// (`numDocsWithField == maxDoc` is what makes it dense), so a doc past
/// that is out of range -- not a silently-answered constant.
#[test]
fn sorted_numeric_dense_doc_past_max_doc_rejected() {
    let mut data = Vec::new();
    // A constant-encoded (`bitsPerValue == 0`) dense field over 3 docs:
    // `decode_value` would answer for any ordinal without this check.
    data.extend_from_slice(&[0u8; 8]);
    let mut e = EntryBuilder::dense(0, 0, 3);
    e.min_value = 42;
    let entry = sorted_numeric_entry_no_addresses(e.to_entry());
    assert_eq!(sorted_numeric_values(&data, &entry, 2).unwrap(), vec![42]);
    assert!(matches!(
        sorted_numeric_values(&data, &entry, 3),
        Err(Error::DocOutOfRange(3, 3))
    ));
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
    dvm.extend_from_slice(&(VERSION_WRITE_DEFAULT as u32).to_be_bytes());
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
        check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id).unwrap(),
        VERSION_WRITE_DEFAULT
    );

    let version = check_data_header_footer(&data_bytes, &id, "").unwrap();
    assert_eq!(version, VERSION_WRITE_DEFAULT);

    let fis = field_infos_with(&[0]);
    let (meta_version, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    assert_eq!(meta_version, VERSION_WRITE_DEFAULT);
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
    assert_eq!(entry.min_value, 42);
    // `Lucene90DocValuesConsumer` runs its GCD scan before the
    // `min >= max` branch and writes whatever it produced -- `0` for an
    // all-equal array, since every `v - firstValue` is 0. Nothing reads it
    // back (`bitsPerValue == 0` short-circuits), but the bytes must match
    // real Lucene's.
    assert_eq!(entry.gcd, 0);
    assert_eq!(entry.table, None);
    for doc in 0..values.len() as i32 {
        assert_eq!(numeric_value(&data_bytes, entry, doc).unwrap(), Some(42));
    }
}

/// The GCD scan's overflow guard (`v` outside `[MIN/2, MAX/2]` abandons
/// GCD tracking) fires even for an all-equal array, so the constant case
/// then records `gcd == 1` rather than `0` -- again exactly what Java
/// writes.
#[test]
fn write_single_dense_numeric_field_all_equal_extreme_values_record_gcd_one() {
    let id = [8u8; ID_LENGTH];
    let values = vec![i64::MAX; 4];
    let (meta_bytes, data_bytes, _) =
        write_single_dense_numeric_field(0, &values, values.len() as i32, &id, "").unwrap();
    let fis = field_infos_with(&[0]);
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let entry = meta.numeric_entry(0).unwrap();
    assert_eq!(entry.bits_per_value, 0);
    assert_eq!(entry.gcd, 1);
    for doc in 0..values.len() as i32 {
        assert_eq!(
            numeric_value(&data_bytes, entry, doc).unwrap(),
            Some(i64::MAX)
        );
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
    assert_eq!(version, VERSION_WRITE_DEFAULT);

    let fis = field_infos_with(&[0]);
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let entry = meta.numeric_entry(0).unwrap();
    assert!(!entry.is_dense());
    assert!(!entry.is_empty_field());

    // `numeric_value` opens a fresh `DisiCursor` per call, so it walks
    // this field's block headers from the start each time. That is
    // O(blocks), not O(cardinality), but 200,000 individual lookups is
    // still needless: every present doc plus a stride of absent ones
    // exercises both branches. `NumericReader`, which keeps one cursor
    // across calls, is checked doc-for-doc against this in
    // `numeric_reader_sparse_agrees_with_numeric_value_in_any_order`.
    let present: std::collections::HashMap<i32, i64> = doc_values.iter().copied().collect();
    for &(doc, want) in doc_values.iter().step_by(97) {
        assert_eq!(numeric_value(&data_bytes, entry, doc).unwrap(), Some(want));
    }
    for doc in (0..max_doc).step_by(97) {
        let got = numeric_value(&data_bytes, entry, doc).unwrap();
        assert_eq!(got, present.get(&doc).copied(), "doc {doc}");
    }
}

/// [`NumericReader`] holds a forward-only `DisiCursor` rather than a decoded
/// `Vec<i32>` of every present doc. The two access patterns that
/// distinguishes are checked against [`numeric_value`], which is the
/// independent implementation (fresh cursor per call):
///
/// - ascending, the order a sort or facet count uses, where the cursor is
///   never rewound and the whole scan walks the region once;
/// - pseudo-random, where every backward step must rewind the cursor. A
///   reader that dropped the rewind would answer `None` for a legitimate
///   document, which is exactly the failure the old `Vec` hid.
///
/// Checked over every doc id, including the absent ones -- the `None`s are
/// the interesting answers.
#[test]
fn numeric_reader_sparse_agrees_with_numeric_value_in_any_order() {
    // Block 0: 10 present docs -> SPARSE. Block 1: every doc -> ALL.
    // Block 2: every 3rd doc -> DENSE. All three shapes, in one field.
    let id = [21u8; ID_LENGTH];
    let max_doc = 196_608i32;
    let mut doc_values: Vec<(i32, i64)> = Vec::new();
    for i in 0..10 {
        doc_values.push((i * 1000, i as i64 - 5));
    }
    for doc in 65_536..131_072 {
        doc_values.push((doc, doc as i64 * 3));
    }
    for doc in (131_072..196_608).step_by(3) {
        doc_values.push((doc, -(doc as i64)));
    }
    let present: std::collections::HashMap<i32, i64> = doc_values.iter().copied().collect();

    let (meta_bytes, data_bytes, _skip) =
        write_single_sparse_numeric_field(0, &doc_values, max_doc, &id, "").unwrap();
    let fis = field_infos_with(&[0]);
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let entry = meta.numeric_entry(0).unwrap();

    // Ascending: one walk of the region for all 196,608 lookups.
    let mut reader = NumericReader::new(&data_bytes, entry);
    for doc in 0..max_doc {
        assert_eq!(
            reader.value(doc).unwrap(),
            present.get(&doc).copied(),
            "ascending, doc {doc}"
        );
    }

    // Pseudo-random, every backward step rewinding the cursor.
    let mut reader = NumericReader::new(&data_bytes, entry);
    let mut probe = 987_654_321u32;
    for _ in 0..3_000 {
        probe = probe.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let doc = (probe % max_doc as u32) as i32;
        assert_eq!(
            reader.value(doc).unwrap(),
            present.get(&doc).copied(),
            "random, doc {doc}"
        );
        // Same document twice in a row must not change the answer either.
        assert_eq!(reader.value(doc).unwrap(), present.get(&doc).copied());
    }
}

/// `for_each_value` is `value` over a range, for every field shape the
/// writer produces: constant, GCD-compressed, table-compressed, plain
/// packed at several widths (dense, so the chunked path), and sparse (the
/// per-document fallback) -- over the whole range, a sub-range that starts
/// and ends mid-chunk, and an empty one.
#[test]
fn for_each_value_agrees_with_value_on_every_shape() {
    let id = [23u8; ID_LENGTH];
    let fis = field_infos_with(&[0]);
    let n = 1_000i32;
    let shapes: Vec<Vec<i64>> = vec![
        vec![42; n as usize],                                  // constant
        (0..n as i64).map(|i| 1_000 + 7 * (i % 13)).collect(), // gcd 7
        (0..n as i64)
            .map(|i| [5, -3, 1 << 40][(i % 3) as usize])
            .collect(), // table
        (0..n as i64).map(|i| i * 3 % 251).collect(),          // 8 bits
        (0..n as i64).map(|i| (i * 7919) % 65_000).collect(),  // 16 bits
        (0..n as i64)
            .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15u64 as i64))
            .collect(), // 64 bits
    ];
    let check = |meta_bytes: &[u8], data: &[u8]| {
        let (_, meta) = parse_meta(meta_bytes, &id, "", &fis).unwrap();
        let entry = meta.numeric_entry(0).unwrap();
        for (start, end) in [(0, n), (3, 700), (255, 257), (500, 500), (-5, 10)] {
            let mut expected = Vec::new();
            let mut by_value = NumericReader::new(data, entry);
            for doc in start.max(0)..end {
                if let Some(v) = by_value.value(doc).unwrap() {
                    expected.push((doc, v));
                }
            }
            let mut got = Vec::new();
            NumericReader::new(data, entry)
                .for_each_value(start, end, |d, v| got.push((d, v)))
                .unwrap();
            assert_eq!(got, expected, "range {start}..{end}");
        }
    };
    for values in &shapes {
        let (meta_bytes, data, _) =
            write_single_dense_numeric_field(0, values, n, &id, "").unwrap();
        check(&meta_bytes, &data);
    }
    let sparse: Vec<(i32, i64)> = (0..n).step_by(3).map(|d| (d, d as i64 * 11)).collect();
    let (meta_bytes, data, _) = write_single_sparse_numeric_field(0, &sparse, n, &id, "").unwrap();
    check(&meta_bytes, &data);
}

/// `fill_window` is `value` over a window, for every field shape: the
/// dense shapes `for_each_value` covers, and a sparse field with all three
/// `IndexedDISI` block encodings (SPARSE, ALL, DENSE) -- windows walked
/// forward across block boundaries, one straddling the end of the
/// documents, one going backwards (a rewind), and a short `present`.
#[test]
fn fill_window_agrees_with_value_on_every_shape() {
    let id = [24u8; ID_LENGTH];
    let fis = field_infos_with(&[0]);
    let check = |meta_bytes: &[u8], data: &[u8], max_doc: i32, windows: &[(i32, usize)]| {
        let (_, meta) = parse_meta(meta_bytes, &id, "", &fis).unwrap();
        let entry = meta.numeric_entry(0).unwrap();
        let mut windowed = NumericReader::new(data, entry);
        for &(start, len) in windows {
            let mut values = vec![i64::MIN; len];
            let mut present = vec![u64::MAX; len.div_ceil(64) + 1];
            windowed
                .fill_window(start, &mut values, &mut present)
                .unwrap();
            let mut by_value = NumericReader::new(data, entry);
            for (i, &v) in values.iter().enumerate() {
                let doc = start + i as i32;
                let want = if doc < max_doc {
                    by_value.value(doc).unwrap()
                } else {
                    None
                };
                let has = present[i >> 6] >> (i & 63) & 1 == 1;
                assert_eq!(has.then_some(v), want, "window {start}+{len}, doc {doc}");
            }
            // The words past the window are left alone.
            assert_eq!(present[len.div_ceil(64)], u64::MAX);
        }
    };
    let n = 3_000i32;
    let dense: Vec<Vec<i64>> = vec![
        vec![42; n as usize],
        (0..n as i64).map(|i| 1_000 + 7 * (i % 13)).collect(),
        (0..n as i64)
            .map(|i| [5, -3, 1 << 40][(i % 3) as usize])
            .collect(),
        (0..n as i64).map(|i| (i * 7919) % 65_000).collect(),
    ];
    for values in &dense {
        let (meta_bytes, data, _) =
            write_single_dense_numeric_field(0, values, n, &id, "").unwrap();
        check(
            &meta_bytes,
            &data,
            n,
            &[(0, 1024), (1024, 1024), (2048, 952), (100, 300)],
        );
    }
    let max_doc = 196_608i32;
    let mut sparse: Vec<(i32, i64)> = (0..10).map(|i| (i * 1000, i as i64 - 5)).collect();
    sparse.extend((65_536..131_072).map(|d| (d, d as i64 * 3)));
    sparse.extend((131_072..196_608).step_by(3).map(|d| (d, -(d as i64))));
    let (meta_bytes, data, _) =
        write_single_sparse_numeric_field(0, &sparse, max_doc, &id, "").unwrap();
    let mut windows: Vec<(i32, usize)> = (0..max_doc).step_by(1024).map(|s| (s, 1024)).collect();
    windows.extend([(3_000, 2_000), (65_000, 1_100), (131_000, 200), (1, 63)]);
    check(&meta_bytes, &data, max_doc, &windows);
    // Too few words for the window, or a negative start: an error.
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let mut reader = NumericReader::new(&data, meta.numeric_entry(0).unwrap());
    assert!(reader.fill_window(0, &mut [0; 65], &mut [0]).is_err());
    assert!(reader.fill_window(-1, &mut [0; 4], &mut [0]).is_err());
    assert!(reader.fill_window(i32::MAX, &mut [0; 4], &mut [0]).is_err());
}

/// A sparse entry whose docs-with-field byte range is not inside the data
/// must still report the error rather than answer "no value": the reader
/// builds no cursor and falls through to [`numeric_value`].
#[test]
fn numeric_reader_sparse_region_outside_the_data_is_an_error() {
    let entry = NumericEntry {
        disi: Default::default(),
        field_number: 0,
        docs_with_field_offset: 4,
        docs_with_field_length: 1_000,
        jump_table_entry_count: 0,
        dense_rank_power: indexed_disi::NO_RANK,
        num_values: 3,
        table: None,
        bits_per_value: 0,
        min_value: 7,
        gcd: 1,
        values_offset: 0,
        values_length: 0,
        block_shift: None,
        value_jump_table_offset: 0,
        adaptive: None,
    };
    let mut reader = NumericReader::new(&[0u8; 8], &entry);
    assert!(reader.value(0).is_err());
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
    let err = write_single_sparse_numeric_field(0, &[(1, 10), (1, 20)], 5, &id, "").unwrap_err();
    assert!(matches!(err, WriteError::DocIdsNotAscending(1)));
}

#[test]
fn write_single_sparse_numeric_field_rejects_out_of_range_doc_id() {
    let id = [14u8; ID_LENGTH];
    let err = write_single_sparse_numeric_field(0, &[(0, 10), (5, 20)], 5, &id, "").unwrap_err();
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
        check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id).unwrap(),
        VERSION_WRITE_DEFAULT
    );
    let version = check_data_header_footer(&data_bytes, &id, "").unwrap();
    assert_eq!(version, VERSION_WRITE_DEFAULT);

    let fis = field_infos_with(&[0]);
    let (meta_version, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    assert_eq!(meta_version, VERSION_WRITE_DEFAULT);
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
        VERSION_WRITE_DEFAULT,
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
    dvm.extend_from_slice(&(VERSION_WRITE_DEFAULT as u32).to_be_bytes());
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

#[test]
fn sorted_numeric_reader_caches_varying_bpv_blocks_exactly() {
    let id = [4u8; ID_LENGTH];
    let values: Vec<Vec<i64>> = (0..40_000usize)
        .map(|d| {
            let block = (2 * d) >> 14;
            let span = 1i64 << (2 + (block * 9) % 40);
            let v = (d as i64 * 7919) % span;
            vec![v, v + 3]
        })
        .collect();
    let (dvm, dvd, _) = write_single_dense_sorted_numeric_field(0, &values, &id, "").unwrap();
    let (_, meta) = parse_meta(&dvm, &id, "", &sorted_numeric_field_infos()).unwrap();
    let entry = meta.sorted_numeric_entry(0).unwrap();
    assert!(entry.numeric.block_shift.is_some());
    let mut r = SortedNumericReader::new(&dvd, entry);
    let mut out = Vec::new();
    // Forward, then jumping back across blocks: the cached header must
    // follow every block change.
    let order = (0..40_000)
        .step_by(1)
        .chain([39_999, 5, 20_000, 8_191, 8_192, 0]);
    for d in order {
        r.values(d, &mut out).unwrap();
        assert_eq!(out, values[d as usize], "doc {d}");
    }
    // By rank (a dense field's rank is its document), and a rank past
    // the field's documents refused.
    for d in [0i64, 17, 39_999, 8_192] {
        r.values_at_rank(d, &mut out).unwrap();
        assert_eq!(out, values[d as usize], "rank {d}");
    }
    assert!(matches!(
        r.values_at_rank(40_000, &mut out),
        Err(Error::DocOutOfRange(..))
    ));
    assert!(r.values_at_rank(-1, &mut out).is_err());
    assert!(out.is_empty());
}

/// `gcd_i64` runs Euclid in `u64` where it once did in `u128`; it agrees
/// with that and with `MathUtil.gcd` on every input, `i64::MIN` included.
#[test]
fn gcd_i64_matches_u128_euclid_and_math_util() {
    fn euclid(a: i64, b: i64) -> i64 {
        let (mut a, mut b) = ((a as i128).unsigned_abs(), (b as i128).unsigned_abs());
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a as i64
    }
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x as i64
    };
    for i in 0..20_000u32 {
        let common = next() >> (i % 63);
        let (a, b) = (
            next().wrapping_rem(1 << (i % 20)).wrapping_mul(common),
            next().wrapping_rem(1 << (i % 23)).wrapping_mul(common),
        );
        for (p, q) in [
            (a, b),
            (0, b),
            (a, 0),
            (i64::MIN, b),
            (a, i64::MIN),
            (i64::MIN, 0),
        ] {
            assert_eq!(gcd_i64(p, q), euclid(p, q), "{p} {q}");
            assert_eq!(lucene_util::math_util::gcd(p, q), euclid(p, q), "{p} {q}");
        }
    }
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
        check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id).unwrap(),
        VERSION_WRITE_DEFAULT
    );
    assert_eq!(
        check_data_header_footer(&data_bytes, &id, "").unwrap(),
        VERSION_WRITE_DEFAULT
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

/// `DIRECT_MONOTONIC_BLOCK_SHIFT` guards the *metadata* size: address
/// arrays are bit-packed in blocks of 2^16, so one `(min, avgInc, offset,
/// bitsPerValue)` record covers 65536 addresses instead of one. Writing a
/// variable-length BINARY field with 10x the docs must not grow `.dvm` by
/// anything like 10x -- with a block shift of 0 it would grow by exactly 21
/// bytes per extra doc.
#[test]
fn direct_monotonic_block_shift_keeps_meta_size_constant() {
    let id = [1u8; ID_LENGTH];
    let meta_len = |docs: usize| {
        let values: Vec<Vec<u8>> = (0..docs).map(|i| vec![b'x'; 1 + i % 7]).collect();
        let (meta, _, _) =
            write_single_dense_binary_field(0, &values, docs as i32, &id, "").unwrap();
        meta.len()
    };
    let small = meta_len(100);
    let large = meta_len(1_000);
    assert_eq!(
        small, large,
        "both doc counts fit in one 2^16 address block, so `.dvm` must be the same size"
    );
    // And the addresses still decode: the same 1000-doc field round-trips.
    let values: Vec<Vec<u8>> = (0..1_000).map(|i| vec![b'x'; 1 + i % 7]).collect();
    let (meta, data, _) = write_single_dense_binary_field(0, &values, 1_000, &id, "").unwrap();
    let (_, parsed) = parse_meta(&meta, &id, "", &binary_field_infos()).unwrap();
    let entry = parsed.binary_entry(0).unwrap();
    for (doc, want) in values.iter().enumerate() {
        assert_eq!(
            binary_value(&data, entry, doc as i32).unwrap(),
            Some(want.as_slice()),
            "doc {doc}"
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
fn write_single_sparse_binary_field_interspersed_missing_docs_round_trips() {
    // Missing docs interspersed throughout, not just trailing: doc 0
    // present, doc 1 missing, doc 2 present, etc, plus a fixed-length
    // shape so both the fixed and variable address paths get exercised
    // once each in this module (variable-length is covered by the next
    // test).
    let id = [20u8; ID_LENGTH];
    let max_doc = 10i32;
    let doc_values: Vec<(i32, Vec<u8>)> = (0..max_doc)
        .step_by(2)
        .map(|doc| (doc, vec![doc as u8, doc as u8]))
        .collect();

    let (meta_bytes, data_bytes, skip_bytes) =
        write_single_sparse_binary_field(0, &doc_values, max_doc, &id, "").unwrap();

    assert_eq!(
        check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id).unwrap(),
        VERSION_WRITE_DEFAULT
    );
    assert_eq!(
        check_data_header_footer(&data_bytes, &id, "").unwrap(),
        VERSION_WRITE_DEFAULT
    );

    let fis = binary_field_infos();
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let entry = meta.binary_entry(0).unwrap();
    assert!(!entry.is_dense());
    assert!(!entry.is_empty_field());
    assert!(entry.is_fixed_length());

    let present: std::collections::HashMap<i32, &Vec<u8>> =
        doc_values.iter().map(|(d, v)| (*d, v)).collect();
    for doc in 0..max_doc {
        let got = binary_value(&data_bytes, entry, doc).unwrap();
        assert_eq!(got, present.get(&doc).map(|v| v.as_slice()), "doc {doc}");
    }
}

#[test]
fn write_single_sparse_binary_field_variable_length_round_trips() {
    let id = [21u8; ID_LENGTH];
    let max_doc = 8i32;
    let doc_values: Vec<(i32, Vec<u8>)> = vec![
        (0, b"a".to_vec()),
        (2, b"".to_vec()),
        (3, b"bbbbb".to_vec()),
        (6, b"cc".to_vec()),
    ];

    let (meta_bytes, data_bytes, _skip_bytes) =
        write_single_sparse_binary_field(0, &doc_values, max_doc, &id, "").unwrap();

    let fis = binary_field_infos();
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let entry = meta.binary_entry(0).unwrap();
    assert!(!entry.is_dense());
    assert!(!entry.is_fixed_length());

    let present: std::collections::HashMap<i32, &Vec<u8>> =
        doc_values.iter().map(|(d, v)| (*d, v)).collect();
    for doc in 0..max_doc {
        let got = binary_value(&data_bytes, entry, doc).unwrap();
        assert_eq!(got, present.get(&doc).map(|v| v.as_slice()), "doc {doc}");
    }
}

#[test]
fn write_single_sparse_binary_field_first_doc_missing() {
    let id = [22u8; ID_LENGTH];
    let max_doc = 5i32;
    let doc_values: Vec<(i32, Vec<u8>)> = vec![
        (1, b"a".to_vec()),
        (2, b"b".to_vec()),
        (3, b"c".to_vec()),
        (4, b"d".to_vec()),
    ];

    let (meta_bytes, data_bytes, _skip_bytes) =
        write_single_sparse_binary_field(0, &doc_values, max_doc, &id, "").unwrap();

    let fis = binary_field_infos();
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let entry = meta.binary_entry(0).unwrap();
    assert_eq!(binary_value(&data_bytes, entry, 0).unwrap(), None);
    for (doc, want) in &doc_values {
        assert_eq!(
            binary_value(&data_bytes, entry, *doc).unwrap(),
            Some(want.as_slice())
        );
    }
}

#[test]
fn write_single_sparse_binary_field_last_doc_missing() {
    let id = [23u8; ID_LENGTH];
    let max_doc = 5i32;
    let doc_values: Vec<(i32, Vec<u8>)> = vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"c".to_vec()),
        (3, b"d".to_vec()),
    ];

    let (meta_bytes, data_bytes, _skip_bytes) =
        write_single_sparse_binary_field(0, &doc_values, max_doc, &id, "").unwrap();

    let fis = binary_field_infos();
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let entry = meta.binary_entry(0).unwrap();
    assert_eq!(binary_value(&data_bytes, entry, 4).unwrap(), None);
    for (doc, want) in &doc_values {
        assert_eq!(
            binary_value(&data_bytes, entry, *doc).unwrap(),
            Some(want.as_slice())
        );
    }
}

#[test]
fn write_single_sparse_binary_field_all_but_one_missing() {
    let id = [24u8; ID_LENGTH];
    let max_doc = 1000i32;
    let doc_values: Vec<(i32, Vec<u8>)> = vec![(500, b"only".to_vec())];

    let (meta_bytes, data_bytes, _skip_bytes) =
        write_single_sparse_binary_field(0, &doc_values, max_doc, &id, "").unwrap();

    let fis = binary_field_infos();
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let entry = meta.binary_entry(0).unwrap();
    assert!(!entry.is_dense());
    assert_eq!(entry.num_docs_with_field, 1);
    for doc in (0..max_doc).step_by(37) {
        let got = binary_value(&data_bytes, entry, doc).unwrap();
        if doc == 500 {
            assert_eq!(got, Some(b"only".as_slice()));
        } else {
            assert_eq!(got, None, "doc {doc}");
        }
    }
    assert_eq!(
        binary_value(&data_bytes, entry, 500).unwrap(),
        Some(b"only".as_slice())
    );
}

#[test]
fn write_single_sparse_binary_field_rejects_duplicate_doc_id() {
    let id = [25u8; ID_LENGTH];
    let err =
        write_single_sparse_binary_field(0, &[(1, b"a".to_vec()), (1, b"b".to_vec())], 5, &id, "")
            .unwrap_err();
    assert!(matches!(err, WriteError::DocIdsNotAscending(1)));
}

#[test]
fn write_single_sparse_binary_field_rejects_out_of_range_doc_id() {
    let id = [26u8; ID_LENGTH];
    let err =
        write_single_sparse_binary_field(0, &[(0, b"a".to_vec()), (5, b"b".to_vec())], 5, &id, "")
            .unwrap_err();
    assert!(matches!(err, WriteError::DocIdOutOfRange(5, 5)));
}

#[test]
fn write_single_dense_binary_field_still_dense_after_sparse_addition() {
    // Regression: adding the sparse write path must not change the
    // dense path's output at all. Same values/assertions as
    // `write_single_dense_binary_field_fixed_length_round_trips`.
    let id = [27u8; ID_LENGTH];
    let values: Vec<Vec<u8>> = vec![b"aa".to_vec(), b"bb".to_vec(), b"cc".to_vec()];
    let (meta_bytes, data_bytes, _skip_bytes) =
        write_single_dense_binary_field(0, &values, values.len() as i32, &id, "").unwrap();

    let fis = binary_field_infos();
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let entry = meta.binary_entry(0).unwrap();
    assert!(entry.is_dense());
    assert!(entry.is_fixed_length());
    for (doc, want) in values.iter().enumerate() {
        assert_eq!(
            binary_value(&data_bytes, entry, doc as i32).unwrap(),
            Some(want.as_slice())
        );
    }
}

#[test]
fn write_single_dense_sorted_numeric_field_multi_valued_round_trips() {
    let id = [12u8; ID_LENGTH];
    let values: Vec<Vec<i64>> = vec![vec![1], vec![2, 3], vec![4, 5, 6], vec![0]];
    let (meta_bytes, data_bytes, skip_bytes) =
        write_single_dense_sorted_numeric_field(0, &values, &id, "").unwrap();

    assert_eq!(
        check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id).unwrap(),
        VERSION_WRITE_DEFAULT
    );
    assert_eq!(
        check_data_header_footer(&data_bytes, &id, "").unwrap(),
        VERSION_WRITE_DEFAULT
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

/// `doBlocks` on a SORTED_NUMERIC column's flat values: a constant
/// block (`0u8` + value), a narrow block and a wide one, all multiples of
/// the column's GCD, plus a partial trailing block -- read back value
/// for value, and the meta says blocks (`tableSize = -16`, width 0xFF).
#[test]
fn sorted_numeric_values_split_into_varying_width_blocks() {
    let id = [15u8; ID_LENGTH];
    let block = NUMERIC_BLOCK_SIZE;
    // Two values per document, so the three full blocks plus a tail
    // cover 1.5 * `block` + 100 documents.
    let flat: Vec<i64> = (0..3 * block + 200)
        .map(|i| match i / block {
            0 => 42 * 3,
            1 => (i % 4) as i64 * 3,
            _ => (i as i64 * 7919 % 1_000_003) * 3 * 1_000_000,
        })
        .collect();
    let values: Vec<Vec<i64>> = flat
        .chunks(2)
        .map(|c| {
            let mut v = c.to_vec();
            v.sort_unstable();
            v
        })
        .collect();
    let (meta_bytes, data_bytes, _skip) =
        write_single_dense_sorted_numeric_field(0, &values, &id, "").unwrap();
    let fis = sorted_numeric_field_infos();
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let entry = meta.sorted_numeric_entry(0).unwrap();
    assert_eq!(entry.numeric.block_shift, Some(NUMERIC_BLOCK_SHIFT));
    assert_eq!(entry.numeric.bits_per_value, 0xFF);
    assert_eq!(entry.numeric.gcd, 3);
    for (doc, want) in values.iter().enumerate() {
        assert_eq!(
            sorted_numeric_values(&data_bytes, entry, doc as i32).unwrap(),
            *want,
            "doc {doc}"
        );
    }
    // Blocks that do not save 10%: one uniform width.
    let uniform: Vec<i64> = (0..2 * block as i64).map(|i| i % 1000).collect();
    let mut meta = Vec::new();
    let mut data = Vec::new();
    write_numeric_values_body(&mut meta, &mut data, &uniform);
    assert_eq!(&meta[8..12], &(-1i32).to_be_bytes());
    assert_eq!(space_in_bits(&[]), 0);
    assert_eq!(space_in_bits(&[5, 5]), 0);
    assert_eq!(space_in_bits(&[0, 3]), 4);
}

#[test]
fn write_single_dense_sorted_numeric_field_rejects_empty_doc() {
    let id = [1u8; ID_LENGTH];
    let values: Vec<Vec<i64>> = vec![vec![1], Vec::new(), vec![2]];
    let err = write_single_dense_sorted_numeric_field(0, &values, &id, "").unwrap_err();
    assert!(matches!(err, WriteError::EmptyMultiValuedDoc(1)));
}

// --- write_single_sparse_sorted_numeric_field ---

#[test]
fn write_single_sparse_sorted_numeric_field_interspersed_missing_docs_round_trips() {
    let id = [40u8; ID_LENGTH];
    let max_doc = 10i32;
    let doc_values: Vec<(i32, Vec<i64>)> = (0..max_doc)
        .step_by(2)
        .map(|doc| (doc, vec![doc as i64 * 10]))
        .collect();

    let (meta_bytes, data_bytes, skip_bytes) =
        write_single_sparse_sorted_numeric_field(0, &doc_values, max_doc, &id, "").unwrap();

    assert_eq!(
        check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id).unwrap(),
        VERSION_WRITE_DEFAULT
    );
    assert_eq!(
        check_data_header_footer(&data_bytes, &id, "").unwrap(),
        VERSION_WRITE_DEFAULT
    );

    let fis = sorted_numeric_field_infos();
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let entry = meta.sorted_numeric_entry(0).unwrap();
    let present: std::collections::HashMap<i32, Vec<i64>> = doc_values.into_iter().collect();
    for doc in 0..max_doc {
        let got = sorted_numeric_values(&data_bytes, entry, doc).unwrap();
        match present.get(&doc) {
            Some(want) => assert_eq!(got, *want, "doc {doc}"),
            None => assert_eq!(got, Vec::<i64>::new(), "doc {doc}"),
        }
    }
}

#[test]
fn write_single_sparse_sorted_numeric_field_first_doc_missing() {
    let id = [41u8; ID_LENGTH];
    let max_doc = 5i32;
    let doc_values: Vec<(i32, Vec<i64>)> =
        vec![(1, vec![10]), (2, vec![20]), (3, vec![30]), (4, vec![40])];

    let (meta_bytes, data_bytes, _skip_bytes) =
        write_single_sparse_sorted_numeric_field(0, &doc_values, max_doc, &id, "").unwrap();

    let fis = sorted_numeric_field_infos();
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let entry = meta.sorted_numeric_entry(0).unwrap();
    assert_eq!(
        sorted_numeric_values(&data_bytes, entry, 0).unwrap(),
        Vec::<i64>::new()
    );
    for (doc, want) in &doc_values {
        assert_eq!(
            sorted_numeric_values(&data_bytes, entry, *doc).unwrap(),
            *want
        );
    }
}

#[test]
fn write_single_sparse_sorted_numeric_field_last_doc_missing() {
    let id = [42u8; ID_LENGTH];
    let max_doc = 5i32;
    let doc_values: Vec<(i32, Vec<i64>)> =
        vec![(0, vec![10]), (1, vec![20]), (2, vec![30]), (3, vec![40])];

    let (meta_bytes, data_bytes, _skip_bytes) =
        write_single_sparse_sorted_numeric_field(0, &doc_values, max_doc, &id, "").unwrap();

    let fis = sorted_numeric_field_infos();
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let entry = meta.sorted_numeric_entry(0).unwrap();
    assert_eq!(
        sorted_numeric_values(&data_bytes, entry, 4).unwrap(),
        Vec::<i64>::new()
    );
    for (doc, want) in &doc_values {
        assert_eq!(
            sorted_numeric_values(&data_bytes, entry, *doc).unwrap(),
            *want
        );
    }
}

#[test]
fn write_single_sparse_sorted_numeric_field_all_but_one_missing() {
    let id = [43u8; ID_LENGTH];
    let max_doc = 1000i32;
    let doc_values: Vec<(i32, Vec<i64>)> = vec![(500, vec![7, 8, 9])];

    let (meta_bytes, data_bytes, _skip_bytes) =
        write_single_sparse_sorted_numeric_field(0, &doc_values, max_doc, &id, "").unwrap();

    let fis = sorted_numeric_field_infos();
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let entry = meta.sorted_numeric_entry(0).unwrap();
    for doc in (0..max_doc).step_by(37) {
        let got = sorted_numeric_values(&data_bytes, entry, doc).unwrap();
        if doc == 500 {
            assert_eq!(got, vec![7, 8, 9]);
        } else {
            assert_eq!(got, Vec::<i64>::new(), "doc {doc}");
        }
    }
}

#[test]
fn write_single_sparse_sorted_numeric_field_present_doc_multiple_values() {
    // A present doc with 3 values, interspersed with missing docs and
    // single-valued present docs -- exercises the address-array branch
    // together with sparsity.
    let id = [44u8; ID_LENGTH];
    let max_doc = 6i32;
    let doc_values: Vec<(i32, Vec<i64>)> = vec![(0, vec![1]), (2, vec![2, 3, 4]), (5, vec![5])];

    let (meta_bytes, data_bytes, _skip_bytes) =
        write_single_sparse_sorted_numeric_field(0, &doc_values, max_doc, &id, "").unwrap();

    let fis = sorted_numeric_field_infos();
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let entry = meta.sorted_numeric_entry(0).unwrap();
    assert!(entry.addresses.is_some());

    let present: std::collections::HashMap<i32, Vec<i64>> = doc_values.into_iter().collect();
    for doc in 0..max_doc {
        let got = sorted_numeric_values(&data_bytes, entry, doc).unwrap();
        match present.get(&doc) {
            Some(want) => assert_eq!(got, *want, "doc {doc}"),
            None => assert_eq!(got, Vec::<i64>::new(), "doc {doc}"),
        }
    }
}

#[test]
fn write_single_sparse_sorted_numeric_field_rejects_duplicate_doc_id() {
    let id = [45u8; ID_LENGTH];
    let err =
        write_single_sparse_sorted_numeric_field(0, &[(1, vec![10]), (1, vec![20])], 5, &id, "")
            .unwrap_err();
    assert!(matches!(err, WriteError::DocIdsNotAscending(1)));
}

#[test]
fn write_single_sparse_sorted_numeric_field_rejects_out_of_range_doc_id() {
    let id = [46u8; ID_LENGTH];
    let err =
        write_single_sparse_sorted_numeric_field(0, &[(0, vec![10]), (5, vec![20])], 5, &id, "")
            .unwrap_err();
    assert!(matches!(err, WriteError::DocIdOutOfRange(5, 5)));
}

#[test]
fn write_single_sparse_sorted_numeric_field_rejects_empty_per_doc_value_list() {
    let id = [47u8; ID_LENGTH];
    let err = write_single_sparse_sorted_numeric_field(
        0,
        &[(0, vec![10]), (1, vec![]), (2, vec![30])],
        5,
        &id,
        "",
    )
    .unwrap_err();
    assert!(matches!(err, WriteError::EmptyMultiValuedDoc(1)));
}

#[test]
fn write_single_dense_sorted_numeric_field_still_dense_after_sparse_addition() {
    // Regression: adding the sparse SORTED_NUMERIC write path must not
    // change the dense path's output at all. Same values/assertions as
    // `write_single_dense_sorted_numeric_field_multi_valued_round_trips`.
    let id = [12u8; ID_LENGTH];
    let values: Vec<Vec<i64>> = vec![vec![1], vec![2, 3], vec![4, 5, 6], vec![0]];
    let (meta_bytes, data_bytes, skip_bytes) =
        write_single_dense_sorted_numeric_field(0, &values, &id, "").unwrap();

    assert_eq!(
        check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id).unwrap(),
        VERSION_WRITE_DEFAULT
    );
    assert_eq!(
        check_data_header_footer(&data_bytes, &id, "").unwrap(),
        VERSION_WRITE_DEFAULT
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

#[test]
fn an_overflowing_values_region_is_a_decode_error_not_an_overflow() {
    // `valuesOffset` and `valuesLength` are two independent `i64`s in
    // `.dvm` with nothing on the wire relating them. `offset + length`
    // panics in a debug build and, in a release build, wraps to a
    // *plausible in-range* offset that decodes valid-looking doc values
    // out of the wrong part of `.dvd` -- the same silent-wrong-answer
    // shape.
    let id = [11u8; ID_LENGTH];
    let (meta_bytes, data_bytes, _) =
        write_single_dense_numeric_field(0, &[1i64, 2, 3, 4], 4, &id, "").unwrap();
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &field_infos_with(&[0])).unwrap();
    let mut entry = meta.numeric_entry(0).unwrap().clone();
    assert_ne!(entry.bits_per_value, 0, "the test needs a packed field");
    entry.values_offset = i64::MAX;
    entry.values_length = 1;
    assert!(numeric_value(&data_bytes, &entry, 0).is_err());
}

#[test]
fn an_overflowing_docs_with_field_region_is_a_decode_error_not_an_overflow() {
    // Same two-independent-`i64`s shape on the `IndexedDISI` region, the
    // one every sparse field's per-doc lookup goes through.
    let id = [12u8; ID_LENGTH];
    let (meta_bytes, data_bytes, _) =
        write_single_sparse_numeric_field(0, &[(0i32, 5i64), (3, 7)], 4, &id, "").unwrap();
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &field_infos_with(&[0])).unwrap();
    let mut entry = meta.numeric_entry(0).unwrap().clone();
    assert!(!entry.is_dense() && !entry.is_empty_field());
    entry.docs_with_field_offset = i64::MAX;
    entry.docs_with_field_length = 1;
    assert!(numeric_value(&data_bytes, &entry, 0).is_err());
}

#[test]
fn a_sorted_numeric_address_range_past_the_value_array_is_a_decode_error_not_a_terabyte_vec() {
    // A multi-valued field's address array names a `[start, end)` range
    // in the field's flat value array. Both ends come out of a
    // `direct_monotonic` array in `.dvd`, and nothing tied them to the
    // `numValues` the entry declares. For a *constant*-encoded field
    // (`bitsPerValue == 0`) `decode_value` never indexes anything and so
    // never fails, and the `collect` below therefore builds one `i64` per
    // element of whatever range the file named -- 8 TB here, which is an
    // allocation **abort** that `catch_unwind` at the FFI boundary cannot
    // intercept.
    //
    // The address array is built through `direct_monotonic`'s own public
    // writer, so this is a range a real `.dvd` can carry, not a
    // hand-forged `Meta`.
    let (addr_meta_bytes, addr_data) = direct_monotonic::write(&[0i64, 1 << 40], 10);
    let mut meta_input = SliceInput::new(&addr_meta_bytes);
    let addr_meta = direct_monotonic::load_meta(&mut meta_input, 2, 10).unwrap();
    let entry = SortedNumericEntry {
        field_number: 0,
        numeric: NumericEntry {
            disi: Default::default(),
            field_number: 0,
            docs_with_field_offset: DOCS_WITH_FIELD_DENSE,
            docs_with_field_length: 0,
            jump_table_entry_count: -1,
            dense_rank_power: 0,
            num_values: 2,
            table: None,
            bits_per_value: 0,
            min_value: 42,
            gcd: 1,
            values_offset: 0,
            values_length: 0,
            block_shift: None,
            value_jump_table_offset: -1,
            adaptive: None,
        },
        num_docs_with_field: 1,
        addresses: Some(MultiValueAddresses {
            offset: 0,
            length: addr_data.len() as i64,
            meta: addr_meta,
        }),
    };
    let err = sorted_numeric_values(&addr_data, &entry, 0).unwrap_err();
    assert!(
        matches!(err, Error::CorruptAddressRange { start: 0, end, .. } if end == 1 << 40),
        "{err}"
    );
}

/// Recomputes the trailing 8-byte CRC32 of a codec-footer-terminated
/// buffer in place, so a corruption sweep cannot be "caught" by the
/// checksum and only semantic invariants can fire (c15/c19/c25's shape).
fn resign_footer(buf: &mut [u8]) {
    let len = buf.len();
    let checksum = crc32fast::hash(&buf[..len - 8]) as u64;
    buf[len - 8..].copy_from_slice(&checksum.to_be_bytes());
}

/// Re-signed single-byte corruption sweep over `.dvm` and `.dvd` for
/// every doc-values type this module can write.
///
/// The point is not the rejection rate (a great many single-bit flips
/// produce a self-consistent file that decodes to different but
/// well-formed values -- that is what a checksum is for). The point is
/// that **nothing panics and nothing aborts**: `clippy::
/// arithmetic_side_effects` covers arithmetic and shifts, not indexing,
/// slicing or allocation. An allocation abort is the worst case,
/// because `catch_unwind` at the FFI boundary cannot intercept it.
#[test]
fn every_resigned_single_byte_dvm_and_dvd_corruption_is_an_error_or_a_clean_decode() {
    let id = [77u8; ID_LENGTH];
    let footer = codec_util::FOOTER_LENGTH;

    // NUMERIC (dense), BINARY (variable width, so an address array),
    // SORTED (a terms dictionary), SORTED_NUMERIC and SORTED_SET (both
    // multi-valued, so a second address array) -- five different meta
    // shapes over the same sweep.
    let numeric = write_single_dense_numeric_field(0, &[7i64, 7, 9, 11], 4, &id, "").unwrap();
    let binary = write_single_dense_binary_field(
        0,
        &[b"a".to_vec(), b"bb".to_vec(), b"ccc".to_vec()],
        3,
        &id,
        "",
    )
    .unwrap();
    let sorted = write_single_dense_sorted_field(
        0,
        &[b"banana".to_vec(), b"apple".to_vec(), b"cherry".to_vec()],
        3,
        &id,
        "",
    )
    .unwrap();
    let sorted_numeric =
        write_single_dense_sorted_numeric_field(0, &[vec![1i64, 2], vec![3]], &id, "").unwrap();
    let sorted_set = write_single_dense_sorted_set_field(
        0,
        &[vec![b"x".to_vec(), b"y".to_vec()], vec![b"z".to_vec()]],
        2,
        &id,
        "",
    )
    .unwrap();

    let mut total = 0usize;
    let mut rejected = 0usize;

    let mut sweep = |meta: &[u8], data: &[u8], fis: &FieldInfos, max_doc: i32| {
        let read = |meta: &[u8], data: &[u8]| -> Result<()> {
            let (_, m) = parse_meta(meta, &id, "", fis)?;
            check_data_header_footer(data, &id, "")?;
            for doc in 0..max_doc {
                if let Some(e) = m.numeric_entry(0) {
                    numeric_value(data, e, doc)?;
                }
                if let Some(e) = m.binary_entry(0) {
                    binary_value(data, e, doc)?;
                }
                if let Some(e) = m.sorted_entry(0) {
                    sorted_ord(data, e, doc)?;
                    terms_dict::decode_all_terms(data, &e.terms)?;
                }
                if let Some(e) = m.sorted_numeric_entry(0) {
                    sorted_numeric_values(data, e, doc)?;
                }
                if let Some(e) = m.sorted_set_entry(0) {
                    match &e.kind {
                        SortedSetKind::Single(s) => {
                            sorted_ord(data, s, doc)?;
                            terms_dict::decode_all_terms(data, &s.terms)?;
                        }
                        SortedSetKind::Multi { ords, terms } => {
                            sorted_numeric_values(data, ords, doc)?;
                            terms_dict::decode_all_terms(data, terms)?;
                        }
                    }
                }
            }
            Ok(())
        };
        for off in 0..meta.len() - footer {
            for mask in [0x01u8, 0x80] {
                let mut corrupt = meta.to_vec();
                corrupt[off] ^= mask;
                resign_footer(&mut corrupt);
                total += 1;
                if read(&corrupt, data).is_err() {
                    rejected += 1;
                }
            }
        }
        for off in 0..data.len() - footer {
            for mask in [0x01u8, 0x80] {
                let mut corrupt = data.to_vec();
                corrupt[off] ^= mask;
                resign_footer(&mut corrupt);
                total += 1;
                if read(meta, &corrupt).is_err() {
                    rejected += 1;
                }
            }
        }
    };

    // The four sparse encoders are the only paths that emit an
    // `IndexedDISI` region, so without them the sweep never
    // exercises the sparse *reader* either (`DisiCursor` + rank
    // ordinals). One multi-field container covers all four at once.
    let sparse = write_dense_fields(
        &[
            DenseField::SparseBinary(0, &[(0, b"aa".to_vec()), (3, b"cccc".to_vec())]),
            DenseField::SparseSorted(1, &[(1, b"pear".to_vec()), (4, b"apple".to_vec())]),
            DenseField::SparseSortedNumeric(2, &[(0, vec![7, 9]), (2, vec![4])]),
            DenseField::SparseSortedSet(
                3,
                &[
                    (2, vec![b"x".to_vec(), b"y".to_vec()]),
                    (4, vec![b"z".to_vec()]),
                ],
            ),
        ],
        5,
        &id,
        "",
    )
    .unwrap();

    sweep(&numeric.0, &numeric.1, &field_infos_with(&[0]), 4);
    let mut fis = field_infos_with(&[0]);
    fis.fields[0].doc_values_type = DocValuesType::Binary;
    sweep(&binary.0, &binary.1, &fis, 3);
    sweep(&sorted.0, &sorted.1, &sorted_field_infos(), 3);
    let mut fis = field_infos_with(&[0]);
    fis.fields[0].doc_values_type = DocValuesType::SortedNumeric;
    sweep(&sorted_numeric.0, &sorted_numeric.1, &fis, 2);
    let mut fis = field_infos_with(&[0]);
    fis.fields[0].doc_values_type = DocValuesType::SortedSet;
    sweep(&sorted_set.0, &sorted_set.1, &fis, 2);
    let sparse_fis = mixed_field_infos(
        &[0, 1, 2, 3],
        &[
            DocValuesType::Binary,
            DocValuesType::Sorted,
            DocValuesType::SortedNumeric,
            DocValuesType::SortedSet,
        ],
    );
    sweep(&sparse.0, &sparse.1, &sparse_fis, 5);

    // Measured when this was written: 2 151 of 4 298 (50%), over six
    // containers covering all five doc-values types in their dense shape
    // and four of them in their sparse shape. For scale, c19/c25
    // measured `.nvm` at 85/99, `.tip` at 44/99, `.dvd` at 18/99 and
    // `.tvd` at 15/43. The floor is what the assertion is for; the exact
    // number is recorded so a change that stops bounding something is
    // visible in the diff rather than only in a crash.
    assert_eq!(total, 4298);
    assert!(
        rejected >= 2100,
        "only {rejected} of {total} re-signed .dvm/.dvd corruptions were rejected"
    );
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
        check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id).unwrap(),
        VERSION_WRITE_DEFAULT
    );
    assert_eq!(
        check_data_header_footer(&data_bytes, &id, "").unwrap(),
        VERSION_WRITE_DEFAULT
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
    // sample past the first.
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

// --- write_single_sparse_sorted_field ---

fn resolved_sparse_sorted_values(
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
fn write_single_sparse_sorted_field_interspersed_missing_docs_round_trips() {
    let id = [30u8; ID_LENGTH];
    let max_doc = 10i32;
    let doc_values: Vec<(i32, Vec<u8>)> = (0..max_doc)
        .step_by(2)
        .map(|doc| (doc, format!("term{doc}").into_bytes()))
        .collect();

    let (meta_bytes, data_bytes, skip_bytes) =
        write_single_sparse_sorted_field(0, &doc_values, max_doc, &id, "").unwrap();

    assert_eq!(
        check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id).unwrap(),
        VERSION_WRITE_DEFAULT
    );
    assert_eq!(
        check_data_header_footer(&data_bytes, &id, "").unwrap(),
        VERSION_WRITE_DEFAULT
    );

    let fis = sorted_field_infos();
    let entry = read_sorted_field(&meta_bytes, &id, &fis);
    let expected: std::collections::HashMap<i32, Vec<u8>> = doc_values.into_iter().collect();
    let resolved = resolved_sparse_sorted_values(&data_bytes, &entry, max_doc);
    for doc in 0..max_doc {
        assert_eq!(
            resolved[doc as usize],
            expected.get(&doc).cloned(),
            "doc {doc}"
        );
    }
}

#[test]
fn write_single_sparse_sorted_field_first_doc_missing() {
    let id = [31u8; ID_LENGTH];
    let max_doc = 5i32;
    let doc_values: Vec<(i32, Vec<u8>)> = vec![
        (1, b"a".to_vec()),
        (2, b"b".to_vec()),
        (3, b"c".to_vec()),
        (4, b"d".to_vec()),
    ];

    let (meta_bytes, data_bytes, _skip_bytes) =
        write_single_sparse_sorted_field(0, &doc_values, max_doc, &id, "").unwrap();

    let fis = sorted_field_infos();
    let entry = read_sorted_field(&meta_bytes, &id, &fis);
    assert_eq!(sorted_ord(&data_bytes, &entry, 0).unwrap(), None);
    let resolved = resolved_sparse_sorted_values(&data_bytes, &entry, max_doc);
    for (doc, want) in &doc_values {
        assert_eq!(resolved[*doc as usize], Some(want.clone()));
    }
}

#[test]
fn write_single_sparse_sorted_field_last_doc_missing() {
    let id = [32u8; ID_LENGTH];
    let max_doc = 5i32;
    let doc_values: Vec<(i32, Vec<u8>)> = vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"c".to_vec()),
        (3, b"d".to_vec()),
    ];

    let (meta_bytes, data_bytes, _skip_bytes) =
        write_single_sparse_sorted_field(0, &doc_values, max_doc, &id, "").unwrap();

    let fis = sorted_field_infos();
    let entry = read_sorted_field(&meta_bytes, &id, &fis);
    assert_eq!(sorted_ord(&data_bytes, &entry, 4).unwrap(), None);
    let resolved = resolved_sparse_sorted_values(&data_bytes, &entry, max_doc);
    for (doc, want) in &doc_values {
        assert_eq!(resolved[*doc as usize], Some(want.clone()));
    }
}

#[test]
fn write_single_sparse_sorted_field_all_but_one_missing() {
    let id = [33u8; ID_LENGTH];
    let max_doc = 1000i32;
    let doc_values: Vec<(i32, Vec<u8>)> = vec![(500, b"only".to_vec())];

    let (meta_bytes, data_bytes, _skip_bytes) =
        write_single_sparse_sorted_field(0, &doc_values, max_doc, &id, "").unwrap();

    let fis = sorted_field_infos();
    let entry = read_sorted_field(&meta_bytes, &id, &fis);
    for doc in (0..max_doc).step_by(37) {
        let got = sorted_ord(&data_bytes, &entry, doc).unwrap();
        if doc == 500 {
            assert!(got.is_some());
        } else {
            assert_eq!(got, None, "doc {doc}");
        }
    }
    let dict = terms_dict::decode_all_terms(&data_bytes, &entry.terms).unwrap();
    assert_eq!(dict, vec![b"only".to_vec()]);
    let ord = sorted_ord(&data_bytes, &entry, 500).unwrap().unwrap();
    assert_eq!(dict[ord as usize], b"only".to_vec());
}

#[test]
fn write_single_sparse_sorted_field_rejects_duplicate_doc_id() {
    let id = [34u8; ID_LENGTH];
    let err =
        write_single_sparse_sorted_field(0, &[(1, b"a".to_vec()), (1, b"b".to_vec())], 5, &id, "")
            .unwrap_err();
    assert!(matches!(err, WriteError::DocIdsNotAscending(1)));
}

#[test]
fn write_single_sparse_sorted_field_rejects_out_of_range_doc_id() {
    let id = [35u8; ID_LENGTH];
    let err =
        write_single_sparse_sorted_field(0, &[(0, b"a".to_vec()), (5, b"b".to_vec())], 5, &id, "")
            .unwrap_err();
    assert!(matches!(err, WriteError::DocIdOutOfRange(5, 5)));
}

#[test]
fn write_single_dense_sorted_field_still_dense_after_sparse_addition() {
    // Regression: adding the sparse SORTED write path must not change
    // the dense path's output at all. Same values/assertions as
    // `write_single_dense_sorted_field_round_trips_small_dictionary`.
    let id = [36u8; ID_LENGTH];
    let values: Vec<Vec<u8>> = vec![
        b"banana".to_vec(),
        b"apple".to_vec(),
        b"cherry".to_vec(),
        b"apple".to_vec(),
    ];
    let (meta_bytes, data_bytes, skip_bytes) =
        write_single_dense_sorted_field(0, &values, values.len() as i32, &id, "").unwrap();

    assert_eq!(
        check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id).unwrap(),
        VERSION_WRITE_DEFAULT
    );
    assert_eq!(
        check_data_header_footer(&data_bytes, &id, "").unwrap(),
        VERSION_WRITE_DEFAULT
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
        check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id).unwrap(),
        VERSION_WRITE_DEFAULT
    );
    assert_eq!(
        check_data_header_footer(&data_bytes, &id, "").unwrap(),
        VERSION_WRITE_DEFAULT
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
    let err =
        write_single_dense_sorted_set_field(0, &values, values.len() as i32, &id, "").unwrap_err();
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

// --- write_single_sparse_sorted_set_field ---

#[test]
fn write_single_sparse_sorted_set_field_interspersed_missing_docs_round_trips() {
    let id = [50u8; ID_LENGTH];
    let max_doc = 10i32;
    let doc_values: Vec<(i32, Vec<Vec<u8>>)> = (0..max_doc)
        .step_by(2)
        .map(|doc| (doc, vec![format!("v{doc}").into_bytes()]))
        .collect();

    let (meta_bytes, data_bytes, skip_bytes) =
        write_single_sparse_sorted_set_field(0, &doc_values, max_doc, &id, "").unwrap();

    assert_eq!(
        check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id).unwrap(),
        VERSION_WRITE_DEFAULT
    );
    assert_eq!(
        check_data_header_footer(&data_bytes, &id, "").unwrap(),
        VERSION_WRITE_DEFAULT
    );

    let fis = sorted_set_field_infos();
    let entry = read_sorted_set_field(&meta_bytes, &id, &fis);
    // One value per present doc: Java's `isSingleValued` collapse.
    assert!(matches!(entry.kind, SortedSetKind::Single(_)));

    let resolved = resolved_sorted_set_values(&data_bytes, &entry, max_doc);
    let present: std::collections::HashMap<i32, Vec<Vec<u8>>> = doc_values.into_iter().collect();
    for doc in 0..max_doc {
        match present.get(&doc) {
            Some(want) => assert_eq!(resolved[doc as usize], *want, "doc {doc}"),
            None => assert_eq!(resolved[doc as usize], Vec::<Vec<u8>>::new(), "doc {doc}"),
        }
    }
}

#[test]
fn write_single_sparse_sorted_set_field_first_doc_missing() {
    let id = [51u8; ID_LENGTH];
    let max_doc = 5i32;
    let doc_values: Vec<(i32, Vec<Vec<u8>>)> = vec![
        (1, vec![b"a".to_vec()]),
        (2, vec![b"b".to_vec()]),
        (3, vec![b"c".to_vec()]),
        (4, vec![b"d".to_vec()]),
    ];

    let (meta_bytes, data_bytes, _skip_bytes) =
        write_single_sparse_sorted_set_field(0, &doc_values, max_doc, &id, "").unwrap();

    let fis = sorted_set_field_infos();
    let entry = read_sorted_set_field(&meta_bytes, &id, &fis);
    let resolved = resolved_sorted_set_values(&data_bytes, &entry, max_doc);
    assert_eq!(resolved[0], Vec::<Vec<u8>>::new());
    for (doc, want) in &doc_values {
        assert_eq!(resolved[*doc as usize], *want);
    }
}

#[test]
fn write_single_sparse_sorted_set_field_last_doc_missing() {
    let id = [52u8; ID_LENGTH];
    let max_doc = 5i32;
    let doc_values: Vec<(i32, Vec<Vec<u8>>)> = vec![
        (0, vec![b"a".to_vec()]),
        (1, vec![b"b".to_vec()]),
        (2, vec![b"c".to_vec()]),
        (3, vec![b"d".to_vec()]),
    ];

    let (meta_bytes, data_bytes, _skip_bytes) =
        write_single_sparse_sorted_set_field(0, &doc_values, max_doc, &id, "").unwrap();

    let fis = sorted_set_field_infos();
    let entry = read_sorted_set_field(&meta_bytes, &id, &fis);
    let resolved = resolved_sorted_set_values(&data_bytes, &entry, max_doc);
    assert_eq!(resolved[4], Vec::<Vec<u8>>::new());
    for (doc, want) in &doc_values {
        assert_eq!(resolved[*doc as usize], *want);
    }
}

#[test]
fn write_single_sparse_sorted_set_field_all_but_one_missing() {
    let id = [53u8; ID_LENGTH];
    let max_doc = 100i32;
    let doc_values: Vec<(i32, Vec<Vec<u8>>)> = vec![(42, vec![b"only".to_vec()])];

    let (meta_bytes, data_bytes, _skip_bytes) =
        write_single_sparse_sorted_set_field(0, &doc_values, max_doc, &id, "").unwrap();

    let fis = sorted_set_field_infos();
    let entry = read_sorted_set_field(&meta_bytes, &id, &fis);
    let resolved = resolved_sorted_set_values(&data_bytes, &entry, max_doc);
    for doc in 0..max_doc {
        if doc == 42 {
            assert_eq!(resolved[doc as usize], vec![b"only".to_vec()]);
        } else {
            assert_eq!(resolved[doc as usize], Vec::<Vec<u8>>::new(), "doc {doc}");
        }
    }
}

#[test]
fn write_single_sparse_sorted_set_field_present_doc_multiple_values_dedups_and_sorts() {
    let id = [54u8; ID_LENGTH];
    let max_doc = 3i32;
    let doc_values: Vec<(i32, Vec<Vec<u8>>)> = vec![
        (
            0,
            vec![
                b"cherry".to_vec(),
                b"apple".to_vec(),
                b"apple".to_vec(),
                b"banana".to_vec(),
            ],
        ),
        (2, vec![b"apple".to_vec()]),
    ];

    let (meta_bytes, data_bytes, _skip_bytes) =
        write_single_sparse_sorted_set_field(0, &doc_values, max_doc, &id, "").unwrap();

    let fis = sorted_set_field_infos();
    let entry = read_sorted_set_field(&meta_bytes, &id, &fis);
    assert!(matches!(entry.kind, SortedSetKind::Multi { .. }));

    let resolved = resolved_sorted_set_values(&data_bytes, &entry, max_doc);
    assert_eq!(
        resolved[0],
        vec![b"apple".to_vec(), b"banana".to_vec(), b"cherry".to_vec()]
    );
    assert_eq!(resolved[1], Vec::<Vec<u8>>::new());
    assert_eq!(resolved[2], vec![b"apple".to_vec()]);
}

#[test]
fn write_single_sparse_sorted_set_field_rejects_duplicate_doc_id() {
    let id = [55u8; ID_LENGTH];
    let err = write_single_sparse_sorted_set_field(
        0,
        &[(1, vec![b"a".to_vec()]), (1, vec![b"b".to_vec()])],
        5,
        &id,
        "",
    )
    .unwrap_err();
    assert!(matches!(err, WriteError::DocIdsNotAscending(1)));
}

#[test]
fn write_single_sparse_sorted_set_field_rejects_out_of_range_doc_id() {
    let id = [56u8; ID_LENGTH];
    let err = write_single_sparse_sorted_set_field(
        0,
        &[(0, vec![b"a".to_vec()]), (5, vec![b"b".to_vec()])],
        5,
        &id,
        "",
    )
    .unwrap_err();
    assert!(matches!(err, WriteError::DocIdOutOfRange(5, 5)));
}

#[test]
fn write_single_sparse_sorted_set_field_rejects_empty_per_doc_value_list() {
    let id = [57u8; ID_LENGTH];
    let err = write_single_sparse_sorted_set_field(
        0,
        &[(0, vec![b"a".to_vec()]), (1, Vec::new())],
        5,
        &id,
        "",
    )
    .unwrap_err();
    assert!(matches!(err, WriteError::EmptyMultiValuedDoc(1)));
}

#[test]
fn write_single_dense_sorted_set_field_still_dense_after_sparse_addition() {
    // Regression: adding the sparse write path must not change the
    // dense path's output at all. Same values/assertions as
    // `write_single_dense_sorted_set_field_small_dictionary_shared_across_docs`.
    let id = [58u8; ID_LENGTH];
    let values: Vec<Vec<Vec<u8>>> = vec![
        vec![b"apple".to_vec(), b"cherry".to_vec()],
        vec![b"banana".to_vec()],
        vec![b"apple".to_vec(), b"banana".to_vec(), b"cherry".to_vec()],
        vec![b"banana".to_vec()],
    ];
    let (meta_bytes, data_bytes, _skip_bytes) =
        write_single_dense_sorted_set_field(0, &values, values.len() as i32, &id, "").unwrap();

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

// --- write_dense_fields (multi-field .dvm/.dvd/.dvs) ---

/// Builds a [`FieldInfos`] with `numbers.len()` fields, field `numbers[i]`
/// given `types[i]` -- unlike [`field_infos_with`] (always NUMERIC), this
/// lets a multi-field test mix doc-values types the way
/// [`write_dense_fields`] itself does.
fn mixed_field_infos(numbers: &[i32], types: &[DocValuesType]) -> FieldInfos {
    FieldInfos {
        fields: numbers
            .iter()
            .zip(types.iter())
            .map(|(&n, &ty)| {
                let mut fi = numeric_field(n);
                fi.doc_values_type = ty;
                fi
            })
            .collect(),
    }
}

#[test]
fn write_dense_fields_rejects_empty_slice() {
    let id = [90u8; ID_LENGTH];
    let err = write_dense_fields(&[], 4, &id, "").unwrap_err();
    assert!(matches!(err, WriteError::EmptyFieldList));
}

#[test]
fn no_field_close_matches_actual_java_files_and_empty_metadata() {
    let id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let (meta, data, skip_index) = write_no_fields(&id, "s");
    assert_eq!(
        meta,
        include_bytes!("../../../../fixtures/testframework/data/cranky_doc_values_empty.dvm")
    );
    assert_eq!(
        data,
        include_bytes!("../../../../fixtures/testframework/data/cranky_doc_values_empty.dvd")
    );
    assert_eq!(
        skip_index,
        include_bytes!("../../../../fixtures/testframework/data/cranky_doc_values_empty.dvs")
    );
    let (version, fields) = parse_meta(&meta, &id, "s", &FieldInfos { fields: vec![] }).unwrap();
    assert_eq!(version, VERSION_WRITE_DEFAULT);
    assert!(fields.numeric.is_empty());
    assert!(fields.binary.is_empty());
    assert!(fields.sorted.is_empty());
    assert!(fields.sorted_numeric.is_empty());
    assert!(fields.sorted_set.is_empty());
    assert!(fields.skippers.is_empty());
    assert_eq!(check_data_header_footer(&data, &id, "s").unwrap(), version);
    assert_eq!(
        check_skip_index_header_footer(&skip_index, &id, "s").unwrap(),
        version
    );
}

#[test]
fn write_dense_fields_rejects_duplicate_field_numbers() {
    let id = [91u8; ID_LENGTH];
    let a = vec![1i64, 2, 3];
    let b = vec![4i64, 5, 6];
    let err = write_dense_fields(
        &[DenseField::Numeric(0, &a), DenseField::Numeric(0, &b)],
        3,
        &id,
        "",
    )
    .unwrap_err();
    assert!(matches!(err, WriteError::DuplicateFieldNumber(0)));
}

#[test]
fn write_dense_fields_rejects_a_field_that_is_not_dense_over_max_doc() {
    let id = [92u8; ID_LENGTH];
    let a = vec![1i64, 2, 3];
    let err = write_dense_fields(&[DenseField::Numeric(0, &a)], 4, &id, "").unwrap_err();
    assert!(matches!(
        err,
        WriteError::NotDense {
            values: 3,
            max_doc: 4
        }
    ));
}

#[test]
fn write_dense_fields_two_numeric_fields_round_trip_independently() {
    let id = [93u8; ID_LENGTH];
    let max_doc = 4i32;
    let values_a = vec![5i64, 250, 0, 100];
    let values_b = vec![-7i64, -7, 42, 1000];
    let (meta_bytes, data_bytes, skip_bytes) = write_dense_fields(
        &[
            DenseField::Numeric(0, &values_a),
            DenseField::Numeric(1, &values_b),
        ],
        max_doc,
        &id,
        "",
    )
    .unwrap();

    assert_eq!(
        check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id).unwrap(),
        VERSION_WRITE_DEFAULT
    );
    assert_eq!(
        check_data_header_footer(&data_bytes, &id, "").unwrap(),
        VERSION_WRITE_DEFAULT
    );

    let fis = field_infos_with(&[0, 1]);
    let (meta_version, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    assert_eq!(meta_version, VERSION_WRITE_DEFAULT);
    assert_eq!(meta.numeric.len(), 2);

    let entry_a = meta.numeric_entry(0).unwrap();
    for (doc, &want) in values_a.iter().enumerate() {
        assert_eq!(
            numeric_value(&data_bytes, entry_a, doc as i32).unwrap(),
            Some(want)
        );
    }
    let entry_b = meta.numeric_entry(1).unwrap();
    for (doc, &want) in values_b.iter().enumerate() {
        assert_eq!(
            numeric_value(&data_bytes, entry_b, doc as i32).unwrap(),
            Some(want)
        );
    }
}

/// `Lucene90DocValuesConsumer.writeValues`' `numDocsWithValue == 0`
/// branch: `meta[-2, 0]`, no `IndexedDISI` structure at all. This port
/// reaches it whenever a doc-values update generation *resets* every
/// document's value -- `update_doc_values` with a `None` value against a
/// term that matches the whole segment.
#[test]
fn a_sparse_numeric_field_with_no_values_writes_javas_empty_marker() {
    let id = [111u8; ID_LENGTH];
    let (meta_bytes, data_bytes, _) =
        write_single_sparse_numeric_field(0, &[], 5, &id, "").unwrap();
    let fis = field_infos_with(&[0]);
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let entry = meta.numeric_entry(0).unwrap();
    assert!(entry.is_empty_field(), "docsWithFieldOffset must be -2");
    assert!(!entry.is_dense());
    assert_eq!(entry.docs_with_field_length, 0);
    assert_eq!(entry.jump_table_entry_count, -1);
    assert_eq!(entry.dense_rank_power, 0xFF, "denseRankPower is Java's -1");
    assert_eq!(entry.num_values, 0);
    for doc in 0..5 {
        assert_eq!(numeric_value(&data_bytes, entry, doc).unwrap(), None);
    }
}

/// The BINARY twin. Java's `minLength`/`maxLength` trackers start at
/// `(Integer.MAX_VALUE, 0)` and stay there, which is what it writes -- and
/// what keeps `maxLength > minLength` false so no address array follows.
#[test]
fn a_sparse_binary_field_with_no_values_writes_javas_empty_marker() {
    let id = [112u8; ID_LENGTH];
    let (meta_bytes, data_bytes, _) = write_single_sparse_binary_field(0, &[], 4, &id, "").unwrap();
    let fis = FieldInfos {
        fields: vec![FieldInfo {
            doc_values_type: crate::field_infos::DocValuesType::Binary,
            ..numeric_field(0)
        }],
    };
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let entry = meta.binary_entry(0).unwrap();
    assert!(entry.is_empty_field());
    assert_eq!(entry.num_docs_with_field, 0);
    assert_eq!(entry.min_length, i32::MAX);
    assert_eq!(entry.max_length, 0);
    assert!(
        entry.addresses.is_none(),
        "maxLength > minLength is false, so no address array is written"
    );
    for doc in 0..4 {
        assert_eq!(binary_value(&data_bytes, entry, doc).unwrap(), None);
    }
}

#[test]
fn write_dense_fields_single_field_matches_write_single_dense_numeric_field() {
    let id = [94u8; ID_LENGTH];
    let values = vec![5i64, 250, 0, 100];
    let max_doc = values.len() as i32;

    let via_multi =
        write_dense_fields(&[DenseField::Numeric(0, &values)], max_doc, &id, "").unwrap();
    let via_single = write_single_dense_numeric_field(0, &values, max_doc, &id, "").unwrap();
    assert_eq!(via_multi, via_single);
}

#[test]
fn write_dense_fields_single_field_matches_write_single_dense_binary_field() {
    let id = [96u8; ID_LENGTH];
    let values: Vec<Vec<u8>> = vec![b"a".to_vec(), b"bb".to_vec(), b"ccc".to_vec()];
    let max_doc = values.len() as i32;

    let via_multi =
        write_dense_fields(&[DenseField::Binary(0, &values)], max_doc, &id, "").unwrap();
    let via_single = write_single_dense_binary_field(0, &values, max_doc, &id, "").unwrap();
    assert_eq!(via_multi, via_single);
}

#[test]
fn write_dense_fields_single_field_matches_write_single_dense_sorted_field() {
    let id = [97u8; ID_LENGTH];
    let values: Vec<Vec<u8>> = vec![b"banana".to_vec(), b"apple".to_vec(), b"cherry".to_vec()];
    let max_doc = values.len() as i32;

    let via_multi =
        write_dense_fields(&[DenseField::Sorted(0, &values)], max_doc, &id, "").unwrap();
    let via_single = write_single_dense_sorted_field(0, &values, max_doc, &id, "").unwrap();
    assert_eq!(via_multi, via_single);
}

#[test]
fn write_dense_fields_single_field_matches_write_single_dense_sorted_numeric_field() {
    let id = [98u8; ID_LENGTH];
    let values: Vec<Vec<i64>> = vec![vec![1, 2], vec![3], vec![4, 5, 6]];
    let max_doc = values.len() as i32;

    let via_multi =
        write_dense_fields(&[DenseField::SortedNumeric(0, &values)], max_doc, &id, "").unwrap();
    let via_single = write_single_dense_sorted_numeric_field(0, &values, &id, "").unwrap();
    assert_eq!(via_multi, via_single);
}

#[test]
fn write_dense_fields_single_field_matches_write_single_dense_sorted_set_field() {
    let id = [99u8; ID_LENGTH];
    let values: Vec<Vec<Vec<u8>>> = vec![
        vec![b"red".to_vec(), b"blue".to_vec()],
        vec![b"green".to_vec()],
        vec![b"red".to_vec(), b"green".to_vec()],
    ];
    let max_doc = values.len() as i32;

    let via_multi =
        write_dense_fields(&[DenseField::SortedSet(0, &values)], max_doc, &id, "").unwrap();
    let via_single = write_single_dense_sorted_set_field(0, &values, max_doc, &id, "").unwrap();
    assert_eq!(via_multi, via_single);
}

#[test]
fn write_dense_fields_numeric_and_sorted_fields_do_not_cross_contaminate() {
    let id = [95u8; ID_LENGTH];
    let max_doc = 3i32;
    let numeric_values = vec![10i64, 20, 30];
    let sorted_values: Vec<Vec<u8>> =
        vec![b"banana".to_vec(), b"apple".to_vec(), b"cherry".to_vec()];

    let (meta_bytes, data_bytes, _skip_bytes) = write_dense_fields(
        &[
            DenseField::Numeric(0, &numeric_values),
            DenseField::Sorted(1, &sorted_values),
        ],
        max_doc,
        &id,
        "",
    )
    .unwrap();

    let fis = mixed_field_infos(&[0, 1], &[DocValuesType::Numeric, DocValuesType::Sorted]);
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    assert_eq!(meta.numeric.len(), 1);
    assert_eq!(meta.sorted.len(), 1);

    let numeric_entry = meta.numeric_entry(0).unwrap();
    for (doc, &want) in numeric_values.iter().enumerate() {
        assert_eq!(
            numeric_value(&data_bytes, numeric_entry, doc as i32).unwrap(),
            Some(want)
        );
    }

    let sorted_entry = meta.sorted_entry(1).unwrap();
    let dict = terms_dict::decode_all_terms(&data_bytes, &sorted_entry.terms).unwrap();
    for (doc, want) in sorted_values.iter().enumerate() {
        let ord = sorted_ord(&data_bytes, sorted_entry, doc as i32)
            .unwrap()
            .unwrap();
        assert_eq!(&dict[ord as usize], want);
    }
}

#[test]
fn write_dense_fields_all_four_sparse_types_together_round_trip() {
    // `DocValuesConsumer.getMerged*DocValues` drops a
    // source reader that has no column for the field, so merging an
    // ordinary index that started writing a doc-values field partway
    // through its life produces a *sparse* column for every one of the
    // five types. Only NUMERIC could be expressed here before, so
    // `execute_merge` had to fail such a merge outright.
    let id = [121u8; ID_LENGTH];
    let max_doc = 5i32;
    let binary: Vec<(i32, Vec<u8>)> = vec![(0, b"aa".to_vec()), (3, b"cccc".to_vec())];
    let sorted: Vec<(i32, Vec<u8>)> = vec![(1, b"pear".to_vec()), (4, b"apple".to_vec())];
    let sorted_numeric: Vec<(i32, Vec<i64>)> = vec![(0, vec![7, 9]), (2, vec![4])];
    let sorted_set: Vec<(i32, Vec<Vec<u8>>)> = vec![
        (2, vec![b"x".to_vec(), b"y".to_vec()]),
        (4, vec![b"z".to_vec()]),
    ];

    let (meta_bytes, data_bytes, _) = write_dense_fields(
        &[
            DenseField::SparseBinary(0, &binary),
            DenseField::SparseSorted(1, &sorted),
            DenseField::SparseSortedNumeric(2, &sorted_numeric),
            DenseField::SparseSortedSet(3, &sorted_set),
        ],
        max_doc,
        &id,
        "",
    )
    .unwrap();

    let fis = mixed_field_infos(
        &[0, 1, 2, 3],
        &[
            DocValuesType::Binary,
            DocValuesType::Sorted,
            DocValuesType::SortedNumeric,
            DocValuesType::SortedSet,
        ],
    );
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();

    let binary_entry = meta.binary_entry(0).unwrap();
    for doc in 0..max_doc {
        let want = binary
            .iter()
            .find(|(d, _)| *d == doc)
            .map(|(_, v)| v.as_slice());
        assert_eq!(binary_value(&data_bytes, binary_entry, doc).unwrap(), want);
    }

    let sorted_entry = meta.sorted_entry(1).unwrap();
    let dict = terms_dict::decode_all_terms(&data_bytes, &sorted_entry.terms).unwrap();
    for doc in 0..max_doc {
        let got = sorted_ord(&data_bytes, sorted_entry, doc)
            .unwrap()
            .map(|ord| dict[ord as usize].clone());
        let want = sorted
            .iter()
            .find(|(d, _)| *d == doc)
            .map(|(_, v)| v.clone());
        assert_eq!(got, want, "sorted doc {doc}");
    }

    let sn_entry = meta.sorted_numeric_entry(2).unwrap();
    for doc in 0..max_doc {
        let want: Vec<i64> = sorted_numeric
            .iter()
            .find(|(d, _)| *d == doc)
            .map(|(_, v)| v.clone())
            .unwrap_or_default();
        assert_eq!(
            sorted_numeric_values(&data_bytes, sn_entry, doc).unwrap(),
            want,
            "sorted-numeric doc {doc}"
        );
    }

    let ss_entry = meta.sorted_set_entry(3).unwrap();
    let SortedSetKind::Multi { ords, terms } = &ss_entry.kind else {
        panic!("a sparse SORTED_SET always takes the multiValued shape");
    };
    let ss_dict = terms_dict::decode_all_terms(&data_bytes, terms).unwrap();
    for doc in 0..max_doc {
        let got: Vec<Vec<u8>> = sorted_numeric_values(&data_bytes, ords, doc)
            .unwrap()
            .into_iter()
            .map(|ord| ss_dict[ord as usize].clone())
            .collect();
        let mut want: Vec<Vec<u8>> = sorted_set
            .iter()
            .find(|(d, _)| *d == doc)
            .map(|(_, v)| v.clone())
            .unwrap_or_default();
        want.sort_unstable();
        want.dedup();
        assert_eq!(got, want, "sorted-set doc {doc}");
    }
}

#[test]
fn a_sparse_column_that_covers_every_document_writes_javas_dense_marker() {
    // `Lucene90DocValuesConsumer` writes the `-1` docs-with-field marker
    // whenever `numDocsWithField == maxDoc` -- `addBinaryField` does it
    // inline and the other four types get it from `writeValues`. A full
    // `IndexedDISI` region for a complete column is readable but is not a
    // byte sequence real Lucene ever produces.
    let id = [122u8; ID_LENGTH];
    let values: Vec<Vec<u8>> = vec![b"aa".to_vec(), b"bbb".to_vec()];
    let pairs: Vec<(i32, Vec<u8>)> = vec![(0, values[0].clone()), (1, values[1].clone())];

    let (sparse_meta, sparse_data, _) =
        write_single_sparse_binary_field(0, &pairs, 2, &id, "").unwrap();
    let (dense_meta, dense_data, _) =
        write_single_dense_binary_field(0, &values, 2, &id, "").unwrap();
    assert_eq!(sparse_meta, dense_meta);
    assert_eq!(sparse_data, dense_data);

    let sorted_pairs: Vec<(i32, Vec<u8>)> = vec![(0, b"b".to_vec()), (1, b"a".to_vec())];
    let sorted_values: Vec<Vec<u8>> = vec![b"b".to_vec(), b"a".to_vec()];
    let (sparse_meta, sparse_data, _) =
        write_single_sparse_sorted_field(0, &sorted_pairs, 2, &id, "").unwrap();
    let (dense_meta, dense_data, _) =
        write_single_dense_sorted_field(0, &sorted_values, 2, &id, "").unwrap();
    assert_eq!(sparse_meta, dense_meta);
    assert_eq!(sparse_data, dense_data);

    // SORTED_NUMERIC: identical bytes, since neither shape has a
    // `multiValued` flag to disagree about.
    let sn_pairs: Vec<(i32, Vec<i64>)> = vec![(0, vec![1, 2]), (1, vec![3])];
    let sn_values: Vec<Vec<i64>> = vec![vec![1, 2], vec![3]];
    let (sparse_meta, sparse_data, _) =
        write_single_sparse_sorted_numeric_field(0, &sn_pairs, 2, &id, "").unwrap();
    let (dense_meta, dense_data, _) =
        write_single_dense_sorted_numeric_field(0, &sn_values, &id, "").unwrap();
    assert_eq!(sparse_meta, dense_meta);
    assert_eq!(sparse_data, dense_data);

    // SORTED_SET is the documented exception: the sparse writer always
    // takes the `multiValued = 1` shape, where the dense one collapses to
    // a plain `SortedEntry` when every doc has exactly one ordinal. The
    // *docs-with-field* marker still has to agree, which is what this
    // asserts -- both entries must carry `-1`, not an `IndexedDISI`
    // region.
    let ss_pairs: Vec<(i32, Vec<Vec<u8>>)> = vec![
        (0, vec![b"x".to_vec(), b"y".to_vec()]),
        (1, vec![b"z".to_vec()]),
    ];
    let ss_values: Vec<Vec<Vec<u8>>> =
        vec![vec![b"x".to_vec(), b"y".to_vec()], vec![b"z".to_vec()]];
    let (sparse_meta, sparse_data, _) =
        write_single_sparse_sorted_set_field(0, &ss_pairs, 2, &id, "").unwrap();
    let (dense_meta, dense_data, _) =
        write_single_dense_sorted_set_field(0, &ss_values, 2, &id, "").unwrap();
    assert_eq!(sparse_meta, dense_meta);
    assert_eq!(sparse_data, dense_data);

    let fis = mixed_field_infos(&[0], &[DocValuesType::SortedSet]);
    let (_, meta) = parse_meta(&sparse_meta, &id, "", &fis).unwrap();
    let SortedSetKind::Multi { ords, .. } = &meta.sorted_set_entry(0).unwrap().kind else {
        panic!("multiValued shape expected");
    };
    assert_eq!(ords.numeric.docs_with_field_offset, DOCS_WITH_FIELD_DENSE);
}

#[test]
fn sparse_multi_valued_fields_with_no_values_write_javas_empty_marker_and_shape() {
    let id = [123u8; ID_LENGTH];
    let (meta_bytes, data_bytes, _) = write_dense_fields(
        &[
            DenseField::SparseSortedNumeric(0, &[]),
            DenseField::SparseSortedSet(1, &[]),
        ],
        4,
        &id,
        "",
    )
    .unwrap();
    let fis = mixed_field_infos(
        &[0, 1],
        &[DocValuesType::SortedNumeric, DocValuesType::SortedSet],
    );
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    let sn = meta.sorted_numeric_entry(0).unwrap();
    assert_eq!(sn.numeric.docs_with_field_offset, DOCS_WITH_FIELD_EMPTY);
    for doc in 0..4 {
        assert!(sorted_numeric_values(&data_bytes, sn, doc)
            .unwrap()
            .is_empty());
    }
    let ss = meta.sorted_set_entry(1).unwrap();
    // No document has two values, so Java's `isSingleValued` holds
    // vacuously and the SORTED shape is written.
    let SortedSetKind::Single(sorted) = &ss.kind else {
        panic!("single-valued shape expected");
    };
    assert_eq!(sorted.ords.docs_with_field_offset, DOCS_WITH_FIELD_EMPTY);
}

#[test]
fn sparse_variants_reject_a_duplicate_or_out_of_range_doc_id() {
    let id = [124u8; ID_LENGTH];
    let dup: Vec<(i32, Vec<u8>)> = vec![(1, b"a".to_vec()), (1, b"b".to_vec())];
    assert!(matches!(
        write_dense_fields(&[DenseField::SparseBinary(0, &dup)], 4, &id, ""),
        Err(WriteError::DocIdsNotAscending(1))
    ));
    let oob: Vec<(i32, Vec<i64>)> = vec![(9, vec![1])];
    assert!(matches!(
        write_dense_fields(&[DenseField::SparseSortedNumeric(0, &oob)], 4, &id, ""),
        Err(WriteError::DocIdOutOfRange(9, 4))
    ));
    let empty: Vec<(i32, Vec<Vec<u8>>)> = vec![(1, Vec::new())];
    assert!(matches!(
        write_dense_fields(&[DenseField::SparseSortedSet(0, &empty)], 4, &id, ""),
        Err(WriteError::EmptyMultiValuedDoc(1))
    ));
}

#[test]
fn write_dense_fields_all_five_types_together_round_trip() {
    let id = [96u8; ID_LENGTH];
    let max_doc = 3i32;
    let numeric_values = vec![10i64, 20, 30];
    let binary_values: Vec<Vec<u8>> = vec![b"aa".to_vec(), b"bb".to_vec(), b"cc".to_vec()];
    let sorted_values: Vec<Vec<u8>> =
        vec![b"banana".to_vec(), b"apple".to_vec(), b"cherry".to_vec()];
    let sorted_numeric_field_values: Vec<Vec<i64>> = vec![vec![1, 2], vec![3], vec![4, 5, 6]];
    let sorted_set_values: Vec<Vec<Vec<u8>>> = vec![
        vec![b"x".to_vec(), b"y".to_vec()],
        vec![b"y".to_vec()],
        vec![b"z".to_vec()],
    ];

    let (meta_bytes, data_bytes, skip_bytes) = write_dense_fields(
        &[
            DenseField::Numeric(0, &numeric_values),
            DenseField::Binary(1, &binary_values),
            DenseField::Sorted(2, &sorted_values),
            DenseField::SortedNumeric(3, &sorted_numeric_field_values),
            DenseField::SortedSet(4, &sorted_set_values),
        ],
        max_doc,
        &id,
        "",
    )
    .unwrap();

    assert_eq!(
        check_data_header_footer_generic(&skip_bytes, "Lucene90DocValuesSkipIndex", &id).unwrap(),
        VERSION_WRITE_DEFAULT
    );

    let fis = mixed_field_infos(
        &[0, 1, 2, 3, 4],
        &[
            DocValuesType::Numeric,
            DocValuesType::Binary,
            DocValuesType::Sorted,
            DocValuesType::SortedNumeric,
            DocValuesType::SortedSet,
        ],
    );
    let (_, meta) = parse_meta(&meta_bytes, &id, "", &fis).unwrap();
    assert_eq!(meta.numeric.len(), 1);
    assert_eq!(meta.binary.len(), 1);
    assert_eq!(meta.sorted.len(), 1);
    assert_eq!(meta.sorted_numeric.len(), 1);
    assert_eq!(meta.sorted_set.len(), 1);

    // Numeric.
    let numeric_entry = meta.numeric_entry(0).unwrap();
    for (doc, &want) in numeric_values.iter().enumerate() {
        assert_eq!(
            numeric_value(&data_bytes, numeric_entry, doc as i32).unwrap(),
            Some(want)
        );
    }

    // Binary.
    let binary_entry = meta.binary_entry(1).unwrap();
    for (doc, want) in binary_values.iter().enumerate() {
        assert_eq!(
            binary_value(&data_bytes, binary_entry, doc as i32).unwrap(),
            Some(want.as_slice())
        );
    }

    // Sorted.
    let sorted_entry = meta.sorted_entry(2).unwrap();
    let sorted_dict = terms_dict::decode_all_terms(&data_bytes, &sorted_entry.terms).unwrap();
    for (doc, want) in sorted_values.iter().enumerate() {
        let ord = sorted_ord(&data_bytes, sorted_entry, doc as i32)
            .unwrap()
            .unwrap();
        assert_eq!(&sorted_dict[ord as usize], want);
    }

    // Sorted numeric.
    let sorted_numeric_entry = meta.sorted_numeric_entry(3).unwrap();
    for (doc, want) in sorted_numeric_field_values.iter().enumerate() {
        let got = sorted_numeric_values(&data_bytes, sorted_numeric_entry, doc as i32).unwrap();
        assert_eq!(&got, want);
    }

    // Sorted set.
    let sorted_set_entry = meta.sorted_set_entry(4).unwrap();
    let SortedSetKind::Multi { ords, terms } = &sorted_set_entry.kind else {
        panic!("expected multi-valued SORTED_SET entry");
    };
    let set_dict = terms_dict::decode_all_terms(&data_bytes, terms).unwrap();
    for (doc, want) in sorted_set_values.iter().enumerate() {
        let doc_ords = sorted_numeric_values(&data_bytes, ords, doc as i32).unwrap();
        let mut got: Vec<Vec<u8>> = doc_ords
            .iter()
            .map(|&ord| set_dict[ord as usize].clone())
            .collect();
        got.sort_unstable();
        let mut want_sorted = want.clone();
        want_sorted.sort_unstable();
        want_sorted.dedup();
        assert_eq!(got, want_sorted);
    }
}

/// `TermsDict`'s random access over a dictionary of many LZ4 blocks and
/// several reverse-index intervals, written by [`write_terms_dict`] (the
/// writer `addSortedField` and `addSortedSetField` share): every ordinal
/// seeks to its term in any order, `lookupTerm` gives a term's ordinal
/// and a key between terms `-insertionPoint - 1`, `seekCeil` lands on the
/// ceiling or reports the end, and an ordinal outside the dictionary is
/// refused.
#[test]
fn a_terms_dictionary_seeks_by_ordinal_and_by_term() {
    use crate::terms_dict::{self, SeekStatus};
    let n = 3000i64;
    let terms: Vec<Vec<u8>> = (0..n)
        .map(|i| format!("t{:05}", i * 3).into_bytes())
        .collect();
    let (mut meta, mut data) = (Vec::new(), Vec::new());
    write_terms_dict(&mut meta, &mut data, &terms);
    let entry = terms_dict::read_term_dict_entry(&mut SliceInput::new(&meta)).unwrap();
    let mut dict = terms_dict::TermsDict::open(&data, &entry).unwrap();
    assert_eq!(dict.size(), n);
    let ords: Vec<i64> = (0..n)
        .step_by(7)
        .chain((0..n).rev().step_by(13))
        .chain([0, n - 1, 64, 63, 1024, 1023, 2047, 2048])
        .collect();
    for ord in ords {
        assert_eq!(
            dict.lookup_ord(ord).unwrap(),
            &terms[ord as usize][..],
            "ord {ord}"
        );
        assert_eq!(dict.ord(), ord);
    }
    for i in (0..n).step_by(5).chain([n - 1]) {
        let t = &terms[i as usize];
        assert_eq!(dict.lookup_term(t).unwrap(), i);
        assert_eq!(dict.raw_term(), &t[..]);
        // Just past `t`, before the next term.
        let between = format!("t{:05}", i * 3 + 1).into_bytes();
        let want = if i + 1 < n { -(i + 1) - 1 } else { -n - 1 };
        assert_eq!(dict.lookup_term(&between).unwrap(), want, "{i}");
    }
    assert_eq!(dict.lookup_term(b"a").unwrap(), -1);
    assert_eq!(dict.lookup_term(b"u").unwrap(), -n - 1);
    assert_eq!(dict.seek_ceil(b"t00004").unwrap(), SeekStatus::NotFound);
    assert_eq!(dict.raw_term(), b"t00006");
    assert_eq!(dict.seek_ceil(b"t00006").unwrap(), SeekStatus::Found);
    assert_eq!(dict.seek_ceil(b"zzz").unwrap(), SeekStatus::End);
    for bad in [-1, n, i64::MAX] {
        assert!(dict.lookup_ord(bad).is_err(), "{bad}");
    }
    let no_index = terms_dict::TermsDictEntry {
        mask: None,
        index: None,
        ..entry.clone()
    };
    assert!(terms_dict::TermsDict::open(&data, &no_index).is_err());
    // An empty dictionary has no term to land on.
    let (mut meta, mut data) = (Vec::new(), Vec::new());
    write_terms_dict::<Vec<u8>>(&mut meta, &mut data, &[]);
    let empty = terms_dict::read_term_dict_entry(&mut SliceInput::new(&meta)).unwrap();
    if let Ok(mut d) = terms_dict::TermsDict::open(&data, &empty) {
        assert_eq!(d.seek_ceil(b"a").unwrap(), SeekStatus::End);
        assert_eq!(d.lookup_term(b"a").unwrap(), -1);
    }
}

/// A masked terms dictionary shows the masked bytes for an ordinal and answers every seek and
/// ordinal-by-term lookup against the stored ones; the stored bytes are what
/// `raw_term_at` and the sequential cursor read; a mask that fails fails the lookup.
#[test]
fn a_masked_terms_dictionary_shows_masked_terms_and_seeks_by_stored_ones() {
    use crate::terms_dict::{self, SeekStatus};
    use crate::value_mask::{SharedMask, ValueMask};

    #[derive(Debug)]
    struct Tag;
    impl ValueMask for Tag {
        fn apply(&self, raw: &[u8]) -> lucene_store::Result<Vec<u8>> {
            let mut out = b"#".to_vec();
            out.extend_from_slice(raw);
            Ok(out)
        }
    }
    #[derive(Debug)]
    struct Broken;
    impl ValueMask for Broken {
        fn apply(&self, _: &[u8]) -> lucene_store::Result<Vec<u8>> {
            Err(crate::value_mask::masked_error("the mask"))
        }
    }

    let terms: Vec<Vec<u8>> = (0..200i64)
        .map(|i| format!("t{:05}", i * 3).into_bytes())
        .collect();
    let (mut meta, mut data) = (Vec::new(), Vec::new());
    write_terms_dict(&mut meta, &mut data, &terms);
    let entry = terms_dict::read_term_dict_entry(&mut SliceInput::new(&meta)).unwrap();
    let masked = terms_dict::TermsDictEntry {
        mask: Some(std::sync::Arc::new(Tag) as SharedMask),
        ..entry.clone()
    };
    let mut dict = terms_dict::TermsDict::open(&data, &masked).unwrap();
    for ord in [0i64, 1, 63, 64, 65, 199, 100] {
        let mut want = b"#".to_vec();
        want.extend_from_slice(&terms[ord as usize]);
        assert_eq!(dict.lookup_ord(ord).unwrap(), &want[..], "ord {ord}");
        assert_eq!(dict.raw_term_at(ord).unwrap(), &terms[ord as usize][..]);
    }
    // Seeks and ordinal lookups compare the stored terms: the masked ones are not in the
    // dictionary.
    assert_eq!(dict.lookup_term(&terms[77]).unwrap(), 77);
    assert!(dict.lookup_term(b"#t00231").unwrap() < 0);
    assert_eq!(dict.seek_ceil(b"t00004").unwrap(), SeekStatus::NotFound);
    assert_eq!(dict.raw_term(), b"t00006");
    // The sequential cursor, which builds ordinal maps, reads the stored terms.
    let mut cursor = terms_dict::TermsCursor::open(&data, &masked).unwrap();
    assert_eq!(cursor.next_term().unwrap(), Some(&terms[0][..]));
    // A mask that fails never falls back to the stored bytes.
    let broken = terms_dict::TermsDictEntry {
        mask: Some(std::sync::Arc::new(Broken) as SharedMask),
        ..entry
    };
    assert!(terms_dict::TermsDict::open(&data, &broken)
        .unwrap()
        .lookup_ord(3)
        .is_err());
}

/// A masked binary field shows masked values through `value_ref`, and every path that would
/// hand out the stored bytes refuses.
#[test]
fn a_masked_binary_field_shows_masked_values_and_refuses_the_stored_ones() {
    use crate::value_mask::{SharedMask, ValueMask};

    #[derive(Debug)]
    struct Tag;
    impl ValueMask for Tag {
        fn apply(&self, raw: &[u8]) -> lucene_store::Result<Vec<u8>> {
            let mut out = b"#".to_vec();
            out.extend_from_slice(raw);
            Ok(out)
        }
    }

    let id = [7u8; 16];
    let dense: Vec<Vec<u8>> = (0..100)
        .map(|d| format!("d{}", d % 7).into_bytes())
        .collect();
    let (dvm, dvd, _) = write_single_dense_binary_field(0, &dense, 100, &id, "").unwrap();
    let (_, meta) = parse_meta(&dvm, &id, "", &binary_field_infos()).unwrap();
    let mut entry = meta.binary_entry(0).unwrap().clone();
    entry.mask = Some(std::sync::Arc::new(Tag) as SharedMask);

    let mut r = BinaryReader::new(&dvd, &entry);
    for doc in [0, 5, 99, 3] {
        let mut want = b"#".to_vec();
        want.extend_from_slice(&dense[doc as usize]);
        assert_eq!(r.value_ref(doc).unwrap(), Some(&want[..]), "doc {doc}");
    }
    assert!(r.value(0).is_err(), "the zero-copy read refuses");
    assert!(
        binary_value(&dvd, &entry, 0).is_err(),
        "the free read refuses"
    );
    // Unmasked, the same entry reads as before.
    entry.mask = None;
    let mut r = BinaryReader::new(&dvd, &entry);
    assert_eq!(r.value_ref(5).unwrap(), Some(&dense[5][..]));
    assert_eq!(r.value(5).unwrap(), Some(&dense[5][..]));
}

/// The cursor readers (`NumericReader`, `SortedNumericReader`,
/// `BinaryReader`) answer what the one-shot decoders (`numeric_value`,
/// `sorted_numeric_values`, `binary_value`) answer, over fields this
/// module writes dense and sparse across several DISI blocks: every
/// document in order, out of order where the cursor allows it (one that
/// has to start over), by value index, a window at a time and through
/// `for_each_value`.
#[test]
fn cursor_readers_agree_with_the_one_shot_decoders() {
    let id = [9u8; ID_LENGTH];
    let max_doc = 200_000i32;
    // Sparse: every 3rd document (DISI's sparse and dense blocks both
    // show up: blocks of 65536 with ~21845 documents are dense).
    let sparse_docs: Vec<i32> = (0..max_doc).filter(|d| d % 3 == 0).collect();
    let val = |d: i32| i64::from(d % 1000) * 7 - 3000 + i64::from(d / 50_000);
    let order: Vec<i32> = (0..max_doc)
        .step_by(997)
        .chain((0..max_doc).rev().step_by(4999))
        .chain([0, 1, 3, 65535, 65536, 65537, max_doc - 1])
        .collect();

    // NUMERIC, dense and sparse.
    let dense: Vec<i64> = (0..max_doc).map(val).collect();
    let sparse: Vec<(i32, i64)> = sparse_docs.iter().map(|&d| (d, val(d))).collect();
    for (dvm, dvd) in [
        {
            let (m, d, _) = write_single_dense_numeric_field(0, &dense, max_doc, &id, "").unwrap();
            (m, d)
        },
        {
            let (m, d, _) =
                write_single_sparse_numeric_field(0, &sparse, max_doc, &id, "").unwrap();
            (m, d)
        },
    ] {
        let (_, meta) = parse_meta(&dvm, &id, "", &field_infos_with(&[0])).unwrap();
        let entry = meta.numeric_entry(0).unwrap();
        let mut r = NumericReader::new(&dvd, entry);
        assert!(std::ptr::eq(r.entry(), entry));
        for &doc in &order {
            let want = numeric_value(&dvd, entry, doc).unwrap();
            assert_eq!(r.value(doc).unwrap(), want, "doc {doc}");
            // The fast path answers or defers, never contradicts.
            if let Some(v) = r.dense_value(doc).filter(|_| r.has_dense_fast()) {
                assert_eq!(Some(v), want, "doc {doc}");
            }
        }
        let present: Vec<i64> = (0..max_doc)
            .filter_map(|d| numeric_value(&dvd, entry, d).unwrap())
            .collect();
        for i in (0..present.len()).step_by(1013) {
            assert_eq!(r.value_at_index(i as i64).unwrap(), present[i]);
        }
        assert!(r.value_at_index(present.len() as i64).is_err());
        assert!(r.value_at_index(-1).is_err());
        let mut seen = Vec::new();
        r.for_each_value(65_000, 66_100, |d, v| seen.push((d, v)))
            .unwrap();
        let want: Vec<(i32, i64)> = (65_000..66_100)
            .filter_map(|d| numeric_value(&dvd, entry, d).unwrap().map(|v| (d, v)))
            .collect();
        assert_eq!(seen, want);
        let (mut values, mut bits) = (vec![0i64; 640], vec![0u64; 10]);
        r.fill_window(65_500, &mut values, &mut bits).unwrap();
        for (i, &v) in values.iter().enumerate() {
            let doc = 65_500 + i as i32;
            let has = bits[i >> 6] & (1 << (i & 63)) != 0;
            let want = numeric_value(&dvd, entry, doc).unwrap();
            assert_eq!(has, want.is_some(), "doc {doc}");
            if has {
                assert_eq!(Some(v), want, "doc {doc}");
            }
        }
    }

    // SORTED_NUMERIC, sparse, one to three values a document.
    let multi: Vec<(i32, Vec<i64>)> = sparse_docs
        .iter()
        .map(|&d| (d, (0..1 + d % 3).map(|k| val(d) + i64::from(k)).collect()))
        .collect();
    let (dvm, dvd, _) =
        write_single_sparse_sorted_numeric_field(0, &multi, max_doc, &id, "").unwrap();
    let (_, meta) = parse_meta(&dvm, &id, "", &sorted_numeric_field_infos()).unwrap();
    let entry = meta.sorted_numeric_entry(0).unwrap();
    let mut r = SortedNumericReader::new(&dvd, entry);
    assert!(std::ptr::eq(r.entry(), entry));
    // Documents in non-decreasing order, the cursor's contract.
    let mut ascending = order.clone();
    ascending.sort_unstable();
    let mut got = Vec::new();
    for &doc in &ascending {
        r.values(doc, &mut got).unwrap();
        assert_eq!(
            got,
            sorted_numeric_values(&dvd, entry, doc).unwrap(),
            "doc {doc}"
        );
    }
    // Out of order (a reordering merge's access), every backward step
    // rewinding the cursor.
    let mut r = SortedNumericReader::new(&dvd, entry);
    for &doc in order.iter().rev() {
        r.values(doc, &mut got).unwrap();
        assert_eq!(
            got,
            sorted_numeric_values(&dvd, entry, doc).unwrap(),
            "doc {doc}"
        );
    }
    let mut r = SortedNumericReader::new(&dvd, entry);
    let (mut offsets, mut values) = (Vec::new(), Vec::new());
    // A window across two DISI blocks is left to `values`; one inside a
    // block, when answered (values in one width), is each document's.
    assert!(!r
        .fill_window(65_530, 20, &mut offsets, &mut values)
        .unwrap());
    let mut answered = 0;
    for start in [70_016, 140_032] {
        let mut r = SortedNumericReader::new(&dvd, entry);
        if r.fill_window(start, 320, &mut offsets, &mut values)
            .unwrap()
        {
            answered += 1;
            for i in 0..320 {
                let doc = start + i as i32;
                let (a, b) = (offsets[i] as usize, offsets[i + 1] as usize);
                assert_eq!(
                    values[a..b],
                    sorted_numeric_values(&dvd, entry, doc).unwrap()[..],
                    "doc {doc}"
                );
            }
        }
    }

    // A dense field's windows always come in one piece.
    let dense_multi: Vec<Vec<i64>> = (0..5000)
        .map(|d| (0..1 + d % 3).map(|k| i64::from(d % 100 + k)).collect())
        .collect();
    let (ddvm, ddvd, _) =
        write_single_dense_sorted_numeric_field(0, &dense_multi, &id, "").unwrap();
    let (_, dmeta) = parse_meta(&ddvm, &id, "", &sorted_numeric_field_infos()).unwrap();
    let dentry = dmeta.sorted_numeric_entry(0).unwrap();
    for start in [0, 1000, 4900] {
        let mut r = SortedNumericReader::new(&ddvd, dentry);
        if r.fill_window(start, 100, &mut offsets, &mut values)
            .unwrap()
        {
            answered += 1;
            for i in 0..100 {
                let doc = start + i as i32;
                let (a, b) = (offsets[i] as usize, offsets[i + 1] as usize);
                assert_eq!(values[a..b], dense_multi[doc as usize][..], "doc {doc}");
            }
        }
    }
    assert!(answered > 0, "a window answered");
    // The dense field read document by document and streamed.
    let mut r = SortedNumericReader::new(&ddvd, dentry);
    for doc in (0..5000).step_by(7) {
        r.values(doc, &mut got).unwrap();
        assert_eq!(got, dense_multi[doc as usize], "doc {doc}");
    }
    assert!(r.values(5000, &mut got).is_err() || got.is_empty());
    // Consecutive documents (each start the previous end, cached), then
    // steps back and over (looked up afresh).
    let mut r = SortedNumericReader::new(&ddvd, dentry);
    for doc in (10..40).chain([12, 13, 5, 4999, 0]) {
        r.values(doc, &mut got).unwrap();
        assert_eq!(got, dense_multi[doc as usize], "doc {doc}");
    }
    for (lo, hi) in [(0, 5000), (123, 4567)] {
        let mut seen = Vec::new();
        SortedNumericReader::new(&ddvd, dentry)
            .for_each_doc(lo, hi, |d, v| seen.push((d, v.to_vec())))
            .unwrap();
        let want: Vec<(i32, Vec<i64>)> = (lo..hi)
            .map(|d| (d, dense_multi[d as usize].clone()))
            .collect();
        assert_eq!(seen, want);
        let mut some = Vec::new();
        SortedNumericReader::new(&ddvd, dentry)
            .for_each_accepted(lo, hi, |d| d % 5 == 0, |d, v| some.push((d, v.to_vec())))
            .unwrap();
        let want_some: Vec<(i32, Vec<i64>)> =
            want.into_iter().filter(|(d, _)| d % 5 == 0).collect();
        assert_eq!(some, want_some);
    }
    // `for_each_doc` / `for_each_accepted`: every document of a range
    // with values, or only the accepted ones, with its values.
    for (lo, hi) in [(0, 5_000), (64_000, 67_000), (max_doc - 500, max_doc)] {
        let want: Vec<(i32, Vec<i64>)> = (lo..hi)
            .map(|d| (d, sorted_numeric_values(&dvd, entry, d).unwrap()))
            .filter(|(_, v)| !v.is_empty())
            .collect();
        let mut seen = Vec::new();
        SortedNumericReader::new(&dvd, entry)
            .for_each_doc(lo, hi, |d, v| seen.push((d, v.to_vec())))
            .unwrap();
        assert_eq!(seen, want, "[{lo}, {hi})");
        let mut odd = Vec::new();
        SortedNumericReader::new(&dvd, entry)
            .for_each_accepted(lo, hi, |d| d % 2 == 1, |d, v| odd.push((d, v.to_vec())))
            .unwrap();
        let want_odd: Vec<(i32, Vec<i64>)> =
            want.iter().filter(|(d, _)| d % 2 == 1).cloned().collect();
        assert_eq!(odd, want_odd, "[{lo}, {hi}) odd");
    }

    // A sparse field of small values, its windows read off its DISI
    // blocks a window at a time.
    let small: Vec<(i32, Vec<i64>)> = sparse_docs
        .iter()
        .take(30_000)
        .map(|&d| (d, (0..1 + d % 2).map(|k| i64::from(d % 13 + k)).collect()))
        .collect();
    let (sdvm, sdvd, _) =
        write_single_sparse_sorted_numeric_field(0, &small, max_doc, &id, "").unwrap();
    let (_, smeta) = parse_meta(&sdvm, &id, "", &sorted_numeric_field_infos()).unwrap();
    let sentry = smeta.sorted_numeric_entry(0).unwrap();
    let mut windows = 0;
    for start in [0, 640, 64_000, 66_048] {
        let mut r = SortedNumericReader::new(&sdvd, sentry);
        if r.fill_window(start, 640, &mut offsets, &mut values)
            .unwrap()
        {
            windows += 1;
            for i in 0..640 {
                let doc = start + i as i32;
                let (a, b) = (offsets[i] as usize, offsets[i + 1] as usize);
                assert_eq!(
                    values[a..b],
                    sorted_numeric_values(&sdvd, sentry, doc).unwrap()[..],
                    "doc {doc}"
                );
            }
        }
    }
    assert!(windows > 0);
    // Streamed whole from the first document: its DISI's documents
    // straight off the words.
    for end in [max_doc, 70_000] {
        let mut seen = Vec::new();
        SortedNumericReader::new(&sdvd, sentry)
            .for_each_accepted(0, end, |d| d % 4 != 1, |d, v| seen.push((d, v.to_vec())))
            .unwrap();
        let want: Vec<(i32, Vec<i64>)> = small
            .iter()
            .filter(|(d, _)| *d < end && d % 4 != 1)
            .cloned()
            .collect();
        assert_eq!(seen, want, "end {end}");
    }

    // BINARY, dense: read forwards and backwards.
    let dense_bins: Vec<Vec<u8>> = (0..3000)
        .map(|d| format!("d{}", d % 91).into_bytes())
        .collect();
    let (bdvm, bdvd, _) = write_single_dense_binary_field(0, &dense_bins, 3000, &id, "").unwrap();
    let (_, bmeta) = parse_meta(&bdvm, &id, "", &binary_field_infos()).unwrap();
    let bentry = bmeta.binary_entry(0).unwrap();
    let mut r = BinaryReader::new(&bdvd, bentry);
    for doc in (0..3000).step_by(11).chain((0..3000).rev().step_by(17)) {
        assert_eq!(r.value(doc).unwrap(), Some(&dense_bins[doc as usize][..]));
    }

    // BINARY, dense and of one length: addressed by `doc * length`.
    let fixed: Vec<Vec<u8>> = (0..3000)
        .map(|d| format!("f{}", d % 10).into_bytes())
        .collect();
    let (fdvm, fdvd, _) = write_single_dense_binary_field(0, &fixed, 3000, &id, "").unwrap();
    let (_, fmeta) = parse_meta(&fdvm, &id, "", &binary_field_infos()).unwrap();
    let fentry = fmeta.binary_entry(0).unwrap();
    let mut r = BinaryReader::new(&fdvd, fentry);
    for doc in [0, 1, 2999, 1234] {
        assert_eq!(r.value(doc).unwrap(), Some(&fixed[doc as usize][..]));
    }

    // BINARY, sparse.
    let bins: Vec<(i32, Vec<u8>)> = sparse_docs
        .iter()
        .map(|&d| (d, format!("v{}", d % 777).into_bytes()))
        .collect();
    let (dvm, dvd, _) = write_single_sparse_binary_field(0, &bins, max_doc, &id, "").unwrap();
    let (_, meta) = parse_meta(&dvm, &id, "", &binary_field_infos()).unwrap();
    let entry = meta.binary_entry(0).unwrap();
    let mut r = BinaryReader::new(&dvd, entry);
    for &doc in &order {
        assert_eq!(
            r.value(doc).unwrap(),
            binary_value(&dvd, entry, doc).unwrap(),
            "doc {doc}"
        );
    }
}
