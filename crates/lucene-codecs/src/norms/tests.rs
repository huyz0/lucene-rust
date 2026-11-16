#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

use super::*;

/// `Lucene90NormsProducer.readFields`' `FieldInfos` half, which
/// [`validate_fields`] carries in this port.
mod validate_fields_tests {
    use super::*;
    use crate::field_infos::{
        DocValuesSkipIndexType, DocValuesType, FieldInfo, VectorEncoding, VectorSimilarityFunction,
    };

    fn field(number: i32, name: &str, index_options: IndexOptions, omit_norms: bool) -> FieldInfo {
        FieldInfo {
            name: name.to_string(),
            number,
            store_term_vectors: false,
            omit_norms,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: Vec::new(),
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        }
    }

    fn norms_for(field_numbers: &[i32]) -> Norms {
        Norms {
            entries: field_numbers
                .iter()
                .map(|&field_number| NormsEntry {
                    field_number,
                    docs_with_field_offset: DOCS_WITH_FIELD_DENSE,
                    docs_with_field_length: 0,
                    jump_table_entry_count: -1,
                    dense_rank_power: 0xFF,
                    num_docs_with_field: 1,
                    bytes_per_norm: 1,
                    norms_offset: 0,
                })
                .collect(),
        }
    }

    #[test]
    fn entries_naming_real_norms_fields_are_accepted() {
        let infos = FieldInfos {
            fields: vec![
                field(0, "body", IndexOptions::DocsAndFreqs, false),
                field(3, "title", IndexOptions::Docs, false),
            ],
        };
        assert!(validate_fields(&norms_for(&[0, 3]), &infos).is_ok());
    }

    #[test]
    fn an_entry_for_a_field_the_fnm_does_not_have_is_rejected() {
        let infos = FieldInfos {
            fields: vec![field(0, "body", IndexOptions::DocsAndFreqs, false)],
        };
        assert!(matches!(
            validate_fields(&norms_for(&[0, 7]), &infos),
            Err(Error::UnknownFieldNumber(7))
        ));
    }

    #[test]
    fn an_entry_for_a_field_with_no_norms_is_rejected() {
        // `FieldInfo.hasNorms()` is `indexOptions != NONE && !omitNorms`;
        // both halves must reject.
        let omitted = FieldInfos {
            fields: vec![field(0, "body", IndexOptions::DocsAndFreqs, true)],
        };
        assert!(matches!(
            validate_fields(&norms_for(&[0]), &omitted),
            Err(Error::FieldHasNoNorms(name)) if name == "body"
        ));

        let unindexed = FieldInfos {
            fields: vec![field(0, "body", IndexOptions::None, false)],
        };
        assert!(matches!(
            validate_fields(&norms_for(&[0]), &unindexed),
            Err(Error::FieldHasNoNorms(name)) if name == "body"
        ));
    }
}

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
            dense_rank_power: indexed_disi::NO_RANK,
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

/// A sparse `.nvm` entry whose `docsWithFieldOffset + docsWithFieldLength`
/// overflows `i64`.
///
/// Both halves are read straight off `.nvm` with no relationship between
/// them on the wire, so a corrupt file can pick any pair. A plain `+`
/// feeding a slice range would, in a debug build, be
/// `attempt to add with overflow`, i.e. a panic where the caller (and,
/// through the FFI, the JVM) expected a decode error. Java is not exposed
/// to it because `IndexInput.slice(offset, length)` takes the two
/// separately and range-checks each.
#[test]
fn a_sparse_entry_whose_offset_plus_length_overflows_is_rejected() {
    let mut e = EntryBuilder::dense(0, 1, 3, 0);
    e.docs_with_field_offset = i64::MAX;
    e.docs_with_field_length = 1;
    let entry = to_entry(&e);
    assert!(!entry.is_dense() && !entry.is_empty_field());
    let got = norm_value(&[0, 0, 0], &entry, 0);
    assert!(got.is_err(), "{got:?}");

    // A negative offset that is neither of the two sentinels
    // (`DOCS_WITH_FIELD_DENSE` = -1, `DOCS_WITH_FIELD_EMPTY` = -2) takes
    // the same route rather than sign-extending into a ~2^64 slice start.
    let mut e = EntryBuilder::dense(0, 1, 3, 0);
    e.docs_with_field_offset = -3;
    e.docs_with_field_length = 4;
    let entry = to_entry(&e);
    let got = norm_value(&[0, 0, 0], &entry, 0);
    assert!(got.is_err(), "{got:?}");
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

/// A `.nvm` entry's `normsOffset` is an unconstrained `i64` off disk.
/// `normsOffset + ordinal * bytesPerNorm` overflowed before it was
/// bounded: a panic in a debug build (dead JVM through the FFI) and, in a
/// release build, a wrap to a small in-range offset that reads a
/// *plausible* norm out of the wrong place in the `.nvd`.
#[test]
fn corrupt_norms_offset_is_a_decode_error_not_an_overflow() {
    let id = [1u8; ID_LENGTH];
    let e = EntryBuilder::dense(0, 1, 5, i64::MAX);
    let buf = build_nvm(&id, &[e]);
    let (_, norms) = parse_meta(&buf, &id, "").unwrap();
    let entry = norms.entry(0).unwrap();
    // doc 0 needs no scaling and merely lands past EOF; doc 1 onwards
    // overflows the `i64` addition itself.
    for doc in 1..5 {
        assert!(
            matches!(
                norm_value(&[0u8; 8], entry, doc),
                Err(Error::Store(lucene_store::Error::Corrupted(_)))
            ),
            "doc {doc}"
        );
    }
}

/// The same shape from the other side: a negative `normsOffset` must not
/// become a huge `usize` through an `as` cast and merely look like EOF.
#[test]
fn negative_norms_offset_is_a_decode_error() {
    let id = [1u8; ID_LENGTH];
    let e = EntryBuilder::dense(0, 2, 5, -8);
    let buf = build_nvm(&id, &[e]);
    let (_, norms) = parse_meta(&buf, &id, "").unwrap();
    let entry = norms.entry(0).unwrap();
    assert!(matches!(
        norm_value(&[0u8; 16], entry, 0),
        Err(Error::Store(lucene_store::Error::Corrupted(_)))
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

/// `NormsReader` answers exactly what `norm_value` answers, doc for doc,
/// for every shape the writer produces -- one-byte dense (the fast path),
/// wider and constant dense (the fallback), sparse (the cursor, walked
/// forwards and then backwards so it has to rewind), and empty -- and for
/// docs outside the field (negative, past `max_doc`).
#[test]
fn norms_reader_agrees_with_norm_value_on_every_shape() {
    let id = [8u8; ID_LENGTH];
    let max_doc = 300i32;
    let dense_1: Vec<i64> = (0..max_doc as i64).map(|d| d % 100 - 50).collect();
    let dense_2: Vec<i64> = (0..max_doc as i64).map(|d| d * 300 - 40_000).collect();
    let constant = vec![9i64; max_doc as usize];
    let sparse: Vec<(i32, i64)> = (0..max_doc)
        .step_by(7)
        .map(|d| (d, d as i64 % 13))
        .collect();
    let mut files = Vec::new();
    for values in [&dense_1, &dense_2, &constant] {
        files.push(write_single_dense_field(0, values, max_doc, &id, "").unwrap());
    }
    files.push(write_single_sparse_field(0, &sparse, max_doc, &id, "").unwrap());
    files.push(write_single_sparse_field(0, &[], max_doc, &id, "").unwrap());
    for (meta_bytes, data) in &files {
        let (_, norms) = parse_meta(meta_bytes, &id, "").unwrap();
        let entry = norms.entry(0).unwrap();
        let docs: Vec<i32> = (0..max_doc).chain((0..max_doc).rev()).collect();
        let mut reader = NormsReader::new(data, entry);
        for doc in docs {
            assert_eq!(
                reader.value(doc).unwrap(),
                norm_value(data, entry, doc).unwrap(),
                "doc {doc}"
            );
        }
        for doc in [-1, max_doc, max_doc + 5] {
            assert_eq!(
                reader.value(doc).is_ok(),
                norm_value(data, entry, doc).is_ok(),
                "doc {doc}"
            );
        }
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

/// `Lucene90NormsConsumer.numBytesPerValue` picks the narrowest of
/// 0/1/2/4/8 bytes that fits the value range. Every width must round-trip
/// through this module's own reader, including the negative values a
/// custom `Similarity` can produce (the reader sign-extends).
#[test]
fn write_single_dense_field_uses_every_per_value_width() {
    let id = [10u8; ID_LENGTH];
    let cases: [(&[i64], u8); 5] = [
        (&[3, 3, 3], 0),
        (&[-128, 0, 127], 1),
        (&[-129, 0, 300], 2),
        (&[i16::MIN as i64 - 1, 0, 70_000], 4),
        (&[i32::MIN as i64 - 1, 0, 5_000_000_000], 8),
    ];
    for (values, want_width) in cases {
        let (meta_bytes, data_bytes) =
            write_single_dense_field(1, values, values.len() as i32, &id, "").unwrap();
        let (_, norms) = parse_meta(&meta_bytes, &id, "").unwrap();
        let entry = norms.entry(1).unwrap();
        assert_eq!(entry.bytes_per_norm, want_width, "values {values:?}");
        assert!(entry.is_dense());
        check_data_header_footer(&data_bytes, &id, "").unwrap();
        for (doc, &want) in values.iter().enumerate() {
            assert_eq!(
                norm_value(&data_bytes, entry, doc as i32).unwrap(),
                Some(want),
                "values {values:?} doc {doc}"
            );
        }
    }
}

/// The sparse branch (`numDocsWithValue != maxDoc`): only the listed docs
/// get a norm, the value array is indexed by rank rather than doc id, and
/// every other doc reads back as `None`.
#[test]
fn write_single_sparse_field_round_trips_and_skips_absent_docs() {
    let id = [11u8; ID_LENGTH];
    let max_doc = 10;
    // Deliberately unsorted input, first and last doc missing.
    let doc_values = [(7i32, 400i64), (1, 100), (4, -300), (6, 350)];
    let (meta_bytes, data_bytes) =
        write_single_sparse_field(2, &doc_values, max_doc, &id, "").unwrap();
    let (_, norms) = parse_meta(&meta_bytes, &id, "").unwrap();
    let entry = norms.entry(2).unwrap();
    assert!(!entry.is_dense() && !entry.is_empty_field());
    assert_eq!(entry.num_docs_with_field, 4);
    // -300..=400 needs 2 bytes per value.
    assert_eq!(entry.bytes_per_norm, 2);
    check_data_header_footer(&data_bytes, &id, "").unwrap();

    let mut want = std::collections::HashMap::new();
    for (doc, v) in doc_values {
        want.insert(doc, v);
    }
    for doc in 0..max_doc {
        assert_eq!(
            norm_value(&data_bytes, entry, doc).unwrap(),
            want.get(&doc).copied(),
            "doc {doc}"
        );
    }
}

/// `addNormsField` branches on the *count*, not on how the caller phrased
/// it: a sparse list covering every doc is written as dense (`-1`), and an
/// empty one as "no document has this field" (`-2`).
#[test]
fn write_single_sparse_field_degenerates_to_the_dense_and_empty_markers() {
    let id = [12u8; ID_LENGTH];
    let all: Vec<(i32, i64)> = (0..4).map(|d| (d, d as i64 + 1)).collect();
    let (meta_bytes, data_bytes) = write_single_sparse_field(0, &all, 4, &id, "").unwrap();
    let (_, norms) = parse_meta(&meta_bytes, &id, "").unwrap();
    let entry = norms.entry(0).unwrap();
    assert!(entry.is_dense());
    for (doc, v) in &all {
        assert_eq!(norm_value(&data_bytes, entry, *doc).unwrap(), Some(*v));
    }

    let (meta_bytes, data_bytes) = write_single_sparse_field(0, &[], 4, &id, "").unwrap();
    let (_, norms) = parse_meta(&meta_bytes, &id, "").unwrap();
    let entry = norms.entry(0).unwrap();
    assert!(entry.is_empty_field());
    assert_eq!(entry.num_docs_with_field, 0);
    for doc in 0..4 {
        assert_eq!(norm_value(&data_bytes, entry, doc).unwrap(), None);
    }
}

#[test]
fn write_single_sparse_field_rejects_duplicate_and_out_of_range_doc_ids() {
    let id = [13u8; ID_LENGTH];
    assert!(matches!(
        write_single_sparse_field(0, &[(1, 1), (1, 2)], 4, &id, ""),
        Err(WriteError::DocIdsNotAscending(1))
    ));
    assert!(matches!(
        write_single_sparse_field(0, &[(4, 1)], 4, &id, ""),
        Err(WriteError::DocIdOutOfRange(4, 4))
    ));
    assert!(matches!(
        write_single_sparse_field(0, &[(-1, 1)], 4, &id, ""),
        Err(WriteError::DocIdOutOfRange(-1, 4))
    ));
}

/// `Lucene90NormsConsumer` puts every norms field of a segment into one
/// `.nvm`/`.nvd` pair; entries must not cross-contaminate, and each field
/// picks its own per-value width.
#[test]
fn write_fields_interleaves_several_fields_in_one_pair() {
    let id = [14u8; ID_LENGTH];
    let max_doc = 6;
    let dense: Vec<i64> = (0..max_doc as i64).map(|d| d * 1000).collect();
    let sparse = [(0i32, 5i64), (5, 9)];
    let constant = vec![42i64; max_doc as usize];
    let (meta_bytes, data_bytes) = write_fields(
        &[
            NormsField::Dense(1, &dense),
            NormsField::Sparse(2, &sparse),
            NormsField::Dense(3, &constant),
        ],
        max_doc,
        &id,
        "",
    )
    .unwrap();
    let (_, norms) = parse_meta(&meta_bytes, &id, "").unwrap();
    assert_eq!(norms.entries.len(), 3);
    check_data_header_footer(&data_bytes, &id, "").unwrap();

    let f1 = norms.entry(1).unwrap();
    assert_eq!(f1.bytes_per_norm, 2);
    for (doc, &want) in dense.iter().enumerate() {
        assert_eq!(norm_value(&data_bytes, f1, doc as i32).unwrap(), Some(want));
    }

    let f2 = norms.entry(2).unwrap();
    assert_eq!(f2.num_docs_with_field, 2);
    assert_eq!(norm_value(&data_bytes, f2, 0).unwrap(), Some(5));
    assert_eq!(norm_value(&data_bytes, f2, 3).unwrap(), None);
    assert_eq!(norm_value(&data_bytes, f2, 5).unwrap(), Some(9));

    let f3 = norms.entry(3).unwrap();
    assert_eq!(f3.bytes_per_norm, 0);
    for doc in 0..max_doc {
        assert_eq!(norm_value(&data_bytes, f3, doc).unwrap(), Some(42));
    }
}

#[test]
fn write_fields_rejects_an_empty_or_duplicated_field_list() {
    let id = [15u8; ID_LENGTH];
    assert!(matches!(
        write_fields(&[], 1, &id, ""),
        Err(WriteError::EmptyFieldList)
    ));
    assert!(matches!(
        write_fields(
            &[NormsField::Dense(0, &[1]), NormsField::Dense(0, &[2])],
            1,
            &id,
            ""
        ),
        Err(WriteError::DuplicateFieldNumber(0))
    ));
}
