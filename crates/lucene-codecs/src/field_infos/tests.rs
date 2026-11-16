#![cfg(test)]
use super::*;

/// Test-only `.fnm` byte builder, independent of the Java fixtures under
/// `tests/field_infos_fixtures.rs` (which exercise a real IndexWriter's
/// output): this covers the parser's own error/consistency handling with
/// deliberately-invalid field combinations no real Lucene writer would
/// ever produce.
struct FieldBuilder {
    name: String,
    number: i32,
    bits: u8,
    index_options: u8,
    doc_values_type: u8,
    doc_values_skip_index_type: Option<u8>, // None => omit (format < 2)
    doc_values_gen: i64,
    point_dimension_count: i32,
    point_index_dimension_count: i32,
    point_num_bytes: i32,
    vector_dimension: i32,
    vector_encoding: u8,
    vector_similarity_function: u8,
}

impl FieldBuilder {
    fn valid(name: &str, number: i32) -> Self {
        Self {
            name: name.to_string(),
            number,
            bits: 0,
            index_options: 1, // Docs
            doc_values_type: 0,
            doc_values_skip_index_type: Some(0),
            doc_values_gen: -1,
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: 0,
            vector_similarity_function: 0,
        }
    }

    fn build(&self, out: &mut Vec<u8>) {
        write_string(out, &self.name);
        write_vint(out, self.number);
        out.push(self.bits);
        out.push(self.index_options);
        out.push(self.doc_values_type);
        if let Some(skip) = self.doc_values_skip_index_type {
            out.push(skip);
        }
        out.extend_from_slice(&self.doc_values_gen.to_le_bytes());
        write_vint(out, 0); // attributes: empty map
        write_vint(out, self.point_dimension_count);
        if self.point_dimension_count != 0 {
            write_vint(out, self.point_index_dimension_count);
            write_vint(out, self.point_num_bytes);
        }
        write_vint(out, self.vector_dimension);
        out.push(self.vector_encoding);
        out.push(self.vector_similarity_function);
    }
}

struct FnmBuilder {
    id: [u8; ID_LENGTH],
    suffix: String,
    format_version: i32,
    fields: Vec<FieldBuilder>,
}

impl FnmBuilder {
    fn valid() -> Self {
        Self {
            id: [4u8; ID_LENGTH],
            suffix: String::new(),
            format_version: FORMAT_CURRENT,
            fields: vec![],
        }
    }

    fn build(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut out, CODEC_NAME);
        out.extend_from_slice(&(self.format_version as u32).to_be_bytes());
        out.extend_from_slice(&self.id);
        out.push(self.suffix.len() as u8);
        out.extend_from_slice(self.suffix.as_bytes());

        write_vint(&mut out, self.fields.len() as i32);
        for f in &self.fields {
            f.build(&mut out);
        }

        out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&out) as u64;
        out.extend_from_slice(&checksum.to_be_bytes());
        out
    }
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

#[test]
fn empty_field_infos_parses() {
    let b = FnmBuilder::valid();
    let fis = parse(&b.build(), &b.id, &b.suffix).unwrap();
    assert_eq!(fis.fields.len(), 0);
}

#[test]
fn single_plain_field_parses() {
    let mut b = FnmBuilder::valid();
    b.fields.push(FieldBuilder::valid("id", 0));
    let fis = parse(&b.build(), &b.id, &b.suffix).unwrap();
    assert_eq!(fis.fields.len(), 1);
    assert_eq!(fis.fields[0].name, "id");
    assert_eq!(fis.fields[0].index_options, IndexOptions::Docs);
}

/// Port of `FieldInfos`' constructor checks -- Java reaches these from
/// `Lucene94FieldInfosFormat.read` itself (`return new
/// FieldInfos(infos)`), so a `.fnm` tripping one of them is rejected by
/// real Lucene, not merely frowned upon.
#[test]
fn duplicate_field_names_rejected() {
    let mut b = FnmBuilder::valid();
    b.fields.push(FieldBuilder::valid("dup", 0));
    b.fields.push(FieldBuilder::valid("dup", 1));
    assert!(matches!(
        parse(&b.build(), &b.id, &b.suffix),
        Err(Error::InvalidFieldInfos(msg)) if msg.contains("duplicate field names")
    ));
}

#[test]
fn duplicate_field_numbers_rejected() {
    let mut b = FnmBuilder::valid();
    b.fields.push(FieldBuilder::valid("a", 4));
    b.fields.push(FieldBuilder::valid("b", 4));
    assert!(matches!(
        parse(&b.build(), &b.id, &b.suffix),
        Err(Error::InvalidFieldInfos(msg)) if msg.contains("duplicate field numbers")
    ));
}

#[test]
fn multiple_soft_deletes_fields_rejected() {
    let mut b = FnmBuilder::valid();
    let mut a = FieldBuilder::valid("a", 0);
    a.bits = SOFT_DELETES_FIELD;
    let mut c = FieldBuilder::valid("c", 1);
    c.bits = SOFT_DELETES_FIELD;
    b.fields.push(a);
    b.fields.push(c);
    assert!(matches!(
        parse(&b.build(), &b.id, &b.suffix),
        Err(Error::InvalidFieldInfos(msg)) if msg.contains("multiple soft-deletes fields")
    ));
}

#[test]
fn multiple_parent_fields_rejected() {
    let mut b = FnmBuilder::valid();
    let mut a = FieldBuilder::valid("a", 0);
    a.bits = PARENT_FIELD_FIELD;
    let mut c = FieldBuilder::valid("c", 1);
    c.bits = PARENT_FIELD_FIELD;
    b.fields.push(a);
    b.fields.push(c);
    assert!(matches!(
        parse(&b.build(), &b.id, &b.suffix),
        Err(Error::InvalidFieldInfos(msg)) if msg.contains("multiple parent fields")
    ));
}

/// One soft-deletes field and one parent field on *different* fields is
/// legal (only the same field being both is not -- see
/// `FieldInfo.checkConsistency`); the accessors report each one.
#[test]
fn distinct_soft_deletes_and_parent_fields_accepted() {
    let mut b = FnmBuilder::valid();
    let mut a = FieldBuilder::valid("__soft_deletes", 0);
    a.bits = SOFT_DELETES_FIELD;
    let mut c = FieldBuilder::valid("_parent", 1);
    c.bits = PARENT_FIELD_FIELD;
    b.fields.push(a);
    b.fields.push(c);
    let fis = parse(&b.build(), &b.id, &b.suffix).unwrap();
    assert_eq!(fis.soft_deletes_field(), Some("__soft_deletes"));
    assert_eq!(fis.parent_field(), Some("_parent"));
    assert_eq!(fis.field_by_name("_parent").unwrap().number, 1);
    assert!(fis.field_by_name("nope").is_none());
}

#[test]
fn no_soft_deletes_or_parent_field_reports_none() {
    let mut b = FnmBuilder::valid();
    b.fields.push(FieldBuilder::valid("id", 0));
    let fis = parse(&b.build(), &b.id, &b.suffix).unwrap();
    assert_eq!(fis.soft_deletes_field(), None);
    assert_eq!(fis.parent_field(), None);
}

#[test]
fn negative_field_number_rejected() {
    let mut b = FnmBuilder::valid();
    b.fields.push(FieldBuilder::valid("bad", -1));
    assert!(matches!(
        parse(&b.build(), &b.id, &b.suffix),
        Err(Error::InvalidFieldNumber(name, -1)) if name == "bad"
    ));
}

#[test]
fn unused_bits_rejected() {
    let mut b = FnmBuilder::valid();
    let mut f = FieldBuilder::valid("f", 0);
    f.bits = 0x80; // top bit, always unused
    b.fields.push(f);
    assert!(matches!(
        parse(&b.build(), &b.id, &b.suffix),
        Err(Error::UnusedBitsSet(0x80))
    ));
}

#[test]
fn parent_field_bit_rejected_when_format_too_old() {
    let mut b = FnmBuilder::valid();
    b.format_version = FORMAT_START;
    let mut f = FieldBuilder::valid("f", 0);
    f.bits = PARENT_FIELD_FIELD;
    b.fields.push(f);
    assert!(matches!(
        parse(&b.build(), &b.id, &b.suffix),
        Err(Error::ParentFieldBitSetButTooOld(_))
    ));
}

#[test]
fn parent_field_bit_accepted_at_current_format() {
    let mut b = FnmBuilder::valid();
    let mut f = FieldBuilder::valid("f", 0);
    f.bits = PARENT_FIELD_FIELD;
    b.fields.push(f);
    let fis = parse(&b.build(), &b.id, &b.suffix).unwrap();
    assert!(fis.fields[0].parent_field);
}

#[test]
fn corrupt_field_counts_are_errors_not_allocations() {
    let b = FnmBuilder::valid();
    let good = b.build();
    // The field count is the vint right after the index header.
    let at = codec_util::index_header_length(CODEC_NAME, &b.suffix);
    // -1, then i32::MAX: re-signed, so only the count is wrong.
    for (count, negative) in [
        ([0xff, 0xff, 0xff, 0xff, 0x0f], true),
        ([0xff, 0xff, 0xff, 0xff, 0x07], false),
    ] {
        let mut bad = good[..at].to_vec();
        bad.extend_from_slice(&count);
        bad.extend_from_slice(&good[at + 1..good.len() - 8]);
        let checksum = crc32fast::hash(&bad) as u64;
        bad.extend_from_slice(&checksum.to_be_bytes());
        let err = parse(&bad, &b.id, &b.suffix).unwrap_err();
        assert_eq!(
            err.to_string().contains("negative field count"),
            negative,
            "{err}"
        );
    }
}

#[test]
fn doc_values_skipper_bit_rejected_when_format_too_old() {
    let mut b = FnmBuilder::valid();
    b.format_version = FORMAT_PARENT_FIELD; // < FORMAT_DOCVALUE_SKIPPER
    let mut f = FieldBuilder::valid("f", 0);
    f.bits = DOCVALUES_SKIPPER;
    f.doc_values_skip_index_type = None; // omitted at this format version
    b.fields.push(f);
    assert!(matches!(
        parse(&b.build(), &b.id, &b.suffix),
        Err(Error::DocValuesSkipperBitSetButTooOld(_))
    ));
}

#[test]
fn doc_values_skip_index_range_accepted_with_compatible_type() {
    let mut b = FnmBuilder::valid();
    let mut f = FieldBuilder::valid("f", 0);
    f.doc_values_type = 1; // Numeric
    f.doc_values_skip_index_type = Some(1); // Range
    b.fields.push(f);
    let fis = parse(&b.build(), &b.id, &b.suffix).unwrap();
    assert_eq!(
        fis.fields[0].doc_values_skip_index_type,
        DocValuesSkipIndexType::Range
    );
}

#[test]
fn doc_values_skip_index_range_incompatible_with_none_type_rejected() {
    let mut b = FnmBuilder::valid();
    let mut f = FieldBuilder::valid("f", 0);
    f.doc_values_type = 0; // None
    f.doc_values_skip_index_type = Some(1); // Range: incompatible with None
    b.fields.push(f);
    assert!(matches!(
        parse(&b.build(), &b.id, &b.suffix),
        Err(Error::Inconsistent(_, _))
    ));
}

#[test]
fn invalid_index_options_byte_rejected() {
    let mut b = FnmBuilder::valid();
    let mut f = FieldBuilder::valid("f", 0);
    f.index_options = 6; // out of range
    b.fields.push(f);
    assert!(matches!(
        parse(&b.build(), &b.id, &b.suffix),
        Err(Error::InvalidIndexOptions(6))
    ));
}

#[test]
fn invalid_doc_values_type_byte_rejected() {
    let mut b = FnmBuilder::valid();
    let mut f = FieldBuilder::valid("f", 0);
    f.doc_values_type = 6;
    b.fields.push(f);
    assert!(matches!(
        parse(&b.build(), &b.id, &b.suffix),
        Err(Error::InvalidDocValuesType(6))
    ));
}

#[test]
fn invalid_doc_values_skip_index_byte_rejected() {
    let mut b = FnmBuilder::valid();
    let mut f = FieldBuilder::valid("f", 0);
    f.doc_values_skip_index_type = Some(2); // out of range
    b.fields.push(f);
    assert!(matches!(
        parse(&b.build(), &b.id, &b.suffix),
        Err(Error::InvalidDocValuesSkipIndexType(2))
    ));
}

#[test]
fn invalid_vector_encoding_byte_rejected() {
    let mut b = FnmBuilder::valid();
    let mut f = FieldBuilder::valid("f", 0);
    f.vector_encoding = 2; // out of range
    b.fields.push(f);
    assert!(matches!(
        parse(&b.build(), &b.id, &b.suffix),
        Err(Error::InvalidVectorEncoding(2))
    ));
}

#[test]
fn invalid_vector_similarity_byte_rejected() {
    let mut b = FnmBuilder::valid();
    let mut f = FieldBuilder::valid("f", 0);
    f.vector_similarity_function = 4; // out of range
    b.fields.push(f);
    assert!(matches!(
        parse(&b.build(), &b.id, &b.suffix),
        Err(Error::InvalidVectorSimilarityFunction(4))
    ));
}

#[test]
fn points_field_parses_dimensions() {
    let mut b = FnmBuilder::valid();
    let mut f = FieldBuilder::valid("point", 0);
    f.point_dimension_count = 1;
    f.point_index_dimension_count = 1;
    f.point_num_bytes = 8;
    b.fields.push(f);
    let fis = parse(&b.build(), &b.id, &b.suffix).unwrap();
    assert_eq!(fis.fields[0].point_dimension_count, 1);
    assert_eq!(fis.fields[0].point_num_bytes, 8);
}

// --- check_consistency ---

#[test]
fn payloads_without_positions_rejected() {
    let mut b = FnmBuilder::valid();
    let mut f = FieldBuilder::valid("f", 0);
    f.index_options = 2; // DocsAndFreqs: no positions
    f.bits = STORE_PAYLOADS;
    b.fields.push(f);
    assert!(matches!(
        parse(&b.build(), &b.id, &b.suffix),
        Err(Error::Inconsistent(_, _))
    ));
}

#[test]
fn payloads_with_positions_accepted() {
    let mut b = FnmBuilder::valid();
    let mut f = FieldBuilder::valid("f", 0);
    f.index_options = 3; // DocsAndFreqsAndPositions
    f.bits = STORE_PAYLOADS;
    b.fields.push(f);
    let fis = parse(&b.build(), &b.id, &b.suffix).unwrap();
    assert!(fis.fields[0].store_payloads);
}

/// A non-indexed field carrying any of the three indexed-only bits is a
/// file real Lucene **reads**: `Lucene94FieldInfosFormat.read` builds each
/// `FieldInfo` through the constructor, whose "for non-indexed fields,
/// leave defaults" branch clears all three *before* `checkConsistency()`
/// runs. This port rejected the file instead, which made a segment its own
/// writer had produced unopenable. Real Lucene's own answer for the
/// same bytes is pinned by `fixtures/src/VerifyFieldInfos.java`'s
/// `noindex_flags` field.
#[test]
fn non_indexed_field_has_its_indexed_only_bits_coerced_away_not_rejected() {
    for bits in [
        STORE_TERMVECTOR,
        STORE_PAYLOADS,
        OMIT_NORMS,
        STORE_TERMVECTOR | STORE_PAYLOADS | OMIT_NORMS,
    ] {
        let mut b = FnmBuilder::valid();
        let mut f = FieldBuilder::valid("f", 0);
        f.index_options = 0; // None
        f.bits = bits;
        b.fields.push(f);
        let fis = parse(&b.build(), &b.id, &b.suffix)
            .unwrap_or_else(|e| panic!("bits {bits:#04x} must parse, got {e}"));
        let f = &fis.fields[0];
        assert!(!f.store_term_vectors, "bits {bits:#04x}");
        assert!(!f.store_payloads, "bits {bits:#04x}");
        assert!(!f.omit_norms, "bits {bits:#04x}");
    }
}

// --- the constructor (`FieldInfo::new` / `with_*` / `checked`) ---

#[test]
fn the_constructor_seed_is_consistent_and_carries_javas_non_vector_defaults() {
    let f = FieldInfo::new("f", 3).checked().unwrap();
    assert_eq!(f.name, "f");
    assert_eq!(f.number, 3);
    assert_eq!(f.index_options, IndexOptions::None);
    assert_eq!(f.doc_values_type, DocValuesType::None);
    assert_eq!(f.doc_values_gen, -1);
    assert_eq!(f.vector_encoding, VectorEncoding::Float32);
    assert_eq!(
        f.vector_similarity_function,
        VectorSimilarityFunction::Euclidean
    );
}

#[test]
fn the_constructor_coerces_the_three_indexed_only_flags_off_a_non_indexed_field() {
    let f = FieldInfo::new("f", 0)
        .with_store_term_vectors(true)
        .with_store_payloads(true)
        .with_omit_norms(true)
        .checked()
        .expect("Java's constructor coerces rather than throwing here");
    assert!(!f.store_term_vectors);
    assert!(!f.store_payloads);
    assert!(!f.omit_norms);
}

#[test]
fn the_constructor_rejects_payloads_without_positions() {
    // The one indexed-field combination the coercion does NOT rescue: the
    // field *is* indexed, so `storePayloads` survives to `checkConsistency`.
    let err = FieldInfo::new("f", 0)
        .with_index_options(IndexOptions::DocsAndFreqs)
        .with_store_payloads(true)
        .checked()
        .unwrap_err();
    assert!(matches!(err, Error::Inconsistent(_, _)), "{err}");
    // ... and accepts it once positions are indexed.
    assert!(FieldInfo::new("f", 0)
        .with_index_options(IndexOptions::DocsAndFreqsAndPositions)
        .with_store_payloads(true)
        .checked()
        .is_ok());
}

#[test]
fn the_constructor_rejects_every_other_check_consistency_violation() {
    let cases: Vec<FieldInfo> = vec![
        // docValuesSkipIndex incompatible with the doc-values type
        FieldInfo::new("f", 0).with_doc_values(
            DocValuesType::None,
            DocValuesSkipIndexType::Range,
            -1,
        ),
        // a docvalues update generation without doc values
        FieldInfo::new("f", 0).with_doc_values(
            DocValuesType::None,
            DocValuesSkipIndexType::None,
            5,
        ),
        FieldInfo::new("f", 0).with_points(-1, 0, 0),
        FieldInfo::new("f", 0).with_points(1, -1, 4),
        FieldInfo::new("f", 0).with_points(1, 1, -1),
        // pointNumBytes must be > 0 when pointDimensionCount != 0
        FieldInfo::new("f", 0).with_points(1, 1, 0),
        // pointIndexDimensionCount must be 0 when pointDimensionCount == 0
        FieldInfo::new("f", 0).with_points(0, 1, 0),
        // pointDimensionCount must be > 0 when pointNumBytes != 0
        FieldInfo::new("f", 0).with_points(0, 0, 4),
        FieldInfo::new("f", 0).with_vectors(
            -1,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Cosine,
        ),
        FieldInfo::new("f", 0)
            .with_soft_deletes_field(true)
            .with_parent_field(true),
    ];
    for case in cases {
        let described = format!("{case:?}");
        assert!(
            matches!(case.checked(), Err(Error::Inconsistent(_, _))),
            "must be rejected: {described}"
        );
    }
}

#[test]
fn the_constructor_setters_each_reach_the_field_they_name() {
    let f = FieldInfo::new("f", 7)
        .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
        .with_store_term_vectors(true)
        .with_omit_norms(true)
        .with_store_payloads(true)
        .with_doc_values(
            DocValuesType::SortedNumeric,
            DocValuesSkipIndexType::Range,
            4,
        )
        .with_attributes(vec![("k".to_string(), "v".to_string())])
        .with_points(2, 1, 8)
        .with_vectors(
            16,
            VectorEncoding::Byte,
            VectorSimilarityFunction::MaximumInnerProduct,
        )
        .with_soft_deletes_field(true)
        .checked()
        .unwrap();
    assert_eq!(
        f.index_options,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets
    );
    assert!(f.store_term_vectors && f.omit_norms && f.store_payloads);
    assert_eq!(f.doc_values_type, DocValuesType::SortedNumeric);
    assert_eq!(f.doc_values_skip_index_type, DocValuesSkipIndexType::Range);
    assert_eq!(f.doc_values_gen, 4);
    assert_eq!(f.attributes, vec![("k".to_string(), "v".to_string())]);
    assert_eq!(
        (
            f.point_dimension_count,
            f.point_index_dimension_count,
            f.point_num_bytes
        ),
        (2, 1, 8)
    );
    assert_eq!(f.vector_dimension, 16);
    assert_eq!(f.vector_encoding, VectorEncoding::Byte);
    assert_eq!(
        f.vector_similarity_function,
        VectorSimilarityFunction::MaximumInnerProduct
    );
    assert!(f.soft_deletes_field);
    assert!(
        FieldInfo::new("f", 0)
            .with_parent_field(true)
            .checked()
            .unwrap()
            .parent_field
    );
}

#[test]
fn field_infos_new_runs_both_the_per_field_and_the_cross_field_checks() {
    // Per-field: the payloads-without-positions violation above.
    assert!(matches!(
        FieldInfos::new(vec![FieldInfo::new("f", 0)
            .with_index_options(IndexOptions::DocsAndFreqs)
            .with_store_payloads(true)]),
        Err(Error::Inconsistent(_, _))
    ));
    // Cross-field: duplicate names, duplicate numbers, two soft-deletes
    // fields, two parent fields.
    assert!(matches!(
        FieldInfos::new(vec![FieldInfo::new("f", 0), FieldInfo::new("f", 1)]),
        Err(Error::InvalidFieldInfos(_))
    ));
    assert!(matches!(
        FieldInfos::new(vec![FieldInfo::new("a", 0), FieldInfo::new("b", 0)]),
        Err(Error::InvalidFieldInfos(_))
    ));
    assert!(matches!(
        FieldInfos::new(vec![
            FieldInfo::new("a", 0).with_soft_deletes_field(true),
            FieldInfo::new("b", 1).with_soft_deletes_field(true),
        ]),
        Err(Error::InvalidFieldInfos(_))
    ));
    assert!(matches!(
        FieldInfos::new(vec![
            FieldInfo::new("a", 0).with_parent_field(true),
            FieldInfo::new("b", 1).with_parent_field(true),
        ]),
        Err(Error::InvalidFieldInfos(_))
    ));
    // ... and the coercion is applied on the way through.
    let infos = FieldInfos::new(vec![FieldInfo::new("a", 0)
        .with_omit_norms(true)
        .with_store_term_vectors(true)])
    .unwrap();
    assert!(!infos.fields[0].omit_norms);
    assert!(!infos.fields[0].store_term_vectors);
}

#[test]
fn doc_values_gen_without_doc_values_rejected() {
    let mut b = FnmBuilder::valid();
    let mut f = FieldBuilder::valid("f", 0);
    f.doc_values_type = 0; // None
    f.doc_values_gen = 5;
    b.fields.push(f);
    assert!(matches!(
        parse(&b.build(), &b.id, &b.suffix),
        Err(Error::Inconsistent(_, _))
    ));
}

#[test]
fn point_num_bytes_zero_with_nonzero_dimension_count_rejected() {
    // Bypass the builder's own `if point_dimension_count != 0` write-side
    // branch (which always writes a consistent numBytes) by writing the
    // point fields by hand: dimension count 1, index dim count 1, num
    // bytes 0 — an invariant violation only reachable via crafted bytes.
    let mut b = FnmBuilder::valid();
    b.fields.push(FieldBuilder::valid("f", 0));
    let mut bytes = b.build();
    // Find the point_dimension_count vint (0x00) just before the trailing
    // vector fields + footer, and hand-patch a violation in a fresh
    // buffer instead, since the builder always keeps points consistent.
    bytes.clear();
    let mut out = Vec::new();
    out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut out, CODEC_NAME);
    out.extend_from_slice(&(FORMAT_CURRENT as u32).to_be_bytes());
    out.extend_from_slice(&b.id);
    out.push(0); // empty suffix
    write_vint(&mut out, 1); // one field
    write_string(&mut out, "f");
    write_vint(&mut out, 0); // number
    out.push(0); // bits
    out.push(0); // index options: None
    out.push(0); // doc values type: None
    out.push(0); // doc values skip index: None
    out.extend_from_slice(&(-1i64).to_le_bytes()); // doc values gen
    write_vint(&mut out, 0); // attributes
    write_vint(&mut out, 1); // pointDimensionCount = 1
    write_vint(&mut out, 1); // pointIndexDimensionCount = 1
    write_vint(&mut out, 0); // pointNumBytes = 0 (invalid: must be >0)
    write_vint(&mut out, 0); // vectorDimension
    out.push(0); // vector encoding
    out.push(0); // vector similarity
    out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&out) as u64;
    out.extend_from_slice(&checksum.to_be_bytes());

    assert!(matches!(
        parse(&out, &b.id, &b.suffix),
        Err(Error::Inconsistent(_, _))
    ));
}

// --- write() round-trips through parse() ---

fn sample_field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::Docs,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Float32,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

#[test]
fn write_empty_round_trips() {
    let id = [7u8; ID_LENGTH];
    let bytes = write(&[], &id, "");
    let fis = parse(&bytes, &id, "").unwrap();
    assert_eq!(fis.fields.len(), 0);
}

#[test]
fn write_plain_field_round_trips() {
    let id = [7u8; ID_LENGTH];
    let field = sample_field("id", 0);
    let bytes = write(&[field], &id, "");
    let fis = parse(&bytes, &id, "").unwrap();
    assert_eq!(fis.fields.len(), 1);
    assert_eq!(fis.fields[0].name, "id");
    assert_eq!(fis.fields[0].number, 0);
    assert_eq!(fis.fields[0].index_options, IndexOptions::Docs);
}

#[test]
fn write_term_vectors_and_payloads_round_trip() {
    let id = [7u8; ID_LENGTH];
    let mut field = sample_field("with_tv", 1);
    field.store_term_vectors = true;
    field.store_payloads = true;
    field.index_options = IndexOptions::DocsAndFreqsAndPositions;
    let bytes = write(&[field], &id, "sfx");
    let fis = parse(&bytes, &id, "sfx").unwrap();
    assert!(fis.fields[0].store_term_vectors);
    assert!(fis.fields[0].store_payloads);
}

#[test]
fn write_soft_deletes_and_parent_field_round_trip() {
    let id = [7u8; ID_LENGTH];
    let mut soft = sample_field("__soft_deletes", 2);
    soft.soft_deletes_field = true;
    let mut parent = sample_field("__parent", 3);
    parent.parent_field = true;
    let bytes = write(&[soft, parent], &id, "");
    let fis = parse(&bytes, &id, "").unwrap();
    assert!(fis.fields[0].soft_deletes_field);
    assert!(fis.fields[1].parent_field);
}

#[test]
fn write_doc_values_and_skip_index_round_trip() {
    let id = [7u8; ID_LENGTH];
    let mut field = sample_field("num_dv", 0);
    field.doc_values_type = DocValuesType::Numeric;
    field.doc_values_skip_index_type = DocValuesSkipIndexType::Range;
    field.doc_values_gen = 42;
    field.attributes = vec![("k1".to_string(), "v1".to_string())];
    let bytes = write(std::slice::from_ref(&field), &id, "");
    // Only the skip-index-type byte differs from the same field without
    // one (and the checksum): Java sets no 0x20 bit in the flags byte.
    let mut plain = field.clone();
    plain.doc_values_skip_index_type = DocValuesSkipIndexType::None;
    let plain_bytes = write(&[plain], &id, "");
    let body = bytes.len() - lucene_store::codec_util::FOOTER_LENGTH;
    let differing = (0..body).filter(|&i| bytes[i] != plain_bytes[i]).count();
    assert_eq!((bytes.len(), differing), (plain_bytes.len(), 1));
    let fis = parse(&bytes, &id, "").unwrap();
    assert_eq!(fis.fields[0].doc_values_type, DocValuesType::Numeric);
    assert_eq!(
        fis.fields[0].doc_values_skip_index_type,
        DocValuesSkipIndexType::Range
    );
    assert_eq!(fis.fields[0].doc_values_gen, 42);
    assert_eq!(
        fis.fields[0].attributes,
        vec![("k1".to_string(), "v1".to_string())]
    );
}

#[test]
fn write_points_field_round_trips() {
    let id = [7u8; ID_LENGTH];
    let mut field = sample_field("point_field", 0);
    field.point_dimension_count = 1;
    field.point_index_dimension_count = 1;
    field.point_num_bytes = 8;
    let bytes = write(&[field], &id, "");
    let fis = parse(&bytes, &id, "").unwrap();
    assert_eq!(fis.fields[0].point_dimension_count, 1);
    assert_eq!(fis.fields[0].point_index_dimension_count, 1);
    assert_eq!(fis.fields[0].point_num_bytes, 8);
}

#[test]
fn write_vector_field_round_trips() {
    let id = [7u8; ID_LENGTH];
    let mut field = sample_field("vector_field", 0);
    field.vector_dimension = 3;
    field.vector_encoding = VectorEncoding::Byte;
    field.vector_similarity_function = VectorSimilarityFunction::Cosine;
    let bytes = write(&[field], &id, "");
    let fis = parse(&bytes, &id, "").unwrap();
    assert_eq!(fis.fields[0].vector_dimension, 3);
    assert_eq!(fis.fields[0].vector_encoding, VectorEncoding::Byte);
    assert_eq!(
        fis.fields[0].vector_similarity_function,
        VectorSimilarityFunction::Cosine
    );
}

#[test]
fn write_multiple_fields_preserve_order() {
    let id = [7u8; ID_LENGTH];
    let fields = vec![
        sample_field("a", 0),
        sample_field("b", 1),
        sample_field("c", 2),
    ];
    let bytes = write(&fields, &id, "");
    let fis = parse(&bytes, &id, "").unwrap();
    assert_eq!(fis.fields.len(), 3);
    assert_eq!(fis.fields[0].name, "a");
    assert_eq!(fis.fields[1].name, "b");
    assert_eq!(fis.fields[2].name, "c");
}

#[test]
fn wrong_id_rejected() {
    let b = FnmBuilder::valid();
    let wrong_id = [9u8; ID_LENGTH];
    assert!(matches!(
        parse(&b.build(), &wrong_id, &b.suffix),
        Err(Error::Store(_))
    ));
}
