//! `BaseFieldInfoFormatTestCase`: the `.fnm` format's round trip, over
//! `lucene_codecs::field_infos::{write, parse}` (`Lucene94FieldInfosFormat`)
//! through a directory.
//!
//! Differs:
//!
//! - The format is two byte functions: [`write_field_infos`] and
//!   [`read_field_infos`] are its `write(dir, si, "", infos)` and
//!   `read(dir, si, "")`, the file named as `IndexFileNames` names it.
//! - `FieldInfos.Builder` is [`FieldInfos::new`] over the fields numbered
//!   in the order they are added (what `FieldNumbers.addOrGet` hands a new
//!   name); its consistency checks are the same constructor checks.
//! - `testImmutableAttributes` checks the attributes round trip: a parsed
//!   [`FieldInfo`] is a value the caller owns, so there is no map to refuse
//!   a `put`. `testExceptionOnCloseInput` is not ported: a read has no
//!   input `close` to fail.
//! - `testRandom`'s field names are Java's draws kept in draw order (not
//!   `HashSet` order), so a seed does not number the fields as Java's does.

use lucene_codecs::field_infos::{
    self, DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos, IndexOptions,
    VectorEncoding, VectorSimilarityFunction,
};
use lucene_index::document::{Document, Field, StoredValue};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::Directory;
use lucene_util::java_random::JavaRandom;

use super::base_index_file_format::{write_file, IndexFileFormatSuite, SuiteTest};
use crate::store::base_directory_wrapper::BaseDirectoryWrapper as _;
use crate::store::mock_directory_wrapper::{
    is_fake_io_exception, FailOn, FailureContext, FailureOp, MockDirectoryWrapper,
};
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::random_numbers::random_from;
use crate::util::test_util;

/// `IndexOptions.values()`, in Java's order.
const INDEX_OPTIONS: [IndexOptions; 6] = [
    IndexOptions::None,
    IndexOptions::Docs,
    IndexOptions::DocsAndFreqs,
    IndexOptions::DocsAndFreqsAndPositions,
    IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
    IndexOptions::DocsAndCustomFreqs,
];

/// `DocValuesType.values()`, in Java's order.
const DOC_VALUES_TYPES: [DocValuesType; 6] = [
    DocValuesType::None,
    DocValuesType::Numeric,
    DocValuesType::Binary,
    DocValuesType::Sorted,
    DocValuesType::SortedNumeric,
    DocValuesType::SortedSet,
];

/// `VectorEncoding.values()`.
const VECTOR_ENCODINGS: [VectorEncoding; 2] = [VectorEncoding::Byte, VectorEncoding::Float32];

/// `VectorSimilarityFunction.values()`.
const SIMILARITIES: [VectorSimilarityFunction; 4] = [
    VectorSimilarityFunction::Euclidean,
    VectorSimilarityFunction::DotProduct,
    VectorSimilarityFunction::Cosine,
    VectorSimilarityFunction::MaximumInnerProduct,
];

/// `PointValues.MAX_DIMENSIONS`, `MAX_INDEX_DIMENSIONS`, `MAX_NUM_BYTES`.
const MAX_DIMENSIONS: i32 = 16;
const MAX_INDEX_DIMENSIONS: i32 = 8;
const MAX_NUM_BYTES: i32 = 16;

/// `Lucene99HnswVectorsFormat.getMaxDimensions(field)`.
const MAX_VECTOR_DIMENSIONS: i32 = 1024;

/// The field-info suite's `addRandomFields`: a stored `foobar`.
pub struct FieldInfoSuite;

impl IndexFileFormatSuite for FieldInfoSuite {
    fn add_random_fields(&self, t: &mut LuceneTestCase, doc: &mut Document) {
        let value = test_util::random_simple_string(t.random());
        doc.add(Field::stored("foobar", StoredValue::String(value)));
    }
}

/// `codec.fieldInfosFormat().write(dir, segmentInfo, "", infos)`.
///
/// # Errors
/// What the directory refuses.
pub fn write_field_infos(
    dir: &dyn Directory,
    segment: &str,
    id: &[u8; ID_LENGTH],
    infos: &FieldInfos,
) -> lucene_store::Result<()> {
    let bytes = field_infos::write(&infos.fields, id, "");
    write_file(dir, &format!("{segment}.fnm"), &bytes)
}

/// `codec.fieldInfosFormat().read(dir, segmentInfo, "")`.
///
/// # Errors
/// The directory's refusal or the parser's, as text.
pub fn read_field_infos(
    dir: &dyn Directory,
    segment: &str,
    id: &[u8; ID_LENGTH],
) -> Result<FieldInfos, String> {
    let bytes = dir
        .open(&format!("{segment}.fnm"))
        .map_err(|e| e.to_string())?;
    field_infos::parse(&bytes, id, "").map_err(|e| e.to_string())
}

/// `newSegmentInfo(dir, "_123")`'s part the format reads: its id. Draws
/// Java's `minVersion` coin.
fn new_segment_id(t: &mut LuceneTestCase) -> [u8; ID_LENGTH] {
    let _min_version_is_null = t.random().next_boolean();
    lucene_util::string_helper::random_id()
}

/// `createFieldInfo()`: `field`, indexed as `TextField.TYPE_STORED`.
fn create_field_info() -> FieldInfo {
    FieldInfo::new("field", 0).with_index_options(IndexOptions::DocsAndFreqsAndPositions)
}

/// `FieldInfos.Builder.add(...).finish()`: numbered in add order.
fn finish(fields: Vec<FieldInfo>) -> FieldInfos {
    let numbered = fields
        .into_iter()
        .enumerate()
        .map(|(n, f)| FieldInfo {
            number: n as i32,
            ..f
        })
        .collect();
    FieldInfos::new(numbered).expect("a consistent FieldInfos")
}

fn round_trip(t: &mut LuceneTestCase, infos: &FieldInfos) -> FieldInfos {
    let dir = t.new_directory();
    let id = new_segment_id(t);
    write_field_infos(&*dir, "_123", &id, infos).expect("write");
    let read = read_field_infos(&*dir, "_123", &id).expect("read");
    dir.close().expect("close");
    read
}

fn has_norms(f: &FieldInfo) -> bool {
    f.index_options != IndexOptions::None && !f.omit_norms
}

/// `testOneField`.
pub fn test_one_field(t: &mut LuceneTestCase) {
    let infos = finish(vec![create_field_info()]);
    let infos2 = round_trip(t, &infos);
    assert_eq!(infos2.fields.len(), 1);
    let f = infos2.field_by_name("field").expect("field");
    assert_ne!(f.index_options, IndexOptions::None);
    assert_eq!(f.doc_values_type, DocValuesType::None);
    assert!(!f.omit_norms);
    assert!(!f.store_payloads);
    assert!(!f.store_term_vectors);
    assert_eq!(f.point_dimension_count, 0);
    assert_eq!(f.vector_dimension, 0);
    assert!(!f.soft_deletes_field);
}

/// `testImmutableAttributes`: the attributes come back as written.
pub fn test_immutable_attributes(t: &mut LuceneTestCase) {
    let attributes = vec![
        ("foo".to_string(), "bar".to_string()),
        ("bar".to_string(), "baz".to_string()),
    ];
    let infos = finish(vec![create_field_info().with_attributes(attributes.clone())]);
    let infos2 = round_trip(t, &infos);
    assert_eq!(infos2.fields.len(), 1);
    let mut read = infos2
        .field_by_name("field")
        .expect("field")
        .attributes
        .clone();
    let mut expected = attributes;
    read.sort();
    expected.sort();
    assert_eq!(read, expected);
}

fn failing(
    t: &mut LuceneTestCase,
    op: FailureOp,
) -> (MockDirectoryWrapper, std::sync::Arc<FailOn>) {
    let dir = t.new_mock_directory();
    let fail = FailOn::new(move |ctx: &FailureContext<'_>| ctx.op == op);
    fail.clear_do_fail();
    dir.fail_on(fail.clone());
    (dir, fail)
}

fn exception_on_write(t: &mut LuceneTestCase, op: FailureOp) {
    let (dir, fail) = failing(t, op);
    let id = new_segment_id(t);
    let infos = finish(vec![create_field_info()]);
    fail.set_do_fail();
    let err = write_field_infos(&dir, "_123", &id, &infos).expect_err("the write must fail");
    assert!(is_fake_io_exception(&err), "{err}");
    fail.clear_do_fail();
    assert!(fail.fired() > 0);
    dir.close().expect("close");
}

/// `testExceptionOnCreateOutput`.
pub fn test_exception_on_create_output(t: &mut LuceneTestCase) {
    exception_on_write(t, FailureOp::CreateOutput);
}

/// `testExceptionOnCloseOutput`.
pub fn test_exception_on_close_output(t: &mut LuceneTestCase) {
    exception_on_write(t, FailureOp::CloseOutput);
}

/// `testExceptionOnOpenInput`.
pub fn test_exception_on_open_input(t: &mut LuceneTestCase) {
    let (dir, fail) = failing(t, FailureOp::OpenInput);
    let id = new_segment_id(t);
    let infos = finish(vec![create_field_info()]);
    write_field_infos(&dir, "_123", &id, &infos).expect("write");
    fail.set_do_fail();
    let err = read_field_infos(&dir, "_123", &id).expect_err("the read must fail");
    assert!(err.contains("FakeIOException"), "{err}");
    fail.clear_do_fail();
    assert!(fail.fired() > 0);
    dir.close().expect("close");
}

/// The parts of `randomFieldType(r, name)` a `FieldInfo` keeps.
struct RandomType {
    index_options: IndexOptions,
    omit_norms: bool,
    store_term_vectors: bool,
    doc_values_type: DocValuesType,
    skip: DocValuesSkipIndexType,
    points: (i32, i32, i32),
    vector: (i32, VectorEncoding, VectorSimilarityFunction),
}

/// `randomFieldType(r, fieldName)`, Java's draws in Java's order.
fn random_field_type(r: &mut JavaRandom) -> RandomType {
    let mut ty = RandomType {
        index_options: IndexOptions::None,
        omit_norms: false,
        store_term_vectors: false,
        doc_values_type: DocValuesType::None,
        skip: DocValuesSkipIndexType::None,
        points: (0, 0, 0),
        vector: (
            0,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        ),
    };
    if r.next_boolean() {
        ty.index_options = INDEX_OPTIONS[r.next_int_bounded(INDEX_OPTIONS.len() as i32) as usize];
        ty.omit_norms = r.next_boolean();
        if r.next_boolean() {
            ty.store_term_vectors = true;
            if ty.index_options.subsumes_positions() {
                let positions = r.next_boolean();
                let _offsets = r.next_boolean();
                if positions {
                    let _payloads = r.next_boolean();
                }
            }
        }
    }
    if r.next_boolean() {
        let n = DOC_VALUES_TYPES.len() as i32;
        let current = DOC_VALUES_TYPES[r.next_int_bounded(n) as usize];
        // Java sets a second draw, and keys the skip index off the first.
        ty.doc_values_type = DOC_VALUES_TYPES[r.next_int_bounded(n) as usize];
        if matches!(
            current,
            DocValuesType::Numeric
                | DocValuesType::SortedNumeric
                | DocValuesType::Sorted
                | DocValuesType::SortedSet
        ) {
            ty.skip = DocValuesSkipIndexType::Range;
        }
    }
    if r.next_boolean() {
        let dimension = 1 + r.next_int_bounded(MAX_DIMENSIONS);
        let index_dimension = 1 + r.next_int_bounded(dimension.min(MAX_INDEX_DIMENSIONS));
        let num_bytes = 1 + r.next_int_bounded(MAX_NUM_BYTES);
        ty.points = (dimension, index_dimension, num_bytes);
    }
    if r.next_boolean() {
        let dimension = 1 + r.next_int_bounded(MAX_VECTOR_DIMENSIONS);
        let similarity = *random_from(r, &SIMILARITIES);
        let encoding = *random_from(r, &VECTOR_ENCODINGS);
        ty.vector = (dimension, encoding, similarity);
    }
    ty
}

/// `testRandom`.
pub fn test_random(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let id = new_segment_id(t);
    let num_fields = t.at_least(2000);
    let r = t.random();
    let mut names: Vec<String> = Vec::new();
    for _ in 0..num_fields {
        let name = test_util::random_unicode_string(r);
        if !names.contains(&name) {
            names.push(name);
        }
    }
    let soft_deletes = r
        .next_boolean()
        .then(|| test_util::random_unicode_string(r));
    let mut parent = r
        .next_boolean()
        .then(|| test_util::random_unicode_string(r));
    if soft_deletes.is_some() && soft_deletes == parent {
        parent = None;
    }
    let mut fields = Vec::with_capacity(names.len());
    for name in names {
        let ty = random_field_type(r);
        let (mut store_term_vectors, mut store_payloads, mut omit_norms) = (false, false, false);
        if ty.index_options != IndexOptions::None {
            store_term_vectors = ty.store_term_vectors;
            omit_norms = ty.omit_norms;
            if ty.index_options.subsumes_positions() {
                store_payloads = r.next_boolean();
            }
        }
        let skip = if matches!(
            ty.doc_values_type,
            DocValuesType::Numeric
                | DocValuesType::Sorted
                | DocValuesType::SortedNumeric
                | DocValuesType::SortedSet
        ) {
            ty.skip
        } else {
            DocValuesSkipIndexType::None
        };
        let soft = soft_deletes.as_deref() == Some(name.as_str());
        let is_parent = parent.as_deref() == Some(name.as_str());
        fields.push(FieldInfo {
            store_term_vectors,
            omit_norms,
            store_payloads,
            index_options: ty.index_options,
            doc_values_type: ty.doc_values_type,
            doc_values_skip_index_type: skip,
            point_dimension_count: ty.points.0,
            point_index_dimension_count: ty.points.1,
            point_num_bytes: ty.points.2,
            vector_dimension: ty.vector.0,
            vector_encoding: ty.vector.1,
            vector_similarity_function: ty.vector.2,
            soft_deletes_field: soft,
            parent_field: is_parent,
            ..FieldInfo::new(name, -1)
        });
    }
    let infos = finish(fields);
    write_field_infos(&*dir, "_123", &id, &infos).expect("write");
    let infos2 = read_field_infos(&*dir, "_123", &id).expect("read");
    assert_same_infos(&infos, &infos2);
    dir.close().expect("close");
}

/// `assertEquals(FieldInfos, FieldInfos)`.
pub fn assert_same_infos(expected: &FieldInfos, actual: &FieldInfos) {
    assert_eq!(expected.fields.len(), actual.fields.len());
    for e in &expected.fields {
        let a = actual
            .field_by_number(e.number)
            .expect("the same field number");
        assert_same_field(e, a);
    }
}

/// `assertEquals(FieldInfo, FieldInfo)`.
pub fn assert_same_field(expected: &FieldInfo, actual: &FieldInfo) {
    assert_eq!(expected.number, actual.number);
    assert_eq!(expected.name, actual.name);
    assert_eq!(expected.doc_values_type, actual.doc_values_type);
    assert_eq!(
        expected.doc_values_skip_index_type,
        actual.doc_values_skip_index_type
    );
    assert_eq!(expected.index_options, actual.index_options);
    assert_eq!(has_norms(expected), has_norms(actual));
    assert_eq!(expected.store_payloads, actual.store_payloads);
    assert_eq!(expected.store_term_vectors, actual.store_term_vectors);
    assert_eq!(expected.omit_norms, actual.omit_norms);
    assert_eq!(expected.doc_values_gen, actual.doc_values_gen);
}

/// Every `BaseFieldInfoFormatTestCase` test this port runs.
pub const TESTS: &[SuiteTest] = &[
    ("testOneField", test_one_field),
    ("testImmutableAttributes", test_immutable_attributes),
    (
        "testExceptionOnCreateOutput",
        test_exception_on_create_output,
    ),
    ("testExceptionOnCloseOutput", test_exception_on_close_output),
    ("testExceptionOnOpenInput", test_exception_on_open_input),
    ("testRandom", test_random),
];
