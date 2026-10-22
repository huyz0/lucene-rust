//! `BaseSegmentInfoFormatTestCase`: the `.si` format's round trip, over
//! `lucene_index::segment_info::{write, parse}` (`Lucene99SegmentInfoFormat`)
//! through a directory.
//!
//! Differs:
//!
//! - The format is two byte functions, not a `SegmentInfoFormat` object:
//!   [`write_segment_info`] and [`read_segment_info`] are its
//!   `write(dir, info)`/`read(dir, name, id)`, and they, not the port, add the
//!   `.si` to the segment's files as Java's `write` does (the port's callers
//!   each add it themselves).
//! - Java's immutability checks (`files()`, `getDiagnostics()` and
//!   `getAttributes()` refusing `add`/`put`) have nothing to check: a parsed
//!   [`SegmentInfo`] is a value the caller owns.
//! - `testExceptionOnCloseInput` is not ported: a directory hands out a
//!   whole file, so a read has no input `close` a failure could hit.
//! - `getVersions()` is `Version.LATEST` alone, as
//!   `TestLucene99SegmentInfoFormat`'s.
//! - `testRandom`'s files and diagnostics are Java's draws, kept in draw
//!   order (Java's `HashSet`/`HashMap` iteration order is not reproduced);
//!   the round trip compares them as sets.

use std::collections::BTreeSet;

use lucene_index::document::{Document, Field, StoredValue};
use lucene_index::segment_info::{
    self, IndexSortField, IndexSortKind, LuceneVersion, NumericSortKey, SegmentInfo,
    SortedNumericSelector, SortedSetSelector, StringMissingValue,
};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::Directory;
use lucene_util::java_random::JavaRandom;

use super::base_index_file_format::{write_file, IndexFileFormatSuite, SuiteTest, VERSION};
use crate::store::base_directory_wrapper::BaseDirectoryWrapper;
use crate::store::mock_directory_wrapper::{
    is_fake_io_exception, FailOn, FailureContext, FailureOp,
};
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::test_util;

/// `getVersions()`.
pub const VERSIONS: [LuceneVersion; 1] = [VERSION];

/// `IndexWriter.MAX_DOCS`.
const MAX_DOCS: i32 = lucene_index::index_writer::MAX_DOCS as i32;

/// The segment-info suite's `addRandomFields`: a stored `foobar`.
pub struct SegmentInfoSuite;

impl IndexFileFormatSuite for SegmentInfoSuite {
    fn add_random_fields(&self, t: &mut LuceneTestCase, doc: &mut Document) {
        let value = test_util::random_simple_string(t.random());
        doc.add(Field::stored("foobar", StoredValue::String(value)));
    }
}

/// `codec.segmentInfoFormat().write(dir, info)`: adds `<name>.si` to
/// `info.files`, then writes it.
///
/// # Errors
/// What the directory refuses.
pub fn write_segment_info(
    dir: &dyn Directory,
    name: &str,
    info: &mut SegmentInfo,
) -> lucene_store::Result<()> {
    let file = format!("{name}.si");
    if !info.files.contains(&file) {
        info.files.push(file.clone());
    }
    write_file(dir, &file, &segment_info::write(info, ""))
}

/// `codec.segmentInfoFormat().read(dir, name, id)`.
///
/// # Errors
/// The directory's refusal or the parser's, as text.
pub fn read_segment_info(
    dir: &dyn Directory,
    name: &str,
    id: &[u8; ID_LENGTH],
) -> Result<SegmentInfo, String> {
    let bytes = dir.open(&format!("{name}.si")).map_err(|e| e.to_string())?;
    segment_info::parse(&bytes, id).map_err(|e| e.to_string())
}

/// `StringHelper.randomId()`: its own stream, not the test's.
fn random_id() -> [u8; ID_LENGTH] {
    lucene_util::string_helper::random_id()
}

/// `new SegmentInfo(dir, v, minV, "_123", 1, false, hasBlocks, codec, diags,
/// id, attributes, sort)` with no files.
fn segment(
    min_version: Option<LuceneVersion>,
    id: [u8; ID_LENGTH],
    has_blocks: bool,
    diagnostics: Vec<(String, String)>,
    attributes: Vec<(String, String)>,
    index_sort: Option<Vec<IndexSortField>>,
) -> SegmentInfo {
    SegmentInfo {
        id,
        version: VERSIONS[0],
        min_version,
        doc_count: 1,
        is_compound_file: false,
        has_blocks,
        diagnostics,
        files: Vec::new(),
        attributes,
        index_sort,
    }
}

fn plain() -> SegmentInfo {
    let id = random_id();
    segment(Some(VERSIONS[0]), id, false, Vec::new(), Vec::new(), None)
}

fn set(v: &[String]) -> BTreeSet<&str> {
    v.iter().map(String::as_str).collect()
}

fn round_trip(t: &mut LuceneTestCase, info: &mut SegmentInfo) -> SegmentInfo {
    let dir = t.new_directory();
    round_trip_in(dir, info)
}

/// [`round_trip`] in a directory made before the info's own draws.
fn round_trip_in(dir: Box<dyn BaseDirectoryWrapper>, info: &mut SegmentInfo) -> SegmentInfo {
    write_segment_info(&*dir, "_123", info).expect("write");
    let read = read_segment_info(&*dir, "_123", &info.id).expect("read");
    dir.close().expect("close");
    read
}

/// `testFiles`.
pub fn test_files(t: &mut LuceneTestCase) {
    let mut info = plain();
    let info2 = round_trip(t, &mut info);
    assert_eq!(set(&info.files), set(&info2.files));
}

/// `testHasBlocks`.
pub fn test_has_blocks(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let id = random_id();
    let has_blocks = t.random().next_boolean();
    let mut info = segment(
        Some(VERSIONS[0]),
        id,
        has_blocks,
        Vec::new(),
        Vec::new(),
        None,
    );
    let info2 = round_trip_in(dir, &mut info);
    assert_eq!(info.has_blocks, info2.has_blocks);
}

/// `testAddsSelfToFiles`.
pub fn test_adds_self_to_files(t: &mut LuceneTestCase) {
    let mut info = plain();
    let original = vec!["_123.a".to_string()];
    info.files = original.clone();
    let info2 = round_trip(t, &mut info);
    let modified = set(&info.files);
    assert!(modified.is_superset(&set(&original)));
    assert!(
        modified.len() > original.len(),
        "did you forget to add yourself to files()"
    );
    assert_eq!(modified, set(&info2.files));
}

fn pairs(entries: &[(&str, &str)]) -> Vec<(String, String)> {
    entries
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// `testDiagnostics`.
pub fn test_diagnostics(t: &mut LuceneTestCase) {
    let diagnostics = pairs(&[("key1", "value1"), ("key2", "value2")]);
    let id = random_id();
    let mut info = segment(
        Some(VERSIONS[0]),
        id,
        false,
        diagnostics.clone(),
        Vec::new(),
        None,
    );
    let info2 = round_trip(t, &mut info);
    assert_eq!(sorted(diagnostics), sorted(info2.diagnostics));
}

/// `testAttributes`.
pub fn test_attributes(t: &mut LuceneTestCase) {
    let attributes = pairs(&[("key1", "value1"), ("key2", "value2")]);
    let id = random_id();
    let mut info = segment(
        Some(VERSIONS[0]),
        id,
        false,
        Vec::new(),
        attributes.clone(),
        None,
    );
    let info2 = round_trip(t, &mut info);
    assert_eq!(sorted(attributes), sorted(info2.attributes));
}

fn sorted(mut v: Vec<(String, String)>) -> Vec<(String, String)> {
    v.sort();
    v
}

/// `testUniqueID`.
pub fn test_unique_id(t: &mut LuceneTestCase) {
    let mut info = plain();
    let info2 = round_trip(t, &mut info);
    assert_eq!(info.id, info2.id);
}

/// `testVersions`.
pub fn test_versions(t: &mut LuceneTestCase) {
    for v in VERSIONS {
        for min_v in [Some(v), None] {
            let id = random_id();
            let mut info = segment(min_v, id, false, Vec::new(), Vec::new(), None);
            info.version = v;
            let info2 = round_trip(t, &mut info);
            assert_eq!(info2.version, v);
            assert_eq!(info2.min_version, min_v);
        }
    }
}

/// `randomIndexSortField()`.
fn random_index_sort_field(r: &mut JavaRandom) -> IndexSortField {
    let reverse = r.next_boolean();
    let choice = r.next_int_bounded(10);
    let field = test_util::random_simple_string(r);
    let numeric = |r: &mut JavaRandom, kind: i32| -> NumericSortKey {
        match kind {
            0 => NumericSortKey::Int(r.next_boolean().then(|| r.next_int())),
            1 => NumericSortKey::Long(r.next_boolean().then(|| r.next_long())),
            2 => NumericSortKey::Float(r.next_boolean().then(|| r.next_float())),
            _ => NumericSortKey::Double(r.next_boolean().then(|| r.next_double())),
        }
    };
    let string_missing = |r: &mut JavaRandom| {
        if r.next_boolean() {
            StringMissingValue::Last
        } else {
            StringMissingValue::None
        }
    };
    let kind = match choice {
        0..=7 if choice % 2 == 0 => IndexSortKind::Numeric(numeric(r, choice / 2)),
        0..=7 => IndexSortKind::SortedNumeric {
            key: numeric(r, choice / 2),
            selector: SortedNumericSelector::Min,
        },
        8 => IndexSortKind::String(string_missing(r)),
        _ => IndexSortKind::SortedSet {
            selector: SortedSetSelector::Min,
            missing: string_missing(r),
        },
    };
    IndexSortField {
        field,
        reverse,
        kind,
    }
}

/// `testSort`.
pub fn test_sort(t: &mut LuceneTestCase) {
    let iters = t.at_least(5);
    for i in 0..iters {
        let sort = (i != 0).then(|| {
            let num = test_util::next_int(t.random(), 1, 3);
            (0..num)
                .map(|_| random_index_sort_field(t.random()))
                .collect::<Vec<_>>()
        });
        let id = random_id();
        let mut info = segment(
            Some(VERSIONS[0]),
            id,
            false,
            Vec::new(),
            Vec::new(),
            sort.clone(),
        );
        let info2 = round_trip(t, &mut info);
        assert_eq!(sort, info2.index_sort);
    }
}

/// A mock directory failing `op` while armed, and the failure.
fn failing(
    t: &mut LuceneTestCase,
    op: FailureOp,
) -> (
    crate::store::mock_directory_wrapper::MockDirectoryWrapper,
    std::sync::Arc<FailOn>,
) {
    let dir = t.new_mock_directory();
    let fail = FailOn::new(move |ctx: &FailureContext<'_>| ctx.op == op);
    fail.clear_do_fail();
    dir.fail_on(fail.clone());
    (dir, fail)
}

fn exception_on_write(t: &mut LuceneTestCase, op: FailureOp) {
    let (dir, fail) = failing(t, op);
    let mut info = plain();
    fail.set_do_fail();
    let err = write_segment_info(&dir, "_123", &mut info).expect_err("the write must fail");
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
    let mut info = plain();
    write_segment_info(&dir, "_123", &mut info).expect("write");
    fail.set_do_fail();
    let err = read_segment_info(&dir, "_123", &info.id).expect_err("the read must fail");
    assert!(err.contains("FakeIOException"), "{err}");
    fail.clear_do_fail();
    assert!(fail.fired() > 0);
    dir.close().expect("close");
}

/// `testRandom`.
pub fn test_random(t: &mut LuceneTestCase) {
    for _ in 0..10 {
        let dir = t.new_directory();
        let r = t.random();
        let version = VERSIONS[r.next_int_bounded(VERSIONS.len() as i32) as usize];
        let index = r.next_long().wrapping_abs();
        let index = if index == i64::MIN {
            i64::from(r.next_int_bounded(i32::MAX))
        } else {
            index
        };
        let name = format!("_{}", lucene_util::base36::to_base36(index));
        let doc_count = test_util::next_int(r, 1, MAX_DOCS);
        let is_compound_file = r.next_boolean();
        let mut files = Vec::new();
        for j in 0..r.next_int_bounded(10) {
            let file = format!("{name}.{j}");
            if !files.contains(&file) {
                files.push(file.clone());
            }
            write_file(&*dir, &file, &[]).expect("createOutput");
        }
        let mut diagnostics: Vec<(String, String)> = Vec::new();
        for _ in 0..r.next_int_bounded(10) {
            let key = test_util::random_unicode_string(r);
            let value = test_util::random_unicode_string(r);
            put(&mut diagnostics, key, value);
        }
        let mut id = [0; ID_LENGTH];
        r.next_bytes(&mut id);
        let mut attributes: Vec<(String, String)> = Vec::new();
        for _ in 0..r.next_int_bounded(10) {
            let key = test_util::random_unicode_string(r);
            let value = test_util::random_unicode_string(r);
            put(&mut attributes, key, value);
        }
        let mut info = SegmentInfo {
            id,
            version,
            min_version: None,
            doc_count,
            is_compound_file,
            has_blocks: false,
            diagnostics,
            files,
            attributes,
            index_sort: None,
        };
        write_segment_info(&*dir, &name, &mut info).expect("write");
        let info2 = read_segment_info(&*dir, &name, &id).expect("read");
        assert_same(&info, &info2);
        dir.close().expect("close");
    }
}

/// `Map.put`: a repeated key replaces its value.
fn put(map: &mut Vec<(String, String)>, key: String, value: String) {
    match map.iter_mut().find(|(k, _)| *k == key) {
        Some(entry) => entry.1 = value,
        None => map.push((key, value)),
    }
}

/// `assertEquals(SegmentInfo, SegmentInfo)`.
pub fn assert_same(expected: &SegmentInfo, actual: &SegmentInfo) {
    assert_eq!(set(&expected.files), set(&actual.files));
    assert_eq!(
        sorted(expected.diagnostics.clone()),
        sorted(actual.diagnostics.clone())
    );
    assert_eq!(expected.doc_count, actual.doc_count);
    assert_eq!(expected.id, actual.id);
    assert_eq!(expected.is_compound_file, actual.is_compound_file);
    assert_eq!(expected.version, actual.version);
    assert_eq!(
        sorted(expected.attributes.clone()),
        sorted(actual.attributes.clone())
    );
}

/// Every `BaseSegmentInfoFormatTestCase` test this port runs.
pub const TESTS: &[SuiteTest] = &[
    ("testFiles", test_files),
    ("testHasBlocks", test_has_blocks),
    ("testAddsSelfToFiles", test_adds_self_to_files),
    ("testDiagnostics", test_diagnostics),
    ("testAttributes", test_attributes),
    ("testUniqueID", test_unique_id),
    ("testVersions", test_versions),
    ("testSort", test_sort),
    (
        "testExceptionOnCreateOutput",
        test_exception_on_create_output,
    ),
    ("testExceptionOnCloseOutput", test_exception_on_close_output),
    ("testExceptionOnOpenInput", test_exception_on_open_input),
    ("testRandom", test_random),
];
