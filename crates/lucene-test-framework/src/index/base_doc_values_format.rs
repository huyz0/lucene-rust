//! `BaseDocValuesFormatTestCase` and `BaseCompressingDocValuesFormatTestCase`:
//! the doc values skip index checks, `testMismatchedFields`, the
//! `intoBitSet` tests and the compression bounds, over
//! `Lucene90DocValuesFormat`. The rest of the suite is
//! [`super::legacy_base_doc_values_format`].
//!
//! Differs:
//!
//! - `DocIdSetIterator.intoBitSet` has no port; [`fill_bit_set`] is Java's
//!   default loop over `nextDoc`, and the expected bits come from
//!   `advanceExact` on each document, so the test still compares two ways
//!   of reaching the same documents.
//! - `CheckIndex.testDocValues` is
//!   [`super::legacy_base_doc_values_format::check_leaf_doc_values`].
//! - `testMismatchedFields` reaches a mismatched field numbering by indexing
//!   the source's fields in the other order (see
//!   [`super::base_postings_format`]) and copies with `addIndexes(Directory...)`.
//! - `@Nightly` `test*WithSkipperBig` take their size as a parameter
//!   ([`numeric_doc_values_with_skipper`] and its siblings), so a default
//!   run reaches the same code at a smaller size.
//! - `RandomPicks.randomFrom(random(), values)` is `values.get(nextInt(size))`,
//!   which is what it draws.

use super::base_index_file_format::SuiteTest;
use lucene_index::document::{
    BinaryDocValuesField, Document, IndexableField, NumericDocValuesField, SortedDocValuesField,
    SortedNumericDocValuesField, SortedSetDocValuesField,
};
use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_search::reader::DocValuesSkipper;
use lucene_search::reader::IndexReader as _;
use lucene_search::reader::{
    DocIdSetIterator, DocValuesIterator, LeafReader, SortedNumericDocValues, SortedSetDocValues,
    NO_MORE_DOCS,
};
use lucene_store::{ByteBuffersDirectory, Directory};
use lucene_util::java_random::JavaRandom;

use super::base_index_file_format::{
    force_merge, mock_analyzer, new_riw, only_leaf, plain_writer, riw_force_merge,
};
use super::legacy_base_doc_values_format::{
    add, advance, binary, check_leaf_doc_values, current_term, doc_of, exact, merged_away,
    merged_away_large, next, numeric, reader_then_close, sorted, sorted_numeric, sorted_set,
    values_of, DvTermsEnum,
};
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::test_util;

// --- merged-away fields --------------------------------------------------

/// The skip index of a field whose every value was merged away: no
/// documents, no values, exhausted at once.
fn assert_empty_skipper(leaf: &SegmentReader, field: &str) {
    let mut skipper = leaf
        .doc_values_skipper(field)
        .expect("getDocValuesSkipper")
        .unwrap_or_else(|| panic!("no skipper for {field}"));
    assert_eq!(0, skipper.global_doc_count());
    assert_eq!(0, skipper.max_value_count());
    skipper.advance(0);
    assert_eq!(NO_MORE_DOCS, skipper.min_doc_id(0));
}

/// The terms enum checks of a merged-away `SORTED`/`SORTED_SET` field.
fn assert_no_terms(mut te: DvTermsEnum<'_>) {
    assert!(!te.seek_exact_term(b"lucene"));
    assert_eq!(
        lucene_codecs::blocktree::SeekStatus::End,
        te.seek_ceil_term(b"lucene")
    );
}

/// `testSortedMergeAwayAllValuesWithSkipper` (and, with `large`, its
/// `LargeSegment` twin).
fn sorted_merge_away_with_skipper(t: &mut LuceneTestCase, large: bool) {
    let dir = ByteBuffersDirectory::new();
    let field = Box::new(SortedDocValuesField::indexed_field(
        "field",
        b"hello".to_vec(),
    ));
    let reader = if large {
        merged_away_large(t, &dir, field)
    } else {
        merged_away(t, &dir, field)
    };
    let leaf = only_leaf(&reader);
    let mut dv = sorted(leaf, "field");
    assert_eq!(NO_MORE_DOCS, next(dv.as_mut()));
    assert_empty_skipper(leaf, "field");
    assert_no_terms(DvTermsEnum::sorted(leaf, "field"));
    assert_eq!(-1, dv.lookup_term(b"lucene").expect("lookupTerm"));
}

/// `testSortedSetMergeAwayAllValuesWithSkipper` (and its `LargeSegment`
/// twin, which steps the iterator rather than counting values).
fn sorted_set_merge_away_with_skipper(t: &mut LuceneTestCase, large: bool) {
    let dir = ByteBuffersDirectory::new();
    let field = Box::new(SortedSetDocValuesField::indexed_field(
        "field",
        b"hello".to_vec(),
    ));
    let reader = if large {
        merged_away_large(t, &dir, field)
    } else {
        merged_away(t, &dir, field)
    };
    let leaf = only_leaf(&reader);
    let mut dv = sorted_set(leaf, "field");
    if large {
        assert_eq!(NO_MORE_DOCS, next(dv.as_mut()));
    } else {
        assert_eq!(0, dv.value_count());
    }
    assert_empty_skipper(leaf, "field");
    assert_no_terms(DvTermsEnum::sorted_set(leaf, "field"));
    assert_eq!(-1, dv.lookup_term(b"lucene").expect("lookupTerm"));
}

/// `testNumberMergeAwayAllValuesWithSkipper` and its `LargeSegment` twin.
fn number_merge_away_with_skipper(t: &mut LuceneTestCase, large: bool) {
    let dir = ByteBuffersDirectory::new();
    let reader = if large {
        merged_away_large(
            t,
            &dir,
            Box::new(NumericDocValuesField::indexed_field("field", 42)),
        )
    } else {
        merged_away(
            t,
            &dir,
            Box::new(NumericDocValuesField::indexed_field("field", 5)),
        )
    };
    let leaf = only_leaf(&reader);
    assert_eq!(NO_MORE_DOCS, next(numeric(leaf, "field").as_mut()));
    assert_empty_skipper(leaf, "field");
}

/// `testSortedNumberMergeAwayAllValuesWithSkipper` and its `LargeSegment`
/// twin.
fn sorted_number_merge_away_with_skipper(t: &mut LuceneTestCase, large: bool) {
    let dir = ByteBuffersDirectory::new();
    let reader = if large {
        merged_away_large(
            t,
            &dir,
            Box::new(SortedNumericDocValuesField::indexed_field("field", 42)),
        )
    } else {
        merged_away(
            t,
            &dir,
            Box::new(SortedNumericDocValuesField::indexed_field("field", 5)),
        )
    };
    let leaf = only_leaf(&reader);
    assert_eq!(NO_MORE_DOCS, next(sorted_numeric(leaf, "field").as_mut()));
    assert_empty_skipper(leaf, "field");
}

/// `testSortedMergeAwayAllValuesWithSkipper`.
pub fn test_sorted_merge_away_all_values_with_skipper(t: &mut LuceneTestCase) {
    sorted_merge_away_with_skipper(t, false);
}

/// `testSortedSetMergeAwayAllValuesWithSkipper`.
pub fn test_sorted_set_merge_away_all_values_with_skipper(t: &mut LuceneTestCase) {
    sorted_set_merge_away_with_skipper(t, false);
}

/// `testNumberMergeAwayAllValuesWithSkipper`.
pub fn test_number_merge_away_all_values_with_skipper(t: &mut LuceneTestCase) {
    number_merge_away_with_skipper(t, false);
}

/// `testSortedNumberMergeAwayAllValuesWithSkipper`.
pub fn test_sorted_number_merge_away_all_values_with_skipper(t: &mut LuceneTestCase) {
    sorted_number_merge_away_with_skipper(t, false);
}

/// `testSortedMergeAwayAllValuesLargeSegmentWithSkipper`.
pub fn test_sorted_merge_away_all_values_large_segment_with_skipper(t: &mut LuceneTestCase) {
    sorted_merge_away_with_skipper(t, true);
}

/// `testSortedSetMergeAwayAllValuesLargeSegmentWithSkipper`.
pub fn test_sorted_set_merge_away_all_values_large_segment_with_skipper(t: &mut LuceneTestCase) {
    sorted_set_merge_away_with_skipper(t, true);
}

/// `testNumericMergeAwayAllValuesLargeSegmentWithSkipper`.
pub fn test_numeric_merge_away_all_values_large_segment_with_skipper(t: &mut LuceneTestCase) {
    number_merge_away_with_skipper(t, true);
}

/// `testSortedNumericMergeAwayAllValuesLargeSegmentWithSkipper`.
pub fn test_sorted_numeric_merge_away_all_values_large_segment_with_skipper(
    t: &mut LuceneTestCase,
) {
    sorted_number_merge_away_with_skipper(t, true);
}

/// `testMaxValueCountWithSkipper`.
pub fn test_max_value_count_with_skipper(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let mut w = new_riw(t, &dir);
    add(
        &mut w,
        &doc_of(vec![
            Box::new(NumericDocValuesField::indexed_field("numeric", 1)),
            Box::new(SortedDocValuesField::indexed_field("sorted", b"a".to_vec())),
            Box::new(SortedNumericDocValuesField::indexed_field(
                "sorted_numeric",
                1,
            )),
            Box::new(SortedSetDocValuesField::indexed_field(
                "sorted_set",
                b"a".to_vec(),
            )),
        ]),
    );
    add(
        &mut w,
        &doc_of(vec![
            Box::new(NumericDocValuesField::indexed_field("numeric", 2)),
            Box::new(SortedDocValuesField::indexed_field("sorted", b"b".to_vec())),
            Box::new(SortedNumericDocValuesField::indexed_field(
                "sorted_numeric",
                1,
            )),
            Box::new(SortedNumericDocValuesField::indexed_field(
                "sorted_numeric",
                2,
            )),
            Box::new(SortedNumericDocValuesField::indexed_field(
                "sorted_numeric",
                3,
            )),
            Box::new(SortedSetDocValuesField::indexed_field(
                "sorted_set",
                b"a".to_vec(),
            )),
            Box::new(SortedSetDocValuesField::indexed_field(
                "sorted_set",
                b"b".to_vec(),
            )),
        ]),
    );
    riw_force_merge(&mut w, 1);
    let reader = reader_then_close(w);
    let leaf = only_leaf(&reader);
    for (field, count) in [
        ("numeric", 1),
        ("sorted", 1),
        ("sorted_numeric", 3),
        ("sorted_set", 2),
    ] {
        let skipper = leaf
            .doc_values_skipper(field)
            .expect("getDocValuesSkipper")
            .expect("a skipper");
        assert_eq!(count, skipper.max_value_count(), "{field}");
    }
}

// --- the random skipper tests --------------------------------------------

/// The four doc-value kinds the skipper tests index under `test`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipperKind {
    /// `NumericDocValuesField.indexedField`.
    Numeric,
    /// `SortedNumericDocValuesField.indexedField`, one to four values.
    SortedNumeric,
    /// `SortedDocValuesField.indexedField` of a random binary term.
    Sorted,
    /// `SortedSetDocValuesField.indexedField`, one to four random terms.
    SortedSet,
}

impl SkipperKind {
    /// `populateDoc(doc)`.
    fn populate_doc(self, r: &mut JavaRandom, doc: &mut Document) {
        match self {
            SkipperKind::Numeric => {
                doc.add(NumericDocValuesField::indexed_field("test", r.next_long()));
            }
            SkipperKind::SortedNumeric => {
                // Java re-draws the bound on every iteration.
                let mut j = 0;
                while j < test_util::next_int_in(r, 1, 5) {
                    doc.add(SortedNumericDocValuesField::indexed_field(
                        "test",
                        r.next_long(),
                    ));
                    j += 1;
                }
            }
            SkipperKind::Sorted => {
                let term = test_util::random_binary_term(r);
                doc.add(SortedDocValuesField::indexed_field("test", term));
            }
            SkipperKind::SortedSet => {
                let mut j = 0;
                while j < test_util::next_int_in(r, 1, 5) {
                    let term = test_util::random_binary_term(r);
                    doc.add(SortedSetDocValuesField::indexed_field("test", term));
                    j += 1;
                }
            }
        }
    }

    /// `docValuesWrapper(leafReader)`.
    fn wrapper<'a>(self, leaf: &'a SegmentReader) -> DocValuesWrapper<'a> {
        let values = match self {
            SkipperKind::Numeric => Values::Numeric(numeric(leaf, "test")),
            SkipperKind::SortedNumeric => Values::SortedNumeric(sorted_numeric(leaf, "test")),
            SkipperKind::Sorted => Values::Sorted(sorted(leaf, "test")),
            SkipperKind::SortedSet => Values::SortedSet(sorted_set(leaf, "test")),
        };
        DocValuesWrapper {
            values,
            min: i64::MAX,
            max: i64::MIN,
            count: 0,
        }
    }
}

enum Values<'a> {
    Numeric(Box<dyn lucene_search::reader::NumericDocValues + 'a>),
    SortedNumeric(Box<dyn SortedNumericDocValues + 'a>),
    Sorted(Box<dyn lucene_search::reader::SortedDocValues + 'a>),
    SortedSet(Box<dyn SortedSetDocValues + 'a>),
}

/// `DocValuesWrapper`: an iterator over one leaf's `test` values that
/// knows each document's smallest and largest value (an ordinal for the
/// sorted kinds) and its value count.
pub struct DocValuesWrapper<'a> {
    values: Values<'a>,
    min: i64,
    max: i64,
    count: i32,
}

impl DocValuesWrapper<'_> {
    fn iterator(&mut self) -> &mut dyn DocValuesIterator {
        match &mut self.values {
            Values::Numeric(v) => v.as_mut(),
            Values::SortedNumeric(v) => v.as_mut(),
            Values::Sorted(v) => v.as_mut(),
            Values::SortedSet(v) => v.as_mut(),
        }
    }

    /// `readValues()`, or the single value of a one-valued kind.
    fn read_values(&mut self) {
        let values: Vec<i64> = match &mut self.values {
            Values::Numeric(v) => vec![v.long_value()],
            Values::Sorted(v) => vec![i64::from(v.ord_value())],
            Values::SortedNumeric(v) => values_of(v.as_mut()),
            Values::SortedSet(v) => super::legacy_base_doc_values_format::ords_of(v.as_mut()),
        };
        self.count = values.len() as i32;
        self.min = values.iter().copied().min().unwrap_or(i64::MAX);
        self.max = values.iter().copied().max().unwrap_or(i64::MIN);
    }

    /// `advance(target)`.
    pub fn advance(&mut self, target: i32) -> i32 {
        let doc = advance(self.iterator(), target);
        if doc != NO_MORE_DOCS {
            self.read_values();
        }
        doc
    }

    /// `advanceExact(target)`.
    pub fn advance_exact(&mut self, target: i32) -> bool {
        if exact(self.iterator(), target) {
            self.read_values();
            true
        } else {
            false
        }
    }

    /// `docID()`.
    pub fn doc_id(&mut self) -> i32 {
        self.iterator().doc_id()
    }
}

/// `assertDocValuesWithSkipper(totalDocs, kind)`.
pub fn assert_doc_values_with_skipper(t: &mut LuceneTestCase, total_docs: i32, kind: SkipperKind) {
    let supplier = t.random().next_int_bounded(3);
    let dir = ByteBuffersDirectory::new();
    let mut w = new_riw(t, &dir);
    let mut num_docs = 0;
    for _ in 0..total_docs {
        let mut doc = Document::new();
        let populate = match supplier {
            0 => true,
            1 => t.random().next_boolean(),
            _ => t.random().next_boolean() && t.random().next_boolean(),
        };
        if populate {
            kind.populate_doc(t.random(), &mut doc);
            num_docs += 1;
        }
        add(&mut w, &doc);
        if t.rarely() {
            w.commit().expect("commit");
        }
    }
    w.flush().expect("flush");
    if t.random().next_boolean() {
        riw_force_merge(&mut w, 1);
    }
    let reader = w.get_reader().expect("getReader");
    let mut read_docs = 0;
    for leaf in reader.segment_readers() {
        check_leaf_doc_values(leaf).expect("testDocValues");
        let skipper = leaf
            .doc_values_skipper("test")
            .expect("getDocValuesSkipper");
        if skipper.is_none() {
            // A segment none of whose documents has the field: Java's
            // wrapper holds a null iterator its null-skipper checks never
            // touch, and neither check draws.
            continue;
        }
        read_docs += assert_doc_values_skip_sequential(&mut kind.wrapper(leaf), skipper);
        for _ in 0..10 {
            let skipper = leaf
                .doc_values_skipper("test")
                .expect("getDocValuesSkipper");
            assert_doc_values_skip_random(
                t.random(),
                &mut kind.wrapper(leaf),
                skipper,
                leaf.max_doc(),
            );
        }
    }
    assert_eq!(num_docs, read_docs);
    drop(reader);
    w.close().expect("close");
}

/// `assertDocValuesSkipSequential(iterator, skipper)`: every level-0
/// interval against the documents it covers. The documents counted.
pub fn assert_doc_values_skip_sequential(
    iterator: &mut DocValuesWrapper<'_>,
    skipper: Option<DocValuesSkipper<'_>>,
) -> i32 {
    let Some(mut skipper) = skipper else {
        return 0;
    };
    assert_eq!(-1, iterator.doc_id());
    assert_eq!(-1, skipper.min_doc_id(0));
    assert_eq!(-1, skipper.max_doc_id(0));

    iterator.advance(0);
    let mut doc_count = 0;
    let mut max_value_count = 0;
    loop {
        let previous_max_doc = skipper.max_doc_id(0);
        skipper.advance(previous_max_doc + 1);
        assert!(skipper.min_doc_id(0) > previous_max_doc);
        assert_eq!(iterator.doc_id(), skipper.min_doc_id(0));

        if skipper.min_doc_id(0) == NO_MORE_DOCS {
            assert_eq!(NO_MORE_DOCS, skipper.max_doc_id(0));
            break;
        }
        assert!(skipper.doc_count(0) > 0);

        let mut max_doc = -1;
        let mut min_val = i64::MAX;
        let mut max_val = i64::MIN;
        for _ in 0..skipper.doc_count(0) {
            let doc = iterator.doc_id();
            assert_ne!(NO_MORE_DOCS, doc);
            max_doc = max_doc.max(doc);
            min_val = min_val.min(iterator.min);
            max_val = max_val.max(iterator.max);
            max_value_count = max_value_count.max(iterator.count);
            iterator.advance(doc + 1);
        }
        assert_eq!(max_doc, skipper.max_doc_id(0));
        assert_eq!(min_val, skipper.min_value(0));
        assert_eq!(max_val, skipper.max_value(0));
        doc_count += skipper.doc_count(0);
        for level in 1..skipper.num_levels() {
            assert!(skipper.min_doc_id(0) >= skipper.min_doc_id(level));
            assert!(skipper.max_doc_id(0) <= skipper.max_doc_id(level));
            assert!(skipper.min_value(0) >= skipper.min_value(level));
            assert!(skipper.max_value(0) <= skipper.max_value(level));
            assert!(skipper.doc_count(0) < skipper.doc_count(level));
        }
    }
    assert_eq!(doc_count, skipper.global_doc_count());
    assert_eq!(max_value_count, skipper.max_value_count());
    doc_count
}

/// `assertDocValuesSkipRandom(iterator, skipper, maxDoc)`: random jumps,
/// each document found inside every level's interval.
pub fn assert_doc_values_skip_random(
    r: &mut JavaRandom,
    iterator: &mut DocValuesWrapper<'_>,
    skipper: Option<DocValuesSkipper<'_>>,
    max_doc: i32,
) {
    let Some(mut skipper) = skipper else {
        return;
    };
    let mut next_level = 0;
    loop {
        let doc = test_util::next_int_in(r, skipper.max_doc_id(next_level), max_doc + 1) + 1;
        skipper.advance(doc);
        if skipper.min_doc_id(0) == NO_MORE_DOCS {
            assert_eq!(NO_MORE_DOCS, skipper.max_doc_id(0));
            return;
        }
        if iterator.advance_exact(doc) {
            for level in 0..skipper.num_levels() {
                assert!(iterator.doc_id() >= skipper.min_doc_id(level));
                assert!(iterator.doc_id() <= skipper.max_doc_id(level));
                assert!(iterator.min >= skipper.min_value(level));
                assert!(iterator.max <= skipper.max_value(level));
            }
        }
        next_level = r.next_int_bounded(skipper.num_levels() as i32) as usize;
    }
}

/// `doTestNumericDocValuesWithSkipper(totalDocs)`.
pub fn numeric_doc_values_with_skipper(t: &mut LuceneTestCase, total_docs: i32) {
    assert_doc_values_with_skipper(t, total_docs, SkipperKind::Numeric);
}

/// `doTestSortedNumericDocValuesWithSkipper(totalDocs)`.
pub fn sorted_numeric_doc_values_with_skipper(t: &mut LuceneTestCase, total_docs: i32) {
    assert_doc_values_with_skipper(t, total_docs, SkipperKind::SortedNumeric);
}

/// `doTestSortedDocValuesWithSkipper(totalDocs)`.
pub fn sorted_doc_values_with_skipper(t: &mut LuceneTestCase, total_docs: i32) {
    assert_doc_values_with_skipper(t, total_docs, SkipperKind::Sorted);
}

/// `doTestSortedSetDocValuesWithSkipper(totalDocs)`.
pub fn sorted_set_doc_values_with_skipper(t: &mut LuceneTestCase, total_docs: i32) {
    assert_doc_values_with_skipper(t, total_docs, SkipperKind::SortedSet);
}

/// A size drawn by `random().nextInt(origin, bound)`, then `kind`'s test.
fn sized(t: &mut LuceneTestCase, origin: i32, bound: i32, kind: SkipperKind) {
    let total_docs = test_util::next_int_in(t.random(), origin, bound);
    assert_doc_values_with_skipper(t, total_docs, kind);
}

/// `testNumericDocValuesWithSkipperSmall`.
pub fn test_numeric_doc_values_with_skipper_small(t: &mut LuceneTestCase) {
    sized(t, 1, 1000, SkipperKind::Numeric);
}

/// `testNumericDocValuesWithSkipperMedium`.
pub fn test_numeric_doc_values_with_skipper_medium(t: &mut LuceneTestCase) {
    sized(t, 1000, 20_000, SkipperKind::Numeric);
}

/// `@Nightly testNumericDocValuesWithSkipperBig`.
pub fn test_numeric_doc_values_with_skipper_big(t: &mut LuceneTestCase) {
    sized(t, 50_000, 100_000, SkipperKind::Numeric);
}

/// `testSortedNumericDocValuesWithSkipperSmall`.
pub fn test_sorted_numeric_doc_values_with_skipper_small(t: &mut LuceneTestCase) {
    sized(t, 1, 1000, SkipperKind::SortedNumeric);
}

/// `testSortedNumericDocValuesWithSkipperMedium`.
pub fn test_sorted_numeric_doc_values_with_skipper_medium(t: &mut LuceneTestCase) {
    sized(t, 1000, 20_000, SkipperKind::SortedNumeric);
}

/// `@Nightly testSortedNumericDocValuesWithSkipperBig`.
pub fn test_sorted_numeric_doc_values_with_skipper_big(t: &mut LuceneTestCase) {
    sized(t, 50_000, 100_000, SkipperKind::SortedNumeric);
}

/// `testSortedDocValuesWithSkipperSmall`.
pub fn test_sorted_doc_values_with_skipper_small(t: &mut LuceneTestCase) {
    sized(t, 1, 1000, SkipperKind::Sorted);
}

/// `testSortedDocValuesWithSkipperMedium`.
pub fn test_sorted_doc_values_with_skipper_medium(t: &mut LuceneTestCase) {
    sized(t, 1000, 20_000, SkipperKind::Sorted);
}

/// `@Nightly testSortedDocValuesWithSkipperBig`.
pub fn test_sorted_doc_values_with_skipper_big(t: &mut LuceneTestCase) {
    sized(t, 50_000, 100_000, SkipperKind::Sorted);
}

/// `testSortedSetDocValuesWithSkipperSmall`.
pub fn test_sorted_set_doc_values_with_skipper_small(t: &mut LuceneTestCase) {
    sized(t, 1, 1000, SkipperKind::SortedSet);
}

/// `testSortedSetDocValuesWithSkipperMedium`.
pub fn test_sorted_set_doc_values_with_skipper_medium(t: &mut LuceneTestCase) {
    sized(t, 10_000, 20_000, SkipperKind::SortedSet);
}

/// `@Nightly testSortedSetDocValuesWithSkipperBig`.
pub fn test_sorted_set_doc_values_with_skipper_big(t: &mut LuceneTestCase) {
    sized(t, 50_000, 100_000, SkipperKind::SortedSet);
}

// --- mismatched fields -----------------------------------------------------

/// The five fields `testMismatchedFields` indexes.
fn mismatched_fields_doc(reverse: bool) -> Document {
    let mut fields: Vec<Box<dyn IndexableField>> = vec![
        Box::new(BinaryDocValuesField::new("binary", b"lucene".to_vec())),
        Box::new(NumericDocValuesField::new("numeric", 0)),
        Box::new(SortedDocValuesField::new("sorted", b"search".to_vec())),
        Box::new(SortedNumericDocValuesField::new("sorted_numeric", 1)),
        Box::new(SortedSetDocValuesField::new(
            "sorted_set",
            b"engine".to_vec(),
        )),
    ];
    if reverse {
        fields.reverse();
    }
    doc_of(fields)
}

/// `testMismatchedFields`.
pub fn test_mismatched_fields(_t: &mut LuceneTestCase) {
    let dir1 = ByteBuffersDirectory::new();
    let mut w1 = plain_writer(&dir1);
    // `MismatchedCodecReader` renumbers the fields; adding them in the
    // other order gives the source the other numbering.
    w1.add_fields_document(&mismatched_fields_doc(true))
        .expect("addDocument");
    w1.commit().expect("commit");
    w1.close().expect("close");

    let dir2 = ByteBuffersDirectory::new();
    let mut w2 = plain_writer(&dir2);
    w2.add_fields_document(&mismatched_fields_doc(false))
        .expect("addDocument");
    w2.commit().expect("commit");
    w2.add_indexes(&[&dir1]).expect("addIndexes");
    force_merge(&mut w2, 1);
    w2.close().expect("close");

    let reader = DirectoryReader::open(&dir2).expect("open");
    let leaf = only_leaf(&reader);

    let mut bdv = binary(leaf, "binary");
    for doc in 0..2 {
        assert_eq!(doc, next(bdv.as_mut()));
        assert_eq!(b"lucene", bdv.binary_value());
    }
    assert_eq!(NO_MORE_DOCS, next(bdv.as_mut()));

    let mut ndv = numeric(leaf, "numeric");
    for doc in 0..2 {
        assert_eq!(doc, next(ndv.as_mut()));
        assert_eq!(0, ndv.long_value());
    }
    assert_eq!(NO_MORE_DOCS, next(ndv.as_mut()));

    let mut sdv = sorted(leaf, "sorted");
    for doc in 0..2 {
        assert_eq!(doc, next(sdv.as_mut()));
        assert_eq!(b"search".to_vec(), current_term(sdv.as_mut()));
    }
    assert_eq!(NO_MORE_DOCS, next(sdv.as_mut()));

    let mut sndv = sorted_numeric(leaf, "sorted_numeric");
    for doc in 0..2 {
        assert_eq!(doc, next(sndv.as_mut()));
        assert_eq!(1, sndv.next_value().expect("nextValue"));
    }
    assert_eq!(NO_MORE_DOCS, next(sndv.as_mut()));

    let mut ssdv = sorted_set(leaf, "sorted_set");
    for doc in 0..2 {
        assert_eq!(doc, next(ssdv.as_mut()));
        let ord = ssdv.next_ord().expect("nextOrd");
        assert_eq!(b"engine".to_vec(), ssdv.lookup_ord(ord).expect("lookupOrd"));
    }
    assert_eq!(NO_MORE_DOCS, next(ssdv.as_mut()));
}

// --- intoBitSet ------------------------------------------------------------

/// `DocIdSetIterator.intoBitSet(upTo, bitSet, offset)`, Java's default:
/// every document from the current one below `up_to`, the iterator left on
/// the first at or past it.
pub fn fill_bit_set<I: DocIdSetIterator + ?Sized>(
    it: &mut I,
    up_to: i32,
    bits: &mut [bool],
    offset: i32,
) {
    let mut doc = it.doc_id();
    while doc < up_to {
        bits[(doc - offset) as usize] = true;
        doc = next(it);
    }
}

/// The kinds `doTestRandomIntoBitSet` reads back, each under `num`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitSetKind {
    /// One numeric value.
    Numeric,
    /// Two sorted-numeric values.
    SortedNumeric,
    /// One sorted value, a long's decimal string.
    Sorted,
    /// Two sorted-set values.
    SortedSet,
    /// Ten random bytes.
    Binary,
}

impl BitSetKind {
    fn consume(self, r: &mut JavaRandom, doc: &mut Document) {
        match self {
            BitSetKind::Numeric => doc.add(NumericDocValuesField::new("num", r.next_long())),
            BitSetKind::SortedNumeric => {
                for _ in 0..2 {
                    doc.add(SortedNumericDocValuesField::new("num", r.next_long()));
                }
            }
            BitSetKind::Sorted => {
                let v = r.next_long().to_string().into_bytes();
                doc.add(SortedDocValuesField::new("num", v));
            }
            BitSetKind::SortedSet => {
                for _ in 0..2 {
                    let v = r.next_long().to_string().into_bytes();
                    doc.add(SortedSetDocValuesField::new("num", v));
                }
            }
            BitSetKind::Binary => {
                let mut bytes = vec![0u8; 10];
                r.next_bytes(&mut bytes);
                doc.add(BinaryDocValuesField::new("num", bytes));
            }
        }
    }

    fn produce<'a>(self, leaf: &'a SegmentReader) -> Option<Box<dyn DocValuesIterator + 'a>> {
        let e = "doc values";
        match self {
            BitSetKind::Numeric => leaf
                .numeric_doc_values("num")
                .expect(e)
                .map(|v| v as Box<dyn DocValuesIterator>),
            BitSetKind::SortedNumeric => leaf
                .sorted_numeric_doc_values("num")
                .expect(e)
                .map(|v| v as Box<dyn DocValuesIterator>),
            BitSetKind::Sorted => leaf
                .sorted_doc_values("num")
                .expect(e)
                .map(|v| v as Box<dyn DocValuesIterator>),
            BitSetKind::SortedSet => leaf
                .sorted_set_doc_values("num")
                .expect(e)
                .map(|v| v as Box<dyn DocValuesIterator>),
            BitSetKind::Binary => leaf
                .binary_doc_values("num")
                .expect(e)
                .map(|v| v as Box<dyn DocValuesIterator>),
        }
    }
}

/// `doTestRandomIntoBitSet(consumer, producer, addDocument)`, the
/// `addDocument` supplier being `add`.
pub fn random_into_bit_set(
    t: &mut LuceneTestCase,
    kind: BitSetKind,
    add_document: &mut dyn FnMut(&mut JavaRandom) -> bool,
) {
    let num_docs = t.at_least(100);
    let mut docs_with_field = 0;
    let dir = ByteBuffersDirectory::new();
    let mut w = plain_writer(&dir);
    for _ in 0..num_docs {
        let mut doc = Document::new();
        if add_document(t.random()) {
            docs_with_field += 1;
            kind.consume(t.random(), &mut doc);
        }
        w.add_fields_document(&doc).expect("addDocument");
    }
    force_merge(&mut w, 1);
    w.close().expect("close");
    let reader = DirectoryReader::open(&dir).expect("open");
    let leaf = only_leaf(&reader);
    for _ in 0..20 {
        let start = t.random().next_int_bounded(num_docs - 1);
        let up_to = test_util::next_int_in(t.random(), start, num_docs);
        let offset = if start == 0 {
            0
        } else {
            t.random().next_int_bounded(start)
        };
        let len = (num_docs - offset) as usize;
        let mut bits = vec![false; len];
        let mut expected_bits = vec![false; len];

        let values = kind.produce(leaf);
        let expected = kind.produce(leaf);
        if docs_with_field == 0 {
            assert!(values.is_none());
            return; // no more to be tested
        }
        let (mut values, mut expected) = (values.expect("values"), expected.expect("values"));
        advance(values.as_mut(), start);
        if values.doc_id() == NO_MORE_DOCS {
            assert_eq!(NO_MORE_DOCS, advance(expected.as_mut(), start));
            continue;
        }
        fill_bit_set(values.as_mut(), up_to, &mut bits, offset);
        // The expected bits by `advanceExact`, document by document.
        for doc in start..up_to {
            if exact(expected.as_mut(), doc) {
                expected_bits[(doc - offset) as usize] = true;
            }
        }
        let after = kind.produce(leaf).map(|mut it| advance(it.as_mut(), up_to));
        assert_eq!(after, Some(values.doc_id()));
        assert_eq!(expected_bits, bits);
    }
}

/// `random().nextInt(n) == 0` for a fixed `n`, `nextInt(50) + 1` drawn
/// first.
fn one_in_n(t: &mut LuceneTestCase) -> impl FnMut(&mut JavaRandom) -> bool {
    let n = t.random().next_int_bounded(50) + 1;
    move |r| r.next_int_bounded(n) == 0
}

/// `testRandomDenseNumericIntoBitSet`.
pub fn test_random_dense_numeric_into_bit_set(t: &mut LuceneTestCase) {
    random_into_bit_set(t, BitSetKind::Numeric, &mut |_| true);
}

/// `testRandomSparseNumericIntoBitSet`.
pub fn test_random_sparse_numeric_into_bit_set(t: &mut LuceneTestCase) {
    random_into_bit_set(t, BitSetKind::Numeric, &mut |r| {
        if r.next_boolean() {
            r.next_boolean()
        } else {
            r.next_int_bounded(50) == 0
        }
    });
}

/// `testRandomDenseSortedNumericIntoBitSet`.
pub fn test_random_dense_sorted_numeric_into_bit_set(t: &mut LuceneTestCase) {
    random_into_bit_set(t, BitSetKind::SortedNumeric, &mut |_| true);
}

/// `testRandomSparseSortedNumericIntoBitSet`.
pub fn test_random_sparse_sorted_numeric_into_bit_set(t: &mut LuceneTestCase) {
    let mut f = one_in_n(t);
    random_into_bit_set(t, BitSetKind::SortedNumeric, &mut f);
}

/// `testRandomDenseSortedIntoBitSet`.
pub fn test_random_dense_sorted_into_bit_set(t: &mut LuceneTestCase) {
    random_into_bit_set(t, BitSetKind::Sorted, &mut |_| true);
}

/// `testRandomSparseSortedIntoBitSet`.
pub fn test_random_sparse_sorted_into_bit_set(t: &mut LuceneTestCase) {
    let mut f = one_in_n(t);
    random_into_bit_set(t, BitSetKind::Sorted, &mut f);
}

/// `testRandomDenseSortedSetIntoBitSet`.
pub fn test_random_dense_sorted_set_into_bit_set(t: &mut LuceneTestCase) {
    random_into_bit_set(t, BitSetKind::SortedSet, &mut |_| true);
}

/// `testRandomSparseSortedSetIntoBitSet`.
pub fn test_random_sparse_sorted_set_into_bit_set(t: &mut LuceneTestCase) {
    let mut f = one_in_n(t);
    random_into_bit_set(t, BitSetKind::SortedSet, &mut f);
}

/// `testRandomDenseBinaryIntoBitSet`.
pub fn test_random_dense_binary_into_bit_set(t: &mut LuceneTestCase) {
    random_into_bit_set(t, BitSetKind::Binary, &mut |_| true);
}

/// `testRandomSparseBinaryIntoBitSet`.
pub fn test_random_sparse_binary_into_bit_set(t: &mut LuceneTestCase) {
    let mut f = one_in_n(t);
    random_into_bit_set(t, BitSetKind::Binary, &mut f);
}

// --- BaseCompressingDocValuesFormatTestCase -------------------------------

/// `dirSize(d)`: the bytes of every file.
pub fn dir_size(d: &dyn Directory) -> u64 {
    d.list_all()
        .expect("listAll")
        .iter()
        .map(|f| d.file_length(f).expect("fileLength"))
        .sum()
}

/// Index `values` as `dv` (one document each), merge to one segment, and
/// measure the directory.
fn add_numbers(
    w: &mut lucene_index::index_writer::IndexWriter<'_>,
    dir: &dyn Directory,
    values: &[i64],
) -> u64 {
    for &v in values {
        w.add_fields_document(&doc_of(vec![Box::new(NumericDocValuesField::new("dv", v))]))
            .expect("addDocument");
    }
    force_merge(w, 1);
    dir_size(dir)
}

/// `testUniqueValuesCompression`.
pub fn test_unique_values_compression(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let _analyzer = mock_analyzer(t);
    let mut w = plain_writer(&dir);
    let unique_value_count = test_util::next_int(t.random(), 1, 256);
    let mut values: Vec<i64> = Vec::new();
    let mut first = Vec::new();
    for _ in 0..300 {
        let value = if (values.len() as i32) < unique_value_count {
            let v = t.random().next_long();
            values.push(v);
            v
        } else {
            values[t.random().next_int_bounded(values.len() as i32) as usize]
        };
        first.push(value);
    }
    let size1 = add_numbers(&mut w, &dir, &first);
    let second: Vec<i64> = (0..20)
        .map(|_| values[t.random().next_int_bounded(values.len() as i32) as usize])
        .collect();
    let size2 = add_numbers(&mut w, &dir, &second);
    // make sure the new longs did not cost 8 bytes each
    assert!(size2 < size1 + 8 * 20, "{size2} >= {size1} + 160");
    w.close().expect("close");
}

/// `PackedInts.bitsRequired(1000L * 60 * 60 * 24)`.
const DAY_BITS: u64 = 27;

/// `testDateCompression`.
pub fn test_date_compression(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let _analyzer = mock_analyzer(t);
    let mut w = plain_writer(&dir);
    let base = 13; // prime
    let day = 1000i64 * 60 * 60 * 24;
    assert_eq!(DAY_BITS, u64::from(64 - day.leading_zeros()));
    let first: Vec<i64> = (0..300)
        .map(|_| base + i64::from(t.random().next_int_bounded(1000)) * day)
        .collect();
    let size1 = add_numbers(&mut w, &dir, &first);
    let second: Vec<i64> = (0..50)
        .map(|_| base + i64::from(t.random().next_int_bounded(1000)) * day)
        .collect();
    let size2 = add_numbers(&mut w, &dir, &second);
    // make sure the new longs costed less than if they had only been packed
    assert!(size2 < size1 + (DAY_BITS * 50) / 8, "{size2} vs {size1}");
    w.close().expect("close");
}

/// `testSingleBigValueCompression`.
pub fn test_single_big_value_compression(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let _analyzer = mock_analyzer(t);
    let mut w = plain_writer(&dir);
    let first: Vec<i64> = (0..20_000).map(|i| i & 1023).collect();
    let size1 = add_numbers(&mut w, &dir, &first);
    let size2 = add_numbers(&mut w, &dir, &[i64::MAX]);
    // make sure the new value did not grow the bpv for every other value
    assert!(
        size2 < size1 + (20_000 * (63 - 10)) / 8,
        "{size2} vs {size1}"
    );
    w.close().expect("close");
}

/// Every test of `BaseDocValuesFormatTestCase` itself, with its name. The
/// `@Nightly` four are last.
pub const TESTS: &[SuiteTest] = &[
    (
        "testSortedMergeAwayAllValuesWithSkipper",
        test_sorted_merge_away_all_values_with_skipper,
    ),
    (
        "testSortedSetMergeAwayAllValuesWithSkipper",
        test_sorted_set_merge_away_all_values_with_skipper,
    ),
    (
        "testNumberMergeAwayAllValuesWithSkipper",
        test_number_merge_away_all_values_with_skipper,
    ),
    (
        "testSortedNumberMergeAwayAllValuesWithSkipper",
        test_sorted_number_merge_away_all_values_with_skipper,
    ),
    (
        "testSortedMergeAwayAllValuesLargeSegmentWithSkipper",
        test_sorted_merge_away_all_values_large_segment_with_skipper,
    ),
    (
        "testSortedSetMergeAwayAllValuesLargeSegmentWithSkipper",
        test_sorted_set_merge_away_all_values_large_segment_with_skipper,
    ),
    (
        "testNumericMergeAwayAllValuesLargeSegmentWithSkipper",
        test_numeric_merge_away_all_values_large_segment_with_skipper,
    ),
    (
        "testSortedNumericMergeAwayAllValuesLargeSegmentWithSkipper",
        test_sorted_numeric_merge_away_all_values_large_segment_with_skipper,
    ),
    (
        "testMaxValueCountWithSkipper",
        test_max_value_count_with_skipper,
    ),
    (
        "testNumericDocValuesWithSkipperSmall",
        test_numeric_doc_values_with_skipper_small,
    ),
    (
        "testNumericDocValuesWithSkipperMedium",
        test_numeric_doc_values_with_skipper_medium,
    ),
    (
        "testSortedNumericDocValuesWithSkipperSmall",
        test_sorted_numeric_doc_values_with_skipper_small,
    ),
    (
        "testSortedNumericDocValuesWithSkipperMedium",
        test_sorted_numeric_doc_values_with_skipper_medium,
    ),
    (
        "testSortedDocValuesWithSkipperSmall",
        test_sorted_doc_values_with_skipper_small,
    ),
    (
        "testSortedDocValuesWithSkipperMedium",
        test_sorted_doc_values_with_skipper_medium,
    ),
    (
        "testSortedSetDocValuesWithSkipperSmall",
        test_sorted_set_doc_values_with_skipper_small,
    ),
    (
        "testSortedSetDocValuesWithSkipperMedium",
        test_sorted_set_doc_values_with_skipper_medium,
    ),
    ("testMismatchedFields", test_mismatched_fields),
    (
        "testRandomDenseNumericIntoBitSet",
        test_random_dense_numeric_into_bit_set,
    ),
    (
        "testRandomSparseNumericIntoBitSet",
        test_random_sparse_numeric_into_bit_set,
    ),
    (
        "testRandomDenseSortedNumericIntoBitSet",
        test_random_dense_sorted_numeric_into_bit_set,
    ),
    (
        "testRandomSparseSortedNumericIntoBitSet",
        test_random_sparse_sorted_numeric_into_bit_set,
    ),
    (
        "testRandomDenseSortedIntoBitSet",
        test_random_dense_sorted_into_bit_set,
    ),
    (
        "testRandomSparseSortedIntoBitSet",
        test_random_sparse_sorted_into_bit_set,
    ),
    (
        "testRandomDenseSortedSetIntoBitSet",
        test_random_dense_sorted_set_into_bit_set,
    ),
    (
        "testRandomSparseSortedSetIntoBitSet",
        test_random_sparse_sorted_set_into_bit_set,
    ),
    (
        "testRandomDenseBinaryIntoBitSet",
        test_random_dense_binary_into_bit_set,
    ),
    (
        "testRandomSparseBinaryIntoBitSet",
        test_random_sparse_binary_into_bit_set,
    ),
    (
        "testNumericDocValuesWithSkipperBig",
        test_numeric_doc_values_with_skipper_big,
    ),
    (
        "testSortedNumericDocValuesWithSkipperBig",
        test_sorted_numeric_doc_values_with_skipper_big,
    ),
    (
        "testSortedDocValuesWithSkipperBig",
        test_sorted_doc_values_with_skipper_big,
    ),
    (
        "testSortedSetDocValuesWithSkipperBig",
        test_sorted_set_doc_values_with_skipper_big,
    ),
];

/// `BaseCompressingDocValuesFormatTestCase`'s own tests.
pub const COMPRESSING_TESTS: &[SuiteTest] = &[
    (
        "testUniqueValuesCompression",
        test_unique_values_compression,
    ),
    ("testDateCompression", test_date_compression),
    (
        "testSingleBigValueCompression",
        test_single_big_value_compression,
    ),
];

/// The `@Nightly` tests, at the end of [`TESTS`].
pub const NIGHTLY: &[&str] = &[
    "testNumericDocValuesWithSkipperBig",
    "testSortedNumericDocValuesWithSkipperBig",
    "testSortedDocValuesWithSkipperBig",
    "testSortedSetDocValuesWithSkipperBig",
];

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_index::document::{Store, StringField};

    #[test]
    fn every_java_test_is_listed_once() {
        let names: std::collections::BTreeSet<&str> = TESTS.iter().map(|(n, _)| *n).collect();
        assert_eq!(names.len(), TESTS.len());
        assert_eq!(32, TESTS.len());
        assert_eq!(3, COMPRESSING_TESTS.len());
        for n in NIGHTLY {
            assert!(names.contains(n));
        }
    }

    #[test]
    fn into_bit_set_stops_at_up_to() {
        struct Docs(Vec<i32>, usize);
        impl DocIdSetIterator for Docs {
            fn doc_id(&self) -> i32 {
                if self.1 == 0 {
                    -1
                } else {
                    self.0.get(self.1 - 1).copied().unwrap_or(NO_MORE_DOCS)
                }
            }
            fn next_doc(&mut self) -> lucene_search::Result<i32> {
                self.1 += 1;
                Ok(self.doc_id())
            }
            fn advance(&mut self, target: i32) -> lucene_search::Result<i32> {
                while self.doc_id() < target {
                    self.next_doc()?;
                }
                Ok(self.doc_id())
            }
            fn cost(&self) -> i64 {
                self.0.len() as i64
            }
        }
        let mut it = Docs(vec![1, 3, 6], 0);
        advance(&mut it, 2);
        let mut bits = vec![false; 5];
        fill_bit_set(&mut it, 6, &mut bits, 1);
        assert_eq!(vec![false, false, true, false, false], bits);
        assert_eq!(6, it.doc_id());
    }

    #[test]
    fn a_leaf_without_a_skip_index_checks_nothing() {
        let dir = ByteBuffersDirectory::new();
        let mut w = plain_writer(&dir);
        w.add_fields_document(&doc_of(vec![Box::new(NumericDocValuesField::new(
            "test", 1,
        ))]))
        .expect("addDocument");
        w.add_fields_document(&doc_of(vec![Box::new(StringField::new(
            "id",
            "1",
            Store::No,
        ))]))
        .expect("addDocument");
        w.close().expect("close");
        let reader = DirectoryReader::open(&dir).expect("open");
        let leaf = only_leaf(&reader);
        assert!(leaf.doc_values_skipper("test").expect("skipper").is_none());
        let mut it = SkipperKind::Numeric.wrapper(leaf);
        assert_eq!(0, assert_doc_values_skip_sequential(&mut it, None));
        assert_doc_values_skip_random(&mut JavaRandom::new(1), &mut it, None, 2);
        // A document without a value: `advanceExact` reads nothing.
        assert!(!it.advance_exact(1));
        assert_eq!(NO_MORE_DOCS, it.advance(1));
    }

    #[test]
    fn no_documents_with_the_field_means_no_iterator() {
        let mut t = LuceneTestCase::with_config("noField", 3, false, 1);
        random_into_bit_set(&mut t, BitSetKind::Binary, &mut |_| false);
    }
}
