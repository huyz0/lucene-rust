//! `BaseNormsFormatTestCase`: norms written through a `CannedNormSimilarity`
//! that hands out chosen values, read back against a numeric doc-values
//! field holding the same values.
//!
//! Differs:
//!
//! - `NumericDocValues.longValues(size, docs, values, default)`, the bulk
//!   fetch, is not ported: its check walks the same documents with
//!   `advanceExact`, which Java's default `longValues` calls.
//! - `maybeWrapWithMergingReader` and `applyCreatedVersionMajor` are not
//!   ported: the readers are `DirectoryReader`s, the index the current
//!   major.
//! - Every directory is a `ByteBuffersDirectory`, not `newDirectory()`'s
//!   (no draw, no `MockDirectoryWrapper` close checks).
//! - `testThreads`' threads run `CheckIndex` over the directory, not
//!   `TestUtil.checkReader` over the shared reader (not ported).
//! - `testMergeStability` is skipped in Java's suite too ("The MockRandom PF
//!   randomizes content on the fly").

use std::sync::{Arc, Barrier, Mutex};

use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    Document, IndexableField, NumericDocValuesField, Store, StringField, TextField,
};
use lucene_index::merge_policy::NoMergePolicy;
use lucene_index::similarity::{FieldInvertState, NormSimilarity};
use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_search::reader::{multi_doc_values, IndexReader as _, LeafReader as _, NO_MORE_DOCS};
use lucene_store::ByteBuffersDirectory;
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::java_random::JavaRandom;

use super::base_index_file_format::{
    new_config, only_leaf, riw_force_merge, riw_with, IndexFileFormatSuite, SuiteTest,
};
use super::index_writer_config::{new_log_merge_policy, Draws, LiveMergePolicy, PolicyKind};
use super::legacy_base_doc_values_format::{doc_of, next};
use super::random_index_writer::RandomIndexWriter;
use crate::analysis::mock_analyzer::MockAnalyzer;
use crate::analysis::mock_tokenizer;
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::test_util;

/// The norms suite's `addRandomFields`: an unstored `foobar` text field.
pub struct NormsSuite;

impl IndexFileFormatSuite for NormsSuite {
    fn add_random_fields(&self, t: &mut LuceneTestCase, doc: &mut Document) {
        let value = test_util::random_simple_string(t.random());
        doc.add(TextField::new("foobar", value, Store::No));
    }
}

/// `CannedNormSimilarity`: each norm is the next non-zero value of `norms`.
#[derive(Debug)]
pub struct CannedNormSimilarity {
    norms: Vec<i64>,
    index: Mutex<usize>,
}

impl CannedNormSimilarity {
    /// `new CannedNormSimilarity(norms)`.
    pub fn new(norms: Vec<i64>) -> Self {
        Self {
            norms,
            index: Mutex::new(0),
        }
    }
}

impl NormSimilarity for CannedNormSimilarity {
    fn compute_norm(&self, _field: &str, state: &FieldInvertState) -> i64 {
        assert!(state.length > 0);
        let mut index = self.index.lock().expect("canned norms");
        loop {
            let norm = self.norms[*index];
            *index += 1;
            if norm != 0 {
                return norm;
            }
        }
    }
}

/// `codecSupportsSparsity()`: the default codec's norms are sparse.
const SPARSE: bool = true;

/// A long supplier drawing from the test's stream.
pub type Longs = Box<dyn FnMut(&mut JavaRandom) -> i64>;

/// A uniform draw in `min..=max` (`TestUtil.nextLong`).
pub fn ranged(min: i64, max: i64) -> Longs {
    Box::new(move |r| test_util::next_long(r, min, max))
}

fn full_long_range() -> Longs {
    Box::new(|r| match r.next_int_bounded(3) {
        0 => i64::MIN,
        1 => i64::MAX,
        _ => test_util::next_long(r, i64::MIN, i64::MAX),
    })
}

fn few_values(a: i64, b: i64) -> Longs {
    Box::new(move |r| if r.next_boolean() { a } else { b })
}

fn all_zeros() -> Longs {
    Box::new(|_| 0)
}

fn most_zeros() -> Longs {
    Box::new(|r| {
        if r.next_int_bounded(100) == 0 {
            byte(r)
        } else {
            0
        }
    })
}

fn byte(r: &mut JavaRandom) -> i64 {
    test_util::next_long(r, i64::from(i8::MIN), i64::from(i8::MAX))
}

/// `testOutliers`' supplier: a common byte value, one draw in a hundred
/// another.
fn outliers(r: &mut JavaRandom) -> Longs {
    let common = byte(r);
    Box::new(move |r| {
        if r.next_int_bounded(100) == 0 {
            byte(r)
        } else {
            common
        }
    })
}

/// `testOutliers2`' supplier: one common and one uncommon byte value.
fn outliers2(r: &mut JavaRandom) -> Longs {
    let common = byte(r);
    let uncommon = byte(r);
    Box::new(move |r| {
        if r.next_int_bounded(100) == 0 {
            uncommon
        } else {
            common
        }
    })
}

/// `testNCommon`'s supplier: `n` common byte values, one draw in a hundred
/// one of up to `256 - n` others.
fn n_common(r: &mut JavaRandom, n: i32) -> Longs {
    let common: Vec<i64> = (0..n).map(|_| byte(r)).collect();
    let num_other = test_util::next_int(r, 2, 256 - n);
    let other: Vec<i64> = (0..num_other).map(|_| byte(r)).collect();
    Box::new(move |r| {
        if r.next_int_bounded(100) == 0 {
            other[r.next_int_bounded(num_other - 1) as usize]
        } else {
            common[r.next_int_bounded(n - 1) as usize]
        }
    })
}

/// `density` 1 for the dense tests, `random().nextDouble()` for the sparse.
fn density(t: &mut LuceneTestCase, sparse: bool) -> f64 {
    if sparse {
        t.random().next_double()
    } else {
        1.0
    }
}

/// The documents with the field: all, or `with_field` of them drawn.
fn docs_with_field(t: &mut LuceneTestCase, num_docs: i32, with_field: i32) -> (FixedBitSet, usize) {
    let mut docs = FixedBitSet::new(num_docs as usize);
    let len = docs.len();
    if with_field == num_docs {
        docs.set_range(0, len);
    } else {
        let mut i = 0;
        while i < with_field {
            let doc = t.random().next_int_bounded(num_docs) as usize;
            if !docs.get(doc) {
                docs.set(doc);
                i += 1;
            }
        }
    }
    (docs, with_field as usize)
}

/// `new MockAnalyzer(random(), MockTokenizer.WHITESPACE, false)`.
fn whitespace_analyzer(t: &mut LuceneTestCase) -> Arc<lucene_analysis::Analyzer> {
    Arc::new(MockAnalyzer::with_tokenizer(t.random(), mock_tokenizer::whitespace(), false).build())
}

/// Adds `num_docs` documents: `id`, and where `docs` says so the `indexed`
/// text (empty for a 0 norm) and the `dv` value, committing one time in 31.
fn index_norms(
    t: &mut LuceneTestCase,
    w: &mut RandomIndexWriter<'_>,
    num_docs: i32,
    docs: &FixedBitSet,
    norms: &[i64],
) {
    assert_eq!(docs.len(), num_docs as usize, "one bit per document");
    let mut j = 0;
    for i in 0..num_docs {
        let id: Box<dyn IndexableField> =
            Box::new(StringField::new("id", i.to_string(), Store::No));
        let doc = if docs.get(i as usize) {
            let value = norms[j];
            j += 1;
            let text = if value == 0 { "" } else { "a" };
            doc_of(vec![
                id,
                Box::new(TextField::new("indexed", text, Store::No)),
                Box::new(NumericDocValuesField::new("dv", value)),
            ])
        } else {
            doc_of(vec![id])
        };
        w.add_document(&doc).expect("addDocument");
        if t.random().next_int_bounded(31) == 0 {
            w.commit().expect("commit");
        }
    }
}

/// `checkNormsVsDocValues(ir)`. Returns the number of norms compared.
pub fn check_norms_vs_doc_values(reader: &DirectoryReader) -> usize {
    reader.segment_readers().iter().map(check_leaf).sum()
}

fn check_leaf(leaf: &SegmentReader) -> usize {
    let expected = leaf.numeric_doc_values("dv").expect("getNumericDocValues");
    let actual = leaf.norm_values("indexed").expect("getNormValues");
    assert_eq!(expected.is_none(), actual.is_none());
    let (Some(mut expected), Some(mut actual)) = (expected, actual) else {
        return 0;
    };
    let mut compared = 0;
    let mut d = next(expected.as_mut());
    while d != NO_MORE_DOCS {
        assert_eq!(d, next(actual.as_mut()));
        assert_eq!(expected.long_value(), actual.long_value(), "doc {d}");
        compared += 1;
        d = next(expected.as_mut());
    }
    assert_eq!(NO_MORE_DOCS, next(actual.as_mut()));

    // Java's bulk `longValues`: the same strides through `advanceExact`.
    let mut expected = leaf.numeric_doc_values("dv").expect("dv").expect("dv");
    let mut actual = leaf.norm_values("indexed").expect("norms").expect("norms");
    let max_doc = leaf.max_doc();
    let mut doc = -1;
    while doc < max_doc {
        for j in 0..16 {
            doc += 1 + (j & 0x03);
            if doc >= max_doc {
                break;
            }
            let e = expected.advance_exact(doc).expect("advanceExact");
            let a = actual.advance_exact(doc).expect("advanceExact");
            assert_eq!(e, a, "doc {doc}");
            if e {
                assert_eq!(expected.long_value(), actual.long_value(), "doc {doc}");
            }
        }
    }
    compared
}

/// `doTestNormsVersusDocValues(density, longs)`. Returns the number of
/// norms compared, before and after the merge.
pub fn norms_versus_doc_values(t: &mut LuceneTestCase, density: f64, longs: &mut Longs) -> usize {
    let num_docs = t.at_least(500);
    // `Math.max(1, (int) (density * numDocs))`, in double.
    let with_field = 1.max((density * f64::from(num_docs)) as i32);
    let (docs, with_field) = docs_with_field(t, num_docs, with_field);
    let norms: Vec<i64> = (0..with_field).map(|_| longs(t.random())).collect();

    let dir = ByteBuffersDirectory::new();
    let analyzer = whitespace_analyzer(t);
    let config = new_config(t, analyzer);
    let mut w = riw_with(t, &dir, config);
    w.writer()
        .set_similarity(Some(Arc::new(CannedNormSimilarity::new(norms.clone()))));
    index_norms(t, &mut w, num_docs, &docs, &norms);

    let num_deletions = t.random().next_int_bounded(num_docs / 20);
    for _ in 0..num_deletions {
        let id = t.random().next_int_bounded(num_docs);
        w.delete_documents(Term::new("id", id.to_string()))
            .expect("deleteDocuments");
    }
    w.commit().expect("commit");

    let reader = DirectoryReader::open(&dir).expect("open");
    let mut compared = check_norms_vs_doc_values(&reader);
    drop(reader);

    riw_force_merge(&mut w, 1);
    let reader = DirectoryReader::open(&dir).expect("open");
    compared += check_norms_vs_doc_values(&reader);
    w.close().expect("close");
    compared
}

/// `atLeast(1)` runs of `doTestNormsVersusDocValues`, each with a supplier
/// `make` draws (before the density, as Java's do).
fn iterate(t: &mut LuceneTestCase, sparse: bool, make: impl Fn(&mut JavaRandom) -> Longs) {
    if sparse && !SPARSE {
        return;
    }
    let iterations = t.at_least(1);
    for _ in 0..iterations {
        let mut longs = make(t.random());
        let d = density(t, sparse);
        norms_versus_doc_values(t, d, &mut longs);
    }
}

const BYTE: (i64, i64) = (i8::MIN as i64, i8::MAX as i64);
const SHORT: (i64, i64) = (i16::MIN as i64, i16::MAX as i64);

/// `testByteRange`.
pub fn test_byte_range(t: &mut LuceneTestCase) {
    iterate(t, false, |_| ranged(BYTE.0, BYTE.1));
}

/// `testSparseByteRange`.
pub fn test_sparse_byte_range(t: &mut LuceneTestCase) {
    iterate(t, true, |_| ranged(BYTE.0, BYTE.1));
}

/// `testShortRange`.
pub fn test_short_range(t: &mut LuceneTestCase) {
    iterate(t, false, |_| ranged(SHORT.0, SHORT.1));
}

/// `testSparseShortRange`.
pub fn test_sparse_short_range(t: &mut LuceneTestCase) {
    iterate(t, true, |_| ranged(SHORT.0, SHORT.1));
}

/// `testLongRange`.
pub fn test_long_range(t: &mut LuceneTestCase) {
    iterate(t, false, |_| ranged(i64::MIN, i64::MAX));
}

/// `testSparseLongRange`.
pub fn test_sparse_long_range(t: &mut LuceneTestCase) {
    iterate(t, true, |_| ranged(i64::MIN, i64::MAX));
}

/// `testFullLongRange`.
pub fn test_full_long_range(t: &mut LuceneTestCase) {
    iterate(t, false, |_| full_long_range());
}

/// `testSparseFullLongRange`.
pub fn test_sparse_full_long_range(t: &mut LuceneTestCase) {
    iterate(t, true, |_| full_long_range());
}

/// `testFewValues`.
pub fn test_few_values(t: &mut LuceneTestCase) {
    iterate(t, false, |_| few_values(20, 3));
}

/// `testFewSparseValues`.
pub fn test_few_sparse_values(t: &mut LuceneTestCase) {
    iterate(t, true, |_| few_values(20, 3));
}

/// `testFewLargeValues`.
pub fn test_few_large_values(t: &mut LuceneTestCase) {
    iterate(t, false, |_| few_values(1_000_000, -5000));
}

/// `testFewSparseLargeValues`.
pub fn test_few_sparse_large_values(t: &mut LuceneTestCase) {
    iterate(t, true, |_| few_values(1_000_000, -5000));
}

/// `testAllZeros`.
pub fn test_all_zeros(t: &mut LuceneTestCase) {
    iterate(t, false, |_| all_zeros());
}

/// `testSparseAllZeros`.
pub fn test_sparse_all_zeros(t: &mut LuceneTestCase) {
    iterate(t, true, |_| all_zeros());
}

/// `testMostZeros`.
pub fn test_most_zeros(t: &mut LuceneTestCase) {
    iterate(t, false, |_| most_zeros());
}

/// `testOutliers`.
pub fn test_outliers(t: &mut LuceneTestCase) {
    iterate(t, false, outliers);
}

/// `testSparseOutliers`.
pub fn test_sparse_outliers(t: &mut LuceneTestCase) {
    iterate(t, true, outliers);
}

/// `testOutliers2`.
pub fn test_outliers2(t: &mut LuceneTestCase) {
    iterate(t, false, outliers2);
}

/// `testSparseOutliers2`.
pub fn test_sparse_outliers2(t: &mut LuceneTestCase) {
    iterate(t, true, outliers2);
}

/// `testNCommon`.
pub fn test_n_common(t: &mut LuceneTestCase) {
    let n = test_util::next_int(t.random(), 2, 15);
    let mut longs = n_common(t.random(), n);
    norms_versus_doc_values(t, 1.0, &mut longs);
}

/// `testSparseNCommon`.
pub fn test_sparse_n_common(t: &mut LuceneTestCase) {
    let n = test_util::next_int(t.random(), 2, 15);
    let mut longs = n_common(t.random(), n);
    let d = t.random().next_double();
    norms_versus_doc_values(t, d, &mut longs);
}

/// `@Nightly testNCommonBig` (`sparse` false) and `testSparseNCommonBig`:
/// every `n` from 2 to 15.
pub fn n_common_big(t: &mut LuceneTestCase, sparse: bool) {
    let iterations = t.at_least(1);
    for _ in 0..iterations {
        for n in 2..16 {
            let mut longs = n_common(t.random(), n);
            let d = density(t, sparse);
            norms_versus_doc_values(t, d, &mut longs);
        }
    }
}

/// `testUndeadNorms`: once every document with `content` is deleted and
/// merged away, its norms exist and are empty.
pub fn test_undead_norms(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = super::base_index_file_format::mock_analyzer(t);
    let config = new_config(t, analyzer);
    let mut w = riw_with(t, &dir, config);
    let num_docs = t.at_least(500);
    let mut to_delete = Vec::new();
    for i in 0..num_docs {
        let mut fields: Vec<Box<dyn IndexableField>> =
            vec![Box::new(StringField::new("id", i.to_string(), Store::No))];
        if t.random().next_int_bounded(5) == 1 {
            to_delete.push(i);
            fields.push(Box::new(TextField::new(
                "content",
                "some content",
                Store::No,
            )));
        }
        w.add_document(&doc_of(fields)).expect("addDocument");
    }
    for id in to_delete {
        w.delete_documents(Term::new("id", id.to_string()))
            .expect("deleteDocuments");
    }
    riw_force_merge(&mut w, 1);
    let reader = w.get_reader().expect("getReader");
    assert!(!reader.has_deletions());
    let mut norms = multi_doc_values::norm_values(&reader, "content")
        .expect("getNormValues")
        .expect("norms exist");
    const { assert!(SPARSE) };
    assert_eq!(NO_MORE_DOCS, next(norms.as_mut()));
    drop(norms);
    drop(reader);
    w.close().expect("close");
}

/// `testThreads`: several threads check one reader at once.
pub fn test_threads(t: &mut LuceneTestCase) {
    let d: f32 = if !SPARSE || t.random().next_boolean() {
        1.0
    } else {
        t.random().next_float()
    };
    let num_docs = t.at_least(500);
    // `Math.max(1, (int) (density * numDocs))`, in float: Java's `density`
    // here is a `float`.
    #[allow(clippy::cast_precision_loss)]
    let with_field = 1.max((d * num_docs as f32) as i32);
    let (docs, with_field) = docs_with_field(t, num_docs, with_field);
    let norms: Vec<i64> = (0..with_field).map(|_| t.random().next_long()).collect();

    let dir = ByteBuffersDirectory::new();
    let analyzer = whitespace_analyzer(t);
    let mut config = new_config(t, analyzer);
    config.merge_policy = LiveMergePolicy::new(PolicyKind::NoMerge(NoMergePolicy::default()));
    let mut w = riw_with(t, &dir, config);
    w.writer()
        .set_similarity(Some(Arc::new(CannedNormSimilarity::new(norms.clone()))));
    index_norms(t, &mut w, num_docs, &docs, &norms);
    let reader = w.get_reader().expect("getReader");
    w.close().expect("close");

    let num_threads = test_util::next_int(t.random(), 3, 30) as usize;
    let gun = Barrier::new(num_threads);
    std::thread::scope(|s| {
        for _ in 0..num_threads {
            s.spawn(|| {
                gun.wait();
                check_norms_vs_doc_values(&reader);
                // Java's `TestUtil.checkReader(reader)`: not ported, so
                // `CheckIndex` over the directory under the reader.
                test_util::check_index(&dir).expect("CheckIndex");
            });
        }
    });
}

/// `testIndependantIterators` (`sparse` false) and
/// `testIndependantSparseIterators`: two norms iterators over one field
/// do not move each other.
fn independent_iterators(t: &mut LuceneTestCase, sparse: bool) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = super::base_index_file_format::mock_analyzer(t);
    let mut config = new_config(t, analyzer);
    let (nightly, multiplier) = (t.nightly(), t.multiplier());
    let log = new_log_merge_policy(&mut Draws::new(t.random()).knobs(nightly, multiplier));
    config.merge_policy = LiveMergePolicy::new(PolicyKind::Log(log));
    let mut w = riw_with(t, &dir, config);
    w.writer()
        .set_similarity(Some(Arc::new(CannedNormSimilarity::new(vec![42, 10, 20]))));
    let mut doc = Document::new();
    doc.add(TextField::new("indexed", "a", Store::No));
    for _ in 0..3 {
        w.add_document(&doc).expect("addDocument");
        if sparse {
            w.add_document(&Document::new()).expect("addDocument");
        }
    }
    riw_force_merge(&mut w, 1);
    let reader = w.get_reader().expect("getReader");
    let leaf = only_leaf(&reader);
    let mut n1 = leaf.norm_values("indexed").expect("norms").expect("norms");
    let mut n2 = leaf.norm_values("indexed").expect("norms").expect("norms");
    let step = if sparse { 2 } else { 1 };
    let check = |it: &mut dyn lucene_search::reader::NumericDocValues, doc: i32, v: i64| {
        assert_eq!(doc, next(it));
        assert_eq!(v, it.long_value());
    };
    check(n1.as_mut(), 0, 42);
    check(n1.as_mut(), step, 10);
    check(n2.as_mut(), 0, 42);
    check(n2.as_mut(), step, 10);
    check(n2.as_mut(), 2 * step, 20);
    check(n1.as_mut(), 2 * step, 20);
    assert_eq!(NO_MORE_DOCS, next(n1.as_mut()));
    assert_eq!(NO_MORE_DOCS, next(n2.as_mut()));
    drop((n1, n2));
    drop(reader);
    w.close().expect("close");
}

/// `testIndependantIterators`.
pub fn test_independant_iterators(t: &mut LuceneTestCase) {
    independent_iterators(t, false);
}

/// `testIndependantSparseIterators`.
pub fn test_independant_sparse_iterators(t: &mut LuceneTestCase) {
    independent_iterators(t, true);
}

/// Every `BaseNormsFormatTestCase` test this port runs by default.
pub const TESTS: &[SuiteTest] = &[
    ("testByteRange", test_byte_range),
    ("testSparseByteRange", test_sparse_byte_range),
    ("testShortRange", test_short_range),
    ("testSparseShortRange", test_sparse_short_range),
    ("testLongRange", test_long_range),
    ("testSparseLongRange", test_sparse_long_range),
    ("testFullLongRange", test_full_long_range),
    ("testSparseFullLongRange", test_sparse_full_long_range),
    ("testFewValues", test_few_values),
    ("testFewSparseValues", test_few_sparse_values),
    ("testFewLargeValues", test_few_large_values),
    ("testFewSparseLargeValues", test_few_sparse_large_values),
    ("testAllZeros", test_all_zeros),
    ("testSparseAllZeros", test_sparse_all_zeros),
    ("testMostZeros", test_most_zeros),
    ("testOutliers", test_outliers),
    ("testSparseOutliers", test_sparse_outliers),
    ("testOutliers2", test_outliers2),
    ("testSparseOutliers2", test_sparse_outliers2),
    ("testNCommon", test_n_common),
    ("testSparseNCommon", test_sparse_n_common),
    ("testUndeadNorms", test_undead_norms),
    ("testThreads", test_threads),
    ("testIndependantIterators", test_independant_iterators),
    (
        "testIndependantSparseIterators",
        test_independant_sparse_iterators,
    ),
];
