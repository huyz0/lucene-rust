//! `BasePointsFormatTestCase`: the BKD points format (`Lucene90PointsFormat`)
//! written through `IndexWriter` and read back with `PointValues.intersect`
//! and the index tree, against the values the test drew.
//!
//! Differs:
//!
//! - `assertSize` walks [`PointsReader::point_tree`]'s nodes, which are
//!   values: `size()` is [`PointsReader::point_tree_size`], `visitDocIDs`
//!   [`PointsReader::visit_doc_ids`], and `visitDocValues` a walk down to
//!   [`PointsReader::visit_leaf`]. On top of Java's checks, a node's count
//!   must be its children's sum and every point it visits must lie in its
//!   cell. `randomPointTreeNavigation` checks that each child's cell lies
//!   in its parent's: a value node cannot fail to come back to where it
//!   started, and a clone is equal by construction.
//! - `switchIndex`'s two reader paths (`addIndexes(CodecReader...)` and
//!   `TestUtil.addIndexesSlowly`) commit the side writer and merge its
//!   directory in (`add_indexes_merged`): the port merges from directories,
//!   not from an NRT reader.
//! - `testMismatchedFields`' `MismatchedCodecReader` is the first index
//!   adding its fields in the other order, so the two number them
//!   differently; `DirectoryReader.open(writer)` is a reader over the commit.
//! - `testAddIndexes`' `searcher.count(IntPoint.newExactQuery(..))` is
//!   `pointCount` ([`PointsReader::count_points`]) on a leaf `Weight.count`
//!   would answer, and an exact-value `intersect` over the live documents
//!   on any other.
//! - `testDocCountEdgeCases` and `testRandomDocCount`'s fake `PointValues`
//!   is the three inputs `estimateDocCount` reads
//!   ([`estimate_doc_count`]): its one-node tree's `estimatePointCount` is
//!   the node's size.
//! - `testWithExceptions` does not suppress `ConcurrentMergeScheduler`
//!   exceptions (a merge error reaches the writer's caller, and counts as
//!   injected when it is), and gives up after 100 runs without one.
//! - A run of more than 10,000 values (`useRealWriter`) adds through the
//!   `RandomIndexWriter`'s `IndexWriter` but keeps `newIndexWriterConfig`.
//!   `@Nightly testRandomBinaryBig` is [`random_binary`] with its count as
//!   a parameter, and `getDirectory` is always `newDirectory()`.
//! - `testMergeStability` is suppressed, as in Java (`mergeIsStable`).

use lucene_codecs::points::{IntersectVisitor, PointTreeNode, PointsReader, Relation};
use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    BinaryPoint, Document, FieldType, IntPoint, LongPoint, NumericDocValuesField, Store,
    StringField, MAX_DIMENSIONS, MAX_INDEX_DIMENSIONS, MAX_NUM_BYTES,
};
use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_search::points_query::estimate_doc_count;
use lucene_search::reader::{IndexReader as _, LeafReader as _};
use lucene_store::Directory;
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::numeric_utils;
use lucene_util::test_support::TempDir;

use super::base_index_file_format::{
    force_merge, mock_analyzer, new_config, new_riw, only_leaf, riw_force_merge, writer_env,
    writer_with, IndexFileFormatSuite, SuiteTest,
};
use super::index_writer_config::{
    new_log_merge_policy, Draws, LiveMergePolicy, MergeSchedulerChoice, PolicyKind,
    RandomIndexWriterConfig,
};
use super::legacy_base_doc_values_format::next;
use super::random_index_writer::RandomIndexWriter;
use crate::store::base_directory_wrapper::BaseDirectoryWrapper;
use crate::store::mock_directory_wrapper::is_injected_io_exception;
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::test_util;

/// The points suite's `addRandomFields`: up to two `IntPoint`s in `f`.
pub struct PointsSuite;

impl IndexFileFormatSuite for PointsSuite {
    fn add_random_fields(&self, t: &mut LuceneTestCase, doc: &mut Document) {
        let num_values = t.random().next_int_bounded(3);
        for _ in 0..num_values {
            let value = t.random().next_int();
            doc.add(IntPoint::new("f", &[value]).expect("IntPoint"));
        }
    }

    /// `mergeIsStable()`: a merged BKD tree is shaped unlike the trees
    /// flushed point by point.
    fn merge_is_stable(&self) -> bool {
        false
    }
}

/// Packed points by document (or by value, see `ids`): `[value][dim]`.
pub type DocValues = Vec<Vec<Vec<u8>>>;

/// `getDirectory(numPoints)`: `newDirectory()` (Java's filesystem
/// directory past 100,000 points is a nightly size).
fn get_directory(t: &mut LuceneTestCase, _num_points: usize) -> Box<dyn BaseDirectoryWrapper> {
    t.new_directory()
}

/// `newIndexWriterConfig()` with `newLogMergePolicy()`: docids keep their
/// order.
fn log_config(t: &mut LuceneTestCase) -> RandomIndexWriterConfig {
    let analyzer = mock_analyzer(t);
    let mut config = new_config(t, analyzer);
    let (nightly, multiplier) = (t.nightly(), t.multiplier());
    let log = new_log_merge_policy(&mut Draws::new(t.random()).knobs(nightly, multiplier));
    config.merge_policy = LiveMergePolicy::new(PolicyKind::Log(log));
    config
}

fn int_point(i: i32) -> Vec<u8> {
    let mut point = vec![0u8; 4];
    numeric_utils::int_to_sortable_bytes(i, &mut point, 0);
    point
}

fn binary_point(name: &str, point: &[u8]) -> BinaryPoint {
    BinaryPoint::new(name, &[point]).expect("BinaryPoint")
}

/// A visitor that crosses every cell and collects each point's document,
/// for the tests whose every point is its document's sortable int.
struct SeeAll<'a> {
    seen: FixedBitSet,
    live: Option<&'a FixedBitSet>,
    expect: &'a dyn Fn(i32) -> i32,
}

impl IntersectVisitor for SeeAll<'_> {
    fn compare(&mut self, _min: &[u8], _max: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }
    fn visit(&mut self, _doc_id: i32) {
        panic!("IllegalStateException: visit(docID) under CELL_CROSSES_QUERY");
    }
    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        // FBS: `seen` and `live` are both `maxDoc` long, and the reader
        // hands over only its own documents.
        if self.live.is_none_or(|l| l.get(doc_id as usize)) {
            self.seen.set(doc_id as usize);
        }
        assert_eq!(
            (self.expect)(doc_id),
            numeric_utils::sortable_bytes_to_int(packed, 0)
        );
    }
}

/// `testBasic` (`commit_at` `None`) and `testMerge` (a commit after
/// document 10): twenty sortable ints, force-merged, each visited once.
fn basic(t: &mut LuceneTestCase, commit_at: Option<i32>) {
    let dir = get_directory(t, 20);
    let config = log_config(t);
    let mut w = writer_with(&*dir, &config);
    for i in 0..20 {
        let mut doc = Document::new();
        doc.add(binary_point("dim", &int_point(i)));
        w.add_fields_document(&doc).expect("addDocument");
        if Some(i) == commit_at {
            w.commit().expect("commit");
        }
    }
    force_merge(&mut w, 1);
    w.close().expect("close");

    let r = DirectoryReader::open(&*dir).expect("open");
    let sub = only_leaf(&r);
    let values = sub.point_values("dim").expect("points").expect("dim");
    let mut seen = SeeAll {
        seen: FixedBitSet::new(20),
        live: None,
        expect: &|doc| doc,
    };
    values.intersect(&mut seen).expect("intersect");
    assert_eq!(20, seen.seen.cardinality());
    drop(values);
    drop(r);
    dir.close().expect("close");
}

/// `testBasic`.
pub fn test_basic(t: &mut LuceneTestCase) {
    basic(t, None);
}

/// `testMerge`.
pub fn test_merge(t: &mut LuceneTestCase) {
    basic(t, Some(10));
}

/// `testAllPointDocsDeletedInSegment`.
pub fn test_all_point_docs_deleted_in_segment(t: &mut LuceneTestCase) {
    let dir = get_directory(t, 20);
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let mut w = writer_with(&*dir, &config);
    for i in 0..10 {
        let mut doc = Document::new();
        doc.add(binary_point("dim", &int_point(i)));
        doc.add(NumericDocValuesField::new("id", i64::from(i)));
        doc.add(StringField::new("x", "x", Store::No));
        w.add_fields_document(&doc).expect("addDocument");
    }
    w.add_fields_document(&Document::new())
        .expect("addDocument");
    w.delete_documents_by_term(&[Term::new("x", "x")])
        .expect("deleteDocuments");
    if t.random().next_boolean() {
        force_merge(&mut w, 1);
    }
    w.close().expect("close");
    let r = DirectoryReader::open(&*dir).expect("open");
    assert_eq!(1, r.num_docs());
    for leaf in r.segment_readers() {
        let Some(mut id_values) = leaf.numeric_doc_values("id").expect("id") else {
            // The ten documents flushed and merged away before the eleventh.
            continue;
        };
        let mut doc_id_to_id = vec![0i32; leaf.max_doc() as usize];
        loop {
            let doc = next(id_values.as_mut());
            if doc == lucene_search::reader::NO_MORE_DOCS {
                break;
            }
            doc_id_to_id[doc as usize] = id_values.long_value() as i32;
        }
        if let Some(values) = leaf.point_values("dim").expect("points") {
            let mut seen = SeeAll {
                seen: FixedBitSet::new(leaf.max_doc() as usize),
                live: leaf.live_docs(),
                expect: &|doc| doc_id_to_id[doc as usize],
            };
            values.intersect(&mut seen).expect("intersect");
            assert_eq!(0, seen.seen.cardinality());
        }
    }
    drop(r);
    dir.close().expect("close");
}

/// `handlePossiblyFakeException`: an injected error, or a merge that hit
/// one.
fn is_fake(text: &str) -> bool {
    text.contains("a random IOException")
        || text.contains("background merge hit exception")
        || is_injected_io_exception(&text)
}

/// `testWithExceptions`: [`verify_in`] over a directory failing one
/// operation in twenty, until an injected error stops a run. Returns the
/// runs; Java's loop has no bound, this one panics after
/// `MAX_EXCEPTION_RUNS` (100) runs rather than hang when no error is ever injected.
pub fn test_with_exceptions(t: &mut LuceneTestCase) -> usize {
    /// Runs without an injected error before the test gives up.
    const MAX_EXCEPTION_RUNS: usize = 100;
    let num_docs = t.at_least(1000);
    let bytes = test_util::next_int(t.random(), 2, MAX_NUM_BYTES);
    let num_dims = test_util::next_int(t.random(), 1, MAX_DIMENSIONS);
    let num_index_dims = test_util::next_int(t.random(), 1, num_dims.min(MAX_INDEX_DIMENSIONS));
    let doc_values = random_values(t, num_docs as usize, num_dims, bytes);
    let mut runs = 0;
    loop {
        runs += 1;
        assert!(
            runs <= MAX_EXCEPTION_RUNS,
            "no injected error stopped {MAX_EXCEPTION_RUNS} runs"
        );
        let temp = TempDir::new("points-exceptions");
        let mut dir = t.new_mock_fs_directory(temp.path());
        dir.keep_alive(temp);
        dir.set_random_io_exception_rate(0.05);
        dir.set_random_io_exception_rate_on_open(0.05);
        let result = verify_in(
            t,
            &dir,
            &doc_values,
            None,
            num_dims as usize,
            num_index_dims as usize,
            bytes as usize,
        );
        dir.set_random_io_exception_rate(0.0);
        dir.set_random_io_exception_rate_on_open(0.0);
        let done = match result {
            Ok(()) => false,
            Err(text) if is_fake(&text) || text.contains("does not exist; files=") => true,
            // Too small a sort heap for the leaf size: retry.
            Err(text) if text.contains("either increase maxMBSortInHeap") => false,
            Err(text) => panic!("unexpected exception: {text}"),
        };
        dir.close().expect("close");
        if done {
            return runs;
        }
    }
}

/// `numValues` values of `num_dims` random dimensions of `bytes` bytes.
fn random_values(
    t: &mut LuceneTestCase,
    num_values: usize,
    num_dims: i32,
    bytes: i32,
) -> DocValues {
    (0..num_values)
        .map(|_| random_value(t, num_dims, bytes))
        .collect()
}

fn random_value(t: &mut LuceneTestCase, num_dims: i32, bytes: i32) -> Vec<Vec<u8>> {
    (0..num_dims)
        .map(|_| {
            let mut v = vec![0u8; bytes as usize];
            t.random().next_bytes(&mut v);
            v
        })
        .collect()
}

/// `testMultiValued`: one to five values per document.
pub fn test_multi_valued(t: &mut LuceneTestCase) {
    let bytes = test_util::next_int(t.random(), 2, MAX_NUM_BYTES);
    let num_dims = test_util::next_int(t.random(), 1, MAX_DIMENSIONS);
    let num_index_dims = test_util::next_int(t.random(), 1, MAX_INDEX_DIMENSIONS.min(num_dims));
    let num_docs = if t.nightly() {
        t.at_least(1000)
    } else {
        t.at_least(100)
    };
    let mut doc_values = Vec::new();
    let mut doc_ids = Vec::new();
    for doc_id in 0..num_docs {
        let num_values_in_doc = test_util::next_int(t.random(), 1, 5);
        for _ in 0..num_values_in_doc {
            doc_ids.push(doc_id);
            doc_values.push(random_value(t, num_dims, bytes));
        }
    }
    verify(
        t,
        &doc_values,
        Some(&doc_ids),
        num_dims as usize,
        num_index_dims as usize,
        bytes as usize,
    );
}

/// `testAllEqual`: every document the first one's point.
pub fn test_all_equal(t: &mut LuceneTestCase) {
    let bytes = test_util::next_int(t.random(), 2, MAX_NUM_BYTES);
    let num_dims = test_util::next_int(t.random(), 1, MAX_INDEX_DIMENSIONS);
    let num_docs = t.at_least(1000) as usize;
    let first = random_value(t, num_dims, bytes);
    let doc_values = vec![first; num_docs];
    let dims = num_dims as usize;
    verify(t, &doc_values, None, dims, dims, bytes as usize);
}

/// `testOneDimEqual`: one dimension the same in every document.
pub fn test_one_dim_equal(t: &mut LuceneTestCase) {
    let bytes = test_util::next_int(t.random(), 2, MAX_NUM_BYTES);
    let num_dims = test_util::next_int(t.random(), 1, MAX_INDEX_DIMENSIONS);
    let num_docs = t.at_least(1000) as usize;
    let the_equal_dim = t.random().next_int_bounded(num_dims) as usize;
    let mut doc_values: DocValues = Vec::with_capacity(num_docs);
    for doc_id in 0..num_docs {
        let mut values = random_value(t, num_dims, bytes);
        if doc_id > 0 {
            values[the_equal_dim] = doc_values[0][the_equal_dim].clone();
        }
        doc_values.push(values);
    }
    let dims = num_dims as usize;
    verify(t, &doc_values, None, dims, dims, bytes as usize);
}

/// `testOneDimTwoValues`: one dimension takes one of two values (run
/// lengths past 255).
pub fn test_one_dim_two_values(t: &mut LuceneTestCase) {
    let bytes = test_util::next_int(t.random(), 2, MAX_NUM_BYTES);
    let num_dims = test_util::next_int(t.random(), 1, MAX_INDEX_DIMENSIONS);
    let num_docs = t.at_least(1000) as usize;
    let the_dim = t.random().next_int_bounded(num_dims);
    let mut value1 = vec![0u8; bytes as usize];
    t.random().next_bytes(&mut value1);
    let mut value2 = vec![0u8; bytes as usize];
    t.random().next_bytes(&mut value2);
    let mut doc_values: DocValues = Vec::with_capacity(num_docs);
    for _ in 0..num_docs {
        let values = (0..num_dims)
            .map(|dim| {
                if dim == the_dim {
                    if t.random().next_boolean() {
                        value1.clone()
                    } else {
                        value2.clone()
                    }
                } else {
                    let mut v = vec![0u8; bytes as usize];
                    t.random().next_bytes(&mut v);
                    v
                }
            })
            .collect();
        doc_values.push(values);
    }
    let dims = num_dims as usize;
    verify(t, &doc_values, None, dims, dims, bytes as usize);
}

/// `randomBigInt(numBytes)`: `new BigInteger(numBytes * 8 - 1, random())`,
/// negated half the time. At most 127 bits, so an `i128`.
fn random_big_int(t: &mut LuceneTestCase, num_bytes: usize) -> i128 {
    // BigInteger.randomBits: numBytes bytes, the top bit masked off.
    let mut magnitude = vec![0u8; num_bytes];
    t.random().next_bytes(&mut magnitude);
    magnitude[0] &= 0x7f;
    let x = magnitude
        .iter()
        .fold(0i128, |acc, &b| (acc << 8) | i128::from(b));
    if t.random().next_boolean() {
        -x
    } else {
        x
    }
}

/// `NumericUtils.bigIntToSortableBytes(value, numBytes, ...)`.
fn big_int_to_sortable(value: i128, num_bytes: usize) -> Vec<u8> {
    let full = value.to_be_bytes();
    let mut out = vec![0u8; num_bytes];
    numeric_utils::big_int_to_sortable_bytes(&full[16 - num_bytes..], num_bytes, &mut out, 0)
        .expect("fits");
    out
}

/// `NumericUtils.sortableBytesToBigInt(packed, offset, numBytes)`.
fn sortable_to_big_int(packed: &[u8], offset: usize, num_bytes: usize) -> i128 {
    let twos = numeric_utils::sortable_bytes_to_big_int(packed, offset, num_bytes);
    let fill = if twos[0] & 0x80 != 0 { 0xff } else { 0 };
    let mut full = [fill; 16];
    full[16 - num_bytes..].copy_from_slice(&twos);
    i128::from_be_bytes(full)
}

/// The `BigInteger` rectangle visitor of `testBigIntNDims`.
struct BigIntRect<'a> {
    query_min: &'a [i128],
    query_max: &'a [i128],
    bytes: usize,
    doc_base: usize,
    hits: &'a mut FixedBitSet,
}

impl IntersectVisitor for BigIntRect<'_> {
    fn visit(&mut self, doc_id: i32) {
        // FBS: `hits` is the reader's `maxDoc` long, past every leaf's base.
        self.hits.set(self.doc_base + doc_id as usize);
    }
    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        for dim in 0..self.query_min.len() {
            let x = sortable_to_big_int(packed, dim * self.bytes, self.bytes);
            if x < self.query_min[dim] || x > self.query_max[dim] {
                return;
            }
        }
        // FBS: as in `visit`.
        self.hits.set(self.doc_base + doc_id as usize);
    }
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        let mut crosses = false;
        for dim in 0..self.query_min.len() {
            let min = sortable_to_big_int(min_packed, dim * self.bytes, self.bytes);
            let max = sortable_to_big_int(max_packed, dim * self.bytes, self.bytes);
            assert!(max >= min);
            if max < self.query_min[dim] || min > self.query_max[dim] {
                return Relation::CellOutsideQuery;
            } else if min < self.query_min[dim] || max > self.query_max[dim] {
                crosses = true;
            }
        }
        if crosses {
            Relation::CellCrossesQuery
        } else {
            Relation::CellInsideQuery
        }
    }
}

/// `testBigIntNDims`: N-dimension points of `BigInteger`s, rectangle
/// queries checked against the values.
pub fn test_big_int_n_dims(t: &mut LuceneTestCase) {
    let num_docs = t.at_least(200) as usize;
    let dir = get_directory(t, num_docs);
    let bytes = test_util::next_int(t.random(), 2, MAX_NUM_BYTES) as usize;
    let num_dims = test_util::next_int(t.random(), 1, MAX_INDEX_DIMENSIONS) as usize;
    let config = log_config(t);
    let env = writer_env(t);
    let mut w =
        RandomIndexWriter::with_config(t.random(), &*dir, config, env).expect("RandomIndexWriter");
    let mut docs = Vec::with_capacity(num_docs);
    for _ in 0..num_docs {
        let values: Vec<i128> = (0..num_dims).map(|_| random_big_int(t, bytes)).collect();
        let packed: Vec<Vec<u8>> = values
            .iter()
            .map(|&v| big_int_to_sortable(v, bytes))
            .collect();
        let dims: Vec<&[u8]> = packed.iter().map(Vec::as_slice).collect();
        docs.push(values);
        let mut doc = Document::new();
        doc.add(BinaryPoint::new("field", &dims).expect("BinaryPoint"));
        w.add_document(&doc).expect("addDocument");
    }
    let r = w.get_reader().expect("getReader");
    w.close().expect("close");

    let iters = t.at_least(100);
    for _ in 0..iters {
        let mut query_min = vec![0i128; num_dims];
        let mut query_max = vec![0i128; num_dims];
        for dim in 0..num_dims {
            query_min[dim] = random_big_int(t, bytes);
            query_max[dim] = random_big_int(t, bytes);
            if query_min[dim] > query_max[dim] {
                std::mem::swap(&mut query_min[dim], &mut query_max[dim]);
            }
        }
        let mut hits = FixedBitSet::new(r.max_doc() as usize);
        let mut doc_base = 0usize;
        for leaf in r.segment_readers() {
            if let Some(values) = leaf.point_values("field").expect("points") {
                let mut v = BigIntRect {
                    query_min: &query_min,
                    query_max: &query_max,
                    bytes,
                    doc_base,
                    hits: &mut hits,
                };
                values.intersect(&mut v).expect("intersect");
            }
            doc_base += leaf.max_doc() as usize;
        }
        for (doc_id, values) in docs.iter().enumerate() {
            let expected =
                (0..num_dims).all(|d| values[d] >= query_min[d] && values[d] <= query_max[d]);
            // FBS: `hits` is `maxDoc` long, at least one per drawn document.
            assert_eq!(expected, hits.get(doc_id), "docID={doc_id}");
        }
    }
    drop(r);
    dir.close().expect("close");
}

/// `doTestRandomBinary(count)`: `count..=2*count` random points.
pub fn random_binary(t: &mut LuceneTestCase, count: i32) {
    let num_docs = test_util::next_int(t.random(), count, count * 2);
    let bytes = test_util::next_int(t.random(), 2, MAX_NUM_BYTES);
    let num_data_dims = test_util::next_int(t.random(), 1, MAX_INDEX_DIMENSIONS);
    let num_index_dims = test_util::next_int(t.random(), 1, num_data_dims);
    let doc_values = random_values(t, num_docs as usize, num_data_dims, bytes);
    verify(
        t,
        &doc_values,
        None,
        num_data_dims as usize,
        num_index_dims as usize,
        bytes as usize,
    );
}

/// `testRandomBinaryTiny`.
pub fn test_random_binary_tiny(t: &mut LuceneTestCase) {
    random_binary(t, 10);
}

/// `testRandomBinaryMedium`.
pub fn test_random_binary_medium(t: &mut LuceneTestCase) {
    random_binary(t, 200);
}

/// `verify(docValues, docIDs, numDataDims, numIndexDims, numBytesPerDim)`.
///
/// # Panics
/// When a query's hits differ from the values', or the writer fails.
pub fn verify(
    t: &mut LuceneTestCase,
    doc_values: &[Vec<Vec<u8>>],
    ids: Option<&[i32]>,
    num_dims: usize,
    num_index_dims: usize,
    bytes: usize,
) {
    let dir = get_directory(t, doc_values.len());
    loop {
        match verify_in(t, &*dir, doc_values, ids, num_dims, num_index_dims, bytes) {
            Ok(()) => break,
            // A too-small maxMBSortInHeap for maxPointsInLeafNode: retry.
            Err(e)
                if e.contains(
                    "either increase maxMBSortInHeap or decrease maxPointsInLeafNode",
                ) => {}
            Err(e) => panic!("{e}"),
        }
    }
    dir.close().expect("close");
}

fn flatten(value: &[Vec<u8>]) -> Vec<u8> {
    value.concat()
}

fn s(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// One of `verify`'s writers: through the `RandomIndexWriter`, or past
/// 10,000 values straight to its `IndexWriter` (`w.w.addDocument`).
fn add(w: &mut RandomIndexWriter<'_>, real: bool, doc: &Document) -> Result<(), String> {
    if real {
        w.writer().add_fields_document(doc).map(drop).map_err(s)
    } else {
        w.add_document(doc).map(drop).map_err(s)
    }
}

fn delete_nukeme(w: &mut RandomIndexWriter<'_>, real: bool) -> Result<(), String> {
    let term = Term::new("nukeme", "yes");
    if real {
        w.writer()
            .delete_documents_by_term(&[term])
            .map(drop)
            .map_err(s)
    } else {
        w.delete_documents(term).map(drop).map_err(s)
    }
}

/// `verify(dir, docValues, ids, numDims, numIndexDims, numBytesPerDim,
/// expectExceptions)`: index the values (sometimes through a second
/// directory brought in by `addIndexes`), check each leaf's tree and
/// min/max, then random rectangle queries against the values. `ids` maps
/// each value to its document (`None`: one value per document).
///
/// # Errors
/// What the writers or readers fail with, as text.
pub fn verify_in(
    t: &mut LuceneTestCase,
    dir: &dyn Directory,
    doc_values: &[Vec<Vec<u8>>],
    ids: Option<&[i32]>,
    num_dims: usize,
    num_index_dims: usize,
    bytes: usize,
) -> Result<(), String> {
    let num_values = doc_values.len();
    let use_real_writer = num_values > 10_000;
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let env = writer_env(t);
    let side: Option<Box<dyn BaseDirectoryWrapper>>;
    let mut w = RandomIndexWriter::with_config(t.random(), dir, config, env).map_err(s)?;

    // The values' own min and max, per dimension.
    let mut expected_min: Vec<Vec<u8>> = doc_values[0].clone();
    let mut expected_max: Vec<Vec<u8>> = doc_values[0].clone();
    for value in &doc_values[1..] {
        for dim in 0..num_dims {
            if value[dim] < expected_min[dim] {
                expected_min[dim].clone_from(&value[dim]);
            }
            if value[dim] > expected_max[dim] {
                expected_max[dim].clone_from(&value[dim]);
            }
        }
    }

    // One time in five the values go to a second directory first, brought
    // in by addIndexes at a random value.
    let mut save_w = None;
    let add_indexes_at;
    if t.random().next_int_bounded(5) == 1 {
        side = Some(get_directory(t, num_values));
        let analyzer = mock_analyzer(t);
        let config = new_config(t, analyzer);
        let env = writer_env(t);
        let side_dir: &dyn Directory = &**side.as_ref().expect("side");
        let side_w =
            RandomIndexWriter::with_config(t.random(), side_dir, config, env).map_err(s)?;
        save_w = Some(std::mem::replace(&mut w, side_w));
        add_indexes_at = test_util::next_int(t.random(), 1, num_values as i32 - 1) as usize;
    } else {
        side = None;
        add_indexes_at = 0;
    }

    let mut field_type = FieldType::new();
    field_type
        .set_dimensions_with_index(num_dims as i32, num_index_dims as i32, bytes as i32)
        .map_err(s)?;
    field_type.freeze();

    let mut doc: Option<Document> = None;
    let mut last_id = -1;
    for (ord, value) in doc_values.iter().enumerate() {
        let id = ids.map_or(ord as i32, |ids| ids[ord]);
        if id != last_id {
            if let Some(d) = doc.take() {
                add(&mut w, use_real_writer, &d)?;
            }
            let mut d = Document::new();
            d.add(NumericDocValuesField::new("id", i64::from(id)));
            doc = Some(d);
        }
        let point =
            BinaryPoint::from_packed("field", flatten(value), field_type.clone()).map_err(s)?;
        doc.as_mut().expect("doc").add(point);
        last_id = id;

        if t.random().next_int_bounded(30) == 17 {
            // A document without the field.
            add(&mut w, use_real_writer, &Document::new())?;
        }
        if t.random().next_int_bounded(30) == 17 {
            // A document with the field, deleted later.
            let mut xdoc = Document::new();
            xdoc.add(
                BinaryPoint::from_packed("field", flatten(value), field_type.clone()).map_err(s)?,
            );
            xdoc.add(StringField::new("nukeme", "yes", Store::No));
            add(&mut w, use_real_writer, &xdoc)?;
            if t.random().next_int_bounded(5) == 1 {
                delete_nukeme(&mut w, use_real_writer)?;
            }
        }
        if ord >= add_indexes_at {
            if let Some(main) = save_w.take() {
                let side_w = std::mem::replace(&mut w, main);
                let side_dir: &dyn Directory = &**side.as_ref().expect("side");
                switch_index(t, side_w, side_dir, &mut w)?;
            }
        }
    }
    add(&mut w, false, &doc.expect("a last document"))?;
    w.delete_documents(Term::new("nukeme", "yes")).map_err(s)?;
    if t.random().next_boolean() {
        w.force_merge(1).map_err(s)?;
    }
    let r = w.get_reader().map_err(s)?;
    w.close().map_err(s)?;
    if let Some(side) = &side {
        side.close()?;
    }

    // Each document's id; deleted documents' too.
    let max_doc = r.max_doc() as usize;
    let mut doc_id_to_id = vec![0i32; max_doc];
    let mut bases = Vec::new();
    let mut base = 0usize;
    for leaf in r.segment_readers() {
        bases.push(base);
        if let Some(mut id_values) = leaf.numeric_doc_values("id").map_err(s)? {
            loop {
                let doc = next(id_values.as_mut());
                if doc == lucene_search::reader::NO_MORE_DOCS {
                    break;
                }
                doc_id_to_id[base + doc as usize] = id_values.long_value() as i32;
            }
        }
        base += leaf.max_doc() as usize;
    }

    // Min and max over the leaves.
    let width = num_index_dims * bytes;
    let mut min_values = vec![0xffu8; width];
    let mut max_values = vec![0u8; width];
    for leaf in r.segment_readers() {
        let Some(values) = leaf.point_values("field").map_err(s)? else {
            continue;
        };
        assert_size_of(t, leaf, "field", values.size(), num_index_dims, bytes)?;
        let (leaf_min, leaf_max) = (values.min_packed_value(), values.max_packed_value());
        for dim in 0..num_index_dims {
            let range = dim * bytes..(dim + 1) * bytes;
            if leaf_min[range.clone()] < min_values[range.clone()] {
                min_values[range.clone()].copy_from_slice(&leaf_min[range.clone()]);
            }
            if leaf_max[range.clone()] > max_values[range.clone()] {
                max_values[range.clone()].copy_from_slice(&leaf_max[range]);
            }
        }
    }
    for dim in 0..num_index_dims {
        let range = dim * bytes..(dim + 1) * bytes;
        assert_eq!(
            expected_min[dim],
            min_values[range.clone()],
            "min of dim {dim}"
        );
        assert_eq!(expected_max[dim], max_values[range], "max of dim {dim}");
    }

    let num_ids = ids.map_or(num_values, |ids| ids.last().map_or(0, |&l| l as usize + 1));
    let iters = t.at_least(100);
    for _ in 0..iters {
        let mut query_min = Vec::with_capacity(num_index_dims);
        let mut query_max = Vec::with_capacity(num_index_dims);
        for _ in 0..num_index_dims {
            let mut a = vec![0u8; bytes];
            t.random().next_bytes(&mut a);
            let mut b = vec![0u8; bytes];
            t.random().next_bytes(&mut b);
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            query_min.push(a);
            query_max.push(b);
        }
        let mut hits = vec![false; num_ids];
        for (leaf, &doc_base) in r.segment_readers().iter().zip(&bases) {
            let Some(values) = leaf.point_values("field").map_err(s)? else {
                continue;
            };
            let mut v = Rect {
                query_min: &query_min,
                query_max: &query_max,
                bytes,
                doc_base,
                live: leaf.live_docs(),
                doc_id_to_id: &doc_id_to_id,
                hits: &mut hits,
            };
            values.intersect(&mut v).map_err(s)?;
        }
        let mut expected = vec![false; num_ids];
        for (ord, value) in doc_values.iter().enumerate() {
            let matches =
                (0..num_index_dims).all(|d| value[d] >= query_min[d] && value[d] <= query_max[d]);
            if matches {
                expected[ids.map_or(ord, |ids| ids[ord] as usize)] = true;
            }
        }
        let fail_count = expected.iter().zip(&hits).filter(|(e, h)| e != h).count();
        assert_eq!(
            fail_count,
            0,
            "{fail_count} docs failed; {} docs succeeded",
            num_ids - fail_count
        );
    }
    Ok(())
}

/// `verify`'s rectangle visitor over the index dimensions, hits by id.
struct Rect<'a> {
    query_min: &'a [Vec<u8>],
    query_max: &'a [Vec<u8>],
    bytes: usize,
    doc_base: usize,
    live: Option<&'a FixedBitSet>,
    doc_id_to_id: &'a [i32],
    hits: &'a mut [bool],
}

impl Rect<'_> {
    fn dim<'p>(&self, packed: &'p [u8], dim: usize) -> &'p [u8] {
        &packed[dim * self.bytes..(dim + 1) * self.bytes]
    }
}

impl IntersectVisitor for Rect<'_> {
    fn visit(&mut self, doc_id: i32) {
        // FBS: `live` is the leaf's `maxDoc` long; its documents only.
        if self.live.is_none_or(|l| l.get(doc_id as usize)) {
            self.hits[self.doc_id_to_id[self.doc_base + doc_id as usize] as usize] = true;
        }
    }
    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        // FBS: as in `visit`.
        if self.live.is_some_and(|l| !l.get(doc_id as usize)) {
            return;
        }
        for d in 0..self.query_min.len() {
            let x = self.dim(packed, d);
            if x < self.query_min[d].as_slice() || x > self.query_max[d].as_slice() {
                return;
            }
        }
        self.hits[self.doc_id_to_id[self.doc_base + doc_id as usize] as usize] = true;
    }
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        let mut crosses = false;
        for d in 0..self.query_min.len() {
            let (min, max) = (self.dim(min_packed, d), self.dim(max_packed, d));
            if max < self.query_min[d].as_slice() || min > self.query_max[d].as_slice() {
                return Relation::CellOutsideQuery;
            } else if min < self.query_min[d].as_slice() || max > self.query_max[d].as_slice() {
                crosses = true;
            }
        }
        if crosses {
            Relation::CellCrossesQuery
        } else {
            Relation::CellInsideQuery
        }
    }
}

/// `switchIndex(w, dir, saveW)`: the side writer's documents into the main
/// one, through readers (see the module documentation) or `addIndexes`.
fn switch_index(
    t: &mut LuceneTestCase,
    mut w: RandomIndexWriter<'_>,
    dir: &dyn Directory,
    save_w: &mut RandomIndexWriter<'_>,
) -> Result<(), String> {
    if t.random().next_boolean() {
        // addIndexes(CodecReader...) or TestUtil.addIndexesSlowly.
        let _codec_readers = t.random().next_boolean();
        w.commit().map_err(s)?;
        save_w.writer().add_indexes_merged(&[dir]).map_err(s)?;
        w.close().map_err(s)
    } else {
        w.close().map_err(s)?;
        save_w.add_indexes(&[dir]).map(drop).map_err(s)
    }
}

/// The points under `node`, each checked against `node`'s cell.
fn subtree_count(
    reader: &PointsReader<'_>,
    node: &PointTreeNode,
    num_index_dims: usize,
    bytes: usize,
) -> Result<i64, String> {
    if node.is_leaf() {
        let mut v = CellCheck {
            min: node.min_packed(),
            max: node.max_packed(),
            num_index_dims,
            bytes,
            count: 0,
        };
        reader.visit_leaf(node, &mut v).map_err(s)?;
        Ok(v.count)
    } else {
        let (left, right) = reader
            .point_tree_children(node)
            .map_err(s)?
            .expect("an inner node has children");
        let mut count = 0;
        for child in [&left, &right] {
            // A child's points are in its own cell, which is in this one's.
            count += subtree_count(reader, child, num_index_dims, bytes)?;
        }
        Ok(count)
    }
}

/// Counts a leaf's points, each inside the leaf's cell.
struct CellCheck<'a> {
    min: &'a [u8],
    max: &'a [u8],
    num_index_dims: usize,
    bytes: usize,
    count: i64,
}

impl IntersectVisitor for CellCheck<'_> {
    fn compare(&mut self, _min: &[u8], _max: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }
    fn visit(&mut self, _doc_id: i32) {
        self.count += 1;
    }
    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        for d in 0..self.num_index_dims {
            let range = d * self.bytes..(d + 1) * self.bytes;
            assert!(
                packed[range.clone()] >= self.min[range.clone()]
                    && packed[range.clone()] <= self.max[range.clone()],
                "doc {doc_id}'s point is outside its leaf's cell in dim {d}"
            );
        }
        self.count += 1;
    }
}

/// `assertSize(values.getPointTree())` over `field` of `leaf`, whose
/// `PointValues.size()` is `size`.
fn assert_size_of(
    t: &mut LuceneTestCase,
    leaf: &SegmentReader,
    field: &str,
    size: i64,
    num_index_dims: usize,
    bytes: usize,
) -> Result<(), String> {
    let number = leaf
        .field_infos()
        .field_by_name(field)
        .expect("the field")
        .number;
    let reader = leaf.points_reader().map_err(s)?;
    let root = reader.point_tree(number).map_err(s)?;
    assert_eq!(size, assert_size(t, &reader, &root, num_index_dims, bytes)?);
    Ok(())
}

/// Counts the document ids `visitDocIDs` hands over.
struct IdCount(i64);

impl IntersectVisitor for IdCount {
    fn compare(&mut self, _min: &[u8], _max: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }
    fn visit(&mut self, _doc: i32) {
        self.0 += 1;
    }
    fn visit_with_value(&mut self, _doc: i32, _packed: &[u8]) {
        unreachable!("visitDocIDs decodes no value");
    }
}

/// `assertSize(tree)`: the node's `visitDocIDs` and `visitDocValues`
/// counts, in a random order, must both be its `size()`; then each child's
/// (after a random excursion below it), which must also add up.
fn assert_size(
    t: &mut LuceneTestCase,
    reader: &PointsReader<'_>,
    node: &PointTreeNode,
    num_index_dims: usize,
    bytes: usize,
) -> Result<i64, String> {
    let clone = node.clone();
    let tree_size = reader.point_tree_size(node).map_err(s)?;
    assert_eq!(reader.point_tree_size(&clone).map_err(s)?, tree_size);
    // Rarely continue with the clone.
    let node = if t.rarely() { &clone } else { node };
    let doc_ids = |reader: &PointsReader<'_>| -> Result<i64, String> {
        let mut ids = IdCount(0);
        reader.visit_doc_ids(node, &mut ids).map_err(s)?;
        Ok(ids.0)
    };
    let (ids, size) = if t.random().next_boolean() {
        let ids = doc_ids(reader)?;
        (ids, subtree_count(reader, node, num_index_dims, bytes)?)
    } else {
        let size = subtree_count(reader, node, num_index_dims, bytes)?;
        (doc_ids(reader)?, size)
    };
    assert_eq!(ids, size, "visitDocIDs and visitDocValues disagree");
    assert_eq!(
        ids, tree_size,
        "a node's visits are not its point_tree_size"
    );
    if let Some((left, right)) = reader.point_tree_children(node).map_err(s)? {
        let mut children = 0;
        for child in [&left, &right] {
            random_point_tree_navigation(t, reader, child, num_index_dims, bytes)?;
            children += assert_size(t, reader, child, num_index_dims, bytes)?;
        }
        assert_eq!(size, children, "a node's size is its children's");
    }
    Ok(size)
}

/// Whether `inner`'s cell lies within `outer`'s, dimension by dimension.
fn assert_cell_within(
    inner: &PointTreeNode,
    outer: &PointTreeNode,
    num_index_dims: usize,
    bytes: usize,
) {
    for d in 0..num_index_dims {
        let range = d * bytes..(d + 1) * bytes;
        assert!(inner.min_packed()[range.clone()] >= outer.min_packed()[range.clone()]);
        assert!(inner.max_packed()[range.clone()] <= outer.max_packed()[range]);
    }
}

/// `randomPointTreeNavigation(tree)`: down to a random depth and back.
fn random_point_tree_navigation(
    t: &mut LuceneTestCase,
    reader: &PointsReader<'_>,
    node: &PointTreeNode,
    num_index_dims: usize,
    bytes: usize,
) -> Result<(), String> {
    if t.random().next_boolean() {
        if let Some((left, right)) = reader.point_tree_children(node).map_err(s)? {
            assert_cell_within(&left, node, num_index_dims, bytes);
            random_point_tree_navigation(t, reader, &left, num_index_dims, bytes)?;
            if t.random().next_boolean() {
                assert_cell_within(&right, node, num_index_dims, bytes);
                random_point_tree_navigation(t, reader, &right, num_index_dims, bytes)?;
            }
        }
    }
    Ok(())
}

/// `PointRangeQuery`'s counting visitor for one exact value: a cell is
/// inside when its bounds are both the value.
struct ExactCount<'a> {
    packed: &'a [u8],
    count: i64,
}

impl IntersectVisitor for ExactCount<'_> {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        if min > self.packed || max < self.packed {
            Relation::CellOutsideQuery
        } else if min == self.packed && max == self.packed {
            Relation::CellInsideQuery
        } else {
            Relation::CellCrossesQuery
        }
    }
    fn visit(&mut self, _doc: i32) {
        unreachable!("pointCount hands over values");
    }
    fn visit_with_value(&mut self, _doc: i32, packed: &[u8]) {
        if packed == self.packed {
            self.count += 1;
        }
    }
}

/// The live documents whose `field` point is exactly `packed`: what
/// `searcher.count(IntPoint.newExactQuery(field, value))` counts. A leaf
/// without deletions whose field has one value per document is counted as
/// `Weight.count` does (`pointCount`, a cell inside the query adding its
/// `size()`); any other leaf by an `intersect` over its live documents.
pub fn count_exact(reader: &DirectoryReader, field: &str, packed: &[u8]) -> usize {
    let mut count = 0;
    for leaf in reader.segment_readers() {
        let Some(values) = leaf.point_values(field).expect("points") else {
            continue;
        };
        if leaf.live_docs().is_none() && values.size() == i64::from(values.doc_count()) {
            let number = leaf
                .field_infos()
                .field_by_name(field)
                .expect("the field")
                .number;
            let points = leaf.points_reader().expect("points reader");
            let mut v = ExactCount { packed, count: 0 };
            let inside = points.count_points(number, &mut v).expect("pointCount");
            count += (inside + v.count) as usize;
            continue;
        }
        let bytes = packed.len();
        let min = vec![packed.to_vec()];
        let max = vec![packed.to_vec()];
        let ids: Vec<i32> = (0..leaf.max_doc()).collect();
        let mut seen = vec![false; leaf.max_doc() as usize];
        let mut v = Rect {
            query_min: &min,
            query_max: &max,
            bytes,
            doc_base: 0,
            live: leaf.live_docs(),
            doc_id_to_id: &ids,
            hits: &mut seen,
        };
        values.intersect(&mut v).expect("intersect");
        count += seen.iter().filter(|&&hit| hit).count();
    }
    count
}

/// `testAddIndexes`: two indexes numbering `int1` and `int2` differently,
/// added and merged.
pub fn test_add_indexes(t: &mut LuceneTestCase) {
    let dir1 = t.new_directory();
    let mut w = new_riw(t, &*dir1);
    let mut doc = Document::new();
    doc.add(IntPoint::new("int1", &[17]).expect("IntPoint"));
    w.add_document(&doc).expect("addDocument");
    let mut doc = Document::new();
    doc.add(IntPoint::new("int2", &[42]).expect("IntPoint"));
    w.add_document(&doc).expect("addDocument");
    w.close().expect("close");

    // Different field number assignments.
    let dir2 = t.new_directory();
    let mut w = new_riw(t, &*dir2);
    let mut doc = Document::new();
    doc.add(IntPoint::new("int2", &[42]).expect("IntPoint"));
    w.add_document(&doc).expect("addDocument");
    let mut doc = Document::new();
    doc.add(IntPoint::new("int1", &[17]).expect("IntPoint"));
    w.add_document(&doc).expect("addDocument");
    w.close().expect("close");

    let dir = t.new_directory();
    let mut w = new_riw(t, &*dir);
    w.add_indexes(&[&*dir1, &*dir2]).expect("addIndexes");
    riw_force_merge(&mut w, 1);
    let r = w.get_reader().expect("getReader");
    let int1 = IntPoint::pack(&[17]).expect("pack");
    let int2 = IntPoint::pack(&[42]).expect("pack");
    assert_eq!(2, count_exact(&r, "int1", &int1));
    assert_eq!(2, count_exact(&r, "int2", &int2));
    drop(r);
    w.close().expect("close");
    dir.close().expect("close");
    dir1.close().expect("close");
    dir2.close().expect("close");
}

/// `testMergeMissing` (LUCENE-7491): a merge of a segment with points in
/// `int` and one with points in `id` only.
pub fn test_merge_missing(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let env = writer_env(t);
    let mut w =
        RandomIndexWriter::with_config(t.random(), &*dir, config, env).expect("RandomIndexWriter");
    // Java sets it on the live config after the writer opened.
    w.writer()
        .set_max_buffered_docs(2)
        .expect("setMaxBufferedDocs");
    for i in 0..2 {
        let mut doc = Document::new();
        doc.add(IntPoint::new("int", &[i]).expect("IntPoint"));
        w.add_document(&doc).expect("addDocument");
    }
    // One segment of two documents, with points.
    let mut doc = Document::new();
    doc.add(IntPoint::new("id", &[0]).expect("IntPoint"));
    w.add_document(&doc).expect("addDocument");
    // Another segment, whose points are in id.
    riw_force_merge(&mut w, 1);
    w.close().expect("close");
    dir.close().expect("close");
}

/// The visitor every `estimateDocCount` test passes: inside everywhere.
fn inside_doc_count(size: i64, doc_count: i32, estimated_point_count: i64) -> i64 {
    estimate_doc_count(estimated_point_count, size, doc_count)
}

/// `testDocCountEdgeCases`.
pub fn test_doc_count_edge_cases(_t: &mut LuceneTestCase) {
    assert_eq!(1, inside_doc_count(i64::MAX, 1, i64::MAX));
    assert_eq!(1, inside_doc_count(i64::MAX, 1, 1));
    assert_eq!(
        i64::from(i32::MAX),
        inside_doc_count(i64::MAX, i32::MAX, i64::MAX)
    );
    assert_eq!(
        i64::from(i32::MAX),
        inside_doc_count(i64::MAX, i32::MAX, i64::MAX / 2)
    );
    assert_eq!(1, inside_doc_count(i64::MAX, i32::MAX, 1));
}

/// `testRandomDocCount`.
pub fn test_random_doc_count(t: &mut LuceneTestCase) {
    for _ in 0..100 {
        let size = test_util::next_long(t.random(), 1, i64::MAX);
        let max_doc = i32::try_from(size).unwrap_or(i32::MAX);
        let doc_count = test_util::next_int(t.random(), 1, max_doc);
        let estimated_point_count = test_util::next_long(t.random(), 0, size);
        let docs = inside_doc_count(size, doc_count, estimated_point_count);
        assert!(docs <= estimated_point_count);
        assert!(docs <= i64::from(max_doc));
        assert!(docs >= estimated_point_count / (size / i64::from(doc_count)));
    }
}

/// `testMismatchedFields`: two indexes numbering `f` and `g` differently,
/// merged into one segment whose points keep their documents' order.
pub fn test_mismatched_fields(t: &mut LuceneTestCase) {
    let f = LongPoint::new("f", &[1]).expect("LongPoint");
    let g = LongPoint::new("g", &[42, 43]).expect("LongPoint");

    // MismatchedCodecReader: the fields numbered the other way around.
    let dir1 = t.new_directory();
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let mut w1 = writer_with(&*dir1, &config);
    let mut doc1 = Document::new();
    doc1.add(g.clone());
    doc1.add(f.clone());
    w1.add_fields_document(&doc1).expect("addDocument");

    let dir2 = t.new_directory();
    let analyzer = mock_analyzer(t);
    let mut config = new_config(t, analyzer);
    config.merge_scheduler = MergeSchedulerChoice::Serial;
    let mut w2 = writer_with(&*dir2, &config);
    let mut doc = Document::new();
    doc.add(f);
    doc.add(g);
    w2.add_fields_document(&doc).expect("addDocument");
    w2.commit().expect("commit");

    w1.commit().expect("commit");
    w1.close().expect("close");
    w2.add_indexes_merged(&[&*dir1]).expect("addIndexes");
    force_merge(&mut w2, 1);
    w2.close().expect("close");
    let reader = DirectoryReader::open(&*dir2).expect("open");
    let leaf = only_leaf(&reader);
    assert_eq!(2, leaf.max_doc());

    for (field, want) in [
        ("f", LongPoint::pack(&[1]).expect("pack")),
        ("g", LongPoint::pack(&[42, 43]).expect("pack")),
    ] {
        let points = leaf.point_values(field).expect("points").expect(field);
        assert_eq!(2, points.size());
        let mut v = InOrder {
            want: &want,
            expected_doc: 0,
        };
        points.intersect(&mut v).expect("intersect");
        assert_eq!(2, v.expected_doc);
    }
    drop(reader);
    dir1.close().expect("close");
    dir2.close().expect("close");
}

/// `testMismatchedFields`' visitor: the one value, documents in order.
struct InOrder<'a> {
    want: &'a [u8],
    expected_doc: i32,
}

impl IntersectVisitor for InOrder<'_> {
    fn compare(&mut self, _min: &[u8], _max: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }
    fn visit(&mut self, _doc_id: i32) {
        panic!("UnsupportedOperationException: visit(docID)");
    }
    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        assert_eq!(self.want, packed);
        assert_eq!(self.expected_doc, doc_id);
        self.expected_doc += 1;
    }
}

/// Every `BasePointsFormatTestCase` test this port runs by default.
pub const TESTS: &[SuiteTest] = &[
    ("testBasic", test_basic),
    ("testMerge", test_merge),
    (
        "testAllPointDocsDeletedInSegment",
        test_all_point_docs_deleted_in_segment,
    ),
    ("testMultiValued", test_multi_valued),
    ("testAllEqual", test_all_equal),
    ("testOneDimEqual", test_one_dim_equal),
    ("testOneDimTwoValues", test_one_dim_two_values),
    ("testBigIntNDims", test_big_int_n_dims),
    ("testRandomBinaryTiny", test_random_binary_tiny),
    ("testRandomBinaryMedium", test_random_binary_medium),
    ("testAddIndexes", test_add_indexes),
    ("testMergeMissing", test_merge_missing),
    ("testDocCountEdgeCases", test_doc_count_edge_cases),
    ("testRandomDocCount", test_random_doc_count),
    ("testMismatchedFields", test_mismatched_fields),
];
