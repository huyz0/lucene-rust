//! `BaseKnnVectorsFormatTestCase`: KNN vector fields written through
//! `IndexWriter` with the default codec (`Lucene99HnswVectorsFormat` behind
//! `PerFieldKnnVectorsFormat`) and read back as the flat vectors, their
//! ordinals and documents, and nearest-neighbour searches on the stored graph.
//!
//! Differs:
//!
//! - A leaf's `getFloatVectorValues`/`getByteVectorValues` is the field's
//!   flat reader (`flat_vectors_reader_for`), Java's `CodecReader`'s
//!   `KnnVectorsReader`. Its `iterator()` is the ordinals in order with
//!   [`ord_to_doc`](lucene_codecs::vectors::FloatVectorValues::ord_to_doc)
//!   (`nextDoc`, `index()`), and `advance(target)` the first document from
//!   `target` the doc-to-ordinal cursor has an ordinal for.
//! - `scorer(target)` is the flat scorer over ordinals with no iterator of
//!   its own, so `testFloatVectorScorerIteration` scores every ordinal
//!   (each score non-negative) and checks the values' documents ascend,
//!   where Java walks the scorer's iterator against the values'; an empty
//!   field has a scorer too (Java's is `null`).
//! - `LeafReader.searchNearestVectors` opens the default format's graph
//!   beside the flat vectors, through a compound reader when necessary.
//!   A codec collector reports visits and early termination directly;
//!   there is no query-level exact-search fallback. `assertOffHeapByteSize`
//!   and `testMergeOffHeapByteSizeMaps` (`getOffHeapByteSize`) have no counterpart.
//! - A field's schema cannot change within a segment either: the writer
//!   keeps one schema per index, so `testIllegalDimChangeTwoDocs` and
//!   `testIllegalSimilarityFunctionChange` get
//!   `FieldInfo.verifySameVectorOptions`' message in the same segment, not
//!   `FieldSchema`'s "Inconsistency ... of doc \[1\]".
//! - `testWriterRamEstimate`/`testWriterByteVectorRamEstimate` measure
//!   [`IndexWriter::ram_bytes_used`] around the vectors buffered for one
//!   field: the KNN writers keep no account of their own.
//! - `addIndexes(CodecReader...)` and `TestUtil.addIndexesSlowly` are
//!   [`IndexWriter::add_indexes_merged`] over the source directory;
//!   `testMismatchedFields`' `MismatchedCodecReader` is the first index
//!   adding its fields in the other order.
//! - Vector fields are immutable values with an explicit similarity:
//!   `testFieldConstructor` passes `EUCLIDEAN`, `testFieldSetValue` and
//!   the setters of `testInvalidKnnVectorFieldUsage` have nothing to call,
//!   and `testIndexedValueNotAliased` cannot alias (each field owns its
//!   vector) but runs its three documents.
//! - `testDifferentCodecs1`/`2` need a `SimpleText` segment written through
//!   `IndexWriter`, which the writer cannot produce;
//!   `testReadQuantizedVectorWithEmptyRawVectors` is skipped, as in Java,
//!   since the default format has no float-vector fallback.
//! - `testCheckIndexIncludesVectors` reads CheckIndex's vector counts, not
//!   its printed `test: vectors...` line.
//! - `testMergeStability` is suppressed, as in Java (`mergeIsStable`).

use lucene_codecs::field_infos::{FieldInfo, VectorEncoding, VectorSimilarityFunction};
use lucene_codecs::hnsw::{KnnCollector, VectorScorer};
use lucene_codecs::hnsw_vectors::{self, HnswVectorsReader, SearchOptions};
use lucene_codecs::vectors::{ByteVectorValues, FloatVectorValues};
use lucene_index::buffered_updates::Term;
use lucene_index::compound_reader::CompoundReader;
use lucene_index::document::{
    Document, Field, IndexableField, KnnByteVectorField, KnnFloatVectorField,
    NumericDocValuesField, Store, StoredValue, StringField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::{self, IndexSortField, IndexSortKind, NumericSortKey};
use lucene_search::collector::{ScoreDoc, TotalHits, TotalHitsRelation};
use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_store::Directory;
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::java_random::JavaRandom;
use lucene_util::vector_util;

use super::base_index_file_format::{
    force_merge, get, leaf_stored, mock_analyzer, new_config, new_riw, new_writer, only_leaf,
    plain_writer, riw_force_merge, set_index_sort, writer_with, IndexFileFormatSuite, SuiteTest,
};
use super::index_writer_config::{new_log_merge_policy, Draws, LiveMergePolicy, PolicyKind};
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::test_util;

/// `DocIdSetIterator.NO_MORE_DOCS`.
const NO_MORE_DOCS: i32 = i32::MAX;

/// `KnnVectorsFormat.DEFAULT_MAX_DIMENSIONS`: what
/// `Codec.getDefault().knnVectorsFormat().getMaxDimensions(field)` returns.
const MAX_DIMENSIONS: usize = 1024;

/// The KNN suite's `addRandomFields`: one `v2` vector of 30 dimensions, in
/// the encoding and similarity the test drew in `@Before init()`.
#[derive(Debug, Clone, Copy)]
pub struct KnnVectorsSuite {
    /// `vectorEncoding`.
    pub encoding: VectorEncoding,
    /// `similarityFunction`.
    pub similarity: VectorSimilarityFunction,
}

impl KnnVectorsSuite {
    /// `init()`: `randomVectorEncoding()`, then `randomSimilarity()`.
    pub fn new(t: &mut LuceneTestCase) -> Self {
        let encoding = random_vector_encoding(t.random());
        let similarity = random_similarity(t.random());
        KnnVectorsSuite {
            encoding,
            similarity,
        }
    }
}

impl IndexFileFormatSuite for KnnVectorsSuite {
    fn add_random_fields(&self, t: &mut LuceneTestCase, doc: &mut Document) {
        match self.encoding {
            VectorEncoding::Byte => doc.add(
                KnnByteVectorField::new("v2", random_vector8(t.random(), 30), self.similarity)
                    .expect("KnnByteVectorField"),
            ),
            VectorEncoding::Float32 => doc.add(
                KnnFloatVectorField::new(
                    "v2",
                    random_normalized_vector(t.random(), 30),
                    self.similarity,
                )
                .expect("KnnFloatVectorField"),
            ),
        }
    }

    /// `mergeIsStable()`: merges for KNN graphs are not stable, because of
    /// the connected-components logic.
    fn merge_is_stable(&self) -> bool {
        false
    }
}

/// Every `VectorSimilarityFunction`, in declaration order (`values()`).
const SIMILARITIES: [VectorSimilarityFunction; 4] = [
    VectorSimilarityFunction::Euclidean,
    VectorSimilarityFunction::DotProduct,
    VectorSimilarityFunction::Cosine,
    VectorSimilarityFunction::MaximumInnerProduct,
];

/// Every `VectorEncoding`, in declaration order (`values()`).
const ENCODINGS: [VectorEncoding; 2] = [VectorEncoding::Byte, VectorEncoding::Float32];

/// `randomSimilarity()`.
pub fn random_similarity(r: &mut JavaRandom) -> VectorSimilarityFunction {
    SIMILARITIES[r.next_int_bounded(SIMILARITIES.len() as i32) as usize]
}

/// `randomVectorEncoding()`.
pub fn random_vector_encoding(r: &mut JavaRandom) -> VectorEncoding {
    ENCODINGS[r.next_int_bounded(ENCODINGS.len() as i32) as usize]
}

/// `randomVector(dim)`: uniform components, drawn again until the vector is
/// not all zeros.
pub fn random_vector(r: &mut JavaRandom, dim: usize) -> Vec<f32> {
    assert!(dim > 0);
    let mut v = vec![0f32; dim];
    let mut square_sum = 0.0f64;
    while square_sum == 0.0 {
        square_sum = 0.0;
        for x in &mut v {
            *x = r.next_float();
            square_sum += f64::from(*x * *x);
        }
    }
    v
}

/// `randomNormalizedVector(dim)`.
pub fn random_normalized_vector(r: &mut JavaRandom, dim: usize) -> Vec<f32> {
    let mut v = random_vector(r, dim);
    vector_util::l2normalize(&mut v, true).expect("a non-zero vector");
    v
}

/// `randomVector8(dim)`: a normalized vector scaled to `(byte) (v * 127)`.
pub fn random_vector8(r: &mut JavaRandom, dim: usize) -> Vec<u8> {
    random_normalized_vector(r, dim)
        .into_iter()
        .map(|x| (x * 127.0) as i8 as u8)
        .collect()
}

/// An even dimension of at least `atLeast(10)`.
fn even_dimension(t: &mut LuceneTestCase) -> usize {
    let dimension = t.at_least(10);
    (dimension + dimension % 2) as usize
}

fn float_field(name: &str, v: Vec<f32>, sim: VectorSimilarityFunction) -> KnnFloatVectorField {
    KnnFloatVectorField::new(name, v, sim).expect("KnnFloatVectorField")
}

fn byte_field(name: &str, v: Vec<u8>, sim: VectorSimilarityFunction) -> KnnByteVectorField {
    KnnByteVectorField::new(name, v, sim).expect("KnnByteVectorField")
}

fn one_field(field: impl IndexableField + 'static) -> Document {
    let mut doc = Document::new();
    doc.add(field);
    doc
}

fn add(w: &mut IndexWriter<'_>, doc: &Document) {
    w.add_fields_document(doc).expect("addDocument");
}

fn add_err(w: &mut IndexWriter<'_>, doc: &Document) -> String {
    w.add_fields_document(doc)
        .expect_err("addDocument must fail")
        .to_string()
}

/// The vector a document carries in `add(iw, field, id, sortkey, vector)`.
enum Vector {
    Float(Vec<f32>),
    Byte(Vec<u8>),
}

/// `add(iw, field, id, sortkey, vector, similarityFunction)`: the vector (if
/// any), `sortkey` doc values and a stored `id`, as an update of `id`.
fn add_vector(
    w: &mut IndexWriter<'_>,
    field: &str,
    id: i32,
    sort_key: i32,
    vector: Option<Vector>,
    sim: VectorSimilarityFunction,
) {
    let mut doc = Document::new();
    match vector {
        Some(Vector::Float(v)) => doc.add(float_field(field, v, sim)),
        Some(Vector::Byte(v)) => doc.add(byte_field(field, v, sim)),
        None => {}
    }
    doc.add(NumericDocValuesField::new("sortkey", i64::from(sort_key)));
    let id = id.to_string();
    doc.add(StringField::new("id", id.clone(), Store::Yes));
    w.update_fields_documents(Term::new("id", id), &[doc])
        .expect("updateDocument");
}

/// `add(iw, field, id, vector, similarityFunction)`: a random sort key.
fn add_random_key(
    t: &mut LuceneTestCase,
    w: &mut IndexWriter<'_>,
    field: &str,
    id: i32,
    vector: Option<Vector>,
    sim: VectorSimilarityFunction,
) {
    let key = t.random().next_int_bounded(100);
    add_vector(w, field, id, key, vector, sim);
}

/// `iwc.setIndexSort(new Sort(new SortField("sortkey", SortField.Type.INT)))`.
fn sort_by_key(w: &mut IndexWriter<'_>) {
    set_index_sort(
        w,
        &[IndexSortField {
            field: "sortkey".to_string(),
            reverse: false,
            kind: IndexSortKind::Numeric(NumericSortKey::Int(None)),
        }],
    );
}

/// `DirectoryReader.open(w)`: the writer's changes committed, and a reader
/// over the commit.
fn open(w: &mut IndexWriter<'_>, dir: &dyn Directory) -> DirectoryReader {
    w.commit().expect("commit");
    DirectoryReader::open(dir).expect("open a reader")
}

/// `leaf.getFieldInfos().fieldInfo(field)` when it indexes vectors.
fn vector_field<'a>(leaf: &'a SegmentReader, field: &str) -> Option<&'a FieldInfo> {
    leaf.field_infos()
        .fields
        .iter()
        .find(|f| f.name == field && f.vector_dimension > 0)
}

/// `leaf.getFloatVectorValues(field)`.
pub fn float_values<'a>(leaf: &'a SegmentReader, field: &str) -> Option<FloatVectorValues<'a>> {
    let fi = vector_field(leaf, field)?;
    if fi.vector_encoding != VectorEncoding::Float32 {
        return None;
    }
    let reader = leaf
        .flat_vectors_reader_for(fi.number)
        .expect("open the vectors")
        .expect("a vector field has a vectors reader");
    Some(
        reader
            .float_vector_values(fi.number)
            .expect("getFloatVectorValues"),
    )
}

/// `leaf.getByteVectorValues(field)`.
pub fn byte_values<'a>(leaf: &'a SegmentReader, field: &str) -> Option<ByteVectorValues<'a>> {
    let fi = vector_field(leaf, field)?;
    if fi.vector_encoding != VectorEncoding::Byte {
        return None;
    }
    let reader = leaf
        .flat_vectors_reader_for(fi.number)
        .expect("open the vectors")
        .expect("a vector field has a vectors reader");
    Some(
        reader
            .byte_vector_values(fi.number)
            .expect("getByteVectorValues"),
    )
}

/// `iterator()` run to `NO_MORE_DOCS`: each document with its ordinal
/// (`index()`), in order.
fn float_docs(values: &FloatVectorValues<'_>) -> Vec<(i32, i32)> {
    docs_of(values.size(), |ord| {
        values.ord_to_doc(ord).expect("ordToDoc")
    })
}

fn byte_docs(values: &ByteVectorValues<'_>) -> Vec<(i32, i32)> {
    docs_of(values.size(), |ord| {
        values.ord_to_doc(ord).expect("ordToDoc")
    })
}

fn docs_of(size: i32, ord_to_doc: impl Fn(i32) -> i32) -> Vec<(i32, i32)> {
    let docs: Vec<(i32, i32)> = (0..size).map(|ord| (ord_to_doc(ord), ord)).collect();
    assert!(
        docs.windows(2).all(|w| w[0].0 < w[1].0),
        "a vector iterator's documents ascend: {docs:?}"
    );
    docs
}

fn float_vector(values: &FloatVectorValues<'_>, ord: i32) -> Vec<f32> {
    values.vector(ord).expect("vectorValue")
}

fn byte_vector(values: &ByteVectorValues<'_>, ord: i32) -> Vec<u8> {
    values.vector(ord).expect("vectorValue").to_vec()
}

/// The stored `id` of `doc`, as a string.
fn stored_id(leaf: &SegmentReader, doc: i32) -> String {
    let fields = leaf_stored(leaf, doc);
    match get(&fields, "id") {
        Some(StoredValue::String(s)) => s.clone(),
        Some(StoredValue::Int(i)) => i.to_string(),
        other => panic!("doc {doc} has no stored id: {other:?}"),
    }
}

fn is_live(leaf: &SegmentReader, doc: i32) -> bool {
    leaf.live_docs().is_none_or(|l| l.get(doc as usize))
}

/// `TopDocs` returned by a leaf's vector search. `total_hits` counts scored
/// vectors, rather than the number of retained hits.
pub struct NearestVectorsResult {
    pub total_hits: TotalHits,
    pub score_docs: Vec<ScoreDoc>,
}

/// Open the default format's graph using the field's per-format suffix.
/// The test framework owns these temporary inputs, as Java's leaf reader does.
fn with_graph<R>(
    dir: &dyn Directory,
    leaf: &SegmentReader,
    field: &str,
    f: impl FnOnce(&HnswVectorsReader<'_>, i32) -> R,
) -> R {
    let fi = vector_field(leaf, field).expect("a vector field");
    let attr = |name: &str| {
        &fi.attributes
            .iter()
            .find(|(key, _)| key == name)
            .expect("per-field vector format attribute")
            .1
    };
    assert_eq!(
        attr("PerFieldKnnVectorsFormat.format"),
        "Lucene99HnswVectorsFormat"
    );
    let suffix = format!(
        "{}_{}",
        attr("PerFieldKnnVectorsFormat.format"),
        attr("PerFieldKnnVectorsFormat.suffix")
    );
    let si = dir
        .open(&format!("{}.si", leaf.segment_name))
        .expect("segment info");
    let info = segment_info::parse(&si, &leaf.segment_id()).expect("read segment info");
    let compound = info.is_compound_file.then(|| {
        CompoundReader::open(dir, &leaf.segment_name, &leaf.segment_id())
            .expect("getCompoundReader")
    });
    let source: &dyn Directory = compound.as_ref().map_or(dir, |c| c);
    let prefix = format!("{}_{suffix}", leaf.segment_name);
    let meta = source
        .open(&format!("{prefix}.vem"))
        .expect("open graph metadata");
    let index = source
        .open(&format!("{prefix}.vex"))
        .expect("open graph data");
    let graph = HnswVectorsReader::open(&meta, &index, &leaf.segment_id(), &suffix)
        .expect("open HNSW reader");
    f(&graph, fi.number)
}

/// `LeafReader.searchNearestVectors`, using Java's collector/codec path.
fn search_vectors<S: VectorScorer>(
    reader: &HnswVectorsReader<'_>,
    field: i32,
    leaf: &SegmentReader,
    scorer: &mut S,
    ord_to_doc: impl Fn(i32) -> i32,
    k: usize,
    visited_limit: u64,
) -> NearestVectorsResult {
    let graph = reader.graph(field).expect("getGraph");
    let accept = leaf.live_docs().map(|live| {
        let mut bits = FixedBitSet::new(scorer.max_ord() as usize);
        let bits_len = bits.len();
        for ord in 0..bits_len {
            let doc = ord_to_doc(ord as i32) as usize;
            let live_len = live.len();
            assert!(doc < live_len);
            if live.get(doc) {
                bits.set(ord);
            }
        }
        bits
    });
    let mut collector = KnnCollector::new(k, visited_limit);
    hnsw_vectors::search_with(
        scorer,
        graph.as_ref(),
        &mut collector,
        SearchOptions {
            accept_ords: accept.as_ref(),
            // Lucene's DenseLiveDocs/SparseLiveDocs are Bits, not BitSet:
            // AcceptDocs.cost() returns maxDoc even when there are deletions.
            ..SearchOptions::default()
        },
    )
    .expect("searchNearestVectors");
    let total_hits = TotalHits {
        value: collector.visited_count(),
        relation: if collector.early_terminated() {
            TotalHitsRelation::GreaterThanOrEqualTo
        } else {
            TotalHitsRelation::EqualTo
        },
    };
    let score_docs = collector
        .top_docs()
        .into_iter()
        .map(|(ord, score)| ScoreDoc {
            doc_id: ord_to_doc(ord),
            score,
        })
        .collect();
    NearestVectorsResult {
        total_hits,
        score_docs,
    }
}

/// Java: `LeafReader.searchNearestVectors(field, float[], k, liveDocs, visitedLimit)`.
///
/// # Panics
/// Invalid test input, a non-default vector format, or unreadable vector files.
pub fn search_nearest_float(
    dir: &dyn Directory,
    leaf: &SegmentReader,
    field: &str,
    target: &[f32],
    k: usize,
    visited_limit: u64,
) -> NearestVectorsResult {
    let Some(values) = float_values(leaf, field) else {
        return empty_nearest_vectors();
    };
    let k = k.min(values.size() as usize);
    if k == 0 {
        return empty_nearest_vectors();
    }
    let mut scorer = values.scorer(target).expect("float scorer");
    with_graph(dir, leaf, field, |reader, number| {
        search_vectors(
            reader,
            number,
            leaf,
            &mut scorer,
            |ord| values.ord_to_doc(ord).expect("ordToDoc"),
            k,
            visited_limit,
        )
    })
}

/// Java: `LeafReader.searchNearestVectors(field, byte[], k, liveDocs, visitedLimit)`.
///
/// # Panics
/// Invalid test input, a non-default vector format, or unreadable vector files.
pub fn search_nearest_byte(
    dir: &dyn Directory,
    leaf: &SegmentReader,
    field: &str,
    target: &[u8],
    k: usize,
    visited_limit: u64,
) -> NearestVectorsResult {
    let Some(values) = byte_values(leaf, field) else {
        return empty_nearest_vectors();
    };
    let k = k.min(values.size() as usize);
    if k == 0 {
        return empty_nearest_vectors();
    }
    let mut scorer = values.scorer(target).expect("byte scorer");
    with_graph(dir, leaf, field, |reader, number| {
        search_vectors(
            reader,
            number,
            leaf,
            &mut scorer,
            |ord| values.ord_to_doc(ord).expect("ordToDoc"),
            k,
            visited_limit,
        )
    })
}

fn empty_nearest_vectors() -> NearestVectorsResult {
    NearestVectorsResult {
        total_hits: TotalHits {
            value: 0,
            relation: TotalHitsRelation::EqualTo,
        },
        score_docs: Vec::new(),
    }
}

/// `leaf.searchNearestVectors(field, target, k, liveDocs, MAX_VALUE)`.
fn search_nearest(
    dir: &dyn Directory,
    leaf: &SegmentReader,
    field: &str,
    target: Vec<f32>,
    k: usize,
) -> Vec<ScoreDoc> {
    search_nearest_float(dir, leaf, field, &target, k, i32::MAX as u64).score_docs
}

/// `testFieldConstructor`.
pub fn test_field_constructor(_t: &mut LuceneTestCase) {
    let v = vec![0f32; 1];
    let field = float_field("f", v.clone(), VectorSimilarityFunction::Euclidean);
    assert_eq!(1, field.field_type().vector_dimension());
    assert_eq!(
        VectorSimilarityFunction::Euclidean,
        field.field_type().vector_similarity_function()
    );
    assert_eq!(v, field.vector());
}

/// `testFieldConstructorExceptions`: what a Rust caller can pass wrongly.
pub fn test_field_constructor_exceptions(_t: &mut LuceneTestCase) {
    let e = KnnFloatVectorField::new("f", Vec::new(), VectorSimilarityFunction::Euclidean)
        .expect_err("an empty vector");
    assert!(
        e.to_string().contains("cannot index an empty vector"),
        "{e}"
    );
    assert!(KnnByteVectorField::new("f", Vec::new(), VectorSimilarityFunction::Euclidean).is_err());
    assert!(
        KnnFloatVectorField::new("f", vec![f32::NAN], VectorSimilarityFunction::Euclidean).is_err()
    );
}

/// The vector-shape change message `FieldInfo.verifySameVectorOptions`
/// throws.
fn change_message(from: (usize, &str), to: (usize, &str)) -> String {
    format!(
        "cannot change field \"f\" from vector dimension={}, vector encoding=FLOAT32, vector \
         similarity function={} to inconsistent vector dimension={}, vector encoding=FLOAT32, \
         vector similarity function={}",
        from.0, from.1, to.0, to.1
    )
}

/// `testIllegalDimChangeTwoDocs`: in one segment and across two.
pub fn test_illegal_dim_change_two_docs(t: &mut LuceneTestCase) {
    for commit in [false, true] {
        let dir = t.new_directory();
        let mut w = new_writer(t, &*dir);
        let dot = VectorSimilarityFunction::DotProduct;
        add(&mut w, &one_field(float_field("f", vec![0.0; 4], dot)));
        if commit {
            w.commit().expect("commit");
        }
        let e = add_err(&mut w, &one_field(float_field("f", vec![0.0; 6], dot)));
        assert!(
            e.contains(&change_message((4, "DOT_PRODUCT"), (6, "DOT_PRODUCT"))),
            "{e}"
        );
    }
}

/// `testIllegalSimilarityFunctionChange`: in one segment and across two.
pub fn test_illegal_similarity_function_change(t: &mut LuceneTestCase) {
    for commit in [false, true] {
        let dir = t.new_directory();
        let mut w = new_writer(t, &*dir);
        add(
            &mut w,
            &one_field(float_field(
                "f",
                vec![0.0; 4],
                VectorSimilarityFunction::DotProduct,
            )),
        );
        if commit {
            w.commit().expect("commit");
        }
        let e = add_err(
            &mut w,
            &one_field(float_field(
                "f",
                vec![0.0; 4],
                VectorSimilarityFunction::Euclidean,
            )),
        );
        assert!(
            e.contains(&change_message((4, "DOT_PRODUCT"), (4, "EUCLIDEAN"))),
            "{e}"
        );
    }
}

/// `testIllegalDimChangeTwoWriters` and
/// `testIllegalSimilarityFunctionChangeTwoWriters`: the second writer reads
/// the first's schema.
fn change_two_writers(t: &mut LuceneTestCase, dim: usize, sim: VectorSimilarityFunction) {
    let dir = t.new_directory();
    let dot = VectorSimilarityFunction::DotProduct;
    {
        let mut w = new_writer(t, &*dir);
        add(&mut w, &one_field(float_field("f", vec![0.0; 4], dot)));
        w.close().expect("close");
    }
    let mut w2 = new_writer(t, &*dir);
    let e = add_err(&mut w2, &one_field(float_field("f", vec![0.0; dim], sim)));
    let name = if sim == dot {
        "DOT_PRODUCT"
    } else {
        "EUCLIDEAN"
    };
    assert!(
        e.contains(&change_message((4, "DOT_PRODUCT"), (dim, name))),
        "{e}"
    );
}

/// `testIllegalDimChangeTwoWriters`.
pub fn test_illegal_dim_change_two_writers(t: &mut LuceneTestCase) {
    change_two_writers(t, 2, VectorSimilarityFunction::DotProduct);
}

/// `testIllegalSimilarityFunctionChangeTwoWriters`.
pub fn test_illegal_similarity_function_change_two_writers(t: &mut LuceneTestCase) {
    change_two_writers(t, 4, VectorSimilarityFunction::Euclidean);
}

/// `testMergingWithDifferentKnnFields` and its byte twin: two segments,
/// each with its own vector field, force-merged without a merge error.
fn merging_with_different_fields(t: &mut LuceneTestCase, bytes: bool) {
    let dir = t.new_directory();
    let mut w = plain_writer(&*dir);
    let euclidean = VectorSimilarityFunction::Euclidean;
    for field in ["field", "otherVector"] {
        for i in 0..10u8 {
            let doc = if bytes {
                one_field(byte_field(field, vec![i; 4], euclidean))
            } else {
                let x = f32::from(i);
                let v = if field == "field" {
                    vec![x, x + 1.0, x + 2.0, x + 3.0]
                } else {
                    vec![x; 4]
                };
                one_field(float_field(field, v, euclidean))
            };
            add(&mut w, &doc);
        }
        w.commit().expect("commit");
    }
    force_merge(&mut w, 1);
    let r = open(&mut w, &*dir);
    let leaf = only_leaf(&r);
    for field in ["field", "otherVector"] {
        let size = if bytes {
            byte_values(leaf, field).expect("values").size()
        } else {
            float_values(leaf, field).expect("values").size()
        };
        assert_eq!(10, size, "{field}");
    }
}

/// `testMergingWithDifferentKnnFields`.
pub fn test_merging_with_different_knn_fields(t: &mut LuceneTestCase) {
    merging_with_different_fields(t, false);
}

/// `testMergingWithDifferentByteKnnFields`.
pub fn test_merging_with_different_byte_knn_fields(t: &mut LuceneTestCase) {
    merging_with_different_fields(t, true);
}

/// `testWriterRamEstimate` and `testWriterByteVectorRamEstimate`.
fn writer_ram_estimate(t: &mut LuceneTestCase, bytes: bool) {
    let dir = t.new_directory();
    let mut w = plain_writer(&*dir);
    let ram = w.ram_bytes_used();
    let dim = (t.random().next_int_bounded(64) + 1) as usize;
    let dim = dim + dim % 2;
    let num_docs = t.at_least(100) as usize;
    let dot = VectorSimilarityFunction::DotProduct;
    for _ in 0..num_docs {
        let doc = if bytes {
            one_field(byte_field("fieldA", random_vector8(t.random(), dim), dot))
        } else {
            one_field(float_field("fieldA", random_vector(t.random(), dim), dot))
        };
        add(&mut w, &doc);
    }
    let ram2 = w.ram_bytes_used();
    let raw = dim * num_docs * if bytes { 1 } else { 4 };
    assert!(ram2 > ram, "{ram2} > {ram}");
    assert!(
        ram2 - ram >= raw,
        "ramBytesUsed grew by {} < raw vector data {raw}",
        ram2 - ram
    );
}

/// `testWriterRamEstimate`.
pub fn test_writer_ram_estimate(t: &mut LuceneTestCase) {
    writer_ram_estimate(t, false);
}

/// `testWriterByteVectorRamEstimate`.
pub fn test_writer_byte_vector_ram_estimate(t: &mut LuceneTestCase) {
    writer_ram_estimate(t, true);
}

/// `testAddIndexesDirectory0`.
pub fn test_add_indexes_directory0(t: &mut LuceneTestCase) {
    let doc = one_field(float_field(
        "field",
        vec![0.0; 4],
        VectorSimilarityFunction::DotProduct,
    ));
    let dir = t.new_directory();
    let dir2 = t.new_directory();
    {
        let mut w = new_writer(t, &*dir);
        add(&mut w, &doc);
        w.close().expect("close");
    }
    let mut w2 = new_writer(t, &*dir2);
    w2.add_indexes(&[&*dir]).expect("addIndexes");
    force_merge(&mut w2, 1);
    let r = open(&mut w2, &*dir2);
    let values = float_values(only_leaf(&r), "field").expect("values");
    assert_eq!(vec![(0, 0)], float_docs(&values));
    assert_eq!(0.0, float_vector(&values, 0)[0]);
}

/// `testAddIndexesDirectory1`: the vector field only in the target.
pub fn test_add_indexes_directory1(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let dir2 = t.new_directory();
    {
        let mut w = new_writer(t, &*dir);
        add(&mut w, &Document::new());
        w.close().expect("close");
    }
    let doc = one_field(float_field(
        "field",
        vec![0.0; 4],
        VectorSimilarityFunction::DotProduct,
    ));
    let mut w2 = new_writer(t, &*dir2);
    add(&mut w2, &doc);
    w2.add_indexes(&[&*dir]).expect("addIndexes");
    force_merge(&mut w2, 1);
    let r = open(&mut w2, &*dir2);
    let values = float_values(only_leaf(&r), "field").expect("values");
    let docs = float_docs(&values);
    assert_eq!(1, docs.len());
    assert_eq!(0.0, float_vector(&values, docs[0].1)[0]);
}

/// `testAddIndexesDirectory01`: a vector on both sides.
pub fn test_add_indexes_directory01(t: &mut LuceneTestCase) {
    let dot = VectorSimilarityFunction::DotProduct;
    let dir = t.new_directory();
    let dir2 = t.new_directory();
    {
        let mut w = new_writer(t, &*dir);
        add(&mut w, &one_field(float_field("field", vec![0.0; 2], dot)));
        w.close().expect("close");
    }
    let mut w2 = new_writer(t, &*dir2);
    add(&mut w2, &one_field(float_field("field", vec![1.0; 2], dot)));
    w2.add_indexes(&[&*dir]).expect("addIndexes");
    force_merge(&mut w2, 1);
    let r = open(&mut w2, &*dir2);
    let values = float_values(only_leaf(&r), "field").expect("values");
    assert_eq!(vec![(0, 0), (1, 1)], float_docs(&values));
    // The merge order is randomized: 0 or 1 first.
    let first = float_vector(&values, 0)[0];
    assert!(first == 0.0 || first == 1.0, "{first}");
    assert_eq!(1.0, first + float_vector(&values, 1)[0]);
}

/// The add-indexes schema tests: `f` as `(4, DOT_PRODUCT)` in a source,
/// and as `(dim, sim)` in the target, which refuses the source three ways
/// (`addIndexes(Directory)`, `addIndexes(CodecReader)`,
/// `TestUtil.addIndexesSlowly`) with `verifySameVectorOptions`' message.
fn change_via_add_indexes(t: &mut LuceneTestCase, dim: usize, sim: VectorSimilarityFunction) {
    let dot = VectorSimilarityFunction::DotProduct;
    let dir = t.new_directory();
    let dir2 = t.new_directory();
    {
        let mut w = new_writer(t, &*dir);
        add(&mut w, &one_field(float_field("f", vec![0.0; 4], dot)));
        w.close().expect("close");
    }
    let mut w2 = new_writer(t, &*dir2);
    add(&mut w2, &one_field(float_field("f", vec![0.0; dim], sim)));
    let name = if sim == dot {
        "DOT_PRODUCT"
    } else {
        "EUCLIDEAN"
    };
    let expected = change_message((dim, name), (4, "DOT_PRODUCT"));
    let copied = w2.add_indexes(&[&*dir]).expect_err("addIndexes(Directory)");
    assert!(copied.to_string().contains(&expected), "{copied}");
    let merged = w2
        .add_indexes_merged(&[&*dir])
        .expect_err("addIndexes(CodecReader)");
    assert!(merged.to_string().contains(&expected), "{merged}");
}

/// `testIllegalDimChangeViaAddIndexesDirectory`, `...CodecReader` and
/// `...SlowCodecReader`.
pub fn test_illegal_dim_change_via_add_indexes(t: &mut LuceneTestCase) {
    change_via_add_indexes(t, 6, VectorSimilarityFunction::DotProduct);
}

/// `testIllegalSimilarityFunctionChangeViaAddIndexesDirectory`,
/// `...CodecReader` and `...SlowCodecReader`.
pub fn test_illegal_similarity_function_change_via_add_indexes(t: &mut LuceneTestCase) {
    change_via_add_indexes(t, 4, VectorSimilarityFunction::Euclidean);
}

/// `testIllegalMultipleValues`.
pub fn test_illegal_multiple_values(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let mut w = new_writer(t, &*dir);
    let dot = VectorSimilarityFunction::DotProduct;
    let mut doc = Document::new();
    doc.add(float_field("f", vec![0.0; 4], dot));
    doc.add(float_field("f", vec![0.0; 4], dot));
    let e = add_err(&mut w, &doc);
    assert!(
        e.contains(
            "VectorValuesField \"f\" appears more than once in this document (only one value is \
             allowed per field)"
        ),
        "{e}"
    );
}

/// `testIllegalDimensionTooLarge`.
pub fn test_illegal_dimension_too_large(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let mut w = new_writer(t, &*dir);
    let dot = VectorSimilarityFunction::DotProduct;
    let too_large = || one_field(float_field("f", vec![0.0; MAX_DIMENSIONS + 1], dot));
    let message = format!("vector's dimensions must be <= [{MAX_DIMENSIONS}]");
    let e = add_err(&mut w, &too_large());
    assert!(e.contains(&message), "{e}");
    add(&mut w, &one_field(float_field("f", vec![0.0; 2], dot)));
    let e = add_err(&mut w, &too_large());
    assert!(
        e.contains("Inconsistency of field data structures across documents for field [f]")
            || e.contains(&message),
        "{e}"
    );
    w.flush().expect("flush");
    let e = add_err(&mut w, &too_large());
    assert!(e.contains(&message), "{e}");
}

/// `testIllegalEmptyVector`.
pub fn test_illegal_empty_vector(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let mut w = new_writer(t, &*dir);
    let euclidean = VectorSimilarityFunction::Euclidean;
    let e = KnnFloatVectorField::new("f", Vec::new(), euclidean).expect_err("an empty vector");
    assert!(
        e.to_string().contains("cannot index an empty vector"),
        "{e}"
    );
    add(
        &mut w,
        &one_field(float_field("f", vec![0.0; 2], euclidean)),
    );
}

/// `testInvalidKnnVectorFieldUsage`: a vector field has no numeric value.
pub fn test_invalid_knn_vector_field_usage(_t: &mut LuceneTestCase) {
    let field = float_field("field", vec![0.0; 2], VectorSimilarityFunction::Euclidean);
    assert!(field.numeric_value().is_none());
    assert!(field.string_value().is_none());
}

/// `testDeleteAllVectorDocs`.
pub fn test_delete_all_vector_docs(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let mut w = new_writer(t, &*dir);
    let mut doc = Document::new();
    doc.add(StringField::new("id", "0", Store::No));
    doc.add(float_field(
        "v",
        vec![2.0, 3.0, 5.0, 6.0],
        VectorSimilarityFunction::DotProduct,
    ));
    add(&mut w, &doc);
    add(&mut w, &Document::new());
    let r = open(&mut w, &*dir);
    assert_eq!(1, float_values(only_leaf(&r), "v").expect("values").size());
    drop(r);
    w.delete_documents_by_term(&[Term::new("id", "0")])
        .expect("deleteDocuments");
    force_merge(&mut w, 1);
    let r = open(&mut w, &*dir);
    let leaf = only_leaf(&r);
    assert_eq!(0, float_values(leaf, "v").expect("values").size());
    // A KNN search over a field whose every document is deleted.
    let target = random_normalized_vector(t.random(), 4);
    assert!(search_nearest(&*dir, leaf, "v", target, 1).is_empty());
}

/// `testKnnVectorFieldMissingFromOneSegment`.
pub fn test_knn_vector_field_missing_from_one_segment(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let mut w = new_writer(t, &*dir);
    let dot = VectorSimilarityFunction::DotProduct;
    let mut doc = Document::new();
    doc.add(StringField::new("id", "0", Store::No));
    doc.add(float_field("v0", vec![2.0, 3.0, 5.0, 6.0], dot));
    add(&mut w, &doc);
    w.commit().expect("commit");
    add(
        &mut w,
        &one_field(float_field("v1", vec![2.0, 3.0, 5.0, 6.0], dot)),
    );
    force_merge(&mut w, 1);
    let r = open(&mut w, &*dir);
    let leaf = only_leaf(&r);
    assert_eq!(vec![(0, 0)], float_docs(&float_values(leaf, "v0").unwrap()));
    assert_eq!(vec![(1, 0)], float_docs(&float_values(leaf, "v1").unwrap()));
}

/// `testSparseVectors`.
pub fn test_sparse_vectors(t: &mut LuceneTestCase) {
    let num_docs = t.at_least(1000);
    let num_fields = test_util::next_int(t.random(), 1, 10) as usize;
    let mut field_doc_counts = vec![0i32; num_fields];
    let mut field_totals = vec![0f64; num_fields];
    let mut field_dims = vec![0usize; num_fields];
    let mut field_sims = vec![VectorSimilarityFunction::Euclidean; num_fields];
    let mut field_encodings = vec![VectorEncoding::Float32; num_fields];
    for i in 0..num_fields {
        let dim = (t.random().next_int_bounded(20) + 1) as usize;
        field_dims[i] = dim + dim % 2;
        field_sims[i] = random_similarity(t.random());
        field_encodings[i] = random_vector_encoding(t.random());
    }
    let dir = t.new_directory();
    let mut w = new_riw(t, &*dir);
    for _ in 0..num_docs {
        let mut doc = Document::new();
        for field in 0..num_fields {
            let name = format!("int{field}");
            if t.random().next_int_bounded(100) == 17 {
                match field_encodings[field] {
                    VectorEncoding::Byte => {
                        let b = random_vector8(t.random(), field_dims[field]);
                        field_totals[field] += f64::from(b[0] as i8);
                        doc.add(byte_field(&name, b, field_sims[field]));
                    }
                    VectorEncoding::Float32 => {
                        let v = random_normalized_vector(t.random(), field_dims[field]);
                        field_totals[field] += f64::from(v[0]);
                        doc.add(float_field(&name, v, field_sims[field]));
                    }
                }
                field_doc_counts[field] += 1;
            }
        }
        w.add_document(&doc).expect("addDocument");
    }
    let r = w.get_reader().expect("getReader");
    for field in 0..num_fields {
        let name = format!("int{field}");
        let (mut doc_count, mut checksum) = (0, 0f64);
        for leaf in r.segment_readers() {
            match field_encodings[field] {
                VectorEncoding::Byte => {
                    if let Some(values) = byte_values(leaf, &name) {
                        doc_count += values.size();
                        for (_, ord) in byte_docs(&values) {
                            checksum += f64::from(byte_vector(&values, ord)[0] as i8);
                        }
                    }
                }
                VectorEncoding::Float32 => {
                    if let Some(values) = float_values(leaf, &name) {
                        doc_count += values.size();
                        for (_, ord) in float_docs(&values) {
                            checksum += f64::from(float_vector(&values, ord)[0]);
                        }
                    }
                }
            }
        }
        assert_eq!(field_doc_counts[field], doc_count, "{name}");
        let delta = if field_encodings[field] == VectorEncoding::Byte {
            f64::from(num_docs) * 0.01
        } else {
            1e-5
        };
        assert!(
            (field_totals[field] - checksum).abs() <= delta,
            "{name}: {} vs {checksum}",
            field_totals[field]
        );
    }
    w.close().expect("close");
}

/// The documents `testFloatVectorScorerIteration` and its byte twin index:
/// `numDoc` updates of random ids with a random vector six times in seven,
/// with deletes and commits.
fn index_for_scoring(
    t: &mut LuceneTestCase,
    w: &mut IndexWriter<'_>,
    dimension: usize,
    bytes: bool,
    sim: VectorSimilarityFunction,
) {
    let num_doc = t.at_least(100);
    for i in 0..num_doc {
        let vector = if t.random().next_int_bounded(7) != 3 {
            Some(if bytes {
                Vector::Byte(random_vector8(t.random(), dimension))
            } else {
                Vector::Float(random_normalized_vector(t.random(), dimension))
            })
        } else {
            None
        };
        add_random_key(t, w, "field", i, vector, sim);
        if t.random().next_int_bounded(10) == 2 {
            let id = t.random().next_int_bounded(i + 1).to_string();
            w.delete_documents_by_term(&[Term::new("id", id)])
                .expect("deleteDocuments");
        }
        if t.random().next_int_bounded(10) == 3 {
            w.commit().expect("commit");
        }
    }
}

/// `testFloatVectorScorerIteration` and `testByteVectorScorerIteration`:
/// a scorer over every ordinal scores each one, and its ordinals' documents
/// are the iterator's.
fn scorer_iteration(t: &mut LuceneTestCase, bytes: bool) {
    let suite = KnnVectorsSuite::new(t);
    let dir = t.new_directory();
    let mut w = new_writer(t, &*dir);
    if t.random().next_boolean() {
        sort_by_key(&mut w);
    }
    let dimension = even_dimension(t);
    index_for_scoring(t, &mut w, dimension, bytes, suite.similarity);
    // `vectorToScore`, drawn once before the reader opens.
    let target = if bytes {
        Vector::Byte(random_vector8(t.random(), dimension))
    } else {
        Vector::Float(random_normalized_vector(t.random(), dimension))
    };
    let r = open(&mut w, &*dir);
    let mut scored = 0;
    for leaf in r.segment_readers() {
        use lucene_codecs::hnsw::VectorScorer as _;
        let (docs, scores) = if let Vector::Byte(target) = &target {
            let Some(values) = byte_values(leaf, "field") else {
                continue;
            };
            let mut scorer = values.scorer(target).expect("scorer");
            let scores: Vec<f32> = (0..values.size())
                .map(|ord| scorer.score(ord).expect("score"))
                .collect();
            (byte_docs(&values), scores)
        } else if let Vector::Float(target) = &target {
            let Some(values) = float_values(leaf, "field") else {
                continue;
            };
            let mut scorer = values.scorer(target).expect("scorer");
            let scores: Vec<f32> = (0..values.size())
                .map(|ord| scorer.score(ord).expect("score"))
                .collect();
            (float_docs(&values), scores)
        } else {
            unreachable!("one target per encoding")
        };
        // The iterator's documents ascend, one per ordinal.
        assert!(docs.windows(2).all(|p| p[0] < p[1]), "{docs:?}");
        for score in scores {
            assert!(score >= 0.0, "{score}");
            scored += 1;
        }
    }
    assert!(scored > 0, "nothing was scored");
}

/// `testFloatVectorScorerIteration`.
pub fn test_float_vector_scorer_iteration(t: &mut LuceneTestCase) {
    scorer_iteration(t, false);
}

/// `testByteVectorScorerIteration`.
pub fn test_byte_vector_scorer_iteration(t: &mut LuceneTestCase) {
    scorer_iteration(t, true);
}

/// `testEmptyFloatVectorData` and `testEmptyByteVectorData`: a merged
/// segment whose one vector document is deleted keeps an empty field.
fn empty_vector_data(t: &mut LuceneTestCase, bytes: bool) {
    let dir = t.new_directory();
    let mut w = new_writer(t, &*dir);
    let dot = VectorSimilarityFunction::DotProduct;
    let mut doc1 = Document::new();
    doc1.add(StringField::new("id", "0", Store::No));
    if bytes {
        doc1.add(byte_field("v", vec![2, 3, 5, 6], dot));
    } else {
        doc1.add(float_field("v", vec![2.0, 3.0, 5.0, 6.0], dot));
    }
    add(&mut w, &doc1);
    let mut doc2 = Document::new();
    doc2.add(StringField::new("id", "1", Store::No));
    add(&mut w, &doc2);
    w.delete_documents_by_term(&[Term::new("id", "0")])
        .expect("deleteDocuments");
    w.commit().expect("commit");
    force_merge(&mut w, 1);
    let r = open(&mut w, &*dir);
    let leaf = only_leaf(&r);
    if bytes {
        let values = byte_values(leaf, "v").expect("values");
        assert_eq!(0, values.size());
        assert!(
            search_nearest_byte(&*dir, leaf, "v", &[2, 3, 5, 6], 1, i32::MAX as u64)
                .score_docs
                .is_empty()
        );
    } else {
        let values = float_values(leaf, "v").expect("values");
        assert_eq!(0, values.size());
        assert!(search_nearest(&*dir, leaf, "v", vec![2.0, 3.0, 5.0, 6.0], 1).is_empty());
    }
}

/// `testEmptyFloatVectorData`.
pub fn test_empty_float_vector_data(t: &mut LuceneTestCase) {
    empty_vector_data(t, false);
}

/// `testEmptyByteVectorData`.
pub fn test_empty_byte_vector_data(t: &mut LuceneTestCase) {
    empty_vector_data(t, true);
}

/// `testIndexedValueNotAliased`: three documents in one segment, read back
/// with the values they were added with.
pub fn test_indexed_value_not_aliased(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let mut w = plain_writer(&*dir);
    let euclidean = VectorSimilarityFunction::Euclidean;
    let mut v = vec![0.0f32, 0.0];
    let doc1 = one_field(float_field("field", v.clone(), euclidean));
    v[0] = 1.0;
    let doc2 = one_field(float_field("field", v.clone(), euclidean));
    add(&mut w, &doc1);
    add(&mut w, &doc2);
    v[0] = 2.0;
    add(
        &mut w,
        &one_field(float_field("field", v.clone(), euclidean)),
    );
    force_merge(&mut w, 1);
    let r = open(&mut w, &*dir);
    let values = float_values(only_leaf(&r), "field").expect("values");
    assert_eq!(3, values.size());
    assert_eq!(vec![(0, 0), (1, 1), (2, 2)], float_docs(&values));
    for (ord, expected) in [0.0, 1.0, 2.0].into_iter().enumerate() {
        assert_eq!(expected, float_vector(&values, ord as i32)[0]);
    }
}

/// `testSortedIndex` and `testSortedIndexBytes`.
fn sorted_index(t: &mut LuceneTestCase, bytes: bool) {
    let dir = t.new_directory();
    let mut w = new_writer(t, &*dir);
    sort_by_key(&mut w);
    let euclidean = VectorSimilarityFunction::Euclidean;
    let vector = |x: i8, y: i8| {
        Some(if bytes {
            Vector::Byte(vec![x as u8, y as u8])
        } else {
            Vector::Float(vec![f32::from(x), f32::from(y)])
        })
    };
    add_vector(&mut w, "field", 1, 1, vector(-1, 0), euclidean);
    add_vector(&mut w, "field", 4, 4, vector(0, 1), euclidean);
    add_vector(&mut w, "field", 3, 3, None, euclidean);
    add_vector(&mut w, "field", 2, 2, vector(1, 0), euclidean);
    force_merge(&mut w, 1);
    let r = open(&mut w, &*dir);
    let leaf = only_leaf(&r);
    let (docs, firsts): (Vec<(i32, i32)>, Vec<f32>) = if bytes {
        let values = byte_values(leaf, "field").expect("values");
        assert_eq!(2, values.dimension());
        let firsts = (0..3)
            .map(|o| f32::from(byte_vector(&values, o)[0] as i8))
            .collect();
        (byte_docs(&values), firsts)
    } else {
        let values = float_values(leaf, "field").expect("values");
        assert_eq!(2, values.dimension());
        let firsts = (0..3).map(|o| float_vector(&values, o)[0]).collect();
        (float_docs(&values), firsts)
    };
    let ids: Vec<String> = docs.iter().map(|&(doc, _)| stored_id(leaf, doc)).collect();
    assert_eq!(["1", "2", "4"], ids.as_slice());
    assert_eq!(vec![-1.0, 1.0, 0.0], firsts);
}

/// `testSortedIndex`.
pub fn test_sorted_index(t: &mut LuceneTestCase) {
    sorted_index(t, false);
}

/// `testSortedIndexBytes`.
pub fn test_sorted_index_bytes(t: &mut LuceneTestCase) {
    sorted_index(t, true);
}

/// `testIndexMultipleKnnVectorFields`.
pub fn test_index_multiple_knn_vector_fields(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let analyzer = mock_analyzer(t);
    let mut config = new_config(t, analyzer);
    let (nightly, multiplier) = (t.nightly(), t.multiplier());
    let log = new_log_merge_policy(&mut Draws::new(t.random()).knobs(nightly, multiplier));
    config.merge_policy = LiveMergePolicy::new(PolicyKind::Log(log));
    let mut w = writer_with(&*dir, &config);
    let euclidean = VectorSimilarityFunction::Euclidean;
    let two_fields = |first: f32| {
        let mut doc = Document::new();
        doc.add(float_field("field1", vec![first, 2.0], euclidean));
        doc.add(float_field("field2", vec![1.0, 2.0, 3.0, 4.0], euclidean));
        doc
    };
    add(&mut w, &two_fields(1.0));
    add(&mut w, &two_fields(2.0));
    add(
        &mut w,
        &one_field(float_field(
            "field3",
            vec![1.0, 2.0, 3.0, 4.0],
            VectorSimilarityFunction::DotProduct,
        )),
    );
    force_merge(&mut w, 1);
    let r = open(&mut w, &*dir);
    let leaf = only_leaf(&r);
    let v1 = float_values(leaf, "field1").expect("field1");
    assert_eq!((2, 2), (v1.dimension(), v1.size()));
    assert_eq!(2, float_docs(&v1).len());
    assert_eq!(1.0, float_vector(&v1, 0)[0]);
    assert_eq!(2.0, float_vector(&v1, 1)[0]);
    let v2 = float_values(leaf, "field2").expect("field2");
    assert_eq!((4, 2), (v2.dimension(), v2.size()));
    assert_eq!(2, float_docs(&v2).len());
    assert_eq!(2.0, float_vector(&v2, 0)[1]);
    assert_eq!(2.0, float_vector(&v2, 1)[1]);
    let v3 = float_values(leaf, "field3").expect("field3");
    assert_eq!((4, 1), (v3.dimension(), v3.size()));
    assert_eq!(1, float_docs(&v3).len());
    assert!((float_vector(&v3, 0)[0] - 1.0).abs() <= 0.1);
}

/// `testRandom` and `testRandomBytes`: random vectors, sometimes skipped,
/// sometimes deleted, under commits and maybe an index sort, read back by
/// stored id.
fn random_vectors(t: &mut LuceneTestCase, bytes: bool) {
    let suite = KnnVectorsSuite::new(t);
    let dir = t.new_directory();
    let mut w = new_writer(t, &*dir);
    if t.random().next_boolean() {
        sort_by_key(&mut w);
    }
    let num_doc = t.at_least(100) as usize;
    let dimension = even_dimension(t);
    let mut num_values = 0;
    let mut values: Vec<Option<Vector>> = Vec::with_capacity(num_doc);
    for i in 0..num_doc {
        let value = if t.random().next_int_bounded(7) != 3 {
            num_values += 1;
            Some(if bytes {
                Vector::Byte(random_vector8(t.random(), dimension))
            } else {
                Vector::Float(random_normalized_vector(t.random(), dimension))
            })
        } else {
            None
        };
        // "sometimes use a shared scratch array": a draw, nothing to share.
        let _ = t.random().next_boolean() && value.is_some();
        let copy = match &value {
            Some(Vector::Float(v)) => Some(Vector::Float(v.clone())),
            Some(Vector::Byte(v)) => Some(Vector::Byte(v.clone())),
            None => None,
        };
        values.push(value);
        add_random_key(t, &mut w, "field", i as i32, copy, suite.similarity);
        if t.random().next_int_bounded(10) == 2 {
            let id_to_delete = t.random().next_int_bounded(i as i32 + 1) as usize;
            w.delete_documents_by_term(&[Term::new("id", id_to_delete.to_string())])
                .expect("deleteDocuments");
            if values[id_to_delete].take().is_some() {
                num_values -= 1;
            }
        }
        if t.random().next_int_bounded(10) == 3 {
            w.commit().expect("commit");
        }
    }
    let r = open(&mut w, &*dir);
    let (mut value_count, mut total_size, mut num_deletes) = (0, 0, 0);
    for leaf in r.segment_readers() {
        let read: Vec<(i32, Vector)> = if bytes {
            let Some(v) = byte_values(leaf, "field") else {
                continue;
            };
            total_size += v.size();
            byte_docs(&v)
                .into_iter()
                .map(|(doc, ord)| (doc, Vector::Byte(byte_vector(&v, ord))))
                .collect()
        } else {
            let Some(v) = float_values(leaf, "field") else {
                continue;
            };
            total_size += v.size();
            float_docs(&v)
                .into_iter()
                .map(|(doc, ord)| (doc, Vector::Float(float_vector(&v, ord))))
                .collect()
        };
        for (doc, vector) in read {
            let id_string = stored_id(leaf, doc);
            let id: usize = id_string.parse().expect("an int id");
            if is_live(leaf, doc) {
                match (&values[id], &vector) {
                    (Some(Vector::Float(a)), Vector::Float(b)) => {
                        assert_eq!(dimension, b.len());
                        assert_eq!(a, b, "{id_string} {doc}");
                    }
                    (Some(Vector::Byte(a)), Vector::Byte(b)) => {
                        assert_eq!(dimension, b.len());
                        assert_eq!(a, b, "{id_string} {doc}");
                    }
                    _ => panic!("live doc {doc} (id {id_string}) has an unexpected vector"),
                }
                value_count += 1;
            } else {
                num_deletes += 1;
                assert!(
                    values[id].is_none(),
                    "deleted id {id_string} kept its value"
                );
            }
        }
    }
    assert_eq!(num_values, value_count);
    assert_eq!(num_values, total_size - num_deletes);
}

/// `testRandom`.
pub fn test_random(t: &mut LuceneTestCase) {
    random_vectors(t, false);
}

/// `testRandomBytes`.
pub fn test_random_bytes(t: &mut LuceneTestCase) {
    random_vectors(t, true);
}

/// Java: `BaseKnnVectorsFormatTestCase.testSearchWithVisitedLimit`.
/// Sparse vectors and deletions exercise ordinal-to-document translation;
/// small budgets report a lower bound and a budget above the vector count
/// finishes exactly. Off-heap byte accounting has no counterpart here.
pub fn test_search_with_visited_limit(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let mut w = new_writer(t, &*dir);
    let dimension = 10;
    for id in 0..300 {
        let vector = if t.random().next_int_bounded(7) != 3 {
            Some(Vector::Float(random_normalized_vector(
                t.random(),
                dimension,
            )))
        } else {
            None
        };
        add_random_key(
            t,
            &mut w,
            "field",
            id,
            vector,
            VectorSimilarityFunction::Euclidean,
        );
    }
    force_merge(&mut w, 1);
    for _ in 0..30 {
        let id = t.random().next_int_bounded(300);
        w.delete_documents_by_term(&[Term::new("id", id.to_string())])
            .expect("deleteDocuments");
    }
    let r = open(&mut w, &*dir);
    let mut searched = 0;
    for leaf in r.segment_readers() {
        let Some(values) = float_values(leaf, "field") else {
            continue;
        };
        let k = 5 + t.random().next_int_bounded(45) as usize;
        let visited_limit = k as u64 + t.random().next_int_bounded(5) as u64;
        let target = random_normalized_vector(t.random(), dimension);
        let limited = search_nearest_float(&*dir, leaf, "field", &target, k, visited_limit);
        assert_eq!(
            limited.total_hits.relation,
            TotalHitsRelation::GreaterThanOrEqualTo
        );
        // Java: Lucene99HnswVectorsReader.EXHAUSTIVE_BULK_SCORE_ORDS.
        let size = 64;
        assert!(limited.total_hits.value <= visited_limit.div_ceil(size) * size);
        assert!(limited
            .score_docs
            .iter()
            .all(|hit| is_live(leaf, hit.doc_id)));

        let k = values.size() as usize;
        let visited_limit = k as u64 + 30;
        let target = random_normalized_vector(t.random(), dimension);
        let complete = search_nearest_float(&*dir, leaf, "field", &target, k, visited_limit);
        assert_eq!(complete.total_hits.relation, TotalHitsRelation::EqualTo);
        assert!(complete.total_hits.value <= visited_limit);
        let live_vectors = float_docs(&values)
            .iter()
            .filter(|(doc, _)| is_live(leaf, *doc))
            .count();
        assert_eq!(complete.score_docs.len(), live_vectors);
        assert!(complete
            .score_docs
            .iter()
            .all(|hit| is_live(leaf, hit.doc_id)));
        searched += 1;
    }
    assert!(searched > 0, "no leaf was searched");
}

/// `testRandomWithUpdatesAndGraph`: random updates of random ids, read back
/// by id, then a nearest-neighbour search per leaf returning `k` results in
/// descending score order.
pub fn test_random_with_updates_and_graph(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let mut w = new_writer(t, &*dir);
    let num_doc = t.at_least(100);
    let dimension = even_dimension(t);
    let mut id2value: Vec<Option<Vec<f32>>> = vec![None; num_doc as usize];
    for _ in 0..num_doc {
        let id = t.random().next_int_bounded(num_doc);
        let value = if t.random().next_int_bounded(7) != 3 {
            Some(random_normalized_vector(t.random(), dimension))
        } else {
            None
        };
        id2value[id as usize] = value.clone();
        add_random_key(
            t,
            &mut w,
            "field",
            id,
            value.map(Vector::Float),
            VectorSimilarityFunction::Euclidean,
        );
    }
    let r = open(&mut w, &*dir);
    let mut searched = 0;
    for (ord, leaf) in r.segment_readers().iter().enumerate() {
        let Some(values) = float_values(leaf, "field") else {
            continue;
        };
        let mut num_live_docs_with_vectors = 0;
        for (doc, vord) in float_docs(&values) {
            let v = float_vector(&values, vord);
            assert_eq!(dimension, v.len());
            let id_string = stored_id(leaf, doc);
            let id: usize = id_string.parse().expect("an int id");
            if is_live(leaf, doc) {
                assert_eq!(
                    id2value[id].as_ref(),
                    Some(&v),
                    "values differ for id={id_string}, docid={doc} leaf={ord}"
                );
                num_live_docs_with_vectors += 1;
            } else if let Some(expected) = &id2value[id] {
                assert_ne!(expected, &v);
            }
        }
        if num_live_docs_with_vectors == 0 {
            continue;
        }
        let size = values.size();
        let k = (t.random().next_int_bounded(size / 10 + 1) + 1).min(num_live_docs_with_vectors);
        let target = random_normalized_vector(t.random(), dimension);
        let results = search_nearest(&*dir, leaf, "field", target, k as usize);
        assert_eq!(k.min(size) as usize, results.len());
        assert!(results.windows(2).all(|p| p[0].score >= p[1].score));
        searched += 1;
    }
    assert!(searched > 0, "no leaf was searched");
}

/// `testCheckIndexIncludesVectors`.
pub fn test_check_index_includes_vectors(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    {
        let mut w = new_writer(t, &*dir);
        let euclidean = VectorSimilarityFunction::Euclidean;
        let mut doc = Document::new();
        doc.add(float_field(
            "v1",
            random_normalized_vector(t.random(), 4),
            euclidean,
        ));
        add(&mut w, &doc);
        doc.add(float_field(
            "v2",
            random_normalized_vector(t.random(), 4),
            euclidean,
        ));
        add(&mut w, &doc);
        w.close().expect("close");
    }
    let results = lucene_index::check_index::check_directory(&*dir).expect("CheckIndex");
    // The commit's own result, then one per segment.
    assert_eq!(2, results.len());
    let segment = &results[1];
    assert!(segment.all_passed(), "{:?}", segment.failures());
    // Three vector values were indexed, across two fields.
    assert_eq!(3, segment.stats.vector_values);
    assert_eq!(2, segment.stats.vector_fields);
}

/// `testSimilarityFunctionIdentifiers`.
pub fn test_similarity_function_identifiers(_t: &mut LuceneTestCase) {
    assert_eq!(0, VectorSimilarityFunction::Euclidean as i32);
    assert_eq!(1, VectorSimilarityFunction::DotProduct as i32);
    assert_eq!(2, VectorSimilarityFunction::Cosine as i32);
    assert_eq!(3, VectorSimilarityFunction::MaximumInnerProduct as i32);
    assert_eq!(4, SIMILARITIES.len());
}

/// `testVectorEncodingOrdinals`.
pub fn test_vector_encoding_ordinals(_t: &mut LuceneTestCase) {
    assert_eq!(0, VectorEncoding::Byte as i32);
    assert_eq!(1, VectorEncoding::Float32 as i32);
    assert_eq!(2, ENCODINGS.len());
}

/// `testAdvance`: `advance(target)` through the doc-to-ordinal cursor lands
/// where the ordinal iterator says.
pub fn test_advance(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let mut w = new_writer(t, &*dir);
    let numdocs = t.at_least(1500);
    for _ in 0..numdocs {
        let mut doc = Document::new();
        if t.random().next_int_bounded(4) == 3 {
            doc.add(float_field(
                "field",
                vec![0.0; 4],
                VectorSimilarityFunction::Euclidean,
            ));
        }
        add(&mut w, &doc);
    }
    force_merge(&mut w, 1);
    let r = open(&mut w, &*dir);
    let leaf = only_leaf(&r);
    let values = float_values(leaf, "field").expect("values");
    let mut vector_docs: Vec<i32> = float_docs(&values).into_iter().map(|(d, _)| d).collect();
    vector_docs.push(NO_MORE_DOCS);
    let mut cursor = values.doc_to_ord().expect("docToOrd");
    let max_doc = leaf.max_doc;
    let mut doc_id = -1;
    let mut advance = |target: i32| -> i32 {
        doc_id = (target..max_doc)
            .find(|&d| cursor.ordinal(d).expect("ordinal").is_some())
            .unwrap_or(NO_MORE_DOCS);
        doc_id
    };
    let mut cur: usize = 0;
    let mut i = 0;
    let mut advanced = 0;
    let mut first = true;
    while i < numdocs {
        if t.random().next_int_bounded(4) == 3 {
            if first {
                first = false;
            } else {
                cur += 1;
            }
            while vector_docs[cur] < i {
                cur += 1;
            }
            let landed = advance(i);
            assert_eq!(vector_docs[cur], landed);
            advanced += 1;
            if landed == NO_MORE_DOCS {
                break;
            }
            // Past the document advanced to, for the next iteration.
            i = landed;
        }
        i += 1;
    }
    assert!(advanced > 0, "advance never ran");
}

/// `testVectorValuesReportCorrectDocs`: the iterator's documents and
/// `ordToDoc` agree with what was indexed, by stored id sum and checksum.
pub fn test_vector_values_report_correct_docs(t: &mut LuceneTestCase) {
    let suite = KnnVectorsSuite::new(t);
    let num_docs = t.at_least(1000);
    let dim = (t.random().next_int_bounded(20) + 1) as usize;
    let dim = dim + dim % 2;
    let (mut values_checksum, mut field_doc_count, mut field_sum_doc_ids) = (0f64, 0, 0i64);
    let dir = t.new_directory();
    let mut w = new_riw(t, &*dir);
    for _ in 0..num_docs {
        let mut doc = Document::new();
        let doc_id = t.random().next_int_bounded(num_docs);
        doc.add(Field::stored("id", StoredValue::Int(doc_id)));
        if t.random().next_int_bounded(4) == 3 {
            match suite.encoding {
                VectorEncoding::Byte => {
                    let b = random_vector8(t.random(), dim);
                    values_checksum += f64::from(b[0] as i8);
                    doc.add(byte_field("knn_vector", b, suite.similarity));
                }
                VectorEncoding::Float32 => {
                    let v = random_normalized_vector(t.random(), dim);
                    values_checksum += f64::from(v[0]);
                    doc.add(float_field("knn_vector", v, suite.similarity));
                }
            }
            field_doc_count += 1;
            field_sum_doc_ids += i64::from(doc_id);
        }
        w.add_document(&doc).expect("addDocument");
    }
    if t.random().next_boolean() {
        riw_force_merge(&mut w, 1);
    }
    let r = w.get_reader().expect("getReader");
    let (mut checksum, mut doc_count, mut sum_doc_ids, mut sum_ord_to_doc_ids) =
        (0f64, 0, 0i64, 0i64);
    let id_of = |leaf: &SegmentReader, doc: i32| -> i64 {
        stored_id(leaf, doc).parse().expect("an int id")
    };
    for leaf in r.segment_readers() {
        let (docs, firsts) = match suite.encoding {
            VectorEncoding::Byte => {
                let Some(v) = byte_values(leaf, "knn_vector") else {
                    continue;
                };
                let docs = byte_docs(&v);
                let firsts: Vec<f64> = docs
                    .iter()
                    .map(|&(_, o)| f64::from(byte_vector(&v, o)[0] as i8))
                    .collect();
                (docs, firsts)
            }
            VectorEncoding::Float32 => {
                let Some(v) = float_values(leaf, "knn_vector") else {
                    continue;
                };
                let docs = float_docs(&v);
                let firsts: Vec<f64> = docs
                    .iter()
                    .map(|&(_, o)| f64::from(float_vector(&v, o)[0]))
                    .collect();
                (docs, firsts)
            }
        };
        doc_count += docs.len() as i32;
        checksum += firsts.iter().sum::<f64>();
        for &(doc, _) in &docs {
            sum_doc_ids += id_of(leaf, doc);
        }
        // `ordToDoc(ord)` for every ordinal, independently of the iterator.
        let mut by_ord = docs.clone();
        by_ord.sort_by_key(|&(_, ord)| ord);
        for (ord, &(doc, o)) in by_ord.iter().enumerate() {
            assert_eq!(ord as i32, o);
            sum_ord_to_doc_ids += id_of(leaf, doc);
        }
    }
    let delta = if suite.encoding == VectorEncoding::Byte {
        f64::from(num_docs) * 0.2
    } else {
        1e-5
    };
    assert!(
        (values_checksum - checksum).abs() <= delta,
        "encoding={:?}: {values_checksum} vs {checksum}",
        suite.encoding
    );
    assert_eq!(field_doc_count, doc_count);
    assert_eq!(field_sum_doc_ids, sum_doc_ids);
    assert_eq!(field_sum_doc_ids, sum_ord_to_doc_ids);
    w.close().expect("close");
}

/// `testMismatchedFields`: a second index numbering `float` and `byte` the
/// other way round, merged in and force-merged.
pub fn test_mismatched_fields(t: &mut LuceneTestCase) {
    let euclidean = VectorSimilarityFunction::Euclidean;
    let float = || float_field("float", vec![1.0, 2.0], euclidean);
    let byte = || byte_field("byte", vec![42], euclidean);
    let dir1 = t.new_directory();
    {
        let mut w1 = new_writer(t, &*dir1);
        let mut doc = Document::new();
        doc.add(byte());
        doc.add(float());
        add(&mut w1, &doc);
        w1.close().expect("close");
    }
    let dir2 = t.new_directory();
    let mut w2 = new_writer(t, &*dir2);
    let mut doc = Document::new();
    doc.add(float());
    doc.add(byte());
    add(&mut w2, &doc);
    w2.commit().expect("commit");
    w2.add_indexes_merged(&[&*dir1]).expect("addIndexes");
    force_merge(&mut w2, 1);
    let r = open(&mut w2, &*dir2);
    let leaf = only_leaf(&r);
    let bytes = byte_values(leaf, "byte").expect("byte vectors");
    assert_eq!(vec![(0, 0), (1, 1)], byte_docs(&bytes));
    assert_eq!(vec![42u8], byte_vector(&bytes, 0));
    assert_eq!(vec![42u8], byte_vector(&bytes, 1));
    let floats = float_values(leaf, "float").expect("float vectors");
    assert_eq!(vec![(0, 0), (1, 1)], float_docs(&floats));
    assert_eq!(vec![1.0f32, 2.0], float_vector(&floats, 0));
    assert_eq!(vec![1.0f32, 2.0], float_vector(&floats, 1));
}

/// This suite's tests, by Java name.
pub const TESTS: &[SuiteTest] = &[
    ("testFieldConstructor", test_field_constructor),
    (
        "testFieldConstructorExceptions",
        test_field_constructor_exceptions,
    ),
    (
        "testIllegalDimChangeTwoDocs",
        test_illegal_dim_change_two_docs,
    ),
    (
        "testIllegalSimilarityFunctionChange",
        test_illegal_similarity_function_change,
    ),
    (
        "testIllegalDimChangeTwoWriters",
        test_illegal_dim_change_two_writers,
    ),
    (
        "testMergingWithDifferentKnnFields",
        test_merging_with_different_knn_fields,
    ),
    (
        "testMergingWithDifferentByteKnnFields",
        test_merging_with_different_byte_knn_fields,
    ),
    ("testWriterRamEstimate", test_writer_ram_estimate),
    (
        "testWriterByteVectorRamEstimate",
        test_writer_byte_vector_ram_estimate,
    ),
    (
        "testIllegalSimilarityFunctionChangeTwoWriters",
        test_illegal_similarity_function_change_two_writers,
    ),
    ("testAddIndexesDirectory0", test_add_indexes_directory0),
    ("testAddIndexesDirectory1", test_add_indexes_directory1),
    ("testAddIndexesDirectory01", test_add_indexes_directory01),
    (
        "testIllegalDimChangeViaAddIndexesDirectory",
        test_illegal_dim_change_via_add_indexes,
    ),
    (
        "testIllegalSimilarityFunctionChangeViaAddIndexesDirectory",
        test_illegal_similarity_function_change_via_add_indexes,
    ),
    ("testIllegalMultipleValues", test_illegal_multiple_values),
    (
        "testIllegalDimensionTooLarge",
        test_illegal_dimension_too_large,
    ),
    ("testIllegalEmptyVector", test_illegal_empty_vector),
    (
        "testInvalidKnnVectorFieldUsage",
        test_invalid_knn_vector_field_usage,
    ),
    ("testDeleteAllVectorDocs", test_delete_all_vector_docs),
    (
        "testKnnVectorFieldMissingFromOneSegment",
        test_knn_vector_field_missing_from_one_segment,
    ),
    ("testSparseVectors", test_sparse_vectors),
    (
        "testFloatVectorScorerIteration",
        test_float_vector_scorer_iteration,
    ),
    (
        "testByteVectorScorerIteration",
        test_byte_vector_scorer_iteration,
    ),
    ("testEmptyFloatVectorData", test_empty_float_vector_data),
    ("testEmptyByteVectorData", test_empty_byte_vector_data),
    ("testIndexedValueNotAliased", test_indexed_value_not_aliased),
    ("testSortedIndex", test_sorted_index),
    ("testSortedIndexBytes", test_sorted_index_bytes),
    (
        "testIndexMultipleKnnVectorFields",
        test_index_multiple_knn_vector_fields,
    ),
    ("testRandom", test_random),
    ("testRandomBytes", test_random_bytes),
    ("testSearchWithVisitedLimit", test_search_with_visited_limit),
    (
        "testRandomWithUpdatesAndGraph",
        test_random_with_updates_and_graph,
    ),
    (
        "testCheckIndexIncludesVectors",
        test_check_index_includes_vectors,
    ),
    (
        "testSimilarityFunctionIdentifiers",
        test_similarity_function_identifiers,
    ),
    ("testVectorEncodingOrdinals", test_vector_encoding_ordinals),
    ("testAdvance", test_advance),
    (
        "testVectorValuesReportCorrectDocs",
        test_vector_values_report_correct_docs,
    ),
    ("testMismatchedFields", test_mismatched_fields),
];
