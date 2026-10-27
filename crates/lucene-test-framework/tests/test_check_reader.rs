//! `TestUtil.checkReader` over a real multi-segment index with every kind
//! of field and some deletions, directly and through the test framework's
//! reader views; and the failures it reports when a view lies.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use lucene_index::buffered_updates::Term;
use lucene_index::document::IndexOptions;
use lucene_index::document::{
    BinaryDocValuesField, Document, Field, FieldType, IntPoint, KnnByteVectorField,
    KnnFloatVectorField, NumericDocValuesField, SortedDocValuesField, SortedNumericDocValuesField,
    SortedSetDocValuesField, Store, StringField, TextField, VectorSimilarityFunction,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::filter::{FilterLeafReader, LeafFilter};
use lucene_search::reader::{
    ByteVectorValues, DocIdSetIterator, DocValuesIterator, FloatVectorValues, IntersectVisitor,
    LeafReader, NumericDocValues, PointValues, Terms, TermsEnum,
};
use lucene_search::Result;
use lucene_store::ByteBuffersDirectory;
use lucene_test_framework::index::asserting_leaf_reader::AssertingLeafReader;
use lucene_test_framework::index::reader_wrappers::{
    AllDeletedFilterReader, FieldFilterLeafReader,
};
use lucene_test_framework::util::check_reader::{check_leaf_reader, check_reader};
use lucene_util::fixed_bit_set::FixedBitSet;

fn index(dir: &ByteBuffersDirectory) {
    let mut w = IndexWriter::open(
        dir,
        Vec::new(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    let mut body = FieldType::copy_of(&TextField::type_stored());
    body.set_store_term_vectors(true).unwrap();
    body.set_store_term_vector_positions(true).unwrap();
    body.set_store_term_vector_offsets(true).unwrap();
    body.set_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
        .unwrap();
    for seg in 0..3 {
        for i in 0..40 {
            let k = seg * 40 + i;
            let mut d = Document::new();
            d.add(StringField::new("id", k.to_string(), Store::Yes));
            let text = format!("word{} common w{} common tail{}", k % 7, k % 3, k);
            d.add(Field::from_string("body", text, body.clone()).unwrap());
            d.add(NumericDocValuesField::new("n", i64::from(k)));
            d.add(BinaryDocValuesField::new(
                "b",
                vec![u8::try_from(k).unwrap()],
            ));
            d.add(SortedDocValuesField::new("s", format!("s{}", k % 5)));
            d.add(SortedSetDocValuesField::new("ss", format!("ss{}", k % 4)));
            d.add(SortedSetDocValuesField::new("ss", format!("tt{}", k % 2)));
            d.add(SortedNumericDocValuesField::new("sn", i64::from(k % 9)));
            d.add(SortedNumericDocValuesField::new("sn", -i64::from(k)));
            d.add(IntPoint::new("p", &[k, -k]).unwrap());
            if k % 3 != 0 {
                d.add(IntPoint::new("p1", &[k % 11]).unwrap());
                let f = f32::from(u8::try_from(k % 13).unwrap());
                d.add(
                    KnnFloatVectorField::new(
                        "fv",
                        vec![f, 1.0, -f],
                        VectorSimilarityFunction::Euclidean,
                    )
                    .unwrap(),
                );
                d.add(
                    KnnByteVectorField::new(
                        "bv",
                        vec![1, 2, u8::try_from(k % 100).unwrap()],
                        VectorSimilarityFunction::Euclidean,
                    )
                    .unwrap(),
                );
            }
            w.add_fields_document(&d).unwrap();
        }
        w.delete_documents_by_term(&[Term::new("id", (seg * 40 + 3).to_string())])
            .unwrap();
        w.commit().unwrap();
    }
    w.close().unwrap();
}

fn open() -> (ByteBuffersDirectory, DirectoryReader) {
    let dir = ByteBuffersDirectory::new();
    index(&dir);
    let reader = DirectoryReader::open(&dir).unwrap();
    (dir, reader)
}

#[test]
fn a_real_index_passes() {
    let (_dir, reader) = open();
    assert!(reader.segment_readers().len() > 1);
    assert!(reader.num_docs() < reader.max_doc());
    check_reader(&reader).unwrap();
    for leaf in reader.segment_readers() {
        check_leaf_reader(leaf, false).unwrap();
    }
}

#[test]
fn reader_views_pass() {
    let (_dir, reader) = open();
    for leaf in reader.segment_readers() {
        let inner: Arc<dyn LeafReader> = Arc::new(leaf.clone());
        check_leaf_reader(&AssertingLeafReader::new(inner.clone()), true).unwrap();
        let fields: HashSet<String> = ["body", "ss", "p"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        check_leaf_reader(
            &FieldFilterLeafReader::new(inner.clone(), fields.clone(), false),
            true,
        )
        .unwrap();
        check_leaf_reader(
            &FieldFilterLeafReader::new(inner.clone(), fields, true),
            true,
        )
        .unwrap();
        check_leaf_reader(&AllDeletedFilterReader::new(inner), true).unwrap();
    }
}

/// What a lying view gets wrong.
#[derive(Clone, Copy, Debug)]
enum Lie {
    SumDocFreq,
    SumTotalTermFreq,
    DocCount,
    DocCountLow,
    Size,
    HasFreqs,
    NoMin,
    NoMax,
    MinAfterTerms,
    MaxBeforeTerms,
    DriftingNumeric,
    ZeroNorms,
    ShortLiveDocs,
    PointSize,
    PointDocCount,
    PointDims,
    VectorDimension,
    ByteVectorDimension,
}

struct Liar {
    lie: Lie,
    calls: AtomicUsize,
    live: FixedBitSet,
}

struct LyingTerms<'a>(Box<dyn Terms + 'a>, Lie);

impl Terms for LyingTerms<'_> {
    fn iterator(&self) -> Result<Box<dyn TermsEnum + '_>> {
        self.0.iterator()
    }
    fn size(&self) -> i64 {
        self.0.size() + i64::from(matches!(self.1, Lie::Size))
    }
    fn sum_total_term_freq(&self) -> i64 {
        self.0.sum_total_term_freq() + i64::from(matches!(self.1, Lie::SumTotalTermFreq))
    }
    fn sum_doc_freq(&self) -> i64 {
        self.0.sum_doc_freq() + i64::from(matches!(self.1, Lie::SumDocFreq))
    }
    fn doc_count(&self) -> i32 {
        self.0.doc_count() + i32::from(matches!(self.1, Lie::DocCount))
            - i32::from(matches!(self.1, Lie::DocCountLow))
    }
    fn has_freqs(&self) -> bool {
        self.0.has_freqs() != matches!(self.1, Lie::HasFreqs)
    }
    fn has_offsets(&self) -> bool {
        self.0.has_offsets()
    }
    fn has_positions(&self) -> bool {
        self.0.has_positions()
    }
    fn has_payloads(&self) -> bool {
        self.0.has_payloads()
    }
    fn min(&self) -> Result<Option<Vec<u8>>> {
        match self.1 {
            Lie::NoMin => Ok(None),
            Lie::MinAfterTerms => Ok(Some(vec![0xff])),
            _ => self.0.min(),
        }
    }
    fn max(&self) -> Result<Option<Vec<u8>>> {
        match self.1 {
            Lie::NoMax => Ok(None),
            Lie::MaxBeforeTerms => Ok(Some(vec![0x01])),
            _ => self.0.max(),
        }
    }
}

/// A numeric iterator whose values are shifted by `delta`, or all zero.
struct Shifted<'a>(Box<dyn NumericDocValues + 'a>, i64, bool);

impl DocIdSetIterator for Shifted<'_> {
    fn doc_id(&self) -> i32 {
        self.0.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.0.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.0.advance(target)
    }
    fn cost(&self) -> i64 {
        self.0.cost()
    }
}

impl DocValuesIterator for Shifted<'_> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        self.0.advance_exact(target)
    }
}

impl NumericDocValues for Shifted<'_> {
    fn long_value(&self) -> i64 {
        if self.2 {
            0
        } else {
            self.0.long_value() + self.1
        }
    }
}

struct LyingPoints<'a>(Box<dyn PointValues + 'a>, Lie);

impl PointValues for LyingPoints<'_> {
    fn intersect(&self, visitor: &mut dyn IntersectVisitor) -> Result<()> {
        self.0.intersect(visitor)
    }
    fn estimate_point_count(&self, visitor: &mut dyn IntersectVisitor) -> Result<i64> {
        self.0.estimate_point_count(visitor)
    }
    fn min_packed_value(&self) -> &[u8] {
        self.0.min_packed_value()
    }
    fn max_packed_value(&self) -> &[u8] {
        self.0.max_packed_value()
    }
    fn num_dimensions(&self) -> i32 {
        self.0.num_dimensions() + i32::from(matches!(self.1, Lie::PointDims))
    }
    fn num_index_dimensions(&self) -> i32 {
        self.0.num_index_dimensions()
    }
    fn bytes_per_dimension(&self) -> i32 {
        self.0.bytes_per_dimension()
    }
    fn size(&self) -> i64 {
        self.0.size() + i64::from(matches!(self.1, Lie::PointSize))
    }
    fn doc_count(&self) -> i32 {
        self.0.doc_count() + i32::from(matches!(self.1, Lie::PointDocCount))
    }
}

struct WideFloats<'a>(Box<dyn FloatVectorValues + 'a>);

impl FloatVectorValues for WideFloats<'_> {
    fn dimension(&self) -> usize {
        self.0.dimension() + 1
    }
    fn size(&self) -> i32 {
        self.0.size()
    }
    fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        self.0.ord_to_doc(ord)
    }
    fn vector_value(&self, ord: i32) -> Result<Vec<f32>> {
        self.0.vector_value(ord)
    }
}

struct WideBytes<'a>(Box<dyn ByteVectorValues + 'a>);

impl ByteVectorValues for WideBytes<'_> {
    fn dimension(&self) -> usize {
        self.0.dimension() + 1
    }
    fn size(&self) -> i32 {
        self.0.size()
    }
    fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        self.0.ord_to_doc(ord)
    }
    fn vector_value(&self, ord: i32) -> Result<Vec<u8>> {
        self.0.vector_value(ord)
    }
}

impl LeafFilter for Liar {
    fn wrap_terms<'a>(
        &'a self,
        field: &str,
        terms: Box<dyn Terms + 'a>,
    ) -> Result<Box<dyn Terms + 'a>> {
        if field == "body" {
            Ok(Box::new(LyingTerms(terms, self.lie)))
        } else {
            Ok(terms)
        }
    }
    fn wrap_numeric<'a>(
        &'a self,
        _field: &str,
        values: Box<dyn NumericDocValues + 'a>,
    ) -> Result<Box<dyn NumericDocValues + 'a>> {
        if !matches!(self.lie, Lie::DriftingNumeric) {
            return Ok(values);
        }
        let n = self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(Box::new(Shifted(
            values,
            i64::try_from(n % 2).unwrap(),
            false,
        )))
    }
    fn wrap_norms<'a>(
        &'a self,
        _field: &str,
        values: Box<dyn NumericDocValues + 'a>,
    ) -> Result<Box<dyn NumericDocValues + 'a>> {
        Ok(Box::new(Shifted(
            values,
            0,
            matches!(self.lie, Lie::ZeroNorms),
        )))
    }
    fn wrap_points<'a>(
        &'a self,
        _field: &str,
        values: Box<dyn PointValues + 'a>,
    ) -> Result<Box<dyn PointValues + 'a>> {
        Ok(Box::new(LyingPoints(values, self.lie)))
    }
    fn wrap_float_vectors<'a>(
        &'a self,
        _field: &str,
        values: Box<dyn FloatVectorValues + 'a>,
    ) -> Result<Box<dyn FloatVectorValues + 'a>> {
        if matches!(self.lie, Lie::VectorDimension) {
            Ok(Box::new(WideFloats(values)))
        } else {
            Ok(values)
        }
    }
    fn wrap_byte_vectors<'a>(
        &'a self,
        _field: &str,
        values: Box<dyn ByteVectorValues + 'a>,
    ) -> Result<Box<dyn ByteVectorValues + 'a>> {
        if matches!(self.lie, Lie::ByteVectorDimension) {
            Ok(Box::new(WideBytes(values)))
        } else {
            Ok(values)
        }
    }
    fn live_docs<'a>(&'a self, inner: Option<&'a FixedBitSet>) -> Option<&'a FixedBitSet> {
        if matches!(self.lie, Lie::ShortLiveDocs) {
            Some(&self.live)
        } else {
            inner
        }
    }
}

#[test]
fn lying_views_fail_with_lucenes_messages() {
    let (_dir, reader) = open();
    let inner: Arc<dyn LeafReader> = Arc::new(reader.segment_readers()[0].clone());
    let mut live = FixedBitSet::new(3);
    live.set(0);
    for (lie, message) in [
        (Lie::SumDocFreq, "!= recomputed sumDocFreq"),
        (Lie::SumTotalTermFreq, "!= recomputed sumTotalTermFreq"),
        (Lie::DocCount, "docCount > maxDoc"),
        (Lie::DocCountLow, "> docCount="),
        (Lie::Size, "termCount mismatch"),
        (Lie::HasFreqs, "should have hasFreqs=true but got false"),
        (Lie::NoMin, "has null minTerm but non-null maxTerm"),
        (Lie::NoMax, "has non-null minTerm but null maxTerm"),
        (Lie::MinAfterTerms, "invalid term"),
        (Lie::MaxBeforeTerms, "invalid term"),
        (Lie::DriftingNumeric, "report different values"),
        (Lie::ZeroNorms, "its norm value is 0"),
        (Lie::ShortLiveDocs, "!= maxDoc"),
        (Lie::PointSize, "claims to have size="),
        (Lie::PointDocCount, "claims to have docCount="),
        (Lie::PointDims, "but FieldInfo says"),
        (Lie::VectorDimension, "has vector values of dimension"),
        (Lie::ByteVectorDimension, "has vector values of dimension"),
    ] {
        let liar = Liar {
            lie,
            calls: AtomicUsize::new(0),
            live: live.clone(),
        };
        let view = FilterLeafReader::new(inner.clone(), Box::new(liar));
        let e = check_leaf_reader(&view, true).unwrap_err();
        assert!(e.contains(message), "{lie:?}: {e}");
    }
}

/// Indexes Lucene itself wrote -- postings with positions and offsets,
/// term vectors, every doc-values type, points, vectors -- pass.
#[test]
fn lucene_written_indexes_pass() {
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data");
    for name in [
        "document_fields/index",
        "doc_values_index",
        "points_index",
        "term_vectors_index",
        "vectors_index",
        "geo_points/index",
        "highlight/tvindex",
    ] {
        let dir = lucene_store::FsDirectory::open(data.join(name));
        let reader = DirectoryReader::open(&dir).unwrap();
        check_reader(&reader).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}
