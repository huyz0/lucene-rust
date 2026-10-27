//! The read side of the test framework's `AssertingCodec`
//! ([`AssertingCodecReader`]): an index written under `Asserting`
//! (`IndexWriter::set_codec`) with every kind of field, deletions and a
//! `Lucene104` segment beside it is read through the asserting producers:
//! every term and posting, every doc-values field through all five getters,
//! norms, skippers, points (range and whole-field walks), vectors, stored
//! fields, term vectors and live docs, with each answer checked against
//! what was written. Faulty segments make the checks fire.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use lucene_codecs::codec::{DocValuesWrite, Lucene104Codec};
use lucene_codecs::doc_values::{self, DenseField, SortedOrdsColumn};
use lucene_codecs::per_field_doc_values::Lucene90DocValuesFormat;
use lucene_codecs::per_field_postings::Lucene104PostingsFormat;

use lucene_codecs::field_infos::{DocValuesType, FieldInfo, FieldInfos, VectorSimilarityFunction};
use lucene_codecs::stored_fields::{self, StoredFieldVisitor, VisitStatus};
use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    BinaryDocValuesField, Document, Field, FieldType, IndexOptions, IntPoint, KnnByteVectorField,
    KnnFloatVectorField, NumericDocValuesField, SortedDocValuesField, SortedNumericDocValuesField,
    SortedSetDocValuesField, Store, StringField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::{IndexSortField, LuceneVersion};
use lucene_search::reader::filter::{FilterLeafReader, LeafFilter};
use lucene_search::reader::{
    BinaryDocValues, ByteVectorValues, CacheHelper, DocValuesSkipper, FloatVectorValues,
    IndexReader, IntersectVisitor, LeafReader, LeafReaderContext, NumericDocValues, PointValues,
    PostingsFlags, Relation, SortedDocValues, SortedNumericDocValues, SortedSetDocValues,
    TermVectorsDocument, Terms, NO_MORE_DOCS,
};
use lucene_search::Result;
use lucene_store::{Directory, FsDirectory};
use lucene_test_framework::codecs::asserting_installed::InstalledAssertingCodec;
use lucene_test_framework::codecs::asserting_reader::{
    AssertingCodecLeafReader, AssertingCodecPointValues, AssertingCodecReader,
    AssertingIntersectVisitor,
};
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};
const PER_SEGMENT: i64 = 60;
const DELETED: [i64; 3] = [3, 70, 150];

/// Document `i`: every field kind, the sparse ones on some documents only.
fn doc(i: i64) -> Document {
    let mut d = Document::new();
    d.add(StringField::new("id", format!("d{i}"), Store::Yes));
    let mut body = FieldType::new();
    body.set_index_options(IndexOptions::DocsAndFreqsAndPositions)
        .unwrap();
    body.set_store_term_vectors(true).unwrap();
    body.set_store_term_vector_positions(true).unwrap();
    d.add(Field::from_string("body", format!("w{} x{} common", i % 5, i % 7), body).unwrap());
    d.add(NumericDocValuesField::indexed_field("n", i));
    if i % 3 != 0 {
        d.add(BinaryDocValuesField::new("b", format!("b{i}").into_bytes()));
        d.add(SortedDocValuesField::new(
            "s",
            format!("s{}", i % 11).into_bytes(),
        ));
    }
    d.add(SortedNumericDocValuesField::new("sn", i % 4));
    d.add(SortedNumericDocValuesField::new("sn", -i));
    d.add(SortedSetDocValuesField::new(
        "ss",
        format!("t{}", i % 6).into_bytes(),
    ));
    d.add(SortedSetDocValuesField::new("ss", b"all".to_vec()));
    d.add(IntPoint::new("p", &[(i % 50) as i32]).unwrap());
    if i % 2 == 0 {
        d.add(IntPoint::new("p2", &[i as i32, -(i as i32)]).unwrap());
        d.add(
            KnnByteVectorField::new(
                "bv",
                vec![(i % 7) as u8, 1, (i % 3) as u8],
                VectorSimilarityFunction::Euclidean,
            )
            .unwrap(),
        );
    }
    d.add(
        KnnFloatVectorField::new(
            "v",
            vec![(i % 10) as f32, 1.0, (i % 3) as f32],
            VectorSimilarityFunction::Euclidean,
        )
        .unwrap(),
    );
    d
}

/// Three `Asserting` segments with deletions, then one `Lucene104` one.
fn write(dir: &dyn Directory) -> i64 {
    lucene_test_framework::codecs::register_codecs();
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_codec(Some(Arc::new(InstalledAssertingCodec::new())))
        .unwrap();
    w.set_max_full_flush_merge_wait_millis(0);
    for seg in 0..3 {
        for i in seg * PER_SEGMENT..(seg + 1) * PER_SEGMENT {
            w.add_fields_document(&doc(i)).unwrap();
        }
        w.commit().unwrap();
    }
    let terms: Vec<Term> = DELETED
        .iter()
        .map(|i| Term::new("id", format!("d{i}")))
        .collect();
    w.delete_documents_by_term(&terms).unwrap();
    w.set_codec(None).unwrap();
    for i in 3 * PER_SEGMENT..4 * PER_SEGMENT {
        w.add_fields_document(&doc(i)).unwrap();
    }
    w.commit().unwrap();
    4 * PER_SEGMENT
}

/// The stored id of a document.
#[derive(Default)]
struct IdVisitor(Option<i64>);

impl StoredFieldVisitor for IdVisitor {
    fn needs_field(&mut self, _field_number: i32) -> stored_fields::Result<VisitStatus> {
        Ok(VisitStatus::Yes)
    }
    // `id` is the only stored field.
    fn string_field(&mut self, _field_number: i32, value: &str) -> stored_fields::Result<()> {
        self.0 = Some(value[1..].parse().unwrap());
        Ok(())
    }
}

/// Collects the documents of a point walk.
#[derive(Default)]
struct Collect {
    lo: Vec<u8>,
    hi: Vec<u8>,
    docs: Vec<i32>,
}

impl IntersectVisitor for Collect {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        if max < self.lo.as_slice() || min > self.hi.as_slice() {
            Relation::CellOutsideQuery
        } else if min >= self.lo.as_slice() && max <= self.hi.as_slice() {
            Relation::CellInsideQuery
        } else {
            Relation::CellCrossesQuery
        }
    }
    fn visit(&mut self, doc: i32) {
        self.docs.push(doc);
    }
    fn visit_with_value(&mut self, doc: i32, value: &[u8]) {
        if value >= self.lo.as_slice() && value <= self.hi.as_slice() {
            self.docs.push(doc);
        }
    }
}

fn sortable(v: i32) -> Vec<u8> {
    ((v as u32) ^ 0x8000_0000).to_be_bytes().to_vec()
}

/// What one live document holds, as read back.
#[derive(Debug, Default, PartialEq)]
struct Read {
    n: i64,
    b: Option<Vec<u8>>,
    s: Option<Vec<u8>>,
    sn: Vec<i64>,
    ss: Vec<Vec<u8>>,
    v: Vec<f32>,
    bv: Option<Vec<u8>>,
    body_terms: usize,
}

/// What document `i` must read back as.
fn expected(i: i64) -> Read {
    let mut sn = vec![i % 4, -i];
    sn.sort();
    let mut ss = vec![format!("t{}", i % 6).into_bytes(), b"all".to_vec()];
    ss.sort();
    Read {
        n: i,
        b: (i % 3 != 0).then(|| format!("b{i}").into_bytes()),
        s: (i % 3 != 0).then(|| format!("s{}", i % 11).into_bytes()),
        sn,
        ss,
        v: vec![(i % 10) as f32, 1.0, (i % 3) as f32],
        bv: (i % 2 == 0).then(|| vec![(i % 7) as u8, 1, (i % 3) as u8]),
        body_terms: 3,
    }
}

/// Reads everything `leaf` holds through its getters, per live document by
/// stored id, plus the ids a range walk over `p` finds.
fn read_leaf(leaf: &dyn LeafReader) -> (BTreeMap<i64, Read>, Vec<i64>) {
    leaf.check_integrity().unwrap();
    let max_doc = leaf.max_doc();
    let live = |doc: i32| leaf.live_docs().is_none_or(|bits| bits.get(doc as usize));
    let mut ids = Vec::new();
    for doc in 0..max_doc {
        let mut v = IdVisitor::default();
        leaf.document(doc, &mut v).unwrap();
        ids.push(v.0.unwrap());
    }
    let mut out: BTreeMap<i32, Read> = (0..max_doc)
        .filter(|&d| live(d))
        .map(|d| (d, Read::default()))
        .collect();
    // Every field through every getter: only the field's own type answers.
    let names: Vec<String> = leaf
        .field_infos()
        .fields
        .iter()
        .map(|f| f.name.clone())
        .collect();
    for name in &names {
        let fi = leaf.field_infos().field_by_name(name).unwrap().clone();
        let ty = fi.doc_values_type;
        if let Some(mut dv) = leaf.numeric_doc_values(name).unwrap() {
            assert_eq!(ty, DocValuesType::Numeric);
            while dv.next_doc().unwrap() != NO_MORE_DOCS {
                if let Some(r) = out.get_mut(&dv.doc_id()) {
                    r.n = dv.long_value();
                }
            }
        }
        if let Some(mut dv) = leaf.binary_doc_values(name).unwrap() {
            assert_eq!(ty, DocValuesType::Binary);
            while dv.next_doc().unwrap() != NO_MORE_DOCS {
                if let Some(r) = out.get_mut(&dv.doc_id()) {
                    r.b = Some(dv.binary_value().to_vec());
                }
            }
        }
        if let Some(mut dv) = leaf.sorted_doc_values(name).unwrap() {
            assert_eq!(ty, DocValuesType::Sorted);
            while dv.next_doc().unwrap() != NO_MORE_DOCS {
                let ord = dv.ord_value();
                let value = dv.lookup_ord(ord).unwrap();
                if let Some(r) = out.get_mut(&dv.doc_id()) {
                    r.s = Some(value);
                }
            }
        }
        if let Some(mut dv) = leaf.sorted_numeric_doc_values(name).unwrap() {
            assert_eq!(ty, DocValuesType::SortedNumeric);
            while dv.next_doc().unwrap() != NO_MORE_DOCS {
                let values: Vec<i64> = (0..dv.doc_value_count())
                    .map(|_| dv.next_value().unwrap())
                    .collect();
                if let Some(r) = out.get_mut(&dv.doc_id()) {
                    r.sn = values;
                }
            }
        }
        if let Some(mut dv) = leaf.sorted_set_doc_values(name).unwrap() {
            assert_eq!(ty, DocValuesType::SortedSet);
            while dv.next_doc().unwrap() != NO_MORE_DOCS {
                let doc = dv.doc_id();
                let ords: Vec<i64> = (0..dv.doc_value_count())
                    .map(|_| dv.next_ord().unwrap())
                    .collect();
                let mut values = Vec::new();
                for ord in ords {
                    values.push(lookup_set_ord(&mut *dv, ord));
                }
                if let Some(r) = out.get_mut(&doc) {
                    r.ss = values;
                }
            }
        }
        if let Some(mut skipper) = leaf.doc_values_skipper(name).unwrap() {
            assert_eq!(name, "n");
            let mut target = 0;
            loop {
                skipper.advance(target);
                if skipper.min_doc_id(0) == NO_MORE_DOCS {
                    break;
                }
                assert!(skipper.min_doc_id(0) <= skipper.max_doc_id(0));
                assert!(skipper.min_value(0) <= skipper.max_value(0));
                target = skipper.max_doc_id(0) + 1;
            }
        }
        if let Some(mut norms) = leaf.norm_values(name).unwrap() {
            assert_eq!(name, "body");
            let mut count = 0;
            while norms.next_doc().unwrap() != NO_MORE_DOCS {
                count += 1;
            }
            assert_eq!(count, max_doc);
        }
        if let Some(values) = leaf.float_vector_values(name).unwrap() {
            for ord in 0..values.size() {
                let doc = values.ord_to_doc(ord).unwrap();
                let v = values.vector_value(ord).unwrap();
                if let Some(r) = out.get_mut(&doc) {
                    r.v = v;
                }
            }
        }
        if let Some(values) = leaf.byte_vector_values(name).unwrap() {
            for ord in 0..values.size() {
                let doc = values.ord_to_doc(ord).unwrap();
                let v = values.vector_value(ord).unwrap();
                if let Some(r) = out.get_mut(&doc) {
                    r.bv = Some(v);
                }
            }
        }
        if let Some(terms) = leaf.terms(name).unwrap() {
            let mut te = terms.iterator().unwrap();
            while te.next().unwrap().is_some() {
                let mut postings = te.postings(PostingsFlags::Positions).unwrap();
                while postings.next_doc().unwrap() != NO_MORE_DOCS {
                    for _ in 0..postings.freq() {
                        postings.next_position().unwrap();
                    }
                    if name == "body" {
                        if let Some(r) = out.get_mut(&postings.doc_id()) {
                            r.body_terms += 1;
                        }
                    }
                }
            }
        }
    }
    for doc in 0..max_doc {
        let tv = leaf
            .term_vectors(doc)
            .unwrap()
            .expect("body's term vectors");
        assert_eq!(tv.fields.len(), 1);
    }
    // Points: a whole-field walk on `p2` and a range on `p`.
    let p2 = leaf.point_values("p2").unwrap().expect("p2's points");
    let mut all = Collect {
        lo: vec![0; 8],
        hi: vec![0xff; 8],
        docs: Vec::new(),
    };
    p2.intersect(&mut all).unwrap();
    assert_eq!(all.docs.len() as i64, p2.size());
    let p = leaf.point_values("p").unwrap().expect("p's points");
    let mut range = Collect {
        lo: sortable(10),
        hi: sortable(29),
        docs: Vec::new(),
    };
    assert!(p.estimate_point_count(&mut range).unwrap() > 0);
    p.intersect(&mut range).unwrap();
    let mut in_range: Vec<i64> = range
        .docs
        .iter()
        .filter(|&&d| live(d))
        .map(|&d| ids[d as usize])
        .collect();
    in_range.sort();
    let reads = out
        .into_iter()
        .map(|(doc, r)| (ids[doc as usize], r))
        .collect();
    (reads, in_range)
}

/// `lookupOrd` of a sorted-set field's ordinal, through a fresh `ss`.
fn lookup_set_ord(dv: &mut dyn SortedSetDocValues, ord: i64) -> Vec<u8> {
    dv.lookup_ord(ord).unwrap()
}

#[test]
fn asserting_segments_read_through_the_asserting_producers() {
    let tmp = TempDir::new("asserting-read-path");
    let dir = FsDirectory::open(tmp.path());
    let total = write(&dir);
    let reader = AssertingCodecReader::open(&dir).unwrap();
    let codecs: Vec<&str> = reader
        .delegate()
        .segment_infos
        .segments
        .iter()
        .map(|s| s.codec_name.as_str())
        .collect();
    assert_eq!(codecs, ["Asserting", "Asserting", "Asserting", "Lucene104"]);
    assert_eq!(reader.leaf_readers().len(), 4);
    assert_eq!(reader.max_doc(), total as i32);
    assert_eq!(reader.num_docs(), (total - DELETED.len() as i64) as i32);
    let mut all = BTreeMap::new();
    let mut in_range = Vec::new();
    for ctx in reader.leaves() {
        let (reads, range) = read_leaf(ctx.reader);
        all.extend(reads);
        in_range.extend(range);
    }
    in_range.sort();
    let live = |i: &i64| !DELETED.contains(i);
    let want: BTreeMap<i64, Read> = (0..total).filter(live).map(|i| (i, expected(i))).collect();
    assert_eq!(all, want);
    let want_range: Vec<i64> = (0..total)
        .filter(live)
        .filter(|i| (10..30).contains(&(i % 50)))
        .collect();
    assert_eq!(in_range, want_range);

    // A reopen after a merge under the codec reads through it again.
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_codec(Some(Arc::new(InstalledAssertingCodec::new())))
        .unwrap();
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    drop(w);
    let reopened = reader.open_if_changed(&dir).unwrap().expect("a new commit");
    assert_eq!(reopened.leaf_readers().len(), 1);
    let (reads, _) = read_leaf(reopened.leaves()[0].reader);
    assert_eq!(reads, want);
}

// ---------------------------------------------------------------------------
// Faulty segments
// ---------------------------------------------------------------------------

/// Runs `f`, which must panic with a message containing `needle`.
fn fires(needle: &str, f: impl FnOnce()) {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err(needle);
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(msg.contains(needle), "{msg:?} lacks {needle:?}");
}

/// A segment whose field infos (and, when given, live docs) say what it is
/// told.
struct Relabeled {
    in_: Arc<dyn LeafReader>,
    infos: FieldInfos,
    live: Option<FixedBitSet>,
}

impl IndexReader for Relabeled {
    fn max_doc(&self) -> i32 {
        self.in_.max_doc()
    }
    fn num_docs(&self) -> i32 {
        self.in_.num_docs()
    }
    fn leaves(&self) -> Vec<LeafReaderContext<'_>> {
        Vec::new()
    }
    fn reader_cache_helper(&self) -> Option<&CacheHelper> {
        None
    }
}

impl LeafReader for Relabeled {
    fn field_infos(&self) -> &FieldInfos {
        &self.infos
    }
    fn live_docs(&self) -> Option<&FixedBitSet> {
        self.live.as_ref().or_else(|| self.in_.live_docs())
    }
    fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms + '_>>> {
        self.in_.terms(field)
    }
    fn numeric_doc_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        self.in_.numeric_doc_values(field)
    }
    fn binary_doc_values(&self, field: &str) -> Result<Option<Box<dyn BinaryDocValues + '_>>> {
        self.in_.binary_doc_values(field)
    }
    fn sorted_doc_values(&self, field: &str) -> Result<Option<Box<dyn SortedDocValues + '_>>> {
        self.in_.sorted_doc_values(field)
    }
    fn sorted_numeric_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedNumericDocValues + '_>>> {
        self.in_.sorted_numeric_doc_values(field)
    }
    fn sorted_set_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedSetDocValues + '_>>> {
        self.in_.sorted_set_doc_values(field)
    }
    fn norm_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        self.in_.norm_values(field)
    }
    fn point_values(&self, field: &str) -> Result<Option<Box<dyn PointValues + '_>>> {
        self.in_.point_values(field)
    }
    fn float_vector_values(&self, field: &str) -> Result<Option<Box<dyn FloatVectorValues + '_>>> {
        self.in_.float_vector_values(field)
    }
    fn byte_vector_values(&self, field: &str) -> Result<Option<Box<dyn ByteVectorValues + '_>>> {
        self.in_.byte_vector_values(field)
    }
    fn document(&self, doc: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        self.in_.document(doc, visitor)
    }
    fn term_vectors(&self, doc: i32) -> Result<Option<TermVectorsDocument>> {
        self.in_.term_vectors(doc)
    }
    fn index_sort(&self) -> Option<&[IndexSortField]> {
        None
    }
    fn core_cache_helper(&self) -> Option<&CacheHelper> {
        None
    }
    fn doc_values_skipper(&self, field: &str) -> Result<Option<DocValuesSkipper<'_>>> {
        self.in_.doc_values_skipper(field)
    }
}

/// Live docs of the wrong length.
struct WrongLiveDocs(FixedBitSet);

impl LeafFilter for WrongLiveDocs {
    fn live_docs<'a>(&'a self, _inner: Option<&'a FixedBitSet>) -> Option<&'a FixedBitSet> {
        Some(&self.0)
    }
}

#[test]
fn faulty_segments_fire() {
    let tmp = TempDir::new("asserting-read-faults");
    let dir = FsDirectory::open(tmp.path());
    write(&dir);
    let reader = AssertingCodecReader::open(&dir).unwrap();
    let inner: Arc<dyn LeafReader> = Arc::new(reader.delegate().segment_readers()[0].clone());
    let relabel = |f: &dyn Fn(&mut FieldInfos)| {
        let mut infos = inner.field_infos().clone();
        f(&mut infos);
        AssertingCodecLeafReader::new(Arc::new(Relabeled {
            in_: Arc::clone(&inner),
            infos,
            live: None,
        }))
    };
    let set = |infos: &mut FieldInfos,
               name: &str,
               edit: &dyn Fn(&mut lucene_codecs::field_infos::FieldInfo)| {
        let fi = infos.fields.iter_mut().find(|f| f.name == name).unwrap();
        edit(fi);
    };
    fires(
        "Numeric doc values for n, whose field info has none",
        || {
            let r = relabel(&|i| set(i, "n", &|f| f.doc_values_type = DocValuesType::None));
            let _ = r.numeric_doc_values("n");
        },
    );
    fires("the Binary doc-values producer has no values for n", || {
        let r = relabel(&|i| set(i, "n", &|f| f.doc_values_type = DocValuesType::Binary));
        let _ = r.binary_doc_values("n");
    });
    fires("norms for body, whose field info has none", || {
        let r = relabel(&|i| set(i, "body", &|f| f.omit_norms = true));
        let _ = r.norm_values("body");
    });
    fires("the norms producer has no norms for id", || {
        let r = relabel(&|i| set(i, "id", &|f| f.omit_norms = false));
        let _ = r.norm_values("id");
    });
    fires(
        "point values for p, whose field info has no point dimensions",
        || {
            let r = relabel(&|i| set(i, "p", &|f| f.point_dimension_count = 0));
            let _ = r.point_values("p");
        },
    );
    fires("float vectors for v, whose field info has none", || {
        let r = relabel(&|i| set(i, "v", &|f| f.vector_dimension = 0));
        let _ = r.float_vector_values("v");
    });
    fires(
        "a skipper for n, whose field info has no skip index",
        || {
            let r = relabel(&|i| {
                set(i, "n", &|f| {
                    f.doc_values_skip_index_type =
                        lucene_codecs::field_infos::DocValuesSkipIndexType::None
                })
            });
            let _ = r.doc_values_skipper("n");
        },
    );
    fires("stored fields of doc 60 of 60", || {
        let r = AssertingCodecLeafReader::new(Arc::clone(&inner));
        let _ = r.document(60, &mut IdVisitor::default());
    });
    fires("maxDoc=60 but live docs hold 9 bits", || {
        let r = AssertingCodecLeafReader::new(Arc::new(FilterLeafReader::new(
            Arc::clone(&inner),
            Box::new(WrongLiveDocs(FixedBitSet::new(9))),
        )));
        let _ = r.live_docs();
    });
    fires("1 deleted documents but live docs clear 59", || {
        let mut bits = FixedBitSet::new(60);
        bits.set(0);
        let r = AssertingCodecLeafReader::new(Arc::new(Relabeled {
            in_: Arc::clone(&inner),
            infos: inner.field_infos().clone(),
            live: Some(bits),
        }));
        let _ = r.live_docs();
    });
    // The point statistics, straight on the asserting values.
    fires("points in 60 documents of 59", || {
        AssertingCodecPointValues::new(inner.point_values("p").unwrap().unwrap(), 59);
    });
}

/// A visitor that takes whatever it is handed.
struct Sink;

impl IntersectVisitor for Sink {
    fn compare(&mut self, _min: &[u8], _max: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }
    fn visit(&mut self, _doc: i32) {}
    fn visit_with_value(&mut self, _doc: i32, _value: &[u8]) {}
}

/// A one-dimensional, four-byte visitor with a budget of four.
fn visitor(sink: &mut Sink) -> AssertingIntersectVisitor<'_> {
    let mut v = AssertingIntersectVisitor::new(1, 1, 4, sink);
    v.grow(4);
    v
}

#[test]
fn the_intersect_visitor_checks_fire() {
    fires(
        "called add() more times than the last call to grow() reserved",
        || {
            let mut sink = Sink;
            let mut v = AssertingIntersectVisitor::new(1, 1, 4, &mut sink);
            v.visit(0);
        },
    );
    fires("point values are out of order", || {
        let mut sink = Sink;
        let mut v = visitor(&mut sink);
        v.visit_with_value(0, &[0, 0, 0, 2]);
        v.visit_with_value(1, &[0, 0, 0, 1]);
    });
    fires(
        "doc ids are out of order when point values are the same!",
        || {
            let mut sink = Sink;
            let mut v = visitor(&mut sink);
            v.visit_with_value(5, &[0, 0, 0, 2]);
            v.visit_with_value(4, &[0, 0, 0, 2]);
        },
    );
    fires("dim=0 of 1", || {
        let mut sink = Sink;
        let mut v = visitor(&mut sink);
        v.compare(&[0, 0, 0, 1], &[0, 0, 0, 3]);
        v.visit_with_value(0, &[0, 0, 0, 4]);
    });
    fires("visit(doc) after Some(CellCrossesQuery)", || {
        let mut sink = Sink;
        let mut v = visitor(&mut sink);
        v.compare(&[0, 0, 0, 1], &[0, 0, 0, 3]);
        v.visit(0);
    });
    fires("cell min above max in dim 0", || {
        let mut sink = Sink;
        let mut v = visitor(&mut sink);
        v.compare(&[0, 0, 0, 3], &[0, 0, 0, 1]);
    });
    fires("a packed value's length", || {
        let mut sink = Sink;
        let mut v = visitor(&mut sink);
        v.visit_with_value(0, &[0, 0, 1]);
    });
    // In order, within budget and cell: no complaint.
    let mut sink = Sink;
    let mut v = visitor(&mut sink);
    v.compare(&[0, 0, 0, 1], &[0, 0, 0, 3]);
    v.visit_with_value(0, &[0, 0, 0, 1]);
    v.visit_with_value(1, &[0, 0, 0, 1]);
    v.visit_with_value(0, &[0, 0, 0, 3]);
}

// ---------------------------------------------------------------------------
// The doc-values write checks
// ---------------------------------------------------------------------------

/// `Asserting` whose doc-values checks also record what they were shown:
/// each instance's suffix, fields and document count.
#[derive(Debug, Default)]
struct Recording {
    inner: InstalledAssertingCodec,
    seen: Mutex<Vec<(String, Vec<String>, i32)>>,
}

impl Lucene104Codec for Recording {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn doc_values_format_for_field(&self, field: &str) -> Lucene90DocValuesFormat {
        self.inner.doc_values_format_for_field(field)
    }
    fn check_doc_values(&self, write: &DocValuesWrite<'_>) -> std::io::Result<()> {
        let names = write
            .field_infos
            .fields
            .iter()
            .map(|f| f.name.clone())
            .collect();
        self.seen
            .lock()
            .unwrap()
            .push((write.segment_suffix.to_string(), names, write.max_doc));
        self.inner.check_doc_values(write)
    }
}

#[test]
fn doc_values_writes_are_checked_at_flush_merge_and_update() {
    lucene_test_framework::codecs::register_codecs();
    let tmp = TempDir::new("asserting-dv-checks");
    let dir = FsDirectory::open(tmp.path());
    let codec = Arc::new(Recording::default());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_codec(Some(Arc::clone(&codec) as Arc<dyn Lucene104Codec>))
        .unwrap();
    w.set_max_full_flush_merge_wait_millis(0);
    for seg in 0..2 {
        for i in seg * 10..(seg + 1) * 10 {
            w.add_fields_document(&doc(i)).unwrap();
        }
        w.commit().unwrap();
    }
    let flushed = codec.seen.lock().unwrap().clone();
    assert_eq!(flushed.len(), 2, "{flushed:?}");
    for (suffix, mut names, max_doc) in flushed {
        names.sort();
        assert_eq!(suffix, "Asserting_0");
        assert_eq!(names, ["b", "n", "s", "sn", "ss"]);
        assert_eq!(max_doc, 10);
    }
    w.update_numeric_doc_value(Term::new("id", "d4"), "n", -4)
        .unwrap();
    w.commit().unwrap();
    let updated = codec.seen.lock().unwrap()[2].clone();
    assert_eq!(updated.1, ["n"]);
    assert_eq!(updated.2, 10);
    assert!(
        updated.0.starts_with('1'),
        "a generation's suffix: {}",
        updated.0
    );
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    drop(w);
    let seen = codec.seen.lock().unwrap();
    let merged = seen.last().unwrap();
    assert_eq!((merged.0.as_str(), merged.2), ("Asserting_0", 20));
    assert_eq!(seen.len(), 4, "{seen:?}");
}

#[test]
fn a_sorted_field_with_an_unused_ordinal_fails_the_check() {
    let mut field = FieldInfo::new("s", 0);
    field.doc_values_type = DocValuesType::Sorted;
    let infos = FieldInfos {
        fields: vec![field],
    };
    let id = [7u8; 16];
    let column = SortedOrdsColumn {
        dict: vec![b"a".to_vec(), b"b".to_vec()],
        docs: vec![0, 1],
        ords: vec![0, 0],
    };
    let (dvm, dvd, dvs) = doc_values::write_fields_with_format(
        &[DenseField::SortedOrds(0, &column)],
        &[],
        2,
        &id,
        "Asserting_0",
        Lucene90DocValuesFormat::default(),
    )
    .unwrap();
    let write = DocValuesWrite {
        max_doc: 2,
        segment_id: &id,
        segment_suffix: "Asserting_0",
        format_name: "Asserting",
        field_infos: &infos,
        dvm: &dvm,
        dvd: &dvd,
        dvs: &dvs,
    };
    // `assert seen.cardinality() == values.getValueCount()`: one of two.
    fires("left: 1\n right: 2", || {
        let _ = InstalledAssertingCodec::new().check_doc_values(&write);
    });
    // Every ordinal used: no complaint.
    let column = SortedOrdsColumn {
        ords: vec![1, 0],
        ..column
    };
    let (dvm, dvd, dvs) = doc_values::write_fields_with_format(
        &[DenseField::SortedOrds(0, &column)],
        &[],
        2,
        &id,
        "Asserting_0",
        Lucene90DocValuesFormat::default(),
    )
    .unwrap();
    InstalledAssertingCodec::new()
        .check_doc_values(&DocValuesWrite {
            dvm: &dvm,
            dvd: &dvd,
            dvs: &dvs,
            ..write
        })
        .unwrap();
}

/// A subclass overriding the per-field choices: `body`'s postings and `n`'s
/// doc values on the plain defaults, the rest on `Asserting`'s. Each format
/// writes its own instance and the index reads back whole.
#[test]
fn overridden_per_field_formats_write_their_own_instances() {
    lucene_test_framework::codecs::register_codecs();
    let tmp = TempDir::new("asserting-overrides");
    let dir = FsDirectory::open(tmp.path());
    let codec = InstalledAssertingCodec::new()
        .with_postings_format(|field| {
            let format = Lucene104PostingsFormat::default();
            if field == "body" {
                format
            } else {
                format.named("Asserting")
            }
        })
        .with_doc_values_format(|field| {
            let format = Lucene90DocValuesFormat::default();
            if field == "n" {
                format
            } else {
                format.named("Asserting")
            }
        });
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_codec(Some(Arc::new(codec))).unwrap();
    w.set_use_compound_file(false);
    for i in 0..PER_SEGMENT {
        w.add_fields_document(&doc(i)).unwrap();
    }
    w.commit().unwrap();
    drop(w);
    let files = dir.list_all().unwrap();
    for f in [
        "_0_Lucene104_0.tim",
        "_0_Asserting_0.tim",
        "_0_Lucene90_0.dvd",
        "_0_Asserting_0.dvd",
    ] {
        assert!(files.iter().any(|n| n == f), "{f} in {files:?}");
    }
    for result in lucene_index::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
    let reader = AssertingCodecReader::open(&dir).unwrap();
    let (reads, _) = read_leaf(reader.leaves()[0].reader);
    let want: BTreeMap<i64, Read> = (0..PER_SEGMENT).map(|i| (i, expected(i))).collect();
    assert_eq!(reads, want);
}
