//! `AssertingLeafReader`: correct use of a real index passes through every
//! wrapper unchanged, and every check is seen to fire -- on misuse of a
//! real reader, or on a faulty inner one.

use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::field_infos::{FieldInfos, IndexOptions};
use lucene_codecs::stored_fields::{DocumentVisitor, StoredFieldVisitor};
use lucene_index::document::{
    BinaryDocValuesField, Document, Field, FieldType, NumericDocValuesField, SortedDocValuesField,
    SortedNumericDocValuesField, SortedSetDocValuesField, Store, StringField, TextField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::{IndexSortField, LuceneVersion};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::{
    BinaryDocValues, ByteVectorValues, CacheHelper, DocIdSetIterator, DocValuesIterator,
    FloatVectorValues, ImpactsEnum, IndexReader, LeafReader, LeafReaderContext,
    MaterializedPostings, NumericDocValues, PointValues, PostingsEnum, PostingsFlags,
    SortedDocValues, SortedNumericDocValues, SortedSetDocValues, TermVectorsDocument, Terms,
    TermsEnum, NO_MORE_DOCS,
};
use lucene_search::Result;
use lucene_store::ByteBuffersDirectory;
use lucene_test_framework::index::asserting_leaf_reader::{
    asserting_directory_reader, AssertingBinaryDocValues, AssertingLeafReader,
    AssertingNumericDocValues, AssertingPostingsEnum, AssertingSortedDocValues,
    AssertingSortedNumericDocValues, AssertingSortedSetDocValues, AssertingTerms,
    AssertingTermsEnum,
};
use lucene_util::fixed_bit_set::FixedBitSet;

const DOCS: i32 = 6;

fn index(dir: &ByteBuffersDirectory, segments: i32) {
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
    body.set_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
        .unwrap();
    for seg in 0..segments {
        for i in 0..DOCS {
            let k = seg * DOCS + i;
            let mut d = Document::new();
            d.add(StringField::new("id", k.to_string(), Store::Yes));
            d.add(
                Field::from_string("body", format!("word{k} common common"), body.clone()).unwrap(),
            );
            // Every other document has doc values, so an iterator can land
            // on a document without one.
            if k % 2 == 0 {
                d.add(NumericDocValuesField::new("n", i64::from(k)));
                d.add(BinaryDocValuesField::new("b", vec![k as u8]));
                d.add(SortedDocValuesField::new("s", format!("s{k}")));
                d.add(SortedSetDocValuesField::new("ss", format!("a{k}")));
                d.add(SortedSetDocValuesField::new("ss", format!("b{k}")));
                d.add(SortedNumericDocValuesField::new("sn", i64::from(k)));
                d.add(SortedNumericDocValuesField::new("sn", i64::from(k) + 100));
            }
            w.add_fields_document(&d).unwrap();
        }
        w.commit().unwrap();
    }
    w.delete_documents_by_term(&[lucene_index::buffered_updates::Term::new("id", "1")])
        .unwrap();
    w.commit().unwrap();
    w.close().unwrap();
}

fn leaf(dir: &ByteBuffersDirectory) -> (DirectoryReader, Arc<dyn LeafReader>) {
    index(dir, 1);
    let reader = DirectoryReader::open(dir).unwrap();
    let leaf: Arc<dyn LeafReader> = Arc::new(reader.segment_readers()[0].clone());
    (reader, leaf)
}

/// Runs `f`, which must panic with a message containing `expected`.
#[track_caller]
fn fires(expected: &str, f: impl FnOnce()) {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("the check did not fire");
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap();
    assert!(msg.contains(expected), "{msg:?} lacks {expected:?}");
}

// ---------------------------------------------------------------------------
// Correct use passes through
// ---------------------------------------------------------------------------

#[test]
fn a_well_behaved_walk_passes_every_check() {
    let dir = ByteBuffersDirectory::new();
    let (_r, inner) = leaf(&dir);
    let r = AssertingLeafReader::new(Arc::clone(&inner));
    assert_eq!(r.max_doc(), DOCS);
    assert_eq!(r.num_docs(), DOCS - 1);
    assert_eq!(r.live_docs().unwrap().len(), DOCS as usize);
    assert_eq!(r.leaves().len(), 1);
    assert!(Arc::ptr_eq(r.delegate(), &inner));
    assert_eq!(
        r.reader_cache_helper().map(CacheHelper::key_id),
        inner.reader_cache_helper().map(CacheHelper::key_id)
    );
    assert_eq!(
        r.core_cache_helper().map(CacheHelper::key_id),
        inner.core_cache_helper().map(CacheHelper::key_id)
    );
    assert_eq!(
        r.field_infos().fields.len(),
        inner.field_infos().fields.len()
    );

    // Terms, statistics and the full postings of every term.
    let terms = r.terms("body").unwrap().unwrap();
    assert_eq!(terms.doc_count(), DOCS);
    assert!(terms.sum_doc_freq() >= i64::from(DOCS));
    assert!(terms.sum_total_term_freq() >= terms.sum_doc_freq());
    assert!(terms.has_freqs() && terms.has_positions() && terms.has_offsets());
    assert!(!terms.has_payloads());
    assert_eq!(terms.size(), inner.terms("body").unwrap().unwrap().size());
    assert_eq!(terms.min().unwrap().unwrap(), b"common");
    assert_eq!(terms.max().unwrap().unwrap(), b"word5");
    let mut te = terms.iterator().unwrap();
    let mut seen = 0;
    while let Some(term) = te.next().unwrap() {
        let term = term.to_vec();
        assert_eq!(te.term().unwrap(), term.as_slice());
        let df = te.doc_freq().unwrap();
        assert!(te.total_term_freq().unwrap() >= i64::from(df));
        let mut p = te.postings(PostingsFlags::All).unwrap();
        assert_eq!(p.doc_id(), -1);
        assert!(p.cost() > 0);
        while p.next_doc().unwrap() != NO_MORE_DOCS {
            assert_eq!(p.doc_id(), p.doc_id());
            for _ in 0..p.freq() {
                assert!(p.next_position().unwrap() >= 0);
                assert!(p.start_offset() >= 0);
                assert!(p.end_offset() > p.start_offset());
                assert!(p.payload().is_none());
            }
        }
        let _ = te.impacts(PostingsFlags::Freqs).unwrap();
        seen += 1;
    }
    assert_eq!(seen, DOCS + 1);
    // Seeks reposition it.
    assert_eq!(te.try_seek_ceil(b"word2").unwrap(), SeekStatus::Found);
    let mut p = te.postings(PostingsFlags::Freqs).unwrap();
    assert_eq!(p.advance(1).unwrap(), 2);
    assert_eq!(p.advance(3).unwrap(), NO_MORE_DOCS);
    assert!(te.try_seek_exact(b"common").unwrap());
    assert_eq!(te.doc_freq().unwrap(), DOCS);
    // intersect is wrapped too.
    let dfa = lucene_codecs::regexp::RegexpPattern::parse("word[0-9]")
        .unwrap()
        .to_dfa()
        .unwrap();
    let mut it = terms.intersect(&dfa, None).unwrap();
    assert_eq!(it.next().unwrap().unwrap(), b"word0");
    assert_eq!(it.doc_freq().unwrap(), 1);

    // Doc values: nextDoc, advance and advanceExact on each kind.
    let mut n = r.numeric_doc_values("n").unwrap().unwrap();
    assert!(n.cost() > 0);
    assert_eq!(n.next_doc().unwrap(), 0);
    assert_eq!(n.long_value(), 0);
    assert_eq!(n.advance(1).unwrap(), 2);
    assert!(!n.advance_exact(3).unwrap());
    assert!(n.advance_exact(4).unwrap());
    assert_eq!(n.long_value(), 4);
    assert_eq!(n.next_doc().unwrap(), NO_MORE_DOCS);

    let mut b = r.binary_doc_values("b").unwrap().unwrap();
    assert_eq!(b.next_doc().unwrap(), 0);
    assert_eq!(b.binary_value(), [0]);
    assert_eq!(b.doc_id(), 0);

    let mut s = r.sorted_doc_values("s").unwrap().unwrap();
    assert_eq!(s.value_count(), 3);
    assert!(s.advance_exact(2).unwrap());
    assert_eq!(s.ord_value(), 1);
    assert_eq!(s.lookup_ord(1).unwrap(), b"s2");
    assert_eq!(s.lookup_term(b"s4").unwrap(), 2);
    assert_eq!(s.lookup_term(b"s3").unwrap(), -3);

    let mut sn = r.sorted_numeric_doc_values("sn").unwrap().unwrap();
    assert_eq!(sn.next_doc().unwrap(), 0);
    assert_eq!(sn.doc_value_count(), 2);
    assert_eq!(sn.next_value().unwrap(), 0);
    assert_eq!(sn.next_value().unwrap(), 100);
    assert_eq!(sn.advance(2).unwrap(), 2);
    if let Some(values) = sn.take_values() {
        assert_eq!(values, [2, 102]);
    }
    assert_eq!(
        r.sorted_numeric_is_singleton("sn"),
        inner.sorted_numeric_is_singleton("sn")
    );

    let mut ss = r.sorted_set_doc_values("ss").unwrap().unwrap();
    assert_eq!(ss.value_count(), 6);
    assert_eq!(ss.next_doc().unwrap(), 0);
    assert_eq!(ss.doc_value_count(), 2);
    assert_eq!(ss.next_ord().unwrap(), 0);
    assert_eq!(ss.next_ord().unwrap(), 3);
    assert_eq!(ss.lookup_ord(3).unwrap(), b"b0");
    assert_eq!(ss.lookup_term(b"b0").unwrap(), 3);
    assert!(ss.advance_exact(4).unwrap());
    if let Some(ords) = ss.take_ords() {
        assert_eq!(ords, [2, 5]);
    }

    let mut norms = r.norm_values("body").unwrap().unwrap();
    assert_eq!(norms.next_doc().unwrap(), 0);
    let _ = norms.long_value();

    // Absent fields stay absent; the rest is the inner reader's.
    assert!(r.terms("nope").unwrap().is_none());
    assert!(r.numeric_doc_values("b").unwrap().is_none());
    assert!(r.binary_doc_values("n").unwrap().is_none());
    assert!(r.sorted_doc_values("n").unwrap().is_none());
    assert!(r.sorted_numeric_doc_values("n").unwrap().is_none());
    assert!(r.sorted_set_doc_values("n").unwrap().is_none());
    assert!(r.norm_values("id").unwrap().is_none());
    assert!(r.point_values("n").unwrap().is_none());
    assert!(r.float_vector_values("n").unwrap().is_none());
    assert!(r.byte_vector_values("n").unwrap().is_none());
    assert!(r.doc_values_skipper("n").unwrap().is_none());
    assert!(r.index_sort().is_none());
    let mut v = DocumentVisitor::all();
    r.document(2, &mut v).unwrap();
    assert_eq!(v.into_document().fields.len(), 2);
    assert!(r.term_vectors(2).unwrap().is_none());
    r.check_integrity().unwrap();
}

#[test]
fn the_directory_reader_wraps_every_leaf() {
    let dir = ByteBuffersDirectory::new();
    index(&dir, 3);
    let reader = Arc::new(DirectoryReader::open(&dir).unwrap());
    let wrapped = asserting_directory_reader(Arc::clone(&reader)).unwrap();
    assert_eq!(wrapped.wrapped_leaves().len(), 3);
    assert_eq!(wrapped.max_doc(), 3 * DOCS);
    assert_eq!(wrapped.num_docs(), 3 * DOCS - 1);
    for leaf in wrapped.leaves() {
        let terms = leaf.reader.terms("body").unwrap().unwrap();
        let te = terms.iterator().unwrap();
        // The wrapped enum's state machine is there: term() before next().
        fires("term() called on unpositioned TermsEnum", || {
            let _ = te.term();
        });
    }
}

// ---------------------------------------------------------------------------
// Misuse of a real reader
// ---------------------------------------------------------------------------

#[test]
fn terms_enum_misuse_fires() {
    let dir = ByteBuffersDirectory::new();
    let (_r, inner) = leaf(&dir);
    let r = AssertingLeafReader::new(inner);
    let terms = r.terms("body").unwrap().unwrap();
    for (msg, call) in [
        ("term() called on unpositioned", 0),
        ("docFreq() called on unpositioned", 1),
        ("totalTermFreq() called on unpositioned", 2),
        ("docs(...) called on unpositioned", 3),
        ("docs(...) called on unpositioned", 4),
        ("ord() called on unpositioned", 5),
    ] {
        let mut te = terms.iterator().unwrap();
        fires(msg, || match call {
            0 => {
                let _ = te.term();
            }
            1 => {
                let _ = te.doc_freq();
            }
            2 => {
                let _ = te.total_term_freq();
            }
            3 => {
                let _ = te.postings(PostingsFlags::Freqs);
            }
            4 => {
                let _ = te.impacts(PostingsFlags::Freqs);
            }
            _ => {
                let _ = te.ord();
            }
        });
    }
    // Exhausted, a failed seekExact and a seekCeil past the end all leave
    // it unpositioned.
    let mut te = terms.iterator().unwrap();
    while te.next().unwrap().is_some() {}
    fires("next() called on unpositioned TermsEnum", || {
        let _ = te.next();
    });
    let mut te = terms.iterator().unwrap();
    assert!(!te.try_seek_exact(b"absent").unwrap());
    fires("docFreq() called on unpositioned", || {
        let _ = te.doc_freq();
    });
    assert_eq!(te.try_seek_ceil(b"zzz").unwrap(), SeekStatus::End);
    fires("next() called on unpositioned", || {
        let _ = te.next();
    });
    assert_eq!(te.try_seek_ceil(b"a").unwrap(), SeekStatus::NotFound);
    fires("Freqs should be requested on impacts", || {
        let _ = te.impacts(PostingsFlags::None);
    });
}

#[test]
fn postings_misuse_fires() {
    let dir = ByteBuffersDirectory::new();
    let (_r, inner) = leaf(&dir);
    let r = AssertingLeafReader::new(inner);
    let terms = r.terms("body").unwrap().unwrap();
    let postings = || {
        let mut te = terms.iterator().unwrap();
        assert!(te.try_seek_exact(b"word3").unwrap());
        te.postings(PostingsFlags::All).unwrap()
    };
    fires("freq() called before nextDoc()/advance()", || {
        let _ = postings().freq();
    });
    fires("nextPosition() called before nextDoc()/advance()", || {
        let _ = postings().next_position();
    });
    for (msg, call) in [
        ("startOffset() called before nextPosition()!", 0),
        ("endOffset() called before nextPosition()!", 1),
        ("getPayload() called before nextPosition()!", 2),
    ] {
        let mut p = postings();
        p.next_doc().unwrap();
        fires(msg, || match call {
            0 => {
                let _ = p.start_offset();
            }
            1 => {
                let _ = p.end_offset();
            }
            _ => {
                let _ = p.payload();
            }
        });
    }
    let mut p = postings();
    p.next_doc().unwrap();
    p.next_position().unwrap();
    fires("nextPosition() called more than freq() times!", || {
        let _ = p.next_position();
    });
    let mut p = postings();
    fires("target must be > docID(), got -1 <= -1", || {
        let _ = p.advance(-1);
    });
    let mut p = postings();
    assert_eq!(p.next_doc().unwrap(), 3);
    fires("target must be > docID(), got 3 <= 3", || {
        let _ = p.advance(3);
    });
    let mut p = postings();
    p.next_doc().unwrap();
    assert_eq!(p.next_doc().unwrap(), NO_MORE_DOCS);
    fires("freq() called after NO_MORE_DOCS", || {
        let _ = p.freq();
    });
    let mut p2 = postings();
    p2.advance(4).unwrap();
    fires("nextDoc() called after NO_MORE_DOCS", || {
        let _ = p2.next_doc();
    });
    fires("advance() called after NO_MORE_DOCS", || {
        let _ = p.advance(NO_MORE_DOCS);
    });
}

#[test]
fn doc_values_misuse_fires() {
    let dir = ByteBuffersDirectory::new();
    let (_r, inner) = leaf(&dir);
    let r = AssertingLeafReader::new(inner);
    fires("longValue() called on a document without a value", || {
        let _ = r.numeric_doc_values("n").unwrap().unwrap().long_value();
    });
    fires("longValue() called on a document without a value", || {
        let mut n = r.numeric_doc_values("n").unwrap().unwrap();
        assert!(!n.advance_exact(1).unwrap());
        let _ = n.long_value();
    });
    fires("binaryValue() called on a document without a value", || {
        let _ = r.binary_doc_values("b").unwrap().unwrap().binary_value();
    });
    fires("ordValue() called on a document without a value", || {
        let _ = r.sorted_doc_values("s").unwrap().unwrap().ord_value();
    });
    fires("lookupOrd(3) of valueCount=3", || {
        let _ = r.sorted_doc_values("s").unwrap().unwrap().lookup_ord(3);
    });
    fires("lookupOrd(-1) of valueCount=6", || {
        let _ = r
            .sorted_set_doc_values("ss")
            .unwrap()
            .unwrap()
            .lookup_ord(-1);
    });
    fires(
        "docValueCount() called on a document without a value",
        || {
            let _ = r
                .sorted_numeric_doc_values("sn")
                .unwrap()
                .unwrap()
                .doc_value_count();
        },
    );
    fires("nextValue() called on a document without a value", || {
        let _ = r
            .sorted_numeric_doc_values("sn")
            .unwrap()
            .unwrap()
            .next_value();
    });
    fires("takeValues() called on a document without a value", || {
        let _ = r
            .sorted_numeric_doc_values("sn")
            .unwrap()
            .unwrap()
            .take_values();
    });
    fires("valueUpto=2 in.docValueCount()=2", || {
        let mut sn = r.sorted_numeric_doc_values("sn").unwrap().unwrap();
        sn.next_doc().unwrap();
        sn.next_value().unwrap();
        sn.next_value().unwrap();
        let _ = sn.next_value();
    });
    fires("nextOrd() called on a document without a value", || {
        let _ = r.sorted_set_doc_values("ss").unwrap().unwrap().next_ord();
    });
    fires("takeOrds() called on a document without a value", || {
        let _ = r.sorted_set_doc_values("ss").unwrap().unwrap().take_ords();
    });
    // Java's sorted-set docValueCount forwards even before positioning.
    assert_eq!(
        r.sorted_set_doc_values("ss")
            .unwrap()
            .unwrap()
            .doc_value_count(),
        0
    );
    fires("nextOrd() called more than docValueCount() times", || {
        let mut ss = r.sorted_set_doc_values("ss").unwrap().unwrap();
        ss.next_doc().unwrap();
        ss.next_ord().unwrap();
        ss.next_ord().unwrap();
        let _ = ss.next_ord();
    });
    fires("advance(2) from doc 2", || {
        let mut n = r.numeric_doc_values("n").unwrap().unwrap();
        n.advance(2).unwrap();
        let _ = n.advance(2);
    });
    fires("advanceExact(6) from doc -1, maxDoc=6", || {
        let _ = r.numeric_doc_values("n").unwrap().unwrap().advance_exact(6);
    });
    fires("advanceExact(1) from doc 2", || {
        let mut n = r.numeric_doc_values("n").unwrap().unwrap();
        n.advance(2).unwrap();
        let _ = n.advance_exact(1);
    });
}

// ---------------------------------------------------------------------------
// Faulty inner readers
// ---------------------------------------------------------------------------

/// A leaf that reports what it is told: document counts, live docs, and
/// (when `dv_everywhere`) doc values and norms for every field, or (when
/// `dv_nowhere`) for none. The rest is `in_`'s.
struct FakeLeaf {
    in_: Arc<dyn LeafReader>,
    max_doc: i32,
    num_docs: i32,
    live: Option<FixedBitSet>,
    dv_everywhere: bool,
    dv_nowhere: bool,
}

impl FakeLeaf {
    fn over(in_: Arc<dyn LeafReader>) -> Self {
        Self {
            max_doc: in_.max_doc(),
            num_docs: in_.num_docs(),
            live: in_.live_docs().cloned(),
            in_,
            dv_everywhere: false,
            dv_nowhere: false,
        }
    }
    fn dv<T>(&self, inner: Result<Option<T>>, fake: impl FnOnce() -> T) -> Result<Option<T>> {
        if self.dv_everywhere {
            Ok(Some(fake()))
        } else if self.dv_nowhere {
            Ok(None)
        } else {
            inner
        }
    }
}

impl IndexReader for FakeLeaf {
    fn max_doc(&self) -> i32 {
        self.max_doc
    }
    fn num_docs(&self) -> i32 {
        self.num_docs
    }
    fn leaves(&self) -> Vec<LeafReaderContext<'_>> {
        Vec::new()
    }
    fn reader_cache_helper(&self) -> Option<&CacheHelper> {
        self.in_.reader_cache_helper()
    }
}

impl LeafReader for FakeLeaf {
    fn field_infos(&self) -> &FieldInfos {
        self.in_.field_infos()
    }
    fn live_docs(&self) -> Option<&FixedBitSet> {
        self.live.as_ref()
    }
    fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms + '_>>> {
        self.in_.terms(field)
    }
    fn numeric_doc_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        self.dv(self.in_.numeric_doc_values(field), || {
            Box::new(FakeDv::default()) as Box<dyn NumericDocValues>
        })
    }
    fn binary_doc_values(&self, field: &str) -> Result<Option<Box<dyn BinaryDocValues + '_>>> {
        self.dv(self.in_.binary_doc_values(field), || {
            Box::new(FakeDv::default()) as Box<dyn BinaryDocValues>
        })
    }
    fn sorted_doc_values(&self, field: &str) -> Result<Option<Box<dyn SortedDocValues + '_>>> {
        self.dv(self.in_.sorted_doc_values(field), || {
            Box::new(FakeDv::default()) as Box<dyn SortedDocValues>
        })
    }
    fn sorted_numeric_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedNumericDocValues + '_>>> {
        self.dv(self.in_.sorted_numeric_doc_values(field), || {
            Box::new(FakeDv::default()) as Box<dyn SortedNumericDocValues>
        })
    }
    fn sorted_set_doc_values(
        &self,
        field: &str,
    ) -> Result<Option<Box<dyn SortedSetDocValues + '_>>> {
        self.dv(self.in_.sorted_set_doc_values(field), || {
            Box::new(FakeDv::default()) as Box<dyn SortedSetDocValues>
        })
    }
    fn norm_values(&self, field: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        self.dv(self.in_.norm_values(field), || {
            Box::new(FakeDv::default()) as Box<dyn NumericDocValues>
        })
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
        self.in_.core_cache_helper()
    }
}

#[test]
fn a_faulty_leaf_fires() {
    let dir = ByteBuffersDirectory::new();
    let (_r, inner) = leaf(&dir);
    let fake = |f: &dyn Fn(&mut FakeLeaf)| {
        let mut leaf = FakeLeaf::over(Arc::clone(&inner));
        f(&mut leaf);
        Arc::new(leaf) as Arc<dyn LeafReader>
    };
    fires("maxDoc=-1", || {
        AssertingLeafReader::new(fake(&|l| l.max_doc = -1));
    });
    fires("numDocs=7 > maxDoc=6", || {
        AssertingLeafReader::new(fake(&|l| l.num_docs = 7));
    });
    fires("hasDeletions() but numDeletedDocs=0", || {
        AssertingLeafReader::new(fake(&|l| l.num_docs = 6));
    });
    fires("maxDoc=6 but live docs hold 9 bits", || {
        let r = AssertingLeafReader::new(fake(&|l| l.live = Some(FixedBitSet::new(9))));
        let _ = r.live_docs();
    });
    fires("no live docs but maxDoc=6 numDocs=5", || {
        let r = AssertingLeafReader::new(fake(&|l| l.live = None));
        let _ = r.live_docs();
    });
    // Doc values the field infos do not promise, and promised ones missing.
    let everywhere = AssertingLeafReader::new(fake(&|l| l.dv_everywhere = true));
    let nowhere = AssertingLeafReader::new(fake(&|l| l.dv_nowhere = true));
    fires(
        "Numeric doc values for b, whose field info says Some(Binary)",
        || {
            let _ = everywhere.numeric_doc_values("b");
        },
    );
    fires(
        "no Numeric doc values for n, whose field info has them",
        || {
            let _ = nowhere.numeric_doc_values("n");
        },
    );
    fires(
        "Binary doc values for nope, whose field info says None",
        || {
            let _ = everywhere.binary_doc_values("nope");
        },
    );
    fires("no Sorted doc values for s", || {
        let _ = nowhere.sorted_doc_values("s");
    });
    fires("SortedNumeric doc values for n", || {
        let _ = everywhere.sorted_numeric_doc_values("n");
    });
    fires("SortedSet doc values for n", || {
        let _ = everywhere.sorted_set_doc_values("n");
    });
    fires("norms for id, whose field info has none", || {
        let _ = everywhere.norm_values("id");
    });
    fires("no norms for body, whose field info has them", || {
        let _ = nowhere.norm_values("body");
    });
}

/// `Terms` reporting what it is told.
struct FakeTerms {
    doc_count: i32,
    sum_df: i64,
    sum_ttf: i64,
    freqs: bool,
}

impl Terms for FakeTerms {
    fn iterator(&self) -> Result<Box<dyn TermsEnum + '_>> {
        Ok(Box::new(FakeTermsEnum {
            df: 1,
            ttf: 1,
            positioned: false,
        }))
    }
    fn size(&self) -> i64 {
        -1
    }
    fn sum_total_term_freq(&self) -> i64 {
        self.sum_ttf
    }
    fn sum_doc_freq(&self) -> i64 {
        self.sum_df
    }
    fn doc_count(&self) -> i32 {
        self.doc_count
    }
    fn has_freqs(&self) -> bool {
        self.freqs
    }
    fn has_offsets(&self) -> bool {
        false
    }
    fn has_positions(&self) -> bool {
        false
    }
    fn has_payloads(&self) -> bool {
        false
    }
}

/// Statistics checks preserve delegate order and stop at Java's first failure.
#[test]
fn term_statistics_preserve_java_callback_order_and_short_circuit() {
    use std::cell::RefCell;

    struct ObservedTerms<'a> {
        inner: FakeTerms,
        calls: &'a RefCell<Vec<&'static str>>,
    }
    impl Terms for ObservedTerms<'_> {
        fn iterator(&self) -> Result<Box<dyn TermsEnum + '_>> {
            self.inner.iterator()
        }
        fn size(&self) -> i64 {
            self.inner.size()
        }
        fn sum_total_term_freq(&self) -> i64 {
            self.calls.borrow_mut().push("sum-total");
            self.inner.sum_total_term_freq()
        }
        fn sum_doc_freq(&self) -> i64 {
            self.calls.borrow_mut().push("sum-doc");
            self.inner.sum_doc_freq()
        }
        fn doc_count(&self) -> i32 {
            self.calls.borrow_mut().push("doc-count");
            self.inner.doc_count()
        }
        fn has_freqs(&self) -> bool {
            self.calls.borrow_mut().push("has-freqs");
            self.inner.has_freqs()
        }
        fn has_offsets(&self) -> bool {
            false
        }
        fn has_positions(&self) -> bool {
            false
        }
        fn has_payloads(&self) -> bool {
            false
        }
    }
    for freqs in [true, false] {
        let calls = RefCell::new(Vec::new());
        let terms = AssertingTerms::new(Box::new(ObservedTerms {
            inner: FakeTerms {
                doc_count: 1,
                sum_df: 2,
                sum_ttf: 2,
                freqs,
            },
            calls: &calls,
        }));
        assert_eq!(terms.sum_total_term_freq(), 2);
        let expected = if freqs {
            vec!["sum-total", "has-freqs", "sum-doc", "doc-count"]
        } else {
            vec!["sum-total", "has-freqs", "sum-doc", "sum-doc", "doc-count"]
        };
        assert_eq!(*calls.borrow(), expected);
    }
    let calls = RefCell::new(Vec::new());
    let terms = AssertingTerms::new(Box::new(ObservedTerms {
        inner: FakeTerms {
            doc_count: 0,
            sum_df: 2,
            sum_ttf: 3,
            freqs: false,
        },
        calls: &calls,
    }));
    fires("sumTotalTermFreq=3 != sumDocFreq=2 without freqs", || {
        terms.sum_total_term_freq();
    });
    assert_eq!(*calls.borrow(), ["sum-total", "has-freqs", "sum-doc"]);
}

/// A `TermsEnum` of one term with the given statistics.
struct FakeTermsEnum {
    df: i32,
    ttf: i64,
    positioned: bool,
}

impl TermsEnum for FakeTermsEnum {
    fn next(&mut self) -> Result<Option<&[u8]>> {
        self.positioned = !self.positioned;
        Ok(self.positioned.then_some(b"t".as_slice()))
    }
    fn term(&self) -> Option<&[u8]> {
        self.positioned.then_some(b"t".as_slice())
    }
    fn try_seek_ceil(&mut self, _target: &[u8]) -> Result<SeekStatus> {
        Ok(SeekStatus::Found)
    }
    fn doc_freq(&mut self) -> Result<i32> {
        Ok(self.df)
    }
    fn total_term_freq(&mut self) -> Result<i64> {
        Ok(self.ttf)
    }
    fn postings(&mut self, _flags: PostingsFlags) -> Result<Box<dyn PostingsEnum>> {
        Ok(Box::new(MaterializedPostings::new(vec![0], vec![1], None)?))
    }
    fn impacts(&mut self, flags: PostingsFlags) -> Result<Box<dyn ImpactsEnum>> {
        Ok(Box::new(lucene_search::reader::SlowImpactsEnum::new(
            self.postings(flags)?,
        )))
    }
    fn seek_exact_ord(&mut self, _ord: i64) -> Result<()> {
        Ok(())
    }
    fn ord(&self) -> Result<i64> {
        Ok(0)
    }
}

#[test]
fn faulty_terms_fire() {
    let terms = |doc_count, sum_df, sum_ttf, freqs| {
        AssertingTerms::new(Box::new(FakeTerms {
            doc_count,
            sum_df,
            sum_ttf,
            freqs,
        }))
    };
    fires("docCount=0", || {
        terms(0, 0, 0, true).doc_count();
    });
    fires("sumDocFreq=1 < docCount=2", || {
        terms(2, 1, 1, true).sum_doc_freq();
    });
    fires("sumTotalTermFreq=3 != sumDocFreq=2 without freqs", || {
        terms(1, 2, 3, false).sum_total_term_freq();
    });
    fires("sumTotalTermFreq=1 < sumDocFreq=2", || {
        terms(1, 2, 1, true).sum_total_term_freq();
    });
    let ok = terms(1, 2, 3, true);
    assert_eq!(ok.size(), -1);
    assert!(!ok.has_offsets() && !ok.has_positions() && !ok.has_payloads());
    assert_eq!(ok.min().unwrap().unwrap(), b"t");
    assert_eq!(ok.max().unwrap().unwrap(), b"t");

    let te = |df, ttf, freqs| {
        let mut te = AssertingTermsEnum::new(
            Box::new(FakeTermsEnum {
                df,
                ttf,
                positioned: false,
            }),
            freqs,
        );
        te.next().unwrap();
        te
    };
    fires("docFreq=0", || {
        let _ = te(0, 0, true).doc_freq();
    });
    fires("totalTermFreq=1 < docFreq=2", || {
        let _ = te(2, 1, true).total_term_freq();
    });
    fires("totalTermFreq=3 != docFreq=2 without freqs", || {
        let _ = te(2, 3, false).total_term_freq();
    });
    // seekExact(ord) positions the enum; ord() then answers.
    let mut by_ord = AssertingTermsEnum::new(
        Box::new(FakeTermsEnum {
            df: 1,
            ttf: 1,
            positioned: false,
        }),
        true,
    );
    by_ord.seek_exact_ord(0).unwrap();
    assert_eq!(by_ord.ord().unwrap(), 0);
}

/// A `PostingsEnum` that goes where it is told, whatever the caller asked.
#[derive(Default)]
struct FakePostings {
    /// What successive `next_doc`/`advance` calls return.
    moves: Vec<i32>,
    upto: usize,
    /// When set, what `doc_id` answers from its `n`th call on.
    lie_after: Option<(usize, i32)>,
    calls: Cell<usize>,
    doc: i32,
    freq: i32,
    position: i32,
    payload: Option<Vec<u8>>,
}

impl FakePostings {
    fn new(moves: &[i32]) -> Self {
        Self {
            moves: moves.to_vec(),
            doc: -1,
            freq: 1,
            ..Self::default()
        }
    }
    fn step(&mut self) -> i32 {
        self.doc = self.moves[self.upto];
        self.upto += 1;
        self.doc
    }
}

impl DocIdSetIterator for FakePostings {
    fn doc_id(&self) -> i32 {
        let n = self.calls.get();
        self.calls.set(n + 1);
        match self.lie_after {
            Some((after, doc)) if n >= after => doc,
            _ => self.doc,
        }
    }
    fn next_doc(&mut self) -> Result<i32> {
        Ok(self.step())
    }
    fn advance(&mut self, _target: i32) -> Result<i32> {
        Ok(self.step())
    }
    fn cost(&self) -> i64 {
        1
    }
}

impl PostingsEnum for FakePostings {
    fn freq(&self) -> i32 {
        self.freq
    }
    fn next_position(&mut self) -> Result<i32> {
        Ok(self.position)
    }
    fn start_offset(&self) -> i32 {
        -1
    }
    fn end_offset(&self) -> i32 {
        -1
    }
    fn payload(&self) -> Option<&[u8]> {
        self.payload.as_deref()
    }
}

#[test]
fn faulty_postings_fire() {
    let wrap = |p: FakePostings| AssertingPostingsEnum::new(Box::new(p));
    fires("backwards nextDoc from 3 to 2", || {
        let mut p = wrap(FakePostings::new(&[3, 2]));
        p.next_doc().unwrap();
        let _ = p.next_doc();
    });
    fires("backwards advance from: 5 to: 4", || {
        let mut p = wrap(FakePostings::new(&[4]));
        let _ = p.advance(5);
    });
    fires("docID() disagrees with 3", || {
        let mut p = wrap(FakePostings {
            lie_after: Some((1, 9)),
            ..FakePostings::new(&[3])
        });
        let _ = p.next_doc();
    });
    fires(" invalid docID() 9 instead of 3", || {
        let mut p = wrap(FakePostings {
            lie_after: Some((2, 9)),
            ..FakePostings::new(&[3])
        });
        p.next_doc().unwrap();
        let _ = p.doc_id();
    });
    fires("freq=0", || {
        let mut p = wrap(FakePostings {
            freq: 0,
            ..FakePostings::new(&[3])
        });
        p.next_doc().unwrap();
        let _ = p.freq();
    });
    fires("invalid position: -2", || {
        let mut p = wrap(FakePostings {
            position: -2,
            ..FakePostings::new(&[3])
        });
        p.next_doc().unwrap();
        let _ = p.next_position();
    });
    fires("getPayload() returned payload with invalid length!", || {
        let mut p = wrap(FakePostings {
            payload: Some(Vec::new()),
            ..FakePostings::new(&[3])
        });
        p.next_doc().unwrap();
        p.next_position().unwrap();
        let _ = p.payload();
    });
    // A position of -1 (none indexed) and a real payload pass.
    let mut p = wrap(FakePostings {
        position: -1,
        payload: Some(vec![1]),
        ..FakePostings::new(&[3])
    });
    p.next_doc().unwrap();
    assert_eq!(p.next_position().unwrap(), -1);
    assert_eq!(p.payload(), Some([1u8].as_slice()));
    assert_eq!((p.start_offset(), p.end_offset(), p.cost()), (-1, -1, 1));
}

/// Doc values of every kind that go where they are told.
struct FakeDv {
    /// What successive `next_doc`/`advance`/`advance_exact` land on.
    moves: Vec<i32>,
    upto: usize,
    doc: i32,
    /// What `doc_id` answers instead of the true document, when set.
    doc_id_lie: Option<i32>,
    cost: i64,
    /// Successive `value_count` answers; the last repeats.
    value_counts: Vec<i64>,
    value_count_calls: Cell<usize>,
    ord: i64,
    count: i32,
    lookup_term: i64,
    take: Vec<i64>,
}

impl Default for FakeDv {
    fn default() -> Self {
        Self {
            moves: vec![0, 1, 2, 3],
            upto: 0,
            doc: -1,
            doc_id_lie: None,
            cost: 1,
            value_counts: vec![2],
            value_count_calls: Cell::new(0),
            ord: 0,
            count: 1,
            lookup_term: 0,
            take: vec![0],
        }
    }
}

impl FakeDv {
    fn step(&mut self) -> i32 {
        self.doc = self.moves[self.upto];
        self.upto += 1;
        self.doc
    }
    fn count(&self) -> i64 {
        let n = self.value_count_calls.get();
        self.value_count_calls.set(n + 1);
        self.value_counts[n.min(self.value_counts.len() - 1)]
    }
}

impl DocIdSetIterator for FakeDv {
    fn doc_id(&self) -> i32 {
        self.doc_id_lie.unwrap_or(self.doc)
    }
    fn next_doc(&mut self) -> Result<i32> {
        Ok(self.step())
    }
    fn advance(&mut self, _target: i32) -> Result<i32> {
        Ok(self.step())
    }
    fn cost(&self) -> i64 {
        self.cost
    }
}

impl DocValuesIterator for FakeDv {
    fn advance_exact(&mut self, _target: i32) -> Result<bool> {
        self.step();
        Ok(true)
    }
}

impl NumericDocValues for FakeDv {
    fn long_value(&self) -> i64 {
        7
    }
}

impl BinaryDocValues for FakeDv {
    fn binary_value(&self) -> &[u8] {
        b"v"
    }
}

impl SortedDocValues for FakeDv {
    fn ord_value(&self) -> i32 {
        self.ord as i32
    }
    fn lookup_ord(&mut self, _ord: i32) -> Result<Vec<u8>> {
        Ok(b"v".to_vec())
    }
    fn value_count(&self) -> i32 {
        self.count() as i32
    }
    fn lookup_term(&mut self, _key: &[u8]) -> Result<i32> {
        Ok(self.lookup_term as i32)
    }
}

impl SortedNumericDocValues for FakeDv {
    fn doc_value_count(&self) -> i32 {
        self.count
    }
    fn next_value(&mut self) -> Result<i64> {
        Ok(7)
    }
    fn take_values(&mut self) -> Option<&[i64]> {
        Some(&self.take)
    }
}

impl SortedSetDocValues for FakeDv {
    fn doc_value_count(&self) -> i32 {
        self.count
    }
    fn next_ord(&mut self) -> Result<i64> {
        Ok(self.ord)
    }
    fn take_ords(&mut self) -> Option<&[i64]> {
        Some(&self.take)
    }
    fn lookup_ord(&mut self, _ord: i64) -> Result<Vec<u8>> {
        Ok(b"v".to_vec())
    }
    fn value_count(&self) -> i64 {
        self.count()
    }
    fn lookup_term(&mut self, _key: &[u8]) -> Result<i64> {
        Ok(self.lookup_term)
    }
}

fn numeric(dv: FakeDv) -> AssertingNumericDocValues<'static> {
    AssertingNumericDocValues::new(Box::new(dv), 4)
}

#[test]
fn doc_value_constructors_preserve_javas_position_checks() {
    let positioned = || {
        Box::new(FakeDv {
            doc: 2,
            moves: vec![3],
            ..FakeDv::default()
        })
    };
    let mut sorted = AssertingSortedDocValues::new(positioned(), 4);
    let mut numeric = AssertingSortedNumericDocValues::new(positioned(), 4);
    let mut set = AssertingSortedSetDocValues::new(positioned(), 4);
    assert_eq!(sorted.doc_id(), 2);
    assert_eq!(numeric.doc_id(), 2);
    assert_eq!(set.doc_id(), 2);
    assert_eq!(sorted.next_doc().unwrap(), 3);
    assert_eq!(numeric.next_doc().unwrap(), 3);
    assert_eq!(set.next_doc().unwrap(), 3);
    assert_eq!(sorted.ord_value(), 0);
    assert_eq!(numeric.next_value().unwrap(), 7);
    assert_eq!(set.next_ord().unwrap(), 0);
    fires("Binary doc values start on doc 2", || {
        AssertingBinaryDocValues::new(positioned(), 4);
    });
}

#[test]
fn faulty_doc_value_iterators_fire() {
    fires("Numeric doc values start on doc 2", || {
        numeric(FakeDv {
            doc: 2,
            ..FakeDv::default()
        });
    });
    fires("backwards nextDoc from 1 to 0", || {
        let mut dv = numeric(FakeDv {
            moves: vec![1, 0],
            ..FakeDv::default()
        });
        dv.next_doc().unwrap();
        let _ = dv.next_doc();
    });
    fires("went to doc 4 of maxDoc=4", || {
        let _ = numeric(FakeDv {
            moves: vec![4],
            ..FakeDv::default()
        })
        .next_doc();
    });
    fires("Numeric doc values: docID() disagrees", || {
        // A doc id that answers -1 whatever the iterator did: the first
        // nextDoc (to 0) then disagrees with it.
        let mut lying = numeric(FakeDv {
            doc_id_lie: Some(-1),
            ..FakeDv::default()
        });
        let _ = lying.next_doc();
    });
    // Java's single-valued advance does not query docID() after delegation.
    assert_eq!(
        numeric(FakeDv {
            doc_id_lie: Some(-1),
            ..FakeDv::default()
        })
        .advance(0)
        .unwrap(),
        0
    );
    fires("backwards advance from: 2 to: 1", || {
        let _ = numeric(FakeDv {
            moves: vec![1],
            ..FakeDv::default()
        })
        .advance(2);
    });
    fires("advanceExact(2) left the iterator on 0", || {
        let _ = numeric(FakeDv::default()).advance_exact(2);
    });
    fires("Numeric doc values: cost=-1", || {
        let _ = numeric(FakeDv {
            cost: -1,
            ..FakeDv::default()
        })
        .cost();
    });

    // Sorted.
    let sorted = |dv: FakeDv| AssertingSortedDocValues::new(Box::new(dv), 4);
    fires("Sorted doc values: valueCount=5 of maxDoc=4", || {
        sorted(FakeDv {
            value_counts: vec![5],
            ..FakeDv::default()
        });
    });
    fires("ord=2 of valueCount=2", || {
        let mut dv = sorted(FakeDv {
            ord: 2,
            ..FakeDv::default()
        });
        dv.next_doc().unwrap();
        let _ = dv.ord_value();
    });
    fires("valueCount changed from 2 to 3", || {
        let _ = sorted(FakeDv {
            value_counts: vec![2, 3],
            ..FakeDv::default()
        })
        .value_count();
    });
    fires("lookupTerm returned 2 of valueCount=2", || {
        let _ = sorted(FakeDv {
            lookup_term: 2,
            ..FakeDv::default()
        })
        .lookup_term(b"x");
    });

    // Sorted numeric.
    let sorted_numeric = |dv: FakeDv| AssertingSortedNumericDocValues::new(Box::new(dv), 4);
    fires("docValueCount=0", || {
        let mut dv = sorted_numeric(FakeDv {
            count: 0,
            ..FakeDv::default()
        });
        dv.next_doc().unwrap();
        let _ = dv.doc_value_count();
    });
    let mut dv = sorted_numeric(FakeDv::default());
    assert!(dv.advance_exact(0).unwrap());
    assert_eq!(dv.take_values(), Some([0i64].as_slice()));
    fires("valueUpto=1 in.docValueCount()=1", || {
        let _ = dv.next_value();
    });

    // Sorted set.
    let sorted_set = |dv: FakeDv| AssertingSortedSetDocValues::new(Box::new(dv), 4);
    fires("Sorted set doc values: valueCount=-1", || {
        sorted_set(FakeDv {
            value_counts: vec![-1],
            ..FakeDv::default()
        });
    });
    let mut empty = sorted_set(FakeDv {
        count: 0,
        ..FakeDv::default()
    });
    // Java forwards this getter even before positioning and for zero counts.
    assert_eq!(empty.doc_value_count(), 0);
    empty.next_doc().unwrap();
    assert_eq!(empty.doc_value_count(), 0);
    fires("ord=2 of valueCount=2", || {
        let mut dv = sorted_set(FakeDv {
            ord: 2,
            ..FakeDv::default()
        });
        dv.next_doc().unwrap();
        let _ = dv.next_ord();
    });
    fires("ords [5] of valueCount=2", || {
        let mut dv = sorted_set(FakeDv {
            take: vec![5],
            ..FakeDv::default()
        });
        dv.next_doc().unwrap();
        let _ = dv.take_ords();
    });
    fires("valueCount changed from 2 to 3", || {
        let _ = sorted_set(FakeDv {
            value_counts: vec![2, 3],
            ..FakeDv::default()
        })
        .value_count();
    });
    fires("lookupTerm returned 2 of valueCount=2", || {
        let _ = sorted_set(FakeDv {
            lookup_term: 2,
            ..FakeDv::default()
        })
        .lookup_term(b"x");
    });
    let mut ok = sorted_set(FakeDv::default());
    ok.next_doc().unwrap();
    assert_eq!(ok.take_ords(), Some([0i64].as_slice()));
    assert_eq!(ok.lookup_ord(1).unwrap(), b"v");

    // Binary passes its value through.
    let mut b = AssertingBinaryDocValues::new(Box::new(FakeDv::default()), 4);
    assert_eq!(b.next_doc().unwrap(), 0);
    assert_eq!(b.binary_value(), b"v");
}
