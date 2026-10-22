//! The leaf and composite reader wrappers over a small real index:
//! `AllDeletedFilterReader`, `FieldFilterLeafReader`, the mismatched
//! readers and `OwnCacheKeyMultiReader`.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use lucene_codecs::stored_fields::{DocumentVisitor, FieldValue};
use lucene_index::document::{
    BinaryDocValuesField, Document, Field, FieldType, NumericDocValuesField, SortedDocValuesField,
    SortedNumericDocValuesField, SortedSetDocValuesField, Store, StringField, TextField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::{CodecReader, CompositeReader, IndexReader, LeafReader, ReaderHandle};
use lucene_store::ByteBuffersDirectory;
use lucene_test_framework::index::reader_wrappers::{
    mismatched_directory_reader, shuffle_infos, AllDeletedFilterReader, FieldFilterLeafReader,
    MismatchedCodecReader, MismatchedLeafReader, OwnCacheKeyMultiReader,
};
use lucene_util::java_random::JavaRandom;

const FIELDS: [&str; 7] = ["id", "body", "n", "b", "s", "ss", "sn"];

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
    body.set_store_term_vectors(true).unwrap();
    for seg in 0..segments {
        for i in 0..5 {
            let k = seg * 5 + i;
            let mut d = Document::new();
            d.add(StringField::new("id", k.to_string(), Store::Yes));
            d.add(Field::from_string("body", format!("word{k} common"), body.clone()).unwrap());
            d.add(NumericDocValuesField::new("n", i64::from(k)));
            d.add(BinaryDocValuesField::new("b", vec![k as u8]));
            d.add(SortedDocValuesField::new("s", format!("s{k}")));
            d.add(SortedSetDocValuesField::new("ss", format!("ss{k}")));
            d.add(SortedNumericDocValuesField::new("sn", i64::from(k)));
            w.add_fields_document(&d).unwrap();
        }
        w.commit().unwrap();
    }
    w.close().unwrap();
}

fn leaf(dir: &ByteBuffersDirectory) -> (DirectoryReader, Arc<dyn LeafReader>) {
    index(dir, 1);
    let reader = DirectoryReader::open(dir).unwrap();
    let leaf: Arc<dyn LeafReader> = Arc::new(reader.segment_readers()[0].clone());
    (reader, leaf)
}

/// The stored fields of `doc`, as (field name, string value) pairs named
/// through `reader`'s own field infos.
fn stored(reader: &dyn LeafReader, doc: i32) -> Vec<(String, String)> {
    let mut v = DocumentVisitor::all();
    reader.document(doc, &mut v).unwrap();
    v.into_document()
        .fields
        .into_iter()
        .map(|f| {
            let name = reader
                .field_infos()
                .field_by_number(f.field_number)
                .unwrap()
                .name
                .clone();
            let value = match f.value {
                FieldValue::String(s) => s,
                other => format!("{other:?}"),
            };
            (name, value)
        })
        .collect()
}

/// Term-vector field names of `doc`, named through `reader`'s field infos.
fn vector_fields(reader: &dyn LeafReader, doc: i32) -> Option<Vec<String>> {
    reader.term_vectors(doc).unwrap().map(|tv| {
        tv.fields
            .iter()
            .map(|f| {
                reader
                    .field_infos()
                    .field_by_number(f.field_number)
                    .unwrap()
                    .name
                    .clone()
            })
            .collect()
    })
}

fn layout(infos: &lucene_codecs::field_infos::FieldInfos) -> Vec<(String, i32)> {
    infos
        .fields
        .iter()
        .map(|f| (f.name.clone(), f.number))
        .collect()
}

#[test]
fn all_deleted_filter_reader_deletes_everything() {
    let dir = ByteBuffersDirectory::new();
    let (_r, inner) = leaf(&dir);
    let r = AllDeletedFilterReader::new(Arc::clone(&inner));
    assert_eq!(r.max_doc(), 5);
    assert_eq!(r.num_docs(), 0);
    assert!(r.has_deletions());
    let live = r.live_docs().unwrap();
    assert!((0..5).all(|d| !live.get(d)));
    assert!(r.reader_cache_helper().is_none());
    assert_eq!(
        r.core_cache_helper().map(|h| h.key_id()),
        inner.core_cache_helper().map(|h| h.key_id())
    );
    // Everything else is the inner reader's.
    assert!(r.terms("body").unwrap().is_some());
    assert!(r.numeric_doc_values("n").unwrap().is_some());
    assert_eq!(stored(&r, 0), stored(inner.as_ref(), 0));
    assert_eq!(r.leaves().len(), 1);
    assert!(r.point_values("n").unwrap().is_none());
    assert!(r.float_vector_values("n").unwrap().is_none());
    assert!(r.byte_vector_values("n").unwrap().is_none());
    assert!(r.index_sort().is_none());
    r.check_integrity().unwrap();
}

#[test]
fn field_filter_leaf_reader_hides_the_other_fields() {
    let dir = ByteBuffersDirectory::new();
    let (_r, inner) = leaf(&dir);
    let keep: HashSet<String> = ["id", "s", "n"].iter().map(|s| s.to_string()).collect();
    let r = FieldFilterLeafReader::new(Arc::clone(&inner), keep.clone(), false);
    let names: Vec<&str> = r
        .field_infos()
        .fields
        .iter()
        .map(|f| f.name.as_str())
        .collect();
    assert_eq!(names.len(), 3);
    assert!(names.iter().all(|n| keep.contains(*n)));
    assert!(r.terms("id").unwrap().is_some());
    assert!(r.terms("body").unwrap().is_none());
    assert!(r.norm_values("body").unwrap().is_none());
    assert!(r.sorted_doc_values("s").unwrap().is_some());
    assert!(r.binary_doc_values("b").unwrap().is_none());
    assert!(r.sorted_set_doc_values("ss").unwrap().is_none());
    assert!(r.sorted_numeric_doc_values("sn").unwrap().is_none());
    // Java's filter does not override getNumericDocValues: a numeric field
    // stays readable whether kept or not.
    assert!(r.numeric_doc_values("n").unwrap().is_some());
    let only_id = FieldFilterLeafReader::new(
        Arc::clone(&inner),
        ["id".to_string()].into_iter().collect(),
        false,
    );
    assert!(only_id.numeric_doc_values("n").unwrap().is_some());
    assert_eq!(stored(&r, 2), vec![("id".to_string(), "2".to_string())]);
    // body is the only field with term vectors.
    assert_eq!(vector_fields(&r, 2), None);
    assert!(r.core_cache_helper().is_none());
    assert!(r.reader_cache_helper().is_none());
    assert_eq!(r.to_string(), "FieldFilterLeafReader(fields=[id, n, s])");

    let negated = FieldFilterLeafReader::new(Arc::clone(&inner), keep, true);
    assert_eq!(negated.field_infos().fields.len(), FIELDS.len() - 3);
    assert!(negated.terms("body").unwrap().is_some());
    assert!(negated.terms("id").unwrap().is_none());
    assert!(negated.norm_values("body").unwrap().is_some());
    assert!(negated.binary_doc_values("b").unwrap().is_some());
    assert_eq!(
        stored(&negated, 2),
        vec![("body".into(), "word2 common".into())]
    );
    assert_eq!(vector_fields(&negated, 2), Some(vec!["body".to_string()]));
    assert!(negated.to_string().contains("fields=![id, n, s]"));
}

/// Every field keeps its name and gets a new number; stored fields and term
/// vectors arrive numbered as the shuffled infos say.
fn assert_mismatched(inner: &dyn LeafReader, r: &dyn LeafReader) {
    let infos = r.field_infos();
    let mut numbers: Vec<i32> = infos.fields.iter().map(|f| f.number).collect();
    numbers.sort_unstable();
    assert_eq!(numbers, (0..FIELDS.len() as i32).collect::<Vec<_>>());
    for f in &inner.field_infos().fields {
        let s = infos.field_by_name(&f.name).unwrap();
        assert_eq!(
            (s.index_options, s.doc_values_type, s.store_term_vectors),
            (f.index_options, f.doc_values_type, f.store_term_vectors)
        );
    }
    for doc in 0..r.max_doc() {
        assert_eq!(stored(r, doc), stored(inner, doc));
        assert_eq!(vector_fields(r, doc), vector_fields(inner, doc));
    }
}

#[test]
fn mismatched_readers_renumber_the_fields() {
    let dir = ByteBuffersDirectory::new();
    let (_r, inner) = leaf(&dir);
    let mut renumbered = 0;
    for seed in 0..20 {
        let r = MismatchedLeafReader::new(Arc::clone(&inner), &mut JavaRandom::new(seed));
        assert_mismatched(inner.as_ref(), &r);
        if r.field_infos()
            .fields
            .iter()
            .map(|f| f.name.as_str())
            .ne(inner.field_infos().fields.iter().map(|f| f.name.as_str()))
        {
            renumbered += 1;
        }
        // The same seed shuffles the same way.
        let again = shuffle_infos(inner.field_infos(), &mut JavaRandom::new(seed));
        assert_eq!(layout(&again), layout(r.field_infos()));
        // Cache helpers are the inner reader's.
        assert_eq!(
            r.core_cache_helper().map(|h| h.key_id()),
            inner.core_cache_helper().map(|h| h.key_id())
        );
    }
    assert!(renumbered > 15, "{renumbered}");

    let codec: Arc<dyn CodecReader> =
        Arc::new(DirectoryReader::open(&dir).unwrap().segment_readers()[0].clone());
    let r = MismatchedCodecReader::new(codec, &mut JavaRandom::new(3));
    assert_mismatched(inner.as_ref(), &r);
}

#[test]
fn mismatched_directory_reader_wraps_every_leaf() {
    let dir = ByteBuffersDirectory::new();
    index(&dir, 3);
    let reader = Arc::new(DirectoryReader::open(&dir).unwrap());
    let r = mismatched_directory_reader(Arc::clone(&reader), JavaRandom::new(9)).unwrap();
    assert_eq!(r.wrapped_leaves().len(), 3);
    assert_eq!(r.num_docs(), 15);
    // One stream for all the leaves: Java's shared Random.
    let mut random = JavaRandom::new(9);
    for (wrapped, seg) in r.wrapped_leaves().iter().zip(reader.segment_readers()) {
        assert_eq!(
            layout(wrapped.field_infos()),
            layout(&shuffle_infos(seg.field_infos(), &mut random))
        );
        assert_mismatched(seg, wrapped.as_ref());
    }
}

#[test]
fn own_cache_key_multi_reader_has_its_own_key() {
    let a = ByteBuffersDirectory::new();
    let b = ByteBuffersDirectory::new();
    index(&a, 1);
    index(&b, 2);
    let ra = Arc::new(DirectoryReader::open(&a).unwrap());
    let rb = Arc::new(DirectoryReader::open(&b).unwrap());
    let r = OwnCacheKeyMultiReader::new(vec![
        ReaderHandle::Composite(ra.clone()),
        ReaderHandle::Composite(rb),
    ])
    .unwrap();
    assert_eq!(r.max_doc(), 15);
    assert_eq!(r.num_docs(), 15);
    assert_eq!(r.leaves().len(), 3);
    assert_eq!(r.leaf_handles().len(), 3);
    assert_eq!(r.sequential_sub_readers().len(), 2);
    assert!(r.multi_reader().reader_cache_helper().is_none());
    let helper = r.reader_cache_helper().unwrap();
    assert_ne!(
        Some(helper.key_id()),
        IndexReader::reader_cache_helper(ra.as_ref()).map(|h| h.key_id())
    );
    // A single child: MultiReader delegates, this one still has its own.
    let one = OwnCacheKeyMultiReader::new(vec![ReaderHandle::Composite(ra.clone())]).unwrap();
    assert_eq!(
        one.multi_reader().reader_cache_helper().map(|h| h.key_id()),
        IndexReader::reader_cache_helper(ra.as_ref()).map(|h| h.key_id())
    );
    assert_ne!(
        one.reader_cache_helper().map(|h| h.key_id()),
        IndexReader::reader_cache_helper(ra.as_ref()).map(|h| h.key_id())
    );
    // Closed listeners run when the reader goes away.
    let closed = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&closed);
    let id = helper.key_id();
    helper.add_closed_listener(Box::new(move |key| {
        assert_eq!(key, id);
        c.fetch_add(1, Ordering::SeqCst);
    }));
    drop(r);
    assert_eq!(closed.load(Ordering::SeqCst), 1);
    drop(one);
}

/// Every stored value type passes through the filtering and the renumbering
/// visitors unchanged, and a document without term vectors has none through
/// either wrapper.
#[test]
fn wrappers_pass_every_stored_type_and_missing_vectors() {
    use lucene_index::document::StoredValue;
    let dir = ByteBuffersDirectory::new();
    {
        let mut w = IndexWriter::open(
            &dir,
            Vec::new(),
            "Lucene104",
            LuceneVersion {
                major: 10,
                minor: 5,
                bugfix: 0,
            },
        )
        .unwrap();
        let mut d = Document::new();
        d.add(StringField::new("id", "0", Store::Yes));
        d.add(Field::stored("i", StoredValue::Int(-7)));
        d.add(Field::stored("l", StoredValue::Long(1 << 40)));
        d.add(Field::stored("f", StoredValue::Float(1.5)));
        d.add(Field::stored("d", StoredValue::Double(-2.25)));
        d.add(Field::stored("y", StoredValue::Binary(vec![1, 2, 3])));
        w.add_fields_document(&d).unwrap();
        w.close().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let inner: Arc<dyn LeafReader> = Arc::new(reader.segment_readers()[0].clone());
    let values = |r: &dyn LeafReader| {
        let mut v = DocumentVisitor::all();
        r.document(0, &mut v).unwrap();
        let mut out: Vec<(String, String)> = v
            .into_document()
            .fields
            .into_iter()
            .map(|f| {
                let name = r.field_infos().field_by_number(f.field_number).unwrap();
                (name.name.clone(), format!("{:?}", f.value))
            })
            .collect();
        out.sort();
        out
    };
    let want = values(inner.as_ref());
    assert_eq!(want.len(), 6);
    let all: HashSet<String> = ["id", "i", "l", "f", "d", "y"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let filtered = FieldFilterLeafReader::new(Arc::clone(&inner), all, false);
    assert_eq!(values(&filtered), want);
    assert!(filtered.term_vectors(0).unwrap().is_none());
    for seed in 0..4 {
        let m = MismatchedLeafReader::new(Arc::clone(&inner), &mut JavaRandom::new(seed));
        assert_eq!(values(&m), want);
        assert!(m.term_vectors(0).unwrap().is_none());
        assert_eq!(
            m.reader_cache_helper().map(|h| h.key_id()),
            inner.reader_cache_helper().map(|h| h.key_id())
        );
    }
    let codec: Arc<dyn CodecReader> = Arc::new(reader.segment_readers()[0].clone());
    let m = MismatchedCodecReader::new(codec, &mut JavaRandom::new(1));
    assert_eq!(values(&m), want);
    assert!(m.term_vectors(0).unwrap().is_none());
    // AllDeletedFilterReader's term vectors are the inner reader's.
    let gone = AllDeletedFilterReader::new(Arc::clone(&inner));
    assert!(gone.term_vectors(0).unwrap().is_none());
}
