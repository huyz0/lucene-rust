//! `DocHelper`, `PerThreadPKLookup` and `PointsStackTracker`.

use std::sync::Arc;

use lucene_codecs::stored_fields::{DocumentVisitor, FieldValue};
use lucene_index::buffered_updates::Term;
use lucene_index::document::{Document, IndexableField, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::{LeafReader, PostingsFlags, NO_MORE_DOCS};
use lucene_store::ByteBuffersDirectory;
use lucene_test_framework::index::doc_helper::{self, NameValue};
use lucene_test_framework::index::per_thread_pk_lookup::PerThreadPkLookup;
use lucene_test_framework::index::points_stack_tracker::{Cell, PointsStackTracker};
use lucene_util::java_random::JavaRandom;

fn writer(dir: &ByteBuffersDirectory) -> IndexWriter<'_> {
    IndexWriter::open(
        dir,
        Vec::new(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap()
}

#[test]
fn doc_helper_sets_match_javas() {
    let s = doc_helper::field_sets();
    let set = |names: &[&str]| names.iter().map(|n| n.to_string()).collect();
    assert_eq!(s.all.len(), 14);
    assert_eq!(
        s.unindexed,
        set(&[
            doc_helper::UNINDEXED_FIELD_KEY,
            doc_helper::LAZY_FIELD_BINARY_KEY
        ])
    );
    assert_eq!(s.indexed.len(), 12);
    assert_eq!(
        s.termvector,
        set(&[
            doc_helper::TEXT_FIELD_2_KEY,
            doc_helper::UNSTORED_FIELD_2_KEY,
            doc_helper::TEXT_FIELD_UTF2_KEY
        ])
    );
    assert_eq!(s.notermvector.len(), 9);
    assert_eq!(
        s.unstored,
        set(&[
            doc_helper::UNSTORED_FIELD_1_KEY,
            doc_helper::UNSTORED_FIELD_2_KEY
        ])
    );
    assert_eq!(s.stored.len(), 12);
    // `keyField` is a `StringField`: `DOCS` only, so in Java's `noTf` too.
    assert_eq!(
        s.no_tf,
        set(&[doc_helper::NO_TF_KEY, doc_helper::KEYWORD_FIELD_KEY])
    );
    assert_eq!(
        s.no_norms,
        set(&[
            doc_helper::TEXT_FIELD_3_KEY,
            doc_helper::KEYWORD_FIELD_KEY,
            doc_helper::NO_NORMS_KEY
        ])
    );
    assert!(s.lazy.is_empty());
    let values = doc_helper::name_values();
    assert_eq!(values.len(), 14);
    assert_eq!(
        values[doc_helper::LAZY_FIELD_BINARY_KEY],
        NameValue::Bytes(doc_helper::LAZY_FIELD_BINARY_BYTES.to_vec())
    );
    assert_eq!(doc_helper::LARGE_LAZY_FIELD_TEXT.len(), 55 * 10_000);
    for f in doc_helper::fields() {
        assert!(values.contains_key(f.name()), "{}", f.name());
    }
    // The frozen types refuse a change.
    assert!(doc_helper::text_type_stored_with_tvs()
        .set_stored(false)
        .is_err());
}

#[test]
fn doc_helper_writes_its_document() {
    let dir = ByteBuffersDirectory::new();
    let mut doc = Document::new();
    doc_helper::setup_doc(&mut doc);
    assert_eq!(doc_helper::num_fields(&doc), 14);
    let info = doc_helper::write_doc(&mut JavaRandom::new(42), &dir, &doc).unwrap();
    assert_eq!(info.del_count, 0);

    let reader = DirectoryReader::open(&dir).unwrap();
    let leaf = &reader.segment_readers()[0];
    assert_eq!(leaf.segment_name, info.segment_name);
    // Stored fields come back with Java's values.
    let mut v = DocumentVisitor::all();
    leaf.document(0, &mut v).unwrap();
    let stored = v.into_document();
    let values = doc_helper::name_values();
    let sets = doc_helper::field_sets();
    assert_eq!(stored.fields.len(), sets.stored.len());
    for f in &stored.fields {
        let name = &leaf
            .field_infos()
            .field_by_number(f.field_number)
            .unwrap()
            .name;
        let expected = &values[name.as_str()];
        match (&f.value, expected) {
            (FieldValue::String(s), NameValue::Text(t)) => assert_eq!(s, t),
            (FieldValue::Binary(b), NameValue::Bytes(t)) => assert_eq!(b, t),
            other => panic!("{name}: {other:?}"),
        }
    }
    // FIELD_2_FREQS: field, text, two.
    let terms = leaf.terms(doc_helper::TEXT_FIELD_2_KEY).unwrap().unwrap();
    let mut te = terms.iterator().unwrap();
    let mut freqs = Vec::new();
    while te.next().unwrap().is_some() {
        let mut p = te.postings(PostingsFlags::Freqs).unwrap();
        assert_eq!(p.next_doc().unwrap(), 0);
        freqs.push(p.freq());
    }
    assert_eq!(freqs, doc_helper::FIELD_2_FREQS);
    // No norms where the type omits them; none for the unindexed fields.
    for name in &sets.all {
        let has_norms = leaf.norm_values(name).unwrap().is_some();
        let expected = sets.indexed.contains(name) && !sets.no_norms.contains(name);
        assert_eq!(has_norms, expected, "{name}");
    }
    assert!(leaf.term_vectors(0).unwrap().is_some());
}

#[test]
fn doc_helper_creates_documents() {
    let doc = doc_helper::create_document(7, "idx", 3);
    assert_eq!(doc_helper::num_fields(&doc), 5);
    assert_eq!(doc.get("id").unwrap(), "7");
    assert_eq!(doc.get("indexname").unwrap(), "idx");
    assert_eq!(doc.get("field1").unwrap(), "a7");
    assert_eq!(doc.get("field2").unwrap(), "a7 b7");
    assert_eq!(doc.get("field3").unwrap(), "a7 b7");
    assert!(doc
        .get_field("id")
        .unwrap()
        .field_type()
        .store_term_vector_offsets());
}

#[test]
fn pk_lookup_finds_live_documents_largest_segment_first() {
    let dir = ByteBuffersDirectory::new();
    let mut w = writer(&dir);
    // Segments of 2, 5 and 3 ids, one of the 5 deleted, then one without
    // the id field.
    for (seg, n) in [2, 5, 3].into_iter().enumerate() {
        for i in 0..n {
            let mut d = Document::new();
            d.add(StringField::new("id", format!("{seg}-{i}"), Store::No));
            w.add_fields_document(&d).unwrap();
        }
        w.commit().unwrap();
    }
    w.delete_documents_by_term(&[Term::new("id", "1-0")])
        .unwrap();
    w.commit().unwrap();
    let mut d = Document::new();
    d.add(StringField::new("other", "x", Store::No));
    w.add_fields_document(&d).unwrap();
    w.commit().unwrap();

    let reader = DirectoryReader::open(&dir).unwrap();
    let lookup = PerThreadPkLookup::new(&reader, "id").unwrap();
    // Largest by numDocs first (5 docs, one deleted: 4; then 3; then 2);
    // the "other"-only segment has no id terms and is left out.
    assert_eq!(lookup.doc_bases(), vec![2, 7, 0]);
    assert_eq!(lookup.lookup(b"0-1").unwrap(), Some(1));
    assert_eq!(lookup.lookup(b"1-4").unwrap(), Some(6));
    assert_eq!(lookup.lookup(b"2-2").unwrap(), Some(9));
    assert_eq!(lookup.lookup(b"1-0").unwrap(), None);
    assert_eq!(lookup.lookup(b"nope").unwrap(), None);

    // reopen follows a new reader.
    let mut d = Document::new();
    d.add(StringField::new("id", "3-0", Store::No));
    w.add_fields_document(&d).unwrap();
    w.commit().unwrap();
    let reader2 = DirectoryReader::open(&dir).unwrap();
    let lookup2 = lookup.reopen(Some(&reader2)).unwrap().unwrap();
    assert_eq!(lookup2.lookup(b"3-0").unwrap(), Some(reader2.max_doc() - 1));
    assert!(lookup.reopen(None).unwrap().is_none());
    w.close().unwrap();

    // Every id in a reader without deletions is found; an absent one is not.
    let dir = ByteBuffersDirectory::new();
    let mut w = writer(&dir);
    for i in 0..4 {
        let mut d = Document::new();
        d.add(StringField::new("id", i.to_string(), Store::No));
        w.add_fields_document(&d).unwrap();
    }
    w.close().unwrap();
    let reader = DirectoryReader::open(&dir).unwrap();
    let leaf: Arc<dyn LeafReader> = Arc::new(reader.segment_readers()[0].clone());
    let lookup = PerThreadPkLookup::new(leaf.as_ref(), "id").unwrap();
    for i in 0..4 {
        assert_eq!(lookup.lookup(i.to_string().as_bytes()).unwrap(), Some(i));
    }
    let mut p = leaf
        .postings("id", b"0", PostingsFlags::None)
        .unwrap()
        .unwrap();
    assert_eq!(p.next_doc().unwrap(), 0);
    assert_eq!(p.next_doc().unwrap(), NO_MORE_DOCS);
}

#[test]
fn points_stack_tracker_pops_to_the_containing_cell() {
    let mut t = PointsStackTracker::new(2, 1);
    let cell = |min: [u8; 2], max: [u8; 2]| Cell {
        min_packed_value: min.to_vec(),
        max_packed_value: max.to_vec(),
    };
    t.on_compare(&[0, 0], &[255, 255]);
    t.on_compare(&[0, 0], &[127, 255]);
    t.on_compare(&[0, 0], &[127, 127]);
    assert_eq!(t.stack.len(), 3);
    // A sibling of the last cell: pops it, stays under its parent.
    t.on_compare(&[0, 128], &[127, 255]);
    assert_eq!(
        t.stack,
        vec![
            cell([0, 0], [255, 255]),
            cell([0, 0], [127, 255]),
            cell([0, 128], [127, 255])
        ]
    );
    // The other half of the root: pops back to the root.
    t.on_compare(&[128, 0], &[255, 255]);
    assert_eq!(t.stack.len(), 2);
    // Bytes compare unsigned: 200 is not below 128.
    assert!(t.contains(&cell([128, 0], [255, 255]), &cell([200, 0], [210, 9])));
    assert!(!t.contains(&cell([128, 0], [255, 255]), &cell([100, 0], [210, 9])));
    assert!(!t.contains(&cell([128, 0], [200, 255]), &cell([128, 0], [210, 9])));
    // A cell outside every frame empties the stack before it is pushed.
    let mut t = PointsStackTracker::new(1, 1);
    t.on_compare(&[10], &[20]);
    t.on_compare(&[30], &[40]);
    assert_eq!(
        t.stack,
        vec![Cell {
            min_packed_value: vec![30],
            max_packed_value: vec![40],
        }]
    );
}
