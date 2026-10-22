//! `BaseStoredFieldsFormatTestCase`: the stored fields suite, run on the
//! one stored fields format this port writes (`Lucene90StoredFieldsFormat`,
//! `BEST_SPEED`, through `Lucene104`).
//!
//! Differs:
//!
//! - `maybeWrapWithMergingReader` is not applied: the merging stored fields
//!   reader is the same reader here.
//! - `newField`'s random field types are the plain ones (see
//!   [`super::base_index_file_format`]).
//! - `testWriteReadMerge` cannot switch to `SimpleTextCodec` between
//!   writers (this port writes `Lucene104` only), so it reopens the writer
//!   on the same codec; its `IntPoint.newRangeQuery` deletes are term
//!   deletes over an unstored `StringField` id, which select the same
//!   documents.
//! - `testMergeFilterReader`'s `DummyFilterDirectoryReader` reverses the
//!   document order of stored fields only, which `addIndexes` would copy:
//!   this port's merge reads the source segments' files, not a reader, so
//!   no wrapper can change what it copies. What remains is the rest of the
//!   test: every stored type, deletes, `addIndexesSlowly` and the check.
//! - `testMismatchedFields`' `MismatchedDirectoryReader` shuffles the field
//!   numbers a reader reports, which the merge would not see (as above).
//!   Field numbers are mixed up where the merge does see them: each source
//!   index adds the ten fields in its own random order, so the same name
//!   has a different number in each.
//! - `testBinaryFieldOffsetLength`: a Rust slice has no offset into a
//!   larger array, so the stored value is the slice.
//! - `testIndexedBit` reads the field's index options from the segment's
//!   field infos, which is where Java's stored document gets them.
//! - `testReadSkip`'s `document(docID, Set<String>)` is a
//!   `DocumentVisitor::for_fields` over the field's number.

use super::base_index_file_format::SuiteTest;
use lucene_search::reader::IndexReader as _;
use std::collections::BTreeMap;

use lucene_codecs::stored_fields::DocumentVisitor;
use lucene_index::buffered_updates::{DeleteQuery, Term};
use lucene_index::document::{
    Document, Field, FieldType, IndexOptions, IntPoint, NumericDocValuesField, Store, StoredValue,
    StringField, TextField,
};
use lucene_index::merge_policy::NoMergePolicy;
use lucene_index::segment_info::{IndexSortField, IndexSortKind, NumericSortKey};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::LeafReader;
use lucene_store::{ByteBuffersDirectory, Directory, FsDirectory};

use super::base_index_file_format::{
    add_indexes_slowly, docs_with_term, force_merge, get, get_all, leaf_stored, mock_analyzer,
    new_config, new_riw, only_leaf, plain_writer, random_byte_array, riw_force_merge, riw_with,
    set_index_sort, stored, IndexFileFormatSuite,
};
use super::index_writer_config::{LiveMergePolicy, PolicyKind};
use super::merge_policies::shuffle;
use crate::store::mock_directory_wrapper::{MockDirectoryWrapper, Throttling};
use crate::util::line_file_docs::LineFileDocs;
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::test_util;

/// `BaseStoredFieldsFormatTestCase` itself: its `addRandomFields`.
#[derive(Debug, Default, Clone, Copy)]
pub struct StoredFieldsSuite;

impl IndexFileFormatSuite for StoredFieldsSuite {
    fn add_random_fields(&self, t: &mut LuceneTestCase, d: &mut Document) {
        let num_values = t.random().next_int_bounded(3);
        for _ in 0..num_values {
            let s = test_util::random_simple_string_max(t.random(), 100);
            d.add(Field::stored("f", StoredValue::String(s)));
        }
    }
}

/// `new FieldType()` with `setStored(true)`: stored, not indexed.
pub fn stored_only() -> FieldType {
    let mut ft = FieldType::new();
    ft.set_stored(true).expect("an unfrozen FieldType");
    ft
}

fn text(v: &StoredValue) -> Option<&str> {
    match v {
        StoredValue::String(s) => Some(s),
        _ => None,
    }
}

fn int_of(v: &StoredValue) -> i32 {
    match v {
        StoredValue::Int(i) => *i,
        other => panic!("not an int: {other:?}"),
    }
}

/// `testRandomStoredFields`.
pub fn test_random_stored_fields(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let mut config = new_config(t, analyzer);
    config.max_buffered_docs = Some(test_util::next_int(t.random(), 5, 20));
    let mut w = riw_with(t, &dir, config);
    let doc_count = t.at_least(200);
    let field_count = test_util::next_int(t.random(), 1, 5);
    let mut field_ids: Vec<i32> = (0..field_count).collect();
    let mut id_type = FieldType::copy_of(&TextField::type_stored());
    id_type.set_tokenized(false).expect("an unfrozen FieldType");
    let custom_type2 = stored_only();
    // id -> (field name, value)
    let mut docs: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for i in 0..doc_count {
        let id = i.to_string();
        let mut doc = Document::new();
        doc.add(Field::from_string("id", id.clone(), id_type.clone()).expect("id"));
        let mut expected = Vec::new();
        for &field in &field_ids {
            if t.random().next_int_bounded(4) != 3 {
                let s = test_util::random_unicode_string_max(t.random(), 1000);
                let name = format!("f{field}");
                doc.add(
                    Field::from_string(name.clone(), s.clone(), custom_type2.clone()).expect("f"),
                );
                expected.push((name, s));
            }
        }
        w.add_document(&doc).expect("addDocument");
        docs.insert(id, expected);
        if t.random().next_int_bounded(50) == 17 {
            // mixup binding of field name -> Number every so often
            shuffle(t.random(), &mut field_ids);
        }
        if t.random().next_int_bounded(5) == 3 && i > 0 {
            let del_id = t.random().next_int_bounded(i).to_string();
            w.delete_documents(Term::new("id", del_id.clone()))
                .expect("deleteDocuments");
            docs.remove(&del_id);
        }
    }
    if !docs.is_empty() {
        let ids: Vec<&String> = docs.keys().collect();
        for _ in 0..2 {
            let r = w.get_reader().expect("getReader");
            let num = t.at_least(100);
            for _ in 0..num {
                let test_id = ids[t.random().next_int_bounded(ids.len() as i32) as usize];
                let hits = docs_with_term(&r, "id", test_id.as_bytes());
                assert_eq!(hits.len(), 1, "id {test_id}");
                let doc = stored(&r, hits[0]);
                let expected = &docs[test_id];
                for i in 0..field_count {
                    let name = format!("f{i}");
                    let want = expected
                        .iter()
                        .find(|(n, _)| *n == name)
                        .map(|(_, v)| v.as_str());
                    assert_eq!(
                        want,
                        get(&doc, &name).and_then(text),
                        "doc {test_id}, field f{i} is wrong"
                    );
                }
            }
            riw_force_merge(&mut w, 1);
        }
    }
    w.close().expect("close");
}

/// `testStoredFieldsOrder` (LUCENE-1727): fields come back in the order
/// they were added, not by name or number.
pub fn test_stored_fields_order(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let mut w = super::base_index_file_format::writer_with(&dir, &config);
    let ft = stored_only();
    let mut doc = Document::new();
    for (name, value) in [("zzz", "a b c"), ("aaa", "a b c"), ("zzz", "1 2 3")] {
        doc.add(Field::from_string(name, value, ft.clone()).expect("field"));
    }
    w.add_fields_document(&doc).expect("addDocument");
    w.commit().expect("commit");
    let r = DirectoryReader::open(&dir).expect("open");
    let fields = stored(&r, 0);
    let got: Vec<(&str, Option<&str>)> =
        fields.iter().map(|(n, v)| (n.as_str(), text(v))).collect();
    assert_eq!(
        got,
        [
            ("zzz", Some("a b c")),
            ("aaa", Some("a b c")),
            ("zzz", Some("1 2 3"))
        ]
    );
    w.close().expect("close");
}

/// `testBinaryFieldOffsetLength` (LUCENE-1219).
pub fn test_binary_field_offset_length(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let mut w = super::base_index_file_format::writer_with(&dir, &config);
    let b: Vec<u8> = (0..50u8).map(|i| i + 77).collect();
    let mut doc = Document::new();
    doc.add(Field::stored(
        "binary",
        StoredValue::Binary(b[10..27].to_vec()),
    ));
    w.add_fields_document(&doc).expect("addDocument");
    w.close().expect("close");
    let r = DirectoryReader::open(&dir).expect("open");
    let fields = stored(&r, 0);
    let Some(StoredValue::Binary(b2)) = get(&fields, "binary") else {
        panic!("no binary value: {fields:?}");
    };
    assert_eq!(b2.len(), 17);
    assert_eq!(b2[0], 87);
}

/// `testNumericField`: each stored numeric comes back with its value and
/// its type.
pub fn test_numeric_field(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let mut w = new_riw(t, &dir);
    let num_docs = t.at_least(500);
    let mut answers = Vec::with_capacity(num_docs as usize);
    for id in 0..num_docs {
        let r = t.random();
        let answer = if r.next_boolean() {
            if r.next_boolean() {
                StoredValue::Float(r.next_float())
            } else {
                StoredValue::Double(r.next_double())
            }
        } else if r.next_boolean() {
            StoredValue::Int(r.next_int())
        } else {
            StoredValue::Long(r.next_long())
        };
        let mut doc = Document::new();
        doc.add(Field::stored("nf", answer.clone()));
        answers.push(answer);
        doc.add(Field::stored("id", StoredValue::Int(id)));
        doc.add(IntPoint::new("id", &[id]).expect("IntPoint"));
        doc.add(NumericDocValuesField::new("id", i64::from(id)));
        w.add_document(&doc).expect("addDocument");
    }
    let r = w.get_reader().expect("getReader");
    w.close().expect("close");
    assert_eq!(r.num_docs(), num_docs);
    for sub in r.segment_readers() {
        let mut ids = sub
            .numeric_doc_values("id")
            .expect("doc values")
            .expect("an id doc value");
        for doc_id in 0..sub.num_docs() {
            let fields = leaf_stored(sub, doc_id);
            let f = get(&fields, "nf").expect("nf");
            assert_eq!(ids.next_doc().expect("nextDoc"), doc_id);
            // `assertEquals(Number, Number)` and its class: the enum
            // variant is the class.
            assert_eq!(f, &answers[ids.long_value() as usize]);
        }
    }
}

/// `testIndexedBit`: a stored-only field is not indexed; a `StringField`
/// is.
pub fn test_indexed_bit(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let mut w = new_riw(t, &dir);
    let mut doc = Document::new();
    doc.add(Field::from_string("field", "value", stored_only()).expect("field"));
    doc.add(StringField::new("field2", "value", Store::Yes));
    w.add_document(&doc).expect("addDocument");
    let r = w.get_reader().expect("getReader");
    w.close().expect("close");
    let leaf = &r.segment_readers()[0];
    let fields = leaf_stored(leaf, 0);
    assert!(get(&fields, "field").is_some() && get(&fields, "field2").is_some());
    let infos = leaf.field_infos();
    assert_eq!(
        infos.field_by_name("field").expect("field").index_options,
        IndexOptions::None
    );
    assert_ne!(
        infos.field_by_name("field2").expect("field2").index_options,
        IndexOptions::None
    );
}

/// `testReadSkip`: each field read alone, skipping the others' bytes.
pub fn test_read_skip(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let mut config = new_config(t, analyzer);
    config.max_buffered_docs = Some(test_util::next_int(t.random(), 2, 30));
    let mut iw = riw_with(t, &dir, config);
    let r = t.random();
    let string = test_util::random_simple_string_max(r, 50);
    let bytes = string.as_bytes().to_vec();
    let l = if r.next_boolean() {
        i64::from(r.next_int_bounded(42))
    } else {
        r.next_long()
    };
    let i = if r.next_boolean() {
        r.next_int_bounded(42)
    } else {
        r.next_int()
    };
    let f = r.next_float();
    let d = r.next_double();
    let ft = stored_only();
    let fields = [
        ("bytes", StoredValue::Binary(bytes.clone())),
        ("string", StoredValue::String(string.clone())),
        ("long", StoredValue::Long(l)),
        ("int", StoredValue::Int(i)),
        ("float", StoredValue::Float(f)),
        ("double", StoredValue::Double(d)),
    ];
    for _ in 0..100 {
        let mut doc = Document::new();
        doc.add(Field::from_bytes("bytes", bytes.clone(), ft.clone()).expect("bytes"));
        doc.add(Field::from_string("string", string.clone(), ft.clone()).expect("string"));
        for (name, value) in &fields[2..] {
            doc.add(Field::stored(*name, value.clone()));
        }
        iw.writer().add_fields_document(&doc).expect("addDocument");
    }
    iw.commit().expect("commit");
    let reader = DirectoryReader::open(&dir).expect("open");
    let doc_id = t.random().next_int_bounded(100);
    let (leaf, local) = super::base_index_file_format::leaf_of(&reader, doc_id);
    for (name, value) in &fields {
        let number = leaf
            .field_infos()
            .field_by_name(name)
            .expect("field")
            .number;
        let mut visitor = DocumentVisitor::for_fields(&[number]);
        leaf.document(local, &mut visitor).expect("document");
        let doc = visitor.into_document();
        assert_eq!(doc.fields.len(), 1, "{name}");
        assert_eq!(&doc.fields[0].value, value, "{name}");
    }
    iw.close().expect("close");
}

/// `testEmptyDocs`: documents without stored fields read back empty.
pub fn test_empty_docs(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let mut config = new_config(t, analyzer);
    config.max_buffered_docs = Some(test_util::next_int(t.random(), 2, 30));
    let mut iw = riw_with(t, &dir, config);
    let num_docs = if t.random().next_boolean() {
        1
    } else {
        t.at_least(1000)
    };
    for _ in 0..num_docs {
        iw.add_document(&Document::new()).expect("addDocument");
    }
    iw.commit().expect("commit");
    let rd = DirectoryReader::open(&dir).expect("open");
    for i in 0..num_docs {
        let (leaf, local) = super::base_index_file_format::leaf_of(&rd, i);
        let doc = leaf.stored_document(local).expect("document");
        // Java's `assertNotNull(doc)`: the segment has stored fields files.
        let doc = doc.expect("a stored document");
        assert!(doc.fields.is_empty());
    }
    iw.close().expect("close");
}

/// `testConcurrentReads`: threads looking up and reading documents through
/// one reader.
pub fn test_concurrent_reads(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let mut config = new_config(t, analyzer);
    config.max_buffered_docs = Some(test_util::next_int(t.random(), 2, 30));
    let mut iw = riw_with(t, &dir, config);
    let num_docs = t.at_least(1000);
    for i in 0..num_docs {
        let mut doc = Document::new();
        doc.add(StringField::new("fld", i.to_string(), Store::Yes));
        iw.add_document(&doc).expect("addDocument");
    }
    iw.commit().expect("commit");
    let rd = DirectoryReader::open(&dir).expect("open");
    let concurrent_reads = t.at_least(5);
    let reads_per_thread = t.at_least(50);
    let queries: Vec<Vec<i32>> = (0..concurrent_reads)
        .map(|_| {
            (0..reads_per_thread)
                .map(|_| t.random().next_int_bounded(num_docs))
                .collect()
        })
        .collect();
    let rd = &rd;
    std::thread::scope(|s| {
        let threads: Vec<_> = queries
            .iter()
            .map(|qs| {
                s.spawn(move || -> Result<(), String> {
                    for &q in qs {
                        let hits = docs_with_term(rd, "fld", q.to_string().as_bytes());
                        if hits.len() != 1 {
                            return Err(format!("Expected 1 hit, got {}", hits.len()));
                        }
                        let doc = stored(rd, hits[0]);
                        match get(&doc, "fld").and_then(text) {
                            None => return Err(format!("Could not find document {q}")),
                            Some(v) if v != q.to_string() => {
                                return Err(format!("Expected {q}, but got {v}"))
                            }
                            Some(_) => {}
                        }
                    }
                    Ok(())
                })
            })
            .collect();
        for th in threads {
            th.join().expect("a reader thread").expect("a read");
        }
    });
    iw.close().expect("close");
}

/// `testWriteReadMerge`: random binary fields through writers reopened
/// now and then, deletes, a merge down to two segments.
pub fn test_write_read_merge(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let mut config = new_config(t, analyzer);
    config.max_buffered_docs = Some(test_util::next_int(t.random(), 2, 30));
    let mut iw = riw_with(t, &dir, config);
    let doc_count = t.at_least(200);
    let mut data: Vec<Vec<Vec<u8>>> = Vec::with_capacity(doc_count as usize);
    for _ in 0..doc_count {
        let field_count = if t.rarely() {
            test_util::next_int(t.random(), 1, 500)
        } else {
            test_util::next_int(t.random(), 1, 5)
        };
        let mut fields = Vec::with_capacity(field_count as usize);
        for _ in 0..field_count {
            let length = if t.rarely() {
                t.random().next_int_bounded(1000)
            } else {
                t.random().next_int_bounded(10)
            };
            let max = if t.rarely() { 256 } else { 2 };
            fields.push(random_byte_array(t.random(), length as usize, max));
        }
        data.push(fields);
    }
    let ft = stored_only();
    let tenth = (doc_count / 10).max(1);
    for (i, fields) in data.iter().enumerate() {
        let i = i as i32;
        let mut doc = Document::new();
        doc.add(IntPoint::new("id", &[i]).expect("IntPoint"));
        doc.add(StringField::new("idt", i.to_string(), Store::No));
        doc.add(Field::stored("id", StoredValue::Int(i)));
        for (j, bytes) in fields.iter().enumerate() {
            doc.add(
                Field::from_bytes(format!("bytes{j}"), bytes.clone(), ft.clone()).expect("bytes"),
            );
        }
        iw.writer().add_fields_document(&doc).expect("addDocument");
        if t.random().next_boolean() && i % tenth == 0 {
            iw.close().expect("close");
            let analyzer = mock_analyzer(t);
            let config = new_config(t, analyzer);
            iw = riw_with(t, &dir, config);
        }
    }
    for _ in 0..10 {
        let min = t.random().next_int_bounded(doc_count);
        let max = min + t.random().next_int_bounded(20);
        let terms = (min..max)
            .map(|id| DeleteQuery::Term(Term::new("idt", id.to_string())))
            .collect();
        iw.delete_documents_by_query(DeleteQuery::Any(terms))
            .expect("deleteDocuments");
    }
    // force merges with deletions
    riw_force_merge(&mut iw, 2);
    iw.commit().expect("commit");
    let ir = DirectoryReader::open(&dir).expect("open");
    assert!(ir.num_docs() > 0);
    let mut num_docs = 0;
    for i in 0..ir.max_doc() {
        let doc = stored(&ir, i);
        if doc.is_empty() {
            continue;
        }
        num_docs += 1;
        let doc_id = int_of(get(&doc, "id").expect("id")) as usize;
        assert_eq!(data[doc_id].len() + 1, doc.len());
        for (j, arr) in data[doc_id].iter().enumerate() {
            assert_eq!(
                get(&doc, &format!("bytes{j}")),
                Some(&StoredValue::Binary(arr.clone()))
            );
        }
    }
    assert!(ir.num_docs() <= num_docs);
    iw.delete_all().expect("deleteAll");
    iw.commit().expect("commit");
    riw_force_merge(&mut iw, 1);
    iw.close().expect("close");
}

/// `testMergeFilterReader`, without the reversing filter (see the module
/// documentation): every stored type through deletes and
/// `addIndexesSlowly`.
pub fn test_merge_filter_reader(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let mut w = new_riw(t, &dir);
    let num_docs = t.at_least(200);
    let string_values: Vec<String> = (0..10)
        .map(|_| test_util::random_realistic_unicode_string_between(t.random(), 10, 10))
        .collect();
    let mut docs = Vec::with_capacity(num_docs as usize);
    for i in 0..num_docs {
        let r = t.random();
        let to_delete = if r.next_boolean() { "yes" } else { "no" };
        let s = string_values[r.next_int_bounded(10) as usize].clone();
        let b = string_values[r.next_int_bounded(10) as usize]
            .clone()
            .into_bytes();
        let values = [
            ("id", StoredValue::Int(i)),
            ("i", StoredValue::Int(r.next_int_bounded(50))),
            ("l", StoredValue::Long(r.next_long())),
            ("d", StoredValue::Double(r.next_double())),
            ("f", StoredValue::Float(r.next_float())),
            ("s", StoredValue::String(s)),
            ("b", StoredValue::Binary(b.clone())),
            ("b2", StoredValue::Binary(b)),
        ];
        let mut doc = Document::new();
        doc.add(StringField::new("to_delete", to_delete, Store::No));
        for (name, v) in &values {
            doc.add(Field::stored(*name, v.clone()));
        }
        w.add_document(&doc).expect("addDocument");
        docs.push(values);
    }
    if t.random().next_boolean() {
        w.delete_documents(Term::new("to_delete", "yes"))
            .expect("deleteDocuments");
    }
    w.commit().expect("commit");
    w.close().expect("close");

    let dir2 = ByteBuffersDirectory::new();
    let mut w = new_riw(t, &dir2);
    add_indexes_slowly(&mut w.writer(), &[&dir]);
    let reader = w.get_reader().expect("getReader");
    for i in 0..reader.max_doc() {
        let doc = stored(&reader, i);
        let id = int_of(get(&doc, "id").expect("id")) as usize;
        for (name, v) in &docs[id] {
            assert_eq!(get(&doc, name), Some(v), "{name}");
        }
    }
    w.close().expect("close");
    test_util::check_index(&dir2).expect("CheckIndex");
}

/// `testBigDocuments` (`@Nightly`): documents much bigger than a chunk --
/// half a million small fields, or one field of megabytes -- through a
/// merge.
pub fn test_big_documents(t: &mut LuceneTestCase) {
    if t.nightly() {
        big_documents(t, (500_000, 1_000_000), (1_000_000, 5_000_000));
    }
}

/// [`test_big_documents`] with its sizes as parameters: `num_fields` small
/// fields in the many-fields document, a field of `big_len` bytes in the
/// other (inclusive ranges).
pub fn big_documents(t: &mut LuceneTestCase, num_fields: (i32, i32), big_len: (i32, i32)) {
    let tmp = lucene_util::test_support::TempDir::new("testBigDocuments");
    let dir = MockDirectoryWrapper::new(t.random(), FsDirectory::open(tmp.path()));
    let analyzer = mock_analyzer(t);
    let mut config = new_config(t, analyzer);
    config.max_buffered_docs = Some(test_util::next_int(t.random(), 2, 30));
    let mut iw = riw_with(t, &dir, config);
    dir.set_throttling(Throttling::Never);
    let ft = stored_only();
    let small_len = t.random().next_int_bounded(10) as usize;
    let small = random_byte_array(t.random(), small_len, 256);
    let num_fields = test_util::next_int(t.random(), num_fields.0, num_fields.1);
    let big_len = test_util::next_int(t.random(), big_len.0, big_len.1) as usize;
    let big = random_byte_array(t.random(), big_len, 2);
    // 0: empty, 1: many small fields, 2: one big field.
    let num_docs = t.at_least(5);
    let kinds: Vec<i32> = (0..num_docs)
        .map(|_| t.random().next_int_bounded(3))
        .collect();
    for (i, &kind) in kinds.iter().enumerate() {
        let mut doc = Document::new();
        doc.add(StringField::new("id", i.to_string(), Store::No));
        match kind {
            1 => {
                for _ in 0..num_fields {
                    doc.add(Field::from_bytes("fld", small.clone(), ft.clone()).expect("fld"));
                }
            }
            2 => doc.add(Field::from_bytes("fld", big.clone(), ft.clone()).expect("fld")),
            _ => {}
        }
        iw.add_document(&doc).expect("addDocument");
        if t.random().next_int_bounded(num_docs) == 0 {
            iw.commit().expect("commit");
        }
    }
    iw.commit().expect("commit");
    // look at what happens when big docs are merged
    riw_force_merge(&mut iw, 1);
    let rd = DirectoryReader::open(&dir).expect("open");
    for (i, &kind) in kinds.iter().enumerate() {
        let hits = docs_with_term(&rd, "id", i.to_string().as_bytes());
        assert_eq!(hits.len(), 1, "{i}");
        let doc = stored(&rd, hits[0]);
        let values = get_all(&doc, "fld");
        let (count, first) = match kind {
            1 => (num_fields as usize, &small),
            2 => (1, &big),
            _ => (0, &small),
        };
        assert_eq!(values.len(), count);
        if count > 0 {
            assert_eq!(values[0], &StoredValue::Binary(first.clone()));
        }
    }
    drop(rd);
    iw.close().expect("close");
}

/// `testBulkMergeWithDeletes`.
pub fn test_bulk_merge_with_deletes(t: &mut LuceneTestCase) {
    let num_docs = t.at_least(200);
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let mut config = new_config(t, analyzer);
    config.merge_policy = LiveMergePolicy::new(PolicyKind::NoMerge(NoMergePolicy::default()));
    let mut w = riw_with(t, &dir, config);
    for i in 0..num_docs {
        let mut doc = Document::new();
        doc.add(StringField::new("id", i.to_string(), Store::Yes));
        let f = test_util::random_simple_string(t.random());
        doc.add(Field::stored("f", StoredValue::String(f)));
        w.add_document(&doc).expect("addDocument");
    }
    let delete_count = test_util::next_int(t.random(), 5, num_docs);
    for _ in 0..delete_count {
        let id = t.random().next_int_bounded(num_docs);
        w.delete_documents(Term::new("id", id.to_string()))
            .expect("deleteDocuments");
    }
    w.commit().expect("commit");
    w.close().expect("close");
    let mut w = new_riw(t, &dir);
    let n = test_util::next_int(t.random(), 1, 3) as usize;
    riw_force_merge(&mut w, n);
    w.commit().expect("commit");
    w.close().expect("close");
    test_util::check_index(&dir).expect("CheckIndex");
}

/// `testMismatchedFields`: ten indexes numbering the same ten fields
/// differently (see the module documentation), added together and merged:
/// every value must still sit under its own name.
pub fn test_mismatched_fields(t: &mut LuceneTestCase) {
    let dirs: Vec<ByteBuffersDirectory> = (0..10).map(|_| ByteBuffersDirectory::new()).collect();
    for target in &dirs {
        let dir = ByteBuffersDirectory::new();
        let mut iw = plain_writer(&dir);
        let mut order: Vec<i32> = (0..10).collect();
        if t.random().next_boolean() {
            shuffle(t.random(), &mut order);
        }
        let mut doc = Document::new();
        for j in order {
            // name=value (e.g. 3=3), so a mix-up shows.
            doc.add(StringField::new(j.to_string(), j.to_string(), Store::Yes));
        }
        for _ in 0..10 {
            iw.add_fields_document(&doc).expect("addDocument");
        }
        iw.commit().expect("commit");
        let mut adder = plain_writer(target);
        add_indexes_slowly(&mut adder, &[&dir]);
        adder.commit().expect("commit");
        adder.close().expect("close");
        iw.close().expect("close");
    }
    let everything = ByteBuffersDirectory::new();
    let mut iw = plain_writer(&everything);
    let sources: Vec<&dyn Directory> = dirs.iter().map(|d| d as &dyn Directory).collect();
    iw.add_indexes(&sources).expect("addIndexes");
    force_merge(&mut iw, 1);
    iw.commit().expect("commit");
    let r = DirectoryReader::open(&everything).expect("open");
    let ir = only_leaf(&r);
    assert_eq!(ir.max_doc(), 100);
    for i in 0..ir.max_doc() {
        let doc = leaf_stored(ir, i);
        assert_eq!(doc.len(), 10);
        for j in 0..10 {
            let name = j.to_string();
            assert_eq!(get(&doc, &name).and_then(text), Some(name.as_str()));
        }
    }
    iw.close().expect("close");
}

/// One document of `testRandomStoredFieldsWithIndexSort`: its id, whether
/// that id is stored, its stored text fields and its sort values.
struct SortDoc {
    id: String,
    id_stored: bool,
    stored: Vec<(String, String)>,
    sort_values: Vec<(String, i64)>,
}

impl SortDoc {
    fn document(&self) -> Document {
        let mut doc = Document::new();
        let store = if self.id_stored {
            Store::Yes
        } else {
            Store::No
        };
        doc.add(StringField::new("id", self.id.clone(), store));
        let ty = TextField::type_stored();
        for (name, s) in &self.stored {
            doc.add(Field::from_string(name.clone(), s.clone(), ty.clone()).expect("text"));
        }
        for (name, v) in &self.sort_values {
            doc.add(NumericDocValuesField::new(name.clone(), *v));
        }
        doc
    }

    fn expected_stored(&self) -> Vec<(String, String)> {
        let mut e = Vec::new();
        if self.id_stored {
            e.push(("id".to_string(), self.id.clone()));
        }
        e.extend(self.stored.iter().cloned());
        e
    }
}

/// `testRandomStoredFieldsWithIndexSort`: documents added, through foreign
/// readers now and then, updated and deleted in a sorted index; each
/// document's stored fields follow it through every sort and merge.
pub fn test_random_stored_fields_with_index_sort(t: &mut LuceneTestCase) {
    let sort_fields: Vec<(String, NumericSortKey)> = if t.random().next_boolean() {
        vec![
            ("sort-1".into(), NumericSortKey::Long(None)),
            ("sort-2".into(), NumericSortKey::Int(None)),
        ]
    } else {
        vec![("sort-1".into(), NumericSortKey::Long(None))]
    };
    let index_sort: Vec<IndexSortField> = sort_fields
        .iter()
        .map(|(field, key)| IndexSortField {
            field: field.clone(),
            reverse: false,
            kind: IndexSortKind::Numeric(*key),
        })
        .collect();
    let num_fields = test_util::next_int(t.random(), 1, 10);
    let mut stored_fields: Vec<String> = (0..num_fields).map(|i| format!("f-{i}")).collect();
    let mut document_factory = |t: &mut LuceneTestCase, id: &str| {
        let id_stored = t.random().next_boolean();
        if t.random().next_int_bounded(100) <= 5 {
            shuffle(t.random(), &mut stored_fields);
        }
        let mut stored = Vec::new();
        for name in &stored_fields {
            if t.random().next_boolean() {
                stored.push((
                    name.clone(),
                    test_util::random_unicode_string_max(t.random(), 100),
                ));
            }
        }
        let sort_values = sort_fields
            .iter()
            .map(|(f, _)| {
                (
                    f.clone(),
                    i64::from(test_util::next_int(t.random(), 0, 10000)),
                )
            })
            .collect();
        SortDoc {
            id: id.to_string(),
            id_stored,
            stored,
            sort_values,
        }
    };
    let num_docs = t.at_least(100);
    let mut docs: BTreeMap<String, SortDoc> = BTreeMap::new();
    for i in 0..num_docs {
        let id = i.to_string();
        let d = document_factory(t, &id);
        docs.insert(id, d);
    }
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let mut iwc = new_config(t, analyzer);
    iwc.max_buffered_docs = Some(test_util::next_int(t.random(), 5, 20));
    let mut iw = riw_with(t, &dir, iwc);
    set_index_sort(&mut iw.writer(), &index_sort);
    let mut added_ids: Vec<String> = Vec::new();
    let verify = |t: &mut LuceneTestCase,
                  iw: &mut super::random_index_writer::RandomIndexWriter<'_>,
                  added_ids: &[String],
                  docs: &BTreeMap<String, SortDoc>| {
        if added_ids.is_empty() {
            return;
        }
        let reader = iw.get_reader().expect("getReader");
        let iters = test_util::next_int(t.random(), 1, 10);
        for _ in 0..iters {
            let test_id = &added_ids[t.random().next_int_bounded(added_ids.len() as i32) as usize];
            let hits = docs_with_term(&reader, "id", test_id.as_bytes());
            assert_eq!(hits.len(), 1, "id {test_id}");
            let expected = docs[test_id].expected_stored();
            let actual = stored(&reader, hits[0]);
            assert_eq!(expected.len(), actual.len());
            for (name, value) in &expected {
                let values = get_all(&actual, name);
                assert_eq!(values.len(), 1, "{name}");
                assert_eq!(text(values[0]), Some(value.as_str()), "{name}");
            }
        }
    };
    let mut ids: Vec<String> = docs.keys().cloned().collect();
    shuffle(t.random(), &mut ids);
    for id in ids {
        if t.random().next_int_bounded(100) < 5 {
            // add via foreign reader
            let other_dir = ByteBuffersDirectory::new();
            let analyzer = mock_analyzer(t);
            let other_iwc = new_config(t, analyzer);
            let mut other = riw_with(t, &other_dir, other_iwc);
            set_index_sort(&mut other.writer(), &index_sort);
            other
                .add_document(&docs[&id].document())
                .expect("addDocument");
            other.commit().expect("commit");
            add_indexes_slowly(&mut iw.writer(), &[&other_dir]);
            other.close().expect("close");
        } else {
            // add normally
            iw.add_document(&docs[&id].document()).expect("addDocument");
        }
        added_ids.push(id);
        if t.random().next_int_bounded(100) < 5 {
            let at = t.random().next_int_bounded(added_ids.len() as i32) as usize;
            let deleting_id = added_ids.remove(at);
            if t.random().next_boolean() {
                iw.delete_documents_by_query(DeleteQuery::Term(Term::new(
                    "id",
                    deleting_id.clone(),
                )))
                .expect("deleteDocuments");
                added_ids.retain(|a| *a != deleting_id);
            } else {
                let new_doc = document_factory(t, &deleting_id);
                iw.update_document(Term::new("id", deleting_id.clone()), &new_doc.document())
                    .expect("updateDocument");
                docs.insert(deleting_id, new_doc);
            }
        }
        if t.random().next_int_bounded(100) < 5 {
            verify(t, &mut iw, &added_ids, &docs);
        }
        if t.random().next_int_bounded(100) < 2 {
            let n = test_util::next_int(t.random(), 1, 3) as usize;
            riw_force_merge(&mut iw, n);
        }
    }
    verify(t, &mut iw, &added_ids, &docs);
    let n = test_util::next_int(t.random(), 1, 3) as usize;
    riw_force_merge(&mut iw, n);
    verify(t, &mut iw, &added_ids, &docs);
    iw.close().expect("close");
}

/// `testLineFileDocs`: realistic text, which compresses better than random
/// bytes, stored only.
pub fn test_line_file_docs(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    {
        let mut docs = LineFileDocs::new(t.random()).expect("LineFileDocs");
        let mut w = plain_writer(&dir);
        let num_docs = t.at_least(10_000);
        for _ in 0..num_docs {
            // Only keep stored fields
            let doc = docs.next_doc().expect("nextDoc").to_document();
            let mut stored_doc = Document::new();
            for field in doc.fields() {
                if field.field_type().stored() {
                    if let Some(s) = field.string_value() {
                        // Disable indexing
                        stored_doc.add(Field::stored(
                            field.name().to_string(),
                            StoredValue::String(s.into_owned()),
                        ));
                    }
                }
            }
            w.add_fields_document(&stored_doc).expect("addDocument");
        }
        force_merge(&mut w, 1);
        w.close().expect("close");
    }
    test_util::check_index(&dir).expect("CheckIndex");
}

/// Every test of the suite, with its name, for a runner to loop over.
pub const TESTS: &[SuiteTest] = &[
    ("testRandomStoredFields", test_random_stored_fields),
    ("testStoredFieldsOrder", test_stored_fields_order),
    (
        "testBinaryFieldOffsetLength",
        test_binary_field_offset_length,
    ),
    ("testNumericField", test_numeric_field),
    ("testIndexedBit", test_indexed_bit),
    ("testReadSkip", test_read_skip),
    ("testEmptyDocs", test_empty_docs),
    ("testConcurrentReads", test_concurrent_reads),
    ("testWriteReadMerge", test_write_read_merge),
    ("testMergeFilterReader", test_merge_filter_reader),
    ("testBigDocuments", test_big_documents),
    ("testBulkMergeWithDeletes", test_bulk_merge_with_deletes),
    ("testMismatchedFields", test_mismatched_fields),
    (
        "testRandomStoredFieldsWithIndexSort",
        test_random_stored_fields_with_index_sort,
    ),
    ("testLineFileDocs", test_line_file_docs),
];
