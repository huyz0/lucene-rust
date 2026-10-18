use super::*;
use crate::facets::taxonomy::{DirectoryTaxonomyWriter, FacetLabel, TaxonomyWriter};
use lucene_index::document::{Document, NumericDocValuesField, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_store::ByteBuffersDirectory;

fn plain_index(docs: &[Option<i64>]) -> ByteBuffersDirectory {
    let d = ByteBuffersDirectory::new();
    let mut iw = IndexWriter::open(
        &d,
        Vec::new(),
        "Lucene104",
        lucene_index::merge_policy::upgrade::LATEST,
    )
    .unwrap();
    for parent in docs {
        let mut doc = Document::new();
        doc.add(StringField::new("f", "x", Store::No));
        if let Some(p) = parent {
            doc.add(NumericDocValuesField::new(FIELD_PARENT_ORDINAL_NDV, *p));
        }
        iw.add_fields_document(&doc).unwrap();
    }
    iw.commit().unwrap();
    drop(iw);
    d
}

fn arrays_of(d: &ByteBuffersDirectory) -> Result<TaxonomyIndexArrays> {
    TaxonomyIndexArrays::new(&DirectoryReader::open(d).unwrap())
}

#[test]
fn oversize_is_java_array_util() {
    assert_eq!(oversize_ints(0), 0);
    assert_eq!(oversize_ints(1), 4);
    assert_eq!(oversize_ints(8), 12);
    assert_eq!(oversize_ints(100), 112);
    assert_eq!(oversize_ints(MAX_ARRAY_LENGTH - 2), MAX_ARRAY_LENGTH);
}

/// Over a chunk boundary: the parents read in two chunks, the children and
/// siblings derived, and a reopen grows them, sharing the full chunks.
#[test]
fn spans_chunks_and_grows() {
    let d = ByteBuffersDirectory::new();
    let w = DirectoryTaxonomyWriter::open(&d).unwrap();
    // root, d, then d/1 .. d/(n-1).
    let n = CHUNK_SIZE + 5;
    w.add_category(&FacetLabel::new(["d"]).unwrap()).unwrap();
    for i in 1..n {
        let ord = w.add_category(&FacetLabel::new(["d".to_string(), i.to_string()]).unwrap());
        assert_eq!(ord.unwrap(), i as i32 + 1);
    }
    w.commit().unwrap();
    let r1 = DirectoryReader::open(&d).unwrap();
    let a1 = TaxonomyIndexArrays::new(&r1).unwrap();
    let len = n as i32 + 1;
    assert_eq!(a1.parents().length(), len);
    assert_eq!(a1.parents().get(0), INVALID_ORDINAL);
    assert_eq!(a1.parents().get(1), 0);
    assert_eq!(a1.parents().get(len - 1), 1);
    assert_eq!(a1.children().get(1), len - 1);
    assert_eq!(a1.siblings().get(len - 1), len - 2);
    assert_eq!(a1.siblings().get(2), INVALID_ORDINAL);
    assert_eq!(a1.siblings().get(0), INVALID_ORDINAL);
    assert_eq!(a1.children().length(), len);

    w.add_category(&FacetLabel::new(["e"]).unwrap()).unwrap();
    w.add_category(&FacetLabel::new(["d", "new"]).unwrap())
        .unwrap();
    w.commit().unwrap();
    let r2 = DirectoryReader::open(&d).unwrap();
    let a2 = TaxonomyIndexArrays::copy_from(&r2, &a1).unwrap();
    assert!(Arc::ptr_eq(&a1.parents.values[0], &a2.parents.values[0]));
    assert_eq!(a2.parents().length(), len + 2);
    assert_eq!(a2.parents().get(len), 0);
    assert_eq!(a2.parents().get(len + 1), 1);
    assert_eq!(a2.children().get(0), len);
    assert_eq!(a2.children().get(1), len + 1);
    assert_eq!(a2.siblings().get(len + 1), len - 1);
    assert_eq!(a2.siblings().get(len), 1);
    // Nothing new: the same values.
    let a3 = TaxonomyIndexArrays::copy_from(&r2, &a2).unwrap();
    assert_eq!(a3.parents().length(), len + 2);
    assert_eq!(a3.children().get(1), len + 1);
    // Copied from arrays without children: computed on demand.
    let a4 = TaxonomyIndexArrays::copy_from(&r2, &TaxonomyIndexArrays::new(&r1).unwrap()).unwrap();
    assert!(a4.children_siblings.get().is_none());
    assert_eq!(a4.children().get(0), len);

    // Read through its chunks, as a walk does: the same values.
    use crate::facets::taxonomy::parallel_taxonomy_arrays::IntArrayReader;
    let siblings = IntArrayReader::new(a2.siblings());
    assert!(matches!(siblings, IntArrayReader::Chunks(_)));
    for i in [0, 1, 2, len - 1, len, len + 1] {
        assert_eq!(siblings.get(i), a2.siblings().get(i), "ordinal {i}");
    }
    let small = TaxonomyIndexArrays::new(
        &DirectoryReader::open(&plain_index(&[Some(-1), Some(0)])).unwrap(),
    )
    .unwrap();
    let one = IntArrayReader::new(small.parents());
    assert!(matches!(one, IntArrayReader::Single(_)));
    assert_eq!((one.get(0), one.get(1)), (INVALID_ORDINAL, 0));

    // A write to a chunk shared with another reader's arrays copies it
    // first: the other arrays keep their value.
    let mut a5 = TaxonomyIndexArrays::copy_from(&r2, &a2).unwrap();
    assert!(Arc::ptr_eq(&a2.parents.values[0], &a5.parents.values[0]));
    a5.add(1, 7);
    assert!(!Arc::ptr_eq(&a2.parents.values[0], &a5.parents.values[0]));
    assert_eq!((a5.parents().get(1), a2.parents().get(1)), (7, 0));
}

/// The writer's `add`: grows by `ArrayUtil.oversize`, forgets children.
#[test]
fn add_grows_parents() {
    let d = plain_index(&[Some(-1)]);
    let mut a = arrays_of(&d).unwrap();
    assert_eq!(a.children().get(0), INVALID_ORDINAL);
    a.add(-3, 0);
    assert_eq!(a.parents().length(), 1);
    a.add(1, 0);
    // `ArrayUtil.oversize(2, Integer.BYTES)`.
    assert_eq!(a.parents().length(), 6);
    assert_eq!(a.parents().get(1), 0);
    assert!(a.children_siblings.get().is_none());
    a.add(2, 1);
    assert_eq!(a.parents().length(), 6);
    assert_eq!(a.children().get(1), 2);
    assert_eq!(a.ram_bytes_used(), 41 + 32);
}

/// Indexes Lucene would not have written: no parent field, a document
/// without a parent, a parent beyond `int`, a parent outside the taxonomy.
#[test]
fn corrupt_parents() {
    let err = arrays_of(&plain_index(&[None])).unwrap_err();
    assert!(
        format!("{err:?}").contains("Parent data field $parent_ndv$ does not exist"),
        "{err:?}"
    );
    let err = arrays_of(&plain_index(&[Some(-1), None])).unwrap_err();
    assert!(
        format!("{err:?}").contains("Missing parent data for category 1"),
        "{err:?}"
    );
    let err = arrays_of(&plain_index(&[Some(-1), Some(i64::MAX)])).unwrap_err();
    assert!(matches!(err, Error::IllegalArgument(ref m) if m == "integer overflow"));
    let a = arrays_of(&plain_index(&[Some(-1), Some(99), Some(0)])).unwrap();
    assert_eq!(a.siblings().get(1), INVALID_ORDINAL);
    assert_eq!(a.children().get(0), 2);
    // An empty index: empty arrays.
    let a = arrays_of(&plain_index(&[])).unwrap();
    assert_eq!(a.parents().length(), 0);
    assert_eq!(a.children().length(), 0);
}
