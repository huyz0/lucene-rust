use super::*;
use crate::facets::taxonomy::TaxonomyWriter;
use lucene_index::deletion_policy::OpenMode;
use lucene_index::index_writer::IndexWriter;
use lucene_store::ByteBuffersDirectory;

fn label(c: &[&str]) -> FacetLabel {
    FacetLabel::new(c.iter().copied()).unwrap()
}

fn err_message<T: std::fmt::Debug>(r: Result<T>) -> String {
    match r.unwrap_err() {
        Error::IllegalArgument(m) | Error::IllegalState(m) | Error::AlreadyClosed(m) => m,
        e => format!("{e:?}"),
    }
}

/// root, a(1), a/b(2), c(3) -- committed.
fn small_taxonomy() -> ByteBuffersDirectory {
    let d = ByteBuffersDirectory::new();
    let w = DirectoryTaxonomyWriter::open(&d).unwrap();
    w.add_category(&label(&["a", "b"])).unwrap();
    w.add_category(&label(&["c"])).unwrap();
    w.close().unwrap();
    d
}

/// `TestDirectoryTaxonomyReader`: ordinals, paths and their bulk forms,
/// cached or not.
#[test]
fn ordinals_and_paths() {
    let d = small_taxonomy();
    let r = DirectoryTaxonomyReader::open(&d).unwrap();
    assert_eq!(r.get_size().unwrap(), 4);
    assert_eq!(r.get_ordinal(&FacetLabel::root()).unwrap(), 0);
    assert_eq!(r.get_ordinal(&label(&["a", "b"])).unwrap(), 2);
    // Cached now.
    assert_eq!(r.get_ordinal(&label(&["a", "b"])).unwrap(), 2);
    assert_eq!(r.get_ordinal(&label(&["nope"])).unwrap(), INVALID_ORDINAL);
    assert_eq!(r.get_ordinal_of("a", &["b"]).unwrap(), 2);
    assert_eq!(r.get_bulk_ordinals(&[]).unwrap(), Vec::<i32>::new());
    assert_eq!(r.get_bulk_ordinals(&[label(&["c"])]).unwrap(), [3]);
    let bulk = [
        label(&["c"]),
        label(&["zz"]),
        label(&["a"]),
        label(&["a", "b"]),
        label(&["a"]),
    ];
    assert_eq!(r.get_bulk_ordinals(&bulk).unwrap(), [3, -1, 1, 2, 1]);
    // Every one cached.
    assert_eq!(
        r.get_bulk_ordinals(&[label(&["c"]), label(&["a"])])
            .unwrap(),
        [3, 1]
    );

    assert!(r.get_path(0).unwrap().is_empty());
    assert_eq!(r.get_path(2).unwrap(), label(&["a", "b"]));
    assert_eq!(r.get_path(2).unwrap(), label(&["a", "b"]));
    assert_eq!(
        r.get_bulk_path(&[3, 1, 2, 3]).unwrap(),
        [
            label(&["c"]),
            label(&["a"]),
            label(&["a", "b"]),
            label(&["c"])
        ]
    );
    let m = err_message(r.get_path(4));
    assert!(
        m.starts_with(
            "ordinal 4 is out of the range of the indexReader StandardDirectoryReader(segments_1:"
        ),
        "{m}"
    );
    assert!(
        m.ends_with(" _0:1 _1:3). The maximum possible ordinal number is 3"),
        "{m}"
    );
    assert!(err_message(r.get_bulk_path(&[0, -1])).starts_with("ordinal -1 "));
    assert_eq!(r.internal_index_reader().unwrap().max_doc(), 4);
    assert!(format!("{r:?}").contains("StandardDirectoryReader(segments_1"));
    assert_eq!(
        r.to_string(10).unwrap(),
        "0: EMPTY STRING!! \n1: FacetLabel: [a]\n2: FacetLabel: [a, b]\n3: FacetLabel: [c]\n"
    );
    assert_eq!(r.to_string(1).unwrap(), "0: EMPTY STRING!! \n");
}

/// The parallel arrays and `getChildren`: youngest child first.
#[test]
fn arrays_and_children() {
    let d = small_taxonomy();
    let r = DirectoryTaxonomyReader::open(&d).unwrap();
    let arrays = r.parallel_taxonomy_arrays().unwrap();
    let parents: Vec<i32> = (0..4).map(|i| arrays.parents().get(i)).collect();
    assert_eq!(parents, [-1, 0, 1, 0]);
    assert_eq!(r.get_children(0).unwrap().collect::<Vec<_>>(), [3, 1]);
    assert_eq!(r.get_children(1).unwrap().collect::<Vec<_>>(), [2]);
    assert_eq!(
        r.get_children(2).unwrap().collect::<Vec<_>>(),
        Vec::<i32>::new()
    );
    let mut none = r.get_children(-1).unwrap();
    assert_eq!(none.next_ordinal(), INVALID_ORDINAL);
    assert_eq!(none.next_ordinal(), INVALID_ORDINAL);
    assert_eq!(
        err_message(r.get_children(4)),
        "Index 4 out of bounds for length 4"
    );
    // The same arrays the second time.
    assert!(Arc::ptr_eq(&arrays, &r.parallel_taxonomy_arrays().unwrap()));
}

/// `TestDirectoryTaxonomyReader.testOpenIfChangedReuseAfterRecreate` and
/// friends, for a reader over commits: nothing new is `None`; a commit is
/// a reader sharing caches and arrays; a recreated taxonomy shares nothing.
#[test]
fn open_if_changed_over_commits() {
    let d = small_taxonomy();
    let r1 = DirectoryTaxonomyReader::open(&d).unwrap();
    assert!(r1.open_if_changed().unwrap().is_none());
    assert_eq!(r1.get_ordinal(&label(&["c"])).unwrap(), 3);
    let a1 = r1.parallel_taxonomy_arrays().unwrap();
    assert_eq!(a1.children().get(0), 3);

    {
        let w = DirectoryTaxonomyWriter::open(&d).unwrap();
        w.add_category(&label(&["a", "e"])).unwrap();
        w.close().unwrap();
    }
    let r2 = r1.open_if_changed().unwrap().unwrap();
    assert_eq!(r2.get_size().unwrap(), 5);
    assert_eq!(r2.get_ordinal(&label(&["a", "e"])).unwrap(), 4);
    let a2 = r2.parallel_taxonomy_arrays().unwrap();
    assert_eq!(a2.parents().get(4), 1);
    assert_eq!(a2.children().get(1), 4);
    assert_eq!(a2.siblings().get(4), 2);
    // Caches shared: the old reader sees what the new one looked up.
    assert!(Arc::ptr_eq(&r1.ordinal_cache, &r2.ordinal_cache));
    assert!(Arc::ptr_eq(&r1.category_cache, &r2.category_cache));

    {
        let w = DirectoryTaxonomyWriter::with_mode(&d, OpenMode::Create).unwrap();
        for c in ["x", "y", "z", "w", "v"] {
            w.add_category(&label(&[c])).unwrap();
        }
        w.close().unwrap();
    }
    let r3 = r2.open_if_changed().unwrap().unwrap();
    // `c` was cached at 3 < maxDoc; a recreated taxonomy must not answer it.
    assert!(!Arc::ptr_eq(&r2.ordinal_cache, &r3.ordinal_cache));
    assert_eq!(r3.get_ordinal(&label(&["c"])).unwrap(), INVALID_ORDINAL);
    assert_eq!(r3.get_ordinal(&label(&["z"])).unwrap(), 3);
    assert_eq!(r3.parallel_taxonomy_arrays().unwrap().parents().get(5), 0);
}

/// A taxonomy whose first commit has no epoch (another writer made it):
/// the first commit with one counts as recreated.
#[test]
fn open_if_changed_from_no_epoch() {
    let d = ByteBuffersDirectory::new();
    {
        let mut iw = IndexWriter::open(
            &d,
            Vec::new(),
            "Lucene104",
            lucene_index::merge_policy::upgrade::LATEST,
        )
        .unwrap();
        iw.commit().unwrap();
    }
    let r1 = DirectoryTaxonomyReader::open(&d).unwrap();
    assert_eq!(r1.get_size().unwrap(), 0);
    // A cache entry the recreated reader must not inherit.
    lock(&r1.ordinal_cache).put(label(&["a"]), 0);
    {
        let w = DirectoryTaxonomyWriter::open(&d).unwrap();
        w.close().unwrap();
    }
    let r2 = r1.open_if_changed().unwrap().unwrap();
    assert_eq!(r2.get_size().unwrap(), 1);
    assert_eq!(r2.get_ordinal(&label(&["a"])).unwrap(), INVALID_ORDINAL);
}

/// A near-real-time reader sees what its writer added before any commit;
/// the writer's epoch decides "recreated".
#[test]
fn open_if_changed_near_real_time() {
    let d = ByteBuffersDirectory::new();
    let w = DirectoryTaxonomyWriter::open(&d).unwrap();
    w.add_category(&label(&["a"])).unwrap();
    let r1 = DirectoryTaxonomyReader::from_writer(&w).unwrap();
    assert_eq!(r1.get_size().unwrap(), 2);
    assert!(r1.open_if_changed().unwrap().is_none());
    assert!(format!("{r1:?}").contains(":nrt"));
    assert_eq!(r1.get_ordinal(&label(&["a"])).unwrap(), 1);
    r1.parallel_taxonomy_arrays().unwrap();

    w.add_category(&label(&["b"])).unwrap();
    let r2 = r1.open_if_changed().unwrap().unwrap();
    assert_eq!(r2.get_size().unwrap(), 3);
    assert_eq!(r2.get_ordinal(&label(&["b"])).unwrap(), 2);
    assert_eq!(r2.parallel_taxonomy_arrays().unwrap().parents().get(2), 0);

    w.delete_all().unwrap();
    w.add_category(&FacetLabel::root()).unwrap();
    w.add_category(&label(&["x"])).unwrap();
    w.add_category(&label(&["y"])).unwrap();
    let r3 = r2.open_if_changed().unwrap().unwrap();
    assert_eq!(r3.get_size().unwrap(), 3);
    assert_eq!(r3.get_ordinal(&label(&["a"])).unwrap(), INVALID_ORDINAL);
    assert_eq!(r3.get_ordinal(&label(&["y"])).unwrap(), 2);
    w.rollback().unwrap();
    assert!(r3.open_if_changed().is_err());
}

/// `setCacheSize`, `ramBytesUsed` and reference counting.
#[test]
fn cache_size_ram_and_ref_count() {
    let d = small_taxonomy();
    let r = DirectoryTaxonomyReader::open(&d).unwrap();
    assert_eq!(r.ram_bytes_used().unwrap(), 0);
    r.get_ordinal(&label(&["a"])).unwrap();
    r.get_path(1).unwrap();
    assert_eq!(r.ram_bytes_used().unwrap(), 2 * BYTES_PER_CACHE_ENTRY);
    let arrays = r.parallel_taxonomy_arrays().unwrap();
    assert_eq!(r.ram_bytes_used().unwrap(), 160 + 41);
    arrays.children();
    assert_eq!(r.ram_bytes_used().unwrap(), 160 + 73);
    r.set_cache_size(1).unwrap();
    r.get_ordinal(&label(&["c"])).unwrap();
    r.get_ordinal(&label(&["a", "b"])).unwrap();
    assert_eq!(lock(&r.ordinal_cache).len(), 1);

    assert_eq!(r.ref_count(), 1);
    r.inc_ref().unwrap();
    assert_eq!(r.ref_count(), 2);
    r.dec_ref().unwrap();
    assert!(r.try_inc_ref());
    r.dec_ref().unwrap();
    r.close().unwrap();
    assert_eq!(r.ref_count(), 0);
    r.close().unwrap();
    assert!(!r.try_inc_ref());
    let closed = "this TaxonomyReader is closed";
    assert_eq!(err_message(r.dec_ref()), closed);
    assert_eq!(err_message(r.inc_ref()), closed);
    assert_eq!(err_message(r.get_ordinal(&label(&["a"]))), closed);
    assert_eq!(err_message(r.get_bulk_ordinals(&[])), closed);
    assert_eq!(err_message(r.get_path(0)), closed);
    assert_eq!(err_message(r.get_bulk_path(&[])), closed);
    assert_eq!(err_message(r.get_size()), closed);
    assert_eq!(err_message(r.commit_user_data()), closed);
    assert_eq!(err_message(r.parallel_taxonomy_arrays()), closed);
    assert_eq!(err_message(r.internal_index_reader()), closed);
    assert_eq!(err_message(r.set_cache_size(3)), closed);
    assert_eq!(err_message(r.ram_bytes_used()), closed);
    assert_eq!(err_message(r.to_string(3)), closed);
    assert_eq!(err_message(r.open_if_changed()), closed);
}

/// A taxonomy index whose documents lack their path doc value (not one
/// Lucene writes): a lookup by ordinal fails rather than inventing one.
#[test]
fn missing_path_doc_value() {
    use lucene_index::document::{Document, NumericDocValuesField};
    let d = ByteBuffersDirectory::new();
    {
        let mut iw = IndexWriter::open(
            &d,
            Vec::new(),
            "Lucene104",
            lucene_index::merge_policy::upgrade::LATEST,
        )
        .unwrap();
        let mut doc = Document::new();
        doc.add(NumericDocValuesField::new(
            super::super::consts::FIELD_PARENT_ORDINAL_NDV,
            -1,
        ));
        iw.add_fields_document(&doc).unwrap();
        let mut doc = Document::new();
        doc.add(lucene_index::document::BinaryDocValuesField::new(
            FULL,
            b"x".as_slice(),
        ));
        doc.add(NumericDocValuesField::new(
            super::super::consts::FIELD_PARENT_ORDINAL_NDV,
            0,
        ));
        iw.add_fields_document(&doc).unwrap();
        iw.commit().unwrap();
    }
    let r = DirectoryTaxonomyReader::open(&d).unwrap();
    assert_eq!(
        err_message(r.get_path(0)),
        "no $full_path$ doc value for ordinal 0"
    );
    assert_eq!(r.get_path(1).unwrap(), label(&["x"]));
    assert_eq!(r.get_ordinal(&label(&["x"])).unwrap(), INVALID_ORDINAL);
    assert_eq!(
        r.get_bulk_ordinals(&[label(&["x"]), label(&["y"])])
            .unwrap(),
        [-1, -1]
    );
    assert!(matches!(r.to_string(1), Err(Error::IllegalState(_))));
}
