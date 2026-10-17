use super::*;
use crate::facets::taxonomy::{DirectoryTaxonomyReader, TaxonomyReader};
use lucene_store::ByteBuffersDirectory;
use lucene_util::test_support::TempDir;

fn label(c: &[&str]) -> FacetLabel {
    FacetLabel::new(c.iter().copied()).unwrap()
}

fn err_message<T: std::fmt::Debug>(r: Result<T>) -> String {
    match r.unwrap_err() {
        Error::IllegalArgument(m) | Error::IllegalState(m) | Error::AlreadyClosed(m) => m,
        e => format!("{e:?}"),
    }
}

/// `TestDirectoryTaxonomyWriter`/`TestTaxonomyCombined`: ancestors are
/// added first, ordinals count up, parents chain to the root.
#[test]
fn adds_ancestors_and_answers_parents() {
    let d = ByteBuffersDirectory::new();
    let w = DirectoryTaxonomyWriter::open(&d).unwrap();
    assert_eq!(w.get_size().unwrap(), 1);
    assert_eq!(w.add_category(&FacetLabel::root()).unwrap(), 0);
    assert_eq!(w.add_category(&label(&["a", "b", "c"])).unwrap(), 3);
    assert_eq!(w.add_category(&label(&["a", "b"])).unwrap(), 2);
    assert_eq!(w.add_category(&label(&["a", "d"])).unwrap(), 4);
    assert_eq!(w.get_size().unwrap(), 5);
    let parents: Vec<i32> = (0..5).map(|o| w.get_parent(o).unwrap()).collect();
    assert_eq!(parents, [-1, 0, 1, 2, 1]);
    assert_eq!(
        err_message(w.get_parent(5)),
        "Index 5 out of bounds for length 5"
    );
    assert_eq!(
        err_message(w.get_parent(-1)),
        "Index -1 out of bounds for length 5"
    );
    assert_eq!(w.taxonomy_epoch(), 1);
    assert_eq!(w.with_cache_ref(|c| c.size()), 5);
    assert!(format!("{w:?}").starts_with("DirectoryTaxonomyWriter"));
    assert!(std::ptr::eq(
        w.directory() as *const dyn Directory as *const u8,
        &d as *const ByteBuffersDirectory as *const u8
    ));
}

/// Open modes: `APPEND` needs an index, `CREATE` starts over under a new
/// epoch, `CREATE_OR_APPEND` keeps what is there.
#[test]
fn open_modes_and_epochs() {
    let d = ByteBuffersDirectory::new();
    let err = DirectoryTaxonomyWriter::with_mode(&d, OpenMode::Append).unwrap_err();
    assert!(
        format!("{err:?}").contains("no segments* file found in directory"),
        "{err:?}"
    );
    {
        let w = DirectoryTaxonomyWriter::open(&d).unwrap();
        w.add_category(&label(&["a", "b"])).unwrap();
        w.close().unwrap();
    }
    {
        let w = DirectoryTaxonomyWriter::with_mode(&d, OpenMode::Append).unwrap();
        assert_eq!(w.get_size().unwrap(), 3);
        assert_eq!(w.taxonomy_epoch(), 1);
        // Found in the index, not the (empty) cache.
        assert_eq!(w.add_category(&label(&["a", "b"])).unwrap(), 2);
        assert_eq!(w.add_category(&label(&["a", "c"])).unwrap(), 3);
        w.close().unwrap();
    }
    {
        let w = DirectoryTaxonomyWriter::with_mode(&d, OpenMode::Create).unwrap();
        assert_eq!(w.get_size().unwrap(), 1);
        assert_eq!(w.taxonomy_epoch(), 2);
        assert_eq!(w.add_category(&label(&["z"])).unwrap(), 1);
        w.close().unwrap();
    }
    let r = DirectoryTaxonomyReader::open(&d).unwrap();
    assert_eq!(r.get_size().unwrap(), 2);
    assert_eq!(r.get_ordinal(&label(&["a"])).unwrap(), -1);
    assert!(r
        .commit_user_data()
        .unwrap()
        .contains(&(INDEX_EPOCH.to_string(), "2".to_string())));
}

/// An index another writer made: no epoch is epoch 1, an unparsable one is
/// refused as Java's `Long.parseLong` refuses it.
#[test]
fn epoch_from_a_foreign_index() {
    let plain = |data: Vec<(String, String)>| {
        let d = ByteBuffersDirectory::new();
        let mut iw = IndexWriter::open(
            &d,
            Vec::new(),
            "Lucene104",
            lucene_index::merge_policy::upgrade::LATEST,
        )
        .unwrap();
        iw.set_live_commit_data(data);
        iw.commit().unwrap();
        drop(iw);
        d
    };
    let d = plain(Vec::new());
    let w = DirectoryTaxonomyWriter::open(&d).unwrap();
    assert_eq!(w.taxonomy_epoch(), 1);
    assert_eq!(
        w.get_size().unwrap(),
        1,
        "the root is added to an empty index"
    );
    drop(w);
    let d = plain(vec![(INDEX_EPOCH.to_string(), "zz".to_string())]);
    assert_eq!(
        err_message(DirectoryTaxonomyWriter::open(&d)),
        "For input string: \"zz\" under radix 16"
    );
    let d = plain(vec![(INDEX_EPOCH.to_string(), "1f".to_string())]);
    let w = DirectoryTaxonomyWriter::with_mode(&d, OpenMode::Create).unwrap();
    assert_eq!(w.taxonomy_epoch(), 0x20);
}

#[test]
fn hex_and_commit_data() {
    assert_eq!(java_long_hex(255), "ff");
    assert_eq!(java_long_hex(-255), "-ff");
    assert_eq!(java_long_hex(i64::MIN), "-8000000000000000");
    assert_eq!(parse_java_long_hex("-ff").unwrap(), -255);
    let data = vec![
        ("k".to_string(), "v".to_string()),
        (INDEX_EPOCH.to_string(), "old".to_string()),
        ("k".to_string(), "w".to_string()),
    ];
    let combined = combined_commit_data(&data, 26);
    assert_eq!(combined.len(), 2);
    assert!(combined.contains(&("k".to_string(), "w".to_string())));
    assert!(combined.contains(&(INDEX_EPOCH.to_string(), "1a".to_string())));
}

/// `prepareCommit` then `commit`; a second `prepareCommit` is refused;
/// live commit data carries the epoch.
#[test]
fn two_phase_commit_and_commit_data() {
    let d = ByteBuffersDirectory::new();
    let w = DirectoryTaxonomyWriter::open(&d).unwrap();
    w.set_live_commit_data(vec![("k".to_string(), "v".to_string())])
        .unwrap();
    let data = w.live_commit_data().unwrap();
    assert!(data.contains(&("k".to_string(), "v".to_string())));
    assert!(data.contains(&(INDEX_EPOCH.to_string(), "1".to_string())));
    let generation = w.prepare_commit().unwrap();
    assert_eq!(
        err_message(w.prepare_commit()),
        "prepareCommit was already called with no corresponding call to commit"
    );
    assert_eq!(w.commit().unwrap(), generation);
    w.add_category(&label(&["x"])).unwrap();
    assert_eq!(w.commit().unwrap(), generation + 1);
    let r = DirectoryTaxonomyReader::open(&d).unwrap();
    assert!(r
        .commit_user_data()
        .unwrap()
        .contains(&("k".to_string(), "v".to_string())));
    assert_eq!(r.get_size().unwrap(), 2);
}

/// `rollback` drops what was not committed and closes the writer; every
/// later call is refused, `close` is a no-op.
#[test]
fn rollback_and_close() {
    let d = ByteBuffersDirectory::new();
    let w = DirectoryTaxonomyWriter::open(&d).unwrap();
    w.add_category(&label(&["kept"])).unwrap();
    w.commit().unwrap();
    w.add_category(&label(&["dropped"])).unwrap();
    w.prepare_commit().unwrap();
    w.rollback().unwrap();
    let closed = "The taxonomy writer has already been closed";
    assert_eq!(err_message(w.add_category(&label(&["y"]))), closed);
    assert_eq!(err_message(w.get_size()), closed);
    assert_eq!(err_message(w.get_parent(0)), closed);
    assert_eq!(err_message(w.commit()), closed);
    assert_eq!(err_message(w.rollback()), closed);
    assert_eq!(err_message(w.set_cache_misses_until_fill(3)), closed);
    let none = "this IndexWriter is closed";
    assert_eq!(err_message(w.set_live_commit_data(Vec::new())), none);
    assert_eq!(err_message(w.live_commit_data()), none);
    assert_eq!(err_message(w.replace_taxonomy(&d)), none);
    assert_eq!(err_message(w.delete_all()), none);
    assert_eq!(
        err_message(w.add_taxonomy(&d, &mut MemoryOrdinalMap::new())),
        closed
    );
    w.close().unwrap();
    let r = DirectoryTaxonomyReader::open(&d).unwrap();
    assert_eq!(r.get_size().unwrap(), 2);
    assert_eq!(r.get_ordinal(&label(&["dropped"])).unwrap(), -1);

    let w = DirectoryTaxonomyWriter::open(&d).unwrap();
    w.close().unwrap();
    w.close().unwrap();
}

/// `deleteAll`: nothing left, the root included, under a new epoch.
#[test]
fn delete_all_starts_over() {
    let d = ByteBuffersDirectory::new();
    let w = DirectoryTaxonomyWriter::open(&d).unwrap();
    w.add_category(&label(&["a", "b"])).unwrap();
    w.delete_all().unwrap();
    assert_eq!(w.get_size().unwrap(), 0);
    assert_eq!(w.taxonomy_epoch(), 2);
    assert_eq!(w.with_cache_ref(|c| c.size()), 0);
    assert_eq!(w.add_category(&FacetLabel::root()).unwrap(), 0);
    assert_eq!(w.add_category(&label(&["b"])).unwrap(), 1);
    w.close().unwrap();
    let r = DirectoryTaxonomyReader::open(&d).unwrap();
    assert_eq!(r.get_size().unwrap(), 2);
    assert_eq!(r.get_ordinal(&label(&["a"])).unwrap(), -1);
}

/// `replaceTaxonomy`: this taxonomy becomes the other one, a new epoch.
#[test]
fn replace_taxonomy_copies_the_source() {
    let src = ByteBuffersDirectory::new();
    {
        let w = DirectoryTaxonomyWriter::open(&src).unwrap();
        w.add_category(&label(&["s", "t"])).unwrap();
        w.close().unwrap();
    }
    let d = ByteBuffersDirectory::new();
    let w = DirectoryTaxonomyWriter::open(&d).unwrap();
    w.add_category(&label(&["x"])).unwrap();
    w.replace_taxonomy(&src).unwrap();
    assert_eq!(w.taxonomy_epoch(), 2);
    assert_eq!(w.get_size().unwrap(), 3);
    assert_eq!(w.get_parent(2).unwrap(), 1);
    assert_eq!(w.add_category(&label(&["s", "t"])).unwrap(), 2);
    assert_eq!(w.add_category(&label(&["x"])).unwrap(), 3);
}

/// The cache fill: with `cacheMissesUntilFill` 0 the first miss reads the
/// whole taxonomy into the cache (complete, so new categories are known to
/// be new); a cache too small for it stops filling when full.
#[test]
fn cache_fill_complete_and_aborted() {
    let d = ByteBuffersDirectory::new();
    {
        let w = DirectoryTaxonomyWriter::open(&d).unwrap();
        for c in ["a", "b", "c", "d"] {
            w.add_category(&label(&[c])).unwrap();
        }
        w.close().unwrap();
    }
    let w = DirectoryTaxonomyWriter::open(&d).unwrap();
    w.set_cache_misses_until_fill(0).unwrap();
    assert_eq!(w.add_category(&label(&["c"])).unwrap(), 3);
    assert_eq!(w.with_cache_ref(|c| c.size()), 5);
    assert_eq!(w.add_category(&label(&["e"])).unwrap(), 5);
    assert_eq!(w.add_category(&label(&["a"])).unwrap(), 1);
    w.rollback().unwrap();

    let w = DirectoryTaxonomyWriter::with_cache(
        &d,
        OpenMode::Append,
        Box::new(LruTaxonomyWriterCache::new(2)),
    )
    .unwrap();
    w.set_cache_misses_until_fill(0).unwrap();
    assert_eq!(w.add_category(&label(&["d"])).unwrap(), 4);
    assert_eq!(w.add_category(&label(&["b"])).unwrap(), 2);
    assert_eq!(w.add_category(&label(&["e"])).unwrap(), 5);
    assert_eq!(w.add_category(&label(&["a"])).unwrap(), 1);
}

fn source_and_dest() -> (ByteBuffersDirectory, ByteBuffersDirectory) {
    let src = ByteBuffersDirectory::new();
    {
        let w = DirectoryTaxonomyWriter::open(&src).unwrap();
        w.add_category(&label(&["a", "b"])).unwrap();
        w.add_category(&label(&["c"])).unwrap();
        w.close().unwrap();
    }
    let dest = ByteBuffersDirectory::new();
    (src, dest)
}

/// `addTaxonomy` with both ordinal maps: every source category mapped to
/// its ordinal here (the root included), in term order.
#[test]
fn add_taxonomy_with_both_maps() {
    let (src, dest) = source_and_dest();
    let w = DirectoryTaxonomyWriter::open(&dest).unwrap();
    w.add_category(&label(&["c"])).unwrap();
    w.add_category(&label(&["d"])).unwrap();
    let mut memory = MemoryOrdinalMap::new();
    w.add_taxonomy(&src, &mut memory).unwrap();
    assert_eq!(memory.get_map().unwrap(), [0, 3, 4, 1]);
    assert_eq!(w.get_size().unwrap(), 5);

    let tmp = TempDir::new("taxo-disk-map");
    let file = tmp.path().join("map");
    let mut disk = DiskOrdinalMap::new(&file).unwrap();
    w.add_taxonomy(&src, &mut disk).unwrap();
    assert_eq!(disk.get_map().unwrap(), [0, 3, 4, 1]);
    assert!(!file.exists(), "read back and deleted");
    assert_eq!(disk.get_map().unwrap(), [0, 3, 4, 1]);
    assert!(format!("{disk:?}").contains("DiskOrdinalMap"));
    assert!(matches!(disk.add_mapping(0, 0), Err(Error::Store(_))));
}

#[test]
fn ordinal_map_refusals() {
    let mut memory = MemoryOrdinalMap::default();
    assert_eq!(err_message(memory.set_size(-1)), "Illegal Capacity: -1");
    memory.set_size(2).unwrap();
    assert_eq!(
        err_message(memory.add_mapping(5, 0)),
        "Index 5 out of bounds for length 2"
    );
    assert_eq!(
        err_message(memory.add_mapping(-1, 0)),
        "Index -1 out of bounds for length 2"
    );
    memory.add_done().unwrap();

    let tmp = TempDir::new("taxo-disk-map-bad");
    let disk_map = |name: &str, f: &dyn Fn(&mut DiskOrdinalMap)| {
        let mut m = DiskOrdinalMap::new(tmp.path().join(name)).unwrap();
        f(&mut m);
        m.get_map().map(<[i32]>::to_vec)
    };
    let negative = disk_map("neg", &|m| m.set_size(-1).unwrap());
    assert_eq!(err_message(negative), "-1 is a negative array size");
    let outside = disk_map("out", &|m| {
        m.set_size(1).unwrap();
        m.add_mapping(7, 0).unwrap();
    });
    assert_eq!(err_message(outside), "Index 7 out of bounds for length 1");
    let truncated = disk_map("short", &|m| {
        m.set_size(2).unwrap();
        m.add_mapping(0, 0).unwrap();
    });
    assert!(matches!(truncated, Err(Error::Store(_))));
    assert!(DiskOrdinalMap::new(tmp.path().join("no/such/dir/map")).is_err());
}

/// A cache that evicts makes the writer refresh its reader, so a category
/// evicted from the cache is still found in the index.
#[test]
fn evicting_cache_falls_back_to_the_index() {
    let d = ByteBuffersDirectory::new();
    let w = DirectoryTaxonomyWriter::with_cache(
        &d,
        OpenMode::Create,
        Box::new(LruTaxonomyWriterCache::new(3)),
    )
    .unwrap();
    // No fill: every miss goes to the index.
    w.set_cache_misses_until_fill(i32::MAX).unwrap();
    let names: Vec<String> = (0..20).map(|i| format!("c{i}")).collect();
    for (i, n) in names.iter().enumerate() {
        assert_eq!(w.add_category(&label(&[n])).unwrap(), i as i32 + 1);
    }
    for (i, n) in names.iter().enumerate().rev() {
        assert_eq!(w.add_category(&label(&[n])).unwrap(), i as i32 + 1);
    }
    assert_eq!(w.get_size().unwrap(), 21);
}
