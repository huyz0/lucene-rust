//! M13 T13.4d2 against `fixtures/src/GenTaxonomyUtils.java`: the taxonomy
//! utilities.
//!
//! - `TaxonomyMergeUtils.merge` of the Java-written source index and taxonomy
//!   into a copy of the destination: the ordinal map, the merged taxonomy,
//!   every live document's facet and other sorted-numeric values and the
//!   merged counts (`merge.tsv`).
//! - `OrdinalMappingLeafReader` over each source leaf (`mapping.tsv`).
//! - `PrintTaxonomyStats` of the merged taxonomy (`stats.txt`,
//!   `stats_tree.txt`).
//! - `ReindexingEnrichedDirectoryTaxonomyWriter`'s documents at each stage,
//!   written here and read back from Lucene's copy (`enriched.tsv`).
//! - `SearcherTaxonomyManager`, near-real-time and over directories, step
//!   for step (`manager.tsv`).
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    BinaryDocValuesField, Document, NumericDocValuesField, SortedNumericDocValuesField, Store,
    StringField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::facets::taxonomy::directory::{
    DirectoryTaxonomyReader, DirectoryTaxonomyWriter, MemoryOrdinalMap, OrdinalMap,
    ReindexingEnrichedDirectoryTaxonomyWriter,
};
use lucene_search::facets::taxonomy::ordinal_mapping_leaf_reader::ordinal_mapping_leaf_reader;
use lucene_search::facets::taxonomy::{
    print_taxonomy_stats, taxonomy_merge_utils, FacetLabel, FastTaxonomyFacetCounts,
    SearcherTaxonomyManager, TaxonomyReader, TaxonomyWriter,
};
use lucene_search::facets::{FacetEntry, FacetField, FacetResult, Facets, FacetsConfig};
use lucene_search::reader::{IndexReader, LeafReader, SortedNumericDocValues, NO_MORE_DOCS};
use lucene_search::Error;
use lucene_store::{ByteBuffersDirectory, Directory, FsDirectory};

fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/taxonomy_utils")
}

fn read(p: &Path) -> String {
    fs::read_to_string(p).unwrap_or_else(|e| {
        panic!(
            "{}: {e} (run scripts/gen-fixtures.sh --only GenTaxonomyUtils)",
            p.display()
        )
    })
}

/// Line-by-line comparison that names the first differing lines.
fn assert_text_eq(what: &str, got: &str, want: &str) {
    let (g, w): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
    let mut diffs = Vec::new();
    for i in 0..g.len().max(w.len()) {
        let (a, b) = (g.get(i).copied(), w.get(i).copied());
        if a != b {
            diffs.push(format!(
                "line {}:\n  rust: {}\n  java: {}",
                i + 1,
                a.unwrap_or("<none>"),
                b.unwrap_or("<none>")
            ));
        }
    }
    assert!(
        diffs.is_empty(),
        "{what}: {} of {} lines differ\n{}",
        diffs.len(),
        w.len(),
        diffs.into_iter().take(12).collect::<Vec<_>>().join("\n")
    );
}

/// `GenTaxonomyUtils.config()`.
fn config() -> FacetsConfig {
    let mut c = FacetsConfig::new();
    c.set_hierarchical("Publish Date", true);
    c.set_multi_valued("Tag", true);
    c.set_index_field_name("Tag", "$tags");
    c
}

fn ff(dim: &str, path: &[&str]) -> FacetEntry {
    FacetField::new(dim, path).unwrap().into()
}

/// `GenTaxonomyUtils.srcDoc(i)`, split into its facet fields and the rest.
fn src_doc(i: i64) -> (Vec<FacetEntry>, Document) {
    let mut d = Document::new();
    d.add(StringField::new("q", "all", Store::No));
    if i % 9 == 4 {
        d.add(StringField::new("del", "y", Store::No));
    }
    let f = vec![
        ff("Author", &[&format!("a{}", i % 6)]),
        ff(
            "Publish Date",
            &[&(2010 + i % 3).to_string(), &(1 + i % 4).to_string()],
        ),
        ff("Tag", &[&format!("t{}", i % 4)]),
        ff("Tag", &[&format!("u{}", i % 5)]),
    ];
    d.add(SortedNumericDocValuesField::new("num", i));
    d.add(SortedNumericDocValuesField::new("num", (i * 3) % 7));
    (f, d)
}

fn line(sb: &mut String, cols: &[&str]) {
    sb.push_str(&cols.join("\t"));
    sb.push('\n');
}

fn label_path(l: &FacetLabel) -> String {
    l.path().join("/")
}

fn values(dv: &mut dyn SortedNumericDocValues) -> String {
    let mut out = Vec::new();
    for _ in 0..dv.doc_value_count() {
        out.push(dv.next_value().unwrap().to_string());
    }
    out.join(",")
}

/// `GenTaxonomyUtils.res`: values as plain ints.
fn res(r: &FacetResult) -> String {
    let children: Vec<String> = r
        .label_values
        .iter()
        .map(|lv| format!("{}={}={}", lv.label, lv.value.int_value(), lv.count))
        .collect();
    format!(
        "{}|{}|{}|{}|{}",
        r.dim,
        r.path.join("/"),
        r.value.int_value(),
        r.child_count,
        children.join(",")
    )
}

fn list(l: &[FacetResult]) -> String {
    l.iter().map(res).collect::<Vec<_>>().join(";")
}

/// Copies every file of `from` but its lock into a new in-memory directory.
fn copy_dir(from: &dyn Directory) -> ByteBuffersDirectory {
    let to = ByteBuffersDirectory::new();
    for f in from.list_all().unwrap() {
        if f != "write.lock" {
            to.copy_from(from, &f, &f).unwrap();
        }
    }
    to
}

/// `GenTaxonomyUtils.dumpIndex`.
fn dump_index(sb: &mut String, reader: &DirectoryReader) {
    for ctx in reader.leaves() {
        let r = ctx.reader;
        let leaf = ctx.ord.to_string();
        line(
            sb,
            &[
                "leaf",
                &leaf,
                &r.max_doc().to_string(),
                &r.num_docs().to_string(),
            ],
        );
        let live = r.live_docs();
        for field in ["$facets", "$tags", "num"] {
            let Some(mut dv) = r.sorted_numeric_doc_values(field).unwrap() else {
                continue;
            };
            let mut doc = dv.next_doc().unwrap();
            while doc != NO_MORE_DOCS {
                if live.is_none_or(|l| l.get(doc as usize)) {
                    let v = values(dv.as_mut());
                    line(sb, &["doc", &leaf, &doc.to_string(), field, &v]);
                }
                doc = dv.next_doc().unwrap();
            }
        }
    }
}

fn stats(taxo: &dyn TaxonomyReader, tree: bool) -> String {
    let mut out = Vec::new();
    print_taxonomy_stats::print_stats(taxo, &mut out, tree).unwrap();
    String::from_utf8(out).unwrap()
}

/// `merge.tsv`, `mapping.tsv`, `stats.txt` and `stats_tree.txt`.
#[test]
fn merge_mapping_and_stats_match_lucene() {
    let config = config();
    let sid = FsDirectory::open(data().join("src/index"));
    let std_ = FsDirectory::open(data().join("src/taxo"));
    let mid = copy_dir(&FsDirectory::open(data().join("dest/index")));
    let mtd = copy_dir(&FsDirectory::open(data().join("dest/taxo")));

    let mut merge = String::new();
    let ord_map: Vec<i32>;
    {
        let mut w = IndexWriter::open(&mid, Vec::new(), "Lucene104", LATEST).unwrap();
        let tw = DirectoryTaxonomyWriter::open(&mtd).unwrap();
        let mut map = MemoryOrdinalMap::new();
        taxonomy_merge_utils::merge(&sid, &std_, &mut map, &mut w, &tw, &config).unwrap();
        ord_map = map.get_map().unwrap().to_vec();
        drop(w);
        tw.close().unwrap();
    }
    let ints: Vec<String> = ord_map.iter().map(i32::to_string).collect();
    line(&mut merge, &["map", &ints.join(",")]);
    let taxo = DirectoryTaxonomyReader::open(&mtd).unwrap();
    let reader = DirectoryReader::open(&mid).unwrap();
    for ord in 0..taxo.get_size().unwrap() {
        let p = label_path(&taxo.get_path(ord).unwrap());
        line(&mut merge, &["taxo", &ord.to_string(), &p]);
    }
    dump_index(&mut merge, &reader);
    for field in ["$facets", "$tags"] {
        let counts = FastTaxonomyFacetCounts::count_all(field, &reader, &taxo, &config).unwrap();
        line(
            &mut merge,
            &["counts", field, &list(&counts.get_all_dims(10).unwrap())],
        );
    }
    assert_text_eq("merge.tsv", &merge, &read(&data().join("merge.tsv")));
    assert_text_eq(
        "stats.txt",
        &stats(&taxo, false),
        &read(&data().join("stats.txt")),
    );
    assert_text_eq(
        "stats_tree.txt",
        &stats(&taxo, true),
        &read(&data().join("stats_tree.txt")),
    );

    let mut mapping = String::new();
    let src = DirectoryReader::open(&sid).unwrap();
    let map: Arc<[i32]> = ord_map.into();
    for (ord, seg) in src.segment_readers().iter().enumerate() {
        let r = ordinal_mapping_leaf_reader(Arc::new(seg.clone()), Arc::clone(&map), &config);
        let leaf = ord.to_string();
        line(&mut mapping, &["leaf", &leaf, &r.max_doc().to_string()]);
        let mut facets = r.sorted_numeric_doc_values("$facets").unwrap().unwrap();
        let mut doc = facets.next_doc().unwrap();
        while doc != NO_MORE_DOCS {
            let v = values(facets.as_mut());
            line(
                &mut mapping,
                &["doc", &leaf, &doc.to_string(), "$facets", &v],
            );
            doc = facets.next_doc().unwrap();
        }
        let mut tags = r.sorted_numeric_doc_values("$tags").unwrap().unwrap();
        for doc in (0..r.max_doc()).step_by(3) {
            if tags.advance_exact(doc).unwrap() {
                let v = values(tags.as_mut());
                line(&mut mapping, &["doc", &leaf, &doc.to_string(), "$tags", &v]);
            }
        }
        let mut num = r.sorted_numeric_doc_values("num").unwrap().unwrap();
        let mut doc = num.advance(5).unwrap();
        while doc != NO_MORE_DOCS {
            let v = values(num.as_mut());
            line(&mut mapping, &["doc", &leaf, &doc.to_string(), "num", &v]);
            doc = num.advance(doc + 7).unwrap();
        }
        let absent = r.sorted_numeric_doc_values("nope").unwrap();
        line(
            &mut mapping,
            &[
                "absent",
                &leaf,
                if absent.is_none() { "null" } else { "present" },
            ],
        );
    }
    assert_text_eq("mapping.tsv", &mapping, &read(&data().join("mapping.tsv")));
}

/// `GenTaxonomyUtils.dumpEnriched`.
fn dump_enriched(sb: &mut String, stage: &str, d: &dyn Directory) {
    let taxo = DirectoryTaxonomyReader::open(d).unwrap();
    let ir = taxo.internal_index_reader().unwrap();
    for ctx in ir.leaves() {
        let r = ctx.reader;
        let mut len = r.numeric_doc_values("len").unwrap();
        let mut tag = r.binary_doc_values("tag").unwrap();
        for doc in 0..r.max_doc() {
            let ord = ctx.doc_base + doc;
            let mut l = "-".to_string();
            if let Some(dv) = len.as_mut() {
                if dv.advance_exact(doc).unwrap() {
                    l = dv.long_value().to_string();
                }
            }
            let mut t = "-".to_string();
            if let Some(dv) = tag.as_mut() {
                if dv.advance_exact(doc).unwrap() {
                    t = String::from_utf8(dv.binary_value().to_vec()).unwrap();
                }
            }
            let p = label_path(&taxo.get_path(ord).unwrap());
            line(sb, &[stage, &ord.to_string(), &p, &l, &t]);
        }
    }
    let size = taxo.get_size().unwrap().to_string();
    line(sb, &[&format!("{stage}_size"), &size]);
}

fn label(components: &[&str]) -> FacetLabel {
    FacetLabel::new(components.iter().copied()).unwrap()
}

/// `enriched.tsv`, written here, and its `final` stage read back from
/// Lucene's taxonomy.
#[test]
fn reindexing_enriched_writer_matches_lucene() {
    let want = read(&data().join("enriched.tsv"));
    let d = ByteBuffersDirectory::new();
    let mut sb = String::new();
    {
        let tw = ReindexingEnrichedDirectoryTaxonomyWriter::new(
            &d,
            Some(Box::new(|cp: &FacetLabel, doc: &mut Document| {
                doc.add(NumericDocValuesField::new("len", cp.len() as i64));
            })),
        )
        .unwrap();
        tw.add_category(&label(&["A", "x"])).unwrap();
        tw.add_category(&label(&["A", "y"])).unwrap();
        tw.add_category(&label(&["B", "p", "q"])).unwrap();
        tw.add_category(&label(&["C"])).unwrap();
        tw.commit().unwrap();
        line(
            &mut sb,
            &["epoch", "before", &tw.taxonomy_epoch().to_string()],
        );
        dump_enriched(&mut sb, "before", &d);
        tw.reindex_with_new_ordinal_data(Some(Box::new(|cp: &FacetLabel, doc: &mut Document| {
            doc.add(BinaryDocValuesField::new(
                "tag",
                format!("{}!", cp.path().join("/")).into_bytes(),
            ));
        })))
        .unwrap();
        line(
            &mut sb,
            &["epoch", "after", &tw.taxonomy_epoch().to_string()],
        );
        dump_enriched(&mut sb, "after", &d);
        let a = tw.add_category(&label(&["D", "z"])).unwrap();
        line(&mut sb, &["add", &a.to_string()]);
        let b = tw.add_category(&label(&["A", "x"])).unwrap();
        line(&mut sb, &["add", &b.to_string()]);
        tw.commit().unwrap();
        dump_enriched(&mut sb, "final", &d);
        tw.writer().close().unwrap();
        assert!(matches!(
            tw.reindex_with_new_ordinal_data(None),
            Err(Error::AlreadyClosed(_))
        ));
    }
    assert_text_eq("enriched.tsv", &sb, &want);

    let mut java = String::new();
    dump_enriched(
        &mut java,
        "final",
        &FsDirectory::open(data().join("enriched/taxo")),
    );
    let want_final: String = want
        .lines()
        .filter(|l| l.starts_with("final"))
        .map(|l| format!("{l}\n"))
        .collect();
    assert_text_eq("enriched/taxo", &java, &want_final);
}

/// `GenTaxonomyUtils.step`.
fn step(
    sb: &mut String,
    mode: &str,
    name: &str,
    m: &SearcherTaxonomyManager<'_>,
    config: &FacetsConfig,
) {
    let sat = m.acquire().unwrap();
    let r = &sat.searcher;
    let counts =
        FastTaxonomyFacetCounts::count_all("$facets", r, sat.taxonomy_reader.as_ref(), config)
            .unwrap();
    let mut row = String::new();
    write!(
        row,
        "{}\t{}\t{}\t{}\t{}",
        r.num_docs(),
        sat.taxonomy_reader.get_size().unwrap(),
        m.is_searcher_current().unwrap(),
        m.is_taxonomy_current().unwrap(),
        list(&counts.get_all_dims(10).unwrap())
    )
    .unwrap();
    line(sb, &[mode, name, &row]);
    m.release(sat);
}

fn add(
    w: &Mutex<IndexWriter<'_>>,
    tw: &DirectoryTaxonomyWriter<'_>,
    config: &FacetsConfig,
    i: i64,
) {
    let (facets, d) = src_doc(i);
    let built = config.build(Some(tw), &facets, d).unwrap();
    w.lock().unwrap().add_fields_document(&built).unwrap();
}

/// `GenTaxonomyUtils.nrtManager`.
fn nrt_manager(sb: &mut String, config: &FacetsConfig) {
    let id = ByteBuffersDirectory::new();
    let td = ByteBuffersDirectory::new();
    let other = ByteBuffersDirectory::new();
    {
        let ow = DirectoryTaxonomyWriter::open(&other).unwrap();
        ow.add_category(&label(&["Zed", "z"])).unwrap();
        ow.commit().unwrap();
        ow.close().unwrap();
    }
    let w = Mutex::new(IndexWriter::open(&id, Vec::new(), "Lucene104", LATEST).unwrap());
    let tw = DirectoryTaxonomyWriter::open(&td).unwrap();
    for i in 0..3 {
        add(&w, &tw, config, i);
    }
    let m = SearcherTaxonomyManager::from_writers(&w, &tw).unwrap();
    step(sb, "nrt", "init", &m, config);
    for i in 3..8 {
        add(&w, &tw, config, i);
    }
    step(sb, "nrt", "added", &m, config);
    let before = m.acquire().unwrap();
    line(
        sb,
        &["nrt", "refresh", &m.maybe_refresh().unwrap().to_string()],
    );
    step(sb, "nrt", "refreshed", &m, config);
    let after = m.acquire().unwrap();
    line(
        sb,
        &[
            "nrt",
            "changed",
            &(!Arc::ptr_eq(&before, &after)).to_string(),
        ],
    );
    m.release(before);
    line(
        sb,
        &["nrt", "refresh", &m.maybe_refresh().unwrap().to_string()],
    );
    let again = m.acquire().unwrap();
    line(
        sb,
        &[
            "nrt",
            "changed",
            &(!Arc::ptr_eq(&again, &after)).to_string(),
        ],
    );
    // A delete alone: the index reader changes, the taxonomy does not.
    let taxo_before = Arc::clone(&again.taxonomy_reader);
    m.release(again);
    m.release(after);
    w.lock()
        .unwrap()
        .delete_documents_by_term(&[Term::new("del", "y")])
        .unwrap();
    line(
        sb,
        &["nrt", "refresh", &m.maybe_refresh().unwrap().to_string()],
    );
    step(sb, "nrt", "deleted", &m, config);
    assert!(Arc::ptr_eq(
        &taxo_before,
        &m.acquire().unwrap().taxonomy_reader
    ));
    tw.replace_taxonomy(&other).unwrap();
    add(&w, &tw, config, 9);
    match m.maybe_refresh() {
        Ok(_) => line(sb, &["nrt", "replace", "no error"]),
        Err(Error::IllegalState(msg)) => line(sb, &["nrt", "replace", &format!("ISE:{msg}")]),
        Err(e) => panic!("{e:?}"),
    }
    m.close();
    assert!(matches!(m.acquire(), Err(Error::AlreadyClosed(_))));
    assert!(matches!(m.maybe_refresh(), Err(Error::AlreadyClosed(_))));
}

/// `GenTaxonomyUtils.dirManager`.
fn dir_manager(sb: &mut String, config: &FacetsConfig) {
    let id = ByteBuffersDirectory::new();
    let td = ByteBuffersDirectory::new();
    let mut w = IndexWriter::open(&id, Vec::new(), "Lucene104", LATEST).unwrap();
    let tw = DirectoryTaxonomyWriter::open(&td).unwrap();
    let add = |w: &mut IndexWriter<'_>, i: i64| {
        let (facets, d) = src_doc(i);
        w.add_fields_document(&config.build(Some(&tw), &facets, d).unwrap())
            .unwrap();
    };
    for i in 0..4 {
        add(&mut w, i);
    }
    tw.commit().unwrap();
    w.commit().unwrap();
    let m = SearcherTaxonomyManager::from_directories(&id, &td).unwrap();
    step(sb, "dir", "init", &m, config);
    let generation = m.searcher_commit_generation().unwrap().to_string();
    line(sb, &["dir", "generation", &generation]);
    for i in 4..10 {
        add(&mut w, i);
    }
    line(
        sb,
        &["dir", "refresh", &m.maybe_refresh().unwrap().to_string()],
    );
    step(sb, "dir", "uncommitted", &m, config);
    tw.commit().unwrap();
    w.commit().unwrap();
    step(sb, "dir", "committed", &m, config);
    line(
        sb,
        &["dir", "refresh", &m.maybe_refresh().unwrap().to_string()],
    );
    step(sb, "dir", "refreshed", &m, config);
    let generation = m.searcher_commit_generation().unwrap().to_string();
    line(sb, &["dir", "generation", &generation]);
    m.close();
}

/// `manager.tsv`.
#[test]
fn searcher_taxonomy_manager_matches_lucene() {
    let config = config();
    let mut sb = String::new();
    nrt_manager(&mut sb, &config);
    dir_manager(&mut sb, &config);
    assert_text_eq("manager.tsv", &sb, &read(&data().join("manager.tsv")));
}

/// `PrintTaxonomyStats.main`: the usage for bad arguments, the stats of the
/// taxonomy at a path otherwise.
#[test]
fn print_taxonomy_stats_main() {
    let taxo = data().join("src/taxo");
    let taxo = taxo.to_str().unwrap();
    for bad in [
        &[][..],
        &["-printTree"][..],
        &["a", "b"][..],
        &["-printTree", "a", "b"][..],
    ] {
        let mut out = Vec::new();
        assert_eq!(print_taxonomy_stats::run(bad, &mut out).unwrap(), 1);
        assert_eq!(
            String::from_utf8(out).unwrap(),
            format!("{}\n", print_taxonomy_stats::USAGE)
        );
    }
    let mut out = Vec::new();
    assert_eq!(print_taxonomy_stats::run(&[taxo], &mut out).unwrap(), 0);
    let plain = String::from_utf8(out).unwrap();
    let mut out = Vec::new();
    assert_eq!(
        print_taxonomy_stats::run(&["-printTree", taxo], &mut out).unwrap(),
        0
    );
    let tree = String::from_utf8(out).unwrap();
    let dir = FsDirectory::open(data().join("src/taxo"));
    let r = DirectoryTaxonomyReader::open(&dir).unwrap();
    assert_eq!(plain, stats(&r, false));
    assert_eq!(tree, stats(&r, true));
    assert!(tree.len() > plain.len());
}
