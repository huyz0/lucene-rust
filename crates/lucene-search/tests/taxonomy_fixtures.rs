//! M13 T13.4a against `fixtures/src/GenTaxonomy.java`: the taxonomy index.
//!
//! - Every taxonomy Lucene wrote reads, through [`DirectoryTaxonomyReader`],
//!   exactly as Lucene's reader dumps it (ordinals, parents, children,
//!   siblings, paths, lookups, bulk lookups, `toString`, commit data).
//! - [`DirectoryTaxonomyWriter`] replays every scenario (four writer caches,
//!   three stages: a new index, a reopened one, a recreated one) and hands
//!   out the ordinals and parents Lucene did; the taxonomies it writes dump
//!   as Lucene's do. `scripts/verify-write-path.sh`'s `taxonomy` case has
//!   Lucene read them.
//! - `addTaxonomy`, `replaceTaxonomy`, `FacetsConfig.build`, `FacetLabel`'s
//!   hashes/ordering and the refusals' messages match.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use lucene_index::deletion_policy::OpenMode;
use lucene_index::document::{DocValuesType, Document, Number, Store, StringField};
use lucene_search::facets::taxonomy::directory::{
    DirectoryTaxonomyReader, DirectoryTaxonomyWriter, MemoryOrdinalMap, OrdinalMap,
};
use lucene_search::facets::taxonomy::writercache::{
    LruTaxonomyWriterCache, LruType, TaxonomyWriterCache,
};
use lucene_search::facets::taxonomy::{FacetLabel, TaxonomyReader, TaxonomyWriter};
use lucene_search::facets::{
    DrillDownTermsIndexing, FacetEntry, FacetField, FacetsConfig, SortedSetDocValuesFacetField,
};
use lucene_search::Error;
use lucene_store::{ByteBuffersDirectory, Directory, FsDirectory};
use lucene_util::test_support::TempDir;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/taxonomy")
}

fn read(p: &Path) -> String {
    fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

const MAX: usize = FacetLabel::MAX_CATEGORY_PATH_LENGTH;

fn long_exact() -> String {
    "x".repeat(MAX - 5)
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

/// `GenTaxonomy.stage(n)`.
fn stage(n: usize) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    match n {
        1 => {
            for i in 0..12 {
                out.push(vec!["Author".into(), format!("author{i}")]);
            }
            for y in 2000..2003 {
                for m in 1..=3 {
                    out.push(vec![
                        "Pub Date".into(),
                        y.to_string(),
                        m.to_string(),
                        "1".into(),
                    ]);
                    out.push(vec![
                        "Pub Date".into(),
                        y.to_string(),
                        m.to_string(),
                        "15".into(),
                    ]);
                }
            }
            out.push(s(&["Ünïcødé", "日本", "語"]));
            out.push(s(&["Ünïcødé", "中文"]));
            out.push(s(&["Ünïcødé", "\u{FFFF}"]));
            out.push(s(&["Ünïcødé", "😀z"]));
            out.push(s(&["sep\u{1F}dim", "a\u{1F}b", "c\u{1E}"]));
            out.push(s(&["esc\u{1E}dim", "\u{1E}\u{1E}", "\u{1F}"]));
            out.push(vec!["long".into(), long_exact()]);
            out.push(vec!["long".into(), "y".repeat(1000), "z".repeat(1000)]);
            out.push(s(&["Author", "author3"]));
        }
        2 => {
            for i in (0..12).rev().step_by(2) {
                out.push(vec!["Author".into(), format!("author{i}")]);
            }
            out.push(s(&["Pub Date", "2001", "2", "15"]));
            out.push(s(&["Pub Date", "2001", "2", "16"]));
            out.push(s(&["Pub Date", "2004"]));
            out.push(s(&["Pub Date", "2004", "1", "1"]));
            for i in 0..15 {
                out.push(vec!["Tag".into(), format!("tag{}", i * 7 % 15)]);
            }
            for i in 0..12 {
                out.push(vec!["Author".into(), format!("author{i}")]);
            }
            let mut deep = vec!["deep".to_string()];
            for i in 0..30 {
                deep.push(format!("d{i}"));
            }
            out.push(deep);
            out.push(s(&["Ünïcødé", "日本"]));
            out.push(s(&["Ünïcødé", "한국어"]));
            out.push(vec!["long".into(), long_exact()]);
        }
        3 => {
            out.push(s(&["Tag", "tag1"]));
            out.push(s(&["Author", "author5"]));
            out.push(s(&["Pub Date", "1999", "12", "31"]));
            out.push(s(&["Tag", "tag2"]));
            out.push(s(&["Author", "author5"]));
        }
        _ => unreachable!(),
    }
    out
}

/// `GenTaxonomy.PROBES`.
fn probes() -> Vec<Vec<String>> {
    vec![
        vec![],
        s(&["Author"]),
        s(&["Author", "author3"]),
        s(&["Author", "author99"]),
        s(&["Pub Date", "2001", "2"]),
        s(&["Pub Date", "2004", "1", "1"]),
        s(&["Tag", "tag1"]),
        s(&["Ünïcødé", "😀z"]),
        s(&["sep\u{1F}dim", "a\u{1F}b", "c\u{1E}"]),
        s(&["sep\u{1F}dim", "a", "b"]),
        s(&["esc\u{1E}dim", "\u{1E}\u{1E}", "\u{1F}"]),
        vec!["long".into(), long_exact()],
        s(&["deep", "d0", "d1", "d2"]),
        s(&["nope"]),
    ]
}

fn label(p: &[String]) -> FacetLabel {
    FacetLabel::new(p.iter().cloned()).unwrap()
}

/// `GenTaxonomy.dump(reader)`.
fn dump(r: &DirectoryTaxonomyReader<'_>) -> String {
    let mut sb = String::new();
    let size = r.get_size().unwrap();
    writeln!(sb, "size\t{size}").unwrap();
    let mut user = r.commit_user_data().unwrap();
    user.sort();
    for (k, v) in user {
        writeln!(sb, "user\t{k}\t{v}").unwrap();
    }
    let arrays = r.parallel_taxonomy_arrays().unwrap();
    for o in 0..size {
        write!(
            sb,
            "ord\t{o}\t{}\t{}\t{}",
            arrays.parents().get(o),
            arrays.children().get(o),
            arrays.siblings().get(o)
        )
        .unwrap();
        for c in r.get_path(o).unwrap().components() {
            write!(sb, "\t{c}").unwrap();
        }
        sb.push('\n');
    }
    for o in 0..size {
        write!(sb, "children\t{o}").unwrap();
        for c in r.get_children(o).unwrap() {
            write!(sb, "\t{c}").unwrap();
        }
        sb.push('\n');
    }
    let probes: Vec<FacetLabel> = probes().iter().map(|p| label(p)).collect();
    for (i, p) in probes.iter().enumerate() {
        writeln!(sb, "lookup\t{i}\t{}", r.get_ordinal(p).unwrap()).unwrap();
    }
    sb.push_str("bulk_ord");
    for o in r.get_bulk_ordinals(&probes).unwrap() {
        write!(sb, "\t{o}").unwrap();
    }
    sb.push('\n');
    let ords: Vec<i32> = (0..size).rev().collect();
    sb.push_str("bulk_path");
    for l in r.get_bulk_path(&ords).unwrap() {
        write!(sb, "\t{}", l.len()).unwrap();
    }
    sb.push('\n');
    sb
}

fn assert_text_eq(what: &str, actual: &str, expected: &str) {
    if actual == expected {
        return;
    }
    let a: Vec<&str> = actual.lines().collect();
    let e: Vec<&str> = expected.lines().collect();
    for (i, (x, y)) in a.iter().zip(&e).enumerate() {
        if x != y {
            let cut = |s: &str| s.chars().take(300).collect::<String>();
            panic!(
                "{what}: line {} differs\n  rust: {}\n  java: {}",
                i + 1,
                cut(x),
                cut(y)
            );
        }
    }
    panic!("{what}: {} lines, expected {}", a.len(), e.len());
}

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() == "write.lock" {
            continue;
        }
        fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
}

const SCENARIOS: [&str; 4] = ["default", "lru_hashed_3", "lru_string_3", "lru_unbounded"];

fn scenario_cache(name: &str) -> Option<Box<dyn TaxonomyWriterCache>> {
    match name {
        "default" => None,
        "lru_hashed_3" => Some(Box::new(LruTaxonomyWriterCache::with_type(
            3,
            LruType::LruHashed,
        ))),
        "lru_string_3" => Some(Box::new(LruTaxonomyWriterCache::with_type(
            3,
            LruType::LruString,
        ))),
        "lru_unbounded" => Some(Box::new(LruTaxonomyWriterCache::new(i32::MAX))),
        _ => unreachable!(),
    }
}

fn open_writer<'d>(
    dir: &'d dyn Directory,
    mode: OpenMode,
    cache: Option<Box<dyn TaxonomyWriterCache>>,
) -> DirectoryTaxonomyWriter<'d> {
    match cache {
        Some(c) => DirectoryTaxonomyWriter::with_cache(dir, mode, c).unwrap(),
        None => DirectoryTaxonomyWriter::with_mode(dir, mode).unwrap(),
    }
}

#[test]
fn java_taxonomies_read_as_lucene_reads_them() {
    let base = fixtures();
    let mut dirs: Vec<(PathBuf, PathBuf, Option<PathBuf>)> = Vec::new();
    for sc in SCENARIOS {
        for st in 1..=3 {
            dirs.push((
                base.join(sc).join(format!("stage{st}")),
                base.join(sc).join(format!("stage{st}.tsv")),
                Some(base.join(sc).join(format!("stage{st}.tostring.txt"))),
            ));
        }
    }
    dirs.push((
        base.join("add_taxonomy/dest"),
        base.join("add_taxonomy/dest.tsv"),
        None,
    ));
    dirs.push((
        base.join("replace/dest"),
        base.join("replace/dest.tsv"),
        None,
    ));
    dirs.push((base.join("build/taxo"), base.join("build/taxo.tsv"), None));
    for (dir, tsv, tostring) in dirs {
        let fs_dir = FsDirectory::open(&dir);
        let r = DirectoryTaxonomyReader::open(&fs_dir).unwrap();
        assert_text_eq(&dir.display().to_string(), &dump(&r), &read(&tsv));
        if let Some(t) = tostring {
            // A fresh reader: `toString` fills the path cache in order.
            let r = DirectoryTaxonomyReader::open(&fs_dir).unwrap();
            assert_text_eq(
                &t.display().to_string(),
                &r.to_string(1000).unwrap(),
                &read(&t),
            );
        }
    }
}

/// Replays one scenario with the Rust writer: the ops it logs and the
/// taxonomies it leaves must be Lucene's.
fn replay(sc: &str, out: &Path) {
    let base = fixtures().join(sc);
    let mut ops = String::new();
    for st in 1..=3usize {
        let dir = out.join(format!("stage{st}"));
        if st > 1 {
            copy_dir(&out.join(format!("stage{}", st - 1)), &dir);
        } else {
            fs::create_dir_all(&dir).unwrap();
        }
        let mode = if st == 2 {
            OpenMode::CreateOrAppend
        } else {
            OpenMode::Create
        };
        {
            let fs_dir = FsDirectory::open(&dir);
            let w = open_writer(&fs_dir, mode, scenario_cache(sc));
            for path in stage(st) {
                let ord = w.add_category(&label(&path)).unwrap();
                write!(ops, "{st}\tadd\t{ord}").unwrap();
                for c in &path {
                    write!(ops, "\t{c}").unwrap();
                }
                ops.push('\n');
            }
            for o in 0..w.get_size().unwrap() {
                writeln!(ops, "{st}\tparent\t{o}\t{}", w.get_parent(o).unwrap()).unwrap();
            }
            writeln!(ops, "{st}\tsize\t{}", w.get_size().unwrap()).unwrap();
            if st == 1 {
                w.set_live_commit_data(vec![("user".into(), "stage1".into())])
                    .unwrap();
            }
            w.close().unwrap();
        }
        let expected_ops: String = read(&base.join("ops.tsv"))
            .lines()
            .filter(|l| l.starts_with(&format!("{st}	")))
            .map(|l| {
                format!(
                    "{l}
"
                )
            })
            .collect();
        let stage_ops: String = ops
            .lines()
            .filter(|l| l.starts_with(&format!("{st}	")))
            .map(|l| {
                format!(
                    "{l}
"
                )
            })
            .collect();
        assert_text_eq(&format!("{sc} stage{st} ops"), &stage_ops, &expected_ops);
        let fs_dir = FsDirectory::open(&dir);
        let r = DirectoryTaxonomyReader::open(&fs_dir).unwrap();
        assert_text_eq(
            &format!("{sc} stage{st} (rust-written)"),
            &dump(&r),
            &read(&base.join(format!("stage{st}.tsv"))),
        );
    }
    assert_text_eq(&format!("{sc} ops"), &ops, &read(&base.join("ops.tsv")));
}

fn replay_in_temp(sc: &str) {
    let tmp = TempDir::new(&format!("taxo-{sc}"));
    replay(sc, tmp.path());
}

#[test]
fn rust_writer_replays_default_cache() {
    replay_in_temp(SCENARIOS[0]);
}

#[test]
fn rust_writer_replays_lru_hashed_3() {
    replay_in_temp(SCENARIOS[1]);
}

#[test]
fn rust_writer_replays_lru_string_3() {
    replay_in_temp(SCENARIOS[2]);
}

#[test]
fn rust_writer_replays_lru_unbounded() {
    replay_in_temp(SCENARIOS[3]);
}

#[test]
fn add_taxonomy_maps_as_lucene_does() {
    let tmp = TempDir::new("taxo-add");
    let dest = tmp.path().join("dest");
    fs::create_dir_all(&dest).unwrap();
    let mut out = String::new();
    {
        let d = FsDirectory::open(&dest);
        let src = FsDirectory::open(fixtures().join("default/stage2"));
        let w = DirectoryTaxonomyWriter::open(&d).unwrap();
        w.add_category(&FacetLabel::new(["Tag", "tag3"]).unwrap())
            .unwrap();
        w.add_category(&FacetLabel::new(["Other", "x", "y"]).unwrap())
            .unwrap();
        let mut map = MemoryOrdinalMap::new();
        w.add_taxonomy(&src, &mut map).unwrap();
        for (i, m) in map.get_map().unwrap().iter().enumerate() {
            writeln!(out, "map\t{i}\t{m}").unwrap();
        }
        writeln!(out, "size\t{}", w.get_size().unwrap()).unwrap();
        w.close().unwrap();
    }
    assert_text_eq(
        "add_taxonomy map",
        &out,
        &read(&fixtures().join("add_taxonomy/map.tsv")),
    );
    let d = FsDirectory::open(&dest);
    let r = DirectoryTaxonomyReader::open(&d).unwrap();
    // The epoch is the one difference a fresh index cannot avoid: none.
    assert_text_eq(
        "add_taxonomy dest",
        &dump(&r),
        &read(&fixtures().join("add_taxonomy/dest.tsv")),
    );
}

#[test]
fn replace_taxonomy_as_lucene_does() {
    let tmp = TempDir::new("taxo-replace");
    let dest = tmp.path().join("dest");
    copy_dir(&fixtures().join("default/stage1"), &dest);
    let mut out = String::new();
    {
        let d = FsDirectory::open(&dest);
        // `replaceTaxonomy` is `addIndexes`, which write-locks its source: a
        // copy, so the committed fixture is never written.
        let src_path = tmp.path().join("src");
        copy_dir(&fixtures().join("lru_string_3/stage2"), &src_path);
        let src = FsDirectory::open(&src_path);
        let w = DirectoryTaxonomyWriter::open(&d).unwrap();
        w.replace_taxonomy(&src).unwrap();
        writeln!(out, "epoch\t{:x}", w.taxonomy_epoch()).unwrap();
        for p in [["Author", "author1"], ["New", "one"], ["Tag", "tag9"]] {
            write!(
                out,
                "add\t{}",
                w.add_category(&FacetLabel::new(p).unwrap()).unwrap()
            )
            .unwrap();
            for c in p {
                write!(out, "\t{c}").unwrap();
            }
            out.push('\n');
        }
        writeln!(out, "size\t{}", w.get_size().unwrap()).unwrap();
        w.close().unwrap();
    }
    assert_text_eq(
        "replace ops",
        &out,
        &read(&fixtures().join("replace/ops.tsv")),
    );
    let d = FsDirectory::open(&dest);
    let r = DirectoryTaxonomyReader::open(&d).unwrap();
    assert_text_eq(
        "replace dest",
        &dump(&r),
        &read(&fixtures().join("replace/dest.tsv")),
    );
}

/// `GenTaxonomy.config()`.
fn config() -> FacetsConfig {
    let mut c = FacetsConfig::new();
    c.set_hierarchical("Publish Date", true);
    c.set_multi_valued("Tags", true);
    c.set_hierarchical("Cat", true);
    c.set_multi_valued("Cat", true);
    c.set_multi_valued("Year", true);
    c.set_require_dim_count("Year", true);
    c.set_hierarchical("Loc", true);
    c.set_multi_valued("Loc", true);
    c.set_require_dim_count("Loc", true);
    c.set_index_field_name("Loc", "$loc");
    c.set_index_field_name("Price", "$loc");
    c.set_index_field_name("Color", "$color");
    c.set_drill_down_terms_indexing("Color", DrillDownTermsIndexing::None);
    c.set_hierarchical("Size", true);
    c.set_drill_down_terms_indexing("Size", DrillDownTermsIndexing::DimensionAndFullPath);
    c.set_hierarchical("Shape", true);
    c.set_drill_down_terms_indexing("Shape", DrillDownTermsIndexing::AllPathsNoDim);
    c.set_drill_down_terms_indexing("Flat", DrillDownTermsIndexing::FullPathOnly);
    c.set_hierarchical("Flat", true);
    c.set_index_field_name("A1", "$a");
    c.set_index_field_name("B1", "$b");
    c.set_index_field_name("Z1", "$z9");
    c.set_index_field_name("E1", "$€");
    c.set_hierarchical("E1", true);
    c.set_multi_valued("E1", true);
    c.set_index_field_name("Brand", "$ssdv");
    c.set_index_field_name("Region", "$ssdv");
    c.set_hierarchical("Region", true);
    c.set_multi_valued("Kind", true);
    c.set_require_dim_count("Kind", true);
    c
}

/// `GenTaxonomy.docs()`: `F` facet fields, `S` sorted-set facet fields, `I`
/// the stored `id`.
fn docs() -> Vec<Vec<Vec<&'static str>>> {
    vec![
        vec![
            vec!["I", "d0"],
            vec!["F", "Author", "Bob"],
            vec!["F", "Publish Date", "2010", "10", "15"],
            vec!["F", "Tags", "x"],
            vec!["F", "Tags", "y"],
        ],
        vec![
            vec!["F", "Cat", "a", "b", "c"],
            vec!["I", "d1"],
            vec!["F", "Cat", "a", "d"],
            vec!["F", "Year", "2010"],
            vec!["F", "Year", "2012"],
            vec!["F", "Loc", "eu", "fr", "paris"],
            vec!["F", "Price", "cheap"],
        ],
        vec![
            vec!["F", "Color", "red"],
            vec!["F", "Size", "s", "m", "l"],
            vec!["F", "Shape", "round", "small", "tiny"],
            vec!["F", "Flat", "a", "b"],
            vec!["I", "d2"],
        ],
        vec![
            vec!["F", "A1", "v"],
            vec!["F", "B1", "v"],
            vec!["F", "Z1", "v"],
            vec!["F", "E1", "€", "¢"],
            vec!["F", "E1", "€", "£"],
            vec!["F", "Author", "Lisa"],
            vec!["S", "Brand", "acme"],
            vec!["S", "Region", "eu", "de"],
            vec!["S", "Kind", "k1"],
            vec!["S", "Kind", "k2"],
            vec!["I", "d3"],
        ],
        vec![
            vec!["F", "Author", "Bob"],
            vec!["F", "Cat", "a", "b", "c"],
            vec!["F", "Cat", "a", "b"],
            vec!["F", "Loc", "eu", "fr"],
            vec!["F", "sep\u{1F}dim", "a\u{1F}b"],
            vec!["F", "Ünïcødé", "😀"],
        ],
        vec![vec!["I", "only"]],
    ]
}

/// `GenTaxonomy.toDoc(spec)`, split into the facet fields and the rest.
fn to_doc(spec: &[Vec<&str>]) -> lucene_search::Result<(Vec<FacetEntry>, Document)> {
    let mut facets = Vec::new();
    let mut doc = Document::new();
    for f in spec {
        let path = &f[2.min(f.len())..];
        match f[0] {
            "F" => facets.push(FacetField::new(f[1], path)?.into()),
            "S" => facets.push(SortedSetDocValuesFacetField::new(f[1], path)?.into()),
            "I" => doc.add(StringField::new("id", f[1], Store::Yes)),
            _ => unreachable!(),
        }
    }
    Ok((facets, doc))
}

fn describe(i: usize, doc: &Document) -> String {
    let mut sb = String::new();
    for f in doc.fields() {
        let ft = f.field_type();
        let (kind, value) = match ft.doc_values_type() {
            DocValuesType::SortedNumeric => (
                "sndv",
                match f.numeric_value() {
                    Some(Number::Long(v)) => v.to_string(),
                    other => panic!("{other:?}"),
                },
            ),
            DocValuesType::SortedSet => (
                "ssdv",
                String::from_utf8(f.binary_value().unwrap().into_owned()).unwrap(),
            ),
            DocValuesType::None => (
                if ft.stored() {
                    "string_stored"
                } else {
                    "string"
                },
                f.string_value().unwrap().into_owned(),
            ),
            other => panic!("{other:?}"),
        };
        writeln!(sb, "{i}\t{}\t{kind}\t{value}", f.name()).unwrap();
    }
    sb
}

#[test]
fn facets_config_build_matches_lucene() {
    let tmp = TempDir::new("taxo-build");
    let taxo = tmp.path().join("taxo");
    fs::create_dir_all(&taxo).unwrap();
    let config = config();
    let mut out = String::new();
    {
        let d = FsDirectory::open(&taxo);
        let w = DirectoryTaxonomyWriter::open(&d).unwrap();
        for (i, spec) in docs().iter().enumerate() {
            let (facets, doc) = to_doc(spec).unwrap();
            let built = config.build(Some(&w), &facets, doc).unwrap();
            out.push_str(&describe(i, &built));
        }
        w.close().unwrap();
    }
    assert_text_eq(
        "build.tsv",
        &out,
        &read(&fixtures().join("build/build.tsv")),
    );
    let d = FsDirectory::open(&taxo);
    let r = DirectoryTaxonomyReader::open(&d).unwrap();
    assert_text_eq(
        "build taxo",
        &dump(&r),
        &read(&fixtures().join("build/taxo.tsv")),
    );
}

/// `GenTaxonomy.LABELS`.
fn labels() -> Vec<Vec<String>> {
    vec![
        vec![],
        s(&["a"]),
        s(&["a", "b"]),
        s(&["a", "b", "c"]),
        s(&["b"]),
        s(&["ab"]),
        s(&["a", "bc"]),
        s(&["Ünïcødé", "日本"]),
        s(&["\u{FFFF}"]),
        s(&["😀"]),
        s(&["😀z"]),
        s(&["sep\u{1F}dim", "a\u{1F}b"]),
        s(&["Author", "author3"]),
        s(&["Pub Date", "2001", "2", "15"]),
        vec!["long".into(), long_exact()],
    ]
}

fn java_len(s: &str) -> usize {
    s.encode_utf16().count()
}

fn long_or(s: &str) -> &str {
    if java_len(s) > 200 {
        "LONG"
    } else {
        s
    }
}

#[test]
fn facet_label_hashes_and_order_match_lucene() {
    let mut sb = String::new();
    let labels = labels();
    for p in &labels {
        let l = label(p);
        let shown = l.to_string();
        write!(
            sb,
            "label\t{}\t{}\t{}",
            l.hash_code(),
            l.long_hash_code(),
            long_or(&shown)
        )
        .unwrap();
        for c in p {
            write!(sb, "\t{}", long_or(c)).unwrap();
        }
        sb.push('\n');
        for n in 0..=p.len() {
            let sub = l.subpath(n);
            let last = if p.is_empty() {
                "-".to_string()
            } else {
                long_or(sub.last_component().unwrap()).to_string()
            };
            writeln!(
                sb,
                "subpath\t{n}\t{}\t{}\t{}\t{last}",
                sub.len(),
                sub.hash_code(),
                sub.long_hash_code()
            )
            .unwrap();
        }
    }
    for (i, a) in labels.iter().enumerate() {
        for (j, b) in labels.iter().enumerate() {
            let (a, b) = (label(a), label(b));
            let signum = match a.compare_to(&b) {
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Greater => 1,
            };
            writeln!(sb, "compare\t{i}\t{j}\t{signum}\t{}", a == b).unwrap();
        }
    }
    assert_text_eq("labels.tsv", &sb, &read(&fixtures().join("labels.tsv")));
}

/// `(class, message)` of a refusal, Java's class named after this port's
/// variant.
fn outcome<T>(r: lucene_search::Result<T>) -> (String, String) {
    match r {
        Ok(_) => ("none".into(), String::new()),
        Err(e) => {
            let class = match &e {
                Error::IllegalArgument(_) => "IllegalArgument",
                Error::IllegalState(_) => "IllegalState",
                Error::Unsupported(_) => "Unsupported",
                Error::AlreadyClosed(_) => "AlreadyClosed",
                other => panic!("unexpected {other:?}"),
            };
            (class.into(), e.to_string())
        }
    }
}

fn java_class(class: &str) -> &'static str {
    match class {
        "none" => "none",
        "java.lang.IllegalArgumentException" | "java.lang.IndexOutOfBoundsException" => {
            "IllegalArgument"
        }
        "java.lang.IllegalStateException" => "IllegalState",
        "java.lang.UnsupportedOperationException" => "Unsupported",
        "org.apache.lucene.store.AlreadyClosedException" => "AlreadyClosed",
        other => panic!("{other}"),
    }
}

#[test]
fn refusals_match_lucene() {
    let mut got: Vec<(String, (String, String))> = Vec::new();
    let mut put = |name: &str, o: (String, String)| got.push((name.to_string(), o));
    let too_long = "x".repeat(MAX - 4);
    put(
        "label_too_long",
        outcome(FacetLabel::new(["long", too_long.as_str()])),
    );
    put(
        "label_exact_max",
        outcome(FacetLabel::new(vec!["long".to_string(), long_exact()])),
    );
    put(
        "label_too_long_one_component",
        outcome(FacetLabel::new(["y".repeat(8192)])),
    );
    put(
        "label_empty_component",
        outcome(FacetLabel::new(["a", "", "b"])),
    );
    put(
        "label_empty_dim",
        outcome(FacetLabel::from_dim_path("", &["b"])),
    );
    put(
        "last_component_empty",
        outcome(FacetLabel::root().last_component()),
    );
    put(
        "facet_field_empty_dim",
        outcome(FacetField::new("", &["a"])),
    );
    put(
        "facet_field_empty_label",
        outcome(FacetField::new("a", &["b", ""])),
    );
    put(
        "facet_field_no_path",
        outcome(FacetField::new::<&str>("a", &[])),
    );
    let config = config();
    let d = ByteBuffersDirectory::new();
    let w = DirectoryTaxonomyWriter::open(&d).unwrap();
    let build = |taxo: Option<&dyn TaxonomyWriter>, spec: Vec<Vec<&str>>| {
        let (facets, doc) = to_doc(&spec)?;
        config.build(taxo, &facets, doc)
    };
    put(
        "build_not_hierarchical",
        outcome(build(Some(&w), vec![vec!["F", "Author", "a", "b"]])),
    );
    put(
        "build_not_multivalued",
        outcome(build(
            Some(&w),
            vec![vec!["F", "Author", "a"], vec!["F", "Author", "b"]],
        )),
    );
    put(
        "build_not_multivalued_mixed",
        outcome(build(
            Some(&w),
            vec![vec!["S", "Brand", "a"], vec!["F", "Brand", "b"]],
        )),
    );
    put(
        "build_ssdv_not_hierarchical",
        outcome(build(Some(&w), vec![vec!["S", "Brand", "a", "b"]])),
    );
    put(
        "build_no_taxo_writer",
        outcome(build(None, vec![vec!["F", "Author", "a"]])),
    );
    put(
        "build_ssdv_only_no_taxo_writer",
        outcome(build(None, vec![vec!["S", "Brand", "a"]])),
    );
    put(
        "writer_parent_out_of_range",
        outcome(w.get_parent(w.get_size().unwrap())),
    );
    put("writer_parent_negative", outcome(w.get_parent(-1)));
    w.commit().unwrap();
    {
        let r = DirectoryTaxonomyReader::open(&d).unwrap();
        put(
            "reader_path_out_of_range",
            outcome(r.get_path(r.get_size().unwrap())),
        );
        put("reader_path_negative", outcome(r.get_path(-1)));
        put(
            "reader_bulk_path_out_of_range",
            outcome(r.get_bulk_path(&[0, 99])),
        );
    }
    w.close().unwrap();
    put(
        "writer_closed",
        outcome(w.add_category(&FacetLabel::new(["a"]).unwrap())),
    );
    put("writer_closed_size", outcome(w.get_size()));

    let expected = read(&fixtures().join("errors.tsv"));
    let expected: Vec<&str> = expected.lines().collect();
    assert_eq!(got.len(), expected.len());
    for ((name, (class, message)), line) in got.iter().zip(expected) {
        let cols: Vec<&str> = line.splitn(3, '\t').collect();
        assert_eq!(name, cols[0]);
        assert_eq!(class, java_class(cols[1]), "{name}");
        if name.starts_with("reader_") {
            // Java quotes its reader's `toString()`, which names the Lucene
            // version that wrote each segment; the rest must match.
            let (head, tail) = cols[2].split_once("StandardDirectoryReader(").unwrap();
            let tail = &tail[tail.find("). ").unwrap() + 1..];
            assert!(message.starts_with(head), "{name}: {message}");
            assert!(message.ends_with(tail), "{name}: {message}");
        } else {
            assert_eq!(message, cols[2], "{name}");
        }
    }
}
