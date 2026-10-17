//! Writes taxonomy indexes for `VerifyTaxonomy` (M13 T13.4a), each in its
//! own directory beside a `dump.tsv` of what this port's reader reads back:
//!
//! - `small-cache`: an `LRU_STRING` cache of 3, so nearly every add evicts
//!   and the writer looks categories up in (and refreshes) its
//!   near-real-time reader -- one segment per refresh, merged by
//!   `LogByteSizeMergePolicy`; two commits, user commit data on the second.
//! - `default-cache`: the default cache, one commit, then a second session
//!   appending to it.
//! - `replaced`: `replaceTaxonomy` from `small-cache`, then more categories.
//!
//! Usage: `write_taxonomy_fixture <output-dir>`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::fmt::Write as _;
use std::path::Path;

use lucene_index::deletion_policy::OpenMode;
use lucene_search::facets::taxonomy::writercache::{LruTaxonomyWriterCache, LruType};
use lucene_search::facets::taxonomy::{
    DirectoryTaxonomyReader, DirectoryTaxonomyWriter, FacetLabel, TaxonomyReader, TaxonomyWriter,
};
use lucene_store::FsDirectory;

/// The categories of one session.
fn categories(session: usize) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut add = |c: &[&str]| out.push(c.iter().map(|s| s.to_string()).collect());
    match session {
        0 => {
            for i in 0..12 {
                add(&["Author", &format!("author{i}")]);
            }
            for y in 2000..2003 {
                for m in 1..=3 {
                    add(&["Pub Date", &y.to_string(), &m.to_string(), "1"]);
                    add(&["Pub Date", &y.to_string(), &m.to_string(), "15"]);
                }
            }
            add(&["Ünïcødé", "日本", "語"]);
            add(&["Ünïcødé", "😀z"]);
            add(&["Ünïcødé", "\u{ffff}"]);
            add(&["sep\u{1f}dim", "a\u{1f}b", "c\u{1e}"]);
            let long = "x".repeat(FacetLabel::MAX_CATEGORY_PATH_LENGTH - 5);
            add(&["long", &long]);
            add(&["Author", "author3"]);
        }
        1 => {
            for i in (0..12).rev().step_by(2) {
                add(&["Author", &format!("author{i}")]);
            }
            add(&["Pub Date", "2001", "2", "16"]);
            add(&["Pub Date", "2004", "1", "1"]);
            for i in 0..15 {
                add(&["Tag", &format!("tag{}", i * 7 % 15)]);
            }
            let deep: Vec<String> = std::iter::once("deep".to_string())
                .chain((0..30).map(|i| format!("d{i}")))
                .collect();
            let deep: Vec<&str> = deep.iter().map(String::as_str).collect();
            add(&deep);
        }
        _ => {
            add(&["Tag", "tag1"]);
            add(&["Replaced", "new"]);
            add(&["Author", "author12"]);
        }
    }
    out
}

fn add_all(w: &DirectoryTaxonomyWriter<'_>, session: usize) {
    for c in categories(session) {
        w.add_category(&FacetLabel::new(c).unwrap()).unwrap();
    }
}

/// What `VerifyTaxonomy.dump` prints for the same taxonomy.
fn dump(r: &DirectoryTaxonomyReader<'_>) -> String {
    let mut s = String::new();
    let size = r.get_size().unwrap();
    writeln!(s, "size\t{size}").unwrap();
    let mut user = r.commit_user_data().unwrap();
    user.sort();
    for (k, v) in user {
        writeln!(s, "user\t{k}\t{v}").unwrap();
    }
    let arrays = r.parallel_taxonomy_arrays().unwrap();
    for o in 0..size {
        write!(
            s,
            "ord\t{o}\t{}\t{}\t{}",
            arrays.parents().get(o),
            arrays.children().get(o),
            arrays.siblings().get(o)
        )
        .unwrap();
        for c in r.get_path(o).unwrap().path() {
            write!(s, "\t{c}").unwrap();
        }
        s.push('\n');
    }
    s
}

fn finish(dir: &Path) {
    let fs = FsDirectory::open(dir);
    let r = DirectoryTaxonomyReader::open(&fs).unwrap();
    std::fs::write(dir.join("dump.tsv"), dump(&r)).unwrap();
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_taxonomy_fixture <output-dir>");
    let out = Path::new(&out);

    let small = out.join("small-cache");
    std::fs::create_dir_all(&small).unwrap();
    {
        let fs = FsDirectory::open(&small);
        let cache = LruTaxonomyWriterCache::with_type(3, LruType::LruString);
        let w =
            DirectoryTaxonomyWriter::with_cache(&fs, OpenMode::Create, Box::new(cache)).unwrap();
        add_all(&w, 0);
        w.commit().unwrap();
        add_all(&w, 1);
        w.set_live_commit_data(vec![("app".to_string(), "rust".to_string())])
            .unwrap();
        w.close().unwrap();
    }
    finish(&small);

    let default = out.join("default-cache");
    std::fs::create_dir_all(&default).unwrap();
    {
        let fs = FsDirectory::open(&default);
        let w = DirectoryTaxonomyWriter::open(&fs).unwrap();
        add_all(&w, 0);
        w.close().unwrap();
        let w = DirectoryTaxonomyWriter::with_mode(&fs, OpenMode::Append).unwrap();
        add_all(&w, 1);
        w.close().unwrap();
    }
    finish(&default);

    let replaced = out.join("replaced");
    std::fs::create_dir_all(&replaced).unwrap();
    {
        let fs = FsDirectory::open(&replaced);
        let w = DirectoryTaxonomyWriter::open(&fs).unwrap();
        add_all(&w, 2);
        w.commit().unwrap();
        let src = FsDirectory::open(&small);
        w.replace_taxonomy(&src).unwrap();
        add_all(&w, 2);
        w.close().unwrap();
    }
    finish(&replaced);
}
