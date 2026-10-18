//! M14 T14.3 `misc`, differentially against Lucene 10.5.0:
//! `fixtures/src/GenMiscRearrange.java`'s `misc/rearrange.tsv`.
//!
//! `IndexRearranger` with `BinaryDocValueSelector`s rearranges
//! `misc/rearrange_src` (three segments, three deleted documents); every
//! output must have the same segments in the same order, each with the same
//! size and live documents, or fail with the same message.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::misc::index::{BinaryDocValueSelector, DocumentSelector, IndexRearranger};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn data() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/misc")
}

/// A private copy of the committed fixture index `name` under `tmp`:
/// `addIndexes` write-locks its sources, and a test must never write into
/// `fixtures/`.
fn source(name: &str, tmp: &TempDir) -> FsDirectory {
    let to = tmp.path().join(name);
    std::fs::create_dir_all(&to).unwrap();
    for entry in std::fs::read_dir(data().join(name)).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() != "write.lock" {
            std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
        }
    }
    FsDirectory::open(to)
}

fn id(n: usize) -> String {
    format!("k{n:03}")
}

fn keys(from: usize, to: usize, step: usize) -> HashSet<Vec<u8>> {
    (from..to)
        .step_by(step)
        .map(|n| id(n).into_bytes())
        .collect()
}

fn sel(keys: HashSet<Vec<u8>>) -> Box<dyn DocumentSelector> {
    Box::new(BinaryDocValueSelector::new("key", keys))
}

/// A rearrangement to run: its name, segment selectors and deletes selector.
type Case = (
    &'static str,
    Vec<Box<dyn DocumentSelector>>,
    Option<Box<dyn DocumentSelector>>,
);

/// What Lucene made: `Ok(segments as (max_doc, ids))` or its failure message.
type Outcome = Result<Vec<(i32, Vec<String>)>, String>;

fn expected() -> BTreeMap<String, Outcome> {
    let tsv = std::fs::read_to_string(data().join("rearrange.tsv")).unwrap();
    let mut out: BTreeMap<String, Outcome> = BTreeMap::new();
    for line in tsv.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        match c[0] {
            "R" => {
                let ids = c.get(4).copied().unwrap_or("");
                let ids = ids.split(',').filter(|s| !s.is_empty()).map(String::from);
                let seg = (c[3].parse().unwrap(), ids.collect());
                match out
                    .entry(c[1].to_string())
                    .or_insert_with(|| Ok(Vec::new()))
                {
                    Ok(v) => {
                        assert_eq!(v.len(), c[2].parse::<usize>().unwrap());
                        v.push(seg);
                    }
                    Err(_) => panic!("rows after a failure"),
                }
            }
            "E" => {
                out.insert(c[1].to_string(), Err(c[2].to_string()));
            }
            other => panic!("unknown row {other}"),
        }
    }
    out
}

fn read(dir: &FsDirectory) -> Vec<(i32, Vec<String>)> {
    let reader = DirectoryReader::open(dir).unwrap();
    reader
        .segment_readers()
        .iter()
        .map(|s| {
            let field = s.field_infos().field_by_name("id").unwrap().number;
            let mut ids = Vec::new();
            for d in 0..s.max_doc {
                if s.live_docs()
                    .is_some_and(|l| !l.get(usize::try_from(d).unwrap()))
                {
                    continue;
                }
                let doc = s.stored_document(d).unwrap().unwrap();
                let f = doc.fields.iter().find(|f| f.field_number == field).unwrap();
                match &f.value {
                    lucene_codecs::stored_fields::FieldValue::String(v) => ids.push(v.clone()),
                    other => panic!("id is not a string: {other:?}"),
                }
            }
            (s.max_doc, ids)
        })
        .collect()
}

fn boxed(v: Vec<BinaryDocValueSelector>) -> Vec<Box<dyn DocumentSelector>> {
    v.into_iter()
        .map(|s| Box::new(s) as Box<dyn DocumentSelector>)
        .collect()
}

#[test]
fn index_rearranger_matches_lucene() {
    let want = expected();
    let src = TempDir::new("rearrange-src");
    let input = source("rearrange_src", &src);
    let delete_from_index = || {
        Some(Box::new(
            BinaryDocValueSelector::create_delete_selector_from_index("key", &input).unwrap(),
        ) as Box<dyn DocumentSelector>)
    };
    let cases: Vec<Case> = vec![
        (
            "geometry",
            boxed(BinaryDocValueSelector::create_live_selectors_from_index("key", &input).unwrap()),
            delete_from_index(),
        ),
        (
            "even_odd",
            vec![sel(keys(0, 18, 2)), sel(keys(1, 18, 2))],
            None,
        ),
        (
            "even_odd_deletes",
            vec![sel(keys(0, 18, 2)), sel(keys(1, 18, 2))],
            delete_from_index(),
        ),
        (
            "three_ways",
            vec![
                sel(keys(12, 18, 1)),
                sel(keys(0, 5, 1)),
                sel(keys(5, 12, 1)),
            ],
            Some(sel(keys(3, 15, 4))),
        ),
        (
            "overlap",
            vec![sel(keys(0, 10, 1)), sel(keys(5, 13, 1))],
            None,
        ),
        (
            "whole_segment_deleted",
            vec![sel(keys(0, 7, 1)), sel(keys(7, 18, 1))],
            Some(sel(keys(0, 7, 1))),
        ),
    ];
    assert_eq!(cases.len(), want.len());
    for (name, selectors, deletes) in cases {
        let tmp = TempDir::new("rearrange");
        let output = FsDirectory::open(tmp.path().join("out"));
        let got = IndexRearranger::new(&input, &output, selectors, deletes)
            .execute()
            .map(|()| read(&output))
            .map_err(|e| e.to_string());
        match (&want[name], got) {
            (Ok(w), Ok(g)) => assert_eq!(&g, w, "{name}"),
            (Err(w), Err(g)) => {
                // Java names the selector by object identity, the port by position.
                let tail = w.trim_start_matches("Document selector # has matched");
                assert!(g.ends_with(tail), "{name}: {g} vs {w}");
            }
            (w, g) => panic!("{name}: Lucene {w:?}, port {g:?}"),
        }
    }
}

#[test]
fn binary_doc_value_selector_edges() {
    let src = TempDir::new("rearrange-src");
    let input = source("rearrange_src", &src);
    // A field the segment has no binary doc values for selects nothing.
    let reader = DirectoryReader::open(&input).unwrap();
    let none = BinaryDocValueSelector::new("absent", keys(0, 18, 1));
    for s in reader.segment_readers() {
        let bits = none.get_filtered_docs(s).unwrap();
        assert_eq!(bits.len(), usize::try_from(s.max_doc).unwrap());
        assert!(bits.iter().all(|b| !b));
    }
    // A document without a key is Java's AssertionError.
    assert!(BinaryDocValueSelector::create_live_selectors_from_index("absent", &input).is_err());
    // An index with no deletions has nothing to select for deletion.
    let tmp = TempDir::new("rearrange-nodel");
    let copy = FsDirectory::open(tmp.path().join("out"));
    IndexRearranger::new(&input, &copy, vec![sel(keys(0, 18, 1))], None)
        .execute()
        .unwrap();
    let selector = BinaryDocValueSelector::create_delete_selector_from_index("key", &copy).unwrap();
    let copy_reader = DirectoryReader::open(&copy).unwrap();
    for s in copy_reader.segment_readers() {
        assert!(selector.get_filtered_docs(s).unwrap().iter().all(|b| !b));
    }
}
