//! M14 T14.3 `misc`, differentially against Lucene 10.5.0:
//! `fixtures/src/GenMiscIndexSplitter.java`'s `misc/index_splitter.tsv`.
//!
//! `IndexSplitter` lists, splits and removes segments of
//! `misc/index_splitter_src` (four segments, one with a deletion); Lucene's
//! listing, the split index's segments, counter and minimum version, and what
//! a remove leaves behind must all be reproduced.
#![allow(clippy::arithmetic_side_effects)]

use std::path::{Path, PathBuf};

use lucene_index::segment_infos;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::misc::index::IndexSplitter;
use lucene_store::directory::Directory;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/misc")
}

fn tsv() -> Vec<Vec<String>> {
    std::fs::read_to_string(data().join("index_splitter.tsv"))
        .unwrap()
        .lines()
        .map(|l| l.split('\t').map(String::from).collect())
        .collect()
}

/// `(segment name, max_doc, live ids)` in commit order.
fn read(dir: &FsDirectory) -> Vec<(String, i32, String)> {
    let reader = DirectoryReader::open(dir).unwrap();
    let infos = &reader.segment_infos;
    reader
        .segment_readers()
        .iter()
        .zip(&infos.segments)
        .map(|(s, sci)| {
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
            (sci.segment_name.clone(), s.max_doc, ids.join(","))
        })
        .collect()
}

fn rows(tag: &str) -> Vec<Vec<String>> {
    tsv().into_iter().filter(|r| r[0] == tag).collect()
}

fn expected_segments(tag: &str) -> Vec<(String, i32, String)> {
    rows(tag)
        .into_iter()
        .enumerate()
        .map(|(i, r)| {
            assert_eq!(r[1].parse::<usize>().unwrap(), i);
            (r[2].clone(), r[3].parse().unwrap(), r[4].clone())
        })
        .collect()
}

#[test]
fn index_splitter_lists_splits_and_removes_as_lucene_does() {
    let src_dir = data().join("index_splitter_src");
    let src = FsDirectory::open(&src_dir);
    let splitter = IndexSplitter::new(&src).unwrap();

    // listSegments
    let want: Vec<String> = rows("L")
        .iter()
        .map(|r| format!("{} {}", r[1], r[2]))
        .collect();
    assert_eq!(splitter.list_segments().unwrap(), want);

    // split
    let tmp = TempDir::new("index-splitter");
    std::fs::create_dir_all(tmp.path().join("dest")).unwrap();
    let dest = FsDirectory::open(tmp.path().join("dest"));
    splitter.split(&dest, &["_2", "_0"]).unwrap();
    assert_eq!(read(&dest), expected_segments("S"));
    let infos = segment_infos::read_latest(&dest).unwrap();
    assert_eq!(infos.counter, rows("C")[0][1].parse::<i64>().unwrap());
    let min = infos.min_segment_lucene_version.unwrap();
    assert_eq!(
        format!("{}.{}.{}", min.major, min.minor, min.bugfix),
        rows("M")[0][1]
    );
    assert!(splitter.split(&dest, &["_9"]).is_err());

    // remove, on a copy
    let copy_dir = tmp.path().join("copy");
    std::fs::create_dir_all(&copy_dir).unwrap();
    for entry in std::fs::read_dir(&src_dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() != "write.lock" {
            std::fs::copy(entry.path(), copy_dir.join(entry.file_name())).unwrap();
        }
    }
    let copy = FsDirectory::open(&copy_dir);
    let mut removing = IndexSplitter::new(&copy).unwrap();
    removing.remove(&["_1", "_3", "none"]).unwrap();
    assert_eq!(read(&copy), expected_segments("R"));
    let mut kept: Vec<String> = copy
        .list_all()
        .unwrap()
        .into_iter()
        .filter(|f| src_dir.join(f).exists())
        .collect();
    kept.sort();
    let mut want: Vec<String> = rows("F").iter().map(|r| r[1].clone()).collect();
    want.sort();
    assert_eq!(kept, want, "a removed segment's files stay");
}
