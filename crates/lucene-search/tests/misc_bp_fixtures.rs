//! M14 T14.3 `misc`, differentially against Lucene 10.5.0:
//! `fixtures/src/GenMiscBp.java`'s `misc/bp.tsv`.
//!
//! `BPIndexReorderer.computeDocMap` over a flat index and over an index of
//! parent blocks, in nine configurations: the new document order must be
//! Lucene's, document for document.
#![allow(clippy::arithmetic_side_effects)]

use std::path::{Path, PathBuf};

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::misc::index::BpIndexReorderer;
use lucene_store::FsDirectory;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/misc")
}

#[test]
fn bp_doc_maps_equal_lucenes() {
    let tsv = std::fs::read_to_string(data().join("bp.tsv")).unwrap();
    let mut checked = 0;
    for line in tsv.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        assert_eq!(c[0], "D");
        let (config, index) = (c[1], c[2]);
        let dir = FsDirectory::open(data().join(format!("bp_{index}")));
        let reader = DirectoryReader::open(&dir).unwrap();
        assert_eq!(reader.segment_readers().len(), 1);
        let leaf = &reader.segment_readers()[0];

        let mut r = BpIndexReorderer::new();
        r.set_ram_budget_mb(256.0);
        r.set_min_doc_freq(c[3].parse().unwrap()).unwrap();
        r.set_min_partition_size(c[4].parse().unwrap()).unwrap();
        r.set_max_doc_freq(c[5].parse().unwrap()).unwrap();
        r.set_max_iters(c[6].parse().unwrap()).unwrap();
        if config != "all_fields" {
            r.set_fields(Some(vec!["body".to_string()]));
        }
        let want: Vec<i32> = c[8].split(' ').map(|v| v.parse().unwrap()).collect();
        assert_eq!(want.len(), c[7].parse::<usize>().unwrap());

        let map = r.compute_doc_map(leaf).unwrap();
        let got: Vec<i32> = (0..map.size()).map(|d| map.new_to_old(d)).collect();
        assert_eq!(got, want, "{config}");
        checked += 1;
    }
    assert_eq!(checked, 9);
}
