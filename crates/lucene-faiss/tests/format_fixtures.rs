#![allow(clippy::arithmetic_side_effects)]
//! Faiss's own files (python faiss 1.16 `write_index`, see
//! `scripts/gen-faiss-fixtures.py`) read and written back byte for byte.

use lucene_faiss::format::{read_index, write_index, Index};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/faiss")
}

fn cases() -> Vec<String> {
    std::fs::read_to_string(dir().join("manifest.tsv"))
        .unwrap()
        .lines()
        .map(|l| l.split('\t').next().unwrap().to_string())
        .collect()
}

#[test]
fn every_fixture_reads_and_writes_back_identically() {
    for name in cases() {
        let bytes = std::fs::read(dir().join(format!("{name}.idx"))).unwrap();
        let index = read_index(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(index.ntotal(), 1500, "{name}");
        assert_eq!(index.d(), 24, "{name}");
        assert_eq!(write_index(&index), bytes, "{name} does not round-trip");
    }
}

#[test]
fn hnsw_graph_matches_the_factory_default_levels() {
    let bytes = std::fs::read(dir().join("hnsw_plain_l2.idx")).unwrap();
    let Index::Hnsw { hnsw, storage } = read_index(&bytes).unwrap() else {
        panic!("not an hnsw index");
    };
    assert_eq!(storage.len(), 1500);
    assert_eq!(hnsw.ef_construction, 40);
    assert_eq!(hnsw.cum_nneighbor_per_level[..3], [0, 16, 24]);
}
