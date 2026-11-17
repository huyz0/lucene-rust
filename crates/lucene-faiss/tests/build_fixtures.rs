#![allow(clippy::arithmetic_side_effects)]
//! Indexes this crate builds are valid Faiss indexes with Faiss-grade recall:
//! built over the fixture vectors, searched like the Faiss-built ones, and
//! written back to bytes that read as the same index.

use lucene_faiss::{build_index, read_index, write_index, Index, Metric, SearchParams};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/faiss")
}

fn floats(name: &str) -> Vec<f32> {
    std::fs::read(dir().join(name))
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

const D: usize = 24;

fn recall(index: &Index, flat: &Index, queries: &[f32], params: &SearchParams) -> f64 {
    let mut hit = 0;
    let mut total = 0;
    for q in queries.chunks_exact(D) {
        let want: Vec<i64> = flat
            .search(q, 10, &SearchParams::default(), None)
            .unwrap()
            .iter()
            .map(|r| r.0)
            .collect();
        let got: Vec<i64> = index
            .search(q, 10, params, None)
            .unwrap()
            .iter()
            .map(|r| r.0)
            .collect();
        hit += got.iter().filter(|g| want.contains(g)).count();
        total += want.len();
    }
    hit as f64 / total as f64
}

fn ids(n: usize) -> Vec<i64> {
    (0..n as i64).map(|i| i * 3 + 7).collect()
}

#[test]
fn built_indexes_have_high_recall_and_round_trip() {
    let x = floats("hnsw_idmap_l2.vectors");
    let queries = floats("hnsw_idmap_l2.queries");
    let n = x.len() / D;
    for metric in [Metric::L2, Metric::InnerProduct] {
        let flat = build_index("IDMap,Flat", "", metric, D, &x, &ids(n)).unwrap();
        let cases: [(&str, &str, SearchParams, f64); 3] = [
            (
                "IDMap,HNSW8",
                "efConstruction=40",
                SearchParams {
                    ef_search: Some(64),
                    nprobe: None,
                },
                0.95,
            ),
            (
                "IDMap,IVF16,Flat",
                "",
                SearchParams {
                    ef_search: None,
                    nprobe: Some(8),
                },
                0.8,
            ),
            ("IDMap,Flat", "", SearchParams::default(), 1.0),
        ];
        for (desc, params, search, min) in cases {
            let index = build_index(desc, params, metric, D, &x, &ids(n)).unwrap();
            let r = recall(&index, &flat, &queries, &search);
            assert!(r >= min, "{desc} {metric:?}: recall {r}");
            let bytes = write_index(&index);
            assert_eq!(read_index(&bytes).unwrap(), index, "{desc}");
        }
    }
}

#[test]
fn a_filter_never_returns_a_rejected_id() {
    let x = floats("hnsw_idmap_l2.vectors");
    let queries = floats("hnsw_idmap_l2.queries");
    let n = x.len() / D;
    let index = build_index(
        "IDMap,HNSW8",
        "efConstruction=40",
        Metric::L2,
        D,
        &x,
        &ids(n),
    )
    .unwrap();
    let even = |id: i64| id % 2 == 0;
    for q in queries.chunks_exact(D) {
        let got = index
            .search(
                q,
                10,
                &SearchParams {
                    ef_search: Some(64),
                    nprobe: None,
                },
                Some(&even),
            )
            .unwrap();
        assert!(!got.is_empty() && got.iter().all(|r| r.0 % 2 == 0));
    }
}

#[test]
fn unsupported_descriptions_are_refused_with_the_supported_list() {
    for bad in [
        "PQ8",
        "IDMap,IVF16,PQ4",
        "IDMap2,Flat",
        "IDMap,HNSW16,SQ4",
        "OPQ16,IVF4,Flat",
        "",
        "IVF0,Flat",
        "IDMap,HNSWx",
    ] {
        let err = build_index(bad, "", Metric::L2, 4, &[], &[])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("unsupported faiss index description"),
            "{bad}: {err}"
        );
    }
    let err = build_index("HNSW8", "", Metric::L2, 2, &[0.0, 1.0], &[5])
        .unwrap_err()
        .to_string();
    assert!(err.contains("IDMap"), "{err}");
    assert!(build_index("IDMap,Flat", "efConstruction=x", Metric::L2, 2, &[], &[]).is_err());
    assert!(build_index("IDMap,Flat", "bogus=1", Metric::L2, 2, &[], &[]).is_err());
}
