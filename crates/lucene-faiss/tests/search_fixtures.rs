#![allow(clippy::arithmetic_side_effects)]
//! Searches over Faiss's own files return what Faiss itself returned for the
//! same queries (`scripts/gen-faiss-fixtures.py`), with and without a filter.

use lucene_faiss::{read_index, SearchParams};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/faiss")
}

fn floats(path: PathBuf) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

struct Case {
    name: String,
    factory: String,
    param: String,
}

fn cases() -> Vec<Case> {
    std::fs::read_to_string(dir().join("manifest.tsv"))
        .unwrap()
        .lines()
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            Case {
                name: f[0].into(),
                factory: f[1].into(),
                param: f.get(6).copied().unwrap_or("").into(),
            }
        })
        .collect()
}

fn params(param: &str) -> SearchParams {
    let mut p = SearchParams::default();
    if let Some(v) = param.strip_prefix("efSearch=") {
        p.ef_search = Some(v.parse().unwrap());
    }
    if let Some(v) = param.strip_prefix("nprobe=") {
        p.nprobe = Some(v.parse().unwrap());
    }
    p
}

#[test]
fn searches_match_faiss_ranks_and_distances() {
    const D: usize = 24;
    for case in cases() {
        let index =
            read_index(&std::fs::read(dir().join(format!("{}.idx", case.name))).unwrap()).unwrap();
        let queries = floats(dir().join(format!("{}.queries", case.name)));
        let params = params(&case.param);
        let mut checked = 0;
        for line in std::fs::read_to_string(dir().join(format!("{}.expected", case.name)))
            .unwrap()
            .lines()
        {
            let f: Vec<&str> = line.split('\t').collect();
            let qi: usize = f[1].parse().unwrap();
            let q = &queries[qi * D..(qi + 1) * D];
            let even = |id: i64| id % 2 == 0;
            let filter: Option<&dyn Fn(i64) -> bool> = (f[0] == "filtered").then_some(&even);
            let got = index.search(q, 10, &params, filter).unwrap();
            let want: Vec<(i64, f32)> = f[2..]
                .iter()
                .map(|c| {
                    let (id, bits) = c.split_once(':').unwrap();
                    (
                        id.parse().unwrap(),
                        f32::from_bits(u32::from_str_radix(bits, 16).unwrap()),
                    )
                })
                .collect();
            let ids: Vec<i64> = got.iter().map(|g| g.0).collect();
            let want_ids: Vec<i64> = want.iter().map(|w| w.0).collect();
            assert_eq!(ids, want_ids, "{} {} query {qi}", case.name, f[0]);
            for (g, w) in got.iter().zip(&want) {
                let tol = 1e-4 * w.1.abs().max(1.0);
                assert!(
                    (g.1 - w.1).abs() <= tol,
                    "{} q{qi}: {} vs {}",
                    case.name,
                    g.1,
                    w.1
                );
            }
            checked += 1;
        }
        assert!(checked >= 20, "{} checked {checked}", case.name);
        assert!(!case.factory.is_empty());
    }
}
