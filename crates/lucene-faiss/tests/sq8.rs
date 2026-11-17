#![allow(clippy::arithmetic_side_effects)]
//! `SQ8` storage: built indexes, their files, and refusals.

use lucene_faiss::{build_index, read_index, write_index, Index, Metric, SearchParams};
use std::path::PathBuf;

fn data(n: usize, d: usize) -> Vec<f32> {
    let mut s = 4242u64;
    (0..n * d)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/data/faiss")
            .join(format!("{name}.idx")),
    )
    .unwrap()
}

#[test]
fn built_indexes_find_their_own_vectors_and_round_trip() {
    let (n, d) = (600, 13);
    let v = data(n, d);
    let ids: Vec<i64> = (0..n as i64).map(|i| i * 2).collect();
    for metric in [Metric::L2, Metric::InnerProduct] {
        for desc in ["IDMap,SQ8", "IDMap,HNSW8,SQ8", "IDMap,IVF8,SQ8"] {
            let idx = build_index(
                desc,
                "efConstruction=40,efSearch=64,nprobe=8",
                metric,
                d,
                &v,
                &ids,
            )
            .unwrap();
            let bytes = write_index(&idx);
            let back = read_index(&bytes).unwrap();
            assert_eq!(back, idx, "{desc}");
            assert_eq!(write_index(&back), bytes, "{desc}");
            let mut found = 0;
            for i in (0..n).step_by(30) {
                let hits = back
                    .search(&v[i * d..(i + 1) * d], 5, &SearchParams::default(), None)
                    .unwrap();
                let top = hits.iter().map(|h| h.0).collect::<Vec<_>>();
                found += usize::from(top.contains(&(i as i64 * 2)));
            }
            assert!(
                found >= 19,
                "{desc} {metric:?}: only {found} of 20 found themselves"
            );
        }
    }
}

#[test]
fn constant_dimensions_and_empty_builds() {
    let (n, d) = (50, 9);
    let mut v = data(n, d);
    for row in v.chunks_exact_mut(d) {
        row[3] = 0.25; // no range in this dimension
    }
    let ids: Vec<i64> = (0..n as i64).collect();
    let idx = build_index("IDMap,SQ8", "", Metric::L2, d, &v, &ids).unwrap();
    let hits = idx
        .search(&v[..d], 1, &SearchParams::default(), None)
        .unwrap();
    assert_eq!(hits[0].0, 0);
    for desc in ["IDMap,SQ8", "IDMap,IVF4,SQ8", "IDMap,HNSW8,SQ8"] {
        let idx = build_index(desc, "", Metric::L2, d, &[], &[]).unwrap();
        assert_eq!(idx.ntotal(), 0);
        let back = read_index(&write_index(&idx)).unwrap();
        assert!(back
            .search(&[0.0; 9], 3, &SearchParams::default(), None)
            .unwrap()
            .is_empty());
    }
}

#[test]
fn filters_apply_and_residual_free_lists_search() {
    let (n, d) = (300, 16);
    let v = data(n, d);
    let ids: Vec<i64> = (0..n as i64).collect();
    let even = |id: i64| id % 2 == 0;
    for desc in ["IDMap,SQ8", "IDMap,HNSW8,SQ8", "IDMap,IVF4,SQ8"] {
        let idx = build_index(desc, "nprobe=4", Metric::L2, d, &v, &ids).unwrap();
        let hits = idx
            .search(&v[d..2 * d], 10, &SearchParams::default(), Some(&even))
            .unwrap();
        assert!(hits.iter().all(|h| h.0 % 2 == 0), "{desc}");
    }
    // Codes that are not residuals (a file Faiss wrote with by_residual off).
    let Index::IdMap { sub, ids: map } =
        build_index("IDMap,IVF4,SQ8", "nprobe=4", Metric::L2, d, &v, &ids).unwrap()
    else {
        unreachable!()
    };
    let Index::IvfSq(mut ivf) = *sub else {
        unreachable!()
    };
    ivf.by_residual = false;
    let idx = Index::IdMap {
        sub: Box::new(Index::IvfSq(ivf)),
        ids: map,
    };
    let back = read_index(&write_index(&idx)).unwrap();
    assert_eq!(back, idx);
    assert_eq!(
        back.search(&v[..d], 3, &SearchParams::default(), None)
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn other_quantizers_and_factory_strings_are_refused() {
    let v = data(10, 4);
    let ids: Vec<i64> = (0..10).collect();
    for desc in [
        "IDMap,SQ4",
        "IDMap,IVF4,SQfp16",
        "SQ8,HNSW8",
        "IDMap,PQ4",
        "HNSW8,SQ8",
    ] {
        assert!(
            build_index(desc, "", Metric::L2, 4, &v, &ids).is_err(),
            "{desc}"
        );
    }
    // A graph without an id map is refused whatever its storage.
    let e = build_index("HNSW8,SQ8", "", Metric::L2, 4, &v, &ids)
        .unwrap_err()
        .to_string();
    assert!(e.contains("IDMap"), "{e}");
    // A scalar quantizer of another type in a real file: refused by name.
    let mut b = fixture("sq8_idmap_l2");
    let qtype_at = 74;
    b[qtype_at] = 2;
    let e = read_index(&b).unwrap_err().to_string();
    assert!(e.contains("scalar quantizer type 2"), "{e}");
}

#[test]
fn mutated_sq_files_never_panic() {
    for name in ["sq8_idmap_l2", "ivfsq8_idmap_ip", "hnswsq8_idmap_l2"] {
        let bytes = fixture(name);
        for cut in (0..bytes.len()).step_by(97) {
            let _ = read_index(&bytes[..cut]);
        }
        let limit = bytes.len().min(2400);
        for pos in 0..limit {
            for flip in [0xffu8, 0x01] {
                let mut b = bytes.clone();
                b[pos] ^= flip;
                if let Ok(idx) = read_index(&b) {
                    let _ = idx.search(&[0.1; 24], 3, &SearchParams::default(), None);
                }
            }
        }
    }
}
