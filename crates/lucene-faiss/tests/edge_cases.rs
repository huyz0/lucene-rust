#![allow(clippy::arithmetic_side_effects)]
use lucene_faiss::{build_index, parse_params, read_index, write_index, Metric, SearchParams};

fn data(n: usize, d: usize) -> Vec<f32> {
    let mut s = 99u64;
    (0..n * d)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

#[test]
fn odd_dimension_and_every_kind_search() {
    let d = 11;
    let v = data(200, d);
    let ids: Vec<i64> = (0..200).collect();
    for (desc, m) in [
        ("IDMap,Flat", Metric::L2),
        ("IDMap,HNSW8", Metric::InnerProduct),
        ("IDMap,IVF4,Flat", Metric::L2),
    ] {
        let idx = build_index(
            desc,
            "efConstruction=30,nprobe=4,efSearch=40",
            m,
            d,
            &v,
            &ids,
        )
        .unwrap();
        let hits = idx
            .search(&v[..d], 3, &SearchParams::default(), None)
            .unwrap();
        assert_eq!(hits.len(), 3, "{desc}");
    }
}

#[test]
fn empty_and_tiny_builds() {
    for desc in ["IDMap,Flat", "IDMap,HNSW8", "IDMap,IVF4,Flat"] {
        let idx = build_index(desc, "", Metric::L2, 4, &[], &[]).unwrap();
        assert_eq!(idx.ntotal(), 0);
        let back = read_index(&write_index(&idx)).unwrap();
        assert!(back
            .search(&[0.0; 4], 5, &SearchParams::default(), None)
            .unwrap()
            .is_empty());
    }
    let idx = build_index(
        "IDMap,IVF64,Flat",
        "",
        Metric::L2,
        4,
        &data(3, 4),
        &[1, 2, 3],
    )
    .unwrap();
    assert_eq!(idx.ntotal(), 3);
}

#[test]
fn duplicate_vectors_leave_clusters_empty() {
    let d = 4;
    let mut v = data(8, d);
    let first = v[..d].to_vec();
    for i in 1..200 {
        let _ = i;
        v.extend_from_slice(&first);
    }
    let ids: Vec<i64> = (0..(v.len() / d) as i64).collect();
    let idx = build_index("IDMap,IVF16,Flat", "nprobe=16", Metric::L2, d, &v, &ids).unwrap();
    let hits = idx
        .search(&first, 5, &SearchParams::default(), None)
        .unwrap();
    assert!(hits[0].1 < 1e-9);
    let ip = build_index("IDMap,IVF16,Flat", "", Metric::InnerProduct, d, &v, &ids).unwrap();
    assert_eq!(
        ip.search(&first, 5, &SearchParams::default(), None)
            .unwrap()
            .len(),
        5
    );
}

#[test]
fn bad_arguments() {
    let v = data(4, 4);
    assert!(build_index("IDMap,Flat", "", Metric::L2, 0, &v, &[1, 2, 3, 4]).is_err());
    assert!(build_index("IDMap,Flat", "", Metric::L2, 4, &v, &[1, 2, 3]).is_err());
    assert!(build_index("HNSW8", "", Metric::L2, 4, &v, &[1, 2, 3, 4]).is_err());
    assert!(parse_params("efSearch").is_err());
    assert!(parse_params("efSearch=abc").is_err());
    assert!(parse_params("bogus=3").is_err());
}

#[test]
fn mutated_and_truncated_indexes_never_panic() {
    let d = 6;
    let v = data(60, d);
    let ids: Vec<i64> = (0..60).collect();
    for (desc, m) in [
        ("IDMap,HNSW4", Metric::L2),
        ("IDMap,IVF4,Flat", Metric::InnerProduct),
        ("IDMap,Flat", Metric::L2),
    ] {
        let bytes = write_index(&build_index(desc, "efConstruction=20", m, d, &v, &ids).unwrap());
        for cut in (0..bytes.len()).step_by(7) {
            let _ = read_index(&bytes[..cut]);
        }
        for pos in (0..bytes.len().min(600)).step_by(1) {
            for flip in [0xffu8, 0x7f, 0x01] {
                let mut b = bytes.clone();
                b[pos] ^= flip;
                if let Ok(idx) = read_index(&b) {
                    let _ = idx.search(&v[..d], 3, &SearchParams::default(), None);
                }
            }
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(read_index(&extra).is_err());
    }
}
