#![allow(clippy::arithmetic_side_effects)]
use lucene_faiss::codec::{read_segment, FaissFilesWriter};
use lucene_faiss::{FaissField, SearchParams, Similarity};

fn data(n: usize, d: usize) -> Vec<f32> {
    let mut s = 12345u64;
    (0..n * d)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn exact_l2(v: &[f32], d: usize, q: &[f32]) -> usize {
    v.chunks_exact(d)
        .enumerate()
        .map(|(i, x)| {
            (
                i,
                x.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum::<f32>(),
            )
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
        .0
}

#[test]
fn euclidean_default_factory_finds_nearest_and_scores() {
    let d = 8;
    let v = data(300, d);
    let docs: Vec<i32> = (0..300).map(|i| i * 2).collect();
    let f = FaissField::build(Similarity::Euclidean, d, &v, &docs, None, None, 16, 100).unwrap();
    let q = &v[5 * d..6 * d];
    let r = f
        .search(
            q,
            3,
            &SearchParams {
                ef_search: Some(64),
                nprobe: None,
            },
            None,
        )
        .unwrap();
    assert_eq!(r[0].0, 10);
    assert!((r[0].1 - 1.0).abs() < 1e-6);
    assert!(r[1].1 < r[0].1);
    assert_eq!(exact_l2(&v, d, q), 5);
}

#[test]
fn filter_and_ivf_and_cosine() {
    let d = 8;
    let v = data(400, d);
    let docs: Vec<i32> = (0..400).collect();
    let f = FaissField::build(
        Similarity::Cosine,
        d,
        &v,
        &docs,
        Some("IDMap,IVF8,Flat"),
        Some("nprobe=8"),
        16,
        100,
    )
    .unwrap();
    let q: Vec<f32> = v[3 * d..4 * d].iter().map(|x| x * 5.0).collect();
    let r = f.search(&q, 5, &SearchParams::default(), None).unwrap();
    assert_eq!(r[0].0, 3);
    assert!((r[0].1 - 1.0).abs() < 1e-4);
    let accept = |doc: i32| doc != 3;
    let r = f
        .search(&q, 5, &SearchParams::default(), Some(&accept))
        .unwrap();
    assert!(r.iter().all(|(doc, _)| *doc != 3));
    assert!(f
        .search(&q[..4], 5, &SearchParams::default(), None)
        .is_err());
    let dot = FaissField::build(
        Similarity::DotProduct,
        d,
        &v,
        &docs,
        Some("IDMap,Flat"),
        None,
        16,
        100,
    )
    .unwrap();
    let r = dot
        .search(&v[..d], 2, &SearchParams::default(), None)
        .unwrap();
    assert!(r[0].1 >= r[1].1 && r[0].1 >= 0.0);
}

#[test]
fn refusals() {
    let v = data(10, 4);
    let docs: Vec<i32> = (0..10).collect();
    assert!(FaissField::build(
        Similarity::MaximumInnerProduct,
        4,
        &v,
        &docs,
        None,
        None,
        16,
        100
    )
    .is_err());
    let e = FaissField::build(
        Similarity::Euclidean,
        4,
        &v,
        &docs,
        Some("PQ4"),
        None,
        16,
        100,
    )
    .unwrap_err();
    assert!(e.to_string().contains("PQ4"));
    let f = FaissField::build(
        Similarity::Euclidean,
        4,
        &v,
        &docs,
        Some("IDMap,Flat"),
        None,
        16,
        100,
    )
    .unwrap();
    assert!(FaissField::from_index(
        Similarity::DotProduct,
        lucene_faiss::read_index(&lucene_faiss::write_index(f.index())).unwrap()
    )
    .is_err());
    let ok = FaissField::from_index(
        Similarity::Euclidean,
        lucene_faiss::read_index(&lucene_faiss::write_index(f.index())).unwrap(),
    )
    .unwrap();
    assert_eq!(ok.index().ntotal(), 10);
    assert_eq!(Similarity::DotProduct.score(-5.0), 0.0);
}

#[test]
fn segment_files_round_trip_and_corruption() {
    let d = 4;
    let v = data(50, d);
    let docs: Vec<i32> = (0..50).collect();
    let f = FaissField::build(Similarity::Euclidean, d, &v, &docs, None, None, 8, 40).unwrap();
    let id = [7u8; 16];
    let mut w = FaissFilesWriter::new(id, "Faiss_0");
    w.add_field(2, f.index());
    w.add_field(5, f.index());
    let (meta, dat) = w.finish();
    let seg = read_segment(&meta, &dat, &id, "Faiss_0").unwrap();
    assert_eq!(
        seg.fields.iter().map(|f| f.0).collect::<Vec<_>>(),
        vec![2, 5]
    );
    assert_eq!(seg.fields[0].1.ntotal(), 50);
    assert!(read_segment(&meta, &dat, &[8u8; 16], "Faiss_0").is_err());
    assert!(read_segment(&meta, &dat, &id, "other").is_err());
    let mut bad = dat.clone();
    bad[40] ^= 1;
    assert!(read_segment(&meta, &bad, &id, "Faiss_0").is_err());
    assert!(read_segment(&meta[..meta.len() - 3], &dat, &id, "Faiss_0").is_err());
    // an extent past the data file, and a duplicated field
    let mut w = FaissFilesWriter::new(id, "s");
    w.add_field(1, f.index());
    w.add_field(1, f.index());
    let (m, dd) = w.finish();
    assert!(read_segment(&m, &dd, &id, "s").is_err());
}
