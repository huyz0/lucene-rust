//! A grouped search returns each of the best groups' best document, as a
//! block-join k-NN search returns each parent's best child.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use lucene_ivf::{
    FieldVectors, Grouping, IvfParams, IvfReader, IvfWriter, Probes, Rerank, SearchParams,
    Similarity,
};
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::QuantizerRegistry;

const ID: [u8; 16] = [5; 16];
const DIM: usize = 16;

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut x = seed ^ 0x2545_F491_4F6C_DD1D;
    move || {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

/// `n` vectors in groups of 1-8 consecutive documents whose vectors lie
/// close together, as a document's chunks do, around 12 centres. Returns
/// the vectors and each document's group.
fn dataset(n: usize, spill: Option<f32>) -> (IvfReader, Vec<f32>, Vec<i32>) {
    let mut next = rng(7);
    let centres: Vec<Vec<f32>> = (0..12)
        .map(|_| (0..DIM).map(|_| next() * 8.0).collect())
        .collect();
    let mut data = Vec::with_capacity(n * DIM);
    let mut groups = Vec::with_capacity(n);
    let mut g = 0i32;
    while groups.len() < n {
        let size = 1 + (next() + 0.5).mul_add(8.0, 0.0) as usize % 8;
        let centre = &centres[g as usize % 12];
        let base: Vec<f32> = centre.iter().map(|c| c + next() * 2.0).collect();
        for _ in 0..size.min(n - groups.len()) {
            data.extend(base.iter().map(|b| b + next() * 0.3));
            groups.push(g);
        }
        g += 1;
    }
    let docs: Vec<i32> = (0..n as i32).collect();
    let mut w = IvfWriter::new(&ID, "");
    let params = IvfParams {
        target_cluster_size: 100,
        spill,
        ..IvfParams::default()
    };
    w.add_field(
        &FieldVectors {
            number: 1,
            similarity: Similarity::Euclidean,
            dimension: DIM,
            docs: &docs,
            vectors: &data,
        },
        &OsqFactory::bbq(),
        &params,
    )
    .unwrap();
    let files = w.finish();
    let r = IvfReader::open(
        &files.meta,
        Arc::new(files.postings.clone()),
        Arc::new(files.vectors.clone()),
        &ID,
        "",
        &QuantizerRegistry::with_builtins(),
    )
    .unwrap();
    (r, data, groups)
}

/// The exact best `k` groups: each one's best accepted document and score.
fn exact(
    data: &[f32],
    groups: &[i32],
    q: &[f32],
    k: usize,
    accept: impl Fn(i32) -> bool,
) -> Vec<(i32, f32)> {
    let mut best: BTreeMap<i32, (i32, f32)> = BTreeMap::new();
    for (d, g) in groups.iter().enumerate() {
        if !accept(d as i32) {
            continue;
        }
        let s = Similarity::Euclidean.score(q, &data[d * DIM..(d + 1) * DIM]);
        let e = best.entry(*g).or_insert((d as i32, f32::NEG_INFINITY));
        if s > e.1 {
            *e = (d as i32, s);
        }
    }
    let mut v: Vec<(i32, f32)> = best.into_values().collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(k);
    v
}

#[test]
fn every_cluster_probed_finds_the_exact_best_child_of_each_group() {
    for spill in [None, Some(1.0)] {
        let n = 3000;
        let (r, data, groups) = dataset(n, spill);
        let group = |d: i32| groups[d as usize];
        let grouping = Grouping {
            group: &group,
            spread: 5,
        };
        let clusters = r.field(1).unwrap().clusters();
        for qi in 0..10 {
            let mut noise = rng(100 + qi);
            let q: Vec<f32> = data[(qi as usize * 97) * DIM..][..DIM]
                .iter()
                .map(|x| x + noise())
                .collect();
            let k = 10;
            for rerank in [Rerank::Fixed(n), Rerank::Bounded { cap: n }] {
                let params = SearchParams {
                    k,
                    nprobe: Probes::Fixed(clusters),
                    rerank,
                };
                let out = r
                    .search_grouped(1, &q, &params, None, None, &grouping)
                    .unwrap();
                let got: Vec<(i32, f32)> = out.hits.iter().map(|h| (h.doc, h.score)).collect();
                assert_eq!(got, exact(&data, &groups, &q, k, |_| true), "{rerank:?}");
            }
            // Filtered: odd documents only.
            let odd = |d: i32| d % 2 == 1;
            let accept: &dyn Fn(i32) -> bool = &odd;
            let params = SearchParams::new(k, clusters, n);
            let out = r
                .search_grouped(1, &q, &params, Some(accept), None, &grouping)
                .unwrap();
            let got: Vec<(i32, f32)> = out.hits.iter().map(|h| (h.doc, h.score)).collect();
            assert_eq!(got, exact(&data, &groups, &q, k, odd));
        }
    }
}

#[test]
fn estimates_and_calibrated_probes_still_keep_one_child_per_group() {
    let (r, data, groups) = dataset(3000, None);
    let group = |d: i32| groups[d as usize];
    let grouping = Grouping {
        group: &group,
        spread: 5,
    };
    let q = &data[123 * DIM..124 * DIM];
    for params in [
        SearchParams::new(10, 4, 0),
        SearchParams::auto(10),
        SearchParams::new(0, 4, 40),
    ] {
        let out = r
            .search_grouped(1, q, &params, None, None, &grouping)
            .unwrap();
        let distinct: BTreeSet<i32> = out.hits.iter().map(|h| group(h.doc)).collect();
        assert_eq!(distinct.len(), out.hits.len());
        assert_eq!(out.hits.len(), params.k);
        assert!(out.hits.windows(2).all(|w| w[0].score >= w[1].score));
    }
    assert!(format!("{grouping:?}").contains("spread: 5"));
}
