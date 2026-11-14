//! Sparse retrieval prototype bench: Seismic as shipped, Seismic with a copy of
//! each cluster's vectors, and block/superblock pruning.
//!
//!   cargo run --release -p lucene-sparse --example proto_bench -- \
//!       [--docs N] [--queries N] [--corpus flat|topical] [--npost N]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing
)]

use lucene_sparse::proto::*;
use lucene_sparse::search::{search, SegmentView, SeismicQuery};
use std::time::Instant;

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn pct(v: &mut Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() as f64 - 1.0) * p) as usize]
}

struct Row {
    name: String,
    recall: f64,
    mean_us: f64,
    p99_us: f64,
    pages: f64,
}

fn main() {
    let n: usize = arg("--docs", 200_000);
    let nq: usize = arg("--queries", 200);
    let kind: String = arg("--corpus", "flat".to_string());
    let npost: i32 = arg("--npost", 2000);
    let passes: usize = arg("--passes", 3);
    let t0 = Instant::now();
    let data: String = arg("--data", String::new());
    let corpus = if !data.is_empty() {
        real_corpus(&data, n, nq)
    } else if kind == "topical" {
        topical_corpus(n, nq, 400, 20261114)
    } else {
        flat_corpus(n, nq, 20261104)
    };
    println!(
        "corpus {kind}: {n} docs, {nq} queries, n_postings {npost} ({:.1}s)",
        t0.elapsed().as_secs_f32()
    );

    let t0 = Instant::now();
    let seis = build_seismic(&corpus.docs, npost);
    println!(
        "seismic build {:.1}s: sit {:.1} MiB, sip {:.1} MiB, forward {:.1} MiB",
        t0.elapsed().as_secs_f32(),
        seis.sit_bytes as f64 / 1048576.0,
        seis.sip_bytes as f64 / 1048576.0,
        seis.forward.heap_bytes() as f64 / 1048576.0
    );
    let store = ClusterStore::build(&seis);
    println!(
        "per-cluster copies: {} vectors, {:.1} MiB ({:.1}x forward)",
        store.copies(),
        store.copy_bytes() as f64 / 1048576.0,
        store.copy_bytes() as f64 / seis.forward.heap_bytes() as f64
    );

    let t0 = Instant::now();
    let order = bp_order(&corpus.docs, 16);
    let bp_secs = t0.elapsed().as_secs_f32();
    let identity: Vec<u32> = (0..n as u32).collect();
    println!("graph bisection {bp_secs:.1}s");

    let mut indexes = Vec::new();
    for &(b, c, bp) in &[(8usize, 16usize, true), (8, 64, true), (8, 16, false)]
        [..if std::env::args().any(|a| a == "--quick") {
            2
        } else {
            3
        }]
    {
        let t0 = Instant::now();
        let o = if bp { order.clone() } else { identity.clone() };
        let bi = BlockIndex::build(&corpus.docs, o, b, c, seis.field.ceiling_ingest);
        println!(
            "block index b={b} c={c} bp={bp}: build {:.1}s, bounds {:.1} MiB (dense table {:.1} MiB), vectors {:.1} MiB",
            t0.elapsed().as_secs_f32(),
            bi.bounds_bytes() as f64 / 1048576.0,
            bi.dense_table_bytes() as f64 / 1048576.0,
            bi.vector_bytes() as f64 / 1048576.0
        );
        indexes.push((format!("b{b}c{c}{}", if bp { "" } else { "-noBP" }), bi));
    }

    let view = SegmentView {
        max_doc: seis.max_doc,
        live_docs: None,
        forward: Some(&seis.forward),
        postings: &seis.postings,
    };

    for &k in &[10usize, 100] {
        println!("\n=== k = {k} ===");
        let mut rows: Vec<Row> = Vec::new();
        // Queries per grid point: top_n and heap_factor of the Seismic search.
        let queries: Vec<Vec<SeismicQuery>> = [10usize, 20, 40]
            .iter()
            .flat_map(|&top_n| {
                [8.0f32, 2.0, 1.0, 0.7, 0.4]
                    .iter()
                    .map(move |&hf| (top_n, hf))
            })
            .map(|(top_n, hf)| {
                corpus
                    .queries
                    .iter()
                    .map(|q| SeismicQuery::new(q, top_n, k, hf, 1.0, &seis.field).unwrap())
                    .collect()
            })
            .collect();
        let params: Vec<(usize, f32)> = [10usize, 20, 40]
            .iter()
            .flat_map(|&t| [8.0f32, 2.0, 1.0, 0.7, 0.4].iter().map(move |&h| (t, h)))
            .collect();
        let exact_q = SeismicQuery::new(&corpus.queries[0], 10, k, 1.0, 1.0, &seis.field).unwrap();
        let _ = exact_q;
        let full: Vec<SeismicQuery> = corpus
            .queries
            .iter()
            .map(|q| SeismicQuery::new(q, usize::MAX, k, 1.0, 1.0, &seis.field).unwrap())
            .collect();
        let exact: Vec<Vec<i32>> = full
            .iter()
            .map(|q| exact_top(&seis.forward, n, &q.dense, k))
            .collect();
        {
            let mut lats = Vec::new();
            for q in full.iter().take(20) {
                let t = Instant::now();
                let _ = exact_top(&seis.forward, n, &q.dense, k);
                lats.push(t.elapsed().as_secs_f64() * 1e6);
            }
            let mean = lats.iter().sum::<f64>() / lats.len() as f64;
            rows.push(Row {
                name: "exact scan (every doc)".into(),
                recall: 1.0,
                mean_us: mean,
                p99_us: mean,
                pages: 0.0,
            });
        }

        // --- Seismic: production search, then the same loop over each layout.
        for (pi, qs) in queries.iter().enumerate() {
            let (top_n, hf) = params[pi];
            let mut lats = Vec::new();
            let mut rec = 0.0;
            for pass in 0..=passes {
                for (qi, q) in qs.iter().enumerate() {
                    let t = Instant::now();
                    let hits = search(&view, &seis.field, q, None).unwrap();
                    let dt = t.elapsed().as_secs_f64() * 1e6;
                    if pass > 0 {
                        lats.push(dt);
                    } else {
                        let got: Vec<(i32, i32)> = hits
                            .iter()
                            .map(|&(d, _)| (d, seis.forward.dot(d as usize, &q.dense).unwrap_or(0)))
                            .collect();
                        rec += recall(&exact[qi], &got) / qs.len() as f64;
                    }
                }
            }
            let mean = lats.iter().sum::<f64>() / lats.len() as f64;
            let p99 = pct(&mut lats, 0.99);
            // Pages with the shared layout (instrumented loop).
            let mut pages = 0.0;
            for q in qs {
                let mut pg = Vec::new();
                let _ = store.search(&seis.forward, Layout::Shared, q, n, Some(&mut pg));
                pg.sort_unstable();
                pg.dedup();
                pages += pg.len() as f64 / qs.len() as f64;
            }
            rows.push(Row {
                name: format!("seismic shared    top_n={top_n} hf={hf}"),
                recall: rec,
                mean_us: mean,
                p99_us: p99,
                pages,
            });

            let mut lats = Vec::new();
            let mut rec2 = 0.0;
            let mut pages = 0.0;
            for pass in 0..=passes {
                for (qi, q) in qs.iter().enumerate() {
                    let t = Instant::now();
                    let hits = store.search(&seis.forward, Layout::PerCluster, q, n, None);
                    let dt = t.elapsed().as_secs_f64() * 1e6;
                    if pass > 0 {
                        lats.push(dt);
                    } else {
                        rec2 += recall(&exact[qi], &hits) / qs.len() as f64;
                        let mut pg = Vec::new();
                        let _ =
                            store.search(&seis.forward, Layout::PerCluster, q, n, Some(&mut pg));
                        pg.sort_unstable();
                        pg.dedup();
                        pages += pg.len() as f64 / qs.len() as f64;
                    }
                }
            }
            let mean = lats.iter().sum::<f64>() / lats.len() as f64;
            let p99 = pct(&mut lats, 0.99);
            assert!(
                (rec2 - rows.last().unwrap().recall).abs() < 1e-9,
                "layouts disagree"
            );
            rows.push(Row {
                name: format!("seismic per-cluster top_n={top_n} hf={hf}"),
                recall: rec2,
                mean_us: mean,
                p99_us: p99,
                pages,
            });
        }

        // --- Block / superblock pruning.
        for (name, bi) in &indexes {
            for &bt in &[10usize, 20, usize::MAX] {
                for &(gamma, mu) in &[
                    (1usize, 1.0f32),
                    (2, 1.0),
                    (4, 1.0),
                    (8, 1.0),
                    (16, 1.0),
                    (usize::MAX, 1.0),
                    (usize::MAX, 0.9),
                    (usize::MAX, 0.8),
                    (usize::MAX, 0.6),
                ] {
                    if bt != usize::MAX && gamma == usize::MAX && mu < 1.0 && bt < 20 {
                        continue;
                    }
                    let p = BlockParams {
                        gamma,
                        mu,
                        bound_terms: bt,
                    };
                    let mut lats = Vec::new();
                    let mut rec = 0.0;
                    let mut pages = 0.0;
                    let bounds: Vec<Vec<(usize, i32)>> =
                        full.iter().map(|q| bound_terms(q, bt)).collect();
                    for pass in 0..=passes {
                        for (qi, q) in full.iter().enumerate() {
                            let t = Instant::now();
                            let hits = bi.search(q, &bounds[qi], p, None);
                            let dt = t.elapsed().as_secs_f64() * 1e6;
                            if pass > 0 {
                                lats.push(dt);
                            } else {
                                rec += recall(&exact[qi], &hits) / full.len() as f64;
                                let mut pg = Vec::new();
                                let _ = bi.search(q, &bounds[qi], p, Some(&mut pg));
                                pg.sort_unstable();
                                pg.dedup();
                                pages += pg.len() as f64 / full.len() as f64;
                            }
                        }
                    }
                    let mean = lats.iter().sum::<f64>() / lats.len() as f64;
                    let p99 = pct(&mut lats, 0.99);
                    let g = if gamma == usize::MAX {
                        "all".to_string()
                    } else {
                        gamma.to_string()
                    };
                    let b = if bt == usize::MAX {
                        "all".to_string()
                    } else {
                        bt.to_string()
                    };
                    rows.push(Row {
                        name: format!("blocks {name} bound={b} gamma={g} mu={mu}"),
                        recall: rec,
                        mean_us: mean,
                        p99_us: p99,
                        pages,
                    });
                }
            }
        }
        println!(
            "{:<58} {:>7} {:>10} {:>10} {:>9}",
            "config", "recall", "mean us", "p99 us", "pages"
        );
        for r in &rows {
            println!(
                "{:<58} {:>7.3} {:>10.0} {:>10.0} {:>9.0}",
                r.name, r.recall, r.mean_us, r.p99_us, r.pages
            );
        }
    }
}
