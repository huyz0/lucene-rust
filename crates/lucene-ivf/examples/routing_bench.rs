//! Speed and accuracy of centroid routing: the flat f32 scan against the
//! 8-bit router, with and without the routing tree.
//!
//! ```text
//! cargo run --release -p lucene-ivf --example routing_bench -- <prefix> <l2|cosine> [options]
//!   --base N      use the first N base vectors (default all)
//!   --queries N   queries (default 1000)
//!   --targets L   comma list of vectors per cluster (default 384,96,24)
//!   --nprobe L    comma list of probe counts (default 16,64,256)
//! ```
//!
//! For each cluster count it prints, per mode and probe count, the mean
//! routing time and how many of the exact `nprobe` nearest centroids the
//! router returned (1.0 = the same set). Clustering uses no full Lloyd
//! rounds, to keep large cluster counts quick to build.
#![allow(clippy::arithmetic_side_effects)]

use std::time::Instant;

use lucene_ivf::kmeans::{self, KMeansParams, Metric};
use lucene_ivf::routing::{exact_route, Router, Routing};
use lucene_ivf::Similarity;

fn read_fbin(path: &str) -> (usize, usize, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let data = bytes[8..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (n, d, data)
}

fn list(s: &str) -> Vec<usize> {
    s.split(',').map(|x| x.parse().expect("number")).collect()
}

fn normalize(v: &mut [f32], dim: usize) {
    for r in v.chunks_exact_mut(dim) {
        let _ = lucene_util::vector_util::l2normalize(r, false);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prefix = &args[1];
    let similarity = match args[2].as_str() {
        "l2" => Similarity::Euclidean,
        "cosine" => Similarity::Cosine,
        other => panic!("unknown metric {other}"),
    };
    let mut base_n = usize::MAX;
    let mut queries_n = 1000;
    let mut targets = vec![384, 96, 24];
    let mut nprobes = vec![16, 64, 256];
    let mut covers = vec![16];
    let mut i = 3;
    while i < args.len() {
        let v = &args[i + 1];
        match args[i].as_str() {
            "--base" => base_n = v.parse().unwrap(),
            "--queries" => queries_n = v.parse().unwrap(),
            "--targets" => targets = list(v),
            "--nprobe" => nprobes = list(v),
            "--cover" => covers = list(v),
            other => panic!("unknown option {other}"),
        }
        i += 2;
    }
    let (n, dim, mut base) = read_fbin(&format!("{prefix}.base.fbin"));
    let (qn, _, mut queries) = read_fbin(&format!("{prefix}.query.fbin"));
    let n = n.min(base_n);
    base.truncate(n * dim);
    let qn = qn.min(queries_n);
    queries.truncate(qn * dim);
    let metric = if similarity == Similarity::Cosine {
        normalize(&mut base, dim);
        normalize(&mut queries, dim);
        Metric::InnerProduct
    } else {
        Metric::L2
    };
    println!("nlist,mode,nprobe,route_us,overlap,build_ms");
    for &target in &targets {
        let params = KMeansParams {
            full_rounds: 0,
            ..KMeansParams::default()
        };
        let clustering = kmeans::cluster(&base, dim, target, &params, metric);
        let centroids = clustering.centroids;
        let nlist = centroids.len() / dim;
        let exact: Vec<Vec<Vec<usize>>> = nprobes
            .iter()
            .map(|&np| {
                queries
                    .chunks_exact(dim)
                    .map(|q| exact_route(&centroids, dim, similarity, q, np))
                    .collect()
            })
            .collect();
        for &np in &nprobes {
            let t = Instant::now();
            let mut sink = 0usize;
            for q in queries.chunks_exact(dim) {
                sink += exact_route(&centroids, dim, similarity, q, np)[0];
            }
            let us = t.elapsed().as_secs_f64() * 1e6 / qn as f64;
            std::hint::black_box(sink);
            println!("{nlist},f32,{np},{us:.1},1.0000,0");
        }
        let mut modes = vec![("int8".to_string(), Routing::Quantized)];
        for &cover in &covers {
            modes.push((format!("tree{cover}"), Routing::Tree { cover }));
        }
        for (mode, routing) in modes {
            let t = Instant::now();
            let router = Router::new(&centroids, dim, similarity, routing, None);
            let build_ms = t.elapsed().as_secs_f64() * 1e3;
            if mode != "int8" && !router.has_tree() {
                continue;
            }
            for (pi, &np) in nprobes.iter().enumerate() {
                let t = Instant::now();
                let mut found = 0usize;
                let mut got = Vec::with_capacity(qn);
                for q in queries.chunks_exact(dim) {
                    got.push(router.route(&centroids, q, np));
                }
                let us = t.elapsed().as_secs_f64() * 1e6 / qn as f64;
                let mut total = 0usize;
                for (g, e) in got.iter().zip(&exact[pi]) {
                    found += g.iter().filter(|c| e.contains(c)).count();
                    total += e.len();
                }
                println!(
                    "{nlist},{mode},{np},{us:.1},{:.4},{build_ms:.0}",
                    found as f64 / total as f64
                );
            }
        }
    }
}
