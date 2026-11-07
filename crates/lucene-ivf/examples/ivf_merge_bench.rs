//! Merge time and recall of an IVF merge that reuses the largest source's
//! clusters against one that re-clusters every vector.
//!
//! ```text
//! cargo run --release -p lucene-ivf --example ivf_merge_bench -- <prefix> <l2|cosine> [options]
//!   --split F     share of the base vectors in segment A, the rest in B (default 0.9)
//!   --delete F    share of A's documents deleted before the merge (default 0)
//!   --queries N   queries to run (default 1000)
//!   --target N    IVF vectors per cluster (default 384)
//!   --nprobe N    clusters probed (default 64); --rerank N  candidates reranked (default 100)
//!   --refine N    neighbourhood rounds of the reusing merge (default 1)
//!   --generations G  instead of one merge, a stream of G merges: each merges
//!                 the previous result with the next 1/G of B, deleting the
//!                 --delete share of the previous result's live documents,
//!                 and prints a row per policy and generation
//! ```
//!
//! `<prefix>` names `.base.fbin`, `.query.fbin` and `.gt.ibin` as
//! `benchmarks/ivf/prepare.py` writes them; the ground truth is over all
//! base vectors, so recall counts only live ones when `--delete` is set.
//! Segments are built with BBQ. Prints one CSV row per merge policy.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use lucene_ivf::{
    merge_field, FieldVectors, IvfFiles, IvfParams, IvfReader, IvfWriter, MergeClustering,
    MergeSource, SearchParams, Similarity,
};
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::QuantizerRegistry;

const K: usize = 10;
const ID: [u8; 16] = [5; 16];

fn read_bin<T: Copy>(path: &str, decode: fn([u8; 4]) -> T) -> (usize, usize, Vec<T>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let data = bytes[8..]
        .chunks_exact(4)
        .map(|c| decode(c.try_into().unwrap()))
        .collect();
    (n, d, data)
}

fn open(files: &IvfFiles) -> IvfReader {
    IvfReader::open(
        &files.meta,
        Arc::new(files.postings.clone()),
        Arc::new(files.vectors.clone()),
        &ID,
        "",
        &QuantizerRegistry::with_builtins(),
    )
    .unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prefix = &args[1];
    let similarity = match args[2].as_str() {
        "l2" => Similarity::Euclidean,
        "cosine" => Similarity::Cosine,
        m => panic!("unknown metric {m}"),
    };
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let split: f64 = opt("--split").map_or(0.9, |s| s.parse().unwrap());
    let delete: f64 = opt("--delete").map_or(0.0, |s| s.parse().unwrap());
    let target = opt("--target").map_or(384, |s| s.parse().unwrap());
    let nprobe = opt("--nprobe").map_or(64, |s| s.parse().unwrap());
    let rerank = opt("--rerank").map_or(100, |s| s.parse().unwrap());
    let refine_rounds = opt("--refine").map_or(1, |s| s.parse().unwrap());
    let (n, dim, base) = read_bin(&format!("{prefix}.base.fbin"), f32::from_le_bytes);
    let (nq_all, _, queries) = read_bin(&format!("{prefix}.query.fbin"), f32::from_le_bytes);
    let (_, gt_k, gt) = read_bin(&format!("{prefix}.gt.ibin"), i32::from_le_bytes);
    let nq = opt("--queries")
        .map_or(1000, |s| s.parse().unwrap())
        .min(nq_all);
    let na = (n as f64 * split) as usize;
    let factory = OsqFactory::bbq();
    let params = IvfParams {
        target_cluster_size: target,
        ..IvfParams::default()
    };
    let segment = |from: usize, to: usize| {
        let docs: Vec<i32> = (0..(to - from) as i32).collect();
        let mut w = IvfWriter::new(&ID, "");
        w.add_field(
            &FieldVectors {
                number: 0,
                similarity,
                dimension: dim,
                docs: &docs,
                vectors: &base[from * dim..to * dim],
            },
            &factory,
            &params,
        )
        .unwrap();
        open(&w.finish())
    };
    if let Some(g) = opt("--generations") {
        let generations: usize = g.parse().unwrap();
        let search = SearchParams::new(K, nprobe, rerank);
        println!("policy,generation,merge_s,clusters,min_size,max_size,recall@{K},search_us");
        for (name, policy) in [
            ("recluster", MergeClustering::Recluster),
            (
                "reuse",
                MergeClustering::ReuseLargest {
                    min_fraction: 0.5,
                    refine_rounds,
                },
            ),
        ] {
            let p = IvfParams {
                merge_clustering: policy,
                ..params
            };
            let mut current = segment(0, na);
            let mut deleted: HashSet<i32> = HashSet::new();
            let mut live: Vec<i32> = (0..na as i32).collect();
            let mut total = 0.0;
            for gen in 0..generations {
                let from = na + (n - na) * gen / generations;
                let to = na + (n - na) * (gen + 1) / generations;
                let fresh = segment(from, to);
                // Deletes spread over the live documents, a different set
                // each generation.
                let step = if delete > 0.0 {
                    (1.0 / delete).round() as usize
                } else {
                    0
                };
                let mut gone = HashSet::new();
                if step > 0 {
                    for (i, &d) in live.iter().enumerate() {
                        if (i + gen) % step == 0 {
                            gone.insert(d);
                        }
                    }
                }
                let map_cur = |d: i32| (!gone.contains(&d)).then_some(d);
                let offset = from as i32;
                let map_new = |d: i32| Some(offset + d);
                let sources = [
                    MergeSource {
                        reader: &current,
                        doc_map: &map_cur,
                    },
                    MergeSource {
                        reader: &fresh,
                        doc_map: &map_new,
                    },
                ];
                let t = Instant::now();
                let mut w = IvfWriter::new(&ID, "");
                merge_field(&mut w, 0, &sources, &factory, &p).unwrap();
                let files = w.finish();
                let secs = t.elapsed().as_secs_f64();
                total += secs;
                current = open(&files);
                live.retain(|d| !gone.contains(d));
                live.extend(from as i32..to as i32);
                deleted.extend(gone);
                let visible = to;
                let f = current.field(0).unwrap();
                let sizes: Vec<usize> = (0..f.clusters()).map(|c| f.cluster_size(c)).collect();
                let mut found = 0usize;
                let mut wanted = 0usize;
                let t = Instant::now();
                for q in 0..nq {
                    let truth: Vec<i32> = gt[q * gt_k..(q + 1) * gt_k]
                        .iter()
                        .copied()
                        .filter(|&d| (d as usize) < visible && !deleted.contains(&d))
                        .take(K)
                        .collect();
                    let hits = current
                        .search(0, &queries[q * dim..(q + 1) * dim], &search, None)
                        .unwrap();
                    found += hits.iter().filter(|h| truth.contains(&h.doc)).count();
                    wanted += truth.len();
                }
                let search_us = t.elapsed().as_secs_f64() * 1e6 / nq as f64;
                println!(
                    "{name},{},{secs:.2},{},{},{},{:.4},{search_us:.0}",
                    gen + 1,
                    sizes.len(),
                    sizes.iter().min().unwrap(),
                    sizes.iter().max().unwrap(),
                    found as f64 / wanted as f64
                );
            }
            eprintln!("{name}: {total:.1}s merging");
        }
        return;
    }
    let t = Instant::now();
    let a = segment(0, na);
    let b = segment(na, n);
    eprintln!(
        "built A ({na}) and B ({}) in {:.1}s",
        n - na,
        t.elapsed().as_secs_f64()
    );
    // Every `step`th document of A is deleted; merged documents keep their
    // base ordinal so the ground truth applies.
    let step = if delete > 0.0 {
        (1.0 / delete).round() as i32
    } else {
        0
    };
    let deleted = |d: i32| step > 0 && d % step == 0;
    let map_a = |d: i32| (!deleted(d)).then_some(d);
    let offset = na as i32;
    let map_b = |d: i32| Some(offset + d);
    let sources = [
        MergeSource {
            reader: &a,
            doc_map: &map_a,
        },
        MergeSource {
            reader: &b,
            doc_map: &map_b,
        },
    ];
    println!("policy,merge_s,clusters,min_size,max_size,recall@{K},nprobe,rerank");
    for (name, policy) in [
        ("recluster", MergeClustering::Recluster),
        (
            "reuse",
            MergeClustering::ReuseLargest {
                min_fraction: 0.5,
                refine_rounds,
            },
        ),
    ] {
        let p = IvfParams {
            merge_clustering: policy,
            ..params
        };
        let t = Instant::now();
        let mut w = IvfWriter::new(&ID, "");
        merge_field(&mut w, 0, &sources, &factory, &p).unwrap();
        let files = w.finish();
        let secs = t.elapsed().as_secs_f64();
        let merged = open(&files);
        let f = merged.field(0).unwrap();
        let sizes: Vec<usize> = (0..f.clusters()).map(|c| f.cluster_size(c)).collect();
        let search = SearchParams::new(K, nprobe, rerank);
        let mut found = 0usize;
        let mut wanted = 0usize;
        for q in 0..nq {
            let truth: Vec<i32> = gt[q * gt_k..(q + 1) * gt_k]
                .iter()
                .copied()
                .filter(|&d| (d as usize) >= na || !deleted(d))
                .take(K)
                .collect();
            let hits = merged
                .search(0, &queries[q * dim..(q + 1) * dim], &search, None)
                .unwrap();
            found += hits.iter().filter(|h| truth.contains(&h.doc)).count();
            wanted += truth.len();
        }
        println!(
            "{name},{secs:.2},{},{},{},{:.4},{nprobe},{rerank}",
            sizes.len(),
            sizes.iter().min().unwrap(),
            sizes.iter().max().unwrap(),
            found as f64 / wanted as f64
        );
    }
}
