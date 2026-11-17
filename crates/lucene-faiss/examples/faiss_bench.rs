#![allow(clippy::arithmetic_side_effects)]
//! Single-thread recall/latency of the native Faiss indexes on the files
//! `benchmarks/ivf/prepare.py` writes:
//! `faiss_bench <prefix> <l2|cosine> [queries] [hnsw|ivf|both]`.
//! Prints `index,config,build_s,recall@10,latency_us,qps,pinned_mb,vectors_mb`.

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

use lucene_faiss::{build_index, write_index, Metric, SearchParams};
use std::time::Instant;

fn read_bin(path: &str) -> (usize, usize, Vec<u8>) {
    let b = std::fs::read(path).unwrap();
    let n = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    (n, d, b[8..].to_vec())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let prefix = &a[1];
    let cosine = a[2] == "cosine";
    let nq: usize = a.get(3).map_or(1000, |s| s.parse().unwrap());
    let which = a.get(4).map_or("both", String::as_str);
    let (n, d, raw) = read_bin(&format!("{prefix}.base.fbin"));
    let mut base: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let (_, _, rawq) = read_bin(&format!("{prefix}.query.fbin"));
    let mut queries: Vec<f32> = rawq
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    queries.truncate(nq * d);
    let (_, kg, rawg) = read_bin(&format!("{prefix}.gt.ibin"));
    let gt: Vec<i32> = rawg
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if cosine {
        for v in base.chunks_exact_mut(d).chain(queries.chunks_exact_mut(d)) {
            let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= nrm);
        }
    }
    let metric = if cosine {
        Metric::InnerProduct
    } else {
        Metric::L2
    };
    let ids: Vec<i64> = (0..n as i64).collect();
    let vectors_mb = (n * d * 4) as f64 / 1e6;
    println!("index,config,build_s,recall@10,latency_us,qps,pinned_mb,vectors_mb");
    let run = |name: &str,
               desc: &str,
               params: &str,
               settings: &[(Option<usize>, Option<usize>)],
               cfg: &dyn Fn(&SearchParams) -> String| {
        let t = Instant::now();
        let index = build_index(desc, params, metric, d, &base, &ids).unwrap();
        let build = t.elapsed().as_secs_f64();
        let mb = write_index(&index).len() as f64 / 1e6;
        for &(ef, np) in settings {
            let sp = SearchParams {
                ef_search: ef,
                nprobe: np,
            };
            let t = Instant::now();
            let mut hits = 0usize;
            for (qi, q) in queries.chunks_exact(d).enumerate() {
                let found = index.search(q, 10, &sp, None).unwrap();
                let truth = &gt[qi * kg..qi * kg + 10];
                hits += found
                    .iter()
                    .filter(|(id, _)| truth.contains(&(*id as i32)))
                    .count();
            }
            let secs = t.elapsed().as_secs_f64();
            println!(
                "{name},{},{build:.1},{:.4},{:.0},{:.0},{mb:.1},{vectors_mb:.1}",
                cfg(&sp),
                hits as f64 / (nq * 10) as f64,
                secs / nq as f64 * 1e6,
                nq as f64 / secs
            );
        }
    };
    if which != "ivf" {
        let s: Vec<_> = [16, 32, 64, 128, 256]
            .iter()
            .map(|&e| (Some(e), None))
            .collect();
        run(
            "native-faiss-hnsw",
            "IDMap,HNSW16",
            "efConstruction=100",
            &s,
            &|p| format!("M=16 efC=100 ef={}", p.ef_search.unwrap()),
        );
    }
    if which != "hnsw" {
        let nlist = (n / 384).max(1);
        let s: Vec<_> = [4, 8, 16, 32, 64, 128]
            .iter()
            .map(|&p| (None, Some(p)))
            .collect();
        run(
            "native-faiss-ivf",
            &format!("IDMap,IVF{nlist},Flat"),
            "",
            &s,
            &|p| format!("nlist={nlist} nprobe={}", p.nprobe.unwrap()),
        );
    }
}
