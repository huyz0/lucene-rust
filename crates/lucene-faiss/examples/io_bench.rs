#![allow(clippy::arithmetic_side_effects)]
//! Serialization speed: `io_bench <prefix>` builds nothing, it round-trips a
//! flat index and an IVF index over the base vectors.
use lucene_faiss::{build_index, read_index, write_index, Metric};
use std::time::Instant;

fn main() {
    let prefix = std::env::args().nth(1).unwrap();
    let b = std::fs::read(format!("{prefix}.base.fbin")).unwrap();
    let n = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let x: Vec<f32> = b[8..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let ids: Vec<i64> = (0..n as i64).collect();
    for desc in ["IDMap,Flat", "IDMap,IVF1015,Flat", "IDMap,HNSW16"] {
        let idx = build_index(desc, "efConstruction=40", Metric::L2, d, &x, &ids).unwrap();
        let t = Instant::now();
        let bytes = write_index(&idx);
        let w = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let back = read_index(&bytes).unwrap();
        let r = t.elapsed().as_secs_f64();
        println!(
            "{desc}: {:.1} MB write {:.0} ms ({:.2} GB/s) read {:.0} ms ({:.2} GB/s) n={}",
            bytes.len() as f64 / 1e6,
            w * 1e3,
            bytes.len() as f64 / w / 1e9,
            r * 1e3,
            bytes.len() as f64 / r / 1e9,
            back.ntotal()
        );
    }
}
