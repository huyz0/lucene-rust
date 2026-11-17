#![allow(clippy::arithmetic_side_effects)]
//! Writes indexes this crate builds, for `scripts/verify-faiss-read.py` to
//! open with the real Faiss library: `write_for_faiss <out-dir>`.

use lucene_faiss::{build_index, write_index, Metric};

fn main() {
    let out = std::env::args().nth(1).expect("output directory");
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/faiss");
    let x: Vec<f32> = std::fs::read(root.join("hnsw_idmap_l2.vectors"))
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let n = x.len() / 24;
    let ids: Vec<i64> = (0..n as i64).map(|i| i * 3 + 7).collect();
    for (name, desc, params, metric) in [
        (
            "ours_hnsw_l2",
            "IDMap,HNSW8",
            "efConstruction=40,efSearch=48",
            Metric::L2,
        ),
        (
            "ours_hnsw_ip",
            "IDMap,HNSW8",
            "efConstruction=40,efSearch=48",
            Metric::InnerProduct,
        ),
        ("ours_ivf_l2", "IDMap,IVF16,Flat", "nprobe=6", Metric::L2),
        (
            "ours_ivf_ip",
            "IDMap,IVF16,Flat",
            "nprobe=6",
            Metric::InnerProduct,
        ),
    ] {
        let index = build_index(desc, params, metric, 24, &x, &ids).unwrap();
        std::fs::write(format!("{out}/{name}.idx"), write_index(&index)).unwrap();
    }
}
