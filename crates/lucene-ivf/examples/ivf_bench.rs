//! Recall and latency of the IVF index against HNSW + BBQ.
//!
//! ```text
//! cargo run --release -p lucene-ivf --example ivf_bench -- <prefix> <l2|cosine> [options]
//!   --base N         use the first N base vectors (ground truth must match: default all)
//!   --queries N      queries to run (default 1000)
//!   --target N       IVF vectors per cluster (default 384)
//!   --quantizers L   comma list of bbq,osq2,osq4,osq7,sq7,sq4,erq2..erq8, or any
//!                    osq/sq name prefixed r for its randomly rotated form
//!                    (rbbq, rosq2, ...) (default bbq,osq4,sq7)
//!   --nprobe L       comma list of probe counts (default 4,8,16,32,64,128)
//!   --rerank L       comma list of candidates reranked (default 0,50,100)
//!   --refine N       k-means neighbourhood rounds; --full N  full Lloyd rounds
//!   --iters N        Lloyd iterations per split
//!   --spill X        SOAR spill with lambda X (default: no spilling)
//!   --hnsw           also build and search Lucene104 HNSW + BBQ (M 16, beam 100)
//!   --ef L           HNSW candidate counts (default 10,20,40,80,160,320)
//! ```
//!
//! `<prefix>` names `.base.fbin`, `.query.fbin` and `.gt.ibin` as
//! `benchmarks/ivf/prepare.py` writes them. Builds use every core (rayon);
//! searches run one query at a time on one thread, so QPS is single-thread
//! QPS. Prints one CSV row per configuration.
#![allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]

use std::sync::Arc;
use std::time::Instant;

use lucene_codecs::field_infos::VectorSimilarityFunction as CodecSim;
use lucene_codecs::scalar_quantized_vectors::{
    binarized_query_data, merge_scorer_supplier, QuantizedQueryScorer, QuantizedVectorsField,
    ScalarQuantizedVectorsReader, ScalarQuantizedVectorsWriter,
};
use lucene_codecs::{hnsw, hnsw_concurrent, hnsw_vectors};
use lucene_ivf::{FieldVectors, IvfParams, IvfReader, IvfWriter, SearchParams, Similarity};
use lucene_util::quantization::ScalarEncoding;
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::rabitq::ExRabitqFactory;
use lucene_vector_quant::rotation::RotatedFactory;
use lucene_vector_quant::scalar::ScalarFactory;
use lucene_vector_quant::{QuantizerFactory, QuantizerRegistry};

const K: usize = 10;
const ID: [u8; 16] = [3; 16];

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

fn list(s: &str) -> Vec<usize> {
    s.split(',').map(|x| x.parse().expect("number")).collect()
}

fn factory(name: &str) -> Box<dyn QuantizerFactory> {
    if let Some(inner) = name.strip_prefix('r') {
        return Box::new(RotatedFactory::new(Arc::from(factory(inner))));
    }
    match name {
        "bbq" => Box::new(OsqFactory::bbq()),
        "osq2" => Box::new(OsqFactory::new(ScalarEncoding::DibitQueryNibble)),
        "osq4" => Box::new(OsqFactory::new(ScalarEncoding::PackedNibble)),
        "osq7" => Box::new(OsqFactory::new(ScalarEncoding::SevenBit)),
        "sq7" => Box::new(ScalarFactory::int7()),
        "sq4" => Box::new(ScalarFactory::int4()),
        erq if erq.starts_with("erq") => match erq[3..].parse() {
            Ok(bits @ 2..=8) => Box::new(ExRabitqFactory::new(bits)),
            _ => panic!("unknown quantizer {erq}: erq takes 2..=8 bits"),
        },
        other => panic!("unknown quantizer {other}"),
    }
}

fn recall(hits: impl Iterator<Item = i32>, truth: &[i32]) -> usize {
    hits.filter(|d| truth[..K].contains(d)).count()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prefix = &args[1];
    let similarity = match args[2].as_str() {
        "l2" => Similarity::Euclidean,
        "cosine" => Similarity::Cosine,
        "dot" => Similarity::DotProduct,
        "mip" => Similarity::MaximumInnerProduct,
        m => panic!("unknown metric {m}"),
    };
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let (n_all, dim, base) = read_bin(&format!("{prefix}.base.fbin"), f32::from_le_bytes);
    let (nq_all, _, queries) = read_bin(&format!("{prefix}.query.fbin"), f32::from_le_bytes);
    let (_, gt_k, gt) = read_bin(&format!("{prefix}.gt.ibin"), i32::from_le_bytes);
    let n = opt("--base")
        .map_or(n_all, |s| s.parse().unwrap())
        .min(n_all);
    let nq = opt("--queries")
        .map_or(1000, |s| s.parse().unwrap())
        .min(nq_all);
    let target = opt("--target").map_or(384, |s| s.parse().unwrap());
    let spill: Option<f32> = opt("--spill").map(|s| s.parse().unwrap());
    let mut kmeans = lucene_ivf::kmeans::KMeansParams::default();
    if let Some(r) = opt("--refine") {
        kmeans.refine_rounds = r.parse().unwrap();
    }
    if let Some(r) = opt("--full") {
        kmeans.full_rounds = r.parse().unwrap();
    }
    if let Some(r) = opt("--iters") {
        kmeans.iterations = r.parse().unwrap();
    }
    let quantizers = opt("--quantizers").unwrap_or_else(|| "bbq,osq4,sq7".into());
    let nprobes = list(&opt("--nprobe").unwrap_or_else(|| "4,8,16,32,64,128".into()));
    let reranks = list(&opt("--rerank").unwrap_or_else(|| "0,50,100".into()));
    let efs = list(&opt("--ef").unwrap_or_else(|| "10,20,40,80,160,320".into()));
    let base = &base[..n * dim];
    let docs: Vec<i32> = (0..n as i32).collect();
    let truth = |q: usize| &gt[q * gt_k..(q + 1) * gt_k];
    eprintln!("{prefix}: {n} x {dim}, {nq} queries, {:?}", similarity);
    println!("index,config,build_s,recall@10,latency_us,qps,pinned_mb,vectors_mb");

    let registry = QuantizerRegistry::with_builtins();
    for qname in quantizers.split(',') {
        let f = factory(qname);
        let t = Instant::now();
        let mut w = IvfWriter::new(&ID, "");
        w.add_field(
            &FieldVectors {
                number: 0,
                similarity,
                dimension: dim,
                docs: &docs,
                vectors: base,
            },
            f.as_ref(),
            &IvfParams {
                target_cluster_size: target,
                spill,
                kmeans,
                ..IvfParams::default()
            },
        )
        .unwrap();
        let files = w.finish();
        let build = t.elapsed().as_secs_f64();
        let pinned = (files.meta.len() + files.postings.len()) as f64 / 1e6;
        let vectors_mb = files.vectors.len() as f64 / 1e6;
        let r = IvfReader::open(
            &files.meta,
            Arc::new(files.postings),
            Arc::new(files.vectors),
            &ID,
            "",
            &registry,
        )
        .unwrap();
        let clusters = r.field(0).unwrap().clusters();
        let entries = r.field(0).unwrap().entries();
        for &nprobe in &nprobes {
            for &rerank in &reranks {
                let params = SearchParams::new(K, nprobe, rerank);
                let mut found = 0;
                let t = Instant::now();
                for q in 0..nq {
                    let hits = r
                        .search(0, &queries[q * dim..(q + 1) * dim], &params, None)
                        .unwrap();
                    found += recall(hits.iter().map(|h| h.doc), truth(q));
                }
                let secs = t.elapsed().as_secs_f64();
                println!(
                    "ivf-{qname},clusters={clusters} entries={entries} nprobe={nprobe} rerank={rerank},{build:.1},{:.4},{:.0},{:.0},{pinned:.1},{vectors_mb:.1}",
                    found as f64 / (nq * K) as f64,
                    secs / nq as f64 * 1e6,
                    nq as f64 / secs
                );
            }
        }
    }

    if args.iter().any(|a| a == "--hnsw") {
        hnsw_bbq(
            similarity,
            dim,
            base,
            &docs,
            &queries[..nq * dim],
            &gt,
            gt_k,
            &efs,
        );
    }
}

/// `Lucene104HnswScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE, 16,
/// 100)` as a merged segment has it: 1-bit codes and a graph built on the
/// asymmetric quantized scorer, searched on the codes, then the candidates
/// reranked with the full-precision vectors.
fn hnsw_bbq(
    similarity: Similarity,
    dim: usize,
    base: &[f32],
    docs: &[i32],
    queries: &[f32],
    gt: &[i32],
    gt_k: usize,
    efs: &[usize],
) {
    let sim = match similarity {
        Similarity::Euclidean => CodecSim::Euclidean,
        Similarity::DotProduct => CodecSim::DotProduct,
        Similarity::Cosine => CodecSim::Cosine,
        Similarity::MaximumInnerProduct => CodecSim::MaximumInnerProduct,
    };
    let n = docs.len();
    let t = Instant::now();
    let mut w =
        ScalarQuantizedVectorsWriter::new(ScalarEncoding::SingleBitQueryNibble, n as i32, &ID, "");
    w.write_field(&QuantizedVectorsField {
        field_number: 0,
        similarity: sim,
        dimension: dim as i32,
        docs,
        vectors: base,
    })
    .unwrap();
    let (veq, vemq) = w.finish();
    let reader = ScalarQuantizedVectorsReader::open(&vemq, &veq, &ID, "").unwrap();
    let values = reader.quantized_vector_values(0).unwrap();
    let query_side = binarized_query_data(&values, sim, base).unwrap();
    let threads = rayon::current_num_threads();
    let scorers: Vec<_> = (0..threads)
        .map(|_| merge_scorer_supplier(&values, sim, Some(&query_side)).unwrap())
        .collect();
    let graph = hnsw_concurrent::build_concurrent(
        scorers,
        hnsw::DEFAULT_MAX_CONN,
        hnsw::DEFAULT_BEAM_WIDTH,
        hnsw::DEFAULT_RAND_SEED,
        hnsw::OnHeapHnswGraph::with_size(hnsw::DEFAULT_MAX_CONN, n as i32),
        None,
        n as i32,
        hnsw_concurrent::DEFAULT_BATCH_SIZE,
    )
    .unwrap();
    let build = t.elapsed().as_secs_f64();
    let mut level0 = 0usize;
    for node in 0..n as i32 {
        level0 += graph.neighbors(0, node).size();
    }
    // Codes plus level-0 adjacency (4 bytes a neighbour), what must stay hot.
    let pinned = (veq.len() + level0 * 4) as f64 / 1e6;
    let vectors_mb = (base.len() * 4) as f64 / 1e6;
    let nq = queries.len() / dim;
    let mut q = vec![0f32; dim];
    for &ef in efs {
        let mut found = 0;
        let t = Instant::now();
        for qi in 0..nq {
            q.copy_from_slice(&queries[qi * dim..(qi + 1) * dim]);
            let mut scorer = QuantizedQueryScorer::new(&values, sim, &q).unwrap();
            let (cands, _) = hnsw_vectors::search(
                &mut scorer,
                Some(&graph),
                ef.max(K),
                u64::MAX,
                hnsw_vectors::SearchOptions::default(),
            )
            .unwrap();
            let mut exact: Vec<(f32, i32)> = cands
                .iter()
                .map(|&(ord, _)| {
                    let v = &base[ord as usize * dim..(ord as usize + 1) * dim];
                    (similarity.score(&q, v), ord)
                })
                .collect();
            exact.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            found += recall(
                exact.iter().take(K).map(|e| e.1),
                &gt[qi * gt_k..(qi + 1) * gt_k],
            );
        }
        let secs = t.elapsed().as_secs_f64();
        println!(
            "hnsw-bbq,M=16 beam=100 ef={ef} rerank={ef},{build:.1},{:.4},{:.0},{:.0},{pinned:.1},{vectors_mb:.1}",
            found as f64 / (nq * K) as f64,
            secs / nq as f64 * 1e6,
            nq as f64 / secs
        );
    }
}
