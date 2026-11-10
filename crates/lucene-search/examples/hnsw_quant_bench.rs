//! Quantizers on an HNSW graph: recall@10, single-thread latency and code
//! bytes of every `lucene-vector-quant` quantizer walking the same graph,
//! with the `ef` candidates reranked on the float vectors.
//!
//! ```text
//! hnsw_quant_bench <prefix> <cosine|l2> [--quantizers L] [--ef L]
//!     [--queries N] [--build-on Q] [--threads T]
//! ```
//!
//! The graph (M 16, beam 100) is built once on the float vectors, with `T`
//! threads; `--build-on Q` also builds one on Q's codes (each inserted
//! vector's float query against the codes) and searches that graph with Q.
//! `<prefix>` has `.base.fbin`, `.query.fbin` and `.gt.ibin`
//! (`benchmarks/ivf/prepare.py`).

use std::sync::Arc;
use std::time::Instant;

use lucene_codecs::hnsw::{
    HnswGraphSearcher, KnnCollector, OnHeapHnswGraph, UpdateableVectorScorer, VectorScorer,
};
use lucene_codecs::hnsw_concurrent::{build_concurrent, DEFAULT_BATCH_SIZE};
use lucene_util::quantization::ScalarEncoding;
use lucene_util::simd::{dot_f32, prefetch_bytes, square_distance_f32};
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::rabitq::ExRabitqFactory;
use lucene_vector_quant::rotation::RotatedFactory;
use lucene_vector_quant::scalar::ScalarFactory;
use lucene_vector_quant::{Quantizer, QuantizerFactory, QueryScorer, Similarity};

type R<T> = std::result::Result<T, lucene_codecs::vectors::Error>;
const K: usize = 10;

fn read_bin<T>(path: &str, decode: fn([u8; 4]) -> T) -> (usize, usize, Vec<T>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let data = bytes[8..]
        .chunks_exact(4)
        .map(|c| decode(c.try_into().unwrap()))
        .collect();
    (n, d, data)
}

fn factory(name: &str) -> Arc<dyn QuantizerFactory> {
    let osq = |e| Arc::new(OsqFactory::new(e)) as Arc<dyn QuantizerFactory>;
    let base = |n: &str| -> Arc<dyn QuantizerFactory> {
        match n {
            "bbq" | "osq1" => osq(ScalarEncoding::SingleBitQueryNibble),
            "osq2" => osq(ScalarEncoding::DibitQueryNibble),
            "osq4" => osq(ScalarEncoding::PackedNibble),
            "osq7" => osq(ScalarEncoding::SevenBit),
            "osq8" => osq(ScalarEncoding::UnsignedByte),
            "int7" => Arc::new(ScalarFactory::int7()),
            n if n.starts_with("erq") => Arc::new(ExRabitqFactory::new(n[3..].parse().unwrap())),
            n => panic!("unknown quantizer {n}"),
        }
    };
    match name.strip_prefix("rot-") {
        Some(inner) => Arc::new(RotatedFactory::new(base(inner))),
        None => base(name),
    }
}

/// Exact similarity on prepared (cosine: unit) vectors, Lucene's scale.
#[derive(Clone, Copy)]
struct Exact {
    l2: bool,
}

impl Exact {
    fn score(self, a: &[f32], b: &[f32]) -> f32 {
        if self.l2 {
            1.0 / (1.0 + square_distance_f32(a, b))
        } else {
            (1.0 + dot_f32(a, b)) / 2.0
        }
    }
}

#[derive(Clone)]
struct FloatScorer<'a> {
    data: &'a [f32],
    dim: usize,
    exact: Exact,
    query: Vec<f32>,
}

impl VectorScorer for FloatScorer<'_> {
    fn score(&mut self, node: i32) -> R<f32> {
        let o = node as usize * self.dim;
        Ok(self.exact.score(&self.query, &self.data[o..o + self.dim]))
    }
    fn max_ord(&self) -> i32 {
        (self.data.len() / self.dim) as i32
    }
    fn prefetch(&self, nodes: &[i32]) {
        for &n in nodes {
            let o = n as usize * self.dim;
            prefetch_bytes(bytemuck(&self.data[o..o + self.dim]));
        }
    }
}

impl UpdateableVectorScorer for FloatScorer<'_> {
    fn set_scoring_ordinal(&mut self, ord: i32) -> R<()> {
        let o = ord as usize * self.dim;
        self.query.clear();
        self.query.extend_from_slice(&self.data[o..o + self.dim]);
        Ok(())
    }
}

fn bytemuck(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and any byte pattern is a valid u8.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// Scores codes for one prepared query.
struct CodeScorer<'a> {
    qs: Box<dyn QueryScorer + 'a>,
    codes: &'a [u8],
    len: usize,
    n: i32,
}

impl VectorScorer for CodeScorer<'_> {
    fn score(&mut self, node: i32) -> R<f32> {
        let o = node as usize * self.len;
        Ok(self.qs.score(&self.codes[o..o + self.len]))
    }
    fn max_ord(&self) -> i32 {
        self.n
    }
    fn prefetch(&self, nodes: &[i32]) {
        for &n in nodes {
            let o = n as usize * self.len;
            prefetch_bytes(&self.codes[o..o + self.len]);
        }
    }
}

/// Builds on codes: each inserted vector's float query against the codes.
struct CodeBuildScorer<'a> {
    q: &'a dyn Quantizer,
    centroid: &'a [f32],
    data: &'a [f32],
    dim: usize,
    inner: CodeScorer<'a>,
}

impl VectorScorer for CodeBuildScorer<'_> {
    fn score(&mut self, node: i32) -> R<f32> {
        self.inner.score(node)
    }
    fn max_ord(&self) -> i32 {
        self.inner.n
    }
    fn prefetch(&self, nodes: &[i32]) {
        self.inner.prefetch(nodes)
    }
}

impl UpdateableVectorScorer for CodeBuildScorer<'_> {
    fn set_scoring_ordinal(&mut self, ord: i32) -> R<()> {
        let o = ord as usize * self.dim;
        let mut qs = self.q.query_scorer(&self.data[o..o + self.dim]);
        qs.set_prepared_centroid(self.centroid);
        self.inner.qs = qs;
        Ok(())
    }
}

fn build(scorers: Vec<impl UpdateableVectorScorer + Send>, n: usize) -> (OnHeapHnswGraph, f64) {
    let t = Instant::now();
    let g = build_concurrent(
        scorers,
        16,
        100,
        42,
        OnHeapHnswGraph::with_size(16, n as i32),
        None,
        n as i32,
        DEFAULT_BATCH_SIZE,
    )
    .unwrap();
    (g, t.elapsed().as_secs_f64())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let opt = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .map(|i| args[i + 1].clone())
    };
    let prefix = &args[1];
    let l2 = args[2] == "l2";
    let exact = Exact { l2 };
    let sim = if l2 {
        Similarity::Euclidean
    } else {
        Similarity::Cosine
    };
    let quantizers = opt("--quantizers").unwrap_or_else(|| {
        "bbq,osq2,osq4,osq7,rot-bbq,rot-osq2,rot-osq4,erq2,erq3,erq4,erq5,erq6,erq7,erq8".into()
    });
    let efs: Vec<usize> = opt("--ef")
        .unwrap_or_else(|| "10,20,40,80,160".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let nq: usize = opt("--queries").map_or(2000, |s| s.parse().unwrap());
    let threads: usize = opt("--threads").map_or(4, |s| s.parse().unwrap());

    let (n, dim, mut base) = read_bin(&format!("{prefix}.base.fbin"), f32::from_le_bytes);
    let (_, _, mut queries) = read_bin(&format!("{prefix}.query.fbin"), f32::from_le_bytes);
    let (_, gt_k, gt) = read_bin(&format!("{prefix}.gt.ibin"), i32::from_le_bytes);
    if !l2 {
        for v in base
            .chunks_exact_mut(dim)
            .chain(queries.chunks_exact_mut(dim))
        {
            let norm = dot_f32(v, v).sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
        }
    }
    let nq = nq.min(queries.len() / dim);
    let mut centroid = vec![0f32; dim];
    for v in base.chunks_exact(dim) {
        centroid.iter_mut().zip(v).for_each(|(c, x)| *c += x);
    }
    centroid.iter_mut().for_each(|c| *c /= n as f32);

    let float_scorer = || FloatScorer {
        data: &base,
        dim,
        exact,
        query: Vec::new(),
    };
    let (graph, build_s) = build((0..threads).map(|_| float_scorer()).collect(), n);
    eprintln!("float graph built in {build_s:.1}s");
    println!("quantizer,graph,build_s,encode_s,code_bytes,ef,recall_codes,recall_rerank,mean_us");

    let recall = |hits: &[i32], qi: usize| -> f64 {
        let truth = &gt[qi * gt_k..qi * gt_k + K];
        hits.iter().take(K).filter(|h| truth.contains(h)).count() as f64 / K as f64
    };

    // Float baseline.
    let mut searcher = HnswGraphSearcher::new(K, n as i32);
    for &ef in &efs {
        let mut s = float_scorer();
        let (mut r, t) = (0f64, Instant::now());
        for qi in 0..nq {
            s.query.clear();
            s.query
                .extend_from_slice(&queries[qi * dim..(qi + 1) * dim]);
            let mut c = KnnCollector::unlimited(ef);
            searcher.search(&mut c, &mut s, &graph, None).unwrap();
            let hits: Vec<i32> = c.top_docs().into_iter().map(|h| h.0).collect();
            r += recall(&hits, qi);
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / nq as f64;
        println!(
            "float,float,{build_s:.1},0,{},{ef},{:.4},{:.4},{us:.0}",
            dim * 4,
            r / nq as f64,
            r / nq as f64
        );
    }

    let build_on = opt("--build-on");
    for name in quantizers.split(',') {
        let f = factory(name);
        let sample_n = n.min(65536);
        let q = f.train(dim, sim, &base[..sample_n * dim]).unwrap();
        let prepared = q
            .prepare_centroid(&centroid)
            .unwrap_or_else(|| centroid.clone());
        let len = q.code_len();
        let t = Instant::now();
        let mut codes = vec![0u8; len * n];
        for (v, out) in base.chunks_exact(dim).zip(codes.chunks_exact_mut(len)) {
            q.encode_prepared(v, &prepared, out);
        }
        let encode_s = t.elapsed().as_secs_f64();
        let mut graphs: Vec<(&str, f64, OnHeapHnswGraph)> = Vec::new();
        if build_on.as_deref() == Some(name) {
            let scorers: Vec<CodeBuildScorer> = (0..threads)
                .map(|_| CodeBuildScorer {
                    q: q.as_ref(),
                    centroid: &prepared,
                    data: &base,
                    dim,
                    inner: CodeScorer {
                        qs: q.query_scorer(&base[..dim]),
                        codes: &codes,
                        len,
                        n: n as i32,
                    },
                })
                .collect();
            // SAFETY of Send: each scorer is used by one worker thread only.
            struct SendScorer<'a>(CodeBuildScorer<'a>);
            unsafe impl Send for SendScorer<'_> {}
            impl VectorScorer for SendScorer<'_> {
                fn score(&mut self, node: i32) -> R<f32> {
                    self.0.score(node)
                }
                fn max_ord(&self) -> i32 {
                    self.0.max_ord()
                }
                fn prefetch(&self, nodes: &[i32]) {
                    self.0.prefetch(nodes)
                }
            }
            impl UpdateableVectorScorer for SendScorer<'_> {
                fn set_scoring_ordinal(&mut self, ord: i32) -> R<()> {
                    self.0.set_scoring_ordinal(ord)
                }
            }
            let (g, s) = build(scorers.into_iter().map(SendScorer).collect(), n);
            eprintln!("{name} graph built in {s:.1}s");
            graphs.push(("codes", s, g));
        }
        let mut runs: Vec<(&str, f64, &OnHeapHnswGraph)> = vec![("float", build_s, &graph)];
        runs.extend(graphs.iter().map(|(a, b, g)| (*a, *b, g)));
        for (gname, gs, g) in runs {
            for &ef in &efs {
                let (mut rc, mut rr, t) = (0f64, 0f64, Instant::now());
                let mut fs = float_scorer();
                for qi in 0..nq {
                    let qv = &queries[qi * dim..(qi + 1) * dim];
                    let mut qs = q.query_scorer(qv);
                    qs.set_prepared_centroid(&prepared);
                    let mut s = CodeScorer {
                        qs,
                        codes: &codes,
                        len,
                        n: n as i32,
                    };
                    let mut c = KnnCollector::unlimited(ef);
                    searcher.search(&mut c, &mut s, g, None).unwrap();
                    let cands = c.top_docs();
                    rc += recall(&cands.iter().map(|h| h.0).collect::<Vec<_>>(), qi);
                    fs.query.clear();
                    fs.query.extend_from_slice(qv);
                    let ords: Vec<i32> = cands.iter().map(|h| h.0).collect();
                    fs.prefetch(&ords);
                    let mut re: Vec<(i32, f32)> =
                        ords.iter().map(|&o| (o, fs.score(o).unwrap())).collect();
                    re.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                    rr += recall(&re.iter().map(|h| h.0).collect::<Vec<_>>(), qi);
                }
                let us = t.elapsed().as_secs_f64() * 1e6 / nq as f64;
                println!(
                    "{name},{gname},{gs:.1},{encode_s:.1},{len},{ef},{:.4},{:.4},{us:.0}",
                    rc / nq as f64,
                    rr / nq as f64
                );
            }
        }
    }
}
