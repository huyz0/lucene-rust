//! Realtime k-NN over an indexing buffer: which structure a search should
//! read buffered vectors from.
//!
//! The first `--committed` base vectors stand for the committed segments
//! (they train the IVF centroids and the int8 ranges); the next vectors
//! arrive into the buffer one at a time. At each buffer size the search
//! reads only the buffered vectors, and recall@10 is against the exact top
//! 10 of those same buffered vectors (the committed leaves are searched as
//! today and merged exactly, so the buffer's recall bounds what realtime
//! adds).
//!
//! Options:
//! - `exact`: f32 scan of every buffered vector.
//! - `int8`: int8 scan (ranges from the committed vectors), f32 rerank of
//!   the best `4k`.
//! - `hnsw`: one graph over the buffer, each vector inserted on arrival
//!   (M 16, beam 100), searched with ef 100.
//! - `hnsw-chunks`: a separate graph per `--chunk` vectors, each searched.
//! - `ivf`: each vector assigned on arrival to the nearest committed
//!   centroid and appended to its list; a search ranks centroids and scans
//!   the f32 vectors of the best `nprobe` lists.
//!
//! ```text
//! cargo run --release -p lucene-search --example realtime_knn_tail -- /tmp/data/glove100 \
//!     --committed 290000 --sizes 1000,10000,50000,100000 --queries 1000
//! ```
//!
//! `--random D` replaces the data with `D`-dimensional Gaussian unit
//! vectors (latency only; recall there means little).

use std::time::Instant;

use lucene_codecs::hnsw::{
    HnswGraphBuilder, HnswGraphSearcher, KnnCollector, OnHeapHnswGraph, UpdateableVectorScorer,
    VectorScorer,
};
use lucene_ivf::kmeans::{self, KMeansParams, Metric};
use lucene_util::simd::dot_f32;

type R<T> = std::result::Result<T, lucene_codecs::vectors::Error>;

/// One option's search: a query's top hits as (buffer ordinal, score).
type Search<'a> = dyn FnMut(&[f32]) -> Vec<(u32, f32)> + 'a;
fn read_bin(path: &str) -> (usize, usize, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let data = bytes[8..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (n, d, data)
}

fn normalize(v: &mut [f32], dim: usize) {
    for x in v.chunks_exact_mut(dim) {
        let n = dot_f32(x, x).sqrt();
        if n > 0.0 {
            x.iter_mut().for_each(|y| *y /= n);
        }
    }
}

fn random_unit(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n * dim);
    while out.len() < n * dim {
        let (u1, u2) = (next().max(1e-12), next());
        let r = (-2.0 * u1.ln()).sqrt();
        out.push((r * (std::f64::consts::TAU * u2).cos()) as f32);
    }
    normalize(&mut out, dim);
    out
}

/// Scores buffered vectors by inner product (unit vectors: cosine).
struct BufScorer<'a> {
    data: &'a [f32],
    dim: usize,
    query: Vec<f32>,
    max_ord: i32,
}

impl VectorScorer for BufScorer<'_> {
    fn score(&mut self, node: i32) -> R<f32> {
        let o = node as usize * self.dim;
        Ok((1.0 + dot_f32(&self.query, &self.data[o..o + self.dim])) / 2.0)
    }
    fn max_ord(&self) -> i32 {
        self.max_ord
    }
}

impl UpdateableVectorScorer for BufScorer<'_> {
    fn set_scoring_ordinal(&mut self, ord: i32) -> R<()> {
        let o = ord as usize * self.dim;
        self.query.clear();
        self.query.extend_from_slice(&self.data[o..o + self.dim]);
        Ok(())
    }
}

fn top_k(scores: impl Iterator<Item = (u32, f32)>, k: usize) -> Vec<(u32, f32)> {
    let mut best: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    for (i, s) in scores {
        if best.len() < k || s > best[best.len() - 1].1 {
            let at = best.partition_point(|&(_, b)| b >= s);
            best.insert(at, (i, s));
            best.truncate(k);
        }
    }
    best
}

fn exact(buf: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<(u32, f32)> {
    top_k(
        buf.chunks_exact(dim)
            .enumerate()
            .map(|(i, v)| (i as u32, dot_f32(q, v))),
        k,
    )
}

struct Int8 {
    lo: Vec<f32>,
    step: Vec<f32>,
}

impl Int8 {
    fn train(data: &[f32], dim: usize) -> Self {
        let mut lo = vec![f32::MAX; dim];
        let mut hi = vec![f32::MIN; dim];
        for v in data.chunks_exact(dim) {
            for (j, &x) in v.iter().enumerate() {
                lo[j] = lo[j].min(x);
                hi[j] = hi[j].max(x);
            }
        }
        let step = lo
            .iter()
            .zip(&hi)
            .map(|(l, h)| (h - l).max(1e-9) / 127.0)
            .collect();
        Int8 { lo, step }
    }
    fn encode(&self, v: &[f32], out: &mut Vec<u8>) {
        for (j, &x) in v.iter().enumerate() {
            out.push((((x - self.lo[j]) / self.step[j]).round()).clamp(0.0, 127.0) as u8);
        }
    }
}

/// Asymmetric int8 scan: q . (lo + step * c) = q.lo + sum (q*step)_j c_j.
fn int8_search(
    codes: &[u8],
    buf: &[f32],
    dim: usize,
    q8: &Int8,
    q: &[f32],
    k: usize,
) -> Vec<(u32, f32)> {
    let w: Vec<f32> = q.iter().zip(&q8.step).map(|(a, s)| a * s).collect();
    let m = w.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30);
    let wq: Vec<u8> = w
        .iter()
        .map(|x| ((x * 127.0 / m).round() as i8) as u8)
        .collect();
    let cand = top_k(
        codes
            .chunks_exact(dim)
            .enumerate()
            .map(|(i, c)| (i as u32, lucene_util::simd::dot_u7_i8(c, &wq) as f32)),
        4 * k,
    );
    top_k(
        cand.into_iter().map(|(i, _)| {
            let o = i as usize * dim;
            (i, dot_f32(q, &buf[o..o + dim]))
        }),
        k,
    )
}

fn hnsw_search(
    graph: &OnHeapHnswGraph,
    buf: &[f32],
    dim: usize,
    n: usize,
    q: &[f32],
    k: usize,
    ef: usize,
) -> Vec<(u32, f32)> {
    let mut scorer = BufScorer {
        data: buf,
        dim,
        query: q.to_vec(),
        max_ord: n as i32,
    };
    let mut collector = KnnCollector::new(ef, u64::MAX);
    let mut searcher = HnswGraphSearcher::new(ef, n as i32);
    searcher
        .search(&mut collector, &mut scorer, graph, None)
        .unwrap();
    let mut hits: Vec<(u32, f32)> = collector
        .top_docs()
        .into_iter()
        .map(|(d, s)| (d as u32, s * 2.0 - 1.0))
        .collect();
    hits.sort_by(|a, b| b.1.total_cmp(&a.1));
    hits.truncate(k);
    hits
}

struct Ivf {
    centroids: Vec<f32>,
    lists: Vec<Vec<u32>>,
    /// Vectors copied in list order (what a search scans contiguously).
    vecs: Vec<Vec<f32>>,
}

impl Ivf {
    fn nearest(&self, v: &[f32], dim: usize) -> usize {
        let mut best = (0, f32::MIN);
        for (c, cen) in self.centroids.chunks_exact(dim).enumerate() {
            let s = dot_f32(v, cen);
            if s > best.1 {
                best = (c, s);
            }
        }
        best.0
    }
    fn add(
        &mut self,
        id: u32,
        v: &[f32],
        dim: usize,
        router: Option<&lucene_ivf::routing::Router>,
    ) {
        let c = match router {
            Some(r) => r.route(&self.centroids, v, 1)[0],
            None => self.nearest(v, dim),
        };
        self.lists[c].push(id);
        self.vecs[c].extend_from_slice(v);
    }
    fn search(&self, dim: usize, q: &[f32], k: usize, nprobe: usize) -> Vec<(u32, f32)> {
        let order = top_k(
            self.centroids
                .chunks_exact(dim)
                .enumerate()
                .map(|(c, cen)| (c as u32, dot_f32(q, cen))),
            nprobe,
        );
        top_k(
            order.iter().flat_map(|&(c, _)| {
                let c = c as usize;
                self.lists[c]
                    .iter()
                    .zip(self.vecs[c].chunks_exact(dim))
                    .map(|(&id, v)| (id, dot_f32(q, v)))
            }),
            k,
        )
    }
}

fn recall(got: &[(u32, f32)], truth: &[(u32, f32)]) -> f64 {
    let t: std::collections::HashSet<u32> = truth.iter().map(|x| x.0).collect();
    got.iter().filter(|x| t.contains(&x.0)).count() as f64 / truth.len().max(1) as f64
}

fn pct(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let committed: usize = opt("--committed").map_or(290_000, |s| s.parse().unwrap());
    let sizes: Vec<usize> = opt("--sizes")
        .unwrap_or_else(|| "1000,10000,50000,100000".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let nq: usize = opt("--queries").map_or(1000, |s| s.parse().unwrap());
    let chunk: usize = opt("--chunk").map_or(16384, |s| s.parse().unwrap());
    let cluster_size: usize = opt("--cluster-size").map_or(384, |s| s.parse().unwrap());
    let nprobes: Vec<usize> = opt("--nprobes")
        .unwrap_or_else(|| "8,16,32".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let ef: usize = opt("--ef").map_or(100, |s| s.parse().unwrap());
    let skip: Vec<String> = opt("--skip")
        .unwrap_or_default()
        .split(',')
        .map(String::from)
        .collect();
    let k = 10;
    let max = *sizes.iter().max().unwrap();

    let (dim, mut base, mut queries) = if let Some(d) = opt("--random") {
        let d: usize = d.parse().unwrap();
        (
            d,
            random_unit(committed + max, d, 7),
            random_unit(nq, d, 11),
        )
    } else {
        let prefix = &args[1];
        let (_, dim, base) = read_bin(&format!("{prefix}.base.fbin"));
        let (_, _, queries) = read_bin(&format!("{prefix}.query.fbin"));
        (dim, base, queries)
    };
    normalize(&mut base, dim);
    normalize(&mut queries, dim);
    queries.truncate(nq * dim);
    let nq = queries.len() / dim;
    assert!(
        base.len() / dim >= committed + max,
        "not enough base vectors"
    );
    let (old, rest) = base.split_at(committed * dim);
    let buf = &rest[..max * dim];

    // Committed side: IVF centroids and int8 ranges, trained once (what the
    // committed segments already carry).
    let t = Instant::now();
    let clustering = kmeans::cluster(
        old,
        dim,
        cluster_size,
        &KMeansParams::default(),
        Metric::InnerProduct,
    );
    let nclusters = clustering.len(dim);
    let mut centroids = clustering.centroids;
    normalize(&mut centroids, dim);
    eprintln!(
        "dim {dim}, committed {committed}, {nclusters} centroids in {:.1}s",
        t.elapsed().as_secs_f64()
    );
    let q8 = Int8::train(old, dim);

    // Arrival: every structure takes the buffer's vectors one at a time; the
    // per-vector cost is timed separately.
    let do_hnsw = !skip.iter().any(|s| s == "hnsw");
    let mut builder = HnswGraphBuilder::new(
        BufScorer {
            data: buf,
            dim,
            query: Vec::new(),
            max_ord: max as i32,
        },
        16,
        100,
        42,
    )
    .unwrap();
    let mut chunk_graphs: Vec<OnHeapHnswGraph> = Vec::new();
    let mut ivf = Ivf {
        centroids: centroids.clone(),
        lists: vec![Vec::new(); nclusters],
        vecs: vec![Vec::new(); nclusters],
    };
    let router = lucene_ivf::routing::Router::new(
        &centroids,
        dim,
        lucene_ivf::Similarity::DotProduct,
        lucene_ivf::routing::Routing::Tree { cover: 32 },
        None,
    );
    eprintln!("router tree: {}", router.has_tree());
    let mut ivf_tree = Ivf {
        centroids: centroids.clone(),
        lists: vec![Vec::new(); nclusters],
        vecs: vec![Vec::new(); nclusters],
    };
    let mut t_tree = 0f64;
    let mut codes: Vec<u8> = Vec::with_capacity(max * dim);
    let (mut t_hnsw, mut t_ivf, mut t_int8, mut t_chunks) = (0f64, 0f64, 0f64, 0f64);

    // Committed leaves, searched exactly here so only the buffer's search
    // loses recall in the merged top 10.
    let old_top: Vec<Vec<(u32, f32)>> = (0..nq)
        .map(|i| {
            exact(old, dim, &queries[i * dim..(i + 1) * dim], k)
                .into_iter()
                .map(|(d, s)| (d + max as u32, s))
                .collect()
        })
        .collect();
    let merged = |i: usize, tail: &[(u32, f32)]| {
        top_k(old_top[i].iter().copied().chain(tail.iter().copied()), k)
    };
    println!(
        "dim,buffer,option,recall@10,recall@10_merged,p50_us,p99_us,mean_us,add_us_per_vec,bytes_per_vec_extra"
    );
    let mut added = 0usize;
    for &size in &sizes {
        while added < size {
            let v = &buf[added * dim..(added + 1) * dim];
            if do_hnsw {
                let t = Instant::now();
                builder.add_graph_node(added as i32).unwrap();
                t_hnsw += t.elapsed().as_secs_f64();
            }
            let t = Instant::now();
            ivf.add(added as u32, v, dim, None);
            t_ivf += t.elapsed().as_secs_f64();
            let t = Instant::now();
            ivf_tree.add(added as u32, v, dim, Some(&router));
            t_tree += t.elapsed().as_secs_f64();
            let t = Instant::now();
            q8.encode(v, &mut codes);
            t_int8 += t.elapsed().as_secs_f64();
            added += 1;
            if do_hnsw && added.is_multiple_of(chunk) {
                let start = added - chunk;
                let t = Instant::now();
                let g = HnswGraphBuilder::new(
                    BufScorer {
                        data: &buf[start * dim..added * dim],
                        dim,
                        query: Vec::new(),
                        max_ord: chunk as i32,
                    },
                    16,
                    100,
                    42,
                )
                .unwrap()
                .build(chunk as i32)
                .unwrap();
                t_chunks += t.elapsed().as_secs_f64();
                chunk_graphs.push(g);
            }
        }
        let n = size;
        let bufn = &buf[..n * dim];
        let truths: Vec<_> = (0..nq)
            .map(|i| exact(bufn, dim, &queries[i * dim..(i + 1) * dim], k))
            .collect();
        let per = |s: f64| s * 1e6 / n as f64;
        let run = |name: &str, add_us: f64, extra: f64, f: &mut Search<'_>| {
            let mut lat = Vec::with_capacity(nq);
            let mut rec = 0.0;
            let mut rec_m = 0.0;
            for i in 0..nq {
                let q = &queries[i * dim..(i + 1) * dim];
                let t = Instant::now();
                let got = f(q);
                lat.push(t.elapsed().as_secs_f64() * 1e6);
                rec += recall(&got, &truths[i]);
                rec_m += recall(&merged(i, &got), &merged(i, &truths[i]));
            }
            let mean = lat.iter().sum::<f64>() / lat.len() as f64;
            println!(
                "{dim},{n},{name},{:.4},{:.4},{:.1},{:.1},{:.1},{:.2},{:.0}",
                rec / nq as f64,
                rec_m / nq as f64,
                pct(lat.clone(), 0.5),
                pct(lat, 0.99),
                mean,
                add_us,
                extra
            );
        };
        run("exact", 0.0, 0.0, &mut |q| exact(bufn, dim, q, k));
        run("int8", per(t_int8), dim as f64, &mut |q| {
            int8_search(&codes[..n * dim], bufn, dim, &q8, q, k)
        });
        if do_hnsw {
            let graph = builder.graph();
            // Level-0 neighbour arrays: up to 2M ints + scores per node.
            run("hnsw", per(t_hnsw), (32 * 8) as f64, &mut |q| {
                hnsw_search(graph, bufn, dim, n, q, k, ef)
            });
            let full = n / chunk;
            if full > 0 {
                let tail = full * chunk;
                run(
                    &format!("hnsw-chunks{chunk}"),
                    t_chunks * 1e6 / (full * chunk) as f64,
                    (32 * 8) as f64,
                    &mut |q| {
                        let mut all = Vec::new();
                        for (c, g) in chunk_graphs.iter().take(full).enumerate() {
                            let s = c * chunk;
                            for (d, sc) in hnsw_search(
                                g,
                                &buf[s * dim..(s + chunk) * dim],
                                dim,
                                chunk,
                                q,
                                k,
                                ef,
                            ) {
                                all.push((d + s as u32, sc));
                            }
                        }
                        let rest = exact(&bufn[tail * dim..], dim, q, k);
                        all.extend(rest.into_iter().map(|(d, s)| (d + tail as u32, s)));
                        top_k(all.into_iter(), k)
                    },
                );
            }
        }
        for &np in &nprobes {
            run(&format!("ivf-np{np}"), per(t_ivf), 4.0, &mut |q| {
                ivf.search(dim, q, k, np)
            });
            run(&format!("ivf-tree-np{np}"), per(t_tree), 4.0, &mut |q| {
                ivf_tree.search(dim, q, k, np)
            });
        }
    }
}
