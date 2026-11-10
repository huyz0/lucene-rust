//! Realtime k-NN with an HNSW graph growing in the indexing buffer while
//! searches read it: what a search pays while the inserter runs.
//!
//! One inserter thread adds buffered vectors to one graph, in arrival
//! order (the same builder and seed a flush uses, so the graph equals the
//! flush's). Documents arrive at `--rate` per second; the vectors the
//! inserter has not reached are the gap, which a search scans exactly.
//! `--searchers` threads search the whole buffer (graph + gap) in a loop.
//!
//! Modes:
//! - `rwlock`: the builder behind a `RwLock`; each insert takes the write
//!   lock, each search the read lock.
//! - `concurrent`: a [`ConcurrentHnswGraph`]: a lock per node, held while
//!   a neighbour list is copied out or rewritten; nothing graph-wide.
//! - `idle`: no inserter (the graph already holds the vectors): the floor.
//!
//! ```text
//! cargo run --release -p lucene-search --example realtime_knn_concurrent -- /tmp/data/glove100 \
//!     --buffer 50000 --rate 2000 --searchers 2 --seconds 10
//! ```

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use lucene_codecs::hnsw::{
    HnswGraphBuilder, HnswGraphSearcher, HnswGraphView, KnnCollector, UpdateableVectorScorer,
    VectorScorer,
};
use lucene_codecs::hnsw_concurrent::{ConcurrentHnswGraph, GraphInserter};
use lucene_util::simd::dot_f32;

type R<T> = Result<T, lucene_codecs::vectors::Error>;

fn read_bin(path: &str) -> (usize, usize, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let data = bytes[8..8 + n * d * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (n, d, data)
}

fn normalize(v: &mut [f32], dim: usize) {
    for x in v.chunks_exact_mut(dim) {
        let n = x.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-12);
        x.iter_mut().for_each(|a| *a /= n);
    }
}

struct BufScorer {
    data: &'static [f32],
    dim: usize,
    query: Vec<f32>,
    max_ord: i32,
}

impl VectorScorer for BufScorer {
    fn score(&mut self, node: i32) -> R<f32> {
        let o = node as usize * self.dim;
        Ok((1.0 + dot_f32(&self.query, &self.data[o..o + self.dim])) / 2.0)
    }
    fn max_ord(&self) -> i32 {
        self.max_ord
    }
}

impl UpdateableVectorScorer for BufScorer {
    fn set_scoring_ordinal(&mut self, ord: i32) -> R<()> {
        let o = ord as usize * self.dim;
        self.query.clear();
        self.query.extend_from_slice(&self.data[o..o + self.dim]);
        Ok(())
    }
}

fn top_k(best: &mut Vec<(u32, f32)>, i: u32, s: f32, k: usize) {
    if best.len() < k || s > best[best.len() - 1].1 {
        let at = best.partition_point(|&(_, b)| b >= s);
        best.insert(at, (i, s));
        best.truncate(k);
    }
}

/// The graph's top `k` over its first `n` nodes, then the gap
/// `n..arrived` scanned exactly.
fn search<G: HnswGraphView>(
    graph: &G,
    n: usize,
    buf: &'static [f32],
    dim: usize,
    arrived: usize,
    q: &[f32],
    k: usize,
) -> Vec<(u32, f32)> {
    let mut best = Vec::with_capacity(k + 1);
    if n > 0 {
        let mut scorer = BufScorer {
            data: buf,
            dim,
            query: q.to_vec(),
            max_ord: n as i32,
        };
        let mut collector = KnnCollector::new(100, u64::MAX);
        HnswGraphSearcher::new(100, n as i32)
            .search(&mut collector, &mut scorer, graph, None)
            .unwrap();
        // A node the inserter linked after `n` was read is in the gap.
        for (d, s) in collector.top_docs() {
            if (d as usize) < n {
                top_k(&mut best, d as u32, s * 2.0 - 1.0, k);
            }
        }
    }
    for i in n..arrived {
        let s = dot_f32(q, &buf[i * dim..(i + 1) * dim]);
        top_k(&mut best, i as u32, s, k);
    }
    best
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let num = |name: &str, d: usize| opt(name).map_or(d, |v| v.parse().unwrap());
    let (buffer, rate, searchers, seconds) = (
        num("--buffer", 50_000),
        num("--rate", 2000),
        num("--searchers", 2),
        num("--seconds", 10),
    );
    let mode = opt("--mode").unwrap_or_else(|| "rwlock".into());
    let prefix = &args[1];
    let (_, dim, mut base) = read_bin(&format!("{prefix}.base.fbin"));
    let (_, _, mut queries) = read_bin(&format!("{prefix}.query.fbin"));
    normalize(&mut base, dim);
    normalize(&mut queries, dim);
    let buf: &'static [f32] = Box::leak(base[..buffer * dim].to_vec().into_boxed_slice());
    let nq = queries.len() / dim;

    let builder = HnswGraphBuilder::new(
        BufScorer {
            data: buf,
            dim,
            query: Vec::new(),
            max_ord: buffer as i32,
        },
        16,
        100,
        42,
    )
    .unwrap();
    if mode == "concurrent" {
        drop(builder);
        return run_concurrent(buf, dim, buffer, rate, searchers, seconds, &queries);
    }
    let graph = Arc::new(RwLock::new(builder));
    let inserted = Arc::new(AtomicUsize::new(0));
    let arrived = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    // Half the buffer is in the graph before the clock starts.
    let start_at = buffer / 2;
    {
        let mut b = graph.write().unwrap();
        let upto = if mode == "idle" { buffer } else { start_at };
        for i in 0..upto {
            b.add_graph_node(i as i32).unwrap();
        }
        inserted.store(upto, Ordering::Release);
        arrived.store(upto, Ordering::Release);
    }
    let started = Instant::now();
    let mut threads = Vec::new();
    if mode != "idle" {
        // Arrivals at `rate` per second.
        let (arrived2, stop2) = (Arc::clone(&arrived), Arc::clone(&stop));
        threads.push(std::thread::spawn(move || {
            let (arrived, stop) = (arrived2, stop2);
            while !stop.load(Ordering::Acquire) {
                let due = start_at + (started.elapsed().as_secs_f64() * rate as f64) as usize;
                arrived.store(due.min(buffer), Ordering::Release);
                std::thread::sleep(Duration::from_millis(1));
            }
        }));
        let (graph, inserted, arrived, stop) = (
            Arc::clone(&graph),
            Arc::clone(&inserted),
            Arc::clone(&arrived),
            Arc::clone(&stop),
        );
        threads.push(std::thread::spawn(move || {
            let mut next = start_at;
            let mut write_wait = Duration::ZERO;
            let mut write_hold = Duration::ZERO;
            while !stop.load(Ordering::Acquire) && next < buffer {
                if next >= arrived.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_micros(200));
                    continue;
                }
                let t = Instant::now();
                let mut b = graph.write().unwrap();
                let got = Instant::now();
                b.add_graph_node(next as i32).unwrap();
                drop(b);
                write_wait += got - t;
                write_hold += got.elapsed();
                next += 1;
                inserted.store(next, Ordering::Release);
            }
            eprintln!(
                "inserter: write-lock wait {:.0} ms, held {:.0} ms",
                write_wait.as_secs_f64() * 1e3,
                write_hold.as_secs_f64() * 1e3
            );
        }));
    }
    let lat: Arc<std::sync::Mutex<Vec<f64>>> = Arc::default();
    let gaps: Arc<std::sync::Mutex<Vec<usize>>> = Arc::default();
    let mut searchers_h = Vec::new();
    for s in 0..searchers {
        let (graph, inserted, arrived, stop, lat, gaps) = (
            Arc::clone(&graph),
            Arc::clone(&inserted),
            Arc::clone(&arrived),
            Arc::clone(&stop),
            Arc::clone(&lat),
            Arc::clone(&gaps),
        );
        let queries = queries.clone();
        searchers_h.push(std::thread::spawn(move || {
            let mut qi = s;
            let mut mine = Vec::new();
            let mut my_gaps = Vec::new();
            while !stop.load(Ordering::Acquire) {
                let q = &queries[(qi % nq) * dim..(qi % nq + 1) * dim];
                let t = Instant::now();
                let g = graph.read().unwrap();
                let n = inserted.load(Ordering::Acquire);
                let a = arrived.load(Ordering::Acquire).max(n);
                let hits = search(g.graph(), n, buf, dim, a, q, 10);
                drop(g);
                mine.push(t.elapsed().as_secs_f64() * 1e6);
                my_gaps.push(a - n);
                std::hint::black_box(hits);
                qi += searchers;
            }
            lat.lock().unwrap().extend(mine);
            gaps.lock().unwrap().extend(my_gaps);
        }));
    }
    std::thread::sleep(Duration::from_secs(seconds as u64));
    stop.store(true, Ordering::Release);
    for t in searchers_h {
        t.join().unwrap();
    }
    for t in threads {
        let _ = t.join();
    }
    let mut lat = lat.lock().unwrap().clone();
    let mut gaps = gaps.lock().unwrap().clone();
    lat.sort_by(|a, b| a.total_cmp(b));
    gaps.sort_unstable();
    let p = |v: &[f64], q: f64| v[((v.len() - 1) as f64 * q) as usize];
    let ins = inserted.load(Ordering::Acquire) - if mode == "idle" { buffer } else { start_at };
    println!(
        "mode {mode} dim {dim} buffer {buffer} rate {rate}/s searchers {searchers}: {} searches, p50 {:.0} us, p99 {:.0} us; inserted {ins} ({:.0}/s), gap p50 {} p99 {}",
        lat.len(),
        p(&lat, 0.5),
        p(&lat, 0.99),
        ins as f64 / seconds as f64,
        gaps[gaps.len() / 2],
        gaps[(gaps.len() - 1) * 99 / 100],
    );
}

fn run_concurrent(
    buf: &'static [f32],
    dim: usize,
    buffer: usize,
    rate: usize,
    searchers: usize,
    seconds: usize,
    queries: &[f32],
) {
    let nq = queries.len() / dim;
    let graph: &'static ConcurrentHnswGraph =
        Box::leak(Box::new(ConcurrentHnswGraph::growable(16)));
    let scorer = || BufScorer {
        data: buf,
        dim,
        query: Vec::new(),
        max_ord: buffer as i32,
    };
    let mut inserter = GraphInserter::new(graph, scorer(), 16, 100, 42).unwrap();
    let start_at = buffer / 2;
    for i in 0..start_at {
        inserter.insert(i as i32).unwrap();
    }
    let inserted = Arc::new(AtomicUsize::new(start_at));
    let arrived = Arc::new(AtomicUsize::new(start_at));
    let stop = Arc::new(AtomicBool::new(false));
    let started = Instant::now();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            while !stop.load(Ordering::Acquire) {
                let due = start_at + (started.elapsed().as_secs_f64() * rate as f64) as usize;
                arrived.store(due.min(buffer), Ordering::Release);
                std::thread::sleep(Duration::from_millis(1));
            }
        });
        scope.spawn(|| {
            let mut next = start_at;
            while !stop.load(Ordering::Acquire) && next < buffer {
                if next >= arrived.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_micros(200));
                    continue;
                }
                inserter.insert(next as i32).unwrap();
                next += 1;
                inserted.store(next, Ordering::Release);
            }
        });
        let handles: Vec<_> = (0..searchers)
            .map(|s| {
                let (arrived, stop) = (&arrived, &stop);
                scope.spawn(move || {
                    let (mut lat, mut gaps) = (Vec::new(), Vec::new());
                    let mut qi = s;
                    while !stop.load(Ordering::Acquire) {
                        let q = &queries[(qi % nq) * dim..(qi % nq + 1) * dim];
                        let t = Instant::now();
                        let published = graph.published();
                        let n = published.len() as usize;
                        let a = arrived.load(Ordering::Acquire).max(n);
                        std::hint::black_box(search(&published, n, buf, dim, a, q, 10));
                        lat.push(t.elapsed().as_secs_f64() * 1e6);
                        gaps.push(a - n);
                        qi += searchers;
                    }
                    (lat, gaps)
                })
            })
            .collect();
        std::thread::sleep(Duration::from_secs(seconds as u64));
        stop.store(true, Ordering::Release);
        let (mut lat, mut gaps) = (Vec::new(), Vec::new());
        for h in handles {
            let (l, g) = h.join().unwrap();
            lat.extend(l);
            gaps.extend(g);
        }
        lat.sort_by(|a, b| a.total_cmp(b));
        gaps.sort_unstable();
        let p = |v: &[f64], q: f64| v[((v.len() - 1) as f64 * q) as usize];
        let ins = inserted.load(Ordering::Acquire) - start_at;
        println!(
            "mode concurrent dim {dim} buffer {buffer} rate {rate}/s searchers {searchers}: {} searches, p50 {:.0} us, p99 {:.0} us; inserted {ins} ({:.0}/s), gap p50 {} p99 {}",
            lat.len(),
            p(&lat, 0.5),
            p(&lat, 0.99),
            ins as f64 / seconds as f64,
            gaps[gaps.len() / 2],
            gaps[(gaps.len() - 1) * 99 / 100],
        );
    });
}
