//! Recall, latency and I/O of HNSW layouts and IVF under a memory budget.
//!
//! ```text
//! cargo run --release -p lucene-disk-hnsw --example disk_hnsw_bench -- <prefix> <l2|cosine> [options]
//!   --base N        use the first N base vectors (ground truth must match: default all)
//!   --queries N     measured queries (default 1000), after as many warm-up queries
//!   --budgets L     comma list of memory budgets in MiB, or inf (default 16,32,64,128,256,inf)
//!   --lists L       disk HNSW list lengths and stock HNSW ef (default 40,80,160)
//!   --beams L       disk HNSW beam widths (default 4)
//!   --nprobe L      IVF probes (default 32,64,128)
//!   --indexes L     comma list of disk,lean,stock-bbq,stock-bbq-pinned,stock-f32,ivf
//!                   (default all)
//!   --encoding E    disk HNSW record vectors: f32|f16|bf16|int8 (default f32)
//! ```
//!
//! Every index runs over the same files a segment would hold, read through
//! one simulated page cache per run: 4 KiB pages, CLOCK eviction, holding the
//! budget minus what the index keeps in memory regardless (its pinned
//! structures). A budget smaller than the pinned structures is not run.
//! The cache counts the pages it fetches (misses) and the round trips a
//! search waits for: a read that misses is one round trip, and so are all the
//! ranges announced together with `prefetch` before they are read.
//!
//! - `disk`: this crate's layout, BBQ codes in memory, records paged.
//! - `lean`: the same with inline neighbour codes (LM-DiskANN) and no codes
//!   in memory.
//! - `stock-bbq`: stock Lucene's layout (`Lucene99HnswVectorsFormat` over
//!   `Lucene104ScalarQuantizedVectorsFormat` 1-bit): level-0 adjacency as
//!   `.vex` stores it (sorted vint deltas, offsets in memory), codes in node
//!   order (`.veq`), float vectors in node order (`.vec`), all paged. The
//!   search is Lucene's (`HnswGraphSearcher`) on the codes, then an exact
//!   rerank of the `ef` results; each expanded node's neighbours' codes are
//!   prefetched together, which is the best a page cache can do for it.
//! - `stock-bbq-pinned`: the same with `.vex` and `.veq` pinned in memory,
//!   only `.vec` paged.
//! - `stock-f32`: stock float HNSW, scoring neighbours on `.vec`.
//! - `ivf`: `lucene-ivf` with BBQ codes, target 384 vectors per cluster,
//!   rerank 100 (`.ivm` pinned, `.ivp` and `.ivv` paged).
//!
//! All HNSW variants share one graph (Lucene's builder, M 16, beam 100, on
//! full-precision vectors). Prints one CSV row per index, setting and
//! budget: `est_nvme_us` and `est_ebs_us` add 100 us and 500 us per round
//! trip to the measured CPU time.
#![allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]

use std::borrow::Cow;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use lucene_codecs::hnsw::{HnswGraphView, OnHeapHnswGraph};
use lucene_disk_hnsw::{
    build_graph, DiskHnswParams, DiskHnswReader, DiskHnswWriter, FieldVectors, GraphParams,
    OpenOptions, RangeSource, SearchParams, Similarity, VectorEncoding,
};
use lucene_ivf::{IvfParams, IvfReader, IvfWriter};
use lucene_store::DataOutput;
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::{Quantizer, QuantizerFactory, QuantizerRegistry};

const ID: [u8; 16] = *b"disk-hnsw-bench!";
const K: usize = 10;
const PAGE: usize = 4096;

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

#[derive(Default, Clone, Copy)]
struct Counters {
    misses: u64,
    round_trips: u64,
}

struct CacheState {
    map: HashMap<u64, usize>,
    slots: Vec<(u64, bool)>,
    hand: usize,
    pending: bool,
    counters: Counters,
}

/// A page cache of `capacity` pages with CLOCK eviction, shared by the
/// files of one index.
struct Cache {
    capacity: usize,
    state: Mutex<CacheState>,
}

impl Cache {
    fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Cache {
            capacity,
            state: Mutex::new(CacheState {
                map: HashMap::new(),
                slots: Vec::new(),
                hand: 0,
                pending: false,
                counters: Counters::default(),
            }),
        })
    }

    fn touch(&self, file: u64, offset: usize, len: usize, prefetch: bool) {
        let mut s = self.state.lock().unwrap();
        let mut missed = false;
        let first = offset / PAGE;
        let last = (offset + len.max(1) - 1) / PAGE;
        for page in first..=last {
            let key = file << 40 | page as u64;
            if let Some(&slot) = s.map.get(&key) {
                s.slots[slot].1 = true;
                continue;
            }
            missed = true;
            s.counters.misses += 1;
            if self.capacity == 0 {
                continue;
            }
            if s.slots.len() < self.capacity {
                let slot = s.slots.len();
                s.slots.push((key, true));
                s.map.insert(key, slot);
                continue;
            }
            loop {
                let hand = s.hand;
                s.hand = (hand + 1) % self.capacity;
                if s.slots[hand].1 {
                    s.slots[hand].1 = false;
                } else {
                    let old = s.slots[hand].0;
                    s.map.remove(&old);
                    s.slots[hand] = (key, true);
                    s.map.insert(key, hand);
                    break;
                }
            }
        }
        if prefetch {
            s.pending |= missed;
        } else if missed || s.pending {
            s.counters.round_trips += 1;
            s.pending = false;
        }
    }

    /// Loads every page of a file, as an unbounded budget holds it.
    fn preload(&self, file: u64, len: usize) {
        self.touch(file, 0, len, false);
        self.take();
    }

    fn take(&self) -> Counters {
        let mut s = self.state.lock().unwrap();
        std::mem::take(&mut s.counters)
    }
}

/// A file read through a [`Cache`].
struct Paged {
    id: u64,
    bytes: Arc<Vec<u8>>,
    cache: Arc<Cache>,
}

impl RangeSource for Paged {
    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn read(&self, offset: usize, len: usize) -> lucene_ivf::Result<Cow<'_, [u8]>> {
        self.cache.touch(self.id, offset, len, false);
        Ok(Cow::Borrowed(&self.bytes[offset..offset + len]))
    }

    fn prefetch(&self, offset: usize, len: usize) {
        self.cache.touch(self.id, offset, len, true);
    }
}

fn paged(id: u64, bytes: &Arc<Vec<u8>>, cache: &Arc<Cache>) -> Arc<Paged> {
    if cache.capacity > bytes.len() / PAGE {
        cache.preload(id, bytes.len());
    }
    Arc::new(Paged {
        id,
        bytes: bytes.clone(),
        cache: cache.clone(),
    })
}

/// Stock Lucene's HNSW files and search, read through a cache.
struct Stock<'g> {
    graph: &'g OnHeapHnswGraph,
    similarity: Similarity,
    dim: usize,
    quantizer: Option<(&'g dyn Quantizer, &'g [f32])>,
    code_len: usize,
    vex_offsets: &'g [usize],
    vex: Arc<dyn RangeSource>,
    veq: Arc<dyn RangeSource>,
    vec: Arc<dyn RangeSource>,
}

#[derive(PartialEq)]
struct Scored(f32, u32);
impl Eq for Scored {}
impl PartialOrd for Scored {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Scored {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0).then(o.1.cmp(&self.1))
    }
}

impl Stock<'_> {
    fn vector<'a>(&'a self, node: u32) -> Cow<'a, [u8]> {
        let len = self.dim * 4;
        self.vec.read(node as usize * len, len).unwrap()
    }

    fn exact(&self, q: &[f32], node: u32) -> f32 {
        let b = self.vector(node);
        let v: Vec<f32> = b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        self.similarity.score(q, &v)
    }

    fn prefetch_scores(&self, nodes: &[u32]) {
        for &n in nodes {
            match self.quantizer {
                Some(_) => self.veq.prefetch(n as usize * self.code_len, self.code_len),
                None => self.vec.prefetch(n as usize * self.dim * 4, self.dim * 4),
            }
        }
    }

    fn score(
        &self,
        q: &[f32],
        scorer: &Option<Box<dyn lucene_vector_quant::QueryScorer + '_>>,
        node: u32,
    ) -> f32 {
        match scorer {
            Some(s) => s.score(
                &self
                    .veq
                    .read(node as usize * self.code_len, self.code_len)
                    .unwrap(),
            ),
            None => self.exact(q, node),
        }
    }

    fn level0(&self, node: u32) -> Vec<u32> {
        let at = self.vex_offsets[node as usize];
        let len = self.vex_offsets[node as usize + 1] - at;
        let bytes = self.vex.read(at, len).unwrap();
        let mut input = lucene_store::SliceInput::new(&bytes);
        use lucene_store::DataInput;
        let count = input.read_vint().unwrap() as usize;
        let mut out = Vec::with_capacity(count);
        let mut prev = 0u32;
        for _ in 0..count {
            prev += input.read_vint().unwrap() as u32;
            out.push(prev);
        }
        out
    }

    fn search(&self, q: &[f32], ef: usize) -> Vec<u32> {
        let scorer = self.quantizer.map(|(quant, prepared)| {
            let mut s = quant.query_scorer(q);
            s.set_prepared_centroid(prepared);
            s
        });
        let mut ep = self.graph.entry_node() as u32;
        let mut ep_score = self.score(q, &scorer, ep);
        for level in (1..self.graph.num_levels()).rev() {
            loop {
                let nbs: Vec<u32> = self
                    .graph
                    .neighbors(level, ep as i32)
                    .nodes()
                    .iter()
                    .map(|&n| n as u32)
                    .collect();
                self.prefetch_scores(&nbs);
                let mut moved = false;
                for nb in nbs {
                    let s = self.score(q, &scorer, nb);
                    if s > ep_score {
                        ep_score = s;
                        ep = nb;
                        moved = true;
                    }
                }
                if !moved {
                    break;
                }
            }
        }
        let n = self.graph.size() as usize;
        let mut visited = vec![false; n];
        visited[ep as usize] = true;
        let mut candidates = BinaryHeap::new();
        let mut results: BinaryHeap<Reverse<Scored>> = BinaryHeap::new();
        candidates.push(Scored(ep_score, ep));
        results.push(Reverse(Scored(ep_score, ep)));
        while let Some(Scored(s, node)) = candidates.pop() {
            if results.len() >= ef && s < results.peek().unwrap().0 .0 {
                break;
            }
            let nbs: Vec<u32> = self
                .level0(node)
                .into_iter()
                .filter(|&nb| !std::mem::replace(&mut visited[nb as usize], true))
                .collect();
            self.prefetch_scores(&nbs);
            for nb in nbs {
                let s = self.score(q, &scorer, nb);
                if results.len() < ef || s > results.peek().unwrap().0 .0 {
                    candidates.push(Scored(s, nb));
                    results.push(Reverse(Scored(s, nb)));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        let nodes: Vec<u32> = results.into_iter().map(|Reverse(Scored(_, n))| n).collect();
        let mut exact: Vec<(f32, u32)> = if scorer.is_some() {
            for &n in &nodes {
                self.vec.prefetch(n as usize * self.dim * 4, self.dim * 4);
            }
            nodes.iter().map(|&n| (self.exact(q, n), n)).collect()
        } else {
            nodes.iter().map(|&n| (self.exact(q, n), n)).collect()
        };
        exact.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        exact.into_iter().take(K).map(|e| e.1).collect()
    }
}

struct Row {
    index: &'static str,
    config: String,
    budget: String,
    pinned_mb: f64,
    paged_mb: f64,
    recall: f64,
    cpu_us: f64,
    misses: f64,
    round_trips: f64,
}

fn print(r: &Row) {
    println!(
        "{},{},{},{:.1},{:.1},{:.4},{:.0},{:.1},{:.1},{:.0},{:.0}",
        r.index,
        r.config,
        r.budget,
        r.pinned_mb,
        r.paged_mb,
        r.recall,
        r.cpu_us,
        r.misses,
        r.round_trips,
        r.cpu_us + r.round_trips * 100.0,
        r.cpu_us + r.round_trips * 500.0
    );
}

/// Runs `search` over the warm-up then the measured queries, returning
/// recall, CPU microseconds and the cache's counts per measured query.
fn measure<'t>(
    cache: &Cache,
    queries: &[f32],
    dim: usize,
    nq: usize,
    truth: &dyn Fn(usize) -> &'t [i32],
    search: &mut dyn FnMut(&[f32]) -> Vec<i32>,
) -> (f64, f64, f64, f64) {
    for i in 0..nq {
        search(&queries[(nq + i) * dim..][..dim]);
    }
    cache.take();
    let mut found = 0;
    let t = Instant::now();
    for i in 0..nq {
        let hits = search(&queries[i * dim..][..dim]);
        found += hits.iter().filter(|d| truth(i)[..K].contains(d)).count();
    }
    let secs = t.elapsed().as_secs_f64();
    let c = cache.take();
    (
        found as f64 / (nq * K) as f64,
        secs / nq as f64 * 1e6,
        c.misses as f64 / nq as f64,
        c.round_trips as f64 / nq as f64,
    )
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
    let list = |s: String| -> Vec<usize> { s.split(',').map(|x| x.parse().unwrap()).collect() };
    let (n_all, dim, mut base) = read_bin(&format!("{prefix}.base.fbin"), f32::from_le_bytes);
    let (nq_all, _, mut queries) = read_bin(&format!("{prefix}.query.fbin"), f32::from_le_bytes);
    let (_, gt_k, gt) = read_bin(&format!("{prefix}.gt.ibin"), i32::from_le_bytes);
    let n = opt("--base")
        .map_or(n_all, |s| s.parse().unwrap())
        .min(n_all);
    let nq = opt("--queries")
        .map_or(1000, |s| s.parse().unwrap())
        .min(nq_all / 2);
    let budgets: Vec<Option<usize>> = opt("--budgets")
        .unwrap_or_else(|| "16,32,64,128,256,inf".into())
        .split(',')
        .map(|b| (b != "inf").then(|| b.parse::<usize>().unwrap() << 20))
        .collect();
    let lists = list(opt("--lists").unwrap_or_else(|| "40,80,160".into()));
    let beams = list(opt("--beams").unwrap_or_else(|| "4".into()));
    let nprobes = list(opt("--nprobe").unwrap_or_else(|| "32,64,128".into()));
    let indexes = opt("--indexes")
        .unwrap_or_else(|| "disk,lean,stock-bbq,stock-bbq-pinned,stock-f32,ivf".into());
    let want = |name: &str| indexes.split(',').any(|i| i == name);
    let encoding = opt("--encoding").map_or(VectorEncoding::F32, |s| {
        VectorEncoding::from_name(&s).expect("--encoding")
    });
    base.truncate(n * dim);
    if similarity == Similarity::Cosine {
        // Stored unit length, as the disk layout and IVF store them.
        for v in base
            .chunks_exact_mut(dim)
            .chain(queries.chunks_exact_mut(dim))
        {
            let _ = lucene_util::vector_util::l2normalize(v, false);
        }
    }
    let docs: Vec<i32> = (0..n as i32).collect();
    let truth = |q: usize| &gt[q * gt_k..(q + 1) * gt_k];
    eprintln!("{prefix}: {n} x {dim}, {nq} queries, {similarity:?}");

    let t = Instant::now();
    let graph = build_graph(&base, dim, similarity, &GraphParams::default()).unwrap();
    eprintln!("graph built in {:.1}s", t.elapsed().as_secs_f64());
    let field = FieldVectors {
        number: 0,
        similarity,
        dimension: dim,
        docs: &docs,
        vectors: &base,
    };
    let registry = QuantizerRegistry::with_builtins();
    let bbq = OsqFactory::bbq();
    println!(
        "index,config,budget_mib,pinned_mb,paged_mb,recall@10,cpu_us,misses,round_trips,est_nvme_us,est_ebs_us"
    );
    let budget_name = |b: Option<usize>| b.map_or("inf".to_string(), |b| (b >> 20).to_string());

    // This crate's layout, with resident codes and lean.
    for (name, inline) in [("disk", false), ("lean", true)] {
        if !want(name) {
            continue;
        }
        let mut w = DiskHnswWriter::new(&ID, "");
        let params = DiskHnswParams {
            inline_codes: inline,
            encoding,
            ..DiskHnswParams::default()
        };
        let t = Instant::now();
        w.add_field_with_graph(&field, &graph, &bbq, &params)
            .unwrap();
        let files = w.finish();
        eprintln!(
            "{name}: written in {:.1}s, nodes {} MB",
            t.elapsed().as_secs_f64(),
            files.nodes.len() / 1_000_000
        );
        let nodes = Arc::new(files.nodes);
        for &budget in &budgets {
            let probe = DiskHnswReader::open(
                &files.meta,
                Arc::new(files.codes.clone()),
                paged(1, &nodes, &Cache::new(0)),
                &ID,
                "",
                &registry,
                OpenOptions {
                    resident_codes: !inline,
                },
            )
            .unwrap();
            let pinned = probe.resident_bytes();
            drop(probe);
            let Some(capacity) = budget.map_or(Some(usize::MAX / PAGE), |b| {
                b.checked_sub(pinned).map(|c| c / PAGE)
            }) else {
                continue;
            };
            let cache = Cache::new(capacity.min(nodes.len() / PAGE + 1));
            let reader = DiskHnswReader::open(
                &files.meta,
                Arc::new(files.codes.clone()),
                paged(1, &nodes, &cache),
                &ID,
                "",
                &registry,
                OpenOptions {
                    resident_codes: !inline,
                },
            )
            .unwrap();
            for (&l, &beam) in lists.iter().flat_map(|l| beams.iter().map(move |b| (l, b))) {
                let params = SearchParams {
                    k: K,
                    list_len: l,
                    beam_width: beam,
                };
                let mut expanded = 0usize;
                let mut estimated = 0usize;
                let (recall, cpu_us, misses, round_trips) =
                    measure(&cache, &queries, dim, nq, &truth, &mut |q| {
                        let out = reader.search(0, q, &params, None).unwrap();
                        expanded += out.stats.expanded;
                        estimated += out.stats.estimated;
                        out.hits.iter().map(|h| h.doc).collect()
                    });
                // Counted over the warm-up and the measured queries.
                let (expanded, estimated) = (expanded / (2 * nq), estimated / (2 * nq));
                print(&Row {
                    index: name,
                    config: format!(
                        "L={l} W={beam} enc={} expanded={expanded} estimated={estimated}",
                        encoding.name()
                    ),
                    budget: budget_name(budget),
                    pinned_mb: pinned as f64 / 1e6,
                    paged_mb: nodes.len() as f64 / 1e6,
                    recall,
                    cpu_us,
                    misses,
                    round_trips,
                });
            }
        }
    }

    // Stock Lucene's layout.
    let quantizer = bbq
        .train(dim, similarity, &base[..base.len().min(25_000 * dim)])
        .unwrap();
    let mut centroid = vec![0f32; dim];
    for v in base.chunks_exact(dim) {
        for (c, x) in centroid.iter_mut().zip(v) {
            *c += x / n as f32;
        }
    }
    let prepared = quantizer
        .prepare_centroid(&centroid)
        .unwrap_or(centroid.clone());
    let code_len = quantizer.code_len();
    let mut veq = vec![0u8; n * code_len];
    for (out, v) in veq.chunks_exact_mut(code_len).zip(base.chunks_exact(dim)) {
        quantizer.encode_prepared(v, &prepared, out);
    }
    let mut vex = Vec::new();
    let mut vex_offsets = Vec::with_capacity(n + 1);
    for node in 0..n {
        vex_offsets.push(vex.len());
        let mut nbs: Vec<i32> = graph.neighbors(0, node as i32).nodes().to_vec();
        nbs.sort_unstable();
        vex.write_vint(nbs.len() as i32);
        let mut prev = 0;
        for nb in nbs {
            vex.write_vint(nb - prev);
            prev = nb;
        }
    }
    vex_offsets.push(vex.len());
    let vec_bytes: Vec<u8> = base.iter().flat_map(|x| x.to_le_bytes()).collect();
    let (vex, veq, vec_bytes) = (Arc::new(vex), Arc::new(veq), Arc::new(vec_bytes));
    for (name, quantized, pin) in [
        ("stock-bbq", true, false),
        ("stock-bbq-pinned", true, true),
        ("stock-f32", false, false),
    ] {
        if !want(name) {
            continue;
        }
        // Offsets (a few bits a node in `DirectMonotonic`) and the upper
        // levels are counted as 4 bytes a node.
        let mut pinned = n * 4;
        let mut paged_bytes = vec_bytes.len() + vex.len();
        if quantized {
            paged_bytes += veq.len();
        }
        if pin {
            pinned += vex.len() + veq.len();
            paged_bytes = vec_bytes.len();
        }
        for &budget in &budgets {
            let Some(capacity) = budget.map_or(Some(usize::MAX / PAGE), |b| {
                b.checked_sub(pinned).map(|c| c / PAGE)
            }) else {
                continue;
            };
            let cache = Cache::new(capacity.min(paged_bytes / PAGE + 3));
            let memory = Cache::new(usize::MAX / PAGE);
            let (vex_src, veq_src): (Arc<dyn RangeSource>, Arc<dyn RangeSource>) = if pin {
                (paged(2, &vex, &memory), paged(3, &veq, &memory))
            } else {
                (paged(2, &vex, &cache), paged(3, &veq, &cache))
            };
            let stock = Stock {
                graph: &graph,
                similarity,
                dim,
                quantizer: quantized.then(|| (quantizer.as_ref(), prepared.as_slice())),
                code_len,
                vex_offsets: &vex_offsets,
                vex: vex_src,
                veq: veq_src,
                vec: paged(4, &vec_bytes, &cache),
            };
            for &ef in &lists {
                let (recall, cpu_us, misses, round_trips) =
                    measure(&cache, &queries, dim, nq, &truth, &mut |q| {
                        stock.search(q, ef).into_iter().map(|n| n as i32).collect()
                    });
                print(&Row {
                    index: name,
                    config: format!("ef={ef}"),
                    budget: budget_name(budget),
                    pinned_mb: pinned as f64 / 1e6,
                    paged_mb: paged_bytes as f64 / 1e6,
                    recall,
                    cpu_us,
                    misses,
                    round_trips,
                });
            }
        }
    }

    if want("ivf") {
        let t = Instant::now();
        let mut w = IvfWriter::new(&ID, "");
        w.add_field(
            &lucene_ivf::FieldVectors {
                number: 0,
                similarity,
                dimension: dim,
                docs: &docs,
                vectors: &base,
            },
            &bbq,
            &IvfParams::default(),
        )
        .unwrap();
        let files = w.finish();
        eprintln!("ivf: built in {:.1}s", t.elapsed().as_secs_f64());
        let postings = Arc::new(files.postings);
        let vectors = Arc::new(files.vectors);
        let pinned = files.meta.len();
        let paged_bytes = postings.len() + vectors.len();
        for &budget in &budgets {
            let Some(capacity) = budget.map_or(Some(usize::MAX / PAGE), |b| {
                b.checked_sub(pinned).map(|c| c / PAGE)
            }) else {
                continue;
            };
            let cache = Cache::new(capacity.min(paged_bytes / PAGE + 2));
            let reader = IvfReader::open(
                &files.meta,
                paged(5, &postings, &cache),
                paged(6, &vectors, &cache),
                &ID,
                "",
                &registry,
            )
            .unwrap();
            for &nprobe in &nprobes {
                let params = lucene_ivf::SearchParams::new(K, nprobe, 100);
                let (recall, cpu_us, misses, round_trips) =
                    measure(&cache, &queries, dim, nq, &truth, &mut |q| {
                        reader
                            .search(0, q, &params, None)
                            .unwrap()
                            .iter()
                            .map(|h| h.doc)
                            .collect()
                    });
                print(&Row {
                    index: "ivf",
                    config: format!("nprobe={nprobe} rerank=100"),
                    budget: budget_name(budget),
                    pinned_mb: pinned as f64 / 1e6,
                    paged_mb: paged_bytes as f64 / 1e6,
                    recall,
                    cpu_us,
                    misses,
                    round_trips,
                });
            }
        }
    }
}
