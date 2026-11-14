//! Prototype pieces for a sparse-retrieval experiment (not wired into the plugin):
//!
//! - [`ClusterStore`]: Seismic's clusters with a copy of each cluster's
//!   vectors stored next to each other (OpenSearch 3.9's `per_block` forward
//!   index), searched by [`ClusterStore::search`].
//! - [`BlockIndex`]: documents reordered by recursive graph bisection and cut
//!   into blocks of `b` documents and superblocks of `c` blocks, each with
//!   per-term maximum weights, searched with superblock then block pruning
//!   (BMP / superblock pruning / lightweight superblock pruning).
//! - a synthetic corpus and an exact search to measure against.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc
)]

use crate::postings::{self, ClusteredPostings, FieldClusters};
use crate::search::{ForwardVectors, ScoreHeap, SeismicField, SeismicQuery};
use crate::vector::SparseVector;
use crate::writer::FieldClusterer;
use lucene_codecs::field_infos::FieldInfo;
use lucene_store::Input;
use lucene_util::fixed_bit_set::FixedBitSet;
use std::collections::HashMap;
use std::sync::Arc;

pub const VOCAB: usize = 30000;
pub const ID: [u8; 16] = [9; 16];

/// xorshift64*.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f64()
    }
}

pub type Pairs = Vec<(u32, f32)>;

/// `random.paretovariate(0.6)`, as the plugin bench draws tokens.
fn pareto_token(rng: &mut Rng) -> u32 {
    let u = 1.0 - rng.f64();
    let x = 1.0 / u.powf(1.0 / 0.6);
    ((x as usize).saturating_sub(1)).min(VOCAB - 1) as u32
}

fn push_unique(v: &mut Pairs, t: u32, w: f32) -> bool {
    if v.iter().any(|&(x, _)| x == t) {
        return false;
    }
    v.push((t, w));
    true
}

pub struct Corpus {
    pub docs: Vec<Pairs>,
    pub queries: Vec<Pairs>,
}

/// The plugin bench's corpus: heavy-tailed tokens, no structure between documents.
pub fn flat_corpus(n: usize, nq: usize, seed: u64) -> Corpus {
    let mut rng = Rng(seed);
    let mut make = |nnz: usize| {
        let mut v = Pairs::new();
        while v.len() < nnz {
            let t = pareto_token(&mut rng);
            let w = rng.range(0.01, 3.5) as f32;
            push_unique(&mut v, t, w);
        }
        v
    };
    let mut docs = Vec::with_capacity(n);
    let mut sizes = Rng(seed ^ 77);
    for _ in 0..n {
        let nnz = 60 + sizes.below(101);
        docs.push(make(nnz));
    }
    let queries = (0..nq).map(|_| make(10 + sizes.below(21))).collect();
    Corpus { docs, queries }
}

/// Documents about a topic share a pool of tokens, as real text does; the
/// same head tokens appear everywhere.
pub fn topical_corpus(n: usize, nq: usize, topics: usize, seed: u64) -> Corpus {
    let mut rng = Rng(seed);
    let pools: Vec<Vec<u32>> = (0..topics)
        .map(|_| {
            let mut p: Vec<u32> = Vec::new();
            while p.len() < 300 {
                let t = pareto_token(&mut rng);
                if !p.contains(&t) {
                    p.push(t);
                }
            }
            p
        })
        .collect();
    let mut make = |rng: &mut Rng, topic: usize, nnz: usize| {
        let mut v = Pairs::new();
        while v.len() < nnz {
            if rng.f64() < 0.65 {
                let t = pools[topic][rng.below(300)];
                push_unique(&mut v, t, rng.range(0.5, 3.5) as f32);
            } else {
                let t = pareto_token(rng);
                push_unique(&mut v, t, rng.range(0.01, 1.5) as f32);
            }
        }
        v
    };
    let mut docs = Vec::with_capacity(n);
    for _ in 0..n {
        let topic = rng.below(topics);
        let nnz = 60 + rng.below(101);
        docs.push(make(&mut rng, topic, nnz));
    }
    let queries = (0..nq)
        .map(|_| {
            let topic = rng.below(topics);
            let nnz = 10 + rng.below(21);
            make(&mut rng, topic, nnz)
        })
        .collect();
    Corpus { docs, queries }
}

pub fn field(n_postings: i32) -> (FieldInfo, SeismicField) {
    let info = FieldInfo::new("emb", 1).with_attributes(vec![
        ("sparse_vector_field".into(), "true".into()),
        ("approximate_threshold".into(), "10000".into()),
        ("n_postings".into(), n_postings.to_string()),
    ]);
    let f = SeismicField::from_field_info(&info).unwrap().unwrap();
    (info, f)
}

pub fn forward_bytes(pairs: &Pairs) -> Vec<u8> {
    let mut b = Vec::with_capacity(pairs.len() * 8);
    for &(t, w) in pairs {
        b.extend_from_slice(&(t as i32).to_be_bytes());
        b.extend_from_slice(&w.to_be_bytes());
    }
    b
}

/// A built Seismic segment: what the plugin searches today, plus the clusters
/// in memory for the variants.
pub struct SeismicIndex {
    pub field: SeismicField,
    pub postings: ClusteredPostings,
    pub forward: ForwardVectors,
    pub clusters: FieldClusters,
    pub sit_bytes: usize,
    pub sip_bytes: usize,
    pub max_doc: i32,
}

pub fn build_seismic(docs: &[Pairs], n_postings: i32) -> SeismicIndex {
    let (info, field) = field(n_postings);
    let max_doc = docs.len() as i32;
    let mut cl = FieldClusterer::new(&info, max_doc).unwrap();
    let mut vectors = Vec::with_capacity(docs.len());
    for (d, p) in docs.iter().enumerate() {
        cl.add(d as i32, &forward_bytes(p)).unwrap();
        vectors.push(Some(SparseVector::quantized(p, field.ceiling_ingest)));
    }
    let clusters = cl.finish(&ID);
    let (sit, sip) = postings::write(&ID, std::slice::from_ref(&clusters));
    let (sit_bytes, sip_bytes) = (sit.len(), sip.len());
    let postings = ClusteredPostings::open(&sit, Arc::new(Input::external(sip)), &ID).unwrap();
    SeismicIndex {
        field,
        postings,
        forward: ForwardVectors::from_vectors(&vectors),
        clusters,
        sit_bytes,
        sip_bytes,
        max_doc,
    }
}

/// Where a Seismic search reads a scored document's vector from.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// One copy of every vector, ordered by document id (what the plugin does).
    Shared,
    /// Each cluster's vectors next to each other.
    PerCluster,
}

struct ClInfo {
    cd_begin: usize,
    n: usize,
    skip: bool,
    summary: Option<SparseVector>,
    vec_begin: usize,
    vec_end: usize,
}

/// Seismic's clusters, with a copy of each cluster's vectors.
pub struct ClusterStore {
    by_term: HashMap<Vec<u8>, (usize, usize)>,
    infos: Vec<ClInfo>,
    cd_doc: Vec<i32>,
    cd_start: Vec<usize>,
    tokens: Vec<u16>,
    weights: Vec<u8>,
}

fn page(region: usize, byte: usize) -> usize {
    (region << 40) | (byte / 4096)
}

impl ClusterStore {
    pub fn build(idx: &SeismicIndex) -> Self {
        let mut s = ClusterStore {
            by_term: HashMap::new(),
            infos: Vec::new(),
            cd_doc: Vec::new(),
            cd_start: Vec::new(),
            tokens: Vec::new(),
            weights: Vec::new(),
        };
        for (term, clusters) in &idx.clusters.terms {
            let first = s.infos.len();
            for c in clusters {
                let vec_begin = s.tokens.len();
                let cd_begin = s.cd_doc.len();
                for &d in &c.docs {
                    let (a, b) = idx.forward.span(d as usize);
                    s.cd_doc.push(d);
                    s.cd_start.push(s.tokens.len());
                    let (t, w) = idx.forward.vector(a, b);
                    s.tokens.extend_from_slice(t);
                    s.weights.extend_from_slice(w);
                }
                s.infos.push(ClInfo {
                    cd_begin,
                    n: c.docs.len(),
                    skip: c.should_not_skip,
                    summary: c.summary.clone(),
                    vec_begin,
                    vec_end: s.tokens.len(),
                });
            }
            s.by_term.insert(term.clone(), (first, clusters.len()));
        }
        s
    }

    /// Bytes the copies would take on disk: tokens and weights, plus about
    /// one byte per document for its length.
    pub fn copy_bytes(&self) -> usize {
        self.tokens.len() * 3 + self.cd_doc.len()
    }

    pub fn copies(&self) -> usize {
        self.cd_doc.len()
    }

    /// `SeismicBaseScorer.searchUpfront` over the in-memory clusters.
    /// Returns `(doc, raw score)` by ascending doc. `pages` collects the
    /// 4 KiB pages of the vector data read.
    pub fn search(
        &self,
        fwd: &ForwardVectors,
        layout: Layout,
        q: &SeismicQuery,
        max_doc: usize,
        mut pages: Option<&mut Vec<usize>>,
    ) -> Vec<(i32, i32)> {
        let mut visited = FixedBitSet::new(max_doc);
        let mut skip_heap = ScoreHeap::new(10);
        let mut results = ScoreHeap::new(q.k);
        for token in &q.top_tokens {
            let Some(&(first, count)) = self.by_term.get(token.as_bytes()) else {
                continue;
            };
            for ci in first..first + count {
                let c = &self.infos[ci];
                if !c.skip && skip_heap.is_full() {
                    let summary = c.summary.as_ref().map_or(0, |s| s.dot(&q.dense));
                    let bound = skip_heap.min_score().unwrap_or(0) as f32 / q.heap_factor;
                    if (summary as f32) < bound {
                        continue;
                    }
                }
                if layout == Layout::PerCluster {
                    if let Some(p) = pages.as_deref_mut() {
                        let (a, b) = (c.vec_begin, c.vec_end);
                        if b > a {
                            for pg in a * 2 / 4096..=(b * 2 - 1) / 4096 {
                                p.push((0 << 40) | pg);
                            }
                            for pg in a / 4096..=(b - 1) / 4096 {
                                p.push((1 << 40) | pg);
                            }
                        }
                    }
                }
                for j in 0..c.n {
                    let doc = self.cd_doc[c.cd_begin + j];
                    let d = doc as usize;
                    if visited.get(d) {
                        continue;
                    }
                    visited.set(d);
                    let s = match layout {
                        Layout::Shared => {
                            if let Some(p) = pages.as_deref_mut() {
                                let (a, b) = fwd.span(d);
                                if b > a {
                                    p.push(page(0, a * 2));
                                    p.push(page(0, b * 2 - 1));
                                    p.push(page(1, a));
                                    p.push(page(1, b - 1));
                                }
                            }
                            fwd.dot(d, &q.dense).unwrap_or(0)
                        }
                        Layout::PerCluster => {
                            let a = self.cd_start[c.cd_begin + j];
                            let b = if j + 1 < c.n {
                                self.cd_start[c.cd_begin + j + 1]
                            } else {
                                c.vec_end
                            };
                            dot(&self.tokens[a..b], &self.weights[a..b], &q.dense)
                        }
                    };
                    skip_heap.add(doc, s);
                    results.add(doc, s);
                }
            }
        }
        results.into_sorted_by_doc()
    }
}

#[inline]
fn dot(tokens: &[u16], weights: &[u8], dense: &[u8]) -> i32 {
    let mut score = 0i32;
    for (&t, &w) in tokens.iter().zip(weights) {
        let Some(&q) = dense.get(usize::from(t)) else {
            break;
        };
        score += i32::from(w) * i32::from(q);
    }
    score
}

/// Every document's exact score, the top `k` scores descending.
pub fn exact_top(fwd: &ForwardVectors, max_doc: usize, dense: &[u8], k: usize) -> Vec<i32> {
    let mut heap = ScoreHeap::new(k);
    for d in 0..max_doc {
        if let Some(s) = fwd.dot(d, dense) {
            heap.add(d as i32, s);
        }
    }
    let mut v: Vec<i32> = heap
        .into_sorted_by_doc()
        .into_iter()
        .map(|(_, s)| s)
        .collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v
}

/// Fraction of the exact top `k` a result reaches: results scoring at least
/// the exact `k`th score (ties count as found).
pub fn recall(exact: &[i32], got: &[(i32, i32)]) -> f64 {
    let Some(&kth) = exact.last() else { return 1.0 };
    let hits = got
        .iter()
        .filter(|&&(_, s)| s >= kth)
        .count()
        .min(exact.len());
    hits as f64 / exact.len() as f64
}

// ---------------------------------------------------------------------------
// Recursive graph bisection (a simplified BP).

struct Scratch {
    l: Vec<u32>,
    r: Vec<u32>,
}

fn bp_rec(
    docs: &[Vec<u16>],
    lg: &[f32],
    ids: &mut [u32],
    scratch: &mut Scratch,
    depth: usize,
    leaf: usize,
) {
    let n = ids.len();
    if n <= leaf {
        return;
    }
    let mid = n / 2;
    for _ in 0..6 {
        let (left, right) = ids.split_at(mid);
        for &d in left {
            for &t in &docs[d as usize] {
                scratch.l[usize::from(t)] += 1;
            }
        }
        for &d in right {
            for &t in &docs[d as usize] {
                scratch.r[usize::from(t)] += 1;
            }
        }
        let (nl, nr) = (left.len(), right.len());
        let gain = |d: u32, from_left: bool, s: &Scratch| -> f32 {
            let mut g = 0f32;
            for &t in &docs[d as usize] {
                let (l, r) = (s.l[usize::from(t)] as usize, s.r[usize::from(t)] as usize);
                // Cost of a term with l docs left and r right: l*log(nl/(l+1)) + r*log(nr/(r+1)).
                let cost = |l: usize, r: usize| {
                    l as f32 * (lg[nl] - lg[l + 1]) + r as f32 * (lg[nr] - lg[r + 1])
                };
                let (nl2, nr2) = if from_left {
                    (l - 1, r + 1)
                } else {
                    (l + 1, r - 1)
                };
                g += cost(l, r) - cost(nl2, nr2);
            }
            g
        };
        let mut gl: Vec<(f32, u32)> = left.iter().map(|&d| (gain(d, true, scratch), d)).collect();
        let mut gr: Vec<(f32, u32)> = right
            .iter()
            .map(|&d| (gain(d, false, scratch), d))
            .collect();
        for &d in left {
            for &t in &docs[d as usize] {
                scratch.l[usize::from(t)] = 0;
            }
        }
        for &d in right {
            for &t in &docs[d as usize] {
                scratch.r[usize::from(t)] = 0;
            }
        }
        gl.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        gr.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let mut swapped = 0;
        for (a, b) in gl.iter().zip(&gr) {
            if a.0 + b.0 <= 0.0 {
                break;
            }
            swapped += 1;
        }
        let mut fixed: Vec<u32> = Vec::with_capacity(n);
        fixed.extend(gl[swapped..].iter().map(|x| x.1));
        fixed.extend(gr[..swapped].iter().map(|x| x.1));
        fixed.extend(gr[swapped..].iter().map(|x| x.1));
        fixed.extend(gl[..swapped].iter().map(|x| x.1));
        ids.copy_from_slice(&fixed);
        if swapped == 0 {
            break;
        }
    }
    let (l, r) = ids.split_at_mut(mid);
    if depth < 2 {
        std::thread::scope(|s| {
            s.spawn(|| {
                let mut sc = Scratch {
                    l: vec![0; VOCAB],
                    r: vec![0; VOCAB],
                };
                bp_rec(docs, lg, l, &mut sc, depth + 1, leaf);
            });
            bp_rec(docs, lg, r, scratch, depth + 1, leaf);
        });
    } else {
        bp_rec(docs, lg, l, scratch, depth + 1, leaf);
        bp_rec(docs, lg, r, scratch, depth + 1, leaf);
    }
}

/// New order of the documents: `order[new] = old`.
pub fn bp_order(docs: &[Pairs], leaf: usize) -> Vec<u32> {
    let tokens: Vec<Vec<u16>> = docs
        .iter()
        .map(|d| d.iter().map(|&(t, _)| t as u16).collect())
        .collect();
    let lg: Vec<f32> = (0..docs.len() + 3)
        .map(|i| (i.max(1) as f32).log2())
        .collect();
    let mut ids: Vec<u32> = (0..docs.len() as u32).collect();
    let mut sc = Scratch {
        l: vec![0; VOCAB],
        r: vec![0; VOCAB],
    };
    bp_rec(&tokens, &lg, &mut ids, &mut sc, 0, leaf);
    ids
}

// ---------------------------------------------------------------------------
// Block / superblock index.

pub struct BlockIndex {
    pub b: usize,
    pub c: usize,
    n: usize,
    nsb: usize,
    order: Vec<u32>,
    starts: Vec<usize>,
    tokens: Vec<u16>,
    weights: Vec<u8>,
    /// `[term * nsb + sb]`: the largest weight of the term in the superblock.
    sb_max: Vec<u8>,
    sb_term_off: Vec<usize>,
    sb_terms: Vec<u16>,
    run_off: Vec<u32>,
    run_blk: Vec<u8>,
    run_max: Vec<u8>,
}

#[derive(Clone, Copy)]
pub struct BlockParams {
    /// Superblocks visited at most (`usize::MAX`: until pruned).
    pub gamma: usize,
    /// Prune when `bound * mu <= kth score`; 1.0 is rank-safe for the bound terms.
    pub mu: f32,
    /// Query terms (by weight) the bounds are computed over; the score uses all of them.
    pub bound_terms: usize,
}

impl BlockIndex {
    pub fn build(docs: &[Pairs], order: Vec<u32>, b: usize, c: usize, ceiling: f32) -> Self {
        let n = docs.len();
        let nblocks = n.div_ceil(b);
        let nsb = nblocks.div_ceil(c);
        let mut idx = BlockIndex {
            b,
            c,
            n,
            nsb,
            starts: Vec::with_capacity(n + 1),
            tokens: Vec::new(),
            weights: Vec::new(),
            sb_max: vec![0; VOCAB * nsb],
            sb_term_off: vec![0],
            sb_terms: Vec::new(),
            run_off: vec![0],
            run_blk: Vec::new(),
            run_max: Vec::new(),
            order,
        };
        for &old in &idx.order.clone() {
            idx.starts.push(idx.tokens.len());
            let v = SparseVector::quantized(&docs[old as usize], ceiling);
            idx.tokens.extend_from_slice(v.tokens());
            idx.weights.extend_from_slice(v.weights());
        }
        idx.starts.push(idx.tokens.len());
        for sb in 0..nsb {
            let mut entries: Vec<(u16, u8, u8)> = Vec::new();
            let first_doc = sb * c * b;
            let last_doc = ((sb + 1) * c * b).min(n);
            for d in first_doc..last_doc {
                let blk = ((d - first_doc) / b) as u8;
                for i in idx.starts[d]..idx.starts[d + 1] {
                    entries.push((idx.tokens[i], blk, idx.weights[i]));
                }
            }
            entries.sort_unstable_by_key(|&(t, blk, _)| (t, blk));
            let mut i = 0;
            while i < entries.len() {
                let t = entries[i].0;
                let mut smax = 0u8;
                while i < entries.len() && entries[i].0 == t {
                    let blk = entries[i].1;
                    let mut m = 0u8;
                    while i < entries.len() && entries[i].0 == t && entries[i].1 == blk {
                        m = m.max(entries[i].2);
                        i += 1;
                    }
                    idx.run_blk.push(blk);
                    idx.run_max.push(m);
                    smax = smax.max(m);
                }
                idx.sb_terms.push(t);
                idx.run_off.push(idx.run_blk.len() as u32);
                idx.sb_max[usize::from(t) * nsb + sb] = smax;
            }
            idx.sb_term_off.push(idx.sb_terms.len());
        }
        idx
    }

    /// Bytes of bounds and block layout without the vectors, with the dense
    /// superblock table counted as the sparse (term, superblock, byte) triples
    /// it holds.
    pub fn bounds_bytes(&self) -> usize {
        self.sb_terms.len() * 3 + self.run_blk.len() * 2 + self.run_off.len() * 4
    }

    pub fn dense_table_bytes(&self) -> usize {
        self.sb_max.len()
    }

    pub fn vector_bytes(&self) -> usize {
        self.tokens.len() * 3 + self.starts.len() * 8
    }

    /// `(doc, raw score)` of the best `k`, `pages` collecting the 4 KiB pages read.
    pub fn search(
        &self,
        q: &SeismicQuery,
        bound: &[(usize, i32)],
        p: BlockParams,
        mut pages: Option<&mut Vec<usize>>,
    ) -> Vec<(i32, i32)> {
        let nsb = self.nsb;
        let mut ub = vec![0u32; nsb];
        for &(t, qw) in bound {
            let row = &self.sb_max[t * nsb..(t + 1) * nsb];
            for (u, &m) in ub.iter_mut().zip(row) {
                *u += qw as u32 * u32::from(m);
            }
        }
        let mut sbs: Vec<(u32, u32)> = ub
            .iter()
            .enumerate()
            .filter(|&(_, &u)| u > 0)
            .map(|(i, &u)| (u, i as u32))
            .collect();
        sbs.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        let mut heap = ScoreHeap::new(q.k);
        let mut blk_ub = vec![0u32; self.c];
        let mut order: Vec<usize> = Vec::with_capacity(self.c);
        for (visited, &(sb_ub, sb)) in sbs.iter().enumerate() {
            if visited >= p.gamma {
                break;
            }
            if heap.is_full() && sb_ub as f32 * p.mu <= heap.min_score().unwrap_or(0) as f32 {
                break;
            }
            let sb = sb as usize;
            blk_ub.iter_mut().for_each(|x| *x = 0);
            let terms = &self.sb_terms[self.sb_term_off[sb]..self.sb_term_off[sb + 1]];
            for &(t, qw) in bound {
                if let Ok(pos) = terms.binary_search(&(t as u16)) {
                    let g = self.sb_term_off[sb] + pos;
                    for r in self.run_off[g] as usize..self.run_off[g + 1] as usize {
                        blk_ub[usize::from(self.run_blk[r])] +=
                            qw as u32 * u32::from(self.run_max[r]);
                    }
                }
            }
            order.clear();
            order.extend((0..self.c).filter(|&i| blk_ub[i] > 0));
            order.sort_unstable_by(|&a, &b| blk_ub[b].cmp(&blk_ub[a]));
            for &bi in &order {
                if heap.is_full()
                    && blk_ub[bi] as f32 * p.mu <= heap.min_score().unwrap_or(0) as f32
                {
                    break;
                }
                let first = (sb * self.c + bi) * self.b;
                let last = (first + self.b).min(self.n);
                if first >= last {
                    continue;
                }
                if let Some(pg) = pages.as_deref_mut() {
                    let (a, e) = (self.starts[first], self.starts[last]);
                    if e > a {
                        for x in a * 2 / 4096..=(e * 2 - 1) / 4096 {
                            pg.push(x);
                        }
                        for x in a / 4096..=(e - 1) / 4096 {
                            pg.push((1 << 40) | x);
                        }
                    }
                }
                for d in first..last {
                    let (a, e) = (self.starts[d], self.starts[d + 1]);
                    let s = dot(&self.tokens[a..e], &self.weights[a..e], &q.dense);
                    heap.add(self.order[d] as i32, s);
                }
            }
        }
        heap.into_sorted_by_doc()
    }
}

/// The `n` heaviest tokens of the quantized query as `(token, weight)`.
pub fn bound_terms(q: &SeismicQuery, n: usize) -> Vec<(usize, i32)> {
    let mut v: Vec<(usize, i32)> = q
        .dense
        .iter()
        .enumerate()
        .filter(|&(_, &w)| w > 0)
        .map(|(t, &w)| (t, i32::from(w)))
        .collect();
    v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(n);
    v
}
