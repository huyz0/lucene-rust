//! Building the indexes of [`crate::format`] from vectors: `index_factory`,
//! `ParameterSpace::set_index_parameters`, `train` and `add_with_ids`.
//!
//! The graph is Faiss's (`HNSW::add_with_locks`): levels drawn from
//! `assign_probas`, a greedy descent through the upper levels, an
//! `efConstruction` beam at each level the vertex reaches, the neighbour
//! heuristic of `shrink_neighbor_list`, reverse links shrunk the same way.
//! Vertices go in highest level first, in parallel past the first. The
//! result is a valid Faiss graph, not Faiss's graph bit for bit: its random
//! levels and its insertion races are not reproducible between runs either.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicI32, Ordering as Atomic};
use std::sync::Mutex;

use rayon::prelude::*;

use crate::distance::distance;
use crate::error::{Error, Result};
use crate::factory::{parse_description, parse_params, Description, IndexParams, Kind};
use crate::format::{Flat, Hnsw, Index, InvList, IvfFlat, Metric};
use crate::kmeans::{self, Rng};

const SEED: u64 = 12345;

/// A `(distance, vertex)` pair ordered by distance, nearest first on top of
/// a [`BinaryHeap`] when wrapped in [`std::cmp::Reverse`].
#[derive(Clone, Copy, PartialEq)]
struct Node(f32, u32);

impl Eq for Node {}

impl Ord for Node {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0).then(self.1.cmp(&o.1))
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Builds the index `description` names over `vectors` (`n` rows of `d`
/// floats) with `ids` as their external ids.
///
/// # Errors
/// [`Error::Description`] for an unsupported factory string,
/// [`Error::Invalid`] for unsupported index parameters, mismatched
/// lengths, or `HNSW` without `IDMap` (Faiss refuses `add_with_ids` there).
pub fn build_index(
    description: &str,
    index_params: &str,
    metric: Metric,
    d: usize,
    vectors: &[f32],
    ids: &[i64],
) -> Result<Index> {
    let desc = parse_description(description)?;
    let params = parse_params(index_params)?;
    if d == 0 || !vectors.len().is_multiple_of(d) || vectors.len() / d != ids.len() {
        return Err(Error::Invalid(format!(
            "{} floats and {} ids do not make vectors of {d} dimensions",
            vectors.len(),
            ids.len()
        )));
    }
    build(&desc, &params, metric, d, vectors, ids)
}

fn build(
    desc: &Description,
    params: &IndexParams,
    metric: Metric,
    d: usize,
    vectors: &[f32],
    ids: &[i64],
) -> Result<Index> {
    let n = ids.len();
    let storage = Flat {
        d,
        metric,
        data: vectors.to_vec(),
    };
    let sub = match desc.kind {
        Kind::Flat => Index::Flat(storage),
        Kind::Hnsw { m } => {
            if !desc.id_map {
                return Err(Error::Invalid(
                    "add_with_ids is not implemented for a graph index; describe it as IDMap,HNSW<M>".into(),
                ));
            }
            let mut hnsw = new_hnsw(m, n);
            hnsw.ef_construction = params.ef_construction.map_or(40, |v| v as i32);
            hnsw.ef_search = params.ef_search.map_or(16, |v| v as i32);
            insert_all(&mut hnsw, &storage);
            Index::Hnsw { hnsw, storage }
        }
        Kind::Ivf { nlist } => {
            let nlist = nlist.min(n.max(1));
            let quantizer = if n == 0 {
                Flat {
                    d,
                    metric,
                    data: vec![0.0; d * nlist],
                }
            } else {
                kmeans::train(vectors, d, nlist, metric, SEED)
            };
            let assigned: Vec<usize> = (0..n)
                .into_par_iter()
                .map(|i| kmeans::nearest(&quantizer, &vectors[i * d..(i + 1) * d]))
                .collect();
            let mut lists = vec![InvList::default(); nlist];
            for (i, &l) in assigned.iter().enumerate() {
                // Without an id map the lists keep the external ids; with
                // one they keep ordinals, which the map translates.
                lists[l]
                    .ids
                    .push(if desc.id_map { i as i64 } else { ids[i] });
                lists[l]
                    .codes
                    .extend_from_slice(&vectors[i * d..(i + 1) * d]);
            }
            Index::IvfFlat(IvfFlat {
                d,
                metric,
                nprobe: params.nprobe.unwrap_or(1),
                quantizer,
                lists,
            })
        }
    };
    Ok(if desc.id_map {
        Index::IdMap {
            sub: Box::new(sub),
            ids: ids.to_vec(),
        }
    } else {
        sub
    })
}

/// `HNSW::set_default_probas` and the empty graph over `n` vertices.
fn new_hnsw(m: usize, n: usize) -> Hnsw {
    let level_mult = 1.0 / (m as f32).ln();
    let mut assign_probas = Vec::new();
    let mut cum = vec![0i32];
    let mut nn = 0i32;
    for level in 0.. {
        let proba = (-(level as f32) / level_mult).exp() * (1.0 - (-1.0 / level_mult).exp());
        if proba < 1e-9 {
            break;
        }
        assign_probas.push(f64::from(proba));
        nn += if level == 0 { 2 * m as i32 } else { m as i32 };
        cum.push(nn);
    }
    Hnsw {
        assign_probas,
        cum_nneighbor_per_level: cum,
        levels: vec![0; n],
        offsets: vec![0; n + 1],
        neighbors: Vec::new(),
        entry_point: -1,
        max_level: -1,
        ef_construction: 40,
        ef_search: 16,
        upper_beam: 1,
    }
}

fn random_level(h: &Hnsw, rng: &mut Rng) -> i32 {
    let mut f = f64::from(rng.unit_float());
    for (level, &p) in h.assign_probas.iter().enumerate() {
        if f < p {
            return level as i32;
        }
        f -= p;
    }
    h.assign_probas.len() as i32 - 1
}

struct Shared<'a> {
    h: &'a Hnsw,
    storage: &'a Flat,
    neighbors: Vec<AtomicI32>,
    locks: Vec<Mutex<()>>,
    sign: f32,
}

impl Shared<'_> {
    fn dis_vec(&self, q: &[f32], i: usize) -> f32 {
        self.sign * distance(self.storage.metric, q, self.storage.vector(i))
    }

    fn dis_pair(&self, a: usize, b: usize) -> f32 {
        self.sign
            * distance(
                self.storage.metric,
                self.storage.vector(a),
                self.storage.vector(b),
            )
    }

    fn range(&self, no: usize, level: usize) -> std::ops::Range<usize> {
        let o = self.h.offsets[no] as usize;
        o + self.h.cum_nneighbor_per_level[level] as usize
            ..o + self.h.cum_nneighbor_per_level[level + 1] as usize
    }

    fn slot(&self, i: usize) -> i32 {
        self.neighbors[i].load(Atomic::Relaxed)
    }

    /// `shrink_neighbor_list`: keeps a candidate only when it is nearer the
    /// query than to every candidate already kept.
    fn shrink(&self, mut input: Vec<Node>, max: usize, keep_max: bool) -> Vec<Node> {
        input.sort_unstable();
        let mut out: Vec<Node> = Vec::with_capacity(max);
        let mut returned = Vec::new();
        for v1 in input {
            let good = out
                .iter()
                .all(|v2| self.dis_pair(v2.1 as usize, v1.1 as usize) >= v1.0);
            if good {
                out.push(v1);
                if out.len() >= max {
                    return out;
                }
            } else if keep_max {
                returned.push(v1);
            }
        }
        for v in returned {
            if out.len() >= max {
                break;
            }
            out.push(v);
        }
        out
    }

    fn add_link(&self, src: usize, dest: usize, level: usize) {
        let _guard = self.locks[src].lock().unwrap_or_else(|e| e.into_inner());
        let range = self.range(src, level);
        if self.slot(range.end - 1) == -1 {
            if let Some(free) = range.clone().find(|&i| self.slot(i) == -1) {
                self.neighbors[free].store(dest as i32, Atomic::Relaxed);
                return;
            }
        }
        let mut set = vec![Node(self.dis_pair(src, dest), dest as u32)];
        for i in range.clone() {
            let nb = self.slot(i);
            if nb >= 0 {
                set.push(Node(self.dis_pair(src, nb as usize), nb as u32));
            }
        }
        let kept = self.shrink(set, range.len(), false);
        for (i, slot) in range.clone().enumerate() {
            let v = kept.get(i).map_or(-1, |n| n.1 as i32);
            self.neighbors[slot].store(v, Atomic::Relaxed);
        }
    }

    fn greedy(&self, q: &[f32], level: usize, nearest: &mut usize, d_nearest: &mut f32) {
        loop {
            let prev = *nearest;
            let _guard = self.locks[prev].lock().unwrap_or_else(|e| e.into_inner());
            for slot in self.range(prev, level) {
                let v = self.slot(slot);
                if v < 0 {
                    break;
                }
                let d = self.dis_vec(q, v as usize);
                if d < *d_nearest {
                    *nearest = v as usize;
                    *d_nearest = d;
                }
            }
            drop(_guard);
            if *nearest == prev {
                return;
            }
        }
    }

    /// `search_neighbors_to_add`: the `efConstruction` nearest found by a
    /// beam search of `level` from `entry`.
    fn beam(
        &self,
        q: &[f32],
        entry: usize,
        d_entry: f32,
        level: usize,
        visited: &mut Visited,
    ) -> Vec<Node> {
        let ef = self.h.ef_construction.max(1) as usize;
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();
        candidates.push(std::cmp::Reverse(Node(d_entry, entry as u32)));
        results.push(Node(d_entry, entry as u32));
        visited.set(entry);
        while let Some(std::cmp::Reverse(cur)) = candidates.peek().copied() {
            if cur.0 > results.peek().map_or(f32::INFINITY, |r: &Node| r.0) {
                break;
            }
            candidates.pop();
            let _guard = self.locks[cur.1 as usize]
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let nbrs: Vec<i32> = self
                .range(cur.1 as usize, level)
                .map(|i| self.slot(i))
                .take_while(|&v| v >= 0)
                .collect();
            drop(_guard);
            for v in nbrs {
                let v = v as usize;
                if visited.get(v) {
                    continue;
                }
                visited.set(v);
                let d = self.dis_vec(q, v);
                if results.len() < ef || results.peek().is_some_and(|r| r.0 > d) {
                    results.push(Node(d, v as u32));
                    candidates.push(std::cmp::Reverse(Node(d, v as u32)));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        results.into_vec()
    }

    fn insert(
        &self,
        pt: usize,
        pt_level: usize,
        entry: usize,
        max_level: usize,
        visited: &mut Visited,
    ) {
        let q = self.storage.vector(pt);
        let mut nearest = entry;
        let mut d_nearest = self.dis_vec(q, nearest);
        for level in ((pt_level + 1)..=max_level).rev() {
            self.greedy(q, level, &mut nearest, &mut d_nearest);
        }
        for level in (0..=pt_level.min(max_level)).rev() {
            visited.clear();
            let found = self.beam(q, nearest, d_nearest, level, visited);
            let m = self.range(pt, level).len();
            let chosen = self.shrink(found, m, level == 0);
            if let Some(n) = chosen.iter().min() {
                nearest = n.1 as usize;
                d_nearest = n.0;
            }
            for n in &chosen {
                self.add_link(pt, n.1 as usize, level);
                self.add_link(n.1 as usize, pt, level);
            }
        }
    }
}

/// The visited set of one search, cleared by bumping an epoch.
struct Visited {
    marks: Vec<u32>,
    epoch: u32,
}

impl Visited {
    fn new(n: usize) -> Self {
        Visited {
            marks: vec![0; n],
            epoch: 1,
        }
    }
    fn clear(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
    }
    fn set(&mut self, i: usize) {
        self.marks[i] = self.epoch;
    }
    fn get(&self, i: usize) -> bool {
        self.marks[i] == self.epoch
    }
}

/// `hnsw_add_vertices`: levels first, then every vertex, highest level first.
fn insert_all(hnsw: &mut Hnsw, storage: &Flat) {
    let n = storage.len();
    if n == 0 {
        return;
    }
    let mut rng = Rng(SEED);
    let levels: Vec<i32> = (0..n).map(|_| random_level(hnsw, &mut rng)).collect();
    hnsw.levels = levels.iter().map(|l| l + 1).collect();
    let mut off = 0u64;
    for i in 0..n {
        hnsw.offsets[i] = off;
        off += hnsw.cum_nneighbor_per_level[hnsw.levels[i] as usize] as u64;
    }
    hnsw.offsets[n] = off;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(levels[i]));
    let max_level = levels[order[0]] as usize;
    let shared = Shared {
        h: hnsw,
        storage,
        neighbors: (0..off).map(|_| AtomicI32::new(-1)).collect(),
        locks: (0..n).map(|_| Mutex::new(())).collect(),
        sign: if storage.metric == Metric::InnerProduct {
            -1.0
        } else {
            1.0
        },
    };
    let entry = order[0];
    // The entry point alone, then everything else in parallel.
    order[1..].par_iter().for_each_init(
        || Visited::new(n),
        |visited, &pt| shared.insert(pt, levels[pt] as usize, entry, max_level, visited),
    );
    let neighbors: Vec<i32> = shared
        .neighbors
        .iter()
        .map(|a| a.load(Atomic::Relaxed))
        .collect();
    hnsw.neighbors = neighbors;
    hnsw.entry_point = entry as i32;
    hnsw.max_level = max_level as i32;
}
