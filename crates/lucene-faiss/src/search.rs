//! Searching the indexes of [`crate::format`] the way Faiss does
//! (`IndexFlat`, `IndexHNSW`, `IndexIVFFlat`, `IndexIDMap`): same traversal,
//! same heaps, same tie order, with an optional id filter that applies to
//! the ids the caller sees, as `SearchParameters::sel` does.

use crate::distance::distance;
use crate::error::{Error, Result};
use crate::format::{Flat, Hnsw, Index, IvfFlat, IvfSq, Metric, SqFlat};
use crate::heap::{Max, Min, Minimax, Order, TopK};
use crate::store::VectorStore;
use crate::visited::VISITED;

/// Search settings Faiss takes as `SearchParameters`; `None` keeps what the
/// index holds (`efSearch`, `nprobe`).
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchParams {
    /// `SearchParametersHNSW::efSearch`.
    pub ef_search: Option<usize>,
    /// `SearchParametersIVF::nprobe`.
    pub nprobe: Option<usize>,
}

/// A predicate over ids: `SearchParameters::sel`.
pub type Selector<'a> = &'a dyn Fn(i64) -> bool;

impl Index {
    /// The `k` nearest vectors to `query`, nearest first, as `(id, distance)`
    /// with Faiss's distances (squared L2, or the inner product). `filter`
    /// restricts the results to ids it accepts; a graph is still walked
    /// through the rejected ones.
    ///
    /// # Errors
    /// [`Error::Invalid`] when `query` is not `d` floats long.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
        filter: Option<Selector<'_>>,
    ) -> Result<Vec<(i64, f32)>> {
        if query.len() != self.d() {
            return Err(Error::Invalid(format!(
                "query of {} floats for an index of {} dimensions",
                query.len(),
                self.d()
            )));
        }
        match self {
            Index::IdMap { sub, ids } => {
                let translate =
                    |ord: i64| usize::try_from(ord).ok().and_then(|o| ids.get(o).copied());
                let accept = |ord: i64| match (filter, translate(ord)) {
                    (Some(f), Some(id)) => f(id),
                    (None, Some(_)) => true,
                    (_, None) => false,
                };
                let found = sub.search_inner(query, k, params, Some(&accept))?;
                Ok(found
                    .into_iter()
                    .filter_map(|(ord, d)| translate(ord).map(|id| (id, d)))
                    .collect())
            }
            other => other.search_inner(query, k, params, filter),
        }
    }

    fn search_inner(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
        filter: Option<Selector<'_>>,
    ) -> Result<Vec<(i64, f32)>> {
        Ok(match self {
            Index::Flat(f) => search_flat(f, query, k, filter),
            Index::Hnsw { hnsw, storage } => search_hnsw(hnsw, storage, query, k, params, filter),
            Index::IvfFlat(ivf) => search_ivf(ivf, query, k, params, filter),
            Index::Sq(f) => search_sq(f, query, k, filter),
            Index::HnswSq { hnsw, storage } => search_hnsw(hnsw, storage, query, k, params, filter),
            Index::IvfSq(ivf) => search_ivf_sq(ivf, query, k, params, filter),
            Index::IdMap { .. } => {
                return Err(Error::Unsupported("an id map inside an id map".into()))
            }
        })
    }
}

fn collect<C: Order>(
    metric_scan: impl Iterator<Item = (i64, f32)>,
    k: usize,
    filter: Option<Selector<'_>>,
) -> Vec<(i64, f32)> {
    let mut top = TopK::<C>::new(k);
    for (id, d) in metric_scan {
        if filter.is_none_or(|f| f(id)) {
            top.add(d, id);
        }
    }
    top.finish()
}

fn search_flat(f: &Flat, q: &[f32], k: usize, filter: Option<Selector<'_>>) -> Vec<(i64, f32)> {
    let scan = (0..f.len()).map(|i| (i as i64, distance(f.metric, q, f.vector(i))));
    match f.metric {
        Metric::L2 => collect::<Max>(scan, k, filter),
        Metric::InnerProduct => collect::<Min>(scan, k, filter),
    }
}

fn search_ivf(
    ivf: &IvfFlat,
    q: &[f32],
    k: usize,
    params: &SearchParams,
    filter: Option<Selector<'_>>,
) -> Vec<(i64, f32)> {
    let nprobe = params
        .nprobe
        .unwrap_or(ivf.nprobe)
        .clamp(1, ivf.lists.len().max(1));
    let probes: Vec<usize> = search_flat(&ivf.quantizer, q, nprobe, None)
        .into_iter()
        .filter_map(|(l, _)| usize::try_from(l).ok())
        .collect();
    match ivf.metric {
        Metric::L2 => scan_lists::<Max>(ivf, &probes, q, k, filter),
        Metric::InnerProduct => scan_lists::<Min>(ivf, &probes, q, k, filter),
    }
}

/// Scans the probed lists in order, skipping the heap for any vector no
/// nearer than the current `k`-th, and prefetching the next list's start.
fn scan_lists<C: Order>(
    ivf: &IvfFlat,
    probes: &[usize],
    q: &[f32],
    k: usize,
    filter: Option<Selector<'_>>,
) -> Vec<(i64, f32)> {
    let mut top = TopK::<C>::new(k);
    for (at, &l) in probes.iter().enumerate() {
        if let Some(&next) = probes.get(at + 1) {
            let codes = &ivf.lists[next].codes;
            for line in 0..4 {
                lucene_util::simd::prefetch(codes, line * 16);
            }
        }
        let list = &ivf.lists[l];
        for (code, &id) in list.codes.chunks_exact(ivf.d).zip(&list.ids) {
            let d = distance(ivf.metric, q, code);
            if C::worse(top.threshold(), d) && filter.is_none_or(|f| f(id)) {
                top.add(d, id);
            }
        }
    }
    top.finish()
}

fn search_sq(f: &SqFlat, q: &[f32], k: usize, filter: Option<Selector<'_>>) -> Vec<(i64, f32)> {
    let d = f.sq.d();
    let scan = f
        .codes
        .chunks_exact(d.max(1))
        .enumerate()
        .map(|(i, code)| (i as i64, f.sq.distance(f.metric, q, code)));
    match f.metric {
        Metric::L2 => collect::<Max>(scan, k, filter),
        Metric::InnerProduct => collect::<Min>(scan, k, filter),
    }
}

fn search_ivf_sq(
    ivf: &IvfSq,
    q: &[f32],
    k: usize,
    params: &SearchParams,
    filter: Option<Selector<'_>>,
) -> Vec<(i64, f32)> {
    let nprobe = params
        .nprobe
        .unwrap_or(ivf.nprobe)
        .clamp(1, ivf.lists.len().max(1));
    let probes: Vec<(usize, f32)> = search_flat(&ivf.quantizer, q, nprobe, None)
        .into_iter()
        .filter_map(|(l, coarse)| usize::try_from(l).ok().map(|l| (l, coarse)))
        .collect();
    match ivf.metric {
        Metric::L2 => scan_sq_lists::<Max>(ivf, &probes, q, k, filter),
        Metric::InnerProduct => scan_sq_lists::<Min>(ivf, &probes, q, k, filter),
    }
}

/// `IndexIVFScalarQuantizer`'s scan: with residual codes Euclidean measures
/// from the query's residual to the list's centroid, and the inner product
/// adds the query's product with the centroid (the coarse distance) to the
/// product with the decoded residual.
fn scan_sq_lists<C: Order>(
    ivf: &IvfSq,
    probes: &[(usize, f32)],
    q: &[f32],
    k: usize,
    filter: Option<Selector<'_>>,
) -> Vec<(i64, f32)> {
    let mut top = TopK::<C>::new(k);
    let mut residual = vec![0f32; q.len()];
    for &(l, coarse) in probes {
        let list = &ivf.lists[l];
        let (query, base) = match (ivf.by_residual, ivf.metric) {
            (true, Metric::L2) => {
                for ((r, &x), &c) in residual.iter_mut().zip(q).zip(ivf.quantizer.vector(l)) {
                    *r = x - c;
                }
                (&residual[..], 0.0)
            }
            (true, Metric::InnerProduct) => (q, coarse),
            (false, _) => (q, 0.0),
        };
        for (code, &id) in list.codes.chunks_exact(ivf.d).zip(&list.ids) {
            let d = base + ivf.sq.distance(ivf.metric, query, code);
            if C::worse(top.threshold(), d) && filter.is_none_or(|f| f(id)) {
                top.add(d, id);
            }
        }
    }
    top.finish()
}

fn neighbor_range(h: &Hnsw, no: usize, level: usize) -> std::ops::Range<usize> {
    let offset = h.offsets[no] as usize;
    offset + h.cum_nneighbor_per_level[level] as usize
        ..offset + h.cum_nneighbor_per_level[level + 1] as usize
}

fn search_hnsw<S: VectorStore>(
    h: &Hnsw,
    storage: &S,
    q: &[f32],
    k: usize,
    params: &SearchParams,
    filter: Option<Selector<'_>>,
) -> Vec<(i64, f32)> {
    if h.entry_point < 0 || storage.len() == 0 {
        return Vec::new();
    }
    // Faiss searches on distances to minimise; a similarity metric is
    // negated through the search and back after it.
    let sign = if storage.metric() == Metric::InnerProduct {
        -1.0
    } else {
        1.0
    };
    let dis = |i: usize| sign * storage.dis_query(q, i);
    let ef_search = params
        .ef_search
        .unwrap_or_else(|| usize::try_from(h.ef_search).unwrap_or(16));
    let mut nearest = h.entry_point as usize;
    let mut d_nearest = dis(nearest);
    for level in (1..=h.max_level.max(0) as usize).rev() {
        loop {
            let prev = nearest;
            for slot in neighbor_range(h, nearest, level) {
                let v = h.neighbors[slot];
                if v < 0 {
                    break;
                }
                let d = dis(v as usize);
                if d < d_nearest {
                    nearest = v as usize;
                    d_nearest = d;
                }
            }
            if nearest == prev {
                break;
            }
        }
    }
    let ef = ef_search.max(k);
    let mut candidates = Minimax::new(ef);
    candidates.push(nearest as i64, d_nearest);
    let mut visited = VISITED.with(|v| v.take());
    visited.begin(storage.len());
    let mut res = TopK::<Max>::new(k);
    let member = |id: usize| filter.is_none_or(|f| f(id as i64));

    let mut seed = vec![(nearest, d_nearest)];
    for &(v, d) in &seed {
        if member(v) && d < res.threshold() {
            res.add(d, v as i64);
        }
        visited.set(v);
    }
    seed.clear();
    let mut nstep = 0usize;
    while candidates.size() > 0 {
        let Some((v0, d0)) = candidates.pop_min() else {
            break;
        };
        // More than `efSearch` processed distances are already below `d0`.
        if candidates.count_below(d0) >= ef_search {
            break;
        }
        let mut batch = [0usize; 64];
        let mut count = 0usize;
        for slot in neighbor_range(h, v0 as usize, 0) {
            let n = h.neighbors[slot];
            if n < 0 {
                break;
            }
            let n = n as usize;
            if visited.seen(n) {
                continue;
            }
            visited.set(n);
            if let Some(slot) = batch.get_mut(count) {
                *slot = n;
                count += 1;
                // Start every vector's cache misses before computing any.
                storage.prefetch(n);
            }
        }
        for &n in &batch[..count] {
            let d = dis(n);
            if member(n) && d < res.threshold() {
                res.add(d, n as i64);
            }
            candidates.push(n as i64, d);
        }
        nstep += 1;
        let _ = nstep;
    }
    VISITED.with(|v| v.set(visited));
    res.finish()
        .into_iter()
        .map(|(id, d)| (id, sign * d))
        .collect()
}
