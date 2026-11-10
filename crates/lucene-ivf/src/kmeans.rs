//! Partitioning a field's vectors into clusters of about a target size.
//!
//! Flat k-means with thousands of centroids costs `n * k * dim` per
//! iteration, which is what makes IVF builds slow. Here the clusters come
//! from **hierarchical k-means** instead: a cluster over the target size is
//! split with a small k-means (at most [`MAX_BRANCHING`] parts, trained on a
//! sample), and the parts are split again until every leaf is at most the
//! target size. Each vector is compared with `O(MAX_BRANCHING * depth)`
//! centroids instead of all of them.
//!
//! Descending a tree assigns a few vectors to a leaf that is not their
//! nearest (the split happened higher up). A **neighbourhood refinement**
//! fixes most of that: each leaf finds its [`REFINE_NEIGHBORS`] nearest leaf
//! centroids, and a few Lloyd rounds move every vector to the nearest
//! centroid among its own leaf and those neighbours, recomputing the means
//! after each round. That is `O(n * REFINE_NEIGHBORS * dim)` per round.
//!
//! Neighbouring centroids are not always where a vector's nearest centroid
//! is: in a hundred or more dimensions about one vector in ten has its
//! nearest centroid outside the leaf's 64 nearest. So the last few rounds
//! are **full Lloyd rounds** that compare every vector with every centroid,
//! using a register-tiled kernel (4 vectors by 3 centroids at a time). They
//! cost `O(n * k * dim)` each, and on GloVe they lift recall at a fixed
//! number of probes by about as much as flat k-means gains over the tree.
//!
//! Vectors go to the centroid nearest by a [`Metric`]: Euclidean distance,
//! or the largest inner product, which callers use for angular and
//! inner-product fields (their `Cosine` vectors normalized first). Centroids
//! are plain means either way; the search ranks them by the same metric.
//!
//! Everything is deterministic for a given input and seed: the samples are
//! drawn from a seeded generator and parallel reductions combine in a fixed
//! order.

use rayon::prelude::*;

use lucene_util::simd;

/// The most parts one split makes.
pub const MAX_BRANCHING: usize = 64;
/// How many neighbouring leaves the refinement considers per leaf.
pub const REFINE_NEIGHBORS: usize = 64;

/// What "nearest centroid" means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// The smallest squared Euclidean distance.
    L2,
    /// Spherical k-means, for unit vectors: centroids are normalized while
    /// clustering and a vector goes to the largest inner product. It
    /// partitions angular data better than Euclidean distance to the means
    /// does. The centroids returned are still the clusters' plain means
    /// (what quantization centres on); rank them by inner product over
    /// centroid norm to reproduce the assignment.
    InnerProduct,
}

impl Metric {
    /// Normalizes freshly computed means for spherical k-means.
    pub(crate) fn adjust(self, centroids: &mut [f32], dim: usize) {
        if self == Metric::InnerProduct {
            centroids.par_chunks_mut(dim).for_each(|c| {
                // A zero mean stays zero.
                let _ = lucene_util::vector_util::l2normalize(c, false);
            });
        }
    }

    /// The assignment cost of `v` to `centroid`: lower is nearer.
    #[inline]
    pub fn cost(self, v: &[f32], centroid: &[f32]) -> f32 {
        match self {
            Metric::L2 => simd::square_distance_f32(v, centroid),
            Metric::InnerProduct => -simd::dot_f32(v, centroid),
        }
    }
}

/// Clustering parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KMeansParams {
    /// Lloyd iterations of each split.
    pub iterations: usize,
    /// Training points per centroid of a split (the split trains on at most
    /// `parts * sample_per_centroid` of its vectors).
    pub sample_per_centroid: usize,
    /// Neighbourhood refinement rounds over the leaves.
    pub refine_rounds: usize,
    /// Full Lloyd rounds (every vector against every centroid) after the
    /// neighbourhood rounds.
    pub full_rounds: usize,
    /// The generator seed.
    pub seed: u64,
}

impl Default for KMeansParams {
    fn default() -> Self {
        KMeansParams {
            iterations: 8,
            sample_per_centroid: 128,
            refine_rounds: 8,
            full_rounds: 2,
            seed: 0x5eed_1f5e_ed1f_5eed,
        }
    }
}

/// A partition of vectors into clusters.
#[derive(Debug, Clone, PartialEq)]
pub struct Clustering {
    /// `clusters * dim` floats: each cluster's mean.
    pub centroids: Vec<f32>,
    /// Each vector's cluster, in input order.
    pub assignments: Vec<u32>,
}

impl Clustering {
    /// The number of clusters.
    pub fn len(&self, dim: usize) -> usize {
        self.centroids.len().checked_div(dim).unwrap_or(0)
    }

    /// Whether there are no clusters (no vectors).
    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }
}

/// SplitMix64: a small, fast, seedable generator for sampling.
#[derive(Debug, Clone)]
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..bound` (`bound > 0`), by multiply-shift.
    pub(crate) fn below(&mut self, bound: usize) -> usize {
        let wide = u128::from(self.next_u64()).wrapping_mul(bound as u128);
        // The high word of a 64x64 product is below `bound`.
        usize::try_from(wide >> 64).unwrap_or(0)
    }
}

/// `count` distinct indices of `0..n` (all of them when `count >= n`),
/// ascending.
// ARITH: `i < n` and `j <= i` throughout the partial Fisher-Yates shuffle.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn sample_indices(n: usize, count: usize, rng: &mut SplitMix64) -> Vec<usize> {
    if count >= n {
        return (0..n).collect();
    }
    // Floyd's algorithm: `count` distinct draws without materializing `n`.
    let mut chosen = std::collections::BTreeSet::new();
    for i in n - count..n {
        let j = rng.below(i + 1);
        if !chosen.insert(j) {
            chosen.insert(i);
        }
    }
    chosen.into_iter().collect()
}

/// The vector `i` of a flat `dim`-wide array.
#[inline]
// ARITH: callers pass `i < data.len() / dim`, so `i * dim + dim <= data.len()`.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn row(data: &[f32], dim: usize, i: usize) -> &[f32] {
    &data[i * dim..][..dim]
}

/// The index of `v`'s nearest centroid among `centroids` (non-empty) and
/// its cost; ties go to the lower index.
#[inline]
pub(crate) fn nearest(v: &[f32], centroids: &[f32], dim: usize, metric: Metric) -> (usize, f32) {
    let mut best = (0, f32::INFINITY);
    for (c, centroid) in centroids.chunks_exact(dim).enumerate() {
        let d = metric.cost(v, centroid);
        if d < best.1 {
            best = (c, d);
        }
    }
    best
}

/// Means of the vectors `members` (indices into `data`) grouped by
/// `labels` (one per member, `< k`): `k * dim` floats. A group with no
/// members keeps its `previous` centroid.
// ARITH: labels are `< k` (callers' invariant), so every offset is inside the
// `k * dim` buffers; counts are bounded by `members.len()`.
#[allow(clippy::arithmetic_side_effects)]
fn means(
    data: &[f32],
    dim: usize,
    members: &[u32],
    labels: &[u32],
    k: usize,
    previous: &[f32],
) -> (Vec<f32>, Vec<usize>) {
    const CHUNK: usize = 4096;
    let partials: Vec<(Vec<f64>, Vec<usize>)> = members
        .par_chunks(CHUNK)
        .zip(labels.par_chunks(CHUNK))
        .map(|(ms, ls)| {
            let mut sums = vec![0f64; k * dim];
            let mut counts = vec![0usize; k];
            for (&m, &l) in ms.iter().zip(ls) {
                let l = l as usize;
                counts[l] += 1;
                let sum = &mut sums[l * dim..][..dim];
                for (s, &x) in sum.iter_mut().zip(row(data, dim, m as usize)) {
                    *s += f64::from(x);
                }
            }
            (sums, counts)
        })
        .collect();
    let mut sums = vec![0f64; k * dim];
    let mut counts = vec![0usize; k];
    for (s, c) in partials {
        for (a, b) in sums.iter_mut().zip(&s) {
            *a += b;
        }
        for (a, b) in counts.iter_mut().zip(&c) {
            *a += b;
        }
    }
    let mut out = previous.to_vec();
    for (c, &count) in counts.iter().enumerate() {
        if count > 0 {
            for (o, s) in out[c * dim..][..dim]
                .iter_mut()
                .zip(&sums[c * dim..][..dim])
            {
                *o = (s / count as f64) as f32;
            }
        }
    }
    (out, counts)
}

/// Each member's nearest centroid.
fn assign(
    data: &[f32],
    dim: usize,
    members: &[u32],
    centroids: &[f32],
    metric: Metric,
) -> Vec<u32> {
    members
        .par_iter()
        .map(|&m| {
            let (c, _) = nearest(row(data, dim, m as usize), centroids, dim, metric);
            // Fewer than `u32::MAX` centroids: at most `MAX_BRANCHING`.
            u32::try_from(c).unwrap_or(u32::MAX)
        })
        .collect()
}

/// Splits `members` into at most `parts` groups with k-means trained on a
/// sample; returns the groups' centroids and each member's group.
// ARITH: `parts >= 2` and `members.len() > parts` (caller); sizes are bounded
// by `members.len()`; `largest` and `c` index the `parts`-long vectors.
#[allow(clippy::arithmetic_side_effects)]
fn split(
    data: &[f32],
    dim: usize,
    members: &[u32],
    parts: usize,
    params: &KMeansParams,
    metric: Metric,
    rng: &mut SplitMix64,
) -> (Vec<f32>, Vec<u32>) {
    let sample_size = parts.saturating_mul(params.sample_per_centroid.max(1));
    let sample: Vec<u32> = sample_indices(members.len(), sample_size, rng)
        .into_iter()
        .map(|i| members[i])
        .collect();
    let mut centroids = Vec::with_capacity(parts * dim);
    for i in sample_indices(sample.len(), parts, rng) {
        centroids.extend_from_slice(row(data, dim, sample[i] as usize));
    }
    for _ in 0..params.iterations {
        let labels = assign(data, dim, &sample, &centroids, metric);
        let (next, counts) = means(data, dim, &sample, &labels, parts, &centroids);
        centroids = next;
        metric.adjust(&mut centroids, dim);
        // An empty group takes half of the largest one: its centroid becomes
        // a perturbed copy of the largest group's.
        let mut counts = counts;
        for c in 0..parts {
            if counts[c] > 0 {
                continue;
            }
            let largest = (0..parts).max_by_key(|&i| counts[i]).unwrap_or(0);
            let half = counts[largest] / 2;
            counts[c] = half;
            counts[largest] -= half;
            let (from, to) = (largest * dim, c * dim);
            for j in 0..dim {
                let x = centroids[from + j];
                let eps = 1e-4 * (x.abs() + 1e-3);
                let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
                centroids[to + j] = x + sign * eps;
                centroids[from + j] = x - sign * eps;
            }
        }
    }
    let labels = assign(data, dim, members, &centroids, metric);
    (centroids, labels)
}

/// Splits `members` with [`split`] until every part holds at most `target`
/// vectors (`target >= 1`), appending the parts to `leaves`.
// ARITH: `target >= 1`, so `parts >= 2` after the clamp; labels are `< parts`.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
fn split_down(
    data: &[f32],
    dim: usize,
    members: Vec<u32>,
    target: usize,
    params: &KMeansParams,
    metric: Metric,
    rng: &mut SplitMix64,
    leaves: &mut Vec<Vec<u32>>,
) {
    let mut pending = vec![members];
    while let Some(members) = pending.pop() {
        if members.len() <= target {
            leaves.push(members);
            continue;
        }
        let parts = members.len().div_ceil(target).clamp(2, MAX_BRANCHING);
        let (_, labels) = split(data, dim, &members, parts, params, metric, rng);
        let mut groups = vec![Vec::new(); parts];
        for (&m, &l) in members.iter().zip(&labels) {
            groups[l as usize].push(m);
        }
        let largest = groups.iter().map(Vec::len).max().unwrap_or(0);
        if largest == members.len() {
            // Identical vectors cannot be split by distance: cut them in
            // order instead.
            leaves.extend(members.chunks(target).map(<[u32]>::to_vec));
            continue;
        }
        pending.extend(groups.into_iter().filter(|g| !g.is_empty()));
    }
}

/// One neighbourhood Lloyd assignment: every vector (or, with `active`,
/// every vector of an active cluster) moves to the nearest centroid among
/// its own and that cluster's `neighbors`. A cluster's members share their
/// candidates, so they are scored together with the tiled kernel of
/// [`full_assign`], and as there a winner replaces the own centroid only
/// when the exact [`Metric::cost`] says it is strictly nearer.
// ARITH: cluster ids are below `k`, member indices below `n`, and row
// offsets below the padded lengths.
#[allow(clippy::arithmetic_side_effects)]
fn refine_round(
    data: &[f32],
    dim: usize,
    centroids: &[f32],
    assignments: &[u32],
    neighbors: &[Vec<u32>],
    metric: Metric,
    active: Option<&[bool]>,
) -> Vec<u32> {
    let k = centroids.len() / dim;
    let groups = dim.div_ceil(LANES);
    let mut padded = Vec::new();
    if metric == Metric::InnerProduct {
        pad_rows(centroids, dim, groups, &mut padded);
    }
    // Members of each cluster, in vector order (a counting sort).
    let mut starts = vec![0usize; k + 1];
    for &a in assignments {
        starts[a as usize + 1] += 1;
    }
    for c in 0..k {
        starts[c + 1] += starts[c];
    }
    let mut members = vec![0u32; assignments.len()];
    let mut next = starts.clone();
    for (i, &a) in assignments.iter().enumerate() {
        members[next[a as usize]] = i as u32;
        next[a as usize] += 1;
    }
    let moves: Vec<Vec<(u32, u32)>> = (0..k)
        .into_par_iter()
        .map(|own| {
            let mine = &members[starts[own]..starts[own + 1]];
            if mine.is_empty() || active.is_some_and(|a| !a[own]) {
                return Vec::new();
            }
            // Own centroid first: ties keep a vector where it is.
            let mut ids: Vec<usize> = Vec::with_capacity(neighbors[own].len() + 1);
            ids.push(own);
            ids.extend(neighbors[own].iter().map(|&c| c as usize));
            let last = ids.len() - 1;
            let tiles = ids.len().div_ceil(TILE_CENTROIDS);
            // Euclidean ranks by `x . c - |c|^2 / 2`, which loses precision
            // far from the origin: measure from the own centroid instead,
            // which every candidate is near.
            let origin = match metric {
                Metric::L2 => Some(row(centroids, dim, own)),
                Metric::InnerProduct => None,
            };
            let shifted = |x: &[f32], out: &mut Vec<f32>| match origin {
                Some(o) => out.extend(x.iter().zip(o).map(|(a, b)| a - b)),
                None => out.extend_from_slice(x),
            };
            let mut rows = Vec::with_capacity(mine.len() * dim);
            for &m in mine {
                shifted(row(data, dim, m as usize), &mut rows);
            }
            let mut vecs = Vec::new();
            pad_rows(&rows, dim, groups, &mut vecs);
            // The candidates' padded rows and biases, from the same origin.
            let (local, bias) = if origin.is_some() {
                let mut cs = Vec::with_capacity(ids.len() * dim);
                for &c in &ids {
                    shifted(row(centroids, dim, c), &mut cs);
                }
                let bias: Vec<f32> = cs
                    .chunks_exact(dim)
                    .map(|c| -0.5 * simd::dot_f32(c, c))
                    .collect();
                let mut p = Vec::new();
                pad_rows(&cs, dim, groups, &mut p);
                (Some(p), bias)
            } else {
                (None, vec![0f32; ids.len()])
            };
            // Candidate `j` of `ids`.
            let centroid = |j: usize| match &local {
                Some(p) => &p[j * groups..][..groups],
                None => &padded[ids[j] * groups..][..groups],
            };
            let mut out = Vec::new();
            for base in (0..mine.len()).step_by(TILE_VECTORS) {
                let live = (mine.len() - base).min(TILE_VECTORS);
                let v: [&[Padded]; TILE_VECTORS] =
                    std::array::from_fn(|r| &vecs[(base + r.min(live - 1)) * groups..][..groups]);
                let mut tb = [(f32::NEG_INFINITY, own); TILE_VECTORS];
                for t in 0..tiles {
                    // A repeat of the last candidate pads the tile; it never
                    // wins, as only a strictly higher score replaces.
                    let at: [usize; TILE_CENTROIDS] =
                        std::array::from_fn(|q| (t * TILE_CENTROIDS + q).min(last));
                    let dots = dot_tile(v, at.map(centroid));
                    for (r, best) in tb.iter_mut().enumerate() {
                        for (q, &j) in at.iter().enumerate() {
                            let s = dots[r][q] + bias[j];
                            if s > best.0 {
                                *best = (s, ids[j]);
                            }
                        }
                    }
                }
                for (r, &(_, won)) in tb.iter().enumerate().take(live) {
                    let i = mine[base + r] as usize;
                    let v = row(data, dim, i);
                    if won != own
                        && metric.cost(v, row(centroids, dim, won))
                            < metric.cost(v, row(centroids, dim, own))
                    {
                        out.push((i as u32, won as u32));
                    }
                }
            }
            out
        })
        .collect();
    let mut result = assignments.to_vec();
    for (i, c) in moves.into_iter().flatten() {
        result[i as usize] = c;
    }
    result
}

/// Clusters the `data.len() / dim` vectors of `data` into leaves of at most
/// `target_size` vectors (one cluster when there are no more than that).
/// An empty input has no clusters.
///
/// # Panics
///
/// When `dim` is zero or `data` is not a whole number of vectors.
// ARITH: every count is bounded by the number of vectors (`< u32::MAX`
// asserted below); leaf and cluster indices index vectors of that length.
#[allow(clippy::arithmetic_side_effects)]
pub fn cluster(
    data: &[f32],
    dim: usize,
    target_size: usize,
    params: &KMeansParams,
    metric: Metric,
) -> Clustering {
    assert!(dim > 0, "dimension must be positive");
    assert!(data.len().is_multiple_of(dim), "ragged vector data");
    let n = data.len() / dim;
    assert!(n < u32::MAX as usize, "too many vectors to cluster");
    let target = target_size.max(1);
    if n == 0 {
        return Clustering {
            centroids: Vec::new(),
            assignments: Vec::new(),
        };
    }
    let mut rng = SplitMix64::new(params.seed);
    let mut leaves: Vec<Vec<u32>> = Vec::new();
    split_down(
        data,
        dim,
        all_members(n),
        target,
        params,
        metric,
        &mut rng,
        &mut leaves,
    );
    let mut assignments = vec![0u32; n];
    for (c, leaf) in leaves.iter().enumerate() {
        for &m in leaf {
            assignments[m as usize] = c as u32;
        }
    }
    let k = leaves.len();
    let everyone = all_members(n);
    let (mut centroids, _) = means(data, dim, &everyone, &assignments, k, &vec![0f32; k * dim]);
    metric.adjust(&mut centroids, dim);
    if k > 1 {
        for _ in 0..params.refine_rounds {
            let neighbors = neighbor_lists(&centroids, dim, REFINE_NEIGHBORS);
            assignments = refine_round(
                data,
                dim,
                &centroids,
                &assignments,
                &neighbors,
                metric,
                None,
            );
            centroids = means(data, dim, &everyone, &assignments, k, &centroids).0;
            metric.adjust(&mut centroids, dim);
        }
        for _ in 0..params.full_rounds {
            assignments = full_assign(data, dim, &centroids, &assignments, metric);
            centroids = means(data, dim, &everyone, &assignments, k, &centroids).0;
            metric.adjust(&mut centroids, dim);
        }
    }
    if metric == Metric::InnerProduct {
        centroids = means(data, dim, &everyone, &assignments, k, &centroids).0;
    }
    compact(centroids, assignments, dim, k)
}

/// [`cluster`] applied to each run of consecutive vectors on its own
/// (`runs` gives their lengths, summing to the vector count): every
/// cluster holds vectors of one run only. The clusters of each run follow
/// those of the runs before it.
///
/// # Panics
///
/// As [`cluster`], and when `runs` does not sum to the vector count.
// ARITH: run lengths sum to the vector count, so offsets stay in `data`.
#[allow(clippy::arithmetic_side_effects)]
pub fn cluster_runs(
    data: &[f32],
    dim: usize,
    runs: &[usize],
    target_size: usize,
    params: &KMeansParams,
    metric: Metric,
) -> Clustering {
    assert!(dim > 0, "dimension must be positive");
    assert_eq!(
        runs.iter().sum::<usize>() * dim,
        data.len(),
        "runs cover the vectors"
    );
    let mut out = Clustering {
        centroids: Vec::new(),
        assignments: Vec::with_capacity(data.len() / dim),
    };
    let mut from = 0;
    for &len in runs {
        let run = cluster(
            &data[from * dim..(from + len) * dim],
            dim,
            target_size,
            params,
            metric,
        );
        let base = (out.centroids.len() / dim) as u32;
        out.assignments
            .extend(run.assignments.iter().map(|&a| a + base));
        out.centroids.extend_from_slice(&run.centroids);
        from += len;
    }
    out
}

/// Clusters larger than this many times the target size are split by
/// [`reuse`].
pub const REUSE_SPLIT_FACTOR: usize = 2;
/// Clusters smaller than the target size divided by this are dissolved by
/// [`reuse`].
pub const REUSE_DISSOLVE_DIVISOR: usize = 4;

/// No cluster: a [`reuse`] hint for a vector to place.
pub const NO_CLUSTER: u32 = u32::MAX;

/// Clusters `data` starting from an existing partition (SPFresh-style
/// incremental rebalancing), instead of clustering it afresh.
///
/// `centroids` are the existing clusters' plain means (as [`cluster`]
/// returns them) and `hints` gives each vector's place: `[c, NO_CLUSTER]`
/// keeps it in cluster `c`, `[a, b]` puts it in the nearer of the two (a
/// vector stored in its primary cluster and spilled into another, when the
/// two cannot be told apart), and `[NO_CLUSTER, _]` puts it in its nearest
/// cluster by the full-assignment kernel. Then:
///
/// 1. the means are recomputed and `refine_rounds` neighbourhood Lloyd
///    rounds run over the clusters that received new vectors and the
///    clusters next to them ([`REFINE_NEIGHBORS`]);
/// 2. clusters below `target_size / REUSE_DISSOLVE_DIVISOR` vectors (left
///    small by deletions) are dissolved, their vectors moving to the nearest
///    remaining cluster;
/// 3. clusters above `REUSE_SPLIT_FACTOR * target_size` are split with the
///    same sampled k-means as [`cluster`] down to the target size.
///
/// No cluster ends larger than `REUSE_SPLIT_FACTOR * target_size` or, when
/// there are two or more, smaller than `target_size / REUSE_DISSOLVE_DIVISOR`
/// unless a split made it. The cost is `n * REFINE_NEIGHBORS` comparisons
/// and `k * k` between centroids per round, plus the placed and dissolved
/// vectors against every centroid, against `n * k` per full Lloyd round
/// for [`cluster`]. With no existing clusters this is
/// [`cluster`]. Centroids are compared as [`cluster`] compares them
/// (normalized for [`Metric::InnerProduct`]) and returned as plain means.
/// Deterministic for a given input and `params.seed`.
///
/// # Panics
///
/// When `dim` is zero, `data` or `centroids` is not a whole number of
/// vectors, `hints` is not one per vector, or a hint names no cluster.
// ARITH: counts are bounded by the number of vectors (`< u32::MAX`, asserted)
// and cluster ids by the cluster count, which never exceeds the vector count
// plus the existing clusters.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub fn reuse(
    data: &[f32],
    dim: usize,
    centroids: &[f32],
    hints: &[[u32; 2]],
    target_size: usize,
    params: &KMeansParams,
    refine_rounds: usize,
    metric: Metric,
) -> Clustering {
    assert!(dim > 0, "dimension must be positive");
    assert!(data.len().is_multiple_of(dim), "ragged vector data");
    assert!(centroids.len().is_multiple_of(dim), "ragged centroids");
    let n = data.len() / dim;
    assert!(n < u32::MAX as usize, "too many vectors to cluster");
    assert_eq!(hints.len(), n, "one hint per vector");
    let mut k = centroids.len() / dim;
    assert!(
        hints
            .iter()
            .flatten()
            .all(|&c| c == NO_CLUSTER || (c as usize) < k),
        "hint names no cluster"
    );
    if k == 0 || n == 0 {
        return cluster(data, dim, target_size, params, metric);
    }
    let target = target_size.max(1);
    let mut work = centroids.to_vec();
    metric.adjust(&mut work, dim);
    let mut assignments: Vec<u32> = hints
        .par_iter()
        .enumerate()
        .map(|(i, &[a, b])| {
            if a == NO_CLUSTER || b == NO_CLUSTER {
                return a;
            }
            let v = row(data, dim, i);
            if metric.cost(v, row(&work, dim, b as usize))
                < metric.cost(v, row(&work, dim, a as usize))
            {
                b
            } else {
                a
            }
        })
        .collect();
    let placed: Vec<u32> = (0..n as u32)
        .filter(|&i| assignments[i as usize] == NO_CLUSTER)
        .collect();
    let mut touched = vec![false; k];
    for (&m, &c) in placed
        .iter()
        .zip(&nearest_of(data, dim, &placed, &work, metric))
    {
        assignments[m as usize] = c;
        touched[c as usize] = true;
    }
    let everyone = all_members(n);
    let mut means_now = means(data, dim, &everyone, &assignments, k, centroids).0;
    if k > 1 && touched.contains(&true) {
        for _ in 0..refine_rounds {
            work.clone_from(&means_now);
            metric.adjust(&mut work, dim);
            let neighbors = neighbor_lists(&work, dim, REFINE_NEIGHBORS);
            let active: Vec<bool> = neighbors
                .iter()
                .enumerate()
                .map(|(c, ns)| touched[c] || ns.iter().any(|&o| touched[o as usize]))
                .collect();
            assignments = refine_round(
                data,
                dim,
                &work,
                &assignments,
                &neighbors,
                metric,
                Some(&active),
            );
            means_now = means(data, dim, &everyone, &assignments, k, &means_now).0;
        }
    }
    work.clone_from(&means_now);
    metric.adjust(&mut work, dim);

    // Dissolve the small clusters into the nearest of the others.
    let mut counts = vec![0usize; k];
    for &a in &assignments {
        counts[a as usize] += 1;
    }
    let min_size = (target / REUSE_DISSOLVE_DIVISOR).max(1);
    let mut keep: Vec<bool> = counts.iter().map(|&c| c >= min_size).collect();
    if !keep.contains(&true) {
        let largest = (0..k)
            .max_by_key(|&c| (counts[c], std::cmp::Reverse(c)))
            .unwrap_or(0);
        keep[largest] = true;
    }
    let survivors: Vec<u32> = (0..k as u32).filter(|&c| keep[c as usize]).collect();
    let orphans: Vec<u32> = (0..n as u32)
        .filter(|&i| !keep[assignments[i as usize] as usize])
        .collect();
    if !orphans.is_empty() {
        let mut kept = Vec::with_capacity(survivors.len() * dim);
        for &c in &survivors {
            kept.extend_from_slice(row(&work, dim, c as usize));
        }
        for (&m, &c) in orphans
            .iter()
            .zip(&nearest_of(data, dim, &orphans, &kept, metric))
        {
            let to = survivors[c as usize];
            assignments[m as usize] = to;
            counts[to as usize] += 1;
        }
    }

    // Split the large ones: the first part keeps the cluster's id.
    let max_size = target.saturating_mul(REUSE_SPLIT_FACTOR);
    let large: Vec<usize> = (0..k)
        .filter(|&c| keep[c] && counts[c] > max_size)
        .collect();
    if !large.is_empty() {
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); k];
        for (i, &a) in assignments.iter().enumerate() {
            if counts[a as usize] > max_size {
                members[a as usize].push(i as u32);
            }
        }
        let mut rng = SplitMix64::new(params.seed ^ 0x5b11_7c1a_55e5);
        for c in large {
            let mut leaves = Vec::new();
            split_down(
                data,
                dim,
                std::mem::take(&mut members[c]),
                target,
                params,
                metric,
                &mut rng,
                &mut leaves,
            );
            for (j, leaf) in leaves.iter().enumerate() {
                let id = if j == 0 {
                    c as u32
                } else {
                    k += 1;
                    (k - 1) as u32
                };
                for &m in leaf {
                    assignments[m as usize] = id;
                }
            }
        }
        means_now.resize(k * dim, 0.0);
    }
    let centroids = means(data, dim, &everyone, &assignments, k, &means_now).0;
    compact(centroids, assignments, dim, k)
}

/// The nearest of `centroids` (non-empty) to each of `members`, by the
/// full-assignment kernel over a copy of their vectors.
fn nearest_of(
    data: &[f32],
    dim: usize,
    members: &[u32],
    centroids: &[f32],
    metric: Metric,
) -> Vec<u32> {
    if members.is_empty() {
        return Vec::new();
    }
    let mut rows = Vec::with_capacity(members.len().saturating_mul(dim));
    for &m in members {
        rows.extend_from_slice(row(data, dim, m as usize));
    }
    full_assign(&rows, dim, centroids, &vec![0; members.len()], metric)
}

/// Lanes of the full-assignment kernel.
const LANES: usize = 8;
/// Rows of one kernel tile: vectors, and centroids.
const TILE_VECTORS: usize = 4;
const TILE_CENTROIDS: usize = 3;
/// Vectors per parallel block of the full assignment.
const BLOCK_VECTORS: usize = 256;
/// Centroid tiles per cache-resident run.
const BLOCK_TILES: usize = 32;

/// A row zero-padded to whole [`LANES`] groups.
type Padded = [f32; LANES];

/// Copies `rows` rows of `dim` floats into `out`, each zero-padded to
/// `groups` lanes groups.
// ARITH: `i < rows`, `j < dim <= groups * LANES`.
#[allow(clippy::arithmetic_side_effects)]
fn pad_rows(src: &[f32], dim: usize, groups: usize, out: &mut Vec<Padded>) {
    let rows = src.len() / dim;
    out.clear();
    out.resize(rows * groups, [0f32; LANES]);
    for (i, r) in src.chunks_exact(dim).enumerate() {
        for (j, &x) in r.iter().enumerate() {
            out[i * groups + j / LANES][j % LANES] = x;
        }
    }
}

/// `a * b + c`, fused only where the target has the instruction: without
/// it `f32::mul_add` is a software `fmaf` call per element.
#[inline(always)]
fn fmadd(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(any(target_feature = "fma", target_arch = "aarch64")) {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// The dot products of [`TILE_VECTORS`] vectors with [`TILE_CENTROIDS`]
/// centroids, all padded to the same number of lane groups.
#[inline(always)]
// ARITH: fixed-size array indices.
#[allow(clippy::arithmetic_side_effects)]
fn dot_tile(
    v: [&[Padded]; TILE_VECTORS],
    c: [&[Padded]; TILE_CENTROIDS],
) -> [[f32; TILE_CENTROIDS]; TILE_VECTORS] {
    let groups = c[0].len();
    let c = c.map(|c| &c[..groups]);
    let v = v.map(|v| &v[..groups]);
    let mut acc = [[[0f32; LANES]; TILE_CENTROIDS]; TILE_VECTORS];
    for g in 0..groups {
        let cg = c.map(|c| c[g]);
        let vg = v.map(|v| v[g]);
        for (ar, x) in acc.iter_mut().zip(&vg) {
            for (a, cl) in ar.iter_mut().zip(&cg) {
                for l in 0..LANES {
                    a[l] = fmadd(x[l], cl[l], a[l]);
                }
            }
        }
    }
    acc.map(|row| row.map(|a| a.iter().sum()))
}

/// Each vector's nearest centroid among all of `centroids`, by `metric`.
/// The kernel's winner replaces the `current` centroid only when the exact
/// [`Metric::cost`] says it is strictly nearer: Euclidean scores are ranked
/// as `x . c - |c|^2 / 2`, which loses precision for large norms, and a tie
/// keeps the vector where it is, so duplicates cut apart by size stay apart.
// ARITH: offsets are below `n * groups` and `k * groups`; `k < u32::MAX`
// (one centroid per leaf, fewer leaves than vectors).
#[allow(clippy::arithmetic_side_effects)]
fn full_assign(
    data: &[f32],
    dim: usize,
    centroids: &[f32],
    current: &[u32],
    metric: Metric,
) -> Vec<u32> {
    let groups = dim.div_ceil(LANES);
    let k = centroids.len() / dim;
    let mut padded = Vec::new();
    pad_rows(centroids, dim, groups, &mut padded);
    // Euclidean: the nearest centroid maximizes `x . c - |c|^2 / 2`.
    let bias: Vec<f32> = match metric {
        Metric::L2 => centroids
            .chunks_exact(dim)
            .map(|c| -0.5 * simd::dot_f32(c, c))
            .collect(),
        Metric::InnerProduct => vec![0f32; k],
    };
    let centroid = |c: usize| &padded[c * groups..][..groups];
    // Pad the centroid count to whole tiles by repeating the last centroid:
    // a repeat never wins, ties go to the lower index.
    let tiles = k.div_ceil(TILE_CENTROIDS);
    data.par_chunks(BLOCK_VECTORS * dim)
        .zip(current.par_chunks(BLOCK_VECTORS))
        .flat_map_iter(|(rows, cur)| {
            let count = rows.len() / dim;
            let mut vecs = Vec::new();
            pad_rows(rows, dim, groups, &mut vecs);
            let mut best = vec![(f32::NEG_INFINITY, 0usize); count];
            // Centroids in cache-sized runs, each against the whole block
            // of vectors, so a run is read from memory once per block.
            for run in (0..tiles).step_by(BLOCK_TILES) {
                let run_end = (run + BLOCK_TILES).min(tiles);
                for base in (0..count).step_by(TILE_VECTORS) {
                    let live = (count - base).min(TILE_VECTORS);
                    let v: [&[Padded]; TILE_VECTORS] = std::array::from_fn(|r| {
                        &vecs[(base + r.min(live - 1)) * groups..][..groups]
                    });
                    let mut tb = [(f32::NEG_INFINITY, 0usize); TILE_VECTORS];
                    for t in run..run_end {
                        let ids: [usize; TILE_CENTROIDS] =
                            std::array::from_fn(|q| (t * TILE_CENTROIDS + q).min(k - 1));
                        let dots = dot_tile(v, ids.map(centroid));
                        for r in 0..TILE_VECTORS {
                            for (q, &id) in ids.iter().enumerate() {
                                let s = dots[r][q] + bias[id];
                                if s > tb[r].0 {
                                    tb[r] = (s, id);
                                }
                            }
                        }
                    }
                    for (b, t) in best[base..base + live].iter_mut().zip(tb) {
                        if t.0 > b.0 {
                            *b = t;
                        }
                    }
                }
            }
            (0..count).map(move |i| {
                let v = row(rows, dim, i);
                let (own, won) = (cur[i] as usize, best[i].1);
                if won != own
                    && metric.cost(v, row(centroids, dim, won))
                        < metric.cost(v, row(centroids, dim, own))
                {
                    won as u32
                } else {
                    cur[i]
                }
            })
        })
        .collect()
}

fn all_members(n: usize) -> Vec<u32> {
    // `n < u32::MAX`, checked by `cluster`.
    (0..u32::try_from(n).unwrap_or(u32::MAX)).collect()
}

/// Drops clusters the refinement emptied and renumbers the rest.
// ARITH: cluster ids are `< k` and `next <= k`.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn compact(
    centroids: Vec<f32>,
    mut assignments: Vec<u32>,
    dim: usize,
    k: usize,
) -> Clustering {
    let mut counts = vec![0usize; k];
    for &a in &assignments {
        counts[a as usize] += 1;
    }
    if counts.iter().all(|&c| c > 0) {
        return Clustering {
            centroids,
            assignments,
        };
    }
    let mut renumber = vec![u32::MAX; k];
    let mut kept = Vec::with_capacity(centroids.len());
    let mut next = 0u32;
    for (c, &count) in counts.iter().enumerate() {
        if count > 0 {
            renumber[c] = next;
            next += 1;
            kept.extend_from_slice(row(&centroids, dim, c));
        }
    }
    for a in &mut assignments {
        *a = renumber[*a as usize];
    }
    Clustering {
        centroids: kept,
        assignments,
    }
}

/// The candidates [`soar_assign`] considers per vector: the primary
/// cluster's nearest neighbouring clusters.
pub const SOAR_CANDIDATES: usize = 32;

/// Spilling with orthogonality-amplified residuals (SOAR, Sun et al., 2023):
/// a second cluster for every vector, chosen so that the two assignments
/// fail on different queries.
///
/// A query misses a vector's primary cluster when the query is far from that
/// centroid along the vector's residual `r = x - c1`. The second cluster
/// minimizes `|x - c|^2 + lambda * <r, x - c>^2 / |r|^2`: close to `x`,
/// with a residual as orthogonal to `r` as possible. Candidates are the
/// primary cluster's [`SOAR_CANDIDATES`] nearest clusters. A vector equal to
/// its centroid, or alone in a one-cluster field, gets `u32::MAX` (none).
// ARITH: assignments and neighbour ids are cluster indices, below `k`.
#[allow(clippy::arithmetic_side_effects)]
pub fn soar_assign(data: &[f32], dim: usize, clustering: &Clustering, lambda: f32) -> Vec<u32> {
    let k = clustering.len(dim);
    if k < 2 {
        return vec![u32::MAX; clustering.assignments.len()];
    }
    let neighbors = neighbor_lists(&clustering.centroids, dim, SOAR_CANDIDATES);
    clustering
        .assignments
        .par_iter()
        .enumerate()
        .map_init(
            || vec![0f32; dim],
            |r, (i, &primary)| {
                let x = row(data, dim, i);
                let c1 = row(&clustering.centroids, dim, primary as usize);
                for ((r, &a), &b) in r.iter_mut().zip(x).zip(c1) {
                    *r = a - b;
                }
                let rn = simd::dot_f32(r, r);
                if rn <= 0.0 {
                    return u32::MAX;
                }
                let rx = simd::dot_f32(r, x);
                let mut best = (u32::MAX, f32::INFINITY);
                for &c in &neighbors[primary as usize] {
                    let centroid = row(&clustering.centroids, dim, c as usize);
                    let parallel = rx - simd::dot_f32(r, centroid);
                    let loss =
                        simd::square_distance_f32(x, centroid) + lambda * parallel * parallel / rn;
                    if loss < best.1 {
                        best = (c, loss);
                    }
                }
                best.0
            },
        )
        .collect()
}

/// Each centroid's `m` nearest other centroids, nearest first.
// ARITH: indices are below the centroid count.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn neighbor_lists(centroids: &[f32], dim: usize, m: usize) -> Vec<Vec<u32>> {
    let k = centroids.len() / dim;
    (0..k)
        .into_par_iter()
        .map(|c| {
            let v = row(centroids, dim, c);
            let mut d: Vec<(f32, u32)> = (0..k)
                .filter(|&o| o != c)
                .map(|o| {
                    (
                        simd::square_distance_f32(v, row(centroids, dim, o)),
                        o as u32,
                    )
                })
                .collect();
            let take = m.min(d.len());
            if take < d.len() {
                d.select_nth_unstable_by(take, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                d.truncate(take);
            }
            d.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            d.into_iter().map(|(_, o)| o).collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;

    /// `blobs` well-separated Gaussian-ish blobs of `per` points each.
    fn blobs(blobs: usize, per: usize, dim: usize) -> Vec<f32> {
        let mut rng = SplitMix64::new(1);
        let mut out = Vec::new();
        for b in 0..blobs {
            for _ in 0..per {
                for j in 0..dim {
                    let center = if j == b % dim {
                        100.0 * (b / dim + 1) as f32
                    } else {
                        0.0
                    };
                    let noise = (rng.below(1000) as f32 / 1000.0) - 0.5;
                    out.push(center + noise);
                }
            }
        }
        out
    }

    #[test]
    fn separated_blobs_become_their_own_leaves() {
        let dim = 8;
        let data = blobs(6, 50, dim);
        let c = cluster(&data, dim, 60, &KMeansParams::default(), Metric::L2);
        let k = c.len(dim);
        assert!(k >= 6, "{k} clusters");
        assert!(!c.is_empty());
        // No cluster mixes blobs.
        for b in 0..6 {
            let first = c.assignments[b * 50];
            for i in 0..50 {
                assert_eq!(c.assignments[b * 50 + i], first);
            }
        }
        let mut sizes = vec![0; k];
        for &a in &c.assignments {
            sizes[a as usize] += 1;
        }
        assert!(sizes.iter().all(|&s| s > 0 && s <= 60), "{sizes:?}");
    }

    #[test]
    fn deterministic_for_a_seed() {
        let data = blobs(10, 40, 4);
        let p = KMeansParams::default();
        assert_eq!(
            cluster(&data, 4, 25, &p, Metric::L2),
            cluster(&data, 4, 25, &p, Metric::L2)
        );
        let ip = cluster(&data, 4, 25, &p, Metric::InnerProduct);
        assert_eq!(ip.assignments.len(), 400);
        assert_eq!(Metric::InnerProduct.cost(&[1.0, 2.0], &[3.0, 1.0]), -5.0);
    }

    #[test]
    fn small_and_degenerate_inputs() {
        let empty = cluster(&[], 3, 10, &KMeansParams::default(), Metric::L2);
        assert!(empty.is_empty() && empty.assignments.is_empty());
        assert_eq!(empty.len(3), 0);
        let one = cluster(
            &[1.0, 2.0, 3.0, 5.0],
            2,
            10,
            &KMeansParams::default(),
            Metric::L2,
        );
        assert_eq!(one.centroids, vec![2.0, 3.5]);
        assert_eq!(one.assignments, vec![0, 0]);
        // Identical vectors: cut in order.
        let same = vec![1.0f32; 2 * 25];
        let c = cluster(&same, 2, 10, &KMeansParams::default(), Metric::L2);
        assert_eq!(c.len(2), 3);
        // A zero target behaves as one.
        let c = cluster(&[0.0, 1.0, 2.0], 1, 0, &KMeansParams::default(), Metric::L2);
        assert_eq!(c.len(1), 3);
    }

    #[test]
    fn refinement_off_still_partitions() {
        let data = blobs(5, 30, 6);
        let p = KMeansParams {
            refine_rounds: 0,
            full_rounds: 0,
            ..KMeansParams::default()
        };
        let c = cluster(&data, 6, 40, &p, Metric::L2);
        assert_eq!(c.assignments.len(), 150);
        assert!(c.len(6) >= 4);
    }

    #[test]
    fn empty_clusters_are_compacted() {
        let c = compact(vec![0.0, 1.0, 2.0], vec![0, 2, 2], 1, 3);
        assert_eq!(c.centroids, vec![0.0, 2.0]);
        assert_eq!(c.assignments, vec![0, 1, 1]);
    }

    #[test]
    fn samples_are_distinct_and_sorted() {
        let mut rng = SplitMix64::new(3);
        let s = sample_indices(100, 30, &mut rng);
        assert_eq!(s.len(), 30);
        assert!(s.windows(2).all(|w| w[0] < w[1]) && s[29] < 100);
        assert_eq!(sample_indices(3, 5, &mut rng), vec![0, 1, 2]);
    }

    #[test]
    fn soar_picks_a_different_nearby_cluster() {
        let dim = 4;
        let data = blobs(6, 30, dim);
        let c = cluster(&data, dim, 30, &KMeansParams::default(), Metric::L2);
        let spill = soar_assign(&data, dim, &c, 1.0);
        assert_eq!(spill.len(), c.assignments.len());
        let k = c.len(dim) as u32;
        for (s, a) in spill.iter().zip(&c.assignments) {
            assert!(*s == u32::MAX || (*s != *a && *s < k));
        }
        assert!(spill.iter().any(|&s| s != u32::MAX));
        let one = cluster(&data[..8], dim, 30, &KMeansParams::default(), Metric::L2);
        assert_eq!(soar_assign(&data[..8], dim, &one, 1.0), vec![u32::MAX; 2]);
        // A vector on its centroid does not spill.
        let two = Clustering {
            centroids: vec![0.0, 0.0, 5.0, 5.0],
            assignments: vec![0, 1],
        };
        assert_eq!(
            soar_assign(&[0.0, 0.0, 5.0, 4.0], 2, &two, 1.0),
            vec![u32::MAX, 0]
        );
    }

    #[test]
    fn neighbors_exclude_self_and_sort() {
        let n = neighbor_lists(&[0.0, 10.0, 1.0, 3.0], 1, 2);
        assert_eq!(n[0], vec![2, 3]);
        assert_eq!(n[1], vec![3, 2]);
        assert_eq!(neighbor_lists(&[0.0, 1.0], 1, 5)[0], vec![1]);
    }

    #[test]
    fn full_assignment_matches_brute_force() {
        let mut rng = SplitMix64::new(7);
        let mut random = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| rng.below(2001) as f32 / 1000.0 - 1.0)
                .collect()
        };
        // Dimensions around the lane width; counts that leave partial
        // tiles and partial blocks.
        for dim in [1, 7, 8, 13] {
            for (n, k) in [(1, 1), (5, 2), (300, 7), (517, 100)] {
                let data = random(n * dim);
                let centroids = random(k * dim);
                let current = vec![0u32; n];
                for metric in [Metric::L2, Metric::InnerProduct] {
                    let got = full_assign(&data, dim, &centroids, &current, metric);
                    for (i, &g) in got.iter().enumerate() {
                        let v = row(&data, dim, i);
                        let (want, best) = nearest(v, &centroids, dim, metric);
                        let cost = metric.cost(v, row(&centroids, dim, g as usize));
                        assert!(
                            g as usize == want || (cost - best).abs() <= 1e-4 * (1.0 + best.abs()),
                            "dim {dim} n {n} k {k} {metric:?} vector {i}: {g} vs {want}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn full_assignment_keeps_ties_in_place() {
        // Two identical centroids: each vector stays with the one it has.
        let centroids = [1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
        let data = [1.0, 0.0, 1.0, 0.0, 0.9, 0.1];
        for metric in [Metric::L2, Metric::InnerProduct] {
            assert_eq!(
                full_assign(&data, 2, &centroids, &[1, 0, 2], metric),
                vec![1, 0, 0]
            );
        }
    }

    #[test]
    fn full_rounds_alone_partition() {
        let data = blobs(6, 30, 3);
        let p = KMeansParams {
            refine_rounds: 0,
            full_rounds: 3,
            ..KMeansParams::default()
        };
        let c = cluster(&data, 3, 30, &p, Metric::L2);
        // No cluster mixes blobs.
        let mut blob_of = vec![usize::MAX; c.len(3)];
        for (i, &a) in c.assignments.iter().enumerate() {
            let b = &mut blob_of[a as usize];
            assert!(*b == usize::MAX || *b == i / 30, "cluster {a} mixes blobs");
            *b = i / 30;
        }
    }

    fn sizes(c: &Clustering, dim: usize) -> Vec<usize> {
        let mut out = vec![0; c.len(dim)];
        for &a in &c.assignments {
            out[a as usize] += 1;
        }
        out
    }

    /// Each blob's mean, as `cluster` would compute it.
    fn blob_means(data: &[f32], dim: usize, blobs: usize, per: usize) -> Vec<f32> {
        let labels: Vec<u32> = (0..blobs * per).map(|i| (i / per) as u32).collect();
        let members = all_members(blobs * per);
        means(data, dim, &members, &labels, blobs, &vec![0.0; blobs * dim]).0
    }

    #[test]
    fn reuse_keeps_hinted_vectors_and_places_the_rest() {
        let dim = 4;
        let data = blobs(6, 40, dim);
        let centroids = blob_means(&data, dim, 5, 40);
        // Blobs 0..5 keep their cluster; blob 5 is new.
        let hints: Vec<[u32; 2]> = (0..240)
            .map(|i| {
                if i < 200 {
                    [(i / 40) as u32, NO_CLUSTER]
                } else {
                    [NO_CLUSTER; 2]
                }
            })
            .collect();
        for (rounds, metric) in [(0, Metric::L2), (2, Metric::L2), (1, Metric::InnerProduct)] {
            let p = KMeansParams::default();
            let c = reuse(&data, dim, &centroids, &hints, 40, &p, rounds, metric);
            assert_eq!(
                c,
                reuse(&data, dim, &centroids, &hints, 40, &p, rounds, metric)
            );
            if metric == Metric::InnerProduct {
                continue;
            }
            assert_eq!(c.len(dim), 5, "{rounds}");
            for i in 0..200 {
                assert_eq!(c.assignments[i], (i / 40) as u32);
            }
            // Blob 5 sits nearest blob 1 and joins it whole (80 is not over
            // twice the target).
            assert!(c.assignments[200..].iter().all(|&a| a == 1));
            assert_eq!(sizes(&c, dim), vec![40, 80, 40, 40, 40]);
            // Centroids are the clusters' means.
            assert_eq!(&c.centroids[..dim], &centroids[..dim]);
        }
    }

    #[test]
    fn reuse_splits_large_clusters() {
        let dim = 4;
        let data = blobs(6, 40, dim);
        let centroids = blob_means(&data, dim, 2, 40);
        let hints = vec![[NO_CLUSTER; 2]; 240];
        let c = reuse(
            &data,
            dim,
            &centroids,
            &hints,
            40,
            &KMeansParams::default(),
            1,
            Metric::L2,
        );
        let s = sizes(&c, dim);
        assert_eq!(s.iter().sum::<usize>(), 240);
        assert!(s.iter().all(|&x| x <= 80), "{s:?}");
        assert!(s.len() >= 3, "{s:?}");
        // Identical vectors over the limit are cut in order.
        let same = vec![1.0f32; 2 * 25];
        let c = reuse(
            &same,
            2,
            &[1.0, 1.0],
            &[[0, NO_CLUSTER]; 25],
            10,
            &KMeansParams::default(),
            0,
            Metric::L2,
        );
        assert_eq!(sizes(&c, 2), vec![10, 10, 5]);
    }

    #[test]
    fn reuse_dissolves_small_clusters() {
        let dim = 4;
        let all = blobs(6, 40, dim);
        let centroids = blob_means(&all, dim, 6, 40);
        // Blob 5 lost all but five vectors.
        let data = &all[..205 * dim];
        let hints: Vec<[u32; 2]> = (0..205).map(|i| [(i / 40) as u32, NO_CLUSTER]).collect();
        let p = KMeansParams::default();
        let c = reuse(data, dim, &centroids, &hints, 40, &p, 1, Metric::L2);
        assert_eq!(sizes(&c, dim), vec![40, 45, 40, 40, 40]);
        assert!(c.assignments[200..].iter().all(|&a| a == 1));
        // When every cluster is small the largest one takes everything.
        let c = reuse(
            &[0.0, 1.0, 10.0, 11.0, 20.0],
            1,
            &[0.5, 10.5, 20.0],
            &[
                [0, 1],
                [0, NO_CLUSTER],
                [1, NO_CLUSTER],
                [1, 0],
                [2, NO_CLUSTER],
            ],
            100,
            &p,
            1,
            Metric::L2,
        );
        assert_eq!(c.assignments, vec![0; 5]);
        assert_eq!(c.centroids, vec![8.4]);
    }

    #[test]
    fn refinement_moves_only_vectors_of_active_clusters() {
        let data = [4.0, 6.0];
        let centroids = [0.0, 10.0];
        let neighbors = neighbor_lists(&centroids, 1, 1);
        let r = |active: Option<&[bool]>| {
            refine_round(
                &data,
                1,
                &centroids,
                &[1, 0],
                &neighbors,
                Metric::L2,
                active,
            )
        };
        assert_eq!(r(None), vec![0, 1]);
        assert_eq!(r(Some(&[true, false])), vec![1, 1]);
    }

    #[test]
    fn reuse_picks_the_nearer_of_two_hinted_clusters() {
        let c = reuse(
            &[9.0, 0.0, 1.0, 0.0],
            2,
            &[0.0, 0.0, 10.0, 0.0],
            &[[0, 1], [1, 0]],
            1,
            &KMeansParams::default(),
            0,
            Metric::L2,
        );
        assert_eq!(c.assignments, vec![1, 0]);
        assert_eq!(c.centroids, vec![1.0, 0.0, 9.0, 0.0]);
    }

    #[test]
    fn reuse_without_clusters_or_vectors_clusters_afresh() {
        let data = blobs(3, 20, 2);
        let p = KMeansParams::default();
        let hints = vec![[NO_CLUSTER; 2]; 60];
        assert_eq!(
            reuse(&data, 2, &[], &hints, 20, &p, 1, Metric::L2),
            cluster(&data, 2, 20, &p, Metric::L2)
        );
        assert!(reuse(&[], 2, &[1.0, 2.0], &[], 20, &p, 1, Metric::L2).is_empty());
    }

    #[test]
    #[should_panic(expected = "hint names no cluster")]
    fn reuse_rejects_a_hint_past_the_clusters() {
        reuse(
            &[1.0],
            1,
            &[1.0],
            &[[1, NO_CLUSTER]],
            4,
            &KMeansParams::default(),
            0,
            Metric::L2,
        );
    }

    #[test]
    #[should_panic(expected = "one hint per vector")]
    fn reuse_needs_one_hint_per_vector() {
        reuse(
            &[1.0],
            1,
            &[1.0],
            &[],
            4,
            &KMeansParams::default(),
            0,
            Metric::L2,
        );
    }

    #[test]
    #[should_panic(expected = "ragged")]
    fn ragged_input_panics() {
        cluster(&[1.0, 2.0, 3.0], 2, 4, &KMeansParams::default(), Metric::L2);
    }
}
