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
    /// The generator seed.
    pub seed: u64,
}

impl Default for KMeansParams {
    fn default() -> Self {
        KMeansParams {
            iterations: 8,
            sample_per_centroid: 128,
            refine_rounds: 8,
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
    let all: Vec<u32> = (0..n as u32).collect();
    let mut pending = vec![all];
    let mut leaves: Vec<Vec<u32>> = Vec::new();
    while let Some(members) = pending.pop() {
        if members.len() <= target {
            leaves.push(members);
            continue;
        }
        let parts = members.len().div_ceil(target).clamp(2, MAX_BRANCHING);
        let (_, labels) = split(data, dim, &members, parts, params, metric, &mut rng);
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
    let mut assignments = vec![0u32; n];
    for (c, leaf) in leaves.iter().enumerate() {
        for &m in leaf {
            assignments[m as usize] = c as u32;
        }
    }
    let k = leaves.len();
    let (mut centroids, _) = means(
        data,
        dim,
        &all_members(n),
        &assignments,
        k,
        &vec![0f32; k * dim],
    );
    metric.adjust(&mut centroids, dim);
    if k > 1 {
        for _ in 0..params.refine_rounds {
            let neighbors = neighbor_lists(&centroids, dim, REFINE_NEIGHBORS);
            assignments = (0..n)
                .into_par_iter()
                .map(|i| {
                    let own = assignments[i] as usize;
                    let v = row(data, dim, i);
                    let mut best = (own, metric.cost(v, row(&centroids, dim, own)));
                    for &c in &neighbors[own] {
                        let d = metric.cost(v, row(&centroids, dim, c as usize));
                        // Ties keep the vector where it is, so duplicates
                        // cut apart by size stay apart.
                        if d < best.1 {
                            best = (c as usize, d);
                        }
                    }
                    best.0 as u32
                })
                .collect();
            centroids = means(data, dim, &all_members(n), &assignments, k, &centroids).0;
            metric.adjust(&mut centroids, dim);
        }
    }
    if metric == Metric::InnerProduct {
        centroids = means(data, dim, &all_members(n), &assignments, k, &centroids).0;
    }
    compact(centroids, assignments, dim, k)
}

fn all_members(n: usize) -> Vec<u32> {
    // `n < u32::MAX`, checked by `cluster`.
    (0..u32::try_from(n).unwrap_or(u32::MAX)).collect()
}

/// Drops clusters the refinement emptied and renumbers the rest.
// ARITH: cluster ids are `< k` and `next <= k`.
#[allow(clippy::arithmetic_side_effects)]
fn compact(centroids: Vec<f32>, mut assignments: Vec<u32>, dim: usize, k: usize) -> Clustering {
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
    #[should_panic(expected = "ragged")]
    fn ragged_input_panics() {
        cluster(&[1.0, 2.0, 3.0], 2, 4, &KMeansParams::default(), Metric::L2);
    }
}
