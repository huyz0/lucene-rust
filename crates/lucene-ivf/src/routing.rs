//! Centroid routing: which clusters a query probes.
//!
//! A flat scan of every f32 centroid costs `nlist * dim` multiply-adds and
//! reads `4 * nlist * dim` bytes per query, which at a few thousand
//! centroids of a few hundred dimensions is a large share of a search. The
//! router scans 8-bit centroid codes instead (a quarter of the bytes) and
//! rescores only the best few at full precision, which returns the same
//! probe order as the f32 scan except for near ties. A field with at least
//! [`TREE_MIN_CLUSTERS`] clusters also stores a routing tree: each centroid's
//! parent, from k-means over the centroids. A query then scores the parents
//! and only the codes of the nearest parents' children, which trades a
//! little recall at a given `nprobe` for much less routing work.
//!
//! The 8-bit codes and the parent centroids are derived from the stored f32
//! centroids (and parents) when a field is opened.

use lucene_util::{simd, vector_util};

use crate::kmeans::{self, KMeansParams, Metric};
use crate::Similarity;

/// How a field ranks its centroids for a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Routing {
    /// Score every centroid at full precision.
    Exact,
    /// Score every centroid's 8-bit code, rescore the best exactly.
    Quantized,
    /// Score parent centroids, then the 8-bit codes of the nearest parents'
    /// children until `cover * nprobe` centroids are scored, then rescore
    /// the best exactly.
    Tree {
        /// Centroids scored per probe.
        cover: usize,
    },
    /// [`Routing::Tree`] with [`TREE_COVER`] when the field stores a
    /// routing tree, [`Routing::Quantized`] otherwise.
    Auto,
}

/// From this many clusters on, a field stores a routing tree by default
/// (`IvfParams::routing_tree_min`), which [`Routing::Auto`] routes through.
pub const TREE_MIN_CLUSTERS: usize = 8192;
/// Centroids per parent in the routing tree.
pub const TREE_FANOUT: usize = 64;

/// How a field's similarity ranks centroids: `score = w * dot(q, c) + b`
/// per centroid, larger is nearer.
fn weights(similarity: Similarity, centroids: &[f32], dim: usize) -> (Vec<f32>, Vec<f32>) {
    let n = centroids.len() / dim;
    match similarity {
        // -|q - c|^2 = 2 q.c - |c|^2 - |q|^2; the last term is the same for
        // every centroid.
        Similarity::Euclidean => (
            vec![2.0; n],
            centroids
                .chunks_exact(dim)
                .map(|c| -simd::dot_f32(c, c))
                .collect(),
        ),
        // Spherically clustered fields rank by the cosine with the centroid.
        s if crate::writer::spherical(s) => (
            centroids
                .chunks_exact(dim)
                .map(|c| {
                    let norm = simd::dot_f32(c, c).sqrt();
                    if norm > 0.0 {
                        1.0 / norm
                    } else {
                        0.0
                    }
                })
                .collect(),
            vec![0.0; n],
        ),
        _ => (vec![1.0; n], vec![0.0; n]),
    }
}

/// Signed 8-bit codes of `rows` after subtracting `mean`, one scale per row:
/// `row - mean ~= scale * code`.
// ARITH: indices stay inside `rows`, which is a whole number of `dim` rows.
#[allow(clippy::arithmetic_side_effects)]
fn encode(rows: &[f32], dim: usize, mean: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let n = rows.len() / dim;
    let mut codes = vec![0u8; n * dim];
    let mut scales = Vec::with_capacity(n);
    for (row, out) in rows.chunks_exact(dim).zip(codes.chunks_exact_mut(dim)) {
        let max = row
            .iter()
            .zip(mean)
            .fold(0f32, |m, (&x, &u)| m.max((x - u).abs()));
        let scale = max / 127.0;
        let inv = if max > 0.0 { 127.0 / max } else { 0.0 };
        for ((o, &x), &u) in out.iter_mut().zip(row).zip(mean) {
            *o = ((x - u) * inv).round().clamp(-127.0, 127.0) as i8 as u8;
        }
        scales.push(scale);
    }
    (codes, scales)
}

/// A set of centroids scored through 8-bit codes.
#[derive(Debug, Clone)]
struct Level {
    dim: usize,
    /// The mean of the rows, subtracted before encoding so the codes spend
    /// their range on what tells the rows apart.
    mean: Vec<f32>,
    codes: Vec<u8>,
    /// Per row: `w * scale`, what an 8-bit dot is multiplied by.
    code_weight: Vec<f32>,
    /// Per row: the score weight `w` and offset `b`.
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Level {
    fn new(rows: &[f32], dim: usize, similarity: Similarity) -> Self {
        let n = rows.len() / dim;
        let mut mean = vec![0f32; dim];
        for r in rows.chunks_exact(dim) {
            for (m, &x) in mean.iter_mut().zip(r) {
                *m += x;
            }
        }
        if n > 0 {
            let inv = 1.0 / n as f32;
            mean.iter_mut().for_each(|m| *m *= inv);
        }
        let (codes, scales) = encode(rows, dim, &mean);
        let (w, b) = weights(similarity, rows, dim);
        let code_weight = w.iter().zip(&scales).map(|(&w, &s)| w * s).collect();
        Level {
            dim,
            mean,
            codes,
            code_weight,
            w,
            b,
        }
    }

    fn len(&self) -> usize {
        self.w.len()
    }

    /// Approximate scores of rows `rows` (indices into this level) or of
    /// every row, appended to `out` as `(score, row)`.
    // ARITH: row indices are below `len()`; code offsets stay in `codes`.
    #[allow(clippy::arithmetic_side_effects)]
    fn score(&self, q: &QueryCode, rows: Option<&[u32]>, out: &mut Vec<(f32, u32)>) {
        let qm = simd::dot_f32(&q.raw, &self.mean);
        let mut one = |r: usize| {
            let code = &self.codes[r * self.dim..(r + 1) * self.dim];
            let dot = vector_util::dot_product_i8(&q.codes, code) as f32;
            let s = self.code_weight[r] * q.scale * dot + self.w[r] * qm + self.b[r];
            out.push((s, r as u32));
        };
        match rows {
            Some(rows) => rows.iter().for_each(|&r| one(r as usize)),
            None => (0..self.len()).for_each(one),
        }
    }
}

/// A query's 8-bit code: `raw ~= scale * codes`.
#[derive(Debug)]
struct QueryCode {
    raw: Vec<f32>,
    codes: Vec<u8>,
    scale: f32,
}

impl QueryCode {
    fn new(q: &[f32]) -> Self {
        let zero = vec![0f32; q.len()];
        let (codes, scales) = encode(q, q.len(), &zero);
        QueryCode {
            raw: q.to_vec(),
            codes,
            scale: scales[0],
        }
    }
}

/// The parents of a routing tree and which centroids each one covers.
#[derive(Debug, Clone)]
struct Tree {
    parents: Level,
    /// `children[starts[p]..starts[p + 1]]` are parent `p`'s centroids.
    starts: Vec<usize>,
    children: Vec<u32>,
}

/// What the tree clusters and compares: unit centroids unless the field
/// is Euclidean.
fn tree_input(centroids: &[f32], dim: usize, similarity: Similarity) -> (Vec<f32>, Similarity) {
    let mut rows = centroids.to_vec();
    if similarity == Similarity::Euclidean {
        return (rows, similarity);
    }
    for c in rows.chunks_exact_mut(dim) {
        let _ = vector_util::l2normalize(c, false);
    }
    (rows, Similarity::Cosine)
}

/// Each centroid's parent in a routing tree over `centroids`: k-means over
/// the centroids into groups of about [`TREE_FANOUT`]. Parents are numbered
/// from zero, and every parent has at least one centroid.
pub fn tree_parents(centroids: &[f32], dim: usize, similarity: Similarity) -> Vec<u32> {
    let (input, routed) = tree_input(centroids, dim, similarity);
    let metric = if routed == Similarity::Euclidean {
        Metric::L2
    } else {
        Metric::InnerProduct
    };
    let params = KMeansParams {
        refine_rounds: 4,
        full_rounds: 2,
        ..KMeansParams::default()
    };
    kmeans::cluster(&input, dim, TREE_FANOUT, &params, metric).assignments
}

impl Tree {
    /// The tree whose centroid `c` has parent `parents[c]`; parents are the
    /// means of their centroids (unit centroids unless Euclidean).
    // ARITH: parent numbers are small (checked by the reader), and rows
    // index `centroids`.
    #[allow(clippy::arithmetic_side_effects)]
    fn new(centroids: &[f32], dim: usize, similarity: Similarity, parents: &[u32]) -> Tree {
        let (input, routed) = tree_input(centroids, dim, similarity);
        let count = parents.iter().map(|&p| p as usize + 1).max().unwrap_or(0);
        let mut starts = vec![0usize; count + 1];
        let mut means = vec![0f32; count * dim];
        for (c, &p) in parents.iter().enumerate() {
            let p = p as usize;
            starts[p + 1] += 1;
            let row = kmeans::row(&input, dim, c);
            for (m, &x) in means[p * dim..(p + 1) * dim].iter_mut().zip(row) {
                *m += x;
            }
        }
        for p in 0..count {
            let size = starts[p + 1];
            if size > 0 {
                let inv = 1.0 / size as f32;
                means[p * dim..(p + 1) * dim]
                    .iter_mut()
                    .for_each(|m| *m *= inv);
            }
            starts[p + 1] += starts[p];
        }
        let mut fill = starts.clone();
        let mut children = vec![0u32; parents.len()];
        for (c, &p) in parents.iter().enumerate() {
            children[fill[p as usize]] = c as u32;
            fill[p as usize] += 1;
        }
        Tree {
            parents: Level::new(&means, dim, routed),
            starts,
            children,
        }
    }
}

/// Ranks a field's clusters for a query.
#[derive(Debug, Clone)]
pub struct Router {
    dim: usize,
    exact: bool,
    leaves: Level,
    tree: Option<Tree>,
    /// Through a tree, how many centroids are scored per probe.
    cover: usize,
}

/// Centroids scored per probe by [`Routing::Auto`]'s tree.
pub const TREE_COVER: usize = 32;

/// Approximate candidates rescored at full precision for `nprobe` probes.
fn rescored(nprobe: usize) -> usize {
    nprobe.saturating_mul(2).max(nprobe.saturating_add(32))
}

/// Ties by cluster number, so the order is deterministic.
fn by_score(a: &(f32, u32), b: &(f32, u32)) -> std::cmp::Ordering {
    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
}

/// Keeps the `n` best of `v`, unordered.
fn keep_best(v: &mut Vec<(f32, u32)>, n: usize) {
    if n < v.len() {
        v.select_nth_unstable_by(n, by_score);
        v.truncate(n);
    }
}

impl Router {
    /// A router over `centroids` (`nlist * dim` floats). `parents` is the
    /// routing tree stored with the field, if any (see [`tree_parents`]):
    /// [`Routing::Auto`] uses it when present and never clusters at open;
    /// [`Routing::Tree`] clusters the centroids when it is absent.
    pub fn new(
        centroids: &[f32],
        dim: usize,
        similarity: Similarity,
        routing: Routing,
        parents: Option<&[u32]>,
    ) -> Self {
        let nlist = centroids.len().checked_div(dim).unwrap_or(0);
        let leaves = Level::new(centroids, dim, similarity);
        let (exact, tree, cover) = match routing {
            Routing::Exact => (true, None, 1),
            Routing::Quantized => (false, None, 1),
            Routing::Tree { cover } => {
                let tree = match parents {
                    Some(p) => Some(Tree::new(centroids, dim, similarity, p)),
                    None => (nlist >= 2 * TREE_FANOUT).then(|| {
                        let p = tree_parents(centroids, dim, similarity);
                        Tree::new(centroids, dim, similarity, &p)
                    }),
                };
                (false, tree, cover.max(1))
            }
            Routing::Auto => (
                false,
                parents.map(|p| Tree::new(centroids, dim, similarity, p)),
                TREE_COVER,
            ),
        };
        Router {
            dim,
            exact,
            leaves,
            tree,
            cover,
        }
    }

    /// The number of clusters.
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    /// Whether there are no clusters.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether queries route through a tree.
    pub fn has_tree(&self) -> bool {
        self.tree.is_some()
    }

    /// The exact score of centroid `c` (see [`weights`]).
    fn exact(&self, centroids: &[f32], q: &[f32], c: usize) -> f32 {
        let row = kmeans::row(centroids, self.dim, c);
        self.leaves.w[c] * simd::dot_f32(q, row) + self.leaves.b[c]
    }

    /// The `nprobe` clusters nearest to `q`, nearest first. `centroids` are
    /// the f32 centroids the router was built from, used to rescore the
    /// best approximate candidates exactly.
    pub fn route(&self, centroids: &[f32], q: &[f32], nprobe: usize) -> Vec<usize> {
        let nprobe = nprobe.min(self.len());
        if nprobe == 0 {
            return Vec::new();
        }
        // A full order (as calibration asks for) is exact anyway.
        if self.exact || nprobe == self.len() {
            let mut all: Vec<(f32, u32)> = (0..self.len())
                .map(|c| (self.exact(centroids, q, c), c as u32))
                .collect();
            keep_best(&mut all, nprobe);
            all.sort_unstable_by(by_score);
            return all.into_iter().map(|(_, c)| c as usize).collect();
        }
        let code = QueryCode::new(q);
        let mut cand = Vec::new();
        match &self.tree {
            None => self.leaves.score(&code, None, &mut cand),
            Some(tree) => {
                let mut parents = Vec::with_capacity(tree.parents.len());
                tree.parents.score(&code, None, &mut parents);
                parents.sort_unstable_by(by_score);
                // Cover a few times more centroids than are probed: a
                // probe's nearest centroids can sit under several parents.
                let want = nprobe.saturating_mul(self.cover).max(rescored(nprobe));
                let mut covered = 0usize;
                let mut rows = Vec::with_capacity(want.saturating_mul(2));
                for (i, &(_, p)) in parents.iter().enumerate() {
                    if covered >= want && i >= 4 {
                        break;
                    }
                    let p = p as usize;
                    let kids = &tree.children[tree.starts[p]..tree.starts[p + 1]];
                    rows.extend_from_slice(kids);
                    covered = covered.saturating_add(kids.len());
                }
                self.leaves.score(&code, Some(&rows), &mut cand);
            }
        }
        keep_best(&mut cand, rescored(nprobe));
        for c in &mut cand {
            c.0 = self.exact(centroids, q, c.1 as usize);
        }
        keep_best(&mut cand, nprobe);
        cand.sort_unstable_by(by_score);
        cand.into_iter().map(|(_, c)| c as usize).collect()
    }
}

/// The `nprobe` clusters nearest to `q` by a full f32 scan, nearest first:
/// what [`Router::route`] approximates.
pub fn exact_route(
    centroids: &[f32],
    dim: usize,
    similarity: Similarity,
    q: &[f32],
    nprobe: usize,
) -> Vec<usize> {
    let (w, b) = weights(similarity, centroids, dim);
    let mut scored: Vec<(f32, u32)> = centroids
        .chunks_exact(dim)
        .enumerate()
        .map(|(c, row)| (w[c] * simd::dot_f32(q, row) + b[c], c as u32))
        .collect();
    keep_best(&mut scored, nprobe);
    scored.sort_unstable_by(by_score);
    scored.into_iter().map(|(_, c)| c as usize).collect()
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;
    use crate::kmeans::SplitMix64;

    fn random(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        (0..n * dim)
            .map(|_| (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
            .collect()
    }

    const SIMILARITIES: [Similarity; 4] = [
        Similarity::Euclidean,
        Similarity::Cosine,
        Similarity::DotProduct,
        Similarity::MaximumInnerProduct,
    ];

    #[test]
    fn exact_routing_matches_the_f32_scan() {
        let dim = 24;
        let centroids = random(300, dim, 1);
        let queries = random(20, dim, 2);
        for s in SIMILARITIES {
            let r = Router::new(&centroids, dim, s, Routing::Exact, None);
            for q in queries.chunks_exact(dim) {
                assert_eq!(
                    r.route(&centroids, q, 17),
                    exact_route(&centroids, dim, s, q, 17)
                );
            }
        }
    }

    #[test]
    fn quantized_routing_finds_the_nearest_centroids() {
        let dim = 64;
        let centroids = random(2000, dim, 3);
        let queries = random(50, dim, 4);
        for s in SIMILARITIES {
            let r = Router::new(&centroids, dim, s, Routing::Quantized, None);
            let mut found = 0;
            for q in queries.chunks_exact(dim) {
                let got = r.route(&centroids, q, 32);
                let want = exact_route(&centroids, dim, s, q, 32);
                assert_eq!(got.len(), 32);
                // The nearest centroid is always found and ranked first.
                assert_eq!(got[0], want[0], "{s:?}");
                found += got.iter().filter(|c| want.contains(c)).count();
            }
            assert!(found * 100 >= 99 * 50 * 32, "{s:?}: {found}");
        }
    }

    #[test]
    fn tree_routing_returns_distinct_clusters_nearest_first() {
        let dim = 16;
        let centroids = random(3000, dim, 5);
        let queries = random(30, dim, 6);
        for s in SIMILARITIES {
            let r = Router::new(&centroids, dim, s, Routing::Tree { cover: 32 }, None);
            assert!(r.has_tree());
            let (w, b) = weights(s, &centroids, dim);
            let mut found = 0;
            for q in queries.chunks_exact(dim) {
                let got = r.route(&centroids, q, 20);
                assert_eq!(got.len(), 20);
                let mut seen = got.clone();
                seen.sort_unstable();
                seen.dedup();
                assert_eq!(seen.len(), 20);
                let score =
                    |c: usize| w[c] * simd::dot_f32(q, kmeans::row(&centroids, dim, c)) + b[c];
                assert!(got.windows(2).all(|p| score(p[0]) >= score(p[1])));
                let want = exact_route(&centroids, dim, s, q, 20);
                found += got.iter().filter(|c| want.contains(c)).count();
            }
            assert!(found * 10 >= 8 * 30 * 20, "{s:?}: {found}");
        }
    }

    #[test]
    fn small_and_empty_fields() {
        let dim = 8;
        for routing in [Routing::Exact, Routing::Quantized, Routing::Auto] {
            let r = Router::new(&[], dim, Similarity::Euclidean, routing, None);
            assert!(r.is_empty());
            assert!(r.route(&[], &[0.0; 8], 4).is_empty());
            let centroids = random(5, dim, 7);
            let r = Router::new(&centroids, dim, Similarity::Cosine, routing, None);
            let mut got = r.route(&centroids, &centroids[..dim], 10);
            assert_eq!(got[0], 0);
            got.sort_unstable();
            assert_eq!(got, vec![0, 1, 2, 3, 4]);
        }
        // Too few clusters for a tree: routes flat.
        let centroids = random(50, dim, 8);
        let r = Router::new(
            &centroids,
            dim,
            Similarity::Euclidean,
            Routing::Tree { cover: 4 },
            None,
        );
        assert!(!r.has_tree());
        assert_eq!(r.route(&centroids, &centroids[..dim], 50).len(), 50);
    }

    #[test]
    fn zero_query_and_identical_centroids() {
        let dim = 8;
        let centroids = vec![0.25f32; 200 * dim];
        for s in SIMILARITIES {
            for routing in [Routing::Quantized, Routing::Tree { cover: 2 }] {
                let r = Router::new(&centroids, dim, s, routing, None);
                for q in [vec![0f32; dim], vec![1f32; dim]] {
                    let got = r.route(&centroids, &q, 7);
                    assert_eq!(got.len(), 7);
                }
            }
        }
    }
}
