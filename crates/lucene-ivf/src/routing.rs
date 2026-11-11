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

/// A centroid's exact routing score, larger is nearer: minus the squared
/// distance for Euclidean, the dot product times `1 / |c|` for the
/// spherically clustered Cosine and DotProduct, and the dot product for
/// MaximumInnerProduct, whose score is the dot product itself.
#[derive(Debug, Clone)]
struct ExactScore {
    euclidean: bool,
    /// Per centroid, the factor on the dot product (not for Euclidean).
    scale: Vec<f32>,
}

impl ExactScore {
    fn new(similarity: Similarity, centroids: &[f32], dim: usize) -> Self {
        let euclidean = similarity == Similarity::Euclidean;
        let scale = if euclidean {
            Vec::new()
        } else if crate::writer::spherical(similarity) {
            centroids.chunks_exact(dim).map(inverse_norm).collect()
        } else {
            let n = centroids.len().checked_div(dim).unwrap_or(0);
            vec![1.0; n]
        };
        ExactScore { euclidean, scale }
    }

    fn score(&self, centroids: &[f32], dim: usize, q: &[f32], c: usize) -> f32 {
        let row = kmeans::row(centroids, dim, c);
        if self.euclidean {
            -simd::square_distance_f32(q, row)
        } else {
            simd::dot_f32(q, row) * self.scale.get(c).copied().unwrap_or(1.0)
        }
    }
}

fn inverse_norm(c: &[f32]) -> f32 {
    let norm = simd::dot_f32(c, c).sqrt();
    if norm > 0.0 {
        1.0 / norm
    } else {
        0.0
    }
}

/// Signed 8-bit code of `row - mean` into `out`, and its scale:
/// `row - mean ~= scale * code`.
fn encode_row(row: &[f32], mean: Option<&[f32]>, out: &mut [u8]) -> f32 {
    let at = |i: usize, x: f32| mean.map_or(x, |m| x - m[i]);
    let max = row
        .iter()
        .enumerate()
        .fold(0f32, |m, (i, &x)| m.max(at(i, x).abs()));
    let inv = if max > 0.0 { 127.0 / max } else { 0.0 };
    for (i, (o, &x)) in out.iter_mut().zip(row).enumerate() {
        *o = (at(i, x) * inv).round().clamp(-127.0, 127.0) as i8 as u8;
    }
    max / 127.0
}

/// The query side of [`simd::dot_u7_i8`]: `q ~= scale * (code - 64)`, each
/// code in `1..=127`, padded with 64 (zero) to `out`'s length.
fn encode_query(q: &[f32], out: &mut [u8]) -> f32 {
    let max = q.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let inv = if max > 0.0 { 63.0 / max } else { 0.0 };
    out.fill(QUERY_ZERO);
    for (o, &x) in out.iter_mut().zip(q) {
        // ARITH: the rounded value is in -63..=63.
        #[allow(clippy::arithmetic_side_effects)]
        let v = (x * inv).round().clamp(-63.0, 63.0) as i8 + QUERY_ZERO as i8;
        *o = v as u8;
    }
    max / 63.0
}

/// The query code of zero (see [`encode_query`]).
const QUERY_ZERO: u8 = 64;

/// Code rows are padded to whole 32-byte vectors for the scan kernel.
const CODE_ALIGN: usize = 32;

/// A set of centroids scored through 8-bit codes.
///
/// A row's score is `code_weight * scale * dot + add * a`, where `dot` is
/// the 8-bit dot product and `scale` the query code's scale. Euclidean rows
/// score `-|q - c|^2 + |q - m|^2 = 2 (q - m).(c - m) - |c - m|^2`, all in
/// coordinates centred on the level's mean `m`, which keeps data far from
/// the origin well conditioned: `add` is `-|c - m|^2` and `a` is 1.
/// Dot-scored rows score `w * q.c = w * (q.(c - m) + q.m)`: `add` is `w`
/// and `a` is `q.m`.
#[derive(Debug, Clone)]
struct Level {
    /// Bytes per code row: the dimension rounded up to [`CODE_ALIGN`].
    stride: usize,
    /// The mean of the rows, subtracted before encoding so the codes spend
    /// their range on what tells the rows apart.
    mean: Vec<f32>,
    /// One zero-padded row of signed codes per centroid.
    codes: Vec<u8>,
    /// Per row: the sum of its codes, which undoes the query codes' offset
    /// of [`QUERY_ZERO`].
    code_sums: Vec<i32>,
    /// Per row: what an 8-bit dot product (times the query's scale) is
    /// multiplied by.
    code_weight: Vec<f32>,
    /// Per row: the score's term outside the dot product.
    add: Vec<f32>,
    euclidean: bool,
}

/// A query's 8-bit code, reused across the levels of one routing.
#[derive(Debug, Default)]
struct QueryCode {
    centred: Vec<f32>,
    codes: Vec<u8>,
    scale: f32,
}

impl Level {
    /// `rows` scored as Euclidean distances, or (`weights` given, one per
    /// level row) as their dot products with the query times `weights`.
    /// With `order`, level row `i` is `rows` row `order[i]`.
    // ARITH: indices stay inside `rows`, a whole number of `dim` rows.
    #[allow(clippy::arithmetic_side_effects)]
    fn new(rows: &[f32], dim: usize, order: Option<&[u32]>, weights: Option<Vec<f32>>) -> Self {
        let n = order.map_or(rows.len().checked_div(dim).unwrap_or(0), <[u32]>::len);
        let row = |i: usize| kmeans::row(rows, dim, order.map_or(i, |o| o[i] as usize));
        let rows = || (0..n).map(row);
        let mut mean = vec![0f32; dim];
        for r in rows() {
            for (m, &x) in mean.iter_mut().zip(r) {
                *m += x;
            }
        }
        if n > 0 {
            let inv = 1.0 / n as f32;
            mean.iter_mut().for_each(|m| *m *= inv);
        }
        let stride = dim.next_multiple_of(CODE_ALIGN);
        let mut codes = vec![0u8; n * stride];
        let scales: Vec<f32> = rows()
            .zip(codes.chunks_exact_mut(stride))
            .map(|(r, out)| encode_row(r, Some(&mean), &mut out[..dim]))
            .collect();
        let code_sums = codes
            .chunks_exact(stride)
            .map(|c| c.iter().map(|&b| i32::from(b as i8)).sum())
            .collect();
        let euclidean = weights.is_none();
        let (code_weight, add) = match weights {
            None => (
                scales.iter().map(|&s| 2.0 * s).collect(),
                rows()
                    .map(|r| -simd::square_distance_f32(r, &mean))
                    .collect(),
            ),
            Some(w) => (scales.iter().zip(&w).map(|(&s, &w)| s * w).collect(), w),
        };
        Level {
            stride,
            mean,
            codes,
            code_sums,
            code_weight,
            add,
            euclidean,
        }
    }

    fn len(&self) -> usize {
        self.code_weight.len()
    }

    /// Encodes `q` as this level scores it.
    fn encode(&self, q: &[f32], code: &mut QueryCode) {
        code.codes.resize(self.stride, QUERY_ZERO);
        code.scale = if self.euclidean {
            code.centred.clear();
            code.centred
                .extend(q.iter().zip(&self.mean).map(|(&x, &m)| x - m));
            encode_query(&code.centred, &mut code.codes)
        } else {
            encode_query(q, &mut code.codes)
        };
    }

    /// Approximate scores of the rows in `ranges` (row index ranges of
    /// this level) or of every row, offered to `out` as `(score, row)`.
    // ARITH: row indices are below `len()`; code offsets stay in `codes`.
    #[allow(clippy::arithmetic_side_effects)]
    fn score(
        &self,
        q: &[f32],
        code: &mut QueryCode,
        ranges: Option<&[std::ops::Range<usize>]>,
        out: &mut Best,
    ) {
        self.encode(q, code);
        let a = if self.euclidean {
            1.0
        } else {
            simd::dot_f32(q, &self.mean)
        };
        let zero = i32::from(QUERY_ZERO);
        let scale = code.scale;
        let qc = &code.codes[..];
        let row = |r: usize| &self.codes[r * self.stride..(r + 1) * self.stride];
        let mut finish = |r: usize, dot: i32| {
            let dot = (dot - zero * self.code_sums[r]) as f32;
            out.offer(
                self.code_weight[r] * scale * dot + self.add[r] * a,
                r as u32,
            );
        };
        let mut scan = |range: std::ops::Range<usize>| {
            let whole = range.start + range.len() / 4 * 4;
            for r in (range.start..whole).step_by(4) {
                let dots = simd::dot_u7_i8_x4(qc, [row(r), row(r + 1), row(r + 2), row(r + 3)]);
                for (i, d) in dots.into_iter().enumerate() {
                    finish(r + i, d);
                }
            }
            for r in whole..range.end {
                finish(r, simd::dot_u7_i8(qc, row(r)));
            }
        };
        match ranges {
            Some(ranges) => ranges.iter().for_each(|r| scan(r.clone())),
            None => scan(0..self.len()),
        }
    }
}

/// The parents of a routing tree and which centroids each one covers.
#[derive(Debug, Clone)]
struct Tree {
    parents: Level,
    /// `children[starts[p]..starts[p + 1]]` are parent `p`'s centroids. The
    /// leaf codes are stored in this order, so a parent's children are one
    /// contiguous run of rows.
    starts: Vec<usize>,
    children: Vec<u32>,
}

/// What the tree clusters: unit centroids unless the field is Euclidean.
fn tree_input(centroids: &[f32], dim: usize, similarity: Similarity) -> Vec<f32> {
    let mut rows = centroids.to_vec();
    if similarity != Similarity::Euclidean {
        for c in rows.chunks_exact_mut(dim) {
            let _ = vector_util::l2normalize(c, false);
        }
    }
    rows
}

/// Each centroid's parent in a routing tree over `centroids`: k-means over
/// the centroids (unit centroids unless Euclidean) into groups of about
/// [`TREE_FANOUT`]. Parents are numbered densely from zero: every parent
/// below [`parent_count`] has a centroid.
pub fn tree_parents(centroids: &[f32], dim: usize, similarity: Similarity) -> Vec<u32> {
    let input = tree_input(centroids, dim, similarity);
    let metric = if similarity == Similarity::Euclidean {
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

/// The number of parents `parents` names: one more than the largest.
pub fn parent_count(parents: &[u32]) -> usize {
    parents
        .iter()
        .map(|&p| (p as usize).saturating_add(1))
        .max()
        .unwrap_or(0)
}

impl Tree {
    /// The tree whose centroid `c` has parent `parents[c]`, or `None` unless
    /// there is one parent per centroid and the parents are dense (as
    /// [`tree_parents`] returns and the reader checks). A parent is the mean
    /// of its centroids (unit centroids unless Euclidean). Dot-scored
    /// parents rank by the cosine with that mean, times, for
    /// MaximumInnerProduct, the largest norm among its centroids, since the
    /// raw dot product ranks the centroids themselves.
    // ARITH: parents are checked below `nlist`, and rows index `centroids`.
    #[allow(clippy::arithmetic_side_effects)]
    fn new(centroids: &[f32], dim: usize, similarity: Similarity, parents: &[u32]) -> Option<Tree> {
        let nlist = centroids.len().checked_div(dim).unwrap_or(0);
        let count = parent_count(parents);
        if parents.len() != nlist || count > nlist {
            return None;
        }
        let input = tree_input(centroids, dim, similarity);
        let mut starts = vec![0usize; count + 1];
        let mut means = vec![0f32; count * dim];
        let mut max_norm = vec![0f32; count];
        for (c, &p) in parents.iter().enumerate() {
            let p = p as usize;
            starts[p + 1] += 1;
            for (m, &x) in means[p * dim..(p + 1) * dim]
                .iter_mut()
                .zip(kmeans::row(&input, dim, c))
            {
                *m += x;
            }
            let row = kmeans::row(centroids, dim, c);
            max_norm[p] = max_norm[p].max(simd::dot_f32(row, row).sqrt());
        }
        if starts[1..].contains(&0) {
            return None;
        }
        for p in 0..count {
            let inv = 1.0 / starts[p + 1] as f32;
            means[p * dim..(p + 1) * dim]
                .iter_mut()
                .for_each(|m| *m *= inv);
            starts[p + 1] += starts[p];
        }
        let mut fill = starts.clone();
        let mut children = vec![0u32; nlist];
        for (c, &p) in parents.iter().enumerate() {
            children[fill[p as usize]] = c as u32;
            fill[p as usize] += 1;
        }
        let weights = (similarity != Similarity::Euclidean).then(|| {
            means
                .chunks_exact(dim)
                .zip(&max_norm)
                .map(|(m, &norm)| {
                    let w = inverse_norm(m);
                    if similarity == Similarity::MaximumInnerProduct {
                        w * norm
                    } else {
                        w
                    }
                })
                .collect()
        });
        Some(Tree {
            parents: Level::new(&means, dim, None, weights),
            starts,
            children,
        })
    }
}

/// Ranks a field's clusters for a query.
#[derive(Debug, Clone)]
pub struct Router {
    dim: usize,
    nlist: usize,
    exact: ExactScore,
    /// The centroids' 8-bit codes, in the tree's child order when there is
    /// a tree; `None` for [`Routing::Exact`].
    leaves: Option<Level>,
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

/// Ties by row number, so the order is deterministic.
fn by_score(a: &(f32, u32), b: &(f32, u32)) -> std::cmp::Ordering {
    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
}

/// Collects the `n` best of many `(score, row)` offers. Offers below the
/// worst of the best `n` seen at the last compaction are dropped on sight,
/// so most of a large scan never reaches the buffer.
#[derive(Debug)]
struct Best {
    n: usize,
    floor: f32,
    items: Vec<(f32, u32)>,
}

impl Best {
    /// The best `n` of at most `offers` offers.
    fn new(n: usize, offers: usize) -> Self {
        Best {
            n,
            floor: f32::NEG_INFINITY,
            items: Vec::with_capacity(n.saturating_mul(4).max(16).min(offers.max(n))),
        }
    }

    #[inline]
    fn offer(&mut self, score: f32, row: u32) {
        if score < self.floor {
            return;
        }
        if self.items.len() == self.items.capacity() {
            keep_best(&mut self.items, self.n);
            self.floor = self.items.iter().fold(f32::INFINITY, |m, &(s, _)| m.min(s));
            if score < self.floor {
                return;
            }
        }
        self.items.push((score, row));
    }

    /// The best `n`, unordered.
    fn into_items(mut self) -> Vec<(f32, u32)> {
        keep_best(&mut self.items, self.n);
        self.items
    }
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
    /// [`Routing::Tree`] clusters the centroids when it is absent. Parents
    /// that do not describe a tree over these centroids are ignored.
    pub fn new(
        centroids: &[f32],
        dim: usize,
        similarity: Similarity,
        routing: Routing,
        parents: Option<&[u32]>,
    ) -> Self {
        let nlist = centroids.len().checked_div(dim).unwrap_or(0);
        let stored = || parents.and_then(|p| Tree::new(centroids, dim, similarity, p));
        let (tree, cover) = match routing {
            Routing::Exact | Routing::Quantized => (None, 1),
            Routing::Tree { cover } => {
                let tree = stored().or_else(|| {
                    (nlist >= TREE_FANOUT.saturating_mul(2)).then(|| {
                        let p = tree_parents(centroids, dim, similarity);
                        Tree::new(centroids, dim, similarity, &p)
                    })?
                });
                (tree, cover.max(1))
            }
            Routing::Auto => (stored(), TREE_COVER),
        };
        let exact = ExactScore::new(similarity, centroids, dim);
        let leaves = (routing != Routing::Exact).then(|| match &tree {
            None => {
                let weights = (!exact.euclidean).then(|| exact.scale.clone());
                Level::new(centroids, dim, None, weights)
            }
            Some(tree) => {
                let weights = (!exact.euclidean).then(|| {
                    tree.children
                        .iter()
                        .map(|&c| exact.scale[c as usize])
                        .collect()
                });
                Level::new(centroids, dim, Some(&tree.children), weights)
            }
        });
        Router {
            dim,
            nlist,
            exact,
            leaves,
            tree,
            cover,
        }
    }

    /// The number of clusters.
    pub fn len(&self) -> usize {
        self.nlist
    }

    /// Whether there are no clusters.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether queries route through a tree.
    pub fn has_tree(&self) -> bool {
        self.tree.is_some()
    }

    /// The `nprobe` clusters nearest to `q`, nearest first. `centroids` are
    /// the f32 centroids the router was built from, used to rescore the
    /// best approximate candidates exactly.
    // ARITH: parent numbers index `starts`, which has one more entry.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn route(&self, centroids: &[f32], q: &[f32], nprobe: usize) -> Vec<usize> {
        let nprobe = nprobe.min(self.len());
        if nprobe == 0 {
            return Vec::new();
        }
        let exact = |c: usize| self.exact.score(centroids, self.dim, q, c);
        let leaves = match &self.leaves {
            // A full order is exact anyway.
            Some(leaves) if nprobe < self.len() => leaves,
            _ => {
                let mut all: Vec<(f32, u32)> =
                    (0..self.len()).map(|c| (exact(c), c as u32)).collect();
                keep_best(&mut all, nprobe);
                all.sort_unstable_by(by_score);
                return all.into_iter().map(|(_, c)| c as usize).collect();
            }
        };
        assert_eq!(q.len(), self.dim, "query dimension");
        let mut code = QueryCode::default();
        let n = rescored(nprobe);
        let mut best;
        match &self.tree {
            None => {
                best = Best::new(n, leaves.len());
                leaves.score(q, &mut code, None, &mut best);
            }
            Some(tree) => {
                let count = tree.parents.len();
                let mut parents = Best::new(count, count);
                tree.parents.score(q, &mut code, None, &mut parents);
                let mut parents = parents.into_items();
                parents.sort_unstable_by(by_score);
                // Cover several times more centroids than are probed: a
                // probe's nearest centroids can sit under several parents.
                let want = nprobe.saturating_mul(self.cover).max(rescored(nprobe));
                let mut ranges = Vec::new();
                let mut covered = 0;
                for (i, &(_, p)) in parents.iter().enumerate() {
                    if covered >= want && i >= 4 {
                        break;
                    }
                    let p = p as usize;
                    covered += tree.starts[p + 1] - tree.starts[p];
                    ranges.push(tree.starts[p]..tree.starts[p + 1]);
                }
                best = Best::new(n, covered);
                leaves.score(q, &mut code, Some(&ranges), &mut best);
            }
        }
        let mut cand = best.into_items();
        for c in &mut cand {
            if let Some(tree) = &self.tree {
                c.1 = tree.children[c.1 as usize];
            }
            c.0 = exact(c.1 as usize);
        }
        keep_best(&mut cand, nprobe);
        cand.sort_unstable_by(by_score);
        cand.into_iter().map(|(_, c)| c as usize).collect()
    }
}

impl Router {
    /// The `nprobe` clusters of `among` nearest to `q`, nearest first, by
    /// their exact scores: a filter's few clusters need no approximation.
    pub fn route_among(
        &self,
        centroids: &[f32],
        q: &[f32],
        among: &[usize],
        nprobe: usize,
    ) -> Vec<usize> {
        let nprobe = nprobe.min(among.len());
        let mut all: Vec<(f32, u32)> = among
            .iter()
            .filter(|&&c| c < self.len())
            .map(|&c| (self.exact.score(centroids, self.dim, q, c), c as u32))
            .collect();
        keep_best(&mut all, nprobe);
        all.sort_unstable_by(by_score);
        all.into_iter().map(|(_, c)| c as usize).collect()
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
    Router::new(centroids, dim, similarity, Routing::Exact, None).route(centroids, q, nprobe)
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
            let exact = ExactScore::new(s, &centroids, dim);
            let mut found = 0;
            for q in queries.chunks_exact(dim) {
                let got = r.route(&centroids, q, 20);
                assert_eq!(got.len(), 20);
                let mut seen = got.clone();
                seen.sort_unstable();
                seen.dedup();
                assert_eq!(seen.len(), 20);
                let score = |c: usize| exact.score(&centroids, dim, q, c);
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

    #[test]
    fn euclidean_routing_far_from_the_origin() {
        let dim = 32;
        let s = Similarity::Euclidean;
        // How many of the exact 16 nearest centroids each routing finds, with
        // every coordinate shifted by `offset`.
        let found = |offset: f32| {
            let mut centroids = random(2000, dim, 9);
            let mut queries = random(40, dim, 10);
            for x in centroids.iter_mut().chain(queries.iter_mut()) {
                *x += offset;
            }
            // The exact path is the f32 squared distance itself.
            let r = Router::new(&centroids, dim, s, Routing::Exact, None);
            for q in queries.chunks_exact(dim) {
                let got = r.route(&centroids, q, 8);
                let d = |c: usize| simd::square_distance_f32(q, kmeans::row(&centroids, dim, c));
                let best = (0..2000).map(d).fold(f32::INFINITY, f32::min);
                assert_eq!(d(got[0]), best);
            }
            [Routing::Quantized, Routing::Tree { cover: 32 }].map(|routing| {
                let r = Router::new(&centroids, dim, s, routing, None);
                queries
                    .chunks_exact(dim)
                    .map(|q| {
                        let want = exact_route(&centroids, dim, s, q, 16);
                        let got = r.route(&centroids, q, 16);
                        got.iter().filter(|c| want.contains(c)).count()
                    })
                    .sum::<usize>()
            })
        };
        let near = found(0.0);
        let far = found(1000.0);
        // The codes are centred, so routing far from the origin is as good
        // as near it (the tree is weak on structureless data either way).
        assert!(far[0] * 10 >= 9 * 40 * 16, "{far:?}");
        assert!(far[1] * 20 >= near[1] * 19, "{near:?} {far:?}");
    }

    #[test]
    fn bad_parents_are_ignored() {
        let dim = 8;
        let centroids = random(300, dim, 11);
        let good = tree_parents(&centroids, dim, Similarity::Euclidean);
        assert_eq!(parent_count(&good), {
            let mut p = good.clone();
            p.sort_unstable();
            p.dedup();
            p.len()
        });
        let short = &good[..299];
        let mut gap = good.clone();
        gap.iter_mut().for_each(|p| *p *= 2);
        let mut huge = good.clone();
        huge[0] = u32::MAX;
        for parents in [short, &gap[..], &huge[..]] {
            let r = Router::new(
                &centroids,
                dim,
                Similarity::Euclidean,
                Routing::Auto,
                Some(parents),
            );
            assert!(!r.has_tree());
            assert_eq!(r.route(&centroids, &centroids[..dim], 5)[0], 0);
        }
        let r = Router::new(
            &centroids,
            dim,
            Similarity::Euclidean,
            Routing::Auto,
            Some(&good),
        );
        assert!(r.has_tree());
    }
}
