//! A vector field's IVF cluster lists kept inside an indexing buffer, so a
//! realtime search of an IVF field scores the buffered vectors of the
//! clusters nearest the query instead of every one of them.
//!
//! The clusters are a flushed segment's ([`ClusterRouter`]): the buffer
//! trains none. The buffer's indexing thread appends each vector
//! ([`BufferLists::push`]); one background task routes the appended
//! ordinals to their nearest centroid, in order ([`BufferLists::assign_pending`]),
//! and a vector far from every centroid goes to an outlier list that every
//! search scores. Searches read the lists while they grow
//! ([`BufferLists::search`]): the ordinals routed so far through the lists,
//! the rest exactly. The flush does not read the lists: it clusters the
//! segment's vectors as it does without them.
//!
//! The vectors are shared with the buffer's documents (`Arc`s), not copied.

use std::sync::atomic::{fence, AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::buffer_vectors::{
    count_set, AppendOnly, BufferHit, BufferQuery, BufferSearch, BufferVector, Column,
    ColumnScorer, EXACT_COMPONENTS,
};
use crate::field_infos::{VectorEncoding, VectorSimilarityFunction};
use crate::hnsw::{KnnCollector, VectorScorer};
use crate::vectors::{Error, Result};

/// Routes vectors to an IVF field's clusters.
pub trait ClusterRouter: Send + Sync {
    /// How many clusters there are.
    fn clusters(&self) -> usize;

    /// The cluster vector `v` joins, or `None` when it is too far from
    /// every centroid for probing to find it.
    fn assign(&self, v: &[f32]) -> Option<usize>;

    /// The `nprobe` clusters nearest to `query`, nearest first.
    fn probe(&self, query: &[f32], nprobe: usize) -> Vec<usize>;

    /// The clusters a segment of the field probes for a query that names
    /// no count.
    fn default_nprobe(&self) -> usize;
}

/// Vectors of a list scored per call, their memory fetched first.
const SCORE_BATCH: usize = 16;

/// One vector field's cluster lists over an indexing buffer: see the
/// module docs.
pub struct BufferLists {
    column: Arc<Column>,
    /// Ordinal to the buffer's document.
    docs: AppendOnly<i32>,
    router: Arc<dyn ClusterRouter>,
    /// Per cluster its ordinals, ascending; then the outliers.
    lists: Box<[RwLock<Vec<i32>>]>,
    /// The ordinals below it are in the lists.
    assigned: AtomicUsize,
    /// Held by the one task that routes.
    assigner: Mutex<()>,
    /// Whether an [`Self::assign_pending`] run is due or running: the
    /// buffer schedules one when it sets this.
    scheduled: AtomicBool,
    /// Routing stopped ([`Self::retire`], or a panic while routing).
    retired: AtomicBool,
}

impl std::fmt::Debug for BufferLists {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferLists")
            .field("vectors", &self.len())
            .field("assigned", &self.assigned())
            .field("clusters", &self.router.clusters())
            .field("outliers", &self.outliers())
            .finish_non_exhaustive()
    }
}

impl BufferLists {
    /// Empty lists for a float field of `similarity` and `dimension`, over
    /// `router`'s clusters.
    pub fn new(
        similarity: VectorSimilarityFunction,
        dimension: usize,
        router: Arc<dyn ClusterRouter>,
    ) -> Self {
        let lists = (0..router.clusters().saturating_add(1))
            .map(|_| RwLock::new(Vec::new()))
            .collect();
        BufferLists {
            column: Arc::new(Column {
                encoding: VectorEncoding::Float32,
                similarity,
                dimension,
                vectors: AppendOnly::new(),
            }),
            docs: AppendOnly::new(),
            router,
            lists,
            assigned: AtomicUsize::new(0),
            assigner: Mutex::new(()),
            scheduled: AtomicBool::new(false),
            retired: AtomicBool::new(false),
        }
    }

    /// How many vectors were appended.
    pub fn len(&self) -> i32 {
        i32::try_from(self.column.vectors.len()).unwrap_or(i32::MAX)
    }

    /// Whether none was.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How many of them are in the lists.
    pub fn assigned(&self) -> i32 {
        i32::try_from(self.assigned.load(Ordering::Acquire)).unwrap_or(i32::MAX)
    }

    /// How many routed vectors were too far from every centroid.
    pub fn outliers(&self) -> usize {
        self.lists
            .last()
            .map_or(0, |l| l.read().unwrap_or_else(|e| e.into_inner()).len())
    }

    /// About the heap it holds beyond the vectors, which the documents
    /// share: per vector its slot, document and list entry.
    // ARITH: small sizes times a count below the table's capacity.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn ram_bytes(&self) -> usize {
        let per_vector = std::mem::size_of::<std::sync::OnceLock<BufferVector>>()
            + std::mem::size_of::<std::sync::OnceLock<i32>>()
            + std::mem::size_of::<i32>();
        usize::try_from(self.len()).unwrap_or(0) * per_vector
    }

    /// Appends document `doc`'s vector, the next ordinal (one thread
    /// appends, in document order). Call [`Self::assign_pending`] when
    /// [`Self::schedule`] says so.
    ///
    /// # Errors
    /// A byte vector, or one of another dimension than the field's.
    pub fn push(&self, doc: i32, vector: BufferVector) -> Result<()> {
        match &vector {
            BufferVector::Float32(v) if v.len() == self.column.dimension => {}
            _ => {
                return Err(Error::InvalidGraphParameter(format!(
                    "not a float vector of {} dimensions",
                    self.column.dimension
                )))
            }
        }
        // The document first: a reader that sees the vector finds it.
        self.docs.push(doc)?;
        self.column.vectors.push(vector)
    }

    /// Whether the caller is to run [`Self::assign_pending`] now: true once
    /// until that run ends.
    pub fn schedule(&self) -> bool {
        // As `BufferGraph::schedule`: the pusher's length store and the
        // run's flag clearing are each fenced before the other's read.
        fence(Ordering::SeqCst);
        !self.retired.load(Ordering::Acquire)
            && self.assigned() < self.len()
            && !self.scheduled.swap(true, Ordering::AcqRel)
    }

    /// Stops routing: a run still going returns after its batch, and the
    /// ordinals it did not reach are scored exactly.
    pub fn retire(&self) {
        self.retired.store(true, Ordering::Release);
    }

    /// Routes the appended ordinals the lists have not reached, `batch` at
    /// a time. Returns once none is left, after the batch when another run
    /// took over, or once retired.
    pub fn assign_pending(&self, batch: usize) {
        loop {
            self.assign_batch(batch);
            self.scheduled.store(false, Ordering::Release);
            fence(Ordering::SeqCst);
            // A vector appended after the last batch but before the flag
            // cleared would wait for the next append: take it now.
            if !self.schedule() {
                return;
            }
        }
    }

    /// Routes up to `batch` pending ordinals.
    fn assign_batch(&self, batch: usize) {
        let Ok(_guard) = self.assigner.lock() else {
            // A run panicked mid-batch: the lists stop where it left them.
            self.retire();
            return;
        };
        let end = self.column.vectors.len();
        let mut next = self.assigned.load(Ordering::Acquire);
        let stop = next.saturating_add(batch).min(end);
        while next < stop && !self.retired.load(Ordering::Acquire) {
            // Neither can happen to a pushed vector; a run that cannot go
            // on stops routing rather than be scheduled again.
            let Ok(ord) = i32::try_from(next) else {
                self.retire();
                return;
            };
            let Ok(v) = self.column.floats(ord) else {
                self.retire();
                return;
            };
            let list = self
                .router
                .assign(v)
                .filter(|&c| c < self.router.clusters())
                .unwrap_or(self.router.clusters());
            if let Some(list) = self.lists.get(list) {
                list.write().unwrap_or_else(|e| e.into_inner()).push(ord);
            }
            next = next.saturating_add(1);
            // The entry is in its list before the count shows it.
            self.assigned.store(next, Ordering::Release);
        }
    }

    /// The buffer's document of ordinal `ord`.
    pub fn doc(&self, ord: i32) -> Option<i32> {
        usize::try_from(ord)
            .ok()
            .and_then(|o| self.docs.get(o))
            .copied()
    }

    /// The best `k` of the first `count` vectors for `query` (the vectors a
    /// reader that saw `count` of them reads), among the ordinals `accept`
    /// holds when given: the buffer's documents and their scores, best
    /// first, the lower document on a tie.
    ///
    /// The routed ordinals are scored in the `nprobe` clusters nearest the
    /// query (the router's default for `0`), as a segment probes; then
    /// twice as many again while fewer than `k` accepted vectors were
    /// found. Scored exactly, the probed lists find more of the neighbours
    /// than a segment's quantized search of as many clusters. The
    /// outliers and the ordinals not routed yet are scored exactly, and so
    /// is every vector of small lists, or under a filter that leaves `k` or
    /// fewer.
    ///
    /// # Errors
    /// A query that is not a float vector of the field's dimension.
    pub fn search(
        &self,
        query: &BufferQuery,
        k: usize,
        count: i32,
        accept: Option<&FixedBitSet>,
        nprobe: usize,
    ) -> Result<Vec<BufferHit>> {
        let count = count.clamp(0, self.len());
        let mut scorer = ColumnScorer::new(Arc::clone(&self.column), query.clone(), count)?;
        let BufferQuery::Float32(q) = query else {
            return Err(Error::InvalidGraphParameter(
                "a byte query of a float field".into(),
            ));
        };
        let accepts = |ord: i32| {
            accept.is_none_or(|a| usize::try_from(ord).is_ok_and(|o| o < a.len() && a.get(o)))
        };
        let mut best = KnnCollector::new(k, u64::MAX);
        let routed = self.assigned().min(count);
        let routed_ords = usize::try_from(routed).unwrap_or(0);
        let accepted = accept.map_or(routed_ords, |a| count_set(a, routed_ords));
        let nlist = self.router.clusters();
        let exact = nlist == 0
            || routed_ords.saturating_mul(self.column.dimension) < EXACT_COMPONENTS
            || accepted <= k;
        let mut exact_from = 0;
        if !exact {
            let mut score_list = |list: usize, found: &mut usize| -> Result<()> {
                let Some(list) = self.lists.get(list) else {
                    return Ok(());
                };
                // Copied out so the router's push into this list does not
                // wait for the scoring. Ascending: the ordinals past the
                // reader's count end it.
                let ords: Vec<i32> = {
                    let all = list.read().unwrap_or_else(|e| e.into_inner());
                    let end = all.partition_point(|&o| o < routed);
                    all.get(..end).unwrap_or_default().to_vec()
                };
                let end = ords.len();
                let mut batch = [0i32; SCORE_BATCH];
                let mut scores = [0f32; SCORE_BATCH];
                for chunk in ords.get(..end).unwrap_or_default().chunks(SCORE_BATCH) {
                    let mut n = 0usize;
                    for &ord in chunk {
                        if accepts(ord) {
                            if let Some(slot) = batch.get_mut(n) {
                                *slot = ord;
                                n = n.saturating_add(1);
                            }
                        }
                    }
                    let (nodes, out) = (&batch[..n], &mut scores[..n]);
                    // Each vector is its own allocation: fetched together.
                    scorer.bulk_score(nodes, out)?;
                    for (&ord, &score) in nodes.iter().zip(out.iter()) {
                        best.collect(ord, score);
                    }
                    *found = found.saturating_add(n);
                }
                Ok(())
            };
            let base = if nprobe == 0 {
                self.router.default_nprobe()
            } else {
                nprobe
            };
            let mut probes = base.clamp(1, nlist);
            let mut probed = vec![false; nlist];
            let mut found = 0usize;
            loop {
                for c in self.router.probe(q, probes) {
                    if probed.get(c).copied().unwrap_or(true) {
                        continue;
                    }
                    probed[c] = true;
                    score_list(c, &mut found)?;
                }
                if found >= k || probes >= nlist {
                    break;
                }
                probes = probes.saturating_mul(2).min(nlist);
            }
            score_list(nlist, &mut found)?;
            exact_from = routed;
        }
        // The ordinals the lists do not hold yet (all of them for an exact
        // search), scored one by one.
        let mut ord = exact_from;
        while ord < count {
            if accepts(ord) {
                best.collect(ord, scorer.score(ord)?);
            }
            ord = ord.saturating_add(1);
        }
        let mut hits: Vec<BufferHit> = best
            .top_docs()
            .into_iter()
            .filter_map(|(ord, score)| Some((self.doc(ord)?, score)))
            .collect();
        hits.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(hits)
    }
}

impl BufferSearch for BufferLists {
    fn doc(&self, ord: i32) -> Option<i32> {
        BufferLists::doc(self, ord)
    }

    fn search(
        &self,
        query: &BufferQuery,
        k: usize,
        count: i32,
        accept: Option<&FixedBitSet>,
        nprobe: usize,
    ) -> Result<Vec<BufferHit>> {
        BufferLists::search(self, query, k, count, accept, nprobe)
    }
}

#[cfg(test)]
mod tests {
    // ARITH: test fixtures of a few thousand vectors.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    /// Deterministic pseudo-random unit vectors, around `centers` clusters.
    fn vectors(n: usize, dim: usize, centers: usize) -> Vec<Vec<f32>> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let means: Vec<Vec<f32>> = (0..centers)
            .map(|_| (0..dim).map(|_| next() * 4.0).collect())
            .collect();
        (0..n)
            .map(|i| {
                let m = &means[i % centers];
                let mut v: Vec<f32> = m.iter().map(|x| x + next()).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= norm);
                v
            })
            .collect()
    }

    /// Brute-force routing over fixed centroids; vectors whose best dot
    /// product is below `outlier` are outliers.
    struct Brute {
        centroids: Vec<Vec<f32>>,
        outlier: f32,
        nprobe: usize,
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    impl Brute {
        fn ranked(&self, q: &[f32]) -> Vec<(usize, f32)> {
            let mut r: Vec<(usize, f32)> = self
                .centroids
                .iter()
                .enumerate()
                .map(|(c, m)| (c, dot(q, m)))
                .collect();
            r.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            r
        }
    }

    impl ClusterRouter for Brute {
        fn clusters(&self) -> usize {
            self.centroids.len()
        }
        fn assign(&self, v: &[f32]) -> Option<usize> {
            let (c, s) = self.ranked(v)[0];
            (s >= self.outlier).then_some(c)
        }
        fn probe(&self, q: &[f32], nprobe: usize) -> Vec<usize> {
            self.ranked(q)
                .into_iter()
                .take(nprobe)
                .map(|(c, _)| c)
                .collect()
        }
        fn default_nprobe(&self) -> usize {
            self.nprobe
        }
    }

    fn unit_means(dim: usize, centers: usize) -> Vec<Vec<f32>> {
        // The data's own means: every fourth vector of a fresh draw is near
        // its cluster's mean, so averaging recovers them.
        let vs = vectors(centers * 50, dim, centers);
        (0..centers)
            .map(|c| {
                let mut m = vec![0.0f32; dim];
                for v in vs.iter().skip(c).step_by(centers) {
                    m.iter_mut().zip(v).for_each(|(a, b)| *a += b);
                }
                let norm = m.iter().map(|x| x * x).sum::<f32>().sqrt();
                m.iter_mut().for_each(|x| *x /= norm);
                m
            })
            .collect()
    }

    fn lists(vs: &[Vec<f32>], router: Brute) -> BufferLists {
        let dim = vs[0].len();
        let l = BufferLists::new(VectorSimilarityFunction::DotProduct, dim, Arc::new(router));
        for (i, v) in vs.iter().enumerate() {
            l.push(i as i32 * 2, BufferVector::Float32(Arc::new(v.clone())))
                .unwrap();
        }
        l
    }

    fn exact(vs: &[Vec<f32>], q: &[f32], k: usize, keep: impl Fn(usize) -> bool) -> Vec<i32> {
        let mut all: Vec<(i32, f32)> = vs
            .iter()
            .enumerate()
            .filter(|(i, _)| keep(*i))
            .map(|(i, v)| {
                (
                    i as i32 * 2,
                    VectorSimilarityFunction::DotProduct.score(q, v),
                )
            })
            .collect();
        all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        all.into_iter().take(k).map(|(d, _)| d).collect()
    }

    /// Probing the nearest clusters finds what an exact scan finds, with
    /// the unrouted tail and the outliers scored too.
    #[test]
    fn probed_lists_find_the_exact_neighbours() {
        let (dim, centers) = (32, 40);
        let vs = vectors(20_000, dim, centers);
        let l = lists(
            &vs,
            Brute {
                centroids: unit_means(dim, centers),
                outlier: 0.5,
                nprobe: 4,
            },
        );
        assert!(l.schedule());
        assert!(!l.schedule(), "one run at a time");
        l.assign_pending(1000);
        assert_eq!(l.assigned(), 20_000);
        let outliers = l.lists[centers].read().unwrap().len();
        assert!(outliers < 2000, "{outliers} outliers");
        let mut recall = 0usize;
        for q in vectors(50, dim, centers).iter().skip(3) {
            let hits = l
                .search(&BufferQuery::Float32(q.clone()), 10, 20_000, None, 0)
                .unwrap();
            let want = exact(&vs, q, 10, |_| true);
            recall += hits.iter().filter(|h| want.contains(&h.0)).count();
        }
        assert!(recall as f64 / 470.0 > 0.95, "recall {recall}/470");
    }

    /// A reader that saw fewer vectors than the lists hold reads only its
    /// own; one that saw more scores the unrouted ones exactly.
    #[test]
    fn a_reader_reads_its_own_count() {
        let (dim, centers) = (32, 40);
        let vs = vectors(20_000, dim, centers);
        let l = lists(
            &vs,
            Brute {
                centroids: unit_means(dim, centers),
                outlier: -1.0,
                nprobe: 4,
            },
        );
        l.assign_batch(15_000);
        assert_eq!(l.assigned(), 15_000);
        let q = vs[17_000].clone();
        let hits = l
            .search(&BufferQuery::Float32(q.clone()), 1, 20_000, None, 0)
            .unwrap();
        assert_eq!(hits[0].0, 34_000, "unrouted, scored exactly");
        let hits = l
            .search(&BufferQuery::Float32(q), 5, 16_000, None, 0)
            .unwrap();
        assert!(hits.iter().all(|h| h.0 < 32_000), "{hits:?}");
    }

    /// A selective filter widens the probes until `k` accepted vectors are
    /// found.
    #[test]
    fn a_filter_widens_the_probes() {
        let (dim, centers) = (32, 40);
        let vs = vectors(20_000, dim, centers);
        let l = lists(
            &vs,
            Brute {
                centroids: unit_means(dim, centers),
                outlier: -1.0,
                nprobe: 1,
            },
        );
        l.assign_pending(usize::MAX);
        // Only cluster 7's vectors pass: far from most queries.
        let mut accept = FixedBitSet::new(20_000);
        for i in (7..20_000).step_by(centers).take(100) {
            // FBS: `i < 20_000`, the bitset's length.
            accept.set(i);
        }
        let q = vs[3].clone();
        let hits = l
            .search(
                &BufferQuery::Float32(q.clone()),
                10,
                20_000,
                Some(&accept),
                1,
            )
            .unwrap();
        let want = exact(&vs, &q, 10, |i| accept.get(i));
        assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), want);
    }

    #[test]
    fn small_bad_and_retired() {
        let vs = vectors(100, 8, 4);
        let l = lists(
            &vs,
            Brute {
                centroids: unit_means(8, 4),
                outlier: -1.0,
                nprobe: 1,
            },
        );
        assert!(l
            .push(1000, BufferVector::Float32(Arc::new(vec![0.0; 3])))
            .is_err());
        assert!(l
            .push(1000, BufferVector::Byte(Arc::new(vec![0; 8])))
            .is_err());
        assert!(l
            .search(&BufferQuery::Byte(vec![0; 8]), 1, 100, None, 0)
            .is_err());
        l.retire();
        assert!(!l.schedule());
        l.assign_pending(10);
        assert_eq!(l.assigned(), 0);
        // Small: exact.
        let hits = l
            .search(&BufferQuery::Float32(vs[5].clone()), 1, 100, None, 0)
            .unwrap();
        assert_eq!(hits[0].0, 10);
        assert!(l.ram_bytes() > 0);
        assert!(format!("{l:?}").contains("BufferLists"));
    }
}
