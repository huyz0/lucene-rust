//! A vector field's HNSW graph kept inside an indexing buffer, so realtime
//! searches walk a graph over the buffered vectors instead of scoring every
//! one of them, and the flush writes that graph instead of building it.
//!
//! The buffer's indexing thread appends each vector ([`BufferGraph::push`]);
//! one inserter adds the appended ordinals to the graph in order
//! ([`BufferGraph::insert_pending`]), so the graph is exactly the one
//! `HnswGraphBuilder` builds over the same vectors with the same seed and
//! parameters, whatever the timing ([`BufferGraph::graph`]). Searches read
//! it while it grows ([`BufferGraph::search`]): the ordinals the inserter
//! finished are searched through the graph, the ones it has not reached yet
//! are scored exactly. Nothing a search reads takes the buffer's lock.
//!
//! The vectors are shared with the buffer's documents (`Arc`s), not copied.

use std::sync::atomic::{fence, AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::field_infos::{VectorEncoding, VectorSimilarityFunction};
use crate::hnsw::{
    HnswGraphSearcher, KnnCollector, NeighborArray, OnHeapHnswGraph, UpdateableVectorScorer,
    VectorScorer,
};
use crate::hnsw_concurrent::{ConcurrentHnswGraph, GraphInserter};
use crate::vectors::{cosine_with_norm, normalize_to_unit_interval, query_norm, Error, Result};

/// Slots per chunk of an [`AppendOnly`].
const CHUNK: usize = 4096;

/// The most vectors a buffer's graph holds: a growing graph's capacity.
const MAX_VECTORS: usize = 1 << 24;

/// Graph ordinals below which, times the dimension, a search scores every
/// vector instead: under about 4k 100-d (2k 300-d) vectors an exact scan
/// is as fast as a graph walk and finds every neighbour.
pub(crate) const EXACT_COMPONENTS: usize = 400_000;

/// One chunk of an [`AppendOnly`]: its slots, each set once.
type Chunk<T> = Box<[OnceLock<T>]>;

/// A table one thread appends to while others read what it appended:
/// slots in chunks allocated on first use and never moved.
pub(crate) struct AppendOnly<T> {
    chunks: Box<[OnceLock<Chunk<T>>]>,
    len: AtomicUsize,
}

impl<T> AppendOnly<T> {
    pub(crate) fn new() -> Self {
        AppendOnly {
            chunks: (0..MAX_VECTORS.div_ceil(CHUNK))
                .map(|_| OnceLock::new())
                .collect(),
            len: AtomicUsize::new(0),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    // ARITH: `i / CHUNK` and `i % CHUNK` of a slot index.
    #[allow(clippy::arithmetic_side_effects)]
    pub(crate) fn get(&self, i: usize) -> Option<&T> {
        if i >= self.len() {
            return None;
        }
        self.chunks.get(i / CHUNK)?.get()?.get(i % CHUNK)?.get()
    }

    /// Appends `value` (one thread appends). The slot is set before the
    /// length shows it.
    // ARITH: as `get`; `i` is below `MAX_VECTORS`.
    #[allow(clippy::arithmetic_side_effects)]
    pub(crate) fn push(&self, value: T) -> Result<()> {
        let i = self.len.load(Ordering::Acquire);
        let full = || Error::InvalidGraphParameter(format!("more than {MAX_VECTORS} vectors"));
        let chunk = self
            .chunks
            .get(i / CHUNK)
            .ok_or_else(full)?
            .get_or_init(|| (0..CHUNK).map(|_| OnceLock::new()).collect());
        chunk
            .get(i % CHUNK)
            .ok_or_else(full)?
            .set(value)
            .map_err(|_| Error::InvalidGraphParameter("two threads appended a vector".into()))?;
        self.len.store(i + 1, Ordering::Release);
        Ok(())
    }
}

/// A buffered vector, shared with the document that carries it.
#[derive(Debug, Clone, PartialEq)]
pub enum BufferVector {
    Float32(Arc<Vec<f32>>),
    Byte(Arc<Vec<u8>>),
}

/// The vectors by ordinal, and what scores them.
pub(crate) struct Column {
    pub(crate) encoding: VectorEncoding,
    pub(crate) similarity: VectorSimilarityFunction,
    pub(crate) dimension: usize,
    pub(crate) vectors: AppendOnly<BufferVector>,
}

impl Column {
    pub(crate) fn floats(&self, ord: i32) -> Result<&[f32]> {
        match usize::try_from(ord).ok().and_then(|o| self.vectors.get(o)) {
            Some(BufferVector::Float32(v)) => Ok(v),
            _ => Err(Error::InvalidGraphParameter(format!(
                "no float vector at ordinal {ord}"
            ))),
        }
    }

    fn bytes(&self, ord: i32) -> Result<&[u8]> {
        match usize::try_from(ord).ok().and_then(|o| self.vectors.get(o)) {
            Some(BufferVector::Byte(v)) => Ok(v),
            _ => Err(Error::InvalidGraphParameter(format!(
                "no byte vector at ordinal {ord}"
            ))),
        }
    }

    fn prefetch(&self, nodes: &[i32]) {
        for &n in nodes {
            let Some(v) = usize::try_from(n).ok().and_then(|o| self.vectors.get(o)) else {
                continue;
            };
            match v {
                BufferVector::Float32(v) => prefetch_all(v),
                BufferVector::Byte(v) => lucene_util::simd::prefetch_bytes(v),
            }
        }
    }
}

/// Every cache line of `v`.
fn prefetch_all(v: &[f32]) {
    let mut i = 0usize;
    while i < v.len() {
        lucene_util::simd::prefetch(v, i);
        i = i.saturating_add(16);
    }
}

/// A query to score buffered vectors against.
#[derive(Debug, Clone)]
pub enum BufferQuery {
    Float32(Vec<f32>),
    Byte(Vec<u8>),
}

/// The flat format's scorers (`DefaultFlatVectorScorer`) over a
/// [`Column`]: the same scores, bit for bit, as the flushed `.vec` gives.
pub(crate) struct ColumnScorer {
    column: Arc<Column>,
    query: BufferQuery,
    /// `query . query`, for a cosine float scorer.
    query_norm: f32,
    max_ord: i32,
}

impl ColumnScorer {
    pub(crate) fn new(column: Arc<Column>, query: BufferQuery, max_ord: i32) -> Result<Self> {
        let len = match &query {
            BufferQuery::Float32(q) if column.encoding == VectorEncoding::Float32 => q.len(),
            BufferQuery::Byte(q) if column.encoding == VectorEncoding::Byte => q.len(),
            _ => {
                return Err(Error::InvalidGraphParameter(
                    "a query of another encoding than the field's".into(),
                ))
            }
        };
        if len != column.dimension {
            return Err(Error::QueryDimensionMismatch(
                i32::try_from(column.dimension).unwrap_or(i32::MAX),
                i32::try_from(len).unwrap_or(i32::MAX),
            ));
        }
        let query_norm = match &query {
            BufferQuery::Float32(q) => query_norm(q),
            BufferQuery::Byte(_) => 0.0,
        };
        Ok(ColumnScorer {
            column,
            query,
            query_norm,
            max_ord,
        })
    }
}

impl VectorScorer for ColumnScorer {
    fn score(&mut self, node: i32) -> Result<f32> {
        let similarity = self.column.similarity;
        Ok(match &self.query {
            BufferQuery::Float32(q) => {
                let candidate = self.column.floats(node)?;
                if similarity == VectorSimilarityFunction::Cosine {
                    normalize_to_unit_interval(cosine_with_norm(q, self.query_norm, candidate))
                } else {
                    similarity.score(q, candidate)
                }
            }
            BufferQuery::Byte(q) => similarity.score_bytes(q, self.column.bytes(node)?),
        })
    }

    fn max_ord(&self) -> i32 {
        self.max_ord
    }

    fn prefetch(&self, nodes: &[i32]) {
        self.column.prefetch(nodes);
    }

    fn bulk_score(&mut self, nodes: &[i32], scores: &mut [f32]) -> Result<f32> {
        if nodes.len() > 1 {
            self.prefetch(nodes);
        }
        let mut max = f32::NEG_INFINITY;
        for (&node, out) in nodes.iter().zip(scores.iter_mut()) {
            *out = self.score(node)?;
            max = max.max(*out);
        }
        Ok(max)
    }
}

impl UpdateableVectorScorer for ColumnScorer {
    fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()> {
        match &mut self.query {
            BufferQuery::Float32(q) => {
                q.clear();
                q.extend_from_slice(self.column.floats(ord)?);
                self.query_norm = query_norm(q);
            }
            BufferQuery::Byte(q) => {
                q.clear();
                q.extend_from_slice(self.column.bytes(ord)?);
            }
        }
        Ok(())
    }
}

/// One vector field's graph over an indexing buffer: see the module docs.
pub struct BufferGraph {
    column: Arc<Column>,
    /// Ordinal to the buffer's document.
    docs: AppendOnly<i32>,
    graph: Arc<ConcurrentHnswGraph>,
    inserter: Mutex<GraphInserter<Arc<ConcurrentHnswGraph>, ColumnScorer>>,
    m: i32,
    beam_width: i32,
    /// Whether an [`Self::insert_pending`] run is due or running: the
    /// buffer schedules one when it sets this.
    scheduled: AtomicBool,
    /// An insertion failed or panicked ([`Self::fail`]).
    failed: AtomicBool,
}

impl std::fmt::Debug for BufferGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferGraph")
            .field("vectors", &self.len())
            .field("inserted", &self.inserted())
            .field("m", &self.m)
            .field("beam_width", &self.beam_width)
            .finish_non_exhaustive()
    }
}

/// One search hit: the buffer's document and its score.
pub type BufferHit = (i32, f32);

/// An index a vector field keeps over an indexing buffer's vectors, which
/// realtime searches read while it grows: [`BufferGraph`] for an HNSW
/// format, [`crate::buffer_lists::BufferLists`] for an IVF one.
pub trait BufferSearch: Send + Sync + std::fmt::Debug {
    /// The buffer's document of ordinal `ord`.
    fn doc(&self, ord: i32) -> Option<i32>;

    /// The best `k` of the first `count` vectors for `query`, among the
    /// ordinals `accept` holds when given: the buffer's documents and
    /// their scores (the flat format's), best first, the lower document on
    /// a tie. `nprobe` is the clusters an IVF segment of the field probes
    /// (`0` for its default); an index without clusters ignores it.
    ///
    /// # Errors
    /// A query of another encoding or dimension than the field's.
    fn search(
        &self,
        query: &BufferQuery,
        k: usize,
        count: i32,
        accept: Option<&FixedBitSet>,
        nprobe: usize,
    ) -> Result<Vec<BufferHit>>;
}

impl BufferSearch for BufferGraph {
    fn doc(&self, ord: i32) -> Option<i32> {
        BufferGraph::doc(self, ord)
    }

    fn search(
        &self,
        query: &BufferQuery,
        k: usize,
        count: i32,
        accept: Option<&FixedBitSet>,
        _nprobe: usize,
    ) -> Result<Vec<BufferHit>> {
        BufferGraph::search(self, query, k, count, accept)
    }
}

impl BufferGraph {
    /// An empty graph for a field of `encoding`, `similarity` and
    /// `dimension`, built with `m` and `beam_width` from `seed` (the flush's
    /// `HnswGraphBuilder` parameters).
    ///
    /// # Errors
    /// Parameters out of range.
    pub fn new(
        encoding: VectorEncoding,
        similarity: VectorSimilarityFunction,
        dimension: usize,
        m: i32,
        beam_width: i32,
        seed: u64,
    ) -> Result<Self> {
        let column = Arc::new(Column {
            encoding,
            similarity,
            dimension,
            vectors: AppendOnly::new(),
        });
        let empty = match encoding {
            VectorEncoding::Float32 => BufferQuery::Float32(vec![0.0; dimension]),
            VectorEncoding::Byte => BufferQuery::Byte(vec![0; dimension]),
        };
        let scorer = ColumnScorer::new(Arc::clone(&column), empty, i32::MAX)?;
        let graph = Arc::new(ConcurrentHnswGraph::growable(m));
        let inserter = GraphInserter::new(Arc::clone(&graph), scorer, m, beam_width, seed)?;
        Ok(BufferGraph {
            column,
            docs: AppendOnly::new(),
            graph,
            inserter: Mutex::new(inserter),
            m,
            beam_width,
            scheduled: AtomicBool::new(false),
            failed: AtomicBool::new(false),
        })
    }

    /// The `(m, beam_width)` it is built with.
    pub fn params(&self) -> (i32, i32) {
        (self.m, self.beam_width)
    }

    /// How many vectors were appended.
    pub fn len(&self) -> i32 {
        i32::try_from(self.column.vectors.len()).unwrap_or(i32::MAX)
    }

    /// Whether none was.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// About the heap it holds beyond the vectors, which the documents
    /// share: per vector its slot, document, and level-0 neighbour array
    /// (the upper levels add about `1 / m` of that).
    // ARITH: a few small sizes, and a count below `MAX_VECTORS` times them.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn ram_bytes(&self) -> usize {
        let m = usize::try_from(self.m).unwrap_or(0);
        // Per node: its slot, its one-level array and the level-0 lists.
        let per_vector = 2 * std::mem::size_of::<OnceLock<BufferVector>>()
            + std::mem::size_of::<std::sync::RwLock<Vec<NeighborArray>>>()
            + std::mem::size_of::<NeighborArray>()
            + (2 * m + 1) * 8;
        usize::try_from(self.len()).unwrap_or(0) * per_vector
    }

    /// How many of them are in the graph.
    pub fn inserted(&self) -> i32 {
        self.graph.published().len()
    }

    /// Appends document `doc`'s vector, the next ordinal (one thread
    /// appends, in document order). Call [`Self::insert_pending`] when
    /// [`Self::schedule`] says so.
    ///
    /// # Errors
    /// A vector of another encoding or dimension than the field's.
    pub fn push(&self, doc: i32, vector: BufferVector) -> Result<()> {
        let (encoding, len) = match &vector {
            BufferVector::Float32(v) => (VectorEncoding::Float32, v.len()),
            BufferVector::Byte(v) => (VectorEncoding::Byte, v.len()),
        };
        if encoding != self.column.encoding || len != self.column.dimension {
            return Err(Error::InvalidGraphParameter(format!(
                "a {encoding:?} vector of {len} dimensions for a {:?} field of {}",
                self.column.encoding, self.column.dimension
            )));
        }
        // The document first: a reader that sees the vector finds it.
        self.docs.push(doc)?;
        self.column.vectors.push(vector)
    }

    /// Whether the caller is to run [`Self::insert_pending`] now: true once
    /// until that run ends.
    pub fn schedule(&self) -> bool {
        // Orders the pusher's length store before its read of the flag, as
        // the fence in `insert_pending` orders the flag's clearing before
        // its read of the length: one of the two sees the other.
        fence(Ordering::SeqCst);
        !self.failed.load(Ordering::Acquire)
            && self.inserted() < self.len()
            && !self.scheduled.swap(true, Ordering::AcqRel)
    }

    /// Stops inserting: a failed or panicked insertion left the graph
    /// unlike the flush builder's. Searches still read what was published
    /// before it, and the flush builds the field's graph again.
    pub fn fail(&self) {
        self.failed.store(true, Ordering::Release);
    }

    /// Stops inserting once the buffer's flush is done with the graph: an
    /// inserter still running returns after its batch, and a later
    /// [`Self::graph`] fails.
    pub fn retire(&self) {
        self.fail();
    }

    /// Inserts the appended ordinals the graph has not reached, `batch` at
    /// a time so a flush waiting on it waits for one batch at most. Returns
    /// once none is left, or after the batch when another run took over.
    ///
    /// # Errors
    /// What inserting a node fails with.
    pub fn insert_pending(&self, batch: usize) -> Result<()> {
        loop {
            self.insert_batch(batch)?;
            self.scheduled.store(false, Ordering::Release);
            fence(Ordering::SeqCst);
            // A vector appended after the last batch but before the flag
            // cleared would wait for the next append: take it now.
            if !self.schedule() {
                return Ok(());
            }
        }
    }

    /// The inserter, unless an insertion failed or panicked.
    fn inserter(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, GraphInserter<Arc<ConcurrentHnswGraph>, ColumnScorer>>>
    {
        let failed = || Error::InvalidGraphParameter("the buffer's graph failed".into());
        let guard = self.inserter.lock().map_err(|_| {
            self.fail();
            failed()
        })?;
        if self.failed.load(Ordering::Acquire) {
            return Err(failed());
        }
        Ok(guard)
    }

    /// Inserts up to `batch` pending ordinals; the number left.
    fn insert_batch(&self, batch: usize) -> Result<i32> {
        let mut inserter = self.inserter()?;
        let mut next = inserter.graph().published().len();
        let end = self.len();
        let stop = next.saturating_add(i32::try_from(batch).unwrap_or(i32::MAX));
        while next < end.min(stop) {
            if let Err(e) = inserter.insert(next) {
                self.fail();
                return Err(e);
            }
            next = next.saturating_add(1);
        }
        Ok(end.saturating_sub(next))
    }

    /// The graph over all `n` vectors appended, inserting what is still
    /// pending first: what `HnswGraphBuilder` builds over them.
    ///
    /// # Errors
    /// Not `n` vectors appended, or an insertion failing (now or before).
    pub fn graph(&self, n: i32) -> Result<OnHeapHnswGraph> {
        if n != self.len() {
            return Err(Error::InvalidGraphParameter(format!(
                "a graph of {n} vectors from a buffer of {}",
                self.len()
            )));
        }
        // Held across the copy: no insertion changes a list mid-copy.
        let mut inserter = self.inserter()?;
        let mut next = inserter.graph().published().len();
        while next < n {
            if let Err(e) = inserter.insert(next) {
                self.fail();
                return Err(e);
            }
            next = next.saturating_add(1);
        }
        if next != n {
            return Err(Error::InvalidGraphParameter(format!(
                "{next} vectors inserted, {n} wanted"
            )));
        }
        self.graph.to_graph(n)
    }

    /// The field's encoding, similarity and dimension.
    pub fn shape(&self) -> (VectorEncoding, VectorSimilarityFunction, usize) {
        (
            self.column.encoding,
            self.column.similarity,
            self.column.dimension,
        )
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
    /// Ordinals the inserter has finished are searched through the graph,
    /// the rest exactly. A small graph, or a filter that leaves fewer
    /// ordinals than the graph walk would visit, is scored exactly too.
    /// A reader that saw fewer vectors than the inserter has since added
    /// walks the graph without the later nodes: lists those pruned in favour
    /// of later nodes are sparser than the builder's over the prefix, which
    /// costs a little recall for a view searched long after it opened.
    ///
    /// # Errors
    /// A query of another encoding or dimension than the field's.
    pub fn search(
        &self,
        query: &BufferQuery,
        k: usize,
        count: i32,
        accept: Option<&FixedBitSet>,
    ) -> Result<Vec<BufferHit>> {
        let count = count.clamp(0, self.len());
        let published = self.graph.published_upto(count);
        let in_graph = published.len();
        let mut scorer = ColumnScorer::new(Arc::clone(&self.column), query.clone(), in_graph)?;
        let accepts = |ord: i32| {
            accept.is_none_or(|a| usize::try_from(ord).is_ok_and(|o| o < a.len() && a.get(o)))
        };
        let mut best = KnnCollector::new(k, u64::MAX);
        let graph_ords = usize::try_from(in_graph).unwrap_or(0);
        let accepted_in_graph = accept.map_or(graph_ords, |a| count_set(a, graph_ords));
        let exact = graph_ords.saturating_mul(self.column.dimension) < EXACT_COMPONENTS
            || accepted_in_graph <= k;
        let mut exact_from = 0;
        if !exact {
            // `HnswGraphSearcher` with `AbstractKnnVectorQuery`'s fallback: a
            // walk that would visit more ordinals than the filter accepts
            // (`cost + 1`), or that finds fewer than `k`, is redone exactly.
            // Unlike Lucene's leaf search it does not switch to the filtered
            // searcher for a selective filter: a buffer is one leaf, small
            // beside the committed segments.
            let limit = (accepted_in_graph as u64).saturating_add(1);
            let mut collector = KnnCollector::new(k, limit);
            HnswGraphSearcher::new(k, in_graph).search(
                &mut collector,
                &mut scorer,
                &published,
                accept,
            )?;
            if !collector.early_terminated() && collector.size() >= k.min(accepted_in_graph) {
                for (ord, score) in collector.top_docs() {
                    best.collect(ord, score);
                }
                exact_from = in_graph;
            }
        }
        // The ordinals the graph does not cover (all of them for an exact
        // search), scored one by one.
        let mut ord = exact_from;
        while ord < count {
            if accepts(ord) {
                let score = scorer.score(ord)?;
                best.collect(ord, score);
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

/// The set bits of `bits` below `n`.
pub(crate) fn count_set(bits: &FixedBitSet, n: usize) -> usize {
    if n >= bits.len() {
        return bits.cardinality();
    }
    let mut count = 0usize;
    let mut at = 0usize;
    while let Some(i) = bits.next_set_bit(at) {
        if i >= n {
            break;
        }
        count = count.saturating_add(1);
        at = i.saturating_add(1);
    }
    count
}

#[cfg(test)]
mod tests {
    // ARITH: test fixtures of a few thousand vectors.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use crate::hnsw::{HnswGraphBuilder, HnswGraphView, DEFAULT_RAND_SEED};

    /// Deterministic pseudo-random unit vectors.
    fn vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                    })
                    .collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= norm);
                v
            })
            .collect()
    }

    fn filled(vs: &[Vec<f32>], similarity: VectorSimilarityFunction) -> BufferGraph {
        let g = BufferGraph::new(
            VectorEncoding::Float32,
            similarity,
            vs[0].len(),
            16,
            100,
            DEFAULT_RAND_SEED,
        )
        .unwrap();
        for (i, v) in vs.iter().enumerate() {
            // Every other document carries the field.
            g.push(2 * i as i32, BufferVector::Float32(Arc::new(v.clone())))
                .unwrap();
        }
        g
    }

    fn edges(g: &OnHeapHnswGraph) -> Vec<Vec<Vec<i32>>> {
        g.graph
            .iter()
            .map(|levels| levels.iter().map(|a| a.nodes().to_vec()).collect())
            .collect()
    }

    fn exact(vs: &[Vec<f32>], q: &[f32], k: usize, keep: impl Fn(usize) -> bool) -> Vec<i32> {
        let mut scored: Vec<(i32, f32)> = vs
            .iter()
            .enumerate()
            .filter(|(i, _)| keep(*i))
            .map(|(i, v)| {
                (
                    2 * i as i32,
                    VectorSimilarityFunction::DotProduct.score(q, v),
                )
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored.into_iter().take(k).map(|(d, _)| d).collect()
    }

    /// The graph a buffer ends with is the flush builder's, for both
    /// similarities that score differently, whenever the inserts ran.
    #[test]
    fn the_graph_is_the_flush_builders() {
        let vs = vectors(1500, 24);
        for similarity in [
            VectorSimilarityFunction::DotProduct,
            VectorSimilarityFunction::Cosine,
            VectorSimilarityFunction::Euclidean,
        ] {
            let g = filled(&vs, similarity);
            assert!(g.schedule());
            assert!(!g.schedule(), "scheduled once until the run ends");
            g.insert_batch(700).unwrap();
            // The flush's builder, over the flat file the flush writes.
            let id = *b"buffervectors001";
            let field = crate::vectors::FlatVectorsField {
                field_number: 0,
                similarity,
                dimension: 24,
                docs: (0..1500).map(|i| 2 * i).collect(),
                values: crate::vectors::FieldVectorData::Float32(
                    vs.iter().flatten().copied().collect(),
                ),
            };
            let (vec, vemf) = crate::vectors::write_flat_vectors(&[field], 3000, &id, "").unwrap();
            let flat = crate::vectors::FlatVectorsReader::open(&vemf, &vec, &id, "").unwrap();
            let values = flat.float_vector_values(0).unwrap();
            let built = HnswGraphBuilder::new(values.ord_scorer(), 16, 100, DEFAULT_RAND_SEED)
                .unwrap()
                .build(1500)
                .unwrap();
            let ours = g.graph(1500).unwrap();
            assert_eq!(edges(&ours), edges(&built), "{similarity:?}");
            assert_eq!(ours.entry_node(), built.entry_node());
            assert_eq!(ours.num_levels(), built.num_levels());
            g.insert_pending(64).unwrap();
            assert_eq!(g.inserted(), 1500);
            assert!(!g.schedule());
        }
    }

    /// Searches find what an exact scan finds, over the graph, the
    /// vectors not inserted yet, a reader's shorter count, and a filter.
    #[test]
    fn searches_match_an_exact_scan() {
        let n = 6000;
        let vs = vectors(n, 96);
        let g = filled(&vs, VectorSimilarityFunction::DotProduct);
        g.insert_batch(4500).unwrap();
        let queries = vectors(30, 96);
        let mut found = 0usize;
        let mut wanted = 0usize;
        for (qi, q) in queries.iter().enumerate() {
            let query = BufferQuery::Float32(q.clone());
            for (count, keep) in [(n, None), (5000, None), (n, Some(3usize)), (n, Some(500))] {
                let accept = keep.map(|m| {
                    let mut bits = FixedBitSet::new(n);
                    (0..n)
                        .filter(|i| (i + qi).is_multiple_of(m))
                        .for_each(|i| bits.set(i));
                    bits
                });
                let hits = g.search(&query, 10, count as i32, accept.as_ref()).unwrap();
                let truth = exact(&vs[..count], q, 10, |i| {
                    keep.is_none_or(|m| (i + qi).is_multiple_of(m))
                });
                assert!(hits.windows(2).all(|w| w[0].1 >= w[1].1));
                assert!(hits.iter().all(|&(d, _)| (d as usize) < 2 * count));
                if let Some(m) = keep {
                    assert!(hits
                        .iter()
                        .all(|&(d, _)| (d as usize / 2 + qi).is_multiple_of(m)));
                }
                assert_eq!(hits.len(), truth.len());
                found += hits.iter().filter(|(d, _)| truth.contains(d)).count();
                wanted += truth.len();
            }
        }
        let recall = found as f64 / wanted as f64;
        assert!(recall > 0.95, "recall {recall}");
    }

    /// A small buffer, or no vectors at all, is scored exactly; a query
    /// of the wrong shape is refused.
    #[test]
    fn small_and_bad_searches() {
        let vs = vectors(50, 8);
        let g = filled(&vs, VectorSimilarityFunction::DotProduct);
        let q = BufferQuery::Float32(vs[7].clone());
        let hits = g.search(&q, 3, 50, None).unwrap();
        assert_eq!(
            hits.iter().map(|h| h.0).collect::<Vec<_>>(),
            exact(&vs, &vs[7], 3, |_| true)
        );
        assert!(g.search(&q, 3, 0, None).unwrap().is_empty());
        assert!(g
            .search(&BufferQuery::Float32(vec![0.0; 3]), 3, 50, None)
            .is_err());
        assert!(g
            .search(&BufferQuery::Byte(vec![0; 8]), 3, 50, None)
            .is_err());
        assert!(g
            .push(100, BufferVector::Float32(Arc::new(vec![0.0; 3])))
            .is_err());
        assert!(g
            .push(100, BufferVector::Byte(Arc::new(vec![0; 8])))
            .is_err());
        assert!(g.graph(51).is_err());
        assert_eq!(g.doc(3), Some(6));
        assert_eq!(g.doc(50), None);
        assert_eq!(g.params(), (16, 100));
        assert!(!g.is_empty());
        assert!(format!("{g:?}").contains("vectors: 50"));
    }

    /// Byte vectors score as the flat byte scorer does.
    #[test]
    fn byte_vectors() {
        let g = BufferGraph::new(
            VectorEncoding::Byte,
            VectorSimilarityFunction::Euclidean,
            4,
            8,
            20,
            DEFAULT_RAND_SEED,
        )
        .unwrap();
        for i in 0..100u8 {
            g.push(i32::from(i), BufferVector::Byte(Arc::new(vec![i, i, 0, 1])))
                .unwrap();
        }
        g.insert_pending(10).unwrap();
        let hits = g
            .search(&BufferQuery::Byte(vec![40, 40, 0, 1]), 2, 100, None)
            .unwrap();
        assert_eq!(hits[0], (40, 1.0));
        assert_eq!(hits[1].0, 39);
    }

    /// A byte field's graph is the flush builder's over the flat byte
    /// scorer too.
    #[test]
    fn byte_graph_is_the_flush_builders() {
        let n = 800usize;
        let bytes: Vec<Vec<u8>> = vectors(n, 12)
            .iter()
            .map(|v| v.iter().map(|x| (x * 120.0) as i8 as u8).collect())
            .collect();
        for similarity in [
            VectorSimilarityFunction::Euclidean,
            VectorSimilarityFunction::DotProduct,
        ] {
            let g = BufferGraph::new(
                VectorEncoding::Byte,
                similarity,
                12,
                8,
                40,
                DEFAULT_RAND_SEED,
            )
            .unwrap();
            for (i, v) in bytes.iter().enumerate() {
                g.push(i as i32, BufferVector::Byte(Arc::new(v.clone())))
                    .unwrap();
            }
            let id = *b"buffervectors002";
            let field = crate::vectors::FlatVectorsField {
                field_number: 0,
                similarity,
                dimension: 12,
                docs: (0..n as i32).collect(),
                values: crate::vectors::FieldVectorData::Byte(
                    bytes.iter().flatten().copied().collect(),
                ),
            };
            let (vec, vemf) =
                crate::vectors::write_flat_vectors(&[field], n as i32, &id, "").unwrap();
            let flat = crate::vectors::FlatVectorsReader::open(&vemf, &vec, &id, "").unwrap();
            let values = flat.byte_vector_values(0).unwrap();
            let built = HnswGraphBuilder::new(values.ord_scorer(), 8, 40, DEFAULT_RAND_SEED)
                .unwrap()
                .build(n as i32)
                .unwrap();
            let ours = g.graph(n as i32).unwrap();
            assert_eq!(edges(&ours), edges(&built), "{similarity:?}");
            assert_eq!(g.shape(), (VectorEncoding::Byte, similarity, 12));
        }
    }

    /// A failed graph is never handed to a flush and schedules no more
    /// inserts.
    #[test]
    fn a_failed_graph_is_refused() {
        let vs = vectors(100, 8);
        let g = filled(&vs, VectorSimilarityFunction::DotProduct);
        assert!(g.graph(99).is_err(), "not every vector");
        g.fail();
        assert!(!g.schedule());
        assert!(g.insert_pending(10).is_err());
        assert!(g.graph(100).is_err());
        // What was published stays searchable, the rest scored exactly.
        let hits = g
            .search(&BufferQuery::Float32(vs[3].clone()), 1, 100, None)
            .unwrap();
        assert_eq!(hits[0].0, 6);
    }

    /// Searches read the graph while the inserter grows it.
    #[test]
    fn searches_while_inserting() {
        let n = 5000;
        let vs = vectors(n, 64);
        let g = filled(&vs, VectorSimilarityFunction::DotProduct);
        std::thread::scope(|s| {
            s.spawn(|| g.insert_pending(32).unwrap());
            for q in vectors(50, 64) {
                let hits = g
                    .search(&BufferQuery::Float32(q.clone()), 5, n as i32, None)
                    .unwrap();
                assert_eq!(hits.len(), 5);
            }
        });
        let _ = g.graph(n as i32).unwrap();
    }
}
