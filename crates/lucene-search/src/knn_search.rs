//! KNN searches over a reader's segments, in the shapes OpenSearch's k-NN
//! plugin runs them.
//!
//! - [`KnnLeafQuery`] searches each segment on its own: its top `k` documents
//!   (the `faiss` engine's `KNNQuery`, which returns `k` hits per segment), or
//!   every document whose score reaches a threshold (a radial search). It is a
//!   [`DocumentQuery`], so it sits in a query tree like any leaf, and its
//!   filter is its [`crate::extended_query::DocumentClause`]'s.
//! - [`knn_top_hits`] is `AbstractKnnVectorQuery.rewrite`: the top `k` of the
//!   whole reader (the `lucene` engine's `KnnFloatVectorQuery`), which a
//!   caller turns into a [`crate::extended_query::DocAndScoreQuery`].
//!
//! A segment's vectors are found through its reader
//! ([`SegmentReader::knn_vectors_for`]): the flat vectors and, when the
//! field has one, its HNSW graph or IVF index. A field with neither is
//! searched exhaustively, as Java does.

use lucene_codecs::buffer_vectors::{BufferGraph, BufferQuery};
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::collector::{ScoreDoc, ScoringCollector};
use crate::directory_reader::SegmentReader;
use crate::document::DocumentQuery;
use crate::join::BitSetProducer;
use crate::multi_segment::OpenSegment;
use crate::query::Clause;
use crate::vector_query::{
    self, ByteVectorSimilarityQuery, FloatVectorSimilarityQuery, GraphReader, KnnByteVectorQuery,
    KnnFloatVectorQuery, KnnSegment, VectorsInput,
};
use crate::{Error, Result};

/// A KNN query vector: `float[]` or Java's signed `byte[]` verbatim.
#[derive(Debug, Clone, PartialEq)]
pub enum KnnTarget {
    Float(Vec<f32>),
    Byte(Vec<u8>),
}

/// What a [`KnnLeafQuery`] keeps of each segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KnnLeafKind {
    /// The segment's `k` best documents, the graph walked with a beam of
    /// `max(k, ef_search)`.
    TopK { k: usize, ef_search: usize },
    /// Every document scoring at least `min_score`
    /// (`FloatVectorSimilarityQuery`'s `resultSimilarity`).
    Radial { min_score: f32 },
}

/// A KNN search run segment by segment: see the module doc.
#[derive(Debug, Clone, PartialEq)]
pub struct KnnLeafQuery {
    pub field: String,
    pub target: KnnTarget,
    pub kind: KnnLeafKind,
}

impl KnnLeafQuery {
    /// The query, checked as Java's constructors check it: `k >= 1`, a
    /// threshold that is a number.
    pub fn new(field: impl Into<String>, target: KnnTarget, kind: KnnLeafKind) -> Result<Self> {
        match kind {
            KnnLeafKind::TopK { k: 0, .. } => {
                return Err(Error::InvalidKnnQuery("k must be at least 1".into()))
            }
            KnnLeafKind::Radial { min_score } if min_score.is_nan() => {
                return Err(Error::InvalidKnnQuery(format!(
                    "resultSimilarity must have a valid value; got {min_score}"
                )))
            }
            _ => {}
        }
        Ok(Self {
            field: field.into(),
            target,
            kind,
        })
    }

    /// This segment's hits, local doc ids, in no particular order.
    fn leaf_hits(
        &self,
        reader: &SegmentReader,
        live_docs: Option<&FixedBitSet>,
        filter: Option<&FixedBitSet>,
    ) -> Result<Vec<ScoreDoc>> {
        let Some(owned) = open_leaf(reader, &self.field)? else {
            return Ok(Vec::new());
        };
        let input = owned.input(reader, live_docs, filter);
        // A beam wider than the segment finds what one of its size does; the
        // bound keeps a request's `k` or `ef_search` from sizing the heaps.
        let kind = match self.kind {
            KnnLeafKind::TopK { k, ef_search } => {
                let docs = usize::try_from(reader.max_doc).unwrap_or(0).max(1);
                KnnLeafKind::TopK {
                    k: k.min(docs),
                    ef_search: ef_search.min(docs),
                }
            }
            radial => radial,
        };
        match (kind, &self.target) {
            (KnnLeafKind::TopK { k, ef_search }, KnnTarget::Float(t)) => {
                let q = KnnFloatVectorQuery::new(self.field.clone(), t.clone(), k)?
                    .with_ef_search(ef_search);
                vector_query::search_knn_float_vector_query(&input, &q)
            }
            (KnnLeafKind::TopK { k, ef_search }, KnnTarget::Byte(t)) => {
                let q = KnnByteVectorQuery::new(self.field.clone(), t.clone(), k)?
                    .with_ef_search(ef_search);
                vector_query::search_knn_byte_vector_query(&input, &q)
            }
            (KnnLeafKind::Radial { min_score }, KnnTarget::Float(t)) => {
                let q = FloatVectorSimilarityQuery::new(
                    self.field.clone(),
                    t.clone(),
                    min_score,
                    FloatVectorSimilarityQuery::DEFAULT_DECAY,
                )?;
                let seg = [KnnSegment {
                    vectors: input,
                    doc_base: 0,
                }];
                vector_query::float_vector_similarity_hits(&seg, &q)
            }
            (KnnLeafKind::Radial { min_score }, KnnTarget::Byte(t)) => {
                let q = ByteVectorSimilarityQuery::new(
                    self.field.clone(),
                    t.clone(),
                    min_score,
                    ByteVectorSimilarityQuery::DEFAULT_DECAY,
                )?;
                let seg = [KnnSegment {
                    vectors: input,
                    doc_base: 0,
                }];
                vector_query::byte_vector_similarity_hits(&seg, &q)
            }
        }
    }
}

impl DocumentQuery for KnnLeafQuery {
    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        self.score_leaf_filtered(leaf, boost, None, collector)
    }

    fn score_leaf_filtered(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        filter: Option<&FixedBitSet>,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some(reader) = leaf.reader else {
            return Err(Error::MissingSegmentReader(format!(
                "KNN query on {:?}",
                self.field
            )));
        };
        let mut hits = self.leaf_hits(reader, leaf.live_docs, filter)?;
        hits.sort_unstable_by_key(|h| h.doc_id);
        for h in hits {
            collector.collect(h.doc_id, h.score * boost);
        }
        Ok(())
    }

    fn constant_score(&self) -> bool {
        false
    }

    fn needs_live_docs(&self) -> bool {
        true
    }

    fn empty_filter_matches_nothing(&self) -> bool {
        true
    }

    fn equals(&self, other: &dyn DocumentQuery) -> bool {
        other
            .as_any_query()
            .downcast_ref::<Self>()
            .is_some_and(|o| o == self)
    }
}

/// One segment's opened vector readers for a field.
struct LeafVectors<'r> {
    flat: lucene_codecs::vectors::FlatVectorsReader<'r>,
    graph: Option<lucene_codecs::hnsw_vectors::HnswVectorsReader<'r>>,
    ivf: Option<std::sync::Arc<lucene_ivf::IvfReader>>,
    scalar_quantized:
        Option<lucene_codecs::scalar_quantized_vectors::ScalarQuantizedVectorsReader<'r>>,
}

impl<'r> LeafVectors<'r> {
    fn input<'a>(
        &'a self,
        reader: &'a SegmentReader,
        live_docs: Option<&'a FixedBitSet>,
        filter: Option<&'a FixedBitSet>,
    ) -> VectorsInput<'a>
    where
        'r: 'a,
    {
        VectorsInput {
            flat: self.flat.clone(),
            hnsw: match (&self.ivf, &self.scalar_quantized) {
                (Some(ivf), _) => Some(GraphReader::Ivf(ivf.clone())),
                (None, Some(codes)) => Some(GraphReader::ScalarQuantized {
                    graph: self.graph.clone(),
                    codes: codes.clone(),
                }),
                (None, None) => self.graph.clone().map(GraphReader::from),
            },
            field_infos: reader.field_infos(),
            live_docs,
            filter,
            max_doc: reader.max_doc,
        }
    }
}

/// `field`'s vector readers in `reader`, `None` when the segment has no
/// vectors for it.
fn open_leaf<'r>(reader: &'r SegmentReader, field: &str) -> Result<Option<LeafVectors<'r>>> {
    let Some(info) = reader
        .field_infos()
        .fields
        .iter()
        .find(|f| f.name == field && f.vector_dimension > 0)
    else {
        return Ok(None);
    };
    let Some((flat, graph)) = reader.knn_vectors_for(info.number)? else {
        return Ok(None);
    };
    let ivf = reader.ivf_for(info.number)?;
    let scalar_quantized = reader.scalar_quantized_for(info.number)?;
    Ok(Some(LeafVectors {
        flat,
        graph,
        ivf,
        scalar_quantized,
    }))
}

/// `KnnFloatVectorQuery(field, target, candidates, filter)` rewritten over
/// `segments`, then cut to its best `k` (OpenSearch's `OSKnnFloatVectorQuery`,
/// which searches `max(k, ef_search)` candidates and keeps `k`): global doc
/// ids, best first. Every segment must carry its reader; its `live_docs` are
/// the deletions the search skips.
///
/// # Errors
/// A `k` or `candidates` below `1`, a target whose dimension or encoding is
/// not the field's, unreadable vector files.
pub fn knn_top_hits(
    segments: &[OpenSegment<'_>],
    field: &str,
    target: &KnnTarget,
    k: usize,
    candidates: usize,
    filter: Option<&Clause>,
) -> Result<Vec<ScoreDoc>> {
    top_hits(segments, field, target, k, candidates, filter, None, 0, &[])
}

/// [`knn_top_hits`] probing `ivf_nprobes` clusters in each segment whose
/// field has an IVF index (`LuceneRustIvfVectorsFormat`); `0` probes the
/// depth each segment calibrated. Other formats ignore it.
///
/// # Errors
/// As [`knn_top_hits`].
pub fn knn_top_hits_with(
    segments: &[OpenSegment<'_>],
    field: &str,
    target: &KnnTarget,
    k: usize,
    candidates: usize,
    filter: Option<&Clause>,
    ivf_nprobes: usize,
) -> Result<Vec<ScoreDoc>> {
    top_hits(
        segments,
        field,
        target,
        k,
        candidates,
        filter,
        None,
        ivf_nprobes,
        &[],
    )
}

/// The documents of an indexing buffer, read as consecutive segments of a
/// reader, whose vectors of a field have a graph kept over them as they
/// arrived ([`BufferGraph`]).
#[derive(Debug, Clone)]
pub struct BufferedVectors<'a> {
    pub graph: &'a BufferGraph,
    /// How many of the graph's vectors the reader holds.
    pub count: i32,
    /// The segments holding the buffer's documents, in order from its first
    /// document: each segment's documents are the next `max_doc` of the
    /// buffer.
    pub segments: std::ops::Range<usize>,
}

/// [`knn_top_hits_with`] where each of `buffers` is searched through its
/// graph (the vectors it has not reached yet exactly) instead of scanning
/// its segments one by one. The hits merge as every segment's do: the
/// scores are the flat format's either way.
///
/// # Errors
/// As [`knn_top_hits`], and a `buffers` segment outside `segments`.
#[allow(clippy::too_many_arguments)]
pub fn knn_top_hits_buffered(
    segments: &[OpenSegment<'_>],
    field: &str,
    target: &KnnTarget,
    k: usize,
    candidates: usize,
    filter: Option<&Clause>,
    ivf_nprobes: usize,
    buffers: &[BufferedVectors<'_>],
) -> Result<Vec<ScoreDoc>> {
    top_hits(
        segments,
        field,
        target,
        k,
        candidates,
        filter,
        None,
        ivf_nprobes,
        buffers,
    )
}

/// [`knn_top_hits`] over a nested field (the `lucene` engine's
/// `OSDiversifyingChildrenFloatKnnVectorQuery`): the best child of each of
/// the best `k` parents, the blocks `parents` closes.
///
/// With `expand` (`expand_nested_docs`, k-NN's `ExpandNestedDocsQuery`)
/// every child of those parents that passes the filter and has a vector is
/// scored too, and all of them are returned, best first.
///
/// # Errors
/// As [`knn_top_hits`], and the parent filter's own errors.
#[allow(clippy::too_many_arguments)]
pub fn knn_top_hits_nested(
    segments: &[OpenSegment<'_>],
    field: &str,
    target: &KnnTarget,
    k: usize,
    candidates: usize,
    filter: Option<&Clause>,
    parents: &dyn BitSetProducer,
    expand: bool,
) -> Result<Vec<ScoreDoc>> {
    top_hits(
        segments,
        field,
        target,
        k,
        candidates,
        filter,
        Some((parents, expand)),
        0,
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn top_hits(
    segments: &[OpenSegment<'_>],
    field: &str,
    target: &KnnTarget,
    k: usize,
    candidates: usize,
    filter: Option<&Clause>,
    nested: Option<(&dyn BitSetProducer, bool)>,
    ivf_nprobes: usize,
    buffers: &[BufferedVectors<'_>],
) -> Result<Vec<ScoreDoc>> {
    if buffers.iter().any(|b| b.segments.end > segments.len()) {
        return Err(Error::MissingSegmentReader(format!(
            "KNN query on {field:?}: a buffer's segments outside the reader"
        )));
    }
    let buffered = |i: usize| buffers.iter().any(|b| b.segments.contains(&i));
    let filters = match filter {
        Some(f) => Some(vector_query::filter_bitsets(segments, f)?),
        None => None,
    };
    let mut opened = Vec::with_capacity(segments.len()); // alloc-ok: one per segment
    for (i, seg) in segments.iter().enumerate() {
        let Some(reader) = seg.reader else {
            return Err(Error::MissingSegmentReader(format!(
                "KNN query on {field:?}"
            )));
        };
        // A buffer's segments are searched through its graph: their own
        // vector readers are not opened.
        let vectors = if nested.is_none() && buffered(i) {
            None
        } else {
            open_leaf(reader, field)?
        };
        opened.push((reader, vectors));
    }
    let mut knn = Vec::with_capacity(segments.len()); // alloc-ok: one per segment
                                                      // The segment each `knn` entry is, and (nested) its parent bit set.
    let mut leaves = Vec::with_capacity(segments.len()); // alloc-ok: one per segment
    let mut parent_sets = Vec::with_capacity(segments.len()); // alloc-ok: one per segment
    for (i, (seg, (reader, vectors))) in segments.iter().zip(&opened).enumerate() {
        let Some(vectors) = vectors else { continue };
        if nested.is_none() && buffered(i) {
            continue;
        }
        let filter = filters.as_ref().map(|f| &f[i]);
        knn.push(KnnSegment {
            vectors: vectors.input(reader, seg.live_docs, filter),
            doc_base: seg.doc_base,
        });
        leaves.push(i);
        if let Some((parents, _)) = nested {
            parent_sets.push(parents.bit_set(seg)?);
        }
    }
    if knn.is_empty() && (buffers.is_empty() || nested.is_some()) {
        return Ok(Vec::new());
    }
    // More candidates than the reader has documents find no more of them; the
    // bound keeps a request's `ef_search` from sizing the search's heaps.
    let docs = opened
        .iter()
        .map(|(reader, _)| usize::try_from(reader.max_doc).unwrap_or(0))
        .fold(0usize, usize::saturating_add)
        .max(1);
    let candidates = candidates.max(k).min(docs);
    let parents: Vec<Option<&FixedBitSet>> = parent_sets.iter().map(|b| b.as_deref()).collect();
    let mut hits = match (target, nested.is_some()) {
        _ if knn.is_empty() => Vec::new(),
        (KnnTarget::Float(t), false) => {
            let q = KnnFloatVectorQuery::new(field, t.clone(), candidates)?
                .with_ivf_nprobes(ivf_nprobes);
            vector_query::search_knn_float_vector_query_multi_segment(&knn, &q)?
        }
        (KnnTarget::Byte(t), false) => {
            let q = KnnByteVectorQuery::new(field, t.clone(), candidates)?;
            vector_query::search_knn_byte_vector_query_multi_segment(&knn, &q)?
        }
        (KnnTarget::Float(t), true) => {
            let q = KnnFloatVectorQuery::new(field, t.clone(), candidates)?;
            vector_query::search_diversifying_children_float_knn_multi_segment(&knn, &parents, &q)?
        }
        (KnnTarget::Byte(t), true) => {
            let q = KnnByteVectorQuery::new(field, t.clone(), candidates)?;
            vector_query::search_diversifying_children_byte_knn_multi_segment(&knn, &parents, &q)?
        }
    };
    if nested.is_none() && !buffers.is_empty() {
        let query = match target {
            KnnTarget::Float(t) => BufferQuery::Float32(t.clone()),
            KnnTarget::Byte(t) => BufferQuery::Byte(t.clone()),
        };
        for b in buffers {
            hits.extend(buffer_hits(
                b,
                segments,
                filters.as_deref(),
                &query,
                candidates,
            )?);
        }
        // `TopDocs.merge`'s order: the better score first, the lower doc on
        // a tie.
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
    }
    hits.truncate(k);
    if !matches!(nested, Some((_, true))) {
        return Ok(hits);
    }
    let mut expanded = Vec::new();
    for ((seg, &leaf), parents) in knn.iter().zip(&leaves).zip(&parents) {
        let Some(parents) = parents else { continue };
        let Some(siblings) = siblings(&hits, seg, parents, filters.as_ref().map(|f| &f[leaf]))
        else {
            continue;
        };
        let (reader, vectors) = &opened[leaf];
        let Some(vectors) = vectors else { continue };
        let input = vectors.input(reader, segments[leaf].live_docs, Some(&siblings));
        let n = siblings.cardinality().max(1);
        let local = match target {
            KnnTarget::Float(t) => {
                let q = KnnFloatVectorQuery::new(field, t.clone(), n)?;
                vector_query::search_knn_float_vector_query(&input, &q)?
            }
            KnnTarget::Byte(t) => {
                let q = KnnByteVectorQuery::new(field, t.clone(), n)?;
                vector_query::search_knn_byte_vector_query(&input, &q)?
            }
        };
        expanded.extend(local.into_iter().map(|h| ScoreDoc {
            doc_id: h.doc_id.saturating_add(seg.doc_base),
            score: h.score,
        }));
    }
    // `TopDocs.merge`'s order: the better score first, the lower doc on a tie.
    expanded.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });
    Ok(expanded)
}

/// The best `candidates` of buffer `b` for `query`, as global doc ids:
/// the ordinals its segments' deletes and `filters` accept.
fn buffer_hits(
    b: &BufferedVectors<'_>,
    segments: &[OpenSegment<'_>],
    filters: Option<&[FixedBitSet]>,
    query: &BufferQuery,
    candidates: usize,
) -> Result<Vec<ScoreDoc>> {
    let segs = segments.get(b.segments.clone()).unwrap_or_default();
    // Each segment's first buffer document, then the end of the last.
    let mut starts = Vec::with_capacity(segs.len().saturating_add(1)); // alloc-ok: one per segment
    let mut at = 0i32;
    for seg in segs {
        starts.push(at);
        let max_doc = seg.reader.map_or(0, |r| r.max_doc);
        at = at.saturating_add(max_doc);
    }
    starts.push(at);
    let locate = |doc: i32| -> Option<(usize, i32)> {
        let j = starts.partition_point(|&s| s <= doc).checked_sub(1)?;
        (j < segs.len()).then(|| (j, doc.saturating_sub(starts[j])))
    };
    let masked = segs.iter().any(|s| s.live_docs.is_some()) || filters.is_some();
    let accept = if masked {
        let n = usize::try_from(b.count).unwrap_or(0);
        let mut bits = FixedBitSet::new(n);
        for ord in 0..n {
            let Some((j, local)) = b.graph.doc(ord as i32).and_then(locate) else {
                continue;
            };
            let live = segs[j].live_docs.is_none_or(|l| l.get_doc(local));
            // FBS: `sets` is the slice of every segment's filter, not a bitset.
            let passes = filters
                .and_then(|sets| sets.get(b.segments.start.saturating_add(j)))
                .is_none_or(|f| f.get_doc(local));
            if live && passes {
                // FBS: `ord < n`, the bitset's length.
                bits.set(ord);
            }
        }
        Some(bits)
    } else {
        None
    };
    let hits = b
        .graph
        .search(query, candidates, b.count, accept.as_ref())?;
    Ok(hits
        .into_iter()
        .filter_map(|(doc, score)| {
            let (j, local) = locate(doc)?;
            Some(ScoreDoc {
                doc_id: segs[j].doc_base.saturating_add(local),
                score,
            })
        })
        .collect())
}

/// k-NN's `GroupedNestedDocIdSetIterator`: in `seg`'s doc space, every child
/// of the parents of the `hits` that fall in it, and that pass `filter`.
/// `None` when no hit is in this segment.
fn siblings(
    hits: &[ScoreDoc],
    seg: &KnnSegment<'_>,
    parents: &FixedBitSet,
    filter: Option<&FixedBitSet>,
) -> Option<FixedBitSet> {
    let max_doc = usize::try_from(seg.vectors.max_doc).unwrap_or(0);
    let mut out: Option<FixedBitSet> = None;
    for hit in hits {
        let Some(local) = hit
            .doc_id
            .checked_sub(seg.doc_base)
            .and_then(|d| usize::try_from(d).ok())
            .filter(|&d| d < max_doc && d < parents.len())
        else {
            continue;
        };
        let Some(parent) = parents.next_set_bit(local) else {
            continue;
        };
        let first = match local.checked_sub(1) {
            Some(before) => parents
                .prev_set_bit(before)
                .map_or(0, |p| p.saturating_add(1)),
            None => 0,
        };
        let bits = out.get_or_insert_with(|| FixedBitSet::new(max_doc));
        for child in first..parent.min(bits.len()) {
            if filter.is_none_or(|f| f.get_doc(child as i32)) {
                bits.set(child);
            }
        }
    }
    out
}
