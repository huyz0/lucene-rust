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
//! field has one, its HNSW graph. A field without a graph is searched
//! exhaustively, as Java does.

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::collector::{ScoreDoc, ScoringCollector};
use crate::directory_reader::SegmentReader;
use crate::document::DocumentQuery;
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
            hnsw: self.graph.clone().map(GraphReader::from),
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
    Ok(reader
        .knn_vectors_for(info.number)?
        .map(|(flat, graph)| LeafVectors { flat, graph }))
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
    let filters = match filter {
        Some(f) => Some(vector_query::filter_bitsets(segments, f)?),
        None => None,
    };
    let mut opened = Vec::with_capacity(segments.len()); // alloc-ok: one per segment
    for seg in segments {
        let Some(reader) = seg.reader else {
            return Err(Error::MissingSegmentReader(format!(
                "KNN query on {field:?}"
            )));
        };
        opened.push((reader, open_leaf(reader, field)?));
    }
    let mut knn = Vec::with_capacity(segments.len()); // alloc-ok: one per segment
    for (i, (seg, (reader, vectors))) in segments.iter().zip(&opened).enumerate() {
        let Some(vectors) = vectors else { continue };
        let filter = filters.as_ref().map(|f| &f[i]);
        knn.push(KnnSegment {
            vectors: vectors.input(reader, seg.live_docs, filter),
            doc_base: seg.doc_base,
        });
    }
    if knn.is_empty() {
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
    let mut hits = match target {
        KnnTarget::Float(t) => {
            let q = KnnFloatVectorQuery::new(field, t.clone(), candidates)?;
            vector_query::search_knn_float_vector_query_multi_segment(&knn, &q)?
        }
        KnnTarget::Byte(t) => {
            let q = KnnByteVectorQuery::new(field, t.clone(), candidates)?;
            vector_query::search_knn_byte_vector_query_multi_segment(&knn, &q)?
        }
    };
    hits.truncate(k);
    Ok(hits)
}
