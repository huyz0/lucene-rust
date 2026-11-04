//! The approximate `neural_sparse` query as a [`DocumentQuery`]:
//! neural-search's `SparseVectorQuery` and `SparseQueryWeight`.
//!
//! On a segment at or over the field's `approximate_threshold` it runs
//! [`crate::search()`] over the segment's clustered postings, under the
//! clause's filter when it has one. On any other segment it does not score:
//! the clause's fallback (neural-search's `fallbackQuery`, a sum of linear
//! `FeatureField` queries) runs there instead. So does a segment written
//! without [`crate::SparseCodec`], which has no clusters for the field: its
//! `FeatureField` postings give the exact scores.

use crate::postings::{self, ClusteredPostings};
use crate::search::{SegmentView, SeismicField, SeismicQuery};
use lucene_search::collector::ScoringCollector;
use lucene_search::directory_reader::SegmentReader;
use lucene_search::document::DocumentQuery;
use lucene_search::multi_segment::OpenSegment;
use lucene_util::fixed_bit_set::FixedBitSet;
use std::sync::Arc;

fn error(e: impl std::fmt::Display) -> lucene_search::Error {
    lucene_search::Error::DocumentQuery(e.to_string())
}

/// A segment's clustered postings, opened once per segment core.
struct SegmentPostings(Option<ClusteredPostings>);

/// The clustered postings of `reader`'s segment, `None` when it has none.
fn clustered_postings(reader: &SegmentReader) -> lucene_search::Result<Arc<SegmentPostings>> {
    reader.core_attachment(|r| {
        let (Some(sit), Some(sip)) = (
            r.codec_file(postings::TERMS_EXTENSION),
            r.codec_file(postings::POSTINGS_EXTENSION),
        ) else {
            return Ok(SegmentPostings(None));
        };
        ClusteredPostings::open(sit, Arc::clone(sip), &r.segment_id())
            .map(|p| SegmentPostings(Some(p)))
            .map_err(error)
    })
}

/// `neural_sparse` with `method_parameters` on a `sparse_vector` field.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseAnnQuery {
    pub field: String,
    /// The query tokens and weights, as given.
    pub tokens: Vec<(u32, f32)>,
    /// `top_n`: how many of the heaviest tokens' postings are walked.
    pub top_n: usize,
    /// `k`: results per segment; 0 means 10.
    pub k: usize,
    pub heap_factor: f32,
}

impl SparseAnnQuery {
    fn seismic_field(&self, reader: &SegmentReader) -> lucene_search::Result<Option<SeismicField>> {
        match reader
            .field_infos()
            .fields
            .iter()
            .find(|f| f.name == self.field)
        {
            None => Ok(None),
            Some(info) => SeismicField::from_field_info(info).map_err(error),
        }
    }
}

impl DocumentQuery for SparseAnnQuery {
    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn constant_score(&self) -> bool {
        false
    }

    fn equals(&self, other: &dyn DocumentQuery) -> bool {
        other
            .as_any_query()
            .downcast_ref::<Self>()
            .is_some_and(|o| o == self)
    }

    /// `SparseQueryWeight`: a `lucene` engine segment under the threshold is
    /// the fallback's, and so is one without clusters for the field.
    fn scores_leaf(&self, leaf: &OpenSegment<'_>) -> lucene_search::Result<bool> {
        let Some(reader) = leaf.reader else {
            return Ok(true);
        };
        let Some(field) = self.seismic_field(reader)? else {
            return Ok(true);
        };
        if field.native_engine {
            return Ok(true);
        }
        if !field.is_seismic_segment(reader.max_doc) {
            return Ok(false);
        }
        Ok(clustered_postings(reader)?
            .0
            .as_ref()
            .is_some_and(|p| p.has_field(field.number)))
    }

    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> lucene_search::Result<()> {
        self.score_leaf_filtered(leaf, boost, None, collector)
    }

    fn score_leaf_filtered(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        filter: Option<&FixedBitSet>,
        collector: &mut dyn ScoringCollector,
    ) -> lucene_search::Result<()> {
        let Some(reader) = leaf.reader else {
            return Err(error("the sparse query needs the segment's reader"));
        };
        let Some(field) = self.seismic_field(reader)? else {
            return Ok(());
        };
        if field.native_engine {
            return Err(error(format!(
                "field [{}] uses the native engine, which this node does not search",
                field.name
            )));
        }
        let postings = clustered_postings(reader)?;
        let Some(postings) = &postings.0 else {
            return Ok(());
        };
        let doc_values = reader
            .doc_values_for_field(field.number)
            .and_then(|(meta, data)| meta.binary_entry(field.number).map(|e| (data, e)));
        let view = SegmentView {
            max_doc: reader.max_doc,
            live_docs: reader.live_docs(),
            doc_values,
            postings,
        };
        let query = SeismicQuery::new(
            &self.tokens,
            self.top_n,
            self.k,
            self.heap_factor,
            boost,
            &field,
        )
        .map_err(error)?;
        for (doc, score) in crate::search(&view, &field, &query, filter).map_err(error)? {
            collector.collect(doc, score);
        }
        Ok(())
    }
}
