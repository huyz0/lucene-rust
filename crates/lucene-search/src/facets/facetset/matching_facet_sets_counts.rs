//! `MatchingFacetSetsCounts`: per matcher, the documents with a facet set
//! it matches.

use std::fmt;

use super::{FacetSetDecoder, FacetSetMatcher};
use crate::facets::facet_counts_with_filter_query::FacetCountsWithFilterQuery;
use crate::facets::facet_result::{FacetResult, FacetValue, LabelAndValue};
use crate::facets::facet_utils::{for_each_hit, validate_dim_and_path};
use crate::facets::facets_collector::FacetsCollector;
use crate::facets::facets_trait::{validate_top_n, Facets};
use crate::facets::taxonomy::facet_label::java_compare;
use crate::facets::top_ord_and_number_queue::PriorityQueue;
use crate::index_searcher::IndexSearcher;
use crate::query::BooleanQuery;
use crate::reader::doc_values::get_binary;
use crate::reader::NO_MORE_DOCS;
use crate::{Error, Result};

/// `MatchingFacetSetsCounts`.
pub struct MatchingFacetSetsCounts {
    labels: Vec<String>,
    counts: Vec<i32>,
    field: String,
    tot_count: i32,
}

/// The decoding state of one count.
struct Counter<'m> {
    matchers: &'m [&'m dyn FacetSetMatcher],
    decoder: &'m FacetSetDecoder,
    counts: Vec<i32>,
    tot_count: i32,
    /// `dimValues`, sized by the first document's dimension count.
    dim_values: Option<Vec<i64>>,
}

impl Counter<'_> {
    /// One document's facet sets.
    fn count_doc(&mut self, bytes: &[u8]) -> Result<()> {
        let Some(head) = bytes.get(..4) else {
            return Err(Error::IllegalState(format!(
                "facet sets value of {} bytes has no dimension count",
                bytes.len()
            )));
        };
        let dim_values = match &mut self.dim_values {
            Some(v) => v,
            None => {
                let num_dims = lucene_util::numeric_utils::sortable_bytes_to_int(head, 0);
                let n = usize::try_from(num_dims).map_err(|_| {
                    Error::IllegalState(format!("negative facet set dimensions: {num_dims}"))
                })?;
                // Four bytes per dimension at least: a larger count is a
                // corrupt value, not an allocation to attempt.
                if n > (bytes.len() - 4) / 4 {
                    return Err(Error::IllegalState(format!(
                        "facet sets value of {} bytes cannot hold {n} dimensions",
                        bytes.len()
                    )));
                }
                self.dim_values.insert(vec![0; n])
            }
        };
        let mut should_count = false;
        let mut start = 4usize;
        while start < bytes.len() {
            let read = (self.decoder)(bytes, start, dim_values)?;
            if read == 0 {
                // Java loops forever here; refuse the value instead.
                return Err(Error::IllegalState(format!(
                    "facet sets value of {} bytes: no progress decoding at byte {start}",
                    bytes.len()
                )));
            }
            start = start.saturating_add(read);
            for (j, m) in self.matchers.iter().enumerate() {
                if m.matches(dim_values) {
                    self.counts[j] = self.counts[j].wrapping_add(1);
                    should_count = true;
                }
            }
        }
        if should_count {
            self.tot_count = self.tot_count.wrapping_add(1);
        }
        Ok(())
    }
}

impl MatchingFacetSetsCounts {
    /// `new MatchingFacetSetsCounts(field, hits, facetSetDecoder,
    /// facetSetMatchers...)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for no matchers, or matchers of different
    /// dimensions; [`Error::IllegalState`] for a field with doc values other
    /// than `BINARY`; whatever decoding and reading report.
    pub fn new(
        field: &str,
        hits: &FacetsCollector<'_>,
        decoder: &FacetSetDecoder,
        matchers: &[&dyn FacetSetMatcher],
    ) -> Result<Self> {
        Self::with_fast_match(field, None, hits, decoder, None, matchers)
    }

    /// `new MatchingFacetSetsCounts(field, hits, facetSetDecoder,
    /// fastMatchQuery, facetSetMatchers...)`: only the hits
    /// `fast_match_query` matches, run through `searcher` (the one the hits
    /// came from; needed only with a query).
    ///
    /// # Errors
    /// As [`Self::new`]; [`Error::IllegalState`] for a query without a
    /// searcher.
    pub fn with_fast_match(
        field: &str,
        searcher: Option<&IndexSearcher<'_, '_>>,
        hits: &FacetsCollector<'_>,
        decoder: &FacetSetDecoder,
        fast_match_query: Option<BooleanQuery>,
        matchers: &[&dyn FacetSetMatcher],
    ) -> Result<Self> {
        let Some(first) = matchers.first() else {
            return Err(Error::IllegalArgument(
                "facetSetMatchers cannot be null or empty".into(),
            ));
        };
        if matchers.iter().any(|m| m.dims() != first.dims()) {
            return Err(Error::IllegalArgument(
                "All facet set matchers must be the same dimensionality".into(),
            ));
        }
        let base = FacetCountsWithFilterQuery::new(fast_match_query);
        let mut counter = Counter {
            matchers,
            decoder,
            counts: vec![0; matchers.len()],
            tot_count: 0,
            dim_values: None,
        };
        for hits in hits.matching_docs() {
            let mut values = get_binary(hits.context.reader, field)?;
            // `createIterator(hits, binaryDocValues)`: the hits the field
            // has a value for (and the query matches).
            if base.fast_match_query().is_none() {
                // The first document fixes the dimension count of
                // `dimValues` for the segment.
                counter.dim_values = None;
                for_each_hit(&hits.bits, values.as_mut(), |dv| {
                    counter.count_doc(dv.binary_value())
                })?;
                continue;
            }
            let searcher = searcher.ok_or_else(|| {
                Error::IllegalState(
                    "a fast-match query needs the searcher the hits came from".into(),
                )
            })?;
            let Some(mut it) = base.create_iterator(searcher, hits, Vec::new())? else {
                continue;
            };
            counter.dim_values = None;
            let mut doc = it.next_doc()?;
            while doc != NO_MORE_DOCS {
                if values.advance_exact(doc)? {
                    counter.count_doc(values.binary_value())?;
                }
                doc = it.next_doc()?;
            }
        }
        Ok(MatchingFacetSetsCounts {
            labels: matchers.iter().map(|m| m.label().to_string()).collect(),
            counts: counter.counts,
            field: field.to_string(),
            tot_count: counter.tot_count,
        })
    }
}

/// `MatchingFacetSetsCounts.Entry`.
struct Entry<'a> {
    label: &'a str,
    count: i32,
}

impl Facets for MatchingFacetSetsCounts {
    fn get_all_children(&self, dim: &str, path: &[&str]) -> Result<Option<FacetResult>> {
        validate_dim_and_path(&self.field, dim, path)?;
        let label_values: Vec<LabelAndValue> = self
            .labels
            .iter()
            .zip(&self.counts)
            .map(|(l, &c)| LabelAndValue::new(l.clone(), FacetValue::Int(c)))
            .collect();
        let n = i32::try_from(label_values.len()).unwrap_or(i32::MAX);
        Ok(Some(FacetResult::new(
            dim.to_string(),
            path.iter().map(|p| p.to_string()).collect(),
            FacetValue::Int(self.tot_count),
            label_values,
            n,
        )))
    }

    /// Java does not check `dim` and `path` here; neither does this.
    ///
    /// Java fills its queue with zero-count sentinels and replaces the top;
    /// the children kept, and their order, are those of the bounded queue
    /// here (the order is total but for equal labels with equal counts,
    /// which print the same).
    fn get_top_children(
        &self,
        top_n: i32,
        dim: &str,
        path: &[&str],
    ) -> Result<Option<FacetResult>> {
        validate_top_n(top_n)?;
        let max = usize::try_from(top_n).unwrap_or(0).min(self.counts.len());
        let mut pq = PriorityQueue::new(max, |a: &Entry<'_>, b: &Entry<'_>| {
            a.count < b.count || (a.count == b.count && java_compare(b.label, a.label).is_lt())
        });
        let mut child_count = 0i32;
        for (label, &count) in self.labels.iter().zip(&self.counts) {
            if count > 0 {
                child_count = child_count.wrapping_add(1);
                pq.insert_with_overflow(Entry { label, count });
            }
        }
        let mut results = Vec::with_capacity(pq.size());
        while let Some(e) = pq.pop() {
            results.push(LabelAndValue::new(
                e.label.to_string(),
                FacetValue::Int(e.count),
            ));
        }
        results.reverse();
        Ok(Some(FacetResult::new(
            dim.to_string(),
            path.iter().map(|p| p.to_string()).collect(),
            FacetValue::Int(self.tot_count),
            results,
            child_count,
        )))
    }

    fn get_specific_value(&self, _dim: &str, _path: &[&str]) -> Result<FacetValue> {
        Err(Error::Unsupported(
            "MatchingFacetSetsCounts.getSpecificValue".into(),
        ))
    }

    fn get_all_dims(&self, top_n: i32) -> Result<Vec<FacetResult>> {
        validate_top_n(top_n)?;
        Ok(self
            .get_top_children(top_n, &self.field, &[])?
            .into_iter()
            .collect())
    }
}

impl fmt::Debug for MatchingFacetSetsCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatchingFacetSetsCounts")
            .field("field", &self.field)
            .field("tot_count", &self.tot_count)
            .field("counts", &self.counts)
            .finish()
    }
}

#[cfg(test)]
mod tests;
