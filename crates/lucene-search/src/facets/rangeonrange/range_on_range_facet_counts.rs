//! `RangeOnRangeFacetCounts`, `LongRangeOnRangeFacetCounts` and
//! `DoubleRangeOnRangeFacetCounts`: each document's range compared with
//! every query range.

use std::fmt;

use lucene_index::document::RangeQueryType;

use super::{DoubleRange, LongRange};
use crate::facets::facet_counts_with_filter_query::FacetCountsWithFilterQuery;
use crate::facets::facet_result::{FacetResult, FacetValue, LabelAndValue};
use crate::facets::facet_utils::{for_each_iter, validate_dim_and_path};
use crate::facets::facets_collector::FacetsCollector;
use crate::facets::facets_trait::{validate_top_n, Facets};
use crate::facets::taxonomy::facet_label::java_compare;
use crate::facets::top_ord_and_number_queue::PriorityQueue;
use crate::index_searcher::IndexSearcher;
use crate::query::BooleanQuery;
use crate::reader::doc_values::get_binary;
use crate::reader::BinaryDocValues;
use crate::{Error, Result};

/// `RangeOnRangeFacetCounts`.
struct RangeOnRangeFacetCounts {
    labels: Vec<String>,
    counts: Vec<i32>,
    field: String,
    tot_count: i32,
}

/// The arguments every constructor shares.
struct Args<'a, 'h> {
    field: &'a str,
    hits: &'a FacetsCollector<'h>,
    query_type: RangeQueryType,
    fast_match_query: Option<BooleanQuery>,
}

impl RangeOnRangeFacetCounts {
    fn new(
        args: Args<'_, '_>,
        searcher: Option<&IndexSearcher<'_, '_>>,
        encoded_ranges: Vec<Vec<u8>>,
        labels: Vec<String>,
    ) -> Result<Self> {
        let mut c = RangeOnRangeFacetCounts {
            counts: vec![0; labels.len()],
            labels,
            field: args.field.to_string(),
            tot_count: 0,
        };
        c.count(args, searcher, &encoded_ranges)?;
        Ok(c)
    }

    /// `count(field, matchingDocs, encodedRanges, numEncodedValueBytes,
    /// queryType)`.
    fn count(
        &mut self,
        args: Args<'_, '_>,
        searcher: Option<&IndexSearcher<'_, '_>>,
        encoded_ranges: &[Vec<u8>],
    ) -> Result<()> {
        const BYTES: usize = 8;
        let packed_len = encoded_ranges.first().map_or(0, Vec::len);
        let dims = packed_len / (2 * BYTES);
        let base = FacetCountsWithFilterQuery::new(args.fast_match_query);
        let mut missing = 0i32;
        for hits in args.hits.matching_docs() {
            let mut values = get_binary(hits.context.reader, &self.field)?;
            let it = match base.fast_match_query() {
                None => Some(Box::new(hits.bits.iterator()) as Box<_>),
                Some(_) => {
                    let searcher = searcher.ok_or_else(|| {
                        Error::IllegalState(
                            "a fast-match query needs the searcher the hits came from".into(),
                        )
                    })?;
                    base.create_iterator(searcher, hits, Vec::new())?
                }
            };
            let Some(mut it) = it else {
                continue;
            };
            self.tot_count = self.tot_count.wrapping_add(hits.total_hits);
            let counts = &mut self.counts;
            for_each_iter(it.as_mut(), |doc| {
                if !values.advance_exact(doc)? {
                    missing = missing.wrapping_add(1);
                    return Ok(());
                }
                let packed = packed_value(values.as_ref(), packed_len)?;
                let mut has_valid_range = false;
                for (r, encoded) in encoded_ranges.iter().enumerate() {
                    if args.query_type.matches(encoded, packed, dims, BYTES) {
                        counts[r] = counts[r].wrapping_add(1);
                        has_valid_range = true;
                    }
                }
                if !has_valid_range {
                    missing = missing.wrapping_add(1);
                }
                Ok(())
            })?;
        }
        self.tot_count = self.tot_count.wrapping_sub(missing);
        Ok(())
    }
}

/// `BinaryRangeDocValues.getPackedValue()`: the first `len` bytes of the
/// document's value.
fn packed_value<'v>(values: &'v (dyn BinaryDocValues + '_), len: usize) -> Result<&'v [u8]> {
    values.binary_value().get(..len).ok_or_else(|| {
        Error::IllegalState(format!(
            "range doc value of {} bytes is shorter than the ranges' {len}",
            values.binary_value().len()
        ))
    })
}

/// `RangeOnRangeFacetCounts.Entry`.
struct Entry<'a> {
    count: i32,
    label: &'a str,
}

impl Facets for RangeOnRangeFacetCounts {
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

    fn get_top_children(
        &self,
        top_n: i32,
        dim: &str,
        path: &[&str],
    ) -> Result<Option<FacetResult>> {
        validate_top_n(top_n)?;
        validate_dim_and_path(&self.field, dim, path)?;
        let max = usize::try_from(top_n).unwrap_or(0).min(self.counts.len());
        let mut pq = PriorityQueue::new(max, |a: &Entry<'_>, b: &Entry<'_>| {
            a.count < b.count || (a.count == b.count && java_compare(b.label, a.label).is_lt())
        });
        let mut child_count = 0i32;
        for (label, &count) in self.labels.iter().zip(&self.counts) {
            if count != 0 {
                child_count = child_count.wrapping_add(1);
                pq.insert_with_overflow(Entry { count, label });
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
            "RangeOnRangeFacetCounts.getSpecificValue".into(),
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

/// `getEncodedRanges(ranges)`: every range encoded, all of the first one's
/// dimensions.
fn encoded<R>(
    ranges: &[R],
    dims: impl Fn(&R) -> usize,
    encode: impl Fn(&R) -> Result<Vec<u8>>,
) -> Result<Vec<Vec<u8>>> {
    let Some(first) = ranges.first() else {
        return Err(Error::IllegalArgument("ranges cannot be empty".into()));
    };
    let d = dims(first);
    ranges
        .iter()
        .map(|r| {
            if dims(r) != d {
                return Err(Error::IllegalArgument(format!(
                    "all ranges must have {d} dimensions, found {}",
                    dims(r)
                )));
            }
            encode(r)
        })
        .collect()
}

macro_rules! range_on_range_facet_counts {
    ($(#[$doc:meta])* $name:ident, $range:ty, $java:literal) => {
        $(#[$doc])*
        pub struct $name {
            inner: RangeOnRangeFacetCounts,
        }

        impl $name {
            #[doc = concat!("`new ", $java, "(field, hits, queryType, ranges...)`.")]
            ///
            /// # Errors
            /// [`Error::IllegalArgument`] for no ranges, ranges of mixed
            /// dimensions, or one that does not encode;
            /// [`Error::IllegalState`] for a field with doc values other
            /// than `BINARY`; whatever reading them reports.
            pub fn new(
                field: &str,
                hits: &FacetsCollector<'_>,
                query_type: RangeQueryType,
                ranges: &[$range],
            ) -> Result<Self> {
                Self::with_fast_match(field, None, hits, query_type, None, ranges)
            }

            #[doc = concat!("`new ", $java, "(field, hits, queryType, fastMatchQuery, ranges...)`:")]
            /// only the hits `fast_match_query` matches, run through
            /// `searcher` (the one the hits came from; needed only with a
            /// query).
            ///
            /// # Errors
            /// As [`Self::new`]; [`Error::IllegalState`] for a query
            /// without a searcher.
            pub fn with_fast_match(
                field: &str,
                searcher: Option<&IndexSearcher<'_, '_>>,
                hits: &FacetsCollector<'_>,
                query_type: RangeQueryType,
                fast_match_query: Option<BooleanQuery>,
                ranges: &[$range],
            ) -> Result<Self> {
                let encoded_ranges = encoded(ranges, <$range>::dims, <$range>::encode)?;
                let labels = ranges.iter().map(|r| r.label.clone()).collect();
                let inner = RangeOnRangeFacetCounts::new(
                    Args {
                        field,
                        hits,
                        query_type,
                        fast_match_query,
                    },
                    searcher,
                    encoded_ranges,
                    labels,
                )?;
                Ok($name { inner })
            }
        }

        impl Facets for $name {
            fn get_all_children(&self, dim: &str, path: &[&str]) -> Result<Option<FacetResult>> {
                self.inner.get_all_children(dim, path)
            }

            fn get_top_children(
                &self,
                top_n: i32,
                dim: &str,
                path: &[&str],
            ) -> Result<Option<FacetResult>> {
                self.inner.get_top_children(top_n, dim, path)
            }

            fn get_specific_value(&self, dim: &str, path: &[&str]) -> Result<FacetValue> {
                self.inner.get_specific_value(dim, path)
            }

            fn get_all_dims(&self, top_n: i32) -> Result<Vec<FacetResult>> {
                self.inner.get_all_dims(top_n)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("field", &self.inner.field)
                    .field("tot_count", &self.inner.tot_count)
                    .field("counts", &self.inner.counts)
                    .finish()
            }
        }
    };
}

range_on_range_facet_counts!(
    /// `LongRangeOnRangeFacetCounts`: a `LongRangeDocValuesField` counted
    /// into [`LongRange`]s.
    LongRangeOnRangeFacetCounts,
    LongRange,
    "LongRangeOnRangeFacetCounts"
);

range_on_range_facet_counts!(
    /// `DoubleRangeOnRangeFacetCounts`: a `DoubleRangeDocValuesField`
    /// counted into [`DoubleRange`]s.
    DoubleRangeOnRangeFacetCounts,
    DoubleRange,
    "DoubleRangeOnRangeFacetCounts"
);

#[cfg(test)]
mod tests;
