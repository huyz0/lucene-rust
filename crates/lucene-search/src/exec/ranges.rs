//! Doc-values, index-sort and point ranges as scorer-tree leaves.

use super::build::LeafContext;
use super::{BoxScorer, Mode};
use crate::extended_query::*;
use crate::Result;

pub(crate) fn numeric_range<'a>(
    _ctx: &LeafContext<'a>,
    _q: &NumericDocValuesRangeQuery,
    _boost: f32,
    _mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    Err(crate::Error::InvalidQuery("todo".into()))
}

pub(crate) fn index_sort_range<'a>(
    _ctx: &LeafContext<'a>,
    _q: &IndexSortSortedNumericDocValuesRangeQuery,
    _boost: f32,
    _mode: Mode,
    _top_level: bool,
) -> Result<Option<BoxScorer<'a>>> {
    Err(crate::Error::InvalidQuery("todo".into()))
}

pub(crate) fn point_range<'a>(
    _ctx: &LeafContext<'a>,
    _q: &PointRangeQuery,
    _boost: f32,
    _mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    Err(crate::Error::InvalidQuery("todo".into()))
}

pub(crate) fn point_in_set<'a>(
    _ctx: &LeafContext<'a>,
    _q: &PointInSetQuery,
    _boost: f32,
    _mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    Err(crate::Error::InvalidQuery("todo".into()))
}

pub(crate) fn doc_values_rewrite<'a>(
    _ctx: &LeafContext<'a>,
    _q: &MultiTermQuery,
    _boost: f32,
    _mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    Err(crate::Error::InvalidQuery("todo".into()))
}
