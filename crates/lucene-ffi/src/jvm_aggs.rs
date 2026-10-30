//! The plugin's aggregation trees (read path R5): a tree blob decoded into
//! [`lucene_search::bucket_aggs::AggNode`]s, run over a reader handle, and
//! the shard results encoded for the plugin's `NativeAggregations.Tree` to
//! turn into OpenSearch's `InternalAggregation`s.
//!
//! # The tree blob
//!
//! Little-endian; every length an `i32`, every string UTF-8 behind its
//! length; trailing bytes are an error.
//!
//! ```text
//! blob     := nodes global slices
//! nodes    := count:u8 node*                  (at most MAX_NODES in all)
//! global   := 0 | 1 query:bytes nodes         (the global pass: its query and nodes)
//! node     := METRIC value_kind:u8 source:u8 needs:u8 field
//!           | CARDINALITY card_kind:u8 precision:u8 field
//!           | PERCENTILES value_kind:u8 compression:f64 field   (a t-digest per bucket;
//!                                              compression in (0, MAX_COMPRESSION])
//!           | HDR_PERCENTILES value_kind:u8 field  (each bucket's values, for an
//!                                              HDR `DoubleHistogram`)
//!           | TOP_HITS top_n:i32 sorted:u8 [sort:bytes]  (top_n > 0; sorted 0: by
//!                  score; 1: by the sort blob, `jvm_reader::decode_sort`'s, its
//!                  track-max-score option `track_scores`, without search-after,
//!                  terminate_after, slices or index-sort flags)
//!           | SIGNIFICANT_TERMS flags:u8 [missing:bytes] field nodes
//!                  (flag 1: a numeric field, its longs the keys; 2: a keyword's
//!                   keys answered as terms; 4: a keyword `missing` term follows)
//!           | TERMS field shard_size:i32 order:u8 flags:u8 [missing:bytes]
//!                   [ords:i64 words:i32 word:i64*] nodes
//!                  (flag 1: `min_doc_count: 0`; 2: a `missing` term follows;
//!                   4: `include`/`exclude`, the accepted ordinals' bits)
//!           | NUMERIC_TERMS field value_kind:u8 shard_size:i32 order:u8 nodes
//!           | HISTOGRAM field value_kind:u8 interval:f64 offset:f64 bounds_f64 nodes
//!           | DATE_HISTOGRAM field unit:u8 [interval:i64 when unit = UNIT_INTERVAL]
//!                            zone_ms:i64 offset:i64 bounds_i64 nodes
//!           | RANGE field value_kind:u8 count:i32 (from:f64 to:f64)* nodes
//!           | FILTERS count:i32 query:bytes* other:u8 nodes
//!           | GLOBAL nodes
//!           | NESTED parents:bytes children:bytes nodes  (the parent and child
//!                                              filters' queries; no scores below)
//!           | REVERSE_NESTED parents:bytes nodes
//!           | MISSING value:i64 node        (a numeric node's `missing`, in its
//!                                            column's encoding; doc-values metrics,
//!                                            percentiles)
//!           | COMPOSITE size:i32 count:u8 source* after:u8 [part*] segments:i32 u8*
//!                       nodes                 (top level only; `after` 1: one part
//!                                              per source; a flag per segment: its
//!                                              first source read sorted)
//! source   := field values flags:u8 missing_order:u8
//!             (flag 1: descending, 2: `missing_bucket`; order 0 default, 1 first,
//!              2 last)
//! values   := 0 (keyword) | 1 (long) | 2 value_kind:u8 (double)
//!           | 3 value_kind:u8 interval:f64 (histogram)
//!           | 4 unit:u8 [interval:i64] zone_ms:i64 offset:i64 (date_histogram)
//! part     := 0 (missing) | 1 key:i64 | 2 term:bytes
//! bounds_*  := flags:u8 [min] [max]           (flag 1: a minimum, 2: a maximum)
//! ```
//!
//! Queries are query blobs ([`crate::jvm_reader::decode_query`]); the slices
//! are [`crate::jvm_reader`]'s.
//!
//! # The result
//!
//! Per slice of the main pass, each top node's result in blob order; then,
//! with a global pass, per slice again its nodes' results.
//! A node's result, recursively (`n` the owning bucket count, 1 at the top):
//!
//! ```text
//! METRIC       n:i32 (count:i64 sum delta min max min_of_mins max_of_maxes:f64)*n
//! PERCENTILES  n:i32 nan:u8 (0 | 1 digest)*n   each bucket's `MergingDigest` (0: the
//!                                              bucket collected nothing; `nan` 1: a
//!                                              value was NaN, which Java refuses)
//! digest       merges:i32 last:i32 total:f64 unmerged:f64 temp_used:i32 min:f64
//!              max:f64 (mean:f64 weight:f64)*last (mean:f64 weight:f64)*temp_used
//! TOP_HITS     n:i32 (0 | 1 total:i64 count:i32 (doc:i32 score:f32)*)*n
//!                                              each bucket's top hits by score,
//!                                              global documents (0: the bucket
//!                                              collected nothing)
//! RECORDED     n:i32 (0 | 1 count:i32 value:f64*count)*n  each bucket's values
//!              in collection order (an HDR percentiles'; 0: the bucket collected
//!              nothing)
//! SORTED_HITS  n:i32 (0 | 1 total:i64 max_score:f32 count:i32 doc:i32*count
//!              score:f32*count value:i64*(count*keys) terms:bytes)*n  a sorted
//!              `top_hits`'s: each hit's global document, score (NaN without
//!              `track_scores`) and one value per key, and the keyword keys'
//!              terms (`jvm_reader::encode_terms`)
//! SIGNIFICANT  n:i32 (subset:i64 count:i32 (key docs:i64)*)*n  subs  each bucket's
//!                                              documents, and its keys in
//!                                              first-collected order (key: an
//!                                              i64, or a term:bytes under flag 2)
//! CARDINALITY  n:i32 (len:i32 sketch)*n       each bucket's `HyperLogLogPlusPlus`
//!                                              as `writeTo` writes it (`len` 0:
//!                                              the bucket saw nothing)
//! TERMS        n:i32 (other:i64 count:i32 (docs:i64 term:bytes)*)*n  subs
//! NUMERIC_TERMS n:i32 (other:i64 count:i32 (docs:i64 key:i64)*)*n  subs
//!                                              (a double's key its sortable long)
//! HISTOGRAM    n:i32 (count:i32 (key:f64 docs:i64)*)*n  subs
//! DATE_HIST.   n:i32 (count:i32 (key:i64 docs:i64)*)*n  subs
//! RANGE/FILTERS/GLOBAL  n:i32 width:i32 docs:i64*(n*width)  subs
//! COMPOSITE    count:i32 (docs:i64 value*)*  subs   (one owning bucket; a value
//!                                              per source: 0 missing, 1 long:i64,
//!                                              2 double:f64, 3 term:bytes)
//! ```
//!
//! and `subs` is each sub-node's result, its owning buckets being the
//! node's buckets in the order listed.

use std::collections::HashMap;
use std::sync::Arc;

use lucene_search::aggs::{MinScore, Source, ValueKind};
use lucene_search::bucket_aggs::{
    self, AcceptedOrds, AggNode, AggResult, CardinalityKind, CompositeAfter, CompositeKeyValue,
    CompositeSource, CompositeValues, DateRounding, DateUnit, Globals, MissingOrder, RoundingKind,
    SignificantKey, TermsOrder,
};
use lucene_search::field_norms::FieldNorms;
use lucene_search::multi_segment::OpenSegment;
use lucene_search::BooleanQuery;

use crate::error::{set_last_error, FfiStatus};
use crate::jvm_reader::{self, Cursor};
use crate::query::map_search_error;

const AGG_METRIC: u8 = 0;
const AGG_CARDINALITY: u8 = 1;
const AGG_TERMS: u8 = 2;
const AGG_HISTOGRAM: u8 = 3;
const AGG_DATE_HISTOGRAM: u8 = 4;
const AGG_RANGE: u8 = 5;
const AGG_FILTERS: u8 = 6;
const AGG_GLOBAL: u8 = 7;
const AGG_NUMERIC_TERMS: u8 = 8;
/// A prefix, not a node: the next node's `missing` value.
const AGG_MISSING: u8 = 9;
const AGG_COMPOSITE: u8 = 10;
const AGG_PERCENTILES: u8 = 11;
const AGG_TOP_HITS: u8 = 12;
const AGG_SIGNIFICANT_TERMS: u8 = 13;
const AGG_HDR_PERCENTILES: u8 = 14;
const AGG_NESTED: u8 = 15;
const AGG_REVERSE_NESTED: u8 = 16;
/// The largest t-digest compression a tree may ask for: a digest holds
/// about `12 * compression` doubles.
const MAX_COMPRESSION: f64 = 10_000.0;
/// The most `composite` sources a tree may name.
const MAX_COMPOSITE_SOURCES: usize = 16;

/// `date_histogram` units; [`UNIT_INTERVAL`] is a fixed interval.
const UNITS: [DateUnit; 8] = [
    DateUnit::Week,
    DateUnit::Year,
    DateUnit::Quarter,
    DateUnit::Month,
    DateUnit::Day,
    DateUnit::Hour,
    DateUnit::Minute,
    DateUnit::Second,
];
const UNIT_INTERVAL: u8 = 8;

/// At most this many nodes in one blob, and this deep.
const MAX_NODES: usize = 256;
const MAX_DEPTH: usize = 16;
/// At most this many ranges or filters in one node.
const MAX_WIDTH: usize = 4096;

/// A decoded tree blob: the main nodes, the global pass, the slices.
pub(crate) struct DecodedTree {
    pub(crate) nodes: Vec<AggNode>,
    pub(crate) global: Option<(BooleanQuery, Vec<AggNode>)>,
    pub(crate) slices: Vec<Vec<usize>>,
}

fn bad(msg: String) -> FfiStatus {
    set_last_error(msg);
    FfiStatus::InvalidArgument
}

fn value_kind(k: u8) -> Result<ValueKind, FfiStatus> {
    match k {
        0 => Ok(ValueKind::Long),
        1 => Ok(ValueKind::Double),
        2 => Ok(ValueKind::Float),
        other => Err(bad(format!("aggregation tree: value kind {other}"))),
    }
}

fn field(c: &mut Cursor<'_>) -> Result<String, FfiStatus> {
    Ok(std::str::from_utf8(c.bytes()?)
        .map_err(|_| FfiStatus::InvalidUtf8)?
        .to_string())
}

fn f64_of(c: &mut Cursor<'_>) -> Result<f64, FfiStatus> {
    Ok(f64::from_bits(c.i64()? as u64))
}

/// A `date_histogram`'s rounding: `unit [interval] zone_ms offset`.
fn date_rounding(c: &mut Cursor<'_>) -> Result<DateRounding, FfiStatus> {
    let unit = c.u8()?;
    let kind = match unit {
        UNIT_INTERVAL => {
            let interval = c.i64()?;
            if interval < 1 {
                return Err(bad(format!("aggregation tree: interval {interval}")));
            }
            RoundingKind::Interval(interval)
        }
        u => RoundingKind::Unit(
            *UNITS
                .get(usize::from(u))
                .ok_or_else(|| bad(format!("aggregation tree: date unit {u}")))?,
        ),
    };
    let zone_ms = c.i64()?;
    let offset = c.i64()?;
    Ok(DateRounding {
        kind,
        zone_ms,
        offset,
    })
}

/// A `terms`' effective `shard_size`: at least one.
fn shard_size(c: &mut Cursor<'_>) -> Result<usize, FfiStatus> {
    let n = c.i32()?;
    usize::try_from(n)
        .ok()
        .filter(|&n| n >= 1)
        .ok_or_else(|| bad(format!("aggregation tree: shard_size {n}")))
}

/// A `terms` order: 0 `_count` descending, 1 `_count` ascending, 2 `_key`
/// ascending, 3 `_key` descending.
fn terms_order(c: &mut Cursor<'_>) -> Result<TermsOrder, FfiStatus> {
    Ok(match c.u8()? {
        0 => TermsOrder::CountDesc,
        1 => TermsOrder::CountAsc,
        2 => TermsOrder::KeyAsc,
        3 => TermsOrder::KeyDesc,
        other => return Err(bad(format!("aggregation tree: terms order {other}"))),
    })
}

fn width(c: &mut Cursor<'_>) -> Result<usize, FfiStatus> {
    let n = c.len()?;
    if n == 0 || n > MAX_WIDTH {
        return Err(bad(format!(
            "aggregation tree: {n} buckets, want 1..={MAX_WIDTH}"
        )));
    }
    Ok(n)
}

fn query_of(blob: &[u8]) -> Result<BooleanQuery, FfiStatus> {
    Ok(jvm_reader::boolean_of(&jvm_reader::decode_query(blob)?))
}

fn nodes(c: &mut Cursor<'_>, depth: usize, count: &mut usize) -> Result<Vec<AggNode>, FfiStatus> {
    if depth > MAX_DEPTH {
        return Err(bad(format!("aggregation tree: deeper than {MAX_DEPTH}")));
    }
    let n = usize::from(c.u8()?);
    let mut out = Vec::new();
    for _ in 0..n {
        *count += 1;
        if *count > MAX_NODES {
            return Err(bad(format!(
                "aggregation tree: more than {MAX_NODES} nodes"
            )));
        }
        out.push(node(c, depth, count)?);
    }
    Ok(out)
}

fn node(c: &mut Cursor<'_>, depth: usize, count: &mut usize) -> Result<AggNode, FfiStatus> {
    let kind = c.u8()?;
    if kind != AGG_MISSING {
        return node_of(kind, c, depth, count);
    }
    let value = c.i64()?;
    let mut n = match c.u8()? {
        AGG_MISSING => return Err(bad("aggregation tree: missing twice".to_string())),
        k => node_of(k, c, depth, count)?,
    };
    match &mut n {
        AggNode::Metric {
            missing,
            source: Source::DocValues,
            ..
        }
        | AggNode::Percentiles { missing, .. }
        | AggNode::NumericTerms { missing, .. }
        | AggNode::SignificantTerms {
            missing,
            numeric: true,
            ..
        }
        | AggNode::Histogram { missing, .. }
        | AggNode::DateHistogram { missing, .. }
        | AggNode::Range { missing, .. } => *missing = Some(value),
        _ => {
            return Err(bad(
                "aggregation tree: missing before a node without one".to_string()
            ))
        }
    }
    Ok(n)
}

fn node_of(
    kind: u8,
    c: &mut Cursor<'_>,
    depth: usize,
    count: &mut usize,
) -> Result<AggNode, FfiStatus> {
    Ok(match kind {
        AGG_METRIC => {
            let kind = value_kind(c.u8()?)?;
            let source = match c.u8()? {
                0 => Source::DocValues,
                1 if depth == 0 => Source::PointsMin,
                2 if depth == 0 => Source::PointsMax,
                other => {
                    return Err(bad(format!(
                        "aggregation tree: metric source {other} at depth {depth}"
                    )))
                }
            };
            let needs = c.u8()?;
            if needs == 0 || needs & !lucene_search::aggs::NEED_ALL != 0 {
                return Err(bad(format!("aggregation tree: metric needs {needs:#x}")));
            }
            AggNode::Metric {
                missing: None,
                field: field(c)?,
                kind,
                source,
                needs,
            }
        }
        AGG_PERCENTILES => {
            let kind = value_kind(c.u8()?)?;
            let compression = f64_of(c)?;
            if !(compression > 0.0 && compression <= MAX_COMPRESSION) {
                return Err(bad(format!(
                    "aggregation tree: percentiles compression {compression}"
                )));
            }
            AggNode::Percentiles {
                field: field(c)?,
                missing: None,
                kind,
                compression,
                hdr: false,
            }
        }
        AGG_HDR_PERCENTILES => AggNode::Percentiles {
            kind: value_kind(c.u8()?)?,
            field: field(c)?,
            missing: None,
            compression: 0.0,
            hdr: true,
        },
        AGG_SIGNIFICANT_TERMS => {
            let flags = c.u8()?;
            let (numeric, terms) = (flags & 1 != 0, flags & 2 != 0);
            if flags > 7 || (numeric && flags & 6 != 0) {
                return Err(bad(format!(
                    "aggregation tree: significant_terms flags {flags}"
                )));
            }
            let missing_term = if flags & 4 != 0 {
                Some(c.bytes()?.to_vec())
            } else {
                None
            };
            AggNode::SignificantTerms {
                field: field(c)?,
                numeric,
                terms,
                missing_term,
                missing: None,
                subs: nodes(c, depth + 1, count)?,
            }
        }
        AGG_TOP_HITS => {
            let top_n = c.i32()?;
            let Ok(top_n @ 1..) = usize::try_from(top_n) else {
                return Err(bad(format!("aggregation tree: top_hits size {top_n}")));
            };
            match c.u8()? {
                0 => AggNode::TopHits { top_n },
                1 => {
                    let (sort, after, track_scores, slices, terminate, index_sorted) =
                        jvm_reader::decode_sort(c.bytes()?)?;
                    if after.is_some()
                        || terminate.is_some()
                        || !slices.is_empty()
                        || !index_sorted.is_empty()
                    {
                        return Err(bad(
                            "aggregation tree: a top_hits sort with search options".to_string()
                        ));
                    }
                    AggNode::SortedTopHits {
                        top_n,
                        sort: Arc::new(sort),
                        track_scores,
                    }
                }
                other => return Err(bad(format!("aggregation tree: top_hits sorted {other}"))),
            }
        }
        AGG_CARDINALITY => {
            let kind = match c.u8()? {
                0 => CardinalityKind::Keyword,
                k @ 1..=3 => CardinalityKind::Numeric(value_kind(k - 1)?),
                other => return Err(bad(format!("aggregation tree: cardinality kind {other}"))),
            };
            // `AbstractCardinalityAlgorithm`'s bounds, refused here rather
            // than at the first bucket.
            let precision = match c.u8()? {
                p @ 4..=18 => u32::from(p),
                other => {
                    return Err(bad(format!(
                        "aggregation tree: cardinality precision {other}"
                    )))
                }
            };
            AggNode::Cardinality {
                field: field(c)?,
                kind,
                precision: Some(precision),
            }
        }
        AGG_TERMS => {
            let field = field(c)?;
            let shard_size = shard_size(c)?;
            let order = terms_order(c)?;
            // `min_doc_count: 0`, which the plan sends without sub-aggregations,
            // and the `missing` term.
            let flags = c.u8()?;
            if flags > 7 {
                return Err(bad(format!("aggregation tree: terms flags {flags}")));
            }
            let zero_buckets = flags & 1 != 0;
            let missing = if flags & 2 != 0 {
                Some(c.bytes()?.to_vec())
            } else {
                None
            };
            let accepted = if flags & 4 != 0 {
                let count = u64::try_from(c.i64()?)
                    .map_err(|_| bad("aggregation tree: negative ordinal count".to_string()))?;
                let words = c.len()?;
                if u64::try_from(words).ok() != Some(count.div_ceil(64)) {
                    return Err(bad(format!(
                        "aggregation tree: {words} words for {count} ordinals"
                    )));
                }
                // Grown as read: a length past the blob fails at its end.
                let mut bits = Vec::new();
                for _ in 0..words {
                    bits.push(u64::from_le_bytes(c.i64()?.to_le_bytes()));
                }
                Some(AcceptedOrds {
                    count,
                    bits: bits.into(),
                })
            } else {
                None
            };
            let subs = nodes(c, depth + 1, count)?;
            if zero_buckets && !subs.is_empty() {
                return Err(bad(
                    "aggregation tree: min_doc_count 0 with sub-aggregations".to_string(),
                ));
            }
            AggNode::Terms {
                field,
                missing,
                accepted,
                shard_size,
                order,
                zero_buckets,
                subs,
            }
        }
        AGG_NUMERIC_TERMS => {
            let field = field(c)?;
            let kind = value_kind(c.u8()?)?;
            let shard_size = shard_size(c)?;
            AggNode::NumericTerms {
                missing: None,
                field,
                kind,
                shard_size,
                order: terms_order(c)?,
                subs: nodes(c, depth + 1, count)?,
            }
        }
        AGG_HISTOGRAM => {
            let field = field(c)?;
            let kind = value_kind(c.u8()?)?;
            let interval = f64_of(c)?;
            let offset = f64_of(c)?;
            // `AbstractHistogramAggregator`: not zero or negative.
            if interval <= 0.0 {
                return Err(bad(format!("aggregation tree: interval {interval}")));
            }
            let flags = c.u8()?;
            let min = if flags & 1 != 0 {
                Some(f64_of(c)?)
            } else {
                None
            };
            let max = if flags & 2 != 0 {
                Some(f64_of(c)?)
            } else {
                None
            };
            AggNode::Histogram {
                missing: None,
                field,
                kind,
                interval,
                offset,
                hard_bounds: (min, max),
                subs: nodes(c, depth + 1, count)?,
            }
        }
        AGG_DATE_HISTOGRAM => {
            let field = field(c)?;
            let rounding = date_rounding(c)?;
            let flags = c.u8()?;
            let min = if flags & 1 != 0 { Some(c.i64()?) } else { None };
            let max = if flags & 2 != 0 { Some(c.i64()?) } else { None };
            AggNode::DateHistogram {
                missing: None,
                field,
                rounding,
                hard_bounds: (min, max),
                subs: nodes(c, depth + 1, count)?,
            }
        }
        AGG_COMPOSITE if depth == 0 => {
            let size = usize::try_from(c.i32()?)
                .map_err(|_| bad("aggregation tree: negative composite size".to_string()))?;
            let n = usize::from(c.u8()?);
            if n == 0 || n > MAX_COMPOSITE_SOURCES {
                return Err(bad(format!("aggregation tree: {n} composite sources")));
            }
            let mut sources = Vec::new();
            for _ in 0..n {
                let field = field(c)?;
                let values = match c.u8()? {
                    0 => CompositeValues::Keyword,
                    1 => CompositeValues::Long,
                    2 => CompositeValues::Double(value_kind(c.u8()?)?),
                    3 => {
                        let kind = value_kind(c.u8()?)?;
                        let interval = f64_of(c)?;
                        if !(interval > 0.0 && interval.is_finite()) {
                            return Err(bad(format!(
                                "aggregation tree: histogram interval {interval}"
                            )));
                        }
                        CompositeValues::Histogram { kind, interval }
                    }
                    4 => CompositeValues::Date(date_rounding(c)?),
                    v => return Err(bad(format!("aggregation tree: composite values {v}"))),
                };
                let flags = c.u8()?;
                if flags > 3 {
                    return Err(bad(format!("aggregation tree: composite flags {flags}")));
                }
                let missing_order = match c.u8()? {
                    0 => MissingOrder::Default,
                    1 => MissingOrder::First,
                    2 => MissingOrder::Last,
                    o => return Err(bad(format!("aggregation tree: missing order {o}"))),
                };
                sources.push(CompositeSource {
                    field,
                    values,
                    reverse: flags & 1 != 0,
                    missing_bucket: flags & 2 != 0,
                    missing_order,
                });
            }
            let after = match c.u8()? {
                0 => None,
                1 => {
                    let mut parts = Vec::new();
                    for s in &sources {
                        parts.push(match (c.u8()?, &s.values) {
                            (0, _) if s.missing_bucket => CompositeAfter::Missing,
                            (2, CompositeValues::Keyword) => {
                                CompositeAfter::Term(c.bytes()?.to_vec())
                            }
                            (1, v) if *v != CompositeValues::Keyword => {
                                CompositeAfter::Value(c.i64()?)
                            }
                            (t, _) => {
                                return Err(bad(format!(
                                    "aggregation tree: after part {t} for {}",
                                    s.field
                                )))
                            }
                        });
                    }
                    Some(parts)
                }
                f => return Err(bad(format!("aggregation tree: after flag {f}"))),
            };
            let segments = c.len()?;
            let mut sorted_lead = Vec::new();
            for _ in 0..segments {
                sorted_lead.push(match c.u8()? {
                    0 => false,
                    1 => true,
                    f => return Err(bad(format!("aggregation tree: sorted lead flag {f}"))),
                });
            }
            AggNode::Composite {
                sources,
                size,
                after,
                sorted_lead,
                subs: nodes(c, depth + 1, count)?,
            }
        }
        AGG_RANGE => {
            let field = field(c)?;
            let kind = value_kind(c.u8()?)?;
            let n = width(c)?;
            let mut ranges = Vec::new();
            for _ in 0..n {
                ranges.push((f64_of(c)?, f64_of(c)?));
            }
            AggNode::Range {
                missing: None,
                field,
                kind,
                ranges,
                subs: nodes(c, depth + 1, count)?,
            }
        }
        AGG_FILTERS => {
            let n = width(c)?;
            let mut filters = Vec::new();
            for _ in 0..n {
                filters.push(query_of(c.bytes()?)?);
            }
            let other = match c.u8()? {
                0 => false,
                1 => true,
                o => return Err(bad(format!("aggregation tree: other bucket flag {o}"))),
            };
            AggNode::Filters {
                filters,
                other,
                subs: nodes(c, depth + 1, count)?,
            }
        }
        AGG_GLOBAL => AggNode::Global {
            subs: nodes(c, depth + 1, count)?,
        },
        AGG_NESTED | AGG_REVERSE_NESTED => {
            let parents = query_of(c.bytes()?)?;
            let children = if kind == AGG_NESTED {
                Some(query_of(c.bytes()?)?)
            } else {
                None
            };
            let subs = nodes(c, depth + 1, count)?;
            // A child document has no score of its own (Java hands it its
            // parent's).
            if subs.iter().any(AggNode::needs_scores) {
                return Err(bad(
                    "aggregation tree: a scored aggregation under nested".to_string()
                ));
            }
            match children {
                Some(children) => AggNode::Nested {
                    parents,
                    children,
                    subs,
                },
                None => AggNode::ReverseNested { parents, subs },
            }
        }
        other => return Err(bad(format!("aggregation tree: node kind {other}"))),
    })
}

/// Decodes a tree blob (see the module doc).
pub(crate) fn decode_tree(blob: &[u8]) -> Result<DecodedTree, FfiStatus> {
    let mut c = Cursor::new(blob);
    let mut count = 0;
    let main = nodes(&mut c, 0, &mut count)?;
    if main.iter().any(|n| matches!(n, AggNode::Global { .. })) {
        return Err(bad(
            "aggregation tree: global outside the global pass".to_string()
        ));
    }
    let global = match c.u8()? {
        0 => None,
        1 => {
            let q = query_of(c.bytes()?)?;
            let g = nodes(&mut c, 0, &mut count)?;
            if g.iter().any(|n| !matches!(n, AggNode::Global { .. })) {
                return Err(bad(
                    "aggregation tree: the global pass holds only globals".to_string()
                ));
            }
            Some((q, g))
        }
        f => return Err(bad(format!("aggregation tree: global flag {f}"))),
    };
    let slices = jvm_reader::decode_slices(&mut c, &bad)?;
    if c.pos() != blob.len() {
        return Err(bad(format!(
            "aggregation tree: {} trailing bytes",
            blob.len() - c.pos()
        )));
    }
    if main.is_empty() && global.is_none() {
        return Err(bad("aggregation tree: no aggregations".to_string()));
    }
    Ok(DecodedTree {
        nodes: main,
        global,
        slices,
    })
}

fn put_i32(out: &mut Vec<u8>, v: usize) -> Result<(), FfiStatus> {
    let v = i32::try_from(v).map_err(|_| bad(format!("aggregation result: {v} too many")))?;
    out.extend_from_slice(&v.to_le_bytes());
    Ok(())
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&i64::try_from(v).unwrap_or(i64::MAX).to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) -> Result<(), FfiStatus> {
    put_i32(out, b.len())?;
    out.extend_from_slice(b);
    Ok(())
}

/// Appends one node's result (see the module doc).
pub(crate) fn encode_result(r: &AggResult, out: &mut Vec<u8>) -> Result<(), FfiStatus> {
    match r {
        AggResult::Metric(states) => {
            put_i32(out, states.len())?;
            for s in states {
                put_u64(out, s.count);
                for v in jvm_reader::metric_values(s) {
                    out.extend_from_slice(&v.to_bits().to_le_bytes());
                }
            }
        }
        AggResult::Digests { digests, nan } => {
            put_i32(out, digests.len())?;
            out.push(u8::from(*nan));
            for d in digests {
                let Some(d) = d else {
                    out.push(0);
                    continue;
                };
                out.push(1);
                out.extend_from_slice(&d.merge_count.to_le_bytes());
                put_i32(out, d.last_used_cell)?;
                out.extend_from_slice(&d.total_weight.to_bits().to_le_bytes());
                out.extend_from_slice(&d.unmerged_weight.to_bits().to_le_bytes());
                put_i32(out, d.temp_used)?;
                out.extend_from_slice(&d.min.to_bits().to_le_bytes());
                out.extend_from_slice(&d.max.to_bits().to_le_bytes());
                let merged = d.mean.iter().zip(&d.weight).take(d.last_used_cell);
                let buffered = d.temp_mean.iter().zip(&d.temp_weight).take(d.temp_used);
                for (m, w) in merged.chain(buffered) {
                    out.extend_from_slice(&m.to_bits().to_le_bytes());
                    out.extend_from_slice(&w.to_bits().to_le_bytes());
                }
            }
        }
        AggResult::Significant { buckets, subs } => {
            put_i32(out, buckets.len())?;
            for (subset, listed) in buckets {
                put_u64(out, *subset);
                put_i32(out, listed.len())?;
                for (key, docs) in listed {
                    match key {
                        SignificantKey::Long(k) => out.extend_from_slice(&k.to_le_bytes()),
                        SignificantKey::Term(t) => put_bytes(out, t)?,
                    }
                    put_u64(out, *docs);
                }
            }
            for s in subs {
                encode_result(s, out)?;
            }
        }
        AggResult::TopHits(per) => {
            put_i32(out, per.len())?;
            for bucket in per {
                let Some((total, hits)) = bucket else {
                    out.push(0);
                    continue;
                };
                out.push(1);
                put_u64(out, *total);
                put_i32(out, hits.len())?;
                for h in hits {
                    out.extend_from_slice(&h.doc_id.to_le_bytes());
                    out.extend_from_slice(&h.score.to_bits().to_le_bytes());
                }
            }
        }
        AggResult::Recorded(per) => {
            put_i32(out, per.len())?;
            for bucket in per {
                let Some(values) = bucket else {
                    out.push(0);
                    continue;
                };
                out.push(1);
                put_i32(out, values.len())?;
                for v in values {
                    out.extend_from_slice(&v.to_bits().to_le_bytes());
                }
            }
        }
        AggResult::SortedTopHits(per) => {
            put_i32(out, per.len())?;
            for bucket in per {
                let Some(hits) = bucket else {
                    out.push(0);
                    continue;
                };
                out.push(1);
                put_u64(out, hits.top.total.value);
                out.extend_from_slice(&hits.max_score.to_bits().to_le_bytes());
                let docs = &hits.top.hits;
                put_i32(out, docs.len())?;
                if hits.scores.len() != docs.len() {
                    return Err(bad(format!(
                        "aggregation tree: {} scores for {} sorted hits",
                        hits.scores.len(),
                        docs.len()
                    )));
                }
                for h in docs {
                    out.extend_from_slice(&h.doc.to_le_bytes());
                }
                for s in &hits.scores {
                    out.extend_from_slice(&s.to_bits().to_le_bytes());
                }
                let keys = hits.sort.len();
                for h in docs {
                    if h.values.len() != keys {
                        return Err(bad(format!(
                            "aggregation tree: a sorted hit of {} values for {keys} keys",
                            h.values.len()
                        )));
                    }
                    for v in &h.values {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
                put_bytes(out, &jvm_reader::encode_terms(&hits.sort, docs)?)?;
            }
        }
        AggResult::CardinalitySketch(per) => {
            put_i32(out, per.len())?;
            for sketch in per {
                put_bytes(out, sketch)?;
            }
        }
        // The tree always asks for sketches (`CARDINALITY` carries a precision).
        AggResult::Cardinality(_) => {
            return Err(bad(
                "aggregation tree: a cardinality without a precision".to_string()
            ))
        }
        AggResult::Terms { buckets, subs } => {
            put_i32(out, buckets.len())?;
            for (other, list) in buckets {
                put_u64(out, *other);
                put_i32(out, list.len())?;
                for (term, docs) in list {
                    put_u64(out, *docs);
                    put_bytes(out, term)?;
                }
            }
            encode_subs(subs, out)?;
        }
        AggResult::NumericTerms { buckets, subs } => {
            put_i32(out, buckets.len())?;
            for (other, list) in buckets {
                put_u64(out, *other);
                put_i32(out, list.len())?;
                for (key, docs) in list {
                    put_u64(out, *docs);
                    out.extend_from_slice(&key.to_le_bytes());
                }
            }
            encode_subs(subs, out)?;
        }
        AggResult::Histogram { buckets, subs } => {
            put_i32(out, buckets.len())?;
            for list in buckets {
                put_i32(out, list.len())?;
                for (key, docs) in list {
                    out.extend_from_slice(&key.to_bits().to_le_bytes());
                    put_u64(out, *docs);
                }
            }
            encode_subs(subs, out)?;
        }
        AggResult::DateHistogram { buckets, subs } => {
            put_i32(out, buckets.len())?;
            for list in buckets {
                put_i32(out, list.len())?;
                for (key, docs) in list {
                    out.extend_from_slice(&key.to_le_bytes());
                    put_u64(out, *docs);
                }
            }
            encode_subs(subs, out)?;
        }
        AggResult::Composite { buckets, subs } => {
            put_i32(out, buckets.len())?;
            for (key, docs) in buckets {
                put_u64(out, *docs);
                for v in key {
                    match v {
                        CompositeKeyValue::Missing => out.push(0),
                        CompositeKeyValue::Long(l) => {
                            out.push(1);
                            out.extend_from_slice(&l.to_le_bytes());
                        }
                        CompositeKeyValue::Double(d) => {
                            out.push(2);
                            out.extend_from_slice(&d.to_bits().to_le_bytes());
                        }
                        CompositeKeyValue::Term(t) => {
                            out.push(3);
                            put_bytes(out, t)?;
                        }
                    }
                }
            }
            encode_subs(subs, out)?;
        }
        AggResult::Fixed { width, docs, subs } => {
            put_i32(out, docs.len().checked_div(*width).unwrap_or(0))?;
            put_i32(out, *width)?;
            for &d in docs {
                put_u64(out, d);
            }
            encode_subs(subs, out)?;
        }
    }
    Ok(())
}

fn encode_subs(subs: &[AggResult], out: &mut Vec<u8>) -> Result<(), FfiStatus> {
    for s in subs {
        encode_result(s, out)?;
    }
    Ok(())
}

/// Runs a tree blob over `handle`'s reader for the query blob (a
/// `min_score` in front of it applies to the main pass only, as
/// OpenSearch's `MinimumScoreCollector` wraps only the query's collectors)
/// and encodes the results.
pub(crate) fn aggregate_tree_blobs(
    handle: u64,
    query_blob: &[u8],
    tree_blob: &[u8],
) -> Result<Vec<u8>, FfiStatus> {
    let (query, min_score, sim) = jvm_reader::decode_request(query_blob)?;
    let sim = sim.as_ref();
    let tree = decode_tree(tree_blob)?;
    let h = jvm_reader::lookup(
        handle,
        "ffi_jvm_reader_aggregate_tree: unknown or already-closed handle",
    )?;
    let mut opened = h.reader.open_segments().map_err(|e| {
        set_last_error(format!("opening segment postings: {e}"));
        FfiStatus::Decode
    })?;
    let q = jvm_reader::boolean_of(&query);
    let mut filters = Vec::new();
    for n in tree
        .nodes
        .iter()
        .chain(tree.global.iter().flat_map(|(_, g)| g.iter()))
    {
        n.filter_queries(&mut filters);
    }
    let global_query = tree.global.as_ref().map(|(gq, _)| gq);
    let points = jvm_reader::query_uses_points(&query)
        || filters
            .iter()
            .copied()
            .chain(global_query)
            .any(jvm_reader::boolean_uses_points)
        || tree.nodes.iter().any(AggNode::reads_points)
        // A top-level histogram or range may be counted from the points.
        || tree
            .nodes
            .iter()
            .any(|n| matches!(n, AggNode::DateHistogram { .. } | AggNode::Range { .. }));
    if points {
        opened.open_points().map_err(|e| {
            set_last_error(format!("opening segment points: {e}"));
            FfiStatus::Decode
        })?;
    }
    let segments: Vec<OpenSegment<'_>> = opened
        .as_open_segments()
        .into_iter()
        .zip(&h.live_docs)
        .map(|(mut s, live)| {
            s.live_docs = live.as_ref();
            s.similarity = sim.map(|p| p as &dyn lucene_search::similarities::Similarity);
            s
        })
        .collect();
    let readers = h.reader.segment_readers();
    let all: Vec<usize> = (0..segments.len().min(readers.len())).collect();
    let slices = if tree.slices.is_empty() {
        vec![all]
    } else {
        tree.slices
    };
    let mut fields = Vec::new();
    for n in tree
        .nodes
        .iter()
        .chain(tree.global.iter().flat_map(|(_, g)| g.iter()))
    {
        n.keyword_fields(&mut fields);
    }
    let mut ords = HashMap::new();
    for f in fields {
        let g: Arc<_> = h
            .reader
            .global_ords(f)
            .map_err(crate::query::map_search_error)?;
        ords.insert(f.to_string(), g);
    }
    let globals = Globals::new(&ords);
    // A pass scores its query behind `min_score`, or for a `top_hits`
    // (behind none: `-inf`), with the norms of the fields it scores.
    let main_scores = tree.nodes.iter().any(AggNode::needs_scores);
    let global_scores = tree
        .global
        .as_ref()
        .is_some_and(|(_, g)| g.iter().any(AggNode::needs_scores));
    let mut norm_fields: Vec<String> = Vec::new();
    if min_score.is_some() || main_scores {
        norm_fields.extend(
            crate::query::clause_field_names(&q)
                .into_iter()
                .map(str::to_string),
        );
    }
    if let (true, Some(gq)) = (global_scores, global_query) {
        for f in crate::query::clause_field_names(gq) {
            if !norm_fields.iter().any(|n| n == f) {
                norm_fields.push(f.to_string());
            }
        }
    }
    let owned = h
        .reader
        .field_norms_by_field(&norm_fields)
        .map_err(|error| map_search_error(error.into()))?;
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> =
        owned.iter().map(|m| (!m.is_empty()).then_some(m)).collect();
    let min = min_score
        .or(main_scores.then_some(f32::NEG_INFINITY))
        .map(|min| MinScore { min, norms: &norms });
    let global_min = global_scores.then_some(MinScore {
        min: f32::NEG_INFINITY,
        norms: &norms,
    });
    let mut out = Vec::new();
    if !tree.nodes.is_empty() {
        let sliced = bucket_aggs::aggregate_tree(
            &segments,
            readers,
            &q,
            &tree.nodes,
            &globals,
            &slices,
            min.as_ref(),
        )
        .map_err(crate::query::map_search_error)?;
        for results in &sliced {
            for r in results {
                encode_result(r, &mut out)?;
            }
        }
    }
    if let Some((gq, gnodes)) = &tree.global {
        // The aggregation processor's `postProcess` searches the global
        // query with the same slices (one collector over every segment, in
        // order, without them).
        let results = bucket_aggs::aggregate_tree(
            &segments,
            readers,
            gq,
            gnodes,
            &globals,
            &slices,
            global_min.as_ref(),
        )
        .map_err(crate::query::map_search_error)?;
        for results in &results {
            for r in results {
                encode_result(r, &mut out)?;
            }
        }
    }
    Ok(out)
}

/// The aggregations of a tree blob over the live matches of the query blob:
/// the encoded results (see the module doc) into `out`, their length into
/// `out_len` either way -- a too-small `out` is
/// [`FfiStatus::BufferTooSmall`], so a caller can size a second call.
///
/// # Safety
/// `query`/`tree` must be valid for `query_len`/`tree_len` bytes, `out` for
/// `cap` bytes (null when 0), `out_len` writable.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_reader_aggregate_tree(
    handle: u64,
    query: *const u8,
    query_len: usize,
    tree: *const u8,
    tree_len: usize,
    out: *mut u8,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    crate::error::guard(|| {
        if out_len.is_null() || (out.is_null() && cap > 0) {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let q = unsafe { crate::raw::bytes_from_raw(query, query_len)? };
        // SAFETY: caller contract.
        let t = unsafe { crate::raw::bytes_from_raw(tree, tree_len)? };
        let encoded = aggregate_tree_blobs(handle, q, t)?;
        // SAFETY: `out_len` is non-null and writable (caller contract).
        unsafe { *out_len = encoded.len() };
        if encoded.len() > cap {
            set_last_error(format!(
                "aggregation result: {} bytes, buffer holds {cap}",
                encoded.len()
            ));
            return Err(FfiStatus::BufferTooSmall);
        }
        if !encoded.is_empty() {
            // SAFETY: `out` is valid for `cap >= encoded.len()` bytes.
            unsafe { std::ptr::copy_nonoverlapping(encoded.as_ptr(), out, encoded.len()) };
        }
        Ok(())
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::jvm_reader::tests::{open, term_blob};

    fn string(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as i32).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    fn f64s(b: &mut Vec<u8>, v: &[f64]) {
        for x in v {
            b.extend_from_slice(&x.to_bits().to_le_bytes());
        }
    }

    fn metric(b: &mut Vec<u8>, kind: u8, source: u8, field: &str) {
        b.extend_from_slice(&[AGG_METRIC, kind, source, lucene_search::aggs::NEED_ALL]);
        string(b, field);
    }

    /// A tree over the fixture's `n` (a long) and `body`: every node kind.
    pub(crate) fn every_kind(global: bool) -> Vec<u8> {
        let mut b = vec![6u8];
        // A top-level metric reading the points bound.
        metric(&mut b, 0, 1, "n");
        b.push(AGG_CARDINALITY);
        b.push(1);
        b.push(14);
        string(&mut b, "n");
        // histogram(n, 1e9) > [metric, terms(body) > []]
        b.push(AGG_HISTOGRAM);
        string(&mut b, "n");
        b.push(0);
        f64s(&mut b, &[1e9, 0.0]);
        b.push(3);
        f64s(&mut b, &[-1e12, 1e12]);
        b.push(2);
        metric(&mut b, 0, 0, "n");
        b.push(AGG_TERMS);
        string(&mut b, "body");
        b.extend_from_slice(&5i32.to_le_bytes());
        b.push(0);
        b.push(0);
        b.push(0);
        // date_histogram(n, unit day, then interval) > []
        for unit in [4u8, UNIT_INTERVAL] {
            b.push(AGG_DATE_HISTOGRAM);
            string(&mut b, "n");
            b.push(unit);
            if unit == UNIT_INTERVAL {
                b.extend_from_slice(&3i64.to_le_bytes());
            }
            b.extend_from_slice(&0i64.to_le_bytes());
            b.extend_from_slice(&0i64.to_le_bytes());
            b.push(if unit == UNIT_INTERVAL { 3 } else { 0 });
            if unit == UNIT_INTERVAL {
                b.extend_from_slice(&i64::MIN.to_le_bytes());
                b.extend_from_slice(&i64::MAX.to_le_bytes());
            }
            b.push(0);
        }
        // range(n, [-inf, 3), [3, inf)) > filters(fox, other) > metric
        b.push(AGG_RANGE);
        string(&mut b, "n");
        b.push(0);
        b.extend_from_slice(&2i32.to_le_bytes());
        f64s(&mut b, &[f64::NEG_INFINITY, 3.0, 3.0, f64::INFINITY]);
        b.push(1);
        b.push(AGG_FILTERS);
        b.extend_from_slice(&1i32.to_le_bytes());
        let fox = term_blob("body", "fox");
        b.extend_from_slice(&(fox.len() as i32).to_le_bytes());
        b.extend_from_slice(&fox);
        b.push(1);
        b.push(1);
        metric(&mut b, 1, 0, "n");
        if global {
            b.push(1);
            let all = term_blob("body", "the");
            b.extend_from_slice(&(all.len() as i32).to_le_bytes());
            b.extend_from_slice(&all);
            b.push(1);
            b.push(AGG_GLOBAL);
            b.push(1);
            metric(&mut b, 0, 0, "n");
        } else {
            b.push(0);
        }
        b.extend_from_slice(&0i32.to_le_bytes());
        b
    }

    #[test]
    fn every_node_kind_decodes() {
        let t = decode_tree(&every_kind(true)).unwrap();
        assert_eq!(t.nodes.len(), 6);
        assert!(matches!(
            t.nodes[0],
            AggNode::Metric {
                source: Source::PointsMin,
                ..
            }
        ));
        assert!(matches!(
            t.nodes[1],
            AggNode::Cardinality {
                kind: CardinalityKind::Numeric(ValueKind::Long),
                precision: Some(14),
                ..
            }
        ));
        let AggNode::Histogram {
            interval,
            hard_bounds,
            subs,
            ..
        } = &t.nodes[2]
        else {
            panic!()
        };
        assert_eq!((*interval, *hard_bounds), (1e9, (Some(-1e12), Some(1e12))));
        assert_eq!(subs.len(), 2);
        assert!(matches!(
            t.nodes[3],
            AggNode::DateHistogram {
                rounding: DateRounding {
                    kind: RoundingKind::Unit(DateUnit::Day),
                    ..
                },
                ..
            }
        ));
        assert!(matches!(
            t.nodes[4],
            AggNode::DateHistogram {
                rounding: DateRounding {
                    kind: RoundingKind::Interval(3),
                    ..
                },
                hard_bounds: (Some(i64::MIN), Some(i64::MAX)),
                ..
            }
        ));
        let AggNode::Range { ranges, subs, .. } = &t.nodes[5] else {
            panic!()
        };
        assert_eq!(ranges.len(), 2);
        assert!(
            matches!(&subs[0], AggNode::Filters { other: true, filters, .. } if filters.len() == 1)
        );
        assert!(t.global.is_some());
        assert!(t.slices.is_empty());
    }

    /// `NUMERIC_TERMS field kind shard_size subs`, decoded and run: the
    /// result is `n other count (docs key)*`, keys ascending, docs summing to
    /// the matches with a value.
    #[test]
    fn numeric_terms_decode_and_run() {
        let tree = |kind: u8, shard_size: i32| {
            let mut b = vec![1u8, AGG_NUMERIC_TERMS];
            string(&mut b, "n");
            b.push(kind);
            b.extend_from_slice(&shard_size.to_le_bytes());
            b.push(2);
            b.push(1);
            metric(&mut b, 0, 0, "n");
            b.push(0);
            b.extend_from_slice(&0i32.to_le_bytes());
            b
        };
        // A `missing` prefix sets the node's value.
        let mut with_missing = vec![1u8, AGG_MISSING];
        with_missing.extend_from_slice(&(-9i64).to_le_bytes());
        with_missing.extend_from_slice(&tree(0, 2)[1..]);
        let t = decode_tree(&with_missing).unwrap();
        assert!(matches!(
            &t.nodes[0],
            AggNode::NumericTerms {
                missing: Some(-9),
                ..
            }
        ));
        let t = decode_tree(&tree(0, 2)).unwrap();
        let AggNode::NumericTerms {
            kind,
            shard_size,
            order,
            subs,
            ..
        } = &t.nodes[0]
        else {
            panic!()
        };
        assert_eq!(
            (*kind, *shard_size, *order, subs.len()),
            (ValueKind::Long, 2, TermsOrder::KeyAsc, 1)
        );
        assert!(decode_tree(&tree(9, 2)).is_err(), "value kind 9");
        assert!(decode_tree(&tree(0, 0)).is_err(), "shard_size 0");
        let full = tree(0, 2);
        for n in 1..full.len() {
            assert!(decode_tree(&full[..n]).is_err(), "truncated at {n}");
        }

        let h = open();
        let fox = term_blob("body", "fox");
        let out = aggregate_tree_blobs(h, &fox, &tree(0, 2)).unwrap();
        let mut at = 0;
        assert_eq!(i32_at(&out, &mut at), 1, "one owning bucket");
        let other = i64_at(&out, &mut at);
        let m = i32_at(&out, &mut at);
        assert!((0..=2).contains(&m));
        let mut keys = Vec::new();
        for _ in 0..m {
            assert!(i64_at(&out, &mut at) >= 1);
            keys.push(i64_at(&out, &mut at));
        }
        assert!(other >= 0);
        assert!(keys.windows(2).all(|w| w[0] < w[1]), "{keys:?}");
        crate::jvm_reader::ffi_close_jvm_reader(h);
    }

    #[test]
    fn malformed_trees_are_refused() {
        let bad = |b: &[u8]| decode_tree(b).err();
        let invalid = Some(FfiStatus::InvalidArgument);
        // Nothing at all, or no aggregation.
        assert_eq!(bad(&[]), invalid);
        assert_eq!(bad(&[0, 0, 0, 0, 0, 0]), invalid);
        // Unknown kinds and codes.
        let tail = [0u8, 0, 0, 0, 0];
        let with = |node: &[u8]| {
            let mut b = vec![1u8];
            b.extend_from_slice(node);
            b.extend_from_slice(&tail);
            b
        };
        assert_eq!(bad(&with(&[99])), invalid);
        assert_eq!(bad(&with(&[AGG_METRIC, 9, 0, 1, 0, 0, 0, 0])), invalid);
        assert_eq!(bad(&with(&[AGG_METRIC, 0, 7, 1, 0, 0, 0, 0])), invalid);
        assert_eq!(
            bad(&with(&[AGG_METRIC, 0, 0, 0, 0, 0, 0, 0])),
            invalid,
            "no needs"
        );
        assert_eq!(
            bad(&with(&[AGG_METRIC, 0, 0, 0x80, 0, 0, 0, 0])),
            invalid,
            "unknown needs"
        );
        assert_eq!(bad(&with(&[AGG_CARDINALITY, 9, 14, 0, 0, 0, 0])), invalid);
        assert_eq!(
            bad(&with(&[AGG_CARDINALITY, 0, 3, 0, 0, 0, 0])),
            invalid,
            "precision"
        );
        assert_eq!(
            bad(&with(&[AGG_CARDINALITY, 0, 19, 0, 0, 0, 0])),
            invalid,
            "precision"
        );
        // A points source below the top, a zero interval, zero ranges.
        let mut deep = vec![AGG_GLOBAL, 1];
        metric(&mut deep, 0, 1, "n");
        let mut g = vec![0u8, 1];
        g.extend_from_slice(&(term_blob("b", "x").len() as i32).to_le_bytes());
        g.extend_from_slice(&term_blob("b", "x"));
        g.push(1);
        g.extend_from_slice(&deep);
        g.extend_from_slice(&0i32.to_le_bytes());
        assert_eq!(bad(&g), invalid);
        let mut h = vec![AGG_HISTOGRAM];
        string(&mut h, "n");
        h.push(0);
        f64s(&mut h, &[0.0, 0.0]);
        h.extend_from_slice(&[0, 0]);
        assert_eq!(bad(&with(&h)), invalid);
        let mut r = vec![AGG_RANGE];
        string(&mut r, "n");
        r.push(0);
        r.extend_from_slice(&0i32.to_le_bytes());
        assert_eq!(bad(&with(&r)), invalid);
        let mut d = vec![AGG_DATE_HISTOGRAM];
        string(&mut d, "n");
        d.push(UNIT_INTERVAL);
        d.extend_from_slice(&0i64.to_le_bytes());
        assert_eq!(bad(&with(&d)), invalid);
        let mut d = vec![AGG_DATE_HISTOGRAM];
        string(&mut d, "n");
        d.push(42);
        assert_eq!(bad(&with(&d)), invalid);
        let mut t = vec![AGG_TERMS];
        string(&mut t, "k");
        t.extend_from_slice(&0i32.to_le_bytes());
        assert_eq!(bad(&with(&t)), invalid);
        // `missing` twice, before a keyword `terms`, before a points metric.
        let mut m = vec![AGG_MISSING];
        m.extend_from_slice(&1i64.to_le_bytes());
        let twice = [
            m.clone(),
            m.clone(),
            vec![AGG_METRIC, 0, 0, 1, 1, 0, 0, 0, b'n'],
        ]
        .concat();
        assert_eq!(bad(&with(&twice)), invalid);
        let mut kw = m.clone();
        kw.push(AGG_TERMS);
        string(&mut kw, "k");
        kw.extend_from_slice(&1i32.to_le_bytes());
        kw.extend_from_slice(&[0, 0]);
        assert_eq!(bad(&with(&kw)), invalid);
        let points = [m.clone(), vec![AGG_METRIC, 0, 1, 1, 1, 0, 0, 0, b'n']].concat();
        assert_eq!(bad(&with(&points)), invalid);
        let doc_values = [m, vec![AGG_METRIC, 0, 0, 1, 1, 0, 0, 0, b'n']].concat();
        assert!(decode_tree(&with(&doc_values)).is_ok());
        // Terms flags past 3, and min_doc_count 0 over a sub-aggregation.
        let mut t = vec![AGG_TERMS];
        string(&mut t, "k");
        t.extend_from_slice(&1i32.to_le_bytes());
        t.extend_from_slice(&[0, 8, 0]);
        assert_eq!(bad(&with(&t)), invalid);
        // A missing term: decoded with min_doc_count 0.
        let mut t = vec![AGG_TERMS];
        string(&mut t, "k");
        t.extend_from_slice(&1i32.to_le_bytes());
        t.extend_from_slice(&[0, 3]);
        string(&mut t, "none");
        t.push(0);
        let tree = decode_tree(&with(&t)).unwrap();
        assert!(matches!(
            &tree.nodes[0],
            AggNode::Terms { missing: Some(m), zero_buckets: true, .. } if m == b"none"
        ));
        // Accepted ordinals: 65 of them in two words; one word too few.
        let accepted = |words: i32| {
            let mut t = vec![AGG_TERMS];
            string(&mut t, "k");
            t.extend_from_slice(&1i32.to_le_bytes());
            t.extend_from_slice(&[0, 4]);
            t.extend_from_slice(&65i64.to_le_bytes());
            t.extend_from_slice(&words.to_le_bytes());
            for w in 0..words {
                t.extend_from_slice(&i64::from(w + 5).to_le_bytes());
            }
            t.push(0);
            t
        };
        let tree = decode_tree(&with(&accepted(2))).unwrap();
        assert!(matches!(
            &tree.nodes[0],
            AggNode::Terms { accepted: Some(a), .. } if a.count == 65 && a.bits[..] == [5, 6]
        ));
        assert_eq!(bad(&with(&accepted(1))), invalid);
        let mut t = vec![AGG_TERMS];
        string(&mut t, "k");
        t.extend_from_slice(&1i32.to_le_bytes());
        t.extend_from_slice(&[0, 1, 1, AGG_METRIC, 0, 0, 1, 1, 0, 0, 0, b'n']);
        assert_eq!(bad(&with(&t)), invalid);
        // An order past `_key` descending.
        let mut t = vec![AGG_TERMS];
        string(&mut t, "k");
        t.extend_from_slice(&1i32.to_le_bytes());
        t.push(4);
        assert_eq!(bad(&with(&t)), invalid);
        let mut f = vec![AGG_FILTERS];
        f.extend_from_slice(&1i32.to_le_bytes());
        let q = term_blob("b", "x");
        f.extend_from_slice(&(q.len() as i32).to_le_bytes());
        f.extend_from_slice(&q);
        f.push(2);
        assert_eq!(bad(&with(&f)), invalid);
        // A global in the main pass; a non-global in the global pass.
        assert_eq!(bad(&with(&[AGG_GLOBAL, 0])), invalid);
        let mut ng = vec![0u8, 1];
        ng.extend_from_slice(&(q.len() as i32).to_le_bytes());
        ng.extend_from_slice(&q);
        ng.push(1);
        metric(&mut ng, 0, 0, "n");
        ng.extend_from_slice(&0i32.to_le_bytes());
        assert_eq!(bad(&ng), invalid);
        assert_eq!(bad(&[0, 7]), invalid, "a bad global flag");
        // Too deep, trailing bytes.
        let mut nest = Vec::new();
        for _ in 0..=MAX_DEPTH + 1 {
            nest.extend_from_slice(&[1, AGG_FILTERS, 1, 0, 0, 0]);
            nest.extend_from_slice(&(q.len() as i32).to_le_bytes());
            nest.extend_from_slice(&q);
            nest.push(0);
        }
        nest.push(0);
        assert_eq!(bad(&nest), invalid);
        let mut trailing = every_kind(false);
        trailing.push(0);
        assert_eq!(bad(&trailing), invalid);
        // Too many nodes: two filters of 255 metrics each.
        let mut wide = vec![2u8];
        for _ in 0..2 {
            wide.push(AGG_FILTERS);
            wide.extend_from_slice(&1i32.to_le_bytes());
            wide.extend_from_slice(&(q.len() as i32).to_le_bytes());
            wide.extend_from_slice(&q);
            wide.push(0);
            wide.push(255);
            for _ in 0..255 {
                metric(&mut wide, 0, 0, "n");
            }
        }
        wide.extend_from_slice(&tail);
        assert_eq!(bad(&wide), invalid);
    }

    #[test]
    fn percentiles_decode_and_encode() {
        let with = |node: &[u8]| {
            let mut b = vec![1u8];
            b.extend_from_slice(node);
            b.extend_from_slice(&[0u8, 0, 0, 0, 0]);
            b
        };
        let percentiles = |compression: f64| {
            let mut t = vec![AGG_PERCENTILES, 1];
            f64s(&mut t, &[compression]);
            string(&mut t, "d");
            t
        };
        let tree = decode_tree(&with(&percentiles(100.0))).unwrap();
        assert!(matches!(
            &tree.nodes[0],
            AggNode::Percentiles { field, kind: ValueKind::Double, compression, missing: None, hdr: false }
                if field == "d" && *compression == 100.0
        ));
        // `missing` in front of it.
        let mut m = vec![AGG_MISSING];
        m.extend_from_slice(&7i64.to_le_bytes());
        m.extend_from_slice(&percentiles(50.0));
        let tree = decode_tree(&with(&m)).unwrap();
        assert!(matches!(
            &tree.nodes[0],
            AggNode::Percentiles {
                missing: Some(7),
                ..
            }
        ));
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, MAX_COMPRESSION * 2.0] {
            assert_eq!(
                decode_tree(&with(&percentiles(bad))).err(),
                Some(FfiStatus::InvalidArgument),
                "{bad}"
            );
        }
        // The result: a bucket that collected nothing, then a digest.
        let mut d = lucene_search::tdigest::MergingDigest::new(100.0);
        for x in [3.0, 1.0, 2.0] {
            d.add(x).unwrap();
        }
        let mut out = Vec::new();
        encode_result(
            &AggResult::Digests {
                digests: vec![None, Some(d)],
                nan: false,
            },
            &mut out,
        )
        .unwrap();
        let mut want = 2i32.to_le_bytes().to_vec();
        want.extend_from_slice(&[0, 0, 1]);
        want.extend_from_slice(&0i32.to_le_bytes());
        want.extend_from_slice(&0i32.to_le_bytes());
        f64s(&mut want, &[0.0, 3.0]);
        want.extend_from_slice(&3i32.to_le_bytes());
        f64s(&mut want, &[1.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.0, 1.0]);
        assert_eq!(out, want);
    }

    #[test]
    fn hdr_percentiles_decode_and_encode() {
        let mut b = vec![1u8, AGG_MISSING];
        b.extend_from_slice(&7i64.to_le_bytes());
        b.extend_from_slice(&[AGG_HDR_PERCENTILES, 0]);
        string(&mut b, "n");
        b.extend_from_slice(&[0u8, 0, 0, 0, 0]);
        let tree = decode_tree(&b).unwrap();
        assert!(matches!(
            &tree.nodes[0],
            AggNode::Percentiles { field, kind: ValueKind::Long, missing: Some(7), hdr: true, .. }
                if field == "n"
        ));
        let mut out = Vec::new();
        encode_result(
            &AggResult::Recorded(vec![Some(vec![2.5, -1.0]), None, Some(Vec::new())]),
            &mut out,
        )
        .unwrap();
        let mut want = 3i32.to_le_bytes().to_vec();
        want.push(1);
        want.extend_from_slice(&2i32.to_le_bytes());
        f64s(&mut want, &[2.5, -1.0]);
        want.extend_from_slice(&[0, 1]);
        want.extend_from_slice(&0i32.to_le_bytes());
        assert_eq!(out, want);
    }

    #[test]
    fn top_hits_decode_and_encode() {
        let with = |node: &[u8]| {
            let mut b = vec![1u8];
            b.extend_from_slice(node);
            b.extend_from_slice(&[0u8, 0, 0, 0, 0]);
            b
        };
        let top_hits = |n: i32| {
            let mut t = vec![AGG_TOP_HITS];
            t.extend_from_slice(&n.to_le_bytes());
            t.push(0);
            t
        };
        let tree = decode_tree(&with(&top_hits(3))).unwrap();
        assert!(matches!(tree.nodes[0], AggNode::TopHits { top_n: 3 }));
        assert!(tree.nodes[0].needs_scores());
        for bad in [0, -1, i32::MIN] {
            assert_eq!(
                decode_tree(&with(&top_hits(bad))).err(),
                Some(FfiStatus::InvalidArgument),
                "{bad}"
            );
        }
        let hit = |doc_id, score| lucene_search::collector::ScoreDoc { doc_id, score };
        let mut out = Vec::new();
        encode_result(
            &AggResult::TopHits(vec![None, Some((5, vec![hit(7, 2.5), hit(1, 1.0)]))]),
            &mut out,
        )
        .unwrap();
        let mut want = 2i32.to_le_bytes().to_vec();
        want.extend_from_slice(&[0, 1]);
        want.extend_from_slice(&5u64.to_le_bytes());
        want.extend_from_slice(&2i32.to_le_bytes());
        for (d, s) in [(7i32, 2.5f32), (1, 1.0)] {
            want.extend_from_slice(&d.to_le_bytes());
            want.extend_from_slice(&s.to_bits().to_le_bytes());
        }
        assert_eq!(out, want);
    }

    #[test]
    fn sorted_top_hits_decode_and_encode() {
        use lucene_search::bucket_aggs::SortedHits;
        use lucene_search::top_field::{FieldDoc, SortField, SortType, TopFieldDocs};
        // A sort blob: `kw` ascending (missing last), then `n` (long) descending.
        let sort_blob = |after: bool, options: u8| {
            let mut b = vec![2u8, jvm_reader::SORT_STRING, 0];
            b.extend_from_slice(&2i32.to_le_bytes());
            b.extend_from_slice(b"kw");
            b.extend_from_slice(&1i64.to_le_bytes());
            b.extend_from_slice(&[jvm_reader::SORT_LONG, 1]);
            b.extend_from_slice(&1i32.to_le_bytes());
            b.push(b'n');
            b.extend_from_slice(&0i64.to_le_bytes());
            if after {
                b.push(1);
                b.extend_from_slice(&0i32.to_le_bytes());
                b.push(0);
                b.extend_from_slice(&0i64.to_le_bytes());
            } else {
                b.push(0);
            }
            b.push(options);
            b.extend_from_slice(&0i32.to_le_bytes());
            b
        };
        let tree = |blob: &[u8]| {
            let mut b = vec![1u8, AGG_TOP_HITS];
            b.extend_from_slice(&3i32.to_le_bytes());
            b.push(1);
            b.extend_from_slice(&i32::try_from(blob.len()).unwrap().to_le_bytes());
            b.extend_from_slice(blob);
            b.extend_from_slice(&[0u8, 0, 0, 0, 0]);
            decode_tree(&b)
        };
        let got = tree(&sort_blob(false, 1)).unwrap();
        let AggNode::SortedTopHits {
            top_n: 3,
            sort,
            track_scores: true,
        } = &got.nodes[0]
        else {
            panic!("{:?}", got.nodes[0])
        };
        assert_eq!(sort.len(), 2);
        assert_eq!(
            (sort[0].ty, sort[1].ty, sort[1].reverse),
            (SortType::String, SortType::Long, true)
        );
        assert!(got.nodes[0].needs_scores() && got.nodes[0].reads_points());
        assert!(!tree(&sort_blob(false, 0)).unwrap().nodes[0].needs_scores());
        // Search-after and terminate_after are a search's, not a bucket's.
        for bad in [sort_blob(true, 0), sort_blob(false, 2)] {
            assert_eq!(tree(&bad).err(), Some(FfiStatus::InvalidArgument));
        }
        let sort = Arc::new(vec![
            SortField::string("kw", false),
            SortField::numeric("n", SortType::Long, true),
        ]);
        let hits = SortedHits {
            sort: Arc::clone(&sort),
            top: TopFieldDocs {
                hits: vec![
                    FieldDoc {
                        doc: 4,
                        values: vec![0, 9],
                        terms: vec![Some(b"a".to_vec()), None],
                    },
                    FieldDoc {
                        doc: 1,
                        values: vec![0, -2],
                        terms: vec![None, None],
                    },
                ],
                total: lucene_search::collector::TotalHits {
                    value: 6,
                    relation: lucene_search::collector::TotalHitsRelation::EqualTo,
                },
                max_score: f32::NAN,
            },
            scores: vec![1.5, 0.5],
            max_score: 2.0,
        };
        let mut out = Vec::new();
        encode_result(&AggResult::SortedTopHits(vec![Some(hits), None]), &mut out).unwrap();
        let mut want = 2i32.to_le_bytes().to_vec();
        want.push(1);
        want.extend_from_slice(&6u64.to_le_bytes());
        want.extend_from_slice(&2.0f32.to_bits().to_le_bytes());
        want.extend_from_slice(&2i32.to_le_bytes());
        for d in [4i32, 1] {
            want.extend_from_slice(&d.to_le_bytes());
        }
        for s in [1.5f32, 0.5] {
            want.extend_from_slice(&s.to_bits().to_le_bytes());
        }
        for v in [0i64, 9, 0, -2] {
            want.extend_from_slice(&v.to_le_bytes());
        }
        let mut terms = 1i32.to_le_bytes().to_vec();
        terms.push(b'a');
        terms.extend_from_slice(&(-1i32).to_le_bytes());
        want.extend_from_slice(&i32::try_from(terms.len()).unwrap().to_le_bytes());
        want.extend_from_slice(&terms);
        want.push(0);
        assert_eq!(out, want);
    }

    #[test]
    fn significant_terms_decode_and_encode() {
        // A keyword's, as terms, with a `missing` term and no sub-aggregation.
        let mut b = vec![1u8, AGG_SIGNIFICANT_TERMS, 6];
        string(&mut b, "none");
        string(&mut b, "kw");
        b.push(0);
        b.extend_from_slice(&[0u8, 0, 0, 0, 0]);
        let tree = decode_tree(&b).unwrap();
        assert!(matches!(
            &tree.nodes[0],
            AggNode::SignificantTerms { field, numeric: false, terms: true, missing_term: Some(m), missing: None, subs }
                if field == "kw" && m == b"none" && subs.is_empty()
        ));
        // A numeric one behind `missing`.
        let mut b = vec![1u8, AGG_MISSING];
        b.extend_from_slice(&7i64.to_le_bytes());
        b.extend_from_slice(&[AGG_SIGNIFICANT_TERMS, 1]);
        string(&mut b, "n");
        b.push(0);
        b.extend_from_slice(&[0u8, 0, 0, 0, 0]);
        let tree = decode_tree(&b).unwrap();
        assert!(matches!(
            &tree.nodes[0],
            AggNode::SignificantTerms {
                numeric: true,
                missing: Some(7),
                ..
            }
        ));
        // Flags a numeric field cannot have.
        for flags in [3u8, 5, 8] {
            let mut b = vec![1u8, AGG_SIGNIFICANT_TERMS, flags];
            string(&mut b, "x");
            string(&mut b, "n");
            b.push(0);
            b.extend_from_slice(&[0u8, 0, 0, 0, 0]);
            assert_eq!(
                decode_tree(&b).err(),
                Some(FfiStatus::InvalidArgument),
                "{flags}"
            );
        }
        let mut out = Vec::new();
        encode_result(
            &AggResult::Significant {
                buckets: vec![
                    (0, vec![]),
                    (
                        4,
                        vec![
                            (SignificantKey::Long(9), 3),
                            (SignificantKey::Term(b"ab".to_vec()), 1),
                        ],
                    ),
                ],
                subs: vec![],
            },
            &mut out,
        )
        .unwrap();
        let mut want = 2i32.to_le_bytes().to_vec();
        want.extend_from_slice(&0u64.to_le_bytes());
        want.extend_from_slice(&0i32.to_le_bytes());
        want.extend_from_slice(&4u64.to_le_bytes());
        want.extend_from_slice(&2i32.to_le_bytes());
        for v in [9i64, 3] {
            want.extend_from_slice(&v.to_le_bytes());
        }
        want.extend_from_slice(&2i32.to_le_bytes());
        want.extend_from_slice(b"ab");
        want.extend_from_slice(&1i64.to_le_bytes());
        assert_eq!(out, want);
    }

    #[test]
    fn composite_trees_decode() {
        let tail = [0u8, 0, 0, 0, 0];
        let with = |node: &[u8]| {
            let mut b = vec![1u8];
            b.extend_from_slice(node);
            b.extend_from_slice(&tail);
            b
        };
        let invalid = Some(FfiStatus::InvalidArgument);
        // composite(size 10, [terms k desc missing_bucket first, histogram n
        // 5.0]) after [missing, 15] over two segments > [metric(n)].
        let composite = |sources: u8, after: &[u8], flags: u8| {
            let mut t = vec![AGG_COMPOSITE];
            t.extend_from_slice(&10i32.to_le_bytes());
            t.push(sources);
            string(&mut t, "k");
            t.extend_from_slice(&[0, flags, 1]);
            string(&mut t, "n");
            t.extend_from_slice(&[3, 0]);
            f64s(&mut t, &[5.0]);
            t.extend_from_slice(&[0, 0]);
            t.extend_from_slice(after);
            t.extend_from_slice(&2i32.to_le_bytes());
            t.extend_from_slice(&[1, 0, 1]);
            metric(&mut t, 0, 0, "n");
            t
        };
        let mut after = vec![1u8, 0, 1];
        after.extend_from_slice(&15i64.to_le_bytes());
        let tree = decode_tree(&with(&composite(2, &after, 3))).unwrap();
        let AggNode::Composite {
            sources,
            size,
            after: Some(after_parts),
            sorted_lead,
            subs,
        } = &tree.nodes[0]
        else {
            panic!("{:?}", tree.nodes[0])
        };
        assert_eq!(*size, 10);
        assert_eq!(sources.len(), 2);
        assert!(sources[0].reverse && sources[0].missing_bucket);
        assert_eq!(sources[0].missing_order, MissingOrder::First);
        assert!(matches!(
            sources[1].values,
            CompositeValues::Histogram { interval, .. } if interval == 5.0
        ));
        assert_eq!(
            after_parts[..],
            [CompositeAfter::Missing, CompositeAfter::Value(15)]
        );
        assert_eq!(sorted_lead[..], [true, false]);
        assert_eq!(subs.len(), 1);
        // No `after`.
        assert!(decode_tree(&with(&composite(2, &[0], 0))).is_ok());
        // A missing `after` part without `missing_bucket`, a term for a
        // numeric source, unknown flags, no sources, a source count past the
        // ones written.
        assert_eq!(decode_tree(&with(&composite(2, &after, 1))).err(), invalid);
        let mut term = vec![1u8, 2];
        string(&mut term, "a");
        term.push(2);
        string(&mut term, "b");
        assert_eq!(decode_tree(&with(&composite(2, &term, 0))).err(), invalid);
        assert_eq!(decode_tree(&with(&composite(2, &[0], 4))).err(), invalid);
        assert_eq!(decode_tree(&with(&composite(0, &[0], 0))).err(), invalid);
        assert_eq!(decode_tree(&with(&composite(3, &[0], 0))).err(), invalid);
        // Only at the top level.
        let mut nested = vec![AGG_GLOBAL, 1];
        nested.extend_from_slice(&composite(2, &[0], 0));
        assert_eq!(decode_tree(&with(&nested)).err(), invalid);
    }

    fn i32_at(b: &[u8], at: &mut usize) -> i32 {
        let v = i32::from_le_bytes(b[*at..*at + 4].try_into().unwrap());
        *at += 4;
        v
    }

    fn i64_at(b: &[u8], at: &mut usize) -> i64 {
        let v = i64::from_le_bytes(b[*at..*at + 8].try_into().unwrap());
        *at += 8;
        v
    }

    #[test]
    fn a_tree_runs_over_a_handle_and_encodes_its_results() {
        let h = open();
        let fox = term_blob("body", "fox");
        let out = aggregate_tree_blobs(h, &fox, &every_kind(true)).unwrap();
        // The first node, a top-level metric: one owner, its state.
        let mut at = 0;
        assert_eq!(i32_at(&out, &mut at), 1);
        let count = i64_at(&out, &mut at);
        assert!(count >= 0);
        // The metrics blob path agrees on the same query and field.
        let flat = jvm_reader::aggregate_blobs(h, &fox, &{
            let mut b = vec![1u8, 0, 0, lucene_search::aggs::NEED_ALL];
            string(&mut b, "n");
            b.push(0);
            b.extend_from_slice(&0i32.to_le_bytes());
            b
        })
        .unwrap();
        assert_eq!(count as u64, flat.0[0].count);
        // Everything else decodes to the end: encode again from a fresh run
        // and compare byte for byte (deterministic).
        assert_eq!(
            aggregate_tree_blobs(h, &fox, &every_kind(true)).unwrap(),
            out
        );
        // Through the C entry point: a buffer too small reports the size.
        let tree = every_kind(false);
        let mut len = 0usize;
        let rc = unsafe {
            ffi_jvm_reader_aggregate_tree(
                h,
                fox.as_ptr(),
                fox.len(),
                tree.as_ptr(),
                tree.len(),
                std::ptr::null_mut(),
                0,
                &mut len,
            )
        };
        assert_eq!(rc, FfiStatus::BufferTooSmall.code());
        assert!(len > 0);
        let mut buf = vec![0u8; len];
        let rc = unsafe {
            ffi_jvm_reader_aggregate_tree(
                h,
                fox.as_ptr(),
                fox.len(),
                tree.as_ptr(),
                tree.len(),
                buf.as_mut_ptr(),
                buf.len(),
                &mut len,
            )
        };
        assert_eq!(rc, 0);
        assert_eq!(buf, aggregate_tree_blobs(h, &fox, &tree).unwrap());
        let rc = unsafe {
            ffi_jvm_reader_aggregate_tree(
                h,
                fox.as_ptr(),
                fox.len(),
                tree.as_ptr(),
                tree.len(),
                std::ptr::null_mut(),
                4,
                &mut len,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        // Behind min_score, and sliced.
        let mut min = vec![jvm_reader::QUERY_MIN_SCORE];
        min.extend_from_slice(&0.0f32.to_bits().to_le_bytes());
        min.extend_from_slice(&fox);
        let mut sliced = every_kind(false);
        let n = sliced.len();
        sliced.truncate(n - 4);
        sliced.extend_from_slice(&2i32.to_le_bytes());
        for s in [0i32, 1] {
            sliced.extend_from_slice(&1i32.to_le_bytes());
            sliced.extend_from_slice(&s.to_le_bytes());
        }
        assert!(aggregate_tree_blobs(h, &min, &sliced).is_ok());
        // A closed handle.
        crate::jvm_reader::ffi_close_jvm_reader(h);
        assert!(aggregate_tree_blobs(h, &fox, &tree).is_err());
    }

    #[test]
    fn results_encode_as_documented() {
        let mut out = Vec::new();
        let r = AggResult::Terms {
            buckets: vec![(3, vec![(b"a".to_vec(), 2)])],
            subs: vec![
                AggResult::CardinalitySketch(vec![vec![4, 0, 1, 0, 0, 0, 7]]),
                AggResult::Histogram {
                    buckets: vec![vec![(1.5, 4)]],
                    subs: vec![],
                },
                AggResult::DateHistogram {
                    buckets: vec![vec![(9, 1)]],
                    subs: vec![],
                },
                AggResult::Fixed {
                    width: 2,
                    docs: vec![5, 6],
                    subs: vec![],
                },
            ],
        };
        encode_result(&r, &mut out).unwrap();
        // Values rather than a sketch never reach the encoder from a tree.
        assert!(encode_result(&AggResult::Cardinality(vec![]), &mut Vec::new()).is_err());
        let mut at = 0;
        assert_eq!(i32_at(&out, &mut at), 1);
        assert_eq!(i64_at(&out, &mut at), 3);
        assert_eq!(i32_at(&out, &mut at), 1);
        assert_eq!(i64_at(&out, &mut at), 2);
        assert_eq!(i32_at(&out, &mut at), 1);
        assert_eq!(out[at], b'a');
        at += 1;
        // Cardinality: one owner, its sketch's bytes.
        assert_eq!(i32_at(&out, &mut at), 1);
        assert_eq!(i32_at(&out, &mut at), 7);
        assert_eq!(&out[at..at + 7], &[4, 0, 1, 0, 0, 0, 7]);
        at += 7;
        // Histogram, date histogram, fixed.
        assert_eq!((i32_at(&out, &mut at), i32_at(&out, &mut at)), (1, 1));
        assert_eq!(f64::from_bits(i64_at(&out, &mut at) as u64), 1.5);
        assert_eq!(i64_at(&out, &mut at), 4);
        assert_eq!((i32_at(&out, &mut at), i32_at(&out, &mut at)), (1, 1));
        assert_eq!((i64_at(&out, &mut at), i64_at(&out, &mut at)), (9, 1));
        assert_eq!((i32_at(&out, &mut at), i32_at(&out, &mut at)), (1, 2));
        assert_eq!((i64_at(&out, &mut at), i64_at(&out, &mut at)), (5, 6));
        assert_eq!(at, out.len());
    }
}
