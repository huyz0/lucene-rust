//! The plugin's aggregation trees: a tree blob decoded into
//! [`lucene_aggs::bucket_aggs::AggNode`]s, run over a reader handle, and
//! the shard results encoded for the plugin's `NativeAggregations.Tree` to
//! turn into OpenSearch's `InternalAggregation`s.
//!
//! # The tree blob
//!
//! Little-endian; every length an `i32`, every string UTF-8 behind its
//! length; trailing bytes are an error.
//!
//! ```text
//! blob     := nodes global slices [bucket_limit:i32]
//!             (the buckets the request may still make, `search.max_buckets`
//!              less those made; past them the call stops with
//!              `TOO_MANY_BUCKETS`, the count made in the last error)
//! nodes    := count:u8 node*                  (at most MAX_NODES in all)
//! global   := 0 | 1 query:bytes nodes         (the global pass: its query and nodes)
//! node     := METRIC value_kind:u8 source:u8 needs:u8 field
//!           | SCRIPT_METRIC needs:u8 script   (a metric over a Painless
//!                                              script's doubles; the script in
//!                                              `jvm_painless_score`'s layout)
//!           | CARDINALITY card_kind:u8 precision:u8 field
//!           | PERCENTILES value_kind:u8 compression:f64 field   (a t-digest per bucket;
//!                                              compression in (0, MAX_COMPRESSION])
//!           | HDR_PERCENTILES value_kind:u8 field  (each bucket's values, for an
//!                                              HDR `DoubleHistogram`)
//!           | TOP_HITS top_n:i32 sorted:u8 [sort:bytes]  (top_n > 0; sorted 0: by
//!                  score; 1: by the sort blob, `crate::reader_search::decode_sort`'s, its
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
//! Queries are query blobs ([`crate::query_blob::decode_query`]); the slices
//! are `reader_search`'s.
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
//!              terms (`crate::reader_search::encode_terms`)
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

use lucene_aggs::bucket_aggs::{
    self, AcceptedOrds, AggNode, AggResult, CardinalityKind, CompositeAfter, CompositeKeyValue,
    CompositeSource, CompositeValues, DateRounding, DateUnit, Globals, MissingOrder, RoundingKind,
    SignificantKey, TermsOrder,
};
use lucene_aggs::metric_aggs::{Source, ValueKind};
use lucene_search::field_norms::FieldNorms;
use lucene_search::multi_segment::OpenSegment;
use lucene_search::segment_match::MinScore;
use lucene_search::BooleanQuery;

use crate::clauses::map_search_error;
use crate::query_blob::Cursor;
use crate::reader_handle::JvmReaderHandle;
use crate::status::{set_last_error, Status};

pub const AGG_METRIC: u8 = 0;
pub const AGG_CARDINALITY: u8 = 1;
pub const AGG_TERMS: u8 = 2;
pub const AGG_HISTOGRAM: u8 = 3;
pub const AGG_DATE_HISTOGRAM: u8 = 4;
pub const AGG_RANGE: u8 = 5;
pub const AGG_FILTERS: u8 = 6;
pub const AGG_GLOBAL: u8 = 7;
pub const AGG_NUMERIC_TERMS: u8 = 8;
/// A prefix, not a node: the next node's `missing` value.
pub const AGG_MISSING: u8 = 9;
pub const AGG_COMPOSITE: u8 = 10;
pub const AGG_PERCENTILES: u8 = 11;
pub const AGG_TOP_HITS: u8 = 12;
pub const AGG_SIGNIFICANT_TERMS: u8 = 13;
pub const AGG_HDR_PERCENTILES: u8 = 14;
pub const AGG_NESTED: u8 = 15;
pub const AGG_REVERSE_NESTED: u8 = 16;
pub const AGG_SIGNIFICANT_TEXT: u8 = 17;
/// A metric over a Painless script's values ([`crate::painless_score::PainlessAggValues`]).
pub const AGG_SCRIPT_METRIC: u8 = 18;
/// The most source fields a `significant_text` may read.
pub const MAX_SOURCE_FIELDS: i32 = 1024;
/// The largest t-digest compression a tree may ask for: a digest holds
/// about `12 * compression` doubles.
pub const MAX_COMPRESSION: f64 = 10_000.0;
/// The most `composite` sources a tree may name.
pub const MAX_COMPOSITE_SOURCES: usize = 16;

/// `date_histogram` units; [`UNIT_INTERVAL`] is a fixed interval.
pub const UNITS: [DateUnit; 8] = [
    DateUnit::Week,
    DateUnit::Year,
    DateUnit::Quarter,
    DateUnit::Month,
    DateUnit::Day,
    DateUnit::Hour,
    DateUnit::Minute,
    DateUnit::Second,
];
pub const UNIT_INTERVAL: u8 = 8;

/// At most this many nodes in one blob, and this deep.
pub const MAX_NODES: usize = 256;
pub const MAX_DEPTH: usize = 16;
/// At most this many ranges or filters in one node.
pub const MAX_WIDTH: usize = 4096;

/// A decoded tree blob: the main nodes, the global pass, the slices.
pub struct DecodedTree {
    pub nodes: Vec<AggNode>,
    pub global: Option<(BooleanQuery, Vec<AggNode>)>,
    pub slices: Vec<Vec<usize>>,
    /// The buckets the request may still make (`search.max_buckets` less
    /// those made); `usize::MAX` when the blob gives none.
    pub bucket_limit: usize,
}

pub fn bad(msg: String) -> Status {
    set_last_error(msg);
    Status::InvalidArgument
}

pub fn value_kind(k: u8) -> Result<ValueKind, Status> {
    match k {
        0 => Ok(ValueKind::Long),
        1 => Ok(ValueKind::Double),
        2 => Ok(ValueKind::Float),
        other => Err(bad(format!("aggregation tree: value kind {other}"))),
    }
}

pub fn field(c: &mut Cursor<'_>) -> Result<String, Status> {
    Ok(std::str::from_utf8(c.bytes()?)
        .map_err(|_| Status::InvalidUtf8)?
        .to_string())
}

pub fn f64_of(c: &mut Cursor<'_>) -> Result<f64, Status> {
    Ok(f64::from_bits(c.i64()? as u64))
}

/// A `date_histogram`'s rounding: `unit [interval] zone_ms offset`.
pub fn date_rounding(c: &mut Cursor<'_>) -> Result<DateRounding, Status> {
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
pub fn shard_size(c: &mut Cursor<'_>) -> Result<usize, Status> {
    let n = c.i32()?;
    usize::try_from(n)
        .ok()
        .filter(|&n| n >= 1)
        .ok_or_else(|| bad(format!("aggregation tree: shard_size {n}")))
}

/// A `terms` order: 0 `_count` descending, 1 `_count` ascending, 2 `_key`
/// ascending, 3 `_key` descending.
pub fn terms_order(c: &mut Cursor<'_>) -> Result<TermsOrder, Status> {
    Ok(match c.u8()? {
        0 => TermsOrder::CountDesc,
        1 => TermsOrder::CountAsc,
        2 => TermsOrder::KeyAsc,
        3 => TermsOrder::KeyDesc,
        other => return Err(bad(format!("aggregation tree: terms order {other}"))),
    })
}

pub fn width(c: &mut Cursor<'_>) -> Result<usize, Status> {
    let n = c.length()?;
    if n == 0 || n > MAX_WIDTH {
        return Err(bad(format!(
            "aggregation tree: {n} buckets, want 1..={MAX_WIDTH}"
        )));
    }
    Ok(n)
}

pub fn query_of(blob: &[u8]) -> Result<BooleanQuery, Status> {
    Ok(crate::query_blob::boolean_of(
        &crate::query_blob::decode_query(blob)?,
    ))
}

pub fn nodes(c: &mut Cursor<'_>, depth: usize, count: &mut usize) -> Result<Vec<AggNode>, Status> {
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

pub fn node(c: &mut Cursor<'_>, depth: usize, count: &mut usize) -> Result<AggNode, Status> {
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

pub fn node_of(
    kind: u8,
    c: &mut Cursor<'_>,
    depth: usize,
    count: &mut usize,
) -> Result<AggNode, Status> {
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
            if needs == 0 || needs & !lucene_aggs::metric_aggs::NEED_ALL != 0 {
                return Err(bad(format!("aggregation tree: metric needs {needs:#x}")));
            }
            AggNode::Metric {
                missing: None,
                field: field(c)?,
                kind,
                source,
                needs,
                script: None,
            }
        }
        AGG_SCRIPT_METRIC => {
            let needs = c.u8()?;
            if needs == 0 || needs & !lucene_aggs::metric_aggs::NEED_ALL != 0 {
                return Err(bad(format!("aggregation tree: metric needs {needs:#x}")));
            }
            let values = crate::painless_score::PainlessAggValues::decode(c)?;
            AggNode::Metric {
                missing: None,
                field: String::new(),
                kind: ValueKind::Double,
                source: Source::DocValues,
                needs,
                script: Some(Arc::new(values)),
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
                text: None,
                subs: nodes(c, depth + 1, count)?,
            }
        }
        AGG_SIGNIFICANT_TEXT => {
            let dedup = match c.u8()? {
                0 => false,
                1 => true,
                f => return Err(bad(format!("aggregation tree: significant_text flags {f}"))),
            };
            let analyzer = crate::analysis::analyzer(c.bytes()?)?;
            let name = field(c)?;
            let n = c.i32()?;
            if !(1..=MAX_SOURCE_FIELDS).contains(&n) {
                return Err(bad(format!(
                    "aggregation tree: significant_text with {n} source fields"
                )));
            }
            let mut source_fields = Vec::new();
            for _ in 0..n {
                source_fields.push(field(c)?);
            }
            AggNode::SignificantTerms {
                field: name.clone(),
                numeric: false,
                terms: true,
                missing_term: None,
                missing: None,
                text: Some(std::sync::Arc::new(
                    lucene_aggs::significant_text::TextSource {
                        source_fields,
                        field: name,
                        analyzer,
                        dedup,
                    },
                )),
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
                        crate::reader_search::decode_sort(c.bytes()?)?;
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
                let words = c.length()?;
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
            let segments = c.length()?;
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
pub fn decode_tree(blob: &[u8]) -> Result<DecodedTree, Status> {
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
    let slices = crate::reader_search::decode_slices(&mut c, &bad)?;
    let bucket_limit = if c.pos() == blob.len() {
        usize::MAX
    } else {
        usize::try_from(c.i32()?).unwrap_or(0)
    };
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
        bucket_limit,
    })
}

pub fn put_i32(out: &mut Vec<u8>, v: usize) -> Result<(), Status> {
    let v = i32::try_from(v).map_err(|_| bad(format!("aggregation result: {v} too many")))?;
    out.extend_from_slice(&v.to_le_bytes());
    Ok(())
}

pub fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&i64::try_from(v).unwrap_or(i64::MAX).to_le_bytes());
}

pub fn put_bytes(out: &mut Vec<u8>, b: &[u8]) -> Result<(), Status> {
    put_i32(out, b.len())?;
    out.extend_from_slice(b);
    Ok(())
}

/// Appends one node's result (see the module doc).
pub fn encode_result(r: &AggResult, out: &mut Vec<u8>) -> Result<(), Status> {
    match r {
        AggResult::Metric(states) => {
            put_i32(out, states.len())?;
            for s in states {
                put_u64(out, s.count);
                for v in crate::reader_search::metric_values(s) {
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
        // Java reads a count of -1 as its aggregator's refusal.
        AggResult::Refused(_) => out.extend_from_slice(&(-1i32).to_le_bytes()),
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
                put_bytes(out, &crate::reader_search::encode_terms(&hits.sort, docs)?)?;
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

pub fn encode_subs(subs: &[AggResult], out: &mut Vec<u8>) -> Result<(), Status> {
    for s in subs {
        encode_result(s, out)?;
    }
    Ok(())
}

/// Runs a tree blob over `handle`'s reader for the query blob (a
/// `min_score` in front of it applies to the main pass only, as
/// OpenSearch's `MinimumScoreCollector` wraps only the query's collectors)
/// and encodes the results.
pub fn aggregate_tree_blobs(
    h: &JvmReaderHandle,
    query_blob: &[u8],
    tree_blob: &[u8],
) -> Result<Vec<u8>, Status> {
    aggregate_tree_blobs_cached(h, query_blob, tree_blob, false)
}

/// [`aggregate_tree_blobs`], with `cached` each committed segment's part
/// read from and kept in the process's aggregation segment cache
/// ([`lucene_aggs::bucket_aggs::segment_cache`]): the same results.
pub fn aggregate_tree_blobs_cached(
    h: &JvmReaderHandle,
    query_blob: &[u8],
    tree_blob: &[u8],
    cached: bool,
) -> Result<Vec<u8>, Status> {
    let (query, min_score, sim) = crate::query_blob::decode_request(query_blob)?;
    let sim = sim.as_ref();
    let tree = decode_tree(tree_blob)?;
    let query = crate::query_blob::similarity_checked(sim, crate::join::resolve(h, query, sim))?;
    let mut opened = h.reader.open_segments().map_err(|e| {
        set_last_error(format!("opening segment postings: {e}"));
        Status::Decode
    })?;
    let q = crate::query_blob::boolean_of(&query);
    let mut filters = Vec::new();
    for n in tree
        .nodes
        .iter()
        .chain(tree.global.iter().flat_map(|(_, g)| g.iter()))
    {
        n.filter_queries(&mut filters);
    }
    let global_query = tree.global.as_ref().map(|(gq, _)| gq);
    let points = crate::query_blob::query_uses_points(&query)
        || filters
            .iter()
            .copied()
            .chain(global_query)
            .any(crate::query_blob::boolean_uses_points)
        || tree.nodes.iter().any(AggNode::reads_points)
        // A top-level histogram or range may be counted from the points.
        || tree
            .nodes
            .iter()
            .any(|n| matches!(n, AggNode::DateHistogram { .. } | AggNode::Range { .. }));
    if points {
        opened.open_points().map_err(|e| {
            set_last_error(format!("opening segment points: {e}"));
            Status::Decode
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
            .map_err(crate::clauses::map_search_error)?;
        ords.insert(f.to_string(), g);
    }
    let globals = Globals::new(&ords).with_bucket_limit(tree.bucket_limit);
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
            crate::clauses::clause_field_names(&q)
                .into_iter()
                .map(str::to_string),
        );
    }
    if let (true, Some(gq)) = (global_scores, global_query) {
        for f in crate::clauses::clause_field_names(gq) {
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
    let cache = bucket_aggs::segment_cache::AggSegmentCache::global();
    let request =
        (cached && cache.budget() > 0).then(|| bucket_aggs::segment_cache::CacheRequest {
            cache,
            segments: h.agg_identities(),
        });
    let mut out = Vec::new();
    if !tree.nodes.is_empty() {
        let sliced = bucket_aggs::aggregate_tree_cached(
            &segments,
            readers,
            &q,
            &tree.nodes,
            &globals,
            &slices,
            min.as_ref(),
            request.as_ref(),
        )
        .map_err(crate::clauses::map_search_error);
        // A similarity script's error first: the search met it before
        // anything that ended it.
        let sliced = crate::query_blob::similarity_checked(sim, sliced)?;
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
        let results = bucket_aggs::aggregate_tree_cached(
            &segments,
            readers,
            gq,
            gnodes,
            &globals,
            &slices,
            global_min.as_ref(),
            request.as_ref(),
        )
        .map_err(crate::clauses::map_search_error);
        let results = crate::query_blob::similarity_checked(sim, results)?;
        for results in &results {
            for r in results {
                encode_result(r, &mut out)?;
            }
        }
    }
    crate::query_blob::similarity_checked(sim, Ok(out))
}
