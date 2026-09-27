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
//!           | CARDINALITY card_kind:u8 field
//!           | TERMS field shard_size:i32 nodes
//!           | HISTOGRAM field value_kind:u8 interval:f64 offset:f64 bounds_f64 nodes
//!           | DATE_HISTOGRAM field unit:u8 [interval:i64 when unit = UNIT_INTERVAL]
//!                            zone_ms:i64 offset:i64 bounds_i64 nodes
//!           | RANGE field value_kind:u8 count:i32 (from:f64 to:f64)* nodes
//!           | FILTERS count:i32 query:bytes* other:u8 nodes
//!           | GLOBAL nodes
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
//! CARDINALITY  n:i32 (count:i32 value*)*n      value: term:bytes (keyword) | i64
//! TERMS        n:i32 (other:i64 count:i32 (docs:i64 term:bytes)*)*n  subs
//! HISTOGRAM    n:i32 (count:i32 (key:f64 docs:i64)*)*n  subs
//! DATE_HIST.   n:i32 (count:i32 (key:i64 docs:i64)*)*n  subs
//! RANGE/FILTERS/GLOBAL  n:i32 width:i32 docs:i64*(n*width)  subs
//! ```
//!
//! and `subs` is each sub-node's result, its owning buckets being the
//! node's buckets in the order listed.

use std::collections::HashMap;
use std::sync::Arc;

use lucene_search::aggs::{MinScore, Source, ValueKind};
use lucene_search::bucket_aggs::{
    self, AggNode, AggResult, CardinalityKind, CardinalityValue, DateRounding, DateUnit, Globals,
    RoundingKind,
};
use lucene_search::field_norms::FieldNorms;
use lucene_search::multi_segment::OpenSegment;
use lucene_search::BooleanQuery;

use crate::error::{set_last_error, FfiStatus};
use crate::jvm_reader::{self, Cursor};

const AGG_METRIC: u8 = 0;
const AGG_CARDINALITY: u8 = 1;
const AGG_TERMS: u8 = 2;
const AGG_HISTOGRAM: u8 = 3;
const AGG_DATE_HISTOGRAM: u8 = 4;
const AGG_RANGE: u8 = 5;
const AGG_FILTERS: u8 = 6;
const AGG_GLOBAL: u8 = 7;

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
                field: field(c)?,
                kind,
                source,
                needs,
            }
        }
        AGG_CARDINALITY => {
            let kind = match c.u8()? {
                0 => CardinalityKind::Keyword,
                k @ 1..=3 => CardinalityKind::Numeric(value_kind(k - 1)?),
                other => return Err(bad(format!("aggregation tree: cardinality kind {other}"))),
            };
            AggNode::Cardinality {
                field: field(c)?,
                kind,
            }
        }
        AGG_TERMS => {
            let field = field(c)?;
            let shard_size = c.i32()?;
            let shard_size = usize::try_from(shard_size)
                .ok()
                .filter(|&n| n >= 1)
                .ok_or_else(|| bad(format!("aggregation tree: shard_size {shard_size}")))?;
            AggNode::Terms {
                field,
                shard_size,
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
            let flags = c.u8()?;
            let min = if flags & 1 != 0 { Some(c.i64()?) } else { None };
            let max = if flags & 2 != 0 { Some(c.i64()?) } else { None };
            AggNode::DateHistogram {
                field,
                rounding: DateRounding {
                    kind,
                    zone_ms,
                    offset,
                },
                hard_bounds: (min, max),
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
        AggResult::Cardinality(per) => {
            put_i32(out, per.len())?;
            for values in per {
                put_i32(out, values.len())?;
                for v in values {
                    match v {
                        CardinalityValue::Term(t) => put_bytes(out, t)?,
                        CardinalityValue::Long(l) => out.extend_from_slice(&l.to_le_bytes()),
                    }
                }
            }
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
    let (query, min_score) = jvm_reader::decode_request(query_blob)?;
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
    let globals = Globals { ords: &ords };
    let norm_fields: Vec<String> = if min_score.is_some() {
        crate::query::clause_field_names(&q)
            .into_iter()
            .map(str::to_string)
            .collect()
    } else {
        Vec::new()
    };
    let owned = h.reader.field_norms_by_field(&norm_fields);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> =
        owned.iter().map(|m| (!m.is_empty()).then_some(m)).collect();
    let min = min_score.map(|min| MinScore { min, norms: &norms });
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
        let results =
            bucket_aggs::aggregate_tree(&segments, readers, gq, gnodes, &globals, &slices, None)
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
mod tests {
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
    fn every_kind(global: bool) -> Vec<u8> {
        let mut b = vec![6u8];
        // A top-level metric reading the points bound.
        metric(&mut b, 0, 1, "n");
        b.push(AGG_CARDINALITY);
        b.push(1);
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
        assert_eq!(bad(&with(&[AGG_CARDINALITY, 9, 0, 0, 0, 0])), invalid);
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
                AggResult::Cardinality(vec![vec![
                    CardinalityValue::Term(b"x".to_vec()),
                    CardinalityValue::Long(-1),
                ]]),
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
        let mut at = 0;
        assert_eq!(i32_at(&out, &mut at), 1);
        assert_eq!(i64_at(&out, &mut at), 3);
        assert_eq!(i32_at(&out, &mut at), 1);
        assert_eq!(i64_at(&out, &mut at), 2);
        assert_eq!(i32_at(&out, &mut at), 1);
        assert_eq!(out[at], b'a');
        at += 1;
        // Cardinality: one owner, two values.
        assert_eq!(i32_at(&out, &mut at), 1);
        assert_eq!(i32_at(&out, &mut at), 2);
        assert_eq!(i32_at(&out, &mut at), 1);
        at += 1;
        assert_eq!(i64_at(&out, &mut at), -1);
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
