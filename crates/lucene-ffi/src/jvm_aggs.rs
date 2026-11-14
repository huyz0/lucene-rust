//! The aggregation-tree entry points over [`lucene_engine::agg_tree`], which
//! holds the tree blob layout, the decoders and the evaluation.

use lucene_aggs::bucket_aggs;

use crate::error::{set_last_error, FfiStatus};
use crate::jvm_reader;

fn handle(handle: u64) -> Result<std::sync::Arc<crate::registry::JvmReaderHandle>, FfiStatus> {
    jvm_reader::lookup(
        handle,
        "ffi_jvm_reader_aggregate_tree: unknown or already-closed handle",
    )
}

/// [`lucene_engine::agg_tree::aggregate_tree_blobs`] over `handle`'s reader.
pub(crate) fn aggregate_tree_blobs(
    h: u64,
    query_blob: &[u8],
    tree_blob: &[u8],
) -> Result<Vec<u8>, FfiStatus> {
    lucene_engine::agg_tree::aggregate_tree_blobs(&*handle(h)?, query_blob, tree_blob)
}

/// [`lucene_engine::agg_tree::aggregate_tree_blobs_cached`] over `handle`'s reader.
pub(crate) fn aggregate_tree_blobs_cached(
    h: u64,
    query_blob: &[u8],
    tree_blob: &[u8],
    cached: bool,
) -> Result<Vec<u8>, FfiStatus> {
    lucene_engine::agg_tree::aggregate_tree_blobs_cached(
        &*handle(h)?,
        query_blob,
        tree_blob,
        cached,
    )
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

/// Sets the bytes the aggregation segment cache keeps parts in, across
/// every index of the process (0 turns it off and drops every part); the
/// parts past a smaller budget are dropped now.
#[no_mangle]
pub extern "C" fn ffi_agg_segment_cache_set_budget(budget_bytes: u64) -> i32 {
    crate::error::guard(|| {
        bucket_aggs::segment_cache::AggSegmentCache::global().set_budget(budget_bytes);
        if budget_bytes == 0 {
            bucket_aggs::segment_cache::AggSegmentCache::global().clear();
        }
        Ok(())
    })
}

/// The aggregation segment cache's counters into `out` (at least
/// [`AGG_CACHE_STATS`] `u64`s): hits, misses, evictions, entries, bytes,
/// budget.
///
/// # Safety
/// `out` must be valid for `len` `u64` writes.
#[no_mangle]
pub unsafe extern "C" fn ffi_agg_segment_cache_stats(out: *mut u64, len: usize) -> i32 {
    crate::error::guard(|| {
        if out.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        if len < AGG_CACHE_STATS {
            set_last_error(format!("{len} slots for {AGG_CACHE_STATS} counters"));
            return Err(FfiStatus::BufferTooSmall);
        }
        let s = bucket_aggs::segment_cache::AggSegmentCache::global().stats();
        let values = [s.hits, s.misses, s.evictions, s.entries, s.bytes, s.budget];
        // SAFETY: `out` is valid for `len >= AGG_CACHE_STATS` writes.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len()) };
        Ok(())
    })
}

/// The counters [`ffi_agg_segment_cache_stats`] writes.
pub const AGG_CACHE_STATS: usize = 6;

#[cfg(test)]
pub(crate) mod tests {
    #![allow(unused_imports)]
    use super::{aggregate_tree_blobs, aggregate_tree_blobs_cached};
    use lucene_engine::agg_tree::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    use lucene_aggs::bucket_aggs::{
        self, AcceptedOrds, AggNode, AggResult, CardinalityKind, CompositeAfter, CompositeKeyValue,
        CompositeSource, CompositeValues, DateRounding, DateUnit, Globals, MissingOrder,
        RoundingKind, SignificantKey, TermsOrder,
    };
    use lucene_aggs::metric_aggs::{Source, ValueKind};
    use lucene_search::field_norms::FieldNorms;
    use lucene_search::multi_segment::OpenSegment;
    use lucene_search::segment_match::MinScore;
    use lucene_search::BooleanQuery;

    use lucene_engine::clauses::map_search_error;
    use lucene_engine::query_blob::Cursor;

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
        b.extend_from_slice(&[AGG_METRIC, kind, source, lucene_aggs::metric_aggs::NEED_ALL]);
        string(b, field);
    }

    /// `terms(field, size)` alone.
    pub(crate) fn terms_tree(field: &str, size: i32) -> Vec<u8> {
        let mut b = vec![1u8, AGG_TERMS];
        string(&mut b, field);
        b.extend_from_slice(&size.to_le_bytes());
        b.extend_from_slice(&[0, 0, 0]);
        b.push(0);
        b.extend_from_slice(&0i32.to_le_bytes());
        b
    }

    /// `histogram(field, interval) > filters(filter, other) > count(field)`
    /// next to a top-level `count(field)`.
    pub(crate) fn histogram_of_filters(field: &str, interval: f64, filter: &[u8]) -> Vec<u8> {
        let mut b = vec![2u8, AGG_HISTOGRAM];
        string(&mut b, field);
        b.push(0);
        f64s(&mut b, &[interval, 0.0]);
        b.push(0);
        b.push(1);
        b.push(AGG_FILTERS);
        b.extend_from_slice(&1i32.to_le_bytes());
        b.extend_from_slice(&(filter.len() as i32).to_le_bytes());
        b.extend_from_slice(filter);
        b.push(1);
        b.push(1);
        metric(&mut b, 0, 0, field);
        metric(&mut b, 0, 0, field);
        b.push(0);
        b.extend_from_slice(&0i32.to_le_bytes());
        b
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

    /// A dashboard: `date_histogram(n, 3) > terms(body, 5) > stats(n)`
    /// beside `range(n) > filters(fox, other) > stats(n)`, `kind` the
    /// metrics' value kind.
    pub(crate) fn dashboard_tree(kind: u8) -> Vec<u8> {
        let mut b = vec![2u8];
        b.push(AGG_DATE_HISTOGRAM);
        string(&mut b, "n");
        b.push(UNIT_INTERVAL);
        b.extend_from_slice(&3i64.to_le_bytes());
        b.extend_from_slice(&0i64.to_le_bytes());
        b.extend_from_slice(&0i64.to_le_bytes());
        b.push(0);
        b.push(1);
        b.push(AGG_TERMS);
        string(&mut b, "body");
        b.extend_from_slice(&5i32.to_le_bytes());
        b.push(0);
        b.push(0);
        b.push(1);
        metric(&mut b, kind, 0, "n");
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
        metric(&mut b, kind, 0, "n");
        b.push(0);
        b.extend_from_slice(&0i32.to_le_bytes());
        b
    }

    #[test]
    fn the_segment_cache_answers_as_the_pass_without_it() {
        use crate::jvm_reader::tests::{open_live, range_blob, term_and_range_blob};
        use lucene_aggs::bucket_aggs::segment_cache::AggSegmentCache;
        let cache = AggSegmentCache::global();
        cache.set_budget(64 << 20);
        let trees = [
            dashboard_tree(0),
            dashboard_tree(1),
            terms_tree("body", 2),
            histogram_of_filters("n", 2.0, &term_blob("body", "fox")),
            every_kind(true),
        ];
        let queries = [
            term_blob("body", "fox"),
            term_blob("body", "the"),
            range_blob("n", i64::MIN, i64::MAX),
            range_blob("n", 2, 5),
            range_blob("n", -10, 1),
            term_and_range_blob("body", "the", "n", 1, 100),
            term_and_range_blob("body", "fox", "n", 3, 3),
        ];
        // No deletions, then one in each segment, then a whole segment.
        let lives: [&[&[u64]]; 3] = [&[], &[&[0b1110], &[0b0111]], &[&[0], &[0b1111]]];
        let before = cache.stats();
        for live in lives {
            let (rc, h) = open_live(&[4, 4], 0, live);
            assert_eq!(rc, 0, "{}", crate::error::last_error());
            for (t, tree) in trees.iter().enumerate() {
                for (q, query) in queries.iter().enumerate() {
                    let plain = aggregate_tree_blobs(h, query, tree).unwrap();
                    for round in 0..2 {
                        assert_eq!(
                            aggregate_tree_blobs_cached(h, query, tree, true).unwrap(),
                            plain,
                            "tree {t}, query {q}, live {live:?}, round {round}"
                        );
                    }
                }
            }
            crate::jvm_reader::ffi_close_jvm_reader(h);
        }
        let after = cache.stats();
        assert!(after.hits > before.hits, "{before:?} {after:?}");
        assert!(after.misses > before.misses, "{before:?} {after:?}");
        assert!(after.entries > 0 && after.bytes > 0);
        // The counters through the C entry point.
        let mut out = [0u64; AGG_CACHE_STATS];
        assert_eq!(
            unsafe { ffi_agg_segment_cache_stats(out.as_mut_ptr(), out.len()) },
            0
        );
        assert!(out[0] >= after.hits && out[5] == 64 << 20);
        assert_eq!(
            unsafe { ffi_agg_segment_cache_stats(out.as_mut_ptr(), 2) },
            FfiStatus::BufferTooSmall.code()
        );
        assert_eq!(
            unsafe { ffi_agg_segment_cache_stats(std::ptr::null_mut(), 6) },
            FfiStatus::NullPointer.code()
        );
        assert_eq!(ffi_agg_segment_cache_set_budget(64 << 20), 0);
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
        let mut d = lucene_aggs::tdigest::MergingDigest::new(100.0);
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
        use lucene_aggs::bucket_aggs::SortedHits;
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
    fn significant_text_decodes_and_a_refusal_encodes_as_minus_one() {
        let spec = lucene_engine::analysis::test_support::spec(
            &[],
            ("whitespace", &[]),
            &[("lowercase", &[])],
            &[],
        );
        let mut b = vec![1u8, AGG_SIGNIFICANT_TEXT, 1];
        b.extend_from_slice(&(spec.len() as i32).to_le_bytes());
        b.extend_from_slice(&spec);
        string(&mut b, "body");
        b.extend_from_slice(&2i32.to_le_bytes());
        string(&mut b, "title");
        string(&mut b, "body");
        b.push(0);
        b.extend_from_slice(&[0u8, 0, 0, 0, 0]);
        let tree = decode_tree(&b).unwrap();
        let AggNode::SignificantTerms {
            field,
            terms: true,
            text: Some(t),
            subs,
            ..
        } = &tree.nodes[0]
        else {
            panic!("{:?}", tree.nodes[0]);
        };
        assert_eq!(field, "body");
        assert!(t.dedup && subs.is_empty());
        assert_eq!(t.source_fields, ["title", "body"]);
        // No source fields, or a flag past 1.
        for (flag, n) in [(1u8, 0i32), (2, 1)] {
            let mut b = vec![1u8, AGG_SIGNIFICANT_TEXT, flag];
            b.extend_from_slice(&(spec.len() as i32).to_le_bytes());
            b.extend_from_slice(&spec);
            string(&mut b, "body");
            b.extend_from_slice(&n.to_le_bytes());
            if n > 0 {
                string(&mut b, "body");
            }
            b.push(0);
            b.extend_from_slice(&[0u8, 0, 0, 0, 0]);
            assert_eq!(decode_tree(&b).err(), Some(FfiStatus::InvalidArgument));
        }
        let mut out = Vec::new();
        encode_result(&AggResult::Refused("not JSON".into()), &mut out).unwrap();
        assert_eq!(out, (-1i32).to_le_bytes());
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
            AggNode::SignificantTerms { field, numeric: false, terms: true, missing_term: Some(m), missing: None, text: None, subs }
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
            let mut b = vec![1u8, 0, 0, lucene_aggs::metric_aggs::NEED_ALL];
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
        let mut min = vec![lucene_engine::query_blob::QUERY_MIN_SCORE];
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
