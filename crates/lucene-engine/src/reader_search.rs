//! The reader the OpenSearch plugin searches: one handle per Java
//! `DirectoryReader`, opened from that reader's own segment list and masked
//! with its live docs, so a doc ID means the same document on both sides.
//!
//! An OpenSearch searcher is usually an NRT reader, not the last commit on
//! disk: it lists segments flushed since the commit and carries deletions
//! that exist only in the JVM. So the JVM hands over:
//!
//! - the segment list, as the bytes `SegmentInfos.write(IndexOutput)` writes
//!   (the `segments_N` format `lucene_index::segment_infos::parse` reads);
//! - each segment's `maxDoc`, which `ffi_open_jvm_reader` checks against
//!   what it opened, since a wrong doc base returns the wrong documents;
//! - each segment's live docs from the leaf's `getLiveDocs()`, which replace
//!   whatever `.liv` or a soft-deletes field would give.
//!
//! A refresh passes the previous handle as `previous`, and unchanged
//! segments are shared (`DirectoryReader::reopen_at`).
//!
//! `ffi_jvm_reader_search` decodes the query blob ([`decode_query`]
//! documents the layout), runs it, counts total hits when asked, and writes
//! the top hits into the caller's buffers in one crossing.

use crate::clauses::map_search_error;
use crate::query_blob::{
    boolean_of, decode_query, decode_request, invalid, query_uses_points, similarity_checked,
    Cursor, JvmQuery,
};
use crate::reader_handle::JvmReaderHandle;
use crate::status::{set_last_error, Status};
use lucene_aggs::metric_aggs::{MetricSpec, MetricState, Source, ValueKind, NEED_ALL};
use lucene_aggs::terms_agg::TermsSpec;
use lucene_search::field_norms::FieldNorms;
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_search::similarities::{PerFieldSimilarity, Similarity};
use lucene_search::top_field::{
    CollapseKind, CollapseTopDocs, CollapseValue, FieldDoc, NestedSort, Selector, SortField,
    SortType,
};
use lucene_search::weight_count::count_term_query;
use lucene_search::{
    count_boolean_query_segment_with, search_boolean_query_multi_segment_maxscore_counting,
    search_term_query_multi_segment_counting, ScoreDoc, TotalHits, TotalHitsRelation,
};
use std::sync::Arc;

/// Sort-key types in a sort blob ([`decode_sort`]).
pub const SORT_SCORE: u8 = 0;
pub const SORT_DOC: u8 = 1;
pub const SORT_LONG: u8 = 2;
pub const SORT_INT: u8 = 3;
pub const SORT_DOUBLE: u8 = 4;
pub const SORT_FLOAT: u8 = 5;
pub const SORT_STRING: u8 = 6;
/// A geo-distance key ([`decode_geo_sort`]).
pub const SORT_GEO_DISTANCE: u8 = 7;
/// OpenSearch's `_shard_doc` key ([`decode_opensearch_sort`]): flags
/// [`SORT_REVERSE`] only, then `shardKeyPrefix: i64`.
pub const SORT_SHARD_DOC: u8 = 8;
/// OpenSearch's `half_float` key: the flags of a numeric key without
/// [`SORT_NESTED`], the field, then the missing float as its sortable int
/// (`i64`).
pub const SORT_HALF_FLOAT: u8 = 9;
/// OpenSearch's `unsigned_long` key: as [`SORT_HALF_FLOAT`] but no
/// [`SORT_MODE_MEDIAN`], the missing value the unsigned long's bits.
pub const SORT_UNSIGNED_LONG: u8 = 10;
/// A Painless `_script` key of type `number`
/// ([`crate::painless_score::PainlessSort`]): flags [`SORT_REVERSE`]
/// only, then the script ([`crate::painless_score`]'s layout, context
/// `3` or `4`).
pub const SORT_SCRIPT: u8 = 11;
/// At most this many origins in a geo-distance key: each costs a distance
/// per value per document.
pub const MAX_GEO_ORIGINS: usize = 1024;
/// Sort-key flags.
pub const SORT_REVERSE: u8 = 1;
pub const SORT_MAX: u8 = 2;
/// Sort-key flags' mode field (`SORT_MODE_MASK`): OpenSearch's
/// `MultiValueMode` for a numeric key it sorts with its own comparator source
/// -- `0` is the `min`/`max` selector [`SORT_MAX`] says.
pub const SORT_MODE_MASK: u8 = 12;
pub const SORT_MODE_SUM: u8 = 4;
pub const SORT_MODE_AVG: u8 = 8;
pub const SORT_MODE_MEDIAN: u8 = 12;
/// Key flag: a nested key ([`NestedSort`]); after its missing value come the
/// root filter and the inner query (query blobs, `len: i32` each) and
/// `max_children: i32`.
pub const SORT_NESTED: u8 = 16;
/// Sort-blob options: track the max score over every match.
pub const SORT_TRACK_MAX_SCORE: u8 = 1;
/// Sort-blob options: `terminate_after`, its count (`i32`, at least 1)
/// following the options byte.
pub const SORT_TERMINATE_AFTER: u8 = 2;
/// Sort-blob options, beside [`SORT_TERMINATE_AFTER`]: the total is a `size:
/// 0` search's, whole segments counted where `Weight.count` answers
/// ([`lucene_search::terminate::count_until`]).
pub const SORT_COUNT_SEGMENTS: u8 = 4;
/// Sort-blob option: after the slices, `count: i32` and one `u8` per segment,
/// `1` where the sort is a prefix of that segment's index sort
/// (`TopFieldCollector.canEarlyTerminate`, which the JVM decides with
/// `SortField.equals`), `1 + p` where it is the index sort's keys from the
/// `p`th on, `0` where neither.
pub const SORT_INDEX_SORTED: u8 = 8;
/// At most this many keys: OpenSearch's sorts are a handful, and each key
/// costs a value per hit on the way back.
pub const MAX_SORT_KEYS: usize = 16;

/// A decoded sort blob: the keys, the search-after document, whether to
/// track the max score, the slices, `terminate_after` with whether its
/// total counts whole segments ([`SORT_COUNT_SEGMENTS`]), and per segment
/// where the sort sits in its index sort ([`SORT_INDEX_SORTED`]): the index
/// sort key it begins at, `None` when it is none of the index sort's runs of
/// keys.
pub type DecodedSort = (
    Vec<SortField>,
    Option<FieldDoc>,
    bool,
    Vec<Vec<usize>>,
    Option<(u64, bool)>,
    Vec<Option<usize>>,
);

/// The concurrent-search slices ending a sort or metrics blob: `count: i32`,
/// then per slice `len: i32` and that many segment indices (`i32`). None
/// means the search is not concurrent.
pub fn decode_slices(
    c: &mut Cursor<'_>,
    bad: &dyn Fn(String) -> Status,
) -> Result<Vec<Vec<usize>>, Status> {
    let slice_count =
        usize::try_from(c.i32()?).map_err(|_| bad("blob: negative slice count".into()))?;
    if slice_count > MAX_SLICES {
        return Err(bad(format!(
            "blob: {slice_count} slices, want at most {MAX_SLICES}"
        )));
    }
    let mut slices = Vec::new();
    let mut total = 0usize;
    for _ in 0..slice_count {
        let len =
            usize::try_from(c.i32()?).map_err(|_| bad("blob: negative slice length".into()))?;
        total = total.saturating_add(len);
        if total > MAX_SLICED_SEGMENTS {
            return Err(bad(format!(
                "blob: more than {MAX_SLICED_SEGMENTS} sliced segments"
            )));
        }
        let mut slice = Vec::new();
        for _ in 0..len {
            slice
                .push(usize::try_from(c.i32()?).map_err(|_| bad("blob: negative segment".into()))?);
        }
        slices.push(slice);
    }
    Ok(slices)
}

/// Decodes a sort blob: `key_count: u8`, then per key `type: u8`,
/// `flags: u8` ([`SORT_REVERSE`], [`SORT_MAX`]) and, for a field's type, the
/// field (`len: i32`, UTF-8) and the missing value: a comparable `i64` for a
/// numeric key (see [`lucene_search::top_field`]), `1` (`STRING_LAST`) or `0`
/// (`STRING_FIRST`) for a keyword one; then `has_after: u8` and, when 1, the
/// search-after document (`doc: i32`) and one value per key, encoded as the
/// search returns them -- an `i64`, or for a keyword key `present: u8` and,
/// when 1, the term (`len: i32`, bytes); then `options: u8`
/// ([`SORT_TRACK_MAX_SCORE`]); and last the concurrent-search slices
/// ([`decode_slices`]). Little-endian, and trailing bytes are an error.
pub fn decode_sort(blob: &[u8]) -> Result<DecodedSort, Status> {
    let mut c = Cursor { buf: blob, pos: 0 };
    let bad = |msg: String| {
        set_last_error(msg);
        Status::InvalidArgument
    };
    let n = usize::from(c.u8()?);
    if n == 0 || n > MAX_SORT_KEYS {
        return Err(bad(format!(
            "sort blob: {n} keys, want 1..={MAX_SORT_KEYS}"
        )));
    }
    let mut keys = Vec::new();
    // Per key: a geo-distance key in Lucene's own comparator.
    let mut lucene_geo = Vec::new();
    for _ in 0..n {
        let ty = c.u8()?;
        let flags = c.u8()?;
        if flags & !(SORT_REVERSE | SORT_MAX | SORT_MODE_MASK | SORT_NESTED) != 0 {
            return Err(bad(format!("sort blob: unknown flags {flags:#x}")));
        }
        let mode = flags & SORT_MODE_MASK;
        if ty == SORT_SCRIPT {
            if flags & !SORT_REVERSE != 0 {
                return Err(bad(format!("sort blob: _script key flags {flags:#x}")));
            }
            let source = crate::painless_score::PainlessSort::decode(&mut c)?;
            keys.push(SortField::with_source(
                "_script",
                Arc::new(source),
                flags & SORT_REVERSE != 0,
            ));
            lucene_geo.push(false);
            continue;
        }
        if matches!(ty, SORT_SHARD_DOC | SORT_HALF_FLOAT | SORT_UNSIGNED_LONG) {
            keys.push(decode_opensearch_sort(&mut c, ty, flags)?);
            lucene_geo.push(false);
            continue;
        }
        let ty = match ty {
            SORT_SCORE => SortType::Score,
            SORT_DOC => SortType::Doc,
            SORT_LONG => SortType::Long,
            SORT_INT => SortType::Int,
            SORT_DOUBLE => SortType::Double,
            SORT_FLOAT => SortType::Float,
            SORT_STRING => SortType::String,
            SORT_GEO_DISTANCE => {
                let (key, lucene) = decode_geo_sort(&mut c, flags)?;
                keys.push(key);
                lucene_geo.push(lucene);
                continue;
            }
            other => return Err(bad(format!("sort blob: unknown key type {other}"))),
        };
        let (field, missing) = match ty {
            SortType::Score | SortType::Doc => (String::new(), 0),
            _ => {
                let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
                let missing = c.i64()?;
                if ty == SortType::String && !(0..=1).contains(&missing) {
                    return Err(bad(format!(
                        "sort blob: keyword missing value {missing}, want 0 or 1"
                    )));
                }
                (field.to_string(), missing)
            }
        };
        let numeric = matches!(
            ty,
            SortType::Long | SortType::Int | SortType::Double | SortType::Float
        );
        let selector = match mode {
            0 if flags & SORT_MAX != 0 => Selector::Max,
            0 => Selector::Min,
            _ if !numeric || flags & SORT_MAX != 0 => {
                return Err(bad(format!(
                    "sort blob: flags {flags:#x} name a mode for a non-numeric key or with max"
                )));
            }
            SORT_MODE_SUM => Selector::Sum,
            SORT_MODE_AVG => Selector::Avg,
            SORT_MODE_MEDIAN => Selector::Median,
            other => return Err(bad(format!("sort blob: unknown mode {other:#x}"))),
        };
        let nested = if flags & SORT_NESTED != 0 {
            // `MultiValueMode.MEDIAN` has no nested pick.
            if !numeric || selector == Selector::Median {
                return Err(bad(format!(
                    "sort blob: flags {flags:#x} nest a key that cannot be nested"
                )));
            }
            let parents = boolean_of(&decode_query(c.bytes()?)?);
            let children = boolean_of(&decode_query(c.bytes()?)?);
            let max_children = c.i32()?;
            Some(Arc::new(NestedSort {
                parents,
                children,
                max_children,
            }))
        } else {
            None
        };
        keys.push(SortField {
            field,
            ty,
            reverse: flags & SORT_REVERSE != 0,
            selector,
            missing,
            nested,
            rewritten: None,
        });
        lucene_geo.push(false);
    }
    let after = match c.u8()? {
        0 => None,
        1 => {
            let doc = c.i32()?;
            let mut values = Vec::new();
            let mut terms = Vec::new();
            for (k, &lucene) in keys.iter().zip(&lucene_geo) {
                if lucene {
                    // `LatLonPointDistanceComparator.compareTop` compares
                    // metres, not the sort keys this side holds: the JVM
                    // runs such a page itself.
                    return Err(bad(
                        "sort blob: search_after on Lucene's geo-distance key".to_string()
                    ));
                }
                if k.ty != SortType::String {
                    values.push(c.i64()?);
                    terms.push(None);
                    continue;
                }
                values.push(0);
                terms.push(match c.u8()? {
                    0 => None,
                    1 => Some(c.bytes()?.to_vec()),
                    other => return Err(bad(format!("sort blob: term present is {other}"))),
                });
            }
            if keys.iter().all(|k| k.ty != SortType::String) {
                terms.clear();
            }
            Some(FieldDoc { doc, values, terms })
        }
        other => return Err(bad(format!("sort blob: has_after is {other}"))),
    };
    let options = c.u8()?;
    if options
        & !(SORT_TRACK_MAX_SCORE | SORT_TERMINATE_AFTER | SORT_COUNT_SEGMENTS | SORT_INDEX_SORTED)
        != 0
    {
        return Err(bad(format!("sort blob: unknown options {options:#x}")));
    }
    let count_segments = options & SORT_COUNT_SEGMENTS != 0;
    let terminate_after = if options & SORT_TERMINATE_AFTER != 0 {
        let n = c.i32()?;
        if n < 1 {
            return Err(bad(format!("sort blob: terminate_after {n}")));
        }
        Some((u64::from(n.unsigned_abs()), count_segments))
    } else if count_segments {
        return Err(bad(
            "sort blob: segment counts without terminate_after".to_string()
        ));
    } else {
        None
    };
    let slices = decode_slices(&mut c, &bad)?;
    // A sorted search's slices partition the segments
    // (`IndexSearcher.LeafSlice`) and their totals are summed: a segment in
    // two slices would count its hits twice, past the index's document count.
    let mut seen = std::collections::HashSet::new();
    if let Some(segment) = slices.iter().flatten().find(|&&s| !seen.insert(s)) {
        return Err(bad(format!(
            "sort blob: segment {segment} in more than one slice"
        )));
    }
    let mut index_sorted = Vec::new();
    if options & SORT_INDEX_SORTED != 0 {
        let n = c.length()?;
        for _ in 0..n {
            index_sorted.push(match c.u8()? {
                0 => None,
                at @ 1..=127 => Some(usize::from(at - 1)),
                other => return Err(bad(format!("sort blob: index-sort flag {other}"))),
            });
        }
    }
    if c.pos != blob.len() {
        return Err(bad(format!(
            "sort blob: {} trailing bytes",
            blob.len() - c.pos
        )));
    }
    Ok((
        keys,
        after,
        options & SORT_TRACK_MAX_SCORE != 0,
        slices,
        terminate_after,
        index_sorted,
    ))
}

/// The flags a geo-distance key may carry: [`SORT_REVERSE`], [`SORT_MAX`]
/// and a mode.
pub const GEO_SORT_FLAGS: u8 = SORT_REVERSE | SORT_MAX | SORT_MODE_MASK;

/// A [`SORT_GEO_DISTANCE`] key, after its type and flags: the field
/// (`len: i32`, UTF-8), the missing value (`i64`, unused: missing documents
/// sort at `+Infinity` in both comparators), `comparator: u8`, `count: i32`
/// and `count` origins (`lat`, `lon`: `f64`), and `unit_meters: f64`.
///
/// | comparator | Java | value |
/// |---|---|---|
/// | `0` | `LatLonPointSortField` (`LatLonDocValuesField.newDistanceSort`): one origin, metres, ascending, no mode | the haversine sort key (`haversin2` of it is the distance) |
/// | `1` | OpenSearch's `GeoDistanceSortBuilder` comparator ([`OpenSearchGeoDistanceSort`](lucene_search::document::geo::OpenSearchGeoDistanceSort)): `ARC`, any origins, unit, mode (flags as a numeric key's) and order | the distance in the unit |
///
/// Each value crosses as `double_to_sortable_long` of the double. Returns
/// the key, and whether it is Lucene's comparator (which takes no
/// `search_after`: its `compareTop` compares metres, not the keys).
/// A key of one of OpenSearch's own comparator sources: [`SORT_SHARD_DOC`],
/// [`SORT_HALF_FLOAT`] or [`SORT_UNSIGNED_LONG`], after its type and flags.
pub fn decode_opensearch_sort(c: &mut Cursor<'_>, ty: u8, flags: u8) -> Result<SortField, Status> {
    use lucene_search::top_field::{HalfFloatSort, Selector, ShardDocSort, UnsignedLongSort};
    let reverse = flags & SORT_REVERSE != 0;
    if ty == SORT_SHARD_DOC {
        if flags & !SORT_REVERSE != 0 {
            return Err(invalid(format!(
                "sort blob: _shard_doc key flags {flags:#x}"
            )));
        }
        let prefix = c.i64()?;
        return Ok(SortField::with_source(
            "_shard_doc",
            Arc::new(ShardDocSort { prefix }),
            reverse,
        ));
    }
    let selector = match (flags & SORT_MODE_MASK, flags & SORT_MAX != 0) {
        (0, false) => Selector::Min,
        (0, true) => Selector::Max,
        (SORT_MODE_SUM, false) => Selector::Sum,
        (SORT_MODE_AVG, false) => Selector::Avg,
        (SORT_MODE_MEDIAN, false) if ty == SORT_HALF_FLOAT => Selector::Median,
        _ => return Err(invalid(format!("sort blob: key {ty} flags {flags:#x}"))),
    };
    if flags & SORT_NESTED != 0 {
        return Err(invalid(format!("sort blob: key {ty} flags {flags:#x}")));
    }
    let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
    let missing = c.i64()?;
    Ok(if ty == SORT_HALF_FLOAT {
        let bits = i32::try_from(missing)
            .map_err(|_| invalid(format!("sort blob: half_float missing value {missing}")))?;
        let missing = lucene_util::numeric_utils::sortable_int_to_float(bits);
        SortField::with_source(
            field,
            Arc::new(HalfFloatSort { selector, missing }),
            reverse,
        )
    } else {
        SortField::with_source(
            field,
            Arc::new(UnsignedLongSort { selector, missing }),
            reverse,
        )
    })
}

pub fn decode_geo_sort(c: &mut Cursor<'_>, flags: u8) -> Result<(SortField, bool), Status> {
    use lucene_search::document::geo::{
        DistanceMode, LatLonPointSortField, OpenSearchGeoDistanceSort,
    };
    if flags & !GEO_SORT_FLAGS != 0 {
        return Err(invalid(format!(
            "sort blob: geo-distance key flags {flags:#x}"
        )));
    }
    let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
    let _missing = c.i64()?;
    let comparator = c.u8()?;
    let count = c.length()?;
    if count == 0 || count > MAX_GEO_ORIGINS {
        return Err(invalid(format!(
            "sort blob: {count} geo-distance origins, want 1..={MAX_GEO_ORIGINS}"
        )));
    }
    let mut origins = Vec::new();
    for _ in 0..count {
        let (lat, lon) = (c.f64()?, c.f64()?);
        if !lat.is_finite() || !lon.is_finite() {
            return Err(invalid(format!(
                "sort blob: geo-distance origin ({lat}, {lon})"
            )));
        }
        origins.push((lat, lon));
    }
    let unit_meters = c.f64()?;
    if !unit_meters.is_finite() || unit_meters <= 0.0 {
        return Err(invalid(format!(
            "sort blob: geo-distance unit of {unit_meters} metres"
        )));
    }
    let reverse = flags & SORT_REVERSE != 0;
    let source: std::sync::Arc<dyn lucene_search::top_field::FieldComparatorSource> =
        match comparator {
            0 => {
                #[allow(clippy::float_cmp)]
                let metres = unit_meters == 1.0;
                if count != 1 || !metres || flags != 0 {
                    return Err(invalid(
                    "sort blob: Lucene's geo-distance key takes one origin, metres, ascending and no mode"
                        .to_string(),
                ));
                }
                let (lat, lon) = origins[0];
                LatLonPointSortField::new(field, lat, lon)
                    .map_err(|e| invalid(format!("sort blob: {e}")))?
                    .comparator_source()
            }
            1 => {
                let mode = match flags & SORT_MODE_MASK {
                    0 if flags & SORT_MAX != 0 => DistanceMode::Max,
                    0 => DistanceMode::Min,
                    _ if flags & SORT_MAX != 0 => {
                        return Err(invalid(format!("sort blob: geo-distance flags {flags:#x}")))
                    }
                    SORT_MODE_SUM => DistanceMode::Sum,
                    SORT_MODE_AVG => DistanceMode::Avg,
                    _ => DistanceMode::Median,
                };
                std::sync::Arc::new(OpenSearchGeoDistanceSort {
                    origins,
                    unit_meters,
                    mode,
                })
            }
            other => {
                return Err(invalid(format!(
                    "sort blob: geo-distance comparator {other} (want 0 or 1)"
                )))
            }
        };
    Ok((
        SortField::with_source(field, source, reverse),
        comparator == 0,
    ))
}

/// Runs the query blob `query` sorted by the sort blob `sort`
/// ([`decode_sort`]) -- Lucene's `TopFieldCollectorManager(sort, top_n,
/// after, count_limit)` -- writing up to `top_n` hits' global doc ids into
/// `out_docs` and their sort values, `keys` per hit in key order, into
/// `out_values` (a keyword key's value there is 0), and the keyword keys'
/// terms into `out_terms`: per hit, per keyword key in key order, a
/// little-endian `i32` length (`-1` for a missing value) and the bytes.
/// `out_terms_len` receives their length; when that exceeds `terms_cap`
/// nothing is written but it, and the call returns
/// [`Status::BufferTooSmall`] for the caller to retry with room.
///
/// Total hits as `ffi_jvm_reader_search` reports them; `top_n` must be at
/// least 1 (a `size: 0` search has no order to keep).
///
/// # Safety
/// `query`/`sort` must be valid for `query_len`/`sort_len` bytes;
/// `out_docs` for `buf_len` elements and `out_values` for `buf_len * keys`
/// (`keys` being the sort's key count), with `buf_len >= top_n`; `out_terms`
/// for `terms_cap` bytes (it may be null when `terms_cap` is 0);
/// A sorted search's answer, ready to hand back.
pub struct SortedOut {
    /// The sort's key count.
    pub keys: usize,
    pub hits: Vec<FieldDoc>,
    pub total: i64,
    pub lower_bound: bool,
    /// [`encode_terms`] of `hits`.
    pub terms: Vec<u8>,
    /// Whether any key is a keyword key (the terms are then an answer,
    /// empty or not).
    pub has_terms: bool,
    /// The max score over every match when tracked, else `NaN`.
    pub max_score: f32,
    /// Whether `terminate_after` ended the search early.
    pub terminated: bool,
}

/// `ffi_jvm_reader_search_sorted` up to its output buffers: the blobs
/// decoded, the search run, the terms encoded (`ffm_bridge` hands them back
/// as a buffer of their exact length).
pub fn search_sorted_blobs(
    h: &JvmReaderHandle,
    query_blob: &[u8],
    sort_blob: &[u8],
    top_n: usize,
    count_limit: i64,
) -> Result<SortedOut, Status> {
    let (query, min_score, sim) = decode_request(query_blob)?;
    let (keys, after, track, slices, terminate_after, index_sorted) = decode_sort(sort_blob)?;
    if min_score.is_some() && terminate_after.is_some() {
        set_last_error("ffi_jvm_reader_search_sorted: min_score is not run behind terminate_after");
        return Err(Status::InvalidArgument);
    }
    let query = similarity_checked(sim.as_ref(), crate::join::resolve(h, query, sim.as_ref()))?;
    let (hits, total, lower_bound, max_score, terminated) = similarity_checked(
        sim.as_ref(),
        search_sorted(
            h,
            &query,
            &keys,
            after.as_ref(),
            top_n,
            count_limit,
            track,
            &slices,
            terminate_after,
            &index_sorted,
            min_score,
            sim.as_ref(),
        ),
    )?;
    let terms = encode_terms(&keys, &hits)?;
    Ok(SortedOut {
        has_terms: keys.iter().any(|k| k.ty == SortType::String),
        max_score,
        keys: keys.len(),
        hits,
        total,
        lower_bound,
        terms,
        terminated,
    })
}

/// Collapse-blob kinds: OpenSearch's `CollapsingDocValuesSource.Numeric`
/// and `.Keyword`.
pub const COLLAPSE_NUMERIC: u8 = 0;
pub const COLLAPSE_KEYWORD: u8 = 1;
/// Collapse values in a collapsed search's answer: none (the group of the
/// documents without one), a long, a term.
pub const COLLAPSE_VALUE_MISSING: u8 = 0;
pub const COLLAPSE_VALUE_LONG: u8 = 1;
pub const COLLAPSE_VALUE_BYTES: u8 = 2;

/// Decodes a collapse blob: the field (`len: i32`, UTF-8) and `kind: u8`
/// ([`COLLAPSE_NUMERIC`], [`COLLAPSE_KEYWORD`]). Trailing bytes are an error.
pub fn decode_collapse(blob: &[u8]) -> Result<(String, CollapseKind), Status> {
    let mut c = Cursor { buf: blob, pos: 0 };
    let field = std::str::from_utf8(c.bytes()?)
        .map_err(|_| Status::InvalidUtf8)?
        .to_string();
    let kind = match c.u8()? {
        COLLAPSE_NUMERIC => CollapseKind::Numeric,
        COLLAPSE_KEYWORD => CollapseKind::Keyword,
        other => {
            set_last_error(format!("collapse blob: unknown kind {other}"));
            return Err(Status::InvalidArgument);
        }
    };
    if c.pos != blob.len() {
        set_last_error(format!(
            "collapse blob: {} trailing bytes",
            blob.len() - c.pos
        ));
        return Err(Status::InvalidArgument);
    }
    Ok((field, kind))
}

/// A collapsed search (OpenSearch's `collapse`): the query blob's matches
/// through `CollapsingTopDocsCollector` by the sort blob's keys
/// ([`decode_sort`]: its `search_after`, `track_scores` option and slices,
/// or `terminate_after` over one slice) on the collapse blob's field
/// ([`decode_collapse`]), keeping `top_n` groups per slice
/// ([`lucene_search::top_field::search_collapsed`]). Encoded, little-endian:
/// `slices: i32`, then per slice `total: i64`, `max_score: i32` (the float's
/// bits, `NaN` untracked), `hits: i32` and per hit `doc: i32`, each key's
/// value -- a keyword key's term as `len: i32` (`-1` for none) and the bytes,
/// any other key's comparable `i64` (a score as its float bits) -- and the
/// group's value: `kind: u8` ([`COLLAPSE_VALUE_MISSING`],
/// [`COLLAPSE_VALUE_LONG`] and an `i64`, [`COLLAPSE_VALUE_BYTES`] and
/// `len: i32` and the bytes); last `terminated: u8`.
pub fn search_collapsed_blobs(
    h: &JvmReaderHandle,
    query_blob: &[u8],
    sort_blob: &[u8],
    collapse_blob: &[u8],
    top_n: usize,
) -> Result<Vec<u8>, Status> {
    let (query, min_score, sim) = decode_request(query_blob)?;
    let sim = sim.as_ref();
    let (keys, after, track, slices, terminate_after, _) = decode_sort(sort_blob)?;
    let (field, kind) = decode_collapse(collapse_blob)?;
    if let Some((_, count_segments)) = terminate_after {
        // OpenSearch runs a terminate_after search sequentially, one
        // collector; the size-0 count is not a collapsed search's.
        if count_segments || slices.len() > 1 || min_score.is_some() {
            set_last_error(
                "collapse: terminate_after runs on one slice, without min_score or segment counts",
            );
            return Err(Status::InvalidArgument);
        }
    }
    if top_n == 0 {
        set_last_error("collapse: topN must be at least 1");
        return Err(Status::InvalidArgument);
    }
    let query = similarity_checked(sim, crate::join::resolve(h, query, sim))?;
    let mut opened = h.reader.open_segments().map_err(|e| {
        set_last_error(format!("opening segment postings: {e}"));
        Status::Decode
    })?;
    if query_uses_points(&query) {
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
            s.similarity = sim.map(|p| p as &dyn Similarity);
            s
        })
        .collect();
    let q = match query {
        JvmQuery::Term(t) => BooleanQuery {
            must: vec![Clause::Term(t)],
            ..Default::default()
        },
        JvmQuery::Boolean(b) => b,
    };
    let needs_scores = track || min_score.is_some() || keys.iter().any(|k| k.ty == SortType::Score);
    let fields: Vec<String> = if needs_scores {
        crate::clauses::clause_field_names(&q)
            .into_iter()
            .map(str::to_string)
            .collect()
    } else {
        Vec::new()
    };
    let owned = h
        .reader
        .field_norms_by_field(&fields)
        .map_err(|error| map_search_error(error.into()))?;
    let norms: Vec<Option<&std::collections::HashMap<String, FieldNorms<'_>>>> =
        owned.iter().map(|m| (!m.is_empty()).then_some(m)).collect();
    if let Some((n, _)) = terminate_after {
        let top_cut = lucene_search::top_field::search_collapsed_until(
            &segments,
            h.reader.segment_readers(),
            &q,
            &norms,
            &keys,
            &field,
            kind,
            top_n,
            after.as_ref(),
            track,
            n,
        )
        .map_err(map_search_error);
        let (top, cut) = similarity_checked(sim, top_cut)?;
        return encode_collapsed(&keys, &[top], cut.terminated);
    }
    let parts = lucene_search::top_field::search_collapsed(
        &segments,
        h.reader.segment_readers(),
        &q,
        &norms,
        &keys,
        &field,
        kind,
        top_n,
        after.as_ref(),
        track,
        &slices,
        min_score,
    )
    .map_err(map_search_error);
    let parts = similarity_checked(sim, parts)?;
    encode_collapsed(&keys, &parts, false)
}

/// [`search_collapsed_blobs`]'s encoding of the slices' answers, then
/// `terminated: u8` (a `terminate_after` search that ended early).
pub fn encode_collapsed(
    keys: &[SortField],
    parts: &[CollapseTopDocs],
    terminated: bool,
) -> Result<Vec<u8>, Status> {
    let too_big = |what: &str| {
        set_last_error(format!("collapse result: {what} past i32"));
        Status::InvalidArgument
    };
    let len = |n: usize, what: &str| i32::try_from(n).map_err(|_| too_big(what));
    let mut out = Vec::new();
    out.extend_from_slice(&len(parts.len(), "slice count")?.to_le_bytes());
    for p in parts {
        out.extend_from_slice(&i64::try_from(p.total).unwrap_or(i64::MAX).to_le_bytes());
        out.extend_from_slice(&p.max_score.to_bits().to_le_bytes());
        out.extend_from_slice(&len(p.hits.len(), "hit count")?.to_le_bytes());
        for (hit, value) in p.hits.iter().zip(&p.values) {
            out.extend_from_slice(&hit.doc.to_le_bytes());
            for (k, key) in keys.iter().enumerate() {
                if key.ty != SortType::String {
                    out.extend_from_slice(&hit.values.get(k).copied().unwrap_or(0).to_le_bytes());
                    continue;
                }
                match hit.terms.get(k).and_then(Option::as_ref) {
                    None => out.extend_from_slice(&(-1i32).to_le_bytes()),
                    Some(t) => {
                        out.extend_from_slice(&len(t.len(), "a sort term")?.to_le_bytes());
                        out.extend_from_slice(t);
                    }
                }
            }
            match value {
                CollapseValue::Missing => out.push(COLLAPSE_VALUE_MISSING),
                CollapseValue::Long(v) => {
                    out.push(COLLAPSE_VALUE_LONG);
                    out.extend_from_slice(&v.to_le_bytes());
                }
                CollapseValue::Bytes(b) => {
                    out.push(COLLAPSE_VALUE_BYTES);
                    out.extend_from_slice(&len(b.len(), "a collapse term")?.to_le_bytes());
                    out.extend_from_slice(b);
                }
            }
        }
    }
    out.push(u8::from(terminated));
    Ok(out)
}

/// OpenSearch's `QueryRescorer` over a search's hits
/// ([`lucene_search::rescorer::opensearch_query_rescore`]): `docs`/`scores`
/// (global ids, best first) rescored in place by the query blob's query
/// over the top `window`, `mode` being the `QueryRescoreMode` ordinal (`avg`,
/// `max`, `min`, `total`, `multiply`). Returns how many hits were rescored.
#[allow(clippy::too_many_arguments)]
pub fn rescore_hits(
    h: &JvmReaderHandle,
    query_blob: &[u8],
    window: usize,
    query_weight: f32,
    rescore_query_weight: f32,
    mode: i32,
    docs: &mut [i32],
    scores: &mut [f32],
) -> Result<usize, Status> {
    use lucene_search::rescorer::{opensearch_query_rescore, QueryRescoreMode};
    let mode = match mode {
        0 => QueryRescoreMode::Avg,
        1 => QueryRescoreMode::Max,
        2 => QueryRescoreMode::Min,
        3 => QueryRescoreMode::Total,
        4 => QueryRescoreMode::Multiply,
        other => {
            set_last_error(format!("rescore: unknown score mode {other}"));
            return Err(Status::InvalidArgument);
        }
    };
    if docs.len() != scores.len() {
        set_last_error("rescore: as many scores as documents");
        return Err(Status::InvalidArgument);
    }
    let (query, min_score, sim) = decode_request(query_blob)?;
    let sim = sim.as_ref();
    if min_score.is_some() {
        set_last_error("rescore: a rescore query has no min_score");
        return Err(Status::InvalidArgument);
    }
    let query = similarity_checked(sim, crate::join::resolve(h, query, sim))?;
    let mut opened = h.reader.open_segments().map_err(|e| {
        set_last_error(format!("opening segment postings: {e}"));
        Status::Decode
    })?;
    if query_uses_points(&query) {
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
            s.similarity = sim.map(|p| p as &dyn Similarity);
            s
        })
        .collect();
    let q = match query {
        JvmQuery::Term(t) => BooleanQuery {
            must: vec![Clause::Term(t)],
            ..Default::default()
        },
        JvmQuery::Boolean(b) => b,
    };
    let fields: Vec<String> = crate::clauses::clause_field_names(&q)
        .into_iter()
        .map(str::to_string)
        .collect();
    let owned = h
        .reader
        .field_norms_by_field(&fields)
        .map_err(|error| map_search_error(error.into()))?;
    let norms: Vec<Option<&std::collections::HashMap<String, FieldNorms<'_>>>> =
        owned.iter().map(|m| (!m.is_empty()).then_some(m)).collect();
    let searcher = lucene_search::index_searcher::IndexSearcher::new(&segments, &norms)
        .map_err(map_search_error)?;
    let ctx = lucene_search::values_source::ValuesContext::new(&searcher);
    let mut hits: Vec<lucene_search::top_docs::ShardScoreDoc> = docs
        .iter()
        .zip(scores.iter())
        .map(|(&d, &s)| lucene_search::top_docs::ShardScoreDoc::new(d, s))
        .collect();
    let rescored = opensearch_query_rescore(
        &ctx,
        &mut hits,
        q,
        window,
        query_weight,
        rescore_query_weight,
        mode,
    )
    .map_err(map_search_error);
    let rescored = similarity_checked(sim, rescored)?;
    for (i, hit) in hits.iter().enumerate() {
        docs[i] = hit.doc;
        scores[i] = hit.score;
    }
    Ok(rescored)
}

/// Value kinds in a metrics blob ([`decode_metrics`]).
pub const METRIC_LONG: u8 = 0;
pub const METRIC_DOUBLE: u8 = 1;
pub const METRIC_FLOAT: u8 = 2;
/// Sources in a metrics blob ([`lucene_aggs::metric_aggs::Source`]).
pub const METRIC_DOC_VALUES: u8 = 0;
pub const METRIC_POINTS_MIN: u8 = 1;
pub const METRIC_POINTS_MAX: u8 = 2;
/// At most this many fields in one metrics blob.
pub const MAX_METRICS: usize = 64;
/// At most this many slices, and segments across them, in one metrics blob.
pub const MAX_SLICES: usize = 4096;
pub const MAX_SLICED_SEGMENTS: usize = 1 << 16;
/// Doubles per field in `ffi_jvm_reader_aggregate`'s output.
pub const METRIC_VALUES: usize = 6;

/// Decodes a metrics blob: `count: u8`, then per field `kind: u8`
/// ([`METRIC_LONG`], [`METRIC_DOUBLE`], [`METRIC_FLOAT`]), `source: u8`
/// ([`METRIC_DOC_VALUES`], [`METRIC_POINTS_MIN`], [`METRIC_POINTS_MAX`]),
/// `needs: u8` (the parts read back, [`lucene_aggs::metric_aggs::NEED_ALL`]'s bits,
/// at least one) and the field (`len: i32`, UTF-8); then `count: u8` `terms`
/// aggregations, each its field (`len: i32`, UTF-8) and `shard_size: i32` (at
/// least 1) -- at least one aggregation in all; then the concurrent-search
/// slices ([`decode_slices`]). Little-endian, and trailing bytes are an error.
pub fn decode_metrics(blob: &[u8]) -> Result<DecodedAggs, Status> {
    let mut c = Cursor { buf: blob, pos: 0 };
    let bad = |msg: String| {
        set_last_error(msg);
        Status::InvalidArgument
    };
    let n = usize::from(c.u8()?);
    if n > MAX_METRICS {
        return Err(bad(format!(
            "metrics blob: {n} fields, want at most {MAX_METRICS}"
        )));
    }
    let mut specs = Vec::new();
    for _ in 0..n {
        let kind = match c.u8()? {
            METRIC_LONG => ValueKind::Long,
            METRIC_DOUBLE => ValueKind::Double,
            METRIC_FLOAT => ValueKind::Float,
            other => return Err(bad(format!("metrics blob: unknown kind {other}"))),
        };
        let source = match c.u8()? {
            METRIC_DOC_VALUES => Source::DocValues,
            METRIC_POINTS_MIN => Source::PointsMin,
            METRIC_POINTS_MAX => Source::PointsMax,
            other => return Err(bad(format!("metrics blob: unknown source {other}"))),
        };
        let needs = c.u8()?;
        if needs == 0 || needs & !NEED_ALL != 0 {
            return Err(bad(format!("metrics blob: needs {needs:#x}")));
        }
        let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
        specs.push(MetricSpec {
            field: field.to_string(),
            kind,
            source,
            needs,
        });
    }
    let t = usize::from(c.u8()?);
    if t > MAX_METRICS || n + t == 0 {
        return Err(bad(format!(
            "metrics blob: {n} fields and {t} terms, want 1..={MAX_METRICS} of each"
        )));
    }
    let mut terms = Vec::new();
    for _ in 0..t {
        let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
        let shard_size = c.i32()?;
        let shard_size = usize::try_from(shard_size)
            .ok()
            .filter(|&n| n >= 1)
            .ok_or_else(|| {
                bad(format!(
                    "metrics blob: shard_size {shard_size}, want at least 1"
                ))
            })?;
        terms.push(TermsSpec {
            field: field.to_string(),
            shard_size,
        });
    }
    // The concurrent-search slices; none means one slice over every segment.
    let slices = decode_slices(&mut c, &bad)?;
    if c.pos != blob.len() {
        return Err(bad(format!(
            "metrics blob: {} trailing bytes",
            blob.len() - c.pos
        )));
    }
    Ok((specs, terms, slices))
}

/// A decoded aggregations blob: the metrics, the `terms`, the slices.
pub type DecodedAggs = (Vec<MetricSpec>, Vec<TermsSpec>, Vec<Vec<usize>>);

/// The aggregations of an aggregations blob ([`decode_metrics`]) over the
/// live matches of the query blob `query`, in one pass: per metric field,
/// its value count into `out_counts` and [`METRIC_VALUES`] doubles into
/// `out_values` -- the compensated sum and its delta, the minimum and maximum
/// over every value, and the minimum of each document's first value and the
/// maximum of each document's last (see [`lucene_aggs::metric_aggs`]); and the
/// `terms` results, encoded ([`encode_terms_results`]; slice by slice, each
/// slice's aggregations in blob order), into `out_terms`, their length into
/// `out_terms_len` either way (a too-small `out_terms` is
/// [`Status::BufferTooSmall`], so a caller can size a second call).
///
/// # Safety
/// `query`/`aggs` must be valid for `query_len`/`aggs_len` bytes;
/// The `terms` results' encoding: per slice `other_doc_count: i64`,
/// `buckets: i32`, and per bucket, by term ascending, `doc_count: i64` and the
/// term (`len: i32`, bytes).
pub fn encode_terms_results(
    results: &[lucene_aggs::terms_agg::TermsResult],
) -> Result<Vec<u8>, Status> {
    let too_big = || {
        set_last_error("terms result: a term or bucket count past i32".to_string());
        Status::InvalidArgument
    };
    let mut out = Vec::new();
    for r in results {
        out.extend_from_slice(
            &i64::try_from(r.other_doc_count)
                .unwrap_or(i64::MAX)
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &i32::try_from(r.buckets.len())
                .map_err(|_| too_big())?
                .to_le_bytes(),
        );
        for (term, count) in &r.buckets {
            out.extend_from_slice(&i64::try_from(*count).unwrap_or(i64::MAX).to_le_bytes());
            out.extend_from_slice(
                &i32::try_from(term.len())
                    .map_err(|_| too_big())?
                    .to_le_bytes(),
            );
            out.extend_from_slice(term);
        }
    }
    Ok(out)
}

/// A field's doubles, in `ffi_jvm_reader_aggregate`'s order.
pub fn metric_values(s: &MetricState) -> [f64; METRIC_VALUES] {
    [s.sum, s.delta, s.min, s.max, s.min_of_mins, s.max_of_maxes]
}

/// `ffi_jvm_reader_aggregate` up to its output buffers.
pub fn aggregate_blobs(
    h: &JvmReaderHandle,
    query_blob: &[u8],
    aggs_blob: &[u8],
) -> Result<(Vec<MetricState>, Vec<u8>), Status> {
    aggregate_counting_blobs(h, query_blob, aggs_blob, -1).map(|(s, t, _)| (s, t))
}

/// [`aggregate_blobs`], and with a positive `count_limit` the `size: 0`
/// search's `(total, is_lower_bound)` for the same query, as
/// `ffi_jvm_reader_search` would count it -- from the matches the
/// aggregations already visited, as Lucene's `MultiCollector` counts them in
/// the same pass. Any other `count_limit` counts nothing (`None`).
#[allow(clippy::type_complexity)]
pub fn aggregate_counting_blobs(
    h: &JvmReaderHandle,
    query_blob: &[u8],
    aggs_blob: &[u8],
    count_limit: i64,
) -> Result<(Vec<MetricState>, Vec<u8>, Option<(i64, bool)>), Status> {
    aggregate_counting_blobs_seen(h, query_blob, aggs_blob, count_limit)
        .map(|(s, t, total, _)| (s, t, total))
}

/// Whether a terms aggregation counts buffered documents' keyword runs in
/// place; a test turns it off to compare with their encoded doc values.
pub fn count_runs_in_place() -> bool {
    #[cfg(test)]
    {
        !test_support::ENCODED_RUNS.with(std::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        true
    }
}

/// [`aggregate_counting_blobs`], also handing back each segment's live
/// match count where the aggregations' pass counted it (`None` for a
/// segment whose aggregations were answered without visiting its matches):
/// what a `size: 0` search's count collector would iterate there, so the
/// caller can tell whether that collector stops early (`terminated_early`,
/// [`lucene_search::terminate::count_terminates`]) without running the
/// query again.
#[allow(clippy::type_complexity)]
pub fn aggregate_counting_blobs_seen(
    h: &JvmReaderHandle,
    query_blob: &[u8],
    aggs_blob: &[u8],
    count_limit: i64,
) -> Result<
    (
        Vec<MetricState>,
        Vec<u8>,
        Option<(i64, bool)>,
        Vec<Option<u64>>,
    ),
    Status,
> {
    let (query, min_score, sim) = decode_request(query_blob)?;
    let sim = sim.as_ref();
    let (specs, terms, slices) = decode_metrics(aggs_blob)?;
    let query = similarity_checked(sim, crate::join::resolve(h, query, sim))?;
    let mut opened = h.reader.open_segments().map_err(|e| {
        set_last_error(format!("opening segment postings: {e}"));
        Status::Decode
    })?;
    if query_uses_points(&query) || specs.iter().any(|s| s.source != Source::DocValues) {
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
            s.similarity = sim.map(|p| p as &dyn Similarity);
            s
        })
        .collect();
    let q = match &query {
        JvmQuery::Term(t) => BooleanQuery {
            must: vec![Clause::Term(t.clone())],
            ..Default::default()
        },
        JvmQuery::Boolean(b) => b.clone(),
    };
    let readers = h.reader.segment_readers();
    let slices = if slices.is_empty() {
        vec![(0..segments.len().min(readers.len())).collect()]
    } else {
        slices
    };
    // Each terms field's global ordinals, built once per reader and kept.
    let globals = terms
        .iter()
        .map(|t| {
            if count_runs_in_place() {
                h.reader.global_ords_skipping_runs(&t.field)
            } else {
                h.reader.global_ords(&t.field)
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(map_search_error)?;
    // min_score scores the query: its fields' norms.
    let fields: Vec<String> = if min_score.is_some() {
        crate::clauses::clause_field_names(&q)
            .into_iter()
            .map(str::to_string)
            .collect()
    } else {
        Vec::new()
    };
    let owned = h
        .reader
        .field_norms_by_field(&fields)
        .map_err(|error| map_search_error(error.into()))?;
    let norms: Vec<Option<&std::collections::HashMap<String, FieldNorms<'_>>>> =
        owned.iter().map(|m| (!m.is_empty()).then_some(m)).collect();
    let min = min_score.map(|min| lucene_search::segment_match::MinScore { min, norms: &norms });
    let sliced = lucene_aggs::metric_aggs::aggregate_sliced_counting(
        &segments,
        readers,
        &q,
        &specs,
        &terms,
        &globals,
        &slices,
        min.as_ref(),
    )
    .map_err(map_search_error);
    let (sliced, seen) = similarity_checked(sim, sliced)?;
    let mut states = Vec::new();
    let mut results = Vec::new();
    for (m, t) in sliced {
        states.extend(m);
        results.extend(t);
    }
    // Behind `min_score` the aggregations saw the passing documents, which
    // is what its count counts; otherwise every live match.
    let total = match (count_limit > 0, min_score) {
        (false, _) => None,
        (true, None) => Some(total_hits_with(h, &segments, &query, count_limit, &seen)?),
        (true, Some(_)) => min_score_total(&seen, count_limit),
    };
    Ok((states, encode_terms_results(&results)?, total, seen))
}

/// A decoded count blob: the limit, the iterate flags, the slices.
pub type CountSpec = (u64, Vec<bool>, Vec<Vec<usize>>);

/// Decodes a count-terminates blob: `n: i32` (at least 1), the segments'
/// iterate flags (`count: i32`, then one `u8` each: 1 where `Weight.count`
/// gave `-1`), then the slices ([`decode_slices`]).
pub fn decode_count_spec(blob: &[u8]) -> Result<CountSpec, Status> {
    let mut c = Cursor { buf: blob, pos: 0 };
    let bad = |msg: String| {
        set_last_error(msg);
        Status::InvalidArgument
    };
    let n = c.i32()?;
    if n < 1 {
        return Err(bad(format!("count blob: n {n}")));
    }
    let count = c.length()?;
    let mut iterate = Vec::new();
    for _ in 0..count {
        iterate.push(match c.u8()? {
            0 => false,
            1 => true,
            other => return Err(bad(format!("count blob: iterate flag {other}"))),
        });
    }
    let slices = decode_slices(&mut c, &bad)?;
    if c.pos != blob.len() {
        return Err(bad(format!(
            "count blob: {} trailing bytes",
            blob.len() - c.pos
        )));
    }
    Ok((u64::from(n.unsigned_abs()), iterate, slices))
}

/// `IndexReader.docFreq(new Term(field, term))` over the handle's segments:
/// OpenSearch's total-hits shortcut for a term query over a reader with no
/// deletions (`TopDocsCollectorContext.shortcutTotalHitCount`), counted here
/// rather than by Lucene's own term dictionaries in the JVM -- the native
/// search reads the same dictionary blocks straight after.
pub fn doc_freq(h: &JvmReaderHandle, field: &[u8], term: &[u8]) -> Result<i64, Status> {
    let field = std::str::from_utf8(field).map_err(|_| {
        set_last_error("doc_freq: the field name is not UTF-8");
        Status::InvalidArgument
    })?;
    h.reader.doc_freq(field, term).map_err(|e| {
        set_last_error(format!("doc_freq: {e}"));
        Status::Decode
    })
}

/// `ffi_jvm_reader_count_terminates` up to its output.
pub fn count_terminates_blobs(
    h: &JvmReaderHandle,
    query_blob: &[u8],
    spec_blob: &[u8],
) -> Result<bool, Status> {
    let (query, min_score, sim) = decode_request(query_blob)?;
    let sim = sim.as_ref();
    let (n, iterate, slices) = decode_count_spec(spec_blob)?;
    let query = similarity_checked(sim, crate::join::resolve(h, query, sim))?;
    let mut opened = h.reader.open_segments().map_err(|e| {
        set_last_error(format!("opening segment postings: {e}"));
        Status::Decode
    })?;
    if query_uses_points(&query) {
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
            s.similarity = sim.map(|p| p as &dyn Similarity);
            s
        })
        .collect();
    let q = match &query {
        JvmQuery::Term(t) => BooleanQuery {
            must: vec![Clause::Term(t.clone())],
            ..Default::default()
        },
        JvmQuery::Boolean(b) => b.clone(),
    };
    let slices = if slices.is_empty() {
        vec![(0..segments.len()).collect()]
    } else {
        slices
    };
    let fields: Vec<String> = if min_score.is_some() {
        crate::clauses::clause_field_names(&q)
            .into_iter()
            .map(str::to_string)
            .collect()
    } else {
        Vec::new()
    };
    let owned = h
        .reader
        .field_norms_by_field(&fields)
        .map_err(|error| map_search_error(error.into()))?;
    let norms: Vec<Option<&std::collections::HashMap<String, FieldNorms<'_>>>> =
        owned.iter().map(|m| (!m.is_empty()).then_some(m)).collect();
    let min = min_score.map(|min| lucene_search::segment_match::MinScore { min, norms: &norms });
    let counted = lucene_search::terminate::count_terminates(
        &segments,
        &q,
        &slices,
        &iterate,
        n,
        min.as_ref(),
    )
    .map_err(map_search_error);
    similarity_checked(sim, counted)
}

/// The keyword keys' terms of `hits`, as `ffi_jvm_reader_search_sorted`
/// hands them back.
pub fn encode_terms(keys: &[SortField], hits: &[FieldDoc]) -> Result<Vec<u8>, Status> {
    let mut out = Vec::new();
    if keys.iter().all(|k| k.ty != SortType::String) {
        return Ok(out);
    }
    for hit in hits {
        for (k, key) in keys.iter().enumerate() {
            if key.ty != SortType::String {
                continue;
            }
            match hit.terms.get(k).and_then(Option::as_ref) {
                None => out.extend_from_slice(&(-1i32).to_le_bytes()),
                Some(t) => {
                    let len = i32::try_from(t.len()).map_err(|_| {
                        set_last_error(format!("a {}-byte sort term", t.len()));
                        Status::InvalidArgument
                    })?;
                    out.extend_from_slice(&len.to_le_bytes());
                    out.extend_from_slice(t);
                }
            }
        }
    }
    Ok(out)
}

/// The search behind `ffi_jvm_reader_search_sorted`.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn search_sorted(
    h: &JvmReaderHandle,
    query: &JvmQuery,
    keys: &[SortField],
    after: Option<&FieldDoc>,
    top_n: usize,
    count_limit: i64,
    track_max_score: bool,
    slices: &[Vec<usize>],
    terminate_after: Option<(u64, bool)>,
    index_sorted: &[Option<usize>],
    min_score: Option<f32>,
    sim: Option<&PerFieldSimilarity>,
) -> Result<(Vec<FieldDoc>, i64, bool, f32, bool), Status> {
    let mut opened = h.reader.open_segments().map_err(|e| {
        set_last_error(format!("opening segment postings: {e}"));
        Status::Decode
    })?;
    // A numeric key skips with its field's points, as `NumericComparator` does.
    let numeric_key = keys
        .iter()
        .any(|k| !matches!(k.ty, SortType::Score | SortType::Doc | SortType::String));
    if numeric_key || query_uses_points(query) {
        opened.open_points().map_err(|e| {
            set_last_error(format!("opening segment points: {e}"));
            Status::Decode
        })?;
    }
    let q = match query {
        JvmQuery::Term(t) => BooleanQuery {
            must: vec![Clause::Term(t.clone())],
            ..Default::default()
        },
        JvmQuery::Boolean(b) => b.clone(),
    };
    let mut segments: Vec<OpenSegment<'_>> = opened
        .as_open_segments()
        .into_iter()
        .zip(&h.live_docs)
        .map(|(mut s, live)| {
            s.live_docs = live.as_ref();
            s.similarity = sim.map(|p| p as &dyn Similarity);
            s
        })
        .collect();
    // The documents come in the sort's order where it begins the index
    // sort, or where it follows index sort keys the query pins to one
    // value each (`tenant = a`, sorted by the timestamp after it).
    for (s, at) in segments.iter_mut().zip(index_sorted) {
        s.index_sort_prefix = match *at {
            None => false,
            Some(0) => true,
            Some(p) => lucene_search::pinned_sort_keys(s, &q).map_err(map_search_error)? >= p,
        };
    }
    let needs_scores =
        track_max_score || min_score.is_some() || keys.iter().any(|k| k.ty == SortType::Score);
    let fields: Vec<String> = if needs_scores {
        crate::clauses::clause_field_names(&q)
            .into_iter()
            .map(str::to_string)
            .collect()
    } else {
        Vec::new()
    };
    let owned = h
        .reader
        .field_norms_by_field(&fields)
        .map_err(|error| map_search_error(error.into()))?;
    let norms: Vec<Option<&std::collections::HashMap<String, FieldNorms<'_>>>> =
        owned.iter().map(|m| (!m.is_empty()).then_some(m)).collect();
    // `i64::MAX` is Java's `Integer.MAX_VALUE` threshold: count exactly, and
    // so never prune (`TopFieldCollector`'s exhaustive score modes).
    let threshold = match count_limit {
        i64::MAX => u64::MAX,
        n => u64::try_from(n).unwrap_or(0),
    };
    // `terminate_after` (never concurrent): the top of the first n matches,
    // each counted, as OpenSearch's EarlyTerminatingCollector lets them through.
    if let Some((n, count_segments)) = terminate_after {
        // `size: 0` (the segment counts): no hits to find, only the cut.
        if count_segments {
            let cut = lucene_search::terminate::terminate_after(&segments, &q, n)
                .map_err(map_search_error)?;
            let counted = lucene_search::terminate::count_until(&segments, &q, &cut)
                .map_err(map_search_error)?
                .ok_or_else(|| {
                    set_last_error("terminate_after: no segment count for this query");
                    Status::InvalidArgument
                })?;
            let total = if count_limit <= 0 {
                -1
            } else {
                i64::try_from(counted).unwrap_or(i64::MAX)
            };
            return Ok((Vec::new(), total, false, f32::NAN, cut.terminated));
        }
        if lucene_search::terminate::cannot_reach(&segments, &q, n) {
            let top = lucene_search::top_field::search_sorted_min_score(
                &segments,
                h.reader.segment_readers(),
                &q,
                &norms,
                keys,
                top_n,
                u64::MAX,
                after,
                track_max_score,
                &[],
                None,
            )
            .map_err(map_search_error)?;
            let total = if count_limit <= 0 {
                -1
            } else {
                i64::try_from(top.total.value).unwrap_or(i64::MAX)
            };
            return Ok((top.hits, total, false, top.max_score, false));
        }
        let (top, cut) = lucene_search::terminate::search_sorted_until(
            &segments,
            h.reader.segment_readers(),
            &q,
            &norms,
            keys,
            top_n,
            after,
            track_max_score,
            n,
        )
        .map_err(map_search_error)?;
        let total = if count_limit <= 0 {
            -1
        } else {
            i64::try_from(cut.collected).unwrap_or(i64::MAX)
        };
        return Ok((top.hits, total, false, top.max_score, cut.terminated));
    }
    // A concurrent search's slices each get a collector, as Lucene's do --
    // except beside a tracked max score, which stays one pass.
    let top = lucene_search::top_field::search_sorted_min_score(
        &segments,
        h.reader.segment_readers(),
        &q,
        &norms,
        keys,
        top_n,
        threshold,
        after,
        track_max_score,
        slices,
        min_score,
    )
    .map_err(map_search_error)?;
    if count_limit <= 0 {
        return Ok((top.hits, -1, false, top.max_score, false));
    }
    Ok((
        top.hits,
        i64::try_from(top.total.value).unwrap_or(i64::MAX),
        top.total.relation == TotalHitsRelation::GreaterThanOrEqualTo,
        top.max_score,
        false,
    ))
}

/// The search behind `ffi_jvm_reader_search`, on an already-validated
/// handle: the reader's segments with the JVM's live docs swapped in.
pub fn search(
    h: &JvmReaderHandle,
    query: &JvmQuery,
    top_n: usize,
    count_limit: i64,
    sim: Option<&PerFieldSimilarity>,
) -> Result<(Vec<ScoreDoc>, i64, bool), Status> {
    let mut opened = h.reader.open_segments().map_err(|e| {
        set_last_error(format!("opening segment postings: {e}"));
        Status::Decode
    })?;
    if query_uses_points(query) {
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
            s.similarity = sim.map(|p| p as &dyn Similarity);
            s
        })
        .collect();

    // Lucene's `TopScoreDocCollectorManager(n, totalHitsThreshold)`: the
    // search itself counts, exactly up to the threshold, and may prune past
    // it. `u64::MAX` (an exact count) disables pruning, as Java's
    // `Integer.MAX_VALUE` does; no count at all lets it prune at once.
    let threshold = u64::try_from(count_limit).unwrap_or(0);
    let searched: Option<(Vec<ScoreDoc>, TotalHits)> = if top_n == 0 {
        None
    } else {
        Some(match query {
            JvmQuery::Term(q) => {
                let owned = h
                    .reader
                    .field_norms(&q.field)
                    .map_err(|error| map_search_error(error.into()))?;
                let norms: Vec<Option<&FieldNorms<'_>>> =
                    owned.iter().map(Option::as_ref).collect();
                search_term_query_multi_segment_counting(&segments, q, &norms, top_n, threshold)
                    .map_err(map_search_error)?
            }
            JvmQuery::Boolean(q) => {
                let fields: Vec<String> = crate::clauses::clause_field_names(q)
                    .into_iter()
                    .map(str::to_string)
                    .collect();
                let owned = h
                    .reader
                    .field_norms_by_field(&fields)
                    .map_err(|error| map_search_error(error.into()))?;
                let norms: Vec<Option<&std::collections::HashMap<String, FieldNorms<'_>>>> =
                    owned.iter().map(|m| (!m.is_empty()).then_some(m)).collect();
                search_boolean_query_multi_segment_maxscore_counting(
                    &segments, q, &norms, top_n, threshold,
                )
                .map_err(map_search_error)?
            }
        })
    };

    Ok(match searched {
        _ if count_limit <= 0 => (searched.map(|(h, _)| h).unwrap_or_default(), -1, false),
        Some((hits, total)) => {
            let value = i64::try_from(total.value).unwrap_or(i64::MAX);
            (
                hits,
                value,
                total.relation == TotalHitsRelation::GreaterThanOrEqualTo,
            )
        }
        // `size: 0`: no top hits to collect, so the count is its own pass.
        None => {
            let (total, lower_bound) = total_hits(h, &segments, query, count_limit)?;
            (Vec::new(), total, lower_bound)
        }
    })
}

/// At most this many sub-queries in a hybrid blob (neural-search allows 5).
pub const MAX_HYBRID_QUERIES: usize = 16;

/// neural-search's hybrid query shard phase: `HybridBulkScorer` feeding a
/// `HybridTopScoreDocCollector`, one collector per concurrent-search slice.
///
/// Per slice, the documents the collector counted and each sub-query's
/// `top_n` best documents (best first, ties by ascending doc ID). Stock's
/// count is not the union's: the bulk scorer scores each segment in windows
/// of [`HYBRID_WINDOW`] documents, and a document reaches the collector only
/// when some sub-query scores it above that sub-query's threshold at the
/// window's start -- 0 at each segment's start, then the best score the
/// slice's queue has turned away (`setMinCompetitiveScore` lets a scorer skip
/// what falls below it). [`HybridCollector`] replays exactly that.
///
/// The blob: `count: u8` sub-queries, each a request blob ([`decode_request`],
/// `len: i32` first) with no `min_score`; `top_n: i32`; the slices
/// ([`decode_slices`], none meaning one slice over every segment).
///
/// The answer, little-endian: `slices: i32`, then per slice `total: i64` and
/// per sub-query `hits: i32` and per hit `doc: i32` and `score` (`f32` bits).
pub fn search_hybrid_blobs(h: &JvmReaderHandle, blob: &[u8]) -> Result<Vec<u8>, Status> {
    let bad = |m: String| {
        set_last_error(format!("hybrid blob: {m}"));
        Status::InvalidArgument
    };
    let mut c = Cursor::new(blob);
    let count = usize::from(c.u8()?);
    if count == 0 || count > MAX_HYBRID_QUERIES {
        return Err(bad(format!(
            "{count} sub-queries, want 1 to {MAX_HYBRID_QUERIES}"
        )));
    }
    let mut subs = Vec::with_capacity(count); // alloc-ok: count is checked to be at most MAX_HYBRID_QUERIES
    let mut sim = None;
    for _ in 0..count {
        let (query, min_score, s) = decode_request(c.bytes()?)?;
        if min_score.is_some() {
            return Err(bad("a sub-query with min_score".into()));
        }
        // Every sub-query is scored with the searcher's one similarity.
        if s.is_some() {
            sim = s;
        }
        subs.push(boolean_of(&query));
    }
    let top_n = usize::try_from(c.i32()?).map_err(|_| bad("negative top_n".into()))?;
    if top_n == 0 {
        return Err(bad("top_n must be at least 1".into()));
    }
    let slices = decode_slices(&mut c, &bad)?;
    if c.remaining() != 0 {
        return Err(bad(format!("{} trailing bytes", c.remaining())));
    }
    let union = BooleanQuery {
        should: subs
            .iter()
            .map(|q| Clause::Boolean(Box::new(q.clone())))
            .collect(),
        ..Default::default()
    };
    let fields: Vec<String> = crate::clauses::clause_field_names(&union)
        .into_iter()
        .map(str::to_string)
        .collect();
    let union_query = JvmQuery::Boolean(union);
    let mut opened = h.reader.open_segments().map_err(|e| {
        set_last_error(format!("opening segment postings: {e}"));
        Status::Decode
    })?;
    if query_uses_points(&union_query) {
        opened.open_points().map_err(|e| {
            set_last_error(format!("opening segment points: {e}"));
            Status::Decode
        })?;
    }
    let owned = h
        .reader
        .field_norms_by_field(&fields)
        .map_err(|error| map_search_error(error.into()))?;
    let norms: Vec<Option<&std::collections::HashMap<String, FieldNorms<'_>>>> =
        owned.iter().map(|m| (!m.is_empty()).then_some(m)).collect();
    let segment_count = h.live_docs.len();
    let slices = if slices.is_empty() {
        vec![(0..segment_count).collect::<Vec<_>>()]
    } else {
        slices
    };
    let mut seen = vec![false; segment_count]; // alloc-ok: segment count of the opened reader
    for &s in slices.iter().flatten() {
        if s >= segment_count || std::mem::replace(&mut seen[s], true) {
            return Err(bad(format!("segment {s} out of range or in two slices")));
        }
    }
    let segments: Vec<OpenSegment<'_>> = opened
        .as_open_segments()
        .into_iter()
        .zip(&h.live_docs)
        .map(|(mut s, live)| {
            s.live_docs = live.as_ref();
            s.similarity = sim.as_ref().map(|p| p as &dyn Similarity);
            s
        })
        .collect();
    // Each sub-query's weight, over the whole shard: its statistics are the
    // reader's whichever slice scores it.
    let prepared = subs
        .iter()
        .map(|q| lucene_search::multi_segment::PreparedBoolean::new(&segments, q))
        .collect::<Result<Vec<_>, _>>()
        .map_err(map_search_error)?;
    let mut out = Vec::new();
    out.extend_from_slice(
        &i32::try_from(slices.len())
            .unwrap_or(i32::MAX)
            .to_le_bytes(),
    );
    for slice in &slices {
        let mut collector = HybridCollector::new(prepared.len(), top_n);
        for &s in slice {
            let seg = &segments[s];
            let scored = prepared
                .iter()
                .map(|p| p.score_all(seg, norms[s]))
                .collect::<Result<Vec<_>, _>>()
                .map_err(map_search_error)?;
            collector.segment(&scored, seg.doc_base);
        }
        out.extend_from_slice(&collector.total.to_le_bytes());
        for queue in collector.queues {
            let hits = queue.into_sorted();
            out.extend_from_slice(&i32::try_from(hits.len()).unwrap_or(i32::MAX).to_le_bytes());
            for hit in hits {
                out.extend_from_slice(&hit.doc.to_le_bytes());
                out.extend_from_slice(&hit.score.to_bits().to_le_bytes());
            }
        }
    }
    similarity_checked(sim.as_ref(), Ok(out))
}

/// `HybridBulkScorer`'s window: `1 << SHIFT` documents.
pub const HYBRID_WINDOW: i32 = 1 << 12;

/// A hit in a [`HitQueue`]: greater is better (higher score, then lower doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridHit {
    pub score: f32,
    pub doc: i32,
}

impl Eq for HybridHit {}

impl PartialOrd for HybridHit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HybridHit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(other.doc.cmp(&self.doc))
    }
}

/// Lucene's `HitQueue(n, false)`: the `n` best hits, the worst on top.
pub struct HitQueue {
    pub heap: std::collections::BinaryHeap<std::cmp::Reverse<HybridHit>>,
    pub size: usize,
}

impl HitQueue {
    pub fn new(size: usize) -> Self {
        Self {
            heap: std::collections::BinaryHeap::with_capacity(size.min(1 << 16)), // alloc-ok: capped at 65536
            size,
        }
    }

    /// `insertWithOverflow`: the hit pushed out (the new one when it does not
    /// beat the worst), or `None` while the queue fills.
    pub fn insert_with_overflow(&mut self, hit: HybridHit) -> Option<HybridHit> {
        if self.heap.len() < self.size {
            self.heap.push(std::cmp::Reverse(hit));
            return None;
        }
        let mut worst = self.heap.peek_mut()?;
        if worst.0 < hit {
            Some(std::mem::replace(&mut worst.0, hit))
        } else {
            Some(hit)
        }
    }

    pub fn into_sorted(self) -> Vec<HybridHit> {
        let mut hits: Vec<HybridHit> = self.heap.into_iter().map(|r| r.0).collect();
        hits.sort_unstable_by(|a, b| b.cmp(a));
        hits
    }
}

/// One slice's `HybridTopScoreDocCollector`, fed as `HybridBulkScorer` feeds
/// it.
pub struct HybridCollector {
    pub queues: Vec<HitQueue>,
    pub total: i64,
    /// One window's scores per sub-query, 0 where it did not compete.
    pub window: Vec<Vec<f32>>,
}

impl HybridCollector {
    pub fn new(subs: usize, top_n: usize) -> Self {
        Self {
            queues: (0..subs).map(|_| HitQueue::new(top_n)).collect(),
            total: 0,
            window: vec![vec![0.0; HYBRID_WINDOW as usize]; subs], // alloc-ok: at most MAX_HYBRID_QUERIES windows of a fixed size
        }
    }

    /// One segment: each sub-query's matches with their scores, ascending by
    /// segment-local doc ID.
    pub fn segment(&mut self, scored: &[Vec<lucene_search::collector::ScoreDoc>], doc_base: i32) {
        // The segment's own HybridSubQueryScorer: thresholds start at 0.
        let mut min_scores = vec![0.0f32; scored.len()]; // alloc-ok: one per sub-query, at most MAX_HYBRID_QUERIES
        let mut next = vec![0usize; scored.len()]; // alloc-ok: one per sub-query, at most MAX_HYBRID_QUERIES
        let mut matching = vec![0u64; HYBRID_WINDOW as usize / 64]; // alloc-ok: fixed window size
        loop {
            let Some(top) = scored
                .iter()
                .zip(&next)
                .filter_map(|(hits, &i)| hits.get(i).map(|h| h.doc_id))
                .min()
            else {
                return;
            };
            let base = top & !(HYBRID_WINDOW - 1);
            let end = base.saturating_add(HYBRID_WINDOW);
            for (q, hits) in scored.iter().enumerate() {
                let min = min_scores[q];
                while let Some(h) = hits.get(next[q]).filter(|h| h.doc_id < end) {
                    // Only a score above the threshold is collected.
                    if h.score > min {
                        let d = (h.doc_id - base) as usize;
                        matching[d / 64] |= 1 << (d % 64);
                        self.window[q][d] = h.score;
                    }
                    next[q] += 1;
                }
            }
            for (word, bits) in matching.iter_mut().enumerate() {
                while *bits != 0 {
                    let d = word * 64 + bits.trailing_zeros() as usize;
                    *bits &= *bits - 1;
                    self.total += 1;
                    let doc = doc_base + base + d as i32;
                    for (q, queue) in self.queues.iter_mut().enumerate() {
                        let score = std::mem::take(&mut self.window[q][d]);
                        if score <= 0.0 {
                            continue;
                        }
                        if let Some(out) = queue.insert_with_overflow(HybridHit { score, doc }) {
                            min_scores[q] = min_scores[q].max(out.score);
                        }
                    }
                }
            }
        }
    }
}

/// [`search`] behind OpenSearch's `min_score` (its `MinimumScoreCollector`
/// around the top-docs collector): only documents scoring at least `min` are
/// kept and counted; `size: 0` counts them.
pub fn search_min_score(
    h: &JvmReaderHandle,
    query: &JvmQuery,
    top_n: usize,
    count_limit: i64,
    min: f32,
    sim: Option<&PerFieldSimilarity>,
) -> Result<(Vec<ScoreDoc>, i64, bool), Status> {
    let mut opened = h.reader.open_segments().map_err(|e| {
        set_last_error(format!("opening segment postings: {e}"));
        Status::Decode
    })?;
    if query_uses_points(query) {
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
            s.similarity = sim.map(|p| p as &dyn Similarity);
            s
        })
        .collect();
    let q = boolean_of(query);
    let fields: Vec<String> = crate::clauses::clause_field_names(&q)
        .into_iter()
        .map(str::to_string)
        .collect();
    let owned = h
        .reader
        .field_norms_by_field(&fields)
        .map_err(|error| map_search_error(error.into()))?;
    let norms: Vec<Option<&std::collections::HashMap<String, FieldNorms<'_>>>> =
        owned.iter().map(|m| (!m.is_empty()).then_some(m)).collect();
    let threshold = u64::try_from(count_limit).unwrap_or(0);
    if top_n == 0 {
        if count_limit <= 0 {
            return Ok((Vec::new(), -1, false));
        }
        let (count, lower) = lucene_search::multi_segment::count_boolean_query_min_score(
            &segments, &q, &norms, min, threshold,
        )
        .map_err(map_search_error)?;
        return Ok((Vec::new(), i64::try_from(count).unwrap_or(i64::MAX), lower));
    }
    let (hits, total) = lucene_search::multi_segment::search_boolean_query_multi_segment_min_score(
        &segments, &q, &norms, top_n, threshold, min,
    )
    .map_err(map_search_error)?;
    if count_limit <= 0 {
        return Ok((hits, -1, false));
    }
    Ok((
        hits,
        i64::try_from(total.value).unwrap_or(i64::MAX),
        total.relation == TotalHitsRelation::GreaterThanOrEqualTo,
    ))
}

/// The total-hits count behind a `size: 0` `ffi_jvm_reader_search` -- one
/// with no top hits, and so no collector to count them -- as `(count,
/// is_lower_bound)`.
///
/// Lucene stops counting at `totalHitsThreshold` and reports "at least"; so
/// does this, in two steps, cheapest first:
///
/// 1. **A lower bound from the term dictionary alone** ([`lower_bound`]): a
///    term matches at least `docFreq - deletedDocs` live documents of a
///    segment, and a pure disjunction at least as many as its best clause.
///    When that already exceeds `count_limit`, no postings are read at all.
/// 2. **Counting segment by segment**, stopping at the first segment
///    boundary where the running total exceeds `count_limit`.
///
/// Either way the answer is exact whenever it is at most `count_limit`,
/// which is all Lucene promises.
pub fn total_hits(
    h: &JvmReaderHandle,
    segments: &[OpenSegment<'_>],
    query: &JvmQuery,
    count_limit: i64,
) -> Result<(i64, bool), Status> {
    total_hits_with(h, segments, query, count_limit, &[])
}

/// `min_score`'s `size: 0` count ([`search_min_score`]) from each segment's
/// passing documents: summed in segment order, stopping at the first segment
/// boundary past `limit`. `None` when a segment on the way was not visited
/// (its aggregations answered from points), which then counts it itself.
pub fn min_score_total(seen: &[Option<u64>], limit: i64) -> Option<(i64, bool)> {
    let limit = u64::try_from(limit).unwrap_or(0);
    let mut count = 0u64;
    for n in seen {
        count = count.saturating_add((*n)?);
        if count > limit {
            return Some((i64::try_from(count).unwrap_or(i64::MAX), true));
        }
    }
    Some((i64::try_from(count).unwrap_or(i64::MAX), false))
}

/// [`total_hits`], with `known[i]` segment `i`'s live match count where it is
/// already known: those segments are not counted again.
pub fn total_hits_with(
    h: &JvmReaderHandle,
    segments: &[OpenSegment<'_>],
    query: &JvmQuery,
    count_limit: i64,
    known: &[Option<u64>],
) -> Result<(i64, bool), Status> {
    let mut bound = 0i64;
    for (i, seg) in segments.iter().enumerate() {
        let deleted = h.deleted.get(i).copied().unwrap_or(0);
        let lb = match query {
            JvmQuery::Term(q) => term_lower_bound(seg, deleted, q)?,
            JvmQuery::Boolean(q) => lower_bound_boolean(seg, deleted, q)?,
        };
        bound = bound.saturating_add(lb);
    }
    if bound > count_limit {
        return Ok((bound, true));
    }
    // A function query's reader-wide preparation, once for every segment.
    let counted = (0..segments.len()).all(|i| known.get(i).copied().flatten().is_some());
    let global = match query {
        JvmQuery::Boolean(q) if !counted => {
            lucene_search::count_preparation(segments, q).map_err(map_search_error)?
        }
        _ => None,
    };
    let mut total = 0i64;
    for (i, seg) in segments.iter().enumerate() {
        let n = match known.get(i).copied().flatten() {
            Some(n) => i64::try_from(n).unwrap_or(i64::MAX),
            None => count_segment(seg, query, global.as_ref())?,
        };
        total = total.saturating_add(n);
        if total > count_limit {
            return Ok((total, true));
        }
    }
    Ok((total, false))
}

/// Live documents of `seg` certainly matching `q`: its `docFreq` less every
/// deleted document of the segment (each could have been one of them).
pub fn term_lower_bound(seg: &OpenSegment<'_>, deleted: i64, q: &TermQuery) -> Result<i64, Status> {
    let Some(field) = seg.fields.field(&q.field) else {
        return Ok(0);
    };
    let df = field
        .try_seek_exact(&q.term)
        .map_err(|e| map_search_error(e.into()))?
        .map_or(0, |s| i64::from(s.doc_freq));
    Ok(df.saturating_sub(deleted).max(0))
}

/// [`lower_bound`] for a boolean: a pure disjunction (only `SHOULD` clauses,
/// `minimum_should_match <= 1`) as its best clause; exactly one
/// `MUST`/`FILTER` clause and nothing else as that clause; anything else --
/// a conjunction, anything with a `MUST_NOT` -- `0`.
pub fn lower_bound_boolean(
    seg: &OpenSegment<'_>,
    deleted: i64,
    b: &BooleanQuery,
) -> Result<i64, Status> {
    if !b.must_not.is_empty() {
        return Ok(0);
    }
    if b.minimum_should_match > b.should.len() {
        // Lucene matches nothing when fewer SHOULD clauses exist than must match.
        return Ok(0);
    }
    let required = b.must.len() + b.filter.len();
    if required == 0 && b.minimum_should_match <= 1 {
        let mut best = 0;
        for c in &b.should {
            best = best.max(lower_bound(seg, deleted, c)?);
        }
        Ok(best)
    } else if required == 1 && b.should.is_empty() {
        let only = b
            .must
            .first()
            .or(b.filter.first())
            .expect("one required clause");
        lower_bound(seg, deleted, only)
    } else {
        Ok(0)
    }
}

/// A cheap lower bound on the live documents of `seg` matching `clause`, from
/// the term dictionary only. Deliberately narrow: a term (see
/// [`term_lower_bound`]); a wrapper, as its inner clause; a boolean, see
/// [`lower_bound_boolean`]. Anything else is `0`, which is always a valid
/// lower bound. Recursion depth is the clause tree's, which the decoder caps
/// at `MAX_CLAUSE_DEPTH`.
pub fn lower_bound(seg: &OpenSegment<'_>, deleted: i64, clause: &Clause) -> Result<i64, Status> {
    match clause {
        Clause::Term(t) => term_lower_bound(seg, deleted, t),
        Clause::ConstantScore(c) => lower_bound(seg, deleted, &c.inner),
        Clause::Boost(b) => lower_bound(seg, deleted, &b.inner),
        Clause::Boolean(b) => lower_bound_boolean(seg, deleted, b),
        _ => Ok(0),
    }
}

/// `IndexSearcher.count` for one segment.
pub fn count_segment(
    seg: &OpenSegment<'_>,
    query: &JvmQuery,
    global: Option<&lucene_search::GlobalStats>,
) -> Result<i64, Status> {
    Ok({
        match query {
            JvmQuery::Term(q) => count_term_query(seg.fields, seg.doc_in, seg.live_docs, q),
            // The scored search's scorer tree, in `COMPLETE_NO_SCORES`.
            JvmQuery::Boolean(q) => count_boolean_query_segment_with(seg, q, global)
                .map(|n| i64::try_from(n).unwrap_or(i64::MAX)),
        }
        .map_err(map_search_error)?
    })
}

/// Hooks the tests of the reader entry points use to steer the engine.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    thread_local! {
        /// Whether a terms aggregation counts buffered documents' keyword
        /// runs from their encoded doc values instead of in place.
        pub static ENCODED_RUNS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
}
