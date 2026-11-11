//! The top `n` by one numeric key without visiting every match: "newest N"
//! by a timestamp, the shape log and event searches run all day.
//!
//! Lucene's `NumericComparator` prunes with the field's points or skip index
//! only once the queue is full, and over segments written in time order a
//! descending sort fills it from the oldest documents first: every later
//! document is competitive, nothing is skipped, and the search visits every
//! match. This path finds a bound `t` first, from the index structures alone,
//! such that at least `k` values of the field are at least as good as `t`:
//!
//! * a best-first walk over every segment's BKD cells (`PointTree`), or its
//!   doc-values skip index blocks, popping the cell with the best bound and
//!   taking whole leaves until `k` values are taken; `t` is the worst bound
//!   among the taken leaves;
//!
//! then runs the ordinary sorted search over `query AND key within t`. Every
//! document outside that range sorts strictly behind `t` (it has no value at
//! least as good as `t`, or none at all, and the missing value is required to
//! sort behind `t`), so when the restricted search fills `n` hits whose last
//! is at least as good as `t`, they are exactly the hits the whole search
//! keeps: the same documents in the same order, ties broken by document as
//! everywhere else. Otherwise (a selective query, deletions, several values a
//! document) `k` grows and the search runs again, until the range would cover
//! most of the index, where the ordinary search runs instead.
//!
//! The total is counted on its own, up to the search's threshold: exact below
//! it, a lower bound (`GreaterThanOrEqualTo`) once it is reached -- what
//! `TopFieldCollector` promises, and what OpenSearch reports (it caps the
//! total at `track_total_hits` either way).

use lucene_codecs::field_infos::DocValuesType;
use lucene_codecs::points::{PointTreeNode, PointsReader};

use super::{search_segments, FieldDoc, Run, Selector, SortField, SortType, TopFieldDocs};
use crate::collector::{TotalHits, TotalHitsRelation};
use crate::exec::{self, Mode, NO_MORE_DOCS};
use crate::extended_query::{ExtendedQuery, NumericDocValuesRangeQuery};
use crate::multi_segment::OpenSegment;
use crate::query::{BooleanQuery, Clause, PointsRangeQuery};
use crate::Result;

#[cfg(test)]
pub(super) mod tests;

/// The restricted search is not tried when the first bound would already
/// take more than this fraction of the field's values (`1 / 8`): the
/// ordinary search is as cheap there.
const MAX_SHARE_SHIFT: u32 = 3;

/// See `limit` in [`search`]: the share of the query's estimated matches
/// the slices may cover.
const MATCH_FACTOR: u64 = 1;

/// How much the count grows after a slice without a hit.
const GROWTH_EMPTY: u64 = 8;

/// Whether `run` is a search this path answers: one numeric key (not nested,
/// not custom), no score wanted anywhere, every segment searched whole, and
/// no segment already sorted by the key (whose search stops early anyway).
fn eligible(run: &Run<'_, '_>) -> bool {
    let [key] = run.sort else {
        return false;
    };
    // A `DOUBLE`/`FLOAT` column stores raw bits (`FloatDocValuesField`),
    // which a doc-values range or skip index does not order as the key does.
    matches!(key.ty, SortType::Long | SortType::Int)
        // A sum or an average is no value of the document's own, which the
        // bounds and the range are over.
        && matches!(
            key.selector,
            Selector::Min | Selector::Max | Selector::MiddleMin | Selector::MiddleMax
        )
        && key.nested.is_none()
        && key.rewritten.is_none()
        && !run.track
        && run.min_score.is_none()
        && run.until.is_none()
        && !run.scorer_only
        && run.global.is_none()
        && run.top_n > 0
        && !run.segments.iter().any(|s| s.index_sort_prefix)
}

/// The key's "goodness": larger is better, so a descending and an ascending
/// sort walk the same way. `i128` keeps `-i64::MIN` representable.
fn good(key: &SortField, v: i64) -> i128 {
    if key.reverse {
        i128::from(v)
    } else {
        -i128::from(v)
    }
}

/// One candidate cell of the best-first walk: its best and worst bound (as
/// [`good`] values) and where it comes from.
struct Cell {
    best: i128,
    worst: i128,
    kind: CellKind,
}

enum CellKind {
    /// A BKD node of segment `seg`; its size is read when it is taken.
    Node { seg: usize, node: PointTreeNode },
    /// A segment's whole skip index, expanded into its blocks when popped.
    Skipper { seg: usize },
    /// One skip-index block of `count` documents.
    Block { count: u64 },
}

/// Where each segment's bounds for the key come from.
enum Source<'a> {
    None,
    Points {
        reader: &'a PointsReader<'a>,
        field: i32,
        bytes: usize,
    },
    Skipper(&'a lucene_codecs::doc_values::DocValuesSkipIndex),
}

/// A packed point value as the comparable long the key compares: the
/// sortable bytes decoded, sign bit flipped back (`NumericUtils`).
fn unpack(bytes: &[u8]) -> Option<i64> {
    match bytes.len() {
        8 => Some(i64::from_be_bytes(bytes.try_into().ok()?) ^ i64::MIN),
        4 => Some(i64::from(
            i32::from_be_bytes(bytes.try_into().ok()?) ^ i32::MIN,
        )),
        _ => None,
    }
}

/// Every segment's bound source, and whether the points cover every
/// segment holding the field's doc values (so a point range restricts
/// correctly); `None` when the key's points have a shape the key cannot
/// read.
fn sources<'r>(run: &Run<'r, '_>, key: &SortField) -> Result<(Vec<Source<'r>>, bool)> {
    let want = if key.ty == SortType::Long { 8 } else { 4 };
    let mut points_cover = true;
    let ranges = required_ranges(run.query);
    let mut out = Vec::with_capacity(run.segments.len());
    for (seg, reader) in run.segments.iter().zip(run.readers) {
        // A segment the query matches nothing in holds no hit: its values do
        // not count toward a bound (the range still covers it).
        if crate::slices::estimated_matches(std::slice::from_ref(seg), run.query) == 0
            || ranges.iter().any(|r| misses(seg, r))
        {
            out.push(Source::None);
            continue;
        }
        let Some(info) = reader.field_infos().field_by_name(&key.field) else {
            out.push(Source::None);
            continue;
        };
        let points = seg.points.and_then(|p| {
            let num = p.field_number(&key.field)?;
            let f = p.reader.field(num)?;
            (f.num_dims == 1 && f.bytes_per_dim as usize == want).then_some((p, num))
        });
        if info.doc_values_type != DocValuesType::None && points.is_none() {
            points_cover = false;
        }
        if let Some((p, num)) = points {
            out.push(Source::Points {
                reader: &p.reader,
                field: num,
                bytes: want,
            });
            continue;
        }
        let skip = if info.doc_values_type == DocValuesType::None {
            None
        } else {
            reader.doc_values_skip_index(info.number)?
        };
        out.push(match skip {
            Some(index) => Source::Skipper(index),
            None => Source::None,
        });
    }
    Ok((out, points_cover))
}

/// The cells to start the walk from: each segment's root.
fn roots(key: &SortField, sources: &[Source<'_>]) -> Result<(Vec<Cell>, u64)> {
    let mut cells = Vec::new();
    let mut values = 0u64;
    for (seg, source) in sources.iter().enumerate() {
        match source {
            Source::None => {}
            Source::Points { reader, field, .. } => {
                let node = reader.point_tree(*field)?;
                let (Some(lo), Some(hi)) = (unpack(node.min_packed()), unpack(node.max_packed()))
                else {
                    continue;
                };
                values = values
                    .saturating_add(u64::try_from(reader.point_tree_size(&node)?).unwrap_or(0));
                cells.push(Cell {
                    best: good(key, hi).max(good(key, lo)),
                    worst: good(key, hi).min(good(key, lo)),
                    kind: CellKind::Node { seg, node },
                });
            }
            Source::Skipper(index) => {
                if index.doc_count <= 0 {
                    continue;
                }
                values = values.saturating_add(u64::try_from(index.doc_count).unwrap_or(0));
                let (a, b) = (good(key, index.min_value), good(key, index.max_value));
                cells.push(Cell {
                    best: a.max(b),
                    worst: a.min(b),
                    kind: CellKind::Skipper { seg },
                });
            }
        }
    }
    Ok((cells, values))
}

/// The best-first walk over every segment's cells, resumable: each
/// [`Walk::until`] takes cells until `k` values are taken, from where the
/// last stopped.
struct Walk<'s, 'a> {
    key: &'s SortField,
    sources: &'s [Source<'a>],
    /// The [`good`] values a hit can have (`searchAfter`'s value at best,
    /// and the query's own required ranges on the key): a cell wholly
    /// outside holds no hit, and is dropped.
    window: (i128, i128),
    /// A max-heap of `(best bound, index into cells)`.
    heap: std::collections::BinaryHeap<(i128, usize)>,
    cells: Vec<Option<Cell>>,
    taken: u64,
    worst: i128,
}

impl<'s, 'a> Walk<'s, 'a> {
    fn new(
        key: &'s SortField,
        sources: &'s [Source<'a>],
        start: Vec<Cell>,
        window: (i128, i128),
    ) -> Self {
        let mut w = Walk {
            key,
            sources,
            window,
            heap: std::collections::BinaryHeap::new(),
            cells: Vec::new(),
            taken: 0,
            worst: i128::MAX,
        };
        for c in start {
            w.push(c);
        }
        w
    }

    fn push(&mut self, cell: Cell) {
        if cell.worst > self.window.1 || cell.best < self.window.0 {
            return;
        }
        self.heap.push((cell.best, self.cells.len()));
        self.cells.push(Some(cell));
    }

    /// A cell of `bytes`-wide packed bounds, `None` for one it cannot read.
    fn node_cell(&self, seg: usize, node: PointTreeNode, bytes: usize) -> Option<Cell> {
        let lo = node.min_packed().get(..bytes).and_then(unpack)?;
        let hi = node.max_packed().get(..bytes).and_then(unpack)?;
        let (a, b) = (good(self.key, lo), good(self.key, hi));
        Some(Cell {
            best: a.max(b),
            worst: a.min(b),
            kind: CellKind::Node { seg, node },
        })
    }

    /// The bound `t` (as a [`good`] value) with at least `k` of the field's
    /// values at least as good, over every walked segment; `None` when they
    /// hold fewer than `k` values.
    fn until(&mut self, k: u64) -> Result<Option<i128>> {
        while self.taken < k {
            let Some((_, i)) = self.heap.pop() else {
                return Ok(None);
            };
            let Some(cell) = self.cells.get_mut(i).and_then(Option::take) else {
                continue;
            };
            match cell.kind {
                CellKind::Node { seg, node } => {
                    let Some(Source::Points { reader, bytes, .. }) = self.sources.get(seg) else {
                        continue;
                    };
                    let (reader, bytes) = (*reader, *bytes);
                    match reader.point_tree_children(&node)? {
                        Some((l, r)) => {
                            for child in [l, r] {
                                if let Some(c) = self.node_cell(seg, child, bytes) {
                                    self.push(c);
                                }
                            }
                        }
                        None => {
                            let size = u64::try_from(reader.point_tree_size(&node)?).unwrap_or(0);
                            self.taken = self.taken.saturating_add(size);
                            self.worst = self.worst.min(cell.worst);
                        }
                    }
                }
                CellKind::Skipper { seg } => {
                    let Some(Source::Skipper(index)) = self.sources.get(seg) else {
                        continue;
                    };
                    for interval in &index.intervals {
                        let Some(l0) = interval.levels.first() else {
                            continue;
                        };
                        if l0.doc_count <= 0 {
                            continue;
                        }
                        let (a, b) = (good(self.key, l0.min_value), good(self.key, l0.max_value));
                        self.push(Cell {
                            best: a.max(b),
                            worst: a.min(b),
                            kind: CellKind::Block {
                                count: u64::try_from(l0.doc_count).unwrap_or(0),
                            },
                        });
                    }
                }
                CellKind::Block { count } => {
                    self.taken = self.taken.saturating_add(count);
                    self.worst = self.worst.min(cell.worst);
                }
            }
        }
        Ok(Some(self.worst))
    }
}

/// `query AND key within [t, prev)` (`prev` open when `None`): the key's
/// point range where points cover every segment with its doc values (a
/// `LONG` key, whose points are the 8-byte sortable longs a point range
/// packs), its doc-values range (read through the skip index where there is
/// one) otherwise.
fn restricted(
    query: &BooleanQuery,
    key: &SortField,
    t: i64,
    prev: Option<i64>,
    points: bool,
) -> BooleanQuery {
    let (lo, hi) = if key.reverse {
        (t, prev.map_or(i64::MAX, |p| p.saturating_sub(1)))
    } else {
        (prev.map_or(i64::MIN, |p| p.saturating_add(1)), t)
    };
    let range = if points && key.ty == SortType::Long {
        Clause::PointsRange(PointsRangeQuery::new(key.field.clone(), lo, hi))
    } else {
        Clause::Extended(Box::new(ExtendedQuery::NumericDocValuesRange(
            NumericDocValuesRangeQuery::new(key.field.clone(), lo, hi),
        )))
    };
    BooleanQuery {
        filter: vec![Clause::Boolean(Box::new(query.clone())), range],
        ..Default::default()
    }
}

/// The search `run` describes, through bounds, or `None` when this path
/// does not apply (the caller runs the ordinary search).
///
/// The search runs in slices of the key's range, best first: slice `i`
/// holds the values in `[t_i, t_{i-1})` (`t_0` open), each `t_i` the bound
/// of a growing count `k_i`. A slice's hits are the best among the
/// documents whose value lies in it, so the slices' hits in order are the
/// whole search's; a document whose sorted value lies outside the slice it
/// matched (several values) is left to the slice that holds it. Past
/// [`MAX_SHARE_SHIFT`] the last slice is the rest of the range, searched
/// unrestricted after the last bound.
pub(super) fn search(run: &Run<'_, '_>) -> Result<Option<TopFieldDocs>> {
    #[cfg(test)]
    if tests::DISABLED.with(std::cell::Cell::get) {
        return Ok(None);
    }
    if !eligible(run) {
        return Ok(None);
    }
    let key = &run.sort[0];
    let (sources, points_cover) = sources(run, key)?;
    let (start, values) = roots(key, &sources)?;
    // Every bound is at most the best root's: a missing value at least as
    // good never sorts behind one.
    let best = start.iter().map(|c| c.best).max();
    if best.is_none_or(|b| good(key, key.missing) >= b) {
        return Ok(None);
    }
    let all: Vec<usize> = (0..run.segments.len().min(run.readers.len())).collect();
    // Slices pay until they would cover the larger of an eighth of the
    // values and the query's matches (measured: past that, a filter whose
    // newest match lies far back costs the ordinary search's time plus the
    // slices').
    let matches = crate::slices::estimated_matches(run.segments, run.query);
    let limit = (values >> MAX_SHARE_SHIFT).max(matches.saturating_mul(MATCH_FACTOR));
    let n = u64::try_from(run.top_n).unwrap_or(u64::MAX);
    // `searchAfter`: no value better than its own is a hit, so cells wholly
    // better than it hold none.
    let mut window = (i128::MIN, i128::MAX);
    if let Some(&v) = run.after.and_then(|a| a.values.first()) {
        window.1 = good(key, v);
    }
    for r in required_ranges(run.query) {
        if r.field == key.field && key.ty == SortType::Long {
            let (a, b) = (good(key, r.min), good(key, r.max));
            window = (window.0.max(a.min(b)), window.1.min(a.max(b)));
        }
    }
    let missing = good(key, key.missing);
    let mut walk = Walk::new(key, &sources, start, window);
    let mut hits: Vec<FieldDoc> = Vec::new();
    // The previous slice's bound, as the key's value.
    let mut prev: Option<i64> = None;
    let mut k = n;
    let mut k_prev = 0u64;
    loop {
        let t = if k > limit { None } else { walk.until(k)? };
        // A document without a value sorts as the missing value: it must
        // sort behind `t`, since the range leaves it out.
        let t = t.filter(|&t| missing < t);
        let tv = t.and_then(|t| i64::try_from(if key.reverse { t } else { -t }).ok());
        let after = slice_after(key, run.after, prev);
        let remaining = run.top_n.saturating_sub(hits.len());
        let (Some(t), Some(tv)) = (t, tv) else {
            if prev.is_none() {
                return Ok(None);
            }
            // The rest of the range: the ordinary search after the last bound.
            let r = Run {
                top_n: remaining,
                total_hits_threshold: 0,
                after: after.as_ref(),
                ..*run
            };
            hits.extend(search_segments(&r, &all)?.hits);
            return Ok(Some(finish(run, hits, n)?));
        };
        let q = restricted(run.query, key, tv, prev, points_cover);
        let rewritten = crate::multi_segment::rewrite_points_ranges(&q, run.segments);
        let q = rewritten.unwrap_or(q);
        let r = Run {
            query: &q,
            top_n: remaining,
            total_hits_threshold: 0,
            after: after.as_ref(),
            ..*run
        };
        let slice = search_segments(&r, &all)?.hits;
        #[cfg(test)]
        tests::SLICES.with(|c| c.set(c.get().saturating_add(1)));
        let before = hits.len();
        hits.extend(
            slice
                .into_iter()
                .take_while(|h| h.values.first().is_some_and(|&v| good(key, v) >= t)),
        );
        if hits.len() == run.top_n {
            return Ok(Some(finish(run, hits, n)?));
        }
        let got = u64::try_from(hits.len().saturating_sub(before)).unwrap_or(0);
        let need = u64::try_from(run.top_n.saturating_sub(hits.len())).unwrap_or(u64::MAX);
        let covered = k.saturating_sub(k_prev);
        // Enough more values for the hits still wanted at the rate this
        // slice found them (twice over), at least doubling.
        let next = match covered.saturating_mul(need).checked_div(got) {
            Some(per) => k.saturating_add(per.saturating_mul(2)),
            None => k.saturating_mul(GROWTH_EMPTY),
        };
        k_prev = k;
        k = next.max(k.saturating_mul(2));
        prev = Some(tv);
    }
}

/// The hits, with the total counted for the original query.
fn finish(run: &Run<'_, '_>, hits: Vec<FieldDoc>, n: u64) -> Result<TopFieldDocs> {
    Ok(TopFieldDocs {
        hits,
        total: count(run.segments, run.query, run.total_hits_threshold, n)?,
        max_score: f32::NAN,
    })
}

/// The `searchAfter` of a slice: the request's own for the first, else the
/// last bound with no document after it at that value (`(prev,
/// Integer.MAX_VALUE)`: only worse values pass), whichever comes later.
fn slice_after(key: &SortField, after: Option<&FieldDoc>, prev: Option<i64>) -> Option<FieldDoc> {
    let bound = prev.map(|p| FieldDoc {
        doc: i32::MAX,
        values: vec![p],
        terms: Vec::new(),
    });
    match (after, bound) {
        (None, b) => b,
        (Some(a), None) => Some(a.clone()),
        (Some(a), Some(b)) => {
            let av = a.values.first().map_or(i128::MAX, |&v| good(key, v));
            let bv = b.values.first().map_or(i128::MAX, |&v| good(key, v));
            // The later of the two in the sort's order.
            Some(
                if (av, std::cmp::Reverse(a.doc)) < (bv, std::cmp::Reverse(b.doc)) {
                    a.clone()
                } else {
                    b
                },
            )
        }
    }
}

/// The total `TopFieldCollector` reports for `query`: exact while below
/// `threshold` (raised to the queue size, as `TopFieldCollectorManager`
/// does), a lower bound once counting stops there.
fn count(
    segments: &[OpenSegment<'_>],
    query: &BooleanQuery,
    threshold: u64,
    n: u64,
) -> Result<TotalHits> {
    let limit = threshold.max(n).saturating_add(1);
    let mut total = 0u64;
    for seg in segments {
        if let crate::terminate::LeafCount::Whole(c) = crate::terminate::leaf_count(seg, query)? {
            total = total.saturating_add(c);
            continue;
        }
        let ctx = exec::LeafContext {
            fields: seg.fields,
            doc_in: seg.doc_in,
            pos_in: seg.pos_in,
            pay_in: seg.pay_in,
            live_docs: seg.live_docs,
            points: seg.points,
            norms: None,
            global: None,
            max_doc: seg.max_doc,
            cache: seg.cache,
            reader: seg.reader,
            similarity: seg.similarity,
        };
        let Some(mut scorer) =
            exec::build::build_boolean_root(&ctx, query, 1.0, Mode::NoScores, true)?
        else {
            continue;
        };
        let mut doc = exec::exact_next(scorer.as_mut())?;
        while doc != NO_MORE_DOCS {
            if seg.live_docs.is_none_or(|l| l.get_doc(doc)) {
                total = total.saturating_add(1);
                if total >= limit {
                    return Ok(TotalHits {
                        value: total,
                        relation: TotalHitsRelation::GreaterThanOrEqualTo,
                    });
                }
            }
            doc = exec::exact_next(scorer.as_mut())?;
        }
    }
    Ok(TotalHits {
        value: total,
        relation: TotalHitsRelation::EqualTo,
    })
}

/// The point ranges every match of `query` must fall in: its required
/// (`must`/`filter`) point ranges, through required nested booleans and
/// constant-score and boost wrappers.
fn required_ranges(query: &BooleanQuery) -> Vec<&PointsRangeQuery> {
    fn clause<'q>(c: &'q Clause, out: &mut Vec<&'q PointsRangeQuery>) {
        match c {
            Clause::PointsRange(r) => out.push(r),
            Clause::Boolean(b) => boolean(b, out),
            Clause::ConstantScore(c) => clause(&c.inner, out),
            Clause::Boost(b) => clause(&b.inner, out),
            _ => {}
        }
    }
    fn boolean<'q>(b: &'q BooleanQuery, out: &mut Vec<&'q PointsRangeQuery>) {
        for c in b.must.iter().chain(&b.filter) {
            clause(c, out);
        }
        // A lone `should` is required too.
        if let ([], [], [only]) = (&b.must[..], &b.filter[..], &b.should[..]) {
            if b.minimum_should_match <= 1 {
                clause(only, out);
            }
        }
    }
    let mut out = Vec::new();
    boolean(query, &mut out);
    out
}

/// Whether `range` (8-byte, as [`PointsRangeQuery`] packs it) matches
/// nothing in `seg`: the segment's points for the field lie wholly outside,
/// or it has none.
fn misses(seg: &OpenSegment<'_>, range: &PointsRangeQuery) -> bool {
    let Some(points) = seg.points else {
        return false;
    };
    let Some(field) = points
        .field_number(&range.field)
        .and_then(|n| points.reader.field(n))
    else {
        return true;
    };
    if field.num_dims != 1 || field.bytes_per_dim != 8 {
        return false;
    }
    match (
        unpack(&field.min_packed_value),
        unpack(&field.max_packed_value),
    ) {
        (Some(lo), Some(hi)) => range.max < lo || range.min > hi,
        _ => false,
    }
}
