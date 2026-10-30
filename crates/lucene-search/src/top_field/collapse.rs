//! Field collapsing: OpenSearch's `CollapsingTopDocsCollector` (shipped in
//! Lucene's `org.apache.lucene.search.grouping` package), the
//! `FirstPassGroupingCollector` it extends and its two group selectors,
//! `CollapsingDocValuesSource.Numeric` and `.Keyword`.
//!
//! The top `topN` groups by the sort, each group being the documents that
//! share the collapse field's value (those without one form one group of
//! their own) and sorting as its best document does; each group is reported
//! by that document, with its sort values and the group's value. Every
//! match is counted, and the collector sees every match (`COMPLETE` or
//! `COMPLETE_NO_SCORES`): no comparator skips.
//!
//! The comparators are [`TopField`]'s own, every one at `Pruning.NONE`
//! (`sortField.getComparator(topNGroups + 1, Pruning.NONE)`), driven slot by
//! slot as `FirstPassGroupingCollector` drives them: `copy`, `compare`,
//! `setBottom`, `compareBottom`, and `compareTop` for `search_after` (which
//! OpenSearch allows only on a sort by the collapse field alone).

use std::collections::HashMap;

use lucene_codecs::field_infos::DocValuesType;

use super::{
    global_stats, key_needs_scores, leaf_bulk, open_leaf, passes, FieldDoc, Leaf, Pruning,
    SortError, SortField, TopField,
};
use crate::collector::{ScoreMode, ScoringCollector};
use crate::directory_reader::SegmentReader;
use crate::exec::{self, Mode, NO_MORE_DOCS};
use crate::field_norms::FieldNorms;
use crate::grouping::TreeSet;
use crate::multi_segment::OpenSegment;
use crate::query::BooleanQuery;
use crate::reader::doc_values as dv;
use crate::reader::{
    NumericDocValues, SortedDocValues, SortedNumericDocValues, SortedSetDocValues,
};
use crate::{Error, Result};

/// Which `CollapsingDocValuesSource` reads the collapse field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollapseKind {
    /// `Numeric`: a `NUMERIC` column, or a single-valued `SORTED_NUMERIC`
    /// one; the group value is the stored long.
    Numeric,
    /// `Keyword`: a `SORTED` column, or a single-valued `SORTED_SET` one;
    /// the group value is the term.
    Keyword,
}

/// A group's value (`CollapseTopFieldDocs.collapseValues`): `Missing` is the
/// group of the documents without one (Java's `null`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CollapseValue {
    Missing,
    Long(i64),
    Bytes(Vec<u8>),
}

/// One collector's answer (`CollapsingTopDocsCollector.getTopDocs()`): the
/// groups best first, each its top document with its sort values, beside
/// the group values; every match counted; and the max score over every
/// match when it was tracked (`MaxScoreCollector`), else `NaN`.
#[derive(Debug, Clone)]
pub struct CollapseTopDocs {
    pub hits: Vec<FieldDoc>,
    pub values: Vec<CollapseValue>,
    pub total: u64,
    pub max_score: f32,
}

/// The message that starts every refusal of a multi-valued collapse field,
/// as `CollapsingDocValuesSource` words it.
pub const MULTI_VALUED: &str = "failed to collapse";

/// `IndexSearcher.search(query, CollapsingTopDocsCollector.create*(field,
/// sort, topN, after))` -- beside a `MaxScoreCollector` when
/// `track_max_score`, behind OpenSearch's `MinimumScoreCollector` when
/// `min_score` -- one collector per slice of a concurrent search (each
/// slice's segments in doc-base order; one collector over every segment
/// without slices), as `CollapsingTopDocsCollectorContext` runs them. The
/// caller merges the slices' answers (`CollapseTopFieldDocs.merge`).
///
/// `readers`, `norms` and the opened `segments` as for
/// [`super::search_sorted`]; `after` is the previous page's last hit, whose
/// one value is the collapse field's.
///
/// # Errors
/// A sort without keys or a `top_n` of 0; `search_after` on a sort that is
/// not one key ([`Error::IllegalArgument`], OpenSearch's own check); a
/// collapse field with more than one value in a document the collector
/// reaches, or doc values of another type ([`Error::IllegalState`], the
/// message starting [`MULTI_VALUED`] for the former); what reading the
/// columns or scoring reports.
#[allow(clippy::too_many_arguments)]
pub fn search_collapsed(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    query: &BooleanQuery,
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    sort: &[SortField],
    field: &str,
    kind: CollapseKind,
    top_n: usize,
    after: Option<&FieldDoc>,
    track_max_score: bool,
    slices: &[Vec<usize>],
    min_score: Option<f32>,
) -> Result<Vec<CollapseTopDocs>> {
    search_collapsed_in(
        segments,
        readers,
        query,
        norms,
        sort,
        field,
        kind,
        top_n,
        after,
        track_max_score,
        slices,
        min_score,
        None,
    )
}

/// [`search_collapsed`], the segment `until.0` searched below doc `until.1`.
#[allow(clippy::too_many_arguments)]
fn search_collapsed_in(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    query: &BooleanQuery,
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    sort: &[SortField],
    field: &str,
    kind: CollapseKind,
    top_n: usize,
    after: Option<&FieldDoc>,
    track_max_score: bool,
    slices: &[Vec<usize>],
    min_score: Option<f32>,
    until: Option<(usize, i32)>,
) -> Result<Vec<CollapseTopDocs>> {
    if sort.is_empty() {
        return Err(SortError::NoKeys.into());
    }
    if top_n == 0 {
        return Err(Error::IllegalArgument(
            "topNGroups must be >= 1 (got 0)".into(),
        ));
    }
    if after.is_some() && sort.len() != 1 {
        return Err(Error::IllegalArgument(
            "The after parameter can only be used when the sort is based on the collapse field"
                .into(),
        ));
    }
    let rewritten = crate::multi_segment::rewrite_points_ranges(query, segments);
    let query = rewritten.as_ref().unwrap_or(query);
    // The score leading the sort gives the max score itself (the first hit).
    let by_score = sort[0] == SortField::score();
    let track = track_max_score && !by_score;
    let global = global_stats(segments, query, sort, track || min_score.is_some(), None)?;
    let run = Run {
        segments,
        readers,
        query,
        norms,
        sort,
        field,
        kind,
        top_n,
        after,
        track,
        global: global.as_ref(),
        min_score,
    };
    if slices.len() <= 1 {
        let all: Vec<usize> = match slices.first() {
            Some(s) => {
                let mut s = s.clone();
                s.sort_unstable();
                s
            }
            None => (0..segments.len().min(readers.len())).collect(),
        };
        return Ok(vec![collapse_segments(&run, &all, until)?]);
    }
    let parallel =
        crate::slices::estimated_matches(segments, query) >= crate::slices::SEQUENTIAL_BELOW;
    crate::slices::run_slices_if(parallel, slices, |slice| {
        let mut order = slice.to_vec();
        order.sort_unstable();
        collapse_segments(&run, &order, None)
    })
    .into_iter()
    .collect()
}

/// [`search_collapsed`] behind `terminate_after(n)`, which OpenSearch never
/// runs concurrently: one collector over the first `n` matches in index order
/// ([`crate::terminate::terminate_after`]'s prefix of the index), and the
/// [`crate::terminate::Cut`]. Its total is the documents let through.
///
/// # Errors
/// As [`search_collapsed`].
#[allow(clippy::too_many_arguments)]
pub fn search_collapsed_until(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    query: &BooleanQuery,
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    sort: &[SortField],
    field: &str,
    kind: CollapseKind,
    top_n: usize,
    after: Option<&FieldDoc>,
    track_max_score: bool,
    n: u64,
) -> Result<(CollapseTopDocs, crate::terminate::Cut)> {
    let cut = crate::terminate::terminate_after(segments, query, n)?;
    let searched: Vec<usize> = (0..cut.leaves.min(readers.len())).collect();
    let until = cut.last.map(|(i, last)| (i, last.saturating_add(1)));
    let mut out = search_collapsed_in(
        segments,
        readers,
        query,
        norms,
        sort,
        field,
        kind,
        top_n,
        after,
        track_max_score,
        &[searched],
        None,
        until,
    )?;
    Ok((out.remove(0), cut))
}

/// What one collapsing search over a list of segments needs.
struct Run<'r, 'a> {
    segments: &'r [OpenSegment<'a>],
    readers: &'r [SegmentReader],
    query: &'r BooleanQuery,
    norms: &'r [Option<&'r HashMap<String, FieldNorms<'a>>>],
    sort: &'r [SortField],
    field: &'r str,
    kind: CollapseKind,
    top_n: usize,
    after: Option<&'r FieldDoc>,
    track: bool,
    global: Option<&'r crate::GlobalStats>,
    min_score: Option<f32>,
}

/// A collected group (`CollectedSearchGroup`).
struct Group {
    value: CollapseValue,
    top_doc: i32,
    slot: usize,
    /// The group's ordinal in the current segment, when a keyword document
    /// of this segment mapped it there.
    ord: Option<i32>,
}

/// The collapse field in one segment.
enum Selector<'a> {
    /// No doc values for the field in this segment (`DocValues.emptyNumeric()`,
    /// `emptySorted()`): every document is missing.
    Empty,
    Numeric(Box<dyn NumericDocValues + 'a>),
    NumericMulti(Box<dyn SortedNumericDocValues + 'a>),
    Keyword(Box<dyn SortedDocValues + 'a>),
    KeywordMulti(Box<dyn SortedSetDocValues + 'a>),
}

/// A document's place in the collapse field: no value, a long, or an
/// ordinal of the segment's dictionary.
enum Key {
    Missing,
    Long(i64),
    Ord(i32),
}

fn multi_valued(doc: i32) -> Error {
    Error::IllegalState(format!(
        "{MULTI_VALUED} {doc}, the collapse field must be single valued"
    ))
}

impl<'a> Selector<'a> {
    /// `setNextReader`: the field's column by its doc-values type.
    fn open(reader: &'a SegmentReader, field: &str, kind: CollapseKind) -> Result<Self> {
        let ty = reader
            .field_infos()
            .field_by_name(field)
            .map_or(DocValuesType::None, |fi| fi.doc_values_type);
        let unexpected = || {
            Error::IllegalState(format!(
                "unexpected doc values type {}` for field `{field}`",
                dv::doc_values_type_name(ty)
            ))
        };
        Ok(match (kind, ty) {
            (_, DocValuesType::None) => Selector::Empty,
            (CollapseKind::Numeric, DocValuesType::Numeric) => {
                Selector::Numeric(dv::get_numeric(reader, field)?)
            }
            (CollapseKind::Numeric, DocValuesType::SortedNumeric) => {
                Selector::NumericMulti(dv::get_sorted_numeric(reader, field)?)
            }
            (CollapseKind::Keyword, DocValuesType::Sorted) => {
                Selector::Keyword(dv::get_sorted(reader, field)?)
            }
            (CollapseKind::Keyword, DocValuesType::SortedSet) => {
                Selector::KeywordMulti(dv::get_sorted_set(reader, field)?)
            }
            _ => return Err(unexpected()),
        })
    }

    /// `advanceTo(doc)`: the document's key.
    fn advance(&mut self, doc: i32) -> Result<Key> {
        Ok(match self {
            Selector::Empty => Key::Missing,
            Selector::Numeric(v) => {
                if v.advance_exact(doc)? {
                    Key::Long(v.long_value())
                } else {
                    Key::Missing
                }
            }
            Selector::NumericMulti(v) => {
                if !v.advance_exact(doc)? {
                    Key::Missing
                } else if v.doc_value_count() > 1 {
                    return Err(multi_valued(doc));
                } else {
                    Key::Long(v.next_value()?)
                }
            }
            Selector::Keyword(v) => {
                if v.advance_exact(doc)? {
                    Key::Ord(v.ord_value())
                } else {
                    Key::Missing
                }
            }
            Selector::KeywordMulti(v) => {
                if !v.advance_exact(doc)? {
                    Key::Missing
                } else {
                    let ord = v.next_ord()?;
                    if v.doc_value_count() != 1 {
                        return Err(multi_valued(doc));
                    }
                    Key::Ord(i32::try_from(ord).map_err(|_| SortError::Ordinal(ord))?)
                }
            }
        })
    }

    /// `lookupOrd(ord)`.
    fn term(&mut self, ord: i32) -> Result<Vec<u8>> {
        match self {
            Selector::Keyword(v) => v.lookup_ord(ord),
            Selector::KeywordMulti(v) => v.lookup_ord(i64::from(ord)),
            _ => Err(SortError::Ordinal(i64::from(ord)).into()),
        }
    }
}

/// `CollapsingTopDocsCollector`'s state across segments.
struct Collapser {
    tf: TopField,
    groups: Vec<Group>,
    /// `groupMap`: a group value to its index in `groups`.
    map: HashMap<CollapseValue, usize>,
    /// This segment's ordinals to their groups (a keyword field).
    ords: HashMap<i32, usize>,
    /// `orderedGroups`, once `topN` groups were collected.
    ordered: Option<TreeSet<usize>>,
    spare: usize,
    top_n: usize,
    total: u64,
    after_doc: Option<i32>,
    doc_base: i32,
}

/// `buildSortedSet`'s comparator: the slots, then the top documents.
fn group_order(tf: &TopField, groups: &[Group], a: usize, b: usize) -> std::cmp::Ordering {
    let (ga, gb) = (&groups[a], &groups[b]);
    for c in &tf.comps {
        let r = c.compare(ga.slot, gb.slot);
        if r != 0 {
            return r.cmp(&0);
        }
    }
    ga.top_doc.cmp(&gb.top_doc)
}

impl Collapser {
    fn new(sort: &[SortField], top_n: usize, after: Option<&FieldDoc>) -> Self {
        let slots = top_n.saturating_add(1);
        let mut tf = TopField::new(sort, slots, u64::MAX, after);
        // `getComparator(topNGroups + 1, Pruning.NONE)`: nothing skips.
        for c in &mut tf.comps {
            c.pruning = Pruning::None;
        }
        tf.doc_first = false;
        tf.can_set_min_score = false;
        Self {
            tf,
            groups: Vec::new(),
            map: HashMap::new(),
            ords: HashMap::new(),
            ordered: None,
            spare: top_n,
            top_n,
            total: 0,
            after_doc: after.map(|a| a.doc),
            doc_base: 0,
        }
    }

    fn set_bottom(&mut self, leaf: &mut Leaf<'_>) -> Result<()> {
        let Some(last) = self.ordered.as_ref().and_then(TreeSet::last) else {
            return Ok(());
        };
        let slot = self.groups[last].slot;
        leaf.set_bottom(&mut self.tf, slot)
    }

    fn build_sorted_set(&mut self) {
        let (tf, groups) = (&self.tf, &self.groups);
        self.ordered = Some(TreeSet::from_items(
            (0..groups.len()).collect(),
            &mut |a: &usize, b: &usize| group_order(tf, groups, *a, *b),
        ));
        // A custom key's comparator sets its bottom on a new segment once
        // the groups are full.
        self.tf.queue_full = true;
    }

    /// `CollapsingTopDocsCollector.collect(doc)`.
    fn collect(
        &mut self,
        leaf: &mut Leaf<'_>,
        selector: &mut Selector<'_>,
        doc: i32,
        score: f32,
    ) -> Result<()> {
        self.total += 1;
        leaf.score = score;
        leaf.score_doc = doc;
        if let Some(after_doc) = self.after_doc {
            // `isAfterDoc`: the top first, then the document -- the
            // segment's own id against the page's global one, as Java
            // compares them.
            let cmp = leaf.compare_top(&self.tf, doc, &mut None)?;
            let is_after = if cmp != 0 { cmp < 0 } else { doc > after_doc };
            if !is_after {
                return Ok(());
            }
        }
        // `isCompetitive`.
        if self.ordered.is_some() && leaf.compare_bottom(&self.tf, doc, &mut None)? <= 0 {
            return Ok(());
        }
        let key = selector.advance(doc)?;
        let group = match key {
            Key::Missing => self.map.get(&CollapseValue::Missing).copied(),
            Key::Long(v) => self.map.get(&CollapseValue::Long(v)).copied(),
            Key::Ord(o) => match self.ords.get(&o) {
                Some(&g) => Some(g),
                None => {
                    let term = selector.term(o)?;
                    let g = self.map.get(&CollapseValue::Bytes(term)).copied();
                    if let Some(g) = g {
                        self.ords.insert(o, g);
                        self.groups[g].ord = Some(o);
                    }
                    g
                }
            },
        };
        match group {
            None => self.collect_new_group(leaf, selector, key, doc),
            Some(g) => self.collect_existing_group(leaf, g, doc),
        }
    }

    /// The value `key` names (`copyValue`).
    fn value_of(selector: &mut Selector<'_>, key: &Key) -> Result<CollapseValue> {
        Ok(match *key {
            Key::Missing => CollapseValue::Missing,
            Key::Long(v) => CollapseValue::Long(v),
            Key::Ord(o) => CollapseValue::Bytes(selector.term(o)?),
        })
    }

    /// `collectNewGroup(doc)`.
    fn collect_new_group(
        &mut self,
        leaf: &mut Leaf<'_>,
        selector: &mut Selector<'_>,
        key: Key,
        doc: i32,
    ) -> Result<()> {
        let value = Self::value_of(selector, &key)?;
        let ord = match key {
            Key::Ord(o) => Some(o),
            _ => None,
        };
        if self.groups.len() < self.top_n {
            let slot = self.groups.len();
            leaf.copy(&mut self.tf, slot, doc, &mut None)?;
            let g = self.groups.len();
            self.map.insert(value.clone(), g);
            if let Some(o) = ord {
                self.ords.insert(o, g);
            }
            self.groups.push(Group {
                value,
                top_doc: self.doc_base + doc,
                slot,
                ord,
            });
            if self.groups.len() >= self.top_n {
                self.build_sorted_set();
                self.set_bottom(leaf)?;
            }
            return Ok(());
        }
        let Some(bottom) = self.ordered.as_mut().and_then(TreeSet::poll_last) else {
            return Ok(());
        };
        let old = std::mem::replace(&mut self.groups[bottom].value, value.clone());
        self.map.remove(&old);
        if let Some(o) = self.groups[bottom].ord.take() {
            self.ords.remove(&o);
        }
        self.groups[bottom].top_doc = self.doc_base + doc;
        self.groups[bottom].ord = ord;
        let slot = self.groups[bottom].slot;
        leaf.copy(&mut self.tf, slot, doc, &mut None)?;
        self.map.insert(value, bottom);
        if let Some(o) = ord {
            self.ords.insert(o, bottom);
        }
        let (tf, groups) = (&self.tf, &self.groups);
        if let Some(ordered) = self.ordered.as_mut() {
            ordered.add(bottom, &mut |a: &usize, b: &usize| {
                group_order(tf, groups, *a, *b)
            });
        }
        self.set_bottom(leaf)
    }

    /// `collectExistingGroup(doc, group)`.
    fn collect_existing_group(&mut self, leaf: &mut Leaf<'_>, g: usize, doc: i32) -> Result<()> {
        leaf.copy(&mut self.tf, self.spare, doc, &mut None)?;
        let slot = self.groups[g].slot;
        let mut better = false;
        for c in &self.tf.comps {
            let r = c.compare(slot, self.spare);
            if r != 0 {
                better = r > 0;
                break;
            }
        }
        if !better {
            return Ok(());
        }
        let mut prev_last = None;
        let mut skip_heavy_ops = false;
        if let Some(ordered) = self.ordered.as_mut() {
            prev_last = ordered.last();
            if ordered.first() == Some(g) {
                skip_heavy_ops = true;
            } else {
                let (tf, groups) = (&self.tf, &self.groups);
                ordered.remove(&g, &mut |a: &usize, b: &usize| {
                    group_order(tf, groups, *a, *b)
                });
            }
        }
        self.groups[g].top_doc = self.doc_base + doc;
        std::mem::swap(&mut self.spare, &mut self.groups[g].slot);
        if self.ordered.is_some() {
            let (tf, groups) = (&self.tf, &self.groups);
            let ordered = self.ordered.as_mut().ok_or(SortError::NoKeys)?;
            if !skip_heavy_ops {
                ordered.add(g, &mut |a: &usize, b: &usize| {
                    group_order(tf, groups, *a, *b)
                });
            }
            let new_last = ordered.last();
            if new_last == Some(g) || prev_last != new_last {
                self.set_bottom(leaf)?;
            }
        }
        Ok(())
    }

    /// `getTopDocs()`.
    fn top_docs(mut self, max_score: f32) -> CollapseTopDocs {
        if self.ordered.is_none() && !self.groups.is_empty() {
            self.build_sorted_set();
        }
        let any_terms = self.tf.any_terms();
        let order: Vec<usize> = self
            .ordered
            .as_ref()
            .map(|o| o.iter().copied().collect())
            .unwrap_or_default();
        let mut hits = Vec::with_capacity(order.len());
        let mut values = Vec::with_capacity(order.len());
        for g in order {
            let group = &self.groups[g];
            hits.push(self.tf.slot_hit(group.slot, group.top_doc, any_terms));
            values.push(group.value.clone());
        }
        CollapseTopDocs {
            hits,
            values,
            total: self.total,
            max_score,
        }
    }
}

/// The leaf collector as a [`ScoringCollector`]: every match, behind the
/// `min_score` test, toward the max score and then the collapse.
struct CollapseLeaf<'l, 'a> {
    c: &'l mut Collapser,
    leaf: &'l mut Leaf<'a>,
    selector: &'l mut Selector<'a>,
    min_score: Option<f32>,
    max: f32,
    error: Option<Error>,
}

impl ScoringCollector for CollapseLeaf<'_, '_> {
    fn collect(&mut self, doc_id: i32, score: f32) {
        if self.error.is_some() || !passes(self.min_score, score) {
            return;
        }
        self.max = self.max.max(score);
        if let Err(e) = self.c.collect(self.leaf, self.selector, doc_id, score) {
            self.error = Some(e);
        }
    }

    fn min_competitive_score(&self) -> Option<f32> {
        None
    }

    fn score_mode(&self) -> ScoreMode {
        ScoreMode::Complete
    }
}

/// One collector over the segments at `order`, in that order; the segment
/// `until.0` only below doc `until.1`.
fn collapse_segments(
    run: &Run<'_, '_>,
    order: &[usize],
    until: Option<(usize, i32)>,
) -> Result<CollapseTopDocs> {
    let mut c = Collapser::new(run.sort, run.top_n, run.after);
    let want_scores = run.sort.iter().any(key_needs_scores) || run.track || run.min_score.is_some();
    let mode = if want_scores {
        Mode::Complete
    } else {
        Mode::NoScores
    };
    let mut max_score = f32::NEG_INFINITY;
    for &i in order {
        let (Some(seg), Some(reader)) = (run.segments.get(i), run.readers.get(i)) else {
            return Err(Error::SliceOutOfRange {
                segment: i,
                segments: run.segments.len().min(run.readers.len()),
            });
        };
        let ctx = exec::LeafContext {
            fields: seg.fields,
            doc_in: seg.doc_in,
            pos_in: seg.pos_in,
            pay_in: seg.pay_in,
            live_docs: seg.live_docs,
            points: seg.points,
            norms: run.norms.get(i).copied().flatten(),
            global: run.global,
            max_doc: seg.max_doc,
            cache: seg.cache,
            reader: seg.reader,
            similarity: None,
        };
        let Some(mut bulk) = leaf_bulk(&ctx, run.query, mode, false)? else {
            continue;
        };
        // `doSetNextReader`: the comparators, then the selector.
        let mut leaf = open_leaf(&mut c.tf, reader, seg, i64::from(reader.max_doc))?;
        let mut selector = Selector::open(reader, run.field, run.kind)?;
        c.doc_base = seg.doc_base;
        c.ords.clear();
        for g in &mut c.groups {
            g.ord = None;
        }
        let mut lc = CollapseLeaf {
            c: &mut c,
            leaf: &mut leaf,
            selector: &mut selector,
            min_score: run.min_score,
            max: f32::NEG_INFINITY,
            error: None,
        };
        let end = match until {
            Some((u, end)) if u == i => end,
            _ => NO_MORE_DOCS,
        };
        exec::score_segment_below(&mut bulk, mode, seg.live_docs, &mut lc, end)?;
        if let Some(e) = lc.error {
            return Err(e);
        }
        max_score = max_score.max(lc.max);
        leaf.finish(&mut c.tf)?;
    }
    let max = if run.track && max_score.is_finite() {
        max_score
    } else {
        f32::NAN
    };
    Ok(c.top_docs(max))
}

#[cfg(test)]
mod tests {
    use super::super::{search_sorted, SortType};
    use super::*;
    use crate::directory_reader::DirectoryReader;
    use crate::query::{Clause, MatchAllDocsQuery, TermQuery};
    use lucene_store::FsDirectory;

    fn fixture(name: &str) -> DirectoryReader {
        let dir = format!("{}/../../fixtures/data/{name}", env!("CARGO_MANIFEST_DIR"));
        DirectoryReader::open(&FsDirectory::open(dir)).expect("open fixture")
    }

    fn all() -> BooleanQuery {
        let mut q = BooleanQuery::new();
        q.must.push(Clause::MatchAllDocs(MatchAllDocsQuery::new(0)));
        q
    }

    fn term(field: &str, text: &str) -> BooleanQuery {
        let mut q = BooleanQuery::new();
        q.must
            .push(Clause::Term(TermQuery::new(field, text.as_bytes())));
        q
    }

    /// Every live document's collapse value, by global id.
    fn values(reader: &DirectoryReader, field: &str, kind: CollapseKind) -> Vec<CollapseValue> {
        let mut out = Vec::new();
        for r in reader.segment_readers() {
            match kind {
                CollapseKind::Numeric => {
                    let mut v = dv::get_sorted_numeric(r, field).unwrap();
                    for doc in 0..r.max_doc {
                        out.push(if v.advance_exact(doc).unwrap() {
                            CollapseValue::Long(v.next_value().unwrap())
                        } else {
                            CollapseValue::Missing
                        });
                    }
                }
                CollapseKind::Keyword => {
                    let mut v = dv::get_sorted_set(r, field).unwrap();
                    for doc in 0..r.max_doc {
                        out.push(if v.advance_exact(doc).unwrap() {
                            let o = v.next_ord().unwrap();
                            CollapseValue::Bytes(v.lookup_ord(o).unwrap())
                        } else {
                            CollapseValue::Missing
                        });
                    }
                }
            }
        }
        out
    }

    /// The collapse by its definition: every match in the sort's order (ties
    /// to the lower document), each group's first one kept, the first
    /// `top_n` groups.
    fn brute(
        reader: &DirectoryReader,
        q: &BooleanQuery,
        sort: &[SortField],
        field: &str,
        kind: CollapseKind,
        top_n: usize,
    ) -> (Vec<FieldDoc>, Vec<CollapseValue>, u64) {
        let mut opened = reader.open_segments().unwrap();
        opened.open_points().unwrap();
        let segments = opened.as_open_segments();
        let norms = vec![None; segments.len()];
        let every = search_sorted(
            &segments,
            reader.segment_readers(),
            q,
            &norms,
            sort,
            reader.max_doc() as usize,
            u64::MAX,
            None,
        )
        .unwrap();
        let by_doc = values(reader, field, kind);
        let mut seen = std::collections::HashSet::new();
        let (mut hits, mut groups) = (Vec::new(), Vec::new());
        for hit in every.hits {
            let v = by_doc[hit.doc as usize].clone();
            if seen.insert(v.clone()) && hits.len() < top_n {
                hits.push(hit);
                groups.push(v);
            }
        }
        (hits, groups, every.total.value)
    }

    #[allow(clippy::too_many_arguments)]
    fn collapsed(
        reader: &DirectoryReader,
        q: &BooleanQuery,
        sort: &[SortField],
        field: &str,
        kind: CollapseKind,
        top_n: usize,
        after: Option<&FieldDoc>,
        slices: &[Vec<usize>],
    ) -> Result<Vec<CollapseTopDocs>> {
        let mut opened = reader.open_segments().unwrap();
        opened.open_points().unwrap();
        let segments = opened.as_open_segments();
        let norms = vec![None; segments.len()];
        search_collapsed(
            &segments,
            reader.segment_readers(),
            q,
            &norms,
            sort,
            field,
            kind,
            top_n,
            after,
            false,
            slices,
            None,
        )
    }

    fn keyword(field: &str, reverse: bool, last: bool) -> SortField {
        SortField {
            missing: i64::from(last),
            ..SortField::string(field, reverse)
        }
    }

    #[test]
    fn collapsing_keeps_each_groups_best_document_and_the_best_groups() {
        let numeric = fixture("sorted_search_index");
        let words = fixture("keyword_sort_index");
        let long = |f: &str, reverse| SortField::numeric(f, SortType::Long, reverse);
        let cases: Vec<(
            &DirectoryReader,
            BooleanQuery,
            Vec<SortField>,
            &str,
            CollapseKind,
        )> = vec![
            (
                &numeric,
                all(),
                vec![long("l", false)],
                "i",
                CollapseKind::Numeric,
            ),
            (
                &numeric,
                all(),
                vec![long("l", true), SortField::doc()],
                "i",
                CollapseKind::Numeric,
            ),
            (
                &numeric,
                term("body", "w3"),
                vec![SortField::score()],
                "s",
                CollapseKind::Numeric,
            ),
            (
                &numeric,
                term("body", "w0"),
                vec![
                    SortField::numeric("i", SortType::Int, false),
                    SortField::score(),
                ],
                "l",
                CollapseKind::Numeric,
            ),
            (
                &numeric,
                all(),
                vec![SortField::doc()],
                "l",
                CollapseKind::Numeric,
            ),
            (
                &words,
                all(),
                vec![keyword("kd", false, true)],
                "k",
                CollapseKind::Keyword,
            ),
            (
                &words,
                all(),
                vec![
                    keyword("k", true, false),
                    SortField::numeric("i", SortType::Int, false),
                ],
                "kd",
                CollapseKind::Keyword,
            ),
            (
                &words,
                all(),
                vec![SortField::numeric("i", SortType::Int, true)],
                "kn",
                CollapseKind::Keyword,
            ),
        ];
        for (reader, q, sort, field, kind) in &cases {
            for top_n in [1, 7, 120, 5000] {
                let (want_hits, want_values, want_total) =
                    brute(reader, q, sort, field, *kind, top_n);
                for slices in [vec![], vec![vec![2, 0, 1]]] {
                    let got =
                        collapsed(reader, q, sort, field, *kind, top_n, None, &slices).unwrap();
                    assert_eq!(got.len(), 1);
                    let got = &got[0];
                    let docs = |h: &[FieldDoc]| h.iter().map(|h| h.doc).collect::<Vec<_>>();
                    assert_eq!(
                        docs(&got.hits),
                        docs(&want_hits),
                        "{field} {sort:?} top {top_n}"
                    );
                    assert_eq!(got.hits, want_hits, "{field} {sort:?} top {top_n}");
                    assert_eq!(got.values, want_values, "{field} {sort:?} top {top_n}");
                    assert_eq!(got.total, want_total);
                    assert!(got.max_score.is_nan());
                }
            }
        }
    }

    #[test]
    fn slices_each_collapse_on_their_own() {
        let reader = fixture("sorted_search_index");
        let sort = vec![SortField::numeric("l", SortType::Long, false)];
        let slices = vec![vec![1], vec![0, 2]];
        let got = collapsed(
            &reader,
            &all(),
            &sort,
            "i",
            CollapseKind::Numeric,
            10,
            None,
            &slices,
        )
        .unwrap();
        assert_eq!(got.len(), 2);
        let per_slice: u64 = got.iter().map(|g| g.total).sum();
        assert_eq!(per_slice, u64::from(reader.num_docs() as u32));
        // The second slice's segments: its hits come from segments 0 and 2.
        let bases: Vec<i32> = reader
            .segment_readers()
            .iter()
            .scan(0, |b, r| {
                let base = *b;
                *b += r.max_doc;
                Some(base)
            })
            .collect();
        for h in &got[0].hits {
            assert!(h.doc >= bases[1] && h.doc < bases[2], "{}", h.doc);
        }
        for h in &got[1].hits {
            assert!(h.doc < bases[1] || h.doc >= bases[2], "{}", h.doc);
        }
    }

    #[test]
    fn search_after_continues_past_the_previous_page() {
        let reader = fixture("keyword_sort_index");
        for reverse in [false, true] {
            let sort = vec![keyword("k", reverse, false)];
            let (every, values, _) =
                brute(&reader, &all(), &sort, "k", CollapseKind::Keyword, 10_000);
            let page = 25;
            let first = &collapsed(
                &reader,
                &all(),
                &sort,
                "k",
                CollapseKind::Keyword,
                page,
                None,
                &[],
            )
            .unwrap()[0];
            assert_eq!(first.values, values[..page]);
            let last = first.hits.last().unwrap().clone();
            let second = &collapsed(
                &reader,
                &all(),
                &sort,
                "k",
                CollapseKind::Keyword,
                page,
                Some(&last),
                &[],
            )
            .unwrap()[0];
            assert_eq!(second.values, values[page..2 * page], "reverse {reverse}");
            assert_eq!(second.hits[0].doc, every[page].doc);
            // Every match is counted, before the page or after it.
            assert_eq!(second.total, first.total);
        }
        let numeric = fixture("sorted_search_index");
        let sort = vec![SortField::numeric("i", SortType::Int, true)];
        let first = &collapsed(
            &numeric,
            &all(),
            &sort,
            "i",
            CollapseKind::Numeric,
            3,
            None,
            &[],
        )
        .unwrap()[0];
        let next = &collapsed(
            &numeric,
            &all(),
            &sort,
            "i",
            CollapseKind::Numeric,
            3,
            first.hits.last(),
            &[],
        )
        .unwrap()[0];
        let (_, values, _) = brute(&numeric, &all(), &sort, "i", CollapseKind::Numeric, 6);
        assert_eq!(first.values, values[..3]);
        // Java's search_after: a later document tied with the last hit on its
        // value opens that group again on the next page.
        let mut both = [first.values.clone(), next.values.clone()].concat();
        assert!(
            next.values[0] == first.values[2] || next.values[0] == values[3],
            "{both:?}"
        );
        both.dedup();
        assert_eq!(both, values[..both.len()]);
    }

    #[test]
    fn scores_track_and_min_score_filter() {
        let reader = fixture("sorted_search_index");
        let mut opened = reader.open_segments().unwrap();
        opened.open_points().unwrap();
        let segments = opened.as_open_segments();
        let norms = vec![None; segments.len()];
        let q = term("body", "w0");
        let sort = vec![SortField::numeric("l", SortType::Long, false)];
        let run = |track, min| {
            search_collapsed(
                &segments,
                reader.segment_readers(),
                &q,
                &norms,
                &sort,
                "i",
                CollapseKind::Numeric,
                5,
                None,
                track,
                &[],
                min,
            )
            .unwrap()
            .remove(0)
        };
        let tracked = run(true, None);
        assert!(tracked.max_score > 0.0);
        let by_score = search_sorted(
            &segments,
            reader.segment_readers(),
            &q,
            &norms,
            &[SortField::score()],
            1,
            u64::MAX,
            None,
        )
        .unwrap();
        assert_eq!(
            tracked.max_score.to_bits(),
            by_score.hits[0].values[0] as u32
        );
        let none = run(true, Some(f32::MAX));
        assert_eq!((none.total, none.hits.len()), (0, 0));
        assert!(none.max_score.is_nan());
        // A threshold between the distinct scores: only the documents above it count.
        let every = search_sorted(
            &segments,
            reader.segment_readers(),
            &q,
            &norms,
            &[SortField::score()],
            100_000,
            u64::MAX,
            None,
        )
        .unwrap();
        let mut scores: Vec<f32> = every
            .hits
            .iter()
            .map(|h| f32::from_bits(h.values[0] as u32))
            .collect();
        scores.dedup();
        assert!(scores.len() > 1, "{scores:?}");
        let cut = scores[scores.len() / 2];
        let above = every
            .hits
            .iter()
            .filter(|h| f32::from_bits(h.values[0] as u32) >= cut)
            .count() as u64;
        let some = run(false, Some(cut));
        assert_eq!(some.total, above);
        assert!(some.total > 0 && some.total < tracked.total);
    }

    #[test]
    fn refusals_are_javas() {
        let reader = fixture("sorted_search_index");
        let sort = vec![SortField::numeric("l", SortType::Long, false)];
        // A multi-valued field, once a document with two values competes.
        let e = collapsed(
            &reader,
            &all(),
            &sort,
            "m",
            CollapseKind::Numeric,
            10,
            None,
            &[],
        )
        .unwrap_err();
        assert!(e.to_string().contains(MULTI_VALUED), "{e}");
        let words = fixture("keyword_sort_index");
        let e = collapsed(
            &words,
            &all(),
            &[keyword("k", false, true)],
            "km",
            CollapseKind::Keyword,
            10,
            None,
            &[],
        )
        .unwrap_err();
        assert!(e.to_string().contains(MULTI_VALUED), "{e}");
        // The wrong kind of doc values for the selector.
        let e = collapsed(
            &words,
            &all(),
            &sort,
            "kd",
            CollapseKind::Numeric,
            10,
            None,
            &[],
        )
        .unwrap_err();
        assert!(e.to_string().contains("unexpected doc values type"), "{e}");
        let e = collapsed(
            &reader,
            &all(),
            &sort,
            "l",
            CollapseKind::Keyword,
            10,
            None,
            &[],
        )
        .unwrap_err();
        assert!(e.to_string().contains("unexpected doc values type"), "{e}");
        // A field no segment has: one group of documents without a value.
        let none = &collapsed(
            &reader,
            &all(),
            &sort,
            "absent",
            CollapseKind::Keyword,
            10,
            None,
            &[],
        )
        .unwrap()[0];
        assert_eq!(none.values, vec![CollapseValue::Missing]);
        // The arguments OpenSearch checks.
        assert!(collapsed(
            &reader,
            &all(),
            &[],
            "i",
            CollapseKind::Numeric,
            10,
            None,
            &[]
        )
        .is_err());
        assert!(collapsed(
            &reader,
            &all(),
            &sort,
            "i",
            CollapseKind::Numeric,
            0,
            None,
            &[]
        )
        .is_err());
        let two = vec![sort[0].clone(), SortField::doc()];
        let after = FieldDoc {
            doc: 0,
            values: vec![0, 0],
            terms: Vec::new(),
        };
        assert!(collapsed(
            &reader,
            &all(),
            &two,
            "i",
            CollapseKind::Numeric,
            10,
            Some(&after),
            &[]
        )
        .is_err());
        assert!(collapsed(
            &reader,
            &all(),
            &sort,
            "i",
            CollapseKind::Numeric,
            10,
            None,
            &[vec![99]]
        )
        .is_err());
    }
}
