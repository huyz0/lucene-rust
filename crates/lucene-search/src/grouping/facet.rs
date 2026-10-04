//! Grouped facets: `GroupFacetCollector` and its `TermGroupFacetCollector`
//! (`SV` for a `SORTED` facet field, `MV` for a `SORTED_SET` one) -- facet
//! counts in which each (group, facet value) pair counts once, merged
//! across segments by term.

use std::collections::{HashMap, HashSet};

use crate::collector::ScoreMode;
use crate::leaf_collector::SegmentCollector;
use crate::multi_segment::OpenSegment;
use crate::reader::doc_values as dv;
use crate::reader::{SortedDocValues, SortedSetDocValues};
use crate::{Error, Result};

/// `UnicodeUtil.BIG_TERM`: ten `0xff` bytes, after every UTF-8 term.
const BIG_TERM: [u8; 10] = [0xff; 10];

/// `GroupFacetCollector.FacetEntry`: a facet value and its count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FacetEntry {
    pub value: Vec<u8>,
    pub count: i32,
}

/// `GroupFacetCollector.GroupedFacetResult`: the facet entries kept --
/// by count (highest first, ties by value) or by value -- with the total
/// and missing counts.
#[derive(Debug, Clone)]
pub struct GroupedFacetResult {
    max_size: usize,
    order_by_count: bool,
    /// `facetEntries`, a `TreeSet` in its comparator's order.
    entries: Vec<FacetEntry>,
    total_missing_count: i32,
    total_count: i32,
    current_min: i32,
}

impl GroupedFacetResult {
    /// `new GroupedFacetResult(size, minCount, orderByCount, totalCount,
    /// totalMissingCount)`.
    pub fn new(
        size: usize,
        min_count: i32,
        order_by_count: bool,
        total_count: i32,
        total_missing_count: i32,
    ) -> Self {
        Self {
            max_size: size,
            order_by_count,
            entries: Vec::new(),
            total_missing_count,
            total_count,
            current_min: min_count,
        }
    }

    /// The set's comparator.
    fn cmp(&self, a: &FacetEntry, b: &FacetEntry) -> std::cmp::Ordering {
        if self.order_by_count {
            // `b.count - a.count`: highest count first, then by value.
            b.count
                .wrapping_sub(a.count)
                .cmp(&0)
                .then_with(|| a.value.cmp(&b.value))
        } else {
            a.value.cmp(&b.value)
        }
    }

    /// `addFacetCount(facetValue, count)`.
    pub fn add_facet_count(&mut self, value: Vec<u8>, count: i32) {
        if count < self.current_min {
            return;
        }
        let entry = FacetEntry { value, count };
        if self.entries.len() == self.max_size {
            // `facetEntries.higher(entry) == null`: nothing sorts after it.
            let higher = self
                .entries
                .iter()
                .any(|e| self.cmp(e, &entry) == std::cmp::Ordering::Greater);
            if !higher {
                return;
            }
            self.entries.pop();
        }
        match self.entries.binary_search_by(|e| self.cmp(e, &entry)) {
            Ok(_) => {}
            Err(i) => self.entries.insert(i, entry),
        }
        if self.entries.len() == self.max_size {
            if let Some(last) = self.entries.last() {
                self.current_min = last.count;
            }
        }
    }

    /// `getFacetEntries(offset, limit)`.
    pub fn facet_entries(&self, offset: usize, limit: usize) -> Vec<FacetEntry> {
        self.entries
            .iter()
            .skip(offset)
            .take(limit)
            .cloned()
            .collect()
    }

    /// `getTotalCount()`.
    pub fn total_count(&self) -> i32 {
        self.total_count
    }

    /// `getTotalMissingCount()`.
    pub fn total_missing_count(&self) -> i32 {
        self.total_missing_count
    }
}

/// A segment's facet doc values: one `SORTED` field (`SV`) or a
/// `SORTED_SET` one (`MV`).
enum FacetValues<'a> {
    Single(Box<dyn SortedDocValues + 'a>),
    Multi(Box<dyn SortedSetDocValues + 'a>),
}

impl FacetValues<'_> {
    fn lookup_ord(&mut self, ord: i32) -> Result<Vec<u8>> {
        match self {
            Self::Single(v) => v.lookup_ord(ord),
            Self::Multi(v) => v.lookup_ord(i64::from(ord)),
        }
    }
}

/// `GroupFacetCollector.SegmentResult`: one segment's counts, merged by
/// term.
struct SegmentResult<'a> {
    counts: Vec<i32>,
    total: i32,
    missing: i32,
    max_term_pos: i32,
    merge_pos: i32,
    /// The term at `merge_pos` (`mergeTerm`).
    merge_term: Option<Vec<u8>>,
    values: FacetValues<'a>,
    /// What `merge_pos` is off the ordinal by (`SV` counts sit one up, the
    /// missing count at `0`).
    shift: i32,
}

impl SegmentResult<'_> {
    /// `nextTerm()`: `tenum.next()`.
    fn next_term(&mut self) -> Result<()> {
        let ord = self.merge_pos.wrapping_sub(self.shift);
        self.merge_term = Some(self.values.lookup_ord(ord)?);
        Ok(())
    }
}

/// `TermGroupFacetCollector.GroupedFacetHit`: a (group, facet value) pair,
/// `None` for a missing one.
type GroupedFacetHit = (Option<Vec<u8>>, Option<Vec<u8>>);

/// `TermGroupFacetCollector` (`createTermGroupFacetCollector(groupField,
/// facetField, facetFieldMultivalued, facetPrefix, initialSize)`).
pub struct TermGroupFacetCollector<'a> {
    group_field: String,
    facet_field: String,
    facet_prefix: Option<Vec<u8>>,
    multivalued: bool,
    /// `groupedFacetHits`: every (group, facet value) pair counted.
    grouped_facet_hits: Vec<GroupedFacetHit>,
    /// `segmentGroupedFacetHits`: the segment's pairs, as
    /// `groupOrd * (facetValueCount + 1) + facetOrd` in `int` arithmetic.
    segment_grouped_facet_hits: HashSet<i32>,
    segment_results: Vec<SegmentResult<'a>>,
    segment_facet_counts: Vec<i32>,
    segment_total_count: i32,
    start_facet_ord: i32,
    end_facet_ord: i32,
    group_index: Option<Box<dyn SortedDocValues + 'a>>,
    facet: Option<FacetValues<'a>>,
    /// `MV`'s `facetFieldNumTerms`.
    facet_num_terms: i32,
}

/// `seekCeil(term)` over a `SORTED_SET` dictionary: the first ordinal at
/// or after `term`, `value_count` when none is.
fn seek_ceil(values: &mut dyn SortedSetDocValues, term: &[u8]) -> Result<i64> {
    let ord = values.lookup_term(term)?;
    // SENTINEL: a negative ordinal is `-insertionPoint - 1`.
    Ok(if ord < 0 {
        ord.saturating_neg().saturating_sub(1)
    } else {
        ord
    })
}

/// A segment's `lookupTerm` answers, by term.
#[derive(Default)]
struct TermOrds<'k>(HashMap<&'k [u8], i64>);

impl<'k> TermOrds<'k> {
    /// `term`'s ordinal (or `-insertionPoint - 1`), looked up once.
    fn get(&mut self, term: &'k [u8], lookup: impl FnOnce(&[u8]) -> Result<i64>) -> Result<i64> {
        if let Some(&o) = self.0.get(term) {
            return Ok(o);
        }
        let o = lookup(term)?;
        self.0.insert(term, o);
        Ok(o)
    }
}

/// `BytesRefBuilder(facetPrefix).append(BIG_TERM)`.
fn end_prefix(prefix: &[u8]) -> Vec<u8> {
    let mut end = prefix.to_vec();
    end.extend_from_slice(&BIG_TERM);
    end
}

/// A segment ordinal as Java's `int`.
fn as_int(ord: i64) -> i32 {
    i32::try_from(ord).unwrap_or(i32::MAX)
}

impl<'a> TermGroupFacetCollector<'a> {
    /// `TermGroupFacetCollector.createTermGroupFacetCollector(groupField,
    /// facetField, facetFieldMultivalued, facetPrefix, initialSize)`.
    pub fn new(
        group_field: &str,
        facet_field: &str,
        facet_field_multivalued: bool,
        facet_prefix: Option<Vec<u8>>,
    ) -> Self {
        Self {
            group_field: group_field.to_string(),
            facet_field: facet_field.to_string(),
            facet_prefix,
            multivalued: facet_field_multivalued,
            grouped_facet_hits: Vec::new(),
            segment_grouped_facet_hits: HashSet::new(),
            segment_results: Vec::new(),
            segment_facet_counts: Vec::new(),
            segment_total_count: 0,
            start_facet_ord: 0,
            end_facet_ord: 0,
            group_index: None,
            facet: None,
            facet_num_terms: 0,
        }
    }

    /// `mergeSegmentResults(size, minCount, orderByCount)`: the segments'
    /// counts summed term by term into the top `size` entries.
    ///
    /// # Errors
    /// What reading a dictionary reports.
    pub fn merge_segment_results(
        &mut self,
        size: usize,
        min_count: i32,
        order_by_count: bool,
    ) -> Result<GroupedFacetResult> {
        let mut total_count = 0i32;
        let mut missing_count = 0i32;
        let mut segments: Vec<usize> = Vec::new();
        for (i, r) in self.segment_results.iter_mut().enumerate() {
            missing_count = missing_count.wrapping_add(r.missing);
            if r.merge_pos >= r.max_term_pos {
                continue;
            }
            total_count = total_count.wrapping_add(r.total);
            segments.push(i);
        }
        let mut result =
            GroupedFacetResult::new(size, min_count, order_by_count, total_count, missing_count);
        // `SegmentResultPriorityQueue`: the segment whose current term sorts
        // first.
        let results = &mut self.segment_results;
        let top = |segments: &[usize], results: &[SegmentResult<'_>]| -> Option<usize> {
            segments
                .iter()
                .copied()
                .min_by(|&a, &b| results[a].merge_term.cmp(&results[b].merge_term))
        };
        while let Some(mut s) = top(&segments, results) {
            let current = results[s].merge_term.clone();
            let mut count = 0i32;
            loop {
                let r = &mut results[s];
                let pos = usize::try_from(r.merge_pos).unwrap_or(usize::MAX);
                count = count.wrapping_add(r.counts.get(pos).copied().unwrap_or(0));
                r.merge_pos = r.merge_pos.wrapping_add(1);
                if r.merge_pos < r.max_term_pos {
                    r.next_term()?;
                } else {
                    segments.retain(|&x| x != s);
                }
                match top(&segments, results) {
                    Some(t) if results[t].merge_term == current => s = t,
                    _ => break,
                }
            }
            result.add_facet_count(current.unwrap_or_default(), count);
        }
        Ok(result)
    }

    /// `segmentGroupedFacetsIndex`.
    fn pair_index(group_ord: i32, value_count: i32, facet_ord: i32) -> i32 {
        group_ord
            .wrapping_mul(value_count.wrapping_add(1))
            .wrapping_add(facet_ord)
    }

    fn group_ord(&mut self, doc: i32) -> Result<i32> {
        let g = self
            .group_index
            .as_mut()
            .ok_or_else(|| Error::IllegalState("collected before entering a segment".into()))?;
        if doc > g.doc_id() {
            g.advance(doc)?;
        }
        Ok(if doc == g.doc_id() { g.ord_value() } else { -1 })
    }

    fn group_key(&mut self, group_ord: i32) -> Result<Option<Vec<u8>>> {
        match (group_ord, self.group_index.as_mut()) {
            (-1, _) | (_, None) => Ok(None),
            (ord, Some(g)) => Ok(Some(g.lookup_ord(ord)?)),
        }
    }

    /// `MV.process(groupOrd, facetOrd)`.
    fn process(&mut self, group_ord: i32, facet_ord: i32) -> Result<()> {
        if facet_ord < self.start_facet_ord || facet_ord >= self.end_facet_ord {
            return Ok(());
        }
        let index = Self::pair_index(group_ord, self.facet_num_terms, facet_ord);
        if !self.segment_grouped_facet_hits.insert(index) {
            return Ok(());
        }
        self.segment_total_count = self.segment_total_count.wrapping_add(1);
        if let Some(c) = usize::try_from(facet_ord)
            .ok()
            .and_then(|i| self.segment_facet_counts.get_mut(i))
        {
            *c = c.wrapping_add(1);
        }
        let group_key = self.group_key(group_ord)?;
        let facet_value = if facet_ord == self.facet_num_terms {
            None
        } else {
            match self.facet.as_mut() {
                Some(f) => Some(f.lookup_ord(facet_ord)?),
                None => None,
            }
        };
        self.grouped_facet_hits.push((group_key, facet_value));
        Ok(())
    }

    fn collect_sv(&mut self, doc: i32) -> Result<()> {
        let Some(FacetValues::Single(f)) = self.facet.as_mut() else {
            return Ok(());
        };
        if doc > f.doc_id() {
            f.advance(doc)?;
        }
        let facet_ord = if doc == f.doc_id() { f.ord_value() } else { -1 };
        let value_count = f.value_count();
        if facet_ord < self.start_facet_ord || facet_ord >= self.end_facet_ord {
            return Ok(());
        }
        let group_ord = self.group_ord(doc)?;
        let index = Self::pair_index(group_ord, value_count, facet_ord);
        if !self.segment_grouped_facet_hits.insert(index) {
            return Ok(());
        }
        self.segment_total_count = self.segment_total_count.wrapping_add(1);
        if let Some(c) = usize::try_from(facet_ord.wrapping_add(1))
            .ok()
            .and_then(|i| self.segment_facet_counts.get_mut(i))
        {
            *c = c.wrapping_add(1);
        }
        let group_key = self.group_key(group_ord)?;
        let facet_key = match (facet_ord, self.facet.as_mut()) {
            (-1, _) | (_, None) => None,
            (ord, Some(f)) => Some(f.lookup_ord(ord)?),
        };
        self.grouped_facet_hits.push((group_key, facet_key));
        Ok(())
    }

    fn collect_mv(&mut self, doc: i32) -> Result<()> {
        let group_ord = self.group_ord(doc)?;
        if self.facet_num_terms == 0 {
            let index = Self::pair_index(group_ord, 0, 0);
            if self.facet_prefix.is_some() || self.segment_grouped_facet_hits.contains(&index) {
                return Ok(());
            }
            self.segment_total_count = self.segment_total_count.wrapping_add(1);
            if let Some(c) = self.segment_facet_counts.first_mut() {
                *c = c.wrapping_add(1);
            }
            self.segment_grouped_facet_hits.insert(index);
            let group_key = self.group_key(group_ord)?;
            self.grouped_facet_hits.push((group_key, None));
            return Ok(());
        }
        let mut ords = Vec::new();
        if let Some(FacetValues::Multi(f)) = self.facet.as_mut() {
            if doc > f.doc_id() {
                f.advance(doc)?;
            }
            if doc == f.doc_id() {
                for _ in 0..f.doc_value_count() {
                    ords.push(as_int(f.next_ord()?));
                }
            }
        }
        if ords.is_empty() {
            // The facet ordinal reserved for documents without the field.
            return self.process(group_ord, self.facet_num_terms);
        }
        for ord in ords {
            self.process(group_ord, ord)?;
        }
        Ok(())
    }

    fn set_next_reader_sv(&mut self, leaf: &OpenSegment<'a>) -> Result<()> {
        let reader = leaf
            .reader
            .ok_or_else(|| Error::MissingSegmentReader("TermGroupFacetCollector".into()))?;
        let mut group_index = dv::get_sorted(reader, &self.group_field)?;
        let mut facet = dv::get_sorted(reader, &self.facet_field)?;
        let value_count = facet.value_count();
        self.segment_facet_counts =
            vec![0; usize::try_from(value_count).unwrap_or(0).saturating_add(1)];
        self.segment_total_count = 0;
        self.segment_grouped_facet_hits.clear();
        // Each distinct term is looked up once per segment (Java looks up
        // both terms of every pair; the ordinals are the same).
        let (mut groups, mut facets) = (TermOrds::default(), TermOrds::default());
        for (group_value, facet_value) in &self.grouped_facet_hits {
            let facet_ord = match facet_value {
                None => -1,
                Some(v) => as_int(facets.get(v, |t| Ok(i64::from(facet.lookup_term(t)?)))?),
            };
            if facet_value.is_some() && facet_ord < 0 {
                continue;
            }
            let group_ord = match group_value {
                None => -1,
                Some(v) => as_int(groups.get(v, |t| Ok(i64::from(group_index.lookup_term(t)?)))?),
            };
            if group_value.is_some() && group_ord < 0 {
                continue;
            }
            self.segment_grouped_facet_hits.insert(Self::pair_index(
                group_ord,
                value_count,
                facet_ord,
            ));
        }
        match &self.facet_prefix {
            Some(prefix) => {
                let start = facet.lookup_term(prefix)?;
                // SENTINEL: a negative ordinal is `-insertionPoint - 1`.
                self.start_facet_ord = if start < 0 {
                    start.wrapping_neg().wrapping_sub(1)
                } else {
                    start
                };
                let end = facet.lookup_term(&end_prefix(prefix))?;
                // SENTINEL: `BIG_TERM` is in no dictionary, so negative.
                self.end_facet_ord = if end < 0 {
                    end.wrapping_neg().wrapping_sub(1)
                } else {
                    end
                };
            }
            None => {
                self.start_facet_ord = -1;
                self.end_facet_ord = value_count;
            }
        }
        self.group_index = Some(group_index);
        self.facet = Some(FacetValues::Single(facet));
        Ok(())
    }

    fn set_next_reader_mv(&mut self, leaf: &OpenSegment<'a>) -> Result<()> {
        let reader = leaf
            .reader
            .ok_or_else(|| Error::MissingSegmentReader("TermGroupFacetCollector".into()))?;
        let mut group_index = dv::get_sorted(reader, &self.group_field)?;
        let mut facet = dv::get_sorted_set(reader, &self.facet_field)?;
        self.facet_num_terms = as_int(facet.value_count());
        self.segment_facet_counts = vec![
            0;
            usize::try_from(self.facet_num_terms)
                .unwrap_or(0)
                .saturating_add(1)
        ];
        self.segment_total_count = 0;
        self.segment_grouped_facet_hits.clear();
        // Each distinct term is looked up once per segment, as above.
        let (mut groups, mut facets) = (TermOrds::default(), TermOrds::default());
        for (group_value, facet_value) in &self.grouped_facet_hits {
            let group_ord = match group_value {
                None => -1,
                Some(v) => as_int(groups.get(v, |t| Ok(i64::from(group_index.lookup_term(t)?)))?),
            };
            if group_value.is_some() && group_ord < 0 {
                continue;
            }
            let facet_ord = match facet_value {
                Some(v) => {
                    if self.facet_num_terms == 0 {
                        continue;
                    }
                    let o = facets.get(v, |t| facet.lookup_term(t))?;
                    // SENTINEL: a negative ordinal is absent.
                    if o < 0 {
                        continue;
                    }
                    as_int(o)
                }
                None => self.facet_num_terms,
            };
            self.segment_grouped_facet_hits.insert(Self::pair_index(
                group_ord,
                self.facet_num_terms,
                facet_ord,
            ));
        }
        self.group_index = Some(group_index);
        match &self.facet_prefix {
            Some(prefix) => {
                let start = if self.facet_num_terms == 0 {
                    None
                } else {
                    Some(seek_ceil(facet.as_mut(), prefix)?).filter(|&o| o < facet.value_count())
                };
                let Some(start) = start else {
                    self.start_facet_ord = 0;
                    self.end_facet_ord = 0;
                    self.facet = Some(FacetValues::Multi(facet));
                    return Ok(());
                };
                self.start_facet_ord = as_int(start);
                let end = seek_ceil(facet.as_mut(), &end_prefix(prefix))?;
                self.end_facet_ord = if end < facet.value_count() {
                    as_int(end)
                } else {
                    // Don't include null...
                    self.facet_num_terms
                };
            }
            None => {
                self.start_facet_ord = 0;
                self.end_facet_ord = self.facet_num_terms.wrapping_add(1);
            }
        }
        self.facet = Some(FacetValues::Multi(facet));
        Ok(())
    }

    /// `createSegmentResult()`.
    fn create_segment_result(&mut self) -> Result<Option<SegmentResult<'a>>> {
        let Some(values) = self.facet.take() else {
            return Ok(None);
        };
        let counts = std::mem::take(&mut self.segment_facet_counts);
        let total = self.segment_total_count;
        let r = if self.multivalued {
            let missing_index = self.facet_num_terms;
            let missing = usize::try_from(missing_index)
                .ok()
                .and_then(|i| counts.get(i))
                .copied()
                .unwrap_or(0);
            let max_term_pos = if self.end_facet_ord == missing_index.wrapping_add(1) {
                missing_index
            } else {
                self.end_facet_ord
            };
            let mut r = SegmentResult {
                counts,
                total: total.wrapping_sub(missing),
                missing,
                max_term_pos,
                merge_pos: self.start_facet_ord,
                merge_term: None,
                values,
                shift: 0,
            };
            // `tenum != null`: a dictionary with terms is read at the
            // start, whether or not anything is left to merge.
            if self.facet_num_terms > 0 && r.merge_pos < self.facet_num_terms {
                r.next_term()?;
            }
            r
        } else {
            let missing = counts.first().copied().unwrap_or(0);
            let start = self.start_facet_ord;
            let mut r = SegmentResult {
                counts,
                total: total.wrapping_sub(missing),
                missing,
                max_term_pos: self.end_facet_ord.wrapping_add(1),
                merge_pos: if start == -1 {
                    1
                } else {
                    start.wrapping_add(1)
                },
                merge_term: None,
                values,
                shift: 1,
            };
            if r.merge_pos < r.max_term_pos {
                r.next_term()?;
            }
            r
        };
        Ok(Some(r))
    }
}

impl<'a> SegmentCollector<'a> for TermGroupFacetCollector<'a> {
    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }

    fn set_next_reader(&mut self, _ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        if self.multivalued {
            self.set_next_reader_mv(leaf)
        } else {
            self.set_next_reader_sv(leaf)
        }
    }

    fn collect(&mut self, doc: i32, _score: f32) -> Result<()> {
        if self.multivalued {
            self.collect_mv(doc)
        } else {
            self.collect_sv(doc)
        }
    }

    /// `finish()`: the segment's result.
    fn finish(&mut self) -> Result<()> {
        if let Some(r) = self.create_segment_result()? {
            self.segment_results.push(r);
        }
        Ok(())
    }
}
