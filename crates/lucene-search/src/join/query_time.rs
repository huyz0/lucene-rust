//! `lucene-join`'s query-time joins (Lucene 10.5.0): `JoinUtil` and what it
//! builds -- the from-side collectors that gather the join values of the
//! documents a query matches, and the to-side queries that match the
//! documents holding them.
//!
//! - [`create_join_query`] (`JoinUtil.createJoinQuery(fromField,
//!   multipleValuesPerDocument, toField, fromQuery, fromSearcher,
//!   scoreMode)`): terms from the from-side's `SORTED`/`SORTED_SET` doc values
//!   ([`TermsCollector`], [`TermsWithScoreCollector`]), matched against the
//!   to-field's **terms** -- [`terms_query`] (`TermsQuery`, a multi-term query
//!   over [`SeekingTermSet`]) without scores, [`TermsIncludingScoreQuery`]
//!   with them.
//! - [`create_numeric_join_query`] (the `Class<? extends Number>` overload):
//!   longs from `NUMERIC`/`SORTED_NUMERIC` doc values, matched against the
//!   to-field's one-dimensional **points** -- a [`PointInSetQuery`] without
//!   scores, a [`PointInSetIncludingScoreQuery`] with them.
//! - [`create_global_ordinals_join_query`] (the `OrdinalMap` overloads, with
//!   `min`/`max`): one `SORTED` join field on both sides, joined by global
//!   ordinal ([`GlobalOrdinalsCollector`], [`GlobalOrdinalsWithScoreCollector`],
//!   [`GlobalOrdinalsQuery`], [`GlobalOrdinalsWithScoreQuery`]).
//!
//! The from side is collected by [`crate::leaf_collector`] as Java's
//! `fromSearcher.search(fromQuery, collector)` collects it: segment by
//! segment, each segment's doc values opened when the search enters it
//! (`DocValues.getSorted` and friends, [`crate::reader::doc_values`]), every
//! float accumulated in Java's order and precision. How the to-side queries
//! run per segment is `exec::query_join`.
//!
//! # What differs from Java
//!
//! - `indexReaderContextId`: Java's to-side queries remember the reader they
//!   were built against, compare equal only for the same one, and the global
//!   ordinal ones refuse a weight over another
//!   (`IllegalStateException`). This port has no reader identity; equality
//!   compares the collected join data instead (the same `JoinUtil` call, or
//!   data equal to it), and running a global-ordinals query over another
//!   reader is the caller's mistake to avoid.
//! - `BytesRefHash`, `LongBitSet`, the blocked `Scores`/`Occurrences`
//!   arrays and HPPC's hash maps are their plain Rust counterparts
//!   ([`BytesRefHash`] itself for the terms, so ids and their order are
//!   Java's).
//! - `explain` is ported for [`TermsIncludingScoreQuery`] and
//!   [`PointInSetIncludingScoreQuery`]; the global-ordinals queries read doc
//!   values, which the explain path (no segment reader) cannot, and report
//!   [`Error::MissingSegmentReader`].

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use lucene_util::bytes_ref_hash::BytesRefHash;
use lucene_util::fixed_bit_set::FixedBitSet;

use super::ScoreMode;
use crate::collector::ScoreMode as CollectorScoreMode;
use crate::extended_query::{
    ExtendedQuery, MultiTermQuery, MultiTermSource, PointInSetQuery, RewriteMethod,
};
use crate::index_searcher::IndexSearcher;
use crate::leaf_collector::{search_segments, SegmentCollector};
use crate::multi_segment::OpenSegment;
use crate::ordinal_map::OrdinalMap;
use crate::query::{BooleanQuery, Clause, MatchNoDocsQuery};
use crate::reader::doc_values as dv;
use crate::reader::filtered_terms_enum::{AcceptStatus, FilteredTermsEnum, TermFilter};
use crate::reader::{
    NumericDocValues, SortedDocValues, SortedNumericDocValues, SortedSetDocValues, TermsEnum,
};
use crate::{Error, Result};

// ---------------------------------------------------------------------------
// Java's float arithmetic
// ---------------------------------------------------------------------------

/// `Math.min(float, float)`: `NaN` if either is, `-0.0` below `0.0`.
pub(crate) fn java_min_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else if a <= b {
        a
    } else {
        b
    }
}

/// `Math.max(float, float)`: `NaN` if either is, `0.0` above `-0.0`.
pub(crate) fn java_max_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_positive() {
            a
        } else {
            b
        }
    } else if a >= b {
        a
    } else {
        b
    }
}

/// `Float.compare(a, 0.0f) == 0`: `a` is `+0.0` exactly (`-0.0` is not).
fn is_positive_zero(a: f32) -> bool {
    a.to_bits() == 0
}

/// A leaf's reader, which every collector here reads doc values through.
fn leaf_reader<'a>(leaf: &OpenSegment<'a>) -> Result<&'a crate::directory_reader::SegmentReader> {
    leaf.reader
        .ok_or_else(|| Error::MissingSegmentReader("a query-time join collector".into()))
}

fn hash_err(e: lucene_util::bytes_ref_hash::BytesRefHashError) -> Error {
    Error::IllegalArgument(e.to_string())
}

/// `BytesRefHash.add`'s id: the new one, or the existing `-(id + 1)` turned
/// back into `id`; and whether it was new.
fn add_term(hash: &mut BytesRefHash, term: &[u8]) -> Result<(usize, bool)> {
    let id = hash.add(term).map_err(hash_err)?;
    // SENTINEL-OK: `BytesRefHash.add`'s `-(id + 1)` for a present term,
    // decoded here and nowhere else.
    if id < 0 {
        let id = id.checked_neg().and_then(|i| i.checked_sub(1)).unwrap_or(0);
        Ok((usize::try_from(id).unwrap_or(0), false))
    } else {
        Ok((usize::try_from(id).unwrap_or(0), true))
    }
}

/// The collected terms in id order and the ids in term order
/// (`BytesRefHash.get` and `sort()`).
fn sorted_terms(mut hash: BytesRefHash) -> (Vec<Vec<u8>>, Vec<u32>) {
    let terms: Vec<Vec<u8>> = (0..hash.size())
        .map(|i| hash.get(i).unwrap_or_default().to_vec())
        .collect();
    let ords: Vec<u32> = hash
        .sort()
        .into_iter()
        .take(terms.len())
        .filter_map(|o| u32::try_from(o).ok())
        .collect();
    (terms, ords)
}

// ---------------------------------------------------------------------------
// The from side: terms
// ---------------------------------------------------------------------------

/// `TermsCollector` (`SV`, `MV`): every join term of the matching documents
/// (`COMPLETE_NO_SCORES`). A single-valued document without a value
/// contributes the empty term, as Java's `SV` adds `BytesRef.EMPTY_BYTES`.
pub struct TermsCollector<'a> {
    field: String,
    multiple_values_per_document: bool,
    sv: Option<Box<dyn SortedDocValues + 'a>>,
    mv: Option<Box<dyn SortedSetDocValues + 'a>>,
    terms: BytesRefHash,
}

impl<'a> TermsCollector<'a> {
    /// `TermsCollector.create(field, multipleValuesPerDocument)`.
    pub fn new(field: &str, multiple_values_per_document: bool) -> Self {
        Self {
            field: field.to_string(),
            multiple_values_per_document,
            sv: None,
            mv: None,
            terms: BytesRefHash::new(),
        }
    }

    /// `getCollectorTerms()`: the terms in id order and the ids in term order.
    pub fn into_terms(self) -> (Vec<Vec<u8>>, Vec<u32>) {
        sorted_terms(self.terms)
    }
}

impl<'a> SegmentCollector<'a> for TermsCollector<'a> {
    fn score_mode(&self) -> CollectorScoreMode {
        CollectorScoreMode::CompleteNoScores
    }

    fn set_next_reader(&mut self, _ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        let reader = leaf_reader(leaf)?;
        if self.multiple_values_per_document {
            self.mv = Some(dv::get_sorted_set(reader, &self.field)?);
        } else {
            self.sv = Some(dv::get_sorted(reader, &self.field)?);
        }
        Ok(())
    }

    fn collect(&mut self, doc: i32, _score: f32) -> Result<()> {
        if let Some(values) = self.mv.as_mut() {
            // `MV.collect`: `advance` when behind, then every ordinal.
            if doc > values.doc_id() {
                values.advance(doc)?;
            }
            if doc == values.doc_id() {
                for _ in 0..values.doc_value_count() {
                    let ord = values.next_ord()?;
                    let term = values.lookup_ord(ord)?;
                    add_term(&mut self.terms, &term)?;
                }
            }
        } else if let Some(values) = self.sv.as_mut() {
            let term = if values.advance_exact(doc)? {
                values.lookup_ord(values.ord_value())?
            } else {
                Vec::new()
            };
            add_term(&mut self.terms, &term)?;
        }
        Ok(())
    }
}

/// `TermsWithScoreCollector` (`SV`, `SV.Avg`, `MV`, `MV.Avg`): every join
/// term of the matching documents with the scores of the documents holding
/// it combined by `score_mode` (`COMPLETE`), in `float`, as Java combines
/// them:
///
/// - single-valued: the first score a term gets is taken as is; a later one
///   is added (`Total`, `Avg`) or kept if lower (`Min`) or higher (`Max`) --
///   except that a combined score of exactly `+0.0` is overwritten by the next
///   one, as Java's `Float.compare(existing, 0.0f) == 0` test does;
/// - multi-valued: every score is folded in (`+=`, `Math.min`, `Math.max`
///   from `0`, `+Inf`, `-Inf`).
///
/// `Avg` divides the sum by the count when the scores are taken.
pub struct TermsWithScoreCollector<'a> {
    field: String,
    multiple_values_per_document: bool,
    score_mode: ScoreMode,
    sv: Option<Box<dyn SortedDocValues + 'a>>,
    mv: Option<Box<dyn SortedSetDocValues + 'a>>,
    terms: BytesRefHash,
    score_sums: Vec<f32>,
    score_counts: Vec<i32>,
}

impl<'a> TermsWithScoreCollector<'a> {
    /// `TermsWithScoreCollector.create(field, multipleValuesPerDocument,
    /// scoreMode)`.
    pub fn new(field: &str, multiple_values_per_document: bool, score_mode: ScoreMode) -> Self {
        Self {
            field: field.to_string(),
            multiple_values_per_document,
            score_mode,
            sv: None,
            mv: None,
            terms: BytesRefHash::new(),
            score_sums: Vec::new(),
            score_counts: Vec::new(),
        }
    }

    /// The value a new term's slot starts from (`ArrayUtil.grow` then the
    /// `Arrays.fill` of `Min`/`Max`; `Avg`'s arrays are not filled).
    fn unset(&self) -> f32 {
        match self.score_mode {
            ScoreMode::Min => f32::INFINITY,
            ScoreMode::Max => f32::NEG_INFINITY,
            _ => 0.0,
        }
    }

    /// A term's slot, created when the term is new.
    fn slot(&mut self, term: &[u8]) -> Result<usize> {
        let (id, _) = add_term(&mut self.terms, term)?;
        while self.score_sums.len() <= id {
            let unset = self.unset();
            self.score_sums.push(unset);
            self.score_counts.push(0);
        }
        Ok(id)
    }

    fn collect_sv(&mut self, id: usize, current: f32) {
        let existing = self.score_sums[id];
        if self.score_mode == ScoreMode::Avg {
            // `SV.Avg.collect`.
            if is_positive_zero(existing) {
                self.score_sums[id] = current;
                self.score_counts[id] = 1;
            } else {
                self.score_sums[id] += current;
                self.score_counts[id] = self.score_counts[id].wrapping_add(1);
            }
            return;
        }
        if is_positive_zero(existing) {
            self.score_sums[id] = current;
        } else {
            match self.score_mode {
                ScoreMode::Total => self.score_sums[id] += current,
                ScoreMode::Min => {
                    if current < existing {
                        self.score_sums[id] = current;
                    }
                }
                ScoreMode::Max => {
                    if current > existing {
                        self.score_sums[id] = current;
                    }
                }
                // `AssertionError("unexpected: " + scoreMode)` in Java; the
                // factory never builds one.
                ScoreMode::None | ScoreMode::Avg => {}
            }
        }
    }

    fn collect_mv(&mut self, id: usize, score: f32) {
        match self.score_mode {
            ScoreMode::Total => self.score_sums[id] += score,
            ScoreMode::Min => self.score_sums[id] = java_min_f32(self.score_sums[id], score),
            ScoreMode::Max => self.score_sums[id] = java_max_f32(self.score_sums[id], score),
            ScoreMode::Avg => {
                self.score_sums[id] += score;
                self.score_counts[id] = self.score_counts[id].wrapping_add(1);
            }
            ScoreMode::None => {}
        }
    }

    /// `getCollectedTerms()` and `getScoresPerTerm()`: the terms in id
    /// order, the ids in term order, and each term's score by id.
    pub fn into_terms_and_scores(self) -> (Vec<Vec<u8>>, Vec<u32>, Vec<f32>) {
        let mut scores = self.score_sums;
        if self.score_mode == ScoreMode::Avg {
            for (s, &c) in scores.iter_mut().zip(&self.score_counts) {
                *s /= c as f32;
            }
        }
        let (terms, ords) = sorted_terms(self.terms);
        scores.truncate(terms.len());
        (terms, ords, scores)
    }
}

impl<'a> SegmentCollector<'a> for TermsWithScoreCollector<'a> {
    fn score_mode(&self) -> CollectorScoreMode {
        CollectorScoreMode::Complete
    }

    fn set_next_reader(&mut self, _ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        let reader = leaf_reader(leaf)?;
        if self.multiple_values_per_document {
            self.mv = Some(dv::get_sorted_set(reader, &self.field)?);
        } else {
            self.sv = Some(dv::get_sorted(reader, &self.field)?);
        }
        Ok(())
    }

    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        if let Some(mut values) = self.mv.take() {
            let r = (|| -> Result<()> {
                if values.advance_exact(doc)? {
                    for _ in 0..values.doc_value_count() {
                        let ord = values.next_ord()?;
                        let term = values.lookup_ord(ord)?;
                        let id = self.slot(&term)?;
                        self.collect_mv(id, score);
                    }
                }
                Ok(())
            })();
            self.mv = Some(values);
            return r;
        }
        let Some(values) = self.sv.as_mut() else {
            return Ok(());
        };
        let term = if values.advance_exact(doc)? {
            values.lookup_ord(values.ord_value())?
        } else {
            Vec::new()
        };
        let id = self.slot(&term)?;
        self.collect_sv(id, score);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The to side: terms
// ---------------------------------------------------------------------------

/// The term set a `TermsQuery` enumerates ([`MultiTermSource::TermSet`]):
/// the join terms, distinct and ascending, with the from side it came from
/// (part of `TermsQuery.equals`, and of its `toString`).
#[derive(Clone)]
pub struct TermSetSource {
    pub field: String,
    /// Distinct, ascending (unsigned bytes).
    pub terms: Arc<[Vec<u8>]>,
    pub from_field: String,
    pub from_query: Arc<Clause>,
}

impl TermSetSource {
    /// Whether `term` is one of the set.
    pub fn contains(&self, term: &[u8]) -> bool {
        self.terms
            .binary_search_by(|t| t.as_slice().cmp(term))
            .is_ok()
    }
}

impl fmt::Debug for TermSetSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `TermsQuery.toString`, and the set by identity: printing the terms
        // would make every key this query is filed under as long as the set.
        write!(
            f,
            "TermsQuery{{field={}fromQuery={:?}}}@{:p}/{}",
            self.field,
            self.from_query,
            Arc::as_ptr(&self.terms),
            self.terms.len()
        )
    }
}

impl PartialEq for TermSetSource {
    fn eq(&self, o: &Self) -> bool {
        self.field == o.field
            && self.from_field == o.from_field
            && self.from_query == o.from_query
            && (Arc::ptr_eq(&self.terms, &o.terms) || self.terms == o.terms)
    }
}

/// `TermsQuery`: the documents whose `field` has any of the set's terms,
/// constant-scored -- a `MultiTermQuery` under
/// `CONSTANT_SCORE_BLENDED_REWRITE` whose terms enum is
/// [`SeekingTermSet`] (`TermsEnum.EMPTY` for an empty set).
pub fn terms_query(source: TermSetSource) -> Clause {
    Clause::from(MultiTermQuery::new(
        MultiTermSource::TermSet(source),
        RewriteMethod::ConstantScoreBlended,
    ))
}

/// `SeekingTermSetTermsEnum`'s filter: walks a sorted term set alongside a
/// terms enum, seeking to each set term in turn and accepting exactly the
/// ones the enum has.
#[derive(Debug, Clone)]
pub struct SeekingTermSet {
    terms: Arc<[Vec<u8>]>,
    /// `upto`: the set term being looked for.
    upto: usize,
    /// `seekTerm`: where the enum seeks next (`None`: nowhere).
    seek: Option<usize>,
}

impl SeekingTermSet {
    /// `new SeekingTermSetTermsEnum(tenum, terms, ords)`'s state: the first
    /// seek is to the smallest term. `terms` must be non-empty (Java's
    /// `TermsQuery` hands over `TermsEnum.EMPTY` instead).
    pub fn new(terms: Arc<[Vec<u8>]>) -> Self {
        Self {
            terms,
            upto: 0,
            seek: Some(0),
        }
    }

    fn last(&self) -> usize {
        self.terms.len().saturating_sub(1)
    }

    /// `nextSeekTerm(currentTerm)`: the pending seek, once.
    pub fn next_seek(&mut self) -> Option<&[u8]> {
        let at = self.seek.take()?;
        self.terms.get(at).map(Vec::as_slice)
    }

    /// `accept(term)`.
    pub fn accept_term(&mut self, term: &[u8]) -> AcceptStatus {
        let Some(last_term) = self.terms.get(self.last()) else {
            return AcceptStatus::End;
        };
        if term > last_term.as_slice() {
            return AcceptStatus::End;
        }
        let last = self.last();
        if term == self.terms[self.upto].as_slice() {
            if self.upto == last {
                return AcceptStatus::Yes;
            }
            self.upto = self.upto.saturating_add(1);
            self.seek = Some(self.upto);
            return AcceptStatus::YesAndSeek;
        }
        if self.upto == last {
            return AcceptStatus::No;
        }
        // Behind the enum's term by one step or more: catch up.
        let cmp = loop {
            if self.upto == last {
                return AcceptStatus::No;
            }
            self.upto = self.upto.saturating_add(1);
            self.seek = Some(self.upto);
            let cmp = self.terms[self.upto].as_slice().cmp(term);
            if cmp != std::cmp::Ordering::Less {
                break cmp;
            }
        };
        if cmp == std::cmp::Ordering::Equal {
            if self.upto == last {
                // `return YES` with `seekTerm` still set: the next `next()`
                // reads on, and the `END` test above stops it.
                return AcceptStatus::Yes;
            }
            self.upto = self.upto.saturating_add(1);
            self.seek = Some(self.upto);
            AcceptStatus::YesAndSeek
        } else {
            AcceptStatus::NoAndSeek
        }
    }
}

impl TermFilter for SeekingTermSet {
    fn accept(&mut self, term: &[u8]) -> Result<AcceptStatus> {
        Ok(self.accept_term(term))
    }

    fn next_seek_term(
        &mut self,
        _current: Option<&[u8]>,
        _initial: Option<Vec<u8>>,
    ) -> Result<Option<Vec<u8>>> {
        Ok(self.next_seek().map(<[u8]>::to_vec))
    }
}

/// `SeekingTermSetTermsEnum`: the terms of `tenum` that are in `terms`
/// (non-empty, distinct, ascending), found by seeking.
pub fn seeking_term_set_terms_enum<'a>(
    tenum: Box<dyn TermsEnum + 'a>,
    terms: Arc<[Vec<u8>]>,
) -> FilteredTermsEnum<'a, SeekingTermSet> {
    FilteredTermsEnum::new(tenum, SeekingTermSet::new(terms), true)
}

/// `TermsIncludingScoreQuery`: the documents whose `to_field` has a join
/// term, each scored its term's from-side score times the boost. With
/// several of its terms a document takes the score of the last one in term
/// order (`SVInOrderScorer`) or of the first (`MVInOrderScorer`,
/// `multiple_values_per_document`). Without scores it is the
/// [`terms_query`] of the same terms.
#[derive(Clone)]
pub struct TermsIncludingScoreQuery {
    pub score_mode: ScoreMode,
    pub to_field: String,
    pub multiple_values_per_document: bool,
    /// The join terms in term order (`terms.get(ords[i])`).
    pub(crate) terms: Arc<[Vec<u8>]>,
    /// Each term's score, aligned with `terms` (`scores[ords[i]]`).
    pub(crate) scores: Arc<[f32]>,
    pub from_field: String,
    pub from_query: Arc<Clause>,
}

impl TermsIncludingScoreQuery {
    /// The terms in term order, each with its score.
    pub fn terms_and_scores(&self) -> impl Iterator<Item = (&[u8], f32)> + '_ {
        self.terms
            .iter()
            .map(Vec::as_slice)
            .zip(self.scores.iter().copied())
    }

    /// The `TermsQuery` the query runs as when scores are not needed.
    pub(crate) fn as_terms_query(&self) -> Clause {
        terms_query(TermSetSource {
            field: self.to_field.clone(),
            terms: Arc::clone(&self.terms),
            from_field: self.from_field.clone(),
            from_query: Arc::clone(&self.from_query),
        })
    }
}

impl fmt::Debug for TermsIncludingScoreQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TermsIncludingScoreQuery{{field={};fromQuery={:?}}}@{:p}",
            self.to_field,
            self.from_query,
            Arc::as_ptr(&self.terms)
        )
    }
}

impl PartialEq for TermsIncludingScoreQuery {
    fn eq(&self, o: &Self) -> bool {
        self.score_mode == o.score_mode
            && self.to_field == o.to_field
            && self.from_field == o.from_field
            && self.from_query == o.from_query
            && self.multiple_values_per_document == o.multiple_values_per_document
            && (Arc::ptr_eq(&self.terms, &o.terms)
                || (self.terms == o.terms
                    && self
                        .scores
                        .iter()
                        .map(|s| s.to_bits())
                        .eq(o.scores.iter().map(|s| s.to_bits()))))
    }
}

/// `JoinUtil.createJoinQuery(fromField, multipleValuesPerDocument, toField,
/// fromQuery, fromSearcher, scoreMode)`: the join terms of the documents
/// `from_query` matches in `from_searcher` (read from `from_field`'s
/// `SORTED`, or with `multiple_values_per_document` `SORTED_SET`, doc
/// values), as a query over `to_field`'s terms.
///
/// # Errors
/// What searching the from side reports, and `DocValues`' error for a
/// from-field with doc values of another kind.
pub fn create_join_query(
    from_field: &str,
    multiple_values_per_document: bool,
    to_field: &str,
    from_query: &Clause,
    from_searcher: &IndexSearcher<'_, '_>,
    score_mode: ScoreMode,
) -> Result<Clause> {
    let query = BooleanQuery {
        must: vec![from_query.clone()],
        ..Default::default()
    };
    let from = Arc::new(from_query.clone());
    if score_mode == ScoreMode::None {
        let mut c = TermsCollector::new(from_field, multiple_values_per_document);
        search_segments(from_searcher, &query, &mut c)?;
        let (terms, ords) = c.into_terms();
        let sorted: Arc<[Vec<u8>]> = ords
            .iter()
            .filter_map(|&o| terms.get(usize::try_from(o).ok()?).cloned())
            .collect();
        return Ok(terms_query(TermSetSource {
            field: to_field.to_string(),
            terms: sorted,
            from_field: from_field.to_string(),
            from_query: from,
        }));
    }
    let mut c = TermsWithScoreCollector::new(from_field, multiple_values_per_document, score_mode);
    search_segments(from_searcher, &query, &mut c)?;
    let (terms, ords, scores) = c.into_terms_and_scores();
    let (sorted, sorted_scores): (Vec<Vec<u8>>, Vec<f32>) = ords
        .iter()
        .filter_map(|&o| {
            let o = usize::try_from(o).ok()?;
            Some((terms.get(o)?.clone(), *scores.get(o)?))
        })
        .unzip();
    Ok(Clause::Extended(Box::new(
        ExtendedQuery::TermsIncludingScore(TermsIncludingScoreQuery {
            score_mode,
            to_field: to_field.to_string(),
            multiple_values_per_document,
            terms: sorted.into(),
            scores: sorted_scores.into(),
            from_field: from_field.to_string(),
            from_query: from,
        }),
    )))
}

// ---------------------------------------------------------------------------
// Numeric joins
// ---------------------------------------------------------------------------

/// The `Class<? extends Number>` of the numeric `createJoinQuery`: how a
/// join value (a doc-values long) is encoded as the to-field's point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumericType {
    /// `Integer`: `IntPoint`, the value cast to `int`.
    Int,
    /// `Long`: `LongPoint`.
    Long,
    /// `Float`: `FloatPoint`, the value's low 32 bits as the float's bits.
    Float,
    /// `Double`: `DoublePoint`, the value as the double's bits.
    Double,
}

impl NumericType {
    /// `Integer.BYTES` / `Long.BYTES` / ...
    pub fn bytes(self) -> usize {
        match self {
            NumericType::Int | NumericType::Float => 4,
            NumericType::Long | NumericType::Double => 8,
        }
    }

    /// The `encodeDimension` of the matching point class for join value
    /// `v`: sortable big-endian bytes, `floatToIntBits`/`doubleToLongBits`
    /// (one `NaN`) for the floating types.
    pub fn encode(self, v: i64) -> Vec<u8> {
        match self {
            NumericType::Int => sortable_int(v as i32),
            NumericType::Long => sortable_long(v),
            NumericType::Float => {
                let f = f32::from_bits(v as i32 as u32);
                let bits = if f.is_nan() {
                    0x7fc0_0000
                } else {
                    f.to_bits() as i32
                };
                sortable_int(bits ^ ((bits >> 31) & 0x7fff_ffff))
            }
            NumericType::Double => {
                let d = f64::from_bits(v as u64);
                let bits = if d.is_nan() {
                    0x7ff8_0000_0000_0000
                } else {
                    d.to_bits() as i64
                };
                sortable_long(bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff))
            }
        }
    }

    /// `PointInSetIncludingScoreQuery.toString.apply(value, numericType)`.
    fn render(self, p: &[u8]) -> String {
        match (self, p.len()) {
            (NumericType::Int, 4) | (NumericType::Float, 4) => {
                let mut a = [0u8; 4];
                a.copy_from_slice(p);
                let i = (u32::from_be_bytes(a) ^ 0x8000_0000) as i32;
                if self == NumericType::Int {
                    i.to_string()
                } else {
                    let bits = i ^ ((i >> 31) & 0x7fff_ffff);
                    format!("{:?}", f32::from_bits(bits as u32))
                }
            }
            (NumericType::Long, 8) | (NumericType::Double, 8) => {
                let mut a = [0u8; 8];
                a.copy_from_slice(p);
                let l = (u64::from_be_bytes(a) ^ (1 << 63)) as i64;
                if self == NumericType::Long {
                    l.to_string()
                } else {
                    let bits = l ^ ((l >> 63) & 0x7fff_ffff_ffff_ffff);
                    format!("{:?}", f64::from_bits(bits as u64))
                }
            }
            _ => "unsupported".to_string(),
        }
    }
}

fn sortable_int(i: i32) -> Vec<u8> {
    ((i as u32) ^ 0x8000_0000).to_be_bytes().to_vec()
}

fn sortable_long(l: i64) -> Vec<u8> {
    ((l as u64) ^ (1 << 63)).to_be_bytes().to_vec()
}

/// `BytesRef.toString()`: `[` hex bytes, space separated `]`.
fn bytes_ref_string(b: &[u8]) -> String {
    let hex: Vec<String> = b.iter().map(|x| format!("{x:x}")).collect();
    format!("[{}]", hex.join(" "))
}

/// The numeric `JoinUtil` collector: every join value of the matching
/// documents (a single-valued document without one contributes `0`, as
/// Java's does), with the scores aggregated per value as `LongFloatHashMap`
/// aggregates them (`Max`/`Min` through `Math.max`/`Math.min` from the
/// first score, `Total`/`Avg` summed from `0`, `Avg` also counted).
pub struct NumericJoinCollector<'a> {
    field: String,
    multiple_values_per_document: bool,
    score_mode: ScoreMode,
    sv: Option<Box<dyn NumericDocValues + 'a>>,
    mv: Option<Box<dyn SortedNumericDocValues + 'a>>,
    values: std::collections::HashSet<i64>,
    scores: HashMap<i64, f32>,
    occurrences: HashMap<i64, i32>,
}

impl<'a> NumericJoinCollector<'a> {
    pub fn new(field: &str, multiple_values_per_document: bool, score_mode: ScoreMode) -> Self {
        Self {
            field: field.to_string(),
            multiple_values_per_document,
            score_mode,
            sv: None,
            mv: None,
            values: Default::default(),
            scores: HashMap::new(),
            occurrences: HashMap::new(),
        }
    }

    fn add(&mut self, value: i64, score: f32) {
        self.values.insert(value);
        match self.score_mode {
            ScoreMode::None => {}
            ScoreMode::Max => {
                let e = self.scores.entry(value).or_insert(score);
                *e = java_max_f32(*e, score);
            }
            ScoreMode::Min => {
                let e = self.scores.entry(value).or_insert(score);
                *e = java_min_f32(*e, score);
            }
            ScoreMode::Total => *self.scores.entry(value).or_insert(0.0) += score,
            ScoreMode::Avg => {
                *self.scores.entry(value).or_insert(0.0) += score;
                let o = self.occurrences.entry(value).or_insert(0);
                *o = o.wrapping_add(1);
            }
        }
    }

    /// The join values, ascending, each with its score (`joinScorer`).
    pub fn into_sorted(self) -> Vec<(i64, f32)> {
        let mut values: Vec<i64> = self.values.into_iter().collect();
        values.sort_unstable();
        values
            .into_iter()
            .map(|v| {
                let s = self.scores.get(&v).copied().unwrap_or(0.0);
                let s = if self.score_mode == ScoreMode::Avg {
                    s / self.occurrences.get(&v).copied().unwrap_or(0) as f32
                } else {
                    s
                };
                (v, s)
            })
            .collect()
    }
}

impl<'a> SegmentCollector<'a> for NumericJoinCollector<'a> {
    fn score_mode(&self) -> CollectorScoreMode {
        if self.score_mode == ScoreMode::None {
            CollectorScoreMode::CompleteNoScores
        } else {
            CollectorScoreMode::Complete
        }
    }

    fn set_next_reader(&mut self, _ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        let reader = leaf_reader(leaf)?;
        if self.multiple_values_per_document {
            self.mv = Some(dv::get_sorted_numeric(reader, &self.field)?);
        } else {
            self.sv = Some(dv::get_numeric(reader, &self.field)?);
        }
        Ok(())
    }

    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        if let Some(mut values) = self.mv.take() {
            let r = (|| -> Result<()> {
                if values.advance_exact(doc)? {
                    for _ in 0..values.doc_value_count() {
                        let v = values.next_value()?;
                        self.add(v, score);
                    }
                }
                Ok(())
            })();
            self.mv = Some(values);
            return r;
        }
        let Some(values) = self.sv.as_mut() else {
            return Ok(());
        };
        let value = if values.advance_exact(doc)? {
            values.long_value()
        } else {
            0
        };
        self.add(value, score);
        Ok(())
    }
}

/// `PointInSetIncludingScoreQuery`: the documents with a one-dimensional
/// point on `field` equal to a join value, each scored its value's
/// from-side score (no boost, as Java's scorer). A document with several
/// matching points takes the score of the last one visited, or of the first
/// with `multiple_values_per_document`.
#[derive(Clone)]
pub struct PointInSetIncludingScoreQuery {
    pub score_mode: ScoreMode,
    pub original_query: Arc<Clause>,
    pub multiple_values_per_document: bool,
    pub field: String,
    pub bytes_per_dim: usize,
    pub numeric_type: NumericType,
    /// Packed points, strictly ascending.
    pub(crate) points: Arc<[Vec<u8>]>,
    /// Each point's score.
    pub(crate) scores: Arc<[f32]>,
}

impl PointInSetIncludingScoreQuery {
    /// The constructor's checks over `(packed point, score)`s in stream
    /// order: each `bytes_per_dim` long, strictly ascending.
    ///
    /// # Errors
    /// Java's `IllegalArgumentException`s, as [`Error::IllegalArgument`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        score_mode: ScoreMode,
        original_query: Arc<Clause>,
        multiple_values_per_document: bool,
        field: &str,
        bytes_per_dim: usize,
        numeric_type: NumericType,
        stream: Vec<(Vec<u8>, f32)>,
    ) -> Result<Self> {
        if !(1..=16).contains(&bytes_per_dim) {
            return Err(Error::IllegalArgument(format!(
                "bytesPerDim must be > 0 and <= 16; got {bytes_per_dim}"
            )));
        }
        let mut points: Vec<Vec<u8>> = Vec::with_capacity(stream.len());
        let mut scores = Vec::with_capacity(stream.len());
        for (p, s) in stream {
            if p.len() != bytes_per_dim {
                return Err(Error::IllegalArgument(format!(
                    "packed point length should be {bytes_per_dim} but got {}; field=\"{field}\"\
                     bytesPerDim={bytes_per_dim}",
                    p.len()
                )));
            }
            if let Some(prev) = points.last() {
                match prev.as_slice().cmp(p.as_slice()) {
                    std::cmp::Ordering::Equal => {
                        return Err(Error::IllegalArgument(format!(
                            "unexpected duplicated value: {}",
                            bytes_ref_string(&p)
                        )))
                    }
                    std::cmp::Ordering::Greater => {
                        return Err(Error::IllegalArgument(format!(
                            "values are out of order: saw {} before {}",
                            bytes_ref_string(prev),
                            bytes_ref_string(&p)
                        )))
                    }
                    std::cmp::Ordering::Less => {}
                }
            }
            points.push(p);
            scores.push(s);
        }
        Ok(Self {
            score_mode,
            original_query,
            multiple_values_per_document,
            field: field.to_string(),
            bytes_per_dim,
            numeric_type,
            points: points.into(),
            scores: scores.into(),
        })
    }

    /// The packed points, ascending, each with its score.
    pub fn points_and_scores(&self) -> impl Iterator<Item = (&[u8], f32)> + '_ {
        self.points
            .iter()
            .map(Vec::as_slice)
            .zip(self.scores.iter().copied())
    }
}

impl fmt::Debug for PointInSetIncludingScoreQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `toString(field)` with a foreign field: `field:{v v ...}`.
        let values: Vec<String> = self
            .points
            .iter()
            .map(|p| self.numeric_type.render(p))
            .collect();
        write!(
            f,
            "PointInSetIncludingScoreQuery({}:{{{}}}, {}, {:?})",
            self.field,
            values.join(" "),
            self.score_mode,
            self.original_query
        )
    }
}

impl PartialEq for PointInSetIncludingScoreQuery {
    fn eq(&self, o: &Self) -> bool {
        self.score_mode == o.score_mode
            && self.field == o.field
            && self.original_query == o.original_query
            && self.bytes_per_dim == o.bytes_per_dim
            && self.multiple_values_per_document == o.multiple_values_per_document
            && self.points == o.points
            && self
                .scores
                .iter()
                .map(|s| s.to_bits())
                .eq(o.scores.iter().map(|s| s.to_bits()))
    }
}

/// `JoinUtil.createJoinQuery(fromField, multipleValuesPerDocument, toField,
/// numericType, fromQuery, fromSearcher, scoreMode)`: the join values of
/// the documents `from_query` matches (from `from_field`'s `NUMERIC`, or
/// with `multiple_values_per_document` `SORTED_NUMERIC`, doc values),
/// sorted as longs, encoded as `numeric_type`'s points, as a query over
/// `to_field`'s points.
///
/// # Errors
/// Java's: values that encode out of order or twice (negative floats sort
/// one way as longs and the other as points; distinct longs can cast to the
/// same `int`) -- `IllegalArgumentException`, as [`Error::IllegalArgument`];
/// and what searching the from side reports.
pub fn create_numeric_join_query(
    from_field: &str,
    multiple_values_per_document: bool,
    to_field: &str,
    numeric_type: NumericType,
    from_query: &Clause,
    from_searcher: &IndexSearcher<'_, '_>,
    score_mode: ScoreMode,
) -> Result<Clause> {
    let query = BooleanQuery {
        must: vec![from_query.clone()],
        ..Default::default()
    };
    let mut c = NumericJoinCollector::new(from_field, multiple_values_per_document, score_mode);
    search_segments(from_searcher, &query, &mut c)?;
    let stream: Vec<(Vec<u8>, f32)> = c
        .into_sorted()
        .into_iter()
        .map(|(v, s)| (numeric_type.encode(v), s))
        .collect();
    let bytes = numeric_type.bytes();
    if score_mode != ScoreMode::None {
        return Ok(Clause::Extended(Box::new(
            ExtendedQuery::PointInSetIncludingScore(PointInSetIncludingScoreQuery::new(
                score_mode,
                Arc::new(from_query.clone()),
                multiple_values_per_document,
                to_field,
                bytes,
                numeric_type,
                stream,
            )?),
        )));
    }
    // `new PointInSetQuery(toField, 1, bytesPerDim, stream)`: equal
    // neighbours are dropped, a value below its predecessor is refused.
    let mut points: Vec<Vec<u8>> = Vec::with_capacity(stream.len());
    for (p, _) in stream {
        if let Some(prev) = points.last() {
            match prev.as_slice().cmp(p.as_slice()) {
                std::cmp::Ordering::Equal => continue,
                std::cmp::Ordering::Greater => {
                    return Err(Error::IllegalArgument(format!(
                        "values are out of order: saw {} before {}",
                        bytes_ref_string(prev),
                        bytes_ref_string(&p)
                    )))
                }
                std::cmp::Ordering::Less => {}
            }
        }
        points.push(p);
    }
    Ok(Clause::from(
        PointInSetQuery::new(to_field, 1, bytes, points)
            .map_err(|e| Error::IllegalArgument(e.to_string()))?,
    ))
}

// ---------------------------------------------------------------------------
// Global ordinals
// ---------------------------------------------------------------------------

/// The ordinal map `JoinUtil`'s global-ordinals join takes: `field`'s
/// `SORTED` dictionaries of every segment of `searcher` merged into one
/// global ordinal space (`OrdinalMap.build(null, sortedDocValues,
/// PackedInts.DEFAULT)`, which a Java caller builds itself).
///
/// # Errors
/// A field whose doc values are not `SORTED`/`SORTED_SET`, or a dictionary
/// that cannot be read.
pub fn ordinal_map(searcher: &IndexSearcher<'_, '_>, field: &str) -> Result<Arc<OrdinalMap>> {
    let readers: Vec<&crate::directory_reader::SegmentReader> = searcher
        .segments()
        .iter()
        .map(leaf_reader)
        .collect::<Result<_>>()?;
    Ok(Arc::new(crate::terms_agg::ordinal_map_of(&readers, field)?))
}

/// Segment cores (name and id) to their position among a searcher's
/// segments: the `context.ord` a to-side scorer looks its segment's
/// ordinal mapping up by.
#[derive(Debug, Default)]
pub(crate) struct LeafOrds(HashMap<(String, [u8; 16]), usize>);

impl LeafOrds {
    fn of(segments: &[OpenSegment<'_>]) -> Self {
        LeafOrds(
            segments
                .iter()
                .enumerate()
                .filter_map(|(i, s)| {
                    Some(((s.reader?.segment_name.clone(), s.reader?.segment_id()), i))
                })
                .collect(),
        )
    }

    /// The position of `reader`'s segment.
    pub(crate) fn ord_of(&self, reader: &crate::directory_reader::SegmentReader) -> Option<usize> {
        self.0
            .get(&(reader.segment_name.clone(), reader.segment_id()))
            .copied()
    }
}

/// A leaf's segment ordinals to global ones: the map's, or the identity
/// without one (a single segment).
fn segment_map(map: Option<&OrdinalMap>, ord: usize) -> Result<Option<&[i64]>> {
    match map {
        None => Ok(None),
        Some(m) => m
            .segment_ords(ord)
            .map(Some)
            .ok_or_else(|| Error::IllegalArgument(format!("the ordinal map has no segment {ord}"))),
    }
}

/// `segmentOrdToGlobalOrdLookup.get(segmentOrd)`, or the segment ordinal
/// itself without a map.
pub(crate) fn global_ord(map: Option<&[i64]>, segment_ord: i32) -> Result<i64> {
    match map {
        None => Ok(i64::from(segment_ord)),
        Some(m) => usize::try_from(segment_ord)
            .ok()
            .and_then(|o| m.get(o).copied())
            .ok_or_else(|| {
                Error::IllegalState(format!("segment ordinal {segment_ord} is not in the map"))
            }),
    }
}

/// Whether bit `ord` of `bits` is set; an ordinal outside the set is not.
pub(crate) fn bit(bits: &FixedBitSet, ord: i64) -> bool {
    // FBS: bounded by `bits.len()` here.
    usize::try_from(ord).is_ok_and(|o| o < bits.len() && bits.get(o))
}

fn set_bit(bits: &mut FixedBitSet, ord: i64) {
    if let Ok(o) = usize::try_from(ord) {
        // FBS: bounded by `bits.len()` here; the set is the map's value count.
        if o < bits.len() {
            bits.set(o);
        }
    }
}

/// `GlobalOrdinalsCollector`: the global ordinal of every matching document
/// that has one (`COMPLETE_NO_SCORES`).
pub struct GlobalOrdinalsCollector<'a> {
    field: String,
    ordinal_map: Option<Arc<OrdinalMap>>,
    values: Option<Box<dyn SortedDocValues + 'a>>,
    map: Option<Vec<i64>>,
    collected: FixedBitSet,
}

impl<'a> GlobalOrdinalsCollector<'a> {
    /// `new GlobalOrdinalsCollector(field, ordinalMap, valueCount)`.
    pub fn new(field: &str, ordinal_map: Option<Arc<OrdinalMap>>, value_count: usize) -> Self {
        Self {
            field: field.to_string(),
            ordinal_map,
            values: None,
            map: None,
            collected: FixedBitSet::new(value_count),
        }
    }

    /// `getCollectorOrdinals()`.
    pub fn into_ordinals(self) -> FixedBitSet {
        self.collected
    }
}

impl<'a> SegmentCollector<'a> for GlobalOrdinalsCollector<'a> {
    fn score_mode(&self) -> CollectorScoreMode {
        CollectorScoreMode::CompleteNoScores
    }

    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.values = Some(dv::get_sorted(leaf_reader(leaf)?, &self.field)?);
        self.map = segment_map(self.ordinal_map.as_deref(), ord)?.map(<[i64]>::to_vec);
        Ok(())
    }

    fn collect(&mut self, doc: i32, _score: f32) -> Result<()> {
        if let Some(values) = self.values.as_mut() {
            if values.advance_exact(doc)? {
                let g = global_ord(self.map.as_deref(), values.ord_value())?;
                set_bit(&mut self.collected, g);
            }
        }
        Ok(())
    }
}

/// What a [`GlobalOrdinalsWithScoreCollector`] collected: which global
/// ordinals matched, how often, and their combined scores.
#[derive(Debug, Clone)]
pub struct CollectedOrdinals {
    pub(crate) score_mode: ScoreMode,
    pub(crate) collected: FixedBitSet,
    /// `Scores`, every slot starting at `unset()`; empty for `None`.
    pub(crate) scores: Vec<f32>,
    /// `Occurrences`; empty unless `Avg` or a min/max bound.
    pub(crate) occurrences: Vec<i32>,
    pub(crate) do_min_max: bool,
    pub(crate) min: i32,
    pub(crate) max: i32,
}

impl CollectedOrdinals {
    /// `match(globalOrd)`.
    pub fn matches(&self, ord: i64) -> bool {
        if !bit(&self.collected, ord) {
            return false;
        }
        if !self.do_min_max {
            return true;
        }
        let n = usize::try_from(ord)
            .ok()
            .and_then(|o| self.occurrences.get(o).copied())
            .unwrap_or(0);
        n >= self.min && n <= self.max
    }

    /// `score(globalOrdinal)`: `1` for `None`, the mean for `Avg`.
    pub fn score(&self, ord: i64) -> f32 {
        let Ok(o) = usize::try_from(ord) else {
            return 0.0;
        };
        match self.score_mode {
            ScoreMode::None => 1.0,
            ScoreMode::Avg => {
                let s = self.scores.get(o).copied().unwrap_or(0.0);
                s / self.occurrences.get(o).copied().unwrap_or(0) as f32
            }
            _ => self.scores.get(o).copied().unwrap_or(0.0),
        }
    }
}

/// `GlobalOrdinalsWithScoreCollector` (`Sum`, `Min`, `Max`, `Avg`,
/// `NoScore`): per global ordinal, whether it matched, its scores combined
/// in `float` (`Math.min`/`Math.max` from `+Inf`/`-Inf`, sums from `0`), and
/// -- for `Avg` or a min/max bound -- how many documents had it.
pub struct GlobalOrdinalsWithScoreCollector<'a> {
    field: String,
    ordinal_map: Option<Arc<OrdinalMap>>,
    values: Option<Box<dyn SortedDocValues + 'a>>,
    map: Option<Vec<i64>>,
    state: CollectedOrdinals,
}

impl<'a> GlobalOrdinalsWithScoreCollector<'a> {
    /// The constructor of the `score_mode`'s subclass.
    ///
    /// # Errors
    /// Java's `IllegalStateException` for more than `Integer.MAX_VALUE`
    /// ordinals.
    pub fn new(
        field: &str,
        ordinal_map: Option<Arc<OrdinalMap>>,
        value_count: usize,
        score_mode: ScoreMode,
        min: i32,
        max: i32,
    ) -> Result<Self> {
        if value_count > i32::MAX as usize {
            return Err(Error::IllegalState(format!(
                "Can't collect more than [{}] ids",
                i32::MAX
            )));
        }
        let do_min_max = min > 1 || max < i32::MAX;
        let unset = match score_mode {
            ScoreMode::Min => f32::INFINITY,
            ScoreMode::Max => f32::NEG_INFINITY,
            _ => 0.0,
        };
        let scores = if score_mode == ScoreMode::None {
            Vec::new()
        } else {
            vec![unset; value_count]
        };
        let occurrences = if score_mode == ScoreMode::Avg || do_min_max {
            vec![0; value_count]
        } else {
            Vec::new()
        };
        Ok(Self {
            field: field.to_string(),
            ordinal_map,
            values: None,
            map: None,
            state: CollectedOrdinals {
                score_mode,
                collected: FixedBitSet::new(value_count),
                scores,
                occurrences,
                do_min_max,
                min,
                max,
            },
        })
    }

    /// What was collected.
    pub fn into_collected(self) -> CollectedOrdinals {
        self.state
    }
}

impl<'a> SegmentCollector<'a> for GlobalOrdinalsWithScoreCollector<'a> {
    fn score_mode(&self) -> CollectorScoreMode {
        if self.state.score_mode == ScoreMode::None {
            CollectorScoreMode::CompleteNoScores
        } else {
            CollectorScoreMode::Complete
        }
    }

    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.values = Some(dv::get_sorted(leaf_reader(leaf)?, &self.field)?);
        self.map = segment_map(self.ordinal_map.as_deref(), ord)?.map(<[i64]>::to_vec);
        Ok(())
    }

    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        let Some(values) = self.values.as_mut() else {
            return Ok(());
        };
        if !values.advance_exact(doc)? {
            return Ok(());
        }
        let g = global_ord(self.map.as_deref(), values.ord_value())?;
        let s = &mut self.state;
        set_bit(&mut s.collected, g);
        let Ok(o) = usize::try_from(g) else {
            return Ok(());
        };
        if let Some(existing) = s.scores.get_mut(o) {
            *existing = match s.score_mode {
                ScoreMode::Min => java_min_f32(*existing, score),
                ScoreMode::Max => java_max_f32(*existing, score),
                ScoreMode::Total | ScoreMode::Avg => *existing + score,
                ScoreMode::None => *existing,
            };
        }
        if let Some(n) = s.occurrences.get_mut(o) {
            *n = n.wrapping_add(1);
        }
        Ok(())
    }
}

/// `GlobalOrdinalsQuery`: the documents `to_query` matches whose join value
/// (their `join_field`'s ordinal, mapped to a global one) is among the
/// collected ones, constant-scored.
#[derive(Clone)]
pub struct GlobalOrdinalsQuery {
    pub(crate) found_ords: Arc<FixedBitSet>,
    pub join_field: String,
    pub(crate) ordinal_map: Option<Arc<OrdinalMap>>,
    /// Each segment's position in the searcher the map was built over
    /// (`context.ord`), by segment core.
    pub(crate) leaves: Arc<LeafOrds>,
    pub to_query: Box<Clause>,
    pub from_query: Arc<Clause>,
}

impl fmt::Debug for GlobalOrdinalsQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GlobalOrdinalsQuery{{joinField={}}}({:?}, {:?})@{:p}",
            self.join_field,
            self.to_query,
            self.from_query,
            Arc::as_ptr(&self.found_ords)
        )
    }
}

impl PartialEq for GlobalOrdinalsQuery {
    fn eq(&self, o: &Self) -> bool {
        self.join_field == o.join_field
            && self.to_query == o.to_query
            && self.from_query == o.from_query
            && (Arc::ptr_eq(&self.found_ords, &o.found_ords) || self.found_ords == o.found_ords)
    }
}

/// `GlobalOrdinalsWithScoreQuery`: [`GlobalOrdinalsQuery`] keeping only the
/// ordinals matched between `min` and `max` times, each document scored its
/// ordinal's combined score times the boost (`1` for `None`). Without
/// scores and bounds it runs as the [`GlobalOrdinalsQuery`] of the same
/// ordinals.
#[derive(Clone)]
pub struct GlobalOrdinalsWithScoreQuery {
    pub(crate) collected: Arc<CollectedOrdinals>,
    pub score_mode: ScoreMode,
    pub join_field: String,
    pub(crate) ordinal_map: Option<Arc<OrdinalMap>>,
    pub(crate) leaves: Arc<LeafOrds>,
    pub to_query: Box<Clause>,
    pub from_query: Arc<Clause>,
    pub min: i32,
    pub max: i32,
}

impl GlobalOrdinalsWithScoreQuery {
    /// The [`GlobalOrdinalsQuery`] it runs as without scores.
    pub(crate) fn as_global_ordinals_query(&self) -> GlobalOrdinalsQuery {
        GlobalOrdinalsQuery {
            found_ords: Arc::new(self.collected.collected.clone()),
            join_field: self.join_field.clone(),
            ordinal_map: self.ordinal_map.clone(),
            leaves: Arc::clone(&self.leaves),
            to_query: self.to_query.clone(),
            from_query: Arc::clone(&self.from_query),
        }
    }
}

impl fmt::Debug for GlobalOrdinalsWithScoreQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GlobalOrdinalsQuery{{joinField={},min={},max={},fromQuery={:?}}}({:?}, {})@{:p}",
            self.join_field,
            self.min,
            self.max,
            self.from_query,
            self.to_query,
            self.score_mode,
            Arc::as_ptr(&self.collected)
        )
    }
}

impl PartialEq for GlobalOrdinalsWithScoreQuery {
    fn eq(&self, o: &Self) -> bool {
        self.min == o.min
            && self.max == o.max
            && self.score_mode == o.score_mode
            && self.join_field == o.join_field
            && self.from_query == o.from_query
            && self.to_query == o.to_query
            && Arc::ptr_eq(&self.collected, &o.collected)
    }
}

/// `JoinUtil.createJoinQuery(joinField, fromQuery, toQuery, searcher,
/// scoreMode, ordinalMap, min, max)`: a join over one `SORTED` field both
/// sides share, by global ordinal. `ordinal_map` ([`ordinal_map`]) is
/// required with more than one segment and ignored with one; `min`/`max`
/// bound how many from-documents a join value needs (`0` and `i32::MAX`:
/// no bound).
///
/// # Errors
/// [`Error::IllegalArgument`] when the searcher has several segments and no
/// map is given (Java's message), and what searching reports.
#[allow(clippy::too_many_arguments)]
pub fn create_global_ordinals_join_query(
    join_field: &str,
    from_query: &Clause,
    to_query: &Clause,
    searcher: &IndexSearcher<'_, '_>,
    score_mode: ScoreMode,
    ordinal_map: Option<Arc<OrdinalMap>>,
    min: i32,
    max: i32,
) -> Result<Clause> {
    let segments = searcher.segments();
    let (ordinal_map, value_count) = match segments {
        [] => {
            return Ok(Clause::MatchNoDocs(
                MatchNoDocsQuery::new().with_reason("JoinUtil.createJoinQuery with no segments"),
            ))
        }
        [one] => {
            use crate::reader::LeafReader;
            match leaf_reader(one)?.sorted_doc_values(join_field)? {
                Some(values) => (None, usize::try_from(values.value_count()).unwrap_or(0)),
                None => {
                    return Ok(Clause::MatchNoDocs(
                        MatchNoDocsQuery::new()
                            .with_reason("JoinUtil.createJoinQuery: no join values"),
                    ))
                }
            }
        }
        _ => {
            let Some(map) = ordinal_map else {
                return Err(Error::IllegalArgument(
                    "OrdinalMap is required, because there is more than 1 segment".into(),
                ));
            };
            let n = usize::try_from(map.value_count()).unwrap_or(0);
            (Some(map), n)
        }
    };
    let leaves = Arc::new(LeafOrds::of(segments));
    let from = BooleanQuery {
        must: vec![from_query.clone()],
        ..Default::default()
    };
    if score_mode == ScoreMode::None && min <= 1 && max == i32::MAX {
        let mut c = GlobalOrdinalsCollector::new(join_field, ordinal_map.clone(), value_count);
        search_segments(searcher, &from, &mut c)?;
        return Ok(Clause::Extended(Box::new(ExtendedQuery::GlobalOrdinals(
            GlobalOrdinalsQuery {
                found_ords: Arc::new(c.into_ordinals()),
                join_field: join_field.to_string(),
                ordinal_map,
                leaves,
                to_query: Box::new(to_query.clone()),
                from_query: Arc::new(from_query.clone()),
            },
        ))));
    }
    let mut c = GlobalOrdinalsWithScoreCollector::new(
        join_field,
        ordinal_map.clone(),
        value_count,
        score_mode,
        min,
        max,
    )?;
    search_segments(searcher, &from, &mut c)?;
    Ok(Clause::Extended(Box::new(
        ExtendedQuery::GlobalOrdinalsWithScore(GlobalOrdinalsWithScoreQuery {
            collected: Arc::new(c.into_collected()),
            score_mode,
            join_field: join_field.to_string(),
            ordinal_map,
            leaves,
            to_query: Box::new(to_query.clone()),
            from_query: Arc::new(from_query.clone()),
            min,
            max,
        }),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::TermQuery;

    fn set(terms: &[&str]) -> Arc<[Vec<u8>]> {
        terms.iter().map(|t| t.as_bytes().to_vec()).collect()
    }

    #[test]
    fn java_float_min_and_max() {
        assert!(java_min_f32(f32::NAN, 1.0).is_nan());
        assert!(java_max_f32(1.0, f32::NAN).is_nan());
        assert!(java_min_f32(0.0, -0.0).is_sign_negative());
        assert!(java_min_f32(-0.0, 0.0).is_sign_negative());
        assert!(java_max_f32(-0.0, 0.0).is_sign_positive());
        assert!(java_max_f32(0.0, -0.0).is_sign_positive());
        assert_eq!(java_min_f32(1.0, 2.0), 1.0);
        assert_eq!(java_min_f32(2.0, 1.0), 1.0);
        assert_eq!(java_max_f32(1.0, 2.0), 2.0);
        assert_eq!(java_max_f32(2.0, 1.0), 2.0);
        assert!(is_positive_zero(0.0) && !is_positive_zero(-0.0));
    }

    /// `SeekingTermSetTermsEnum.accept` over a walk of the enum's terms:
    /// each seek, each match, the catch-up loop and the end.
    #[test]
    fn the_term_set_filter_seeks_and_accepts_as_java_does() {
        let mut f = SeekingTermSet::new(set(&["b", "d", "f"]));
        assert_eq!(f.next_seek(), Some(&b"b"[..]));
        assert_eq!(f.next_seek(), None, "a seek is handed over once");
        // The enum landed on `b`: accepted, seek on to `d`.
        assert_eq!(f.accept_term(b"b"), AcceptStatus::YesAndSeek);
        assert_eq!(f.next_seek(), Some(&b"d"[..]));
        // It landed past `d`, on `e`: catch up to `f`, which is past `e`.
        assert_eq!(f.accept_term(b"e"), AcceptStatus::NoAndSeek);
        assert_eq!(f.next_seek(), Some(&b"f"[..]));
        // On the last term: accepted, nothing more to seek.
        assert_eq!(f.accept_term(b"f"), AcceptStatus::Yes);
        assert_eq!(f.accept_term(b"g"), AcceptStatus::End);

        // Catching up onto an exact match, with more to come and at the end.
        let mut f = SeekingTermSet::new(set(&["a", "c", "e"]));
        f.next_seek();
        assert_eq!(f.accept_term(b"c"), AcceptStatus::YesAndSeek);
        assert_eq!(f.next_seek(), Some(&b"e"[..]));
        let mut f = SeekingTermSet::new(set(&["a", "c"]));
        f.next_seek();
        assert_eq!(f.accept_term(b"c"), AcceptStatus::Yes);
        // Behind on the last term: rejected.
        let mut f = SeekingTermSet::new(set(&["a", "c"]));
        f.next_seek();
        assert_eq!(f.accept_term(b"b"), AcceptStatus::NoAndSeek);
        assert_eq!(f.accept_term(b"bb"), AcceptStatus::No);
        let mut f = SeekingTermSet::new(set(&["a"]));
        assert_eq!(f.accept_term(b"0"), AcceptStatus::No);
        // An empty set ends at once.
        let mut f = SeekingTermSet::new(set(&[]));
        assert_eq!(f.next_seek(), None);
        assert_eq!(f.accept_term(b"a"), AcceptStatus::End);
        let mut f = SeekingTermSet::new(set(&["x"]));
        assert_eq!(f.accept(b"x").unwrap(), AcceptStatus::Yes);
        assert_eq!(f.next_seek_term(None, None).unwrap(), Some(b"x".to_vec()));
    }

    #[test]
    fn numeric_types_encode_as_their_points() {
        assert_eq!(NumericType::Int.encode(1), vec![0x80, 0, 0, 1]);
        assert_eq!(NumericType::Int.encode(-1), vec![0x7f, 0xff, 0xff, 0xff]);
        assert_eq!(NumericType::Long.encode(0), vec![0x80, 0, 0, 0, 0, 0, 0, 0]);
        let f = NumericType::Float.encode(i64::from(1.5f32.to_bits() as i32));
        assert_eq!(f.len(), 4);
        let d = NumericType::Double.encode(2.5f64.to_bits() as i64);
        assert_eq!(d.len(), 8);
        // Every NaN encodes as Java's one NaN.
        assert_eq!(
            NumericType::Float.encode(0x7fc0_0001),
            NumericType::Float.encode(0x7fc0_0000)
        );
        assert_eq!(
            NumericType::Double.encode(0x7ff8_0000_0000_0001),
            NumericType::Double.encode(0x7ff8_0000_0000_0000)
        );
        assert_eq!(NumericType::Int.render(&NumericType::Int.encode(-7)), "-7");
        assert_eq!(NumericType::Long.render(&NumericType::Long.encode(9)), "9");
        assert_eq!(
            NumericType::Float.render(&NumericType::Float.encode(i64::from((-1.5f32).to_bits()))),
            "-1.5"
        );
        assert_eq!(
            NumericType::Double.render(&NumericType::Double.encode(0.25f64.to_bits() as i64)),
            "0.25"
        );
        assert_eq!(NumericType::Int.render(&[1, 2]), "unsupported");
        assert_eq!(NumericType::Long.bytes(), 8);
        assert_eq!(NumericType::Float.bytes(), 4);
        assert_eq!(bytes_ref_string(&[0x80, 0x0a]), "[80 a]");
    }

    #[test]
    fn point_set_with_scores_checks_its_stream() {
        let q = || Arc::new(Clause::Term(TermQuery::new("f", b"x".to_vec())));
        let ok = PointInSetIncludingScoreQuery::new(
            ScoreMode::Max,
            q(),
            false,
            "p",
            4,
            NumericType::Int,
            vec![
                (NumericType::Int.encode(1), 1.0),
                (NumericType::Int.encode(2), 2.0),
            ],
        )
        .unwrap();
        assert_eq!(ok.points_and_scores().count(), 2);
        assert!(format!("{ok:?}").contains("p:{1 2}"));
        assert_eq!(ok, ok.clone());
        let err = |stream: Vec<(Vec<u8>, f32)>, bytes: usize| {
            PointInSetIncludingScoreQuery::new(
                ScoreMode::Max,
                q(),
                false,
                "p",
                bytes,
                NumericType::Int,
                stream,
            )
            .unwrap_err()
            .to_string()
        };
        assert!(err(Vec::new(), 0).contains("bytesPerDim must be > 0"));
        assert!(err(vec![(vec![1, 2], 1.0)], 4).contains("packed point length should be 4"));
        let one = NumericType::Int.encode(1);
        let two = NumericType::Int.encode(2);
        assert!(err(vec![(one.clone(), 1.0), (one.clone(), 1.0)], 4)
            .contains("unexpected duplicated value: [80 0 0 1]"));
        assert!(err(vec![(two, 1.0), (one, 1.0)], 4).contains("values are out of order"));
    }

    #[test]
    fn collected_ordinals_match_and_score() {
        let mut c = CollectedOrdinals {
            score_mode: ScoreMode::Avg,
            collected: FixedBitSet::new(4),
            scores: vec![6.0, 0.0, 0.0, 0.0],
            occurrences: vec![3, 0, 0, 0],
            do_min_max: true,
            min: 2,
            max: 3,
        };
        c.collected.set(0);
        assert!(c.matches(0));
        assert!(!c.matches(1), "never collected");
        assert!(!c.matches(-1) && !c.matches(9));
        assert_eq!(c.score(0), 2.0);
        assert_eq!(c.score(-1), 0.0);
        c.max = 2;
        assert!(!c.matches(0), "three occurrences, at most two");
        c.score_mode = ScoreMode::None;
        assert_eq!(c.score(0), 1.0);
        c.score_mode = ScoreMode::Max;
        assert_eq!(c.score(0), 6.0);
        c.do_min_max = false;
        assert!(c.matches(0));
    }

    #[test]
    fn global_ordinal_lookups() {
        assert_eq!(global_ord(None, 5).unwrap(), 5);
        assert_eq!(global_ord(Some(&[3, 7]), 1).unwrap(), 7);
        assert!(global_ord(Some(&[3]), 1).is_err());
        assert!(global_ord(Some(&[3]), -1).is_err());
        let mut b = FixedBitSet::new(3);
        set_bit(&mut b, 2);
        set_bit(&mut b, 9);
        set_bit(&mut b, -1);
        assert!(bit(&b, 2) && !bit(&b, 9) && !bit(&b, -1) && !bit(&b, 0));
        assert!(segment_map(None, 0).unwrap().is_none());
    }

    #[test]
    fn collectors_score_in_javas_order() {
        // Single-valued: the first score is taken, a later one combined --
        // and a combined score of exactly `+0.0` is overwritten.
        let mut c = TermsWithScoreCollector::new("f", false, ScoreMode::Total);
        let id = c.slot(b"a").unwrap();
        c.collect_sv(id, 1.0);
        c.collect_sv(id, 2.0);
        assert_eq!(c.score_sums[id], 3.0);
        c.collect_sv(id, -3.0);
        c.collect_sv(id, 5.0);
        assert_eq!(c.score_sums[id], 5.0, "a zero sum restarts");
        for (mode, want) in [(ScoreMode::Min, 1.0), (ScoreMode::Max, 4.0)] {
            let mut c = TermsWithScoreCollector::new("f", false, mode);
            let id = c.slot(b"a").unwrap();
            for s in [2.0, 1.0, 4.0] {
                c.collect_sv(id, s);
            }
            assert_eq!(c.score_sums[id], want, "{mode}");
        }
        let mut c = TermsWithScoreCollector::new("f", false, ScoreMode::Avg);
        let id = c.slot(b"a").unwrap();
        for s in [1.0, 2.0, 6.0] {
            c.collect_sv(id, s);
        }
        let (terms, ords, scores) = c.into_terms_and_scores();
        assert_eq!((terms.len(), ords, scores), (1, vec![0], vec![3.0]));

        for (mode, want) in [
            (ScoreMode::Total, 7.0),
            (ScoreMode::Min, 1.0),
            (ScoreMode::Max, 4.0),
            (ScoreMode::Avg, 7.0 / 3.0),
            (ScoreMode::None, 0.0),
        ] {
            let mut c = TermsWithScoreCollector::new("f", true, mode);
            let id = c.slot(b"z").unwrap();
            let other = c.slot(b"a").unwrap();
            for s in [2.0, 1.0, 4.0] {
                c.collect_mv(id, s);
            }
            assert_eq!(other, 1);
            let (terms, ords, scores) = c.into_terms_and_scores();
            assert_eq!(terms, vec![b"z".to_vec(), b"a".to_vec()]);
            assert_eq!(ords, vec![1, 0]);
            assert_eq!(scores[0], want as f32, "{mode}");
        }
        // `SV` with `None` (which the factory never builds) does nothing.
        let mut c = TermsWithScoreCollector::new("f", false, ScoreMode::None);
        let id = c.slot(b"a").unwrap();
        c.collect_sv(id, 2.0);
        c.collect_sv(id, 3.0);
        assert_eq!(c.score_sums[id], 2.0);
    }

    #[test]
    fn numeric_collector_aggregates_per_value() {
        for (mode, want) in [
            (ScoreMode::Max, 4.0f32),
            (ScoreMode::Min, 1.0),
            (ScoreMode::Total, 7.0),
            (ScoreMode::Avg, 7.0 / 3.0),
            (ScoreMode::None, 0.0),
        ] {
            let mut c = NumericJoinCollector::new("f", false, mode);
            for s in [2.0, 1.0, 4.0] {
                c.add(5, s);
            }
            c.add(-1, 1.0);
            let sorted = c.into_sorted();
            assert_eq!(sorted.len(), 2);
            assert_eq!(sorted[0].0, -1);
            assert_eq!(sorted[1], (5, want), "{mode}");
        }
    }

    #[test]
    fn queries_compare_and_print() {
        let from = Arc::new(Clause::Term(TermQuery::new("f", b"x".to_vec())));
        let source = TermSetSource {
            field: "to".into(),
            terms: set(&["a", "b"]),
            from_field: "from".into(),
            from_query: Arc::clone(&from),
        };
        assert!(source.contains(b"b") && !source.contains(b"c"));
        let mut copy = source.clone();
        copy.terms = set(&["a", "b"]);
        assert_eq!(source, copy, "equal sets");
        copy.terms = set(&["a"]);
        assert_ne!(source, copy);
        assert!(format!("{source:?}").starts_with("TermsQuery{field=to"));

        let q = TermsIncludingScoreQuery {
            score_mode: ScoreMode::Max,
            to_field: "to".into(),
            multiple_values_per_document: false,
            terms: set(&["a"]),
            scores: vec![1.5].into(),
            from_field: "from".into(),
            from_query: Arc::clone(&from),
        };
        let mut other = q.clone();
        assert_eq!(q, other);
        other.terms = set(&["a"]);
        assert_eq!(q, other, "equal terms and scores");
        other.scores = vec![2.5].into();
        assert_ne!(q, other);
        assert!(format!("{q:?}").starts_with("TermsIncludingScoreQuery{field=to;"));
        assert!(matches!(
            q.as_terms_query(),
            Clause::Extended(e) if matches!(*e, ExtendedQuery::MultiTerm(_))
        ));

        let g = GlobalOrdinalsQuery {
            found_ords: Arc::new(FixedBitSet::new(2)),
            join_field: "gj".into(),
            ordinal_map: None,
            leaves: Arc::new(LeafOrds::default()),
            to_query: Box::new((*from).clone()),
            from_query: Arc::clone(&from),
        };
        let mut g2 = g.clone();
        assert_eq!(g, g2);
        g2.found_ords = Arc::new(FixedBitSet::new(2));
        assert_eq!(g, g2, "equal ordinals");
        assert!(format!("{g:?}").starts_with("GlobalOrdinalsQuery{joinField=gj}"));
        let w = GlobalOrdinalsWithScoreQuery {
            collected: Arc::new(CollectedOrdinals {
                score_mode: ScoreMode::None,
                collected: FixedBitSet::new(2),
                scores: Vec::new(),
                occurrences: Vec::new(),
                do_min_max: false,
                min: 0,
                max: i32::MAX,
            }),
            score_mode: ScoreMode::None,
            join_field: "gj".into(),
            ordinal_map: None,
            leaves: Arc::new(LeafOrds::default()),
            to_query: Box::new((*from).clone()),
            from_query: from,
            min: 0,
            max: i32::MAX,
        };
        assert_eq!(w, w.clone());
        assert!(format!("{w:?}").contains("min=0"));
        assert_eq!(w.as_global_ordinals_query().join_field, "gj");
    }
}
