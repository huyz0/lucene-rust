//! `PhraseWildcardQuery`: a phrase whose positions may be multi-term queries
//! (prefix, wildcard, regexp, range, automaton, fuzzy), each expanded per
//! segment within a shared budget.
//!
//! `createWeight` is reproduced at the searcher's rewrite
//! ([`PhraseWildcardQuery::resolve`]): the segments sorted by their field's
//! term count, ascending (`SegmentTermsSizeComparator`, a stable sort);
//! the single terms first, each looked up in every remaining segment, a
//! segment without it dropped from the list (segment optimisation, on by
//! default), the query matching nothing once a term matches nowhere; then
//! each multi-term in phrase order with `remaining / remainingMultiTerms` of
//! the budget, expanded segment by segment in term order until its share is
//! spent (a segment that spends it ends the walk), a segment it does not
//! expand in dropped. The similarity is built from the single terms'
//! reader-wide statistics and each expanded term's statistics summed over
//! the segments it was expanded in; each segment's phrase runs over the
//! union of its own expansions per position (`UnionPostingsEnum`), exact or
//! sloppy.
//!
//! Differences: an expanded term's statistics join the similarity in term
//! order (Java's `HashMap` order varies by JVM run; BM25 sums the idfs in a
//! `double`, so the order does not change a score here); `toString`,
//! `equals`/`hashCode`, `visit` and the test counters are not ported.

use std::collections::BTreeMap;
use std::ops::ControlFlow;

use lucene_codecs::blocktree::{BlockTreeFields, FieldTerms};
use lucene_codecs::field_infos::IndexOptions;
use lucene_codecs::fuzzy::FuzzyMatch;

use crate::exec::extended::{eager_phrase, sim_scorer, visit_terms};
use crate::exec::{BoxScorer, LeafContext, Mode};
use crate::extended_query::{ExtendedQuery, MultiTermSource};
use crate::multi_segment::OpenSegment;
use crate::query::{Clause, FuzzyQuery, MatchNoDocsQuery, TermQuery};
use crate::similarities::{CollectionStatistics, TermStatistics};
use crate::sloppy_phrase::PhraseRepeats;
use crate::{Error, Result};

/// One position of the phrase.
#[derive(Debug, Clone, PartialEq)]
pub enum PhraseWildcardTerm {
    /// `SingleTerm`.
    Term(Vec<u8>),
    /// `MultiTerm` over a term enumeration, with the query it came from
    /// (what a one-position query rewrites to, its rewrite method intact).
    Multi(MultiTermSource, Box<Clause>),
    /// `MultiTerm` over a `FuzzyQuery` (`FuzzyTermsEnum`).
    Fuzzy(FuzzyQuery),
}

/// `PhraseWildcardQuery`, built with [`PhraseWildcardBuilder`].
#[derive(Debug, Clone, PartialEq)]
pub struct PhraseWildcardQuery {
    pub field: String,
    pub terms: Vec<PhraseWildcardTerm>,
    pub slop: u32,
    pub max_multi_term_expansions: usize,
    pub segment_optimization: bool,
}

/// `PhraseWildcardQuery.Builder`.
#[derive(Debug, Clone)]
pub struct PhraseWildcardBuilder {
    query: PhraseWildcardQuery,
}

impl PhraseWildcardBuilder {
    /// `Builder(field, maxMultiTermExpansions)`, segment optimisation on.
    pub fn new(field: impl Into<String>, max_multi_term_expansions: usize) -> Self {
        Self::with_segment_optimization(field, max_multi_term_expansions, true)
    }

    /// `Builder(field, maxMultiTermExpansions, segmentOptimizationEnabled)`.
    pub fn with_segment_optimization(
        field: impl Into<String>,
        max_multi_term_expansions: usize,
        segment_optimization: bool,
    ) -> Self {
        Self {
            query: PhraseWildcardQuery {
                field: field.into(),
                terms: Vec::new(),
                slop: 0,
                max_multi_term_expansions,
                segment_optimization,
            },
        }
    }

    /// `addTerm(bytes)`.
    pub fn add_term(&mut self, term: impl Into<Vec<u8>>) -> &mut Self {
        self.query.terms.push(PhraseWildcardTerm::Term(term.into()));
        self
    }

    /// `addTerm(Term)`.
    ///
    /// # Errors
    /// A term of another field.
    pub fn add_field_term(&mut self, q: &TermQuery) -> Result<&mut Self> {
        self.check_field("Term", &q.field)?;
        Ok(self.add_term(q.term.clone()))
    }

    /// `addMultiTerm(query)`: a prefix, wildcard, regexp, fuzzy or other
    /// multi-term query clause.
    ///
    /// # Errors
    /// A query of another field, or a clause that is not a multi-term query.
    pub fn add_multi_term(&mut self, clause: &Clause) -> Result<&mut Self> {
        let (name, term) = match clause {
            Clause::Prefix(q) => (
                "PrefixQuery",
                PhraseWildcardTerm::Multi(
                    MultiTermSource::Prefix(q.clone()),
                    Box::new(clause.clone()),
                ),
            ),
            Clause::Wildcard(q) => (
                "WildcardQuery",
                PhraseWildcardTerm::Multi(
                    MultiTermSource::Wildcard(q.clone()),
                    Box::new(clause.clone()),
                ),
            ),
            Clause::Regexp(q) => (
                "RegexpQuery",
                PhraseWildcardTerm::Multi(
                    MultiTermSource::Regexp(q.clone()),
                    Box::new(clause.clone()),
                ),
            ),
            Clause::Fuzzy(q) => ("FuzzyQuery", PhraseWildcardTerm::Fuzzy(q.clone())),
            Clause::Extended(e) => match e.as_ref() {
                ExtendedQuery::MultiTerm(m) => (
                    "MultiTermQuery",
                    PhraseWildcardTerm::Multi(m.source.clone(), Box::new(clause.clone())),
                ),
                _ => return Err(not_multi_term()),
            },
            _ => return Err(not_multi_term()),
        };
        let field = match &term {
            PhraseWildcardTerm::Multi(s, _) => s.field().to_string(),
            PhraseWildcardTerm::Fuzzy(q) => q.field.clone(),
            PhraseWildcardTerm::Term(_) => self.query.field.clone(),
        };
        self.check_field(name, &field)?;
        self.query.terms.push(term);
        Ok(self)
    }

    fn check_field(&self, name: &str, field: &str) -> Result<()> {
        if field == self.query.field {
            return Ok(());
        }
        Err(Error::InvalidQuery(format!(
            "{name} field \"{field}\" cannot be different from the PhraseWildcardQuery field \"{}\"",
            self.query.field
        )))
    }

    /// `setSlop(slop)`.
    ///
    /// # Errors
    /// A negative slop.
    pub fn set_slop(&mut self, slop: i32) -> Result<&mut Self> {
        self.query.slop = u32::try_from(slop)
            .map_err(|_| Error::InvalidQuery("slop value cannot be negative".into()))?;
        Ok(self)
    }

    pub fn build(&self) -> PhraseWildcardQuery {
        self.query.clone()
    }
}

fn not_multi_term() -> Error {
    Error::InvalidQuery("PhraseWildcardQuery.addMultiTerm takes a multi-term query".into())
}

/// The weight `createWeight` builds, per segment: what each position
/// expands to in it.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedPhraseWildcard {
    pub(crate) field: String,
    pub(crate) slop: u32,
    /// `(segment fields' address, terms per position)`, for the segments
    /// every position has terms in. The address identifies a segment only
    /// while the segments it was resolved on are open: a resolved query is
    /// for those segments alone, as Java's per-leaf `TermStates` are for the
    /// reader they were built on (`TermStates.wasBuiltFor`).
    pub(crate) segments: Vec<(usize, Vec<Vec<Vec<u8>>>)>,
    /// Every position's terms over the whole reader (`TermData.terms`), what
    /// a sloppy phrase's repeated-term check reads.
    pub(crate) all_terms: Vec<Vec<Vec<u8>>>,
    pub(crate) collection: CollectionStatistics,
    pub(crate) stats: Vec<TermStatistics>,
}

fn address(fields: &BlockTreeFields) -> usize {
    std::ptr::from_ref(fields) as usize
}

impl PhraseWildcardQuery {
    /// `rewrite` then `createWeight` against `segments`.
    ///
    /// # Errors
    /// A field indexed without positions (`IllegalStateException`); what
    /// reading the term dictionaries reports.
    pub fn resolve(&self, segments: &[OpenSegment<'_>]) -> Result<Clause> {
        let no_match = || {
            Clause::MatchNoDocs(MatchNoDocsQuery::new().with_reason("Empty PhraseWildcardQuery"))
        };
        match self.terms.as_slice() {
            [] => return Ok(no_match()),
            [only] => {
                return Ok(match only {
                    PhraseWildcardTerm::Term(t) => {
                        Clause::Term(TermQuery::new(self.field.clone(), t.clone()))
                    }
                    // Java returns the multi-term query it was given.
                    PhraseWildcardTerm::Multi(_, original) => original.as_ref().clone(),
                    PhraseWildcardTerm::Fuzzy(q) => Clause::Fuzzy(q.clone()),
                });
            }
            _ => {}
        }
        let field = self.field.as_str();
        // `SegmentTermsSizeComparator`: by the field's term count, stable.
        let mut order: Vec<usize> = (0..segments.len()).collect();
        order.sort_by_key(|&i| segments[i].fields.field(field).map_or(0, |ft| ft.num_terms));
        let positions = self.terms.len();
        let mut per_position: Vec<BTreeMap<usize, Vec<Vec<u8>>>> = vec![BTreeMap::new(); positions];
        let mut all_terms: Vec<Vec<Vec<u8>>> = vec![Vec::new(); positions];
        let mut stats = Vec::new();
        let mut num_multi = 0usize;
        for (p, term) in self.terms.iter().enumerate() {
            let PhraseWildcardTerm::Term(term) = term else {
                num_multi = num_multi.saturating_add(1);
                continue;
            };
            // `collectSingleTermData`.
            all_terms[p].push(term.clone());
            if let Some(g) = crate::multi_segment::global_term_stats(segments, field, term)? {
                if g.doc_freq > 0 {
                    stats.push(g.term_statistics(term));
                }
            }
            let mut matches = 0usize;
            let mut kept = Vec::with_capacity(order.len());
            for &i in &order {
                let mut found = false;
                if let Some(ft) = segments[i].fields.field(field) {
                    self.check_positions(ft)?;
                    if ft.try_seek_exact(term)?.is_some() {
                        found = true;
                        matches = matches.saturating_add(1);
                        per_position[p].insert(i, vec![term.clone()]);
                    }
                }
                if found || !self.segment_optimization {
                    kept.push(i);
                }
            }
            order = kept;
            if matches == 0 {
                return Ok(no_match());
            }
        }
        let mut remaining = self.max_multi_term_expansions;
        let mut remaining_multi = num_multi;
        for (p, term) in self.terms.iter().enumerate() {
            if matches!(term, PhraseWildcardTerm::Term(_)) {
                continue;
            }
            // `collectMultiTermData`.
            let max_for_term = remaining / remaining_multi.max(1);
            let mut term_stats: BTreeMap<Vec<u8>, (i64, i64)> = BTreeMap::new();
            let mut expansions = 0usize;
            let mut kept = Vec::with_capacity(order.len());
            let mut stop = false;
            for (k, &i) in order.iter().enumerate() {
                if stop {
                    kept.extend_from_slice(&order[k..]);
                    break;
                }
                let budget = max_for_term.saturating_sub(expansions);
                let mut expanded = Vec::new();
                if let Some(ft) = segments[i].fields.field(field) {
                    self.check_positions(ft)?;
                    expand(segments[i].fields, ft, term, budget, &mut expanded)?;
                    stop = expanded.len() == budget;
                }
                if expanded.is_empty() {
                    if !self.segment_optimization {
                        kept.push(i);
                    }
                    continue;
                }
                kept.push(i);
                expansions = expansions.saturating_add(expanded.len());
                let mut terms = Vec::with_capacity(expanded.len());
                for (t, df, ttf) in expanded {
                    let e = term_stats.entry(t.clone()).or_insert((0, 0));
                    e.0 = e.0.saturating_add(i64::from(df));
                    e.1 = if e.1 >= 0 && ttf >= 0 {
                        e.1.saturating_add(ttf)
                    } else {
                        -1
                    };
                    terms.push(t);
                }
                per_position[p].insert(i, terms);
            }
            order = kept;
            for (t, (df, ttf)) in term_stats {
                if df > 0 {
                    stats.push(TermStatistics {
                        doc_freq: df,
                        total_term_freq: ttf.max(df),
                        term: Some(std::sync::Arc::from(t.as_slice())),
                    });
                }
                all_terms[p].push(t);
            }
            if expansions == 0 {
                return Ok(no_match());
            }
            remaining = remaining.saturating_sub(expansions);
            remaining_multi = remaining_multi.saturating_sub(1);
        }
        // The segments every position has terms in; another's
        // `getPhraseMatcher` returns null.
        let mut resolved = Vec::new();
        for (i, seg) in segments.iter().enumerate() {
            let slots: Option<Vec<Vec<Vec<u8>>>> =
                per_position.iter().map(|m| m.get(&i).cloned()).collect();
            if let Some(slots) = slots {
                resolved.push((address(seg.fields), slots));
            }
        }
        let collection = crate::multi_segment::global_term_stats(segments, field, &[])?
            .map(|c| c.collection_statistics());
        let Some(collection) = collection else {
            return Ok(no_match());
        };
        Ok(Clause::Extended(Box::new(
            ExtendedQuery::PhraseWildcardWeight(ResolvedPhraseWildcard {
                field: self.field.clone(),
                slop: self.slop,
                segments: resolved,
                all_terms,
                collection,
                stats,
            }),
        )))
    }

    /// `checkTermsHavePositions`.
    fn check_positions(&self, ft: &FieldTerms) -> Result<()> {
        if matches!(
            ft.index_options(),
            IndexOptions::DocsAndFreqsAndPositions
                | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        ) {
            return Ok(());
        }
        Err(Error::IllegalState(format!(
            "field \"{}\" was indexed without position data; cannot run PhraseWildcardQuery",
            self.field
        )))
    }
}

/// `collectMultiTermDataForSegment`: up to `budget` of the terms `term`
/// enumerates in `ft`, in term order, with their `docFreq` and
/// `totalTermFreq`.
fn expand(
    fields: &BlockTreeFields,
    ft: &FieldTerms,
    term: &PhraseWildcardTerm,
    budget: usize,
    out: &mut Vec<(Vec<u8>, i32, i64)>,
) -> Result<()> {
    if budget == 0 {
        return Ok(());
    }
    match term {
        PhraseWildcardTerm::Term(_) => {}
        PhraseWildcardTerm::Multi(source, _) => {
            visit_terms(fields, source, Some(budget), &mut |t, seeked| {
                out.push((t, seeked.stats.doc_freq, seeked.stats.total_term_freq));
                Ok(ControlFlow::Continue(()))
            })?;
        }
        // `FuzzyQuery.getTermsEnum`: the exact term alone at zero edits.
        PhraseWildcardTerm::Fuzzy(q) if q.max_edits == 0 => {
            if let Some(s) = ft.try_seek_exact(&q.term)? {
                out.push((q.term.clone(), s.doc_freq, s.total_term_freq));
            }
        }
        PhraseWildcardTerm::Fuzzy(q) => {
            let pattern = FuzzyMatch::new(&q.term, q.max_edits, q.prefix_length, q.transpositions);
            for next in ft.fuzzy_intersect(&pattern).take(budget) {
                let (t, s) = next?;
                out.push((t, s.doc_freq, s.total_term_freq));
            }
        }
    }
    Ok(())
}

/// `PhraseWeight.scorer` over this segment's expansions.
pub(crate) fn scorer<'a>(
    ctx: &LeafContext<'a>,
    q: &ResolvedPhraseWildcard,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let here = address(ctx.fields);
    let Some((_, slots)) = q.segments.iter().find(|(a, _)| *a == here) else {
        return Ok(None);
    };
    if ctx.fields.field(&q.field).is_none() {
        return Ok(None);
    }
    let Some(pos_in) = ctx.pos_in else {
        return Err(Error::MissingPosInput);
    };
    let scorer = sim_scorer(ctx, &q.field, boost, &q.collection, &q.stats);
    let repeats = PhraseRepeats::for_multi_phrase(&q.all_terms);
    let offsets: Vec<i32> = (0..slots.len())
        .map(|i| i32::try_from(i).unwrap_or(i32::MAX))
        .collect();
    eager_phrase(
        ctx, pos_in, &q.field, slots, &offsets, &repeats, q.slop, &scorer, mode,
    )
}
