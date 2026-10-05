//! The queries M7 brings into the scorer tree, behind one [`Clause::Extended`]
//! variant: `SynonymQuery`, `CombinedFieldQuery`, `NGramPhraseQuery`, the
//! `MultiTermQuery` rewrite methods (`TermRangeQuery`, `AutomatonQuery`, and
//! the prefix/wildcard/regexp family under a chosen `RewriteMethod`),
//! `BlendedTermQuery`, the Indri and fusion queries (`IndriAndQuery`,
//! `LogOddsFusionQuery`, `BayesianScoreQuery`), the doc-values and
//! index-sort ranges, general points, `DocAndScoreQuery` (what a KNN or
//! vector-similarity query rewrites to), and any query of the `document`
//! package ([`DocumentClause`]: the geo point and shape queries).
//!
//! Each is a plain description of the Java query; how it runs per segment is
//! `exec::extended`, and the reader-wide work its `Weight` does
//! (`TermStates.build`, a `TermCollectingRewrite`) is gathered with the other
//! statistics in [`crate::multi_segment::global_boolean_stats`].

use std::sync::Arc;

use lucene_util::automaton::Automaton;

use crate::query::{Clause, PhraseQuery, PrefixQuery, RegexpQuery, WildcardQuery};
use crate::{Error, Result};

/// `IndexSearcher.getMaxClauseCount()`'s default.
pub const MAX_CLAUSE_COUNT: usize = 1024;

/// One of the queries [`Clause::Extended`] carries.
#[derive(Debug, Clone, PartialEq)]
pub enum ExtendedQuery {
    Synonym(SynonymQuery),
    CombinedField(CombinedFieldQuery),
    NGramPhrase(NGramPhraseQuery),
    MultiTerm(MultiTermQuery),
    Blended(BlendedTermQuery),
    IndriAnd(IndriAndQuery),
    LogOddsFusion(LogOddsFusionQuery),
    BayesianScore(BayesianScoreQuery),
    DocAndScore(DocAndScoreQuery),
    NumericDocValuesRange(NumericDocValuesRangeQuery),
    IndexSortRange(IndexSortSortedNumericDocValuesRangeQuery),
    PointRange(PointRangeQuery),
    PointInSet(PointInSetQuery),
    IndexOrDocValues(IndexOrDocValuesQuery),
    /// Searched only once [`crate::rescorer::rewrite_rescore_clauses`] (which
    /// `IndexSearcher`'s searches run) has made it a [`DocAndScoreQuery`].
    RescoreTopN(crate::rescorer::RescoreTopNQuery),
    /// A query of the `document` package as a leaf of the tree.
    Document(DocumentClause),
    /// `lucene-join`'s block joins ([`crate::join`]).
    ToParentBlockJoin(crate::join::ToParentBlockJoinQuery),
    ToChildBlockJoin(crate::join::ToChildBlockJoinQuery),
    ParentChildrenBlockJoin(crate::join::ParentChildrenBlockJoinQuery),
    ParentsChildrenBlockJoin(crate::join::ParentsChildrenBlockJoinQuery),
    /// `lucene-join`'s query-time joins' to-side queries ([`crate::join`]).
    TermsIncludingScore(crate::join::TermsIncludingScoreQuery),
    GlobalOrdinals(crate::join::GlobalOrdinalsQuery),
    GlobalOrdinalsWithScore(crate::join::GlobalOrdinalsWithScoreQuery),
    PointInSetIncludingScore(crate::join::PointInSetIncludingScoreQuery),
    /// `lucene-queries`' function queries ([`crate::function`]).
    Function(crate::function::FunctionQuery),
    FunctionRange(crate::function::FunctionRangeQuery),
    FunctionMatch(crate::function::FunctionMatchQuery),
    FunctionScore(crate::function::FunctionScoreQuery),
    /// `lucene-queries`' `IntervalQuery` ([`crate::intervals`]).
    Interval(crate::intervals::IntervalQuery),
}

macro_rules! into_clause {
    ($($ty:ident => $variant:ident),* $(,)?) => {
        $(
            impl From<$ty> for Clause {
                fn from(q: $ty) -> Self {
                    Clause::Extended(Box::new(ExtendedQuery::$variant(q)))
                }
            }
        )*
    };
}

into_clause! {
    SynonymQuery => Synonym,
    CombinedFieldQuery => CombinedField,
    NGramPhraseQuery => NGramPhrase,
    MultiTermQuery => MultiTerm,
    BlendedTermQuery => Blended,
    IndriAndQuery => IndriAnd,
    LogOddsFusionQuery => LogOddsFusion,
    BayesianScoreQuery => BayesianScore,
    DocAndScoreQuery => DocAndScore,
    NumericDocValuesRangeQuery => NumericDocValuesRange,
    IndexSortSortedNumericDocValuesRangeQuery => IndexSortRange,
    PointRangeQuery => PointRange,
    PointInSetQuery => PointInSet,
    IndexOrDocValuesQuery => IndexOrDocValues,
    DocumentClause => Document,
}

impl From<crate::function::FunctionQuery> for Clause {
    fn from(q: crate::function::FunctionQuery) -> Self {
        Clause::Extended(Box::new(ExtendedQuery::Function(q)))
    }
}

impl From<crate::function::FunctionRangeQuery> for Clause {
    fn from(q: crate::function::FunctionRangeQuery) -> Self {
        Clause::Extended(Box::new(ExtendedQuery::FunctionRange(q)))
    }
}

impl From<crate::function::FunctionMatchQuery> for Clause {
    fn from(q: crate::function::FunctionMatchQuery) -> Self {
        Clause::Extended(Box::new(ExtendedQuery::FunctionMatch(q)))
    }
}

impl From<crate::function::FunctionScoreQuery> for Clause {
    fn from(q: crate::function::FunctionScoreQuery) -> Self {
        Clause::Extended(Box::new(ExtendedQuery::FunctionScore(q)))
    }
}

impl From<crate::rescorer::RescoreTopNQuery> for Clause {
    fn from(q: crate::rescorer::RescoreTopNQuery) -> Self {
        Clause::Extended(Box::new(ExtendedQuery::RescoreTopN(q)))
    }
}

impl ExtendedQuery {
    /// `Query.toString`-like name, for explanations and errors.
    pub fn name(&self) -> &'static str {
        match self {
            ExtendedQuery::Synonym(_) => "SynonymQuery",
            ExtendedQuery::CombinedField(_) => "CombinedFieldQuery",
            ExtendedQuery::NGramPhrase(_) => "NGramPhraseQuery",
            ExtendedQuery::MultiTerm(q) => q.source.name(),
            ExtendedQuery::Blended(_) => "BlendedTermQuery",
            ExtendedQuery::IndriAnd(_) => "IndriAndQuery",
            ExtendedQuery::LogOddsFusion(_) => "LogOddsFusionQuery",
            ExtendedQuery::BayesianScore(_) => "BayesianScoreQuery",
            ExtendedQuery::DocAndScore(_) => "DocAndScoreQuery",
            ExtendedQuery::NumericDocValuesRange(_) => "SortedNumericDocValuesRangeQuery",
            ExtendedQuery::IndexSortRange(_) => "IndexSortSortedNumericDocValuesRangeQuery",
            ExtendedQuery::PointRange(_) => "PointRangeQuery",
            ExtendedQuery::PointInSet(_) => "PointInSetQuery",
            ExtendedQuery::IndexOrDocValues(_) => "IndexOrDocValuesQuery",
            ExtendedQuery::RescoreTopN(_) => "RescoreTopNQuery",
            ExtendedQuery::Document(_) => "DocumentQuery",
            ExtendedQuery::ToParentBlockJoin(_) => "ToParentBlockJoinQuery",
            ExtendedQuery::ToChildBlockJoin(_) => "ToChildBlockJoinQuery",
            ExtendedQuery::ParentChildrenBlockJoin(_) => "ParentChildrenBlockJoinQuery",
            ExtendedQuery::ParentsChildrenBlockJoin(_) => "ParentsChildrenBlockJoinQuery",
            ExtendedQuery::TermsIncludingScore(_) => "TermsIncludingScoreQuery",
            ExtendedQuery::GlobalOrdinals(_) => "GlobalOrdinalsQuery",
            ExtendedQuery::GlobalOrdinalsWithScore(_) => "GlobalOrdinalsWithScoreQuery",
            ExtendedQuery::PointInSetIncludingScore(_) => "PointInSetIncludingScoreQuery",
            ExtendedQuery::Function(_) => "FunctionQuery",
            ExtendedQuery::FunctionRange(_) => "FunctionRangeQuery",
            ExtendedQuery::FunctionMatch(_) => "FunctionMatchQuery",
            ExtendedQuery::Interval(_) => "IntervalQuery",
            ExtendedQuery::FunctionScore(_) => "FunctionScoreQuery",
        }
    }

    /// The sub-clauses a wrapper query holds, for walks that recurse.
    pub fn children(&self) -> Vec<&Clause> {
        match self {
            ExtendedQuery::IndriAnd(q) => q.clauses.iter().collect(),
            ExtendedQuery::LogOddsFusion(q) => q.clauses.iter().collect(),
            ExtendedQuery::BayesianScore(q) => vec![q.query.as_ref()],
            ExtendedQuery::IndexSortRange(q) => vec![q.fallback.as_ref()],
            ExtendedQuery::IndexOrDocValues(q) => vec![q.index_query.as_ref(), q.dv_query.as_ref()],
            ExtendedQuery::ToParentBlockJoin(q) => vec![q.child.as_ref()],
            ExtendedQuery::ToChildBlockJoin(q) => vec![q.parent.as_ref()],
            ExtendedQuery::ParentChildrenBlockJoin(q) => vec![q.child.as_ref()],
            ExtendedQuery::ParentsChildrenBlockJoin(q) => vec![q.parent.as_ref(), q.child.as_ref()],
            // The to-query runs inside the global-ordinals joins' weights.
            ExtendedQuery::GlobalOrdinals(q) => vec![q.to_query.as_ref()],
            ExtendedQuery::GlobalOrdinalsWithScore(q) => vec![q.to_query.as_ref()],
            ExtendedQuery::FunctionScore(q) => vec![q.in_query.as_ref()],
            _ => Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// SynonymQuery, CombinedFieldQuery
// ---------------------------------------------------------------------------

/// `SynonymQuery`: terms of one field scored as if they were one term -- the
/// document's frequency is the sum of each present term's `boost * freq`,
/// `docFreq` the largest of theirs, `totalTermFreq` the sum.
#[derive(Debug, Clone, PartialEq)]
pub struct SynonymQuery {
    pub field: String,
    /// `(term, boost)`, sorted by term as `Builder.build` sorts them.
    pub terms: Vec<(Vec<u8>, f32)>,
}

impl SynonymQuery {
    /// `SynonymQuery.Builder`: every boost must be in `(0, 1]`, and there may
    /// be at most [`MAX_CLAUSE_COUNT`] terms.
    pub fn new(
        field: impl Into<String>,
        terms: impl IntoIterator<Item = (impl Into<Vec<u8>>, f32)>,
    ) -> Result<Self> {
        let mut out: Vec<(Vec<u8>, f32)> = Vec::new();
        for (t, boost) in terms {
            if boost.is_nan() || boost <= 0.0 || boost > 1.0 {
                return Err(Error::InvalidQuery(
                    "boost must be a positive float between 0 (exclusive) and 1 (inclusive)".into(),
                ));
            }
            out.push((t.into(), boost));
            if out.len() > MAX_CLAUSE_COUNT {
                return Err(Error::InvalidQuery("too many clauses".into()));
            }
        }
        // `terms.sort(Comparator.comparing(a -> a.term))`: a stable sort.
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(Self {
            field: field.into(),
            terms: out,
        })
    }
}

/// `CombinedFieldQuery` (BM25F): one term searched across several fields as
/// if they were one, each field's frequency and length scaled by its weight.
#[derive(Debug, Clone, PartialEq)]
pub struct CombinedFieldQuery {
    pub term: Vec<u8>,
    /// `(field, weight)`, sorted by field (`TreeMap`), one entry per field.
    pub fields: Vec<(String, f32)>,
}

impl CombinedFieldQuery {
    /// `CombinedFieldQuery.Builder`: every weight must be at least `1`; a
    /// field added twice keeps its last weight (`HashMap.put`).
    pub fn new(
        term: impl Into<Vec<u8>>,
        fields: impl IntoIterator<Item = (impl Into<String>, f32)>,
    ) -> Result<Self> {
        let mut map = std::collections::BTreeMap::new();
        for (f, w) in fields {
            // `weight < 1` rejects too; a NaN weight passes Java's check.
            if w < 1.0 {
                return Err(Error::InvalidQuery(
                    "weight must be greater or equal to 1".into(),
                ));
            }
            map.insert(f.into(), w);
        }
        if map.len() > MAX_CLAUSE_COUNT {
            return Err(Error::InvalidQuery("too many clauses".into()));
        }
        Ok(Self {
            term: term.into(),
            fields: map.into_iter().collect(),
        })
    }
}

/// `NGramPhraseQuery`: a phrase over n-gram tokens, rewritten to keep only
/// every `n`th gram (and the last) when the phrase is exact and contiguous.
#[derive(Debug, Clone, PartialEq)]
pub struct NGramPhraseQuery {
    pub n: usize,
    pub phrase: PhraseQuery,
}

impl NGramPhraseQuery {
    pub fn new(n: usize, phrase: PhraseQuery) -> Self {
        Self { n, phrase }
    }

    /// `NGramPhraseQuery.rewrite`.
    pub fn rewrite(&self) -> PhraseQuery {
        let terms = &self.phrase.terms;
        let positions = self.phrase.positions();
        let mut optimizable = self.phrase.slop == 0 && self.n >= 2 && terms.len() >= 3;
        if optimizable {
            optimizable = positions
                .windows(2)
                .all(|w| i64::from(w[1]) == i64::from(w[0]) + 1);
        }
        if !optimizable {
            return self.phrase.clone();
        }
        let last = terms.len() - 1;
        let mut out_terms = Vec::new();
        let mut out_positions = Vec::new();
        for (i, t) in terms.iter().enumerate() {
            if i % self.n == 0 || i == last {
                out_terms.push(t.clone());
                // Builder.add(term, i): positions are the index, whatever the
                // original positions were.
                out_positions.push(i32::try_from(i).unwrap_or(i32::MAX));
            }
        }
        PhraseQuery {
            field: self.phrase.field.clone(),
            terms: out_terms,
            slop: 0,
            positions: out_positions,
        }
    }
}

// ---------------------------------------------------------------------------
// MultiTermQuery rewrite methods
// ---------------------------------------------------------------------------

/// `MultiTermQuery.RewriteMethod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RewriteMethod {
    /// `CONSTANT_SCORE_BLENDED_REWRITE`, the default.
    #[default]
    ConstantScoreBlended,
    /// `CONSTANT_SCORE_REWRITE` (`MultiTermQueryConstantScoreWrapper`).
    ConstantScore,
    /// `SCORING_BOOLEAN_REWRITE`: a `BooleanQuery` of scoring terms.
    ScoringBoolean,
    /// `CONSTANT_SCORE_BOOLEAN_REWRITE`: the same boolean, constant-scored.
    ConstantScoreBoolean,
    /// `TopTermsScoringBooleanQueryRewrite(size)`.
    TopTermsScoringBoolean(usize),
    /// `TopTermsBoostOnlyBooleanQueryRewrite(size)`.
    TopTermsBoostOnlyBoolean(usize),
    /// `TopTermsBlendedFreqScoringRewrite(size)`.
    TopTermsBlendedFreqScoring(usize),
    /// `DOC_VALUES_REWRITE` (`DocValuesRewriteMethod`): matched against the
    /// field's `SORTED_SET` doc values instead of its terms.
    DocValues,
}

/// `TermRangeQuery`: the terms between two bounds, either end open.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermRangeQuery {
    pub field: String,
    pub lower: Option<Vec<u8>>,
    pub upper: Option<Vec<u8>>,
    pub include_lower: bool,
    pub include_upper: bool,
}

impl TermRangeQuery {
    pub fn new(
        field: impl Into<String>,
        lower: Option<Vec<u8>>,
        upper: Option<Vec<u8>>,
        include_lower: bool,
        include_upper: bool,
    ) -> Self {
        Self {
            field: field.into(),
            lower,
            upper,
            include_lower,
            include_upper,
        }
    }

    /// Whether `term` is inside the range (`Automata.makeBinaryInterval`'s
    /// language).
    pub fn accepts(&self, term: &[u8]) -> bool {
        let above = match &self.lower {
            None => true,
            Some(lo) if self.include_lower => term >= lo.as_slice(),
            Some(lo) => term > lo.as_slice(),
        };
        let below = match &self.upper {
            None => true,
            Some(hi) if self.include_upper => term <= hi.as_slice(),
            Some(hi) => term < hi.as_slice(),
        };
        above && below
    }
}

/// `AutomatonQuery`: every term an automaton accepts. `binary` is
/// `isBinary`: the automaton is over bytes rather than code points.
#[derive(Debug, Clone)]
pub struct AutomatonQuery {
    pub field: String,
    pub automaton: Arc<Automaton>,
    pub binary: bool,
}

impl AutomatonQuery {
    pub fn new(field: impl Into<String>, automaton: Automaton, binary: bool) -> Self {
        Self {
            field: field.into(),
            automaton: Arc::new(automaton),
            binary,
        }
    }
}

impl PartialEq for AutomatonQuery {
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field
            && self.binary == other.binary
            && *self.automaton == *other.automaton
    }
}

/// Which terms a [`MultiTermQuery`] enumerates.
#[derive(Debug, Clone, PartialEq)]
pub enum MultiTermSource {
    Prefix(PrefixQuery),
    Wildcard(WildcardQuery),
    Regexp(RegexpQuery),
    TermRange(TermRangeQuery),
    Automaton(AutomatonQuery),
    /// `lucene-join`'s `TermsQuery`: a sorted set of terms, found by seeking
    /// (`SeekingTermSetTermsEnum`).
    TermSet(crate::join::TermSetSource),
}

impl MultiTermSource {
    pub fn field(&self) -> &str {
        match self {
            MultiTermSource::Prefix(q) => &q.field,
            MultiTermSource::Wildcard(q) => &q.field,
            MultiTermSource::Regexp(q) => &q.field,
            MultiTermSource::TermRange(q) => &q.field,
            MultiTermSource::Automaton(q) => &q.field,
            MultiTermSource::TermSet(q) => &q.field,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            MultiTermSource::Prefix(_) => "PrefixQuery",
            MultiTermSource::Wildcard(_) => "WildcardQuery",
            MultiTermSource::Regexp(_) => "RegexpQuery",
            MultiTermSource::TermRange(_) => "TermRangeQuery",
            MultiTermSource::Automaton(_) => "AutomatonQuery",
            MultiTermSource::TermSet(_) => "TermsQuery",
        }
    }
}

/// `MultiTermQuery`: a term enumeration and the `RewriteMethod` that turns
/// its terms into a query.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiTermQuery {
    pub source: MultiTermSource,
    pub rewrite: RewriteMethod,
}

impl MultiTermQuery {
    pub fn new(source: MultiTermSource, rewrite: RewriteMethod) -> Self {
        Self { source, rewrite }
    }

    pub fn field(&self) -> &str {
        self.source.field()
    }
}

// ---------------------------------------------------------------------------
// BlendedTermQuery
// ---------------------------------------------------------------------------

/// `BlendedTermQuery.RewriteMethod`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendedRewrite {
    /// `BOOLEAN_REWRITE`: the blended terms as `SHOULD` clauses.
    Boolean,
    /// `DisjunctionMaxRewrite(tieBreakerMultiplier)`.
    DisjunctionMax(f32),
}

impl Default for BlendedRewrite {
    /// `DISJUNCTION_MAX_REWRITE`: `DisjunctionMaxRewrite(0.01f)`.
    fn default() -> Self {
        BlendedRewrite::DisjunctionMax(0.01)
    }
}

/// `BlendedTermQuery`: terms (of any fields) that all score with the same
/// `docFreq` -- the largest of theirs -- and the sum of their
/// `totalTermFreq`s.
#[derive(Debug, Clone, PartialEq)]
pub struct BlendedTermQuery {
    /// `(field, term, boost)`, sorted by `Term.compareTo` (field, then
    /// bytes), stably.
    pub terms: Vec<(String, Vec<u8>, f32)>,
    pub rewrite: BlendedRewrite,
}

impl BlendedTermQuery {
    pub fn new(
        terms: impl IntoIterator<Item = (impl Into<String>, impl Into<Vec<u8>>, f32)>,
        rewrite: BlendedRewrite,
    ) -> Result<Self> {
        let mut out: Vec<(String, Vec<u8>, f32)> = terms
            .into_iter()
            .map(|(f, t, b)| (f.into(), t.into(), b))
            .collect();
        if out.len() > MAX_CLAUSE_COUNT {
            return Err(Error::InvalidQuery("too many clauses".into()));
        }
        // `InPlaceMergeSorter`: stable.
        out.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
        Ok(Self {
            terms: out,
            rewrite,
        })
    }
}

// ---------------------------------------------------------------------------
// Indri, fusion, calibration
// ---------------------------------------------------------------------------

/// `IndriAndQuery`: every clause's documents, scored by `IndriAndScorer`.
/// The clauses' occurs are not read by the scorer (every clause is part of
/// the disjunction), so they are not kept.
#[derive(Debug, Clone, PartialEq)]
pub struct IndriAndQuery {
    pub clauses: Vec<Clause>,
}

impl IndriAndQuery {
    pub fn new(clauses: impl IntoIterator<Item = impl Into<Clause>>) -> Self {
        Self {
            clauses: clauses.into_iter().map(Into::into).collect(),
        }
    }
}

/// `LogOddsFusionQuery`: a disjunction whose score is the sigmoid of the
/// (weighted) mean of its clauses' gated log-odds, scaled by `n^alpha`.
#[derive(Debug, Clone, PartialEq)]
pub struct LogOddsFusionQuery {
    pub clauses: Vec<Clause>,
    pub alpha: f32,
    pub weights: Option<Vec<f32>>,
    /// `logitMin`/`logitMax`: both or neither.
    pub logit_bounds: Option<(Vec<f32>, Vec<f32>)>,
}

impl LogOddsFusionQuery {
    /// The constructor's checks: `alpha` in `[0, 1]`; weights, when given,
    /// one per clause, finite, non-negative, summing to `1` within `1e-3`;
    /// logit bounds, when given, one pair per clause.
    pub fn new(
        clauses: impl IntoIterator<Item = impl Into<Clause>>,
        alpha: f32,
        weights: Option<Vec<f32>>,
        logit_bounds: Option<(Vec<f32>, Vec<f32>)>,
    ) -> Result<Self> {
        let clauses: Vec<Clause> = clauses.into_iter().map(Into::into).collect();
        if alpha.is_nan() || !(0.0..=1.0).contains(&alpha) {
            return Err(Error::InvalidQuery(format!(
                "alpha must be in [0, 1], got {alpha}"
            )));
        }
        if let Some(w) = &weights {
            if w.len() != clauses.len() {
                return Err(Error::InvalidQuery(format!(
                    "weights length {} must equal clauses size {}",
                    w.len(),
                    clauses.len()
                )));
            }
            let mut sum = 0.0f32;
            for &x in w {
                if !x.is_finite() || x < 0.0 {
                    return Err(Error::InvalidQuery(format!(
                        "weights must be non-negative and finite, got {x}"
                    )));
                }
                sum += x;
            }
            if (sum - 1.0).abs() > 1e-3 {
                return Err(Error::InvalidQuery(format!(
                    "weights must sum to 1.0, got {sum}"
                )));
            }
        }
        if let Some((min, max)) = &logit_bounds {
            if min.len() != clauses.len() || max.len() != clauses.len() {
                return Err(Error::InvalidQuery(format!(
                    "logit bounds must have one entry per clause ({})",
                    clauses.len()
                )));
            }
        }
        Ok(Self {
            clauses,
            alpha,
            weights,
            logit_bounds,
        })
    }
}

/// `BayesianScoreQuery`: `sigmoid(alpha * (score - beta) + logit(baseRate))`
/// of the wrapped query's score.
#[derive(Debug, Clone, PartialEq)]
pub struct BayesianScoreQuery {
    pub query: Box<Clause>,
    pub alpha: f32,
    pub beta: f32,
    pub base_rate: f32,
}

impl BayesianScoreQuery {
    /// The constructor's checks: `alpha` positive and finite, `beta` finite,
    /// `baseRate` in `[0, 1)`.
    // `Range::contains` would reject a NaN `baseRate`, which Java's two
    // comparisons let through.
    #[allow(clippy::manual_range_contains)]
    pub fn new(query: impl Into<Clause>, alpha: f32, beta: f32, base_rate: f32) -> Result<Self> {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(Error::InvalidQuery(format!(
                "alpha must be a positive finite value, got {alpha}"
            )));
        }
        if !beta.is_finite() {
            return Err(Error::InvalidQuery(format!(
                "beta must be a finite value, got {beta}"
            )));
        }
        // `baseRate < 0 || baseRate >= 1`: a NaN passes, as in Java.
        if base_rate < 0.0 || base_rate >= 1.0 {
            return Err(Error::InvalidQuery(format!(
                "baseRate must be in [0, 1), got {base_rate}"
            )));
        }
        Ok(Self {
            query: Box::new(query.into()),
            alpha,
            beta,
            base_rate,
        })
    }

    /// `logitBaseRate`: `(float) Math.log(baseRate / (1.0 - baseRate))`, `0`
    /// without a base rate.
    pub fn logit_base_rate(&self) -> f32 {
        if self.base_rate > 0.0 {
            let b = f64::from(self.base_rate);
            (b / (1.0 - b)).ln() as f32
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// DocAndScoreQuery
// ---------------------------------------------------------------------------

/// `DocAndScoreQuery`: fixed documents with fixed scores -- what
/// `AbstractKnnVectorQuery.rewrite` turns a KNN search into, so the result
/// can sit in a boolean like any other clause.
#[derive(Debug, Clone, PartialEq)]
pub struct DocAndScoreQuery {
    /// Global doc ids, ascending.
    pub docs: Vec<i32>,
    /// `scores[i]` is `docs[i]`'s.
    pub scores: Vec<f32>,
    /// `segmentStarts`: the index in `docs` where each segment's documents
    /// start, one entry per segment plus a final `docs.len()`.
    pub segment_starts: Vec<usize>,
    /// Each segment's `docBase`, parallel to `segment_starts` minus its last
    /// entry.
    pub doc_bases: Vec<i32>,
}

impl DocAndScoreQuery {
    /// `createRewrittenQuery`: `hits` in any order, as `(global doc, score)`,
    /// grouped by the segments whose `docBase`s are `doc_bases` (ascending).
    pub fn new(mut hits: Vec<(i32, f32)>, doc_bases: &[i32]) -> Self {
        hits.sort_by_key(|&(d, _)| d);
        let docs: Vec<i32> = hits.iter().map(|&(d, _)| d).collect();
        let scores: Vec<f32> = hits.iter().map(|&(_, s)| s).collect();
        let mut segment_starts = Vec::with_capacity(doc_bases.len() + 1);
        for &base in doc_bases {
            segment_starts.push(docs.partition_point(|&d| d < base));
        }
        segment_starts.push(docs.len());
        Self {
            docs,
            scores,
            segment_starts,
            doc_bases: doc_bases.to_vec(),
        }
    }

    /// The segment whose `docBase` is `doc_base`: its documents (local ids)
    /// and scores.
    pub fn segment(&self, doc_base: i32) -> Option<(Vec<i32>, &[f32])> {
        let i = self.doc_bases.iter().position(|&b| b == doc_base)?;
        let (lo, hi) = (self.segment_starts[i], self.segment_starts[i + 1]);
        let docs = self.docs[lo..hi].iter().map(|&d| d - doc_base).collect();
        Some((docs, &self.scores[lo..hi]))
    }
}

// ---------------------------------------------------------------------------
// Doc-values and index-sort ranges
// ---------------------------------------------------------------------------

/// `SortedNumericDocValuesRangeQuery` (the concrete class behind
/// `NumericDocValuesField.newSlowRangeQuery` and
/// `SortedNumericDocValuesField.newSlowRangeQuery`, whose base is
/// `NumericDocValuesRangeQuery`): documents with a value in
/// `[lower, upper]`, from the field's doc values, constant-scored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumericDocValuesRangeQuery {
    pub field: String,
    pub lower: i64,
    pub upper: i64,
}

impl NumericDocValuesRangeQuery {
    pub fn new(field: impl Into<String>, lower: i64, upper: i64) -> Self {
        Self {
            field: field.into(),
            lower,
            upper,
        }
    }
}

/// `IndexSortSortedNumericDocValuesRangeQuery`: a range over the field the
/// segment is sorted by, answered by two binary searches over the doc
/// values; any segment it cannot take runs `fallback`.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexSortSortedNumericDocValuesRangeQuery {
    pub field: String,
    pub lower: i64,
    pub upper: i64,
    pub fallback: Box<Clause>,
}

impl IndexSortSortedNumericDocValuesRangeQuery {
    pub fn new(
        field: impl Into<String>,
        lower: i64,
        upper: i64,
        fallback: impl Into<Clause>,
    ) -> Self {
        Self {
            field: field.into(),
            lower,
            upper,
            fallback: Box::new(fallback.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Points
// ---------------------------------------------------------------------------

/// `PointRangeQuery` over any number of dimensions and any width: the
/// packed lower and upper corners, inclusive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointRangeQuery {
    pub field: String,
    pub num_dims: usize,
    pub bytes_per_dim: usize,
    pub lower: Vec<u8>,
    pub upper: Vec<u8>,
}

impl PointRangeQuery {
    /// `new PointRangeQuery(field, lowerPoint, upperPoint, numDims)`, with
    /// `PointRangeQuery.checkArgs`' checks and messages: the corners must be
    /// `num_dims * bytes_per_dim` bytes each.
    pub fn new(
        field: impl Into<String>,
        num_dims: usize,
        lower: Vec<u8>,
        upper: Vec<u8>,
    ) -> Result<Self> {
        if num_dims == 0 {
            return Err(Error::InvalidQuery(
                "numDims must be positive, got 0".into(),
            ));
        }
        if lower.is_empty() {
            return Err(Error::InvalidQuery("lowerPoint has length of zero".into()));
        }
        if !lower.len().is_multiple_of(num_dims) {
            return Err(Error::InvalidQuery(
                "lowerPoint is not a fixed multiple of numDims".into(),
            ));
        }
        if lower.len() != upper.len() {
            return Err(Error::InvalidQuery(format!(
                "lowerPoint has length={} but upperPoint has different length={}",
                lower.len(),
                upper.len()
            )));
        }
        Ok(Self {
            field: field.into(),
            num_dims,
            bytes_per_dim: lower.len() / num_dims,
            lower,
            upper,
        })
    }

    /// `IntPoint.newRangeQuery(field, int[], int[])`.
    pub fn int_range(field: impl Into<String>, lower: &[i32], upper: &[i32]) -> Result<Self> {
        let pack = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|&x| int_bytes(x)).collect() };
        Self::new(field, lower.len(), pack(lower), pack(upper))
    }

    /// `LongPoint.newRangeQuery(field, long[], long[])`.
    pub fn long_range(field: impl Into<String>, lower: &[i64], upper: &[i64]) -> Result<Self> {
        let pack = |v: &[i64]| -> Vec<u8> { v.iter().flat_map(|&x| long_bytes(x)).collect() };
        Self::new(field, lower.len(), pack(lower), pack(upper))
    }

    /// `InetAddressPoint.newRangeQuery`: IPv4 addresses as their IPv4-mapped
    /// IPv6 form, 16 bytes.
    pub fn inet_range(
        field: impl Into<String>,
        lower: std::net::IpAddr,
        upper: std::net::IpAddr,
    ) -> Result<Self> {
        Self::new(
            field,
            1,
            inet_bytes(lower).to_vec(),
            inet_bytes(upper).to_vec(),
        )
    }
}

/// `PointInSetQuery`: documents with a point equal to one of a set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointInSetQuery {
    pub field: String,
    pub num_dims: usize,
    pub bytes_per_dim: usize,
    /// Packed points, sorted and deduplicated (`PointInSetQuery`'s
    /// `PrefixCodedTerms` of the sorted set).
    pub points: Vec<Vec<u8>>,
}

impl PointInSetQuery {
    /// `new PointInSetQuery(field, numDims, bytesPerDim, packedPoints)`, with
    /// its checks and messages.
    pub fn new(
        field: impl Into<String>,
        num_dims: usize,
        bytes_per_dim: usize,
        points: impl IntoIterator<Item = Vec<u8>>,
    ) -> Result<Self> {
        if !(1..=16).contains(&bytes_per_dim) {
            return Err(Error::InvalidQuery(format!(
                "bytesPerDim must be > 0 and <= 16; got {bytes_per_dim}"
            )));
        }
        if !(1..=8).contains(&num_dims) {
            return Err(Error::InvalidQuery(format!(
                "numDims must be > 0 and <= 8; got {num_dims}"
            )));
        }
        let mut points: Vec<Vec<u8>> = points.into_iter().collect();
        let want = num_dims.saturating_mul(bytes_per_dim);
        if let Some(bad) = points.iter().find(|p| p.len() != want) {
            return Err(Error::InvalidQuery(format!(
                "packed point length should be {want} but got {}",
                bad.len()
            )));
        }
        points.sort();
        points.dedup();
        Ok(Self {
            field: field.into(),
            num_dims,
            bytes_per_dim,
            points,
        })
    }

    /// `IntPoint.newSetQuery`.
    pub fn int_set(field: impl Into<String>, values: &[i32]) -> Result<Self> {
        Self::new(field, 1, 4, values.iter().map(|&v| int_bytes(v).to_vec()))
    }

    /// `LongPoint.newSetQuery`.
    pub fn long_set(field: impl Into<String>, values: &[i64]) -> Result<Self> {
        Self::new(field, 1, 8, values.iter().map(|&v| long_bytes(v).to_vec()))
    }

    /// `InetAddressPoint.newSetQuery`.
    pub fn inet_set(field: impl Into<String>, values: &[std::net::IpAddr]) -> Result<Self> {
        Self::new(field, 1, 16, values.iter().map(|&v| inet_bytes(v).to_vec()))
    }
}

/// `IndexOrDocValuesQuery`: one set of matches two ways -- `index_query`
/// (points or terms: costly to set up, a good lead iterator) and `dv_query`
/// (doc values: cheap to start, good at verifying documents another clause
/// leads). Both must match the same documents with the same constant score.
///
/// Per segment, as `IndexOrDocValuesQuery.createWeight`'s `ScorerSupplier`
/// decides: nothing when either side has no scorer supplier (the segment lacks
/// the points or the doc values); run alone (`bulkScorer`) the index side;
/// inside a boolean, [`crate::doc_value_query::plan_index_or_doc_values`] over
/// the index side's `cost()` and the boolean's lead cost
/// (`ScorerSupplier.get(leadCost)`).
#[derive(Debug, Clone, PartialEq)]
pub struct IndexOrDocValuesQuery {
    pub index_query: Box<Clause>,
    pub dv_query: Box<Clause>,
}

impl IndexOrDocValuesQuery {
    /// `new IndexOrDocValuesQuery(indexQuery, dvQuery)`.
    pub fn new(index_query: impl Into<Clause>, dv_query: impl Into<Clause>) -> Self {
        Self {
            index_query: Box::new(index_query.into()),
            dv_query: Box::new(dv_query.into()),
        }
    }
}

/// `NumericUtils.intToSortableBytes`.
pub fn int_bytes(v: i32) -> [u8; 4] {
    ((v as u32) ^ 0x8000_0000).to_be_bytes()
}

/// `NumericUtils.longToSortableBytes`.
pub fn long_bytes(v: i64) -> [u8; 8] {
    ((v as u64) ^ 0x8000_0000_0000_0000).to_be_bytes()
}

/// `InetAddressPoint.encode`: IPv4 as IPv4-mapped IPv6.
pub fn inet_bytes(v: std::net::IpAddr) -> [u8; 16] {
    match v {
        std::net::IpAddr::V4(a) => a.to_ipv6_mapped().octets(),
        std::net::IpAddr::V6(a) => a.octets(),
    }
}

// ---------------------------------------------------------------------------
// The document package's queries
// ---------------------------------------------------------------------------

/// A [`crate::document::DocumentQuery`] -- `LatLonPointDistanceQuery`,
/// `LatLonPointQuery`, `LatLonShapeQuery` and the rest of the geo package,
/// each a `ConstantScoreWeight` -- as a leaf of the scorer tree: per segment
/// its matches before deletions (the tree applies them, as Lucene's bulk
/// scorer does), every one scored the boost.
///
/// `key` is the query's identity, which Java's `Query.equals` is: two
/// clauses are equal exactly when their keys are, and the key is what
/// [`std::fmt::Debug`] prints, so the query cache (which keys a clause by its
/// debug form) tells two geo queries apart by it. The caller builds it from
/// everything the query depends on -- the FFI decoder uses the query's own
/// wire bytes.
#[derive(Clone)]
pub struct DocumentClause {
    pub key: String,
    pub query: Arc<dyn crate::document::DocumentQuery>,
}

impl DocumentClause {
    /// `query`, identified by `key`.
    pub fn new(key: impl Into<String>, query: Arc<dyn crate::document::DocumentQuery>) -> Self {
        Self {
            key: key.into(),
            query,
        }
    }
}

impl std::fmt::Debug for DocumentClause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DocumentClause({})", self.key)
    }
}

impl PartialEq for DocumentClause {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::TermQuery;

    fn t(w: &str) -> Clause {
        Clause::Term(TermQuery::new("body", w.as_bytes().to_vec()))
    }

    fn invalid<T: std::fmt::Debug>(r: Result<T>) -> String {
        match r {
            Err(Error::InvalidQuery(m)) => m,
            other => panic!("accepted: {other:?}"),
        }
    }

    /// The builders' argument checks, with Java's messages.
    #[test]
    fn builders_refuse_what_javas_refuse() {
        for boost in [0.0, -1.0, 1.5, f32::NAN] {
            let m = invalid(SynonymQuery::new("f", [(b"a".to_vec(), boost)]));
            assert!(
                m.starts_with("boost must be a positive float"),
                "{boost}: {m}"
            );
        }
        let many: Vec<(Vec<u8>, f32)> = (0..=MAX_CLAUSE_COUNT)
            .map(|i| (i.to_string().into_bytes(), 1.0))
            .collect();
        assert_eq!(invalid(SynonymQuery::new("f", many)), "too many clauses");

        assert_eq!(
            invalid(CombinedFieldQuery::new(b"t".to_vec(), [("a", 0.5)])),
            "weight must be greater or equal to 1"
        );
        let fields: Vec<(String, f32)> = (0..=MAX_CLAUSE_COUNT)
            .map(|i| (format!("f{i}"), 1.0))
            .collect();
        assert_eq!(
            invalid(CombinedFieldQuery::new(b"t".to_vec(), fields)),
            "too many clauses"
        );
        // A field added twice keeps its last weight.
        let c =
            CombinedFieldQuery::new(b"t".to_vec(), [("b", 2.0), ("a", 1.0), ("b", 3.0)]).unwrap();
        assert_eq!(c.fields, [("a".to_string(), 1.0), ("b".to_string(), 3.0)]);

        let terms: Vec<(String, Vec<u8>, f32)> = (0..=MAX_CLAUSE_COUNT)
            .map(|i| ("f".to_string(), i.to_string().into_bytes(), 1.0))
            .collect();
        assert_eq!(
            invalid(BlendedTermQuery::new(terms, BlendedRewrite::default())),
            "too many clauses"
        );
        assert_eq!(
            BlendedRewrite::default(),
            BlendedRewrite::DisjunctionMax(0.01)
        );

        let two = || [t("a"), t("b")];
        for alpha in [-0.1, 1.1, f32::NAN] {
            let m = invalid(LogOddsFusionQuery::new(two(), alpha, None, None));
            assert!(m.starts_with("alpha must be in [0, 1]"), "{m}");
        }
        assert_eq!(
            invalid(LogOddsFusionQuery::new(two(), 0.5, Some(vec![1.0]), None)),
            "weights length 1 must equal clauses size 2"
        );
        for w in [vec![-0.5, 1.5], vec![f32::INFINITY, 0.0]] {
            let m = invalid(LogOddsFusionQuery::new(two(), 0.5, Some(w), None));
            assert!(
                m.starts_with("weights must be non-negative and finite"),
                "{m}"
            );
        }
        assert_eq!(
            invalid(LogOddsFusionQuery::new(
                two(),
                0.5,
                Some(vec![0.5, 0.25]),
                None
            )),
            "weights must sum to 1.0, got 0.75"
        );
        assert_eq!(
            invalid(LogOddsFusionQuery::new(
                two(),
                0.5,
                None,
                Some((vec![0.0], vec![1.0, 1.0]))
            )),
            "logit bounds must have one entry per clause (2)"
        );
        assert!(LogOddsFusionQuery::new(two(), 0.5, Some(vec![0.5, 0.5]), None).is_ok());

        for alpha in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            let m = invalid(BayesianScoreQuery::new(t("a"), alpha, 0.0, 0.5));
            assert!(
                m.starts_with("alpha must be a positive finite value"),
                "{m}"
            );
        }
        let m = invalid(BayesianScoreQuery::new(t("a"), 1.0, f32::NAN, 0.5));
        assert!(m.starts_with("beta must be a finite value"), "{m}");
        for base_rate in [-0.1, 1.0] {
            let m = invalid(BayesianScoreQuery::new(t("a"), 1.0, 0.0, base_rate));
            assert!(m.starts_with("baseRate must be in [0, 1)"), "{m}");
        }
        // `baseRate < 0 || baseRate >= 1`: a NaN passes Java's check too.
        assert!(BayesianScoreQuery::new(t("a"), 1.0, 0.0, f32::NAN).is_ok());
    }

    /// `getClass().getSimpleName()` of each query, as explanations and
    /// errors name it.
    #[test]
    fn every_query_names_its_java_class() {
        let multi = |s: MultiTermSource| {
            ExtendedQuery::MultiTerm(MultiTermQuery::new(s, RewriteMethod::default()))
        };
        let cases: Vec<(ExtendedQuery, &str)> = vec![
            (
                ExtendedQuery::CombinedField(
                    CombinedFieldQuery::new(b"t".to_vec(), [("a", 1.0)]).unwrap(),
                ),
                "CombinedFieldQuery",
            ),
            (
                ExtendedQuery::NGramPhrase(NGramPhraseQuery::new(
                    2,
                    PhraseQuery::new("f", [b"ab".to_vec()]),
                )),
                "NGramPhraseQuery",
            ),
            (
                multi(MultiTermSource::Prefix(PrefixQuery::new(
                    "f",
                    b"a".to_vec(),
                ))),
                "PrefixQuery",
            ),
            (
                multi(MultiTermSource::Wildcard(WildcardQuery::new(
                    "f",
                    b"a*".to_vec(),
                ))),
                "WildcardQuery",
            ),
            (
                multi(MultiTermSource::Regexp(RegexpQuery::new("f", "a.*"))),
                "RegexpQuery",
            ),
            (
                multi(MultiTermSource::TermRange(TermRangeQuery::new(
                    "f", None, None, true, true,
                ))),
                "TermRangeQuery",
            ),
            (
                multi(MultiTermSource::Automaton(AutomatonQuery::new(
                    "f",
                    lucene_util::automaton::automata::make_string("ab"),
                    false,
                ))),
                "AutomatonQuery",
            ),
            (
                ExtendedQuery::Blended(
                    BlendedTermQuery::new([("f", b"a".to_vec(), 1.0)], BlendedRewrite::Boolean)
                        .unwrap(),
                ),
                "BlendedTermQuery",
            ),
            (
                ExtendedQuery::IndriAnd(IndriAndQuery::new([t("a")])),
                "IndriAndQuery",
            ),
            (
                ExtendedQuery::LogOddsFusion(
                    LogOddsFusionQuery::new([t("a")], 0.5, None, None).unwrap(),
                ),
                "LogOddsFusionQuery",
            ),
            (
                ExtendedQuery::BayesianScore(
                    BayesianScoreQuery::new(t("a"), 1.0, 0.0, 0.5).unwrap(),
                ),
                "BayesianScoreQuery",
            ),
            (
                ExtendedQuery::NumericDocValuesRange(NumericDocValuesRangeQuery::new("n", 1, 2)),
                "SortedNumericDocValuesRangeQuery",
            ),
            (
                ExtendedQuery::IndexSortRange(IndexSortSortedNumericDocValuesRangeQuery::new(
                    "n",
                    1,
                    2,
                    t("a"),
                )),
                "IndexSortSortedNumericDocValuesRangeQuery",
            ),
            (
                ExtendedQuery::PointRange(PointRangeQuery::int_range("p", &[1], &[2]).unwrap()),
                "PointRangeQuery",
            ),
            (
                ExtendedQuery::PointInSet(PointInSetQuery::int_set("p", &[1, 2]).unwrap()),
                "PointInSetQuery",
            ),
            (
                ExtendedQuery::IndexOrDocValues(IndexOrDocValuesQuery::new(t("a"), t("b"))),
                "IndexOrDocValuesQuery",
            ),
        ];
        for (q, name) in cases {
            assert_eq!(q.name(), name, "{q:?}");
        }
    }

    /// An automaton query is equal by field, binary flag and language.
    #[test]
    fn automaton_queries_compare_by_field_and_automaton() {
        use lucene_util::automaton::automata::make_string;
        let a = AutomatonQuery::new("f", make_string("ab"), false);
        assert_eq!(a, AutomatonQuery::new("f", make_string("ab"), false));
        assert_ne!(a, AutomatonQuery::new("g", make_string("ab"), false));
        assert_ne!(a, AutomatonQuery::new("f", make_string("ab"), true));
        assert_ne!(a, AutomatonQuery::new("f", make_string("ac"), false));
    }

    /// `InetAddressPoint.encode`: an IPv4 address as its IPv4-mapped IPv6
    /// form, an IPv6 address as its 16 bytes.
    #[test]
    fn inet_bytes_encode_both_families_in_sixteen_bytes() {
        let v6: std::net::IpAddr = "2001:db8::1".parse().unwrap();
        let mut want = [0u8; 16];
        want[..4].copy_from_slice(&[0x20, 0x01, 0x0d, 0xb8]);
        want[15] = 1;
        assert_eq!(inet_bytes(v6), want);
        let v4: std::net::IpAddr = "10.1.2.3".parse().unwrap();
        let mut mapped = [0u8; 16];
        mapped[10] = 0xff;
        mapped[11] = 0xff;
        mapped[12..].copy_from_slice(&[10, 1, 2, 3]);
        assert_eq!(inet_bytes(v4), mapped);
    }
}
