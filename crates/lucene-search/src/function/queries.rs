//! The function queries -- `FunctionQuery`, `FunctionRangeQuery` (with
//! `ValueSourceScorer`), `FunctionMatchQuery`, `FunctionScoreQuery` (with
//! `boostByValue`/`boostByQuery`'s `MultiplicativeBoostValuesSource` and
//! `QueryBoostValuesSource`) -- and `ValueSource`'s bridges to the
//! `DoubleValuesSource` API (`asDoubleValuesSource`, `asLongValuesSource`,
//! `fromDoubleValuesSource`, with its `ScorableView`) and its sort
//! (`getSortField`'s `ValueSourceSortField`).
//!
//! The queries are plain descriptions; how each runs over a segment is
//! `exec::function`.

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

use super::{
    java_float, BoxValues, FunctionContext, FunctionValues, RangeMatcher, TopLevel, ValueLeaf,
    ValueSource,
};
use crate::explain::Explanation;
use crate::query::Clause;
use crate::values_source::{
    BoxDoubleValues, BoxLongValues, DoubleValues, DoubleValuesSource, LongValues, LongValuesSource,
    ValuesContext,
};
use crate::Result;

// ---------------------------------------------------------------------------
// FunctionQuery
// ---------------------------------------------------------------------------

/// `FunctionQuery`: every document, scored by the source's `floatVal` times
/// the boost (a negative or `NaN` value scoring `0`).
#[derive(Clone)]
pub struct FunctionQuery {
    pub source: Arc<dyn ValueSource>,
}

impl FunctionQuery {
    pub fn new(source: Arc<dyn ValueSource>) -> Self {
        Self { source }
    }

    /// `getValueSource()`.
    pub fn value_source(&self) -> &Arc<dyn ValueSource> {
        &self.source
    }
}

impl fmt::Debug for FunctionQuery {
    /// `toString(field)`: the source's description.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.source.description())
    }
}

impl PartialEq for FunctionQuery {
    fn eq(&self, o: &Self) -> bool {
        super::same_source(self.source.as_ref(), o.source.as_ref())
    }
}

// ---------------------------------------------------------------------------
// FunctionRangeQuery, ValueSourceScorer
// ---------------------------------------------------------------------------

/// `FunctionRangeQuery`: the documents whose value is in a range, as the
/// source's values compare it (`FunctionValues.getRangeScorer`), each
/// scored by its `floatVal` (unboosted).
#[derive(Clone)]
pub struct FunctionRangeQuery {
    pub source: Arc<dyn ValueSource>,
    /// The bounds as strings (`FunctionValues.getRangeScorer` parses them);
    /// `None` unbounded.
    pub lower_val: Option<String>,
    pub upper_val: Option<String>,
    pub include_lower: bool,
    pub include_upper: bool,
}

impl FunctionRangeQuery {
    pub fn new(
        source: Arc<dyn ValueSource>,
        lower_val: Option<&str>,
        upper_val: Option<&str>,
        include_lower: bool,
        include_upper: bool,
    ) -> Self {
        Self {
            source,
            lower_val: lower_val.map(str::to_string),
            upper_val: upper_val.map(str::to_string),
            include_lower,
            include_upper,
        }
    }
}

impl fmt::Debug for FunctionRangeQuery {
    /// `toString(field)`: `frange(source):[lower TO upper]`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "frange({}):{}{} TO {}{}",
            self.source.description(),
            if self.include_lower { '[' } else { '{' },
            self.lower_val.as_deref().unwrap_or("*"),
            self.upper_val.as_deref().unwrap_or("*"),
            if self.include_upper { ']' } else { '}' }
        )
    }
}

impl PartialEq for FunctionRangeQuery {
    fn eq(&self, o: &Self) -> bool {
        self.include_lower == o.include_lower
            && self.include_upper == o.include_upper
            && super::same_source(self.source.as_ref(), o.source.as_ref())
            && self.lower_val == o.lower_val
            && self.upper_val == o.upper_val
    }
}

/// `ValueSourceScorer`: every document of the segment as the approximation,
/// confirmed by a range (or nothing: `FunctionValues.getScorer` matches
/// every document), scored by `floatVal` (negative infinity and `NaN`
/// as `-Float.MAX_VALUE`).
pub struct ValueSourceScorer<'a> {
    pub(crate) values: BoxValues<'a>,
    /// `None`: `getScorer`'s scorer, which matches every document at no
    /// cost.
    pub(crate) range: Option<RangeMatcher>,
    pub(crate) doc: i32,
    pub(crate) max_doc: i32,
    /// A batch's candidate documents (`exec::function`'s batches).
    pub(crate) candidates: Vec<i32>,
}

/// `ValueSourceScorer.DEF_COST`.
const DEF_COST: f32 = 5.0;

impl<'a> ValueSourceScorer<'a> {
    /// `FunctionValues.getRangeScorer(...)`'s scorer.
    ///
    /// # Errors
    /// An unparseable bound.
    pub fn range(
        mut values: BoxValues<'a>,
        max_doc: i32,
        lower: Option<&str>,
        upper: Option<&str>,
        include_lower: bool,
        include_upper: bool,
    ) -> Result<Self> {
        let range = values.range_matcher(lower, upper, include_lower, include_upper)?;
        Ok(Self {
            values,
            range: Some(range),
            doc: -1,
            max_doc,
            candidates: Vec::new(),
        })
    }

    /// `FunctionValues.getScorer(...)`'s scorer.
    pub fn all(values: BoxValues<'a>, max_doc: i32) -> Self {
        Self {
            values,
            range: None,
            doc: -1,
            max_doc,
            candidates: Vec::new(),
        }
    }

    /// `matches(doc)`.
    ///
    /// # Errors
    /// Whatever reading the value reports.
    pub fn matches_doc(&mut self, doc: i32) -> Result<bool> {
        match &self.range {
            None => Ok(true),
            Some(r) => r.matches(self.values.as_mut(), doc),
        }
    }

    /// `matchCost()`: `DEF_COST` plus the values' cost (`0` for
    /// `getScorer`'s).
    pub fn match_cost_of(&self) -> f32 {
        if self.range.is_none() {
            0.0
        } else {
            DEF_COST + self.values.cost()
        }
    }

    /// `score()`: `floatVal`, with negative infinity and `NaN` as
    /// `-Float.MAX_VALUE`.
    ///
    /// # Errors
    /// Whatever reading the value reports.
    pub fn score_doc(&mut self, doc: i32) -> Result<f32> {
        let score = self.values.float_val(doc)?;
        Ok(if score > f32::NEG_INFINITY {
            score
        } else {
            -f32::MAX
        })
    }
}

// ---------------------------------------------------------------------------
// FunctionMatchQuery
// ---------------------------------------------------------------------------

/// `FunctionMatchQuery.DEFAULT_MATCH_COST`.
pub const DEFAULT_MATCH_COST: f32 = 100.0;

/// The predicate a [`FunctionMatchQuery`] tests values with
/// (`DoublePredicate`).
pub type DoublePredicate = dyn Fn(f64) -> bool + Send + Sync;

/// `FunctionMatchQuery`: the documents whose value passes a predicate,
/// constant-scored.
#[derive(Clone)]
pub struct FunctionMatchQuery {
    pub source: Arc<dyn DoubleValuesSource>,
    pub filter: Arc<DoublePredicate>,
    /// Not part of equality, as in Java.
    pub match_cost: f32,
}

impl FunctionMatchQuery {
    pub fn new(source: Arc<dyn DoubleValuesSource>, filter: Arc<DoublePredicate>) -> Self {
        Self::with_match_cost(source, filter, DEFAULT_MATCH_COST)
    }

    pub fn with_match_cost(
        source: Arc<dyn DoubleValuesSource>,
        filter: Arc<DoublePredicate>,
        match_cost: f32,
    ) -> Self {
        Self {
            source,
            filter,
            match_cost,
        }
    }
}

impl fmt::Debug for FunctionMatchQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FunctionMatchQuery({})", self.source.describe())
    }
}

impl PartialEq for FunctionMatchQuery {
    /// The same source and the same predicate (by identity, as Java's
    /// lambdas compare).
    fn eq(&self, o: &Self) -> bool {
        Arc::ptr_eq(&self.filter, &o.filter) && self.source.describe() == o.source.describe()
    }
}

// ---------------------------------------------------------------------------
// FunctionScoreQuery
// ---------------------------------------------------------------------------

/// `FunctionScoreQuery`: `in`'s matches, scored by a values source (which
/// may read `in`'s score); a missing, negative or `NaN` value scores `0`.
#[derive(Clone)]
pub struct FunctionScoreQuery {
    pub in_query: Box<Clause>,
    pub source: Arc<dyn DoubleValuesSource>,
}

impl FunctionScoreQuery {
    pub fn new(in_query: impl Into<Clause>, source: Arc<dyn DoubleValuesSource>) -> Self {
        Self {
            in_query: Box::new(in_query.into()),
            source,
        }
    }

    /// `getWrappedQuery()`.
    pub fn wrapped_query(&self) -> &Clause {
        &self.in_query
    }

    /// `getSource()`.
    pub fn source(&self) -> &Arc<dyn DoubleValuesSource> {
        &self.source
    }

    /// `boostByValue(in, boost)`: `in`'s score times the boost's value (`1`
    /// where it has none).
    pub fn boost_by_value(in_query: impl Into<Clause>, boost: Arc<dyn DoubleValuesSource>) -> Self {
        Self::new(
            in_query,
            Arc::new(MultiplicativeBoostValuesSource { boost }),
        )
    }

    /// `boostByQuery(in, boostMatch, boostValue)`: `in`'s score, times
    /// `boostValue` where `boostMatch` matches too.
    pub fn boost_by_query(
        in_query: impl Into<Clause>,
        boost_match: impl Into<Clause>,
        boost_value: f32,
    ) -> Self {
        Self::new(
            in_query,
            Arc::new(MultiplicativeBoostValuesSource {
                boost: Arc::new(QueryBoostValuesSource {
                    query: crate::values_source::from_clause(boost_match.into()),
                    boost: boost_value,
                }),
            }),
        )
    }
}

impl fmt::Debug for FunctionScoreQuery {
    /// `FunctionScoreQuery(in, scored by source)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FunctionScoreQuery({}, scored by {})",
            crate::explain::describe_clause(&self.in_query),
            self.source.describe()
        )
    }
}

impl PartialEq for FunctionScoreQuery {
    fn eq(&self, o: &Self) -> bool {
        self.in_query == o.in_query && self.source.describe() == o.source.describe()
    }
}

/// `DoubleValues.withDefault(in, missingValue)`: every document has a
/// value, `missingValue` where `in` has none.
struct WithDefault<'c> {
    inner: BoxDoubleValues<'c>,
    missing: f64,
    has_value: bool,
}

impl DoubleValues for WithDefault<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.has_value = self.inner.advance_exact(doc)?;
        Ok(true)
    }
    fn double_value(&mut self) -> Result<f64> {
        if self.has_value {
            self.inner.double_value()
        } else {
            Ok(self.missing)
        }
    }
}

/// `FunctionScoreQuery.MultiplicativeBoostValuesSource`: the score times
/// the boost's value (`1` where it has none).
struct MultiplicativeBoostValuesSource {
    boost: Arc<dyn DoubleValuesSource>,
}

struct MultiplicativeValues<'c> {
    scores: BoxDoubleValues<'c>,
    boost: WithDefault<'c>,
    /// The scores' batch, for [`DoubleValues::fill_batch`].
    score_values: Vec<f64>,
    score_has: Vec<bool>,
}

impl DoubleValues for MultiplicativeValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.boost.advance_exact(doc)
    }
    fn double_value(&mut self) -> Result<f64> {
        Ok(self.scores.double_value()? * self.boost.double_value()?)
    }
    fn batch_capable(&self) -> bool {
        self.scores.batch_capable() && self.boost.inner.batch_capable()
    }
    /// Every document has a value: the score times the boost's value, or
    /// `1` where the boost has none (`WithDefault`).
    fn fill_batch(
        &mut self,
        docs: &[i32],
        scores: &[f32],
        out: &mut [f64],
        has: &mut [bool],
    ) -> Result<()> {
        let n = docs.len();
        self.boost.inner.fill_batch(docs, scores, out, has)?;
        self.score_values.resize(n, 0.0);
        self.score_has.resize(n, false);
        self.scores
            .fill_batch(docs, scores, &mut self.score_values, &mut self.score_has)?;
        for i in 0..n {
            let boost = if has[i] { out[i] } else { 1.0 };
            out[i] = self.score_values[i] * boost;
            has[i] = true;
        }
        Ok(())
    }
}

impl DoubleValuesSource for MultiplicativeBoostValuesSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        let scores = scores
            .ok_or_else(|| crate::Error::IllegalState("boost(): the scores are needed".into()))?;
        let inner = self.boost.get_values(ctx, leaf, None)?;
        Ok(Box::new(MultiplicativeValues {
            scores,
            boost: WithDefault {
                inner,
                missing: 1.0,
                has_value: false,
            },
            score_values: Vec::new(),
            score_has: Vec::new(),
        }))
    }
    fn needs_scores(&self) -> bool {
        true
    }
    fn needs_searcher(&self) -> bool {
        self.boost.needs_searcher()
    }
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        self.boost.is_cacheable(ctx, leaf)
    }
    fn describe(&self) -> String {
        format!("boost({})", self.boost.describe())
    }
    fn rewrite(&self, top: &TopLevel<'_>) -> Result<Option<Arc<dyn DoubleValuesSource>>> {
        Ok(self.boost.rewrite(top)?.map(|boost| {
            Arc::new(MultiplicativeBoostValuesSource { boost }) as Arc<dyn DoubleValuesSource>
        }))
    }
    fn queries(&self) -> Vec<&Clause> {
        self.boost.queries()
    }
    /// The score explanation times the boost's (the score's alone where the
    /// boost has no value).
    fn explain(
        &self,
        ctx: &ValuesContext<'_>,
        leaf: usize,
        doc: i32,
        score_explanation: &Explanation,
    ) -> Result<Explanation> {
        if !score_explanation.matched {
            return Ok(score_explanation.clone());
        }
        let boost = self.boost.explain(ctx, leaf, doc, score_explanation)?;
        if !boost.matched {
            return Ok(score_explanation.clone());
        }
        Ok(Explanation::match_double(
            score_explanation.double_value() * boost.double_value(),
            "product of:",
        )
        .with_details(vec![score_explanation.clone(), boost]))
    }
}

/// `FunctionScoreQuery.QueryBoostValuesSource`: `boost` where the query
/// matches, `1` elsewhere.
struct QueryBoostValuesSource {
    query: Arc<dyn DoubleValuesSource>,
    boost: f32,
}

/// `queryboost`'s values: the boost where the query has a value.
struct QueryBoostValues<'c> {
    inner: BoxDoubleValues<'c>,
    boost: f32,
}

impl DoubleValues for QueryBoostValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.inner.advance_exact(doc)
    }
    fn double_value(&mut self) -> Result<f64> {
        Ok(f64::from(self.boost))
    }
}

impl DoubleValuesSource for QueryBoostValuesSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        let inner = self.query.get_values(ctx, leaf, None)?;
        Ok(Box::new(WithDefault {
            inner: Box::new(QueryBoostValues {
                inner,
                boost: self.boost,
            }),
            missing: 1.0,
            has_value: false,
        }))
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn needs_searcher(&self) -> bool {
        self.query.needs_searcher()
    }
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        self.query.is_cacheable(ctx, leaf)
    }
    fn describe(&self) -> String {
        format!(
            "queryboost({})^{}",
            self.query.describe(),
            java_float(self.boost)
        )
    }
    fn rewrite(&self, top: &TopLevel<'_>) -> Result<Option<Arc<dyn DoubleValuesSource>>> {
        Ok(self.query.rewrite(top)?.map(|query| {
            Arc::new(QueryBoostValuesSource {
                query,
                boost: self.boost,
            }) as Arc<dyn DoubleValuesSource>
        }))
    }
    fn queries(&self) -> Vec<&Clause> {
        self.query.queries()
    }
    /// `Matched boosting query ...` at the boost where the query matches.
    fn explain(
        &self,
        ctx: &ValuesContext<'_>,
        leaf: usize,
        doc: i32,
        score_explanation: &Explanation,
    ) -> Result<Explanation> {
        let inner = self.query.explain(ctx, leaf, doc, score_explanation)?;
        if !inner.matched {
            return Ok(inner);
        }
        Ok(Explanation::match_(
            self.boost,
            format!("Matched boosting query {}", self.query.describe()),
        ))
    }
}

// ---------------------------------------------------------------------------
// ValueSource <-> DoubleValuesSource
// ---------------------------------------------------------------------------

/// `ValueSource.ScorableView`: the scores a wrapping values source was
/// handed, read for the document the wrapper was last moved to (and at
/// most once per document).
pub(crate) struct ScorableView<'a> {
    scores: Option<BoxDoubleValues<'a>>,
    pub(crate) doc_id: i32,
    scores_doc_id: i32,
    score: f32,
}

impl<'a> ScorableView<'a> {
    fn new(scores: Option<BoxDoubleValues<'a>>) -> Self {
        Self {
            scores,
            doc_id: -1,
            scores_doc_id: -1,
            score: 0.0,
        }
    }

    fn fixed(doc: i32, score: f32) -> Self {
        Self {
            scores: None,
            doc_id: doc,
            scores_doc_id: doc,
            score,
        }
    }

    /// `score()`.
    pub(crate) fn score(&mut self) -> Result<f32> {
        if self.scores_doc_id != self.doc_id {
            self.scores_doc_id = self.doc_id;
            self.score = match &mut self.scores {
                Some(s) => {
                    if s.advance_exact(self.doc_id)? {
                        s.double_value()? as f32
                    } else {
                        0.0
                    }
                }
                None => 0.0,
            };
        }
        Ok(self.score)
    }
}

/// `DoubleValuesSource.fromScorer(scorer)` over a [`ScorableView`].
pub(crate) struct FromDoubleValues<'a>(pub(crate) Rc<RefCell<ScorableView<'a>>>);

impl DoubleValues for FromDoubleValues<'_> {
    fn advance_exact(&mut self, _doc: i32) -> Result<bool> {
        Ok(true)
    }
    fn double_value(&mut self) -> Result<f64> {
        Ok(f64::from(self.0.borrow_mut().score()?))
    }
}

/// The leaf a wrapped source reads: the running search's segment, or
/// segment `leaf` of the context's searcher (or its reader alone).
fn value_leaf<'c>(ctx: &ValuesContext<'c>, leaf: usize) -> Result<ValueLeaf<'c>> {
    if let Some(lc) = ctx.exec_leaf() {
        return Ok(ValueLeaf::new(lc));
    }
    if let Ok(searcher) = ctx.searcher() {
        return ValueLeaf::of_searcher(searcher, leaf, None);
    }
    let reader = ctx.leaf_reader(leaf)?;
    Err(crate::Error::IllegalState(format!(
        "segment {}: a wrapped value source needs the searcher or the running search",
        reader.segment_name
    )))
}

/// `ValueSource.WrappedDoubleValuesSource`: a value source as a
/// `DoubleValuesSource` -- every document has a value (the source's
/// `doubleVal`).
struct WrappedDoubleValuesSource {
    inner: Arc<dyn ValueSource>,
    /// The context `rewrite(searcher)` bound it to (Java's
    /// `newContext(searcher)`).
    fcx: Option<Arc<FunctionContext>>,
}

struct WrappedDoubleValues<'c> {
    fv: BoxValues<'c>,
    scorer: Rc<RefCell<ScorableView<'c>>>,
    /// Nothing in `fv` holds the scorer view (no `fromDoubleValuesSource`
    /// under it reads the scores), so the values are the documents' alone.
    batch: bool,
}

impl DoubleValues for WrappedDoubleValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.scorer.borrow_mut().doc_id = doc;
        Ok(true)
    }
    fn double_value(&mut self) -> Result<f64> {
        let doc = self.scorer.borrow().doc_id;
        self.fv.double_val(doc)
    }
    fn batch_capable(&self) -> bool {
        self.batch
    }
    /// Every document has a value, the source's `doubleVal`.
    fn fill_batch(
        &mut self,
        docs: &[i32],
        _scores: &[f32],
        out: &mut [f64],
        has: &mut [bool],
    ) -> Result<()> {
        if let Some(&last) = docs.last() {
            self.scorer.borrow_mut().doc_id = last;
        }
        has[..docs.len()].fill(true);
        self.fv.double_val_batch(docs, out)
    }
}

/// A rewritten wrapper's context: everything `createWeight` computes, but
/// marked as Java's (which holds the searcher and saw no `createWeight`).
fn searcher_context(source: &dyn ValueSource, top: &TopLevel<'_>) -> Result<Arc<FunctionContext>> {
    let mut fcx = FunctionContext::create(source, top)?;
    fcx.searcher_only = true;
    Ok(Arc::new(fcx))
}

impl DoubleValuesSource for WrappedDoubleValuesSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        let empty = FunctionContext::new();
        let fcx = self.fcx.as_deref().unwrap_or(&empty);
        let scorer = Rc::new(RefCell::new(ScorableView::new(scores)));
        let mut vleaf = value_leaf(ctx, leaf)?;
        vleaf.scorer = Some(Rc::clone(&scorer));
        let fv = self.inner.get_values(fcx, &vleaf)?;
        drop(vleaf);
        // Only this wrapper still holds the view: no value under it reads it.
        let batch = Rc::strong_count(&scorer) == 1;
        Ok(Box::new(WrappedDoubleValues { fv, scorer, batch }))
    }
    /// On the safe side, as Java.
    fn needs_scores(&self) -> bool {
        true
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        false
    }
    fn describe(&self) -> String {
        self.inner.description()
    }
    fn rewrite(&self, top: &TopLevel<'_>) -> Result<Option<Arc<dyn DoubleValuesSource>>> {
        Ok(Some(Arc::new(WrappedDoubleValuesSource {
            inner: Arc::clone(&self.inner),
            fcx: Some(searcher_context(self.inner.as_ref(), top)?),
        })))
    }
    fn queries(&self) -> Vec<&Clause> {
        self.inner.queries()
    }
    /// The values' own explanation, with the explained score as the
    /// scorer's.
    fn explain(
        &self,
        ctx: &ValuesContext<'_>,
        leaf: usize,
        doc: i32,
        score_explanation: &Explanation,
    ) -> Result<Explanation> {
        let empty = FunctionContext::new();
        let fcx = self.fcx.as_deref().unwrap_or(&empty);
        let mut vleaf = value_leaf(ctx, leaf)?;
        vleaf.scorer = Some(Rc::new(RefCell::new(ScorableView::fixed(
            doc,
            score_explanation.value,
        ))));
        let mut fv = self.inner.get_values(fcx, &vleaf)?;
        fv.explain(doc)
    }
    fn wrapped_value_source(&self) -> Option<Arc<dyn ValueSource>> {
        Some(Arc::clone(&self.inner))
    }
}

/// `ValueSource.asDoubleValuesSource()`.
pub fn as_double_values_source(source: Arc<dyn ValueSource>) -> Arc<dyn DoubleValuesSource> {
    Arc::new(WrappedDoubleValuesSource {
        inner: source,
        fcx: None,
    })
}

/// `ValueSource.WrappedLongValuesSource`: a value source as a
/// `LongValuesSource` (a document has a value where it `exists`).
struct WrappedLongValuesSource {
    inner: Arc<dyn ValueSource>,
    fcx: Option<Arc<FunctionContext>>,
}

struct WrappedLongValues<'c> {
    fv: BoxValues<'c>,
    scorer: Rc<RefCell<ScorableView<'c>>>,
}

impl LongValues for WrappedLongValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.scorer.borrow_mut().doc_id = doc;
        self.fv.exists(doc)
    }
    fn long_value(&mut self) -> Result<i64> {
        let doc = self.scorer.borrow().doc_id;
        self.fv.long_val(doc)
    }
}

impl LongValuesSource for WrappedLongValuesSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxLongValues<'c>> {
        let empty = FunctionContext::new();
        let fcx = self.fcx.as_deref().unwrap_or(&empty);
        let scorer = Rc::new(RefCell::new(ScorableView::new(scores)));
        let mut vleaf = value_leaf(ctx, leaf)?;
        vleaf.scorer = Some(Rc::clone(&scorer));
        let fv = self.inner.get_values(fcx, &vleaf)?;
        Ok(Box::new(WrappedLongValues { fv, scorer }))
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        false
    }
    fn describe(&self) -> String {
        self.inner.description()
    }
    fn rewrite(&self, top: &TopLevel<'_>) -> Result<Option<Arc<dyn LongValuesSource>>> {
        Ok(Some(Arc::new(WrappedLongValuesSource {
            inner: Arc::clone(&self.inner),
            fcx: Some(searcher_context(self.inner.as_ref(), top)?),
        })))
    }
}

/// `ValueSource.asLongValuesSource()`.
pub fn as_long_values_source(source: Arc<dyn ValueSource>) -> Arc<dyn LongValuesSource> {
    Arc::new(WrappedLongValuesSource {
        inner: source,
        fcx: None,
    })
}

/// `ValueSource.FromDoubleValuesSource`: a `DoubleValuesSource` as a value
/// source (`floatVal`/`doubleVal` its value, `0` where it has none).
struct FromDoubleValuesSource {
    inner: Arc<dyn DoubleValuesSource>,
}

/// The source `createWeight` rewrote (Java rewrites it in `getValues`
/// against the context's searcher).
struct Rewritten(Arc<dyn DoubleValuesSource>);

struct FromValues<'a> {
    inner: BoxDoubleValues<'a>,
    description: String,
}

impl FunctionValues for FromValues<'_> {
    fn float_val(&mut self, doc: i32) -> Result<f32> {
        if !self.inner.advance_exact(doc)? {
            return Ok(0.0);
        }
        Ok(self.inner.double_value()? as f32)
    }
    fn double_val(&mut self, doc: i32) -> Result<f64> {
        if !self.inner.advance_exact(doc)? {
            return Ok(0.0);
        }
        self.inner.double_value()
    }
    fn exists(&mut self, doc: i32) -> Result<bool> {
        self.inner.advance_exact(doc)
    }
    fn to_string_doc(&mut self, _doc: i32) -> Result<String> {
        Ok(self.description.clone())
    }
}

impl ValueSource for FromDoubleValuesSource {
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>> {
        let scores: Option<BoxDoubleValues<'a>> = leaf
            .scorer
            .as_ref()
            .map(|s| Box::new(FromDoubleValues(Rc::clone(s))) as BoxDoubleValues<'a>);
        let source = fcx.get::<Rewritten, _>(self).map_or(&self.inner, |r| &r.0);
        let inner = source.get_values(&ValuesContext::for_leaf(leaf.ctx), 0, scores)?;
        Ok(Box::new(FromValues {
            inner,
            description: self.inner.describe(),
        }))
    }
    fn description(&self) -> String {
        self.inner.describe()
    }
    fn create_weight(&self, fcx: &mut FunctionContext, top: &TopLevel<'_>) -> Result<()> {
        if let Some(r) = self.inner.rewrite(top)? {
            fcx.put(self, Rewritten(r));
        }
        Ok(())
    }
    fn queries(&self) -> Vec<&Clause> {
        self.inner.queries()
    }
}

/// `ValueSource.fromDoubleValuesSource(in)`: the wrapped value source when
/// `in` wraps one.
pub fn from_double_values_source(source: Arc<dyn DoubleValuesSource>) -> Arc<dyn ValueSource> {
    match source.wrapped_value_source() {
        Some(vs) => vs,
        None => Arc::new(FromDoubleValuesSource { inner: source }),
    }
}

// ---------------------------------------------------------------------------
// Sorting
// ---------------------------------------------------------------------------

/// `ValueSource.getSortField(reverse)`: the source's own sort where it has
/// one (the field sources'), else `ValueSourceSortField` -- a `REWRITEABLE`
/// key over `doubleVal`, compared by `Double.compare`, which
/// `IndexSearcher.search(query, n, sort)` rewrites against the searcher
/// (`createWeight` run on a fresh context).
pub fn sort_field(source: Arc<dyn ValueSource>, reverse: bool) -> crate::top_field::SortField {
    if let Some(f) = source.native_sort_field(reverse) {
        return f;
    }
    let field = source.description();
    let id = crate::top_field::register_comparator_source(Arc::new(ValueSourceComparatorSource {
        source,
    }));
    crate::top_field::SortField::custom(&field, id, reverse)
}

/// `ValueSourceComparatorSource`, rewritten per searcher.
struct ValueSourceComparatorSource {
    source: Arc<dyn ValueSource>,
}

impl crate::top_field::FieldComparatorSource for ValueSourceComparatorSource {
    fn new_comparator(
        &self,
        _field: &str,
        _num_hits: usize,
        _reverse: bool,
    ) -> Box<dyn crate::top_field::FieldComparator> {
        Box::new(UnrewrittenComparator)
    }

    /// `ValueSourceSortField.rewrite(searcher)`: `createWeight` on a fresh
    /// context, then every leaf's `doubleVal`s (Java reads them per
    /// competitive document; the values are the same).
    fn rewrite(
        &self,
        ctx: &ValuesContext<'_>,
    ) -> Result<Option<Arc<dyn crate::top_field::FieldComparatorSource>>> {
        let searcher = ctx.searcher()?;
        let stats = super::queries_stats(searcher, self.source.as_ref())?;
        let top = TopLevel::of_searcher(searcher, Some(&stats));
        let fcx = FunctionContext::create(self.source.as_ref(), &top)?;
        let mut leaves = std::collections::HashMap::new();
        for (i, leaf) in top.leaves().enumerate() {
            let max_doc = leaf.max_doc()?;
            let mut values = self.source.get_values(&fcx, &leaf)?;
            let mut out = Vec::with_capacity(usize::try_from(max_doc).unwrap_or(0));
            for doc in 0..max_doc {
                out.push(crate::values_source::double_to_sortable_long(
                    values.double_val(doc)?,
                ));
            }
            let base = searcher.segments().get(i).map_or(0, |s| s.doc_base);
            leaves.insert(base, Arc::new(out));
        }
        Ok(Some(Arc::new(RewrittenValueSourceSort {
            leaves: Arc::new(leaves),
        })))
    }
}

/// A `ValueSourceSortField` used before its rewrite.
struct UnrewrittenComparator;

impl crate::top_field::FieldComparator for UnrewrittenComparator {
    fn leaf<'a>(
        &self,
        _ctx: crate::top_field::LeafCtx<'a>,
    ) -> Result<Box<dyn crate::top_field::LeafFieldComparator + 'a>> {
        Err(crate::Error::IllegalState(
            "a ValueSource sort must be rewritten against the searcher \
             (top_field::rewrite_sort)"
                .into(),
        ))
    }
    fn compare_values(
        &self,
        _a: &crate::top_field::SortValue,
        _b: &crate::top_field::SortValue,
    ) -> std::cmp::Ordering {
        std::cmp::Ordering::Equal
    }
}

/// The rewritten sort: each leaf's values as sortable longs, by doc base.
struct RewrittenValueSourceSort {
    leaves: Arc<std::collections::HashMap<i32, Arc<Vec<i64>>>>,
}

impl crate::top_field::FieldComparatorSource for RewrittenValueSourceSort {
    fn new_comparator(
        &self,
        _field: &str,
        _num_hits: usize,
        _reverse: bool,
    ) -> Box<dyn crate::top_field::FieldComparator> {
        Box::new(RewrittenValueSourceSort {
            leaves: Arc::clone(&self.leaves),
        })
    }
}

impl crate::top_field::FieldComparator for RewrittenValueSourceSort {
    fn leaf<'a>(
        &self,
        ctx: crate::top_field::LeafCtx<'a>,
    ) -> Result<Box<dyn crate::top_field::LeafFieldComparator + 'a>> {
        let values = self.leaves.get(&ctx.doc_base).cloned().ok_or_else(|| {
            crate::Error::IllegalState(format!(
                "the sort was rewritten against another reader: no leaf at docBase {}",
                ctx.doc_base
            ))
        })?;
        Ok(Box::new(RewrittenLeaf { values }))
    }
    fn compare_values(
        &self,
        a: &crate::top_field::SortValue,
        b: &crate::top_field::SortValue,
    ) -> std::cmp::Ordering {
        match (a, b) {
            (crate::top_field::SortValue::Long(x), crate::top_field::SortValue::Long(y)) => {
                x.cmp(y)
            }
            _ => std::cmp::Ordering::Equal,
        }
    }
}

struct RewrittenLeaf {
    values: Arc<Vec<i64>>,
}

impl crate::top_field::LeafFieldComparator for RewrittenLeaf {
    fn value(&mut self, doc: i32, _score: f32) -> Result<crate::top_field::SortValue> {
        let v = usize::try_from(doc)
            .ok()
            .and_then(|d| self.values.get(d).copied())
            .unwrap_or(0);
        Ok(crate::top_field::SortValue::Long(v))
    }
}
