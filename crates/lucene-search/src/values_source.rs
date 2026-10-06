//! Lucene's values-source API: per-document `double`/`long` values read
//! from doc values, a query's scores, vector similarities or late-interaction
//! multi-vectors -- `DoubleValues`, `DoubleValuesSource`, `LongValues`,
//! `LongValuesSource`, the vector similarity sources
//! (`VectorSimilarityValuesSource`, `FloatVectorSimilarityValuesSource`,
//! `ByteVectorSimilarityValuesSource`,
//! `FullPrecisionFloatVectorSimilarityValuesSource`),
//! `LateInteractionFloatValuesSource` with `MultiVectorSimilarity`, and
//! `NumericFieldStats`.
//!
//! # Shape
//!
//! A source is a trait object ([`DoubleValuesSource`], [`LongValuesSource`]),
//! as in Java, so a caller can write its own; the static factories are free
//! functions ([`from_long_field`], [`constant`], [`from_query`], ...). A
//! source reads a leaf through a [`ValuesContext`]: the searcher (its
//! segments, their readers, reader-wide statistics) and, when vector sources
//! are used, each segment's opened vectors -- `LeafReaderContext` has both,
//! this port opens vectors separately ([`crate::vector_query::VectorsInput`]).
//!
//! # Deviations
//!
//! * `rewrite(IndexSearcher)` is not needed: a query-backed source
//!   ([`from_query`]) scores through the searcher it is handed at
//!   `get_values` time, which is what Java's rewrite to a
//!   `WeightDoubleValuesSource` captures. Its scorer is the segment's whole
//!   scoring pass (deleted documents included, as `Weight.scorer` includes
//!   them), then looked up per document, rather than an iterator advanced to
//!   each: the values are the same.
//! * `hashCode`/`equals` are not ported; `toString` is [`std::fmt::Display`]
//!   through [`DoubleValuesSource::describe`].
//! * Sorting by a source ([`double_sort_field`], [`long_sort_field`]) reads
//!   each segment through [`ValuesContext::for_reader`]: a source that needs
//!   the searcher (a query's scores) or opened vectors cannot sort.

use std::collections::HashMap;
use std::sync::Arc;

use lucene_codecs::doc_values::NumericReader;
use lucene_codecs::field_infos::{
    DocValuesType, FieldInfo, VectorEncoding, VectorSimilarityFunction,
};

use crate::explain::Explanation;
use crate::index_searcher::IndexSearcher;
use crate::query::BooleanQuery;
use crate::vector_query::VectorsInput;
use crate::{Error, Result};

/// `DoubleValues`: a per-leaf cursor over `double` values.
pub trait DoubleValues {
    /// `advanceExact(doc)`: whether `doc` (leaf-local, not before the last
    /// target) has a value.
    fn advance_exact(&mut self, doc: i32) -> Result<bool>;
    /// `doubleValue()`: the value at the current document; only valid after
    /// [`Self::advance_exact`] returned `true`.
    fn double_value(&mut self) -> Result<f64>;

    /// Whether [`Self::fill_batch`] may stand in for the per-document calls:
    /// the values read the current scorer's score, if at all, only from
    /// `fill_batch`'s `scores` -- never from state a caller sets per document
    /// (`FunctionScoreQuery`'s score cell). `false` unless overridden.
    fn batch_capable(&self) -> bool {
        false
    }

    /// For each of `docs` (ascending, the first not before the last target),
    /// with `scores[i]` the score at `docs[i]`: `advance_exact`, then --
    /// where it has a value -- `double_value`. `has[i]` is whether `docs[i]`
    /// has a value and `out[i]` that value (unspecified without one); both
    /// as long as `docs`. Only valid when [`Self::batch_capable`]. By
    /// default the per-document calls, statically dispatched within the
    /// implementing type.
    ///
    /// # Errors
    /// Whatever reading a value reports.
    fn fill_batch(
        &mut self,
        docs: &[i32],
        _scores: &[f32],
        out: &mut [f64],
        has: &mut [bool],
    ) -> Result<()> {
        for ((&doc, o), h) in docs.iter().zip(out.iter_mut()).zip(has.iter_mut()) {
            *h = self.advance_exact(doc)?;
            if *h {
                *o = self.double_value()?;
            }
        }
        Ok(())
    }
}

/// `LongValues`: a per-leaf cursor over `long` values.
pub trait LongValues {
    fn advance_exact(&mut self, doc: i32) -> Result<bool>;
    fn long_value(&mut self) -> Result<i64>;
}

/// `DoubleValues.EMPTY`: no document has a value.
#[derive(Debug, Default, Clone, Copy)]
pub struct EmptyDoubleValues;

impl DoubleValues for EmptyDoubleValues {
    fn advance_exact(&mut self, _doc: i32) -> Result<bool> {
        Ok(false)
    }
    fn double_value(&mut self) -> Result<f64> {
        Ok(0.0)
    }
    fn batch_capable(&self) -> bool {
        true
    }
}

/// Every document has `value` (`ConstantValuesSource`'s values, and the
/// scores `explain` hands a source).
#[derive(Debug, Clone, Copy)]
pub struct ConstantDoubleValues(pub f64);

impl DoubleValues for ConstantDoubleValues {
    fn advance_exact(&mut self, _doc: i32) -> Result<bool> {
        Ok(true)
    }
    fn double_value(&mut self) -> Result<f64> {
        Ok(self.0)
    }
    fn batch_capable(&self) -> bool {
        true
    }
}

/// A boxed per-leaf cursor.
pub type BoxDoubleValues<'c> = Box<dyn DoubleValues + 'c>;
pub type BoxLongValues<'c> = Box<dyn LongValues + 'c>;

/// What a source reads a leaf through: the searcher, and each segment's
/// opened vectors (parallel to `searcher.segments()`; empty when no vector
/// source is used) -- or, for a sort's comparator, one segment's reader
/// alone ([`Self::for_reader`], its leaf `0`), which serves every source that
/// reads only the segment (fields, vectors, late interaction, constants).
#[derive(Clone, Copy)]
pub struct ValuesContext<'c> {
    searcher: Option<&'c IndexSearcher<'c, 'c>>,
    reader: Option<&'c crate::directory_reader::SegmentReader>,
    vectors: &'c [Option<&'c VectorsInput<'c>>],
    /// One segment of a running search (a function query's): its reader as
    /// leaf `0`, and its postings and statistics, which a query-backed
    /// source scores through as `weight.scorer(ctx)` does.
    leaf: Option<crate::exec::LeafContext<'c>>,
}

impl<'c> ValuesContext<'c> {
    pub fn new(searcher: &'c IndexSearcher<'c, 'c>) -> Self {
        Self {
            searcher: Some(searcher),
            reader: None,
            vectors: &[],
            leaf: None,
        }
    }

    /// One segment's reader as leaf `0`, without a searcher: what a sort's
    /// comparator has (`getLeafComparator(context)`).
    pub fn for_reader(reader: &'c crate::directory_reader::SegmentReader) -> Self {
        Self {
            searcher: None,
            reader: Some(reader),
            vectors: &[],
            leaf: None,
        }
    }

    /// One segment of a running search as leaf `0`.
    pub(crate) fn for_leaf(ctx: crate::exec::LeafContext<'c>) -> Self {
        Self {
            searcher: None,
            reader: ctx.reader,
            vectors: &[],
            leaf: Some(ctx),
        }
    }

    /// The running search's segment, for a context of one.
    pub(crate) fn exec_leaf(&self) -> Option<crate::exec::LeafContext<'c>> {
        self.leaf
    }

    /// The searcher, which a query-backed source needs.
    ///
    /// # Errors
    /// [`Error::IllegalState`] for a context of one reader.
    pub fn searcher(&self) -> Result<&'c IndexSearcher<'c, 'c>> {
        self.searcher.ok_or_else(|| {
            Error::IllegalState(
                "this values source needs a searcher (it reads the whole index)".to_string(),
            )
        })
    }

    pub fn with_vectors(mut self, vectors: &'c [Option<&'c VectorsInput<'c>>]) -> Self {
        self.vectors = vectors;
        self
    }

    fn reader(&self, leaf: usize) -> Result<&'c crate::directory_reader::SegmentReader> {
        let found = match (self.searcher, self.reader) {
            (Some(s), _) => s.segments().get(leaf).and_then(|s| s.reader),
            (None, Some(r)) if leaf == 0 => Some(r),
            _ => None,
        };
        found.ok_or_else(|| {
            Error::IllegalState(format!(
                "leaf {leaf}: a values source needs the segment's reader"
            ))
        })
    }

    /// Leaf `leaf`'s reader, as the spatial value sources (and a caller's
    /// own sources, through [`crate::reader::doc_values`]) read it.
    ///
    /// # Errors
    /// [`Error::IllegalState`] when the context has no reader for `leaf`.
    pub fn leaf_reader(&self, leaf: usize) -> Result<&'c crate::directory_reader::SegmentReader> {
        self.reader(leaf)
    }

    /// Leaf `leaf`'s doc base (`LeafReaderContext.docBase`): `0` for a
    /// context of one reader.
    pub(crate) fn doc_base(&self, leaf: usize) -> i32 {
        self.searcher
            .and_then(|s| s.segments().get(leaf))
            .map_or(0, |s| s.doc_base)
    }

    fn vectors(&self, leaf: usize) -> Result<&'c VectorsInput<'c>> {
        self.vectors.get(leaf).copied().flatten().ok_or_else(|| {
            Error::IllegalState(format!(
                "leaf {leaf}: a vector values source needs the segment's vectors"
            ))
        })
    }
}

/// `DoubleValuesSource`.
pub trait DoubleValuesSource: Send + Sync {
    /// `getValues(ctx, scores)`: the values of leaf `leaf`; `scores` are the
    /// current scorer's (for a source that [needs them](Self::needs_scores)).
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>>;

    /// `needsScores()`.
    fn needs_scores(&self) -> bool;

    /// Whether the values need more than the leaf's reader -- the searcher
    /// (a query-backed source) or the opened vectors of a [`ValuesContext`]
    /// -- which a sort's comparator only has once the sort is rewritten
    /// ([`crate::top_field::rewrite_sort`], Java's `rewrite(searcher)`).
    fn needs_searcher(&self) -> bool {
        false
    }

    /// `SegmentCacheable.isCacheable(ctx)`.
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool;

    /// `toString()`.
    fn describe(&self) -> String;

    /// `rewrite(searcher)`: a source bound to the reader (`None`: this one
    /// as it is). [`crate::function::FunctionScoreQuery`] and
    /// [`crate::function::FunctionMatchQuery`] search through it, as their
    /// `createWeight` does.
    ///
    /// # Errors
    /// Whatever reading the reader reports.
    fn rewrite(
        &self,
        _top: &crate::function::TopLevel<'_>,
    ) -> Result<Option<Arc<dyn DoubleValuesSource>>> {
        Ok(None)
    }

    /// The queries the source scores, whose reader-wide statistics a search
    /// gathers.
    fn queries(&self) -> Vec<&crate::query::Clause> {
        Vec::new()
    }

    /// The value source this wraps (`ValueSource.asDoubleValuesSource()`'s
    /// `WrappedDoubleValuesSource`), which `fromDoubleValuesSource` unwraps.
    fn wrapped_value_source(&self) -> Option<Arc<dyn crate::function::ValueSource>> {
        None
    }

    /// `explain(ctx, docId, scoreExplanation)`: the value of `doc` computed
    /// with the explained score as the scores.
    fn explain(
        &self,
        ctx: &ValuesContext<'_>,
        leaf: usize,
        doc: i32,
        score_explanation: &Explanation,
    ) -> Result<Explanation> {
        let scores = Box::new(ConstantDoubleValues(f64::from(score_explanation.value)));
        let mut dv = self.get_values(ctx, leaf, Some(scores))?;
        if dv.advance_exact(doc)? {
            Ok(Explanation::match_double(
                dv.double_value()?,
                self.describe(),
            ))
        } else {
            Ok(Explanation::no_match(self.describe()))
        }
    }
}

/// `LongValuesSource`.
pub trait LongValuesSource: Send + Sync {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxLongValues<'c>>;
    fn needs_scores(&self) -> bool;
    /// As [`DoubleValuesSource::needs_searcher`].
    fn needs_searcher(&self) -> bool {
        false
    }
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool;
    fn describe(&self) -> String;
    /// As [`DoubleValuesSource::rewrite`].
    ///
    /// # Errors
    /// Whatever reading the reader reports.
    fn rewrite(
        &self,
        _top: &crate::function::TopLevel<'_>,
    ) -> Result<Option<Arc<dyn LongValuesSource>>> {
        Ok(None)
    }
}

impl std::fmt::Debug for dyn DoubleValuesSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.describe())
    }
}

impl std::fmt::Debug for dyn LongValuesSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.describe())
    }
}

// ---------------------------------------------------------------------------
// Doc-values sources
// ---------------------------------------------------------------------------

/// `DocValues.isCacheable(ctx, field)`: a field with no doc-values updates
/// (`dvGen == -1`), or a field the segment lacks.
pub fn doc_values_cacheable(ctx: &ValuesContext<'_>, leaf: usize, field: &str) -> bool {
    match ctx.reader(leaf) {
        Ok(r) => r
            .field_infos()
            .field_by_name(field)
            .is_none_or(|fi| fi.doc_values_gen == -1),
        Err(_) => false,
    }
}

/// `DocValues.getNumeric(reader, field)`: the field's `NUMERIC` column, or
/// `None` (Java's empty values) when the segment has no such field; a field
/// with doc values of another type is `IllegalStateException`.
fn numeric_column<'c>(
    ctx: &ValuesContext<'c>,
    leaf: usize,
    field: &str,
) -> Result<Option<NumericReader<'c>>> {
    let reader = ctx.reader(leaf)?;
    let Some(info) = reader.field_infos().field_by_name(field) else {
        return Ok(None);
    };
    check_doc_values_type(info, DocValuesType::Numeric)?;
    let Some((meta, data)) = reader.doc_values_for_field(info.number) else {
        return Ok(None);
    };
    Ok(meta
        .numeric_entry(info.number)
        .map(|e| NumericReader::new(data, e)))
}

/// `DocValues.checkField`: a field whose doc values are of another type.
fn check_doc_values_type(info: &FieldInfo, want: DocValuesType) -> Result<()> {
    if info.doc_values_type != DocValuesType::None && info.doc_values_type != want {
        return Err(Error::IllegalState(format!(
            "unexpected docvalues type {:?} for field '{}' (expected={want:?})",
            info.doc_values_type, info.name
        )));
    }
    Ok(())
}

/// A leaf's `NUMERIC` values as `LongValues`.
struct NumericLongValues<'c> {
    column: Option<NumericReader<'c>>,
    current: i64,
}

impl LongValues for NumericLongValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        let Some(c) = self.column.as_mut() else {
            return Ok(false);
        };
        match c.value(doc)? {
            Some(v) => {
                self.current = v;
                Ok(true)
            }
            None => Ok(false),
        }
    }
    fn long_value(&mut self) -> Result<i64> {
        Ok(self.current)
    }
}

/// How `fromField` decodes a stored long into a double.
#[derive(Clone)]
pub enum Decoder {
    /// `(double) v` (`fromLongField`, `fromIntField`).
    Long,
    /// `Double.longBitsToDouble(v)` (`fromDoubleField`).
    Double,
    /// `Float.intBitsToFloat((int) v)` widened (`fromFloatField`).
    Float,
    /// Any `LongToDoubleFunction` (`fromField(field, decoder)`).
    Custom(Arc<dyn Fn(i64) -> f64 + Send + Sync>),
}

impl Decoder {
    fn apply(&self, v: i64) -> f64 {
        match self {
            Decoder::Long => v as f64,
            Decoder::Double => f64::from_bits(v as u64),
            Decoder::Float => f64::from(f32::from_bits(v as i32 as u32)),
            Decoder::Custom(f) => f(v),
        }
    }
}

/// `DoubleValuesSource.FieldValuesSource`.
struct FieldDoubleSource {
    field: String,
    decoder: Decoder,
}

struct FieldDoubleValues<'c> {
    inner: NumericLongValues<'c>,
    decoder: Decoder,
}

impl DoubleValues for FieldDoubleValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.inner.advance_exact(doc)
    }
    fn double_value(&mut self) -> Result<f64> {
        Ok(self.decoder.apply(self.inner.current))
    }
    fn batch_capable(&self) -> bool {
        true
    }
}

impl DoubleValuesSource for FieldDoubleSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        Ok(Box::new(FieldDoubleValues {
            inner: NumericLongValues {
                column: numeric_column(ctx, leaf, &self.field)?,
                current: 0,
            },
            decoder: self.decoder.clone(),
        }))
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        doc_values_cacheable(ctx, leaf, &self.field)
    }
    fn describe(&self) -> String {
        format!("double({})", self.field)
    }
    /// `FieldValuesSource.explain` reads the field, not the scores.
    fn explain(
        &self,
        ctx: &ValuesContext<'_>,
        leaf: usize,
        doc: i32,
        _score_explanation: &Explanation,
    ) -> Result<Explanation> {
        let mut v = self.get_values(ctx, leaf, None)?;
        if v.advance_exact(doc)? {
            Ok(Explanation::match_double(
                v.double_value()?,
                self.describe(),
            ))
        } else {
            Ok(Explanation::no_match(self.describe()))
        }
    }
}

/// `DoubleValuesSource.fromField(field, decoder)`.
pub fn from_field(field: &str, decoder: Decoder) -> Arc<dyn DoubleValuesSource> {
    Arc::new(FieldDoubleSource {
        field: field.to_string(),
        decoder,
    })
}

/// `DoubleValuesSource.fromDoubleField(field)`.
pub fn from_double_field(field: &str) -> Arc<dyn DoubleValuesSource> {
    from_field(field, Decoder::Double)
}

/// `DoubleValuesSource.fromFloatField(field)`.
pub fn from_float_field(field: &str) -> Arc<dyn DoubleValuesSource> {
    from_field(field, Decoder::Float)
}

/// `DoubleValuesSource.fromLongField(field)`.
pub fn from_long_field(field: &str) -> Arc<dyn DoubleValuesSource> {
    from_field(field, Decoder::Long)
}

/// `DoubleValuesSource.fromIntField(field)`.
pub fn from_int_field(field: &str) -> Arc<dyn DoubleValuesSource> {
    from_long_field(field)
}

// ---------------------------------------------------------------------------
// Scores and constants
// ---------------------------------------------------------------------------

/// `DoubleValuesSource.SCORES`.
struct ScoresSource;

impl DoubleValuesSource for ScoresSource {
    fn get_values<'c>(
        &self,
        _ctx: &ValuesContext<'c>,
        _leaf: usize,
        scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        scores.ok_or_else(|| {
            Error::IllegalArgument("the SCORES source needs the scorer's scores".to_string())
        })
    }
    fn needs_scores(&self) -> bool {
        true
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        false
    }
    fn describe(&self) -> String {
        "scores".to_string()
    }
    fn explain(
        &self,
        _ctx: &ValuesContext<'_>,
        _leaf: usize,
        _doc: i32,
        score_explanation: &Explanation,
    ) -> Result<Explanation> {
        Ok(score_explanation.clone())
    }
}

/// `DoubleValuesSource.SCORES`: the scores it is handed.
pub fn scores() -> Arc<dyn DoubleValuesSource> {
    Arc::new(ScoresSource)
}

/// `DoubleValuesSource.ConstantValuesSource`.
struct ConstantSource(f64);

impl DoubleValuesSource for ConstantSource {
    fn get_values<'c>(
        &self,
        _ctx: &ValuesContext<'c>,
        _leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        Ok(Box::new(ConstantDoubleValues(self.0)))
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        true
    }
    fn describe(&self) -> String {
        format!("constant({})", java_double(self.0))
    }
    fn explain(
        &self,
        _ctx: &ValuesContext<'_>,
        _leaf: usize,
        _doc: i32,
        _score_explanation: &Explanation,
    ) -> Result<Explanation> {
        Ok(Explanation::match_double(self.0, self.describe()))
    }
}

/// `Double.toString` for the plain cases `toString` prints (`3.0`, not `3`).
fn java_double(v: f64) -> String {
    if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e7 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

/// `DoubleValuesSource.constant(value)`.
pub fn constant(value: f64) -> Arc<dyn DoubleValuesSource> {
    Arc::new(ConstantSource(value))
}

// ---------------------------------------------------------------------------
// A query's scores
// ---------------------------------------------------------------------------

/// `QueryDoubleValuesSource`/`WeightDoubleValuesSource`: a document's score
/// under `query` (`ScoreMode.COMPLETE`, boost 1), where it matches.
/// The query (as the boolean a searcher scores), its rewrite, and the query
/// as given (its `toString`).
struct QuerySource(BooleanQuery, crate::query::Clause, crate::query::Clause);

/// A leaf's matches and scores, ascending by document.
struct SortedScores {
    docs: Vec<i32>,
    scores: Vec<f32>,
    at: usize,
}

impl DoubleValues for SortedScores {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        while self.at < self.docs.len() && self.docs[self.at] < doc {
            self.at += 1;
        }
        Ok(self.docs.get(self.at) == Some(&doc))
    }
    fn double_value(&mut self) -> Result<f64> {
        Ok(f64::from(self.scores.get(self.at).copied().unwrap_or(0.0)))
    }
}

impl DoubleValuesSource for QuerySource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        if let Some(lc) = ctx.exec_leaf() {
            // `WeightDoubleValuesSource.getValues`: the query's scorer
            // (`ScoreMode.COMPLETE`), advanced with the documents.
            let scorer =
                crate::exec::build::build(&lc, &self.1, 1.0, crate::exec::Mode::Complete, false)?;
            return Ok(match scorer {
                None => Box::new(EmptyDoubleValues),
                Some(scorer) => Box::new(WeightScores {
                    scorer,
                    tpi_match: None,
                }),
            });
        }
        let hits = ctx.searcher()?.leaf_scores(&self.0, leaf, true)?;
        let (docs, scores) = hits.into_iter().unzip();
        Ok(Box::new(SortedScores {
            docs,
            scores,
            at: 0,
        }))
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn needs_searcher(&self) -> bool {
        true
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        false
    }
    fn describe(&self) -> String {
        format!("score({})", crate::explain::describe_clause(&self.2))
    }
    /// `WeightDoubleValuesSource.explain`: the query's own explanation.
    fn explain(
        &self,
        ctx: &ValuesContext<'_>,
        leaf: usize,
        doc: i32,
        _score_explanation: &Explanation,
    ) -> Result<Explanation> {
        if let Some(lc) = ctx.exec_leaf() {
            return crate::explain::explain_clause_with_stats(
                lc.fields,
                lc.doc_in,
                lc.pos_in,
                lc.pay_in,
                lc.live_docs,
                lc.points,
                &self.1,
                doc,
                lc.norms,
                lc.global,
            );
        }
        let searcher = ctx.searcher()?;
        let base = searcher.segments().get(leaf).map_or(0, |s| s.doc_base);
        searcher.explain(&self.0, base + doc)
    }
    fn queries(&self) -> Vec<&crate::query::Clause> {
        vec![&self.1]
    }
}

/// `WeightDoubleValuesSource`'s values: the scorer's score where it
/// matches (its two-phase check run once per document).
struct WeightScores<'c> {
    scorer: crate::exec::BoxScorer<'c>,
    tpi_match: Option<bool>,
}

impl DoubleValues for WeightScores<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        if self.scorer.doc_id() < doc {
            self.scorer.advance(doc)?;
            self.tpi_match = None;
        }
        if self.scorer.doc_id() == doc {
            if !self.scorer.two_phase() {
                return Ok(true);
            }
            if self.tpi_match.is_none() {
                self.tpi_match = Some(self.scorer.matches()?);
            }
            return Ok(self.tpi_match == Some(true));
        }
        Ok(false)
    }
    fn double_value(&mut self) -> Result<f64> {
        Ok(f64::from(self.scorer.score()?))
    }
}

/// `DoubleValuesSource.fromQuery(query)`.
pub fn from_query(query: BooleanQuery) -> Arc<dyn DoubleValuesSource> {
    from_clause(crate::query::Clause::Boolean(Box::new(query)))
}

/// `DoubleValuesSource.fromQuery(query)` for any query.
pub fn from_clause(query: crate::query::Clause) -> Arc<dyn DoubleValuesSource> {
    let rewritten = query.clone().rewrite();
    let as_boolean = match &rewritten {
        crate::query::Clause::Boolean(b) => (**b).clone(),
        other => BooleanQuery {
            must: vec![other.clone()],
            ..Default::default()
        },
    };
    Arc::new(QuerySource(as_boolean, rewritten, query))
}

// ---------------------------------------------------------------------------
// Long sources, and conversions
// ---------------------------------------------------------------------------

/// `LongValuesSource.FieldValuesSource`.
struct FieldLongSource(String);

impl LongValuesSource for FieldLongSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxLongValues<'c>> {
        Ok(Box::new(NumericLongValues {
            column: numeric_column(ctx, leaf, &self.0)?,
            current: 0,
        }))
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        doc_values_cacheable(ctx, leaf, &self.0)
    }
    fn describe(&self) -> String {
        format!("long({})", self.0)
    }
}

/// `LongValuesSource.fromLongField(field)`.
pub fn long_from_long_field(field: &str) -> Arc<dyn LongValuesSource> {
    Arc::new(FieldLongSource(field.to_string()))
}

/// `LongValuesSource.fromIntField(field)`.
pub fn long_from_int_field(field: &str) -> Arc<dyn LongValuesSource> {
    long_from_long_field(field)
}

/// `LongValuesSource.ConstantLongValuesSource`.
struct ConstantLongSource(i64);

struct ConstantLongValues(i64);

impl LongValues for ConstantLongValues {
    fn advance_exact(&mut self, _doc: i32) -> Result<bool> {
        Ok(true)
    }
    fn long_value(&mut self) -> Result<i64> {
        Ok(self.0)
    }
}

impl LongValuesSource for ConstantLongSource {
    fn get_values<'c>(
        &self,
        _ctx: &ValuesContext<'c>,
        _leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxLongValues<'c>> {
        Ok(Box::new(ConstantLongValues(self.0)))
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        true
    }
    fn describe(&self) -> String {
        format!("constant({})", self.0)
    }
}

/// `LongValuesSource.constant(value)`.
pub fn long_constant(value: i64) -> Arc<dyn LongValuesSource> {
    Arc::new(ConstantLongSource(value))
}

/// How a double source becomes a long one.
#[derive(Clone, Copy)]
enum DoubleToLong {
    /// `toLongValuesSource()`: `(long) value`.
    Cast,
    /// `toSortableLongDoubleValuesSource()`:
    /// `NumericUtils.doubleToSortableLong(value)`.
    Sortable,
}

struct DoubleAsLongSource {
    inner: Arc<dyn DoubleValuesSource>,
    how: DoubleToLong,
}

struct DoubleAsLongValues<'c> {
    inner: BoxDoubleValues<'c>,
    how: DoubleToLong,
}

/// `NumericUtils.doubleToSortableLong` (over `Double.doubleToLongBits`: one
/// NaN).
pub fn double_to_sortable_long(d: f64) -> i64 {
    let bits = if d.is_nan() {
        0x7ff8_0000_0000_0000
    } else {
        d.to_bits() as i64
    };
    bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff)
}

impl LongValues for DoubleAsLongValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.inner.advance_exact(doc)
    }
    fn long_value(&mut self) -> Result<i64> {
        let v = self.inner.double_value()?;
        Ok(match self.how {
            // Java's `(long)` saturates and maps NaN to 0, as `as` does.
            DoubleToLong::Cast => v as i64,
            DoubleToLong::Sortable => double_to_sortable_long(v),
        })
    }
}

impl LongValuesSource for DoubleAsLongSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxLongValues<'c>> {
        Ok(Box::new(DoubleAsLongValues {
            inner: self.inner.get_values(ctx, leaf, scores)?,
            how: self.how,
        }))
    }
    fn needs_scores(&self) -> bool {
        self.inner.needs_scores()
    }
    fn needs_searcher(&self) -> bool {
        self.inner.needs_searcher()
    }
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        match self.how {
            DoubleToLong::Cast => self.inner.is_cacheable(ctx, leaf),
            DoubleToLong::Sortable => false,
        }
    }
    fn describe(&self) -> String {
        match self.how {
            DoubleToLong::Cast => format!("long({})", self.inner.describe()),
            DoubleToLong::Sortable => format!("sortableLong({})", self.inner.describe()),
        }
    }
    /// The inner source's rewrite, converted the same way.
    fn rewrite(
        &self,
        top: &crate::function::TopLevel<'_>,
    ) -> Result<Option<Arc<dyn LongValuesSource>>> {
        Ok(self.inner.rewrite(top)?.map(|inner| {
            Arc::new(DoubleAsLongSource {
                inner,
                how: self.how,
            }) as Arc<dyn LongValuesSource>
        }))
    }
}

/// `DoubleValuesSource.toLongValuesSource()`.
pub fn to_long_values_source(inner: Arc<dyn DoubleValuesSource>) -> Arc<dyn LongValuesSource> {
    Arc::new(DoubleAsLongSource {
        inner,
        how: DoubleToLong::Cast,
    })
}

/// `DoubleValuesSource.toSortableLongDoubleValuesSource()`.
pub fn to_sortable_long_values_source(
    inner: Arc<dyn DoubleValuesSource>,
) -> Arc<dyn LongValuesSource> {
    Arc::new(DoubleAsLongSource {
        inner,
        how: DoubleToLong::Sortable,
    })
}

/// `LongValuesSource.DoubleLongValuesSource`.
struct LongAsDoubleSource(Arc<dyn LongValuesSource>);

struct LongAsDoubleValues<'c>(BoxLongValues<'c>);

impl DoubleValues for LongAsDoubleValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.0.advance_exact(doc)
    }
    fn double_value(&mut self) -> Result<f64> {
        Ok(self.0.long_value()? as f64)
    }
}

impl DoubleValuesSource for LongAsDoubleSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        Ok(Box::new(LongAsDoubleValues(
            self.0.get_values(ctx, leaf, scores)?,
        )))
    }
    fn needs_scores(&self) -> bool {
        self.0.needs_scores()
    }
    fn needs_searcher(&self) -> bool {
        self.0.needs_searcher()
    }
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        self.0.is_cacheable(ctx, leaf)
    }
    fn describe(&self) -> String {
        format!("double({})", self.0.describe())
    }
    /// `inner.rewrite(searcher).toDoubleValuesSource()`.
    fn rewrite(
        &self,
        top: &crate::function::TopLevel<'_>,
    ) -> Result<Option<Arc<dyn DoubleValuesSource>>> {
        Ok(self
            .0
            .rewrite(top)?
            .map(|r| Arc::new(LongAsDoubleSource(r)) as Arc<dyn DoubleValuesSource>))
    }
}

/// `LongValuesSource.toDoubleValuesSource()`.
pub fn to_double_values_source(inner: Arc<dyn LongValuesSource>) -> Arc<dyn DoubleValuesSource> {
    Arc::new(LongAsDoubleSource(inner))
}

// ---------------------------------------------------------------------------
// Vector similarity sources
// ---------------------------------------------------------------------------

/// The field's vector metadata in `leaf`, or `None` when the segment lacks
/// the field (`FloatVectorValues.checkField`: a field that exists without
/// vectors of this encoding is `IllegalStateException`).
fn vector_field<'c>(
    ctx: &ValuesContext<'c>,
    leaf: usize,
    field: &str,
    encoding: VectorEncoding,
) -> Result<Option<(&'c VectorsInput<'c>, i32, VectorSimilarityFunction, i32)>> {
    let reader = ctx.reader(leaf)?;
    let Some(info) = reader.field_infos().field_by_name(field) else {
        return Ok(None);
    };
    if info.vector_dimension == 0 || info.vector_encoding != encoding {
        return Err(Error::IllegalState(format!(
            "field \"{field}\" does not have {} vectors indexed",
            match encoding {
                VectorEncoding::Float32 => "float",
                VectorEncoding::Byte => "byte",
            }
        )));
    }
    let vectors = ctx.vectors(leaf)?;
    Ok(Some((
        vectors,
        info.number,
        info.vector_similarity_function,
        info.vector_dimension,
    )))
}

/// Documents with a vector (ascending) and each one's ordinal.
fn doc_ords(size: i32, ord_to_doc: impl Fn(i32) -> Result<i32>) -> Result<Vec<i32>> {
    (0..size).map(ord_to_doc).collect()
}

/// A leaf's vector scores, computed per document as it is reached.
struct VectorScores<'c> {
    docs: Vec<i32>,
    at: usize,
    score: Box<dyn FnMut(i32) -> Result<f64> + 'c>,
}

impl DoubleValues for VectorScores<'_> {
    /// `doc >= iterator.docID() && (iterator.docID() == doc ||
    /// iterator.advance(doc) == doc)`.
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        while self.at < self.docs.len() && self.docs[self.at] < doc {
            self.at += 1;
        }
        Ok(self.docs.get(self.at) == Some(&doc))
    }
    fn double_value(&mut self) -> Result<f64> {
        let ord = i32::try_from(self.at).map_err(|_| Error::IllegalState("ord".into()))?;
        (self.score)(ord)
    }
}

fn check_dimension(query: usize, field: i32) -> Result<()> {
    if i32::try_from(query).ok() != Some(field) {
        return Err(Error::IllegalArgument(format!(
            "vector query dimension: {query} differs from field dimension: {field}"
        )));
    }
    Ok(())
}

/// `FloatVectorSimilarityValuesSource`: the similarity of each document's
/// vector to `query`, under the field's similarity function.
struct FloatVectorSource {
    field: String,
    query: Vec<f32>,
}

impl DoubleValuesSource for FloatVectorSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        float_vector_values(ctx, leaf, &self.field, self.query.clone(), None, true)
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn needs_searcher(&self) -> bool {
        true
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        true
    }
    fn describe(&self) -> String {
        format!(
            "FloatVectorSimilarityValuesSource(fieldName={} queryVector={:?})",
            self.field, self.query
        )
    }
}

fn float_vector_values<'c>(
    ctx: &ValuesContext<'c>,
    leaf: usize,
    field: &str,
    query: Vec<f32>,
    function: Option<VectorSimilarityFunction>,
    check_dim_as_scorer: bool,
) -> Result<BoxDoubleValues<'c>> {
    let Some((vectors, number, sim, dim)) =
        vector_field(ctx, leaf, field, VectorEncoding::Float32)?
    else {
        return Ok(Box::new(EmptyDoubleValues));
    };
    if check_dim_as_scorer || function.is_some() {
        check_dimension(query.len(), dim)?;
    } else if i32::try_from(query.len()).ok() != Some(dim) {
        return Err(Error::IllegalArgument(format!(
            "Query vector dimension does not match field dimension: {} != {dim}",
            query.len()
        )));
    }
    let values = vectors.flat.float_vector_values(number)?;
    let docs = doc_ords(values.size(), |o| Ok(values.ord_to_doc(o)?))?;
    let sim = function.unwrap_or(sim);
    Ok(Box::new(VectorScores {
        docs,
        at: 0,
        score: Box::new(move |ord| {
            let v = values.vector(ord)?;
            Ok(f64::from(sim.score(&query, &v)))
        }),
    }))
}

/// `DoubleValuesSource.similarityToQueryVector(ctx, float[], field)`'s
/// source.
pub fn float_vector_similarity(field: &str, query: Vec<f32>) -> Arc<dyn DoubleValuesSource> {
    Arc::new(FloatVectorSource {
        field: field.to_string(),
        query,
    })
}

/// `ByteVectorSimilarityValuesSource`.
struct ByteVectorSource {
    field: String,
    query: Vec<u8>,
}

impl DoubleValuesSource for ByteVectorSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        let Some((vectors, number, sim, dim)) =
            vector_field(ctx, leaf, &self.field, VectorEncoding::Byte)?
        else {
            return Ok(Box::new(EmptyDoubleValues));
        };
        check_dimension(self.query.len(), dim)?;
        let values = vectors.flat.byte_vector_values(number)?;
        let docs = doc_ords(values.size(), |o| Ok(values.ord_to_doc(o)?))?;
        let query = self.query.clone();
        Ok(Box::new(VectorScores {
            docs,
            at: 0,
            score: Box::new(move |ord| {
                let v = values.vector(ord)?;
                Ok(f64::from(sim.score_bytes(&query, v)))
            }),
        }))
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn needs_searcher(&self) -> bool {
        true
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        true
    }
    fn describe(&self) -> String {
        format!(
            "ByteVectorSimilarityValuesSource(fieldName={} queryVector={:?})",
            self.field,
            self.query.iter().map(|&b| b as i8).collect::<Vec<_>>()
        )
    }
}

/// `DoubleValuesSource.similarityToQueryVector(ctx, byte[], field)`'s source
/// (`query` as the bytes of Java's `byte[]`).
pub fn byte_vector_similarity(field: &str, query: Vec<u8>) -> Arc<dyn DoubleValuesSource> {
    Arc::new(ByteVectorSource {
        field: field.to_string(),
        query,
    })
}

/// `FullPrecisionFloatVectorSimilarityValuesSource`: the raw float vectors'
/// similarity, under `function` or the field's own. Over the flat format
/// this port reads, the raw vectors are the only ones, so it scores as
/// [`float_vector_similarity`] does; it exists for quantized formats, whose
/// `rescorer` reads the full-precision copy.
struct FullPrecisionSource {
    field: String,
    query: Vec<f32>,
    function: Option<VectorSimilarityFunction>,
}

impl DoubleValuesSource for FullPrecisionSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        float_vector_values(
            ctx,
            leaf,
            &self.field,
            self.query.clone(),
            self.function,
            false,
        )
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn needs_searcher(&self) -> bool {
        true
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        true
    }
    fn describe(&self) -> String {
        format!(
            "FullPrecisionFloatVectorSimilarityValuesSource(fieldName={} \
             vectorSimilarityFunction={} queryVector={:?})",
            self.field,
            self.function
                .map_or("null".to_string(), |f| format!("{f:?}")),
            self.query
        )
    }
}

/// `new FullPrecisionFloatVectorSimilarityValuesSource(vector, field[,
/// function])`.
pub fn full_precision_float_vector_similarity(
    field: &str,
    query: Vec<f32>,
    function: Option<VectorSimilarityFunction>,
) -> Arc<dyn DoubleValuesSource> {
    Arc::new(FullPrecisionSource {
        field: field.to_string(),
        query,
        function,
    })
}

// ---------------------------------------------------------------------------
// Late interaction
// ---------------------------------------------------------------------------

/// `MultiVectorSimilarity`: a score between two multi-vectors.
pub trait MultiVectorSimilarity: Send + Sync {
    /// `compare(queryVector, docVector, vectorSimilarityFunction)`.
    fn compare(
        &self,
        query: &[Vec<f32>],
        doc: &[Vec<f32>],
        function: VectorSimilarityFunction,
    ) -> Result<f32>;
}

/// `LateInteractionFloatValuesSource.ScoreFunction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreFunction {
    /// `SUM_MAX_SIM`: over the query's token vectors, the sum of each one's
    /// best similarity to any of the document's (`Float.MIN_VALUE` for a
    /// document with none).
    SumMaxSim,
}

impl MultiVectorSimilarity for ScoreFunction {
    fn compare(
        &self,
        query: &[Vec<f32>],
        doc: &[Vec<f32>],
        function: VectorSimilarityFunction,
    ) -> Result<f32> {
        if doc.is_empty() {
            return Ok(f32::from_bits(1));
        }
        let mut result = 0f32;
        for q in query {
            let mut max_sim = f32::from_bits(1);
            for d in doc {
                if q.len() != d.len() {
                    return Err(Error::IllegalArgument(format!(
                        "Provided multi-vectors are incompatible. Their composing token vectors \
                         should have the same dimension, got {} != {}",
                        q.len(),
                        d.len()
                    )));
                }
                max_sim = java_float_max(max_sim, function.score(q, d));
            }
            result += max_sim;
        }
        Ok(result)
    }
}

/// `Float.max`: NaN wins, and `+0.0` beats `-0.0`.
fn java_float_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else if a >= b {
        a
    } else {
        b
    }
}

/// `LateInteractionField.decode(payload)`: a little-endian `int` token
/// dimension, then the token vectors' floats.
pub fn decode_multi_vector(payload: &[u8]) -> Result<Vec<Vec<f32>>> {
    let bad = || {
        Error::IllegalArgument(
            "Provided payload does not appear to have been encoded via \
             LateInteractionField.encode"
                .to_string(),
        )
    };
    let head: [u8; 4] = payload
        .get(..4)
        .and_then(|h| h.try_into().ok())
        .ok_or_else(bad)?;
    let dim = usize::try_from(i32::from_le_bytes(head)).map_err(|_| bad())?;
    let body = &payload[4..];
    let width = dim.checked_mul(4).filter(|&w| w > 0).ok_or_else(bad)?;
    if !body.len().is_multiple_of(width) {
        return Err(bad());
    }
    Ok(body
        .chunks_exact(width)
        .map(|token| {
            token
                .chunks_exact(4)
                .map(|f| f32::from_le_bytes([f[0], f[1], f[2], f[3]]))
                .collect()
        })
        .collect())
}

/// `LateInteractionField.encode(value)`.
pub fn encode_multi_vector(value: &[Vec<f32>]) -> Result<Vec<u8>> {
    let first = value
        .first()
        .ok_or_else(|| Error::IllegalArgument("Value should not be null or empty".to_string()))?;
    if first.is_empty() {
        return Err(Error::IllegalArgument(
            "Composing token vectors should not be null or empty".to_string(),
        ));
    }
    let dim = first.len();
    let mut out = Vec::with_capacity(4 + value.len() * dim * 4);
    out.extend_from_slice(
        &i32::try_from(dim)
            .map_err(|_| Error::IllegalArgument("dimension".to_string()))?
            .to_le_bytes(),
    );
    for (i, token) in value.iter().enumerate() {
        if token.len() != dim {
            return Err(Error::IllegalArgument(format!(
                "Composing token vectors should have the same dimension. Mismatching dimensions \
                 detected between token[0] and token[{i}], {dim} != {}",
                token.len()
            )));
        }
        for f in token {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }
    Ok(out)
}

/// `LateInteractionFloatValuesSource`.
pub struct LateInteractionFloatValuesSource {
    field: String,
    query: Vec<Vec<f32>>,
    function: VectorSimilarityFunction,
    score_function: Arc<dyn MultiVectorSimilarity>,
}

impl LateInteractionFloatValuesSource {
    /// `new LateInteractionFloatValuesSource(field, queryVector, function,
    /// scoreFunction)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an empty query, an empty first token
    /// vector, or token vectors of different lengths.
    pub fn new(
        field: &str,
        query: Vec<Vec<f32>>,
        function: VectorSimilarityFunction,
        score_function: Arc<dyn MultiVectorSimilarity>,
    ) -> Result<Self> {
        let first = query.first().ok_or_else(|| {
            Error::IllegalArgument("queryVector must not be null or empty".to_string())
        })?;
        if first.is_empty() {
            return Err(Error::IllegalArgument(
                "composing token vectors in provided query vector should not be null or empty"
                    .to_string(),
            ));
        }
        if query.iter().any(|q| q.len() != first.len()) {
            return Err(Error::IllegalArgument(
                "all composing token vectors in provided query vector should have the same length"
                    .to_string(),
            ));
        }
        Ok(Self {
            field: field.to_string(),
            query,
            function,
            score_function,
        })
    }

    /// `new LateInteractionFloatValuesSource(field, queryVector[, function])`:
    /// `COSINE` and `SUM_MAX_SIM` by default.
    pub fn with_function(
        field: &str,
        query: Vec<Vec<f32>>,
        function: VectorSimilarityFunction,
    ) -> Result<Self> {
        Self::new(field, query, function, Arc::new(ScoreFunction::SumMaxSim))
    }
}

struct LateInteractionValues<'c> {
    column: Option<lucene_codecs::doc_values::BinaryReader<'c>>,
    current: Option<&'c [u8]>,
    query: Vec<Vec<f32>>,
    function: VectorSimilarityFunction,
    score_function: Arc<dyn MultiVectorSimilarity>,
}

impl DoubleValues for LateInteractionValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.current = match self.column.as_mut() {
            Some(c) => c.value(doc)?,
            None => None,
        };
        Ok(self.current.is_some())
    }
    fn double_value(&mut self) -> Result<f64> {
        let doc = decode_multi_vector(self.current.unwrap_or_default())?;
        Ok(f64::from(self.score_function.compare(
            &self.query,
            &doc,
            self.function,
        )?))
    }
}

impl DoubleValuesSource for LateInteractionFloatValuesSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        let reader = ctx.reader(leaf)?;
        // `getBinaryDocValues(field)`: null (EMPTY) unless the field has
        // BINARY doc values.
        let column = reader
            .field_infos()
            .field_by_name(&self.field)
            .filter(|i| i.doc_values_type == DocValuesType::Binary)
            .and_then(|i| {
                let (meta, data) = reader.doc_values_for_field(i.number)?;
                meta.binary_entry(i.number)
                    .map(|e| lucene_codecs::doc_values::BinaryReader::new(data, e))
            });
        if column.is_none() {
            return Ok(Box::new(EmptyDoubleValues));
        }
        Ok(Box::new(LateInteractionValues {
            column,
            current: None,
            query: self.query.clone(),
            function: self.function,
            score_function: Arc::clone(&self.score_function),
        }))
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        true
    }
    fn describe(&self) -> String {
        format!(
            "LateInteractionFloatValuesSource(fieldName={} similarityFunction={:?} \
             queryVector={:?})",
            self.field, self.function, self.query
        )
    }
}

// ---------------------------------------------------------------------------
// Sorting by a values source
// ---------------------------------------------------------------------------

/// The score a sort's comparator hands a source that needs scores
/// (`DoubleValuesSource.fromScorer(scorer)`).
struct ScoreCell(std::rc::Rc<std::cell::Cell<f32>>);

impl DoubleValues for ScoreCell {
    fn advance_exact(&mut self, _doc: i32) -> Result<bool> {
        Ok(true)
    }
    fn double_value(&mut self) -> Result<f64> {
        Ok(f64::from(self.0.get()))
    }
}

/// `DoubleValuesComparatorSource`/`LongValuesComparatorSource`: a sort key
/// over a values source, compared as Java's `DoubleComparator`
/// (`Double.compare`, over `doubleToSortableLong`) or `LongComparator`, with
/// `missing` for a document without a value.
struct ValuesSortSource {
    source: ValuesSortKind,
    missing: i64,
}

#[derive(Clone)]
enum ValuesSortKind {
    Double(Arc<dyn DoubleValuesSource>),
    Long(Arc<dyn LongValuesSource>),
}

impl crate::top_field::FieldComparatorSource for ValuesSortSource {
    fn new_comparator(
        &self,
        _field: &str,
        _num_hits: usize,
        _reverse: bool,
    ) -> Box<dyn crate::top_field::FieldComparator> {
        Box::new(ValuesSortComparator {
            source: self.source.clone(),
            missing: self.missing,
        })
    }

    /// `DoubleValuesSortField.rewrite(searcher)` (and the long one's): a
    /// source that needs the searcher or the opened vectors
    /// ([`DoubleValuesSource::needs_searcher`]) is rewritten against `ctx`
    /// -- Java binds a query-backed source to its `Weight` there -- here by
    /// reading each leaf's values once, so the comparator needs only the
    /// document's leaf. A source that also needs the scores is left as it is
    /// (its values depend on each hit's score).
    fn rewrite(
        &self,
        ctx: &ValuesContext<'_>,
    ) -> Result<Option<Arc<dyn crate::top_field::FieldComparatorSource>>> {
        let (needs_searcher, needs_scores) = match &self.source {
            ValuesSortKind::Double(s) => (s.needs_searcher(), s.needs_scores()),
            ValuesSortKind::Long(s) => (s.needs_searcher(), s.needs_scores()),
        };
        if !needs_searcher || needs_scores {
            return Ok(None);
        }
        let searcher = ctx.searcher()?;
        let mut leaves = HashMap::new();
        for (leaf, seg) in searcher.segments().iter().enumerate() {
            let max_doc = ctx.reader(leaf)?.max_doc;
            let mut values = vec![None; usize::try_from(max_doc).unwrap_or(0)];
            match &self.source {
                ValuesSortKind::Double(s) => {
                    let mut v = s.get_values(ctx, leaf, None)?;
                    for (doc, slot) in (0..max_doc).zip(values.iter_mut()) {
                        if v.advance_exact(doc)? {
                            *slot = Some(double_to_sortable_long(v.double_value()?));
                        }
                    }
                }
                ValuesSortKind::Long(s) => {
                    let mut v = s.get_values(ctx, leaf, None)?;
                    for (doc, slot) in (0..max_doc).zip(values.iter_mut()) {
                        if v.advance_exact(doc)? {
                            *slot = Some(v.long_value()?);
                        }
                    }
                }
            }
            leaves.insert(seg.doc_base, Arc::new(values));
        }
        Ok(Some(Arc::new(RewrittenValuesSortSource {
            leaves: Arc::new(leaves),
            missing: self.missing,
        })))
    }
}

/// A values-source sort key rewritten against a searcher: each leaf's
/// values (the comparable long a hit carries), by the leaf's `docBase`.
struct RewrittenValuesSortSource {
    leaves: Arc<HashMap<i32, Arc<Vec<Option<i64>>>>>,
    missing: i64,
}

impl crate::top_field::FieldComparatorSource for RewrittenValuesSortSource {
    fn new_comparator(
        &self,
        _field: &str,
        _num_hits: usize,
        _reverse: bool,
    ) -> Box<dyn crate::top_field::FieldComparator> {
        Box::new(RewrittenValuesSortSource {
            leaves: Arc::clone(&self.leaves),
            missing: self.missing,
        })
    }
}

impl crate::top_field::FieldComparator for RewrittenValuesSortSource {
    fn leaf<'a>(
        &self,
        ctx: crate::top_field::LeafCtx<'a>,
    ) -> Result<Box<dyn crate::top_field::LeafFieldComparator + 'a>> {
        let values = self.leaves.get(&ctx.doc_base).cloned().ok_or_else(|| {
            Error::IllegalState(format!(
                "the sort was rewritten against another reader: no leaf at docBase {}",
                ctx.doc_base
            ))
        })?;
        Ok(Box::new(RewrittenValuesLeaf {
            values,
            missing: self.missing,
        }))
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

struct RewrittenValuesLeaf {
    values: Arc<Vec<Option<i64>>>,
    missing: i64,
}

impl crate::top_field::LeafFieldComparator for RewrittenValuesLeaf {
    fn value(&mut self, doc: i32, _score: f32) -> Result<crate::top_field::SortValue> {
        let v = usize::try_from(doc)
            .ok()
            .and_then(|d| self.values.get(d).copied().flatten())
            .unwrap_or(self.missing);
        Ok(crate::top_field::SortValue::Long(v))
    }
}

struct ValuesSortComparator {
    source: ValuesSortKind,
    missing: i64,
}

struct ValuesSortLeaf<'a> {
    score: std::rc::Rc<std::cell::Cell<f32>>,
    values: ValuesSortLeafValues<'a>,
    missing: i64,
}

enum ValuesSortLeafValues<'a> {
    Double(BoxDoubleValues<'a>),
    Long(BoxLongValues<'a>),
}

impl crate::top_field::FieldComparator for ValuesSortComparator {
    fn leaf<'a>(
        &self,
        ctx: crate::top_field::LeafCtx<'a>,
    ) -> Result<Box<dyn crate::top_field::LeafFieldComparator + 'a>> {
        let vctx = ValuesContext::for_reader(ctx.reader);
        let score = std::rc::Rc::new(std::cell::Cell::new(0f32));
        let scores = || -> Option<BoxDoubleValues<'a>> {
            Some(Box::new(ScoreCell(std::rc::Rc::clone(&score))))
        };
        let values = match &self.source {
            ValuesSortKind::Double(s) => ValuesSortLeafValues::Double(s.get_values(
                &vctx,
                0,
                s.needs_scores().then(scores).flatten(),
            )?),
            ValuesSortKind::Long(s) => ValuesSortLeafValues::Long(s.get_values(
                &vctx,
                0,
                s.needs_scores().then(scores).flatten(),
            )?),
        };
        Ok(Box::new(ValuesSortLeaf {
            score,
            values,
            missing: self.missing,
        }))
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

    fn needs_scores(&self) -> bool {
        match &self.source {
            ValuesSortKind::Double(s) => s.needs_scores(),
            ValuesSortKind::Long(s) => s.needs_scores(),
        }
    }
}

impl crate::top_field::LeafFieldComparator for ValuesSortLeaf<'_> {
    fn value(&mut self, doc: i32, score: f32) -> Result<crate::top_field::SortValue> {
        self.score.set(score);
        let v = match &mut self.values {
            ValuesSortLeafValues::Double(v) => {
                if v.advance_exact(doc)? {
                    double_to_sortable_long(v.double_value()?)
                } else {
                    self.missing
                }
            }
            ValuesSortLeafValues::Long(v) => {
                if v.advance_exact(doc)? {
                    v.long_value()?
                } else {
                    self.missing
                }
            }
        };
        Ok(crate::top_field::SortValue::Long(v))
    }
}

/// `DoubleValuesSource.getSortField(reverse, missingValue)`: a sort key over
/// the source's values (a hit's value is `doubleToSortableLong` of it). A
/// source that needs the searcher or the vectors (a query-backed or vector
/// source, [`DoubleValuesSource::needs_searcher`]) sorts once the sort is
/// rewritten against them ([`crate::top_field::rewrite_sort`], which
/// `IndexSearcher.search(query, n, sort)` does in Java).
pub fn double_sort_field(
    source: Arc<dyn DoubleValuesSource>,
    reverse: bool,
    missing: f64,
) -> crate::top_field::SortField {
    let field = source.describe();
    let id = crate::top_field::register_comparator_source(Arc::new(ValuesSortSource {
        source: ValuesSortKind::Double(source),
        missing: double_to_sortable_long(missing),
    }));
    crate::top_field::SortField::custom(&field, id, reverse)
}

/// `LongValuesSource.getSortField(reverse, missingValue)`.
pub fn long_sort_field(
    source: Arc<dyn LongValuesSource>,
    reverse: bool,
    missing: i64,
) -> crate::top_field::SortField {
    let field = source.describe();
    let id = crate::top_field::register_comparator_source(Arc::new(ValuesSortSource {
        source: ValuesSortKind::Long(source),
        missing,
    }));
    crate::top_field::SortField::custom(&field, id, reverse)
}

// ---------------------------------------------------------------------------
// NumericFieldStats
// ---------------------------------------------------------------------------

/// `NumericFieldStats.Stats`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub min: i64,
    pub max: i64,
    pub doc_count: i32,
}

/// `NumericFieldStats.getStats(reader, field)`: the field's global minimum,
/// maximum and document count from its points (a value of at most 8 bytes),
/// else from its doc-values skip index; `None` when neither answers.
///
/// Needs each segment's points opened (`OpenedSegments::open_points`) and its
/// reader.
pub fn numeric_field_stats(searcher: &IndexSearcher<'_, '_>, field: &str) -> Result<Option<Stats>> {
    if let Some(s) = stats_from_points(searcher, field) {
        return Ok(Some(s));
    }
    stats_from_skipper(searcher, field)
}

/// `PointValues.getMinPackedValue`/`getMaxPackedValue`/`getDocCount` over the
/// reader, then `decodeLong`.
fn stats_from_points(searcher: &IndexSearcher<'_, '_>, field: &str) -> Option<Stats> {
    let mut min: Option<Vec<u8>> = None;
    let mut max: Option<Vec<u8>> = None;
    let mut doc_count: i32 = 0;
    for seg in searcher.segments() {
        let Some(points) = seg.points else { continue };
        let Some(f) = points
            .field_number(field)
            .and_then(|n| points.reader.field(n))
        else {
            continue;
        };
        let bytes = usize::try_from(f.bytes_per_dim).ok()?;
        let dims = usize::try_from(f.num_index_dims).ok()?;
        doc_count = doc_count.saturating_add(f.doc_count);
        let merge = |acc: &mut Option<Vec<u8>>, v: &[u8], want_less: bool| match acc {
            None => *acc = Some(v.to_vec()),
            Some(a) => {
                for d in 0..dims {
                    let r = d * bytes..(d + 1) * bytes;
                    let (x, y) = (&v[r.clone()], &a[r.clone()]);
                    if (want_less && x < y) || (!want_less && x > y) {
                        a[r].copy_from_slice(x);
                    }
                }
            }
        };
        merge(&mut min, &f.min_packed_value, true);
        merge(&mut max, &f.max_packed_value, false);
    }
    let (min, max) = (min?, max?);
    if min.len() > 8 || max.len() > 8 {
        return None;
    }
    Some(Stats {
        min: decode_long(&min),
        max: decode_long(&max),
        doc_count,
    })
}

/// `NumericFieldStats.decodeLong`: the first byte's sign flipped, then the
/// rest big-endian.
fn decode_long(packed: &[u8]) -> i64 {
    let Some((&first, rest)) = packed.split_first() else {
        return 0;
    };
    let mut result = i64::from((first ^ 0x80) as i8);
    for &b in rest {
        result = (result << 8) | i64::from(b);
    }
    result
}

fn stats_from_skipper(searcher: &IndexSearcher<'_, '_>, field: &str) -> Result<Option<Stats>> {
    let mut acc: Option<(i64, i64)> = None;
    let mut doc_count: i32 = 0;
    for (i, seg) in searcher.segments().iter().enumerate() {
        let reader = seg.reader.ok_or_else(|| {
            Error::IllegalState(format!(
                "leaf {i}: NumericFieldStats needs the segment's reader"
            ))
        })?;
        let Some(info) = reader.field_infos().field_by_name(field) else {
            continue;
        };
        let skipper = reader
            .doc_values_meta()
            .and_then(|m| m.skipper_meta(info.number));
        let Some(skipper) = skipper else {
            return Ok(None);
        };
        acc = Some(match acc {
            None => (skipper.min_value, skipper.max_value),
            Some((lo, hi)) => (lo.min(skipper.min_value), hi.max(skipper.max_value)),
        });
        doc_count = doc_count.saturating_add(skipper.doc_count);
    }
    Ok(acc.map(|(min, max)| Stats {
        min,
        max,
        doc_count,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_vector_round_trip_and_errors() {
        let v = vec![vec![1.0, -2.0], vec![0.5, 3.0]];
        let bytes = encode_multi_vector(&v).unwrap();
        assert_eq!(decode_multi_vector(&bytes).unwrap(), v);
        assert!(encode_multi_vector(&[]).is_err());
        assert!(encode_multi_vector(&[vec![]]).is_err());
        assert!(encode_multi_vector(&[vec![1.0], vec![1.0, 2.0]]).is_err());
        assert!(decode_multi_vector(&[1, 0]).is_err());
        assert!(decode_multi_vector(&[0, 0, 0, 0]).is_err());
        let mut odd = bytes.clone();
        odd.push(0);
        assert!(decode_multi_vector(&odd).is_err());
    }

    #[test]
    fn sum_max_sim() {
        let f = ScoreFunction::SumMaxSim;
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(
            f.compare(&q, &[], VectorSimilarityFunction::DotProduct)
                .unwrap(),
            f32::from_bits(1)
        );
        let d = vec![vec![1.0, 0.0]];
        // dot: (1+1)/2 = 1 for the first token, (1+0)/2 = 0.5 for the second.
        assert_eq!(
            f.compare(&q, &d, VectorSimilarityFunction::DotProduct)
                .unwrap(),
            1.5
        );
        assert!(f
            .compare(&q, &[vec![1.0]], VectorSimilarityFunction::DotProduct)
            .is_err());
        assert!(LateInteractionFloatValuesSource::with_function(
            "f",
            vec![],
            VectorSimilarityFunction::Cosine
        )
        .is_err());
        assert!(LateInteractionFloatValuesSource::with_function(
            "f",
            vec![vec![]],
            VectorSimilarityFunction::Cosine
        )
        .is_err());
        assert!(LateInteractionFloatValuesSource::with_function(
            "f",
            vec![vec![1.0], vec![1.0, 2.0]],
            VectorSimilarityFunction::Cosine
        )
        .is_err());
    }

    #[test]
    fn float_max_is_javas() {
        assert!(java_float_max(f32::NAN, 1.0).is_nan());
        assert_eq!(java_float_max(-0.0, 0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(java_float_max(0.0, -0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(java_float_max(2.0, 1.0), 2.0);
        assert_eq!(java_float_max(1.0, 2.0), 2.0);
    }

    #[test]
    fn decode_long_is_numeric_field_stats() {
        assert_eq!(decode_long(&[0x80, 0, 0, 0, 0, 0, 0, 5]), 5);
        assert_eq!(decode_long(&[0x7f, 0xff, 0xff, 0xff]), -1);
        assert_eq!(decode_long(&[0x80, 0, 0, 7]), 7);
        assert_eq!(decode_long(&[]), 0);
    }

    #[test]
    fn decoders_and_conversions() {
        assert_eq!(Decoder::Long.apply(-3), -3.0);
        assert_eq!(Decoder::Double.apply(2.5f64.to_bits() as i64), 2.5);
        assert_eq!(
            Decoder::Float.apply(i64::from(1.5f32.to_bits() as i32)),
            1.5
        );
        assert_eq!(Decoder::Custom(Arc::new(|v| v as f64 * 2.0)).apply(4), 8.0);
        assert_eq!(double_to_sortable_long(0.0), 0);
        assert!(double_to_sortable_long(-1.0) < double_to_sortable_long(-0.0));
        assert!(double_to_sortable_long(f64::NAN) > double_to_sortable_long(f64::INFINITY));
        assert_eq!(java_double(3.0), "3.0");
        assert_eq!(java_double(0.25), "0.25");
        assert_eq!(constant(3.0).describe(), "constant(3.0)");
        assert_eq!(scores().describe(), "scores");
        assert!(scores().needs_scores());
        assert_eq!(from_long_field("n").describe(), "double(n)");
        assert_eq!(long_from_int_field("n").describe(), "long(n)");
        assert_eq!(
            to_long_values_source(from_float_field("f")).describe(),
            "long(double(f))"
        );
        assert_eq!(
            to_sortable_long_values_source(from_double_field("d")).describe(),
            "sortableLong(double(d))"
        );
        assert_eq!(
            to_double_values_source(long_constant(4)).describe(),
            "double(constant(4))"
        );
    }

    #[test]
    fn empty_and_constant_values() {
        let mut e = EmptyDoubleValues;
        assert!(!e.advance_exact(0).unwrap());
        assert_eq!(e.double_value().unwrap(), 0.0);
        let mut c = ConstantDoubleValues(2.0);
        assert!(c.advance_exact(9).unwrap());
        assert_eq!(c.double_value().unwrap(), 2.0);
        let mut l = ConstantLongValues(7);
        assert!(l.advance_exact(1).unwrap());
        assert_eq!(l.long_value().unwrap(), 7);
    }
}
