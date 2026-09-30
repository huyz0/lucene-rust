//! Second-pass rescoring: `Rescorer`, `QueryRescorer`, `SortRescorer`,
//! `DoubleValuesSourceRescorer`, `LateInteractionRescorer`, and
//! `RescoreTopNQuery`'s rewrite.
//!
//! Each takes the first pass's [`TopDocs`] (global document ids, as
//! [`IndexSearcher::search`] returns them) and a [`ValuesContext`] over the
//! searcher that produced them, and follows Lucene 10.5.0's algorithm step
//! for step: the hits walked in document order leaf by leaf, the second
//! pass read per hit, the combination in Java's arithmetic (a `float` plus a
//! `double` product narrowed once), then Java's final ordering -- including
//! `SortRescorer`'s reassignment of first-pass scores by document rank, which
//! pairs scores with the wrong documents once `topN` drops a hit (kept, since
//! it is what Lucene returns).
//!
//! Verified against Lucene by `tests/rescore_fixtures.rs`
//! (`fixtures/src/GenValuesRescore.java`).

use std::cmp::Ordering;
use std::sync::Arc;

use lucene_codecs::doc_values::{NumericReader, SortedNumericReader};
use lucene_codecs::field_infos::{DocValuesType, VectorSimilarityFunction};
use lucene_codecs::terms_dict::TermsDict;

use crate::collector::{TotalHits, TotalHitsRelation};
use crate::explain::Explanation;
use crate::index_searcher::IndexSearcher;
use crate::query::BooleanQuery;
use crate::top_docs::{ShardFieldDoc, ShardScoreDoc, ShardTopFieldDocs, TopDocs};
use crate::top_field::{compare_keys, FieldDoc, SortField, SortType};
use crate::values_source::{
    DoubleValuesSource, LateInteractionFloatValuesSource, ScoreFunction, ValuesContext,
};
use crate::{Error, Result};

/// `Rescorer`.
pub trait Rescorer {
    /// `rescore(searcher, firstPassTopDocs, topN)`.
    fn rescore(
        &self,
        ctx: &ValuesContext<'_>,
        first_pass: &TopDocs,
        top_n: usize,
    ) -> Result<TopDocs>;

    /// `explain(searcher, firstPassExplanation, docID)`.
    fn explain(
        &self,
        ctx: &ValuesContext<'_>,
        first_pass: &Explanation,
        doc: i32,
    ) -> Result<Explanation>;
}

/// The leaf holding global `doc` and its local id, walking forward from
/// `leaf` as Lucene's rescorers do (`while (docID >= endDoc) readerUpto++`).
fn leaf_of(searcher: &IndexSearcher<'_, '_>, doc: i32) -> Result<(usize, i32)> {
    let leaf = searcher.segment_of(doc).ok_or_else(|| {
        Error::IllegalState(format!(
            "hit docId={doc} is not in any leaf of the searcher; ensure firstPassTopDocs were \
             produced by the searcher provided to rescore"
        ))
    })?;
    Ok((leaf, doc - searcher.segments()[leaf].doc_base))
}

/// `ScoreDoc.COMPARATOR` (and `QueryRescorer`'s identical comparator): the
/// higher score first, else the lower document. Java's `>`/`<` leave a NaN
/// score unordered against every other; it sorts here as the lowest.
fn by_score_then_doc(a: &ShardScoreDoc, b: &ShardScoreDoc) -> Ordering {
    if a.score > b.score {
        Ordering::Less
    } else if a.score < b.score {
        Ordering::Greater
    } else if a.score.is_nan() != b.score.is_nan() {
        if a.score.is_nan() {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    } else {
        a.doc.cmp(&b.doc)
    }
}

/// `ArrayUtil.select(hits, 0, len, topN, cmp)` + `copyOfSubArray` +
/// `Arrays.sort`: the best `top_n`, in order.
fn select_sorted(mut hits: Vec<ShardScoreDoc>, top_n: usize) -> Vec<ShardScoreDoc> {
    hits.sort_by(by_score_then_doc);
    hits.truncate(top_n);
    hits
}

/// The combination a [`QueryRescorer`] applies: `combine(firstPassScore,
/// secondPassMatches, secondPassScore)`.
pub type QueryCombine = dyn Fn(f32, bool, f32) -> f32 + Send + Sync;

/// `QueryRescorer`: a second query's score for each first-pass hit,
/// combined with the first by [`QueryCombine`].
pub struct QueryRescorer {
    query: BooleanQuery,
    combine: Arc<QueryCombine>,
}

impl QueryRescorer {
    /// `new QueryRescorer(query) { combine(...) }`.
    pub fn new(query: BooleanQuery, combine: Arc<QueryCombine>) -> Self {
        Self { query, combine }
    }

    /// The combination of `QueryRescorer.rescore(searcher, topDocs, query,
    /// weight, topN)`: `score += weight * secondPassScore` when the second
    /// pass matches -- a `float` plus a `double`, narrowed once.
    pub fn with_weight(query: BooleanQuery, weight: f64) -> Self {
        Self::new(
            query,
            Arc::new(move |first: f32, matches: bool, second: f32| {
                if matches {
                    (f64::from(first) + weight * f64::from(second)) as f32
                } else {
                    first
                }
            }),
        )
    }

    /// `QueryRescorer.rescore(searcher, topDocs, query, weight, topN)`.
    pub fn rescore_with_weight(
        ctx: &ValuesContext<'_>,
        top_docs: &TopDocs,
        query: BooleanQuery,
        weight: f64,
        top_n: usize,
    ) -> Result<TopDocs> {
        Self::with_weight(query, weight).rescore(ctx, top_docs, top_n)
    }

    /// The second pass's score of global `doc`, if the query matches it.
    fn second_pass(&self, searcher: &IndexSearcher<'_, '_>, doc: i32) -> Result<Option<f32>> {
        let (leaf, local) = leaf_of(searcher, doc)?;
        let scores = searcher.leaf_scores(&self.query, leaf, true)?;
        Ok(scores
            .binary_search_by_key(&local, |&(d, _)| d)
            .ok()
            .map(|i| scores[i].1))
    }
}

impl Rescorer for QueryRescorer {
    fn rescore(
        &self,
        ctx: &ValuesContext<'_>,
        first_pass: &TopDocs,
        top_n: usize,
    ) -> Result<TopDocs> {
        let searcher = ctx.searcher()?;
        let mut hits = first_pass.score_docs.clone();
        hits.sort_by_key(|h| h.doc);
        // One scorer per leaf, advanced through the leaf's hits in order.
        let mut current: Option<(usize, Vec<(i32, f32)>)> = None;
        for hit in &mut hits {
            let (leaf, local) = leaf_of(searcher, hit.doc)?;
            if current.as_ref().is_none_or(|(l, _)| *l != leaf) {
                current = Some((leaf, searcher.leaf_scores(&self.query, leaf, true)?));
            }
            let scores = current.as_ref().map(|(_, s)| s.as_slice()).unwrap_or(&[]);
            hit.score = match scores.binary_search_by_key(&local, |&(d, _)| d) {
                Ok(i) => (self.combine)(hit.score, true, scores[i].1),
                Err(_) => (self.combine)(hit.score, false, 0.0),
            };
        }
        Ok(TopDocs {
            total_hits: first_pass.total_hits,
            score_docs: select_sorted(hits, top_n),
        })
    }

    fn explain(
        &self,
        ctx: &ValuesContext<'_>,
        first_pass: &Explanation,
        doc: i32,
    ) -> Result<Explanation> {
        let second = self.second_pass(ctx.searcher()?, doc)?;
        let score = match second {
            Some(s) => (self.combine)(first_pass.value, true, s),
            None => (self.combine)(first_pass.value, false, 0.0),
        };
        let first = Explanation::match_(first_pass.value, "first pass score")
            .with_details(vec![first_pass.clone()]);
        let second = match second {
            Some(s) => Explanation::match_(s, "second pass score"),
            None => Explanation::no_match("no second pass score"),
        };
        Ok(Explanation::match_(
            score,
            "combined first and second pass score using QueryRescorer",
        )
        .with_details(vec![first, second]))
    }
}

/// The combination a [`DoubleValuesSourceRescorer`] applies:
/// `combine(firstPassScore, valuePresent, sourceValue)`.
pub type ValuesCombine = dyn Fn(f32, bool, f64) -> f32 + Send + Sync;

/// `DoubleValuesSourceRescorer`: a values source's value for each hit,
/// combined with the first-pass score.
pub struct DoubleValuesSourceRescorer {
    source: Arc<dyn DoubleValuesSource>,
    combine: Arc<ValuesCombine>,
}

impl DoubleValuesSourceRescorer {
    pub fn new(source: Arc<dyn DoubleValuesSource>, combine: Arc<ValuesCombine>) -> Self {
        Self { source, combine }
    }
}

impl Rescorer for DoubleValuesSourceRescorer {
    /// Java opens the leaf's values once per hit; here once per leaf, with
    /// the hits in document order -- the values read are the same.
    fn rescore(
        &self,
        ctx: &ValuesContext<'_>,
        first_pass: &TopDocs,
        top_n: usize,
    ) -> Result<TopDocs> {
        let searcher = ctx.searcher()?;
        let mut hits = first_pass.score_docs.clone();
        hits.sort_by_key(|h| h.doc);
        let mut current: Option<(usize, crate::values_source::BoxDoubleValues<'_>)> = None;
        let mut last_local = -1;
        for hit in &mut hits {
            let (leaf, local) = leaf_of(searcher, hit.doc)?;
            // A fresh cursor per leaf, and per repeated document (a cursor
            // only moves forward).
            if current.as_ref().is_none_or(|(l, _)| *l != leaf) || local <= last_local {
                current = Some((leaf, self.source.get_values(ctx, leaf, None)?));
            }
            last_local = local;
            let Some((_, values)) = current.as_mut() else {
                continue;
            };
            let present = values.advance_exact(local)?;
            let value = if present { values.double_value()? } else { 0.0 };
            hit.score = (self.combine)(hit.score, present, value);
        }
        Ok(TopDocs {
            total_hits: first_pass.total_hits,
            score_docs: select_sorted(hits, top_n),
        })
    }

    fn explain(
        &self,
        ctx: &ValuesContext<'_>,
        first_pass: &Explanation,
        doc: i32,
    ) -> Result<Explanation> {
        let searcher = ctx.searcher()?;
        let (leaf, local) = searcher
            .segment_of(doc)
            .map(|l| (l, doc - searcher.segments()[l].doc_base))
            .ok_or_else(|| {
                Error::IllegalArgument(format!(
                    "docId={doc} not found in any leaf in provided searcher"
                ))
            })?;
        let first = Explanation::match_(first_pass.value, "first pass score")
            .with_details(vec![first_pass.clone()]);
        let values = self.source.explain(
            ctx,
            leaf,
            local,
            &Explanation::no_match("DoubleValuesSource was not initialized with query scores"),
        )?;
        let second = if values.matched {
            Explanation::match_(values.value, "value from DoubleValuesSource")
                .with_details(vec![values.clone()])
        } else {
            Explanation::no_match("no value in DoubleValuesSource")
        };
        let score = (self.combine)(first_pass.value, values.matched, f64::from(values.value));
        Ok(Explanation::match_(
            score,
            format!(
                "combined score from firstPass and DoubleValuesSource={} using \
                 DoubleValuesSourceRescorer",
                self.source.describe()
            ),
        )
        .with_details(vec![first, second]))
    }
}

/// `LateInteractionRescorer.create(field, queryVector[, function])`: the
/// late-interaction similarity replaces the first-pass score; a hit without
/// a multi-vector scores `0`.
pub fn late_interaction_rescorer(
    field: &str,
    query: Vec<Vec<f32>>,
    function: VectorSimilarityFunction,
) -> Result<DoubleValuesSourceRescorer> {
    let source = LateInteractionFloatValuesSource::new(
        field,
        query,
        function,
        Arc::new(ScoreFunction::SumMaxSim),
    )?;
    Ok(DoubleValuesSourceRescorer::new(
        Arc::new(source),
        Arc::new(
            |_first: f32, present: bool, value: f64| {
                if present {
                    value as f32
                } else {
                    0.0
                }
            },
        ),
    ))
}

/// `LateInteractionRescorer.withFallbackToFirstPassScore(...)`: a hit without
/// a multi-vector keeps its first-pass score.
pub fn late_interaction_rescorer_with_fallback(
    field: &str,
    query: Vec<Vec<f32>>,
    function: VectorSimilarityFunction,
) -> Result<DoubleValuesSourceRescorer> {
    let source = LateInteractionFloatValuesSource::with_function(field, query, function)?;
    Ok(DoubleValuesSourceRescorer::new(
        Arc::new(source),
        Arc::new(
            |first: f32, present: bool, value: f64| {
                if present {
                    value as f32
                } else {
                    first
                }
            },
        ),
    ))
}

// ---------------------------------------------------------------------------
// SortRescorer
// ---------------------------------------------------------------------------

fn store_err(e: lucene_store::Error) -> Error {
    Error::from(lucene_codecs::doc_values::Error::from(e))
}

/// One sort key's value for leaf-local `doc` of `leaf`, in
/// [`crate::top_field`]'s encoding: the comparable long, and for a keyword
/// key the term (`None` when the document has none).
fn key_value(
    searcher: &IndexSearcher<'_, '_>,
    leaf: usize,
    doc: i32,
    key: &SortField,
    score: f32,
) -> Result<(i64, Option<Vec<u8>>)> {
    let seg = &searcher.segments()[leaf];
    match key.ty {
        SortType::Score => return Ok((i64::from(score.to_bits()), None)),
        SortType::Doc => return Ok((i64::from(seg.doc_base + doc), None)),
        _ => {}
    }
    let reader = seg.reader.ok_or_else(|| {
        Error::IllegalState(format!(
            "leaf {leaf}: a field sort needs the segment's reader"
        ))
    })?;
    if let SortType::Custom(id) = key.ty {
        let c = crate::top_field::custom_comparator(id, key, 1)?;
        let mut l = c.leaf(crate::top_field::LeafCtx {
            reader,
            doc_base: seg.doc_base,
        })?;
        return Ok(match l.value(doc, score)? {
            crate::top_field::SortValue::Long(v) => (v, None),
            crate::top_field::SortValue::Bytes(b) => (0, b),
        });
    }
    let info = reader.field_infos().field_by_name(&key.field);
    let column = info.and_then(|i| reader.doc_values_for_field(i.number).map(|c| (i, c)));
    if key.ty == SortType::StringVal {
        let Some((info, (meta, data))) = column else {
            return Ok((0, None));
        };
        return match info.doc_values_type {
            DocValuesType::Binary => {
                let e = meta
                    .binary_entry(info.number)
                    .ok_or_else(|| Error::IllegalState("binary entry".to_string()))?;
                let v = lucene_codecs::doc_values::BinaryReader::new(data, e).value(doc)?;
                Ok((0, v.map(<[u8]>::to_vec)))
            }
            DocValuesType::None => Ok((0, None)),
            _ => Err(crate::top_field::SortError::BinaryType(key.field.clone()).into()),
        };
    }
    if key.ty == SortType::String {
        let Some((info, (meta, data))) = column else {
            return Ok((0, None));
        };
        let (ord, terms) = match info.doc_values_type {
            DocValuesType::Sorted => {
                let e = meta
                    .sorted_entry(info.number)
                    .ok_or_else(|| Error::IllegalState("sorted entry".to_string()))?;
                (NumericReader::new(data, &e.ords).value(doc)?, &e.terms)
            }
            DocValuesType::SortedSet => {
                let e = meta
                    .sorted_set_entry(info.number)
                    .ok_or_else(|| Error::IllegalState("sorted set entry".to_string()))?;
                match &e.kind {
                    lucene_codecs::doc_values::SortedSetKind::Single(se) => {
                        (NumericReader::new(data, &se.ords).value(doc)?, &se.terms)
                    }
                    lucene_codecs::doc_values::SortedSetKind::Multi { ords, terms } => {
                        let mut buf = Vec::new();
                        SortedNumericReader::new(data, ords).values(doc, &mut buf)?;
                        (key.selector.pick_ord(&buf), terms)
                    }
                }
            }
            DocValuesType::None => return Ok((0, None)),
            _ => return Err(crate::top_field::SortError::KeywordType(key.field.clone()).into()),
        };
        let Some(ord) = ord else {
            return Ok((0, None));
        };
        let mut dict = TermsDict::open(data, terms).map_err(store_err)?;
        let term = dict.seek_ord(ord).map_err(store_err)?.to_vec();
        return Ok((0, Some(term)));
    }
    // A numeric key: the stored long (a `SORTED_NUMERIC` value through the
    // selector), `(int)` for a 32-bit type, the missing value when absent.
    let Some((info, (meta, data))) = column else {
        return Ok((key.missing, None));
    };
    let v = match info.doc_values_type {
        DocValuesType::Numeric => {
            let e = meta
                .numeric_entry(info.number)
                .ok_or_else(|| Error::IllegalState("numeric entry".to_string()))?;
            NumericReader::new(data, e).value(doc)?
        }
        DocValuesType::SortedNumeric => {
            let e = meta
                .sorted_numeric_entry(info.number)
                .ok_or_else(|| Error::IllegalState("sorted numeric entry".to_string()))?;
            let mut buf = Vec::new();
            SortedNumericReader::new(data, e).values(doc, &mut buf)?;
            key.selector.pick(key.ty, &buf)
        }
        DocValuesType::None => None,
        _ => return Err(crate::top_field::SortError::DocValuesType(key.field.clone()).into()),
    };
    Ok(match v {
        Some(v) if matches!(key.ty, SortType::Int | SortType::Float) => (i64::from(v as i32), None),
        Some(v) => (v, None),
        None => (key.missing, None),
    })
}

/// `SortRescorer`: the first-pass hits re-ordered by a sort, as a
/// `TopFieldCollector` over just those documents collects them (the
/// first-pass score is the score a `SCORE` key reads).
pub struct SortRescorer {
    sort: Vec<SortField>,
}

impl SortRescorer {
    pub fn new(sort: Vec<SortField>) -> Self {
        Self { sort }
    }

    /// The rescored hits with their sort values (`TopFieldDocs`), and the
    /// score each carries after Lucene's reassignment.
    pub fn rescore_field_docs(
        &self,
        ctx: &ValuesContext<'_>,
        first_pass: &TopDocs,
        top_n: usize,
    ) -> Result<ShardTopFieldDocs> {
        let searcher = ctx.searcher()?;
        let mut hits = first_pass.score_docs.clone();
        hits.sort_by_key(|h| h.doc);
        let mut collected: Vec<ShardFieldDoc> = Vec::with_capacity(hits.len());
        for hit in &hits {
            let (leaf, local) = leaf_of(searcher, hit.doc)?;
            let mut values = Vec::with_capacity(self.sort.len());
            let mut terms = Vec::with_capacity(self.sort.len());
            for key in &self.sort {
                let (v, t) = key_value(searcher, leaf, local, key, hit.score)?;
                values.push(v);
                terms.push(t);
            }
            if !self.sort.iter().any(|k| match k.ty {
                SortType::String | SortType::StringVal => true,
                SortType::Custom(id) => crate::top_field::custom_comparator(id, k, 1)
                    .is_ok_and(|c| c.values_are_bytes()),
                _ => false,
            }) {
                terms.clear();
            }
            collected.push(ShardFieldDoc {
                fields: FieldDoc {
                    doc: hit.doc,
                    values,
                    terms,
                },
                score: hit.score,
                shard_index: -1,
            });
        }
        // `TopFieldCollector` over documents collected in ascending order: a
        // later document ties lose.
        collected.sort_by(|a, b| {
            compare_keys(&self.sort, &a.fields, &b.fields).then(a.fields.doc.cmp(&b.fields.doc))
        });
        collected.truncate(top_n);
        // `rescoredDocsClone` sorted by document gets `hits[i].score`: the
        // i-th lowest rescored document takes the i-th lowest first-pass
        // hit's score.
        let mut by_doc: Vec<usize> = (0..collected.len()).collect();
        by_doc.sort_by_key(|&i| collected[i].fields.doc);
        for (rank, &i) in by_doc.iter().enumerate() {
            if let Some(h) = hits.get(rank) {
                collected[i].score = h.score;
            }
        }
        Ok(ShardTopFieldDocs {
            total_hits: first_pass.total_hits,
            hits: collected,
        })
    }
}

impl Rescorer for SortRescorer {
    fn rescore(
        &self,
        ctx: &ValuesContext<'_>,
        first_pass: &TopDocs,
        top_n: usize,
    ) -> Result<TopDocs> {
        let docs = self.rescore_field_docs(ctx, first_pass, top_n)?;
        Ok(TopDocs {
            total_hits: docs.total_hits,
            score_docs: docs
                .hits
                .into_iter()
                .map(|h| ShardScoreDoc::new(h.fields.doc, h.score))
                .collect(),
        })
    }

    /// `SortRescorer.explain`: the sort's values for `doc`, scored `0`.
    fn explain(
        &self,
        ctx: &ValuesContext<'_>,
        first_pass: &Explanation,
        doc: i32,
    ) -> Result<Explanation> {
        let one = TopDocs {
            total_hits: TotalHits {
                value: 1,
                relation: TotalHitsRelation::EqualTo,
            },
            score_docs: vec![ShardScoreDoc::new(doc, first_pass.value)],
        };
        let hits = self.rescore_field_docs(ctx, &one, 1)?;
        let mut subs = vec![Explanation::match_(first_pass.value, "first pass score")
            .with_details(vec![first_pass.clone()])];
        if let Some(h) = hits.hits.first() {
            for (i, key) in self.sort.iter().enumerate() {
                let value = match h.fields.terms.get(i) {
                    Some(Some(t)) => String::from_utf8_lossy(t).into_owned(),
                    Some(None) => "null".to_string(),
                    None => h.fields.values.get(i).copied().unwrap_or(0).to_string(),
                };
                subs.push(Explanation::match_(
                    0.0,
                    format!("sort field {} value={value}", key.field),
                ));
            }
        }
        Ok(Explanation::match_(0.0, "sort field values").with_details(subs))
    }
}

// ---------------------------------------------------------------------------
// RescoreTopNQuery
// ---------------------------------------------------------------------------

/// `RescoreTopNQuery`: the `n` best matches of a query by a values source's
/// value, as a query over just those documents with those scores
/// (`DocAndScoreQuery`). A [`crate::query::Clause`] too
/// (`Clause::from(query)`): a search through [`IndexSearcher`] rewrites it
/// first ([`rewrite_rescore_clauses`]), as `IndexSearcher.rewrite` does.
#[derive(Clone)]
pub struct RescoreTopNQuery {
    query: BooleanQuery,
    source: Arc<dyn DoubleValuesSource>,
    n: usize,
}

impl std::fmt::Debug for RescoreTopNQuery {
    /// `toString`: `RescoreTopNQuery:<query>:<source>[n]`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RescoreTopNQuery:{:?}:{}[{}]",
            self.query,
            self.source.describe(),
            self.n
        )
    }
}

impl PartialEq for RescoreTopNQuery {
    /// `equals`: the same query, source and `n` (a source is equal only to
    /// itself here, having no `equals` of its own).
    fn eq(&self, other: &Self) -> bool {
        self.query == other.query && Arc::ptr_eq(&self.source, &other.source) && self.n == other.n
    }
}

/// `IndexSearcher.rewrite(query)` for the [`RescoreTopNQuery`] clauses a
/// query holds (inside booleans, dis-max, boosts and constant-score
/// wrappers): each becomes the `DocAndScoreQuery` its `rewrite(searcher)`
/// returns, over `ctx`'s searcher. `None` when there are none.
///
/// # Errors
/// What a [`RescoreTopNQuery::rewrite`] reports.
pub fn rewrite_rescore_clauses(
    query: &BooleanQuery,
    ctx: &ValuesContext<'_>,
) -> Result<Option<BooleanQuery>> {
    use crate::extended_query::{DocAndScoreQuery, ExtendedQuery};
    use crate::query::Clause;
    fn clause(c: &Clause, ctx: &ValuesContext<'_>) -> Result<Option<Clause>> {
        Ok(match c {
            Clause::Extended(e) => match e.as_ref() {
                ExtendedQuery::RescoreTopN(q) => {
                    let r = q.rewrite(ctx)?;
                    let searcher = ctx.searcher()?;
                    let bases: Vec<i32> = searcher.segments().iter().map(|s| s.doc_base).collect();
                    let hits: Vec<(i32, f32)> = r
                        .docs
                        .iter()
                        .copied()
                        .zip(r.scores.iter().copied())
                        .collect();
                    Some(Clause::from(DocAndScoreQuery::new(hits, &bases)))
                }
                _ => None,
            },
            Clause::Boolean(b) => boolean(b, ctx)?.map(|b| Clause::Boolean(Box::new(b))),
            Clause::DisjunctionMax(d) => {
                let mut changed = false;
                let disjuncts = list(&d.disjuncts, ctx, &mut changed)?;
                changed.then(|| {
                    let mut d = (**d).clone();
                    d.disjuncts = disjuncts;
                    Clause::DisjunctionMax(Box::new(d))
                })
            }
            Clause::ConstantScore(cs) => clause(&cs.inner, ctx)?.map(|inner| {
                let mut cs = (**cs).clone();
                cs.inner = Box::new(inner);
                Clause::ConstantScore(Box::new(cs))
            }),
            Clause::Boost(b) => clause(&b.inner, ctx)?.map(|inner| {
                let mut b = (**b).clone();
                b.inner = Box::new(inner);
                Clause::Boost(Box::new(b))
            }),
            _ => None,
        })
    }
    fn list(v: &[Clause], ctx: &ValuesContext<'_>, changed: &mut bool) -> Result<Vec<Clause>> {
        v.iter()
            .map(|c| {
                Ok(match clause(c, ctx)? {
                    Some(n) => {
                        *changed = true;
                        n
                    }
                    None => c.clone(),
                })
            })
            .collect()
    }
    fn boolean(b: &BooleanQuery, ctx: &ValuesContext<'_>) -> Result<Option<BooleanQuery>> {
        let mut changed = false;
        let mut out = b.clone();
        out.must = list(&b.must, ctx, &mut changed)?;
        out.filter = list(&b.filter, ctx, &mut changed)?;
        out.should = list(&b.should, ctx, &mut changed)?;
        out.must_not = list(&b.must_not, ctx, &mut changed)?;
        Ok(changed.then_some(out))
    }
    boolean(query, ctx)
}

/// Whether a query holds a [`RescoreTopNQuery`] clause anywhere.
pub(crate) fn has_rescore_clauses(query: &BooleanQuery) -> bool {
    use crate::extended_query::ExtendedQuery;
    use crate::query::Clause;
    fn clause(c: &Clause) -> bool {
        match c {
            Clause::Extended(e) => {
                matches!(e.as_ref(), ExtendedQuery::RescoreTopN(_))
                    || e.children().into_iter().any(clause)
            }
            Clause::Boolean(b) => has_rescore_clauses(b),
            Clause::DisjunctionMax(d) => d.disjuncts.iter().any(clause),
            Clause::ConstantScore(cs) => clause(&cs.inner),
            Clause::Boost(b) => clause(&b.inner),
            _ => false,
        }
    }
    query
        .must
        .iter()
        .chain(&query.filter)
        .chain(&query.should)
        .chain(&query.must_not)
        .any(clause)
}

/// `DocAndScoreQuery`'s data: the kept documents ascending, their scores,
/// and how many documents the inner query matched.
#[derive(Debug, Clone, PartialEq)]
pub struct DocAndScores {
    pub docs: Vec<i32>,
    pub scores: Vec<f32>,
    pub original_count: u64,
}

impl RescoreTopNQuery {
    /// # Errors
    /// [`Error::IllegalArgument`] when `n < 1`.
    pub fn new(query: BooleanQuery, source: Arc<dyn DoubleValuesSource>, n: usize) -> Result<Self> {
        if n < 1 {
            return Err(Error::IllegalArgument("n must be >= 1".to_string()));
        }
        Ok(Self { query, source, n })
    }

    /// `RescoreTopNQuery.createFullPrecisionRescorerQuery(in, target, field, n)`.
    pub fn full_precision(
        query: BooleanQuery,
        target: Vec<f32>,
        field: &str,
        n: usize,
    ) -> Result<Self> {
        Self::new(
            query,
            crate::values_source::full_precision_float_vector_similarity(field, target, None),
            n,
        )
    }

    /// `RescoreTopNQuery.createLateInteractionQuery(in, n, field, query,
    /// function)`.
    pub fn late_interaction(
        query: BooleanQuery,
        n: usize,
        field: &str,
        vectors: Vec<Vec<f32>>,
        function: VectorSimilarityFunction,
    ) -> Result<Self> {
        Self::new(
            query,
            Arc::new(LateInteractionFloatValuesSource::with_function(
                field, vectors, function,
            )?),
            n,
        )
    }

    /// `rewrite(searcher)`: every match of the inner query -- deleted ones
    /// too, as `Weight.scorer` iterates them -- valued by the source (`0` for
    /// none), the best `n` kept by `HitQueue` (the higher value, then the
    /// lower document).
    pub fn rewrite(&self, ctx: &ValuesContext<'_>) -> Result<DocAndScores> {
        let searcher = ctx.searcher()?;
        let mut all: Vec<ShardScoreDoc> = Vec::new();
        let mut original_count: u64 = 0;
        for leaf in 0..searcher.segments().len() {
            let matches = searcher.leaf_scores(&self.query, leaf, true)?;
            if matches.is_empty() {
                continue;
            }
            let doc_base = searcher.segments()[leaf].doc_base;
            let scores: Option<crate::values_source::BoxDoubleValues<'_>> =
                if self.source.needs_scores() {
                    Some(Box::new(MatchScores {
                        matches: matches.clone(),
                        at: 0,
                    }))
                } else {
                    None
                };
            let mut values = self.source.get_values(ctx, leaf, scores)?;
            for &(doc, _) in &matches {
                let v = if values.advance_exact(doc)? {
                    values.double_value()? as f32
                } else {
                    0.0
                };
                all.push(ShardScoreDoc::new(doc_base + doc, v));
                original_count += 1;
            }
        }
        // `HitQueue.lessThan`: `Float.compare`, then the higher document is
        // the lesser.
        all.sort_by(|a, b| java_float_compare(b.score, a.score).then(a.doc.cmp(&b.doc)));
        all.truncate(self.n);
        all.sort_by_key(|h| h.doc);
        Ok(DocAndScores {
            docs: all.iter().map(|h| h.doc).collect(),
            scores: all.iter().map(|h| h.score).collect(),
            original_count,
        })
    }

    /// `searcher.search(rescoreTopNQuery, topN)`: the rewritten documents that
    /// are live, by score then document, and their count.
    pub fn search(&self, ctx: &ValuesContext<'_>, top_n: usize) -> Result<TopDocs> {
        let rewritten = self.rewrite(ctx)?;
        let searcher = ctx.searcher()?;
        let mut hits: Vec<ShardScoreDoc> = Vec::new();
        for (&doc, &score) in rewritten.docs.iter().zip(&rewritten.scores) {
            let (leaf, local) = leaf_of(searcher, doc)?;
            let live = searcher.segments()[leaf]
                .live_docs
                .is_none_or(|l| l.get_doc(local));
            if live {
                hits.push(ShardScoreDoc::new(doc, score));
            }
        }
        let total = hits.len() as u64;
        hits.sort_by(|a, b| java_float_compare(b.score, a.score).then(a.doc.cmp(&b.doc)));
        hits.truncate(top_n);
        Ok(TopDocs {
            total_hits: TotalHits {
                value: total,
                relation: TotalHitsRelation::EqualTo,
            },
            score_docs: hits,
        })
    }
}

/// `Float.compare`: `-0.0 < 0.0`, NaN above everything.
fn java_float_compare(a: f32, b: f32) -> Ordering {
    let key = |f: f32| {
        let bits = if f.is_nan() {
            0x7fc0_0000
        } else {
            f.to_bits() as i32
        };
        bits ^ ((bits >> 31) & 0x7fff_ffff)
    };
    key(a).cmp(&key(b))
}

/// The inner query's scores, handed to a source that needs them
/// (`DoubleValuesSource.fromScorer(innerScorer)`).
struct MatchScores {
    matches: Vec<(i32, f32)>,
    at: usize,
}

impl crate::values_source::DoubleValues for MatchScores {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        while self.at < self.matches.len() && self.matches[self.at].0 < doc {
            self.at += 1;
        }
        Ok(self.matches.get(self.at).is_some_and(|m| m.0 == doc))
    }
    fn double_value(&mut self) -> Result<f64> {
        Ok(f64::from(self.matches.get(self.at).map_or(0.0, |m| m.1)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sd(doc: i32, score: f32) -> ShardScoreDoc {
        ShardScoreDoc::new(doc, score)
    }

    #[test]
    fn ordering_is_scores_then_docs() {
        let got = select_sorted(vec![sd(3, 1.0), sd(1, 2.0), sd(2, 1.0), sd(4, f32::NAN)], 3);
        assert_eq!(got.iter().map(|h| h.doc).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(
            by_score_then_doc(&sd(1, f32::NAN), &sd(2, f32::NAN)),
            Ordering::Less
        );
        assert_eq!(
            by_score_then_doc(&sd(1, 0.0), &sd(2, f32::NAN)),
            Ordering::Less
        );
    }

    #[test]
    fn float_compare_is_javas() {
        assert_eq!(java_float_compare(-0.0, 0.0), Ordering::Less);
        assert_eq!(
            java_float_compare(f32::NAN, f32::INFINITY),
            Ordering::Greater
        );
        assert_eq!(java_float_compare(1.0, 1.0), Ordering::Equal);
    }

    #[test]
    fn weight_combination_narrows_once() {
        let r = QueryRescorer::with_weight(BooleanQuery::new(), 2.0);
        let first = 0.1f32;
        let second = 0.3f32;
        assert_eq!(
            (r.combine)(first, true, second),
            (f64::from(first) + 2.0 * f64::from(second)) as f32
        );
        assert_eq!((r.combine)(first, false, 9.0), first);
    }

    #[test]
    fn rescore_top_n_needs_n() {
        assert!(
            RescoreTopNQuery::new(BooleanQuery::new(), crate::values_source::constant(1.0), 0)
                .is_err()
        );
        let mut m = MatchScores {
            matches: vec![(1, 2.0), (5, 3.0)],
            at: 0,
        };
        use crate::values_source::DoubleValues;
        assert!(!m.advance_exact(0).unwrap());
        assert!(m.advance_exact(5).unwrap());
        assert_eq!(m.double_value().unwrap(), 3.0);
        assert!(!m.advance_exact(9).unwrap());
        assert_eq!(m.double_value().unwrap(), 0.0);
    }

    /// A `RescoreTopNQuery` clause anywhere in a query -- under a boolean,
    /// a dis-max, a boost, a constant-score wrapper -- is rewritten to its
    /// `DocAndScoreQuery` by the searcher, and searches like the standalone
    /// query; unrewritten, the tree refuses it. A values-source sort that
    /// needs the searcher sorts once rewritten, and only over its reader.
    #[test]
    fn rescore_clauses_are_rewritten_wherever_they_sit() {
        use crate::directory_reader::DirectoryReader;
        use crate::query::{
            BoostQuery, Clause, ConstantScoreQuery, DisjunctionMaxQuery, TermQuery,
        };
        let dir = lucene_store::FsDirectory::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/values_rescore_index"
        ));
        let reader = DirectoryReader::open(&dir).unwrap();
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let norms = vec![None; segments.len()];
        let searcher = IndexSearcher::new(&segments, &norms).unwrap();
        let ctx = ValuesContext::new(&searcher);
        let term = |w: &str| Clause::Term(TermQuery::new("body", w.as_bytes().to_vec()));
        let mut inner = BooleanQuery::new();
        inner.must.push(term("w0"));
        let rtn =
            RescoreTopNQuery::new(inner, crate::values_source::from_float_field("f"), 6).unwrap();
        assert_eq!(rtn, rtn.clone());
        assert!(format!("{rtn:?}").starts_with("RescoreTopNQuery:"));
        let alone = rtn.search(&ctx, 20).unwrap();
        assert!(!alone.score_docs.is_empty());

        let wrap = |c: Clause| {
            let mut q = BooleanQuery::new();
            q.must.push(c);
            q
        };
        let wrapped = [
            wrap(Clause::from(rtn.clone())),
            wrap(Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery {
                disjuncts: vec![Clause::from(rtn.clone())],
                tie_breaker: 0.0,
            }))),
            wrap(Clause::Boost(Box::new(BoostQuery {
                inner: Box::new(Clause::from(rtn.clone())),
                boost: 1.0,
            }))),
            wrap(Clause::Boolean(Box::new(wrap(Clause::from(rtn.clone()))))),
        ];
        for q in &wrapped {
            assert!(has_rescore_clauses(q));
            let got = searcher.search(q, 20).unwrap();
            let docs = |t: &TopDocs| {
                t.score_docs
                    .iter()
                    .map(|h| (h.doc, h.score.to_bits()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(docs(&got), docs(&alone));
            assert_eq!(searcher.count(q).unwrap(), alone.total_hits.value);
        }
        let constant = wrap(Clause::ConstantScore(Box::new(ConstantScoreQuery {
            inner: Box::new(Clause::from(rtn.clone())),
            score: 2.0,
        })));
        assert_eq!(
            searcher.search(&constant, 20).unwrap().score_docs.len(),
            alone.score_docs.len()
        );
        let plain = wrap(term("w1"));
        assert!(!has_rescore_clauses(&plain));
        assert!(rewrite_rescore_clauses(&plain, &ctx).unwrap().is_none());
        // The tree itself refuses an unrewritten one.
        let err = crate::search_boolean_query_scored_segment(
            &segments[0],
            &wrapped[0],
            None,
            None,
            &mut crate::collector::TopDocsCollector::new(5),
        );
        assert!(err.is_err());

        // A query-backed sort: refused until rewritten; the rewritten key
        // is bound to this reader.
        let sort = vec![crate::values_source::double_sort_field(
            crate::values_source::from_query(wrap(term("w2"))),
            true,
            0.0,
        )];
        let readers = reader.segment_readers();
        let all = wrap(Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(
            i32::MAX,
        )));
        let run = |s: &[crate::top_field::SortField]| {
            crate::top_field::search_sorted(&segments, readers, &all, &norms, s, 5, u64::MAX, None)
        };
        assert!(run(&sort).is_err());
        let rewritten = crate::top_field::rewrite_sort(&sort, &ctx).unwrap();
        assert_ne!(rewritten, sort);
        assert_eq!(
            crate::top_field::rewrite_sort(&rewritten, &ctx).unwrap(),
            rewritten
        );
        assert!(format!("{:?}", rewritten[0].rewritten).contains("RewrittenSource"));
        let top = run(&rewritten).unwrap();
        assert_eq!(top.hits.len(), 5);
        // Over one segment only: the other leaves are unknown to the key.
        let one = IndexSearcher::new(&segments[1..], &norms[1..]).unwrap();
        let other = crate::top_field::rewrite_sort(&sort, &ValuesContext::new(&one)).unwrap();
        assert!(run(&other).is_err());
    }
}
