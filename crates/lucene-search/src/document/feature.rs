//! `FeatureField`'s search side: `FeatureQuery` and its four scoring
//! functions (`newLinearQuery`, `newLogQuery`, `newSaturationQuery`,
//! `newSigmoidQuery`), `FeatureSortField` (`newFeatureSort`) and
//! `FeatureDoubleValuesSource` (`newDoubleValues`).
//!
//! A feature's value is the term frequency its field indexed for the
//! feature's term, decoded by
//! [`FeatureField::decode_feature_value`](lucene_index::document::FeatureField::decode_feature_value);
//! a query scores each document holding the term with its function of that
//! value, the query weight being the `BoostQuery` boost `createWeight` hands
//! the `SimScorer`. The arithmetic is Java's, `float` where Java's is and
//! `double` (`Math.log`, `Math.pow`) where Java's is.

use lucene_codecs::blocktree::FieldTerms;
use lucene_index::document::FeatureField;

use super::point_queries::boxed_boost;
use super::{reader, DocumentQuery};
use crate::collector::ScoringCollector;
use crate::multi_segment::OpenSegment;
use crate::{Error, Result};

fn illegal(message: impl Into<String>) -> Error {
    Error::DocumentQuery(message.into())
}

/// `FeatureField.FeatureFunction`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureFunction {
    /// `LinearFunction`: `w * S`.
    Linear,
    /// `LogFunction`: `w * log(a + S)`.
    Log { scaling_factor: f32 },
    /// `SaturationFunction`: `w * S / (S + k)`; `pivot: None` until the
    /// query is rewritten, when `k` becomes the feature's average value.
    Saturation { pivot: Option<f32> },
    /// `SigmoidFunction`: `w * S^a / (S^a + k^a)`.
    Sigmoid { pivot: f32, a: f32 },
}

impl FeatureFunction {
    /// `SimScorer.score(freq, norm)` of `scorer(weight)`: the score of a
    /// document whose term frequency is `freq`.
    pub fn score(&self, weight: f32, freq: f32) -> Result<f32> {
        let f = FeatureField::decode_feature_value(freq);
        Ok(match *self {
            FeatureFunction::Linear => weight * f,
            FeatureFunction::Log { scaling_factor } => {
                (f64::from(weight) * f64::from(scaling_factor + f).ln()) as f32
            }
            FeatureFunction::Saturation { pivot } => {
                let pivot = pivot.ok_or_else(|| Error::DocumentQuery("Rewrite first".into()))?;
                weight * (1.0 - pivot / (f + pivot))
            }
            FeatureFunction::Sigmoid { pivot, a } => {
                let pivot_pa = f64::from(pivot).powf(f64::from(a));
                (f64::from(weight)
                    * (1.0 - pivot_pa / (f64::from(f).powf(f64::from(a)) + pivot_pa)))
                    as f32
            }
        })
    }
}

/// The feature's term in `leaf`, when the segment's field holds it.
fn feature_terms<'a>(leaf: &OpenSegment<'a>, field: &str) -> Option<&'a FieldTerms> {
    leaf.fields.field(field)
}

/// The live documents holding the feature, with their frequencies.
fn feature_postings(leaf: &OpenSegment<'_>, field: &str, feature: &str) -> Result<Vec<(i32, i32)>> {
    let Some(terms) = feature_terms(leaf, field) else {
        return Ok(Vec::new());
    };
    let Some(p) = terms.postings(feature.as_bytes(), leaf.doc_in)? else {
        return Ok(Vec::new());
    };
    Ok(p.docs
        .into_iter()
        .zip(p.freqs)
        .filter(|(d, _)| leaf.live_docs.is_none_or(|bits| bits.get_doc(*d)))
        .collect())
}

/// `FeatureQuery`: the documents holding `feature` in `field`, scored by the
/// function of their feature value.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureQuery {
    pub field: String,
    pub feature: String,
    pub function: FeatureFunction,
}

impl FeatureQuery {
    pub fn new(
        field: impl Into<String>,
        feature: impl Into<String>,
        function: FeatureFunction,
    ) -> Self {
        FeatureQuery {
            field: field.into(),
            feature: feature.into(),
            function,
        }
    }
}

/// `FeatureField.computePivotFeatureValue`: the feature's mean value over
/// the whole reader (deleted documents included, as `TermStates` counts
/// them), or 1 when no document has it.
pub fn compute_pivot_feature_value(
    leaves: &[OpenSegment<'_>],
    field: &str,
    feature: &str,
) -> Result<f32> {
    let (mut df, mut ttf) = (0i64, 0i64);
    for leaf in leaves {
        if let Some(stats) = feature_terms(leaf, field)
            .map(|t| t.try_seek_exact(feature.as_bytes()))
            .transpose()?
            .flatten()
        {
            df = df.saturating_add(i64::from(stats.doc_freq));
            ttf = ttf.saturating_add(stats.total_term_freq);
        }
    }
    if df == 0 {
        return Ok(1.0);
    }
    let avg_freq = (ttf as f64 / df as f64) as f32;
    Ok(FeatureField::decode_feature_value(avg_freq))
}

impl DocumentQuery for FeatureQuery {
    /// `rewrite`: a saturation function without a pivot gets the feature's
    /// mean value.
    fn rewrite(&self, leaves: &[OpenSegment<'_>]) -> Result<Option<Box<dyn DocumentQuery>>> {
        if let FeatureFunction::Saturation { pivot: None } = self.function {
            let pivot = compute_pivot_feature_value(leaves, &self.field, &self.feature)?;
            return Ok(Some(Box::new(FeatureQuery {
                function: FeatureFunction::Saturation { pivot: Some(pivot) },
                ..self.clone()
            })));
        }
        Ok(None)
    }

    /// `TermScorer` over the feature's postings with the function's
    /// `SimScorer`.
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        for (doc, freq) in feature_postings(leaf, &self.field, &self.feature)? {
            collector.collect(doc, self.function.score(boost, freq as f32)?);
        }
        Ok(())
    }
}

/// `FeatureSortField` (`newFeatureSort`): documents by their feature value,
/// highest first, a document without the feature at `0`; ties by doc id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSortField {
    pub field: String,
    pub feature: String,
}

impl FeatureSortField {
    pub fn new(field: impl Into<String>, feature: impl Into<String>) -> Self {
        FeatureSortField {
            field: field.into(),
            feature: feature.into(),
        }
    }

    /// `setMissingValue`: not supported.
    pub fn set_missing_value(&mut self) -> Result<()> {
        Err(illegal("Missing value not supported for FeatureSortField"))
    }

    /// `IndexSearcher.search(query, n, new Sort(this))`: the top `n` hits of
    /// `query` by this sort, as `(global doc, value)`.
    pub fn search(
        &self,
        leaves: &[OpenSegment<'_>],
        query: &dyn DocumentQuery,
        n: usize,
    ) -> Result<Vec<(i32, f32)>> {
        let hits = super::search_all(leaves, query)?;
        let mut valued = Vec::with_capacity(hits.len());
        let mut hit = hits.iter().peekable();
        for leaf in leaves {
            let mut values = self.values(leaf)?;
            let end = leaf.doc_base.saturating_add(reader(leaf)?.max_doc);
            while let Some(h) = hit.next_if(|h| h.doc_id < end) {
                let local = h.doc_id.saturating_sub(leaf.doc_base);
                valued.push((h.doc_id, values.value_for_doc(local)));
            }
        }
        // `FeatureComparator` under `reverse = true`: `Float.compare`, highest
        // first; `FieldValueHitQueue` breaks ties by doc id.
        valued.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        valued.truncate(n);
        Ok(valued)
    }

    /// `FeatureComparator.doSetNextReader`: the leaf's feature values.
    pub fn values(&self, leaf: &OpenSegment<'_>) -> Result<FeatureDoubleValues> {
        FeatureDoubleValuesSource::new(self.field.clone(), self.feature.clone()).get_values(leaf)
    }
}

/// `FeatureDoubleValuesSource` (`newDoubleValues`): each document's feature
/// value, or none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureDoubleValuesSource {
    pub field: String,
    pub feature: String,
}

impl FeatureDoubleValuesSource {
    pub fn new(field: impl Into<String>, feature: impl Into<String>) -> Self {
        FeatureDoubleValuesSource {
            field: field.into(),
            feature: feature.into(),
        }
    }

    /// `needsScores()`.
    pub fn needs_scores(&self) -> bool {
        false
    }

    /// `getValues(ctx, scores)`: the leaf's values (deleted documents
    /// included, as a postings enum without live docs reads them).
    pub fn get_values(&self, leaf: &OpenSegment<'_>) -> Result<FeatureDoubleValues> {
        let postings = match feature_terms(leaf, &self.field) {
            None => Vec::new(),
            Some(terms) => match terms.postings(self.feature.as_bytes(), leaf.doc_in)? {
                None => Vec::new(),
                Some(p) => p.docs.into_iter().zip(p.freqs).collect(),
            },
        };
        Ok(FeatureDoubleValues {
            postings,
            at: 0,
            current: -1,
        })
    }
}

/// `FeatureDoubleValues`: a forward-only cursor over a leaf's postings.
#[derive(Debug, Clone)]
pub struct FeatureDoubleValues {
    postings: Vec<(i32, i32)>,
    at: usize,
    /// The doc id the cursor sits on (`-1` before the first).
    current: i32,
}

impl FeatureDoubleValues {
    /// `advanceExact(doc)`: whether `doc` holds the feature. Targets must not
    /// go backwards (a target behind the cursor answers `false`, as Java's
    /// does).
    pub fn advance_exact(&mut self, doc: i32) -> bool {
        if doc < self.current {
            return false;
        }
        while let Some(&(d, _)) = self.postings.get(self.at) {
            if d >= doc {
                self.current = d;
                return d == doc;
            }
            self.at = self.at.saturating_add(1);
        }
        self.current = i32::MAX;
        false
    }

    /// `doubleValue()`: the value at the cursor.
    pub fn double_value(&self) -> f64 {
        let freq = self.postings.get(self.at).map_or(0, |p| p.1);
        f64::from(FeatureField::decode_feature_value(freq as f32))
    }

    /// `FeatureComparator.getValueForDoc`: the value, `0` without one.
    pub fn value_for_doc(&mut self, doc: i32) -> f32 {
        if self.advance_exact(doc) {
            self.double_value() as f32
        } else {
            0.0
        }
    }
}

/// `FeatureField`'s query factories, sort field and value source.
pub mod feature_field {
    use super::*;

    fn check_weight(weight: f32) -> Result<()> {
        if weight <= 0.0 || weight > FeatureField::MAX_WEIGHT || weight.is_nan() {
            return Err(illegal(format!(
                "weight must be in (0, {:?}], got: {weight:?}",
                FeatureField::MAX_WEIGHT
            )));
        }
        Ok(())
    }

    fn query(
        field: &str,
        feature: &str,
        function: FeatureFunction,
        weight: f32,
    ) -> Box<dyn DocumentQuery> {
        boxed_boost(
            Box::new(FeatureQuery::new(field, feature, function)),
            weight,
        )
    }

    /// `newLinearQuery(fieldName, featureName, weight)`.
    pub fn new_linear_query(
        field: &str,
        feature: &str,
        weight: f32,
    ) -> Result<Box<dyn DocumentQuery>> {
        check_weight(weight)?;
        Ok(query(field, feature, FeatureFunction::Linear, weight))
    }

    /// `newLogQuery(fieldName, featureName, weight, scalingFactor)`.
    pub fn new_log_query(
        field: &str,
        feature: &str,
        weight: f32,
        scaling_factor: f32,
    ) -> Result<Box<dyn DocumentQuery>> {
        check_weight(weight)?;
        if !(scaling_factor >= 1.0 && scaling_factor.is_finite()) {
            return Err(illegal(format!(
                "scalingFactor must be >= 1, got: {scaling_factor:?}"
            )));
        }
        Ok(query(
            field,
            feature,
            FeatureFunction::Log { scaling_factor },
            weight,
        ))
    }

    /// `newSaturationQuery(fieldName, featureName, weight, pivot)`.
    pub fn new_saturation_query(
        field: &str,
        feature: &str,
        weight: f32,
        pivot: f32,
    ) -> Result<Box<dyn DocumentQuery>> {
        saturation(field, feature, weight, Some(pivot))
    }

    /// `newSaturationQuery(fieldName, featureName)`: weight 1, the pivot
    /// computed from the index at rewrite time.
    pub fn new_saturation_query_auto(field: &str, feature: &str) -> Result<Box<dyn DocumentQuery>> {
        saturation(field, feature, 1.0, None)
    }

    fn saturation(
        field: &str,
        feature: &str,
        weight: f32,
        pivot: Option<f32>,
    ) -> Result<Box<dyn DocumentQuery>> {
        check_weight(weight)?;
        if let Some(p) = pivot {
            if !(p > 0.0 && p.is_finite()) {
                return Err(illegal(format!("pivot must be > 0, got: {p:?}")));
            }
        }
        Ok(query(
            field,
            feature,
            FeatureFunction::Saturation { pivot },
            weight,
        ))
    }

    /// `newSigmoidQuery(fieldName, featureName, weight, pivot, exp)`.
    pub fn new_sigmoid_query(
        field: &str,
        feature: &str,
        weight: f32,
        pivot: f32,
        exp: f32,
    ) -> Result<Box<dyn DocumentQuery>> {
        check_weight(weight)?;
        if !(pivot > 0.0 && pivot.is_finite()) {
            return Err(illegal(format!("pivot must be > 0, got: {pivot:?}")));
        }
        if !(exp > 0.0 && exp.is_finite()) {
            return Err(illegal(format!("exp must be > 0, got: {exp:?}")));
        }
        Ok(query(
            field,
            feature,
            FeatureFunction::Sigmoid { pivot, a: exp },
            weight,
        ))
    }

    /// `newFeatureSort(field, featureName)`.
    pub fn new_feature_sort(field: &str, feature: &str) -> FeatureSortField {
        FeatureSortField::new(field, feature)
    }

    /// `newDoubleValues(field, featureName)`.
    pub fn new_double_values(field: &str, feature: &str) -> FeatureDoubleValuesSource {
        FeatureDoubleValuesSource::new(field, feature)
    }
}
