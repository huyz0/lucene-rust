//! `org.apache.lucene.classification.document`: classifiers of a whole
//! document, each text field analysed by its own analyzer.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use lucene_analysis::Analyzer;
use lucene_index::document::Document;
use lucene_util::strict_math::log;

use super::knn::boolean_clause;
use super::naive_bayes::{avg_unique_terms, doc_freq, field_terms};
use super::{
    analyze, best, norm_classification_results, parse_boost, sort_results, split_boost, sub_list,
    ClassificationResult, KNearestNeighborClassifier, SharedSimilarity, SimpleNaiveBayesClassifier,
    TrainingIndex,
};
use crate::query::{BooleanQuery, Clause, TermQuery};
use crate::{Error, Result};

/// `DocumentClassifier<T>`.
pub trait DocumentClassifier<T> {
    /// `assignClass(document)`.
    ///
    /// # Errors
    /// What reading the index or analysing the document reports.
    fn assign_class(&self, document: &Document) -> Result<Option<ClassificationResult<T>>>;

    /// `getClasses(document)`: every class, best first.
    ///
    /// # Errors
    /// As [`Self::assign_class`].
    fn get_classes(&self, document: &Document) -> Result<Vec<ClassificationResult<T>>>;

    /// `getClasses(document, max)`.
    ///
    /// # Errors
    /// As [`Self::assign_class`]; `max` past the classes there are, where
    /// Java's `subList` throws.
    fn get_classes_max(
        &self,
        document: &Document,
        max: usize,
    ) -> Result<Vec<ClassificationResult<T>>> {
        sub_list(self.get_classes(document)?, max)
    }
}

/// The analyzer of `field`, as `field2analyzer.get(field)`; Java's
/// `NullPointerException` without one.
fn analyzer_of<'m>(map: &'m HashMap<String, Arc<Analyzer>>, field: &str) -> Result<&'m Analyzer> {
    map.get(field)
        .map(|a| &**a)
        .ok_or_else(|| Error::IllegalArgument(format!("no analyzer for field {field}")))
}

/// `KNearestNeighborDocumentClassifier`: a `MoreLikeThis` query per value of
/// each text field of the document.
pub struct KNearestNeighborDocumentClassifier<'s, 'a> {
    knn: KNearestNeighborClassifier<'s, 'a>,
    field_analyzers: HashMap<String, Arc<Analyzer>>,
}

impl<'s, 'a> KNearestNeighborDocumentClassifier<'s, 'a> {
    /// `new KNearestNeighborDocumentClassifier(reader, similarity, query, k,
    /// minDocsFreq, minTermFreq, classFieldName, field2analyzer,
    /// textFieldNames...)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        index: TrainingIndex<'s, 'a>,
        similarity: SharedSimilarity,
        query: Option<Clause>,
        k: usize,
        min_doc_freq: i32,
        min_term_freq: i32,
        class_field: impl Into<String>,
        field_analyzers: HashMap<String, Arc<Analyzer>>,
        text_fields: &[&str],
    ) -> Self {
        Self {
            knn: KNearestNeighborClassifier::new(
                index,
                similarity,
                None,
                query,
                k,
                min_doc_freq,
                min_term_freq,
                class_field,
                text_fields,
            ),
            field_analyzers,
        }
    }

    fn knn_search(&self, document: &Document) -> Result<crate::top_docs::TopDocs> {
        let mut should = Vec::new();
        for name in &self.knn.text_fields {
            let (field, boost) = split_boost(name);
            let analyzer = self.field_analyzers.get(field).map(|a| &**a);
            let mut mlt = self.knn.mlt(analyzer);
            if let Some(b) = boost {
                mlt.boost_factor = parse_boost(b)?;
            }
            for value in document.get_values(field) {
                should.push(boolean_clause(mlt.like_texts(field, &[value.as_str()])?));
            }
        }
        self.knn.search_like(should)
    }
}

impl DocumentClassifier<Vec<u8>> for KNearestNeighborDocumentClassifier<'_, '_> {
    fn assign_class(&self, document: &Document) -> Result<Option<ClassificationResult<Vec<u8>>>> {
        Ok(best(&self.knn.build_list(&self.knn_search(document)?)?))
    }

    fn get_classes(&self, document: &Document) -> Result<Vec<ClassificationResult<Vec<u8>>>> {
        let mut c = self.knn.build_list(&self.knn_search(document)?)?;
        sort_results(&mut c);
        Ok(c)
    }

    /// `getClasses(document, max)`: at most `max`, never an error.
    fn get_classes_max(
        &self,
        document: &Document,
        max: usize,
    ) -> Result<Vec<ClassificationResult<Vec<u8>>>> {
        let mut c = self.get_classes(document)?;
        c.truncate(max);
        Ok(c)
    }
}

/// `SimpleNaiveBayesDocumentClassifier`: naive Bayes summed over the text
/// fields' values, each value's likelihood averaged over its tokens and
/// multiplied by its field's boost.
///
/// Java strips the `^boost` suffixes off its text field names in place the
/// first time it analyses a document, so only that first document is scored
/// with the boosts; this classifier does the same.
pub struct SimpleNaiveBayesDocumentClassifier<'s, 'a> {
    base: SimpleNaiveBayesClassifier<'s, 'a>,
    field_analyzers: HashMap<String, Arc<Analyzer>>,
    /// `textFieldNames`, rewritten by the first `analyzeSeedDocument`.
    text_fields: Mutex<Vec<String>>,
}

impl<'s, 'a> SimpleNaiveBayesDocumentClassifier<'s, 'a> {
    /// `new SimpleNaiveBayesDocumentClassifier(reader, query, classFieldName,
    /// field2analyzer, textFieldNames...)`.
    pub fn new(
        index: TrainingIndex<'s, 'a>,
        query: Option<Clause>,
        class_field: impl Into<String>,
        field_analyzers: HashMap<String, Arc<Analyzer>>,
        text_fields: &[&str],
    ) -> Self {
        // The base's analyzer is never used (Java passes `null`).
        let unused = Arc::new(Analyzer::keyword());
        Self {
            base: SimpleNaiveBayesClassifier::new(index, unused, query, class_field, text_fields),
            field_analyzers,
            text_fields: Mutex::new(text_fields.iter().map(|s| s.to_string()).collect()),
        }
    }

    /// `assignNormClasses`.
    fn assign_norm_classes(
        &self,
        document: &Document,
    ) -> Result<Vec<ClassificationResult<Vec<u8>>>> {
        let mut assigned = Vec::new();
        if let Some(classes) = field_terms(&self.base.index, &self.base.class_field)? {
            let (fields, tokens, boosts) = self.analyze_seed_document(document)?;
            let docs_with_class = self.base.count_docs_with_class()?;
            for class in classes {
                let mut class_score = 0.0;
                for field in &fields {
                    let mut field_score = 0.0;
                    let boost = f64::from(boosts.get(field).copied().unwrap_or(1.0));
                    for arr in tokens.get(field).map(Vec::as_slice).unwrap_or_default() {
                        field_score += self.base.log_prior(&class, docs_with_class)?
                            + self.log_likelihood(arr, field, &class, docs_with_class)? * boost;
                    }
                    class_score += field_score;
                }
                assigned.push(ClassificationResult::new(class, class_score));
            }
        }
        Ok(norm_classification_results(assigned))
    }

    /// `analyzeSeedDocument`: each text field's values analysed, its boost,
    /// and the field names without their boosts (kept from now on).
    #[allow(clippy::type_complexity)]
    fn analyze_seed_document(
        &self,
        document: &Document,
    ) -> Result<(
        Vec<String>,
        HashMap<String, Vec<Vec<String>>>,
        HashMap<String, f32>,
    )> {
        let mut names = self.text_fields.lock().unwrap_or_else(|e| e.into_inner());
        let mut tokens = HashMap::new();
        let mut boosts = HashMap::new();
        for name in names.iter_mut() {
            let (field, boost) = split_boost(name);
            let boost = boost.map(parse_boost).transpose()?.unwrap_or(1.0);
            let field = field.to_string();
            let mut values = Vec::new();
            for v in document.get_values(&field) {
                values.push(analyze(
                    analyzer_of(&self.field_analyzers, &field)?,
                    &field,
                    &v,
                )?);
            }
            tokens.insert(field.clone(), values);
            boosts.insert(field.clone(), boost);
            *name = field;
        }
        Ok((names.clone(), tokens, boosts))
    }

    /// `calculateLogLikelihood`: the log probabilities of the tokens,
    /// averaged (`NaN` for no tokens).
    fn log_likelihood(
        &self,
        tokens: &[String],
        field: &str,
        class: &[u8],
        docs_with_class: i32,
    ) -> Result<f64> {
        let index = &self.base.index;
        let mut result = 0.0;
        for word in tokens {
            let sub = BooleanQuery {
                should: vec![Clause::Term(TermQuery::new(field, word.as_bytes()))],
                ..Default::default()
            };
            let mut q = BooleanQuery {
                must: vec![
                    Clause::Boolean(Box::new(sub)),
                    Clause::Term(TermQuery::new(self.base.class_field.as_str(), class)),
                ],
                ..Default::default()
            };
            if let Some(f) = &self.base.query {
                q.must.push(f.clone());
            }
            let hits = index.searcher(None)?.count(&q)? as i32;
            let num = f64::from(hits + 1);
            let avg = avg_unique_terms(index, &[field.to_string()])?;
            let den = avg * f64::from(doc_freq(index, &self.base.class_field, class)?)
                + f64::from(docs_with_class);
            result += log(num / den);
        }
        Ok(result / tokens.len() as f64)
    }
}

impl DocumentClassifier<Vec<u8>> for SimpleNaiveBayesDocumentClassifier<'_, '_> {
    fn assign_class(&self, document: &Document) -> Result<Option<ClassificationResult<Vec<u8>>>> {
        Ok(best(&self.assign_norm_classes(document)?))
    }

    fn get_classes(&self, document: &Document) -> Result<Vec<ClassificationResult<Vec<u8>>>> {
        let mut c = self.assign_norm_classes(document)?;
        sort_results(&mut c);
        Ok(c)
    }
}
