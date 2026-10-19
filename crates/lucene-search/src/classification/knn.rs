//! `KNearestNeighborClassifier` and `KNearestFuzzyClassifier`: the classes
//! of the `k` documents most like the input, by `MoreLikeThis` or by
//! [`NearestFuzzyQuery`].

use std::sync::Arc;

use lucene_analysis::Analyzer;

use super::{
    best, parse_boost, sort_results, split_boost, ClassificationResult, Classifier,
    NearestFuzzyQuery, SharedSimilarity, TrainingIndex,
};
use crate::mlt::{java_hash_order_of, MoreLikeThis};
use crate::query::{BooleanQuery, Clause, MatchNoDocsQuery, WildcardQuery};
use crate::top_docs::TopDocs;
use crate::Result;

/// A rewritten query as a clause: an empty `BooleanQuery` matches nothing.
pub(crate) fn boolean_clause(q: BooleanQuery) -> Clause {
    if q.must.is_empty() && q.filter.is_empty() && q.should.is_empty() && q.must_not.is_empty() {
        Clause::MatchNoDocs(MatchNoDocsQuery::default())
    } else {
        Clause::Boolean(Box::new(q))
    }
}

/// `IndexSearcher.rewrite(query)`: [`IndexSearcher::search`] scores the
/// query as given, but Java scores it rewritten -- and the rewrite sums the
/// boosts of duplicate `SHOULD` terms (two text values' like-queries, two
/// fuzzy variants of one term) into one clause, which can round differently
/// from scoring them apart.
///
/// [`IndexSearcher::search`]: crate::index_searcher::IndexSearcher::search
pub(crate) fn rewritten(q: BooleanQuery) -> BooleanQuery {
    match Clause::Boolean(Box::new(q)).rewrite() {
        Clause::Boolean(b) => *b,
        other => BooleanQuery {
            must: vec![other],
            ..Default::default()
        },
    }
}

/// `classField:*`.
pub(crate) fn any_class(class_field: &str) -> Clause {
    Clause::Wildcard(WildcardQuery::new(class_field, b"*".to_vec()))
}

/// `buildListFromTopDocs`: each class of the hits scored by its hits'
/// scores relative to the best, over `k` (over the hits, when fewer than `k`
/// have a class). `all_values` reads every stored value of the class field
/// (`getFields`), else the first (`getField`). The classes come in
/// `HashMap<BytesRef, _>` order.
pub(crate) fn build_list_from_top_docs(
    index: &TrainingIndex<'_, '_>,
    top: &TopDocs,
    class_field: &str,
    k: usize,
    all_values: bool,
) -> Result<Vec<ClassificationResult<Vec<u8>>>> {
    let max_score = match top.score_docs.first() {
        Some(sd) if top.total_hits.value > 0 => sd.score,
        _ => f32::NAN,
    };
    // (class, count, summed boost), in first-seen order.
    let mut classes: Vec<(Vec<u8>, i32, f64)> = Vec::new();
    for sd in &top.score_docs {
        let mut values = index.stored_values(sd.doc, class_field)?;
        if !all_values {
            values.truncate(1);
        }
        for v in values {
            let single = f64::from(sd.score / max_score);
            match classes
                .iter_mut()
                .find(|(c, _, _)| c.as_slice() == v.as_bytes())
            {
                Some(entry) => {
                    entry.1 += 1;
                    entry.2 += single;
                }
                None => classes.push((v.into_bytes(), 1, single)),
            }
        }
    }
    let hashes: Vec<i32> = classes
        .iter()
        .map(|(c, _, _)| index.bytes_hash(c))
        .collect();
    let k = k as f64;
    let mut temporary = Vec::with_capacity(classes.len());
    let mut sum_doc = 0i32;
    for i in java_hash_order_of(&hashes) {
        let (class, count, boost) = &classes[i];
        let norm_boost = boost / f64::from(*count);
        temporary.push(ClassificationResult::new(
            class.clone(),
            (f64::from(*count) * norm_boost) / k,
        ));
        sum_doc += count;
    }
    if f64::from(sum_doc) < k {
        let sum_doc = f64::from(sum_doc);
        Ok(temporary
            .into_iter()
            .map(|cr| ClassificationResult::new(cr.assigned_class, cr.score * k / sum_doc))
            .collect())
    } else {
        Ok(temporary)
    }
}

/// `KNearestNeighborClassifier`: a `MoreLikeThis` query per text field (a
/// field may carry a `^boost`), restricted to documents with a class.
pub struct KNearestNeighborClassifier<'s, 'a> {
    pub(crate) index: TrainingIndex<'s, 'a>,
    pub(crate) similarity: SharedSimilarity,
    pub(crate) analyzer: Option<Arc<Analyzer>>,
    pub(crate) query: Option<Clause>,
    pub(crate) k: usize,
    pub(crate) min_doc_freq: i32,
    pub(crate) min_term_freq: i32,
    pub(crate) class_field: String,
    pub(crate) text_fields: Vec<String>,
}

impl<'s, 'a> KNearestNeighborClassifier<'s, 'a> {
    /// `new KNearestNeighborClassifier(reader, similarity, analyzer, query,
    /// k, minDocsFreq, minTermFreq, classFieldName, textFieldNames...)`: a
    /// `similarity` of `None` is BM25; `MoreLikeThis` keeps its own minimum
    /// frequencies unless one given is positive.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        index: TrainingIndex<'s, 'a>,
        similarity: SharedSimilarity,
        analyzer: Option<Arc<Analyzer>>,
        query: Option<Clause>,
        k: usize,
        min_doc_freq: i32,
        min_term_freq: i32,
        class_field: impl Into<String>,
        text_fields: &[&str],
    ) -> Self {
        Self {
            index,
            similarity,
            analyzer,
            query,
            k,
            min_doc_freq,
            min_term_freq,
            class_field: class_field.into(),
            text_fields: text_fields.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// The classifier's `MoreLikeThis`, configured as Java's constructor
    /// configures it, with `analyzer` and `setBoost(true)`.
    pub(crate) fn mlt<'r>(&'r self, analyzer: Option<&'r Analyzer>) -> MoreLikeThis<'r> {
        let mut mlt = MoreLikeThis::new(self.index.reader);
        mlt.analyzer = analyzer;
        mlt.field_names = Some(self.text_fields.clone());
        if self.min_doc_freq > 0 {
            mlt.min_doc_freq = self.min_doc_freq;
        }
        if self.min_term_freq > 0 {
            mlt.min_term_freq = self.min_term_freq;
        }
        mlt.boost = true;
        mlt
    }

    /// The `SHOULD` like-queries, then `classField:*` and the filter as
    /// `MUST`, searched for the top `k`.
    pub(crate) fn search_like(&self, should: Vec<Clause>) -> Result<TopDocs> {
        let mut q = BooleanQuery {
            should,
            must: vec![any_class(&self.class_field)],
            ..Default::default()
        };
        if let Some(f) = &self.query {
            q.must.push(f.clone());
        }
        let searcher = self.index.searcher(self.similarity.as_deref())?;
        searcher.search(&rewritten(q), self.k)
    }

    fn knn_search(&self, text: &str) -> Result<TopDocs> {
        let mut should = Vec::new();
        for name in &self.text_fields {
            let (field, boost) = split_boost(name);
            let mut mlt = self.mlt(self.analyzer.as_deref());
            if let Some(b) = boost {
                mlt.boost_factor = parse_boost(b)?;
            }
            should.push(boolean_clause(mlt.like_texts(field, &[text])?));
        }
        self.search_like(should)
    }

    pub(crate) fn build_list(&self, top: &TopDocs) -> Result<Vec<ClassificationResult<Vec<u8>>>> {
        build_list_from_top_docs(&self.index, top, &self.class_field, self.k, true)
    }
}

impl Classifier<Vec<u8>> for KNearestNeighborClassifier<'_, '_> {
    fn assign_class(&self, text: &str) -> Result<Option<ClassificationResult<Vec<u8>>>> {
        Ok(best(&self.build_list(&self.knn_search(text)?)?))
    }

    fn get_classes(&self, text: &str) -> Result<Option<Vec<ClassificationResult<Vec<u8>>>>> {
        let mut c = self.build_list(&self.knn_search(text)?)?;
        sort_results(&mut c);
        Ok(Some(c))
    }
}

/// `KNearestFuzzyClassifier`: the `k` documents a [`NearestFuzzyQuery`] of
/// the input finds among those with a class.
pub struct KNearestFuzzyClassifier<'s, 'a> {
    index: TrainingIndex<'s, 'a>,
    similarity: SharedSimilarity,
    analyzer: Arc<Analyzer>,
    query: Option<Clause>,
    k: usize,
    class_field: String,
    text_fields: Vec<String>,
}

impl<'s, 'a> KNearestFuzzyClassifier<'s, 'a> {
    /// `new KNearestFuzzyClassifier(reader, similarity, analyzer, query, k,
    /// classFieldName, textFieldNames...)`.
    pub fn new(
        index: TrainingIndex<'s, 'a>,
        similarity: SharedSimilarity,
        analyzer: Arc<Analyzer>,
        query: Option<Clause>,
        k: usize,
        class_field: impl Into<String>,
        text_fields: &[&str],
    ) -> Self {
        Self {
            index,
            similarity,
            analyzer,
            query,
            k,
            class_field: class_field.into(),
            text_fields: text_fields.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn knn_search(&self, text: &str) -> Result<TopDocs> {
        let mut nfq = NearestFuzzyQuery::new(Arc::clone(&self.analyzer));
        for f in &self.text_fields {
            nfq.add_terms(text, f);
        }
        let mut q = BooleanQuery {
            must: vec![
                boolean_clause(nfq.rewrite(&self.index)?),
                any_class(&self.class_field),
            ],
            ..Default::default()
        };
        if let Some(f) = &self.query {
            q.must.push(f.clone());
        }
        let searcher = self.index.searcher(self.similarity.as_deref())?;
        searcher.search(&rewritten(q), self.k)
    }

    fn build_list(&self, top: &TopDocs) -> Result<Vec<ClassificationResult<Vec<u8>>>> {
        build_list_from_top_docs(&self.index, top, &self.class_field, self.k, false)
    }
}

impl Classifier<Vec<u8>> for KNearestFuzzyClassifier<'_, '_> {
    fn assign_class(&self, text: &str) -> Result<Option<ClassificationResult<Vec<u8>>>> {
        Ok(best(&self.build_list(&self.knn_search(text)?)?))
    }

    fn get_classes(&self, text: &str) -> Result<Option<Vec<ClassificationResult<Vec<u8>>>>> {
        let mut c = self.build_list(&self.knn_search(text)?)?;
        sort_results(&mut c);
        Ok(Some(c))
    }
}
