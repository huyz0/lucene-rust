//! `SimpleNaiveBayesClassifier`, `CachingNaiveBayesClassifier` and
//! `BM25NBClassifier`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use lucene_analysis::Analyzer;
use lucene_util::strict_math::log;

use super::{
    best, norm_classification_results, sort_results, tokenize, ClassificationResult, Classifier,
    TrainingIndex,
};
use crate::multi_terms::MultiTerms;
use crate::query::{BooleanQuery, Clause, TermQuery, WildcardQuery};
use crate::reader::Terms;
use crate::{Error, Result};

/// `MultiTerms.getTerms(reader, field)`'s terms, in order.
pub(crate) fn field_terms(
    index: &TrainingIndex<'_, '_>,
    field: &str,
) -> Result<Option<Vec<Vec<u8>>>> {
    let Some(terms) = MultiTerms::get_terms(index.reader, field)? else {
        return Ok(None);
    };
    let mut te = terms.iterator()?;
    let mut out = Vec::new();
    while let Some(t) = te.next()? {
        out.push(t.to_vec());
    }
    Ok(Some(out))
}

/// `terms.getSumDocFreq() / (double) terms.getDocCount()` summed over
/// `fields`: the average number of unique terms per document. A field with
/// no terms is Java's `NullPointerException`.
pub(crate) fn avg_unique_terms(index: &TrainingIndex<'_, '_>, fields: &[String]) -> Result<f64> {
    let mut avg = 0.0;
    for f in fields {
        let terms: Box<dyn Terms + '_> = MultiTerms::get_terms(index.reader, f)?
            .ok_or_else(|| Error::IllegalState(format!("no terms for field {f}")))?;
        avg += terms.sum_doc_freq() as f64 / f64::from(terms.doc_count());
    }
    Ok(avg)
}

/// `indexReader.docFreq(term)`, an `int`.
pub(crate) fn doc_freq(index: &TrainingIndex<'_, '_>, field: &str, term: &[u8]) -> Result<i32> {
    Ok(index.reader.doc_freq(field, term)? as i32)
}

/// `query` and the optional filter, as one more `MUST` clause each.
fn with_filter(mut q: BooleanQuery, filter: &Option<Clause>) -> BooleanQuery {
    if let Some(f) = filter {
        q.must.push(f.clone());
    }
    q
}

/// `SimpleNaiveBayesClassifier`: `P(c) * prod P(w|c)` with add-one
/// smoothing, the counts read off the index by searching.
pub struct SimpleNaiveBayesClassifier<'s, 'a> {
    pub(crate) index: TrainingIndex<'s, 'a>,
    pub(crate) analyzer: Arc<Analyzer>,
    pub(crate) query: Option<Clause>,
    pub(crate) class_field: String,
    pub(crate) text_fields: Vec<String>,
}

impl<'s, 'a> SimpleNaiveBayesClassifier<'s, 'a> {
    /// `new SimpleNaiveBayesClassifier(reader, analyzer, query,
    /// classFieldName, textFieldNames...)`.
    pub fn new(
        index: TrainingIndex<'s, 'a>,
        analyzer: Arc<Analyzer>,
        query: Option<Clause>,
        class_field: impl Into<String>,
        text_fields: &[&str],
    ) -> Self {
        Self {
            index,
            analyzer,
            query,
            class_field: class_field.into(),
            text_fields: text_fields.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn assign_class_normalized_list(
        &self,
        text: &str,
    ) -> Result<Vec<ClassificationResult<Vec<u8>>>> {
        let mut assigned = Vec::new();
        if let Some(classes) = field_terms(&self.index, &self.class_field)? {
            let tokens = tokenize(&self.analyzer, &self.text_fields, text)?;
            let docs_with_class = self.count_docs_with_class()?;
            for class in classes {
                if class.is_empty() {
                    continue;
                }
                let score = self.log_prior(&class, docs_with_class)?
                    + self.log_likelihood(&tokens, &class, docs_with_class)?;
                assigned.push(ClassificationResult::new(class, score));
            }
        }
        Ok(norm_classification_results(assigned))
    }

    /// `countDocsWithClass()`: the class field's `docCount`, or a count of
    /// `class:* AND query` where the codec has none.
    pub(crate) fn count_docs_with_class(&self) -> Result<i32> {
        let doc_count = MultiTerms::get_terms(self.index.reader, &self.class_field)?
            .map(|t| t.doc_count())
            .filter(|&c| c != -1);
        match doc_count {
            Some(c) => Ok(c),
            None => {
                let q = with_filter(
                    BooleanQuery {
                        must: vec![Clause::Wildcard(WildcardQuery::new(
                            self.class_field.as_str(),
                            b"*".to_vec(),
                        ))],
                        ..Default::default()
                    },
                    &self.query,
                );
                Ok(self.index.searcher(None)?.count(&q)? as i32)
            }
        }
    }

    fn log_likelihood(&self, tokens: &[String], class: &[u8], docs_with_class: i32) -> Result<f64> {
        let mut result = 0.0;
        // Java recomputes the denominator per token; it does not depend on
        // the token, so it is read once (and only when there is a token)
        let mut den = None;
        for word in tokens {
            let hits = self.word_freq_for_class(word, class)?;
            let num = f64::from(hits + 1);
            let den = match den {
                Some(d) => d,
                None => {
                    *den.insert(self.text_term_freq_for_class(class)? + f64::from(docs_with_class))
                }
            };
            result += log(num / den);
        }
        Ok(result)
    }

    /// `getTextTermFreqForClass`: the average unique terms per document
    /// times the documents of `class`.
    fn text_term_freq_for_class(&self, class: &[u8]) -> Result<f64> {
        let avg = avg_unique_terms(&self.index, &self.text_fields)?;
        let docs_with_c = doc_freq(&self.index, &self.class_field, class)?;
        Ok(avg * f64::from(docs_with_c))
    }

    /// `getWordFreqForClass`: the documents of `class` (and the filter)
    /// holding `word` in any text field.
    fn word_freq_for_class(&self, word: &str, class: &[u8]) -> Result<i32> {
        let sub = BooleanQuery {
            should: self
                .text_fields
                .iter()
                .map(|f| Clause::Term(TermQuery::new(f.as_str(), word.as_bytes())))
                .collect(),
            ..Default::default()
        };
        let q = with_filter(
            BooleanQuery {
                must: vec![
                    Clause::Boolean(Box::new(sub)),
                    Clause::Term(TermQuery::new(self.class_field.as_str(), class)),
                ],
                ..Default::default()
            },
            &self.query,
        );
        Ok(self.index.searcher(None)?.count(&q)? as i32)
    }

    /// `calculateLogPrior`: `log(docFreq(class)) - log(docsWithClass)`.
    pub(crate) fn log_prior(&self, class: &[u8], docs_with_class: i32) -> Result<f64> {
        let df = doc_freq(&self.index, &self.class_field, class)?;
        Ok(log(f64::from(df)) - log(f64::from(docs_with_class)))
    }
}

impl Classifier<Vec<u8>> for SimpleNaiveBayesClassifier<'_, '_> {
    fn assign_class(&self, text: &str) -> Result<Option<ClassificationResult<Vec<u8>>>> {
        Ok(best(&self.assign_class_normalized_list(text)?))
    }

    fn get_classes(&self, text: &str) -> Result<Option<Vec<ClassificationResult<Vec<u8>>>>> {
        let mut c = self.assign_class_normalized_list(text)?;
        sort_results(&mut c);
        Ok(Some(c))
    }
}

/// `CachingNaiveBayesClassifier`: [`SimpleNaiveBayesClassifier`] with the
/// classes, their term frequencies and each word's per-class hits kept. Like
/// Java's, it scores without the prior, and (built with `justCachedTerms`)
/// gives a word absent from the text fields no hits without searching.
pub struct CachingNaiveBayesClassifier<'s, 'a> {
    base: SimpleNaiveBayesClassifier<'s, 'a>,
    classes: Vec<Vec<u8>>,
    /// `termCClassHitCache`: a cached word's hits per class (indexed as
    /// `classes`), empty -- or all zero, which Java's empty map also is --
    /// until it is first searched with a hit.
    hit_cache: Mutex<HashMap<String, Vec<i32>>>,
    /// `classTermFreq`, indexed as `classes`.
    class_term_freq: Vec<f64>,
    just_cached_terms: bool,
    docs_with_class: i32,
}

impl<'s, 'a> CachingNaiveBayesClassifier<'s, 'a> {
    /// `new CachingNaiveBayesClassifier(...)`: the cache built by
    /// `reInitCache(0, true)`.
    ///
    /// # Errors
    /// What reading the index reports (Java's `RuntimeException`).
    pub fn new(
        index: TrainingIndex<'s, 'a>,
        analyzer: Arc<Analyzer>,
        query: Option<Clause>,
        class_field: impl Into<String>,
        text_fields: &[&str],
    ) -> Result<Self> {
        let mut c = Self {
            base: SimpleNaiveBayesClassifier::new(index, analyzer, query, class_field, text_fields),
            classes: Vec::new(),
            hit_cache: Mutex::new(HashMap::new()),
            class_term_freq: Vec::new(),
            just_cached_terms: true,
            docs_with_class: 0,
        };
        c.re_init_cache(0, true)?;
        Ok(c)
    }

    /// `reInitCache(minTermOccurrenceInCache, justCachedTerms)`: cache the
    /// words whose summed `docFreq` over the text fields exceeds the minimum;
    /// with `just_cached_terms`, other words are never searched.
    ///
    /// # Errors
    /// What reading the index reports; a text or class field with no terms.
    pub fn re_init_cache(
        &mut self,
        min_term_occurrence: i32,
        just_cached_terms: bool,
    ) -> Result<()> {
        self.just_cached_terms = just_cached_terms;
        self.docs_with_class = self.base.count_docs_with_class()?;
        let index = self.base.index;
        let mut frequency: HashMap<String, i64> = HashMap::new();
        for f in &self.base.text_fields {
            let terms = MultiTerms::get_terms(index.reader, f)?
                .ok_or_else(|| Error::IllegalState(format!("no terms for field {f}")))?;
            let mut te = terms.iterator()?;
            while let Some(t) = te.next()? {
                let text = String::from_utf8_lossy(t).into_owned();
                let df = i64::from(te.doc_freq()?);
                *frequency.entry(text).or_insert(0) += df;
            }
        }
        let cache: HashMap<String, Vec<i32>> = frequency
            .into_iter()
            .filter(|&(_, f)| f > i64::from(min_term_occurrence))
            .map(|(w, _)| (w, Vec::new()))
            .collect();
        *self.hit_cache.lock().unwrap_or_else(|e| e.into_inner()) = cache;
        self.classes = field_terms(&index, &self.base.class_field)?.ok_or_else(|| {
            Error::IllegalState(format!("no terms for field {}", self.base.class_field))
        })?;
        self.class_term_freq.clear();
        for class in &self.classes {
            let avg = avg_unique_terms(&index, &self.base.text_fields)?;
            let docs_with_c = doc_freq(&index, &self.base.class_field, class)?;
            self.class_term_freq.push(avg * f64::from(docs_with_c));
        }
        Ok(())
    }

    fn assign_class_normalized_list(
        &self,
        text: &str,
    ) -> Result<Vec<ClassificationResult<Vec<u8>>>> {
        let tokens = tokenize(&self.base.analyzer, &self.base.text_fields, text)?;
        Ok(norm_classification_results(self.log_likelihood(&tokens)?))
    }

    /// `calculateLogLikelihood`: Java moves a class's result to the end of
    /// the list each time it adds a word's term to it, and the list's order
    /// decides the order of equal scores, so this does the same.
    fn log_likelihood(&self, tokens: &[String]) -> Result<Vec<ClassificationResult<Vec<u8>>>> {
        // (class index, score), in Java's list order.
        let mut ret: Vec<(usize, f64)> = (0..self.classes.len()).map(|c| (c, 0.0)).collect();
        let mut hits = Vec::with_capacity(self.classes.len());
        for word in tokens {
            self.word_freq_for_classes(word, &mut hits)?;
            for (class, &n) in hits.iter().enumerate() {
                let num = f64::from(n + 1);
                let den = self.class_term_freq[class] + f64::from(self.docs_with_class);
                let probability = num / den;
                if let Some(at) = ret.iter().position(|&(c, _)| c == class) {
                    let (_, score) = ret.remove(at);
                    ret.push((class, score + log(probability)));
                }
            }
        }
        Ok(ret
            .into_iter()
            .map(|(c, score)| ClassificationResult::new(self.classes[c].clone(), score))
            .collect())
    }

    /// `getWordFreqForClassess`: `word`'s hits per class into `hits`, from
    /// the cache or searched (and then cached, when `word` is a cached term).
    fn word_freq_for_classes(&self, word: &str, hits: &mut Vec<i32>) -> Result<()> {
        hits.clear();
        let cached = {
            let cache = self.hit_cache.lock().unwrap_or_else(|e| e.into_inner());
            match cache.get(word) {
                Some(h) if h.iter().any(|&n| n != 0) => {
                    hits.extend_from_slice(h);
                    return Ok(());
                }
                Some(_) => true,
                None => false,
            }
        };
        hits.resize(self.classes.len(), 0);
        if cached || !self.just_cached_terms {
            let searcher = self.base.index.searcher(None)?;
            for (class, n) in self.classes.iter().zip(hits.iter_mut()) {
                let sub = BooleanQuery {
                    should: self
                        .base
                        .text_fields
                        .iter()
                        .map(|f| Clause::Term(TermQuery::new(f.as_str(), word.as_bytes())))
                        .collect(),
                    ..Default::default()
                };
                let q = with_filter(
                    BooleanQuery {
                        must: vec![
                            Clause::Boolean(Box::new(sub)),
                            Clause::Term(TermQuery::new(
                                self.base.class_field.as_str(),
                                class.as_slice(),
                            )),
                        ],
                        ..Default::default()
                    },
                    &self.base.query,
                );
                *n = searcher.count(&q)? as i32;
            }
            if cached {
                let mut cache = self.hit_cache.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(e) = cache.get_mut(word) {
                    e.clone_from(hits);
                }
            }
        }
        Ok(())
    }
}

impl Classifier<Vec<u8>> for CachingNaiveBayesClassifier<'_, '_> {
    fn assign_class(&self, text: &str) -> Result<Option<ClassificationResult<Vec<u8>>>> {
        Ok(best(&self.assign_class_normalized_list(text)?))
    }

    fn get_classes(&self, text: &str) -> Result<Option<Vec<ClassificationResult<Vec<u8>>>>> {
        let mut c = self.assign_class_normalized_list(text)?;
        sort_results(&mut c);
        Ok(Some(c))
    }
}

/// `BM25NBClassifier`: naive Bayes whose prior and word probabilities are
/// BM25 scores -- the best score of the class's documents, alone and with
/// the word.
pub struct Bm25NbClassifier<'s, 'a> {
    index: TrainingIndex<'s, 'a>,
    analyzer: Arc<Analyzer>,
    query: Option<Clause>,
    class_field: String,
    text_fields: Vec<String>,
}

impl<'s, 'a> Bm25NbClassifier<'s, 'a> {
    /// `new BM25NBClassifier(reader, analyzer, query, classFieldName,
    /// textFieldNames...)`.
    pub fn new(
        index: TrainingIndex<'s, 'a>,
        analyzer: Arc<Analyzer>,
        query: Option<Clause>,
        class_field: impl Into<String>,
        text_fields: &[&str],
    ) -> Self {
        Self {
            index,
            analyzer,
            query,
            class_field: class_field.into(),
            text_fields: text_fields.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn assign_class_normalized_list(
        &self,
        text: &str,
    ) -> Result<Vec<ClassificationResult<Vec<u8>>>> {
        let classes = field_terms(&self.index, &self.class_field)?.ok_or_else(|| {
            Error::IllegalState(format!("no terms for field {}", self.class_field))
        })?;
        let tokens = tokenize(&self.analyzer, &self.text_fields, text)?;
        let mut assigned = Vec::new();
        for class in classes {
            if class.is_empty() {
                continue;
            }
            let score = self.log_prior(&class)? + self.log_likelihood(&tokens, &class)?;
            assigned.push(ClassificationResult::new(class, score));
        }
        Ok(norm_classification_results(assigned))
    }

    fn log_likelihood(&self, tokens: &[String], class: &[u8]) -> Result<f64> {
        let mut result = 0.0;
        for word in tokens {
            result += log(self.term_prob_for_class(class, word)?);
        }
        Ok(result)
    }

    /// `getTermProbForClass`: the best score of `class:c AND (word...)`,
    /// `1` when nothing matches.
    fn term_prob_for_class(&self, class: &[u8], word: &str) -> Result<f64> {
        let mut q = BooleanQuery {
            must: vec![Clause::Term(TermQuery::new(
                self.class_field.as_str(),
                class,
            ))],
            should: self
                .text_fields
                .iter()
                .map(|f| Clause::Term(TermQuery::new(f.as_str(), word.as_bytes())))
                .collect(),
            ..Default::default()
        };
        q = with_filter(q, &self.query);
        let top = self.index.searcher(None)?.search(&q, 1)?;
        Ok(match top.score_docs.first() {
            Some(sd) if top.total_hits.value > 0 => f64::from(sd.score),
            _ => 1.0,
        })
    }

    /// `calculateLogPrior`: the log of the best score of `class:c`, `0` when
    /// nothing matches.
    fn log_prior(&self, class: &[u8]) -> Result<f64> {
        let q = with_filter(
            BooleanQuery {
                must: vec![Clause::Term(TermQuery::new(
                    self.class_field.as_str(),
                    class,
                ))],
                ..Default::default()
            },
            &self.query,
        );
        let top = self.index.searcher(None)?.search(&q, 1)?;
        Ok(match top.score_docs.first() {
            Some(sd) if top.total_hits.value > 0 => log(f64::from(sd.score)),
            _ => 0.0,
        })
    }
}

impl Classifier<Vec<u8>> for Bm25NbClassifier<'_, '_> {
    /// `assignClass`: the first of the normalised list (sorted, best first);
    /// an index without classes is Java's `IndexOutOfBoundsException`.
    fn assign_class(&self, text: &str) -> Result<Option<ClassificationResult<Vec<u8>>>> {
        let list = self.assign_class_normalized_list(text)?;
        match list.into_iter().next() {
            Some(first) => Ok(Some(first)),
            None => Err(Error::IllegalArgument(
                "Index 0 out of bounds for length 0".into(),
            )),
        }
    }

    fn get_classes(&self, text: &str) -> Result<Option<Vec<ClassificationResult<Vec<u8>>>>> {
        let mut c = self.assign_class_normalized_list(text)?;
        sort_results(&mut c);
        Ok(Some(c))
    }
}
