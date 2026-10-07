//! Port of `org.apache.lucene.analysis.query.QueryAutoStopWordAnalyzer`
//! (analysis-common): an `AnalyzerWrapper` that drops, per field, the terms
//! found in more than a given number of an index's documents.
//!
//! It lives here rather than in `lucene-analysis` because its constructors
//! read an index: `FieldInfos.getIndexedFields`, `MultiTerms` and their
//! `docFreq`s ([`crate::multi_terms`]), which `lucene-analysis` sits below.
//! As in Java, a term's document frequency counts deleted documents while
//! the percentage constructors scale `numDocs()`, which does not.
//!
//! Differs: a term whose bytes are not UTF-8 becomes its lossy decoding,
//! where Java's `CharsRefBuilder.copyUTF8Bytes` decodes without checking.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lucene_analysis::{Analyzer, AnalyzerWrapper, CharArraySet, StopFilter, TokenStreamComponents};

use crate::multi_terms::{indexed_fields, MultiTerms};
use crate::reader::IndexReader;
use crate::Result;

/// `QueryAutoStopWordAnalyzer.defaultMaxDocFreqPercent`.
pub const DEFAULT_MAX_DOC_FREQ_PERCENT: f32 = 0.4;

/// `QueryAutoStopWordAnalyzer`. Build the [`Analyzer`] with
/// [`Self::into_analyzer`] (Java passes the delegate's reuse strategy up).
pub struct QueryAutoStopWordAnalyzer {
    delegate: Analyzer,
    /// Each field's stop words, as the `StopFilter`'s set.
    stop_words_per_field: HashMap<String, Arc<CharArraySet>>,
    /// The same words, for `getStopWords`.
    words_per_field: HashMap<String, HashSet<String>>,
}

impl QueryAutoStopWordAnalyzer {
    /// `QueryAutoStopWordAnalyzer(Analyzer, IndexReader)`: every indexed
    /// field, [`DEFAULT_MAX_DOC_FREQ_PERCENT`].
    ///
    /// # Errors
    /// The index's terms fail to read.
    pub fn new<R: IndexReader + ?Sized>(delegate: Analyzer, reader: &R) -> Result<Self> {
        Self::with_max_percent_docs(delegate, reader, DEFAULT_MAX_DOC_FREQ_PERCENT)
    }

    /// `QueryAutoStopWordAnalyzer(Analyzer, IndexReader, int maxDocFreq)`.
    ///
    /// # Errors
    /// The index's terms fail to read.
    pub fn with_max_doc_freq<R: IndexReader + ?Sized>(
        delegate: Analyzer,
        reader: &R,
        max_doc_freq: i32,
    ) -> Result<Self> {
        let fields = indexed_fields(reader);
        Self::for_fields(delegate, reader, &fields, max_doc_freq)
    }

    /// `QueryAutoStopWordAnalyzer(Analyzer, IndexReader, float maxPercentDocs)`.
    ///
    /// # Errors
    /// The index's terms fail to read.
    pub fn with_max_percent_docs<R: IndexReader + ?Sized>(
        delegate: Analyzer,
        reader: &R,
        max_percent_docs: f32,
    ) -> Result<Self> {
        let fields = indexed_fields(reader);
        Self::for_fields_percent(delegate, reader, &fields, max_percent_docs)
    }

    /// `QueryAutoStopWordAnalyzer(Analyzer, IndexReader, Collection<String>,
    /// float maxPercentDocs)`: `(int) (reader.numDocs() * maxPercentDocs)`.
    ///
    /// # Errors
    /// The index's terms fail to read.
    pub fn for_fields_percent<R: IndexReader + ?Sized, S: AsRef<str>>(
        delegate: Analyzer,
        reader: &R,
        fields: &[S],
        max_percent_docs: f32,
    ) -> Result<Self> {
        // Java's float product and saturating `(int)` cast.
        let max_doc_freq = (reader.num_docs() as f32 * max_percent_docs) as i32;
        Self::for_fields(delegate, reader, fields, max_doc_freq)
    }

    /// `QueryAutoStopWordAnalyzer(Analyzer, IndexReader, Collection<String>,
    /// int maxDocFreq)`: each field's terms with `docFreq() > maxDocFreq`
    /// (a field without terms gets an empty set).
    ///
    /// # Errors
    /// The index's terms fail to read.
    pub fn for_fields<R: IndexReader + ?Sized, S: AsRef<str>>(
        delegate: Analyzer,
        reader: &R,
        fields: &[S],
        max_doc_freq: i32,
    ) -> Result<Self> {
        let mut words_per_field = HashMap::new();
        for field in fields {
            let field = field.as_ref();
            let mut words = HashSet::new();
            if let Some(terms) = MultiTerms::get_terms(reader, field)? {
                let mut te = terms.iterator()?;
                while let Some(text) = te.next()? {
                    let word = String::from_utf8_lossy(text).into_owned();
                    if te.doc_freq()? > max_doc_freq {
                        words.insert(word);
                    }
                }
            }
            words_per_field.insert(field.to_string(), words);
        }
        let stop_words_per_field = words_per_field
            .iter()
            .map(|(f, w)| (f.clone(), Arc::new(CharArraySet::from_words(w, false))))
            .collect();
        Ok(QueryAutoStopWordAnalyzer {
            delegate,
            stop_words_per_field,
            words_per_field,
        })
    }

    /// `getStopWords(String fieldName)`: empty for a field not analysed
    /// (Java's array order is its `HashSet`'s; this is sorted).
    pub fn stop_words(&self, field_name: &str) -> Vec<String> {
        let mut w: Vec<String> = self
            .words_per_field
            .get(field_name)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();
        w.sort();
        w
    }

    /// `getStopWords()`: every `(field, word)` pair, sorted.
    pub fn all_stop_words(&self) -> Vec<(String, String)> {
        let mut all: Vec<(String, String)> = self
            .words_per_field
            .iter()
            .flat_map(|(f, ws)| ws.iter().map(move |w| (f.clone(), w.clone())))
            .collect();
        all.sort();
        all
    }

    /// The [`Analyzer`], with the delegate's reuse strategy.
    pub fn into_analyzer(self) -> Analyzer {
        let strategy = self.delegate.reuse_strategy();
        Analyzer::with_reuse_strategy(self, strategy)
    }
}

impl AnalyzerWrapper for QueryAutoStopWordAnalyzer {
    fn wrapped_analyzer(&self, _field_name: &str) -> &Analyzer {
        &self.delegate
    }

    // Java: QueryAutoStopWordAnalyzer.wrapComponents
    fn wrap_components(
        &self,
        field_name: &str,
        components: TokenStreamComponents,
    ) -> TokenStreamComponents {
        let Some(stop_words) = self.stop_words_per_field.get(field_name) else {
            return components;
        };
        let (source, sink) = components.into_parts();
        TokenStreamComponents::from_parts(
            source,
            Box::new(StopFilter::new(sink, Arc::clone(stop_words))),
        )
    }
}
