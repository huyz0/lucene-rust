//! `lucene-queries`' more-like-this (`org.apache.lucene.queries.mlt`,
//! Lucene 10.5.0): [`MoreLikeThis`], which picks a document's (or a text's)
//! most interesting terms -- term frequency in the source times
//! `ClassicSimilarity`'s `idf` over the reader -- and makes a disjunction of
//! them; and [`MoreLikeThisQuery`], which does so for a text when the
//! searcher rewrites it.
//!
//! The source's terms come from its term vectors where the field has them,
//! else from its stored values (or the given text) through the analyzer.
//!
//! # What has to be Java's exactly
//!
//! The candidate terms are visited in a `HashMap`'s iteration order (fields,
//! then each field's words), and the bounded queue keeps the first of two
//! equal scores; so which of several equally interesting terms is kept, and
//! the order of the query's clauses among equal scores, follow that order.
//! [`java_hash_order`] reproduces it: Java's `String.hashCode`, spread and
//! masked to the table size a `HashMap` grows to for that many entries,
//! insertion order within a bucket. The queue is `util.PriorityQueue`'s heap
//! ([`crate::intervals::iterators::IndexQueue`]).

use std::collections::HashMap;

use lucene_analysis::Analyzer;

use crate::directory_reader::DirectoryReader;
use crate::index_searcher::IndexSearcher;
use crate::intervals::iterators::IndexQueue;
use crate::query::{BooleanQuery, BoostQuery, Clause, TermQuery};
use crate::similarities::ClassicSimilarity;
use crate::{Error, Result};

/// `MoreLikeThis.DEFAULT_MAX_NUM_TOKENS_PARSED`.
pub const DEFAULT_MAX_NUM_TOKENS_PARSED: usize = 5000;
/// `MoreLikeThis.DEFAULT_MIN_TERM_FREQ`.
pub const DEFAULT_MIN_TERM_FREQ: i32 = 2;
/// `MoreLikeThis.DEFAULT_MIN_DOC_FREQ`.
pub const DEFAULT_MIN_DOC_FREQ: i32 = 5;
/// `MoreLikeThis.DEFAULT_MAX_DOC_FREQ`.
pub const DEFAULT_MAX_DOC_FREQ: i32 = i32::MAX;
/// `MoreLikeThis.DEFAULT_MAX_QUERY_TERMS`.
pub const DEFAULT_MAX_QUERY_TERMS: usize = 25;

/// What [`MoreLikeThis`] reads from the index: term statistics over the
/// whole reader.
pub trait TermStatsReader {
    /// `IndexReader.docFreq(term)`.
    fn doc_freq(&self, field: &str, term: &[u8]) -> Result<i64>;
    /// `IndexReader.getDocCount(field)`.
    fn doc_count(&self, field: &str) -> Result<i64>;
    /// `IndexReader.maxDoc()`.
    fn max_doc(&self) -> i64;
}

impl TermStatsReader for DirectoryReader {
    fn doc_freq(&self, field: &str, term: &[u8]) -> Result<i64> {
        Ok(DirectoryReader::doc_freq(self, field, term)?)
    }
    fn doc_count(&self, field: &str) -> Result<i64> {
        Ok(self
            .segment_readers()
            .iter()
            .filter_map(|r| r.field_stats(field))
            .map(|(_, doc_count)| i64::from(doc_count))
            .sum())
    }
    fn max_doc(&self) -> i64 {
        i64::from(DirectoryReader::max_doc(self))
    }
}

impl TermStatsReader for IndexSearcher<'_, '_> {
    fn doc_freq(&self, field: &str, term: &[u8]) -> Result<i64> {
        let mut df = 0i64;
        for seg in self.segments() {
            if let Some(ft) = seg.fields.field(field) {
                if let Some(stats) = ft.try_seek_exact(term)? {
                    df = df.saturating_add(i64::from(stats.doc_freq));
                }
            }
        }
        Ok(df)
    }
    fn doc_count(&self, field: &str) -> Result<i64> {
        Ok(self
            .segments()
            .iter()
            .filter_map(|s| s.fields.field(field))
            .map(|ft| i64::from(ft.doc_count))
            .sum())
    }
    fn max_doc(&self) -> i64 {
        i64::from(IndexSearcher::max_doc(self))
    }
}

/// Java's `String.hashCode`.
fn java_string_hash(s: &str) -> i32 {
    s.encode_utf16()
        .fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(i32::from(c)))
}

/// The order a `java.util.HashMap` of `String` keys, filled in `keys`'
/// order, iterates them: by bucket (`(h ^ (h >>> 16)) & (capacity - 1)`,
/// the capacity it has grown to from 16 at load factor 0.75), insertion
/// order within a bucket (a resize keeps it). Returns indices into `keys`.
pub fn java_hash_order(keys: &[&str]) -> Vec<usize> {
    let mut capacity: usize = 16;
    while keys.len() > capacity / 4 * 3 {
        capacity *= 2;
    }
    let mask = (capacity - 1) as u32;
    let mut order: Vec<(u32, usize)> = keys
        .iter()
        .enumerate()
        .map(|(i, k)| {
            let h = java_string_hash(k) as u32;
            ((h ^ (h >> 16)) & mask, i)
        })
        .collect();
    order.sort_unstable();
    order.into_iter().map(|(_, i)| i).collect()
}

/// `MoreLikeThis.ScoreTerm`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreTerm {
    pub word: String,
    pub top_field: String,
    pub score: f32,
}

/// A field's words and their frequencies, in first-seen order (a
/// `HashMap<String, Int>` and the order its entries were made).
#[derive(Debug, Default, Clone)]
struct TermFreqs {
    words: Vec<String>,
    freqs: Vec<i32>,
    index: HashMap<String, usize>,
}

impl TermFreqs {
    fn add(&mut self, word: &str, freq: i32) {
        match self.index.get(word) {
            Some(&i) => self.freqs[i] = self.freqs[i].wrapping_add(freq),
            None => {
                self.index.insert(word.to_string(), self.words.len());
                self.words.push(word.to_string());
                self.freqs.push(freq);
            }
        }
    }
}

/// `Map<String, Map<String, Int>>`, fields in first-seen order.
#[derive(Debug, Default)]
struct FieldTermFreqs {
    fields: Vec<(String, TermFreqs)>,
}

impl FieldTermFreqs {
    fn field(&mut self, name: &str) -> &mut TermFreqs {
        if let Some(i) = self.fields.iter().position(|(f, _)| f == name) {
            return &mut self.fields[i].1;
        }
        self.fields.push((name.to_string(), TermFreqs::default()));
        let last = self.fields.len() - 1;
        &mut self.fields[last].1
    }
}

/// `MoreLikeThis`: the settings, and the reader statistics are read from.
pub struct MoreLikeThis<'r> {
    stats: &'r dyn TermStatsReader,
    reader: Option<&'r DirectoryReader>,
    pub analyzer: Option<&'r Analyzer>,
    pub min_term_freq: i32,
    pub min_doc_freq: i32,
    pub max_doc_freq: i32,
    pub boost: bool,
    pub boost_factor: f32,
    pub field_names: Option<Vec<String>>,
    pub max_num_tokens_parsed: usize,
    pub min_word_len: usize,
    pub max_word_len: usize,
    pub stop_words: Option<std::collections::HashSet<String>>,
    pub max_query_terms: usize,
}

impl<'r> MoreLikeThis<'r> {
    /// `new MoreLikeThis(ir)`, with `ClassicSimilarity`.
    pub fn new(reader: &'r DirectoryReader) -> Self {
        let mut m = Self::over(reader);
        m.reader = Some(reader);
        m
    }

    /// A `MoreLikeThis` over term statistics alone: `like(field, text)`
    /// only (a document's terms need its reader).
    pub fn over(stats: &'r dyn TermStatsReader) -> Self {
        MoreLikeThis {
            stats,
            reader: None,
            analyzer: None,
            min_term_freq: DEFAULT_MIN_TERM_FREQ,
            min_doc_freq: DEFAULT_MIN_DOC_FREQ,
            max_doc_freq: DEFAULT_MAX_DOC_FREQ,
            boost: false,
            boost_factor: 1.0,
            field_names: Some(vec!["contents".to_string()]),
            max_num_tokens_parsed: DEFAULT_MAX_NUM_TOKENS_PARSED,
            min_word_len: 0,
            max_word_len: 0,
            stop_words: None,
            max_query_terms: DEFAULT_MAX_QUERY_TERMS,
        }
    }

    /// `setMaxDocFreqPct(maxPercentage)`.
    pub fn set_max_doc_freq_pct(&mut self, max_percentage: i32) -> Result<()> {
        let v = i64::from(max_percentage).saturating_mul(self.stats.max_doc()) / 100;
        self.max_doc_freq = i32::try_from(v)
            .map_err(|_| Error::IllegalArgument(format!("integer overflow: {v}")))?;
        Ok(())
    }

    /// `like(docNum)`.
    ///
    /// # Errors
    /// [`Error::Unsupported`] when a field without term vectors must be
    /// analyzed and no analyzer is set; a document outside the reader; and
    /// what reading the index reports.
    pub fn like_doc(&mut self, doc: i32) -> Result<BooleanQuery> {
        self.resolve_field_names()?;
        let q = self.retrieve_terms_of_doc(doc)?;
        Ok(self.create_query(q))
    }

    /// `like(fieldName, readers...)`: the texts analyzed as `field`.
    ///
    /// # Errors
    /// [`Error::Unsupported`] without an analyzer; the analyzer's errors.
    pub fn like_texts(&self, field: &str, texts: &[&str]) -> Result<BooleanQuery> {
        let mut freqs = FieldTermFreqs::default();
        for t in texts {
            self.add_text_frequencies(t, &mut freqs, field)?;
        }
        let q = self.create_queue(&freqs)?;
        Ok(self.create_query(q))
    }

    /// `like(Map<String, Collection<Object>>)`: each listed field's values,
    /// analyzed.
    ///
    /// # Errors
    /// As [`Self::like_texts`].
    pub fn like_fields(&mut self, values: &[(&str, Vec<String>)]) -> Result<BooleanQuery> {
        self.resolve_field_names()?;
        let mut freqs = FieldTermFreqs::default();
        for name in self.field_names.clone().unwrap_or_default() {
            let Some((_, vs)) = values.iter().find(|(f, _)| *f == name) else {
                continue;
            };
            for v in vs {
                self.add_text_frequencies(v, &mut freqs, &name)?;
            }
        }
        let q = self.create_queue(&freqs)?;
        Ok(self.create_query(q))
    }

    /// `retrieveInterestingTerms(docNum)`.
    ///
    /// # Errors
    /// As [`Self::like_doc`].
    pub fn retrieve_interesting_terms(&mut self, doc: i32) -> Result<Vec<String>> {
        let q = self.retrieve_terms_of_doc(doc)?;
        Ok(Self::interesting(q, self.max_query_terms))
    }

    /// `retrieveInterestingTerms(reader, fieldName)`.
    ///
    /// # Errors
    /// As [`Self::like_texts`].
    pub fn retrieve_interesting_terms_of_text(
        &self,
        text: &str,
        field: &str,
    ) -> Result<Vec<String>> {
        let mut freqs = FieldTermFreqs::default();
        self.add_text_frequencies(text, &mut freqs, field)?;
        let q = self.create_queue(&freqs)?;
        Ok(Self::interesting(q, self.max_query_terms))
    }

    fn interesting(mut q: FreqQueue, lim: usize) -> Vec<String> {
        let mut out = Vec::new();
        let mut lim = lim;
        while let Some(t) = q.pop() {
            if lim == 0 {
                break;
            }
            lim -= 1;
            out.push(t.word);
        }
        out
    }

    /// `fieldNames == null`: every indexed field of the reader.
    fn resolve_field_names(&mut self) -> Result<()> {
        if self.field_names.is_some() {
            return Ok(());
        }
        let reader = self
            .reader
            .ok_or_else(|| Error::IllegalState("no reader to list the indexed fields of".into()))?;
        let mut names: Vec<String> = Vec::new();
        for r in reader.segment_readers() {
            for fi in &r.field_infos().fields {
                if fi.index_options != lucene_codecs::field_infos::IndexOptions::None
                    && !names.contains(&fi.name)
                {
                    names.push(fi.name.clone());
                }
            }
        }
        self.field_names = Some(names);
        Ok(())
    }

    /// `createQuery(queue)`: the terms, least interesting first, as
    /// `SHOULD` term queries, boosted relative to the first when `boost`.
    fn create_query(&self, mut q: FreqQueue) -> BooleanQuery {
        let mut query = BooleanQuery::new();
        let mut best_score = -1.0f32;
        while let Some(st) = q.pop() {
            let mut tq = Clause::Term(TermQuery::new(st.top_field.clone(), st.word.into_bytes()));
            if self.boost {
                if best_score == -1.0 {
                    best_score = st.score;
                }
                let my_score = st.score;
                tq = Clause::Boost(Box::new(BoostQuery::new(
                    tq,
                    self.boost_factor * my_score / best_score,
                )));
            }
            if query.should.len() >= crate::extended_query::MAX_CLAUSE_COUNT {
                break;
            }
            query.should.push(tq);
        }
        query
    }

    /// `createQueue(perFieldTermFrequencies)`.
    fn create_queue(&self, freqs: &FieldTermFreqs) -> Result<FreqQueue> {
        let total: usize = freqs.fields.iter().map(|(_, t)| t.words.len()).sum();
        let limit = self.max_query_terms.min(total);
        let mut queue = FreqQueue::new(limit);
        let field_keys: Vec<&str> = freqs.fields.iter().map(|(f, _)| f.as_str()).collect();
        for fi in java_hash_order(&field_keys) {
            let (field, words) = &freqs.fields[fi];
            let mut num_docs = self.stats.doc_count(field)?;
            if num_docs == -1 {
                num_docs = self.stats.max_doc();
            }
            let word_keys: Vec<&str> = words.words.iter().map(String::as_str).collect();
            for wi in java_hash_order(&word_keys) {
                let word = &words.words[wi];
                let tf = words.freqs[wi];
                if self.min_term_freq > 0 && tf < self.min_term_freq {
                    continue;
                }
                let doc_freq = self.stats.doc_freq(field, word.as_bytes())?;
                if self.min_doc_freq > 0 && doc_freq < i64::from(self.min_doc_freq) {
                    continue;
                }
                if doc_freq > i64::from(self.max_doc_freq) {
                    continue;
                }
                if doc_freq == 0 {
                    continue;
                }
                let idf = ClassicSimilarity::idf(doc_freq, num_docs);
                let score = tf as f32 * idf;
                if queue.size() < limit {
                    queue.add(ScoreTerm {
                        word: word.clone(),
                        top_field: field.clone(),
                        score,
                    });
                } else if let Some(top) = queue.top_mut() {
                    if top.score < score {
                        *top = ScoreTerm {
                            word: word.clone(),
                            top_field: field.clone(),
                            score,
                        };
                        queue.update_top();
                    }
                }
            }
        }
        Ok(queue)
    }

    /// `retrieveTerms(docNum)`: each field's term vector in the document,
    /// else its stored values analyzed.
    fn retrieve_terms_of_doc(&self, doc: i32) -> Result<FreqQueue> {
        let reader = self
            .reader
            .ok_or_else(|| Error::IllegalState("like(docNum) needs the index reader".into()))?;
        let (seg, local) = locate(reader, doc)?;
        let mut freqs = FieldTermFreqs::default();
        let field_names = self.field_names.clone().unwrap_or_default();
        let vectors = match seg.term_vectors_reader()? {
            Some(tv) => tv.document(local)?,
            None => None,
        };
        for name in &field_names {
            let number = seg.field_infos().field_by_name(name).map(|fi| fi.number);
            let vector = vectors
                .as_ref()
                .and_then(|v| v.fields.iter().find(|f| Some(f.field_number) == number));
            match vector {
                Some(v) => {
                    let tf = freqs.field(name);
                    for t in &v.terms {
                        let term = String::from_utf8_lossy(&t.term);
                        if self.is_noise_word(&term) {
                            continue;
                        }
                        tf.add(&term, t.freq);
                    }
                }
                None => {
                    let document = seg.stored_document(local)?.unwrap_or_default();
                    let values: Vec<String> = document
                        .fields
                        .iter()
                        .filter(|f| Some(f.field_number) == number)
                        .filter_map(|f| match &f.value {
                            lucene_codecs::stored_fields::FieldValue::String(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect();
                    if values.is_empty() {
                        // `field2termFreqMap` gets no entry for the field.
                        continue;
                    }
                    for v in values {
                        self.add_text_frequencies(&v, &mut freqs, name)?;
                    }
                }
            }
        }
        self.create_queue(&freqs)
    }

    /// `addTermFrequencies(reader, map, fieldName)`: the text analyzed as
    /// `field`, at most `maxNumTokensParsed` tokens.
    fn add_text_frequencies(
        &self,
        text: &str,
        freqs: &mut FieldTermFreqs,
        field: &str,
    ) -> Result<()> {
        let analyzer = self.analyzer.ok_or_else(|| {
            Error::Unsupported(
                "To use MoreLikeThis without term vectors, you must provide an Analyzer".into(),
            )
        })?;
        let analysis = |e: lucene_analysis::AnalysisError| Error::IllegalArgument(e.to_string());
        let tf = freqs.field(field);
        let mut ts = analyzer.token_stream(field, text).map_err(analysis)?;
        use lucene_analysis::TokenStream;
        ts.reset().map_err(analysis)?;
        let mut token_count = 0usize;
        while ts.increment_token().map_err(analysis)? {
            let atts = ts.attributes();
            let word = atts.term().to_string();
            token_count += 1;
            if token_count > self.max_num_tokens_parsed {
                break;
            }
            if self.is_noise_word(&word) {
                continue;
            }
            tf.add(&word, atts.term_frequency());
        }
        ts.end().map_err(analysis)?;
        Ok(())
    }

    /// `isNoiseWord(term)`: too short, too long, or a stop word (lengths in
    /// UTF-16 units, as `String.length()`).
    fn is_noise_word(&self, term: &str) -> bool {
        let len = term.encode_utf16().count();
        if self.min_word_len > 0 && len < self.min_word_len {
            return true;
        }
        if self.max_word_len > 0 && len > self.max_word_len {
            return true;
        }
        self.stop_words.as_ref().is_some_and(|s| s.contains(term))
    }
}

/// The segment holding global `doc`, and `doc` within it.
fn locate(
    reader: &DirectoryReader,
    doc: i32,
) -> Result<(&crate::directory_reader::SegmentReader, i32)> {
    let mut base = 0i32;
    for r in reader.segment_readers() {
        let max = r.max_doc;
        if doc >= base && doc < base.saturating_add(max) {
            return Ok((r, doc - base));
        }
        base = base.saturating_add(max);
    }
    Err(Error::IllegalArgument(format!(
        "doc {doc} is in no segment"
    )))
}

/// `MoreLikeThis.FreqQ`: `util.PriorityQueue<ScoreTerm>` by score.
struct FreqQueue {
    terms: Vec<ScoreTerm>,
    heap: IndexQueue,
}

impl FreqQueue {
    fn new(max_size: usize) -> Self {
        FreqQueue {
            terms: Vec::new(),
            heap: IndexQueue::new(max_size),
        }
    }

    fn size(&self) -> usize {
        self.heap.size()
    }

    fn add(&mut self, t: ScoreTerm) {
        self.terms.push(t);
        let i = self.terms.len() - 1;
        let terms = &self.terms;
        self.heap.add(i, &|a, b| terms[a].score < terms[b].score);
    }

    fn top_mut(&mut self) -> Option<&mut ScoreTerm> {
        let i = self.heap.top()?;
        self.terms.get_mut(i)
    }

    fn update_top(&mut self) {
        let terms = &self.terms;
        self.heap
            .update_top(&|a, b| terms[a].score < terms[b].score);
    }

    fn pop(&mut self) -> Option<ScoreTerm> {
        let terms = &self.terms;
        let i = self.heap.pop(&|a, b| terms[a].score < terms[b].score)?;
        Some(self.terms[i].clone())
    }
}

/// `MoreLikeThisQuery`: the documents like a text, its interesting terms
/// found when the searcher rewrites it.
#[derive(Clone)]
pub struct MoreLikeThisQuery {
    pub like_text: String,
    pub more_like_fields: Vec<String>,
    pub analyzer: std::sync::Arc<Analyzer>,
    pub field_name: String,
    pub percent_terms_to_match: f32,
    pub min_term_frequency: i32,
    pub max_query_terms: usize,
    pub stop_words: Option<std::collections::HashSet<String>>,
    pub min_doc_freq: i32,
}

impl std::fmt::Debug for MoreLikeThisQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoreLikeThisQuery")
            .field("like_text", &self.like_text)
            .field("more_like_fields", &self.more_like_fields)
            .field("field_name", &self.field_name)
            .field("percent_terms_to_match", &self.percent_terms_to_match)
            .field("min_term_frequency", &self.min_term_frequency)
            .field("max_query_terms", &self.max_query_terms)
            .field("stop_words", &self.stop_words)
            .field("min_doc_freq", &self.min_doc_freq)
            .finish_non_exhaustive()
    }
}

impl PartialEq for MoreLikeThisQuery {
    /// `MoreLikeThisQuery.equals`: every setting, the analyzer by identity.
    fn eq(&self, o: &Self) -> bool {
        self.like_text == o.like_text
            && self.more_like_fields == o.more_like_fields
            && std::sync::Arc::ptr_eq(&self.analyzer, &o.analyzer)
            && self.field_name == o.field_name
            && self.percent_terms_to_match == o.percent_terms_to_match
            && self.min_term_frequency == o.min_term_frequency
            && self.max_query_terms == o.max_query_terms
            && self.stop_words == o.stop_words
            && self.min_doc_freq == o.min_doc_freq
    }
}

impl MoreLikeThisQuery {
    /// `new MoreLikeThisQuery(likeText, moreLikeFields, analyzer, fieldName)`.
    pub fn new(
        like_text: impl Into<String>,
        more_like_fields: Vec<String>,
        analyzer: std::sync::Arc<Analyzer>,
        field_name: impl Into<String>,
    ) -> Self {
        MoreLikeThisQuery {
            like_text: like_text.into(),
            more_like_fields,
            analyzer,
            field_name: field_name.into(),
            percent_terms_to_match: 0.3,
            min_term_frequency: 1,
            max_query_terms: 5,
            stop_words: None,
            min_doc_freq: -1,
        }
    }

    /// `rewrite(indexSearcher)`: `like(fieldName, likeText)`, with at least
    /// `percentTermsToMatch` of its clauses required.
    ///
    /// # Errors
    /// The analyzer's, and reading the term dictionaries.
    pub fn rewrite(&self, searcher: &IndexSearcher<'_, '_>) -> Result<Clause> {
        let mut mlt = MoreLikeThis::over(searcher);
        mlt.field_names = Some(self.more_like_fields.clone());
        mlt.analyzer = Some(&self.analyzer);
        mlt.min_term_freq = self.min_term_frequency;
        if self.min_doc_freq >= 0 {
            mlt.min_doc_freq = self.min_doc_freq;
        }
        mlt.max_query_terms = self.max_query_terms;
        mlt.stop_words = self.stop_words.clone();
        let mut bq = mlt.like_texts(&self.field_name, &[&self.like_text])?;
        bq.minimum_should_match = (bq.should.len() as f32 * self.percent_terms_to_match) as usize;
        Ok(Clause::Boolean(Box::new(bq)))
    }
}

impl std::fmt::Display for MoreLikeThisQuery {
    /// `toString(field)`: `like:` and the text.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "like:{}", self.like_text)
    }
}

impl From<MoreLikeThisQuery> for Clause {
    fn from(q: MoreLikeThisQuery) -> Self {
        Clause::Extended(Box::new(
            crate::extended_query::ExtendedQuery::MoreLikeThis(q),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Java's `String.hashCode` and a `HashMap`'s iteration order, as
    /// `new HashMap<String, ?>()` filled with these keys iterates them
    /// (printed by Lucene's JDK for the same keys).
    #[test]
    fn hash_order_is_a_java_hash_maps() {
        assert_eq!(java_string_hash("apple"), 93029210);
        assert_eq!(java_string_hash(""), 0);
        assert_eq!(java_string_hash("Aa"), java_string_hash("BB"));
        let keys = ["red", "blue", "green", "fast", "slow"];
        let order: Vec<&str> = java_hash_order(&keys).iter().map(|&i| keys[i]).collect();
        assert_eq!(order.len(), 5);
        // Colliding keys keep their insertion order.
        let same = ["Aa", "BB"];
        assert_eq!(java_hash_order(&same), [0, 1]);
        let rev = ["BB", "Aa"];
        assert_eq!(java_hash_order(&rev), [0, 1]);
        // Past 12 entries the table doubles.
        let many: Vec<String> = (0..40).map(|i| format!("w{i}")).collect();
        let refs: Vec<&str> = many.iter().map(String::as_str).collect();
        let order = java_hash_order(&refs);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..40).collect::<Vec<_>>());
    }

    #[test]
    fn the_queue_keeps_the_highest_scores_and_pops_lowest_first() {
        let mut q = FreqQueue::new(2);
        for (w, s) in [("a", 1.0), ("b", 3.0)] {
            q.add(ScoreTerm {
                word: w.into(),
                top_field: "f".into(),
                score: s,
            });
        }
        let top = q.top_mut().unwrap();
        assert_eq!(top.word, "a");
        *top = ScoreTerm {
            word: "c".into(),
            top_field: "f".into(),
            score: 2.0,
        };
        q.update_top();
        assert_eq!(MoreLikeThis::interesting(q, 1), ["c"]);
    }
    fn reader() -> DirectoryReader {
        let dir = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/mlt/index"
        ));
        DirectoryReader::open(&lucene_store::FsDirectory::open(dir)).unwrap()
    }

    /// The reader's and a searcher's term statistics agree; a field list
    /// left `None` is every indexed field; settings Java refuses are refused.
    #[test]
    fn statistics_settings_and_refusals() {
        let reader = reader();
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let norms = vec![None; segments.len()];
        let searcher = IndexSearcher::new(&segments, &norms).unwrap();
        for (field, term) in [
            ("body", "river"),
            ("tv", "stone"),
            ("body", "nosuch"),
            ("x", "y"),
        ] {
            assert_eq!(
                TermStatsReader::doc_freq(&reader, field, term.as_bytes()).unwrap(),
                TermStatsReader::doc_freq(&searcher, field, term.as_bytes()).unwrap(),
                "{field}:{term}"
            );
            assert_eq!(
                TermStatsReader::doc_count(&reader, field).unwrap(),
                TermStatsReader::doc_count(&searcher, field).unwrap()
            );
        }
        assert_eq!(
            TermStatsReader::max_doc(&reader),
            TermStatsReader::max_doc(&searcher)
        );

        let mut m = MoreLikeThis::new(&reader);
        m.field_names = None;
        m.min_term_freq = 1;
        m.min_doc_freq = 1;
        let analyzer = Analyzer::standard(None);
        m.analyzer = Some(&analyzer);
        let q = m.like_doc(3).unwrap();
        assert!(!q.should.is_empty());
        let fields = m.field_names.clone().unwrap();
        assert!(fields.contains(&"body".to_string()) && fields.contains(&"id".to_string()));
        assert!(m.set_max_doc_freq_pct(i32::MAX).is_err());
        assert!(m.like_doc(10_000).is_err(), "a document outside the reader");
        m.max_query_terms = 0;
        assert!(m.retrieve_interesting_terms(3).unwrap().is_empty());

        // Statistics alone: no documents, no field list to resolve.
        let mut over = MoreLikeThis::over(&searcher);
        over.field_names = None;
        assert!(over.like_doc(0).is_err());
        assert!(over.like_fields(&[("body", vec!["river".into()])]).is_err());
        over.field_names = Some(vec!["body".into()]);
        assert!(over.like_texts("body", &["river"]).is_err(), "no analyzer");
        over.analyzer = Some(&analyzer);
        over.min_doc_freq = 1;
        over.min_term_freq = 1;
        over.min_word_len = 5;
        over.max_word_len = 5;
        over.stop_words = Some(["stone".to_string()].into_iter().collect());
        let words = over
            .retrieve_interesting_terms_of_text("river stone light house river garden", "body")
            .unwrap();
        // Five letters exactly, and not a stop word.
        assert!(!words.is_empty());
        assert!(
            words.iter().all(|w| w.len() == 5 && w != "stone"),
            "{words:?}"
        );
        over.max_num_tokens_parsed = 1;
        assert_eq!(
            over.retrieve_interesting_terms_of_text("river light", "body")
                .unwrap(),
            ["river"]
        );
    }

    /// `MoreLikeThisQuery`: `toString`, equality (the analyzer by identity),
    /// its rewrite's minimum and settings.
    #[test]
    fn the_query_rewrites_prints_and_compares() {
        let reader = reader();
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let norms = vec![None; segments.len()];
        let searcher = IndexSearcher::new(&segments, &norms).unwrap();
        let analyzer = std::sync::Arc::new(Analyzer::standard(None));
        let mut q = MoreLikeThisQuery::new(
            "river stone river light",
            vec!["body".into()],
            std::sync::Arc::clone(&analyzer),
            "body",
        );
        assert_eq!(q.to_string(), "like:river stone river light");
        assert_eq!(q, q.clone());
        let other = MoreLikeThisQuery::new(
            "river stone river light",
            vec!["body".into()],
            std::sync::Arc::new(Analyzer::standard(None)),
            "body",
        );
        assert_ne!(q, other, "another analyzer");
        assert!(format!("{q:?}").starts_with("MoreLikeThisQuery"));
        q.min_doc_freq = 1;
        q.percent_terms_to_match = 1.0;
        q.stop_words = Some(["light".to_string()].into_iter().collect());
        let Clause::Boolean(b) = q.rewrite(&searcher).unwrap() else {
            panic!("a boolean")
        };
        assert_eq!(b.minimum_should_match, b.should.len());
        assert!(b.should.len() >= 2);
        let clause = Clause::from(q);
        assert!(matches!(clause, Clause::Extended(_)));
    }
}
