//! `WeightedSpanTermExtractor`, `QueryTermExtractor`, `QueryScorer` and
//! `QueryTermScorer`: how the classic highlighter decides which tokens of the
//! text the query matched, and how much.
//!
//! `WeightedSpanTermExtractor` indexes the text's token stream into a
//! one-document [`MemoryIndex`] (every field of the query is read from the
//! stream's one field, as Lucene's `DelegatingLeafReader` does) and asks the
//! span queries -- phrases and multi-phrases are turned into span queries
//! first -- where they match; a term of such a query is then only scored at a
//! position inside one of those spans.
//!
//! Not ported: queries Lucene reads through a rewrite this crate has no
//! counterpart for (`TermInSetQuery`, range and automaton multi-term
//! queries, `FunctionScoreQuery`, `IndexOrDocValuesQuery`), payload span
//! queries inside a span query, and `setWrapIfNotCachingTokenFilter(false)`:
//! the stream is always cached, as Java does for a stream that is not a
//! `CachingTokenFilter` already (caching a cached stream again only copies).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::{Arc, RwLock};

use lucene_analysis::{
    AnalysisError, AttributeSource, CachingTokenFilter, TokenFilter, TokenStream,
};
use lucene_codecs::fuzzy::FuzzyMatch;
use lucene_codecs::regexp::RegexpPattern;
use lucene_codecs::wildcard::WildcardPattern;
use lucene_index::document::{FieldToken, FieldTokens};

use super::{DynStream, Error, Result, Scorer};
use crate::extended_query::{ExtendedQuery, MultiTermQuery, MultiTermSource};
use crate::memory::MemoryIndex;
use crate::query::{Clause, PhraseQuery};
use crate::query_visitor::{self, Occur, QueryVisitor, SubVisitor, Term, TermCollector};
use crate::spans::SpanNode;

/// The one field every query field is read from.
const SHADOW_FIELD: &str = "shadowed_field";

/// `WeightedTerm`: a term and its weight.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedTerm {
    weight: f32,
    term: String,
}

impl WeightedTerm {
    /// `new WeightedTerm(weight, term)`.
    pub fn new(weight: f32, term: impl Into<String>) -> Self {
        WeightedTerm {
            weight,
            term: term.into(),
        }
    }

    /// `getTerm()`.
    pub fn term(&self) -> &str {
        &self.term
    }

    /// `getWeight()`.
    pub fn weight(&self) -> f32 {
        self.weight
    }
}

/// `PositionSpan`: the first and last position of a span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositionSpan {
    start: i32,
    end: i32,
}

impl PositionSpan {
    /// `new PositionSpan(start, end)`.
    pub fn new(start: i32, end: i32) -> Self {
        PositionSpan { start, end }
    }

    /// The first position.
    pub fn start(&self) -> i32 {
        self.start
    }

    /// The last position.
    pub fn end(&self) -> i32 {
        self.end
    }
}

/// `WeightedSpanTerm`: a weighted term that, for a phrase or span query, only
/// counts inside the spans the query matched.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedSpanTerm {
    weight: f32,
    term: String,
    position_sensitive: bool,
    position_spans: Vec<PositionSpan>,
}

impl WeightedSpanTerm {
    /// `new WeightedSpanTerm(weight, term)`.
    pub fn new(weight: f32, term: impl Into<String>) -> Self {
        WeightedSpanTerm {
            weight,
            term: term.into(),
            position_sensitive: false,
            position_spans: Vec::new(),
        }
    }

    /// `new WeightedSpanTerm(weight, term, positionSensitive)`.
    pub fn with_position_sensitive(
        weight: f32,
        term: impl Into<String>,
        position_sensitive: bool,
    ) -> Self {
        WeightedSpanTerm {
            position_sensitive,
            ..Self::new(weight, term)
        }
    }

    /// `getTerm()`.
    pub fn term(&self) -> &str {
        &self.term
    }

    /// `getWeight()`.
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// `isPositionSensitive()`.
    pub fn is_position_sensitive(&self) -> bool {
        self.position_sensitive
    }

    /// `getPositionSpans()`.
    pub fn position_spans(&self) -> &[PositionSpan] {
        &self.position_spans
    }

    /// `checkPosition(position)`.
    pub fn check_position(&self, position: i32) -> bool {
        self.position_spans
            .iter()
            .any(|s| position >= s.start && position <= s.end)
    }

    /// `addPositionSpans(positionSpans)`.
    pub fn add_position_spans(&mut self, spans: &[PositionSpan]) {
        self.position_spans.extend_from_slice(spans);
    }
}

/// What `IndexReader.maxDoc()` and `docFreq(term)` give the idf-weighted
/// scorers.
pub trait DocFreqSource {
    /// `maxDoc()`.
    fn max_doc(&self) -> i32;
    /// `docFreq(new Term(field, term))`.
    fn doc_freq(&self, field: &str, term: &str) -> i32;
}

/// The idf of `QueryTermExtractor` and `getWeightedSpanTermsWithScores`:
/// `(float) (Math.log(maxDoc / (double) (docFreq + 1)) + 1.0)`.
fn idf(max_doc: i32, doc_freq: i32) -> f32 {
    (lucene_util::strict_math::log(f64::from(max_doc) / f64::from(doc_freq.wrapping_add(1))) + 1.0)
        as f32
}

fn text_of(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

// ---------------------------------------------------------------------------
// QueryTermExtractor
// ---------------------------------------------------------------------------

/// `QueryTermExtractor`: the terms of a query, weighted by their boosts.
pub struct QueryTermExtractor;

struct BoostedTermExtractor {
    boost: f32,
    terms: Rc<RefCell<Vec<WeightedTerm>>>,
    include_prohibited: bool,
    field: Option<String>,
}

impl QueryVisitor for BoostedTermExtractor {
    fn accept_field(&self, field: &str) -> bool {
        self.field.as_deref().is_none_or(|f| f == field)
    }

    fn consume_terms(&mut self, _query: &Clause, terms: &[Term]) {
        let mut out = self.terms.borrow_mut();
        for t in terms {
            out.push(WeightedTerm::new(self.boost, text_of(&t.bytes)));
        }
    }

    fn sub_visitor(&mut self, occur: Occur, parent: &Clause) -> SubVisitor {
        if let Clause::Boost(b) = parent {
            return SubVisitor::Other(Box::new(BoostedTermExtractor {
                boost: self.boost * b.boost,
                terms: Rc::clone(&self.terms),
                include_prohibited: self.include_prohibited,
                field: self.field.clone(),
            }));
        }
        if occur == Occur::MustNot && !self.include_prohibited {
            return SubVisitor::Empty;
        }
        SubVisitor::Same
    }
}

impl QueryTermExtractor {
    /// `getTerms(query)`.
    pub fn get_terms(query: &Clause) -> Vec<WeightedTerm> {
        Self::get_terms_in(query, false, None)
    }

    /// `getTerms(query, prohibited, fieldName)`: the terms of `field` (every
    /// field for `None`), those of `MUST_NOT` clauses too when `prohibited`.
    pub fn get_terms_in(
        query: &Clause,
        prohibited: bool,
        field: Option<&str>,
    ) -> Vec<WeightedTerm> {
        let terms = Rc::new(RefCell::new(Vec::new()));
        let mut v = BoostedTermExtractor {
            boost: 1.0,
            terms: Rc::clone(&terms),
            include_prohibited: prohibited,
            field: field.map(str::to_string),
        };
        query_visitor::visit(query, &mut v);
        let out = terms.borrow().clone();
        out
    }

    /// `getIdfWeightedTerms(query, reader, fieldName)`.
    pub fn get_idf_weighted_terms(
        query: &Clause,
        reader: &dyn DocFreqSource,
        field: &str,
    ) -> Vec<WeightedTerm> {
        let mut terms = Self::get_terms_in(query, false, Some(field));
        let max_doc = reader.max_doc();
        for t in &mut terms {
            t.weight *= idf(max_doc, reader.doc_freq(field, &t.term));
        }
        terms
    }
}

// ---------------------------------------------------------------------------
// OffsetLimitTokenFilter
// ---------------------------------------------------------------------------

/// `OffsetLimitTokenFilter`: stops once the tokens passed cover
/// `offset_limit` characters.
struct OffsetLimitTokenFilter<I> {
    input: I,
    offset_count: i32,
    offset_limit: i32,
}

impl<I: TokenStream> TokenFilter for OffsetLimitTokenFilter<I> {
    type Input = I;
    fn input(&self) -> &I {
        &self.input
    }
    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }
    fn increment(&mut self) -> std::result::Result<bool, AnalysisError> {
        if self.offset_count < self.offset_limit && self.input.increment_token()? {
            let a = self.input.attributes();
            self.offset_count += a.end_offset() - a.start_offset();
            return Ok(true);
        }
        Ok(false)
    }
    fn reset_filter(&mut self) -> std::result::Result<(), AnalysisError> {
        self.input.reset()?;
        self.offset_count = 0;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// WeightedSpanTermExtractor
// ---------------------------------------------------------------------------

/// The result of an extraction: the weighted terms, and the stream the
/// highlighter reads next.
pub struct Extraction<'a> {
    /// The weighted span terms by term text.
    pub terms: HashMap<String, WeightedSpanTerm>,
    /// The token stream to read the text from: the extractor's replayable
    /// copy once it has consumed the stream, else the stream it was given.
    pub stream: DynStream<'a>,
    /// Whether `stream` is the extractor's cached copy
    /// (`isCachedTokenStream()`).
    pub cached: bool,
}

/// `WeightedSpanTermExtractor`.
#[derive(Debug, Clone)]
pub struct WeightedSpanTermExtractor {
    default_field: Option<String>,
    expand_multi_term_query: bool,
    max_doc_chars_to_analyze: i32,
    use_payloads: bool,
}

impl Default for WeightedSpanTermExtractor {
    fn default() -> Self {
        Self::new(None)
    }
}

/// The document being extracted from: its stream, and once a query needs it,
/// the one-document index and its terms.
struct Leaf<'a> {
    stream: Option<DynStream<'a>>,
    replay: Option<DynStream<'a>>,
    index: Option<Arc<crate::memory::MemoryIndexReader>>,
    terms: Vec<Vec<u8>>,
}

impl WeightedSpanTermExtractor {
    /// `new WeightedSpanTermExtractor(defaultField)`.
    pub fn new(default_field: Option<&str>) -> Self {
        WeightedSpanTermExtractor {
            default_field: default_field.map(str::to_string),
            // Java's field default is `false`; `QueryScorer` sets it
            expand_multi_term_query: false,
            max_doc_chars_to_analyze: 0,
            use_payloads: false,
        }
    }

    /// `setExpandMultiTermQuery(expand)`.
    pub fn set_expand_multi_term_query(&mut self, expand: bool) {
        self.expand_multi_term_query = expand;
    }

    /// `getExpandMultiTermQuery()`.
    pub fn expand_multi_term_query(&self) -> bool {
        self.expand_multi_term_query
    }

    /// `setUsePayloads(usePayloads)`.
    pub fn set_use_payloads(&mut self, use_payloads: bool) {
        self.use_payloads = use_payloads;
    }

    /// `setMaxDocCharsToAnalyze(max)`.
    pub fn set_max_doc_chars_to_analyze(&mut self, max: i32) {
        self.max_doc_chars_to_analyze = max;
    }

    /// `getWeightedSpanTerms(query, boost, tokenStream, fieldName)`.
    pub fn get_weighted_span_terms<'a>(
        &self,
        query: &Clause,
        boost: f32,
        token_stream: DynStream<'a>,
        field_name: Option<&str>,
    ) -> Result<Extraction<'a>> {
        let mut run = Run {
            cfg: self,
            field_name: field_name.map(str::to_string),
            leaf: Leaf {
                stream: Some(token_stream),
                replay: None,
                index: None,
                terms: Vec::new(),
            },
            out: HashMap::new(),
        };
        run.extract(query, boost)?;
        let cached = run.leaf.replay.is_some();
        // the stream is moved into the replay copy when the document is
        // indexed, and is otherwise still here
        let stream = run
            .leaf
            .replay
            .take()
            .or_else(|| run.leaf.stream.take())
            .ok_or_else(|| Error::IllegalArgument("the token stream is gone".into()))?;
        Ok(Extraction {
            terms: run.out,
            stream,
            cached,
        })
    }

    /// `getWeightedSpanTermsWithScores(query, boost, tokenStream, fieldName,
    /// reader)`: the terms' weights times their idf.
    pub fn get_weighted_span_terms_with_scores<'a>(
        &self,
        query: &Clause,
        boost: f32,
        token_stream: DynStream<'a>,
        field_name: &str,
        reader: &dyn DocFreqSource,
    ) -> Result<Extraction<'a>> {
        let mut e = self.get_weighted_span_terms(query, boost, token_stream, Some(field_name))?;
        let max_doc = reader.max_doc();
        for w in e.terms.values_mut() {
            w.weight *= idf(max_doc, reader.doc_freq(field_name, &w.term));
        }
        Ok(e)
    }
}

struct Run<'e, 'a> {
    cfg: &'e WeightedSpanTermExtractor,
    field_name: Option<String>,
    leaf: Leaf<'a>,
    out: HashMap<String, WeightedSpanTerm>,
}

impl Run<'_, '_> {
    /// `fieldNameComparator(fieldNameToCheck)`.
    fn field_ok(&self, field: &str) -> bool {
        self.field_name.as_deref().is_none_or(|f| f == field)
            || self.cfg.default_field.as_deref() == Some(field)
    }

    /// `PositionCheckingMap.put`.
    fn put(&mut self, key: String, mut new: WeightedSpanTerm) {
        if let Some(prev) = self.out.get(&key) {
            if !prev.position_sensitive {
                new.position_sensitive = false;
            }
        }
        self.out.insert(key, new);
    }

    /// `getLeafContext()`: the stream indexed, on first need.
    fn ensure_leaf(&mut self) -> Result<()> {
        if self.leaf.index.is_some() {
            return Ok(());
        }
        let stream = self
            .leaf
            .stream
            .take()
            .ok_or_else(|| Error::IllegalArgument("the token stream is gone".into()))?;
        let limited = OffsetLimitTokenFilter {
            input: stream,
            offset_count: 0,
            offset_limit: self.cfg.max_doc_chars_to_analyze,
        };
        let mut cache = DynStream::new(CachingTokenFilter::new(limited));
        // `MemoryIndex.addField(name, stream)`: reset, read, end, close
        cache.reset()?;
        let mut tokens = Vec::new();
        while cache.increment_token()? {
            let a = cache.attributes();
            tokens.push(FieldToken {
                term: a.term_bytes().to_vec(),
                start_offset: a.start_offset(),
                end_offset: a.end_offset(),
                position_increment: a.position_increment(),
                term_frequency: 1,
                payload: if self.cfg.use_payloads {
                    a.payload().filter(|p| !p.is_empty()).map(<[u8]>::to_vec)
                } else {
                    None
                },
            });
        }
        cache.end()?;
        let field_tokens = FieldTokens {
            tokens,
            final_position_increment: cache.attributes().position_increment(),
            final_offset: cache.attributes().end_offset(),
            end_attributes: None,
        };
        cache.close()?;
        let mut terms: Vec<Vec<u8>> = field_tokens.tokens.iter().map(|t| t.term.clone()).collect();
        terms.sort_unstable();
        terms.dedup();
        let mut indexer = MemoryIndex::with_options(true, self.cfg.use_payloads);
        indexer.add_field_tokens(SHADOW_FIELD, &field_tokens, 0, 1)?;
        self.leaf.index = Some(indexer.create_reader()?);
        self.leaf.terms = terms;
        self.leaf.replay = Some(cache);
        Ok(())
    }

    /// `extract(query, boost, terms)`.
    fn extract(&mut self, query: &Clause, boost: f32) -> Result<()> {
        match query {
            Clause::Boost(b) => self.extract(&b.inner, boost * b.boost),
            Clause::Boolean(b) => {
                for c in b.must.iter().chain(&b.filter).chain(&b.should) {
                    self.extract(c, boost)?;
                }
                Ok(())
            }
            Clause::Phrase(p) => self.extract_phrase(p, boost),
            Clause::Term(_) => self.extract_weighted_terms(query, boost),
            Clause::Span(s) => self.extract_weighted_span_terms(&SpanNode::from(s), boost),
            Clause::ConstantScore(c) => self.extract(&c.inner, boost),
            Clause::DisjunctionMax(d) => {
                for c in &d.disjuncts {
                    self.extract(c, boost)?;
                }
                Ok(())
            }
            Clause::MultiPhrase(m) => self.extract_multi_phrase(m, boost),
            Clause::MatchAllDocs(_) => Ok(()),
            Clause::Extended(e) => match e.as_ref() {
                ExtendedQuery::Synonym(_) => self.extract_weighted_terms(query, boost),
                ExtendedQuery::Span(node) => self.extract_weighted_span_terms(node, boost),
                _ => self.extract_other(query, boost),
            },
            _ => self.extract_other(query, boost),
        }
    }

    /// The final branch of `extract`: a multi-term query read through the
    /// terms of the document, or a query nothing is known about.
    fn extract_other(&mut self, query: &Clause, boost: f32) -> Result<()> {
        let field = match query {
            Clause::Prefix(q) => Some(q.field.as_str()),
            Clause::Wildcard(q) => Some(q.field.as_str()),
            Clause::Regexp(q) => Some(q.field.as_str()),
            Clause::Fuzzy(q) => Some(q.field.as_str()),
            _ => None,
        };
        if let Some(f) = field {
            if !self.cfg.expand_multi_term_query || !self.field_ok(f) {
                return Ok(());
            }
        }
        self.ensure_leaf()?;
        let Some(field) = field else {
            // `extractUnknownQuery`: nothing
            return Ok(());
        };
        // `SCORING_BOOLEAN_REWRITE`: a boosted `TermQuery` per matching term
        let expanded = expand_clause(query, &self.leaf.terms)?;
        for (term, term_boost) in expanded.unwrap_or_default() {
            let text = text_of(&term);
            if self.field_ok(field) {
                self.put(
                    text.clone(),
                    WeightedSpanTerm::new(boost * term_boost, text),
                );
            }
        }
        Ok(())
    }

    /// `extractWeightedTerms(terms, query, boost)`.
    fn extract_weighted_terms(&mut self, query: &Clause, boost: f32) -> Result<()> {
        self.ensure_leaf()?;
        let mut collector = TermCollector::default();
        query_visitor::visit(query, &mut collector);
        for t in collector.terms {
            if self.field_ok(&t.field) {
                let text = text_of(&t.bytes);
                self.put(text.clone(), WeightedSpanTerm::new(boost, text));
            }
        }
        Ok(())
    }

    fn extract_phrase(&mut self, p: &PhraseQuery, boost: f32) -> Result<()> {
        if p.terms.len() == 1 {
            let node = SpanNode::term(p.field.clone(), p.terms[0].clone());
            return self.extract_weighted_span_terms(&node, boost);
        }
        let clauses: Vec<SpanNode> = p
            .terms
            .iter()
            .map(|t| SpanNode::term(p.field.clone(), t.clone()))
            .collect();
        let positions = p.positions();
        let mut position_gaps = 0i32;
        if positions.len() >= 2 {
            let n = i32::try_from(positions.len()).unwrap_or(i32::MAX);
            position_gaps = (positions[positions.len() - 1] - positions[0] - n + 1).max(0);
        }
        let slop = i32::try_from(p.slop).unwrap_or(i32::MAX);
        let node = SpanNode::near(clauses, slop.saturating_add(position_gaps), slop == 0)
            .map_err(Error::Search)?;
        self.extract_weighted_span_terms(&node, boost)
    }

    fn extract_multi_phrase(
        &mut self,
        m: &crate::query::MultiPhraseQuery,
        boost: f32,
    ) -> Result<()> {
        let positions = m.positions();
        if positions.is_empty() {
            return Ok(());
        }
        let max_position = positions.iter().copied().max().unwrap_or(0);
        let len = usize::try_from(max_position).unwrap_or(0) + 1;
        let mut disjunct_lists: Vec<Option<Vec<SpanNode>>> = vec![None; len];
        for (i, array) in m.term_arrays.iter().enumerate() {
            let at = usize::try_from(positions[i]).unwrap_or(0);
            let list = disjunct_lists[at].get_or_insert_with(Vec::new);
            for t in array {
                list.push(SpanNode::term(m.field.clone(), t.clone()));
            }
        }
        let mut position_gaps = 0i32;
        let mut clauses = Vec::new();
        for list in disjunct_lists {
            match list {
                Some(l) => clauses.push(SpanNode::or(l).map_err(Error::Search)?),
                None => position_gaps += 1,
            }
        }
        if clauses.len() == 1 {
            return self.extract_weighted_span_terms(&clauses[0], boost);
        }
        let slop = i32::try_from(m.slop).unwrap_or(i32::MAX);
        let node = SpanNode::near(clauses, slop.saturating_add(position_gaps), slop == 0)
            .map_err(Error::Search)?;
        self.extract_weighted_span_terms(&node, boost)
    }

    /// `extractWeightedSpanTerms(terms, spanQuery, boost)`.
    fn extract_weighted_span_terms(&mut self, span_query: &SpanNode, boost: f32) -> Result<()> {
        let mut query_field_names = HashSet::new();
        collect_span_query_fields(span_query, &mut query_field_names);
        if let Some(f) = &self.field_name {
            if !query_field_names.contains(f)
                && self
                    .cfg
                    .default_field
                    .as_ref()
                    .is_none_or(|d| !query_field_names.contains(d))
            {
                return Ok(());
            }
        }
        self.ensure_leaf()?;
        let node = if must_rewrite(span_query, self.cfg.expand_multi_term_query) {
            match expand_node(span_query, &self.leaf.terms)? {
                Some(n) => n,
                None => return Ok(()),
            }
        } else {
            span_query.clone()
        };
        // `query.visit(QueryVisitor.termCollector(terms))`: a span query's terms
        // (a not query's include side only), each once
        let mut pairs = Vec::new();
        crate::spans::weight_terms(&node, &mut pairs);
        let non_weighted_terms: std::collections::BTreeSet<Term> =
            pairs.into_iter().map(|(f, b)| Term::new(f, b)).collect();
        if non_weighted_terms.is_empty() {
            return Ok(());
        }
        let Some(shadowed) = shadow_node(&node)? else {
            return Ok(());
        };
        let leaf = self
            .leaf
            .index
            .as_ref()
            .ok_or_else(|| Error::IllegalArgument("the document is not indexed".to_string()))?;
        let span_positions = span_positions(leaf, &Clause::from(shadowed))?;
        if span_positions.is_empty() {
            return Ok(());
        }
        for query_term in non_weighted_terms {
            if self.field_ok(&query_term.field) {
                let text = text_of(&query_term.bytes);
                match self.out.get_mut(&text) {
                    None => {
                        let mut w = WeightedSpanTerm::new(boost, text.clone());
                        w.add_position_spans(&span_positions);
                        w.position_sensitive = true;
                        self.put(text, w);
                    }
                    Some(w) => w.add_position_spans(&span_positions),
                }
            }
        }
        Ok(())
    }
}

/// Where `query` matches the document's one field: `Spans`'
/// `startPosition()` and `endPosition() - 1`, as `SpanWeight.matches` reports
/// them.
fn span_positions(
    reader: &crate::memory::MemoryIndexReader,
    query: &Clause,
) -> Result<Vec<PositionSpan>> {
    let found = reader.with_searcher(|searcher| -> crate::Result<Vec<PositionSpan>> {
        let mut out = Vec::new();
        let Some(m) = crate::matches::matches(searcher, query, 0)? else {
            return Ok(out);
        };
        if let Some(mut it) = m.get_matches(SHADOW_FIELD)? {
            while it.next()? {
                out.push(PositionSpan::new(it.start_position(), it.end_position()));
            }
        }
        Ok(out)
    })?;
    Ok(found?)
}

/// `collectSpanQueryFields`.
fn collect_span_query_fields(node: &SpanNode, fields: &mut HashSet<String>) {
    match node {
        SpanNode::FieldMasking { inner, .. } | SpanNode::First { inner, .. } => {
            collect_span_query_fields(inner, fields);
        }
        SpanNode::Near { clauses, .. } | SpanNode::Or { clauses } => {
            for c in clauses {
                collect_span_query_fields(c, fields);
            }
        }
        SpanNode::Not { include, .. } => collect_span_query_fields(include, fields),
        other => {
            if let Some(f) = other.field() {
                fields.insert(f.to_string());
            }
        }
    }
}

/// `mustRewriteQuery(spanQuery)`.
fn must_rewrite(node: &SpanNode, expand: bool) -> bool {
    if !expand {
        return false;
    }
    match node {
        SpanNode::FieldMasking { inner, .. } | SpanNode::First { inner, .. } => {
            must_rewrite(inner, expand)
        }
        SpanNode::Near { clauses, .. } | SpanNode::Or { clauses } => {
            clauses.iter().any(|c| must_rewrite(c, expand))
        }
        SpanNode::Not {
            include, exclude, ..
        } => must_rewrite(include, expand) || must_rewrite(exclude, expand),
        SpanNode::Term { .. } => false,
        _ => true,
    }
}

/// A multi-term clause's matching terms, each with its boost.
type Expansion = Vec<(Vec<u8>, f32)>;

/// The terms (and their boosts) a multi-term clause matches among a
/// document's `terms`, as `SCORING_BOOLEAN_REWRITE` finds them.
pub(crate) fn expand_clause(query: &Clause, terms: &[Vec<u8>]) -> Result<Option<Expansion>> {
    let plain = |keep: &dyn Fn(&[u8]) -> bool| -> Expansion {
        terms
            .iter()
            .filter(|t| keep(t))
            .map(|t| (t.clone(), 1.0))
            .collect()
    };
    Ok(match query {
        Clause::Prefix(q) => {
            let p = WildcardPattern::prefix(&q.prefix);
            Some(plain(&|t| p.matches(t)))
        }
        Clause::Wildcard(q) => {
            let p = WildcardPattern::new(&q.pattern);
            Some(plain(&|t| p.matches(t)))
        }
        Clause::Regexp(q) => {
            let p = RegexpPattern::new(q.pattern.as_bytes())
                .map_err(|e| Error::IllegalArgument(e.to_string()))?;
            Some(plain(&|t| p.matches(t)))
        }
        Clause::Fuzzy(q) => {
            let p = FuzzyMatch::new(&q.term, q.max_edits, q.prefix_length, q.transpositions);
            Some(
                terms
                    .iter()
                    .filter_map(|t| p.boost(t).map(|b| (t.clone(), b)))
                    .collect(),
            )
        }
        _ => None,
    })
}

/// `searcher.rewrite(spanQuery)` for what the queries here contain: every
/// `SpanMultiTermQueryWrapper` becomes a `SpanOr` of the document's terms it
/// matches. `None` when the rewritten query can match nothing.
fn expand_node(node: &SpanNode, terms: &[Vec<u8>]) -> Result<Option<SpanNode>> {
    let one = |n: &SpanNode| expand_node(n, terms);
    Ok(match node {
        SpanNode::Term { .. } => Some(node.clone()),
        SpanNode::Near {
            clauses,
            slop,
            in_order,
        } => {
            let mut out = Vec::with_capacity(clauses.len());
            for c in clauses {
                match one(c)? {
                    Some(n) => out.push(n),
                    None => return Ok(None),
                }
            }
            Some(SpanNode::near(out, *slop, *in_order).map_err(Error::Search)?)
        }
        SpanNode::Or { clauses } => {
            let mut out = Vec::new();
            for c in clauses {
                if let Some(n) = one(c)? {
                    out.push(n);
                }
            }
            if out.is_empty() {
                None
            } else {
                Some(SpanNode::or(out).map_err(Error::Search)?)
            }
        }
        SpanNode::First { inner, end } => one(inner)?.map(|n| SpanNode::first(n, *end)),
        SpanNode::PositionRange { inner, start, end } => {
            one(inner)?.map(|n| SpanNode::position_range(n, *start, *end))
        }
        SpanNode::Not {
            include,
            exclude,
            pre,
            post,
        } => match (one(include)?, one(exclude)?) {
            (None, _) => None,
            (Some(i), None) => Some(i),
            (Some(i), Some(e)) => Some(SpanNode::not(i, e, *pre, *post).map_err(Error::Search)?),
        },
        SpanNode::Containing { big, little } => match (one(big)?, one(little)?) {
            (Some(b), Some(l)) => Some(SpanNode::containing(b, l).map_err(Error::Search)?),
            _ => None,
        },
        SpanNode::Within { big, little } => match (one(big)?, one(little)?) {
            (Some(b), Some(l)) => Some(SpanNode::within(b, l).map_err(Error::Search)?),
            _ => None,
        },
        SpanNode::FieldMasking { inner, field } => {
            one(inner)?.map(|n| SpanNode::field_masking(n, field.clone()))
        }
        SpanNode::MultiTerm(q) => {
            let clause = multi_term_clause(q);
            let expanded = match clause {
                Some(c) => expand_clause(&c, terms)?.unwrap_or_default(),
                None => Vec::new(),
            };
            let field = q.field().to_string();
            let clauses: Vec<SpanNode> = expanded
                .into_iter()
                .map(|(t, _)| SpanNode::term(field.clone(), t))
                .collect();
            if clauses.is_empty() {
                None
            } else {
                Some(SpanNode::or(clauses).map_err(Error::Search)?)
            }
        }
        // payload queries are kept as they are
        other => Some(other.clone()),
    })
}

/// The plain clause behind a multi-term span wrapper, where there is one.
fn multi_term_clause(q: &MultiTermQuery) -> Option<Clause> {
    match &q.source {
        MultiTermSource::Prefix(p) => Some(Clause::Prefix(p.clone())),
        MultiTermSource::Wildcard(w) => Some(Clause::Wildcard(w.clone())),
        MultiTermSource::Regexp(r) => Some(Clause::Regexp(r.clone())),
        _ => None,
    }
}

/// The span query with every field replaced by the one the document is
/// indexed under (`DelegatingLeafReader`).
fn shadow_node(node: &SpanNode) -> Result<Option<SpanNode>> {
    let one = |n: &SpanNode| shadow_node(n);
    Ok(match node {
        SpanNode::Term { term, .. } => Some(SpanNode::term(SHADOW_FIELD, term.clone())),
        SpanNode::Near {
            clauses,
            slop,
            in_order,
        } => {
            let mut out = Vec::with_capacity(clauses.len());
            for c in clauses {
                match one(c)? {
                    Some(n) => out.push(n),
                    None => return Ok(None),
                }
            }
            Some(SpanNode::near(out, *slop, *in_order).map_err(Error::Search)?)
        }
        SpanNode::Or { clauses } => {
            let mut out = Vec::new();
            for c in clauses {
                if let Some(n) = one(c)? {
                    out.push(n);
                }
            }
            if out.is_empty() {
                None
            } else {
                Some(SpanNode::or(out).map_err(Error::Search)?)
            }
        }
        SpanNode::First { inner, end } => one(inner)?.map(|n| SpanNode::first(n, *end)),
        SpanNode::PositionRange { inner, start, end } => {
            one(inner)?.map(|n| SpanNode::position_range(n, *start, *end))
        }
        SpanNode::Not {
            include,
            exclude,
            pre,
            post,
        } => match (one(include)?, one(exclude)?) {
            (Some(i), Some(e)) => Some(SpanNode::not(i, e, *pre, *post).map_err(Error::Search)?),
            (Some(i), None) => Some(i),
            _ => None,
        },
        SpanNode::Containing { big, little } => match (one(big)?, one(little)?) {
            (Some(b), Some(l)) => Some(SpanNode::containing(b, l).map_err(Error::Search)?),
            _ => None,
        },
        SpanNode::Within { big, little } => match (one(big)?, one(little)?) {
            (Some(b), Some(l)) => Some(SpanNode::within(b, l).map_err(Error::Search)?),
            _ => None,
        },
        SpanNode::FieldMasking { inner, .. } => one(inner)?,
        // an expanded query has no multi-term wrapper left
        SpanNode::MultiTerm(_) => None,
        other => Some(other.clone()),
    })
}

// ---------------------------------------------------------------------------
// QueryScorer
// ---------------------------------------------------------------------------

type SharedTerms = Arc<RwLock<HashMap<String, WeightedSpanTerm>>>;

/// `QueryScorer`: scores a token by the weight of the query term it is, where
/// the query's phrase or span leaves it in a matching position.
pub struct QueryScorer {
    total_score: f32,
    found_terms: HashSet<String>,
    terms: SharedTerms,
    max_term_weight: f32,
    position: i32,
    default_field: Option<String>,
    expand_multi_term_query: bool,
    query: Option<Clause>,
    field: Option<String>,
    reader: Option<Arc<dyn DocFreqSource>>,
    skip_init_extractor: bool,
    max_chars_to_analyze: i32,
    use_payloads: bool,
}

impl QueryScorer {
    fn base(query: Option<Clause>) -> Self {
        QueryScorer {
            total_score: 0.0,
            found_terms: HashSet::new(),
            terms: Arc::new(RwLock::new(HashMap::new())),
            max_term_weight: 0.0,
            position: -1,
            default_field: None,
            expand_multi_term_query: true,
            query,
            field: None,
            reader: None,
            skip_init_extractor: false,
            max_chars_to_analyze: 0,
            use_payloads: false,
        }
    }

    /// `new QueryScorer(query)`.
    pub fn new(query: &Clause) -> Self {
        Self::base(Some(query.clone()))
    }

    /// `new QueryScorer(query, field)`: only the terms of `field`.
    pub fn for_field(query: &Clause, field: &str) -> Self {
        let mut s = Self::new(query);
        s.field = Some(field.to_string());
        s
    }

    /// `new QueryScorer(query, field, defaultField)`: the terms of `field` and
    /// of `defaultField`.
    pub fn for_field_with_default(query: &Clause, field: &str, default_field: &str) -> Self {
        let mut s = Self::for_field(query, field);
        s.default_field = Some(default_field.to_string());
        s
    }

    /// `new QueryScorer(query, reader, field)`: weights times idf.
    pub fn with_reader(query: &Clause, reader: Arc<dyn DocFreqSource>, field: &str) -> Self {
        let mut s = Self::for_field(query, field);
        s.reader = Some(reader);
        s
    }

    /// `new QueryScorer(WeightedSpanTerm[])`: no query to extract from.
    pub fn from_weighted_span_terms(weighted_terms: Vec<WeightedSpanTerm>) -> Self {
        let mut s = Self::base(None);
        {
            let mut map = s.terms.write().unwrap_or_else(|e| e.into_inner());
            for w in weighted_terms {
                let replace = map.get(&w.term).is_none_or(|e| e.weight < w.weight);
                if replace {
                    s.max_term_weight = s.max_term_weight.max(w.weight);
                    map.insert(w.term.clone(), w);
                }
            }
        }
        s.skip_init_extractor = true;
        s
    }

    /// The weighted span terms, shared with a [`super::SimpleSpanFragmenter`].
    pub(super) fn shared_terms(&self) -> SharedTerms {
        Arc::clone(&self.terms)
    }

    /// `getMaxTermWeight()`.
    pub fn max_term_weight(&self) -> f32 {
        self.max_term_weight
    }

    /// `getWeightedSpanTerm(token)`.
    pub fn weighted_span_term(&self, token: &str) -> Option<WeightedSpanTerm> {
        self.terms
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(token)
            .cloned()
    }

    /// `isExpandMultiTermQuery()`.
    pub fn is_expand_multi_term_query(&self) -> bool {
        self.expand_multi_term_query
    }

    /// `setExpandMultiTermQuery(expand)`.
    pub fn set_expand_multi_term_query(&mut self, expand: bool) {
        self.expand_multi_term_query = expand;
    }

    /// `isUsePayloads()`.
    pub fn is_use_payloads(&self) -> bool {
        self.use_payloads
    }

    /// `setUsePayloads(usePayloads)`.
    pub fn set_use_payloads(&mut self, use_payloads: bool) {
        self.use_payloads = use_payloads;
    }
}

impl Scorer for QueryScorer {
    fn init<'a>(&mut self, stream: DynStream<'a>) -> Result<DynStream<'a>> {
        self.position = -1;
        if self.skip_init_extractor {
            return Ok(stream);
        }
        self.terms
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        let Some(query) = self.query.clone() else {
            return Ok(stream);
        };
        let mut qse = WeightedSpanTermExtractor::new(self.default_field.as_deref());
        qse.set_max_doc_chars_to_analyze(self.max_chars_to_analyze);
        qse.set_expand_multi_term_query(self.expand_multi_term_query);
        qse.set_use_payloads(self.use_payloads);
        let extraction = match &self.reader {
            None => qse.get_weighted_span_terms(&query, 1.0, stream, self.field.as_deref())?,
            Some(r) => qse.get_weighted_span_terms_with_scores(
                &query,
                1.0,
                stream,
                self.field.as_deref().unwrap_or(""),
                r.as_ref(),
            )?,
        };
        *self.terms.write().unwrap_or_else(|e| e.into_inner()) = extraction.terms;
        Ok(extraction.stream)
    }

    fn start_fragment(&mut self) {
        self.found_terms = HashSet::new();
        self.total_score = 0.0;
    }

    fn token_score(&mut self, token: &AttributeSource) -> f32 {
        self.position += token.position_increment();
        let term_text = token.term();
        let (weight, sensitive, ok) = {
            let map = self.terms.read().unwrap_or_else(|e| e.into_inner());
            let Some(w) = map.get(term_text) else {
                return 0.0;
            };
            (
                w.weight,
                w.position_sensitive,
                w.check_position(self.position),
            )
        };
        if sensitive && !ok {
            return 0.0;
        }
        if self.found_terms.insert(term_text.to_string()) {
            self.total_score += weight;
        }
        weight
    }

    fn fragment_score(&self) -> f32 {
        self.total_score
    }

    fn set_max_doc_chars_to_analyze(&mut self, max: i32) {
        self.max_chars_to_analyze = max;
    }
}

// ---------------------------------------------------------------------------
// QueryTermScorer
// ---------------------------------------------------------------------------

/// `QueryTermScorer`: scores a token by its term's weight, wherever it is.
pub struct QueryTermScorer {
    unique_terms_in_fragment: HashSet<String>,
    total_score: f32,
    max_term_weight: f32,
    terms_to_find: HashMap<String, WeightedTerm>,
}

impl QueryTermScorer {
    /// `new QueryTermScorer(query)`.
    pub fn new(query: &Clause) -> Self {
        Self::from_weighted_terms(QueryTermExtractor::get_terms(query))
    }

    /// `new QueryTermScorer(query, fieldName)`.
    pub fn for_field(query: &Clause, field: &str) -> Self {
        Self::from_weighted_terms(QueryTermExtractor::get_terms_in(query, false, Some(field)))
    }

    /// `new QueryTermScorer(query, reader, fieldName)`.
    pub fn with_reader(query: &Clause, reader: &dyn DocFreqSource, field: &str) -> Self {
        Self::from_weighted_terms(QueryTermExtractor::get_idf_weighted_terms(
            query, reader, field,
        ))
    }

    /// `new QueryTermScorer(WeightedTerm[])`.
    pub fn from_weighted_terms(weighted_terms: Vec<WeightedTerm>) -> Self {
        let mut terms_to_find: HashMap<String, WeightedTerm> = HashMap::new();
        let mut max_term_weight = 0.0f32;
        for w in weighted_terms {
            if terms_to_find
                .get(&w.term)
                .is_none_or(|e| e.weight < w.weight)
            {
                max_term_weight = max_term_weight.max(w.weight);
                terms_to_find.insert(w.term.clone(), w);
            }
        }
        QueryTermScorer {
            unique_terms_in_fragment: HashSet::new(),
            total_score: 0.0,
            max_term_weight,
            terms_to_find,
        }
    }

    /// `getMaxTermWeight()`.
    pub fn max_term_weight(&self) -> f32 {
        self.max_term_weight
    }
}

impl Scorer for QueryTermScorer {
    fn init<'a>(&mut self, stream: DynStream<'a>) -> Result<DynStream<'a>> {
        Ok(stream)
    }

    fn start_fragment(&mut self) {
        self.unique_terms_in_fragment = HashSet::new();
        self.total_score = 0.0;
    }

    fn token_score(&mut self, token: &AttributeSource) -> f32 {
        let term_text = token.term();
        let Some(query_term) = self.terms_to_find.get(term_text) else {
            return 0.0;
        };
        if self.unique_terms_in_fragment.insert(term_text.to_string()) {
            self.total_score += query_term.weight;
        }
        query_term.weight
    }

    fn fragment_score(&self) -> f32 {
        self.total_score
    }
}
