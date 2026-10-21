//! `org.apache.lucene.search.uhighlight`: the unified highlighter.
//!
//! A field's offsets come from postings, term vectors or re-analysis of the
//! stored text (see [`Strategy`]); the text is cut into passages by a
//! [`BreakIterator`], the passages scored ([`PassageScorer`]) and the best
//! formatted ([`PassageFormatter`]).
//!
//! Offsets are UTF-16 code units, as Java's `String` indexes, so the text is
//! held as UTF-16; a lone surrogate in a snippet reads as U+FFFD.
//!
//! Differences from Lucene:
//!
//! - `with_max_length` and the fragment-alignment constructor panic where
//!   Java throws `IllegalArgumentException`; see their `# Panics`.
//! - The one-document index of an `ANALYSIS` or `TERM_VECTORS` field is
//!   overlaid on the segment leaf by leaf of the query (see
//!   [`strategies`]): a leaf that reads both the highlighted field and
//!   another (a span query masking one field as another, a
//!   `CommonTermsQuery` over both) reads the other from neither.
//! - A term vector without positions is read with the positions its tokens
//!   imply (Java's `TermVectorLeafReader` then fails a phrase or span
//!   query); one without offsets reports them as -1, as Java's does.
//! - `TermVectorReusingLeafReader` (a cache of the current document's term
//!   vector when two or more fields read them) is not needed: a term
//!   vector is read once per field and document either way.
//! - A span query's term on a field the matcher rejects is left out of the
//!   `PhraseHelper`'s offsets by its bytes: where the same bytes are also a
//!   term of an accepted field, the rejected occurrences count too.
//! - `POSTINGS_WITH_TERM_VECTORS` filters the terms of the classic term and
//!   automaton paths by the document's term vector, but a query rewritten
//!   from the index (`WEIGHT_MATCHES`, a multi-term span) enumerates the
//!   segment's terms, not the term vector's: a top-terms or clause-limited
//!   rewrite may keep other terms than Java's.

pub mod break_iterators;
pub mod field_highlighter;
pub mod matches_enum;
pub mod passage;
pub mod phrase_helper;
pub mod strategies;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_codecs::stored_fields::FieldValue;

use self::break_iterators::{sentence_instance, BreakIterator, SplittingBreakIterator};
pub use self::field_highlighter::{passage_by_start_offset, FieldHighlighter, PassageOrder};
pub use self::passage::{
    DefaultPassageFormatter, EmptyOffsets, HighlightObject, HighlightValue, OffsetsEnum, Passage,
    PassageFormatter, PassageScorer,
};
pub use self::strategies::Flags;
use self::strategies::{
    multi_fields_enum, Components, FieldMatcher, LeafCtx, Strategy, MULTIVAL_SEP_CHAR,
};
use super::{Error, Result};
use crate::directory_reader::DirectoryReader;
use crate::highlighter::{offset_source_for_field, OffsetSource};
use crate::index_searcher::IndexSearcher;
use crate::query::Clause;
use crate::query_visitor::extract_terms;

/// `UnifiedHighlighter.DEFAULT_MAX_LENGTH`.
pub const DEFAULT_MAX_LENGTH: i32 = 10_000;

/// `UnifiedHighlighter.DEFAULT_CACHE_CHARS_THRESHOLD`: ~1 MB of UTF-16.
pub const DEFAULT_CACHE_CHARS_THRESHOLD: i32 = 524_288;

/// A factory of the base `BreakIterator` (`Supplier<BreakIterator>`).
pub type BreakIteratorFactory = Arc<dyn Fn() -> Box<dyn BreakIterator> + Send + Sync>;

/// `Function<String, Set<String>>`: the fields masked by a field.
pub type MaskedFields = Arc<dyn Fn(&str) -> Option<BTreeSet<String>> + Send + Sync>;

/// `UnifiedHighlighter`.
pub struct UnifiedHighlighter<'r> {
    reader: Option<&'r DirectoryReader>,
    analyzer: Option<Arc<Analyzer>>,
    field_matcher: Option<FieldMatcher>,
    masked_fields: Option<MaskedFields>,
    flags: Flags,
    max_length: i32,
    cache_field_val_chars_threshold: i32,
    break_iterator: BreakIteratorFactory,
    scorer: PassageScorer,
    formatter: Arc<dyn PassageFormatter + Send + Sync>,
    max_no_highlight_passages: i32,
    passage_sort_comparator: PassageOrder,
}

impl<'r> UnifiedHighlighter<'r> {
    /// `UnifiedHighlighter.builder(searcher, analyzer).build()`; `reader` is
    /// the searcher's index. Every default is Lucene's.
    pub fn new(reader: Option<&'r DirectoryReader>, analyzer: Option<Arc<Analyzer>>) -> Self {
        UnifiedHighlighter {
            reader,
            analyzer,
            field_matcher: None,
            masked_fields: None,
            flags: Flags::evaluate(true, true, true, true),
            max_length: DEFAULT_MAX_LENGTH,
            cache_field_val_chars_threshold: DEFAULT_CACHE_CHARS_THRESHOLD,
            break_iterator: Arc::new(sentence_instance),
            scorer: PassageScorer::default(),
            formatter: Arc::new(DefaultPassageFormatter::default()),
            max_no_highlight_passages: -1,
            passage_sort_comparator: passage_by_start_offset,
        }
    }

    /// The flags `Builder.withHandleMultiTermQuery`,
    /// `withHighlightPhrasesStrictly`, `withPassageRelevancyOverSpeed` and
    /// `withWeightMatches` evaluate to.
    pub fn with_flags(
        mut self,
        multi_term_query: bool,
        phrases: bool,
        passage_relevancy_over_speed: bool,
        weight_matches: bool,
    ) -> Self {
        self.flags = Flags::evaluate(
            multi_term_query,
            phrases,
            passage_relevancy_over_speed,
            weight_matches,
        );
        self
    }

    /// `withMaxLength(value)`.
    ///
    /// # Panics
    /// When `max_length` is outside `0..i32::MAX` (`IllegalArgumentException`).
    pub fn with_max_length(mut self, max_length: i32) -> Self {
        assert!(
            (0..i32::MAX).contains(&max_length),
            "maxLength must be < Integer.MAX_VALUE"
        );
        self.max_length = max_length;
        self
    }

    /// `withCacheFieldValCharsThreshold(value)`: how many characters of
    /// stored values a batch of documents loads before it is highlighted.
    pub fn with_cache_field_val_chars_threshold(mut self, value: i32) -> Self {
        self.cache_field_val_chars_threshold = value;
        self
    }

    /// `getCacheFieldValCharsThreshold()`.
    pub fn cache_field_val_chars_threshold(&self) -> i32 {
        self.cache_field_val_chars_threshold
    }

    /// `withBreakIterator(value)`.
    pub fn with_break_iterator(mut self, factory: BreakIteratorFactory) -> Self {
        self.break_iterator = factory;
        self
    }

    /// `withFieldMatcher(value)`.
    pub fn with_field_matcher(mut self, matcher: FieldMatcher) -> Self {
        self.field_matcher = Some(matcher);
        self
    }

    /// `withMaskedFieldsFunc(value)`.
    pub fn with_masked_fields(mut self, masked: MaskedFields) -> Self {
        self.masked_fields = Some(masked);
        self
    }

    /// `withScorer(value)`.
    pub fn with_scorer(mut self, scorer: PassageScorer) -> Self {
        self.scorer = scorer;
        self
    }

    /// `withFormatter(value)`.
    pub fn with_formatter(mut self, formatter: Arc<dyn PassageFormatter + Send + Sync>) -> Self {
        self.formatter = formatter;
        self
    }

    /// `withMaxNoHighlightPassages(value)`.
    pub fn with_max_no_highlight_passages(mut self, value: i32) -> Self {
        self.max_no_highlight_passages = value;
        self
    }

    /// `withPassageSortComparator(value)`.
    pub fn with_passage_sort_comparator(mut self, order: PassageOrder) -> Self {
        self.passage_sort_comparator = order;
        self
    }

    /// `getMaxLength()`.
    pub fn max_length(&self) -> i32 {
        self.max_length
    }

    /// `getFlags(field)`.
    pub fn flags(&self) -> Flags {
        self.flags
    }

    /// `getFieldMatcher(field)`: by default the query's field must be the
    /// highlighted one (`requireFieldMatch`).
    fn field_matcher_for(&self, field: &str) -> FieldMatcher {
        match &self.field_matcher {
            Some(m) => Arc::clone(m),
            None => {
                let field = field.to_string();
                Arc::new(move |qf: &str| field == qf)
            }
        }
    }

    /// `getOffsetSource(field)`, from the index's merged field infos.
    pub fn offset_source(&self, field: &str) -> OffsetSource {
        let Some(reader) = self.reader else {
            return OffsetSource::Analysis;
        };
        let mut index_options = None;
        let mut has_tv = false;
        for seg in reader.segment_readers() {
            if let Some(fi) = seg.field_infos().field_by_name(field) {
                if index_options.is_none() {
                    index_options = Some(fi.index_options);
                }
                has_tv |= fi.store_term_vectors;
            }
        }
        offset_source_for_field(index_options, has_tv)
    }

    fn strategy(
        &self,
        field: &str,
        query: &Clause,
        all_terms: &BTreeSet<crate::query_visitor::Term>,
    ) -> Result<Strategy> {
        let components = Components::new(
            field,
            self.field_matcher_for(field),
            query,
            all_terms,
            self.flags,
        );
        let source = components.optimized_offset_source(self.offset_source(field));
        Strategy::new(source, components, self.analyzer.clone())
    }

    /// `getFieldHighlighter(field, query, allTerms, maxPassages)`: the
    /// strategies (several when the field masks others) and the passage
    /// logic.
    fn field_highlighter(
        &self,
        field: &str,
        query: &Clause,
        all_terms: &BTreeSet<crate::query_visitor::Term>,
        max_passages: i32,
    ) -> Result<(Vec<Strategy>, FieldHighlighter)> {
        let masked = self.masked_fields.as_ref().and_then(|f| f(field));
        let mut strategies = Vec::new();
        match masked {
            Some(masked) if !masked.is_empty() => {
                for m in &masked {
                    strategies.push(self.strategy(m, query, all_terms)?);
                }
                // adding the original field as well
                strategies.push(self.strategy(field, query, all_terms)?);
            }
            _ => strategies.push(self.strategy(field, query, all_terms)?),
        }
        let highlighter = FieldHighlighter {
            field: field.to_string(),
            break_iterator: Box::new(SplittingBreakIterator::new(
                (self.break_iterator)(),
                MULTIVAL_SEP_CHAR,
            )),
            passage_scorer: self.scorer,
            max_passages,
            max_no_highlight_passages: self.max_no_highlight_passages,
            passage_formatter: Box::new(SharedFormatter(Arc::clone(&self.formatter))),
            passage_sort_comparator: self.passage_sort_comparator,
        };
        Ok((strategies, highlighter))
    }

    /// `highlightFields(fields, query, docIds, maxPassages)`: for each field,
    /// the snippet of each document, `None` where there is nothing to show
    /// -- the `toString()` of [`Self::highlight_fields_as_objects`]'.
    ///
    /// # Errors
    /// As [`Self::highlight_fields_as_objects`].
    pub fn highlight_fields(
        &self,
        searcher: &IndexSearcher<'_, '_>,
        fields: &[&str],
        query: &Clause,
        doc_ids: &[i32],
        max_passages: &[i32],
    ) -> Result<BTreeMap<String, Vec<Option<String>>>> {
        let objects =
            self.highlight_fields_as_objects(searcher, fields, query, doc_ids, max_passages)?;
        Ok(objects
            .into_iter()
            .map(|(field, docs)| {
                let snippets = docs.into_iter().map(|o| o.map(|o| o.to_string())).collect();
                (field, snippets)
            })
            .collect())
    }

    /// `highlightFieldsAsObjects(fields, query, docIds, maxPassages)`: for
    /// each field, what the formatter made of each document, `None` where
    /// there is nothing to show. The documents are highlighted in doc id
    /// order, in batches whose stored values are loaded together
    /// (`loadFieldValues` with the threshold
    /// `calculateOptimalCacheCharsThreshold` picks), and within a batch
    /// field by field.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for no fields, or `fields` and
    /// `max_passages` of different lengths, or a field indexed without
    /// offsets that is highlighted from its postings; `IllegalStateException`
    /// without a reader.
    pub fn highlight_fields_as_objects(
        &self,
        searcher: &IndexSearcher<'_, '_>,
        fields: &[&str],
        query: &Clause,
        doc_ids: &[i32],
        max_passages: &[i32],
    ) -> Result<BTreeMap<String, Vec<Option<HighlightObject>>>> {
        if fields.is_empty() {
            return Err(Error::IllegalArgument("fieldsIn must not be empty".into()));
        }
        if fields.len() != max_passages.len() {
            return Err(Error::IllegalArgument(
                "invalid number of maxPassagesIn".into(),
            ));
        }
        let reader = self.reader.ok_or_else(|| {
            Error::Search(crate::Error::IllegalState(
                "This method requires that an indexSearcher was passed in the \
                 constructor.  Perhaps you mean to call highlightWithoutSearcher?"
                    .into(),
            ))
        })?;
        // sort docs and fields for sequential i/o: each doc with its index in
        // the input, each field with its maxPassages (both sorts stable)
        let mut docs: Vec<(i32, usize)> = doc_ids.iter().copied().zip(0..).collect();
        docs.sort_by_key(|&(doc, _)| doc);
        let mut sorted: Vec<(&str, i32)> = fields
            .iter()
            .copied()
            .zip(max_passages.iter().copied())
            .collect();
        sorted.sort_by(|a, b| a.0.cmp(b.0));
        let all_terms = extract_terms(query);
        let mut highlighters = Vec::new();
        let (mut num_term_vectors, mut num_postings) = (0, 0);
        for (field, max) in &sorted {
            let (strategies, highlighter) =
                self.field_highlighter(field, query, &all_terms, *max)?;
            // a masking field's strategy reports its first masked field's source
            match strategies[0].source {
                OffsetSource::TermVectors => num_term_vectors += 1,
                OffsetSource::Postings => num_postings += 1,
                OffsetSource::PostingsWithTermVectors => {
                    num_term_vectors += 1;
                    num_postings += 1;
                }
                OffsetSource::Analysis | OffsetSource::NoneNeeded => {}
            }
            highlighters.push((strategies, highlighter));
        }
        let cache_chars_threshold =
            self.calculate_optimal_cache_chars_threshold(num_term_vectors, num_postings);
        let names: Vec<&str> = sorted.iter().map(|(f, _)| *f).collect();
        let mut results: Vec<Vec<Option<HighlightObject>>> = names
            .iter()
            .map(|_| (0..doc_ids.len()).map(|_| None).collect())
            .collect();
        let mut batch_start = 0;
        while batch_start < docs.len() {
            let batch = self.load_field_values(
                reader,
                searcher,
                &names,
                &docs[batch_start..],
                cache_chars_threshold,
            )?;
            // per field first, then by doc
            for (f, (strategies, highlighter)) in highlighters.iter_mut().enumerate() {
                for (values, &(doc_id, doc_in_index)) in batch.iter().zip(&docs[batch_start..]) {
                    let Some(content) = &values[f] else {
                        continue;
                    };
                    let leaf = searcher.segment_of(doc_id).ok_or_else(|| {
                        Error::IllegalArgument(format!("doc {doc_id} is in no segment"))
                    })?;
                    let segment = &searcher.segments()[leaf];
                    let matches = |c: &Clause| {
                        crate::matches::matches(searcher, &c.clone().rewrite(), doc_id)
                    };
                    let ctx = LeafCtx {
                        reader: &reader.segment_readers()[leaf],
                        segment,
                        doc: doc_id - segment.doc_base,
                        matches: &matches,
                    };
                    results[f][doc_in_index] =
                        highlight_field_for_doc(strategies, highlighter, Some(&ctx), content)?;
                }
            }
            batch_start += batch.len();
        }
        Ok(names.into_iter().map(str::to_string).zip(results).collect())
    }

    /// `calculateOptimalCacheCharsThreshold`: one document at a time when
    /// every field is analysed (nothing else is read from disk) or when two
    /// or more read term vectors; else [`Self::cache_field_val_chars_threshold`].
    fn calculate_optimal_cache_chars_threshold(
        &self,
        num_term_vectors: i32,
        num_postings: i32,
    ) -> i32 {
        if (num_postings == 0 && num_term_vectors == 0) || num_term_vectors >= 2 {
            0
        } else {
            self.cache_field_val_chars_threshold
        }
    }

    /// `highlight(field, query, docIds, maxPassages)`.
    ///
    /// # Errors
    /// As [`Self::highlight_fields`].
    pub fn highlight(
        &self,
        searcher: &IndexSearcher<'_, '_>,
        field: &str,
        query: &Clause,
        doc_ids: &[i32],
        max_passages: i32,
    ) -> Result<Vec<Option<String>>> {
        let mut m = self.highlight_fields(searcher, &[field], query, doc_ids, &[max_passages])?;
        Ok(m.remove(field).unwrap_or_default())
    }

    /// `highlightWithoutSearcher(field, query, content, maxPassages)` as
    /// text: the text is analyzed, as there is no index to read offsets
    /// from.
    ///
    /// # Errors
    /// As [`Self::highlight_without_searcher_as_object`].
    pub fn highlight_without_searcher(
        &self,
        field: &str,
        query: &Clause,
        content: &str,
        max_passages: i32,
    ) -> Result<Option<String>> {
        Ok(self
            .highlight_without_searcher_as_object(field, query, content, max_passages)?
            .map(|o| o.to_string()))
    }

    /// `highlightWithoutSearcher(field, query, content, maxPassages)`: what
    /// the formatter made of the text.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when the highlighter has a reader.
    pub fn highlight_without_searcher_as_object(
        &self,
        field: &str,
        query: &Clause,
        content: &str,
        max_passages: i32,
    ) -> Result<Option<HighlightObject>> {
        if self.reader.is_some() {
            return Err(Error::IllegalArgument(
                "highlightWithoutSearcher should only be called on a UnifiedHighlighter \
                 without an IndexSearcher."
                    .into(),
            ));
        }
        let all_terms = extract_terms(query);
        let (strategies, mut highlighter) =
            self.field_highlighter(field, query, &all_terms, max_passages)?;
        let content: Vec<u16> = content.encode_utf16().collect();
        highlight_field_for_doc(&strategies, &mut highlighter, None, &content)
    }

    /// `loadFieldValues(fields, docIter, cacheCharsThreshold)` with
    /// `LimitedStoredFieldVisitor`: from the first of `docs` on, each
    /// document's fields (string values joined by [`MULTIVAL_SEP_CHAR`] and
    /// cut to `maxLength` UTF-16 units; `None` for a field without a value),
    /// until their characters pass the threshold -- one document when it is
    /// 0 or less.
    fn load_field_values(
        &self,
        reader: &DirectoryReader,
        searcher: &IndexSearcher<'_, '_>,
        fields: &[&str],
        docs: &[(i32, usize)],
        cache_chars_threshold: i32,
    ) -> Result<Vec<Vec<Option<Vec<u16>>>>> {
        let mut batch = Vec::new();
        let mut sum_chars = 0usize;
        let threshold = usize::try_from(cache_chars_threshold).unwrap_or(0);
        for &(doc_id, _) in docs {
            let values = self.load_doc_field_values(reader, searcher, fields, doc_id)?;
            sum_chars += values.iter().flatten().map(Vec::len).sum::<usize>();
            batch.push(values);
            // Java's `sumChars <= threshold && threshold != 0` to go on
            if cache_chars_threshold <= 0 || sum_chars > threshold {
                break;
            }
        }
        Ok(batch)
    }

    /// One document's values, as `LimitedStoredFieldVisitor` reads them.
    fn load_doc_field_values(
        &self,
        reader: &DirectoryReader,
        searcher: &IndexSearcher<'_, '_>,
        fields: &[&str],
        doc_id: i32,
    ) -> Result<Vec<Option<Vec<u16>>>> {
        let leaf = searcher
            .segment_of(doc_id)
            .ok_or_else(|| Error::IllegalArgument(format!("doc {doc_id} is in no segment")))?;
        let local = doc_id - searcher.segments()[leaf].doc_base;
        let segment = &reader.segment_readers()[leaf];
        let mut values: Vec<Option<Vec<u16>>> = vec![None; fields.len()];
        let max_length = usize::try_from(self.max_length).unwrap_or(0);
        let doc = segment
            .stored_document(local)
            .map_err(|e| Error::Io(e.to_string()))?;
        let Some(doc) = doc else {
            return Ok(values);
        };
        for f in &doc.fields {
            let Some(fi) = segment.field_infos().field_by_number(f.field_number) else {
                continue;
            };
            let Ok(slot) = fields.binary_search(&fi.name.as_str()) else {
                continue;
            };
            // `needsField`: a field that is full needs no more
            if values[slot].as_ref().is_some_and(|v| v.len() >= max_length) {
                continue;
            }
            // `stringField`: only string values reach it
            let FieldValue::String(text) = &f.value else {
                continue;
            };
            let value: Vec<u16> = text.encode_utf16().collect();
            match &mut values[slot] {
                None => values[slot] = Some(value[..value.len().min(max_length)].to_vec()),
                Some(cur) => {
                    let budget = max_length - cur.len();
                    if budget == 0 {
                        continue;
                    }
                    cur.push(MULTIVAL_SEP_CHAR);
                    cur.extend_from_slice(&value[..(budget - 1).min(value.len())]);
                }
            }
        }
        Ok(values)
    }
}

/// `FieldHighlighter.highlightFieldForDoc` over its strategies.
fn highlight_field_for_doc(
    strategies: &[Strategy],
    highlighter: &mut FieldHighlighter,
    ctx: Option<&LeafCtx<'_>>,
    content: &[u16],
) -> Result<Option<HighlightObject>> {
    // nothing to do
    if content.is_empty() {
        return Ok(None);
    }
    highlighter.break_iterator.set_text(content);
    let enum_ = if strategies.len() == 1 {
        strategies[0].offsets_enum(ctx, content)?
    } else {
        multi_fields_enum(strategies, ctx, content)?
    };
    let mut off: Box<dyn OffsetsEnum> = enum_.unwrap_or_else(|| Box::new(EmptyOffsets));
    highlighter.highlight_content(content, off.as_mut())
}

/// An `Arc`'d formatter as a boxed one.
struct SharedFormatter(Arc<dyn PassageFormatter + Send + Sync>);

impl PassageFormatter for SharedFormatter {
    fn format(&self, passages: &[Passage], content: &[u16]) -> String {
        self.0.format(passages, content)
    }
    fn format_object(&self, passages: &[Passage], content: &[u16]) -> HighlightObject {
        self.0.format_object(passages, content)
    }
}
