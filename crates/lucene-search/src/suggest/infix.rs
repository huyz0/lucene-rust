//! `org.apache.lucene.search.suggest.analyzing.AnalyzingInfixSuggester` and
//! `BlendedInfixSuggester` (M13 T13.3c): suggestions matched anywhere in the
//! text, not only at its start. Every suggestion is a document with its
//! analyzed `text`, the edge n-grams of each token (`textgrams`, so a prefix
//! shorter than `minPrefixChars` is a plain term lookup), its exact text, its
//! weight, its payload and its contexts; a lookup is a boolean query over the
//! query's tokens (the last one a prefix unless the query ends in a
//! separator), ordered by weight, descending.
//!
//! Java keeps those documents in a Lucene index in a private `Directory`,
//! sorted by weight, and reads them back through a `SearcherManager`. This
//! port keeps them in memory, with the same visibility rules: `build` makes
//! its documents searchable at once, `add`/`update` only after `refresh`.
//! Because nothing is persisted, there is no `Directory` argument and no
//! "load an index built by an earlier instance". Contexts are reported per
//! flushed segment (as Java's per-segment `SortedSetDocValues` do): one
//! segment per `build`, `refresh` or reopen of the writer; a merge of
//! segments is not modelled.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ops::Bound;
use std::sync::Arc;

use lucene_analysis::{collect_tokens, AnalysisError, Analyzer};
use lucene_store::{DataInput, DataOutput, Error, Result};

use super::input::InputIterator;
use super::lookup::{charsequence_compare, illegal_state, Lookup, LookupResult};
pub use crate::query_visitor::Occur;

/// `DEFAULT_MIN_PREFIX_CHARS`.
pub const DEFAULT_MIN_PREFIX_CHARS: i32 = 4;
/// `DEFAULT_ALL_TERMS_REQUIRED`.
pub const DEFAULT_ALL_TERMS_REQUIRED: bool = true;
/// `DEFAULT_HIGHLIGHT`.
pub const DEFAULT_HIGHLIGHT: bool = true;
/// `DEFAULT_CLOSE_INDEXWRITER_ON_BUILD`.
pub const DEFAULT_CLOSE_INDEXWRITER_ON_BUILD: bool = true;

fn analysis_err(e: AnalysisError) -> Error {
    Error::IllegalArgument(e.to_string())
}

/// One suggestion, as indexed.
#[derive(Clone)]
struct Doc {
    text: String,
    weight: i64,
    payload: Option<Vec<u8>>,
    contexts: Option<BTreeSet<Vec<u8>>>,
    /// The `text` field's terms with the position of each first occurrence
    /// (the term vector `BlendedInfixSuggester` reads).
    terms: BTreeMap<String, i32>,
    /// The `textgrams` field's terms.
    grams: BTreeSet<String>,
    /// The flushed segment this document lives in.
    segment: usize,
}

/// A fixed view of the index, as a `SearcherManager` hands out.
#[derive(Clone, Default)]
struct Snapshot {
    /// The live documents in index order: weight, descending.
    docs: Vec<Doc>,
    /// Segments with a `contexts` doc-values field.
    segments_with_contexts: HashSet<usize>,
    /// Whether a segment of the view has a `payloads` doc-values field, so
    /// `MultiDocValues` hands out payloads (empty for a document without one).
    has_payloads: bool,
    /// The postings of the `text` field: each term's documents, ascending.
    text: BTreeMap<String, Vec<usize>>,
    /// The postings of the `textgrams` field.
    grams: HashMap<String, Vec<usize>>,
    /// The postings of the `contexts` field.
    contexts: HashMap<Vec<u8>, Vec<usize>>,
}

/// The fields a [`Query`] looks in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Field {
    Text,
    TextGrams,
    Contexts,
}

/// The `BooleanQuery`/`TermQuery`/`PrefixQuery` subset a lookup builds.
enum Query {
    Term(Field, Vec<u8>),
    Prefix(Vec<u8>),
    Bool(Vec<(Occur, Query)>),
}

impl Snapshot {
    /// The view over `docs` (already in index order) with its postings.
    fn new(docs: Vec<Doc>, segments_with_contexts: HashSet<usize>, has_payloads: bool) -> Self {
        let mut text: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut grams: HashMap<String, Vec<usize>> = HashMap::new();
        let mut contexts: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();
        for (i, doc) in docs.iter().enumerate() {
            for term in doc.terms.keys() {
                text.entry(term.clone()).or_default().push(i);
            }
            for gram in &doc.grams {
                grams.entry(gram.clone()).or_default().push(i);
            }
            for context in doc.contexts.iter().flatten() {
                contexts.entry(context.clone()).or_default().push(i);
            }
        }
        Snapshot {
            docs,
            segments_with_contexts,
            has_payloads,
            text,
            grams,
            contexts,
        }
    }

    /// The documents `query` matches, ascending (so in index order).
    fn matching(&self, query: &Query) -> Vec<usize> {
        let postings = |list: Option<&Vec<usize>>| list.cloned().unwrap_or_default();
        match query {
            Query::Term(Field::Text, t) => {
                postings(std::str::from_utf8(t).ok().and_then(|t| self.text.get(t)))
            }
            Query::Term(Field::TextGrams, t) => {
                postings(std::str::from_utf8(t).ok().and_then(|t| self.grams.get(t)))
            }
            Query::Term(Field::Contexts, t) => postings(self.contexts.get(t)),
            Query::Prefix(p) => {
                let Ok(p) = std::str::from_utf8(p) else {
                    return Vec::new();
                };
                // terms sort by their bytes, so a prefix's terms are contiguous
                let mut out: Vec<usize> = self
                    .text
                    .range::<str, _>((Bound::Included(p), Bound::Unbounded))
                    .take_while(|(term, _)| term.starts_with(p))
                    .flat_map(|(_, docs)| docs.iter().copied())
                    .collect();
                out.sort_unstable();
                out.dedup();
                out
            }
            Query::Bool(clauses) => {
                let mut required: Option<Vec<usize>> = None;
                let mut optional: Option<Vec<usize>> = None;
                let mut excluded: Vec<usize> = Vec::new();
                for (occur, q) in clauses {
                    let docs = self.matching(q);
                    match occur {
                        Occur::Must | Occur::Filter => {
                            required = Some(match required {
                                None => docs,
                                Some(r) => intersect(&r, &docs),
                            });
                        }
                        Occur::Should => {
                            let mut u = optional.unwrap_or_default();
                            u.extend(docs);
                            optional = Some(u);
                        }
                        Occur::MustNot => excluded.extend(docs),
                    }
                }
                // a query with no positive clause matches nothing; with only
                // optional clauses, one of them must match
                let mut out = match (required, optional) {
                    (Some(r), _) => r,
                    (None, Some(mut o)) => {
                        o.sort_unstable();
                        o.dedup();
                        o
                    }
                    (None, None) => return Vec::new(),
                };
                if !excluded.is_empty() {
                    excluded.sort_unstable();
                    out.retain(|d| excluded.binary_search(d).is_err());
                }
                out
            }
        }
    }
}

/// The documents in both ascending lists.
fn intersect(a: &[usize], b: &[usize]) -> Vec<usize> {
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    small
        .iter()
        .copied()
        .filter(|d| large.binary_search(d).is_ok())
        .collect()
}

/// `BlendedInfixSuggester.BlenderType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlenderType {
    /// Application dependent: see [`BlendedInfixSuggester::with_custom_coefficient`].
    Custom,
    /// `weight*(1 - 0.10*position)`.
    PositionLinear,
    /// `weight/(1+position)`.
    PositionReciprocal,
    /// `weight/pow(1+position, exponent)`.
    PositionExponentialReciprocal,
}

/// The blending a [`BlendedInfixSuggester`] applies to the hits.
struct Blend<'a> {
    blender: BlenderType,
    num_factor: usize,
    exponent: f64,
    custom: Option<&'a (dyn Fn(i32) -> f64 + Send + Sync)>,
}

/// `AnalyzingInfixSuggester`.
pub struct AnalyzingInfixSuggester {
    index_analyzer: Arc<Analyzer>,
    query_analyzer: Arc<Analyzer>,
    min_prefix_chars: i32,
    commit_on_build: bool,
    all_terms_required: bool,
    highlight: bool,
    close_index_writer_on_build: bool,
    /// The live documents of the index writer, in insertion order.
    docs: Vec<Doc>,
    segments_with_contexts: HashSet<usize>,
    /// Segments with a `payloads` doc-values field.
    segments_with_payloads: HashSet<usize>,
    /// The segment new documents go to, and whether it has any yet.
    segment: usize,
    segment_has_docs: bool,
    writer_open: bool,
    searcher: Option<Snapshot>,
}

impl AnalyzingInfixSuggester {
    /// `new AnalyzingInfixSuggester(dir, analyzer)`.
    pub fn new(analyzer: Arc<Analyzer>) -> Self {
        Self::with_options(
            analyzer.clone(),
            analyzer,
            DEFAULT_MIN_PREFIX_CHARS,
            false,
            DEFAULT_ALL_TERMS_REQUIRED,
            DEFAULT_HIGHLIGHT,
            DEFAULT_CLOSE_INDEXWRITER_ON_BUILD,
        )
        .expect("the defaults are valid")
    }

    /// The full constructor.
    pub fn with_options(
        index_analyzer: Arc<Analyzer>,
        query_analyzer: Arc<Analyzer>,
        min_prefix_chars: i32,
        commit_on_build: bool,
        all_terms_required: bool,
        highlight: bool,
        close_index_writer_on_build: bool,
    ) -> Result<Self> {
        if min_prefix_chars < 0 {
            return Err(Error::IllegalArgument(format!(
                "minPrefixChars must be >= 0; got: {min_prefix_chars}"
            )));
        }
        Ok(Self {
            index_analyzer,
            query_analyzer,
            min_prefix_chars,
            commit_on_build,
            all_terms_required,
            highlight,
            close_index_writer_on_build,
            docs: Vec::new(),
            segments_with_contexts: HashSet::new(),
            segments_with_payloads: HashSet::new(),
            segment: 0,
            segment_has_docs: false,
            writer_open: false,
            searcher: None,
        })
    }

    /// `buildDocument`.
    fn build_document(
        &self,
        text: &[u8],
        contexts: Option<&BTreeSet<Vec<u8>>>,
        weight: i64,
        payload: Option<&[u8]>,
    ) -> Result<Doc> {
        let text = String::from_utf8_lossy(text).into_owned();
        let tokens = self
            .index_analyzer
            .token_stream("text", &text)
            .map_err(analysis_err)
            .and_then(|mut ts| collect_tokens(&mut ts).map_err(analysis_err))?;
        let mut terms: BTreeMap<String, i32> = BTreeMap::new();
        let mut grams = BTreeSet::new();
        let mut position = -1i32;
        for token in &tokens.tokens {
            position = position.saturating_add(token.position_increment);
            terms.entry(token.term.clone()).or_insert(position);
            if self.min_prefix_chars > 0 {
                // EdgeNGramTokenFilter(1, minPrefixChars, false)
                let chars: Vec<char> = token.term.chars().collect();
                let longest = chars
                    .len()
                    .min(usize::try_from(self.min_prefix_chars).unwrap_or(usize::MAX));
                for n in 1..=longest {
                    grams.insert(chars[..n].iter().collect());
                }
            }
        }
        Ok(Doc {
            text,
            weight,
            payload: payload.map(<[u8]>::to_vec),
            contexts: contexts.cloned(),
            terms,
            grams,
            segment: self.segment,
        })
    }

    fn push(&mut self, doc: Doc) {
        if doc.contexts.as_ref().is_some_and(|c| !c.is_empty()) {
            self.segments_with_contexts.insert(doc.segment);
        }
        if doc.payload.is_some() {
            self.segments_with_payloads.insert(doc.segment);
        }
        self.segment_has_docs = true;
        self.docs.push(doc);
    }

    /// The writer's reader sees every document; the next ones go to a new
    /// segment.
    fn flush_and_snapshot(&mut self) {
        if self.segment_has_docs {
            self.segment += 1;
            self.segment_has_docs = false;
        }
        // the field survives the documents that brought it: Lucene 10.5.0
        // still hands out payloads after every such document is updated away
        // (`GenInfixSuggest`'s `payloads.tsv`)
        let has_payloads = !self.segments_with_payloads.is_empty();
        // the index is sorted by weight, descending (Java's index sort on the
        // weight doc values); a stable sort keeps ties in insertion order
        let mut docs = self.docs.clone();
        docs.sort_by_key(|d| std::cmp::Reverse(d.weight));
        self.searcher = Some(Snapshot::new(
            docs,
            self.segments_with_contexts.clone(),
            has_payloads,
        ));
    }

    /// `ensureOpen()`.
    fn ensure_open(&mut self) {
        if !self.writer_open {
            self.writer_open = true;
            self.flush_and_snapshot();
        }
    }

    /// `commit()`.
    pub fn commit(&mut self) -> Result<()> {
        if self.writer_open {
            if self.segment_has_docs {
                self.segment += 1;
                self.segment_has_docs = false;
            }
            Ok(())
        } else if self.searcher.is_none() || !self.close_index_writer_on_build {
            Err(illegal_state(
                "Cannot commit on an closed writer. Add documents first",
            ))
        } else {
            Ok(())
        }
    }

    /// `add(text, contexts, weight, payload)`: searchable after [`Self::refresh`].
    pub fn add(
        &mut self,
        text: &[u8],
        contexts: Option<&BTreeSet<Vec<u8>>>,
        weight: i64,
        payload: Option<&[u8]>,
    ) -> Result<()> {
        self.ensure_open();
        let doc = self.build_document(text, contexts, weight, payload)?;
        self.push(doc);
        Ok(())
    }

    /// `update(text, contexts, weight, payload)`: replaces the suggestion
    /// with exactly this text.
    pub fn update(
        &mut self,
        text: &[u8],
        contexts: Option<&BTreeSet<Vec<u8>>>,
        weight: i64,
        payload: Option<&[u8]>,
    ) -> Result<()> {
        self.ensure_open();
        let doc = self.build_document(text, contexts, weight, payload)?;
        self.docs.retain(|d| d.text != doc.text);
        self.push(doc);
        Ok(())
    }

    /// `refresh()`.
    pub fn refresh(&mut self) -> Result<()> {
        if self.searcher.is_none() {
            return Err(illegal_state("suggester was not built"));
        }
        if self.writer_open {
            self.flush_and_snapshot();
        }
        Ok(())
    }

    /// `close()`.
    pub fn close(&mut self) {
        self.searcher = None;
        self.writer_open = false;
    }

    /// `getLastTokenQuery(token)`.
    fn last_token_query(&self, token: &str) -> Query {
        let length = i64::try_from(token.encode_utf16().count()).unwrap_or(i64::MAX);
        if length < i64::from(self.min_prefix_chars) {
            // The leading ngram was directly indexed
            Query::Term(Field::TextGrams, token.as_bytes().to_vec())
        } else {
            Query::Prefix(token.as_bytes().to_vec())
        }
    }

    /// `toQuery(Set<BytesRef>)`: every context optional.
    fn context_query_of_set(contexts: Option<&BTreeSet<Vec<u8>>>) -> Option<Vec<(Vec<u8>, Occur)>> {
        let contexts = contexts?;
        if contexts.is_empty() {
            return None;
        }
        Some(
            contexts
                .iter()
                .map(|c| (c.clone(), Occur::Should))
                .collect(),
        )
    }

    /// `lookup(key, contexts, num, allTermsRequired, doHighlight)`.
    pub fn lookup_with(
        &self,
        key: &str,
        contexts: Option<&BTreeSet<Vec<u8>>>,
        num: usize,
        all_terms_required: bool,
        do_highlight: bool,
    ) -> Result<Vec<LookupResult>> {
        let context_query = Self::context_query_of_set(contexts);
        self.search(
            key,
            context_query.as_deref(),
            num,
            all_terms_required,
            do_highlight,
            None,
        )
    }

    /// `lookup(key, Map<BytesRef, Occur> contextInfo, num, allTermsRequired,
    /// doHighlight)`: each context with its own clause type.
    pub fn lookup_context_info(
        &self,
        key: &str,
        context_info: &BTreeMap<Vec<u8>, Occur>,
        num: usize,
        all_terms_required: bool,
        do_highlight: bool,
    ) -> Result<Vec<LookupResult>> {
        let context_query = Self::context_query_of_info(context_info);
        self.search(
            key,
            context_query.as_deref(),
            num,
            all_terms_required,
            do_highlight,
            None,
        )
    }

    /// The context clauses of a `Map<BytesRef, Occur>`, `None` when empty.
    fn context_query_of_info(
        context_info: &BTreeMap<Vec<u8>, Occur>,
    ) -> Option<Vec<(Vec<u8>, Occur)>> {
        let context_query: Vec<(Vec<u8>, Occur)> =
            context_info.iter().map(|(c, o)| (c.clone(), *o)).collect();
        (!context_query.is_empty()).then_some(context_query)
    }

    /// The `lookup(key, BooleanQuery contextQuery, ...)` all the others end
    /// in, with the context clauses `(context, occur)`.
    #[allow(clippy::too_many_arguments)]
    fn search(
        &self,
        key: &str,
        context_query: Option<&[(Vec<u8>, Occur)]>,
        num: usize,
        all_terms_required: bool,
        do_highlight: bool,
        blend: Option<&Blend<'_>>,
    ) -> Result<Vec<LookupResult>> {
        let Some(snapshot) = &self.searcher else {
            return Err(illegal_state("suggester was not built"));
        };
        let occur = if all_terms_required {
            Occur::Must
        } else {
            Occur::Should
        };

        let mut matched_tokens: HashSet<String> = HashSet::new();
        let mut prefix_token: Option<String> = None;
        let mut query: Vec<(Occur, Query)> = Vec::new();
        {
            let mut ts = self
                .query_analyzer
                .token_stream("", key)
                .map_err(analysis_err)?;
            let analyzed = collect_tokens(&mut ts).map_err(analysis_err)?;
            let mut last_token: Option<&str> = None;
            let mut max_end_offset = -1;
            for token in &analyzed.tokens {
                if let Some(last) = last_token {
                    matched_tokens.insert(last.to_string());
                    query.push((occur, Query::Term(Field::Text, last.as_bytes().to_vec())));
                }
                last_token = Some(&token.term);
                max_end_offset = max_end_offset.max(token.end_offset);
            }
            if let Some(last) = last_token {
                let last_query = if max_end_offset == analyzed.final_offset {
                    // Use PrefixQuery (or the ngram equivalent) when there
                    // was no trailing discarded chars in the string (e.g.
                    // whitespace), so that a query not ending with a space
                    // shows prefix matches for that token
                    prefix_token = Some(last.to_string());
                    self.last_token_query(last)
                } else {
                    // Use TermQuery for an exact match if there were
                    // trailing discarded chars (e.g. whitespace)
                    matched_tokens.insert(last.to_string());
                    Query::Term(Field::Text, last.as_bytes().to_vec())
                };
                query.push((occur, last_query));
            }
        }

        let query = if let Some(clauses) = context_query {
            let all_must_not = clauses.iter().all(|(_, o)| *o == Occur::MustNot);
            let context_bool = |clauses: &[(Vec<u8>, Occur)]| {
                Query::Bool(
                    clauses
                        .iter()
                        .map(|(c, o)| (*o, Query::Term(Field::Contexts, c.clone())))
                        .collect(),
                )
            };
            if all_must_not {
                // All are MUST_NOT: add the contextQuery to the main query
                // instead (not as sub-query)
                for (c, o) in clauses {
                    query.push((*o, Query::Term(Field::Contexts, c.clone())));
                }
                Query::Bool(query)
            } else if !all_terms_required {
                // We must carefully upgrade the query clauses to MUST
                Query::Bool(vec![
                    (Occur::Must, Query::Bool(query)),
                    (Occur::Must, context_bool(clauses)),
                ])
            } else {
                query.push((Occur::Must, context_bool(clauses)));
                Query::Bool(query)
            }
        } else {
            Query::Bool(query)
        };

        if num == 0 {
            return Err(Error::IllegalArgument(
                "numHits must be > 0; please use TotalHitCountCollector if you just need the total hit count"
                    .into(),
            ));
        }
        // The view is in index order (weight, descending) and the matches
        // come ascending: the first `num` are the top hits, as with Java's
        // index sort
        let hits: Vec<&Doc> = snapshot
            .matching(&query)
            .into_iter()
            .take(num)
            .filter_map(|d| snapshot.docs.get(d))
            .collect();

        match blend {
            None => self.create_results(
                snapshot,
                &hits,
                do_highlight,
                &matched_tokens,
                prefix_token.as_deref(),
            ),
            Some(blend) => self.create_blended_results(
                snapshot,
                &hits,
                num,
                key,
                do_highlight,
                &matched_tokens,
                prefix_token.as_deref(),
                blend,
            ),
        }
    }

    /// The payload doc values, `None` when no segment of the view has any.
    fn payload_of(snapshot: &Snapshot, doc: &Doc) -> Option<Vec<u8>> {
        if snapshot.has_payloads {
            Some(doc.payload.clone().unwrap_or_default())
        } else {
            None
        }
    }

    /// `createResults`.
    fn create_results(
        &self,
        snapshot: &Snapshot,
        hits: &[&Doc],
        do_highlight: bool,
        matched_tokens: &HashSet<String>,
        prefix_token: Option<&str>,
    ) -> Result<Vec<LookupResult>> {
        let mut results = Vec::with_capacity(hits.len());
        for doc in hits {
            let payload = Self::payload_of(snapshot, doc);
            // doc values per segment: a segment with none has no contexts
            let contexts = if snapshot.segments_with_contexts.contains(&doc.segment) {
                Some(doc.contexts.clone().unwrap_or_default())
            } else {
                None
            };
            let highlight_key = if do_highlight {
                Some(self.highlight_text(&doc.text, matched_tokens, prefix_token)?)
            } else {
                None
            };
            results.push(LookupResult::with_all(
                doc.text.clone(),
                highlight_key,
                doc.weight,
                payload,
                contexts,
            ));
        }
        Ok(results)
    }

    /// `highlight(text, matchedTokens, prefixToken)` with the default
    /// `addNonMatch`/`addWholeMatch`/`addPrefixMatch`.
    fn highlight_text(
        &self,
        text: &str,
        matched_tokens: &HashSet<String>,
        prefix_token: Option<&str>,
    ) -> Result<String> {
        // offsets are UTF-16 units
        let units: Vec<u16> = text.encode_utf16().collect();
        let slice = |from: i32, to: i32| -> String {
            let from = usize::try_from(from).unwrap_or(0).min(units.len());
            let to = usize::try_from(to).unwrap_or(0).min(units.len()).max(from);
            String::from_utf16_lossy(&units[from..to])
        };
        let mut ts = self
            .query_analyzer
            .token_stream("text", text)
            .map_err(analysis_err)?;
        let analyzed = collect_tokens(&mut ts).map_err(analysis_err)?;
        let mut sb = String::new();
        let mut upto = 0i32;
        for token in &analyzed.tokens {
            let (start, end) = (token.start_offset, token.end_offset);
            if upto < start {
                sb.push_str(&slice(upto, start));
                upto = start;
            } else if upto > start {
                continue;
            }
            if matched_tokens.contains(&token.term) {
                // Token matches.
                sb.push_str("<b>");
                sb.push_str(&slice(start, end));
                sb.push_str("</b>");
                upto = end;
            } else if prefix_token.is_some_and(|p| token.term.starts_with(p)) {
                let surface = slice(start, end);
                let prefix_len =
                    i32::try_from(prefix_token.map_or(0, |p| p.encode_utf16().count()))
                        .unwrap_or(i32::MAX);
                let surface_len = end.saturating_sub(start);
                if prefix_len >= surface_len {
                    sb.push_str("<b>");
                    sb.push_str(&surface);
                    sb.push_str("</b>");
                } else {
                    sb.push_str("<b>");
                    sb.push_str(&slice(start, start.saturating_add(prefix_len)));
                    sb.push_str("</b>");
                    sb.push_str(&slice(start.saturating_add(prefix_len), end));
                }
                upto = end;
            }
        }
        let end_offset = analyzed.final_offset;
        if upto < end_offset {
            sb.push_str(&slice(upto, i32::MAX));
        }
        Ok(sb)
    }

    /// `BlendedInfixSuggester.createResults`.
    #[allow(clippy::too_many_arguments)]
    fn create_blended_results(
        &self,
        snapshot: &Snapshot,
        hits: &[&Doc],
        num: usize,
        key: &str,
        do_highlight: bool,
        matched_tokens: &HashSet<String>,
        prefix_token: Option<&str>,
        blend: &Blend<'_>,
    ) -> Result<Vec<LookupResult>> {
        // we reduce the num to the one initially requested
        let actual_num = num / blend.num_factor.max(1);
        // a TreeSet ordered by (value, key, payload)
        let mut results: Vec<LookupResult> = Vec::new();
        for doc in hits {
            let mut weight = doc.weight;
            let payload = Self::payload_of(snapshot, doc);
            let coefficient = if doc.text.starts_with(key) {
                // if hit starts with the key, we don't change the score
                1.0
            } else {
                Self::create_coefficient(doc, matched_tokens, prefix_token, blend)
            };
            if weight == 0 {
                weight = 1;
            }
            const LINEAR_COEF: f64 = 0.10;
            if (weight as f64) < 1.0 / LINEAR_COEF && (weight as f64) > -1.0 / LINEAR_COEF {
                weight = (weight as f64 * (1.0 / LINEAR_COEF)) as i64;
            }
            let score = (weight as f64 * coefficient) as i64;
            let highlight_key = if do_highlight {
                Some(self.highlight_text(&doc.text, matched_tokens, prefix_token)?)
            } else {
                None
            };
            let result =
                LookupResult::with_all(doc.text.clone(), highlight_key, score, payload, None);
            bounded_tree_add(&mut results, result, actual_num);
        }
        results.reverse(); // descendingSet()
        Ok(results)
    }

    /// `createCoefficient`: from the position of the first matching word.
    fn create_coefficient(
        doc: &Doc,
        matched_tokens: &HashSet<String>,
        prefix_token: Option<&str>,
        blend: &Blend<'_>,
    ) -> f64 {
        let mut position = i32::MAX;
        for (term, p) in &doc.terms {
            if matched_tokens.contains(term) || prefix_token.is_some_and(|pt| term.starts_with(pt))
            {
                position = position.min(*p);
            }
        }
        blend.coefficient(position)
    }
}

impl Blend<'_> {
    /// `calculateCoefficient(position)`.
    fn coefficient(&self, position: i32) -> f64 {
        match self.blender {
            BlenderType::PositionLinear => 1.0 - 0.10 * f64::from(position),
            // `position + 1` is an `int` addition in Java
            BlenderType::PositionReciprocal => 1.0 / f64::from(position.wrapping_add(1)),
            BlenderType::PositionExponentialReciprocal => {
                1.0 / (f64::from(position) + 1.0).powf(self.exponent)
            }
            BlenderType::Custom => self.custom.map_or(1.0, |f| f(position)),
        }
    }
}

/// `LookUpComparator`: by weight, then key, then payload.
fn compare_results(a: &LookupResult, b: &LookupResult) -> Ordering {
    a.value
        .cmp(&b.value)
        .then_with(|| charsequence_compare(&a.key, &b.key))
        .then_with(|| match (&a.payload, &b.payload) {
            (Some(x), Some(y)) => x.cmp(y),
            _ => Ordering::Equal,
        })
}

/// `boundedTreeAdd`: a `TreeSet` (equal elements collapse) of at most `num`
/// results; a full set gives up its smallest for a bigger newcomer.
fn bounded_tree_add(results: &mut Vec<LookupResult>, result: LookupResult, num: usize) {
    if results.len() >= num {
        match results.first() {
            Some(first) if first.value < result.value => {
                results.remove(0);
            }
            _ => return,
        }
    }
    if let Err(at) = results.binary_search_by(|r| compare_results(r, &result)) {
        results.insert(at, result);
    }
}

impl Lookup for AnalyzingInfixSuggester {
    fn get_count(&self) -> Result<u64> {
        Ok(self.searcher.as_ref().map_or(0, |s| s.docs.len() as u64))
    }

    fn build(&mut self, input: &mut dyn InputIterator) -> Result<()> {
        // a new index replaces the old one
        self.docs.clear();
        self.segments_with_contexts.clear();
        self.segments_with_payloads.clear();
        self.segment = 0;
        self.segment_has_docs = false;
        self.writer_open = true;
        let result = (|| -> Result<()> {
            while let Some(text) = input.next() {
                let payload = if input.has_payloads() {
                    input.payload().map(<[u8]>::to_vec)
                } else {
                    None
                };
                let doc = self.build_document(
                    &text,
                    input.contexts(),
                    input.weight(),
                    payload.as_deref(),
                )?;
                self.push(doc);
            }
            Ok(())
        })();
        if let Err(e) = result {
            self.docs.clear();
            self.writer_open = false;
            return Err(e);
        }
        if self.commit_on_build || self.close_index_writer_on_build {
            self.commit()?;
        }
        self.flush_and_snapshot();
        if self.close_index_writer_on_build {
            self.writer_open = false;
        }
        Ok(())
    }

    fn lookup(
        &self,
        key: &str,
        contexts: Option<&BTreeSet<Vec<u8>>>,
        _only_more_popular: bool,
        num: usize,
    ) -> Result<Vec<LookupResult>> {
        self.lookup_with(key, contexts, num, self.all_terms_required, self.highlight)
    }

    fn store(&self, _output: &mut dyn DataOutput) -> Result<bool> {
        Ok(false)
    }

    fn load(&mut self, _input: &mut dyn DataInput) -> Result<bool> {
        Ok(false)
    }
}

/// `BlendedInfixSuggester`: the hits of an [`AnalyzingInfixSuggester`]
/// (`numFactor` times as many as asked for) re-weighted by the position of
/// the first matching word in the text.
pub struct BlendedInfixSuggester {
    inner: AnalyzingInfixSuggester,
    blender: BlenderType,
    num_factor: usize,
    exponent: f64,
    custom: Option<Box<dyn Fn(i32) -> f64 + Send + Sync>>,
}

impl BlendedInfixSuggester {
    /// Coefficient used for linear blending.
    pub const LINEAR_COEF: f64 = 0.10;
    /// `DEFAULT_NUM_FACTOR`.
    pub const DEFAULT_NUM_FACTOR: usize = 10;

    /// `new BlendedInfixSuggester(dir, analyzer)`.
    pub fn new(analyzer: Arc<Analyzer>) -> Self {
        Self {
            inner: AnalyzingInfixSuggester::new(analyzer),
            blender: BlenderType::PositionLinear,
            num_factor: Self::DEFAULT_NUM_FACTOR,
            exponent: 2.0,
            custom: None,
        }
    }

    /// `new BlendedInfixSuggester(dir, indexAnalyzer, queryAnalyzer,
    /// minPrefixChars, blenderType, numFactor, commitOnBuild)`.
    pub fn with_blender(
        index_analyzer: Arc<Analyzer>,
        query_analyzer: Arc<Analyzer>,
        min_prefix_chars: i32,
        blender: BlenderType,
        num_factor: usize,
        commit_on_build: bool,
    ) -> Result<Self> {
        Self::with_options(
            index_analyzer,
            query_analyzer,
            min_prefix_chars,
            blender,
            num_factor,
            None,
            commit_on_build,
            DEFAULT_ALL_TERMS_REQUIRED,
            DEFAULT_HIGHLIGHT,
        )
    }

    /// The full constructor (`exponent` is only used by
    /// [`BlenderType::PositionExponentialReciprocal`]; `None` is 2).
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        index_analyzer: Arc<Analyzer>,
        query_analyzer: Arc<Analyzer>,
        min_prefix_chars: i32,
        blender: BlenderType,
        num_factor: usize,
        exponent: Option<f64>,
        commit_on_build: bool,
        all_terms_required: bool,
        highlight: bool,
    ) -> Result<Self> {
        Ok(Self {
            inner: AnalyzingInfixSuggester::with_options(
                index_analyzer,
                query_analyzer,
                min_prefix_chars,
                commit_on_build,
                all_terms_required,
                highlight,
                DEFAULT_CLOSE_INDEXWRITER_ON_BUILD,
            )?,
            blender,
            num_factor,
            exponent: exponent.unwrap_or(2.0),
            custom: None,
        })
    }

    /// What a subclass of Java's `BlendedInfixSuggester` does by overriding
    /// `calculateCoefficient`: the coefficient for [`BlenderType::Custom`].
    pub fn with_custom_coefficient(
        mut self,
        coefficient: impl Fn(i32) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.custom = Some(Box::new(coefficient));
        self
    }

    /// The underlying infix suggester: `add`, `update`, `refresh`, `commit`
    /// and `close` are its own.
    pub fn infix(&mut self) -> &mut AnalyzingInfixSuggester {
        &mut self.inner
    }

    fn blend(&self) -> Blend<'_> {
        Blend {
            blender: self.blender,
            num_factor: self.num_factor,
            exponent: self.exponent,
            custom: self.custom.as_deref(),
        }
    }

    /// `lookup(key, contexts, num, allTermsRequired, doHighlight)`.
    pub fn lookup_with(
        &self,
        key: &str,
        contexts: Option<&BTreeSet<Vec<u8>>>,
        num: usize,
        all_terms_required: bool,
        do_highlight: bool,
    ) -> Result<Vec<LookupResult>> {
        let context_query = AnalyzingInfixSuggester::context_query_of_set(contexts);
        self.inner.search(
            key,
            context_query.as_deref(),
            // the number of hits is multiplied once, in the last call of the chain
            num.saturating_mul(self.num_factor),
            all_terms_required,
            do_highlight,
            Some(&self.blend()),
        )
    }

    /// `lookup(key, Map<BytesRef, Occur> contextInfo, num, allTermsRequired,
    /// doHighlight)`, blended like [`Self::lookup_with`].
    pub fn lookup_context_info(
        &self,
        key: &str,
        context_info: &BTreeMap<Vec<u8>, Occur>,
        num: usize,
        all_terms_required: bool,
        do_highlight: bool,
    ) -> Result<Vec<LookupResult>> {
        let context_query = AnalyzingInfixSuggester::context_query_of_info(context_info);
        self.inner.search(
            key,
            context_query.as_deref(),
            num.saturating_mul(self.num_factor),
            all_terms_required,
            do_highlight,
            Some(&self.blend()),
        )
    }
}

impl Lookup for BlendedInfixSuggester {
    fn get_count(&self) -> Result<u64> {
        self.inner.get_count()
    }

    fn build(&mut self, input: &mut dyn InputIterator) -> Result<()> {
        self.inner.build(input)
    }

    fn lookup(
        &self,
        key: &str,
        contexts: Option<&BTreeSet<Vec<u8>>>,
        _only_more_popular: bool,
        num: usize,
    ) -> Result<Vec<LookupResult>> {
        self.lookup_with(
            key,
            contexts,
            num,
            self.inner.all_terms_required,
            self.inner.highlight,
        )
    }

    fn store(&self, _output: &mut dyn DataOutput) -> Result<bool> {
        Ok(false)
    }

    fn load(&mut self, _input: &mut dyn DataInput) -> Result<bool> {
        Ok(false)
    }
}
