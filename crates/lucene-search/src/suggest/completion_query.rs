//! The query side of `org.apache.lucene.search.suggest.document` (M13
//! T13.3d): [`CompletionAnalyzer`] (with `CompletionTokenStream.toAutomaton`),
//! [`CompletionQuery`] and its prefix, regex, fuzzy and context kinds, the
//! [`CompletionWeight`] each becomes (`setNextMatch`, `boost`, `context`),
//! [`BitsProducer`], and [`SuggestIndexSearcher`], which runs a query over
//! every segment through [`super::document::lookup_weighted`].
//!
//! Rust-forced departures: the query and weight hierarchies are the enums
//! [`CompletionQuery`] and [`CompletionWeight`]; a filter's `Bits` is a
//! closure ([`FilterBits`]); the fuzzy `refs` are a `Vec` (only scanned); an
//! unsupported `maxEdits` is an `IllegalArgument` from `suggest` where Java's
//! `null` automaton fails later; a code point `char` cannot hold becomes
//! U+FFFD where `UnicodeUtil.newString` throws; and [`ContextQuery`] keeps its
//! contexts in insertion order rather than `HashMap` order, which the union's
//! `determinize` makes immaterial.

use std::rc::Rc;
use std::sync::Arc;

use lucene_analysis::miscellaneous::{
    ConcatenateGraphFilter, DEFAULT_MAX_GRAPH_EXPANSIONS, SEP_LABEL,
};
use lucene_analysis::Analyzer;
use lucene_codecs::field_infos::IndexOptions;
use lucene_codecs::per_field_postings::COMPLETION_FORMAT_NAME;
use lucene_util::automaton::{
    automata, operations, Automaton, FiniteStringsIterator, LevenshteinAutomata, RegExp,
    Utf32ToUtf8, DEFAULT_DETERMINIZE_WORK_LIMIT,
};

use super::document::{
    lookup_weighted, MatchWeight, SegmentView, SuggestScoreDoc, TopSuggestDocsCollector,
};
use super::suggest_field::{ContextSuggestField, SuggestField, HOLE_CHARACTER};
use crate::directory_reader::{DirectoryReader, SegmentReader};
use crate::{Error, Result};

fn illegal(e: impl std::fmt::Display) -> Error {
    Error::IllegalArgument(e.to_string())
}

/// `CompletionAnalyzer`: an analyzer whose token stream becomes the
/// completion automaton, keeping (or dropping) token separators and
/// position increments.
#[derive(Clone)]
pub struct CompletionAnalyzer {
    analyzer: Arc<Analyzer>,
    preserve_sep: bool,
    preserve_position_increments: bool,
    max_graph_expansions: i32,
}

impl CompletionAnalyzer {
    /// `new CompletionAnalyzer(analyzer)`: separators and position
    /// increments preserved, no limit on graph expansions.
    pub fn new(analyzer: Arc<Analyzer>) -> Self {
        Self::with_options(analyzer, true, true, DEFAULT_MAX_GRAPH_EXPANSIONS)
    }

    /// `new CompletionAnalyzer(analyzer, preserveSep,
    /// preservePositionIncrements, maxGraphExpansions)`.
    pub fn with_options(
        analyzer: Arc<Analyzer>,
        preserve_sep: bool,
        preserve_position_increments: bool,
        max_graph_expansions: i32,
    ) -> Self {
        CompletionAnalyzer {
            analyzer,
            preserve_sep,
            preserve_position_increments,
            max_graph_expansions,
        }
    }

    /// `preserveSep()`.
    pub fn preserve_sep(&self) -> bool {
        self.preserve_sep
    }

    /// `preservePositionIncrements()`.
    pub fn preserve_position_increments(&self) -> bool {
        self.preserve_position_increments
    }

    /// `maxGraphExpansions`.
    pub fn max_graph_expansions(&self) -> i32 {
        self.max_graph_expansions
    }

    /// The wrapped analyzer.
    pub fn analyzer(&self) -> &Arc<Analyzer> {
        &self.analyzer
    }

    /// `((CompletionTokenStream) tokenStream(field, text)).toAutomaton(unicodeAware)`.
    pub fn to_automaton(&self, field: &str, text: &str, unicode_aware: bool) -> Result<Automaton> {
        let input = self.analyzer.token_stream(field, text).map_err(illegal)?;
        let mut stream = ConcatenateGraphFilter::with_options(
            input,
            self.preserve_sep.then_some(SEP_LABEL),
            self.preserve_position_increments,
            self.max_graph_expansions,
        );
        stream.to_automaton_unicode(unicode_aware).map_err(illegal)
    }
}

/// What `BitsProducer.getBits` returns for one segment.
pub enum FilterBits {
    /// `Bits.MatchNoBits`: the segment is skipped.
    MatchNoBits,
    /// Any other `Bits`: whether a segment-local doc is in scope.
    Bits(Box<dyn Fn(i32) -> bool>),
}

/// `org.apache.lucene.search.suggest.BitsProducer`: the documents a
/// completion query is restricted to, per segment.
pub trait BitsProducer: Send + Sync {
    /// `getBits(context)`.
    fn get_bits(&self, segment: &SegmentReader) -> Result<FilterBits>;
}

/// `CompletionQuery.validate`: the term text may not hold the hole or the
/// separator.
fn validate(text: &str) -> Result<()> {
    for c in text.chars() {
        if c == HOLE_CHARACTER {
            return Err(illegal(
                "Term text cannot contain HOLE character U+001E; this character is reserved",
            ));
        }
        if c == SEP_LABEL {
            return Err(illegal(
                "Term text cannot contain unit separator character U+001F; this character is \
                 reserved",
            ));
        }
    }
    Ok(())
}

/// The term and filter every `CompletionQuery` has.
#[derive(Clone)]
struct Base {
    field: String,
    text: String,
    filter: Option<Arc<dyn BitsProducer>>,
}

impl Base {
    fn new(field: &str, text: &str, filter: Option<Arc<dyn BitsProducer>>) -> Result<Self> {
        validate(text)?;
        Ok(Base {
            field: field.to_string(),
            text: text.to_string(),
            filter,
        })
    }
}

/// `PrefixCompletionQuery`: the analyzed term text as a prefix.
#[derive(Clone)]
pub struct PrefixCompletionQuery {
    base: Base,
    analyzer: CompletionAnalyzer,
}

impl PrefixCompletionQuery {
    /// `new PrefixCompletionQuery(analyzer, term, filter)`.
    pub fn new(
        analyzer: CompletionAnalyzer,
        field: &str,
        text: &str,
        filter: Option<Arc<dyn BitsProducer>>,
    ) -> Result<Self> {
        Ok(PrefixCompletionQuery {
            base: Base::new(field, text, filter)?,
            analyzer,
        })
    }

    /// `getAnalyzer()`.
    pub fn analyzer(&self) -> &CompletionAnalyzer {
        &self.analyzer
    }

    fn create_weight(&self) -> Result<CompletionWeight> {
        let automaton = self
            .analyzer
            .to_automaton(&self.base.field, &self.base.text, false)?;
        Ok(CompletionWeight::plain(automaton))
    }
}

/// `RegexCompletionQuery`: the term text as a regular expression prefix.
#[derive(Clone)]
pub struct RegexCompletionQuery {
    base: Base,
    flags: i32,
    determinize_work_limit: i32,
}

impl RegexCompletionQuery {
    /// `new RegexCompletionQuery(term, filter)`: every optional syntax
    /// ([`RegExp::ALL`]), the default determinize work limit.
    pub fn new(field: &str, text: &str, filter: Option<Arc<dyn BitsProducer>>) -> Result<Self> {
        Self::with_flags(
            field,
            text,
            RegExp::ALL,
            DEFAULT_DETERMINIZE_WORK_LIMIT,
            filter,
        )
    }

    /// `new RegexCompletionQuery(term, flags, determinizeWorkLimit, filter)`.
    pub fn with_flags(
        field: &str,
        text: &str,
        flags: i32,
        determinize_work_limit: i32,
        filter: Option<Arc<dyn BitsProducer>>,
    ) -> Result<Self> {
        Ok(RegexCompletionQuery {
            base: Base::new(field, text, filter)?,
            flags,
            determinize_work_limit,
        })
    }

    /// `getFlags()`.
    pub fn flags(&self) -> i32 {
        self.flags
    }

    /// `getDeterminizeWorkLimit()`.
    pub fn determinize_work_limit(&self) -> i32 {
        self.determinize_work_limit
    }

    fn create_weight(&self) -> Result<CompletionWeight> {
        // an empty regex matches nothing, as an empty prefix does
        let automaton = if self.base.text.is_empty() {
            automata::make_empty()
        } else {
            let regexp = RegExp::with_flags(&self.base.text, self.flags, 0).map_err(illegal)?;
            let a = regexp.to_automaton().map_err(illegal)?;
            operations::determinize(&a, self.determinize_work_limit).map_err(illegal)?
        };
        Ok(CompletionWeight::plain(automaton))
    }
}

/// `FuzzyCompletionQuery`: the analyzed term text as a prefix, within an
/// edit distance, boosted by how much of it a match shares.
#[derive(Clone)]
pub struct FuzzyCompletionQuery {
    prefix: PrefixCompletionQuery,
    max_edits: i32,
    transpositions: bool,
    non_fuzzy_prefix: i32,
    min_fuzzy_length: i32,
    unicode_aware: bool,
    determinize_work_limit: i32,
}

impl FuzzyCompletionQuery {
    /// `FuzzyCompletionQuery.DEFAULT_UNICODE_AWARE`.
    pub const DEFAULT_UNICODE_AWARE: bool = false;
    /// `DEFAULT_MIN_FUZZY_LENGTH`.
    pub const DEFAULT_MIN_FUZZY_LENGTH: i32 = 3;
    /// `DEFAULT_NON_FUZZY_PREFIX`.
    pub const DEFAULT_NON_FUZZY_PREFIX: i32 = 1;
    /// `DEFAULT_MAX_EDITS`.
    pub const DEFAULT_MAX_EDITS: i32 = 1;
    /// `DEFAULT_TRANSPOSITIONS`.
    pub const DEFAULT_TRANSPOSITIONS: bool = true;

    /// `new FuzzyCompletionQuery(analyzer, term, filter)` with the defaults.
    pub fn new(
        analyzer: CompletionAnalyzer,
        field: &str,
        text: &str,
        filter: Option<Arc<dyn BitsProducer>>,
    ) -> Result<Self> {
        Self::with_options(
            analyzer,
            field,
            text,
            filter,
            Self::DEFAULT_MAX_EDITS,
            Self::DEFAULT_TRANSPOSITIONS,
            Self::DEFAULT_NON_FUZZY_PREFIX,
            Self::DEFAULT_MIN_FUZZY_LENGTH,
            Self::DEFAULT_UNICODE_AWARE,
            DEFAULT_DETERMINIZE_WORK_LIMIT,
        )
    }

    /// `new FuzzyCompletionQuery(analyzer, term, filter, maxEdits,
    /// transpositions, nonFuzzyPrefix, minFuzzyLength, unicodeAware,
    /// determinizeWorkLimit)`.
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        analyzer: CompletionAnalyzer,
        field: &str,
        text: &str,
        filter: Option<Arc<dyn BitsProducer>>,
        max_edits: i32,
        transpositions: bool,
        non_fuzzy_prefix: i32,
        min_fuzzy_length: i32,
        unicode_aware: bool,
        determinize_work_limit: i32,
    ) -> Result<Self> {
        Ok(FuzzyCompletionQuery {
            prefix: PrefixCompletionQuery::new(analyzer, field, text, filter)?,
            max_edits,
            transpositions,
            non_fuzzy_prefix,
            min_fuzzy_length,
            unicode_aware,
            determinize_work_limit,
        })
    }

    /// `getMaxEdits()`.
    pub fn max_edits(&self) -> i32 {
        self.max_edits
    }

    /// `isTranspositions()`.
    pub fn is_transpositions(&self) -> bool {
        self.transpositions
    }

    /// `getNonFuzzyPrefix()`.
    pub fn non_fuzzy_prefix(&self) -> i32 {
        self.non_fuzzy_prefix
    }

    /// `getMinFuzzyLength()`.
    pub fn min_fuzzy_length(&self) -> i32 {
        self.min_fuzzy_length
    }

    /// `isUnicodeAware()`.
    pub fn is_unicode_aware(&self) -> bool {
        self.unicode_aware
    }

    fn create_weight(&self) -> Result<CompletionWeight> {
        let base = &self.prefix.base;
        let original =
            self.prefix
                .analyzer
                .to_automaton(&base.field, &base.text, self.unicode_aware)?;
        let mut refs = Vec::new();
        let mut automaton = self.to_levenshtein_automata(&original, &mut refs)?;
        if self.unicode_aware {
            let utf8 = Utf32ToUtf8::new().convert(&automaton);
            automaton =
                operations::determinize(&utf8, self.determinize_work_limit).map_err(illegal)?;
        }
        Ok(CompletionWeight::Fuzzy {
            automaton,
            refs,
            current_boost: 0,
        })
    }

    /// `toLevenshteinAutomata`: every finite string of the analyzed prefix
    /// as a Levenshtein automaton past its non-fuzzy prefix (or exactly, when
    /// too short), their union determinized; `refs` gets each string.
    fn to_levenshtein_automata(
        &self,
        automaton: &Automaton,
        refs: &mut Vec<Vec<i32>>,
    ) -> Result<Automaton> {
        let mut subs = Vec::new();
        let mut strings = FiniteStringsIterator::new(automaton);
        while let Some(string) = strings.next_string().map_err(illegal)? {
            if !refs.contains(&string) {
                refs.push(string.clone());
            }
            let len = i32::try_from(string.len()).unwrap_or(i32::MAX);
            if len <= self.non_fuzzy_prefix || len < self.min_fuzzy_length {
                subs.push(automata::make_string_ints(&string));
            } else {
                let split = usize::try_from(self.non_fuzzy_prefix).map_err(illegal)?;
                let (prefix, rest) = string.split_at(split);
                let alpha_max = if self.unicode_aware { 0x10FFFF } else { 255 };
                let lev = LevenshteinAutomata::with_alphabet(
                    rest.to_vec(),
                    alpha_max,
                    self.transpositions,
                )
                .map_err(illegal)?;
                // `UnicodeUtil.newString(ints, offset, nonFuzzyPrefix)`: each
                // int a code point (a byte, when not unicode aware)
                let prefix: String = prefix
                    .iter()
                    .map(|&c| {
                        u32::try_from(c)
                            .ok()
                            .and_then(char::from_u32)
                            .unwrap_or('\u{FFFD}')
                    })
                    .collect();
                let a = lev
                    .to_automaton_with_prefix(self.max_edits, &prefix)
                    .ok_or_else(|| {
                        illegal(format!("maxEdits {} is not supported", self.max_edits))
                    })?;
                subs.push(a);
            }
        }
        Ok(match subs.len() {
            0 => automata::make_empty(),
            1 => subs.swap_remove(0),
            _ => {
                let refs: Vec<&Automaton> = subs.iter().collect();
                operations::determinize(&operations::union(&refs), self.determinize_work_limit)
                    .map_err(illegal)?
            }
        })
    }
}

/// `ContextQuery.ContextMetaData`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ContextMetaData {
    boost: f32,
    exact: bool,
}

/// `ContextQuery`: a wrapped completion query whose suggestions are
/// filtered and boosted by the contexts a [`ContextSuggestField`] indexed.
#[derive(Clone)]
pub struct ContextQuery {
    base: Base,
    inner: Box<CompletionQuery>,
    /// `contexts`, a `HashMap` in Java: insertion order here, a later
    /// `add_context` of the same context replacing the earlier one.
    contexts: Vec<(Vec<i32>, ContextMetaData)>,
    match_all_contexts: bool,
}

impl ContextQuery {
    /// `new ContextQuery(query)`.
    pub fn new(query: CompletionQuery) -> Result<Self> {
        if matches!(query, CompletionQuery::Context(_)) {
            return Err(illegal(
                "'query' parameter must not be of type ContextQuery",
            ));
        }
        Ok(ContextQuery {
            base: query.base().clone(),
            inner: Box::new(query),
            contexts: Vec::new(),
            match_all_contexts: false,
        })
    }

    /// `addContext(context, boost, exact)`.
    pub fn add_context(&mut self, context: &str, boost: f32, exact: bool) -> Result<()> {
        if boost < 0.0 {
            return Err(illegal("'boost' must be >= 0"));
        }
        if let Some((i, c)) = context
            .encode_utf16()
            .enumerate()
            .find(|&(_, c)| u32::from(c) == ContextSuggestField::CONTEXT_SEPARATOR as u32)
        {
            return Err(illegal(format!(
                "Illegal value [{context}] UTF-16 codepoint [0x{c:x}] at position {i} is a \
                 reserved character"
            )));
        }
        let key: Vec<i32> = context.bytes().map(i32::from).collect();
        let meta = ContextMetaData { boost, exact };
        match self.contexts.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1 = meta,
            None => self.contexts.push((key, meta)),
        }
        Ok(())
    }

    /// `addAllContexts()`.
    pub fn add_all_contexts(&mut self) {
        self.match_all_contexts = true;
    }

    fn create_weight(&self) -> Result<CompletionWeight> {
        let inner = self.inner.create_weight()?;
        // an inner automaton matching nothing: no context is worth walking
        if operations::is_empty(inner.automaton()) {
            return Ok(CompletionWeight::plain(automata::make_empty()));
        }
        // with separators preserved the FST holds a SEP_LABEL behind each
        // gap, so the query takes an optional one too
        let optional_sep = operations::optional(&automata::make_char(SEP_LABEL as i32));
        let contexts = self.to_context_automaton();
        let a = operations::concatenate(&[&contexts, &optional_sep, inner.automaton()]);
        let automaton =
            operations::determinize(&a, DEFAULT_DETERMINIZE_WORK_LIMIT).map_err(illegal)?;
        let context_map: Vec<(Vec<i32>, f32)> = self
            .contexts
            .iter()
            .map(|(k, m)| (k.clone(), m.boost))
            .collect();
        let mut context_lengths: Vec<usize> = context_map.iter().map(|(k, _)| k.len()).collect();
        context_lengths.sort_unstable_by(|a, b| b.cmp(a));
        context_lengths.dedup();
        Ok(CompletionWeight::Context {
            automaton,
            inner: Box::new(inner),
            context_map,
            context_lengths,
            current_boost: 0.0,
            current_context: None,
        })
    }

    /// `toContextAutomaton(contexts, matchAllContexts)`.
    fn to_context_automaton(&self) -> Automaton {
        let match_all = operations::repeat(&automata::make_any_string());
        let sep = automata::make_char(ContextSuggestField::CONTEXT_SEPARATOR as i32);
        if self.match_all_contexts || self.contexts.is_empty() {
            return operations::concatenate(&[&match_all, &sep]);
        }
        let list: Vec<Automaton> = self
            .contexts
            .iter()
            .map(|(key, meta)| {
                let context = automata::make_string_ints(key);
                if meta.exact {
                    operations::concatenate(&[&context, &sep])
                } else {
                    operations::concatenate(&[&context, &match_all, &sep])
                }
            })
            .collect();
        let refs: Vec<&Automaton> = list.iter().collect();
        operations::union(&refs)
    }
}

/// `CompletionQuery`: the kinds of completion query.
#[derive(Clone)]
pub enum CompletionQuery {
    /// [`PrefixCompletionQuery`].
    Prefix(PrefixCompletionQuery),
    /// [`RegexCompletionQuery`].
    Regex(RegexCompletionQuery),
    /// [`FuzzyCompletionQuery`].
    Fuzzy(FuzzyCompletionQuery),
    /// [`ContextQuery`].
    Context(ContextQuery),
}

impl CompletionQuery {
    fn base(&self) -> &Base {
        match self {
            CompletionQuery::Prefix(q) => &q.base,
            CompletionQuery::Regex(q) => &q.base,
            CompletionQuery::Fuzzy(q) => &q.prefix.base,
            CompletionQuery::Context(q) => &q.base,
        }
    }

    /// `getField()`.
    pub fn field(&self) -> &str {
        &self.base().field
    }

    /// `getTerm().text()`.
    pub fn text(&self) -> &str {
        &self.base().text
    }

    /// `getFilter()`.
    pub fn filter(&self) -> Option<&Arc<dyn BitsProducer>> {
        self.base().filter.as_ref()
    }

    /// `createWeight`.
    pub fn create_weight(&self) -> Result<CompletionWeight> {
        match self {
            CompletionQuery::Prefix(q) => q.create_weight(),
            CompletionQuery::Regex(q) => q.create_weight(),
            CompletionQuery::Fuzzy(q) => q.create_weight(),
            CompletionQuery::Context(q) => q.create_weight(),
        }
    }

    /// `rewrite(searcher)`: a query over a field that indexed contexts
    /// becomes a [`ContextQuery`] over every context; a [`ContextQuery`]
    /// over a field that did not is refused, as are values of both types in
    /// one field.
    pub fn rewrite(&self, reader: &DirectoryReader) -> Result<CompletionQuery> {
        let mut found: Option<u8> = None;
        for segment in reader.segment_readers() {
            let Some(t) = segment.completion_type(self.field()) else {
                continue;
            };
            match found {
                None => found = Some(t),
                Some(f) if f != t => {
                    return Err(Error::IllegalState(format!(
                        "{} has values of multiple types",
                        self.field()
                    )))
                }
                Some(_) => {}
            }
        }
        match (found, self) {
            (Some(SuggestField::TYPE), CompletionQuery::Context(_)) => {
                Err(Error::IllegalState(format!(
                "ContextQuery can not be executed against a non context-enabled SuggestField: {}",
                self.field()
            )))
            }
            (Some(ContextSuggestField::TYPE), q) if !matches!(q, CompletionQuery::Context(_)) => {
                Ok(CompletionQuery::Context(ContextQuery::new(q.clone())?))
            }
            _ => Ok(self.clone()),
        }
    }
}

/// `CompletionWeight` and its fuzzy and context subclasses: the query
/// automaton, plus what a matched path prefix makes of the boost and the
/// context (`setNextMatch`, `boost()`, `context()`).
#[derive(Clone, Debug)]
pub enum CompletionWeight {
    /// `CompletionWeight`: boost 0, no context.
    Plain {
        /// `getAutomaton()`.
        automaton: Automaton,
    },
    /// `FuzzyCompletionQuery.FuzzyCompletionWeight`.
    Fuzzy {
        /// `getAutomaton()`.
        automaton: Automaton,
        /// Every finite string of the analyzed prefix.
        refs: Vec<Vec<i32>>,
        /// The longest common prefix of the last match with a ref.
        current_boost: usize,
    },
    /// `ContextQuery.ContextCompletionWeight`.
    Context {
        /// `getAutomaton()`.
        automaton: Automaton,
        /// The wrapped query's weight.
        inner: Box<CompletionWeight>,
        /// Each context's bytes (as ints) and boost.
        context_map: Vec<(Vec<i32>, f32)>,
        /// The contexts' distinct lengths, longest first.
        context_lengths: Vec<usize>,
        /// `currentBoost`.
        current_boost: f32,
        /// `currentContext`.
        current_context: Option<Rc<str>>,
    },
}

impl CompletionWeight {
    /// `new CompletionWeight(query, automaton)`.
    pub fn plain(automaton: Automaton) -> Self {
        CompletionWeight::Plain { automaton }
    }

    /// `getAutomaton()`.
    pub fn automaton(&self) -> &Automaton {
        match self {
            CompletionWeight::Plain { automaton }
            | CompletionWeight::Fuzzy { automaton, .. }
            | CompletionWeight::Context { automaton, .. } => automaton,
        }
    }
}

impl MatchWeight for CompletionWeight {
    /// `setNextMatch(pathPrefix)`.
    fn set_next_match(&mut self, path_prefix: &[i32]) {
        match self {
            CompletionWeight::Plain { .. } => {}
            CompletionWeight::Fuzzy {
                refs,
                current_boost,
                ..
            } => {
                // the last letter of an exact match's prefix never gets here,
                // so an exact match and one edited at its end boost the same
                *current_boost = refs
                    .iter()
                    .map(|r| {
                        r.iter()
                            .zip(path_prefix)
                            .take_while(|(a, b)| a == b)
                            .count()
                    })
                    .max()
                    .unwrap_or(0);
            }
            CompletionWeight::Context {
                inner,
                context_map,
                context_lengths,
                current_boost,
                current_context,
                ..
            } => {
                // the longest defined context the path starts with
                for &length in context_lengths.iter() {
                    if length > path_prefix.len() {
                        continue;
                    }
                    if let Some((_, boost)) = context_map
                        .iter()
                        .find(|(k, _)| k[..] == path_prefix[..length])
                    {
                        *current_boost = *boost;
                        set_inner_weight(inner, current_context, path_prefix, length);
                        return;
                    }
                }
                // an unknown context
                *current_boost = 0.0;
                set_inner_weight(inner, current_context, path_prefix, 0);
            }
        }
    }

    /// `boost()`.
    fn boost(&self) -> f32 {
        match self {
            CompletionWeight::Plain { .. } => 0.0,
            CompletionWeight::Fuzzy { current_boost, .. } => *current_boost as f32,
            CompletionWeight::Context {
                inner,
                current_boost,
                ..
            } => current_boost + inner.boost(),
        }
    }

    /// `context()`.
    fn context(&self) -> Option<Rc<str>> {
        match self {
            CompletionWeight::Context {
                current_context, ..
            } => current_context.clone(),
            _ => None,
        }
    }
}

/// `ContextCompletionWeight.setInnerWeight`: the context is what precedes
/// the first context separator at or after `offset`, and the inner weight
/// sees what follows it (past one `SEP_LABEL`). With no separator, neither
/// changes, as in Java.
fn set_inner_weight(
    inner: &mut CompletionWeight,
    current_context: &mut Option<Rc<str>>,
    path: &[i32],
    offset: usize,
) {
    let separator = ContextSuggestField::CONTEXT_SEPARATOR as i32;
    let Some(i) = path
        .iter()
        .enumerate()
        .skip(offset)
        .find(|&(_, &c)| c == separator)
        .map(|(i, _)| i)
    else {
        return;
    };
    *current_context = if i > 0 {
        // `Util.toBytesRef`: each int is a byte
        let bytes: Vec<u8> = path[..i].iter().map(|&c| c as u8).collect();
        Some(Rc::from(String::from_utf8_lossy(&bytes).as_ref()))
    } else {
        None
    };
    let mut start = i + 1;
    if path.get(start) == Some(&(SEP_LABEL as i32)) {
        start += 1;
    }
    inner.set_next_match(path.get(start..).unwrap_or_default());
}

/// `SuggestIndexSearcher`: completion queries over a reader's segments.
pub struct SuggestIndexSearcher<'a> {
    reader: &'a DirectoryReader,
}

impl<'a> SuggestIndexSearcher<'a> {
    /// `new SuggestIndexSearcher(reader)`.
    pub fn new(reader: &'a DirectoryReader) -> Self {
        SuggestIndexSearcher { reader }
    }

    /// `suggest(query, n, skipDuplicates)`: the top `n` hits.
    pub fn suggest(
        &self,
        query: &CompletionQuery,
        n: usize,
        skip_duplicates: bool,
    ) -> Result<Vec<SuggestScoreDoc>> {
        let mut collector = TopSuggestDocsCollector::new(n, skip_duplicates)?;
        self.suggest_into(query, &mut collector)?;
        Ok(collector.get())
    }

    /// `suggest(query, collector)`: every segment's hits into `collector`.
    pub fn suggest_into(
        &self,
        query: &CompletionQuery,
        collector: &mut TopSuggestDocsCollector,
    ) -> Result<()> {
        let query = query.rewrite(self.reader)?;
        let mut weight = query.create_weight()?;
        // the automaton is fixed; only the weight's match state changes
        let automaton = weight.automaton().clone();
        for segment in self.reader.segment_readers() {
            let Some(suggester) = completion_suggester(segment, query.field())? else {
                continue;
            };
            let filter = match query.filter() {
                Some(producer) => match producer.get_bits(segment)? {
                    FilterBits::MatchNoBits => continue,
                    FilterBits::Bits(bits) => Some(bits),
                },
                None => None,
            };
            let live = segment.live_docs();
            let accept = |doc: i32| {
                filter.as_ref().is_none_or(|f| f(doc))
                    && live.is_none_or(|l| usize::try_from(doc).is_ok_and(|d| l.get(d)))
            };
            let view = SegmentView {
                num_docs: segment.num_docs(),
                max_doc: segment.max_doc,
                filtered: query.filter().is_some(),
                accept: &accept,
            };
            collector.set_next_reader(segment.doc_base);
            lookup_weighted(&suggester, &automaton, &mut weight, &view, collector)?;
            collector.finish();
        }
        Ok(())
    }
}

/// `CompletionWeight.bulkScorer`'s look at the field: no suggester for a
/// segment without the field's terms (or without an FST), and an error for
/// a field whose terms are not a completion field's.
fn completion_suggester(
    segment: &SegmentReader,
    field: &str,
) -> Result<Option<std::sync::Arc<lucene_codecs::completion::NrtSuggester>>> {
    if segment.completion_type(field).is_none() {
        let indexed_elsewhere = segment
            .field_infos()
            .field_by_name(field)
            .is_some_and(|info| {
                info.index_options != IndexOptions::None
                    && !info.attributes.iter().any(|(k, v)| {
                        k == "PerFieldPostingsFormat.format" && v == COMPLETION_FORMAT_NAME
                    })
            });
        if indexed_elsewhere {
            return Err(illegal(format!("{field} is not a SuggestField")));
        }
        return Ok(None);
    }
    Ok(segment.suggester(field)?)
}
