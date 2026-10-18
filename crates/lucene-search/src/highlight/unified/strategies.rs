//! `UHComponents` and the `FieldOffsetStrategy` family: where a field's
//! highlightable offsets come from (postings, term vectors, re-analysis of the
//! text, or nowhere) and what they are for the query.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use lucene_analysis::{Analyzer, TokenStream};
use lucene_index::document::{FieldToken, FieldTokens};

use super::super::{Error, Result};
use super::matches_enum::{query_to_term, OfMatchesWithSubs};
use super::passage::{MatchEntry, MultiOffsets, OfMatches, OfPostings, OffsetsEnum};
use crate::highlighter::OffsetSource;
use crate::matches::BoxMatches;
use crate::memory::MemoryIndex;
use crate::query::{Clause, MatchAllDocsQuery, MatchNoDocsQuery};
use crate::query_visitor::{self, Occur, QueryVisitor, SubVisitor, Term, TermMatcher};
use crate::reader::{LeafReader, PostingsFlags, Terms};

/// `UnifiedHighlighter.MULTIVAL_SEP_CHAR`.
pub const MULTIVAL_SEP_CHAR: u16 = 0;

/// `Predicate<String>` over field names.
pub type FieldMatcher = Arc<dyn Fn(&str) -> bool + Send + Sync>;

/// `UnifiedHighlighter.HighlightFlag`, as the four switches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Flags {
    /// `MULTI_TERM_QUERY`.
    pub multi_term_query: bool,
    /// `PHRASES`.
    pub phrases: bool,
    /// `PASSAGE_RELEVANCY_OVER_SPEED`.
    pub passage_relevancy_over_speed: bool,
    /// `WEIGHT_MATCHES`.
    pub weight_matches: bool,
}

impl Flags {
    /// `evaluateFlags(...)`: `WEIGHT_MATCHES` only stands when the other
    /// three do and the caller did not opt out.
    pub fn evaluate(
        multi_term_query: bool,
        phrases: bool,
        passage_relevancy_over_speed: bool,
        weight_matches: bool,
    ) -> Self {
        Flags {
            multi_term_query,
            phrases,
            passage_relevancy_over_speed,
            weight_matches: multi_term_query
                && phrases
                && passage_relevancy_over_speed
                && weight_matches,
        }
    }
}

/// A reduced `PhraseHelper`: the phrase and span clauses of the query on
/// matching fields (highlighted only where they match as a whole) and the
/// terms that need no such care. Span clauses holding a multi-term query are
/// not looked into (Java ignores a query it must rewrite).
#[derive(Clone, Default)]
pub struct PhraseHelper {
    /// `positionInsensitiveTerms`, sorted.
    insensitive: Vec<Vec<u8>>,
    /// `spanQueries`.
    spans: Vec<Clause>,
}

impl PhraseHelper {
    /// The helper for `query`.
    pub fn new(query: &Clause, field_matcher: &dyn Fn(&str) -> bool) -> Self {
        fn walk(
            c: &Clause,
            fm: &dyn Fn(&str) -> bool,
            ins: &mut BTreeSet<Vec<u8>>,
            spans: &mut Vec<Clause>,
        ) {
            match c {
                Clause::Boolean(b) => {
                    for x in b.must.iter().chain(&b.filter).chain(&b.should) {
                        walk(x, fm, ins, spans);
                    }
                }
                Clause::DisjunctionMax(d) => {
                    for x in &d.disjuncts {
                        walk(x, fm, ins, spans);
                    }
                }
                Clause::ConstantScore(q) => walk(&q.inner, fm, ins, spans),
                Clause::Boost(q) => walk(&q.inner, fm, ins, spans),
                Clause::Term(t) => {
                    if fm(&t.field) {
                        ins.insert(t.term.clone());
                    }
                }
                Clause::Phrase(p) => {
                    if fm(&p.field) {
                        spans.push(c.clone());
                    }
                }
                Clause::Span(_) if clause_fields(c).iter().all(|f| fm(f)) => {
                    spans.push(c.clone());
                }
                _ => {}
            }
        }
        let (mut ins, mut spans) = (BTreeSet::new(), Vec::new());
        walk(query, field_matcher, &mut ins, &mut spans);
        PhraseHelper {
            insensitive: ins.into_iter().collect(),
            spans,
        }
    }

    /// `hasPositionSensitivity`.
    pub fn has_position_sensitivity(&self) -> bool {
        !self.spans.is_empty()
    }
}

/// The fields a clause refers to.
fn clause_fields(c: &Clause) -> BTreeSet<String> {
    struct Rec(BTreeSet<String>);
    impl QueryVisitor for Rec {
        fn accept_field(&self, _field: &str) -> bool {
            true
        }
        fn consume_terms(&mut self, _q: &Clause, terms: &[Term]) {
            self.0.extend(terms.iter().map(|t| t.field.clone()));
        }
        fn consume_terms_matching(&mut self, _q: &Clause, field: &str, _m: &dyn TermMatcher) {
            self.0.insert(field.to_string());
        }
        fn visit_leaf(&mut self, _q: &Clause) {}
    }
    let mut r = Rec(BTreeSet::new());
    query_visitor::visit(c, &mut r);
    r.0
}

/// `c` with its leaves replaced by `leaf(clause, required)`, where `required`
/// is whether the leaf must match for the whole to.
fn map_leaves(c: &Clause, required: bool, leaf: &dyn Fn(&Clause, bool) -> Clause) -> Clause {
    match c {
        Clause::Boolean(b) => {
            let mut out = b.clone();
            out.must = b.must.iter().map(|x| map_leaves(x, true, leaf)).collect();
            out.filter = b.filter.iter().map(|x| map_leaves(x, true, leaf)).collect();
            out.should = b
                .should
                .iter()
                .map(|x| map_leaves(x, false, leaf))
                .collect();
            out.must_not = b
                .must_not
                .iter()
                .map(|x| map_leaves(x, false, leaf))
                .collect();
            Clause::Boolean(out)
        }
        Clause::DisjunctionMax(d) => {
            let mut out = d.clone();
            out.disjuncts = d
                .disjuncts
                .iter()
                .map(|x| map_leaves(x, false, leaf))
                .collect();
            Clause::DisjunctionMax(out)
        }
        Clause::ConstantScore(q) => {
            let mut out = q.clone();
            out.inner = Box::new(map_leaves(&q.inner, required, leaf));
            Clause::ConstantScore(out)
        }
        Clause::Boost(q) => {
            let mut out = q.clone();
            out.inner = Box::new(map_leaves(&q.inner, required, leaf));
            Clause::Boost(out)
        }
        other => leaf(other, required),
    }
}

/// What the `FilterLeafReader` of the weight-matches path (and of
/// `PhraseHelper`) does: every field the matcher accepts is read as `to`.
fn retarget(c: &Clause, fm: &dyn Fn(&str) -> bool, to: &str) -> Clause {
    map_leaves(c, true, &|leaf, _| match leaf {
        Clause::Term(t) if fm(&t.field) => {
            let mut t = t.clone();
            t.field = to.to_string();
            Clause::Term(t)
        }
        Clause::Phrase(p) if fm(&p.field) => {
            let mut p = p.clone();
            p.field = to.to_string();
            Clause::Phrase(p)
        }
        Clause::Wildcard(q) if fm(&q.field) => {
            let mut q = q.clone();
            q.field = to.to_string();
            Clause::Wildcard(q)
        }
        Clause::Prefix(q) if fm(&q.field) => {
            let mut q = q.clone();
            q.field = to.to_string();
            Clause::Prefix(q)
        }
        Clause::Fuzzy(q) if fm(&q.field) => {
            let mut q = q.clone();
            q.field = to.to_string();
            Clause::Fuzzy(q)
        }
        Clause::Regexp(q) if fm(&q.field) => {
            let mut q = q.clone();
            q.field = to.to_string();
            Clause::Regexp(q)
        }
        other => other.clone(),
    })
}

/// One term's occurrences in the document: its frequency and their offsets.
type Postings = (i32, Vec<(i32, i32)>);

/// `Weight.matches` of a query, over the index or over a memory index.
type MatchesFn<'a> = &'a dyn Fn(&Clause) -> Result<Option<BoxMatches>>;

/// `OffsetSpanCollector`: the offsets of the terms of every match.
fn collect_leaf_offsets(
    it: &mut crate::matches::BoxMatchesIterator,
    out: &mut BTreeMap<Vec<u8>, Vec<(i32, i32)>>,
) -> Result<()> {
    while it.next()? {
        match it.sub_matches()? {
            Some(mut subs) => collect_leaf_offsets(&mut subs, out)?,
            None => out
                .entry(query_to_term(it.query()))
                .or_default()
                .push((it.start_offset(), it.end_offset())),
        }
    }
    Ok(())
}

/// `UHComponents`.
#[derive(Clone)]
pub struct Components {
    /// The field being highlighted.
    pub field: String,
    /// Which query fields count as this one.
    pub field_matcher: FieldMatcher,
    /// The query, rewritten.
    pub query: Clause,
    /// The sorted terms of the query in matching fields; `None` is "unknown".
    pub terms: Option<Vec<Vec<u8>>>,
    /// The labels of the query's multi-term automata, in visiting order;
    /// `None` is "unknown".
    pub automata: Option<Vec<String>>,
    /// Whether `AutomataCollector` looks inside span queries.
    pub look_in_span: bool,
    /// `hasUnrecognizedQueryPart`.
    pub has_unrecognized_query_part: bool,
    /// The `PhraseHelper` (empty without `PHRASES`).
    pub phrase: PhraseHelper,
    /// The flags.
    pub flags: Flags,
}

fn is_span(c: &Clause) -> bool {
    matches!(c, Clause::Span(_))
        || matches!(
            c,
            Clause::Extended(e) if matches!(e.as_ref(), crate::extended_query::ExtendedQuery::Span(_))
        )
}

/// `MultiTermHighlighting.AutomataCollector`: every multi-term clause on a
/// matching field, with the matcher its terms are tested by.
pub struct AutomataCollector<'a> {
    look_in_span: bool,
    field_matcher: &'a dyn Fn(&str) -> bool,
    on_automaton: &'a mut dyn FnMut(&str, &dyn TermMatcher),
}

impl QueryVisitor for AutomataCollector<'_> {
    fn accept_field(&self, field: &str) -> bool {
        (self.field_matcher)(field)
    }
    fn sub_visitor(&mut self, occur: Occur, parent: &Clause) -> SubVisitor {
        if !self.look_in_span && is_span(parent) {
            return SubVisitor::Empty;
        }
        // `super.getSubVisitor`: this, but not for a `MUST_NOT` clause
        if occur == Occur::MustNot {
            SubVisitor::Empty
        } else {
            SubVisitor::Same
        }
    }
    fn consume_terms_matching(&mut self, query: &Clause, _field: &str, matcher: &dyn TermMatcher) {
        (self.on_automaton)(&crate::explain::describe_clause(query), matcher);
    }
}

/// `MultiTermHighlighting.extractAutomata` -- each automaton handed to
/// `on_automaton` with its label (`query.toString()`).
pub fn for_each_automaton(
    query: &Clause,
    field_matcher: &dyn Fn(&str) -> bool,
    look_in_span: bool,
    on_automaton: &mut dyn FnMut(&str, &dyn TermMatcher),
) {
    let mut collector = AutomataCollector {
        look_in_span,
        field_matcher,
        on_automaton,
    };
    query_visitor::visit(query, &mut collector);
}

/// `UnifiedHighlighter.hasUnrecognizedQuery`.
pub fn has_unrecognized_query(field_matcher: &dyn Fn(&str) -> bool, query: &Clause) -> bool {
    struct V<'a> {
        unknown: bool,
        field_matcher: &'a dyn Fn(&str) -> bool,
    }
    impl QueryVisitor for V<'_> {
        fn accept_field(&self, field: &str) -> bool {
            // checking `unknown` is a trick to exit early
            !self.unknown && (self.field_matcher)(field)
        }
        fn visit_leaf(&mut self, query: &Clause) {
            // `canExtractAutomataFromLeafQuery`: an `AutomatonQuery` or a
            // `FuzzyQuery`
            let extractable = matches!(
                query,
                Clause::Prefix(_) | Clause::Wildcard(_) | Clause::Regexp(_) | Clause::Fuzzy(_)
            );
            if !extractable && !matches!(query, Clause::MatchAllDocs(_) | Clause::MatchNoDocs(_)) {
                self.unknown = true;
            }
        }
    }
    let mut v = V {
        unknown: false,
        field_matcher,
    };
    query_visitor::visit(query, &mut v);
    v.unknown
}

impl Components {
    /// `getHighlightComponents(field, query, allTerms)` (no `PhraseHelper`).
    pub fn new(
        field: &str,
        field_matcher: FieldMatcher,
        query: &Clause,
        all_terms: &BTreeSet<Term>,
        flags: Flags,
    ) -> Self {
        let has_unrecognized = has_unrecognized_query(&*field_matcher, query);
        let mut terms = None;
        let mut automata = None;
        // WEIGHT_MATCHES with an unrecognized part needs neither
        if !flags.weight_matches || !has_unrecognized {
            // strip the redundant field and sort the remaining terms
            let set: BTreeSet<Vec<u8>> = all_terms
                .iter()
                .filter(|t| field_matcher(&t.field))
                .map(|t| t.bytes.clone())
                .collect();
            terms = Some(set.into_iter().collect());
            let mut labels = Vec::new();
            if flags.multi_term_query {
                for_each_automaton(
                    query,
                    &*field_matcher,
                    Self::look_in_span(flags),
                    &mut |l, _| labels.push(l.to_string()),
                );
            }
            automata = Some(labels);
        }
        let phrase = if flags.phrases {
            PhraseHelper::new(query, &*field_matcher)
        } else {
            PhraseHelper::default()
        };
        Components {
            field: field.to_string(),
            field_matcher,
            query: query.clone(),
            terms,
            automata,
            look_in_span: Self::look_in_span(flags),
            has_unrecognized_query_part: has_unrecognized,
            phrase,
            flags,
        }
    }

    /// Whether automata are looked for inside span queries, rather than
    /// left to the `PhraseHelper`.
    fn look_in_span(flags: Flags) -> bool {
        !flags.phrases || flags.weight_matches
    }

    /// `getOptimizedOffsetSource(components)` over the field's own source.
    pub fn optimized_offset_source(&self, source: OffsetSource) -> OffsetSource {
        // null automata means unknown, so assume a possibility
        let mtq_or_rewrite = self.automata.as_ref().is_none_or(|a| !a.is_empty())
            || self.has_unrecognized_query_part;
        // null terms means unknown, so assume something to highlight
        let has_terms = self.terms.as_ref().is_none_or(|t| !t.is_empty());
        crate::highlighter::optimize_offset_source(source, mtq_or_rewrite, has_terms)
    }
}

// ---------------------------------------------------------------------------
// What a strategy reads
// ---------------------------------------------------------------------------

/// The leaf a document is highlighted in: its reader, the document in it, and
/// how to compute the query's `Matches` over the index itself.
pub struct LeafCtx<'a> {
    /// The segment's reader.
    pub reader: &'a dyn LeafReader,
    /// The document, leaf-local.
    pub doc: i32,
    /// `Weight.matches(leaf, doc)` of a query over the index.
    pub matches: &'a dyn Fn(&Clause) -> crate::Result<Option<BoxMatches>>,
}

/// A strategy: an offset source with the components it works from.
#[derive(Clone)]
pub struct Strategy {
    /// What `getOffsetSource()` reports.
    pub source: OffsetSource,
    /// The components.
    pub components: Components,
    /// The index analyzer, for `ANALYSIS`.
    pub analyzer: Option<Arc<Analyzer>>,
}

impl Strategy {
    /// `getOffsetStrategy(offsetSource, components)`.
    pub fn new(
        source: OffsetSource,
        components: Components,
        analyzer: Option<Arc<Analyzer>>,
    ) -> Result<Self> {
        if source == OffsetSource::Analysis {
            let analyzer = analyzer
                .as_ref()
                .ok_or_else(|| Error::IllegalArgument("an analyzer is required".into()))?;
            // 1 is the default; it is rarely changed
            if analyzer.offset_gap_for_field(&components.field) != 1 {
                return Err(Error::IllegalArgument(format!(
                    "offset gap of the provided analyzer should be 1 (field {})",
                    components.field
                )));
            }
        }
        Ok(Strategy {
            source,
            components,
            analyzer,
        })
    }

    /// `getOffsetsEnum(reader, docId, content)`; `None` is
    /// `OffsetsEnum.EMPTY`. `ctx` is `None` for a highlight without a
    /// searcher.
    pub fn offsets_enum(
        &self,
        ctx: Option<&LeafCtx<'_>>,
        content: &[u16],
    ) -> Result<Option<Box<dyn OffsetsEnum>>> {
        match self.source {
            OffsetSource::NoneNeeded => Ok(None),
            OffsetSource::Postings | OffsetSource::PostingsWithTermVectors => {
                let ctx = ctx.ok_or_else(|| {
                    Error::IllegalArgument("a postings strategy needs a reader".into())
                })?;
                let own = |q: &Clause| -> Result<Option<BoxMatches>> {
                    let q = retarget(q, &*self.components.field_matcher, &self.components.field);
                    (ctx.matches)(&q).map_err(Into::into)
                };
                self.enums_from_reader(ctx.reader, ctx.doc, &own)
            }
            OffsetSource::TermVectors => {
                let ctx = ctx.ok_or_else(|| {
                    Error::IllegalArgument("a term vector strategy needs a reader".into())
                })?;
                let field = &self.components.field;
                let Some(fi) = ctx.reader.field_infos().field_by_name(field) else {
                    return Ok(None);
                };
                let number = fi.number;
                let Some(doc) = ctx.reader.term_vectors(ctx.doc)? else {
                    return Ok(None);
                };
                let Some(vector) = doc.fields.iter().find(|f| f.field_number == number) else {
                    return Ok(None);
                };
                let mut stream = super::super::TokenStreamFromTermVector::new(vector, -1)?;
                let tokens = drain_tokens(&mut stream)?;
                let found = self.enums_from_memory_index(&tokens, Some(ctx), false)?;
                if vector.has_offsets {
                    return Ok(found);
                }
                // `TermVectorLeafReader` postings report offset -1 without
                // offsets, which `FieldHighlighter` refuses at the first match
                Ok(found.map(|e| Box::new(NoOffsets(e)) as Box<dyn OffsetsEnum>))
            }
            OffsetSource::Analysis => {
                let analyzer = self.analyzer.as_ref().expect("checked at construction");
                let tokens = analyze_content(analyzer, &self.components.field, content)?;
                let c = &self.components;
                let position_sensitive = c.phrase.has_position_sensitivity();
                if !position_sensitive
                    && !c.flags.passage_relevancy_over_speed
                    && !c.flags.weight_matches
                {
                    // pure term filtering: no memory index
                    Ok(self.token_stream_enum(&tokens))
                } else {
                    self.enums_from_memory_index(&tokens, ctx, true)
                }
            }
        }
    }

    /// `TokenStreamOffsetStrategy`: the tokens equal to a query term or
    /// matched by one of its automata, with the claimed frequency
    /// `Integer.MAX_VALUE`.
    fn token_stream_enum(&self, tokens: &FieldTokens) -> Option<Box<dyn OffsetsEnum>> {
        let c = &self.components;
        let terms = c.terms.clone().unwrap_or_default();
        let mut automata: Vec<(String, Vec<bool>)> = Vec::new();
        if c.flags.multi_term_query {
            let distinct: Vec<&[u8]> = tokens.tokens.iter().map(|t| t.term.as_slice()).collect();
            for_each_automaton(
                &c.query,
                &*c.field_matcher,
                c.look_in_span,
                &mut |label, m| {
                    automata.push((
                        label.to_string(),
                        distinct.iter().map(|t| m.matches(t)).collect(),
                    ));
                },
            );
        }
        let mut entries = Vec::new();
        for (i, t) in tokens.tokens.iter().enumerate() {
            // matchers: the terms first, then the automata
            let hit = terms
                .iter()
                .position(|q| *q == t.term)
                .map(|k| (k, terms[k].clone()))
                .or_else(|| {
                    automata.iter().position(|(_, hits)| hits[i]).map(|k| {
                        // each matcher is its own term, as Java's identity
                        // `toString()` makes it
                        (
                            terms.len() + k,
                            format!("{}#{}", k, automata[k].0).into_bytes(),
                        )
                    })
                });
            if let Some((_, term)) = hit {
                entries.push(MatchEntry {
                    term,
                    start: t.start_offset,
                    end: t.end_offset,
                    freq: i32::MAX,
                });
            }
        }
        Some(Box::new(OfMatches::new(entries)))
    }

    /// `MemoryIndexOffsetStrategy`'s pre-filter: only the tokens that equal
    /// a query term or match an automaton reach the memory index, the
    /// positions of the dropped ones still counting. (Java filters the
    /// `ANALYSIS` tokens only, and only when it knows both sets.)
    fn keep_words(&self, tokens: &FieldTokens) -> FieldTokens {
        let c = &self.components;
        let (Some(terms), Some(_)) = (&c.terms, &c.automata) else {
            return tokens.clone();
        };
        let mut hit = vec![false; tokens.tokens.len()];
        if c.flags.multi_term_query {
            for_each_automaton(&c.query, &*c.field_matcher, c.look_in_span, &mut |_, m| {
                for (h, t) in hit.iter_mut().zip(&tokens.tokens) {
                    *h = *h || m.matches(&t.term);
                }
            });
        }
        let mut kept = Vec::new();
        let mut carried = 0;
        for (t, h) in tokens.tokens.iter().zip(hit) {
            if h || terms.contains(&t.term) {
                let mut t = t.clone();
                t.position_increment += carried;
                carried = 0;
                kept.push(t);
            } else {
                carried += t.position_increment;
            }
        }
        FieldTokens {
            tokens: kept,
            final_position_increment: tokens.final_position_increment,
            final_offset: tokens.final_offset,
            end_attributes: None,
        }
    }

    /// `MemoryIndexOffsetStrategy`: the tokens as a one-document index that
    /// the reader-based extraction then reads.
    fn enums_from_memory_index(
        &self,
        tokens: &FieldTokens,
        ctx: Option<&LeafCtx<'_>>,
        prefilter: bool,
    ) -> Result<Option<Box<dyn OffsetsEnum>>> {
        let c = &self.components;
        let filtered;
        let tokens = if prefilter {
            filtered = self.keep_words(tokens);
            &filtered
        } else {
            tokens
        };
        let mut index = MemoryIndex::with_options(true, false);
        index
            .add_field_tokens(&c.field, tokens, 0, 1)
            .map_err(|e| Error::IllegalArgument(e.to_string()))?;
        let reader = index
            .create_reader()
            .map_err(|e| Error::IllegalArgument(e.to_string()))?;
        let overlay = ctx.is_some();
        let own = |q: &Clause| -> Result<Option<BoxMatches>> {
            let q = retarget(q, &*c.field_matcher, &c.field);
            let q = memory_index_query(&q, &|f| f == c.field, overlay).rewrite();
            reader
                .with_searcher(|s| crate::matches::matches(s, &q, 0))
                .map_err(|e| Error::IllegalArgument(e.to_string()))?
                .map_err(Into::into)
        };
        if c.flags.weight_matches {
            // the in-memory searcher answers; the codec segment the
            // reader-level APIs need is written only on the classic path
            return self.enums_from_matches(own(&c.query)?);
        }
        let segments = reader
            .directory_reader()
            .map_err(|e| Error::IllegalArgument(e.to_string()))?;
        let leaf: &dyn LeafReader = &segments.segment_readers()[0];
        self.enums_from_reader(leaf, 0, &own)
    }

    /// `createOffsetsEnumFromReader(leafReader, doc)`.
    fn enums_from_reader(
        &self,
        leaf: &dyn LeafReader,
        doc: i32,
        matches_of: MatchesFn<'_>,
    ) -> Result<Option<Box<dyn OffsetsEnum>>> {
        let c = &self.components;
        let Some(terms_index) = leaf.terms(&c.field)? else {
            return Ok(None);
        };
        let mut enums: Vec<Box<dyn OffsetsEnum>> = Vec::new();
        if c.flags.weight_matches {
            let matches = matches_of(&c.query)?;
            if let Some(matches) = matches {
                enums.extend(self.matches_enums(matches)?);
            }
        } else {
            // classic approach
            let position_sensitive = c.phrase.has_position_sensitivity();
            let terms = if position_sensitive {
                c.phrase.insensitive.clone()
            } else {
                c.terms.clone().unwrap_or_default()
            };
            if !terms.is_empty() {
                self.enums_for_terms(&terms, &*terms_index, doc, &mut enums)?;
            }
            if position_sensitive {
                self.enums_for_spans(matches_of, &mut enums)?;
            }
            if c.automata.as_ref().is_some_and(|a| !a.is_empty()) {
                self.enums_for_automata(&*terms_index, doc, &mut enums)?;
            }
        }
        collapse(enums)
    }

    fn enums_from_matches(
        &self,
        matches: Option<BoxMatches>,
    ) -> Result<Option<Box<dyn OffsetsEnum>>> {
        match matches {
            None => Ok(None),
            Some(m) => collapse(self.matches_enums(m)?),
        }
    }

    /// One enum per accepted field of the matches.
    fn matches_enums(&self, matches: BoxMatches) -> Result<Vec<Box<dyn OffsetsEnum>>> {
        let mut out: Vec<Box<dyn OffsetsEnum>> = Vec::new();
        for field in matches.fields() {
            if (self.components.field_matcher)(&field) {
                if let Some(it) = matches.get_matches(&field)? {
                    out.push(Box::new(OfMatchesWithSubs::new(it)));
                }
            }
        }
        Ok(out)
    }

    /// `PhraseHelper.createOffsetsEnumsForSpans`.
    fn enums_for_spans(
        &self,
        matches_of: MatchesFn<'_>,
        results: &mut Vec<Box<dyn OffsetsEnum>>,
    ) -> Result<()> {
        let c = &self.components;
        let mut by_term: BTreeMap<Vec<u8>, Vec<(i32, i32)>> = BTreeMap::new();
        for span in &c.phrase.spans {
            let Some(m) = matches_of(span)? else {
                continue;
            };
            let Some(mut it) = m.get_matches(&c.field)? else {
                continue;
            };
            let Clause::Phrase(phrase) = span else {
                collect_leaf_offsets(&mut it, &mut by_term)?;
                continue;
            };
            // a phrase match reports no terms: take each term's own offsets
            // that fall inside a match
            let mut ranges = Vec::new();
            while it.next()? {
                ranges.push((it.start_offset(), it.end_offset()));
            }
            for term in &phrase.terms {
                let single =
                    Clause::Term(crate::query::TermQuery::new(c.field.clone(), term.clone()));
                let Some(tm) = matches_of(&single)? else {
                    continue;
                };
                let Some(mut tit) = tm.get_matches(&c.field)? else {
                    continue;
                };
                while tit.next()? {
                    let (s, e) = (tit.start_offset(), tit.end_offset());
                    if ranges.iter().any(|&(rs, re)| rs <= s && e <= re) {
                        by_term.entry(term.clone()).or_default().push((s, e));
                    }
                }
            }
        }
        for (term, mut offsets) in by_term {
            // handled with the position-insensitive terms
            if c.phrase.insensitive.contains(&term) {
                continue;
            }
            offsets.sort_unstable();
            offsets.dedup();
            let freq = i32::try_from(offsets.len()).unwrap_or(i32::MAX);
            results.push(Box::new(OfPostings::new(term, freq, offsets)));
        }
        Ok(())
    }

    /// `createOffsetsEnumsForTerms`.
    fn enums_for_terms(
        &self,
        source_terms: &[Vec<u8>],
        terms_index: &dyn Terms,
        doc: i32,
        results: &mut Vec<Box<dyn OffsetsEnum>>,
    ) -> Result<()> {
        let mut te = terms_index.iterator()?;
        for term in source_terms {
            if te.try_seek_exact(term)? {
                let mut pe = te.postings(PostingsFlags::Offsets)?;
                if pe.advance(doc)? == doc {
                    let freq = pe.freq();
                    let mut offsets = Vec::new();
                    for _ in 0..freq {
                        pe.next_position()?;
                        offsets.push((pe.start_offset(), pe.end_offset()));
                    }
                    results.push(Box::new(OfPostings::new(term.clone(), freq, offsets)));
                }
            }
        }
        Ok(())
    }

    /// `createOffsetsEnumsForAutomata`.
    fn enums_for_automata(
        &self,
        terms_index: &dyn Terms,
        doc: i32,
        results: &mut Vec<Box<dyn OffsetsEnum>>,
    ) -> Result<()> {
        let c = &self.components;
        let mut failure: Option<Error> = None;
        let mut found: Vec<(String, Vec<Postings>)> = Vec::new();
        for_each_automaton(
            &c.query,
            &*c.field_matcher,
            c.look_in_span,
            &mut |label, m| {
                if failure.is_some() {
                    return;
                }
                let mut postings: Vec<Postings> = Vec::new();
                let scan = (|| -> Result<()> {
                    let mut te = terms_index.iterator()?;
                    while let Some(term) = te.next()? {
                        if m.matches(term) {
                            let mut pe = te.postings(PostingsFlags::Offsets)?;
                            if pe.advance(doc)? == doc {
                                let freq = pe.freq();
                                let mut offsets = Vec::new();
                                for _ in 0..freq {
                                    pe.next_position()?;
                                    offsets.push((pe.start_offset(), pe.end_offset()));
                                }
                                postings.push((freq, offsets));
                            }
                        }
                    }
                    Ok(())
                })();
                match scan {
                    Ok(()) => found.push((label.to_string(), postings)),
                    Err(e) => failure = Some(e),
                }
            },
        );
        if let Some(e) = failure {
            return Err(e);
        }
        for (label, postings) in found {
            if postings.is_empty() {
                continue;
            }
            let sum_freq: i32 = postings.iter().map(|(f, _)| *f).sum();
            for (_, offsets) in postings {
                results.push(Box::new(OfPostings::new(
                    label.clone().into_bytes(),
                    sum_freq,
                    offsets,
                )));
            }
        }
        Ok(())
    }
}

/// `switch (offsetsEnums.size())`: none, the one, or a merge.
fn collapse(mut enums: Vec<Box<dyn OffsetsEnum>>) -> Result<Option<Box<dyn OffsetsEnum>>> {
    match enums.len() {
        0 => Ok(None),
        1 => Ok(enums.pop()),
        _ => Ok(Some(Box::new(MultiOffsets::new(enums)?))),
    }
}

/// `MultiFieldsOffsetStrategy.getOffsetsEnum`: the strategies' enums merged.
pub fn multi_fields_enum(
    strategies: &[Strategy],
    ctx: Option<&LeafCtx<'_>>,
    content: &[u16],
) -> Result<Option<Box<dyn OffsetsEnum>>> {
    let mut enums = Vec::new();
    for s in strategies {
        if let Some(e) = s.offsets_enum(ctx, content)? {
            enums.push(e);
        }
    }
    Ok(Some(Box::new(MultiOffsets::new(enums)?)))
}

/// The query as the one-document index can answer it. Java overlays the
/// index's other fields on the memory index; here a clause on a field the
/// highlighter does not match is taken as the hit it must have been
/// (`MUST`/`FILTER`) or as absent (`SHOULD`/`MUST_NOT`).
fn memory_index_query(
    query: &Clause,
    field_matcher: &dyn Fn(&str) -> bool,
    overlay: bool,
) -> Clause {
    if !overlay {
        return query.clone();
    }
    map_leaves(query, true, &|leaf, required| {
        let fields = clause_fields(leaf);
        if fields.is_empty() || fields.iter().any(|f| field_matcher(f)) {
            leaf.clone()
        } else if required {
            Clause::MatchAllDocs(MatchAllDocsQuery::new(1))
        } else {
            Clause::MatchNoDocs(MatchNoDocsQuery::new())
        }
    })
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

fn drain_tokens(stream: &mut dyn TokenStream) -> Result<FieldTokens> {
    stream.reset()?;
    let mut tokens = Vec::new();
    while stream.increment_token()? {
        let a = stream.attributes();
        tokens.push(FieldToken {
            term: a.term_bytes().to_vec(),
            start_offset: a.start_offset(),
            end_offset: a.end_offset(),
            position_increment: a.position_increment(),
            term_frequency: 1,
            payload: None,
        });
    }
    stream.end()?;
    let (final_position_increment, final_offset) = {
        let a = stream.attributes();
        (a.position_increment(), a.end_offset())
    };
    stream.close()?;
    Ok(FieldTokens {
        tokens,
        final_position_increment,
        final_offset,
        end_attributes: None,
    })
}

/// `AnalysisOffsetStrategy.tokenStream(content)`: the analyzer over the
/// content, which holds a field's values joined by [`MULTIVAL_SEP_CHAR`]
/// (`MultiValueTokenStream` -- offsets shifted by where a value starts, the
/// gaps between values carried in the position increment).
pub fn analyze_content(analyzer: &Analyzer, field: &str, content: &[u16]) -> Result<FieldTokens> {
    let values: Vec<&[u16]> = content.split(|&c| c == MULTIVAL_SEP_CHAR).collect();
    let mut tokens: Vec<FieldToken> = Vec::new();
    let mut start_val_idx = 0i32;
    let mut remaining_pos_inc = 0i32;
    let (mut final_position_increment, mut final_offset) = (0, 0);
    for (i, value) in values.iter().enumerate() {
        let text = String::from_utf16_lossy(value);
        let mut stream = analyzer.token_stream(field, &text)?;
        stream.reset()?;
        while stream.increment_token()? {
            let a = stream.attributes();
            let mut pos_inc = a.position_increment();
            if remaining_pos_inc > 0 {
                pos_inc += remaining_pos_inc;
                remaining_pos_inc = 0;
            }
            tokens.push(FieldToken {
                term: a.term_bytes().to_vec(),
                start_offset: start_val_idx + a.start_offset(),
                end_offset: start_val_idx + a.end_offset(),
                position_increment: pos_inc,
                term_frequency: 1,
                payload: None,
            });
        }
        stream.end()?;
        let (end_pos_inc, end_offset) = {
            let a = stream.attributes();
            (a.position_increment(), a.end_offset())
        };
        stream.close()?;
        if i + 1 == values.len() {
            final_position_increment = end_pos_inc;
            final_offset = start_val_idx + end_offset;
        } else {
            remaining_pos_inc += end_pos_inc;
            remaining_pos_inc += analyzer.position_increment_gap_for_field(field);
            start_val_idx += i32::try_from(value.len()).unwrap_or(i32::MAX) + 1;
        }
    }
    Ok(FieldTokens {
        tokens,
        final_position_increment,
        final_offset,
        end_attributes: None,
    })
}

/// The matches of a term vector without offsets: offset -1, as Java's
/// `TermVectorLeafReader` postings report them.
struct NoOffsets(Box<dyn OffsetsEnum>);

impl OffsetsEnum for NoOffsets {
    fn next_position(&mut self) -> Result<bool> {
        self.0.next_position()
    }
    fn freq(&self) -> i32 {
        self.0.freq()
    }
    fn term(&self) -> Option<&[u8]> {
        self.0.term()
    }
    // SENTINEL: `-1` = no offsets, `OffsetsEnum.startOffset`'s contract;
    // `FieldHighlighter` compares it and refuses.
    fn start_offset(&self) -> i32 {
        -1
    }
    // SENTINEL: `-1` = no offsets, as `start_offset`; compared only.
    fn end_offset(&self) -> i32 {
        -1
    }
}
