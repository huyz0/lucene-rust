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

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::ops::Bound;
use std::sync::Arc;

use lucene_analysis::{collect_tokens, AnalysisError, Analyzer};
use lucene_store::{DataInput, DataOutput, Error, Result};
use rayon::prelude::*;

use crate::grouping::{FxHashMap, FxHasher};

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
    /// The document's place in the writer's insertion order (increasing;
    /// set when it is added), which breaks ties of weight in index order.
    seq: u64,
}

impl Doc {
    /// The document's place in index order: weight, descending, then
    /// insertion order (a stable sort by weight over the insertion order).
    #[inline]
    fn order_key(&self) -> (std::cmp::Reverse<i64>, u64) {
        (std::cmp::Reverse(self.weight), self.seq)
    }
}

/// A run of documents in index order with their postings: the base of a
/// [`View`], or the documents added since. Its documents are the writer's,
/// shared.
#[derive(Clone, Default)]
struct Snapshot {
    /// The documents in index order ([`Doc::order_key`]).
    docs: Vec<Arc<Doc>>,
    /// The postings of the `text` field: each term's documents, ascending.
    text: BTreeMap<String, Vec<usize>>,
    /// The postings of the `textgrams` field.
    grams: FxHashMap<String, Vec<usize>>,
    /// The postings of the `contexts` field.
    contexts: FxHashMap<Vec<u8>, Vec<usize>>,
}

/// The postings of a run of a snapshot's documents, keyed by the documents'
/// own terms.
#[derive(Default)]
struct Inverted<'d> {
    text: FxHashMap<&'d str, Vec<usize>>,
    grams: FxHashMap<&'d str, Vec<usize>>,
    contexts: FxHashMap<&'d [u8], Vec<usize>>,
}

impl<'d> Inverted<'d> {
    /// The postings of `docs`, the first of them document `base`.
    fn of(docs: &'d [Arc<Doc>], base: usize) -> Self {
        let mut inv = Inverted::default();
        for (i, doc) in docs.iter().enumerate() {
            let i = base + i;
            for term in doc.terms.keys() {
                inv.text.entry(term.as_str()).or_default().push(i);
            }
            for gram in &doc.grams {
                inv.grams.entry(gram.as_str()).or_default().push(i);
            }
            for context in doc.contexts.iter().flatten() {
                inv.contexts.entry(context.as_slice()).or_default().push(i);
            }
        }
        inv
    }

    /// `later`'s postings (of documents after all of this run's) appended.
    fn append(&mut self, later: Self) {
        fn merge<K: std::hash::Hash + Eq>(
            to: &mut FxHashMap<K, Vec<usize>>,
            from: FxHashMap<K, Vec<usize>>,
        ) {
            to.reserve(from.len());
            for (k, v) in from {
                match to.entry(k) {
                    std::collections::hash_map::Entry::Occupied(mut e) => {
                        e.get_mut().extend_from_slice(&v)
                    }
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(v);
                    }
                }
            }
        }
        merge(&mut self.text, later.text);
        merge(&mut self.grams, later.grams);
        merge(&mut self.contexts, later.contexts);
    }
}

/// Below this many documents a snapshot is inverted on the calling thread.
const PARALLEL_INVERSION_DOCS: usize = 8_192;

/// How many input rows `build` reads before analysing them.
const BUILD_BATCH: usize = 4_096;

/// Below this many rows a `build` batch is analysed on the calling thread.
const PARALLEL_ANALYSIS_ROWS: usize = 256;

/// An input row as `build` reads it: text, contexts, weight, payload.
type Row = (Vec<u8>, Option<BTreeSet<Vec<u8>>>, i64, Option<Vec<u8>>);

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
    /// The run of `docs` (already in index order) with its postings.
    /// Runs of documents are inverted in parallel and their postings
    /// appended in document order, so each list is ascending.
    fn new(docs: Vec<Arc<Doc>>) -> Self {
        let (text, grams, contexts) = {
            let inv = if docs.len() < PARALLEL_INVERSION_DOCS {
                Inverted::of(&docs, 0)
            } else {
                let chunk = docs.len().div_ceil(rayon::current_num_threads().max(1));
                let runs: Vec<Inverted<'_>> = docs
                    .par_chunks(chunk)
                    .enumerate()
                    .map(|(c, run)| Inverted::of(run, c * chunk))
                    .collect();
                let mut runs = runs.into_iter();
                let mut inv = runs.next().unwrap_or_default();
                for later in runs {
                    inv.append(later);
                }
                inv
            };
            let mut text: Vec<(&str, Vec<usize>)> = inv.text.into_iter().collect();
            text.par_sort_unstable_by(|a, b| a.0.cmp(b.0));
            let text: BTreeMap<String, Vec<usize>> =
                text.into_iter().map(|(k, v)| (k.to_owned(), v)).collect();
            let grams: FxHashMap<String, Vec<usize>> = inv
                .grams
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v))
                .collect();
            let contexts: FxHashMap<Vec<u8>, Vec<usize>> = inv
                .contexts
                .into_iter()
                .map(|(k, v)| (k.to_vec(), v))
                .collect();
            (text, grams, contexts)
        };
        Snapshot {
            docs,
            text,
            grams,
            contexts,
        }
    }

    /// The documents `query` matches, ascending (so in index order). A term's
    /// are its posting list, borrowed; only a combination allocates.
    fn matching(&self, query: &Query) -> Cow<'_, [usize]> {
        fn postings(list: Option<&Vec<usize>>) -> Cow<'_, [usize]> {
            match list {
                Some(l) => Cow::Borrowed(l.as_slice()),
                None => Cow::Owned(Vec::new()),
            }
        }
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
                    return Cow::Owned(Vec::new());
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
                Cow::Owned(out)
            }
            Query::Bool(clauses) => {
                let mut required: Option<Cow<'_, [usize]>> = None;
                let mut optional: Option<Vec<usize>> = None;
                let mut excluded: Vec<usize> = Vec::new();
                for (occur, q) in clauses {
                    let docs = self.matching(q);
                    match occur {
                        Occur::Must | Occur::Filter => {
                            required = Some(match required {
                                None => docs,
                                Some(r) => Cow::Owned(intersect(&r, &docs)),
                            });
                        }
                        Occur::Should => {
                            let mut u = optional.unwrap_or_default();
                            u.extend_from_slice(&docs);
                            optional = Some(u);
                        }
                        Occur::MustNot => excluded.extend_from_slice(&docs),
                    }
                }
                // a query with no positive clause matches nothing; with only
                // optional clauses, one of them must match
                let mut out = match (required, optional) {
                    (Some(r), _) => r,
                    (None, Some(mut o)) => {
                        o.sort_unstable();
                        o.dedup();
                        Cow::Owned(o)
                    }
                    (None, None) => return Cow::Owned(Vec::new()),
                };
                if !excluded.is_empty() {
                    excluded.sort_unstable();
                    out.to_mut().retain(|d| excluded.binary_search(d).is_err());
                }
                out
            }
        }
    }

    /// Whether document `d` matches `query`: [`Self::matching`] for one
    /// document, read from the document's own fields (which its postings
    /// were inverted from).
    fn doc_matches(&self, d: usize, query: &Query) -> bool {
        let Some(doc) = self.docs.get(d) else {
            return false;
        };
        match query {
            Query::Term(Field::Text, t) => {
                std::str::from_utf8(t).is_ok_and(|t| doc.terms.contains_key(t))
            }
            Query::Term(Field::TextGrams, t) => {
                std::str::from_utf8(t).is_ok_and(|t| doc.grams.contains(t))
            }
            Query::Term(Field::Contexts, t) => doc.contexts.as_ref().is_some_and(|c| c.contains(t)),
            Query::Prefix(p) => std::str::from_utf8(p).is_ok_and(|p| {
                doc.terms
                    .range::<str, _>((Bound::Included(p), Bound::Unbounded))
                    .next()
                    .is_some_and(|(term, _)| term.starts_with(p))
            }),
            Query::Bool(clauses) => {
                let mut has_required = false;
                let mut has_optional = false;
                let mut any_optional = false;
                for (occur, q) in clauses {
                    match occur {
                        Occur::Must | Occur::Filter => {
                            has_required = true;
                            if !self.doc_matches(d, q) {
                                return false;
                            }
                        }
                        Occur::Should => {
                            has_optional = true;
                            any_optional = any_optional || self.doc_matches(d, q);
                        }
                        Occur::MustNot => {
                            if self.doc_matches(d, q) {
                                return false;
                            }
                        }
                    }
                }
                // as `matching`: the required clauses decide; without any,
                // one of the optional ones must match
                has_required || (has_optional && any_optional)
            }
        }
    }

    /// The first `num` documents (in index order) `query` matches, of the
    /// first `end` not in `dead` (a bit set over the documents; documents
    /// past its end are live). A conjunction is driven by its rarest term clause and
    /// stops at the `num`th hit -- Java's `TopFieldCollector`
    /// early-terminating over an index sorted as the hits are -- with the
    /// other clauses checked per document; any other query is
    /// [`Self::matching`] cut to `num`.
    fn top(&self, query: &Query, num: usize, dead: &[u64], end: usize) -> Vec<usize> {
        let end = end.min(self.docs.len());
        let live = |d: usize| dead.get(d / 64).is_none_or(|w| w >> (d % 64) & 1 == 0);
        if let Query::Bool(clauses) = query {
            let mut driver: Option<(usize, Cow<'_, [usize]>)> = None;
            for (i, (occur, q)) in clauses.iter().enumerate() {
                if matches!(occur, Occur::Must | Occur::Filter) && matches!(q, Query::Term(..)) {
                    let docs = self.matching(q);
                    if driver
                        .as_ref()
                        .is_none_or(|(_, best)| docs.len() < best.len())
                    {
                        driver = Some((i, docs));
                    }
                }
            }
            if let Some((skip, docs)) = driver {
                return docs
                    .iter()
                    .copied()
                    .take_while(|&d| d < end)
                    .filter(|&d| {
                        live(d)
                            && clauses
                                .iter()
                                .enumerate()
                                .all(|(i, (occur, q))| i == skip || self.clause_holds(d, *occur, q))
                    })
                    .take(num)
                    .collect();
            }
            // a required prefix and no term: the documents in index order,
            // checked one by one, while that costs less than the prefix's
            // postings would (a frequent prefix finds its hits early)
            let prefix_postings = clauses
                .iter()
                .filter_map(|(occur, q)| match (occur, q) {
                    (Occur::Must | Occur::Filter, Query::Prefix(p)) => {
                        Some(self.prefix_postings(p))
                    }
                    _ => None,
                })
                .min();
            // a disjunction (no required clause): its lists merged in
            // order, up to the `num`th document no excluded clause matches
            if !clauses.is_empty()
                && clauses
                    .iter()
                    .all(|(occur, _)| matches!(occur, Occur::Should | Occur::MustNot))
            {
                let lists: Vec<Cow<'_, [usize]>> = clauses
                    .iter()
                    .filter(|(occur, _)| *occur == Occur::Should)
                    .map(|(_, q)| self.matching(q))
                    .collect();
                let mut at = vec![0usize; lists.len()];
                let mut hits = Vec::new();
                while hits.len() < num {
                    let Some(d) = lists
                        .iter()
                        .zip(&at)
                        .filter_map(|(l, &i)| l.get(i).copied())
                        .min()
                    else {
                        break;
                    };
                    if d >= end {
                        break;
                    }
                    for (l, i) in lists.iter().zip(at.iter_mut()) {
                        if l.get(*i) == Some(&d) {
                            *i += 1;
                        }
                    }
                    if live(d)
                        && clauses
                            .iter()
                            .all(|(occur, q)| *occur != Occur::MustNot || !self.doc_matches(d, q))
                    {
                        hits.push(d);
                    }
                }
                return hits;
            }
            if let Some(budget) = prefix_postings {
                let mut hits = Vec::with_capacity(num.min(budget));
                for d in 0..end.min(budget) {
                    if live(d)
                        && clauses
                            .iter()
                            .all(|(occur, q)| self.clause_holds(d, *occur, q))
                    {
                        hits.push(d);
                        if hits.len() == num {
                            return hits;
                        }
                    }
                }
                if budget >= end {
                    return hits;
                }
            }
        }
        self.matching(query)
            .iter()
            .copied()
            .take_while(|&d| d < end)
            .filter(|&d| live(d))
            .take(num)
            .collect()
    }

    /// How many postings the `text` terms starting with `p` have.
    fn prefix_postings(&self, p: &[u8]) -> usize {
        let Ok(p) = std::str::from_utf8(p) else {
            return 0;
        };
        self.text
            .range::<str, _>((Bound::Included(p), Bound::Unbounded))
            .take_while(|(term, _)| term.starts_with(p))
            .map(|(_, docs)| docs.len())
            .sum()
    }

    /// Whether clause `(occur, q)` of a conjunction holds for document `d`
    /// (an optional clause beside a required one never decides).
    fn clause_holds(&self, d: usize, occur: Occur, q: &Query) -> bool {
        match occur {
            Occur::Must | Occur::Filter => self.doc_matches(d, q),
            Occur::MustNot => !self.doc_matches(d, q),
            Occur::Should => true,
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

/// A fixed view of the index, as a `SearcherManager` hands out: a base run
/// of documents with the ones deleted since it was made marked dead, and
/// the documents added since (Java's newer NRT segments) in a second run.
/// A lookup takes the top hits of each and merges them in index order, so
/// the view reads as one run of its live documents in index order.
#[derive(Default)]
struct View {
    base: Snapshot,
    /// The base documents deleted since it was made, a bit per document.
    dead: Vec<u64>,
    /// How many bits of `dead` are set.
    base_dead: usize,
    /// The writer's sequence number past the base's documents: a document
    /// of the view with a smaller one is in the base, else in `added`.
    base_seq_end: u64,
    /// The live documents added since the base was made.
    added: Snapshot,
    /// Segments with a `contexts` doc-values field.
    segments_with_contexts: HashSet<usize>,
    /// Whether a segment of the view has a `payloads` doc-values field, so
    /// `MultiDocValues` hands out payloads (empty for a document without one).
    has_payloads: bool,
}

/// A refresh that would leave a view's added and dead documents past
/// `1 / COMPACT_FRACTION` of its base compacts it instead: a new base of
/// every live document, nothing added.
const COMPACT_FRACTION: usize = 16;

impl View {
    /// The live documents.
    fn len(&self) -> usize {
        self.base.docs.len() - self.base_dead + self.added.docs.len()
    }

    /// Every live document, in index order.
    #[cfg(test)]
    fn live_docs(&self) -> Vec<&Doc> {
        let mut docs: Vec<&Doc> = (0..self.base.docs.len())
            .filter(|&d| self.dead[d / 64] >> (d % 64) & 1 == 0)
            .map(|d| &*self.base.docs[d])
            .chain(self.added.docs.iter().map(|d| &**d))
            .collect();
        docs.sort_by_key(|d| d.order_key());
        docs
    }

    /// The first `num` live documents (in index order) `query` matches.
    fn top(&self, query: &Query, num: usize) -> Vec<&Doc> {
        let base = self.base.top(query, num, &self.dead, usize::MAX);
        if self.added.docs.is_empty() {
            return base.into_iter().map(|d| &*self.base.docs[d]).collect();
        }
        // with `num` base hits, an added one comes before the last of them
        // or not at all
        let end = match base.last() {
            Some(&last) if base.len() == num => {
                let bound = self.base.docs[last].order_key();
                self.added.docs.partition_point(|d| d.order_key() < bound)
            }
            _ => self.added.docs.len(),
        };
        let base = base.into_iter().map(|d| &self.base.docs[d]);
        let added = if end == 0 {
            Vec::new()
        } else {
            self.added.top(query, num, &[], end)
        };
        let mut added = added.into_iter().map(|d| &self.added.docs[d]).peekable();
        let mut base = base.peekable();
        let mut hits = Vec::with_capacity(num.min(64));
        while hits.len() < num {
            let next = match (base.peek(), added.peek()) {
                (Some(b), Some(a)) if a.order_key() < b.order_key() => added.next(),
                (Some(_), _) => base.next(),
                (None, Some(_)) => added.next(),
                (None, None) => break,
            };
            hits.extend(next.map(|d| &**d));
        }
        hits
    }
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
    /// The documents of the index writer, in insertion order; `None` where
    /// one was updated away.
    docs: Vec<Option<Arc<Doc>>>,
    /// How many of `docs` are `None`.
    deleted: usize,
    /// The slots in `docs` of each text's live documents, by the text's
    /// hash (`update`'s `deleteDocuments(exactText)`).
    by_text: FxHashMap<u64, Vec<usize>>,
    /// Whether the documents changed since the last snapshot.
    dirty: bool,
    /// The sequence number ([`Doc::seq`]) the next document gets.
    next_seq: u64,
    /// The slots of `docs` before this one were there when the view was
    /// made (the later ones are added since).
    visible_slots: usize,
    /// The documents of the view deleted since it was made.
    deleted_visible: Vec<Arc<Doc>>,
    /// The next view must be made from every document (a new `build`).
    rebuild: bool,
    segments_with_contexts: HashSet<usize>,
    /// Segments with a `payloads` doc-values field.
    segments_with_payloads: HashSet<usize>,
    /// The segment new documents go to, and whether it has any yet.
    segment: usize,
    segment_has_docs: bool,
    writer_open: bool,
    searcher: Option<View>,
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
            deleted: 0,
            by_text: FxHashMap::default(),
            dirty: false,
            next_seq: 0,
            visible_slots: 0,
            deleted_visible: Vec::new(),
            rebuild: false,
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
        analyze_document(
            &self.index_analyzer,
            self.min_prefix_chars,
            self.segment,
            text,
            contexts,
            weight,
            payload,
        )
    }

    fn push(&mut self, mut doc: Doc) {
        doc.seq = self.next_seq;
        self.next_seq += 1;
        if doc.contexts.as_ref().is_some_and(|c| !c.is_empty()) {
            self.segments_with_contexts.insert(doc.segment);
        }
        if doc.payload.is_some() {
            self.segments_with_payloads.insert(doc.segment);
        }
        self.segment_has_docs = true;
        self.dirty = true;
        let slot = self.docs.len();
        self.by_text
            .entry(text_hash(&doc.text))
            .or_default()
            .push(slot);
        self.docs.push(Some(Arc::new(doc)));
    }

    /// Deletes every document whose text is `text`.
    fn delete_text(&mut self, text: &str) {
        let Some(slots) = self.by_text.get_mut(&text_hash(text)) else {
            return;
        };
        let docs = &mut self.docs;
        let (visible, deleted_visible) = (self.visible_slots, &mut self.deleted_visible);
        let mut deleted = 0;
        slots.retain(|&slot| {
            let same = docs[slot].as_ref().is_some_and(|d| d.text == text);
            if same {
                let doc = docs[slot].take();
                if slot < visible {
                    deleted_visible.extend(doc);
                }
                deleted += 1;
            }
            !same
        });
        if slots.is_empty() {
            self.by_text.remove(&text_hash(text));
        }
        if deleted > 0 {
            self.deleted += deleted;
            self.dirty = true;
            // drop the holes once they are most of the slots
            if self.deleted > 1024 && self.deleted * 2 > self.docs.len() {
                self.compact();
            }
        }
    }

    /// `docs` without its holes, `by_text` and `visible_slots` renumbered.
    fn compact(&mut self) {
        self.visible_slots = self.docs[..self.visible_slots]
            .iter()
            .filter(|d| d.is_some())
            .count();
        self.docs.retain(Option::is_some);
        self.deleted = 0;
        self.by_text.clear();
        for (slot, doc) in self.docs.iter().enumerate() {
            if let Some(doc) = doc {
                self.by_text
                    .entry(text_hash(&doc.text))
                    .or_default()
                    .push(slot);
            }
        }
    }

    /// Forgets every document (a new `build`).
    fn clear_docs(&mut self) {
        self.docs.clear();
        self.deleted = 0;
        self.by_text.clear();
        self.dirty = true;
        self.visible_slots = 0;
        self.deleted_visible.clear();
        self.rebuild = true;
    }

    /// The writer's reader sees every document; the next ones go to a new
    /// segment. A view with nothing changed since the last one is kept
    /// (Java's `maybeRefresh` finds the reader current).
    fn flush_and_snapshot(&mut self) {
        if self.segment_has_docs {
            self.segment += 1;
            self.segment_has_docs = false;
        }
        if !self.dirty && self.searcher.is_some() {
            return;
        }
        // the field survives the documents that brought it: Lucene 10.5.0
        // still hands out payloads after every such document is updated away
        // (`GenInfixSuggest`'s `payloads.tsv`)
        let has_payloads = !self.segments_with_payloads.is_empty();
        if !self.refresh_view(has_payloads) {
            // the index is sorted by weight, descending (Java's index sort on
            // the weight doc values); a stable sort keeps ties in insertion
            // order
            let mut docs: Vec<Arc<Doc>> = self.docs.iter().flatten().cloned().collect();
            docs.par_sort_by_key(|d| std::cmp::Reverse(d.weight));
            self.searcher = Some(View {
                dead: vec![0; docs.len().div_ceil(64)],
                base: Snapshot::new(docs),
                base_dead: 0,
                base_seq_end: self.next_seq,
                added: Snapshot::default(),
                segments_with_contexts: self.segments_with_contexts.clone(),
                has_payloads,
            });
        }
        self.visible_slots = self.docs.len();
        self.deleted_visible.clear();
        self.rebuild = false;
        self.dirty = false;
    }

    /// The view brought up to the writer without a new base -- the base's
    /// deleted documents marked dead, the added run made again with the
    /// documents added since -- unless there is no view, a `build` replaced
    /// the documents, or the added and dead documents would pass
    /// `1 / COMPACT_FRACTION` of the base. Whether it was.
    fn refresh_view(&mut self, has_payloads: bool) -> bool {
        let Some(view) = self.searcher.as_mut().filter(|_| !self.rebuild) else {
            return false;
        };
        let added: Vec<Arc<Doc>> = self.docs[self.visible_slots..]
            .iter()
            .flatten()
            .cloned()
            .collect();
        let in_base = self
            .deleted_visible
            .iter()
            .filter(|d| d.seq < view.base_seq_end)
            .count();
        let added_len =
            view.added.docs.len() - (self.deleted_visible.len() - in_base) + added.len();
        if (added_len + view.base_dead + in_base).saturating_mul(COMPACT_FRACTION)
            > view.base.docs.len()
        {
            return false;
        }
        let mut gone: Vec<u64> = Vec::new();
        for doc in &self.deleted_visible {
            if doc.seq >= view.base_seq_end {
                gone.push(doc.seq);
                continue;
            }
            let key = doc.order_key();
            if let Ok(d) = view.base.docs.binary_search_by(|x| x.order_key().cmp(&key)) {
                view.dead[d / 64] |= 1 << (d % 64);
                view.base_dead += 1;
            }
        }
        if !gone.is_empty() || !added.is_empty() {
            gone.sort_unstable();
            let mut docs: Vec<Arc<Doc>> = view
                .added
                .docs
                .iter()
                .filter(|d| gone.binary_search(&d.seq).is_err())
                .cloned()
                .chain(added)
                .collect();
            docs.sort_unstable_by_key(|d| d.order_key());
            view.added = Snapshot::new(docs);
        }
        view.segments_with_contexts
            .clone_from(&self.segments_with_contexts);
        view.has_payloads = has_payloads;
        true
    }
}

/// The hash `by_text` keys a text by.
fn text_hash(text: &str) -> u64 {
    use std::hash::BuildHasher;
    std::hash::BuildHasherDefault::<FxHasher>::default().hash_one(text)
}

/// `buildDocument`, for a document of segment `segment`.
fn analyze_document(
    index_analyzer: &Analyzer,
    min_prefix_chars: i32,
    segment: usize,
    text: &[u8],
    contexts: Option<&BTreeSet<Vec<u8>>>,
    weight: i64,
    payload: Option<&[u8]>,
) -> Result<Doc> {
    let text = String::from_utf8_lossy(text).into_owned();
    let tokens = index_analyzer
        .token_stream("text", &text)
        .map_err(analysis_err)
        .and_then(|mut ts| collect_tokens(&mut ts).map_err(analysis_err))?;
    let mut terms: BTreeMap<String, i32> = BTreeMap::new();
    let mut grams = BTreeSet::new();
    let mut position = -1i32;
    for token in &tokens.tokens {
        position = position.saturating_add(token.position_increment);
        if !terms.contains_key(&token.term) {
            terms.insert(token.term.clone(), position);
        }
        if min_prefix_chars > 0 && !token.term.is_empty() {
            // EdgeNGramTokenFilter(1, minPrefixChars, false): the first
            // 1..=minPrefixChars code points, each gram ending where the
            // next code point starts (or at the term's end)
            let longest = usize::try_from(min_prefix_chars).unwrap_or(usize::MAX);
            for end in token
                .term
                .char_indices()
                .map(|(i, _)| i)
                .skip(1)
                .chain(std::iter::once(token.term.len()))
                .take(longest)
            {
                let gram = &token.term[..end];
                if !grams.contains(gram) {
                    grams.insert(gram.to_owned());
                }
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
        segment,
        seq: 0,
    })
}

impl AnalyzingInfixSuggester {
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
        self.delete_text(&doc.text);
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
        let Some(view) = &self.searcher else {
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
        let hits: Vec<&Doc> = view.top(&query, num);

        match blend {
            None => self.create_results(
                view,
                &hits,
                do_highlight,
                &matched_tokens,
                prefix_token.as_deref(),
            ),
            Some(blend) => self.create_blended_results(
                view,
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
    fn payload_of(view: &View, doc: &Doc) -> Option<Vec<u8>> {
        if view.has_payloads {
            Some(doc.payload.clone().unwrap_or_default())
        } else {
            None
        }
    }

    /// `createResults`.
    fn create_results(
        &self,
        view: &View,
        hits: &[&Doc],
        do_highlight: bool,
        matched_tokens: &HashSet<String>,
        prefix_token: Option<&str>,
    ) -> Result<Vec<LookupResult>> {
        let mut results = Vec::with_capacity(hits.len());
        for doc in hits {
            let payload = Self::payload_of(view, doc);
            // doc values per segment: a segment with none has no contexts
            let contexts = if view.segments_with_contexts.contains(&doc.segment) {
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
        view: &View,
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
            let payload = Self::payload_of(view, doc);
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
        Ok(self.searcher.as_ref().map_or(0, |s| s.len() as u64))
    }

    fn build(&mut self, input: &mut dyn InputIterator) -> Result<()> {
        // a new index replaces the old one
        self.clear_docs();
        self.segments_with_contexts.clear();
        self.segments_with_payloads.clear();
        self.segment = 0;
        self.segment_has_docs = false;
        self.writer_open = true;
        // the input read a batch at a time and each batch analysed in
        // parallel, its documents added in input order up to the first that
        // fails (the rest of that batch is read but never added)
        let result = (|| -> Result<()> {
            let mut batch: Vec<Row> = Vec::with_capacity(BUILD_BATCH);
            loop {
                batch.clear();
                while batch.len() < BUILD_BATCH {
                    let Some(text) = input.next() else {
                        break;
                    };
                    let payload = if input.has_payloads() {
                        input.payload().map(<[u8]>::to_vec)
                    } else {
                        None
                    };
                    batch.push((text, input.contexts().cloned(), input.weight(), payload));
                }
                if batch.is_empty() {
                    return Ok(());
                }
                let (analyzer, min_prefix_chars, segment) =
                    (&*self.index_analyzer, self.min_prefix_chars, self.segment);
                let analyze = |(text, contexts, weight, payload): &Row| {
                    analyze_document(
                        analyzer,
                        min_prefix_chars,
                        segment,
                        text,
                        contexts.as_ref(),
                        *weight,
                        payload.as_deref(),
                    )
                };
                let docs: Vec<Result<Doc>> = if batch.len() < PARALLEL_ANALYSIS_ROWS {
                    batch.iter().map(analyze).collect()
                } else {
                    batch.par_iter().with_min_len(64).map(analyze).collect()
                };
                for doc in docs {
                    self.push(doc?);
                }
            }
        })();
        if let Err(e) = result {
            self.clear_docs();
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `(text, weight, contexts)` rows.
    /// A test row: text, weight, contexts.
    type TestRow = (String, i64, Option<BTreeSet<Vec<u8>>>);

    struct Rows {
        rows: Vec<TestRow>,
        next: usize,
    }

    impl InputIterator for Rows {
        fn next(&mut self) -> Option<Vec<u8>> {
            let row = self.rows.get(self.next)?;
            self.next += 1;
            Some(row.0.as_bytes().to_vec())
        }
        fn weight(&self) -> i64 {
            self.rows[self.next - 1].1
        }
        fn payload(&self) -> Option<&[u8]> {
            None
        }
        fn has_payloads(&self) -> bool {
            false
        }
        fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
            self.rows[self.next - 1].2.as_ref()
        }
        fn has_contexts(&self) -> bool {
            true
        }
    }

    const WORDS: [&str; 12] = [
        "apple", "apply", "ape", "banana", "band", "bandit", "cat", "catalog", "dog", "dot",
        "éclair", "écho",
    ];

    fn rows(n: usize) -> Vec<TestRow> {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |m: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % m as u64) as usize
        };
        (0..n)
            .map(|i| {
                let words = 1 + next(4);
                let text: Vec<&str> = (0..words).map(|_| WORDS[next(WORDS.len())]).collect();
                let contexts = match next(3) {
                    0 => None,
                    1 => Some(BTreeSet::from([b"red".to_vec()])),
                    _ => Some(BTreeSet::from([b"red".to_vec(), b"blue".to_vec()])),
                };
                // weights collide, so ties keep insertion order
                (format!("{} {i}", text.join(" ")), next(50) as i64, contexts)
            })
            .collect()
    }

    fn suggester(n: usize) -> AnalyzingInfixSuggester {
        let analyzer = Arc::new(Analyzer::standard(None));
        let mut s = AnalyzingInfixSuggester::with_options(
            analyzer.clone(),
            analyzer,
            4,
            false,
            true,
            false,
            false,
        )
        .unwrap();
        s.build(&mut Rows {
            rows: rows(n),
            next: 0,
        })
        .unwrap();
        s
    }

    fn term(field: Field, t: &str) -> Query {
        Query::Term(field, t.as_bytes().to_vec())
    }

    fn leaves() -> Vec<Query> {
        vec![
            term(Field::Text, "apple"),
            term(Field::Text, "dot"),
            term(Field::Text, "nowhere"),
            term(Field::TextGrams, "ba"),
            term(Field::TextGrams, "é"),
            term(Field::Contexts, "blue"),
            Query::Prefix(b"band".to_vec()),
            Query::Prefix(b"zzzz".to_vec()),
            Query::Prefix(b"cata".to_vec()),
            Query::Term(Field::Text, vec![0xff]),
            Query::Prefix(vec![0xff]),
        ]
    }

    #[test]
    fn top_hits_are_the_first_of_all_matches_for_every_query_shape() {
        // past the parallel inversion and analysis thresholds
        let s = suggester(PARALLEL_INVERSION_DOCS + 500);
        let snap = &s.searcher.as_ref().unwrap().base;
        // the parallel inversion is the sequential one
        let inv = Inverted::of(&snap.docs, 0);
        assert_eq!(inv.text.len(), snap.text.len());
        for (k, v) in &inv.text {
            assert_eq!(snap.text.get(*k), Some(v), "{k}");
        }
        assert_eq!(inv.grams.len(), snap.grams.len());
        for (k, v) in &inv.grams {
            assert_eq!(snap.grams.get(*k), Some(v), "{k}");
        }
        assert_eq!(inv.contexts.len(), snap.contexts.len());

        let n = leaves().len();
        let mut queries = Vec::new();
        for a in 0..n {
            for b in 0..n {
                for (oa, ob) in [
                    (Occur::Must, Occur::Must),
                    (Occur::Must, Occur::MustNot),
                    (Occur::Must, Occur::Should),
                    (Occur::Should, Occur::Should),
                    (Occur::Filter, Occur::Must),
                    (Occur::MustNot, Occur::MustNot),
                ] {
                    let (la, lb) = (leaves().swap_remove(a), leaves().swap_remove(b));
                    queries.push(Query::Bool(vec![(oa, la), (ob, lb)]));
                }
            }
            // a lone clause, and one beside a context sub-query
            queries.push(Query::Bool(vec![(Occur::Must, leaves().swap_remove(a))]));
            queries.push(Query::Bool(vec![
                (Occur::Must, leaves().swap_remove(a)),
                (
                    Occur::Must,
                    Query::Bool(vec![
                        (Occur::Should, term(Field::Contexts, "red")),
                        (Occur::Should, term(Field::Contexts, "green")),
                    ]),
                ),
                (Occur::MustNot, term(Field::Contexts, "blue")),
            ]));
        }
        queries.push(Query::Bool(Vec::new()));
        for q in &queries {
            let all = snap.matching(q);
            for num in [1, 7, 100, usize::MAX] {
                let want: Vec<usize> = all.iter().copied().take(num).collect();
                assert_eq!(snap.top(q, num, &[], usize::MAX), want, "num {num}");
                // every third document dead, the documents cut at `end`
                let dead: Vec<u64> = (0..snap.docs.len().div_ceil(64))
                    .map(|w| {
                        (0..64)
                            .filter(|b| (w * 64 + b) % 3 == 0)
                            .fold(0u64, |m, b| m | 1 << b)
                    })
                    .collect();
                for end in [0, 5, snap.docs.len() / 2] {
                    let want: Vec<usize> = all
                        .iter()
                        .copied()
                        .filter(|&d| d < end && d % 3 != 0)
                        .take(num)
                        .collect();
                    assert_eq!(snap.top(q, num, &dead, end), want, "num {num} end {end}");
                }
            }
            for d in [0, 1, all.first().copied().unwrap_or(2)] {
                assert_eq!(snap.doc_matches(d, q), all.contains(&d));
            }
        }
        assert!(!snap.doc_matches(snap.docs.len(), &term(Field::Text, "apple")));
    }

    #[test]
    fn a_rare_prefix_falls_back_to_its_postings() {
        // the only "bandit" documents are the last ones: the scan's budget
        // (the prefix's postings) runs out before it reaches them
        let s = suggester(300);
        let snap = &s.searcher.as_ref().unwrap().base;
        let q = Query::Bool(vec![(Occur::Must, Query::Prefix(b"bandi".to_vec()))]);
        let all = snap.matching(&q);
        assert!(!all.is_empty());
        assert_eq!(snap.top(&q, 1000, &[], usize::MAX), all.to_vec());
        assert_eq!(snap.top(&q, 1, &[], usize::MAX), all[..1].to_vec());
    }

    #[test]
    fn edge_ngrams_are_code_point_prefixes() {
        let s = suggester(0);
        let doc = s
            .build_document("éclair ab".as_bytes(), None, 1, None)
            .unwrap();
        let grams: Vec<&str> = doc.grams.iter().map(String::as_str).collect();
        assert_eq!(grams, ["a", "ab", "é", "éc", "écl", "écla"]);
    }

    #[test]
    fn update_replaces_every_copy_and_refresh_skips_an_unchanged_view() {
        let mut s = suggester(50);
        let first = s.searcher.as_ref().unwrap().base.docs.as_ptr();
        s.refresh().unwrap();
        assert_eq!(s.searcher.as_ref().unwrap().base.docs.as_ptr(), first);

        // a model of the writer: (text, weight) in insertion order
        let mut model: Vec<(String, i64)> = rows(50).into_iter().map(|r| (r.0, r.1)).collect();
        s.add(b"dup", None, 3, None).unwrap();
        s.add(b"dup", None, 4, None).unwrap();
        model.push(("dup".into(), 3));
        model.push(("dup".into(), 4));
        let texts: Vec<String> = model.iter().map(|m| m.0.clone()).collect();
        for round in 0..3000usize {
            let text = &texts[round * 7 % texts.len()];
            let weight = (round % 40) as i64;
            s.update(text.as_bytes(), None, weight, None).unwrap();
            model.retain(|(t, _)| t != text);
            model.push((text.clone(), weight));
        }
        // the holes were dropped along the way
        assert!(s.deleted < s.docs.len());
        let live: Vec<(String, i64)> = s
            .docs
            .iter()
            .flatten()
            .map(|d| (d.text.clone(), d.weight))
            .collect();
        assert_eq!(live, model);
        s.refresh().unwrap();
        let mut sorted = model.clone();
        sorted.sort_by_key(|m| std::cmp::Reverse(m.1));
        let view: Vec<(String, i64)> = s
            .searcher
            .as_ref()
            .unwrap()
            .live_docs()
            .into_iter()
            .map(|d| (d.text.clone(), d.weight))
            .collect();
        assert_eq!(view, sorted);
        assert_eq!(s.get_count().unwrap(), 51);
        let hits = s.lookup_with("dup", None, 5, true, false).unwrap();
        assert_eq!(hits.len(), 1);

        // a closed suggester reopened by `add` sees its documents again
        s.close();
        s.add(b"late", None, 1, None).unwrap();
        s.refresh().unwrap();
        assert_eq!(s.get_count().unwrap(), 52);
    }

    /// Every lookup shape the test below asks, as `Debug` text.
    fn lookups(s: &AnalyzingInfixSuggester) -> Vec<String> {
        let red = BTreeSet::from([b"red".to_vec()]);
        let not_blue = BTreeMap::from([(b"blue".to_vec(), Occur::MustNot)]);
        let red_and_blue = BTreeMap::from([
            (b"red".to_vec(), Occur::Must),
            (b"blue".to_vec(), Occur::Should),
        ]);
        let mut out = Vec::new();
        for key in [
            "a", "ap", "appl", "apple", "apple ", "band", "ba ca", "cat dog", "é", "écl", "zzz",
            "dup",
        ] {
            for num in [1, 4, 30] {
                for all in [true, false] {
                    out.push(format!("{:?}", s.lookup_with(key, None, num, all, true)));
                    out.push(format!(
                        "{:?}",
                        s.lookup_with(key, Some(&red), num, all, false)
                    ));
                    out.push(format!(
                        "{:?}",
                        s.lookup_context_info(key, &not_blue, num, all, false)
                    ));
                    out.push(format!(
                        "{:?}",
                        s.lookup_context_info(key, &red_and_blue, num, all, false)
                    ));
                }
            }
        }
        out.push(format!("{:?}", s.get_count()));
        out
    }

    /// An incremental refresh (dead base documents, an added run, compaction)
    /// reads as a view made again from every document: the same lookups, in
    /// the same order, under random interleaved adds, updates, refreshes,
    /// commits and closes.
    #[test]
    fn incremental_refreshes_read_as_full_rebuilds() {
        let base_rows = rows(1500);
        let mut s = suggester(1500);
        let mut model = suggester(1500);
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move |m: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % m as u64) as usize
        };
        let mut texts: Vec<String> = base_rows.iter().map(|r| r.0.clone()).collect();
        let (mut incremental, mut compactions) = (0, 0);
        for step in 0..1200 {
            let op = next(100);
            let weight = next(60) as i64;
            let contexts = match next(3) {
                0 => None,
                1 => Some(BTreeSet::from([b"red".to_vec()])),
                _ => Some(BTreeSet::from([b"blue".to_vec()])),
            };
            let payload = (next(4) == 0).then(|| format!("p{step}").into_bytes());
            if op < 40 {
                // an update of a text already there (often one just added)
                let text = if next(3) == 0 {
                    texts[texts.len() - 1 - next(texts.len().min(20))].clone()
                } else {
                    texts[next(texts.len())].clone()
                };
                for t in [&mut s, &mut model] {
                    t.update(
                        text.as_bytes(),
                        contexts.as_ref(),
                        weight,
                        payload.as_deref(),
                    )
                    .unwrap();
                }
            } else if op < 70 {
                let words = 1 + next(3);
                let mut text: Vec<&str> = (0..words).map(|_| WORDS[next(WORDS.len())]).collect();
                if next(10) == 0 {
                    text = vec!["dup"];
                }
                let text = format!(
                    "{} {}",
                    text.join(" "),
                    if text == ["dup"] { 0 } else { step }
                );
                texts.push(text.clone());
                for t in [&mut s, &mut model] {
                    t.add(
                        text.as_bytes(),
                        contexts.as_ref(),
                        weight,
                        payload.as_deref(),
                    )
                    .unwrap();
                }
            } else if op < 90 {
                let before = s.searcher.as_ref().map(|v| v.base.docs.as_ptr());
                s.refresh().unwrap();
                model.rebuild = true;
                model.refresh().unwrap();
                let view = s.searcher.as_ref().unwrap();
                if before == Some(view.base.docs.as_ptr()) {
                    incremental += usize::from(!view.added.docs.is_empty() || view.base_dead > 0);
                } else {
                    compactions += 1;
                }
                assert_eq!(model.searcher.as_ref().unwrap().base_dead, 0);
                let docs = |v: &View| -> Vec<(String, i64, u64)> {
                    v.live_docs()
                        .into_iter()
                        .map(|d| (d.text.clone(), d.weight, d.seq))
                        .collect()
                };
                assert_eq!(
                    docs(view),
                    docs(model.searcher.as_ref().unwrap()),
                    "step {step}"
                );
            } else if op < 97 {
                // lookups between refreshes see the last view only
                assert_eq!(lookups(&s), lookups(&model), "step {step}");
            } else if op < 99 {
                s.commit().unwrap();
                model.commit().unwrap();
            } else {
                s.close();
                model.close();
                // reopened by the next write; a lookup needs a view again
                for t in [&mut s, &mut model] {
                    t.add(b"reopened", None, 5, None).unwrap();
                    t.refresh().unwrap();
                }
            }
        }
        s.refresh().unwrap();
        model.rebuild = true;
        model.refresh().unwrap();
        assert_eq!(lookups(&s), lookups(&model));
        // both paths ran
        assert!(incremental > 20, "{incremental}");
        assert!(compactions > 2, "{compactions}");
    }
}
