//! Collection and term statistics gathered across segments for scoring.

use super::*;

/// Reader-wide term and collection statistics, for scoring a multi-segment
/// search the way Lucene does.
///
/// Lucene's `IndexSearcher` computes `TermStatistics`/`CollectionStatistics`
/// once across the whole reader and hands the same values to every leaf, so a
/// term's idf is identical in every segment; scoring each segment from its own
/// `docFreq`/`docCount` would let the top-k fill from whichever segment makes
/// the term look rarest.
///
/// Beside the two numbers BM25 reads, it carries the rest of what
/// `IndexSearcher.termStatistics`/`collectionStatistics` hand any
/// [`similarities::Similarity`]: the term's `totalTermFreq` and the field's
/// `maxDoc`, `sumTotalTermFreq` and `sumDocFreq`, each summed the same way.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CollectionStats {
    /// Number of documents containing the term, summed across every segment.
    pub doc_freq: i64,
    /// Number of documents that have the field, summed across every segment.
    pub doc_count: i64,
    /// The term's occurrences, summed across every segment
    /// (`TermStates.totalTermFreq`).
    pub total_term_freq: i64,
    /// `IndexReader.maxDoc()`: every segment's `maxDoc`, deleted documents
    /// included.
    pub max_doc: i64,
    /// The field's tokens (`Terms.getSumTotalTermFreq`), summed.
    pub sum_total_term_freq: i64,
    /// The field's postings (`Terms.getSumDocFreq`), summed.
    pub sum_doc_freq: i64,
}

impl CollectionStats {
    /// `IndexSearcher.collectionStatistics(field)`, as a similarity takes it.
    /// The counts come from a valid index, so `CollectionStatistics`' own
    /// invariants hold; they are clamped rather than checked so a statistic
    /// a caller built by hand with only `doc_freq`/`doc_count` still scores.
    pub fn collection_statistics(&self) -> similarities::CollectionStatistics {
        let doc_count = self.doc_count.max(1);
        let sum_doc_freq = self.sum_doc_freq.max(doc_count);
        similarities::CollectionStatistics {
            max_doc: self.max_doc.max(doc_count),
            doc_count,
            sum_total_term_freq: self.sum_total_term_freq.max(sum_doc_freq),
            sum_doc_freq,
        }
    }

    /// `IndexSearcher.termStatistics(term, docFreq, totalTermFreq)`, clamped
    /// like [`Self::collection_statistics`]; `term` is `term.bytes()`.
    pub fn term_statistics(&self, term: &[u8]) -> similarities::TermStatistics {
        let doc_freq = self.doc_freq.max(1);
        similarities::TermStatistics {
            doc_freq,
            total_term_freq: self.total_term_freq.max(doc_freq),
            term: Some(std::sync::Arc::from(term)),
        }
    }
}

/// Reader-wide statistics for one [`FuzzyQuery`] clause: the expansion
/// `MultiTermQuery.TopTermsBlendedFreqScoringRewrite` produces over the
/// **whole reader**, not one leaf.
///
/// Java's `TopTermsRewrite.rewrite` drives one collector across every leaf's
/// `TermsEnum` (`TermCollectingRewrite.collectTerms`), so the selected terms,
/// their boosts and each one's `docFreq` are all reader-wide; then
/// `BlendedTermQuery.rewrite` takes `df = max` over them and every clause
/// scores with that one number against the reader-wide
/// `CollectionStatistics.docCount`. Expanding per segment instead selects a
/// *different term set* in each leaf and blends a different frequency -- the
/// multi-term twin of the defect [`CollectionStats`] documents.
#[derive(Debug, Clone, PartialEq)]
pub struct FuzzyCollectionStats {
    /// The selected terms and their `FuzzyTermsEnum` boosts, in
    /// `TopTermsRewrite`'s final order (boost descending, then term bytes
    /// ascending). Boosts are **not** clamped -- `TopTermsRewrite.build` does
    /// `Math.max(0.0f, st.boost)` when it turns the queue into clauses, and so
    /// does the scorer.
    pub terms: Vec<(Vec<u8>, f32)>,
    /// `BlendedTermQuery.rewrite`'s `df = Math.max(df, ctx.docFreq())` over
    /// every selected term's reader-wide `docFreq`.
    pub blended_doc_freq: i64,
    /// `CollectionStatistics.docCount` for the field, summed over every
    /// segment that has it.
    pub doc_count: i64,
}

/// Reader-wide statistics for the terms and fuzzy clauses a query mentions.
/// Built once per search by the multi-segment layer.
///
/// Two maps rather than one, because a fuzzy clause's statistic is not a
/// term's: it is keyed by the *query* (a fuzzy clause and a `TermQuery` on the
/// same `(field, term)` can appear in one boolean query and mean different
/// things), and it carries the whole expansion rather than a single
/// `docFreq`/`docCount` pair.
/// One term's entry in [`GlobalStats`].
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TermEntry {
    pub(super) stats: CollectionStats,
    pub(super) states: Vec<(usize, Option<lucene_codecs::blocktree::SeekedTerm>)>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GlobalStats {
    /// By field, then term: the reader-wide statistics, and where gathering
    /// them found the term in each segment.
    pub(super) terms: HashMap<String, HashMap<Vec<u8>, TermEntry>>,
    pub(super) fuzzy: HashMap<FuzzyQuery, FuzzyCollectionStats>,
    /// A multi-term query's reader-wide rewrite (`TermCollectingRewrite`),
    /// keyed by the query's `Debug` form; see
    /// [`crate::exec::extended::rewrite_multi_term`].
    pub(super) extended: HashMap<String, Clause>,
    /// The function queries' reader-wide state: each value source's
    /// `createWeight` context and each values source's rewrite.
    pub(super) functions: function::FunctionStats,
}

impl GlobalStats {
    /// An empty set of statistics -- every lookup falls back to the segment's
    /// own counters, which is what a single-segment search wants.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one term's reader-wide `docFreq`/`docCount`.
    pub fn insert_term(
        &mut self,
        field: impl Into<String>,
        term: impl Into<Vec<u8>>,
        stats: CollectionStats,
    ) {
        self.terms.entry(field.into()).or_default().insert(
            term.into(),
            TermEntry {
                stats,
                states: Vec::new(),
            },
        );
    }

    /// [`Self::insert_term`] with the term's state in each segment the
    /// statistics were summed over (keyed by the segment's
    /// [`BlockTreeFields`]' address; `None` where the term is absent) -- what
    /// Lucene's `TermStates` keeps so `TermWeight.scorer` never seeks a term
    /// twice.
    pub(crate) fn insert_term_states(
        &mut self,
        field: String,
        term: Vec<u8>,
        stats: CollectionStats,
        states: Vec<(usize, Option<lucene_codecs::blocktree::SeekedTerm>)>,
    ) {
        self.terms
            .entry(field)
            .or_default()
            .insert(term, TermEntry { stats, states });
    }

    /// The term's state in the segment whose fields are `fields`, as the
    /// statistics pass found it: `Some(None)` when it is absent there, `None`
    /// when that segment was not recorded (the caller seeks).
    pub(crate) fn term_state(
        &self,
        field: &str,
        term: &[u8],
        fields: &BlockTreeFields,
    ) -> Option<Option<lucene_codecs::blocktree::SeekedTerm>> {
        let addr = std::ptr::from_ref(fields).addr();
        self.terms
            .get(field)?
            .get(term)?
            .states
            .iter()
            .find(|(a, _)| *a == addr)
            .map(|(_, s)| *s)
    }

    /// Replaces every gathered term's statistics with what `source` says
    /// (`IndexSearcher.termStatistics` / `collectionStatistics` overridden):
    /// each field's collection statistics once, from the reader's, then
    /// each term the reader holds (a term absent here has no
    /// `TermStatistics` to ask about, as in Java, and matches nothing here).
    ///
    /// # Errors
    /// Whatever `source` reports.
    pub(crate) fn override_statistics(
        &mut self,
        source: &dyn index_searcher::StatisticsSource,
    ) -> Result<()> {
        for (field, terms) in &mut self.terms {
            let Some(local) = terms.values().map(|e| e.stats).find(|s| s.max_doc > 0) else {
                continue;
            };
            let local = similarities::CollectionStatistics {
                max_doc: local.max_doc,
                doc_count: local.doc_count,
                sum_total_term_freq: local.sum_total_term_freq,
                sum_doc_freq: local.sum_doc_freq,
            };
            let collection = source
                .collection_statistics(field, Some(local))?
                .unwrap_or(local);
            for (term, entry) in terms.iter_mut() {
                let stats = &mut entry.stats;
                if stats.doc_freq <= 0 {
                    continue;
                }
                let t = source.term_statistics(
                    field,
                    term,
                    similarities::TermStatistics {
                        doc_freq: stats.doc_freq,
                        total_term_freq: stats.total_term_freq,
                        term: Some(std::sync::Arc::from(term.as_slice())),
                    },
                )?;
                *stats = CollectionStats {
                    doc_freq: t.doc_freq,
                    doc_count: collection.doc_count,
                    total_term_freq: t.total_term_freq,
                    max_doc: collection.max_doc,
                    sum_total_term_freq: collection.sum_total_term_freq,
                    sum_doc_freq: collection.sum_doc_freq,
                };
            }
        }
        Ok(())
    }

    /// One term's reader-wide statistics, or `None` when the caller did not
    /// gather them (in which case the segment's own are already right).
    pub fn term(&self, field: &str, term: &[u8]) -> Option<&CollectionStats> {
        self.terms.get(field)?.get(term).map(|e| &e.stats)
    }

    /// Every fuzzy clause's recorded expansion.
    pub(crate) fn fuzzy_entries(
        &self,
    ) -> impl Iterator<Item = (&FuzzyQuery, &FuzzyCollectionStats)> {
        self.fuzzy.iter()
    }

    /// Records a multi-term query's reader-wide rewrite under `key`.
    pub(crate) fn insert_extended_rewrite(&mut self, key: String, rewritten: Clause) {
        self.extended.insert(key, rewritten);
    }

    /// The reader-wide rewrite recorded under `key`, if any.
    pub(crate) fn extended_rewrite(&self, key: &str) -> Option<&Clause> {
        self.extended.get(key)
    }

    /// Records one fuzzy clause's reader-wide expansion.
    pub fn insert_fuzzy(&mut self, query: FuzzyQuery, stats: FuzzyCollectionStats) {
        self.fuzzy.insert(query, stats);
    }

    /// One fuzzy clause's reader-wide expansion, or `None` when the caller did
    /// not gather it.
    pub fn fuzzy(&self, query: &FuzzyQuery) -> Option<&FuzzyCollectionStats> {
        self.fuzzy.get(query)
    }

    /// True when nothing was gathered at all.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty() && self.fuzzy.is_empty() && self.functions.is_empty()
    }

    /// The function queries' state the statistics pass computed.
    pub(crate) fn functions(&self) -> &function::FunctionStats {
        &self.functions
    }

    /// Records the function queries' state.
    pub(crate) fn set_functions(&mut self, functions: function::FunctionStats) {
        self.functions = functions;
    }
}
