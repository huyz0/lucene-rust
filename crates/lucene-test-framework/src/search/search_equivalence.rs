//! `SearchEquivalenceTestBase`: a random corpus of one-letter terms, and
//! assertions that one query's hits are a subset of (or the same set as, or
//! scored the same as) another's -- alone, under a random filter, and
//! filtered.
//!
//! The corpus is written through `RandomIndexWriter`
//! ([`TestIndex::write_random`]) and searched with two `newSearcher`s
//! ([`NewSearcher`]): `q1` on the first, `q2` on the second. Differs:
//! `Sort.INDEXORDER` is every matching document collected unscored (the
//! same set `TopFieldDocs` would hold at `n = maxDoc`); `setQueryCache(null)`
//! has nothing to disable (the searchers cache no queries).

use std::collections::BTreeSet;
use std::sync::Arc;

use lucene_index::buffered_updates::{DeleteQuery, Term};
use lucene_index::document::{Document, Store, StringField, TextField};
use lucene_search::collector::ScoreMode;
use lucene_search::extended_query::{
    MultiTermQuery, MultiTermSource, RewriteMethod, TermRangeQuery,
};
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause, PhraseQuery};
use lucene_util::automaton::automata::make_string;
use lucene_util::automaton::CharacterRunAutomaton;
use lucene_util::java_random::JavaRandom;

use super::base_explanation::top_level;
use super::new_searcher::NewSearcher;
use super::test_index::TestIndex;
use super::{check_hits, query_utils};
use crate::analysis::mock_analyzer::MockAnalyzer;
use crate::analysis::mock_tokenizer;
use crate::util::{lucene_test_case, test_util};

/// The field every document's text is in.
pub const FIELD: &str = "field";

/// `beforeClass`'s corpus, and the assertions over it.
pub struct SearchEquivalence {
    index: TestIndex,
    s1: NewSearcher,
    s2: NewSearcher,
    stopword: char,
    num_docs: i32,
    contents: Vec<String>,
    deleted: Vec<i32>,
    nightly: bool,
    multiplier: i32,
}

/// `randomChar()`: a letter, skewed towards `a`.
pub fn random_char(random: &mut JavaRandom) -> char {
    let mut c = test_util::next_int(random, i32::from(b'a'), i32::from(b'z'));
    if random.next_boolean() {
        c = test_util::next_int(random, i32::from(b'a'), c);
    }
    char::from(u8::try_from(c).expect("a letter"))
}

/// `randomFieldContents()`: up to 14 random letters, space-separated.
pub fn random_field_contents(random: &mut JavaRandom) -> String {
    let num_terms = random.next_int_bounded(15);
    let mut sb = String::new();
    for _ in 0..num_terms {
        if !sb.is_empty() {
            sb.push(' ');
        }
        sb.push(random_char(random));
    }
    sb
}

/// `randomTerm()`'s text (its field is [`FIELD`]).
pub fn random_term(random: &mut JavaRandom) -> Vec<u8> {
    random_char(random).to_string().into_bytes()
}

/// `randomFilter()`: a term range from `a`, or a phrase of two letters
/// within slop 100.
pub fn random_filter(random: &mut JavaRandom) -> Clause {
    if random.next_boolean() {
        let upper = random_term(random);
        MultiTermQuery::new(
            MultiTermSource::TermRange(TermRangeQuery::new(
                FIELD,
                Some(b"a".to_vec()),
                Some(upper),
                true,
                true,
            )),
            RewriteMethod::default(),
        )
        .into()
    } else {
        let t1 = random_term(random);
        let t2 = random_term(random);
        Clause::Phrase(PhraseQuery::new(FIELD, [t1, t2]).with_slop(100))
    }
}

/// `filteredQuery(query, filter)`: `query` required, `filter` as a filter.
#[must_use]
pub fn filtered_query(query: Clause, filter: Clause) -> Clause {
    Clause::Boolean(Box::new(BooleanQuery {
        must: vec![query],
        filter: vec![filter],
        ..BooleanQuery::default()
    }))
}

impl SearchEquivalence {
    /// `beforeClass` under the given `TEST_NIGHTLY` and multiplier: a
    /// random stop letter, `atLeast(100)` documents (`atLeast(1000)`
    /// nightly) of [`random_field_contents`], and `numDocs / 20` random
    /// deletes by id.
    pub fn new(random: &mut JavaRandom, nightly: bool, multiplier: i32) -> Self {
        let stopword = random_char(random);
        let stopset = Arc::new(
            CharacterRunAutomaton::new(&make_string(&stopword.to_string()))
                .expect("a one-letter automaton"),
        );
        let analyzer = Arc::new(
            MockAnalyzer::with_automaton(random, mock_tokenizer::whitespace(), false, stopset)
                .test_knobs(nightly, multiplier)
                .build(),
        );
        let mut num_docs = 0;
        let mut contents = Vec::new();
        let mut deleted = Vec::new();
        let index = TestIndex::write_random(random, analyzer, &[FIELD], |iw, random| {
            num_docs =
                lucene_test_case::at_least(random, multiplier, if nightly { 1000 } else { 100 });
            for i in 0..num_docs {
                let text = random_field_contents(random);
                let mut doc = Document::new();
                doc.add(StringField::new("id", i.to_string(), Store::No));
                doc.add(TextField::new(FIELD, text.clone(), Store::No));
                iw.add_document(&doc).expect("addDocument");
                contents.push(text);
            }
            // delete some docs, by term or by the same term as a `TermQuery`
            for _ in 0..num_docs / 20 {
                let id = random.next_int_bounded(num_docs);
                let term = Term::new("id", id.to_string());
                if random.next_boolean() {
                    iw.delete_documents(term).expect("deleteDocuments");
                } else {
                    iw.delete_documents_by_query(DeleteQuery::Term(term))
                        .expect("deleteDocuments");
                }
                deleted.push(id);
            }
        });
        let s1 = NewSearcher::draw(random, nightly, multiplier);
        let s2 = NewSearcher::draw(random, nightly, multiplier);
        Self {
            index,
            s1,
            s2,
            stopword,
            num_docs,
            contents,
            deleted,
            nightly,
            multiplier,
        }
    }

    /// The stop letter the analyzer drops.
    #[must_use]
    pub fn stopword(&self) -> char {
        self.stopword
    }

    /// How many documents were written (before deletes).
    #[must_use]
    pub fn num_docs(&self) -> i32 {
        self.num_docs
    }

    /// Each document's text, in id order.
    #[must_use]
    pub fn contents(&self) -> &[String] {
        &self.contents
    }

    /// The ids deleted, in the order drawn (repeats included).
    #[must_use]
    pub fn deleted(&self) -> &[i32] {
        &self.deleted
    }

    /// Runs `f` over the corpus's searcher.
    pub fn with_searcher<R>(&self, f: impl FnOnce(&IndexSearcher<'_, '_>) -> R) -> R {
        self.index.with_searcher(f)
    }

    /// Runs `f` over `s1` and `s2`, the two `newSearcher`s, and runs each
    /// search through its searcher's asserting wrappers.
    fn with_searchers<R>(
        &self,
        f: impl FnOnce(&Searching<'_, '_, '_>, &Searching<'_, '_, '_>) -> R,
    ) -> R {
        self.index.with_segments(|segments, norms| {
            let mut s1 = IndexSearcher::new(segments, norms).expect("searcher");
            let mut s2 = IndexSearcher::new(segments, norms).expect("searcher");
            self.s1.configure(&mut s1);
            self.s2.configure(&mut s2);
            f(
                &Searching {
                    searcher: &s1,
                    draws: &self.s1,
                },
                &Searching {
                    searcher: &s2,
                    draws: &self.s2,
                },
            )
        })
    }

    fn num_filters(&self, random: &mut JavaRandom) -> i32 {
        lucene_test_case::at_least(random, self.multiplier, if self.nightly { 10 } else { 3 })
    }

    /// `assertSameSet(q1, q2)`: each a subset of the other.
    pub fn assert_same_set(&self, random: &mut JavaRandom, q1: &Clause, q2: &Clause) {
        self.assert_subset_of(random, q1, q2);
        self.assert_subset_of(random, q2, q1);
    }

    /// `assertSubsetOf(q1, q2)`: alone, then under `atLeast(3)` random
    /// filters, each both as a filter clause and through
    /// [`filtered_query`].
    pub fn assert_subset_of(&self, random: &mut JavaRandom, q1: &Clause, q2: &Clause) {
        self.assert_subset_of_filtered(q1, q2, None);
        for _ in 0..self.num_filters(random) {
            let filter = random_filter(random);
            self.assert_subset_of_filtered(q1, q2, Some(&filter));
            self.assert_subset_of_filtered(
                &filtered_query(q1.clone(), filter.clone()),
                &filtered_query(q2.clone(), filter),
                None,
            );
        }
    }

    /// `assertSubsetOf(q1, q2, filter)`: in index order and by relevance,
    /// `q1` hits no more documents than `q2`, and none `q2` misses.
    ///
    /// # Panics
    /// When `q1` hits a document `q2` does not.
    pub fn assert_subset_of_filtered(&self, q1: &Clause, q2: &Clause, filter: Option<&Clause>) {
        let (q1, q2) = with_filter(q1, q2, filter);
        query_utils::check(&q1);
        query_utils::check(&q2);
        self.with_searchers(|s1, s2| {
            let n = usize::try_from(s1.searcher.max_doc()).expect("maxDoc is non-negative");
            for relevance in [false, true] {
                let (td1, td2): (Vec<i32>, Vec<i32>) = if relevance {
                    let docs = |s: &Searching<'_, '_, '_>, q: &BooleanQuery| -> Vec<i32> {
                        s.run(|s| s.search(q, n))
                            .expect("search")
                            .score_docs
                            .iter()
                            .map(|d| d.doc)
                            .collect()
                    };
                    (docs(s1, &q1), docs(s2, &q2))
                } else {
                    let docs = |s: &Searching<'_, '_, '_>, q: &BooleanQuery| -> Vec<i32> {
                        s.run(|s| check_hits::all_hits(s, q, ScoreMode::CompleteNoScores))
                            .expect("search")
                            .into_iter()
                            .map(|(d, _)| d)
                            .collect()
                    };
                    (docs(s1, &q1), docs(s2, &q2))
                };
                assert!(
                    td1.len() <= td2.len(),
                    "too many hits: {} > {}",
                    td1.len(),
                    td2.len()
                );
                let bitset: BTreeSet<i32> = td2.into_iter().collect();
                for d in td1 {
                    assert!(bitset.contains(&d), "{q1:?} hits {d}, {q2:?} does not");
                }
            }
        });
    }

    /// `assertSameScores(q1, q2)`: the same set, then the same scores --
    /// alone and under `atLeast(3)` random filters.
    pub fn assert_same_scores(&self, random: &mut JavaRandom, q1: &Clause, q2: &Clause) {
        self.assert_same_set(random, q1, q2);
        self.assert_same_scores_filtered(q1, q2, None);
        for _ in 0..self.num_filters(random) {
            let filter = random_filter(random);
            self.assert_same_scores_filtered(q1, q2, Some(&filter));
            self.assert_same_scores_filtered(
                &filtered_query(q1.clone(), filter.clone()),
                &filtered_query(q2.clone(), filter),
                None,
            );
        }
    }

    /// `assertSameScores(q1, q2, filter)`: the same hit count, and the same
    /// documents in the same order with scores within `10e-5`.
    ///
    /// # Panics
    /// When the hits differ.
    pub fn assert_same_scores_filtered(&self, q1: &Clause, q2: &Clause, filter: Option<&Clause>) {
        let (q1, q2) = with_filter(q1, q2, filter);
        self.with_searchers(|s1, s2| {
            let n = usize::try_from(s1.searcher.max_doc()).expect("maxDoc is non-negative");
            let td1 = s1.run(|s| s.search(&q1, n)).expect("search");
            let td2 = s2.run(|s| s.search(&q2, n)).expect("search");
            assert_eq!(
                td1.total_hits.value, td2.total_hits.value,
                "{q1:?} vs {q2:?}"
            );
            for (a, b) in td1.score_docs.iter().zip(&td2.score_docs) {
                assert_eq!(a.doc, b.doc, "{q1:?} vs {q2:?}");
                assert!(
                    (a.score - b.score).abs() <= 10e-5,
                    "doc {}: {} vs {} ({q1:?} vs {q2:?})",
                    a.doc,
                    a.score,
                    b.score
                );
            }
        });
    }
}

/// One of the two searchers, with what `newSearcher` drew for it.
struct Searching<'r, 's, 'a> {
    searcher: &'r IndexSearcher<'s, 'a>,
    draws: &'r NewSearcher,
}

impl<'s, 'a> Searching<'_, 's, 'a> {
    /// One search, under this searcher's asserting wrappers.
    fn run<R>(&self, f: impl FnOnce(&IndexSearcher<'s, 'a>) -> R) -> R {
        self.draws.searching(|| f(self.searcher))
    }
}

fn with_filter(q1: &Clause, q2: &Clause, filter: Option<&Clause>) -> (BooleanQuery, BooleanQuery) {
    match filter {
        Some(f) => (
            top_level(filtered_query(q1.clone(), f.clone())),
            top_level(filtered_query(q2.clone(), f.clone())),
        ),
        None => (top_level(q1.clone()), top_level(q2.clone())),
    }
}
