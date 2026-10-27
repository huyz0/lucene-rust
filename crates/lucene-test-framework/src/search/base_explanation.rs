//! `BaseExplanationTestCase`: four small documents, and `qtest`, which runs
//! a query through [`check_hits::check_hit_collector`] -- every
//! `QueryUtils` check, explanations and matches included -- and asserts
//! exactly which documents it hits.
//!
//! The documents are written as Java writes them: `newStringField` and
//! `newTextField`'s random field types ([`NewFields`]), through a
//! `RandomIndexWriter` under a random configuration with a random
//! `LogMergePolicy` ([`TestIndex::write_random_with`]), searched through a
//! `newSearcher` ([`NewSearcher`]). Differs: the codec is the default (the
//! writer's), and the searcher's similarity is the default or the one
//! [`BaseExplanation::with_similarity`] sets, not the class environment's.

use std::sync::Arc;

use lucene_index::document::{Document, SortedDocValuesField, Store, StringField, TextField};
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_search::similarities::Similarity;
use lucene_util::java_random::JavaRandom;

use super::check_hits;
use super::new_searcher::NewSearcher;
use super::test_index::TestIndex;
use crate::analysis::mock_analyzer::MockAnalyzer;
use crate::index::index_writer_config::{
    new_index_writer_config, new_log_merge_policy, Draws, LiveMergePolicy, PolicyKind,
};
use crate::index::random_index_writer::RandomIndexWriter;
use crate::util::lucene_test_case;
use crate::util::new_field::NewFields;

/// `KEY`: each document's number, as a string field and sorted doc values.
pub const KEY: &str = "KEY";
/// `FIELD`: the document text.
pub const FIELD: &str = "field";
/// `ALTFIELD`: the same text again.
pub const ALTFIELD: &str = "alt";

/// `docFields`.
pub const DOC_FIELDS: [&str; 4] = [
    "w1 w2 w3 w4 w5",
    "w1 w3 w2 w3 zz",
    "w1 xx w2 yy w3",
    "w1 w3 xx w2 yy w3 zz",
];

/// `createDoc(index)`, its fields of `newStringField`/`newTextField`'s
/// random types.
pub fn create_random_doc(
    fields: &mut NewFields,
    random: &mut JavaRandom,
    index: usize,
) -> Document {
    let mut doc = Document::new();
    doc.add(fields.new_string_field(random, KEY, &index.to_string(), Store::No));
    doc.add(SortedDocValuesField::new(
        KEY,
        index.to_string().into_bytes(),
    ));
    doc.add(fields.new_text_field(random, FIELD, DOC_FIELDS[index], Store::No));
    doc.add(fields.new_text_field(random, ALTFIELD, DOC_FIELDS[index], Store::No));
    doc
}

/// `createDoc(index)` with the plain field types (no random upgrade).
#[must_use]
pub fn create_doc(index: usize) -> Document {
    let mut doc = Document::new();
    doc.add(StringField::new(KEY, index.to_string(), Store::No));
    doc.add(SortedDocValuesField::new(
        KEY,
        index.to_string().into_bytes(),
    ));
    doc.add(TextField::new(FIELD, DOC_FIELDS[index], Store::No));
    doc.add(TextField::new(ALTFIELD, DOC_FIELDS[index], Store::No));
    doc
}

/// `beforeClassTestExplanations`'s index, and the test case's helpers.
pub struct BaseExplanation {
    index: TestIndex,
    searcher: NewSearcher,
    similarity: Option<Box<dyn Similarity>>,
}

/// A query as the top-level [`BooleanQuery`] a searcher takes: a boolean
/// clause is itself, any other clause its single required clause (which
/// `BooleanQuery.rewrite` reduces to the clause, scores unchanged).
#[must_use]
pub fn top_level(q: Clause) -> BooleanQuery {
    match q {
        Clause::Boolean(b) => *b,
        q => BooleanQuery {
            must: vec![q],
            ..BooleanQuery::default()
        },
    }
}

fn never_match() -> Clause {
    Clause::Term(TermQuery::new("NEVER", "MATCH"))
}

impl BaseExplanation {
    /// `beforeClassTestExplanations`: the documents analysed by a
    /// `MockAnalyzer(random)`, written through a `RandomIndexWriter`, and
    /// the `newSearcher` over them, everything drawn from `random` (the
    /// environment's `TEST_NIGHTLY` and multiplier).
    #[must_use]
    pub fn new(random: &mut JavaRandom) -> Self {
        let (nightly, multiplier) = lucene_test_case::knobs();
        let analyzer = Arc::new(MockAnalyzer::new(random).build());
        let index = TestIndex::write_random_with(
            random,
            |random, dir, env| {
                let mut config = new_index_writer_config(&mut Draws::new(random), analyzer, 0);
                let log = new_log_merge_policy(&mut Draws::new(random));
                config.merge_policy = LiveMergePolicy::new(PolicyKind::Log(log));
                RandomIndexWriter::with_config(random, dir, config, env)
            },
            &[FIELD, ALTFIELD],
            |w, random| {
                let mut fields = NewFields::new();
                for i in 0..DOC_FIELDS.len() {
                    let doc = create_random_doc(&mut fields, random, i);
                    w.add_document(&doc).expect("addDocument");
                }
            },
        );
        Self {
            index,
            searcher: NewSearcher::draw(random, nightly, multiplier),
            similarity: None,
        }
    }

    /// `searcher.setSimilarity(similarity)` for every later search.
    #[must_use]
    pub fn with_similarity(mut self, similarity: Box<dyn Similarity>) -> Self {
        self.similarity = Some(similarity);
        self
    }

    /// Runs `f` over the index's `newSearcher`, under its asserting
    /// wrappers.
    pub fn with_searcher<R>(&self, f: impl FnOnce(&IndexSearcher<'_, '_>) -> R) -> R {
        self.index.with_segments(|segments, norms| {
            let mut s = IndexSearcher::new(segments, norms).expect("searcher");
            if let Some(sim) = self.similarity.as_deref() {
                s.set_similarity(sim);
            }
            self.searcher.configure(&mut s);
            self.searcher.searching(|| f(&s))
        })
    }

    /// `qtest(q, expDocNrs)`: half the time `q` is first wrapped as a
    /// `SHOULD` clause beside one that never matches.
    ///
    /// # Panics
    /// When a check fails or searching reports an error.
    pub fn qtest(&self, random: &mut JavaRandom, q: impl Into<Clause>, exp_doc_nrs: &[i32]) {
        let mut q = q.into();
        if random.next_boolean() {
            q = Clause::Boolean(Box::new(BooleanQuery {
                should: vec![q, never_match()],
                ..BooleanQuery::default()
            }));
        }
        let q = top_level(q);
        self.with_searcher(|s| check_hits::check_hit_collector(random, &q, s, exp_doc_nrs))
            .expect("search");
    }

    /// `bqtest(q, expDocNrs)`: [`Self::qtest`] of [`req_b`] then of
    /// [`opt_b`].
    pub fn bqtest(&self, random: &mut JavaRandom, q: impl Into<Clause>, exp_doc_nrs: &[i32]) {
        let q = q.into();
        self.qtest(random, req_b(q.clone()), exp_doc_nrs);
        self.qtest(random, opt_b(q), exp_doc_nrs);
    }
}

/// `matchTheseItems(terms)`: a disjunction of `KEY` terms.
#[must_use]
pub fn match_these_items(terms: &[i32]) -> Clause {
    Clause::Boolean(Box::new(BooleanQuery {
        should: terms
            .iter()
            .map(|t| Clause::Term(TermQuery::new(KEY, t.to_string())))
            .collect(),
        ..BooleanQuery::default()
    }))
}

/// `ta(s)`: the strings as `FIELD` terms' bytes.
#[must_use]
pub fn ta(s: &[&str]) -> Vec<Vec<u8>> {
    s.iter().map(|t| t.as_bytes().to_vec()).collect()
}

/// `optB(q)`: `q` optional, beside a prohibited clause that never matches.
#[must_use]
pub fn opt_b(q: Clause) -> Clause {
    Clause::Boolean(Box::new(BooleanQuery {
        should: vec![q],
        must_not: vec![never_match()],
        ..BooleanQuery::default()
    }))
}

/// `reqB(q)`: `q` required, beside an optional `FIELD:w1`.
#[must_use]
pub fn req_b(q: Clause) -> Clause {
    Clause::Boolean(Box::new(BooleanQuery {
        must: vec![q],
        should: vec![Clause::Term(TermQuery::new(FIELD, "w1"))],
        ..BooleanQuery::default()
    }))
}
