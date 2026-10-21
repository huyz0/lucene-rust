//! `MatchesTestBase`: the same documents in fields indexed with offsets,
//! positions, freqs and docs only, and checks of the `Matches` a query
//! reports for each document -- positions and offsets, labels, sub-matches,
//! named matches, term matches.
//!
//! Java writes the documents through `RandomIndexWriter` and force-merges
//! to one segment; here they are one segment under a plain `IndexWriter`.
//! `NamedQuery` has no `Clause` here, so [`MatchesTestBase::check_sub_matches`]
//! takes the named clauses `NamedMatches.wrapQuery` would wrap (see
//! `lucene_search::matches::named_matches`).

use std::collections::BTreeSet;
use std::sync::Arc;

use lucene_index::document::{
    Document, Field, FieldType, IndexOptions, IntPoint, NumericDocValuesField, TextField,
};
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::matches::{self, BoxMatchesIterator};
use lucene_search::query::Clause;
use lucene_util::java_random::JavaRandom;

use super::asserting_matches::AssertingMatches;
use super::test_index::TestIndex;
use crate::analysis::mock_analyzer::MockAnalyzer;

/// `FIELD_WITH_OFFSETS`.
pub const FIELD_WITH_OFFSETS: &str = "field_offsets";
/// `FIELD_NO_OFFSETS`.
pub const FIELD_NO_OFFSETS: &str = "field_no_offsets";
/// `FIELD_DOCS_ONLY`.
pub const FIELD_DOCS_ONLY: &str = "field_docs_only";
/// `FIELD_FREQS`.
pub const FIELD_FREQS: &str = "field_freqs";
/// `FIELD_POINT`.
pub const FIELD_POINT: &str = "field_point";

/// `TermMatch`: one term's position and offsets within a match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TermMatch {
    /// The term's position.
    pub position: i32,
    /// Its start offset.
    pub start_offset: i32,
    /// Its end offset.
    pub end_offset: i32,
}

impl TermMatch {
    /// `new TermMatch(position, startOffset, endOffset)`.
    #[must_use]
    pub fn new(position: i32, start_offset: i32, end_offset: i32) -> Self {
        Self {
            position,
            start_offset,
            end_offset,
        }
    }
}

impl std::fmt::Display for TermMatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}[{}->{}]",
            self.position, self.start_offset, self.end_offset
        )
    }
}

/// `TextField.TYPE_STORED` with other index options.
fn text_type(options: IndexOptions) -> FieldType {
    let mut ft = FieldType::copy_of(&TextField::type_stored());
    ft.set_index_options(options).expect("unfrozen");
    ft.frozen()
}

/// `setUp`'s index over [`Self::new`]'s documents.
pub struct MatchesTestBase {
    index: TestIndex,
}

impl MatchesTestBase {
    /// `setUp`: each of `documents` in every field, with an int point and
    /// numeric doc values of 10 and an `id`.
    ///
    /// # Panics
    /// When building a field fails.
    pub fn new(random: &mut JavaRandom, documents: &[&str]) -> Self {
        let analyzer = Arc::new(MockAnalyzer::new(random).build());
        let offsets = text_type(IndexOptions::DocsAndFreqsAndPositionsAndOffsets);
        let docs_only = text_type(IndexOptions::Docs);
        let freqs = text_type(IndexOptions::DocsAndFreqs);
        let field = |name: &str, text: &str, ft: &FieldType| {
            Field::from_string(name, text, ft.clone()).expect("an indexed field")
        };
        let docs: Vec<Document> = documents
            .iter()
            .zip(0i64..)
            .map(|(text, i)| {
                let mut doc = Document::new();
                doc.add(field(FIELD_WITH_OFFSETS, text, &offsets));
                doc.add(field(FIELD_NO_OFFSETS, text, &TextField::type_stored()));
                doc.add(field(FIELD_DOCS_ONLY, text, &docs_only));
                doc.add(field(FIELD_FREQS, text, &freqs));
                doc.add(IntPoint::new(FIELD_POINT, &[10]).expect("a point"));
                doc.add(NumericDocValuesField::new(FIELD_POINT, 10));
                doc.add(NumericDocValuesField::new("id", i));
                doc.add(field("id", &i.to_string(), &TextField::type_stored()));
                doc
            })
            .collect();
        Self {
            index: TestIndex::build(analyzer, &[docs], &[], &[]),
        }
    }

    /// Runs `f` over the index's searcher.
    pub fn with_searcher<R>(&self, f: impl FnOnce(&IndexSearcher<'_, '_>) -> R) -> R {
        self.index.with_searcher(f)
    }

    /// `q`'s matches of `doc`, wrapped as `AssertingWeight.matches` wraps
    /// them.
    fn matches(&self, q: &Clause, doc: i32) -> Option<matches::BoxMatches> {
        self.with_searcher(|s| matches::matches(s, q, doc))
            .expect("matches")
            .map(|m| Box::new(AssertingMatches::new(m)) as matches::BoxMatches)
    }

    /// `checkMatches(q, field, expected)`: each row is a document then
    /// `(startPosition, endPosition, startOffset, endOffset)` per match; a
    /// row of the document alone expects no matches in `field`.
    ///
    /// # Panics
    /// When the matches differ.
    pub fn check_matches(&self, q: &Clause, field: &str, expected: &[&[i32]]) {
        for row in expected {
            let Some(m) = self.matches(q, row[0]) else {
                assert_eq!(row.len(), 1, "no matches on doc {}", row[0]);
                continue;
            };
            let it = m.get_matches(field).expect("matches");
            if row.len() == 1 {
                assert!(it.is_none(), "unexpected matches on doc {}", row[0]);
                continue;
            }
            check_field_matches(it.expect("matches in field"), row);
            // test multiple calls
            check_field_matches(m.get_matches(field).expect("matches").expect("again"), row);
        }
    }

    /// `checkLabelCount(q, field, expected)`: how many distinct queries
    /// (by identity) report matches in `field`, per document.
    ///
    /// # Panics
    /// When a count differs.
    pub fn check_label_count(&self, q: &Clause, field: &str, expected: &[usize]) {
        for (doc, &want) in (0..).zip(expected) {
            let Some(m) = self.matches(q, doc) else {
                assert_eq!(want, 0, "Expected to get matches on document {doc}");
                continue;
            };
            let it = m.get_matches(field).expect("matches");
            if want == 0 {
                assert!(it.is_none(), "unexpected matches on doc {doc}");
                continue;
            }
            let mut it = it.expect("matches in field");
            let mut labels = BTreeSet::new();
            while it.next().expect("next") {
                labels.insert(std::ptr::from_ref(it.query()) as usize);
            }
            assert_eq!(want, labels.len(), "labels on doc {doc}");
        }
    }

    /// `checkNoPositionsMatches(q, field, expected)`: a matching document's
    /// matches have no positions (`-1`).
    ///
    /// # Panics
    /// When a document matches unexpectedly or reports a position.
    pub fn check_no_positions_matches(&self, q: &Clause, field: &str, expected: &[bool]) {
        for (doc, &want) in (0..).zip(expected) {
            let m = self.matches(q, doc);
            if !want {
                assert!(m.is_none(), "unexpected matches on doc {doc}");
                continue;
            }
            let mut mi = m
                .expect("matches")
                .get_matches(field)
                .expect("matches")
                .expect("matches in field");
            assert!(mi.next().expect("next"));
            assert_eq!(-1, mi.start_position());
            while mi.next().expect("next") {
                assert_eq!(-1, mi.start_position());
            }
        }
    }

    /// `checkSubMatches(q, expectedNames)` with `q` a disjunction of named
    /// clauses: exactly the expected names match each document.
    ///
    /// # Panics
    /// When a name is missing or unexpected.
    pub fn check_sub_matches(&self, named: &[(String, Clause)], expected_names: &[&[&str]]) {
        self.with_searcher(|s| {
            for (doc, names) in (0..).zip(expected_names) {
                let leaf = s.segment_of(doc).expect("a document");
                let seg = &s.segments()[leaf];
                let found = matches::named_matches(seg, named, doc - seg.doc_base).expect("named");
                let actual: BTreeSet<&str> =
                    found.iter().map(matches::NamedMatches::name).collect();
                let want: BTreeSet<&str> = names.iter().copied().collect();
                let unexpected: Vec<_> = actual.difference(&want).collect();
                assert!(
                    unexpected.is_empty(),
                    "Unexpected matching leaf queries: {unexpected:?}"
                );
                let missing: Vec<_> = want.difference(&actual).collect();
                assert!(
                    missing.is_empty(),
                    "Missing matching leaf queries: {missing:?}"
                );
            }
        });
    }

    /// `assertIsLeafMatch(q, field)`: up to the first document without
    /// matches in `field`, no match has sub-matches.
    ///
    /// # Panics
    /// When a match has sub-matches.
    pub fn assert_is_leaf_match(&self, q: &Clause, field: &str) {
        let max_doc = self.with_searcher(|s| s.max_doc());
        for doc in 0..max_doc {
            let Some(m) = self.matches(q, doc) else {
                return;
            };
            let Some(mut mi) = m.get_matches(field).expect("matches") else {
                return;
            };
            while mi.next().expect("next") {
                assert!(
                    mi.sub_matches().expect("sub-matches").is_none(),
                    "doc {doc}"
                );
            }
        }
    }

    /// `checkTermMatches(q, field, expected)`: per document, per match, the
    /// set of its terms' positions and offsets.
    ///
    /// # Panics
    /// When a term match is missing or unexpected.
    pub fn check_term_matches(&self, q: &Clause, field: &str, expected: &[&[&[TermMatch]]]) {
        for (doc, want) in (0..).zip(expected) {
            let Some(m) = self.matches(q, doc) else {
                assert!(want.is_empty(), "no matches on doc {doc}");
                continue;
            };
            let it = m.get_matches(field).expect("matches");
            if want.is_empty() {
                assert!(it.is_none(), "unexpected matches on doc {doc}");
                continue;
            }
            check_terms(want, it.expect("matches in field"));
        }
    }
}

/// `checkFieldMatches(it, expected)`.
///
/// # Panics
/// When a position or offset differs, or the count does.
pub fn check_field_matches(mut it: BoxMatchesIterator, expected: &[i32]) {
    let mut pos = 1;
    while it.next().expect("next") {
        assert_eq!(expected[pos], it.start_position(), "Wrong start position");
        assert_eq!(expected[pos + 1], it.end_position(), "Wrong end position");
        assert_eq!(expected[pos + 2], it.start_offset(), "Wrong start offset");
        assert_eq!(expected[pos + 3], it.end_offset(), "Wrong end offset");
        pos += 4;
    }
    assert_eq!(expected.len(), pos);
}

fn check_terms(expected: &[&[TermMatch]], mut it: BoxMatchesIterator) {
    let mut up_to = 0;
    while it.next().expect("next") {
        let mut want: BTreeSet<TermMatch> = expected[up_to].iter().copied().collect();
        let mut sub = it
            .sub_matches()
            .expect("sub-matches")
            .expect("a match has term sub-matches");
        while sub.next().expect("next") {
            let tm = TermMatch::new(sub.start_position(), sub.start_offset(), sub.end_offset());
            assert!(want.remove(&tm), "Unexpected term match: {tm}");
        }
        let missing: Vec<String> = want.iter().map(ToString::to_string).collect();
        assert!(
            missing.is_empty(),
            "Missing term matches: {}",
            missing.join(", ")
        );
        up_to += 1;
    }
    assert!(up_to + 1 >= expected.len(), "Missing expected match");
}
