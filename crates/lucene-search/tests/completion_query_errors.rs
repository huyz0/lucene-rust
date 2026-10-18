//! The refusals and accessors of the completion queries: what Lucene throws
//! for a bad term, context, regular expression or field, over the index
//! `GenCompletionQuery` wrote (see `completion_query_fixtures.rs`) and one
//! written here that mixes both suggestion types in a field.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::path::PathBuf;
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_codecs::per_field_postings::Lucene104PostingsFormat;
use lucene_index::document::Document;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::suggest::completion_query::{
    CompletionAnalyzer, CompletionQuery, ContextQuery, FuzzyCompletionQuery, PrefixCompletionQuery,
    RegexCompletionQuery, SuggestIndexSearcher,
};
use lucene_search::suggest::suggest_field::{ContextSuggestField, SuggestField};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn analyzer() -> CompletionAnalyzer {
    CompletionAnalyzer::new(Arc::new(Analyzer::standard(None)))
}

fn fixture_reader() -> DirectoryReader {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data/completion_query/index");
    DirectoryReader::open(&FsDirectory::open(&dir)).unwrap()
}

fn message<T, E: std::fmt::Display>(r: Result<T, E>) -> String {
    match r {
        Ok(_) => panic!("expected an error"),
        Err(e) => e.to_string(),
    }
}

fn prefix(field: &str, text: &str) -> CompletionQuery {
    CompletionQuery::Prefix(PrefixCompletionQuery::new(analyzer(), field, text, None).unwrap())
}

#[test]
fn reserved_characters_in_the_term_are_refused() {
    let hole = PrefixCompletionQuery::new(analyzer(), "suggest", "a\u{1e}", None);
    let e = message(hole);
    assert!(e.contains("cannot contain HOLE character U+001E"), "{e}");
    let sep = RegexCompletionQuery::new("suggest", "a\u{1f}", None);
    let e = message(sep);
    assert!(
        e.contains("cannot contain unit separator character U+001F"),
        "{e}"
    );
}

#[test]
fn context_queries_validate_their_input() {
    let mut q = ContextQuery::new(prefix("ctx", "b")).unwrap();
    let e = message(q.add_context("x", -1.0, true));
    assert!(e.contains("'boost' must be >= 0"), "{e}");
    let e = message(q.add_context("a\u{1d}b", 1.0, true));
    assert!(e.contains("[0x1d] at position 1"), "{e}");
    q.add_context("music", 1.0, true).unwrap();
    q.add_context("music", 3.0, true).unwrap();
    let e = message(ContextQuery::new(CompletionQuery::Context(q)));
    assert!(
        e.contains("'query' parameter must not be of type ContextQuery"),
        "{e}"
    );
}

#[test]
fn a_context_query_over_plain_suggestions_is_refused() {
    let reader = fixture_reader();
    let s = SuggestIndexSearcher::new(&reader);
    let q = CompletionQuery::Context(ContextQuery::new(prefix("suggest", "b")).unwrap());
    let e = message(s.suggest(&q, 3, false));
    let want =
        "ContextQuery can not be executed against a non context-enabled SuggestField: suggest";
    assert!(e.contains(want), "{e}");
}

#[test]
fn a_field_that_is_not_a_suggest_field_is_refused_and_an_absent_one_is_empty() {
    let reader = fixture_reader();
    let s = SuggestIndexSearcher::new(&reader);
    let e = message(s.suggest(&prefix("id", "1"), 3, false));
    assert!(e.contains("id is not a SuggestField"), "{e}");
    assert!(s
        .suggest(&prefix("absent", "1"), 3, false)
        .unwrap()
        .is_empty());
}

#[test]
fn bad_regexes_and_edit_distances_are_refused() {
    let reader = fixture_reader();
    let s = SuggestIndexSearcher::new(&reader);
    let bad = RegexCompletionQuery::new("suggest", "(", None).unwrap();
    assert!(s.suggest(&CompletionQuery::Regex(bad), 3, false).is_err());
    let fuzzy = FuzzyCompletionQuery::with_options(
        analyzer(),
        "suggest",
        "beatles",
        None,
        9,
        true,
        1,
        3,
        false,
        10_000,
    );
    let fuzzy = CompletionQuery::Fuzzy(fuzzy.unwrap());
    let e = message(s.suggest(&fuzzy, 3, false));
    assert!(e.contains("maxEdits 9 is not supported"), "{e}");
}

#[test]
fn the_accessors_report_what_the_query_was_built_with() {
    let q = CompletionQuery::Prefix(
        PrefixCompletionQuery::new(analyzer(), "suggest", "abc", None).unwrap(),
    );
    assert_eq!((q.field(), q.text()), ("suggest", "abc"));
    assert!(q.filter().is_none());
    let f = FuzzyCompletionQuery::new(analyzer(), "suggest", "abc", None).unwrap();
    assert_eq!(f.max_edits(), FuzzyCompletionQuery::DEFAULT_MAX_EDITS);
    assert_eq!(
        f.non_fuzzy_prefix(),
        FuzzyCompletionQuery::DEFAULT_NON_FUZZY_PREFIX
    );
}

/// One field holding `SuggestField`s in one segment and
/// `ContextSuggestField`s in another has values of multiple types.
#[test]
fn a_field_with_both_suggestion_types_is_refused() {
    let tmp = TempDir::new("completion-mixed-types");
    let fs = FsDirectory::open(tmp.path());
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
    w.set_analyzer(Some(Arc::new(Analyzer::keyword())));
    w.set_use_compound_file(false);
    w.set_postings_format_for_field("s", Lucene104PostingsFormat::completion104());
    let mut d = Document::new();
    d.add_boxed(Box::new(SuggestField::new("s", "plain", 1).unwrap()));
    w.add_fields_document(&d).unwrap();
    w.commit().unwrap();
    let mut d = Document::new();
    d.add_boxed(Box::new(
        ContextSuggestField::new("s", "context", 1, &["c"]).unwrap(),
    ));
    w.add_fields_document(&d).unwrap();
    w.commit().unwrap();
    let reader = DirectoryReader::open(&fs).unwrap();
    assert_eq!(reader.segment_readers().len(), 2);
    let s = SuggestIndexSearcher::new(&reader);
    let q = CompletionQuery::Prefix(
        PrefixCompletionQuery::new(
            CompletionAnalyzer::new(Arc::new(Analyzer::keyword())),
            "s",
            "p",
            None,
        )
        .unwrap(),
    );
    let e = message(s.suggest(&q, 3, false));
    assert!(e.contains("s has values of multiple types"), "{e}");
}

#[test]
fn every_option_is_readable_back() {
    let a = CompletionAnalyzer::with_options(Arc::new(Analyzer::keyword()), false, false, 7);
    assert!(!a.preserve_sep() && !a.preserve_position_increments());
    assert_eq!(a.max_graph_expansions(), 7);
    let _ = a.analyzer();
    let f = FuzzyCompletionQuery::new(analyzer(), "suggest", "abc", None).unwrap();
    assert!(f.is_transpositions());
    assert_eq!(
        f.min_fuzzy_length(),
        FuzzyCompletionQuery::DEFAULT_MIN_FUZZY_LENGTH
    );
    assert_eq!(
        f.is_unicode_aware(),
        FuzzyCompletionQuery::DEFAULT_UNICODE_AWARE
    );
    let r = RegexCompletionQuery::new("suggest", "a", None).unwrap();
    let _ = (r.flags(), r.determinize_work_limit());
    let p = PrefixCompletionQuery::new(analyzer(), "suggest", "a", None).unwrap();
    assert!(p.analyzer().preserve_sep());
}
