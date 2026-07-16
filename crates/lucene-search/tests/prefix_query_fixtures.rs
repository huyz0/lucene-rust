//! Differential test for `PrefixQuery` (task #35) against the same real
//! `IndexWriter`-produced segment `tests/wildcard_query_fixtures.rs` (task
//! #34) already validates (`fixtures/data/blocktree_index/`) -- see that
//! file's module doc for the shared fixture-reuse rationale.
//!
//! `body`'s real terms/postings in this fixture (see `GenBlockTree.java`'s
//! own doc comment): `cat` = {0, 2}, `dog` = {0, 1}, `bird` = {1, 4}.
//!
//! `fixtures/src/AppendPrefixManifest.java` opens this exact fixture
//! directory read-only and runs genuine
//! `org.apache.lucene.search.PrefixQuery`s (a prefix matching one term, a
//! prefix matching several, the empty prefix, a prefix equal to a full
//! existing term, a no-match prefix, and a prefix containing literal
//! `*`/`?` bytes) through a real `IndexSearcher`, recording real Lucene's own
//! matched doc IDs into `manifest.properties`' `prefix.*` keys -- this is the
//! actual cross-engine proof this port's `PrefixQuery` matching is checked
//! against, not a hand-derived expectation.

use lucene_codecs::blocktree;
use lucene_codecs::postings::DocInput;
use lucene_codecs::wildcard::WildcardPattern;
use lucene_search::{search_boolean_query, BooleanQuery, Clause, PrefixQuery, VecCollector};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenBlockTree + AppendPrefixManifest)");
        let kv = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Manifest { kv }
    }

    fn get(&self, key: &str) -> &str {
        self.kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

fn read_raw(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}{}.raw", dir(), name)).unwrap_or_else(|_| panic!("missing {name}.raw"))
}

fn open_segment() -> (blocktree::BlockTreeFields, Vec<u8>, [u8; 16], String) {
    let m = Manifest::load();
    let id = id_from_hex(m.get("id_hex"));
    let suffix = m.get("segment_suffix").to_string();
    let max_doc: i32 = m.get("max_doc").parse().unwrap();

    let fnm = read_raw(m.get("fnm_file_name"));
    let field_infos = lucene_codecs::field_infos::parse(&fnm, &id, "").expect("parse .fnm");

    let tim = read_raw(m.get("tim_file_name"));
    let tip = read_raw(m.get("tip_file_name"));
    let tmd = read_raw(m.get("tmd_file_name"));
    let fields = blocktree::open(&tim, &tip, &tmd, &field_infos, &id, &suffix, max_doc)
        .expect("open blocktree");

    let doc = read_raw(m.get("doc_file_name"));
    (fields, doc, id, suffix)
}

/// Runs `search_boolean_query` with a single `Clause::Prefix(field, prefix)`
/// `must` clause and returns the matched, ascending doc-ID list.
fn prefix_docs(field: &str, prefix: &str) -> Vec<i32> {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let query = BooleanQuery::new().with_must([Clause::from(PrefixQuery::new(
        field,
        prefix.as_bytes().to_vec(),
    ))]);
    let mut c = VecCollector::default();
    search_boolean_query(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query,
        &mut c,
    )
    .unwrap();
    c.docs
}

/// Every `prefix.<case>.*` manifest entry `AppendPrefixManifest.java`
/// recorded must be reproduced exactly by this port's own `Clause::Prefix`
/// matching -- the actual cross-engine check (not a self-consistency test).
#[test]
fn prefix_matches_real_lucenes_own_prefixquery_output_for_every_case() {
    let m = Manifest::load();
    let cases: Vec<&str> = m.get("prefix.cases").split(',').collect();
    assert!(!cases.is_empty(), "manifest must record prefix cases");

    for case in cases {
        let field = m.get(&format!("prefix.{case}.field"));
        let prefix = m.get(&format!("prefix.{case}.prefix"));
        let docs_key = m.get(&format!("prefix.{case}.docs"));
        let expected: Vec<i32> = if docs_key.is_empty() {
            Vec::new()
        } else {
            docs_key.split(',').map(|d| d.parse().unwrap()).collect()
        };

        let got = prefix_docs(field, prefix);
        assert_eq!(got, expected, "case={case} field={field} prefix={prefix}");
    }
}

/// Unit-level checks of `PrefixQuery` matching semantics not already pinned
/// by the real-Lucene cases above.
#[test]
fn prefix_missing_field_matches_nothing() {
    assert_eq!(prefix_docs("no_such_field", "ca"), Vec::<i32>::new());
}

/// A prefix containing literal `*`/`?`/`\` bytes must be treated as plain
/// literal bytes to match against, never reinterpreted as wildcard syntax --
/// confirms `PrefixQuery` doesn't route through `WildcardPattern::new`'s
/// escape/glob parser at all (see `PrefixQuery`'s doc comment for the design
/// rationale). `body` has no term containing these bytes, so on the real
/// fixture this is (necessarily) still a no-match case; the next test below
/// is what actually proves the discriminating property against terms that
/// would behave differently under glob- vs literal-byte interpretation.
#[test]
fn prefix_containing_literal_wildcard_bytes_is_not_glob_interpreted() {
    assert_eq!(prefix_docs("body", "a*b?c"), Vec::<i32>::new());
    assert_eq!(prefix_docs("body", "\\"), Vec::<i32>::new());
}

/// Directly proves `WildcardPattern::prefix` (what `PrefixQuery` matching is
/// built on -- see `PrefixQuery`'s doc comment) treats `*`/`?` bytes in the
/// prefix as literal, not glob metacharacters, using terms the real fixture
/// can't provide (its dictionary has no term containing these bytes). A
/// prefix of `a*` requires the literal two-byte sequence `a*` to start the
/// term: `a*XYZ` must match (literal prefix present, `AnyMany` covers the
/// rest) while `aXYZ` must NOT (no literal `*` byte at all) -- a
/// glob-interpreting implementation would get this backwards, treating `*`
/// as "any run of bytes" and matching `aXYZ` too. Same shape for `a?`.
#[test]
fn wildcard_pattern_prefix_treats_star_and_question_mark_as_literal_bytes() {
    let star_prefix = WildcardPattern::prefix(b"a*");
    assert!(
        star_prefix.matches(b"a*XYZ"),
        "literal `a*` followed by anything must match"
    );
    assert!(
        !star_prefix.matches(b"aXYZ"),
        "no literal `*` byte present -- a glob-interpreting impl would wrongly match this"
    );
    assert!(star_prefix.matches(b"a*"), "prefix equal to the full term");

    let question_prefix = WildcardPattern::prefix(b"a?");
    assert!(
        question_prefix.matches(b"a?XYZ"),
        "literal `a?` followed by anything must match"
    );
    assert!(
        !question_prefix.matches(b"aXYZ"),
        "no literal `?` byte present -- a glob-interpreting impl would wrongly match this"
    );
    assert!(
        !question_prefix.matches(b"aYXYZ"),
        "`?` is not a literal Y -- confirms it's the literal `?` byte, not a wildcard match either"
    );
}

/// Composing a `Clause::Prefix` inside a `BooleanQuery` alongside a plain
/// `Clause::Term` must intersect (`must`) exactly like any other clause pair
/// -- `b` (bird={1,4}, matching prefix "b") AND `dog` ({0,1}) -> {1}.
#[test]
fn prefix_composes_inside_boolean_query_must() {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let query = BooleanQuery::new().with_must([
        Clause::from(PrefixQuery::new("body", b"b".to_vec())),
        Clause::Term(lucene_search::TermQuery::new("body", "dog")),
    ]);
    let mut c = VecCollector::default();
    search_boolean_query(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query,
        &mut c,
    )
    .unwrap();
    assert_eq!(c.docs, vec![1]);
}
